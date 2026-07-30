#pragma once
#include "prefill.cuh"  // ffpa::prefill
#include "tma.cuh"      // ffpa::tma

// ============================================================================
// ffpa_attn_split_d_fwd_template_sm120
// ----------------------------------------------------------------------------
// SM120a (Blackwell) TMA + MMA variant of the split-D prefill attention
// kernel. Supports two execution modes controlled by ``kNonWS``:
//
//   kNonWS=1 (non-warp-specialised, DEFAULT for sm_120a):
//     All 256 threads participate in MMA compute. Thread 0 issues TMA
//     bulk-tensor copies inline with N-stage-ahead prefetch. No dedicated
//     producer warp-group.
//
//   kNonWS=0 (warp-specialised, for sm_90/100 where setmaxnreg works):
//     128 producer threads (only tid==0 issues TMA) + 256 consumer threads.
//     Producer/consumer communicate via mbarrier-gated smem stages.
//
// Why NOT warp-specialised on sm_120a (NCU profiling, D=320, B=1,H=32,N=8192):
//
//   1) setmaxnreg is ineffective on sm_120a (ptxas C7506: cp.async.bulk.tensor
//      treated as implicit extern boundary). Producer cannot release registers
//      to consumer. Per-thread budget = 65536 / total_threads.
//
//   2) WS if/else structure confuses the register allocator. Even with 227
//      regs/thread budget (288 threads), ptxas allocates only 168 regs:
//        WS 384T: 168 regs, 465M spills, 18.7% occ → 152T
//        WS 288T: 168 regs, 0 spills,   18.7% occ → 153T
//        WS 256T: 255 regs, 10.6M spills, 10.4% occ → 147T (64x64 tile)
//      The compiler cannot prove producer-only and consumer-only registers
//      are disjoint across the if/else, so it conservatively caps allocation.
//
//   3) Non-WS eliminates the if/else → compiler allocates 255 regs freely:
//        NonWS 256T: 255 regs, 0 spills, 16.7% occ → 168T (D=320)
//                                            → 152T (D=512)
//      vs legacy cp.async baseline: 255 regs, 266M spills, 16.7% occ → 165T
//
//   4) TMA's core benefit in non-WS mode: eliminates cp.async inflight
//      register overhead. cp.async requires per-thread bookkeeping registers
//      for each in-flight copy group; TMA offloads all addressing to the
//      hardware TMA engine, freeing those registers for R_D/R_S/R_K.
//      Result: 0 spills (vs 266M with cp.async) at identical occupancy.
//
// Barrier protocol (both modes):
//   arrive_count = kConsumerThreads + 1 (256 consumer arrives + 1 TMA issuer
//   arrive_expect_tx). wait(arrive()) pattern: arrive() returns phase token,
//   wait(token) blocks until phase flips. fence inside wait_barrier.
// ============================================================================
template <typename kDataType, const int kHeadDim, const int kMmaAtomM,
          const int kMmaAtomN, const int kMmaAtomK, const int kMmaTileSeqLenQ,
          const int kMmaTileSeqLenK, const int kMmaTileSeqLenP,
          const int kMmaTileHeadDimV, const int kValTileSeqLenQ,
          const int kValTileSeqLenK, const int kValTileSeqLenP,
          const int kValTileHeadDimV, const int kMmaAccFloat32QK,
          const int kMmaAccFloat32PV, const int kOStorageAccFloat32,
          const int kStageQK, const int kStagePV, const int kPadQ,
          const int kPadK, const int kPadV, const int kQKDChunk = kMmaAtomK,
          const int kVDChunk = kMmaAtomN * 2, const int kShareSmemQKV = 0,
          const int kPersistQg2s = 0, const int kProducerThreads = 128,
          const int kNonWS = 0>
__global__ void __launch_bounds__(WARP_SIZE* kMmaTileSeqLenQ* kMmaTileSeqLenK +
                                      (kNonWS ? 0 : kProducerThreads),
                                  1)
    ffpa_attn_split_d_fwd_template_sm120(
        const CUtensorMap* __restrict__ tma_q,
        const CUtensorMap* __restrict__ tma_k,
        const CUtensorMap* __restrict__ tma_v, kDataType* __restrict__ O,
        float* __restrict__ softmax_lse, const int Nq, const int Nkv,
        const int Nh, const int Nh_kv, const float scale, const int Tc,
        const int causal, const void* __restrict__ attn_bias,
        const int attn_bias_dtype, const long long attn_bias_stride_b,
        const long long attn_bias_stride_h, const long long attn_bias_stride_m,
        const long long attn_bias_stride_n, const float dropout_p,
        const unsigned long long philox_seed,
        const unsigned long long philox_offset) {
  static_assert(kMmaAtomM == 16 && kMmaAtomN == 8 && kMmaAtomK == 16);
  static_assert(kValTileSeqLenQ == 1 && kValTileSeqLenP == 1);
  constexpr int Br = kMmaAtomM * kMmaTileSeqLenQ * kValTileSeqLenQ;
  constexpr int Bc = kMmaAtomN * kMmaTileSeqLenK * kValTileSeqLenK;
  constexpr int kConsumerThreads =
      WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK;
  constexpr int kEffProducerThreads = kNonWS ? 0 : kProducerThreads;
  constexpr int kTotalThreads = kConsumerThreads + kEffProducerThreads;
  // kQKDChunk: headdim chunk per TMA load for QK (16/32/64). kQKSubTiles =
  // number of 16-col m16n8k16 sub-tiles inside one QK TMA box (1/2/4).
  static_assert(kQKDChunk == 16 || kQKDChunk == 32 || kQKDChunk == 64);
  constexpr int kQKDChunks = kHeadDim / kQKDChunk;
  constexpr int kQKSubTiles = kQKDChunk / kMmaAtomK;
  // kVDChunk: headdim chunk per TMA load for V (16/32/64). kVDChunks = number
  // of V TMA loads per kv_tile. kSubTilesV = number of 16-col m16n8k16
  // sub-tiles inside one V TMA box (1/2/4). Each sub-tile is consumed by 2
  // j iterations (j even/odd select the 8-col halves via ldmatrix.x2.trans).
  static_assert(kVDChunk == 16 || kVDChunk == 32 || kVDChunk == 64);
  constexpr int kVDChunks = kHeadDim / kVDChunk;
  constexpr int kSubTilesV = kVDChunk / (kMmaAtomN * 2);
  // kShareSmemQKV: V reuses QK smem after QK phase completes. The barrier
  // protocol guarantees the consumer has finished all QK ldmatrix before the
  // producer starts V TMA writes (the producer's last QK wait on qk_empty
  // requires the consumer's arrive after the last QK d_chunk ldmatrix).
  static_assert(!kShareSmemQKV || (kStageQK == kStagePV),
                "kShareSmemQKV requires kStageQK == kStagePV");
  static_assert(!(kPersistQg2s && kShareSmemQKV),
                "kPersistQg2s and kShareSmemQKV are mutually exclusive");
  static_assert(!kPersistQg2s || (kStageQK == kStagePV),
                "kPersistQg2s requires kStageQK == kStagePV (KV share)");

#ifdef ENABLE_FFPA_LAUNCH_GRID_DNHB
  const int Nb_id = blockIdx.z;
  const int Nh_id = blockIdx.y;
#else
  const int Nb_id = blockIdx.y / Nh;
  const int Nh_id = blockIdx.y % Nh;
#endif
  const int Q_tile_id = blockIdx.x;
  const int group_size = Nh / Nh_kv;
  const int kv_head_idx = Nh_id / group_size;
  // TMA major_coord (global row index) for Q / K / V. The TMA descriptor
  // covers [B*H*N, D] so the major axis is the flattened (batch, head, seq)
  // row index; minor axis is the head-dim element offset.
  const int Q_major_base = (Nb_id * Nh + Nh_id) * Nq + Q_tile_id * Br;
  const int KV_major_base = (Nb_id * Nh_kv + kv_head_idx) * Nkv;
  const int O_gmem_offset =
      ((Nb_id * Nh * Nq * kHeadDim) + (Nh_id * Nq * kHeadDim));

  // Whole-block early exit: grid may launch one extra CTA when Nq % Br != 0.
  if ((Q_tile_id * Br) >= Nq)
    return;

  // TMA SWIZZLE (32B/128B) requires a 1024B-aligned smem base so the hardware
  // swizzle phase starts at zero; ``__align__(16)`` would shift the phase and
  // corrupt every TMA-written tile.
  extern __shared__ __align__(1024) unsigned char ffpa_smem_raw[];
  kDataType* smem = reinterpret_cast<kDataType*>(ffpa_smem_raw);
  constexpr int Q_tile_size = Br * (kQKDChunk + kPadQ);
  constexpr int K_tile_size = Bc * (kQKDChunk + kPadK);
  constexpr int V_tile_size = Bc * (kVDChunk + kPadV);
  // kPersistQg2s: Q persisted in dedicated region, K/V share staged region.
  // kShareSmemQKV: V reuses QK smem. Stage stride = max(QK combined, V).
  // Neither: Q/K/V fully independent staged regions.
  constexpr int kKVStageStride =
      kPersistQg2s ? (K_tile_size > V_tile_size ? K_tile_size : V_tile_size)
                   : 0;
  constexpr int kStageStride = kShareSmemQKV
                                   ? ((Q_tile_size + K_tile_size) > V_tile_size
                                          ? (Q_tile_size + K_tile_size)
                                          : V_tile_size)
                                   : 0;
  constexpr int kQSmemStride =
      kPersistQg2s ? Q_tile_size : (kShareSmemQKV ? kStageStride : Q_tile_size);
  constexpr int kKSmemStride =
      kPersistQg2s ? kKVStageStride
                   : (kShareSmemQKV ? kStageStride : K_tile_size);
  constexpr int kVSmemStride =
      kPersistQg2s ? kKVStageStride
                   : (kShareSmemQKV ? kStageStride : V_tile_size);
  kDataType* Q_tile_smem = smem;
  kDataType* K_tile_smem =
      kPersistQg2s    ? (Q_tile_smem + kQKDChunks * Q_tile_size)
      : kShareSmemQKV ? (Q_tile_smem + Q_tile_size)
                      : (Q_tile_smem + kStageQK * Q_tile_size);
  kDataType* V_tile_smem = kPersistQg2s ? K_tile_smem
                           : kShareSmemQKV
                               ? Q_tile_smem
                               : (K_tile_smem + kStageQK * K_tile_size);
  const uint32_t smem_Q_base_ptr = __cvta_generic_to_shared(Q_tile_smem);
  const uint32_t smem_K_base_ptr = __cvta_generic_to_shared(K_tile_smem);
  const uint32_t smem_V_base_ptr = __cvta_generic_to_shared(V_tile_smem);

  // Barriers live at the end of dynamic smem (after Q/K/V tiles).
  constexpr int kQKVSmemElems =
      kPersistQg2s    ? (kQKDChunks * Q_tile_size + kStageQK * kKVStageStride)
      : kShareSmemQKV ? kStageQK * kStageStride
                      : (kStageQK * Q_tile_size + kStageQK * K_tile_size +
                         kStagePV * V_tile_size);
  ffpa::tma::barrier_t* qk_full =
      reinterpret_cast<ffpa::tma::barrier_t*>(smem + kQKVSmemElems);
  ffpa::tma::barrier_t* qk_empty = qk_full + kStageQK;
  ffpa::tma::barrier_t* v_full = qk_empty + kStageQK;
  ffpa::tma::barrier_t* v_empty = v_full + kStagePV;
  // kShareSmemQKV / kPersistQg2s: dedicated phase-transition barriers.
  // qk_done: consumer signals after QK loop → producer waits before V writes.
  // v_done: consumer signals after V loop → producer waits before next QK.
  ffpa::tma::barrier_t* qk_done = v_empty + kStagePV;
  ffpa::tma::barrier_t* v_done =
      qk_done + ((kShareSmemQKV || kPersistQg2s) ? 1 : 0);
  // kPersistQg2s: q_ready barrier for initial Q load completion.
  ffpa::tma::barrier_t* q_ready =
      v_done + ((kShareSmemQKV || kPersistQg2s) ? 1 : 0);

  // Barrier init (thread 0 only): arrive_count=257 for all barriers
  // (256 consumer arrives + 1 producer arrive/arrive_expect_tx). Uses the
  // ``wait(arrive())`` protocol:
  //   - full: 256 consumer ``wait(arrive())`` (arrive contributes 1 each) +
  //           1 producer ``arrive_expect_tx`` (contributes 1 + sets tx-count).
  //   - empty: 256 consumer ``arrive()`` + 1 producer ``wait(arrive())``.
  // No explicit phase tracking needed (arrive() returns a phase token that
  // wait(token) checks). Fence: producer calls fence_async_shared after TMA
  // issue; consumer gets fence inside wait_barrier (barrier.wait(arrive())).
  if (threadIdx.x == 0) {
    for (int s = 0; s < kStageQK; ++s) {
      ffpa::tma::init_barrier(&qk_full[s], kConsumerThreads + 1);
      ffpa::tma::init_barrier(&qk_empty[s], kConsumerThreads + 1);
    }
    for (int s = 0; s < kStagePV; ++s) {
      ffpa::tma::init_barrier(&v_full[s], kConsumerThreads + 1);
      ffpa::tma::init_barrier(&v_empty[s], kConsumerThreads + 1);
    }
    if constexpr (kShareSmemQKV || kPersistQg2s) {
      ffpa::tma::init_barrier(qk_done, kConsumerThreads + 1);
      ffpa::tma::init_barrier(v_done, kConsumerThreads + 1);
    }
    if constexpr (kPersistQg2s) {
      ffpa::tma::init_barrier(q_ready, kConsumerThreads + 1);
    }
    ffpa::tma::fence_async_shared();
  }
  __syncthreads();

  const bool is_producer = kNonWS ? false : (threadIdx.x < kProducerThreads);
  const int wg_tid =
      is_producer ? threadIdx.x : (threadIdx.x - kEffProducerThreads);
  const int warp_QP = wg_tid / WARP_SIZE;
  constexpr int warp_KV = 0;

  constexpr int kQTileBytes = Br * kQKDChunk * sizeof(kDataType);
  constexpr int kKTileBytes = Bc * kQKDChunk * sizeof(kDataType);
  constexpr int kVTileBytes = Bc * kVDChunk * sizeof(kDataType);

  const int Br_base = Q_tile_id * Br;
  const int kv_offset = Nkv - Nq;
  const int causal_thresh_row0 = Br_base + kv_offset;
  const int Tc_eff =
      causal ? min(Tc, ((Br_base + Br - 1 + kv_offset) / Bc) + 1) : Tc;
  const int mask_start_tile =
      causal ? max(0, (causal_thresh_row0 + 1) / Bc) : INT_MAX;
  const float inv_scale = 1.0f / scale;

  if constexpr (!kNonWS) {
    if (is_producer) {
      // setmaxnreg.dec: release registers from the producer warpgroup so the
      // consumer can borrow more. Effective on sm_90a/sm_100a; no-op on sm_120a
      // (ptxas C7506, suppressed in env.py). All 128 producer threads execute.
      ffpa::tma::warpgroup_reg_dealloc<40>();
      // ======================================================================
      // Producer warp-group (WG0): only tid==0 issues TMA. For each kv_tile,
      // load all QK d_chunks then all V v_chunks into barrier-gated stages.
      //
      // Protocol (wait(arrive()) mode, aligned with flash_attn_tma_mma_ws):
      //   - empty.wait(empty.arrive()): arrive contributes 1 (the 257th after
      //     256 consumer arrives), wait returns immediately as phase flips.
      //   - load_2d_no_arrive x2 (Q+K) + arrive_expect_tx(combined bytes):
      //     arrive_expect_tx contributes 1 arrival + sets tx-count; the 256
      //     consumer arrives (from wait(arrive()) on full) provide the rest.
      //   - NO explicit fence_async_shared() here: the consumer's
      //     wait_barrier(full) already includes the required
      //     fence.proxy.async.shared::cta (see tma.cuh). Placing fence on
      //     the producer side would stall the producer on TMA completion
      //     before the barrier signal, reducing pipeline overlap.
      // ======================================================================
      if (wg_tid == 0) {
        // kPersistQg2s: one-time Q load into persistent region before kv loop.
        if constexpr (kPersistQg2s) {
          for (int d_chunk = 0; d_chunk < kQKDChunks; ++d_chunk) {
            kDataType* q_dst = Q_tile_smem + d_chunk * Q_tile_size;
            const int minor = d_chunk * kQKDChunk;
            ffpa::tma::load_2d_no_arrive(q_dst, tma_q, minor, Q_major_base,
                                         *q_ready);
          }
          ffpa::tma::arrive_expect_tx(*q_ready, kQKDChunks * kQTileBytes);
        }
        for (int kv_tile = 0; kv_tile < Tc_eff; ++kv_tile) {
          const int kv_major = KV_major_base + kv_tile * Bc;
          // Transition: wait for consumer V phase before overwriting shared
          // smem.
          if constexpr (kShareSmemQKV || kPersistQg2s) {
            if (kv_tile > 0)
              ffpa::tma::wait_barrier(*v_done);
          }
          // QK phase: load Q+K (or K-only if persist) per d_chunk.
          for (int d_chunk = 0; d_chunk < kQKDChunks; ++d_chunk) {
            ffpa::tma::wait_barrier(qk_empty[d_chunk % kStageQK]);
            kDataType* k_dst =
                K_tile_smem + (d_chunk % kStageQK) * kKSmemStride;
            const int minor = d_chunk * kQKDChunk;
            if constexpr (!kPersistQg2s) {
              kDataType* q_dst =
                  Q_tile_smem + (d_chunk % kStageQK) * kQSmemStride;
              ffpa::tma::load_2d_no_arrive(q_dst, tma_q, minor, Q_major_base,
                                           qk_full[d_chunk % kStageQK]);
            }
            ffpa::tma::load_2d_no_arrive(k_dst, tma_k, minor, kv_major,
                                         qk_full[d_chunk % kStageQK]);
            ffpa::tma::arrive_expect_tx(
                qk_full[d_chunk % kStageQK],
                kPersistQg2s ? kKTileBytes : (kQTileBytes + kKTileBytes));
          }
          // Transition: wait for consumer QK ldmatrix before V overwrites.
          if constexpr (kShareSmemQKV || kPersistQg2s) {
            ffpa::tma::wait_barrier(*qk_done);
          }
          // V phase: one V tile per v_chunk.
          for (int v_chunk = 0; v_chunk < kVDChunks; ++v_chunk) {
            ffpa::tma::wait_barrier(v_empty[v_chunk % kStagePV]);
            kDataType* v_dst =
                V_tile_smem + (v_chunk % kStagePV) * kVSmemStride;
            const int minor = v_chunk * kVDChunk;
            ffpa::tma::load_2d_no_arrive(v_dst, tma_v, minor, kv_major,
                                         v_full[v_chunk % kStagePV]);
            ffpa::tma::arrive_expect_tx(v_full[v_chunk % kStagePV],
                                        kVTileBytes);
          }
        }
      }
      return;
    }  // if (is_producer)
  }  // if constexpr (!kNonWS)
  // Consumer path: all threads when kNonWS=1, non-producer threads when WS.
  // setmaxnreg.inc: borrow registers released by the producer, raising the
  // consumer per-thread budget from the static 170 (65536/384) toward ~232
  // so R_D stops spilling for D>=320. Effective on sm_90a/sm_100a; no-op on
  // sm_120a (ptxas C7506, suppressed in env.py). All 256 consumer threads
  // (2 warpgroups) execute.
  if constexpr (!kNonWS) {
    ffpa::tma::warpgroup_reg_alloc<232>();
  }
  // ======================================================================
  // Consumer warp-group (WG1): 256 threads, 8 warps. Reuses the existing
  // split-D compute pipeline; cp.async wait is replaced by barrier waits.
  // ======================================================================
  // Persistent softmax state.
  float lane_block_row_max_old[kValTileSeqLenQ][2];
  float lane_block_row_sum_old[kValTileSeqLenQ][2];
  ffpa::utils::fill_2D_regs<float, kValTileSeqLenQ, 2>(lane_block_row_max_old,
                                                       -INFINITY);
  ffpa::utils::fill_2D_regs<float, kValTileSeqLenQ, 2>(lane_block_row_sum_old,
                                                       0.0f);

  uint32_t R_Q[kValTileSeqLenQ][1][4];
  uint32_t R_K[kValTileSeqLenK][2];
  uint32_t R_V[1][2];
  uint32_t R_S[kValTileSeqLenQ][kValTileSeqLenK][(kMmaAccFloat32QK) ? 4 : 2];
  uint32_t R_O[(kMmaAccFloat32PV) ? 4 : 2];
  uint32_t R_D[kValTileSeqLenP][kValTileHeadDimV]
              [(kOStorageAccFloat32) ? 4 : 2];
  ffpa::utils::fill_3D_regs<uint32_t, kValTileSeqLenP, kValTileHeadDimV,
                            ((kOStorageAccFloat32) ? 4 : 2)>(R_D, 0);

  // Signal all empty stages as consumed so the producer can start filling.
  // Each consumer thread calls arrive() once per stage (256 arrivals); the
  // producer's wait(arrive()) provides the 257th arrival that flips phase.
  for (int s = 0; s < kStageQK; ++s) {
    [[maybe_unused]] auto token = qk_empty[s].arrive();
  }
  for (int s = 0; s < kStagePV; ++s) {
    [[maybe_unused]] auto token = v_empty[s].arrive();
  }
  // kNonWS: thread 0 prefetches first kStageQK QK stages inline.
  if constexpr (kNonWS) {
    if (threadIdx.x == 0) {
      const int kv_major_0 = KV_major_base;
      for (int d = 0; d < kStageQK && d < kQKDChunks; ++d) {
        ffpa::tma::wait_barrier(qk_empty[d]);
        kDataType* q_dst = Q_tile_smem + d * kQSmemStride;
        kDataType* k_dst = K_tile_smem + d * kKSmemStride;
        const int minor = d * kQKDChunk;
        ffpa::tma::load_2d_no_arrive(q_dst, tma_q, minor, Q_major_base,
                                     qk_full[d]);
        ffpa::tma::load_2d_no_arrive(k_dst, tma_k, minor, kv_major_0,
                                     qk_full[d]);
        ffpa::tma::arrive_expect_tx(qk_full[d], kQTileBytes + kKTileBytes);
      }
    }
  }
  // kPersistQg2s: wait for Q initial load to complete before entering loop.
  if constexpr (kPersistQg2s) {
    ffpa::tma::wait_barrier(*q_ready);
  }

#pragma unroll 1
  for (int tile_K_seqlen = 0; tile_K_seqlen < Tc_eff; ++tile_K_seqlen) {
    // kNonWS: prefetch first kStageQK QK stages for kv_tile > 0.
    // (kv_tile 0 was prefetched before the loop.)
    if constexpr (kNonWS) {
      if (tile_K_seqlen > 0 && threadIdx.x == 0) {
        const int kv_major = KV_major_base + tile_K_seqlen * Bc;
        for (int d = 0; d < kStageQK && d < kQKDChunks; ++d) {
          ffpa::tma::wait_barrier(qk_empty[d]);
          kDataType* q_dst = Q_tile_smem + d * kQSmemStride;
          kDataType* k_dst = K_tile_smem + d * kKSmemStride;
          const int minor = d * kQKDChunk;
          ffpa::tma::load_2d_no_arrive(q_dst, tma_q, minor, Q_major_base,
                                       qk_full[d]);
          ffpa::tma::load_2d_no_arrive(k_dst, tma_k, minor, kv_major,
                                       qk_full[d]);
          ffpa::tma::arrive_expect_tx(qk_full[d], kQTileBytes + kKTileBytes);
        }
      }
    }
    ffpa::utils::fill_3D_regs<uint32_t, kValTileSeqLenQ, kValTileSeqLenK,
                              (kMmaAccFloat32QK) ? 4 : 2>(R_S, 0);
    // QK phase: kQKDChunks TMA loads, each Q[Br,kQKDChunk]+K[Bc,kQKDChunk].
    // Each TMA tile is consumed by kQKSubTiles m16n8k16 sub-tiles
    // (kQKDChunk/16).
#pragma unroll
    for (int tile_K_d = 0; tile_K_d < kQKDChunks; ++tile_K_d) {
      const int stage = tile_K_d % kStageQK;
      // wait_barrier = barrier.wait(barrier.arrive()): the arrive()
      // contributes 1 arrival (256 consumer + 1 producer arrive_expect_tx
      // = 257 -> phase flips), wait(token) blocks until flip. fence inside.
      ffpa::tma::wait_barrier(qk_full[stage]);

#pragma unroll
      for (int sub = 0; sub < kQKSubTiles; ++sub) {
        const int sub_col = sub * kMmaAtomK;
        // Q s2r: kSmemColStride=kQKDChunk selects the wide-tile row stride +
        // swizzle width; subtile_col_offset picks the 16-col sub-block.
        // kPersistQg2s: Q read from persistent region (stage=tile_K_d).
        ffpa::prefill::sync_fetch_qkv_frags_s2r<0, 4, kQSmemStride, kMmaAtomM,
                                                kMmaAtomN, kMmaAtomK, kPadQ,
                                                kDataType, kQKDChunk>(
            smem_Q_base_ptr, &R_Q[0][0][0], warp_QP, 0, 0,
            kPersistQg2s ? tile_K_d : stage, sub_col);
        // K s2r
#pragma unroll
        for (int j = 0; j < kValTileSeqLenK; ++j) {
          ffpa::prefill::sync_fetch_qkv_frags_s2r<0, 2, kKSmemStride, kMmaAtomM,
                                                  kMmaAtomN, kMmaAtomK, kPadK,
                                                  kDataType, kQKDChunk>(
              smem_K_base_ptr, &R_K[j][0], warp_KV, j, 0, stage, sub_col);
        }
        // Q@K^T MMA
#pragma unroll
        for (int j = 0; j < kValTileSeqLenK; ++j) {
          if constexpr (kMmaAccFloat32QK) {
            ffpa::mma::m16n8k16_abf32<kDataType,
                                      ffpa::mma::MMAMode::kInplaceUpdate>(
                &R_S[0][j][0], &R_S[0][j][1], &R_S[0][j][2], &R_S[0][j][3],
                &R_Q[0][0][0], &R_Q[0][0][1], &R_Q[0][0][2], &R_Q[0][0][3],
                &R_K[j][0], &R_K[j][1]);
          } else {
            ffpa::mma::m16n8k16_f16f16f16<ffpa::mma::MMAMode::kInplaceUpdate>(
                &R_S[0][j][0], &R_S[0][j][1], &R_Q[0][0][0], &R_Q[0][0][1],
                &R_Q[0][0][2], &R_Q[0][0][3], &R_K[j][0], &R_K[j][1]);
          }
        }
      }
      // Release QK stage for producer reuse (256 consumer arrives; producer's
      // wait(arrive()) on empty provides the 257th arrival).
      { [[maybe_unused]] auto token = qk_empty[stage].arrive(); }
      // kNonWS: thread 0 prefetches next QK d_chunk into the just-freed stage.
      if constexpr (kNonWS) {
        const int d_next = tile_K_d + kStageQK;
        if (d_next < kQKDChunks && threadIdx.x == 0) {
          const int s_next = d_next % kStageQK;
          ffpa::tma::wait_barrier(qk_empty[s_next]);
          kDataType* q_dst = Q_tile_smem + s_next * kQSmemStride;
          kDataType* k_dst = K_tile_smem + s_next * kKSmemStride;
          const int minor = d_next * kQKDChunk;
          const int kv_major = KV_major_base + tile_K_seqlen * Bc;
          ffpa::tma::load_2d_no_arrive(q_dst, tma_q, minor, Q_major_base,
                                       qk_full[s_next]);
          ffpa::tma::load_2d_no_arrive(k_dst, tma_k, minor, kv_major,
                                       qk_full[s_next]);
          ffpa::tma::arrive_expect_tx(qk_full[s_next],
                                      kQTileBytes + kKTileBytes);
        }
      }
    }
    // NOTE: no __syncthreads() here -- producer does not participate in the
    // consumer loop and would deadlock on a CTA-wide barrier. The QK MMA
    // results live entirely in registers (R_S); softmax below uses only
    // warp shuffles, so no cross-warp smem ordering is needed.

    // kShareSmemQKV / kPersistQg2s: signal producer that all QK ldmatrix is
    // done and the shared smem is safe to overwrite with V data.
    if constexpr (kShareSmemQKV || kPersistQg2s) {
      {
        [[maybe_unused]] auto token = qk_done->arrive();
      }
    }

    // Online safe softmax (reused unchanged).
    float lane_row_max_new[kValTileSeqLenQ][2];
    float lane_row_sum_new[kValTileSeqLenQ][2];
    ffpa::utils::fill_2D_regs<float, kValTileSeqLenQ, 2>(lane_row_max_new,
                                                         -INFINITY);
    ffpa::utils::fill_2D_regs<float, kValTileSeqLenQ, 2>(lane_row_sum_new,
                                                         0.0f);
    {
      const int kv_valid_local = Nkv - tile_K_seqlen * Bc;
      if (kv_valid_local < Bc) {
        ffpa::prefill::sync_apply_kv_mask<kValTileSeqLenK, kMmaAccFloat32QK,
                                          kDataType>(&R_S[0][0][0],
                                                     kv_valid_local);
      }
    }
    if (tile_K_seqlen >= mask_start_tile) {
      ffpa::prefill::sync_apply_causal_mask<kValTileSeqLenK, kMmaAccFloat32QK,
                                            kDataType>(
          &R_S[0][0][0], warp_QP, Br_base, tile_K_seqlen * Bc, kv_offset);
    }
    if (attn_bias != nullptr) {
      ffpa::prefill::sync_apply_attn_bias<kValTileSeqLenK, kMmaAccFloat32QK,
                                          kDataType>(
          &R_S[0][0][0], attn_bias, attn_bias_dtype, attn_bias_stride_b,
          attn_bias_stride_h, attn_bias_stride_m, attn_bias_stride_n, Nb_id,
          Nh_id, warp_QP, Br_base, tile_K_seqlen * Bc, Nq, Nkv, inv_scale);
    }
    ffpa::prefill::sync_online_safe_softmax<kValTileSeqLenK, kMmaAccFloat32QK,
                                            kDataType>(
        &R_S[0][0][0], scale, &lane_row_max_new[0][0], &lane_row_sum_new[0][0],
        &lane_block_row_max_old[0][0], &lane_block_row_sum_old[0][0]);
    if (dropout_p > 0.0f) {
      ffpa::prefill::sync_apply_dropout_to_p<kValTileSeqLenK, kMmaAccFloat32QK,
                                             kDataType>(
          &R_S[0][0][0], dropout_p, philox_seed, philox_offset, Nb_id, Nh_id,
          Nh, warp_QP, Br_base, tile_K_seqlen * Bc, Nq, Nkv);
    }

    // PV phase: kVDChunks V TMA stages, each containing kSubTilesV 16-col
    // sub-tiles. Each sub-tile is consumed by 2 j iterations (jj=0,1 select
    // the 8-col halves via ldmatrix.x2.trans). Global j index maps as:
    //   j = v_chunk * kSubTilesV * 2 + sub * 2 + jj
    static_assert(kValTileSeqLenP == 1);
    {
      float rescale_o_factor_0[1];
      float rescale_o_factor_1[1];
      ffpa::prefill::sync_precompute_rescale_factors(
          &rescale_o_factor_0[0], &rescale_o_factor_1[0],
          &lane_row_max_new[0][0], &lane_block_row_max_old[0][0],
          tile_K_seqlen);

      // kNonWS: thread 0 prefetches first kStagePV V stages.
      if constexpr (kNonWS) {
        if (threadIdx.x == 0) {
          const int kv_major = KV_major_base + tile_K_seqlen * Bc;
          for (int v = 0; v < kStagePV && v < kVDChunks; ++v) {
            ffpa::tma::wait_barrier(v_empty[v]);
            kDataType* v_dst = V_tile_smem + v * kVSmemStride;
            const int minor = v * kVDChunk;
            ffpa::tma::load_2d_no_arrive(v_dst, tma_v, minor, kv_major,
                                         v_full[v]);
            ffpa::tma::arrive_expect_tx(v_full[v], kVTileBytes);
          }
        }
      }

#pragma unroll
      for (int v_chunk = 0; v_chunk < kVDChunks; ++v_chunk) {
        const int v_stage = v_chunk % kStagePV;
        // Wait for V TMA stage (one wait per kVDChunk-wide tile).
        ffpa::tma::wait_barrier(v_full[v_stage]);

#pragma unroll
        for (int sub = 0; sub < kSubTilesV; ++sub) {
          const int subtile_col_offset = sub * (kMmaAtomN * 2);
#pragma unroll
          for (int jj = 0; jj < 2; ++jj) {
            const int j = v_chunk * (kSubTilesV * 2) + sub * 2 + jj;
            ffpa::utils::fill_1D_regs<uint32_t, (kMmaAccFloat32PV) ? 4 : 2>(R_O,
                                                                            0);
#pragma unroll
            for (int tile_V_Bc = 0; tile_V_Bc < (Bc / kMmaAtomK); ++tile_V_Bc) {
              ffpa::prefill::sync_fetch_qkv_frags_s2r<
                  1, 2, kVSmemStride, kMmaAtomM, kMmaAtomN, kMmaAtomK, kPadV,
                  kDataType, kVDChunk>(smem_V_base_ptr, &R_V[0][0], warp_KV, jj,
                                       tile_V_Bc, v_stage, subtile_col_offset);
              const int p_offset = tile_V_Bc * 2;
              if constexpr (kMmaAccFloat32PV) {
                ffpa::mma::m16n8k16_abf32<kDataType,
                                          ffpa::mma::MMAMode::kInplaceUpdate>(
                    &R_O[0], &R_O[1], &R_O[2], &R_O[3], &R_S[0][p_offset][0],
                    &R_S[0][p_offset][1], &R_S[0][p_offset + 1][0],
                    &R_S[0][p_offset + 1][1], &R_V[0][0], &R_V[0][1]);
              } else {
                ffpa::mma::m16n8k16_f16f16f16<
                    ffpa::mma::MMAMode::kInplaceUpdate>(
                    &R_O[0], &R_O[1], &R_S[0][p_offset][0],
                    &R_S[0][p_offset][1], &R_S[0][p_offset + 1][0],
                    &R_S[0][p_offset + 1][1], &R_V[0][0], &R_V[0][1]);
              }
            }
            ffpa::prefill::sync_rescaling_tiling_o<kOStorageAccFloat32,
                                                   kMmaAccFloat32PV, kDataType>(
                &R_D[0][0][0], &R_O[0], &rescale_o_factor_0[0],
                &rescale_o_factor_1[0], tile_K_seqlen, j);
          }
        }
        // Release V stage: all kSubTilesV*2 j's consumed this tile.
        { [[maybe_unused]] auto token = v_empty[v_stage].arrive(); }
        // kNonWS: thread 0 prefetches next V chunk into the just-freed stage.
        if constexpr (kNonWS) {
          const int v_next = v_chunk + kStagePV;
          if (v_next < kVDChunks && threadIdx.x == 0) {
            const int s_next = v_next % kStagePV;
            ffpa::tma::wait_barrier(v_empty[s_next]);
            kDataType* v_dst = V_tile_smem + s_next * kVSmemStride;
            const int minor = v_next * kVDChunk;
            const int kv_major = KV_major_base + tile_K_seqlen * Bc;
            ffpa::tma::load_2d_no_arrive(v_dst, tma_v, minor, kv_major,
                                         v_full[s_next]);
            ffpa::tma::arrive_expect_tx(v_full[s_next], kVTileBytes);
          }
        }
      }
      ffpa::prefill::sync_update_max_expsum(
          &lane_row_max_new[0][0], &lane_row_sum_new[0][0],
          &lane_block_row_max_old[0][0], &lane_block_row_sum_old[0][0],
          &rescale_o_factor_0[0], &rescale_o_factor_1[0]);
    }
    // kShareSmemQKV / kPersistQg2s: signal producer that all V ldmatrix is
    // done and the shared smem is safe to overwrite with next kv_tile's QK.
    if constexpr (kShareSmemQKV || kPersistQg2s) {
      {
        [[maybe_unused]] auto token = v_done->arrive();
      }
    }
  }

  // Epilogue: final rescale + store O + LSE (reused unchanged).
  ffpa::prefill::sync_rescaling_final_o<kValTileHeadDimV, kOStorageAccFloat32,
                                        kDataType>(
      &R_D[0][0][0], &lane_block_row_sum_old[0][0]);
  ffpa::prefill::sync_store_o_r2g<Br, kHeadDim, kMmaAtomM, kMmaAtomN,
                                  kValTileHeadDimV, kOStorageAccFloat32,
                                  kDataType>(O, O_gmem_offset, Q_tile_id,
                                             warp_QP, &R_D[0][0][0],
                                             &R_Q[0][0][0], &R_K[0][0], Nq);
  const int softmax_lse_offset = Nb_id * Nh * Nq + Nh_id * Nq;
  ffpa::prefill::sync_store_lse_r2g<Br, kMmaAtomM, kValTileSeqLenQ>(
      softmax_lse, softmax_lse_offset, Q_tile_id, warp_QP,
      &lane_block_row_max_old[0][0], &lane_block_row_sum_old[0][0], Nq);
}

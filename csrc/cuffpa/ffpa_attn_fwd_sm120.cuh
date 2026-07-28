#pragma once
#include "prefill.cuh"  // ffpa::prefill
#include "tma.cuh"      // ffpa::tma

// ============================================================================
// ffpa_attn_split_d_fwd_template_sm120
// ----------------------------------------------------------------------------
// SM120a (Blackwell) TMA + MMA warp-specialised variant of the split-D
// prefill attention kernel. The compute logic (ldmatrix / mma / softmax /
// rescale / epilogue) is identical to ``ffpa_attn_split_d_fwd_template``;
// only the G->S data movement is replaced: a dedicated producer warp-group
// issues TMA bulk-tensor copies into barrier-gated shared-memory stages,
// while a consumer warp-group runs the existing MMA pipeline.
//
// Design (mirrors ``ffpa_attn_tma_mma_ws_split_d_cute`` in LeetCUDA, but in
// pure CUDA with kMmaAtomK=16 split-D chunks and 256 consumer threads):
//   - 384 threads = 128 producer (WG0) + 256 consumer (WG1)
//   - Producer (only tid==0 issues TMA) loads combined Q+K per d_chunk into
//     a single ``qk_full`` barrier stage, and V per v_chunk into ``v_full``.
//   - Consumer reuses ``sync_fetch_qkv_frags_s2r`` / ``m16n8k16`` /
//     ``sync_online_safe_softmax`` / ``sync_rescaling_*`` / ``sync_store_*``
//     unchanged; cp.async wait/wait_group is replaced by ``wait_barrier``
//     (``barrier.wait(barrier.arrive())``) + ``arrive``.
//   - Barrier protocol (wait(arrive()) mode, aligned with
//     flash_attn_tma_mma_ws_stages_split_q):
//       full[s]/empty[s]: arrive_count = 257
//         (256 consumer arrives + 1 producer arrive/arrive_expect_tx)
//       No explicit phase tracking (arrive() returns a phase token).
//   - No ``__syncthreads()`` inside producer/consumer branches: it is a
//     CTA-wide barrier requiring all 384 threads, but producer and consumer
//     run independent loops and would deadlock. The only ``__syncthreads``
//     is the single one after barrier init (before the if/else split).
//
// First version: kStageQK=2, kStagePV=2, D=512. Q persistence and register
// ping-pong are disabled (kPersistQg2s=0, kPersistQs2r=0, kRegPipeKV=0,
// kShareSmemQKV=0) for simplicity.
//
// V TMA d_chunk (kVDChunk): V TMA loads use a kVDChunk-wide box (16/32/64)
// instead of the original fixed 16-col strips, reducing the number of TMA
// issues per kv_tile from D/16 to D/kVDChunk. The consumer PV loop iterates
// v_chunk -> sub (kSubTilesV = kVDChunk/16 sub-tiles) -> jj (2 halves of each
// 16-col sub-tile); ldmatrix.x2.trans selects the 8-col half via ``jj`` and
// the sub-tile base via ``subtile_col_offset`` (kSmemColStride=kVDChunk).
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
          const int kVDChunk = kMmaAtomN * 2, const int kShareSmemQKV = 0>
__global__ void
// minBlocksPerMultiprocessor=1: let the compiler use the full per-thread
// register budget (65536/384 = 170 regs). Without this hint the compiler's
// occupancy heuristic picks minBlocks=2 for some dtypes (e.g. bf16), capping
// registers at ~85 and forcing massive R_D spilling. Note: the compiler still
// allocates only ~168 regs (vs baseline's 255) due to the if/else WS
// structure confusing the register allocator; R_D[1][64][4]=256-reg spilling
// remains the main overhead (~12% vs baseline).
__launch_bounds__(WARP_SIZE* kMmaTileSeqLenQ* kMmaTileSeqLenK + 128, 1)
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
  constexpr int kProducerThreads = 128;
  constexpr int kTotalThreads = kConsumerThreads + kProducerThreads;
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
  // corrupt every TMA-written tile (illegal instruction / wrong ldmatrix
  // addresses). See /memories/repo/leetcuda-sm120-tma-mma-ws.md.
  extern __shared__ __align__(1024) unsigned char ffpa_smem_raw[];
  kDataType* smem = reinterpret_cast<kDataType*>(ffpa_smem_raw);
  constexpr int Q_tile_size = Br * (kQKDChunk + kPadQ);
  constexpr int K_tile_size = Bc * (kQKDChunk + kPadK);
  constexpr int V_tile_size = Bc * (kVDChunk + kPadV);
  // kShareSmemQKV: V reuses QK smem. Stage stride = max(QK combined, V).
  // Q[stage] = base + stage * kStageStride
  // K[stage] = base + Q_tile_size + stage * kStageStride
  // V[stage] = base + stage * kStageStride (overlaps Q+K)
  constexpr int kStageStride = kShareSmemQKV
                                   ? ((Q_tile_size + K_tile_size) > V_tile_size
                                          ? (Q_tile_size + K_tile_size)
                                          : V_tile_size)
                                   : 0;
  constexpr int kQSmemStride = kShareSmemQKV ? kStageStride : Q_tile_size;
  constexpr int kKSmemStride = kShareSmemQKV ? kStageStride : K_tile_size;
  constexpr int kVSmemStride = kShareSmemQKV ? kStageStride : V_tile_size;
  kDataType* Q_tile_smem = smem;
  kDataType* K_tile_smem = kShareSmemQKV
                               ? (Q_tile_smem + Q_tile_size)
                               : (Q_tile_smem + kStageQK * Q_tile_size);
  kDataType* V_tile_smem =
      kShareSmemQKV ? Q_tile_smem : (K_tile_smem + kStageQK * K_tile_size);
  const uint32_t smem_Q_base_ptr = __cvta_generic_to_shared(Q_tile_smem);
  const uint32_t smem_K_base_ptr = __cvta_generic_to_shared(K_tile_smem);
  const uint32_t smem_V_base_ptr = __cvta_generic_to_shared(V_tile_smem);

  // Barriers live at the end of dynamic smem (after Q/K/V tiles).
  constexpr int kQKVSmemElems =
      kShareSmemQKV ? kStageQK * kStageStride
                    : (kStageQK * Q_tile_size + kStageQK * K_tile_size +
                       kStagePV * V_tile_size);
  ffpa::tma::barrier_t* qk_full =
      reinterpret_cast<ffpa::tma::barrier_t*>(smem + kQKVSmemElems);
  ffpa::tma::barrier_t* qk_empty = qk_full + kStageQK;
  ffpa::tma::barrier_t* v_full = qk_empty + kStageQK;
  ffpa::tma::barrier_t* v_empty = v_full + kStagePV;
  // kShareSmemQKV: dedicated phase-transition barriers.
  // qk_done: consumer signals after QK loop → producer waits before V writes.
  // v_done: consumer signals after V loop → producer waits before next QK.
  ffpa::tma::barrier_t* qk_done = v_empty + kStagePV;
  ffpa::tma::barrier_t* v_done = qk_done + (kShareSmemQKV ? 1 : 0);

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
    if constexpr (kShareSmemQKV) {
      // Phase-transition barriers: 256 consumer arrives + 1 producer wait.
      ffpa::tma::init_barrier(qk_done, kConsumerThreads + 1);
      ffpa::tma::init_barrier(v_done, kConsumerThreads + 1);
    }
    ffpa::tma::fence_async_shared();
  }
  __syncthreads();

  const bool is_producer = threadIdx.x < kProducerThreads;
  const int wg_tid =
      is_producer ? threadIdx.x : (threadIdx.x - kProducerThreads);
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

  if (is_producer) {
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
      for (int kv_tile = 0; kv_tile < Tc_eff; ++kv_tile) {
        const int kv_major = KV_major_base + kv_tile * Bc;
        // kShareSmemQKV: wait for consumer to finish V phase of previous
        // kv_tile before overwriting shared smem with new QK data.
        if constexpr (kShareSmemQKV) {
          if (kv_tile > 0)
            ffpa::tma::wait_barrier(*v_done);
        }
        // QK phase: combined Q+K per d_chunk into one qk_full stage.
        for (int d_chunk = 0; d_chunk < kQKDChunks; ++d_chunk) {
          ffpa::tma::wait_barrier(qk_empty[d_chunk % kStageQK]);
          kDataType* q_dst = Q_tile_smem + (d_chunk % kStageQK) * kQSmemStride;
          kDataType* k_dst = K_tile_smem + (d_chunk % kStageQK) * kKSmemStride;
          const int minor = d_chunk * kQKDChunk;
          ffpa::tma::load_2d_no_arrive(q_dst, tma_q, minor, Q_major_base,
                                       qk_full[d_chunk % kStageQK]);
          ffpa::tma::load_2d_no_arrive(k_dst, tma_k, minor, kv_major,
                                       qk_full[d_chunk % kStageQK]);
          ffpa::tma::arrive_expect_tx(qk_full[d_chunk % kStageQK],
                                      kQTileBytes + kKTileBytes);
        }
        // kShareSmemQKV: wait for consumer to finish all QK ldmatrix before
        // writing V into the shared smem region.
        if constexpr (kShareSmemQKV) {
          ffpa::tma::wait_barrier(*qk_done);
        }
        // V phase: one V tile per v_chunk (kVDChunk cols per TMA load).
        for (int v_chunk = 0; v_chunk < kVDChunks; ++v_chunk) {
          ffpa::tma::wait_barrier(v_empty[v_chunk % kStagePV]);
          kDataType* v_dst = V_tile_smem + (v_chunk % kStagePV) * kVSmemStride;
          const int minor = v_chunk * kVDChunk;
          ffpa::tma::load_2d_no_arrive(v_dst, tma_v, minor, kv_major,
                                       v_full[v_chunk % kStagePV]);
          ffpa::tma::arrive_expect_tx(v_full[v_chunk % kStagePV], kVTileBytes);
        }
      }
    }
  } else {
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

#pragma unroll 1
    for (int tile_K_seqlen = 0; tile_K_seqlen < Tc_eff; ++tile_K_seqlen) {
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
          ffpa::prefill::sync_fetch_qkv_frags_s2r<0, 4, kQSmemStride, kMmaAtomM,
                                                  kMmaAtomN, kMmaAtomK, kPadQ,
                                                  kDataType, kQKDChunk>(
              smem_Q_base_ptr, &R_Q[0][0][0], warp_QP, 0, 0, stage, sub_col);
          // K s2r
#pragma unroll
          for (int j = 0; j < kValTileSeqLenK; ++j) {
            ffpa::prefill::sync_fetch_qkv_frags_s2r<
                0, 2, kKSmemStride, kMmaAtomM, kMmaAtomN, kMmaAtomK, kPadK,
                kDataType, kQKDChunk>(smem_K_base_ptr, &R_K[j][0], warp_KV, j,
                                      0, stage, sub_col);
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
      }
      // NOTE: no __syncthreads() here -- producer does not participate in the
      // consumer loop and would deadlock on a CTA-wide barrier. The QK MMA
      // results live entirely in registers (R_S); softmax below uses only
      // warp shuffles, so no cross-warp smem ordering is needed.

      // kShareSmemQKV: signal producer that all QK ldmatrix is done and the
      // shared smem is safe to overwrite with V data.
      if constexpr (kShareSmemQKV) {
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
          &R_S[0][0][0], scale, &lane_row_max_new[0][0],
          &lane_row_sum_new[0][0], &lane_block_row_max_old[0][0],
          &lane_block_row_sum_old[0][0]);
      if (dropout_p > 0.0f) {
        ffpa::prefill::sync_apply_dropout_to_p<kValTileSeqLenK,
                                               kMmaAccFloat32QK, kDataType>(
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
              ffpa::utils::fill_1D_regs<uint32_t, (kMmaAccFloat32PV) ? 4 : 2>(
                  R_O, 0);
#pragma unroll
              for (int tile_V_Bc = 0; tile_V_Bc < (Bc / kMmaAtomK);
                   ++tile_V_Bc) {
                ffpa::prefill::sync_fetch_qkv_frags_s2r<
                    1, 2, kVSmemStride, kMmaAtomM, kMmaAtomN, kMmaAtomK, kPadV,
                    kDataType, kVDChunk>(smem_V_base_ptr, &R_V[0][0], warp_KV,
                                         jj, tile_V_Bc, v_stage,
                                         subtile_col_offset);
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
              ffpa::prefill::sync_rescaling_tiling_o<
                  kOStorageAccFloat32, kMmaAccFloat32PV, kDataType>(
                  &R_D[0][0][0], &R_O[0], &rescale_o_factor_0[0],
                  &rescale_o_factor_1[0], tile_K_seqlen, j);
            }
          }
          // Release V stage: all kSubTilesV*2 j's consumed this tile.
          { [[maybe_unused]] auto token = v_empty[v_stage].arrive(); }
        }
        ffpa::prefill::sync_update_max_expsum(
            &lane_row_max_new[0][0], &lane_row_sum_new[0][0],
            &lane_block_row_max_old[0][0], &lane_block_row_sum_old[0][0],
            &rescale_o_factor_0[0], &rescale_o_factor_1[0]);
      }
      // kShareSmemQKV: signal producer that all V ldmatrix is done and the
      // shared smem is safe to overwrite with next kv_tile's QK data.
      if constexpr (kShareSmemQKV) {
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
}

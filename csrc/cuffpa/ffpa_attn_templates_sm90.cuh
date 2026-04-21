// ============================================================================
// FFPA experimental SM90+ kernel templates with TMA-based K/V staging.
//
// This header isolates all TMA-related kernel changes from the
// architecture-agnostic ``ffpa_attn_templates.cuh`` (which remains the
// fallback for SM < 9.0 and for ``tma=False`` callers). The SM90 kernel
// reuses the FFPA large-d Split-Q (FlashAttention-2) algorithm and only
// substitutes the per-tile K/V global-to-shared transfer with a TMA
// bulk tensor copy (``cp.async.bulk.tensor.2d.global.shared``), with a
// scratch + repack step to remain compatible with the existing handcrafted
// shared-memory swizzle / padding layout.
//
// Multi-stage (kStageQK / kStagePV >= 1) is supported. The TMA repack is
// synchronous (load -> barrier_wait -> repack -> __syncthreads) and writes
// directly to the requested destination stage slot, so the surrounding
// cp.async-based pipeline keeps issuing Q loads and waiting on its own
// commit groups exactly as in the fallback kernel; only the K/V transfer
// path differs.
// ============================================================================
#pragma once

#include "cuffpa/prefill.cuh"
#include "cuffpa/tma.cuh"

namespace ffpa {
namespace sm90 {

// Eligibility check for the experimental TMA SM90 large-d kernel.
//
// * ``kEligibleHeadDim``  : large-d kernel only kicks in for D > 64
//                           (small-d kernel is a different template).
// * ``kRequiresPaddedSmem``: TMA repack writes to padded shared (no XOR
//                           swizzle) since matching the kernel's
//                           hand-crafted swizzle inside repack would
//                           require an exact 128 B TMA swizzle layout
//                           map, which is not yet implemented.
// * ``kSupportsAllStages``: multi-stage is supported (kStageQK and
//                           kStagePV may each be >= 1).
template <const int kHeadDim, const int kStageQK, const int kStagePV, const int kPadQ,
          const int kPadK, const int kPadV>
struct ExperimentalTmaLargeDConfig {
  static constexpr bool kEligibleHeadDim = (kHeadDim > 64);
  static constexpr bool kRequiresPaddedSmem = (kPadK > 0 && kPadV > 0);
  static constexpr bool kSupportsAllStages = (kStageQK >= 1 && kStagePV >= 1);
  static constexpr bool kCanAttempt =
      kEligibleHeadDim && kRequiresPaddedSmem && kSupportsAllStages && (kPadQ >= 0);
};

}  // namespace sm90
}  // namespace ffpa

// ============================================================================
// ffpa_stages_split_q_large_d_sm90_template
// ----------------------------------------------------------------------------
// Mirror of ``ffpa_stages_split_q_large_d_template`` from
// ``ffpa_attn_templates.cuh`` but with K/V tile staging delegated to TMA
// (``cp.async.bulk.tensor.2d.global.shared``) followed by a thread-block
// repack into the existing padded shared-memory layout. All other
// algorithmic behavior (online softmax, causal mask, GQA/MQA,
// kStageQK/kStagePV pipeline flow, kPersistQg2s/kPersistQs2r/kRegPipeKV
// options) is identical to the fallback kernel; do NOT diverge those paths
// here.
// ============================================================================
template <typename kDataType, const int kHeadDim, const int kMmaAtomM, const int kMmaAtomN,
          const int kMmaAtomK, const int kMmaTileSeqLenQ, const int kMmaTileSeqLenK,
          const int kMmaTileSeqLenP, const int kMmaTileHeadDimV, const int kValTileSeqLenQ,
          const int kValTileSeqLenK, const int kValTileSeqLenP, const int kValTileHeadDimV,
          const int kMmaAccFloat32QK, const int kMmaAccFloat32PV, const int kOStorageAccFloat32,
          const int kPrefetchQK, const int kPrefetchPV, const int kShareSmemQKV,
          const int kPersistQs2r, const int kPersistQg2s, const int kRegPipeKV, const int kStageQK,
          const int kStagePV, const int kPadQ, const int kPadK, const int kPadV>
__global__ void __launch_bounds__(WARP_SIZE* kMmaTileSeqLenQ* kMmaTileSeqLenK)
    ffpa_stages_split_q_large_d_sm90_template(
        const kDataType* __restrict__ Q, const kDataType* __restrict__ K,
        const kDataType* __restrict__ V, kDataType* __restrict__ O, const int Nq, const int Nkv,
        const int Nh, const int Nh_kv, const float scale, const int Tc, const int causal,
        const CUtensorMap* __restrict__ K_tma_desc, const CUtensorMap* __restrict__ V_tma_desc) {
  ffpa::prefill::check_large_d_compiling_states<
      kHeadDim, kMmaAtomM, kMmaAtomN, kMmaAtomK, kMmaTileSeqLenQ, kMmaTileSeqLenK, kMmaTileSeqLenP,
      kMmaTileHeadDimV, kValTileSeqLenQ, kValTileSeqLenK, kValTileSeqLenP, kValTileHeadDimV,
      kMmaAccFloat32QK, kMmaAccFloat32PV, kOStorageAccFloat32, kPrefetchQK, kPrefetchPV,
      kShareSmemQKV, kPersistQs2r, kPersistQg2s, kRegPipeKV, kStageQK, kStagePV, kPadQ, kPadK,
      kPadV>();
  constexpr int Br = kMmaAtomM * kMmaTileSeqLenQ * kValTileSeqLenQ;
  constexpr int Bc = kMmaAtomN * kMmaTileSeqLenK * kValTileSeqLenK;
  constexpr int kNumThreads = WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK;

#ifdef ENABLE_FFPA_LAUNCH_GRID_DNHB
  const int Nb_id = blockIdx.z;
  const int Nh_id = blockIdx.y;
#else
  const int Nb_id = blockIdx.y / Nh;
  const int Nh_id = blockIdx.y % Nh;
#endif
  const int Q_tile_id = blockIdx.x;
  const int O_tile_id = Q_tile_id;
  const int warp_QP = threadIdx.x / WARP_SIZE;
  constexpr int warp_KV = 0;
  const int group_size = Nh / Nh_kv;
  const int kv_head_idx = Nh_id / group_size;
  const int Q_gmem_offset = ((Nb_id * Nh * Nq * kHeadDim) + (Nh_id * Nq * kHeadDim));
  const int K_gmem_offset = ((Nb_id * Nh_kv * Nkv * kHeadDim) + (kv_head_idx * Nkv * kHeadDim));
  const int V_gmem_offset = K_gmem_offset;
  const int O_gmem_offset = Q_gmem_offset;

  if ((Q_tile_id * Br) >= Nq)
    return;

  extern __shared__ __align__(16) unsigned char ffpa_smem_raw[];
  kDataType* smem = reinterpret_cast<kDataType*>(ffpa_smem_raw);
  constexpr int Q_tile_size = Br * (kMmaAtomK + kPadQ);
  constexpr int K_tile_size = Bc * (kMmaAtomK + kPadK);
  constexpr int V_tile_size = Bc * (kMmaAtomN * 2 + kPadV);
  kDataType* Q_tile_smem = smem;
  kDataType* K_tile_smem = (Q_tile_smem + (kPersistQg2s ? ((kHeadDim / kMmaAtomK) * Q_tile_size)
                                                        : (kStageQK * Q_tile_size)));
  kDataType* V_tile_smem = (kShareSmemQKV ? Q_tile_smem : K_tile_smem + kStageQK * K_tile_size);
  // TMA scratch: a single contiguous tile-sized slot for K and V each is
  // sufficient because each repack call is internally synchronous (load,
  // barrier wait, repack, syncthreads) and the temp slot is consumed
  // before the next call. The launcher reserves the matching dynamic
  // shared-memory size via getExperimentalTmaSm90ScratchSize().
  constexpr int kTmaTmpKTileSize = Bc * kMmaAtomK;
  constexpr int kTmaTmpVTileSize = Bc * (kMmaAtomN * 2);
  kDataType* K_tma_tmp_smem = (kShareSmemQKV ? (K_tile_smem + kStageQK * K_tile_size)
                                             : (V_tile_smem + kStagePV * V_tile_size));
  kDataType* V_tma_tmp_smem = K_tma_tmp_smem + kTmaTmpKTileSize;
  (void)kTmaTmpVTileSize;
  const uint32_t smem_Q_base_ptr = __cvta_generic_to_shared(Q_tile_smem);
  const uint32_t smem_K_base_ptr = __cvta_generic_to_shared(K_tile_smem);
  const uint32_t smem_V_base_ptr = __cvta_generic_to_shared(V_tile_smem);

  __shared__ alignas(alignof(
      ffpa::tma::barrier_t)) unsigned char K_tma_barrier_storage[sizeof(ffpa::tma::barrier_t)];
  __shared__ alignas(alignof(
      ffpa::tma::barrier_t)) unsigned char V_tma_barrier_storage[sizeof(ffpa::tma::barrier_t)];
  ffpa::tma::barrier_t* K_tma_barrier =
      reinterpret_cast<ffpa::tma::barrier_t*>(K_tma_barrier_storage);
  ffpa::tma::barrier_t* V_tma_barrier =
      reinterpret_cast<ffpa::tma::barrier_t*>(V_tma_barrier_storage);
  if (threadIdx.x == 0) {
    ffpa::tma::init_barrier(K_tma_barrier, 1);
    ffpa::tma::init_barrier(V_tma_barrier, 1);
  }
  __syncthreads();

  // ---------- helper lambdas: TMA-or-cpasync K/V tile stagers ----------
  // These wrap the repack helper with a cp.async fallback so tail tiles
  // (where Bc would overrun Nkv) and any future eligibility miss still
  // produce a correct K/V tile in the destination shared slot.
  auto stage_K_tile = [&](int tile_K_seqlen, int d_tile, int dst_stage) -> void {
    if (!ffpa::tma::load_2d_to_smem_repack<Bc, K_tile_size, kHeadDim, kMmaAtomK, kNumThreads,
                                           kPadK>(K_tile_smem, K_tma_tmp_smem, K_tma_desc,
                                                  tile_K_seqlen * Bc, d_tile, dst_stage, 0, Nkv,
                                                  *K_tma_barrier)) {
      ffpa::prefill::cp_async_qkv_g2s<Bc, K_tile_size, kHeadDim, kMmaAtomK, kNumThreads, kPadK>(
          smem_K_base_ptr, K, K_gmem_offset, tile_K_seqlen, d_tile, dst_stage, Nkv);
    }
  };
  auto stage_V_tile = [&](int tile_K_seqlen, int d_tile, int dst_stage) -> void {
    if (!ffpa::tma::load_2d_to_smem_repack<Bc, V_tile_size, kHeadDim, kMmaAtomN * 2, kNumThreads,
                                           kPadV>(V_tile_smem, V_tma_tmp_smem, V_tma_desc,
                                                  tile_K_seqlen * Bc, d_tile, dst_stage, 0, Nkv,
                                                  *V_tma_barrier)) {
      ffpa::prefill::cp_async_qkv_g2s<Bc, V_tile_size, kHeadDim, kMmaAtomN * 2, kNumThreads, kPadV>(
          smem_V_base_ptr, V, V_gmem_offset, tile_K_seqlen, d_tile, dst_stage, Nkv);
    }
  };

  if constexpr (kPersistQg2s) {
#pragma unroll
    for (int tile_K_d = 0; tile_K_d < (kHeadDim / kMmaAtomK); ++tile_K_d) {
      ffpa::prefill::cp_async_qkv_g2s<Br, Q_tile_size, kHeadDim, kMmaAtomK, kNumThreads, kPadQ>(
          smem_Q_base_ptr, Q, Q_gmem_offset, Q_tile_id, tile_K_d, tile_K_d, Nq);
      ffpa::cp_async::commit_group();
    }
  }

  float lane_block_row_max_old[kValTileSeqLenQ][2];
  float lane_block_row_sum_old[kValTileSeqLenQ][2];
  ffpa::utils::fill_2D_regs<float, kValTileSeqLenQ, 2>(lane_block_row_max_old, -INFINITY);
  ffpa::utils::fill_2D_regs<float, kValTileSeqLenQ, 2>(lane_block_row_sum_old, 0.0f);

  uint32_t R_Q[kValTileSeqLenQ][(kPersistQs2r) ? (kHeadDim / kMmaAtomK) : 1][4];
  uint32_t R_K[(kRegPipeKV) ? 2 : kValTileSeqLenK][2];
  uint32_t R_V[(kRegPipeKV) ? 2 : 1][2];
  uint32_t R_S[kValTileSeqLenQ][kValTileSeqLenK][(kMmaAccFloat32QK) ? 4 : 2];
  uint32_t R_O[(kMmaAccFloat32PV) ? 4 : 2];
  uint32_t R_D[kValTileSeqLenP][kValTileHeadDimV][(kOStorageAccFloat32) ? 4 : 2];
  ffpa::utils::fill_3D_regs<uint32_t, kValTileSeqLenP, kValTileHeadDimV,
                            ((kOStorageAccFloat32) ? 4 : 2)>(R_D, 0);

  uint32_t reg_st_idx = 0;
  uint32_t reg_ld_idx = 1;

  const int Br_base = Q_tile_id * Br;
  const int kv_offset = Nkv - Nq;
  const int causal_thresh_row0 = Br_base + kv_offset;
  const int Tc_eff = causal ? min(Tc, ((Br_base + Br - 1 + kv_offset) / Bc) + 1) : Tc;
  const int mask_start_tile = causal ? max(0, (causal_thresh_row0 + 1) / Bc) : INT_MAX;

#pragma unroll 1
  for (int tile_K_seqlen = 0; tile_K_seqlen < Tc_eff; ++tile_K_seqlen) {
    if constexpr (kPrefetchQK) {
      if constexpr (kStageQK > 1) {
        if (tile_K_seqlen == 0) {
#pragma unroll
          for (int stage = 0; stage < (kStageQK - 1); ++stage) {
            if constexpr (!kPersistQg2s) {
              ffpa::prefill::cp_async_qkv_g2s<Br, Q_tile_size, kHeadDim, kMmaAtomK, kNumThreads,
                                              kPadQ>(smem_Q_base_ptr, Q, Q_gmem_offset, Q_tile_id,
                                                     stage, stage, Nq);
            }
            stage_K_tile(tile_K_seqlen, stage, stage);
            ffpa::cp_async::commit_group();
          }
          ffpa::cp_async::wait_group<(kStageQK - 2)>();
          __syncthreads();
        } else {
          ffpa::cp_async::wait_group<(kStageQK - 2)>();
          __syncthreads();
        }
      }
    } else {
      if constexpr (kStageQK > 1) {
#pragma unroll
        for (int stage = 0; stage < (kStageQK - 1); ++stage) {
          if constexpr (kPersistQs2r) {
            if (tile_K_seqlen == 0) {
              ffpa::prefill::cp_async_qkv_g2s<Br, Q_tile_size, kHeadDim, kMmaAtomK, kNumThreads,
                                              kPadQ>(smem_Q_base_ptr, Q, Q_gmem_offset, Q_tile_id,
                                                     stage, stage, Nq);
            }
          } else {
            if constexpr (!kPersistQg2s) {
              ffpa::prefill::cp_async_qkv_g2s<Br, Q_tile_size, kHeadDim, kMmaAtomK, kNumThreads,
                                              kPadQ>(smem_Q_base_ptr, Q, Q_gmem_offset, Q_tile_id,
                                                     stage, stage, Nq);
            }
          }
          stage_K_tile(tile_K_seqlen, stage, stage);
          ffpa::cp_async::commit_group();
        }
        ffpa::cp_async::wait_group<(kStageQK - 2)>();
        __syncthreads();
      }
    }

    if constexpr ((!kShareSmemQKV) && kPrefetchPV && kStagePV > 1) {
      if constexpr (kStagePV > 1) {
#pragma unroll
        for (int stage = 0; stage < (kStagePV - 1); ++stage) {
          stage_V_tile(tile_K_seqlen, stage, stage);
          ffpa::cp_async::commit_group();
        }
      }
    }

    ffpa::utils::fill_3D_regs<uint32_t, kValTileSeqLenQ, kValTileSeqLenK,
                              (kMmaAccFloat32QK) ? 4 : 2>(R_S, 0);
#pragma unroll
    for (int tile_K_d = 0; tile_K_d < (kHeadDim / kMmaAtomK); ++tile_K_d) {
      const int smem_sel = (tile_K_d) % kStageQK;
      const int smem_sel_next = (tile_K_d + (kStageQK - 1)) % kStageQK;
      if constexpr (kPersistQs2r) {
        if (tile_K_seqlen == 0) {
          ffpa::prefill::cp_async_qkv_g2s<Br, Q_tile_size, kHeadDim, kMmaAtomK, kNumThreads, kPadQ>(
              smem_Q_base_ptr, Q, Q_gmem_offset, Q_tile_id,
              (kStageQK > 1) ? (tile_K_d + (kStageQK - 1)) : tile_K_d,
              (kStageQK > 1) ? smem_sel_next : smem_sel, Nq);
        }
      } else {
        if constexpr (!kPersistQg2s) {
          ffpa::prefill::cp_async_qkv_g2s<Br, Q_tile_size, kHeadDim, kMmaAtomK, kNumThreads, kPadQ>(
              smem_Q_base_ptr, Q, Q_gmem_offset, Q_tile_id,
              (kStageQK > 1) ? (tile_K_d + (kStageQK - 1)) : tile_K_d,
              (kStageQK > 1) ? smem_sel_next : smem_sel, Nq);
        }
      }
      stage_K_tile(tile_K_seqlen, (kStageQK > 1) ? (tile_K_d + (kStageQK - 1)) : tile_K_d,
                   (kStageQK > 1) ? smem_sel_next : smem_sel);
      ffpa::cp_async::commit_group();

      if constexpr (kStageQK <= 1) {
        ffpa::cp_async::wait_group<0>();
        __syncthreads();
      }

      static_assert(kValTileSeqLenQ == 1);
      {
        if constexpr (kPersistQs2r) {
          if (tile_K_seqlen == 0) {
            ffpa::prefill::sync_fetch_qkv_frags_s2r<0, 4, Q_tile_size, kMmaAtomM, kMmaAtomN,
                                                    kMmaAtomK, kPadQ, kDataType>(
                smem_Q_base_ptr, &R_Q[0][tile_K_d][0], warp_QP, 0, 0, smem_sel);
          }
        } else {
          if constexpr (!kPersistQg2s) {
            ffpa::prefill::sync_fetch_qkv_frags_s2r<0, 4, Q_tile_size, kMmaAtomM, kMmaAtomN,
                                                    kMmaAtomK, kPadQ, kDataType>(
                smem_Q_base_ptr, &R_Q[0][0][0], warp_QP, 0, 0, smem_sel);
          } else {
            ffpa::prefill::sync_fetch_qkv_frags_s2r<0, 4, Q_tile_size, kMmaAtomM, kMmaAtomN,
                                                    kMmaAtomK, kPadQ, kDataType>(
                smem_Q_base_ptr, &R_Q[0][0][0], warp_QP, 0, 0, tile_K_d);
          }
        }
      }

      reg_st_idx = 0;
      reg_ld_idx = 1;
      if constexpr (!kRegPipeKV) {
#pragma unroll
        for (int j = 0; j < kValTileSeqLenK; ++j) {
          ffpa::prefill::sync_fetch_qkv_frags_s2r<0, 2, K_tile_size, kMmaAtomM, kMmaAtomN,
                                                  kMmaAtomK, kPadK, kDataType>(
              smem_K_base_ptr, &R_K[j][0], warp_KV, j, 0, smem_sel);
        }
      } else {
        ffpa::prefill::sync_fetch_qkv_frags_s2r<0, 2, K_tile_size, kMmaAtomM, kMmaAtomN, kMmaAtomK,
                                                kPadK, kDataType>(
            smem_K_base_ptr, &R_K[reg_st_idx][0], warp_KV, 0, 0, smem_sel);
      }

      if constexpr ((kShareSmemQKV) && kPrefetchPV && kStagePV > 1) {
        if (tile_K_d == (kHeadDim / kMmaAtomK - 1)) {
          __syncthreads();
          if constexpr (kStagePV > 1) {
#pragma unroll
            for (int stage = 0; stage < (kStagePV - 1); ++stage) {
              stage_V_tile(tile_K_seqlen, stage, stage);
              ffpa::cp_async::commit_group();
            }
          }
        }
      }

      static_assert(kValTileSeqLenQ == 1);
      {
        const int q_offset = (kPersistQs2r) ? (tile_K_d) : 0;
#pragma unroll
        for (int j = 0; j < kValTileSeqLenK; ++j) {
          reg_st_idx ^= 1;
          reg_ld_idx ^= 1;
          if constexpr (kRegPipeKV) {
            if ((j + 1) < kValTileSeqLenK) {
              ffpa::prefill::sync_fetch_qkv_frags_s2r<0, 2, K_tile_size, kMmaAtomM, kMmaAtomN,
                                                      kMmaAtomK, kPadK, kDataType>(
                  smem_K_base_ptr, &R_K[reg_st_idx][0], warp_KV, (j + 1), 0, smem_sel);
            }
          }
          const int k_offset = (kRegPipeKV) ? reg_ld_idx : j;
          if constexpr (kMmaAccFloat32QK) {
            ffpa::mma::m16n8k16_abf32<kDataType, ffpa::mma::MMAMode::kInplaceUpdate>(
                &R_S[0][j][0], &R_S[0][j][1], &R_S[0][j][2], &R_S[0][j][3], &R_Q[0][q_offset][0],
                &R_Q[0][q_offset][1], &R_Q[0][q_offset][2], &R_Q[0][q_offset][3], &R_K[k_offset][0],
                &R_K[k_offset][1]);
          } else {
            ffpa::mma::m16n8k16_f16f16f16<ffpa::mma::MMAMode::kInplaceUpdate>(
                &R_S[0][j][0], &R_S[0][j][1], &R_Q[0][q_offset][0], &R_Q[0][q_offset][1],
                &R_Q[0][q_offset][2], &R_Q[0][q_offset][3], &R_K[k_offset][0], &R_K[k_offset][1]);
          }
        }
      }

      if constexpr (kStageQK > 1) {
        if (tile_K_d < (kHeadDim / kMmaAtomK - 1)) {
          ffpa::cp_async::wait_group<(kStageQK - 2)>();
          __syncthreads();
        }
      }
      if constexpr (kStageQK < 2) {
        __syncthreads();
      }
    }
    __syncthreads();

    static_assert(kValTileSeqLenP == 1);
    if constexpr (!kPrefetchPV) {
      if constexpr (kStagePV > 1) {
#pragma unroll
        for (int stage = 0; stage < (kStagePV - 1); ++stage) {
          stage_V_tile(tile_K_seqlen, stage, stage);
          ffpa::cp_async::commit_group();
        }
      }
    }

    float lane_row_max_new[kValTileSeqLenQ][2];
    float lane_row_sum_new[kValTileSeqLenQ][2];
    ffpa::utils::fill_2D_regs<float, kValTileSeqLenQ, 2>(lane_row_max_new, -INFINITY);
    ffpa::utils::fill_2D_regs<float, kValTileSeqLenQ, 2>(lane_row_sum_new, 0.0f);

    static_assert(kValTileSeqLenQ == 1);
    {
      const int kv_valid_local = Nkv - tile_K_seqlen * Bc;
      if (kv_valid_local < Bc) {
        ffpa::prefill::sync_apply_kv_mask<kValTileSeqLenK, kMmaAccFloat32QK, kDataType>(
            &R_S[0][0][0], kv_valid_local);
      }
    }
    if (tile_K_seqlen >= mask_start_tile) {
      ffpa::prefill::sync_apply_causal_mask<kValTileSeqLenK, kMmaAccFloat32QK, kDataType>(
          &R_S[0][0][0], warp_QP, Br_base, tile_K_seqlen * Bc, kv_offset);
    }
    ffpa::prefill::sync_online_safe_softmax<kValTileSeqLenK, kMmaAccFloat32QK, kDataType>(
        &R_S[0][0][0], scale, &lane_row_max_new[0][0], &lane_row_sum_new[0][0],
        &lane_block_row_max_old[0][0], &lane_block_row_sum_old[0][0]);

    if constexpr (kStagePV > 1) {
      ffpa::cp_async::wait_group<(kStagePV - 2)>();
      __syncthreads();
    }

    if constexpr ((!kShareSmemQKV) && kPrefetchQK && kStageQK > 1) {
      if ((tile_K_seqlen + 1) < Tc_eff) {
#pragma unroll
        for (int stage = 0; stage < (kStageQK - 1); ++stage) {
          if constexpr (!kPersistQs2r) {
            ffpa::prefill::cp_async_qkv_g2s<Br, Q_tile_size, kHeadDim, kMmaAtomK, kNumThreads,
                                            kPadQ>(smem_Q_base_ptr, Q, Q_gmem_offset, Q_tile_id,
                                                   stage, stage, Nq);
          }
          stage_K_tile(tile_K_seqlen + 1, stage, stage);
          ffpa::cp_async::commit_group();
        }
      }
    }

    static_assert(kValTileSeqLenP == 1);
    {
      float rescale_o_factor_0[1];
      float rescale_o_factor_1[1];
      ffpa::prefill::sync_precompute_rescale_factors(&rescale_o_factor_0[0], &rescale_o_factor_1[0],
                                                     &lane_row_max_new[0][0],
                                                     &lane_block_row_max_old[0][0], tile_K_seqlen);

#pragma unroll
      for (int j = 0; j < kValTileHeadDimV; ++j) {
        const int tile_V_d = (j >> 1);
        const int smem_sel_v = (tile_V_d) % kStagePV;
        const int smem_sel_v_next = (tile_V_d + (kStagePV - 1)) % kStagePV;
        if (j % 2 == 0) {
          stage_V_tile(tile_K_seqlen, (kStagePV > 1) ? (tile_V_d + (kStagePV - 1)) : tile_V_d,
                       (kStagePV > 1) ? smem_sel_v_next : smem_sel_v);
          ffpa::cp_async::commit_group();
          if constexpr (kStagePV <= 1) {
            ffpa::cp_async::wait_group<0>();
            __syncthreads();
          }
        }

        reg_st_idx = 0;
        reg_ld_idx = 1;
        if constexpr (kRegPipeKV) {
          ffpa::prefill::sync_fetch_qkv_frags_s2r<1, 2, V_tile_size, kMmaAtomM, kMmaAtomN,
                                                  kMmaAtomK, kPadV, kDataType>(
              smem_V_base_ptr, &R_V[reg_st_idx][0], warp_KV, (j % 2), 0, smem_sel_v);
        }

        ffpa::utils::fill_1D_regs<uint32_t, (kMmaAccFloat32PV) ? 4 : 2>(R_O, 0);
#pragma unroll
        for (int tile_V_Bc = 0; tile_V_Bc < (Bc / kMmaAtomK); ++tile_V_Bc) {
          if constexpr ((kShareSmemQKV) && kPrefetchQK && kStageQK > 1) {
            if (j == (kValTileHeadDimV - 1) && tile_V_Bc == (Bc / kMmaAtomK - 1) &&
                (tile_K_seqlen + 1) < Tc_eff) {
              __syncthreads();
              if constexpr (kStageQK > 1) {
#pragma unroll
                for (int stage = 0; stage < (kStageQK - 1); ++stage) {
                  if constexpr (!kPersistQs2r) {
                    ffpa::prefill::cp_async_qkv_g2s<Br, Q_tile_size, kHeadDim, kMmaAtomK,
                                                    kNumThreads, kPadQ>(
                        smem_Q_base_ptr, Q, Q_gmem_offset, Q_tile_id, stage, stage, Nq);
                  }
                  stage_K_tile(tile_K_seqlen + 1, stage, stage);
                  ffpa::cp_async::commit_group();
                }
              }
            }
          }

          reg_st_idx ^= 1;
          reg_ld_idx ^= 1;
          if constexpr (!kRegPipeKV) {
            ffpa::prefill::sync_fetch_qkv_frags_s2r<1, 2, V_tile_size, kMmaAtomM, kMmaAtomN,
                                                    kMmaAtomK, kPadV, kDataType>(
                smem_V_base_ptr, &R_V[0][0], warp_KV, (j % 2), tile_V_Bc, smem_sel_v);
          } else {
            if ((tile_V_Bc + 1) < (Bc / kMmaAtomK)) {
              ffpa::prefill::sync_fetch_qkv_frags_s2r<1, 2, V_tile_size, kMmaAtomM, kMmaAtomN,
                                                      kMmaAtomK, kPadV, kDataType>(
                  smem_V_base_ptr, &R_V[reg_st_idx][0], warp_KV, (j % 2), (tile_V_Bc + 1),
                  smem_sel_v);
            }
          }

          const int p_offset = tile_V_Bc * 2;
          const int v_offset = (kRegPipeKV) ? reg_ld_idx : 0;
          if constexpr (kMmaAccFloat32PV) {
            ffpa::mma::m16n8k16_abf32<kDataType, ffpa::mma::MMAMode::kInplaceUpdate>(
                &R_O[0], &R_O[1], &R_O[2], &R_O[3], &R_S[0][p_offset][0], &R_S[0][p_offset][1],
                &R_S[0][p_offset + 1][0], &R_S[0][p_offset + 1][1], &R_V[v_offset][0],
                &R_V[v_offset][1]);
          } else {
            ffpa::mma::m16n8k16_f16f16f16<ffpa::mma::MMAMode::kInplaceUpdate>(
                &R_O[0], &R_O[1], &R_S[0][p_offset][0], &R_S[0][p_offset][1],
                &R_S[0][p_offset + 1][0], &R_S[0][p_offset + 1][1], &R_V[v_offset][0],
                &R_V[v_offset][1]);
          }
        }
        if constexpr (kStagePV < 2) {
          __syncthreads();
        }

        ffpa::prefill::sync_rescaling_tiling_o<kOStorageAccFloat32, kMmaAccFloat32PV, kDataType>(
            &R_D[0][0][0], &R_O[0], &rescale_o_factor_0[0], &rescale_o_factor_1[0], tile_K_seqlen,
            j);

        if constexpr (kStagePV > 1) {
          if (j < (kValTileHeadDimV - 1)) {
            ffpa::cp_async::wait_group<(kStagePV - 2)>();
            __syncthreads();
          }
        }
      }

      ffpa::prefill::sync_update_max_expsum(
          &lane_row_max_new[0][0], &lane_row_sum_new[0][0], &lane_block_row_max_old[0][0],
          &lane_block_row_sum_old[0][0], &rescale_o_factor_0[0], &rescale_o_factor_1[0]);
    }
    __syncthreads();
  }
  __syncthreads();

  static_assert(kValTileSeqLenP == 1);
  ffpa::prefill::sync_rescaling_final_o<kValTileHeadDimV, kOStorageAccFloat32, kDataType>(
      &R_D[0][0][0], &lane_block_row_sum_old[0][0]);

  static_assert(kValTileSeqLenP == 1);
  ffpa::prefill::sync_store_o_r2g<Br, kHeadDim, kMmaAtomM, kMmaAtomN, kValTileHeadDimV,
                                  kOStorageAccFloat32, kDataType>(
      O, O_gmem_offset, O_tile_id, warp_QP, &R_D[0][0][0], &R_Q[0][0][0], &R_K[0][0], Nq);
}

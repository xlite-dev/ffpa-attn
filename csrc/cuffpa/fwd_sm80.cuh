#pragma once
#include "prefill.cuh"  // ffpa::prefill

// ============================================================================
// ffpa_attn_split_d_fwd_template
// ----------------------------------------------------------------------------
// Split-Q (FlashAttention-2) prefill attention kernel tuned for *large* head
// dimensions (D > ~128). Each thread block owns a fixed slice of Q rows
// ([Br,D]) and streams K/V tiles ([Bc,D]) through an online safe-softmax.
// The "large-d" variant tiles the head-dim axis into `kHeadDim / kMmaAtomK`
// inner steps so SMEM stays O(1) in D.
//
// Supports cross-attention (Nq may differ from Nkv; Nk == Nv is required
// and is asserted by the launcher), grouped-query / multi-query attention
// (K/V use Nh_kv heads shared across group_size = Nh / Nh_kv Q heads), and
// causal attention as a runtime parameter.
//
// ----------------------------------------------------------------------------
// Template parameters (compile-time configuration)
// ----------------------------------------------------------------------------
//   kDataType             Activation dtype: __half (fp16) or __nv_bfloat16.
//   kHeadDim              Head dimension D (32, 64, ..., 1024), multiple of
//                         kMmaAtomK.
//   kMmaAtomM/N/K         MMA atom shape for m16n8k16 (fixed to 16/8/16).
//   kMmaTileSeqLenQ/K/P   Warp-level MMA tiling along Q rows / K cols / P rows.
//                         Thread block uses warps = kMmaTileSeqLenQ *
//                         kMmaTileSeqLenK.
//   kMmaTileHeadDimV      Warp-level MMA tiling along V cols (head-dim).
//   kValTileSeqLenQ/K/P   Extra value-level tiling multipliers yielding the
//                         tile sizes Br =
//                         kMmaAtomM*kMmaTileSeqLenQ*kValTileSeqLenQ, Bc =
//                         kMmaAtomN*kMmaTileSeqLenK*kValTileSeqLenK.
//   kValTileHeadDimV      Value tiling over head-dim for P@V.
//   kMmaAccFloat32QK/PV   0 -> fp16 MMA accumulator, 1 -> fp32 accumulator for
//                         the Q@K^T / P@V stage respectively (bf16 requires 1).
//   kOStorageAccFloat32   0 -> store O accumulator as fp16, 1 -> as fp32.
//   kPrefetchQK/PV        0/1, enable async prefetch of next QK / PV tiles.
//   kShareSmemQKV         0/1, let V reuse QK SMEM after Q@K^T to save SRAMz .
//   kPersistQs2r          0/1, persist Q in registers across all KV tiles.
//   kPersistQg2s          0/1, persist Q in SMEM across all KV tiles.
//   kRegPipeKV            0/1, register ping-pong between ldmatrix and mma.
//   kStageQK/kStagePV     Multi-stage cp.async pipeline depth (<= 4) for QK /
//   PV. kPadQ/kPadK/kPadV     0 selects SMEM swizzle (bank-conflict free), >0
//                         selects padded layout with that padding width.
//
// ----------------------------------------------------------------------------
// Runtime parameters (grid/block invariants)
// ----------------------------------------------------------------------------
//   Q     in : fp16/bf16 tensor, shape [Nb, Nh, Nq,  D] (BHND layout).
//   K     in : fp16/bf16 tensor, shape [Nb, Nh_kv, Nkv, D].
//   V     in : fp16/bf16 tensor, shape [Nb, Nh_kv, Nkv, D].
//   O     out: fp16/bf16 tensor, shape [Nb, Nh, Nq,  D]; written in place.
//   Nb       : Batch size; combined with Nh to fan out across gridDim.y
//              (or gridDim.{y,z} under the DNHB layout).
//   Nq       : Query sequence length. Drives gridDim.x = div_ceil(Nq, Br) and
//              the per-row store predicate; need not be a multiple of Br.
//   Nkv      : Key/Value sequence length (Nk == Nv). Drives the KV loop bound
//              Tc = div_ceil(Nkv, Bc) and the tail-tile softmax mask; need not
//              be a multiple of Bc.
//   Nh       : Number of Q attention heads (Nh_q); used to decode
//              (batch, head) from blockIdx.y when the grid is laid out as
//              (Tr, Nb*Nh).
//   Nh_kv    : Number of K/V heads (GQA/MQA). Must divide Nh; the Q head
//              index maps to ``kv_head_idx = Nh_id / (Nh / Nh_kv)``.
//   scale    : Softmax pre-scale, typically 1 / sqrt(D).
//   Tc       : Precomputed KV tile count = div_ceil(Nkv, Bc).
//   causal   : 0/1 runtime flag. When non-zero, Q row ``r`` only attends
//              to KV positions ``k <= r + (Nkv - Nq)`` (standard causal
//              with queries aligned to the KV tail; the launcher enforces
//              Nkv >= Nq). The kernel clips the KV loop to the last tile
//              with any visible key and applies a per-fragment -inf mask
//              on tiles that straddle the diagonal; tiles fully below the
//              diagonal pay only one compare-and-branch.
//
// ----------------------------------------------------------------------------
// Grid / block layout
// ----------------------------------------------------------------------------
//   Block: (WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK, 1, 1).
//   Grid : default     (div_ceil(Nq, Br), Nb*Nh, 1);
//          DNHB layout (div_ceil(Nq, Br), Nh, Nb) if
//          ENABLE_FFPA_LAUNCH_GRID_DNHB.
//   Dynamic SMEM size is computed by getConfigQKVSmemMaxSize<...>() in the
//   launcher and must be set via cudaFuncSetAttribute before launch.
// ============================================================================
template <typename kDataType, const int kHeadDim, const int kMmaAtomM,
          const int kMmaAtomN, const int kMmaAtomK, const int kMmaTileSeqLenQ,
          const int kMmaTileSeqLenK, const int kMmaTileSeqLenP,
          const int kMmaTileHeadDimV, const int kValTileSeqLenQ,
          const int kValTileSeqLenK, const int kValTileSeqLenP,
          const int kValTileHeadDimV, const int kMmaAccFloat32QK,
          const int kMmaAccFloat32PV, const int kOStorageAccFloat32,
          const int kPrefetchQK, const int kPrefetchPV, const int kShareSmemQKV,
          const int kPersistQs2r, const int kPersistQg2s, const int kRegPipeKV,
          const int kStageQK, const int kStagePV, const int kPadQ,
          const int kPadK, const int kPadV>
__global__ void __launch_bounds__(WARP_SIZE* kMmaTileSeqLenQ* kMmaTileSeqLenK)
    ffpa_attn_split_d_fwd_template(
        const kDataType* __restrict__ Q, const kDataType* __restrict__ K,
        const kDataType* __restrict__ V, kDataType* __restrict__ O,
        float* __restrict__ softmax_lse, const int Nq, const int Nkv,
        const int Nh, const int Nh_kv, const float scale, const int Tc,
        const int causal, const void* __restrict__ attn_bias,
        const int attn_bias_dtype, const long long attn_bias_stride_b,
        const long long attn_bias_stride_h, const long long attn_bias_stride_m,
        const long long attn_bias_stride_n, const float dropout_p,
        const unsigned long long philox_seed,
        const unsigned long long philox_offset) {
  ffpa::prefill::check_large_d_compiling_states<
      kHeadDim, kMmaAtomM, kMmaAtomN, kMmaAtomK, kMmaTileSeqLenQ,
      kMmaTileSeqLenK, kMmaTileSeqLenP, kMmaTileHeadDimV, kValTileSeqLenQ,
      kValTileSeqLenK, kValTileSeqLenP, kValTileHeadDimV, kMmaAccFloat32QK,
      kMmaAccFloat32PV, kOStorageAccFloat32, kPrefetchQK, kPrefetchPV,
      kShareSmemQKV, kPersistQs2r, kPersistQg2s, kRegPipeKV, kStageQK, kStagePV,
      kPadQ, kPadK, kPadV>();
  constexpr int Br = kMmaAtomM * kMmaTileSeqLenQ * kValTileSeqLenQ;
  constexpr int Bc = kMmaAtomN * kMmaTileSeqLenK * kValTileSeqLenK;
  constexpr int kNumThreads = WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK;

#ifdef ENABLE_FFPA_LAUNCH_GRID_DNHB
  // grid(div_ceil(Nq, Br), Nh, Nb), (x,y,z)
  const int Nb_id = blockIdx.z;  // Batch size
  const int Nh_id = blockIdx.y;  // Head num
#else
  // grid(div_ceil(Nq, Br), Nb * Nh), (x,y,z)
  const int Nb_id = blockIdx.y / Nh;  // Batch size
  const int Nh_id = blockIdx.y % Nh;  // Head num
#endif
  const int Q_tile_id = blockIdx.x;             // Q tile_id, range [0, Tr]
  const int O_tile_id = Q_tile_id;              // O tile_id, same as Q.
  const int warp_QP = threadIdx.x / WARP_SIZE;  // 0,1,2,3 or 0~7
  constexpr int warp_KV = 0;                    // 0
  // GQA: K/V have Nh_kv heads shared across group_size = Nh / Nh_kv Q heads;
  // MHA path preserved when Nh_kv == Nh (group_size == 1, Nh_id ==
  // kv_head_idx).
  const int group_size = Nh / Nh_kv;
  const int kv_head_idx = Nh_id / group_size;
  const int Q_gmem_offset =
      ((Nb_id * Nh * Nq * kHeadDim) + (Nh_id * Nq * kHeadDim));  // Q [seqlen,d]
  const int K_gmem_offset = ((Nb_id * Nh_kv * Nkv * kHeadDim) +
                             (kv_head_idx * Nkv * kHeadDim));  // K [seqlen,d]
  const int V_gmem_offset = K_gmem_offset;                     // V [seqlen,d]
  const int O_gmem_offset = Q_gmem_offset;                     // O [seqlen,d]

  // int load_gmem_Q_Br = Q_tile_id * Br + load_smem_Q_Br;
  // if ((Q_tile_id * Br + (threadIdx.x / (kNumThreads / Br))) >= Nq) return;
  if ((Q_tile_id * Br) >= Nq)
    return;

  extern __shared__ __align__(16) unsigned char ffpa_smem_raw[];
  kDataType* smem = reinterpret_cast<kDataType*>(ffpa_smem_raw);
  constexpr int Q_tile_size =
      Br * (kMmaAtomK + kPadQ);  // Q[Br,16], 64*16*2=2048 bytes
  constexpr int K_tile_size = Bc * (kMmaAtomK + kPadK);      // K[Bc,16]
  constexpr int V_tile_size = Bc * (kMmaAtomN * 2 + kPadV);  // V[Bc,16]
  kDataType* Q_tile_smem = smem;
  kDataType* K_tile_smem =
      (Q_tile_smem +
       (kPersistQg2s
            ? ((kHeadDim / kMmaAtomK) * Q_tile_size)
            : (kStageQK * Q_tile_size)));  // kPersistQg2s -> e.g d=64, Q smem
                                           // [4][Br][16] [tile_K_d][Br][16]
  // V may reuse all Q+K smem after Q@K^T.
  kDataType* V_tile_smem =
      (kShareSmemQKV ? Q_tile_smem : K_tile_smem + kStageQK * K_tile_size);
  const uint32_t smem_Q_base_ptr = __cvta_generic_to_shared(Q_tile_smem);
  const uint32_t smem_K_base_ptr = __cvta_generic_to_shared(K_tile_smem);
  const uint32_t smem_V_base_ptr = __cvta_generic_to_shared(V_tile_smem);

  // load Q g2s at very beginning if kPersistQg2s is enabled.
  // Put Q g2s before registers init, enable overlap between kPersistQg2s
  // and init states.
  if constexpr (kPersistQg2s) {
#pragma unroll
    for (int tile_K_d = 0; tile_K_d < (kHeadDim / kMmaAtomK); ++tile_K_d) {
      ffpa::prefill::cp_async_qkv_g2s<Br, Q_tile_size, kHeadDim, kMmaAtomK,
                                      kNumThreads, kPadQ>(
          smem_Q_base_ptr, Q, Q_gmem_offset, Q_tile_id, tile_K_d, tile_K_d, Nq);
      ffpa::cp_async::commit_group();
    }
  }

  // Registers/SMEM for thread block: m|l_old, store in lane.
  float lane_block_row_max_old[kValTileSeqLenQ][2];  // [1][2]
  float lane_block_row_sum_old[kValTileSeqLenQ][2];  // [1][2]
  ffpa::utils::fill_2D_regs<float, kValTileSeqLenQ, 2>(lane_block_row_max_old,
                                                       -INFINITY);
  ffpa::utils::fill_2D_regs<float, kValTileSeqLenQ, 2>(lane_block_row_sum_old,
                                                       0.0f);

  // Registers for S=Q@K^T/O=P@V, e.g, 64, !kPersistQs2r -> [1][4] 4 regs,
  // kPersistQs2r -> [1][4*4] 16 regs.
  uint32_t R_Q[kValTileSeqLenQ][(kPersistQs2r) ? (kHeadDim / kMmaAtomK) : 1][4];
  // R_K [8][2] w/o registers ping pong buffers, [2][2] w/ registers ping pong
  // buffers.
  uint32_t R_K[(kRegPipeKV) ? 2 : kValTileSeqLenK][2];  // [8][2] or [2][2]
  // R_V [2][2] w registers ping pong buffers, [1][2] w/o registers ping pong
  // buffers.
  uint32_t R_V[(kRegPipeKV) ? 2 : 1]
              [2];  // [1][2], S=Q@K, only use 2 32bits registers.
  // e.g [1][8][2], MMA Acc fp16; [1][8][4], MMA Acc fp32;
  uint32_t R_S[kValTileSeqLenQ][kValTileSeqLenK][(kMmaAccFloat32QK) ? 4 : 2];
  uint32_t R_O[(kMmaAccFloat32PV) ? 4 : 2];  // registers for O=PV[Br,d]=P@V, [4
                                             // or 2]
  uint32_t R_D[kValTileSeqLenP][kValTileHeadDimV]
              [(kOStorageAccFloat32) ? 4 : 2];
  ffpa::utils::fill_3D_regs<uint32_t, kValTileSeqLenP, kValTileHeadDimV,
                            ((kOStorageAccFloat32) ? 4 : 2)>(R_D, 0);

  // Additional load/store controllers for kRegPipeKV
  uint32_t reg_st_idx = 0;
  uint32_t reg_ld_idx = 1;

  // Causal attention bounds (large-d kernel). Queries are aligned to the
  // tail of the KV sequence: the global KV column visible to Q row r is
  // ``k <= r + kv_offset`` with ``kv_offset = Nkv - Nq``. The launcher
  // enforces ``Nkv >= Nq`` whenever ``causal != 0`` so ``kv_offset >= 0``.
  // * ``Tc_eff``         : KV tile loop bound clipped by the last tile that
  //                       still contains any visible key for this Q tile.
  // * ``mask_start_tile``: first tile where at least one column exceeds
  //                       the causal threshold of row 0 and therefore
  //                       needs the per-fragment causal mask.
  const int Br_base = Q_tile_id * Br;
  const int kv_offset = Nkv - Nq;
  const int causal_thresh_row0 = Br_base + kv_offset;  // max k for row 0
  const int Tc_eff =
      causal ? min(Tc, ((Br_base + Br - 1 + kv_offset) / Bc) + 1) : Tc;
  const int mask_start_tile =
      causal ? max(0, (causal_thresh_row0 + 1) / Bc) : INT_MAX;
  const float inv_scale = 1.0f / scale;

  // Now, N must be mutliples of Bc(32/64) for KV tiling across seqlen.
  // <loop over K seqlen>: for K^T[d,seqlen] with K^T_tile[d,Bc]
#pragma unroll 1
  for (int tile_K_seqlen = 0; tile_K_seqlen < Tc_eff; ++tile_K_seqlen) {
    // TODO: process last tile_K_seqlen ? pad to multiple of Bc.
    if constexpr (kPrefetchQK) {
      if constexpr (kStageQK > 1) {
        if (tile_K_seqlen == 0) {
#pragma unroll
          for (int stage = 0; stage < (kStageQK - 1); ++stage) {
            if constexpr (!kPersistQg2s) {
              ffpa::prefill::cp_async_qkv_g2s<Br, Q_tile_size, kHeadDim,
                                              kMmaAtomK, kNumThreads, kPadQ>(
                  smem_Q_base_ptr, Q, Q_gmem_offset, Q_tile_id, stage, stage,
                  Nq);
            }

            ffpa::prefill::cp_async_qkv_g2s<Bc, K_tile_size, kHeadDim,
                                            kMmaAtomK, kNumThreads, kPadK>(
                smem_K_base_ptr, K, K_gmem_offset, tile_K_seqlen, stage, stage,
                Nkv);
            ffpa::cp_async::commit_group();  // pack QK as 1 group.
          }  // end for stage
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
            // We only load Q g2s and s2r once if kPersistQs2r is enabled.
            if (tile_K_seqlen == 0) {
              ffpa::prefill::cp_async_qkv_g2s<Br, Q_tile_size, kHeadDim,
                                              kMmaAtomK, kNumThreads, kPadQ>(
                  smem_Q_base_ptr, Q, Q_gmem_offset, Q_tile_id, stage, stage,
                  Nq);
            }
          } else {
            if constexpr (!kPersistQg2s) {
              ffpa::prefill::cp_async_qkv_g2s<Br, Q_tile_size, kHeadDim,
                                              kMmaAtomK, kNumThreads, kPadQ>(
                  smem_Q_base_ptr, Q, Q_gmem_offset, Q_tile_id, stage, stage,
                  Nq);
            }
          }

          ffpa::prefill::cp_async_qkv_g2s<Bc, K_tile_size, kHeadDim, kMmaAtomK,
                                          kNumThreads, kPadK>(
              smem_K_base_ptr, K, K_gmem_offset, tile_K_seqlen, stage, stage,
              Nkv);
          ffpa::cp_async::commit_group();  // pack QK as 1 group.
        }  // end for stage
        ffpa::cp_async::wait_group<(kStageQK - 2)>();
        __syncthreads();
      }  // end if kStageQK > 1
    }

    // !kShareSmemQKV: Prefetch V g2s before all Q@K^T iteration.
    if constexpr ((!kShareSmemQKV) && kPrefetchPV && kStagePV > 1) {
      // Prefetch V g2s before all Q@K^T iteration.
      if constexpr (kStagePV > 1) {
#pragma unroll
        for (int stage = 0; stage < (kStagePV - 1); ++stage) {
          ffpa::prefill::cp_async_qkv_g2s<Bc, V_tile_size, kHeadDim,
                                          kMmaAtomN * 2, kNumThreads, kPadV>(
              smem_V_base_ptr, V, V_gmem_offset, tile_K_seqlen, stage, stage,
              Nkv);
          ffpa::cp_async::commit_group();
        }
      }
    }

    // <loop over K d>: tile_K_d, kMmaAtomK = 16, K_tile_d[kMmaAtomK,Bc]
    ffpa::utils::fill_3D_regs<uint32_t, kValTileSeqLenQ, kValTileSeqLenK,
                              (kMmaAccFloat32QK) ? 4 : 2>(R_S, 0);
#pragma unroll
    for (int tile_K_d = 0; tile_K_d < (kHeadDim / kMmaAtomK); ++tile_K_d) {
      const int smem_sel = (tile_K_d) % kStageQK;
      const int smem_sel_next = (tile_K_d + (kStageQK - 1)) % kStageQK;
      // QK g2s, kPersistQs2r or not.
      if constexpr (kPersistQs2r) {
        // We only load Q g2s and s2r once if kPersistQs2r is enabled.
        if (tile_K_seqlen == 0) {
          ffpa::prefill::cp_async_qkv_g2s<Br, Q_tile_size, kHeadDim, kMmaAtomK,
                                          kNumThreads, kPadQ>(
              smem_Q_base_ptr, Q, Q_gmem_offset, Q_tile_id,
              (kStageQK > 1) ? (tile_K_d + (kStageQK - 1)) : tile_K_d,
              (kStageQK > 1) ? smem_sel_next : smem_sel, Nq);
        }
      } else {
        if constexpr (!kPersistQg2s) {
          ffpa::prefill::cp_async_qkv_g2s<Br, Q_tile_size, kHeadDim, kMmaAtomK,
                                          kNumThreads, kPadQ>(
              smem_Q_base_ptr, Q, Q_gmem_offset, Q_tile_id,
              (kStageQK > 1) ? (tile_K_d + (kStageQK - 1)) : tile_K_d,
              (kStageQK > 1) ? smem_sel_next : smem_sel, Nq);
        }
      }

      ffpa::prefill::cp_async_qkv_g2s<Bc, K_tile_size, kHeadDim, kMmaAtomK,
                                      kNumThreads, kPadK>(
          smem_K_base_ptr, K, K_gmem_offset, tile_K_seqlen,
          (kStageQK > 1) ? (tile_K_d + (kStageQK - 1)) : tile_K_d,
          (kStageQK > 1) ? smem_sel_next : smem_sel, Nkv);
      ffpa::cp_async::commit_group();  // pack QK as 1 group.

      if constexpr (kStageQK <= 1) {
        ffpa::cp_async::wait_group<0>();
        __syncthreads();
      }

      // Q s2r
      static_assert(kValTileSeqLenQ == 1);
      {
        if constexpr (kPersistQs2r) {
          // We only load Q g2s and s2r once if kPersistQs2r is enabled.
          if (tile_K_seqlen == 0) {
            ffpa::prefill::sync_fetch_qkv_frags_s2r<
                0, 4, Q_tile_size, kMmaAtomM, kMmaAtomN, kMmaAtomK, kPadQ,
                kDataType>(smem_Q_base_ptr, &R_Q[0][tile_K_d][0], warp_QP, 0, 0,
                           smem_sel);
          }
        } else {
          if constexpr (!kPersistQg2s) {
            ffpa::prefill::sync_fetch_qkv_frags_s2r<
                0, 4, Q_tile_size, kMmaAtomM, kMmaAtomN, kMmaAtomK, kPadQ,
                kDataType>(smem_Q_base_ptr, &R_Q[0][0][0], warp_QP, 0, 0,
                           smem_sel);
          } else {
            ffpa::prefill::sync_fetch_qkv_frags_s2r<
                0, 4, Q_tile_size, kMmaAtomM, kMmaAtomN, kMmaAtomK, kPadQ,
                kDataType>(smem_Q_base_ptr, &R_Q[0][0][0], warp_QP, 0, 0,
                           tile_K_d);
          }
        }
      }

      // K s2r
      reg_st_idx = 0;
      reg_ld_idx = 1;
      if constexpr (!kRegPipeKV) {
#pragma unroll
        for (int j = 0; j < kValTileSeqLenK; ++j) {
          ffpa::prefill::sync_fetch_qkv_frags_s2r<0, 2, K_tile_size, kMmaAtomM,
                                                  kMmaAtomN, kMmaAtomK, kPadK,
                                                  kDataType>(
              smem_K_base_ptr, &R_K[j][0], warp_KV, j, 0, smem_sel);
        }
      } else {
        // kRegPipeKV is enabled, load first K tile frags from kValTileSeqLenK.
        ffpa::prefill::sync_fetch_qkv_frags_s2r<0, 2, K_tile_size, kMmaAtomM,
                                                kMmaAtomN, kMmaAtomK, kPadK,
                                                kDataType>(
            smem_K_base_ptr, &R_K[reg_st_idx][0], warp_KV, 0, 0, smem_sel);
      }

      // kShareSmemQKV: Prefetch V g2s before last Q@K^T iteration.
      if constexpr ((kShareSmemQKV) && kPrefetchPV && kStagePV > 1) {
        if (tile_K_d == (kHeadDim / kMmaAtomK - 1)) {
          __syncthreads();  // wait all QK s2r ready
          // Prefetch V g2s before last Q@K^T iteration.
          if constexpr (kStagePV > 1) {
#pragma unroll
            for (int stage = 0; stage < (kStagePV - 1); ++stage) {
              ffpa::prefill::cp_async_qkv_g2s<
                  Bc, V_tile_size, kHeadDim, kMmaAtomN * 2, kNumThreads, kPadV>(
                  smem_V_base_ptr, V, V_gmem_offset, tile_K_seqlen, stage,
                  stage, Nkv);
              ffpa::cp_async::commit_group();
            }
          }
        }
      }  // end kPrefetchQKV

      // Q@K^T MMA compute
      static_assert(kValTileSeqLenQ == 1);
      {  // kValTileSeqLenQ = 1
        const int q_offset = (kPersistQs2r) ? (tile_K_d) : 0;  // (tile_K_d)
#pragma unroll
        for (int j = 0; j < kValTileSeqLenK; ++j) {
          reg_st_idx ^= 1;  // 0->1
          reg_ld_idx ^= 1;  // 1->0
          if constexpr (kRegPipeKV) {
            // load next (j+1) K tile frags
            if ((j + 1) < kValTileSeqLenK) {
              ffpa::prefill::sync_fetch_qkv_frags_s2r<
                  0, 2, K_tile_size, kMmaAtomM, kMmaAtomN, kMmaAtomK, kPadK,
                  kDataType>(smem_K_base_ptr, &R_K[reg_st_idx][0], warp_KV,
                             (j + 1), 0, smem_sel);
            }
          }
          const int k_offset = (kRegPipeKV) ? reg_ld_idx : j;
          if constexpr (kMmaAccFloat32QK) {
            ffpa::mma::m16n8k16_abf32<kDataType,
                                      ffpa::mma::MMAMode::kInplaceUpdate>(
                &R_S[0][j][0], &R_S[0][j][1], &R_S[0][j][2], &R_S[0][j][3],
                &R_Q[0][q_offset][0], &R_Q[0][q_offset][1],
                &R_Q[0][q_offset][2], &R_Q[0][q_offset][3], &R_K[k_offset][0],
                &R_K[k_offset][1]);
          } else {
            ffpa::mma::m16n8k16_f16f16f16<ffpa::mma::MMAMode::kInplaceUpdate>(
                &R_S[0][j][0], &R_S[0][j][1], &R_Q[0][q_offset][0],
                &R_Q[0][q_offset][1], &R_Q[0][q_offset][2],
                &R_Q[0][q_offset][3], &R_K[k_offset][0], &R_K[k_offset][1]);
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
        // must wait all MMAs ready before next iteration
        // if kStageQK == 1 to avoid K smem overwrite.
        __syncthreads();
      }
    }  // end loop over d, S=Q@K^T
    __syncthreads();

    // Prefetch V g2s before row max/sum for P@V if kStagePV > 1
    static_assert(kValTileSeqLenP == 1);
    if constexpr (!kPrefetchPV) {
      if constexpr (kStagePV > 1) {
#pragma unroll
        for (int stage = 0; stage < (kStagePV - 1); ++stage) {
          ffpa::prefill::cp_async_qkv_g2s<Bc, V_tile_size, kHeadDim,
                                          kMmaAtomN * 2, kNumThreads, kPadV>(
              smem_V_base_ptr, V, V_gmem_offset, tile_K_seqlen, stage, stage,
              Nkv);
          ffpa::cp_async::commit_group();
        }
      }
    }

    // MMA = m16n8k16, Br=16x4=64, Bc=8x8=64, layout: 4 warps
    // |   64x64   |      warp_KV 0       |
    // | warp_QP 0 | MMA 0 ... MMA 0 (x8) |
    // | warp_QP 1 | MMA 1 ... MMA 1 (x8) |
    // | warp_QP 2 | MMA 2 ... MMA 2 (x8) |
    // | warp_QP 3 | MMA 3 ... MMA 3 (x8) |

    // Online safe softmax, warp/block reduce max/sum, row wise
    float lane_row_max_new[kValTileSeqLenQ][2];  // [1][2]
    float lane_row_sum_new[kValTileSeqLenQ][2];  // [1][2]
    ffpa::utils::fill_2D_regs<float, kValTileSeqLenQ, 2>(lane_row_max_new,
                                                         -INFINITY);
    ffpa::utils::fill_2D_regs<float, kValTileSeqLenQ, 2>(lane_row_sum_new,
                                                         0.0f);

    static_assert(kValTileSeqLenQ == 1);
    // reference:
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
    // #matrix-fragments-for-mma-m16n8k16-with-floating-point-type
    // The layout of the fragments held by different threads for C. (m16n8k16)
    // Row\Col  0    1    2    3    4    5    6    7
    // 0        T0: {c0, c1}  T1: {c0, c1}  T2: {c0, c1}  T3: {c0, c1}
    // 1        T4: {c0, c1}  T5: {c0, c1}  T6: {c0, c1}  T7: {c0, c1}
    // 2        ...
    // ...
    // 7        T28: {c0, c1}  T29: {c0, c1}  T30: {c0, c1}  T31: {c0, c1}
    // 8        T0: {c2, c3}   T1: {c2, c3}   T2: {c2, c3}   T3: {c2, c3}
    // 9        T4: {c2, c3}   T5: {c2, c3}   T6: {c2, c3}   T7: {c2, c3}
    // 10       ...
    // ...
    // 15       T28: {c2, c3}  T29: {c2, c3}  T30: {c2, c3}  T31: {c2, c3}
    // Seqlen boundary: on the last KV tile with Nq % Bc != 0, mask
    // padding columns to -inf so online softmax ignores them.
    {
      const int kv_valid_local = Nkv - tile_K_seqlen * Bc;
      if (kv_valid_local < Bc) {
        ffpa::prefill::sync_apply_kv_mask<kValTileSeqLenK, kMmaAccFloat32QK,
                                          kDataType>(&R_S[0][0][0],
                                                     kv_valid_local);
      }
    }
    // Causal mask: skipped on tiles fully below the diagonal thanks to
    // ``mask_start_tile`` (INT_MAX when causal == 0), so non-causal paths
    // pay only one compare-and-branch per tile.
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

    // Wait V g2s stages ready.
    if constexpr (kStagePV > 1) {
      ffpa::cp_async::wait_group<(kStagePV - 2)>();  // s2->0, s3->1, s4->2
      __syncthreads();
    }

    // !kShareSmemQKV: Prefetch QK g2s before all P@V.
    if constexpr ((!kShareSmemQKV) && kPrefetchQK && kStageQK > 1) {
      if ((tile_K_seqlen + 1) < Tc_eff) {
#pragma unroll
        for (int stage = 0; stage < (kStageQK - 1); ++stage) {
          // We do not need to load Q g2s & s2r while tile_K_seqlen > 0,
          // if kPersistQs2r is enabled.
          if constexpr (!kPersistQs2r) {
            ffpa::prefill::cp_async_qkv_g2s<Br, Q_tile_size, kHeadDim,
                                            kMmaAtomK, kNumThreads, kPadQ>(
                smem_Q_base_ptr, Q, Q_gmem_offset, Q_tile_id, stage, stage, Nq);
          }

          ffpa::prefill::cp_async_qkv_g2s<Bc, K_tile_size, kHeadDim, kMmaAtomK,
                                          kNumThreads, kPadK>(
              smem_K_base_ptr, K, K_gmem_offset, tile_K_seqlen + 1, stage,
              stage, Nkv);
          ffpa::cp_async::commit_group();  // pack QK as 1 group.
        }  // end for stage
      }
    }

    static_assert(kValTileSeqLenP == 1);
    {
      float rescale_o_factor_0[1];
      float rescale_o_factor_1[1];
      ffpa::prefill::sync_precompute_rescale_factors(
          &rescale_o_factor_0[0], &rescale_o_factor_1[0],
          &lane_row_max_new[0][0], &lane_block_row_max_old[0][0],
          tile_K_seqlen);

      // <HGEMM in registers>
#pragma unroll
      for (int j = 0; j < kValTileHeadDimV; ++j) {  // 8, 16, 32, ...
        // Compute d tile, P[Br,Bc]@V[Bc,16] = O[Br,16]
        const int tile_V_d = (j >> 1);  // (j / 2)
        const int smem_sel_v = (tile_V_d) % kStagePV;
        const int smem_sel_v_next = (tile_V_d + (kStagePV - 1)) % kStagePV;
        // V g2s, V tile smem [Bc,kMmaAtomN*2]=[64,16]
        if (j % 2 == 0) {  // 0,2,4,6,...// curr K tile g2s
          ffpa::prefill::cp_async_qkv_g2s<Bc, V_tile_size, kHeadDim,
                                          kMmaAtomN * 2, kNumThreads, kPadV>(
              smem_V_base_ptr, V, V_gmem_offset, tile_K_seqlen,
              (kStagePV > 1) ? (tile_V_d + (kStagePV - 1)) : tile_V_d,
              (kStagePV > 1) ? smem_sel_v_next : smem_sel_v, Nkv);
          ffpa::cp_async::commit_group();
          if constexpr (kStagePV <= 1) {
            ffpa::cp_async::wait_group<0>();
            __syncthreads();
          }
        }

        // reinit controllers
        reg_st_idx = 0;
        reg_ld_idx = 1;
        // kRegPipeKV V s2r
        if constexpr (kRegPipeKV) {
          // load first tile_V_Bc V tile frags from (Bc / kMmaAtomK).
          ffpa::prefill::sync_fetch_qkv_frags_s2r<1, 2, V_tile_size, kMmaAtomM,
                                                  kMmaAtomN, kMmaAtomK, kPadV,
                                                  kDataType>(
              smem_V_base_ptr, &R_V[reg_st_idx][0], warp_KV, (j % 2), 0,
              smem_sel_v);
        }

        ffpa::utils::fill_1D_regs<uint32_t, (kMmaAccFloat32PV) ? 4 : 2>(R_O, 0);
#pragma unroll
        for (int tile_V_Bc = 0; tile_V_Bc < (Bc / kMmaAtomK); ++tile_V_Bc) {
          // kShareSmemQKV: Prefetch next QK g2s before last P@V iteration.
          if constexpr ((kShareSmemQKV) && kPrefetchQK && kStageQK > 1) {
            if (j == (kValTileHeadDimV - 1) &&
                tile_V_Bc == (Bc / kMmaAtomK - 1) &&
                (tile_K_seqlen + 1) < Tc_eff) {
              __syncthreads();  // wait all V s2r ready
              // Prefetch next QK g2s before last P@V iteration.
              if constexpr (kStageQK > 1) {
#pragma unroll
                for (int stage = 0; stage < (kStageQK - 1); ++stage) {
                  // We do not need to load Q g2s & s2r while tile_K_seqlen > 0,
                  // if kPersistQs2r is enabled.
                  if constexpr (!kPersistQs2r) {
                    ffpa::prefill::cp_async_qkv_g2s<Br, Q_tile_size, kHeadDim,
                                                    kMmaAtomK, kNumThreads,
                                                    kPadQ>(
                        smem_Q_base_ptr, Q, Q_gmem_offset, Q_tile_id, stage,
                        stage, Nq);
                  }

                  ffpa::prefill::cp_async_qkv_g2s<
                      Bc, K_tile_size, kHeadDim, kMmaAtomK, kNumThreads, kPadK>(
                      smem_K_base_ptr, K, K_gmem_offset, tile_K_seqlen + 1,
                      stage, stage, Nkv);
                  ffpa::cp_async::commit_group();  // pack QK as 1 group.
                }  // end for stage
              }
            }
          }  // end if kPrefetchQKV && kStageQK > 1

          // V s2r
          reg_st_idx ^= 1;  // 0->1
          reg_ld_idx ^= 1;  // 1->0
          if constexpr (!kRegPipeKV) {
            ffpa::prefill::sync_fetch_qkv_frags_s2r<
                1, 2, V_tile_size, kMmaAtomM, kMmaAtomN, kMmaAtomK, kPadV,
                kDataType>(smem_V_base_ptr, &R_V[0][0], warp_KV, (j % 2),
                           tile_V_Bc, smem_sel_v);
          } else {
            // load next (tile_V_Bc + 1) V tile frags
            if ((tile_V_Bc + 1) < (Bc / kMmaAtomK)) {
              ffpa::prefill::sync_fetch_qkv_frags_s2r<
                  1, 2, V_tile_size, kMmaAtomM, kMmaAtomN, kMmaAtomK, kPadV,
                  kDataType>(smem_V_base_ptr, &R_V[reg_st_idx][0], warp_KV,
                             (j % 2), (tile_V_Bc + 1), smem_sel_v);
            }
          }

          // Compute P[Br,Bc]@V[Bc,d] = O[Br,d]
          // For R_S[1][8][2], mapping the layout below of P matrix.
          // MMA = m16n8k16, Br=16x4=64, Bc=8x8=64, layout: 4 warps
          // |   64x64   |      warp_KV 0       |
          // | warp_QP 0 | MMA 0 ... MMA 0 (x8) |
          // | warp_QP 1 | MMA 1 ... MMA 1 (x8) |
          // | warp_QP 2 | MMA 2 ... MMA 2 (x8) |
          // | warp_QP 3 | MMA 3 ... MMA 3 (x8) |
          // tile_V_Bc = 0, all curr MMAs(0~4) need slice P[:,  0:16], 0, 1;
          // stored in all MMAs. tile_V_Bc = 1, all curr MMAs(0~4) need slice
          // P[:, 16:32], 2, 3; stored in all MMAs. tile_V_Bc = 2, all curr
          // MMAs(0~4) need slice P[:, 32:48], 4, 5; stored in all MMAs.
          // tile_V_Bc = 3, all curr MMAs(0~4) need slice P[:, 48:64], 6, 7;
          // stored in all MMAs.
          const int p_offset = tile_V_Bc * 2;  // MMA(Warp) selected, 0, 2, 4, 6
          const int v_offset = (kRegPipeKV) ? reg_ld_idx : 0;
          if constexpr (kMmaAccFloat32PV) {
            // MMA accumulate with F32 dtype for high precision.
            ffpa::mma::m16n8k16_abf32<kDataType,
                                      ffpa::mma::MMAMode::kInplaceUpdate>(
                &R_O[0], &R_O[1], &R_O[2], &R_O[3], &R_S[0][p_offset][0],
                &R_S[0][p_offset][1], &R_S[0][p_offset + 1][0],
                &R_S[0][p_offset + 1][1], &R_V[v_offset][0], &R_V[v_offset][1]);
          } else {
            // MMA accumulate with F16 dtype for high throughput.
            ffpa::mma::m16n8k16_f16f16f16<ffpa::mma::MMAMode::kInplaceUpdate>(
                &R_O[0], &R_O[1], &R_S[0][p_offset][0], &R_S[0][p_offset][1],
                &R_S[0][p_offset + 1][0], &R_S[0][p_offset + 1][1],
                &R_V[v_offset][0], &R_V[v_offset][1]);
          }
        }  // end for V Bc.
        if constexpr (kStagePV < 2) {
          // Wait curr P@V tile ready if kStage < 2 in order to avoid
          // the next V tile g2s overwrite.
          __syncthreads();
        }
        // according to the A matrix layout for MMA m16n8k16 instruction.
        // reference:
        // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
        // #matrix-fragments-for-mma-m16n8k16-with-floating-point-type
        // The layout of the fragments held by different threads for A matrix
        // with .f16. R\C  0    1    2    3    4    5    6    7    8    9   10
        // 11   12   13   14   15 0    T0: {a0, a1}  T1: {a0, a1}  T2: {a0, a1}
        // T3: {a0, a1}  T0: {a4, a5}  T1: {a4, a5} T2: {a4, a5}  T3: {a4, a5} 1
        // T4: {a0, a1}  T5: {a0, a1}  T6: {a0, a1}  T7: {a0, a1} T4: {a4, a5}
        // T5: {a4, a5}  T6: {a4, a5}  T7: {a4, a5} 2    (dashed arrow pointing
        // right)
        // ...
        // 7    T28: {a0, a1}  T29: {a0, a1}  T30: {a0, a1}  T31: {a0, a1}  T28:
        // {a4, a5}  T29: {a4, a5}  T30: {a4, a5}  T31: {a4, a5} 8    T0: {a2,
        // a3}   T1: {a2, a3}   T2: {a2, a3}   T3: {a2, a3}   T0: {a6, a7}   T1:
        // {a6, a7}   T2: {a6, a7}   T3: {a6, a7} 9    T4: {a2, a3} T5: {a2, a3}
        // T6: {a2, a3}   T7: {a2, a3}   T4: {a6, a7}   T5: {a6, a7}   T6: {a6,
        // a7} T7: {a6, a7} 10   (dashed arrow pointing right)
        // ...
        // 15   T28: {a2, a3}  T29: {a2, a3}  T30: {a2, a3}  T31: {a2, a3}  T28:
        // {a6, a7}  T29: {a6, a7}  T30: {a6, a7}  T31: {a6, a7}

        // Now, we get [Br,8] slice of [Br,d], each warp(MMA) contains m16n8.
        // 0. Rescale O: Online rescaling O each tile_K_seqlen step, need m_new,
        // m_old. m = max(m_old, m_new), O_new[Br,d] = exp(m_old - m) * O_old +
        // P@V
        ffpa::prefill::sync_rescaling_tiling_o<kOStorageAccFloat32,
                                               kMmaAccFloat32PV, kDataType>(
            &R_D[0][0][0], &R_O[0], &rescale_o_factor_0[0],
            &rescale_o_factor_1[0], tile_K_seqlen, j);

        if constexpr (kStagePV > 1) {
          // Wait next V tile g2s ready.
          if (j < (kValTileHeadDimV - 1)) {
            ffpa::cp_async::wait_group<(kStagePV - 2)>();
            __syncthreads();
          }
        }
      }  // end for kValTileHeadDimV (end P@V)

      // Now, we can update m, l after O has been scaled.
      ffpa::prefill::sync_update_max_expsum(
          &lane_row_max_new[0][0], &lane_row_sum_new[0][0],
          &lane_block_row_max_old[0][0], &lane_block_row_sum_old[0][0],
          &rescale_o_factor_0[0], &rescale_o_factor_1[0]);
    }  // end P@V
    __syncthreads();

  }  // end loop over N
  __syncthreads();

  // Finaly, we still have to rescale O once more.
  // O_output(D) = ( 1/l_final ) * O_final (FA2 paper)
  static_assert(kValTileSeqLenP == 1);
  ffpa::prefill::sync_rescaling_final_o<kValTileHeadDimV, kOStorageAccFloat32,
                                        kDataType>(
      &R_D[0][0][0], &lane_block_row_sum_old[0][0]);

  // Store O(D): Write O[Br,d] from regs -> gmem, collective store
  // with reg reuse & warp shuffle.
  static_assert(kValTileSeqLenP == 1);
  ffpa::prefill::sync_store_o_r2g<Br, kHeadDim, kMmaAtomM, kMmaAtomN,
                                  kValTileHeadDimV, kOStorageAccFloat32,
                                  kDataType>(O, O_gmem_offset, O_tile_id,
                                             warp_QP, &R_D[0][0][0],
                                             &R_Q[0][0][0], &R_K[0][0], Nq);

  // Store softmax LSE: LSE = log(row_sum) + row_max, written per-row to the
  // [B, Nh, Nq] float32 buffer.
  const int softmax_lse_offset = Nb_id * Nh * Nq + Nh_id * Nq;
  ffpa::prefill::sync_store_lse_r2g<Br, kMmaAtomM, kValTileSeqLenQ>(
      softmax_lse, softmax_lse_offset, O_tile_id, warp_QP,
      &lane_block_row_max_old[0][0], &lane_block_row_sum_old[0][0], Nq);
}

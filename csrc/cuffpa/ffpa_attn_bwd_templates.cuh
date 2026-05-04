#pragma once

// ============================================================================
// FFPA Split-D Backward Kernel — Algorithm Design Document
// ============================================================================
//
// The goal is a native Split-D backward kernel for large head dimensions
// (D > 256) that replaces the current SDPA backward delegation path.
// This file documents the intended algorithm and serves as an implementation
// guide.  The kernel body is currently stubbed out (static_assert(false)).
//
// ----------------------------------------------------------------------------
// 1. FA-2 Backward Dataflow
// ----------------------------------------------------------------------------
//
// Grid: KV-driven — (div_ceil(Nkv, Bc), Nb * Nh_kv).  Each block owns one
// KV tile [Bc, D] and iterates over Q tiles in REVERSE order (FA-2 schedule).
//
// Per block:
//   Load K[Bc, D], V[Bc, D] into SMEM (via cp.async, Split-D tiling).
//   Initialize dK_accum, dV_accum registers or SMEM.
//   For each Q tile (m_block from Tr-1 down to m_block_min):
//     For each Q head in GQA group:
//       Phase 1 (inner D/16 loop, SAME as forward):
//         For tile_K_d in 0..D/16-1:
//           Load Q[Br,16], dO[Br,16], O[Br,16], K[Bc,16] via cp.async.
//           MMA: R_S += Q[Br,16] @ K^T[16,Bc]     (same as forward Q@K^T)
//           MMA: R_dP += dO[Br,16] @ V^T[16,Bc]   (new, for dP=grad_out@V^T)
//           Accumulate dP_sum = row_sum(dO * O) per tile_K_d.
//       Phase 1b:
//         P = exp(S*scale - LSE)  — softmax reconstruction from forward LSE.
//         dS = P * (dP - rowsum(dO*O))  — softmax backward gradient.
//       Phase 2 (backward gradient accumulation):
//         dQ += dS @ K   (MMA or element-wise)
//         dK += dS^T @ Q (MMA or element-wise)
//         dV += P^T @ dO (MMA or element-wise)
//     Write dQ to global (atomicAdd across KV tiles).
//   Write dK, dV to global.
//
// ----------------------------------------------------------------------------
// 2. Key Challenges
// ----------------------------------------------------------------------------
//
// 2a. Fragment Layout Mismatch
//     Phase 1 outputs dS and P as C-fragments (4 fp32 values/thread from
//     m16n8k16 MMA output).  Phase 2 needs A/B-fragments (8 fp16 values/
//     thread from ldmatrix.x4 / 2 fp16 from ldmatrix.x2) for efficient MMA.
//     The C→A layout conversion requires register-level transposition.
//
//     Mitigation options:
//       a) Store dS/P to SMEM row-major, reload via ldmatrix.x4 — clean but
//          adds SMEM pressure (~8 KB for Br=64, Bc=64).
//       b) Compute dQ/dK/dV element-wise using the known C-fragment layout
//          (documented in the forward template comments) — correct but slow.
//       c) Implement a register-level C→A transpose helper.
//
// 2b. SMEM Budget
//     For D=320, Br=64, Bc=64, stage=1:
//       Q pipeline (kStageQK * Br * 16)  = 2 KB
//       dO pipeline (same)                = 2 KB
//       O pipeline (same)                 = 2 KB
//       K pipeline (kStageQK * Bc * 16)  = 2 KB
//       V pipeline (kStagePV * Bc * 16)  = 2 KB
//       Total pipeline                    = 10 KB
//
//     If using SMEM accumulators for dK/dV:
//       dK_accum [Bc, D] fp16           = 40 KB
//       dV_accum [Bc, D] fp16           = 40 KB
//       Total with accumulators          = 90 KB (fits L20 100 KB)
//
//     L20 has 100 KB configurable SMEM.  The current launcher uses only
//     10 KB (pipeline only, writing dK/dV directly to global via atomicAdd).
//     This avoids SMEM pressure at the cost of global memory bandwidth.
//
// 2c. dK/dV Atomic Contention
//     When multiple Q tiles contribute to the same KV tile, dK/dV accumulate
//     via atomicAdd.  SMEM accumulators avoid this but increase SMEM usage.
//     With stage=1, SMEM accumulators fit; with stage>=2 (multi-stage
//     pipeline), extra SMEM is needed for ring buffers.
//
// 2d. Future Optimization: Small-Bc Persistent-KV Backward
//     Split-D is primarily needed because large D makes full K/V residency too
//     expensive in SMEM.  For moderate D, a separate FA-style backward path can
//     reduce global-memory traffic by shrinking Bc and keeping K[Bc,D] and
//     V[Bc,D] resident in SMEM for the whole KV-owning block.  This should be a
//     distinct small-Bc persistent-KV kernel, not a replacement for the current
//     O(1)-in-D Split-D fallback.
//
//     With dtype fp16/bf16, persistent K+V costs:
//       2 * Bc * D * sizeof(dtype)
//
//     For Bc=32:
//       D=512  -> 64 KB for K+V only
//       D=768  -> 96 KB for K+V only
//       D=1024 -> 128 KB for K+V only
//
//     The kernel still needs Q/dO streaming buffers, P/dS transpose scratch,
//     padding, and synchronization space.  On a 100 KB-class SMEM budget,
//     Bc=32 is therefore realistic for D<=512, marginal or too large for D=768,
//     and not viable for D=1024.  D=1024 may require Bc=16, but that increases
//     the number of KV tiles and can worsen atomic/grid overhead.
//
//     Expected dispatch policy:
//       if persistent-KV smem fits, use small-Bc persistent-KV backward;
//       otherwise use this Split-D backward with D-slice staging.
//
//     Minimal first implementation target:
//       Br=64, Bc=32, D=512.  Load K[Bc,D] and V[Bc,D] once into SMEM at block
//       entry, stream Q/dO/O by D-slice, compute S/dP in Phase 1, then reuse the
//       resident K in Phase 2 for dQ.  Q/dO still need to be streamed for dK/dV,
//       and dQ plus dK/dV may still use atomicAdd initially.  The main expected
//       win is removing repeated global K/V loads across Q tiles and avoiding
//       the Phase-2 K reload when SMEM budget allows it.
//
// ----------------------------------------------------------------------------
// 3. Split-D Backward Algorithm (FA-2 Schedule + Head-Dim Tiling)
// ----------------------------------------------------------------------------
//
// The Split-D backward fuses the FA-2 KV-driven reverse schedule with
// FFPA's head-dimension tiling strategy.  The head dim D is decomposed
// into kMmaAtomK = 16 column sub-tiles to keep SMEM O(1) in D, exactly
// mirroring the forward kernel's Split-D design.
//
// Grid:  (div_ceil(Nkv, Bc), Nb * Nh_kv)  — one block per KV tile
// Block: (WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK, 1, 1)
//
// Register arrays (same as forward, plus backward-specific additions):
//   R_S  [1][kValTileSeqLenK][4]  — S=Q@K^T accumulator, reused as P
//   R_dP [1][kValTileSeqLenK][4]  — dP=dO@V^T accumulator, reused as dS
//   R_dQ [1][kValTileHeadDimV][4] — dQ running accumulator (all D slices)
//   R_Q  [1][4], R_K [*][2], R_V [*][2] — MMA operand fragments
//   R_dO_frag [4], R_O_frag [4] — dO/O fragments for dP_sum
//
// SMEM layout (mirrors forward):
//   Q_smem [kStageQK][Br][16] + dO_smem [kStageQK][Br][16]
//     + O_smem [kStageQK][Br][16]      — Q/dO/O ring buffer
//   K_smem [kStageQK][Bc][16]          — K ring buffer
//   V_smem [kStagePV][Bc][16]          — V ring buffer
//   [optional] dK_accum [Bc][D]        — dK SMEM accumulator
//   [optional] dV_accum [Bc][D]        — dV SMEM accumulator
//
// ----------------------------------------------------------------------------
// Per-block algorithm:
// ----------------------------------------------------------------------------
//
//   // ---- Initialization ----
//   Zero dK_accum and dV_accum (if using SMEM).
//   Zero R_dQ registers (once, reused across Q tiles).
//
//   // ---- Q-tile reverse iteration (FA-2 schedule) ----
//   for m_block = Tr-1 down to m_block_min:
//     for each Q head in GQA group:
//
//  ╔═══════════════════════════════════════════════════════════════╗
//  ║  PHASE 1 — Inner D/16 loop (Split-D: tile over head dim)    ║
//  ║  Accumulate S and dP across D/16 sub-tiles in registers.    ║
//  ╚═══════════════════════════════════════════════════════════════╝
//
//   for tile_K_d = 0 .. D/16 - 1:
//     1. cp.async G2S (global→shared):
//          K[Bc,16] → K_smem[stage][Bc,16]
//          Q[Br,16] → Q_smem[stage][Br,16]
//          dO[Br,16] → dO_smem[stage][Br,16]
//          O[Br,16]  → O_smem[stage][Br,16]
//        commit_group(); wait_group();
//        __syncthreads();
//
//     2. ldmatrix S2R (shared→register):
//          Q → R_Q     (ldmatrix.x4, 8 fp16/thread)
//          K → R_K     (ldmatrix.x2, 2 fp16/thread)
//          dO → R_dO_frag  (ldmatrix.x4)
//          O  → R_O_frag   (ldmatrix.x4)
//
//     3. MMA (Split-D accumulate, D dimension accumulates across tiles):
//          for kc in 0..kValTileSeqLenK-1:
//            R_S[0][kc]  += Q[Br,16]  @ K^T[16, Bc_slice]
//            R_dP[0][kc] += dO[Br,16] @ V^T[16, Bc_slice]
//
//     4. dP_sum accumulation (per tile_K_d, row-wise dot product):
//          dP_sum[0] += Σ_i dO[row0, i] * O[row0, i]  (row group 0)
//          dP_sum[1] += Σ_i dO[row1, i] * O[row1, i]  (row group 1)
//        (leveraging the A-fragment ldmatrix.x4 layout:
//         values 0-3 → row g*2, values 4-7 → row g*2+8)
//   end tile_K_d  // R_S and R_dP now hold FULL S[Br,Bc] and dP[Br,Bc]
//
//  ╔═══════════════════════════════════════════════════════════════╗
//  ║  PHASE 1b — Softmax reconstruction from forward LSE         ║
//  ╚═══════════════════════════════════════════════════════════════╝
//
//   Load LSE[16] for this warp's Q rows (1 float/row from global).
//   Warp-reduce dP_sum across 4-thread subgroups.
//   for each (row, col) in R_S C-fragment:
//     P[row][col] = exp(S[row][col] * scale - LSE[row])
//   Overwrite R_S with P (activation dtype).
//
//  ╔═══════════════════════════════════════════════════════════════╗
//  ║  PHASE 1c — Softmax backward gradient dS                    ║
//  ╚═══════════════════════════════════════════════════════════════╝
//
//   for each (row, col) in R_dP C-fragment:
//     dS[row][col] = P[row][col] * (dP[row][col] - dP_sum[row])
//   Overwrite R_dP with dS in-place.
//
//  ╔═══════════════════════════════════════════════════════════════╗
//  ║  PHASE 2 — Outer D/8 loop (Split-D: slice over head dim)    ║
//  ║  Accumulate dQ/dK/dV per D/8 column slice.                  ║
//  ╚═══════════════════════════════════════════════════════════════╝
//
//   for j = 0 .. kValTileHeadDimV-1:      // j = D/8 iterations
//     // Each j processes columns [j*8, j*8+8) of dQ/dK/dV.
//     // Reload Q[Br,16] and K[Bc,16] and dO[Br,16] from SMEM
//     // for this D slice (same SMEM ring-buffer, stage = j % kStageQK).
//
//     // --- dQ[j] += dS @ K ---
//     // dS from R_dP (C-fragment, full Br×Bc), K from ldmatrix.x2
//     for kc in 0..kValTileSeqLenK-1:
//       R_dQ[j] += dS[Br, Bc_slice] @ K[Bc_slice, 8]
//     // R_dQ[j] now holds partial dQ for D columns [j*8, j*8+8)
//
//     // --- dK += dS^T @ Q ---
//     // dS^T from R_dP (C-fragment, transposed), Q from ldmatrix.x4
//     for kc in 0..kValTileSeqLenK-1:
//       R_temp += dS^T[Bc_slice, Br] @ Q[Br, 8]
//     // atomicAdd R_temp → dK_accum[Bc, j*8:j*8+8]
//
//     // --- dV += P^T @ dO ---
//     // P from R_S (C-fragment), dO from ldmatrix.x4
//     for kc in 0..kValTileSeqLenK-1:
//       R_temp += P^T[Bc_slice, Br] @ dO[Br, 8]
//     // atomicAdd R_temp → dV_accum[Bc, j*8:j*8+8]
//   end j
//
//   // --- dQ writeback ---
//   R_dQ[0..kValTileHeadDimV-1] atomicAdd → global dQ
//   (across KV tiles, each tile contributes partial dQ to same Q rows)
//
//   // After all Q tiles:
//   dK_accum[Bc, D] → global dK  (one write per KV tile, no atomic needed)
//   dV_accum[Bc, D] → global dV  (one write per KV tile, no atomic needed)
//
// ----------------------------------------------------------------------------
// Split-D register budget (D=512, Br=64, Bc=64, 4 warps):
//   R_S:   1*8*4 = 32 regs (fp32 S, overwritten as fp16 P)
//   R_dP:  1*8*4 = 32 regs (fp32 dP, overwritten as dS)
//   R_dQ:  1*64*4 = 256 regs → TOO LARGE for 255-reg limit
//   → Mitigation: use fp16 for R_dQ (1*64*2 = 128 regs) or reduce
//     kMmaTileSeqLenQ to 4 (Br=64 gives 128 threads, 2 warps, more regs/thread)
//
// Gradient formulas (same as standard FA-2):
//   dV = P^T @ dO                        (no scale factor)
//   dP = dO @ V^T                        (no scale factor)
//   dS = P ⊙ (dP - dP_sum)               (dP_sum ≡ rowsum(dO * O))
//   dQ = scale * dS @ K                  (chain rule factor)
//   dK = scale * dS^T @ Q                (chain rule factor)
//
// Causal masking: applies to Phase 1b softmax.  Same logic as forward.
//   m_block_min = max(0, (KV_tile_id*Bc - kv_offset) / Br)
//   kv_offset = Nkv - Nq (queries aligned to KV tail)
//   S rows beyond Nq are masked to -∞ via LSE sentinel.
//
// GQA support: gridDim.y uses Nh_kv.  Each block maps kv_head_idx
//   to Q head range [kv_head_idx*group_size, (kv_head_idx+1)*group_size).
// ============================================================================

#include "prefill.cuh"

namespace ffpa {
namespace bwd {

__device__ __forceinline__ int c_frag_local_row(const int lane_id, const int reg_id) {
  return (lane_id / 4) + ((reg_id >= 2) ? 8 : 0);
}

__device__ __forceinline__ int c_frag_local_col(const int lane_id, const int n_tile_id,
                                                const int reg_id) {
  return n_tile_id * 8 + (lane_id % 4) * 2 + (reg_id & 1);
}

template <typename kDataType, const int kNumNTiles>
__device__ __forceinline__ void compute_p_and_ds_from_lse(
    const uint32_t (&R_S)[kNumNTiles][4], const uint32_t (&R_dP)[kNumNTiles][4],
    uint32_t (&R_P)[kNumNTiles][2], uint32_t (&R_dS)[kNumNTiles][2], const float lse_row0,
    const float lse_row8, const float dp_sum_row0, const float dp_sum_row8, const float scale) {
  using Traits = DtypeTraits<kDataType>;
#pragma unroll
  for (int n_tile = 0; n_tile < kNumNTiles; ++n_tile) {
    const float* scores = reinterpret_cast<const float*>(&R_S[n_tile][0]);
    const float* dP = reinterpret_cast<const float*>(&R_dP[n_tile][0]);
    kDataType* p_frag = reinterpret_cast<kDataType*>(&R_P[n_tile][0]);
    kDataType* ds_frag = reinterpret_cast<kDataType*>(&R_dS[n_tile][0]);
    const float p0 = __expf(__fmaf_rn(scores[0], scale, -lse_row0));
    const float p1 = __expf(__fmaf_rn(scores[1], scale, -lse_row0));
    const float p2 = __expf(__fmaf_rn(scores[2], scale, -lse_row8));
    const float p3 = __expf(__fmaf_rn(scores[3], scale, -lse_row8));
    p_frag[0] = Traits::from_float(p0);
    p_frag[1] = Traits::from_float(p1);
    p_frag[2] = Traits::from_float(p2);
    p_frag[3] = Traits::from_float(p3);
    ds_frag[0] = Traits::from_float(p0 * (dP[0] - dp_sum_row0));
    ds_frag[1] = Traits::from_float(p1 * (dP[1] - dp_sum_row0));
    ds_frag[2] = Traits::from_float(p2 * (dP[2] - dp_sum_row8));
    ds_frag[3] = Traits::from_float(p3 * (dP[3] - dp_sum_row8));
  }
}

template <typename kDataType, const int kHeadDim>
__device__ __forceinline__ void atomic_add_c_frag_to_dq(
    const uint32_t (&R_C)[4], kDataType* __restrict__ dQ, const int q_gmem_offset,
    const int Q_tile_id, const int warp_QP, const int d_col_base, const int Nq, const float scale) {
  using Traits = DtypeTraits<kDataType>;
  const int lane_id = threadIdx.x % WARP_SIZE;
  const float* c_frag = reinterpret_cast<const float*>(&R_C[0]);
#pragma unroll
  for (int reg_id = 0; reg_id < 4; ++reg_id) {
    const int row = Q_tile_id * 64 + warp_QP * 16 + c_frag_local_row(lane_id, reg_id);
    const int col = d_col_base + (lane_id % 4) * 2 + (reg_id & 1);
    if (row < Nq && col < kHeadDim) {
      atomicAdd(&dQ[q_gmem_offset + row * kHeadDim + col],
                Traits::from_float(c_frag[reg_id] * scale));
    }
  }
}

template <typename kDataType, const int kNumNTiles, const int kSmemStride>
__device__ __forceinline__ void store_packed_c_frag_transposed_to_smem(
    const uint32_t (&R_C)[kNumNTiles][2], kDataType* __restrict__ smem_tile) {
  static_assert(sizeof(kDataType) == 2, "packed C-fragment store expects 16-bit activation dtype.");
  const int lane_id = threadIdx.x % WARP_SIZE;
  const int row_group = lane_id / 4;
  const int lane_col = lane_id % 4;
  const bool store_row_pair = ((row_group & 1) == 0);
#pragma unroll
  for (int n_tile = 0; n_tile < kNumNTiles; ++n_tile) {
    const uint32_t frag01 = R_C[n_tile][0];
    const uint32_t frag23 = R_C[n_tile][1];
    const uint32_t next_frag01 = __shfl_down_sync(0xffffffff, frag01, 4);
    const uint32_t next_frag23 = __shfl_down_sync(0xffffffff, frag23, 4);
    if (store_row_pair) {
      const int row0 = row_group;
      const int row8 = row_group + 8;
      const int col0 = n_tile * 8 + lane_col * 2;
      const int col1 = col0 + 1;
      const uint32_t packed0 = (frag01 & 0x0000ffffu) | ((next_frag01 & 0x0000ffffu) << 16);
      const uint32_t packed1 = ((frag01 & 0xffff0000u) >> 16) | (next_frag01 & 0xffff0000u);
      const uint32_t packed2 = (frag23 & 0x0000ffffu) | ((next_frag23 & 0x0000ffffu) << 16);
      const uint32_t packed3 = ((frag23 & 0xffff0000u) >> 16) | (next_frag23 & 0xffff0000u);
      *reinterpret_cast<uint32_t*>(&smem_tile[col0 * kSmemStride + row0]) = packed0;
      *reinterpret_cast<uint32_t*>(&smem_tile[col1 * kSmemStride + row0]) = packed1;
      *reinterpret_cast<uint32_t*>(&smem_tile[col0 * kSmemStride + row8]) = packed2;
      *reinterpret_cast<uint32_t*>(&smem_tile[col1 * kSmemStride + row8]) = packed3;
    }
  }
}

template <typename kDataType, const int kHeadDim>
__device__ __forceinline__ void atomic_add_c_frag_to_dkv(const uint32_t (&R_C)[4],
                                                         kDataType* __restrict__ dKV,
                                                         const int kv_gmem_offset,
                                                         const int K_tile_id, const int kv_frag,
                                                         const int d_col_base, const int Nkv,
                                                         const float scale) {
  using Traits = DtypeTraits<kDataType>;
  const int lane_id = threadIdx.x % WARP_SIZE;
  const float* c_frag = reinterpret_cast<const float*>(&R_C[0]);
#pragma unroll
  for (int reg_id = 0; reg_id < 4; ++reg_id) {
    const int row = K_tile_id * 64 + kv_frag * 16 + c_frag_local_row(lane_id, reg_id);
    const int col = d_col_base + (lane_id % 4) * 2 + (reg_id & 1);
    if (row < Nkv && col < kHeadDim) {
      atomicAdd(&dKV[kv_gmem_offset + row * kHeadDim + col],
                Traits::from_float(c_frag[reg_id] * scale));
    }
  }
}

template <typename kDataType, const int kHeadDim, const int Bc>
__device__ __forceinline__ void atomic_add_c_frag_to_dkv_tile(
    const uint32_t (&R_C)[4], kDataType* __restrict__ dKV, const int kv_gmem_offset,
    const int K_tile_id, const int kv_frag, const int d_col_base, const int Nkv,
    const float scale) {
  using Traits = DtypeTraits<kDataType>;
  const int lane_id = threadIdx.x % WARP_SIZE;
  const float* c_frag = reinterpret_cast<const float*>(&R_C[0]);
#pragma unroll
  for (int reg_id = 0; reg_id < 4; ++reg_id) {
    const int row = K_tile_id * Bc + kv_frag * 16 + c_frag_local_row(lane_id, reg_id);
    const int col = d_col_base + (lane_id % 4) * 2 + (reg_id & 1);
    if (row < Nkv && col < kHeadDim) {
      atomicAdd(&dKV[kv_gmem_offset + row * kHeadDim + col],
                Traits::from_float(c_frag[reg_id] * scale));
    }
  }
}

template <typename kDataType, const int kHeadDim>
__device__ __forceinline__ float dot_do_o_row_4lane(const kDataType* __restrict__ dO,
                                                    const kDataType* __restrict__ O,
                                                    const int q_gmem_offset, const int row,
                                                    const int Nq) {
  using Traits = DtypeTraits<kDataType>;
  const int lane_id = threadIdx.x % WARP_SIZE;
  const int lane_col = lane_id % 4;
  float acc = 0.0f;
  if (row < Nq) {
#pragma unroll 1
    for (int d = lane_col; d < kHeadDim; d += 4) {
      const int addr = q_gmem_offset + row * kHeadDim + d;
      acc += Traits::to_float(dO[addr]) * Traits::to_float(O[addr]);
    }
  }
  acc += __shfl_xor_sync(0xffffffff, acc, 1);
  acc += __shfl_xor_sync(0xffffffff, acc, 2);
  return acc;
}

template <typename kDataType, const int Br, const int Bc, const int QTileSize, const int KTileSize,
          const int VTileSize, const int kHeadDim, const int kMmaAtomK, const int kNumThreads,
          const int kPadQ, const int kPadK, const int kPadV>
__device__ __forceinline__ void cp_async_phase1_g2s(
    const uint32_t smem_Q_base_ptr, const uint32_t smem_dO_base_ptr, const uint32_t smem_K_base_ptr,
    const uint32_t smem_V_base_ptr, const kDataType* __restrict__ Q,
    const kDataType* __restrict__ dO, const kDataType* __restrict__ K,
    const kDataType* __restrict__ V, const int q_gmem_offset, const int kv_gmem_offset,
    const int Q_tile_id, const int K_tile_id, const int tile_d, const int stage, const int Nq,
    const int Nkv) {
  ffpa::prefill::cp_async_qkv_g2s<Br, QTileSize, kHeadDim, kMmaAtomK, kNumThreads, kPadQ>(
      smem_Q_base_ptr, Q, q_gmem_offset, Q_tile_id, tile_d, stage, Nq);
  ffpa::prefill::cp_async_qkv_g2s<Br, QTileSize, kHeadDim, kMmaAtomK, kNumThreads, kPadQ>(
      smem_dO_base_ptr, dO, q_gmem_offset, Q_tile_id, tile_d, stage, Nq);
  ffpa::prefill::cp_async_qkv_g2s<Bc, KTileSize, kHeadDim, kMmaAtomK, kNumThreads, kPadK>(
      smem_K_base_ptr, K, kv_gmem_offset, K_tile_id, tile_d, stage, Nkv);
  ffpa::prefill::cp_async_qkv_g2s<Bc, VTileSize, kHeadDim, kMmaAtomK, kNumThreads, kPadV>(
      smem_V_base_ptr, V, kv_gmem_offset, K_tile_id, tile_d, stage, Nkv);
  ffpa::cp_async::commit_group();
}

template <typename kDataType, const int Br, const int Bc, const int QTileSize, const int KTileSize,
          const int kHeadDim, const int kMmaAtomK, const int kNumThreads, const int kPadQ,
          const int kPadK>
__device__ __forceinline__ void cp_async_phase2_g2s(
    const uint32_t smem_Q_base_ptr, const uint32_t smem_dO_base_ptr, const uint32_t smem_K_base_ptr,
    const kDataType* __restrict__ Q, const kDataType* __restrict__ dO,
    const kDataType* __restrict__ K, const int q_gmem_offset, const int kv_gmem_offset,
    const int Q_tile_id, const int K_tile_id, const int tile_d, const int stage, const int Nq,
    const int Nkv) {
  ffpa::prefill::cp_async_qkv_g2s<Br, QTileSize, kHeadDim, kMmaAtomK, kNumThreads, kPadQ>(
      smem_Q_base_ptr, Q, q_gmem_offset, Q_tile_id, tile_d, stage, Nq);
  ffpa::prefill::cp_async_qkv_g2s<Br, QTileSize, kHeadDim, kMmaAtomK, kNumThreads, kPadQ>(
      smem_dO_base_ptr, dO, q_gmem_offset, Q_tile_id, tile_d, stage, Nq);
  ffpa::prefill::cp_async_qkv_g2s<Bc, KTileSize, kHeadDim, kMmaAtomK, kNumThreads, kPadK>(
      smem_K_base_ptr, K, kv_gmem_offset, K_tile_id, tile_d, stage, Nkv);
  ffpa::cp_async::commit_group();
}

template <typename kDataType, const int Br, const int QTileSize, const int kHeadDim,
          const int kMmaAtomK, const int kNumThreads, const int kPadQ>
__device__ __forceinline__ void cp_async_phase1_persistent_qdo_g2s(
    const uint32_t smem_Q_base_ptr, const uint32_t smem_dO_base_ptr,
    const kDataType* __restrict__ Q, const kDataType* __restrict__ dO, const int q_gmem_offset,
    const int Q_tile_id, const int tile_d, const int stage, const int Nq) {
  ffpa::prefill::cp_async_qkv_g2s<Br, QTileSize, kHeadDim, kMmaAtomK, kNumThreads, kPadQ>(
      smem_Q_base_ptr, Q, q_gmem_offset, Q_tile_id, tile_d, stage, Nq);
  ffpa::prefill::cp_async_qkv_g2s<Br, QTileSize, kHeadDim, kMmaAtomK, kNumThreads, kPadQ>(
      smem_dO_base_ptr, dO, q_gmem_offset, Q_tile_id, tile_d, stage, Nq);
  ffpa::cp_async::commit_group();
}

template <typename kDataType, const int Br, const int QTileSize, const int kHeadDim,
          const int kMmaAtomK, const int kNumThreads, const int kPadQ>
__device__ __forceinline__ void cp_async_phase2_persistent_qdo_g2s(
    const uint32_t smem_Q_base_ptr, const uint32_t smem_dO_base_ptr,
    const kDataType* __restrict__ Q, const kDataType* __restrict__ dO, const int q_gmem_offset,
    const int Q_tile_id, const int tile_d, const int stage, const int Nq) {
  ffpa::prefill::cp_async_qkv_g2s<Br, QTileSize, kHeadDim, kMmaAtomK, kNumThreads, kPadQ>(
      smem_Q_base_ptr, Q, q_gmem_offset, Q_tile_id, tile_d, stage, Nq);
  ffpa::prefill::cp_async_qkv_g2s<Br, QTileSize, kHeadDim, kMmaAtomK, kNumThreads, kPadQ>(
      smem_dO_base_ptr, dO, q_gmem_offset, Q_tile_id, tile_d, stage, Nq);
  ffpa::cp_async::commit_group();
}

template <typename kDataType, const int Bc, const int KTileSize, const int kHeadDim,
          const int kMmaAtomK, const int kPad>
__device__ __forceinline__ void cp_async_persistent_kv_g2s(const uint32_t smem_base_ptr,
                                                           const kDataType* __restrict__ gmem_ptr,
                                                           const int gmem_offset,
                                                           const int K_tile_id, const int tile_d,
                                                           const int stage, const int Nkv) {
  static_assert(Bc == 32 || Bc == 64, "Persistent-K resident loader supports Bc=32 or Bc=64.");
  static_assert(kMmaAtomK == 16, "Persistent-KV resident loader expects 16-wide D subtiles.");
  const int tid = threadIdx.x;
  if (tid >= (Bc * 2)) {
    return;
  }
  constexpr bool kSwizzle = (kPad == 0) ? true : false;
  const int row = tid / 2;
  const int col = (tid & 1) * 8;
  const int gmem_row = K_tile_id * Bc + row;
  const int gmem_col = tile_d * kMmaAtomK + col;
  const int gmem_addr = gmem_offset + gmem_row * kHeadDim + gmem_col;
  const uint32_t smem_ptr =
      (smem_base_ptr + (stage * KTileSize + row * (kMmaAtomK + kPad) +
                        (kSwizzle ? ffpa::swizzle::permuted<kMmaAtomK>(row, col) : col)) *
                           sizeof(kDataType));
  ffpa::cp_async::cp_async_zfill<16>(smem_ptr, &(gmem_ptr[gmem_addr]), gmem_row < Nkv);
}

}  // namespace bwd
}  // namespace ffpa

template <typename kDataType, const int kHeadDim, const int kMmaAtomM, const int kMmaAtomN,
          const int kMmaAtomK, const int kMmaTileSeqLenQ, const int kMmaTileSeqLenK,
          const int kMmaTileSeqLenP, const int kMmaTileHeadDimV, const int kValTileSeqLenQ,
          const int kValTileSeqLenK, const int kValTileSeqLenP, const int kValTileHeadDimV,
          const int kMmaAccFloat32QK, const int kMmaAccFloat32PV, const int kOStorageAccFloat32,
          const int kPrefetchQK, const int kPrefetchPV, const int kShareSmemQKV,
          const int kPersistQs2r, const int kPersistQg2s, const int kRegPipeKV, const int kStageQK,
          const int kStagePV, const int kPadQ, const int kPadK, const int kPadV>
__global__ void __launch_bounds__(WARP_SIZE* kMmaTileSeqLenQ* kMmaTileSeqLenK)
    ffpa_attn_split_d_bwd_template(const kDataType* __restrict__ Q, const kDataType* __restrict__ K,
                                   const kDataType* __restrict__ V,
                                   const float* __restrict__ softmax_lse,
                                   const kDataType* __restrict__ dO,
                                   const kDataType* __restrict__ O, kDataType* __restrict__ dQ,
                                   kDataType* __restrict__ dK, kDataType* __restrict__ dV,
                                   const int Nq, const int Nkv, const int Nh, const int Nh_kv,
                                   const float scale, const int Tc, const int causal) {
  ffpa::prefill::check_large_d_compiling_states<
      kHeadDim, kMmaAtomM, kMmaAtomN, kMmaAtomK, kMmaTileSeqLenQ, kMmaTileSeqLenK, kMmaTileSeqLenP,
      kMmaTileHeadDimV, kValTileSeqLenQ, kValTileSeqLenK, kValTileSeqLenP, kValTileHeadDimV,
      kMmaAccFloat32QK, kMmaAccFloat32PV, kOStorageAccFloat32, kPrefetchQK, kPrefetchPV,
      kShareSmemQKV, kPersistQs2r, kPersistQg2s, kRegPipeKV, kStageQK, kStagePV, kPadQ, kPadK,
      kPadV>();

  constexpr int Br = kMmaAtomM * kMmaTileSeqLenQ * kValTileSeqLenQ;
  constexpr int Bc = kMmaAtomN * kMmaTileSeqLenK * kValTileSeqLenK;
  constexpr int kNumThreads = WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK;
  // cp.async.wait_group counts committed groups, not individual cp.async
  // instructions and not calls to cp_async_qkv_g2s. Each D tile currently has
  // one commit_group: Phase 1 packs 4 logical copies (Q/dO/K/V) into it, and
  // Phase 2 packs 3 logical copies (Q/dO/K). If those logical copies are later
  // split across multiple commit_group calls, update these constants.
  constexpr int kPhase1CpAsyncCommitGroupsPerTile = 1;
  constexpr int kPhase2CpAsyncCommitGroupsPerTile = 1;
  constexpr int kPhase1CpAsyncWaitGroups =
      (kStageQK > 1) ? ((kStageQK - 2) * kPhase1CpAsyncCommitGroupsPerTile) : 0;
  constexpr int kPhase2CpAsyncWaitGroups =
      (kStageQK > 1) ? ((kStageQK - 2) * kPhase2CpAsyncCommitGroupsPerTile) : 0;

  static_assert(Br == 64, "Split-D backward stage-1 target currently requires Br=64.");
  static_assert(Bc == 64, "Split-D backward stage-1 target currently requires Bc=64.");
  static_assert(kMmaAccFloat32QK == 1 && kMmaAccFloat32PV == 1,
                "Split-D backward requires fp32 MMA accumulators.");
  static_assert(kStageQK <= 3 && kStagePV <= 3,
                "Split-D backward currently supports only stage<=3.");
  static_assert(kStageQK == kStagePV,
                "Split-D backward Phase 1 pipelines Q/dO/K/V with the same stage count.");

#ifdef ENABLE_FFPA_LAUNCH_GRID_DNHB
  const int Nb_id = blockIdx.z;
  const int kv_head_idx = blockIdx.y;
#else
  const int Nb_id = blockIdx.y / Nh_kv;
  const int kv_head_idx = blockIdx.y % Nh_kv;
#endif
  const int K_tile_id = blockIdx.x;
  if (K_tile_id >= Tc || K_tile_id * Bc >= Nkv) {
    return;
  }

  const int group_size = Nh / Nh_kv;
  const int first_q_head = kv_head_idx * group_size;
  const int kv_gmem_offset = (Nb_id * Nh_kv * Nkv * kHeadDim) + (kv_head_idx * Nkv * kHeadDim);
  const int q_batch_offset = Nb_id * Nh * Nq * kHeadDim;
  const int lse_batch_offset = Nb_id * Nh * Nq;

  extern __shared__ __align__(16) unsigned char ffpa_smem_raw[];
  kDataType* smem = reinterpret_cast<kDataType*>(ffpa_smem_raw);
  constexpr int Q_tile_size = Br * (kMmaAtomK + kPadQ);
  constexpr int K_tile_size = Bc * (kMmaAtomK + kPadK);
  constexpr int V_tile_size = Bc * (kMmaAtomK + kPadV);
  constexpr int kTransposeScratchPad = 8;
  constexpr int Transpose_tile_size = Bc * (kMmaAtomK + kTransposeScratchPad);
  kDataType* Q_tile_smem = smem;
  kDataType* dO_tile_smem = Q_tile_smem + kStageQK * Q_tile_size;
  kDataType* O_tile_smem = dO_tile_smem + kStageQK * Q_tile_size;
  kDataType* K_tile_smem = O_tile_smem + kStageQK * Q_tile_size;
  kDataType* V_tile_smem = K_tile_smem + kStageQK * K_tile_size;
  kDataType* P_t_smem = V_tile_smem + kStagePV * V_tile_size;
  kDataType* dS_t_smem = P_t_smem + kMmaTileSeqLenQ * Transpose_tile_size;
  const uint32_t smem_Q_base_ptr = __cvta_generic_to_shared(Q_tile_smem);
  const uint32_t smem_dO_base_ptr = __cvta_generic_to_shared(dO_tile_smem);
  const uint32_t smem_K_base_ptr = __cvta_generic_to_shared(K_tile_smem);
  const uint32_t smem_V_base_ptr = __cvta_generic_to_shared(V_tile_smem);

  const int warp_QP = threadIdx.x / WARP_SIZE;
  constexpr int warp_KV = 0;
  const int Tr = ffpa::utils::div_ceil(Nq, Br);

  uint32_t R_Q[4];
  uint32_t R_dO[4];
  uint32_t R_K[2];
  uint32_t R_VK[2];
  uint32_t R_S[kValTileSeqLenK][4];
  uint32_t R_dP[kValTileSeqLenK][4];
  uint32_t R_P[kValTileSeqLenK][2];
  uint32_t R_dS[kValTileSeqLenK][2];
  uint32_t R_dQ[4];
  uint32_t R_QB[2];
  uint32_t R_dOB[2];
  uint32_t R_P_T[4];
  uint32_t R_dS_T[4];
  uint32_t R_dK[4];
  uint32_t R_dV[4];

#pragma unroll 1
  for (int q_head_offset = 0; q_head_offset < group_size; ++q_head_offset) {
    const int q_head_idx = first_q_head + q_head_offset;
    const int q_gmem_offset = q_batch_offset + q_head_idx * Nq * kHeadDim;

#pragma unroll 1
    for (int Q_tile_id = Tr - 1; Q_tile_id >= 0; --Q_tile_id) {
      if (Q_tile_id * Br >= Nq) {
        continue;
      }
      const int Br_base = Q_tile_id * Br;
      const int kv_offset = Nkv - Nq;
      if (causal && (Br_base + Br - 1 + kv_offset) < K_tile_id * Bc) {
        continue;
      }

      ffpa::utils::fill_2D_regs<uint32_t, kValTileSeqLenK, 4>(R_S, 0);
      ffpa::utils::fill_2D_regs<uint32_t, kValTileSeqLenK, 4>(R_dP, 0);
      ffpa::utils::fill_2D_regs<uint32_t, kValTileSeqLenK, 2>(R_P, 0);
      ffpa::utils::fill_2D_regs<uint32_t, kValTileSeqLenK, 2>(R_dS, 0);

      const int lane_id = threadIdx.x % WARP_SIZE;
      const int row_in_warp0 = warp_QP * kMmaAtomM + lane_id / 4;
      const int row_in_warp8 = row_in_warp0 + 8;
      const int q_row0 = Q_tile_id * Br + row_in_warp0;
      const int q_row8 = Q_tile_id * Br + row_in_warp8;

      // Softmax backward row correction:
      //   dP_sum[row] = rowsum(dO[row, :] * O[row, :])
      // This is independent of the KV tile and is consumed by
      //   dS = P * (dP - dP_sum).
      // Current implementation computes it inside every KV-tile block, so the
      // same Q row is recomputed across K_tile_id. A future faster design could
      // precompute D[row] once per (B,H,Q) row, as FlashAttention backward does,
      // or fuse it into a Q-driven prepass to remove this repeated global IO.
      const float dp_sum_row0 =
          ffpa::bwd::dot_do_o_row_4lane<kDataType, kHeadDim>(dO, O, q_gmem_offset, q_row0, Nq);
      const float dp_sum_row8 =
          ffpa::bwd::dot_do_o_row_4lane<kDataType, kHeadDim>(dO, O, q_gmem_offset, q_row8, Nq);

      // Phase 1: reconstruct the attention logits and dP for this Q/KV tile.
      // MMA math, accumulated across the full head dimension:
      //   S  = Q  @ K^T
      //   dP = dO @ V^T
      // Note: Q/dO/K are loaded again in Phase 2 below. That duplicate global
      // load is one of the main current bottlenecks. The stage=2 path below
      // overlaps each phase's gmem->smem copies with MMA on the previous D
      // tile, but it does not remove the cross-phase duplicate global pass.
      if constexpr (kStageQK > 1) {
#pragma unroll
        for (int stage = 0; stage < (kStageQK - 1); ++stage) {
          ffpa::bwd::cp_async_phase1_g2s<kDataType, Br, Bc, Q_tile_size, K_tile_size, V_tile_size,
                                         kHeadDim, kMmaAtomK, kNumThreads, kPadQ, kPadK, kPadV>(
              smem_Q_base_ptr, smem_dO_base_ptr, smem_K_base_ptr, smem_V_base_ptr, Q, dO, K, V,
              q_gmem_offset, kv_gmem_offset, Q_tile_id, K_tile_id, stage, stage, Nq, Nkv);
        }
        ffpa::cp_async::wait_group<kPhase1CpAsyncWaitGroups>();
        __syncthreads();
      }

#pragma unroll 1
      for (int tile_d = 0; tile_d < (kHeadDim / kMmaAtomK); ++tile_d) {
        const int smem_sel = tile_d % kStageQK;
        if constexpr (kStageQK > 1) {
          constexpr int kPipelineDistance = kStageQK - 1;
          const int next_tile_d = tile_d + kPipelineDistance;
          const int smem_sel_next = next_tile_d % kStageQK;
          if (next_tile_d < (kHeadDim / kMmaAtomK)) {
            ffpa::bwd::cp_async_phase1_g2s<kDataType, Br, Bc, Q_tile_size, K_tile_size, V_tile_size,
                                           kHeadDim, kMmaAtomK, kNumThreads, kPadQ, kPadK, kPadV>(
                smem_Q_base_ptr, smem_dO_base_ptr, smem_K_base_ptr, smem_V_base_ptr, Q, dO, K, V,
                q_gmem_offset, kv_gmem_offset, Q_tile_id, K_tile_id, next_tile_d, smem_sel_next, Nq,
                Nkv);
          }
        } else {
          ffpa::bwd::cp_async_phase1_g2s<kDataType, Br, Bc, Q_tile_size, K_tile_size, V_tile_size,
                                         kHeadDim, kMmaAtomK, kNumThreads, kPadQ, kPadK, kPadV>(
              smem_Q_base_ptr, smem_dO_base_ptr, smem_K_base_ptr, smem_V_base_ptr, Q, dO, K, V,
              q_gmem_offset, kv_gmem_offset, Q_tile_id, K_tile_id, tile_d, smem_sel, Nq, Nkv);
          ffpa::cp_async::wait_group<0>();
          __syncthreads();
        }

        ffpa::prefill::sync_fetch_qkv_frags_s2r<0, 4, Q_tile_size, kMmaAtomM, kMmaAtomN, kMmaAtomK,
                                                kPadQ, kDataType>(smem_Q_base_ptr, &R_Q[0], warp_QP,
                                                                  0, 0, smem_sel);
        ffpa::prefill::sync_fetch_qkv_frags_s2r<0, 4, Q_tile_size, kMmaAtomM, kMmaAtomN, kMmaAtomK,
                                                kPadQ, kDataType>(smem_dO_base_ptr, &R_dO[0],
                                                                  warp_QP, 0, 0, smem_sel);

#pragma unroll
        for (int kv_frag = 0; kv_frag < kValTileSeqLenK; ++kv_frag) {
          ffpa::prefill::sync_fetch_qkv_frags_s2r<0, 2, K_tile_size, kMmaAtomM, kMmaAtomN,
                                                  kMmaAtomK, kPadK, kDataType>(
              smem_K_base_ptr, &R_K[0], warp_KV, kv_frag, 0, smem_sel);
          ffpa::prefill::sync_fetch_qkv_frags_s2r<0, 2, V_tile_size, kMmaAtomM, kMmaAtomN,
                                                  kMmaAtomK, kPadV, kDataType>(
              smem_V_base_ptr, &R_VK[0], warp_KV, kv_frag, 0, smem_sel);
          ffpa::mma::m16n8k16_abf32<kDataType, ffpa::mma::MMAMode::kInplaceUpdate>(
              &R_S[kv_frag][0], &R_S[kv_frag][1], &R_S[kv_frag][2], &R_S[kv_frag][3], &R_Q[0],
              &R_Q[1], &R_Q[2], &R_Q[3], &R_K[0], &R_K[1]);
          ffpa::mma::m16n8k16_abf32<kDataType, ffpa::mma::MMAMode::kInplaceUpdate>(
              &R_dP[kv_frag][0], &R_dP[kv_frag][1], &R_dP[kv_frag][2], &R_dP[kv_frag][3], &R_dO[0],
              &R_dO[1], &R_dO[2], &R_dO[3], &R_VK[0], &R_VK[1]);
        }
        if constexpr (kStageQK > 1) {
          if (tile_d < (kHeadDim / kMmaAtomK - 1)) {
            ffpa::cp_async::wait_group<kPhase1CpAsyncWaitGroups>();
            __syncthreads();
          }
        } else {
          __syncthreads();
        }
      }

      const int kv_valid_local = Nkv - K_tile_id * Bc;
      if (kv_valid_local < Bc) {
        ffpa::prefill::sync_apply_kv_mask<kValTileSeqLenK, kMmaAccFloat32QK, kDataType>(
            &R_S[0][0], kv_valid_local);
      }
      if (causal) {
        ffpa::prefill::sync_apply_causal_mask<kValTileSeqLenK, kMmaAccFloat32QK, kDataType>(
            &R_S[0][0], warp_QP, Br_base, K_tile_id * Bc, kv_offset);
      }

      const int lse_offset = lse_batch_offset + q_head_idx * Nq;
      const float lse_row0 = (q_row0 < Nq) ? softmax_lse[lse_offset + q_row0] : INFINITY;
      const float lse_row8 = (q_row8 < Nq) ? softmax_lse[lse_offset + q_row8] : INFINITY;
      // Phase 1b/1c: reconstruct softmax probability and softmax gradient.
      //   P  = exp(S * scale - LSE)
      //   dS = P * (dP - dP_sum)
      // P and dS are packed back to activation dtype so Tensor Core MMA can be
      // used in Phase 2. This sacrifices some gradient precision versus fp32
      // dS/P storage but keeps register and shared-memory pressure manageable.
      ffpa::bwd::compute_p_and_ds_from_lse<kDataType, kValTileSeqLenK>(
          R_S, R_dP, R_P, R_dS, lse_row0, lse_row8, dp_sum_row0, dp_sum_row8, scale);

      // C-fragment -> transposed shared-memory staging for dK/dV:
      //   dK = scale * dS^T @ Q
      //   dV = P^T @ dO
      // R_P/R_dS are produced in MMA C-fragment layout, but the dK/dV MMA wants
      // them as A operands. The current safe path stores a transposed P/dS tile
      // to shared memory and reloads it with ldmatrix.x4. This extra smem write,
      // read, and barrier are expensive. A high-value future optimization is a
      // register-level C-fragment -> A-fragment transpose that removes this
      // scratch path entirely.
      kDataType* warp_P_t_smem = P_t_smem + warp_QP * Transpose_tile_size;
      kDataType* warp_dS_t_smem = dS_t_smem + warp_QP * Transpose_tile_size;
      ffpa::bwd::store_packed_c_frag_transposed_to_smem<kDataType, kValTileSeqLenK,
                                                        kMmaAtomK + kTransposeScratchPad>(
          R_P, warp_P_t_smem);
      ffpa::bwd::store_packed_c_frag_transposed_to_smem<kDataType, kValTileSeqLenK,
                                                        kMmaAtomK + kTransposeScratchPad>(
          R_dS, warp_dS_t_smem);
      __syncthreads();
      const uint32_t smem_warp_P_t_base_ptr = __cvta_generic_to_shared(warp_P_t_smem);
      const uint32_t smem_warp_dS_t_base_ptr = __cvta_generic_to_shared(warp_dS_t_smem);

      // Phase 2: gradient MMA and global accumulation.
      //   dQ += scale * dS @ K
      //   dK += scale * dS^T @ Q
      //   dV += P^T @ dO
      // Current bottleneck: Phase 2 reloads Q/dO/K from global memory even
      // though Phase 1 has already loaded the same Q/dO/K D-slices to build
      // S and dP. Removing or hiding this duplicate gmem traffic is one of the
      // highest-priority future optimizations; possible directions include a
      // true staged pipeline, keeping selected fragments live across phases, or
      // restructuring the loop so Phase 2 consumes the Phase 1 staged data.
      // dQ is partial across KV tiles, so it needs either atomicAdd or a
      // separate reduction design. dK/dV are partial across Q tiles within this
      // KV-owning block; atomics are correct but very expensive. A future fast
      // design should accumulate dK/dV per block in registers/shared memory by
      // D-slice and write each KV tile once, avoiding global atomics while still
      // staying within the per-block shared-memory budget.
      if constexpr (kStageQK > 1) {
#pragma unroll
        for (int stage = 0; stage < (kStageQK - 1); ++stage) {
          ffpa::bwd::cp_async_phase2_g2s<kDataType, Br, Bc, Q_tile_size, K_tile_size, kHeadDim,
                                         kMmaAtomK, kNumThreads, kPadQ, kPadK>(
              smem_Q_base_ptr, smem_dO_base_ptr, smem_K_base_ptr, Q, dO, K, q_gmem_offset,
              kv_gmem_offset, Q_tile_id, K_tile_id, stage, stage, Nq, Nkv);
        }
        ffpa::cp_async::wait_group<kPhase2CpAsyncWaitGroups>();
        __syncthreads();
      }

#pragma unroll 1
      for (int tile_d = 0; tile_d < (kHeadDim / kMmaAtomK); ++tile_d) {
        const int smem_sel = tile_d % kStageQK;
        if constexpr (kStageQK > 1) {
          constexpr int kPipelineDistance = kStageQK - 1;
          const int next_tile_d = tile_d + kPipelineDistance;
          const int smem_sel_next = next_tile_d % kStageQK;
          if (next_tile_d < (kHeadDim / kMmaAtomK)) {
            ffpa::bwd::cp_async_phase2_g2s<kDataType, Br, Bc, Q_tile_size, K_tile_size, kHeadDim,
                                           kMmaAtomK, kNumThreads, kPadQ, kPadK>(
                smem_Q_base_ptr, smem_dO_base_ptr, smem_K_base_ptr, Q, dO, K, q_gmem_offset,
                kv_gmem_offset, Q_tile_id, K_tile_id, next_tile_d, smem_sel_next, Nq, Nkv);
          }
        } else {
          ffpa::bwd::cp_async_phase2_g2s<kDataType, Br, Bc, Q_tile_size, K_tile_size, kHeadDim,
                                         kMmaAtomK, kNumThreads, kPadQ, kPadK>(
              smem_Q_base_ptr, smem_dO_base_ptr, smem_K_base_ptr, Q, dO, K, q_gmem_offset,
              kv_gmem_offset, Q_tile_id, K_tile_id, tile_d, smem_sel, Nq, Nkv);
          ffpa::cp_async::wait_group<0>();
          __syncthreads();
        }

#pragma unroll
        for (int d_subtile = 0; d_subtile < 2; ++d_subtile) {
          ffpa::utils::fill_1D_regs<uint32_t, 4>(R_dQ, 0);
#pragma unroll
          for (int kv_frag = 0; kv_frag < (Bc / kMmaAtomK); ++kv_frag) {
            ffpa::prefill::sync_fetch_qkv_frags_s2r<1, 2, K_tile_size, kMmaAtomM, kMmaAtomN,
                                                    kMmaAtomK, kPadK, kDataType>(
                smem_K_base_ptr, &R_K[0], warp_KV, d_subtile, kv_frag, smem_sel);
            const int ds_offset = kv_frag * 2;
            ffpa::mma::m16n8k16_abf32<kDataType, ffpa::mma::MMAMode::kInplaceUpdate>(
                &R_dQ[0], &R_dQ[1], &R_dQ[2], &R_dQ[3], &R_dS[ds_offset][0], &R_dS[ds_offset][1],
                &R_dS[ds_offset + 1][0], &R_dS[ds_offset + 1][1], &R_K[0], &R_K[1]);
          }
          const int d_col_base = tile_d * kMmaAtomK + d_subtile * kMmaAtomN;
          ffpa::bwd::atomic_add_c_frag_to_dq<kDataType, kHeadDim>(
              R_dQ, dQ, q_gmem_offset, Q_tile_id, warp_QP, d_col_base, Nq, scale);

          ffpa::prefill::sync_fetch_qkv_frags_s2r<1, 2, Q_tile_size, kMmaAtomM, kMmaAtomN,
                                                  kMmaAtomK, kPadQ, kDataType>(
              smem_Q_base_ptr, &R_QB[0], warp_KV, d_subtile, warp_QP, smem_sel);
          ffpa::prefill::sync_fetch_qkv_frags_s2r<1, 2, Q_tile_size, kMmaAtomM, kMmaAtomN,
                                                  kMmaAtomK, kPadQ, kDataType>(
              smem_dO_base_ptr, &R_dOB[0], warp_KV, d_subtile, warp_QP, smem_sel);
#pragma unroll
          for (int kv_frag = 0; kv_frag < (Bc / kMmaAtomK); ++kv_frag) {
            ffpa::utils::fill_1D_regs<uint32_t, 4>(R_dK, 0);
            ffpa::utils::fill_1D_regs<uint32_t, 4>(R_dV, 0);
            ffpa::prefill::sync_fetch_qkv_frags_s2r<0, 4, Transpose_tile_size, kMmaAtomM, kMmaAtomN,
                                                    kMmaAtomK, kTransposeScratchPad, kDataType>(
                smem_warp_dS_t_base_ptr, &R_dS_T[0], kv_frag, 0, 0, 0);
            ffpa::prefill::sync_fetch_qkv_frags_s2r<0, 4, Transpose_tile_size, kMmaAtomM, kMmaAtomN,
                                                    kMmaAtomK, kTransposeScratchPad, kDataType>(
                smem_warp_P_t_base_ptr, &R_P_T[0], kv_frag, 0, 0, 0);
            ffpa::mma::m16n8k16_abf32<kDataType, ffpa::mma::MMAMode::kInplaceUpdate>(
                &R_dK[0], &R_dK[1], &R_dK[2], &R_dK[3], &R_dS_T[0], &R_dS_T[1], &R_dS_T[2],
                &R_dS_T[3], &R_QB[0], &R_QB[1]);
            ffpa::mma::m16n8k16_abf32<kDataType, ffpa::mma::MMAMode::kInplaceUpdate>(
                &R_dV[0], &R_dV[1], &R_dV[2], &R_dV[3], &R_P_T[0], &R_P_T[1], &R_P_T[2], &R_P_T[3],
                &R_dOB[0], &R_dOB[1]);
            ffpa::bwd::atomic_add_c_frag_to_dkv<kDataType, kHeadDim>(
                R_dK, dK, kv_gmem_offset, K_tile_id, kv_frag, d_col_base, Nkv, scale);
            ffpa::bwd::atomic_add_c_frag_to_dkv<kDataType, kHeadDim>(
                R_dV, dV, kv_gmem_offset, K_tile_id, kv_frag, d_col_base, Nkv, 1.0f);
          }
        }
        if constexpr (kStageQK > 1) {
          if (tile_d < (kHeadDim / kMmaAtomK - 1)) {
            ffpa::cp_async::wait_group<kPhase2CpAsyncWaitGroups>();
            __syncthreads();
          }
        } else {
          __syncthreads();
        }
      }
    }
  }
}

template <typename kDataType, const int kHeadDim, const int kMmaAtomM, const int kMmaAtomN,
          const int kMmaAtomK, const int kMmaTileSeqLenQ, const int kMmaTileSeqLenK,
          const int kMmaTileSeqLenP, const int kMmaTileHeadDimV, const int kValTileSeqLenQ,
          const int kValTileSeqLenK, const int kValTileSeqLenP, const int kValTileHeadDimV,
          const int kMmaAccFloat32QK, const int kMmaAccFloat32PV, const int kOStorageAccFloat32,
          const int kPrefetchQK, const int kPrefetchPV, const int kShareSmemQKV,
          const int kPersistQs2r, const int kPersistQg2s, const int kRegPipeKV, const int kStageQK,
          const int kStagePV, const int kPadQ, const int kPadK, const int kPadV>
__global__ void __launch_bounds__(WARP_SIZE* kMmaTileSeqLenQ* kMmaTileSeqLenK)
    ffpa_attn_persistent_kv_bwd_template(
        const kDataType* __restrict__ Q, const kDataType* __restrict__ K,
        const kDataType* __restrict__ V, const float* __restrict__ softmax_lse,
        const kDataType* __restrict__ dO, const kDataType* __restrict__ O,
        kDataType* __restrict__ dQ, kDataType* __restrict__ dK, kDataType* __restrict__ dV,
        const int Nq, const int Nkv, const int Nh, const int Nh_kv, const float scale, const int Tc,
        const int causal) {
  ffpa::prefill::check_large_d_compiling_states<
      kHeadDim, kMmaAtomM, kMmaAtomN, kMmaAtomK, kMmaTileSeqLenQ, kMmaTileSeqLenK, kMmaTileSeqLenP,
      kMmaTileHeadDimV, kValTileSeqLenQ, kValTileSeqLenK, kValTileSeqLenP, kValTileHeadDimV,
      kMmaAccFloat32QK, kMmaAccFloat32PV, kOStorageAccFloat32, kPrefetchQK, kPrefetchPV,
      kShareSmemQKV, kPersistQs2r, kPersistQg2s, kRegPipeKV, kStageQK, kStagePV, kPadQ, kPadK,
      kPadV>();

  constexpr int Br = kMmaAtomM * kMmaTileSeqLenQ * kValTileSeqLenQ;
  constexpr int Bc = kMmaAtomN * kMmaTileSeqLenK * kValTileSeqLenK;
  constexpr int kNumThreads = WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK;
  constexpr int kNumDSubtiles = kHeadDim / kMmaAtomK;
  constexpr int kPhase1CpAsyncWaitGroups = (kStageQK > 1) ? (kStageQK - 2) : 0;
  constexpr int kPhase2CpAsyncWaitGroups = (kStageQK > 1) ? (kStageQK - 2) : 0;

  static_assert(kHeadDim <= 512, "Persistent-KV backward supports only D<=512.");
  static_assert(Br == 64, "Persistent-KV backward first target requires Br=64.");
  static_assert(Bc == 64, "Persistent-K backward optimized target requires Bc=64.");
  static_assert(kMmaAccFloat32QK == 1 && kMmaAccFloat32PV == 1,
                "Persistent-KV backward requires fp32 MMA accumulators.");
  static_assert(kStageQK <= 2 && kStagePV <= 2,
                "Persistent-KV backward currently supports only stage<=2.");
  static_assert(kStageQK == kStagePV,
                "Persistent-KV backward uses the same stage count for Q/dO/O streams.");

#ifdef ENABLE_FFPA_LAUNCH_GRID_DNHB
  const int Nb_id = blockIdx.z;
  const int kv_head_idx = blockIdx.y;
#else
  const int Nb_id = blockIdx.y / Nh_kv;
  const int kv_head_idx = blockIdx.y % Nh_kv;
#endif
  const int K_tile_id = blockIdx.x;
  if (K_tile_id >= Tc || K_tile_id * Bc >= Nkv) {
    return;
  }

  const int group_size = Nh / Nh_kv;
  const int first_q_head = kv_head_idx * group_size;
  const int kv_gmem_offset = (Nb_id * Nh_kv * Nkv * kHeadDim) + (kv_head_idx * Nkv * kHeadDim);
  const int q_batch_offset = Nb_id * Nh * Nq * kHeadDim;
  const int lse_batch_offset = Nb_id * Nh * Nq;

  extern __shared__ __align__(16) unsigned char ffpa_smem_raw[];
  kDataType* smem = reinterpret_cast<kDataType*>(ffpa_smem_raw);
  constexpr int Q_tile_size = Br * (kMmaAtomK + kPadQ);
  constexpr int K_tile_size = Bc * (kMmaAtomK + kPadK);
  constexpr int V_tile_size = Bc * (kMmaAtomK + kPadV);
  constexpr int kTransposeScratchPad = 8;
  constexpr int Transpose_tile_size = Bc * (kMmaAtomK + kTransposeScratchPad);
  kDataType* Q_tile_smem = smem;
  kDataType* dO_tile_smem = Q_tile_smem + kStageQK * Q_tile_size;
  kDataType* K_tile_smem = dO_tile_smem + kStageQK * Q_tile_size;
  kDataType* V_tile_smem = K_tile_smem + kNumDSubtiles * K_tile_size;
  kDataType* P_t_smem = V_tile_smem + kNumDSubtiles * V_tile_size;
  kDataType* dS_t_smem = P_t_smem + kMmaTileSeqLenQ * Transpose_tile_size;
  const uint32_t smem_Q_base_ptr = __cvta_generic_to_shared(Q_tile_smem);
  const uint32_t smem_dO_base_ptr = __cvta_generic_to_shared(dO_tile_smem);
  const uint32_t smem_K_base_ptr = __cvta_generic_to_shared(K_tile_smem);
  const uint32_t smem_V_base_ptr = __cvta_generic_to_shared(V_tile_smem);

#pragma unroll 1
  for (int tile_d = 0; tile_d < kNumDSubtiles; ++tile_d) {
    ffpa::bwd::cp_async_persistent_kv_g2s<kDataType, Bc, K_tile_size, kHeadDim, kMmaAtomK, kPadK>(
        smem_K_base_ptr, K, kv_gmem_offset, K_tile_id, tile_d, tile_d, Nkv);
    ffpa::bwd::cp_async_persistent_kv_g2s<kDataType, Bc, V_tile_size, kHeadDim, kMmaAtomK, kPadV>(
        smem_V_base_ptr, V, kv_gmem_offset, K_tile_id, tile_d, tile_d, Nkv);
    ffpa::cp_async::commit_group();
    if ((tile_d & 3) == 3) {
      ffpa::cp_async::wait_group<0>();
      __syncthreads();
    }
  }
  ffpa::cp_async::wait_group<0>();
  __syncthreads();

  const int warp_QP = threadIdx.x / WARP_SIZE;
  constexpr int warp_KV = 0;
  const int Tr = ffpa::utils::div_ceil(Nq, Br);

  uint32_t R_Q[4];
  uint32_t R_dO[4];
  uint32_t R_K[2];
  uint32_t R_VK[2];
  uint32_t R_S[kValTileSeqLenK][4];
  uint32_t R_dP[kValTileSeqLenK][4];
  uint32_t R_P[kValTileSeqLenK][2];
  uint32_t R_dS[kValTileSeqLenK][2];
  uint32_t R_dQ[4];
  uint32_t R_QB[2];
  uint32_t R_dOB[2];
  uint32_t R_P_T[4];
  uint32_t R_dS_T[4];
  uint32_t R_dK[4];
  uint32_t R_dV[4];

#pragma unroll 1
  for (int q_head_offset = 0; q_head_offset < group_size; ++q_head_offset) {
    const int q_head_idx = first_q_head + q_head_offset;
    const int q_gmem_offset = q_batch_offset + q_head_idx * Nq * kHeadDim;

#pragma unroll 1
    for (int Q_tile_id = Tr - 1; Q_tile_id >= 0; --Q_tile_id) {
      if (Q_tile_id * Br >= Nq) {
        continue;
      }
      const int Br_base = Q_tile_id * Br;
      const int kv_offset = Nkv - Nq;
      if (causal && (Br_base + Br - 1 + kv_offset) < K_tile_id * Bc) {
        continue;
      }

      ffpa::utils::fill_2D_regs<uint32_t, kValTileSeqLenK, 4>(R_S, 0);
      ffpa::utils::fill_2D_regs<uint32_t, kValTileSeqLenK, 4>(R_dP, 0);
      ffpa::utils::fill_2D_regs<uint32_t, kValTileSeqLenK, 2>(R_P, 0);
      ffpa::utils::fill_2D_regs<uint32_t, kValTileSeqLenK, 2>(R_dS, 0);

      const int lane_id = threadIdx.x % WARP_SIZE;
      const int row_in_warp0 = warp_QP * kMmaAtomM + lane_id / 4;
      const int row_in_warp8 = row_in_warp0 + 8;
      const int q_row0 = Q_tile_id * Br + row_in_warp0;
      const int q_row8 = Q_tile_id * Br + row_in_warp8;

      const float dp_sum_row0 =
          ffpa::bwd::dot_do_o_row_4lane<kDataType, kHeadDim>(dO, O, q_gmem_offset, q_row0, Nq);
      const float dp_sum_row8 =
          ffpa::bwd::dot_do_o_row_4lane<kDataType, kHeadDim>(dO, O, q_gmem_offset, q_row8, Nq);

      if constexpr (kStageQK > 1) {
        ffpa::bwd::cp_async_phase1_persistent_qdo_g2s<kDataType, Br, Q_tile_size, kHeadDim,
                                                      kMmaAtomK, kNumThreads, kPadQ>(
            smem_Q_base_ptr, smem_dO_base_ptr, Q, dO, q_gmem_offset, Q_tile_id, 0, 0, Nq);
        ffpa::cp_async::wait_group<kPhase1CpAsyncWaitGroups>();
        __syncthreads();
      }

#pragma unroll 1
      for (int tile_d = 0; tile_d < kNumDSubtiles; ++tile_d) {
        const int smem_sel = tile_d % kStageQK;
        if constexpr (kStageQK > 1) {
          const int next_tile_d = tile_d + 1;
          const int smem_sel_next = next_tile_d % kStageQK;
          if (next_tile_d < kNumDSubtiles) {
            ffpa::bwd::cp_async_phase1_persistent_qdo_g2s<kDataType, Br, Q_tile_size, kHeadDim,
                                                          kMmaAtomK, kNumThreads, kPadQ>(
                smem_Q_base_ptr, smem_dO_base_ptr, Q, dO, q_gmem_offset, Q_tile_id, next_tile_d,
                smem_sel_next, Nq);
          }
        } else {
          ffpa::bwd::cp_async_phase1_persistent_qdo_g2s<kDataType, Br, Q_tile_size, kHeadDim,
                                                        kMmaAtomK, kNumThreads, kPadQ>(
              smem_Q_base_ptr, smem_dO_base_ptr, Q, dO, q_gmem_offset, Q_tile_id, tile_d, smem_sel,
              Nq);
          ffpa::cp_async::wait_group<0>();
          __syncthreads();
        }

        ffpa::prefill::sync_fetch_qkv_frags_s2r<0, 4, Q_tile_size, kMmaAtomM, kMmaAtomN, kMmaAtomK,
                                                kPadQ, kDataType>(smem_Q_base_ptr, &R_Q[0], warp_QP,
                                                                  0, 0, smem_sel);
        ffpa::prefill::sync_fetch_qkv_frags_s2r<0, 4, Q_tile_size, kMmaAtomM, kMmaAtomN, kMmaAtomK,
                                                kPadQ, kDataType>(smem_dO_base_ptr, &R_dO[0],
                                                                  warp_QP, 0, 0, smem_sel);

#pragma unroll
        for (int kv_frag = 0; kv_frag < kValTileSeqLenK; ++kv_frag) {
          ffpa::prefill::sync_fetch_qkv_frags_s2r<0, 2, K_tile_size, kMmaAtomM, kMmaAtomN,
                                                  kMmaAtomK, kPadK, kDataType>(
              smem_K_base_ptr, &R_K[0], warp_KV, kv_frag, 0, tile_d);
          ffpa::prefill::sync_fetch_qkv_frags_s2r<0, 2, V_tile_size, kMmaAtomM, kMmaAtomN,
                                                  kMmaAtomK, kPadV, kDataType>(
              smem_V_base_ptr, &R_VK[0], warp_KV, kv_frag, 0, tile_d);
          ffpa::mma::m16n8k16_abf32<kDataType, ffpa::mma::MMAMode::kInplaceUpdate>(
              &R_S[kv_frag][0], &R_S[kv_frag][1], &R_S[kv_frag][2], &R_S[kv_frag][3], &R_Q[0],
              &R_Q[1], &R_Q[2], &R_Q[3], &R_K[0], &R_K[1]);
          ffpa::mma::m16n8k16_abf32<kDataType, ffpa::mma::MMAMode::kInplaceUpdate>(
              &R_dP[kv_frag][0], &R_dP[kv_frag][1], &R_dP[kv_frag][2], &R_dP[kv_frag][3], &R_dO[0],
              &R_dO[1], &R_dO[2], &R_dO[3], &R_VK[0], &R_VK[1]);
        }
        if constexpr (kStageQK > 1) {
          if (tile_d < (kNumDSubtiles - 1)) {
            ffpa::cp_async::wait_group<kPhase1CpAsyncWaitGroups>();
            __syncthreads();
          }
        } else {
          __syncthreads();
        }
      }

      const int kv_valid_local = Nkv - K_tile_id * Bc;
      if (kv_valid_local < Bc) {
        ffpa::prefill::sync_apply_kv_mask<kValTileSeqLenK, kMmaAccFloat32QK, kDataType>(
            &R_S[0][0], kv_valid_local);
      }
      if (causal) {
        ffpa::prefill::sync_apply_causal_mask<kValTileSeqLenK, kMmaAccFloat32QK, kDataType>(
            &R_S[0][0], warp_QP, Br_base, K_tile_id * Bc, kv_offset);
      }

      const int lse_offset = lse_batch_offset + q_head_idx * Nq;
      const float lse_row0 = (q_row0 < Nq) ? softmax_lse[lse_offset + q_row0] : INFINITY;
      const float lse_row8 = (q_row8 < Nq) ? softmax_lse[lse_offset + q_row8] : INFINITY;
      ffpa::bwd::compute_p_and_ds_from_lse<kDataType, kValTileSeqLenK>(
          R_S, R_dP, R_P, R_dS, lse_row0, lse_row8, dp_sum_row0, dp_sum_row8, scale);

      kDataType* warp_P_t_smem = P_t_smem + warp_QP * Transpose_tile_size;
      kDataType* warp_dS_t_smem = dS_t_smem + warp_QP * Transpose_tile_size;
      ffpa::bwd::store_packed_c_frag_transposed_to_smem<kDataType, kValTileSeqLenK,
                                                        kMmaAtomK + kTransposeScratchPad>(
          R_P, warp_P_t_smem);
      ffpa::bwd::store_packed_c_frag_transposed_to_smem<kDataType, kValTileSeqLenK,
                                                        kMmaAtomK + kTransposeScratchPad>(
          R_dS, warp_dS_t_smem);
      __syncthreads();
      const uint32_t smem_warp_P_t_base_ptr = __cvta_generic_to_shared(warp_P_t_smem);
      const uint32_t smem_warp_dS_t_base_ptr = __cvta_generic_to_shared(warp_dS_t_smem);

      if constexpr (kStageQK > 1) {
        ffpa::bwd::cp_async_phase2_persistent_qdo_g2s<kDataType, Br, Q_tile_size, kHeadDim,
                                                      kMmaAtomK, kNumThreads, kPadQ>(
            smem_Q_base_ptr, smem_dO_base_ptr, Q, dO, q_gmem_offset, Q_tile_id, 0, 0, Nq);
        ffpa::cp_async::wait_group<kPhase2CpAsyncWaitGroups>();
        __syncthreads();
      }

#pragma unroll 1
      for (int tile_d = 0; tile_d < kNumDSubtiles; ++tile_d) {
        const int smem_sel = tile_d % kStageQK;
        if constexpr (kStageQK > 1) {
          const int next_tile_d = tile_d + 1;
          const int smem_sel_next = next_tile_d % kStageQK;
          if (next_tile_d < kNumDSubtiles) {
            ffpa::bwd::cp_async_phase2_persistent_qdo_g2s<kDataType, Br, Q_tile_size, kHeadDim,
                                                          kMmaAtomK, kNumThreads, kPadQ>(
                smem_Q_base_ptr, smem_dO_base_ptr, Q, dO, q_gmem_offset, Q_tile_id, next_tile_d,
                smem_sel_next, Nq);
          }
        } else {
          ffpa::bwd::cp_async_phase2_persistent_qdo_g2s<kDataType, Br, Q_tile_size, kHeadDim,
                                                        kMmaAtomK, kNumThreads, kPadQ>(
              smem_Q_base_ptr, smem_dO_base_ptr, Q, dO, q_gmem_offset, Q_tile_id, tile_d, smem_sel,
              Nq);
          ffpa::cp_async::wait_group<0>();
          __syncthreads();
        }

#pragma unroll
        for (int d_subtile = 0; d_subtile < 2; ++d_subtile) {
          ffpa::utils::fill_1D_regs<uint32_t, 4>(R_dQ, 0);
#pragma unroll
          for (int kv_frag = 0; kv_frag < (Bc / kMmaAtomK); ++kv_frag) {
            ffpa::prefill::sync_fetch_qkv_frags_s2r<1, 2, K_tile_size, kMmaAtomM, kMmaAtomN,
                                                    kMmaAtomK, kPadK, kDataType>(
                smem_K_base_ptr, &R_K[0], warp_KV, d_subtile, kv_frag, tile_d);
            const int ds_offset = kv_frag * 2;
            ffpa::mma::m16n8k16_abf32<kDataType, ffpa::mma::MMAMode::kInplaceUpdate>(
                &R_dQ[0], &R_dQ[1], &R_dQ[2], &R_dQ[3], &R_dS[ds_offset][0], &R_dS[ds_offset][1],
                &R_dS[ds_offset + 1][0], &R_dS[ds_offset + 1][1], &R_K[0], &R_K[1]);
          }
          const int d_col_base = tile_d * kMmaAtomK + d_subtile * kMmaAtomN;
          ffpa::bwd::atomic_add_c_frag_to_dq<kDataType, kHeadDim>(
              R_dQ, dQ, q_gmem_offset, Q_tile_id, warp_QP, d_col_base, Nq, scale);

          ffpa::prefill::sync_fetch_qkv_frags_s2r<1, 2, Q_tile_size, kMmaAtomM, kMmaAtomN,
                                                  kMmaAtomK, kPadQ, kDataType>(
              smem_Q_base_ptr, &R_QB[0], warp_KV, d_subtile, warp_QP, smem_sel);
          ffpa::prefill::sync_fetch_qkv_frags_s2r<1, 2, Q_tile_size, kMmaAtomM, kMmaAtomN,
                                                  kMmaAtomK, kPadQ, kDataType>(
              smem_dO_base_ptr, &R_dOB[0], warp_KV, d_subtile, warp_QP, smem_sel);
#pragma unroll
          for (int kv_frag = 0; kv_frag < (Bc / kMmaAtomK); ++kv_frag) {
            ffpa::utils::fill_1D_regs<uint32_t, 4>(R_dK, 0);
            ffpa::utils::fill_1D_regs<uint32_t, 4>(R_dV, 0);
            ffpa::prefill::sync_fetch_qkv_frags_s2r<0, 4, Transpose_tile_size, kMmaAtomM, kMmaAtomN,
                                                    kMmaAtomK, kTransposeScratchPad, kDataType>(
                smem_warp_dS_t_base_ptr, &R_dS_T[0], kv_frag, 0, 0, 0);
            ffpa::prefill::sync_fetch_qkv_frags_s2r<0, 4, Transpose_tile_size, kMmaAtomM, kMmaAtomN,
                                                    kMmaAtomK, kTransposeScratchPad, kDataType>(
                smem_warp_P_t_base_ptr, &R_P_T[0], kv_frag, 0, 0, 0);
            ffpa::mma::m16n8k16_abf32<kDataType, ffpa::mma::MMAMode::kInplaceUpdate>(
                &R_dK[0], &R_dK[1], &R_dK[2], &R_dK[3], &R_dS_T[0], &R_dS_T[1], &R_dS_T[2],
                &R_dS_T[3], &R_QB[0], &R_QB[1]);
            ffpa::mma::m16n8k16_abf32<kDataType, ffpa::mma::MMAMode::kInplaceUpdate>(
                &R_dV[0], &R_dV[1], &R_dV[2], &R_dV[3], &R_P_T[0], &R_P_T[1], &R_P_T[2], &R_P_T[3],
                &R_dOB[0], &R_dOB[1]);
            ffpa::bwd::atomic_add_c_frag_to_dkv_tile<kDataType, kHeadDim, Bc>(
                R_dK, dK, kv_gmem_offset, K_tile_id, kv_frag, d_col_base, Nkv, scale);
            ffpa::bwd::atomic_add_c_frag_to_dkv_tile<kDataType, kHeadDim, Bc>(
                R_dV, dV, kv_gmem_offset, K_tile_id, kv_frag, d_col_base, Nkv, 1.0f);
          }
        }
        if constexpr (kStageQK > 1) {
          if (tile_d < (kNumDSubtiles - 1)) {
            ffpa::cp_async::wait_group<kPhase2CpAsyncWaitGroups>();
            __syncthreads();
          }
        } else {
          __syncthreads();
        }
      }
    }
  }
}

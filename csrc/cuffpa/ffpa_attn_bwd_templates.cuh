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

#include "cuffpa/prefill.cuh"

template <typename kDataType, const int kHeadDim, const int kMmaAtomM, const int kMmaAtomN,
          const int kMmaAtomK, const int kMmaTileSeqLenQ, const int kMmaTileSeqLenK,
          const int kMmaTileSeqLenP, const int kMmaTileHeadDimV, const int kValTileSeqLenQ,
          const int kValTileSeqLenK, const int kValTileSeqLenP, const int kValTileHeadDimV,
          const int kMmaAccFloat32QK, const int kMmaAccFloat32PV, const int kOStorageAccFloat32,
          const int kPrefetchQK, const int kPrefetchPV, const int kShareSmemQKV,
          const int kPersistQs2r, const int kPersistQg2s, const int kRegPipeKV, const int kStageQK,
          const int kStagePV, const int kPadQ, const int kPadK, const int kPadV>
__global__ void ffpa_stages_split_q_large_d_bwd_template(const kDataType*, const kDataType*,
                                                         const kDataType*, const float*,
                                                         const kDataType*, const kDataType*,
                                                         kDataType*, kDataType*, kDataType*,
                                                         const int, const int, const int, const int,
                                                         const float, const int, const int) {
  // NOT YET IMPLEMENTED.  See algorithm documentation in the file header.
  // The SDPA backward delegation path (backward_backend="sdpa") is used
  // as the default and only supported option for now.
  //
  // Note: we use sizeof(kDataType) == 0 rather than static_assert(false)
  // because NVCC evaluates static_assert(false) at template *definition*
  // time (not instantiation time), which would break every TU that
  // includes this header via launch_templates.cuh.  The dependent
  // expression sizeof(kDataType) == 0 is deferred to instantiation and
  // only fires if the backward template is actually instantiated.
  static_assert(sizeof(kDataType) == 0,
                "ffpa_stages_split_q_large_d_bwd_template is not yet implemented. "
                "Use backward_backend='sdpa' instead.");
}

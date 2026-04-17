#pragma once
#include <climits>                  // INT_MAX
#include "cuffpa/mma.cuh"           // ffpa::mma
#include "cuffpa/dtype_traits.cuh"  // ffpa::DtypeTraits
#include "cuffpa/warp.cuh"          // ffpa::warp
#include "cuffpa/swizzle.cuh"       // ffpa::swizzle
#include "cuffpa/cp_async.cuh"      // ffpa::cp_async
#include "cuffpa/utils.cuh"         // ffpa::utils

namespace ffpa {
namespace prefill {
// prefill utils: prefetch/load QKV g2s funcs, rescale/softmax funcs etc.

// Compile-time sanity checks for the large-d (D > ~128) FFPA kernel
// template. All invariants are enforced via static_assert so a bad config
// fails at NVCC time instead of silently producing wrong numerics. See
// ``ffpa_stages_split_q_large_d_template`` for the full parameter contract.
template <const int kHeadDim, const int kMmaAtomM, const int kMmaAtomN, const int kMmaAtomK,
          const int kMmaTileSeqLenQ, const int kMmaTileSeqLenK, const int kMmaTileSeqLenP,
          const int kMmaTileHeadDimV, const int kValTileSeqLenQ, const int kValTileSeqLenK,
          const int kValTileSeqLenP, const int kValTileHeadDimV, const int kMmaAccFloat32QK,
          const int kMmaAccFloat32PV, const int kOStorageAccFloat32, const int kPrefetchQK,
          const int kPrefetchPV, const int kShareSmemQKV, const int kPersistQs2r,
          const int kPersistQg2s, const int kRegPipeKV, const int kStageQK, const int kStagePV,
          const int kPadQ, const int kPadK, const int kPadV>
__device__ __forceinline__ void check_large_d_compiling_states() {
  // Matmul Layout: Q[Br,d]@K^T[d,Bc] NT, P[Br,Bc]@V[Bc,d] NN.
  // NOTE: K[Bc,d] with row major means K^T[d,Bc] in col major.
  static_assert(kMmaAtomM == 16 && kMmaAtomN == 8 && kMmaAtomK == 16);  // m16n8k16
  static_assert(kMmaTileSeqLenQ <= 8 && kMmaTileSeqLenK == 1);          // Q@K^T
  static_assert(kMmaTileSeqLenP <= 8 && kMmaTileHeadDimV == 1);         // P@V
  static_assert(kValTileSeqLenQ == 1 && kValTileSeqLenK <= 16);         // Q@K^T
  static_assert(kValTileSeqLenP == 1 &&
                kValTileHeadDimV == (kHeadDim / (kMmaAtomN * kMmaTileHeadDimV)));  // P@V
  static_assert(kMmaAccFloat32QK == 0 || kMmaAccFloat32QK == 1);
  static_assert(kMmaAccFloat32PV == 0 || kMmaAccFloat32PV == 1);
  static_assert(kOStorageAccFloat32 == 0 || kOStorageAccFloat32 == 1);
  // Make sure that Br >= Bc, for shared memory reuse.
  static_assert((kMmaAtomM * kMmaTileSeqLenQ * kValTileSeqLenQ) >=
                (kMmaAtomN * kMmaTileSeqLenK * kValTileSeqLenK));
  static_assert(kPrefetchQK == 0 || kPrefetchQK == 1);
  static_assert(kPrefetchPV == 0 || kPrefetchPV == 1);
  static_assert(kShareSmemQKV == 0 || kShareSmemQKV == 1);
  // Persist load Q s2r for headdim < 512, more registers, but still keep O(1) SRAM.
  static_assert(kPersistQs2r == 0 || kPersistQs2r == 1);
  // Persist load Q g2s for headdim < 512, more SRAM, but still keep register usage.
  static_assert(kPersistQg2s == 0 || kPersistQg2s == 1);
  // kPersistQg2s and kPersistQs2r can not both enabled for large d kernel.
  static_assert((kPersistQg2s & kPersistQs2r) == 0);
  // kPersistQg2s and kShareSmemQKV can not both enabled for large d kernel..
  static_assert((kPersistQg2s & kShareSmemQKV) == 0);
  // Registers Ping pong double buffers for ldmatrix s2r & mma computation overlapping.
  static_assert(kRegPipeKV == 0 || kRegPipeKV == 1);
  // May apply different multi stages policy for QK and V.
  static_assert(kStageQK < 5 && kStageQK > 0);  // QK (<=4)
  static_assert(kStagePV < 5 && kStagePV > 0);  // V  (<=4)
  static_assert(kPadQ >= 0 && kPadQ % 8 == 0);  // 0,8,16
  static_assert(kPadK >= 0 && kPadK % 8 == 0);  // 0,8,16
  static_assert(kPadV >= 0 && kPadV % 8 == 0);  // 0,8,16
}

// Compile-time sanity checks for the small-d (D <= 256) FFPA kernel
// template. Enforces the FA-2-style small-d invariants (e.g. kStageQK ==
// kStagePV == 1, Br >= Bc, kRegPipeKV and kPersistVs2r mutually exclusive).
// See ``ffpa_stages_split_q_small_d_template`` for the full parameter
// contract.
template <const int kHeadDim, const int kMmaAtomM, const int kMmaAtomN, const int kMmaAtomK,
          const int kMmaTileSeqLenQ, const int kMmaTileSeqLenK, const int kMmaTileSeqLenP,
          const int kMmaTileHeadDimV, const int kValTileSeqLenQ, const int kValTileSeqLenK,
          const int kValTileSeqLenP, const int kValTileHeadDimV, const int kMmaAccFloat32QK,
          const int kMmaAccFloat32PV, const int kOStorageAccFloat32, const int kPrefetchQK,
          const int kPrefetchPV, const int kShareSmemQKV, const int kPersistQs2r,
          const int kPersistVs2r, const int kRegPipeKV, const int kStageQK, const int kStagePV,
          const int kPadQ, const int kPadK, const int kPadV>
__device__ __forceinline__ void check_small_d_compiling_states() {
  // Matmul Layout: Q[Br,d]@K^T[d,Bc] NT, P[Br,Bc]@V[Bc,d] NN.
  // NOTE: K[Bc,d] with row major means K^T[d,Bc] in col major.
  static_assert(kMmaAtomM == 16 && kMmaAtomN == 8 && kMmaAtomK == 16);  // m16n8k16
  static_assert(kMmaTileSeqLenQ <= 8 && kMmaTileSeqLenK == 1);          // Q@K^T
  static_assert(kMmaTileSeqLenP <= 8 && kMmaTileHeadDimV == 1);         // P@V
  static_assert(kValTileSeqLenQ == 1 && kValTileSeqLenK <= 16);         // Q@K^T
  static_assert(kValTileSeqLenP == 1 &&
                kValTileHeadDimV == (kHeadDim / (kMmaAtomN * kMmaTileHeadDimV)));  // P@V
  static_assert(kMmaAccFloat32QK == 0 || kMmaAccFloat32QK == 1);
  static_assert(kMmaAccFloat32PV == 0 || kMmaAccFloat32PV == 1);
  static_assert(kOStorageAccFloat32 == 0 || kOStorageAccFloat32 == 1);
  // Make sure that Br >= Bc, for shared memory reuse.
  static_assert((kMmaAtomM * kMmaTileSeqLenQ * kValTileSeqLenQ) >=
                (kMmaAtomN * kMmaTileSeqLenK * kValTileSeqLenK));
  static_assert(kPrefetchQK == 0 || kPrefetchQK == 1);
  static_assert(kPrefetchPV == 0 || kPrefetchPV == 1);
  static_assert(kShareSmemQKV == 0 || kShareSmemQKV == 1);
  // Persist load Q s2r for headdim <= 128, more registers.
  static_assert(kPersistQs2r == 0 || kPersistQs2r == 1);
  // Persist load V s2r for headdim <= 128, more registers.
  static_assert(kPersistVs2r == 0 || kPersistVs2r == 1);
  if constexpr (kShareSmemQKV) {
    // kPersistQs2r must be enabled is set kShareSmemQKV as 1
    static_assert(kPersistQs2r == 1);
  }
  // Registers Ping pong double buffers for ldmatrix s2r & mma
  // computation overlapping.
  static_assert(kRegPipeKV == 0 || kRegPipeKV == 1);
  // kRegPipeKV and kPersistVs2r can not both enabled.
  static_assert((kRegPipeKV & kPersistVs2r) == 0);
  // May apply different multi stages policy for QK and V.
  static_assert(kStageQK < 5 && kStageQK > 0);  // QK (<=4)
  static_assert(kStagePV < 5 && kStagePV > 0);  // V  (<=4)
  static_assert(kPadQ >= 0 && kPadQ % 8 == 0);  // 0,8,16
  static_assert(kPadK >= 0 && kPadK % 8 == 0);  // 0,8,16
  static_assert(kPadV >= 0 && kPadV % 8 == 0);  // 0,8,16
}

// Async g2s (global -> shared) copy for one BrOrBc x kMmaAtomK tile of
// Q / K / V. Uses cp.async.ca.shared.global 16B transactions, with SMEM
// destination addressing chosen by ``kPad`` (swizzle when kPad == 0,
// padded layout otherwise). Rows whose global index >= ``seqlen_bound``
// are zero-filled via cp.async src-size=0 so that tail tiles with
// N % Br != 0 or N % Bc != 0 stay correct without extra branching.
//
// - ``BrOrBc``    tile rows in this call (Br for Q, Bc for K/V).
// - ``kTileSize`` byte-stride between two cp.async stages in SMEM.
// - ``kHeadDim``  global head-dim; ``d_tile_id >= kHeadDim/kMmaAtomK``
//                 early-returns so callers can issue fixed-length loops.
// - ``n_tile_id`` seqlen tile index (Q_tile_id for Q, tile_K_seqlen for K/V).
// - ``d_tile_id`` head-dim sub-tile index; clamped via the early return.
// - ``stage``     cp.async pipeline stage (multi-buffer ring index).
template <const int BrOrBc, const int kTileSize, const int kHeadDim, const int kMmaAtomK,
          const int kNumThreads, const int kPad, typename kDataType = __half>
__device__ __forceinline__ void cp_async_qkv_g2s(
    uint32_t smem_base_ptr,     // QKV smem base ptr
    const kDataType* gmem_ptr,  // QKV gmem ptr
    const int gmem_offset,      // QKV gmem_offset
    const int n_tile_id,        // seqlen offset, Q_tile_id * Br, tile_K_seqlen * Bc
    const int d_tile_id,        // headdim offset, tile_K_d * kMmaAtomK, tile_V_d * kMmaAtomN * 2
    const int stage,            // stage * QKV tile_size
    const int seqlen_bound = INT_MAX  // bound on the seqlen axis; rows with
                                      // global idx >= seqlen_bound are
                                      // zero-filled via cp.async src-size=0
) {
  // QK: tile_K_d < (kHeadDim / kMmaAtomK)
  //  V: tile_V_d < (kHeadDim / kMmaAtomN * 2)
  if (d_tile_id >= (kHeadDim / kMmaAtomK)) {
    return;
  }
  const int tid = threadIdx.x;  // within block
  constexpr bool kSwizzle = (kPad == 0) ? true : false;
  // Mapping QKV tid -> smem, tile size [64/128, 16]
  // Br 64, tid / 2, row 0~64
  const int load_smem_BrOrBc = (tid / (kNumThreads / BrOrBc));
  // (tid % 2) * 8, 0,8,...
  const int load_smem_d = (tid % (kNumThreads / BrOrBc)) * (kMmaAtomK / (kNumThreads / BrOrBc));
  // Mapping QKV tid -> gmem, tile size [64/128, 16], row offset by
  // n_tile_id(seqlen), col offset by d_tile_id(Headdim).
  const int load_gmem_BrOrBc = (n_tile_id * BrOrBc) + load_smem_BrOrBc;
  const int load_gmem_d = (d_tile_id * kMmaAtomK) + load_smem_d;  // 0,8
  // Offset by QKV global gmem_offset.
  const int load_gmem_addr = (gmem_offset + load_gmem_BrOrBc * kHeadDim + load_gmem_d);
  // Seqlen boundary predicate: rows beyond QKV_seqlen must be zero-filled.
  const bool row_valid = (load_gmem_BrOrBc < seqlen_bound);

// cp async & apply swizzle or padding.
#pragma unroll
  for (int i = 0; i < (kMmaAtomK / (kNumThreads / BrOrBc)); i += 8) {
    const uint32_t load_smem_ptr =
        (smem_base_ptr +
         (stage * kTileSize + load_smem_BrOrBc * (kMmaAtomK + kPad) +
          (kSwizzle ? swizzle::permuted<kMmaAtomK>(load_smem_BrOrBc, load_smem_d + i)
                    : load_smem_d + i)) *
             sizeof(kDataType));
    cp_async::cp_async_zfill<16>(load_smem_ptr, &(gmem_ptr[load_gmem_addr + i]), row_valid);
  }
  // cp_async::commit_group();
}

// Sync s2r (shared -> register) load of one MMA fragment of Q, K, or V
// using ``ldmatrix.sync``. Selects ``ldmatrix.m8n8.x2``, ``x2.trans``
// (for V, emits col-major fragments consumed by P@V), or ``x4`` (for Q)
// based on ``kTrans`` / ``kNumRegs``. SMEM addressing mirrors the g2s
// layout (swizzle when kPad == 0, else padded).
template <const int kTrans, const int kNumRegs, const int kTileSize, const int kMmaAtomM,
          const int kMmaAtomN, const int kMmaAtomK, const int kPad, typename kDataType = __half>
__device__ __forceinline__ void sync_fetch_qkv_frags_s2r(
    uint32_t smem_base_ptr,  // QKV smem base ptr
    uint32_t* R,             // Register ptr, R_QKV
    const int mma_tile_id,   // Q warp_QP 0~num MMAs, KV warp_KV 0
    const int warp_tile_id,  // Q 0, KV 0~kValTileSeqLenK
    const int n_tile_id,     // seqlen QK 0, V tile_V_Bc
    const int stage) {
  const int lane_id = threadIdx.x % WARP_SIZE;  // 0~31
  constexpr bool kSwizzle = (kPad == 0) ? true : false;
  if constexpr (kTrans) {
    // load V m8n8x2 via ldmatrix.x2.trans
    static_assert(kNumRegs == 2);
    // mma_tile_id = warp_KV == 0, warp_tile_id = (j % 2), n_tile_id = tile_V_Bc
    // warp_smem_V_d  = warp_KV * (kMmaAtomN * kValTileHeadDimV) + (j % 2) * kMmaAtomN;
    const int warp_smem_d = warp_tile_id * kMmaAtomN;
    const int lane_smem_Bc = n_tile_id * kMmaAtomK + lane_id % 16;
    const int lane_smem_d = warp_smem_d;  // 0,8
    const uint32_t lane_smem_ptr =
        (smem_base_ptr +
         (stage * kTileSize + lane_smem_Bc * (kMmaAtomN * 2 + kPad) +
          (kSwizzle ? swizzle::permuted<kMmaAtomN * 2>(lane_smem_Bc, lane_smem_d) : lane_smem_d)) *
             sizeof(kDataType));
    mma::ldmatrix_m8n8x2_trans(&R[0], &R[1], lane_smem_ptr);
  } else {
    static_assert(kNumRegs == 2 || kNumRegs == 4);
    if constexpr (kNumRegs == 4) {
      // load Q m8n8x4 via ldmatrix.x4
      // mma_tile_id = warp_QP, kValTileSeqLenQ=1
      // warp_smem_Q_Br = warp_QP * (kMmaAtomM * kValTileSeqLenQ) + 0 * kMmaAtomM
      const int warp_smem_Br = mma_tile_id * (kMmaAtomM);
      const int lane_smem_Br = warp_smem_Br + lane_id % 16;  // 0~15
      const int lane_smem_d = (lane_id / 16) * 8;            // 0,8
      const uint32_t lane_smem_ptr =
          (smem_base_ptr +
           (stage * kTileSize + lane_smem_Br * (kMmaAtomK + kPad) +
            (kSwizzle ? swizzle::permuted<kMmaAtomK>(lane_smem_Br, lane_smem_d) : lane_smem_d)) *
               sizeof(kDataType));
      mma::ldmatrix_m8n8x4(&R[0], &R[1], &R[2], &R[3], lane_smem_ptr);
    } else {
      // load K m8n8x2 via ldmatrix.x2
      // mma_tile_id = warp_KV == 0, warp_tile_id = j
      // warp_smem_Bc = warp_KV * (kMmaAtomN * kValTileSeqLenK) + j * kMmaAtomN;
      const int warp_smem_Bc = warp_tile_id * kMmaAtomN;
      const int lane_smem_Bc = warp_smem_Bc + lane_id % 8;  // 0~7
      const int lane_smem_d = ((lane_id / 8) % 2) * 8;      // 0,8
      const uint32_t lane_smem_ptr =
          (smem_base_ptr +
           (stage * kTileSize + lane_smem_Bc * (kMmaAtomK + kPad) +
            (kSwizzle ? swizzle::permuted<kMmaAtomK>(lane_smem_Bc, lane_smem_d) : lane_smem_d)) *
               sizeof(kDataType));
      mma::ldmatrix_m8n8x2(&R[0], &R[1], lane_smem_ptr);
    }
  }
}

// Apply a -inf mask to R_S fragments whose local KV column falls beyond the
// valid KV length of the current tile. Used on the last KV tile when
// ``QKV_seqlen % Bc != 0`` so that online safe-softmax treats padding
// columns as logits of -inf (exp(-inf)=0), making them invisible to both
// row_max and row_sum without changing the main-path performance.
//
// The fragment-to-column mapping follows the ``m16n8k16`` C-layout:
// for j in [0, kValTileSeqLenK), R_S[j] covers local columns
// ``j*8 + (lane_id%4)*2 + {0, 1}`` (and two rows offset by 0 / 8).
// With ``kMmaTileSeqLenK == 1`` the warp_KV base is always 0.
template <const int kValTileSeqLenK, const int kMmaAccFloat32, typename kDataType = __half>
__device__ __forceinline__ void sync_apply_kv_mask(
    uint32_t* R_S,             // &R_S[0][0][0]
    const int kv_valid_local)  // valid KV columns inside this Bc-tile, in (0, Bc]
{
  using Traits = DtypeTraits<kDataType>;
  const int lane_id = threadIdx.x % WARP_SIZE;
  const int col_base = (lane_id % 4) * 2;  // local col within 8-col fragment
  if constexpr (kMmaAccFloat32) {
#pragma unroll
    for (int j = 0; j < kValTileSeqLenK; ++j) {
      float* t_fptr = reinterpret_cast<float*>(R_S + j * 4);
      const int c0 = j * 8 + col_base;
      const int c1 = c0 + 1;
      if (c0 >= kv_valid_local) {
        t_fptr[0] = -INFINITY;
        t_fptr[2] = -INFINITY;
      }
      if (c1 >= kv_valid_local) {
        t_fptr[1] = -INFINITY;
        t_fptr[3] = -INFINITY;
      }
    }
  } else {
    static_assert(std::is_same_v<kDataType, __half>,
                  "MMA Acc F16 mask path is only valid for __half activation dtype.");
    // fp16 acc: 4 half values packed into 2 uint32; -inf in fp16 = 0xFC00.
    const kDataType neg_inf = Traits::from_float(-INFINITY);
#pragma unroll
    for (int j = 0; j < kValTileSeqLenK; ++j) {
      kDataType* t_hptr = reinterpret_cast<kDataType*>(R_S + j * 2);
      const int c0 = j * 8 + col_base;
      const int c1 = c0 + 1;
      if (c0 >= kv_valid_local) {
        t_hptr[0] = neg_inf;
        t_hptr[2] = neg_inf;
      }
      if (c1 >= kv_valid_local) {
        t_hptr[1] = neg_inf;
        t_hptr[3] = neg_inf;
      }
    }
  }
}

// Apply a causal-attention -inf mask to R_S fragments whose local KV
// column falls strictly above the allowed causal boundary for the
// fragment's global Q row. Called only on tiles that straddle or lie
// beyond the causal diagonal (selected by the launcher) so no overhead
// is paid on non-causal paths.
//
// Convention: queries are aligned to the *tail* of the KV sequence so
// that global q position in KV space is ``q_pos = row + kv_offset``
// with ``kv_offset = Nkv - Nq >= 0``. A key is visible iff
// ``k_pos <= q_pos``. This matches the standard decoding-style causal
// mask and degenerates to the usual triangular mask when ``Nkv == Nq``.
//
// The fragment-to-(row,col) mapping follows the ``m16n8k16`` C-layout
// (same mapping used by ``sync_apply_kv_mask``):
//   row0 = warp_QP*16 + lane_id/4,   row8 = row0 + 8
//   col_base = (lane_id % 4) * 2,    per-j cols in [j*8+col_base, j*8+col_base+1]
// Each thread owns four values per fragment j: {row0,c0}, {row0,c1},
// {row8,c0}, {row8,c1}; they may share a single row threshold when the
// tile is fully below the diagonal for one sub-row but not the other.
template <const int kValTileSeqLenK, const int kMmaAccFloat32, typename kDataType = __half>
__device__ __forceinline__ void sync_apply_causal_mask(
    uint32_t* R_S,        // &R_S[0][0][0]
    const int warp_QP,    // warp row index (matches kernel state)
    const int Br_base,    // Q_tile_id * Br (global Q row of tile's row 0)
    const int Bc_base,    // tile_K_seqlen * Bc (global KV col of tile's col 0)
    const int kv_offset)  // Nkv - Nq (>= 0 enforced at launch time)
{
  using Traits = DtypeTraits<kDataType>;
  const int lane_id = threadIdx.x % WARP_SIZE;
  const int row_base = warp_QP * 16 + (lane_id / 4);
  const int col_base = (lane_id % 4) * 2;
  const int thresh_row0 = (Br_base + row_base) + kv_offset;
  const int thresh_row8 = thresh_row0 + 8;
  if constexpr (kMmaAccFloat32) {
#pragma unroll
    for (int j = 0; j < kValTileSeqLenK; ++j) {
      float* t_fptr = reinterpret_cast<float*>(R_S + j * 4);
      const int k0 = Bc_base + j * 8 + col_base;
      const int k1 = k0 + 1;
      if (k0 > thresh_row0)
        t_fptr[0] = -INFINITY;
      if (k1 > thresh_row0)
        t_fptr[1] = -INFINITY;
      if (k0 > thresh_row8)
        t_fptr[2] = -INFINITY;
      if (k1 > thresh_row8)
        t_fptr[3] = -INFINITY;
    }
  } else {
    static_assert(std::is_same_v<kDataType, __half>,
                  "MMA Acc F16 causal mask path is only valid for __half activation dtype.");
    const kDataType neg_inf = Traits::from_float(-INFINITY);
#pragma unroll
    for (int j = 0; j < kValTileSeqLenK; ++j) {
      kDataType* t_hptr = reinterpret_cast<kDataType*>(R_S + j * 2);
      const int k0 = Bc_base + j * 8 + col_base;
      const int k1 = k0 + 1;
      if (k0 > thresh_row0)
        t_hptr[0] = neg_inf;
      if (k1 > thresh_row0)
        t_hptr[1] = neg_inf;
      if (k0 > thresh_row8)
        t_hptr[2] = neg_inf;
      if (k1 > thresh_row8)
        t_hptr[3] = neg_inf;
    }
  }
}

// Online safe-softmax step for one [Br, Bc] S fragment (FA-2 paper algo 1).
//   1. warp-reduces a per-row max across the Bc axis.
//   2. ``m_new = max(m_old, m_new)`` for numerical stability.
//   3. ``P = exp(S * scale - m_new)`` written back into R_S in the
//      activation dtype so R_S can be reused as P for the next P@V MMA.
//   4. warp-reduces the per-row exp-sum into ``lane_row_sum_new``.
// Caller follows up with ``sync_precompute_rescale_factors`` +
// ``sync_update_max_expsum`` to fold the new (m, l) into the running
// (m_old, l_old). The fp32 accumulator path works for both fp16 and
// bf16 activations; the fp16-acc fast path requires kDataType == __half.
template <const int kValTileSeqLenK, const int kMmaAccFloat32, typename kDataType = __half>
__device__ __forceinline__ void sync_online_safe_softmax(
    uint32_t* R_S,                  // &R_S[0][0][0]
    const float scale,              // 1 / sqrt(d)
    float* lane_row_max_new,        // &lane_row_max_new[0][0]
    float* lane_row_sum_new,        // &lane_row_sum_new[0][0]
    float* lane_block_row_max_old,  // &lane_block_row_max_old[0][0]
    float* lane_block_row_sum_old   // &lane_block_row_sum_old[0][0]
) {
  using Traits = DtypeTraits<kDataType>;
  if constexpr (kMmaAccFloat32) {
    // Row max for [Br,Bc] tile, Thread -> Warp -> Block.
    {  // kValTileSeqLenQ = 1
// Thread level reduce max across kValTileSeqLenK dim, namely Bc.
#pragma unroll
      for (int j = 0; j < kValTileSeqLenK; ++j) {
        const float* t_fptr_S_0_1 = reinterpret_cast<float*>(R_S + j * 4);  // &R_S[0][j][0]
        // This should be the row max after S = (Q @ K^T) / sqrt(d)
        const float tmp_max_0 = max(t_fptr_S_0_1[0], t_fptr_S_0_1[1]) * scale;
        const float tmp_max_1 = max(t_fptr_S_0_1[2], t_fptr_S_0_1[3]) * scale;
        lane_row_max_new[0] = max(lane_row_max_new[0], tmp_max_0);
        lane_row_max_new[1] = max(lane_row_max_new[1], tmp_max_1);
      }  // end for kValTileSeqLenK

      // Warp level reduce max, warp_size = 4
      // Each thread contains the maximum of 2 rows of Br,
      // and only the values of T0, T4, ..., T28 are used.
      lane_row_max_new[0] = warp::reduce_max<float, 4>(lane_row_max_new[0]);
      lane_row_max_new[1] = warp::reduce_max<float, 4>(lane_row_max_new[1]);
    }  // end for kValTileSeqLenQ

    // static_assert(kValTileSeqLenQ == 1);
    // Exp sum and mul scale_factor for [Br,Bc] tile, Thread -> Warp -> Block.
    {  // kValTileSeqLenQ = 1
      // Use latest global row max without update.
      // Br 0, row_id, 0~7,  16~23, 32~39, 48~55;
      // Br 1, row_id, 8~15, 24~31, 40~47, 56~63;
      // Apply m_new = max(m_old, m_new) here.
      const float block_row_max_new_0 = max(lane_block_row_max_old[0], lane_row_max_new[0]);
      const float block_row_max_new_1 = max(lane_block_row_max_old[1], lane_row_max_new[1]);

#pragma unroll
      for (int j = 0; j < kValTileSeqLenK; ++j) {
        // R_S[][][4] 4 32bit registers with each contains 1 F32 element.
        // (x,y) 0~7->{c0, c1}, (z,w)->8~15 {c2, c3}
        float* t_fptr_S_0_1 = reinterpret_cast<float*>(R_S + j * 4);
        kDataType* t_hptr_S_0_1 = reinterpret_cast<kDataType*>(R_S + j * 4);
        // P = Exp(S - m_new), fmaf(x, y, z) = x * y + z in registers;
        t_fptr_S_0_1[0] = __expf(__fmaf_rn(t_fptr_S_0_1[0], scale, -block_row_max_new_0));
        t_fptr_S_0_1[1] = __expf(__fmaf_rn(t_fptr_S_0_1[1], scale, -block_row_max_new_0));
        t_fptr_S_0_1[2] = __expf(__fmaf_rn(t_fptr_S_0_1[2], scale, -block_row_max_new_1));
        t_fptr_S_0_1[3] = __expf(__fmaf_rn(t_fptr_S_0_1[3], scale, -block_row_max_new_1));
        lane_row_sum_new[0] += (t_fptr_S_0_1[0] + t_fptr_S_0_1[1]);
        lane_row_sum_new[1] += (t_fptr_S_0_1[2] + t_fptr_S_0_1[3]);
        // Update R_S for P[Br,Bc] = Exp(S-m), point wise.
        // Also convert F32 -> kDataType for P@V MMA, reuse R_S as P.
        t_hptr_S_0_1[0] = Traits::from_float(t_fptr_S_0_1[0]);
        t_hptr_S_0_1[1] = Traits::from_float(t_fptr_S_0_1[1]);
        t_hptr_S_0_1[2] = Traits::from_float(t_fptr_S_0_1[2]);
        t_hptr_S_0_1[3] = Traits::from_float(t_fptr_S_0_1[3]);
      }  // end for kValTileSeqLenK

      // Warp level reduce sum, warp_size = 4
      lane_row_sum_new[0] = warp::reduce_sum<float, 4>(lane_row_sum_new[0]);
      lane_row_sum_new[1] = warp::reduce_sum<float, 4>(lane_row_sum_new[1]);
    }

  } else {
    // MMA Acc F16 (only valid when kDataType == __half; bf16 forces kMmaAccFloat32==1).
    static_assert(std::is_same_v<kDataType, __half>,
                  "MMA Acc F16 path is only valid for __half activation dtype.");
    // Row max for [Br,Bc] tile, Thread -> Warp -> Block.
    {  // kValTileSeqLenQ = 1
// Thread level reduce max across kValTileSeqLenK dim, namely Bc.
#pragma unroll
      for (int j = 0; j < kValTileSeqLenK; ++j) {
        const kDataType* t_hptr_S_0_1 = reinterpret_cast<kDataType*>(R_S + j * 2);
        // This should be the row max after S = (Q @ K^T) / sqrt(d)
        const float tmp_max_0 =
            Traits::to_float(Traits::hmax(t_hptr_S_0_1[0], t_hptr_S_0_1[1])) * scale;
        const float tmp_max_1 =
            Traits::to_float(Traits::hmax(t_hptr_S_0_1[2], t_hptr_S_0_1[3])) * scale;
        lane_row_max_new[0] = max(lane_row_max_new[0], tmp_max_0);
        lane_row_max_new[1] = max(lane_row_max_new[1], tmp_max_1);
      }  // end for kValTileSeqLenK

      // Warp level reduce max, warp_size = 4
      // Each thread contains the maximum of 2 rows of Br,
      // and only the values of T0, T4, ..., T28 are used.
      lane_row_max_new[0] = warp::reduce_max<float, 4>(lane_row_max_new[0]);
      lane_row_max_new[1] = warp::reduce_max<float, 4>(lane_row_max_new[1]);
    }  // end for kValTileSeqLenQ

    // static_assert(kValTileSeqLenQ == 1);
    // Exp sum and mul scale_factor for [Br,Bc] tile, Thread -> Warp -> Block.
    {  // kValTileSeqLenQ = 1
      // Apply m_new = max(m_old, m_new) here.
      const float block_row_max_new_0 = max(lane_block_row_max_old[0], lane_row_max_new[0]);
      const float block_row_max_new_1 = max(lane_block_row_max_old[1], lane_row_max_new[1]);

#pragma unroll
      for (int j = 0; j < kValTileSeqLenK; ++j) {
        kDataType* t_hptr_S_0_1 = reinterpret_cast<kDataType*>(R_S + j * 2);
        // P = Exp(S - m_new), fmaf(x, y, z) = x * y + z;
        float4 t_reg_S_0_1;
        t_reg_S_0_1.x =
            __expf(__fmaf_rn(Traits::to_float(t_hptr_S_0_1[0]), scale, -block_row_max_new_0));
        t_reg_S_0_1.y =
            __expf(__fmaf_rn(Traits::to_float(t_hptr_S_0_1[1]), scale, -block_row_max_new_0));
        t_reg_S_0_1.z =
            __expf(__fmaf_rn(Traits::to_float(t_hptr_S_0_1[2]), scale, -block_row_max_new_1));
        t_reg_S_0_1.w =
            __expf(__fmaf_rn(Traits::to_float(t_hptr_S_0_1[3]), scale, -block_row_max_new_1));
        lane_row_sum_new[0] += (t_reg_S_0_1.x + t_reg_S_0_1.y);
        lane_row_sum_new[1] += (t_reg_S_0_1.z + t_reg_S_0_1.w);
        // Update R_S for P[Br,Bc] = Exp(S-m), point wise.
        t_hptr_S_0_1[0] = Traits::from_float(t_reg_S_0_1.x);
        t_hptr_S_0_1[1] = Traits::from_float(t_reg_S_0_1.y);
        t_hptr_S_0_1[2] = Traits::from_float(t_reg_S_0_1.z);
        t_hptr_S_0_1[3] = Traits::from_float(t_reg_S_0_1.w);
      }  // end for kValTileSeqLenK

      // Warp level reduce sum, warp_size = 4
      lane_row_sum_new[0] = warp::reduce_sum<float, 4>(lane_row_sum_new[0]);
      lane_row_sum_new[1] = warp::reduce_sum<float, 4>(lane_row_sum_new[1]);
    }
  }
}

// Precompute O-rescaling factors ``exp(m_old - m_new)`` for the two
// sub-rows owned by this lane. Factored out of the softmax step so the
// tiling-O rescaling loop over head-dim can reuse them without repeating
// the ``__expf`` call. On the first KV tile (``n_tile_id == 0``) m_old
// is forced to m_new so the factor is 1 and O starts un-rescaled.
__device__ __forceinline__ void sync_precompute_rescale_factors(
    float* rescale_o_factor_0,            // rescale factor
    float* rescale_o_factor_1,            // rescale factor
    const float* lane_row_max_new,        // &lane_row_max_new[0][0]
    const float* lane_block_row_max_old,  // &lane_block_row_max_old[0][0]
    const int n_tile_id                   // tile_K_seqlen
) {
  float block_row_max_new_0 = lane_row_max_new[0];
  float block_row_max_new_1 = lane_row_max_new[1];
  float block_row_max_old_0 = lane_block_row_max_old[0];
  float block_row_max_old_1 = lane_block_row_max_old[1];
  // NOTE: max(-inf, val) = val.
  block_row_max_new_0 = max(block_row_max_old_0, block_row_max_new_0);
  block_row_max_new_1 = max(block_row_max_old_1, block_row_max_new_1);
  // Avoid inf value while using m_old for rescaling O.
  block_row_max_old_0 = (n_tile_id > 0 ? block_row_max_old_0 : block_row_max_new_0);
  block_row_max_old_1 = (n_tile_id > 0 ? block_row_max_old_1 : block_row_max_new_1);
  // Precompute rescale_o_factor_0 & rescale_o_factor_1, avoid redundant exp.
  rescale_o_factor_0[0] = __expf(block_row_max_old_0 - block_row_max_new_0);
  rescale_o_factor_1[0] = __expf(block_row_max_old_1 - block_row_max_new_1);
}

// Apply the rescale ``O_new = exp(m_old - m_new) * O_old + P @ V`` to
// one [Br, 8] head-dim slice of the running O accumulator. Called inside
// the head-dim loop of P@V so the whole O tensor is updated incrementally
// with the latest KV-tile contribution. Supports fp32 and fp16 O storage
// independently of the MMA accumulator dtype.
template <const int kOStorageAccFloat32, const int kMmaAccFloat32, typename kDataType = __half>
__device__ __forceinline__ void sync_rescaling_tiling_o(
    uint32_t* R_D,                    // &R_D[0][0][0]
    uint32_t* R_O,                    // &R_O[0]
    const float* rescale_o_factor_0,  // rescale factor
    const float* rescale_o_factor_1,  // rescale factor
    const int n_tile_id,              // tile_K_seqlen
    const int d_tile_id               // j
) {
  using Traits = DtypeTraits<kDataType>;
  // Now, we get [Br,8] slice of [Br,d], each warp(MMA) contains m16n8.
  // 0. Rescale O: Online rescaling O each tile_K_seqlen step, need m_new, m_old.
  // m = max(m_old, m_new), O_new[Br,d] = exp(m_old - m) * O_old + P@V
  // m = max(m_old, m_new), l = exp(m_old - m) * l_old + l_new (FA2 paper)
  if constexpr (kMmaAccFloat32) {
    const float* t_fptr_O_0_1 = reinterpret_cast<float*>(R_O);
    if constexpr (kOStorageAccFloat32) {
      // (x,y) 0~7->{c0, c1}, (z,w)->8~15 {c2, c3} kValTileSeqLenP=1
      float* t_fptr_D_0_1 = reinterpret_cast<float*>(R_D + d_tile_id * 4);  // &(R_D[0][j][0])
      t_fptr_D_0_1[0] = __fmaf_rn(rescale_o_factor_0[0], t_fptr_D_0_1[0], t_fptr_O_0_1[0]);
      t_fptr_D_0_1[1] = __fmaf_rn(rescale_o_factor_0[0], t_fptr_D_0_1[1], t_fptr_O_0_1[1]);
      t_fptr_D_0_1[2] = __fmaf_rn(rescale_o_factor_1[0], t_fptr_D_0_1[2], t_fptr_O_0_1[2]);
      t_fptr_D_0_1[3] = __fmaf_rn(rescale_o_factor_1[0], t_fptr_D_0_1[3], t_fptr_O_0_1[3]);
    } else {
      kDataType* t_hptr_D_0_1 = reinterpret_cast<kDataType*>(R_D + d_tile_id * 2);
      t_hptr_D_0_1[0] = Traits::from_float(
          __fmaf_rn(rescale_o_factor_0[0], Traits::to_float(t_hptr_D_0_1[0]), t_fptr_O_0_1[0]));
      t_hptr_D_0_1[1] = Traits::from_float(
          __fmaf_rn(rescale_o_factor_0[0], Traits::to_float(t_hptr_D_0_1[1]), t_fptr_O_0_1[1]));
      t_hptr_D_0_1[2] = Traits::from_float(
          __fmaf_rn(rescale_o_factor_1[0], Traits::to_float(t_hptr_D_0_1[2]), t_fptr_O_0_1[2]));
      t_hptr_D_0_1[3] = Traits::from_float(
          __fmaf_rn(rescale_o_factor_1[0], Traits::to_float(t_hptr_D_0_1[3]), t_fptr_O_0_1[3]));
    }
  } else {
    // MMA Acc F16 (only valid when kDataType == __half; bf16 forces kMmaAccFloat32==1).
    static_assert(std::is_same_v<kDataType, __half>,
                  "MMA Acc F16 path is only valid for __half activation dtype.");
    const kDataType* t_hptr_O_0_1 = reinterpret_cast<kDataType*>(R_O);
    if constexpr (kOStorageAccFloat32) {
      // (x,y) 0~7->{c0, c1}, (z,w)->8~15 {c2, c3} kValTileSeqLenP=1
      float* t_fptr_D_0_1 = reinterpret_cast<float*>(R_D + d_tile_id * 4);
      t_fptr_D_0_1[0] =
          __fmaf_rn(rescale_o_factor_0[0], t_fptr_D_0_1[0], Traits::to_float(t_hptr_O_0_1[0]));
      t_fptr_D_0_1[1] =
          __fmaf_rn(rescale_o_factor_0[0], t_fptr_D_0_1[1], Traits::to_float(t_hptr_O_0_1[1]));
      t_fptr_D_0_1[2] =
          __fmaf_rn(rescale_o_factor_1[0], t_fptr_D_0_1[2], Traits::to_float(t_hptr_O_0_1[2]));
      t_fptr_D_0_1[3] =
          __fmaf_rn(rescale_o_factor_1[0], t_fptr_D_0_1[3], Traits::to_float(t_hptr_O_0_1[3]));
    } else {
      kDataType* t_hptr_D_0_1 = reinterpret_cast<kDataType*>(R_D + d_tile_id * 2);
      t_hptr_D_0_1[0] =
          Traits::from_float(__fmaf_rn(rescale_o_factor_0[0], Traits::to_float(t_hptr_D_0_1[0]),
                                       Traits::to_float(t_hptr_O_0_1[0])));
      t_hptr_D_0_1[1] =
          Traits::from_float(__fmaf_rn(rescale_o_factor_0[0], Traits::to_float(t_hptr_D_0_1[1]),
                                       Traits::to_float(t_hptr_O_0_1[1])));
      t_hptr_D_0_1[2] =
          Traits::from_float(__fmaf_rn(rescale_o_factor_1[0], Traits::to_float(t_hptr_D_0_1[2]),
                                       Traits::to_float(t_hptr_O_0_1[2])));
      t_hptr_D_0_1[3] =
          Traits::from_float(__fmaf_rn(rescale_o_factor_1[0], Traits::to_float(t_hptr_D_0_1[3]),
                                       Traits::to_float(t_hptr_O_0_1[3])));
    }
  }
}

// Fold the new (m_new, l_new) into the running (m_old, l_old) after O has
// been rescaled by ``sync_rescaling_tiling_o``. Implements the FA-2
// recurrence ``l_new = exp(m_old - m_new) * l_old + rowsum(P)`` and
// ``m_old <- max(m_old, m_new)``.
__device__ __forceinline__ void sync_update_max_expsum(
    float* lane_row_max_new,          // &lane_row_max_new[0][0]
    float* lane_row_sum_new,          // &lane_row_sum_new[0][0]
    float* lane_block_row_max_old,    // &lane_block_row_max_old[0][0]
    float* lane_block_row_sum_old,    // &lane_block_row_sum_old[0][0]
    const float* rescale_o_factor_0,  // rescale factor 0 exp(m_old - m_new)
    const float* rescale_o_factor_1   // rescale factor 1 exp(m_old - m_new)
) {
  // Now, we can update m, l after O has been scaled.
  // Update l = exp(m_old - m_new) * l_old + row_sum(P).
  lane_block_row_sum_old[0] =
      (__fmaf_rn(rescale_o_factor_0[0], lane_block_row_sum_old[0], lane_row_sum_new[0]));
  lane_block_row_sum_old[1] =
      (__fmaf_rn(rescale_o_factor_1[0], lane_block_row_sum_old[1], lane_row_sum_new[1]));
  // 2. Then, update block row max for each lane.
  lane_block_row_max_old[0] = max(lane_block_row_max_old[0], lane_row_max_new[0]);
  lane_block_row_max_old[1] = max(lane_block_row_max_old[1], lane_row_max_new[1]);
}

// Final O scaling once the KV loop is done: ``O = (1 / l_final) * O`` and
// cast fp32 accumulator back to the activation dtype so the collective
// ``sync_store_o_r2g`` can emit 128-bit global stores.
template <const int kValTileHeadDimV, const int kOStorageAccFloat32, typename kDataType = __half>
__device__ __forceinline__ void sync_rescaling_final_o(
    uint32_t* R_D,                       // Final O after loop over N
    const float* lane_block_row_sum_old  // &lane_block_row_sum_old[0][0]
) {
  using Traits = DtypeTraits<kDataType>;
  // Finaly, we still have to rescale O once more.
  // O_output(D) = ( 1/l_final ) * O_final (FA2 paper)
  // static_assert(kValTileSeqLenP == 1);
  {  // kValTileSeqLenP = 1
    const float rescale_factor_0 = __frcp_rn(lane_block_row_sum_old[0]);
    const float rescale_factor_1 = __frcp_rn(lane_block_row_sum_old[1]);
#pragma unroll
    for (int j = 0; j < kValTileHeadDimV; ++j) {  // 8, 16, 32, ...
      // Scaling in registers & convert F32 -> kDataType for O collective store.
      if constexpr (kOStorageAccFloat32) {
        const float* t_fptr_D_0_1 = reinterpret_cast<float*>(R_D + j * 4);
        kDataType* t_hptr_D_0_1 = reinterpret_cast<kDataType*>(R_D + j * 4);
        t_hptr_D_0_1[0] = Traits::from_float(rescale_factor_0 * t_fptr_D_0_1[0]);
        t_hptr_D_0_1[1] = Traits::from_float(rescale_factor_0 * t_fptr_D_0_1[1]);
        t_hptr_D_0_1[2] = Traits::from_float(rescale_factor_1 * t_fptr_D_0_1[2]);
        t_hptr_D_0_1[3] = Traits::from_float(rescale_factor_1 * t_fptr_D_0_1[3]);
      } else {
        kDataType* t_hptr_D_0_1 = reinterpret_cast<kDataType*>(R_D + j * 2);
        t_hptr_D_0_1[0] = Traits::from_float(rescale_factor_0 * Traits::to_float(t_hptr_D_0_1[0]));
        t_hptr_D_0_1[1] = Traits::from_float(rescale_factor_0 * Traits::to_float(t_hptr_D_0_1[1]));
        t_hptr_D_0_1[2] = Traits::from_float(rescale_factor_1 * Traits::to_float(t_hptr_D_0_1[2]));
        t_hptr_D_0_1[3] = Traits::from_float(rescale_factor_1 * Traits::to_float(t_hptr_D_0_1[3]));
      }
    }  // end for kValTileHeadDimV
  }  // end for kValTileSeqLenP = 1
}

// Collective write-back of the final O tile from registers to gmem.
// Uses warp-shuffles across lane_id % 4 to pack the per-thread fragment
// layout into contiguous 128-bit vectors, then ``st.global.v4`` once per
// (lane % 4 == 0) thread. Scratch registers ``R_Q`` / ``R_K`` are reused
// as the shuffle staging buffers so no extra SMEM is needed. Rows with
// global idx >= ``seqlen_bound`` are not written, keeping tail padding
// (N % Br != 0) untouched.
template <const int Br, const int kHeadDim, const int kMmaAtomM, const int kMmaAtomN,
          const int kValTileHeadDimV, const int kOStorageAccFloat32, typename kDataType = __half>
__device__ __forceinline__ void sync_store_o_r2g(
    kDataType* gmem_ptr,              // O gmem ptr
    const int gmem_offset,            // O gmem global offset
    const int n_tile_id,              // curr tile id (seqlen) O_tile_id
    const int mma_tile_id,            // Q warp_QP 0~num MMAs, KV warp_KV 0
    uint32_t* R_D,                    // Final scaled O
    uint32_t* R_Q,                    // R_Q[1][4] for registers reuse
    uint32_t* R_K,                    // R_K[8][2] for registers reuse
    const int seqlen_bound = INT_MAX  // per-row bound: rows with global idx >=
                                      // seqlen_bound are not written to gmem
) {
  // Store O(D): Write O[Br,d] from regs -> gmem, collective store
  // with reg reuse & warp shuffle.
  const int lane_id = threadIdx.x % WARP_SIZE;  // 0~31
  // static_assert(kValTileSeqLenP == 1);
  {  // kValTileSeqLenP = 1
#pragma unroll
    for (int j = 0; j < kValTileHeadDimV; ++j) {  // 8
      // reuse R_Q[1][4], R_K[8][2] for collective store.
      uint32_t* t_uptr_Z_0 = reinterpret_cast<uint32_t*>(R_Q);
      uint32_t* t_uptr_Z_1 = reinterpret_cast<uint32_t*>(R_K);
      const int offset = (kOStorageAccFloat32) ? j * 4 : j * 2;
      t_uptr_Z_0[0] = R_D[offset + 0];
      t_uptr_Z_1[0] = R_D[offset + 1];
      t_uptr_Z_0[1] = __shfl_sync((0xffffffff), R_D[offset + 0], lane_id + 1, 4);
      t_uptr_Z_0[2] = __shfl_sync((0xffffffff), R_D[offset + 0], lane_id + 2, 4);
      t_uptr_Z_0[3] = __shfl_sync((0xffffffff), R_D[offset + 0], lane_id + 3, 4);
      t_uptr_Z_1[1] = __shfl_sync((0xffffffff), R_D[offset + 1], lane_id + 1, 4);
      t_uptr_Z_1[2] = __shfl_sync((0xffffffff), R_D[offset + 1], lane_id + 2, 4);
      t_uptr_Z_1[3] = __shfl_sync((0xffffffff), R_D[offset + 1], lane_id + 3, 4);

      // st.global.v4 128 bits. [Br,d]
      if (lane_id % 4 == 0) {
        // (0/1)*32 + (0/1)*16=(0,16,32,48), + 0~7 -> 0~56 kValTileSeqLenP = 1
        // int store_warp_regs_O_Br = warp_QP * (kMmaAtomM * kValTileSeqLenP ) + 0 * kMmaAtomM;
        const int store_warp_regs_O_Br = mma_tile_id * (kMmaAtomM);
        const int store_lane_gmem_O_Br =
            n_tile_id * Br + store_warp_regs_O_Br + lane_id / 4;  // 0~7
        // (0~3)*16 + (0/1)*8=(0,8,16,24,...,48,56)  warp_KV = 0
        // int store_warp_regs_O_d = warp_KV * (kMmaAtomN * kValTileHeadDimV) + j * kMmaAtomN;
        const int store_warp_regs_O_d = j * kMmaAtomN;
        const int store_lane_gmem_O_d = store_warp_regs_O_d;  // (0~3)*16+(0/8)
        const int store_gmem_O_addr_0 =
            (gmem_offset + (store_lane_gmem_O_Br + 0) * kHeadDim + store_lane_gmem_O_d);
        const int store_gmem_O_addr_1 =
            (gmem_offset + (store_lane_gmem_O_Br + 8) * kHeadDim + store_lane_gmem_O_d);
        // Row predicate: skip writes past QKV_seqlen to keep output rows
        // beyond the valid length untouched when N % Br != 0.
        if ((store_lane_gmem_O_Br + 0) < seqlen_bound) {
          cp_async::stg_sync_128b(&gmem_ptr[store_gmem_O_addr_0], t_uptr_Z_0);
        }
        if ((store_lane_gmem_O_Br + 8) < seqlen_bound) {
          cp_async::stg_sync_128b(&gmem_ptr[store_gmem_O_addr_1], t_uptr_Z_1);
        }
      }
    }  // end for kValTileHeadDimV
  }  // kValTileSeqLenP = 1
}

}  // namespace prefill
}  // namespace ffpa

// ============================================================================
// FFPA SM>=SM90 kernel templates with TMA-direct K/V staging.
//
// This header isolates all TMA-related kernel changes from the
// architecture-agnostic ``ffpa_attn_templates.cuh`` (which remains the
// fallback for SM < 9.0 and for ``tma=False`` callers). The SM90 kernel
// reuses the FFPA large-d Split-Q (FlashAttention-2) algorithm and only
// substitutes the per-tile K/V global-to-shared transfer with a TMA
// bulk-tensor copy (``cp.async.bulk.tensor.2d.global.shared``).
//
// Direct-write design
// -------------------
// TMA writes directly into the existing ``kPad==0`` XOR-swizzled K/V
// destination slot, no scratch buffer and no repack pass:
//
//   * The K TMA box is widened to ``kKvBoxCols=64`` fp16 cols
//     (``CU_TENSOR_MAP_SWIZZLE_128B``, equivalent to FFPA's
//     ``swizzle::permuted<64>`` = Cute ``Swizzle<3, 4, 3>``) when the
//     head-dim is large enough (``kHeadDim >= 128`` and
//     ``kHeadDim % 64 == 0``) and ``kPadK == 0``. Each K stage holds
//     ``kSubtilesPerKBox = kKvBoxCols/kMmaAtomK`` consumed sub-tiles and
//     the K issue rate drops by the same factor (e.g. 32->8 issues per
//     K-seqlen tile at D=512). Smaller head dims fall back to the
//     ``kKvBoxCols == kMmaAtomK`` (16 cols, ``SWIZZLE_32B``) shape.
//   * V always uses the 16-col ``SWIZZLE_32B`` box. V's per-tile issue
//     count is already half of K's, and widening V doubles its smem
//     footprint with smaller marginal payoff.
//   * For both K and V the FFPA hand-crafted swizzle formula is
//     bit-for-bit equivalent to the chosen TMA swizzle mode, so the
//     existing ldmatrix kPad==0 path consumes the TMA-written tile
//     directly. ``getExperimentalTmaSm90ScratchSize`` accounts for the
//     extra K smem implied by the wider K box.
//
// KV-axis tail tiles (``Nkv % Bc != 0``) are handled by TMA's OOB-fill
// (``CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE`` zero-fills out-of-bounds rows),
// so no cp.async fallback is needed for the K/V loads. Head-dim
// out-of-range speculative prefetches still no-op via the helper's
// ``d_tile_id >= kHeadDim/kCols`` guard.
//
// Why K/V only (and not Q) today
// ------------------------------
// Q is technically a perfectly valid TMA source too -- ``Q[Nb, Nh, Nq, D]``
// is contiguous BHND just like K and V, and a 2D TMA descriptor with a
// (Br, kMmaAtomK) box could load it. We deliberately keep Q on the
// existing cp.async path because:
//
//   1. In the Split-Q FA-2 dataflow, each block reads its single Q tile
//      exactly once and then keeps it resident in registers (kPersistQs2r)
//      or shared (kPersistQg2s) for the entire ``Tc`` outer loop, while
//      every K/V tile is reloaded ``Tc`` times across the head-dim
//      sub-tiling. So the gmem-bandwidth-amortised win from TMA lives
//      almost entirely on K/V; moving Q to TMA buys ~1/Tc of the savings
//      while doubling the descriptor-build / mbarrier bookkeeping cost on
//      the host and adding a third TMA pipeline to schedule.
//   2. Q already participates in the cp.async commit/wait_group pipeline
//      that gates the K loads; replacing only Q with TMA would force us to
//      either drain that group early (costing a __syncthreads) or keep Q
//      out of the existing prefetch window (losing the kPersistQg2s win).
//   3. The SM90 path inherits the kPersistQs2r / kPersistQg2s
//      register/smem reuse modes byte-for-byte from the fallback kernel.
//
// Multi-stage (kStageQK / kStagePV >= 1) is supported and the K/V loads
// are pipelined: each tile is split into an asynchronous ``issue_X_tile``
// (TMA bulk-tensor copy with mbarrier-tx tracking) and a deferred
// ``consume_X_tile`` (mbarrier wait + ``__syncthreads``). The issue
// happens kStageX-1 outer iterations before the consume, so a stage-N
// TMA copy overlaps with the stage-(N-1) MMA consumption. Per-stage
// mbarriers ensure independent stage progress. K stages advance at the
// box cadence (every ``kSubtilesPerKBox`` sub-tiles); Q stages keep the
// per-sub-tile cadence via a separate ``q_smem_sel`` index because Q
// smem is still tiled by ``kMmaAtomK`` in d.
//
// Why Split-D is structurally hostile to TMA
// ------------------------------------------
// FFPA's Split-D dataflow tiles the head dim by ``kMmaAtomK`` (=16 fp16
// = 32 B per row). ``cp.async.bulk.tensor`` is a single-thread SASS
// instruction with a fixed per-issue cost (descriptor lookup, coordinate
// projection, mbarrier-tx write, engine-queue insert). A naive 2 KB box
// makes the fixed:transfer ratio ~1:5, so TMA-engine issue rate is the
// bottleneck -- not gmem bandwidth or LSU port contention. cp.async
// amortises the same bytes across all 256 threads in parallel.
//
// Widening the K box to 64 cols (SWIZZLE_128B, what this kernel does)
// cuts the K issue count 4x and brings the TMA path to parity with the
// well-tuned cp.async + multi-stages baseline on D=512, but does not
// beat it. cp.async's 256-thread parallel dispatch is hard to beat with
// a single-issuer instruction; closing the gap further would require a
// larger redesign (super-tile K/V on TMA + warp-specialised
// producer/consumer), not a drop-in K/V replacement.
//
// Optimisations applied on top of the mbarrier path
// -------------------------------------------------
//   * K-issue is dispatched BEFORE the matching Q cp.async submission in
//     every prefetch slot and inner-loop iter, so the bulk-tensor
//     request is on-the-wire before the cp.async commit (overlap
//     dispatch).
//   * Issuer-lane rotation: the SASS-issuing thread rotates across warps
//     via ``issuer_lane = (idx * WARP_SIZE) mod kNumThreads`` (K and V
//     offset by ``kNumThreads/2`` so they hit disjoint LSU schedulers).
//     Threaded through ``ffpa::tma::load_2d`` /
//     ``issue_load_2d_to_dst_swizzled``.
//   * Wider K TMA box (``kKvBoxCols=64`` / ``SWIZZLE_128B``) cuts K
//     issue count 4x; see the direct-write design notes above.
// ============================================================================
#pragma once

#include "cuffpa/prefill.cuh"
#include "cuffpa/tma.cuh"

namespace ffpa {
namespace sm90 {

// Eligibility check for the experimental TMA SM90 large-d kernel.
//
// * ``kEligibleHeadDim``  : the K TMA box is widened to 64 fp16 cols
//                           (``SWIZZLE_128B``) only when
//                           ``kHeadDim >= 128 && kHeadDim % 64 == 0``.
//                           At smaller / non-aligned head dims the K
//                           box would fall back to the 16-col
//                           ``SWIZZLE_32B`` shape, which is strictly
//                           slower than the cp.async path, so we avoid
//                           instantiating the SM90 TMA kernel there
//                           altogether and let the launcher use the
//                           non-TMA fallback. This also keeps compile
//                           time and binary size in check.
// * ``kRequiresSwizzledSmem``: TMA writes directly into the kPad==0
//                           XOR-swizzled K/V slot, so any non-zero
//                           ``kPadK`` / ``kPadV`` would leave per-row
//                           pad gaps that the bulk-tensor copy cannot
//                           skip.
// * ``kSupportsAllStages``: multi-stage is supported (kStageQK and
//                           kStagePV may each be >= 1).
template <const int kHeadDim, const int kStageQK, const int kStagePV, const int kPadQ,
          const int kPadK, const int kPadV>
struct ExperimentalTmaLargeDConfig {
  static constexpr bool kEligibleHeadDim = (kHeadDim >= 128 && (kHeadDim % 64) == 0);
  // TMA writes directly into the kPad==0 XOR-swizzled destination slot.
  // FFPA's ``swizzle::permuted<16>`` (V, K when ``kKvBoxCols == 16``) is
  // bit-for-bit equivalent to ``CU_TENSOR_MAP_SWIZZLE_32B`` (Cute
  // ``Swizzle<1, 4, 3>``: byte address bit 4 XOR bit 7), and
  // ``swizzle::permuted<64>`` (K when ``kKvBoxCols == 64``) is
  // equivalent to ``CU_TENSOR_MAP_SWIZZLE_128B`` (Cute
  // ``Swizzle<3, 4, 3>``), so the existing ldmatrix kPad==0 path reads
  // the TMA-written tile correctly with zero repack. ``kPad>0`` padded
  // layouts are incompatible because TMA writes the box contiguously
  // in swizzled rows -- it cannot leave per-row pad gaps.
  static constexpr bool kRequiresSwizzledSmem = (kPadK == 0 && kPadV == 0);
  static constexpr bool kSupportsAllStages = (kStageQK >= 1 && kStagePV >= 1);
  static constexpr bool kCanAttempt =
      kEligibleHeadDim && kRequiresSwizzledSmem && kSupportsAllStages && (kPadQ >= 0);
};

}  // namespace sm90
}  // namespace ffpa

// ============================================================================
// ffpa_stages_split_q_large_d_sm90_template
// ----------------------------------------------------------------------------
// Mirror of ``ffpa_stages_split_q_large_d_template`` from
// ``ffpa_attn_templates.cuh`` but with K/V tile staging delegated to TMA
// (``cp.async.bulk.tensor.2d.global.shared``) writing directly into the
// existing kPad==0 XOR-swizzled K/V destination slot (no scratch, no
// repack). All other algorithmic behavior (online softmax, causal mask,
// GQA/MQA,
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
  // K/V global pointers are only addressed via the TMA descriptors on
  // this path; suppress unused-variable warnings without touching the
  // rest of the kernel skeleton.

  (void)K;
  (void)V;
  (void)K_gmem_offset;
  (void)V_gmem_offset;
  const int O_gmem_offset = Q_gmem_offset;

  // Per-block batch/head row offset into the flat 2D K/V tensor that the
  // TMA descriptor addresses. The descriptor was built with
  // globalDim = (kHeadDim, B*Nh_kv*Nkv) so a single ``major_coord`` row
  // index selects the right (batch, kv_head, seqlen) row. Without this
  // offset every block reads from batch 0 / kv_head 0, which produces
  // garbage outputs for any block whose (Nb_id, kv_head_idx) != (0, 0).
  const int kv_row_base = (Nb_id * Nh_kv + kv_head_idx) * Nkv;

  if ((Q_tile_id * Br) >= Nq)
    return;

  // TMA bulk-tensor-load requires the smem destination address to be
  // 128-byte aligned (PTX hardware constraint, holds for all swizzle
  // modes). The hardware swizzle phase additionally depends on the
  // ABSOLUTE smem byte address: SWIZZLE_32B repeats every 256 B,
  // SWIZZLE_128B repeats every 1024 B, so the per-stage K/V slot must
  // be aligned to the largest stride in use for the TMA-written byte
  // layout to match FFPA's relative-row permuted swizzle. We bump the
  // dynamic-smem base alignment to 1024 (covers all swizzle modes) and
  // rely on K_tile_size / V_tile_size being multiples of that stride.
  extern __shared__ __align__(1024) unsigned char ffpa_smem_raw[];
  kDataType* smem = reinterpret_cast<kDataType*>(ffpa_smem_raw);
  constexpr int Q_tile_size = Br * (kMmaAtomK + kPadQ);
  // Widen the K TMA box to 64 fp16 cols (SWIZZLE_128B) when the
  // head-dim is large enough to amortise the bigger per-stage smem
  // footprint and ``kPadK==0`` so the swizzle matches
  // ``swizzle::permuted<64>``. Each K stage then holds a single wider
  // box covering ``kSubtilesPerKBox = kKvBoxCols/kMmaAtomK`` consumed
  // sub-tiles, and the TMA issue count per K-seqlen tile drops from
  // ``kHeadDim/kMmaAtomK`` to ``kHeadDim/kKvBoxCols`` (e.g. 32->8 at
  // D=512). V stays on the 16-col box (its issue count is already half
  // of K's and widening V doubles the V-side smem with smaller
  // marginal payoff).
  constexpr int kKvBoxCols =
      ((kPadK == 0) && (kHeadDim % 64 == 0) && (kHeadDim >= 128)) ? 64 : kMmaAtomK;
  constexpr int kSubtilesPerKBox = kKvBoxCols / kMmaAtomK;
  constexpr int K_tile_size = Bc * (kKvBoxCols + kPadK);
  constexpr int V_tile_size = Bc * (kMmaAtomN * 2 + kPadV);
  kDataType* Q_tile_smem = smem;
  kDataType* K_tile_smem = (Q_tile_smem + (kPersistQg2s ? ((kHeadDim / kMmaAtomK) * Q_tile_size)
                                                        : (kStageQK * Q_tile_size)));
  kDataType* V_tile_smem = (kShareSmemQKV ? Q_tile_smem : K_tile_smem + kStageQK * K_tile_size);
  // TMA writes directly into the K/V destination slots above (kPad==0
  // XOR-swizzled, layout matches the chosen TMA swizzle mode). No
  // scratch buffer is needed: the launcher allocates ``kQKVSmemMaxSize
  // + getExperimentalTmaSm90ScratchSize`` where the scratch size only
  // accounts for the extra K bytes implied by the wider K box.
  const uint32_t smem_Q_base_ptr = __cvta_generic_to_shared(Q_tile_smem);
  const uint32_t smem_K_base_ptr = __cvta_generic_to_shared(K_tile_smem);
  const uint32_t smem_V_base_ptr = __cvta_generic_to_shared(V_tile_smem);

  // One mbarrier per K/V stage so that an outstanding load for stage N+1
  // does not race with the consumer waiting on stage N. arrive_count=1:
  // the only producer arrival is the ``barrier_arrive_tx`` from thread 0
  // inside ``ffpa::tma::load_2d``; consumers do their own arrive() in
  // ``wait_barrier`` to advance the phase for the next reuse.
  __shared__ alignas(alignof(ffpa::tma::barrier_t)) unsigned char
      K_tma_barrier_storage[kStageQK * sizeof(ffpa::tma::barrier_t)];
  __shared__ alignas(alignof(ffpa::tma::barrier_t)) unsigned char
      V_tma_barrier_storage[kStagePV * sizeof(ffpa::tma::barrier_t)];
  ffpa::tma::barrier_t* K_tma_barriers =
      reinterpret_cast<ffpa::tma::barrier_t*>(K_tma_barrier_storage);
  ffpa::tma::barrier_t* V_tma_barriers =
      reinterpret_cast<ffpa::tma::barrier_t*>(V_tma_barrier_storage);
  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < kStageQK; ++s) {
      ffpa::tma::init_barrier(&K_tma_barriers[s], 1);
    }
#pragma unroll
    for (int s = 0; s < kStagePV; ++s) {
      ffpa::tma::init_barrier(&V_tma_barriers[s], 1);
    }
  }
  __syncthreads();

  // Per-slot bookkeeping: track whether each K/V stage slot was actually
  // issued via TMA (i.e. d-tile in range and descriptor non-null).
  // Speculative prefetch loops can call issue with d_tile_id past the
  // head-dim end; the issue helper no-ops in that case, and consume must
  // skip the wait to avoid hanging. KV-axis tail tiles do NOT need a
  // fallback because the TMA descriptor's OOB-fill zero-fills out-of-range
  // rows automatically.
  // ``K_tma_phase[s]`` / ``V_tma_phase[s]`` track the parity bit each
  // consumer must wait on for stage ``s``: 0 on the first issue, 1 on the
  // second, alternating thereafter (mbarriers init'd with arrive_count=1
  // flip phase as soon as the producer tx-count completes).
  bool K_tma_used[kStageQK];
  bool V_tma_used[kStagePV];
  uint32_t K_tma_phase[kStageQK];
  uint32_t V_tma_phase[kStagePV];
#pragma unroll
  for (int s = 0; s < kStageQK; ++s) {
    K_tma_used[s] = false;
    K_tma_phase[s] = 0;
  }
#pragma unroll
  for (int s = 0; s < kStagePV; ++s) {
    V_tma_used[s] = false;
    V_tma_phase[s] = 0;
  }

  // ---------- helper lambdas: issue / consume K/V tile stagers ----------
  // ``issue_X_tile`` issues a TMA bulk-tensor copy directly into the
  // destination swizzled slot. Tail KV rows are auto-zero-filled by TMA;
  // out-of-range head-dim sub-tiles no-op. ``consume_X_tile`` waits on
  // the per-stage mbarrier (parity-based) and ``__syncthreads`` so all
  // threads see the freshly written tile before the s2r ldmatrix reads
  // start, then flips the per-slot phase bit for the next reuse.
  //
  // ``issuer_lane`` rotates the SASS-issuing thread across warps so the
  // per-warp-scheduler LSU dispatch port is not single-thread
  // bottlenecked: each call rotates ``idx * WARP_SIZE`` modulo
  // ``kNumThreads``, spreading inner-loop K issues across all warps'
  // lane 0. K and V use disjoint base offsets so a given inner iter's
  // K and V issues fire from different warps and contend for different
  // LSU schedulers.
  constexpr int kIssuerStep = WARP_SIZE;
  constexpr int kVIssuerOffset = (kNumThreads >= 64) ? (kNumThreads / 2) : 0;
  auto k_issuer_lane = [](int idx) -> int { return (idx * kIssuerStep) % kNumThreads; };
  auto v_issuer_lane = [](int idx) -> int {
    return ((idx * kIssuerStep) + kVIssuerOffset) % kNumThreads;
  };

  auto issue_K_tile = [&](int tile_K_seqlen, int d_tile, int dst_stage,
                          int issuer_lane = 0) -> void {
    // d_tile is in units of ``kKvBoxCols`` (one wide TMA box per stage
    // covering kSubtilesPerKBox consumed sub-tiles).
    K_tma_used[dst_stage] =
        ffpa::tma::issue_load_2d_to_dst_swizzled<Bc, kHeadDim, kKvBoxCols, K_tile_size>(
            K_tile_smem, K_tma_desc, kv_row_base + tile_K_seqlen * Bc, d_tile, dst_stage,
            K_tma_barriers[dst_stage], issuer_lane);
  };
  auto consume_K_tile = [&](int dst_stage) -> void {
    if (K_tma_used[dst_stage]) {
      ffpa::tma::wait_barrier_parity(K_tma_barriers[dst_stage], K_tma_phase[dst_stage]);
      K_tma_phase[dst_stage] ^= 1u;
      K_tma_used[dst_stage] = false;
      __syncthreads();
    }
  };
  auto issue_V_tile = [&](int tile_K_seqlen, int d_tile, int dst_stage,
                          int issuer_lane = 0) -> void {
    V_tma_used[dst_stage] =
        ffpa::tma::issue_load_2d_to_dst_swizzled<Bc, kHeadDim, kMmaAtomN * 2, V_tile_size>(
            V_tile_smem, V_tma_desc, kv_row_base + tile_K_seqlen * Bc, d_tile, dst_stage,
            V_tma_barriers[dst_stage], issuer_lane);
  };
  auto consume_V_tile = [&](int dst_stage) -> void {
    if (V_tma_used[dst_stage]) {
      ffpa::tma::wait_barrier_parity(V_tma_barriers[dst_stage], V_tma_phase[dst_stage]);
      V_tma_phase[dst_stage] ^= 1u;
      V_tma_used[dst_stage] = false;
      __syncthreads();
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
    // K (TMA, long-latency) is dispatched FIRST in every prefetch /
    // inner-loop slot so the bulk-tensor copy is on-the-wire before the
    // matching Q cp.async commit; this lets the two overlap on the gmem
    // path instead of strictly serialising K behind Q.
    if constexpr (kPrefetchQK) {
      if constexpr (kStageQK > 1) {
        if (tile_K_seqlen == 0) {
#pragma unroll
          for (int stage = 0; stage < (kStageQK - 1); ++stage) {
            issue_K_tile(tile_K_seqlen, stage, stage,
                         k_issuer_lane(stage));  // TMA: dispatched first
            ffpa::tma::bulk_commit_group();
            if constexpr (!kPersistQg2s) {
              ffpa::prefill::cp_async_qkv_g2s<Br, Q_tile_size, kHeadDim, kMmaAtomK, kNumThreads,
                                              kPadQ>(smem_Q_base_ptr, Q, Q_gmem_offset, Q_tile_id,
                                                     stage, stage, Nq);
            }
            ffpa::cp_async::commit_group();
          }
          ffpa::cp_async::wait_group<(kStageQK - 2)>();  // drains Q only
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
          issue_K_tile(tile_K_seqlen, stage, stage,
                       k_issuer_lane(stage));  // TMA: dispatched first
          ffpa::tma::bulk_commit_group();
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
          ffpa::cp_async::commit_group();
        }
        ffpa::cp_async::wait_group<(kStageQK - 2)>();  // drains Q only
        __syncthreads();
      }
    }

    // V is on the TMA path: every issue_V_tile is a cp.async.bulk.tensor
    // copy whose completion is tracked by the per-stage mbarrier (signed
    // by ``cuda::device::barrier_arrive_tx`` inside ``load_2d`` and
    // awaited via ``consume_V_tile`` -> ``wait_barrier_parity``). To keep
    // the cp.async-vs-TMA bookkeeping unambiguous for downstream readers
    // and for static analysis tools, we additionally close a TMA-only
    // group with ``bulk_commit_group`` after each issue. The matching
    // ``bulk_wait_group<(kStagePV-2)>`` below provides a coarse fallback
    // drain alongside the per-slot mbarrier (both are correct; mbarrier
    // is finer-grained and is what actually orders MMA reads).
    if constexpr ((!kShareSmemQKV) && kPrefetchPV && kStagePV > 1) {
      if constexpr (kStagePV > 1) {
#pragma unroll
        for (int stage = 0; stage < (kStagePV - 1); ++stage) {
          issue_V_tile(tile_K_seqlen, stage, stage, v_issuer_lane(stage));
          ffpa::tma::bulk_commit_group();
        }
      }
    }

    ffpa::utils::fill_3D_regs<uint32_t, kValTileSeqLenQ, kValTileSeqLenK,
                              (kMmaAccFloat32QK) ? 4 : 2>(R_S, 0);
#pragma unroll
    for (int tile_K_d = 0; tile_K_d < (kHeadDim / kMmaAtomK); ++tile_K_d) {
      // Box-coordinate translation. ``box_idx`` is the K TMA box this
      // sub-tile lives in; ``subtile_idx`` is which of the
      // ``kSubtilesPerKBox`` consumed sub-tiles inside that box; the
      // wider K smem layout has row stride ``kKvBoxCols`` so the s2r
      // path adds ``subtile_col_off`` to the column index. K issue /
      // consume happens once per box (i.e. on subtile_idx == 0); for
      // kKvBoxCols == kMmaAtomK these all degenerate to the per-iter
      // cadence (kSubtilesPerKBox == 1, subtile_idx always 0).
      const int box_idx = tile_K_d / kSubtilesPerKBox;
      const int subtile_idx = tile_K_d % kSubtilesPerKBox;
      const int subtile_col_off = subtile_idx * kMmaAtomK;
      const bool is_box_head = (subtile_idx == 0);
      constexpr int kKBoxCount = kHeadDim / kKvBoxCols;
      // K stage indices step at the box cadence (every kSubtilesPerKBox
      // sub-tiles); Q stage indices keep the original per-sub-tile cadence
      // because Q smem is still tiled by kMmaAtomK in d.
      const int smem_sel = box_idx % kStageQK;
      const int smem_sel_next = (box_idx + (kStageQK - 1)) % kStageQK;
      const int q_smem_sel = (tile_K_d) % kStageQK;
      const int q_smem_sel_next = (tile_K_d + (kStageQK - 1)) % kStageQK;
      // Issue K (TMA) before Q (cp.async) so the bulk-tensor request is
      // on-the-wire before the cp.async submission (overlap dispatch).
      // With wider K boxes we issue one TMA per box, not per sub-tile.
      if (is_box_head) {
        const int issue_box = (kStageQK > 1) ? (box_idx + (kStageQK - 1)) : box_idx;
        // The OOB guard inside ``issue_load_2d_to_dst_swizzled`` skips
        // box_idx >= kKBoxCount; but the prefetch can speculatively
        // address beyond, so we keep the call and rely on its guard.
        (void)kKBoxCount;
        issue_K_tile(tile_K_seqlen, issue_box, (kStageQK > 1) ? smem_sel_next : smem_sel,
                     k_issuer_lane(box_idx));
        ffpa::tma::bulk_commit_group();  // K -> TMA bulk counter (mbarrier-waited)
      }
      if constexpr (kPersistQs2r) {
        if (tile_K_seqlen == 0) {
          ffpa::prefill::cp_async_qkv_g2s<Br, Q_tile_size, kHeadDim, kMmaAtomK, kNumThreads, kPadQ>(
              smem_Q_base_ptr, Q, Q_gmem_offset, Q_tile_id,
              (kStageQK > 1) ? (tile_K_d + (kStageQK - 1)) : tile_K_d,
              (kStageQK > 1) ? q_smem_sel_next : q_smem_sel, Nq);
        }
      } else {
        if constexpr (!kPersistQg2s) {
          ffpa::prefill::cp_async_qkv_g2s<Br, Q_tile_size, kHeadDim, kMmaAtomK, kNumThreads, kPadQ>(
              smem_Q_base_ptr, Q, Q_gmem_offset, Q_tile_id,
              (kStageQK > 1) ? (tile_K_d + (kStageQK - 1)) : tile_K_d,
              (kStageQK > 1) ? q_smem_sel_next : q_smem_sel, Nq);
        }
      }
      ffpa::cp_async::commit_group();  // Q (when loaded above) -> cp.async counter

      if constexpr (kStageQK <= 1) {
        ffpa::cp_async::wait_group<0>();  // drain Q
        __syncthreads();
      }

      // Drain the K TMA load for the slot we are about to read via its
      // per-slot mbarrier (parity-based). bulk_commit/bulk_wait above
      // additionally provide a FIFO-coarse drain of the TMA queue.
      // With wider K boxes we only consume on the box head; subsequent
      // sub-tiles inside the same box read from the already-resident slot.
      if (is_box_head) {
        consume_K_tile(smem_sel);
      }

      static_assert(kValTileSeqLenQ == 1);
      {
        if constexpr (kPersistQs2r) {
          if (tile_K_seqlen == 0) {
            ffpa::prefill::sync_fetch_qkv_frags_s2r<0, 4, Q_tile_size, kMmaAtomM, kMmaAtomN,
                                                    kMmaAtomK, kPadQ, kDataType>(
                smem_Q_base_ptr, &R_Q[0][tile_K_d][0], warp_QP, 0, 0, q_smem_sel);
          }
        } else {
          if constexpr (!kPersistQg2s) {
            ffpa::prefill::sync_fetch_qkv_frags_s2r<0, 4, Q_tile_size, kMmaAtomM, kMmaAtomN,
                                                    kMmaAtomK, kPadQ, kDataType>(
                smem_Q_base_ptr, &R_Q[0][0][0], warp_QP, 0, 0, q_smem_sel);
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
                                                  kMmaAtomK, kPadK, kDataType, kKvBoxCols>(
              smem_K_base_ptr, &R_K[j][0], warp_KV, j, 0, smem_sel, subtile_col_off);
        }
      } else {
        ffpa::prefill::sync_fetch_qkv_frags_s2r<0, 2, K_tile_size, kMmaAtomM, kMmaAtomN, kMmaAtomK,
                                                kPadK, kDataType, kKvBoxCols>(
            smem_K_base_ptr, &R_K[reg_st_idx][0], warp_KV, 0, 0, smem_sel, subtile_col_off);
      }

      if constexpr ((kShareSmemQKV) && kPrefetchPV && kStagePV > 1) {
        if (tile_K_d == (kHeadDim / kMmaAtomK - 1)) {
          __syncthreads();
          if constexpr (kStagePV > 1) {
#pragma unroll
            for (int stage = 0; stage < (kStagePV - 1); ++stage) {
              issue_V_tile(tile_K_seqlen, stage, stage, v_issuer_lane(stage));
              ffpa::tma::bulk_commit_group();  // V is TMA, not cp.async
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
                                                      kMmaAtomK, kPadK, kDataType, kKvBoxCols>(
                  smem_K_base_ptr, &R_K[reg_st_idx][0], warp_KV, (j + 1), 0, smem_sel,
                  subtile_col_off);
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
          ffpa::cp_async::wait_group<(kStageQK - 2)>();  // drains Q
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
          issue_V_tile(tile_K_seqlen, stage, stage, v_issuer_lane(stage));
          ffpa::tma::bulk_commit_group();  // V is TMA, not cp.async
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

    // Drain TMA V copies issued earlier this iter to bound in-flight
    // queue depth. Per-slot mbarrier already guards the smem read in
    // consume_V_tile; this is the matching coarse-group drain.
    if constexpr (kStagePV > 1) {
      ffpa::tma::bulk_wait_group<(kStagePV - 2)>();
      __syncthreads();
    }

    if constexpr ((!kShareSmemQKV) && kPrefetchQK && kStageQK > 1) {
      if ((tile_K_seqlen + 1) < Tc_eff) {
#pragma unroll
        for (int stage = 0; stage < (kStageQK - 1); ++stage) {
          issue_K_tile(tile_K_seqlen + 1, stage, stage,
                       k_issuer_lane(stage));  // TMA: dispatched first
          ffpa::tma::bulk_commit_group();
          if constexpr (!kPersistQs2r) {
            ffpa::prefill::cp_async_qkv_g2s<Br, Q_tile_size, kHeadDim, kMmaAtomK, kNumThreads,
                                            kPadQ>(smem_Q_base_ptr, Q, Q_gmem_offset, Q_tile_id,
                                                   stage, stage, Nq);
          }
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
          issue_V_tile(tile_K_seqlen, (kStagePV > 1) ? (tile_V_d + (kStagePV - 1)) : tile_V_d,
                       (kStagePV > 1) ? smem_sel_v_next : smem_sel_v, v_issuer_lane(tile_V_d));
          ffpa::tma::bulk_commit_group();  // V is TMA, not cp.async
          if constexpr (kStagePV <= 1) {
            ffpa::tma::bulk_wait_group<0>();
            __syncthreads();
          }
          // Drain the V TMA load for the slot we are about to read via
          // its per-slot mbarrier (parity-based). The bulk_wait_group
          // above is a coarser FIFO-drain; this is the precise per-slot
          // wait + __syncthreads that materialises the swizzled tile in
          // the destination smem stage before the s2r ldmatrix reads.
          consume_V_tile(smem_sel_v);
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
                  issue_K_tile(tile_K_seqlen + 1, stage, stage,
                               k_issuer_lane(stage));  // TMA: dispatched first
                  ffpa::tma::bulk_commit_group();
                  if constexpr (!kPersistQs2r) {
                    ffpa::prefill::cp_async_qkv_g2s<Br, Q_tile_size, kHeadDim, kMmaAtomK,
                                                    kNumThreads, kPadQ>(
                        smem_Q_base_ptr, Q, Q_gmem_offset, Q_tile_id, stage, stage, Nq);
                  }
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
            ffpa::tma::bulk_wait_group<(kStagePV - 2)>();
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

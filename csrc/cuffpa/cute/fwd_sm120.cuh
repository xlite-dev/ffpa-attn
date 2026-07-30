#pragma once

#include "attn_traits.cuh"

#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cutlass.h>
#include <cutlass/device_kernel.h>

// exp2f log-domain constants (FFPA_M_LOG2E / FFPA_M_LN2) live in common.cuh.
#include "common.cuh"
#include "attn_bias.cuh"
#include "dropout.cuh"
#include "softmax.cuh"

using TmaBarrier = cutlass::arch::ClusterTransactionBarrier;
using CtaBarrier = cutlass::arch::ClusterBarrier;

template <typename Traits, typename TmaQ, typename TmaK, typename TmaV,
          typename TmaO, int kHasAttnBias = 0, int kHasDropout = 0>
__global__ void __launch_bounds__(Traits::kNumThreads, 1)
    ffpa_attn_split_d_fwd_cute_sm120(
        CUTLASS_GRID_CONSTANT TmaQ const tma_q,
        CUTLASS_GRID_CONSTANT TmaK const tma_k,
        CUTLASS_GRID_CONSTANT TmaV const tma_v,
        CUTLASS_GRID_CONSTANT TmaO const tma_o,
        typename Traits::Element* __restrict__ O,
        float* __restrict__ softmax_lse, int Nq, int Nkv, int Nh, int Nh_kv,
        float scale, int Tc, int causal, int total_q_rows, int total_kv_rows,
        const void* __restrict__ attn_bias = nullptr, int attn_bias_dtype = 0,
        long long attn_bias_stride_b = 0, long long attn_bias_stride_h = 0,
        long long attn_bias_stride_m = 0, long long attn_bias_stride_n = 0,
        float dropout_p = 0.0f, unsigned long long philox_seed = 0,
        unsigned long long philox_offset = 0) {
  // Split-D Flash Attention forward (non-WS, CuTe TMA).
  //
  // Algorithm per KV tile:
  //   1. QK GEMM: S[Br,Bc] += Q[Br,kQKDChunk] @ K[Bc,kQKDChunk]^T
  //      accumulated over kDChunksQK = kHeadDim/kQKDChunk split-D chunks.
  //   2. Online softmax: row-max, exp2, row-sum with rescale factor.
  //   3. PV GEMM: O[Br,kVDChunk] += P[Br,Bc] @ V[Bc,kVDChunk]
  //      accumulated over kDChunksV = kHeadDim/kVDChunk split-D chunks.
  //      O is rescaled by row_scale before each kv_tile > 0.
  //   4. Epilogue: O /= row_sum, convert to half, store to gmem.
  //
  // TMA pipeline: tid=0 issues TMA loads inline (non-WS). All threads
  // participate in MMA. Barriers: qk_full (TmaBarrier, init=1) signals
  // data ready; qk_empty (CtaBarrier, init=kNumThreads) signals stage
  // consumed. Phase tracking: chunk_index = kv_tile*kDChunks + d_chunk,
  // phase = (chunk_index / kStages) & 1.
  //
  // Layout transforms (all defined in gemm.cuh; see those headers for the
  // m16n8k16-fragment rationale and upstream references):
  //   convert_layout_acc_rowcol: MMA C-fragment → [rows, cols] for softmax
  //     row-max/exp/sum (each row's columns land in one thread so __shfl_xor
  //     reduces across the 4 lanes sharing a row).
  //   convert_layout_acc_Aregs:  MMA C-fragment → A-operand regs for PV
  //     (reuses P registers as MMA-A without writing back to smem).
  //   convert_type:              f32 acc → f16 P/O in-register, zero copy.
  //   gemm_ss / gemm_rs:         software-pipelined ldmatrix + mma.sync
  //     (gemm_rs only preloads B=V since A=P is already in regs).
  //   SmemLayoutVt: transposed V layout for gemm_rs B-operand (LDSM_T).
  // Why NOT WS? Please check ../fwd_sm120.cuh for more details.

  using namespace cute;
  using cute::tma_store_arrive;
  using cute::tma_store_wait;
  using Element = typename Traits::Element;
  using SmemLayoutQ = typename Traits::SmemLayoutQ;
  using SmemLayoutK = typename Traits::SmemLayoutK;
  using SmemLayoutV = typename Traits::SmemLayoutV;
  using SmemLayoutVt = typename Traits::SmemLayoutVt;
  using SmemLayoutO = typename Traits::SmemLayoutO;
  using TiledMmaQK = typename Traits::TiledMmaQK;
  using TiledMmaPV = typename Traits::TiledMmaPV;
  using SmemCopyAtom = typename Traits::SmemCopyAtom;
  using SmemCopyAtomTransposed = typename Traits::SmemCopyAtomTransposed;

  constexpr int kBr = Traits::kBr;
  constexpr int kBc = Traits::kBc;
  constexpr int kQKDChunk = Traits::kQKDChunk;
  constexpr int kVDChunk = Traits::kVDChunk;
  constexpr int kHeadDim = Traits::kHeadDim;
  constexpr int kDChunksQK = Traits::kDChunksQK;
  constexpr int kDChunksV = Traits::kDChunksV;
  constexpr int kNumThreads = Traits::kNumThreads;
  constexpr int kStagesQK = Traits::kStagesQK;
  constexpr int kStagesPV = Traits::kStagesPV;

  constexpr int kQChunkElements = cosize(SmemLayoutQ{});
  constexpr int kKChunkElements = cosize(SmemLayoutK{});
  constexpr int kVChunkElements = cosize(SmemLayoutV{});

  // TMA-O epilogue reuses v_base smem as the O staging buffer; guard that it
  // fits. The "no in-flight V TMA at epilogue entry" invariant holds for any
  // kDChunksV/kStagesPV: every v_chunk's V is consumed via TmaBarrier::wait
  // (v_full) inside the PV loop, so by loop exit all V loads are drained and
  // v_base is safe to overwrite after the epilogue's __syncthreads().
  static_assert(cosize(SmemLayoutO{}) <= kStagesPV * cosize(SmemLayoutV{}),
                "TMA-O: O staging buffer must fit in reused V-stage smem");

  const int Nb_id = blockIdx.y / Nh;
  const int Nh_id = blockIdx.y % Nh;
  const int Q_tile_id = blockIdx.x;
  const int group_size = Nh / Nh_kv;
  const int kv_head_idx = Nh_id / group_size;
  const int Br_base = Q_tile_id * kBr;
  const int tid = threadIdx.x;

  if (Br_base >= Nq)
    return;

  const int kv_offset = Nkv - Nq;
  const int causal_thresh_row0 = Br_base + kv_offset;
  const int Tc_eff =
      causal ? min(Tc, ((Br_base + kBr - 1 + kv_offset) / kBc) + 1) : Tc;
  const int mask_start_tile =
      causal ? max(0, (causal_thresh_row0 + 1) / kBc) : INT_MAX;

  // Per-head global row origins injected into the TMA views via domain_offset
  // below. Using the true per-head row count (Nq / Nkv) rather than
  // ceil(N/kBr)*kBr keeps the TMA row coordinate correct when N % kBr != 0
  // (the non-aligned case); folding the head dim into the tile index would
  // accumulate a (kBr - N%kBr) row misalignment per head.
  const int q_row_offset = (Nb_id * Nh + Nh_id) * Nq;
  const int kv_row_offset = (Nb_id * Nh_kv + kv_head_idx) * Nkv;

  // SMEM layout: [q_base | k_base | v_base], each with kStages copies.
  extern __shared__ __align__(1024) Element shm[];
  Element* q_base = shm;
  Element* k_base = q_base + kStagesQK * kQChunkElements;
  Element* v_base = k_base + kStagesQK * kKChunkElements;

  __shared__ uint64_t qk_full[kStagesQK];
  __shared__ uint64_t qk_empty[kStagesQK];
  __shared__ uint64_t v_full[kStagesPV];
  __shared__ uint64_t v_empty[kStagesPV];

  // Barrier roles:
  //   *_full  (TmaBarrier, init=1):   producer→consumer. The `1` is the single
  //     TMA-issuing thread (tid=0) that arrives via arrive_and_expect_tx(bytes)
  //     once its TMA writes land; consumers block on wait(*_full, phase).
  //   *_empty (CtaBarrier, init=kNumThreads): consumer→producer. The
  //     kNumThreads arrivals = every consumer thread has finished reading the
  //     stage; the next producer TMA blocks on wait(*_empty, phase) so it can
  //     safely overwrite that stage's smem.
  //   wait(bar, phase): phase is a 1-bit flip counter for ping-pong reuse of
  //     the SAME stage slot across passes; producer and consumer must pass
  //     matching phases so a stale arrival from pass N-1 cannot release
  //     pass N early. phase = (chunk_index / kStages) & 1 flips every kStages
  //     chunks (0,0,...,0 | 1,1,...,1 | 0,...).
  if (tid == 0) {
    for (int s = 0; s < kStagesQK; ++s) {
      TmaBarrier::init(&qk_full[s], 1);
      CtaBarrier::init(&qk_empty[s], kNumThreads);
    }
    for (int s = 0; s < kStagesPV; ++s) {
      TmaBarrier::init(&v_full[s], 1);
      CtaBarrier::init(&v_empty[s], kNumThreads);
    }
  }
  __syncthreads();

  // TMA tensor views: 2D [total_rows, kHeadDim] with row-major stride. The
  // per-head row origin (q_row_offset / kv_row_offset) is injected via
  // domain_offset so the TMA row coordinate equals head*N + tile*kBr, which
  // stays correct when N % kBr != 0 (non-aligned). The tile coordinate passed
  // to local_tile below is then the per-head local tile id (Q_tile_id /
  // kv_tile_idx), not a head-folded global tile index. Out-of-range rows on
  // the last tile are zero-padded by the TMA descriptor (K tail is masked to
  // -inf in softmax; Q/O tail rows are dropped by the store predicate).
  // domain_offset(coord, layout) returns (same_layout, layout(coord)) -- i.e.
  // the SAME tensor layout with its origin pointer advanced by coord, so the
  // TMA row coordinate equals head*N + tile*kBr without re-wrapping the TMA
  // tensor descriptor. local_tile below then uses per-head local tile ids.
  auto mQ = domain_offset(
      make_coord(q_row_offset, 0),
      tma_q.get_tma_tensor(make_shape(total_q_rows, Int<kHeadDim>{})));
  auto mK = domain_offset(
      make_coord(kv_row_offset, 0),
      tma_k.get_tma_tensor(make_shape(total_kv_rows, Int<kHeadDim>{})));
  auto mV = domain_offset(
      make_coord(kv_row_offset, 0),
      tma_v.get_tma_tensor(make_shape(total_kv_rows, Int<kHeadDim>{})));
  auto q_slice = tma_q.get_slice(_0{});
  auto k_slice = tma_k.get_slice(_0{});
  auto v_slice = tma_v.get_slice(_0{});

  // Dual TiledMma: QK uses Tile<kBr,kBc,16> (full S tile in one MMA),
  // PV uses Tile<kBr,kVDChunk,16> (output d-direction is N of MMA).
  TiledMmaQK tiled_mma_qk;
  TiledMmaPV tiled_mma_pv;
  auto thr_mma_qk = tiled_mma_qk.get_thread_slice(tid);
  auto thr_mma_pv = tiled_mma_pv.get_thread_slice(tid);

  // S2R copy atoms: LDSM_N for Q/K (A/B operands of QK GEMM),
  // LDSM_T for V (transposed B operand of PV GEMM via SmemLayoutVt).
  auto s2r_copy_q = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma_qk);
  auto s2r_copy_k = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma_qk);
  auto s2r_copy_v = make_tiled_copy_B(SmemCopyAtomTransposed{}, tiled_mma_pv);
  auto s2r_thr_q = s2r_copy_q.get_thread_slice(tid);
  auto s2r_thr_k = s2r_copy_k.get_thread_slice(tid);
  auto s2r_thr_v = s2r_copy_v.get_thread_slice(tid);

  // V fragment layout: precompute the register layout for PV B-operand
  // so we can reinterpret raw LDSM_T data without extra copies.
  // sVt0_ns uses get_nonswizzle_portion ONLY to derive which register slots
  // each thread holds (partition_fragment_B's thread↔data map); the register
  // layout is swizzle-independent. V's smem bank conflicts are handled by the
  // TMA write (SmemLayoutV has swizzle) + the ldmatrix read (partition_S on
  // the swizzled sVt inside the PV loop applies swizzle). Doing
  // partition_fragment_B on the swizzled sVt0 would conflict the LDSM_T
  // thread mapping with the swizzle composition.
  // Ref: flash-attention/csrc/flash_attn/src/kernel_traits.h
  //      SmemLayoutVtransposedNoSwizzle (same trick).
  auto sV0 = make_tensor(make_smem_ptr(v_base), SmemLayoutV{});
  auto sVt0_ns =
      make_tensor(sV0.data(), get_nonswizzle_portion(SmemLayoutVt{}));
  auto tCrV_layout = thr_mma_pv.partition_fragment_B(sVt0_ns).layout();

  // OFragType/OFragLayout are compile-time aliases over partition_fragment_C
  // used ONLY to size the o_acc_storage scratch (kOElemsPerFrag) and to derive
  // kORows/kOCols for the rowcol reshape; the runtime O fragment is rebuilt
  // fresh each iteration from o_acc_storage[v_chunk] (see the PV loop).
  using OFragType = decltype(partition_fragment_C(
      tiled_mma_pv, Shape<Int<kBr>, Int<kVDChunk>>{}));
  using OFragLayout = typename OFragType::layout_type;
  constexpr int kOElemsPerFrag = decltype(size(OFragType{}))::value;
  constexpr int kORows = decltype(size<0>(
      make_tensor((float*)nullptr,
                  ffpa_cute::convert_layout_acc_rowcol(OFragLayout{}))))::value;
  constexpr int kOCols = decltype(size<1>(
      make_tensor((float*)nullptr,
                  ffpa_cute::convert_layout_acc_rowcol(OFragLayout{}))))::value;

  auto cS = make_identity_tensor(Shape<Int<kBr>, Int<kBc>>{});
  auto tScS = thr_mma_qk.partition_C(cS);
  auto tScS_rc = make_tensor(
      tScS.data(), ffpa_cute::convert_layout_acc_rowcol(tScS.layout()));
  constexpr int kSRows = decltype(size<0>(tScS_rc))::value;
  constexpr int kSCols = decltype(size<1>(tScS_rc))::value;

  // Online softmax below uses exp2f, which requires the scale in log2 domain:
  // exp(x) == exp2(x * log2(e)). The caller passes the linear-domain scale
  // (1/sqrt(D)); convert it once here so exp2f(scores*scale - max) is correct.
  // (This was the accuracy bug: without log2(e) the P and row_scale were
  // 2^(...) instead of e^(...), compounding across kv_tiles.)
  const float inv_scale = 1.0f / scale;
  scale *= FFPA_M_LOG2E;

  float row_max[kORows];
  float row_sum[kORows];
#pragma unroll
  for (int r = 0; r < kORows; ++r) {
    row_max[r] = -INFINITY;
    row_sum[r] = 0.0f;
  }

  float o_acc_storage[kDChunksV][kOElemsPerFrag];
#pragma unroll
  for (int v = 0; v < kDChunksV; ++v)
#pragma unroll
    for (int i = 0; i < kOElemsPerFrag; ++i)
      o_acc_storage[v][i] = 0.0f;

  // Signal all stages empty so tid=0 can issue initial TMA prefetch.
  for (int s = 0; s < kStagesQK; ++s)
    CtaBarrier::arrive(&qk_empty[s]);
  for (int s = 0; s < kStagesPV; ++s)
    CtaBarrier::arrive(&v_empty[s]);

  // TMA load helpers: tid=0 issues Q+K (or V) TMA copies with
  // arrive_and_expect_tx on the full barrier for the target stage.
  auto issue_qk_tma = [&](int d_chunk, int stage, int kv_tile_idx) {
    auto sQ = make_tensor(make_smem_ptr(q_base + stage * kQChunkElements),
                          SmemLayoutQ{});
    auto sK = make_tensor(make_smem_ptr(k_base + stage * kKChunkElements),
                          SmemLayoutK{});
    auto gQ = local_tile(mQ, Shape<Int<kBr>, Int<kQKDChunk>>{},
                         make_coord(Q_tile_id, d_chunk));
    auto gK = local_tile(mK, Shape<Int<kBc>, Int<kQKDChunk>>{},
                         make_coord(kv_tile_idx, d_chunk));
    auto tQgQ = q_slice.partition_S(gQ);
    auto tQsQ = q_slice.partition_D(sQ);
    auto tKgK = k_slice.partition_S(gK);
    auto tKsK = k_slice.partition_D(sK);
    TmaBarrier::arrive_and_expect_tx(&qk_full[stage],
                                     sizeof(Element) * (size(sQ) + size(sK)));
    copy(tma_q.with(qk_full[stage]), tQgQ, tQsQ);
    copy(tma_k.with(qk_full[stage]), tKgK, tKsK);
  };

  auto issue_v_tma = [&](int v_chunk, int stage, int kv_tile_idx) {
    auto sV = make_tensor(make_smem_ptr(v_base + stage * kVChunkElements),
                          SmemLayoutV{});
    auto gV = local_tile(mV, Shape<Int<kBc>, Int<kVDChunk>>{},
                         make_coord(kv_tile_idx, v_chunk));
    auto tVgV = v_slice.partition_S(gV);
    auto tVsV = v_slice.partition_D(sV);
    TmaBarrier::arrive_and_expect_tx(&v_full[stage],
                                     sizeof(Element) * size(sV));
    copy(tma_v.with(v_full[stage]), tVgV, tVsV);
  };

  // Initial QK prefetch: fill pipeline with first kStagesQK chunks.
  if (tid == 0) {
    for (int d = 0; d < kStagesQK && d < kDChunksQK; ++d) {
      CtaBarrier::wait(&qk_empty[d], 0);
      issue_qk_tma(d, d, 0);
    }
  }

  // Initial V prefetch for kv_tile 0: issue first kStagesPV V chunks so the
  // V TMA overlaps the entire first QK GEMM + softmax window (V is independent
  // of QK, so it can be launched before the QK loop).
  // v_stage = chunk_index % kStagesPV (the smem slot), v_phase flips 0→1→0
  //   every kStagesPV chunks PER slot so a slot's pass N arrival can't be
  //   mistaken for its pass N-1 arrival. Example kStagesPV=2, kDChunksV=2:
  //   chunk 0→slot0 phase0, 1→slot1 phase0, 2→slot0 phase1, 3→slot1 phase1,
  //   4→slot0 phase0, ... (phase flips 0,0,1,1,0,0 across chunk_index).
  if (tid == 0) {
    for (int v = 0; v < kStagesPV && v < kDChunksV; ++v) {
      const int chunk_index = v;  // kv_tile == 0
      const int v_stage = chunk_index % kStagesPV;
      const int v_phase = (chunk_index / kStagesPV) & 1;
      CtaBarrier::wait(&v_empty[v_stage], v_phase);
      issue_v_tma(v, v_stage, 0);
    }
  }

#pragma unroll 1
  for (int kv_tile = 0; kv_tile < Tc_eff; ++kv_tile) {
    // V prefetch for kv_tile > 0: issue first kStagesPV V chunks before the
    // QK loop so the V TMA overlaps QK GEMM + softmax. (kv_tile 0's V initial
    // was issued before the loop; subsequent kv_tiles' QK initial is issued at
    // the end of the previous kv_tile's QK loop, see below.)
    if (kv_tile > 0 && tid == 0) {
      for (int v = 0; v < kStagesPV && v < kDChunksV; ++v) {
        const int chunk_index = kv_tile * kDChunksV + v;
        const int v_stage = chunk_index % kStagesPV;
        const int v_phase = (chunk_index / kStagesPV) & 1;
        CtaBarrier::wait(&v_empty[v_stage], v_phase);
        issue_v_tma(v, v_stage, kv_tile);
      }
    }

    // Phase 1: QK GEMM with split-D accumulation.
    // S[Br,Bc] = sum_{d=0}^{kDChunksQK-1} Q_d @ K_d^T
    auto tCrS = partition_fragment_C(tiled_mma_qk, Shape<Int<kBr>, Int<kBc>>{});
    clear(tCrS);

#pragma unroll
    for (int d_chunk = 0; d_chunk < kDChunksQK; ++d_chunk) {
      // Wait for TMA data, fence, then gemm_ss (smem→regs→MMA).
      // TmaBarrier::wait(qk_full[stage], phase): consumers block until tid=0's
      // arrive_and_expect_tx for this stage's Q+K TMA lands.
      const int chunk_index = kv_tile * kDChunksQK + d_chunk;
      const int stage = chunk_index % kStagesQK;
      const int phase = (chunk_index / kStagesQK) & 1;
      TmaBarrier::wait(&qk_full[stage], phase);
      cutlass::arch::fence_view_async_shared();

      auto sQ = make_tensor(make_smem_ptr(q_base + stage * kQChunkElements),
                            SmemLayoutQ{});
      auto sK = make_tensor(make_smem_ptr(k_base + stage * kKChunkElements),
                            SmemLayoutK{});
      auto tCrQ = thr_mma_qk.partition_fragment_A(sQ);
      auto tCrK = thr_mma_qk.partition_fragment_B(sK);
      auto tQsQ = s2r_thr_q.partition_S(sQ);
      auto tKsK = s2r_thr_k.partition_S(sK);

      ffpa_cute::gemm_ss(tCrS, tCrQ, tCrK, tQsQ, tKsK, tiled_mma_qk, s2r_copy_q,
                         s2r_copy_k, s2r_thr_q, s2r_thr_k);

      // Signal stage consumed; tid=0 prefetches next chunk if available.
      // CtaBarrier::arrive(qk_empty[stage]): each consumer thread arrives once
      // it has finished reading stage's smem; once all kNumThreads arrive, the
      // producer's wait(qk_empty[stage], phase_next) unblocks to overwrite it.
      CtaBarrier::arrive(&qk_empty[stage]);

      // Cannot move this prefetch before gemm_ss: s_next == stage and
      // phase_next == 1-phase, so the wait below gates on THIS iter's
      // arrive(qk_empty[stage]) — moving earlier deadlocks (tid=0 would
      // stall waiting for an arrive that needs tid=0 inside gemm_ss).
      if (tid == 0) {
        const int d_next = d_chunk + kStagesQK;
        if (d_next < kDChunksQK) {
          const int next_index = kv_tile * kDChunksQK + d_next;
          const int s_next = next_index % kStagesQK;
          const int phase_next = (next_index / kStagesQK) & 1;
          CtaBarrier::wait(&qk_empty[s_next], phase_next);
          issue_qk_tma(d_next, s_next, kv_tile);
        }
      }
    }

    // Prefetch next kv_tile's QK initial chunks so the QK TMA overlaps this
    // kv_tile's softmax + PV loop. The QK barriers are disjoint from the
    // softmax/PV barriers, so placing this here is safe (zero-deadlock by the
    // disjoint-barrier-set invariant). kv_tile 0's QK initial was issued
    // before the loop; this replaces the old "kv_tile > 0 top" QK prefetch.
    if (kv_tile < Tc_eff - 1 && tid == 0) {
      for (int d = 0; d < kStagesQK && d < kDChunksQK; ++d) {
        const int chunk_index = (kv_tile + 1) * kDChunksQK + d;
        const int stage = chunk_index % kStagesQK;
        const int phase = (chunk_index / kStagesQK) & 1;
        CtaBarrier::wait(&qk_empty[stage], phase);
        issue_qk_tma(d, stage, kv_tile + 1);
      }
    }

    // Phase 2: Online softmax.
    // Layout transform: MMA C-fragment → [kORows, kSCols] rowcol view.
    {
      auto scores = make_tensor(
          tCrS.data(), ffpa_cute::convert_layout_acc_rowcol(tCrS.layout()));
      float row_scale[kORows];

      // Boundary masking: -inf for OOB KV positions.
      {
        const int kv_valid = Nkv - kv_tile * kBc;
        if (kv_valid < kBc) {
#pragma unroll
          for (int row = 0; row < kSRows; ++row)
#pragma unroll
            for (int col = 0; col < kSCols; ++col) {
              if (get<1>(tScS_rc(row, col)) >= kv_valid)
                scores(row, col) = -INFINITY;
            }
        }
      }

      // Causal masking: -inf where k_pos > q_pos.
      if (kv_tile >= mask_start_tile) {
#pragma unroll
        for (int row = 0; row < kSRows; ++row) {
          const int q_pos = Br_base + get<0>(tScS_rc(row, 0)) + kv_offset;
#pragma unroll
          for (int col = 0; col < kSCols; ++col) {
            const int k_pos = kv_tile * kBc + get<1>(tScS_rc(row, col));
            if (k_pos > q_pos)
              scores(row, col) = -INFINITY;
          }
        }
      }

      // Additive attention bias (pre-softmax, separate pass).
      // NOTE: attn_bias/dropout on CuTe kernel is ~3x slower than the non-WS
      // TMA (../fwd_sm120.cuh) template kernel due to 1 block/SM occupancy (8
      // warps cannot hide scalar gmem load / Philox RNG latency). The launcher
      // should prefer the non-WS TMA fallback when bias/dropout is active;
      // these constexpr paths exist for correctness and future optimization
      // (e.g. vectorized bias load via TMA).
      if constexpr (kHasAttnBias) {
        ffpa_cute::apply_attn_bias_rowcol<decltype(scores), decltype(tScS_rc),
                                          kSRows, kSCols>(
            scores, tScS_rc, attn_bias, attn_bias_dtype, attn_bias_stride_b,
            attn_bias_stride_h, attn_bias_stride_m, attn_bias_stride_n, Nb_id,
            Nh_id, Br_base, kv_tile, kBc, inv_scale);
      }

      // Row-max + exp2 + row-sum (warp-level reduction via shfl_xor).
      ffpa_cute::online_safe_softmax<decltype(scores), decltype(tScS_rc),
                                     kORows>(scores, tScS_rc, scale, row_max,
                                             row_sum, row_scale);

      // Dropout on P (post-softmax, pre-PV, separate pass).
      if constexpr (kHasDropout) {
        ffpa_cute::apply_dropout_rowcol<decltype(scores), decltype(tScS_rc),
                                        kORows, kSCols>(
            scores, tScS_rc, dropout_p, philox_seed, philox_offset, Nb_id, Nh,
            Nh_id, Nq, Nkv, Br_base, kv_tile, kBc);
      }

      // P fragment: convert fp32 scores → Element, then reinterpret
      // C-layout as A-operand registers for PV GEMM (zero-copy reuse).
      auto tCrP = ffpa_cute::convert_type<Element>(tCrS);
      auto tCrPv = make_tensor(
          tCrP.data(),
          ffpa_cute::convert_layout_acc_Aregs<TiledMmaPV>(tCrP.layout()));

      // (V initial prefetch moved to the top of the kv_tile loop so it overlaps
      // QK GEMM + softmax; see the kv_tile > 0 block and the pre-loop block.)

#pragma unroll
      for (int v_chunk = 0; v_chunk < kDChunksV; ++v_chunk) {
        // Wait for V TMA data, fence, then prepare V smem view.
        const int chunk_index = kv_tile * kDChunksV + v_chunk;
        const int v_stage = chunk_index % kStagesPV;
        const int v_phase = (chunk_index / kStagesPV) & 1;
        TmaBarrier::wait(&v_full[v_stage], v_phase);
        cutlass::arch::fence_view_async_shared();

        auto sV = make_tensor(make_smem_ptr(v_base + v_stage * kVChunkElements),
                              SmemLayoutV{});
        auto sVt = make_tensor(sV.data(), SmemLayoutVt{});
        auto tCrVStorage = thr_mma_pv.partition_fragment_B(sV);
        auto tCrV = make_tensor(tCrVStorage.data(), tCrV_layout);
        auto tVsVt = s2r_thr_v.partition_S(sVt);

        // O rescaling: multiply accumulated O by row_scale (kv_tile > 0).
        auto tCrO = make_tensor(make_rmem_ptr(&o_acc_storage[v_chunk][0]),
                                OFragLayout{});
        if (kv_tile > 0) {
          auto tCrO_rc = make_tensor(
              tCrO.data(), ffpa_cute::convert_layout_acc_rowcol(tCrO.layout()));
#pragma unroll
          for (int row = 0; row < kORows; ++row)
#pragma unroll
            for (int col = 0; col < kOCols; ++col)
              tCrO_rc(row, col) *= row_scale[row];
        }

        // gemm_rs: P (register A) @ V (smem B via LDSM_T) → O (register C).
        ffpa_cute::gemm_rs(tCrO, tCrPv, tCrV, tVsVt, tiled_mma_pv, s2r_copy_v,
                           s2r_thr_v);

        // Signal stage consumed; tid=0 prefetches next chunk if available.
        CtaBarrier::arrive(&v_empty[v_stage]);

        // Cannot move this prefetch before gemm_rs: s_next == v_stage and
        // phase_next == 1-v_phase, so the wait below gates on THIS iter's
        // arrive(v_empty[v_stage]) — moving earlier deadlocks (tid=0 would
        // stall waiting for an arrive that needs tid=0 inside gemm_rs).
        if (tid == 0) {
          const int v_next = v_chunk + kStagesPV;
          if (v_next < kDChunksV) {
            const int next_index = kv_tile * kDChunksV + v_next;
            const int s_next = next_index % kStagesPV;
            const int phase_next = (next_index / kStagesPV) & 1;
            CtaBarrier::wait(&v_empty[s_next], phase_next);
            issue_v_tma(v_next, s_next, kv_tile);
          }
        }
      }
    }
  }

  // Phase 4: Epilogue. Normalize O by 1/row_sum, convert to Element, store.
  //   aligned tile: batched R->S(stmatrix)->swizzled smem->TMA store.
  //   kVChunksPerBatch
  //     v_chunks staged in shm (reusing freed QKV smem), TMA stores batched
  //     into one bulk group (one arrive + one wait per batch), reducing wait
  //     count from kDChunksV to kNBatches. LSE write deferred to overlap last
  //     drain.
  //   tail tile: per-element predicated R->G (unchanged, zero risk).
  {
    constexpr int kVChunksPerBatch = Traits::kVChunksPerBatch;
    constexpr int kNBatches = Traits::kNBatches;
    constexpr int kOTileElems = cosize(SmemLayoutO{});

    __syncthreads();  // V smem reads done before R->S overwrites shm

    auto mO_tma = domain_offset(
        make_coord(q_row_offset, 0),
        tma_o.get_tma_tensor(make_shape(total_q_rows, Int<kHeadDim>{})));
    auto o_slice = tma_o.get_slice(_0{});

    auto r2s_copy = make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N, Element>{},
                                      tiled_mma_pv);
    auto r2s_thr = r2s_copy.get_slice(tid);

    const int O_gmem_offset =
        (Nb_id * Nh * Nq * kHeadDim) + (Nh_id * Nq * kHeadDim);
    auto mO = make_tensor(make_gmem_ptr(O + O_gmem_offset),
                          make_shape(Nq, Int<kHeadDim>{}),
                          make_stride(Int<kHeadDim>{}, _1{}));
    auto cO = make_identity_tensor(Shape<Int<kBr>, Int<kVDChunk>>{});
    auto tOcO = thr_mma_pv.partition_C(cO);

    if (Br_base + kBr <= Nq) {
      // aligned: batched R->S->G via TMA store
#pragma unroll
      for (int batch = 0; batch < kNBatches; ++batch) {
        // R->S: stage kVChunksPerBatch v_chunks into disjoint shm regions
#pragma unroll
        for (int v_in = 0; v_in < kVChunksPerBatch; ++v_in) {
          int v_chunk = batch * kVChunksPerBatch + v_in;
          auto tCrO = make_tensor(make_rmem_ptr(&o_acc_storage[v_chunk][0]),
                                  OFragLayout{});
          auto tCrO_rc = make_tensor(
              tCrO.data(), ffpa_cute::convert_layout_acc_rowcol(tCrO.layout()));
#pragma unroll
          for (int row = 0; row < kORows; ++row) {
            const float inv_sum = 1.0f / row_sum[row];
#pragma unroll
            for (int col = 0; col < kOCols; ++col)
              tCrO_rc(row, col) *= inv_sum;
          }
          auto tCrOHalf = ffpa_cute::convert_type<Element>(tCrO);
          auto sO_v = make_tensor(make_smem_ptr(shm + v_in * kOTileElems),
                                  SmemLayoutO{});
          auto tCrOHalf_src = r2s_thr.retile_S(tCrOHalf);
          auto tCsO_dst = r2s_thr.partition_D(sO_v);
          copy(r2s_copy, tCrOHalf_src, tCsO_dst);
        }
        cutlass::arch::fence_view_async_shared();
        __syncthreads();
        // TMA stores: issue all v_chunks in this batch into one bulk group
#pragma unroll
        for (int v_in = 0; v_in < kVChunksPerBatch; ++v_in) {
          int v_chunk = batch * kVChunksPerBatch + v_in;
          auto sO_v = make_tensor(make_smem_ptr(shm + v_in * kOTileElems),
                                  SmemLayoutO{});
          auto gO_tma = local_tile(mO_tma, Shape<Int<kBr>, Int<kVDChunk>>{},
                                   make_coord(Q_tile_id, v_chunk));
          auto tCgO_tma = o_slice.partition_D(gO_tma);
          auto tOsO = o_slice.partition_S(sO_v);
          if (tid == 0) {
            copy(tma_o, tOsO, tCgO_tma);
          }
        }
        tma_store_arrive();
        if (batch < kNBatches - 1)
          tma_store_wait<0>();  // drain for shm reuse
      }
    } else {
      // tail: per-element predicated R->G (unchanged)
#pragma unroll
      for (int v_chunk = 0; v_chunk < kDChunksV; ++v_chunk) {
        auto tCrO = make_tensor(make_rmem_ptr(&o_acc_storage[v_chunk][0]),
                                OFragLayout{});
        auto tCrO_rc = make_tensor(
            tCrO.data(), ffpa_cute::convert_layout_acc_rowcol(tCrO.layout()));
#pragma unroll
        for (int row = 0; row < kORows; ++row) {
          const float inv_sum = 1.0f / row_sum[row];
#pragma unroll
          for (int col = 0; col < kOCols; ++col)
            tCrO_rc(row, col) *= inv_sum;
        }
        auto tCrOHalf = ffpa_cute::convert_type<Element>(tCrO);
        auto gO = local_tile(mO, Shape<Int<kBr>, Int<kVDChunk>>{},
                             make_coord(Q_tile_id, v_chunk));
        auto tCgO = thr_mma_pv.partition_C(gO);
#pragma unroll
        for (int i = 0; i < size(tCrOHalf); ++i) {
          const int global_row = Br_base + get<0>(tOcO(i));
          if (global_row < Nq)
            tCgO(i) = tCrOHalf(i);
        }
      }
    }

    // LSE write: overlaps last batch's TMA drain (aligned) or serial (tail).
    if (softmax_lse != nullptr) {
      const int lse_base = Nb_id * Nh * Nq + Nh_id * Nq;
#pragma unroll
      for (int row = 0; row < kORows; ++row) {
        const float lse = (row_max[row] + log2f(row_sum[row])) * FFPA_M_LN2;
        const int global_row = Br_base + get<0>(tScS_rc(row, 0));
        if (global_row < Nq)
          softmax_lse[lse_base + global_row] = lse;
      }
    }

    // Final drain: only if TMA stores were issued (aligned path).
    if (Br_base + kBr <= Nq)
      tma_store_wait<0>();
  }
}

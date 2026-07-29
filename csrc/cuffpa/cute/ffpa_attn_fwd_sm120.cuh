#pragma once

#include "attn_traits.cuh"

#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cutlass.h>
#include <cutlass/device_kernel.h>

using TmaBarrier = cutlass::arch::ClusterTransactionBarrier;
using CtaBarrier = cutlass::arch::ClusterBarrier;

template <typename Traits, typename TmaQ, typename TmaK, typename TmaV,
          int kStagesQK = 2, int kStagesPV = 2>
__global__ void __launch_bounds__(Traits::kNumThreads, 1)
    ffpa_attn_split_d_fwd_cute_sm120(CUTLASS_GRID_CONSTANT TmaQ const tma_q,
                                     CUTLASS_GRID_CONSTANT TmaK const tma_k,
                                     CUTLASS_GRID_CONSTANT TmaV const tma_v,
                                     typename Traits::Element* __restrict__ O,
                                     float* __restrict__ softmax_lse, int Nq,
                                     int Nkv, int Nh, int Nh_kv, float scale,
                                     int Tc, int causal, int total_q_rows,
                                     int total_kv_rows) {
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
  // Layout transforms:
  //   convert_layout_acc_rowcol: MMA C-fragment → [rows, cols] for softmax.
  //   convert_layout_acc_Aregs:  MMA C-fragment → A-operand regs for PV
  //     (reuses P registers as MMA-A without data movement).
  //   SmemLayoutVt: transposed V layout for gemm_rs B-operand (LDSM_T).

  using namespace cute;
  using Element = typename Traits::Element;
  using SmemLayoutQ = typename Traits::SmemLayoutQ;
  using SmemLayoutK = typename Traits::SmemLayoutK;
  using SmemLayoutV = typename Traits::SmemLayoutV;
  using SmemLayoutVt = typename Traits::SmemLayoutVt;
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

  constexpr int kQChunkElements = cosize(SmemLayoutQ{});
  constexpr int kKChunkElements = cosize(SmemLayoutK{});
  constexpr int kVChunkElements = cosize(SmemLayoutV{});

  const int Nb_id = blockIdx.y / Nh;
  const int Nh_id = blockIdx.y % Nh;
  const int Q_tile_id = blockIdx.x;
  const int group_size = Nh / Nh_kv;
  const int kv_head_idx = Nh_id / group_size;
  const int Br_base = Q_tile_id * kBr;

  if (Br_base >= Nq)
    return;

  const int kv_offset = Nkv - Nq;
  const int causal_thresh_row0 = Br_base + kv_offset;
  const int Tc_eff =
      causal ? min(Tc, ((Br_base + kBr - 1 + kv_offset) / kBc) + 1) : Tc;
  const int mask_start_tile =
      causal ? max(0, (causal_thresh_row0 + 1) / kBc) : INT_MAX;

  const int q_tile = (Nb_id * Nh + Nh_id) * ((Nq + kBr - 1) / kBr) + Q_tile_id;
  const int kv_tiles_total = (Nkv + kBc - 1) / kBc;
  const int kv_base = (Nb_id * Nh_kv + kv_head_idx) * kv_tiles_total;

  // SMEM layout: [q_base | k_base | v_base], each with kStages copies.
  extern __shared__ __align__(1024) Element shm[];
  Element* q_base = shm;
  Element* k_base = q_base + kStagesQK * kQChunkElements;
  Element* v_base = k_base + kStagesQK * kKChunkElements;

  __shared__ uint64_t qk_full[kStagesQK];
  __shared__ uint64_t qk_empty[kStagesQK];
  __shared__ uint64_t v_full[kStagesPV];
  __shared__ uint64_t v_empty[kStagesPV];

  // Barrier init: qk_full/v_full are TmaBarriers (tid=0 arrive_expect_tx),
  // qk_empty/v_empty are CtaBarriers (all threads arrive after consume).
  if (threadIdx.x == 0) {
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

  // TMA tensor views: 2D [total_rows, kHeadDim] with row-major stride.
  auto mQ = tma_q.get_tma_tensor(make_shape(total_q_rows, Int<kHeadDim>{}));
  auto mK = tma_k.get_tma_tensor(make_shape(total_kv_rows, Int<kHeadDim>{}));
  auto mV = tma_v.get_tma_tensor(make_shape(total_kv_rows, Int<kHeadDim>{}));
  auto q_slice = tma_q.get_slice(_0{});
  auto k_slice = tma_k.get_slice(_0{});
  auto v_slice = tma_v.get_slice(_0{});

  // Dual TiledMma: QK uses Tile<kBr,kBc,16> (full S tile in one MMA),
  // PV uses Tile<kBr,kVDChunk,16> (output d-direction is N of MMA).
  TiledMmaQK tiled_mma_qk;
  TiledMmaPV tiled_mma_pv;
  auto thr_mma_qk = tiled_mma_qk.get_thread_slice(threadIdx.x);
  auto thr_mma_pv = tiled_mma_pv.get_thread_slice(threadIdx.x);

  // S2R copy atoms: LDSM_N for Q/K (A/B operands of QK GEMM),
  // LDSM_T for V (transposed B operand of PV GEMM via SmemLayoutVt).
  auto s2r_copy_q = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma_qk);
  auto s2r_copy_k = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma_qk);
  auto s2r_copy_v = make_tiled_copy_B(SmemCopyAtomTransposed{}, tiled_mma_pv);
  auto s2r_thr_q = s2r_copy_q.get_thread_slice(threadIdx.x);
  auto s2r_thr_k = s2r_copy_k.get_thread_slice(threadIdx.x);
  auto s2r_thr_v = s2r_copy_v.get_thread_slice(threadIdx.x);

  // V fragment layout: precompute the register layout for PV B-operand
  // so we can reinterpret raw LDSM_T data without extra copies.
  auto sV0 = make_tensor(make_smem_ptr(v_base), SmemLayoutV{});
  auto sVt0_ns =
      make_tensor(sV0.data(), get_nonswizzle_portion(SmemLayoutVt{}));
  auto tCrV_layout = thr_mma_pv.partition_fragment_B(sVt0_ns).layout();

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
                         make_coord(q_tile, d_chunk));
    auto gK = local_tile(mK, Shape<Int<kBc>, Int<kQKDChunk>>{},
                         make_coord(kv_base + kv_tile_idx, d_chunk));
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
                         make_coord(kv_base + kv_tile_idx, v_chunk));
    auto tVgV = v_slice.partition_S(gV);
    auto tVsV = v_slice.partition_D(sV);
    TmaBarrier::arrive_and_expect_tx(&v_full[stage],
                                     sizeof(Element) * size(sV));
    copy(tma_v.with(v_full[stage]), tVgV, tVsV);
  };

  // Initial QK prefetch: fill pipeline with first kStagesQK chunks.
  if (threadIdx.x == 0) {
    for (int d = 0; d < kStagesQK && d < kDChunksQK; ++d) {
      CtaBarrier::wait(&qk_empty[d], 0);
      issue_qk_tma(d, d, 0);
    }
  }

#pragma unroll 1
  for (int kv_tile = 0; kv_tile < Tc_eff; ++kv_tile) {
    // QK prefetch for kv_tile > 0: issue first kStagesQK chunks.
    if (kv_tile > 0 && threadIdx.x == 0) {
      for (int d = 0; d < kStagesQK && d < kDChunksQK; ++d) {
        const int chunk_index = kv_tile * kDChunksQK + d;
        const int stage = chunk_index % kStagesQK;
        const int phase = (chunk_index / kStagesQK) & 1;
        CtaBarrier::wait(&qk_empty[stage], phase);
        issue_qk_tma(d, stage, kv_tile);
      }
    }

    // Phase 1: QK GEMM with split-D accumulation.
    // S[Br,Bc] = sum_{d=0}^{kDChunksQK-1} Q_d @ K_d^T
    auto tCrS = partition_fragment_C(tiled_mma_qk, Shape<Int<kBr>, Int<kBc>>{});
    clear(tCrS);

#pragma unroll
    for (int d_chunk = 0; d_chunk < kDChunksQK; ++d_chunk) {
      // Wait for TMA data, fence, then gemm_ss (smem→regs→MMA).
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
      CtaBarrier::arrive(&qk_empty[stage]);

      if (threadIdx.x == 0) {
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

      // Row-max + exp2 + row-sum (warp-level reduction via shfl_xor).
      // row_scale = exp2(old_max - new_max) for O rescaling.
#pragma unroll
      for (int row = 0; row < kORows; ++row) {
        float tile_max = -INFINITY;
#pragma unroll
        for (int col = 0; col < size<1>(scores); ++col)
          tile_max = fmaxf(tile_max, scores(row, col) * scale);
        tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 1));
        tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 2));
        const float next_max = fmaxf(row_max[row], tile_max);
        row_scale[row] = exp2f(row_max[row] - next_max);
        float tile_sum = 0.0f;
#pragma unroll
        for (int col = 0; col < size<1>(scores); ++col) {
          const float p = exp2f(scores(row, col) * scale - next_max);
          scores(row, col) = p;
          tile_sum += p;
        }
        tile_sum += __shfl_xor_sync(0xffffffff, tile_sum, 1);
        tile_sum += __shfl_xor_sync(0xffffffff, tile_sum, 2);
        row_sum[row] = row_sum[row] * row_scale[row] + tile_sum;
        row_max[row] = next_max;
      }

      // P fragment: convert fp32 scores → Element, then reinterpret
      // C-layout as A-operand registers for PV GEMM (zero-copy reuse).
      auto tCrP = ffpa_cute::convert_type<Element>(tCrS);
      auto tCrPv = make_tensor(
          tCrP.data(),
          ffpa_cute::convert_layout_acc_Aregs<TiledMmaPV>(tCrP.layout()));

      // V prefetch: issue first kStagesPV V chunks for this kv_tile.
      if (threadIdx.x == 0) {
        for (int v = 0; v < kStagesPV && v < kDChunksV; ++v) {
          const int chunk_index = kv_tile * kDChunksV + v;
          const int v_stage = chunk_index % kStagesPV;
          const int v_phase = (chunk_index / kStagesPV) & 1;
          CtaBarrier::wait(&v_empty[v_stage], v_phase);
          issue_v_tma(v, v_stage, kv_tile);
        }
      }

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

        if (threadIdx.x == 0) {
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

  // Phase 4: Epilogue. Normalize O by 1/row_sum, convert to Element,
  // and store to gmem. Aligned tiles use vectorized copy(); the last
  // partial tile (if any) uses per-element predicated store.
  {
    const int O_gmem_offset =
        (Nb_id * Nh * Nq * kHeadDim) + (Nh_id * Nq * kHeadDim);
    auto mO = make_tensor(make_gmem_ptr(O + O_gmem_offset),
                          make_shape(Nq, Int<kHeadDim>{}),
                          make_stride(Int<kHeadDim>{}, _1{}));
    auto cO = make_identity_tensor(Shape<Int<kBr>, Int<kVDChunk>>{});
    auto tOcO = thr_mma_pv.partition_C(cO);
#pragma unroll
    for (int v_chunk = 0; v_chunk < kDChunksV; ++v_chunk) {
      auto tCrO =
          make_tensor(make_rmem_ptr(&o_acc_storage[v_chunk][0]), OFragLayout{});
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
      if (Br_base + kBr <= Nq) {
        copy(tCrOHalf, tCgO);
      } else {
#pragma unroll
        for (int i = 0; i < size(tCrOHalf); ++i) {
          const int global_row = Br_base + get<0>(tOcO(i));
          if (global_row < Nq)
            tCgO(i) = tCrOHalf(i);
        }
      }
    }
  }

  // Optional: write log-sum-exp for backward pass compatibility.
  if (softmax_lse != nullptr) {
    const int lse_base = Nb_id * Nh * Nq + Nh_id * Nq;
#pragma unroll
    for (int row = 0; row < kORows; ++row) {
      const float lse = row_max[row] + log2f(row_sum[row]);
      const int global_row = Br_base + get<0>(tScS_rc(row, 0));
      if (global_row < Nq)
        softmax_lse[lse_base + global_row] = lse;
    }
  }
}

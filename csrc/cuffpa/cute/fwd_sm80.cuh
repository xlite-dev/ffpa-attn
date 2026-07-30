#pragma once

#include "attn_traits.cuh"

#include <cute/atom/copy_atom.hpp>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>

#include "common.cuh"
#include "attn_bias.cuh"
#include "dropout.cuh"

// Split-D Flash Attention forward (cp.async, sm_80+).
// Same algorithm as the TMA version (fwd_sm120.cuh) but uses cooperative
// cp.async G2S (all 256 threads) instead of TMA (tid=0 only).
template <typename Traits, int kStagesQK = 2, int kStagesPV = 2,
          int kHasAttnBias = 0, int kHasDropout = 0>
__global__ void __launch_bounds__(Traits::kNumThreads, 1)
    ffpa_attn_split_d_fwd_cute(
        typename Traits::Element* __restrict__ Q,
        typename Traits::Element* __restrict__ K,
        typename Traits::Element* __restrict__ V,
        typename Traits::Element* __restrict__ O,
        float* __restrict__ softmax_lse, int Nq, int Nkv, int Nh, int Nh_kv,
        float scale, int Tc, int causal,
        const void* __restrict__ attn_bias = nullptr, int attn_bias_dtype = 0,
        long long attn_bias_stride_b = 0, long long attn_bias_stride_h = 0,
        long long attn_bias_stride_m = 0, long long attn_bias_stride_n = 0,
        float dropout_p = 0.0f, unsigned long long philox_seed = 0,
        unsigned long long philox_offset = 0) {
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
  const int tid = threadIdx.x;

  if (Br_base >= Nq)
    return;

  const int kv_offset = Nkv - Nq;
  const int causal_thresh_row0 = Br_base + kv_offset;
  const int Tc_eff =
      causal ? min(Tc, ((Br_base + kBr - 1 + kv_offset) / kBc) + 1) : Tc;
  const int mask_start_tile =
      causal ? max(0, (causal_thresh_row0 + 1) / kBc) : INT_MAX;

  // Per-head gmem row origins (replaces TMA domain_offset).
  const int q_row_offset = (Nb_id * Nh + Nh_id) * Nq;
  const int kv_row_offset = (Nb_id * Nh_kv + kv_head_idx) * Nkv;

  // SMEM: per-stage base addresses + 2D layout (no stride-0 3D stage-mode).
  extern __shared__ __align__(1024) Element shm[];
  Element* q_base = shm;
  Element* k_base = q_base + kStagesQK * kQChunkElements;
  Element* v_base = k_base + kStagesQK * kKChunkElements;

  // G2S TiledCopy: 256 threads, 128-bit cp.async.
  // Separate copies for QK (kQKDChunk-wide) and V (kVDChunk-wide) tiles.
  using G2SCopyOp = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
  using G2SCopyAtom = Copy_Atom<Copy_Traits<G2SCopyOp>, Element>;
  constexpr int kG2SThrN_QK = kQKDChunk / 8;
  constexpr int kG2SThrM_QK = kNumThreads / kG2SThrN_QK;
  constexpr int kG2SThrN_V = kVDChunk / 8;
  constexpr int kG2SThrM_V = kNumThreads / kG2SThrN_V;
  using G2SCopyQK = decltype(make_tiled_copy(
      G2SCopyAtom{},
      make_layout(make_shape(Int<kG2SThrM_QK>{}, Int<kG2SThrN_QK>{}),
                  make_stride(Int<kG2SThrN_QK>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<8>{}))));
  using G2SCopyV = decltype(make_tiled_copy(
      G2SCopyAtom{},
      make_layout(make_shape(Int<kG2SThrM_V>{}, Int<kG2SThrN_V>{}),
                  make_stride(Int<kG2SThrN_V>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<8>{}))));
  G2SCopyQK g2s_copy_qk;
  G2SCopyV g2s_copy_v;
  auto g2s_thr_qk = g2s_copy_qk.get_slice(tid);
  auto g2s_thr_v = g2s_copy_v.get_slice(tid);

  // Gmem tensors: [total_rows, kHeadDim] row-major.
  auto mQ = make_tensor(make_gmem_ptr(Q + q_row_offset * kHeadDim),
                        make_shape(Nq, Int<kHeadDim>{}),
                        make_stride(Int<kHeadDim>{}, _1{}));
  auto mK = make_tensor(make_gmem_ptr(K + kv_row_offset * kHeadDim),
                        make_shape(Nkv, Int<kHeadDim>{}),
                        make_stride(Int<kHeadDim>{}, _1{}));
  auto mV = make_tensor(make_gmem_ptr(V + kv_row_offset * kHeadDim),
                        make_shape(Nkv, Int<kHeadDim>{}),
                        make_stride(Int<kHeadDim>{}, _1{}));

  // Dual TiledMma (same as TMA version).
  TiledMmaQK tiled_mma_qk;
  TiledMmaPV tiled_mma_pv;
  auto thr_mma_qk = tiled_mma_qk.get_thread_slice(tid);
  auto thr_mma_pv = tiled_mma_pv.get_thread_slice(tid);

  // S2R copy atoms.
  auto s2r_copy_q = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma_qk);
  auto s2r_copy_k = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma_qk);
  auto s2r_copy_v = make_tiled_copy_B(SmemCopyAtomTransposed{}, tiled_mma_pv);
  auto s2r_thr_q = s2r_copy_q.get_thread_slice(tid);
  auto s2r_thr_k = s2r_copy_k.get_thread_slice(tid);
  auto s2r_thr_v = s2r_copy_v.get_thread_slice(tid);

  // V fragment layout for gemm_rs.
  auto sV0 = make_tensor(make_smem_ptr(v_base), SmemLayoutV{});
  auto sVt0_ns =
      make_tensor(sV0.data(), get_nonswizzle_portion(SmemLayoutVt{}));
  auto tCrV_layout = thr_mma_pv.partition_fragment_B(sVt0_ns).layout();

  // O fragment layout.
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

  // Coordinate tensor for softmax indexing.
  auto cS = make_identity_tensor(Shape<Int<kBr>, Int<kBc>>{});
  auto tScS = thr_mma_qk.partition_C(cS);
  auto tScS_rc = make_tensor(
      tScS.data(), ffpa_cute::convert_layout_acc_rowcol(tScS.layout()));
  constexpr int kSRows = decltype(size<0>(tScS_rc))::value;
  constexpr int kSCols = decltype(size<1>(tScS_rc))::value;

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

  // G2S helpers: load Q/K/V tiles via cp.async.
  auto g2s_load_q = [&](int d_chunk, int stage) {
    auto gQ = local_tile(mQ, Shape<Int<kBr>, Int<kQKDChunk>>{},
                         make_coord(Q_tile_id, d_chunk));
    auto sQ = make_tensor(make_smem_ptr(q_base + stage * kQChunkElements),
                          SmemLayoutQ{});
    copy(g2s_copy_qk, g2s_thr_qk.partition_S(gQ), g2s_thr_qk.partition_D(sQ));
  };

  auto g2s_load_k = [&](int kv_tile_idx, int d_chunk, int stage) {
    auto gK = local_tile(mK, Shape<Int<kBc>, Int<kQKDChunk>>{},
                         make_coord(kv_tile_idx, d_chunk));
    auto sK = make_tensor(make_smem_ptr(k_base + stage * kKChunkElements),
                          SmemLayoutK{});
    copy(g2s_copy_qk, g2s_thr_qk.partition_S(gK), g2s_thr_qk.partition_D(sK));
  };

  auto g2s_load_v = [&](int kv_tile_idx, int v_chunk, int stage) {
    auto gV = local_tile(mV, Shape<Int<kBc>, Int<kVDChunk>>{},
                         make_coord(kv_tile_idx, v_chunk));
    auto sV = make_tensor(make_smem_ptr(v_base + stage * kVChunkElements),
                          SmemLayoutV{});
    copy(g2s_copy_v, g2s_thr_v.partition_S(gV), g2s_thr_v.partition_D(sV));
  };

  // Initial V prefetch for kv_tile 0.
  {
    int v_write = 0;
#pragma unroll
    for (int v = 0; v < kStagesPV - 1 && v < kDChunksV; ++v) {
      g2s_load_v(0, v, v_write);
      cp_async_fence();
      v_write = (v_write + 1) % kStagesPV;
    }
    if constexpr (kStagesPV > 1) {
      cp_async_wait<kStagesPV - 2>();
      __syncthreads();
    }
  }

  // Initial QK prefetch for kv_tile 0.
  {
    int qk_write = 0;
#pragma unroll
    for (int d = 0; d < kStagesQK - 1 && d < kDChunksQK; ++d) {
      g2s_load_q(d, qk_write);
      g2s_load_k(0, d, qk_write);
      cp_async_fence();
      qk_write = (qk_write + 1) % kStagesQK;
    }
    if constexpr (kStagesQK > 1) {
      cp_async_wait<kStagesQK - 2>();
      __syncthreads();
    }
  }

#pragma unroll 1
  for (int kv_tile = 0; kv_tile < Tc_eff; ++kv_tile) {
    // V prefetch for kv_tile > 0.
    if (kv_tile > 0) {
      int v_write = 0;
#pragma unroll
      for (int v = 0; v < kStagesPV - 1 && v < kDChunksV; ++v) {
        g2s_load_v(kv_tile, v, v_write);
        cp_async_fence();
        v_write = (v_write + 1) % kStagesPV;
      }
      if constexpr (kStagesPV > 1) {
        cp_async_wait<kStagesPV - 2>();
        __syncthreads();
      }
    }

    // Phase 1: QK GEMM with split-D accumulation.
    auto tCrS = partition_fragment_C(tiled_mma_qk, Shape<Int<kBr>, Int<kBc>>{});
    clear(tCrS);

    int qk_read = 0;
    int qk_write = (kStagesQK > 1) ? (kStagesQK - 1) : 0;

    for (int d_chunk = 0; d_chunk < kDChunksQK; ++d_chunk) {
      // Issue prefetch (non-blocking, overlaps with gemm below).
      const int d_next = d_chunk + kStagesQK - 1;
      if (d_next < kDChunksQK) {
        g2s_load_q(d_next, qk_write);
        g2s_load_k(kv_tile, d_next, qk_write);
        cp_async_fence();
        qk_write = (qk_write + 1) % kStagesQK;
      }

      // Compute on current stage (overlaps with prefetch DMA).
      auto sQ = make_tensor(make_smem_ptr(q_base + qk_read * kQChunkElements),
                            SmemLayoutQ{});
      auto sK = make_tensor(make_smem_ptr(k_base + qk_read * kKChunkElements),
                            SmemLayoutK{});
      auto tCrQ = thr_mma_qk.partition_fragment_A(sQ);
      auto tCrK = thr_mma_qk.partition_fragment_B(sK);
      auto tQsQ = s2r_thr_q.partition_S(sQ);
      auto tKsK = s2r_thr_k.partition_S(sK);

      ffpa_cute::gemm_ss(tCrS, tCrQ, tCrK, tQsQ, tKsK, tiled_mma_qk, s2r_copy_q,
                         s2r_copy_k, s2r_thr_q, s2r_thr_k);
      qk_read = (qk_read + 1) % kStagesQK;

      // Wait for prefetched data before next iteration reads it.
      if (d_chunk < kDChunksQK - 1) {
        if constexpr (kStagesQK > 1) {
          cp_async_wait<kStagesQK - 2>();
        } else {
          cp_async_wait<0>();
        }
        __syncthreads();
      }
    }

    // Prefetch next kv_tile's initial QK chunks (overlaps softmax + PV).
    if (kv_tile < Tc_eff - 1) {
      int qk_write_next = 0;
#pragma unroll
      for (int d = 0; d < kStagesQK - 1 && d < kDChunksQK; ++d) {
        g2s_load_q(d, qk_write_next);
        g2s_load_k(kv_tile + 1, d, qk_write_next);
        cp_async_fence();
        qk_write_next = (qk_write_next + 1) % kStagesQK;
      }
      if constexpr (kStagesQK > 1) {
        cp_async_wait<kStagesQK - 2>();
        __syncthreads();
      }
    }

    // Phase 2: Online softmax.
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

      // Causal masking.
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

      // Additive attention bias (pre-softmax).
      if constexpr (kHasAttnBias) {
        ffpa_cute::apply_attn_bias_rowcol<decltype(scores), decltype(tScS_rc),
                                          kSRows, kSCols>(
            scores, tScS_rc, attn_bias, attn_bias_dtype, attn_bias_stride_b,
            attn_bias_stride_h, attn_bias_stride_m, attn_bias_stride_n, Nb_id,
            Nh_id, Br_base, kv_tile, kBc, inv_scale);
      }

      // Row-max + exp2 + row-sum.
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

      // Dropout on P (post-softmax, pre-PV).
      if constexpr (kHasDropout) {
        ffpa_cute::apply_dropout_rowcol<decltype(scores), decltype(tScS_rc),
                                        kORows, kSCols>(
            scores, tScS_rc, dropout_p, philox_seed, philox_offset, Nb_id, Nh,
            Nh_id, Nq, Nkv, Br_base, kv_tile, kBc);
      }

      // P fragment: convert fp32 → Element, reinterpret as A-operand for PV.
      auto tCrP = ffpa_cute::convert_type<Element>(tCrS);
      auto tCrPv = make_tensor(
          tCrP.data(),
          ffpa_cute::convert_layout_acc_Aregs<TiledMmaPV>(tCrP.layout()));

      // Phase 3: PV with split-D.
      int v_read = 0;
      int v_write_pv = (kStagesPV > 1) ? (kStagesPV - 1) : 0;

      for (int v_chunk = 0; v_chunk < kDChunksV; ++v_chunk) {
        // Issue prefetch (non-blocking, overlaps with gemm below).
        const int v_next = v_chunk + kStagesPV - 1;
        if (v_next < kDChunksV) {
          g2s_load_v(kv_tile, v_next, v_write_pv);
          cp_async_fence();
          v_write_pv = (v_write_pv + 1) % kStagesPV;
        }

        // O rescaling.
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

        // Compute on current stage (overlaps with prefetch DMA).
        auto sV = make_tensor(make_smem_ptr(v_base + v_read * kVChunkElements),
                              SmemLayoutV{});
        auto sVt = make_tensor(sV.data(), SmemLayoutVt{});
        auto tCrVStorage = thr_mma_pv.partition_fragment_B(sV);
        auto tCrV = make_tensor(tCrVStorage.data(), tCrV_layout);
        auto tVsVt = s2r_thr_v.partition_S(sVt);

        ffpa_cute::gemm_rs(tCrO, tCrPv, tCrV, tVsVt, tiled_mma_pv, s2r_copy_v,
                           s2r_thr_v);
        v_read = (v_read + 1) % kStagesPV;

        // Wait for prefetched data before next iteration reads it.
        if (v_chunk < kDChunksV - 1) {
          if constexpr (kStagesPV > 1) {
            cp_async_wait<kStagesPV - 2>();
          } else {
            cp_async_wait<0>();
          }
          __syncthreads();
        }
      }
      __syncthreads();
    }
  }

  // Phase 4: Epilogue. Normalize O, convert, store.
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

  // Write log-sum-exp.
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
}

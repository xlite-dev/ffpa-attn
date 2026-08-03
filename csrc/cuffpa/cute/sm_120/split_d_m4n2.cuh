#pragma once

// tensor.hpp MUST precede any cute/atom/* header (see sm_80/split_d.cuh).
#include <cute/tensor.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/cutlass.h>
#include <cutlass/device_kernel.h>

// namespace ffpa_cute
#include "../gemm.cuh"
#include "../attn_traits.cuh"
#include "../attn_bias.cuh"
#include "../dropout.cuh"
#include "../softmax.cuh"

using TmaBarrier = cutlass::arch::ClusterTransactionBarrier;
using CtaBarrier = cutlass::arch::ClusterBarrier;

// O regs = D/4 per thread.
// Key differences from split_d_fwd_cute_sm120 (M8N1):
//   1. atom_layout=(4,2,1): 2 N-warps split Bc/kVDChunk columns
//   2. Softmax: cross-N-warp reduction via SMEM exchange (2 syncs)
//   3. P→PV: SMEM roundtrip (stmatrix→LDSM_N) instead of register reinterpret
//   4. Epilogue: only n_warp==0 writes LSE (both N-warps share same rows)
//
// A/B benchmark vs M8N1 (constexpr dispatch in launch.cuh),
// RTX 5090 (SM120), torch 2.13.0+cu132, self-attn N=8192, stages=2.
// Table: FFPA time (ms) / TFLOPS, fp16 (bf16 within ±2%); O_err≈1e-4 both.
//   D     M8N1 (ms/TFLOPS)      M4N2 (ms/TFLOPS)       winner
//   320   13.21/13.20  208T     15.35/15.30  179T      M8N1 +16%
//   384   16.37/16.22  202T     18.13/18.04  182T      M8N1 +11%
//   448   20.33/20.25  189T     20.77/20.55  185T      M8N1  +2%
//   512   22.69/22.58  194T     23.73/23.48  185T      M8N1  +4%
//   576   26.54/26.36  186T     26.42/26.07  187T      M4N2  +0.5%
//   640   29.99/29.83  183T     31.28/30.96  176T      M8N1  +4%
//   768   40.55/39.79  163T     37.78/37.31  175T      M4N2  +7%
//   896   54.03/57.97  142T     49.16/48.72  157T      M4N2 +11%
//   1024  88.37/87.75  100T     57.11/56.60  154T      M4N2 +55%
// Cross point lies between 640 and 768. Final dispatch (launch.cuh):
// D<768 -> M8N1 (P regs stay under the 255-reg ceiling), D>=768 -> this
// M4N2 kernel. At D=1024 M8N1's o_acc = D/2 = 512 regs/thread spills to
// local mem and collapses to ~100T; M4N2's D/4 = 256 regs keeps 154T.
template <typename Traits, typename TmaQ, typename TmaK, typename TmaV,
          typename TmaO, int kHasAttnBias = 0, int kHasDropout = 0>
__global__ void __launch_bounds__(Traits::kNumThreads, 1)
    split_d_m4n2_fwd_cute_sm120(
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
  // Body-level arch guard: TMA/stmatrix need sm>=90, but in mixed -gencode
  // builds the sm_89 device pass still compiles this TU; the guard compiles
  // the body into a no-op stub there. Body-level (not file-level) is required
  // because the host launcher references this kernel via <<<>>> and nvcc must
  // see its declaration in every device pass; hiding it file-level fails with
  // "identifier undefined". Runtime safety: launch.cuh dispatches TMA kernels
  // only when prop->major >= 9, so pre-90 devices never execute the stub.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  using namespace cute;
  using cute::tma_store_arrive;
  using cute::tma_store_wait;
  using Element = typename Traits::Element;
  using SmemLayoutQ = typename Traits::SmemLayoutQ;
  using SmemLayoutK = typename Traits::SmemLayoutK;
  using SmemLayoutV = typename Traits::SmemLayoutV;
  using SmemLayoutVt = typename Traits::SmemLayoutVt;
  using SmemLayoutO = typename Traits::SmemLayoutO;
  using SmemLayoutP = typename Traits::SmemLayoutP;
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
  constexpr int kPElements = cosize(SmemLayoutP{});

  static_assert(cosize(SmemLayoutO{}) <= kStagesPV * cosize(SmemLayoutV{}),
                "TMA-O: O staging buffer must fit in reused V-stage smem");

  const int Nb_id = blockIdx.y / Nh;
  const int Nh_id = blockIdx.y % Nh;
  const int Q_tile_id = blockIdx.x;
  const int group_size = Nh / Nh_kv;
  const int kv_head_idx = Nh_id / group_size;
  const int Br_base = Q_tile_id * kBr;
  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;
  const int n_warp = warp_id / 4;  // 0 or 1

  if (Br_base >= Nq)
    return;

  const int kv_offset = Nkv - Nq;
  const int causal_thresh_row0 = Br_base + kv_offset;
  const int Tc_eff =
      causal ? min(Tc, ((Br_base + kBr - 1 + kv_offset) / kBc) + 1) : Tc;
  const int mask_start_tile =
      causal ? max(0, (causal_thresh_row0 + 1) / kBc) : INT_MAX;

  const int q_row_offset = (Nb_id * Nh + Nh_id) * Nq;
  const int kv_row_offset = (Nb_id * Nh_kv + kv_head_idx) * Nkv;

  // SMEM layout: [q_base | k_base | v_base | p_base | exchange]
  extern __shared__ __align__(1024) Element shm[];
  Element* q_base = shm;
  Element* k_base = q_base + kStagesQK * kQChunkElements;
  Element* v_base = k_base + kStagesQK * kKChunkElements;
  Element* p_base = v_base + kStagesPV * kVChunkElements;
  // Cross-N-warp softmax exchange: [8 warps][16 rows] floats (reused max/sum).
  float* smem_exchange = reinterpret_cast<float*>(p_base + kPElements);

  __shared__ uint64_t qk_full[kStagesQK];
  __shared__ uint64_t qk_empty[kStagesQK];
  __shared__ uint64_t v_full[kStagesPV];
  __shared__ uint64_t v_empty[kStagesPV];

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

  TiledMmaQK tiled_mma_qk;
  TiledMmaPV tiled_mma_pv;
  auto thr_mma_qk = tiled_mma_qk.get_thread_slice(tid);
  auto thr_mma_pv = tiled_mma_pv.get_thread_slice(tid);

  auto s2r_copy_q = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma_qk);
  auto s2r_copy_k = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma_qk);
  auto s2r_thr_q = s2r_copy_q.get_thread_slice(tid);
  auto s2r_thr_k = s2r_copy_k.get_thread_slice(tid);

  // P→PV: read full P[kBr,kBc] from SMEM as PV A-operand via LDSM_N
  auto s2r_copy_p = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma_pv);
  auto s2r_copy_v = make_tiled_copy_B(SmemCopyAtomTransposed{}, tiled_mma_pv);
  auto s2r_thr_p = s2r_copy_p.get_thread_slice(tid);
  auto s2r_thr_v = s2r_copy_v.get_thread_slice(tid);

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

  for (int s = 0; s < kStagesQK; ++s)
    CtaBarrier::arrive(&qk_empty[s]);
  for (int s = 0; s < kStagesPV; ++s)
    CtaBarrier::arrive(&v_empty[s]);

  auto issue_qk_tma = [&](int d_chunk, int stage, int kv_tile_idx) {
    cutlass::arch::fence_view_async_shared();
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
    cutlass::arch::fence_view_async_shared();
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

  if (tid == 0) {
    for (int d = 0; d < kStagesQK && d < kDChunksQK; ++d) {
      CtaBarrier::wait(&qk_empty[d], 0);
      issue_qk_tma(d, d, 0);
    }
  }

  if (tid == 0) {
    for (int v = 0; v < kStagesPV && v < kDChunksV; ++v) {
      const int chunk_index = v;
      const int v_stage = chunk_index % kStagesPV;
      const int v_phase = (chunk_index / kStagesPV) & 1;
      CtaBarrier::wait(&v_empty[v_stage], v_phase);
      issue_v_tma(v, v_stage, 0);
    }
  }

#pragma unroll 1
  for (int kv_tile = 0; kv_tile < Tc_eff; ++kv_tile) {
    if (kv_tile > 0)
      __syncthreads();
    if (kv_tile > 0 && tid == 0) {
      for (int v = 0; v < kStagesPV && v < kDChunksV; ++v) {
        const int chunk_index = kv_tile * kDChunksV + v;
        const int v_stage = chunk_index % kStagesPV;
        const int v_phase = (chunk_index / kStagesPV) & 1;
        CtaBarrier::wait(&v_empty[v_stage], v_phase);
        issue_v_tma(v, v_stage, kv_tile);
      }
    }

    // Phase 1: QK GEMM with split-D accumulation
    auto tCrS = partition_fragment_C(tiled_mma_qk, Shape<Int<kBr>, Int<kBc>>{});
    clear(tCrS);

#pragma unroll
    for (int d_chunk = 0; d_chunk < kDChunksQK; ++d_chunk) {
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

      CtaBarrier::arrive(&qk_empty[stage]);

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

    if (kv_tile < Tc_eff - 1 && tid == 0) {
      for (int d = 0; d < kStagesQK && d < kDChunksQK; ++d) {
        const int chunk_index = (kv_tile + 1) * kDChunksQK + d;
        const int stage = chunk_index % kStagesQK;
        const int phase = (chunk_index / kStagesQK) & 1;
        CtaBarrier::wait(&qk_empty[stage], phase);
        issue_qk_tma(d, stage, kv_tile + 1);
      }
    }

    // Phase 2: online softmax with cross-N-warp reduction
    {
      auto scores = make_tensor(
          tCrS.data(), ffpa_cute::convert_layout_acc_rowcol(tCrS.layout()));
      float row_scale[kORows];

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

      if constexpr (kHasAttnBias) {
        ffpa_cute::apply_attn_bias_rowcol<decltype(scores), decltype(tScS_rc),
                                          kSRows, kSCols>(
            scores, tScS_rc, attn_bias, attn_bias_dtype, attn_bias_stride_b,
            attn_bias_stride_h, attn_bias_stride_m, attn_bias_stride_n, Nb_id,
            Nh_id, Br_base, kv_tile, kBc, inv_scale);
      }

      ffpa_cute::online_safe_softmax_m4n2<decltype(scores), decltype(tScS_rc),
                                          kORows>(
          scores, tScS_rc, scale, row_max, row_sum, row_scale, smem_exchange,
          warp_id, lane_id, Traits::kRescaleThreshold);

      bool local_need_rescale = false;
#pragma unroll
      for (int r = 0; r < kORows; ++r)
        local_need_rescale = local_need_rescale || (row_scale[r] < 1.0f);
      const bool need_rescale = __any_sync(0xffffffff, local_need_rescale);

      if constexpr (kHasDropout) {
        ffpa_cute::apply_dropout_rowcol<decltype(scores), decltype(tScS_rc),
                                        kORows, kSCols>(
            scores, tScS_rc, dropout_p, philox_seed, philox_offset, Nb_id, Nh,
            Nh_id, Nq, Nkv, Br_base, kv_tile, kBc);
      }

      // Phase 3: P→SMEM (stmatrix), then LDSM_N into PV A-regs
      auto tCrP = ffpa_cute::convert_type<Element>(tCrS);

      auto sP = make_tensor(make_smem_ptr(p_base), SmemLayoutP{});
      auto r2s_copy_p = make_tiled_copy_C(
          Copy_Atom<SM90_U32x4_STSM_N, Element>{}, tiled_mma_qk);
      auto r2s_thr_p = r2s_copy_p.get_slice(tid);
      auto tCrP_src = r2s_thr_p.retile_S(tCrP);
      auto tCsP_dst = r2s_thr_p.partition_D(sP);
      copy(r2s_copy_p, tCrP_src, tCsP_dst);
      cutlass::arch::fence_view_async_shared();
      __syncthreads();
      // Reuses the P write-read barrier above as the sum-publish sync.
      ffpa_cute::finalize_row_sum_m4n2<kORows>(row_sum, row_scale,
                                               smem_exchange, warp_id, lane_id);

      auto tCrPv_storage = thr_mma_pv.partition_fragment_A(sP);
      auto tPsP = s2r_thr_p.partition_S(sP);
      copy(s2r_copy_p, tPsP, tCrPv_storage);

      // Phase 4: PV GEMM with split-D accumulation
#pragma unroll
      for (int v_chunk = 0; v_chunk < kDChunksV; ++v_chunk) {
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

        auto tCrO = make_tensor(make_rmem_ptr(&o_acc_storage[v_chunk][0]),
                                OFragLayout{});
        if (kv_tile > 0 && need_rescale) {
          auto tCrO_rc = make_tensor(
              tCrO.data(), ffpa_cute::convert_layout_acc_rowcol(tCrO.layout()));
#pragma unroll
          for (int row = 0; row < kORows; ++row)
#pragma unroll
            for (int col = 0; col < kOCols; ++col)
              tCrO_rc(row, col) *= row_scale[row];
        }

        ffpa_cute::gemm_rs(tCrO, tCrPv_storage, tCrV, tVsVt, tiled_mma_pv,
                           s2r_copy_v, s2r_thr_v);

        CtaBarrier::arrive(&v_empty[v_stage]);

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

  // Phase 5: Epilogue — O /= row_sum, R→S→TMA store or predicated R→G.
  // Only n_warp==0 writes LSE (both N-warps share the same Q rows).
  // TMA-store drain race (fixed): only tid=0 issues the store, so
  // tma_store_wait<0>() is a no-op for every other thread. Without a CTA
  // barrier the next batch's R->S would overwrite shm the in-flight TMA
  // store is still reading -> deterministic O corruption whenever
  // kNBatches >= 2 (D=640 stages=2 -> kNBatches=5 fails; stages=3 -> 1
  // passes; D=512 stages=2 -> kNBatches=2 passes only because each batch
  // writes 16KB, long enough for the store to drain). The __syncthreads()
  // after tma_store_wait below gates all threads on the drain; the batch
  // condition is CTA-uniform so it cannot deadlock.
  {
    constexpr int kVChunksPerBatch = Traits::kVChunksPerBatch;
    constexpr int kNBatches = Traits::kNBatches;
    constexpr int kOTileElems = cosize(SmemLayoutO{});

    __syncthreads();

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
#pragma unroll
      for (int batch = 0; batch < kNBatches; ++batch) {
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
#pragma unroll
        for (int v_in = 0; v_in < kVChunksPerBatch; ++v_in) {
          int v_chunk = batch * kVChunksPerBatch + v_in;
          auto sO_v = make_tensor(make_smem_ptr(shm + v_in * kOTileElems),
                                  SmemLayoutO{});
          auto gO_tma = local_tile(mO_tma, Shape<Int<kBr>, Int<kVDChunk>>{},
                                   make_coord(Q_tile_id, v_chunk));
          auto tCgO_tma = o_slice.partition_D(gO_tma);
          auto tOsO = o_slice.partition_S(sO_v);
          if (tid == 0)
            copy(tma_o, tOsO, tCgO_tma);
        }
        tma_store_arrive();
        if (batch < kNBatches - 1) {
          tma_store_wait<0>();
          __syncthreads();  // all threads wait (tma_store_wait is tid=0-only)
        }
      }
    } else {
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

    // LSE write: only n_warp==0 writes (both N-warps share same Q rows)
    if (softmax_lse != nullptr && n_warp == 0) {
      const int lse_base = Nb_id * Nh * Nq + Nh_id * Nq;
#pragma unroll
      for (int row = 0; row < kORows; ++row) {
        const float lse = (row_max[row] + log2f(row_sum[row])) * FFPA_M_LN2;
        const int global_row = Br_base + get<0>(tScS_rc(row, 0));
        if (global_row < Nq)
          softmax_lse[lse_base + global_row] = lse;
      }
    }

    if (Br_base + kBr <= Nq)
      tma_store_wait<0>();
  }
#endif  // defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
}

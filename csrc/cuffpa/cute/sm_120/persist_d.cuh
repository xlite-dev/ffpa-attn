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

// WS persist-D: 128 producer (TMA-only) + 256 consumer (MMA-only), 384 threads.
// sm_120f target: setmaxnreg effective (producer dec 32 / consumer inc 232);
// sm_120a target hits ptxas C7506 and silently ignores it, so build with 120f.
// Epilogue: R->S->TMA store (aligned) or R->G (tail); a __syncthreads would
// deadlock since the producer warpgroup has already fallen through the
// early-return below, so sync with a named barrier (consumer threads only).
// NOTE: 32-multiple small D (D=32/64/96/128) is supported. The smem swizzle
// is auto-selected by Traits from D*2B (SW128 for 64-mult, SW64 for D=32/96);
// TMA descriptors inherit the same swizzle from SmemLayoutO.
// kNhdKV: K/V arrive as a batched 4D (N, D, h, b) TMA over an NHD (diffusers
// BNHD) permute view instead of flat BHND rows; the (kv_head, batch) origin
// rides the local_tile coord (FA3 batched pattern) instead of domain_offset.
// kNhdQ: same for Q - flat (B*N, H*D) rows with the q head as the column
// tile. O stays BHND-packed (caller allocates it packed and re-views).
template <typename Traits, typename TmaQ, typename TmaK, typename TmaV,
          typename TmaO, int kHasAttnBias = 0, int kHasDropout = 0,
          bool kNhdKV = false, bool kNhdQ = false>
__global__ void __launch_bounds__(384, 1) persist_d_ws_fwd_cute_sm120(
    CUTLASS_GRID_CONSTANT TmaQ const tma_q,
    CUTLASS_GRID_CONSTANT TmaK const tma_k,
    CUTLASS_GRID_CONSTANT TmaV const tma_v,
    CUTLASS_GRID_CONSTANT TmaO const tma_o,
    typename Traits::Element* __restrict__ O, float* __restrict__ softmax_lse,
    int Nq, int Nkv, int Nh, int Nh_kv, float scale, int Tc, int causal,
    int total_q_rows, int total_kv_rows,
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
  using Element = typename Traits::Element;
  using SmemLayoutQ = typename Traits::SmemLayoutQ;
  using SmemLayoutKV = typename Traits::SmemLayoutKV;
  using SmemLayoutKVt = typename Traits::SmemLayoutKVt;
  using SmemLayoutO = typename Traits::SmemLayoutO;
  using TiledMmaQK = typename Traits::TiledMmaQK;
  using TiledMmaPV = typename Traits::TiledMmaPV;
  using SmemCopyAtom = typename Traits::SmemCopyAtom;
  using SmemCopyAtomTransposed = typename Traits::SmemCopyAtomTransposed;

  constexpr int kBr = Traits::kBr;
  constexpr int kBc = Traits::kBc;
  constexpr int kHeadDim = Traits::kHeadDim;
  constexpr int kStagesK = Traits::kStagesK;
  constexpr int kStagesV = Traits::kStagesV;
  constexpr int kProducerThreads = 128;
  constexpr int kConsumerThreads = 256;

  constexpr int kQTileElements = cosize(SmemLayoutQ{});
  constexpr int kKVTileElements = cosize(SmemLayoutKV{});

  const int Nb_id = blockIdx.y / Nh;
  const int Nh_id = blockIdx.y % Nh;
  const int Q_tile_id = blockIdx.x;
  const int group_size = Nh / Nh_kv;
  const int kv_head_idx = Nh_id / group_size;
  const int Br_base = Q_tile_id * kBr;
  const int tid = threadIdx.x;
  const bool is_producer = tid < kProducerThreads;
  const int wg_tid = is_producer ? tid : tid - kProducerThreads;

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

  // SMEM: [Q persist | K stages | V stages]
  extern __shared__ __align__(1024) Element shm[];
  Element* q_base = shm;
  Element* k_base = q_base + kQTileElements;
  Element* v_base = k_base + kStagesK * kKVTileElements;

  __shared__ uint64_t q_full;
  __shared__ uint64_t k_full[kStagesK];
  __shared__ uint64_t k_empty[kStagesK];
  __shared__ uint64_t v_full[kStagesV];
  __shared__ uint64_t v_empty[kStagesV];

  if (tid == 0) {
    TmaBarrier::init(&q_full, 1);
    for (int s = 0; s < kStagesK; ++s) {
      TmaBarrier::init(&k_full[s], 1);
      CtaBarrier::init(&k_empty[s], kConsumerThreads);
    }
    for (int s = 0; s < kStagesV; ++s) {
      TmaBarrier::init(&v_full[s], 1);
      CtaBarrier::init(&v_empty[s], kConsumerThreads);
    }
  }
  __syncthreads();

  // Producer warpgroup: only wg_tid==0 issues TMA; the other 127 threads idle.
  // P0 loads Q once; P1 prefetches K/V[0..S-2]; P2 prefetches ahead by S-1
  // tiles (V-first then K-after) so consumer's K/V waits never stall.
  if (is_producer) {
    // Release registers to the CTA pool so the two consumer warpgroups can
    // alloc up to 232 regs/thread.
    cutlass::arch::warpgroup_reg_dealloc<32>();  // sm_120f
    if (wg_tid == 0) {
      auto mQ = [&] {
        if constexpr (kNhdQ)
          // NHD (diffusers BNHD) -> flat (B*N, H*D) rows, batch via
          // domain_offset, head as a kHeadDim-wide column tile.
          return domain_offset(make_coord(static_cast<long>(Nb_id) * Nq, 0),
                               tma_q.get_tma_tensor(make_shape(
                                   static_cast<long>(gridDim.y / Nh) * Nq,
                                   static_cast<long>(Nh) * kHeadDim)));
        else
          return domain_offset(
              make_coord(q_row_offset, 0),
              tma_q.get_tma_tensor(make_shape(total_q_rows, Int<kHeadDim>{})));
      }();
      // K/V gmem view: BHND packed -> flat (B*H*N, D) rows, origin via
      // domain_offset; NHD (diffusers BNHD) -> flat (B*N, H*D) rows (row
      // stride H*D uniform across batch), head selected as a kHeadDim-wide
      // column tile (tile coord 1) and batch via domain_offset. Both stay
      // plain flat-2D TMA; only the tile coord gains the head index.
      const auto make_mKV = [&](auto const& tma) {
        if constexpr (kNhdKV)
          return domain_offset(make_coord(static_cast<long>(Nb_id) * Nkv, 0),
                               tma.get_tma_tensor(make_shape(
                                   static_cast<long>(gridDim.y / Nh) * Nkv,
                                   static_cast<long>(Nh_kv) * kHeadDim)));
        else
          return domain_offset(
              make_coord(kv_row_offset, 0),
              tma.get_tma_tensor(make_shape(total_kv_rows, Int<kHeadDim>{})));
      };
      auto mK = make_mKV(tma_k);
      auto mV = make_mKV(tma_v);
      const auto kv_tile_coord = [&](int tile) {
        if constexpr (kNhdKV)
          return make_coord(tile, kv_head_idx);
        else
          return make_coord(tile, _0{});
      };
      auto q_slice = tma_q.get_slice(_0{});
      auto k_slice = tma_k.get_slice(_0{});
      auto v_slice = tma_v.get_slice(_0{});

      // P0: Q one-shot full-D TMA
      auto sQ = make_tensor(make_smem_ptr(q_base), SmemLayoutQ{});
      auto gQ = [&] {
        if constexpr (kNhdQ)
          return local_tile(mQ, Shape<Int<kBr>, Int<kHeadDim>>{},
                            make_coord(Q_tile_id, Nh_id));
        else
          return local_tile(mQ, Shape<Int<kBr>, Int<kHeadDim>>{},
                            make_coord(Q_tile_id, _0{}));
      }();
      auto tQgQ = q_slice.partition_S(gQ);
      auto tQsQ = q_slice.partition_D(sQ);
      TmaBarrier::arrive_and_expect_tx(&q_full, sizeof(Element) * size(sQ));
      copy(tma_q.with(q_full), tQgQ, tQsQ);

      // P1: prefetch K[0..Sk-2]; waits unblock once consumer arrives k_empty
      for (int s = 0; s < kStagesK - 1; ++s) {
        if (s < Tc_eff) {
          CtaBarrier::wait(&k_empty[s], 0);
          auto sK = make_tensor(make_smem_ptr(k_base + s * kKVTileElements),
                                SmemLayoutKV{});
          auto gK = local_tile(mK, Shape<Int<kBc>, Int<kHeadDim>>{},
                               kv_tile_coord(s));
          auto tKgK = k_slice.partition_S(gK);
          auto tKsK = k_slice.partition_D(sK);
          TmaBarrier::arrive_and_expect_tx(&k_full[s],
                                           sizeof(Element) * size(sK));
          copy(tma_k.with(k_full[s]), tKgK, tKsK);
        }
      }
      // P1b: prefetch V[0..Sv-2]
      for (int s = 0; s < kStagesV - 1; ++s) {
        if (s < Tc_eff) {
          CtaBarrier::wait(&v_empty[s], 0);
          auto sV = make_tensor(make_smem_ptr(v_base + s * kKVTileElements),
                                SmemLayoutKV{});
          auto gV = local_tile(mV, Shape<Int<kBc>, Int<kHeadDim>>{},
                               kv_tile_coord(s));
          auto tVgV = v_slice.partition_S(gV);
          auto tVsV = v_slice.partition_D(sV);
          TmaBarrier::arrive_and_expect_tx(&v_full[s],
                                           sizeof(Element) * size(sV));
          copy(tma_v.with(v_full[s]), tVgV, tVsV);
        }
      }
      // P2: prefetch-ahead loop. s_next == k_stg since
      // (tile+kStagesK)%kStagesK == tile%kStagesK; only the phase flips.
      // phase/stage transform (kStagesK=2): same slot reused, phase flips/cycle
      //   kv_tile  k_stg  k_phase  k_next  s_next  p_next
      //   0        0      0        2       0       1
      //   1        1      0        3       1       1
      //   2        0      1        4       0       0
      //   3        1      1        5       1       0
      for (int tile = 0; tile < Tc_eff; ++tile) {
        // V: prefetch V[tile+Sv-1]
        {
          const int v_tile = tile + kStagesV - 1;
          if (v_tile < Tc_eff) {
            const int stage_v = v_tile % kStagesV;
            const int phase_v = (v_tile / kStagesV) & 1;
            CtaBarrier::wait(&v_empty[stage_v], phase_v);
            auto sV =
                make_tensor(make_smem_ptr(v_base + stage_v * kKVTileElements),
                            SmemLayoutKV{});
            auto gV = local_tile(mV, Shape<Int<kBc>, Int<kHeadDim>>{},
                                 kv_tile_coord(v_tile));
            auto tVgV = v_slice.partition_S(gV);
            auto tVsV = v_slice.partition_D(sV);
            TmaBarrier::arrive_and_expect_tx(&v_full[stage_v],
                                             sizeof(Element) * size(sV));
            copy(tma_v.with(v_full[stage_v]), tVgV, tVsV);
          }
        }
        // K: prefetch K[tile+Sk-1] (phase/stage table same as K above)
        {
          const int k_tile = tile + kStagesK - 1;
          if (k_tile < Tc_eff) {
            const int stage_k = k_tile % kStagesK;
            const int phase_k = (k_tile / kStagesK) & 1;
            CtaBarrier::wait(&k_empty[stage_k], phase_k);
            auto sK =
                make_tensor(make_smem_ptr(k_base + stage_k * kKVTileElements),
                            SmemLayoutKV{});
            auto gK = local_tile(mK, Shape<Int<kBc>, Int<kHeadDim>>{},
                                 kv_tile_coord(k_tile));
            auto tKgK = k_slice.partition_S(gK);
            auto tKsK = k_slice.partition_D(sK);
            TmaBarrier::arrive_and_expect_tx(&k_full[stage_k],
                                             sizeof(Element) * size(sK));
            copy(tma_k.with(k_full[stage_k]), tKgK, tKsK);
          }
        }
      }
    }
    return;
  }

  // Consumer path (wg_tid 0..255): wait Q, release K/V slots, then the full
  // QK->softmax->PV loop. No TMA issue here; no __syncthreads (single WG).
  cutlass::arch::warpgroup_reg_alloc<232>();  // sm_120f

  TmaBarrier::wait(&q_full, 0);
  cutlass::arch::fence_view_async_shared();

  // Mark all K/V slots empty so the producer prefetch can proceed.
  for (int s = 0; s < kStagesK; ++s)
    CtaBarrier::arrive(&k_empty[s]);
  for (int s = 0; s < kStagesV; ++s)
    CtaBarrier::arrive(&v_empty[s]);

  TiledMmaQK tiled_mma_qk;
  TiledMmaPV tiled_mma_pv;
  auto thr_mma_qk = tiled_mma_qk.get_thread_slice(wg_tid);
  auto thr_mma_pv = tiled_mma_pv.get_thread_slice(wg_tid);

  auto s2r_copy_q = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma_qk);
  auto s2r_copy_k = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma_qk);
  auto s2r_copy_v = make_tiled_copy_B(SmemCopyAtomTransposed{}, tiled_mma_pv);
  auto s2r_thr_q = s2r_copy_q.get_thread_slice(wg_tid);
  auto s2r_thr_k = s2r_copy_k.get_thread_slice(wg_tid);
  auto s2r_thr_v = s2r_copy_v.get_thread_slice(wg_tid);

  auto sV0 = make_tensor(make_smem_ptr(v_base), SmemLayoutKV{});
  auto sVt0_ns =
      make_tensor(sV0.data(), get_nonswizzle_portion(SmemLayoutKVt{}));
  auto tCrV_layout = thr_mma_pv.partition_fragment_B(sVt0_ns).layout();

  using OFragType = decltype(partition_fragment_C(
      tiled_mma_pv, Shape<Int<kBr>, Int<kHeadDim>>{}));
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

  float o_acc[kOElemsPerFrag];
#pragma unroll
  for (int i = 0; i < kOElemsPerFrag; ++i)
    o_acc[i] = 0.0f;

  // Persistent Q smem tensor (read-only in main loop)
  auto sQ = make_tensor(make_smem_ptr(q_base), SmemLayoutQ{});
  auto tCrQ = thr_mma_qk.partition_fragment_A(sQ);
  auto tQsQ_s2r = s2r_thr_q.partition_S(sQ);

#pragma unroll 1
  for (int kv_tile = 0; kv_tile < Tc_eff; ++kv_tile) {
    const int k_stg = kv_tile % kStagesK;
    const int k_phase = (kv_tile / kStagesK) & 1;
    const int v_stg = kv_tile % kStagesV;
    const int v_phase = (kv_tile / kStagesV) & 1;

    // QK GEMM: single gemm_ss, full-D Q × full-D K
    TmaBarrier::wait(&k_full[k_stg], k_phase);
    cutlass::arch::fence_view_async_shared();

    auto sK = make_tensor(make_smem_ptr(k_base + k_stg * kKVTileElements),
                          SmemLayoutKV{});
    auto tCrK = thr_mma_qk.partition_fragment_B(sK);
    auto tKsK_s2r = s2r_thr_k.partition_S(sK);

    auto tCrS = partition_fragment_C(tiled_mma_qk, Shape<Int<kBr>, Int<kBc>>{});
    clear(tCrS);
    ffpa_cute::gemm_ss(tCrS, tCrQ, tCrK, tQsQ_s2r, tKsK_s2r, tiled_mma_qk,
                       s2r_copy_q, s2r_copy_k, s2r_thr_q, s2r_thr_k);
    // Release K slot to producer; no inline prefetch here (producer owns it).
    CtaBarrier::arrive(&k_empty[k_stg]);

    // Online softmax
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

    ffpa_cute::online_safe_softmax<decltype(scores), decltype(tScS_rc), kORows>(
        scores, tScS_rc, scale, row_max, row_sum, row_scale,
        Traits::kRescaleThreshold);

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

    // Rescale O accumulator
    if (kv_tile > 0 && need_rescale) {
      auto tCrO = make_tensor(make_rmem_ptr(o_acc), OFragLayout{});
      auto tCrO_rc = make_tensor(
          tCrO.data(), ffpa_cute::convert_layout_acc_rowcol(tCrO.layout()));
#pragma unroll
      for (int row = 0; row < kORows; ++row)
#pragma unroll
        for (int col = 0; col < kOCols; ++col)
          tCrO_rc(row, col) *= row_scale[row];
    }

    // PV GEMM: single gemm_rs, full-D P × full-D V
    TmaBarrier::wait(&v_full[v_stg], v_phase);
    cutlass::arch::fence_view_async_shared();

    auto sV = make_tensor(make_smem_ptr(v_base + v_stg * kKVTileElements),
                          SmemLayoutKV{});
    auto sVt = make_tensor(sV.data(), SmemLayoutKVt{});
    auto tCrVStorage = thr_mma_pv.partition_fragment_B(sV);
    auto tCrV = make_tensor(tCrVStorage.data(), tCrV_layout);
    auto tVsVt_s2r = s2r_thr_v.partition_S(sVt);

    auto tCrP = ffpa_cute::convert_type<Element>(tCrS);
    auto tCrPv = make_tensor(
        tCrP.data(),
        ffpa_cute::convert_layout_acc_Aregs<TiledMmaPV>(tCrP.layout()));
    auto tCrO = make_tensor(make_rmem_ptr(o_acc), OFragLayout{});
    ffpa_cute::gemm_rs(tCrO, tCrPv, tCrV, tVsVt_s2r, tiled_mma_pv, s2r_copy_v,
                       s2r_thr_v);
    // Release V slot to producer; no inline prefetch here (producer owns it).
    CtaBarrier::arrive(&v_empty[v_stg]);
  }

  // Epilogue: O /= row_sum, R->S->TMA store (aligned, single full-D tile) or
  // direct R->G (tail). The producer warpgroup has returned, so CTA-wide
  // __syncthreads would deadlock; sync with a named barrier limited to the
  // consumer threads instead.
  // NOTE: single full-D tile, no shm reuse after the TMA store -> unlike the
  // split-D kernels' batched epilogue (which must __syncthreads/NamedBarrier
  // after tma_store_wait to avoid the next batch overwriting shm the store
  // still reads), no drain barrier is required here.
  {
    cutlass::arch::NamedBarrier::sync(kConsumerThreads, 0);

    auto tCrO = make_tensor(make_rmem_ptr(o_acc), OFragLayout{});
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

    auto r2s_copy = make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N, Element>{},
                                      tiled_mma_pv);
    auto r2s_thr = r2s_copy.get_slice(wg_tid);

    if (Br_base + kBr <= Nq) {
      // aligned: R->S via STSM (reuse the freed Q smem), then TMA store
      auto sO = make_tensor(make_smem_ptr(q_base), SmemLayoutO{});
      auto tCrOHalf_src = r2s_thr.retile_S(tCrOHalf);
      auto tCsO_dst = r2s_thr.partition_D(sO);
      copy(r2s_copy, tCrOHalf_src, tCsO_dst);
      cutlass::arch::fence_view_async_shared();
      cutlass::arch::NamedBarrier::sync(kConsumerThreads, 0);

      auto mO_tma = domain_offset(
          make_coord(q_row_offset, 0),
          tma_o.get_tma_tensor(make_shape(total_q_rows, Int<kHeadDim>{})));
      auto o_slice = tma_o.get_slice(_0{});
      auto gO_tma = local_tile(mO_tma, Shape<Int<kBr>, Int<kHeadDim>>{},
                               make_coord(Q_tile_id, _0{}));
      auto tCgO_tma = o_slice.partition_D(gO_tma);
      auto tOsO = o_slice.partition_S(sO);
      if (wg_tid == 0)
        copy(tma_o, tOsO, tCgO_tma);
      tma_store_arrive();
      tma_store_wait<0>();
    } else {
      // tail: direct R->G store
      const int O_gmem_offset =
          (Nb_id * Nh * Nq * kHeadDim) + (Nh_id * Nq * kHeadDim);
      auto mO = make_tensor(make_gmem_ptr(O + O_gmem_offset),
                            make_shape(Nq, Int<kHeadDim>{}),
                            make_stride(Int<kHeadDim>{}, _1{}));
      auto gO = local_tile(mO, Shape<Int<kBr>, Int<kHeadDim>>{},
                           make_coord(Q_tile_id, _0{}));
      auto tCgO = thr_mma_pv.partition_C(gO);
      auto cO = make_identity_tensor(Shape<Int<kBr>, Int<kHeadDim>>{});
      auto tOcO = thr_mma_pv.partition_C(cO);
#pragma unroll
      for (int i = 0; i < size(tCrOHalf); ++i) {
        const int global_row = Br_base + get<0>(tOcO(i));
        if (global_row < Nq)
          tCgO(i) = tCrOHalf(i);
      }
    }

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
#endif  // defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
}

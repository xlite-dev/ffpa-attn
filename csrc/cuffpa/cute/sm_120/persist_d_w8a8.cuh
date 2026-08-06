#pragma once

#include <cuda_fp8.h>

#include <cute/tensor.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/cutlass.h>
#include <cutlass/device_kernel.h>

#include <algorithm>

#include "../gemm.cuh"
#include "../attn_traits.cuh"
#include "../fp8_pscale.cuh"
#include "../reg2reg_fp8.cuh"
#include "../softmax.cuh"

namespace ffpa_w8a8 {
using TmaBarrier = cutlass::arch::ClusterTransactionBarrier;
using CtaBarrier = cutlass::arch::ClusterBarrier;

// Smooth-K lse correction, per-row partial: dot(Q8_row, km). Softmax is
// shift-invariant, so smoothing K leaves O unchanged, but the returned lse
// must add back scale*qs*dot(Q_row, km) (see the epilogue).
// m16n8 C layout: the 4 peer lanes of a quad share the same rows; each lane
// strides over kHeadDim/16 column chunks of 4, and xor-1/xor-2 complete the
// quad-local reduce (a full-warp butterfly would mix the warp's 8 rows).
// Perf note: scalar smem/gmem reads, correctness-first; the lse path is cold,
// revisit only if it ever shows up in profiles.
template <int kHeadDim, int kRows, typename SmemQTensor, typename CoordTensor>
CUTE_DEVICE void smooth_k_qk_dot(const SmemQTensor& sQ,
                                 const CoordTensor& tScS_rc,
                                 const float* __restrict__ km_bh, float* qkm) {
  constexpr int kVec = 4;
  constexpr int kQuad = 4;
  constexpr int kIters = kHeadDim / (kVec * kQuad);
  const int qlane = cutlass::canonical_lane_idx() % kQuad;
#pragma unroll
  for (int row = 0; row < kRows; ++row) {
    const int r_idx = cute::get<0>(tScS_rc(row, 0));
    float acc = 0.0f;
#pragma unroll
    for (int it = 0; it < kIters; ++it) {
      const int col = (qlane + it * kQuad) * kVec;
#pragma unroll
      for (int d = 0; d < kVec; ++d)
        acc += static_cast<float>(sQ(r_idx, col + d)) * km_bh[col + d];
    }
    qkm[row] = acc;
  }
#pragma unroll
  for (int row = 0; row < kRows; ++row) {
    qkm[row] += __shfl_xor_sync(0xffffffff, qkm[row], 1);
    qkm[row] += __shfl_xor_sync(0xffffffff, qkm[row], 2);
  }
}

// WS persist-D W8A8 (fp8 e4m3 Q/K/V; kQKInt8: Q/K symmetric int8 with s32 QK
// MMA cast to f32, PV stays fp8): same 128 producer + 256 consumer split
// as persist_d.cuh. V is pre-transposed (D x N) by the quantize pre-kernel.
// Blockwise scales: k_scale folded into the log2-domain softmax, v_scale
// absorbed into P's fp8 quantization, q_scale * p_scale applied in epilogue.
// O epilogue: STSM into the freed smem, then TMA store (coalesced). Tail Q
// tiles (partial rows) fall back to direct R->G stores. attn_bias/dropout are
// not supported on this path.
//
// kPQuantPerRow selects the P (softmax probability) quantization granularity
// (see fp8_pscale.cuh for the full math):
//   true  - per-row p_scale = row_max/448; highest accuracy (flat rows fill
//           the e4m3 range); costs a row-max reduction, an extra o_tile
//           fragment and a per-tile rescale pass (higher register pressure).
//   false - fixed p_scale = 1/448 (valid since max(P) <= 1); no reduction,
//           PV accumulates directly into o_acc, single dequant in epilogue.
//           Faster, slightly coarser for rows whose max(P) << 1. Default.
template <typename Traits, typename ElementO, typename TmaQ, typename TmaK,
          typename TmaV, typename TmaO, bool kPQuantPerRow = false>
__global__ void __launch_bounds__(384, 1) persist_d_ws_fwd_cute_w8a8_sm120(
    CUTLASS_GRID_CONSTANT TmaQ const tma_q,
    CUTLASS_GRID_CONSTANT TmaK const tma_k,
    CUTLASS_GRID_CONSTANT TmaV const tma_v,
    CUTLASS_GRID_CONSTANT TmaO const tma_o, ElementO* __restrict__ O,
    float* __restrict__ softmax_lse, const float* __restrict__ q_scale,
    const float* __restrict__ k_scale, const float* __restrict__ v_scale,
    int Nq, int Nkv, int Nh, int Nh_kv, float scale, int Tc, int causal,
    int total_q_rows, int total_kv_rows, int n_rb_q, int n_rb_kv,
    const float* __restrict__ km = nullptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  using namespace cute;
  using Element = typename Traits::Element;      // float_e4m3_t (V/PV)
  using ElementQK = typename Traits::ElementQK;  // int8 (kQKInt8) or e4m3
  using SmemLayoutQ = typename Traits::SmemLayoutQ;
  using SmemLayoutK = typename Traits::SmemLayoutK;
  using SmemLayoutV = typename Traits::SmemLayoutV;
  using TiledMmaQK = typename Traits::TiledMmaQK;
  using TiledMmaPV = typename Traits::TiledMmaPV;
  using SmemCopyAtom = typename Traits::SmemCopyAtom;
  using SmemCopyAtomQK = typename Traits::SmemCopyAtomQK;
  using SmemLayoutO = typename Traits::SmemLayoutO;

  constexpr int kBr = Traits::kBr;
  constexpr int kBc = Traits::kBc;
  constexpr int kHeadDim = Traits::kHeadDim;
  constexpr int kStagesK = Traits::kStagesK;
  constexpr int kStagesV = Traits::kStagesV;
  constexpr int kProducerThreads = 128;
  constexpr int kConsumerThreads = 256;
  static_assert(kHeadDim == 64 || kHeadDim == 128,
                "w8a8 lse correction supports D in {64, 128}");

  constexpr int kQTileElements = cosize(SmemLayoutQ{});
  constexpr int kKTileElements = cosize(SmemLayoutK{});
  constexpr int kVTileElements = cosize(SmemLayoutV{});

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
  const int q_bh = Nb_id * Nh + Nh_id;
  const int kv_bh = Nb_id * Nh_kv + kv_head_idx;
  const float qs = q_scale[static_cast<long>(q_bh) * n_rb_q + Q_tile_id];

  // SMEM: [Q persist | K stages | V stages], 1B per elem (int8 or e4m3).
  extern __shared__ __align__(1024) char shm[];
  ElementQK* q_base = reinterpret_cast<ElementQK*>(shm);
  ElementQK* k_base = q_base + kQTileElements;
  Element* v_base =
      reinterpret_cast<Element*>(k_base + kStagesK * kKTileElements);

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

  if (is_producer) {
    // Release registers to the CTA pool so the two consumer warpgroups can
    // alloc up to 232 regs/thread. Unlike fp16 persist_d, the fp8 path carries
    // P quantization + row-sum + rescale state, so D=128 still needs the extra
    // budget; keeping the static 170-reg ceiling causes ~60 regs of spill.
    cutlass::arch::warpgroup_reg_dealloc<32>();  // sm_120f
    if (wg_tid == 0) {
      auto mQ = domain_offset(
          make_coord(q_row_offset, 0),
          tma_q.get_tma_tensor(make_shape(total_q_rows, Int<kHeadDim>{})));
      auto mK = domain_offset(
          make_coord(kv_row_offset, 0),
          tma_k.get_tma_tensor(make_shape(total_kv_rows, Int<kHeadDim>{})));
      // V^T: per-head (D, Nkv) planes; offset row to the KV head's D plane,
      // col is local n within [0, Nkv). Descriptor dim1 must be Nkv (not the
      // flattened total_kv_rows) so the TMA column coordinate tiles correctly.
      const int v_row_base = (Nb_id * Nh_kv + kv_head_idx) * kHeadDim;
      const int d_total = (total_kv_rows / Nkv) * kHeadDim;
      auto mV = domain_offset(make_coord(v_row_base, _0{}),
                              tma_v.get_tma_tensor(make_shape(d_total, Nkv)));
      auto q_slice = tma_q.get_slice(_0{});
      auto k_slice = tma_k.get_slice(_0{});
      auto v_slice = tma_v.get_slice(_0{});

      auto sQ = make_tensor(make_smem_ptr(q_base), SmemLayoutQ{});
      auto gQ = local_tile(mQ, Shape<Int<kBr>, Int<kHeadDim>>{},
                           make_coord(Q_tile_id, _0{}));
      TmaBarrier::arrive_and_expect_tx(&q_full, sizeof(ElementQK) * size(sQ));
      copy(tma_q.with(q_full), q_slice.partition_S(gQ),
           q_slice.partition_D(sQ));

      for (int s = 0; s < kStagesK - 1; ++s) {
        if (s < Tc_eff) {
          CtaBarrier::wait(&k_empty[s], 0);
          auto sK = make_tensor(make_smem_ptr(k_base + s * kKTileElements),
                                SmemLayoutK{});
          auto gK = local_tile(mK, Shape<Int<kBc>, Int<kHeadDim>>{},
                               make_coord(s, _0{}));
          TmaBarrier::arrive_and_expect_tx(&k_full[s],
                                           sizeof(ElementQK) * size(sK));
          copy(tma_k.with(k_full[s]), k_slice.partition_S(gK),
               k_slice.partition_D(sK));
        }
      }
      for (int s = 0; s < kStagesV - 1; ++s) {
        if (s < Tc_eff) {
          CtaBarrier::wait(&v_empty[s], 0);
          auto sV = make_tensor(make_smem_ptr(v_base + s * kVTileElements),
                                SmemLayoutV{});
          auto gV = local_tile(mV, Shape<Int<kHeadDim>, Int<kBc>>{},
                               make_coord(_0{}, s));
          TmaBarrier::arrive_and_expect_tx(&v_full[s],
                                           sizeof(Element) * size(sV));
          copy(tma_v.with(v_full[s]), v_slice.partition_S(gV),
               v_slice.partition_D(sV));
        }
      }
      for (int tile = 0; tile < Tc_eff; ++tile) {
        {
          const int v_tile = tile + kStagesV - 1;
          if (v_tile < Tc_eff) {
            const int stage_v = v_tile % kStagesV;
            const int phase_v = (v_tile / kStagesV) & 1;
            CtaBarrier::wait(&v_empty[stage_v], phase_v);
            auto sV =
                make_tensor(make_smem_ptr(v_base + stage_v * kVTileElements),
                            SmemLayoutV{});
            auto gV = local_tile(mV, Shape<Int<kHeadDim>, Int<kBc>>{},
                                 make_coord(_0{}, v_tile));
            TmaBarrier::arrive_and_expect_tx(&v_full[stage_v],
                                             sizeof(Element) * size(sV));
            copy(tma_v.with(v_full[stage_v]), v_slice.partition_S(gV),
                 v_slice.partition_D(sV));
          }
        }
        {
          const int k_tile = tile + kStagesK - 1;
          if (k_tile < Tc_eff) {
            const int stage_k = k_tile % kStagesK;
            const int phase_k = (k_tile / kStagesK) & 1;
            CtaBarrier::wait(&k_empty[stage_k], phase_k);
            auto sK =
                make_tensor(make_smem_ptr(k_base + stage_k * kKTileElements),
                            SmemLayoutK{});
            auto gK = local_tile(mK, Shape<Int<kBc>, Int<kHeadDim>>{},
                                 make_coord(k_tile, _0{}));
            TmaBarrier::arrive_and_expect_tx(&k_full[stage_k],
                                             sizeof(ElementQK) * size(sK));
            copy(tma_k.with(k_full[stage_k]), k_slice.partition_S(gK),
                 k_slice.partition_D(sK));
          }
        }
      }
    }
    return;
  }

  // Consumer path. Take the registers released by the producer warpgroup.
  // Unconditional (not gated on kHeadDim != 128 like fp16 persist_d) because
  // the fp8 path carries P quant + row-sum + rescale state even at D=128.
  cutlass::arch::warpgroup_reg_alloc<232>();

  // Release the initial K/V stages *before* waiting on Q. The producer gates
  // its first K/V TMA issues on these arrives; they carry no dependency on Q
  // data, so arriving early lets K(0)/V(0) TMAs overlap the Q TMA in flight
  // instead of serializing behind Q arrival + consumer wakeup.
  for (int s = 0; s < kStagesK; ++s)
    CtaBarrier::arrive(&k_empty[s]);
  for (int s = 0; s < kStagesV; ++s)
    CtaBarrier::arrive(&v_empty[s]);

  TmaBarrier::wait(&q_full, 0);
  cutlass::arch::fence_view_async_shared();

  TiledMmaQK tiled_mma_qk;
  TiledMmaPV tiled_mma_pv;
  auto thr_mma_qk = tiled_mma_qk.get_thread_slice(wg_tid);
  auto thr_mma_pv = tiled_mma_pv.get_thread_slice(wg_tid);

  auto s2r_copy_q = make_tiled_copy_A(SmemCopyAtomQK{}, tiled_mma_qk);
  auto s2r_copy_k = make_tiled_copy_B(SmemCopyAtomQK{}, tiled_mma_qk);
  auto s2r_copy_v = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma_pv);
  auto s2r_thr_q = s2r_copy_q.get_thread_slice(wg_tid);
  auto s2r_thr_k = s2r_copy_k.get_thread_slice(wg_tid);
  auto s2r_thr_v = s2r_copy_v.get_thread_slice(wg_tid);

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

  const float scale_orig = scale;
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

  // Per-row mode only: raw PV output buffer + per-row quant scales. Fixed
  // mode accumulates straight into o_acc and needs neither.
  float o_tile[kPQuantPerRow ? kOElemsPerFrag : 1];
  float p_scale[kPQuantPerRow ? kORows : 1];

  auto sQ = make_tensor(make_smem_ptr(q_base), SmemLayoutQ{});
  auto tCrQ = thr_mma_qk.partition_fragment_A(sQ);
  auto tQsQ_s2r = s2r_thr_q.partition_S(sQ);

  ffpa_cute::ReorgCFp8toAFp8 reorg;

#pragma unroll 1
  for (int kv_tile = 0; kv_tile < Tc_eff; ++kv_tile) {
    const int k_stg = kv_tile % kStagesK;
    const int k_phase = (kv_tile / kStagesK) & 1;
    const int v_stg = kv_tile % kStagesV;
    const int v_phase = (kv_tile / kStagesV) & 1;

    const float ks = k_scale[static_cast<long>(kv_bh) * n_rb_kv + kv_tile];
    const float vs = v_scale[static_cast<long>(kv_bh) * n_rb_kv + kv_tile];
    const float s_dequant = qs * ks;  // linear dequant of S

    // QK GEMM: fp8xfp8->fp32, or int8xint8->s32 when kQKInt8.
    TmaBarrier::wait(&k_full[k_stg], k_phase);
    cutlass::arch::fence_view_async_shared();

    auto sK = make_tensor(make_smem_ptr(k_base + k_stg * kKTileElements),
                          SmemLayoutK{});
    auto tCrK = thr_mma_qk.partition_fragment_B(sK);
    auto tKsK_s2r = s2r_thr_k.partition_S(sK);

    auto tCrS = partition_fragment_C(tiled_mma_qk, Shape<Int<kBr>, Int<kBc>>{});
    clear(tCrS);
    ffpa_cute::gemm_ss(tCrS, tCrQ, tCrK, tQsQ_s2r, tKsK_s2r, tiled_mma_qk,
                       s2r_copy_q, s2r_copy_k, s2r_thr_q, s2r_thr_k);
    CtaBarrier::arrive(&k_empty[k_stg]);

    // int8 QK: cast the s32 acc to f32 in place over the same 4B regs (no
    // extra registers); identity view on the fp8 path.
    auto tCrSf =
        make_tensor(reinterpret_cast<float*>(tCrS.data()), tCrS.layout());
    if constexpr (Traits::kQKInt8) {
#pragma unroll
      for (int i = 0; i < size(tCrS); ++i)
        tCrSf(i) = static_cast<float>(tCrS(i));
    }

    // S -> log2 domain with k_scale folded in (blockwise dequant of S).
    auto scores = make_tensor(
        tCrSf.data(), ffpa_cute::convert_layout_acc_rowcol(tCrS.layout()));
    // Known causal accuracy limit (early-row error): masked upper-triangle
    // columns contribute EXACTLY 0 (P=0 after -inf softmax), the error comes
    // from the VALID terms -- early rows attend only i+1 keys, so per-element
    // quant errors are not averaged out (~1/sqrt(n_valid) decay with row):
    //   row 0 is the pure probe: O0 = V0 quant rounding itself (~0.08 @amp.5);
    //   fp8-P weight error and QK dS->dP error hit the few dominant weights
    //   undiluted. Late rows average hundreds of independent errors instead.
    // Rejected fix: fp16 accurate-PV re-run on masked tiles (VT16 plane) made
    // causal +38~61% slower (gmem-direct fp16 B + split quantize + VT16
    // traffic) and was removed. Current mitigation: int8 QK (auto-default for
    // causal) removes only the dS part. Future candidates: correct only the
    // few attended KV columns of the diagonal tile in fp16 (the PV16 idea
    // scoped down -- its cost was the implementation, not the concept), or
    // finer V quant block granularity (currently 128 keys/block).
    const int kv_valid = Nkv - kv_tile * kBc;
    const bool tile_needs_mask =
        (kv_valid < kBc) || (kv_tile >= mask_start_tile);
    if (tile_needs_mask) {
#pragma unroll
      for (int row = 0; row < kSRows; ++row) {
        const int q_pos = Br_base + get<0>(tScS_rc(row, 0)) + kv_offset;
#pragma unroll
        for (int col = 0; col < kSCols; ++col) {
          float s = scores(row, col) * s_dequant * scale;
          if (get<1>(tScS_rc(row, col)) >= kv_valid)
            s = -INFINITY;
          if (kv_tile >= mask_start_tile) {
            const int k_pos = kv_tile * kBc + get<1>(tScS_rc(row, col));
            if (k_pos > q_pos)
              s = -INFINITY;
          }
          scores(row, col) = s;
        }
      }
    }

    float row_scale[kORows];
    const float vs448 = vs * ffpa_cute::kE4m3Max;
    if constexpr (kPQuantPerRow) {
      if (!tile_needs_mask) {
        const float sd = s_dequant * scale;
#pragma unroll
        for (int row = 0; row < kSRows; ++row)
#pragma unroll
          for (int col = 0; col < kSCols; ++col)
            scores(row, col) *= sd;
      }
      ffpa_cute::online_safe_softmax<decltype(scores), decltype(tScS_rc),
                                     kORows>(scores, tScS_rc, 1.0f, row_max,
                                             row_sum, row_scale,
                                             Traits::kRescaleThreshold);
    } else {
      // Fixed mode: fold the P quant scale vs*448 into the exp2 offset so the
      // softmax emits P*vs*448 directly; row_sum is folded back to the true
      // probability domain inside (see fp8_pscale.cuh). Unmasked tiles defer
      // the s_dequant multiply into the softmax (one multiply per exp instead
      // of a separate scale pass over all 64 scores).
      const float softmax_scale_eff =
          tile_needs_mask ? 1.0f : s_dequant * scale;
      ffpa_cute::online_softmax_fp8_fixed<true, decltype(scores),
                                          decltype(tScS_rc), kORows>(
          scores, tScS_rc, softmax_scale_eff, row_max, row_sum, row_scale,
          log2f(vs448), 1.0f / vs448, Traits::kRescaleThreshold);
    }

    // Rescale o_acc (online softmax); deferred until p_scale is known.
    bool local_need_rescale = false;
#pragma unroll
    for (int r = 0; r < kORows; ++r)
      local_need_rescale = local_need_rescale || (row_scale[r] < 1.0f);
    const bool need_rescale = __any_sync(0xffffffff, local_need_rescale);

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

    // P -> e4m3 A operand (see fp8_pscale.cuh). Per-row mode needs the row
    // max first, then scales+converts; fixed mode was pre-scaled by the
    // softmax and only converts (packed e4m3x2) + reorgs.
    if constexpr (kPQuantPerRow) {
      ffpa_cute::pscale_per_row(scores, p_scale);
      ffpa_cute::quantize_p_frag<true>(scores, tCrSf, vs, p_scale, reorg);
    } else {
      ffpa_cute::quantize_p_frag_prescaled(tCrSf, reorg);
    }

    // PV GEMM: A = P in regs, B = V^T from smem. Per-row dequant factor
    // varies per row -> separate o_tile; fixed mode accumulates into o_acc.
    TmaBarrier::wait(&v_full[v_stg], v_phase);
    cutlass::arch::fence_view_async_shared();

    auto sV = make_tensor(make_smem_ptr(v_base + v_stg * kVTileElements),
                          SmemLayoutV{});
    auto tCrV = thr_mma_pv.partition_fragment_B(sV);
    auto tVsV_s2r = s2r_thr_v.partition_S(sV);

    auto tCrP =
        make_tensor(reinterpret_cast<Element*>(tCrSf.data()),
                    Layout<Shape<Shape<_4, _2, _2>, _1, Int<kBc / 32>>>{});
    if constexpr (kPQuantPerRow) {
      auto tCrTile = make_tensor(make_rmem_ptr(o_tile), OFragLayout{});
#pragma unroll
      for (int i = 0; i < kOElemsPerFrag; ++i)
        o_tile[i] = 0.0f;
      ffpa_cute::gemm_rs(tCrTile, tCrP, tCrV, tVsV_s2r, tiled_mma_pv,
                         s2r_copy_v, s2r_thr_v);
      auto tCrO = make_tensor(make_rmem_ptr(o_acc), OFragLayout{});
      auto tCrO_rc = make_tensor(
          tCrO.data(), ffpa_cute::convert_layout_acc_rowcol(tCrO.layout()));
      auto tCrTile_rc =
          make_tensor(tCrTile.data(),
                      ffpa_cute::convert_layout_acc_rowcol(tCrTile.layout()));
      ffpa_cute::accumulate_p_tile(tCrO_rc, tCrTile_rc, p_scale);
    } else {
      // Tensor-core row sum over the quantized P regs (replaces the fp32
      // FADD/shfl reduction; softmax<true> only rescaled row_sum so far).
      ffpa_cute::pscale_rowsum_mma(tCrP, row_sum, 1.0f / vs448);
      auto tCrO = make_tensor(make_rmem_ptr(o_acc), OFragLayout{});
      ffpa_cute::gemm_rs(tCrO, tCrP, tCrV, tVsV_s2r, tiled_mma_pv, s2r_copy_v,
                         s2r_thr_v);
    }
    CtaBarrier::arrive(&v_empty[v_stg]);
  }

  // Epilogue: O = O / row_sum (per-row mode already dequantized per tile) or
  // O = O * kFP8FixedPScale / row_sum (fixed mode keeps one global domain).
  // Smooth-K lse correction: dot(Q8_row, km) must be read off sQ BEFORE the
  // full-tile STSM aliases q_base as O staging; the lse gmem write itself is
  // deferred to the end of the epilogue.
  float qkm[kORows];
  const bool smooth_lse = (softmax_lse != nullptr) && (km != nullptr);
  {
    cutlass::arch::NamedBarrier::sync(kConsumerThreads, 0);

    if (smooth_lse)
      smooth_k_qk_dot<kHeadDim, kORows>(
          sQ, tScS_rc, km + static_cast<long>(kv_bh) * kHeadDim, qkm);

    auto tCrO = make_tensor(make_rmem_ptr(o_acc), OFragLayout{});
    auto tCrO_rc = make_tensor(
        tCrO.data(), ffpa_cute::convert_layout_acc_rowcol(tCrO.layout()));
#pragma unroll
    for (int row = 0; row < kORows; ++row) {
      const float inv_sum = (row_sum[row] == 0.0f) ? 1.0f : 1.0f / row_sum[row];
      const float mul =
          kPQuantPerRow ? inv_sum : inv_sum * ffpa_cute::kFP8FixedPScale;
#pragma unroll
      for (int col = 0; col < kOCols; ++col)
        tCrO_rc(row, col) *= mul;
    }
    auto tCrOHalf = ffpa_cute::convert_type<ElementO>(tCrO);

    if (Br_base + kBr <= Nq) {
      // Full tile: STSM into the freed smem (Q/K/V all consumed), then one
      // coalesced TMA store. sO aliases q_base: kBr*kHeadDim ElementO elems
      // (32KB for D=128 fp16) fit the freed [Q|K|V] region.
      auto sO = make_tensor(make_smem_ptr(reinterpret_cast<ElementO*>(q_base)),
                            SmemLayoutO{});
      auto r2s_copy = make_tiled_copy_C(
          Copy_Atom<SM90_U32x4_STSM_N, ElementO>{}, tiled_mma_pv);
      auto r2s_thr = r2s_copy.get_slice(wg_tid);
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
      // Tail tile: rows past Nq would alias the next head in the flattened
      // [total_q_rows, D] TMA space, so store R->G with a row guard.
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
        float lse = (row_max[row] + log2f(row_sum[row])) * FFPA_M_LN2;
        if (smooth_lse)
          lse += scale_orig * qs * qkm[row];
        const int global_row = Br_base + get<0>(tScS_rc(row, 0));
        if (global_row < Nq)
          softmax_lse[lse_base + global_row] = lse;
      }
    }
  }
#endif  // defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
}

}  // namespace ffpa_w8a8

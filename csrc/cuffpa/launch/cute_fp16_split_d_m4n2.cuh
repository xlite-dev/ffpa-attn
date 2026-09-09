#pragma once
// CuTe fp16/bf16 split_d M4N2 launcher: kernel include + bias plan +
// variant body (`_v`), split out of launch/cute_fp16.cuh so the per-tag
// variant TUs (env.py) preprocess exactly one kernel table.
#include "launch/common.cuh"
#if defined(ENABLE_FFPA_CUTE_EXT) && defined(ENABLE_FFPA_TMA_EXT)
#include "cute/sm_120/split_d_m4n2.cuh"

namespace ffpa {

// Single-source final bias-mode decision: the umbrella wrapper dispatch
// and the variant body both call this, so a demote-rule drift fails the
// variant's tag TORCH_CHECK instead of silently launching the wrong
// kernel. Mirrors the pre-split inline plan in cute_fp16.cuh.
template <typename kDataType, const int kHeadDim, const int kStage>
inline FfpaBiasTilePlan fp16_split_d_m4n2_bias_plan(
    const FfpaBiasParams& bias_p, int Nb, int Nh, int Nq, int Nkv) {
  using Element = std::conditional_t<std::is_same_v<kDataType, __half>,
                                     cutlass::half_t, cutlass::bfloat16_t>;
  constexpr int kBr = 64;
  constexpr int kBc = 64;
  constexpr int kQKDChunk = 64;
  constexpr int kVDChunk = 64;
  constexpr int kStagesQK = (kStage < 2 ? 2 : (kStage > 3 ? 3 : kStage));
  using Traits = ffpa_cute::FFPAAttnCuTeSplitDM4N2Traits<
      kHeadDim, kBr, kBc, kQKDChunk, kVDChunk, kStagesQK, kStagesQK, Element>;
  constexpr int kBaseSmemBytes = Traits::kSmemElems * sizeof(Element);
  constexpr int kSmemBudgetBytes = 99 * 1024;
  FfpaBiasTilePlan plan = ffpa_bias_tile_plan_of(bias_p, Nb, Nh, Nq, Nkv);
  const int bias_stages = (plan.mode == 2) ? 2 : 1;
  if (kBaseSmemBytes + plan.tile_bytes(kBr, kBc, bias_stages) >
      kSmemBudgetBytes)
    plan.mode = 0;
  // Row-broadcast resident (mode 3): the upgrade must not trade blocks/SM
  // for the removed barrier — keep the smem-driven occupancy of the base
  // layout (measured on fp8 m4n2: mode 3 dropped 3 CTA/SM to 1, 4.9%
  // slower than the gmem fallback).
  if (plan.mode == 2) {
    const long long kv_pad = (Nkv + kBc - 1) / kBc * kBc;
    const long long base_align = (kBaseSmemBytes + 15) & ~15;
    const long long resident = base_align + kv_pad * plan.elem_size;
    if (resident <= kSmemBudgetBytes &&
        101376 / resident >= 101376 / base_align)
      plan.mode = 3;
  }
  return plan;
}

}  // namespace ffpa

// M4N2 launcher: kBr=64, kBc=64, atom_layout=(4,2,1). Uses
// FFPAAttnCuTeSplitDM4N2Traits (P SMEM roundtrip + cross-N-warp softmax).
// Dispatched for D>=768 to avoid M8N1's register spill (O=D/2 > 255).
// Variant tags: kBiasOn (attn_bias tensor present), kBiasPlanMode (bias
// tile mode: 0 = gmem-direct fallback, 1 = dense [kBr,kBc] TMA tile,
// 2 = row-broadcast TMA, 3 = resident row vector), kBias4BytesPerElem
// (1 = 4-byte fp32 mask, 0 = 2-byte fp16/bf16 mask), kHasDropout
// (dropout plumbing). The umbrella wrapper computes the plan and picks
// the exact tag; explicit instantiations live in the generated variant
// TUs.
template <typename kDataType, const int kHeadDim, const int kStage, int kBiasOn,
          int kBiasPlanMode, int kBias4BytesPerElem, int kHasDropout>
void launch_cute_fwd_split_d_m4n2_sm120_v(torch::Tensor Q, torch::Tensor K,
                                          torch::Tensor V, torch::Tensor O,
                                          torch::Tensor attn_bias,
                                          torch::Tensor softmax_lse, int causal,
                                          double softmax_scale,
                                          double dropout_p, int64_t philox_seed,
                                          int64_t philox_offset) {
  using namespace cute;

  constexpr int kBr = 64;
  constexpr int kBc = 64;
  constexpr int kQKDChunk = 64;
  constexpr int kVDChunk = 64;
  constexpr int kStagesQK = (kStage < 2 ? 2 : (kStage > 3 ? 3 : kStage));
  constexpr int kStagesPV = kStagesQK;
  constexpr int kNumThreads = 256;

  using Element = std::conditional_t<std::is_same_v<kDataType, __half>,
                                     cutlass::half_t, cutlass::bfloat16_t>;
  using Traits = ffpa_cute::FFPAAttnCuTeSplitDM4N2Traits<
      kHeadDim, kBr, kBc, kQKDChunk, kVDChunk, kStagesQK, kStagesPV, Element>;
  using SmemLayoutQ = typename Traits::SmemLayoutQ;
  using SmemLayoutK = typename Traits::SmemLayoutK;
  using SmemLayoutV = typename Traits::SmemLayoutV;
  using SmemLayoutO = typename Traits::SmemLayoutO;

  const int Nb = Q.size(0);
  const int Nh = Q.size(1);
  const int Nh_kv = K.size(1);
  const int Nq = Q.size(2);
  const int Nkv = K.size(2);
  const int Tc = utils::div_ceil(Nkv, kBc);
  const float scale = static_cast<float>(softmax_scale);

  const FfpaBiasParams bias = ffpa_bias_params_of(attn_bias, Q, K);
  const int bias_on = bias.ptr != nullptr ? 1 : 0;
  const void* attn_bias_ptr = bias.ptr;
  const int attn_bias_dtype = bias.dtype;
  const float dropout_p_f = static_cast<float>(dropout_p);
  const unsigned long long philox_seed_u =
      static_cast<unsigned long long>(philox_seed);
  const unsigned long long philox_offset_u =
      static_cast<unsigned long long>(philox_offset);

  const dim3 block(kNumThreads, 1, 1);
  const dim3 grid(utils::div_ceil(Nq, kBr), Nb * Nh, 1);

  const c10::cuda::OptionalCUDAGuard device_guard(Q.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  const int total_q_rows = Nb * Nh * Nq;
  const int total_kv_rows = Nb * Nh_kv * Nkv;

  // NHD (diffusers BNHD) permute views — including strided fused-QKV chunk
  // rows — are consumed natively via flat (B*N, H*D) TMA rows carrying the
  // tensor's own row stride (H*D packed; wider for strided views); O stays
  // BHND-packed (the caller allocates it packed and re-views).
  const bool q_nhd = ffpa_is_nhd_view(Q) || ffpa_is_strided_nhd(Q);
  if (ffpa_is_strided_nhd(Q))
    ffpa_check_strided_nhd_aligned(Q, "Q");
  if (!q_nhd)
    TORCH_CHECK(Q.stride(3) == 1 && Q.stride(2) == (long)kHeadDim &&
                    Q.stride(1) == (long)Nq * kHeadDim &&
                    Q.stride(0) == (long)Nh * Nq * kHeadDim,
                "ffpa_attn: Q must be BHND-contiguous or an NHD (BNHD) view");
  const bool k_nhd = ffpa_is_nhd_view(K) || ffpa_is_strided_nhd(K);
  const bool v_nhd = ffpa_is_nhd_view(V) || ffpa_is_strided_nhd(V);
  TORCH_CHECK(k_nhd == v_nhd,
              "ffpa_attn: K and V must share the same memory layout family "
              "(BHND-packed or NHD)");
  if (ffpa_is_strided_nhd(K))
    ffpa_check_strided_nhd_aligned(K, "K");
  if (ffpa_is_strided_nhd(V))
    ffpa_check_strided_nhd_aligned(V, "V");
  const bool kv_nhd = k_nhd;
  if (!kv_nhd) {
    TORCH_CHECK(K.stride(3) == 1 && K.stride(2) == (long)kHeadDim &&
                    K.stride(1) == (long)Nkv * kHeadDim &&
                    K.stride(0) == (long)Nh_kv * Nkv * kHeadDim,
                "ffpa_attn: K must be BHND-contiguous or an NHD (BNHD) view");
    TORCH_CHECK(V.stride(3) == 1 && V.stride(2) == (long)kHeadDim &&
                    V.stride(1) == (long)Nkv * kHeadDim &&
                    V.stride(0) == (long)Nh_kv * Nkv * kHeadDim,
                "ffpa_attn: V must be BHND-contiguous or an NHD (BNHD) view");
  }

  // O output TMA store descriptor: BHND flat [total_q_rows,kHeadDim] or
  // NHD (diffusers BNHD packed) [Nb*Nq, Nh*kHeadDim] with the head
  // selecting the column-tile group (kernel folds Nh_id*kDChunksV into
  // the v_chunk walk). Both branches use dynamic int64 extents/strides so
  // TmaO has a single type and the kernel takes a runtime nhd_out branch.
  const bool nhd_out = ffpa_is_nhd_view(O);
  auto gO =
      nhd_out
          ? make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(O.data_ptr())),
                        make_shape((int64_t)Nb * Nq, (int64_t)Nh * kHeadDim),
                        make_stride((int64_t)Nh * kHeadDim, _1{}))
          : make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(O.data_ptr())),
                        make_shape((int64_t)total_q_rows, (int64_t)kHeadDim),
                        make_stride((int64_t)kHeadDim, _1{}));
  auto tma_o = make_tma_copy(SM90_TMA_STORE{}, gO, SmemLayoutO{},
                             Shape<Int<kBr>, Int<kVDChunk>>{}, _1{});

  auto make_tma_q = [&](auto q_c) {
    if constexpr (decltype(q_c)::value) {
      auto gQ =
          make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(Q.data_ptr())),
                      make_shape((long)Nb * Nq, (long)Nh * kHeadDim),
                      make_stride(Q.stride(2), _1{}));
      return make_tma_copy(SM90_TMA_LOAD{}, gQ, SmemLayoutQ{},
                           Shape<Int<kBr>, Int<kQKDChunk>>{}, _1{});
    } else {
      auto gQ =
          make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(Q.data_ptr())),
                      make_shape(total_q_rows, Int<kHeadDim>{}),
                      make_stride(Int<kHeadDim>{}, _1{}));
      return make_tma_copy(SM90_TMA_LOAD{}, gQ, SmemLayoutQ{},
                           Shape<Int<kBr>, Int<kQKDChunk>>{}, _1{});
    }
  };
  auto make_tma_k = [&](auto kv_c) {
    if constexpr (decltype(kv_c)::value) {
      auto gK =
          make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(K.data_ptr())),
                      make_shape((long)Nb * Nkv, (long)Nh_kv * kHeadDim),
                      make_stride(K.stride(2), _1{}));
      return make_tma_copy(SM90_TMA_LOAD{}, gK, SmemLayoutK{},
                           Shape<Int<kBc>, Int<kQKDChunk>>{}, _1{});
    } else {
      auto gK =
          make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(K.data_ptr())),
                      make_shape(total_kv_rows, Int<kHeadDim>{}),
                      make_stride(Int<kHeadDim>{}, _1{}));
      return make_tma_copy(SM90_TMA_LOAD{}, gK, SmemLayoutK{},
                           Shape<Int<kBc>, Int<kQKDChunk>>{}, _1{});
    }
  };
  auto make_tma_v = [&](auto kv_c) {
    if constexpr (decltype(kv_c)::value) {
      auto gV =
          make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(V.data_ptr())),
                      make_shape((long)Nb * Nkv, (long)Nh_kv * kHeadDim),
                      make_stride(V.stride(2), _1{}));
      return make_tma_copy(SM90_TMA_LOAD{}, gV, SmemLayoutV{},
                           Shape<Int<kBc>, Int<kVDChunk>>{}, _1{});
    } else {
      auto gV =
          make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(V.data_ptr())),
                      make_shape(total_kv_rows, Int<kHeadDim>{}),
                      make_stride(Int<kHeadDim>{}, _1{}));
      return make_tma_copy(SM90_TMA_LOAD{}, gV, SmemLayoutV{},
                           Shape<Int<kBc>, Int<kVDChunk>>{}, _1{});
    }
  };

  // PC-0 bias tile: classify the mask shape; fall back to the gmem-direct
  // path when the tile exceeds the smem budget. The TMA box must be fully
  // static (cute's vectorization inference rejects dynamic modes), so mode
  // and dtype become template splits; the direct fallback reuses the
  // row-vec 2B descriptor shape as a never-issued dummy.
  FfpaBiasTilePlan bias_plan;
  if (bias_on) {
    bias_plan = ffpa::fp16_split_d_m4n2_bias_plan<kDataType, kHeadDim, kStage>(
        bias, Nb, Nh, Nq, Nkv);
  }
  TORCH_CHECK(kBiasOn == bias_on &&
                  kBiasPlanMode == (bias_on ? bias_plan.mode : 0) &&
                  kBias4BytesPerElem ==
                      ((kBiasPlanMode != 0 && bias.dtype == 3) ? 1 : 0) &&
                  kHasDropout == (dropout_p > 0.0 ? 1 : 0),
              "ffpa_attn: fp16 m4n2 D=", kHeadDim,
              " variant tag mismatch (wrapper dispatch vs plan)");
  const int kBaseSmemBytes = Traits::kSmemElems * sizeof(Element);
  const int bias_stages = (bias_plan.mode == 2) ? 2 : 1;
  const auto make_tma_bias = [&](auto mode_c, auto b4_c) {
    constexpr int kBiasModeT = decltype(mode_c)::value;
    constexpr int kBias4B = decltype(b4_c)::value;
    constexpr int bias_cols = kBc * (kBias4B ? 2 : 1);
    // The descriptor is live only when the template mode matches the plan
    // (mode 2 then spans the real [m_total, Nkv] plane); every other
    // combination is a never-issued dummy, where bias_cols keeps the TMA
    // descriptor's 16B outer-stride assert satisfied.
    const bool bias_desc_live = bias_plan.mode == kBiasModeT;
    const int64_t plane_rows =
        bias_desc_live ? (int64_t)bias_plan.m_total : (int64_t)1;
    const int64_t plane_cols =
        (int64_t)std::max<long long>(Nkv, 1) * (kBias4B ? 2 : 1);
    const int64_t plane_row_stride =
        bias_desc_live
            ? (kBiasModeT == 1 ? (int64_t)bias.stride_m * (kBias4B ? 2 : 1)
                               : plane_cols)
            : (int64_t)bias_cols;
    // mode 0 (any demote reason, incl. an unaligned mask ptr) never issues:
    // point every descriptor at the anchor so the 16B address assert holds.
    const uint16_t* bias_desc_base =
        bias_plan.mode != 0 ? reinterpret_cast<const uint16_t*>(attn_bias_ptr)
                            : &kBiasDummyAnchor;
    auto gB = make_tensor(make_gmem_ptr(bias_desc_base),
                          make_shape(plane_rows, plane_cols),
                          make_stride(plane_row_stride, _1{}));
    auto sB = [&] {
      if constexpr (kBiasModeT == 1)
        return make_layout(Shape<Int<kBr>, Int<bias_cols>>{},
                           Stride<Int<bias_cols>, _1>{});
      else
        return make_layout(Shape<_1, Int<bias_cols>>{},
                           Stride<Int<bias_cols>, _1>{});
    }();
    return make_tma_copy(SM90_TMA_LOAD{}, gB, sB, shape(sB), _1{});
  };
  auto tma_bias_d16 = make_tma_bias(std::integral_constant<int, 1>{},
                                    std::integral_constant<int, 0>{});
  auto tma_bias_d32 = make_tma_bias(std::integral_constant<int, 1>{},
                                    std::integral_constant<int, 1>{});
  auto tma_bias_r16 = make_tma_bias(std::integral_constant<int, 2>{},
                                    std::integral_constant<int, 0>{});
  auto tma_bias_r32 = make_tma_bias(std::integral_constant<int, 2>{},
                                    std::integral_constant<int, 1>{});
  const int kSmemBytes =
      ((kBaseSmemBytes + 15) & ~15) +
      (int)((bias_plan.mode == 3)
                ? ((long long)(Nkv + kBc - 1) / kBc * kBc) * bias_plan.elem_size
                : bias_plan.tile_bytes(kBr, kBc, bias_stages));

  float* softmax_lse_ptr =
      softmax_lse.numel() > 0 ? softmax_lse.data_ptr<float>() : nullptr;
  auto O_ptr = reinterpret_cast<Element*>(O.data_ptr());

  const auto run = [&](auto tma_q, auto tma_k, auto tma_v, auto q_c,
                       auto kv_c) {
    constexpr bool kNhdQ = decltype(q_c)::value;
    constexpr bool kNhdKV = decltype(kv_c)::value;

    auto launch_variant = [&](auto kernel_func, auto tma_bias_sel) {
      cudaFuncSetAttribute(
          kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
      kernel_func<<<grid, block, kSmemBytes, stream>>>(
          tma_q, tma_k, tma_v, tma_o, tma_bias_sel, O_ptr, softmax_lse_ptr, Nq,
          Nkv, Nh, Nh_kv, scale, Tc, causal, total_q_rows, total_kv_rows,
          attn_bias_ptr, attn_bias_dtype, bias.stride_b, bias.stride_h,
          bias.stride_m, bias.stride_n, dropout_p_f, philox_seed_u,
          philox_offset_u, nhd_out,
          bias_plan.mode != 0 ? bias_plan.m_total : (long long)1);
    };

    using TmaQ = decltype(tma_q);
    using TmaK = decltype(tma_k);
    using TmaV = decltype(tma_v);
    using TmaO = decltype(tma_o);
    // tag table below is instantiated once per compile-time tag (runtime
    // plan -> tag via the umbrella wrapper); the NHD axes stay a runtime
    // split because the TMA descriptor types are function-local decltype.
    // Bias-bearing rows pass (kBiasPlanMode, kBias4BytesPerElem, 1) -> kernel
    // (kBiasMode, kBias4B, kHasAttnBias); the mode/4B values ride the template
    // variables so the slots cannot rotate (kernel expects mode first).
    if constexpr (kBiasOn == 0) {
      launch_variant(split_d_m4n2_fwd_cute_sm120<Traits, TmaQ, TmaK, TmaV, TmaO,
                                                 decltype(tma_bias_r16), 0, 0,
                                                 0, kHasDropout, kNhdQ, kNhdKV>,
                     tma_bias_r16);
    } else if constexpr (kBiasPlanMode == 1) {
      if constexpr (kBias4BytesPerElem)
        launch_variant(
            split_d_m4n2_fwd_cute_sm120<Traits, TmaQ, TmaK, TmaV, TmaO,
                                        decltype(tma_bias_d32), kBiasPlanMode,
                                        kBias4BytesPerElem, 1, kHasDropout,
                                        kNhdQ, kNhdKV>,
            tma_bias_d32);
      else
        launch_variant(
            split_d_m4n2_fwd_cute_sm120<Traits, TmaQ, TmaK, TmaV, TmaO,
                                        decltype(tma_bias_d16), kBiasPlanMode,
                                        kBias4BytesPerElem, 1, kHasDropout,
                                        kNhdQ, kNhdKV>,
            tma_bias_d16);
    } else if constexpr (kBiasPlanMode == 2) {
      if constexpr (kBias4BytesPerElem)
        launch_variant(
            split_d_m4n2_fwd_cute_sm120<Traits, TmaQ, TmaK, TmaV, TmaO,
                                        decltype(tma_bias_r32), kBiasPlanMode,
                                        kBias4BytesPerElem, 1, kHasDropout,
                                        kNhdQ, kNhdKV>,
            tma_bias_r32);
      else
        launch_variant(
            split_d_m4n2_fwd_cute_sm120<Traits, TmaQ, TmaK, TmaV, TmaO,
                                        decltype(tma_bias_r16), kBiasPlanMode,
                                        kBias4BytesPerElem, 1, kHasDropout,
                                        kNhdQ, kNhdKV>,
            tma_bias_r16);
    } else if constexpr (kBiasPlanMode == 3) {
      // resident row-vector: no TMA issue in-kernel, descriptor unused.
      if constexpr (kBias4BytesPerElem)
        launch_variant(
            split_d_m4n2_fwd_cute_sm120<Traits, TmaQ, TmaK, TmaV, TmaO,
                                        decltype(tma_bias_r32), kBiasPlanMode,
                                        kBias4BytesPerElem, 1, kHasDropout,
                                        kNhdQ, kNhdKV>,
            tma_bias_r32);
      else
        launch_variant(
            split_d_m4n2_fwd_cute_sm120<Traits, TmaQ, TmaK, TmaV, TmaO,
                                        decltype(tma_bias_r16), kBiasPlanMode,
                                        kBias4BytesPerElem, 1, kHasDropout,
                                        kNhdQ, kNhdKV>,
            tma_bias_r16);
    } else {
      // mode 0 demote: gmem-direct bias still needs kHasAttnBias=1.
      launch_variant(
          split_d_m4n2_fwd_cute_sm120<
              Traits, TmaQ, TmaK, TmaV, TmaO, decltype(tma_bias_r16),
              kBiasPlanMode, kBias4BytesPerElem, 1, kHasDropout, kNhdQ, kNhdKV>,
          tma_bias_r16);
    }
  };

  if (kv_nhd) {
    if (q_nhd)
      run(make_tma_q(std::true_type{}), make_tma_k(std::true_type{}),
          make_tma_v(std::true_type{}), std::true_type{}, std::true_type{});
    else
      run(make_tma_q(std::false_type{}), make_tma_k(std::true_type{}),
          make_tma_v(std::true_type{}), std::false_type{}, std::true_type{});
  } else {
    if (q_nhd)
      run(make_tma_q(std::true_type{}), make_tma_k(std::false_type{}),
          make_tma_v(std::false_type{}), std::true_type{}, std::false_type{});
    else
      run(make_tma_q(std::false_type{}), make_tma_k(std::false_type{}),
          make_tma_v(std::false_type{}), std::false_type{}, std::false_type{});
  }
}

#endif  // ENABLE_FFPA_CUTE_EXT && ENABLE_FFPA_TMA_EXT

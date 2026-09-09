#pragma once
// CuTe fp16/bf16 family umbrella: per-impl header includes + the
// tag-dispatch wrappers (same signatures as the pre-split launchers, so
// dispatch/cute_fp16.cuh is unchanged). The kernel tables live in the
// per-impl headers' `_v` templates, explicitly instantiated one tag per
// generated variant TU; the extern table at the tail suppresses
// re-instantiation from the family TUs.
#include "launch/common.cuh"
#ifdef ENABLE_FFPA_CUTE_EXT
#include "launch/cute_fp16_split_d_sm80.cuh"
#ifdef ENABLE_FFPA_TMA_EXT
#include "launch/cute_fp16_split_d.cuh"
#include "launch/cute_fp16_persist_d.cuh"
#include "launch/cute_fp16_split_d_m4n2.cuh"

// Tag-dispatch wrappers: compute the plan once (same single-source plan
// helpers the variant bodies re-check), then pick the exact variant tag.
// A demote-rule drift is a loud TORCH_CHECK in the variant body, never a
// silent wrong-kernel launch.
template <typename kDataType, const int kHeadDim, const int kStage>
void launch_cute_fwd_persist_d_sm120(torch::Tensor Q, torch::Tensor K,
                                     torch::Tensor V, torch::Tensor O,
                                     torch::Tensor attn_bias,
                                     torch::Tensor softmax_lse, int causal,
                                     double softmax_scale, double dropout_p,
                                     int64_t philox_seed,
                                     int64_t philox_offset) {
  if constexpr (kHeadDim % 32 == 0 && kHeadDim >= 32 && kHeadDim <= 256) {
    const FfpaBiasParams bias = ffpa_bias_params_of(attn_bias, Q, K);
    const int bias_on = bias.ptr != nullptr ? 1 : 0;
    FfpaBiasTilePlan plan;
    if (bias_on)
      plan = ffpa::fp16_persist_d_bias_plan<kDataType, kHeadDim, kStage>(
          bias, Q.size(0), Q.size(1), Q.size(2), K.size(2));
    const int mode = bias_on ? plan.mode : 0;
    const int b4 = (mode != 0 && bias.dtype == 3) ? 1 : 0;
#ifndef ENABLE_FFPA_CUDA_MASK_FP32
    // mode 0 (gmem-direct) reads the mask dtype at runtime, so fp32 stays
    // legal there; only the TMA tile modes need the f=1 variants.
    TORCH_CHECK(bias_on == 0 || mode == 0 || bias.dtype != 3,
                "ffpa_attn: fp32 attn_mask requires a build with "
                "ENABLE_FFPA_CUDA_MASK_FP32=1");
#endif
    using Ic1 = std::integral_constant<int, 1>;
    using Ic0 = std::integral_constant<int, 0>;
    using Ic2 = std::integral_constant<int, 2>;
    using Ic3 = std::integral_constant<int, 3>;
    const auto launch_variant = [&](auto has_bias, auto plan_mode, auto bias_4b,
                                    auto has_dropout) {
      launch_cute_fwd_persist_d_sm120_v<
          kDataType, kHeadDim, kStage, decltype(has_bias)::value,
          decltype(plan_mode)::value, decltype(bias_4b)::value,
          decltype(has_dropout)::value>(Q, K, V, O, attn_bias, softmax_lse,
                                        causal, softmax_scale, dropout_p,
                                        philox_seed, philox_offset);
    };
    const auto launch_variant_with_dropout = [&](auto has_bias, auto plan_mode,
                                                 auto bias_4b) {
      if (dropout_p > 0.0)
        launch_variant(has_bias, plan_mode, bias_4b, Ic1{});
      else
        launch_variant(has_bias, plan_mode, bias_4b, Ic0{});
    };
    if (!bias_on)
      launch_variant_with_dropout(Ic0{}, Ic0{}, Ic0{});
    else if (mode == 1) {
#ifdef ENABLE_FFPA_CUDA_MASK_FP32
      if (b4)
        launch_variant_with_dropout(Ic1{}, Ic1{}, Ic1{});
      else
#endif
        launch_variant_with_dropout(Ic1{}, Ic1{}, Ic0{});
    } else if (mode == 2) {
#ifdef ENABLE_FFPA_CUDA_MASK_FP32
      if (b4)
        launch_variant_with_dropout(Ic1{}, Ic2{}, Ic1{});
      else
#endif
        launch_variant_with_dropout(Ic1{}, Ic2{}, Ic0{});
    } else {
      // persist has no mode 3 (the resident upgrade is a split/m4n2-only
      // plan step); mode 0 keeps the gmem-direct tag.
      launch_variant_with_dropout(Ic1{}, Ic0{}, Ic0{});
    }
  } else {
    TORCH_CHECK(false,
                "ffpa_attn: cute_tma fp16 persist_d requires D in {32..256} "
                "step 32, got D=",
                kHeadDim);
  }
}

template <typename kDataType, const int kHeadDim, const int kStage,
          const int kQKDChunk, const int kVDChunk>
void launch_cute_fwd_split_d_sm120(torch::Tensor Q, torch::Tensor K,
                                   torch::Tensor V, torch::Tensor O,
                                   torch::Tensor attn_bias,
                                   torch::Tensor softmax_lse, int causal,
                                   double softmax_scale, double dropout_p,
                                   int64_t philox_seed, int64_t philox_offset) {
  if constexpr (kHeadDim % 32 == 0 && kHeadDim >= 32) {
    const FfpaBiasParams bias = ffpa_bias_params_of(attn_bias, Q, K);
    const int bias_on = bias.ptr != nullptr ? 1 : 0;
    FfpaBiasTilePlan plan;
    if (bias_on)
      plan = ffpa::fp16_split_d_bias_plan<kDataType, kHeadDim, kStage,
                                          kQKDChunk, kVDChunk>(
          bias, Q.size(0), Q.size(1), Q.size(2), K.size(2));
    const int mode = bias_on ? plan.mode : 0;
    const int b4 = (mode != 0 && bias.dtype == 3) ? 1 : 0;
#ifndef ENABLE_FFPA_CUDA_MASK_FP32
    // mode 0 (gmem-direct) reads the mask dtype at runtime, so fp32 stays
    // legal there; only the TMA tile modes need the f=1 variants.
    TORCH_CHECK(bias_on == 0 || mode == 0 || bias.dtype != 3,
                "ffpa_attn: fp32 attn_mask requires a build with "
                "ENABLE_FFPA_CUDA_MASK_FP32=1");
#endif
    using Ic1 = std::integral_constant<int, 1>;
    using Ic0 = std::integral_constant<int, 0>;
    using Ic2 = std::integral_constant<int, 2>;
    using Ic3 = std::integral_constant<int, 3>;
    const auto launch_variant = [&](auto has_bias, auto plan_mode, auto bias_4b,
                                    auto has_dropout) {
      launch_cute_fwd_split_d_sm120_v<
          kDataType, kHeadDim, kStage, kQKDChunk, kVDChunk,
          decltype(has_bias)::value, decltype(plan_mode)::value,
          decltype(bias_4b)::value, decltype(has_dropout)::value>(
          Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale, dropout_p,
          philox_seed, philox_offset);
    };
    const auto launch_variant_with_dropout = [&](auto has_bias, auto plan_mode,
                                                 auto bias_4b) {
      if (dropout_p > 0.0)
        launch_variant(has_bias, plan_mode, bias_4b, Ic1{});
      else
        launch_variant(has_bias, plan_mode, bias_4b, Ic0{});
    };
    if (!bias_on)
      launch_variant_with_dropout(Ic0{}, Ic0{}, Ic0{});
    else if (mode == 1) {
#ifdef ENABLE_FFPA_CUDA_MASK_FP32
      if (b4)
        launch_variant_with_dropout(Ic1{}, Ic1{}, Ic1{});
      else
#endif
        launch_variant_with_dropout(Ic1{}, Ic1{}, Ic0{});
    } else if (mode == 2) {
#ifdef ENABLE_FFPA_CUDA_MASK_FP32
      if (b4)
        launch_variant_with_dropout(Ic1{}, Ic2{}, Ic1{});
      else
#endif
        launch_variant_with_dropout(Ic1{}, Ic2{}, Ic0{});
    } else if (mode == 3) {
#ifdef ENABLE_FFPA_CUDA_MASK_FP32
      if (b4)
        launch_variant_with_dropout(Ic1{}, Ic3{}, Ic1{});
      else
#endif
        launch_variant_with_dropout(Ic1{}, Ic3{}, Ic0{});
    } else {
      launch_variant_with_dropout(Ic1{}, Ic0{}, Ic0{});
    }
  } else {
    TORCH_CHECK(
        false,
        "ffpa_attn: cute_tma fp16 split_d requires D%32==0, got D=", kHeadDim);
  }
}

template <typename kDataType, const int kHeadDim, const int kStage>
void launch_cute_fwd_split_d_m4n2_sm120(torch::Tensor Q, torch::Tensor K,
                                        torch::Tensor V, torch::Tensor O,
                                        torch::Tensor attn_bias,
                                        torch::Tensor softmax_lse, int causal,
                                        double softmax_scale, double dropout_p,
                                        int64_t philox_seed,
                                        int64_t philox_offset) {
  if constexpr (kHeadDim % 64 == 0 && kHeadDim >= 768 && kHeadDim <= 1024) {
    const FfpaBiasParams bias = ffpa_bias_params_of(attn_bias, Q, K);
    const int bias_on = bias.ptr != nullptr ? 1 : 0;
    FfpaBiasTilePlan plan;
    if (bias_on)
      plan = ffpa::fp16_split_d_m4n2_bias_plan<kDataType, kHeadDim, kStage>(
          bias, Q.size(0), Q.size(1), Q.size(2), K.size(2));
    const int mode = bias_on ? plan.mode : 0;
    const int b4 = (mode != 0 && bias.dtype == 3) ? 1 : 0;
#ifndef ENABLE_FFPA_CUDA_MASK_FP32
    // mode 0 (gmem-direct) reads the mask dtype at runtime, so fp32 stays
    // legal there; only the TMA tile modes need the f=1 variants.
    TORCH_CHECK(bias_on == 0 || mode == 0 || bias.dtype != 3,
                "ffpa_attn: fp32 attn_mask requires a build with "
                "ENABLE_FFPA_CUDA_MASK_FP32=1");
#endif
    using Ic1 = std::integral_constant<int, 1>;
    using Ic0 = std::integral_constant<int, 0>;
    using Ic2 = std::integral_constant<int, 2>;
    using Ic3 = std::integral_constant<int, 3>;
    const auto launch_variant = [&](auto has_bias, auto plan_mode, auto bias_4b,
                                    auto has_dropout) {
      launch_cute_fwd_split_d_m4n2_sm120_v<
          kDataType, kHeadDim, kStage, decltype(has_bias)::value,
          decltype(plan_mode)::value, decltype(bias_4b)::value,
          decltype(has_dropout)::value>(Q, K, V, O, attn_bias, softmax_lse,
                                        causal, softmax_scale, dropout_p,
                                        philox_seed, philox_offset);
    };
    const auto launch_variant_with_dropout = [&](auto has_bias, auto plan_mode,
                                                 auto bias_4b) {
      if (dropout_p > 0.0)
        launch_variant(has_bias, plan_mode, bias_4b, Ic1{});
      else
        launch_variant(has_bias, plan_mode, bias_4b, Ic0{});
    };
    if (!bias_on)
      launch_variant_with_dropout(Ic0{}, Ic0{}, Ic0{});
    else if (mode == 1) {
#ifdef ENABLE_FFPA_CUDA_MASK_FP32
      if (b4)
        launch_variant_with_dropout(Ic1{}, Ic1{}, Ic1{});
      else
#endif
        launch_variant_with_dropout(Ic1{}, Ic1{}, Ic0{});
    } else if (mode == 2) {
#ifdef ENABLE_FFPA_CUDA_MASK_FP32
      if (b4)
        launch_variant_with_dropout(Ic1{}, Ic2{}, Ic1{});
      else
#endif
        launch_variant_with_dropout(Ic1{}, Ic2{}, Ic0{});
    } else if (mode == 3) {
#ifdef ENABLE_FFPA_CUDA_MASK_FP32
      if (b4)
        launch_variant_with_dropout(Ic1{}, Ic3{}, Ic1{});
      else
#endif
        launch_variant_with_dropout(Ic1{}, Ic3{}, Ic0{});
    } else {
      launch_variant_with_dropout(Ic1{}, Ic0{}, Ic0{});
    }
  } else {
    TORCH_CHECK(false,
                "ffpa_attn: cute_tma fp16 m4n2 requires D in {768..1024} "
                "step 64, got D=",
                kHeadDim);
  }
}

#include "generated/fwd_cute_fp16_variants.cuh"
#endif  // ENABLE_FFPA_TMA_EXT
#endif  // ENABLE_FFPA_CUTE_EXT

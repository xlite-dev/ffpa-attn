#pragma once
// CuTe fp8 family launchers (persist-D / split-D M8N1 / split-D M4N2 with
// their quantize/smooth/hadamard pre-kernel orchestration), moved
// verbatim out of the old cute/launch.cuh.
#include "launch/common.cuh"
#if defined(ENABLE_FFPA_CUTE_EXT) && defined(ENABLE_FFPA_TMA_EXT)
#include "launch/cute_fp8_persist_d.cuh"
#include "launch/cute_fp8_split_d.cuh"
#include "launch/cute_fp8_split_d_m4n2.cuh"
template <typename kDataType, const int kHeadDim, const int kStage>
void launch_cute_fwd_persist_d_fp8_sm120(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
    torch::Tensor attn_bias, torch::Tensor softmax_lse, int causal,
    double softmax_scale, double dropout_p, int64_t philox_seed,
    int64_t philox_offset, bool fp8_smooth_k, bool fp8_smooth_v,
    int64_t fp8_q_quant_method, int64_t fp8_k_quant_method,
    int64_t fp8_v_quant_method, int64_t fp8_pv_acc_type, int64_t fp8_qk_mm_type,
    int q_start_row = 0, bool fp8_hadamard = false) {
  // qk_mm_type: 0=fp8 (e4m3 QK MMA), 1=int8 (s8xs8->s32). Default fp8;
  // int8 fixes the causal early-row dS accuracy limit at ~zero cost.
  // if constexpr keeps the impl (and its kernel) out of instantiation for
  // unsupported headdims; every headdim TU includes this launcher template.
  if constexpr (kHeadDim % 32 == 0 && kHeadDim >= 32 && kHeadDim <= 224) {
    const bool qk_int8 = (fp8_qk_mm_type == 1);
    const FfpaBiasParams bias = ffpa_bias_params_of(attn_bias, Q, K);
    int max_smem_optin = 0;
    cudaDeviceGetAttribute(&max_smem_optin,
                           cudaDevAttrMaxSharedMemoryPerBlockOptin,
                           Q.get_device());
    const int dyn_limit = max_smem_optin - 256;
    // Runtime dispatch over the variant tags; the plan helper is the same
    // single source the variant body checks against, so a demote-rule drift
    // is a loud TORCH_CHECK, never a silent wrong-kernel launch.
    const auto dispatch_q = [&](auto qk_c) {
      constexpr bool kQ = decltype(qk_c)::value;
      FfpaBiasTilePlan plan;
      if (bias.ptr != nullptr)
        plan = ffpa::fp8_persist_d_bias_plan<kDataType, kHeadDim, kStage, kQ>(
            bias, Q.size(0), Q.size(1), Q.size(2), K.size(2), dyn_limit);
      const int bias_on = bias.ptr != nullptr ? 1 : 0;
      const int mode = bias_on ? plan.mode : 0;
      const int b4 = (mode == 2 && bias.dtype == 3) ? 1 : 0;
      if (!bias_on)
        launch_cute_fwd_persist_d_fp8_sm120_v<kDataType, kHeadDim, kStage, kQ,
                                              0, 0, 0>(
            Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
            dropout_p, philox_seed, philox_offset, fp8_smooth_k, fp8_smooth_v,
            fp8_q_quant_method, fp8_k_quant_method, fp8_v_quant_method,
            fp8_pv_acc_type, q_start_row, fp8_hadamard);
      else if (mode == 2 && b4)
        launch_cute_fwd_persist_d_fp8_sm120_v<kDataType, kHeadDim, kStage, kQ,
                                              1, 2, 1>(
            Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
            dropout_p, philox_seed, philox_offset, fp8_smooth_k, fp8_smooth_v,
            fp8_q_quant_method, fp8_k_quant_method, fp8_v_quant_method,
            fp8_pv_acc_type, q_start_row, fp8_hadamard);
      else if (mode == 2)
        launch_cute_fwd_persist_d_fp8_sm120_v<kDataType, kHeadDim, kStage, kQ,
                                              1, 2, 0>(
            Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
            dropout_p, philox_seed, philox_offset, fp8_smooth_k, fp8_smooth_v,
            fp8_q_quant_method, fp8_k_quant_method, fp8_v_quant_method,
            fp8_pv_acc_type, q_start_row, fp8_hadamard);
      else if (mode == 3)
        launch_cute_fwd_persist_d_fp8_sm120_v<kDataType, kHeadDim, kStage, kQ,
                                              1, 3, 0>(
            Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
            dropout_p, philox_seed, philox_offset, fp8_smooth_k, fp8_smooth_v,
            fp8_q_quant_method, fp8_k_quant_method, fp8_v_quant_method,
            fp8_pv_acc_type, q_start_row, fp8_hadamard);
      else
        launch_cute_fwd_persist_d_fp8_sm120_v<kDataType, kHeadDim, kStage, kQ,
                                              1, 0, 0>(
            Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
            dropout_p, philox_seed, philox_offset, fp8_smooth_k, fp8_smooth_v,
            fp8_q_quant_method, fp8_k_quant_method, fp8_v_quant_method,
            fp8_pv_acc_type, q_start_row, fp8_hadamard);
    };
    if (qk_int8)
      dispatch_q(std::integral_constant<bool, true>{});
    else
      dispatch_q(std::integral_constant<bool, false>{});
  } else {
    TORCH_CHECK(false,
                "ffpa_attn: cute_tma_fp8 persist_d requires D in {32..224} "
                "step 32, got D=",
                kHeadDim);
  }
}

template <typename kDataType, const int kHeadDim, const int kStage>
void launch_cute_fwd_split_d_fp8_sm120(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
    torch::Tensor attn_bias, torch::Tensor softmax_lse, int causal,
    double softmax_scale, double dropout_p, int64_t philox_seed,
    int64_t philox_offset, bool fp8_smooth_k, bool fp8_smooth_v,
    int64_t fp8_q_quant_method, int64_t fp8_k_quant_method,
    int64_t fp8_v_quant_method, int64_t fp8_pv_acc_type, int64_t fp8_qk_mm_type,
    int q_start_row = 0, bool fp8_hadamard = false) {
  // EXPERIMENT: lower bound lowered from >=768 to >=192 so M4N2 can be A/B'd
  // against M8N1 across all large headdims via FFPA_FP8_FORCE_KERNEL.
  // Production dispatch selects M4N2 only for D>=768 via the top-level
  // launcher.
  if constexpr (kHeadDim >= 192 && kHeadDim <= 1024 && kHeadDim % 64 == 0) {
    const bool qk_int8 = (fp8_qk_mm_type == 1);
    const FfpaBiasParams bias = ffpa_bias_params_of(attn_bias, Q, K);
    int max_smem_optin = 0;
    cudaDeviceGetAttribute(&max_smem_optin,
                           cudaDevAttrMaxSharedMemoryPerBlockOptin,
                           Q.get_device());
    const int dyn_limit = max_smem_optin - 256;
    // Runtime dispatch over the variant tags (single-source plan, see the
    // persist_d wrapper).
    const auto dispatch_q = [&](auto qk_c) {
      constexpr bool kQ = decltype(qk_c)::value;
      FfpaBiasTilePlan plan;
      if (bias.ptr != nullptr)
        plan = ffpa::fp8_split_d_bias_plan<kDataType, kHeadDim, kStage, kQ>(
            bias, Q.size(0), Q.size(1), Q.size(2), K.size(2), dyn_limit);
      const int bias_on = bias.ptr != nullptr ? 1 : 0;
      const int mode = bias_on ? plan.mode : 0;
      const int b4 = (mode == 2 && bias.dtype == 3) ? 1 : 0;
      if (!bias_on)
        launch_cute_fwd_split_d_fp8_sm120_v<kDataType, kHeadDim, kStage, kQ, 0,
                                            0, 0>(
            Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
            dropout_p, philox_seed, philox_offset, fp8_smooth_k, fp8_smooth_v,
            fp8_q_quant_method, fp8_k_quant_method, fp8_v_quant_method,
            fp8_pv_acc_type, q_start_row, fp8_hadamard);
      else if constexpr (kHeadDim < 512) {
        // mode 2 is demoted away by the plan for D>=512 (see
        // fp8_split_d_bias_plan), so those tags stay out of the extern
        // table and must not be instantiated here either.
        if (mode == 2 && b4)
          launch_cute_fwd_split_d_fp8_sm120_v<kDataType, kHeadDim, kStage, kQ,
                                              1, 2, 1>(
              Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
              dropout_p, philox_seed, philox_offset, fp8_smooth_k, fp8_smooth_v,
              fp8_q_quant_method, fp8_k_quant_method, fp8_v_quant_method,
              fp8_pv_acc_type, q_start_row, fp8_hadamard);
        else if (mode == 2)
          launch_cute_fwd_split_d_fp8_sm120_v<kDataType, kHeadDim, kStage, kQ,
                                              1, 2, 0>(
              Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
              dropout_p, philox_seed, philox_offset, fp8_smooth_k, fp8_smooth_v,
              fp8_q_quant_method, fp8_k_quant_method, fp8_v_quant_method,
              fp8_pv_acc_type, q_start_row, fp8_hadamard);
        else
          launch_cute_fwd_split_d_fp8_sm120_v<kDataType, kHeadDim, kStage, kQ,
                                              1, 0, 0>(
              Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
              dropout_p, philox_seed, philox_offset, fp8_smooth_k, fp8_smooth_v,
              fp8_q_quant_method, fp8_k_quant_method, fp8_v_quant_method,
              fp8_pv_acc_type, q_start_row, fp8_hadamard);
      } else
        launch_cute_fwd_split_d_fp8_sm120_v<kDataType, kHeadDim, kStage, kQ, 1,
                                            0, 0>(
            Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
            dropout_p, philox_seed, philox_offset, fp8_smooth_k, fp8_smooth_v,
            fp8_q_quant_method, fp8_k_quant_method, fp8_v_quant_method,
            fp8_pv_acc_type, q_start_row, fp8_hadamard);
    };
    if (qk_int8)
      dispatch_q(std::integral_constant<bool, true>{});
    else
      dispatch_q(std::integral_constant<bool, false>{});
  } else {
    TORCH_CHECK(false,
                "ffpa_attn: cute_tma_fp8 split_d requires D in "
                "[192, 1024] with D % 64 == 0, got D=",
                kHeadDim);
  }
}

template <typename kDataType, const int kHeadDim, const int kStage>
void launch_cute_fwd_split_d_m4n2_fp8_sm120(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
    torch::Tensor attn_bias, torch::Tensor softmax_lse, int causal,
    double softmax_scale, double dropout_p, int64_t philox_seed,
    int64_t philox_offset, bool fp8_smooth_k, bool fp8_smooth_v,
    int64_t fp8_q_quant_method, int64_t fp8_k_quant_method,
    int64_t fp8_v_quant_method, int64_t fp8_pv_acc_type, int64_t fp8_qk_mm_type,
    int q_start_row = 0, bool fp8_hadamard = false) {
  // EXPERIMENT: lower bound lowered from >=768 to >=192 so M4N2 can be A/B'd
  // against M8N1 across all large headdims via FFPA_FP8_FORCE_KERNEL.
  // Production dispatch selects M4N2 only for D>=768 via the top-level
  // launcher.
  if constexpr (kHeadDim >= 192 && kHeadDim <= 1024 && kHeadDim % 64 == 0) {
    const bool qk_int8 = (fp8_qk_mm_type == 1);
    const FfpaBiasParams bias = ffpa_bias_params_of(attn_bias, Q, K);
    // Runtime dispatch over the variant tags (single-source plan, see the
    // persist_d wrapper). m4n2 keeps mode 1 (dense tile).
    const auto dispatch_q = [&](auto qk_c) {
      constexpr bool kQ = decltype(qk_c)::value;
      FfpaBiasTilePlan plan;
      if (bias.ptr != nullptr)
        plan = ffpa::fp8_m4n2_bias_plan<kDataType, kHeadDim, kStage, kQ>(
            bias, Q.size(0), Q.size(1), Q.size(2), K.size(2));
      const int bias_on = bias.ptr != nullptr ? 1 : 0;
      const int mode = bias_on ? plan.mode : 0;
      const int b4 = (mode != 0 && bias.dtype == 3) ? 1 : 0;
      if (!bias_on)
        launch_cute_fwd_split_d_m4n2_fp8_sm120_v<kDataType, kHeadDim, kStage,
                                                 kQ, 0, 0, 0>(
            Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
            dropout_p, philox_seed, philox_offset, fp8_smooth_k, fp8_smooth_v,
            fp8_q_quant_method, fp8_k_quant_method, fp8_v_quant_method,
            fp8_pv_acc_type, q_start_row, fp8_hadamard);
      else if (mode == 1 && b4)
        launch_cute_fwd_split_d_m4n2_fp8_sm120_v<kDataType, kHeadDim, kStage,
                                                 kQ, 1, 1, 1>(
            Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
            dropout_p, philox_seed, philox_offset, fp8_smooth_k, fp8_smooth_v,
            fp8_q_quant_method, fp8_k_quant_method, fp8_v_quant_method,
            fp8_pv_acc_type, q_start_row, fp8_hadamard);
      else if (mode == 1)
        launch_cute_fwd_split_d_m4n2_fp8_sm120_v<kDataType, kHeadDim, kStage,
                                                 kQ, 1, 1, 0>(
            Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
            dropout_p, philox_seed, philox_offset, fp8_smooth_k, fp8_smooth_v,
            fp8_q_quant_method, fp8_k_quant_method, fp8_v_quant_method,
            fp8_pv_acc_type, q_start_row, fp8_hadamard);
      else if (mode == 2 && b4)
        launch_cute_fwd_split_d_m4n2_fp8_sm120_v<kDataType, kHeadDim, kStage,
                                                 kQ, 1, 2, 1>(
            Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
            dropout_p, philox_seed, philox_offset, fp8_smooth_k, fp8_smooth_v,
            fp8_q_quant_method, fp8_k_quant_method, fp8_v_quant_method,
            fp8_pv_acc_type, q_start_row, fp8_hadamard);
      else if (mode == 2)
        launch_cute_fwd_split_d_m4n2_fp8_sm120_v<kDataType, kHeadDim, kStage,
                                                 kQ, 1, 2, 0>(
            Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
            dropout_p, philox_seed, philox_offset, fp8_smooth_k, fp8_smooth_v,
            fp8_q_quant_method, fp8_k_quant_method, fp8_v_quant_method,
            fp8_pv_acc_type, q_start_row, fp8_hadamard);
      else if (mode == 3 && b4)
        launch_cute_fwd_split_d_m4n2_fp8_sm120_v<kDataType, kHeadDim, kStage,
                                                 kQ, 1, 3, 1>(
            Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
            dropout_p, philox_seed, philox_offset, fp8_smooth_k, fp8_smooth_v,
            fp8_q_quant_method, fp8_k_quant_method, fp8_v_quant_method,
            fp8_pv_acc_type, q_start_row, fp8_hadamard);
      else if (mode == 3)
        launch_cute_fwd_split_d_m4n2_fp8_sm120_v<kDataType, kHeadDim, kStage,
                                                 kQ, 1, 3, 0>(
            Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
            dropout_p, philox_seed, philox_offset, fp8_smooth_k, fp8_smooth_v,
            fp8_q_quant_method, fp8_k_quant_method, fp8_v_quant_method,
            fp8_pv_acc_type, q_start_row, fp8_hadamard);
      else
        launch_cute_fwd_split_d_m4n2_fp8_sm120_v<kDataType, kHeadDim, kStage,
                                                 kQ, 1, 0, 0>(
            Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
            dropout_p, philox_seed, philox_offset, fp8_smooth_k, fp8_smooth_v,
            fp8_q_quant_method, fp8_k_quant_method, fp8_v_quant_method,
            fp8_pv_acc_type, q_start_row, fp8_hadamard);
    };
    if (qk_int8)
      dispatch_q(std::integral_constant<bool, true>{});
    else
      dispatch_q(std::integral_constant<bool, false>{});
  } else {
    TORCH_CHECK(false,
                "ffpa_attn: cute_tma_fp8 split_d_m4n2 requires D in "
                "[192, 1024] with D % 64 == 0, got D=",
                kHeadDim);
  }
}

// Variant entry extern declarations (generated): included here, after the
// variant template definitions, so every family TU suppresses
// re-instantiation of the whole (kQKInt8, bias-mode) kernel table. The
// variant TUs include the per-impl headers (launch/cute_fp8_{persist_d,
// split_d,split_d_m4n2}.cuh) directly and get exactly one kernel table.
#include "generated/fwd_cute_fp8_variants.cuh"  // extern templates

#endif  // ENABLE_FFPA_CUTE_EXT && ENABLE_FFPA_TMA_EXT

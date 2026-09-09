#pragma once
// CuTe fp4 family launchers (persist-D / split-D / split-D M4N2 with
// their quantize/delta_s/hadamard/kv-mean pre-kernel orchestration),
// moved verbatim out of the old cute/launch.cuh.
#include "launch/common.cuh"
#if defined(ENABLE_FFPA_CUTE_EXT) && defined(ENABLE_FFPA_TMA_EXT)
#include "launch/cute_fp4_persist_d.cuh"
#include "launch/cute_fp4_split_d.cuh"
#include "launch/cute_fp4_split_d_m4n2.cuh"
template <typename kDataType, const int kHeadDim, const int kStage>
void launch_cute_fwd_persist_d_fp4_sm120(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
    torch::Tensor attn_bias, torch::Tensor softmax_lse, int causal,
    double softmax_scale, int q_start_row = 0, bool fp4_hadamard = false,
    int fp4_pv_mm_type = 0, bool fp4_smooth_v = false) {
  (void)kStage;  // kStages (3, or 2 at D=256) fixed by the fp4 traits
  auto prop = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(prop->major == 12,
              "ffpa_attn: the NVFP4 path requires an sm_120 device, got sm_",
              prop->major, prop->minor);
  const bool pv_fp8 = fp4_pv_mm_type == 1;
  TORCH_CHECK(fp4_pv_mm_type == 0 || fp4_pv_mm_type == 1,
              "ffpa_attn: fp4_pv_mm_type must be 0 (fp4) or 1 (fp8)");
  if constexpr (kHeadDim % 64 == 0 && kHeadDim >= 64 && kHeadDim <= 256) {
    if (pv_fp8 && kHeadDim > 192)
      TORCH_CHECK(false,
                  "ffpa_attn: fp4_pv_mm_type=fp8 persist_d supports D in "
                  "{64,128,192} (smem budget), got D=",
                  kHeadDim);
    const FfpaBiasParams bias = ffpa_bias_params_of(attn_bias, Q, K);
    // Runtime dispatch over the variant tags (single-source plan, see
    // fp4_persist_d_bias_plan); the variant body re-checks the tags.
    const auto dispatch_p = [&](auto pv_c) {
      constexpr bool kPv = decltype(pv_c)::value;
      if constexpr (!kPv || kHeadDim <= 192) {
        int max_smem_optin = 0;
        cudaDeviceGetAttribute(&max_smem_optin,
                               cudaDevAttrMaxSharedMemoryPerBlockOptin,
                               Q.get_device());
        const int dyn_limit = max_smem_optin - 256;
        FfpaBiasTilePlan plan;
        if (bias.ptr != nullptr)
          plan = ffpa::fp4_persist_d_bias_plan<kDataType, kHeadDim, kPv>(
              bias, Q.size(0), Q.size(1), Q.size(2), K.size(2), dyn_limit);
        const int bias_on = bias.ptr != nullptr ? 1 : 0;
        const int mode = bias_on ? plan.mode : 0;
        const int b4 = (mode != 0 && bias.dtype == 3) ? 1 : 0;
#ifndef ENABLE_FFPA_CUDA_MASK_FP32
        // mode 0 (gmem-direct) reads the mask dtype at runtime; only the
        // TMA tile modes need the f=1 variants.
        TORCH_CHECK(bias_on == 0 || mode == 0 || bias.dtype != 3,
                    "ffpa_attn: fp32 attn_mask requires a build with "
                    "ENABLE_FFPA_CUDA_MASK_FP32=1");
#endif
        if (!bias_on)
          launch_cute_fwd_persist_d_fp4_sm120_v<kDataType, kHeadDim, kPv, 0, 0,
                                                0>(
              Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
              q_start_row, fp4_hadamard, fp4_smooth_v);
#ifdef ENABLE_FFPA_CUDA_MASK_FP32
        else if (mode == 1 && b4)
          launch_cute_fwd_persist_d_fp4_sm120_v<kDataType, kHeadDim, kPv, 1, 1,
                                                1>(
              Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
              q_start_row, fp4_hadamard, fp4_smooth_v);
#endif
        else if (mode == 1)
          launch_cute_fwd_persist_d_fp4_sm120_v<kDataType, kHeadDim, kPv, 1, 1,
                                                0>(
              Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
              q_start_row, fp4_hadamard, fp4_smooth_v);
#ifdef ENABLE_FFPA_CUDA_MASK_FP32
        else if (mode == 2 && b4)
          launch_cute_fwd_persist_d_fp4_sm120_v<kDataType, kHeadDim, kPv, 1, 2,
                                                1>(
              Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
              q_start_row, fp4_hadamard, fp4_smooth_v);
#endif
        else if (mode == 2)
          launch_cute_fwd_persist_d_fp4_sm120_v<kDataType, kHeadDim, kPv, 1, 2,
                                                0>(
              Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
              q_start_row, fp4_hadamard, fp4_smooth_v);
#ifdef ENABLE_FFPA_CUDA_MASK_FP32
        else if (mode == 3 && b4)
          launch_cute_fwd_persist_d_fp4_sm120_v<kDataType, kHeadDim, kPv, 1, 3,
                                                1>(
              Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
              q_start_row, fp4_hadamard, fp4_smooth_v);
#endif
        else if (mode == 3)
          launch_cute_fwd_persist_d_fp4_sm120_v<kDataType, kHeadDim, kPv, 1, 3,
                                                0>(
              Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
              q_start_row, fp4_hadamard, fp4_smooth_v);
        else
          launch_cute_fwd_persist_d_fp4_sm120_v<kDataType, kHeadDim, kPv, 1, 0,
                                                0>(
              Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
              q_start_row, fp4_hadamard, fp4_smooth_v);
      }
    };
    if (pv_fp8)
      dispatch_p(std::integral_constant<bool, true>{});
    else
      dispatch_p(std::integral_constant<bool, false>{});
  } else {
    TORCH_CHECK(false,
                "ffpa_attn: cute_tma_fp4 persist_d requires D in "
                "{64,128,192,256} (64-multiples), got D=",
                kHeadDim);
  }
}

template <typename kDataType, const int kHeadDim, const int kStage>
void launch_cute_fwd_split_d_fp4_sm120(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
    torch::Tensor attn_bias, torch::Tensor softmax_lse, int causal,
    double softmax_scale, int q_start_row = 0, bool fp4_hadamard = false,
    int fp4_pv_mm_type = 0, bool fp4_smooth_v = false) {
  (void)kStage;  // kStages (3/3) fixed by the fp4 split_d traits
  TORCH_CHECK(fp4_pv_mm_type == 0 || fp4_pv_mm_type == 1,
              "ffpa_attn: fp4_pv_mm_type must be 0 (fp4) or 1 (fp8)");
  auto prop = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(prop->major == 12,
              "ffpa_attn: the NVFP4 path requires an sm_120 device, got sm_",
              prop->major, prop->minor);
  if constexpr (kHeadDim % 64 == 0 && kHeadDim > 256 && kHeadDim < 768) {
    const FfpaBiasParams bias = ffpa_bias_params_of(attn_bias, Q, K);
    // Runtime dispatch over the variant tags (single-source plan, see
    // fp4_split_d_bias_plan); the variant body re-checks the tags.
    const auto dispatch_p = [&](auto pv_c) {
      constexpr bool kPv = decltype(pv_c)::value;
      int max_smem_optin = 0;
      cudaDeviceGetAttribute(&max_smem_optin,
                             cudaDevAttrMaxSharedMemoryPerBlockOptin,
                             Q.get_device());
      const int dyn_limit = max_smem_optin - 256;
      FfpaBiasTilePlan plan;
      if (bias.ptr != nullptr)
        plan = ffpa::fp4_split_d_bias_plan<kDataType, kHeadDim, kPv>(
            bias, Q.size(0), Q.size(1), Q.size(2), K.size(2), dyn_limit);
      const int bias_on = bias.ptr != nullptr ? 1 : 0;
      const int mode = bias_on ? plan.mode : 0;
      const int b4 = (mode == 2 && bias.dtype == 3) ? 1 : 0;
#ifndef ENABLE_FFPA_CUDA_MASK_FP32
      // Only the mode-2 TMA tile needs the f=1 variant (b4); mode 0/3
      // read the mask dtype at runtime.
      TORCH_CHECK(b4 == 0,
                  "ffpa_attn: fp32 attn_mask requires a build with "
                  "ENABLE_FFPA_CUDA_MASK_FP32=1");
#endif
      if (!bias_on)
        launch_cute_fwd_split_d_fp4_sm120_v<kDataType, kHeadDim, kPv, 0, 0, 0>(
            Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
            q_start_row, fp4_hadamard, fp4_smooth_v);
#ifdef ENABLE_FFPA_CUDA_MASK_FP32
      else if (mode == 2 && b4)
        launch_cute_fwd_split_d_fp4_sm120_v<kDataType, kHeadDim, kPv, 1, 2, 1>(
            Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
            q_start_row, fp4_hadamard, fp4_smooth_v);
#endif
      else if (mode == 2)
        launch_cute_fwd_split_d_fp4_sm120_v<kDataType, kHeadDim, kPv, 1, 2, 0>(
            Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
            q_start_row, fp4_hadamard, fp4_smooth_v);
      else if (mode == 3)
        launch_cute_fwd_split_d_fp4_sm120_v<kDataType, kHeadDim, kPv, 1, 3, 0>(
            Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
            q_start_row, fp4_hadamard, fp4_smooth_v);
      else
        launch_cute_fwd_split_d_fp4_sm120_v<kDataType, kHeadDim, kPv, 1, 0, 0>(
            Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
            q_start_row, fp4_hadamard, fp4_smooth_v);
    };
    if (fp4_pv_mm_type == 1)
      dispatch_p(std::integral_constant<bool, true>{});
    else
      dispatch_p(std::integral_constant<bool, false>{});
  } else {
    TORCH_CHECK(false,
                "ffpa_attn: cute_tma_fp4 split_d requires 64-multiple D in "
                "(256,768), got D=",
                kHeadDim);
  }
}

template <typename kDataType, const int kHeadDim, const int kStage>
void launch_cute_fwd_split_d_m4n2_fp4_sm120(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
    torch::Tensor attn_bias, torch::Tensor softmax_lse, int causal,
    double softmax_scale, int q_start_row = 0, bool fp4_hadamard = false,
    int fp4_pv_mm_type = 0, bool fp4_smooth_v = false) {
  (void)kStage;  // kStages (2/2) fixed by the fp4 m4n2 traits
  // NVFP4-only PV: the MXFP8 PV atom (SM120_16x8x128) consumes Tile-K=128
  // tokens per mma, but the m4n2 tiles are kBc=64 - the operand pair
  // cannot be formed. Architectural, not a smem budget.
  TORCH_CHECK(fp4_pv_mm_type == 0,
              "ffpa_attn: fp4_pv_mm_type=fp8 supports persist_d (D<=192) "
              "and split_d (256<D<768) only, got split_d m4n2 D=",
              kHeadDim);
  auto prop = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(prop->major == 12,
              "ffpa_attn: the NVFP4 path requires an sm_120 device, got sm_",
              prop->major, prop->minor);
  if constexpr (kHeadDim % 64 == 0 && kHeadDim >= 768 && kHeadDim <= 1024) {
    // Runtime dispatch over the variant tags (single-source plan, see
    // fp4_m4n2_bias_plan); regular builds always land on mode 0 (PC-0-5)
    // or no-bias, mode 2 exists only in debug builds.
    const FfpaBiasParams bias = ffpa_bias_params_of(attn_bias, Q, K);
    const int bias_on = bias.ptr != nullptr ? 1 : 0;
    if (!bias_on) {
      launch_cute_fwd_split_d_m4n2_fp4_sm120_v<kDataType, kHeadDim, false, 0, 0,
                                               0>(
          Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
          q_start_row, fp4_hadamard, fp4_smooth_v);
      return;
    }
    int max_smem_optin = 0;
    cudaDeviceGetAttribute(&max_smem_optin,
                           cudaDevAttrMaxSharedMemoryPerBlockOptin,
                           Q.get_device());
    FfpaBiasTilePlan plan = ffpa::fp4_m4n2_bias_plan<kDataType, kHeadDim>(
        bias, Q.size(0), Q.size(1), Q.size(2), K.size(2), max_smem_optin - 256);
    if (plan.mode == 2) {
#ifdef ENABLE_FFPA_FP4_BUILD_DEBUG
      // mode 2 is a debug-only tag (PC-0-5); the f=1 variant additionally
      // needs ENABLE_FFPA_CUDA_MASK_FP32 (env.py drops it otherwise).
#ifndef ENABLE_FFPA_CUDA_MASK_FP32
      TORCH_CHECK(bias.dtype != 3,
                  "ffpa_attn: fp32 attn_mask requires a build with "
                  "ENABLE_FFPA_CUDA_MASK_FP32=1");
#endif
#ifdef ENABLE_FFPA_CUDA_MASK_FP32
      if (bias.dtype == 3)
        launch_cute_fwd_split_d_m4n2_fp4_sm120_v<kDataType, kHeadDim, false, 1,
                                                 2, 1>(
            Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
            q_start_row, fp4_hadamard, fp4_smooth_v);
      else
#endif
        launch_cute_fwd_split_d_m4n2_fp4_sm120_v<kDataType, kHeadDim, false, 1,
                                                 2, 0>(
            Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
            q_start_row, fp4_hadamard, fp4_smooth_v);
#else
      TORCH_CHECK(false,
                  "ffpa_attn: fp4 m4n2 mode-2 requires a debug build "
                  "(ENABLE_FFPA_FP4_BUILD_DEBUG)");
#endif
    } else {
      launch_cute_fwd_split_d_m4n2_fp4_sm120_v<kDataType, kHeadDim, false, 1, 0,
                                               0>(
          Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
          q_start_row, fp4_hadamard, fp4_smooth_v);
    }
  } else {
    TORCH_CHECK(false,
                "ffpa_attn: cute_tma_fp4 split_d m4n2 requires 64-multiple D "
                "in [768,1024], got D=",
                kHeadDim);
  }
}

// Variant TUs include the per-impl headers (launch/cute_fp4_{persist_d,
// split_d,split_d_m4n2}.cuh) directly and instantiate exactly one kernel
// table; every other TU gets extern-template declarations only via
// generated/fwd_cute_fp4_variants.cuh.
#include "generated/fwd_cute_fp4_variants.cuh"  // extern templates

#endif  // ENABLE_FFPA_CUTE_EXT && ENABLE_FFPA_TMA_EXT

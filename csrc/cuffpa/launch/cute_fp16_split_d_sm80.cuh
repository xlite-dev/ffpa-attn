#pragma once
// CuTe fp16/bf16 sm80 split_d launcher (cp.async, no TMA), split out of
// launch/cute_fp16.cuh so touching the sm120 kernel headers no longer
// repreprocesses it. Not variant-split: the kernel table is small and the
// family TUs keep instantiating it implicitly (see dispatch/cute_fp16.cuh).
#include "launch/common.cuh"
#ifdef ENABLE_FFPA_CUTE_EXT
#include "cute/sm_80/split_d.cuh"

template <typename kDataType, const int kHeadDim, const int kStage,
          const int kQKDChunk, const int kVDChunk>
void launch_cute_fwd_split_d_sm80(torch::Tensor Q, torch::Tensor K,
                                  torch::Tensor V, torch::Tensor O,
                                  torch::Tensor attn_bias,
                                  torch::Tensor softmax_lse, int causal,
                                  double softmax_scale, double dropout_p,
                                  int64_t philox_seed, int64_t philox_offset) {
  using namespace cute;

  constexpr int kBr = 128;
  constexpr int kBc = 128;

  constexpr int kNumThreads = kBr / 16 * 32;

  using Element = std::conditional_t<std::is_same_v<kDataType, __half>,
                                     cutlass::half_t, cutlass::bfloat16_t>;
  constexpr int kStagesQK = kStage;
  constexpr int kStagesPV = kStagesQK;
  using Traits = ffpa_cute::FFPAAttnCuTeSplitDTraits<
      kHeadDim, kBr, kBc, kQKDChunk, kVDChunk, kStagesQK, kStagesPV, Element>;

  constexpr int kQTileBytes = kBr * kQKDChunk * sizeof(Element);
  constexpr int kKTileBytes = kBc * kQKDChunk * sizeof(Element);
  constexpr int kVTileBytes = kBc * kVDChunk * sizeof(Element);
  constexpr int kSmemPerStage = kQTileBytes + kKTileBytes + kVTileBytes;

  int max_smem_optin = 0;
  cudaDeviceGetAttribute(&max_smem_optin,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin,
                         Q.device().index());
  TORCH_CHECK(kStagesQK * kSmemPerStage <= max_smem_optin,
              "ffpa_attn: CuTe kernel requires ", kStagesQK * kSmemPerStage,
              " bytes smem (stages=", kStagesQK, ", chunk=", kQKDChunk, "/",
              kVDChunk, ") but device supports ", max_smem_optin,
              " bytes opt-in smem");

  const int Nb = Q.size(0);
  const int Nh = Q.size(1);
  const int Nh_kv = K.size(1);
  const int Nq = Q.size(2);
  const int Nkv = K.size(2);
  const int Tc = utils::div_ceil(Nkv, kBc);
  const float scale = static_cast<float>(softmax_scale);

  const bool has_attn_bias = attn_bias.numel() != 0;
  const bool has_dropout = dropout_p > 0.0;

  const void* attn_bias_ptr = nullptr;
  int attn_bias_dtype = 0;
  long long attn_bias_stride_b = 0;
  long long attn_bias_stride_h = 0;
  long long attn_bias_stride_m = 0;
  long long attn_bias_stride_n = 0;
  if (has_attn_bias) {
    TORCH_CHECK(attn_bias.is_cuda(),
                "ffpa_attn: attn_mask must be a CUDA tensor");
    TORCH_CHECK(attn_bias.device() == Q.device(),
                "ffpa_attn: attn_mask must be on the same device as Q/K/V");
    TORCH_CHECK(
        attn_bias.dim() == 4,
        "ffpa_attn: normalized attn_mask must be 4-D [B, Nh_q, Nq, Nkv]");
    TORCH_CHECK(attn_bias.size(0) == 1 || attn_bias.size(0) == Nb,
                "ffpa_attn: attn_mask batch dimension must be 1 or B");
    TORCH_CHECK(attn_bias.size(1) == 1 || attn_bias.size(1) == Nh,
                "ffpa_attn: attn_mask head dimension must be 1 or Nh_q");
    TORCH_CHECK(attn_bias.size(2) == 1 || attn_bias.size(2) == Nq,
                "ffpa_attn: attn_mask query dimension must be 1 or Nq");
    TORCH_CHECK(attn_bias.size(3) == 1 || attn_bias.size(3) == Nkv,
                "ffpa_attn: attn_mask kv dimension must be 1 or Nkv");
    attn_bias_ptr = attn_bias.data_ptr();
    if (attn_bias.scalar_type() == at::ScalarType::Half)
      attn_bias_dtype = 1;
    else if (attn_bias.scalar_type() == at::ScalarType::BFloat16)
      attn_bias_dtype = 2;
    else
      attn_bias_dtype = 3;
    attn_bias_stride_b =
        (attn_bias.size(0) == 1 && Nb > 1) ? 0 : attn_bias.stride(0);
    attn_bias_stride_h =
        (attn_bias.size(1) == 1 && Nh > 1) ? 0 : attn_bias.stride(1);
    attn_bias_stride_m =
        (attn_bias.size(2) == 1 && Nq > 1) ? 0 : attn_bias.stride(2);
    attn_bias_stride_n =
        (attn_bias.size(3) == 1 && Nkv > 1) ? 0 : attn_bias.stride(3);
  }
  const float dropout_p_f = static_cast<float>(dropout_p);
  const unsigned long long philox_seed_u =
      static_cast<unsigned long long>(philox_seed);
  const unsigned long long philox_offset_u =
      static_cast<unsigned long long>(philox_offset);

  // PC-0 bias tile (cp.async sm80 path): shape classification only — no TMA
  // descriptor here; the loader is in-kernel vectorized global loads, so the
  // plan only gates on the smem budget.
  FfpaBiasTilePlan bias_plan;
  if (has_attn_bias) {
    FfpaBiasParams bias_p{attn_bias_ptr,      attn_bias_dtype,
                          attn_bias_stride_b, attn_bias_stride_h,
                          attn_bias_stride_m, attn_bias_stride_n};
    bias_plan = ffpa_bias_tile_plan_of(bias_p, Nb, Nh, Nq, Nkv);
  }
  const int kBaseSmemBytes = kStagesQK * kSmemPerStage;
  if (kBaseSmemBytes + bias_plan.tile_bytes(kBr, kBc, 2) > max_smem_optin)
    bias_plan.mode = 0;
  int kSmemBytes = kBaseSmemBytes + (int)bias_plan.tile_bytes(kBr, kBc, 2);
  // PC-14 dropout bitmap: [kBr,kBc] keep-bits x2 stages past the bias area
  // (same layout/gating as the sm120 split_d launcher; env escape per call,
  // debug builds only).
#ifdef ENABLE_FFPA_FP16_BUILD_DEBUG
  bool dropout_bitmap_on =
      has_dropout && getenv("FFPA_DROPOUT_BITMAP_DISABLE") == nullptr;
#else
  bool dropout_bitmap_on = has_dropout;
#endif
  constexpr int kBitmapBytes = kBr * kBc / 8 * 2;
  if (dropout_bitmap_on) {
    const long long total = (((long long)kSmemBytes + 15) & ~15) + kBitmapBytes;
    const long long base16 = ((long long)kSmemBytes + 15) & ~15;
    if (total > max_smem_optin || 101376 / total < 101376 / base16)
      dropout_bitmap_on = false;
    else
      kSmemBytes = (int)total;
  }

  const dim3 block(kNumThreads, 1, 1);
  const dim3 grid(utils::div_ceil(Nq, kBr), Nb * Nh, 1);

  const c10::cuda::OptionalCUDAGuard device_guard(Q.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  float* softmax_lse_ptr =
      softmax_lse.numel() > 0 ? softmax_lse.data_ptr<float>() : nullptr;
  auto Q_ptr = reinterpret_cast<Element*>(Q.data_ptr());
  auto K_ptr = reinterpret_cast<Element*>(K.data_ptr());
  auto V_ptr = reinterpret_cast<Element*>(V.data_ptr());
  auto O_ptr = reinterpret_cast<Element*>(O.data_ptr());

  auto launch_variant = [&](auto kernel_func) {
    cudaFuncSetAttribute(
        kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
    kernel_func<<<grid, block, kSmemBytes, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, softmax_lse_ptr, Nq, Nkv, Nh, Nh_kv, scale,
        Tc, causal, attn_bias_ptr, attn_bias_dtype, attn_bias_stride_b,
        attn_bias_stride_h, attn_bias_stride_m, attn_bias_stride_n, dropout_p_f,
        philox_seed_u, philox_offset_u, bias_plan.mode,
        bias_plan.mode != 0 ? bias_plan.m_total : (long long)1,
        dropout_bitmap_on ? 1 : 0);
  };

  if (has_attn_bias && has_dropout) {
    launch_variant(split_d_fwd_cute_sm80<Traits, kStagesQK, kStagesPV, 1, 1>);
  } else if (has_attn_bias) {
    launch_variant(split_d_fwd_cute_sm80<Traits, kStagesQK, kStagesPV, 1, 0>);
  } else if (has_dropout) {
    launch_variant(split_d_fwd_cute_sm80<Traits, kStagesQK, kStagesPV, 0, 1>);
  } else {
    launch_variant(split_d_fwd_cute_sm80<Traits, kStagesQK, kStagesPV, 0, 0>);
  }
}

#endif  // ENABLE_FFPA_CUTE_EXT

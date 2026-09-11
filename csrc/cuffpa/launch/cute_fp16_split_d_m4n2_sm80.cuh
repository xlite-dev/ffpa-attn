#pragma once
// CuTe fp16/bf16 sm80 split_d_m4n2 launcher (cp.async, no TMA). PC-12/FC-12:
// sm_80 port of the sm120 split-D M4N2 kernel (kBr=64/kBc=64, atom layout
// (4,2,1), O regs = D/4) with the sm80 split_d's cp.async chunk pipeline.
// Covers D>=768 where the M8N1 split_d's o_acc = D/2 registers would spill.
// Not variant-split: instantiated implicitly through the family TUs.
#include "launch/common.cuh"
#ifdef ENABLE_FFPA_CUTE_EXT
#include "cute/sm_80/split_d_m4n2.cuh"

template <typename kDataType, const int kHeadDim, const int kStage>
void launch_cute_fwd_split_d_m4n2_sm80(torch::Tensor Q, torch::Tensor K,
                                       torch::Tensor V, torch::Tensor O,
                                       torch::Tensor attn_bias,
                                       torch::Tensor softmax_lse, int causal,
                                       double softmax_scale, double dropout_p,
                                       int64_t philox_seed,
                                       int64_t philox_offset) {
  using namespace cute;

  constexpr int kBr = 64;
  constexpr int kBc = 64;
  constexpr int kNumThreads = 256;
  // Same 64-wide chunks as the sm120 m4n2 launcher.
  constexpr int kQKDChunk = 64;
  constexpr int kVDChunk = 64;

  using Element = std::conditional_t<std::is_same_v<kDataType, __half>,
                                     cutlass::half_t, cutlass::bfloat16_t>;
  using Traits = ffpa_cute::FFPAAttnCuTeSplitDM4N2Traits<
      kHeadDim, kBr, kBc, kQKDChunk, kVDChunk, kStage, kStage, Element>;

  int max_smem_optin = 0;
  cudaDeviceGetAttribute(&max_smem_optin,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin,
                         Q.device().index());
  const int kSmemBytes = Traits::kSmemElems * (int)sizeof(Element);
  TORCH_CHECK(kSmemBytes <= max_smem_optin,
              "ffpa_attn: CuTe m4n2 sm80 kernel requires ", kSmemBytes,
              " bytes smem but device supports ", max_smem_optin,
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

  // Bias stays gmem-direct (mode 0): the sm120 tile modes ride TMA +
  // mbarrier handoff, which do not exist on sm_80.
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
        philox_seed_u, philox_offset_u);
  };

  if (has_attn_bias && has_dropout) {
    launch_variant(split_d_m4n2_fwd_cute_sm80<Traits, 1, 1>);
  } else if (has_attn_bias) {
    launch_variant(split_d_m4n2_fwd_cute_sm80<Traits, 1, 0>);
  } else if (has_dropout) {
    launch_variant(split_d_m4n2_fwd_cute_sm80<Traits, 0, 1>);
  } else {
    launch_variant(split_d_m4n2_fwd_cute_sm80<Traits, 0, 0>);
  }
}

#endif  // ENABLE_FFPA_CUTE_EXT

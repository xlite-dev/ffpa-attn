#pragma once
// CuTe fp8 persist_d sm89 launcher: cp.async (non-TMA) host glue around the
// sm_89 persist_d kernel. Reuses the TMA-free fp8 preprocess chain
// (prepare_fp8_inputs / smooth_k) and the shared quantization plumbing of
// the sm120 launcher; diverges in the kernel launch (no TMA descriptors)
// and the v1 scope (bias mode 0 gmem-direct, per_block Q/K/V quant,
// q_start_row=0, no dropout/smooth_v).
#include "launch/common.cuh"
#if defined(ENABLE_FFPA_CUTE_EXT) && defined(ENABLE_FFPA_TMA_EXT)
#include "cute/fp8/quantize_fp8.cuh"
#include "cute/fp8/prepare_inputs.cuh"
#include "generated/fwd_cute_fp8_preprocess.cuh"  // extern templates
#include "cute/fp8/smooth_k.cuh"
#include "cute/hadamard.cuh"
#include "cute/fp8/sm_89/persist_d.cuh"

namespace ffpa {

template <typename kDataType, const int kHeadDim, const int kStage,
          bool kQKInt8, int kHasAttnBias>
void launch_cute_fwd_persist_d_fp8_sm89(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
    torch::Tensor attn_bias, torch::Tensor softmax_lse, int causal,
    double softmax_scale, double dropout_p, int64_t philox_seed,
    int64_t philox_offset, bool fp8_smooth_k, bool fp8_smooth_v,
    int64_t fp8_q_quant_method, int64_t fp8_k_quant_method,
    int64_t fp8_v_quant_method, int64_t fp8_pv_acc_type, int q_start_row,
    bool fp8_hadamard) {
  using namespace cute;
  // v1 scope checks (sm89 cp.async path).
  TORCH_CHECK(dropout_p == 0.0, "fp8 sm89 path does not support dropout");
  TORCH_CHECK(q_start_row == 0, "fp8 sm89 v1 does not support q_start_row");
  TORCH_CHECK(fp8_q_quant_method == 0 && fp8_k_quant_method == 0,
              "fp8 sm89 v1 supports per_block Q/K quant only");
  TORCH_CHECK(fp8_v_quant_method == 0,
              "fp8 sm89 v1 supports per_block V quant only");
  TORCH_CHECK(!fp8_smooth_v, "fp8 sm89 v1 requires per_channel V for smooth_v");
  TORCH_CHECK(fp8_pv_acc_type == 0,
              "fp8 sm89 v1 supports f16 PV accumulator only");
  // The cp.async G2S loader cuts 64-column segments (1B elems, 16B copies):
  // D must be a multiple of 64 for the whole-row Q/K tiles to load.
  static_assert(kHeadDim % 64 == 0, "fp8 sm89 persist-D requires D % 64 == 0");
  // v1 writes O in BHND layout only; the diffusers NHD fast path is not
  // implemented yet (loud reject instead of silent wrong-layout output).
  TORCH_CHECK(!ffpa_is_nhd_view(O),
              "fp8 sm89 v1 supports BHND O only (no NHD fast path)");
  if (fp8_hadamard) {
    // WHT requires BHND-contiguous inputs; materialize packed copies for
    // any NHD-family view (same contract as the sm120 launcher).
    if (!Q.is_contiguous())
      Q = Q.contiguous();
    if (!K.is_contiguous())
      K = K.contiguous();
    if (Q.size(3) < kHeadDim)
      V = torch::constant_pad_nd(V, {0, kHeadDim - Q.size(3)}, 0.0);
    else if (!V.is_contiguous())
      V = V.contiguous();
    Q = ffpa::apply_wht_qk_sm120<kDataType, kHeadDim>(Q);
    K = ffpa::apply_wht_qk_sm120<kDataType, kHeadDim>(K);
  }
  const ffpa_fp8::Fp8InputLayout Lq =
      ffpa_layout_of(Q, Q.size(2), Q.size(3), /*allow_strided_rows=*/true);
  const ffpa_fp8::Fp8InputLayout Lkv =
      ffpa_layout_of(K, K.size(2), K.size(3), /*allow_strided_rows=*/true);
  const ffpa_fp8::Fp8InputLayout Lv =
      ffpa_layout_of(V, V.size(2), V.size(3), /*allow_strided_rows=*/true);
  const FfpaBiasParams bias = ffpa_bias_params_of(attn_bias, Q, K);
  TORCH_CHECK(kHasAttnBias == (bias.ptr != nullptr ? 1 : 0),
              "ffpa_attn: fp8 sm89 persist_d bias tag mismatch");

  constexpr int kBr = 128;
  constexpr int kBc = (kHeadDim <= 128) ? 128 : 64;
  constexpr int kQPersistBytes = kBr * kHeadDim;  // 1B/elem, Q stays in smem
  constexpr int kPerStageBytes = 2 * kBc * kHeadDim;
  constexpr int kMaxStages = (99 * 1024 - kQPersistBytes) / kPerStageBytes;
  constexpr int kStages =
      (kStage < 1) ? 2 : (kStage > kMaxStages ? kMaxStages : kStage);

  using ElementO = std::conditional_t<std::is_same_v<kDataType, __half>,
                                      cutlass::half_t, cutlass::bfloat16_t>;
  using Traits =
      ffpa_cute::FFPAAttnCuTePersistDFP8Traits<kHeadDim, ElementO, kBr, kBc,
                                               kStages, kStages, kQKInt8>;
  using Element = typename Traits::Element;
  using ElementQK = typename Traits::ElementQK;
  constexpr int kNumThreads = Traits::kNumThreads;
  constexpr int kSmemBytes = Traits::kSmemElems;  // 1B/elem, Q persist kept

  const int Nb = Q.size(0);
  const int Nh = Q.size(1);
  const int Nh_kv = K.size(1);
  const int Nq = Q.size(2);
  const int Nkv = K.size(2);
  const int Tc = utils::div_ceil(Nkv, kBc);
  const float scale = static_cast<float>(softmax_scale);
  const int n_rb_q = utils::div_ceil(Nq, kBr);
  const int n_rb_kv = utils::div_ceil(Nkv, kBc);
  const int Nkv_pad = (Nkv + 15) / 16 * 16;
  const int D_og = Q.size(3);

  const c10::cuda::OptionalCUDAGuard device_guard(Q.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const bool qk_per_thread = false;
  const bool v_per_channel = false;
  const bool v_smooth_mean = false;
  const float v_r = 448.0f;
  const bool reorg_free = true;
  const ffpa_fp8::Fp8QuantizedInputs qi =
      ffpa_fp8::prepare_fp8_inputs<kDataType, kBr, kBc, kHeadDim, kQKInt8>(
          Q, K, V, Lq, Lkv, Lv, Nb, Nh, Nh_kv, Nq, Nkv, n_rb_q, n_rb_kv,
          Nkv_pad, D_og, fp8_smooth_k, qk_per_thread, v_per_channel,
          v_smooth_mean, v_r, reorg_free, stream);

  int max_smem_optin = 0;
  cudaDeviceGetAttribute(
      &max_smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, Q.get_device());
  const int dyn_limit = max_smem_optin - 256;
  TORCH_CHECK(kSmemBytes <= dyn_limit,
              "ffpa_attn: fp8 sm89 persist_d D=", kHeadDim, " needs ",
              kSmemBytes, "B smem, device opt-in allows ", dyn_limit);

  float* softmax_lse_ptr =
      softmax_lse.numel() > 0 ? softmax_lse.data_ptr<float>() : nullptr;
  auto O_ptr = reinterpret_cast<ElementO*>(O.data_ptr());
  const dim3 block(kNumThreads, 1, 1);
  const dim3 grid(utils::div_ceil(Nq, kBr), Nb * Nh, 1);

  constexpr bool kPVAccF16 = true;  // fp8_pv_acc_type == 0 (v1)
  const auto kernel = ffpa_fp8::persist_d_fwd_cute_fp8_sm89<
      Traits, ElementO, kHasAttnBias, kPVAccF16,
      /*kVPerChannel=*/false, /*kQKPerThread=*/false, /*kReorgFree=*/true>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       kSmemBytes);
  kernel<<<grid, block, kSmemBytes, stream>>>(
      reinterpret_cast<ElementQK*>(qi.q8.data_ptr()),
      reinterpret_cast<ElementQK*>(qi.k8.data_ptr()),
      reinterpret_cast<Element*>(qi.vt8.data_ptr()), O_ptr, softmax_lse_ptr,
      qi.q_scale.data_ptr<float>(), qi.k_scale.data_ptr<float>(),
      qi.v_scale.data_ptr<float>(), Nq, Nkv, Nh, Nh_kv, n_rb_q, n_rb_kv, scale,
      Tc, causal, Nkv_pad, qi.km_f32_ptr, qi.vm_kernel, bias.ptr, bias.dtype,
      bias.stride_b, bias.stride_h, bias.stride_m, bias.stride_n);
}

}  // namespace ffpa
#endif

#pragma once
// CuTe fp8 split_d sm89 launcher: cp.async (non-TMA) host glue around the
// sm_89 split-D kernel (224 < D <= 512, D % 64 == 0). Reuses the TMA-free
// fp8 preprocess chain (prepare_fp8_inputs / smooth_k) and the shared
// quantization plumbing of the sm89 persist-D launcher; diverges in the
// kernel launch geometry (kBr=128 / 256T, split-D chunks, per-v_chunk O
// accumulators) and the D window. Same scope as the persist-D v1: bias
// mode 0 gmem-direct, per_block/per_thread QK (matched) +
// per_block/per_channel V with smooth_v, f16 PV acc only, no dropout,
// BHND O only, q_start_row for the fp8-hybrid stage-2.
#include "launch/common.cuh"
// The sm89 launcher is TMA-free (cp.async host glue); it compiles under
// the full TMA ext as well as the sm_89-only ext (real Ada builds where
// TMA must stay off).
#if defined(ENABLE_FFPA_CUTE_EXT) && \
    (defined(ENABLE_FFPA_TMA_EXT) || defined(ENABLE_FFPA_FP8_SM89_EXT))
#include "cute/fp8/quantize_fp8.cuh"
#include "cute/fp8/prepare_inputs.cuh"
#include "generated/fwd_cute_fp8_preprocess.cuh"  // extern templates
#include "cute/fp8/smooth_k.cuh"
#include "cute/hadamard.cuh"
#include "cute/fp8/sm_89/split_d.cuh"

namespace ffpa {

template <typename kDataType, const int kHeadDim, const int kStage,
          bool kQKInt8, int kHasAttnBias>
void launch_cute_fwd_split_d_fp8_sm89(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
    torch::Tensor attn_bias, torch::Tensor softmax_lse, int causal,
    double softmax_scale, double dropout_p, int64_t philox_seed,
    int64_t philox_offset, bool fp8_smooth_k, bool fp8_smooth_v,
    int64_t fp8_q_quant_method, int64_t fp8_k_quant_method,
    int64_t fp8_v_quant_method, int64_t fp8_pv_acc_type, int q_start_row,
    bool fp8_hadamard) {
  using namespace cute;
  // v1 scope checks (sm89 cp.async path), mirroring persist_d_sm89.
  static_assert(kHeadDim > 224 && kHeadDim <= 512 && kHeadDim % 64 == 0,
                "fp8 sm89 split-D covers 224 < D <= 512 with D % 64 == 0");
  TORCH_CHECK(dropout_p == 0.0, "fp8 sm89 path does not support dropout");
  TORCH_CHECK((fp8_q_quant_method == 0 && fp8_k_quant_method == 0) ||
                  (fp8_q_quant_method == 2 && fp8_k_quant_method == 2),
              "fp8 sm89 supports per_block or per_thread Q/K quant (matched)");
  TORCH_CHECK(fp8_v_quant_method == 0 || fp8_v_quant_method == 1,
              "fp8 sm89 supports per_block or per_channel V quant");
  TORCH_CHECK(fp8_pv_acc_type == 0,
              "fp8 sm89 v1 supports f16 PV accumulator only");
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
              "ffpa_attn: fp8 sm89 split_d bias tag mismatch");

  // Split-D geometry (mirrors the sm120 fp8 split_d): kBr=128 / 256T with
  // 1 CTA/SM, kBc=128, QK chunks 32 / V chunks 64. The O accumulators are
  // D/2 fp32 per thread, so D=512 spills exactly like the sm120 kernel
  // (accepted; D>512 belongs to the m4n2 family, Phase 3).
  constexpr int kBr = 128;
  constexpr int kBc = 128;
  constexpr int kQKDChunk = 32;
  constexpr int kVDChunk = 64;
  constexpr int kPerStageBytes =
      (kBr + kBc) * kQKDChunk + kBc * kVDChunk;  // 1B/elem QK + V, 16KB
  constexpr int kMaxStages = (99 * 1024) / kPerStageBytes;
  constexpr int kStagesQK =
      (kStage < 2) ? 3 : (kStage > kMaxStages ? kMaxStages : kStage);
  constexpr int kStagesPV = kStagesQK;
  constexpr int kSmemBytes = kStagesQK * kPerStageBytes;

  using ElementO = std::conditional_t<std::is_same_v<kDataType, __half>,
                                      cutlass::half_t, cutlass::bfloat16_t>;
  using Traits =
      ffpa_cute::FFPAAttnCuTeSplitDFP8Traits<kHeadDim, ElementO, kBr, kBc,
                                             kQKDChunk, kVDChunk, kStagesQK,
                                             kStagesPV, kQKInt8>;
  using Element = typename Traits::Element;
  using ElementQK = typename Traits::ElementQK;
  constexpr int kNumThreads = Traits::kNumThreads;

  const int Nb = Q.size(0);
  const int Nh = Q.size(1);
  const int Nh_kv = K.size(1);
  const int Nq = Q.size(2);
  const int Nkv = K.size(2);
  const int Tc = utils::div_ceil(Nkv, kBc);
  const float scale = static_cast<float>(softmax_scale);
  const int n_rb_q = utils::div_ceil(Nq, kBr);
  const int n_rb_kv = utils::div_ceil(Nkv, kBc);
  // The cp.async V loader has no TMA OOB zero-fill: the tail kv tile reads
  // up to ceil(Nkv/kBc)*kBc columns of the V^T plane, and a read past the
  // plane row stride lands in the next D-row (or, for the last D-row of
  // the last (b,h) plane, past the vt8 allocation in foreign memory).
  // Those columns sit at masked kv positions where P8 == 0, but
  // 0 * NaN-garbage == NaN inside the fp8 PV MMA and poisons O. Padding
  // the stride to the full read extent keeps every read inside a valid or
  // zero-filled column: the V^T quantize kernels zero-fill [Nkv, Nkv_pad)
  // (their "0 * NaN garbage poisons O" contract) and their grids cover
  // exactly ceil(Nkv/kBc)*kBc columns.
  const int Nkv_pad = (Nkv + kBc - 1) / kBc * kBc;
  const int D_og = Q.size(3);

  const c10::cuda::OptionalCUDAGuard device_guard(Q.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const bool qk_per_thread = (fp8_q_quant_method == 2);
  const bool v_per_channel = (fp8_v_quant_method == 1);
  const bool v_smooth_mean = v_per_channel && fp8_smooth_v;
  TORCH_CHECK(!fp8_smooth_v || v_per_channel,
              "ffpa_attn: fp8_smooth_v requires fp8_v_quant_method="
              "'per_channel'");
  // Per-channel V + f16 PV acc compresses V8 to v_r=2.25 so one tile's
  // f16 inst_buf stays in range (kBc*448*2.25 <= 65504 would fail at
  // kBc=128; the kernel narrows P to kPQuantScalePerCh=224 there).
  const float v_r = v_per_channel ? 2.25f : 448.0f;
  // The shuffle-free perm pack (in-kernel) binds the permuted V^T the
  // quantize pre-kernel writes under reorg_free (same pairing contract as
  // the sm89 persist-D / sm120 split-D launchers).
  constexpr bool reorg_free = true;
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
              "ffpa_attn: fp8 sm89 split_d D=", kHeadDim, " needs ", kSmemBytes,
              "B smem, device opt-in allows ", dyn_limit);

  float* softmax_lse_ptr =
      softmax_lse.numel() > 0 ? softmax_lse.data_ptr<float>() : nullptr;
  auto O_ptr = reinterpret_cast<ElementO*>(O.data_ptr());
  const dim3 block(kNumThreads, 1, 1);
  TORCH_CHECK(q_start_row >= 0 && q_start_row < Nq,
              "ffpa_attn: q_start_row must be in [0, Nq)");
  // Hybrid stage-2: grid.x covers the rows past q_start_row.
  const dim3 grid(utils::div_ceil(Nq - q_start_row, kBr), Nb * Nh, 1);

  // The quant granularity only changes register-time constants and the
  // epilogue dequant; instantiate the four combinations over one launch
  // site (same table shape as the persist-D sm89 launcher).
  const auto launch = [&](auto qk_pt_c, auto v_pc_c) {
    const auto k =
        ffpa_fp8::split_d_fwd_cute_fp8_sm89<Traits, ElementO, kHasAttnBias,
                                            decltype(qk_pt_c)::value,
                                            decltype(v_pc_c)::value>;
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         kSmemBytes);
    k<<<grid, block, kSmemBytes, stream>>>(
        reinterpret_cast<ElementQK*>(qi.q8.data_ptr()),
        reinterpret_cast<ElementQK*>(qi.k8.data_ptr()),
        reinterpret_cast<Element*>(qi.vt8.data_ptr()), O_ptr, softmax_lse_ptr,
        qi.q_scale.data_ptr<float>(), qi.k_scale.data_ptr<float>(),
        qi.v_scale.data_ptr<float>(), Nq, Nkv, Nh, Nh_kv, n_rb_q, n_rb_kv,
        scale, Tc, causal, Nkv_pad, q_start_row, qi.km_f32_ptr, qi.vm_kernel,
        bias.ptr, bias.dtype, bias.stride_b, bias.stride_h, bias.stride_m,
        bias.stride_n);
  };
  if (qk_per_thread) {
    if (v_per_channel)
      launch(std::true_type{}, std::true_type{});
    else
      launch(std::true_type{}, std::false_type{});
  } else {
    if (v_per_channel)
      launch(std::false_type{}, std::true_type{});
    else
      launch(std::false_type{}, std::false_type{});
  }
}

}  // namespace ffpa

#endif  // ENABLE_FFPA_CUTE_EXT && (ENABLE_FFPA_TMA_EXT ||
        // ENABLE_FFPA_FP8_SM89_EXT)

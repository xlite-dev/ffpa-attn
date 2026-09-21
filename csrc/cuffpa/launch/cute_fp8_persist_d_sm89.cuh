#pragma once
// CuTe fp8 persist_d sm89 launcher: cp.async (non-TMA) host glue around the
// sm_89 persist_d kernel. Reuses the TMA-free fp8 preprocess chain
// (prepare_fp8_inputs / smooth_k) and the shared quantization plumbing of
// the sm120 launcher; diverges in the kernel launch (no TMA descriptors)
// and the scope (bias mode 0 gmem-direct, per_block/per_thread QK (matched)
// + per_block/per_channel V with smooth_v, q_start_row=0, no dropout).
#include <cstdio>
#include <cstdlib>

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
#include "cute/fp8/sm_89/persist_d.cuh"

namespace ffpa {

// One (kBc, kStages) schedule of the unified kBr=64 persist-D kernel.
// kBr=64 / 128T keeps the per-thread tile share (and REG:255 ceiling) of
// the old kBr=128/256T schedule while two CTAs per SM interleave two
// independent QK->softmax->PV chains -- the Sage2 structure. kStages is
// the K/V cp.async pipeline depth; every extra stage spends the smem
// that funds the 2nd resident CTA (D=128: kBc=128 S=1 -> 40KB/2 CTAs,
// S=2 -> 72KB/1 CTA; kBc=64 S=1/2/3 -> 16/32/48KB, all 2 CTAs).
template <typename kDataType, const int kHeadDim, bool kQKInt8,
          int kHasAttnBias, const int kBc, const int kStages>
void persist_d_fp8_sm89_variant(
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
  TORCH_CHECK((fp8_q_quant_method == 0 && fp8_k_quant_method == 0) ||
                  (fp8_q_quant_method == 2 && fp8_k_quant_method == 2),
              "fp8 sm89 supports per_block or per_thread Q/K quant (matched)");
  TORCH_CHECK(fp8_v_quant_method == 0 || fp8_v_quant_method == 1,
              "fp8 sm89 supports per_block or per_channel V quant");
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

  constexpr int kBr = 64;
  constexpr int kKTileBytes = kBc * kHeadDim;  // 1B/elem
  constexpr int kVTileBytes = kBc * kHeadDim;
  // Same-shape Q/K tiles (kBc == kBr) share one swizzle buffer: Q drains
  // before K[0] overwrites it; K stages 1..S-1 and the V stages follow.
  constexpr bool kQSharesK = kBr == kBc;
  constexpr int kSmemBytes =
      (kQSharesK ? 0 : kBr * kHeadDim) + kStages * (kKTileBytes + kVTileBytes);

  using ElementO = std::conditional_t<std::is_same_v<kDataType, __half>,
                                      cutlass::half_t, cutlass::bfloat16_t>;
  using Traits =
      ffpa_cute::FFPAAttnCuTePersistDFP8Traits<kHeadDim, ElementO, kBr, kBc,
                                               kStages, kStages, kQKInt8>;
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
  const int Nkv_pad = (Nkv + 15) / 16 * 16;
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
  // f16 inst_buf stays in range (kBc*448*2.25 <= 65504 for kBc=64).
  const float v_r = v_per_channel ? 2.25f : 448.0f;
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
  TORCH_CHECK(q_start_row >= 0 && q_start_row < Nq,
              "ffpa_attn: q_start_row must be in [0, Nq)");
  // Hybrid stage-2: grid.x covers the rows past q_start_row.
  const dim3 grid(utils::div_ceil(Nq - q_start_row, kBr), Nb * Nh, 1);

  // The quant granularity only changes register-time constants and the
  // epilogue dequant; instantiate the four combinations over one launch site.
  const auto launch = [&](auto qk_pt_c, auto v_pc_c) {
    const auto k =
        ffpa_fp8::persist_d_fwd_cute_fp8_sm89<Traits, ElementO, kHasAttnBias,
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
  // Unified kBr=64 / 128T: two resident CTAs per SM interleave two
  // independent QK->softmax->PV dependency chains (the Sage2 structure)
  // while keeping the old kBr=128/256T per-thread register share.
  // kBc=64 minimizes the smem footprint (16KB: Q shares K stage0) which
  // keeps both CTAs resident AND shrinks the smem carve so the L1 cache
  // left behind is bigger; single-stage K/V (S=1) wins because deeper
  // pipelines spend the smem that funds the 2nd CTA -- S=2 @ kBc=128
  // drops to 1 CTA/SM (+37% @ Nkv=16384), and even with both CTAs held
  // (kBc=64 S=2/3) the extra depth only costs. Bench vs Sage2 (B1 H32
  // D128 dense): -3.8% @ 4096, -1.1% @ 8192, +3.3% @ 16384 (med; min
  // ties). FFPA_SM89_PERSIST_KVCFG="kBc,stages" retunes for experiments.
  int kBc_cfg = 64, kStages_cfg = 1;
  if (const char* cfg = std::getenv("FFPA_SM89_PERSIST_KVCFG"))
    std::sscanf(cfg, "%d,%d", &kBc_cfg, &kStages_cfg);
#define FFPA_PERSIST_D_CFG(BC, ST)                                           \
  persist_d_fp8_sm89_variant<kDataType, kHeadDim, kQKInt8, kHasAttnBias, BC, \
                             ST>(                                            \
      Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale, dropout_p,  \
      philox_seed, philox_offset, fp8_smooth_k, fp8_smooth_v,                \
      fp8_q_quant_method, fp8_k_quant_method, fp8_v_quant_method,            \
      fp8_pv_acc_type, q_start_row, fp8_hadamard)
  const bool valid =
      (kBc_cfg == 128 && (kStages_cfg == 1 || kStages_cfg == 2)) ||
      (kBc_cfg == 64 && kStages_cfg >= 1 && kStages_cfg <= 3);
  TORCH_CHECK(valid, "ffpa_attn: unsupported FFPA_SM89_PERSIST_KVCFG=", kBc_cfg,
              ",", kStages_cfg);
  if (kBc_cfg == 128 && kStages_cfg == 1)
    FFPA_PERSIST_D_CFG(128, 1);
  else if (kBc_cfg == 128 && kStages_cfg == 2)
    FFPA_PERSIST_D_CFG(128, 2);
  else if (kBc_cfg == 64 && kStages_cfg == 1)
    FFPA_PERSIST_D_CFG(64, 1);
  else if (kBc_cfg == 64 && kStages_cfg == 2)
    FFPA_PERSIST_D_CFG(64, 2);
  else
    FFPA_PERSIST_D_CFG(64, 3);
#undef FFPA_PERSIST_D_CFG
}

}  // namespace ffpa
#endif

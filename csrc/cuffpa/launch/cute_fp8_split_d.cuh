#pragma once
// CuTe fp8 split_d launcher: kernel include + bias plan + variant
// body (`_v`), split out of launch/cute_fp8.cuh so the per-tag
// variant TUs (env.py) preprocess exactly one kernel table.
#include "launch/common.cuh"
#if defined(ENABLE_FFPA_CUTE_EXT) && defined(ENABLE_FFPA_TMA_EXT)
#include "cute/fp8/quantize_fp8.cuh"
#include "cute/fp8/prepare_inputs.cuh"
#include "generated/fwd_cute_fp8_preprocess.cuh"  // extern templates
#include "cute/fp8/smooth_k.cuh"
#include "cute/hadamard.cuh"
#include "cute/fp8/sm_120/split_d.cuh"

namespace ffpa {
// Single-source final bias-mode decision per impl: the wrapper dispatch and
// the variant body both call these, so a demote-rule drift fails the
// variant's tag TORCH_CHECK instead of silently launching the wrong kernel.
// The constexpr blocks inside mirror the launcher bodies; keep in sync.

template <typename kDataType, const int kHeadDim, const int kStage,
          bool kQKInt8>
inline FfpaBiasTilePlan fp8_split_d_bias_plan(const FfpaBiasParams& bias_p,
                                              int Nb, int Nh, int Nq, int Nkv,
                                              int dyn_limit) {
  constexpr int kBr = 128;
  constexpr int kBc = 128;
  constexpr int kQKDChunk = 32;
  constexpr int kVDChunk = 64;
  constexpr int kPerStageBytes =
      (kBr + kBc) * kQKDChunk + kBc * kVDChunk;  // 1B/elem QK + V
  constexpr int kMaxStages = (99 * 1024) / kPerStageBytes;
  constexpr int kStagesQK =
      (kStage < 2) ? 3 : (kStage > kMaxStages ? kMaxStages : kStage);
  using ElementO = std::conditional_t<std::is_same_v<kDataType, __half>,
                                      cutlass::half_t, cutlass::bfloat16_t>;
  using Traits =
      ffpa_cute::FFPAAttnCuTeSplitDFP8Traits<kHeadDim, ElementO, kBr, kBc,
                                             kQKDChunk, kVDChunk, kStagesQK,
                                             kStagesQK, kQKInt8>;
  constexpr long long kSmemBytes = Traits::kSmemElems;
  FfpaBiasTilePlan plan = ffpa_bias_tile_plan_of(bias_p, Nb, Nh, Nq, Nkv);
  if (plan.mode == 1)
    plan.mode = 0;
  const int bias_stages = (plan.mode == 2) ? 2 : 1;
  if (kSmemBytes + plan.tile_bytes(kBr, kBc, bias_stages) > dyn_limit)
    plan.mode = 0;
  if (plan.mode == 2 && kHeadDim >= 512)
    plan.mode = 0;  // mode-2 launch variants are compile-time excluded (D>=512)
  return plan;
}

}  // namespace ffpa
// Split-D FP8 launcher (headdim > 128): non-WS M8N1 kernel over quantized
// q8/k8/vt8 buffers. Fixed-P-scale only (FFPA_FP8_PQUANT_PER_ROW applies to
// the persist_d path only and is ignored here).
// Variant tags (kBiasOn, kModeL, kB4): see launch_cute_fwd_persist_d_fp8_
// sm120_v above; explicit instantiations live in the generated variant TUs.
template <typename kDataType, const int kHeadDim, const int kStage,
          bool kQKInt8, int kBiasOn, int kModeL, int kB4>
void launch_cute_fwd_split_d_fp8_sm120_v(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
    torch::Tensor attn_bias, torch::Tensor softmax_lse, int causal,
    double softmax_scale, double dropout_p, int64_t philox_seed,
    int64_t philox_offset, bool fp8_smooth_k, bool fp8_smooth_v,
    int64_t fp8_q_quant_method, int64_t fp8_k_quant_method,
    int64_t fp8_v_quant_method, int64_t fp8_pv_acc_type, int q_start_row,
    bool fp8_hadamard) {
  using namespace cute;
  // Hadamard: rotate Q/K (and zero-pad V) BEFORE anything reads D_og — D_og
  // is the row stride of every fp8 pre-kernel (kv-mean/quantize), so Q/K/V
  // must all become kHeadDim-wide together.
  if (fp8_hadamard) {
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
  // NHD (diffusers BNHD) zero-copy views: the fp8 pre-kernels read the
  // original gmem through Fp8InputLayout strides, including strided
  // fused-QKV chunk rows. V keeps its own descriptor since interleaved
  // chunks give it K's head layout but a wider row stride.
  const ffpa_fp8::Fp8InputLayout Lq =
      ffpa_layout_of(Q, Q.size(2), Q.size(3), /*allow_strided_rows=*/true);
  const ffpa_fp8::Fp8InputLayout Lkv =
      ffpa_layout_of(K, K.size(2), K.size(3), /*allow_strided_rows=*/true);
  const ffpa_fp8::Fp8InputLayout Lv =
      ffpa_layout_of(V, V.size(2), V.size(3), /*allow_strided_rows=*/true);
  TORCH_CHECK(dropout_p == 0.0, "fp8 sm120 path does not support dropout");
  const FfpaBiasParams bias = ffpa_bias_params_of(attn_bias, Q, K);
  const int bias_on = bias.ptr != nullptr ? 1 : 0;
  TORCH_CHECK(
      (fp8_q_quant_method == 0 && fp8_k_quant_method == 0) ||
          (fp8_q_quant_method == 2 && fp8_k_quant_method == 2),
      "ffpa_attn: Q/K quant method must be both per_block or both per_thread");
  const bool qk_per_thread = (fp8_q_quant_method == 2);
  const bool v_per_channel = (fp8_v_quant_method == 1);
  const bool v_smooth_mean = v_per_channel && fp8_smooth_v;
  const bool pv_acc_f16 = (fp8_pv_acc_type == 0);
  const float v_r = (v_per_channel && pv_acc_f16) ? 2.25f : 448.0f;
  TORCH_CHECK(
      !fp8_smooth_v || v_per_channel,
      "ffpa_attn: fp8_smooth_v requires fp8_v_quant_method='per_channel'");
  // Split-D reorg-free: PackC8bitToA8bitPermVT in-kernel + permuted V^T from
  // the quantize pre-kernel (same pairing as persist_d; M8N1 C/A layouts are
  // identical between the two families). Part of the split-d fused-rescale
  // optimization set: all-on measured +8.4% vs the 81dbf75 baseline on RTX
  // PRO 5000 (see the switches note in split_d.cuh); default off with the
  // rest so the off-path stays instruction-identical to the baseline.
  // persist_d keeps reorg_free=true (WS hides the extra pipe pressure).
  constexpr bool kUseFusedRescale = false;
  constexpr bool reorg_free = kUseFusedRescale;

  // kBr/kBc mirrored by env.py::_fp8_variant_blocks; keep in sync.
  constexpr int kBr = 128;
  constexpr int kBc = 128;
  constexpr int kQKDChunk = 32;
  constexpr int kVDChunk = 64;
  constexpr int kSmemBudgetBytes = 99 * 1024;
  constexpr int kPerStageBytes =
      (kBr + kBc) * kQKDChunk + kBc * kVDChunk;  // 1B/elem QK + V
  constexpr int kMaxStages = kSmemBudgetBytes / kPerStageBytes;
  constexpr int kStagesQK =
      (kStage < 2) ? 3 : (kStage > kMaxStages ? kMaxStages : kStage);
  constexpr int kStagesPV = kStagesQK;

  using ElementO = std::conditional_t<std::is_same_v<kDataType, __half>,
                                      cutlass::half_t, cutlass::bfloat16_t>;
  using Traits =
      ffpa_cute::FFPAAttnCuTeSplitDFP8Traits<kHeadDim, ElementO, kBr, kBc,
                                             kQKDChunk, kVDChunk, kStagesQK,
                                             kStagesPV, kQKInt8>;
  using Element = typename Traits::Element;
  using ElementQK = typename Traits::ElementQK;
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
  const int n_rb_q = utils::div_ceil(Nq, kBr);
  const int n_rb_kv = utils::div_ceil(Nkv, kBc);
  // TMA needs a 16-byte-aligned leading stride; fp8 rows are Nkv bytes, so pad.
  const int Nkv_pad = (Nkv + 15) / 16 * 16;
  // D_og: real input head_dim (may be < kHeadDim for non-32-mult pad path).
  const int D_og = Q.size(3);

  const c10::cuda::OptionalCUDAGuard device_guard(Q.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  // Stage-independent preprocessing (allocation + smooth-K mean + Q/K/V
  // quantize) lives in cute/fp8/prepare_inputs.cuh, instantiated once per
  // (dtype, D, kQKInt8) in the generated preprocess TU; the extern
  // template declarations come from generated/fwd_cute_fp8_preprocess.cuh.
  const ffpa_fp8::Fp8QuantizedInputs qi =
      ffpa_fp8::prepare_fp8_inputs<kDataType, kBr, kBc, kHeadDim, kQKInt8>(
          Q, K, V, Lq, Lkv, Lv, Nb, Nh, Nh_kv, Nq, Nkv, n_rb_q, n_rb_kv,
          Nkv_pad, D_og, fp8_smooth_k, qk_per_thread, v_per_channel,
          v_smooth_mean, v_r, reorg_free, stream);
  const torch::Tensor& q8 = qi.q8;
  const torch::Tensor& k8 = qi.k8;
  const torch::Tensor& vt8 = qi.vt8;
  const torch::Tensor& q_scale = qi.q_scale;
  const torch::Tensor& k_scale = qi.k_scale;
  const torch::Tensor& v_scale = qi.v_scale;
  const float* km_f32_ptr = qi.km_f32_ptr;
  const float* vm_kernel = qi.vm_kernel;

  const int total_q_rows = Nb * Nh * Nq;
  const int total_kv_rows = Nb * Nh_kv * Nkv;

  auto gQ =
      make_tensor(make_gmem_ptr(reinterpret_cast<ElementQK*>(q8.data_ptr())),
                  make_shape(total_q_rows, Int<kHeadDim>{}),
                  make_stride(Int<kHeadDim>{}, _1{}));
  auto gK =
      make_tensor(make_gmem_ptr(reinterpret_cast<ElementQK*>(k8.data_ptr())),
                  make_shape(total_kv_rows, Int<kHeadDim>{}),
                  make_stride(Int<kHeadDim>{}, _1{}));

  auto tma_q = make_tma_copy(SM90_TMA_LOAD{}, gQ, SmemLayoutQ{},
                             Shape<Int<kBr>, Int<kQKDChunk>>{}, _1{});
  auto tma_k = make_tma_copy(SM90_TMA_LOAD{}, gK, SmemLayoutK{},
                             Shape<Int<kBc>, Int<kQKDChunk>>{}, _1{});
  // V^T: flat [B*Nh_kv*D, Nkv] with 16B-aligned row stride Nkv_pad; the
  // kernel offsets rows by the KV head's D plane via domain_offset.
  auto mV = make_tensor(
      make_gmem_ptr(reinterpret_cast<Element*>(vt8.data_ptr())),
      make_shape(Nb * Nh_kv * kHeadDim, Nkv), make_stride(Nkv_pad, Int<1>{}));
  auto tma_v = make_tma_copy(SM90_TMA_LOAD{}, mV, SmemLayoutV{},
                             Shape<Int<kVDChunk>, Int<kBc>>{}, _1{});

  // NHD (diffusers BNHD packed) O, detected by storage: flat [Nb*Nq,
  // Nh*kHeadDim] with the head selecting the column-tile group (kernel folds
  // Nh_id*kDChunksV into the v_chunk walk). Both branches use dynamic int64
  // extents/strides so TmaO has a single type and the kernel takes a runtime
  // nhd_out branch (same pattern as the persist-D impls).
  const bool nhd_out = ffpa_is_nhd_view(O);
  auto gO = nhd_out
                ? make_tensor(
                      make_gmem_ptr(reinterpret_cast<ElementO*>(O.data_ptr())),
                      make_shape((int64_t)Nb * Nq, (int64_t)Nh * kHeadDim),
                      make_stride((int64_t)Nh * kHeadDim, _1{}))
                : make_tensor(
                      make_gmem_ptr(reinterpret_cast<ElementO*>(O.data_ptr())),
                      make_shape((int64_t)total_q_rows, (int64_t)kHeadDim),
                      make_stride((int64_t)kHeadDim, _1{}));
  auto tma_o = make_tma_copy(SM90_TMA_STORE{}, gO, SmemLayoutO{},
                             Shape<Int<kBr>, Int<kVDChunk>>{}, _1{});

  constexpr int kSmemBytes = Traits::kSmemElems;
  // PC-0-1 bias tile plan: final mode decided by the single-source helper
  // (the wrapper dispatch uses it too). Mode 3 measured no-win on this
  // family (D=320: 59.2 vs 50.9ms for mode 2) so it is never selected;
  // mode 2 is demoted to 0 for D>=512 (pipeline-starvation analysis in
  // fp8_split_d_bias_plan).
  int max_smem_optin = 0;
  cudaDeviceGetAttribute(
      &max_smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, Q.get_device());
  // Static smem (barrier arrays, <= 144B incl. the bias pair) is invisible
  // to the dynamic budget: reserve 256B so the attribute set cannot land
  // past the true ceiling (fp4 persist_d D=256 lesson).
  const int dyn_limit = max_smem_optin - 256;
  FfpaBiasTilePlan bias_plan;
  if (bias.ptr != nullptr) {
    FfpaBiasParams bias_p{bias.ptr,      bias.dtype,    bias.stride_b,
                          bias.stride_h, bias.stride_m, bias.stride_n};
    bias_plan =
        ffpa::fp8_split_d_bias_plan<kDataType, kHeadDim, kStage, kQKInt8>(
            bias_p, Nb, Nh, Nq, Nkv, dyn_limit);
  }
  TORCH_CHECK(kBiasOn == bias_on && kModeL == (bias_on ? bias_plan.mode : 0) &&
                  kB4 == ((kModeL == 2 && bias.dtype == 3) ? 1 : 0),
              "ffpa_attn: fp8 split_d D=", kHeadDim,
              " variant tag mismatch (wrapper dispatch vs plan)");
  const int bias_stages = (bias_plan.mode == 2) ? 2 : 1;
  const auto make_tma_bias = [&](auto b4_c) {
    constexpr int kBias4B = decltype(b4_c)::value;
    constexpr int bias_cols = kBc * (kBias4B ? 2 : 1);
    // Row-broadcast plane is the real [m_total, Nkv]; every demoted/dummy
    // case (mode 0, or mode 3 which never issues) keeps a 1-row plane where
    // bias_cols satisfies the 16B outer-stride assert (NOT plane_cols --
    // non-16-multiple Nkv would trip it). mode 0 points at the anchor so
    // the 16B address assert holds without touching user memory.
    const bool bias_desc_live = bias_plan.mode == 2;
    const int64_t plane_rows =
        bias_desc_live ? (int64_t)bias_plan.m_total : (int64_t)1;
    const int64_t plane_cols =
        (int64_t)std::max<long long>(Nkv, 1) * (kBias4B ? 2 : 1);
    const uint16_t* bias_desc_base =
        bias_plan.mode != 0 ? reinterpret_cast<const uint16_t*>(bias.ptr)
                            : &kBiasDummyAnchor;
    auto gB = make_tensor(
        make_gmem_ptr(bias_desc_base), make_shape(plane_rows, plane_cols),
        make_stride(bias_desc_live ? plane_cols : (int64_t)bias_cols, _1{}));
    auto sB = Layout<Shape<_1, Int<bias_cols>>, Stride<Int<bias_cols>, _1>>{};
    return make_tma_copy(SM90_TMA_LOAD{}, gB, sB, shape(sB), _1{});
  };
  auto tma_bias_r16 = make_tma_bias(std::integral_constant<int, 0>{});
  [[maybe_unused]] auto tma_bias_r32 =
      make_tma_bias(std::integral_constant<int, 1>{});
  // Mode 3 pads the resident bytes to a whole kBc tile: tail tiles'
  // unclamped injection reads stay in-allocation (pad zero-filled by the
  // resident load).
  const int kSmemBytesBias =
      (int)(((long long)kSmemBytes + 15) & ~15) +
      (int)((bias_plan.mode == 3)
                ? ((long long)(Nkv + kBc - 1) / kBc * kBc) * bias_plan.elem_size
                : bias_plan.tile_bytes(kBr, kBc, bias_stages));
  TORCH_CHECK(kSmemBytesBias <= dyn_limit,
              "ffpa_attn: fp8 split_d D=", kHeadDim, " needs ", kSmemBytesBias,
              "B smem, device opt-in allows ", dyn_limit,
              " (static reserved 256B)");
  float* softmax_lse_ptr =
      softmax_lse.numel() > 0 ? softmax_lse.data_ptr<float>() : nullptr;
  auto O_ptr = reinterpret_cast<ElementO*>(O.data_ptr());

  const dim3 block(Traits::kNumThreads, 1, 1);
  TORCH_CHECK(q_start_row >= 0 && q_start_row < Nq,
              "ffpa_attn: q_start_row must be in [0, Nq)");
  TORCH_CHECK(q_start_row % 128 == 0,
              "ffpa_attn: q_start_row must be a multiple of kBr=128");
  const dim3 grid(utils::div_ceil(Nq - q_start_row, kBr), Nb * Nh, 1);
  using TmaQ = decltype(tma_q);
  using TmaK = decltype(tma_k);
  using TmaV = decltype(tma_v);
  using TmaO = decltype(tma_o);
  auto launch_kernel = [&](auto kernel, auto tma_bias_sel_arg) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         kSmemBytesBias);
    kernel<<<grid, block, kSmemBytesBias, stream>>>(
        tma_q, tma_k, tma_v, tma_o, tma_bias_sel_arg, O_ptr, softmax_lse_ptr,
        q_scale.data_ptr<float>(), k_scale.data_ptr<float>(),
        v_scale.data_ptr<float>(), Nq, Nkv, Nh, Nh_kv, scale, Tc, causal,
        total_q_rows, total_kv_rows, n_rb_q, n_rb_kv, q_start_row, km_f32_ptr,
        vm_kernel, nhd_out, bias.ptr, bias.dtype, bias.stride_b, bias.stride_h,
        bias.stride_m, bias.stride_n,
        bias_plan.mode != 0 ? bias_plan.m_total : (long long)1);
  };
  // kHasAttnBias/kBiasMode are template parameters, so the variant table
  // below is instantiated once per compile-time tag (runtime plan -> tag).
  // The 8 config axes (qk_per_thread x v_per_channel x pv_acc_f16) select
  // the bools; the bias axis selects the tile mode x dtype.
  const auto launch_with = [&](auto bias_tag, auto tma_bias_sel, auto mode_c,
                               auto b4_c) {
    using TmaBiasSel = decltype(tma_bias_sel);
    const auto kernel_of = [&](auto qk_pt, auto v_pc, auto pv_f16) {
      return ffpa_fp8::split_d_fwd_cute_fp8_sm120<
          Traits, ElementO, TmaQ, TmaK, TmaV, TmaO, TmaBiasSel,
          decltype(pv_f16)::value, decltype(v_pc)::value,
          decltype(qk_pt)::value, reorg_free, kModeL, kB4, kBiasOn>;
    };
    using Ic = std::integral_constant<int, 1>;
    using Ic0 = std::integral_constant<int, 0>;
    if (qk_per_thread) {
      if (v_per_channel) {
        if (pv_acc_f16)
          launch_kernel(kernel_of(Ic{}, Ic{}, Ic{}), tma_bias_sel);
        else
          launch_kernel(kernel_of(Ic{}, Ic{}, Ic0{}), tma_bias_sel);
      } else if (pv_acc_f16) {
        launch_kernel(kernel_of(Ic{}, Ic0{}, Ic{}), tma_bias_sel);
      } else {
        launch_kernel(kernel_of(Ic{}, Ic0{}, Ic0{}), tma_bias_sel);
      }
    } else if (v_per_channel) {
      if (pv_acc_f16)
        launch_kernel(kernel_of(Ic0{}, Ic{}, Ic{}), tma_bias_sel);
      else
        launch_kernel(kernel_of(Ic0{}, Ic{}, Ic0{}), tma_bias_sel);
    } else if (pv_acc_f16) {
      launch_kernel(kernel_of(Ic0{}, Ic0{}, Ic{}), tma_bias_sel);
    } else {
      launch_kernel(kernel_of(Ic0{}, Ic0{}, Ic0{}), tma_bias_sel);
    }
  };
  // Compile-time pinned variant: only this tag's kernel table instantiates.
  // D>=512 never instantiates the mode-2 variants (fp8_split_d_bias_plan
  // demotes mode 2 to 0 there; env.py mirrors this in the variant TU set).
  if constexpr (kBiasOn == 0) {
    launch_with(std::integral_constant<int, 0>{}, tma_bias_r16,
                std::integral_constant<int, 0>{},
                std::integral_constant<int, 0>{});
  } else if constexpr (kModeL == 2) {
    if constexpr (kB4 == 1)
      launch_with(std::integral_constant<int, 1>{}, tma_bias_r32,
                  std::integral_constant<int, 2>{},
                  std::integral_constant<int, 1>{});
    else
      launch_with(std::integral_constant<int, 1>{}, tma_bias_r16,
                  std::integral_constant<int, 2>{},
                  std::integral_constant<int, 0>{});
  } else {
    launch_with(std::integral_constant<int, 1>{}, tma_bias_r16,
                std::integral_constant<int, 0>{},
                std::integral_constant<int, 0>{});
  }
}

#endif  // ENABLE_FFPA_CUTE_EXT && ENABLE_FFPA_TMA_EXT

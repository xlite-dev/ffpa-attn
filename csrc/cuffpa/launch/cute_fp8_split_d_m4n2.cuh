#pragma once
// CuTe fp8 split_d_m4n2 launcher: kernel include + bias plan + variant
// body (`_v`), split out of launch/cute_fp8.cuh so the per-tag
// variant TUs (env.py) preprocess exactly one kernel table.
#include "launch/common.cuh"
#if defined(ENABLE_FFPA_CUTE_EXT) && defined(ENABLE_FFPA_TMA_EXT)
#include "cute/fp8/quantize_fp8.cuh"
#include "cute/fp8/prepare_inputs.cuh"
#include "generated/fwd_cute_fp8_preprocess.cuh"  // extern templates
#include "cute/fp8/smooth_k.cuh"
#include "cute/hadamard.cuh"
#include "cute/fp8/sm_120/split_d_m4n2.cuh"

namespace ffpa {
// Single-source final bias-mode decision per impl: the wrapper dispatch and
// the variant body both call these, so a demote-rule drift fails the
// variant's tag TORCH_CHECK instead of silently launching the wrong kernel.
// The constexpr blocks inside mirror the launcher bodies; keep in sync.

template <typename kDataType, const int kHeadDim, const int kStage,
          bool kQKInt8>
inline FfpaBiasTilePlan fp8_m4n2_bias_plan(const FfpaBiasParams& bias_p, int Nb,
                                           int Nh, int Nq, int Nkv) {
  constexpr int kBr = 64;
  constexpr int kBc = 64;
  constexpr int kPerStageBytes = (kBr + kBc) * 64 + kBc * 64;
  constexpr int kFixedSmemBytes = kBr * kBc + 2 * 8 * 16 * 4;
  constexpr int kMaxStages = (99 * 1024 - kFixedSmemBytes) / kPerStageBytes;
  using ElementO = std::conditional_t<std::is_same_v<kDataType, __half>,
                                      cutlass::half_t, cutlass::bfloat16_t>;
  using Traits = ffpa_cute::FFPAAttnCuTeSplitDM4N2FP8Traits<
      kHeadDim, ElementO, kBr, kBc, 64, 64,
      (kStage < 2) ? 2 : (kStage > kMaxStages ? kMaxStages : kStage),
      (kStage < 2) ? 2 : (kStage > kMaxStages ? kMaxStages : kStage), kQKInt8>;
  constexpr int kBiasSmemBudgetBytes = 99 * 1024;
  FfpaBiasTilePlan plan = ffpa_bias_tile_plan_of(bias_p, Nb, Nh, Nq, Nkv);
  const int bias_stages = (plan.mode == 2) ? 2 : 1;
  if ((long long)Traits::kSmemElems + plan.tile_bytes(kBr, kBc, bias_stages) >
      kBiasSmemBudgetBytes)
    plan.mode = 0;
  if (plan.mode == 2) {
    const long long kv_pad = (Nkv + kBc - 1) / kBc * kBc;
    const long long base_align = ((long long)Traits::kSmemElems + 15) & ~15;
    const long long resident = base_align + kv_pad * plan.elem_size;
    if (resident <= kBiasSmemBudgetBytes &&
        101376 / resident >= 101376 / base_align)
      plan.mode = 3;
  }
  return plan;
}

}  // namespace ffpa
// Split-D M4N2 FP8 launcher: m4n2 atom layout (4,2,1) + fp8 e4m3 Q/K/V.
// Dispatched for D>=768 to avoid M8N1's D/2 register spill (O=D/2>255).
// M4N2 uses D/4 regs per thread; P goes through SMEM roundtrip (stmatrix->
// LDSM_N) since each N-warp holds only half the Bc columns.
// Variant tags: kBiasOn (attn_bias tensor present), kBiasPlanMode (bias
// tile mode: 0 = gmem-direct fallback, 1 = dense [kBr,kBc] TMA tile,
// 2 = row-broadcast TMA, 3 = resident row vector; mode 1 is m4n2-only),
// kBias4BytesPerElem (1 = 4-byte fp32 mask, 0 = 2-byte fp16/bf16 mask);
// explicit instantiations live in the generated variant TUs.
template <typename kDataType, const int kHeadDim, const int kStage,
          bool kQKInt8, int kBiasOn, int kBiasPlanMode, int kBias4BytesPerElem>
void launch_cute_fwd_split_d_m4n2_fp8_sm120_v(
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

  // kBr/kBc mirrored by env.py::_fp8_variant_blocks; keep in sync.
  constexpr int kBr = 64;
  constexpr int kBc = 64;
  constexpr int kQKDChunk = 64;
  constexpr int kVDChunk = 64;
  constexpr int kSmemBudgetBytes = 99 * 1024;
  // Per stage = Q(K+D) + K(K+D) + V(D+B): 1B/elem.
  // Fixed smem: P[kBr*kBc] + exchange[2*8*16*4B].
  constexpr int kPerStageBytes = (kBr + kBc) * kQKDChunk + kBc * kVDChunk;
  constexpr int kFixedSmemBytes = kBr * kBc + 2 * 8 * 16 * 4;
  constexpr int kMaxStages =
      (kSmemBudgetBytes - kFixedSmemBytes) / kPerStageBytes;
  constexpr int kStagesQK =
      (kStage < 2) ? 2 : (kStage > kMaxStages ? kMaxStages : kStage);
  constexpr int kStagesPV = kStagesQK;

  using ElementO = std::conditional_t<std::is_same_v<kDataType, __half>,
                                      cutlass::half_t, cutlass::bfloat16_t>;
  using Traits =
      ffpa_cute::FFPAAttnCuTeSplitDM4N2FP8Traits<kHeadDim, ElementO, kBr, kBc,
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
          v_smooth_mean, v_r, /*reorg_free=*/false, stream);
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

  // PC-0-1 bias tile plan: final mode (incl. the dense-tile m4n2 path and
  // the resident-vector upgrade) comes from the single-source helper the
  // wrapper dispatch also uses.
  FfpaBiasTilePlan bias_plan;
  if (bias.ptr != nullptr) {
    FfpaBiasParams bias_p{bias.ptr,      bias.dtype,    bias.stride_b,
                          bias.stride_h, bias.stride_m, bias.stride_n};
    bias_plan = ffpa::fp8_m4n2_bias_plan<kDataType, kHeadDim, kStage, kQKInt8>(
        bias_p, Nb, Nh, Nq, Nkv);
  }
  TORCH_CHECK(kBiasOn == bias_on &&
                  kBiasPlanMode == (bias_on ? bias_plan.mode : 0) &&
                  kBias4BytesPerElem ==
                      ((kBiasPlanMode != 0 && bias.dtype == 3) ? 1 : 0),
              "ffpa_attn: fp8 split_d m4n2 D=", kHeadDim,
              " variant tag mismatch (wrapper dispatch vs plan)");
  constexpr int kBiasSmemBudgetBytes = 99 * 1024;
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
        bias_plan.mode != 0 ? reinterpret_cast<const uint16_t*>(bias.ptr)
                            : &kBiasDummyAnchor;
    auto gB = make_tensor(make_gmem_ptr(bias_desc_base),
                          make_shape(plane_rows, plane_cols),
                          make_stride(plane_row_stride, _1{}));
    auto sB = [&] {
      if constexpr (kBiasModeT == 1)
        return Layout<Shape<Int<kBr>, Int<bias_cols>>,
                      Stride<Int<bias_cols>, _1>>{};
      else
        return Layout<Shape<_1, Int<bias_cols>>, Stride<Int<bias_cols>, _1>>{};
    }();
    return make_tma_copy(SM90_TMA_LOAD{}, gB, sB, shape(sB), _1{});
  };
  [[maybe_unused]] auto tma_bias_d16 = make_tma_bias(
      std::integral_constant<int, 1>{}, std::integral_constant<int, 0>{});
  [[maybe_unused]] auto tma_bias_d32 = make_tma_bias(
      std::integral_constant<int, 1>{}, std::integral_constant<int, 1>{});
  [[maybe_unused]] auto tma_bias_r16 = make_tma_bias(
      std::integral_constant<int, 2>{}, std::integral_constant<int, 0>{});
  [[maybe_unused]] auto tma_bias_r32 = make_tma_bias(
      std::integral_constant<int, 2>{}, std::integral_constant<int, 1>{});
  const int kSmemBytes =
      ((Traits::kSmemElems + 15) & ~15) +
      (int)((bias_plan.mode == 3)
                ? ((long long)(Nkv + kBc - 1) / kBc * kBc) * bias_plan.elem_size
                : bias_plan.tile_bytes(kBr, kBc, bias_stages));
  float* softmax_lse_ptr =
      softmax_lse.numel() > 0 ? softmax_lse.data_ptr<float>() : nullptr;
  auto O_ptr = reinterpret_cast<ElementO*>(O.data_ptr());

  const dim3 block(Traits::kNumThreads, 1, 1);
  TORCH_CHECK(q_start_row >= 0 && q_start_row < Nq,
              "ffpa_attn: q_start_row must be in [0, Nq)");
  TORCH_CHECK(q_start_row % 64 == 0,
              "ffpa_attn: q_start_row must be a multiple of kBr=64");
  const dim3 grid(utils::div_ceil(Nq - q_start_row, kBr), Nb * Nh, 1);
  using TmaQ = decltype(tma_q);
  using TmaK = decltype(tma_k);
  using TmaV = decltype(tma_v);
  using TmaO = decltype(tma_o);
  auto launch_kernel = [&](auto kernel, auto tma_bias_sel) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         kSmemBytes);
    kernel<<<grid, block, kSmemBytes, stream>>>(
        tma_q, tma_k, tma_v, tma_o, tma_bias_sel, O_ptr, softmax_lse_ptr,
        q_scale.data_ptr<float>(), k_scale.data_ptr<float>(),
        v_scale.data_ptr<float>(), Nq, Nkv, Nh, Nh_kv, scale, Tc, causal,
        total_q_rows, total_kv_rows, n_rb_q, n_rb_kv, q_start_row, km_f32_ptr,
        vm_kernel, nhd_out, bias.ptr, bias.dtype, bias.stride_b, bias.stride_h,
        bias.stride_m, bias.stride_n,
        bias_plan.mode != 0 ? bias_plan.m_total : (long long)1);
  };
  // kHasAttnBias/kBiasMode/kBias4B are template parameters (TMA box type),
  // so the variant table below is instantiated once per compile-time bias
  // tag combination; the runtime mode/dtype select the branch.
  const auto dispatch = [&](auto bias_tag, auto tma_bias_sel, auto mode_c,
                            auto b4_c) {
    using TmaBiasSel = decltype(tma_bias_sel);
    if (qk_per_thread) {
      // Per-thread QK quant (sage style): fragment-aligned dequant scales.
      if (v_per_channel && pv_acc_f16) {
        launch_kernel(
            ffpa_fp8::split_d_m4n2_fwd_cute_fp8_sm120<
                Traits, ElementO, TmaQ, TmaK, TmaV, TmaO, TmaBiasSel, true,
                true, true, kBiasPlanMode, kBias4BytesPerElem, kBiasOn>,
            tma_bias_sel);
      } else if (v_per_channel) {
        launch_kernel(
            ffpa_fp8::split_d_m4n2_fwd_cute_fp8_sm120<
                Traits, ElementO, TmaQ, TmaK, TmaV, TmaO, TmaBiasSel, false,
                true, true, kBiasPlanMode, kBias4BytesPerElem, kBiasOn>,
            tma_bias_sel);
      } else if (pv_acc_f16) {
        launch_kernel(
            ffpa_fp8::split_d_m4n2_fwd_cute_fp8_sm120<
                Traits, ElementO, TmaQ, TmaK, TmaV, TmaO, TmaBiasSel, true,
                false, true, kBiasPlanMode, kBias4BytesPerElem, kBiasOn>,
            tma_bias_sel);
      } else {
        launch_kernel(
            ffpa_fp8::split_d_m4n2_fwd_cute_fp8_sm120<
                Traits, ElementO, TmaQ, TmaK, TmaV, TmaO, TmaBiasSel, false,
                false, true, kBiasPlanMode, kBias4BytesPerElem, kBiasOn>,
            tma_bias_sel);
      }
    } else if (v_per_channel && pv_acc_f16) {
      launch_kernel(
          ffpa_fp8::split_d_m4n2_fwd_cute_fp8_sm120<
              Traits, ElementO, TmaQ, TmaK, TmaV, TmaO, TmaBiasSel, true, true,
              false, kBiasPlanMode, kBias4BytesPerElem, kBiasOn>,
          tma_bias_sel);
    } else if (v_per_channel) {
      launch_kernel(
          ffpa_fp8::split_d_m4n2_fwd_cute_fp8_sm120<
              Traits, ElementO, TmaQ, TmaK, TmaV, TmaO, TmaBiasSel, false, true,
              false, kBiasPlanMode, kBias4BytesPerElem, kBiasOn>,
          tma_bias_sel);
    } else if (pv_acc_f16) {
      launch_kernel(
          ffpa_fp8::split_d_m4n2_fwd_cute_fp8_sm120<
              Traits, ElementO, TmaQ, TmaK, TmaV, TmaO, TmaBiasSel, true, false,
              false, kBiasPlanMode, kBias4BytesPerElem, kBiasOn>,
          tma_bias_sel);
    } else {
      launch_kernel(
          ffpa_fp8::split_d_m4n2_fwd_cute_fp8_sm120<
              Traits, ElementO, TmaQ, TmaK, TmaV, TmaO, TmaBiasSel, false,
              false, false, kBiasPlanMode, kBias4BytesPerElem, kBiasOn>,
          tma_bias_sel);
    }
  };
  // Compile-time pinned variant: only this tag's kernel table instantiates.
  if constexpr (kBiasOn == 0) {
    dispatch(std::integral_constant<int, 0>{}, tma_bias_r16,
             std::integral_constant<int, 0>{},
             std::integral_constant<int, 0>{});
  } else if constexpr (kBiasPlanMode == 1) {
    if constexpr (kBias4BytesPerElem == 1)
      dispatch(std::integral_constant<int, 1>{}, tma_bias_d32,
               std::integral_constant<int, 1>{},
               std::integral_constant<int, 1>{});
    else
      dispatch(std::integral_constant<int, 1>{}, tma_bias_d16,
               std::integral_constant<int, 1>{},
               std::integral_constant<int, 0>{});
  } else if constexpr (kBiasPlanMode == 2) {
    if constexpr (kBias4BytesPerElem == 1)
      dispatch(std::integral_constant<int, 1>{}, tma_bias_r32,
               std::integral_constant<int, 2>{},
               std::integral_constant<int, 1>{});
    else
      dispatch(std::integral_constant<int, 1>{}, tma_bias_r16,
               std::integral_constant<int, 2>{},
               std::integral_constant<int, 0>{});
  } else if constexpr (kBiasPlanMode == 3) {
    if constexpr (kBias4BytesPerElem == 1)
      dispatch(std::integral_constant<int, 1>{}, tma_bias_r32,
               std::integral_constant<int, 3>{},
               std::integral_constant<int, 1>{});
    else
      dispatch(std::integral_constant<int, 1>{}, tma_bias_r16,
               std::integral_constant<int, 3>{},
               std::integral_constant<int, 0>{});
  } else {
    dispatch(std::integral_constant<int, 1>{}, tma_bias_r16,
             std::integral_constant<int, 0>{},
             std::integral_constant<int, 0>{});
  }
}

#endif  // ENABLE_FFPA_CUTE_EXT && ENABLE_FFPA_TMA_EXT

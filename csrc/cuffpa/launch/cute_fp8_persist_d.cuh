#pragma once
// CuTe fp8 persist_d launcher: kernel include + bias plan + variant
// body (`_v`), split out of launch/cute_fp8.cuh so the per-tag
// variant TUs (env.py) preprocess exactly one kernel table.
#include "launch/common.cuh"
#if defined(ENABLE_FFPA_CUTE_EXT) && defined(ENABLE_FFPA_TMA_EXT)
#include "cute/fp8/quantize_fp8.cuh"
#include "cute/fp8/prepare_inputs.cuh"
#include "generated/fwd_cute_fp8_preprocess.cuh"  // extern templates
#include "cute/fp8/smooth_k.cuh"
#include "cute/hadamard.cuh"
#include "cute/fp8/sm_120/persist_d.cuh"

namespace ffpa {
// Single-source final bias-mode decision per impl: the wrapper dispatch and
// the variant body both call these, so a demote-rule drift fails the
// variant's tag TORCH_CHECK instead of silently launching the wrong kernel.
// The constexpr blocks inside mirror the launcher bodies; keep in sync.

template <typename kDataType, const int kHeadDim, const int kStage,
          bool kQKInt8>
inline FfpaBiasTilePlan fp8_persist_d_bias_plan(const FfpaBiasParams& bias_p,
                                                int Nb, int Nh, int Nq, int Nkv,
                                                int dyn_limit) {
  constexpr int kBr = 128;
  constexpr int kBc = (kHeadDim <= 128) ? 128 : 64;
  constexpr int kQPersistBytes =
      ffpa_fp8::kPersistQs2rDefault ? 0 : kBr * kHeadDim;
  constexpr int kPerStageBytes = 2 * kBc * kHeadDim;
  constexpr int kMaxStages = (99 * 1024 - kQPersistBytes) / kPerStageBytes;
  constexpr int kStagesK =
      (kStage < 1) ? 2 : (kStage > kMaxStages ? kMaxStages : kStage);
  using ElementO = std::conditional_t<std::is_same_v<kDataType, __half>,
                                      cutlass::half_t, cutlass::bfloat16_t>;
  using Traits =
      ffpa_cute::FFPAAttnCuTePersistDFP8Traits<kHeadDim, ElementO, kBr, kBc,
                                               kStagesK, kStagesK, kQKInt8>;
  constexpr long long kSmemBytes =
      (Traits::kSmemElems -
       (ffpa_fp8::kPersistQs2rDefault ? Traits::kBr * Traits::kHeadDim : 0)) *
      sizeof(typename Traits::Element);
  FfpaBiasTilePlan plan = ffpa_bias_tile_plan_of(bias_p, Nb, Nh, Nq, Nkv);
  if (plan.mode == 1)
    plan.mode = 0;
  const auto bytes_of = [&](int m) {
    return (m == 3) ? (long long)(Nkv + kBc - 1) / kBc * kBc * plan.elem_size
                    : plan.tile_bytes(kBr, kBc, (m == 2) ? 2 : 1);
  };
  if (plan.mode == 2 && kSmemBytes + bytes_of(3) <= dyn_limit)
    plan.mode = 3;
  if (kSmemBytes + bytes_of(plan.mode) > dyn_limit)
    plan.mode = 0;
  return plan;
}

}  // namespace ffpa
// FP8 persist-D: fp16/bf16 in, internally blockwise-quantized (Q/K row-major
// to e4m3 or symmetric int8, V transposed to e4m3), then low-precision
// attention. kQKInt8: QK runs s8xs8->s32 MMA (cast to f32 before softmax).
// The `_v` suffix = variant body of the variant-TU split: a tag-pinned
// template whose explicit instantiations live in the env.py-generated
// per-tag TUs; the suffix-less wrapper does the runtime plan -> tag
// dispatch into these (same contract as the fp4 launchers).
// Variant tags: kBiasOn (attn_bias tensor present), kBiasPlanMode (bias
// tile mode: 0 = gmem-direct fallback, 1 = dense [kBr,kBc] TMA tile,
// 2 = row-broadcast TMA, 3 = resident row vector), kBias4BytesPerElem
// (1 = 4-byte fp32 mask, 0 = 2-byte fp16/bf16 mask). They pin the kernel
// set compiled into this instantiation; explicit instantiations live in
// the generated variant TUs (fwd_cute_fp8_variants.cuh), so family TUs
// only carry extern declarations.
template <typename kDataType, const int kHeadDim, const int kStage,
          bool kQKInt8, int kBiasOn, int kBiasPlanMode, int kBias4BytesPerElem>
void launch_cute_fwd_persist_d_fp8_sm120_v(
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
    // WHT requires BHND-contiguous inputs; materialize packed copies for
    // any NHD-family view (rare combo — the quantize kernels below are
    // NHD-native, only the WHT kernel is not). V must join the same
    // packing: pad it to kHeadDim or materialize it BHND when already wide.
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
  // original gmem through Fp8InputLayout strides, so no permute copy is
  // needed. Strided-NHD rows (fused-QKV chunk views, e.g. FLUX.2
  // single-stream) are accepted via the relaxed gate; V keeps its own
  // descriptor since interleaved chunks give it K's head layout but a
  // wider row stride.
  const ffpa_fp8::Fp8InputLayout Lq =
      ffpa_layout_of(Q, Q.size(2), Q.size(3), /*allow_strided_rows=*/true);
  const ffpa_fp8::Fp8InputLayout Lkv =
      ffpa_layout_of(K, K.size(2), K.size(3), /*allow_strided_rows=*/true);
  const ffpa_fp8::Fp8InputLayout Lv =
      ffpa_layout_of(V, V.size(2), V.size(3), /*allow_strided_rows=*/true);
  TORCH_CHECK(dropout_p == 0.0, "fp8 sm120 path does not support dropout");
  const FfpaBiasParams bias = ffpa_bias_params_of(attn_bias, Q, K);
  const int bias_on = bias.ptr != nullptr ? 1 : 0;
  // q/k only support per-block quant today; per-channel is reserved for
  // future kernel work.
  TORCH_CHECK(
      (fp8_q_quant_method == 0 && fp8_k_quant_method == 0) ||
          (fp8_q_quant_method == 2 && fp8_k_quant_method == 2),
      "ffpa_attn: Q/K quant method must be both per_block or both per_thread");
  const bool qk_per_thread = (fp8_q_quant_method == 2);
  // FP8 V quant / PV acc / smooth_v are API params (v_quant_method:
  // 0=per_block, 1=per_channel; pv_acc_type: 0=f16, 1=f32). All fp8
  // kernels (persist_d, split_d, m4n2) support every combination.
  const bool v_per_channel = (fp8_v_quant_method == 1);
  const bool v_smooth_mean = v_per_channel && fp8_smooth_v;
  const bool pv_acc_f16 = (fp8_pv_acc_type == 0);
  const float v_r = (v_per_channel && pv_acc_f16) ? 2.25f : 448.0f;
  TORCH_CHECK(
      !fp8_smooth_v || v_per_channel,
      "ffpa_attn: fp8_smooth_v requires fp8_v_quant_method='per_channel'");
#ifdef ENABLE_FFPA_FP8_BUILD_DEBUG
  const bool pquant_per_row = getenv("FFPA_FP8_PQUANT_PER_ROW") != nullptr;
#else
  const bool pquant_per_row = false;
#endif
  // Reorg-free PV pack (Phase 3): the attention kernel packs P into the PV A
  // operand without cross-lane shuffles, leaving a permuted k-indexing that
  // the quantize pre-kernel must match by storing V^T columns permuted
  // (VTPermInv32). Both sides derive from this single flag so the pairing can
  // never diverge. Default for EVERY persist_d fp8 config: the mechanism only
  // depends on the shared m16n8k32 fragment layouts, so it is QK element
  // (fp8/int8), PV acc (f16/f32) and Q/K/V/P granularity agnostic; the
  // cross-lane ReorgC8bitToA8bit fallback stays compiled (flip this gate to
  // restore; split_d carries its own identical gate).
  constexpr bool reorg_free = true;

  // kBr/kBc choice mirrored by env.py::_fp8_variant_blocks (the extern-
  // template table); keep the two in sync.
  constexpr int kBr = 128;
  // D>128 must shrink kBc to fit the 99KB smem budget (1B/elem fp8): D=224
  // with kBc=64 -> Q(28KB)+2*stage(28KB)=84KB. Mirrors fp16 persist_d's
  // D-scaled kBc (L354).
  constexpr int kBc = (kHeadDim <= 128) ? 128 : 64;
  constexpr int kSmemBudgetBytes = 99 * 1024;
  // kPersistQs2rDefault: K stage 0 reuses the Q area, so the Q bytes drop
  // out of the smem budget (stages 3 -> 96KB fits the 99KB sm_120 limit).
  constexpr int kQPersistBytes =
      ffpa_fp8::kPersistQs2rDefault ? 0 : kBr * kHeadDim;  // e4m3/int8 = 1B
  constexpr int kPerStageBytes = 2 * kBc * kHeadDim;
  constexpr int kMaxStages =
      (kSmemBudgetBytes - kQPersistBytes) / kPerStageBytes;
  constexpr int kStagesK =
      (kStage < 1) ? 2 : (kStage > kMaxStages ? kMaxStages : kStage);
  constexpr int kStagesV = kStagesK;
  constexpr int kNumThreads = 384;

  using ElementO = std::conditional_t<std::is_same_v<kDataType, __half>,
                                      cutlass::half_t, cutlass::bfloat16_t>;
  using Traits =
      ffpa_cute::FFPAAttnCuTePersistDFP8Traits<kHeadDim, ElementO, kBr, kBc,
                                               kStagesK, kStagesV, kQKInt8>;
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
                             Shape<Int<kBr>, Int<kHeadDim>>{}, _1{});
  auto tma_k = make_tma_copy(SM90_TMA_LOAD{}, gK, SmemLayoutK{},
                             Shape<Int<kBc>, Int<kHeadDim>>{}, _1{});
  // V^T: single flat descriptor over [B*Nh_kv*D, Nkv] with a 16B-aligned row
  // stride Nkv_pad (TMA requires the leading stride % 16 == 0); globalDim[1]
  // stays Nkv so out-of-range columns in the last partial tile zero-fill.
  auto mV = make_tensor(
      make_gmem_ptr(reinterpret_cast<Element*>(vt8.data_ptr())),
      make_shape(Nb * Nh_kv * kHeadDim, Nkv), make_stride(Nkv_pad, Int<1>{}));
  auto tma_v = make_tma_copy(SM90_TMA_LOAD{}, mV, SmemLayoutV{},
                             Shape<Int<kHeadDim>, Int<kBc>>{}, _1{});

  // O store descriptor: full [total_q_rows, kHeadDim] ElementO tensor for a
  // BHND-packed O; the per-(batch,head) origin is injected via domain_offset
  // in the kernel. NHD (diffusers BNHD packed) O, detected by storage: flat
  // [Nb*Nq, Nh*kHeadDim] with the head selecting the column tile — mirrors
  // the fp16 persist-D NHD Q load. Both branches use dynamic int64
  // extents/strides so TmaO has a single type and the kernel takes a runtime
  // nhd_out branch. The smem layout mirrors the kernel's SmemLayoutO staging
  // (SW128, ElementO).
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
                             Shape<Int<kBr>, Int<kHeadDim>>{}, _1{});

  // kPersistQs2rDefault: K stage 0 reuses the one-shot Q tile area in the
  // kernel, so the Q bytes drop out of the smem allocation.
  constexpr int kSmemBytes =
      (Traits::kSmemElems -
       (ffpa_fp8::kPersistQs2rDefault ? Traits::kBr * Traits::kHeadDim : 0)) *
      sizeof(Element);
  // PC-0-1 bias tile plan: final mode (incl. resident-vector upgrade to 3
  // and smem demote to 0) comes from the single-source helper the wrapper
  // dispatch also uses; the tag TORCH_CHECK below fails loudly if the two
  // ever drift apart.
  int max_smem_optin = 0;
  cudaDeviceGetAttribute(
      &max_smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, Q.get_device());
  // Static smem (barrier arrays, ~160B incl. the bias pair) is invisible to
  // the dynamic budget: reserve 256B so the attribute set cannot land past
  // the true ceiling (fp4 persist_d D=256 lesson).
  const int dyn_limit = max_smem_optin - 256;
  FfpaBiasTilePlan bias_plan;
  if (bias.ptr != nullptr) {
    FfpaBiasParams bias_p{bias.ptr,      bias.dtype,    bias.stride_b,
                          bias.stride_h, bias.stride_m, bias.stride_n};
    bias_plan =
        ffpa::fp8_persist_d_bias_plan<kDataType, kHeadDim, kStage, kQKInt8>(
            bias_p, Nb, Nh, Nq, Nkv, dyn_limit);
  }
  TORCH_CHECK(kBiasOn == bias_on &&
                  kBiasPlanMode == (bias_on ? bias_plan.mode : 0) &&
                  kBias4BytesPerElem ==
                      ((kBiasPlanMode == 2 && bias.dtype == 3) ? 1 : 0),
              "ffpa_attn: fp8 persist_d D=", kHeadDim,
              " variant tag mismatch (wrapper dispatch vs plan)");
  const auto bias_bytes_of = [&](int m) {
    // Mode 3 pads the resident bytes to a whole kBc tile (the kernel's
    // resident fill zero-fills the pad segment; tail tiles' unclamped
    // injection reads stay in-allocation).
    return (m == 3)
               ? (long long)(Nkv + kBc - 1) / kBc * kBc * bias_plan.elem_size
               : bias_plan.tile_bytes(kBr, kBc, (m == 2) ? 2 : 1);
  };
  const int kSmemBytesBias = (int)(((long long)kSmemBytes + 15) & ~15) +
                             (int)bias_bytes_of(bias_plan.mode);
  TORCH_CHECK(kSmemBytesBias <= dyn_limit,
              "ffpa_attn: fp8 persist_d D=", kHeadDim, " needs ",
              kSmemBytesBias, "B smem, device opt-in allows ", dyn_limit,
              " (static reserved 256B)");
  const auto make_tma_bias = [&](auto b4_c) {
    constexpr int kBias4B = decltype(b4_c)::value;
    constexpr int bias_cols = kBc * (kBias4B ? 2 : 1);
    // Row-broadcast plane is the real [m_total, Nkv]; demoted/dummy cases
    // (mode 0, or mode 3 which never issues) keep a 1-row plane where
    // bias_cols satisfies the 16B outer-stride assert (a live plane needs
    // Nkv % 8; the dummy must NOT inherit plane_cols or non-16-multiple
    // Nkv values trip the assert). mode 0 points at the anchor so the 16B
    // address assert holds without touching user memory.
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
  float* softmax_lse_ptr =
      softmax_lse.numel() > 0 ? softmax_lse.data_ptr<float>() : nullptr;
  auto O_ptr = reinterpret_cast<ElementO*>(O.data_ptr());

  const dim3 block(kNumThreads, 1, 1);
  TORCH_CHECK(q_start_row >= 0 && q_start_row < Nq,
              "ffpa_attn: q_start_row must be in [0, Nq)");
  TORCH_CHECK(q_start_row % 128 == 0,
              "ffpa_attn: q_start_row must be a multiple of kBr=128");
  const dim3 grid(utils::div_ceil(Nq - q_start_row, kBr), Nb * Nh, 1);
  // P quant granularity: fixed 1/448 scale (fast, default) vs per-row scale
  // (higher accuracy). Opt into per-row with FFPA_FP8_PQUANT_PER_ROW=1.
  using TmaQ = decltype(tma_q);
  using TmaK = decltype(tma_k);
  using TmaV = decltype(tma_v);
  using TmaO = decltype(tma_o);
  const auto launch_kernel = [&](auto kernel, auto tma_bias_sel_arg) {
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
  // Axes: pquant_per_row x qk_per_thread x v_per_channel x pv_acc_f16.
  const auto launch_with = [&](auto bias_tag, auto tma_bias_sel, auto mode_c,
                               auto b4_c) {
    using TmaBiasSel = decltype(tma_bias_sel);
    const auto kernel_of = [&](auto pq_pr, auto qk_pt, auto v_pc, auto pv_f16) {
      return ffpa_fp8::persist_d_ws_fwd_cute_fp8_sm120<
          Traits, ElementO, TmaQ, TmaK, TmaV, TmaO, TmaBiasSel,
          decltype(pq_pr)::value, decltype(pv_f16)::value,
          decltype(v_pc)::value, decltype(qk_pt)::value, reorg_free,
          ffpa_fp8::kPersistQs2rDefault, kBiasPlanMode, kBias4BytesPerElem,
          kBiasOn>;
    };
    using Ic = std::integral_constant<int, 1>;
    using Ic0 = std::integral_constant<int, 0>;
    if (qk_per_thread) {
      // Per-thread QK quant (sage style): fragment-aligned dequant scales.
      if (v_per_channel) {
        if (pv_acc_f16)
          launch_kernel(kernel_of(Ic0{}, Ic{}, Ic{}, Ic{}), tma_bias_sel);
        else
          launch_kernel(kernel_of(Ic0{}, Ic{}, Ic{}, Ic0{}), tma_bias_sel);
      } else if (pv_acc_f16) {
        launch_kernel(kernel_of(Ic0{}, Ic{}, Ic0{}, Ic{}), tma_bias_sel);
      } else {
        launch_kernel(kernel_of(Ic0{}, Ic{}, Ic0{}, Ic0{}), tma_bias_sel);
      }
    } else if (pquant_per_row) {
      launch_kernel(kernel_of(Ic{}, Ic0{}, Ic0{}, Ic0{}), tma_bias_sel);
    } else if (v_per_channel) {
      // Per-channel V (sage-style): V per-D scale, P uses fixed 448;
      // epilogue dequants per-D.
      if (pv_acc_f16)
        launch_kernel(kernel_of(Ic0{}, Ic0{}, Ic{}, Ic{}), tma_bias_sel);
      else
        launch_kernel(kernel_of(Ic0{}, Ic0{}, Ic{}, Ic0{}), tma_bias_sel);
    } else if (pv_acc_f16) {
      // f8f8f16 PV (fp16 MMA accumulator) avoids the 22-bit f8f8f32
      // accumulator loss on causal early rows. See persist_d.cuh kPVAccF16.
      launch_kernel(kernel_of(Ic0{}, Ic0{}, Ic0{}, Ic{}), tma_bias_sel);
    } else {
      launch_kernel(kernel_of(Ic0{}, Ic0{}, Ic0{}, Ic0{}), tma_bias_sel);
    }
  };
  // Compile-time pinned variant: only this tag's kernel table instantiates.
  if constexpr (kBiasOn == 0) {
    launch_with(std::integral_constant<int, 0>{}, tma_bias_r16,
                std::integral_constant<int, 0>{},
                std::integral_constant<int, 0>{});
  } else if constexpr (kBiasPlanMode == 2) {
    if constexpr (kBias4BytesPerElem == 1)
      launch_with(std::integral_constant<int, 1>{}, tma_bias_r32,
                  std::integral_constant<int, 2>{},
                  std::integral_constant<int, 1>{});
    else
      launch_with(std::integral_constant<int, 1>{}, tma_bias_r16,
                  std::integral_constant<int, 2>{},
                  std::integral_constant<int, 0>{});
  } else if constexpr (kBiasPlanMode == 3) {
    // Resident-vector path: no bias TMA (dummy desc), runtime dtype only.
    launch_with(std::integral_constant<int, 1>{}, tma_bias_r16,
                std::integral_constant<int, 3>{},
                std::integral_constant<int, 0>{});
  } else {
    launch_with(std::integral_constant<int, 1>{}, tma_bias_r16,
                std::integral_constant<int, 0>{},
                std::integral_constant<int, 0>{});
  }
}

#endif  // ENABLE_FFPA_CUTE_EXT && ENABLE_FFPA_TMA_EXT

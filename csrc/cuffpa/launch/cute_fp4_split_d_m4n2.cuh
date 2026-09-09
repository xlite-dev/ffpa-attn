#pragma once
// CuTe fp4 split_d_m4n2 launcher: kernel include + bias plan + variant
// body (`_v`), split out of launch/cute_fp4.cuh so the per-tag
// variant TUs (env.py) preprocess exactly one kernel table.
#include "launch/common.cuh"
#if defined(ENABLE_FFPA_CUTE_EXT) && defined(ENABLE_FFPA_TMA_EXT)
#include "cute/fp8/smooth_k.cuh"
#include "cute/fp4/quantize_fp4.cuh"
#include "cute/hadamard.cuh"
#include "cute/fp4/delta_s.cuh"
#include "generated/fwd_cute_fp4_preprocess.cuh"  // extern templates
#include "cute/fp4/sm_120/split_d_m4n2.cuh"

namespace ffpa {
// Single-source final bias-mode decision per fp4 impl: the wrapper
// dispatch and the variant body both call these, so a demote-rule drift
// fails the variant tag TORCH_CHECK instead of silently mismatching.
// All three read the device smem opt-in (dyn_limit = optin - 256, the
// static barrier-array reserve) at the call site and pass it in.

// split_d m4n2: dense is m4n2-native but unwired (demote to 0); PC-0-5
// pins gmem-direct (mode 0) on regular builds - FFPA_BIAS_TILE_KEEP
// (debug build only) keeps the smem tile modes for investigation. No
// mode-3 upgrade on this family (deterministic 8B resident overread).
template <typename kDataType, const int kHeadDim>
inline FfpaBiasTilePlan fp4_m4n2_bias_plan(const FfpaBiasParams& bias_p, int Nb,
                                           int Nh, int Nq, int Nkv,
                                           int dyn_limit) {
  using ElementO = std::conditional_t<std::is_same_v<kDataType, __half>,
                                      cutlass::half_t, cutlass::bfloat16_t>;
  using Traits = ffpa_fp4::FFPAAttnCuTeSplitDM4N2FP4Traits<ElementO, kHeadDim>;
  constexpr int kBr = 64;
  constexpr int kBc = 64;
  FfpaBiasTilePlan plan = ffpa_bias_tile_plan_of(bias_p, Nb, Nh, Nq, Nkv);
  if (plan.mode == 1)
    plan.mode = 0;
#ifdef ENABLE_FFPA_FP4_BUILD_DEBUG
  if (plan.mode != 0 && getenv("FFPA_BIAS_TILE_KEEP") == nullptr)
    plan.mode = 0;
#else
  plan.mode = 0;
#endif
  const auto bias_bytes_of = [&](int m) {
    return (m == 3) ? (long long)(Nkv + kBc - 1) / kBc * kBc * plan.elem_size
                    : plan.tile_bytes(kBr, kBc, (m == 2) ? 2 : 1);
  };
  if ((long long)Traits::kSmemBytes + bias_bytes_of(plan.mode) > dyn_limit)
    plan.mode = 0;
  return plan;
}

}  // namespace ffpa
// NVFP4 split-D M4N2 launcher, headdims in [768, 1024]. Identical
// pre-kernel pipeline to the split-D fp4 launcher (km -> q_block_mean ->
// quantize -> delta_s); only the tile geometry changes (kBr=kBc=64, m4n2
// traits own the TMA descriptor shapes). smooth_v follows the same chain
// as split-D/persist-D (V^T quantize residual + epilogue add-back);
// fp4_pv_mm_type stays NVFP4-only here (see the wrapper: the MXFP8 PV
// atom needs Tile-K=128 but m4n2 tiles are kBc=64).
// Variant tags: kPvMxfp8 (always false here - m4n2 is NVFP4-only),
// kBiasOn (attn_bias tensor present), kBiasPlanMode (bias tile mode,
// see the persist_d notes; regular builds pin mode 0 per PC-0-5,
// FFPA_BIAS_TILE_KEEP (debug) keeps mode 2), kBias4BytesPerElem (1 =
// 4-byte fp32 mask, 0 = 2-byte fp16/bf16 mask).
template <typename kDataType, const int kHeadDim, bool kPvMxfp8, int kBiasOn,
          int kBiasPlanMode, int kBias4BytesPerElem>
void launch_cute_fwd_split_d_m4n2_fp4_sm120_v(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
    torch::Tensor attn_bias, torch::Tensor softmax_lse, int causal,
    double softmax_scale, int q_start_row, bool fp4_hadamard,
    bool fp4_smooth_v) {
  using namespace cute;
  constexpr int kBr = 64;
  constexpr int kBc = 64;
  constexpr int kNumThreads = 256;
  constexpr int kQKDChunk = 64;
  constexpr int kVDChunk = 64;

  using ElementO = std::conditional_t<std::is_same_v<kDataType, __half>,
                                      cutlass::half_t, cutlass::bfloat16_t>;
  using Traits = ffpa_fp4::FFPAAttnCuTeSplitDM4N2FP4Traits<ElementO, kHeadDim>;
  const FfpaBiasParams bias = ffpa_bias_params_of(attn_bias, Q, K);
  const int bias_on = bias.ptr != nullptr ? 1 : 0;
  using Element = typename Traits::Element;
  using ElementSF = typename Traits::ElementSF;
  auto prop = at::cuda::getCurrentDeviceProperties();
  using SmemLayoutQ = typename Traits::SmemLayoutQ;
  using SmemLayoutK = typename Traits::SmemLayoutK;
  using SmemLayoutVt = typename Traits::SmemLayoutVt;
  using SmemLayoutSFQ = typename Traits::SmemLayoutSFQ;
  using SmemLayoutSFK = typename Traits::SmemLayoutSFK;
  using SmemLayoutSFVt = typename Traits::SmemLayoutSFVt;
  using SmemLayoutDS = typename Traits::SmemLayoutDS;
  using SmemLayoutO = typename Traits::SmemLayoutO;
  using SmemLayoutAtomDS = typename Traits::SmemLayoutAtomDS;
  using BlkScaledConfig = typename Traits::BlkScaledConfig;

  const int Nb = Q.size(0);
  const int Nh = Q.size(1);
  const int Nh_kv = K.size(1);
  const int Nq = Q.size(2);
  const int Nkv = K.size(2);
  const int group = Nh / Nh_kv;
  // Padding stays 128-aligned (the SF gmem layouts and the q_block_mean /
  // delta_s 128-row blocks all assume it); only the attention tile is 64.
  const int Nq_pad = utils::div_ceil(Nq, 128) * 128;
  const int Nkv_pad = utils::div_ceil(Nkv, 128) * 128;
  const int Mb = Nq_pad / kBr;
  const int Mb_qm = Nq_pad / 128;
  const int Tc = Nkv_pad / kBc;
  const int total_q_rows = Nb * Nh * Nq;
  const float scale = static_cast<float>(softmax_scale);

  auto opts_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(Q.device());
  auto opts_f32 =
      torch::TensorOptions().dtype(torch::kFloat32).device(Q.device());
  torch::Tensor q4 = torch::empty({Nb, Nh, Nq_pad, kHeadDim / 2}, opts_u8);
  torch::Tensor sfq = torch::empty({Nb, Nh, Nq_pad, kHeadDim / 16}, opts_u8);
  torch::Tensor k4 = torch::empty({Nb, Nh_kv, Nkv_pad, kHeadDim / 2}, opts_u8);
  torch::Tensor sfk =
      torch::empty({Nb, Nh_kv, Nkv_pad, kHeadDim / 16}, opts_u8);
  torch::Tensor vt4 = torch::empty({Nb, Nh_kv, kHeadDim, Nkv_pad / 2}, opts_u8);
  torch::Tensor sfvt =
      torch::empty({Nb, Nh_kv, kHeadDim, Nkv_pad / 16}, opts_u8);
  torch::Tensor qm = torch::empty({Nb, Nh, Mb_qm, kHeadDim}, opts_f32);
  torch::Tensor delta_s;

  const c10::cuda::OptionalCUDAGuard device_guard(Q.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  if (fp4_hadamard) {
    // WHT pre-rotation needs BHND-packed rows; materialize non-packed.
    if (!Q.is_contiguous())
      Q = Q.contiguous();
    if (!K.is_contiguous())
      K = K.contiguous();
    Q = ffpa::apply_wht_qk_sm120<kDataType, kHeadDim>(Q);
    K = ffpa::apply_wht_qk_sm120<kDataType, kHeadDim>(K);
  }
  const kDataType* k_ptr = reinterpret_cast<const kDataType*>(K.data_ptr());
  // NHD (BNHD) permute views — including strided fused-QKV chunk rows —
  // are consumed natively by the pre-kernels: kv-mean/delta_s address rows
  // through the relaxed Lkv/Lv, the tensor-based quantize kernels take the
  // tensors' native strides.
  const ffpa_fp8::Fp8InputLayout Lkv =
      ffpa_layout_of(K, Nkv, K.size(3), /*allow_strided_rows=*/true);
  const ffpa_fp8::Fp8InputLayout Lv =
      ffpa_layout_of(V, Nkv, V.size(3), /*allow_strided_rows=*/true);

  torch::Tensor km_h = torch::empty({Nb * Nh_kv, kHeadDim}, K.options());
  torch::Tensor km_f32 = torch::empty({Nb * Nh_kv, kHeadDim}, opts_f32);
  {
    const int mean_chunks =
        (Nkv + ffpa_fp8::kMeanRowsPerChunk - 1) / ffpa_fp8::kMeanRowsPerChunk;
    torch::Tensor km_partials =
        torch::empty({Nb * Nh_kv, mean_chunks, kHeadDim}, opts_f32);
    ffpa_fp8::launch_kv_mean_sm120<kDataType, kHeadDim>(
        k_ptr, reinterpret_cast<kDataType*>(km_h.data_ptr()),
        km_f32.data_ptr<float>(), km_partials.data_ptr<float>(), Nb, Nh_kv, Nkv,
        static_cast<int>(K.size(3)), stream, &Lkv);
  }

  auto Q_t = Q.transpose(1, 2);
  auto K_t = K.transpose(1, 2);
  auto V_t = V.transpose(1, 2);
  ffpa_fp4::launch_fp4_q_block_mean_sm120<kHeadDim>(Q_t, qm);
  ffpa_fp4::launch_fp4_quant_q_sm120<kHeadDim>(Q_t, q4, sfq, qm, Nq_pad,
                                               /*sub_qm=*/true);
  ffpa_fp4::launch_fp4_quant_k_sm120<kHeadDim>(
      K_t, k4, sfk, km_f32.view({Nb, Nh_kv, kHeadDim}), Nkv_pad,
      /*sub_km=*/true);
  // smooth_v: per-(b,hkv) V column mean (same chain as split-D/persist-D).
  torch::Tensor vm_v;
  if (fp4_smooth_v) {
    vm_v = torch::empty({Nb, Nh_kv, kHeadDim}, opts_f32);
    const int v_mean_chunks =
        (Nkv + ffpa_fp8::kMeanRowsPerChunk - 1) / ffpa_fp8::kMeanRowsPerChunk;
    torch::Tensor vm_partials =
        torch::empty({Nb * Nh_kv, v_mean_chunks, kHeadDim}, opts_f32);
    ffpa_fp8::launch_kv_mean_sm120<kDataType, kHeadDim>(
        reinterpret_cast<const kDataType*>(V.data_ptr()), nullptr,
        vm_v.data_ptr<float>(), vm_partials.data_ptr<float>(), Nb, Nh_kv, Nkv,
        static_cast<int>(V.size(3)), stream, &Lv);
  }
  ffpa_fp4::launch_fp4_quant_vt_sm120<kHeadDim>(V_t, vt4, sfvt, Nkv_pad, vm_v);

  {
    auto qm_h = qm.to(Q.dtype());
    auto qkm = torch::matmul(qm_h.view({Nb, Nh_kv, group, Mb_qm, kHeadDim}),
                             km_h.view({Nb, Nh_kv, 1, kHeadDim, 1}))
                   .reshape({Nb, Nh, Mb_qm});
    delta_s = torch::empty({Nb, Nh, Mb_qm, Nkv_pad}, opts_f32);
    ffpa_fp4::launch_fp4_delta_s_sm120<kDataType, kHeadDim>(
        reinterpret_cast<const kDataType*>(qm_h.data_ptr()), k_ptr,
        reinterpret_cast<const kDataType*>(qkm.data_ptr()),
        delta_s.data_ptr<float>(), Nb, Nh, Nh_kv, Mb_qm, Nkv, Nkv_pad,
        static_cast<int>(K.size(3)), stream, &Lkv);
  }

  const long total_q_pad = (long)Nb * Nh * Nq_pad;
  const long total_kv_pad = (long)Nb * Nh_kv * Nkv_pad;
  const long d_total = (long)Nb * Nh_kv * kHeadDim;
  auto gQ =
      make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(q4.data_ptr())),
                  make_shape(total_q_pad, Int<kHeadDim>{}),
                  make_stride(Int<kHeadDim>{}, _1{}));
  auto tma_q = make_tma_copy(SM90_TMA_LOAD{}, gQ, SmemLayoutQ{},
                             Shape<Int<kBr>, Int<kHeadDim>>{}, _1{});
  auto gK =
      make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(k4.data_ptr())),
                  make_shape(total_kv_pad, Int<kHeadDim>{}),
                  make_stride(Int<kHeadDim>{}, _1{}));
  auto tma_k = make_tma_copy(SM90_TMA_LOAD{}, gK, SmemLayoutK{}(_, _, _0{}),
                             Shape<Int<kBc>, Int<kQKDChunk>>{}, _1{});
  auto gV =
      make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(vt4.data_ptr())),
                  make_shape(d_total, Nkv_pad), make_stride(Nkv_pad, _1{}));
  auto tma_v = make_tma_copy(SM90_TMA_LOAD{}, gV, SmemLayoutVt{}(_, _, _0{}),
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

  auto layout_SFQ = BlkScaledConfig::tile_atom_to_shape_SFQKV(
      make_shape(Nq_pad, Int<kHeadDim>{}, Nh, Nb));
  auto mSFQ = make_tensor(
      make_gmem_ptr(reinterpret_cast<ElementSF*>(sfq.data_ptr())), layout_SFQ);
  auto tma_sfq =
      make_tma_copy<uint16_t>(SM90_TMA_LOAD{}, mSFQ, SmemLayoutSFQ{},
                              Shape<Int<kBr>, Int<kHeadDim>>{}, _1{});
  auto layout_SFK = BlkScaledConfig::tile_atom_to_shape_SFQKV(
      make_shape(Nkv_pad, Int<kHeadDim>{}, Nh_kv, Nb));
  auto mSFK = make_tensor(
      make_gmem_ptr(reinterpret_cast<ElementSF*>(sfk.data_ptr())), layout_SFK);
  auto tma_sfk = make_tma_copy<uint16_t>(
      SM90_TMA_LOAD{}, mSFK, SmemLayoutSFK{}(_, _, _0{}),
      Shape<Int<kBc>, Int<kQKDChunk>>{}, _1{});
  auto layout_SFVt = BlkScaledConfig::tile_atom_to_shape_SFVt(
      make_shape(Int<kHeadDim>{}, Nkv_pad, Nh_kv, Nb));
  auto mSFVt =
      make_tensor(make_gmem_ptr(reinterpret_cast<ElementSF*>(sfvt.data_ptr())),
                  layout_SFVt);
  auto tma_sfvt = make_tma_copy<uint16_t>(
      SM90_TMA_LOAD{}, mSFVt, SmemLayoutSFVt{}(_, _, _0{}),
      Shape<Int<kVDChunk>, Int<kBc>>{}, _1{});
  auto layout_DS =
      tile_to_shape(SmemLayoutAtomDS{}, make_shape(Mb_qm, Nkv_pad, Nh, Nb),
                    Step<_2, _1, _3, _4>{});
  auto mDS = make_tensor(make_gmem_ptr(delta_s.data_ptr<float>()), layout_DS);
  auto tma_ds = make_tma_copy(SM90_TMA_LOAD{}, mDS, SmemLayoutDS{}(_, _, _0{}),
                              Shape<_1, Int<kBc>>{}, _1{});

  constexpr int kSmemBytes = Traits::kSmemBytes;
  TORCH_CHECK(q_start_row >= 0 && q_start_row < Nq,
              "ffpa_attn: q_start_row must be in [0, Nq)");
  TORCH_CHECK(q_start_row % kBr == 0,
              "ffpa_attn: q_start_row must be a multiple of kBr=64");
  int max_smem_optin = 0;
  cudaDeviceGetAttribute(
      &max_smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, Q.get_device());
  // PC-0-5 bias tile plan: single-source helper (the wrapper dispatch
  // calls it too; it owns the PC-0-5 mode-0 pin and the FFPA_BIAS_TILE_KEEP
  // debug switch); the tags below must match or the CHECK fires.
  const int dyn_limit = max_smem_optin - 256;
  FfpaBiasTilePlan bias_plan;
  if (bias.ptr != nullptr) {
    FfpaBiasParams bias_p{bias.ptr,      bias.dtype,    bias.stride_b,
                          bias.stride_h, bias.stride_m, bias.stride_n};
    bias_plan = ffpa::fp4_m4n2_bias_plan<kDataType, kHeadDim>(
        bias_p, Nb, Nh, Nq, Nkv, dyn_limit);
  }
  TORCH_CHECK(kBiasOn == bias_on &&
                  kBiasPlanMode == (bias_on ? bias_plan.mode : 0) &&
                  // b4 only picks the mode-2 TMA smem width; the mode-3
                  // resident fill reads the runtime attn_bias_dtype in-kernel.
                  kBias4BytesPerElem ==
                      ((kBiasPlanMode == 2 && bias.dtype == 3) ? 1 : 0),
              "ffpa_attn: fp4 split_d m4n2 D=", kHeadDim,
              " variant tag mismatch (wrapper dispatch vs plan)");
  const auto bias_bytes_of = [&](int m) {
    return (m == 3)
               ? (long long)(Nkv + kBc - 1) / kBc * kBc * bias_plan.elem_size
               : bias_plan.tile_bytes(kBr, kBc, (m == 2) ? 2 : 1);
  };
  const int kSmemBytesBias = (int)(((long long)kSmemBytes + 15) & ~15) +
                             (int)bias_bytes_of(bias_plan.mode);
  TORCH_CHECK(kSmemBytesBias <= dyn_limit,
              "ffpa_attn: fp4 split_d m4n2 D=", kHeadDim, " needs ",
              kSmemBytesBias, "B smem, device opt-in allows ", dyn_limit,
              " (static reserved 256B)");
  const auto make_tma_bias = [&](auto b4_c) {
    constexpr int kBias4B = decltype(b4_c)::value;
    constexpr int bias_cols = kBc * (kBias4B ? 2 : 1);
    // Row-broadcast plane is the real [m_total, Nkv]; demoted/dummy cases
    // (mode 0, or mode 3 which never issues) keep a 1-row plane where
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
  [[maybe_unused]] auto tma_bias_r16 =
      make_tma_bias(std::integral_constant<int, 0>{});
  [[maybe_unused]] auto tma_bias_r32 =
      make_tma_bias(std::integral_constant<int, 1>{});
  TORCH_CHECK(kSmemBytes <= max_smem_optin,
              "ffpa_attn: fp4 split_d m4n2 D=", kHeadDim, " needs ", kSmemBytes,
              "B smem, device opt-in allows ", max_smem_optin);
  float* softmax_lse_ptr =
      softmax_lse.numel() > 0 ? softmax_lse.data_ptr<float>() : nullptr;
  auto O_ptr = reinterpret_cast<ElementO*>(O.data_ptr());
  const dim3 block(kNumThreads, 1, 1);
  const int mb = (Nq - q_start_row + kBr - 1) / kBr;
  const int total_work = mb * Nb * Nh;
  const int num_ctas =
      causal ? total_work : std::min(total_work, prop->multiProcessorCount);
  const dim3 grid(num_ctas, 1, 1);
  const auto launch_with = [&](auto bias_tag, auto tma_bias_sel, auto mode_c,
                               auto b4_c) {
    auto kernel = ffpa_fp4::split_d_m4n2_fwd_cute_fp4_sm120<
        Traits, ElementO, decltype(tma_q), decltype(tma_k), decltype(tma_v),
        decltype(tma_o), decltype(tma_sfq), decltype(tma_sfk),
        decltype(tma_sfvt), decltype(tma_ds), decltype(tma_bias_sel),
        kBiasPlanMode, kBias4BytesPerElem, kBiasOn>;
    TORCH_CHECK(
        cudaFuncSetAttribute(kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             kSmemBytesBias) == cudaSuccess,
        "ffpa_attn: fp4 split_d m4n2 smem opt-in failed for D=", kHeadDim);
    kernel<<<grid, block, kSmemBytesBias, stream>>>(
        tma_q, tma_k, tma_v, tma_o, tma_sfq, tma_sfk, tma_sfvt, tma_ds,
        tma_bias_sel, O_ptr, softmax_lse_ptr, km_f32.data_ptr<float>(),
        qm.data_ptr<float>(), fp4_smooth_v ? vm_v.data_ptr<float>() : nullptr,
        Nq, Nkv, Nq_pad, Nkv_pad, Nh, Nh_kv, scale, Tc, causal, total_q_rows,
        Nb, q_start_row, nhd_out, bias.ptr, bias.dtype, bias.stride_b,
        bias.stride_h, bias.stride_m, bias.stride_n,
        bias_plan.mode != 0 ? bias_plan.m_total : (long long)1);
  };
  // Compile-time pinned variant: regular builds pin mode 0 (PC-0-5, see
  // fp4_m4n2_bias_plan); only the debug KEEP switch reaches mode 2, so the
  // m2 tags compile solely under ENABLE_FFPA_FP4_BUILD_DEBUG.
  if constexpr (kBiasOn == 0) {
    launch_with(std::integral_constant<int, 0>{}, tma_bias_r16,
                std::integral_constant<int, 0>{},
                std::integral_constant<int, 0>{});
  } else {
#ifdef ENABLE_FFPA_FP4_BUILD_DEBUG
    if constexpr (kBiasPlanMode == 2) {
      if constexpr (kBias4BytesPerElem == 1)
        launch_with(std::integral_constant<int, 1>{}, tma_bias_r32,
                    std::integral_constant<int, 2>{},
                    std::integral_constant<int, 1>{});
      else
        launch_with(std::integral_constant<int, 1>{}, tma_bias_r16,
                    std::integral_constant<int, 2>{},
                    std::integral_constant<int, 0>{});
    } else
#endif
      launch_with(std::integral_constant<int, 1>{}, tma_bias_r16,
                  std::integral_constant<int, 0>{},
                  std::integral_constant<int, 0>{});
  }
}

#endif  // ENABLE_FFPA_CUTE_EXT && ENABLE_FFPA_TMA_EXT

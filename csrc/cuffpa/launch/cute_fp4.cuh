#pragma once
// CuTe fp4 family launchers (persist-D / split-D / split-D M4N2 with
// their quantize/delta_s/hadamard/kv-mean pre-kernel orchestration),
// moved verbatim out of the old cute/launch.cuh.
#include "launch/common.cuh"
#if defined(ENABLE_FFPA_CUTE_EXT) && defined(ENABLE_FFPA_TMA_EXT)
#include "cute/fp8/smooth_k.cuh"
#include "cute/fp4/quantize_fp4.cuh"
#include "cute/hadamard.cuh"
#include "cute/fp4/delta_s.cuh"
#include "generated/fwd_cute_fp4_preprocess.cuh"  // extern templates
#include "cute/fp4/sm_120/persist_d.cuh"
#include "cute/fp4/sm_120/split_d.cuh"
#include "cute/fp4/sm_120/split_d_m4n2.cuh"

// NVFP4 persist-D launcher (D=128). Pipeline: km (two-stage K column mean,
// shared with fp8) -> q_block_mean -> 3 quantize kernels (Q centered by qm,
// K smoothed by km + row-permuted, V transposed) -> delta_s = qm @ (K-km)^T
// - qm.km (GQA broadcast bmm, fp16 domain like sageattn3) -> TMA descriptors
// -> persist_d_ws_fwd_cute_fp4_sm120. Workspaces are 128-padded along
// seqlen; delta_s tail columns zero-fill (masked -inf in-kernel).
template <typename kDataType, const int kHeadDim, bool kPvMxfp8 = false>
void launch_cute_fwd_persist_d_fp4_sm120_impl(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
    torch::Tensor attn_bias, torch::Tensor softmax_lse, int causal,
    double softmax_scale, int q_start_row = 0, bool fp4_hadamard = false,
    bool fp4_smooth_v = false) {
  using namespace cute;
  constexpr int kBr = 128;
  constexpr int kBc = 128;
  constexpr int kNumThreads = 384;

  using ElementO = std::conditional_t<std::is_same_v<kDataType, __half>,
                                      cutlass::half_t, cutlass::bfloat16_t>;
  using Traits =
      ffpa_fp4::FFPAAttnCuTePersistDFP4Traits<ElementO, kHeadDim, kPvMxfp8>;
  const FfpaBiasParams bias = ffpa_bias_params_of(attn_bias, Q, K);
  const int bias_on = bias.ptr != nullptr ? 1 : 0;
  using Element = typename Traits::Element;
  using ElementSF = typename Traits::ElementSF;
  using ElementPV = typename Traits::ElementPV;
  using ElementSFV = typename Traits::ElementSFV;
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
  using BlkScaledConfigV = typename Traits::BlkScaledConfigV;

  const int Nb = Q.size(0);
  const int Nh = Q.size(1);
  const int Nh_kv = K.size(1);
  const int Nq = Q.size(2);
  const int Nkv = K.size(2);
  const int group = Nh / Nh_kv;
  const int Nq_pad = utils::div_ceil(Nq, kBr) * kBr;
  const int Nkv_pad = utils::div_ceil(Nkv, kBc) * kBc;
  const int Mb = Nq_pad / kBr;
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
  torch::Tensor vt4 = torch::empty(
      {Nb, Nh_kv, kHeadDim, Nkv_pad / (kPvMxfp8 ? 1 : 2)}, opts_u8);
  torch::Tensor sfvt = torch::empty(
      {Nb, Nh_kv, kHeadDim, Nkv_pad / (kPvMxfp8 ? 32 : 16)}, opts_u8);
  torch::Tensor qm = torch::empty({Nb, Nh, Mb, kHeadDim}, opts_f32);
  torch::Tensor delta_s;

  const c10::cuda::OptionalCUDAGuard device_guard(Q.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  // fp4_hadamard: rotate Q/K before the preprocessing chain. Exact in fp32
  // math (H orthogonal); only moves where quantization noise lands. The
  // rotated copies are kHeadDim-wide (rotated zero pad cols stored), so all
  // downstream consumers stay in one rotated domain; V and the hybrid
  // stage-1 (fp16, earlier in dispatch) are untouched.
  // Fused-hadamard path (pow2 D): rows are rotated inside the quantize
  // kernel; Q/K stay unrotated, mean/delta_s run in the unrotated domain
  // (WHT is linear, H H^T = I), and the attention kernel gets WHT-pre-
  // rotated qm/km copies for its lse correction. Non-pow2 D keeps the
  // standalone pre-rotation kernels.
  constexpr bool kFuseWht = (kHeadDim & (kHeadDim - 1)) == 0 && kHeadDim <= 512;
  const bool fused_wht = fp4_hadamard && kFuseWht;
  torch::Tensor km_rot_f32, qm_rot;
  if (fp4_hadamard && !fused_wht) {
    // WHT pre-rotation needs BHND-packed rows; materialize NHD-family views.
    if (!Q.is_contiguous())
      Q = Q.contiguous();
    if (!K.is_contiguous())
      K = K.contiguous();
    Q = ffpa::apply_wht_qk_sm120<kDataType, kHeadDim>(Q);
    K = ffpa::apply_wht_qk_sm120<kDataType, kHeadDim>(K);
  }
  const kDataType* k_ptr = reinterpret_cast<const kDataType*>(K.data_ptr());
  // NHD (BNHD) permute views — including strided fused-QKV chunk rows —
  // are consumed natively by the pre-kernels.
  const ffpa_fp8::Fp8InputLayout Lkv =
      ffpa_layout_of(K, Nkv, K.size(3), /*allow_strided_rows=*/true);
  const ffpa_fp8::Fp8InputLayout Lv =
      ffpa_layout_of(V, Nkv, V.size(3), /*allow_strided_rows=*/true);

  // Quantize kernels take (B,S,H,D)-strided inputs; pass the (B,H,N,D)
  // tensors as (B,N,H,D) views (strides only, no copy).
  auto Q_t = Q.transpose(1, 2);
  auto K_t = K.transpose(1, 2);
  auto V_t = V.transpose(1, 2);

  // D <= 128 quantizes Q/K/V in a single launch that also emits qm (fp32
  // + in-dtype) inside the Q-tile blocks, replacing q_mean + the three
  // quantize launches + the qm cast (and the qm_rot WHT under hadamard).
  // Larger head dims keep the separate quantize chain below.
  constexpr bool fused_qkv = kHeadDim <= 128;
  // if constexpr: keeps the fused launcher (static_assert D <= 128, pow2
  // WHT) from being instantiated for larger head dims.
  if constexpr (!fused_qkv) {
    // [smooth Q - always on, mandatory for fp4 accuracy] per-128-row-block
    // Q mean: quantize bias (sub_qm) + the rank-1 delta_s/qkm terms.
    // Hoisted above the K chain so the qkm dot below only waits on the
    // small km kernels; the launch order targets L2 reuse (see the
    // quantize section).
    ffpa_fp4::launch_fp4_q_block_mean_sm120<kHeadDim>(Q_t, qm);
    if (fused_wht)
      qm_rot = ffpa::apply_wht_f32_rows_sm120<kHeadDim>(qm);
  }

  // [smooth K - always on, mandatory for fp4 accuracy] per-(b,hkv) K column
  // mean: km_h/km_f32. Consumed as the quantize bias (sub_km) and by the
  // lse correction; delta_s restores the exact scores (see the kernel
  // header). Shared two-stage kernel from the fp8 path.
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
  if (fused_wht)
    km_rot_f32 = ffpa::apply_wht_f32_rows_sm120<kHeadDim>(km_f32);

  torch::Tensor vm_v;
  // Launch order targets L2 reuse (96MB L2 vs ~67MB per input tensor): the
  // K chain km -> qkm -> K-quant -> delta_s keeps all four K touches back
  // to back, vm -> V-quant pairs the V reads, and Q-quant (the 2nd Q read)
  // runs last, where its L2 copy has aged out anyway. The fused path gets
  // the same locality for free: vm/km run right before the single launch
  // that re-reads all three tensors.
  torch::Tensor qm_h;
  if constexpr (fused_qkv) {
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
    qm_h = torch::empty({Nb, Nh, Mb, kHeadDim}, Q.options());
    if (fused_wht)
      qm_rot = torch::empty({Nb, Nh, Mb, kHeadDim}, opts_f32);
    ffpa_fp4::launch_fp4_quant_qkv_fused_sm120<kHeadDim>(
        Q_t, q4, sfq, qm, qm_h, qm_rot,
        fused_wht ? km_rot_f32.view({Nb, Nh_kv, kHeadDim})
                  : km_f32.view({Nb, Nh_kv, kHeadDim}),
        K_t, k4, sfk, vm_v, V_t, vt4, sfvt, Nq_pad, Nkv_pad,
        /*hadamard=*/fused_wht, /*pv_mxfp8=*/kPvMxfp8);
  } else {
    qm_h = qm.to(Q.dtype());
    if (fused_wht) {
      // pow2-only WHT-fused variant; fused_wht is false for non-pow2 D
      // (standalone pre-rotated path), keep it uninstantiated there.
      if constexpr (kFuseWht)
        ffpa_fp4::launch_fp4_quant_k_wht_sm120<kHeadDim>(
            K_t, k4, sfk, km_rot_f32.view({Nb, Nh_kv, kHeadDim}), Nkv_pad);
    } else {
      ffpa_fp4::launch_fp4_quant_k_sm120<kHeadDim>(
          K_t, k4, sfk, km_f32.view({Nb, Nh_kv, kHeadDim}), Nkv_pad,
          /*sub_km=*/true);
    }
  }
  auto qkm = torch::matmul(qm_h.view({Nb, Nh_kv, group, Mb, kHeadDim}),
                           km_h.view({Nb, Nh_kv, 1, kHeadDim, 1}))
                 .reshape({Nb, Nh, Mb});

  // delta_s per 128-row Q block via the identity qm@(K-km)^T ==
  // qm@K^T - qm.km^T, fused in one wmma kernel (fp32 out, tail columns
  // zero-filled). GQA broadcasts the shared K heads.
  delta_s = torch::empty({Nb, Nh, Mb, Nkv_pad}, opts_f32);
  ffpa_fp4::launch_fp4_delta_s_sm120<kDataType, kHeadDim>(
      reinterpret_cast<const kDataType*>(qm_h.data_ptr()), k_ptr,
      reinterpret_cast<const kDataType*>(qkm.data_ptr()),
      delta_s.data_ptr<float>(), Nb, Nh, Nh_kv, Mb, Nkv, Nkv_pad,
      static_cast<int>(K.size(3)), stream, &Lkv);

  if constexpr (!fused_qkv) {
    // smooth_v: per-(b,hkv) V column mean. The attention kernel computes,
    // in exact math,
    //   O_i = sum_j P_ij V_j / sum_j P_ij,
    // and with V_j = Vhat_j + vm (vm constant per (b,hkv,d) column),
    //   O_i = [sum_j P_ij Vhat_j / sum_j P_ij] + vm,
    // so quantizing the residual Vhat and adding vm back in the epilogue is
    // exactly equivalent while shrinking the quantized dynamic range. The
    // chain: launch_kv_mean_sm120 (generic column-mean kernels shared with
    // the fp8 smooth_k path; km=nullptr skips the in-dtype copy) -> subtract
    // inside the V^T quantize kernel -> epilogue add-back after the softmax
    // normalize. Contrast with K-smoothing: K-mean subtraction changes the
    // scores, which stays exact only through softmax shift invariance plus
    // the delta_s/lse corrections; V-mean subtraction never touches the
    // scores at all.
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
    if constexpr (kPvMxfp8)
      ffpa_fp4::launch_mxfp8_quant_vt_sm120<kHeadDim>(V_t, vt4, sfvt, Nkv_pad,
                                                      vm_v);
    else
      ffpa_fp4::launch_fp4_quant_vt_sm120<kHeadDim>(V_t, vt4, sfvt, Nkv_pad,
                                                    vm_v);

    if (fused_wht) {
      // pow2-only WHT-fused variant (see the K quantize site above).
      if constexpr (kFuseWht)
        ffpa_fp4::launch_fp4_quant_q_wht_sm120<kHeadDim>(Q_t, q4, sfq, qm_rot,
                                                         Nq_pad);
    } else {
      ffpa_fp4::launch_fp4_quant_q_sm120<kHeadDim>(Q_t, q4, sfq, qm, Nq_pad,
                                                   /*sub_qm=*/true);
    }
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
                             Shape<Int<kBc>, Int<kHeadDim>>{}, _1{});
  auto gV =
      make_tensor(make_gmem_ptr(reinterpret_cast<ElementPV*>(vt4.data_ptr())),
                  make_shape(d_total, Nkv_pad), make_stride(Nkv_pad, _1{}));
  auto tma_v = make_tma_copy(SM90_TMA_LOAD{}, gV, SmemLayoutVt{}(_, _, _0{}),
                             Shape<Int<kHeadDim>, Int<kBc>>{}, _1{});
  // BHND-packed O is flat [total_q_rows, D] with the per-(batch,head)
  // origin injected via domain_offset in the kernel; NHD (diffusers BNHD
  // packed) O, detected by storage, is flat [Nb*Nq, Nh*kHeadDim] with the
  // head selecting the column tile. Both branches use dynamic int64
  // extents/strides so TmaO has a single type and the kernel takes a
  // runtime nhd_out branch.
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
      Shape<Int<kBc>, Int<kHeadDim>>{}, _1{});
  auto layout_SFVt = BlkScaledConfigV::tile_atom_to_shape_SFVt(
      make_shape(Int<kHeadDim>{}, Nkv_pad, Nh_kv, Nb));
  auto mSFVt =
      make_tensor(make_gmem_ptr(reinterpret_cast<ElementSFV*>(sfvt.data_ptr())),
                  layout_SFVt);
  auto tma_sfvt = make_tma_copy<uint16_t>(
      SM90_TMA_LOAD{}, mSFVt, SmemLayoutSFVt{}(_, _, _0{}),
      Shape<Int<kHeadDim>, Int<kBc>>{}, _1{});
  auto layout_DS =
      tile_to_shape(SmemLayoutAtomDS{}, make_shape(Nq_pad, Nkv_pad, Nh, Nb),
                    Step<_2, _1, _3, _4>{});
  auto mDS = make_tensor(make_gmem_ptr(delta_s.data_ptr<float>()), layout_DS);
  auto tma_ds = make_tma_copy(SM90_TMA_LOAD{}, mDS, SmemLayoutDS{}(_, _, _0{}),
                              Shape<Int<kBr>, Int<kBc>>{}, _1{});

  // PC-0-1 bias tile plan (tail slack past SFVt, mirroring the fp8 m4n2
  // launcher): dense tiles get a single buffer; row-broadcast double
  // buffers and upgrades to the resident vector (mode 3) when the whole
  // [1,Nkv] row fits AND the smem-driven blocks/SM of the base layout stay
  // intact (this family is 1 CTA/SM, so the guard only rejects the tight
  // D=256 corners).
  FfpaBiasTilePlan bias_plan;
  if (bias.ptr != nullptr) {
    FfpaBiasParams bias_p{bias.ptr,      bias.dtype,    bias.stride_b,
                          bias.stride_h, bias.stride_m, bias.stride_n};
    bias_plan = ffpa_bias_tile_plan_of(bias_p, Nb, Nh, Nq, Nkv);
  }
  TORCH_CHECK(q_start_row >= 0 && q_start_row < Nq,
              "ffpa_attn: q_start_row must be in [0, Nq)");
  TORCH_CHECK(q_start_row % kBr == 0,
              "ffpa_attn: q_start_row must be a multiple of kBr=128");
  // Exceeding the smem opt-in limit fails SILENTLY in cudaFuncSetAttribute
  // (score collapses to zero); check both up front (see sm120-smem-limit).
  int max_smem_optin = 0;
  cudaDeviceGetAttribute(
      &max_smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, Q.get_device());
  // The kernel's STATIC smem (the barrier arrays, <= 144B incl. the bias
  // pair) is invisible to the dynamic budget: D=256's base sits ~200B under
  // the opt-in ceiling and the aligned bias tail pushed the ATTRIBUTE over
  // (the dynamic pre-check still passed). Reserve 256B for it.
  const int dyn_limit = max_smem_optin - 256;
  const int bias_stages = (bias_plan.mode == 2) ? 2 : 1;
  if ((long long)Traits::kSmemBytes +
          bias_plan.tile_bytes(kBr, kBc, bias_stages) >
      dyn_limit)
    bias_plan.mode = 0;
  if (bias_plan.mode == 2) {
    const long long kv_pad = (Nkv + kBc - 1) / kBc * kBc;
    const long long base_align = ((long long)Traits::kSmemBytes + 15) & ~15;
    const long long resident = base_align + kv_pad * bias_plan.elem_size;
    if (resident <= dyn_limit && dyn_limit / resident >= dyn_limit / base_align)
      bias_plan.mode = 3;
  }
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
  auto tma_bias_d16 = make_tma_bias(std::integral_constant<int, 1>{},
                                    std::integral_constant<int, 0>{});
  auto tma_bias_d32 = make_tma_bias(std::integral_constant<int, 1>{},
                                    std::integral_constant<int, 1>{});
  auto tma_bias_r16 = make_tma_bias(std::integral_constant<int, 2>{},
                                    std::integral_constant<int, 0>{});
  auto tma_bias_r32 = make_tma_bias(std::integral_constant<int, 2>{},
                                    std::integral_constant<int, 1>{});
  const int kSmemBytes =
      (int)(((long long)Traits::kSmemBytes + 15) & ~15) +
      (int)((bias_plan.mode == 3)
                ? ((long long)(Nkv + kBc - 1) / kBc * kBc) * bias_plan.elem_size
                : bias_plan.tile_bytes(kBr, kBc, bias_stages));
  TORCH_CHECK(kSmemBytes <= dyn_limit, "ffpa_attn: fp4 persist_d D=", kHeadDim,
              " needs ", kSmemBytes, "B smem, device opt-in allows ", dyn_limit,
              " (static reserved 256B)");
  float* softmax_lse_ptr =
      softmax_lse.numel() > 0 ? softmax_lse.data_ptr<float>() : nullptr;
  auto O_ptr = reinterpret_cast<ElementO*>(O.data_ptr());
  const dim3 block(kNumThreads, 1, 1);
  // Grid dispatch - NOT two kernel variants: the kernel's strided work loop
  // degenerates to one iteration per CTA when gridDim.x == total_work (see
  // the scheduling contract comment in fp4/sm_120/persist_d.cuh). Dense
  // works are long (Tc tiles each) and benefit from the persistent grid
  // (pipeline overlap across works); causal works average half the tiles
  // with many short ones, where the per-work epilogue_done -> Q TMA round
  // trip dominates, so give each work its own CTA and let the HW scheduler
  // load-balance instead.
  const int mb = (Nq - q_start_row + kBr - 1) / kBr;
  const int total_work = mb * Nb * Nh;
  const int num_ctas =
      causal ? total_work : std::min(total_work, prop->multiProcessorCount);
  const dim3 grid(num_ctas, 1, 1);
  const auto launch_with = [&](auto bias_tag, auto tma_bias_sel, auto mode_c,
                               auto b4_c) {
    constexpr int kBiasOn = decltype(bias_tag)::value;
    constexpr int kModeL = decltype(mode_c)::value;
    constexpr int kB4 = decltype(b4_c)::value;
    using TmaBiasSel = decltype(tma_bias_sel);
    auto kernel = ffpa_fp4::persist_d_ws_fwd_cute_fp4_sm120<
        Traits, ElementO, decltype(tma_q), decltype(tma_k), decltype(tma_v),
        decltype(tma_o), decltype(tma_sfq), decltype(tma_sfk),
        decltype(tma_sfvt), decltype(tma_ds), TmaBiasSel, kModeL, kB4, kBiasOn>;
    TORCH_CHECK(cudaFuncSetAttribute(
                    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                    kSmemBytes) == cudaSuccess,
                "ffpa_attn: fp4 persist_d smem opt-in failed for D=", kHeadDim);
    kernel<<<grid, block, kSmemBytes, stream>>>(
        tma_q, tma_k, tma_v, tma_o, tma_sfq, tma_sfk, tma_sfvt, tma_ds,
        tma_bias_sel, O_ptr, softmax_lse_ptr,
        fused_wht ? km_rot_f32.data_ptr<float>() : km_f32.data_ptr<float>(),
        fused_wht ? qm_rot.data_ptr<float>() : qm.data_ptr<float>(),
        fp4_smooth_v ? vm_v.data_ptr<float>() : nullptr, Nq, Nkv, Nq_pad,
        Nkv_pad, Nh, Nh_kv, scale, Tc, causal, total_q_rows, Nb, q_start_row,
        nhd_out, bias.ptr, bias.dtype, bias.stride_b, bias.stride_h,
        bias.stride_m, bias.stride_n,
        bias_plan.mode != 0 ? bias_plan.m_total : (long long)1);
  };
  if (bias.ptr == nullptr) {
    launch_with(std::integral_constant<int, 0>{}, tma_bias_r16,
                std::integral_constant<int, 0>{},
                std::integral_constant<int, 0>{});
  } else {
    const auto bias_on = std::integral_constant<int, 1>{};
    const auto m0 = std::integral_constant<int, 0>{};
    const auto m1 = std::integral_constant<int, 1>{};
    const auto m2 = std::integral_constant<int, 2>{};
    const auto m3 = std::integral_constant<int, 3>{};
    if (bias_plan.mode == 1) {
      if (bias.dtype == 3)
        launch_with(bias_on, tma_bias_d32, m1, m1);
      else
        launch_with(bias_on, tma_bias_d16, m1, m0);
    } else if (bias_plan.mode == 2) {
      if (bias.dtype == 3)
        launch_with(bias_on, tma_bias_r32, m2, m1);
      else
        launch_with(bias_on, tma_bias_r16, m2, m0);
    } else if (bias_plan.mode == 3) {
      // resident row-vector: no TMA issue in-kernel, descriptor unused.
      if (bias.dtype == 3)
        launch_with(bias_on, tma_bias_r32, m3, m1);
      else
        launch_with(bias_on, tma_bias_r16, m3, m0);
    } else {
      launch_with(bias_on, tma_bias_r16, m0, m0);
    }
  }
}

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
    if (pv_fp8) {
      // mxfp8-PV traits static_assert D <= 192; the TORCH_CHECK above
      // already rejected the request, keep the variant uninstantiated.
      if constexpr (kHeadDim <= 192)
        launch_cute_fwd_persist_d_fp4_sm120_impl<kDataType, kHeadDim, true>(
            Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
            q_start_row, fp4_hadamard, fp4_smooth_v);
    } else {
      launch_cute_fwd_persist_d_fp4_sm120_impl<kDataType, kHeadDim, false>(
          Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
          q_start_row, fp4_hadamard, fp4_smooth_v);
    }
  } else {
    TORCH_CHECK(false,
                "ffpa_attn: cute_tma_fp4 persist_d requires D in "
                "{64,128,192,256} (64-multiples), got D=",
                kHeadDim);
  }
}

// NVFP4 split-D launcher, headdims in (256, 768). Same pre-kernel pipeline
// as the persist-D fp4 launcher (km -> q_block_mean -> quantize -> delta_s);
// only the TMA descriptors change shape: K/SFK tiles become [kBc, 64] D
// chunks, V^T/SFVt [64, kBc], and O stores per [kBr, kVDChunk] chunk. The
// kernel itself stays persistent (same grid contract as persist_d fp4).
// kPvMxfp8 switches the PV side to MXFP8 (e4m3 V^T + ue8m0/32 SF + the
// K=128 MXFP8 PV atom - legal here because the split-D PV Tile-K is the
// full kBc=128; the m4n2 family cannot take it, kBc=64 < 128). smooth_v
// quantizes the residual V - vm (kv-mean kernels shared with the fp8 path)
// and the kernel adds vm back in the epilogue (persist_d derivation).
template <typename kDataType, const int kHeadDim, bool kPvMxfp8 = false>
void launch_cute_fwd_split_d_fp4_sm120_impl(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
    torch::Tensor attn_bias, torch::Tensor softmax_lse, int causal,
    double softmax_scale, int q_start_row = 0, bool fp4_hadamard = false,
    bool fp4_smooth_v = false) {
  using namespace cute;
  constexpr int kBr = 128;
  constexpr int kBc = 128;
  constexpr int kNumThreads = 256;
  constexpr int kQKDChunk = 64;
  constexpr int kVDChunk = 64;

  using ElementO = std::conditional_t<std::is_same_v<kDataType, __half>,
                                      cutlass::half_t, cutlass::bfloat16_t>;
  using Traits = ffpa_fp4::FFPAAttnCuTeSplitDFP4Traits<
      ElementO, kHeadDim, kBr, kBc, kQKDChunk, kVDChunk, 3, 3, kPvMxfp8>;
  const FfpaBiasParams bias = ffpa_bias_params_of(attn_bias, Q, K);
  const int bias_on = bias.ptr != nullptr ? 1 : 0;
  using Element = typename Traits::Element;
  using ElementSF = typename Traits::ElementSF;
  using ElementPV = typename Traits::ElementPV;
  using ElementSFV = typename Traits::ElementSFV;
  using BlkScaledConfigV = typename Traits::BlkScaledConfigV;
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
  const int Nq_pad = utils::div_ceil(Nq, kBr) * kBr;
  const int Nkv_pad = utils::div_ceil(Nkv, kBc) * kBc;
  const int Mb = Nq_pad / kBr;
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
  torch::Tensor vt4 = torch::empty(
      {Nb, Nh_kv, kHeadDim, Nkv_pad / (kPvMxfp8 ? 1 : 2)}, opts_u8);
  torch::Tensor sfvt = torch::empty(
      {Nb, Nh_kv, kHeadDim, Nkv_pad / (kPvMxfp8 ? 32 : 16)}, opts_u8);
  torch::Tensor qm = torch::empty({Nb, Nh, Mb, kHeadDim}, opts_f32);
  torch::Tensor delta_s;

  const c10::cuda::OptionalCUDAGuard device_guard(Q.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  // Fused-hadamard path (pow2 D): rows are rotated inside the quantize
  // kernel; Q/K stay unrotated, mean/delta_s run in the unrotated domain
  // (WHT is linear, H H^T = I), and the attention kernel gets WHT-pre-
  // rotated qm/km copies for its lse correction. Non-pow2 D keeps the
  // standalone pre-rotation kernels below.
  constexpr bool kFuseWht = (kHeadDim & (kHeadDim - 1)) == 0 && kHeadDim <= 512;
  const bool fused_wht = fp4_hadamard && kFuseWht;
  torch::Tensor km_rot_f32, qm_rot;
  if (fp4_hadamard && !fused_wht) {
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
  // through the relaxed Lkv, the tensor-based quantize kernels take the
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
  if (fused_wht)
    km_rot_f32 = ffpa::apply_wht_f32_rows_sm120<kHeadDim>(km_f32);

  auto Q_t = Q.transpose(1, 2);
  auto K_t = K.transpose(1, 2);
  auto V_t = V.transpose(1, 2);
  ffpa_fp4::launch_fp4_q_block_mean_sm120<kHeadDim>(Q_t, qm);
  if (fused_wht) {
    // pow2-only WHT-fused variants; fused_wht is false for non-pow2 D
    // (standalone pre-rotated path), keep them uninstantiated there.
    if constexpr (kFuseWht) {
      qm_rot = ffpa::apply_wht_f32_rows_sm120<kHeadDim>(qm);
      ffpa_fp4::launch_fp4_quant_q_wht_sm120<kHeadDim>(Q_t, q4, sfq, qm_rot,
                                                       Nq_pad);
      ffpa_fp4::launch_fp4_quant_k_wht_sm120<kHeadDim>(
          K_t, k4, sfk, km_rot_f32.view({Nb, Nh_kv, kHeadDim}), Nkv_pad);
    }
  } else {
    ffpa_fp4::launch_fp4_quant_q_sm120<kHeadDim>(Q_t, q4, sfq, qm, Nq_pad,
                                                 /*sub_qm=*/true);
    ffpa_fp4::launch_fp4_quant_k_sm120<kHeadDim>(
        K_t, k4, sfk, km_f32.view({Nb, Nh_kv, kHeadDim}), Nkv_pad,
        /*sub_km=*/true);
  }
  // smooth_v: per-(b,hkv) V column mean (persist_d chain: generic column-
  // mean kernels shared with the fp8 smooth_k path; km=nullptr skips the
  // in-dtype copy) -> subtract inside the V^T quantize kernel -> epilogue
  // add-back after the softmax normalize.
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
  if constexpr (kPvMxfp8)
    ffpa_fp4::launch_mxfp8_quant_vt_sm120<kHeadDim>(V_t, vt4, sfvt, Nkv_pad,
                                                    vm_v);
  else
    ffpa_fp4::launch_fp4_quant_vt_sm120<kHeadDim>(V_t, vt4, sfvt, Nkv_pad,
                                                  vm_v);

  {
    auto qm_h = qm.to(Q.dtype());
    auto qkm = torch::matmul(qm_h.view({Nb, Nh_kv, group, Mb, kHeadDim}),
                             km_h.view({Nb, Nh_kv, 1, kHeadDim, 1}))
                   .reshape({Nb, Nh, Mb});
    delta_s = torch::empty({Nb, Nh, Mb, Nkv_pad}, opts_f32);
    ffpa_fp4::launch_fp4_delta_s_sm120<kDataType, kHeadDim>(
        reinterpret_cast<const kDataType*>(qm_h.data_ptr()), k_ptr,
        reinterpret_cast<const kDataType*>(qkm.data_ptr()),
        delta_s.data_ptr<float>(), Nb, Nh, Nh_kv, Mb, Nkv, Nkv_pad,
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
      make_tensor(make_gmem_ptr(reinterpret_cast<ElementPV*>(vt4.data_ptr())),
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
  auto layout_SFVt = BlkScaledConfigV::tile_atom_to_shape_SFVt(
      make_shape(Int<kHeadDim>{}, Nkv_pad, Nh_kv, Nb));
  auto mSFVt =
      make_tensor(make_gmem_ptr(reinterpret_cast<ElementSFV*>(sfvt.data_ptr())),
                  layout_SFVt);
  auto tma_sfvt = make_tma_copy<uint16_t>(
      SM90_TMA_LOAD{}, mSFVt, SmemLayoutSFVt{}(_, _, _0{}),
      Shape<Int<kVDChunk>, Int<kBc>>{}, _1{});
  auto layout_DS =
      tile_to_shape(SmemLayoutAtomDS{}, make_shape(Nq_pad, Nkv_pad, Nh, Nb),
                    Step<_2, _1, _3, _4>{});
  auto mDS = make_tensor(make_gmem_ptr(delta_s.data_ptr<float>()), layout_DS);
  auto tma_ds = make_tma_copy(SM90_TMA_LOAD{}, mDS, SmemLayoutDS{}(_, _, _0{}),
                              Shape<Int<kBr>, Int<kBc>>{}, _1{});

  constexpr int kSmemBytes = Traits::kSmemBytes;
  TORCH_CHECK(q_start_row >= 0 && q_start_row < Nq,
              "ffpa_attn: q_start_row must be in [0, Nq)");
  TORCH_CHECK(q_start_row % kBr == 0,
              "ffpa_attn: q_start_row must be a multiple of kBr=128");
  int max_smem_optin = 0;
  cudaDeviceGetAttribute(
      &max_smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, Q.get_device());
  // PC-0-1 bias tile plan (tail slack past the SFVt block): row-broadcast
  // double buffered (mode 2); dense (mode 1) is m4n2-only per the PC-0-1
  // plan. Try the resident vector (mode 3) first -- the fp4 persist_d
  // family measured it fastest -- and demote 3->2->0 by the smem budget.
  FfpaBiasTilePlan bias_plan;
  if (bias.ptr != nullptr) {
    FfpaBiasParams bias_p{bias.ptr,      bias.dtype,    bias.stride_b,
                          bias.stride_h, bias.stride_m, bias.stride_n};
    bias_plan = ffpa_bias_tile_plan_of(bias_p, Nb, Nh, Nq, Nkv);
    if (bias_plan.mode == 1)
      bias_plan.mode = 0;
  }
  // Static smem (barrier arrays, ~160B incl. the bias pair) is invisible to
  // the dynamic budget: reserve 256B so the attribute set cannot land past
  // the true ceiling (fp4 persist_d D=256 lesson).
  const int dyn_limit = max_smem_optin - 256;
  const auto bias_bytes_of = [&](int m) {
    // Mode 3 pads the resident bytes to a whole kBc tile (the kernel's
    // resident fill zero-fills the pad segment; tail tiles' unclamped
    // injection reads stay in-allocation).
    return (m == 3)
               ? (long long)(Nkv + kBc - 1) / kBc * kBc * bias_plan.elem_size
               : bias_plan.tile_bytes(kBr, kBc, (m == 2) ? 2 : 1);
  };
  if (bias_plan.mode == 2 &&
      (long long)kSmemBytes + bias_bytes_of(3) <= dyn_limit)
    bias_plan.mode = 3;
  if ((long long)kSmemBytes + bias_bytes_of(bias_plan.mode) > dyn_limit)
    bias_plan.mode = 0;
  const int kSmemBytesBias = (int)(((long long)kSmemBytes + 15) & ~15) +
                             (int)bias_bytes_of(bias_plan.mode);
  TORCH_CHECK(kSmemBytesBias <= dyn_limit,
              "ffpa_attn: fp4 split_d D=", kHeadDim, " needs ", kSmemBytesBias,
              "B smem, device opt-in allows ", dyn_limit,
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
  auto tma_bias_r16 = make_tma_bias(std::integral_constant<int, 0>{});
  auto tma_bias_r32 = make_tma_bias(std::integral_constant<int, 1>{});
  TORCH_CHECK(kSmemBytes <= max_smem_optin,
              "ffpa_attn: fp4 split_d D=", kHeadDim, " needs ", kSmemBytes,
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
    constexpr int kBiasOn = decltype(bias_tag)::value;
    constexpr int kModeL = decltype(mode_c)::value;
    constexpr int kB4 = decltype(b4_c)::value;
    auto kernel = ffpa_fp4::split_d_fwd_cute_fp4_sm120<
        Traits, ElementO, decltype(tma_q), decltype(tma_k), decltype(tma_v),
        decltype(tma_o), decltype(tma_sfq), decltype(tma_sfk),
        decltype(tma_sfvt), decltype(tma_ds), decltype(tma_bias_sel), kModeL,
        kB4, kBiasOn>;
    TORCH_CHECK(cudaFuncSetAttribute(
                    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                    kSmemBytesBias) == cudaSuccess,
                "ffpa_attn: fp4 split_d smem opt-in failed for D=", kHeadDim);
    kernel<<<grid, block, kSmemBytesBias, stream>>>(
        tma_q, tma_k, tma_v, tma_o, tma_sfq, tma_sfk, tma_sfvt, tma_ds,
        tma_bias_sel, O_ptr, softmax_lse_ptr,
        fused_wht ? km_rot_f32.data_ptr<float>() : km_f32.data_ptr<float>(),
        fused_wht ? qm_rot.data_ptr<float>() : qm.data_ptr<float>(),
        fp4_smooth_v ? vm_v.data_ptr<float>() : nullptr, Nq, Nkv, Nq_pad,
        Nkv_pad, Nh, Nh_kv, scale, Tc, causal, total_q_rows, Nb, q_start_row,
        nhd_out, bias.ptr, bias.dtype, bias.stride_b, bias.stride_h,
        bias.stride_m, bias.stride_n,
        bias_plan.mode != 0 ? bias_plan.m_total : (long long)1);
  };
  if (bias.ptr == nullptr) {
    launch_with(std::integral_constant<int, 0>{}, tma_bias_r16,
                std::integral_constant<int, 0>{},
                std::integral_constant<int, 0>{});
  } else if (bias_plan.mode == 2) {
    if (bias.dtype == 3)
      launch_with(std::integral_constant<int, 1>{}, tma_bias_r32,
                  std::integral_constant<int, 2>{},
                  std::integral_constant<int, 1>{});
    else
      launch_with(std::integral_constant<int, 1>{}, tma_bias_r16,
                  std::integral_constant<int, 2>{},
                  std::integral_constant<int, 0>{});
  } else if (bias_plan.mode == 3) {
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
    if (fp4_pv_mm_type == 1) {
      launch_cute_fwd_split_d_fp4_sm120_impl<kDataType, kHeadDim, true>(
          Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
          q_start_row, fp4_hadamard, fp4_smooth_v);
    } else {
      launch_cute_fwd_split_d_fp4_sm120_impl<kDataType, kHeadDim, false>(
          Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
          q_start_row, fp4_hadamard, fp4_smooth_v);
    }
  } else {
    TORCH_CHECK(false,
                "ffpa_attn: cute_tma_fp4 split_d requires 64-multiple D in "
                "(256,768), got D=",
                kHeadDim);
  }
}

// NVFP4 split-D M4N2 launcher, headdims in [768, 1024]. Identical
// pre-kernel pipeline to the split-D fp4 launcher (km -> q_block_mean ->
// quantize -> delta_s); only the tile geometry changes (kBr=kBc=64, m4n2
// traits own the TMA descriptor shapes). smooth_v follows the same chain
// as split-D/persist-D (V^T quantize residual + epilogue add-back);
// fp4_pv_mm_type stays NVFP4-only here (see the wrapper: the MXFP8 PV
// atom needs Tile-K=128 but m4n2 tiles are kBc=64).
template <typename kDataType, const int kHeadDim>
void launch_cute_fwd_split_d_m4n2_fp4_sm120_impl(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
    torch::Tensor attn_bias, torch::Tensor softmax_lse, int causal,
    double softmax_scale, int q_start_row = 0, bool fp4_hadamard = false,
    bool fp4_smooth_v = false) {
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
  // PC-0-1 bias tile plan (tail slack past the P/exchange block):
  // row-broadcast double buffered (mode 2); dense (mode 1) tile support is
  // m4n2-native but wired in a follow-up (B-1 wired the fp8 m4n2 dense
  // tile). Try the resident vector (mode 3) first -- measured fastest on
  // both fp4 families so far -- and demote 3->2->0 by the smem budget.
  FfpaBiasTilePlan bias_plan;
  if (bias.ptr != nullptr) {
    FfpaBiasParams bias_p{bias.ptr,      bias.dtype,    bias.stride_b,
                          bias.stride_h, bias.stride_m, bias.stride_n};
    bias_plan = ffpa_bias_tile_plan_of(bias_p, Nb, Nh, Nq, Nkv);
    if (bias_plan.mode == 1)
      bias_plan.mode = 0;
      // PC-0-5: the fp4 m4n2 bias-smem paths (mode 2 TMA double buffer, mode 3
      // resident fill) hit a 100%-reproducible O corruption on cold bias
      // calls (bitwise-unstable across identical launches, lse stable,
      // corruption pinned to one (m-warp, n-warp, v-chunk) PV C tile;
      // protocol audit closed at the language level and ptxas -O2 /
      // producer-warp relocation both reproduce it - see RFC PC-0-5). Pin
      // the gmem direct-read mode (mode 0), which is stable on that
      // sequence, at ~5% attn-mask cost. RESIDUAL (known, accepted): with a
      // heavy GPU-work prelude (any matmul burst) the bias template still
      // shows the same low-probability corruption even in mode 0 - the race
      // is load-timing sensitive at the hardware level, not bias-smem
      // specific; the no-bias template is clean under the same load. fp4
      // m4n2 only serves D>=768 fp4 (rare in production), so this is
      // documented rather than root-caused (NVIDIA report pending).
      // FFPA_BIAS_TILE_KEEP=1 restores the smem tile modes (debug build only).
#ifdef ENABLE_FFPA_FP4_BUILD_DEBUG
    if (bias_plan.mode != 0 && getenv("FFPA_BIAS_TILE_KEEP") == nullptr)
      bias_plan.mode = 0;
#else
    bias_plan.mode = 0;
#endif
  }
  // Static smem (barrier arrays, ~160B incl. the bias pair) is invisible to
  // the dynamic budget: reserve 256B so the attribute set cannot land past
  // the true ceiling (fp4 persist_d D=256 lesson).
  const int dyn_limit = max_smem_optin - 256;
  const auto bias_bytes_of = [&](int m) {
    // Mode 3 pads the resident bytes to a whole kBc tile: tail tiles'
    // unclamped injection reads stay in-allocation (the kernel's resident
    // fill zero-fills the pad segment).
    return (m == 3)
               ? (long long)(Nkv + kBc - 1) / kBc * kBc * bias_plan.elem_size
               : bias_plan.tile_bytes(kBr, kBc, (m == 2) ? 2 : 1);
  };
  // PC-0-5 debug switch: keep mode 2 (TMA row-broadcast double buffer).
  // The mode 3 resident-vector upgrade stays OFF on this family: the
  // fill+inject pair has a deterministic 8B shared overread at whole-tail
  // shapes (Nkv=kBc, repro'd 4/4 on pre-fix builds; ptxas merges the
  // per-column scalar LDS into LDS.64 past the resident segment) on top
  // of the PC-0-5 corruption, so mode 2 is the only KEEP debug path that
  // can run clean. Production pins mode 0 anyway (PC-0-5).
  if ((long long)kSmemBytes + bias_bytes_of(bias_plan.mode) > dyn_limit)
    bias_plan.mode = 0;
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
  auto tma_bias_r16 = make_tma_bias(std::integral_constant<int, 0>{});
  auto tma_bias_r32 = make_tma_bias(std::integral_constant<int, 1>{});
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
    constexpr int kBiasOn = decltype(bias_tag)::value;
    constexpr int kModeL = decltype(mode_c)::value;
    constexpr int kB4 = decltype(b4_c)::value;
    auto kernel = ffpa_fp4::split_d_m4n2_fwd_cute_fp4_sm120<
        Traits, ElementO, decltype(tma_q), decltype(tma_k), decltype(tma_v),
        decltype(tma_o), decltype(tma_sfq), decltype(tma_sfk),
        decltype(tma_sfvt), decltype(tma_ds), decltype(tma_bias_sel), kModeL,
        kB4, kBiasOn>;
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
  if (bias.ptr == nullptr) {
    launch_with(std::integral_constant<int, 0>{}, tma_bias_r16,
                std::integral_constant<int, 0>{},
                std::integral_constant<int, 0>{});
  } else if (bias_plan.mode == 2) {
    if (bias.dtype == 3)
      launch_with(std::integral_constant<int, 1>{}, tma_bias_r32,
                  std::integral_constant<int, 2>{},
                  std::integral_constant<int, 1>{});
    else
      launch_with(std::integral_constant<int, 1>{}, tma_bias_r16,
                  std::integral_constant<int, 2>{},
                  std::integral_constant<int, 0>{});
  } else if (bias_plan.mode == 3) {
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
    launch_cute_fwd_split_d_m4n2_fp4_sm120_impl<kDataType, kHeadDim>(
        Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale, q_start_row,
        fp4_hadamard, fp4_smooth_v);
  } else {
    TORCH_CHECK(false,
                "ffpa_attn: cute_tma_fp4 split_d m4n2 requires 64-multiple D "
                "in [768,1024], got D=",
                kHeadDim);
  }
}

#endif  // ENABLE_FFPA_CUTE_EXT && ENABLE_FFPA_TMA_EXT

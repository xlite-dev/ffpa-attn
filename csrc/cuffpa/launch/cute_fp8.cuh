#pragma once
// CuTe fp8 family launchers (persist-D / split-D M8N1 / split-D M4N2 with
// their quantize/smooth/hadamard pre-kernel orchestration), moved
// verbatim out of the old cute/launch.cuh.
#include "launch/common.cuh"
#if defined(ENABLE_FFPA_CUTE_EXT) && defined(ENABLE_FFPA_TMA_EXT)
#include "cute/fp8/quantize_fp8.cuh"
#include "cute/fp8/smooth_k.cuh"
#include "cute/fp8/sm_120/persist_d.cuh"
#include "cute/fp8/sm_120/split_d.cuh"
#include "cute/fp8/sm_120/split_d_m4n2.cuh"
#include "cute/hadamard.cuh"

// FP8 persist-D: fp16/bf16 in, internally blockwise-quantized (Q/K row-major
// to e4m3 or symmetric int8, V transposed to e4m3), then low-precision
// attention. kQKInt8: QK runs s8xs8->s32 MMA (cast to f32 before softmax).
// D=64/128 only.
template <typename kDataType, const int kHeadDim, const int kStage,
          bool kQKInt8>
void launch_cute_fwd_persist_d_fp8_sm120_impl(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
    torch::Tensor attn_bias, torch::Tensor softmax_lse, int causal,
    double softmax_scale, double dropout_p, int64_t philox_seed,
    int64_t philox_offset, bool fp8_smooth_k, bool fp8_smooth_v,
    int64_t fp8_q_quant_method, int64_t fp8_k_quant_method,
    int64_t fp8_v_quant_method, int64_t fp8_pv_acc_type, int q_start_row = 0,
    bool fp8_hadamard = false) {
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

  auto opts_qk = torch::TensorOptions()
                     .dtype(kQKInt8 ? torch::kChar : torch::kFloat8_e4m3fn)
                     .device(Q.device());
  auto opts_u8 =
      torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(Q.device());
  auto opts_f32 =
      torch::TensorOptions().dtype(torch::kFloat32).device(Q.device());
  torch::Tensor q8 = torch::empty({Nb, Nh, Nq, kHeadDim}, opts_qk);
  torch::Tensor k8 = torch::empty({Nb, Nh_kv, Nkv, kHeadDim}, opts_qk);
  torch::Tensor vt8 = torch::empty({Nb, Nh_kv, kHeadDim, Nkv_pad}, opts_u8);
  // Per-thread QK: 64 scale/Q-block, 4 scale/K-block (fragment-aligned).
  torch::Tensor q_scale =
      torch::empty({Nb * Nh, qk_per_thread ? n_rb_q * 64 : n_rb_q}, opts_f32);
  torch::Tensor k_scale = torch::empty(
      {Nb * Nh_kv, qk_per_thread ? n_rb_kv * 4 : n_rb_kv}, opts_f32);
  // Per-channel V (along D, amax over N) -- sage style. Re-quantize V,
  // overwriting the per-block vt8/v_scale produced above. Scale stays 448.
  // v_per_channel / v_smooth_mean are resolved from API params at the top of
  // this function.
  torch::Tensor v_scale = v_per_channel
                              ? torch::empty({Nb * Nh_kv, kHeadDim}, opts_f32)
                              : torch::empty({Nb * Nh_kv, n_rb_kv}, opts_f32);
  torch::Tensor v_scale_quant =
      v_per_channel ? torch::empty({Nb * Nh_kv, n_rb_kv}, opts_f32) : v_scale;

  const c10::cuda::OptionalCUDAGuard device_guard(Q.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  // Smooth-K (K -= per-(b,h) seq mean before quantize) defaults on; it is
  // mathematically lossless for O, only lse needs the correction done in the
  // attention kernel epilogue. km = per-(b,h) seq mean of K, (B*Nh_kv, D).
  // The mean stays a separate launch, NOT fused into quantize, because:
  //   - mean reduces ALONG seqlen (across all row blocks) while quantize
  //     parallelizes ALONG seqlen (per row block); fusing creates a
  //     cross-block global dependency (atomics + spin barrier) that costs
  //     more than the mean kernel it replaces;
  //   - no DRAM savings: K is cold-read once, mean fills L2 and quantize
  //     re-reads it from L2.
  // Implemented as a custom two-stage kernel (launch_kv_mean_sm120, ~50us at
  // B1 H32 N8192 D128) instead of at::mean + fp32 cast (~85us): it reads K
  // once coalesced with fp32 accumulate and emits both dtypes in one pass.
  torch::Tensor km, km_f32, km_partials;
  const kDataType* km_ptr = nullptr;
  const float* km_f32_ptr = nullptr;
  const kDataType* q_ptr = reinterpret_cast<const kDataType*>(Q.data_ptr());
  const kDataType* k_ptr = reinterpret_cast<const kDataType*>(K.data_ptr());
  const kDataType* v_ptr = reinterpret_cast<const kDataType*>(V.data_ptr());
  if (fp8_smooth_k) {
    // Custom two-stage column mean (~50us) replacing at::mean + fp32 cast
    // (~85us); emits the in-dtype mean and its fp32 copy in one pass.
    const int mean_chunks =
        (Nkv + ffpa_fp8::kMeanRowsPerChunk - 1) / ffpa_fp8::kMeanRowsPerChunk;
    km = torch::empty({Nb * Nh_kv, kHeadDim}, K.options());
    km_f32 = torch::empty({Nb * Nh_kv, kHeadDim}, opts_f32);
    km_partials = torch::empty({Nb * Nh_kv, mean_chunks, kHeadDim}, opts_f32);
    km_ptr = reinterpret_cast<const kDataType*>(km.data_ptr());
    km_f32_ptr = km_f32.data_ptr<float>();
    ffpa_fp8::launch_kv_mean_sm120<kDataType, kHeadDim>(
        k_ptr, reinterpret_cast<kDataType*>(km.data_ptr()),
        km_f32.data_ptr<float>(), km_partials.data_ptr<float>(), Nb, Nh_kv, Nkv,
        D_og, stream, &Lkv);
  }
  if (qk_per_thread) {
    ffpa_fp8::launch_quantize_fp8_perthread_qk_sm120<kDataType, kBr, kBc,
                                                     kHeadDim, kQKInt8>(
        q_ptr, k_ptr, v_ptr, q8.data_ptr(), k8.data_ptr(),
        reinterpret_cast<__nv_fp8_e4m3*>(vt8.data_ptr()),
        q_scale.data_ptr<float>(), k_scale.data_ptr<float>(),
        v_scale_quant.data_ptr<float>(), Nb, Nh, Nh_kv, Nq, Nkv, Nkv_pad, D_og,
        Lq, Lkv, stream, km_ptr, reorg_free, v_per_channel, &Lv);
  } else {
    ffpa_fp8::launch_quantize_fp8_sm120<kDataType, kBr, kBc, kHeadDim, kQKInt8>(
        q_ptr, k_ptr, v_ptr, q8.data_ptr(), k8.data_ptr(),
        reinterpret_cast<__nv_fp8_e4m3*>(vt8.data_ptr()),
        q_scale.data_ptr<float>(), k_scale.data_ptr<float>(),
        v_scale_quant.data_ptr<float>(), Nb, Nh, Nh_kv, Nq, Nkv, Nkv_pad, D_og,
        Lq, Lkv, stream, km_ptr, reorg_free, v_per_channel, &Lv);
  }

  // Per-channel V (sage-style): re-quantize V with per-D scale via coalesced
  // stats (sum+max+min -> mean+amax) + quantize/transpose. smooth_v subtracts
  // the per-D mean (residual amax); the per-block vt8/v_scale are overwritten.
  torch::Tensor vm, v_partials_sum, v_partials_max, v_partials_min;
  float* vm_ptr = nullptr;
  if (v_per_channel) {
    const int stats_chunks = (Nkv + ffpa_fp8::kVStatsRowsPerChunk - 1) /
                             ffpa_fp8::kVStatsRowsPerChunk;
    v_partials_sum =
        torch::empty({Nb * Nh_kv, stats_chunks, kHeadDim}, opts_f32);
    v_partials_max =
        torch::empty({Nb * Nh_kv, stats_chunks, kHeadDim}, opts_f32);
    v_partials_min =
        torch::empty({Nb * Nh_kv, stats_chunks, kHeadDim}, opts_f32);
    vm = torch::empty({Nb * Nh_kv, kHeadDim}, opts_f32);
    vm_ptr = vm.data_ptr<float>();
    if (v_smooth_mean) {
      ffpa_fp8::launch_quantize_fp8_vt_perchannel_sm120<kDataType, kBr, kBc,
                                                        kHeadDim, true>(
          v_ptr, reinterpret_cast<__nv_fp8_e4m3*>(vt8.data_ptr()),
          v_scale.data_ptr<float>(), vm_ptr, v_partials_sum.data_ptr<float>(),
          v_partials_max.data_ptr<float>(), v_partials_min.data_ptr<float>(),
          Nb, Nh_kv, Nkv, Nkv_pad, stream, D_og, v_r, reorg_free, &Lv);
    } else {
      ffpa_fp8::launch_quantize_fp8_vt_perchannel_sm120<kDataType, kBr, kBc,
                                                        kHeadDim, false>(
          v_ptr, reinterpret_cast<__nv_fp8_e4m3*>(vt8.data_ptr()),
          v_scale.data_ptr<float>(), vm_ptr, v_partials_sum.data_ptr<float>(),
          v_partials_max.data_ptr<float>(), v_partials_min.data_ptr<float>(),
          Nb, Nh_kv, Nkv, Nkv_pad, stream, D_og, v_r, reorg_free, &Lv);
    }
  }
  const float* vm_kernel = v_smooth_mean ? vm_ptr : nullptr;

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
  // PC-0-1 bias tile plan (tail slack past the K/V stages): row-broadcast
  // double buffered (mode 2). Dense (mode 1) is m4n2-only per the PC-0-1
  // plan. Mode 3 (resident vector) won the A/B below (14.46 vs 14.68ms at
  // D=128 N=16384) and is the preferred upgrade; mode 2 remains the
  // fallback when the padded resident footprint busts the smem budget.
  FfpaBiasTilePlan bias_plan;
  if (bias.ptr != nullptr) {
    FfpaBiasParams bias_p{bias.ptr,      bias.dtype,    bias.stride_b,
                          bias.stride_h, bias.stride_m, bias.stride_n};
    bias_plan = ffpa_bias_tile_plan_of(bias_p, Nb, Nh, Nq, Nkv);
    if (bias_plan.mode == 1)
      bias_plan.mode = 0;
  }
  int max_smem_optin = 0;
  cudaDeviceGetAttribute(
      &max_smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, Q.get_device());
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
  // Prefer the resident vector (measured 14.46 vs 14.68ms at D=128 N=16384,
  // consistent with the fp16 persist_d family); the double-buffered tile is
  // the fallback when the Nkv footprint busts the smem budget.
  if (bias_plan.mode == 2 &&
      (long long)kSmemBytes + bias_bytes_of(3) <= dyn_limit)
    bias_plan.mode = 3;
  if ((long long)kSmemBytes + bias_bytes_of(bias_plan.mode) > dyn_limit)
    bias_plan.mode = 0;
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
  auto tma_bias_r32 = make_tma_bias(std::integral_constant<int, 1>{});
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
    constexpr int kBiasOn = decltype(bias_tag)::value;
    constexpr int kModeL = decltype(mode_c)::value;
    constexpr int kB4 = decltype(b4_c)::value;
    using TmaBiasSel = decltype(tma_bias_sel);
    const auto kernel_of = [&](auto pq_pr, auto qk_pt, auto v_pc, auto pv_f16) {
      return ffpa_fp8::persist_d_ws_fwd_cute_fp8_sm120<
          Traits, ElementO, TmaQ, TmaK, TmaV, TmaO, TmaBiasSel,
          decltype(pq_pr)::value, decltype(pv_f16)::value,
          decltype(v_pc)::value, decltype(qk_pt)::value, reorg_free,
          ffpa_fp8::kPersistQs2rDefault, kModeL, kB4, kBiasOn>;
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
void launch_cute_fwd_persist_d_fp8_sm120(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
    torch::Tensor attn_bias, torch::Tensor softmax_lse, int causal,
    double softmax_scale, double dropout_p, int64_t philox_seed,
    int64_t philox_offset, bool fp8_smooth_k, bool fp8_smooth_v,
    int64_t fp8_q_quant_method, int64_t fp8_k_quant_method,
    int64_t fp8_v_quant_method, int64_t fp8_pv_acc_type, int64_t fp8_qk_mm_type,
    int q_start_row = 0, bool fp8_hadamard = false) {
  // qk_mm_type: 0=fp8 (e4m3 QK MMA), 1=int8 (s8xs8->s32). Default fp8;
  // int8 fixes the causal early-row dS accuracy limit at ~zero cost.
  // if constexpr keeps the impl (and its kernel) out of instantiation for
  // unsupported headdims; every headdim TU includes this launcher template.
  if constexpr (kHeadDim % 32 == 0 && kHeadDim >= 32 && kHeadDim <= 224) {
    const bool qk_int8 = (fp8_qk_mm_type == 1);
    if (qk_int8)
      launch_cute_fwd_persist_d_fp8_sm120_impl<kDataType, kHeadDim, kStage,
                                               true>(
          Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale, dropout_p,
          philox_seed, philox_offset, fp8_smooth_k, fp8_smooth_v,
          fp8_q_quant_method, fp8_k_quant_method, fp8_v_quant_method,
          fp8_pv_acc_type, q_start_row, fp8_hadamard);
    else
      launch_cute_fwd_persist_d_fp8_sm120_impl<kDataType, kHeadDim, kStage,
                                               false>(
          Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale, dropout_p,
          philox_seed, philox_offset, fp8_smooth_k, fp8_smooth_v,
          fp8_q_quant_method, fp8_k_quant_method, fp8_v_quant_method,
          fp8_pv_acc_type, q_start_row, fp8_hadamard);
  } else {
    TORCH_CHECK(false,
                "ffpa_attn: cute_tma_fp8 persist_d requires D in {32..224} "
                "step 32, got D=",
                kHeadDim);
  }
}

// Split-D FP8 launcher (headdim > 128): non-WS M8N1 kernel over quantized
// q8/k8/vt8 buffers. Fixed-P-scale only (FFPA_FP8_PQUANT_PER_ROW applies to
// the persist_d path only and is ignored here).
template <typename kDataType, const int kHeadDim, const int kStage,
          bool kQKInt8>
void launch_cute_fwd_split_d_fp8_sm120_impl(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
    torch::Tensor attn_bias, torch::Tensor softmax_lse, int causal,
    double softmax_scale, double dropout_p, int64_t philox_seed,
    int64_t philox_offset, bool fp8_smooth_k, bool fp8_smooth_v,
    int64_t fp8_q_quant_method, int64_t fp8_k_quant_method,
    int64_t fp8_v_quant_method, int64_t fp8_pv_acc_type, int q_start_row = 0,
    bool fp8_hadamard = false) {
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

  auto opts_qk = torch::TensorOptions()
                     .dtype(kQKInt8 ? torch::kChar : torch::kFloat8_e4m3fn)
                     .device(Q.device());
  auto opts_u8 =
      torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(Q.device());
  auto opts_f32 =
      torch::TensorOptions().dtype(torch::kFloat32).device(Q.device());
  torch::Tensor q8 = torch::empty({Nb, Nh, Nq, kHeadDim}, opts_qk);
  torch::Tensor k8 = torch::empty({Nb, Nh_kv, Nkv, kHeadDim}, opts_qk);
  torch::Tensor vt8 = torch::empty({Nb, Nh_kv, kHeadDim, Nkv_pad}, opts_u8);
  // Per-thread QK: 64 scale/Q-block, 4 scale/K-block (fragment-aligned).
  torch::Tensor q_scale =
      torch::empty({Nb * Nh, qk_per_thread ? n_rb_q * 64 : n_rb_q}, opts_f32);
  torch::Tensor k_scale = torch::empty(
      {Nb * Nh_kv, qk_per_thread ? n_rb_kv * 4 : n_rb_kv}, opts_f32);
  // Per-channel V (along D): v_scale is (bh, D) for per-channel, (bh,
  // n_rb_kv) for per-block. v_scale_quant feeds the first per-block quantize
  // pass; per-channel overwrites vt8/v_scale afterwards.
  torch::Tensor v_scale = v_per_channel
                              ? torch::empty({Nb * Nh_kv, kHeadDim}, opts_f32)
                              : torch::empty({Nb * Nh_kv, n_rb_kv}, opts_f32);
  torch::Tensor v_scale_quant =
      v_per_channel ? torch::empty({Nb * Nh_kv, n_rb_kv}, opts_f32) : v_scale;

  const c10::cuda::OptionalCUDAGuard device_guard(Q.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  torch::Tensor km, km_f32, km_partials;
  const kDataType* km_ptr = nullptr;
  const float* km_f32_ptr = nullptr;
  const kDataType* q_ptr = reinterpret_cast<const kDataType*>(Q.data_ptr());
  const kDataType* k_ptr = reinterpret_cast<const kDataType*>(K.data_ptr());
  const kDataType* v_ptr = reinterpret_cast<const kDataType*>(V.data_ptr());
  if (fp8_smooth_k) {
    // Custom two-stage column mean (~50us) replacing at::mean + fp32 cast
    // (~85us); emits the in-dtype mean and its fp32 copy in one pass.
    const int mean_chunks =
        (Nkv + ffpa_fp8::kMeanRowsPerChunk - 1) / ffpa_fp8::kMeanRowsPerChunk;
    km = torch::empty({Nb * Nh_kv, kHeadDim}, K.options());
    km_f32 = torch::empty({Nb * Nh_kv, kHeadDim}, opts_f32);
    km_partials = torch::empty({Nb * Nh_kv, mean_chunks, kHeadDim}, opts_f32);
    km_ptr = reinterpret_cast<const kDataType*>(km.data_ptr());
    km_f32_ptr = km_f32.data_ptr<float>();
    ffpa_fp8::launch_kv_mean_sm120<kDataType, kHeadDim>(
        k_ptr, reinterpret_cast<kDataType*>(km.data_ptr()),
        km_f32.data_ptr<float>(), km_partials.data_ptr<float>(), Nb, Nh_kv, Nkv,
        D_og, stream, &Lkv);
  }
  if (qk_per_thread) {
    ffpa_fp8::launch_quantize_fp8_perthread_qk_sm120<kDataType, kBr, kBc,
                                                     kHeadDim, kQKInt8>(
        q_ptr, k_ptr, v_ptr, q8.data_ptr(), k8.data_ptr(),
        reinterpret_cast<__nv_fp8_e4m3*>(vt8.data_ptr()),
        q_scale.data_ptr<float>(), k_scale.data_ptr<float>(),
        v_scale_quant.data_ptr<float>(), Nb, Nh, Nh_kv, Nq, Nkv, Nkv_pad, D_og,
        Lq, Lkv, stream, km_ptr, reorg_free, v_per_channel, &Lv);
  } else {
    ffpa_fp8::launch_quantize_fp8_sm120<kDataType, kBr, kBc, kHeadDim, kQKInt8>(
        q_ptr, k_ptr, v_ptr, q8.data_ptr(), k8.data_ptr(),
        reinterpret_cast<__nv_fp8_e4m3*>(vt8.data_ptr()),
        q_scale.data_ptr<float>(), k_scale.data_ptr<float>(),
        v_scale_quant.data_ptr<float>(), Nb, Nh, Nh_kv, Nq, Nkv, Nkv_pad, D_og,
        Lq, Lkv, stream, km_ptr, reorg_free, v_per_channel, &Lv);
  }

  // Per-channel V (sage-style): re-quantize V with per-D scale via coalesced
  // stats (sum+max+min -> mean+amax) + quantize/transpose. smooth_v subtracts
  // the per-D mean (residual amax); overwrites the per-block vt8/v_scale.
  torch::Tensor vm, v_partials_sum, v_partials_max, v_partials_min;
  float* vm_ptr = nullptr;
  if (v_per_channel) {
    const int stats_chunks = (Nkv + ffpa_fp8::kVStatsRowsPerChunk - 1) /
                             ffpa_fp8::kVStatsRowsPerChunk;
    v_partials_sum =
        torch::empty({Nb * Nh_kv, stats_chunks, kHeadDim}, opts_f32);
    v_partials_max =
        torch::empty({Nb * Nh_kv, stats_chunks, kHeadDim}, opts_f32);
    v_partials_min =
        torch::empty({Nb * Nh_kv, stats_chunks, kHeadDim}, opts_f32);
    vm = torch::empty({Nb * Nh_kv, kHeadDim}, opts_f32);
    vm_ptr = vm.data_ptr<float>();
    if (v_smooth_mean) {
      ffpa_fp8::launch_quantize_fp8_vt_perchannel_sm120<kDataType, kBr, kBc,
                                                        kHeadDim, true>(
          v_ptr, reinterpret_cast<__nv_fp8_e4m3*>(vt8.data_ptr()),
          v_scale.data_ptr<float>(), vm_ptr, v_partials_sum.data_ptr<float>(),
          v_partials_max.data_ptr<float>(), v_partials_min.data_ptr<float>(),
          Nb, Nh_kv, Nkv, Nkv_pad, stream, D_og, v_r, reorg_free, &Lv);
    } else {
      ffpa_fp8::launch_quantize_fp8_vt_perchannel_sm120<kDataType, kBr, kBc,
                                                        kHeadDim, false>(
          v_ptr, reinterpret_cast<__nv_fp8_e4m3*>(vt8.data_ptr()),
          v_scale.data_ptr<float>(), vm_ptr, v_partials_sum.data_ptr<float>(),
          v_partials_max.data_ptr<float>(), v_partials_min.data_ptr<float>(),
          Nb, Nh_kv, Nkv, Nkv_pad, stream, D_og, v_r, reorg_free, &Lv);
    }
  }
  const float* vm_kernel = v_smooth_mean ? vm_ptr : nullptr;

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
  // PC-0-1 bias tile plan (tail slack past the QK/V stages): row-broadcast
  // double buffered (mode 2) only. Dense (mode 1) is m4n2-only per the
  // PC-0-1 plan; mode 3 (resident vector) measured no-win on this family
  // (D=320: 59.2 vs 50.9ms for mode 2) so it is never selected -- see the
  // D>=512 demote below for the pipeline-starvation analysis.
  FfpaBiasTilePlan bias_plan;
  if (bias.ptr != nullptr) {
    FfpaBiasParams bias_p{bias.ptr,      bias.dtype,    bias.stride_b,
                          bias.stride_h, bias.stride_m, bias.stride_n};
    bias_plan = ffpa_bias_tile_plan_of(bias_p, Nb, Nh, Nq, Nkv);
    if (bias_plan.mode == 1)
      bias_plan.mode = 0;
  }
  int max_smem_optin = 0;
  cudaDeviceGetAttribute(
      &max_smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, Q.get_device());
  // Static smem (barrier arrays, <= 144B incl. the bias pair) is invisible
  // to the dynamic budget: reserve 256B so the attribute set cannot land
  // past the true ceiling (fp4 persist_d D=256 lesson).
  const int dyn_limit = max_smem_optin - 256;
  const int bias_stages = (bias_plan.mode == 2) ? 2 : 1;
  if ((long long)kSmemBytes + bias_plan.tile_bytes(kBr, kBc, bias_stages) >
      dyn_limit)
    bias_plan.mode = 0;
  if (bias_plan.mode == 2 && kHeadDim >= 512) {
    // D=512 measured (N=16384 H32): tile 275ms vs gmem 115ms -- the
    // injection enters the consumer's critical path, the 2-deep s2 QK
    // pipeline starves and every warp spins in qk_full phase checks
    // (SYNCS.PHASECHK = 18% of stall samples, sleeping 22% vs gmem's
    // 3.5%). Structural at 1 CTA/SM (REG:255) with no warp overlap: keep
    // the FC-4 gmem path whose LDG injection leaves the smem/MIO pipe
    // alone. Mode 3 does not help either (267ms): the resident load and
    // the injection LDS share the same critical path.
    bias_plan.mode = 0;
  }
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
  auto tma_bias_r32 = make_tma_bias(std::integral_constant<int, 1>{});
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
    constexpr int kBiasOn = decltype(bias_tag)::value;
    constexpr int kModeL = decltype(mode_c)::value;
    constexpr int kB4 = decltype(b4_c)::value;
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
  } else {
    launch_with(std::integral_constant<int, 1>{}, tma_bias_r16,
                std::integral_constant<int, 0>{},
                std::integral_constant<int, 0>{});
  }
}

template <typename kDataType, const int kHeadDim, const int kStage>
void launch_cute_fwd_split_d_fp8_sm120(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
    torch::Tensor attn_bias, torch::Tensor softmax_lse, int causal,
    double softmax_scale, double dropout_p, int64_t philox_seed,
    int64_t philox_offset, bool fp8_smooth_k, bool fp8_smooth_v,
    int64_t fp8_q_quant_method, int64_t fp8_k_quant_method,
    int64_t fp8_v_quant_method, int64_t fp8_pv_acc_type, int64_t fp8_qk_mm_type,
    int q_start_row = 0, bool fp8_hadamard = false) {
  // EXPERIMENT: lower bound lowered from >=768 to >=192 so M4N2 can be A/B'd
  // against M8N1 across all large headdims via FFPA_FP8_FORCE_KERNEL.
  // Production dispatch selects M4N2 only for D>=768 via the top-level
  // launcher.
  if constexpr (kHeadDim >= 192 && kHeadDim <= 1024 && kHeadDim % 64 == 0) {
    const bool qk_int8 = (fp8_qk_mm_type == 1);
    if (qk_int8)
      launch_cute_fwd_split_d_fp8_sm120_impl<kDataType, kHeadDim, kStage, true>(
          Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale, dropout_p,
          philox_seed, philox_offset, fp8_smooth_k, fp8_smooth_v,
          fp8_q_quant_method, fp8_k_quant_method, fp8_v_quant_method,
          fp8_pv_acc_type, q_start_row, fp8_hadamard);
    else
      launch_cute_fwd_split_d_fp8_sm120_impl<kDataType, kHeadDim, kStage,
                                             false>(
          Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale, dropout_p,
          philox_seed, philox_offset, fp8_smooth_k, fp8_smooth_v,
          fp8_q_quant_method, fp8_k_quant_method, fp8_v_quant_method,
          fp8_pv_acc_type, q_start_row, fp8_hadamard);
  } else {
    TORCH_CHECK(false,
                "ffpa_attn: cute_tma_fp8 split_d requires D in "
                "[192, 1024] with D % 64 == 0, got D=",
                kHeadDim);
  }
}

// Split-D M4N2 FP8 launcher: m4n2 atom layout (4,2,1) + fp8 e4m3 Q/K/V.
// Dispatched for D>=768 to avoid M8N1's D/2 register spill (O=D/2>255).
// M4N2 uses D/4 regs per thread; P goes through SMEM roundtrip (stmatrix->
// LDSM_N) since each N-warp holds only half the Bc columns.
template <typename kDataType, const int kHeadDim, const int kStage,
          bool kQKInt8>
void launch_cute_fwd_split_d_m4n2_fp8_sm120_impl(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
    torch::Tensor attn_bias, torch::Tensor softmax_lse, int causal,
    double softmax_scale, double dropout_p, int64_t philox_seed,
    int64_t philox_offset, bool fp8_smooth_k, bool fp8_smooth_v,
    int64_t fp8_q_quant_method, int64_t fp8_k_quant_method,
    int64_t fp8_v_quant_method, int64_t fp8_pv_acc_type, int q_start_row = 0,
    bool fp8_hadamard = false) {
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

  auto opts_qk = torch::TensorOptions()
                     .dtype(kQKInt8 ? torch::kChar : torch::kFloat8_e4m3fn)
                     .device(Q.device());
  auto opts_u8 =
      torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(Q.device());
  auto opts_f32 =
      torch::TensorOptions().dtype(torch::kFloat32).device(Q.device());
  torch::Tensor q8 = torch::empty({Nb, Nh, Nq, kHeadDim}, opts_qk);
  torch::Tensor k8 = torch::empty({Nb, Nh_kv, Nkv, kHeadDim}, opts_qk);
  torch::Tensor vt8 = torch::empty({Nb, Nh_kv, kHeadDim, Nkv_pad}, opts_u8);
  // Per-thread QK: Q uses 128-row quantize blocks (64 scale/block), K uses
  // kBc=64-col blocks (4 scale/block).
  const int n_rb_q_quant = utils::div_ceil(Nq, 128);
  torch::Tensor q_scale = torch::empty(
      {Nb * Nh, qk_per_thread ? n_rb_q_quant * 64 : n_rb_q}, opts_f32);
  torch::Tensor k_scale = torch::empty(
      {Nb * Nh_kv, qk_per_thread ? n_rb_kv * 4 : n_rb_kv}, opts_f32);
  // Per-channel V (along D): v_scale is (bh, D) for per-channel, (bh,
  // n_rb_kv) for per-block. v_scale_quant feeds the first per-block quantize
  // pass; per-channel overwrites vt8/v_scale afterwards.
  torch::Tensor v_scale = v_per_channel
                              ? torch::empty({Nb * Nh_kv, kHeadDim}, opts_f32)
                              : torch::empty({Nb * Nh_kv, n_rb_kv}, opts_f32);
  torch::Tensor v_scale_quant =
      v_per_channel ? torch::empty({Nb * Nh_kv, n_rb_kv}, opts_f32) : v_scale;

  const c10::cuda::OptionalCUDAGuard device_guard(Q.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  torch::Tensor km, km_f32, km_partials;
  const kDataType* km_ptr = nullptr;
  const float* km_f32_ptr = nullptr;
  const kDataType* q_ptr = reinterpret_cast<const kDataType*>(Q.data_ptr());
  const kDataType* k_ptr = reinterpret_cast<const kDataType*>(K.data_ptr());
  const kDataType* v_ptr = reinterpret_cast<const kDataType*>(V.data_ptr());
  if (fp8_smooth_k) {
    // Custom two-stage column mean (~50us) replacing at::mean + fp32 cast
    // (~85us); emits the in-dtype mean and its fp32 copy in one pass.
    const int mean_chunks =
        (Nkv + ffpa_fp8::kMeanRowsPerChunk - 1) / ffpa_fp8::kMeanRowsPerChunk;
    km = torch::empty({Nb * Nh_kv, kHeadDim}, K.options());
    km_f32 = torch::empty({Nb * Nh_kv, kHeadDim}, opts_f32);
    km_partials = torch::empty({Nb * Nh_kv, mean_chunks, kHeadDim}, opts_f32);
    km_ptr = reinterpret_cast<const kDataType*>(km.data_ptr());
    km_f32_ptr = km_f32.data_ptr<float>();
    ffpa_fp8::launch_kv_mean_sm120<kDataType, kHeadDim>(
        k_ptr, reinterpret_cast<kDataType*>(km.data_ptr()),
        km_f32.data_ptr<float>(), km_partials.data_ptr<float>(), Nb, Nh_kv, Nkv,
        D_og, stream, &Lkv);
  }
  if (qk_per_thread) {
    ffpa_fp8::launch_quantize_fp8_perthread_qk_sm120<kDataType, kBr, kBc,
                                                     kHeadDim, kQKInt8>(
        q_ptr, k_ptr, v_ptr, q8.data_ptr(), k8.data_ptr(),
        reinterpret_cast<__nv_fp8_e4m3*>(vt8.data_ptr()),
        q_scale.data_ptr<float>(), k_scale.data_ptr<float>(),
        v_scale_quant.data_ptr<float>(), Nb, Nh, Nh_kv, Nq, Nkv, Nkv_pad, D_og,
        Lq, Lkv, stream, km_ptr, false, v_per_channel, &Lv);
  } else {
    ffpa_fp8::launch_quantize_fp8_sm120<kDataType, kBr, kBc, kHeadDim, kQKInt8>(
        q_ptr, k_ptr, v_ptr, q8.data_ptr(), k8.data_ptr(),
        reinterpret_cast<__nv_fp8_e4m3*>(vt8.data_ptr()),
        q_scale.data_ptr<float>(), k_scale.data_ptr<float>(),
        v_scale_quant.data_ptr<float>(), Nb, Nh, Nh_kv, Nq, Nkv, Nkv_pad, D_og,
        Lq, Lkv, stream, km_ptr, false, v_per_channel, &Lv);
  }
  // Per-channel V (sage-style): re-quantize V with per-D scale via coalesced
  // stats (sum+max+min -> mean+amax) + quantize/transpose. smooth_v subtracts
  // the per-D mean (residual amax); overwrites the per-block vt8/v_scale.
  torch::Tensor vm, v_partials_sum, v_partials_max, v_partials_min;
  float* vm_ptr = nullptr;
  if (v_per_channel) {
    const int stats_chunks = (Nkv + ffpa_fp8::kVStatsRowsPerChunk - 1) /
                             ffpa_fp8::kVStatsRowsPerChunk;
    v_partials_sum =
        torch::empty({Nb * Nh_kv, stats_chunks, kHeadDim}, opts_f32);
    v_partials_max =
        torch::empty({Nb * Nh_kv, stats_chunks, kHeadDim}, opts_f32);
    v_partials_min =
        torch::empty({Nb * Nh_kv, stats_chunks, kHeadDim}, opts_f32);
    vm = torch::empty({Nb * Nh_kv, kHeadDim}, opts_f32);
    vm_ptr = vm.data_ptr<float>();
    if (v_smooth_mean) {
      ffpa_fp8::launch_quantize_fp8_vt_perchannel_sm120<kDataType, kBr, kBc,
                                                        kHeadDim, true>(
          v_ptr, reinterpret_cast<__nv_fp8_e4m3*>(vt8.data_ptr()),
          v_scale.data_ptr<float>(), vm_ptr, v_partials_sum.data_ptr<float>(),
          v_partials_max.data_ptr<float>(), v_partials_min.data_ptr<float>(),
          Nb, Nh_kv, Nkv, Nkv_pad, stream, D_og, v_r, /*perm_vt=*/false, &Lv);
    } else {
      ffpa_fp8::launch_quantize_fp8_vt_perchannel_sm120<kDataType, kBr, kBc,
                                                        kHeadDim, false>(
          v_ptr, reinterpret_cast<__nv_fp8_e4m3*>(vt8.data_ptr()),
          v_scale.data_ptr<float>(), vm_ptr, v_partials_sum.data_ptr<float>(),
          v_partials_max.data_ptr<float>(), v_partials_min.data_ptr<float>(),
          Nb, Nh_kv, Nkv, Nkv_pad, stream, D_og, v_r, /*perm_vt=*/false, &Lv);
    }
  }
  const float* vm_kernel = v_smooth_mean ? vm_ptr : nullptr;

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

  // PC-0-1 bias tile plan (mirrors the fp16 split_d launcher): classify the
  // mask shape, demote to gmem-direct when the tile misses the smem budget,
  // and upgrade row-broadcast to the resident vector (mode 3) when the whole
  // [1,Nkv] row fits past the buffers. kSmemElems counts bytes here (1B
  // elements + exchange), same as the kernel's bias_base rounding.
  FfpaBiasTilePlan bias_plan;
  if (bias.ptr != nullptr) {
    FfpaBiasParams bias_p{bias.ptr,      bias.dtype,    bias.stride_b,
                          bias.stride_h, bias.stride_m, bias.stride_n};
    bias_plan = ffpa_bias_tile_plan_of(bias_p, Nb, Nh, Nq, Nkv);
  }
  constexpr int kBiasSmemBudgetBytes = 99 * 1024;
  const int bias_stages = (bias_plan.mode == 2) ? 2 : 1;
  if (Traits::kSmemElems + bias_plan.tile_bytes(kBr, kBc, bias_stages) >
      kBiasSmemBudgetBytes)
    bias_plan.mode = 0;
  // Mode 3 upgrade must keep the smem-driven blocks/SM of the base layout:
  // this family runs 3 CTA/SM (29.7KB base) and the resident vector's 32KB
  // would drop it to 1 -- measured 4.9% slower than the gmem fallback.
  // Resident bytes are padded to a whole kBc tile (tail tiles' unclamped
  // injection reads stay in-allocation; pad zero-filled by the load).
  if (bias_plan.mode == 2) {
    const long long kv_pad = (Nkv + kBc - 1) / kBc * kBc;
    const long long base_align = (Traits::kSmemElems + 15) & ~15;
    const long long resident = base_align + kv_pad * bias_plan.elem_size;
    if (resident <= kBiasSmemBudgetBytes &&
        101376 / resident >= 101376 / base_align)
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
    constexpr int kBiasOn = decltype(bias_tag)::value;
    constexpr int kModeL = decltype(mode_c)::value;
    constexpr int kB4 = decltype(b4_c)::value;
    using TmaBiasSel = decltype(tma_bias_sel);
    if (qk_per_thread) {
      // Per-thread QK quant (sage style): fragment-aligned dequant scales.
      if (v_per_channel && pv_acc_f16) {
        launch_kernel(ffpa_fp8::split_d_m4n2_fwd_cute_fp8_sm120<
                          Traits, ElementO, TmaQ, TmaK, TmaV, TmaO, TmaBiasSel,
                          true, true, true, kModeL, kB4, kBiasOn>,
                      tma_bias_sel);
      } else if (v_per_channel) {
        launch_kernel(ffpa_fp8::split_d_m4n2_fwd_cute_fp8_sm120<
                          Traits, ElementO, TmaQ, TmaK, TmaV, TmaO, TmaBiasSel,
                          false, true, true, kModeL, kB4, kBiasOn>,
                      tma_bias_sel);
      } else if (pv_acc_f16) {
        launch_kernel(ffpa_fp8::split_d_m4n2_fwd_cute_fp8_sm120<
                          Traits, ElementO, TmaQ, TmaK, TmaV, TmaO, TmaBiasSel,
                          true, false, true, kModeL, kB4, kBiasOn>,
                      tma_bias_sel);
      } else {
        launch_kernel(ffpa_fp8::split_d_m4n2_fwd_cute_fp8_sm120<
                          Traits, ElementO, TmaQ, TmaK, TmaV, TmaO, TmaBiasSel,
                          false, false, true, kModeL, kB4, kBiasOn>,
                      tma_bias_sel);
      }
    } else if (v_per_channel && pv_acc_f16) {
      launch_kernel(ffpa_fp8::split_d_m4n2_fwd_cute_fp8_sm120<
                        Traits, ElementO, TmaQ, TmaK, TmaV, TmaO, TmaBiasSel,
                        true, true, false, kModeL, kB4, kBiasOn>,
                    tma_bias_sel);
    } else if (v_per_channel) {
      launch_kernel(ffpa_fp8::split_d_m4n2_fwd_cute_fp8_sm120<
                        Traits, ElementO, TmaQ, TmaK, TmaV, TmaO, TmaBiasSel,
                        false, true, false, kModeL, kB4, kBiasOn>,
                    tma_bias_sel);
    } else if (pv_acc_f16) {
      launch_kernel(ffpa_fp8::split_d_m4n2_fwd_cute_fp8_sm120<
                        Traits, ElementO, TmaQ, TmaK, TmaV, TmaO, TmaBiasSel,
                        true, false, false, kModeL, kB4, kBiasOn>,
                    tma_bias_sel);
    } else {
      launch_kernel(ffpa_fp8::split_d_m4n2_fwd_cute_fp8_sm120<
                        Traits, ElementO, TmaQ, TmaK, TmaV, TmaO, TmaBiasSel,
                        false, false, false, kModeL, kB4, kBiasOn>,
                    tma_bias_sel);
    }
  };
  if (bias.ptr == nullptr) {
    dispatch(std::integral_constant<int, 0>{}, tma_bias_r16,
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
        dispatch(bias_on, tma_bias_d32, m1, m1);
      else
        dispatch(bias_on, tma_bias_d16, m1, m0);
    } else if (bias_plan.mode == 2) {
      if (bias.dtype == 3)
        dispatch(bias_on, tma_bias_r32, m2, m1);
      else
        dispatch(bias_on, tma_bias_r16, m2, m0);
    } else if (bias_plan.mode == 3) {
      // resident row-vector: no TMA issue in-kernel, descriptor unused.
      if (bias.dtype == 3)
        dispatch(bias_on, tma_bias_r32, m3, m1);
      else
        dispatch(bias_on, tma_bias_r16, m3, m0);
    } else {
      dispatch(bias_on, tma_bias_r16, m0, m0);
    }
  }
}

template <typename kDataType, const int kHeadDim, const int kStage>
void launch_cute_fwd_split_d_m4n2_fp8_sm120(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
    torch::Tensor attn_bias, torch::Tensor softmax_lse, int causal,
    double softmax_scale, double dropout_p, int64_t philox_seed,
    int64_t philox_offset, bool fp8_smooth_k, bool fp8_smooth_v,
    int64_t fp8_q_quant_method, int64_t fp8_k_quant_method,
    int64_t fp8_v_quant_method, int64_t fp8_pv_acc_type, int64_t fp8_qk_mm_type,
    int q_start_row = 0, bool fp8_hadamard = false) {
  // EXPERIMENT: lower bound lowered from >=768 to >=192 so M4N2 can be A/B'd
  // against M8N1 across all large headdims via FFPA_FP8_FORCE_KERNEL.
  // Production dispatch selects M4N2 only for D>=768 via the top-level
  // launcher.
  if constexpr (kHeadDim >= 192 && kHeadDim <= 1024 && kHeadDim % 64 == 0) {
    const bool qk_int8 = (fp8_qk_mm_type == 1);
    if (qk_int8)
      launch_cute_fwd_split_d_m4n2_fp8_sm120_impl<kDataType, kHeadDim, kStage,
                                                  true>(
          Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale, dropout_p,
          philox_seed, philox_offset, fp8_smooth_k, fp8_smooth_v,
          fp8_q_quant_method, fp8_k_quant_method, fp8_v_quant_method,
          fp8_pv_acc_type, q_start_row, fp8_hadamard);
    else
      launch_cute_fwd_split_d_m4n2_fp8_sm120_impl<kDataType, kHeadDim, kStage,
                                                  false>(
          Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale, dropout_p,
          philox_seed, philox_offset, fp8_smooth_k, fp8_smooth_v,
          fp8_q_quant_method, fp8_k_quant_method, fp8_v_quant_method,
          fp8_pv_acc_type, q_start_row, fp8_hadamard);
  } else {
    TORCH_CHECK(false,
                "ffpa_attn: cute_tma_fp8 split_d_m4n2 requires D in "
                "[192, 1024] with D % 64 == 0, got D=",
                kHeadDim);
  }
}

#endif  // ENABLE_FFPA_CUTE_EXT && ENABLE_FFPA_TMA_EXT

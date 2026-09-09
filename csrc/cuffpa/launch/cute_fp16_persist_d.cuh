#pragma once
// CuTe fp16/bf16 persist_d launcher: kernel include + bias plan + variant
// body (`_v`), split out of launch/cute_fp16.cuh so the per-tag variant
// TUs (env.py) preprocess exactly one kernel table.
#include "launch/common.cuh"
#if defined(ENABLE_FFPA_CUTE_EXT) && defined(ENABLE_FFPA_TMA_EXT)
#include "cute/sm_120/persist_d.cuh"

namespace ffpa {

// Single-source final bias-mode decision: the umbrella wrapper dispatch
// and the variant body both call this, so a demote-rule drift fails the
// variant's tag TORCH_CHECK instead of silently launching the wrong
// kernel. Mirrors the pre-split inline plan in cute_fp16.cuh, including
// the dense-tile (mode 1) ceil-split smem accounting. The bias_extra
// re-computation in the variant body is the same pure function of the
// final mode; keep in sync.
template <typename kDataType, const int kHeadDim, const int kStage>
inline FfpaBiasTilePlan fp16_persist_d_bias_plan(const FfpaBiasParams& bias_p,
                                                 int Nb, int Nh, int Nq,
                                                 int Nkv) {
  constexpr int kBr = 128;
  constexpr int kBc = (kHeadDim <= 64) ? 128 : (kHeadDim <= 128) ? 64 : 32;
  constexpr int kSmemBudgetBytes = 99 * 1024;
  constexpr int kElemSize = sizeof(kDataType);
  constexpr int kQPersistBytes = kBr * kHeadDim * kElemSize;
  using Element = std::conditional_t<std::is_same_v<kDataType, __half>,
                                     cutlass::half_t, cutlass::bfloat16_t>;
  // Same stage clamp as the variant body: without it kStagesK can exceed
  // the smem budget and the plan overestimates kBaseSmemBytes, silently
  // demoting bias mode 1/2 tiles to mode 0 (gmem-direct).
  constexpr int kPerStageBytes = 2 * kBc * kHeadDim * kElemSize;
  constexpr int kMaxStages =
      (kSmemBudgetBytes - kQPersistBytes) / kPerStageBytes;
  constexpr int kStagesK =
      (kStage < 1) ? 1 : (kStage > kMaxStages ? kMaxStages : kStage);
  using Traits =
      ffpa_cute::FFPAAttnCuTePersistDTraits<kHeadDim, kBr, kBc, kStagesK,
                                            kStagesK, Element>;
  constexpr int kBaseSmemBytes = Traits::kSmemElems * sizeof(Element);
  FfpaBiasTilePlan plan = ffpa_bias_tile_plan_of(bias_p, Nb, Nh, Nq, Nkv);
  const long long tile_u16_1 = (long long)kBr * kBc * (plan.elem_size / 2);
  const int bias_stages =
      ((long long)kQPersistBytes / 2 >= 2 * tile_u16_1) ? 2 : 1;
  long long bias_extra = std::max(
      0LL, plan.tile_bytes(kBr, kBc, bias_stages) - (long long)kQPersistBytes);
  if (plan.mode == 1) {
    // Mirror persist_d.cuh's ceil-split segment layout: the tail TMA
    // segment still writes a full box, so the extra area must use the
    // same ceil arithmetic (exact sizing under-reserves and the tail
    // writes past the allocation).
    const long long cols_u16 = (long long)kBc * (plan.elem_size / 2);
    const long long q_u16 = (long long)kQPersistBytes / 2;
    const long long box_rows =
        ((long long)kBr * cols_u16 > q_u16) ? q_u16 / cols_u16 : (long long)kBr;
    const long long segs = ((long long)kBr + box_rows - 1) / box_rows;
    bias_extra =
        std::max(0LL, (long long)bias_stages * segs * box_rows * cols_u16 * 2 -
                          (long long)kQPersistBytes);
  }
  if (kBaseSmemBytes + bias_extra > kSmemBudgetBytes)
    plan.mode = 0;
  return plan;
}

}  // namespace ffpa

// WS persist-D launcher (D<=128 via ffpa_fwd_cute_fp16, D<=224/256 via
// the fp8/fp4 hybrid stage-1 entries). Variant tags: kBiasOn (attn_bias
// tensor present), kBiasPlanMode (bias tile mode: 0 = gmem-direct
// fallback, 1 = dense [kBr,kBc] TMA tile, 2 = row-broadcast TMA,
// 3 = resident row vector; persist has no mode 3 - the resident upgrade
// is a split/m4n2-only plan step), kBias4BytesPerElem (1 = 4-byte fp32
// mask, 0 = 2-byte fp16/bf16 mask), kHasDropout (dropout plumbing).
// Explicit instantiations live in the generated variant TUs.
template <typename kDataType, const int kHeadDim, const int kStage, int kBiasOn,
          int kBiasPlanMode, int kBias4BytesPerElem, int kHasDropout>
void launch_cute_fwd_persist_d_sm120_v(torch::Tensor Q, torch::Tensor K,
                                       torch::Tensor V, torch::Tensor O,
                                       torch::Tensor attn_bias,
                                       torch::Tensor softmax_lse, int causal,
                                       double softmax_scale, double dropout_p,
                                       int64_t philox_seed,
                                       int64_t philox_offset) {
  using namespace cute;

  // WS consumer is fixed 256T (8 warps); TiledMma must be 8 warps -> kBr=128.
  // kBc scaled with D so K/V stages fit the 99KB smem budget:
  //   D<=64  -> kBc=128 (per-stage 32KB, S=2)
  //   D=128  -> kBc=64  (per-stage 32KB, S=2; kBc=32 costs ~8%: Tc doubles)
  //   D=256  -> kBc=32  (per-stage 32KB, S=1; Q persist alone is 64KB)
  constexpr int kBr = 128;
  constexpr int kBc = (kHeadDim <= 64) ? 128 : (kHeadDim <= 128) ? 64 : 32;
  constexpr int kSmemBudgetBytes = 99 * 1024;
  constexpr int kElemSize = sizeof(kDataType);
  constexpr int kQPersistBytes = kBr * kHeadDim * kElemSize;
  constexpr int kPerStageBytes = 2 * kBc * kHeadDim * kElemSize;
  constexpr int kMaxStages =
      (kSmemBudgetBytes - kQPersistBytes) / kPerStageBytes;
  constexpr int kStagesK =
      (kStage < 1) ? 1 : (kStage > kMaxStages ? kMaxStages : kStage);
  constexpr int kStagesV = kStagesK;
  // WS: 128 producer + 256 consumer = 384 threads
  constexpr int kNumThreads = 384;

  using Element = std::conditional_t<std::is_same_v<kDataType, __half>,
                                     cutlass::half_t, cutlass::bfloat16_t>;
  using Traits =
      ffpa_cute::FFPAAttnCuTePersistDTraits<kHeadDim, kBr, kBc, kStagesK,
                                            kStagesV, Element>;
  using SmemLayoutQ = typename Traits::SmemLayoutQ;
  using SmemLayoutKV = typename Traits::SmemLayoutKV;
  using SmemLayoutO = typename Traits::SmemLayoutO;

  const int Nb = Q.size(0);
  const int Nh = Q.size(1);
  const int Nh_kv = K.size(1);
  const int Nq = Q.size(2);
  const int Nkv = K.size(2);
  const int Tc = utils::div_ceil(Nkv, kBc);
  const float scale = static_cast<float>(softmax_scale);

  const FfpaBiasParams bias = ffpa_bias_params_of(attn_bias, Q, K);
  const int bias_on = bias.ptr != nullptr ? 1 : 0;
  const void* attn_bias_ptr = bias.ptr;
  const int attn_bias_dtype = bias.dtype;
  const float dropout_p_f = static_cast<float>(dropout_p);
  const unsigned long long philox_seed_u =
      static_cast<unsigned long long>(philox_seed);
  const unsigned long long philox_offset_u =
      static_cast<unsigned long long>(philox_offset);

  // PC-0 bias tile: classify the mask shape; fall back to the gmem-direct
  // path when the tile exceeds the smem budget. The TMA box must be fully
  // static (cute's vectorization inference rejects dynamic modes), so mode
  // and dtype become template splits; the direct fallback reuses the
  // row-vec 2B descriptor shape as a never-issued dummy.
  FfpaBiasTilePlan bias_plan;
  if (bias_on) {
    bias_plan = ffpa::fp16_persist_d_bias_plan<kDataType, kHeadDim, kStage>(
        bias, Nb, Nh, Nq, Nkv);
  }
  TORCH_CHECK(kBiasOn == bias_on &&
                  kBiasPlanMode == (bias_on ? bias_plan.mode : 0) &&
                  kBias4BytesPerElem ==
                      ((kBiasPlanMode != 0 && bias.dtype == 3) ? 1 : 0) &&
                  kHasDropout == (dropout_p > 0.0 ? 1 : 0),
              "ffpa_attn: fp16 persist_d D=", kHeadDim,
              " variant tag mismatch (wrapper dispatch vs plan)");
  const int kBaseSmemBytes = Traits::kSmemElems * sizeof(Element);
  constexpr int kBiasSmemBudgetBytes = 99 * 1024;
  const long long tile_u16_1 = (long long)kBr * kBc * (bias_plan.elem_size / 2);
  const int bias_stages =
      ((long long)kQPersistBytes / 2 >= 2 * tile_u16_1) ? 2 : 1;
  long long bias_extra =
      std::max(0LL, bias_plan.tile_bytes(kBr, kBc, bias_stages) -
                        (long long)kQPersistBytes);
  if (bias_plan.mode == 1) {
    // Same ceil-split accounting as fp16_persist_d_bias_plan (pure
    // function of the final mode); keep in sync.
    const long long cols_u16 = (long long)kBc * (bias_plan.elem_size / 2);
    const long long q_u16 = (long long)kQPersistBytes / 2;
    const long long box_rows =
        ((long long)kBr * cols_u16 > q_u16) ? q_u16 / cols_u16 : (long long)kBr;
    const long long segs = ((long long)kBr + box_rows - 1) / box_rows;
    bias_extra =
        std::max(0LL, (long long)bias_stages * segs * box_rows * cols_u16 * 2 -
                          (long long)kQPersistBytes);
  }
  // PC-14 dropout bitmap: [kBr,kBc] keep-bits x2 stages (ping-pong) past
  // the bias extra area, generated by the consumer threads one tile ahead
  // (inside the K/V TMA wait window). The env escape (debug builds only)
  // turns it off (A/B -> inline per-element Philox) and is read per call so
  // toggling mid-process works. kBc=32 (D=192/256) can't use the half-row
  // scheme (a row must span an even number of 32-bit words) and stays
  // inline.
#ifdef ENABLE_FFPA_FP16_BUILD_DEBUG
  bool dropout_bitmap_on = kHasDropout != 0 && kBc >= 64 &&
                           getenv("FFPA_DROPOUT_BITMAP_DISABLE") == nullptr;
#else
  bool dropout_bitmap_on = kHasDropout != 0 && kBc >= 64;
#endif
  constexpr int kBitmapBytes = kBr * kBc / 8 * 2;
  if (kBaseSmemBytes + bias_extra + kBitmapBytes > kBiasSmemBudgetBytes)
    dropout_bitmap_on = false;
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
        bias_plan.mode != 0 ? reinterpret_cast<const uint16_t*>(attn_bias_ptr)
                            : &kBiasDummyAnchor;
    auto gB = make_tensor(make_gmem_ptr(bias_desc_base),
                          make_shape(plane_rows, plane_cols),
                          make_stride(plane_row_stride, _1{}));
    auto sB = [&] {
      if constexpr (kBiasModeT == 1) {
        // Dense tiles larger than the Q-persist area arrive as multiple
        // box-tall segments (Q area + tail extra, see persist_d.cuh); the
        // box height must match the kernel's kBiasBoxRows.
        constexpr int kQPersistU16 = kQPersistBytes / 2;
        constexpr int box_rows =
            kBr * bias_cols > kQPersistU16 ? kQPersistU16 / bias_cols : kBr;
        return make_layout(Shape<Int<box_rows>, Int<bias_cols>>{},
                           Stride<Int<bias_cols>, _1>{});
      } else
        return make_layout(Shape<_1, Int<bias_cols>>{},
                           Stride<Int<bias_cols>, _1>{});
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
  const int kTotalSmemBytes =
      kBaseSmemBytes + (int)bias_extra + (dropout_bitmap_on ? kBitmapBytes : 0);

  const dim3 block(kNumThreads, 1, 1);
  const dim3 grid(utils::div_ceil(Nq, kBr), Nb * Nh, 1);

  const c10::cuda::OptionalCUDAGuard device_guard(Q.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  const int total_q_rows = Nb * Nh * Nq;
  const int total_kv_rows = Nb * Nh_kv * Nkv;

  // K/V gmem layout: BHND packed (flat 2D TMA rows) or an NHD (diffusers
  // BNHD) permute view consumed natively via a batched 4D TMA descriptor.
  // Q likewise (flat (B*N, H*D) rows, head as a kHeadDim-wide column tile);
  // O stays BHND-packed (the caller allocates it packed and re-views),
  // unless the storage is an NHD (diffusers BNHD) view: then the store is
  // flat [Nb*Nq, Nh*kHeadDim] with the head selecting the column tile,
  // mirroring the NHD Q load. Both branches use dynamic int64 extents/
  // strides so TmaO has a single type and the kernel takes a runtime
  // nhd_out branch.
  const bool nhd_out = ffpa_is_nhd_view(O);
  auto gO =
      nhd_out
          ? make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(O.data_ptr())),
                        make_shape((int64_t)Nb * Nq, (int64_t)Nh * kHeadDim),
                        make_stride((int64_t)Nh * kHeadDim, _1{}))
          : make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(O.data_ptr())),
                        make_shape((int64_t)total_q_rows, (int64_t)kHeadDim),
                        make_stride((int64_t)kHeadDim, _1{}));
  // Per-tensor layout families: BHND-packed, packed-NHD view, or
  // strided-NHD (fused-QKV interleaved chunk rows, e.g. FLUX.2
  // single-stream V). K and V must belong to the same family (the
  // kernel's NHD batch/row domain-offset logic is shared), but their row
  // strides may differ.
  const bool q_nhd = ffpa_is_nhd_view(Q) || ffpa_is_strided_nhd(Q);
  if (ffpa_is_strided_nhd(Q))
    ffpa_check_strided_nhd_aligned(Q, "Q");
  if (!q_nhd) {
    TORCH_CHECK(Q.stride(3) == 1 && Q.stride(2) == (long)kHeadDim &&
                    Q.stride(1) == (long)Nq * kHeadDim &&
                    Q.stride(0) == (long)Nh * Nq * kHeadDim,
                "ffpa_attn: Q must be BHND-contiguous or an NHD (BNHD) view");
  }
  const bool k_nhd = ffpa_is_nhd_view(K) || ffpa_is_strided_nhd(K);
  const bool v_nhd = ffpa_is_nhd_view(V) || ffpa_is_strided_nhd(V);
  TORCH_CHECK(k_nhd == v_nhd,
              "ffpa_attn: K and V must share the same memory layout family "
              "(BHND-packed or NHD)");
  if (ffpa_is_strided_nhd(K))
    ffpa_check_strided_nhd_aligned(K, "K");
  if (ffpa_is_strided_nhd(V))
    ffpa_check_strided_nhd_aligned(V, "V");
  const bool kv_nhd = k_nhd;
  if (!kv_nhd) {
    TORCH_CHECK(K.stride(3) == 1 && K.stride(2) == (long)kHeadDim &&
                    K.stride(1) == (long)Nkv * kHeadDim &&
                    K.stride(0) == (long)Nh_kv * Nkv * kHeadDim,
                "ffpa_attn: K must be BHND-contiguous or an NHD (BNHD) view");
    TORCH_CHECK(V.stride(3) == 1 && V.stride(2) == (long)kHeadDim &&
                    V.stride(1) == (long)Nkv * kHeadDim &&
                    V.stride(0) == (long)Nh_kv * Nkv * kHeadDim,
                "ffpa_attn: V must be BHND-contiguous or an NHD (BNHD) view");
  }

  // Everything from the TMA descriptor build through the kernel dispatch is
  // generic over the Q/K/V descriptor types (2D flat vs batched 4D).
  const auto run = [&](auto tma_q, auto tma_k, auto tma_v, auto q_c,
                       auto kv_c) {
    constexpr bool kNhdQ = decltype(q_c)::value;
    constexpr bool kNhdKV = decltype(kv_c)::value;
    auto tma_o = make_tma_copy(SM90_TMA_STORE{}, gO, SmemLayoutO{},
                               Shape<Int<kBr>, Int<kHeadDim>>{}, _1{});

    const int kSmemBytes = kTotalSmemBytes;

    float* softmax_lse_ptr =
        softmax_lse.numel() > 0 ? softmax_lse.data_ptr<float>() : nullptr;
    auto O_ptr = reinterpret_cast<Element*>(O.data_ptr());

    auto launch_variant = [&](auto kernel_func, auto tma_bias_sel) {
      cudaFuncSetAttribute(
          kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
      kernel_func<<<grid, block, kSmemBytes, stream>>>(
          tma_q, tma_k, tma_v, tma_o, tma_bias_sel, O_ptr, softmax_lse_ptr, Nq,
          Nkv, Nh, Nh_kv, scale, Tc, causal, total_q_rows, total_kv_rows,
          attn_bias_ptr, attn_bias_dtype, bias.stride_b, bias.stride_h,
          bias.stride_m, bias.stride_n, dropout_p_f, philox_seed_u,
          philox_offset_u, nhd_out,
          bias_plan.mode != 0 ? bias_plan.m_total : (long long)1,
          dropout_bitmap_on ? 1 : 0);
    };

    using TmaQ = decltype(tma_q);
    using TmaK = decltype(tma_k);
    using TmaV = decltype(tma_v);
    using TmaO = decltype(tma_o);
    // tag table below is instantiated once per compile-time tag (runtime
    // plan -> tag via the umbrella wrapper); the NHD axes stay a runtime
    // split because the TMA descriptor types are function-local decltype.
    // Bias-bearing rows pass (kBiasPlanMode, kBias4BytesPerElem, 1) -> kernel
    // (kBiasMode, kBias4B, kHasAttnBias); the mode/4B values ride the template
    // variables so the slots cannot rotate (kernel expects mode first).
    if constexpr (kBiasOn == 0) {
      launch_variant(persist_d_ws_fwd_cute_sm120<Traits, TmaQ, TmaK, TmaV, TmaO,
                                                 decltype(tma_bias_r16), 0, 0,
                                                 0, kHasDropout, kNhdKV, kNhdQ>,
                     tma_bias_r16);
    } else if constexpr (kBiasPlanMode == 1) {
      if constexpr (kBias4BytesPerElem)
        launch_variant(
            persist_d_ws_fwd_cute_sm120<Traits, TmaQ, TmaK, TmaV, TmaO,
                                        decltype(tma_bias_d32), kBiasPlanMode,
                                        kBias4BytesPerElem, 1, kHasDropout,
                                        kNhdKV, kNhdQ>,
            tma_bias_d32);
      else
        launch_variant(
            persist_d_ws_fwd_cute_sm120<Traits, TmaQ, TmaK, TmaV, TmaO,
                                        decltype(tma_bias_d16), kBiasPlanMode,
                                        kBias4BytesPerElem, 1, kHasDropout,
                                        kNhdKV, kNhdQ>,
            tma_bias_d16);
    } else if constexpr (kBiasPlanMode == 2) {
      if constexpr (kBias4BytesPerElem)
        launch_variant(
            persist_d_ws_fwd_cute_sm120<Traits, TmaQ, TmaK, TmaV, TmaO,
                                        decltype(tma_bias_r32), kBiasPlanMode,
                                        kBias4BytesPerElem, 1, kHasDropout,
                                        kNhdKV, kNhdQ>,
            tma_bias_r32);
      else
        launch_variant(
            persist_d_ws_fwd_cute_sm120<Traits, TmaQ, TmaK, TmaV, TmaO,
                                        decltype(tma_bias_r16), kBiasPlanMode,
                                        kBias4BytesPerElem, 1, kHasDropout,
                                        kNhdKV, kNhdQ>,
            tma_bias_r16);
    } else {
      // mode 0 demote: gmem-direct bias still needs kHasAttnBias=1.
      launch_variant(
          persist_d_ws_fwd_cute_sm120<
              Traits, TmaQ, TmaK, TmaV, TmaO, decltype(tma_bias_r16),
              kBiasPlanMode, kBias4BytesPerElem, 1, kHasDropout, kNhdKV, kNhdQ>,
          tma_bias_r16);
    }
  };

  // Q TMA: BHND flat (B*H*N, D) rows or NHD flat (B*N, H*D) rows. NHD rows
  // carry the tensor's own row stride: H*D for packed views, wider for
  // strided fused-QKV chunk views.
  const auto make_tma_q = [&](auto q_c) {
    if constexpr (decltype(q_c)::value) {
      auto gQ =
          make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(Q.data_ptr())),
                      make_shape((long)Nb * Nq, (long)Nh * kHeadDim),
                      make_stride(Q.stride(2), _1{}));
      return make_tma_copy(SM90_TMA_LOAD{}, gQ, SmemLayoutQ{},
                           Shape<Int<kBr>, Int<kHeadDim>>{}, _1{});
    } else {
      auto gQ =
          make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(Q.data_ptr())),
                      make_shape(total_q_rows, Int<kHeadDim>{}),
                      make_stride(Int<kHeadDim>{}, _1{}));
      return make_tma_copy(SM90_TMA_LOAD{}, gQ, SmemLayoutQ{},
                           Shape<Int<kBr>, Int<kHeadDim>>{}, _1{});
    }
  };

  if (kv_nhd) {
    // NHD view [B, H, N, D] <- packed [B, N, H, D]: element offset is
    // ((b*N + n)*H + h)*D + d, i.e. a flat (B*N, H*D) row-major matrix with
    // a uniform row stride (H*D packed; wider for strided fused-QKV chunk
    // views). The kernel domain_offsets to the batch's rows and tiles the
    // (H*D) columns by kHeadDim, so the head rides the second tile coord —
    // same flat-2D TMA machinery as BHND.
    auto gK =
        make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(K.data_ptr())),
                    make_shape((long)Nb * Nkv, (long)Nh_kv * kHeadDim),
                    make_stride(K.stride(2), _1{}));
    auto gV =
        make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(V.data_ptr())),
                    make_shape((long)Nb * Nkv, (long)Nh_kv * kHeadDim),
                    make_stride(V.stride(2), _1{}));
    auto tma_k = make_tma_copy(SM90_TMA_LOAD{}, gK, SmemLayoutKV{},
                               Shape<Int<kBc>, Int<kHeadDim>>{}, _1{});
    auto tma_v = make_tma_copy(SM90_TMA_LOAD{}, gV, SmemLayoutKV{},
                               Shape<Int<kBc>, Int<kHeadDim>>{}, _1{});
    if (q_nhd)
      run(make_tma_q(std::true_type{}), tma_k, tma_v, std::true_type{},
          std::true_type{});
    else
      run(make_tma_q(std::false_type{}), tma_k, tma_v, std::false_type{},
          std::true_type{});
  } else {
    auto gK =
        make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(K.data_ptr())),
                    make_shape(total_kv_rows, Int<kHeadDim>{}),
                    make_stride(Int<kHeadDim>{}, _1{}));
    auto gV =
        make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(V.data_ptr())),
                    make_shape(total_kv_rows, Int<kHeadDim>{}),
                    make_stride(Int<kHeadDim>{}, _1{}));
    auto tma_k = make_tma_copy(SM90_TMA_LOAD{}, gK, SmemLayoutKV{},
                               Shape<Int<kBc>, Int<kHeadDim>>{}, _1{});
    auto tma_v = make_tma_copy(SM90_TMA_LOAD{}, gV, SmemLayoutKV{},
                               Shape<Int<kBc>, Int<kHeadDim>>{}, _1{});
    if (q_nhd)
      run(make_tma_q(std::true_type{}), tma_k, tma_v, std::true_type{},
          std::false_type{});
    else
      run(make_tma_q(std::false_type{}), tma_k, tma_v, std::false_type{},
          std::false_type{});
  }
}

#endif  // ENABLE_FFPA_CUTE_EXT && ENABLE_FFPA_TMA_EXT

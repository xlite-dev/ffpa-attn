#pragma once
// CuTe fp16/bf16 family launchers (sm120 persist-D/split-D/M4N2 and the
// sm80 split-D path), moved verbatim out of the old cute/launch.cuh.
#include "launch/common.cuh"
#ifdef ENABLE_FFPA_CUTE_EXT
#include "cute/sm_80/split_d.cuh"
#ifdef ENABLE_FFPA_TMA_EXT
#include "cute/sm_120/split_d.cuh"
#include "cute/sm_120/persist_d.cuh"
#include "cute/sm_120/split_d_m4n2.cuh"

template <typename kDataType, const int kHeadDim, const int kStage,
          const int kQKDChunk, const int kVDChunk>
void launch_cute_fwd_split_d_sm120(torch::Tensor Q, torch::Tensor K,
                                   torch::Tensor V, torch::Tensor O,
                                   torch::Tensor attn_bias,
                                   torch::Tensor softmax_lse, int causal,
                                   double softmax_scale, double dropout_p,
                                   int64_t philox_seed, int64_t philox_offset) {
  using namespace cute;

  constexpr int kBr = 128;
  constexpr int kBc = 128;
  // stages=1: single-buffer makes producer TMA writes (async proxy) collide
  // with consumer ldmatrix reads (generic proxy) on the same smem slot;
  // CtaBarrier (async proxy) can't prove the generic-proxy read finished.
  // Clamp >=2 so double-buffering keeps read/write addresses disjoint.
  constexpr int kStagesQK = (kStage < 2 ? 2 : (kStage > 3 ? 3 : kStage));
  constexpr int kStagesPV = kStagesQK;
  constexpr int kNumThreads = kBr / 16 * 32;

  using Element = std::conditional_t<std::is_same_v<kDataType, __half>,
                                     cutlass::half_t, cutlass::bfloat16_t>;
  using Traits = ffpa_cute::FFPAAttnCuTeSplitDTraits<
      kHeadDim, kBr, kBc, kQKDChunk, kVDChunk, kStagesQK, kStagesPV, Element>;
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

  const bool has_attn_bias = attn_bias.numel() != 0;
  const bool has_dropout = dropout_p > 0.0;

  const void* attn_bias_ptr = nullptr;
  int attn_bias_dtype = 0;
  long long attn_bias_stride_b = 0;
  long long attn_bias_stride_h = 0;
  long long attn_bias_stride_m = 0;
  long long attn_bias_stride_n = 0;
  if (has_attn_bias) {
    TORCH_CHECK(attn_bias.is_cuda(),
                "ffpa_attn: attn_mask must be a CUDA tensor");
    TORCH_CHECK(attn_bias.device() == Q.device(),
                "ffpa_attn: attn_mask must be on the same device as Q/K/V");
    TORCH_CHECK(
        attn_bias.dim() == 4,
        "ffpa_attn: normalized attn_mask must be 4-D [B, Nh_q, Nq, Nkv]");
    TORCH_CHECK(attn_bias.size(0) == 1 || attn_bias.size(0) == Nb,
                "ffpa_attn: attn_mask batch dimension must be 1 or B");
    TORCH_CHECK(attn_bias.size(1) == 1 || attn_bias.size(1) == Nh,
                "ffpa_attn: attn_mask head dimension must be 1 or Nh_q");
    TORCH_CHECK(attn_bias.size(2) == 1 || attn_bias.size(2) == Nq,
                "ffpa_attn: attn_mask query dimension must be 1 or Nq");
    TORCH_CHECK(attn_bias.size(3) == 1 || attn_bias.size(3) == Nkv,
                "ffpa_attn: attn_mask kv dimension must be 1 or Nkv");
    attn_bias_ptr = attn_bias.data_ptr();
    if (attn_bias.scalar_type() == at::ScalarType::Half)
      attn_bias_dtype = 1;
    else if (attn_bias.scalar_type() == at::ScalarType::BFloat16)
      attn_bias_dtype = 2;
    else
      attn_bias_dtype = 3;
    attn_bias_stride_b =
        (attn_bias.size(0) == 1 && Nb > 1) ? 0 : attn_bias.stride(0);
    attn_bias_stride_h =
        (attn_bias.size(1) == 1 && Nh > 1) ? 0 : attn_bias.stride(1);
    attn_bias_stride_m =
        (attn_bias.size(2) == 1 && Nq > 1) ? 0 : attn_bias.stride(2);
    attn_bias_stride_n =
        (attn_bias.size(3) == 1 && Nkv > 1) ? 0 : attn_bias.stride(3);
  }
  const float dropout_p_f = static_cast<float>(dropout_p);
  const unsigned long long philox_seed_u =
      static_cast<unsigned long long>(philox_seed);
  const unsigned long long philox_offset_u =
      static_cast<unsigned long long>(philox_offset);

  const dim3 block(kNumThreads, 1, 1);
  const dim3 grid(utils::div_ceil(Nq, kBr), Nb * Nh, 1);

  const c10::cuda::OptionalCUDAGuard device_guard(Q.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  const int total_q_rows = Nb * Nh * Nq;
  const int total_kv_rows = Nb * Nh_kv * Nkv;

  // NHD (diffusers BNHD) permute views — including strided fused-QKV chunk
  // rows — are consumed natively via flat (B*N, H*D) TMA rows carrying the
  // tensor's own row stride (H*D packed; wider for strided views); O stays
  // BHND-packed (the caller allocates it packed and re-views).
  const bool q_nhd = ffpa_is_nhd_view(Q) || ffpa_is_strided_nhd(Q);
  if (ffpa_is_strided_nhd(Q))
    ffpa_check_strided_nhd_aligned(Q, "Q");
  if (!q_nhd)
    TORCH_CHECK(Q.stride(3) == 1 && Q.stride(2) == (long)kHeadDim &&
                    Q.stride(1) == (long)Nq * kHeadDim &&
                    Q.stride(0) == (long)Nh * Nq * kHeadDim,
                "ffpa_attn: Q must be BHND-contiguous or an NHD (BNHD) view");
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

  // O output TMA store descriptor: BHND flat [total_q_rows,kHeadDim] or
  // NHD (diffusers BNHD packed) [Nb*Nq, Nh*kHeadDim] with the head
  // selecting the column-tile group (kernel folds Nh_id*kDChunksV into
  // the v_chunk walk). Both branches use dynamic int64 extents/strides so
  // TmaO has a single type and the kernel takes a runtime nhd_out branch.
  // Per-head origin injected via domain_offset in kernel; direction =
  // SM90_TMA_STORE (first arg); swizzle auto-inferred from SmemLayoutO.
  const bool nhd_out = ffpa_is_nhd_view(O);
  auto gO =
      nhd_out
          ? make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(O.data_ptr())),
                        make_shape((int64_t)Nb * Nq, (int64_t)Nh * kHeadDim),
                        make_stride((int64_t)Nh * kHeadDim, _1{}))
          : make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(O.data_ptr())),
                        make_shape((int64_t)total_q_rows, (int64_t)kHeadDim),
                        make_stride((int64_t)kHeadDim, _1{}));
  auto tma_o = make_tma_copy(SM90_TMA_STORE{}, gO, SmemLayoutO{},
                             Shape<Int<kBr>, Int<kVDChunk>>{}, _1{});

  auto make_tma_q = [&](auto q_c) {
    if constexpr (decltype(q_c)::value) {
      auto gQ =
          make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(Q.data_ptr())),
                      make_shape((long)Nb * Nq, (long)Nh * kHeadDim),
                      make_stride(Q.stride(2), _1{}));
      return make_tma_copy(SM90_TMA_LOAD{}, gQ, SmemLayoutQ{},
                           Shape<Int<kBr>, Int<kQKDChunk>>{}, _1{});
    } else {
      auto gQ =
          make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(Q.data_ptr())),
                      make_shape(total_q_rows, Int<kHeadDim>{}),
                      make_stride(Int<kHeadDim>{}, _1{}));
      return make_tma_copy(SM90_TMA_LOAD{}, gQ, SmemLayoutQ{},
                           Shape<Int<kBr>, Int<kQKDChunk>>{}, _1{});
    }
  };
  auto make_tma_k = [&](auto kv_c) {
    if constexpr (decltype(kv_c)::value) {
      auto gK =
          make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(K.data_ptr())),
                      make_shape((long)Nb * Nkv, (long)Nh_kv * kHeadDim),
                      make_stride(K.stride(2), _1{}));
      return make_tma_copy(SM90_TMA_LOAD{}, gK, SmemLayoutK{},
                           Shape<Int<kBc>, Int<kQKDChunk>>{}, _1{});
    } else {
      auto gK =
          make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(K.data_ptr())),
                      make_shape(total_kv_rows, Int<kHeadDim>{}),
                      make_stride(Int<kHeadDim>{}, _1{}));
      return make_tma_copy(SM90_TMA_LOAD{}, gK, SmemLayoutK{},
                           Shape<Int<kBc>, Int<kQKDChunk>>{}, _1{});
    }
  };
  auto make_tma_v = [&](auto kv_c) {
    if constexpr (decltype(kv_c)::value) {
      auto gV =
          make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(V.data_ptr())),
                      make_shape((long)Nb * Nkv, (long)Nh_kv * kHeadDim),
                      make_stride(V.stride(2), _1{}));
      return make_tma_copy(SM90_TMA_LOAD{}, gV, SmemLayoutV{},
                           Shape<Int<kBc>, Int<kVDChunk>>{}, _1{});
    } else {
      auto gV =
          make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(V.data_ptr())),
                      make_shape(total_kv_rows, Int<kHeadDim>{}),
                      make_stride(Int<kHeadDim>{}, _1{}));
      return make_tma_copy(SM90_TMA_LOAD{}, gV, SmemLayoutV{},
                           Shape<Int<kBc>, Int<kVDChunk>>{}, _1{});
    }
  };

  constexpr int kQTileBytes = kBr * kQKDChunk * sizeof(Element);
  constexpr int kKTileBytes = kBc * kQKDChunk * sizeof(Element);
  constexpr int kVTileBytes = kBc * kVDChunk * sizeof(Element);
  const int kBaseSmemBytes = kStagesQK * kQTileBytes + kStagesQK * kKTileBytes +
                             kStagesPV * kVTileBytes;

  // PC-0 bias tile: classify the mask shape; fall back to the gmem-direct
  // path when the tile exceeds the smem budget. The TMA box must be fully
  // static (cute's vectorization inference rejects dynamic modes), so mode
  // and dtype become template splits; the direct fallback reuses the
  // row-vec 2B descriptor shape as a never-issued dummy.
  FfpaBiasTilePlan bias_plan;
  if (has_attn_bias) {
    FfpaBiasParams bias_p{attn_bias_ptr,      attn_bias_dtype,
                          attn_bias_stride_b, attn_bias_stride_h,
                          attn_bias_stride_m, attn_bias_stride_n};
    bias_plan = ffpa_bias_tile_plan_of(bias_p, Nb, Nh, Nq, Nkv);
  }
  // sm_120 max opt-in dynamic smem per block is 99KB (101376B), which also
  // caps the dense bias tile next to the QK/V buffers. Row-broadcast keeps
  // a double buffer (tiny tile; hides the TMA latency, see split_d.cuh) —
  // the stages must mirror the kernel's kBiasStages so the dynamic smem
  // covers every slot.
  constexpr int kSmemBudgetBytes = 99 * 1024;
  const int bias_stages = (bias_plan.mode == 2) ? 2 : 1;
  if (kBaseSmemBytes + bias_plan.tile_bytes(kBr, kBc, bias_stages) >
      kSmemBudgetBytes)
    bias_plan.mode = 0;
  // Row-broadcast resident: the whole [1,Nkv] vector fits past the QK/V
  // buffers (16B-aligned) — one plain load before the kv loop removes every
  // per-tile bias TMA/barrier from the hot path (mode 3). The upgrade must
  // not trade blocks/SM for the removed barrier: keep the smem-driven
  // occupancy of the base layout (101376B opt-in per block; measured on
  // fp8 m4n2 where mode 3 dropped 3 CTA/SM to 1 and ran 4.9% slower).
  // The resident bytes are padded to a whole kBc tile: tail kv tiles' bias
  // injection reads tile-local offsets < kBc unclamped, so the pad (zero-
  // filled by the resident load) keeps every read inside the allocation
  // while the hot path stays instruction-identical to the no-guard build.
  if (bias_plan.mode == 2) {
    const long long kv_pad = (Nkv + kBc - 1) / kBc * kBc;
    const long long base_align = (kBaseSmemBytes + 15) & ~15;
    const long long resident = base_align + kv_pad * bias_plan.elem_size;
    if (resident <= kSmemBudgetBytes &&
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
            ? (kBiasModeT == 1 ? (int64_t)attn_bias_stride_m * (kBias4B ? 2 : 1)
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
      if constexpr (kBiasModeT == 1)
        return make_layout(Shape<Int<kBr>, Int<bias_cols>>{},
                           Stride<Int<bias_cols>, _1>{});
      else
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
  int kSmemBytes =
      ((kBaseSmemBytes + 15) & ~15) +
      (int)((bias_plan.mode == 3)
                ? ((long long)(Nkv + kBc - 1) / kBc * kBc) * bias_plan.elem_size
                : bias_plan.tile_bytes(kBr, kBc, bias_stages));
  // PC-14 dropout bitmap: [kBr,kBc] keep-bits x2 stages past the bias area,
  // generated by the 256 threads one tile ahead (see split_d.cuh). The env
  // escapes to the inline per-element Philox path; per-call read. The
  // occupancy guard mirrors mode 3: the extra bytes must not drop the
  // blocks/SM class (the launcher's 16B rounding is >= the kernel's 8B
  // bitmap_base padding in u16 units, so the write never overruns).
#ifdef ENABLE_FFPA_FP16_BUILD_DEBUG
  bool dropout_bitmap_on =
      has_dropout && getenv("FFPA_DROPOUT_BITMAP_DISABLE") == nullptr;
#else
  bool dropout_bitmap_on = has_dropout;
#endif
  constexpr int kBitmapBytes = kBr * kBc / 8 * 2;
  if (dropout_bitmap_on) {
    const long long total = (((long long)kSmemBytes + 15) & ~15) + kBitmapBytes;
    const long long base16 = ((long long)kSmemBytes + 15) & ~15;
    if (total > kSmemBudgetBytes || 101376 / total < 101376 / base16)
      dropout_bitmap_on = false;
    else
      kSmemBytes = (int)total;
  }

  float* softmax_lse_ptr =
      softmax_lse.numel() > 0 ? softmax_lse.data_ptr<float>() : nullptr;
  auto O_ptr = reinterpret_cast<Element*>(O.data_ptr());

  const auto run = [&](auto tma_q, auto tma_k, auto tma_v, auto q_c,
                       auto kv_c) {
    constexpr bool kNhdQ = decltype(q_c)::value;
    constexpr bool kNhdKV = decltype(kv_c)::value;

    auto launch_variant = [&](auto kernel_func, auto tma_bias_sel) {
      cudaFuncSetAttribute(
          kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
      kernel_func<<<grid, block, kSmemBytes, stream>>>(
          tma_q, tma_k, tma_v, tma_o, tma_bias_sel, O_ptr, softmax_lse_ptr, Nq,
          Nkv, Nh, Nh_kv, scale, Tc, causal, total_q_rows, total_kv_rows,
          attn_bias_ptr, attn_bias_dtype, attn_bias_stride_b,
          attn_bias_stride_h, attn_bias_stride_m, attn_bias_stride_n,
          dropout_p_f, philox_seed_u, philox_offset_u, nhd_out,
          bias_plan.mode != 0 ? bias_plan.m_total : (long long)1,
          dropout_bitmap_on ? 1 : 0);
    };

    using TmaQ = decltype(tma_q);
    using TmaK = decltype(tma_k);
    using TmaV = decltype(tma_v);
    using TmaO = decltype(tma_o);
    using TmaBiasD16 = decltype(tma_bias_d16);
    using TmaBiasD32 = decltype(tma_bias_d32);
    using TmaBiasR16 = decltype(tma_bias_r16);
    using TmaBiasR32 = decltype(tma_bias_r32);
    const bool bias_f32 = (attn_bias_dtype == 3);
    // bias-bearing helper: mode/dtype are compile-time (TMA box type), the
    // dropout flag stays a runtime split inside.
    const auto launch_bd = [&](auto tma_bias_sel, auto mode_c, auto b4_c) {
      constexpr int kModeL = decltype(mode_c)::value;
      constexpr int kB4 = decltype(b4_c)::value;
      if (has_dropout)
        launch_variant(split_d_fwd_cute_sm120<Traits, TmaQ, TmaK, TmaV, TmaO,
                                              decltype(tma_bias_sel), kModeL,
                                              kB4, 1, 1, kNhdQ, kNhdKV>,
                       tma_bias_sel);
      else
        launch_variant(split_d_fwd_cute_sm120<Traits, TmaQ, TmaK, TmaV, TmaO,
                                              decltype(tma_bias_sel), kModeL,
                                              kB4, 1, 0, kNhdQ, kNhdKV>,
                       tma_bias_sel);
    };
    if (!has_attn_bias) {
      if (has_dropout)
        launch_variant(
            split_d_fwd_cute_sm120<Traits, TmaQ, TmaK, TmaV, TmaO, TmaBiasR16,
                                   0, 0, 0, 1, kNhdQ, kNhdKV>,
            tma_bias_r16);
      else
        launch_variant(
            split_d_fwd_cute_sm120<Traits, TmaQ, TmaK, TmaV, TmaO, TmaBiasR16,
                                   0, 0, 0, 0, kNhdQ, kNhdKV>,
            tma_bias_r16);
    } else if (bias_plan.mode == 1) {
      if (bias_f32)
        launch_bd(tma_bias_d32, std::integral_constant<int, 1>{},
                  std::integral_constant<int, 1>{});
      else
        launch_bd(tma_bias_d16, std::integral_constant<int, 1>{},
                  std::integral_constant<int, 0>{});
    } else if (bias_plan.mode == 2) {
      if (bias_f32)
        launch_bd(tma_bias_r32, std::integral_constant<int, 2>{},
                  std::integral_constant<int, 1>{});
      else
        launch_bd(tma_bias_r16, std::integral_constant<int, 2>{},
                  std::integral_constant<int, 0>{});
    } else if (bias_plan.mode == 3) {
      // resident row-vector: no TMA issue in-kernel, descriptor unused.
      if (bias_f32)
        launch_bd(tma_bias_r32, std::integral_constant<int, 3>{},
                  std::integral_constant<int, 1>{});
      else
        launch_bd(tma_bias_r16, std::integral_constant<int, 3>{},
                  std::integral_constant<int, 0>{});
    } else {
      launch_bd(tma_bias_r16, std::integral_constant<int, 0>{},
                std::integral_constant<int, 0>{});
    }
  };

  if (kv_nhd) {
    if (q_nhd)
      run(make_tma_q(std::true_type{}), make_tma_k(std::true_type{}),
          make_tma_v(std::true_type{}), std::true_type{}, std::true_type{});
    else
      run(make_tma_q(std::false_type{}), make_tma_k(std::true_type{}),
          make_tma_v(std::true_type{}), std::false_type{}, std::true_type{});
  } else {
    if (q_nhd)
      run(make_tma_q(std::true_type{}), make_tma_k(std::false_type{}),
          make_tma_v(std::false_type{}), std::true_type{}, std::false_type{});
    else
      run(make_tma_q(std::false_type{}), make_tma_k(std::false_type{}),
          make_tma_v(std::false_type{}), std::false_type{}, std::false_type{});
  }
}

// M4N2 launcher: kBr=64, kBc=64, atom_layout=(4,2,1). Uses
// FFPAAttnCuTeSplitDM4N2Traits (P SMEM roundtrip + cross-N-warp softmax).
// Dispatched for D>=512 to avoid M8N1's register spill (O=D/2 > 255).
template <typename kDataType, const int kHeadDim, const int kStage>
void launch_cute_fwd_split_d_m4n2_sm120(torch::Tensor Q, torch::Tensor K,
                                        torch::Tensor V, torch::Tensor O,
                                        torch::Tensor attn_bias,
                                        torch::Tensor softmax_lse, int causal,
                                        double softmax_scale, double dropout_p,
                                        int64_t philox_seed,
                                        int64_t philox_offset) {
  using namespace cute;

  constexpr int kBr = 64;
  constexpr int kBc = 64;
  constexpr int kQKDChunk = 64;
  constexpr int kVDChunk = 64;
  constexpr int kStagesQK = (kStage < 2 ? 2 : (kStage > 3 ? 3 : kStage));
  constexpr int kStagesPV = kStagesQK;
  constexpr int kNumThreads = 256;

  using Element = std::conditional_t<std::is_same_v<kDataType, __half>,
                                     cutlass::half_t, cutlass::bfloat16_t>;
  using Traits = ffpa_cute::FFPAAttnCuTeSplitDM4N2Traits<
      kHeadDim, kBr, kBc, kQKDChunk, kVDChunk, kStagesQK, kStagesPV, Element>;
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

  const bool has_attn_bias = attn_bias.numel() != 0;
  const bool has_dropout = dropout_p > 0.0;

  const void* attn_bias_ptr = nullptr;
  int attn_bias_dtype = 0;
  long long attn_bias_stride_b = 0;
  long long attn_bias_stride_h = 0;
  long long attn_bias_stride_m = 0;
  long long attn_bias_stride_n = 0;
  if (has_attn_bias) {
    TORCH_CHECK(attn_bias.is_cuda(),
                "ffpa_attn: attn_mask must be a CUDA tensor");
    TORCH_CHECK(attn_bias.device() == Q.device(),
                "ffpa_attn: attn_mask must be on the same device as Q/K/V");
    TORCH_CHECK(
        attn_bias.dim() == 4,
        "ffpa_attn: normalized attn_mask must be 4-D [B, Nh_q, Nq, Nkv]");
    TORCH_CHECK(attn_bias.size(0) == 1 || attn_bias.size(0) == Nb,
                "ffpa_attn: attn_mask batch dimension must be 1 or B");
    TORCH_CHECK(attn_bias.size(1) == 1 || attn_bias.size(1) == Nh,
                "ffpa_attn: attn_mask head dimension must be 1 or Nh_q");
    TORCH_CHECK(attn_bias.size(2) == 1 || attn_bias.size(2) == Nq,
                "ffpa_attn: attn_mask query dimension must be 1 or Nq");
    TORCH_CHECK(attn_bias.size(3) == 1 || attn_bias.size(3) == Nkv,
                "ffpa_attn: attn_mask kv dimension must be 1 or Nkv");
    attn_bias_ptr = attn_bias.data_ptr();
    if (attn_bias.scalar_type() == at::ScalarType::Half)
      attn_bias_dtype = 1;
    else if (attn_bias.scalar_type() == at::ScalarType::BFloat16)
      attn_bias_dtype = 2;
    else
      attn_bias_dtype = 3;
    attn_bias_stride_b =
        (attn_bias.size(0) == 1 && Nb > 1) ? 0 : attn_bias.stride(0);
    attn_bias_stride_h =
        (attn_bias.size(1) == 1 && Nh > 1) ? 0 : attn_bias.stride(1);
    attn_bias_stride_m =
        (attn_bias.size(2) == 1 && Nq > 1) ? 0 : attn_bias.stride(2);
    attn_bias_stride_n =
        (attn_bias.size(3) == 1 && Nkv > 1) ? 0 : attn_bias.stride(3);
  }
  const float dropout_p_f = static_cast<float>(dropout_p);
  const unsigned long long philox_seed_u =
      static_cast<unsigned long long>(philox_seed);
  const unsigned long long philox_offset_u =
      static_cast<unsigned long long>(philox_offset);

  const dim3 block(kNumThreads, 1, 1);
  const dim3 grid(utils::div_ceil(Nq, kBr), Nb * Nh, 1);

  const c10::cuda::OptionalCUDAGuard device_guard(Q.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  const int total_q_rows = Nb * Nh * Nq;
  const int total_kv_rows = Nb * Nh_kv * Nkv;

  // NHD (diffusers BNHD) permute views — including strided fused-QKV chunk
  // rows — are consumed natively via flat (B*N, H*D) TMA rows carrying the
  // tensor's own row stride (H*D packed; wider for strided views); O stays
  // BHND-packed (the caller allocates it packed and re-views).
  const bool q_nhd = ffpa_is_nhd_view(Q) || ffpa_is_strided_nhd(Q);
  if (ffpa_is_strided_nhd(Q))
    ffpa_check_strided_nhd_aligned(Q, "Q");
  if (!q_nhd)
    TORCH_CHECK(Q.stride(3) == 1 && Q.stride(2) == (long)kHeadDim &&
                    Q.stride(1) == (long)Nq * kHeadDim &&
                    Q.stride(0) == (long)Nh * Nq * kHeadDim,
                "ffpa_attn: Q must be BHND-contiguous or an NHD (BNHD) view");
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

  // O output TMA store descriptor: BHND flat [total_q_rows,kHeadDim] or
  // NHD (diffusers BNHD packed) [Nb*Nq, Nh*kHeadDim] with the head
  // selecting the column-tile group (kernel folds Nh_id*kDChunksV into
  // the v_chunk walk). Both branches use dynamic int64 extents/strides so
  // TmaO has a single type and the kernel takes a runtime nhd_out branch.
  const bool nhd_out = ffpa_is_nhd_view(O);
  auto gO =
      nhd_out
          ? make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(O.data_ptr())),
                        make_shape((int64_t)Nb * Nq, (int64_t)Nh * kHeadDim),
                        make_stride((int64_t)Nh * kHeadDim, _1{}))
          : make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(O.data_ptr())),
                        make_shape((int64_t)total_q_rows, (int64_t)kHeadDim),
                        make_stride((int64_t)kHeadDim, _1{}));
  auto tma_o = make_tma_copy(SM90_TMA_STORE{}, gO, SmemLayoutO{},
                             Shape<Int<kBr>, Int<kVDChunk>>{}, _1{});

  auto make_tma_q = [&](auto q_c) {
    if constexpr (decltype(q_c)::value) {
      auto gQ =
          make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(Q.data_ptr())),
                      make_shape((long)Nb * Nq, (long)Nh * kHeadDim),
                      make_stride(Q.stride(2), _1{}));
      return make_tma_copy(SM90_TMA_LOAD{}, gQ, SmemLayoutQ{},
                           Shape<Int<kBr>, Int<kQKDChunk>>{}, _1{});
    } else {
      auto gQ =
          make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(Q.data_ptr())),
                      make_shape(total_q_rows, Int<kHeadDim>{}),
                      make_stride(Int<kHeadDim>{}, _1{}));
      return make_tma_copy(SM90_TMA_LOAD{}, gQ, SmemLayoutQ{},
                           Shape<Int<kBr>, Int<kQKDChunk>>{}, _1{});
    }
  };
  auto make_tma_k = [&](auto kv_c) {
    if constexpr (decltype(kv_c)::value) {
      auto gK =
          make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(K.data_ptr())),
                      make_shape((long)Nb * Nkv, (long)Nh_kv * kHeadDim),
                      make_stride(K.stride(2), _1{}));
      return make_tma_copy(SM90_TMA_LOAD{}, gK, SmemLayoutK{},
                           Shape<Int<kBc>, Int<kQKDChunk>>{}, _1{});
    } else {
      auto gK =
          make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(K.data_ptr())),
                      make_shape(total_kv_rows, Int<kHeadDim>{}),
                      make_stride(Int<kHeadDim>{}, _1{}));
      return make_tma_copy(SM90_TMA_LOAD{}, gK, SmemLayoutK{},
                           Shape<Int<kBc>, Int<kQKDChunk>>{}, _1{});
    }
  };
  auto make_tma_v = [&](auto kv_c) {
    if constexpr (decltype(kv_c)::value) {
      auto gV =
          make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(V.data_ptr())),
                      make_shape((long)Nb * Nkv, (long)Nh_kv * kHeadDim),
                      make_stride(V.stride(2), _1{}));
      return make_tma_copy(SM90_TMA_LOAD{}, gV, SmemLayoutV{},
                           Shape<Int<kBc>, Int<kVDChunk>>{}, _1{});
    } else {
      auto gV =
          make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(V.data_ptr())),
                      make_shape(total_kv_rows, Int<kHeadDim>{}),
                      make_stride(Int<kHeadDim>{}, _1{}));
      return make_tma_copy(SM90_TMA_LOAD{}, gV, SmemLayoutV{},
                           Shape<Int<kBc>, Int<kVDChunk>>{}, _1{});
    }
  };

  // PC-0 bias tile: classify the mask shape; fall back to the gmem-direct
  // path when the tile exceeds the smem budget. The TMA box must be fully
  // static (cute's vectorization inference rejects dynamic modes), so mode
  // and dtype become template splits; the direct fallback reuses the
  // row-vec 2B descriptor shape as a never-issued dummy.
  FfpaBiasTilePlan bias_plan;
  if (has_attn_bias) {
    FfpaBiasParams bias_p{attn_bias_ptr,      attn_bias_dtype,
                          attn_bias_stride_b, attn_bias_stride_h,
                          attn_bias_stride_m, attn_bias_stride_n};
    bias_plan = ffpa_bias_tile_plan_of(bias_p, Nb, Nh, Nq, Nkv);
  }
  const int kBaseSmemBytes = Traits::kSmemElems * sizeof(Element);
  // Row-broadcast keeps a double buffer (tiny tile; hides the TMA latency,
  // see split_d_m4n2.cuh); the stages must mirror the kernel's kBiasStages
  // so the dynamic smem covers every slot.
  constexpr int kSmemBudgetBytes = 99 * 1024;
  const int bias_stages = (bias_plan.mode == 2) ? 2 : 1;
  if (kBaseSmemBytes + bias_plan.tile_bytes(kBr, kBc, bias_stages) >
      kSmemBudgetBytes)
    bias_plan.mode = 0;
  // Row-broadcast resident (mode 3): the upgrade must not trade blocks/SM
  // for the removed barrier -- keep the smem-driven occupancy of the base
  // layout (measured on fp8 m4n2: mode 3 dropped 3 CTA/SM to 1, 4.9%
  // slower than the gmem fallback). Resident bytes are padded to a whole
  // kBc tile so tail tiles' unclamped injection reads stay in-allocation
  // (pad is zero-filled by the resident load).
  if (bias_plan.mode == 2) {
    const long long kv_pad = (Nkv + kBc - 1) / kBc * kBc;
    const long long base_align = (kBaseSmemBytes + 15) & ~15;
    const long long resident = base_align + kv_pad * bias_plan.elem_size;
    if (resident <= kSmemBudgetBytes &&
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
            ? (kBiasModeT == 1 ? (int64_t)attn_bias_stride_m * (kBias4B ? 2 : 1)
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
      if constexpr (kBiasModeT == 1)
        return make_layout(Shape<Int<kBr>, Int<bias_cols>>{},
                           Stride<Int<bias_cols>, _1>{});
      else
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
  const int kSmemBytes =
      ((kBaseSmemBytes + 15) & ~15) +
      (int)((bias_plan.mode == 3)
                ? ((long long)(Nkv + kBc - 1) / kBc * kBc) * bias_plan.elem_size
                : bias_plan.tile_bytes(kBr, kBc, bias_stages));

  float* softmax_lse_ptr =
      softmax_lse.numel() > 0 ? softmax_lse.data_ptr<float>() : nullptr;
  auto O_ptr = reinterpret_cast<Element*>(O.data_ptr());

  const auto run = [&](auto tma_q, auto tma_k, auto tma_v, auto q_c,
                       auto kv_c) {
    constexpr bool kNhdQ = decltype(q_c)::value;
    constexpr bool kNhdKV = decltype(kv_c)::value;

    auto launch_variant = [&](auto kernel_func, auto tma_bias_sel) {
      cudaFuncSetAttribute(
          kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
      kernel_func<<<grid, block, kSmemBytes, stream>>>(
          tma_q, tma_k, tma_v, tma_o, tma_bias_sel, O_ptr, softmax_lse_ptr, Nq,
          Nkv, Nh, Nh_kv, scale, Tc, causal, total_q_rows, total_kv_rows,
          attn_bias_ptr, attn_bias_dtype, attn_bias_stride_b,
          attn_bias_stride_h, attn_bias_stride_m, attn_bias_stride_n,
          dropout_p_f, philox_seed_u, philox_offset_u, nhd_out,
          bias_plan.mode != 0 ? bias_plan.m_total : (long long)1);
    };

    using TmaQ = decltype(tma_q);
    using TmaK = decltype(tma_k);
    using TmaV = decltype(tma_v);
    using TmaO = decltype(tma_o);
    using TmaBiasD16 = decltype(tma_bias_d16);
    using TmaBiasD32 = decltype(tma_bias_d32);
    using TmaBiasR16 = decltype(tma_bias_r16);
    using TmaBiasR32 = decltype(tma_bias_r32);
    const bool bias_f32 = (attn_bias_dtype == 3);
    // bias-bearing helper: mode/dtype are compile-time (TMA box type), the
    // dropout flag stays a runtime split inside.
    const auto launch_bd = [&](auto tma_bias_sel, auto mode_c, auto b4_c) {
      constexpr int kModeL = decltype(mode_c)::value;
      constexpr int kB4 = decltype(b4_c)::value;
      if (has_dropout)
        launch_variant(
            split_d_m4n2_fwd_cute_sm120<Traits, TmaQ, TmaK, TmaV, TmaO,
                                        decltype(tma_bias_sel), kModeL, kB4, 1,
                                        1, kNhdQ, kNhdKV>,
            tma_bias_sel);
      else
        launch_variant(
            split_d_m4n2_fwd_cute_sm120<Traits, TmaQ, TmaK, TmaV, TmaO,
                                        decltype(tma_bias_sel), kModeL, kB4, 1,
                                        0, kNhdQ, kNhdKV>,
            tma_bias_sel);
    };
    if (!has_attn_bias) {
      if (has_dropout)
        launch_variant(
            split_d_m4n2_fwd_cute_sm120<Traits, TmaQ, TmaK, TmaV, TmaO,
                                        TmaBiasR16, 0, 0, 0, 1, kNhdQ, kNhdKV>,
            tma_bias_r16);
      else
        launch_variant(
            split_d_m4n2_fwd_cute_sm120<Traits, TmaQ, TmaK, TmaV, TmaO,
                                        TmaBiasR16, 0, 0, 0, 0, kNhdQ, kNhdKV>,
            tma_bias_r16);
    } else if (bias_plan.mode == 1) {
      if (bias_f32)
        launch_bd(tma_bias_d32, std::integral_constant<int, 1>{},
                  std::integral_constant<int, 1>{});
      else
        launch_bd(tma_bias_d16, std::integral_constant<int, 1>{},
                  std::integral_constant<int, 0>{});
    } else if (bias_plan.mode == 2) {
      if (bias_f32)
        launch_bd(tma_bias_r32, std::integral_constant<int, 2>{},
                  std::integral_constant<int, 1>{});
      else
        launch_bd(tma_bias_r16, std::integral_constant<int, 2>{},
                  std::integral_constant<int, 0>{});
    } else if (bias_plan.mode == 3) {
      // resident row-vector: no TMA issue in-kernel, descriptor unused.
      if (bias_f32)
        launch_bd(tma_bias_r32, std::integral_constant<int, 3>{},
                  std::integral_constant<int, 1>{});
      else
        launch_bd(tma_bias_r16, std::integral_constant<int, 3>{},
                  std::integral_constant<int, 0>{});
    } else {
      launch_bd(tma_bias_r16, std::integral_constant<int, 0>{},
                std::integral_constant<int, 0>{});
    }
  };

  if (kv_nhd) {
    if (q_nhd)
      run(make_tma_q(std::true_type{}), make_tma_k(std::true_type{}),
          make_tma_v(std::true_type{}), std::true_type{}, std::true_type{});
    else
      run(make_tma_q(std::false_type{}), make_tma_k(std::true_type{}),
          make_tma_v(std::true_type{}), std::false_type{}, std::true_type{});
  } else {
    if (q_nhd)
      run(make_tma_q(std::true_type{}), make_tma_k(std::false_type{}),
          make_tma_v(std::false_type{}), std::true_type{}, std::false_type{});
    else
      run(make_tma_q(std::false_type{}), make_tma_k(std::false_type{}),
          make_tma_v(std::false_type{}), std::false_type{}, std::false_type{});
  }
}

template <typename kDataType, const int kHeadDim, const int kStage>
void launch_cute_fwd_persist_d_sm120(torch::Tensor Q, torch::Tensor K,
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

  const bool has_attn_bias = attn_bias.numel() != 0;
  const bool has_dropout = dropout_p > 0.0;

  const void* attn_bias_ptr = nullptr;
  int attn_bias_dtype = 0;
  long long attn_bias_stride_b = 0, attn_bias_stride_h = 0,
            attn_bias_stride_m = 0, attn_bias_stride_n = 0;
  if (has_attn_bias) {
    TORCH_CHECK(attn_bias.is_cuda() && attn_bias.device() == Q.device());
    TORCH_CHECK(attn_bias.dim() == 4);
    attn_bias_ptr = attn_bias.data_ptr();
    if (attn_bias.scalar_type() == at::ScalarType::Half)
      attn_bias_dtype = 1;
    else if (attn_bias.scalar_type() == at::ScalarType::BFloat16)
      attn_bias_dtype = 2;
    else
      attn_bias_dtype = 3;
    attn_bias_stride_b =
        (attn_bias.size(0) == 1 && Nb > 1) ? 0 : attn_bias.stride(0);
    attn_bias_stride_h =
        (attn_bias.size(1) == 1 && Nh > 1) ? 0 : attn_bias.stride(1);
    attn_bias_stride_m =
        (attn_bias.size(2) == 1 && Nq > 1) ? 0 : attn_bias.stride(2);
    attn_bias_stride_n =
        (attn_bias.size(3) == 1 && Nkv > 1) ? 0 : attn_bias.stride(3);
  }
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
  if (has_attn_bias) {
    FfpaBiasParams bias_p{attn_bias_ptr,      attn_bias_dtype,
                          attn_bias_stride_b, attn_bias_stride_h,
                          attn_bias_stride_m, attn_bias_stride_n};
    bias_plan = ffpa_bias_tile_plan_of(bias_p, Nb, Nh, Nq, Nkv);
  }
  const int kBaseSmemBytes = Traits::kSmemElems * sizeof(Element);
  // Independent of the K/V stage budget above (kSmemBudgetBytes drives
  // kMaxStages): the bias tile shares the same 99KB per-block opt-in cap.
  // The tile reuses the Q-persist area (Q is s2r'd into regs before the kv
  // loop, see persist_d.cuh), so only the part beyond kQPersistBytes adds
  // smem. bias_stages mirrors the kernel's kBiasStages (double-buffer when
  // the Q area holds two tiles).
  constexpr int kBiasSmemBudgetBytes = 99 * 1024;
  const long long tile_u16_1 = (long long)kBr * kBc * (bias_plan.elem_size / 2);
  const int bias_stages =
      ((long long)kQPersistBytes / 2 >= 2 * tile_u16_1) ? 2 : 1;
  long long bias_extra =
      std::max(0LL, bias_plan.tile_bytes(kBr, kBc, bias_stages) -
                        (long long)kQPersistBytes);
  if (bias_plan.mode == 1) {
    // Mirror persist_d.cuh's ceil-split segment layout: the tail TMA
    // segment still writes a full box, so the extra area must use the
    // same ceil arithmetic. Exact sizing under-reserves (D=96 dense f32
    // by 16KB) and the tail writes past the allocation into whatever
    // follows — now the dropout bitmap. Exact equality with the old
    // formula holds because cols_u16 divides q_u16 (kBc is a power of
    // two and D a multiple of 32).
    const long long cols_u16 = (long long)kBc * (bias_plan.elem_size / 2);
    const long long q_u16 = (long long)kQPersistBytes / 2;
    const long long box_rows =
        ((long long)kBr * cols_u16 > q_u16) ? q_u16 / cols_u16 : (long long)kBr;
    const long long segs = ((long long)kBr + box_rows - 1) / box_rows;
    bias_extra =
        std::max(0LL, (long long)bias_stages * segs * box_rows * cols_u16 * 2 -
                          (long long)kQPersistBytes);
  }
  if (kBaseSmemBytes + bias_extra > kBiasSmemBudgetBytes) {
    // Demoted to gmem-direct: no smem tile at all (the dummy descriptor
    // never issues), so the tail pad must drop out of the allocation or
    // the opt-in size itself would exceed the cap.
    bias_plan.mode = 0;
    bias_extra = 0;
  }
  // PC-14 dropout bitmap: [kBr,kBc] keep-bits x2 stages (ping-pong) past
  // the bias extra area, generated by the consumer threads one tile ahead
  // (inside the K/V TMA wait window). The env escape (debug builds only)
  // turns it off (A/B -> inline per-element Philox) and is read per call so
  // toggling mid-process works. kBc=32 (D=192/256) can't use the half-row
  // scheme (a row must span an even number of 32-bit words) and stays
  // inline.
#ifdef ENABLE_FFPA_FP16_BUILD_DEBUG
  bool dropout_bitmap_on = has_dropout && kBc >= 64 &&
                           getenv("FFPA_DROPOUT_BITMAP_DISABLE") == nullptr;
#else
  bool dropout_bitmap_on = has_dropout && kBc >= 64;
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
            ? (kBiasModeT == 1 ? (int64_t)attn_bias_stride_m * (kBias4B ? 2 : 1)
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
          attn_bias_ptr, attn_bias_dtype, attn_bias_stride_b,
          attn_bias_stride_h, attn_bias_stride_m, attn_bias_stride_n,
          dropout_p_f, philox_seed_u, philox_offset_u, nhd_out,
          bias_plan.mode != 0 ? bias_plan.m_total : (long long)1,
          dropout_bitmap_on ? 1 : 0);
    };

    using TmaQ = decltype(tma_q);
    using TmaK = decltype(tma_k);
    using TmaV = decltype(tma_v);
    using TmaO = decltype(tma_o);
    using TmaBiasD16 = decltype(tma_bias_d16);
    using TmaBiasD32 = decltype(tma_bias_d32);
    using TmaBiasR16 = decltype(tma_bias_r16);
    using TmaBiasR32 = decltype(tma_bias_r32);
    const bool bias_f32 = (attn_bias_dtype == 3);
    // bias-bearing helper: mode/dtype are compile-time (TMA box type), the
    // dropout flag stays a runtime split inside.
    const auto launch_bd = [&](auto tma_bias_sel, auto mode_c, auto b4_c) {
      constexpr int kModeL = decltype(mode_c)::value;
      constexpr int kB4 = decltype(b4_c)::value;
      if (has_dropout)
        launch_variant(
            persist_d_ws_fwd_cute_sm120<Traits, TmaQ, TmaK, TmaV, TmaO,
                                        decltype(tma_bias_sel), kModeL, kB4, 1,
                                        1, kNhdKV, kNhdQ>,
            tma_bias_sel);
      else
        launch_variant(
            persist_d_ws_fwd_cute_sm120<Traits, TmaQ, TmaK, TmaV, TmaO,
                                        decltype(tma_bias_sel), kModeL, kB4, 1,
                                        0, kNhdKV, kNhdQ>,
            tma_bias_sel);
    };
    if (!has_attn_bias) {
      if (has_dropout)
        launch_variant(
            persist_d_ws_fwd_cute_sm120<Traits, TmaQ, TmaK, TmaV, TmaO,
                                        TmaBiasR16, 0, 0, 0, 1, kNhdKV, kNhdQ>,
            tma_bias_r16);
      else
        launch_variant(
            persist_d_ws_fwd_cute_sm120<Traits, TmaQ, TmaK, TmaV, TmaO,
                                        TmaBiasR16, 0, 0, 0, 0, kNhdKV, kNhdQ>,
            tma_bias_r16);
    } else if (bias_plan.mode == 1) {
      if (bias_f32)
        launch_bd(tma_bias_d32, std::integral_constant<int, 1>{},
                  std::integral_constant<int, 1>{});
      else
        launch_bd(tma_bias_d16, std::integral_constant<int, 1>{},
                  std::integral_constant<int, 0>{});
    } else if (bias_plan.mode == 2) {
      if (bias_f32)
        launch_bd(tma_bias_r32, std::integral_constant<int, 2>{},
                  std::integral_constant<int, 1>{});
      else
        launch_bd(tma_bias_r16, std::integral_constant<int, 2>{},
                  std::integral_constant<int, 0>{});
    } else {
      launch_bd(tma_bias_r16, std::integral_constant<int, 0>{},
                std::integral_constant<int, 0>{});
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

#endif  // ENABLE_FFPA_TMA_EXT

template <typename kDataType, const int kHeadDim, const int kStage,
          const int kQKDChunk, const int kVDChunk>
void launch_cute_fwd_split_d_sm80(torch::Tensor Q, torch::Tensor K,
                                  torch::Tensor V, torch::Tensor O,
                                  torch::Tensor attn_bias,
                                  torch::Tensor softmax_lse, int causal,
                                  double softmax_scale, double dropout_p,
                                  int64_t philox_seed, int64_t philox_offset) {
  using namespace cute;

  constexpr int kBr = 128;
  constexpr int kBc = 128;

  constexpr int kNumThreads = kBr / 16 * 32;

  using Element = std::conditional_t<std::is_same_v<kDataType, __half>,
                                     cutlass::half_t, cutlass::bfloat16_t>;
  constexpr int kStagesQK = kStage;
  constexpr int kStagesPV = kStagesQK;
  using Traits = ffpa_cute::FFPAAttnCuTeSplitDTraits<
      kHeadDim, kBr, kBc, kQKDChunk, kVDChunk, kStagesQK, kStagesPV, Element>;

  constexpr int kQTileBytes = kBr * kQKDChunk * sizeof(Element);
  constexpr int kKTileBytes = kBc * kQKDChunk * sizeof(Element);
  constexpr int kVTileBytes = kBc * kVDChunk * sizeof(Element);
  constexpr int kSmemPerStage = kQTileBytes + kKTileBytes + kVTileBytes;

  int max_smem_optin = 0;
  cudaDeviceGetAttribute(&max_smem_optin,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin,
                         Q.device().index());
  TORCH_CHECK(kStagesQK * kSmemPerStage <= max_smem_optin,
              "ffpa_attn: CuTe kernel requires ", kStagesQK * kSmemPerStage,
              " bytes smem (stages=", kStagesQK, ", chunk=", kQKDChunk, "/",
              kVDChunk, ") but device supports ", max_smem_optin,
              " bytes opt-in smem");

  const int Nb = Q.size(0);
  const int Nh = Q.size(1);
  const int Nh_kv = K.size(1);
  const int Nq = Q.size(2);
  const int Nkv = K.size(2);
  const int Tc = utils::div_ceil(Nkv, kBc);
  const float scale = static_cast<float>(softmax_scale);

  const bool has_attn_bias = attn_bias.numel() != 0;
  const bool has_dropout = dropout_p > 0.0;

  const void* attn_bias_ptr = nullptr;
  int attn_bias_dtype = 0;
  long long attn_bias_stride_b = 0;
  long long attn_bias_stride_h = 0;
  long long attn_bias_stride_m = 0;
  long long attn_bias_stride_n = 0;
  if (has_attn_bias) {
    TORCH_CHECK(attn_bias.is_cuda(),
                "ffpa_attn: attn_mask must be a CUDA tensor");
    TORCH_CHECK(attn_bias.device() == Q.device(),
                "ffpa_attn: attn_mask must be on the same device as Q/K/V");
    TORCH_CHECK(
        attn_bias.dim() == 4,
        "ffpa_attn: normalized attn_mask must be 4-D [B, Nh_q, Nq, Nkv]");
    TORCH_CHECK(attn_bias.size(0) == 1 || attn_bias.size(0) == Nb,
                "ffpa_attn: attn_mask batch dimension must be 1 or B");
    TORCH_CHECK(attn_bias.size(1) == 1 || attn_bias.size(1) == Nh,
                "ffpa_attn: attn_mask head dimension must be 1 or Nh_q");
    TORCH_CHECK(attn_bias.size(2) == 1 || attn_bias.size(2) == Nq,
                "ffpa_attn: attn_mask query dimension must be 1 or Nq");
    TORCH_CHECK(attn_bias.size(3) == 1 || attn_bias.size(3) == Nkv,
                "ffpa_attn: attn_mask kv dimension must be 1 or Nkv");
    attn_bias_ptr = attn_bias.data_ptr();
    if (attn_bias.scalar_type() == at::ScalarType::Half)
      attn_bias_dtype = 1;
    else if (attn_bias.scalar_type() == at::ScalarType::BFloat16)
      attn_bias_dtype = 2;
    else
      attn_bias_dtype = 3;
    attn_bias_stride_b =
        (attn_bias.size(0) == 1 && Nb > 1) ? 0 : attn_bias.stride(0);
    attn_bias_stride_h =
        (attn_bias.size(1) == 1 && Nh > 1) ? 0 : attn_bias.stride(1);
    attn_bias_stride_m =
        (attn_bias.size(2) == 1 && Nq > 1) ? 0 : attn_bias.stride(2);
    attn_bias_stride_n =
        (attn_bias.size(3) == 1 && Nkv > 1) ? 0 : attn_bias.stride(3);
  }
  const float dropout_p_f = static_cast<float>(dropout_p);
  const unsigned long long philox_seed_u =
      static_cast<unsigned long long>(philox_seed);
  const unsigned long long philox_offset_u =
      static_cast<unsigned long long>(philox_offset);

  // PC-0 bias tile (cp.async sm80 path): shape classification only — no TMA
  // descriptor here; the loader is in-kernel vectorized global loads, so the
  // plan only gates on the smem budget.
  FfpaBiasTilePlan bias_plan;
  if (has_attn_bias) {
    FfpaBiasParams bias_p{attn_bias_ptr,      attn_bias_dtype,
                          attn_bias_stride_b, attn_bias_stride_h,
                          attn_bias_stride_m, attn_bias_stride_n};
    bias_plan = ffpa_bias_tile_plan_of(bias_p, Nb, Nh, Nq, Nkv);
  }
  const int kBaseSmemBytes = kStagesQK * kSmemPerStage;
  if (kBaseSmemBytes + bias_plan.tile_bytes(kBr, kBc, 2) > max_smem_optin)
    bias_plan.mode = 0;
  int kSmemBytes = kBaseSmemBytes + (int)bias_plan.tile_bytes(kBr, kBc, 2);
  // PC-14 dropout bitmap: [kBr,kBc] keep-bits x2 stages past the bias area
  // (same layout/gating as the sm120 split_d launcher; env escape per call,
  // debug builds only).
#ifdef ENABLE_FFPA_FP16_BUILD_DEBUG
  bool dropout_bitmap_on =
      has_dropout && getenv("FFPA_DROPOUT_BITMAP_DISABLE") == nullptr;
#else
  bool dropout_bitmap_on = has_dropout;
#endif
  constexpr int kBitmapBytes = kBr * kBc / 8 * 2;
  if (dropout_bitmap_on) {
    const long long total = (((long long)kSmemBytes + 15) & ~15) + kBitmapBytes;
    const long long base16 = ((long long)kSmemBytes + 15) & ~15;
    if (total > max_smem_optin || 101376 / total < 101376 / base16)
      dropout_bitmap_on = false;
    else
      kSmemBytes = (int)total;
  }

  const dim3 block(kNumThreads, 1, 1);
  const dim3 grid(utils::div_ceil(Nq, kBr), Nb * Nh, 1);

  const c10::cuda::OptionalCUDAGuard device_guard(Q.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  float* softmax_lse_ptr =
      softmax_lse.numel() > 0 ? softmax_lse.data_ptr<float>() : nullptr;
  auto Q_ptr = reinterpret_cast<Element*>(Q.data_ptr());
  auto K_ptr = reinterpret_cast<Element*>(K.data_ptr());
  auto V_ptr = reinterpret_cast<Element*>(V.data_ptr());
  auto O_ptr = reinterpret_cast<Element*>(O.data_ptr());

  auto launch_variant = [&](auto kernel_func) {
    cudaFuncSetAttribute(
        kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
    kernel_func<<<grid, block, kSmemBytes, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, softmax_lse_ptr, Nq, Nkv, Nh, Nh_kv, scale,
        Tc, causal, attn_bias_ptr, attn_bias_dtype, attn_bias_stride_b,
        attn_bias_stride_h, attn_bias_stride_m, attn_bias_stride_n, dropout_p_f,
        philox_seed_u, philox_offset_u, bias_plan.mode,
        bias_plan.mode != 0 ? bias_plan.m_total : (long long)1,
        dropout_bitmap_on ? 1 : 0);
  };

  if (has_attn_bias && has_dropout) {
    launch_variant(split_d_fwd_cute_sm80<Traits, kStagesQK, kStagesPV, 1, 1>);
  } else if (has_attn_bias) {
    launch_variant(split_d_fwd_cute_sm80<Traits, kStagesQK, kStagesPV, 1, 0>);
  } else if (has_dropout) {
    launch_variant(split_d_fwd_cute_sm80<Traits, kStagesQK, kStagesPV, 0, 1>);
  } else {
    launch_variant(split_d_fwd_cute_sm80<Traits, kStagesQK, kStagesPV, 0, 0>);
  }
}

#endif  // ENABLE_FFPA_CUTE_EXT

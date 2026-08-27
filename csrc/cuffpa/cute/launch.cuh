#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <optional>
#include "common.cuh"
#ifdef ENABLE_FFPA_CUTE_EXT
#include "cute/sm_80/split_d.cuh"
#ifdef ENABLE_FFPA_TMA_EXT
#include "cute/sm_120/split_d.cuh"
#include "cute/sm_120/persist_d.cuh"
#include "cute/sm_120/split_d_m4n2.cuh"
#include "cute/fp8/quantize_fp8.cuh"
#include "cute/fp8/smooth_k.cuh"
#include "cute/fp8/sm_120/persist_d.cuh"
#include "cute/fp8/sm_120/split_d.cuh"
#include "cute/fp8/sm_120/split_d_m4n2.cuh"
#include "cute/fp4/quantize_fp4.cuh"
#include "cute/hadamard.cuh"
#include "cute/fp4/delta_s.cuh"
#include "cute/fp4/sm_120/persist_d.cuh"
#include "cute/fp4/sm_120/split_d.cuh"
#include "cute/fp4/sm_120/split_d_m4n2.cuh"
#endif
#endif

#ifdef ENABLE_FFPA_CUTE_EXT
#ifdef ENABLE_FFPA_TMA_EXT
// NHD (diffusers BNHD) input detection: a [B, H, N, D]-shaped tensor that is
// a zero-copy permute view of a packed [B, N, H, D] tensor, i.e.
// strides (N*H*D, D, H*D, 1). Returns false for BHND-packed tensors.
// B == 1 makes stride(0) an ignored leftover (CP comm primitives leave the
// pre-permute value), so it is exempt from the exact check.
static inline bool ffpa_is_nhd_view(const torch::Tensor& X) {
  const long B = X.size(0), H = X.size(1), N = X.size(2), D = X.size(3);
  return X.dim() == 4 && X.stride(3) == 1 && X.stride(2) == H * D &&
         X.stride(1) == D && (B <= 1 || X.stride(0) == N * H * D);
}

// BHND-packed detection: contiguous [B, H, N, D] strides.
static inline bool ffpa_is_bhnd_packed(const torch::Tensor& X) {
  const long B = X.size(0), H = X.size(1), N = X.size(2), D = X.size(3);
  return X.dim() == 4 && X.stride(3) == 1 && X.stride(2) == D &&
         X.stride(1) == N * D && X.stride(0) == H * N * D;
}

// Strided-NHD predicate: an NHD-family [B, H, N, D] view whose row stride
// exceeds H*D (fused-QKV interleaved chunk layouts) — neither BHND-packed
// nor a packed-NHD permute view. stride(2) >= H*D excludes negative strides
// and head-overlapping rows; stride(0) > 0 excludes reversed batches
// (negative strides can still be %16-aligned).
static inline bool ffpa_is_strided_nhd(const torch::Tensor& X) {
  const long B = X.size(0), H = X.size(1), N = X.size(2), D = X.size(3);
  return X.dim() == 4 && X.stride(3) == 1 && X.stride(1) == D &&
         X.stride(2) >= H * D && !ffpa_is_nhd_view(X) &&
         (B <= 1 || (X.stride(0) == X.stride(2) * N && X.stride(0) > 0));
}

// TMA consumers need 16B alignment on the base pointer and both row/batch
// strides of a strided-NHD input.
static inline void ffpa_check_strided_nhd_aligned(const torch::Tensor& X,
                                                  const char* name) {
  const long es = X.element_size();
  TORCH_CHECK(reinterpret_cast<uintptr_t>(X.data_ptr()) % 16 == 0 &&
                  (X.stride(2) * es) % 16 == 0 && (X.stride(0) * es) % 16 == 0,
              "ffpa_attn: strided NHD ", name,
              " requires a 16B-aligned data_ptr and 16B-aligned row/batch "
              "strides (elemsize ",
              es, ", strides ", X.stride(0), ",", X.stride(1), ",", X.stride(2),
              ",1)");
}

// Fp8InputLayout from a [B, H, N, D] tensor's strides. Accepts BHND-packed
// and NHD-view; anything else (arbitrary strides) is rejected unless
// allow_strided_rows (persist-D opt-in): NHD-family views whose row stride
// exceeds H*D (fused-QKV chunk layouts, e.g. FLUX.2 single-stream V) are
// also accepted — the pre-kernels address rows through s_row, so any
// 16B-aligned positive row/batch stride is legal. Split-D/M4N2 keep the
// default strict gate.
static inline ffpa_fp8::Fp8InputLayout ffpa_layout_of(
    const torch::Tensor& X, long N, long D, bool allow_strided_rows = false) {
  const long B = X.size(0), H = X.size(1);
  TORCH_CHECK(X.dim() == 4 && X.stride(3) == 1,
              "ffpa_attn: Q/K/V must be 4-D with unit stride along D");
  if (ffpa_is_nhd_view(X))
    return {true, static_cast<int>(H), N * H * D, D, H * D};
  if (allow_strided_rows && ffpa_is_strided_nhd(X)) {
    ffpa_check_strided_nhd_aligned(X, "input");
    const long s_row = X.stride(2);
    const long s_batch = (B <= 1) ? N * s_row : X.stride(0);
    return {true, static_cast<int>(H), s_batch, D, s_row};
  }
  TORCH_CHECK(
      X.stride(2) == D && X.stride(1) == N * D && X.stride(0) == H * N * D,
      "ffpa_attn: Q/K/V must be BHND-contiguous or an NHD (BNHD) "
      "permute view, got strides (",
      X.stride(0), ",", X.stride(1), ",", X.stride(2), ",", X.stride(3), ")");
  return {false, 0, 0, N * D, D};
}

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

  // NHD (diffusers BNHD) permute views are consumed natively via flat
  // (B*N, H*D) TMA rows (head as a column tile); O stays BHND-packed (the
  // caller allocates it packed and re-views).
  const bool q_nhd = ffpa_is_nhd_view(Q);
  const bool kv_nhd = ffpa_is_nhd_view(K);
  if (kv_nhd) {
    TORCH_CHECK(ffpa_is_nhd_view(V),
                "ffpa_attn: K and V must share the same memory layout");
  } else {
    TORCH_CHECK(K.stride(3) == 1 && K.stride(2) == (long)kHeadDim &&
                    K.stride(1) == (long)Nkv * kHeadDim &&
                    K.stride(0) == (long)Nh_kv * Nkv * kHeadDim,
                "ffpa_attn: K must be BHND-contiguous or an NHD (BNHD) view");
  }
  if (!q_nhd)
    TORCH_CHECK(Q.stride(3) == 1 && Q.stride(2) == (long)kHeadDim &&
                    Q.stride(1) == (long)Nq * kHeadDim &&
                    Q.stride(0) == (long)Nh * Nq * kHeadDim,
                "ffpa_attn: Q must be BHND-contiguous or an NHD (BNHD) view");

  // O output TMA store descriptor: full O tensor [total_q_rows,kHeadDim],
  // same shape/stride as the BHND Q; per-head origin injected via
  // domain_offset in kernel. Direction = SM90_TMA_STORE (first arg);
  // swizzle auto-inferred from SmemLayoutO.
  auto gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(O.data_ptr())),
                        make_shape(total_q_rows, Int<kHeadDim>{}),
                        make_stride(Int<kHeadDim>{}, _1{}));
  auto tma_o = make_tma_copy(SM90_TMA_STORE{}, gO, SmemLayoutO{},
                             Shape<Int<kBr>, Int<kVDChunk>>{}, _1{});

  auto make_tma_q = [&](auto q_c) {
    if constexpr (decltype(q_c)::value) {
      auto gQ =
          make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(Q.data_ptr())),
                      make_shape((long)Nb * Nq, (long)Nh * kHeadDim),
                      make_stride((long)Nh * kHeadDim, _1{}));
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
                      make_stride((long)Nh_kv * kHeadDim, _1{}));
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
                      make_stride((long)Nh_kv * kHeadDim, _1{}));
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
  constexpr int kSmemBytes = kStagesQK * kQTileBytes + kStagesQK * kKTileBytes +
                             kStagesPV * kVTileBytes;

  float* softmax_lse_ptr =
      softmax_lse.numel() > 0 ? softmax_lse.data_ptr<float>() : nullptr;
  auto O_ptr = reinterpret_cast<Element*>(O.data_ptr());

  const auto run = [&](auto tma_q, auto tma_k, auto tma_v, auto q_c,
                       auto kv_c) {
    constexpr bool kNhdQ = decltype(q_c)::value;
    constexpr bool kNhdKV = decltype(kv_c)::value;

    auto launch_variant = [&](auto kernel_func) {
      cudaFuncSetAttribute(
          kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
      kernel_func<<<grid, block, kSmemBytes, stream>>>(
          tma_q, tma_k, tma_v, tma_o, O_ptr, softmax_lse_ptr, Nq, Nkv, Nh,
          Nh_kv, scale, Tc, causal, total_q_rows, total_kv_rows, attn_bias_ptr,
          attn_bias_dtype, attn_bias_stride_b, attn_bias_stride_h,
          attn_bias_stride_m, attn_bias_stride_n, dropout_p_f, philox_seed_u,
          philox_offset_u);
    };

    using TmaQ = decltype(tma_q);
    using TmaK = decltype(tma_k);
    using TmaV = decltype(tma_v);
    using TmaO = decltype(tma_o);
    if (has_attn_bias && has_dropout) {
      launch_variant(split_d_fwd_cute_sm120<Traits, TmaQ, TmaK, TmaV, TmaO, 1,
                                            1, kNhdQ, kNhdKV>);
    } else if (has_attn_bias) {
      launch_variant(split_d_fwd_cute_sm120<Traits, TmaQ, TmaK, TmaV, TmaO, 1,
                                            0, kNhdQ, kNhdKV>);
    } else if (has_dropout) {
      launch_variant(split_d_fwd_cute_sm120<Traits, TmaQ, TmaK, TmaV, TmaO, 0,
                                            1, kNhdQ, kNhdKV>);
    } else {
      launch_variant(split_d_fwd_cute_sm120<Traits, TmaQ, TmaK, TmaV, TmaO, 0,
                                            0, kNhdQ, kNhdKV>);
    }
  };

  if (kv_nhd)
    run(make_tma_q(std::true_type{}), make_tma_k(std::true_type{}),
        make_tma_v(std::true_type{}), std::true_type{}, std::true_type{});
  else
    run(make_tma_q(std::false_type{}), make_tma_k(std::false_type{}),
        make_tma_v(std::false_type{}), std::false_type{}, std::false_type{});
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

  // NHD (diffusers BNHD) permute views are consumed natively via flat
  // (B*N, H*D) TMA rows (head as a column tile); O stays BHND-packed (the
  // caller allocates it packed and re-views).
  const bool q_nhd = ffpa_is_nhd_view(Q);
  const bool kv_nhd = ffpa_is_nhd_view(K);
  if (kv_nhd) {
    TORCH_CHECK(ffpa_is_nhd_view(V),
                "ffpa_attn: K and V must share the same memory layout");
  } else {
    TORCH_CHECK(K.stride(3) == 1 && K.stride(2) == (long)kHeadDim &&
                    K.stride(1) == (long)Nkv * kHeadDim &&
                    K.stride(0) == (long)Nh_kv * Nkv * kHeadDim,
                "ffpa_attn: K must be BHND-contiguous or an NHD (BNHD) view");
  }
  if (!q_nhd)
    TORCH_CHECK(Q.stride(3) == 1 && Q.stride(2) == (long)kHeadDim &&
                    Q.stride(1) == (long)Nq * kHeadDim &&
                    Q.stride(0) == (long)Nh * Nq * kHeadDim,
                "ffpa_attn: Q must be BHND-contiguous or an NHD (BNHD) view");

  auto gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(O.data_ptr())),
                        make_shape(total_q_rows, Int<kHeadDim>{}),
                        make_stride(Int<kHeadDim>{}, _1{}));
  auto tma_o = make_tma_copy(SM90_TMA_STORE{}, gO, SmemLayoutO{},
                             Shape<Int<kBr>, Int<kVDChunk>>{}, _1{});

  auto make_tma_q = [&](auto q_c) {
    if constexpr (decltype(q_c)::value) {
      auto gQ =
          make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(Q.data_ptr())),
                      make_shape((long)Nb * Nq, (long)Nh * kHeadDim),
                      make_stride((long)Nh * kHeadDim, _1{}));
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
                      make_stride((long)Nh_kv * kHeadDim, _1{}));
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
                      make_stride((long)Nh_kv * kHeadDim, _1{}));
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

  constexpr int kSmemBytes = Traits::kSmemElems * sizeof(Element);

  float* softmax_lse_ptr =
      softmax_lse.numel() > 0 ? softmax_lse.data_ptr<float>() : nullptr;
  auto O_ptr = reinterpret_cast<Element*>(O.data_ptr());

  const auto run = [&](auto tma_q, auto tma_k, auto tma_v, auto q_c,
                       auto kv_c) {
    constexpr bool kNhdQ = decltype(q_c)::value;
    constexpr bool kNhdKV = decltype(kv_c)::value;

    auto launch_variant = [&](auto kernel_func) {
      cudaFuncSetAttribute(
          kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
      kernel_func<<<grid, block, kSmemBytes, stream>>>(
          tma_q, tma_k, tma_v, tma_o, O_ptr, softmax_lse_ptr, Nq, Nkv, Nh,
          Nh_kv, scale, Tc, causal, total_q_rows, total_kv_rows, attn_bias_ptr,
          attn_bias_dtype, attn_bias_stride_b, attn_bias_stride_h,
          attn_bias_stride_m, attn_bias_stride_n, dropout_p_f, philox_seed_u,
          philox_offset_u);
    };

    using TmaQ = decltype(tma_q);
    using TmaK = decltype(tma_k);
    using TmaV = decltype(tma_v);
    using TmaO = decltype(tma_o);
    if (has_attn_bias && has_dropout) {
      launch_variant(split_d_m4n2_fwd_cute_sm120<Traits, TmaQ, TmaK, TmaV, TmaO,
                                                 1, 1, kNhdQ, kNhdKV>);
    } else if (has_attn_bias) {
      launch_variant(split_d_m4n2_fwd_cute_sm120<Traits, TmaQ, TmaK, TmaV, TmaO,
                                                 1, 0, kNhdQ, kNhdKV>);
    } else if (has_dropout) {
      launch_variant(split_d_m4n2_fwd_cute_sm120<Traits, TmaQ, TmaK, TmaV, TmaO,
                                                 0, 1, kNhdQ, kNhdKV>);
    } else {
      launch_variant(split_d_m4n2_fwd_cute_sm120<Traits, TmaQ, TmaK, TmaV, TmaO,
                                                 0, 0, kNhdQ, kNhdKV>);
    }
  };

  if (kv_nhd)
    run(make_tma_q(std::true_type{}), make_tma_k(std::true_type{}),
        make_tma_v(std::true_type{}), std::true_type{}, std::true_type{});
  else
    run(make_tma_q(std::false_type{}), make_tma_k(std::false_type{}),
        make_tma_v(std::false_type{}), std::false_type{}, std::false_type{});
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

    constexpr int kSmemBytes = Traits::kSmemElems * sizeof(Element);

    float* softmax_lse_ptr =
        softmax_lse.numel() > 0 ? softmax_lse.data_ptr<float>() : nullptr;
    auto O_ptr = reinterpret_cast<Element*>(O.data_ptr());

    auto launch_variant = [&](auto kernel_func) {
      cudaFuncSetAttribute(
          kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
      kernel_func<<<grid, block, kSmemBytes, stream>>>(
          tma_q, tma_k, tma_v, tma_o, O_ptr, softmax_lse_ptr, Nq, Nkv, Nh,
          Nh_kv, scale, Tc, causal, total_q_rows, total_kv_rows, attn_bias_ptr,
          attn_bias_dtype, attn_bias_stride_b, attn_bias_stride_h,
          attn_bias_stride_m, attn_bias_stride_n, dropout_p_f, philox_seed_u,
          philox_offset_u, nhd_out);
    };

    using TmaQ = decltype(tma_q);
    using TmaK = decltype(tma_k);
    using TmaV = decltype(tma_v);
    using TmaO = decltype(tma_o);
    if (has_attn_bias && has_dropout) {
      launch_variant(persist_d_ws_fwd_cute_sm120<Traits, TmaQ, TmaK, TmaV, TmaO,
                                                 1, 1, kNhdKV, kNhdQ>);
    } else if (has_attn_bias) {
      launch_variant(persist_d_ws_fwd_cute_sm120<Traits, TmaQ, TmaK, TmaV, TmaO,
                                                 1, 0, kNhdKV, kNhdQ>);
    } else if (has_dropout) {
      launch_variant(persist_d_ws_fwd_cute_sm120<Traits, TmaQ, TmaK, TmaV, TmaO,
                                                 0, 1, kNhdKV, kNhdQ>);
    } else {
      launch_variant(persist_d_ws_fwd_cute_sm120<Traits, TmaQ, TmaK, TmaV, TmaO,
                                                 0, 0, kNhdKV, kNhdQ>);
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
  TORCH_CHECK(attn_bias.numel() == 0 && dropout_p == 0.0,
              "fp8 sm120 path does not support attn_bias/dropout");
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
  const bool pquant_per_row = getenv("FFPA_FP8_PQUANT_PER_ROW") != nullptr;
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
  const auto launch_kernel = [&](auto kernel) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         kSmemBytes);
    kernel<<<grid, block, kSmemBytes, stream>>>(
        tma_q, tma_k, tma_v, tma_o, O_ptr, softmax_lse_ptr,
        q_scale.data_ptr<float>(), k_scale.data_ptr<float>(),
        v_scale.data_ptr<float>(), Nq, Nkv, Nh, Nh_kv, scale, Tc, causal,
        total_q_rows, total_kv_rows, n_rb_q, n_rb_kv, q_start_row, km_f32_ptr,
        vm_kernel, nhd_out);
  };
  if (qk_per_thread) {
    // Per-thread QK quant (sage style): fragment-aligned dequant scales.
    if (v_per_channel && pv_acc_f16) {
      launch_kernel(ffpa_fp8::persist_d_ws_fwd_cute_fp8_sm120<
                    Traits, ElementO, TmaQ, TmaK, TmaV, TmaO, false, true, true,
                    true, reorg_free>);
    } else if (v_per_channel) {
      launch_kernel(ffpa_fp8::persist_d_ws_fwd_cute_fp8_sm120<
                    Traits, ElementO, TmaQ, TmaK, TmaV, TmaO, false, false,
                    true, true, reorg_free>);
    } else if (pv_acc_f16) {
      launch_kernel(ffpa_fp8::persist_d_ws_fwd_cute_fp8_sm120<
                    Traits, ElementO, TmaQ, TmaK, TmaV, TmaO, false, true,
                    false, true, reorg_free>);
    } else {
      launch_kernel(ffpa_fp8::persist_d_ws_fwd_cute_fp8_sm120<
                    Traits, ElementO, TmaQ, TmaK, TmaV, TmaO, false, false,
                    false, true, reorg_free>);
    }
  } else if (pquant_per_row) {
    launch_kernel(
        ffpa_fp8::persist_d_ws_fwd_cute_fp8_sm120<Traits, ElementO, TmaQ, TmaK,
                                                  TmaV, TmaO, true, false,
                                                  false, false, reorg_free>);
  } else if (v_per_channel && pv_acc_f16) {
    // Per-channel V + fp16 PV accumulator: sage-style per-D V scale plus the
    // f8f8f16 PV path that avoids the 22-bit f8f8f32 accumulator loss.
    launch_kernel(
        ffpa_fp8::persist_d_ws_fwd_cute_fp8_sm120<Traits, ElementO, TmaQ, TmaK,
                                                  TmaV, TmaO, false, true, true,
                                                  false, reorg_free>);
  } else if (v_per_channel) {
    // Per-channel V (sage-style): V per-D scale, P uses fixed 448; epilogue
    // dequants per-D. Targets real VLM/diffusion data with per-D outliers
    // (per-block V over-saturates them). See persist_d.cuh kVPerChannel.
    launch_kernel(
        ffpa_fp8::persist_d_ws_fwd_cute_fp8_sm120<Traits, ElementO, TmaQ, TmaK,
                                                  TmaV, TmaO, false, false,
                                                  true, false, reorg_free>);
  } else if (pv_acc_f16) {
    // f8f8f16 PV (fp16 MMA accumulator, absorbs to float o_acc each
    // kv_tile) avoids the 22-bit f8f8f32 accumulator loss on causal early
    // rows. See persist_d.cuh kPVAccF16.
    launch_kernel(
        ffpa_fp8::persist_d_ws_fwd_cute_fp8_sm120<Traits, ElementO, TmaQ, TmaK,
                                                  TmaV, TmaO, false, true,
                                                  false, false, reorg_free>);
  } else {
    launch_kernel(
        ffpa_fp8::persist_d_ws_fwd_cute_fp8_sm120<Traits, ElementO, TmaQ, TmaK,
                                                  TmaV, TmaO, false, false,
                                                  false, false, reorg_free>);
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
    if (ffpa_is_nhd_view(Q))
      Q = Q.contiguous();
    if (ffpa_is_nhd_view(K))
      K = K.contiguous();
    if (Q.size(3) < kHeadDim)
      V = torch::constant_pad_nd(V, {0, kHeadDim - Q.size(3)}, 0.0);
    Q = ffpa::apply_wht_qk_sm120<kDataType, kHeadDim>(Q);
    K = ffpa::apply_wht_qk_sm120<kDataType, kHeadDim>(K);
  }
  // NHD (diffusers BNHD) zero-copy views: the fp8 pre-kernels read the
  // original gmem through Fp8InputLayout strides. V must share K's layout.
  const ffpa_fp8::Fp8InputLayout Lq = ffpa_layout_of(Q, Q.size(2), Q.size(3));
  const ffpa_fp8::Fp8InputLayout Lkv = ffpa_layout_of(K, K.size(2), K.size(3));
  TORCH_CHECK(V.stride(0) == K.stride(0) && V.stride(1) == K.stride(1) &&
                  V.stride(2) == K.stride(2) && V.stride(3) == K.stride(3),
              "ffpa_attn: V must share K's memory layout");
  TORCH_CHECK(attn_bias.numel() == 0 && dropout_p == 0.0,
              "fp8 sm120 path does not support attn_bias/dropout");
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
        Lq, Lkv, stream, km_ptr, reorg_free, v_per_channel);
  } else {
    ffpa_fp8::launch_quantize_fp8_sm120<kDataType, kBr, kBc, kHeadDim, kQKInt8>(
        q_ptr, k_ptr, v_ptr, q8.data_ptr(), k8.data_ptr(),
        reinterpret_cast<__nv_fp8_e4m3*>(vt8.data_ptr()),
        q_scale.data_ptr<float>(), k_scale.data_ptr<float>(),
        v_scale_quant.data_ptr<float>(), Nb, Nh, Nh_kv, Nq, Nkv, Nkv_pad, D_og,
        Lq, Lkv, stream, km_ptr, reorg_free, v_per_channel);
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
          Nb, Nh_kv, Nkv, Nkv_pad, stream, D_og, v_r, reorg_free, &Lkv);
    } else {
      ffpa_fp8::launch_quantize_fp8_vt_perchannel_sm120<kDataType, kBr, kBc,
                                                        kHeadDim, false>(
          v_ptr, reinterpret_cast<__nv_fp8_e4m3*>(vt8.data_ptr()),
          v_scale.data_ptr<float>(), vm_ptr, v_partials_sum.data_ptr<float>(),
          v_partials_max.data_ptr<float>(), v_partials_min.data_ptr<float>(),
          Nb, Nh_kv, Nkv, Nkv_pad, stream, D_og, v_r, reorg_free, &Lkv);
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

  auto gO =
      make_tensor(make_gmem_ptr(reinterpret_cast<ElementO*>(O.data_ptr())),
                  make_shape(total_q_rows, Int<kHeadDim>{}),
                  make_stride(Int<kHeadDim>{}, _1{}));
  auto tma_o = make_tma_copy(SM90_TMA_STORE{}, gO, SmemLayoutO{},
                             Shape<Int<kBr>, Int<kVDChunk>>{}, _1{});

  constexpr int kSmemBytes = Traits::kSmemElems;
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
  auto launch_kernel = [&](auto kernel) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         kSmemBytes);
    kernel<<<grid, block, kSmemBytes, stream>>>(
        tma_q, tma_k, tma_v, tma_o, O_ptr, softmax_lse_ptr,
        q_scale.data_ptr<float>(), k_scale.data_ptr<float>(),
        v_scale.data_ptr<float>(), Nq, Nkv, Nh, Nh_kv, scale, Tc, causal,
        total_q_rows, total_kv_rows, n_rb_q, n_rb_kv, q_start_row, km_f32_ptr,
        vm_kernel);
  };
  if (qk_per_thread) {
    // Per-thread QK quant (sage style): fragment-aligned dequant scales.
    if (v_per_channel && pv_acc_f16) {
      launch_kernel(
          ffpa_fp8::split_d_fwd_cute_fp8_sm120<Traits, ElementO, TmaQ, TmaK,
                                               TmaV, TmaO, true, true, true,
                                               reorg_free>);
    } else if (v_per_channel) {
      launch_kernel(
          ffpa_fp8::split_d_fwd_cute_fp8_sm120<Traits, ElementO, TmaQ, TmaK,
                                               TmaV, TmaO, false, true, true,
                                               reorg_free>);
    } else if (pv_acc_f16) {
      launch_kernel(
          ffpa_fp8::split_d_fwd_cute_fp8_sm120<Traits, ElementO, TmaQ, TmaK,
                                               TmaV, TmaO, true, false, true,
                                               reorg_free>);
    } else {
      launch_kernel(
          ffpa_fp8::split_d_fwd_cute_fp8_sm120<Traits, ElementO, TmaQ, TmaK,
                                               TmaV, TmaO, false, false, true,
                                               reorg_free>);
    }
  } else if (v_per_channel && pv_acc_f16) {
    launch_kernel(
        ffpa_fp8::split_d_fwd_cute_fp8_sm120<Traits, ElementO, TmaQ, TmaK, TmaV,
                                             TmaO, true, true, false,
                                             reorg_free>);
  } else if (v_per_channel) {
    launch_kernel(
        ffpa_fp8::split_d_fwd_cute_fp8_sm120<Traits, ElementO, TmaQ, TmaK, TmaV,
                                             TmaO, false, true, false,
                                             reorg_free>);
  } else if (pv_acc_f16) {
    launch_kernel(
        ffpa_fp8::split_d_fwd_cute_fp8_sm120<Traits, ElementO, TmaQ, TmaK, TmaV,
                                             TmaO, true, false, false,
                                             reorg_free>);
  } else {
    launch_kernel(
        ffpa_fp8::split_d_fwd_cute_fp8_sm120<Traits, ElementO, TmaQ, TmaK, TmaV,
                                             TmaO, false, false, false,
                                             reorg_free>);
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
    if (ffpa_is_nhd_view(Q))
      Q = Q.contiguous();
    if (ffpa_is_nhd_view(K))
      K = K.contiguous();
    if (Q.size(3) < kHeadDim)
      V = torch::constant_pad_nd(V, {0, kHeadDim - Q.size(3)}, 0.0);
    Q = ffpa::apply_wht_qk_sm120<kDataType, kHeadDim>(Q);
    K = ffpa::apply_wht_qk_sm120<kDataType, kHeadDim>(K);
  }
  // NHD (diffusers BNHD) zero-copy views: the fp8 pre-kernels read the
  // original gmem through Fp8InputLayout strides. V must share K's layout.
  const ffpa_fp8::Fp8InputLayout Lq = ffpa_layout_of(Q, Q.size(2), Q.size(3));
  const ffpa_fp8::Fp8InputLayout Lkv = ffpa_layout_of(K, K.size(2), K.size(3));
  TORCH_CHECK(V.stride(0) == K.stride(0) && V.stride(1) == K.stride(1) &&
                  V.stride(2) == K.stride(2) && V.stride(3) == K.stride(3),
              "ffpa_attn: V must share K's memory layout");
  TORCH_CHECK(attn_bias.numel() == 0 && dropout_p == 0.0,
              "fp8 sm120 path does not support attn_bias/dropout");
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
        Lq, Lkv, stream, km_ptr, false, v_per_channel);
  } else {
    ffpa_fp8::launch_quantize_fp8_sm120<kDataType, kBr, kBc, kHeadDim, kQKInt8>(
        q_ptr, k_ptr, v_ptr, q8.data_ptr(), k8.data_ptr(),
        reinterpret_cast<__nv_fp8_e4m3*>(vt8.data_ptr()),
        q_scale.data_ptr<float>(), k_scale.data_ptr<float>(),
        v_scale_quant.data_ptr<float>(), Nb, Nh, Nh_kv, Nq, Nkv, Nkv_pad, D_og,
        Lq, Lkv, stream, km_ptr, false, v_per_channel);
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
          Nb, Nh_kv, Nkv, Nkv_pad, stream, D_og, v_r, /*perm_vt=*/false, &Lkv);
    } else {
      ffpa_fp8::launch_quantize_fp8_vt_perchannel_sm120<kDataType, kBr, kBc,
                                                        kHeadDim, false>(
          v_ptr, reinterpret_cast<__nv_fp8_e4m3*>(vt8.data_ptr()),
          v_scale.data_ptr<float>(), vm_ptr, v_partials_sum.data_ptr<float>(),
          v_partials_max.data_ptr<float>(), v_partials_min.data_ptr<float>(),
          Nb, Nh_kv, Nkv, Nkv_pad, stream, D_og, v_r, /*perm_vt=*/false, &Lkv);
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

  auto gO =
      make_tensor(make_gmem_ptr(reinterpret_cast<ElementO*>(O.data_ptr())),
                  make_shape(total_q_rows, Int<kHeadDim>{}),
                  make_stride(Int<kHeadDim>{}, _1{}));
  auto tma_o = make_tma_copy(SM90_TMA_STORE{}, gO, SmemLayoutO{},
                             Shape<Int<kBr>, Int<kVDChunk>>{}, _1{});

  constexpr int kSmemBytes = Traits::kSmemElems;
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
  auto launch_kernel = [&](auto kernel) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         kSmemBytes);
    kernel<<<grid, block, kSmemBytes, stream>>>(
        tma_q, tma_k, tma_v, tma_o, O_ptr, softmax_lse_ptr,
        q_scale.data_ptr<float>(), k_scale.data_ptr<float>(),
        v_scale.data_ptr<float>(), Nq, Nkv, Nh, Nh_kv, scale, Tc, causal,
        total_q_rows, total_kv_rows, n_rb_q, n_rb_kv, q_start_row, km_f32_ptr,
        vm_kernel);
  };
  if (qk_per_thread) {
    // Per-thread QK quant (sage style): fragment-aligned dequant scales.
    if (v_per_channel && pv_acc_f16) {
      launch_kernel(
          ffpa_fp8::split_d_m4n2_fwd_cute_fp8_sm120<
              Traits, ElementO, TmaQ, TmaK, TmaV, TmaO, true, true, true>);
    } else if (v_per_channel) {
      launch_kernel(
          ffpa_fp8::split_d_m4n2_fwd_cute_fp8_sm120<
              Traits, ElementO, TmaQ, TmaK, TmaV, TmaO, false, true, true>);
    } else if (pv_acc_f16) {
      launch_kernel(
          ffpa_fp8::split_d_m4n2_fwd_cute_fp8_sm120<
              Traits, ElementO, TmaQ, TmaK, TmaV, TmaO, true, false, true>);
    } else {
      launch_kernel(
          ffpa_fp8::split_d_m4n2_fwd_cute_fp8_sm120<
              Traits, ElementO, TmaQ, TmaK, TmaV, TmaO, false, false, true>);
    }
  } else if (v_per_channel && pv_acc_f16) {
    launch_kernel(
        ffpa_fp8::split_d_m4n2_fwd_cute_fp8_sm120<Traits, ElementO, TmaQ, TmaK,
                                                  TmaV, TmaO, true, true>);
  } else if (v_per_channel) {
    launch_kernel(
        ffpa_fp8::split_d_m4n2_fwd_cute_fp8_sm120<Traits, ElementO, TmaQ, TmaK,
                                                  TmaV, TmaO, false, true>);
  } else if (pv_acc_f16) {
    launch_kernel(
        ffpa_fp8::split_d_m4n2_fwd_cute_fp8_sm120<Traits, ElementO, TmaQ, TmaK,
                                                  TmaV, TmaO, true>);
  } else {
    launch_kernel(
        ffpa_fp8::split_d_m4n2_fwd_cute_fp8_sm120<Traits, ElementO, TmaQ, TmaK,
                                                  TmaV, TmaO>);
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
  constexpr int kSmemBytes = kStagesQK * kSmemPerStage;

  int max_smem_optin = 0;
  cudaDeviceGetAttribute(&max_smem_optin,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin,
                         Q.device().index());
  TORCH_CHECK(kSmemBytes <= max_smem_optin, "ffpa_attn: CuTe kernel requires ",
              kSmemBytes, " bytes smem (stages=", kStagesQK,
              ", chunk=", kQKDChunk, "/", kVDChunk, ") but device supports ",
              max_smem_optin, " bytes opt-in smem");

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
        philox_seed_u, philox_offset_u);
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

// NVFP4 persist-D launcher (D=128). Pipeline: km (two-stage K column mean,
// shared with fp8) -> q_block_mean -> 3 quantize kernels (Q centered by qm,
// K smoothed by km + row-permuted, V transposed) -> delta_s = qm @ (K-km)^T
// - qm.km (GQA broadcast bmm, fp16 domain like sageattn3) -> TMA descriptors
// -> persist_d_ws_fwd_cute_fp4_sm120. Workspaces are 128-padded along
// seqlen; delta_s tail columns zero-fill (masked -inf in-kernel).
#ifdef ENABLE_FFPA_TMA_EXT
template <typename kDataType, const int kHeadDim, bool kPvMxfp8 = false>
void launch_cute_fwd_persist_d_fp4_sm120_impl(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
    torch::Tensor softmax_lse, int causal, double softmax_scale,
    int q_start_row = 0, bool fp4_hadamard = false, bool fp4_smooth_v = false) {
  using namespace cute;
  constexpr int kBr = 128;
  constexpr int kBc = 128;
  constexpr int kNumThreads = 384;

  using ElementO = std::conditional_t<std::is_same_v<kDataType, __half>,
                                      cutlass::half_t, cutlass::bfloat16_t>;
  using Traits =
      ffpa_fp4::FFPAAttnCuTePersistDFP4Traits<ElementO, kHeadDim, kPvMxfp8>;
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

  constexpr int kSmemBytes = Traits::kSmemBytes;
  TORCH_CHECK(q_start_row >= 0 && q_start_row < Nq,
              "ffpa_attn: q_start_row must be in [0, Nq)");
  TORCH_CHECK(q_start_row % kBr == 0,
              "ffpa_attn: q_start_row must be a multiple of kBr=128");
  // Exceeding the smem opt-in limit fails SILENTLY in cudaFuncSetAttribute
  // (score collapses to zero); check both up front (see sm120-smem-limit).
  int max_smem_optin = 0;
  cudaDeviceGetAttribute(
      &max_smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, Q.get_device());
  TORCH_CHECK(kSmemBytes <= max_smem_optin,
              "ffpa_attn: fp4 persist_d D=", kHeadDim, " needs ", kSmemBytes,
              "B smem, device opt-in allows ", max_smem_optin);
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
  auto kernel = ffpa_fp4::persist_d_ws_fwd_cute_fp4_sm120<
      Traits, ElementO, decltype(tma_q), decltype(tma_k), decltype(tma_v),
      decltype(tma_o), decltype(tma_sfq), decltype(tma_sfk), decltype(tma_sfvt),
      decltype(tma_ds)>;
  TORCH_CHECK(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           kSmemBytes) == cudaSuccess,
      "ffpa_attn: fp4 persist_d smem opt-in failed for D=", kHeadDim);
  kernel<<<grid, block, kSmemBytes, stream>>>(
      tma_q, tma_k, tma_v, tma_o, tma_sfq, tma_sfk, tma_sfvt, tma_ds, O_ptr,
      softmax_lse_ptr,
      fused_wht ? km_rot_f32.data_ptr<float>() : km_f32.data_ptr<float>(),
      fused_wht ? qm_rot.data_ptr<float>() : qm.data_ptr<float>(),
      fp4_smooth_v ? vm_v.data_ptr<float>() : nullptr, Nq, Nkv, Nq_pad, Nkv_pad,
      Nh, Nh_kv, scale, Tc, causal, total_q_rows, Nb, q_start_row, nhd_out);
}

template <typename kDataType, const int kHeadDim, const int kStage>
void launch_cute_fwd_persist_d_fp4_sm120(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
    torch::Tensor softmax_lse, int causal, double softmax_scale,
    int q_start_row = 0, bool fp4_hadamard = false, int fp4_pv_mm_type = 0,
    bool fp4_smooth_v = false) {
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
            Q, K, V, O, softmax_lse, causal, softmax_scale, q_start_row,
            fp4_hadamard, fp4_smooth_v);
    } else {
      launch_cute_fwd_persist_d_fp4_sm120_impl<kDataType, kHeadDim, false>(
          Q, K, V, O, softmax_lse, causal, softmax_scale, q_start_row,
          fp4_hadamard, fp4_smooth_v);
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
template <typename kDataType, const int kHeadDim>
void launch_cute_fwd_split_d_fp4_sm120_impl(torch::Tensor Q, torch::Tensor K,
                                            torch::Tensor V, torch::Tensor O,
                                            torch::Tensor softmax_lse,
                                            int causal, double softmax_scale,
                                            int q_start_row = 0,
                                            bool fp4_hadamard = false) {
  using namespace cute;
  constexpr int kBr = 128;
  constexpr int kBc = 128;
  constexpr int kNumThreads = 256;
  constexpr int kQKDChunk = 64;
  constexpr int kVDChunk = 64;

  using ElementO = std::conditional_t<std::is_same_v<kDataType, __half>,
                                      cutlass::half_t, cutlass::bfloat16_t>;
  using Traits = ffpa_fp4::FFPAAttnCuTeSplitDFP4Traits<ElementO, kHeadDim>;
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
  torch::Tensor vt4 = torch::empty({Nb, Nh_kv, kHeadDim, Nkv_pad / 2}, opts_u8);
  torch::Tensor sfvt =
      torch::empty({Nb, Nh_kv, kHeadDim, Nkv_pad / 16}, opts_u8);
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
    // WHT pre-rotation needs BHND-packed rows; materialize NHD views.
    if (ffpa_is_nhd_view(Q))
      Q = Q.contiguous();
    if (ffpa_is_nhd_view(K))
      K = K.contiguous();
    Q = ffpa::apply_wht_qk_sm120<kDataType, kHeadDim>(Q);
    K = ffpa::apply_wht_qk_sm120<kDataType, kHeadDim>(K);
  }
  const kDataType* k_ptr = reinterpret_cast<const kDataType*>(K.data_ptr());
  // NHD (BNHD) permute views are consumed natively by the pre-kernels.
  const ffpa_fp8::Fp8InputLayout Lkv = ffpa_layout_of(K, Nkv, K.size(3));

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
  ffpa_fp4::launch_fp4_quant_vt_sm120<kHeadDim>(V_t, vt4, sfvt, Nkv_pad);

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
      make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(vt4.data_ptr())),
                  make_shape(d_total, Nkv_pad), make_stride(Nkv_pad, _1{}));
  auto tma_v = make_tma_copy(SM90_TMA_LOAD{}, gV, SmemLayoutVt{}(_, _, _0{}),
                             Shape<Int<kVDChunk>, Int<kBc>>{}, _1{});
  auto gO =
      make_tensor(make_gmem_ptr(reinterpret_cast<ElementO*>(O.data_ptr())),
                  make_shape((long)total_q_rows, Int<kHeadDim>{}),
                  make_stride(Int<kHeadDim>{}, _1{}));
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
  auto kernel = ffpa_fp4::split_d_fwd_cute_fp4_sm120<
      Traits, ElementO, decltype(tma_q), decltype(tma_k), decltype(tma_v),
      decltype(tma_o), decltype(tma_sfq), decltype(tma_sfk), decltype(tma_sfvt),
      decltype(tma_ds)>;
  TORCH_CHECK(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           kSmemBytes) == cudaSuccess,
      "ffpa_attn: fp4 split_d smem opt-in failed for D=", kHeadDim);
  kernel<<<grid, block, kSmemBytes, stream>>>(
      tma_q, tma_k, tma_v, tma_o, tma_sfq, tma_sfk, tma_sfvt, tma_ds, O_ptr,
      softmax_lse_ptr,
      fused_wht ? km_rot_f32.data_ptr<float>() : km_f32.data_ptr<float>(),
      fused_wht ? qm_rot.data_ptr<float>() : qm.data_ptr<float>(), Nq, Nkv,
      Nq_pad, Nkv_pad, Nh, Nh_kv, scale, Tc, causal, total_q_rows, Nb,
      q_start_row);
}

template <typename kDataType, const int kHeadDim, const int kStage>
void launch_cute_fwd_split_d_fp4_sm120(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
    torch::Tensor softmax_lse, int causal, double softmax_scale,
    int q_start_row = 0, bool fp4_hadamard = false, int fp4_pv_mm_type = 0) {
  (void)kStage;  // kStages (3/3) fixed by the fp4 split_d traits
  TORCH_CHECK(fp4_pv_mm_type == 0,
              "ffpa_attn: fp4_pv_mm_type=fp8 supports persist_d (D<=192) "
              "only, got split_d D=",
              kHeadDim);
  auto prop = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(prop->major == 12,
              "ffpa_attn: the NVFP4 path requires an sm_120 device, got sm_",
              prop->major, prop->minor);
  if constexpr (kHeadDim % 64 == 0 && kHeadDim > 256 && kHeadDim < 768) {
    launch_cute_fwd_split_d_fp4_sm120_impl<kDataType, kHeadDim>(
        Q, K, V, O, softmax_lse, causal, softmax_scale, q_start_row,
        fp4_hadamard);
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
// traits own the TMA descriptor shapes).
template <typename kDataType, const int kHeadDim>
void launch_cute_fwd_split_d_m4n2_fp4_sm120_impl(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
    torch::Tensor softmax_lse, int causal, double softmax_scale,
    int q_start_row = 0, bool fp4_hadamard = false) {
  using namespace cute;
  constexpr int kBr = 64;
  constexpr int kBc = 64;
  constexpr int kNumThreads = 256;
  constexpr int kQKDChunk = 64;
  constexpr int kVDChunk = 64;

  using ElementO = std::conditional_t<std::is_same_v<kDataType, __half>,
                                      cutlass::half_t, cutlass::bfloat16_t>;
  using Traits = ffpa_fp4::FFPAAttnCuTeSplitDM4N2FP4Traits<ElementO, kHeadDim>;
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
    // WHT pre-rotation needs BHND-packed rows; materialize NHD views.
    if (ffpa_is_nhd_view(Q))
      Q = Q.contiguous();
    if (ffpa_is_nhd_view(K))
      K = K.contiguous();
    Q = ffpa::apply_wht_qk_sm120<kDataType, kHeadDim>(Q);
    K = ffpa::apply_wht_qk_sm120<kDataType, kHeadDim>(K);
  }
  const kDataType* k_ptr = reinterpret_cast<const kDataType*>(K.data_ptr());
  // NHD (BNHD) permute views are consumed natively by the pre-kernels.
  const ffpa_fp8::Fp8InputLayout Lkv = ffpa_layout_of(K, Nkv, K.size(3));

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
  ffpa_fp4::launch_fp4_quant_vt_sm120<kHeadDim>(V_t, vt4, sfvt, Nkv_pad);

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
  auto gO =
      make_tensor(make_gmem_ptr(reinterpret_cast<ElementO*>(O.data_ptr())),
                  make_shape((long)total_q_rows, Int<kHeadDim>{}),
                  make_stride(Int<kHeadDim>{}, _1{}));
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
  auto kernel = ffpa_fp4::split_d_m4n2_fwd_cute_fp4_sm120<
      Traits, ElementO, decltype(tma_q), decltype(tma_k), decltype(tma_v),
      decltype(tma_o), decltype(tma_sfq), decltype(tma_sfk), decltype(tma_sfvt),
      decltype(tma_ds)>;
  TORCH_CHECK(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           kSmemBytes) == cudaSuccess,
      "ffpa_attn: fp4 split_d m4n2 smem opt-in failed for D=", kHeadDim);
  kernel<<<grid, block, kSmemBytes, stream>>>(
      tma_q, tma_k, tma_v, tma_o, tma_sfq, tma_sfk, tma_sfvt, tma_ds, O_ptr,
      softmax_lse_ptr, km_f32.data_ptr<float>(), qm.data_ptr<float>(), Nq, Nkv,
      Nq_pad, Nkv_pad, Nh, Nh_kv, scale, Tc, causal, total_q_rows, Nb,
      q_start_row);
}

template <typename kDataType, const int kHeadDim, const int kStage>
void launch_cute_fwd_split_d_m4n2_fp4_sm120(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
    torch::Tensor softmax_lse, int causal, double softmax_scale,
    int q_start_row = 0, bool fp4_hadamard = false, int fp4_pv_mm_type = 0) {
  (void)kStage;  // kStages (2/2) fixed by the fp4 m4n2 traits
  TORCH_CHECK(fp4_pv_mm_type == 0,
              "ffpa_attn: fp4_pv_mm_type=fp8 supports persist_d (D<=192) "
              "only, got split_d m4n2 D=",
              kHeadDim);
  auto prop = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(prop->major == 12,
              "ffpa_attn: the NVFP4 path requires an sm_120 device, got sm_",
              prop->major, prop->minor);
  if constexpr (kHeadDim % 64 == 0 && kHeadDim >= 768 && kHeadDim <= 1024) {
    launch_cute_fwd_split_d_m4n2_fp4_sm120_impl<kDataType, kHeadDim>(
        Q, K, V, O, softmax_lse, causal, softmax_scale, q_start_row,
        fp4_hadamard);
  } else {
    TORCH_CHECK(false,
                "ffpa_attn: cute_tma_fp4 split_d m4n2 requires 64-multiple D "
                "in [768,1024], got D=",
                kHeadDim);
  }
}
#endif  // ENABLE_FFPA_TMA_EXT
#endif  // ENABLE_FFPA_CUTE_EXT

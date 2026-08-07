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
#include "cute/fp8/sm_120/persist_d_fp8.cuh"
#include "cute/fp8/sm_120/split_d_fp8.cuh"
#include "cute/fp8/sm_120/split_d_m4n2_fp8.cuh"
#endif
#endif

#ifdef ENABLE_FFPA_CUTE_EXT
#ifdef ENABLE_FFPA_TMA_EXT
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

  auto gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(Q.data_ptr())),
                        make_shape(total_q_rows, Int<kHeadDim>{}),
                        make_stride(Int<kHeadDim>{}, _1{}));
  auto gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(K.data_ptr())),
                        make_shape(total_kv_rows, Int<kHeadDim>{}),
                        make_stride(Int<kHeadDim>{}, _1{}));
  auto gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(V.data_ptr())),
                        make_shape(total_kv_rows, Int<kHeadDim>{}),
                        make_stride(Int<kHeadDim>{}, _1{}));

  auto tma_q = make_tma_copy(SM90_TMA_LOAD{}, gQ, SmemLayoutQ{},
                             Shape<Int<kBr>, Int<kQKDChunk>>{}, _1{});
  auto tma_k = make_tma_copy(SM90_TMA_LOAD{}, gK, SmemLayoutK{},
                             Shape<Int<kBc>, Int<kQKDChunk>>{}, _1{});
  auto tma_v = make_tma_copy(SM90_TMA_LOAD{}, gV, SmemLayoutV{},
                             Shape<Int<kBc>, Int<kVDChunk>>{}, _1{});

  // O output TMA store descriptor: full O tensor [total_q_rows,kHeadDim],
  // same shape/stride as gQ; per-head origin injected via domain_offset in
  // kernel. Direction = SM90_TMA_STORE (first arg); swizzle auto-inferred
  // from SmemLayoutO (matches the sO staging buffer's actual swizzle).
  auto gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(O.data_ptr())),
                        make_shape(total_q_rows, Int<kHeadDim>{}),
                        make_stride(Int<kHeadDim>{}, _1{}));
  auto tma_o = make_tma_copy(SM90_TMA_STORE{}, gO, SmemLayoutO{},
                             Shape<Int<kBr>, Int<kVDChunk>>{}, _1{});

  constexpr int kQTileBytes = kBr * kQKDChunk * sizeof(Element);
  constexpr int kKTileBytes = kBc * kQKDChunk * sizeof(Element);
  constexpr int kVTileBytes = kBc * kVDChunk * sizeof(Element);
  constexpr int kSmemBytes = kStagesQK * kQTileBytes + kStagesQK * kKTileBytes +
                             kStagesPV * kVTileBytes;

  float* softmax_lse_ptr =
      softmax_lse.numel() > 0 ? softmax_lse.data_ptr<float>() : nullptr;
  auto O_ptr = reinterpret_cast<Element*>(O.data_ptr());

  auto launch_variant = [&](auto kernel_func) {
    cudaFuncSetAttribute(
        kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
    kernel_func<<<grid, block, kSmemBytes, stream>>>(
        tma_q, tma_k, tma_v, tma_o, O_ptr, softmax_lse_ptr, Nq, Nkv, Nh, Nh_kv,
        scale, Tc, causal, total_q_rows, total_kv_rows, attn_bias_ptr,
        attn_bias_dtype, attn_bias_stride_b, attn_bias_stride_h,
        attn_bias_stride_m, attn_bias_stride_n, dropout_p_f, philox_seed_u,
        philox_offset_u);
  };

  using TmaQ = decltype(tma_q);
  using TmaK = decltype(tma_k);
  using TmaV = decltype(tma_v);
  using TmaO = decltype(tma_o);
  if (has_attn_bias && has_dropout) {
    launch_variant(
        split_d_fwd_cute_sm120<Traits, TmaQ, TmaK, TmaV, TmaO, 1, 1>);
  } else if (has_attn_bias) {
    launch_variant(
        split_d_fwd_cute_sm120<Traits, TmaQ, TmaK, TmaV, TmaO, 1, 0>);
  } else if (has_dropout) {
    launch_variant(
        split_d_fwd_cute_sm120<Traits, TmaQ, TmaK, TmaV, TmaO, 0, 1>);
  } else {
    launch_variant(
        split_d_fwd_cute_sm120<Traits, TmaQ, TmaK, TmaV, TmaO, 0, 0>);
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

  auto gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(Q.data_ptr())),
                        make_shape(total_q_rows, Int<kHeadDim>{}),
                        make_stride(Int<kHeadDim>{}, _1{}));
  auto gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(K.data_ptr())),
                        make_shape(total_kv_rows, Int<kHeadDim>{}),
                        make_stride(Int<kHeadDim>{}, _1{}));
  auto gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(V.data_ptr())),
                        make_shape(total_kv_rows, Int<kHeadDim>{}),
                        make_stride(Int<kHeadDim>{}, _1{}));

  auto tma_q = make_tma_copy(SM90_TMA_LOAD{}, gQ, SmemLayoutQ{},
                             Shape<Int<kBr>, Int<kQKDChunk>>{}, _1{});
  auto tma_k = make_tma_copy(SM90_TMA_LOAD{}, gK, SmemLayoutK{},
                             Shape<Int<kBc>, Int<kQKDChunk>>{}, _1{});
  auto tma_v = make_tma_copy(SM90_TMA_LOAD{}, gV, SmemLayoutV{},
                             Shape<Int<kBc>, Int<kVDChunk>>{}, _1{});

  auto gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(O.data_ptr())),
                        make_shape(total_q_rows, Int<kHeadDim>{}),
                        make_stride(Int<kHeadDim>{}, _1{}));
  auto tma_o = make_tma_copy(SM90_TMA_STORE{}, gO, SmemLayoutO{},
                             Shape<Int<kBr>, Int<kVDChunk>>{}, _1{});

  constexpr int kSmemBytes = Traits::kSmemElems * sizeof(Element);

  float* softmax_lse_ptr =
      softmax_lse.numel() > 0 ? softmax_lse.data_ptr<float>() : nullptr;
  auto O_ptr = reinterpret_cast<Element*>(O.data_ptr());

  auto launch_variant = [&](auto kernel_func) {
    cudaFuncSetAttribute(
        kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
    kernel_func<<<grid, block, kSmemBytes, stream>>>(
        tma_q, tma_k, tma_v, tma_o, O_ptr, softmax_lse_ptr, Nq, Nkv, Nh, Nh_kv,
        scale, Tc, causal, total_q_rows, total_kv_rows, attn_bias_ptr,
        attn_bias_dtype, attn_bias_stride_b, attn_bias_stride_h,
        attn_bias_stride_m, attn_bias_stride_n, dropout_p_f, philox_seed_u,
        philox_offset_u);
  };

  using TmaQ = decltype(tma_q);
  using TmaK = decltype(tma_k);
  using TmaV = decltype(tma_v);
  using TmaO = decltype(tma_o);
  if (has_attn_bias && has_dropout) {
    launch_variant(
        split_d_m4n2_fwd_cute_sm120<Traits, TmaQ, TmaK, TmaV, TmaO, 1, 1>);
  } else if (has_attn_bias) {
    launch_variant(
        split_d_m4n2_fwd_cute_sm120<Traits, TmaQ, TmaK, TmaV, TmaO, 1, 0>);
  } else if (has_dropout) {
    launch_variant(
        split_d_m4n2_fwd_cute_sm120<Traits, TmaQ, TmaK, TmaV, TmaO, 0, 1>);
  } else {
    launch_variant(
        split_d_m4n2_fwd_cute_sm120<Traits, TmaQ, TmaK, TmaV, TmaO, 0, 0>);
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

  const dim3 block(kNumThreads, 1, 1);
  const dim3 grid(utils::div_ceil(Nq, kBr), Nb * Nh, 1);

  const c10::cuda::OptionalCUDAGuard device_guard(Q.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  const int total_q_rows = Nb * Nh * Nq;
  const int total_kv_rows = Nb * Nh_kv * Nkv;

  auto gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(Q.data_ptr())),
                        make_shape(total_q_rows, Int<kHeadDim>{}),
                        make_stride(Int<kHeadDim>{}, _1{}));
  auto gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(K.data_ptr())),
                        make_shape(total_kv_rows, Int<kHeadDim>{}),
                        make_stride(Int<kHeadDim>{}, _1{}));
  auto gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(V.data_ptr())),
                        make_shape(total_kv_rows, Int<kHeadDim>{}),
                        make_stride(Int<kHeadDim>{}, _1{}));
  auto gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(O.data_ptr())),
                        make_shape(total_q_rows, Int<kHeadDim>{}),
                        make_stride(Int<kHeadDim>{}, _1{}));

  auto tma_q = make_tma_copy(SM90_TMA_LOAD{}, gQ, SmemLayoutQ{},
                             Shape<Int<kBr>, Int<kHeadDim>>{}, _1{});
  auto tma_k = make_tma_copy(SM90_TMA_LOAD{}, gK, SmemLayoutKV{},
                             Shape<Int<kBc>, Int<kHeadDim>>{}, _1{});
  auto tma_v = make_tma_copy(SM90_TMA_LOAD{}, gV, SmemLayoutKV{},
                             Shape<Int<kBc>, Int<kHeadDim>>{}, _1{});
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
        tma_q, tma_k, tma_v, tma_o, O_ptr, softmax_lse_ptr, Nq, Nkv, Nh, Nh_kv,
        scale, Tc, causal, total_q_rows, total_kv_rows, attn_bias_ptr,
        attn_bias_dtype, attn_bias_stride_b, attn_bias_stride_h,
        attn_bias_stride_m, attn_bias_stride_n, dropout_p_f, philox_seed_u,
        philox_offset_u);
  };

  using TmaQ = decltype(tma_q);
  using TmaK = decltype(tma_k);
  using TmaV = decltype(tma_v);
  using TmaO = decltype(tma_o);
  if (has_attn_bias && has_dropout) {
    launch_variant(
        persist_d_ws_fwd_cute_sm120<Traits, TmaQ, TmaK, TmaV, TmaO, 1, 1>);
  } else if (has_attn_bias) {
    launch_variant(
        persist_d_ws_fwd_cute_sm120<Traits, TmaQ, TmaK, TmaV, TmaO, 1, 0>);
  } else if (has_dropout) {
    launch_variant(
        persist_d_ws_fwd_cute_sm120<Traits, TmaQ, TmaK, TmaV, TmaO, 0, 1>);
  } else {
    launch_variant(
        persist_d_ws_fwd_cute_sm120<Traits, TmaQ, TmaK, TmaV, TmaO, 0, 0>);
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
    int64_t philox_offset, bool smooth_k) {
  using namespace cute;
  TORCH_CHECK(attn_bias.numel() == 0 && dropout_p == 0.0,
              "fp8 sm120 path does not support attn_bias/dropout");

  constexpr int kBr = 128;
  constexpr int kBc = 128;
  constexpr int kSmemBudgetBytes = 99 * 1024;
  constexpr int kQPersistBytes = kBr * kHeadDim;  // e4m3/int8 = 1B
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
  torch::Tensor q_scale = torch::empty({Nb * Nh, n_rb_q}, opts_f32);
  torch::Tensor k_scale = torch::empty({Nb * Nh_kv, n_rb_kv}, opts_f32);
  torch::Tensor v_scale = torch::empty({Nb * Nh_kv, n_rb_kv}, opts_f32);
  const bool pquant_per_row = getenv("FFPA_FP8_PQUANT_PER_ROW") != nullptr;

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
  if (smooth_k) {
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
        stream);
  }
  ffpa_fp8::launch_quantize_fp8_sm120<kDataType, kBr, kBc, kHeadDim, kQKInt8>(
      q_ptr, k_ptr, v_ptr, q8.data_ptr(), k8.data_ptr(),
      reinterpret_cast<__nv_fp8_e4m3*>(vt8.data_ptr()),
      q_scale.data_ptr<float>(), k_scale.data_ptr<float>(),
      v_scale.data_ptr<float>(), Nb, Nh, Nh_kv, Nq, Nkv, Nkv_pad, kHeadDim,
      stream, km_ptr);

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

  // O store descriptor: full [total_q_rows, kHeadDim] ElementO tensor; the
  // per-(batch,head) origin is injected via domain_offset in the kernel. The
  // smem layout mirrors the kernel's SmemLayoutO staging (SW128, ElementO).
  auto gO =
      make_tensor(make_gmem_ptr(reinterpret_cast<ElementO*>(O.data_ptr())),
                  make_shape(total_q_rows, Int<kHeadDim>{}),
                  make_stride(Int<kHeadDim>{}, _1{}));
  auto tma_o = make_tma_copy(SM90_TMA_STORE{}, gO, SmemLayoutO{},
                             Shape<Int<kBr>, Int<kHeadDim>>{}, _1{});

  constexpr int kSmemBytes = Traits::kSmemElems * sizeof(Element);
  float* softmax_lse_ptr =
      softmax_lse.numel() > 0 ? softmax_lse.data_ptr<float>() : nullptr;
  auto O_ptr = reinterpret_cast<ElementO*>(O.data_ptr());

  const dim3 block(kNumThreads, 1, 1);
  const dim3 grid(utils::div_ceil(Nq, kBr), Nb * Nh, 1);
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
        total_q_rows, total_kv_rows, n_rb_q, n_rb_kv, km_f32_ptr);
  };
  if (pquant_per_row) {
    launch_kernel(
        ffpa_fp8::persist_d_ws_fwd_cute_fp8_sm120<Traits, ElementO, TmaQ, TmaK,
                                                  TmaV, TmaO, true>);
  } else {
    launch_kernel(
        ffpa_fp8::persist_d_ws_fwd_cute_fp8_sm120<Traits, ElementO, TmaQ, TmaK,
                                                  TmaV, TmaO, false>);
  }
}

template <typename kDataType, const int kHeadDim, const int kStage>
void launch_cute_fwd_persist_d_fp8_sm120(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
    torch::Tensor attn_bias, torch::Tensor softmax_lse, int causal,
    double softmax_scale, double dropout_p, int64_t philox_seed,
    int64_t philox_offset, bool smooth_k, std::optional<bool> qk_int8_opt) {
  // QK dtype tri-state: explicit qk_int8 wins; else FFPA_FP8_QK_INT8
  // (=1 forces int8, =0 forces fp8); else auto-selects int8 for causal
  // (early-row accuracy limit, see the masking-pass comment in
  // persist_d_fp8.cuh; int8 QK fixes its dS part at ~zero causal cost)
  // and fp8 otherwise (dense pays ~7.5% for no gain).
  // if constexpr keeps the impl (and its kernel) out of instantiation for
  // unsupported headdims; every headdim TU includes this launcher template.
  if constexpr (kHeadDim == 64 || kHeadDim == 128) {
    bool qk_int8;
    if (qk_int8_opt.has_value()) {
      qk_int8 = *qk_int8_opt;
    } else {
      const char* qk_int8_env = getenv("FFPA_FP8_QK_INT8");
      qk_int8 = qk_int8_env != nullptr ? qk_int8_env[0] != '0' : (causal != 0);
    }
    if (qk_int8)
      launch_cute_fwd_persist_d_fp8_sm120_impl<kDataType, kHeadDim, kStage,
                                               true>(
          Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale, dropout_p,
          philox_seed, philox_offset, smooth_k);
    else
      launch_cute_fwd_persist_d_fp8_sm120_impl<kDataType, kHeadDim, kStage,
                                               false>(
          Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale, dropout_p,
          philox_seed, philox_offset, smooth_k);
  } else {
    TORCH_CHECK(false,
                "ffpa_attn: cute_tma_fp8 requires D=64/128, got D=", kHeadDim);
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
    int64_t philox_offset, bool smooth_k) {
  using namespace cute;
  TORCH_CHECK(attn_bias.numel() == 0 && dropout_p == 0.0,
              "fp8 sm120 path does not support attn_bias/dropout");

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
  torch::Tensor q_scale = torch::empty({Nb * Nh, n_rb_q}, opts_f32);
  torch::Tensor k_scale = torch::empty({Nb * Nh_kv, n_rb_kv}, opts_f32);
  torch::Tensor v_scale = torch::empty({Nb * Nh_kv, n_rb_kv}, opts_f32);

  const c10::cuda::OptionalCUDAGuard device_guard(Q.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  torch::Tensor km, km_f32, km_partials;
  const kDataType* km_ptr = nullptr;
  const float* km_f32_ptr = nullptr;
  const kDataType* q_ptr = reinterpret_cast<const kDataType*>(Q.data_ptr());
  const kDataType* k_ptr = reinterpret_cast<const kDataType*>(K.data_ptr());
  const kDataType* v_ptr = reinterpret_cast<const kDataType*>(V.data_ptr());
  if (smooth_k) {
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
        stream);
  }
  ffpa_fp8::launch_quantize_fp8_sm120<kDataType, kBr, kBc, kHeadDim, kQKInt8>(
      q_ptr, k_ptr, v_ptr, q8.data_ptr(), k8.data_ptr(),
      reinterpret_cast<__nv_fp8_e4m3*>(vt8.data_ptr()),
      q_scale.data_ptr<float>(), k_scale.data_ptr<float>(),
      v_scale.data_ptr<float>(), Nb, Nh, Nh_kv, Nq, Nkv, Nkv_pad, kHeadDim,
      stream, km_ptr);

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
  const dim3 grid(utils::div_ceil(Nq, kBr), Nb * Nh, 1);
  using TmaQ = decltype(tma_q);
  using TmaK = decltype(tma_k);
  using TmaV = decltype(tma_v);
  using TmaO = decltype(tma_o);
  auto kernel = ffpa_fp8::split_d_ws_fwd_cute_fp8_sm120<Traits, ElementO, TmaQ,
                                                        TmaK, TmaV, TmaO>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       kSmemBytes);
  kernel<<<grid, block, kSmemBytes, stream>>>(
      tma_q, tma_k, tma_v, tma_o, O_ptr, softmax_lse_ptr,
      q_scale.data_ptr<float>(), k_scale.data_ptr<float>(),
      v_scale.data_ptr<float>(), Nq, Nkv, Nh, Nh_kv, scale, Tc, causal,
      total_q_rows, total_kv_rows, n_rb_q, n_rb_kv, km_f32_ptr);
}

template <typename kDataType, const int kHeadDim, const int kStage>
void launch_cute_fwd_split_d_fp8_sm120(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
    torch::Tensor attn_bias, torch::Tensor softmax_lse, int causal,
    double softmax_scale, double dropout_p, int64_t philox_seed,
    int64_t philox_offset, bool smooth_k, std::optional<bool> qk_int8_opt) {
  // Same qk_int8 tri-state as persist_d: explicit param > FFPA_FP8_QK_INT8
  // env ("1"/"0") > auto (causal -> int8 for early-row accuracy, dense fp8).
  // if constexpr keeps the impl (and its kernel) out of instantiation for
  // unsupported headdims; every headdim TU includes this launcher template.
  if constexpr (kHeadDim > 128 && kHeadDim < 768 && kHeadDim % 64 == 0) {
    bool qk_int8;
    if (qk_int8_opt.has_value()) {
      qk_int8 = *qk_int8_opt;
    } else {
      const char* qk_int8_env = getenv("FFPA_FP8_QK_INT8");
      qk_int8 = qk_int8_env != nullptr ? qk_int8_env[0] != '0' : (causal != 0);
    }
    if (qk_int8)
      launch_cute_fwd_split_d_fp8_sm120_impl<kDataType, kHeadDim, kStage, true>(
          Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale, dropout_p,
          philox_seed, philox_offset, smooth_k);
    else
      launch_cute_fwd_split_d_fp8_sm120_impl<kDataType, kHeadDim, kStage,
                                             false>(
          Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale, dropout_p,
          philox_seed, philox_offset, smooth_k);
  } else {
    TORCH_CHECK(false,
                "ffpa_attn: cute_tma_fp8 split_d requires D in (128, 768) "
                "with D % 64 == 0, got D=",
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
    int64_t philox_offset, bool smooth_k) {
  using namespace cute;
  TORCH_CHECK(attn_bias.numel() == 0 && dropout_p == 0.0,
              "fp8 sm120 path does not support attn_bias/dropout");

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
  torch::Tensor q_scale = torch::empty({Nb * Nh, n_rb_q}, opts_f32);
  torch::Tensor k_scale = torch::empty({Nb * Nh_kv, n_rb_kv}, opts_f32);
  torch::Tensor v_scale = torch::empty({Nb * Nh_kv, n_rb_kv}, opts_f32);

  const c10::cuda::OptionalCUDAGuard device_guard(Q.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  torch::Tensor km, km_f32, km_partials;
  const kDataType* km_ptr = nullptr;
  const float* km_f32_ptr = nullptr;
  const kDataType* q_ptr = reinterpret_cast<const kDataType*>(Q.data_ptr());
  const kDataType* k_ptr = reinterpret_cast<const kDataType*>(K.data_ptr());
  const kDataType* v_ptr = reinterpret_cast<const kDataType*>(V.data_ptr());
  if (smooth_k) {
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
        stream);
  }
  ffpa_fp8::launch_quantize_fp8_sm120<kDataType, kBr, kBc, kHeadDim, kQKInt8>(
      q_ptr, k_ptr, v_ptr, q8.data_ptr(), k8.data_ptr(),
      reinterpret_cast<__nv_fp8_e4m3*>(vt8.data_ptr()),
      q_scale.data_ptr<float>(), k_scale.data_ptr<float>(),
      v_scale.data_ptr<float>(), Nb, Nh, Nh_kv, Nq, Nkv, Nkv_pad, kHeadDim,
      stream, km_ptr);

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
  const dim3 grid(utils::div_ceil(Nq, kBr), Nb * Nh, 1);
  using TmaQ = decltype(tma_q);
  using TmaK = decltype(tma_k);
  using TmaV = decltype(tma_v);
  using TmaO = decltype(tma_o);
  auto kernel =
      ffpa_fp8::split_d_m4n2_fwd_cute_fp8_sm120<Traits, ElementO, TmaQ, TmaK,
                                                TmaV, TmaO>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       kSmemBytes);
  kernel<<<grid, block, kSmemBytes, stream>>>(
      tma_q, tma_k, tma_v, tma_o, O_ptr, softmax_lse_ptr,
      q_scale.data_ptr<float>(), k_scale.data_ptr<float>(),
      v_scale.data_ptr<float>(), Nq, Nkv, Nh, Nh_kv, scale, Tc, causal,
      total_q_rows, total_kv_rows, n_rb_q, n_rb_kv, km_f32_ptr);
}

template <typename kDataType, const int kHeadDim, const int kStage>
void launch_cute_fwd_split_d_m4n2_fp8_sm120(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
    torch::Tensor attn_bias, torch::Tensor softmax_lse, int causal,
    double softmax_scale, double dropout_p, int64_t philox_seed,
    int64_t philox_offset, bool smooth_k, std::optional<bool> qk_int8_opt) {
  if constexpr (kHeadDim >= 768 && kHeadDim <= 1024 && kHeadDim % 64 == 0) {
    bool qk_int8;
    if (qk_int8_opt.has_value()) {
      qk_int8 = *qk_int8_opt;
    } else {
      const char* qk_int8_env = getenv("FFPA_FP8_QK_INT8");
      qk_int8 = qk_int8_env != nullptr ? qk_int8_env[0] != '0' : (causal != 0);
    }
    if (qk_int8)
      launch_cute_fwd_split_d_m4n2_fp8_sm120_impl<kDataType, kHeadDim, kStage,
                                                  true>(
          Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale, dropout_p,
          philox_seed, philox_offset, smooth_k);
    else
      launch_cute_fwd_split_d_m4n2_fp8_sm120_impl<kDataType, kHeadDim, kStage,
                                                  false>(
          Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale, dropout_p,
          philox_seed, philox_offset, smooth_k);
  } else {
    TORCH_CHECK(false,
                "ffpa_attn: cute_tma_fp8 split_d_m4n2 requires D in "
                "[768, 1024] with D % 64 == 0, got D=",
                kHeadDim);
  }
}

template <typename kDataType, const int kHeadDim, const int kStage>
void launch_cute_fwd_split_d_ws_sm120(torch::Tensor Q, torch::Tensor K,
                                      torch::Tensor V, torch::Tensor O,
                                      torch::Tensor attn_bias,
                                      torch::Tensor softmax_lse, int causal,
                                      double softmax_scale, double dropout_p,
                                      int64_t philox_seed,
                                      int64_t philox_offset) {
  using namespace cute;

  // WS split-D reuses FFPAAttnCuTeSplitDTraits (FA-2 split-Q M8N1: 8 warps
  // along M, 1 along N), identical to the non-WS split-D consumer (kBr=128).
  // The WS layer only adds a 128-thread TMA producer warpgroup; the consumer
  // MMA layout is the same proven M8N1 used by
  // split_d_fwd_cute_sm120.
  constexpr int kBr = 128;
  constexpr int kBc = 64;
  constexpr int kQKDChunk = 32;
  constexpr int kVDChunk = 64;
  constexpr int kSmemBudgetBytes = 99 * 1024;
  constexpr int kElemSize = sizeof(kDataType);
  constexpr int kPerStageBytes =
      (kBr * kQKDChunk + kBc * kQKDChunk + kBc * kVDChunk) * kElemSize;
  constexpr int kMaxStages = kSmemBudgetBytes / kPerStageBytes;
  // stages=1: single-buffer makes producer TMA writes (async proxy) collide
  // with consumer ldmatrix reads (generic proxy) on the same smem slot;
  // CtaBarrier (async proxy) can't prove the generic-proxy read finished.
  // Clamp >=2 so double-buffering keeps read/write addresses disjoint.
  constexpr int kStagesQK =
      (kStage < 2) ? 2 : (kStage > kMaxStages ? kMaxStages : kStage);
  constexpr int kStagesPV = kStagesQK;
  // WS: 128 producer + 256 consumer (M8N1) = 384 threads
  constexpr int kNumThreads = 384;

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

  auto gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(Q.data_ptr())),
                        make_shape(total_q_rows, Int<kHeadDim>{}),
                        make_stride(Int<kHeadDim>{}, _1{}));
  auto gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(K.data_ptr())),
                        make_shape(total_kv_rows, Int<kHeadDim>{}),
                        make_stride(Int<kHeadDim>{}, _1{}));
  auto gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(V.data_ptr())),
                        make_shape(total_kv_rows, Int<kHeadDim>{}),
                        make_stride(Int<kHeadDim>{}, _1{}));
  auto gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(O.data_ptr())),
                        make_shape(total_q_rows, Int<kHeadDim>{}),
                        make_stride(Int<kHeadDim>{}, _1{}));

  auto tma_q = make_tma_copy(SM90_TMA_LOAD{}, gQ, SmemLayoutQ{},
                             Shape<Int<kBr>, Int<kQKDChunk>>{}, _1{});
  auto tma_k = make_tma_copy(SM90_TMA_LOAD{}, gK, SmemLayoutK{},
                             Shape<Int<kBc>, Int<kQKDChunk>>{}, _1{});
  auto tma_v = make_tma_copy(SM90_TMA_LOAD{}, gV, SmemLayoutV{},
                             Shape<Int<kBc>, Int<kVDChunk>>{}, _1{});
  auto tma_o = make_tma_copy(SM90_TMA_STORE{}, gO, SmemLayoutO{},
                             Shape<Int<kBr>, Int<kVDChunk>>{}, _1{});

  constexpr int kSmemBytes = Traits::kSmemElems * sizeof(Element);

  float* softmax_lse_ptr =
      softmax_lse.numel() > 0 ? softmax_lse.data_ptr<float>() : nullptr;
  auto O_ptr = reinterpret_cast<Element*>(O.data_ptr());

  auto launch_variant = [&](auto kernel_func) {
    cudaFuncSetAttribute(
        kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
    kernel_func<<<grid, block, kSmemBytes, stream>>>(
        tma_q, tma_k, tma_v, tma_o, O_ptr, softmax_lse_ptr, Nq, Nkv, Nh, Nh_kv,
        scale, Tc, causal, total_q_rows, total_kv_rows, attn_bias_ptr,
        attn_bias_dtype, attn_bias_stride_b, attn_bias_stride_h,
        attn_bias_stride_m, attn_bias_stride_n, dropout_p_f, philox_seed_u,
        philox_offset_u);
  };

  using TmaQ = decltype(tma_q);
  using TmaK = decltype(tma_k);
  using TmaV = decltype(tma_v);
  using TmaO = decltype(tma_o);
  if (has_attn_bias && has_dropout) {
    launch_variant(
        split_d_ws_fwd_cute_sm120<Traits, TmaQ, TmaK, TmaV, TmaO, 1, 1>);
  } else if (has_attn_bias) {
    launch_variant(
        split_d_ws_fwd_cute_sm120<Traits, TmaQ, TmaK, TmaV, TmaO, 1, 0>);
  } else if (has_dropout) {
    launch_variant(
        split_d_ws_fwd_cute_sm120<Traits, TmaQ, TmaK, TmaV, TmaO, 0, 1>);
  } else {
    launch_variant(
        split_d_ws_fwd_cute_sm120<Traits, TmaQ, TmaK, TmaV, TmaO, 0, 0>);
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
#endif  // ENABLE_FFPA_CUTE_EXT

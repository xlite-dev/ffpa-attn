#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "common.cuh"
#ifdef ENABLE_FFPA_CUTE_EXT
#include "cute/sm_80/split_d.cuh"
#ifdef ENABLE_FFPA_TMA_EXT
#include "cute/sm_120/split_d.cuh"
#include "cute/sm_120/persist_d.cuh"
#include "cute/sm_120/split_d_m4n2.cuh"
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
  // stages=1 has a barrier-protocol race on sm_120a; clamp to >=2.
  constexpr int kStagesQK = (kStage < 2 ? 2 : (kStage > 3 ? 3 : kStage));
  constexpr int kStagesPV = kStagesQK;
  constexpr int kNumThreads = kBr / 16 * 32;

  using CuteElement = std::conditional_t<std::is_same_v<kDataType, __half>,
                                         cutlass::half_t, cutlass::bfloat16_t>;
  using Traits =
      ffpa_cute::FFPAAttnCuTeSplitDTraits<kHeadDim, kBr, kBc, kQKDChunk,
                                          kVDChunk, kStagesQK, kStagesPV,
                                          CuteElement>;
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

  auto gQ =
      make_tensor(make_gmem_ptr(reinterpret_cast<CuteElement*>(Q.data_ptr())),
                  make_shape(total_q_rows, Int<kHeadDim>{}),
                  make_stride(Int<kHeadDim>{}, _1{}));
  auto gK =
      make_tensor(make_gmem_ptr(reinterpret_cast<CuteElement*>(K.data_ptr())),
                  make_shape(total_kv_rows, Int<kHeadDim>{}),
                  make_stride(Int<kHeadDim>{}, _1{}));
  auto gV =
      make_tensor(make_gmem_ptr(reinterpret_cast<CuteElement*>(V.data_ptr())),
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
  auto gO =
      make_tensor(make_gmem_ptr(reinterpret_cast<CuteElement*>(O.data_ptr())),
                  make_shape(total_q_rows, Int<kHeadDim>{}),
                  make_stride(Int<kHeadDim>{}, _1{}));
  auto tma_o = make_tma_copy(SM90_TMA_STORE{}, gO, SmemLayoutO{},
                             Shape<Int<kBr>, Int<kVDChunk>>{}, _1{});

  constexpr int kQTileBytes = kBr * kQKDChunk * sizeof(CuteElement);
  constexpr int kKTileBytes = kBc * kQKDChunk * sizeof(CuteElement);
  constexpr int kVTileBytes = kBc * kVDChunk * sizeof(CuteElement);
  constexpr int kSmemBytes = kStagesQK * kQTileBytes + kStagesQK * kKTileBytes +
                             kStagesPV * kVTileBytes;

  float* softmax_lse_ptr =
      softmax_lse.numel() > 0 ? softmax_lse.data_ptr<float>() : nullptr;
  auto O_ptr = reinterpret_cast<CuteElement*>(O.data_ptr());

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

  using CuteElement = std::conditional_t<std::is_same_v<kDataType, __half>,
                                         cutlass::half_t, cutlass::bfloat16_t>;
  using Traits =
      ffpa_cute::FFPAAttnCuTeSplitDM4N2Traits<kHeadDim, kBr, kBc, kQKDChunk,
                                              kVDChunk, kStagesQK, kStagesPV,
                                              CuteElement>;
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

  auto gQ =
      make_tensor(make_gmem_ptr(reinterpret_cast<CuteElement*>(Q.data_ptr())),
                  make_shape(total_q_rows, Int<kHeadDim>{}),
                  make_stride(Int<kHeadDim>{}, _1{}));
  auto gK =
      make_tensor(make_gmem_ptr(reinterpret_cast<CuteElement*>(K.data_ptr())),
                  make_shape(total_kv_rows, Int<kHeadDim>{}),
                  make_stride(Int<kHeadDim>{}, _1{}));
  auto gV =
      make_tensor(make_gmem_ptr(reinterpret_cast<CuteElement*>(V.data_ptr())),
                  make_shape(total_kv_rows, Int<kHeadDim>{}),
                  make_stride(Int<kHeadDim>{}, _1{}));

  auto tma_q = make_tma_copy(SM90_TMA_LOAD{}, gQ, SmemLayoutQ{},
                             Shape<Int<kBr>, Int<kQKDChunk>>{}, _1{});
  auto tma_k = make_tma_copy(SM90_TMA_LOAD{}, gK, SmemLayoutK{},
                             Shape<Int<kBc>, Int<kQKDChunk>>{}, _1{});
  auto tma_v = make_tma_copy(SM90_TMA_LOAD{}, gV, SmemLayoutV{},
                             Shape<Int<kBc>, Int<kVDChunk>>{}, _1{});

  auto gO =
      make_tensor(make_gmem_ptr(reinterpret_cast<CuteElement*>(O.data_ptr())),
                  make_shape(total_q_rows, Int<kHeadDim>{}),
                  make_stride(Int<kHeadDim>{}, _1{}));
  auto tma_o = make_tma_copy(SM90_TMA_STORE{}, gO, SmemLayoutO{},
                             Shape<Int<kBr>, Int<kVDChunk>>{}, _1{});

  constexpr int kSmemBytes = Traits::kSmemElems * sizeof(CuteElement);

  float* softmax_lse_ptr =
      softmax_lse.numel() > 0 ? softmax_lse.data_ptr<float>() : nullptr;
  auto O_ptr = reinterpret_cast<CuteElement*>(O.data_ptr());

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

  using CuteElement = std::conditional_t<std::is_same_v<kDataType, __half>,
                                         cutlass::half_t, cutlass::bfloat16_t>;
  using Traits =
      ffpa_cute::FFPAAttnCuTePersistDTraits<kHeadDim, kBr, kBc, kStagesK,
                                            kStagesV, CuteElement>;
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

  auto gQ =
      make_tensor(make_gmem_ptr(reinterpret_cast<CuteElement*>(Q.data_ptr())),
                  make_shape(total_q_rows, Int<kHeadDim>{}),
                  make_stride(Int<kHeadDim>{}, _1{}));
  auto gK =
      make_tensor(make_gmem_ptr(reinterpret_cast<CuteElement*>(K.data_ptr())),
                  make_shape(total_kv_rows, Int<kHeadDim>{}),
                  make_stride(Int<kHeadDim>{}, _1{}));
  auto gV =
      make_tensor(make_gmem_ptr(reinterpret_cast<CuteElement*>(V.data_ptr())),
                  make_shape(total_kv_rows, Int<kHeadDim>{}),
                  make_stride(Int<kHeadDim>{}, _1{}));
  auto gO =
      make_tensor(make_gmem_ptr(reinterpret_cast<CuteElement*>(O.data_ptr())),
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

  constexpr int kSmemBytes = Traits::kSmemElems * sizeof(CuteElement);

  float* softmax_lse_ptr =
      softmax_lse.numel() > 0 ? softmax_lse.data_ptr<float>() : nullptr;
  auto O_ptr = reinterpret_cast<CuteElement*>(O.data_ptr());

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
  // stages=1 has a barrier-protocol race on sm_120a; clamp to >=2.
  constexpr int kStagesQK =
      (kStage < 2) ? 2 : (kStage > kMaxStages ? kMaxStages : kStage);
  constexpr int kStagesPV = kStagesQK;
  // WS: 128 producer + 256 consumer (M8N1) = 384 threads
  constexpr int kNumThreads = 384;

  using CuteElement = std::conditional_t<std::is_same_v<kDataType, __half>,
                                         cutlass::half_t, cutlass::bfloat16_t>;
  using Traits =
      ffpa_cute::FFPAAttnCuTeSplitDTraits<kHeadDim, kBr, kBc, kQKDChunk,
                                          kVDChunk, kStagesQK, kStagesPV,
                                          CuteElement>;
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

  auto gQ =
      make_tensor(make_gmem_ptr(reinterpret_cast<CuteElement*>(Q.data_ptr())),
                  make_shape(total_q_rows, Int<kHeadDim>{}),
                  make_stride(Int<kHeadDim>{}, _1{}));
  auto gK =
      make_tensor(make_gmem_ptr(reinterpret_cast<CuteElement*>(K.data_ptr())),
                  make_shape(total_kv_rows, Int<kHeadDim>{}),
                  make_stride(Int<kHeadDim>{}, _1{}));
  auto gV =
      make_tensor(make_gmem_ptr(reinterpret_cast<CuteElement*>(V.data_ptr())),
                  make_shape(total_kv_rows, Int<kHeadDim>{}),
                  make_stride(Int<kHeadDim>{}, _1{}));
  auto gO =
      make_tensor(make_gmem_ptr(reinterpret_cast<CuteElement*>(O.data_ptr())),
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

  constexpr int kSmemBytes = Traits::kSmemElems * sizeof(CuteElement);

  float* softmax_lse_ptr =
      softmax_lse.numel() > 0 ? softmax_lse.data_ptr<float>() : nullptr;
  auto O_ptr = reinterpret_cast<CuteElement*>(O.data_ptr());

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

  using CuteElement = std::conditional_t<std::is_same_v<kDataType, __half>,
                                         cutlass::half_t, cutlass::bfloat16_t>;
  constexpr int kStagesQK = kStage;
  constexpr int kStagesPV = kStagesQK;
  using Traits =
      ffpa_cute::FFPAAttnCuTeSplitDTraits<kHeadDim, kBr, kBc, kQKDChunk,
                                          kVDChunk, kStagesQK, kStagesPV,
                                          CuteElement>;

  constexpr int kQTileBytes = kBr * kQKDChunk * sizeof(CuteElement);
  constexpr int kKTileBytes = kBc * kQKDChunk * sizeof(CuteElement);
  constexpr int kVTileBytes = kBc * kVDChunk * sizeof(CuteElement);
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
  auto Q_ptr = reinterpret_cast<CuteElement*>(Q.data_ptr());
  auto K_ptr = reinterpret_cast<CuteElement*>(K.data_ptr());
  auto V_ptr = reinterpret_cast<CuteElement*>(V.data_ptr());
  auto O_ptr = reinterpret_cast<CuteElement*>(O.data_ptr());

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

#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "backend.h"
#include "common.cuh"
#include "native/sm_80/split_d.cuh"
#include "native/sm_80/split_kv.cuh"
#ifdef ENABLE_FFPA_TMA_EXT
#include "native/sm_120/split_d.cuh"
#include "native/tma.cuh"
#endif
using namespace ffpa;

static constexpr int kMaxDForOStoreFloat32 = 1024;
// Use fp16/bf16 for D > 1024 to save registers, since

static inline int select_decode_num_splits(int batch_nheads_mblocks,
                                           int num_sms, int num_n_blocks,
                                           int max_splits, int active_rows) {
  if (batch_nheads_mblocks >=
      static_cast<int>(0.8f * static_cast<float>(num_sms))) {
    return 1;
  }

  max_splits = min(max_splits, min(num_sms, num_n_blocks));
  if (max_splits <= 1) {
    return 1;
  }

  std::vector<float> efficiency(max_splits, 0.0f);
  float max_efficiency = 0.0f;
  int max_efficiency_split = 1;
  auto is_split_eligible = [num_n_blocks](int num_splits) {
    return num_splits == 1 || utils::div_ceil(num_n_blocks, num_splits) !=
                                  utils::div_ceil(num_n_blocks, num_splits - 1);
  };

  for (int num_splits = 1; num_splits <= max_splits; ++num_splits) {
    if (!is_split_eligible(num_splits)) {
      continue;
    }
    const float n_waves =
        static_cast<float>(batch_nheads_mblocks * num_splits) /
        static_cast<float>(num_sms);
    const float eff = n_waves / ceilf(n_waves);
    efficiency[num_splits - 1] = eff;
    if (eff > max_efficiency) {
      max_efficiency = eff;
      max_efficiency_split = num_splits;
    }
  }

  if (active_rows == 1) {
    return max_efficiency_split;
  }

  for (int num_splits = 1; num_splits <= max_splits; ++num_splits) {
    if (!is_split_eligible(num_splits)) {
      continue;
    }
    if (efficiency[num_splits - 1] >= 0.85f * max_efficiency) {
      return num_splits;
    }
  }

  return 1;
}

template <const int kHeadDim>
static constexpr int getConfigMmaTileSeqLenQP() {
  return 8;
}

template <const int kHeadDim>
static constexpr int getConfigValTileSeqLenK() {
  return 16;
}

template <const int kHeadDim>
static constexpr int getConfigValTileHeadDimV() {
  constexpr int kMmaAtomN = 8;
  constexpr int kMmaTileHeadDimV = 1;
  constexpr int kValTileHeadDimV = (kHeadDim / (kMmaAtomN * kMmaTileHeadDimV));
  return kValTileHeadDimV;
}

static constexpr int getConfigShareSmemQKV() {
#if defined(ENABLE_FFPA_QKV_SMEM_SHARE)
  constexpr int kShareSmemQKV = 1;
#else
  constexpr int kShareSmemQKV = 0;
#endif
  return kShareSmemQKV;
}

template <const int kHeadDim>
static constexpr int getConfigOStorageAccFloat32() {
  // 0/1, The precision of the O storage buffer can differ from
  // that of the MMA, supporting either FP32 or Half precision.
  // FP16 can provide precision to approximately 3-4 decimal places.
  // Thus, if the error does not exceed 1e-3, using FP16 storage is
  // sufficient for most applications.
  return ((kHeadDim <= kMaxDForOStoreFloat32)) ? 1 : 0;
}

template <const int kStageQKV>
static constexpr int getConfigPrefetchQKV() {
  // Prefetch QKV at the appropriate time point.
#if defined(ENABLE_FFPA_PREFETCH_QKV)
  constexpr int kPrefetchQKV = (kStageQKV > 1) ? 1 : 0;
#else
  constexpr int kPrefetchQKV = 0;
#endif
  return kPrefetchQKV;
}

template <const int kStageQK, const int kHeadDim>
static constexpr int getConfigPersistQg2s() {
  // Persist load Q g2s for headdim < 512, more SRAM, but still
  // keep register usage.
#if defined(ENABLE_FFPA_PERSIST_Q_G2S)
  constexpr int kPersistQg2s =
      (kHeadDim < 256) ? 1 : ((kHeadDim <= 320) ? ((kStageQK < 3) ? 1 : 0) : 0);
#else
  constexpr int kPersistQg2s = 0;
#endif
  return kPersistQg2s;
}

static constexpr int getConfigPersistQs2r() {
  // Persist load Q s2r for headdim < 512, more registers,
  // but still keep O(1) SRAM.
#ifdef ENABLE_FFPA_PERSIST_Q_S2R
  constexpr int kPersistQs2r = 1;
#else
  constexpr int kPersistQs2r = 0;
#endif
  return kPersistQs2r;
}

static constexpr int getConfigRegistersPipeKV() {
#ifdef ENABLE_FFPA_REGISTERS_PIPE_KV
  constexpr int kRegPipeKV = 1;
#else
  constexpr int kRegPipeKV = 0;
#endif
  return kRegPipeKV;
}

static constexpr int getConfigPadQ() {
#ifdef ENABLE_FFPA_SMEM_SWIZZLE_Q
  constexpr int kPadQ = 0;
#else
  constexpr int kPadQ = 8;
#endif
  return kPadQ;
}

static constexpr int getConfigPadK() {
#ifdef ENABLE_FFPA_SMEM_SWIZZLE_K
  constexpr int kPadK = 0;
#else
  constexpr int kPadK = 8;
#endif
  return kPadK;
}

static constexpr int getConfigPadV() {
#ifdef ENABLE_FFPA_SMEM_SWIZZLE_V
  constexpr int kPadV = 0;
#else
  constexpr int kPadV = 8;
#endif
  return kPadV;
}

template <const int kNumThreads>
static inline dim3 getConfigBlock() {
  dim3 block(kNumThreads);
  return block;
}

template <const int Br>
static inline dim3 getConfigGrid(const int B, const int H, const int N) {
  // Tr(=N/Br), batch_size x num_heads
  // try grid(N/Br, B * H) or grid(N/Br, H, B)
#ifdef ENABLE_FFPA_LAUNCH_GRID_DNHB
  dim3 grid(utils::div_ceil(N, Br), H, B);
#else
  dim3 grid(utils::div_ceil(N, Br), B * H);
#endif
  return grid;
}

template <const int Br, const int Bc, const int kMmaAtomM, const int kMmaAtomN,
          const int kMmaAtomK, const int kHeadDim, const int kShareSmemQKV,
          const int kPersistQg2s, const int kPersistQs2r, const int kStageQK,
          const int kStagePV, const int kPadQ, const int kPadK, const int kPadV>
static constexpr int getConfigQKVSmemMaxSize() {
  // Unified split-D SMEM size calculation for all headdims.
  constexpr int Q_smem_size =
      ((kPersistQg2s ? (kHeadDim / kMmaAtomK) : kStageQK) *
       (Br * (kMmaAtomK + kPadQ))) *
      2;
  constexpr int K_smem_size = ((kStageQK) * (Bc * (kMmaAtomK + kPadK))) * 2;
  constexpr int V_smem_size = (kStagePV * (Bc * (kMmaAtomN * 2 + kPadV))) * 2;
  constexpr int kQKSmemMaxSize = (Q_smem_size + K_smem_size);
  constexpr int kVSmemMaxSize = V_smem_size;
  // try to let V reuse all Q+K smem after Q@K^T, reduce smem usage.
  constexpr int kQKVSmemMaxSize =
      ((kShareSmemQKV && (!kPersistQg2s))
           ? ((kQKSmemMaxSize > kVSmemMaxSize) ? kQKSmemMaxSize : kVSmemMaxSize)
           : (kQKSmemMaxSize + kVSmemMaxSize));
  // NOTE: R_D registers usage, s=2, d=64, 16 regs; d=128, 32 regs;
  // d=256, 64 regs; d=512, 128 regs; d=1024, 256 regs;
  return kQKVSmemMaxSize;
}

// General native cp.async forward launcher (sm>=80). Computes the native
// block-tile config (MMA atoms, Br/Bc, stages, smem/pad flags) from
// kHeadDim/kStage, then dispatches the Nq==1 split-KV decode fast-path or
// the general split-D FA-2 kernel. Shape invariants are validated upstream
// by launch_ffpa_attn_fwd_template; this launcher only extracts dims.
template <typename kDataType, const int kHeadDim, const int kMmaAccFloat32QK,
          const int kMmaAccFloat32PV, const int kStage>
void launch_native_fwd_split_d_sm80(torch::Tensor Q, torch::Tensor K,
                                    torch::Tensor V, torch::Tensor O,
                                    torch::Tensor attn_bias,
                                    torch::Tensor softmax_lse, int causal,
                                    double softmax_scale, double dropout_p,
                                    int64_t philox_seed,
                                    int64_t philox_offset) {
  constexpr int kMmaAtomM = 16;
  constexpr int kMmaAtomN = 8;
  constexpr int kMmaAtomK = 16;
  constexpr int kMmaTileSeqLenQ = getConfigMmaTileSeqLenQP<kHeadDim>();
  constexpr int kMmaTileSeqLenK = 1;
  constexpr int kMmaTileSeqLenP = getConfigMmaTileSeqLenQP<kHeadDim>();
  constexpr int kMmaTileHeadDimV = 1;
  constexpr int kValTileSeqLenQ = 1;
  constexpr int kValTileSeqLenK = getConfigValTileSeqLenK<kHeadDim>();
  constexpr int kValTileSeqLenP = 1;
  constexpr int kValTileHeadDimV = getConfigValTileHeadDimV<kHeadDim>();
  constexpr int Br = kMmaAtomM * kMmaTileSeqLenQ * kValTileSeqLenQ;
  constexpr int Bc = kMmaAtomN * kMmaTileSeqLenK * kValTileSeqLenK;
  static_assert(Br == Bc,
                "Br must be equal Bc to avoid illegal memory access.");
  constexpr int kNumThreads = WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK;
  constexpr int kOStorageAccFloat32 = getConfigOStorageAccFloat32<kHeadDim>();
  constexpr int kStageQK = kStage;
  constexpr int kStagePV = kStage;
  constexpr int kShareSmemQKV = getConfigShareSmemQKV();
  constexpr int kPrefetchQK = getConfigPrefetchQKV<kStageQK>();
  constexpr int kPrefetchPV = getConfigPrefetchQKV<kStagePV>();
  constexpr int kPersistQs2r = getConfigPersistQs2r();
  constexpr int kPersistQg2s = getConfigPersistQg2s<kStageQK, kHeadDim>();
  constexpr int kRegPipeKV = getConfigRegistersPipeKV();
  constexpr int kPadQ = getConfigPadQ();
  constexpr int kPadK = getConfigPadK();
  constexpr int kPadV = getConfigPadV();
  constexpr int kQKVSmemMaxSize =
      getConfigQKVSmemMaxSize<Br, Bc, kMmaAtomM, kMmaAtomN, kMmaAtomK, kHeadDim,
                              kShareSmemQKV, kPersistQg2s, kPersistQs2r,
                              kStageQK, kStagePV, kPadQ, kPadK, kPadV>();

  const int Nb = Q.size(0);
  const int Nh = Q.size(1);
  const int Nh_kv = K.size(1);
  const int Nq = Q.size(2);
  const int Nkv = K.size(2);
  // Real gmem row width of Q/K/V; == kHeadDim unless the api-layer head_dim
  // pad shrank the rows (kernels zero-fill cols >= d_og, FC-8).
  const int d_og = static_cast<int>(Q.size(3));
  const bool has_attn_bias = attn_bias.numel() != 0;
  const bool has_dropout = dropout_p > 0.0;
  const void* attn_bias_ptr = nullptr;
  int attn_bias_dtype = 0;
  int attn_bias_rowvec = 0;
  long long attn_bias_stride_b = 0, attn_bias_stride_h = 0,
            attn_bias_stride_m = 0, attn_bias_stride_n = 0;
  if (has_attn_bias) {
    attn_bias_ptr = attn_bias.data_ptr();
    attn_bias_dtype = attn_bias.scalar_type() == torch::kHalf       ? 1
                      : attn_bias.scalar_type() == torch::kBFloat16 ? 2
                                                                    : 3;
    attn_bias_stride_b =
        (attn_bias.size(0) == 1 && Nb > 1) ? 0 : attn_bias.stride(0);
    attn_bias_stride_h =
        (attn_bias.size(1) == 1 && Nh > 1) ? 0 : attn_bias.stride(1);
    attn_bias_stride_m =
        (attn_bias.size(2) == 1 && Nq > 1) ? 0 : attn_bias.stride(2);
    attn_bias_stride_n =
        (attn_bias.size(3) == 1 && Nkv > 1) ? 0 : attn_bias.stride(3);
    // PC-0-2 rowvec fast path: stride_m == 0 (bias depends only on the KV
    // column) + stride_n == 1 (columns contiguous) + alignment for pair
    // loads (4B fp16/bf16, 8B fp32) + even (b,h) plane strides so every
    // block's base offset stays pair-aligned. Nq == 1 admits non-zero
    // stride_m too: every row index is a padding row except row 0, and the
    // fast path addresses columns only. The env is read per call so
    // toggling mid-process works.
    const int bias_vec_bytes = attn_bias_dtype == 3 ? 8 : 4;
    attn_bias_rowvec =
        (attn_bias_stride_m == 0 || Nq == 1) && attn_bias_stride_n == 1 &&
        (reinterpret_cast<uintptr_t>(attn_bias_ptr) % bias_vec_bytes == 0) &&
        (attn_bias_stride_b % 2 == 0) && (attn_bias_stride_h % 2 == 0) &&
        getenv("FFPA_BIAS_ROWVEC_DISABLE") == nullptr;
  }

  const dim3 block = getConfigBlock<kNumThreads>();
  const dim3 grid = getConfigGrid<Br>(Nb, Nh, Nq);
  const int Tc = utils::div_ceil(Nkv, Bc);
  const float scale = static_cast<float>(softmax_scale);
  const float dropout_p_f = static_cast<float>(dropout_p);
  const unsigned long long philox_seed_u =
      static_cast<unsigned long long>(philox_seed);
  const unsigned long long philox_offset_u =
      static_cast<unsigned long long>(philox_offset);
  float* softmax_lse_ptr = softmax_lse.data_ptr<float>();

  const c10::cuda::OptionalCUDAGuard device_guard(Q.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  const int num_sms_x2 =
      max(1, at::cuda::getCurrentDeviceProperties()->multiProcessorCount * 2);
  const int num_splits = select_decode_num_splits(
      Nb * Nh * utils::div_ceil(Nq, 16), num_sms_x2, Tc, 128, min(Nq, 16));

  // Fast path for Nq=1, num_splits>1, no attn_bias, no dropout (decode).
  if (Nq == 1 && num_splits > 1 && !has_attn_bias && !has_dropout) {
    const int split_size = utils::div_ceil(Tc, num_splits) * Bc;
    auto scratch_options =
        torch::TensorOptions().dtype(torch::kFloat32).device(Q.device());
    auto partial_out =
        torch::empty({Nb, Nh, num_splits, Nq, kHeadDim}, scratch_options);
    auto chunk_lse = torch::empty({Nb, Nh, num_splits, Nq}, scratch_options);
    const dim3 decode_stage1_grid = dim3(num_splits, Nb * Nh, 1);
    const dim3 decode_stage2_grid = dim3(Nq, Nb * Nh, 1);
    const int decode_threads =
        ((kHeadDim / 8) + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;
    const dim3 decode_block = dim3(decode_threads, 1, 1);
    auto decode_stage1_kernel =
        (split_kv_decode_s1_fwd_sm80<kDataType, kHeadDim, true>);
    decode_stage1_kernel<<<decode_stage1_grid, decode_block, 0, stream>>>(
        reinterpret_cast<kDataType*>(Q.data_ptr()),
        reinterpret_cast<kDataType*>(K.data_ptr()),
        reinterpret_cast<kDataType*>(V.data_ptr()),
        partial_out.data_ptr<float>(), chunk_lse.data_ptr<float>(), Nq, Nkv, Nh,
        Nh_kv, scale, num_splits, split_size, causal, d_og);

    auto decode_stage2_kernel =
        (split_kv_decode_s2_fwd_sm80<kDataType, kHeadDim>);
    decode_stage2_kernel<<<decode_stage2_grid, decode_block, 0, stream>>>(
        partial_out.data_ptr<float>(), chunk_lse.data_ptr<float>(),
        reinterpret_cast<kDataType*>(O.data_ptr()), softmax_lse_ptr, Nq, Nh,
        num_splits);
    return;
  }

  // General split-D FA-2 cp.async path for sm>=80.
  const int smem_size_base = kQKVSmemMaxSize;
  constexpr int kEffShareSmemQKV_LargeD = (kPersistQg2s) ? 0 : kShareSmemQKV;
  constexpr int kEffPersistQs2r_LargeD =
      (kPersistQg2s || kHeadDim > 256) ? 0 : kPersistQs2r;

  auto ffpa_mma_large_d_kernel_func =
      (split_d_fwd_sm80<
          kDataType, kHeadDim, kMmaAtomM, kMmaAtomN, kMmaAtomK, kMmaTileSeqLenQ,
          kMmaTileSeqLenK, kMmaTileSeqLenP, kMmaTileHeadDimV, kValTileSeqLenQ,
          kValTileSeqLenK, kValTileSeqLenP, kValTileHeadDimV, kMmaAccFloat32QK,
          kMmaAccFloat32PV, kOStorageAccFloat32, kPrefetchQK, kPrefetchPV,
          kEffShareSmemQKV_LargeD, kEffPersistQs2r_LargeD, kPersistQg2s,
          kRegPipeKV, kStageQK, kStagePV, kPadQ, kPadK, kPadV>);
  cudaFuncSetAttribute(ffpa_mma_large_d_kernel_func,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       smem_size_base);
  ffpa_mma_large_d_kernel_func<<<grid, block, smem_size_base, stream>>>(
      reinterpret_cast<kDataType*>(Q.data_ptr()),
      reinterpret_cast<kDataType*>(K.data_ptr()),
      reinterpret_cast<kDataType*>(V.data_ptr()),
      reinterpret_cast<kDataType*>(O.data_ptr()), softmax_lse_ptr, Nq, Nkv, Nh,
      Nh_kv, scale, Tc, causal, attn_bias_ptr, attn_bias_dtype,
      attn_bias_rowvec, attn_bias_stride_b, attn_bias_stride_h,
      attn_bias_stride_m, attn_bias_stride_n, dropout_p_f, philox_seed_u,
      philox_offset_u, d_og);
}

// Host-side launcher that picks compile-time configuration (block tile,
// stages, prefetch / share-smem flags, pad vs swizzle, etc.) based on
// ``kHeadDim`` and build macros, then launches the
// ``split_d_fwd_sm80`` kernel on the caller's current CUDA
// stream. Validates Q/K/V/O shape invariants up-front via ``TORCH_CHECK``
// (GQA/MQA head ratio, matching Nkv / D, and the
// ``causal => Nkv >= Nq`` rule).
//
// Template parameters:
//   kDataType            Activation dtype: ``__half`` or ``__nv_bfloat16``.
//   kHeadDim             Head dim D (32, 64, ..., 1024); selects block tile
//   config. kMmaAccFloat32QK/PV  0 -> fp16 MMA accumulator, 1 -> fp32
//   accumulator.
//                        Must both be 1 for bf16 activations.
//   kStage               cp.async pipeline depth used for QK (the PV
//                        depth is derived inside the launcher).
//
// The SM120 TMA path is called from inside this function when ``tma`` is set;

// ============================================================================
// launch_native_fwd_split_d_sm120
// ----------------------------------------------------------------------------
// SM120+ (TMA-capable) launcher.  kQKDChunk=32 (SWIZZLE_64B) or 64
// (SWIZZLE_128B) is selected by the caller based on compute capability
// (sm_90/100 → 64, sm_120a → 32, constrained by per-SM shared memory).
// ============================================================================
#ifdef ENABLE_FFPA_TMA_EXT
template <typename kDataType, const int kHeadDim, const int kMmaAccFloat32QK,
          const int kMmaAccFloat32PV, const int kStage, const int kQKDChunk,
          const int kVDChunk, const int kShareSmemQKV, const int kPersistQg2s,
          const int kMmaTileSeqLenQ, const int kValTileSeqLenK,
          const int kProducerThreads, const int kNonWS>
void launch_native_fwd_split_d_sm120(torch::Tensor Q, torch::Tensor K,
                                     torch::Tensor V, torch::Tensor O,
                                     torch::Tensor attn_bias,
                                     torch::Tensor softmax_lse, int causal,
                                     double softmax_scale, double dropout_p,
                                     int64_t philox_seed,
                                     int64_t philox_offset) {
  constexpr int kMmaAtomM = 16;
  constexpr int kMmaAtomN = 8;
  constexpr int kMmaAtomK = 16;
  constexpr int kMmaTileSeqLenK = 1;
  constexpr int kMmaTileSeqLenP = kMmaTileSeqLenQ;
  constexpr int kMmaTileHeadDimV = 1;
  constexpr int kValTileSeqLenQ = 1;
  constexpr int kValTileSeqLenP = 1;
  constexpr int kValTileHeadDimV = getConfigValTileHeadDimV<kHeadDim>();
  constexpr int Br = kMmaAtomM * kMmaTileSeqLenQ * kValTileSeqLenQ;
  constexpr int Bc = kMmaAtomN * kMmaTileSeqLenK * kValTileSeqLenK;
  static_assert(Br == Bc);
  constexpr int kConsumerThreads =
      WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK;
  constexpr int kEffProducerThreads = kNonWS ? 0 : kProducerThreads;
  constexpr int kTotalThreads = kConsumerThreads + kEffProducerThreads;
  constexpr int kPadQ = 0;
  constexpr int kPadK = 0;
  constexpr int kPadV = 0;
  constexpr int kStageQK = kStage;
  constexpr int kStagePV = kStage;
  constexpr int kOStorageAccFloat32 = getConfigOStorageAccFloat32<kHeadDim>();
  constexpr int kQKDChunks = kHeadDim / kQKDChunk;

  const int Nb = Q.size(0);
  const int Nh = Q.size(1);
  const int Nh_kv = K.size(1);
  const int Nq = Q.size(2);
  const int Nkv = K.size(2);
  // Real gmem row width of Q/K/V; == kHeadDim unless the api-layer head_dim
  // pad shrank the rows (TMA OOB fill zero-pads the descriptor minor axis,
  // FC-8). Q.size(3)==kHeadDim when a caller (e.g. the CUTE_TMA bias/dropout
  // fallback) already materialized padded tensors.
  const int d_og = static_cast<int>(Q.size(3));
  const bool has_attn_bias = attn_bias.numel() != 0;
  const bool has_dropout = dropout_p > 0.0;
  const void* attn_bias_ptr = nullptr;
  int attn_bias_dtype = 0;
  int attn_bias_rowvec = 0;
  long long attn_bias_stride_b = 0, attn_bias_stride_h = 0,
            attn_bias_stride_m = 0, attn_bias_stride_n = 0;
  if (has_attn_bias) {
    attn_bias_ptr = attn_bias.data_ptr();
    attn_bias_dtype = attn_bias.scalar_type() == torch::kHalf       ? 1
                      : attn_bias.scalar_type() == torch::kBFloat16 ? 2
                                                                    : 3;
    attn_bias_stride_b =
        (attn_bias.size(0) == 1 && Nb > 1) ? 0 : attn_bias.stride(0);
    attn_bias_stride_h =
        (attn_bias.size(1) == 1 && Nh > 1) ? 0 : attn_bias.stride(1);
    attn_bias_stride_m =
        (attn_bias.size(2) == 1 && Nq > 1) ? 0 : attn_bias.stride(2);
    attn_bias_stride_n =
        (attn_bias.size(3) == 1 && Nkv > 1) ? 0 : attn_bias.stride(3);
    // PC-0-2 rowvec fast path (see the sm80 launcher for the gate rules;
    // Nq == 1 admits non-zero stride_m since only row 0 is a real row and
    // the fast path addresses columns only).
    const int bias_vec_bytes = attn_bias_dtype == 3 ? 8 : 4;
    attn_bias_rowvec =
        (attn_bias_stride_m == 0 || Nq == 1) && attn_bias_stride_n == 1 &&
        (reinterpret_cast<uintptr_t>(attn_bias_ptr) % bias_vec_bytes == 0) &&
        (attn_bias_stride_b % 2 == 0) && (attn_bias_stride_h % 2 == 0) &&
        getenv("FFPA_BIAS_ROWVEC_DISABLE") == nullptr;
  }

  const dim3 block(kTotalThreads, 1, 1);
  const dim3 grid = getConfigGrid<Br>(Nb, Nh, Nq);
  const int Tc = utils::div_ceil(Nkv, Bc);
  const float scale = static_cast<float>(softmax_scale);
  const float dropout_p_f = static_cast<float>(dropout_p);
  const unsigned long long philox_seed_u =
      static_cast<unsigned long long>(philox_seed);
  const unsigned long long philox_offset_u =
      static_cast<unsigned long long>(philox_offset);
  float* softmax_lse_ptr = softmax_lse.data_ptr<float>();

  const c10::cuda::OptionalCUDAGuard device_guard(Q.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  const int total_rows = Nb * Nh * Nq;
  const int total_kv_rows = Nb * Nh_kv * Nkv;
  constexpr CUtensorMapSwizzle qk_swizzle =
      (kQKDChunk == 64)   ? CU_TENSOR_MAP_SWIZZLE_128B
      : (kQKDChunk == 32) ? CU_TENSOR_MAP_SWIZZLE_64B
                          : CU_TENSOR_MAP_SWIZZLE_32B;
  constexpr CUtensorMapSwizzle v_swizzle =
      (kVDChunk == 64)   ? CU_TENSOR_MAP_SWIZZLE_128B
      : (kVDChunk == 32) ? CU_TENSOR_MAP_SWIZZLE_64B
                         : CU_TENSOR_MAP_SWIZZLE_32B;

  auto make_desc = [&](void* gmem_ptr, int rows, int box_minor,
                       CUtensorMapSwizzle sw) -> CUtensorMap {
    ffpa::tma::Copy2DDescriptorParams<kDataType> params;
    params.global_address = reinterpret_cast<kDataType*>(gmem_ptr);
    // minor_dim/stride follow the real row width; boxes reaching past it are
    // zero-filled by TMA (OOB_FILL_NONE = zeros), so pad cols contribute 0.
    params.minor_dim = d_og;
    params.major_dim = rows;
    params.major_stride_bytes = static_cast<uint64_t>(d_og) * sizeof(kDataType);
    params.box_minor_dim = box_minor;
    params.box_major_dim = Bc;
    params.swizzle = sw;
    params.l2_promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
    params.oob_fill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
    return ffpa::tma::make_2d_copy_desc<kDataType>(params);
  };
  CUtensorMap tma_q_desc =
      make_desc(Q.data_ptr(), total_rows, kQKDChunk, qk_swizzle);
  CUtensorMap tma_k_desc =
      make_desc(K.data_ptr(), total_kv_rows, kQKDChunk, qk_swizzle);
  CUtensorMap tma_v_desc =
      make_desc(V.data_ptr(), total_kv_rows, kVDChunk, v_swizzle);
  CUtensorMap *tma_q_d = nullptr, *tma_k_d = nullptr, *tma_v_d = nullptr;
  cudaMalloc(&tma_q_d, sizeof(CUtensorMap));
  cudaMalloc(&tma_k_d, sizeof(CUtensorMap));
  cudaMalloc(&tma_v_d, sizeof(CUtensorMap));
  cudaMemcpy(tma_q_d, &tma_q_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
  cudaMemcpy(tma_k_d, &tma_k_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
  cudaMemcpy(tma_v_d, &tma_v_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);

  constexpr int kQTileBytes = Br * kQKDChunk * sizeof(kDataType);
  constexpr int kKTileBytes = Bc * kQKDChunk * sizeof(kDataType);
  constexpr int kVTileBytes = Bc * kVDChunk * sizeof(kDataType);
  constexpr int kKVStageBytes =
      kKTileBytes > kVTileBytes ? kKTileBytes : kVTileBytes;
  constexpr int kQKVSmemBytes =
      kPersistQg2s
          ? (kHeadDim / kQKDChunk) * kQTileBytes + kStageQK * kKVStageBytes
      : kShareSmemQKV
          ? kStageQK * ((kQTileBytes + kKTileBytes) > kVTileBytes
                            ? (kQTileBytes + kKTileBytes)
                            : kVTileBytes)
          : (kStageQK * (kQTileBytes + kKTileBytes) + kStagePV * kVTileBytes);
  constexpr int kBarrierBytes =
      (kStageQK * 2 + kStagePV * 2 + ((kShareSmemQKV || kPersistQg2s) ? 2 : 0) +
       (kPersistQg2s ? 1 : 0)) *
      sizeof(ffpa::tma::barrier_t);
  constexpr int kSmemBytes = kQKVSmemBytes + kBarrierBytes;

  auto kernel_func =
      (split_d_fwd_sm120<
          kDataType, kHeadDim, kMmaAtomM, kMmaAtomN, kMmaAtomK, kMmaTileSeqLenQ,
          kMmaTileSeqLenK, kMmaTileSeqLenP, kMmaTileHeadDimV, kValTileSeqLenQ,
          kValTileSeqLenK, kValTileSeqLenP, kValTileHeadDimV, kMmaAccFloat32QK,
          kMmaAccFloat32PV, kOStorageAccFloat32, kStageQK, kStagePV, kPadQ,
          kPadK, kPadV, kQKDChunk, kVDChunk, kShareSmemQKV, kPersistQg2s,
          kProducerThreads, kNonWS>);
  cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       kSmemBytes);
  kernel_func<<<grid, block, kSmemBytes, stream>>>(
      tma_q_d, tma_k_d, tma_v_d, reinterpret_cast<kDataType*>(O.data_ptr()),
      softmax_lse_ptr, Nq, Nkv, Nh, Nh_kv, scale, Tc, causal, attn_bias_ptr,
      attn_bias_dtype, attn_bias_rowvec, attn_bias_stride_b, attn_bias_stride_h,
      attn_bias_stride_m, attn_bias_stride_n, dropout_p_f, philox_seed_u,
      philox_offset_u);
  cudaFree(tma_q_d);
  cudaFree(tma_k_d);
  cudaFree(tma_v_d);
}
#endif  // ENABLE_FFPA_TMA_EXT

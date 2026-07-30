#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "fwd_sm80.cuh"
#include "split_kv.cuh"
#ifdef ENABLE_FFPA_TMA_EXT
#include "fwd_sm120.cuh"
#include "tma.cuh"
#endif
#ifdef ENABLE_FFPA_CUTE_EXT
#include "cute/fwd_sm80.cuh"
#ifdef ENABLE_FFPA_TMA_EXT
#include "cute/fwd_sm120.cuh"
#endif
#endif
using namespace ffpa;

static constexpr int kMaxDForOStoreFloat32 = 512;
// for D up to 512; Use fp16/bf16 for D > 512 to save registers, since

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

// Host-side launcher that picks compile-time configuration (block tile,
// stages, prefetch / share-smem flags, pad vs swizzle, etc.) based on
// ``kHeadDim`` and build macros, then launches the
// ``ffpa_attn_split_d_fwd_template`` kernel on the caller's current CUDA
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
// forward declaration (definition follows after the legacy launcher).

#ifdef ENABLE_FFPA_TMA_EXT
template <typename kDataType, const int kHeadDim, const int kAccQK,
          const int kAccPV, const int kStage, const int kQKDChunk,
          const int kVDChunk = kQKDChunk, const int kShareSmemQKV = 0,
          const int kPersistQg2s = 0, const int kMmaTileSeqLenQ = 8,
          const int kValTileSeqLenK = 16, const int kProducerThreads = 128,
          const int kNonWS = 0>
void launch_ffpa_attn_fwd_template_sm120(torch::Tensor Q, torch::Tensor K,
                                         torch::Tensor V, torch::Tensor O,
                                         torch::Tensor attn_bias,
                                         torch::Tensor softmax_lse, int causal,
                                         double softmax_scale, double dropout_p,
                                         int64_t philox_seed,
                                         int64_t philox_offset);
#endif  // ENABLE_FFPA_TMA_EXT

#ifdef ENABLE_FFPA_CUTE_EXT
template <typename kDataType, const int kHeadDim, const int kStage,
          const int kQKDChunk = 32, const int kVDChunk = 64>
void launch_ffpa_attn_split_d_fwd_cute_sm120(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
    torch::Tensor attn_bias, torch::Tensor softmax_lse, int causal,
    double softmax_scale, double dropout_p, int64_t philox_seed,
    int64_t philox_offset);

template <typename kDataType, const int kHeadDim, const int kStage,
          const int kQKDChunk = 32, const int kVDChunk = 64>
void launch_ffpa_attn_split_d_fwd_cute(torch::Tensor Q, torch::Tensor K,
                                       torch::Tensor V, torch::Tensor O,
                                       torch::Tensor attn_bias,
                                       torch::Tensor softmax_lse, int causal,
                                       double softmax_scale, double dropout_p,
                                       int64_t philox_seed,
                                       int64_t philox_offset);
#endif  // ENABLE_FFPA_CUTE_EXT

// Runtime arguments:
//   Q, K, V, O     : BHND tensors as described in the kernel template docs.
//   causal         : 0/1 runtime flag. Non-zero enables causal masking with
//                    queries aligned to the KV tail; requires Nkv >= Nq.
//   softmax_scale  : pre-softmax scaling factor applied to QK^T. Matches the
//                    flash-attn naming; the Python wrapper defaults it to
//                    ``1 / sqrt(D)`` when the caller does not supply one.
// Runtime ``tma`` is accepted for API compatibility but ignored. The legacy
// SM90 TMA CUDA branch is kept under csrc/cuffpa/deprecated; active native
// forward launches always use the architecture-agnostic templates here.
template <typename kDataType, const int kHeadDim, const int kMmaAccFloat32QK,
          const int kMmaAccFloat32PV, const int kStage>
void launch_ffpa_attn_fwd_template(torch::Tensor Q, torch::Tensor K,
                                   torch::Tensor V, torch::Tensor O,
                                   torch::Tensor attn_bias,
                                   torch::Tensor softmax_lse, int causal,
                                   double softmax_scale, double dropout_p,
                                   int64_t philox_seed, int64_t philox_offset,
                                   int tma) {
  // Q,K,V,O with [B, H, N, D] layout, B=batch, H=head, N=seqlen, D=dim
  // TODO: support BNHD layout, Q,K,V,O with [B, N, H, D] layout.
  constexpr int kMmaAtomM = 16;
  constexpr int kMmaAtomN = 8;
  constexpr int kMmaAtomK = 16;
  // Split-Q(FA-2) Algo, Tile MMA across Q and keep KV access for all MMAs.
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
  // Apply different multi stages policy for QK and V.
  // TODO: tune stages for Q@K and P@V.
  constexpr int kStageQK = kStage;  // <= FFPA_BUILD_MAX_STAGES
  constexpr int kStagePV = kStage;  // <= FFPA_BUILD_MAX_STAGES
  // Prefetch QKV, Persist Q g2s/s2r, Shared QKV smem.
  constexpr int kShareSmemQKV = getConfigShareSmemQKV();
  constexpr int kPrefetchQK = getConfigPrefetchQKV<kStageQK>();
  constexpr int kPrefetchPV = getConfigPrefetchQKV<kStagePV>();
  constexpr int kPersistQs2r = getConfigPersistQs2r();
  constexpr int kPersistQg2s = getConfigPersistQg2s<kStageQK, kHeadDim>();
  constexpr int kRegPipeKV = getConfigRegistersPipeKV();
  // QKV smem swizzle, 0 for smem swizzle, !0 for smem padding.
  constexpr int kPadQ = getConfigPadQ();
  constexpr int kPadK = getConfigPadK();
  constexpr int kPadV = getConfigPadV();
  // Calculate SRAM size needed for per block.
  constexpr int kQKVSmemMaxSize =
      getConfigQKVSmemMaxSize<Br, Bc, kMmaAtomM, kMmaAtomN, kMmaAtomK, kHeadDim,
                              kShareSmemQKV, kPersistQg2s, kPersistQs2r,
                              kStageQK, kStagePV, kPadQ, kPadK, kPadV>();
  TORCH_CHECK(K.size(0) == Q.size(0) && V.size(0) == Q.size(0),
              "ffpa_attn: Q/K/V must share the same batch size");
  TORCH_CHECK(K.size(1) == V.size(1),
              "ffpa_attn: K and V must share the same num_heads (Nh_kv)");
  TORCH_CHECK(
      Q.size(1) % K.size(1) == 0,
      "ffpa_attn: Q num_heads must be an integer multiple of K/V num_heads "
      "(GQA/MQA group_size = Nh_q / Nh_kv)");
  TORCH_CHECK(K.size(2) == V.size(2),
              "ffpa_attn: K and V must have identical sequence length (Nkv)");
  TORCH_CHECK(K.size(3) == Q.size(3) && V.size(3) == Q.size(3),
              "ffpa_attn: Q/K/V must share the same head dim");
  TORCH_CHECK(causal == 0 || K.size(2) >= Q.size(2),
              "ffpa_attn: causal attention requires Nkv >= Nq (queries are "
              "aligned to the tail of the KV sequence)");
  const int Nb = Q.size(0);
  const int Nh = Q.size(1);  // Q head count (Nh_q); used for grid fan-out.
  const int Nh_kv =
      K.size(1);  // K/V head count; Nh % Nh_kv == 0 asserted above.
  // Cross-attention: Q seqlen (Nq) may differ from KV seqlen (Nkv).
  const int Nq = Q.size(2);
  const int Nkv = K.size(2);
  const bool has_attn_bias = attn_bias.numel() != 0;
  const bool has_dropout = dropout_p > 0.0;
  TORCH_CHECK(causal == 0 || !has_attn_bias,
              "ffpa_attn: explicit attn_mask should not be set when causal "
              "attention is enabled");
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
                "ffpa_attn: attn_mask key dimension must be 1 or Nkv");
    TORCH_CHECK(attn_bias.stride(3) == 1,
                "ffpa_attn: normalized attn_mask must be contiguous along the "
                "key dimension");
    const auto bias_type = attn_bias.scalar_type();
    if (bias_type == torch::kHalf) {
      attn_bias_dtype = 1;
    } else if (bias_type == torch::kBFloat16) {
      attn_bias_dtype = 2;
    } else if (bias_type == torch::kFloat32) {
      attn_bias_dtype = 3;
    } else {
      TORCH_CHECK(false,
                  "ffpa_attn: attn_mask dtype must be fp16, bf16, or fp32");
    }
    TORCH_CHECK(bias_type == torch::kFloat32 || bias_type == Q.scalar_type(),
                "ffpa_attn: attn_mask dtype must be fp32 or match Q dtype");
    attn_bias_ptr = attn_bias.data_ptr();
    attn_bias_stride_b =
        (attn_bias.size(0) == 1 && Nb > 1) ? 0 : attn_bias.stride(0);
    attn_bias_stride_h =
        (attn_bias.size(1) == 1 && Nh > 1) ? 0 : attn_bias.stride(1);
    attn_bias_stride_m =
        (attn_bias.size(2) == 1 && Nq > 1) ? 0 : attn_bias.stride(2);
    attn_bias_stride_n =
        (attn_bias.size(3) == 1 && Nkv > 1) ? 0 : attn_bias.stride(3);
  }
  // Seqlen (Nq, Nkv) no longer has to be a multiple of max(Br, Bc): the
  // kernel handles the tail tile via cp.async zero-fill, softmax -inf
  // masking and a per-row store predicate. div_ceil(Nkv, Bc) below still
  // yields the right Tc for partial last KV tiles.

  const dim3 block = getConfigBlock<kNumThreads>();  // 4/8 warps per block
  // grid is driven by Q row tiles; KV tile count Tc is driven by Nkv.
  const dim3 grid = getConfigGrid<Br>(Nb, Nh, Nq);
  const int Tc = utils::div_ceil(Nkv, Bc);  // Tc K_tile[Bc,d]
  const float scale = static_cast<float>(softmax_scale);
  const float dropout_p_f = static_cast<float>(dropout_p);
  const unsigned long long philox_seed_u =
      static_cast<unsigned long long>(philox_seed);
  const unsigned long long philox_offset_u =
      static_cast<unsigned long long>(philox_offset);
  float* softmax_lse_ptr = softmax_lse.data_ptr<float>();

  // Launch on the caller's current CUDA stream so the kernel participates
  // correctly in multi-stream pipelines. Without this the kernel would
  // default to stream 0 and race against user-side non-default streams.
  const c10::cuda::OptionalCUDAGuard device_guard(Q.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  const int num_sms_x2 =
      max(1, at::cuda::getCurrentDeviceProperties()->multiProcessorCount * 2);
  const int num_splits = select_decode_num_splits(
      Nb * Nh * utils::div_ceil(Nq, 16), num_sms_x2, Tc, 128, min(Nq, 16));

  // Fast path for Nq=1, num_splits>1, no attn_bias, no dropout. (decode cases)
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
    // Pure gemv implementation for Nq=1 case, do the reduction in stage 2.
    auto decode_stage1_kernel =
        (ffpa_attn_split_kv_decode_stage1_template<kDataType, kHeadDim, true>);
    decode_stage1_kernel<<<decode_stage1_grid, decode_block, 0, stream>>>(
        reinterpret_cast<kDataType*>(Q.data_ptr()),
        reinterpret_cast<kDataType*>(K.data_ptr()),
        reinterpret_cast<kDataType*>(V.data_ptr()),
        partial_out.data_ptr<float>(), chunk_lse.data_ptr<float>(), Nq, Nkv, Nh,
        Nh_kv, scale, num_splits, split_size, causal);

    auto decode_stage2_kernel =
        (ffpa_attn_split_kv_decode_stage2_template<kDataType, kHeadDim>);
    decode_stage2_kernel<<<decode_stage2_grid, decode_block, 0, stream>>>(
        partial_out.data_ptr<float>(), chunk_lse.data_ptr<float>(),
        reinterpret_cast<kDataType*>(O.data_ptr()), softmax_lse_ptr, Nq, Nh,
        num_splits);
    return;
  }

  // SM120 TMA path: when ``tma`` is set and the device is TMA-capable
  // (sm_90+), delegate to the TMA launcher. Falls back to the legacy
  // cp.async path on older hardware. NOTE: NO WGMMA on sm_120a.
  // See fwd_sm120.cuh header for register pressure analysis
  // and why sm_120a uses non-WS mode (kNonWS=1).
  //
  // TMA path dispatch:
  //   sm_120a: non-WS (kNonWS=1). All 256 threads do MMA, thread 0 issues
  //     TMA inline. Eliminates WS if/else register allocation penalty
  //     (168→255 regs, 0 spills). +2-7% vs legacy cp.async baseline.
  //   sm_90/100: WS (kNonWS=0). setmaxnreg effective, 228KB smem allows
  //     deep pipeline. Unverified on real hardware.
#ifdef ENABLE_FFPA_TMA_EXT
  if (tma) {
    auto prop = at::cuda::getCurrentDeviceProperties();
    if (prop->major >= 9) {
      if (prop->major == 9 || prop->major == 10) {
        // sm_90/100 (228 KB smem): WS path, setmaxnreg effective.
        if (!has_attn_bias && !has_dropout && kHeadDim <= 512) {
          // w/ kPersistQg2s = 1
          launch_ffpa_attn_fwd_template_sm120<
              kDataType, kHeadDim, kMmaAccFloat32QK, kMmaAccFloat32PV, kStage,
              64 /*kQKDChunk*/, 64 /*kVDChunk*/, 0 /*kShareSmemQKV*/,
              1 /*kPersistQg2s*/, 8 /*kMmaTileSeqLenQ*/, 16 /*kValTileSeqLenK*/,
              128 /*kProducerThreads*/, 0 /*kNonWS*/>(
              Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
              dropout_p, philox_seed, philox_offset);
        } else {
          // w/ kPersistQg2s = 0
          launch_ffpa_attn_fwd_template_sm120<
              kDataType, kHeadDim, kMmaAccFloat32QK, kMmaAccFloat32PV, kStage,
              64 /*kQKDChunk*/, 64 /*kVDChunk*/, 0 /*kShareSmemQKV*/,
              0 /*kPersistQg2s*/, 8 /*kMmaTileSeqLenQ*/, 16 /*kValTileSeqLenK*/,
              128 /*kProducerThreads*/, 0 /*kNonWS*/>(
              Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
              dropout_p, philox_seed, philox_offset);
        }
      } else {
        // sm_120a (99 KB smem): non-WS path.
#ifdef ENABLE_FFPA_CUTE_EXT
        // CuTe kernel: kHeadDim%64==0 → kVDChunk=64; %32==0 → kVDChunk=32.
        // NOTE: CuTe kernel's bias/dropout paths are functional but ~2x slower
        // than the non-WS TMA template kernel due to register pressure from
        // the 128x128 rowcol tensor abstraction (64 score regs simultaneously
        // live + addressing temps → spills). Prefer the non-WS TMA fallback
        // when bias/dropout is active; CuTe handles the clean path only.
        if constexpr (kHeadDim % 64 == 0) {
          if (!has_attn_bias && !has_dropout) {
            launch_ffpa_attn_split_d_fwd_cute_sm120<kDataType, kHeadDim, kStage,
                                                    32, 64>(
                Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
                dropout_p, philox_seed, philox_offset);
          } else {
            launch_ffpa_attn_fwd_template_sm120<
                kDataType, kHeadDim, kMmaAccFloat32QK, kMmaAccFloat32PV,
                (kStage > 3 ? 3 : kStage), 32 /*kQKDChunk*/, 64 /*kVDChunk*/,
                0 /*kShareSmemQKV*/, 0 /*kPersistQg2s*/, 8 /*kMmaTileSeqLenQ*/,
                16 /*kValTileSeqLenK*/, 128 /*kProducerThreads*/, 1 /*kNonWS*/>(
                Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
                dropout_p, philox_seed, philox_offset);
          }
        } else if constexpr (kHeadDim % 32 == 0) {
          if (!has_attn_bias && !has_dropout) {
            launch_ffpa_attn_split_d_fwd_cute_sm120<kDataType, kHeadDim, kStage,
                                                    32, 32>(
                Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
                dropout_p, philox_seed, philox_offset);
          } else {
            launch_ffpa_attn_fwd_template_sm120<
                kDataType, kHeadDim, kMmaAccFloat32QK, kMmaAccFloat32PV,
                (kStage > 3 ? 3 : kStage), 32 /*kQKDChunk*/, 64 /*kVDChunk*/,
                0 /*kShareSmemQKV*/, 0 /*kPersistQg2s*/, 8 /*kMmaTileSeqLenQ*/,
                16 /*kValTileSeqLenK*/, 128 /*kProducerThreads*/, 1 /*kNonWS*/>(
                Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
                dropout_p, philox_seed, philox_offset);
          }
        } else {
          launch_ffpa_attn_fwd_template_sm120<
              kDataType, kHeadDim, kMmaAccFloat32QK, kMmaAccFloat32PV,
              (kStage > 3 ? 3 : kStage), 32 /*kQKDChunk*/, 64 /*kVDChunk*/,
              0 /*kShareSmemQKV*/, 0 /*kPersistQg2s*/, 8 /*kMmaTileSeqLenQ*/,
              16 /*kValTileSeqLenK*/, 128 /*kProducerThreads*/, 1 /*kNonWS*/>(
              Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
              dropout_p, philox_seed, philox_offset);
        }
#else
        // kQKDChunk=32 (SWIZZLE_64B), kVDChunk=64, S≤3.
        // smem per stage: Q=128×32×2B=8KB, K=128×32×2B=8KB, V=128×64×2B=16KB
        // total: 3×(8+8+16)KB = 96KB < 99KB.
        launch_ffpa_attn_fwd_template_sm120<
            kDataType, kHeadDim, kMmaAccFloat32QK, kMmaAccFloat32PV,
            (kStage > 3 ? 3 : kStage), 32 /*kQKDChunk*/, 64 /*kVDChunk*/,
            0 /*kShareSmemQKV*/, 0 /*kPersistQg2s*/, 8 /*kMmaTileSeqLenQ*/,
            16 /*kValTileSeqLenK*/, 128 /*kProducerThreads*/, 1 /*kNonWS*/>(
            Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
            dropout_p, philox_seed, philox_offset);
#endif
      }
      return;
    }
  }
#endif  // ENABLE_FFPA_TMA_EXT

#ifdef ENABLE_FFPA_CUTE_EXT
  // CuTe cp.async path: sm_80+ without TMA (tma=0 or sm<90).
  if constexpr (kHeadDim % 64 == 0) {
    launch_ffpa_attn_split_d_fwd_cute<kDataType, kHeadDim, kStage, 16, 64>(
        Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale, dropout_p,
        philox_seed, philox_offset);
    return;
  } else if constexpr (kHeadDim % 32 == 0) {
    launch_ffpa_attn_split_d_fwd_cute<kDataType, kHeadDim, kStage, 16, 32>(
        Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale, dropout_p,
        philox_seed, philox_offset);
    return;
  }
#endif  // ENABLE_FFPA_CUTE_EXT

  // General path for sm>=80 architectures.
  const int smem_size_base = kQKVSmemMaxSize;

  constexpr int kEffShareSmemQKV_LargeD = (kPersistQg2s) ? 0 : kShareSmemQKV;
  constexpr int kEffPersistQs2r_LargeD =
      (kPersistQg2s || kHeadDim > 256) ? 0 : kPersistQs2r;

  auto ffpa_mma_large_d_kernel_func =
      (ffpa_attn_split_d_fwd_template<
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
      attn_bias_stride_b, attn_bias_stride_h, attn_bias_stride_m,
      attn_bias_stride_n, dropout_p_f, philox_seed_u, philox_offset_u);
}

// ============================================================================
// launch_ffpa_attn_fwd_template_sm120
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
void launch_ffpa_attn_fwd_template_sm120(torch::Tensor Q, torch::Tensor K,
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
  const bool has_attn_bias = attn_bias.numel() != 0;
  const bool has_dropout = dropout_p > 0.0;
  const void* attn_bias_ptr = nullptr;
  int attn_bias_dtype = 0;
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
    params.minor_dim = kHeadDim;
    params.major_dim = rows;
    params.major_stride_bytes = kHeadDim * sizeof(kDataType);
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
      (ffpa_attn_split_d_fwd_template_sm120<
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
      attn_bias_dtype, attn_bias_stride_b, attn_bias_stride_h,
      attn_bias_stride_m, attn_bias_stride_n, dropout_p_f, philox_seed_u,
      philox_offset_u);
  cudaFree(tma_q_d);
  cudaFree(tma_k_d);
  cudaFree(tma_v_d);
}
#endif  // ENABLE_FFPA_TMA_EXT

#ifdef ENABLE_FFPA_CUTE_EXT
template <typename kDataType, const int kHeadDim, const int kStage,
          const int kQKDChunk, const int kVDChunk>
void launch_ffpa_attn_split_d_fwd_cute_sm120(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
    torch::Tensor attn_bias, torch::Tensor softmax_lse, int causal,
    double softmax_scale, double dropout_p, int64_t philox_seed,
    int64_t philox_offset) {
  using namespace cute;

  constexpr int kBr = 128;
  constexpr int kBc = 128;
  constexpr int kStagesQK = (kStage > 3 ? 3 : kStage);
  constexpr int kStagesPV = kStagesQK;
  constexpr int kNumThreads = kBr / 16 * 32;

  using CuteElement = std::conditional_t<std::is_same_v<kDataType, __half>,
                                         cutlass::half_t, cutlass::bfloat16_t>;
  using Traits = ffpa_cute::FFPAAttnCuTeTraits<kHeadDim, kBr, kBc, kQKDChunk,
                                               kVDChunk, CuteElement>;
  using SmemLayoutQ = typename Traits::SmemLayoutQ;
  using SmemLayoutK = typename Traits::SmemLayoutK;
  using SmemLayoutV = typename Traits::SmemLayoutV;

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
        tma_q, tma_k, tma_v, O_ptr, softmax_lse_ptr, Nq, Nkv, Nh, Nh_kv, scale,
        Tc, causal, total_q_rows, total_kv_rows, attn_bias_ptr, attn_bias_dtype,
        attn_bias_stride_b, attn_bias_stride_h, attn_bias_stride_m,
        attn_bias_stride_n, dropout_p_f, philox_seed_u, philox_offset_u);
  };

  using TmaQ = decltype(tma_q);
  using TmaK = decltype(tma_k);
  using TmaV = decltype(tma_v);
  if (has_attn_bias && has_dropout) {
    launch_variant(
        ffpa_attn_split_d_fwd_cute_sm120<Traits, TmaQ, TmaK, TmaV, kStagesQK,
                                         kStagesPV, 1, 1>);
  } else if (has_attn_bias) {
    launch_variant(
        ffpa_attn_split_d_fwd_cute_sm120<Traits, TmaQ, TmaK, TmaV, kStagesQK,
                                         kStagesPV, 1, 0>);
  } else if (has_dropout) {
    launch_variant(
        ffpa_attn_split_d_fwd_cute_sm120<Traits, TmaQ, TmaK, TmaV, kStagesQK,
                                         kStagesPV, 0, 1>);
  } else {
    launch_variant(
        ffpa_attn_split_d_fwd_cute_sm120<Traits, TmaQ, TmaK, TmaV, kStagesQK,
                                         kStagesPV, 0, 0>);
  }
}

template <typename kDataType, const int kHeadDim, const int kStage,
          const int kQKDChunk, const int kVDChunk>
void launch_ffpa_attn_split_d_fwd_cute(torch::Tensor Q, torch::Tensor K,
                                       torch::Tensor V, torch::Tensor O,
                                       torch::Tensor attn_bias,
                                       torch::Tensor softmax_lse, int causal,
                                       double softmax_scale, double dropout_p,
                                       int64_t philox_seed,
                                       int64_t philox_offset) {
  using namespace cute;

  constexpr int kBr = 128;
  constexpr int kBc = 128;
  constexpr int kStagesQK = (kStage > 3 ? 3 : kStage);
  constexpr int kStagesPV = kStagesQK;
  constexpr int kNumThreads = kBr / 16 * 32;

  using CuteElement = std::conditional_t<std::is_same_v<kDataType, __half>,
                                         cutlass::half_t, cutlass::bfloat16_t>;
  using Traits = ffpa_cute::FFPAAttnCuTeTraits<kHeadDim, kBr, kBc, kQKDChunk,
                                               kVDChunk, CuteElement>;

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

  constexpr int kQTileBytes = kBr * kQKDChunk * sizeof(CuteElement);
  constexpr int kKTileBytes = kBc * kQKDChunk * sizeof(CuteElement);
  constexpr int kVTileBytes = kBc * kVDChunk * sizeof(CuteElement);
  constexpr int kSmemBytes = kStagesQK * kQTileBytes + kStagesQK * kKTileBytes +
                             kStagesPV * kVTileBytes;

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
    launch_variant(
        ffpa_attn_split_d_fwd_cute<Traits, kStagesQK, kStagesPV, 1, 1>);
  } else if (has_attn_bias) {
    launch_variant(
        ffpa_attn_split_d_fwd_cute<Traits, kStagesQK, kStagesPV, 1, 0>);
  } else if (has_dropout) {
    launch_variant(
        ffpa_attn_split_d_fwd_cute<Traits, kStagesQK, kStagesPV, 0, 1>);
  } else {
    launch_variant(
        ffpa_attn_split_d_fwd_cute<Traits, kStagesQK, kStagesPV, 0, 0>);
  }
}
#endif  // ENABLE_FFPA_CUTE_EXT

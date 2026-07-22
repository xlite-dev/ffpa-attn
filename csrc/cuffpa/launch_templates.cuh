#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "ffpa_attn_fwd.cuh"
#include "ffpa_attn_fwd_split_kv.cuh"
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
  (void)tma;
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

  const int smem_size_base = kQKVSmemMaxSize;

#define LAUNCH_TEMPLATE_FUNC_BASE(TEMPLATE_FUNC)                            \
  cudaFuncSetAttribute(TEMPLATE_FUNC,                                       \
                       cudaFuncAttributeMaxDynamicSharedMemorySize,         \
                       smem_size_base);                                     \
  TEMPLATE_FUNC<<<grid, block, smem_size_base, stream>>>(                   \
      reinterpret_cast<kDataType*>(Q.data_ptr()),                           \
      reinterpret_cast<kDataType*>(K.data_ptr()),                           \
      reinterpret_cast<kDataType*>(V.data_ptr()),                           \
      reinterpret_cast<kDataType*>(O.data_ptr()), softmax_lse_ptr, Nq, Nkv, \
      Nh, Nh_kv, scale, Tc, causal, attn_bias_ptr, attn_bias_dtype,         \
      attn_bias_stride_b, attn_bias_stride_h, attn_bias_stride_m,           \
      attn_bias_stride_n, dropout_p_f, philox_seed_u, philox_offset_u);

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
  LAUNCH_TEMPLATE_FUNC_BASE(ffpa_mma_large_d_kernel_func);

#undef LAUNCH_TEMPLATE_FUNC_BASE
}

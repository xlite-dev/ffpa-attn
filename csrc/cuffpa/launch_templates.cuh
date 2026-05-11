#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "ffpa_attn_fwd_templates.cuh"
using namespace ffpa;

static constexpr int kMaxDForSmallDKernel = 128;
// Always use fp32 accumulators for O to reduce numerical instability
// for D up to 512; Use fp16/bf16 for D > 512 to save registers, since
// the larger D may cuase register spilling when using fp32 accumulators.
static constexpr int kMaxDForOStoreFloat32 = 512;
static constexpr int kMaxDForSmallBlockTile = 256;

static inline int select_decode_num_splits(int batch_nheads_mblocks, int num_sms, int num_n_blocks,
                                           int max_splits, int active_rows) {
  if (batch_nheads_mblocks >= static_cast<int>(0.8f * static_cast<float>(num_sms))) {
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
        static_cast<float>(batch_nheads_mblocks * num_splits) / static_cast<float>(num_sms);
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
#ifdef ENABLE_FFPA_PERSIST_KV_G2S
#if defined(BUILD_FFPA_ATTN_MMA_L20)
  constexpr int kMmaTileSeqLenQP = (kHeadDim <= kMaxDForSmallBlockTile) ? 4 : 8;
#else
  constexpr int kMmaTileSeqLenQP = (kHeadDim <= kMaxDForSmallBlockTile) ? 8 : 8;
#endif
#else  // if undef ENABLE_FFPA_PERSIST_KV_G2S
  // O(1) SRAM complexity, may always use large tile for
  // ffpa large d kernel. TODO: tune block size for L20/4090/3080 etc.
#if defined(BUILD_FFPA_ATTN_MMA_L20)
  constexpr int kMmaTileSeqLenQP = (kHeadDim <= kMaxDForSmallBlockTile) ? 4 : 8;
#else
  constexpr int kMmaTileSeqLenQP = (kHeadDim <= kMaxDForSmallBlockTile) ? 8 : 8;
#endif
#endif
  return kMmaTileSeqLenQP;
}

template <const int kHeadDim>
static constexpr int getConfigWarpTileSeqLenK() {
#ifdef ENABLE_FFPA_PERSIST_KV_G2S
#if defined(BUILD_FFPA_ATTN_MMA_L20)
  constexpr int kValTileSeqLenK = (kHeadDim <= kMaxDForSmallBlockTile) ? 8 : 16;
#else
  constexpr int kValTileSeqLenK = (kHeadDim <= kMaxDForSmallBlockTile) ? 16 : 16;
#endif
#else  // if undef ENABLE_FFPA_PERSIST_KV_G2S
#if defined(BUILD_FFPA_ATTN_MMA_L20)
  constexpr int kValTileSeqLenK = (kHeadDim <= kMaxDForSmallBlockTile) ? 8 : 16;
#else
  constexpr int kValTileSeqLenK = (kHeadDim <= kMaxDForSmallBlockTile) ? 16 : 16;
#endif
#endif
  return kValTileSeqLenK;
}

template <const int kHeadDim>
static constexpr int getConfigWarpTileHeadDimV() {
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
#if defined(ENABLE_FFPA_PERSIST_KV_G2S)
  constexpr int kPrefetchQKV = 1;  // kStageQKV is unused
#else
  constexpr int kPrefetchQKV = (kStageQKV > 1) ? 1 : 0;
#endif
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

static constexpr int getConfigPersistVs2r() {
#ifdef ENABLE_FFPA_PERSIST_V_S2R
  constexpr int kPersistVs2r = 1;
#else
  constexpr int kPersistVs2r = 0;
#endif
  return kPersistVs2r;
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

template <const int Br, const int Bc, const int kMmaAtomM, const int kMmaAtomN, const int kMmaAtomK,
          const int kHeadDim, const int kShareSmemQKV, const int kPersistQg2s,
          const int kPersistQs2r, const int kStageQK, const int kStagePV, const int kPadQ,
          const int kPadK, const int kPadV>
static constexpr int getConfigQKVSmemMaxSize() {
#ifdef ENABLE_FFPA_PERSIST_KV_G2S
  if constexpr (kHeadDim <= kMaxDForSmallDKernel) {  // e.g > 128 will use large d kernel
    // Calculate SRAM size needed per block, Q,K,V smem size, V shared the QK smem.
    constexpr int Q_smem_size = ((kHeadDim / kMmaAtomK) * (Br * (kMmaAtomK + kPadQ)));
    constexpr int K_smem_size = ((kHeadDim / kMmaAtomK) * (Bc * (kMmaAtomK + kPadK)));
    constexpr int V_smem_size = ((kHeadDim / (kMmaAtomN * 2)) * (Bc * (kMmaAtomN * 2 + kPadV)));
    constexpr int kQSmemMaxSize = Q_smem_size * 2;
    constexpr int kKSmemMaxSize = K_smem_size * 2;
    constexpr int kVSmemMaxSize = V_smem_size * 2;
    constexpr int kQKSmemMaxSize = (kQSmemMaxSize > kKSmemMaxSize ? kQSmemMaxSize : kKSmemMaxSize);
    constexpr int kQKVSmemMaxSize =
        ((kShareSmemQKV && kPersistQs2r) ? (kQKSmemMaxSize + kVSmemMaxSize)
                                         :  // QK shared the same smem
             (kQSmemMaxSize + kKSmemMaxSize + kVSmemMaxSize));
    return kQKVSmemMaxSize;
  } else {
    // Calculate SRAM size needed per block, Q,K,V smem size, V shared the QK smem.
    constexpr int Q_smem_size =
        ((kPersistQg2s ? (kHeadDim / kMmaAtomK) : kStageQK) * (Br * (kMmaAtomK + kPadQ))) * 2;
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
#else
  // Calculate SRAM size needed per block, Q,K,V smem size, V shared the QK smem.
  constexpr int Q_smem_size =
      ((kPersistQg2s ? (kHeadDim / kMmaAtomK) : kStageQK) * (Br * (kMmaAtomK + kPadQ))) * 2;
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
#endif
}

// Host-side launcher that picks compile-time configuration (block tile,
// stages, prefetch / share-smem flags, pad vs swizzle, etc.) based on
// ``kHeadDim`` and build macros, then launches the correct FFPA kernel
// template (small-d FA-2 style vs large-d split-Q) on the caller's
// current CUDA stream. Validates Q/K/V/O shape invariants up-front via
// ``TORCH_CHECK`` (GQA/MQA head ratio, matching Nkv / D, and the
// ``causal => Nkv >= Nq`` rule).
//
// Template parameters:
//   kDataType            Activation dtype: ``__half`` or ``__nv_bfloat16``.
//   kHeadDim             Head dim D (32, 64, ..., 1024); selects the
//                        small-d vs large-d kernel and the block tile.
//   kMmaAccFloat32QK/PV  0 -> fp16 MMA accumulator, 1 -> fp32 accumulator.
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
void launch_ffpa_attn_fwd_template(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                                   torch::Tensor O, torch::Tensor softmax_lse, int causal,
                                   double softmax_scale, int tma) {
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
  constexpr int kValTileSeqLenK = getConfigWarpTileSeqLenK<kHeadDim>();
  constexpr int kValTileSeqLenP = 1;
  constexpr int kValTileHeadDimV = getConfigWarpTileHeadDimV<kHeadDim>();
  constexpr int Br = kMmaAtomM * kMmaTileSeqLenQ * kValTileSeqLenQ;
  constexpr int Bc = kMmaAtomN * kMmaTileSeqLenK * kValTileSeqLenK;
  static_assert(Br == Bc, "Br must be equal Bc to avoid illegal memory access.");
  constexpr int kNumThreads = WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK;
  constexpr int kOStorageAccFloat32 = getConfigOStorageAccFloat32<kHeadDim>();
  // Apply different multi stages policy for QK and V.
  // TODO: tune stages for Q@K and P@V.
  constexpr int kStageQK = kStage;  // <= 4
  constexpr int kStagePV = kStage;  // <= 4
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
      getConfigQKVSmemMaxSize<Br, Bc, kMmaAtomM, kMmaAtomN, kMmaAtomK, kHeadDim, kShareSmemQKV,
                              kPersistQg2s, kPersistQs2r, kStageQK, kStagePV, kPadQ, kPadK,
                              kPadV>();
  TORCH_CHECK(K.size(0) == Q.size(0) && V.size(0) == Q.size(0),
              "ffpa_attn: Q/K/V must share the same batch size");
  TORCH_CHECK(K.size(1) == V.size(1), "ffpa_attn: K and V must share the same num_heads (Nh_kv)");
  TORCH_CHECK(Q.size(1) % K.size(1) == 0,
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
  const int Nh = Q.size(1);     // Q head count (Nh_q); used for grid fan-out.
  const int Nh_kv = K.size(1);  // K/V head count; Nh % Nh_kv == 0 asserted above.
  // Cross-attention: Q seqlen (Nq) may differ from KV seqlen (Nkv).
  const int Nq = Q.size(2);
  const int Nkv = K.size(2);
  // Seqlen (Nq, Nkv) no longer has to be a multiple of max(Br, Bc): the
  // kernel handles the tail tile via cp.async zero-fill, softmax -inf
  // masking and a per-row store predicate. div_ceil(Nkv, Bc) below still
  // yields the right Tc for partial last KV tiles.

  const dim3 block = getConfigBlock<kNumThreads>();  // 4/8 warps per block
  // grid is driven by Q row tiles; KV tile count Tc is driven by Nkv.
  const dim3 grid = getConfigGrid<Br>(Nb, Nh, Nq);
  const int Tc = utils::div_ceil(Nkv, Bc);  // Tc K_tile[Bc,d]
  const float scale = static_cast<float>(softmax_scale);
  float* softmax_lse_ptr = softmax_lse.data_ptr<float>();

  // Launch on the caller's current CUDA stream so the kernel participates
  // correctly in multi-stream pipelines. Without this the kernel would
  // default to stream 0 and race against user-side non-default streams.
  const c10::cuda::OptionalCUDAGuard device_guard(Q.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  if constexpr (kHeadDim > kMaxDForSmallDKernel) {
    const int num_sms_x2 = max(1, at::cuda::getCurrentDeviceProperties()->multiProcessorCount * 2);
    const int num_splits = select_decode_num_splits(Nb * Nh * utils::div_ceil(Nq, 16), num_sms_x2,
                                                    Tc, 128, min(Nq, 16));
    if (Nq == 1 && num_splits > 1) {
      const int split_size = utils::div_ceil(Tc, num_splits) * Bc;
      auto scratch_options = torch::TensorOptions().dtype(torch::kFloat32).device(Q.device());
      auto partial_out = torch::empty({Nb, Nh, num_splits, Nq, kHeadDim}, scratch_options);
      auto chunk_lse = torch::empty({Nb, Nh, num_splits, Nq}, scratch_options);
      const dim3 decode_stage1_grid = dim3(num_splits, Nb * Nh, 1);
      const dim3 decode_stage2_grid = dim3(Nq, Nb * Nh, 1);
      const int decode_threads = ((kHeadDim / 8) + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;
      const dim3 decode_block = dim3(decode_threads, 1, 1);
      // Pure gemv implementation for Nq=1 case, do the reduction in stage 2.
      auto decode_stage1_kernel =
          (ffpa_attn_splitkv_decode_stage1_template<kDataType, kHeadDim, true>);
      decode_stage1_kernel<<<decode_stage1_grid, decode_block, 0, stream>>>(
          reinterpret_cast<kDataType*>(Q.data_ptr()), reinterpret_cast<kDataType*>(K.data_ptr()),
          reinterpret_cast<kDataType*>(V.data_ptr()), partial_out.data_ptr<float>(),
          chunk_lse.data_ptr<float>(), Nq, Nkv, Nh, Nh_kv, scale, num_splits, split_size, causal);

      auto decode_stage2_kernel = (ffpa_attn_splitkv_decode_stage2_template<kDataType, kHeadDim>);
      decode_stage2_kernel<<<decode_stage2_grid, decode_block, 0, stream>>>(
          partial_out.data_ptr<float>(), chunk_lse.data_ptr<float>(),
          reinterpret_cast<kDataType*>(O.data_ptr()), softmax_lse_ptr, Nq, Nh, num_splits);
      return;
    }
  }

  const int smem_size_base = kQKVSmemMaxSize;

#define LAUNCH_TEMPLATE_FUNC_BASE(TEMPLATE_FUNC)                                              \
  cudaFuncSetAttribute(TEMPLATE_FUNC, cudaFuncAttributeMaxDynamicSharedMemorySize,            \
                       smem_size_base);                                                       \
  TEMPLATE_FUNC<<<grid, block, smem_size_base, stream>>>(                                     \
      reinterpret_cast<kDataType*>(Q.data_ptr()), reinterpret_cast<kDataType*>(K.data_ptr()), \
      reinterpret_cast<kDataType*>(V.data_ptr()), reinterpret_cast<kDataType*>(O.data_ptr()), \
      softmax_lse_ptr, Nq, Nkv, Nh, Nh_kv, scale, Tc, causal);

  constexpr int kEffShareSmemQKV_LargeD = (kPersistQg2s) ? 0 : kShareSmemQKV;
  constexpr int kEffPersistQs2r_LargeD = (kPersistQg2s || kHeadDim > 256) ? 0 : kPersistQs2r;

#ifdef ENABLE_FFPA_PERSIST_KV_G2S
  if constexpr (kHeadDim <= kMaxDForSmallDKernel) {       // e.g > 128 will use large d kernel
    constexpr int kPersistVs2r = getConfigPersistVs2r();  // only for d < 256

    auto ffpa_mma_small_d_kernel_func =
        (ffpa_attn_persistent_d_fwd_template < kDataType, kHeadDim, kMmaAtomM, kMmaAtomN, kMmaAtomK,
         kMmaTileSeqLenQ, kMmaTileSeqLenK, kMmaTileSeqLenP, kMmaTileHeadDimV, kValTileSeqLenQ,
         kValTileSeqLenK, kValTileSeqLenP, kValTileHeadDimV, kMmaAccFloat32QK, kMmaAccFloat32PV,
         kOStorageAccFloat32, kPrefetchQK, kPrefetchPV, kShareSmemQKV, kPersistQs2r, kPersistVs2r,
         // Force disable KV registers ping pong buffers
         // while V s2r is enabled.
         (kPersistVs2r) ? 0 : kRegPipeKV, 1, /*kStageQK unused*/
         1,                                  /*kStagePV unused*/
         kPadQ, kPadK, kPadV >);
    // Small-d kernel never takes the TMA path.
    LAUNCH_TEMPLATE_FUNC_BASE(ffpa_mma_small_d_kernel_func);
  } else {  // large headdim > kMaxDForSmallDKernel (e.g 128)
    auto ffpa_mma_large_d_kernel_func =
        (ffpa_attn_split_d_fwd_template<
            kDataType, kHeadDim, kMmaAtomM, kMmaAtomN, kMmaAtomK, kMmaTileSeqLenQ, kMmaTileSeqLenK,
            kMmaTileSeqLenP, kMmaTileHeadDimV, kValTileSeqLenQ, kValTileSeqLenK, kValTileSeqLenP,
            kValTileHeadDimV, kMmaAccFloat32QK, kMmaAccFloat32PV, kOStorageAccFloat32, kPrefetchQK,
            kPrefetchPV, kEffShareSmemQKV_LargeD, kEffPersistQs2r_LargeD, kPersistQg2s, kRegPipeKV,
            kStageQK, kStagePV, kPadQ, kPadK, kPadV>);
    LAUNCH_TEMPLATE_FUNC_BASE(ffpa_mma_large_d_kernel_func);
  }
#else
  auto ffpa_mma_large_d_kernel_func =
      (ffpa_attn_split_d_fwd_template<
          kDataType, kHeadDim, kMmaAtomM, kMmaAtomN, kMmaAtomK, kMmaTileSeqLenQ, kMmaTileSeqLenK,
          kMmaTileSeqLenP, kMmaTileHeadDimV, kValTileSeqLenQ, kValTileSeqLenK, kValTileSeqLenP,
          kValTileHeadDimV, kMmaAccFloat32QK, kMmaAccFloat32PV, kOStorageAccFloat32, kPrefetchQK,
          kPrefetchPV, kEffShareSmemQKV_LargeD, kEffPersistQs2r_LargeD, kPersistQg2s, kRegPipeKV,
          kStageQK, kStagePV, kPadQ, kPadK, kPadV>);
  LAUNCH_TEMPLATE_FUNC_BASE(ffpa_mma_large_d_kernel_func);
#endif

#undef LAUNCH_TEMPLATE_FUNC_BASE
}

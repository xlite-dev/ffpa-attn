#include "ffpa_attn_templates_L1.cuh"
using namespace ffpa;                                            

static constexpr int kMaxDForSmallDKernel   = 64;
static constexpr int kMaxDForOStoreFloat32  = 64;
static constexpr int kMaxDForSmallBlockTile = 256;

template<const int kHeadDim>
static constexpr int getConfigMmaTileSeqLenQP() {
#ifdef ENABLE_FFPA_PERSIST_KV_G2S 
#if defined(BUILD_FFPA_ATTN_MMA_L20)
  constexpr int kMmaTileSeqLenQP  = (
    kHeadDim <= kMaxDForSmallBlockTile) ? 4: 8;
#else
  constexpr int kMmaTileSeqLenQP  = (
    kHeadDim <= kMaxDForSmallBlockTile) ? 8: 8;
#endif
#else // if undef ENABLE_FFPA_PERSIST_KV_G2S
  // O(1) SRAM complexity, may always use large tile for 
  // ffpa large d kernel. TODO: tune block size for L20/4090/3080 etc. 
#if defined(BUILD_FFPA_ATTN_MMA_L20)
  constexpr int kMmaTileSeqLenQP  = (
    kHeadDim <= kMaxDForSmallBlockTile) ? 4: 8;
#else
  constexpr int kMmaTileSeqLenQP  = (
    kHeadDim <= kMaxDForSmallBlockTile) ? 8: 8;
#endif
#endif
  return kMmaTileSeqLenQP;
}

template<const int kHeadDim>
static constexpr int getConfigWarpTileSeqLenK() {
#ifdef ENABLE_FFPA_PERSIST_KV_G2S 
#if defined(BUILD_FFPA_ATTN_MMA_L20)
  constexpr int kWarpTileSeqLenK  = (
    kHeadDim <= kMaxDForSmallBlockTile) ? 8: 16;
#else
  constexpr int kWarpTileSeqLenK  = (
    kHeadDim <= kMaxDForSmallBlockTile) ? 16: 16;
#endif
#else // if undef ENABLE_FFPA_PERSIST_KV_G2S
#if defined(BUILD_FFPA_ATTN_MMA_L20)
  constexpr int kWarpTileSeqLenK  = (
    kHeadDim <= kMaxDForSmallBlockTile) ? 8: 16;
#else
  constexpr int kWarpTileSeqLenK  = (
    kHeadDim <= kMaxDForSmallBlockTile) ? 16: 16;
#endif
#endif
  return kWarpTileSeqLenK;
}

template<const int kHeadDim>
static constexpr int getConfigWarpTileHeadDimV() {
  constexpr int kMmaAtomN = 8;
  constexpr int kMmaTileHeadDimV = 1;
  constexpr int kWarpTileHeadDimV = (
    kHeadDim / (kMmaAtomN * kMmaTileHeadDimV));
  return kWarpTileHeadDimV;
}

static constexpr int getConfigShareSmemQKV() {
#if defined(ENABLE_FFPA_QKV_SMEM_SHARE)
  constexpr int kShareSmemQKV = 1;
#else
  constexpr int kShareSmemQKV = 0;
#endif 
  return kShareSmemQKV;
}

template<const int kHeadDim>
static constexpr int getConfigOStorageAccFloat32() {
  // 0/1, The precision of the O storage buffer can differ from 
  // that of the MMA, supporting either FP32 or Half precision.
  // FP16 can provide precision to approximately 3-4 decimal places.
  // Thus, if the error does not exceed 1e-3, using FP16 storage is 
  // sufficient for most applications.
  return ((kHeadDim <= kMaxDForOStoreFloat32)) ? 1 : 0;
}

template<const int kStageQKV>
static constexpr int getConfigPrefetchQKV() {
  // Prefetch QKV at the appropriate time point. 
#if defined(ENABLE_FFPA_PREFETCH_QKV)
#if defined(ENABLE_FFPA_PERSIST_KV_G2S)
  constexpr int kPrefetchQKV = 1; // kStageQKV is unused
#else 
  constexpr int kPrefetchQKV = (kStageQKV > 1) ? 1 : 0; 
#endif
#else 
  constexpr int kPrefetchQKV = 0;
#endif
  return kPrefetchQKV;
}

template<const int kStageQK, const int kHeadDim>
static constexpr int getConfigPersistQg2s() {
  // Persist load Q g2s for headdim < 512, more SRAM, but still
  // keep register usage.
#if defined(ENABLE_FFPA_PERSIST_Q_G2S)
  constexpr int kPersistQg2s = (kHeadDim < 256) ? 1 : (
    (kHeadDim <= 320) ? ((kStageQK < 3) ? 1 : 0) : 0 
  );
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

template<const int kNumThreads>
static inline dim3 getConfigBlock() {
  dim3 block(kNumThreads);
  return block;
}

template<const int Br>
static inline dim3 getConfigGrid(
  const int B, const int H, const int N) {
  // Tr(=N/Br), batch_size x num_heads
  // try grid(N/Br, B * H) or grid(N/Br, H, B)
#ifdef ENABLE_FFPA_LAUNCH_GRID_DNHB
  dim3 grid(utils::div_ceil(N, Br), H, B); 
#else
  dim3 grid(utils::div_ceil(N, Br), B * H); 
#endif
  return grid;
}

template<
  const int Br, 
  const int Bc, 
  const int kMmaAtomM, 
  const int kMmaAtomN, 
  const int kMmaAtomK, 
  const int kHeadDim, 
  const int kShareSmemQKV, 
  const int kPersistQg2s, 
  const int kPersistQs2r, 
  const int kStageQK,
  const int kStagePV,
  const int kPadQ, 
  const int kPadK, 
  const int kPadV
>
static constexpr int getConfigQKVSmemMaxSize() {
#ifdef ENABLE_FFPA_PERSIST_KV_G2S
  if constexpr (kHeadDim <= kMaxDForSmallDKernel) { // e.g > 128 will use large d kernel
    // Calculate SRAM size needed per block, Q,K,V smem size, V shared the QK smem.
    constexpr int Q_smem_size = (
      (kHeadDim / kMmaAtomK) * (Br * (kMmaAtomK + kPadQ)));
    constexpr int K_smem_size = (
      (kHeadDim / kMmaAtomK) * (Bc * (kMmaAtomK + kPadK)));
    constexpr int V_smem_size = (
      (kHeadDim / (kMmaAtomN * 2)) * (Bc * (kMmaAtomN * 2 + kPadV)));
    constexpr int kQSmemMaxSize = Q_smem_size * 2;
    constexpr int kKSmemMaxSize = K_smem_size * 2;
    constexpr int kVSmemMaxSize = V_smem_size * 2;
    constexpr int kQKSmemMaxSize = (
      kQSmemMaxSize > kKSmemMaxSize ? kQSmemMaxSize : kKSmemMaxSize);
    constexpr int kQKVSmemMaxSize = (
      (kShareSmemQKV && kPersistQs2r) ? 
      (kQKSmemMaxSize + kVSmemMaxSize) : // QK shared the same smem
      (kQSmemMaxSize + kKSmemMaxSize + kVSmemMaxSize)
    );
    return kQKVSmemMaxSize;
  } else {
    // Calculate SRAM size needed per block, Q,K,V smem size, V shared the QK smem.
    constexpr int Q_smem_size = ((kPersistQg2s ? (kHeadDim / kMmaAtomK) : kStageQK) * 
                                 (Br * (kMmaAtomK + kPadQ))) * 2;
    constexpr int K_smem_size = ((kStageQK) * (Bc * (kMmaAtomK + kPadK))) * 2;
    constexpr int V_smem_size = (kStagePV * (Bc * (kMmaAtomN * 2 + kPadV))) * 2;
    constexpr int kQKSmemMaxSize = (Q_smem_size + K_smem_size);
    constexpr int kVSmemMaxSize  = V_smem_size;
    // try to let V reuse all Q+K smem after Q@K^T, reduce smem usage.
    constexpr int kQKVSmemMaxSize = (
      (kShareSmemQKV && (!kPersistQg2s)) ? 
      ((kQKSmemMaxSize > kVSmemMaxSize) ? kQKSmemMaxSize: kVSmemMaxSize) : 
      (kQKSmemMaxSize + kVSmemMaxSize)
    );
    // NOTE: R_D registers usage, s=2, d=64, 16 regs; d=128, 32 regs; 
    // d=256, 64 regs; d=512, 128 regs; d=1024, 256 regs;
    return kQKVSmemMaxSize;
  }
#else  
  // Calculate SRAM size needed per block, Q,K,V smem size, V shared the QK smem.
  constexpr int Q_smem_size = ((kPersistQg2s ? (kHeadDim / kMmaAtomK) : kStageQK) * 
                               (Br * (kMmaAtomK + kPadQ))) * 2;
  constexpr int K_smem_size = ((kStageQK) * (Bc * (kMmaAtomK + kPadK))) * 2;
  constexpr int V_smem_size = (kStagePV * (Bc * (kMmaAtomN * 2 + kPadV))) * 2;
  constexpr int kQKSmemMaxSize = (Q_smem_size + K_smem_size);
  constexpr int kVSmemMaxSize  = V_smem_size;
  // try to let V reuse all Q+K smem after Q@K^T, reduce smem usage.
  constexpr int kQKVSmemMaxSize = (
    (kShareSmemQKV && (!kPersistQg2s)) ? 
    ((kQKSmemMaxSize > kVSmemMaxSize) ? kQKSmemMaxSize: kVSmemMaxSize) : 
    (kQKSmemMaxSize + kVSmemMaxSize)
  );
  // NOTE: R_D registers usage, s=2, d=64, 16 regs; d=128, 32 regs; 
  // d=256, 64 regs; d=512, 128 regs; d=1024, 256 regs;
  return kQKVSmemMaxSize;
#endif
}

template<
  const int kHeadDim,          // Headdim, 32~1024   
  const int kMmaAccFloat32QK,  // 0/1, Q@K^T, 0 MMA Acc with fp16, 1 MMA Acc with fp32.
  const int kMmaAccFloat32PV,  // 0/1, P@V, 0 MMA Acc with fp16, 1 MMA Acc with fp32.
  const int kStage
>
void launch_ffpa_mma_L1_template(torch::Tensor Q, 
                                 torch::Tensor K, 
                                 torch::Tensor V, 
                                 torch::Tensor O) {
  // Q,K,V,O with [B, H, N, D] layout, B=batch, H=head, N=seqlen, D=dim
  // TODO: support BNHD layout, Q,K,V,O with [B, N, H, D] layout.
  constexpr int kMmaAtomM = 16;
  constexpr int kMmaAtomN = 8;
  constexpr int kMmaAtomK = 16;
  // Split-Q(FA-2) Algo, Tile MMA across Q and keep KV access for all MMAs.
  constexpr int kMmaTileSeqLenQ   = getConfigMmaTileSeqLenQP<kHeadDim>();
  constexpr int kMmaTileSeqLenK   = 1;
  constexpr int kMmaTileSeqLenP   = getConfigMmaTileSeqLenQP<kHeadDim>();
  constexpr int kMmaTileHeadDimV  = 1;
  constexpr int kWarpTileSeqLenQ  = 1;
  constexpr int kWarpTileSeqLenK  = getConfigWarpTileSeqLenK<kHeadDim>();
  constexpr int kWarpTileSeqLenP  = 1;
  constexpr int kWarpTileHeadDimV = getConfigWarpTileHeadDimV<kHeadDim>();
  constexpr int Br = kMmaAtomM * kMmaTileSeqLenQ * kWarpTileSeqLenQ;
  constexpr int Bc = kMmaAtomN * kMmaTileSeqLenK * kWarpTileSeqLenK;
  static_assert(Br == Bc, "Br must be equal Bc to avoid illegal memory access.");
  constexpr int kNumThreads = WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK;
  constexpr int kOStorageAccFloat32 = getConfigOStorageAccFloat32<kHeadDim>();
  // Apply different multi stages policy for QK and V.
  // TODO: tune stages for Q@K and P@V.
  constexpr int kStageQK = kStage; // <= 4
  constexpr int kStagePV = kStage; // <= 4
  // Prefetch QKV, Persist Q g2s/s2r, Shared QKV smem.
  constexpr int kShareSmemQKV = getConfigShareSmemQKV();
  constexpr int kPrefetchQK   = getConfigPrefetchQKV<kStageQK>();
  constexpr int kPrefetchPV   = getConfigPrefetchQKV<kStagePV>(); 
  constexpr int kPersistQs2r  = getConfigPersistQs2r();
  constexpr int kPersistQg2s  = getConfigPersistQg2s<kStageQK, kHeadDim>();
  constexpr int kRegPipeKV    = getConfigRegistersPipeKV();
  // QKV smem swizzle, 0 for smem swizzle, !0 for smem padding.
  constexpr int kPadQ = getConfigPadQ();
  constexpr int kPadK = getConfigPadK();
  constexpr int kPadV = getConfigPadV();
  // Calculate SRAM size needed for per block.
  constexpr int kQKVSmemMaxSize = getConfigQKVSmemMaxSize<
    Br, Bc, kMmaAtomM, kMmaAtomN, kMmaAtomK, kHeadDim, kShareSmemQKV, 
    kPersistQg2s, kPersistQs2r, kStageQK, kStagePV, kPadQ, kPadK,
    kPadV
  >();

  const int QKV_batch  = Q.size(0); 
  const int QKV_head   = Q.size(1);
  const int QKV_seqlen = Q.size(2); // QKV_seqlen
  assert(QKV_seqlen % max(Br, Bc) == 0); // multiple of max(Br, Bc)
  
  const dim3 block = getConfigBlock<kNumThreads>(); // 4/8 warps per block
  const dim3 grid  = getConfigGrid<Br>(QKV_batch, QKV_head, QKV_seqlen);
  // Precompute softmax scale and Tc
  const int Tc = utils::div_ceil(QKV_seqlen, Bc); // Tc K_tile[Bc,d]  
  const float scale = 1.0f / sqrt((float) kHeadDim);

#define LAUNCH_TEMPLATE_FUNC(TEMPLATE_FUNC)                 \
cudaFuncSetAttribute(                                       \
  TEMPLATE_FUNC,                                            \
  cudaFuncAttributeMaxDynamicSharedMemorySize,              \
  kQKVSmemMaxSize                                           \
);                                                          \
TEMPLATE_FUNC<<<grid, block, kQKVSmemMaxSize>>>(            \
  reinterpret_cast<half*>(Q.data_ptr()),                    \
  reinterpret_cast<half*>(K.data_ptr()),                    \
  reinterpret_cast<half*>(V.data_ptr()),                    \
  reinterpret_cast<half*>(O.data_ptr()),                    \
  QKV_seqlen,                                               \
  QKV_head,                                                 \
  scale,                                                    \
  Tc                                                        \
);

#ifdef ENABLE_FFPA_PERSIST_KV_G2S
  if constexpr (kHeadDim <= kMaxDForSmallDKernel) { // e.g > 128 will use large d kernel
    constexpr int kPersistVs2r = getConfigPersistVs2r(); // only for d < 256

    auto ffpa_mma_L1_small_d_kernel_func = (
      ffpa_mma_stages_split_q_L1_small_d_template<
        kHeadDim, 
        kMmaAtomM, 
        kMmaAtomN, 
        kMmaAtomK, 
        kMmaTileSeqLenQ,  
        kMmaTileSeqLenK,
        kMmaTileSeqLenP, 
        kMmaTileHeadDimV, 
        kWarpTileSeqLenQ, 
        kWarpTileSeqLenK, 
        kWarpTileSeqLenP, 
        kWarpTileHeadDimV, 
        kMmaAccFloat32QK,
        kMmaAccFloat32PV,
        kOStorageAccFloat32,
        kPrefetchQK,
        kPrefetchPV,
        kShareSmemQKV,
        kPersistQs2r,
        kPersistVs2r,
        // Force disable KV registers ping pong buffers
        // while V s2r is enabled.
        (kPersistVs2r) ? 0 : kRegPipeKV,
        1, /*kStageQK unused*/
        1, /*kStagePV unused*/
        kPadQ,
        kPadK,
        kPadV
      >
    );
    LAUNCH_TEMPLATE_FUNC(ffpa_mma_L1_small_d_kernel_func);
  } else { // large headdim > kMaxDForSmallDKernel (e.g 128)
    auto ffpa_mma_L1_large_d_kernel_func = (
      ffpa_mma_stages_split_q_L1_large_d_template<
        kHeadDim, 
        kMmaAtomM, 
        kMmaAtomN, 
        kMmaAtomK, 
        kMmaTileSeqLenQ,  
        kMmaTileSeqLenK,
        kMmaTileSeqLenP, 
        kMmaTileHeadDimV, 
        kWarpTileSeqLenQ, 
        kWarpTileSeqLenK, 
        kWarpTileSeqLenP, 
        kWarpTileHeadDimV, 
        kMmaAccFloat32QK,
        kMmaAccFloat32PV,
        kOStorageAccFloat32,
        kPrefetchQK,
        kPrefetchPV,
        (kPersistQg2s) ? 0 : kShareSmemQKV,
        // Force disable Q s2r for d >= 256, Q s2r for large d will 
        // need too many register, thus, introduce performance drops.
        (kPersistQg2s || kHeadDim > 256) ? 0 : kPersistQs2r,
        kPersistQg2s,
        kRegPipeKV,
        kStageQK, 
        kStagePV,
        kPadQ,
        kPadK,
        kPadV
      >
    );
    LAUNCH_TEMPLATE_FUNC(ffpa_mma_L1_large_d_kernel_func);
  }
#else  
  auto ffpa_mma_L1_large_d_kernel_func = (
    ffpa_mma_stages_split_q_L1_large_d_template<
      kHeadDim, 
      kMmaAtomM, 
      kMmaAtomN, 
      kMmaAtomK, 
      kMmaTileSeqLenQ,  
      kMmaTileSeqLenK,
      kMmaTileSeqLenP, 
      kMmaTileHeadDimV, 
      kWarpTileSeqLenQ, 
      kWarpTileSeqLenK, 
      kWarpTileSeqLenP, 
      kWarpTileHeadDimV, 
      kMmaAccFloat32QK,
      kMmaAccFloat32PV,
      kOStorageAccFloat32,
      kPrefetchQK,
      kPrefetchPV,
      (kPersistQg2s) ? 0 : kShareSmemQKV,
      // Force disable Q s2r for d >= 256, Q s2r for large d will 
      // need too many register, thus, introduce performance drops.
      (kPersistQg2s || kHeadDim > 256) ? 0 : kPersistQs2r, 
      kPersistQg2s,
      kRegPipeKV,
      kStageQK, 
      kStagePV,
      kPadQ,
      kPadK,
      kPadV
    >
  );
  LAUNCH_TEMPLATE_FUNC(ffpa_mma_L1_large_d_kernel_func);
#endif

#undef LAUNCH_TEMPLATE_FUNC
}

// dispatch headdim
#define LAUNCHER_L1(D, S)        \
  case D:                        \
    launch_ffpa_mma_L1_template< \
      (D),                       \
      kMmaAccFloat32QK,          \
      kMmaAccFloat32PV,          \
      (S)                        \
    >(Q, K, V, O);               \
    break;

#ifdef ENABLE_FFPA_DEBUG
  // minimal kernels for debug mode
#define DISPATCH_HEADDIM(LAUNCHER, S) \
  {                                   \
    switch (d)                        \
    {                                 \
      LAUNCHER(32,   (S));            \
      LAUNCHER(64,   (S));            \
      LAUNCHER(128,  (S));            \
      LAUNCHER(256,  (S));            \
      LAUNCHER(320,  (S));            \
      LAUNCHER(512,  (S));            \
      LAUNCHER(1024, (S));            \
    default:                          \
      throw std::runtime_error(       \
        "headdim not support!");      \
      break;                          \
    }                                 \
  }

#else
#ifdef ENABLE_FFPA_ALL_HEADDIM
  // multiple of 32
#define DISPATCH_HEADDIM(LAUNCHER, S) \
  {                                   \
    switch (d)                        \
    {                                 \
      LAUNCHER(32,   (S));            \
      LAUNCHER(64,   (S));            \
      LAUNCHER(96,   (S));            \
      LAUNCHER(128,  (S));            \
      LAUNCHER(160,  (S));            \
      LAUNCHER(192,  (S));            \
      LAUNCHER(224,  (S));            \
      LAUNCHER(256,  (S));            \
      LAUNCHER(288,  (S));            \
      LAUNCHER(320,  (S));            \
      LAUNCHER(352,  (S));            \
      LAUNCHER(384,  (S));            \
      LAUNCHER(416,  (S));            \
      LAUNCHER(448,  (S));            \
      LAUNCHER(480,  (S));            \
      LAUNCHER(512,  (S));            \
      LAUNCHER(544,  (S));            \
      LAUNCHER(576,  (S));            \
      LAUNCHER(608,  (S));            \
      LAUNCHER(640,  (S));            \
      LAUNCHER(672,  (S));            \
      LAUNCHER(704,  (S));            \
      LAUNCHER(736,  (S));            \
      LAUNCHER(768,  (S));            \
      LAUNCHER(800,  (S));            \
      LAUNCHER(832,  (S));            \
      LAUNCHER(864,  (S));            \
      LAUNCHER(896,  (S));            \
      LAUNCHER(928,  (S));            \
      LAUNCHER(960,  (S));            \
      LAUNCHER(992,  (S));            \
      LAUNCHER(1024, (S));            \
    default:                          \
      throw std::runtime_error(       \
        "headdim not support!");      \
      break;                          \
    }                                 \
  }
#else
  // multiple of 64
#define DISPATCH_HEADDIM(LAUNCHER, S) \
  {                                   \
    switch (d)                        \
    {                                 \
      LAUNCHER(256,  (S));            \
      LAUNCHER(320,  (S));            \
      LAUNCHER(384,  (S));            \
      LAUNCHER(448,  (S));            \
      LAUNCHER(512,  (S));            \
      LAUNCHER(576,  (S));            \
      LAUNCHER(640,  (S));            \
      LAUNCHER(704,  (S));            \
      LAUNCHER(768,  (S));            \
      LAUNCHER(832,  (S));            \
      LAUNCHER(896,  (S));            \
      LAUNCHER(960,  (S));            \
      LAUNCHER(1024, (S));            \
    default:                          \
      throw std::runtime_error(       \
        "headdim not support!");      \
      break;                          \
    }                                 \
  }
#endif

#endif

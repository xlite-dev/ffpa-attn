#include "ffpa_attn_templates_L1.cuh"
using namespace ffpa;                                            


template<
  const int kHeadDim,              // Headdim, 32~1024   
  const int kMmaAccFloat32QK,      // 0/1, Q@K^T, 0 MMA Acc with fp16, 1 MMA Acc with fp32.
  const int kMmaAccFloat32PV,      // 0/1, P@V, 0 MMA Acc with fp16, 1 MMA Acc with fp32.
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
#if defined(ENABLE_FFPA_QKV_SMEM_SHARE)
  constexpr int kShareSmemQKV = 1;
#else
  constexpr int kShareSmemQKV = 0;
#endif 

#ifdef ENABLE_FFPA_PERSIST_KV_G2S 
  // Need more SRAM, use small tile 64x64 for large headdim
  // and large tile for small headdim. headdim > 128 will 
  // use ffpa-attn, not flash-attn. TODO: tune block size 
  // for L20/4090/3080 etc. Prefer small block size for 
  // ffpa small d kenel on NVIDIA L20 device (64x64) and
  // large block size on NVIDIA 4090 device (128x128)
#if defined(BUILD_FFPA_ATTN_MMA_L20)
  // Enable QKV g2s & s2r with block 64x64 for d <= 128
  // (small d kernel) will get best performance.
  constexpr int kMmaTileSeqLenQ  = (kHeadDim <= 256) ? 4: 8;
  constexpr int kMmaTileSeqLenK  = 1;
  constexpr int kMmaTileSeqLenP  = (kHeadDim <= 256) ? 4: 8;
  constexpr int kMmaTileHeadDimV = 1;
  constexpr int kWarpTileSeqLenQ = 1;
  constexpr int kWarpTileSeqLenK = (kHeadDim <= 256) ? 8: 16;
#else
  // NOTE: On 4090, enable Q g2s for d <= 320 (large d kernel) 
  // will get best performance.
  constexpr int kMmaTileSeqLenQ  = (kHeadDim <= 256) ? 8: 8;
  constexpr int kMmaTileSeqLenK  = 1;
  constexpr int kMmaTileSeqLenP  = (kHeadDim <= 256) ? 8: 8;
  constexpr int kMmaTileHeadDimV = 1;
  constexpr int kWarpTileSeqLenQ = 1;
  constexpr int kWarpTileSeqLenK = (kHeadDim <= 256) ? 16: 16;
#endif

#else // if undef ENABLE_FFPA_PERSIST_KV_G2S
  // O(1) SRAM complexity, may always use large tile for 
  // ffpa large d kernel. TODO: tune block size for L20/4090/3080 etc. 
#if defined(BUILD_FFPA_ATTN_MMA_L20)
  // Enable QKV g2s & s2r with block 64x64 for d <= 128
  // (small d kernel) will get best performance.
  constexpr int kMmaTileSeqLenQ  = (kHeadDim <= 256) ? 4: 8;
  constexpr int kMmaTileSeqLenK  = 1;
  constexpr int kMmaTileSeqLenP  = (kHeadDim <= 256) ? 4: 8;
  constexpr int kMmaTileHeadDimV = 1;
  constexpr int kWarpTileSeqLenQ = 1;
  constexpr int kWarpTileSeqLenK = (kHeadDim <= 256) ? 8: 16;
#else
  // NOTE: On 4090, enable Q g2s for d <= 320 (large d kernel) 
  // will get best performance.
  constexpr int kMmaTileSeqLenQ  = (kHeadDim <= 256) ? 8: 8;
  constexpr int kMmaTileSeqLenK  = 1;
  constexpr int kMmaTileSeqLenP  = (kHeadDim <= 256) ? 8: 8;
  constexpr int kMmaTileHeadDimV = 1;
  constexpr int kWarpTileSeqLenQ = 1;
  constexpr int kWarpTileSeqLenK = (kHeadDim <= 256) ? 16: 16;
#endif

#endif
  constexpr int kWarpTileSeqLenP = 1;
  constexpr int kWarpTileHeadDimV = (kHeadDim / (kMmaAtomN * kMmaTileHeadDimV));
  constexpr int Br = kMmaAtomM * kMmaTileSeqLenQ * kWarpTileSeqLenQ;
  constexpr int Bc = kMmaAtomN * kMmaTileSeqLenK * kWarpTileSeqLenK;
  static_assert(Br == Bc, "Br must be equal Bc in order to avoid illegal memory access.");
  constexpr int kNumThreads = WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK;
  // Apply different multi stages policy for QK and V.
  constexpr int kStageQK = kStage; // <= 4
  constexpr int kStagePV = kStage; // <= 4
  // 0/1, The precision of the O storage buffer can differ from 
  // that of the MMA, supporting either FP32 or Half precision.
  // FP16 can provide precision to approximately 3-4 decimal places.
  // Thus, if the error does not exceed 1e-3, using FP16 storage is 
  // sufficient for most applications.
  constexpr int kOStorageAccFloat32 = ((kHeadDim < 128)) ? 1 : 0;
  // Persist load Q s2r for headdim < 512, more registers, 
  // but still keep O(1) SRAM.
#ifdef ENABLE_FFPA_PERSIST_Q_S2R
  constexpr int kPersistQs2r = 1;
#else
  constexpr int kPersistQs2r = 0;
#endif

  // Persist load Q g2s for headdim < 512, more SRAM, but still
  // keep register usage.
#if defined(ENABLE_FFPA_PERSIST_Q_G2S)
  constexpr int kPersistQg2s = (kHeadDim < 256) ? 1 : (
    (kHeadDim <= 320) ? ((kStageQK < 3) ? 1 : 0) : 0 
  );
#else
  constexpr int kPersistQg2s = 0;
#endif

  // Prefetch QKV at the appropriate time point. 
#if defined(ENABLE_FFPA_PREFETCH_QKV)
#if defined(ENABLE_FFPA_PERSIST_KV_G2S)
  constexpr int kPrefetchQK = 1; // kStageQK is unused
  constexpr int kPrefetchPV = 1; // kStagePV is unused
#else 
  constexpr int kPrefetchQK = (kStageQK > 1) ? 1 : 0; 
  constexpr int kPrefetchPV = (kStagePV > 1) ? 1 : 0; 
#endif
#else 
  constexpr int kPrefetchQK = 0;
  constexpr int kPrefetchPV = 0;
#endif

  // QKV smem swizzle, 0 for smem swizzle, !0 for smem padding.
#ifdef ENABLE_FFPA_SMEM_SWIZZLE_Q
  constexpr int kPadQ = 0;
#else 
  constexpr int kPadQ = 8;
#endif
#ifdef ENABLE_FFPA_SMEM_SWIZZLE_K
  constexpr int kPadK = 0; 
#else
  constexpr int kPadK = 8;
#endif
#ifdef ENABLE_FFPA_SMEM_SWIZZLE_V
  constexpr int kPadV = 0; 
#else 
  constexpr int kPadV = 8;
#endif

  const int QKV_batch  = Q.size(0); 
  const int QKV_head   = Q.size(1);
  const int QKV_seqlen = Q.size(2); // QKV_seqlen
  assert(QKV_seqlen % max(Br, Bc) == 0); // multiple of max(Br, Bc)
  
  dim3 block(kNumThreads); // 4/8 warps per block
  // Tr(=N/Br), batch_size x num_heads
  // try grid(N/Br, B * H) or grid(N/Br, H, B)
#ifdef ENBALE_FFPA_LAUNCH_GRID_DNHB
  dim3 grid(utils::div_ceil(QKV_seqlen, Br), QKV_head, QKV_batch); 
#else
  dim3 grid(utils::div_ceil(QKV_seqlen, Br), QKV_batch * QKV_head); 
#endif
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
  if constexpr (kHeadDim < 256) { // 256 will use large d kernel
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
#ifdef ENABLE_FFPA_PERSIST_V_S2R
    constexpr int kPersistVs2r = 1;
#else
    constexpr int kPersistVs2r = 0;
#endif

    auto ffpa_mma_L1_kernel_func = (
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
        1, /*kStageQK unused*/
        1, /*kStagePV unused*/
        kPadQ,
        kPadK,
        kPadV
      >
    );
    LAUNCH_TEMPLATE_FUNC(ffpa_mma_L1_kernel_func);
  } else { // large headdim >= 256
    // Calculate SRAM size needed per block, Q,K,V smem size, V shared the QK smem.
    constexpr int QK_smem_size = (
      (kPersistQg2s ? (kHeadDim / kMmaAtomK) : kStageQK) * // Q
      (Br * (kMmaAtomK + kPadQ)) + 
      (kStageQK) * (Bc * (kMmaAtomK + kPadK))  // K
    );
    // R_D registers, s=2, d=64, 16 regs; d=128, 32 regs; 
    // d=256, 64 regs; d=512, 128 regs; d=1024, 256 regs;
    constexpr int PV_smem_size = (kStagePV * (Bc * (kMmaAtomN * 2 + kPadV)));
    // try to let V reuse all Q+K smem after Q@K^T, reduce smem usage.
    constexpr int kQKVSmemMaxSize = (
      (kShareSmemQKV && (!kPersistQg2s)) ? 
      ((QK_smem_size > PV_smem_size ? QK_smem_size * 2 : PV_smem_size * 2)) : 
      ((QK_smem_size + PV_smem_size) * 2)
    );

    auto ffpa_mma_L1_kernel_func = (
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
        (kPersistQg2s || kHeadDim > 256) ? 0 : kPersistQs2r,
        kPersistQg2s,
        kStageQK, 
        kStagePV,
        kPadQ,
        kPadK,
        kPadV
      >
    );
    LAUNCH_TEMPLATE_FUNC(ffpa_mma_L1_kernel_func);
  }
#else  
  // Always use large d ffpa kernel for all headdims.
  // Calculate SRAM size needed per block, Q,K,V smem size, V shared the QK smem.
  constexpr int QK_smem_size = (
    (kPersistQg2s ? (kHeadDim / kMmaAtomK) : kStageQK) * // Q
    (Br * (kMmaAtomK + kPadQ)) + 
    (kStageQK) * (Bc * (kMmaAtomK + kPadK))  // K
  );
  // R_D registers, s=2, d=64, 16 regs; d=128, 32 regs; 
  // d=256, 64 regs; d=512, 128 regs; d=1024, 256 regs;
  constexpr int PV_smem_size = (kStagePV * (Bc * (kMmaAtomN * 2 + kPadV)));
  constexpr int kQKVSmemMaxSize = (
    (kShareSmemQKV && (!kPersistQg2s)) ? 
    ((QK_smem_size > PV_smem_size ? QK_smem_size * 2 : PV_smem_size * 2)) : 
    ((QK_smem_size + PV_smem_size) * 2)
  );

  auto ffpa_mma_L1_kernel_func = (
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
      (kPersistQg2s || kHeadDim > 256) ? 0 : kPersistQs2r,
      kPersistQg2s,
      kStageQK, 
      kStagePV,
      kPadQ,
      kPadK,
      kPadV
    >
  );
  LAUNCH_TEMPLATE_FUNC(ffpa_mma_L1_kernel_func);
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

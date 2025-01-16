#include "ffpa_attn_templates_L1.cuh"
using namespace ffpa;  


template<const int kHeadDim, const int kStage>
void launch_ffpa_mma_acc_f32_L1(torch::Tensor Q, 
                                torch::Tensor K, 
                                torch::Tensor V, 
                                torch::Tensor O) {
  // Q,K,V,O with [B, H, N, D] layout, B=batch, H=head, N=seqlen, D=dim
  // TODO: support BNHD layout, Q,K,V,O with [B, N, H, D] layout.
  // Now: fixed tile BrxBc=128x128 for d>= 128, 64x64 for d<128.
  constexpr int kMmaAtomM = 16;
  constexpr int kMmaAtomN = 8;
  constexpr int kMmaAtomK = 16;
  constexpr int kMmaTileSeqLenQ  = (kHeadDim < 128) ? 4 : 8;
  constexpr int kMmaTileSeqLenK  = 1;
  constexpr int kMmaTileSeqLenP  = (kHeadDim < 128) ? 4 : 8;
  constexpr int kMmaTileHeadDimV = 1;
  constexpr int kWarpTileSeqLenQ = 1;
  constexpr int kWarpTileSeqLenK = (kHeadDim < 128) ? 8 : 16;
  constexpr int kWarpTileSeqLenP = 1;
  constexpr int kWarpTileHeadDimV = (kHeadDim / (kMmaAtomN * kMmaTileHeadDimV));
  constexpr int Br = kMmaAtomM * kMmaTileSeqLenQ * kWarpTileSeqLenQ;
  constexpr int Bc = kMmaAtomN * kMmaTileSeqLenK * kWarpTileSeqLenK;
  constexpr int kNumThreads = WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK;
  // Q@K^T or P@V, 0 MMA Acc with fp16, 1 MMA Acc with fp32.
#ifdef ENABLE_FFPA_FORCE_QK_F16
  constexpr int kMmaAccFloat32QK = 0;
#else
  constexpr int kMmaAccFloat32QK = 1;
#endif
#ifdef ENABLE_FFPA_FORCE_PV_F16
  constexpr int kMmaAccFloat32PV = 0;
#else
  constexpr int kMmaAccFloat32PV = 1;
#endif
  // Apply different multi stages policy for QK and V.
  constexpr int kStageQK = kStage; // <= 4
  constexpr int kStagePV = kStage; // <= 4
  // 0/1, The precision of the O storage buffer can differ from 
  // that of the MMA, supporting either FP32 or Half precision.
  // FP16 can provide precision to approximately 3-4 decimal places.
  // Thus, if the error does not exceed 1e-3, using FP16 storage is 
  // sufficient for most applications.
  constexpr int kOStorageAccFloat32 = (kHeadDim < 256) ? 1 : 0;
  // Persist load Q s2r for headdim < 512, but still keep O(1) SRAM.
#ifdef ENABLE_FFPA_PERSIST_Q_S2R
  const int kPersistQs2r = 1;
#else
  const int kPersistQs2r = 0;
#endif
  // Prefetch QKV at the appropriate time point.  
#ifdef ENABLE_FFPA_PREFETCH_QKV
  constexpr int kPrefetchQK = (kStageQK > 1) ? 1 : 0; 
  constexpr int kPrefetchPV = (kStagePV > 1) ? 1 : 0; 
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

  // Calculate SRAM size needed per block, Q,K,V smem size, V shared the QK smem.
  constexpr int QK_smem_size = (kStageQK * (Br * (kMmaAtomK + kPadQ)) + 
                                kStageQK * (Bc * (kMmaAtomK + kPadK)));
  // R_D registers, s=2, d=64, 16 regs; d=128, 32 regs; 
  // d=256, 64 regs; d=512, 128 regs; d=1024, 256 regs;
  constexpr int PV_smem_size = (kStagePV * (Bc * (kMmaAtomN * 2 + kPadV))); 
#ifdef ENABLE_FFPA_QKV_SMEM_SHARE
  constexpr int kShareSmemQKV = 1;
  // try to let V reuse all Q+K smem after Q@K^T, reduce smem usage.
  constexpr int kQKVSmemMaxSize = (QK_smem_size > PV_smem_size ? 
                                   QK_smem_size * 2 : 
                                   PV_smem_size * 2);
#else
  constexpr int kShareSmemQKV = 0;
  constexpr int kQKVSmemMaxSize = (QK_smem_size + PV_smem_size) * 2;
#endif 

  const int QKV_batch  = Q.size(0); 
  const int QKV_head   = Q.size(1);
  const int QKV_seqlen = Q.size(2); // QKV_seqlen
  assert(QKV_seqlen % max(Br, Bc) == 0); // multiple of max(Br, Bc)
  
  // Tr(=N/Br), batch_size x num_heads
  dim3 grid(utils::div_ceil(QKV_seqlen, Br), QKV_batch * QKV_head); 
  dim3 block(kNumThreads); // 4/8 warps per block

  cudaFuncSetAttribute(
    ffpa_mma_stages_split_q_L1_template<
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
      kStageQK, 
      kStagePV,
      kPadQ,
      kPadK,
      kPadV
    >,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    kQKVSmemMaxSize
  );

  ffpa_mma_stages_split_q_L1_template<
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
    kStageQK, 
    kStagePV,
    kPadQ,
    kPadK,
    kPadV
  ><<<grid, block, kQKVSmemMaxSize>>>(
    reinterpret_cast<half*>(Q.data_ptr()),
    reinterpret_cast<half*>(K.data_ptr()),
    reinterpret_cast<half*>(V.data_ptr()),
    reinterpret_cast<half*>(O.data_ptr()),
    QKV_seqlen,
    QKV_head
  );
}

void ffpa_mma_acc_f32_L1(torch::Tensor Q, 
                         torch::Tensor K, 
                         torch::Tensor V, 
                         torch::Tensor O, 
                         int stages) {
  CHECK_TORCH_TENSOR_DTYPE(Q, torch::kHalf) // Q [B,H,N,D]
  CHECK_TORCH_TENSOR_DTYPE(K, torch::kHalf) // K [B,H,N,D]
  CHECK_TORCH_TENSOR_DTYPE(V, torch::kHalf) // V [B,H,N,D]
  CHECK_TORCH_TENSOR_DTYPE(O, torch::kHalf) // O [B,H,N,D]
  const int d = Q.size(3); // B, H, N, d
  
  // dispatch headdim
#define CASE_LAUNCH_KERNEL_F32_L1(D, S)      \
  case D:                                    \
    launch_ffpa_mma_acc_f32_L1<(D), (S)>(    \
      Q, K, V, O);                           \
    break;

#ifdef ENABLE_FFPA_DEBUG
  // minimal kernels for debug mode
#define DISPATCH_KERNEL_F32_L1_HEADDIM(S)    \
  {                                          \
    switch (d)                               \
    {                                        \
      CASE_LAUNCH_KERNEL_F32_L1(64,   (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(128,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(256,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(320,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(512,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(1024, (S));  \
    default:                                 \
      throw std::runtime_error(              \
        "headdim not support!");             \
      break;                                 \
    }                                        \
  }

#else 
#ifdef ENABLE_FFPA_ALL_HEADDIM
  // multiple of 32
#define DISPATCH_KERNEL_F32_L1_HEADDIM(S)    \
  {                                          \
    switch (d)                               \
    {                                        \
      CASE_LAUNCH_KERNEL_F32_L1(32,   (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(64,   (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(96,   (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(128,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(160,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(192,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(224,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(256,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(288,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(320,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(352,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(384,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(416,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(448,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(480,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(512,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(544,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(576,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(608,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(640,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(672,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(704,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(736,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(768,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(800,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(832,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(864,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(896,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(928,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(960,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(992,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(1024, (S));  \
    default:                                 \
      throw std::runtime_error(              \
        "headdim not support!");             \
      break;                                 \
    }                                        \
  }
#else
  // multiple of 64
#define DISPATCH_KERNEL_F32_L1_HEADDIM(S)    \
  {                                          \
    switch (d)                               \
    {                                        \
      CASE_LAUNCH_KERNEL_F32_L1(256,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(320,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(384,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(448,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(512,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(576,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(640,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(704,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(768,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(832,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(896,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(960,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(1024, (S));  \
    default:                                 \
      throw std::runtime_error(              \
        "headdim not support!");             \
      break;                                 \
    }                                        \
  }
#endif

#endif

#ifdef ENABLE_FFPA_ALL_STAGES
  // dispatch stages
  if (stages == 2) {
    DISPATCH_KERNEL_F32_L1_HEADDIM(2);
  } else if (stages == 3) {
    DISPATCH_KERNEL_F32_L1_HEADDIM(3);
  } else if (stages == 4) {
    DISPATCH_KERNEL_F32_L1_HEADDIM(4);
  } else {
    DISPATCH_KERNEL_F32_L1_HEADDIM(1);
  }
#else 
  // dispatch stages
  if (stages == 2) {
    DISPATCH_KERNEL_F32_L1_HEADDIM(2);
  } else {
    DISPATCH_KERNEL_F32_L1_HEADDIM(1);
  }
#endif

#undef CASE_LAUNCH_KERNEL_F32_L1
#undef DISPATCH_KERNEL_F32_L1_HEADDIM
}

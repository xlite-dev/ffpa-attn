#include "launch_templates.cuh"
using namespace ffpa;  


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
  
#ifdef ENABLE_FFPA_ALL_STAGES
  // dispatch stages
  if (stages == 2) {
    DISPATCH_HEADDIM(LAUNCHER_L1, 2);
  } else if (stages == 3) {
    DISPATCH_HEADDIM(LAUNCHER_L1, 3);
  } else if (stages == 4) {
    DISPATCH_HEADDIM(LAUNCHER_L1, 4);
  } else {
    DISPATCH_HEADDIM(LAUNCHER_L1, 1);
  }
#else 
  // dispatch stages
  if (stages == 2) {
    DISPATCH_HEADDIM(LAUNCHER_L1, 2);
  } else {
    DISPATCH_HEADDIM(LAUNCHER_L1, 1);
  }
#endif
}

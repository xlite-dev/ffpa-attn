#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <mma.h>
#include <torch/types.h>
#include <torch/extension.h>
using namespace nvcuda;

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])
#define FLOAT2(value) (reinterpret_cast<float2*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162*>(&(value))[0])
#define LDST32BITS(value) (reinterpret_cast<half2*>(&(value))[0])
#define LDST64BITS(value) (reinterpret_cast<float2*>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])
// gmem -> smem
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n) asm volatile("cp.async.wait_group %0;\n" ::"n"(n))
// ca(cache all, L1 + L2): support 4, 8, 16 bytes, cg(cache global, L2): only support 16 bytes.
#define CP_ASYNC_CA(dst, src, bytes) asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))
#define CP_ASYNC_CG(dst, src, bytes) asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))
// ldmatrix
#define LDMATRIX_X1(R, addr) asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n" : "=r"(R) : "r"(addr))
#define LDMATRIX_X2(R0, R1, addr) asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" : "=r"(R0), "=r"(R1) : "r"(addr))
#define LDMATRIX_X4(R0, R1, R2, R3, addr) asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3) : "r"(addr))
#define LDMATRIX_X1_T(R, addr) asm volatile("ldmatrix.sync.aligned.x1.trans.m8n8.shared.b16 {%0}, [%1];\n" : "=r"(R) : "r"(addr))
#define LDMATRIX_X2_T(R0, R1, addr) asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n" : "=r"(R0), "=r"(R1) : "r"(addr))
#define LDMATRIX_X4_T(R0, R1, R2, R3, addr) asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3) : "r"(addr))
// mma m16n8k16 acc f32 or f16
#define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1) asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n" : "=r"(RD0), "=r"(RD1) : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), "r"(RC1))
#define HMMA16816F32(RD0, RD1, RD2, RD3, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1, RC2, RC3) asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,  %1,  %2,  %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n" : "=r"(RD0), "=r"(RD1), "=r"(RD2), "=r"(RD3): "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), "r"(RC1), "r"(RC2), "r"(RC3))


__device__ __host__ inline 
int div_ceil(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

template<typename T, const int kWarpSize = WARP_SIZE>
__device__ inline T warp_reduce_sum(T val) {
  #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask, kWarpSize);
  }
  return val;
}

template<typename T, const int kWarpSize = WARP_SIZE>
__device__ inline T warp_reduce_max(T val) {
  #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val = max(val, __shfl_xor_sync(0xffffffff, val, mask, kWarpSize));
  }
  return val;
}

template<typename T, int M, const int N, const int K = 2>
__device__ inline void fill_3D_regs(T (&R)[M][N][K], T val) {
  #pragma unroll
  for (int i = 0; i < M; ++i) {
    #pragma unroll
    for (int j = 0; j < N; ++j) {
      #pragma unroll
      for (int k = 0; k < K; ++k) {
        R[i][j][k] = val;
      }
    }
  }
}

template<typename T, int M, const int N = 2>
__device__ inline void fill_2D_regs(T (&R)[M][N], T val) {
  #pragma unroll
  for (int i = 0; i < M; ++i) {
    #pragma unroll
    for (int j = 0; j < N; ++j) {
      R[i][j] = val;
    }
  }
}

template<typename T, int M>
__device__ inline void fill_1D_regs(T (&S)[M], T val) {
  #pragma unroll
  for (int i = 0; i < M; ++i) {
    S[i] = val;
  }
}

#ifdef FFPA_MMA_DEBUG
#define FFPA_MMA_PRINT_T0_REG(R, format, ...)    \
{                                                \
  if (tid == 0) {                                \
    float2 v_reg = __half22float2(HALF2(R));     \
    printf("[T0] " format ", V0=%f, V1=%f\n",    \
           ##__VA_ARGS__, v_reg.x, v_reg.y);     \
  }                                              \
}

#define FFPA_MMA_PRINT_T32_REG(R, format, ...)   \
{                                                \
  if (tid < 32) {                                \
    float2 v_reg = __half22float2(HALF2(R));     \
    printf("[T%d] " format ", V0=%f, V1=%f\n",   \
           tid, ##__VA_ARGS__, v_reg.x, v_reg.y);\
  }                                              \
}

#define FFPA_MMA_PRINT_REG(R, format, ...)       \
{                                                \
  {                                              \
    float2 v_reg = __half22float2(HALF2(R));     \
    printf(format", V0=%f, V1=%f\n",             \
           ##__VA_ARGS__, v_reg.x, v_reg.y);     \
  }                                              \
}

#define FFPA_MMA_CHECK_PRINT_REG(R0, R1, format, ...)                     \
{                                                                         \
  {                                                                       \
    float2 v_reg_0 = __half22float2(HALF2(R0));                           \
    float2 v_reg_1 = __half22float2(HALF2(R1));                           \
    if ((fabs(v_reg_0.x - v_reg_1.x) > 0.01f) ||                          \
        (fabs(v_reg_0.y - v_reg_1.y) > 0.01f)) {                          \
      printf(format", R0, V0=%f, V1=%f, R1, V0=%f, V1=%f\n",              \
             ##__VA_ARGS__, v_reg_0.x, v_reg_0.y, v_reg_1.x, v_reg_1.y);  \
    }                                                                     \
  }                                                                       \
}

#define FFPA_MMA_CHECK_PRINT_T32_REG(R0, R1, format, ...)                 \
{                                                                         \
  if (tid < 32){                                                          \
    float2 v_reg_0 = __half22float2(HALF2(R0));                           \
    float2 v_reg_1 = __half22float2(HALF2(R1));                           \
    if ((fabs(v_reg_0.x - v_reg_1.x) > 0.01f) ||                          \
        (fabs(v_reg_0.y - v_reg_1.y) > 0.01f)) {                          \
      printf(format", R0, V0=%f, V1=%f, R1, V0=%f, V1=%f\n",              \
             ##__VA_ARGS__, v_reg_0.x, v_reg_0.y, v_reg_1.x, v_reg_1.y);  \
    }                                                                     \
  }                                                                       \
}

#define FFPA_MMA_PRINT_T0(format, ...)          \
{                                               \
  if (tid == 0) {                               \
    printf("[T0] " format, ##__VA_ARGS__);      \
  }                                             \
}

#define FFPA_MMA_PRINT_T32(format, ...)         \
{                                               \
  if (tid < 32) {                               \
    printf("[T%d] " format, tid, ##__VA_ARGS__);\
  }                                             \
}

#define FFPA_MMA_PRINT_L0_REG(R, format, ...)     \
{                                                 \
  if (lane_id == 0) {                             \
    float2 v_reg = __half22float2(HALF2(R));      \
    printf("[L0] " format", V0=%f, V1=%f\n",      \
           ##__VA_ARGS__, v_reg.x, v_reg.y);      \
  }                                               \
}

#define FFPA_MMA_PRINT_L0(format, ...)          \
{                                               \
  if (lane_id == 0) {                           \
    printf("[L0] " format, ##__VA_ARGS__);      \
  }                                             \
}

#define FFPA_MMA_PRINT_T0_B0_MATRIX(B, format, ...)        \
{                                                          \
  if (tid == 0 && blockIdx.z == 0) {                       \
    printf("----------------------------------------\n");  \
    printf(format, ##__VA_ARGS__);                         \
    for (int i = 0; i < Br; ++i) {                         \
      for (int j = 0; j < kMmaTileSeqLenK; ++j) {          \
        printf("[%d][%d]=%f", i, j, (B)[i][j]);            \
      }                                                    \
      printf("\n");                                        \
    }                                                      \
    printf("----------------------------------------\n");  \
  }                                                        \
  __syncthreads();                                         \
}

#else

#define FFPA_MMA_PRINT_REG(R, format, ...) {}
#define FFPA_MMA_CHECK_PRINT_REG(R0, R1, format, ...) {}
#define FFPA_MMA_PRINT_T0_REG(R, format, ...) {}
#define FFPA_MMA_PRINT_T32_REG(R, format, ...) {}
#define FFPA_MMA_PRINT_L0_REG(R, format, ...) {}
#define FFPA_MMA_PRINT_T0(format, ...) {}
#define FFPA_MMA_PRINT_T32(format, ...) {}
#define FFPA_MMA_PRINT_L0(format, ...) {}
#define FFPA_MMA_PRINT_T0_B0_MATRIX(B, format, ...) {}

#endif

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                 \
if(((T).options().dtype() != (th_type))) {                   \
  std::cout << "Tensor Info:" << (T).options() << std::endl; \
  throw std::runtime_error("values must be "#th_type);       \
}

#define CHECK_TORCH_TENSOR_SHAPE(T1, T2)             \
if (((T2).size(0) != (T1).size(0)) ||                \
    ((T2).size(1) != (T1).size(1)) ||                \
    ((T2).size(2) != (T1).size(2)) ||                \
    ((T2).size(3) != (T1).size(3))) {                \
  throw std::runtime_error("Tensor size mismatch!"); \
}

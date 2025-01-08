#pragma once
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

#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])


#ifdef ENABLE_FFPA_DEBUG
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


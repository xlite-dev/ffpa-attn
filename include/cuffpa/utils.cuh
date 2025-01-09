#pragma once
#include "cuffpa/logging.cuh" // log

namespace ffpa {
namespace utils {

__device__ __host__ inline 
int div_ceil(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

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

} // utils
} // ffpa


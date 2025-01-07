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

#define WARP_SIZE 32

namespace ffpa {
namespace warp {

template<typename T, const int kWarpSize = WARP_SIZE>
__device__ inline T reduce_sum(T val) {
  #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask, kWarpSize);
  }
  return val;
}

template<typename T, const int kWarpSize = WARP_SIZE>
__device__ inline T reduce_max(T val) {
  #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val = max(val, __shfl_xor_sync(0xffffffff, val, mask, kWarpSize));
  }
  return val;
}

} // warp
} // ffpa

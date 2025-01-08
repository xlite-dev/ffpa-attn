// cp async operations
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

namespace ffpa {
namespace cp_async {

// Simple wrappers for cp.async/ld/st instructions.
__device__ __forceinline__ void commit_group() {
  asm volatile("cp.async.commit_group;\n" ::);
}

// e.g: wait_group<1>();
template <size_t n>
__device__ __forceinline__ void wait_group() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
}

// e.g: cp_async<half, 16>(smem_ptr, gmem_ptr);
template <typename T, const int kBytes = 16>
__device__ __forceinline__ void cp_async(
  uint32_t smem_ptr, const T* gmem_ptr) {
  static_assert(kBytes == 16 || kBytes == 8); // 8 or 4 halfs
  if constexpr (kBytes == 16) {
    asm volatile(
      "cp.async.cg.shared.global.L2::128B "
      "[%0], [%1], %2, %3;\n" 
      ::"r"(smem_ptr), "l"(gmem_ptr), 
        "n"(16), "r"(16)
    );
  } else {
    asm volatile(
      "cp.async.ca.shared.global.L2::128B "
      "[%0], [%1], %2, %3;\n" 
      ::"r"(smem_ptr), "l"(gmem_ptr), 
        "n"(8), "r"(8)
    );
  }
}

// e.g ldg_sync_128b<half, uint32_t>(...);
template <typename T0, typename T1>
__device__ __forceinline__ void ldg_sync_128b(
  T0 * mem_dst_ptr, T1 * gmem_src_ptr) {
  using _128b_t = uint4;
  _128b_t * dst_128b_ptr = reinterpret_cast<_128b_t*>(
    mem_dst_ptr);
  _128b_t * src_128b_ptr = reinterpret_cast<_128b_t*>(
    gmem_src_ptr);
  *(dst_128b_ptr) = *(src_128b_ptr);
}

// e.g stg_sync_128b<half, uint32_t>(...);
template <typename T0, typename T1>
__device__ __forceinline__ void stg_sync_128b(
  T0 * gmem_dst_ptr, T1 * mem_src_ptr) {
  using _128b_t = uint4;
  _128b_t * dst_128b_ptr = reinterpret_cast<_128b_t*>(
    gmem_dst_ptr);
  _128b_t * src_128b_ptr = reinterpret_cast<_128b_t*>(
    mem_src_ptr);
  *(dst_128b_ptr) = *(src_128b_ptr);
}

} // cp_async
} // ffpa

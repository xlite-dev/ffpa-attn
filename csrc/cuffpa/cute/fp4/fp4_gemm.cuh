// NVFP4 blockscaled-GEMM helpers shared across fp4 attention kernels
// (sm_120 persist_d today, a future split-D family later), the fp4
// counterpart of cute/gemm.cuh's gemm_ss/gemm_rs primitives. Depends only
// on fp4_pscale.cuh (P quantization) - no sm_120-only headers here.
#pragma once

#include <cute/tensor.hpp>

namespace ffpa_fp4 {

// K/V^T storage column j -> original token index (the quantize kernels'
// 32-row interleave; bijection inside every 32-window, identity across).
// Table for j in [0,32): [0,1,8,9,16,17,24,25,2,3,10,11,18,19,26,27,
// 4,5,12,13,20,21,28,29,6,7,14,15,22,23,30,31].
CUTE_DEVICE int kv_perm32(int j) {
  const int loc = j & 31;
  return (j & ~31) + (loc / 8) * 2 + ((loc % 8) / 2) * 8 + (loc % 8) % 2;
}

}  // namespace ffpa_fp4

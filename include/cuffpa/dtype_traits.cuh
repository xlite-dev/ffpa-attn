#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace ffpa {

// DtypeTraits<T> abstracts the element-level conversions and arithmetic that
// FFPA kernels use on the *activation* dtype (Q/K/V/O, plus the on-register P
// tile after softmax). Specializations are provided for `__half` and
// `__nv_bfloat16`. The PTX mma/ldmatrix paths are shape-compatible between the
// two dtypes (both occupy 16 bits), so only the float<->dtype conversion and
// pairwise-max helpers need to be dispatched here.
template <typename T>
struct DtypeTraits;

template <>
struct DtypeTraits<__half> {
  using type = __half;

  static __device__ __forceinline__ __half from_float(float x) { return __float2half_rn(x); }

  static __device__ __forceinline__ float to_float(__half x) { return __half2float(x); }

  static __device__ __forceinline__ __half hmax(__half a, __half b) { return __hmax(a, b); }
};

template <>
struct DtypeTraits<__nv_bfloat16> {
  using type = __nv_bfloat16;

  static __device__ __forceinline__ __nv_bfloat16 from_float(float x) {
    return __float2bfloat16_rn(x);
  }

  static __device__ __forceinline__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }

  static __device__ __forceinline__ __nv_bfloat16 hmax(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __hmax(a, b);
  }
};

}  // namespace ffpa

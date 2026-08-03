#pragma once

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <cuda/barrier>

#include <cstdlib>
#include <stdexcept>
#include <type_traits>

namespace ffpa {
namespace tma {

using barrier_t = cuda::barrier<cuda::thread_scope_block>;

// Warpgroup-level register rebalancing via setmaxnreg.
// Effective on sm_90a / sm_100a where ptxas honours the hint.
// On sm_120 / sm_120a (__CUDA_ARCH__ == 1200) the instruction is either
// unsupported (sm_120) or silently ignored by ptxas (sm_120a, C7506:
// cp.async.bulk.tensor treated as implicit extern boundary).  Gate to no-op
// on arch 1200 so builds targeting sm_120 or sm_120a compile cleanly.
template <uint32_t kNumRegs>
__device__ __forceinline__ void warpgroup_reg_dealloc() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && __CUDA_ARCH__ != 1200
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(kNumRegs));
#endif
}

template <uint32_t kNumRegs>
__device__ __forceinline__ void warpgroup_reg_alloc() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && __CUDA_ARCH__ != 1200
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(kNumRegs));
#endif
}

template <typename T>
constexpr CUtensorMapDataType get_tensor_map_dtype() {
  if constexpr (std::is_same_v<T, __half>) {
    return CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
  } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  } else if constexpr (std::is_same_v<T, float>) {
    return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
  } else {
    static_assert(std::is_same_v<T, void>, "Unsupported TMA dtype");
  }
}

inline bool device_supports_tma(int device_index) {
  int major = 0;
  cudaError_t status = cudaDeviceGetAttribute(
      &major, cudaDevAttrComputeCapabilityMajor, device_index);
  if (status != cudaSuccess) {
    return false;
  }
  return major >= 9;
}

inline PFN_cuTensorMapEncodeTiled_v12000 get_cu_tensor_map_encode_tiled() {
  cudaDriverEntryPointQueryResult driver_status;
  void* entry_point = nullptr;

#if CUDA_VERSION >= 12050
  cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &entry_point,
                                   12000, cudaEnableDefault, &driver_status);
#else
  cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &entry_point,
                          cudaEnableDefault, &driver_status);
#endif
  if (driver_status != cudaDriverEntryPointSuccess || entry_point == nullptr) {
    throw std::runtime_error(
        "Failed to resolve cuTensorMapEncodeTiled entry point");
  }
  return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(entry_point);
}

template <typename T>
struct Copy2DDescriptorParams {
  T* global_address;
  uint64_t minor_dim;
  uint64_t major_dim;
  uint64_t major_stride_bytes;
  uint32_t box_minor_dim;
  uint32_t box_major_dim;
  CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE;
  CUtensorMapL2promotion l2_promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
  CUtensorMapFloatOOBfill oob_fill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
};

template <typename T>
inline CUtensorMap make_2d_copy_desc(const Copy2DDescriptorParams<T>& params) {
  CUtensorMap tensor_map{};
  constexpr uint32_t rank = 2;
  uint64_t global_dims[rank] = {params.minor_dim, params.major_dim};
  uint64_t global_stride[rank - 1] = {params.major_stride_bytes};
  uint32_t box_dims[rank] = {params.box_minor_dim, params.box_major_dim};
  uint32_t elem_strides[rank] = {1, 1};

  auto encode = get_cu_tensor_map_encode_tiled();
  CUresult result =
      encode(&tensor_map, get_tensor_map_dtype<T>(), rank,
             params.global_address, global_dims, global_stride, box_dims,
             elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, params.swizzle,
             params.l2_promotion, params.oob_fill);
  if (result != CUDA_SUCCESS) {
    throw std::runtime_error(
        "cuTensorMapEncodeTiled failed for FFPA experimental TMA descriptor");
  }
  return tensor_map;
}

__host__ __device__ __forceinline__ void init_barrier(barrier_t* barrier,
                                                      int arrive_count) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  namespace cde = cuda::device::experimental;
  init(barrier, arrive_count);
#if CUDART_VERSION >= 13020
  cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
#else
  cde::fence_proxy_async_shared_cta();
#endif
#endif
}

__host__ __device__ __forceinline__ void wait_barrier(barrier_t& barrier) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  namespace cde = cuda::device::experimental;
  barrier.wait(barrier.arrive());
#if CUDART_VERSION >= 13020
  cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
#else
  cde::fence_proxy_async_shared_cta();
#endif
#endif
}

__host__ __device__ __forceinline__ void wait_barrier_parity(barrier_t& barrier,
                                                             uint32_t phase) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  namespace cde = cuda::device::experimental;
  barrier.wait_parity(phase != 0);
#if CUDART_VERSION >= 13020
  cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
#else
  cde::fence_proxy_async_shared_cta();
#endif
#endif
}

__host__ __device__ __forceinline__ void fence_async_shared() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  namespace cde = cuda::device::experimental;
#if CUDART_VERSION >= 13020
  cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
#else
  cde::fence_proxy_async_shared_cta();
#endif
#endif
}

__host__ __device__ __forceinline__ void bulk_commit_group() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("cp.async.bulk.commit_group;\n" ::);
#endif
}

template <size_t n>
__host__ __device__ __forceinline__ void bulk_wait_group() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("cp.async.bulk.wait_group %0;\n" ::"n"(n));
#endif
}

__host__ __device__ __forceinline__ void load_2d(
    void* smem_ptr, const CUtensorMap* tensor_map, int32_t minor_coord,
    int32_t major_coord, barrier_t& barrier, uint32_t bytes,
    int issuer_lane = 0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  namespace cde = cuda::device::experimental;
  if (static_cast<int>(threadIdx.x) == issuer_lane) {
#if CUDART_VERSION >= 13020
    const int32_t coords[]{minor_coord, major_coord};
    auto* barrier_handle = cuda::device::barrier_native_handle(barrier);
    cuda::ptx::cp_async_bulk_tensor(cuda::ptx::space_cluster,
                                    cuda::ptx::space_global, smem_ptr,
                                    tensor_map, coords, barrier_handle);
    [[maybe_unused]] auto token = cuda::ptx::mbarrier_arrive_expect_tx(
        cuda::ptx::sem_release, cuda::ptx::scope_cta, cuda::ptx::space_shared,
        barrier_handle, bytes);
#else
    cde::cp_async_bulk_tensor_2d_global_to_shared(
        smem_ptr, tensor_map, minor_coord, major_coord, barrier);
    [[maybe_unused]] auto token =
        cuda::device::barrier_arrive_tx(barrier, 1, bytes);
#endif
  }
#endif
}

__host__ __device__ __forceinline__ void load_2d_no_arrive(
    void* smem_ptr, const CUtensorMap* tensor_map, int32_t minor_coord,
    int32_t major_coord, barrier_t& barrier, int issuer_lane = 0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  namespace cde = cuda::device::experimental;
  if (static_cast<int>(threadIdx.x) == issuer_lane) {
#if CUDART_VERSION >= 13020
    const int32_t coords[]{minor_coord, major_coord};
    auto* barrier_handle = cuda::device::barrier_native_handle(barrier);
    cuda::ptx::cp_async_bulk_tensor(cuda::ptx::space_cluster,
                                    cuda::ptx::space_global, smem_ptr,
                                    tensor_map, coords, barrier_handle);
#else
    cde::cp_async_bulk_tensor_2d_global_to_shared(
        smem_ptr, tensor_map, minor_coord, major_coord, barrier);
#endif
  }
#endif
}

__host__ __device__ __forceinline__ void arrive_expect_tx(barrier_t& barrier,
                                                          uint32_t bytes,
                                                          int issuer_lane = 0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  namespace cde = cuda::device::experimental;
  if (static_cast<int>(threadIdx.x) == issuer_lane) {
#if CUDART_VERSION >= 13020
    auto* barrier_handle = cuda::device::barrier_native_handle(barrier);
    [[maybe_unused]] auto token = cuda::ptx::mbarrier_arrive_expect_tx(
        cuda::ptx::sem_release, cuda::ptx::scope_cta, cuda::ptx::space_shared,
        barrier_handle, bytes);
#else
    [[maybe_unused]] auto token =
        cuda::device::barrier_arrive_tx(barrier, 1, bytes);
#endif
  }
#endif
}

template <const int BrOrBc, const int kHeadDim, const int kCols,
          const int kTileSize, typename T>
__host__ __device__ __forceinline__ bool issue_load_2d_to_dst_swizzled(
    T* dst_smem_base_ptr, const CUtensorMap* tensor_map, const int major_coord,
    const int d_tile_id, const int dst_stage, barrier_t& barrier,
    int issuer_lane = 0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if (tensor_map == nullptr || d_tile_id >= (kHeadDim / kCols)) {
    return false;
  }
  T* dst_stage_ptr = dst_smem_base_ptr + dst_stage * kTileSize;
  load_2d(dst_stage_ptr, tensor_map, d_tile_id * kCols, major_coord, barrier,
          BrOrBc * kCols * sizeof(T), issuer_lane);
  return true;
#else
  return false;
#endif
}

template <const int BrOrBc, const int kHeadDim, const int kCols, typename T>
__host__ __device__ __forceinline__ bool issue_load_2d_to_tmp(
    T* tmp_smem_base_ptr, const CUtensorMap* tensor_map, const int major_coord,
    const int d_tile_id, const int tmp_stage, const int seqlen_bound,
    barrier_t& barrier) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if (tensor_map == nullptr || d_tile_id >= (kHeadDim / kCols) ||
      ((major_coord + BrOrBc) > seqlen_bound)) {
    return false;
  }
  T* tmp_stage_ptr = tmp_smem_base_ptr + tmp_stage * BrOrBc * kCols;
  load_2d(tmp_stage_ptr, tensor_map, d_tile_id * kCols, major_coord, barrier,
          BrOrBc * kCols * sizeof(T));
  return true;
#else
  return false;
#endif
}

template <const int BrOrBc, const int kTileSize, const int kCols,
          const int kNumThreads, const int kPad, typename T>
__host__ __device__ __forceinline__ void wait_and_repack_tmp_to_dst(
    T* dst_smem_base_ptr, T* tmp_smem_base_ptr, const int dst_stage,
    const int tmp_stage, barrier_t& barrier) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  constexpr bool kSwizzle = (kPad == 0);
  constexpr int kElemsPerThread = kCols / (kNumThreads / BrOrBc);
  static_assert(kElemsPerThread * sizeof(T) == 16,
                "Experimental TMA repack expects one 16B vector per thread.");

  T* tmp_stage_ptr = tmp_smem_base_ptr + tmp_stage * BrOrBc * kCols;
  wait_barrier(barrier);
  __syncthreads();

  const int tid = threadIdx.x;
  const int row = tid / (kNumThreads / BrOrBc);
  const int col = (tid % (kNumThreads / BrOrBc)) * kElemsPerThread;
  T* src = tmp_stage_ptr + row * kCols + col;
  T* dst = dst_smem_base_ptr + dst_stage * kTileSize + row * (kCols + kPad) +
           (kSwizzle ? (((col >> 3) ^ (row >> 2)) % (kCols >> 3)) << 3 : col);
  *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<uint4*>(src);
  __syncthreads();
#endif
}

template <const int BrOrBc, const int kTileSize, const int kHeadDim,
          const int kCols, const int kNumThreads, const int kPad, typename T>
__host__ __device__ __forceinline__ bool load_2d_to_smem_repack(
    T* dst_smem_base_ptr, T* tmp_smem_base_ptr, const CUtensorMap* tensor_map,
    const int major_coord, const int d_tile_id, const int dst_stage,
    const int tmp_stage, const int seqlen_bound, barrier_t& barrier) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if (!issue_load_2d_to_tmp<BrOrBc, kHeadDim, kCols, T>(
          tmp_smem_base_ptr, tensor_map, major_coord, d_tile_id, tmp_stage,
          seqlen_bound, barrier)) {
    return false;
  }
  wait_and_repack_tmp_to_dst<BrOrBc, kTileSize, kCols, kNumThreads, kPad, T>(
      dst_smem_base_ptr, tmp_smem_base_ptr, dst_stage, tmp_stage, barrier);
  return true;
#else
  return false;
#endif
}

}  // namespace tma
}  // namespace ffpa

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

inline bool is_experimental_tma_enabled() {
  const char* env = std::getenv("ENABLE_FFPA_EXPERIMENTAL_TMA");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

inline bool device_supports_tma(int device_index) {
  int major = 0;
  cudaError_t status =
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_index);
  if (status != cudaSuccess) {
    return false;
  }
  return major >= 9;
}

inline PFN_cuTensorMapEncodeTiled_v12000 get_cu_tensor_map_encode_tiled() {
  cudaDriverEntryPointQueryResult driver_status;
  void* entry_point = nullptr;

#if CUDA_VERSION >= 12050
  cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &entry_point, 12000, cudaEnableDefault,
                                   &driver_status);
#else
  cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &entry_point, cudaEnableDefault,
                          &driver_status);
#endif
  if (driver_status != cudaDriverEntryPointSuccess || entry_point == nullptr) {
    throw std::runtime_error("Failed to resolve cuTensorMapEncodeTiled entry point");
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
      encode(&tensor_map, get_tensor_map_dtype<T>(), rank, params.global_address, global_dims,
             global_stride, box_dims, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, params.swizzle,
             params.l2_promotion, params.oob_fill);
  if (result != CUDA_SUCCESS) {
    throw std::runtime_error("cuTensorMapEncodeTiled failed for FFPA experimental TMA descriptor");
  }
  return tensor_map;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
namespace cde = cuda::device::experimental;

__device__ __forceinline__ void init_barrier(barrier_t* barrier, int arrive_count) {
  init(barrier, arrive_count);
  cde::fence_proxy_async_shared_cta();
}

__device__ __forceinline__ void wait_barrier(barrier_t& barrier) {
  barrier.wait(barrier.arrive());
}

__device__ __forceinline__ void load_2d(void* smem_ptr, const CUtensorMap* tensor_map,
                                        int32_t minor_coord, int32_t major_coord,
                                        barrier_t& barrier, uint32_t bytes) {
  if (threadIdx.x == 0) {
    cde::cp_async_bulk_tensor_2d_global_to_shared(smem_ptr, tensor_map, minor_coord, major_coord,
                                                  barrier);
    [[maybe_unused]] auto token = cuda::device::barrier_arrive_tx(barrier, 1, bytes);
  }
}

template <const int BrOrBc, const int kTileSize, const int kHeadDim, const int kCols,
          const int kNumThreads, const int kPad, typename T>
__device__ __forceinline__ bool load_2d_to_smem_repack(T* dst_smem_base_ptr, T* tmp_smem_base_ptr,
                                                       const CUtensorMap* tensor_map,
                                                       const int major_coord, const int d_tile_id,
                                                       const int dst_stage, const int tmp_stage,
                                                       const int seqlen_bound, barrier_t& barrier) {
  if (tensor_map == nullptr || d_tile_id >= (kHeadDim / kCols) ||
      ((major_coord + BrOrBc) > seqlen_bound)) {
    return false;
  }

  constexpr bool kSwizzle = (kPad == 0);
  constexpr int kElemsPerThread = kCols / (kNumThreads / BrOrBc);
  static_assert(kElemsPerThread * sizeof(T) == 16,
                "Experimental TMA repack expects one 16B vector per thread.");

  T* tmp_stage_ptr = tmp_smem_base_ptr + tmp_stage * BrOrBc * kCols;
  load_2d(tmp_stage_ptr, tensor_map, d_tile_id * kCols, major_coord, barrier,
          BrOrBc * kCols * sizeof(T));
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
  return true;
}
#else
__device__ __forceinline__ void init_barrier(barrier_t* barrier, int arrive_count) {
  (void)barrier;
  (void)arrive_count;
}

__device__ __forceinline__ void wait_barrier(barrier_t& barrier) {
  (void)barrier;
}

__device__ __forceinline__ void load_2d(void* smem_ptr, const CUtensorMap* tensor_map,
                                        int32_t minor_coord, int32_t major_coord,
                                        barrier_t& barrier, uint32_t bytes) {
  (void)smem_ptr;
  (void)tensor_map;
  (void)minor_coord;
  (void)major_coord;
  (void)barrier;
  (void)bytes;
}

template <const int BrOrBc, const int kTileSize, const int kHeadDim, const int kCols,
          const int kNumThreads, const int kPad, typename T>
__device__ __forceinline__ bool load_2d_to_smem_repack(T* dst_smem_base_ptr, T* tmp_smem_base_ptr,
                                                       const CUtensorMap* tensor_map,
                                                       const int major_coord, const int d_tile_id,
                                                       const int dst_stage, const int tmp_stage,
                                                       const int seqlen_bound, barrier_t& barrier) {
  (void)dst_smem_base_ptr;
  (void)tmp_smem_base_ptr;
  (void)tensor_map;
  (void)major_coord;
  (void)d_tile_id;
  (void)dst_stage;
  (void)tmp_stage;
  (void)seqlen_bound;
  (void)barrier;
  return false;
}
#endif

}  // namespace tma
}  // namespace ffpa

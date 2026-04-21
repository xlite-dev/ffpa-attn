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

// Parity-based wait used by the plan-A consumers. mbarriers initialised
// with ``arrive_count == 1`` and signalled exclusively by the producer's
// ``cuda::device::barrier_arrive_tx`` (inside ``load_2d``) flip phase as
// soon as the TMA's tx-count is satisfied. Consumers must therefore wait
// on the *current* phase bit (0 on the first reuse, 1 on the second,
// alternating thereafter) instead of using ``wait(arrive())`` -- the
// latter would over-arrive and corrupt the next phase.
__device__ __forceinline__ void wait_barrier_parity(barrier_t& barrier, uint32_t phase) {
  barrier.wait_parity(phase != 0);
  // The TMA bulk-tensor copy writes via the async proxy; ldmatrix.sync (used
  // by the consumer) reads via the generic proxy. mbarrier wait_parity
  // alone provides only completion of the TMA, not cross-proxy visibility
  // for sm_120 / Blackwell, where a generic-proxy reader must observe an
  // explicit ``fence.proxy.async.shared::cta`` after the producer arrival.
  cde::fence_proxy_async_shared_cta();
}

// ------------------------------------------------------------------
// TMA bulk-copy commit / wait wrappers
// ------------------------------------------------------------------
// ``cp.async.bulk.commit_group`` / ``cp.async.bulk.wait_group`` are the
// TMA-specific counterparts of the cp.async commit/wait primitives in
// ``ffpa::cp_async``. They track ONLY ``cp.async.bulk{,.tensor}`` copies
// and are independent of ``cp.async.commit_group`` (the legacy cp.async
// group counter does NOT track TMA, and vice versa).
//
// Plan A's data path uses mbarrier-based completion (every TMA copy in
// ``load_2d`` arrives on its per-stage mbarrier via
// ``cuda::device::barrier_arrive_tx`` and is awaited in the kernel via
// ``wait_barrier_parity``). Mbarrier-tracking is strictly more
// expressive than the bulk-group counter (per-slot phases, no FIFO
// constraint), so the kernel does not currently call ``bulk_commit_group``
// / ``bulk_wait_group``. They are exposed here for two reasons:
//
//   1. Code-review clarity: it makes the cp.async vs TMA wait split
//      explicit. ``ffpa::cp_async::commit_group/wait_group`` only sync Q
//      (which is still on cp.async); K/V are synced via
//      ``consume_X_tile`` -> ``wait_barrier_parity``.
//   2. Future use: a fused bulk_wait_group<0> at the very end of the
//      kernel would let us drop the per-slot mbarrier waits if we later
//      decide to coarsen the schedule.
__device__ __forceinline__ void bulk_commit_group() {
  asm volatile("cp.async.bulk.commit_group;\n" ::);
}

template <size_t n>
__device__ __forceinline__ void bulk_wait_group() {
  asm volatile("cp.async.bulk.wait_group %0;\n" ::"n"(n));
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

// Plan A: issue a TMA bulk-tensor copy directly into the destination
// swizzled smem slot. The destination layout is the kernel's existing
// ``kPad==0`` XOR-swizzled K/V slot, which (for kCols == 16 fp16, i.e.
// 32B per row) is bit-for-bit equivalent to ``CU_TENSOR_MAP_SWIZZLE_32B``
// (Cute ``Swizzle<1, 4, 3>``: address bit 4 XOR bit 7). The TMA
// descriptor MUST therefore be configured with
// ``CU_TENSOR_MAP_SWIZZLE_32B`` so the hardware writes the same byte
// pattern that the existing ldmatrix kPad==0 path expects.
//
// Returns ``false`` and skips the issue when (a) the descriptor is null
// or (b) ``d_tile_id`` is past the head-dim end (so speculative prefetch
// loops beyond ``kHeadDim/kCols`` are safe to call). KV-axis tail tiles
// (``major_coord + BrOrBc > seqlen_bound``) are NOT skipped; the TMA
// descriptor's OOB-fill (``CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE``) zero-fills
// out-of-bounds rows so no cp.async fallback is needed.
//
// MUST be paired with ``wait_barrier`` on the same barrier slot before
// the destination smem is read.
template <const int BrOrBc, const int kHeadDim, const int kCols, const int kTileSize, typename T>
__device__ __forceinline__ bool issue_load_2d_to_dst_swizzled(
    T* dst_smem_base_ptr, const CUtensorMap* tensor_map, const int major_coord, const int d_tile_id,
    const int dst_stage, barrier_t& barrier) {
  if (tensor_map == nullptr || d_tile_id >= (kHeadDim / kCols)) {
    return false;
  }
  T* dst_stage_ptr = dst_smem_base_ptr + dst_stage * kTileSize;
  load_2d(dst_stage_ptr, tensor_map, d_tile_id * kCols, major_coord, barrier,
          BrOrBc * kCols * sizeof(T));
  return true;
}

// Plan D legacy: issue an asynchronous TMA load of one (BrOrBc, kCols)
// tile into the caller-owned scratch slot ``tmp_stage`` of
// ``tmp_smem_base_ptr``. Kept for the scratch+repack flow; new code on
// SM90+ should prefer ``issue_load_2d_to_dst_swizzled`` above.
template <const int BrOrBc, const int kHeadDim, const int kCols, typename T>
__device__ __forceinline__ bool issue_load_2d_to_tmp(T* tmp_smem_base_ptr,
                                                     const CUtensorMap* tensor_map,
                                                     const int major_coord, const int d_tile_id,
                                                     const int tmp_stage, const int seqlen_bound,
                                                     barrier_t& barrier) {
  if (tensor_map == nullptr || d_tile_id >= (kHeadDim / kCols) ||
      ((major_coord + BrOrBc) > seqlen_bound)) {
    return false;
  }
  T* tmp_stage_ptr = tmp_smem_base_ptr + tmp_stage * BrOrBc * kCols;
  load_2d(tmp_stage_ptr, tensor_map, d_tile_id * kCols, major_coord, barrier,
          BrOrBc * kCols * sizeof(T));
  return true;
}

// Wait on ``barrier`` (signalled by the matching ``issue_load_2d_to_tmp``
// call with the same ``tmp_stage``) and repack the contiguous TMA scratch
// tile at ``tmp_smem_base_ptr[tmp_stage]`` into the kernel's existing
// padded (kPad>0) or XOR-swizzled (kPad==0) destination slot
// ``dst_smem_base_ptr[dst_stage]``. All threads in the block participate.
//
// MUST only be invoked when the matching ``issue_load_2d_to_tmp`` returned
// ``true``; otherwise the wait will block forever (no producer arrival).
template <const int BrOrBc, const int kTileSize, const int kCols, const int kNumThreads,
          const int kPad, typename T>
__device__ __forceinline__ void wait_and_repack_tmp_to_dst(T* dst_smem_base_ptr,
                                                           T* tmp_smem_base_ptr,
                                                           const int dst_stage, const int tmp_stage,
                                                           barrier_t& barrier) {
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
}

// Legacy synchronous helper kept for completeness: issues, waits and
// repacks in a single call. Equivalent to ``issue_load_2d_to_tmp`` +
// ``wait_and_repack_tmp_to_dst`` back-to-back. New code should prefer
// the split helpers so that the issue and the wait can be hoisted apart
// to enable multi-stage TMA pipelining.
template <const int BrOrBc, const int kTileSize, const int kHeadDim, const int kCols,
          const int kNumThreads, const int kPad, typename T>
__device__ __forceinline__ bool load_2d_to_smem_repack(T* dst_smem_base_ptr, T* tmp_smem_base_ptr,
                                                       const CUtensorMap* tensor_map,
                                                       const int major_coord, const int d_tile_id,
                                                       const int dst_stage, const int tmp_stage,
                                                       const int seqlen_bound, barrier_t& barrier) {
  if (!issue_load_2d_to_tmp<BrOrBc, kHeadDim, kCols, T>(tmp_smem_base_ptr, tensor_map, major_coord,
                                                        d_tile_id, tmp_stage, seqlen_bound,
                                                        barrier)) {
    return false;
  }
  wait_and_repack_tmp_to_dst<BrOrBc, kTileSize, kCols, kNumThreads, kPad, T>(
      dst_smem_base_ptr, tmp_smem_base_ptr, dst_stage, tmp_stage, barrier);
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

__device__ __forceinline__ void wait_barrier_parity(barrier_t& barrier, uint32_t phase) {
  (void)barrier;
  (void)phase;
}

__device__ __forceinline__ void bulk_commit_group() {}

template <size_t n>
__device__ __forceinline__ void bulk_wait_group() {}

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

template <const int BrOrBc, const int kHeadDim, const int kCols, const int kTileSize, typename T>
__device__ __forceinline__ bool issue_load_2d_to_dst_swizzled(
    T* dst_smem_base_ptr, const CUtensorMap* tensor_map, const int major_coord, const int d_tile_id,
    const int dst_stage, barrier_t& barrier) {
  (void)dst_smem_base_ptr;
  (void)tensor_map;
  (void)major_coord;
  (void)d_tile_id;
  (void)dst_stage;
  (void)barrier;
  return false;
}

template <const int BrOrBc, const int kHeadDim, const int kCols, typename T>
__device__ __forceinline__ bool issue_load_2d_to_tmp(T* tmp_smem_base_ptr,
                                                     const CUtensorMap* tensor_map,
                                                     const int major_coord, const int d_tile_id,
                                                     const int tmp_stage, const int seqlen_bound,
                                                     barrier_t& barrier) {
  (void)tmp_smem_base_ptr;
  (void)tensor_map;
  (void)major_coord;
  (void)d_tile_id;
  (void)tmp_stage;
  (void)seqlen_bound;
  (void)barrier;
  return false;
}

template <const int BrOrBc, const int kTileSize, const int kCols, const int kNumThreads,
          const int kPad, typename T>
__device__ __forceinline__ void wait_and_repack_tmp_to_dst(T* dst_smem_base_ptr,
                                                           T* tmp_smem_base_ptr,
                                                           const int dst_stage, const int tmp_stage,
                                                           barrier_t& barrier) {
  (void)dst_smem_base_ptr;
  (void)tmp_smem_base_ptr;
  (void)dst_stage;
  (void)tmp_stage;
  (void)barrier;
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

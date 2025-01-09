#pragma once
#include "cuffpa/mma.cuh" // ffpa::mma
#include "cuffpa/warp.cuh" // ffpa::warp
#include "cuffpa/swizzle.cuh" // ffpa::swizzle
#include "cuffpa/cp_async.cuh" // ffpa::cp_async
#include "cuffpa/utils.cuh" // ffpa::utils

namespace ffpa {
namespace prefill {
// prefill utils: prefetch/load QKV g2s funcs, rescale/softmax funcs etc.
// cp_async & commit_group
template<
  const int BrOrBc,
  const int kTileSize, 
  const int kHeadDim,
  const int kMmaAtomK, 
  const int kNumThreads, 
  const int kPad
>
__device__ __forceinline__ void cp_async_qkv_g2s(
  uint32_t smem_base_ptr, // QKV smem base ptr
  const half * gmem_ptr,  // QKV gmem ptr
  const int gmem_offset,  // QKV gmem_offset
  const int n_tile_id,    // seqlen offset, Q_tile_id * Br, tile_K_seqlen * Bc
  const int d_tile_id,    // headdim offset, tile_K_d * kMmaAtomK, tile_V_d * kMmaAtomN * 2
  const int stage         // stage * QKV tile_size
) {
  // QK: tile_K_d < (kHeadDim / kMmaAtomK)
  //  V: tile_V_d < (kHeadDim / kMmaAtomN * 2)
  if (d_tile_id >= (kHeadDim / kMmaAtomK)) { return; }
  const int tid = threadIdx.x; // within block
  const int Q_tile_id = blockIdx.x; // Q tile_id, range [0, Tr]
  constexpr bool kSwizzle = (kPad == 0) ? true : false;
  // Mapping QKV tid -> smem, tile size [64/128, 16]
  // Br 64, tid / 2, row 0~64
  const int load_smem_BrOrBc = (tid / (kNumThreads / BrOrBc)); 
  // (tid % 2) * 8, 0,8,...
  const int load_smem_d = (
    tid % (kNumThreads / BrOrBc)) * (kMmaAtomK / (kNumThreads / BrOrBc));
  // Mapping QKV tid -> gmem, tile size [64/128, 16], row offset by
  // n_tile_id(seqlen), col offset by d_tile_id(Headdim).
  const int load_gmem_BrOrBc = (n_tile_id * BrOrBc) + load_smem_BrOrBc; 
  const int load_gmem_d = (d_tile_id * kMmaAtomK) + load_smem_d; // 0,8
  // Offset by QKV global gmem_offset.
  const int load_gmem_addr = (
    gmem_offset + load_gmem_BrOrBc * kHeadDim + load_gmem_d);

  // cp async & apply swizzle or padding.
  #pragma unroll
  for (int i = 0; i < (kMmaAtomK / (kNumThreads / BrOrBc)); i += 8) {
    uint32_t load_smem_ptr = (
      smem_base_ptr + (stage * kTileSize + 
                       load_smem_BrOrBc * (kMmaAtomK + kPad) + 
                      (kSwizzle ? swizzle::permuted<kMmaAtomK>(
                       load_smem_BrOrBc, load_smem_d + i) : 
                       load_smem_d + i )
                      ) * sizeof(half));
    cp_async::cp_async<16>(load_smem_ptr, &(gmem_ptr[load_gmem_addr + i]));
  }
  cp_async::commit_group();
}

template<
  const int kTrans,
  const int kNumRegs,
  const int kTileSize, 
  const int kMmaAtomM, 
  const int kMmaAtomN, 
  const int kMmaAtomK, 
  const int kPad
>
__device__ __forceinline__ void sync_fetch_qkv_frags(
  uint32_t smem_base_ptr, // QKV smem base ptr
  uint32_t * R,           // Register ptr, R_QKV
  const int mma_tile_id,  // Q warp_QP 0~num MMAs, KV warp_KV 0
  const int warp_tile_id, // Q 0, KV 0~kWarpTileSeqLenK
  const int n_tile_id,    // seqlen QK 0, V tile_V_Bc
  const int stage
) {
  const int lane_id = threadIdx.x % WARP_SIZE; // 0~31
  constexpr bool kSwizzle = (kPad == 0) ? true : false;
  if constexpr (kTrans) {
    // load V m8n8x2 via ldmatrix.x2.trans
    static_assert(kNumRegs == 2);
    // mma_tile_id = warp_KV == 0, warp_tile_id = (j % 2), n_tile_id = tile_V_Bc
    // warp_smem_V_d  = warp_KV * (kMmaAtomN * kWarpTileHeadDimV) + (j % 2) * kMmaAtomN; 
    const int warp_smem_d = warp_tile_id * kMmaAtomN;
    const int lane_smem_Bc = n_tile_id * kMmaAtomK + lane_id % 16;
    const int lane_smem_d  = warp_smem_d; // 0,8
    uint32_t lane_smem_ptr = (
      smem_base_ptr + (stage * kTileSize + 
                       lane_smem_Bc * (kMmaAtomN * 2 + kPad) + 
                      (kSwizzle ? swizzle::permuted<kMmaAtomN * 2>(
                       lane_smem_Bc, lane_smem_d): 
                       lane_smem_d)
                      ) * sizeof(half)
    );
    mma::ldmatrix_m8n8x2_trans(&R[0], &R[1], lane_smem_ptr);
  } else {
    static_assert(kNumRegs == 2 || kNumRegs == 4);
    if constexpr (kNumRegs == 4) {
      // load Q m8n8x4 via ldmatrix.x4 
      // mma_tile_id = warp_QP, kWarpTileSeqLenQ=1
      // warp_smem_Q_Br = warp_QP * (kMmaAtomM * kWarpTileSeqLenQ) + 0 * kMmaAtomM
      const int warp_smem_Br = mma_tile_id * (kMmaAtomM);
      const int lane_smem_Br = warp_smem_Br + lane_id % 16; // 0~15
      const int lane_smem_d  = (lane_id / 16) * 8; // 0,8
      uint32_t lane_smem_ptr = (
        smem_base_ptr + (stage * kTileSize + 
                         lane_smem_Br * (kMmaAtomK + kPad) + 
                        (kSwizzle ? swizzle::permuted<kMmaAtomK>(
                         lane_smem_Br, lane_smem_d): 
                         lane_smem_d)
                        ) * sizeof(half)
      );
      mma::ldmatrix_m8n8x4(&R[0], &R[1], &R[2], &R[3], lane_smem_ptr);
    } else {
      // load K m8n8x2 via ldmatrix.x2
      // mma_tile_id = warp_KV == 0, warp_tile_id = j
      // warp_smem_Bc = warp_KV * (kMmaAtomN * kWarpTileSeqLenK) + j * kMmaAtomN;
      const int warp_smem_Bc = warp_tile_id * kMmaAtomN;
      const int lane_smem_Bc = warp_smem_Bc + lane_id % 8; // 0~7
      const int lane_smem_d  = ((lane_id / 8) % 2) * 8; // 0,8
      uint32_t lane_smem_ptr = (
        smem_base_ptr + (stage * kTileSize + 
                         lane_smem_Bc * (kMmaAtomK + kPad) + 
                        (kSwizzle ? swizzle::permuted<kMmaAtomK>(
                         lane_smem_Bc, lane_smem_d): 
                         lane_smem_d )
                         ) * sizeof(half)
      );
      mma::ldmatrix_m8n8x2(&R[0], &R[1], lane_smem_ptr);
    }
  }
}

} // prefill 
} // ffpa

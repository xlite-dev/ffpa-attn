#pragma once
#include "cuffpa/mma.cuh" // ffpa::mma
#include "cuffpa/warp.cuh" // ffpa::warp
#include "cuffpa/swizzle.cuh" // ffpa::swizzle
#include "cuffpa/cp_async.cuh" // ffpa::cp_async
#include "cuffpa/utils.cuh" // ffpa::utils

namespace ffpa {
namespace prefill {
// prefill utils: prefetch/load QKV g2s funcs, rescale/softmax funcs etc.


template<
  const int kHeadDim,              // Headdim, 32,64,128     
  const int kMmaAtomM,             // MMA Atom M, 16
  const int kMmaAtomN,             // MMA Atom N, 8
  const int kMmaAtomK,             // MMA Atom K, 16
  const int kMmaTileSeqLenQ,       // 4, more MMA(warp), M=16*4=64, Q@K^T=[Br(M), d(K)]@[d(K),  Bc(N)]  
  const int kMmaTileSeqLenK,       // 1, more MMA(warp), N=8*1 =8,  Q@K^T=[Br(M), d(K)]@[d(K),  Bc(N)]    
  const int kMmaTileSeqLenP,       // 4, more MMA(warp), M=16*4=64, P@V  =[Br(M),Bc(K)]@[Bc(K), d(N) ]
  const int kMmaTileHeadDimV,      // 1, more MMA(warp), N=8*1 =8,  P@V  =[Br(M),Bc(K)]@[Bc(K), d(N) ]       
  const int kWarpTileSeqLenQ,      // 1, more values, M, Br=64*1=64, matmul M 
  const int kWarpTileSeqLenK,      // 8, more values, N, Bc=8*8 =64, matmul N
  const int kWarpTileSeqLenP,      // 1, more values, M, Br=64*1=64, matmul M
  const int kWarpTileHeadDimV,     // 8, more values, N, d=8*(1|2|3|4|...)=8|...|32|64|96|128|...
  const int kMmaAccFloat32QK,      // 0/1, Q@K^T, 0 MMA Acc with fp16, 1 MMA Acc with fp32.
  const int kMmaAccFloat32PV,      // 0/1, P@V, 0 MMA Acc with fp16, 1 MMA Acc with fp32.
  const int kOStorageAccFloat32,   // 0/1, MMA Acc always be f32/f16, but O storage can be fp32 or half.
  const int kPrefetchQK,           // Prefetch QK at the Appropriate Time Point. 
  const int kPrefetchPV,           // Prefetch V at the Appropriate Time Point. 
  const int kShareSmemQKV,         // QKV share the same shared memory, reuse QK smem for V.
  const int kPersistQs2r,          // Persist load Q s2r for headdim < 512, more registers, but still keep O(1) SRAM.
  const int kPersistQg2s,          // Persist load Q g2s for headdim < 512, more SRAM, but still keep register usage.
  const int kStageQK,              // <= 4, may apply different multi stages policy for QK and V (<=4)
  const int kStagePV,              // <= 4, may apply different multi stages policy for QK and V (<=4)
  const int kPadQ,                 // Pad Q/K/V 0,8; 0 -> smem swizzle, > 0 -> padding
  const int kPadK,
  const int kPadV            
>
__device__ __forceinline__ void check_compiling_states() {
  // Matmul Layout: Q[Br,d]@K^T[d,Bc] NT, P[Br,Bc]@V[Bc,d] NN.
  // NOTE: K[Bc,d] with row major means K^T[d,Bc] in col major.
  static_assert(kMmaAtomM == 16 && kMmaAtomN == 8 && kMmaAtomK == 16); // m16n8k16
  static_assert(kMmaTileSeqLenQ  <= 8 && kMmaTileSeqLenK  == 1);  // Q@K^T
  static_assert(kMmaTileSeqLenP  <= 8 && kMmaTileHeadDimV == 1);  // P@V
  static_assert(kWarpTileSeqLenQ == 1 && kWarpTileSeqLenK <= 16); // Q@K^T
  static_assert(kWarpTileSeqLenP == 1 && kWarpTileHeadDimV == (
    kHeadDim / (kMmaAtomN * kMmaTileHeadDimV))); // P@V
  static_assert(kMmaAccFloat32QK == 0 || kMmaAccFloat32QK == 1);
  static_assert(kMmaAccFloat32PV == 0 || kMmaAccFloat32PV == 1);
  static_assert(kOStorageAccFloat32 == 0 || kOStorageAccFloat32 == 1);
  // Make sure that Br >= Bc, for shared memory reuse.
  static_assert(
    (kMmaAtomM * kMmaTileSeqLenQ * kWarpTileSeqLenQ) >= 
    (kMmaAtomN * kMmaTileSeqLenK * kWarpTileSeqLenK)); 
  static_assert(kPrefetchQK == 0 || kPrefetchQK == 1);
  static_assert(kPrefetchPV == 0 || kPrefetchPV == 1);
  static_assert(kShareSmemQKV == 0 || kShareSmemQKV == 1);
  // Persist load Q s2r for headdim < 512, more registers, but still keep O(1) SRAM.
  static_assert(kPersistQs2r == 0 || kPersistQs2r == 1);
  // Persist load Q g2s for headdim < 512, more SRAM, but still keep register usage.
  static_assert(kPersistQg2s == 0 || kPersistQg2s == 1);
  // kPersistQg2s and kPersistQs2r can not both enabled.
  static_assert((kPersistQg2s & kPersistQs2r)  == 0);
  // kPersistQg2s and kShareSmemQKV can not both enabled.
  static_assert((kPersistQg2s & kShareSmemQKV) == 0);
  // May apply different multi stages policy for QK and V.
  static_assert(kStageQK < 5 && kStageQK > 0); // QK (<=4)
  static_assert(kStagePV < 5 && kStagePV > 0); // V  (<=4)
  static_assert(kPadQ >= 0 && kPadQ % 8 == 0); // 0,8,16
  static_assert(kPadK >= 0 && kPadK % 8 == 0); // 0,8,16
  static_assert(kPadV >= 0 && kPadV % 8 == 0); // 0,8,16
}


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
    const uint32_t load_smem_ptr = (
      smem_base_ptr + (stage * kTileSize + 
                       load_smem_BrOrBc * (kMmaAtomK + kPad) + 
                      (kSwizzle ? swizzle::permuted<kMmaAtomK>(
                       load_smem_BrOrBc, load_smem_d + i) : 
                       load_smem_d + i )
                      ) * sizeof(half));
    cp_async::cp_async<16>(load_smem_ptr, &(gmem_ptr[load_gmem_addr + i]));
  }
  // cp_async::commit_group();
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
__device__ __forceinline__ void sync_fetch_qkv_frags_s2r(
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
    const uint32_t lane_smem_ptr = (
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
      const uint32_t lane_smem_ptr = (
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
      const uint32_t lane_smem_ptr = (
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


template<const int kWarpTileSeqLenK, const int kMmaAccFloat32>
__device__ __forceinline__ void sync_online_safe_softmax(
  uint32_t * R_S,                       // &R_S[0][0][0]
  const float scale,                    // 1 / sqrt(d)
  float * lane_row_max_new,       // &lane_row_max_new[0][0]
  float * lane_row_sum_new,       // &lane_row_sum_new[0][0]
  float * lane_block_row_max_old, // &lane_block_row_max_old[0][0]
  float * lane_block_row_sum_old  // &lane_block_row_sum_old[0][0]
) {
  if constexpr (kMmaAccFloat32) {
    // Row max for [Br,Bc] tile, Thread -> Warp -> Block.
    { // kWarpTileSeqLenQ = 1
      // Thread level reduce max across kWarpTileSeqLenK dim, namely Bc.
      #pragma unroll
      for (int j = 0; j < kWarpTileSeqLenK; ++j) {
        const float* t_fptr_S_0_1 = reinterpret_cast<float*>(R_S + j * 4); // &R_S[0][j][0]
        // This should be the row max after S = (Q @ K^T) / sqrt(d)
        const float tmp_max_0 = max(t_fptr_S_0_1[0], t_fptr_S_0_1[1]) * scale;
        const float tmp_max_1 = max(t_fptr_S_0_1[2], t_fptr_S_0_1[3]) * scale;
        lane_row_max_new[0] = max(lane_row_max_new[0], tmp_max_0);
        lane_row_max_new[1] = max(lane_row_max_new[1], tmp_max_1);
      } // end for kWarpTileSeqLenK

      // Warp level reduce max, warp_size = 4
      // Each thread contains the maximum of 2 rows of Br, 
      // and only the values of T0, T4, ..., T28 are used.
      lane_row_max_new[0] = warp::reduce_max<float, 4>(lane_row_max_new[0]);
      lane_row_max_new[1] = warp::reduce_max<float, 4>(lane_row_max_new[1]);
    } // end for kWarpTileSeqLenQ

    // static_assert(kWarpTileSeqLenQ == 1);
    // Exp sum and mul scale_factor for [Br,Bc] tile, Thread -> Warp -> Block.
    { // kWarpTileSeqLenQ = 1
      // Use latest global row max without update.
      // Br 0, row_id, 0~7,  16~23, 32~39, 48~55; 
      // float block_row_max_new_0 = lane_row_max_new[0]; 
      // // Br 1, row_id, 8~15, 24~31, 40~47, 56~63;
      // float block_row_max_new_1 = lane_row_max_new[1];
      // const float block_row_max_old_0 = lane_block_row_max_old[0];
      // const float block_row_max_old_1 = lane_block_row_max_old[1];
      // Apply m_new = max(m_old, m_new) here.
      const float block_row_max_new_0 = max(lane_block_row_max_old[0], lane_row_max_new[0]);
      const float block_row_max_new_1 = max(lane_block_row_max_old[1], lane_row_max_new[1]);

      #pragma unroll
      for (int j = 0; j < kWarpTileSeqLenK; ++j) {
        // R_S[][][4] 4 32bit registers with each contains 1 F32 element.
        // (x,y) 0~7->{c0, c1}, (z,w)->8~15 {c2, c3}
        float* t_fptr_S_0_1 = reinterpret_cast<float*>(R_S + j * 4); 
        half*  t_hptr_S_0_1 = reinterpret_cast< half*>(R_S + j * 4); 
        // P = Exp(S - m_new), fmaf(x, y, z) = x * y + z in registers;
        t_fptr_S_0_1[0] = __expf(__fmaf_rn(t_fptr_S_0_1[0], scale, - block_row_max_new_0));
        t_fptr_S_0_1[1] = __expf(__fmaf_rn(t_fptr_S_0_1[1], scale, - block_row_max_new_0));
        t_fptr_S_0_1[2] = __expf(__fmaf_rn(t_fptr_S_0_1[2], scale, - block_row_max_new_1));
        t_fptr_S_0_1[3] = __expf(__fmaf_rn(t_fptr_S_0_1[3], scale, - block_row_max_new_1));
        lane_row_sum_new[0] += (t_fptr_S_0_1[0] + t_fptr_S_0_1[1]);
        lane_row_sum_new[1] += (t_fptr_S_0_1[2] + t_fptr_S_0_1[3]);
        // Update R_S for P[Br,Bc] = Exp(S-m), point wise.
        // Also convert F32 -> half for P@V MMA, reuse R_S as P.
        t_hptr_S_0_1[0] = __float2half_rn(t_fptr_S_0_1[0]);
        t_hptr_S_0_1[1] = __float2half_rn(t_fptr_S_0_1[1]);
        t_hptr_S_0_1[2] = __float2half_rn(t_fptr_S_0_1[2]);
        t_hptr_S_0_1[3] = __float2half_rn(t_fptr_S_0_1[3]);
      } // end for kWarpTileSeqLenK

      // Warp level reduce sum, warp_size = 4
      lane_row_sum_new[0] = warp::reduce_sum<float, 4>(lane_row_sum_new[0]);
      lane_row_sum_new[1] = warp::reduce_sum<float, 4>(lane_row_sum_new[1]);
    }

  } else {
    // MMA Acc F16
    // Row max for [Br,Bc] tile, Thread -> Warp -> Block.
    { // kWarpTileSeqLenQ = 1
      // Thread level reduce max across kWarpTileSeqLenK dim, namely Bc.
      #pragma unroll
      for (int j = 0; j < kWarpTileSeqLenK; ++j) {
        const half* t_hptr_S_0_1 = reinterpret_cast<half*>(R_S + j * 2); 
        // This should be the row max after S = (Q @ K^T) / sqrt(d)
        const float tmp_max_0 = __half2float(__hmax(t_hptr_S_0_1[0], t_hptr_S_0_1[1])) * scale;
        const float tmp_max_1 = __half2float(__hmax(t_hptr_S_0_1[2], t_hptr_S_0_1[3])) * scale;
        lane_row_max_new[0] = max(lane_row_max_new[0], tmp_max_0);
        lane_row_max_new[1] = max(lane_row_max_new[1], tmp_max_1);
      } // end for kWarpTileSeqLenK

      // Warp level reduce max, warp_size = 4
      // Each thread contains the maximum of 2 rows of Br, 
      // and only the values of T0, T4, ..., T28 are used.
      lane_row_max_new[0] = warp::reduce_max<float, 4>(lane_row_max_new[0]);
      lane_row_max_new[1] = warp::reduce_max<float, 4>(lane_row_max_new[1]);
    } // end for kWarpTileSeqLenQ

    // static_assert(kWarpTileSeqLenQ == 1);
    // Exp sum and mul scale_factor for [Br,Bc] tile, Thread -> Warp -> Block.
    { // kWarpTileSeqLenQ = 1
      // Use latest global row max without update.
      // Br 0, row_id, 0~7,  16~23, 32~39, 48~55; 
      // float block_row_max_new_0 = lane_row_max_new[0]; 
      // // Br 1, row_id, 8~15, 24~31, 40~47, 56~63;
      // float block_row_max_new_1 = lane_row_max_new[1];
      // const float block_row_max_old_0 = lane_block_row_max_old[0];
      // const float block_row_max_old_1 = lane_block_row_max_old[1];
      // Apply m_new = max(m_old, m_new) here.
      const float block_row_max_new_0 = max(lane_block_row_max_old[0], lane_row_max_new[0]);
      const float block_row_max_new_1 = max(lane_block_row_max_old[1], lane_row_max_new[1]);

      #pragma unroll
      for (int j = 0; j < kWarpTileSeqLenK; ++j) {
        half* t_hptr_S_0_1 = reinterpret_cast<half*>(R_S + j * 2); 
        // P = Exp(S - m_new), fmaf(x, y, z) = x * y + z;
        float4 t_reg_S_0_1;
        t_reg_S_0_1.x = __expf(__fmaf_rn(
          __half2float(t_hptr_S_0_1[0]), scale, - block_row_max_new_0));
        t_reg_S_0_1.y = __expf(__fmaf_rn(
          __half2float(t_hptr_S_0_1[1]), scale, - block_row_max_new_0));
        t_reg_S_0_1.z = __expf(__fmaf_rn(
          __half2float(t_hptr_S_0_1[2]), scale, - block_row_max_new_1));
        t_reg_S_0_1.w = __expf(__fmaf_rn(
          __half2float(t_hptr_S_0_1[3]), scale, - block_row_max_new_1));
        lane_row_sum_new[0] += (t_reg_S_0_1.x + t_reg_S_0_1.y);
        lane_row_sum_new[1] += (t_reg_S_0_1.z + t_reg_S_0_1.w);
        // Update R_S for P[Br,Bc] = Exp(S-m), point wise.
        t_hptr_S_0_1[0] = __float2half_rn(t_reg_S_0_1.x);
        t_hptr_S_0_1[1] = __float2half_rn(t_reg_S_0_1.y);
        t_hptr_S_0_1[2] = __float2half_rn(t_reg_S_0_1.z);
        t_hptr_S_0_1[3] = __float2half_rn(t_reg_S_0_1.w);
      } // end for kWarpTileSeqLenK

      // Warp level reduce sum, warp_size = 4
      lane_row_sum_new[0] = warp::reduce_sum<float, 4>(lane_row_sum_new[0]);
      lane_row_sum_new[1] = warp::reduce_sum<float, 4>(lane_row_sum_new[1]);
    }
  }
}


__device__ __forceinline__ void sync_precompute_rescale_factors(
  float * rescale_o_factor_0,           // rescale factor
  float * rescale_o_factor_1,           // rescale factor
  const float * lane_row_max_new,       // &lane_row_max_new[0][0]
  const float * lane_block_row_max_old, // &lane_block_row_max_old[0][0]
  const int n_tile_id                   // tile_K_seqlen
) {
  float block_row_max_new_0 = lane_row_max_new[0]; 
  float block_row_max_new_1 = lane_row_max_new[1];
  float block_row_max_old_0 = lane_block_row_max_old[0];
  float block_row_max_old_1 = lane_block_row_max_old[1];
  // NOTE: max(-inf, val) = val.
  block_row_max_new_0 = max(block_row_max_old_0, block_row_max_new_0);
  block_row_max_new_1 = max(block_row_max_old_1, block_row_max_new_1);   
  // Avoid inf value while using m_old for rescaling O.
  block_row_max_old_0 = (n_tile_id > 0 ? block_row_max_old_0 : 
                                         block_row_max_new_0);                                       
  block_row_max_old_1 = (n_tile_id > 0 ? block_row_max_old_1 : 
                                         block_row_max_new_1);  
  // Precompute rescale_o_factor_0 & rescale_o_factor_1, avoid redundant exp.                                       
  rescale_o_factor_0[0] = __expf(block_row_max_old_0 - block_row_max_new_0);
  rescale_o_factor_1[0] = __expf(block_row_max_old_1 - block_row_max_new_1); 
}

template<const int kOStorageAccFloat32, const int kMmaAccFloat32>
__device__ __forceinline__ void sync_rescaling_tiling_o(
  uint32_t * R_D,             // &R_D[0][0][0]
  uint32_t * R_O,             // &R_O[0]
  const float * rescale_o_factor_0, // rescale factor
  const float * rescale_o_factor_1, // rescale factor
  const int n_tile_id,        // tile_K_seqlen
  const int d_tile_id         // j
) {
  // Now, we get [Br,8] slice of [Br,d], each warp(MMA) contains m16n8.
  // 0. Rescale O: Online rescaling O each tile_K_seqlen step, need m_new, m_old.
  // m = max(m_old, m_new), O_new[Br,d] = exp(m_old - m) * O_old + P@V
  // m = max(m_old, m_new), l = exp(m_old - m) * l_old + l_new (FA2 paper)
  if constexpr (kMmaAccFloat32) {
    const float* t_fptr_O_0_1 = reinterpret_cast<float*>(R_O); 
    if constexpr (kOStorageAccFloat32) {
      // (x,y) 0~7->{c0, c1}, (z,w)->8~15 {c2, c3} kWarpTileSeqLenP=1
      float* t_fptr_D_0_1 = reinterpret_cast<float*>(R_D + d_tile_id * 4); // &(R_D[0][j][0])
      t_fptr_D_0_1[0] = __fmaf_rn(rescale_o_factor_0[0], t_fptr_D_0_1[0], t_fptr_O_0_1[0]);
      t_fptr_D_0_1[1] = __fmaf_rn(rescale_o_factor_0[0], t_fptr_D_0_1[1], t_fptr_O_0_1[1]);
      t_fptr_D_0_1[2] = __fmaf_rn(rescale_o_factor_1[0], t_fptr_D_0_1[2], t_fptr_O_0_1[2]);
      t_fptr_D_0_1[3] = __fmaf_rn(rescale_o_factor_1[0], t_fptr_D_0_1[3], t_fptr_O_0_1[3]);
    } else {
      half* t_hptr_D_0_1 = reinterpret_cast<half*>(R_D + d_tile_id * 2); 
      t_hptr_D_0_1[0] = __float2half_rn(__fmaf_rn(
        rescale_o_factor_0[0], __half2float(t_hptr_D_0_1[0]), t_fptr_O_0_1[0]));
      t_hptr_D_0_1[1] = __float2half_rn(__fmaf_rn(
        rescale_o_factor_0[0], __half2float(t_hptr_D_0_1[1]), t_fptr_O_0_1[1]));
      t_hptr_D_0_1[2] = __float2half_rn(__fmaf_rn(
        rescale_o_factor_1[0], __half2float(t_hptr_D_0_1[2]), t_fptr_O_0_1[2]));
      t_hptr_D_0_1[3] = __float2half_rn(__fmaf_rn(
        rescale_o_factor_1[0], __half2float(t_hptr_D_0_1[3]), t_fptr_O_0_1[3]));
    }
  } else {
    // MMA Acc F16
    const half* t_hptr_O_0_1 = reinterpret_cast<half*>(R_O); 
    if constexpr (kOStorageAccFloat32) {
      // (x,y) 0~7->{c0, c1}, (z,w)->8~15 {c2, c3} kWarpTileSeqLenP=1
      float* t_fptr_D_0_1 = reinterpret_cast<float*>(R_D + d_tile_id * 4);
      t_fptr_D_0_1[0] = __fmaf_rn(
        rescale_o_factor_0[0], t_fptr_D_0_1[0], __half2float(t_hptr_O_0_1[0]));
      t_fptr_D_0_1[1] = __fmaf_rn(
        rescale_o_factor_0[0], t_fptr_D_0_1[1], __half2float(t_hptr_O_0_1[1]));
      t_fptr_D_0_1[2] = __fmaf_rn(
        rescale_o_factor_1[0], t_fptr_D_0_1[2], __half2float(t_hptr_O_0_1[2]));
      t_fptr_D_0_1[3] = __fmaf_rn(
        rescale_o_factor_1[0], t_fptr_D_0_1[3], __half2float(t_hptr_O_0_1[3]));
    } else {
      half* t_hptr_D_0_1 = reinterpret_cast<half*>(R_D + d_tile_id * 2); 
      t_hptr_D_0_1[0] = __float2half_rn(__fmaf_rn(rescale_o_factor_0[0], 
        __half2float(t_hptr_D_0_1[0]), __half2float(t_hptr_O_0_1[0])));
      t_hptr_D_0_1[1] = __float2half_rn(__fmaf_rn(rescale_o_factor_0[0], 
        __half2float(t_hptr_D_0_1[1]), __half2float(t_hptr_O_0_1[1])));
      t_hptr_D_0_1[2] = __float2half_rn(__fmaf_rn(rescale_o_factor_1[0], 
        __half2float(t_hptr_D_0_1[2]), __half2float(t_hptr_O_0_1[2])));
      t_hptr_D_0_1[3] = __float2half_rn(__fmaf_rn(rescale_o_factor_1[0], 
        __half2float(t_hptr_D_0_1[3]), __half2float(t_hptr_O_0_1[3])));
    } 
  }
}

__device__ __forceinline__ void sync_update_max_expsum(
  float * lane_row_max_new,       // &lane_row_max_new[0][0]
  float * lane_row_sum_new,       // &lane_row_sum_new[0][0]
  float * lane_block_row_max_old, // &lane_block_row_max_old[0][0]
  float * lane_block_row_sum_old, // &lane_block_row_sum_old[0][0]
  const float * rescale_o_factor_0,     // rescale factor 0 exp(m_old - m_new)
  const float * rescale_o_factor_1     // rescale factor 1 exp(m_old - m_new)
  // const int n_tile_id             // tile_K_seqlen
) {
  // Now, we can update m, l after O has been scaled.
  // Update l = exp(m_old - m_new) * l_old + row_sum(P).
  lane_block_row_sum_old[0] = (__fmaf_rn(
    rescale_o_factor_0[0], lane_block_row_sum_old[0], lane_row_sum_new[0]));
  lane_block_row_sum_old[1] = (__fmaf_rn(
    rescale_o_factor_1[0], lane_block_row_sum_old[1], lane_row_sum_new[1]));
  // 2. Then, update block row max for each lane.
  lane_block_row_max_old[0] = max(lane_block_row_max_old[0], lane_row_max_new[0]);
  lane_block_row_max_old[1] = max(lane_block_row_max_old[1], lane_row_max_new[1]);                         
}


template<const int kWarpTileHeadDimV, const int kOStorageAccFloat32>
__device__ __forceinline__ void sync_rescaling_final_o(
  uint32_t * R_D,                // Final O after loop over N
  const float * lane_block_row_sum_old // &lane_block_row_sum_old[0][0]
) {
  // Finaly, we still have to rescale O once more.
  // O_output(D) = ( 1/l_final ) * O_final (FA2 paper)
  // static_assert(kWarpTileSeqLenP == 1);
  { // kWarpTileSeqLenP = 1
    const float rescale_factor_0 = __frcp_rn(lane_block_row_sum_old[0]);
    const float rescale_factor_1 = __frcp_rn(lane_block_row_sum_old[1]);
    #pragma unroll
    for (int j = 0; j < kWarpTileHeadDimV; ++j) { // 8, 16, 32, ...
      // Scaling in registers & convert F32 -> half for O collective store.
      if constexpr (kOStorageAccFloat32) {
        const float* t_fptr_D_0_1 = reinterpret_cast<float*>(R_D + j * 4); 
        half*  t_hptr_D_0_1 = reinterpret_cast< half*>(R_D + j * 4); 
        t_hptr_D_0_1[0] = __float2half_rn(rescale_factor_0 * t_fptr_D_0_1[0]);
        t_hptr_D_0_1[1] = __float2half_rn(rescale_factor_0 * t_fptr_D_0_1[1]);
        t_hptr_D_0_1[2] = __float2half_rn(rescale_factor_1 * t_fptr_D_0_1[2]);
        t_hptr_D_0_1[3] = __float2half_rn(rescale_factor_1 * t_fptr_D_0_1[3]);
      } else {
        half* t_hptr_D_0_1 = reinterpret_cast<half*>(R_D + j * 2); 
        t_hptr_D_0_1[0] = __float2half_rn(rescale_factor_0 * __half2float(t_hptr_D_0_1[0]));
        t_hptr_D_0_1[1] = __float2half_rn(rescale_factor_0 * __half2float(t_hptr_D_0_1[1]));
        t_hptr_D_0_1[2] = __float2half_rn(rescale_factor_1 * __half2float(t_hptr_D_0_1[2]));
        t_hptr_D_0_1[3] = __float2half_rn(rescale_factor_1 * __half2float(t_hptr_D_0_1[3]));
      }
    } // end for kWarpTileHeadDimV
  } // end for kWarpTileSeqLenP = 1
}


template<
  const int Br, 
  const int kHeadDim, 
  const int kMmaAtomM, 
  const int kMmaAtomN, 
  const int kWarpTileHeadDimV,
  const int kOStorageAccFloat32
>
__device__ __forceinline__ void sync_store_o_r2g(
  half * gmem_ptr,       // O gmem ptr
  const int gmem_offset, // O gmem global offset 
  const int n_tile_id,   // curr tile id (seqlen) O_tile_id
  const int mma_tile_id, // Q warp_QP 0~num MMAs, KV warp_KV 0
  uint32_t * R_D,        // Final scaled O
  uint32_t * R_Q,        // R_Q[1][4] for registers reuse
  uint32_t * R_K         // R_K[8][2] for registers reuse
) {
  // Store O(D): Write O[Br,d] from regs -> gmem, collective store 
  // with reg reuse & warp shuffle. 
  const int lane_id = threadIdx.x % WARP_SIZE; // 0~31
  // static_assert(kWarpTileSeqLenP == 1);
  { // kWarpTileSeqLenP = 1
    #pragma unroll
    for (int j = 0; j < kWarpTileHeadDimV; ++j) { // 8
      // reuse R_Q[1][4], R_K[8][2] for collective store.
      uint32_t* t_uptr_Z_0 = reinterpret_cast<uint32_t*>(R_Q); 
      uint32_t* t_uptr_Z_1 = reinterpret_cast<uint32_t*>(R_K); 
      const int offset = (kOStorageAccFloat32) ? j * 4 : j * 2;
      t_uptr_Z_0[0] = R_D[offset + 0]; 
      t_uptr_Z_1[0] = R_D[offset + 1]; 
      t_uptr_Z_0[1] = __shfl_sync((0xffffffff), R_D[offset + 0], lane_id + 1, 4);
      t_uptr_Z_0[2] = __shfl_sync((0xffffffff), R_D[offset + 0], lane_id + 2, 4);
      t_uptr_Z_0[3] = __shfl_sync((0xffffffff), R_D[offset + 0], lane_id + 3, 4);
      t_uptr_Z_1[1] = __shfl_sync((0xffffffff), R_D[offset + 1], lane_id + 1, 4);
      t_uptr_Z_1[2] = __shfl_sync((0xffffffff), R_D[offset + 1], lane_id + 2, 4);
      t_uptr_Z_1[3] = __shfl_sync((0xffffffff), R_D[offset + 1], lane_id + 3, 4);

      // st.global.v4 128 bits. [Br,d]
      if (lane_id % 4 == 0) {
        // (0/1)*32 + (0/1)*16=(0,16,32,48), + 0~7 -> 0~56 kWarpTileSeqLenP = 1
        // int store_warp_regs_O_Br = warp_QP * (kMmaAtomM * kWarpTileSeqLenP ) + 0 * kMmaAtomM;
        const int store_warp_regs_O_Br = mma_tile_id * (kMmaAtomM);
        const int store_lane_gmem_O_Br = n_tile_id * Br + store_warp_regs_O_Br + lane_id / 4; // 0~7
        // (0~3)*16 + (0/1)*8=(0,8,16,24,...,48,56)  warp_KV = 0
        // int store_warp_regs_O_d = warp_KV * (kMmaAtomN * kWarpTileHeadDimV) + j * kMmaAtomN;
        const int store_warp_regs_O_d = j * kMmaAtomN;
        const int store_lane_gmem_O_d = store_warp_regs_O_d; // (0~3)*16+(0/8)
        const int store_gmem_O_addr_0 = (
          gmem_offset + (store_lane_gmem_O_Br + 0) * kHeadDim + store_lane_gmem_O_d);
        const int store_gmem_O_addr_1 = (
          gmem_offset + (store_lane_gmem_O_Br + 8) * kHeadDim + store_lane_gmem_O_d);
        cp_async::stg_sync_128b(&gmem_ptr[store_gmem_O_addr_0], t_uptr_Z_0);
        cp_async::stg_sync_128b(&gmem_ptr[store_gmem_O_addr_1], t_uptr_Z_1);
      }
    } // end for kWarpTileHeadDimV
  } // kWarpTileSeqLenP = 1
}

} // prefill 
} // ffpa

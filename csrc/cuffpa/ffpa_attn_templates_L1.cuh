#pragma once
#include "cuffpa/prefill.cuh" // ffpa::prefill
using namespace ffpa;
using mma::MMAMode;                                      


template<
  const int kHeadDim,              // Headdim, 32~1024     
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
__global__ void __launch_bounds__(
  WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK) 
ffpa_mma_stages_split_q_L1_template(half* Q, 
                                    half* K, 
                                    half* V, 
                                    half* O, 
                                    int QKV_seqlen, 
                                    int QKV_head) {
  prefill::check_compiling_states<
    kHeadDim, kMmaAtomM, kMmaAtomN, kMmaAtomK, kMmaTileSeqLenQ, kMmaTileSeqLenK, 
    kMmaTileSeqLenP, kMmaTileHeadDimV, kWarpTileSeqLenQ, kWarpTileSeqLenK, 
    kWarpTileSeqLenP, kWarpTileHeadDimV, kMmaAccFloat32QK, kMmaAccFloat32PV,
    kOStorageAccFloat32, kPrefetchQK, kPrefetchPV, kShareSmemQKV, kPersistQs2r, 
    kPersistQg2s, kStageQK, kStagePV, kPadQ, kPadK, kPadV
  >();
  constexpr int Br = kMmaAtomM * kMmaTileSeqLenQ * kWarpTileSeqLenQ;
  constexpr int Bc = kMmaAtomN * kMmaTileSeqLenK * kWarpTileSeqLenK;
  constexpr int kNumThreads = WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK;
  
#ifdef ENBALE_FFPA_LAUNCH_GRID_DNHB
  // grid(div_ceil(QKV_seqlen, Br), QKV_head, QKV_batch), (x,y,z)
  const int QKV_batch_id = blockIdx.z;            // Batch size
  const int QKV_head_id  = blockIdx.y;            // Head num
#else
  // grid(div_ceil(QKV_seqlen, Br), QKV_batch * QKV_head), (x,y,z)
  const int QKV_batch_id = blockIdx.y / QKV_head; // Batch size
  const int QKV_head_id  = blockIdx.y % QKV_head; // Head num
#endif
  const int Q_tile_id    = blockIdx.x;            // Q tile_id, range [0, Tr]
  const int O_tile_id    = Q_tile_id;             // O tile_id, same as Q.
  const int tid          = threadIdx.x;           // within block
  const int warp_QP      = tid / WARP_SIZE;       // 0,1,2,3 or 0~7
  const int warp_KV      = 0;                     // 0
  const int Q_gmem_offset = ((QKV_batch_id * QKV_head * QKV_seqlen * kHeadDim) + 
                             (QKV_head_id * QKV_seqlen * kHeadDim)); // Q [seqlen,d]
  const int K_gmem_offset = ((QKV_batch_id * QKV_head * QKV_seqlen * kHeadDim) + 
                             (QKV_head_id * QKV_seqlen * kHeadDim)); // K [seqlen,d]                           
  const int V_gmem_offset = Q_gmem_offset; // V [seqlen,d]
  const int O_gmem_offset = Q_gmem_offset; // O [seqlen,d]

  // int load_gmem_Q_Br = Q_tile_id * Br + load_smem_Q_Br; 
  if ((Q_tile_id * Br + (tid / (kNumThreads / Br))) >= QKV_seqlen) return;

  extern __shared__ half smem[];
  constexpr int Q_tile_size = Br * (kMmaAtomK     + kPadQ); // Q[Br,16], 64*16*2=2048 bytes
  constexpr int K_tile_size = Bc * (kMmaAtomK     + kPadK); // K[Bc,16]
  constexpr int V_tile_size = Bc * (kMmaAtomN * 2 + kPadV); // V[Bc,16]
  half* Q_tile_smem = smem; 
  half* K_tile_smem = (
    Q_tile_smem + (kPersistQg2s ? ((kHeadDim / kMmaAtomK) * Q_tile_size) : 
                   (kStageQK * Q_tile_size))
  ); // kPersistQg2s -> e.g d=64, Q smem [4][Br][16] [tile_K_d][Br][16]
  // V may reuse all Q+K smem after Q@K^T.
  half* V_tile_smem = (kShareSmemQKV ? Q_tile_smem : 
                       K_tile_smem + kStageQK * K_tile_size); 
  uint32_t smem_Q_base_ptr = __cvta_generic_to_shared(Q_tile_smem);
  uint32_t smem_K_base_ptr = __cvta_generic_to_shared(K_tile_smem);
  uint32_t smem_V_base_ptr = __cvta_generic_to_shared(V_tile_smem);

  // load Q g2s at very beginning if kPersistQg2s is enabled.
  // Put Q g2s before registers init, enable overlap between kPersistQg2s 
  // and init states.
  if constexpr (kPersistQg2s) {
    #pragma unroll
    for (int tile_K_d = 0; tile_K_d < (kHeadDim / kMmaAtomK); ++tile_K_d) {
      prefill::cp_async_qkv_g2s<
        Br, Q_tile_size, kHeadDim, kMmaAtomK, kNumThreads, kPadQ>(
          smem_Q_base_ptr, Q, Q_gmem_offset, Q_tile_id, 
          tile_K_d, tile_K_d
      );
      cp_async::commit_group();
    }
  }

  // --------------------- Registers/SMEM for thread block -------------------------
  // block m_old, l_old, store in lane, use float to keep precision.
  float lane_block_row_max_old[kWarpTileSeqLenQ][2]; // [1][2]
  float lane_block_row_sum_old[kWarpTileSeqLenQ][2]; // [1][2]
  utils::fill_2D_regs<float, kWarpTileSeqLenQ, 2>(lane_block_row_max_old, -INFINITY);
  utils::fill_2D_regs<float, kWarpTileSeqLenQ, 2>(lane_block_row_sum_old, 0.0f);

  // ---------------------- Registers for S=Q@K^T/O=P@V ----------------------------
  // e.g, 64, !kPersistQs2r -> [1][4] 4 regs, kPersistQs2r -> [1][4*4] 16 regs.
  uint32_t R_Q[kWarpTileSeqLenQ][
    (kPersistQg2s) ? 4: ((kPersistQs2r) ? (kHeadDim / kMmaAtomK) * 4 : 4)]; 
  uint32_t R_K[kWarpTileSeqLenK][2]; // [8][2]
  uint32_t R_V[2]; // [2], S=Q@K, only use 2 32bits registers.
  // e.g [1][8][2], MMA Acc fp16; [1][8][4], MMA Acc fp32; 
  uint32_t R_S[kWarpTileSeqLenQ][kWarpTileSeqLenK][(kMmaAccFloat32QK) ? 4 : 2]; 
  uint32_t R_O[(kMmaAccFloat32PV) ? 4 : 2]; // registers for O=PV[Br,d]=P@V, [4 or 2]
  uint32_t R_D[kWarpTileSeqLenP][kWarpTileHeadDimV][(kOStorageAccFloat32) ? 4 : 2]; 
  utils::fill_3D_regs<uint32_t, kWarpTileSeqLenP, kWarpTileHeadDimV, 
                      ((kOStorageAccFloat32) ? 4 : 2)>(R_D, 0);
  // Now, N must be mutliples of Bc(32/64) for KV tiling across seqlen.
  const int Tc = utils::div_ceil(QKV_seqlen, Bc); // Tc K_tile[Bc,d]
  const float scale = 1.0f / sqrt((float) kHeadDim);
  
  // <loop over K seqlen>: for K^T[d,seqlen] with K^T_tile[d,Bc]
  #pragma unroll 1
  for (int tile_K_seqlen = 0; tile_K_seqlen < Tc; ++tile_K_seqlen) { 
    // TODO: process last tile_K_seqlen ? pad to multiple of Bc.
    if constexpr (kPrefetchQK) {
      if constexpr (kStageQK > 1) {
        if (tile_K_seqlen == 0) {
          #pragma unroll
          for (int stage = 0; stage < (kStageQK - 1); ++stage) {
            if constexpr (!kPersistQg2s) {
              prefill::cp_async_qkv_g2s<
                Br, Q_tile_size, kHeadDim, kMmaAtomK, kNumThreads, kPadQ>(
                  smem_Q_base_ptr, Q, Q_gmem_offset, Q_tile_id, stage, stage
              );
            }
            
            prefill::cp_async_qkv_g2s<
              Bc, K_tile_size, kHeadDim, kMmaAtomK, kNumThreads, kPadK>(
                smem_K_base_ptr, K, K_gmem_offset, tile_K_seqlen, stage, stage
            );
            cp_async::commit_group(); // pack QK as 1 group.
          } // end for stage
          cp_async::wait_group<(kStageQK - 2)>();
          __syncthreads(); 
        } else {
          cp_async::wait_group<(kStageQK - 2)>();
          __syncthreads(); 
        }
      }
    } else {
      if constexpr (kStageQK > 1) {
        #pragma unroll
        for (int stage = 0; stage < (kStageQK - 1); ++stage) {
          if constexpr (kPersistQs2r) {
            // We only load Q g2s and s2r once if kPersistQs2r is enabled.
            if (tile_K_seqlen == 0) {
              prefill::cp_async_qkv_g2s<
                Br, Q_tile_size, kHeadDim, kMmaAtomK, kNumThreads, kPadQ>(
                  smem_Q_base_ptr, Q, Q_gmem_offset, Q_tile_id, stage, stage
              );
            }
          } else {
            if constexpr (!kPersistQg2s) {
              prefill::cp_async_qkv_g2s<
                Br, Q_tile_size, kHeadDim, kMmaAtomK, kNumThreads, kPadQ>(
                  smem_Q_base_ptr, Q, Q_gmem_offset, Q_tile_id, stage, stage
              );
            }
          }
          
          prefill::cp_async_qkv_g2s<
            Bc, K_tile_size, kHeadDim, kMmaAtomK, kNumThreads, kPadK>(
              smem_K_base_ptr, K, K_gmem_offset, tile_K_seqlen, stage, stage
          );
          cp_async::commit_group(); // pack QK as 1 group.
        } // end for stage
        cp_async::wait_group<(kStageQK - 2)>();
        __syncthreads(); 
      } // end if kStageQK > 1
    }

    // !kShareSmemQKV: Prefetch V g2s before all Q@K^T iteration.
    if constexpr ((!kShareSmemQKV) && kPrefetchPV && kStagePV > 1) {
      // Prefetch V g2s before all Q@K^T iteration.
      if constexpr (kStagePV > 1) {
        #pragma unroll
        for (int stage = 0; stage < (kStagePV - 1); ++stage) {
          prefill::cp_async_qkv_g2s<
            Bc, V_tile_size, kHeadDim, kMmaAtomN * 2, kNumThreads, kPadV>(
              smem_V_base_ptr, V, V_gmem_offset, tile_K_seqlen, stage, stage
          );
          cp_async::commit_group();
        }
      }
    }

    // <loop over K d>: tile_K_d, kMmaAtomK = 16, K_tile_d[kMmaAtomK,Bc]
    utils::fill_3D_regs<uint32_t, kWarpTileSeqLenQ, kWarpTileSeqLenK, 
                        (kMmaAccFloat32QK) ? 4 : 2>(R_S, 0);
    #pragma unroll
    for (int tile_K_d = 0; tile_K_d < (kHeadDim / kMmaAtomK); ++tile_K_d) {
      const int smem_sel      = (tile_K_d) % kStageQK;   
      const int smem_sel_next = (tile_K_d + (kStageQK - 1)) % kStageQK;
      // QK g2s, kPersistQs2r or not.
      if constexpr (kPersistQs2r) {
        // We only load Q g2s and s2r once if kPersistQs2r is enabled.
        if (tile_K_seqlen == 0) {
          prefill::cp_async_qkv_g2s<
            Br, Q_tile_size, kHeadDim, kMmaAtomK, kNumThreads, kPadQ>(
              smem_Q_base_ptr, Q, Q_gmem_offset, Q_tile_id, 
              (kStageQK > 1) ? (tile_K_d + (kStageQK - 1)) : tile_K_d, 
              (kStageQK > 1) ? smem_sel_next : smem_sel
          );
        }
      } else {
        if constexpr (!kPersistQg2s) {
          prefill::cp_async_qkv_g2s<
            Br, Q_tile_size, kHeadDim, kMmaAtomK, kNumThreads, kPadQ>(
              smem_Q_base_ptr, Q, Q_gmem_offset, Q_tile_id, 
              (kStageQK > 1) ? (tile_K_d + (kStageQK - 1)) : tile_K_d, 
              (kStageQK > 1) ? smem_sel_next : smem_sel
          );
        }
      }
      
      prefill::cp_async_qkv_g2s<
        Bc, K_tile_size, kHeadDim, kMmaAtomK, kNumThreads, kPadK>(
          smem_K_base_ptr, K, K_gmem_offset, tile_K_seqlen, 
          (kStageQK > 1) ? (tile_K_d + (kStageQK - 1)) : tile_K_d, 
          (kStageQK > 1) ? smem_sel_next : smem_sel
      );
      cp_async::commit_group(); // pack QK as 1 group.

      if constexpr (kStageQK <= 1) {
        cp_async::wait_group<0>();
        __syncthreads(); 
      }

      // QK s2r
      static_assert(kWarpTileSeqLenQ == 1);
      {
        if constexpr (kPersistQs2r) {
          // We only load Q g2s and s2r once if kPersistQs2r is enabled.
          if (tile_K_seqlen == 0) {
            prefill::sync_fetch_qkv_frags_s2r<
              0, 4, Q_tile_size, kMmaAtomM, kMmaAtomN, kMmaAtomK, kPadQ>(
                smem_Q_base_ptr, &R_Q[0][tile_K_d * 4], warp_QP, 0, 0, smem_sel
            );
          }
        } else {
          if constexpr (!kPersistQg2s) {
            prefill::sync_fetch_qkv_frags_s2r<
              0, 4, Q_tile_size, kMmaAtomM, kMmaAtomN, kMmaAtomK, kPadQ>(
                smem_Q_base_ptr, &R_Q[0][0], warp_QP, 0, 0, smem_sel
            );
          } else {
            prefill::sync_fetch_qkv_frags_s2r<
              0, 4, Q_tile_size, kMmaAtomM, kMmaAtomN, kMmaAtomK, kPadQ>(
                smem_Q_base_ptr, &R_Q[0][0], warp_QP, 0, 0, tile_K_d
            );
          }
        }
      }

      #pragma unroll
      for (int j = 0; j < kWarpTileSeqLenK; ++j) {
        prefill::sync_fetch_qkv_frags_s2r<
          0, 2, K_tile_size, kMmaAtomM, kMmaAtomN, kMmaAtomK, kPadK>(
            smem_K_base_ptr, &R_K[j][0], warp_KV, j, 0, smem_sel
        );
      } 
      if constexpr (kStageQK < 2) {
        __syncthreads();
      }

      // kShareSmemQKV: Prefetch V g2s before last Q@K^T iteration.
      if constexpr ((kShareSmemQKV) && kPrefetchPV && kStagePV > 1) {
        if (tile_K_d == (kHeadDim / kMmaAtomK - 1)) {
          __syncthreads(); // wait all QK s2r ready
          // Prefetch V g2s before last Q@K^T iteration.
          if constexpr (kStagePV > 1) {
            #pragma unroll
            for (int stage = 0; stage < (kStagePV - 1); ++stage) {
              prefill::cp_async_qkv_g2s<
                Bc, V_tile_size, kHeadDim, kMmaAtomN * 2, kNumThreads, kPadV>(
                  smem_V_base_ptr, V, V_gmem_offset, tile_K_seqlen, stage, stage
              );
              cp_async::commit_group();
            }
          }
        }
      } // end kPrefetchQKV
      
      // Q@K^T MMA compute
      static_assert(kWarpTileSeqLenQ == 1);
      { // kWarpTileSeqLenQ = 1
        const int q_offset = (kPersistQs2r) ? (tile_K_d << 2) : 0; // (tile_K_d * 4)
        #pragma unroll
        for (int j = 0; j < kWarpTileSeqLenK; ++j) {
          if constexpr (kMmaAccFloat32QK) {
            mma::m16n8k16_f16f16f32<MMAMode::kInplaceUpdate>(
              &R_S[0][j][0], &R_S[0][j][1], &R_S[0][j][2], &R_S[0][j][3],
              &R_Q[0][q_offset + 0],    &R_Q[0][q_offset + 1],    
              &R_Q[0][q_offset + 2],    &R_Q[0][q_offset + 3], 
              &R_K[j][0],    &R_K[j][1]
            );
          } else {
            mma::m16n8k16_f16f16f16<MMAMode::kInplaceUpdate>(
              &R_S[0][j][0], &R_S[0][j][1],
              &R_Q[0][q_offset + 0],    &R_Q[0][q_offset + 1],    
              &R_Q[0][q_offset + 2],    &R_Q[0][q_offset + 3], 
              &R_K[j][0],    &R_K[j][1]
            );
          }
        }
      }

      if constexpr (kStageQK > 1) {
        if (tile_K_d < (kHeadDim / kMmaAtomK - 1)) {
          cp_async::wait_group<(kStageQK - 2)>();
          __syncthreads(); 
        }
      }
    } // end loop over d, S=Q@K^T
    __syncthreads();

    // Prefetch V g2s before row max/sum for P@V if kStagePV > 1
    static_assert(kWarpTileSeqLenP == 1);
    if constexpr (!kPrefetchPV) {
      if constexpr (kStagePV > 1) {
        #pragma unroll
        for (int stage = 0; stage < (kStagePV - 1); ++stage) {
          prefill::cp_async_qkv_g2s<
            Bc, V_tile_size, kHeadDim, kMmaAtomN * 2, kNumThreads, kPadV>(
              smem_V_base_ptr, V, V_gmem_offset, tile_K_seqlen, stage, stage
          );
          cp_async::commit_group();
        }
      }
    }

    // Online safe softmax, warp/block reduce max/sum, row wise
    float lane_row_max_new[kWarpTileSeqLenQ][2]; // [1][2]
    float lane_row_sum_new[kWarpTileSeqLenQ][2]; // [1][2]
    utils::fill_2D_regs<float, kWarpTileSeqLenQ, 2>(lane_row_max_new, -INFINITY);
    utils::fill_2D_regs<float, kWarpTileSeqLenQ, 2>(lane_row_sum_new, 0.0f);

    static_assert(kWarpTileSeqLenQ == 1);
    prefill::sync_online_safe_softmax<kWarpTileSeqLenK, kMmaAccFloat32QK>(
      &R_S[0][0][0], scale, &lane_row_max_new[0][0], &lane_row_sum_new[0][0],
      &lane_block_row_max_old[0][0], &lane_block_row_sum_old[0][0]
    );

    // Wait V g2s stages ready.
    if constexpr (kStagePV > 1) {
      cp_async::wait_group<(kStagePV - 2)>(); // s2->0, s3->1, s4->2
      __syncthreads(); 
    }

    // !kShareSmemQKV: Prefetch QK g2s before all P@V.
    if constexpr ((!kShareSmemQKV) && kPrefetchQK && kStageQK > 1) {
      if ((tile_K_seqlen + 1) < Tc) {
        #pragma unroll
        for (int stage = 0; stage < (kStageQK - 1); ++stage) {
          // We do not need to load Q g2s & s2r while tile_K_seqlen > 0, 
          // if kPersistQs2r is enabled.
          if constexpr (!kPersistQs2r) {
            prefill::cp_async_qkv_g2s<
              Br, Q_tile_size, kHeadDim, kMmaAtomK, kNumThreads, kPadQ>(
                smem_Q_base_ptr, Q, Q_gmem_offset, Q_tile_id, stage, stage
            );
          }

          prefill::cp_async_qkv_g2s<
            Bc, K_tile_size, kHeadDim, kMmaAtomK, kNumThreads, kPadK>(
              smem_K_base_ptr, K, K_gmem_offset, tile_K_seqlen + 1, stage, stage
          );
          cp_async::commit_group(); // pack QK as 1 group.
        } // end for stage
      }
    }
    
    static_assert(kWarpTileSeqLenP == 1);
    {
      float rescale_o_factor_0[1];
      float rescale_o_factor_1[1];
      prefill::sync_precompute_rescale_factors(
        &lane_row_max_new[0][0], &lane_block_row_max_old[0][0], 
        &rescale_o_factor_0[0], &rescale_o_factor_1[0], tile_K_seqlen
      );

      // <HGEMM in registers>
      #pragma unroll
      for (int j = 0; j < kWarpTileHeadDimV; ++j) { // 8, 16, 32, ...
        // Compute d tile, P[Br,Bc]@V[Bc,16] = O[Br,16]
        const int tile_V_d = (j >> 1); // (j / 2)
        const int smem_sel_v = (tile_V_d) % kStagePV;   
        const int smem_sel_v_next = (tile_V_d + (kStagePV - 1)) % kStagePV;
        // V g2s, V tile smem [Bc,kMmaAtomN*2]=[64,16]
        if (j % 2 == 0) { // 0,2,4,6,...// curr K tile g2s
          prefill::cp_async_qkv_g2s<
            Bc, V_tile_size, kHeadDim, kMmaAtomN * 2, kNumThreads, kPadV>(
              smem_V_base_ptr, V, V_gmem_offset, tile_K_seqlen, 
              (kStagePV > 1) ? (tile_V_d + (kStagePV - 1)) : tile_V_d, 
              (kStagePV > 1) ? smem_sel_v_next : smem_sel_v
          );
          cp_async::commit_group();
          if constexpr (kStagePV <= 1) {
            cp_async::wait_group<0>();
            __syncthreads(); 
          }
        }

        utils::fill_1D_regs<uint32_t, (kMmaAccFloat32PV) ? 4 : 2>(R_O, 0); 
        #pragma unroll
        for (int tile_V_Bc = 0; tile_V_Bc < (Bc / kMmaAtomK); ++tile_V_Bc) {
          prefill::sync_fetch_qkv_frags_s2r<
            1, 2, V_tile_size, kMmaAtomM, kMmaAtomN, kMmaAtomK, kPadV>(
              smem_V_base_ptr, &R_V[0], warp_KV, (j % 2), tile_V_Bc, 
              smem_sel_v
          );

          // kShareSmemQKV: Prefetch next QK g2s before last P@V iteration.
          if constexpr ((kShareSmemQKV) && kPrefetchQK && kStageQK > 1) {
            if (j == (kWarpTileHeadDimV - 1) && tile_V_Bc == (Bc / kMmaAtomK - 1) 
                && (tile_K_seqlen + 1) < Tc) {
              __syncthreads(); // wait all V s2r ready
              // Prefetch next QK g2s before last P@V iteration.
              if constexpr (kStageQK > 1) {
                #pragma unroll
                for (int stage = 0; stage < (kStageQK - 1); ++stage) {
                  // We do not need to load Q g2s & s2r while tile_K_seqlen > 0, 
                  // if kPersistQs2r is enabled.
                  if constexpr (!kPersistQs2r) {
                    prefill::cp_async_qkv_g2s<
                      Br, Q_tile_size, kHeadDim, kMmaAtomK, kNumThreads, kPadQ>(
                        smem_Q_base_ptr, Q, Q_gmem_offset, Q_tile_id, stage, stage
                    );
                  }

                  prefill::cp_async_qkv_g2s<
                    Bc, K_tile_size, kHeadDim, kMmaAtomK, kNumThreads, kPadK>(
                      smem_K_base_ptr, K, K_gmem_offset, tile_K_seqlen + 1, stage, stage
                  );
                  cp_async::commit_group(); // pack QK as 1 group.
                } // end for stage
              }
            }
          } // end if kPrefetchQKV && kStageQK > 1

          // Compute P[Br,Bc]@V[Bc,d] = O[Br,d] 
          const int p_offset = tile_V_Bc * 2; // MMA(Warp) selected, 0, 2, 4, 6
          if constexpr (kMmaAccFloat32PV) {
            // MMA accumulate with F32 dtype for high precision.
            mma::m16n8k16_f16f16f32<MMAMode::kInplaceUpdate>(
              &R_O[0], &R_O[1], &R_O[2], &R_O[3],
              &R_S[0][p_offset][0],      &R_S[0][p_offset][1], 
              &R_S[0][p_offset + 1][0],  &R_S[0][p_offset + 1][1], 
              &R_V[0], &R_V[1]
            ); 
          } else {
            // MMA accumulate with F16 dtype for high throughput.
            mma::m16n8k16_f16f16f16<MMAMode::kInplaceUpdate>(
              &R_O[0], &R_O[1],
              &R_S[0][p_offset][0],      &R_S[0][p_offset][1], 
              &R_S[0][p_offset + 1][0],  &R_S[0][p_offset + 1][1], 
              &R_V[0], &R_V[1]
            ); 
          }
        } // end for V Bc.
        if constexpr (kStagePV < 2) {
          // Wait curr P@V tile ready if kStage < 2 in order to avoid 
          // the next V tile g2s overwrite.
          __syncthreads();
        }

        // Now, we get [Br,8] slice of [Br,d], each warp(MMA) contains m16n8.
        // 0. Rescale O: Online rescaling O each tile_K_seqlen step, need m_new, m_old.
        // m = max(m_old, m_new), O_new[Br,d] = exp(m_old - m) * O_old + P@V
        prefill::sync_rescaling_tiling_o<kOStorageAccFloat32, kMmaAccFloat32PV>(
          &R_D[0][0][0], &R_O[0], &rescale_o_factor_0[0], 
          &rescale_o_factor_1[0], tile_K_seqlen, j
        );

        if constexpr (kStagePV > 1) {
          // Wait next V tile g2s ready.
          if (j < (kWarpTileHeadDimV - 1)) {
            cp_async::wait_group<(kStagePV - 2)>();
            __syncthreads();
          }
        }
      } // end for kWarpTileHeadDimV (end P@V)

      // Now, we can update m, l after O has been scaled.
      prefill::sync_update_max_expsum(
        &lane_row_max_new[0][0], &lane_row_sum_new[0][0],
        &lane_block_row_max_old[0][0], &lane_block_row_sum_old[0][0],
        &rescale_o_factor_0[0], &rescale_o_factor_1[0], tile_K_seqlen
      );
    } // end P@V
    __syncthreads(); 

  } // end loop over N
  __syncthreads();

  // Finaly, we still have to rescale O once more.
  // O_output(D) = ( 1/l_final ) * O_final (FA2 paper)
  static_assert(kWarpTileSeqLenP == 1);
  prefill::sync_rescaling_final_o<kWarpTileHeadDimV, kOStorageAccFloat32>(
    &R_D[0][0][0], &lane_block_row_sum_old[0][0]
  );

  // Store O(D): Write O[Br,d] from regs -> gmem, collective store 
  // with reg reuse & warp shuffle. 
  static_assert(kWarpTileSeqLenP == 1);
  prefill::sync_store_o_r2g<
    Br, kHeadDim, kMmaAtomM, kMmaAtomN, kWarpTileHeadDimV, kOStorageAccFloat32>(
      O, O_gmem_offset, O_tile_id, warp_QP, &R_D[0][0][0], &R_Q[0][0], &R_K[0][0]
  );
}

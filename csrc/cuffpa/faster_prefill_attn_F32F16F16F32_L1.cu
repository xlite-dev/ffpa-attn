#include "cuffpa/prefill.cuh" // prefill
using namespace ffpa;  


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
  const int kOStorageAccFloat32,   // 0/1, MMA Acc always be fp16, but O storage can be fp32 or half.
  const int kStage,                // 1,2
  const int kPadQ,                 // Pad Q/K/V 0,8; 0 -> smem swizzle
  const int kPadK,             
  const int kPadV             
>
__global__ void __launch_bounds__(
  WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK) 
ffpa_mma_stages_split_q_acc_f32_L1_kernel(
  half* Q, 
  half* K, 
  half* V, 
  half* O, 
  int QKV_seqlen,
  int QKV_head
) {
  // Matmul Layout: Q[Br,d]@K^T[d,Bc] NT, P[Br,Bc]@V[Bc,d] NN.
  // NOTE: K[Bc,d] with row major means K^T[d,Bc] in col major.
  static_assert(kMmaAtomM == 16 && kMmaAtomN == 8 && kMmaAtomK == 16); // m16n8k16
  static_assert(kMmaTileSeqLenQ  <= 8 && kMmaTileSeqLenK  == 1);  // Q@K^T
  static_assert(kMmaTileSeqLenP  <= 8 && kMmaTileHeadDimV == 1);  // P@V
  static_assert(kWarpTileSeqLenQ == 1 && kWarpTileSeqLenK <= 16); // Q@K^T
  static_assert(kWarpTileSeqLenP == 1 && kWarpTileHeadDimV == (
    kHeadDim / (kMmaAtomN * kMmaTileHeadDimV))); // P@V
  static_assert(kOStorageAccFloat32 == 0 || kOStorageAccFloat32 == 1);
  static_assert(kStage < 5 && kStage > 0); 
  static_assert(kPadQ >= 0 && kPadQ % 8 == 0); // 0,8,16
  static_assert(kPadK >= 0 && kPadK % 8 == 0); // 0,8,16
  static_assert(kPadV >= 0 && kPadV % 8 == 0); // 0,8,16
  constexpr int Br = kMmaAtomM * kMmaTileSeqLenQ * kWarpTileSeqLenQ; // 16*4*1=64
  constexpr int Bc = kMmaAtomN * kMmaTileSeqLenK * kWarpTileSeqLenK; //  8*1*8=64
  static_assert(Br >= Bc); // for shared memory reuse.
  constexpr int kNumThreads = WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK; // 32*4*1=128, num threads
  const int Tc = utils::div_ceil(QKV_seqlen, Bc); // Tc K_tile[Bc,d]
  const float scale = 1.0f / sqrt((float) kHeadDim);
  
  const int QKV_batch_id = blockIdx.y / QKV_head; // Batch size
  const int QKV_head_id  = blockIdx.y % QKV_head; // Head num
  const int Q_tile_id    = blockIdx.x;            // Q tile_id, range [0, Tr]
  const int O_tile_id    = Q_tile_id;             // O tile_id, same as Q.
  const int tid          = threadIdx.x;           // within block
  const int warp_id      = tid / WARP_SIZE;       // 0~7 warp_id within block
  const int lane_id      = tid % WARP_SIZE;       // 0~31
  const int warp_QP      = warp_id;               // 0,1,2,3 or 0~7
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
  half* K_tile_smem = Q_tile_smem + kStage * Q_tile_size;
  half* V_tile_smem = Q_tile_smem; // V may reuse all Q+K smem after Q@K^T.
  uint32_t smem_Q_base_ptr = __cvta_generic_to_shared(Q_tile_smem);
  uint32_t smem_K_base_ptr = __cvta_generic_to_shared(K_tile_smem);
  uint32_t smem_V_base_ptr = __cvta_generic_to_shared(V_tile_smem);

  // --------------------- Registers/SMEM for thread block -------------------------
  // block m_old, l_old, store in lane, use float to keep precision.
  float lane_block_row_max_old[kWarpTileSeqLenQ][2]; // [1][2]
  float lane_block_row_sum_old[kWarpTileSeqLenQ][2]; // [1][2]
  utils::fill_2D_regs<float, kWarpTileSeqLenQ, 2>(lane_block_row_max_old, -INFINITY);
  utils::fill_2D_regs<float, kWarpTileSeqLenQ, 2>(lane_block_row_sum_old, 0.0f);

  // ---------------------- Registers for S=Q@K^T/O=P@V ----------------------------
  uint32_t R_Q[kWarpTileSeqLenQ][ 4]; // [1][4]
  uint32_t R_K[kWarpTileSeqLenK][ 2]; // [8][2]
  uint32_t R_V[2]; // [2], S=Q@K, only use 2 32bits registers.
  uint32_t R_S[kWarpTileSeqLenQ][kWarpTileSeqLenK][ 4]; // [1][8][4], acc f32.
  uint32_t R_O[4]; // registers for O=PV[Br,d]=P@V, [4], only use 4 32bits registers.
  uint32_t R_D[kWarpTileSeqLenP][kWarpTileHeadDimV][(kOStorageAccFloat32) ? 4 : 2]; 
  utils::fill_3D_regs<uint32_t, kWarpTileSeqLenP, kWarpTileHeadDimV, 
                      ((kOStorageAccFloat32) ? 4 : 2)>(R_D, 0);
  
  // <loop over K seqlen>: for K^T[d,seqlen] with K^T_tile[d,Bc]
  #pragma unroll 1
  for (int tile_K_seqlen = 0; tile_K_seqlen < Tc; ++tile_K_seqlen) { 
    // TODO: process last tile_K_seqlen ? pad to multiple of Bc.
    if constexpr (kStage > 1) {
      #pragma unroll
      for (int stage = 0; stage < (kStage - 1); ++stage) {
        prefill::cp_async_qkv_g2s<
          Br, Q_tile_size, kHeadDim, kMmaAtomK, kNumThreads, kPadQ>(
            smem_Q_base_ptr, Q, Q_gmem_offset, Q_tile_id, stage, stage
        );

        prefill::cp_async_qkv_g2s<
          Bc, K_tile_size, kHeadDim, kMmaAtomK, kNumThreads, kPadK>(
            smem_K_base_ptr, K, K_gmem_offset, tile_K_seqlen, stage, stage
        );
      } // end for stage
      cp_async::wait_group<(kStage - 2)>();
      __syncthreads(); 
    } // end if kStage > 1

    // <loop over K d>: tile_K_d, kMmaAtomK = 16, K_tile_d[kMmaAtomK,Bc]
    utils::fill_3D_regs<uint32_t, kWarpTileSeqLenQ, kWarpTileSeqLenK, 4>(R_S, 0);
    #pragma unroll
    for (int tile_K_d = 0; tile_K_d < (kHeadDim / kMmaAtomK); ++tile_K_d) {
      const int smem_sel      = (tile_K_d) % kStage;   
      const int smem_sel_next = (tile_K_d + (kStage - 1)) % kStage;

      prefill::cp_async_qkv_g2s<
        Br, Q_tile_size, kHeadDim, kMmaAtomK, kNumThreads, kPadQ>(
          smem_Q_base_ptr, Q, Q_gmem_offset, Q_tile_id, 
          (kStage > 1) ? (tile_K_d + 1) : tile_K_d, 
          (kStage > 1) ? smem_sel_next : smem_sel
      );

      prefill::cp_async_qkv_g2s<
        Bc, K_tile_size, kHeadDim, kMmaAtomK, kNumThreads, kPadK>(
          smem_K_base_ptr, K, K_gmem_offset, tile_K_seqlen, 
          (kStage > 1) ? (tile_K_d + 1) : tile_K_d, 
          (kStage > 1) ? smem_sel_next : smem_sel
      );
      if constexpr (kStage <= 1) {
        cp_async::wait_group<0>();
        __syncthreads(); 
      }

      // Q s2r
      static_assert(kWarpTileSeqLenQ == 1);
      {
        prefill::sync_fetch_qkv_frags<
          0, 4, Q_tile_size, kMmaAtomM, kMmaAtomN, kMmaAtomK, kPadQ>(
            smem_Q_base_ptr, &R_Q[0][0], warp_QP, 0, 0, smem_sel
        );
      }

      #pragma unroll
      for (int j = 0; j < kWarpTileSeqLenK; ++j) {
        prefill::sync_fetch_qkv_frags<
          0, 2, K_tile_size, kMmaAtomM, kMmaAtomN, kMmaAtomK, kPadK>(
            smem_K_base_ptr, &R_K[j][0], warp_KV, j, 0, smem_sel
        );
      } 
      if constexpr (kStage < 2) {
        __syncthreads();
      }
      
      // MMA compute
      static_assert(kWarpTileSeqLenQ == 1);
      { // kWarpTileSeqLenQ = 1
        #pragma unroll
        for (int j = 0; j < kWarpTileSeqLenK; ++j) { // 8, 16, 32, ...
          // MMA always accumulate with F32 dtype for high precision.
          mma::m16n8k16_f16f16f32(
            &R_S[0][j][0], &R_S[0][j][1], &R_S[0][j][2], &R_S[0][j][3],
            &R_Q[0][0],    &R_Q[0][1],    &R_Q[0][2],    &R_Q[0][3], 
            &R_K[j][0],    &R_K[j][1], 
            &R_S[0][j][0], &R_S[0][j][1], &R_S[0][j][2], &R_S[0][j][3]
          );                      
        }
      }

      if constexpr (kStage > 1) {
        // Wait next Q, K tile g2s ready.
        cp_async::wait_group<(kStage - 2)>();
        __syncthreads(); 
      }
    } // end loop over d, S=Q@K^T
    __syncthreads();

    // Prefetch V g2s before row max/sum for P@V if kStage > 1
    static_assert(kWarpTileSeqLenP == 1);
    {
      if constexpr (kStage > 1) {
        #pragma unroll
        for (int stage = 0; stage < (kStage - 1); ++stage) {
          prefill::cp_async_qkv_g2s<
            Bc, V_tile_size, kHeadDim, kMmaAtomN * 2, kNumThreads, kPadV>(
              smem_V_base_ptr, V, V_gmem_offset, tile_K_seqlen, stage, stage
          );
        }
      }
    }

    // Online safe softmax, warp/block reduce max/sum, row wise
    float lane_row_max_new[kWarpTileSeqLenQ][2]; // [1][2]
    float lane_row_sum_new[kWarpTileSeqLenQ][2]; // [1][2]
    utils::fill_2D_regs<float, kWarpTileSeqLenQ, 2>(lane_row_max_new, -INFINITY);
    utils::fill_2D_regs<float, kWarpTileSeqLenQ, 2>(lane_row_sum_new, 0.0f);

    static_assert(kWarpTileSeqLenQ == 1);
    // Row max for [Br,Bc] tile, Thread -> Warp -> Block.
    { // kWarpTileSeqLenQ = 1
      // Thread level reduce max across kWarpTileSeqLenK dim, namely Bc.
      #pragma unroll
      for (int j = 0; j < kWarpTileSeqLenK; ++j) {
        float* t_fptr_S_0_1 = reinterpret_cast<float*>(&(R_S[0][j][0])); 
        // This should be the row max after S = (Q @ K^T) / sqrt(d)
        float tmp_max_0 = max(t_fptr_S_0_1[0], t_fptr_S_0_1[1]) * scale;
        float tmp_max_1 = max(t_fptr_S_0_1[2], t_fptr_S_0_1[3]) * scale;
        lane_row_max_new[0][0] = max(lane_row_max_new[0][0], tmp_max_0);
        lane_row_max_new[0][1] = max(lane_row_max_new[0][1], tmp_max_1);
      } // end for kWarpTileSeqLenK

      // Warp level reduce max, warp_size = 4
      // Each thread contains the maximum of 2 rows of Br, 
      // and only the values of T0, T4, ..., T28 are used.
      lane_row_max_new[0][0] = warp::reduce_max<float, 4>(lane_row_max_new[0][0]);
      lane_row_max_new[0][1] = warp::reduce_max<float, 4>(lane_row_max_new[0][1]);
    } // end for kWarpTileSeqLenQ

    static_assert(kWarpTileSeqLenQ == 1);
    // Exp sum and mul scale_factor for [Br,Bc] tile, Thread -> Warp -> Block.
    { // kWarpTileSeqLenQ = 1
      // Use latest global row max without update.
      // Br 0, row_id, 0~7,  16~23, 32~39, 48~55; 
      float block_row_max_new_0 = lane_row_max_new[0][0]; 
      // Br 1, row_id, 8~15, 24~31, 40~47, 56~63;
      float block_row_max_new_1 = lane_row_max_new[0][1];
    
      float block_row_max_old_0 = lane_block_row_max_old[0][0];
      float block_row_max_old_1 = lane_block_row_max_old[0][1];
      // Apply m_new = max(m_old, m_new) here.
      block_row_max_new_0 = max(block_row_max_old_0, block_row_max_new_0);
      block_row_max_new_1 = max(block_row_max_old_1, block_row_max_new_1);

      #pragma unroll
      for (int j = 0; j < kWarpTileSeqLenK; ++j) {
        // R_S[][][4] 4 32bit registers with each contains 1 F32 element.
        // (x,y) 0~7->{c0, c1}, (z,w)->8~15 {c2, c3}
        float* t_fptr_S_0_1 = reinterpret_cast<float*>(&(R_S[0][j][0])); 
        half*  t_hptr_S_0_1 = reinterpret_cast< half*>(&(R_S[0][j][0])); 
        // P = Exp(S - m_new), fmaf(x, y, z) = x * y + z in registers;
        t_fptr_S_0_1[0] = __expf(__fmaf_rn(t_fptr_S_0_1[0], scale, - block_row_max_new_0));
        t_fptr_S_0_1[1] = __expf(__fmaf_rn(t_fptr_S_0_1[1], scale, - block_row_max_new_0));
        t_fptr_S_0_1[2] = __expf(__fmaf_rn(t_fptr_S_0_1[2], scale, - block_row_max_new_1));
        t_fptr_S_0_1[3] = __expf(__fmaf_rn(t_fptr_S_0_1[3], scale, - block_row_max_new_1));
        lane_row_sum_new[0][0] += (t_fptr_S_0_1[0] + t_fptr_S_0_1[1]);
        lane_row_sum_new[0][1] += (t_fptr_S_0_1[2] + t_fptr_S_0_1[3]);
        // Update R_S for P[Br,Bc] = Exp(S-m), point wise.
        // Also convert F32 -> half for P@V MMA, reuse R_S as P.
        t_hptr_S_0_1[0] = __float2half_rn(t_fptr_S_0_1[0]);
        t_hptr_S_0_1[1] = __float2half_rn(t_fptr_S_0_1[1]);
        t_hptr_S_0_1[2] = __float2half_rn(t_fptr_S_0_1[2]);
        t_hptr_S_0_1[3] = __float2half_rn(t_fptr_S_0_1[3]);
      } // end for kWarpTileSeqLenK

      // Warp level reduce sum, warp_size = 4
      lane_row_sum_new[0][0] = warp::reduce_sum<float, 4>(lane_row_sum_new[0][0]);
      lane_row_sum_new[0][1] = warp::reduce_sum<float, 4>(lane_row_sum_new[0][1]);
    }
    
    static_assert(kWarpTileSeqLenP == 1);
    {
      // <Prefetch max/sum values>
      // m = max(m_old, m_new), l = exp(m_old - m) * l_old + l_new (FA2 paper)
      // Br 0, row_id, 0~7,  16~23, 32~39, 48~55; Br 1, row_id, 8~15, 24~31, 40~47, 56~63
      float block_row_max_new_0 = lane_row_max_new[0][0]; 
      float block_row_max_new_1 = lane_row_max_new[0][1];
      float block_row_sum_new_0 = lane_row_sum_new[0][0];
      float block_row_sum_new_1 = lane_row_sum_new[0][1];
        
      float block_row_max_old_0 = lane_block_row_max_old[0][0];
      float block_row_max_old_1 = lane_block_row_max_old[0][1];
      // NOTE: max(-inf, val) = val.
      block_row_max_new_0 = max(block_row_max_old_0, block_row_max_new_0);
      block_row_max_new_1 = max(block_row_max_old_1, block_row_max_new_1);   
      // Avoid inf value while using m_old for rescaling O.
      block_row_max_old_0 = (tile_K_seqlen > 0 ? block_row_max_old_0 : 
                                                 block_row_max_new_0);                                       
      block_row_max_old_1 = (tile_K_seqlen > 0 ? block_row_max_old_1 : 
                                                 block_row_max_new_1);  
      // rescale factor for O and l, exp(m_old - m) for curr tile [Br,d].
      float rescale_o_factor_0 = __expf(block_row_max_old_0 - block_row_max_new_0);
      float rescale_o_factor_1 = __expf(block_row_max_old_1 - block_row_max_new_1);
      
      // Wait V g2s stages ready.
      if constexpr (kStage > 1) {
        cp_async::wait_group<(kStage - 2)>(); // s2->0, s3->1, s4->2
        __syncthreads(); 
      }

      // <HGEMM in registers>
      #pragma unroll
      for (int j = 0; j < kWarpTileHeadDimV; ++j) { // 8, 16, 32, ...
        // Compute d tile, P[Br,Bc]@V[Bc,16] = O[Br,16]
        const int tile_V_d = (j >> 1); // (j / 2)
        const int smem_sel_v = (tile_V_d) % kStage;   
        const int smem_sel_v_next = (tile_V_d + (kStage - 1)) % kStage;
        // V g2s, V tile smem [Bc,kMmaAtomN*2]=[64,16]
        if (j % 2 == 0) { // 0,2,4,6,...// curr K tile g2s
          prefill::cp_async_qkv_g2s<
            Bc, V_tile_size, kHeadDim, kMmaAtomN * 2, kNumThreads, kPadV>(
              smem_V_base_ptr, V, V_gmem_offset, tile_K_seqlen, 
              (kStage > 1) ? (tile_V_d + 1) : tile_V_d, 
              (kStage > 1) ? smem_sel_v_next : smem_sel_v
          );
          if constexpr (kStage <= 1) {
            cp_async::wait_group<0>();
            __syncthreads(); 
          }
        }

        utils::fill_1D_regs<uint32_t, 4>(R_O, 0); // must clear 
        #pragma unroll
        for (int tile_V_Bc = 0; tile_V_Bc < (Bc / kMmaAtomK); ++tile_V_Bc) {
          prefill::sync_fetch_qkv_frags<
            1, 2, V_tile_size, kMmaAtomM, kMmaAtomN, kMmaAtomK, kPadV>(
              smem_V_base_ptr, &R_V[0], warp_KV, (j % 2), tile_V_Bc, 
              smem_sel_v
          );
          // Compute P[Br,Bc]@V[Bc,d] = O[Br,d] 
          const int w = tile_V_Bc << 1; // MMA(Warp) selected, 0, 2, 4, 6
          // MMA always accumulate with F32 dtype for high precision.
          mma::m16n8k16_f16f16f32(
            &R_O[0], &R_O[1], &R_O[2], &R_O[3],
            &R_S[0][w][0], &R_S[0][w][1], &R_S[0][w + 1][0],  &R_S[0][w + 1][1], 
            &R_V[0], &R_V[1],
            &R_O[0], &R_O[1], &R_O[2], &R_O[3]
          ); 
        } // end for V Bc.
        if constexpr (kStage < 2) {
          // Wait curr P@V tile ready if kStage < 2 in order to avoid 
          // the next V tile g2s overwrite.
          __syncthreads();
        }

        // Now, we get [Br,8] slice of [Br,d], each warp(MMA) contains m16n8.
        // 0. Rescale O: Online rescaling O each tile_K_seqlen step, need m_new, m_old.
        // m = max(m_old, m_new), O_new[Br,d] = exp(m_old - m) * O_old + P@V
        float* t_fptr_O_0_1 = reinterpret_cast<float*>(&(R_O[0])); 
        if constexpr (kOStorageAccFloat32) {
          // (x,y) 0~7->{c0, c1}, (z,w)->8~15 {c2, c3} kWarpTileSeqLenP=1
          float* t_fptr_D_0_1 = reinterpret_cast<float*>(&(R_D[0][j][0]));
          t_fptr_D_0_1[0] = __fmaf_rn(rescale_o_factor_0, t_fptr_D_0_1[0], t_fptr_O_0_1[0]);
          t_fptr_D_0_1[1] = __fmaf_rn(rescale_o_factor_0, t_fptr_D_0_1[1], t_fptr_O_0_1[1]);
          t_fptr_D_0_1[2] = __fmaf_rn(rescale_o_factor_1, t_fptr_D_0_1[2], t_fptr_O_0_1[2]);
          t_fptr_D_0_1[3] = __fmaf_rn(rescale_o_factor_1, t_fptr_D_0_1[3], t_fptr_O_0_1[3]);
        } else {
          half* t_hptr_D_0_1 = reinterpret_cast<half*>(&(R_D[0][j][0])); 
          t_hptr_D_0_1[0] = __float2half_rn(__fmaf_rn(
            rescale_o_factor_0, __half2float(t_hptr_D_0_1[0]), t_fptr_O_0_1[0]));
          t_hptr_D_0_1[1] = __float2half_rn(__fmaf_rn(
            rescale_o_factor_0, __half2float(t_hptr_D_0_1[1]), t_fptr_O_0_1[1]));
          t_hptr_D_0_1[2] = __float2half_rn(__fmaf_rn(
            rescale_o_factor_1, __half2float(t_hptr_D_0_1[2]), t_fptr_O_0_1[2]));
          t_hptr_D_0_1[3] = __float2half_rn(__fmaf_rn(
            rescale_o_factor_1, __half2float(t_hptr_D_0_1[3]), t_fptr_O_0_1[3]));
        } // end for tile_V_Bc
        if constexpr (kStage > 1) {
          // Wait next V tile g2s ready.
          cp_async::wait_group<(kStage - 2)>();
          __syncthreads();
        }
      } // end for kWarpTileHeadDimV. 
      // Now, we can update m, l after O has been scaled.
      // 1. First, update block row sum Exp for each lane which
      // need both m_new and m_old.
      float block_row_sum_old_0 = lane_block_row_sum_old[0][0];
      float block_row_sum_old_1 = lane_block_row_sum_old[0][1];
      // Update l = exp(m_old - m_new) * l_old + row_sum(P).
      lane_block_row_sum_old[0][0] = (__fmaf_rn(
        rescale_o_factor_0, block_row_sum_old_0, block_row_sum_new_0));
      lane_block_row_sum_old[0][1] = (__fmaf_rn(
        rescale_o_factor_1, block_row_sum_old_1, block_row_sum_new_1));
      // 2. Then, update block row max for each lane.
      lane_block_row_max_old[0][0] = block_row_max_new_0;
      lane_block_row_max_old[0][1] = block_row_max_new_1;
    } // end P@V
    __syncthreads(); 

  } // end loop over N
  __syncthreads();

  // Finaly, we still have to rescale O once more.
  // O_output(D) = ( 1/l_final ) * O_final (FA2 paper)
  static_assert(kWarpTileSeqLenP == 1);
  prefill::sync_recaling_final_o<kWarpTileHeadDimV, kOStorageAccFloat32>(
    &R_D[0][0][0], &lane_block_row_sum_old[0][0]
  );

  // Store O(D): Write O[Br,d] from regs -> gmem, collective store 
  // with reg reuse & warp shuffle. 
  static_assert(kWarpTileSeqLenP == 1);
  prefill::sync_store_o_r2g<
    Br, kHeadDim, kMmaAtomM, kMmaAtomN, kWarpTileHeadDimV>(
      O, O_gmem_offset, O_tile_id, warp_QP, &R_D[0][0][0], 
      &R_Q[0][0], &R_K[0][0]
  );
}

template<const int kHeadDim, const int kStage>
void launch_ffpa_mma_acc_f32_L1(torch::Tensor Q, 
                                torch::Tensor K, 
                                torch::Tensor V, 
                                torch::Tensor O) {
  // Q,K,V,O with [B, H, N, D] layout, B=batch, H=head, N=seqlen, D=dim
  // TODO: support BNHD layout, Q,K,V,O with [B, N, H, D] layout.
  // Now: fixed tile BrxBc=128x128 for d>= 128, 64x64 for d<128.
  constexpr int kMmaAtomM = 16;
  constexpr int kMmaAtomN = 8;
  constexpr int kMmaAtomK = 16;
  constexpr int kMmaTileSeqLenQ  = (kHeadDim < 128) ? 4 : 8;
  constexpr int kMmaTileSeqLenK  = 1;
  constexpr int kMmaTileSeqLenP  = (kHeadDim < 128) ? 4 : 8;
  constexpr int kMmaTileHeadDimV = 1;
  constexpr int kWarpTileSeqLenQ = 1;
  constexpr int kWarpTileSeqLenK = (kHeadDim < 128) ? 8 : 16;
  constexpr int kWarpTileSeqLenP = 1;
  constexpr int kWarpTileHeadDimV = (kHeadDim / (kMmaAtomN * kMmaTileHeadDimV)); 
  constexpr int Br = kMmaAtomM * kMmaTileSeqLenQ * kWarpTileSeqLenQ;
  constexpr int Bc = kMmaAtomN * kMmaTileSeqLenK * kWarpTileSeqLenK;
  constexpr int kNumThreads = WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK; 
  // 0 for smem swizzle, > 0 for smem padding.
  constexpr int kPadQ = 0;
  constexpr int kPadK = 0; 
  constexpr int kPadV = 8; // swizzle V seems can not get good performance.
  // 0/1, MMA Acc always be fp32, but O storage(R_D) can be fp32 or half.
  // FP16 can provide precision to approximately 3-4 decimal places. Thus, if the 
  // error does not exceed 1e-3, using FP16 storage is sufficient for most applications.
  constexpr int kOStorageAccFloat32 = (kHeadDim < 256) ? 1 : 0;
  
  // Calculate SRAM size needed per block, Q,K,V smem size, V shared the QK smem.
  constexpr int QK_smem_size = (kStage * (Br * (kMmaAtomK + kPadQ)) + 
                                kStage * (Bc * (kMmaAtomK + kPadK)));
  // R_D registers, s=2, d=64, 16 regs; d=128, 32 regs; 
  // d=256, 64 regs; d=512, 128 regs; d=1024, 256 regs;
  constexpr int V_smem_size  = (kStage * (Bc * (kMmaAtomN * 2 + kPadV))); 
  // try to let V reuse all Q+K smem after Q@K^T, reduce smem usage.
  constexpr int kQKVSmemMaxSize = (QK_smem_size > V_smem_size ? 
                                   QK_smem_size * 2 : V_smem_size * 2);

  const int QKV_batch  = Q.size(0); 
  const int QKV_head   = Q.size(1);
  const int QKV_seqlen = Q.size(2); // QKV_seqlen
  assert(QKV_seqlen % max(Br, Bc) == 0); // multiple of max(Br, Bc)
  
  // Tr(=N/Br), batch_size x num_heads
  dim3 grid(utils::div_ceil(QKV_seqlen, Br), QKV_batch * QKV_head); 
  dim3 block(kNumThreads); // 4/8 warps per block

  cudaFuncSetAttribute(
    ffpa_mma_stages_split_q_acc_f32_L1_kernel<
      kHeadDim, 
      kMmaAtomM, 
      kMmaAtomN, 
      kMmaAtomK, 
      kMmaTileSeqLenQ, 
      kMmaTileSeqLenK, 
      kMmaTileSeqLenP, 
      kMmaTileHeadDimV, 
      kWarpTileSeqLenQ, 
      kWarpTileSeqLenK, 
      kWarpTileSeqLenP, 
      kWarpTileHeadDimV, 
      kOStorageAccFloat32,
      kStage, 
      kPadQ,
      kPadK,
      kPadV
    >,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    kQKVSmemMaxSize
  );

  ffpa_mma_stages_split_q_acc_f32_L1_kernel<
    kHeadDim, 
    kMmaAtomM, 
    kMmaAtomN, 
    kMmaAtomK, 
    kMmaTileSeqLenQ,  
    kMmaTileSeqLenK,
    kMmaTileSeqLenP, 
    kMmaTileHeadDimV, 
    kWarpTileSeqLenQ, 
    kWarpTileSeqLenK, 
    kWarpTileSeqLenP, 
    kWarpTileHeadDimV, 
    kOStorageAccFloat32,
    kStage, 
    kPadQ,
    kPadK,
    kPadV
  ><<<grid, block, kQKVSmemMaxSize>>>(
    reinterpret_cast<half*>(Q.data_ptr()),
    reinterpret_cast<half*>(K.data_ptr()),
    reinterpret_cast<half*>(V.data_ptr()),
    reinterpret_cast<half*>(O.data_ptr()),
    QKV_seqlen,
    QKV_head
  );
}

void ffpa_mma_acc_f32_L1(torch::Tensor Q, 
                         torch::Tensor K, 
                         torch::Tensor V, 
                         torch::Tensor O, 
                         int stages) {
  CHECK_TORCH_TENSOR_DTYPE(Q, torch::kHalf) // Q [B,H,N,D]
  CHECK_TORCH_TENSOR_DTYPE(K, torch::kHalf) // K [B,H,N,D]
  CHECK_TORCH_TENSOR_DTYPE(V, torch::kHalf) // V [B,H,N,D]
  CHECK_TORCH_TENSOR_DTYPE(O, torch::kHalf) // O [B,H,N,D]
  const int d = Q.size(3); // B, H, N, d
  
  // dispatch headdim
#define CASE_LAUNCH_KERNEL_F32_L1(D, S)      \
  case D:                                    \
    launch_ffpa_mma_acc_f32_L1<(D), (S)>(    \
      Q, K, V, O);                           \
    break;

#ifdef ENABLE_FFPA_DEBUG
  // minimal kernels for debug mode
#define DISPATCH_KERNEL_F32_L1_HEADDIM(S)    \
  {                                          \
    switch (d)                               \
    {                                        \
      CASE_LAUNCH_KERNEL_F32_L1(320,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(512,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(1024, (S));  \
    default:                                 \
      throw std::runtime_error(              \
        "headdim not support!");             \
      break;                                 \
    }                                        \
  }

#else 
#ifdef ENBALE_FFPA_ALL_HEADDIM
  // multiple of 32
#define DISPATCH_KERNEL_F32_L1_HEADDIM(S)    \
  {                                          \
    switch (d)                               \
    {                                        \
      CASE_LAUNCH_KERNEL_F32_L1(32,   (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(64,   (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(96,   (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(128,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(160,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(192,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(224,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(256,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(288,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(320,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(352,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(384,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(416,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(448,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(480,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(512,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(544,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(576,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(608,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(640,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(672,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(704,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(736,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(768,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(800,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(832,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(864,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(896,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(928,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(960,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(992,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(1024, (S));  \
    default:                                 \
      throw std::runtime_error(              \
        "headdim not support!");             \
      break;                                 \
    }                                        \
  }
#else
  // multiple of 64
#define DISPATCH_KERNEL_F32_L1_HEADDIM(S)    \
  {                                          \
    switch (d)                               \
    {                                        \
      CASE_LAUNCH_KERNEL_F32_L1(256,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(320,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(384,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(448,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(512,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(576,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(640,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(704,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(768,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(832,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(896,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(960,  (S));  \
      CASE_LAUNCH_KERNEL_F32_L1(1024, (S));  \
    default:                                 \
      throw std::runtime_error(              \
        "headdim not support!");             \
      break;                                 \
    }                                        \
  }
#endif

#endif

#ifdef ENBALE_FFPA_ALL_STAGES
  // dispatch stages
  if (stages == 2) {
    DISPATCH_KERNEL_F32_L1_HEADDIM(2);
  } else if (stages == 3) {
    DISPATCH_KERNEL_F32_L1_HEADDIM(3);
  } else if (stages == 4) {
    DISPATCH_KERNEL_F32_L1_HEADDIM(4);
  } else {
    DISPATCH_KERNEL_F32_L1_HEADDIM(1);
  }
#else 
  // dispatch stages
  if (stages == 2) {
    DISPATCH_KERNEL_F32_L1_HEADDIM(2);
  } else {
    DISPATCH_KERNEL_F32_L1_HEADDIM(1);
  }
#endif

#undef CASE_LAUNCH_KERNEL_F32_L1
#undef DISPATCH_KERNEL_F32_L1_HEADDIM
}

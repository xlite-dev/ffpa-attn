#pragma once
#include "prefill.cuh"  // ffpa::prefill

// ============================================================================
// ffpa_attn_split_kv_decode_stage1_template, faster for Nq=1, pure GEMV dot.
// ----------------------------------------------------------------------------
// Decode-only stage1 kernel for the large-d CUDA path. Each block owns one
// `(batch, q_head, kv_split)` tuple and processes all query rows for that head
// (only valid for Nq < 16). The kernel keeps the KV split external and loops
// over head-dim internally, so the overall algorithm remains split-kv outside
// and split-d inside.
//
// The kernel writes a normalized partial output plus per-split LSE:
//   partial_out[b, hq, split, row, d] = acc[row, d] / l[row]
//   chunk_lse[b, hq, split, row] = m[row] + log(l[row])
//
//   kStage controls the K smem pipeline depth:
//     1 = no pipeline (K read from gmem directly)
//     2 = double-buffer cp.async
//     N = ring-buffer with N slots, prefetching N-1 rows ahead
// ============================================================================
template <typename kDataType, const int kHeadDim, const bool kUseGemv,
          const int kStage = 2>
__global__ void __launch_bounds__(((kHeadDim / 8 + WARP_SIZE - 1) / WARP_SIZE) *
                                  WARP_SIZE)
    ffpa_attn_split_kv_decode_stage1_template(
        const kDataType* __restrict__ Q, const kDataType* __restrict__ K,
        const kDataType* __restrict__ V, float* __restrict__ partial_out,
        float* __restrict__ chunk_lse, const int Nq, const int Nkv,
        const int Nh, const int Nh_kv, const float scale, const int num_splits,
        const int split_size, const int causal) {
  using Traits = ffpa::DtypeTraits<kDataType>;
  constexpr int kElemsPerThread = 8;
  static_assert(kHeadDim % kElemsPerThread == 0,
                "kHeadDim must be multiple of 8");
  // Round threads up to WARP_SIZE boundary so warp shuffles are well-defined.
  constexpr int kNumActiveThreads = kHeadDim / kElemsPerThread;
  constexpr int kNumThreads =
      ((kNumActiveThreads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
  constexpr int kNumWarps = kNumThreads / WARP_SIZE;
  static_assert(kUseGemv,
                "This kernel is designed for the kUseGemv=true case only");
  constexpr int kMaxRows = kUseGemv ? 1 : 16;
  constexpr int kColsPerThread = kElemsPerThread;

  const int split_id = blockIdx.x;
#ifdef ENABLE_FFPA_LAUNCH_GRID_DNHB
  const int Nb_id = blockIdx.z;
  const int Nh_id = blockIdx.y;
#else
  const int Nb_id = blockIdx.y / Nh;
  const int Nh_id = blockIdx.y % Nh;
#endif
  const int tid = threadIdx.x;
  const int lane = tid % WARP_SIZE;
  const int warp_id = tid / WARP_SIZE;
  const int group_size = Nh / Nh_kv;
  const int kv_head_idx = Nh_id / group_size;
  const int active_rows = kUseGemv ? 1 : min(Nq, 16);
  const int split_start = split_id * split_size;
  const int split_end = min(Nkv, split_start + split_size);
  const int kv_offset = Nkv - Nq;

  // K_tile: ring-buffered smem for K rows (cp.async pipeline, kStage slots).
  __shared__ __align__(16) kDataType K_tile[kStage][kHeadDim];
  // V_tile: single-buffered smem for V row (cp.async overlaps with QK+reduce).
  __shared__ __align__(16) kDataType V_tile[kHeadDim];
  __shared__ kDataType q_tile[kMaxRows][kHeadDim];
  __shared__ float smem_out[kHeadDim];
  __shared__ float warp_partials[kMaxRows][kNumWarps];
  __shared__ float row_scores[kMaxRows];
  __shared__ float row_m[kMaxRows];
  __shared__ float row_l[kMaxRows];
  __shared__ float row_p[kMaxRows];
  __shared__ float row_rescale[kMaxRows];

  const int q_base = ((Nb_id * Nh + Nh_id) * Nq) * kHeadDim;
  const int kv_base = ((Nb_id * Nh_kv + kv_head_idx) * Nkv) * kHeadDim;

  // Vector constants for cp.async row loads (8×half/bf16 = 128-bit).
  constexpr int kVecRow =
      16 / static_cast<int>(sizeof(kDataType));  // 8 for half/bf16
  static_assert(kHeadDim % kVecRow == 0,
                "kHeadDim must be multiple of kVecRow");
  constexpr int kNumVecRow = kHeadDim / kVecRow;

  // Prologue: issue K + V + Q async loads in one cp.async group.
  if constexpr (kStage > 1) {
    for (int s = 0; s < kStage - 1; ++s) {
      const int gk = split_start + s;
      if (gk >= split_end)
        break;
      const int slot = gk % kStage;
      for (int i = tid; i < kNumVecRow; i += kNumThreads) {
        uint32_t smem_ptr =
            __cvta_generic_to_shared(&K_tile[slot][i * kVecRow]);
        ffpa::cp_async::cp_async<16>(smem_ptr,
                                     &K[kv_base + gk * kHeadDim + i * kVecRow]);
      }
    }
  }
  // V preload.
  {
    for (int i = tid; i < kNumVecRow; i += kNumThreads) {
      uint32_t v_smem = __cvta_generic_to_shared(&V_tile[i * kVecRow]);
      ffpa::cp_async::cp_async<16>(
          v_smem, &V[kv_base + split_start * kHeadDim + i * kVecRow]);
    }
  }
  // Q load (async cp.async, overlaps with K+V).
  for (int row = 0; row < active_rows; ++row) {
    for (int i = tid; i < kNumVecRow; i += kNumThreads) {
      uint32_t q_smem = __cvta_generic_to_shared(&q_tile[row][i * kVecRow]);
      ffpa::cp_async::cp_async<16>(q_smem,
                                   &Q[q_base + row * kHeadDim + i * kVecRow]);
    }
  }
  // Single commit: K + V + Q all in one group.
  ffpa::cp_async::commit_group();

  if (tid < kMaxRows) {
    row_m[tid] = -INFINITY;
    row_l[tid] = 0.0f;
    row_p[tid] = 0.0f;
    row_rescale[tid] = 0.0f;
    row_scores[tid] = 0.0f;
  }
  // Wait for K+V+Q async loads to complete.
  ffpa::cp_async::wait_group<0>();
  __syncthreads();

  float acc[kMaxRows][kColsPerThread];
#pragma unroll
  for (int row = 0; row < kMaxRows; ++row) {
#pragma unroll
    for (int col = 0; col < kColsPerThread; ++col) {
      acc[row][col] = 0.0f;
    }
  }

  for (int global_k = split_start; global_k < split_end; ++global_k) {
    // Pipeline stage 1: issue V async (overlaps with QK dot product)
    if (global_k > split_start) {
      for (int i = tid; i < kNumVecRow; i += kNumThreads) {
        uint32_t v_smem = __cvta_generic_to_shared(&V_tile[i * kVecRow]);
        ffpa::cp_async::cp_async<16>(
            v_smem, &V[kv_base + global_k * kHeadDim + i * kVecRow]);
      }
      ffpa::cp_async::commit_group();  // → group 0 = V
    }

    float score_partial[kMaxRows];
#pragma unroll
    for (int row = 0; row < kMaxRows; ++row) {
      score_partial[row] = 0.0f;
    }

    // QK dot product (128-bit smem loads)
    for (int row = 0; row < active_rows; ++row) {
      float qk = 0.0f;
      if (tid < kNumActiveThreads) {
        if constexpr (kStage == 1) {
          // kStage==1: K from gmem (128-bit loads).
          const int d = tid * kElemsPerThread;
          uint4 qv, kv_g;
          ffpa::cp_async::ldg_sync_128b(&qv, &q_tile[row][d]);
          ffpa::cp_async::ldg_sync_128b(&kv_g,
                                        &K[kv_base + global_k * kHeadDim + d]);
          const kDataType* qh = reinterpret_cast<const kDataType*>(&qv);
          const kDataType* kh = reinterpret_cast<const kDataType*>(&kv_g);
          // TODO: Warp reduction here instead of register reduction for better
          // ILP.
#pragma unroll
          for (int v = 0; v < kElemsPerThread; ++v) {
            qk += Traits::to_float(qh[v]) * Traits::to_float(kh[v]);
          }
        } else {
          const int read_slot = global_k % kStage;
          const int d = tid * kElemsPerThread;
          uint4 qv, kv;
          ffpa::cp_async::ldg_sync_128b(&qv, &q_tile[row][d]);
          ffpa::cp_async::ldg_sync_128b(&kv, &K_tile[read_slot][d]);
          const kDataType* qh = reinterpret_cast<const kDataType*>(&qv);
          const kDataType* kh = reinterpret_cast<const kDataType*>(&kv);
#pragma unroll
          for (int v = 0; v < kElemsPerThread; ++v) {
            qk += Traits::to_float(qh[v]) * Traits::to_float(kh[v]);
          }
        }
        score_partial[row] = qk;
      }

      // Pipeline stage 2: issue K async (overlaps with reduce + softmax)
      if constexpr (kStage > 1) {
        const int prefetch_k = global_k + kStage - 1;
        if (prefetch_k < split_end) {
          const int slot = prefetch_k % kStage;
          for (int i = tid; i < kNumVecRow; i += kNumThreads) {
            uint32_t smem_ptr =
                __cvta_generic_to_shared(&K_tile[slot][i * kVecRow]);
            ffpa::cp_async::cp_async<16>(
                smem_ptr, &K[kv_base + prefetch_k * kHeadDim + i * kVecRow]);
          }
          ffpa::cp_async::commit_group();  // → group 0 = K, group 1 = V
        }
      }

      for (int row = 0; row < active_rows; ++row) {
        const float warp_sum =
            ffpa::warp::reduce_sum<float, WARP_SIZE>(score_partial[row]);
        if (lane == 0) {
          warp_partials[row][warp_id] = warp_sum;
        }
      }
      __syncthreads();

      if (warp_id == 0 && lane < active_rows) {
        float block_sum = 0.0f;
#pragma unroll
        for (int warp = 0; warp < kNumWarps; ++warp) {
          block_sum += warp_partials[lane][warp];
        }
        row_scores[lane] = block_sum;
      }
      // Only warp 0 writes row_scores; tid==0 (also warp 0) reads it.
      __syncwarp();

      if (tid == 0) {
        for (int row = 0; row < active_rows; ++row) {
          const bool visible = !causal || (global_k <= (row + kv_offset));
          if (!visible) {
            row_p[row] = 0.0f;
            row_rescale[row] = 1.0f;
            continue;
          }

          const float score = row_scores[row] * scale;
          const float m_old = row_m[row];
          const float l_old = row_l[row];
          const float m_new = max(m_old, score);
          const float rescale = (m_old > -INFINITY)
                                    ? exp2f((m_old - m_new) * FFPA_M_LOG2E)
                                    : 0.0f;
          const float p = exp2f((score - m_new) * FFPA_M_LOG2E);
          row_m[row] = m_new;
          row_l[row] = l_old * rescale + p;
          row_p[row] = p;
          row_rescale[row] = (l_old > 0.0f) ? rescale : 0.0f;
        }
      }
      __syncthreads();

      // Wait for V (group 1, oldest), then V update from smem V_tile.
      ffpa::cp_async::wait_group<1>();
      __syncthreads();

      for (int row = 0; row < active_rows; ++row) {
        if (tid < kNumActiveThreads) {
          const float p = row_p[row];
          const float rescale = row_rescale[row];
          const int d = tid * kElemsPerThread;
          uint4 vv;
          ffpa::cp_async::ldg_sync_128b(&vv, &V_tile[d]);
          const kDataType* vh = reinterpret_cast<const kDataType*>(&vv);
#pragma unroll
          for (int v = 0; v < kElemsPerThread; ++v) {
            acc[row][v] = acc[row][v] * rescale + p * Traits::to_float(vh[v]);
          }
        }
      }
      // Wait for K async load (group 0), then swap buffers for next iteration.
      if constexpr (kStage > 1) {
        if (global_k + kStage - 1 < split_end) {
          ffpa::cp_async::wait_group<0>();
        }
      }
      __syncthreads();
    }

    const int split_row_base =
        (((Nb_id * Nh + Nh_id) * num_splits + split_id) * Nq);
    constexpr int kVecF32 = 4;  // 4×fp32 = 128-bit store
    static_assert(kHeadDim % kVecF32 == 0, "kHeadDim must be multiple of 4");
    constexpr int kNumVecF32 = kHeadDim / kVecF32;
    for (int row = 0; row < active_rows; ++row) {
      const float inv_l = (row_l[row] > 0.0f) ? (1.0f / row_l[row]) : 0.0f;

      // Phase 1: pack acc into 2×uint4, store to smem_out via stg_sync_128b.
      if (tid < kNumActiveThreads) {
        const int d = tid * kElemsPerThread;
        float tmp[8];
#pragma unroll
        for (int v = 0; v < kElemsPerThread; ++v) {
          tmp[v] = acc[row][v] * inv_l;
        }
        ffpa::cp_async::stg_sync_128b(&smem_out[d], &tmp[0]);
        ffpa::cp_async::stg_sync_128b(&smem_out[d + 4], &tmp[4]);
      }
      __syncthreads();

      // Phase 2: vectorized 128-bit store smem_out → partial_out.
      // kNumVecF32 = kHeadDim / 4 (e.g. D=512 → 128), kNumThreads=256,
      // so tid 0..127 each handles one 4×fp32 vector, covering all D elements.
      const int partial_row_base = (split_row_base + row) * kHeadDim;
      for (int i = tid; i < kNumVecF32; i += kNumThreads) {
        ffpa::cp_async::stg_sync_128b(
            &partial_out[partial_row_base + i * kVecF32],
            &smem_out[i * kVecF32]);
      }
      __syncthreads();
    }

    if (tid < active_rows) {
      chunk_lse[split_row_base + tid] =
          (row_l[tid] > 0.0f) ? (row_m[tid] + logf(row_l[tid])) : -INFINITY;
    }
  }
}

// ============================================================================
// ffpa_attn_split_kv_decode_stage2_template
// ----------------------------------------------------------------------------
// Combine per-split partial outputs from stage1 into the final output / LSE.
// Each block handles one `(batch, q_head, row)` tuple and reduces over the
// split dimension using the standard stable log-sum-exp recurrence.
// ============================================================================
template <typename kDataType, const int kHeadDim>
__global__ void __launch_bounds__(((kHeadDim / 8 + WARP_SIZE - 1) / WARP_SIZE) *
                                  WARP_SIZE)
    ffpa_attn_split_kv_decode_stage2_template(
        const float* __restrict__ partial_out,
        const float* __restrict__ chunk_lse, kDataType* __restrict__ O,
        float* __restrict__ softmax_lse, const int Nq, const int Nh,
        const int num_splits) {
  using Traits = ffpa::DtypeTraits<kDataType>;
  constexpr int kNumActiveThreads = kHeadDim / 8;
  constexpr int kNumThreads =
      ((kNumActiveThreads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

  const int row = blockIdx.x;
#ifdef ENABLE_FFPA_LAUNCH_GRID_DNHB
  const int Nb_id = blockIdx.z;
  const int Nh_id = blockIdx.y;
#else
  const int Nb_id = blockIdx.y / Nh;
  const int Nh_id = blockIdx.y % Nh;
#endif
  const int tid = threadIdx.x;

  // smem_o must be declared first for 16-byte alignment (128-bit stores).
  __shared__ __align__(16) kDataType smem_o[kHeadDim];
  __shared__ float row_max;
  __shared__ float row_inv_denom;

  const int split_row_base = (((Nb_id * Nh + Nh_id) * num_splits) * Nq + row);
  const int o_row_base = ((Nb_id * Nh + Nh_id) * Nq + row) * kHeadDim;
  const int lse_row_offset = (Nb_id * Nh + Nh_id) * Nq + row;

  if (tid == 0) {
    float max_lse = -INFINITY;
    for (int split = 0; split < num_splits; ++split) {
      max_lse = max(max_lse, chunk_lse[split_row_base + split * Nq]);
    }

    if (max_lse == -INFINITY) {
      row_max = -INFINITY;
      row_inv_denom = 0.0f;
      softmax_lse[lse_row_offset] = -INFINITY;
    } else {
      float denom = 0.0f;
      for (int split = 0; split < num_splits; ++split) {
        denom += exp2f((chunk_lse[split_row_base + split * Nq] - max_lse) *
                       FFPA_M_LOG2E);
      }
      row_max = max_lse;
      row_inv_denom = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
      softmax_lse[lse_row_offset] =
          (denom > 0.0f) ? (max_lse + logf(denom)) : -INFINITY;
    }
  }
  __syncthreads();

  // Accumulate weighted partial outputs in 8×fp32 chunks (two 128-bit loads
  // per split, packed into one 128-bit smem store of 8×half).
  // kNumVecF32 = kHeadDim / 8 = kNumThreads (e.g. D=512 → 64),
  // so for aligned cases each thread handles one 8×fp32 vector per split loop;
  // for rounded launches (e.g. D=320), padded threads remain idle.
  constexpr int kVecF32 = 8;
  static_assert(kHeadDim % kVecF32 == 0, "kHeadDim must be multiple of 8");
  constexpr int kNumVecF32 = kHeadDim / kVecF32;
  for (int i = tid; i < kNumVecF32; i += kNumThreads) {
    const int d = i * kVecF32;
    float out0 = 0.0f, out1 = 0.0f, out2 = 0.0f, out3 = 0.0f;
    float out4 = 0.0f, out5 = 0.0f, out6 = 0.0f, out7 = 0.0f;
    if (row_inv_denom > 0.0f) {
      for (int split = 0; split < num_splits; ++split) {
        const float weight = exp2f(
            (chunk_lse[split_row_base + split * Nq] - row_max) * FFPA_M_LOG2E);

        // First 128-bit load: partial_out[d..d+3].
        const int partial_offset0 =
            ((split_row_base + split * Nq) * kHeadDim) + d;
        uint4 v0;
        ffpa::cp_async::ldg_sync_128b(&v0, &partial_out[partial_offset0]);
        const float* pf0 = reinterpret_cast<const float*>(&v0);
        out0 += weight * pf0[0];
        out1 += weight * pf0[1];
        out2 += weight * pf0[2];
        out3 += weight * pf0[3];

        // Second 128-bit load: partial_out[d+4..d+7].
        const int partial_offset1 =
            ((split_row_base + split * Nq) * kHeadDim) + d + 4;
        uint4 v1;
        ffpa::cp_async::ldg_sync_128b(&v1, &partial_out[partial_offset1]);
        const float* pf1 = reinterpret_cast<const float*>(&v1);
        out4 += weight * pf1[0];
        out5 += weight * pf1[1];
        out6 += weight * pf1[2];
        out7 += weight * pf1[3];
      }
      out0 *= row_inv_denom;
      out1 *= row_inv_denom;
      out2 *= row_inv_denom;
      out3 *= row_inv_denom;
      out4 *= row_inv_denom;
      out5 *= row_inv_denom;
      out6 *= row_inv_denom;
      out7 *= row_inv_denom;
    }
    // Pack 8×half into one 128-bit smem store.
    kDataType tmp[8];
    tmp[0] = Traits::from_float(out0);
    tmp[1] = Traits::from_float(out1);
    tmp[2] = Traits::from_float(out2);
    tmp[3] = Traits::from_float(out3);
    tmp[4] = Traits::from_float(out4);
    tmp[5] = Traits::from_float(out5);
    tmp[6] = Traits::from_float(out6);
    tmp[7] = Traits::from_float(out7);
    ffpa::cp_async::stg_sync_128b(&smem_o[d], tmp);
  }
  __syncthreads();

  // Vectorized 128-bit O store (kHeadDim/kVecO = D/8 = kNumThreads).
  constexpr int kVecO =
      16 / static_cast<int>(sizeof(kDataType));  // 8 for half/bf16
  static_assert(kHeadDim % kVecO == 0, "kHeadDim must be multiple of kVecO");
  for (int i = tid; i < kHeadDim / kVecO; i += kNumThreads) {
    ffpa::cp_async::stg_sync_128b(&O[o_row_base + i * kVecO],
                                  &smem_o[i * kVecO]);
  }
}

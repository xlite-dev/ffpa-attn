#pragma once

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>

#include "input_layout.cuh"

namespace ffpa_fp8 {

// Reference (V mean smoothing):
// https://github.com/thu-ml/SageAttention/blob/main/csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn.cu
// Smooth-V per-channel stats (mean + symmetric residual amax), two stages,
// deterministic (no atomics): stage 1 reduces a (bh, row-chunk) slab of V into
// fp32 partials (sum + max + min), stage 2 reduces chunks and derives
// mean = sum/N, amax = max(|max-mean|, |min-mean|) (sage smooth_v recipe:
// subtracting the per-D mean centers V so the e4m3 range covers the residual
// symmetrically). Mirrors kv_col_sum_kernel (smooth_k.cuh) for coalesced I/O.
constexpr int kVStatsRowsPerChunk = 512;

template <typename Element, int kD>
__global__ void v_col_stats_kernel(const Element* __restrict__ v,
                                   float* __restrict__ partials_sum,
                                   float* __restrict__ partials_max,
                                   float* __restrict__ partials_min, int Nkv,
                                   int rows_per_chunk, int D_og,
                                   Fp8InputLayout L) {
  constexpr int kVec = 16 / sizeof(Element);  // 8 half/bf16 per 16B
  constexpr int kColsPerRow = kD / kVec;
  constexpr int kThreads = 256;
  constexpr int kRowsPerIter = (kThreads + kColsPerRow - 1) / kColsPerRow;
  const int chunk = blockIdx.x;
  const int bh = blockIdx.y;
  const int row0 = chunk * rows_per_chunk;
  const int row_end = min(row0 + rows_per_chunk, Nkv);
  const Element* v_bh = v + fp8_plane_base(L, bh);
  const int col0 = (threadIdx.x % kColsPerRow) * kVec;

  float sum_acc[kVec], max_acc[kVec], min_acc[kVec];
#pragma unroll
  for (int i = 0; i < kVec; ++i) {
    sum_acc[i] = 0.0f;
    max_acc[i] = -INFINITY;
    min_acc[i] = INFINITY;
  }
  // Pad cols (col0 >= D_og) have no V data; D_og%8==0 keeps each kVec
  // entirely inside or outside the real dims, so the guard is exact.
  if (col0 < D_og) {
    for (int r = row0 + threadIdx.x / kColsPerRow; r < row_end;
         r += kRowsPerIter) {
      const uint4 packed = *reinterpret_cast<const uint4*>(
          v_bh + static_cast<long>(r) * L.s_row + col0);
      const Element* vals = reinterpret_cast<const Element*>(&packed);
#pragma unroll
      for (int i = 0; i < kVec; ++i) {
        const float x = static_cast<float>(vals[i]);
        sum_acc[i] += x;
        max_acc[i] = fmaxf(max_acc[i], x);
        min_acc[i] = fminf(min_acc[i], x);
      }
    }
  } else {
    // No data: collapse max/min to 0 so partials stay well-defined (the
    // finalize kernel skips pad cols anyway).
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      max_acc[i] = 0.0f;
      min_acc[i] = 0.0f;
    }
  }

  // Block reduce: kRowsPerIter partial rows -> one per column (3 stats).
  __shared__ float red_sum[kRowsPerIter][kD];
  __shared__ float red_max[kRowsPerIter][kD];
  __shared__ float red_min[kRowsPerIter][kD];
  if constexpr (kThreads % kColsPerRow != 0) {
    for (int i = threadIdx.x; i < kRowsPerIter * kD; i += kThreads) {
      reinterpret_cast<float*>(red_sum)[i] = 0.0f;
      reinterpret_cast<float*>(red_max)[i] = -INFINITY;
      reinterpret_cast<float*>(red_min)[i] = INFINITY;
    }
    __syncthreads();
  }
  const int row_grp = threadIdx.x / kColsPerRow;
#pragma unroll
  for (int i = 0; i < kVec; ++i) {
    red_sum[row_grp][col0 + i] = sum_acc[i];
    red_max[row_grp][col0 + i] = max_acc[i];
    red_min[row_grp][col0 + i] = min_acc[i];
  }
  __syncthreads();
  float* out_sum =
      partials_sum + (static_cast<long>(bh) * gridDim.x + chunk) * kD;
  float* out_max =
      partials_max + (static_cast<long>(bh) * gridDim.x + chunk) * kD;
  float* out_min =
      partials_min + (static_cast<long>(bh) * gridDim.x + chunk) * kD;
  for (int d = threadIdx.x; d < kD; d += kThreads) {
    float s = 0.0f, mx = -INFINITY, mn = INFINITY;
#pragma unroll
    for (int g = 0; g < kRowsPerIter; ++g) {
      s += red_sum[g][d];
      mx = fmaxf(mx, red_max[g][d]);
      mn = fminf(mn, red_min[g][d]);
    }
    out_sum[d] = s;
    out_max[d] = mx;
    out_min[d] = mn;
  }
}

template <int kD, bool kSmoothV>
__global__ void v_stats_finalize_kernel(const float* __restrict__ partials_sum,
                                        const float* __restrict__ partials_max,
                                        const float* __restrict__ partials_min,
                                        float* __restrict__ vm,
                                        float* __restrict__ v_scale, int chunks,
                                        int Nkv, float v_scale_max, int D_og) {
  const int bh = blockIdx.x;
  const int d = threadIdx.x;
  if (d >= kD)
    return;
  if (d >= D_og) {
    // Pad cols: no data, VT is zero-filled. v_scale=1 (identity) avoids
    // 0/0 in the quantize divide; vm=0 keeps the epilogue dequant (O += mean)
    // at 0. Deliberately NOT scaled by v_scale_max (decoupled on purpose).
    vm[static_cast<long>(bh) * kD + d] = 0.0f;
    v_scale[static_cast<long>(bh) * kD + d] = 1.0f;
    return;
  }
  const float inv_n = 1.0f / static_cast<float>(Nkv);
  float s = 0.0f, mx = -INFINITY, mn = INFINITY;
  for (int c = 0; c < chunks; ++c) {
    const long base = (static_cast<long>(bh) * chunks + c) * kD + d;
    s += partials_sum[base];
    mx = fmaxf(mx, partials_max[base]);
    mn = fminf(mn, partials_min[base]);
  }
  const float mean = s * inv_n;
  const float amax = kSmoothV ? fmaxf(fabsf(mx - mean), fabsf(mn - mean))
                              : fmaxf(fabsf(mx), fabsf(mn));
  vm[static_cast<long>(bh) * kD + d] = mean;
  v_scale[static_cast<long>(bh) * kD + d] = fmaxf(amax, 1e-8f) / v_scale_max;
}

// Per-(b,h) V stats [Nb, Nh_kv, Nkv, D] -> vm[bh, D] (mean) + v_scale[bh, D]
// (amax/v_scale_max). kSmoothV: residual amax max|V-mean| (symmetric, sage
// style); else absolute amax max|V|. v_scale_max controls the V8 range
// (V8 = (V-mean)/v_scale in [-v_scale_max, v_scale_max]); 448 keeps e4m3
// full range, smaller (e.g. 2.25) compresses V8 for fp16 PV acc precision.
// partials_sum/max/min are fp32 scratch (Nb*Nh_kv, chunks, D) with chunks =
// ceil(Nkv / 512), allocated by caller.
template <typename Element, int kHeadDim, bool kSmoothV>
void launch_v_stats_sm120(const Element* v_ptr, float* vm, float* v_scale,
                          float* partials_sum, float* partials_max,
                          float* partials_min, int Nb, int Nh_kv, int Nkv,
                          cudaStream_t stream, int D_og,
                          float v_scale_max = 448.0f,
                          const Fp8InputLayout* L = nullptr) {
  Fp8InputLayout bhnd;
  if (!L) {
    bhnd = {false, 0, 0, static_cast<long>(Nkv) * D_og, D_og};
    L = &bhnd;
  }
  const int bh = Nb * Nh_kv;
  const int chunks = (Nkv + kVStatsRowsPerChunk - 1) / kVStatsRowsPerChunk;
  dim3 grid(chunks, bh);
  v_col_stats_kernel<Element, kHeadDim><<<grid, 256, 0, stream>>>(
      v_ptr, partials_sum, partials_max, partials_min, Nkv, kVStatsRowsPerChunk,
      D_og, *L);
  v_stats_finalize_kernel<kHeadDim, kSmoothV><<<bh, kHeadDim, 0, stream>>>(
      partials_sum, partials_max, partials_min, vm, v_scale, chunks, Nkv,
      v_scale_max, D_og);
}

}  // namespace ffpa_fp8

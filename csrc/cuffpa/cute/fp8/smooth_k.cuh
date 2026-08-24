#pragma once

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>

#include "input_layout.cuh"

namespace ffpa_fp8 {

// Reference (smooth_k semantics + lse correction):
// https://github.com/thu-ml/SageAttention/blob/main/sageattention/core.py
// Smooth-K lse correction, per-row partial: dot(Q8_row, km). Softmax is
// shift-invariant, so smoothing K leaves O unchanged, but the returned lse
// must add back scale*qs*dot(Q_row, km) (see the attention epilogue).
// m16n8 C layout: the 4 peer lanes of a quad share the same rows; each lane
// strides over kHeadDim/16 column chunks of 4, and xor-1/xor-2 complete the
// quad-local reduce (a full-warp butterfly would mix the warp's 8 rows).
// Perf note: scalar smem/gmem reads, correctness-first; the lse path is cold,
// revisit only if it ever shows up in profiles.
// kAccumulate: add into qkm instead of overwriting (split-D calls this once
// per D chunk and accumulates the full dot).
template <int kHeadDim, int kRows, bool kAccumulate = false,
          typename SmemQTensor, typename CoordTensor>
CUTE_DEVICE void smooth_k_qk_dot(const SmemQTensor& sQ,
                                 const CoordTensor& tScS_rc,
                                 const float* __restrict__ km_bh, float* qkm) {
  constexpr int kVec = 4;
  constexpr int kQuad = 4;
  constexpr int kIters = kHeadDim / (kVec * kQuad);
  const int qlane = cutlass::canonical_lane_idx() % kQuad;
  // Accumulate mode: the prior qkm is already quad-reduced, so it must stay
  // out of the shfl reduction of this chunk's partial sum.
  float base[kRows];
#pragma unroll
  for (int row = 0; row < kRows; ++row) {
    base[row] = kAccumulate ? qkm[row] : 0.0f;
    const int r_idx = cute::get<0>(tScS_rc(row, 0));
    float acc = 0.0f;
#pragma unroll
    for (int it = 0; it < kIters; ++it) {
      const int col = (qlane + it * kQuad) * kVec;
#pragma unroll
      for (int d = 0; d < kVec; ++d)
        acc += static_cast<float>(sQ(r_idx, col + d)) * km_bh[col + d];
    }
    qkm[row] = acc;
  }
#pragma unroll
  for (int row = 0; row < kRows; ++row) {
    qkm[row] += __shfl_xor_sync(0xffffffff, qkm[row], 1);
    qkm[row] += __shfl_xor_sync(0xffffffff, qkm[row], 2);
    qkm[row] += base[row];
  }
}

// Smooth-K column mean, two stages, deterministic (no atomics/zero-init):
// stage 1 sums a (bh, row-chunk) slab of K into fp32 partials, stage 2
// reduces the chunks and divides by Nkv, emitting both the in-dtype mean and
// its fp32 copy (lse correction). Replaces at::mean + km.to(fp32) (~85us ->
// ~50us at B1 H32 N8192 D128): one coalesced DRAM pass with fp32 accumulate.
constexpr int kMeanRowsPerChunk = 512;

template <typename Element, int kD>
__global__ void kv_col_sum_kernel(const Element* __restrict__ k,
                                  float* __restrict__ partials, int Nkv,
                                  int rows_per_chunk, int D_og,
                                  ffpa_fp8::Fp8InputLayout L) {
  constexpr int kVec = 16 / sizeof(Element);  // 8 half/bf16 per 16B
  constexpr int kColsPerRow = kD / kVec;      // uint4s per row
  constexpr int kThreads = 256;
  // Ceil'd so headdims where kColsPerRow does not divide kThreads (e.g.
  // D=320/384/448) still map every thread to a valid row group.
  constexpr int kRowsPerIter = (kThreads + kColsPerRow - 1) / kColsPerRow;
  const int chunk = blockIdx.x;
  const int bh = blockIdx.y;
  const int row0 = chunk * rows_per_chunk;
  const int row_end = min(row0 + rows_per_chunk, Nkv);
  const Element* k_bh = k + ffpa_fp8::fp8_plane_base(L, bh);
  const int col0 = (threadIdx.x % kColsPerRow) * kVec;

  float acc[kVec];
#pragma unroll
  for (int i = 0; i < kVec; ++i)
    acc[i] = 0.0f;
  // Pad cols (col0 >= D_og): keep acc=0, don't read (would hit next row).
  if (col0 < D_og) {
    for (int r = row0 + threadIdx.x / kColsPerRow; r < row_end;
         r += kRowsPerIter) {
      const uint4 packed = *reinterpret_cast<const uint4*>(
          k_bh + static_cast<long>(r) * L.s_row + col0);
      const Element* vals = reinterpret_cast<const Element*>(&packed);
#pragma unroll
      for (int i = 0; i < kVec; ++i)
        acc[i] += static_cast<float>(vals[i]);
    }
  }

  // Block reduce: kRowsPerIter partial rows -> one per column.
  __shared__ float red[kRowsPerIter][kD];
  // With a ceil'd row group some (row_grp, col) slots have no owning thread;
  // zero them so they don't pollute the column sums.
  if constexpr (kThreads % kColsPerRow != 0) {
    for (int i = threadIdx.x; i < kRowsPerIter * kD; i += kThreads)
      reinterpret_cast<float*>(red)[i] = 0.0f;
    __syncthreads();
  }
  const int row_grp = threadIdx.x / kColsPerRow;
#pragma unroll
  for (int i = 0; i < kVec; ++i)
    red[row_grp][col0 + i] = acc[i];
  __syncthreads();
  float* out = partials + (static_cast<long>(bh) * gridDim.x + chunk) * kD;
  for (int d = threadIdx.x; d < kD; d += kThreads) {
    float s = 0.0f;
#pragma unroll
    for (int g = 0; g < kRowsPerIter; ++g)
      s += red[g][d];
    out[d] = s;
  }
}

template <typename Element>
__global__ void kv_mean_finalize_kernel(const float* __restrict__ partials,
                                        Element* __restrict__ km,
                                        float* __restrict__ km_f32, int chunks,
                                        int Nkv, int D) {
  const int bh = blockIdx.x;
  const float inv_n = 1.0f / static_cast<float>(Nkv);
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float s = 0.0f;
    for (int c = 0; c < chunks; ++c)
      s += partials[(static_cast<long>(bh) * chunks + c) * D + d];
    const float m = s * inv_n;
    if (km)
      km[static_cast<long>(bh) * D + d] = static_cast<Element>(m);
    km_f32[static_cast<long>(bh) * D + d] = m;
  }
}

// Per-(b,h) seqlen mean of K [Nb, Nh_kv, Nkv, D]; any D % 8 == 0. Writes km
// (in-dtype, may be nullptr to skip - the fp4 smooth_v path only needs the
// fp32 mean) and km_f32; partials is fp32 scratch (Nb*Nh_kv, chunks, D) with
// chunks = ceil(Nkv / 512), allocated by the caller. The mean is exact over
// the true Nkv rows (chunk-tail guard); used both for K smoothing (fp8) and
// V smoothing (fp4) - it is a generic column-mean primitive.
template <typename Element, int kHeadDim>
void launch_kv_mean_sm120(const Element* k_ptr, Element* km, float* km_f32,
                          float* partials, int Nb, int Nh_kv, int Nkv, int D_og,
                          cudaStream_t stream,
                          const Fp8InputLayout* L = nullptr) {
  Fp8InputLayout bhnd;
  if (!L) {
    bhnd = {false, 0, 0, static_cast<long>(Nkv) * D_og, D_og};
    L = &bhnd;
  }
  const int bh = Nb * Nh_kv;
  const int chunks = (Nkv + kMeanRowsPerChunk - 1) / kMeanRowsPerChunk;
  dim3 grid(chunks, bh);
  kv_col_sum_kernel<Element, kHeadDim><<<grid, 256, 0, stream>>>(
      k_ptr, partials, Nkv, kMeanRowsPerChunk, D_og, *L);
  kv_mean_finalize_kernel<Element>
      <<<bh, 128, 0, stream>>>(partials, km, km_f32, chunks, Nkv, kHeadDim);
}

}  // namespace ffpa_fp8

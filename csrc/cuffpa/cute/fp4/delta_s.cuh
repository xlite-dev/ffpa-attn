// Fused delta_s preprocess for the NVFP4 persist_d path:
//   delta_s[b,h,m,n] = qm[b,h,m,:] @ K[b,hkv,n,:] - 2 * qkm[b,h,m]
// (identity of qm@(K-km)^T - qm.km^T). One wmma tile kernel replaces the
// host-side bmm + transpose copy + epilogue cast (~1 ms at N=16384).
// Tail columns n >= Nkv are zero-filled (masked -inf in the attn kernel).
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace ffpa_fp4 {

using namespace nvcuda;

// CTA: one (m_tile 128, n_tile 128) output tile of one (b, h). 8 warps,
// warp w owns rows [16w, 16w+16), full 128 cols via 8 wmma n-tiles.
// Mb < 128 (short Nq) and Nkv tail columns are masked on the epilogue.
template <typename T>
__global__ void delta_s_wmma_kernel(
    const T* __restrict__ qm,   // [Nb, Nh, Mb, 128] row-major
    const T* __restrict__ k,    // [Nb, Nhkv, Nkv, 128] row-major
    const T* __restrict__ qkm,  // [Nb, Nh, Mb] row-major
    float* __restrict__ ds,     // [Nb, Nh, Mb, Nkv_pad] row-major
    int Nh, int Nh_kv, int Mb, int Nkv, int Nkv_pad) {
  namespace w = nvcuda::wmma;
  const int m_tiles = (Mb + 127) / 128;
  const int bh = blockIdx.y / m_tiles;
  const int tile_m = blockIdx.y % m_tiles;
  const int b = bh / Nh;
  const int h = bh % Nh;
  const int n0 = blockIdx.x * 128;
  const int warp = threadIdx.x / 32;
  const int m0 = tile_m * 128 + warp * 16;
  // Mb can be < 128 (short Nq): out-of-range warps contribute nothing.
  const bool m_ok = m0 < Mb;

  const T* q_row = qm + ((long)(bh * Mb) + m0) * 128;
  // K as col-major B[k,n]: element (k,n) at k_ptr[n*128 + k].
  const T* k_base =
      k + ((long)((b * Nh_kv + h / (Nh / Nh_kv)) * Nkv) + n0) * 128;

  w::fragment<w::accumulator, 16, 16, 16, float> acc[8];
#pragma unroll
  for (int n = 0; n < 8; ++n)
    w::fill_fragment(acc[n], 0.0f);
  if (m_ok) {
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
      w::fragment<w::matrix_a, 16, 16, 16, T, w::row_major> fa;
      w::load_matrix_sync(fa, q_row + kk * 16, 128);
#pragma unroll
      for (int n = 0; n < 8; ++n) {
        w::fragment<w::matrix_b, 16, 16, 16, T, w::col_major> fb;
        w::load_matrix_sync(fb, k_base + (long)n * 16 * 128 + kk * 16, 128);
        w::mma_sync(acc[n], fa, fb, acc[n]);
      }
    }
  }

  // Stage through smem (130-col pad avoids bank conflicts) so the epilogue
  // writes full rows and the qkm term is applied per output row. Dynamic
  // smem (64KB) exceeds the 48KB static limit.
  extern __shared__ float tile[];
  constexpr int kTileLd = 130;
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    w::store_matrix_sync(tile + warp * 16 * kTileLd + n * 16, acc[n], kTileLd,
                         w::mem_row_major);
  }
  __syncthreads();

  const int row = threadIdx.x / 2;
  const int c4 = (threadIdx.x % 2) * 64;  // 16 float4 per thread
  const int m_global = tile_m * 128 + row;
  if (m_global >= Mb)
    return;
  const float qk = static_cast<float>(qkm[(long)bh * Mb + m_global]);
  const float* src = tile + row * kTileLd + c4;
  float* out = ds + ((long)(bh * Mb) + m_global) * Nkv_pad + n0 + c4;
  const int n_valid = Nkv - n0 - c4;  // valid cols in this thread's span
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const float* s = src + i * 4;
    float4 v = {s[0] - 2.f * qk, s[1] - 2.f * qk, s[2] - 2.f * qk,
                s[3] - 2.f * qk};
    if (i * 4 + 3 < n_valid) {
      reinterpret_cast<float4*>(out)[i] = v;
    } else {  // tail tile: column-wise valid/zero mix
      float* o = out + i * 4;
      o[0] = i * 4 + 0 < n_valid ? v.x : 0.0f;
      o[1] = i * 4 + 1 < n_valid ? v.y : 0.0f;
      o[2] = i * 4 + 2 < n_valid ? v.z : 0.0f;
      o[3] = i * 4 + 3 < n_valid ? v.w : 0.0f;
    }
  }
}

template <typename T>
inline void launch_fp4_delta_s_sm120(const T* qm, const T* k, const T* qkm,
                                     float* ds, int Nb, int Nh, int Nh_kv,
                                     int Mb, int Nkv, int Nkv_pad,
                                     cudaStream_t stream) {
  const int tc = Nkv_pad / 128;
  const int m_tiles = (Mb + 127) / 128;
  dim3 grid(tc, Nb * Nh * m_tiles);
  const int smem = 128 * 130 * sizeof(float);
  cudaFuncSetAttribute(delta_s_wmma_kernel<T>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  delta_s_wmma_kernel<T><<<grid, 256, smem, stream>>>(qm, k, qkm, ds, Nh, Nh_kv,
                                                      Mb, Nkv, Nkv_pad);
}

}  // namespace ffpa_fp4

// Fused delta_s preprocess for the NVFP4 persist_d path:
//   delta_s[b,h,m,n] = qm[b,h,m,:] @ (K[b,hkv,n,:] - km)
//                    = qm @ K^T - qkm   with qkm[b,h,m] = qm . km
// (the kernel's S = Qhat@Khat^T + delta_s then equals q(k-km)^T, so the
// lse correction needs exactly scale * dot(q_row, km); any extra row
// constant here would shift the lse without changing O). One wmma tile
// kernel replaces the host-side bmm + transpose copy + epilogue cast
// (~1 ms at N=16384).
// Tail columns n >= Nkv are zero-filled (masked -inf in the attn kernel).
// Plain wmma on fp16 operands: the kernel is memory-bound (SM ~10%), so
// the block-scaled NVFP4 CuTe atoms of the attention kernel buy nothing
// here. If MMA ever becomes the limit, a CuTe port (LDSM/swizzled smem,
// tiled HMMA) is the upgrade path.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace ffpa_fp4 {

using namespace nvcuda;

// CTA: one (m_tile 128, n_tile 128) output tile of one (b, h). 8 warps,
// warp w owns rows [16w, 16w+16), full 128 cols via 8 wmma n-tiles.
// The K reduction (kHeadDim) is staged in 64-element chunks: full A/B
// staging would need 128*(D+8)*2*2 bytes and breaks the 99KB sm_120 budget
// at D>=192, so A/B land in a 2*128*72*2 = 36KB window that is reused as
// the f32 accumulator staging tile (128x136 floats = 68KB) for the
// epilogue. Rows m >= Mb and n >= Nkv are zero-filled on load: no OOB gmem
// reads and garbage-free accumulators (out-of-range rows are skipped in the
// epilogue anyway).
template <typename T, int kHeadDim>
__global__ void delta_s_wmma_kernel(
    const T* __restrict__ qm,   // [Nb, Nh, Mb, kHeadDim] row-major
    const T* __restrict__ k,    // [Nb, Nhkv, Nkv, d_og] row-major
    const T* __restrict__ qkm,  // [Nb, Nh, Mb] row-major
    float* __restrict__ ds,     // [Nb, Nh, Mb, Nkv_pad] row-major
    int Nh, int Nh_kv, int Mb, int Nkv, int Nkv_pad, int d_og) {
  namespace w = nvcuda::wmma;
  constexpr int kChunk = 64;   // K elements staged per pass
  constexpr int kLdAB = 72;    // kChunk + 8 halves (ldm multiple of 8)
  constexpr int kLdAcc = 136;  // multiple of 4 floats / 16B rows
  static_assert(kHeadDim % kChunk == 0, "delta_s requires 64-multiple D");
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

  extern __shared__ float tile[];
  T* sA = reinterpret_cast<T*>(tile);
  T* sB = sA + 128 * kLdAB;

  const T* gA = qm + (long)(bh * Mb + tile_m * 128) * kHeadDim;
  const T* gB = k + ((long)((b * Nh_kv + h / (Nh / Nh_kv)) * Nkv) + n0) * d_og;
  constexpr int kVec4PerRow = kChunk / 8;  // 16B loads per staged row

  w::fragment<w::accumulator, 16, 16, 16, float> acc[8];
#pragma unroll
  for (int n = 0; n < 8; ++n)
    w::fill_fragment(acc[n], 0.0f);

  for (int kc = 0; kc < kHeadDim; kc += kChunk) {
    // Cooperative A/B chunk load: 128 rows x 64 halves (8 uint4) each.
#pragma unroll
    for (int i = threadIdx.x; i < 128 * kVec4PerRow; i += 256) {
      const int row = i / kVec4PerRow;
      const int col = (i % kVec4PerRow) * 8;
      if (tile_m * 128 + row < Mb) {
        *reinterpret_cast<uint4*>(sA + row * kLdAB + col) =
            *reinterpret_cast<const uint4*>(gA + row * kHeadDim + kc + col);
      } else {
        *reinterpret_cast<uint4*>(sA + row * kLdAB + col) = uint4{0, 0, 0, 0};
      }
      // K rows are only d_og wide (d_og%8==0: 8-elem uint4 loads stay
      // whole); guarded loads keep pad cols zero. qm is a padded
      // kHeadDim-wide buffer whose pad cols are already 0.
      if (n0 + row < Nkv && kc + col < d_og) {
        *reinterpret_cast<uint4*>(sB + row * kLdAB + col) =
            *reinterpret_cast<const uint4*>(gB + row * d_og + kc + col);
      } else {
        *reinterpret_cast<uint4*>(sB + row * kLdAB + col) = uint4{0, 0, 0, 0};
      }
    }
    __syncthreads();
    if (m_ok) {
#pragma unroll
      for (int kk = 0; kk < kChunk / 16; ++kk) {
        w::fragment<w::matrix_a, 16, 16, 16, T, w::row_major> fa;
        // sA holds one 128-row tile: index by the in-tile row (warp * 16),
        // not the global m0 (tile_m * 128 + warp * 16).
        w::load_matrix_sync(fa, sA + warp * 16 * kLdAB + kk * 16, kLdAB);
#pragma unroll
        for (int n = 0; n < 8; ++n) {
          w::fragment<w::matrix_b, 16, 16, 16, T, w::col_major> fb;
          w::load_matrix_sync(fb, sB + n * 16 * kLdAB + kk * 16, kLdAB);
          w::mma_sync(acc[n], fa, fb, acc[n]);
        }
      }
    }
    // A/B smem is dead for every warp only after all warps finish the mma
    // loop (warps read B rows owned by other warps), so the next chunk load
    // must wait.
    __syncthreads();
  }

  // The A/B staging is dead; the same storage now backs the f32 accumulator
  // tile written below.
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    w::store_matrix_sync(tile + warp * 16 * kLdAcc + n * 16, acc[n], kLdAcc,
                         w::mem_row_major);
  }
  __syncthreads();

  const int row = threadIdx.x / 2;
  const int c4 = (threadIdx.x % 2) * 64;  // 16 float4 per thread
  const int m_global = tile_m * 128 + row;
  if (m_global >= Mb)
    return;
  const float qk = static_cast<float>(qkm[(long)bh * Mb + m_global]);
  const float* src = tile + row * kLdAcc + c4;
  float* out = ds + ((long)(bh * Mb) + m_global) * Nkv_pad + n0 + c4;
  const int n_valid = Nkv - n0 - c4;  // valid cols in this thread's span
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const float* s = src + i * 4;
    // qm @ (k - km)^T = (qm @ k^T) - qm.km: subtract ONE qkm.
    float4 v = {s[0] - qk, s[1] - qk, s[2] - qk, s[3] - qk};
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

template <typename T, int kHeadDim>
inline void launch_fp4_delta_s_sm120(const T* qm, const T* k, const T* qkm,
                                     float* ds, int Nb, int Nh, int Nh_kv,
                                     int Mb, int Nkv, int Nkv_pad, int d_og,
                                     cudaStream_t stream) {
  const int tc = Nkv_pad / 128;
  const int m_tiles = (Mb + 127) / 128;
  dim3 grid(tc, Nb * Nh * m_tiles);
  // Union of the A/B chunk staging (2*128*72*sizeof(T)) and the f32 acc
  // tile (128*136*4); the acc side dominates for fp16/bf16.
  constexpr int kSmemAB = 2 * 128 * 72 * (int)sizeof(T);
  constexpr int kSmemAcc = 128 * 136 * 4;
  constexpr int smem = kSmemAB > kSmemAcc ? kSmemAB : kSmemAcc;
  cudaFuncSetAttribute(delta_s_wmma_kernel<T, kHeadDim>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  delta_s_wmma_kernel<T, kHeadDim><<<grid, 256, smem, stream>>>(
      qm, k, qkm, ds, Nh, Nh_kv, Mb, Nkv, Nkv_pad, d_og);
}

}  // namespace ffpa_fp4

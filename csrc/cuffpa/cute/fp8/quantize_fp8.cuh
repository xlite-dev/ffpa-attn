#pragma once

#include <cuda_fp8.h>
#include <cstdint>
#include <type_traits>

#include "smooth_v.cuh"

namespace ffpa_fp8 {

// 8-elem (16-byte) vector for coalesced fp16/bf16 quantize I/O.
template <typename Element>
union Vec8 {
  uint4 raw;
  Element elem[8];
  static __device__ __forceinline__ Vec8 zero() {
    Vec8 v;
    v.raw = make_uint4(0, 0, 0, 0);
    return v;
  }
};
static_assert(sizeof(Vec8<__half>) == 16, "Vec8 must be 128-bit");

// Symmetric int8 quantize (Sage recipe): round-nearest-even, clamp [-127,127]
// (scale = amax/127 keeps -128 unused, avoiding the asymmetric extreme).
__device__ __forceinline__ int8_t quant_sym_int8(float x) {
  const float q = fminf(fmaxf(rintf(x), -127.0f), 127.0f);
  return static_cast<int8_t>(static_cast<int>(q));
}

// fp8/int8 QK quantize output element (int8 when kQKInt8).
template <bool kQKInt8>
using QKOutT = std::conditional_t<kQKInt8, int8_t, __nv_fp8_e4m3>;

// Row-major blockwise e4m3 quantization of fp16/bf16 Q/K for the FP8
// persist-D kernel. Output keeps the (N, D) row-major layout; scales are
// amax/448 per (b, h, row-block). One block per (head, row-block).
//
// Requires D % 8 == 0. Memory accesses are fully vectorized: 128-bit (8-elem)
// loads, and 64-bit (8 x e4m3) stores whose 8B chunks never straddle a
// D-element row (D % 8 == 0), so every transaction is aligned and coalesced.
// The amax pass keeps the 8-elem chunks in registers, so the quantize pass
// converts straight from regs (no second global read).
// kQKInt8: symmetric int8 output (scale = amax/127) instead of e4m3.
template <typename Element, int kBlockRows, bool kQKInt8, int kD>
__global__ void quantize_fp8_kernel(
    const Element* __restrict__ X,    // (B, H, N, kD) row-major
    QKOutT<kQKInt8>* __restrict__ Y,  // (B, H, N, kD)
    float* __restrict__ scale,        // (B, H, N/kBlockRows)
    int N, int H,
    const Element* __restrict__ km = nullptr,  // (B*H, D) smooth-K col means
    float inv_n = 0.0f) {
  constexpr int kVec = 8;  // elems per vector access (128-bit in, 64-bit out)
  constexpr int kThreads = 128;
  constexpr int D = kD;
  constexpr int dv = kD / kVec;           // vector chunks per row
  constexpr int total = kBlockRows * dv;  // vector chunks per block
  constexpr int n_chunks_per_thread = total / kThreads;
  static_assert(total % kThreads == 0, "tile must divide over threads");
  const int rb = blockIdx.x;
  const int bh = blockIdx.y;
  const int row0 = rb * kBlockRows;
  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid % 32;
  constexpr int kWarps = kThreads / 32;

  const Element* x_bh = X + static_cast<long>(bh) * N * D;
  QKOutT<kQKInt8>* y_bh = Y + static_cast<long>(bh) * N * D;

  // Smooth-K: cache this head's mean vector once (D floats, L1-hot reuse).
  __shared__ float km_sh[kD];
  const bool smooth = km != nullptr;
  if (smooth)
    for (int d = tid; d < D; d += kThreads)
      km_sh[d] = static_cast<float>(km[static_cast<long>(bh) * D + d]) * inv_n;
  __syncthreads();

  using VecIn = Vec8<Element>;
  float amax = 0.0f;
  VecIn regs[n_chunks_per_thread];
#pragma unroll
  for (int j = 0; j < n_chunks_per_thread; ++j) {
    const int i = tid + j * kThreads;
    const int r = i / dv;
    const int c = i % dv;
    const long row = row0 + r;
    VecIn v = Vec8<Element>::zero();
    if (row < N)
      v = reinterpret_cast<const VecIn*>(x_bh + row * D)[c];
    regs[j] = v;
#pragma unroll
    for (int e = 0; e < kVec; ++e) {
      float x = static_cast<float>(v.elem[e]);
      if (smooth)
        x -= km_sh[c * kVec + e];
      amax = fmaxf(amax, fabsf(x));
    }
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
  __shared__ float warp_max[kWarps];
  if (lane == 0)
    warp_max[warp] = amax;
  __syncthreads();
  if (warp == 0) {
    amax = lane < kWarps ? warp_max[lane] : 0.0f;
#pragma unroll
    for (int off = kWarps / 2; off > 0; off >>= 1)
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
    if (lane == 0)
      warp_max[0] = amax;
  }
  __syncthreads();
  amax = warp_max[0];

  float s = kQKInt8 ? amax / 127.0f : amax / 448.0f;
  const float inv_s = (s == 0.0f) ? 0.0f : 1.0f / s;
  const int n_rb = (N + kBlockRows - 1) / kBlockRows;
  if (tid == 0)
    scale[static_cast<long>(bh) * n_rb + rb] = s;

#pragma unroll
  for (int j = 0; j < n_chunks_per_thread; ++j) {
    const int i = tid + j * kThreads;
    const int r = i / dv;
    const int c = i % dv;
    const long row = row0 + r;
    if (row >= N)
      continue;
    const VecIn v = regs[j];
    uint2 out;
    QKOutT<kQKInt8>* ob = reinterpret_cast<QKOutT<kQKInt8>*>(&out);
#pragma unroll
    for (int e = 0; e < kVec; ++e) {
      float x = static_cast<float>(v.elem[e]);
      if (smooth)
        x -= km_sh[c * kVec + e];
      if constexpr (kQKInt8)
        ob[e] = quant_sym_int8(x * inv_s);
      else
        ob[e] = __nv_fp8_e4m3(x * inv_s);
    }
    reinterpret_cast<uint2*>(y_bh + row * D)[c] = out;
  }
}

// Transposed (D, N) V quantize staged through smem so BOTH the gmem read and
// the gmem write are fully coalesced. Grid: (Nkv/kBlockRows, B*Nh_kv); block
// covers one (bh, row-block) tile of (kBlockRows, D). Global reads are
// 128-bit vectorized; 8-elem runs land in smem as 2x uint32 (pad keeps bank
// conflicts low on the transposed read-out).
template <typename Element, int kBlockRows, int kD>
__global__ void quantize_fp8_vt_kernel(
    const Element* __restrict__ V,   // (B, H, N, D) row-major
    __nv_fp8_e4m3* __restrict__ VT,  // flat [B*Nh_kv*D, Nkv_pad]
    float* __restrict__ scale,       // (B*Nh_kv, Nkv/kBlockRows)
    int N, int Nh_kv, int ldy) {
  constexpr int D = kD;
  constexpr int kPad = 16;
  constexpr int kVec = 8;
  constexpr int kThreads = 128;
  constexpr int dv = D / kVec;  // vector chunks per row
  static_assert(D % kVec == 0, "vt kernel requires D % 8 == 0");
  static_assert(kBlockRows * dv % kThreads == 0);
  const int rb = blockIdx.x;
  const int bh = blockIdx.y;
  const int row0 = rb * kBlockRows;
  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid % 32;
  constexpr int kWarps = kThreads / 32;

  // VT staging tile lives in DYNAMIC smem (sized by the launcher). D is
  // tiled into kDChunk-wide columns so the staging tile fits the 48KB
  // static-smem cap (kBlockRows*(kDChunk+kPad)) even for large D.
  extern __shared__ __nv_fp8_e4m3 tile_sh[];
  constexpr int kDChunk = (D < 256) ? D : 256;
  constexpr int kNDChunks = (D + kDChunk - 1) / kDChunk;
  using TileRow = __nv_fp8_e4m3[kDChunk + kPad];
  TileRow* tile = reinterpret_cast<TileRow*>(tile_sh);
  __shared__ float warp_max[kWarps];

  constexpr int total = kBlockRows * dv;  // vector chunks per tile
  constexpr int n_chunks_per_thread = total / kThreads;
  static_assert(total % kThreads == 0);

  const Element* v_bh = V + static_cast<long>(bh) * N * D;
  using VecIn = Vec8<Element>;

  // Pass 1: vectorized coalesced read cached in regs (like the QK kernel),
  // track amax. The quantize pass below converts straight from regs, so V is
  // read from DRAM/L2 exactly once.
  float amax = 0.0f;
  VecIn regs[n_chunks_per_thread];
#pragma unroll
  for (int j = 0; j < n_chunks_per_thread; ++j) {
    const int i = tid + j * kThreads;
    const int r = i / dv;
    const int c = i % dv;
    VecIn v = Vec8<Element>::zero();
    if (row0 + r < N)
      v = reinterpret_cast<const VecIn*>(v_bh +
                                         static_cast<long>(row0 + r) * D)[c];
    regs[j] = v;
#pragma unroll
    for (int e = 0; e < kVec; ++e)
      amax = fmaxf(amax, fabsf(static_cast<float>(v.elem[e])));
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
  if (lane == 0)
    warp_max[warp] = amax;
  __syncthreads();
  if (warp == 0) {
    amax = lane < kWarps ? warp_max[lane] : 0.0f;
#pragma unroll
    for (int off = kWarps / 2; off > 0; off >>= 1)
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
    if (lane == 0)
      warp_max[0] = amax;
  }
  __syncthreads();
  amax = warp_max[0];

  float s = amax / 448.0f;
  const float inv_s = (s == 0.0f) ? 0.0f : 1.0f / s;
  if (tid == 0)
    scale[static_cast<long>(bh) * ((N + kBlockRows - 1) / kBlockRows) + rb] = s;

  const long vt_base = static_cast<long>(bh) * D * ldy;
  // Pass 2+3: per D-chunk quantize regs -> smem -> transpose store. Tiling
  // keeps the staging tile within the 48KB static-smem cap for large D.
  for (int dci = 0; dci < kNDChunks; ++dci) {
    const int dc = dci * kDChunk;
    const int d_end = (dc + kDChunk < D) ? dc + kDChunk : D;
    const int dc_vec = dc / kVec;
    const int d_vec_end = d_end / kVec;
#pragma unroll 1
    for (int j = 0; j < n_chunks_per_thread; ++j) {
      const int i_local = tid + j * kThreads;
      const int r = i_local / dv;
      const int c_vec = i_local % dv;
      if (c_vec < dc_vec || c_vec >= d_vec_end)
        continue;
      const int c_local = c_vec - dc_vec;
      const VecIn v = regs[j];
      uint32_t pack[2];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        __nv_fp8_e4m3* ob = reinterpret_cast<__nv_fp8_e4m3*>(&pack[h]);
#pragma unroll
        for (int e = 0; e < 4; ++e)
          ob[e] = __nv_fp8_e4m3(static_cast<float>(v.elem[h * 4 + e]) * inv_s);
      }
      uint32_t* dst = reinterpret_cast<uint32_t*>(&tile[r][c_local * kVec]);
      dst[0] = pack[0];
      dst[1] = pack[1];
    }
    __syncthreads();

    // Pass 3: coalesced transpose write for this chunk's D range [dc, d_end).
#pragma unroll 1
    for (int it = warp * 4; it < (d_end - dc); it += kWarps * 4) {
#pragma unroll
      for (int cc = 0; cc < 4; ++cc) {
        const int c = dc + it + cc;
#pragma unroll
        for (int t = 0; t < kBlockRows / 32; ++t) {
          const int n = t * 32 + lane;
          if (row0 + n < N)
            VT[vt_base + static_cast<long>(c) * ldy + row0 + n] =
                tile[n][it + cc];
        }
      }
    }
    __syncthreads();  // tile safe to overwrite next chunk
  }
}

// Fused QKV quantize: one launch handles Q (row-major), K (row-major), and
// V (transposed) via blockIdx.z role selection. Requires kBr == kBc and
// self-attention (Nq == Nkv, Nh == Nh_kv) so all three share the same grid
// dimensions. Falls back to separate launches otherwise.
// kQKInt8: roles 0/1 emit symmetric int8 (amax/127); VT stays e4m3.
template <typename Element, int kBlockRows, int kD, int kThreads, bool kQKInt8>
__global__ void quantize_fp8_qkv_fused_kernel(
    const Element* __restrict__ Q, const Element* __restrict__ K,
    const Element* __restrict__ V, QKOutT<kQKInt8>* __restrict__ Q8,
    QKOutT<kQKInt8>* __restrict__ K8, __nv_fp8_e4m3* __restrict__ VT8,
    float* __restrict__ q_scale, float* __restrict__ k_scale,
    float* __restrict__ v_scale, int N, int H, int ldy,
    const Element* __restrict__ km = nullptr,  // (B*H, D) col means
    float inv_n = 0.0f) {
  constexpr int kVec = 8;
  constexpr int kPad = 16;
  constexpr int kWarps = kThreads / 32;
  const int role = blockIdx.z;  // 0=Q, 1=K, 2=VT
  const int rb = blockIdx.x;
  const int bh = blockIdx.y;
  const int row0 = rb * kBlockRows;
  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid % 32;

  constexpr int dv = kD / kVec;
  constexpr int total = kBlockRows * dv;
  constexpr int n_chunks_per_thread = total / kThreads;
  static_assert(total % kThreads == 0, "tile must divide over threads");
  const int n_rb = (N + kBlockRows - 1) / kBlockRows;

  __shared__ float warp_max[kWarps];  // VT staging tile (role==2) lives in
                                      // DYNAMIC smem (sized by the launcher):
  // D=512 needs ~67KB, beyond the 48KB static-smem default cap.
  extern __shared__ __nv_fp8_e4m3
      tile_sh[];  // Smooth-K applies to K only (role==1); cache the head's mean
                  // vector once.
  __shared__ float km_sh[kD];
  const bool smooth = (role == 1) && (km != nullptr);
  if (smooth)
    for (int d = tid; d < kD; d += kThreads)
      km_sh[d] = static_cast<float>(km[static_cast<long>(bh) * kD + d]) * inv_n;

  if (role < 2) {
    const Element* X = (role == 0) ? Q : K;
    QKOutT<kQKInt8>* Y = (role == 0) ? Q8 : K8;
    float* scale_out = (role == 0) ? q_scale : k_scale;
    const Element* x_bh = X + static_cast<long>(bh) * N * kD;
    QKOutT<kQKInt8>* y_bh = Y + static_cast<long>(bh) * N * kD;
    if (smooth)
      __syncthreads();

    using VecIn = Vec8<Element>;
    float amax = 0.0f;
    VecIn regs[n_chunks_per_thread];
#pragma unroll
    for (int j = 0; j < n_chunks_per_thread; ++j) {
      const int i = tid + j * kThreads;
      const int r = i / dv;
      const int c = i % dv;
      const long row = row0 + r;
      VecIn v = Vec8<Element>::zero();
      if (row < N)
        v = reinterpret_cast<const VecIn*>(x_bh + row * kD)[c];
      regs[j] = v;
#pragma unroll
      for (int e = 0; e < kVec; ++e) {
        float x = static_cast<float>(v.elem[e]);
        if (smooth)
          x -= km_sh[c * kVec + e];
        amax = fmaxf(amax, fabsf(x));
      }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
    if (lane == 0)
      warp_max[warp] = amax;
    __syncthreads();
    if (warp == 0) {
      amax = lane < kWarps ? warp_max[lane] : 0.0f;
#pragma unroll
      for (int off = kWarps / 2; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
      if (lane == 0)
        warp_max[0] = amax;
    }
    __syncthreads();
    amax = warp_max[0];

    float s = kQKInt8 ? amax / 127.0f : amax / 448.0f;
    const float inv_s = (s == 0.0f) ? 0.0f : 1.0f / s;
    if (tid == 0)
      scale_out[static_cast<long>(bh) * n_rb + rb] = s;

#pragma unroll 1
    for (int j = 0; j < n_chunks_per_thread; ++j) {
      const int i = tid + j * kThreads;
      if (i >= total)
        break;
      const int r = i / dv;
      const int c = i % dv;
      const long row = row0 + r;
      if (row >= N)
        continue;
      const VecIn v = regs[j];
      uint2 out;
      QKOutT<kQKInt8>* ob = reinterpret_cast<QKOutT<kQKInt8>*>(&out);
#pragma unroll
      for (int e = 0; e < kVec; ++e) {
        float x = static_cast<float>(v.elem[e]);
        if (smooth)
          x -= km_sh[c * kVec + e];
        if constexpr (kQKInt8)
          ob[e] = quant_sym_int8(x * inv_s);
        else
          ob[e] = __nv_fp8_e4m3(x * inv_s);
      }
      reinterpret_cast<uint2*>(y_bh + row * kD)[c] = out;
    }
  } else {
    // V transpose quantize through smem. D is tiled into kDChunk-wide
    // columns so the staging tile fits the 48KB static-smem cap for large D.
    constexpr int kDChunk = (kD < 256) ? kD : 256;
    constexpr int kNDChunks = (kD + kDChunk - 1) / kDChunk;
    using TileRow = __nv_fp8_e4m3[kDChunk + kPad];
    TileRow* tile = reinterpret_cast<TileRow*>(tile_sh);
    const Element* v_bh = V + static_cast<long>(bh) * N * kD;
    using VecIn = Vec8<Element>;

    float amax = 0.0f;
#pragma unroll 1
    for (int j = 0; j < n_chunks_per_thread; ++j) {
      const int i = tid + j * kThreads;
      const int r = i / dv;
      const int c = i % dv;
      VecIn v = Vec8<Element>::zero();
      if (row0 + r < N)
        v = reinterpret_cast<const VecIn*>(v_bh +
                                           static_cast<long>(row0 + r) * kD)[c];
#pragma unroll
      for (int e = 0; e < kVec; ++e)
        amax = fmaxf(amax, fabsf(static_cast<float>(v.elem[e])));
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
    if (lane == 0)
      warp_max[warp] = amax;
    __syncthreads();
    if (warp == 0) {
      amax = lane < kWarps ? warp_max[lane] : 0.0f;
#pragma unroll
      for (int off = kWarps / 2; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
      if (lane == 0)
        warp_max[0] = amax;
    }
    __syncthreads();
    amax = warp_max[0];

    float s = amax / 448.0f;
    const float inv_s = (s == 0.0f) ? 0.0f : 1.0f / s;
    if (tid == 0)
      v_scale[static_cast<long>(bh) * n_rb + rb] = s;

    const long vt_base = static_cast<long>(bh) * kD * ldy;
    // Pass 2+3: per D-chunk read -> quantize -> smem -> transpose store.
    for (int dci = 0; dci < kNDChunks; ++dci) {
      const int dc = dci * kDChunk;
      const int d_end = (dc + kDChunk < kD) ? dc + kDChunk : kD;
      const int dc_vec = dc / kVec;
      const int d_vec_end = d_end / kVec;
#pragma unroll 1
      for (int j = 0; j < n_chunks_per_thread; ++j) {
        const int i = tid + j * kThreads;
        const int r = i / dv;
        const int c_vec = i % dv;
        if (c_vec < dc_vec || c_vec >= d_vec_end)
          continue;
        const int c_local = c_vec - dc_vec;
        VecIn v = Vec8<Element>::zero();
        if (row0 + r < N)
          v = reinterpret_cast<const VecIn*>(
              v_bh + static_cast<long>(row0 + r) * kD)[c_vec];
        uint32_t pack[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          __nv_fp8_e4m3* ob = reinterpret_cast<__nv_fp8_e4m3*>(&pack[h]);
#pragma unroll
          for (int e = 0; e < 4; ++e)
            ob[e] =
                __nv_fp8_e4m3(static_cast<float>(v.elem[h * 4 + e]) * inv_s);
        }
        uint32_t* dst = reinterpret_cast<uint32_t*>(&tile[r][c_local * kVec]);
        dst[0] = pack[0];
        dst[1] = pack[1];
      }
      __syncthreads();

#pragma unroll 1
      for (int it = warp * 4; it < (d_end - dc); it += kWarps * 4) {
#pragma unroll
        for (int cc = 0; cc < 4; ++cc) {
          const int c = dc + it + cc;
#pragma unroll
          for (int t = 0; t < kBlockRows / 32; ++t) {
            const int n = t * 32 + lane;
            if (row0 + n < N)
              VT8[vt_base + static_cast<long>(c) * ldy + row0 + n] =
                  tile[n][it + cc];
          }
        }
      }
      __syncthreads();
    }
  }
}

// Quantize + transpose V[N, D] -> VT[D, N_pad] with a per-D scale. Same smem
// staging tile as quantize_fp8_vt_kernel, but the scale is read per-D from
// v_scale[bh, D] instead of computed as a per-tile amax.
template <typename Element, int kBlockRows, int kD>
__global__ void quantize_fp8_vt_perchannel_kernel(
    const Element* __restrict__ V,    // (B, H, N, D) row-major
    __nv_fp8_e4m3* __restrict__ VT,   // flat [B*Nh_kv*D, Nkv_pad]
    const float* __restrict__ scale,  // (B*Nh_kv, D)
    const float* __restrict__ vm,     // (B*Nh_kv, D) per-D mean, or nullptr
    int N, int Nh_kv, int ldy) {
  constexpr int D = kD;
  constexpr int kPad = 16;
  constexpr int kVec = 8;
  constexpr int kThreads = 128;
  constexpr int dv = D / kVec;
  const int rb = blockIdx.x;
  const int bh = blockIdx.y;
  const int row0 = rb * kBlockRows;
  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid % 32;
  constexpr int kWarps = kThreads / 32;

  extern __shared__ __nv_fp8_e4m3 tile_sh[];
  constexpr int kDChunk = (D < 256) ? D : 256;
  constexpr int kNDChunks = (D + kDChunk - 1) / kDChunk;
  using TileRow = __nv_fp8_e4m3[kDChunk + kPad];
  TileRow* tile = reinterpret_cast<TileRow*>(tile_sh);

  constexpr int total = kBlockRows * dv;
  constexpr int n_chunks_per_thread = total / kThreads;
  static_assert(total % kThreads == 0);

  const Element* v_bh = V + static_cast<long>(bh) * N * D;
  const float* scale_bh = scale + static_cast<long>(bh) * D;
  const float* vm_bh = vm ? (vm + static_cast<long>(bh) * D) : nullptr;
  using VecIn = Vec8<Element>;

  VecIn regs[n_chunks_per_thread];
#pragma unroll
  for (int j = 0; j < n_chunks_per_thread; ++j) {
    const int i = tid + j * kThreads;
    const int r = i / dv;
    const int c = i % dv;
    VecIn v = Vec8<Element>::zero();
    if (row0 + r < N)
      v = reinterpret_cast<const VecIn*>(v_bh +
                                         static_cast<long>(row0 + r) * D)[c];
    regs[j] = v;
  }

  const long vt_base = static_cast<long>(bh) * D * ldy;
  for (int dci = 0; dci < kNDChunks; ++dci) {
    const int dc = dci * kDChunk;
    const int d_end = (dc + kDChunk < D) ? dc + kDChunk : D;
    const int dc_vec = dc / kVec;
    const int d_vec_end = d_end / kVec;
#pragma unroll 1
    for (int j = 0; j < n_chunks_per_thread; ++j) {
      const int i_local = tid + j * kThreads;
      const int r = i_local / dv;
      const int c_vec = i_local % dv;
      if (c_vec < dc_vec || c_vec >= d_vec_end)
        continue;
      const int c_local = c_vec - dc_vec;
      const VecIn v = regs[j];
      uint32_t pack[2];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        __nv_fp8_e4m3* ob = reinterpret_cast<__nv_fp8_e4m3*>(&pack[h]);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int d = dc + c_local * kVec + h * 4 + e;
          float x = static_cast<float>(v.elem[h * 4 + e]);
          if (vm_bh)
            x -= vm_bh[d];
          ob[e] = __nv_fp8_e4m3(x / scale_bh[d]);
        }
      }
      uint32_t* dst = reinterpret_cast<uint32_t*>(&tile[r][c_local * kVec]);
      dst[0] = pack[0];
      dst[1] = pack[1];
    }
    __syncthreads();

#pragma unroll 1
    for (int it = warp * 4; it < (d_end - dc); it += kWarps * 4) {
#pragma unroll
      for (int cc = 0; cc < 4; ++cc) {
        const int c = dc + it + cc;
#pragma unroll
        for (int t = 0; t < kBlockRows / 32; ++t) {
          const int n = t * 32 + lane;
          if (row0 + n < N)
            VT[vt_base + static_cast<long>(c) * ldy + row0 + n] =
                tile[n][it + cc];
        }
      }
    }
    __syncthreads();
  }
}

// Per-channel V quantize: coalesced stats (sum+max+min -> mean+amax) then
// quantize+transpose. kSmoothV: residual amax + mean subtract; else absolute
// amax, no mean subtract (vm_quant=nullptr). v_scale_max controls V8 range
// (default 448 = e4m3 full range); smaller compresses V8 for fp16 PV acc.
template <typename Element, int kBr, int kBc, int kHeadDim, bool kSmoothV>
void launch_quantize_fp8_vt_perchannel_sm120(
    const Element* v_ptr, __nv_fp8_e4m3* vt8, float* v_scale, float* vm,
    float* partials_sum, float* partials_max, float* partials_min, int B,
    int Nh_kv, int Nkv, int Nkv_pad, cudaStream_t stream,
    float v_scale_max = 448.0f) {
  launch_v_stats_sm120<Element, kHeadDim, kSmoothV>(
      v_ptr, vm, v_scale, partials_sum, partials_max, partials_min, B, Nh_kv,
      Nkv, stream, v_scale_max);
  constexpr int kDChunk = (kHeadDim < 256) ? kHeadDim : 256;
  constexpr int kVtSmemBytes = kBc * (kDChunk + 16);
  dim3 grid_q((Nkv + kBc - 1) / kBc, B * Nh_kv);
  auto kernel = quantize_fp8_vt_perchannel_kernel<Element, kBc, kHeadDim>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       kVtSmemBytes);
  kernel<<<grid_q, 128, kVtSmemBytes, stream>>>(
      v_ptr, vt8, v_scale, kSmoothV ? vm : nullptr, Nkv, Nh_kv, Nkv_pad);
}

// Per-thread Q quantize (fragment-aligned, NOT per-token): 64 scales per
// 128-row block. Each group covers a C-fragment row pair {r, r+8} (2 rows),
// paired via shfl_xor(amax, 8). Per-token would use 128 scales (1 row/scale);
// per-block uses 1. This middle ground gives zero-shuffle dequant (each
// thread's C-frag rows share one scale) at the cost of multi-row amax sharing.
// kQKInt8: int8 output (amax/127); else e4m3 (amax/448).
template <typename Element, int kD, bool kQKInt8>
__global__ void quantize_fp8_perthread_q_kernel(
    const Element* __restrict__ X,    // (B, H, N, kD)
    QKOutT<kQKInt8>* __restrict__ Y,  // (B, H, N, kD)
    float* __restrict__ scale,        // (B*H, ceil(N/128)*64)
    int N, int H) {
  constexpr int kVec = 8;
  constexpr int D = kD;
  constexpr int dv = D / kVec;
  const int rb = blockIdx.x;
  const int bh = blockIdx.y;
  const int tid = threadIdx.x;
  const int row = rb * 128 + tid;
  const Element* x_bh = X + static_cast<long>(bh) * N * D;
  QKOutT<kQKInt8>* y_bh = Y + static_cast<long>(bh) * N * D;

  float amax = 0.0f;
  Vec8<Element> regs[dv];
  if (row < N) {
#pragma unroll
    for (int c = 0; c < dv; ++c) {
      regs[c] = reinterpret_cast<const Vec8<Element>*>(x_bh + row * D)[c];
#pragma unroll
      for (int e = 0; e < kVec; ++e)
        amax = fmaxf(amax, fabsf(static_cast<float>(regs[c].elem[e])));
    }
  }
  // Pair rows tid and tid^8 (same C-fragment group).
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 8));
  const float s = kQKInt8 ? amax / 127.0f : amax / 448.0f;
  const float inv_s = (s == 0.0f) ? 0.0f : 1.0f / s;
  const int n_rb = (N + 127) / 128;
  const int g = (tid / 16) * 8 + tid % 8;
  if ((tid & 8) == 0)
    scale[static_cast<long>(bh) * (n_rb * 64) + rb * 64 + g] = s;

  if (row < N) {
#pragma unroll
    for (int c = 0; c < dv; ++c) {
      uint2 out;
      QKOutT<kQKInt8>* ob = reinterpret_cast<QKOutT<kQKInt8>*>(&out);
#pragma unroll
      for (int e = 0; e < kVec; ++e) {
        float v = static_cast<float>(regs[c].elem[e]) * inv_s;
        if constexpr (kQKInt8)
          ob[e] = quant_sym_int8(v);
        else
          ob[e] = __nv_fp8_e4m3(v);
      }
      reinterpret_cast<uint2*>(y_bh + row * D)[c] = out;
    }
  }
}

// Per-thread K quantize (fragment-aligned, NOT per-token): 4 scales per
// kBlockRows-row block, grouped by N_kv row (row%8)/2 matching SM89_16x8x32
// C-fragment N-column layout: thread lane accesses N_kv cols
// {2*(lane%4)+8n, +1}, group=lane%4=(col%8)/2. amax across ALL D per group.
// Per-token would use kBlockRows scales (1 row/scale); per-block uses 1.
// Groups repeat every 8 rows → valid for all warps (including m4n2's 2
// N-warps). Smooth-K (km) subtracted.
template <typename Element, int kBlockRows, int kD, bool kQKInt8>
__global__ void quantize_fp8_perthread_k_kernel(
    const Element* __restrict__ X,    // (B, H, N, kD)
    QKOutT<kQKInt8>* __restrict__ Y,  // (B, H, N, kD)
    float* __restrict__ scale,        // (B*H_kv, ceil(N/128)*4)
    int N, int H,
    const Element* __restrict__ km = nullptr,  // (B*H, D) smooth-K means
    float inv_n = 0.0f) {
  constexpr int kVec = 8;
  constexpr int D = kD;
  constexpr int dv = D / kVec;
  constexpr int kThreads = 128;
  constexpr int kWarps = kThreads / 32;
  const int rb = blockIdx.x;
  const int bh = blockIdx.y;
  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid % 32;
  const int row0 = rb * kBlockRows;
  const Element* x_bh = X + static_cast<long>(bh) * N * D;
  QKOutT<kQKInt8>* y_bh = Y + static_cast<long>(bh) * N * D;

  __shared__ float km_sh[kD];
  const bool smooth = km != nullptr;
  if (smooth)
    for (int d = tid; d < D; d += kThreads)
      km_sh[d] = static_cast<float>(km[static_cast<long>(bh) * D + d]) * inv_n;
  __syncthreads();

  // Group = (row % 8) / 2 (N_kv row dimension, NOT D element index).
  float amax[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  constexpr int total = kBlockRows * dv;
  constexpr int n_chunks_per_thread = total / kThreads;
  Vec8<Element> regs[n_chunks_per_thread];
  int chunk_r[n_chunks_per_thread], chunk_c[n_chunks_per_thread];
#pragma unroll
  for (int j = 0; j < n_chunks_per_thread; ++j) {
    const int i = tid + j * kThreads;
    const int r = i / dv;
    const int c = i % dv;
    chunk_r[j] = r;
    chunk_c[j] = c;
    const long row = row0 + r;
    Vec8<Element> v = Vec8<Element>::zero();
    if (row < N)
      v = reinterpret_cast<const Vec8<Element>*>(x_bh + row * D)[c];
    regs[j] = v;
    const int g = (r % 8) / 2;
#pragma unroll
    for (int e = 0; e < kVec; ++e) {
      float x = static_cast<float>(v.elem[e]);
      if (smooth)
        x -= km_sh[c * kVec + e];
      amax[g] = fmaxf(amax[g], fabsf(x));
    }
  }
  // Warp-reduce 4 amaxes.
#pragma unroll
  for (int g = 0; g < 4; ++g) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      amax[g] = fmaxf(amax[g], __shfl_xor_sync(0xffffffff, amax[g], off));
  }
  __shared__ float warp_max[kWarps * 4];
  if (lane == 0)
#pragma unroll
    for (int g = 0; g < 4; ++g)
      warp_max[warp * 4 + g] = amax[g];
  __syncthreads();
  float s[4];
  if (warp == 0 && lane == 0) {
#pragma unroll
    for (int g = 0; g < 4; ++g) {
      s[g] = warp_max[g];
#pragma unroll
      for (int w = 1; w < kWarps; ++w)
        s[g] = fmaxf(s[g], warp_max[w * 4 + g]);
      warp_max[g] = kQKInt8 ? s[g] / 127.0f : s[g] / 448.0f;
    }
  }
  __syncthreads();
  float sc[4] = {warp_max[0], warp_max[1], warp_max[2], warp_max[3]};
  const int n_rb = (N + kBlockRows - 1) / kBlockRows;
  if (tid == 0)
#pragma unroll
    for (int g = 0; g < 4; ++g)
      scale[static_cast<long>(bh) * (n_rb * 4) + rb * 4 + g] = sc[g];

  const float inv_s[4] = {
      sc[0] == 0.0f ? 0.0f : 1.0f / sc[0], sc[1] == 0.0f ? 0.0f : 1.0f / sc[1],
      sc[2] == 0.0f ? 0.0f : 1.0f / sc[2], sc[3] == 0.0f ? 0.0f : 1.0f / sc[3]};
#pragma unroll
  for (int j = 0; j < n_chunks_per_thread; ++j) {
    const int r = chunk_r[j];
    const int c = chunk_c[j];
    const long row = row0 + r;
    if (row >= N)
      continue;
    const int g = (r % 8) / 2;
    uint2 out;
    QKOutT<kQKInt8>* ob = reinterpret_cast<QKOutT<kQKInt8>*>(&out);
#pragma unroll
    for (int e = 0; e < kVec; ++e) {
      float x = static_cast<float>(regs[j].elem[e]);
      if (smooth)
        x -= km_sh[c * kVec + e];
      if constexpr (kQKInt8)
        ob[e] = quant_sym_int8(x * inv_s[g]);
      else
        ob[e] = __nv_fp8_e4m3(x * inv_s[g]);
    }
    reinterpret_cast<uint2*>(y_bh + row * D)[c] = out;
  }
}

// Per-thread QK quantize launcher: Q 64 scale/block, K 4 scale/block.
// VT (V transposed) uses the regular per-block e4m3 quantize. Q/K quantize
// kernels are launched separately (no fused path). smooth-K applied to K.
template <typename Element, int kBr, int kBc, int kHeadDim, bool kQKInt8>
void launch_quantize_fp8_perthread_qk_sm120(
    const Element* q_ptr, const Element* k_ptr, const Element* v_ptr, void* q8,
    void* k8, __nv_fp8_e4m3* vt8, float* q_scale, float* k_scale,
    float* v_scale, int B, int Nh, int Nh_kv, int Nq, int Nkv, int Nkv_pad,
    cudaStream_t stream, const Element* km = nullptr) {
  using QKOut = QKOutT<kQKInt8>;
  // Q: 128-row blocks, 64 scale per block.
  {
    dim3 grid((Nq + 127) / 128, B * Nh);
    auto qk = quantize_fp8_perthread_q_kernel<Element, kHeadDim, kQKInt8>;
    qk<<<grid, 128, 0, stream>>>(q_ptr, reinterpret_cast<QKOut*>(q8), q_scale,
                                 Nq, Nh);
  }
  // K: kBr-col blocks (kBc for persist_d = 128), 4 scale per block.
  {
    dim3 grid((Nkv + kBc - 1) / kBc, B * Nh_kv);
    auto kk = quantize_fp8_perthread_k_kernel<Element, kBc, kHeadDim, kQKInt8>;
    kk<<<grid, 128, 0, stream>>>(k_ptr, reinterpret_cast<QKOut*>(k8), k_scale,
                                 Nkv, Nh_kv, km, 1.0f);
  }
  // VT: regular per-block e4m3 quantize (same as launch_quantize_fp8_sm120).
  {
    constexpr int kDChunk = (kHeadDim < 256) ? kHeadDim : 256;
    constexpr int kVtSmemBytes = kBc * (kDChunk + 16);
    dim3 grid((Nkv + kBc - 1) / kBc, B * Nh_kv);
    auto vk = quantize_fp8_vt_kernel<Element, kBc, kHeadDim>;
    cudaFuncSetAttribute(vk, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         kVtSmemBytes);
    vk<<<grid, 128, kVtSmemBytes, stream>>>(v_ptr, vt8, v_scale, Nkv, Nh_kv,
                                            Nkv_pad);
  }
}

// Quantize Q/K row-major and V transposed (D, N); scales per (bh, row-block).
// kQKInt8: Q/K are quantized to symmetric int8 (amax/127); VT stays e4m3.
// q8/k8 point to QKOutT<kQKInt8> buffers (passed as void* since the dtype is
// mode-dependent); smooth-K (km != nullptr): K is shifted by its per-(b,h)
// seq mean before quantizing; km holds the per-head mean vector (B*Nh_kv, D).
// inv_n scales km (pass 1.0 when km is already a mean).
template <typename Element, int kBr, int kBc, int kHeadDim, bool kQKInt8>
void launch_quantize_fp8_sm120(const Element* q_ptr, const Element* k_ptr,
                               const Element* v_ptr, void* q8, void* k8,
                               __nv_fp8_e4m3* vt8, float* q_scale,
                               float* k_scale, float* v_scale, int B, int Nh,
                               int Nh_kv, int Nq, int Nkv, int Nkv_pad, int D,
                               cudaStream_t stream,
                               const Element* km = nullptr) {
  using QKOut = QKOutT<kQKInt8>;
  constexpr int kThreads = 128;
  // Dynamic smem for the VT staging tile (1B/elem); must match the kernels'
  // kPad. D is tiled into kDChunk-wide columns so the staging tile fits the
  // 48KB static-smem cap even for large D (kBc*(kDChunk+16) <= ~34KB).
  constexpr int kDChunk = (kHeadDim < 256) ? kHeadDim : 256;
  constexpr int kVtSmemBytes = kBc * (kDChunk + 16);
  // Fused path: single launch for Q+K+VT when grids match (self-attention).
  if constexpr (kBr == kBc) {
    if (Nq == Nkv && Nh == Nh_kv) {
      dim3 grid((Nq + kBr - 1) / kBr, B * Nh, 3);
      auto kernel = quantize_fp8_qkv_fused_kernel<Element, kBr, kHeadDim,
                                                  kThreads, kQKInt8>;
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           kVtSmemBytes);
      kernel<<<grid, kThreads, kVtSmemBytes, stream>>>(
          q_ptr, k_ptr, v_ptr, reinterpret_cast<QKOut*>(q8),
          reinterpret_cast<QKOut*>(k8), vt8, q_scale, k_scale, v_scale, Nq, Nh,
          Nkv_pad, km, 1.0f);
      return;
    }
  }
  {
    dim3 grid_q((Nq + kBr - 1) / kBr, B * Nh);
    quantize_fp8_kernel<Element, kBr, kQKInt8, kHeadDim>
        <<<grid_q, kThreads, 0, stream>>>(q_ptr, reinterpret_cast<QKOut*>(q8),
                                          q_scale, Nq, Nh, nullptr, 0.0f);
  }
  {
    dim3 grid_k((Nkv + kBc - 1) / kBc, B * Nh_kv);
    quantize_fp8_kernel<Element, kBc, kQKInt8, kHeadDim>
        <<<grid_k, kThreads, 0, stream>>>(k_ptr, reinterpret_cast<QKOut*>(k8),
                                          k_scale, Nkv, Nh_kv, km, 1.0f);
  }
  {
    dim3 grid_kv((Nkv + kBc - 1) / kBc, B * Nh_kv);
    auto kernel = quantize_fp8_vt_kernel<Element, kBc, kHeadDim>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         kVtSmemBytes);
    kernel<<<grid_kv, kThreads, kVtSmemBytes, stream>>>(v_ptr, vt8, v_scale,
                                                        Nkv, Nh_kv, Nkv_pad);
  }
}

}  // namespace ffpa_fp8

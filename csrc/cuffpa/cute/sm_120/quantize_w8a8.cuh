#pragma once

#include <cuda_fp8.h>

namespace ffpa_cute {

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

// Row-major blockwise e4m3 quantization of fp16/bf16 Q/K for the W8A8
// persist-D kernel. Output keeps the (N, D) row-major layout; scales are
// amax/448 per (b, h, row-block). One block per (head, row-block).
//
// Requires D % 8 == 0. Memory accesses are fully vectorized: 128-bit (8-elem)
// loads, and 64-bit (8 x e4m3) stores whose 8B chunks never straddle a
// D-element row (D % 8 == 0), so every transaction is aligned and coalesced.
// The amax pass keeps the 8-elem chunks in registers, so the quantize pass
// converts straight from regs (no second global read).
template <typename Element, int kBlockRows>
__global__ void quantize_w8a8_kernel(
    const Element* __restrict__ X,  // (B, H, N, D) row-major
    __nv_fp8_e4m3* __restrict__ Y,  // (B, H, N, D)
    float* __restrict__ scale,      // (B, H, N/kBlockRows)
    int N, int H, int D, int ldy,
    const Element* __restrict__ km = nullptr,  // (B*H, D) smooth-K col means
    float inv_n = 0.0f) {
  constexpr int kVec = 8;  // elems per vector access (128-bit in, 64-bit out)
  constexpr int kThreads = 128;
  const int rb = blockIdx.x;
  const int bh = blockIdx.y;
  const int row0 = rb * kBlockRows;
  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid % 32;
  constexpr int kWarps = kThreads / 32;

  const int dv = D / kVec;            // vector chunks per row
  const int total = kBlockRows * dv;  // vector chunks per block
  const int n_chunks_per_thread = (total + kThreads - 1) / kThreads;

  const Element* x_bh = X + static_cast<long>(bh) * N * D;
  __nv_fp8_e4m3* y_bh = Y + static_cast<long>(bh) * N * D;

  // Smooth-K: cache this head's mean vector once (D floats, L1-hot reuse).
  __shared__ float km_sh[128];
  const bool smooth = km != nullptr;
  if (smooth && tid < D)
    km_sh[tid] =
        static_cast<float>(km[static_cast<long>(bh) * D + tid]) * inv_n;
  __syncthreads();

  using VecIn = Vec8<Element>;
  float amax = 0.0f;
  VecIn regs[8];  // kBlockRows*D/(kVec*kThreads) <= 8 for supported shapes
#pragma unroll 1
  for (int j = 0; j < n_chunks_per_thread; ++j) {
    const int i = tid + j * kThreads;
    if (i >= total)
      break;
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

  float s = amax / 448.0f;
  const float inv_s = (s == 0.0f) ? 0.0f : 1.0f / s;
  const int n_rb = (N + kBlockRows - 1) / kBlockRows;
  if (tid == 0)
    scale[static_cast<long>(bh) * n_rb + rb] = s;

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
    __nv_fp8_e4m3* ob = reinterpret_cast<__nv_fp8_e4m3*>(&out);
#pragma unroll
    for (int e = 0; e < kVec; ++e) {
      float x = static_cast<float>(v.elem[e]);
      if (smooth)
        x -= km_sh[c * kVec + e];
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
__global__ void quantize_w8a8_vt_kernel(
    const Element* __restrict__ V,   // (B, H, N, D) row-major
    __nv_fp8_e4m3* __restrict__ VT,  // flat [B*Nh_kv*D, Nkv_pad]
    float* __restrict__ scale,       // (B*Nh_kv, Nkv/kBlockRows)
    int N, int Nh_kv, int ldy) {
  constexpr int D = kD;
  constexpr int kPad = 16;
  constexpr int kVec = 8;
  constexpr int kThreads = 128;
  static_assert(D % kVec == 0, "vt kernel requires D % 8 == 0");
  const int rb = blockIdx.x;
  const int bh = blockIdx.y;
  const int row0 = rb * kBlockRows;
  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid % 32;
  constexpr int kWarps = kThreads / 32;

  __shared__ __nv_fp8_e4m3 tile[kBlockRows][D + kPad];
  __shared__ float warp_max[kWarps];

  const int dv = D / kVec;            // vector chunks per row
  const int total = kBlockRows * dv;  // vector chunks per tile
  const int n_chunks_per_thread = total / kThreads;

  const Element* v_bh = V + static_cast<long>(bh) * N * D;
  using VecIn = Vec8<Element>;

  // Pass 1: vectorized coalesced read, track amax.
  float amax = 0.0f;
#pragma unroll 1
  for (int j = 0; j < n_chunks_per_thread; ++j) {
    const int i = tid + j * kThreads;
    const int r = i / dv;
    const int c = i % dv;
    VecIn v = Vec8<Element>::zero();
    if (row0 + r < N)
      v = reinterpret_cast<const VecIn*>(v_bh +
                                         static_cast<long>(row0 + r) * D)[c];
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

    // Pass 2: vectorized re-read (L2-hot), quantize into smem as 2x uint32.
#pragma unroll 1
  for (int j = 0; j < n_chunks_per_thread; ++j) {
    const int i = tid + j * kThreads;
    const int r = i / dv;
    const int c = i % dv;
    VecIn v = Vec8<Element>::zero();
    if (row0 + r < N)
      v = reinterpret_cast<const VecIn*>(v_bh +
                                         static_cast<long>(row0 + r) * D)[c];
    uint32_t pack[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      __nv_fp8_e4m3* ob = reinterpret_cast<__nv_fp8_e4m3*>(&pack[h]);
#pragma unroll
      for (int e = 0; e < 4; ++e)
        ob[e] = __nv_fp8_e4m3(static_cast<float>(v.elem[h * 4 + e]) * inv_s);
    }
    uint32_t* dst = reinterpret_cast<uint32_t*>(&tile[r][c * kVec]);
    dst[0] = pack[0];
    dst[1] = pack[1];
  }
  __syncthreads();

  // Pass 3: coalesced transpose write. Each warp writes 4 consecutive output
  // rows c..c+3; lanes sweep the contiguous kv index n. Per-element stores:
  // a wider pack would straddle Nkv-length rows of a neighbouring head plane.
  const long vt_base = static_cast<long>(bh) * D * ldy;
#pragma unroll 1
  for (int it = warp * 4; it < D; it += kWarps * 4) {
#pragma unroll
    for (int cc = 0; cc < 4; ++cc) {
      const int c = it + cc;
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        const int n = t * 32 + lane;  // local kv index within the block
        if (row0 + n < N)
          VT[vt_base + static_cast<long>(c) * ldy + row0 + n] = tile[n][c];
      }
    }
  }
}

// Fused QKV quantize: one launch handles Q (row-major), K (row-major), and
// V (transposed) via blockIdx.z role selection. Requires kBr == kBc and
// self-attention (Nq == Nkv, Nh == Nh_kv) so all three share the same grid
// dimensions. Falls back to separate launches otherwise.
template <typename Element, int kBlockRows, int kD, int kThreads>
__global__ void quantize_w8a8_qkv_fused_kernel(
    const Element* __restrict__ Q, const Element* __restrict__ K,
    const Element* __restrict__ V, __nv_fp8_e4m3* __restrict__ Q8,
    __nv_fp8_e4m3* __restrict__ K8, __nv_fp8_e4m3* __restrict__ VT8,
    float* __restrict__ q_scale, float* __restrict__ k_scale,
    float* __restrict__ v_scale, int N, int H, int ldy,
    const Element* __restrict__ km = nullptr,  // (B*H, D) smooth-K col means
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

  const int dv = kD / kVec;
  const int total = kBlockRows * dv;
  const int n_chunks_per_thread = (total + kThreads - 1) / kThreads;
  const int n_rb = (N + kBlockRows - 1) / kBlockRows;

  __shared__ float warp_max[kWarps];
  // Smooth-K applies to K only (role==1); cache the head's mean vector once.
  __shared__ float km_sh[kD];
  const bool smooth = (role == 1) && (km != nullptr);
  if (smooth && tid < kD)
    km_sh[tid] =
        static_cast<float>(km[static_cast<long>(bh) * kD + tid]) * inv_n;

  if (role < 2) {
    const Element* X = (role == 0) ? Q : K;
    __nv_fp8_e4m3* Y = (role == 0) ? Q8 : K8;
    float* scale_out = (role == 0) ? q_scale : k_scale;
    const Element* x_bh = X + static_cast<long>(bh) * N * kD;
    __nv_fp8_e4m3* y_bh = Y + static_cast<long>(bh) * N * kD;
    if (smooth)
      __syncthreads();

    using VecIn = Vec8<Element>;
    float amax = 0.0f;
    VecIn regs[8];
#pragma unroll 1
    for (int j = 0; j < n_chunks_per_thread; ++j) {
      const int i = tid + j * kThreads;
      if (i >= total)
        break;
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

    float s = amax / 448.0f;
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
      __nv_fp8_e4m3* ob = reinterpret_cast<__nv_fp8_e4m3*>(&out);
#pragma unroll
      for (int e = 0; e < kVec; ++e) {
        float x = static_cast<float>(v.elem[e]);
        if (smooth)
          x -= km_sh[c * kVec + e];
        ob[e] = __nv_fp8_e4m3(x * inv_s);
      }
      reinterpret_cast<uint2*>(y_bh + row * kD)[c] = out;
    }
  } else {
    // V transpose quantize through smem.
    __shared__ __nv_fp8_e4m3 tile[kBlockRows][kD + kPad];
    const Element* v_bh = V + static_cast<long>(bh) * N * kD;
    using VecIn = Vec8<Element>;

    float amax = 0.0f;
#pragma unroll 1
    for (int j = 0; j < n_chunks_per_thread; ++j) {
      const int i = tid + j * kThreads;
      if (i >= total)
        break;
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

#pragma unroll 1
    for (int j = 0; j < n_chunks_per_thread; ++j) {
      const int i = tid + j * kThreads;
      if (i >= total)
        break;
      const int r = i / dv;
      const int c = i % dv;
      VecIn v = Vec8<Element>::zero();
      if (row0 + r < N)
        v = reinterpret_cast<const VecIn*>(v_bh +
                                           static_cast<long>(row0 + r) * kD)[c];
      uint32_t pack[2];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        __nv_fp8_e4m3* ob = reinterpret_cast<__nv_fp8_e4m3*>(&pack[h]);
#pragma unroll
        for (int e = 0; e < 4; ++e)
          ob[e] = __nv_fp8_e4m3(static_cast<float>(v.elem[h * 4 + e]) * inv_s);
      }
      uint32_t* dst = reinterpret_cast<uint32_t*>(&tile[r][c * kVec]);
      dst[0] = pack[0];
      dst[1] = pack[1];
    }
    __syncthreads();

    const long vt_base = static_cast<long>(bh) * kD * ldy;
#pragma unroll 1
    for (int it = warp * 4; it < kD; it += kWarps * 4) {
#pragma unroll
      for (int cc = 0; cc < 4; ++cc) {
        const int c = it + cc;
#pragma unroll
        for (int t = 0; t < 4; ++t) {
          const int n = t * 32 + lane;
          if (row0 + n < N)
            VT8[vt_base + static_cast<long>(c) * ldy + row0 + n] = tile[n][c];
        }
      }
    }
  }
}

// Quantize Q/K row-major and V transposed (D, N); scales per (bh, row-block).
// smooth-K (km != nullptr): K is shifted by its per-(b,h) seq mean before
// quantizing; km holds the per-head mean vector (B*Nh_kv, D). inv_n scales km
// (pass 1.0 when km is already a mean).
template <typename Element, int kBr, int kBc, int kHeadDim>
void launch_quantize_w8a8_sm120(const Element* q_ptr, const Element* k_ptr,
                                const Element* v_ptr, __nv_fp8_e4m3* q8,
                                __nv_fp8_e4m3* k8, __nv_fp8_e4m3* vt8,
                                float* q_scale, float* k_scale, float* v_scale,
                                int B, int Nh, int Nh_kv, int Nq, int Nkv,
                                int Nkv_pad, int D, cudaStream_t stream,
                                const Element* km = nullptr) {
  constexpr int kThreads = 128;
  // Fused path: single launch for Q+K+VT when grids match (self-attention).
  if constexpr (kBr == kBc) {
    if (Nq == Nkv && Nh == Nh_kv) {
      dim3 grid((Nq + kBr - 1) / kBr, B * Nh, 3);
      quantize_w8a8_qkv_fused_kernel<Element, kBr, kHeadDim, kThreads>
          <<<grid, kThreads, 0, stream>>>(q_ptr, k_ptr, v_ptr, q8, k8, vt8,
                                          q_scale, k_scale, v_scale, Nq, Nh,
                                          Nkv_pad, km, 1.0f);
      return;
    }
  }
  dim3 grid_q((Nq + kBr - 1) / kBr, B * Nh);
  quantize_w8a8_kernel<Element, kBr><<<grid_q, kThreads, 0, stream>>>(
      q_ptr, q8, q_scale, Nq, Nh, D, D, nullptr, 0.0f);
  dim3 grid_kv((Nkv + kBc - 1) / kBc, B * Nh_kv);
  quantize_w8a8_kernel<Element, kBc><<<grid_kv, kThreads, 0, stream>>>(
      k_ptr, k8, k_scale, Nkv, Nh_kv, D, D, km, 1.0f);
  quantize_w8a8_vt_kernel<Element, kBc, kHeadDim>
      <<<grid_kv, kThreads, 0, stream>>>(v_ptr, vt8, v_scale, Nkv, Nh_kv,
                                         Nkv_pad);
}

}  // namespace ffpa_cute

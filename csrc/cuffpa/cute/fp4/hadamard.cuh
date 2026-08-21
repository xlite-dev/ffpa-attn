// Walsh-Hadamard pre-rotation for the NVFP4 QK path: rotated Q/K copies
// are fed to the standard preprocessing chain (km/qm/quant/delta_s/lse).
// H is orthogonal, so Q H H^T K^T == Q K^T exactly; the rotation only
// moves where fp4 quantization noise lands (flattens per-16-group outlier
// amplitudes). Width rule: full-width WHT for pow2 D <= 512, blockdiag
// H_64 otherwise (fp4 D is always a 64-multiple). The rotated copy is
// kHeadDim-wide: cols >= d_og load as zeros and their rotated values are
// STORED, so downstream full-width reads (d_og becomes kHeadDim) keep the
// contraction exact instead of dropping energy into dead pad cols.
#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/all.h>

namespace ffpa_fp4 {

// One warp per (row, kWhtWidth-block); lane l owns cols {base + l + 32*j}.
// Butterfly distances >= 32 swap register slots in-lane (bit lives in j),
// distances < 32 use __shfl_xor_sync (bit lives in the lane id). fp32
// accumulation, scale 1/sqrt(kWhtWidth), stores cover the whole block
// including rotated pad cols.
template <typename T, int kWhtWidth>
__global__ void wht_qk_kernel(const T* __restrict__ input,
                              T* __restrict__ output, long num_rows, int d_og,
                              long in_row_stride, long out_row_stride,
                              int blocks_per_row) {
  constexpr int kVec = kWhtWidth / 32;
  const long task =
      (long)blockIdx.x * (blockDim.x >> 5) + (long)(threadIdx.x >> 5);
  if (task >= num_rows * blocks_per_row)
    return;
  const long row = task / blocks_per_row;
  const int col_base = (int)(task % blocks_per_row) * kWhtWidth;
  const int lane = threadIdx.x & 31;

  float x[kVec];
  const T* src = input + row * in_row_stride + col_base;
#pragma unroll
  for (int j = 0; j < kVec; ++j) {
    const int col = lane + 32 * j;
    x[j] = col_base + col < d_og ? (float)src[col] : 0.f;
  }
#pragma unroll
  for (int dist = kWhtWidth >> 1; dist >= 1; dist >>= 1) {
    if (dist >= 32) {
#pragma unroll
      for (int j = 0; j < kVec; ++j) {
        const int jj = j ^ (dist >> 5);
        if (j < jj) {
          const float a = x[j], b = x[jj];
          x[j] = a + b;
          x[jj] = a - b;
        }
      }
    } else {
#pragma unroll
      for (int j = 0; j < kVec; ++j) {
        const float other = __shfl_xor_sync(0xffffffffu, x[j], dist);
        x[j] = (lane & dist) ? other - x[j] : x[j] + other;
      }
    }
  }
  const float s = rsqrtf((float)kWhtWidth);
  T* dst = output + row * out_row_stride + col_base;
#pragma unroll
  for (int j = 0; j < kVec; ++j)
    dst[lane + 32 * j] = (T)(x[j] * s);
}

namespace detail {

template <typename T, int kWhtWidth, int kHeadDim>
void launch_wht_qk_t(const torch::Tensor& input, torch::Tensor& output) {
  constexpr int kBlocksPerRow = kHeadDim / kWhtWidth;
  constexpr int kWarpsPerBlock = 4;
  const long rows = input.numel() / input.size(3);
  const int d_og = static_cast<int>(input.size(3));
  const long blocks =
      (rows * kBlocksPerRow + kWarpsPerBlock - 1) / kWarpsPerBlock;
  TORCH_CHECK(blocks <= 2147483647L, "ffpa_attn: wht_qk grid overflow");
  if (blocks == 0)
    return;
  auto stream = at::cuda::getCurrentCUDAStream();
  wht_qk_kernel<T, kWhtWidth>
      <<<(unsigned int)blocks, kWarpsPerBlock * 32, 0, stream>>>(
          reinterpret_cast<const T*>(input.data_ptr()),
          reinterpret_cast<T*>(output.data_ptr()), rows, d_og,
          input.size(3),  // contiguous: row stride == d_og
          kHeadDim, kBlocksPerRow);
}

}  // namespace detail

// Returns the kHeadDim-wide rotated copy of a (B,H,N,d_og) contiguous
// Q/K tensor (d_og <= kHeadDim, d_og % 8 == 0).
template <typename T, int kHeadDim>
torch::Tensor apply_wht_qk_sm120(const torch::Tensor& input) {
  TORCH_CHECK(input.is_contiguous(),
              "ffpa_attn: fp4 hadamard requires contiguous Q/K");
  TORCH_CHECK(
      input.size(3) % 8 == 0 && input.size(3) <= kHeadDim,
      "ffpa_attn: fp4 hadamard requires head_dim%8==0 and <= ", kHeadDim);
  constexpr bool kPow2 = (kHeadDim & (kHeadDim - 1)) == 0;
  constexpr int kWhtWidth = (kPow2 && kHeadDim <= 512) ? kHeadDim : 64;
  torch::Tensor out = torch::empty(
      {input.size(0), input.size(1), input.size(2), kHeadDim}, input.options());
  detail::launch_wht_qk_t<T, kWhtWidth, kHeadDim>(input, out);
  return out;
}

}  // namespace ffpa_fp4

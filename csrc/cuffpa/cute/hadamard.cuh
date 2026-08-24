// Walsh-Hadamard pre-rotation for the quantized QK paths (fp8 & fp4):
// rotated Q/K copies are fed to the standard preprocessing chain.
//
// Math: the (unnormalized) Walsh-Hadamard matrix is defined recursively,
//   H_1 = [1],   H_{2n} = | H_n  H_n |
//                         | H_n -H_n |
// Entries are +-1; the normalized form Hhat = H_n / sqrt(n) is orthogonal
// (Hhat Hhat^T = I), so (Q Hhat)(K Hhat)^T = Q K^T exactly - softmax
// logits are unchanged. The kernels below apply Hhat to each row via the
// radix-2 butterfly ((a,b) -> (a+b, a-b), one pass per bit, no
// bit-reversal) with the 1/sqrt(n) scale folded into the store.
//
// Why rotate at all: the rotation whitens each row - every output
// coordinate is the signed sum of ALL inputs, so energy spreads uniformly
// across coords (Hadamard "Jackson" property). The quantizers share one
// scale per element block (per 16 for fp4, per block/group for fp8);
// without rotation a single outlier coordinate dominates its block scale
// and crushes the other values, while rotated rows have near-uniform
// amplitudes -> block scales concentrate -> quantization noise drops.
// Same idea as FlashAttention-3's incoherent processing
// (arXiv:2407.08608 Sec 3.3: randomized +/-1 diagonal x Hadamard fused
// into RoPE; up to 2.6x lower FP8 RMSE jointly with block quantization).
//
// Width rule: full-width WHT for pow2 D <= 512, blockdiag H_64 otherwise
// (fp4 D is always a 64-multiple; the blockdiag fallback needs
// kHeadDim % 64 == 0). The rotated copy is kHeadDim-wide: cols >= d_og
// load as zeros and their rotated values are STORED, so downstream
// full-width reads (d_og becomes kHeadDim) keep the contraction exact
// instead of dropping energy into dead pad cols.
#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/all.h>

namespace ffpa {

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
              "ffpa_attn: hadamard requires contiguous Q/K");
  TORCH_CHECK(input.size(3) % 8 == 0 && input.size(3) <= kHeadDim,
              "ffpa_attn: hadamard requires head_dim%8==0 and <= ", kHeadDim);
  constexpr bool kPow2 = (kHeadDim & (kHeadDim - 1)) == 0;
  constexpr int kWhtWidth = (kPow2 && kHeadDim <= 512) ? kHeadDim : 64;
  static_assert(kHeadDim % kWhtWidth == 0,
                "blockdiag H_64 needs kHeadDim % 64 == 0");
  torch::Tensor out = torch::empty(
      {input.size(0), input.size(1), input.size(2), kHeadDim}, input.options());
  detail::launch_wht_qk_t<T, kWhtWidth, kHeadDim>(input, out);
  return out;
}

// fp32 row WHT for the pre-rotation of small mean vectors (qm / km) in the
// fused-hadamard path: WHT is linear so mean(X H) == mean(X) H, which lets
// the mean kernels keep running unrotated while the quantize bias needs
// the rotated copy. One warp per row; rows are kWidth wide and fully
// valid (pad cols already zeroed by the producers).
template <int kWidth>
__global__ void wht_f32_rows_kernel(const float* __restrict__ input,
                                    float* __restrict__ output, long num_rows) {
  constexpr int kVec = kWidth / 32;
  const long row = (long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (row >= num_rows)
    return;
  const int lane = threadIdx.x & 31;
  const float* src = input + row * kWidth;
  float x[kVec];
#pragma unroll
  for (int j = 0; j < kVec; ++j)
    x[j] = src[lane + 32 * j];
#pragma unroll
  for (int dist = kWidth >> 1; dist >= 1; dist >>= 1) {
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
  const float s = rsqrtf((float)kWidth);
  float* dst = output + row * kWidth;
#pragma unroll
  for (int j = 0; j < kVec; ++j)
    dst[lane + 32 * j] = x[j] * s;
}

// Returns the WHT-rotated copy of a contiguous fp32 tensor whose last dim
// is kWidth (qm blocks / km heads). Feed-forward only: tiny traffic.
template <int kWidth>
torch::Tensor apply_wht_f32_rows_sm120(const torch::Tensor& input) {
  TORCH_CHECK(
      input.is_contiguous() && input.scalar_type() == at::ScalarType::Float,
      "ffpa_attn: wht_f32_rows requires contiguous fp32 input");
  TORCH_CHECK(input.size(-1) == kWidth,
              "ffpa_attn: wht_f32_rows last dim must be ", kWidth);
  const long num_rows = input.numel() / kWidth;
  torch::Tensor out = torch::empty_like(input);
  if (num_rows == 0)
    return out;
  constexpr int kWarpsPerBlock = 8;
  const long blocks = (num_rows + kWarpsPerBlock - 1) / kWarpsPerBlock;
  auto stream = at::cuda::getCurrentCUDAStream();
  wht_f32_rows_kernel<kWidth>
      <<<(unsigned int)blocks, kWarpsPerBlock * 32, 0, stream>>>(
          input.data_ptr<float>(), out.data_ptr<float>(), num_rows);
  return out;
}

}  // namespace ffpa

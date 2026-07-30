#pragma once

#include "prefill.cuh"

namespace ffpa_cute {

// Online safe softmax (no bias, no dropout). Baseline hot path.
template <typename ScoresTensor, typename CoordTensor, int kRows>
__device__ __forceinline__ void online_safe_softmax(ScoresTensor& scores,
                                                    const CoordTensor& tScS_rc,
                                                    float scale, float* row_max,
                                                    float* row_sum,
                                                    float* row_scale) {
#pragma unroll
  for (int row = 0; row < kRows; ++row) {
    float tile_max = -INFINITY;
#pragma unroll
    for (int col = 0; col < cute::size<1>(scores); ++col)
      tile_max = fmaxf(tile_max, scores(row, col) * scale);
    tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 1));
    tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 2));
    const float next_max = fmaxf(row_max[row], tile_max);
    row_scale[row] = exp2f(row_max[row] - next_max);
    float tile_sum = 0.0f;
#pragma unroll
    for (int col = 0; col < cute::size<1>(scores); ++col) {
      const float p = exp2f(scores(row, col) * scale - next_max);
      scores(row, col) = p;
      tile_sum += p;
    }
    tile_sum += __shfl_xor_sync(0xffffffff, tile_sum, 1);
    tile_sum += __shfl_xor_sync(0xffffffff, tile_sum, 2);
    row_sum[row] = row_sum[row] * row_scale[row] + tile_sum;
    row_max[row] = next_max;
  }
}

// Online softmax with additive bias fused into the row-max pass.
template <typename ScoresTensor, typename CoordTensor, int kRows>
__device__ __forceinline__ void online_softmax_bias(
    ScoresTensor& scores, const CoordTensor& tScS_rc, float scale,
    float* row_max, float* row_sum, float* row_scale,
    const void* __restrict__ attn_bias, int attn_bias_dtype, int stride_b,
    int stride_h, int stride_m, int stride_n, int Nb_id, int Nh_id, int Br_base,
    int kv_tile, int kBc, float inv_scale) {
  const int bias_base = Nb_id * stride_b + Nh_id * stride_h;
  const int bc_base = kv_tile * kBc;
#pragma unroll
  for (int row = 0; row < kRows; ++row) {
    const int q_row = Br_base + cute::get<0>(tScS_rc(row, 0));
    const int row_off = bias_base + q_row * stride_m;
    float tile_max = -INFINITY;
#pragma unroll
    for (int col = 0; col < cute::size<1>(scores); ++col) {
      const int k_col = bc_base + cute::get<1>(tScS_rc(row, col));
      scores(row, col) += ffpa::prefill::load_attn_bias_value(
                              attn_bias, attn_bias_dtype,
                              (long long)(row_off + k_col * stride_n)) *
                          inv_scale;
      tile_max = fmaxf(tile_max, scores(row, col) * scale);
    }
    tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 1));
    tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 2));
    const float next_max = fmaxf(row_max[row], tile_max);
    row_scale[row] = exp2f(row_max[row] - next_max);
    float tile_sum = 0.0f;
#pragma unroll
    for (int col = 0; col < cute::size<1>(scores); ++col) {
      const float p = exp2f(scores(row, col) * scale - next_max);
      scores(row, col) = p;
      tile_sum += p;
    }
    tile_sum += __shfl_xor_sync(0xffffffff, tile_sum, 1);
    tile_sum += __shfl_xor_sync(0xffffffff, tile_sum, 2);
    row_sum[row] = row_sum[row] * row_scale[row] + tile_sum;
    row_max[row] = next_max;
  }
}

// Online softmax with dropout fused into the exp2 pass.
template <typename ScoresTensor, typename CoordTensor, int kRows>
__device__ __forceinline__ void online_softmax_dropout(
    ScoresTensor& scores, const CoordTensor& tScS_rc, float scale,
    float* row_max, float* row_sum, float* row_scale, float dropout_p,
    unsigned long long philox_seed, unsigned long long philox_offset, int Nb_id,
    int Nh, int Nh_id, int Nq, int Nkv, int Br_base, int kv_tile, int kBc) {
  const float keep_scale = 1.0f / (1.0f - dropout_p);
  const int head_base = (Nb_id * Nh + Nh_id) * Nq;
  const int bc_base = kv_tile * kBc;
#pragma unroll
  for (int row = 0; row < kRows; ++row) {
    float tile_max = -INFINITY;
#pragma unroll
    for (int col = 0; col < cute::size<1>(scores); ++col)
      tile_max = fmaxf(tile_max, scores(row, col) * scale);
    tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 1));
    tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 2));
    const float next_max = fmaxf(row_max[row], tile_max);
    row_scale[row] = exp2f(row_max[row] - next_max);
    const int q_row = Br_base + cute::get<0>(tScS_rc(row, 0));
    const int row_off = (head_base + q_row) * Nkv;
    float tile_sum = 0.0f;
#pragma unroll
    for (int col = 0; col < cute::size<1>(scores); ++col) {
      float p = exp2f(scores(row, col) * scale - next_max);
      const int k_col = bc_base + cute::get<1>(tScS_rc(row, col));
      const unsigned long long off =
          philox_offset + (unsigned long long)(row_off + k_col);
      const float u =
          ffpa::prefill::curand_uniform_from_element_offset(philox_seed, off);
      p = (u > dropout_p) ? p * keep_scale : 0.0f;
      scores(row, col) = p;
      tile_sum += p;
    }
    tile_sum += __shfl_xor_sync(0xffffffff, tile_sum, 1);
    tile_sum += __shfl_xor_sync(0xffffffff, tile_sum, 2);
    row_sum[row] = row_sum[row] * row_scale[row] + tile_sum;
    row_max[row] = next_max;
  }
}

}  // namespace ffpa_cute

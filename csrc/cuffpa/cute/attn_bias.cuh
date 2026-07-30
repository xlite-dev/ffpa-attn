#pragma once

#include "prefill.cuh"

namespace ffpa_cute {

// Additive attention bias on rowcol scores (pre-softmax).
// bias is added as: scores += load(bias[q_row, k_col]) * inv_scale
template <typename ScoresTensor, typename CoordTensor, int kRows, int kCols>
__device__ __forceinline__ void apply_attn_bias_rowcol(
    ScoresTensor& scores, const CoordTensor& tScS_rc,
    const void* __restrict__ attn_bias, int attn_bias_dtype, long long stride_b,
    long long stride_h, long long stride_m, long long stride_n, int Nb_id,
    int Nh_id, int Br_base, int kv_tile, int kBc, float inv_scale) {
  const long long bias_base =
      (long long)Nb_id * stride_b + (long long)Nh_id * stride_h;
#pragma unroll
  for (int row = 0; row < kRows; ++row) {
    const int q_row = Br_base + cute::get<0>(tScS_rc(row, 0));
    const long long row_off = bias_base + (long long)q_row * stride_m;
#pragma unroll
    for (int col = 0; col < kCols; ++col) {
      const int k_col = kv_tile * kBc + cute::get<1>(tScS_rc(row, col));
      const long long offset = row_off + (long long)k_col * stride_n;
      scores(row, col) += ffpa::prefill::load_attn_bias_value(
                              attn_bias, attn_bias_dtype, offset) *
                          inv_scale;
    }
  }
}

}  // namespace ffpa_cute

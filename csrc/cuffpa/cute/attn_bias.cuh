#pragma once

#include "prefill.cuh"

namespace ffpa_cute {

// Additive attention bias on rowcol scores (pre-softmax).
// Partial unroll to avoid icache pressure from 256 scalar loads.
template <typename ScoresTensor, typename CoordTensor, int kRows, int kCols>
__device__ __forceinline__ void apply_attn_bias_rowcol(
    ScoresTensor& scores, const CoordTensor& tScS_rc,
    const void* __restrict__ attn_bias, int attn_bias_dtype, int stride_b,
    int stride_h, int stride_m, int stride_n, int Nb_id, int Nh_id, int Br_base,
    int kv_tile, int kBc, float inv_scale) {
  const int bias_base = Nb_id * stride_b + Nh_id * stride_h;
  const int bc_base = kv_tile * kBc;
#pragma unroll 1
  for (int row = 0; row < kRows; ++row) {
    const int q_row = Br_base + cute::get<0>(tScS_rc(row, 0));
    const int row_off = bias_base + q_row * stride_m;
#pragma unroll 1
    for (int col = 0; col < kCols; ++col) {
      const int k_col = bc_base + cute::get<1>(tScS_rc(row, col));
      scores(row, col) += ffpa::prefill::load_attn_bias_value(
                              attn_bias, attn_bias_dtype,
                              (long long)(row_off + k_col * stride_n)) *
                          inv_scale;
    }
  }
}

}  // namespace ffpa_cute

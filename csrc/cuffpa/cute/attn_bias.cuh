#pragma once

#include "../native/prefill.cuh"

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
#pragma unroll
  for (int row = 0; row < kRows; ++row) {
    const int q_row = Br_base + cute::get<0>(tScS_rc(row, 0));
    const int row_off = bias_base + q_row * stride_m;
#pragma unroll
    for (int col = 0; col < kCols; ++col) {
      const int k_col = bc_base + cute::get<1>(tScS_rc(row, col));
      scores(row, col) += ffpa::prefill::load_attn_bias_value(
                              attn_bias, attn_bias_dtype,
                              (long long)(row_off + k_col * stride_n)) *
                          inv_scale;
    }
  }
}

// Additive attention bias for the fp8 quant kernels, injected in the RAW
// score domain (right after the QK GEMM, before any dequant-scale path).
// Every downstream path scales the raw scores by qs_arr[row]*ks*scale
// (LOG2E folded into scale), so bias*inv_sd[row] with inv_sd[row] =
// 1/(qs_arr[row]*ks*scale_orig) lands as +bias in softmax-input units on
// masked and unmasked tiles alike. q_row_base = q_start_row + Br_base is
// the absolute query row (split/hybrid launches offset rows); kv columns
// are in natural order on the fp8 kernels.
template <typename ScoresTensor, typename CoordTensor, int kRows, int kCols>
__device__ __forceinline__ void apply_attn_bias_quant_rowcol(
    ScoresTensor& scores, const CoordTensor& tScS_rc,
    const void* __restrict__ attn_bias, int attn_bias_dtype, long long stride_b,
    long long stride_h, long long stride_m, long long stride_n, int Nb_id,
    int Nh_id, int q_row_base, int kv_tile, int kBc,
    const float (&inv_sd)[kRows]) {
  const long long bias_base =
      (long long)Nb_id * stride_b + (long long)Nh_id * stride_h;
  const int bc_base = kv_tile * kBc;
#pragma unroll
  for (int row = 0; row < kRows; ++row) {
    const int q_row = q_row_base + cute::get<0>(tScS_rc(row, 0));
    const long long row_off = bias_base + (long long)q_row * stride_m;
    const float inv = inv_sd[row];
#pragma unroll
    for (int col = 0; col < kCols; ++col) {
      const int k_col = bc_base + cute::get<1>(tScS_rc(row, col));
      scores(row, col) += ffpa::prefill::load_attn_bias_value(
                              attn_bias, attn_bias_dtype,
                              row_off + (long long)k_col * stride_n) *
                          inv;
    }
  }
}

}  // namespace ffpa_cute

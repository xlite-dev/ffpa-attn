#pragma once

#include "prefill.cuh"

namespace ffpa_cute {

// Dropout on P (post-softmax, pre-PV). SDPA semantics: mask/scale P
// without modifying row_sum (normalization happens in epilogue).
template <typename ScoresTensor, typename CoordTensor, int kRows, int kCols>
__device__ __forceinline__ void apply_dropout_rowcol(
    ScoresTensor& scores, const CoordTensor& tScS_rc, float dropout_p,
    unsigned long long philox_seed, unsigned long long philox_offset, int Nb_id,
    int Nh, int Nh_id, int Nq, int Nkv, int Br_base, int kv_tile, int kBc) {
  const float keep_scale = 1.0f / (1.0f - dropout_p);
  const unsigned long long head_base =
      (unsigned long long)(Nb_id * Nh + Nh_id) * Nq;
#pragma unroll
  for (int row = 0; row < kRows; ++row) {
    const int q_row = Br_base + cute::get<0>(tScS_rc(row, 0));
    const unsigned long long row_off = (head_base + q_row) * Nkv;
#pragma unroll
    for (int col = 0; col < kCols; ++col) {
      const int k_col = kv_tile * kBc + cute::get<1>(tScS_rc(row, col));
      const unsigned long long elem_off = row_off + k_col;
      const float uniform = ffpa::prefill::curand_uniform_from_element_offset(
          philox_seed, philox_offset + elem_off);
      scores(row, col) =
          (uniform > dropout_p) ? scores(row, col) * keep_scale : 0.0f;
    }
  }
}

}  // namespace ffpa_cute

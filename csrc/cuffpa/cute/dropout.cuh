#pragma once

#include "../native/prefill.cuh"

namespace ffpa_cute {

// Dropout on P (post-softmax, pre-PV). SDPA semantics: mask/scale P
// without modifying row_sum (normalization happens in epilogue).
// Partial unroll + paired Philox to reduce icache/register pressure.
template <typename ScoresTensor, typename CoordTensor, int kRows, int kCols>
__device__ __forceinline__ void apply_dropout_rowcol(
    ScoresTensor& scores, const CoordTensor& tScS_rc, float dropout_p,
    unsigned long long philox_seed, unsigned long long philox_offset, int Nb_id,
    int Nh, int Nh_id, int Nq, int Nkv, int Br_base, int kv_tile, int kBc) {
  const float keep_scale = 1.0f / (1.0f - dropout_p);
  const unsigned long long head_base =
      (static_cast<unsigned long long>(Nb_id) * Nh + Nh_id) * Nq;
  const int bc_base = kv_tile * kBc;
#pragma unroll
  for (int row = 0; row < kRows; ++row) {
    const int q_row = Br_base + cute::get<0>(tScS_rc(row, 0));
    const unsigned long long row_off = (head_base + q_row) * Nkv;
#pragma unroll
    for (int col = 0; col < kCols; col += 2) {
      const int k0 = bc_base + cute::get<1>(tScS_rc(row, col));
      const unsigned long long off0 = philox_offset + row_off + k0;
      const uint4 rng = ffpa::prefill::philox4x32_10(philox_seed, off0 >> 2);
      const unsigned lane0 = (unsigned)(off0 & 3);
      const float u0 = ffpa::prefill::uniform_from_philox_uint(
          ffpa::prefill::select_philox_lane(rng, lane0));
      float u1;
      if (lane0 == 3u) {
        const unsigned long long off1 = off0 + 1;
        u1 = ffpa::prefill::uniform_from_philox_uint(
            ffpa::prefill::select_philox_lane(
                ffpa::prefill::philox4x32_10(philox_seed, off1 >> 2),
                (unsigned)(off1 & 3)));
      } else {
        u1 = ffpa::prefill::uniform_from_philox_uint(
            ffpa::prefill::select_philox_lane(rng, lane0 + 1));
      }
      scores(row, col) =
          (u0 > dropout_p) ? scores(row, col) * keep_scale : 0.0f;
      scores(row, col + 1) =
          (u1 > dropout_p) ? scores(row, col + 1) * keep_scale : 0.0f;
    }
  }
}

}  // namespace ffpa_cute

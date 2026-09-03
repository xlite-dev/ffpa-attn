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

// PC-14 consumer-side keep-bitmap generation, one half-row per call (two
// threads cover a row: thread row*2+half owns columns [half*kBc/2,
// (half+1)*kBc/2)). The decision for element offset e stays
// uniform(philox(seed, e>>2)[e&3]) > dropout_p — the exact query the inline
// path makes — but is iterated on offset-quads: one philox4x32_10 call
// yields up to 4 decisions (half the philox volume of the inline path's
// column-pair loop, which recomputes each quad's philox twice). Bitmap
// word: bitmap[row*kBc/32 + col/32], bit (col & 31).
template <int kBc>
__device__ __forceinline__ void generate_dropout_bitmap_halfrow(
    uint32_t* __restrict__ bitmap, int row, int half, int q_row, int kv_tile,
    float dropout_p, unsigned long long philox_seed,
    unsigned long long philox_offset, unsigned long long head_base, int Nkv) {
  constexpr int kHalf = kBc / 2;
  constexpr int kHalfWords = kBc / 64;
  const unsigned long long base = philox_offset + (head_base + q_row) * Nkv +
                                  (unsigned long long)kv_tile * kBc +
                                  (unsigned long long)half * kHalf;
  const unsigned long long q_end = (base + kHalf - 1) >> 2;
  uint32_t w[kHalfWords];
#pragma unroll
  for (int i = 0; i < kHalfWords; ++i)
    w[i] = 0u;
  for (unsigned long long q = base >> 2; q <= q_end; ++q) {
    const uint4 rng = ffpa::prefill::philox4x32_10(philox_seed, q);
#pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
      const long long c =
          (long long)(q * 4 + (unsigned long long)lane) - (long long)base;
      if (c >= 0 && c < kHalf) {
        const float u = ffpa::prefill::uniform_from_philox_uint(
            ffpa::prefill::select_philox_lane(rng, lane));
        if (u > dropout_p)
          w[(int)c >> 5] |= 1u << ((int)c & 31);
      }
    }
  }
  uint32_t* dst = bitmap + row * (kBc / 32) + half * kHalfWords;
#pragma unroll
  for (int i = 0; i < kHalfWords; ++i)
    dst[i] = w[i];
}

// PC-14 consumer-side bitmap application: same arithmetic as the inline
// path (keep ? score*keep_scale : 0), decision read from the consumer-
// generated bitmap. Words are staged into registers once per row.
// NOTE: must stay __forceinline__ — __noinline__ forces the caller to
// ABI-spill the whole kv-loop live set (~150 regs) around each call,
// 2.4G local sectors/kernel, 5.7x slower (measured).
template <typename ScoresTensor, typename CoordTensor, int kRows, int kCols,
          int kBc>
__device__ __forceinline__ void apply_dropout_bitmap_rowcol(
    ScoresTensor& scores, const CoordTensor& tScS_rc,
    const uint32_t* __restrict__ bitmap, float keep_scale) {
#pragma unroll
  for (int row = 0; row < kRows; ++row) {
    const uint32_t* words = bitmap + cute::get<0>(tScS_rc(row, 0)) * (kBc / 32);
    uint32_t wr[kBc / 32];
#pragma unroll
    for (int i = 0; i < kBc / 32; ++i)
      wr[i] = words[i];
#pragma unroll
    for (int col = 0; col < kCols; ++col) {
      const int c = cute::get<1>(tScS_rc(row, col));
      scores(row, col) = ((wr[c >> 5] >> (c & 31)) & 1u)
                             ? scores(row, col) * keep_scale
                             : 0.0f;
    }
  }
}

}  // namespace ffpa_cute

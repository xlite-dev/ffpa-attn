// Manually SMEM swizzling for bank conflict free.
// ----------------------------------------------------------------
// [INFO] Assert smem store layout col_stride <= 16, prefer 16.   |
// [INFO] For logical_col_stride > 16, we have to permute the     |
// [INFO] smem store layout using col major ZigZag method:        |
// [INFO] e.g, --> Q smem logical layout [Br][64].                |
// [INFO]      --> col major ZigZag permuted -->                  |
// [INFO]      --> Q smem store layout [4][Br][16].               |
// ----------------------------------------------------------------
// ----------------------------------------------------------------
// -------------------------swizzle layout-------------------------
// --------------------logical col 0~64, step 8--------------------
// ---------------------smem col 0~16, step 8----------------------
// ----------------------------------------------------------------
// |bank  |b 0~3 |b 4~7 |b 0~3 |b 4~7 |b 0~3 |b 4~7 |b 0~3 |b 4~7 |
// |row 0 |  0   |  8   |  0   |  8   |  0   |  8   |  0   |  8   |
// |bank  |b 8~11|b12~15|b 8~11|b12~15|b 8~11|b12~15|b 8~11|b12~15|
// |row 1 |  0   |  8   |  0   |  8   |  0   |  8   |  0   |  8   |
// |bank  |b16~19|b20~23|b16~19|b20~23|b16~19|b20~23|b16~19|b20~23|
// |row 2 |  0   |  8   |  0   |  8   |  0   |  8   |  0   |  8   |
// |bank  |b24~27|b28~31|b24~27|b28~31|b24~27|b28~31|b24~27|b28~31|
// |row 3 |  0   |  8   |  0   |  8   |  0   |  8   |  0   |  8   |
// ----------------------------------------------------------------
// |bank  |b 0~3 |b 4~7 |b 0~3 |b 4~7 |b 0~3 |b 4~7 |b 0~3 |b 4~7 |
// |row 4 |  8   |  0   |  8   |  0   |  8   |  0   |  8   |  0   |
// |bank  |b 8~11|b12~15|b 8~11|b12~15|b 8~11|b12~15|b 8~11|b12~15|
// |row 5 |  8   |  0   |  8   |  0   |  8   |  0   |  8   |  0   |
// |bank  |b16~19|b20~23|b16~19|b20~23|b16~19|b20~23|b16~19|b20~23|
// |row 6 |  8   |  0   |  8   |  0   |  8   |  0   |  8   |  0   |
// |bank  |b24~27|b28~31|b24~27|b28~31|b24~27|b28~31|b24~27|b28~31|
// |row 7 |  8   |  0   |  8   |  0   |  8   |  0   |  8   |  0   |
// ----------------------------------------------------------------
// |bank  |b 0~3 |b 4~7 |b 0~3 |b 4~7 |b 0~3 |b 4~7 |b 0~3 |b 4~7 |
// |row 8 |  0   |  8   |  0   |  8   |  0   |  8   |  0   |  8   |
// |bank  |b 8~11|b12~15|b 8~11|b12~15|b 8~11|b12~15|b 8~11|b12~15|
// |row 9 |  0   |  8   |  0   |  8   |  0   |  8   |  0   |  8   |
// |bank  |b16~19|b20~23|b16~19|b20~23|b16~19|b20~23|b16~19|b20~23|
// |row 10|  0   |  8   |  0   |  8   |  0   |  8   |  0   |  8   |
// |bank  |b24~27|b28~31|b24~27|b28~31|b24~27|b28~31|b24~27|b28~31|
// |row 11|  0   |  8   |  0   |  8   |  0   |  8   |  0   |  8   |
// ----------------------------------------------------------------
// |bank  |b 0~3 |b 4~7 |b 0~3 |b 4~7 |b 0~3 |b 4~7 |b 0~3 |b 4~7 |
// |row 12|  8   |  0   |  8   |  0   |  8   |  0   |  8   |  0   |
// |bank  |b 8~11|b12~15|b 8~11|b12~15|b 8~11|b12~15|b 8~11|b12~15|
// |row 13|  8   |  0   |  8   |  0   |  8   |  0   |  8   |  0   |
// |bank  |b16~19|b20~23|b16~19|b20~23|b16~19|b20~23|b16~19|b20~23|
// |row 14|  8   |  0   |  8   |  0   |  8   |  0   |  8   |  0   |
// |bank  |b24~27|b28~31|b24~27|b28~31|b24~27|b28~31|b24~27|b28~31|
// |row 15|  8   |  0   |  8   |  0   |  8   |  0   |  8   |  0   |
// ----------------------------------------------------------------
#pragma once
#include <cuda_runtime.h>

namespace ffpa {
namespace swizzle {

// i: row index; j: col index.
// Returns the chunk-level (8 fp16 = 16 B per chunk) column swizzle, i.e.
// 0 / 8 / 16 / ... that the caller adds to the chunk-aligned base column.
// The within-chunk offset (``j & 7``) is preserved by the caller.
//
// kColStride matches the equivalent CuTe ``Swizzle<B, M, S>`` swizzle
// hardware mode used by TMA bulk-tensor descriptors:
//   * kColStride = 16 fp16 = 32 B/row -> Swizzle<1, 4, 3> = SWIZZLE_32B
//   * kColStride = 32 fp16 = 64 B/row -> Swizzle<2, 4, 3> = SWIZZLE_64B
//   * kColStride = 64 fp16 = 128 B/row -> Swizzle<3, 4, 3> = SWIZZLE_128B
// The XOR mask is derived from the byte-address bit positions used by the
// underlying CuTe swizzle pattern: bits {4..(4+B-1)} XOR bits
// {7..(7+B-1)} of the absolute byte address. Translating to (i, j) with
// row stride ``kColStride * 2`` bytes:
//   * 32B (B=1) : (j>>3) ^ (i>>2)              [chunk in {0, 1}]
//   * 64B (B=2) : chunk(2b) ^ {(i>>1)&1, (i>>2)&1}      [{0..3}]
//   * 128B(B=3) : chunk(3b) ^ {i&1, (i>>1)&1, (i>>2)&1} [{0..7}]
template <const int kColStride = 16, const int kStep = 8>
static __device__ __forceinline__ int permuted(int i, int j) {
  static_assert(kColStride == 16 || kColStride == 32 || kColStride == 64 || kColStride == 8,
                "kColStride must be one of {8, 16, 32, 64} (matches SWIZZLE_32B/64B/128B + the "
                "kStep=4 narrow legacy case).");
  static_assert(kStep == 4 || kStep == 8, "kStep must be 8 or 4.");
  static_assert(kColStride % kStep == 0, "kColStride must be multiple of kStep.");
  if constexpr (kStep == 4) {
    static_assert(kColStride <= 16,
                  "kStep=4 only supports the legacy narrow swizzle (kColStride <= 16).");
    return (((j >> 2) ^ (i >> 2)) % (kColStride >> 2)) << 2;
  } else if constexpr (kColStride == 16) {
    return (((j >> 3) ^ (i >> 2)) & 1) << 3;  // SWIZZLE_32B
  } else if constexpr (kColStride == 32) {
    const int chunk = (j >> 3) & 3;
    const int xor_mask = ((i >> 1) & 1) | (((i >> 2) & 1) << 1);
    return (chunk ^ xor_mask) << 3;  // SWIZZLE_64B
  } else {                           // kColStride == 64
    const int chunk = (j >> 3) & 7;
    const int xor_mask = (i & 1) | (((i >> 1) & 1) << 1) | (((i >> 2) & 1) << 2);
    return (chunk ^ xor_mask) << 3;  // SWIZZLE_128B
  }
}

}  // namespace swizzle
}  // namespace ffpa

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
template<const int kColStride = 16, const int kStep = 8>
static __device__ __forceinline__ int permuted(int i, int j) {
  // swizzle: ((int(j / kStep) ^ int(i / 4)) % int(kColStride / kStep)) * kStep;
  static_assert(kColStride <= 16, "Currently, kColStride must be less than or equal to 16.");
  static_assert(kStep == 4 || kStep == 8, "kStep must be 8 or 4.");
  static_assert(kColStride % kStep == 0, "kColStride must be multiple of kStep.");
  if constexpr (kStep == 8) {
    return (((j >> 3) ^ (i >> 2)) % (kColStride >> 3)) << 3;
  } else {
    static_assert(kStep == 4);
    return (((j >> 2) ^ (i >> 2)) % (kColStride >> 2)) << 2;
  }
}

} // swizzle
} // ffpa


#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <type_traits>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <mma.h>

namespace ffpa {
namespace mma {

enum class MMAMode {
  kAutoZeroFill = 0U,
  kInplaceUpdate = 1U,
};

// Simple wrappers for mma and ldmatrix instructions.
template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void m16n8k16_f16f16f16(uint32_t* RD0, uint32_t* RD1, uint32_t* RA0,
                                                   uint32_t* RA1, uint32_t* RA2, uint32_t* RA3,
                                                   uint32_t* RB0, uint32_t* RB1) {
  if constexpr (mma_mode == MMAMode::kInplaceUpdate) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0, %1}, "
        "{%2, %3, %4, %5}, "
        "{%6, %7}, "
        "{%8, %9};\n"
        : "=r"(RD0[0]), "=r"(RD1[0])
        : "r"(RA0[0]), "r"(RA1[0]), "r"(RA2[0]), "r"(RA3[0]), "r"(RB0[0]), "r"(RB1[0]), "r"(RD0[0]),
          "r"(RD1[0]));
  } else {
    // WARN: seems can not get good performance while stage = 1.
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0, %1}, "
        "{%2, %3, %4, %5}, "
        "{%6, %7}, "
        "{%8, %9};\n"
        : "=r"(RD0[0]), "=r"(RD1[0])
        : "r"(RA0[0]), "r"(RA1[0]), "r"(RA2[0]), "r"(RA3[0]), "r"(RB0[0]), "r"(RB1[0]), "r"(0),
          "r"(0));
  }
}

template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void m16n8k16_f16f16f32(uint32_t* RD0, uint32_t* RD1, uint32_t* RD2,
                                                   uint32_t* RD3, uint32_t* RA0, uint32_t* RA1,
                                                   uint32_t* RA2, uint32_t* RA3, uint32_t* RB0,
                                                   uint32_t* RB1) {
  // "h" = .u16 reg; "r" = .u32 reg; "l" = .u64 reg;
  // "f" = .f32 reg; "d" = .f64 reg
  if constexpr (mma_mode == MMAMode::kInplaceUpdate) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,  %1,  %2,  %3}, "
        "{%4,  %5,  %6,  %7}, "
        "{%8,  %9}, "
        "{%10, %11, %12, %13};\n"
        : "=r"(RD0[0]), "=r"(RD1[0]), "=r"(RD2[0]), "=r"(RD3[0])
        : "r"(RA0[0]), "r"(RA1[0]), "r"(RA2[0]), "r"(RA3[0]), "r"(RB0[0]), "r"(RB1[0]), "r"(RD0[0]),
          "r"(RD1[0]), "r"(RD2[0]), "r"(RD3[0]));
  } else {
    // WARN: seems can not get good performance while stage = 1.
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,  %1,  %2,  %3}, "
        "{%4,  %5,  %6,  %7}, "
        "{%8,  %9}, "
        "{%10, %11, %12, %13};\n"
        : "=r"(RD0[0]), "=r"(RD1[0]), "=r"(RD2[0]), "=r"(RD3[0])
        : "r"(RA0[0]), "r"(RA1[0]), "r"(RA2[0]), "r"(RA3[0]), "r"(RB0[0]), "r"(RB1[0]), "r"(0),
          "r"(0), "r"(0), "r"(0));
  }
}

template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void m16n8k16_bf16bf16f32(uint32_t* RD0, uint32_t* RD1, uint32_t* RD2,
                                                     uint32_t* RD3, uint32_t* RA0, uint32_t* RA1,
                                                     uint32_t* RA2, uint32_t* RA3, uint32_t* RB0,
                                                     uint32_t* RB1) {
  // BF16 m16n8k16 mma. BF16 hardware has only the f32-accumulated variant; no
  // bf16-accumulated mma instruction exists, so the FFPA BF16 path always
  // routes through this wrapper with kMmaAccFloat32QK=1 and kMmaAccFloat32PV=1.
  if constexpr (mma_mode == MMAMode::kInplaceUpdate) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,  %1,  %2,  %3}, "
        "{%4,  %5,  %6,  %7}, "
        "{%8,  %9}, "
        "{%10, %11, %12, %13};\n"
        : "=r"(RD0[0]), "=r"(RD1[0]), "=r"(RD2[0]), "=r"(RD3[0])
        : "r"(RA0[0]), "r"(RA1[0]), "r"(RA2[0]), "r"(RA3[0]), "r"(RB0[0]), "r"(RB1[0]), "r"(RD0[0]),
          "r"(RD1[0]), "r"(RD2[0]), "r"(RD3[0]));
  } else {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,  %1,  %2,  %3}, "
        "{%4,  %5,  %6,  %7}, "
        "{%8,  %9}, "
        "{%10, %11, %12, %13};\n"
        : "=r"(RD0[0]), "=r"(RD1[0]), "=r"(RD2[0]), "=r"(RD3[0])
        : "r"(RA0[0]), "r"(RA1[0]), "r"(RA2[0]), "r"(RA3[0]), "r"(RB0[0]), "r"(RB1[0]), "r"(0),
          "r"(0), "r"(0), "r"(0));
  }
}

// Dtype-dispatching wrapper over the f32-accumulated m16n8k16 mma for the
// activation dtypes supported by FFPA (half and bfloat16). The dispatch is
// resolved at compile time via `if constexpr`, so each specialization of the
// kernel only emits one PTX variant.
template <typename kDataType, MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void m16n8k16_abf32(uint32_t* RD0, uint32_t* RD1, uint32_t* RD2,
                                               uint32_t* RD3, uint32_t* RA0, uint32_t* RA1,
                                               uint32_t* RA2, uint32_t* RA3, uint32_t* RB0,
                                               uint32_t* RB1) {
  if constexpr (std::is_same_v<kDataType, __half>) {
    m16n8k16_f16f16f32<mma_mode>(RD0, RD1, RD2, RD3, RA0, RA1, RA2, RA3, RB0, RB1);
  } else {
    static_assert(std::is_same_v<kDataType, __nv_bfloat16>,
                  "m16n8k16_abf32 only supports __half and __nv_bfloat16.");
    m16n8k16_bf16bf16f32<mma_mode>(RD0, RD1, RD2, RD3, RA0, RA1, RA2, RA3, RB0, RB1);
  }
}

__device__ __forceinline__ void ldmatrix_m8n8x4(uint32_t* R0, uint32_t* R1, uint32_t* R2,
                                                uint32_t* R3, uint32_t smem_ptr) {
  asm volatile(
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(R0[0]), "=r"(R1[0]), "=r"(R2[0]), "=r"(R3[0])
      : "r"(smem_ptr));
}

__device__ __forceinline__ void ldmatrix_m8n8x4_trans(uint32_t* R0, uint32_t* R1, uint32_t* R2,
                                                      uint32_t* R3, uint32_t smem_ptr) {
  asm volatile(
      "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 "
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(R0[0]), "=r"(R1[0]), "=r"(R2[0]), "=r"(R3[0])
      : "r"(smem_ptr));
}

__device__ __forceinline__ void ldmatrix_m8n8x2(uint32_t* R0, uint32_t* R1, uint32_t smem_ptr) {
  asm volatile(
      "ldmatrix.sync.aligned.x2.m8n8.shared.b16 "
      "{%0, %1}, [%2];\n"
      : "=r"(R0[0]), "=r"(R1[0])
      : "r"(smem_ptr));
}

__device__ __forceinline__ void ldmatrix_m8n8x2_trans(uint32_t* R0, uint32_t* R1,
                                                      uint32_t smem_ptr) {
  asm volatile(
      "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 "
      "{%0, %1}, [%2];\n"
      : "=r"(R0[0]), "=r"(R1[0])
      : "r"(smem_ptr));
}

}  // namespace mma
}  // namespace ffpa

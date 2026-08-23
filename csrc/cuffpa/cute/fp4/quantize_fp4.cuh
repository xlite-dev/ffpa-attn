// NVFP4 quantize pre-kernels, ported from SageAttention3
// fp4_quantization_4d.cu and adapted to the ffpa-attn pipeline:
// Reference:
// https://github.com/thu-ml/SageAttention/tree/main/sageattention3_blackwell/sageattn3/quantization/fp4_quantization_4d.cu
//   - inputs keep their native (B,S,H,D) strides (no host transpose);
//   - K-row permutation for the PV accumulator->A-operand layout is folded
//     into the K quantize kernel (same [0,1,8,9,16,17,24,25,...] table);
//   - K smoothing (subtract per-(b,h) km) and Q centering (subtract
//     per-128-row block mean qm) happen in-kernel before the group max;
//   - workspaces are padded to 128-token multiples: kernels launch on the
//     padded grid, guard reads with the true token count and write zeros
//     (data + SF) into the tail so TMA loads are well-defined.
// SF bytes are written in the SfAtom-blocked order the MMA's SF operand
// expects (offset formula must match ffpa_fp4::BlockScaledConfig).
#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/all.h>

#include "fp4_gemm.cuh"

namespace ffpa_fp4 {

constexpr int kCVTFp4EltsPerThread = 16;

// 4 float2 (8 values) -> one uint32 of packed e2m1. cvt e2m1x2 requires
// sm_120+; every caller lives under the __CUDA_ARCH__ >= 1200 guards below.
inline __device__ uint32_t fp32_vec_to_e2m1(float2* array) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  uint32_t val;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
      "}"
      : "=r"(val)
      : "f"(array[0].x), "f"(array[0].y), "f"(array[1].x), "f"(array[1].y),
        "f"(array[2].x), "f"(array[2].y), "f"(array[3].x), "f"(array[3].y));
  return val;
#else
  (void)array;
  return 0u;
#endif
}

// 2 float2 (4 values) -> one uint32 of packed e4m3 (byte i = value i).
inline __device__ uint32_t fp32_vec_to_e4m3(float2* array) {
  uint32_t val;
  asm volatile(
      "{\n"
      ".reg .b16 lo;\n"
      ".reg .b16 hi;\n"
      "cvt.rn.satfinite.e4m3x2.f32   lo, %2, %1;\n"
      "cvt.rn.satfinite.e4m3x2.f32   hi, %4, %3;\n"
      "mov.b32 %0, {lo, hi};\n"
      "}"
      : "=r"(val)
      : "f"(array[0].x), "f"(array[0].y), "f"(array[1].x), "f"(array[1].y));
  return val;
}

template <typename T>
struct Fp4TypeConverter {
  using Type2 = void;
};
template <>
struct Fp4TypeConverter<half> {
  using Type2 = half2;
};
template <>
struct Fp4TypeConverter<__nv_bfloat16> {
  using Type2 = __nv_bfloat162;
};

template <class T>
struct Fp4PackedVec {
  typename Fp4TypeConverter<T>::Type2 elts[8];
};

// Q / K quantization: (B,S,H,D) strided fp16/bf16 -> packed e2m1 (B,H,N,D/2)
// + SF (B,H,N,D/16) in the MMA atom layout. kPermute selects the K row
// permutation; kSubQm / kSubKm select bias subtraction before quantization.
// Mapping: 4 threads per token, each owning a 64-element slice (one SF
// 4-col block) split into kHeadDim/64 16-element PackedVecs; tokens per
// block shrink for large D (see the launchers) so regs*threads stays within
// the SM register file.
// d_og < kHeadDim (padded head_dim): input rows are only d_og wide
// (d_og%8==0); guarded loads keep pad cols zero so data and SF pad cols
// are 0.
// kHadamard: fold the full-width Walsh-Hadamard row rotation into the load
// (replaces the standalone wht_qk_kernel + its whole-tensor read/write).
// The 4 threads of a token are contiguous, so with kE = kHeadDim/4 elems
// per thread the butterfly splits into in-thread pairs (dist < kE), one
// shfl_xor(1) pass (dist = kE) and one shfl_xor(2) pass (dist = 2*kE);
// requires pow2 kHeadDim. qm/km biases must then be pre-rotated (qm_rot /
// km_rot from wht_f32_rows_kernel): WHT is linear, so mean and rotation
// commute and delta_s/qkm stay exact in the unrotated domain.
template <typename T, int kHeadDim, bool kPermute, bool kSubQm, bool kSubKm,
          bool kHadamard = false>
__global__ void fp4_quant_kernel(
    const T* __restrict__ input, uint8_t* __restrict__ output,
    uint8_t* __restrict__ output_sf, int num_tokens, int stride_bz_input,
    int stride_h_input, int stride_seq_input, int stride_bz_output,
    int stride_h_output, int stride_seq_output, int stride_bz_output_sf,
    int stride_h_output_sf, int stride_seq_output_sf,
    const float* __restrict__ qm, int qm_stride_b, int qm_stride_h,
    const float* __restrict__ km, int km_stride_b, int km_stride_h, int d_og) {
  using PackedVec = Fp4PackedVec<T>;
  constexpr int kThreadsPerToken = 4;
  // tokens/block is a launch-config choice (launcher shrinks it for large D
  // to keep regs*threads within the SM register file); derive it from
  // blockDim so the kernel stays agnostic.
  const int kBlock = blockDim.x / kThreadsPerToken;
  constexpr int kVecsPerThread = kHeadDim / 64;
  static_assert(kHeadDim % 64 == 0, "fp4 quantize requires 64-multiple D");

  const int batch_id = blockIdx.y;
  const int head_id = blockIdx.z;
  const int token_block_id = blockIdx.x;
  const int local_token_id = threadIdx.x / kThreadsPerToken;
  const int token_id = token_block_id * kBlock + local_token_id;
  const int slice = threadIdx.x % kThreadsPerToken;

  int load_token_id;
  if constexpr (!kPermute) {
    load_token_id = token_id;
  } else {
    // [0,1,8,9,16,17,24,25,2,3,10,11,18,19,26,27,4,5,12,13,20,21,28,29,6,7,14,15,22,23,30,31]
    load_token_id = token_block_id * kBlock + kv_perm32(local_token_id);
  }
  const bool token_valid = load_token_id < num_tokens;

  PackedVec in_vec[kVecsPerThread];
#pragma unroll
  for (int v = 0; v < kVecsPerThread; v++) {
#pragma unroll
    for (int i = 0; i < kCVTFp4EltsPerThread / 2; i++) {
      reinterpret_cast<uint32_t&>(in_vec[v].elts[i]) = 0;
    }
  }
  if (token_valid) {
    const PackedVec* __restrict__ src = reinterpret_cast<const PackedVec*>(
        input + batch_id * stride_bz_input + head_id * stride_h_input +
        load_token_id * stride_seq_input +
        slice * kVecsPerThread * kCVTFp4EltsPerThread);
    const typename Fp4TypeConverter<T>::Type2* __restrict__ src2 =
        reinterpret_cast<const typename Fp4TypeConverter<T>::Type2*>(src);
#pragma unroll
    for (int v = 0; v < kVecsPerThread; v++) {
      const int off = (slice * kVecsPerThread + v) * kCVTFp4EltsPerThread;
      if (off + kCVTFp4EltsPerThread <= d_og) {
        in_vec[v] = src[v];  // whole 16-elem vec
      } else {
        // Tail vec straddling d_og (d_og%8==0): 8-elem halves. Pad halves
        // stay zero; the shared SF then scales only the real half.
#pragma unroll
        for (int h = 0; h < 2; h++) {
          if (off + h * 8 < d_og) {
#pragma unroll
            for (int i = 0; i < 4; i++)
              in_vec[v].elts[h * 4 + i] = src2[v * 8 + h * 4 + i];
          }
        }
      }
    }
  }

  float2 fp2Vals[kVecsPerThread][kCVTFp4EltsPerThread / 2];
#pragma unroll
  for (int v = 0; v < kVecsPerThread; v++) {
#pragma unroll
    for (int i = 0; i < kCVTFp4EltsPerThread / 2; i++) {
      if constexpr (std::is_same<T, half>::value) {
        fp2Vals[v][i] = __half22float2(in_vec[v].elts[i]);
      } else {
        fp2Vals[v][i] = __bfloat1622float2(in_vec[v].elts[i]);
      }
    }
  }
  if constexpr (kHadamard) {
    // Row WHT folded into the load: kE contiguous elems per thread, so
    // distances < kE stay in-thread; dist kE / 2kE swap with the slice^1 /
    // slice^2 neighbor via one shfl each (invalid tokens load zeros, so
    // the whole warp stays convergent through the shuffles).
    constexpr int kE = kHeadDim / 4;
    static_assert((kHeadDim & (kHeadDim - 1)) == 0,
                  "fused WHT requires pow2 kHeadDim");
    static_assert(kVecsPerThread * kCVTFp4EltsPerThread == kE);
    float x[kE];
#pragma unroll
    for (int v = 0; v < kVecsPerThread; v++)
#pragma unroll
      for (int i = 0; i < kCVTFp4EltsPerThread / 2; i++) {
        x[v * kCVTFp4EltsPerThread + 2 * i] = fp2Vals[v][i].x;
        x[v * kCVTFp4EltsPerThread + 2 * i + 1] = fp2Vals[v][i].y;
      }
#pragma unroll
    for (int dist = kE / 2; dist >= 1; dist >>= 1) {
#pragma unroll
      for (int j = 0; j < kE; j++) {
        const int jj = j ^ dist;
        if (j < jj) {
          const float a = x[j], b = x[jj];
          x[j] = a + b;
          x[jj] = a - b;
        }
      }
    }
#pragma unroll
    for (int j = 0; j < kE; j++) {
      const float p = __shfl_xor_sync(0xffffffffu, x[j], 1);
      x[j] = (slice & 1) ? p - x[j] : x[j] + p;
    }
#pragma unroll
    for (int j = 0; j < kE; j++) {
      const float p = __shfl_xor_sync(0xffffffffu, x[j], 2);
      x[j] = (slice & 2) ? p - x[j] : x[j] + p;
    }
    const float s = rsqrtf((float)kHeadDim);
#pragma unroll
    for (int v = 0; v < kVecsPerThread; v++)
#pragma unroll
      for (int i = 0; i < kCVTFp4EltsPerThread / 2; i++) {
        fp2Vals[v][i].x = x[v * kCVTFp4EltsPerThread + 2 * i] * s;
        fp2Vals[v][i].y = x[v * kCVTFp4EltsPerThread + 2 * i + 1] * s;
      }
  }
  // padded tail rows stay zero: skip the bias there (SF and data both)
  if constexpr (kSubQm) {
    if (token_valid) {
      const float* qm_row = qm +
                            (batch_id * qm_stride_b + head_id * qm_stride_h +
                             (token_id / 128) * kHeadDim) +
                            slice * kVecsPerThread * kCVTFp4EltsPerThread;
#pragma unroll
      for (int v = 0; v < kVecsPerThread; v++) {
#pragma unroll
        for (int i = 0; i < kCVTFp4EltsPerThread / 2; i++) {
          float2 b = *reinterpret_cast<const float2*>(
              qm_row + v * kCVTFp4EltsPerThread + 2 * i);
          fp2Vals[v][i].x -= b.x;
          fp2Vals[v][i].y -= b.y;
        }
      }
    }
  }
  if constexpr (kSubKm) {
    if (token_valid) {
      const float* km_row = km + batch_id * km_stride_b +
                            head_id * km_stride_h +
                            slice * kVecsPerThread * kCVTFp4EltsPerThread;
#pragma unroll
      for (int v = 0; v < kVecsPerThread; v++) {
#pragma unroll
        for (int i = 0; i < kCVTFp4EltsPerThread / 2; i++) {
          float2 b = *reinterpret_cast<const float2*>(
              km_row + v * kCVTFp4EltsPerThread + 2 * i);
          fp2Vals[v][i].x -= b.x;
          fp2Vals[v][i].y -= b.y;
        }
      }
    }
  }

  uint8_t* output_sf_save_base = output_sf + batch_id * stride_bz_output_sf +
                                 head_id * stride_h_output_sf +
                                 (token_id / 64) * 64 * stride_seq_output_sf;
  uint32_t token_id_local = token_id % 64;
#pragma unroll
  for (int v = 0; v < kVecsPerThread; v++) {
    float vecMax = 0.f;
#pragma unroll
    for (int i = 0; i < kCVTFp4EltsPerThread / 2; i++) {
      vecMax =
          fmaxf(vecMax, fmaxf(fabsf(fp2Vals[v][i].x), fabsf(fp2Vals[v][i].y)));
    }

    float SFValue = vecMax / 6.0f;
    uint8_t SFValueFP8;
    reinterpret_cast<__nv_fp8_e4m3&>(SFValueFP8) = __nv_fp8_e4m3(SFValue);
    SFValue = float(reinterpret_cast<__nv_fp8_e4m3&>(SFValueFP8));
    float SFValueInv = (SFValue == 0.0f) ? 0.0f : 1.0f / SFValue;

#pragma unroll
    for (int i = 0; i < kCVTFp4EltsPerThread / 2; i++) {
      fp2Vals[v][i].x = fp2Vals[v][i].x * SFValueInv;
      fp2Vals[v][i].y = fp2Vals[v][i].y * SFValueInv;
    }

    uint32_t e2m1Vals[kCVTFp4EltsPerThread / 8];
#pragma unroll
    for (int i = 0; i < kCVTFp4EltsPerThread / 8; i++) {
      e2m1Vals[i] = fp32_vec_to_e2m1(fp2Vals[v] + i * 4);
    }

    reinterpret_cast<uint64_t*>(
        output + batch_id * stride_bz_output + head_id * stride_h_output +
        token_id * stride_seq_output +
        (slice * kVecsPerThread + v) * kCVTFp4EltsPerThread / 2)[0] =
        reinterpret_cast<uint64_t*>(e2m1Vals)[0];

    uint32_t col_id_local = slice * kVecsPerThread + v;
    uint32_t offset_local = (col_id_local / 4) * 256 + (col_id_local % 4) +
                            (token_id_local / 16) * 4 +
                            (token_id_local % 16) * 16;
    reinterpret_cast<uint8_t*>(output_sf_save_base + offset_local)[0] =
        SFValueFP8;
  }
}

// V quantization with transpose: (B,S,H,D) -> packed e2m1 (B,H,D,N/2) + SF
// (B,H,D,N/16) in the transposed SF atom layout. No bias (V not smoothed).
// Mapping mirrors the Q/K kernel (4 threads per token, 64-element slices);
// after the smem transpose each thread owns one (d, 16-token) SF group and
// iterates d in kHeadDim/64 passes over 64 d-rows each. Tokens per block
// shrink to 64 for D>128 and to 32 for D>768 (96KB+ windows would not fit
// the 101KB opt-in); the staging window goes dynamic (opt-in) once past
// the 48KB static limit (D>=512 at 64 tokens).
// d_og < kHeadDim (d_og%8==0): guarded loads keep pad d-rows zero in data
// and SF.
template <typename T, int kHeadDim>
__global__ void fp4_quant_trans_kernel(
    const T* __restrict__ input, uint8_t* __restrict__ output,
    uint8_t* __restrict__ output_sf, int num_tokens, int stride_bz_input,
    int stride_h_input, int stride_seq_input, int stride_bz_output,
    int stride_h_output, int stride_d_output, int stride_bz_output_sf,
    int stride_h_output_sf, int stride_d_output_sf, int d_og,
    const float* __restrict__ vm = nullptr, int vm_stride_b = 0,
    int vm_stride_h = 0) {
  using PackedVec = Fp4PackedVec<T>;
  constexpr int kTokensPerBlock =
      (kHeadDim <= 128) ? 128 : ((kHeadDim <= 768) ? 64 : 32);
  constexpr int kThreadsPerToken = 4;
  constexpr int kVecsPerThread = kHeadDim / 64;
  constexpr int kSFGroupsPerToken = kHeadDim / 16;
  constexpr int kThreadsPerSeq = kTokensPerBlock / kCVTFp4EltsPerThread;
  constexpr int kDRowsPerPass =
      kTokensPerBlock * kThreadsPerToken / kThreadsPerSeq;
  constexpr bool kDynamicSmem =
      kTokensPerBlock * kHeadDim * int(sizeof(T)) > 48 * 1024;
  static_assert(kHeadDim % 64 == 0 && kHeadDim % kDRowsPerPass == 0,
                "fp4 quantize requires 64-multiple D");

  const int batch_id = blockIdx.y;
  const int head_id = blockIdx.z;
  const int token_block_id = blockIdx.x;
  const int token_id =
      token_block_id * kTokensPerBlock + threadIdx.x / kThreadsPerToken;
  const int slice = threadIdx.x % kThreadsPerToken;

  PackedVec in_vec[kVecsPerThread];
#pragma unroll
  for (int v = 0; v < kVecsPerThread; v++) {
#pragma unroll
    for (int i = 0; i < kCVTFp4EltsPerThread / 2; i++) {
      reinterpret_cast<uint32_t&>(in_vec[v].elts[i]) = 0;
    }
  }
  if (token_id < num_tokens) {
    const PackedVec* __restrict__ src = reinterpret_cast<const PackedVec*>(
        input + batch_id * stride_bz_input + head_id * stride_h_input +
        token_id * stride_seq_input +
        slice * kVecsPerThread * kCVTFp4EltsPerThread);
    const typename Fp4TypeConverter<T>::Type2* __restrict__ src2 =
        reinterpret_cast<const typename Fp4TypeConverter<T>::Type2*>(src);
#pragma unroll
    for (int v = 0; v < kVecsPerThread; v++) {
      const int off = (slice * kVecsPerThread + v) * kCVTFp4EltsPerThread;
      if (off + kCVTFp4EltsPerThread <= d_og) {
        in_vec[v] = src[v];
      } else {
#pragma unroll
        for (int h = 0; h < 2; h++) {
          if (off + h * 8 < d_og) {
#pragma unroll
            for (int i = 0; i < 4; i++)
              in_vec[v].elts[h * 4 + i] = src2[v * 8 + h * 4 + i];
          }
        }
      }
    }
  }

  extern __shared__ __align__(16) char quant_dyn_shm[];
  __shared__ T quant_static_shm[kDynamicSmem ? 1 : kTokensPerBlock * kHeadDim];
  T* shared_input =
      kDynamicSmem ? reinterpret_cast<T*>(quant_dyn_shm) : quant_static_shm;
  PackedVec* shared_pv = reinterpret_cast<PackedVec*>(shared_input);
#pragma unroll
  for (int v = 0; v < kVecsPerThread; v++) {
    shared_pv[(threadIdx.x / kThreadsPerToken) * kSFGroupsPerToken +
              slice * kVecsPerThread + v] = in_vec[v];
  }
  __syncthreads();

  uint8_t* output_sf_base =
      output_sf + batch_id * stride_bz_output_sf + head_id * stride_h_output_sf;
  const float* vm_head =
      vm ? vm + batch_id * vm_stride_b + head_id * vm_stride_h : nullptr;
  uint32_t col_id_local =
      token_block_id * kTokensPerBlock / kCVTFp4EltsPerThread +
      threadIdx.x % kThreadsPerSeq;
  uint32_t sf_col_offset = (col_id_local / 4) * 256 + (col_id_local % 4);
#pragma unroll
  for (int p = 0; p < kHeadDim / kDRowsPerPass; p++) {
    const int d = p * kDRowsPerPass + threadIdx.x / kThreadsPerSeq;
    float2 fp2Vals[kCVTFp4EltsPerThread / 2];
#pragma unroll
    for (int i = 0; i < kCVTFp4EltsPerThread / 2; i++) {
      const int tok0 =
          (threadIdx.x % kThreadsPerSeq) * kCVTFp4EltsPerThread + 2 * i;
      if constexpr (std::is_same<T, half>::value) {
        fp2Vals[i].x = __half2float(shared_input[tok0 * kHeadDim + d]);
        fp2Vals[i].y = __half2float(shared_input[(tok0 + 1) * kHeadDim + d]);
      } else {
        fp2Vals[i].x = __bfloat162float(shared_input[tok0 * kHeadDim + d]);
        fp2Vals[i].y =
            __bfloat162float(shared_input[(tok0 + 1) * kHeadDim + d]);
      }
    }
    // smooth_v: quantize the residual Vhat = V - vm instead of V, where
    //   vm[d] = (1/Nkv) * sum_n V[b, hkv, n, d]   (per-(b,hkv,d) constant)
    // is subtracted per column d here and added back in the attention
    // epilogue after the softmax normalize: because softmax weights sum to
    // 1 and vm is shared by all columns,
    //   O_i = (sum_j P_ij * (Vhat_j + vm)) / sum_j P_ij
    //       = (sum_j P_ij * Vhat_j) / sum_j P_ij + vm,
    // so the subtraction is mathematically exact; it only shrinks the
    // residual dynamic range so the e2m1 blockscale quantizes finer.
    if (vm_head) {
      const float vm_d = vm_head[d];
#pragma unroll
      for (int i = 0; i < kCVTFp4EltsPerThread / 2; i++) {
        fp2Vals[i].x -= vm_d;
        fp2Vals[i].y -= vm_d;
      }
    }

    float vecMax = 0.f;
#pragma unroll
    for (int i = 0; i < kCVTFp4EltsPerThread / 2; i++) {
      vecMax = fmaxf(vecMax, fmaxf(fabsf(fp2Vals[i].x), fabsf(fp2Vals[i].y)));
    }

    float SFValue = vecMax / 6.0f;
    uint8_t SFValueFP8;
    reinterpret_cast<__nv_fp8_e4m3&>(SFValueFP8) = __nv_fp8_e4m3(SFValue);
    SFValue = float(reinterpret_cast<__nv_fp8_e4m3&>(SFValueFP8));
    float SFValueInv = (SFValue == 0.0f) ? 0.0f : 1.0f / SFValue;

#pragma unroll
    for (int i = 0; i < kCVTFp4EltsPerThread / 2; i++) {
      fp2Vals[i].x = fp2Vals[i].x * SFValueInv;
      fp2Vals[i].y = fp2Vals[i].y * SFValueInv;
    }

    uint32_t e2m1Vals[kCVTFp4EltsPerThread / 8];
#pragma unroll
    for (int i = 0; i < kCVTFp4EltsPerThread / 8; i++) {
      e2m1Vals[i] = fp32_vec_to_e2m1(fp2Vals + i * 4);
    }

    reinterpret_cast<uint64_t*>(
        output + batch_id * stride_bz_output + head_id * stride_h_output +
        d * stride_d_output +
        (token_block_id * kTokensPerBlock +
         (threadIdx.x % kThreadsPerSeq) * kCVTFp4EltsPerThread) /
            2)[0] = reinterpret_cast<uint64_t*>(e2m1Vals)[0];

    uint32_t row_id_local = threadIdx.x / kThreadsPerSeq;
    uint32_t offset_local =
        sf_col_offset + (row_id_local / 16) * 4 + (row_id_local % 16) * 16;
    reinterpret_cast<uint8_t*>(
        output_sf_base + (d / 64) * 64 * stride_d_output_sf + offset_local)[0] =
        SFValueFP8;
  }
}

// MXFP8 V^T quantize (fp4_pv_mm_type=fp8): V^T as e4m3 data + ue8m0 SF per
// (d-row, 32-token group). Same staging window, thread mapping and guarded
// loads as fp4_quant_trans_kernel; the only quantization change is the SF:
// two adjacent pass threads (16 tokens each) merge their absmax via
// shfl_xor(1) into the 32-token group scale, ceiled to a power of two
// (2^ceil(log2(amax/448))), so the scaled data peaks within the e4m3
// range. The even thread of each pair stores the single SF byte; the gmem
// SF layout follows BlockScaledConfig<32> (same 64x4-block formula, group
// index = token/32).
template <typename T, int kHeadDim>
__global__ void mxfp8_quant_trans_kernel(
    const T* __restrict__ input, uint8_t* __restrict__ output,
    uint8_t* __restrict__ output_sf, int num_tokens, int stride_bz_input,
    int stride_h_input, int stride_seq_input, int stride_bz_output,
    int stride_h_output, int stride_d_output, int stride_bz_output_sf,
    int stride_h_output_sf, int stride_d_output_sf, int d_og,
    const float* __restrict__ vm = nullptr, int vm_stride_b = 0,
    int vm_stride_h = 0) {
  using PackedVec = Fp4PackedVec<T>;
  constexpr int kTokensPerBlock =
      (kHeadDim <= 128) ? 128 : ((kHeadDim <= 768) ? 64 : 32);
  constexpr int kThreadsPerToken = 4;
  constexpr int kVecsPerThread = kHeadDim / 64;
  constexpr int kSFGroupsPerToken = kHeadDim / 16;
  constexpr int kThreadsPerSeq = kTokensPerBlock / kCVTFp4EltsPerThread;
  constexpr int kDRowsPerPass =
      kTokensPerBlock * kThreadsPerToken / kThreadsPerSeq;
  constexpr bool kDynamicSmem =
      kTokensPerBlock * kHeadDim * int(sizeof(T)) > 48 * 1024;
  static_assert(kHeadDim % 64 == 0 && kHeadDim % kDRowsPerPass == 0,
                "mxfp8 quantize requires 64-multiple D");

  const int batch_id = blockIdx.y;
  const int head_id = blockIdx.z;
  const int token_block_id = blockIdx.x;
  const int token_id =
      token_block_id * kTokensPerBlock + threadIdx.x / kThreadsPerToken;
  const int slice = threadIdx.x % kThreadsPerToken;

  PackedVec in_vec[kVecsPerThread];
#pragma unroll
  for (int v = 0; v < kVecsPerThread; v++) {
#pragma unroll
    for (int i = 0; i < kCVTFp4EltsPerThread / 2; i++) {
      reinterpret_cast<uint32_t&>(in_vec[v].elts[i]) = 0;
    }
  }
  if (token_id < num_tokens) {
    const PackedVec* __restrict__ src = reinterpret_cast<const PackedVec*>(
        input + batch_id * stride_bz_input + head_id * stride_h_input +
        token_id * stride_seq_input +
        slice * kVecsPerThread * kCVTFp4EltsPerThread);
    const typename Fp4TypeConverter<T>::Type2* __restrict__ src2 =
        reinterpret_cast<const typename Fp4TypeConverter<T>::Type2*>(src);
#pragma unroll
    for (int v = 0; v < kVecsPerThread; v++) {
      const int off = (slice * kVecsPerThread + v) * kCVTFp4EltsPerThread;
      if (off + kCVTFp4EltsPerThread <= d_og) {
        in_vec[v] = src[v];
      } else {
#pragma unroll
        for (int h = 0; h < 2; h++) {
          if (off + h * 8 < d_og) {
#pragma unroll
            for (int i = 0; i < 4; i++)
              in_vec[v].elts[h * 4 + i] = src2[v * 8 + h * 4 + i];
          }
        }
      }
    }
  }

  extern __shared__ __align__(16) char quant_dyn_shm[];
  __shared__ T quant_static_shm[kDynamicSmem ? 1 : kTokensPerBlock * kHeadDim];
  T* shared_input =
      kDynamicSmem ? reinterpret_cast<T*>(quant_dyn_shm) : quant_static_shm;
  PackedVec* shared_pv = reinterpret_cast<PackedVec*>(shared_input);
#pragma unroll
  for (int v = 0; v < kVecsPerThread; v++) {
    shared_pv[(threadIdx.x / kThreadsPerToken) * kSFGroupsPerToken +
              slice * kVecsPerThread + v] = in_vec[v];
  }
  __syncthreads();

  uint8_t* output_sf_base =
      output_sf + batch_id * stride_bz_output_sf + head_id * stride_h_output_sf;
  const float* vm_head =
      vm ? vm + batch_id * vm_stride_b + head_id * vm_stride_h : nullptr;
  const uint32_t seq_lane = threadIdx.x % kThreadsPerSeq;
  const uint32_t row_id_local = threadIdx.x / kThreadsPerSeq;
  const uint32_t group_col_local =
      token_block_id * (kTokensPerBlock / 32) + seq_lane / 2;
  const uint32_t sf_col_offset =
      (group_col_local / 4) * 256 + (group_col_local % 4);
  const uint32_t offset_local =
      sf_col_offset + (row_id_local / 16) * 4 + (row_id_local % 16) * 16;
#pragma unroll
  for (int p = 0; p < kHeadDim / kDRowsPerPass; p++) {
    const int d = p * kDRowsPerPass + threadIdx.x / kThreadsPerSeq;
    float2 fp2Vals[kCVTFp4EltsPerThread / 2];
#pragma unroll
    for (int i = 0; i < kCVTFp4EltsPerThread / 2; i++) {
      const int tok0 = seq_lane * kCVTFp4EltsPerThread + 2 * i;
      if constexpr (std::is_same<T, half>::value) {
        fp2Vals[i].x = __half2float(shared_input[tok0 * kHeadDim + d]);
        fp2Vals[i].y = __half2float(shared_input[(tok0 + 1) * kHeadDim + d]);
      } else {
        fp2Vals[i].x = __bfloat162float(shared_input[tok0 * kHeadDim + d]);
        fp2Vals[i].y =
            __bfloat162float(shared_input[(tok0 + 1) * kHeadDim + d]);
      }
    }
    // smooth_v: quantize the residual Vhat = V - vm instead of V, where
    //   vm[d] = (1/Nkv) * sum_n V[b, hkv, n, d]   (per-(b,hkv,d) constant)
    // is subtracted per column d here and added back in the attention
    // epilogue after the softmax normalize: because softmax weights sum to
    // 1 and vm is shared by all columns,
    //   O_i = (sum_j P_ij * (Vhat_j + vm)) / sum_j P_ij
    //       = (sum_j P_ij * Vhat_j) / sum_j P_ij + vm,
    // so the subtraction is mathematically exact; it only shrinks the
    // residual dynamic range so the ue8m0 group scale quantizes finer.
    if (vm_head) {
      const float vm_d = vm_head[d];
#pragma unroll
      for (int i = 0; i < kCVTFp4EltsPerThread / 2; i++) {
        fp2Vals[i].x -= vm_d;
        fp2Vals[i].y -= vm_d;
      }
    }

    float vecMax = 0.f;
#pragma unroll
    for (int i = 0; i < kCVTFp4EltsPerThread / 2; i++) {
      vecMax = fmaxf(vecMax, fmaxf(fabsf(fp2Vals[i].x), fabsf(fp2Vals[i].y)));
    }
    // 32-token group absmax across the neighbouring pass thread.
    const float groupMax =
        fmaxf(vecMax, __shfl_xor_sync(0xFFFFFFFFu, vecMax, 1));

    float sf = groupMax / 448.f;
    int e = sf > 0.f ? int(ceilf(log2f(sf))) : -127;
    if (e < -127)
      e = -127;
    if (e > 128)
      e = 128;
    if (e < 128 && ldexpf(1.f, e) < sf)
      e += 1;
    const float scale = ldexpf(1.f, -e);

#pragma unroll
    for (int i = 0; i < kCVTFp4EltsPerThread / 2; i++) {
      fp2Vals[i].x = fp2Vals[i].x * scale;
      fp2Vals[i].y = fp2Vals[i].y * scale;
    }

    uint32_t e4m3Vals[kCVTFp4EltsPerThread / 4];
#pragma unroll
    for (int i = 0; i < kCVTFp4EltsPerThread / 4; i++) {
      e4m3Vals[i] = fp32_vec_to_e4m3(fp2Vals + i * 2);
    }

    reinterpret_cast<uint4*>(output + batch_id * stride_bz_output +
                             head_id * stride_h_output + d * stride_d_output +
                             (token_block_id * kTokensPerBlock +
                              seq_lane * kCVTFp4EltsPerThread))[0] =
        reinterpret_cast<uint4*>(e4m3Vals)[0];

    if ((seq_lane & 1) == 0) {
      reinterpret_cast<uint8_t*>(output_sf_base +
                                 (d / 64) * 64 * stride_d_output_sf +
                                 offset_local)[0] = uint8_t(e + 127);
    }
  }
}

// Per-(b,h) 128-row block means of Q: qm (B,H,ceil(Nq/128),D) fp32. Tail
// blocks average only the valid rows. Pad cols [d_og, kHeadDim) stay 0.
template <typename T, int kHeadDim>
__global__ void fp4_q_block_mean_kernel(const T* __restrict__ input,
                                        float* __restrict__ qm, int num_tokens,
                                        int stride_bz_input, int stride_h_input,
                                        int stride_seq_input, int stride_bz_qm,
                                        int stride_h_qm, int stride_m_qm,
                                        int d_og) {
  const int m_block = blockIdx.x;
  const int b = blockIdx.y;
  const int h = blockIdx.z;
  const int row0 = m_block * 128;
  const int count = max(0, min(128, num_tokens - row0));
  const T* base = input + b * stride_bz_input + h * stride_h_input;
  float* out = qm + b * stride_bz_qm + h * stride_h_qm + m_block * stride_m_qm;

  const int d = threadIdx.x;
  if (d >= kHeadDim)
    return;
  if (d >= d_og) {
    out[d] = 0.f;
    return;
  }
  float sum = 0.f;
  for (int r = 0; r < count; ++r) {
    T val = base[(row0 + r) * stride_seq_input + d];
    if constexpr (std::is_same<T, half>::value) {
      sum += __half2float(val);
    } else {
      sum += __bfloat162float(val);
    }
  }
  out[d] = count > 0 ? sum / count : 0.f;
}

namespace detail {

template <typename T, int kHeadDim>
void launch_fp4_quant_q_t(const torch::Tensor& input, torch::Tensor& output,
                          torch::Tensor& output_sf, const torch::Tensor& qm,
                          int64_t n_pad, bool sub_qm) {
  constexpr int kTokensPerBlock =
      (kHeadDim <= 128) ? 128 : ((kHeadDim <= 768) ? 64 : 32);
  const int num_tokens = input.size(1);
  const int d_og = static_cast<int>(input.size(3));
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 block(kTokensPerBlock * 4, 1, 1);
  dim3 grid((n_pad + kTokensPerBlock - 1) / kTokensPerBlock, input.size(0),
            input.size(2));
  if (sub_qm) {
    fp4_quant_kernel<T, kHeadDim, false, true, false>
        <<<grid, block, 0, stream>>>(
            reinterpret_cast<const T*>(input.data_ptr()),
            output.data_ptr<uint8_t>(), output_sf.data_ptr<uint8_t>(),
            num_tokens, input.stride(0), input.stride(2), input.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            output_sf.stride(0), output_sf.stride(1), output_sf.stride(2),
            qm.data_ptr<float>(), qm.stride(0), qm.stride(1), nullptr, 0, 0,
            d_og);
  } else {
    fp4_quant_kernel<T, kHeadDim, false, false, false>
        <<<grid, block, 0, stream>>>(
            reinterpret_cast<const T*>(input.data_ptr()),
            output.data_ptr<uint8_t>(), output_sf.data_ptr<uint8_t>(),
            num_tokens, input.stride(0), input.stride(2), input.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            output_sf.stride(0), output_sf.stride(1), output_sf.stride(2),
            nullptr, 0, 0, nullptr, 0, 0, d_og);
  }
}

template <typename T, int kHeadDim>
void launch_fp4_quant_k_t(const torch::Tensor& input, torch::Tensor& output,
                          torch::Tensor& output_sf, const torch::Tensor& km,
                          int64_t n_pad, bool sub_km) {
  constexpr int kTokensPerBlock =
      (kHeadDim <= 128) ? 128 : ((kHeadDim <= 768) ? 64 : 32);
  const int num_tokens = input.size(1);
  const int d_og = static_cast<int>(input.size(3));
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 block(kTokensPerBlock * 4, 1, 1);
  dim3 grid((n_pad + kTokensPerBlock - 1) / kTokensPerBlock, input.size(0),
            input.size(2));
  if (sub_km) {
    fp4_quant_kernel<T, kHeadDim, true, false, true>
        <<<grid, block, 0, stream>>>(
            reinterpret_cast<const T*>(input.data_ptr()),
            output.data_ptr<uint8_t>(), output_sf.data_ptr<uint8_t>(),
            num_tokens, input.stride(0), input.stride(2), input.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            output_sf.stride(0), output_sf.stride(1), output_sf.stride(2),
            nullptr, 0, 0, km.data_ptr<float>(), km.stride(0), km.stride(1),
            d_og);
  } else {
    fp4_quant_kernel<T, kHeadDim, true, false, false>
        <<<grid, block, 0, stream>>>(
            reinterpret_cast<const T*>(input.data_ptr()),
            output.data_ptr<uint8_t>(), output_sf.data_ptr<uint8_t>(),
            num_tokens, input.stride(0), input.stride(2), input.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            output_sf.stride(0), output_sf.stride(1), output_sf.stride(2),
            nullptr, 0, 0, nullptr, 0, 0, d_og);
  }
}

template <typename T, int kHeadDim>
void launch_fp4_quant_vt_t(const torch::Tensor& input, torch::Tensor& output,
                           torch::Tensor& output_sf, int64_t n_pad,
                           const torch::Tensor& vm) {
  constexpr int kTokensPerBlock =
      (kHeadDim <= 128) ? 128 : ((kHeadDim <= 768) ? 64 : 32);
  constexpr bool kDynamicSmem =
      kTokensPerBlock * kHeadDim * int(sizeof(T)) > 48 * 1024;
  constexpr int kSmemBytes =
      kDynamicSmem ? kTokensPerBlock * kHeadDim * int(sizeof(T)) : 0;
  const int num_tokens = input.size(1);
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 block(kTokensPerBlock * 4, 1, 1);
  dim3 grid((n_pad + kTokensPerBlock - 1) / kTokensPerBlock, input.size(0),
            input.size(2));
  auto kernel = fp4_quant_trans_kernel<T, kHeadDim>;
  if constexpr (kDynamicSmem) {
    TORCH_CHECK(cudaFuncSetAttribute(
                    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                    kSmemBytes) == cudaSuccess,
                "fp4 V-transpose quantize smem opt-in failed for D=", kHeadDim);
  }
  kernel<<<grid, block, kSmemBytes, stream>>>(
      reinterpret_cast<const T*>(input.data_ptr()), output.data_ptr<uint8_t>(),
      output_sf.data_ptr<uint8_t>(), num_tokens, input.stride(0),
      input.stride(2), input.stride(1), output.stride(0), output.stride(1),
      output.stride(2), output_sf.stride(0), output_sf.stride(1),
      output_sf.stride(2), static_cast<int>(input.size(3)),
      vm.defined() ? vm.data_ptr<float>() : nullptr,
      vm.defined() ? vm.size(0) * vm.size(1) : 0,
      vm.defined() ? vm.size(2) : 0);
}

template <typename T, int kHeadDim>
void launch_mxfp8_quant_vt_t(const torch::Tensor& input, torch::Tensor& output,
                             torch::Tensor& output_sf, int64_t n_pad,
                             const torch::Tensor& vm) {
  constexpr int kTokensPerBlock =
      (kHeadDim <= 128) ? 128 : ((kHeadDim <= 768) ? 64 : 32);
  constexpr bool kDynamicSmem =
      kTokensPerBlock * kHeadDim * int(sizeof(T)) > 48 * 1024;
  constexpr int kSmemBytes =
      kDynamicSmem ? kTokensPerBlock * kHeadDim * int(sizeof(T)) : 0;
  const int num_tokens = input.size(1);
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 block(kTokensPerBlock * 4, 1, 1);
  dim3 grid((n_pad + kTokensPerBlock - 1) / kTokensPerBlock, input.size(0),
            input.size(2));
  auto kernel = mxfp8_quant_trans_kernel<T, kHeadDim>;
  if constexpr (kDynamicSmem) {
    TORCH_CHECK(
        cudaFuncSetAttribute(kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             kSmemBytes) == cudaSuccess,
        "mxfp8 V-transpose quantize smem opt-in failed for D=", kHeadDim);
  }
  kernel<<<grid, block, kSmemBytes, stream>>>(
      reinterpret_cast<const T*>(input.data_ptr()), output.data_ptr<uint8_t>(),
      output_sf.data_ptr<uint8_t>(), num_tokens, input.stride(0),
      input.stride(2), input.stride(1), output.stride(0), output.stride(1),
      output.stride(2), output_sf.stride(0), output_sf.stride(1),
      output_sf.stride(2), static_cast<int>(input.size(3)),
      vm.defined() ? vm.data_ptr<float>() : nullptr,
      vm.defined() ? vm.size(0) * vm.size(1) : 0,
      vm.defined() ? vm.size(2) : 0);
}

// Fused-WHT variants: same launch shape as the plain Q/K launchers but the
// kernel rotates rows in-register; qm/km must be the PRE-ROTATED copies
// (wht_f32_rows_kernel output), delta_s/qkm keep consuming the unrotated
// ones.
template <typename T, int kHeadDim>
void launch_fp4_quant_q_wht_t(const torch::Tensor& input, torch::Tensor& output,
                              torch::Tensor& output_sf, const torch::Tensor& qm,
                              int64_t n_pad) {
  constexpr int kTokensPerBlock =
      (kHeadDim <= 128) ? 128 : ((kHeadDim <= 768) ? 64 : 32);
  const int num_tokens = input.size(1);
  const int d_og = static_cast<int>(input.size(3));
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 block(kTokensPerBlock * 4, 1, 1);
  dim3 grid((n_pad + kTokensPerBlock - 1) / kTokensPerBlock, input.size(0),
            input.size(2));
  fp4_quant_kernel<T, kHeadDim, false, true, false, true>
      <<<grid, block, 0, stream>>>(
          reinterpret_cast<const T*>(input.data_ptr()),
          output.data_ptr<uint8_t>(), output_sf.data_ptr<uint8_t>(), num_tokens,
          input.stride(0), input.stride(2), input.stride(1), output.stride(0),
          output.stride(1), output.stride(2), output_sf.stride(0),
          output_sf.stride(1), output_sf.stride(2), qm.data_ptr<float>(),
          qm.stride(0), qm.stride(1), nullptr, 0, 0, d_og);
}

template <typename T, int kHeadDim>
void launch_fp4_quant_k_wht_t(const torch::Tensor& input, torch::Tensor& output,
                              torch::Tensor& output_sf, const torch::Tensor& km,
                              int64_t n_pad) {
  constexpr int kTokensPerBlock =
      (kHeadDim <= 128) ? 128 : ((kHeadDim <= 768) ? 64 : 32);
  const int num_tokens = input.size(1);
  const int d_og = static_cast<int>(input.size(3));
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 block(kTokensPerBlock * 4, 1, 1);
  dim3 grid((n_pad + kTokensPerBlock - 1) / kTokensPerBlock, input.size(0),
            input.size(2));
  fp4_quant_kernel<T, kHeadDim, true, false, true, true>
      <<<grid, block, 0, stream>>>(
          reinterpret_cast<const T*>(input.data_ptr()),
          output.data_ptr<uint8_t>(), output_sf.data_ptr<uint8_t>(), num_tokens,
          input.stride(0), input.stride(2), input.stride(1), output.stride(0),
          output.stride(1), output.stride(2), output_sf.stride(0),
          output_sf.stride(1), output_sf.stride(2), nullptr, 0, 0,
          km.data_ptr<float>(), km.stride(0), km.stride(1), d_og);
}

}  // namespace detail

template <int kHeadDim>
inline void launch_fp4_quant_q_sm120(const torch::Tensor& input,
                                     torch::Tensor& output,
                                     torch::Tensor& output_sf,
                                     const torch::Tensor& qm, int64_t n_pad,
                                     bool sub_qm) {
  TORCH_CHECK(input.size(3) % 8 == 0 && input.size(3) <= kHeadDim &&
                  kHeadDim % 64 == 0 && kHeadDim >= 64 && kHeadDim <= 1024,
              "fp4 quantize requires head_dim %8==0, D <= ", kHeadDim);
  if (input.scalar_type() == at::ScalarType::Half) {
    detail::launch_fp4_quant_q_t<half, kHeadDim>(input, output, output_sf, qm,
                                                 n_pad, sub_qm);
  } else {
    detail::launch_fp4_quant_q_t<__nv_bfloat16, kHeadDim>(
        input, output, output_sf, qm, n_pad, sub_qm);
  }
}

template <int kHeadDim>
inline void launch_fp4_quant_k_sm120(const torch::Tensor& input,
                                     torch::Tensor& output,
                                     torch::Tensor& output_sf,
                                     const torch::Tensor& km, int64_t n_pad,
                                     bool sub_km) {
  TORCH_CHECK(input.size(3) % 8 == 0 && input.size(3) <= kHeadDim &&
                  kHeadDim % 64 == 0 && kHeadDim >= 64 && kHeadDim <= 1024,
              "fp4 quantize requires head_dim %8==0, D <= ", kHeadDim);
  if (input.scalar_type() == at::ScalarType::Half) {
    detail::launch_fp4_quant_k_t<half, kHeadDim>(input, output, output_sf, km,
                                                 n_pad, sub_km);
  } else {
    detail::launch_fp4_quant_k_t<__nv_bfloat16, kHeadDim>(
        input, output, output_sf, km, n_pad, sub_km);
  }
}

template <int kHeadDim>
inline void launch_fp4_quant_vt_sm120(const torch::Tensor& input,
                                      torch::Tensor& output,
                                      torch::Tensor& output_sf, int64_t n_pad,
                                      const torch::Tensor& vm = {}) {
  TORCH_CHECK(input.size(3) % 8 == 0 && input.size(3) <= kHeadDim &&
                  kHeadDim % 64 == 0 && kHeadDim >= 64 && kHeadDim <= 1024,
              "fp4 quantize requires head_dim %8==0, D <= ", kHeadDim);
  if (input.scalar_type() == at::ScalarType::Half) {
    detail::launch_fp4_quant_vt_t<half, kHeadDim>(input, output, output_sf,
                                                  n_pad, vm);
  } else {
    detail::launch_fp4_quant_vt_t<__nv_bfloat16, kHeadDim>(
        input, output, output_sf, n_pad, vm);
  }
}

template <int kHeadDim>
inline void launch_mxfp8_quant_vt_sm120(const torch::Tensor& input,
                                        torch::Tensor& output,
                                        torch::Tensor& output_sf, int64_t n_pad,
                                        const torch::Tensor& vm = {}) {
  TORCH_CHECK(input.size(3) % 8 == 0 && input.size(3) <= kHeadDim &&
                  kHeadDim % 64 == 0 && kHeadDim >= 64 && kHeadDim <= 1024,
              "fp4 quantize requires head_dim %8==0, D <= ", kHeadDim);
  if (input.scalar_type() == at::ScalarType::Half) {
    detail::launch_mxfp8_quant_vt_t<half, kHeadDim>(input, output, output_sf,
                                                    n_pad, vm);
  } else {
    detail::launch_mxfp8_quant_vt_t<__nv_bfloat16, kHeadDim>(
        input, output, output_sf, n_pad, vm);
  }
}

// Fused-WHT Q/K quantize: rows are Walsh-Hadamard rotated in-register at
// load time (pow2 kHeadDim only); qm/km args are the pre-rotated copies.
template <int kHeadDim>
inline void launch_fp4_quant_q_wht_sm120(const torch::Tensor& input,
                                         torch::Tensor& output,
                                         torch::Tensor& output_sf,
                                         const torch::Tensor& qm,
                                         int64_t n_pad) {
  TORCH_CHECK((kHeadDim & (kHeadDim - 1)) == 0,
              "fused-WHT fp4 quantize requires pow2 kHeadDim");
  if (input.scalar_type() == at::ScalarType::Half) {
    detail::launch_fp4_quant_q_wht_t<half, kHeadDim>(input, output, output_sf,
                                                     qm, n_pad);
  } else {
    detail::launch_fp4_quant_q_wht_t<__nv_bfloat16, kHeadDim>(
        input, output, output_sf, qm, n_pad);
  }
}

template <int kHeadDim>
inline void launch_fp4_quant_k_wht_sm120(const torch::Tensor& input,
                                         torch::Tensor& output,
                                         torch::Tensor& output_sf,
                                         const torch::Tensor& km,
                                         int64_t n_pad) {
  TORCH_CHECK((kHeadDim & (kHeadDim - 1)) == 0,
              "fused-WHT fp4 quantize requires pow2 kHeadDim");
  if (input.scalar_type() == at::ScalarType::Half) {
    detail::launch_fp4_quant_k_wht_t<half, kHeadDim>(input, output, output_sf,
                                                     km, n_pad);
  } else {
    detail::launch_fp4_quant_k_wht_t<__nv_bfloat16, kHeadDim>(
        input, output, output_sf, km, n_pad);
  }
}

template <int kHeadDim>
inline void launch_fp4_q_block_mean_sm120(const torch::Tensor& input,
                                          torch::Tensor& qm) {
  TORCH_CHECK(input.size(3) % 8 == 0 && input.size(3) <= kHeadDim &&
                  kHeadDim % 64 == 0 && kHeadDim >= 64 && kHeadDim <= 1024,
              "fp4 quantize requires head_dim %8==0, D <= ", kHeadDim);
  const int num_tokens = input.size(1);
  const int n_blocks = qm.size(2);
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(n_blocks, input.size(0), input.size(2));
  dim3 block(kHeadDim, 1, 1);
  if (input.scalar_type() == at::ScalarType::Half) {
    fp4_q_block_mean_kernel<half, kHeadDim><<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(input.data_ptr()), qm.data_ptr<float>(),
        num_tokens, input.stride(0), input.stride(2), input.stride(1),
        qm.stride(0), qm.stride(1), qm.stride(2),
        static_cast<int>(input.size(3)));
  } else {
    fp4_q_block_mean_kernel<__nv_bfloat16, kHeadDim>
        <<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            qm.data_ptr<float>(), num_tokens, input.stride(0), input.stride(2),
            input.stride(1), qm.stride(0), qm.stride(1), qm.stride(2),
            static_cast<int>(input.size(3)));
  }
}

// ===== single-launch Q/K/V quantize with fused q_mean (D <= 128) =====
// Replaces the q_mean + Q-quant + K-quant + V-quant launches (and the
// separate qm fp16 cast) with ONE kernel, following the fp8 path's
// single-launch quantize precedent. blockIdx.x is split into three
// segments [Q tiles | K tiles | V tiles]; blockIdx.y is the batch. The
// Q-tile blocks additionally emit the per-128-row-block mean qm in fp32
// AND in dtype (the mean block == the quant tile, so the reduce is
// block-local over a smem staging of the tile - the strided whole-tensor
// q_mean pass disappears). Under kHadamard the Q segment also runs the
// kHeadDim-point WHT over qm in smem (ping-pong butterfly) so the
// pre-rotated bias qm_rot no longer needs its own launch; km_rot stays an
// external small kernel (its input km is ready before this launch).
// Used by the persist_d launcher as the default quantize path for D <=
// 128 (larger head dims use the separate quant kernels below).
// Staging row stride is kHeadDim+16 elements: 16-element vec aligned and
// the +16 pad keeps the column-sum reads at 2-way bank conflicts.
template <typename T, int kHeadDim, bool kHadamard, bool kPvMxfp8>
__global__ void fp4_quant_fused_qkv_kernel(
    const T* __restrict__ q_in, uint8_t* __restrict__ q_out,
    uint8_t* __restrict__ q_sf, float* __restrict__ qm_f32,
    T* __restrict__ qm_h, float* __restrict__ qm_rot_out, int q_num_tokens,
    int q_s_bz, int q_s_h, int q_s_seq, int q_s_out_bz, int q_s_out_h,
    int q_s_out_seq, int q_s_sf_bz, int q_s_sf_h, int q_s_sf_seq, int qm_s_bz,
    int qm_s_h, int qm_s_m, const float* __restrict__ km, int km_s_b,
    int km_s_h, const T* __restrict__ k_in, uint8_t* __restrict__ k_out,
    uint8_t* __restrict__ k_sf, int k_num_tokens, int k_s_bz, int k_s_h,
    int k_s_seq, int k_s_out_bz, int k_s_out_h, int k_s_out_seq, int k_s_sf_bz,
    int k_s_sf_h, int k_s_sf_seq, const float* __restrict__ vm, int vm_s_b,
    int vm_s_h, const T* __restrict__ v_in, uint8_t* __restrict__ v_out,
    uint8_t* __restrict__ v_sf, int v_num_tokens, int v_s_bz, int v_s_h,
    int v_s_seq, int v_s_out_bz, int v_s_out_h, int v_s_out_d, int v_s_sf_bz,
    int v_s_sf_h, int v_s_sf_d, int d_og, int q_tiles, int k_tiles, int q_heads,
    int kv_heads) {
  using PackedVec = Fp4PackedVec<T>;
  constexpr int kThreadsPerToken = 4;
  constexpr int kVecsPerThread = kHeadDim / 64;
  constexpr int kE = kHeadDim / 4;
  constexpr int kStageRow = kHeadDim + 16;
  constexpr int kStageVecs = kStageRow / kCVTFp4EltsPerThread;
  __shared__ T s_stage[128 * kStageRow];
  __shared__ float s_bias[kHeadDim];
  __shared__ float s_wht[2][kHeadDim];

  const int b = blockIdx.y;
  const int t = blockIdx.x;
  const int q_total = q_heads * q_tiles;
  const int k_total = kv_heads * k_tiles;
  const int local_tok = threadIdx.x / kThreadsPerToken;
  const int slice = threadIdx.x % kThreadsPerToken;

  if (t < q_total) {
    const int h = t / q_tiles;
    const int tb = t % q_tiles;
    const int token_id = tb * 128 + local_tok;
    const bool token_valid = token_id < q_num_tokens;

    PackedVec in_vec[kVecsPerThread];
#pragma unroll
    for (int v = 0; v < kVecsPerThread; v++) {
#pragma unroll
      for (int i = 0; i < kCVTFp4EltsPerThread / 2; i++)
        reinterpret_cast<uint32_t&>(in_vec[v].elts[i]) = 0;
    }
    if (token_valid) {
      const PackedVec* __restrict__ src = reinterpret_cast<const PackedVec*>(
          q_in + b * q_s_bz + h * q_s_h + token_id * q_s_seq + slice * kE);
      const typename Fp4TypeConverter<T>::Type2* __restrict__ src2 =
          reinterpret_cast<const typename Fp4TypeConverter<T>::Type2*>(src);
#pragma unroll
      for (int v = 0; v < kVecsPerThread; v++) {
        const int off = (slice * kVecsPerThread + v) * kCVTFp4EltsPerThread;
        if (off + kCVTFp4EltsPerThread <= d_og) {
          in_vec[v] = src[v];
        } else {
#pragma unroll
          for (int hh = 0; hh < 2; hh++) {
            if (off + hh * 8 < d_og) {
#pragma unroll
              for (int i = 0; i < 4; i++)
                in_vec[v].elts[hh * 4 + i] = src2[v * 8 + hh * 4 + i];
            }
          }
        }
      }
    }
    PackedVec* stage_pv = reinterpret_cast<PackedVec*>(s_stage);
#pragma unroll
    for (int v = 0; v < kVecsPerThread; v++)
      stage_pv[local_tok * kStageVecs + slice * kVecsPerThread + v] = in_vec[v];

    float2 fp2Vals[kVecsPerThread][kCVTFp4EltsPerThread / 2];
#pragma unroll
    for (int v = 0; v < kVecsPerThread; v++) {
#pragma unroll
      for (int i = 0; i < kCVTFp4EltsPerThread / 2; i++) {
        if constexpr (std::is_same<T, half>::value) {
          fp2Vals[v][i] = __half22float2(in_vec[v].elts[i]);
        } else {
          fp2Vals[v][i] = __bfloat1622float2(in_vec[v].elts[i]);
        }
      }
    }
    if constexpr (kHadamard) {
      float x[kE];
#pragma unroll
      for (int v = 0; v < kVecsPerThread; v++)
#pragma unroll
        for (int i = 0; i < kCVTFp4EltsPerThread / 2; i++) {
          x[v * kCVTFp4EltsPerThread + 2 * i] = fp2Vals[v][i].x;
          x[v * kCVTFp4EltsPerThread + 2 * i + 1] = fp2Vals[v][i].y;
        }
#pragma unroll
      for (int dist = kE / 2; dist >= 1; dist >>= 1) {
#pragma unroll
        for (int j = 0; j < kE; j++) {
          const int jj = j ^ dist;
          if (j < jj) {
            const float a = x[j], bb = x[jj];
            x[j] = a + bb;
            x[jj] = a - bb;
          }
        }
      }
#pragma unroll
      for (int j = 0; j < kE; j++) {
        const float p = __shfl_xor_sync(0xffffffffu, x[j], 1);
        x[j] = (slice & 1) ? p - x[j] : x[j] + p;
      }
#pragma unroll
      for (int j = 0; j < kE; j++) {
        const float p = __shfl_xor_sync(0xffffffffu, x[j], 2);
        x[j] = (slice & 2) ? p - x[j] : x[j] + p;
      }
      const float s = rsqrtf((float)kHeadDim);
#pragma unroll
      for (int v = 0; v < kVecsPerThread; v++)
#pragma unroll
        for (int i = 0; i < kCVTFp4EltsPerThread / 2; i++) {
          fp2Vals[v][i].x = x[v * kCVTFp4EltsPerThread + 2 * i] * s;
          fp2Vals[v][i].y = x[v * kCVTFp4EltsPerThread + 2 * i + 1] * s;
        }
    }
    __syncthreads();

    // Block-local qm: one thread per column d sums the staged tile rows.
    float qm_d = 0.f;
    if (threadIdx.x < kHeadDim) {
      const int d = threadIdx.x;
      float sum = 0.f;
      for (int r = 0; r < 128; ++r) {
        if constexpr (std::is_same<T, half>::value) {
          sum += __half2float(s_stage[r * kStageRow + d]);
        } else {
          sum += __bfloat162float(s_stage[r * kStageRow + d]);
        }
      }
      const int cnt = max(0, min(128, q_num_tokens - tb * 128));
      qm_d = cnt > 0 ? sum / cnt : 0.f;
      const long qm_off = (long)b * qm_s_bz + h * qm_s_h + tb * qm_s_m + d;
      qm_f32[qm_off] = qm_d;
      qm_h[qm_off] = (T)qm_d;
      s_wht[0][threadIdx.x] = qm_d;
    }
    __syncthreads();
    int wht_p = 0;  // s_wht ping-pong bank after the WHT loop below
    if constexpr (kHadamard) {
      // Full kHeadDim-point WHT over qm (ping-pong smem butterfly); the
      // block reduce above only produced the unrotated value. The dist
      // loop runs OUTSIDE the threadIdx.x < kHeadDim guard so every
      // thread reaches the per-stage __syncthreads (a CTA-wide barrier
      // inside the guard would let the other 384 threads race ahead into
      // the s_bias read below).
      for (int dist = kHeadDim / 2; dist >= 1; dist >>= 1) {
        if (threadIdx.x < kHeadDim) {
          const int i = threadIdx.x, j = i ^ dist;
          const float a = s_wht[wht_p][i], bb = s_wht[wht_p][j];
          // Same (low, high) -> (sum, low - high) butterfly convention as
          // the Q/K in-register WHT above.
          s_wht[wht_p ^ 1][i] = (i < j) ? a + bb : bb - a;
        }
        wht_p ^= 1;
        __syncthreads();
      }
    }
    if (threadIdx.x < kHeadDim) {
      float bias = qm_d;
      if constexpr (kHadamard) {
        // The main kernel's lse correction needs qm_rot (the rotated mean,
        // same layout as qm); emit it here instead of a separate launch.
        bias = s_wht[wht_p][threadIdx.x] * rsqrtf((float)kHeadDim);
        qm_rot_out[(long)b * qm_s_bz + h * qm_s_h + tb * qm_s_m + threadIdx.x] =
            bias;
      }
      s_bias[threadIdx.x] = bias;
    }
    __syncthreads();
    if (token_valid) {
      const float* bias = s_bias + slice * kE;
#pragma unroll
      for (int v = 0; v < kVecsPerThread; v++) {
#pragma unroll
        for (int i = 0; i < kCVTFp4EltsPerThread / 2; i++) {
          float2 bbv = *reinterpret_cast<const float2*>(
              bias + v * kCVTFp4EltsPerThread + 2 * i);
          fp2Vals[v][i].x -= bbv.x;
          fp2Vals[v][i].y -= bbv.y;
        }
      }
    }

    uint8_t* output = q_out + b * q_s_out_bz + h * q_s_out_h;
    uint8_t* output_sf = q_sf + b * q_s_sf_bz + h * q_s_sf_h;
    uint8_t* sf_tail = output_sf + (token_id / 64) * 64 * q_s_sf_seq;
    const uint32_t token_id_local = token_id % 64;
#pragma unroll
    for (int v = 0; v < kVecsPerThread; v++) {
      float vecMax = 0.f;
#pragma unroll
      for (int i = 0; i < kCVTFp4EltsPerThread / 2; i++)
        vecMax = fmaxf(vecMax,
                       fmaxf(fabsf(fp2Vals[v][i].x), fabsf(fp2Vals[v][i].y)));
      float SFValue = vecMax / 6.0f;
      uint8_t SFValueFP8;
      reinterpret_cast<__nv_fp8_e4m3&>(SFValueFP8) = __nv_fp8_e4m3(SFValue);
      SFValue = float(reinterpret_cast<__nv_fp8_e4m3&>(SFValueFP8));
      const float SFValueInv = (SFValue == 0.0f) ? 0.0f : 1.0f / SFValue;
#pragma unroll
      for (int i = 0; i < kCVTFp4EltsPerThread / 2; i++) {
        fp2Vals[v][i].x = fp2Vals[v][i].x * SFValueInv;
        fp2Vals[v][i].y = fp2Vals[v][i].y * SFValueInv;
      }
      uint32_t e2m1Vals[kCVTFp4EltsPerThread / 8];
#pragma unroll
      for (int i = 0; i < kCVTFp4EltsPerThread / 8; i++)
        e2m1Vals[i] = fp32_vec_to_e2m1(fp2Vals[v] + i * 4);
      reinterpret_cast<uint64_t*>(output + token_id * q_s_out_seq +
                                  (slice * kVecsPerThread + v) *
                                      kCVTFp4EltsPerThread / 2)[0] =
          reinterpret_cast<uint64_t*>(e2m1Vals)[0];
      uint32_t col_id_local = slice * kVecsPerThread + v;
      uint32_t offset_local = (col_id_local / 4) * 256 + (col_id_local % 4) +
                              (token_id_local / 16) * 4 +
                              (token_id_local % 16) * 16;
      reinterpret_cast<uint8_t*>(sf_tail + offset_local)[0] = SFValueFP8;
    }
  } else if (t < q_total + k_total) {
    const int t2 = t - q_total;
    const int h = t2 / k_tiles;
    const int tb = t2 % k_tiles;
    const int load_token_id = tb * 128 + kv_perm32(local_tok);
    const int token_id = tb * 128 + local_tok;
    const bool token_valid = load_token_id < k_num_tokens;

    PackedVec in_vec[kVecsPerThread];
#pragma unroll
    for (int v = 0; v < kVecsPerThread; v++) {
#pragma unroll
      for (int i = 0; i < kCVTFp4EltsPerThread / 2; i++)
        reinterpret_cast<uint32_t&>(in_vec[v].elts[i]) = 0;
    }
    if (token_valid) {
      const PackedVec* __restrict__ src = reinterpret_cast<const PackedVec*>(
          k_in + b * k_s_bz + h * k_s_h + load_token_id * k_s_seq + slice * kE);
      const typename Fp4TypeConverter<T>::Type2* __restrict__ src2 =
          reinterpret_cast<const typename Fp4TypeConverter<T>::Type2*>(src);
#pragma unroll
      for (int v = 0; v < kVecsPerThread; v++) {
        const int off = (slice * kVecsPerThread + v) * kCVTFp4EltsPerThread;
        if (off + kCVTFp4EltsPerThread <= d_og) {
          in_vec[v] = src[v];
        } else {
#pragma unroll
          for (int hh = 0; hh < 2; hh++) {
            if (off + hh * 8 < d_og) {
#pragma unroll
              for (int i = 0; i < 4; i++)
                in_vec[v].elts[hh * 4 + i] = src2[v * 8 + hh * 4 + i];
            }
          }
        }
      }
    }
    float2 fp2Vals[kVecsPerThread][kCVTFp4EltsPerThread / 2];
#pragma unroll
    for (int v = 0; v < kVecsPerThread; v++) {
#pragma unroll
      for (int i = 0; i < kCVTFp4EltsPerThread / 2; i++) {
        if constexpr (std::is_same<T, half>::value) {
          fp2Vals[v][i] = __half22float2(in_vec[v].elts[i]);
        } else {
          fp2Vals[v][i] = __bfloat1622float2(in_vec[v].elts[i]);
        }
      }
    }
    if constexpr (kHadamard) {
      float x[kE];
#pragma unroll
      for (int v = 0; v < kVecsPerThread; v++)
#pragma unroll
        for (int i = 0; i < kCVTFp4EltsPerThread / 2; i++) {
          x[v * kCVTFp4EltsPerThread + 2 * i] = fp2Vals[v][i].x;
          x[v * kCVTFp4EltsPerThread + 2 * i + 1] = fp2Vals[v][i].y;
        }
#pragma unroll
      for (int dist = kE / 2; dist >= 1; dist >>= 1) {
#pragma unroll
        for (int j = 0; j < kE; j++) {
          const int jj = j ^ dist;
          if (j < jj) {
            const float a = x[j], bb = x[jj];
            x[j] = a + bb;
            x[jj] = a - bb;
          }
        }
      }
#pragma unroll
      for (int j = 0; j < kE; j++) {
        const float p = __shfl_xor_sync(0xffffffffu, x[j], 1);
        x[j] = (slice & 1) ? p - x[j] : x[j] + p;
      }
#pragma unroll
      for (int j = 0; j < kE; j++) {
        const float p = __shfl_xor_sync(0xffffffffu, x[j], 2);
        x[j] = (slice & 2) ? p - x[j] : x[j] + p;
      }
      const float s = rsqrtf((float)kHeadDim);
#pragma unroll
      for (int v = 0; v < kVecsPerThread; v++)
#pragma unroll
        for (int i = 0; i < kCVTFp4EltsPerThread / 2; i++) {
          fp2Vals[v][i].x = x[v * kCVTFp4EltsPerThread + 2 * i] * s;
          fp2Vals[v][i].y = x[v * kCVTFp4EltsPerThread + 2 * i + 1] * s;
        }
    }
    if (token_valid) {
      const float* km_row = km + b * km_s_b + h * km_s_h + slice * kE;
#pragma unroll
      for (int v = 0; v < kVecsPerThread; v++) {
#pragma unroll
        for (int i = 0; i < kCVTFp4EltsPerThread / 2; i++) {
          float2 bbv = *reinterpret_cast<const float2*>(
              km_row + v * kCVTFp4EltsPerThread + 2 * i);
          fp2Vals[v][i].x -= bbv.x;
          fp2Vals[v][i].y -= bbv.y;
        }
      }
    }

    uint8_t* output = k_out + b * k_s_out_bz + h * k_s_out_h;
    uint8_t* output_sf = k_sf + b * k_s_sf_bz + h * k_s_sf_h;
    uint8_t* sf_tail = output_sf + (token_id / 64) * 64 * k_s_sf_seq;
    const uint32_t token_id_local = token_id % 64;
#pragma unroll
    for (int v = 0; v < kVecsPerThread; v++) {
      float vecMax = 0.f;
#pragma unroll
      for (int i = 0; i < kCVTFp4EltsPerThread / 2; i++)
        vecMax = fmaxf(vecMax,
                       fmaxf(fabsf(fp2Vals[v][i].x), fabsf(fp2Vals[v][i].y)));
      float SFValue = vecMax / 6.0f;
      uint8_t SFValueFP8;
      reinterpret_cast<__nv_fp8_e4m3&>(SFValueFP8) = __nv_fp8_e4m3(SFValue);
      SFValue = float(reinterpret_cast<__nv_fp8_e4m3&>(SFValueFP8));
      const float SFValueInv = (SFValue == 0.0f) ? 0.0f : 1.0f / SFValue;
#pragma unroll
      for (int i = 0; i < kCVTFp4EltsPerThread / 2; i++) {
        fp2Vals[v][i].x = fp2Vals[v][i].x * SFValueInv;
        fp2Vals[v][i].y = fp2Vals[v][i].y * SFValueInv;
      }
      uint32_t e2m1Vals[kCVTFp4EltsPerThread / 8];
#pragma unroll
      for (int i = 0; i < kCVTFp4EltsPerThread / 8; i++)
        e2m1Vals[i] = fp32_vec_to_e2m1(fp2Vals[v] + i * 4);
      reinterpret_cast<uint64_t*>(output + token_id * k_s_out_seq +
                                  (slice * kVecsPerThread + v) *
                                      kCVTFp4EltsPerThread / 2)[0] =
          reinterpret_cast<uint64_t*>(e2m1Vals)[0];
      uint32_t col_id_local = slice * kVecsPerThread + v;
      uint32_t offset_local = (col_id_local / 4) * 256 + (col_id_local % 4) +
                              (token_id_local / 16) * 4 +
                              (token_id_local % 16) * 16;
      reinterpret_cast<uint8_t*>(sf_tail + offset_local)[0] = SFValueFP8;
    }
  } else {
    // V segment: fp4_quant_trans / mxfp8_quant_trans body on the shared
    // staging window (rows padded, first kHeadDim columns).
    const int t2 = t - q_total - k_total;
    const int h = t2 / k_tiles;
    const int token_block_id = t2 % k_tiles;
    const int token_id = token_block_id * 128 + local_tok;

    PackedVec in_vec[kVecsPerThread];
#pragma unroll
    for (int v = 0; v < kVecsPerThread; v++) {
#pragma unroll
      for (int i = 0; i < kCVTFp4EltsPerThread / 2; i++)
        reinterpret_cast<uint32_t&>(in_vec[v].elts[i]) = 0;
    }
    if (token_id < v_num_tokens) {
      const PackedVec* __restrict__ src = reinterpret_cast<const PackedVec*>(
          v_in + b * v_s_bz + h * v_s_h + token_id * v_s_seq + slice * kE);
      const typename Fp4TypeConverter<T>::Type2* __restrict__ src2 =
          reinterpret_cast<const typename Fp4TypeConverter<T>::Type2*>(src);
#pragma unroll
      for (int v = 0; v < kVecsPerThread; v++) {
        const int off = (slice * kVecsPerThread + v) * kCVTFp4EltsPerThread;
        if (off + kCVTFp4EltsPerThread <= d_og) {
          in_vec[v] = src[v];
        } else {
#pragma unroll
          for (int hh = 0; hh < 2; hh++) {
            if (off + hh * 8 < d_og) {
#pragma unroll
              for (int i = 0; i < 4; i++)
                in_vec[v].elts[hh * 4 + i] = src2[v * 8 + hh * 4 + i];
            }
          }
        }
      }
    }
    PackedVec* stage_pv = reinterpret_cast<PackedVec*>(s_stage);
#pragma unroll
    for (int v = 0; v < kVecsPerThread; v++)
      stage_pv[local_tok * kStageVecs + slice * kVecsPerThread + v] = in_vec[v];
    __syncthreads();

    uint8_t* output_sf_base = v_sf + b * v_s_sf_bz + h * v_s_sf_h;
    const float* vm_head = vm ? vm + b * vm_s_b + h * vm_s_h : nullptr;
    constexpr int kThreadsPerSeq = 128 / kCVTFp4EltsPerThread;
    constexpr int kDRowsPerPass = 128 * kThreadsPerToken / kThreadsPerSeq;
    const uint32_t col_id_local = token_block_id * 128 / kCVTFp4EltsPerThread +
                                  threadIdx.x % kThreadsPerSeq;
#pragma unroll
    for (int p = 0; p < kHeadDim / kDRowsPerPass; p++) {
      const int d = p * kDRowsPerPass + threadIdx.x / kThreadsPerSeq;
      float2 fp2Vals[kCVTFp4EltsPerThread / 2];
#pragma unroll
      for (int i = 0; i < kCVTFp4EltsPerThread / 2; i++) {
        const int tok0 =
            (threadIdx.x % kThreadsPerSeq) * kCVTFp4EltsPerThread + 2 * i;
        if constexpr (std::is_same<T, half>::value) {
          fp2Vals[i].x = __half2float(s_stage[tok0 * kStageRow + d]);
          fp2Vals[i].y = __half2float(s_stage[(tok0 + 1) * kStageRow + d]);
        } else {
          fp2Vals[i].x = __bfloat162float(s_stage[tok0 * kStageRow + d]);
          fp2Vals[i].y = __bfloat162float(s_stage[(tok0 + 1) * kStageRow + d]);
        }
      }
      if (vm_head) {
        const float vm_d = vm_head[d];
#pragma unroll
        for (int i = 0; i < kCVTFp4EltsPerThread / 2; i++) {
          fp2Vals[i].x -= vm_d;
          fp2Vals[i].y -= vm_d;
        }
      }
      float vecMax = 0.f;
#pragma unroll
      for (int i = 0; i < kCVTFp4EltsPerThread / 2; i++)
        vecMax = fmaxf(vecMax, fmaxf(fabsf(fp2Vals[i].x), fabsf(fp2Vals[i].y)));

      if constexpr (kPvMxfp8) {
        const float groupMax =
            fmaxf(vecMax, __shfl_xor_sync(0xFFFFFFFFu, vecMax, 1));
        float sf = groupMax / 448.f;
        int e = sf > 0.f ? int(ceilf(log2f(sf))) : -127;
        if (e < -127)
          e = -127;
        if (e > 128)
          e = 128;
        if (e < 128 && ldexpf(1.f, e) < sf)
          e += 1;
        const float scale = ldexpf(1.f, -e);
#pragma unroll
        for (int i = 0; i < kCVTFp4EltsPerThread / 2; i++) {
          fp2Vals[i].x = fp2Vals[i].x * scale;
          fp2Vals[i].y = fp2Vals[i].y * scale;
        }
        uint32_t e4m3Vals[kCVTFp4EltsPerThread / 4];
#pragma unroll
        for (int i = 0; i < kCVTFp4EltsPerThread / 4; i++)
          e4m3Vals[i] = fp32_vec_to_e4m3(fp2Vals + i * 2);
        reinterpret_cast<uint4*>(
            v_out + b * v_s_out_bz + h * v_s_out_h + d * v_s_out_d +
            (token_block_id * 128 +
             (threadIdx.x % kThreadsPerSeq) * kCVTFp4EltsPerThread))[0] =
            reinterpret_cast<uint4*>(e4m3Vals)[0];
        const uint32_t group_col_local =
            token_block_id * (128 / 32) + (threadIdx.x % kThreadsPerSeq) / 2;
        const uint32_t offset_local = (group_col_local / 4) * 256 +
                                      (group_col_local % 4) +
                                      (threadIdx.x / kThreadsPerSeq / 16) * 4 +
                                      (threadIdx.x / kThreadsPerSeq % 16) * 16;
        if (((threadIdx.x % kThreadsPerSeq) & 1) == 0) {
          reinterpret_cast<uint8_t*>(output_sf_base + (d / 64) * 64 * v_s_sf_d +
                                     offset_local)[0] = uint8_t(e + 127);
        }
      } else {
        float SFValue = vecMax / 6.0f;
        uint8_t SFValueFP8;
        reinterpret_cast<__nv_fp8_e4m3&>(SFValueFP8) = __nv_fp8_e4m3(SFValue);
        SFValue = float(reinterpret_cast<__nv_fp8_e4m3&>(SFValueFP8));
        const float SFValueInv = (SFValue == 0.0f) ? 0.0f : 1.0f / SFValue;
#pragma unroll
        for (int i = 0; i < kCVTFp4EltsPerThread / 2; i++) {
          fp2Vals[i].x = fp2Vals[i].x * SFValueInv;
          fp2Vals[i].y = fp2Vals[i].y * SFValueInv;
        }
        uint32_t e2m1Vals[kCVTFp4EltsPerThread / 8];
#pragma unroll
        for (int i = 0; i < kCVTFp4EltsPerThread / 8; i++)
          e2m1Vals[i] = fp32_vec_to_e2m1(fp2Vals + i * 4);
        reinterpret_cast<uint64_t*>(
            v_out + b * v_s_out_bz + h * v_s_out_h + d * v_s_out_d +
            (token_block_id * 128 +
             (threadIdx.x % kThreadsPerSeq) * kCVTFp4EltsPerThread) /
                2)[0] = reinterpret_cast<uint64_t*>(e2m1Vals)[0];
        const uint32_t offset_local = (col_id_local / 4) * 256 +
                                      (col_id_local % 4) +
                                      (threadIdx.x / kThreadsPerSeq / 16) * 4 +
                                      (threadIdx.x / kThreadsPerSeq % 16) * 16;
        reinterpret_cast<uint8_t*>(output_sf_base + (d / 64) * 64 * v_s_sf_d +
                                   offset_local)[0] = SFValueFP8;
      }
    }
  }
}

namespace detail {

template <typename T, int kHeadDim, bool kHadamard, bool kPvMxfp8>
void launch_fp4_quant_qkv_fused_t(
    const torch::Tensor& q_in, torch::Tensor& q_out, torch::Tensor& q_sf,
    torch::Tensor& qm_f32, torch::Tensor& qm_h, torch::Tensor& qm_rot,
    const torch::Tensor& km, const torch::Tensor& k_in, torch::Tensor& k_out,
    torch::Tensor& k_sf, const torch::Tensor& vm, const torch::Tensor& v_in,
    torch::Tensor& v_out, torch::Tensor& v_sf, int64_t q_pad, int64_t kv_pad) {
  const int q_tiles = static_cast<int>((q_pad + 127) / 128);
  const int k_tiles = static_cast<int>((kv_pad + 127) / 128);
  const int q_heads = static_cast<int>(q_in.size(2));
  const int kv_heads = static_cast<int>(k_in.size(2));
  const int d_og = static_cast<int>(q_in.size(3));
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 block(512, 1, 1);
  dim3 grid(q_heads * q_tiles + kv_heads * k_tiles * 2,
            static_cast<unsigned>(q_in.size(0)));
  fp4_quant_fused_qkv_kernel<T, kHeadDim, kHadamard, kPvMxfp8>
      <<<grid, block, 0, stream>>>(
          reinterpret_cast<const T*>(q_in.data_ptr()),
          q_out.data_ptr<uint8_t>(), q_sf.data_ptr<uint8_t>(),
          qm_f32.data_ptr<float>(), reinterpret_cast<T*>(qm_h.data_ptr()),
          kHadamard ? qm_rot.data_ptr<float>()
                    : qm_f32.data_ptr<float>(),  // placeholder when unused
          static_cast<int>(q_in.size(1)), q_in.stride(0), q_in.stride(2),
          q_in.stride(1), q_out.stride(0), q_out.stride(1), q_out.stride(2),
          q_sf.stride(0), q_sf.stride(1), q_sf.stride(2), qm_f32.stride(0),
          qm_f32.stride(1), qm_f32.stride(2), km.data_ptr<float>(),
          km.stride(0), km.stride(1),
          reinterpret_cast<const T*>(k_in.data_ptr()),
          k_out.data_ptr<uint8_t>(), k_sf.data_ptr<uint8_t>(),
          static_cast<int>(k_in.size(1)), k_in.stride(0), k_in.stride(2),
          k_in.stride(1), k_out.stride(0), k_out.stride(1), k_out.stride(2),
          k_sf.stride(0), k_sf.stride(1), k_sf.stride(2),
          vm.defined() ? vm.data_ptr<float>() : nullptr,
          vm.defined() ? static_cast<int>(vm.size(0) * vm.size(1)) : 0,
          vm.defined() ? static_cast<int>(vm.size(2)) : 0,
          reinterpret_cast<const T*>(v_in.data_ptr()),
          v_out.data_ptr<uint8_t>(), v_sf.data_ptr<uint8_t>(),
          static_cast<int>(v_in.size(1)), v_in.stride(0), v_in.stride(2),
          v_in.stride(1), v_out.stride(0), v_out.stride(1), v_out.stride(2),
          v_sf.stride(0), v_sf.stride(1), v_sf.stride(2), d_og, q_tiles,
          k_tiles, q_heads, kv_heads);
}

}  // namespace detail

// Single-launch Q/K/V fp4 quantize with the fused q_mean. km is the
// quantize bias for the K segment (km_f32, or km_rot under hadamard); vm
// is optional (fp4_smooth_v). qm_h receives the in-dtype copy of qm so the
// separate qm cast disappears. hadamard/pv_mxfp8 pick the kernel template.
template <int kHeadDim>
inline void launch_fp4_quant_qkv_fused_sm120(
    const torch::Tensor& q_in, torch::Tensor& q_out, torch::Tensor& q_sf,
    torch::Tensor& qm_f32, torch::Tensor& qm_h, torch::Tensor& qm_rot,
    const torch::Tensor& km, const torch::Tensor& k_in, torch::Tensor& k_out,
    torch::Tensor& k_sf, const torch::Tensor& vm, const torch::Tensor& v_in,
    torch::Tensor& v_out, torch::Tensor& v_sf, int64_t q_pad, int64_t kv_pad,
    bool hadamard, bool pv_mxfp8) {
  static_assert(kHeadDim <= 128, "fused QKV quantize requires D <= 128");
#define FFPA_FUSED_QKV_CALL(T, HD, MM)                                    \
  detail::launch_fp4_quant_qkv_fused_t<T, kHeadDim, HD, MM>(              \
      q_in, q_out, q_sf, qm_f32, qm_h, qm_rot, km, k_in, k_out, k_sf, vm, \
      v_in, v_out, v_sf, q_pad, kv_pad)
  if (q_in.scalar_type() == at::kHalf) {
    if (hadamard) {
      if (pv_mxfp8)
        FFPA_FUSED_QKV_CALL(half, true, true);
      else
        FFPA_FUSED_QKV_CALL(half, true, false);
    } else {
      if (pv_mxfp8)
        FFPA_FUSED_QKV_CALL(half, false, true);
      else
        FFPA_FUSED_QKV_CALL(half, false, false);
    }
  } else {
    if (hadamard) {
      if (pv_mxfp8)
        FFPA_FUSED_QKV_CALL(__nv_bfloat16, true, true);
      else
        FFPA_FUSED_QKV_CALL(__nv_bfloat16, true, false);
    } else {
      if (pv_mxfp8)
        FFPA_FUSED_QKV_CALL(__nv_bfloat16, false, true);
      else
        FFPA_FUSED_QKV_CALL(__nv_bfloat16, false, false);
    }
  }
#undef FFPA_FUSED_QKV_CALL
}

}  // namespace ffpa_fp4

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
template <typename T, int kHeadDim, bool kPermute, bool kSubQm, bool kSubKm>
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
    int stride_h_output_sf, int stride_d_output_sf, int d_og) {
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
                           torch::Tensor& output_sf, int64_t n_pad) {
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
      output_sf.stride(2), static_cast<int>(input.size(3)));
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
                                      torch::Tensor& output_sf, int64_t n_pad) {
  TORCH_CHECK(input.size(3) % 8 == 0 && input.size(3) <= kHeadDim &&
                  kHeadDim % 64 == 0 && kHeadDim >= 64 && kHeadDim <= 1024,
              "fp4 quantize requires head_dim %8==0, D <= ", kHeadDim);
  if (input.scalar_type() == at::ScalarType::Half) {
    detail::launch_fp4_quant_vt_t<half, kHeadDim>(input, output, output_sf,
                                                  n_pad);
  } else {
    detail::launch_fp4_quant_vt_t<__nv_bfloat16, kHeadDim>(input, output,
                                                           output_sf, n_pad);
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

}  // namespace ffpa_fp4

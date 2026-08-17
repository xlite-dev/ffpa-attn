// NVFP4 quantize pre-kernels, ported from SageAttention3
// fp4_quantization_4d.cu and adapted to the ffpa-attn pipeline:
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
template <typename T, int kHeadDim, bool kPermute, bool kSubQm, bool kSubKm>
__global__ void fp4_quant_kernel(
    const T* __restrict__ input, uint8_t* __restrict__ output,
    uint8_t* __restrict__ output_sf, int num_tokens, int stride_bz_input,
    int stride_h_input, int stride_seq_input, int stride_bz_output,
    int stride_h_output, int stride_seq_output, int stride_bz_output_sf,
    int stride_h_output_sf, int stride_seq_output_sf,
    const float* __restrict__ qm, int qm_stride_b, int qm_stride_h,
    const float* __restrict__ km, int km_stride_b, int km_stride_h) {
  using PackedVec = Fp4PackedVec<T>;
  constexpr int kBlock = 128;
  constexpr int kThreadsPerToken = kHeadDim / kCVTFp4EltsPerThread;

  const int batch_id = blockIdx.y;
  const int head_id = blockIdx.z;
  const int token_block_id = blockIdx.x;
  const int token_id = token_block_id * kBlock + threadIdx.x / kThreadsPerToken;

  int load_token_id;
  if constexpr (!kPermute) {
    load_token_id = token_id;
  } else {
    int local_token_id = threadIdx.x / kThreadsPerToken;
    int local_token_id_residue = local_token_id % 32;
    // [0,1,8,9,16,17,24,25,2,3,10,11,18,19,26,27,4,5,12,13,20,21,28,29,6,7,14,15,22,23,30,31]
    load_token_id = token_block_id * kBlock + (local_token_id / 32) * 32 +
                    (local_token_id_residue / 8) * 2 +
                    ((local_token_id_residue % 8) / 2) * 8 +
                    (local_token_id_residue % 8) % 2;
  }

  PackedVec in_vec;
#pragma unroll
  for (int i = 0; i < kCVTFp4EltsPerThread / 2; i++) {
    reinterpret_cast<uint32_t&>(in_vec.elts[i]) = 0;
  }
  if (load_token_id < num_tokens) {
    in_vec = reinterpret_cast<PackedVec const*>(
        input + batch_id * stride_bz_input + head_id * stride_h_input +
        load_token_id * stride_seq_input +
        (threadIdx.x % kThreadsPerToken) * kCVTFp4EltsPerThread)[0];
  }

  float2 fp2Vals[kCVTFp4EltsPerThread / 2];
#pragma unroll
  for (int i = 0; i < kCVTFp4EltsPerThread / 2; i++) {
    if constexpr (std::is_same<T, half>::value) {
      fp2Vals[i] = __half22float2(in_vec.elts[i]);
    } else {
      fp2Vals[i] = __bfloat1622float2(in_vec.elts[i]);
    }
  }
  // padded tail rows stay zero: skip the bias there (SF and data both)
  if constexpr (kSubQm) {
    if (load_token_id < num_tokens) {
      const float* qm_row =
          qm +
          (batch_id * qm_stride_b + head_id * qm_stride_h +
           (token_id / 128) * kHeadDim) +
          (threadIdx.x % kThreadsPerToken) * kCVTFp4EltsPerThread;
#pragma unroll
      for (int i = 0; i < kCVTFp4EltsPerThread / 2; i++) {
        float2 b = *reinterpret_cast<const float2*>(qm_row + 2 * i);
        fp2Vals[i].x -= b.x;
        fp2Vals[i].y -= b.y;
      }
    }
  }
  if constexpr (kSubKm) {
    if (load_token_id < num_tokens) {
      const float* km_row =
          km + batch_id * km_stride_b + head_id * km_stride_h +
          (threadIdx.x % kThreadsPerToken) * kCVTFp4EltsPerThread;
#pragma unroll
      for (int i = 0; i < kCVTFp4EltsPerThread / 2; i++) {
        float2 b = *reinterpret_cast<const float2*>(km_row + 2 * i);
        fp2Vals[i].x -= b.x;
        fp2Vals[i].y -= b.y;
      }
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
      token_id * stride_seq_output +
      (threadIdx.x % kThreadsPerToken) * kCVTFp4EltsPerThread / 2)[0] =
      reinterpret_cast<uint64_t*>(e2m1Vals)[0];

  uint8_t* output_sf_save_base = output_sf + batch_id * stride_bz_output_sf +
                                 head_id * stride_h_output_sf +
                                 (token_id / 64) * 64 * stride_seq_output_sf;
  uint32_t token_id_local = token_id % 64;
  uint32_t col_id_local = threadIdx.x % kThreadsPerToken;
  uint32_t offset_local = (col_id_local / 4) * 256 + (col_id_local % 4) +
                          (token_id_local / 16) * 4 +
                          (token_id_local % 16) * 16;
  reinterpret_cast<uint8_t*>(output_sf_save_base + offset_local)[0] =
      SFValueFP8;
}

// V quantization with transpose: (B,S,H,D) -> packed e2m1 (B,H,D,N/2) + SF
// (B,H,D,N/16) in the transposed SF atom layout. No bias (V not smoothed).
template <typename T, int kHeadDim>
__global__ void fp4_quant_trans_kernel(
    const T* __restrict__ input, uint8_t* __restrict__ output,
    uint8_t* __restrict__ output_sf, int num_tokens, int stride_bz_input,
    int stride_h_input, int stride_seq_input, int stride_bz_output,
    int stride_h_output, int stride_d_output, int stride_bz_output_sf,
    int stride_h_output_sf, int stride_d_output_sf) {
  using PackedVec = Fp4PackedVec<T>;
  constexpr int kBlock = 128;
  constexpr int kThreadsPerToken = kHeadDim / kCVTFp4EltsPerThread;
  constexpr int kThreadsPerSeq = kBlock / kCVTFp4EltsPerThread;

  const int batch_id = blockIdx.y;
  const int head_id = blockIdx.z;
  const int token_block_id = blockIdx.x;
  const int token_id = token_block_id * kBlock + threadIdx.x / kThreadsPerToken;

  PackedVec in_vec;
#pragma unroll
  for (int i = 0; i < kCVTFp4EltsPerThread / 2; i++) {
    reinterpret_cast<uint32_t&>(in_vec.elts[i]) = 0;
  }
  if (token_id < num_tokens) {
    in_vec = reinterpret_cast<PackedVec const*>(
        input + batch_id * stride_bz_input + head_id * stride_h_input +
        token_id * stride_seq_input +
        (threadIdx.x % kThreadsPerToken) * kCVTFp4EltsPerThread)[0];
  }

  __shared__ T shared_input[kBlock * kHeadDim];
  reinterpret_cast<PackedVec*>(shared_input)[threadIdx.x] = in_vec;
  __syncthreads();
#pragma unroll
  for (int i = 0; i < kCVTFp4EltsPerThread / 2; i++) {
    in_vec.elts[i].x =
        shared_input[(threadIdx.x / kThreadsPerSeq) +
                     ((threadIdx.x % kThreadsPerSeq) * kCVTFp4EltsPerThread +
                      2 * i) *
                         kHeadDim];
    in_vec.elts[i].y =
        shared_input[(threadIdx.x / kThreadsPerSeq) +
                     ((threadIdx.x % kThreadsPerSeq) * kCVTFp4EltsPerThread +
                      2 * i + 1) *
                         kHeadDim];
  }

  float2 fp2Vals[kCVTFp4EltsPerThread / 2];
#pragma unroll
  for (int i = 0; i < kCVTFp4EltsPerThread / 2; i++) {
    if constexpr (std::is_same<T, half>::value) {
      fp2Vals[i] = __half22float2(in_vec.elts[i]);
    } else {
      fp2Vals[i] = __bfloat1622float2(in_vec.elts[i]);
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
      (threadIdx.x / kThreadsPerSeq) * stride_d_output +
      (token_block_id * kBlock +
       (threadIdx.x % kThreadsPerSeq) * kCVTFp4EltsPerThread) /
          2)[0] = reinterpret_cast<uint64_t*>(e2m1Vals)[0];

  uint8_t* output_sf_save_base =
      output_sf + batch_id * stride_bz_output_sf +
      head_id * stride_h_output_sf +
      (threadIdx.x / kThreadsPerSeq / 64) * 64 * stride_d_output_sf;
  uint32_t row_id_local = (threadIdx.x / kThreadsPerSeq) % 64;
  uint32_t col_id_local = token_block_id * kBlock / kCVTFp4EltsPerThread +
                          threadIdx.x % kThreadsPerSeq;
  uint32_t offset_local = (col_id_local / 4) * 256 + (col_id_local % 4) +
                          (row_id_local / 16) * 4 + (row_id_local % 16) * 16;
  reinterpret_cast<uint8_t*>(output_sf_save_base + offset_local)[0] =
      SFValueFP8;
}

// Per-(b,h) 128-row block means of Q: qm (B,H,ceil(Nq/128),D) fp32. Tail
// blocks average only the valid rows.
template <typename T, int kHeadDim>
__global__ void fp4_q_block_mean_kernel(const T* __restrict__ input,
                                        float* __restrict__ qm, int num_tokens,
                                        int stride_bz_input, int stride_h_input,
                                        int stride_seq_input, int stride_bz_qm,
                                        int stride_h_qm, int stride_m_qm) {
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

inline void launch_fp4_quant_q_sm120(const torch::Tensor& input,
                                     torch::Tensor& output,
                                     torch::Tensor& output_sf,
                                     const torch::Tensor& qm, int64_t n_pad,
                                     bool sub_qm) {
  TORCH_CHECK(input.size(3) == 128, "fp4 quantize requires head_dim 128");
  const int num_tokens = input.size(1);
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 block(128 * 128 / kCVTFp4EltsPerThread, 1, 1);
  dim3 grid((n_pad + 127) / 128, input.size(0), input.size(2));
  if (input.scalar_type() == at::ScalarType::Half) {
    using T = half;
    if (sub_qm) {
      fp4_quant_kernel<T, 128, false, true, false><<<grid, block, 0, stream>>>(
          reinterpret_cast<const T*>(input.data_ptr()),
          output.data_ptr<uint8_t>(), output_sf.data_ptr<uint8_t>(), num_tokens,
          input.stride(0), input.stride(2), input.stride(1), output.stride(0),
          output.stride(1), output.stride(2), output_sf.stride(0),
          output_sf.stride(1), output_sf.stride(2), qm.data_ptr<float>(),
          qm.stride(0), qm.stride(1), nullptr, 0, 0);
    } else {
      fp4_quant_kernel<T, 128, false, false, false><<<grid, block, 0, stream>>>(
          reinterpret_cast<const T*>(input.data_ptr()),
          output.data_ptr<uint8_t>(), output_sf.data_ptr<uint8_t>(), num_tokens,
          input.stride(0), input.stride(2), input.stride(1), output.stride(0),
          output.stride(1), output.stride(2), output_sf.stride(0),
          output_sf.stride(1), output_sf.stride(2), nullptr, 0, 0, nullptr, 0,
          0);
    }
  } else {
    using T = __nv_bfloat16;
    if (sub_qm) {
      fp4_quant_kernel<T, 128, false, true, false><<<grid, block, 0, stream>>>(
          reinterpret_cast<const T*>(input.data_ptr()),
          output.data_ptr<uint8_t>(), output_sf.data_ptr<uint8_t>(), num_tokens,
          input.stride(0), input.stride(2), input.stride(1), output.stride(0),
          output.stride(1), output.stride(2), output_sf.stride(0),
          output_sf.stride(1), output_sf.stride(2), qm.data_ptr<float>(),
          qm.stride(0), qm.stride(1), nullptr, 0, 0);
    } else {
      fp4_quant_kernel<T, 128, false, false, false><<<grid, block, 0, stream>>>(
          reinterpret_cast<const T*>(input.data_ptr()),
          output.data_ptr<uint8_t>(), output_sf.data_ptr<uint8_t>(), num_tokens,
          input.stride(0), input.stride(2), input.stride(1), output.stride(0),
          output.stride(1), output.stride(2), output_sf.stride(0),
          output_sf.stride(1), output_sf.stride(2), nullptr, 0, 0, nullptr, 0,
          0);
    }
  }
}

inline void launch_fp4_quant_k_sm120(const torch::Tensor& input,
                                     torch::Tensor& output,
                                     torch::Tensor& output_sf,
                                     const torch::Tensor& km, int64_t n_pad,
                                     bool sub_km) {
  TORCH_CHECK(input.size(3) == 128, "fp4 quantize requires head_dim 128");
  const int num_tokens = input.size(1);
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 block(128 * 128 / kCVTFp4EltsPerThread, 1, 1);
  dim3 grid((n_pad + 127) / 128, input.size(0), input.size(2));
  if (input.scalar_type() == at::ScalarType::Half) {
    using T = half;
    if (sub_km) {
      fp4_quant_kernel<T, 128, true, false, true><<<grid, block, 0, stream>>>(
          reinterpret_cast<const T*>(input.data_ptr()),
          output.data_ptr<uint8_t>(), output_sf.data_ptr<uint8_t>(), num_tokens,
          input.stride(0), input.stride(2), input.stride(1), output.stride(0),
          output.stride(1), output.stride(2), output_sf.stride(0),
          output_sf.stride(1), output_sf.stride(2), nullptr, 0, 0,
          km.data_ptr<float>(), input.size(2) * 128, 128);
    } else {
      fp4_quant_kernel<T, 128, true, false, false><<<grid, block, 0, stream>>>(
          reinterpret_cast<const T*>(input.data_ptr()),
          output.data_ptr<uint8_t>(), output_sf.data_ptr<uint8_t>(), num_tokens,
          input.stride(0), input.stride(2), input.stride(1), output.stride(0),
          output.stride(1), output.stride(2), output_sf.stride(0),
          output_sf.stride(1), output_sf.stride(2), nullptr, 0, 0, nullptr, 0,
          0);
    }
  } else {
    using T = __nv_bfloat16;
    if (sub_km) {
      fp4_quant_kernel<T, 128, true, false, true><<<grid, block, 0, stream>>>(
          reinterpret_cast<const T*>(input.data_ptr()),
          output.data_ptr<uint8_t>(), output_sf.data_ptr<uint8_t>(), num_tokens,
          input.stride(0), input.stride(2), input.stride(1), output.stride(0),
          output.stride(1), output.stride(2), output_sf.stride(0),
          output_sf.stride(1), output_sf.stride(2), nullptr, 0, 0,
          km.data_ptr<float>(), input.size(2) * 128, 128);
    } else {
      fp4_quant_kernel<T, 128, true, false, false><<<grid, block, 0, stream>>>(
          reinterpret_cast<const T*>(input.data_ptr()),
          output.data_ptr<uint8_t>(), output_sf.data_ptr<uint8_t>(), num_tokens,
          input.stride(0), input.stride(2), input.stride(1), output.stride(0),
          output.stride(1), output.stride(2), output_sf.stride(0),
          output_sf.stride(1), output_sf.stride(2), nullptr, 0, 0, nullptr, 0,
          0);
    }
  }
}

inline void launch_fp4_quant_vt_sm120(const torch::Tensor& input,
                                      torch::Tensor& output,
                                      torch::Tensor& output_sf, int64_t n_pad) {
  TORCH_CHECK(input.size(3) == 128, "fp4 quantize requires head_dim 128");
  const int num_tokens = input.size(1);
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 block(128 * 128 / kCVTFp4EltsPerThread, 1, 1);
  dim3 grid((n_pad + 127) / 128, input.size(0), input.size(2));
  if (input.scalar_type() == at::ScalarType::Half) {
    fp4_quant_trans_kernel<half, 128><<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(input.data_ptr()),
        output.data_ptr<uint8_t>(), output_sf.data_ptr<uint8_t>(), num_tokens,
        input.stride(0), input.stride(2), input.stride(1), output.stride(0),
        output.stride(1), output.stride(2), output_sf.stride(0),
        output_sf.stride(1), output_sf.stride(2));
  } else {
    fp4_quant_trans_kernel<__nv_bfloat16, 128><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        output.data_ptr<uint8_t>(), output_sf.data_ptr<uint8_t>(), num_tokens,
        input.stride(0), input.stride(2), input.stride(1), output.stride(0),
        output.stride(1), output.stride(2), output_sf.stride(0),
        output_sf.stride(1), output_sf.stride(2));
  }
}

inline void launch_fp4_q_block_mean_sm120(const torch::Tensor& input,
                                          torch::Tensor& qm) {
  TORCH_CHECK(input.size(3) == 128, "fp4 quantize requires head_dim 128");
  const int num_tokens = input.size(1);
  const int n_blocks = qm.size(2);
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(n_blocks, input.size(0), input.size(2));
  dim3 block(128, 1, 1);
  if (input.scalar_type() == at::ScalarType::Half) {
    fp4_q_block_mean_kernel<half, 128><<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(input.data_ptr()), qm.data_ptr<float>(),
        num_tokens, input.stride(0), input.stride(2), input.stride(1),
        qm.stride(0), qm.stride(1), qm.stride(2));
  } else {
    fp4_q_block_mean_kernel<__nv_bfloat16, 128><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        qm.data_ptr<float>(), num_tokens, input.stride(0), input.stride(2),
        input.stride(1), qm.stride(0), qm.stride(1), qm.stride(2));
  }
}

}  // namespace ffpa_fp4

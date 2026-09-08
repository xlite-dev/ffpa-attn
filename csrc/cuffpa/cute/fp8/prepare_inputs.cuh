// Stage-independent fp8 input preprocessing: output allocation, smooth-K
// column mean, Q/K quantize (per-block or per-thread) and the per-channel
// V re-quantize. Shared verbatim by the persist_d / split_d / m4n2 fp8
// launchers (launch/cute_fp8.cuh). Explicitly instantiated once per
// (dtype, kBr, kBc, kHeadDim, kQKInt8) in the generated preprocess TU;
// every other TU sees extern-template declarations from
// generated/fp8_preprocess_instances.cuh and stops recompiling the
// quantize kernel family per stage variant.
#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/types.h>
#include "cute/fp8/input_layout.cuh"
#include "cute/fp8/quantize_fp8.cuh"
#include "cute/fp8/smooth_k.cuh"
#include "cute/fp8/smooth_v.cuh"
#include "native/utils.cuh"

namespace ffpa_fp8 {

struct Fp8QuantizedInputs {
  torch::Tensor q8, k8, vt8;
  torch::Tensor q_scale, k_scale, v_scale;
  // Smooth-K mean tensors keep the storage alive; empty when !smooth_k.
  torch::Tensor km, km_f32;
  // Per-channel V mean; empty unless v_per_channel.
  torch::Tensor vm;
  // Kernel-facing pointers (null when the corresponding pass is off).
  const float* km_f32_ptr = nullptr;
  const float* vm_kernel = nullptr;
};

template <typename kDataType, const int kBr, const int kBc, const int kHeadDim,
          bool kQKInt8>
Fp8QuantizedInputs prepare_fp8_inputs(
    const torch::Tensor& Q, const torch::Tensor& K, const torch::Tensor& V,
    const Fp8InputLayout& Lq, const Fp8InputLayout& Lkv,
    const Fp8InputLayout& Lv, int Nb, int Nh, int Nh_kv, int Nq, int Nkv,
    int n_rb_q, int n_rb_kv, int Nkv_pad, int D_og, bool smooth_k,
    bool qk_per_thread, bool v_per_channel, bool v_smooth_mean, float v_r,
    bool reorg_free, cudaStream_t stream) {
  auto opts_qk = torch::TensorOptions()
                     .dtype(kQKInt8 ? torch::kChar : torch::kFloat8_e4m3fn)
                     .device(Q.device());
  auto opts_u8 =
      torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(Q.device());
  auto opts_f32 =
      torch::TensorOptions().dtype(torch::kFloat32).device(Q.device());
  torch::Tensor q8 = torch::empty({Nb, Nh, Nq, kHeadDim}, opts_qk);
  torch::Tensor k8 = torch::empty({Nb, Nh_kv, Nkv, kHeadDim}, opts_qk);
  torch::Tensor vt8 = torch::empty({Nb, Nh_kv, kHeadDim, Nkv_pad}, opts_u8);
  // Per-thread QK: the Q quantize block is fixed at 128 rows (64
  // scale/block) for every variant -- m4n2's kBr=64 only sizes the
  // attention tile -- while K follows kBc (4 scale/block).
  const int n_rb_q_quant = ffpa::utils::div_ceil(Nq, 128);
  torch::Tensor q_scale = torch::empty(
      {Nb * Nh, qk_per_thread ? n_rb_q_quant * 64 : n_rb_q}, opts_f32);
  torch::Tensor k_scale = torch::empty(
      {Nb * Nh_kv, qk_per_thread ? n_rb_kv * 4 : n_rb_kv}, opts_f32);
  // Per-channel V (along D): v_scale is (bh, D) for per-channel, (bh,
  // n_rb_kv) for per-block. v_scale_quant feeds the first per-block
  // quantize pass; per-channel overwrites vt8/v_scale afterwards.
  torch::Tensor v_scale = v_per_channel
                              ? torch::empty({Nb * Nh_kv, kHeadDim}, opts_f32)
                              : torch::empty({Nb * Nh_kv, n_rb_kv}, opts_f32);
  torch::Tensor v_scale_quant =
      v_per_channel ? torch::empty({Nb * Nh_kv, n_rb_kv}, opts_f32) : v_scale;

  torch::Tensor km, km_f32, km_partials;
  const kDataType* km_ptr = nullptr;
  const float* km_f32_ptr = nullptr;
  const kDataType* q_ptr = reinterpret_cast<const kDataType*>(Q.data_ptr());
  const kDataType* k_ptr = reinterpret_cast<const kDataType*>(K.data_ptr());
  const kDataType* v_ptr = reinterpret_cast<const kDataType*>(V.data_ptr());
  if (smooth_k) {
    // Custom two-stage column mean (~50us) replacing at::mean + fp32 cast
    // (~85us); emits the in-dtype mean and its fp32 copy in one pass.
    const int mean_chunks = (Nkv + kMeanRowsPerChunk - 1) / kMeanRowsPerChunk;
    km = torch::empty({Nb * Nh_kv, kHeadDim}, K.options());
    km_f32 = torch::empty({Nb * Nh_kv, kHeadDim}, opts_f32);
    km_partials = torch::empty({Nb * Nh_kv, mean_chunks, kHeadDim}, opts_f32);
    km_ptr = reinterpret_cast<const kDataType*>(km.data_ptr());
    km_f32_ptr = km_f32.data_ptr<float>();
    launch_kv_mean_sm120<kDataType, kHeadDim>(
        k_ptr, reinterpret_cast<kDataType*>(km.data_ptr()),
        km_f32.data_ptr<float>(), km_partials.data_ptr<float>(), Nb, Nh_kv, Nkv,
        D_og, stream, &Lkv);
  }
  if (qk_per_thread) {
    launch_quantize_fp8_perthread_qk_sm120<kDataType, kBr, kBc, kHeadDim,
                                           kQKInt8>(
        q_ptr, k_ptr, v_ptr, q8.data_ptr(), k8.data_ptr(),
        reinterpret_cast<__nv_fp8_e4m3*>(vt8.data_ptr()),
        q_scale.data_ptr<float>(), k_scale.data_ptr<float>(),
        v_scale_quant.data_ptr<float>(), Nb, Nh, Nh_kv, Nq, Nkv, Nkv_pad, D_og,
        Lq, Lkv, stream, km_ptr, reorg_free, v_per_channel, &Lv);
  } else {
    launch_quantize_fp8_sm120<kDataType, kBr, kBc, kHeadDim, kQKInt8>(
        q_ptr, k_ptr, v_ptr, q8.data_ptr(), k8.data_ptr(),
        reinterpret_cast<__nv_fp8_e4m3*>(vt8.data_ptr()),
        q_scale.data_ptr<float>(), k_scale.data_ptr<float>(),
        v_scale_quant.data_ptr<float>(), Nb, Nh, Nh_kv, Nq, Nkv, Nkv_pad, D_og,
        Lq, Lkv, stream, km_ptr, reorg_free, v_per_channel, &Lv);
  }

  // Per-channel V (sage-style): re-quantize V with per-D scale via coalesced
  // stats (sum+max+min -> mean+amax) + quantize/transpose. smooth_v subtracts
  // the per-D mean (residual amax); overwrites the per-block vt8/v_scale.
  torch::Tensor vm, v_partials_sum, v_partials_max, v_partials_min;
  float* vm_ptr = nullptr;
  if (v_per_channel) {
    const int stats_chunks =
        (Nkv + kVStatsRowsPerChunk - 1) / kVStatsRowsPerChunk;
    v_partials_sum =
        torch::empty({Nb * Nh_kv, stats_chunks, kHeadDim}, opts_f32);
    v_partials_max =
        torch::empty({Nb * Nh_kv, stats_chunks, kHeadDim}, opts_f32);
    v_partials_min =
        torch::empty({Nb * Nh_kv, stats_chunks, kHeadDim}, opts_f32);
    vm = torch::empty({Nb * Nh_kv, kHeadDim}, opts_f32);
    vm_ptr = vm.data_ptr<float>();
    if (v_smooth_mean) {
      launch_quantize_fp8_vt_perchannel_sm120<kDataType, kBr, kBc, kHeadDim,
                                              true>(
          v_ptr, reinterpret_cast<__nv_fp8_e4m3*>(vt8.data_ptr()),
          v_scale.data_ptr<float>(), vm_ptr, v_partials_sum.data_ptr<float>(),
          v_partials_max.data_ptr<float>(), v_partials_min.data_ptr<float>(),
          Nb, Nh_kv, Nkv, Nkv_pad, stream, D_og, v_r, reorg_free, &Lv);
    } else {
      launch_quantize_fp8_vt_perchannel_sm120<kDataType, kBr, kBc, kHeadDim,
                                              false>(
          v_ptr, reinterpret_cast<__nv_fp8_e4m3*>(vt8.data_ptr()),
          v_scale.data_ptr<float>(), vm_ptr, v_partials_sum.data_ptr<float>(),
          v_partials_max.data_ptr<float>(), v_partials_min.data_ptr<float>(),
          Nb, Nh_kv, Nkv, Nkv_pad, stream, D_og, v_r, reorg_free, &Lv);
    }
  }

  Fp8QuantizedInputs out;
  out.q8 = q8;
  out.k8 = k8;
  out.vt8 = vt8;
  out.q_scale = q_scale;
  out.k_scale = k_scale;
  out.v_scale = v_scale;
  out.km = km;
  out.km_f32 = km_f32;
  out.vm = vm;
  out.km_f32_ptr = km_f32_ptr;
  out.vm_kernel = v_smooth_mean ? vm_ptr : nullptr;
  return out;
}

}  // namespace ffpa_fp8

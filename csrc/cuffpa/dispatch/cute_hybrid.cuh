// Hybrid stage-1 prep shared by the fp8/fp4 dispatch families (moved
// verbatim from the old cute/launch.cuh).
#pragma once
#include <torch/types.h>
#include <ATen/ops/constant_pad_nd.h>
#include "layout.cuh"

namespace ffpa {

#ifdef ENABLE_FFPA_CUTE_EXT
#ifdef ENABLE_FFPA_TMA_EXT
// Hybrid Stage-1 prep: slice the early rows and, when head_dim is padded,
// zero-pad them to kHeadDim so the fp16 launcher's TMA stride matches D_pad.
// Returns new tensors; the original Q/K/V stay D_og-wide (fp8 quantize reads
// D_og natively). Zero-fill keeps QK^T/PV dot products exact.
static inline void prepare_hybrid_stage1(
    torch::Tensor& Q_e, torch::Tensor& K_e, torch::Tensor& V_e,
    const torch::Tensor& Q, const torch::Tensor& K, const torch::Tensor& V,
    int64_t n_early, int64_t Nkv, int64_t Nq, int causal, int64_t D_og,
    int64_t D_pad, bool d_padded) {
  const int64_t kv_offset = Nkv - Nq;
  if (d_padded) {
    const int64_t pad_cols = D_pad - D_og;
    Q_e = torch::constant_pad_nd(Q.slice(2, 0, n_early), {0, pad_cols}, 0.0);
    if (causal != 0) {
      K_e = torch::constant_pad_nd(K.slice(2, 0, kv_offset + n_early),
                                   {0, pad_cols}, 0.0);
      V_e = torch::constant_pad_nd(V.slice(2, 0, kv_offset + n_early),
                                   {0, pad_cols}, 0.0);
    } else {
      K_e = torch::constant_pad_nd(K, {0, pad_cols}, 0.0);
      V_e = torch::constant_pad_nd(V, {0, pad_cols}, 0.0);
    }
  } else {
    Q_e = Q.slice(2, 0, n_early).contiguous();
    if (causal != 0) {
      K_e = K.slice(2, 0, kv_offset + n_early).contiguous();
      V_e = V.slice(2, 0, kv_offset + n_early).contiguous();
    } else {
      // Stage-1 runs an fp16 cute kernel (persist-D or split-D/m4n2 by
      // head_dim) that consumes packed and strided-NHD K/V natively, so
      // pass the original layouts through zero-copy. Mixed K/V layout
      // families (e.g. BHND K + strided-NHD V) are legal for the fp8/fp4
      // stage-2 impls but not for the fp16 stage-1 kernel (k_nhd == v_nhd),
      // so materialize both on a family mismatch.
      const bool k_nhd_family = ffpa_is_nhd_view(K) || ffpa_is_strided_nhd(K);
      const bool v_nhd_family = ffpa_is_nhd_view(V) || ffpa_is_strided_nhd(V);
      if (k_nhd_family != v_nhd_family) {
        K_e = K.contiguous();
        V_e = V.contiguous();
      } else {
        K_e = K;
        V_e = V;
      }
    }
  }
}
#endif
#endif

}  // namespace ffpa

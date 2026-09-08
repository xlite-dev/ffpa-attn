// Dispatch interface layer for the family-split CUDA forward build.
//
// The routing logic itself lives in launch/router.cuh
// (launch_ffpa_attn_fwd_template); this header only carries the shared
// parameter struct and the family entry template declarations so that
// dispatcher TUs stay free of CUTLASS/kernel headers. Definitions live in
// dispatch/{native_fp16,cute_fp16,cute_fp8,cute_fp4,cute_hybrid}.cuh and
// are explicitly instantiated by the generated per-family TUs (see env.py).
#pragma once
#include <torch/types.h>

namespace ffpa {

// Aggregated forward arguments handed from the routing layer to a family
// entry. Derived fields (Nb..Nkv, D_og, d_padded, qkv_padded,
// has_attn_bias) are computed ONCE in the routing layer after any
// NHD/pad rewrite of Q/K/V; family implementations must consume them
// as-is and never recompute them from the (possibly rewritten) tensors.
struct FfpaFwdParams {
  torch::Tensor Q;
  torch::Tensor K;
  torch::Tensor V;
  torch::Tensor O;
  torch::Tensor attn_bias;
  torch::Tensor softmax_lse;
  int causal = 0;
  double softmax_scale = 0.0;
  double dropout_p = 0.0;
  int64_t philox_seed = 0;
  int64_t philox_offset = 0;
  bool fp8_smooth_k = true;
  bool fp8_smooth_v = false;
  int64_t fp8_q_quant_method = 0;
  int64_t fp8_k_quant_method = 0;
  int64_t fp8_v_quant_method = 1;
  int64_t fp8_pv_acc_type = 1;
  int64_t fp8_qk_mm_type = 0;
  bool fp8_hybrid = false;
  int64_t fp8_hybrid_n_early = 256;
  bool fp4_hybrid = false;
  int64_t fp4_hybrid_n_early = 256;
  bool fp8_hadamard = false;
  bool fp4_hadamard = false;
  int64_t fp4_pv_mm_type = 0;
  bool fp4_smooth_v = false;
  int64_t Nb = 0;
  int64_t Nh = 0;
  int64_t Nh_kv = 0;
  int64_t Nq = 0;
  int64_t Nkv = 0;
  int D_og = 0;
  bool d_padded = false;
  bool qkv_padded = false;
  bool has_attn_bias = false;
};

// Native cp.async general path incl. the Nq==1 split-KV decode fast-path
// (wraps launch_native_fwd_split_d_sm80).
template <typename kDataType, const int kHeadDim, const int kMmaAccFloat32QK,
          const int kMmaAccFloat32PV, const int kStage>
void ffpa_fwd_native_sm80(const FfpaFwdParams& p);

// Native TMA path: sm90/100 WS dual-config + sm120 non-WS (runtime prop
// selection inside; the sm120 call clamps kStage to <=3 as before).
template <typename kDataType, const int kHeadDim, const int kMmaAccFloat32QK,
          const int kMmaAccFloat32PV, const int kStage>
void ffpa_fwd_native_tma(const FfpaFwdParams& p);

// CuTe TMA fp16/bf16 sm120 family (persist-D / split-D / M4N2 by the
// headdim gates). Routing-level fallbacks (bias/dropout -> native, D%32
// != 0 -> native) stay in launch/router.cuh and are NOT part of this entry.
template <typename kDataType, const int kHeadDim, const int kStage>
void ffpa_fwd_cute_fp16(const FfpaFwdParams& p);

// CuTe cp.async sm80 path (CUTE hint, no TMA; internal stage clamps for
// sm>=120 preserved verbatim).
template <typename kDataType, const int kHeadDim, const int kStage>
void ffpa_fwd_cute_fp16_sm80(const FfpaFwdParams& p);

// Hybrid stage-1 fp16 kernel selection for the fp8/fp4 paths: the caller
// (ffpa_fwd_fp8 / ffpa_fwd_fp4) prepares the early-row sub-problem
// (Q_e/K_e/V_e/O_e/bias_e/lse_e via prepare_hybrid_stage1) and passes it
// in FfpaFwdParams; this entry only dispatches on the headdim gates:
// persist-D for kHeadDim <= kPersistMaxD (fp8: 224, fp4: 256), split-D
// (32,64) below 768, M4N2 above.
template <typename kDataType, const int kHeadDim, const int kStage,
          const int kPersistMaxD>
void ffpa_fwd_fp16_stage1(const FfpaFwdParams& p);

// CUTE_TMA_FP8 family: force-kernel A/B env, persist/split/M4N2 headdim
// gates, hybrid orchestration (stage-1 via ffpa_fwd_fp16_stage1<...,224>).
template <typename kDataType, const int kHeadDim, const int kStage>
void ffpa_fwd_fp8(const FfpaFwdParams& p);

// CUTE_TMA_FP4 family: headdim gates, hybrid orchestration (stage-1 via
// ffpa_fwd_fp16_stage1<...,256>). kStage is accepted for a uniform TU
// scheme but unused (fp4 stages are fixed by the traits).
template <typename kDataType, const int kHeadDim, const int kStage>
void ffpa_fwd_fp4(const FfpaFwdParams& p);

}  // namespace ffpa

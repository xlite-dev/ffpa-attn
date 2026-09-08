// CUTE_TMA_FP8 family entry (moved verbatim from the force_fp8 branch of
// launch.cuh's TMA super-path). Hybrid stage-1 goes through
// ffpa_fwd_fp16_stage1<...,224> instead of direct fp16 launcher calls.
#pragma once
#include <cstdlib>
#include <cstring>
#include "dispatch.cuh"
#include "dispatch/hybrid.cuh"
#include "cute/launch.cuh"

namespace ffpa {

#ifdef ENABLE_FFPA_TMA_EXT
template <typename kDataType, const int kHeadDim, const int kStage>
void ffpa_fwd_fp8(const FfpaFwdParams& p) {
  // q/k quant: per_block (0) for all headdims; per_thread (2) for
  // all headdims (persist_d + split_d + m4n2 paths).
  TORCH_CHECK((p.fp8_q_quant_method == 0 && p.fp8_k_quant_method == 0) ||
                  (p.fp8_q_quant_method == 2 && p.fp8_k_quant_method == 2),
              "ffpa_attn: Q/K quant method must be both per_block or "
              "both per_thread");
#ifdef ENABLE_FFPA_CUTE_EXT
#ifdef ENABLE_FFPA_FP8_BUILD_DEBUG
  // EXPERIMENT: FFPA_FP8_FORCE_KERNEL=split_d|m4n2 forces a specific
  // split-D kernel to A/B test the M8N1/M4N2 dispatch cross-point.
  // Applies only to 224 < D <= 1024; persist-D (D<=224) is unaffected.
  // Unset -> normal headdim-based dispatch below. Debug-build only: both
  // forced entries instantiate, doubling the fp8 attention codegen per TU.
  if constexpr (kHeadDim > 224 && kHeadDim <= 1024) {
    const char* fk = getenv("FFPA_FP8_FORCE_KERNEL");
    if (fk != nullptr) {
      if (std::strcmp(fk, "split_d") == 0) {
        launch_cute_fwd_split_d_fp8_sm120<kDataType, kHeadDim, kStage>(
            p.Q, p.K, p.V, p.O, p.attn_bias, p.softmax_lse, p.causal,
            p.softmax_scale, p.dropout_p, p.philox_seed, p.philox_offset,
            p.fp8_smooth_k, p.fp8_smooth_v, p.fp8_q_quant_method,
            p.fp8_k_quant_method, p.fp8_v_quant_method, p.fp8_pv_acc_type,
            p.fp8_qk_mm_type,
            /*q_start_row=*/0, p.fp8_hadamard);
        return;
      } else if (std::strcmp(fk, "m4n2") == 0) {
        launch_cute_fwd_split_d_m4n2_fp8_sm120<kDataType, kHeadDim, kStage>(
            p.Q, p.K, p.V, p.O, p.attn_bias, p.softmax_lse, p.causal,
            p.softmax_scale, p.dropout_p, p.philox_seed, p.philox_offset,
            p.fp8_smooth_k, p.fp8_smooth_v, p.fp8_q_quant_method,
            p.fp8_k_quant_method, p.fp8_v_quant_method, p.fp8_pv_acc_type,
            p.fp8_qk_mm_type,
            /*q_start_row=*/0, p.fp8_hadamard);
        return;
      }
    }
  }
#endif  // ENABLE_FFPA_FP8_BUILD_DEBUG
  // NHD (diffusers BNHD) views and strided fused-QKV rows compose
  // with hybrid across persist-D/split-D/m4n2 (RFC FC-3): the fp16
  // stage-1 kernels consume them natively, prepare_hybrid_stage1
  // materializes BHND only for the causal/padded slices, and stage-2
  // quantize is layout-generic with q_start_row offsetting just the
  // attention grid.
  // D<=224: persist-D fp8; 224<D<768: split-D M8N1 fp8;
  // D>=768: split-D M4N2 fp8. Same D<768/D>=768 cross-point as the
  // fp16 dispatch (M4N2 wins only for D>=768; below that M8N1 is
  // faster even with D/2 reg spill, same as fp16).
  if constexpr (kHeadDim <= 224) {
    if (p.fp8_hybrid && p.Nq >= p.fp8_hybrid_n_early) {
      const int n_early = static_cast<int>(p.fp8_hybrid_n_early);
      TORCH_CHECK(n_early % 128 == 0,
                  "ffpa_attn: fp8_hybrid_n_early must be multiple of 128");
      torch::Tensor Q_e, K_e, V_e;
      prepare_hybrid_stage1(Q_e, K_e, V_e, p.Q, p.K, p.V, n_early, p.Nkv, p.Nq,
                            p.causal, p.D_og, kHeadDim, p.d_padded);
      auto O_e = torch::empty_like(Q_e);
      auto lse_e = torch::empty(
          {p.Nb, p.Nh, n_early},
          torch::TensorOptions().dtype(torch::kFloat32).device(p.Q.device()));
      // Stage-1 bias slice: the fp16 family has no q_start_row (rows
      // addressed from 0), so hand it only the [0, n_early) rows.
      auto bias_e = p.attn_bias.numel() > 0 ? p.attn_bias.slice(2, 0, n_early)
                                            : p.attn_bias;
      FfpaFwdParams p1;
      p1.Q = Q_e;
      p1.K = K_e;
      p1.V = V_e;
      p1.O = O_e;
      p1.attn_bias = bias_e;
      p1.softmax_lse = lse_e;
      p1.causal = p.causal;
      p1.softmax_scale = p.softmax_scale;
      ffpa_fwd_fp16_stage1<kDataType, kHeadDim, kStage, 224>(p1);
      p.O.slice(2, 0, n_early).copy_(O_e);
      if (p.softmax_lse.numel() > 0)
        p.softmax_lse.slice(2, 0, n_early).copy_(lse_e);
      // Stage 2: fp8 late rows [n_early:N] via q_start_row offset.
      launch_cute_fwd_persist_d_fp8_sm120<kDataType, kHeadDim, kStage>(
          p.Q, p.K, p.V, p.O, p.attn_bias, p.softmax_lse, p.causal,
          p.softmax_scale, p.dropout_p, p.philox_seed, p.philox_offset,
          p.fp8_smooth_k, p.fp8_smooth_v, p.fp8_q_quant_method,
          p.fp8_k_quant_method, p.fp8_v_quant_method, p.fp8_pv_acc_type,
          p.fp8_qk_mm_type,
          /*q_start_row=*/n_early, p.fp8_hadamard);
    } else {
      launch_cute_fwd_persist_d_fp8_sm120<kDataType, kHeadDim, kStage>(
          p.Q, p.K, p.V, p.O, p.attn_bias, p.softmax_lse, p.causal,
          p.softmax_scale, p.dropout_p, p.philox_seed, p.philox_offset,
          p.fp8_smooth_k, p.fp8_smooth_v, p.fp8_q_quant_method,
          p.fp8_k_quant_method, p.fp8_v_quant_method, p.fp8_pv_acc_type,
          p.fp8_qk_mm_type,
          /*q_start_row=*/0, p.fp8_hadamard);
    }
  } else if constexpr (kHeadDim < 768) {
    if (p.fp8_hybrid && p.Nq >= p.fp8_hybrid_n_early) {
      const int n_early = static_cast<int>(p.fp8_hybrid_n_early);
      TORCH_CHECK(n_early % 128 == 0,
                  "ffpa_attn: fp8_hybrid_n_early must be multiple of 128");
      torch::Tensor Q_e, K_e, V_e;
      prepare_hybrid_stage1(Q_e, K_e, V_e, p.Q, p.K, p.V, n_early, p.Nkv, p.Nq,
                            p.causal, p.D_og, kHeadDim, p.d_padded);
      auto O_e = torch::empty_like(Q_e);
      auto lse_e = torch::empty(
          {p.Nb, p.Nh, n_early},
          torch::TensorOptions().dtype(torch::kFloat32).device(p.Q.device()));
      auto bias_e = p.attn_bias.numel() > 0 ? p.attn_bias.slice(2, 0, n_early)
                                            : p.attn_bias;
      FfpaFwdParams p1;
      p1.Q = Q_e;
      p1.K = K_e;
      p1.V = V_e;
      p1.O = O_e;
      p1.attn_bias = bias_e;
      p1.softmax_lse = lse_e;
      p1.causal = p.causal;
      p1.softmax_scale = p.softmax_scale;
      ffpa_fwd_fp16_stage1<kDataType, kHeadDim, kStage, 224>(p1);
      p.O.slice(2, 0, n_early).copy_(O_e);
      if (p.softmax_lse.numel() > 0)
        p.softmax_lse.slice(2, 0, n_early).copy_(lse_e);
      launch_cute_fwd_split_d_fp8_sm120<kDataType, kHeadDim, kStage>(
          p.Q, p.K, p.V, p.O, p.attn_bias, p.softmax_lse, p.causal,
          p.softmax_scale, p.dropout_p, p.philox_seed, p.philox_offset,
          p.fp8_smooth_k, p.fp8_smooth_v, p.fp8_q_quant_method,
          p.fp8_k_quant_method, p.fp8_v_quant_method, p.fp8_pv_acc_type,
          p.fp8_qk_mm_type,
          /*q_start_row=*/n_early, p.fp8_hadamard);
    } else {
      launch_cute_fwd_split_d_fp8_sm120<kDataType, kHeadDim, kStage>(
          p.Q, p.K, p.V, p.O, p.attn_bias, p.softmax_lse, p.causal,
          p.softmax_scale, p.dropout_p, p.philox_seed, p.philox_offset,
          p.fp8_smooth_k, p.fp8_smooth_v, p.fp8_q_quant_method,
          p.fp8_k_quant_method, p.fp8_v_quant_method, p.fp8_pv_acc_type,
          p.fp8_qk_mm_type,
          /*q_start_row=*/0, p.fp8_hadamard);
    }
  } else {
    if (p.fp8_hybrid && p.Nq >= p.fp8_hybrid_n_early) {
      const int n_early = static_cast<int>(p.fp8_hybrid_n_early);
      TORCH_CHECK(n_early % 64 == 0,
                  "ffpa_attn: fp8_hybrid_n_early must be multiple of 64");
      torch::Tensor Q_e, K_e, V_e;
      prepare_hybrid_stage1(Q_e, K_e, V_e, p.Q, p.K, p.V, n_early, p.Nkv, p.Nq,
                            p.causal, p.D_og, kHeadDim, p.d_padded);
      auto O_e = torch::empty_like(Q_e);
      auto lse_e = torch::empty(
          {p.Nb, p.Nh, n_early},
          torch::TensorOptions().dtype(torch::kFloat32).device(p.Q.device()));
      auto bias_e = p.attn_bias.numel() > 0 ? p.attn_bias.slice(2, 0, n_early)
                                            : p.attn_bias;
      FfpaFwdParams p1;
      p1.Q = Q_e;
      p1.K = K_e;
      p1.V = V_e;
      p1.O = O_e;
      p1.attn_bias = bias_e;
      p1.softmax_lse = lse_e;
      p1.causal = p.causal;
      p1.softmax_scale = p.softmax_scale;
      ffpa_fwd_fp16_stage1<kDataType, kHeadDim, kStage, 224>(p1);
      p.O.slice(2, 0, n_early).copy_(O_e);
      if (p.softmax_lse.numel() > 0)
        p.softmax_lse.slice(2, 0, n_early).copy_(lse_e);
      launch_cute_fwd_split_d_m4n2_fp8_sm120<kDataType, kHeadDim, kStage>(
          p.Q, p.K, p.V, p.O, p.attn_bias, p.softmax_lse, p.causal,
          p.softmax_scale, p.dropout_p, p.philox_seed, p.philox_offset,
          p.fp8_smooth_k, p.fp8_smooth_v, p.fp8_q_quant_method,
          p.fp8_k_quant_method, p.fp8_v_quant_method, p.fp8_pv_acc_type,
          p.fp8_qk_mm_type,
          /*q_start_row=*/n_early, p.fp8_hadamard);
    } else {
      launch_cute_fwd_split_d_m4n2_fp8_sm120<kDataType, kHeadDim, kStage>(
          p.Q, p.K, p.V, p.O, p.attn_bias, p.softmax_lse, p.causal,
          p.softmax_scale, p.dropout_p, p.philox_seed, p.philox_offset,
          p.fp8_smooth_k, p.fp8_smooth_v, p.fp8_q_quant_method,
          p.fp8_k_quant_method, p.fp8_v_quant_method, p.fp8_pv_acc_type,
          p.fp8_qk_mm_type,
          /*q_start_row=*/0, p.fp8_hadamard);
    }
  }
#else
  TORCH_CHECK(false, "ffpa_attn: cute ext not compiled");
#endif
}
#else
template <typename kDataType, const int kHeadDim, const int kStage>
void ffpa_fwd_fp8(const FfpaFwdParams& p) {
  TORCH_CHECK(false, "ffpa_attn: cute_tma_fp8 path not compiled");
}
#endif

}  // namespace ffpa

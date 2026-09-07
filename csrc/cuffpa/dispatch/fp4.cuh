// CUTE_TMA_FP4 family entry (moved verbatim from the force_fp4 branch of
// launch.cuh's TMA super-path). Hybrid stage-1 goes through
// ffpa_fwd_fp16_stage1<...,256> instead of direct fp16 launcher calls.
#pragma once
#include "dispatch.cuh"
#include "dispatch/hybrid.cuh"
#include "cute/launch.cuh"

namespace ffpa {

#ifdef ENABLE_FFPA_TMA_EXT
template <typename kDataType, const int kHeadDim, const int kStage>
void ffpa_fwd_fp4(const FfpaFwdParams& p) {
  // NVFP4 persist-D: quantize pre-kernels + blockscaled mma. No knobs
  // (kStages fixed by traits); dropout unsupported. Causal early
  // rows fall back to the fp16 persist_d kernel (hybrid), same
  // as fp8: P-quantization noise on short-row softmax rows.
  TORCH_CHECK(p.dropout_p == 0.0, "fp4 sm120 path does not support dropout");
#ifdef ENABLE_FFPA_CUTE_EXT
  // NHD (BNHD) views and strided fused-QKV rows are consumed natively
  // by the fp4 pre-kernels (Fp8InputLayout strides) and every fp16
  // stage-1 variant (persist-D, split-D, m4n2); the causal/padded
  // slices inside prepare_hybrid_stage1 materialize BHND. Hybrid
  // therefore composes with any layout family (RFC FC-3).
  // fp4 persist-D covers 64-multiple headdims in [64,256], split-D
  // fp4 covers (256,768); D>=768 lands in the m4n2 branch. The first
  // if constexpr also keeps the hybrid stage-1 fp16 persist-D out of
  // the D>=320 TUs: its smem stages formula yields 0 there (zero-
  // sized array) and fp16's own dispatch never instantiates it for
  // those headdims.
  if constexpr (kHeadDim % 64 == 0 && kHeadDim >= 64 && kHeadDim <= 256) {
    if (p.fp4_hybrid && p.Nq >= p.fp4_hybrid_n_early) {
      const int n_early = static_cast<int>(p.fp4_hybrid_n_early);
      TORCH_CHECK(n_early % 128 == 0,
                  "ffpa_attn: fp4_hybrid_n_early must be multiple of 128");
      torch::Tensor Q_e, K_e, V_e;
      // Stage-1 fp16 kernel needs D_pad-wide inputs: it must pad the
      // early-row slices only on the fused path (Q/K/V still original
      // width); the torch-padded path is already kHeadDim-wide.
      const bool stage1_needs_pad = p.d_padded && !p.qkv_padded;
      prepare_hybrid_stage1(Q_e, K_e, V_e, p.Q, p.K, p.V, n_early, p.Nkv, p.Nq,
                            p.causal,
                            stage1_needs_pad ? p.D_og : (int64_t)kHeadDim,
                            kHeadDim, stage1_needs_pad);
      auto O_e = torch::empty_like(Q_e);
      auto lse_e = torch::empty(
          {p.Nb, p.Nh, n_early},
          torch::TensorOptions().dtype(torch::kFloat32).device(p.Q.device()));
      // Stage-1 sees only the [0, n_early) query rows; slice the bias
      // to match (the fp16 family has no q_start_row, rows are
      // addressed from 0). Stage-2 passes the full bias and offsets
      // rows via q_start_row.
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
      ffpa_fwd_fp16_stage1<kDataType, kHeadDim, kStage, 256>(p1);
      p.O.slice(2, 0, n_early).copy_(O_e);
      if (p.softmax_lse.numel() > 0)
        p.softmax_lse.slice(2, 0, n_early).copy_(lse_e);
      // Stage 2: fp4 late rows [n_early:N) via q_start_row offset.
      launch_cute_fwd_persist_d_fp4_sm120<kDataType, kHeadDim, kStage>(
          p.Q, p.K, p.V, p.O, p.attn_bias, p.softmax_lse, p.causal,
          p.softmax_scale,
          /*q_start_row=*/n_early, p.fp4_hadamard,
          static_cast<int>(p.fp4_pv_mm_type), p.fp4_smooth_v);
    } else {
      launch_cute_fwd_persist_d_fp4_sm120<kDataType, kHeadDim, kStage>(
          p.Q, p.K, p.V, p.O, p.attn_bias, p.softmax_lse, p.causal,
          p.softmax_scale,
          /*q_start_row=*/0, p.fp4_hadamard, static_cast<int>(p.fp4_pv_mm_type),
          p.fp4_smooth_v);
    }
  } else if constexpr (kHeadDim % 64 == 0 && kHeadDim > 256 && kHeadDim < 768) {
    // Split-D fp4. Hybrid stage-1 runs the fp16 split_d kernel (the
    // fp16 persist-D cannot be instantiated at D>=320: zero-sized
    // stage array, see the comment above). smooth_v/MXFP8-PV follow
    // the quantize chain + kernel branches (RFC FC-6).
    if (p.fp4_hybrid && p.Nq >= p.fp4_hybrid_n_early) {
      const int n_early = static_cast<int>(p.fp4_hybrid_n_early);
      TORCH_CHECK(n_early % 128 == 0,
                  "ffpa_attn: fp4_hybrid_n_early must be multiple of 128");
      torch::Tensor Q_e, K_e, V_e;
      const bool stage1_needs_pad = p.d_padded && !p.qkv_padded;
      prepare_hybrid_stage1(Q_e, K_e, V_e, p.Q, p.K, p.V, n_early, p.Nkv, p.Nq,
                            p.causal,
                            stage1_needs_pad ? p.D_og : (int64_t)kHeadDim,
                            kHeadDim, stage1_needs_pad);
      auto O_e = torch::empty_like(Q_e);
      auto lse_e = torch::empty(
          {p.Nb, p.Nh, n_early},
          torch::TensorOptions().dtype(torch::kFloat32).device(p.Q.device()));
      // Stage-1 bias slice: see the persist_d hybrid block above.
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
      ffpa_fwd_fp16_stage1<kDataType, kHeadDim, kStage, 256>(p1);
      p.O.slice(2, 0, n_early).copy_(O_e);
      if (p.softmax_lse.numel() > 0)
        p.softmax_lse.slice(2, 0, n_early).copy_(lse_e);
      // Stage 2: fp4 late rows [n_early:N) via q_start_row offset.
      launch_cute_fwd_split_d_fp4_sm120<kDataType, kHeadDim, kStage>(
          p.Q, p.K, p.V, p.O, p.attn_bias, p.softmax_lse, p.causal,
          p.softmax_scale,
          /*q_start_row=*/n_early, p.fp4_hadamard,
          static_cast<int>(p.fp4_pv_mm_type), p.fp4_smooth_v);
    } else {
      launch_cute_fwd_split_d_fp4_sm120<kDataType, kHeadDim, kStage>(
          p.Q, p.K, p.V, p.O, p.attn_bias, p.softmax_lse, p.causal,
          p.softmax_scale,
          /*q_start_row=*/0, p.fp4_hadamard, static_cast<int>(p.fp4_pv_mm_type),
          p.fp4_smooth_v);
    }
  } else if constexpr (kHeadDim % 64 == 0 && kHeadDim >= 768 &&
                       kHeadDim <= 1024) {
    // Split-D m4n2 fp4. Hybrid stage-1 runs the fp16 m4n2 kernel
    // (same tile geometry); stage-2 takes the q_start_row offset.
    // smooth_v supported (FC-6); MXFP8-PV is architectural N/A (the
    // wrapper rejects it: MXFP8 PV atom needs Tile-K=128 > kBc=64).
    if (p.fp4_hybrid && p.Nq >= p.fp4_hybrid_n_early) {
      const int n_early = static_cast<int>(p.fp4_hybrid_n_early);
      TORCH_CHECK(n_early % 128 == 0,
                  "ffpa_attn: fp4_hybrid_n_early must be multiple of 128");
      torch::Tensor Q_e, K_e, V_e;
      const bool stage1_needs_pad = p.d_padded && !p.qkv_padded;
      prepare_hybrid_stage1(Q_e, K_e, V_e, p.Q, p.K, p.V, n_early, p.Nkv, p.Nq,
                            p.causal,
                            stage1_needs_pad ? p.D_og : (int64_t)kHeadDim,
                            kHeadDim, stage1_needs_pad);
      auto O_e = torch::empty_like(Q_e);
      auto lse_e = torch::empty(
          {p.Nb, p.Nh, n_early},
          torch::TensorOptions().dtype(torch::kFloat32).device(p.Q.device()));
      // Stage-1 bias slice: see the persist_d hybrid block above.
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
      ffpa_fwd_fp16_stage1<kDataType, kHeadDim, kStage, 256>(p1);
      p.O.slice(2, 0, n_early).copy_(O_e);
      if (p.softmax_lse.numel() > 0)
        p.softmax_lse.slice(2, 0, n_early).copy_(lse_e);
      launch_cute_fwd_split_d_m4n2_fp4_sm120<kDataType, kHeadDim, kStage>(
          p.Q, p.K, p.V, p.O, p.attn_bias, p.softmax_lse, p.causal,
          p.softmax_scale,
          /*q_start_row=*/n_early, p.fp4_hadamard,
          static_cast<int>(p.fp4_pv_mm_type), p.fp4_smooth_v);
    } else {
      launch_cute_fwd_split_d_m4n2_fp4_sm120<kDataType, kHeadDim, kStage>(
          p.Q, p.K, p.V, p.O, p.attn_bias, p.softmax_lse, p.causal,
          p.softmax_scale,
          /*q_start_row=*/0, p.fp4_hadamard, static_cast<int>(p.fp4_pv_mm_type),
          p.fp4_smooth_v);
    }
  } else {
    TORCH_CHECK(false,
                "ffpa_attn: fp4 requires 64-multiple head_dim in "
                "[64,1024]");
  }
#else
  TORCH_CHECK(false, "ffpa_attn: cute ext not compiled");
#endif
}
#else
template <typename kDataType, const int kHeadDim, const int kStage>
void ffpa_fwd_fp4(const FfpaFwdParams& p) {
  TORCH_CHECK(false, "ffpa_attn: cute_tma_fp4 path not compiled");
}
#endif

}  // namespace ffpa

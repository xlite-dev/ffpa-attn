#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/constant_pad_nd.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstring>
#include <optional>
#include "backend.h"
#include "native/launch.cuh"
#ifdef ENABLE_FFPA_CUTE_EXT
#include "cute/launch.cuh"
#endif
using namespace ffpa;

// Runtime arguments:
//   Q, K, V, O     : BHND tensors as described in the kernel template docs.
//   causal         : 0/1 runtime flag. Non-zero enables causal masking with
//                    queries aligned to the KV tail; requires Nkv >= Nq.
//   softmax_scale  : pre-softmax scaling factor applied to QK^T. Matches the
//                    flash-attn naming; the Python wrapper defaults it to
//                    ``1 / sqrt(D)`` when the caller does not supply one.
// Runtime ``tma`` is accepted for API compatibility but ignored. The legacy
// SM90 TMA CUDA branch is kept under csrc/cuffpa/deprecated; active native
// forward launches always use the architecture-agnostic templates here.
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
template <typename kDataType, const int kHeadDim, const int kMmaAccFloat32QK,
          const int kMmaAccFloat32PV, const int kStage>
void launch_ffpa_attn_fwd_template(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
    torch::Tensor attn_bias, torch::Tensor softmax_lse, int causal,
    double softmax_scale, double dropout_p, int64_t philox_seed,
    int64_t philox_offset, bool fp8_smooth_k, bool fp8_smooth_v,
    int64_t fp8_q_quant_method, int64_t fp8_k_quant_method,
    int64_t fp8_v_quant_method, int64_t fp8_pv_acc_type, int64_t fp8_qk_mm_type,
    bool fp8_hybrid = false, int64_t fp8_hybrid_n_early = 256,
    bool fp4_hybrid = false, int64_t fp4_hybrid_n_early = 256,
    bool fp8_hadamard = false, bool fp4_hadamard = false,
    int64_t fp4_pv_mm_type = 0, bool fp4_smooth_v = false) {
  // Q,K,V,O with [B, H, N, D] layout, B=batch, H=head, N=seqlen, D=dim
  // TODO: support BNHD layout, Q,K,V,O with [B, N, H, D] layout.
  // Native block-tile config (MMA atoms, Br/Bc, stages, smem/pad flags) and
  // the Nq==1 decode fast-path live in
  // native/launch.cuh::launch_native_fwd_split_d_sm80. CuTe uses its own
  // traits. This top-level entry only validates shapes and dispatches to a
  // backend.
  TORCH_CHECK(K.size(0) == Q.size(0) && V.size(0) == Q.size(0),
              "ffpa_attn: Q/K/V must share the same batch size");
  TORCH_CHECK(K.size(1) == V.size(1),
              "ffpa_attn: K and V must share the same num_heads (Nh_kv)");
  TORCH_CHECK(
      Q.size(1) % K.size(1) == 0,
      "ffpa_attn: Q num_heads must be an integer multiple of K/V num_heads "
      "(GQA/MQA group_size = Nh_q / Nh_kv)");
  TORCH_CHECK(K.size(2) == V.size(2),
              "ffpa_attn: K and V must have identical sequence length (Nkv)");
  TORCH_CHECK(K.size(3) == Q.size(3) && V.size(3) == Q.size(3),
              "ffpa_attn: Q/K/V must share the same head dim");
  TORCH_CHECK(causal == 0 || K.size(2) >= Q.size(2),
              "ffpa_attn: causal attention requires Nkv >= Nq (queries are "
              "aligned to the tail of the KV sequence)");
  const int Nb = Q.size(0);
  const int Nh = Q.size(1);
  const int Nh_kv = K.size(1);
  const int Nq = Q.size(2);
  const int Nkv = K.size(2);
  const bool has_attn_bias = attn_bias.numel() != 0;
  const bool has_dropout = dropout_p > 0.0;
  TORCH_CHECK(causal == 0 || !has_attn_bias,
              "ffpa_attn: explicit attn_mask should not be set when causal "
              "attention is enabled");
  if (has_attn_bias) {
    TORCH_CHECK(attn_bias.is_cuda(),
                "ffpa_attn: attn_mask must be a CUDA tensor");
    TORCH_CHECK(attn_bias.device() == Q.device(),
                "ffpa_attn: attn_mask must be on the same device as Q/K/V");
    TORCH_CHECK(
        attn_bias.dim() == 4,
        "ffpa_attn: normalized attn_mask must be 4-D [B, Nh_q, Nq, Nkv]");
    TORCH_CHECK(attn_bias.size(0) == 1 || attn_bias.size(0) == Nb,
                "ffpa_attn: attn_mask batch dimension must be 1 or B");
    TORCH_CHECK(attn_bias.size(1) == 1 || attn_bias.size(1) == Nh,
                "ffpa_attn: attn_mask head dimension must be 1 or Nh_q");
    TORCH_CHECK(attn_bias.size(2) == 1 || attn_bias.size(2) == Nq,
                "ffpa_attn: attn_mask query dimension must be 1 or Nq");
    TORCH_CHECK(attn_bias.size(3) == 1 || attn_bias.size(3) == Nkv,
                "ffpa_attn: attn_mask key dimension must be 1 or Nkv");
    TORCH_CHECK(attn_bias.stride(3) == 1,
                "ffpa_attn: normalized attn_mask must be contiguous along the "
                "key dimension");
    const auto bias_type = attn_bias.scalar_type();
    TORCH_CHECK(bias_type == torch::kFloat32 || bias_type == torch::kHalf ||
                    bias_type == torch::kBFloat16,
                "ffpa_attn: attn_mask dtype must be fp16, bf16, or fp32");
    TORCH_CHECK(bias_type == torch::kFloat32 || bias_type == Q.scalar_type(),
                "ffpa_attn: attn_mask dtype must be fp32 or match Q dtype");
  }
  // Backend implementation hint: override path selection when explicitly set.
  // AUTO is treated as NATIVE: tma/cute paths are opt-in only.
  const auto impl_hint = ffpa::get_backend_impl_hint();
  const bool force_native = (impl_hint == ffpa::CudaBackendImpl::NATIVE ||
                             impl_hint == ffpa::CudaBackendImpl::AUTO);
  const bool force_tma = (impl_hint == ffpa::CudaBackendImpl::TMA);
  const bool force_cute = (impl_hint == ffpa::CudaBackendImpl::CUTE);
  const bool force_cute_tma = (impl_hint == ffpa::CudaBackendImpl::CUTE_TMA);
  const bool force_fp8 = (impl_hint == ffpa::CudaBackendImpl::CUTE_TMA_FP8);
  const bool force_fp4 = (impl_hint == ffpa::CudaBackendImpl::CUTE_TMA_FP4);
#ifdef ENABLE_FFPA_CUTE_EXT
#ifdef ENABLE_FFPA_TMA_EXT
  // NHD (diffusers BNHD) permute-view inputs are consumed natively by the
  // fp8/fp4 cute paths (pre-kernels via Fp8InputLayout/strides + batched 4D
  // TMA in the persist-D hybrid stage-1 fp16 kernel) and by the whole sm_120
  // fp16/bf16 cute family (persist-D, split-D, M4N2). Every other fp16
  // backend materializes packed copies here (same cost as a caller-side
  // permute+contiguous) instead of silently corrupting.
  bool nhd_in =
      ffpa_is_nhd_view(Q) || ffpa_is_nhd_view(K) || ffpa_is_nhd_view(V);
  // Strided-NHD inputs (fused-QKV interleaved chunk views): neither
  // BHND-packed nor a packed-NHD permute view.
  const bool strided_in = (!ffpa_is_bhnd_packed(Q) && !ffpa_is_nhd_view(Q)) ||
                          (!ffpa_is_bhnd_packed(K) && !ffpa_is_nhd_view(K)) ||
                          (!ffpa_is_bhnd_packed(V) && !ffpa_is_nhd_view(V));
  auto prop_nhd = at::cuda::getCurrentDeviceProperties();
  const bool fp16_nhd_ok = !force_tma && !force_native && !force_cute &&
                           prop_nhd->major >= 12 && kHeadDim % 32 == 0 &&
                           attn_bias.numel() == 0 && dropout_p == 0.0;
  if (nhd_in && !force_fp8 && !force_fp4 && !fp16_nhd_ok) {
    if (ffpa_is_nhd_view(Q))
      Q = Q.contiguous();
    if (ffpa_is_nhd_view(K))
      K = K.contiguous();
    if (ffpa_is_nhd_view(V))
      V = V.contiguous();
    nhd_in = false;
  }
  // Strided-NHD inputs are consumed natively by the fp8/fp4 families
  // (relaxed ffpa_layout_of gate across persist-D/split-D/M4N2) and the
  // whole fp16/bf16 CUTE_TMA family (stride-parameterized TMA rows);
  // every other backend indexes packed storage and would silently
  // mis-index. Materialize them the same way unsupported NHD views are
  // materialized above.
  const bool fp16_strided_ok = fp16_nhd_ok;
  if (strided_in && !force_fp8 && !force_fp4 && !fp16_strided_ok) {
    if (!ffpa_is_bhnd_packed(Q) && !ffpa_is_nhd_view(Q))
      Q = Q.contiguous();
    if (!ffpa_is_bhnd_packed(K) && !ffpa_is_nhd_view(K))
      K = K.contiguous();
    if (!ffpa_is_bhnd_packed(V) && !ffpa_is_nhd_view(V))
      V = V.contiguous();
  }
#endif
#endif

  // NHD (BNHD) O is stored natively by the sm120 CUTE_TMA kernels
  // (persist-D and split-D/M4N2, each with a runtime nhd_out branch).
  // The fp8/fp4 families guard their own dispatch branches below; this
  // check covers the fp16/bf16 family, where every other path (TMA/NATIVE/
  // CUTE hints, sm80/sm90) stores through a BHND-packed descriptor and
  // would silently corrupt an NHD-packed O.
  if (ffpa_is_nhd_view(O) && !force_fp8 && !force_fp4) {
    auto prop_o = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(prop_o->major >= 12 && force_cute_tma && kHeadDim % 32 == 0,
                "ffpa_attn: NHD (BNHD) output requires the fp16 sm120 CUTE_TMA "
                "path (%32 == 0)");
  }

  // fp16/bf16 head_dim pad: non-32-multiple D_og (e.g. 120) zero-pads Q/K/V
  // to the compiled kHeadDim. fp8 skips (quantize reads D_og natively); O is
  // padded by ffpa_api.cc. Only reachable via the CUTE_TMA/CUTE pad paths
  // (native/AUTO always have D_og == kHeadDim), so the TMA and cp.async
  // dispatch below both see D_pad-wide Q/K/V. fp4 also skips when D_og%8==0
  // (the api gate): its quantize/delta_s kernels read the original width and
  // zero-fill pad cols (no pad copy); FFPA_FP4_PAD_TORCH=1 forces the torch
  // pad path for A/B comparison and as a fallback.
  const int D_og = Q.size(3);
  const bool d_padded = D_og != kHeadDim;
  const bool fp4_fused =
      force_fp4 && D_og % 8 == 0 && getenv("FFPA_FP4_PAD_TORCH") == nullptr;
  const bool qkv_padded = d_padded && !force_fp8 && !fp4_fused;
  if (qkv_padded) {
    const int64_t pad_cols = kHeadDim - D_og;
    Q = torch::constant_pad_nd(Q, {0, pad_cols}, 0.0);
    K = torch::constant_pad_nd(K, {0, pad_cols}, 0.0);
    V = torch::constant_pad_nd(V, {0, pad_cols}, 0.0);
  }

  // SM120 TMA path: when ``tma`` is set and the device is TMA-capable
  // (sm_90+), delegate to the TMA launcher. Falls back to the legacy
  // cp.async path on older hardware. NOTE: NO WGMMA on sm_120a.
  // See fwd_sm120.cuh header for register pressure analysis
  // and why sm_120a uses non-WS mode (kNonWS=1).
  //
  // TMA path dispatch:
  //   sm_120a: non-WS (kNonWS=1). All 256 threads do MMA, thread 0 issues
  //     TMA inline. Eliminates WS if/else register allocation penalty
  //     (168→255 regs, 0 spills). +2-7% vs legacy cp.async baseline.
  //   sm_90/100: WS (kNonWS=0). setmaxnreg effective, 228KB smem allows
  //     deep pipeline. Unverified on real hardware.
#ifdef ENABLE_FFPA_TMA_EXT
  if ((force_tma || force_cute_tma || force_fp8 || force_fp4) &&
      !force_native && !force_cute) {
    auto prop = at::cuda::getCurrentDeviceProperties();
    if (prop->major >= 9) {
      if (force_fp4) {
        // NVFP4 persist-D: quantize pre-kernels + blockscaled mma. No knobs
        // (kStages fixed by traits); attn_bias/dropout unsupported. Causal
        // early rows fall back to the fp16 persist_d kernel (hybrid), same
        // as fp8: P-quantization noise on short-row softmax rows.
        TORCH_CHECK(attn_bias.numel() == 0 && dropout_p == 0.0,
                    "fp4 sm120 path does not support attn_bias/dropout");
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
          if (fp4_hybrid && Nq >= fp4_hybrid_n_early) {
            const int n_early = static_cast<int>(fp4_hybrid_n_early);
            TORCH_CHECK(
                n_early % 128 == 0,
                "ffpa_attn: fp4_hybrid_n_early must be multiple of 128");
            torch::Tensor Q_e, K_e, V_e;
            // Stage-1 fp16 kernel needs D_pad-wide inputs: it must pad the
            // early-row slices only on the fused path (Q/K/V still original
            // width); the torch-padded path is already kHeadDim-wide.
            const bool stage1_needs_pad = d_padded && !qkv_padded;
            prepare_hybrid_stage1(Q_e, K_e, V_e, Q, K, V, n_early, Nkv, Nq,
                                  causal,
                                  stage1_needs_pad ? D_og : (int64_t)kHeadDim,
                                  kHeadDim, stage1_needs_pad);
            auto O_e = torch::empty_like(Q_e);
            auto lse_e =
                torch::empty({Nb, Nh, n_early}, torch::TensorOptions()
                                                    .dtype(torch::kFloat32)
                                                    .device(Q.device()));
            auto empty_bias = torch::empty({0}, attn_bias.options());
            launch_cute_fwd_persist_d_sm120<kDataType, kHeadDim, kStage>(
                Q_e, K_e, V_e, O_e, empty_bias, lse_e, causal, softmax_scale,
                0.0, 0, 0);
            O.slice(2, 0, n_early).copy_(O_e);
            if (softmax_lse.numel() > 0)
              softmax_lse.slice(2, 0, n_early).copy_(lse_e);
            // Stage 2: fp4 late rows [n_early:N) via q_start_row offset.
            launch_cute_fwd_persist_d_fp4_sm120<kDataType, kHeadDim, kStage>(
                Q, K, V, O, softmax_lse, causal, softmax_scale,
                /*q_start_row=*/n_early, fp4_hadamard,
                static_cast<int>(fp4_pv_mm_type), fp4_smooth_v);
          } else {
            launch_cute_fwd_persist_d_fp4_sm120<kDataType, kHeadDim, kStage>(
                Q, K, V, O, softmax_lse, causal, softmax_scale,
                /*q_start_row=*/0, fp4_hadamard,
                static_cast<int>(fp4_pv_mm_type), fp4_smooth_v);
          }
        } else if constexpr (kHeadDim % 64 == 0 && kHeadDim > 256 &&
                             kHeadDim < 768) {
          // Split-D fp4. Hybrid stage-1 runs the fp16 split_d kernel (the
          // fp16 persist-D cannot be instantiated at D>=320: zero-sized
          // stage array, see the comment above).
          TORCH_CHECK(!fp4_smooth_v,
                      "ffpa_attn: fp4_smooth_v supports persist_d (D<=256)");
          if (fp4_hybrid && Nq >= fp4_hybrid_n_early) {
            const int n_early = static_cast<int>(fp4_hybrid_n_early);
            TORCH_CHECK(
                n_early % 128 == 0,
                "ffpa_attn: fp4_hybrid_n_early must be multiple of 128");
            torch::Tensor Q_e, K_e, V_e;
            const bool stage1_needs_pad = d_padded && !qkv_padded;
            prepare_hybrid_stage1(Q_e, K_e, V_e, Q, K, V, n_early, Nkv, Nq,
                                  causal,
                                  stage1_needs_pad ? D_og : (int64_t)kHeadDim,
                                  kHeadDim, stage1_needs_pad);
            auto O_e = torch::empty_like(Q_e);
            auto lse_e =
                torch::empty({Nb, Nh, n_early}, torch::TensorOptions()
                                                    .dtype(torch::kFloat32)
                                                    .device(Q.device()));
            auto empty_bias = torch::empty({0}, attn_bias.options());
            launch_cute_fwd_split_d_sm120<kDataType, kHeadDim, kStage, 32, 64>(
                Q_e, K_e, V_e, O_e, empty_bias, lse_e, causal, softmax_scale,
                0.0, 0, 0);
            O.slice(2, 0, n_early).copy_(O_e);
            if (softmax_lse.numel() > 0)
              softmax_lse.slice(2, 0, n_early).copy_(lse_e);
            // Stage 2: fp4 late rows [n_early:N) via q_start_row offset.
            launch_cute_fwd_split_d_fp4_sm120<kDataType, kHeadDim, kStage>(
                Q, K, V, O, softmax_lse, causal, softmax_scale,
                /*q_start_row=*/n_early, fp4_hadamard,
                static_cast<int>(fp4_pv_mm_type));
          } else {
            launch_cute_fwd_split_d_fp4_sm120<kDataType, kHeadDim, kStage>(
                Q, K, V, O, softmax_lse, causal, softmax_scale,
                /*q_start_row=*/0, fp4_hadamard,
                static_cast<int>(fp4_pv_mm_type));
          }
        } else if constexpr (kHeadDim % 64 == 0 && kHeadDim >= 768 &&
                             kHeadDim <= 1024) {
          // Split-D m4n2 fp4. Hybrid stage-1 runs the fp16 m4n2 kernel
          // (same tile geometry); stage-2 takes the q_start_row offset.
          TORCH_CHECK(!fp4_smooth_v,
                      "ffpa_attn: fp4_smooth_v supports persist_d (D<=256)");
          if (fp4_hybrid && Nq >= fp4_hybrid_n_early) {
            const int n_early = static_cast<int>(fp4_hybrid_n_early);
            TORCH_CHECK(
                n_early % 128 == 0,
                "ffpa_attn: fp4_hybrid_n_early must be multiple of 128");
            torch::Tensor Q_e, K_e, V_e;
            const bool stage1_needs_pad = d_padded && !qkv_padded;
            prepare_hybrid_stage1(Q_e, K_e, V_e, Q, K, V, n_early, Nkv, Nq,
                                  causal,
                                  stage1_needs_pad ? D_og : (int64_t)kHeadDim,
                                  kHeadDim, stage1_needs_pad);
            auto O_e = torch::empty_like(Q_e);
            auto lse_e =
                torch::empty({Nb, Nh, n_early}, torch::TensorOptions()
                                                    .dtype(torch::kFloat32)
                                                    .device(Q.device()));
            auto empty_bias = torch::empty({0}, attn_bias.options());
            launch_cute_fwd_split_d_m4n2_sm120<kDataType, kHeadDim, kStage>(
                Q_e, K_e, V_e, O_e, empty_bias, lse_e, causal, softmax_scale,
                0.0, 0, 0);
            O.slice(2, 0, n_early).copy_(O_e);
            if (softmax_lse.numel() > 0)
              softmax_lse.slice(2, 0, n_early).copy_(lse_e);
            launch_cute_fwd_split_d_m4n2_fp4_sm120<kDataType, kHeadDim, kStage>(
                Q, K, V, O, softmax_lse, causal, softmax_scale,
                /*q_start_row=*/n_early, fp4_hadamard,
                static_cast<int>(fp4_pv_mm_type));
          } else {
            launch_cute_fwd_split_d_m4n2_fp4_sm120<kDataType, kHeadDim, kStage>(
                Q, K, V, O, softmax_lse, causal, softmax_scale,
                /*q_start_row=*/0, fp4_hadamard,
                static_cast<int>(fp4_pv_mm_type));
          }
        } else {
          TORCH_CHECK(false,
                      "ffpa_attn: fp4 requires 64-multiple head_dim in "
                      "[64,1024]");
        }
        return;
      }
      if (force_fp8) {
        // q/k quant: per_block (0) for all headdims; per_thread (2) for
        // all headdims (persist_d + split_d + m4n2 paths).
        TORCH_CHECK((fp8_q_quant_method == 0 && fp8_k_quant_method == 0) ||
                        (fp8_q_quant_method == 2 && fp8_k_quant_method == 2),
                    "ffpa_attn: Q/K quant method must be both per_block or "
                    "both per_thread");
#ifdef ENABLE_FFPA_CUTE_EXT
        // EXPERIMENT: FFPA_FP8_FORCE_KERNEL=split_d|m4n2 forces a specific
        // split-D kernel to A/B test the M8N1/M4N2 dispatch cross-point.
        // Applies only to 224 < D <= 1024; persist-D (D<=224) is unaffected.
        // Unset -> normal headdim-based dispatch below.
        if constexpr (kHeadDim > 224 && kHeadDim <= 1024) {
          const char* fk = getenv("FFPA_FP8_FORCE_KERNEL");
          if (fk != nullptr) {
            if (std::strcmp(fk, "split_d") == 0) {
              launch_cute_fwd_split_d_fp8_sm120<kDataType, kHeadDim, kStage>(
                  Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
                  dropout_p, philox_seed, philox_offset, fp8_smooth_k,
                  fp8_smooth_v, fp8_q_quant_method, fp8_k_quant_method,
                  fp8_v_quant_method, fp8_pv_acc_type, fp8_qk_mm_type,
                  /*q_start_row=*/0, fp8_hadamard);
              return;
            } else if (std::strcmp(fk, "m4n2") == 0) {
              launch_cute_fwd_split_d_m4n2_fp8_sm120<kDataType, kHeadDim,
                                                     kStage>(
                  Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
                  dropout_p, philox_seed, philox_offset, fp8_smooth_k,
                  fp8_smooth_v, fp8_q_quant_method, fp8_k_quant_method,
                  fp8_v_quant_method, fp8_pv_acc_type, fp8_qk_mm_type,
                  /*q_start_row=*/0, fp8_hadamard);
              return;
            }
          }
        }
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
          if (fp8_hybrid && Nq >= fp8_hybrid_n_early) {
            const int n_early = static_cast<int>(fp8_hybrid_n_early);
            TORCH_CHECK(
                n_early % 128 == 0,
                "ffpa_attn: fp8_hybrid_n_early must be multiple of 128");
            torch::Tensor Q_e, K_e, V_e;
            prepare_hybrid_stage1(Q_e, K_e, V_e, Q, K, V, n_early, Nkv, Nq,
                                  causal, D_og, kHeadDim, d_padded);
            auto O_e = torch::empty_like(Q_e);
            auto lse_e =
                torch::empty({Nb, Nh, n_early}, torch::TensorOptions()
                                                    .dtype(torch::kFloat32)
                                                    .device(Q.device()));
            auto empty_bias = torch::empty({0}, attn_bias.options());
            launch_cute_fwd_persist_d_sm120<kDataType, kHeadDim, kStage>(
                Q_e, K_e, V_e, O_e, empty_bias, lse_e, causal, softmax_scale,
                0.0, 0, 0);
            O.slice(2, 0, n_early).copy_(O_e);
            if (softmax_lse.numel() > 0)
              softmax_lse.slice(2, 0, n_early).copy_(lse_e);
            // Stage 2: fp8 late rows [n_early:N] via q_start_row offset.
            launch_cute_fwd_persist_d_fp8_sm120<kDataType, kHeadDim, kStage>(
                Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
                dropout_p, philox_seed, philox_offset, fp8_smooth_k,
                fp8_smooth_v, fp8_q_quant_method, fp8_k_quant_method,
                fp8_v_quant_method, fp8_pv_acc_type, fp8_qk_mm_type,
                /*q_start_row=*/n_early, fp8_hadamard);
          } else {
            launch_cute_fwd_persist_d_fp8_sm120<kDataType, kHeadDim, kStage>(
                Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
                dropout_p, philox_seed, philox_offset, fp8_smooth_k,
                fp8_smooth_v, fp8_q_quant_method, fp8_k_quant_method,
                fp8_v_quant_method, fp8_pv_acc_type, fp8_qk_mm_type,
                /*q_start_row=*/0, fp8_hadamard);
          }
        } else if constexpr (kHeadDim < 768) {
          if (fp8_hybrid && Nq >= fp8_hybrid_n_early) {
            const int n_early = static_cast<int>(fp8_hybrid_n_early);
            TORCH_CHECK(
                n_early % 128 == 0,
                "ffpa_attn: fp8_hybrid_n_early must be multiple of 128");
            torch::Tensor Q_e, K_e, V_e;
            prepare_hybrid_stage1(Q_e, K_e, V_e, Q, K, V, n_early, Nkv, Nq,
                                  causal, D_og, kHeadDim, d_padded);
            auto O_e = torch::empty_like(Q_e);
            auto lse_e =
                torch::empty({Nb, Nh, n_early}, torch::TensorOptions()
                                                    .dtype(torch::kFloat32)
                                                    .device(Q.device()));
            auto empty_bias = torch::empty({0}, attn_bias.options());
            launch_cute_fwd_split_d_sm120<kDataType, kHeadDim, kStage, 32, 64>(
                Q_e, K_e, V_e, O_e, empty_bias, lse_e, causal, softmax_scale,
                0.0, 0, 0);
            O.slice(2, 0, n_early).copy_(O_e);
            if (softmax_lse.numel() > 0)
              softmax_lse.slice(2, 0, n_early).copy_(lse_e);
            launch_cute_fwd_split_d_fp8_sm120<kDataType, kHeadDim, kStage>(
                Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
                dropout_p, philox_seed, philox_offset, fp8_smooth_k,
                fp8_smooth_v, fp8_q_quant_method, fp8_k_quant_method,
                fp8_v_quant_method, fp8_pv_acc_type, fp8_qk_mm_type,
                /*q_start_row=*/n_early, fp8_hadamard);
          } else {
            launch_cute_fwd_split_d_fp8_sm120<kDataType, kHeadDim, kStage>(
                Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
                dropout_p, philox_seed, philox_offset, fp8_smooth_k,
                fp8_smooth_v, fp8_q_quant_method, fp8_k_quant_method,
                fp8_v_quant_method, fp8_pv_acc_type, fp8_qk_mm_type,
                /*q_start_row=*/0, fp8_hadamard);
          }
        } else {
          if (fp8_hybrid && Nq >= fp8_hybrid_n_early) {
            const int n_early = static_cast<int>(fp8_hybrid_n_early);
            TORCH_CHECK(n_early % 64 == 0,
                        "ffpa_attn: fp8_hybrid_n_early must be multiple of 64");
            torch::Tensor Q_e, K_e, V_e;
            prepare_hybrid_stage1(Q_e, K_e, V_e, Q, K, V, n_early, Nkv, Nq,
                                  causal, D_og, kHeadDim, d_padded);
            auto O_e = torch::empty_like(Q_e);
            auto lse_e =
                torch::empty({Nb, Nh, n_early}, torch::TensorOptions()
                                                    .dtype(torch::kFloat32)
                                                    .device(Q.device()));
            auto empty_bias = torch::empty({0}, attn_bias.options());
            launch_cute_fwd_split_d_m4n2_sm120<kDataType, kHeadDim, kStage>(
                Q_e, K_e, V_e, O_e, empty_bias, lse_e, causal, softmax_scale,
                0.0, 0, 0);
            O.slice(2, 0, n_early).copy_(O_e);
            if (softmax_lse.numel() > 0)
              softmax_lse.slice(2, 0, n_early).copy_(lse_e);
            launch_cute_fwd_split_d_m4n2_fp8_sm120<kDataType, kHeadDim, kStage>(
                Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
                dropout_p, philox_seed, philox_offset, fp8_smooth_k,
                fp8_smooth_v, fp8_q_quant_method, fp8_k_quant_method,
                fp8_v_quant_method, fp8_pv_acc_type, fp8_qk_mm_type,
                /*q_start_row=*/n_early, fp8_hadamard);
          } else {
            launch_cute_fwd_split_d_m4n2_fp8_sm120<kDataType, kHeadDim, kStage>(
                Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
                dropout_p, philox_seed, philox_offset, fp8_smooth_k,
                fp8_smooth_v, fp8_q_quant_method, fp8_k_quant_method,
                fp8_v_quant_method, fp8_pv_acc_type, fp8_qk_mm_type,
                /*q_start_row=*/0, fp8_hadamard);
          }
        }
#else
        TORCH_CHECK(false, "ffpa_attn: cute ext not compiled");
#endif
      } else if (prop->major == 9 || prop->major == 10) {
        // sm_90/100 (228 KB smem): WS path, setmaxnreg effective.
        if (!has_attn_bias && !has_dropout && kHeadDim <= 512) {
          // w/ kPersistQg2s = 1
          launch_native_fwd_split_d_sm120<
              kDataType, kHeadDim, kMmaAccFloat32QK, kMmaAccFloat32PV, kStage,
              64 /*kQKDChunk*/, 64 /*kVDChunk*/, 0 /*kShareSmemQKV*/,
              1 /*kPersistQg2s*/, 8 /*kMmaTileSeqLenQ*/, 16 /*kValTileSeqLenK*/,
              128 /*kProducerThreads*/, 0 /*kNonWS*/>(
              Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
              dropout_p, philox_seed, philox_offset);
        } else {
          // w/ kPersistQg2s = 0
          launch_native_fwd_split_d_sm120<
              kDataType, kHeadDim, kMmaAccFloat32QK, kMmaAccFloat32PV, kStage,
              64 /*kQKDChunk*/, 64 /*kVDChunk*/, 0 /*kShareSmemQKV*/,
              0 /*kPersistQg2s*/, 8 /*kMmaTileSeqLenQ*/, 16 /*kValTileSeqLenK*/,
              128 /*kProducerThreads*/, 0 /*kNonWS*/>(
              Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
              dropout_p, philox_seed, philox_offset);
        }
      } else {
        // sm_120a (99 KB smem): non-WS path.
#ifdef ENABLE_FFPA_CUTE_EXT
        // CuTe kernel: kHeadDim%64==0 → kVDChunk=64; %32==0 → kVDChunk=32.
        // NOTE: CuTe kernel's bias/dropout paths are functional but ~2x slower
        // than the non-WS TMA template kernel due to register pressure from
        // the 128x128 rowcol tensor abstraction (64 score regs simultaneously
        // live + addressing temps → spills). Prefer the non-WS TMA fallback
        // when bias/dropout is active; CuTe handles the clean path only.
        if (force_tma) {
          launch_native_fwd_split_d_sm120<
              kDataType, kHeadDim, kMmaAccFloat32QK, kMmaAccFloat32PV,
              (kStage > 3 ? 3 : kStage), 32 /*kQKDChunk*/, 64 /*kVDChunk*/,
              0 /*kShareSmemQKV*/, 0 /*kPersistQg2s*/, 8 /*kMmaTileSeqLenQ*/,
              16 /*kValTileSeqLenK*/, 128 /*kProducerThreads*/, 1 /*kNonWS*/>(
              Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
              dropout_p, philox_seed, philox_offset);
        } else if (force_cute_tma || (!has_attn_bias && !has_dropout)) {
          if constexpr (kHeadDim <= 128 && kHeadDim % 32 == 0) {
            // WS persist-D: D=32/64/96/128 (Q persist fits the smem budget).
            // 32-mult small D (32/96) uses SW64 smem swizzle (D*2B=64/192B),
            // auto-selected by Traits; TMA descriptors match via SmemLayoutO.
            launch_cute_fwd_persist_d_sm120<kDataType, kHeadDim, kStage>(
                Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
                dropout_p, philox_seed, philox_offset);
          } else if constexpr (kHeadDim % 64 == 0) {
            // Production dispatch for %64==0 headdims, from the A/B benchmark
            // (RTX 5090, self-attn fp16/bf16, D=320..1024): M8N1 wins for
            // D<768 (+2..16%, cross at D=640), M4N2 wins for D>=768 (+7% @768,
            // +11% @896, +55% @1024 where M8N1's o_acc=D/2 regs spills to
            // local mem and collapses to ~100T). Both are exact (O_err ~1e-4)
            // at every D. Table in fwd_sm120.cuh M4N2 header.
            if constexpr (kHeadDim >= 768) {
              // split-D M4N2 (non-WS): kBr=64, atom_layout=(4,2,1). O regs =
              // D/4 per thread (vs M8N1's D/2 which spills for D>=512).
              launch_cute_fwd_split_d_m4n2_sm120<kDataType, kHeadDim, kStage>(
                  Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
                  dropout_p, philox_seed, philox_offset);
            } else {
              // split-D (non-WS) M8N1. The WS variant
              // (launch_cute_fwd_split_d_ws_sm120) is disabled:
              // setmaxnreg's consumer ceiling (232, CTA-pool max) cannot hold
              // D=512's 256-reg o_acc (per-thread hard cap 255), and D=256/320/
              // 512 show no perf gain over non-WS (o_acc=D*kBr/256 regs spills
              // to local mem either way). WS kernel kept in
              // cute/sm_120/split_d.cuh for reference; FA-1 M4N2 is the path to
              // lower large-D reg pressure (.tmp/plans/ffpa_fa1.md).
              launch_cute_fwd_split_d_sm120<kDataType, kHeadDim, kStage, 32,
                                            64>(
                  Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
                  dropout_p, philox_seed, philox_offset);
            }
          } else if constexpr (kHeadDim % 32 == 0) {
            launch_cute_fwd_split_d_sm120<kDataType, kHeadDim, kStage, 32, 32>(
                Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
                dropout_p, philox_seed, philox_offset);
          } else {
            launch_native_fwd_split_d_sm120<
                kDataType, kHeadDim, kMmaAccFloat32QK, kMmaAccFloat32PV,
                (kStage > 3 ? 3 : kStage), 32 /*kQKDChunk*/, 64 /*kVDChunk*/,
                0 /*kShareSmemQKV*/, 0 /*kPersistQg2s*/, 8 /*kMmaTileSeqLenQ*/,
                16 /*kValTileSeqLenK*/, 128 /*kProducerThreads*/, 1 /*kNonWS*/>(
                Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
                dropout_p, philox_seed, philox_offset);
          }
        } else {
          launch_native_fwd_split_d_sm120<
              kDataType, kHeadDim, kMmaAccFloat32QK, kMmaAccFloat32PV,
              (kStage > 3 ? 3 : kStage), 32 /*kQKDChunk*/, 64 /*kVDChunk*/,
              0 /*kShareSmemQKV*/, 0 /*kPersistQg2s*/, 8 /*kMmaTileSeqLenQ*/,
              16 /*kValTileSeqLenK*/, 128 /*kProducerThreads*/, 1 /*kNonWS*/>(
              Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
              dropout_p, philox_seed, philox_offset);
        }
#else
        // kQKDChunk=32 (SWIZZLE_64B), kVDChunk=64, S≤3.
        // smem per stage: Q=128×32×2B=8KB, K=128×32×2B=8KB, V=128×64×2B=16KB
        // total: 3×(8+8+16)KB = 96KB < 99KB.
        launch_native_fwd_split_d_sm120<
            kDataType, kHeadDim, kMmaAccFloat32QK, kMmaAccFloat32PV,
            (kStage > 3 ? 3 : kStage), 32 /*kQKDChunk*/, 64 /*kVDChunk*/,
            0 /*kShareSmemQKV*/, 0 /*kPersistQg2s*/, 8 /*kMmaTileSeqLenQ*/,
            16 /*kValTileSeqLenK*/, 128 /*kProducerThreads*/, 1 /*kNonWS*/>(
            Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
            dropout_p, philox_seed, philox_offset);
#endif
      }
      return;
    }
  }
#endif  // ENABLE_FFPA_TMA_EXT

#ifdef ENABLE_FFPA_CUTE_EXT
  // CuTe cp.async path: sm_80+ without TMA (tma=0 or sm<90).
  // Architecture-aware dispatch:
  //   sm >= 120 (Blackwell, high compute): stages capped at 2 for (32,32),
  //     3 for (32,64) — sync overhead dominates on fast MMA.
  //   sm < 120 (Ada/Ampere, lower compute): prefer (32,64) for D%64==0,
  //     deeper pipeline (Python-controlled, smem physics cap applies).
  if (!force_native) {
    auto cute_prop = at::cuda::getCurrentDeviceProperties();
    const int sm_arch = cute_prop->major * 10 + cute_prop->minor;
    if (sm_arch >= 120) {
      constexpr int kCuteStage32 = (kStage > 2) ? 2 : kStage;
      constexpr int kCuteStage64 = (kStage > 3) ? 3 : kStage;
      if constexpr (kHeadDim >= 320) {
        launch_cute_fwd_split_d_sm80<kDataType, kHeadDim, kCuteStage32, 32, 32>(
            Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
            dropout_p, philox_seed, philox_offset);
      } else if constexpr (kHeadDim % 64 == 0) {
        launch_cute_fwd_split_d_sm80<kDataType, kHeadDim, kCuteStage64, 32, 64>(
            Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
            dropout_p, philox_seed, philox_offset);
      } else if constexpr (kHeadDim % 32 == 0) {
        launch_cute_fwd_split_d_sm80<kDataType, kHeadDim, kCuteStage32, 32, 32>(
            Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
            dropout_p, philox_seed, philox_offset);
      }
    } else {
      if constexpr (kHeadDim % 64 == 0) {
        launch_cute_fwd_split_d_sm80<kDataType, kHeadDim, kStage, 32, 64>(
            Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
            dropout_p, philox_seed, philox_offset);
      } else if constexpr (kHeadDim % 32 == 0) {
        launch_cute_fwd_split_d_sm80<kDataType, kHeadDim, kStage, 32, 32>(
            Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
            dropout_p, philox_seed, philox_offset);
      }
    }
    return;
  }
#endif  // ENABLE_FFPA_CUTE_EXT

  // Native general cp.async path + Nq==1 split-KV decode fast-path (fallback
  // when no TMA/CuTe backend is selected). Config + decode live in native/.
  launch_native_fwd_split_d_sm80<kDataType, kHeadDim, kMmaAccFloat32QK,
                                 kMmaAccFloat32PV, kStage>(
      Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale, dropout_p,
      philox_seed, philox_offset);
}

#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/constant_pad_nd.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstring>
#include <optional>
#include "backend.h"
#include "layout.cuh"
#include "dispatch.cuh"
#include "native/launch.cuh"
// Green-checkpoint stage: the family entry definitions stay visible in this
// TU (implicit instantiation == pre-split behavior). The C' flip drops these
// includes; the generated per-family TUs then provide the instantiations.
#include "dispatch/native.cuh"
#include "dispatch/cute16.cuh"
#include "dispatch/fp8.cuh"
#include "dispatch/fp4.cuh"
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

  // fp16/bf16 head_dim pad: non-32-multiple D_og (e.g. 120) needs Q/K/V rows
  // widened to the compiled kHeadDim. Which mechanism applies depends on the
  // backend (FC-8):
  //   - AUTO/NATIVE (cp.async) and the sm90+ TMA native path consume D_og-wide
  //     rows directly: kernels zero-fill pad cols (cp.async src-size guard /
  //     TMA OOB zero fill), so no pad copy here.
  //   - TMA hint on pre-sm90 hardware (or without the TMA ext) falls back to
  //     the CUTE sm80 kernel, which still needs the torch pad copy — hence
  //     tma_kernel_active requires both the ext and major >= 9.
  //   - CUTE/CUTE_TMA fp16 paths keep the torch pad copy below.
  //   - fp8 skips (quantize reads D_og natively); O is padded by ffpa_api.cc.
  //   - fp4 skips when D_og%8==0 (the api gate): its quantize/delta_s kernels
  //     read the original width and zero-fill pad cols; FFPA_FP4_PAD_TORCH=1
  //     forces the torch pad path for A/B comparison and as a fallback.
  const int D_og = Q.size(3);
  const bool d_padded = D_og != kHeadDim;
#ifdef ENABLE_FFPA_TMA_EXT
  auto prop_pad = at::cuda::getCurrentDeviceProperties();
  const bool tma_kernel_active = force_tma && prop_pad->major >= 9;
#else
  const bool tma_kernel_active = false;
#endif
  const bool native_kernel_pad = force_native || tma_kernel_active;
  const bool fp4_fused =
      force_fp4 && D_og % 8 == 0 && getenv("FFPA_FP4_PAD_TORCH") == nullptr;
  const bool qkv_padded =
      d_padded && !force_fp8 && !fp4_fused && !native_kernel_pad;
  if (qkv_padded) {
    const int64_t pad_cols = kHeadDim - D_og;
    Q = torch::constant_pad_nd(Q, {0, pad_cols}, 0.0);
    K = torch::constant_pad_nd(K, {0, pad_cols}, 0.0);
    V = torch::constant_pad_nd(V, {0, pad_cols}, 0.0);
  }

  // Family routing: fill the shared FfpaFwdParams once (after any NHD/pad
  // rewrite above) and delegate each backend family to its ffpa:: entry
  // declared in dispatch.cuh. Routing-level fallbacks (bias/dropout and
  // D%32 != 0 to native) stay here; family-internal headdim gates live in
  // the entries (dispatch/*.cuh).
  ffpa::FfpaFwdParams p;
  p.Q = Q;
  p.K = K;
  p.V = V;
  p.O = O;
  p.attn_bias = attn_bias;
  p.softmax_lse = softmax_lse;
  p.causal = causal;
  p.softmax_scale = softmax_scale;
  p.dropout_p = dropout_p;
  p.philox_seed = philox_seed;
  p.philox_offset = philox_offset;
  p.fp8_smooth_k = fp8_smooth_k;
  p.fp8_smooth_v = fp8_smooth_v;
  p.fp8_q_quant_method = fp8_q_quant_method;
  p.fp8_k_quant_method = fp8_k_quant_method;
  p.fp8_v_quant_method = fp8_v_quant_method;
  p.fp8_pv_acc_type = fp8_pv_acc_type;
  p.fp8_qk_mm_type = fp8_qk_mm_type;
  p.fp8_hybrid = fp8_hybrid;
  p.fp8_hybrid_n_early = fp8_hybrid_n_early;
  p.fp4_hybrid = fp4_hybrid;
  p.fp4_hybrid_n_early = fp4_hybrid_n_early;
  p.fp8_hadamard = fp8_hadamard;
  p.fp4_hadamard = fp4_hadamard;
  p.fp4_pv_mm_type = fp4_pv_mm_type;
  p.fp4_smooth_v = fp4_smooth_v;
  p.Nb = Nb;
  p.Nh = Nh;
  p.Nh_kv = Nh_kv;
  p.Nq = Nq;
  p.Nkv = Nkv;
  p.D_og = D_og;
  p.d_padded = d_padded;
  p.qkv_padded = qkv_padded;
  p.has_attn_bias = has_attn_bias;
#ifdef ENABLE_FFPA_TMA_EXT
  if ((force_tma || force_cute_tma || force_fp8 || force_fp4) &&
      !force_native && !force_cute) {
    auto prop = at::cuda::getCurrentDeviceProperties();
    if (prop->major >= 9) {
      if (force_fp4) {
        ffpa::ffpa_fwd_fp4<kDataType, kHeadDim, kStage>(p);
        return;
      }
      if (force_fp8) {
        ffpa::ffpa_fwd_fp8<kDataType, kHeadDim, kStage>(p);
      } else if (prop->major == 9 || prop->major == 10) {
        // sm_90/100 (228 KB smem): WS path, setmaxnreg effective.
        ffpa::ffpa_fwd_native_tma<kDataType, kHeadDim, kMmaAccFloat32QK,
                                  kMmaAccFloat32PV, kStage>(p);
      } else {
        // sm_120a (99 KB smem): non-WS path.
#ifdef ENABLE_FFPA_CUTE_EXT
        if (force_tma) {
          ffpa::ffpa_fwd_native_tma<kDataType, kHeadDim, kMmaAccFloat32QK,
                                    kMmaAccFloat32PV, kStage>(p);
        } else if (force_cute_tma || (!has_attn_bias && !has_dropout)) {
          if constexpr (kHeadDim % 32 == 0) {
            ffpa::ffpa_fwd_cute16<kDataType, kHeadDim, kStage>(p);
          } else {
            // D%32!=0: native non-WS fallback, as before the family split.
            ffpa::ffpa_fwd_native_tma<kDataType, kHeadDim, kMmaAccFloat32QK,
                                      kMmaAccFloat32PV, kStage>(p);
          }
        } else {
          ffpa::ffpa_fwd_native_tma<kDataType, kHeadDim, kMmaAccFloat32QK,
                                    kMmaAccFloat32PV, kStage>(p);
        }
#else
        ffpa::ffpa_fwd_native_tma<kDataType, kHeadDim, kMmaAccFloat32QK,
                                  kMmaAccFloat32PV, kStage>(p);
#endif
      }
      return;
    }
  }
#endif  // ENABLE_FFPA_TMA_EXT

#ifdef ENABLE_FFPA_CUTE_EXT
  // CuTe cp.async path: sm_80+ without TMA (tma=0 or sm<90).
  // Architecture-aware stage clamps live inside the entry.
  if (!force_native) {
    ffpa::ffpa_fwd_cute16_sm80<kDataType, kHeadDim, kStage>(p);
    return;
  }
#endif  // ENABLE_FFPA_CUTE_EXT

  // Native general cp.async path + Nq==1 split-KV decode fast-path (fallback
  // when no TMA/CuTe backend is selected). Config + decode live in native/.
  ffpa::ffpa_fwd_native_sm80<kDataType, kHeadDim, kMmaAccFloat32QK,
                             kMmaAccFloat32PV, kStage>(p);
}

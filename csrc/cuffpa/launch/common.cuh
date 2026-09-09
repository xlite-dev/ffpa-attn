// Shared cute-family launcher helpers (strided-NHD probes, bias tile
// plan), moved verbatim out of the old cute/launch.cuh. Gated on
// ENABLE_FFPA_CUTE_EXT only: the sm80 fp16 launcher uses the bias tile
// plan without TMA (the old file kept helpers inside the TMA gate,
// which broke cute-without-tma builds).
#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <optional>
#include <type_traits>
#include "../common.cuh"
#include "layout.cuh"
// Unqualified ffpa names below (utils::div_ceil, ffpa_fp8::...) relied on
// native/launch.cuh's global `using namespace ffpa;` arriving first in the
// old launch.cuh include chain; family TUs include this header directly.
#include "native/utils.cuh"
// Fp8InputLayout for ffpa_layout_of (was transitive via quantize_fp8.cuh).
#include "cute/fp8/input_layout.cuh"
using namespace ffpa;

#ifdef ENABLE_FFPA_CUTE_EXT
// TMA consumers need 16B alignment on the base pointer and both row/batch
// strides of a strided-NHD input.
static inline void ffpa_check_strided_nhd_aligned(const torch::Tensor& X,
                                                  const char* name) {
  const long es = X.element_size();
  TORCH_CHECK(reinterpret_cast<uintptr_t>(X.data_ptr()) % 16 == 0 &&
                  (X.stride(2) * es) % 16 == 0 && (X.stride(0) * es) % 16 == 0,
              "ffpa_attn: strided NHD ", name,
              " requires a 16B-aligned data_ptr and 16B-aligned row/batch "
              "strides (elemsize ",
              es, ", strides ", X.stride(0), ",", X.stride(1), ",", X.stride(2),
              ",1)");
}

// Fp8InputLayout from a [B, H, N, D] tensor's strides. Accepts BHND-packed
// and NHD-view; anything else (arbitrary strides) is rejected unless
// allow_strided_rows: NHD-family views whose row stride exceeds H*D
// (fused-QKV chunk layouts, e.g. FLUX.2 single-stream V) are also accepted
// — the pre-kernels address rows through s_row, so any 16B-aligned
// positive row/batch stride is legal. Every fp8/fp4 family (persist-D,
// split-D, m4n2) passes allow_strided_rows=true.
static inline ffpa_fp8::Fp8InputLayout ffpa_layout_of(
    const torch::Tensor& X, long N, long D, bool allow_strided_rows = false) {
  const long B = X.size(0), H = X.size(1);
  TORCH_CHECK(X.dim() == 4 && X.stride(3) == 1,
              "ffpa_attn: Q/K/V must be 4-D with unit stride along D");
  if (ffpa_is_nhd_view(X))
    return {true, static_cast<int>(H), N * H * D, D, H * D};
  if (allow_strided_rows && ffpa_is_strided_nhd(X)) {
    ffpa_check_strided_nhd_aligned(X, "input");
    const long s_row = X.stride(2);
    const long s_batch = (B <= 1) ? N * s_row : X.stride(0);
    return {true, static_cast<int>(H), s_batch, D, s_row};
  }
  TORCH_CHECK(
      X.stride(2) == D && X.stride(1) == N * D && X.stride(0) == H * N * D,
      "ffpa_attn: Q/K/V must be BHND-contiguous or an NHD (BNHD) "
      "permute view, got strides (",
      X.stride(0), ",", X.stride(1), ",", X.stride(2), ",", X.stride(3), ")");
  return {false, 0, 0, N * D, D};
}

// Kernel-side attn_bias view: nullptr means no bias. dtype codes match
// ffpa::prefill::load_attn_bias_value (1=half, 2=bf16, 3=other).
struct FfpaBiasParams {
  const void* ptr = nullptr;
  int dtype = 0;
  long long stride_b = 0;
  long long stride_h = 0;
  long long stride_m = 0;
  long long stride_n = 0;
};

// Anchor object for bias-less dummy TMA descriptors (never dereferenced:
// the kernel skips the bias TMA when tile mode is 0).
static constexpr uint16_t kBiasDummyAnchor __attribute__((aligned(16))) = 0;

// Validate/flatten a python-side additive attn_bias (already normalized to
// 4-D [B, Nh_q, Nq, Nkv]; size-1 dims broadcast via stride 0) into kernel
// launch parameters. Shared by the fp8/fp4 quant launchers; mirrors the
// fp16 launcher checks.
static inline FfpaBiasParams ffpa_bias_params_of(const torch::Tensor& attn_bias,
                                                 const torch::Tensor& Q,
                                                 const torch::Tensor& K) {
  FfpaBiasParams p;
  if (attn_bias.numel() == 0)
    return p;
  const long Nb = Q.size(0), Nh = Q.size(1), Nq = Q.size(2);
  const long Nkv = K.size(2);
  TORCH_CHECK(attn_bias.is_cuda(),
              "ffpa_attn: attn_mask must be a CUDA tensor");
  TORCH_CHECK(attn_bias.device() == Q.device(),
              "ffpa_attn: attn_mask must be on the same device as Q/K/V");
  TORCH_CHECK(attn_bias.dim() == 4,
              "ffpa_attn: normalized attn_mask must be 4-D [B, Nh_q, Nq, Nkv]");
  TORCH_CHECK(attn_bias.size(0) == 1 || attn_bias.size(0) == Nb,
              "ffpa_attn: attn_mask batch dimension must be 1 or B");
  TORCH_CHECK(attn_bias.size(1) == 1 || attn_bias.size(1) == Nh,
              "ffpa_attn: attn_mask head dimension must be 1 or Nh_q");
  TORCH_CHECK(attn_bias.size(2) == 1 || attn_bias.size(2) == Nq,
              "ffpa_attn: attn_mask query dimension must be 1 or Nq");
  TORCH_CHECK(attn_bias.size(3) == 1 || attn_bias.size(3) == Nkv,
              "ffpa_attn: attn_mask kv dimension must be 1 or Nkv");
  p.ptr = attn_bias.data_ptr();
  p.dtype = attn_bias.scalar_type() == at::ScalarType::Half       ? 1
            : attn_bias.scalar_type() == at::ScalarType::BFloat16 ? 2
                                                                  : 3;
  p.stride_b = (attn_bias.size(0) == 1 && Nb > 1) ? 0 : attn_bias.stride(0);
  p.stride_h = (attn_bias.size(1) == 1 && Nh > 1) ? 0 : attn_bias.stride(1);
  p.stride_m = (attn_bias.size(2) == 1 && Nq > 1) ? 0 : attn_bias.stride(2);
  p.stride_n = (attn_bias.size(3) == 1 && Nkv > 1) ? 0 : attn_bias.stride(3);
  return p;
}

// PC-0 bias tile plan: classify the broadcast shape and check TMA
// expressibility (contiguous inner dim, 16B-aligned rows) for the smem
// tile prefetch. mode: 0 = gmem-direct fallback (FC-4 path), 1 = dense
// [kBr,kBc] tile, 2 = row-broadcast [1,kBc] tile. For mode 1 the (b,h)
// strides fold into the linear row domain of the (m_total, Nkv) plane
// (validated here), so the kernel offsets rows with its existing
// attn_bias_stride_b/h. Column-broadcast masks (stride_n==0) stay on the
// fallback: a 1-wide TMA box violates the 16B inner-dim requirement.
struct FfpaBiasTilePlan {
  int mode = 0;
  int elem_size = 4;
  long long m_total = 0;
  // stages: 1 for the sm_120 single-buffer TMA tile, 2 for the sm_80
  // double-buffered cp.async loader.
  long long tile_bytes(int kBr, int kBc, int stages = 1) const {
    const long long rows = (mode == 1) ? kBr : 1;
    return (long long)stages * rows * kBc * elem_size;
  }
};

static inline FfpaBiasTilePlan ffpa_bias_tile_plan_of(
    const FfpaBiasParams& bias, long Nb, long Nh, long Nq, long Nkv) {
  FfpaBiasTilePlan plan;
  if (bias.ptr == nullptr)
    return plan;
#if defined(ENABLE_FFPA_FP16_BUILD_DEBUG) || \
    defined(ENABLE_FFPA_FP8_BUILD_DEBUG) ||  \
    defined(ENABLE_FFPA_FP4_BUILD_DEBUG)
  // Debug/A-B switch: force every tile mode to the gmem-direct fallback.
  if (getenv("FFPA_BIAS_TILE_DISABLE") != nullptr)
    return plan;
#endif
  plan.elem_size = bias.dtype == 3 ? 4 : 2;
  const auto aligned16 = [](long long bytes) { return bytes % 16 == 0; };
  if (!aligned16(reinterpret_cast<long long>(bias.ptr)))
    return plan;
  if (bias.stride_n == 1 && bias.stride_m == 0) {
    // Row-broadcast plane is [b_eff*h_eff, Nkv]: (b,h) must fold to an
    // exact row (stride_h in {0, Nkv}, stride_b in {0, h_eff*Nkv}) and
    // the row stride must keep the 16B outer-stride guarantee (TMA
    // descriptor / cp.async vector), else stay on the gmem fallback.
    const long long h_eff = bias.stride_h != 0 ? Nh : 1;
    const long long b_eff = bias.stride_b != 0 ? Nb : 1;
    if (bias.stride_h != 0 && bias.stride_h != Nkv)
      return plan;
    if (bias.stride_b != 0 && bias.stride_b != h_eff * Nkv)
      return plan;
    if (!aligned16(Nkv * plan.elem_size))
      return plan;
    plan.mode = 2;
    plan.m_total = b_eff * h_eff;
    return plan;
  }
  if (bias.stride_n != 1 || !aligned16(bias.stride_m * plan.elem_size))
    return plan;
  const long long h_eff = bias.stride_h != 0 ? Nh : 1;
  const long long b_eff = bias.stride_b != 0 ? Nb : 1;
  if (bias.stride_h != 0 && bias.stride_h != Nq * bias.stride_m)
    return plan;
  if (bias.stride_b != 0 && bias.stride_b != h_eff * Nq * bias.stride_m)
    return plan;
  plan.mode = 1;
  plan.m_total = b_eff * h_eff * Nq;
  return plan;
}

#endif  // ENABLE_FFPA_CUTE_EXT

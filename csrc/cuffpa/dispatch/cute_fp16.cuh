// CuTe fp16/bf16 family entry definitions for the family-split forward
// build: the sm120 TMA headdim gates, the sm80 cp.async dispatch, and the
// fp8/fp4 hybrid stage-1 fp16 launcher. Branch bodies moved verbatim from
// launch.cuh::launch_ffpa_attn_fwd_template.
#pragma once
#include "dispatch.cuh"
#ifdef ENABLE_FFPA_CUTE_EXT
#include "launch/cute_fp16.cuh"
#endif

namespace ffpa {

// CuTe cp.async path: sm_80+ without TMA (tma=0 or sm<90).
// Architecture-aware dispatch:
//   sm >= 120 (Blackwell, high compute): stages capped at 2 for (32,32),
//     3 for (32,64) — sync overhead dominates on fast MMA.
//   sm < 120 (Ada/Ampere, lower compute): prefer (32,64) for D%64==0,
//     deeper pipeline (Python-controlled, smem physics cap applies).
template <typename kDataType, const int kHeadDim, const int kStage>
void ffpa_fwd_cute_fp16_sm80(const FfpaFwdParams& p) {
#ifdef ENABLE_FFPA_CUTE_EXT
  auto cute_prop = at::cuda::getCurrentDeviceProperties();
  const int sm_arch = cute_prop->major * 10 + cute_prop->minor;
  if (sm_arch >= 120) {
    constexpr int kCuteStage32 = (kStage > 2) ? 2 : kStage;
    constexpr int kCuteStage64 = (kStage > 3) ? 3 : kStage;
    if constexpr (kHeadDim >= 320) {
      launch_cute_fwd_split_d_sm80<kDataType, kHeadDim, kCuteStage32, 32, 32>(
          p.Q, p.K, p.V, p.O, p.attn_bias, p.softmax_lse, p.causal,
          p.softmax_scale, p.dropout_p, p.philox_seed, p.philox_offset);
    } else if constexpr (kHeadDim % 64 == 0) {
      launch_cute_fwd_split_d_sm80<kDataType, kHeadDim, kCuteStage64, 32, 64>(
          p.Q, p.K, p.V, p.O, p.attn_bias, p.softmax_lse, p.causal,
          p.softmax_scale, p.dropout_p, p.philox_seed, p.philox_offset);
    } else if constexpr (kHeadDim % 32 == 0) {
      launch_cute_fwd_split_d_sm80<kDataType, kHeadDim, kCuteStage32, 32, 32>(
          p.Q, p.K, p.V, p.O, p.attn_bias, p.softmax_lse, p.causal,
          p.softmax_scale, p.dropout_p, p.philox_seed, p.philox_offset);
    }
  } else {
    if constexpr (kHeadDim % 64 == 0) {
      launch_cute_fwd_split_d_sm80<kDataType, kHeadDim, kStage, 32, 64>(
          p.Q, p.K, p.V, p.O, p.attn_bias, p.softmax_lse, p.causal,
          p.softmax_scale, p.dropout_p, p.philox_seed, p.philox_offset);
    } else if constexpr (kHeadDim % 32 == 0) {
      launch_cute_fwd_split_d_sm80<kDataType, kHeadDim, kStage, 32, 32>(
          p.Q, p.K, p.V, p.O, p.attn_bias, p.softmax_lse, p.causal,
          p.softmax_scale, p.dropout_p, p.philox_seed, p.philox_offset);
    }
  }
#else
  TORCH_CHECK(false, "ffpa_attn: cute sm80 path not compiled");
#endif
}

// CuTe kernel: kHeadDim%64==0 → kVDChunk=64; %32==0 → kVDChunk=32.
// NOTE: CuTe kernel's bias/dropout paths are functional but ~2x slower
// than the non-WS TMA template kernel due to register pressure from
// the 128x128 rowcol tensor abstraction (64 score regs simultaneously
// live + addressing temps → spills). The routing layer therefore sends
// only the clean path here (bias/dropout falls back to native); the
// kernels still accept bias/dropout for the hybrid stage-1 slices.
template <typename kDataType, const int kHeadDim, const int kStage>
void ffpa_fwd_cute_fp16(const FfpaFwdParams& p) {
#if defined(ENABLE_FFPA_CUTE_EXT) && defined(ENABLE_FFPA_TMA_EXT)
  if constexpr (kHeadDim <= 128 && kHeadDim % 32 == 0) {
    // WS persist-D: D=32/64/96/128 (Q persist fits the smem budget).
    // 32-mult small D (32/96) uses SW64 smem swizzle (D*2B=64/192B),
    // auto-selected by Traits; TMA descriptors match via SmemLayoutO.
    launch_cute_fwd_persist_d_sm120<kDataType, kHeadDim, kStage>(
        p.Q, p.K, p.V, p.O, p.attn_bias, p.softmax_lse, p.causal,
        p.softmax_scale, p.dropout_p, p.philox_seed, p.philox_offset);
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
          p.Q, p.K, p.V, p.O, p.attn_bias, p.softmax_lse, p.causal,
          p.softmax_scale, p.dropout_p, p.philox_seed, p.philox_offset);
    } else {
      // split-D (non-WS) M8N1. The WS variant
      // (launch_cute_fwd_split_d_ws_sm120) is disabled:
      // setmaxnreg's consumer ceiling (232, CTA-pool max) cannot hold
      // D=512's 256-reg o_acc (per-thread hard cap 255), and D=256/320/
      // 512 show no perf gain over non-WS (o_acc=D*kBr/256 regs spills
      // to local mem either way). WS kernel kept in
      // cute/sm_120/split_d.cuh for reference; FA-1 M4N2 is the path to
      // lower large-D reg pressure (.tmp/plans/ffpa_fa1.md).
      launch_cute_fwd_split_d_sm120<kDataType, kHeadDim, kStage, 32, 64>(
          p.Q, p.K, p.V, p.O, p.attn_bias, p.softmax_lse, p.causal,
          p.softmax_scale, p.dropout_p, p.philox_seed, p.philox_offset);
    }
  } else if constexpr (kHeadDim % 32 == 0) {
    launch_cute_fwd_split_d_sm120<kDataType, kHeadDim, kStage, 32, 32>(
        p.Q, p.K, p.V, p.O, p.attn_bias, p.softmax_lse, p.causal,
        p.softmax_scale, p.dropout_p, p.philox_seed, p.philox_offset);
  } else {
    TORCH_CHECK(false,
                "ffpa_attn: unreachable, routing guards D%32 for the cute "
                "fp16 family");
  }
#else
  TORCH_CHECK(false, "ffpa_attn: cute sm120 fp16 family not compiled");
#endif
}

// Hybrid stage-1 fp16 kernel selection for the fp8/fp4 paths: the caller
// (ffpa_fwd_fp8 / ffpa_fwd_fp4) prepares the early-row sub-problem tensors
// in FfpaFwdParams (dropout/philox stay 0) and this entry only dispatches
// on the headdim gates: persist-D for kHeadDim <= kPersistMaxD (fp8: 224,
// fp4: 256), split-D (32,64) below 768, M4N2 above.
template <typename kDataType, const int kHeadDim, const int kStage,
          const int kPersistMaxD>
void ffpa_fwd_fp16_stage1(const FfpaFwdParams& p) {
#if defined(ENABLE_FFPA_CUTE_EXT) && defined(ENABLE_FFPA_TMA_EXT)
  if constexpr (kHeadDim <= kPersistMaxD) {
    launch_cute_fwd_persist_d_sm120<kDataType, kHeadDim, kStage>(
        p.Q, p.K, p.V, p.O, p.attn_bias, p.softmax_lse, p.causal,
        p.softmax_scale, p.dropout_p, p.philox_seed, p.philox_offset);
  } else if constexpr (kHeadDim < 768) {
    launch_cute_fwd_split_d_sm120<kDataType, kHeadDim, kStage, 32, 64>(
        p.Q, p.K, p.V, p.O, p.attn_bias, p.softmax_lse, p.causal,
        p.softmax_scale, p.dropout_p, p.philox_seed, p.philox_offset);
  } else {
    launch_cute_fwd_split_d_m4n2_sm120<kDataType, kHeadDim, kStage>(
        p.Q, p.K, p.V, p.O, p.attn_bias, p.softmax_lse, p.causal,
        p.softmax_scale, p.dropout_p, p.philox_seed, p.philox_offset);
  }
#else
  TORCH_CHECK(false, "ffpa_attn: cute sm120 fp16 family not compiled");
#endif
}

}  // namespace ffpa

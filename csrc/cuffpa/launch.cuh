#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <optional>
#include "backend_hint.h"
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
template <typename kDataType, const int kHeadDim, const int kMmaAccFloat32QK,
          const int kMmaAccFloat32PV, const int kStage>
void launch_ffpa_attn_fwd_template(torch::Tensor Q, torch::Tensor K,
                                   torch::Tensor V, torch::Tensor O,
                                   torch::Tensor attn_bias,
                                   torch::Tensor softmax_lse, int causal,
                                   double softmax_scale, double dropout_p,
                                   int64_t philox_seed, int64_t philox_offset,
                                   bool smooth_k, std::optional<bool> qk_int8) {
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
  const bool force_w8a8 = (impl_hint == ffpa::CudaBackendImpl::CUTE_TMA_W8A8);

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
  if ((force_tma || force_cute_tma || force_w8a8) && !force_native &&
      !force_cute) {
    auto prop = at::cuda::getCurrentDeviceProperties();
    if (prop->major >= 9) {
      if (force_w8a8) {
#ifdef ENABLE_FFPA_CUTE_EXT
        // D<=128: persist-D w8a8; 128<D<=512: split-D M8N1 w8a8;
        // D>=768: split-D M4N2 w8a8 (D/4 regs, avoids M8N1's D/2 spill);
        // 512<D<768: fall back to fp16 M4N2 (M8N1 w8a8 spills, M4N2 w8a8
        // not yet built for this range).
        if constexpr (kHeadDim <= 128) {
          launch_cute_fwd_persist_d_w8a8_sm120<kDataType, kHeadDim, kStage>(
              Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
              dropout_p, philox_seed, philox_offset, smooth_k, qk_int8);
        } else if constexpr (kHeadDim <= 512) {
          launch_cute_fwd_split_d_w8a8_sm120<kDataType, kHeadDim, kStage>(
              Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
              dropout_p, philox_seed, philox_offset, smooth_k, qk_int8);
        } else if constexpr (kHeadDim >= 768 && kHeadDim <= 1024) {
          launch_cute_fwd_split_d_m4n2_w8a8_sm120<kDataType, kHeadDim, kStage>(
              Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
              dropout_p, philox_seed, philox_offset, smooth_k, qk_int8);
        } else {
          // 512 < D < 768: fall back to fp16 M4N2 (w8a8 not available).
          launch_cute_fwd_split_d_m4n2_sm120<kDataType, kHeadDim, kStage>(
              Q, K, V, O, attn_bias, softmax_lse, causal, softmax_scale,
              dropout_p, philox_seed, philox_offset);
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
          if constexpr (kHeadDim <= 128 && kHeadDim % 64 == 0) {
            // WS persist-D: D=64/128 (Q persist fits the smem budget).
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

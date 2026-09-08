// Native (cp.async / TMA) family entry definitions for the family-split
// forward build. Branch bodies moved verbatim from
// launch.cuh::launch_ffpa_attn_fwd_template; the routing layer fills
// FfpaFwdParams once and the generated per-family TUs instantiate these.
#pragma once
#include <ATen/cuda/CUDAContext.h>
#include "dispatch.cuh"
#include "launch/native_fp16.cuh"

namespace ffpa {

// Native general cp.async path + Nq==1 split-KV decode fast-path (fallback
// when no TMA/CuTe backend is selected). Config + decode live in native/.
template <typename kDataType, const int kHeadDim, const int kMmaAccFloat32QK,
          const int kMmaAccFloat32PV, const int kStage>
void ffpa_fwd_native_sm80(const FfpaFwdParams& p) {
  launch_native_fwd_split_d_sm80<kDataType, kHeadDim, kMmaAccFloat32QK,
                                 kMmaAccFloat32PV, kStage>(
      p.Q, p.K, p.V, p.O, p.attn_bias, p.softmax_lse, p.causal, p.softmax_scale,
      p.dropout_p, p.philox_seed, p.philox_offset);
}

// TMA path dispatch (moved from the old top-level launch.cuh): sm_90/100 run
// the WS kernel (setmaxnreg effective, 228KB smem allows a deep pipeline);
// sm_120a runs non-WS (all 256 threads do MMA, thread 0 issues TMA inline,
// which removes the WS register-allocation penalty; NOTE: no WGMMA on sm_120a).
// See fwd_sm120.cuh for the register pressure analysis.
template <typename kDataType, const int kHeadDim, const int kMmaAccFloat32QK,
          const int kMmaAccFloat32PV, const int kStage>
void ffpa_fwd_native_tma(const FfpaFwdParams& p) {
#ifdef ENABLE_FFPA_TMA_EXT
  auto prop = at::cuda::getCurrentDeviceProperties();
  if (prop->major == 9 || prop->major == 10) {
    // sm_90/100 (228 KB smem): WS path, setmaxnreg effective.
    if (!p.has_attn_bias && !(p.dropout_p > 0.0) && kHeadDim <= 512) {
      // w/ kPersistQg2s = 1
      launch_native_fwd_split_d_sm120<
          kDataType, kHeadDim, kMmaAccFloat32QK, kMmaAccFloat32PV, kStage,
          64 /*kQKDChunk*/, 64 /*kVDChunk*/, 0 /*kShareSmemQKV*/,
          1 /*kPersistQg2s*/, 8 /*kMmaTileSeqLenQ*/, 16 /*kValTileSeqLenK*/,
          128 /*kProducerThreads*/, 0 /*kNonWS*/>(
          p.Q, p.K, p.V, p.O, p.attn_bias, p.softmax_lse, p.causal,
          p.softmax_scale, p.dropout_p, p.philox_seed, p.philox_offset);
    } else {
      // w/ kPersistQg2s = 0
      launch_native_fwd_split_d_sm120<
          kDataType, kHeadDim, kMmaAccFloat32QK, kMmaAccFloat32PV, kStage,
          64 /*kQKDChunk*/, 64 /*kVDChunk*/, 0 /*kShareSmemQKV*/,
          0 /*kPersistQg2s*/, 8 /*kMmaTileSeqLenQ*/, 16 /*kValTileSeqLenK*/,
          128 /*kProducerThreads*/, 0 /*kNonWS*/>(
          p.Q, p.K, p.V, p.O, p.attn_bias, p.softmax_lse, p.causal,
          p.softmax_scale, p.dropout_p, p.philox_seed, p.philox_offset);
    }
  } else {
    // sm_120a (99 KB smem): non-WS path.
    // kQKDChunk=32 (SWIZZLE_64B), kVDChunk=64, S≤3.
    // smem per stage: Q=128×32×2B=8KB, K=128×32×2B=8KB, V=128×64×2B=16KB
    // total: 3×(8+8+16)KB = 96KB < 99KB.
    launch_native_fwd_split_d_sm120<
        kDataType, kHeadDim, kMmaAccFloat32QK, kMmaAccFloat32PV,
        (kStage > 3 ? 3 : kStage), 32 /*kQKDChunk*/, 64 /*kVDChunk*/,
        0 /*kShareSmemQKV*/, 0 /*kPersistQg2s*/, 8 /*kMmaTileSeqLenQ*/,
        16 /*kValTileSeqLenK*/, 128 /*kProducerThreads*/, 1 /*kNonWS*/>(
        p.Q, p.K, p.V, p.O, p.attn_bias, p.softmax_lse, p.causal,
        p.softmax_scale, p.dropout_p, p.philox_seed, p.philox_offset);
  }
#else
  TORCH_CHECK(false, "ffpa_attn: native TMA path not compiled");
#endif
}

}  // namespace ffpa

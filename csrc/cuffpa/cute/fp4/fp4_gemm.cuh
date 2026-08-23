// NVFP4 blockscaled-GEMM helpers shared across fp4 attention kernels
// (sm_120 persist_d today, a future split-D family later), the fp4
// counterpart of cute/gemm.cuh's gemm_ss/gemm_rs primitives. Depends only
// on fp4_pscale.cuh (P quantization) - no sm_120-only headers here.
#pragma once

#include <cute/tensor.hpp>
#include <cute/tensor_zip.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/arch/barrier.h>

#include "fp4_pscale.cuh"

namespace ffpa_fp4 {

using namespace cute;

// K/V^T storage column j -> original token index (the quantize kernels'
// 32-row interleave; bijection inside every 32-window, identity across).
// Table for j in [0,32): [0,1,8,9,16,17,24,25,2,3,10,11,18,19,26,27,
// 4,5,12,13,20,21,28,29,6,7,14,15,22,23,30,31].
CUTE_DEVICE int kv_perm32(int j) {
  const int loc = j & 31;
  return (j & ~31) + (loc / 8) * 2 + ((loc % 8) / 2) * 8 + (loc % 8) % 2;
}

// QK step of the blockscaled pipeline: S += (Qhat . SFQ) @ (Khat . SFK)^T
// over one kv_tile's smem stage. Named gemm_ss_fp4 after the fp8/fp16
// convention (QK = gemm_ss family, PV = gemm_rs family; the _fp4 suffix
// keeps it apart from ffpa_cute::gemm_ss), with two fp4 deviations from
// ffpa_cute::gemm_ss: (1) A (Q) is a per-work register constant preloaded
// outside the kv loop, so only B (K/SFK) streams smem->regs (gemm-before-
// copy prefetch order, same as the original inline loop); (2) operands are
// data+SF zip pairs and the k_empty arrive replaces the last copy so the
// producer sees the stage freed exactly when the mma chain has consumed
// it. Copy destinations are built in-function via retile_D (same as
// gemm_ss); callers pass raw partition_fragment results.
// NOTE for a future split-D variant: if A must also stream from smem,
// extend this into a gemm_ss-style dual-operand pipeline instead of
// preloading (Q-per-work is a persist_d specialization).
template <typename TensorC, typename TensorQA, typename TensorQSF,
          typename TensorKB, typename TensorKSF, typename TensorSK,
          typename TensorSKSF, typename TiledMma, typename TiledCopyK,
          typename ThreadCopyK, typename TiledCopyKSF, typename ThreadCopyKSF>
CUTE_DEVICE void gemm_ss_fp4(TensorC& acc, TensorQA& tSrQ, TensorQSF& tSrSFQ,
                             TensorKB& tSrK, TensorKSF& tSrSFK,
                             TensorSK const& tSsK, TensorSKSF const& tSsSFK,
                             TiledMma tiled_mma_qk, TiledCopyK tiled_copy_k,
                             ThreadCopyK thread_copy_k,
                             TiledCopyKSF tiled_copy_ksf,
                             ThreadCopyKSF thread_copy_ksf, uint64_t* k_empty,
                             int stage) {
  auto copy_view_k = thread_copy_k.retile_D(tSrK);
  auto copy_view_ksf = thread_copy_ksf.retile_D(tSrSFK);
  auto tSsK_stage = tSsK(_, _, _, stage);
  auto tSsSFK_stage = tSsSFK(_, _, _, stage);
  copy(tiled_copy_k, tSsK_stage(_, _, _0{}), copy_view_k(_, _, _0{}));
  copy(tiled_copy_ksf, tSsSFK_stage(_, _, _0{}), copy_view_ksf(_, _, _0{}));
  CUTLASS_PRAGMA_UNROLL
  for (int k_block = 0; k_block < size<2>(tSrQ); ++k_block) {
    cute::gemm(tiled_mma_qk,
               make_zip_tensor(tSrQ(_, _, k_block), tSrSFQ(_, _, k_block)),
               make_zip_tensor(tSrK(_, _, k_block), tSrSFK(_, _, k_block)),
               acc);
    if (k_block < size<2>(tSrQ) - 1) {
      copy(tiled_copy_k, tSsK_stage(_, _, k_block + 1),
           copy_view_k(_, _, k_block + 1));
      copy(tiled_copy_ksf, tSsSFK_stage(_, _, k_block + 1),
           copy_view_ksf(_, _, k_block + 1));
    } else {
      cutlass::arch::ClusterBarrier::arrive(k_empty + stage);
    }
  }
}

// Split-D QK step: one 64-wide D chunk per call, accumulating into the
// shared [kBr, kBc] S tile (delta_s preloads the accumulator first). The
// chunk TiledMmaQK (Tile-K=64) yields rank-2 per-chunk fragments, so there
// is no k_block loop: A (Q chunk) is copied from the work-resident Q smem
// by the caller - no stage barrier on the A side - and B (K/SFK chunk)
// streams from the chunk stage exactly like gemm_ss_fp4. The trailing
// k_empty arrive releases the stage when the single mma has consumed it.
template <typename TensorC, typename TensorQA, typename TensorQSF,
          typename TensorKB, typename TensorKSF, typename TensorSK,
          typename TensorSKSF, typename TiledMma, typename TiledCopyK,
          typename ThreadCopyK, typename TiledCopyKSF, typename ThreadCopyKSF>
CUTE_DEVICE void gemm_ss_chunk_fp4(
    TensorC& acc, TensorQA& tSrQ, TensorQSF& tSrSFQ, TensorKB& tSrK,
    TensorKSF& tSrSFK, TensorSK const& tSsK, TensorSKSF const& tSsSFK,
    TiledMma tiled_mma_qk, TiledCopyK tiled_copy_k, ThreadCopyK thread_copy_k,
    TiledCopyKSF tiled_copy_ksf, ThreadCopyKSF thread_copy_ksf,
    uint64_t* k_empty, int stage) {
  auto copy_view_k = thread_copy_k.retile_D(tSrK);
  auto copy_view_ksf = thread_copy_ksf.retile_D(tSrSFK);
  copy(tiled_copy_k, tSsK, copy_view_k(_, _, _0{}));
  copy(tiled_copy_ksf, tSsSFK, copy_view_ksf(_, _, _0{}));
  cute::gemm(tiled_mma_qk, make_zip_tensor(tSrQ(_, _, _0{}), tSrSFQ),
             make_zip_tensor(tSrK(_, _, _0{}), tSrSFK(_, _, _0{})), acc);
  cutlass::arch::ClusterBarrier::arrive(k_empty + stage);
}

// PV step of the blockscaled pipeline: O += (P . SFP) @ (V^T . SFVt) over
// one kv_tile's smem stage. Named gemm_rs_fp4 after the fp8/fp16
// convention (PV step = gemm_rs family; the _fp4 suffix keeps it apart
// from ffpa_cute::gemm_rs). A (P) never touches smem: the just-updated
// softmax scores are quantized+packed in regs by quantize_and_pack_p
// (fp4_pscale.cuh), matching gemm_rs's register-A smem-B form. The copy of
// v_block+1, its quantization, and the mma of v_block are issued in that
// order so the S2R loads overlap the mma chain; the trailing v_empty
// arrive doubles as the final "no more prefetch" branch, releasing the
// stage to the producer.
template <typename TensorC, typename TensorPA, typename TensorPSF,
          typename TensorVB, typename TensorVSF, typename TensorSV,
          typename TensorSVSF, typename TiledMma, typename TiledCopyV,
          typename ThreadCopyV, typename TiledCopyVSF, typename ThreadCopyVSF,
          typename AbsMaxTensor, typename AccConvTensor>
CUTE_DEVICE void gemm_rs_fp4(
    TensorC& tgt, TensorPA& tOrP, TensorPSF& tOrSFP, TensorVB& tOrVt,
    TensorVSF& tOrSFVt, TensorSV const& tOsVt, TensorSVSF const& tOsSFVt,
    TiledMma tiled_mma_pv, TiledCopyV tiled_copy_v, ThreadCopyV thread_copy_v,
    TiledCopyVSF tiled_copy_vsf, ThreadCopyVSF thread_copy_vsf,
    AbsMaxTensor& AbsMaxP, AccConvTensor& acc_conversion_view,
    uint64_t* v_empty, int v_stg) {
  auto copy_view_v = thread_copy_v.retile_D(tOrVt);
  auto copy_view_vsf = thread_copy_vsf.retile_D(tOrSFVt);
  auto tOsVt_stage = tOsVt(_, _, _, v_stg);
  auto tOsSFVt_stage = tOsSFVt(_, _, _, v_stg);
  copy(tiled_copy_v, tOsVt_stage(_, _, _0{}), copy_view_v(_, _, _0{}));
  copy(tiled_copy_vsf, tOsSFVt_stage(_, _, _0{}), copy_view_vsf(_, _, _0{}));
  quantize_and_pack_p(_0{}, AbsMaxP, acc_conversion_view, tOrP, tOrSFP);
  CUTLASS_PRAGMA_UNROLL
  for (int v_block = 0; v_block < size<2>(tOrP); ++v_block) {
    cute::gemm(tiled_mma_pv,
               make_zip_tensor(tOrP(_, _, v_block), tOrSFP(_, _, v_block)),
               make_zip_tensor(tOrVt(_, _, v_block), tOrSFVt(_, _, v_block)),
               tgt);
    if (v_block < size<2>(tOrP) - 1) {
      copy(tiled_copy_v, tOsVt_stage(_, _, v_block + 1),
           copy_view_v(_, _, v_block + 1));
      copy(tiled_copy_vsf, tOsSFVt_stage(_, _, v_block + 1),
           copy_view_vsf(_, _, v_block + 1));
      quantize_and_pack_p(v_block + 1, AbsMaxP, acc_conversion_view, tOrP,
                          tOrSFP);
    } else {
      cutlass::arch::ClusterBarrier::arrive(v_empty + v_stg);
    }
  }
}

// PV step with a PRE-QUANTIZED register A (m4n2): P crosses N-warps
// through the f32 smem roundtrip, so quantize+pack (quantize_pack_a_fp4)
// happens before the PV call on the readback fragment. Inlined into
// split_d_m4n2.cuh: B (V^T/SFVt chunk) loads element-wise from the mma's
// own smem partition views (the tiled-copy path under-fills the fragments
// under the m4n2 thr layout), single mma (Tile-K = kBc),
// trailing v_empty arrive.

// PV step of the MXFP8 path (QK NVFP4 x PV MXFP8): O += (P . SFA) @
// (V^T . SFVt) with P/V^T as e4m3 and SFs as ue8m0 (32-element groups).
// Same copy/mma/arrive protocol as gemm_rs_fp4; two deviations: the pack
// is quantize_and_pack_p_mxfp8 (quad-routed A slots + ue8m0 SFA) reading
// the scores through a REDUCTION view, and the k loop walks kBc/32 blocks.
template <typename TensorC, typename TensorPA, typename TensorPSF,
          typename TensorVB, typename TensorVSF, typename TensorSV,
          typename TensorSVSF, typename TiledMma, typename TiledCopyV,
          typename ThreadCopyV, typename TiledCopyVSF, typename ThreadCopyVSF,
          typename AbsMaxTensor, typename AccRedTensor>
CUTE_DEVICE void gemm_rs_mxfp8(
    TensorC& tgt, TensorPA& tOrP, TensorPSF& tOrSFA, TensorVB& tOrVt,
    TensorVSF& tOrSFVt, TensorSV const& tOsVt, TensorSVSF const& tOsSFVt,
    TiledMma tiled_mma_pv, TiledCopyV tiled_copy_v, ThreadCopyV thread_copy_v,
    TiledCopyVSF tiled_copy_vsf, ThreadCopyVSF thread_copy_vsf,
    AbsMaxTensor& AbsMaxP, AccRedTensor& acc_reduction_view, uint64_t* v_empty,
    int v_stg, int lane) {
  auto copy_view_v = thread_copy_v.retile_D(tOrVt);
  auto copy_view_vsf = thread_copy_vsf.retile_D(tOrSFVt);
  auto tOsVt_stage = tOsVt(_, _, _, v_stg);
  auto tOsSFVt_stage = tOsSFVt(_, _, _, v_stg);
  copy(tiled_copy_v, tOsVt_stage(_, _, _0{}), copy_view_v(_, _, _0{}));
  copy(tiled_copy_vsf, tOsSFVt_stage(_, _, _0{}), copy_view_vsf(_, _, _0{}));
  quantize_and_pack_p_mxfp8(0, AbsMaxP, acc_reduction_view, tOrP, tOrSFA, lane);
  CUTLASS_PRAGMA_UNROLL
  for (int v_block = 0; v_block < size<2>(tOrP); ++v_block) {
    cute::gemm(tiled_mma_pv,
               make_zip_tensor(tOrP(_, _, v_block), tOrSFA(_, _, v_block)),
               make_zip_tensor(tOrVt(_, _, v_block), tOrSFVt(_, _, v_block)),
               tgt);
    if (v_block < size<2>(tOrP) - 1) {
      copy(tiled_copy_v, tOsVt_stage(_, _, v_block + 1),
           copy_view_v(_, _, v_block + 1));
      copy(tiled_copy_vsf, tOsSFVt_stage(_, _, v_block + 1),
           copy_view_vsf(_, _, v_block + 1));
      quantize_and_pack_p_mxfp8(v_block + 1, AbsMaxP, acc_reduction_view, tOrP,
                                tOrSFA, lane);
    } else {
      cutlass::arch::ClusterBarrier::arrive(v_empty + v_stg);
    }
  }
}

}  // namespace ffpa_fp4

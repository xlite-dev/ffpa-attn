// NVFP4 split-D M4N2 forward kernel for sm_120, headdims in [768, 1024]
// (64-multiples): the regime where even the chunked split-D O accumulator
// (D/2 f32 regs/thread) passes the 255-register wall, so the atom layout
// switches to (4,2,1) - 4 M-warps x 2 N-warps - halving the per-thread O
// extent to D/4. Everything else is the split_d.cuh pipeline adapted to
// the N-warp split (64x64 tiles, element-wise P-roundtrip STS, rc-view
// delta_s preload):
//   * P crosses N-warps (each holds half the kBc columns): softmax runs in
//     the P domain writing f32 scores to a [kBr, kBc] smem
//     staging tile; after the roundtrip barrier each N-warp reads its PV
//     A-fragment slice back and quantizes it there (quantize_pack_a_fp4,
//     one-level per-16-k SF - the SoftmaxFused two-level fold needs the
//     full-row max, unavailable per half-row);
//   * row max/sum reduce across peer N-warps (warp_id ^ 4) through the
//     exchange buffer (fp8 m4n2 protocol: one barrier for the max, the sum
//     half published by the P roundtrip's __syncthreads and folded by
//     ffpa_cute::finalize_row_sum_m4n2).
// Kept from persist-D/split-D verbatim: persistent work loop with global
// chunk counters (barriers never re-init), SFQ resident per work (Q smem
// resident, data half copied per chunk),
// delta_s rank-1 preload, lazy rescale (row_scale warp vote), masking
// through kv_perm32, the P-domain lse formula with the qkm dot read from
// the resident Q smem before O staging overwrites it, batched R->S(STSM)
// ->TMA epilogue over the freed smem with a tail R->G fallback.
// lse is written by n_warp==0 only (both N-warps compute identical
// row_max/row_sum after the exchange).
#pragma once

#include <cuda_fp8.h>

#include <cute/tensor.hpp>
#include <cute/pointer_flagged.hpp>
#include <cute/tensor_zip.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cutlass.h>
#include <cutlass/device_kernel.h>
#include <cutlass/gemm/collective/collective_builder.hpp>

#include <algorithm>

#include "../../../common.cuh"
#include "../../gemm.cuh"
#include "../../softmax.cuh"
#include "../attn_traits.cuh"
#include "../cute_ext.h"
#include "../fp4_gemm.cuh"
#include "../fp4_pscale.cuh"

namespace ffpa_fp4 {

using TmaBarrier = cutlass::arch::ClusterTransactionBarrier;
using CtaBarrier = cutlass::arch::ClusterBarrier;

template <typename Traits, typename ElementO, typename TmaQ, typename TmaK,
          typename TmaV, typename TmaO, typename TmaSFQ, typename TmaSFK,
          typename TmaSFVt, typename TmaDS, typename TmaBias, int kBiasMode = 0,
          int kBias4B = 0, int kHasAttnBias = 0>
__global__ void __launch_bounds__(Traits::kNumThreads, 1)
    split_d_m4n2_fwd_cute_fp4_sm120(
        CUTLASS_GRID_CONSTANT TmaQ const tma_q,
        CUTLASS_GRID_CONSTANT TmaK const tma_k,
        CUTLASS_GRID_CONSTANT TmaV const tma_v,
        CUTLASS_GRID_CONSTANT TmaO const tma_o,
        CUTLASS_GRID_CONSTANT TmaSFQ const tma_sfq,
        CUTLASS_GRID_CONSTANT TmaSFK const tma_sfk,
        CUTLASS_GRID_CONSTANT TmaSFVt const tma_sfvt,
        CUTLASS_GRID_CONSTANT TmaDS const tma_ds,
        CUTLASS_GRID_CONSTANT TmaBias const tma_bias, ElementO* __restrict__ O,
        float* __restrict__ softmax_lse, const float* __restrict__ km,
        const float* __restrict__ qm, const float* __restrict__ vm, int Nq,
        int Nkv, int Nq_pad, int Nkv_pad, int Nh, int Nh_kv, float scale,
        int Tc, int causal, int total_q_rows, int Nb, int q_start_row = 0,
        bool nhd_out = false, const void* __restrict__ attn_bias = nullptr,
        int attn_bias_dtype = 0, long long attn_bias_stride_b = 0,
        long long attn_bias_stride_h = 0, long long attn_bias_stride_m = 0,
        long long attn_bias_stride_n = 0,
        long long attn_bias_plane_m_total = 0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  using namespace cute;
  using cute::tma_store_arrive;
  using cute::tma_store_wait;
  using Element = typename Traits::Element;
  using ElementSF = typename Traits::ElementSF;
  using SmemLayoutQ = typename Traits::SmemLayoutQ;
  using SmemLayoutK = typename Traits::SmemLayoutK;
  using SmemLayoutVt = typename Traits::SmemLayoutVt;
  using SmemLayoutSFQ = typename Traits::SmemLayoutSFQ;
  using SmemLayoutSFK = typename Traits::SmemLayoutSFK;
  using SmemLayoutSFVt = typename Traits::SmemLayoutSFVt;
  using SmemLayoutDS = typename Traits::SmemLayoutDS;
  using SmemLayoutP = typename Traits::SmemLayoutP;
  using SmemLayoutO = typename Traits::SmemLayoutO;
  using TiledMmaQK = typename Traits::TiledMmaQK;
  using TiledMmaPV = typename Traits::TiledMmaPV;
  using SmemCopyAtomQ = typename Traits::SmemCopyAtomQ;
  using SmemCopyAtomKV = typename Traits::SmemCopyAtomKV;
  using SmemCopyAtomSF = typename Traits::SmemCopyAtomSF;
  using BlkScaledConfig = typename Traits::BlkScaledConfig;

  constexpr int kBr = Traits::kBr;
  constexpr int kBc = Traits::kBc;
  constexpr int kHeadDim = Traits::kHeadDim;
  constexpr int kQKDChunk = Traits::kQKDChunk;
  constexpr int kVDChunk = Traits::kVDChunk;
  constexpr int kDChunksQK = Traits::kDChunksQK;
  constexpr int kDChunksV = Traits::kDChunksV;
  constexpr int kStagesQK = Traits::kStagesQK;
  constexpr int kStagesPV = Traits::kStagesPV;
  constexpr int kNumThreads = Traits::kNumThreads;
  constexpr int kOffQ = Traits::kOffQ;
  constexpr int kOffSFQ = Traits::kOffSFQ;
  constexpr int kOffK = Traits::kOffK;
  constexpr int kOffSFK = Traits::kOffSFK;
  constexpr int kOffDS = Traits::kOffDS;
  constexpr int kOffV = Traits::kOffV;
  constexpr int kOffSFVt = Traits::kOffSFVt;
  constexpr int kOffP = Traits::kOffP;
  (void)kOffSFVt;

  const int group_size = Nh / Nh_kv;
  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;
  const int n_warp = warp_id / 4;

  const int MB = (Nq - q_start_row + kBr - 1) / kBr;
  const int total_work = MB * Nb * Nh;

  extern __shared__ __align__(1024) char shm[];

  __shared__ uint64_t q_full;
  __shared__ uint64_t k_full[kStagesQK];
  __shared__ uint64_t k_empty[kStagesQK];
  __shared__ uint64_t v_full[kStagesPV];
  __shared__ uint64_t v_empty[kStagesPV];
  __shared__ uint64_t epilogue_done;
  // PC-0-1 bias tile: row-broadcast [1,kBc] double buffered (mode 2) or the
  // resident [1,Nkv] vector (mode 3), 16B-aligned past the P/exchange tail
  // -- outside kSmemBytes so neither the P staging nor the O epilogue's
  // batched staging (which reuse the freed area) ever alias it. Like the
  // K/V stages the bias barriers are never re-init: issuer and consumer
  // advance one global bias-tile counter each so phases stay aligned
  // across the grid-strided works.
  constexpr int kBiasStages = (kBiasMode == 2) ? 2 : 1;
  __shared__ uint64_t bias_full[kBiasStages];
  __shared__ uint64_t bias_empty[kBiasStages];
  uint16_t* bias_base =
      reinterpret_cast<uint16_t*>(shm + ((Traits::kSmemBytes + 15) & ~15));

  if (tid == 0) {
    TmaBarrier::init(&q_full, 1);
    for (int s = 0; s < kStagesQK; ++s) {
      TmaBarrier::init(&k_full[s], 1);
      CtaBarrier::init(&k_empty[s], kNumThreads);
    }
    for (int s = 0; s < kStagesPV; ++s) {
      TmaBarrier::init(&v_full[s], 1);
      CtaBarrier::init(&v_empty[s], kNumThreads);
    }
    CtaBarrier::init(&epilogue_done, kNumThreads);
    if constexpr (kHasAttnBias) {
      for (int s = 0; s < kBiasStages; ++s) {
        TmaBarrier::init(&bias_full[s], 1);
        CtaBarrier::init(&bias_empty[s], kNumThreads);
      }
    }
  }
  __syncthreads();

  auto mQ =
      tma_q.get_tma_tensor(make_shape((long)Nb * Nh * Nq_pad, Int<kHeadDim>{}));
  auto mK = tma_k.get_tma_tensor(
      make_shape((long)Nb * Nh_kv * Nkv_pad, Int<kHeadDim>{}));
  auto mV =
      tma_v.get_tma_tensor(make_shape((long)Nb * Nh_kv * kHeadDim, Nkv_pad));
  auto layout_SFQ = BlkScaledConfig::tile_atom_to_shape_SFQKV(
      make_shape(Nq_pad, Int<kHeadDim>{}, Nh, Nb));
  auto layout_SFK = BlkScaledConfig::tile_atom_to_shape_SFQKV(
      make_shape(Nkv_pad, Int<kHeadDim>{}, Nh_kv, Nb));
  auto layout_SFVt = BlkScaledConfig::tile_atom_to_shape_SFVt(
      make_shape(Int<kHeadDim>{}, Nkv_pad, Nh_kv, Nb));
  auto layout_DS = tile_to_shape(typename Traits::SmemLayoutAtomDS{},
                                 make_shape(Nq_pad / 128, Nkv_pad, Nh, Nb),
                                 Step<_2, _1, _3, _4>{});
  auto mSFQ = tma_sfq.get_tma_tensor(shape(layout_SFQ));
  auto mSFK = tma_sfk.get_tma_tensor(shape(layout_SFK));
  auto mSFVt = tma_sfvt.get_tma_tensor(shape(layout_SFVt));
  auto mDS = tma_ds.get_tma_tensor(shape(layout_DS));

  auto q_slice = tma_q.get_slice(_0{});
  auto k_slice = tma_k.get_slice(_0{});
  auto v_slice = tma_v.get_slice(_0{});
  auto sfq_slice = tma_sfq.get_slice(_0{});
  auto sfk_slice = tma_sfk.get_slice(_0{});
  auto sfvt_slice = tma_sfvt.get_slice(_0{});
  auto ds_slice = tma_ds.get_slice(_0{});

  auto sQ = make_tensor(make_smem_ptr<Element>(shm + kOffQ), SmemLayoutQ{});
  auto sSFQ =
      make_tensor(make_smem_ptr<ElementSF>(shm + kOffSFQ), SmemLayoutSFQ{});
  auto sK = make_tensor(make_smem_ptr<Element>(shm + kOffK), SmemLayoutK{});
  auto sSFK =
      make_tensor(make_smem_ptr<ElementSF>(shm + kOffSFK), SmemLayoutSFK{});
  auto sDS = make_tensor(make_smem_ptr<float>(shm + kOffDS), SmemLayoutDS{});
  auto sV = make_tensor(make_smem_ptr<Element>(shm + kOffV), SmemLayoutVt{});
  auto sSFVt =
      make_tensor(make_smem_ptr<ElementSF>(shm + kOffSFVt), SmemLayoutSFVt{});
  auto sP = make_tensor(make_smem_ptr<float>(shm + kOffP), SmemLayoutP{});
  float* smem_exchange = reinterpret_cast<float*>(shm + Traits::kOffExchange);

  auto tQsQ = q_slice.partition_D(sQ);
  auto tQsSFQ = sfq_slice.partition_D(sSFQ);

  // TMA issue helpers (tid==0 only), identical to split_d.cuh: Q+SFQ share
  // q_full; DS rides the kv tile's first K chunk barrier.
  auto issue_q = [&](int q_bh, int Q_tile_id, int Nh_id, int b,
                     int q_tile_abs) {
    const int q_row_offset = q_bh * Nq_pad + q_start_row;
    auto gQ = local_tile(domain_offset(make_coord(q_row_offset, _0{}), mQ),
                         Shape<Int<kBr>, Int<kHeadDim>>{},
                         make_coord(Q_tile_id, _0{}));
    auto gSFQ =
        local_tile(mSFQ(_, _, Nh_id, b), Shape<Int<kBr>, Int<kHeadDim>>{},
                   make_coord(q_tile_abs, _0{}));
    TmaBarrier::arrive_and_expect_tx(&q_full, Traits::kTxBytesQ);
    copy(tma_q.with(q_full), q_slice.partition_S(gQ), tQsQ);
    copy(tma_sfq.with(q_full), sfq_slice.partition_S(gSFQ), tQsSFQ);
  };
  auto issue_k_chunk = [&](int kv_bh, int Nh_id, int b, int q_tile_abs,
                           int kv_tile, int d_chunk, int stage) {
    const int kv_row_offset = kv_bh * Nkv_pad;
    auto gK = local_tile(domain_offset(make_coord(kv_row_offset, _0{}), mK),
                         Shape<Int<kBc>, Int<kQKDChunk>>{},
                         make_coord(kv_tile, d_chunk));
    auto gSFK = local_tile(mSFK(_, _, kv_bh % Nh_kv, kv_bh / Nh_kv),
                           Shape<Int<kBc>, Int<kQKDChunk>>{},
                           make_coord(kv_tile, d_chunk));
    auto sK_st = make_tensor(
        make_smem_ptr<Element>(shm + kOffK + stage * Traits::kKBytesStage),
        typename Traits::SmemLayoutKStage{});
    auto sSFK_st =
        make_tensor(make_smem_ptr<ElementSF>(shm + kOffSFK +
                                             stage * Traits::kSFKBytesStage),
                    typename Traits::SmemLayoutSFKStage{});
    cutlass::arch::fence_view_async_shared();
    if (d_chunk == 0) {
      // DS is a 128-row-block constant vector; index it by the block
      // (q_tile_abs*kBr = q_start_row + Q_tile_id*kBr).
      auto gDS = local_tile(mDS(_, _, Nh_id, b), Shape<_1, Int<kBc>>{},
                            make_coord((q_tile_abs * kBr) / 128, kv_tile));
      auto sDS_st = make_tensor(
          make_smem_ptr<float>(shm + kOffDS + stage * Traits::kDSBytesStage),
          typename Traits::SmemLayoutDSStage{});
      TmaBarrier::arrive_and_expect_tx(&k_full[stage],
                                       Traits::kTxBytesK + Traits::kTxBytesDS);
      copy(tma_k.with(k_full[stage]), k_slice.partition_S(gK),
           k_slice.partition_D(sK_st));
      copy(tma_sfk.with(k_full[stage]), sfk_slice.partition_S(gSFK),
           sfk_slice.partition_D(sSFK_st));
      copy(tma_ds.with(k_full[stage]), ds_slice.partition_S(gDS),
           ds_slice.partition_D(sDS_st));
    } else {
      TmaBarrier::arrive_and_expect_tx(&k_full[stage], Traits::kTxBytesK);
      copy(tma_k.with(k_full[stage]), k_slice.partition_S(gK),
           k_slice.partition_D(sK_st));
      copy(tma_sfk.with(k_full[stage]), sfk_slice.partition_S(gSFK),
           sfk_slice.partition_D(sSFK_st));
    }
  };
  auto issue_v_chunk = [&](int kv_bh, int kv_tile, int v_chunk, int stage) {
    const int v_row_base = kv_bh * kHeadDim + v_chunk * kVDChunk;
    auto gV =
        local_tile(domain_offset(make_coord(v_row_base, _0{}), mV),
                   Shape<Int<kVDChunk>, Int<kBc>>{}, make_coord(_0{}, kv_tile));
    auto gSFVt = local_tile(mSFVt(_, _, kv_bh % Nh_kv, kv_bh / Nh_kv),
                            Shape<Int<kVDChunk>, Int<kBc>>{},
                            make_coord(v_chunk, kv_tile));
    auto sV_st = make_tensor(
        make_smem_ptr<Element>(shm + kOffV + stage * Traits::kVBytesStage),
        typename Traits::SmemLayoutVtStage{});
    auto sSFVt_st =
        make_tensor(make_smem_ptr<ElementSF>(shm + kOffSFVt +
                                             stage * Traits::kSFVtBytesStage),
                    typename Traits::SmemLayoutSFVtStage{});
    cutlass::arch::fence_view_async_shared();
    TmaBarrier::arrive_and_expect_tx(&v_full[stage], Traits::kTxBytesV);
    copy(tma_v.with(v_full[stage]), v_slice.partition_S(gV),
         v_slice.partition_D(sV_st));
    copy(tma_sfvt.with(v_full[stage]), sfvt_slice.partition_S(gSFVt),
         sfvt_slice.partition_D(sSFVt_st));
  };

  // PC-0-1 bias tile (fp4 split_d pattern): the stage/phase come from the
  // global bias_g counter (barriers never re-init), and mBias folds this
  // work's (b,h) row -- row-broadcast rows are Nkv elements wide
  // (host-validated strides).
  int bias_g = 0;  // global bias-tile count; bias_full/empty never re-init
  auto b_slice = tma_bias.get_slice(_0{});
  constexpr int bias_cols = kBc * (kBias4B ? 2 : 1);
  auto issue_bias_tma = [&](int b, int h, int tile) {
    cutlass::arch::fence_view_async_shared();
    const int stage = bias_g % kBiasStages;
    const int phase = (bias_g / kBiasStages) & 1;
    CtaBarrier::wait(&bias_empty[stage], phase);
    auto mBias = domain_offset(
        make_coord(((long long)b * attn_bias_stride_b +
                    (long long)h * attn_bias_stride_h) /
                       (long long)Nkv,
                   0LL),
        tma_bias.get_tma_tensor(make_shape(attn_bias_plane_m_total,
                                           (long long)Nkv * bias_cols / kBc)));
    auto sB = make_tensor(
        make_smem_ptr(bias_base + stage * bias_cols),
        Layout<Shape<_1, Int<bias_cols>>, Stride<Int<bias_cols>, _1>>{});
    auto gB =
        local_tile(mBias, Shape<_1, Int<bias_cols>>{}, make_coord(_0{}, tile));
    TmaBarrier::arrive_and_expect_tx(&bias_full[stage],
                                     sizeof(uint16_t) * bias_cols);
    copy(tma_bias.with(bias_full[stage]), b_slice.partition_S(gB),
         b_slice.partition_D(sB));
    ++bias_g;
  };

  for (int s = 0; s < kStagesQK; ++s)
    CtaBarrier::arrive(&k_empty[s]);
  for (int s = 0; s < kStagesPV; ++s)
    CtaBarrier::arrive(&v_empty[s]);
  if constexpr (kHasAttnBias) {
    for (int s = 0; s < kBiasStages; ++s)
      CtaBarrier::arrive(&bias_empty[s]);
  }

  TiledMmaQK tiled_mma_qk;
  TiledMmaPV tiled_mma_pv;
  auto thread_mma_qk = tiled_mma_qk.get_thread_slice(tid);
  auto thread_mma_pv = tiled_mma_pv.get_thread_slice(tid);

  Tensor tSrK = thread_mma_qk.partition_fragment_B(sK(_, _, Int<0>{}));
  Tensor tOrVt = thread_mma_pv.partition_fragment_B(sV(_, _, Int<0>{}));
  Tensor tSrSFQ = partition_fragment_SFA(sSFQ, thread_mma_qk);
  Tensor tSrSFK = partition_fragment_SFB(sSFK(_, _, Int<0>{}), thread_mma_qk);
  Tensor tOrSFVt = partition_fragment_SFB(sSFVt(_, _, Int<0>{}), thread_mma_pv);

  auto smem_tiled_copy_Q = make_tiled_copy_A(SmemCopyAtomQ{}, tiled_mma_qk);
  auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tid);
  Tensor tSsQ =
      smem_thr_copy_Q.partition_S(as_position_independent_swizzle_tensor(sQ));

  auto tile_shape_mnk = tile_shape(tiled_mma_qk);
  auto smem_tiled_copy_SFQ = make_tiled_copy_impl(
      SmemCopyAtomSF{}, get_layoutSFA_TV(tiled_mma_qk),
      make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk)));
  auto smem_thr_copy_SFQ = smem_tiled_copy_SFQ.get_thread_slice(tid);
  Tensor tSsSFQ = smem_thr_copy_SFQ.partition_S(
      as_position_independent_swizzle_tensor(sSFQ));
  Tensor tSrSFQ_copy_view = smem_thr_copy_SFQ.retile_D(tSrSFQ);

  // K/V/SF fragments load element-wise from the mma's own smem partition
  // views instead of tiled copies: under this m4n2 TiledMma thr layout the
  // SM75_U32x4_LDSM_N / 1-byte tiled copies only populate part of the
  // register fragments (warp1's B and 3/4 of every SF byte stay unloaded,
  // zeroing 3/4 of the k-groups in every mxf4nvf4 mma).

  // P roundtrip plumbing: each N-warp reads its PV A-fragment slice back
  // from sP as f32 (UniversalCopy over the mma's A thread partition - the
  // dst fragment layout matches the e2m1 A operand's, so the register
  // quantizer packs straight into the operand slots) and quantizes in
  // registers. The write side is the permuted element-wise STS in the kv
  // loop (see the P roundtrip block). The e2m1/SF fragment shapes come
  // from shadow (nullptr) tensors of the operand layouts - shape only, no
  // access.
  auto s2r_copy_p =
      make_tiled_copy_A(Copy_Atom<UniversalCopy<float>, float>{}, tiled_mma_pv);
  auto s2r_thr_p = s2r_copy_p.get_thread_slice(tid);
  Tensor tPsP = s2r_thr_p.partition_S(sP);
  // Register fragments carry the persist-D rank-3 LayoutP/LayoutSFP
  // adapters (k-iter = 1): make_zip_tensor(tPA, tPASF) requires equal
  // ranks, and the raw partition_fragment_A layout is rank-2 here.
  Tensor tPf32 = make_tensor_like<float>(typename Traits::LayoutP{});
  Tensor tPA = make_tensor_like<Element>(typename Traits::LayoutP{});
  Tensor tPASF = make_tensor_like<ElementSF>(typename Traits::LayoutSFP{});

  Tensor tSrS = partition_fragment_C(tiled_mma_qk, Shape<Int<kBr>, Int<kBc>>{});
  auto cS = make_identity_tensor(Shape<Int<kBr>, Int<kBc>>{});
  auto tScS = thread_mma_qk.partition_C(cS);
  auto tScS_rc =
      make_tensor(tScS.data(), convert_to_reduction_layout(tScS.layout()));
  constexpr int kSRows = decltype(size<0>(tScS_rc))::value;
  constexpr int kSCols = decltype(size<1>(tScS_rc))::value;
  constexpr int kSoftmaxRows = kSRows;

  using OFragType = decltype(partition_fragment_C(
      tiled_mma_pv, Shape<Int<kBr>, Int<kVDChunk>>{}));
  using OFragLayout = typename OFragType::layout_type;
  constexpr int kOElemsPerFrag = decltype(size(OFragType{}))::value;
  float o_acc_storage[kDChunksV][kOElemsPerFrag];

  const float scale_orig = scale;
  const float softmax_scale_log2 = scale * FFPA_M_LOG2E;

  // delta_s rank-1 preload: the m4n2 C-fragment covers half the kBc cols,
  // so the M8N1 float4-slot math does not apply - assign through the rc
  // view (element-wise STS.32 to registers; runs once per kv tile). The
  // ASSIGN doubles as the per-tile acc clear (split_d does the same via its
  // float4 store): gemm accumulates on top, and the previous tile's P must
  // not leak in.
  // delta_s is computed from the ORIGINAL (unpermuted) K and stored in
  // logical kv order; the C-fragment column j scores K smem row j, whose
  // logical kv is kv_perm32(j) (same mapping the masking uses), so the DS
  // read goes through the permutation.
  auto add_delta_s = [&](auto& acc, int stage) {
    auto acc_rc =
        make_tensor(acc.data(), convert_to_reduction_layout(acc.layout()));
    CUTLASS_PRAGMA_UNROLL
    for (int row = 0; row < kSRows; ++row) {
      CUTLASS_PRAGMA_UNROLL
      for (int col = 0; col < kSCols; ++col) {
        acc_rc(row, col) =
            sDS(_0{}, kv_perm32(cute::get<1>(tScS_rc(row, col))), stage);
      }
    }
  };

  int first_bh, first_Q_tile, first_Nh_id, first_b, first_kv_bh, first_Tc_eff;
  {
    const int work_id = blockIdx.x;
    const int bh = work_id / MB;
    const int Q_tile_id = work_id % MB;
    const int b = bh / Nh;
    const int Nh_id = bh % Nh;
    const int kv_head_idx = Nh_id / group_size;
    const int kv_offset = Nkv - Nq;
    const int Br_base = Q_tile_id * kBr;
    first_bh = bh;
    first_Q_tile = Q_tile_id;
    first_Nh_id = Nh_id;
    first_b = b;
    first_kv_bh = b * Nh_kv + kv_head_idx;
    first_Tc_eff =
        causal
            ? min(Tc, ((q_start_row + Br_base + kBr - 1 + kv_offset) / kBc) + 1)
            : Tc;
  }
  if (tid == 0) {
    issue_q(first_bh, first_Q_tile, first_Nh_id, first_b,
            first_Q_tile + q_start_row / kBr);
    for (int i = 0; i < kStagesQK && i < first_Tc_eff * kDChunksQK; ++i) {
      const int stage = i % kStagesQK;
      const int phase = (i / kStagesQK) & 1;
      CtaBarrier::wait(&k_empty[stage], phase);
      issue_k_chunk(first_kv_bh, first_Nh_id, first_b,
                    first_Q_tile + q_start_row / kBr, i / kDChunksQK,
                    i % kDChunksQK, stage);
    }
    for (int i = 0; i < kStagesPV && i < first_Tc_eff * kDChunksV; ++i) {
      const int stage = i % kStagesPV;
      const int phase = (i / kStagesPV) & 1;
      CtaBarrier::wait(&v_empty[stage], phase);
      issue_v_chunk(first_kv_bh, i / kDChunksV, i % kDChunksV, stage);
    }
    if constexpr (kHasAttnBias && kBiasMode == 2) {
      if (first_Tc_eff > 0)
        issue_bias_tma(first_b, first_Nh_id, 0);
    }
  }
  TmaBarrier::wait(&q_full, 0);
  cutlass::arch::fence_view_async_shared();

  int gK = 0;
  int gV = 0;
  int w = 0;
  int bias_gc = 0;  // consumer-side mirror of bias_g (never re-init)
  for (int work_id = blockIdx.x; work_id < total_work;
       work_id += gridDim.x, ++w) {
    const int kv_offset = Nkv - Nq;
    const int bh = work_id / MB;
    const int Q_tile_id = work_id % MB;
    const int Nb_id = bh / Nh;
    const int Nh_id = bh % Nh;
    const int kv_head_idx = Nh_id / group_size;
    const int Br_base = Q_tile_id * kBr;
    const int causal_thresh_row0 = q_start_row + Br_base + kv_offset;
    const int Tc_eff =
        causal
            ? min(Tc, ((q_start_row + Br_base + kBr - 1 + kv_offset) / kBc) + 1)
            : Tc;
    const int mask_start_tile =
        causal ? max(0, (causal_thresh_row0 + 1) / kBc) : INT_MAX;
    const int q_bh = bh;
    const int kv_bh = Nb_id * Nh_kv + kv_head_idx;
    const int q_tile_abs = Q_tile_id + q_start_row / kBr;
    const int O_row_offset = q_bh * Nq + q_start_row;
    const int work_chunks_qk = Tc_eff * kDChunksQK;
    const int work_chunks_v = Tc_eff * kDChunksV;

    if (w > 0) {
      if (tid == 0) {
        CtaBarrier::wait(&epilogue_done, (w - 1) & 1);
        issue_q(q_bh, Q_tile_id, Nh_id, Nb_id, q_tile_abs);
        for (int i = 0; i < kStagesQK && i < work_chunks_qk; ++i) {
          const int seq = gK + i;
          const int stage = seq % kStagesQK;
          const int phase = (seq / kStagesQK) & 1;
          CtaBarrier::wait(&k_empty[stage], phase);
          issue_k_chunk(kv_bh, Nh_id, Nb_id, q_tile_abs, i / kDChunksQK,
                        i % kDChunksQK, stage);
        }
        for (int i = 0; i < kStagesPV && i < work_chunks_v; ++i) {
          const int seq = gV + i;
          const int stage = seq % kStagesPV;
          const int phase = (seq / kStagesPV) & 1;
          CtaBarrier::wait(&v_empty[stage], phase);
          issue_v_chunk(kv_bh, i / kDChunksV, i % kDChunksV, stage);
        }
        if constexpr (kHasAttnBias && kBiasMode == 2) {
          if (Tc_eff > 0)
            issue_bias_tma(Nb_id, Nh_id, 0);
        }
      }
      TmaBarrier::wait(&q_full, w & 1);
      cutlass::arch::fence_view_async_shared();
    }

    // Mode 3: load this work's (b,h) resident [1,Nkv] row-broadcast vector
    // once per work (plain vector loads, host-guaranteed 16B alignment) --
    // no bias TMA and no per-tile bias barrier in the kv loop. The bias
    // area sits past kSmemBytes so neither the P staging nor the O
    // epilogue's batched staging ever aliases it; a plain CTA sync orders
    // the loads against the injection.
    if constexpr (kHasAttnBias && kBiasMode == 3) {
      const uint16_t* src = reinterpret_cast<const uint16_t*>(attn_bias) +
                            ((long long)Nb_id * attn_bias_stride_b +
                             (long long)Nh_id * attn_bias_stride_h) *
                                ((attn_bias_dtype == 3) ? 2 : 1);
      const int n_u16 = (int)Nkv * ((attn_bias_dtype == 3) ? 2 : 1);
      const int vec_end = n_u16 & ~7;
      for (int i = tid * 8; i < vec_end; i += kNumThreads * 8)
        *reinterpret_cast<uint4*>(bias_base + i) =
            *reinterpret_cast<const uint4*>(src + i);
      for (int i = vec_end + tid; i < n_u16; i += kNumThreads)
        bias_base[i] = src[i];
      __syncthreads();
      // PC5-E1 (diagnostic): sentinel-fill the P staging each work. If the
      // manual readback ever reads an unwritten slot the error explodes to
      // ~1e37; if the error stays ~1e-2 the readback values are fine and
      // the corruption sits further downstream (PV / quant / epilogue).
      {
        float* sP_raw = reinterpret_cast<float*>(shm + kOffP);
        const int nP = (int)kBr * kBc;
        for (int i = tid; i < nP; i += kNumThreads)
          sP_raw[i] = __int_as_float(0x7e7e7e7e);
        __syncthreads();
      }
    }

    copy(smem_tiled_copy_SFQ, tSsSFQ, tSrSFQ_copy_view);

    float row_max[kSoftmaxRows];
    float row_sum[kSoftmaxRows];
    float row_scale[kSoftmaxRows];
#pragma unroll
    for (int r = 0; r < kSoftmaxRows; ++r) {
      row_max[r] = -INFINITY;
      row_sum[r] = 0.0f;
      row_scale[r] = 1.0f;
    }
#pragma unroll
    for (int v = 0; v < kDChunksV; ++v)
#pragma unroll
      for (int i = 0; i < kOElemsPerFrag; ++i)
        o_acc_storage[v][i] = 0.0f;

    const int g0k = gK;
    const int g0v = gV;

#pragma unroll 1
    for (int kv_tile = 0; kv_tile < Tc_eff; ++kv_tile) {
      // Guards the P staging tile across kv tiles: the previous tile's A
      // readback must retire before this tile's r2s write (fp8 m4n2 order).
      if (kv_tile > 0)
        __syncthreads();

      // 2-stage bias prefetch: issue (t+1) before this tile's QK/softmax
      // so the TMA hides behind them; empty-wait(t+1) needs only the
      // previous tile's injection arrive, which finished last iteration.
      // Mode 3 has no per-tile bias traffic at all.
      if constexpr (kHasAttnBias && kBiasMode == 2) {
        if (kv_tile + 1 < Tc_eff && tid == 0)
          issue_bias_tma(Nb_id, Nh_id, kv_tile + 1);
      }

#pragma unroll
      for (int d_chunk = 0; d_chunk < kDChunksQK; ++d_chunk) {
        const int seq = g0k + kv_tile * kDChunksQK + d_chunk;
        const int k_stg = seq % kStagesQK;
        const int k_phase = (seq / kStagesQK) & 1;
        TmaBarrier::wait(&k_full[k_stg], k_phase);
        cutlass::arch::fence_view_async_shared();

        if (d_chunk == 0)
          add_delta_s(tSrS, k_stg);  // '=' not '+=': doubles as the tile clear
        auto sQ_chunk = local_tile(sQ, Shape<Int<kBr>, Int<kQKDChunk>>{},
                                   make_coord(_0{}, d_chunk));
        Tensor tSrQ_c = thread_mma_qk.partition_fragment_A(sQ_chunk);
        Tensor tSrQ_c_view = smem_thr_copy_Q.retile_D(tSrQ_c);
        Tensor tSsQ_chunk = smem_thr_copy_Q.partition_S(
            as_position_independent_swizzle_tensor(sQ_chunk));
        copy(smem_tiled_copy_Q, tSsQ_chunk, tSrQ_c_view);
        auto tSrSFQ_c = tSrSFQ(_, _, d_chunk);
        {
          auto b_smem = recast<uint4_t>(
              flatten(thread_mma_qk.partition_B(sK(_, _, k_stg))));
          auto b_frag = flatten(tSrK);
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < (int)size(b_frag); ++i)
            b_frag(i) = b_smem(i);
          auto sf_smem = filter_zeros(
              flatten(cute::partition_SFB(sSFK(_, _, k_stg), thread_mma_qk)));
          auto sf_frag = filter_zeros(flatten(tSrSFK));
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < (int)size(sf_frag); ++i)
            sf_frag(i) = sf_smem(i);
        }
        cute::gemm(tiled_mma_qk, make_zip_tensor(tSrQ_c(_, _, _0{}), tSrSFQ_c),
                   make_zip_tensor(tSrK(_, _, _0{}), tSrSFK(_, _, _0{})), tSrS);
        cutlass::arch::ClusterBarrier::arrive(k_empty + k_stg);

        if (tid == 0) {
          const int next = kv_tile * kDChunksQK + d_chunk + kStagesQK;
          if (next < work_chunks_qk) {
            const int seq_n = g0k + next;
            const int s_next = seq_n % kStagesQK;
            const int phase_next = (seq_n / kStagesQK) & 1;
            CtaBarrier::wait(&k_empty[s_next], phase_next);
            issue_k_chunk(kv_bh, Nh_id, Nb_id, q_tile_abs, next / kDChunksQK,
                          next % kDChunksQK, s_next);
          }
        }
      }

      // Masking (persist_d verbatim; m4n2's tScS_rc covers half the cols).
      {
        auto scores = make_tensor(tSrS.data(),
                                  convert_to_reduction_layout(tSrS.layout()));
        // Additive attn bias in the dequantized score domain; the softmax
        // below applies softmax_scale_log2, so bias/scale_orig lands as
        // +bias in softmax-input units. The -INFINITY assignments in the
        // masking below simply override it. Tile path (PC-0-1): the smem
        // tile holds ORIGINAL token order, the injection indexes it
        // through kv_perm32 just like the gmem variant (tScS_rc covers
        // half the cols -- same as the gmem call).
        if constexpr (kHasAttnBias && kBiasMode != 0) {
          const int b_stg = bias_gc % kBiasStages;
          if constexpr (kBiasMode != 3) {
            const int b_phase = (bias_gc / kBiasStages) & 1;
            TmaBarrier::wait(&bias_full[b_stg], b_phase);
            cutlass::arch::fence_view_async_shared();
          }
          const int b_slot_u16 = kBc * ((attn_bias_dtype == 3) ? 2 : 1);
          // mode 3: the resident vector's tile-t segment sits at t*kBc.
          const uint16_t* b_slot =
              bias_base + (kBiasMode == 3 ? (long long)kv_tile * kBc *
                                                ((attn_bias_dtype == 3) ? 2 : 1)
                                          : (long long)b_stg * b_slot_u16);
          if (attn_bias_dtype == 3)
            ffpa_fp4::apply_attn_bias_fp4_rowcol_smem<
                float, decltype(scores), decltype(tScS_rc), kSRows, kSCols>(
                scores, tScS_rc, reinterpret_cast<const float*>(b_slot), 0, 1,
                1.0f / scale_orig);
          else if (attn_bias_dtype == 2)
            ffpa_fp4::apply_attn_bias_fp4_rowcol_smem<
                cutlass::bfloat16_t, decltype(scores), decltype(tScS_rc),
                kSRows, kSCols>(
                scores, tScS_rc,
                reinterpret_cast<const cutlass::bfloat16_t*>(b_slot), 0, 1,
                1.0f / scale_orig);
          else
            ffpa_fp4::apply_attn_bias_fp4_rowcol_smem<
                cutlass::half_t, decltype(scores), decltype(tScS_rc), kSRows,
                kSCols>(scores, tScS_rc,
                        reinterpret_cast<const cutlass::half_t*>(b_slot), 0, 1,
                        1.0f / scale_orig);
          if constexpr (kBiasMode != 3)
            CtaBarrier::arrive(&bias_empty[b_stg]);
          ++bias_gc;
        } else if constexpr (kHasAttnBias) {
          ffpa_fp4::apply_attn_bias_fp4_rowcol<
              decltype(scores), decltype(tScS_rc), kSRows, kSCols>(
              scores, tScS_rc, attn_bias, attn_bias_dtype, attn_bias_stride_b,
              attn_bias_stride_h, attn_bias_stride_m, attn_bias_stride_n, Nb_id,
              Nh_id, q_start_row + Br_base, kv_tile, kBc, 1.0f / scale_orig);
        }
        const int kv_valid = Nkv - kv_tile * kBc;
        const bool tail_tile = kv_valid < kBc;
        const bool causal_tile = kv_tile >= mask_start_tile;
        if (tail_tile || causal_tile) {
          CUTLASS_PRAGMA_UNROLL
          for (int row = 0; row < kSRows; ++row) {
            const int q_pos = q_start_row + Br_base +
                              cute::get<0>(tScS_rc(row, 0)) + kv_offset;
            CUTLASS_PRAGMA_UNROLL
            for (int col = 0; col < kSCols; ++col) {
              const int j = cute::get<1>(tScS_rc(row, col));
              const int k_pos = kv_tile * kBc + kv_perm32(j);
              if (tail_tile && kv_perm32(j) >= kv_valid)
                scores(row, col) = -INFINITY;
              if (causal_tile && k_pos > q_pos)
                scores(row, col) = -INFINITY;
            }
          }
        }
      }

      // Cross-N-warp softmax in the P domain (f32 scores in tSrS).
      {
        auto scores_p2 = make_tensor(
            tSrS.data(), convert_to_reduction_layout(tSrS.layout()));
        online_softmax_p2_m4n2<decltype(scores_p2), decltype(tScS_rc),
                               kSoftmaxRows>(
            scores_p2, tScS_rc, softmax_scale_log2, row_max, row_sum, row_scale,
            smem_exchange, warp_id, lane_id);
      }
      // Lazy rescale: row_scale == 1.0f when the running row max did not
      // grow on this tile; warp-vote the skip.
      const bool need_rescale =
          kv_tile > 0 &&
          __any_sync(0xffffffff, row_scale[0] != 1.0f || row_scale[1] != 1.0f);
      if (need_rescale) {
#pragma unroll
        for (int v = 0; v < kDChunksV; ++v) {
          auto tCrO =
              make_tensor(make_rmem_ptr(&o_acc_storage[v][0]), OFragLayout{});
          auto tCrO_rc = make_tensor(
              tCrO.data(), convert_to_reduction_layout(tCrO.layout()));
#pragma unroll
          for (int row = 0; row < kSoftmaxRows; ++row)
#pragma unroll
            for (int col = 0; col < decltype(size<1>(tCrO_rc))::value; ++col)
              tCrO_rc(row, col) *= row_scale[row];
        }
      }

      // P roundtrip: r2s the P-domain scores, barrier, fold peer sums,
      // read back + quantize the A operand, then PV over the V chunks.
      // Column alignment: C fragment col j scores K storage row j =
      // original token kv_perm32(j) (K quantized with kPermute=true), but
      // V is quantized by fp4_quant_trans_kernel with NO permutation, so
      // sVt col k = original token k. The PV mma pairs A slot k with B
      // slot k, hence sP col c must hold P(original c): write C col j to
      // smem col kv_perm32(j) (involution folds the K-side permutation).
      {
        auto scores_rc = make_tensor(
            tSrS.data(), convert_to_reduction_layout(tSrS.layout()));
        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < kSRows; ++row) {
          CUTLASS_PRAGMA_UNROLL
          for (int col = 0; col < kSCols; ++col) {
            const auto c = tScS_rc(row, col);
            sP(cute::get<0>(c), kv_perm32(cute::get<1>(c))) =
                scores_rc(row, col);
          }
        }
        cutlass::arch::fence_view_async_shared();
        __syncthreads();
        ffpa_cute::finalize_row_sum_m4n2<kSoftmaxRows>(
            row_sum, row_scale, smem_exchange, warp_id, lane_id);
        copy(s2r_copy_p, tPsP, s2r_thr_p.retile_D(tPf32));
        // retile_D writes the copy's k-major (v1 = k+8) slot order into
        // tPf32, but the PV A operand needs the PTX fragment order
        // (slot v1 = m+8): reg0/2 = row gid at k = 8*tig (+32), reg1/3 =
        // row gid+8, same k windows (mma_traits_sm120 ALayout). Reload the
        // fragment element-wise straight from sP in the operand's (m,k)
        // order.
        {
          Tensor pf = flatten(tPf32);
          int const tig = lane_id & 3;
          int const gid = lane_id >> 2;
          // warp M base: atoms are (warp_id % 4) along M (see AtomLayoutMNK)
          int const wrow = (warp_id % 4) * 16 + gid;
          CUTLASS_PRAGMA_UNROLL
          for (int v = 0; v < 32; ++v) {
            int const v0 = v & 7, v1 = (v >> 3) & 1, v2 = (v >> 4) & 1;
            pf(v) = sP(wrow + 8 * v1, 8 * tig + v0 + 32 * v2);
          }
        }
        quantize_pack_a_fp4(tPf32, tPA, tPASF);
      }

#pragma unroll
      for (int v_chunk = 0; v_chunk < kDChunksV; ++v_chunk) {
        const int seq = g0v + kv_tile * kDChunksV + v_chunk;
        const int v_stg = seq % kStagesPV;
        const int v_phase = (seq / kStagesPV) & 1;
        TmaBarrier::wait(&v_full[v_stg], v_phase);
        cutlass::arch::fence_view_async_shared();

        auto tCrO = make_tensor(make_rmem_ptr(&o_acc_storage[v_chunk][0]),
                                OFragLayout{});
        {
          auto b_smem = recast<uint4_t>(
              flatten(thread_mma_pv.partition_B(sV(_, _, v_stg))));
          auto b_frag = flatten(tOrVt);
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < (int)size(b_frag); ++i)
            b_frag(i) = b_smem(i);
          auto sf_smem = filter_zeros(
              flatten(cute::partition_SFB(sSFVt(_, _, v_stg), thread_mma_pv)));
          auto sf_frag = filter_zeros(flatten(tOrSFVt));
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < (int)size(sf_frag); ++i)
            sf_frag(i) = sf_smem(i);
        }
        cute::gemm(
            tiled_mma_pv, make_zip_tensor(tPA(_, _, _0{}), tPASF(_, _, _0{})),
            make_zip_tensor(tOrVt(_, _, _0{}), tOrSFVt(_, _, _0{})), tCrO);
        cutlass::arch::ClusterBarrier::arrive(v_empty + v_stg);

        if (tid == 0) {
          const int next = kv_tile * kDChunksV + v_chunk + kStagesPV;
          if (next < work_chunks_v) {
            const int seq_n = g0v + next;
            const int s_next = seq_n % kStagesPV;
            const int phase_next = (seq_n / kStagesPV) & 1;
            CtaBarrier::wait(&v_empty[s_next], phase_next);
            issue_v_chunk(kv_bh, next / kDChunksV, next % kDChunksV, s_next);
          }
        }
      }
    }

    // O = o_acc / row_sum (the P-domain 2688 cancels); every D chunk
    // scales by the same 1/row_sum.
    {
      auto tCrO0 =
          make_tensor(make_rmem_ptr(&o_acc_storage[0][0]), OFragLayout{});
      auto tCrO0_rc = make_tensor(tCrO0.data(),
                                  convert_to_reduction_layout(tCrO0.layout()));
#pragma unroll
      for (int row = 0; row < kSoftmaxRows; ++row) {
        const float sum = row_sum[row];
        const float inv_sum = (sum == 0.f || sum != sum) ? 0.f : 1 / sum;
#pragma unroll
        for (int col = 0; col < decltype(size<1>(tCrO0_rc))::value; ++col)
          tCrO0_rc(row, col) *= inv_sum;
      }
#pragma unroll
      for (int v = 1; v < kDChunksV; ++v) {
        auto tCrO =
            make_tensor(make_rmem_ptr(&o_acc_storage[v][0]), OFragLayout{});
        auto tCrO_rc = make_tensor(tCrO.data(),
                                   convert_to_reduction_layout(tCrO.layout()));
#pragma unroll
        for (int row = 0; row < kSoftmaxRows; ++row) {
          const float sum = row_sum[row];
          const float inv_sum = (sum == 0.f || sum != sum) ? 0.f : 1 / sum;
#pragma unroll
          for (int col = 0; col < decltype(size<1>(tCrO_rc))::value; ++col)
            tCrO_rc(row, col) *= inv_sum;
        }
      }
    }

    // smooth_v epilogue: add the per-(b, hkv) V column mean back (the
    // persist_d derivation; vm factors out of the column sum and cancels
    // against the row normalization above). Per v_chunk the [kBr,
    // kVDChunk] identity partition maps each C-fragment slot onto its d
    // column within the chunk; the m4n2 N-warp split is handled by the
    // partition itself (each N-warp holds its own d half).
    if (vm != nullptr) {
      const float* vm_bh = vm + static_cast<long>(kv_bh) * kHeadDim;
      auto cO_vm = make_identity_tensor(Shape<Int<kBr>, Int<kVDChunk>>{});
      auto tOcO_vm = thread_mma_pv.partition_C(cO_vm);
#pragma unroll
      for (int v_chunk = 0; v_chunk < kDChunksV; ++v_chunk) {
        auto tCrO = make_tensor(make_rmem_ptr(&o_acc_storage[v_chunk][0]),
                                OFragLayout{});
        const float* vm_c = vm_bh + v_chunk * kVDChunk;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tCrO); ++i)
          tCrO(i) += vm_c[cute::get<1>(tOcO_vm(i))];
      }
    }

    // Epilogue (split_d ordering): qkm dot before O staging, batched
    // R->S(STSM)->TMA over the freed smem, tail tiles R->G, lse written
    // by n_warp==0 only.
    float qkm[kSRows];
    const bool smooth_lse =
        (softmax_lse != nullptr) && (km != nullptr) && (qm != nullptr);
    {
      __syncthreads();

      if (smooth_lse) {
        const float* km_bh = km + static_cast<long>(kv_bh) * kHeadDim;
        // qm blocks are 128-row; attention tiles are 64-row (2 tiles per
        // block, tail-aligned because Nq_pad stays 128-aligned).
        const long qm_mb = (long)Nq_pad / 128;
        const int qm_blk_idx = (q_start_row + Q_tile_id * kBr) / 128;
        const float* qm_blk =
            qm + (static_cast<long>(q_bh) * qm_mb + qm_blk_idx) * kHeadDim;
        lse_qkm_dot<kHeadDim, kSRows>(sQ, sSFQ, tScS_rc, km_bh, qm_blk, qkm);
        __syncthreads();
      }

      constexpr int kVChunksPerBatch = Traits::kVChunksPerBatch;
      constexpr int kNBatches = Traits::kNBatches;
      constexpr int kOTileElems = cosize(SmemLayoutO{});

      auto r2s_copy = make_tiled_copy_C(
          Copy_Atom<SM90_U32x2_STSM_N, ElementO>{}, tiled_mma_pv);
      auto r2s_thr = r2s_copy.get_thread_slice(tid);

      // NHD (diffusers BNHD packed) O: rows interleave heads (row stride
      // Nh*kHeadDim); the nhd_out branch only picks coordinates, the
      // batched R->S->TMA copy path is shared (per-work, like fp4
      // persist-D). Column tiles fold the head in (v_chunk stays local).
      const int o_row_base =
          nhd_out ? (Nb_id * Nq + q_start_row) : O_row_offset;
      const int o_rows = nhd_out ? (Nb * Nq) : total_q_rows;
      const int o_cols = nhd_out ? (Nh * kHeadDim) : kHeadDim;
      const int o_col_tile = nhd_out ? (Nh_id * kDChunksV) : 0;
      auto mO_tma =
          domain_offset(make_coord(o_row_base, 0),
                        tma_o.get_tma_tensor(make_shape(o_rows, o_cols)));
      auto o_slice = tma_o.get_slice(_0{});

      if (Br_base + kBr <= Nq - q_start_row) {
#pragma unroll
        for (int batch = 0; batch < kNBatches; ++batch) {
#pragma unroll
          for (int v_in = 0; v_in < kVChunksPerBatch; ++v_in) {
            const int v_chunk = batch * kVChunksPerBatch + v_in;
            auto tCrO = make_tensor(make_rmem_ptr(&o_acc_storage[v_chunk][0]),
                                    OFragLayout{});
            auto tCrOHalf = ffpa_cute::convert_type<ElementO>(tCrO);
            auto sO_v =
                make_tensor(make_smem_ptr(reinterpret_cast<ElementO*>(shm) +
                                          v_in * kOTileElems),
                            SmemLayoutO{});
            auto tCrOHalf_src = r2s_thr.retile_S(tCrOHalf);
            auto tCsO_dst = r2s_thr.partition_D(sO_v);
            copy(r2s_copy, tCrOHalf_src, tCsO_dst);
          }
          cutlass::arch::fence_view_async_shared();
          __syncthreads();
#pragma unroll
          for (int v_in = 0; v_in < kVChunksPerBatch; ++v_in) {
            const int v_chunk = batch * kVChunksPerBatch + v_in;
            auto sO_v =
                make_tensor(make_smem_ptr(reinterpret_cast<ElementO*>(shm) +
                                          v_in * kOTileElems),
                            SmemLayoutO{});
            auto gO_tma =
                local_tile(mO_tma, Shape<Int<kBr>, Int<kVDChunk>>{},
                           make_coord(Q_tile_id, o_col_tile + v_chunk));
            auto tCgO_tma = o_slice.partition_D(gO_tma);
            auto tOsO = o_slice.partition_S(sO_v);
            if (tid == 0)
              copy(tma_o, tOsO, tCgO_tma);
          }
          tma_store_arrive();
          if (batch < kNBatches - 1) {
            tma_store_wait<0>();
            __syncthreads();
          }
        }
      } else {
        const int O_gmem_offset =
            nhd_out ? ((Nb_id * Nq + q_start_row) * Nh + Nh_id) * kHeadDim
                    : (q_bh)*Nq * kHeadDim + q_start_row * kHeadDim;
        const int o_row_stride = nhd_out ? Nh * kHeadDim : kHeadDim;
        auto mO = make_tensor(make_gmem_ptr(O + O_gmem_offset),
                              make_shape(Nq - q_start_row, Int<kHeadDim>{}),
                              make_stride(o_row_stride, _1{}));
        auto cO = make_identity_tensor(Shape<Int<kBr>, Int<kVDChunk>>{});
        auto tOcO = thread_mma_pv.partition_C(cO);
#pragma unroll
        for (int v_chunk = 0; v_chunk < kDChunksV; ++v_chunk) {
          auto tCrO = make_tensor(make_rmem_ptr(&o_acc_storage[v_chunk][0]),
                                  OFragLayout{});
          auto tCrOHalf = ffpa_cute::convert_type<ElementO>(tCrO);
          auto gO = local_tile(mO, Shape<Int<kBr>, Int<kVDChunk>>{},
                               make_coord(Q_tile_id, v_chunk));
          auto tCgO = thread_mma_pv.partition_C(gO);
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(tCrOHalf); ++i) {
            const int global_row = Br_base + cute::get<0>(tOcO(i));
            if (global_row < Nq - q_start_row)
              tCgO(i) = tCrOHalf(i);
          }
        }
      }

      if (softmax_lse != nullptr && n_warp == 0) {
        const int lse_base = Nb_id * Nh * Nq + Nh_id * Nq;
        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < kSRows; ++row) {
          // row_max is log2-domain, row_sum is P2 = P*2688 domain (see
          // online_softmax_p2_m4n2): fold the 2688 back out of the lse.
          float lse = (row_max[row] + log2f(row_sum[row]) +
                       SoftmaxFused<kSoftmaxRows>::fp8_scalexfp4_scale_log2) *
                      FFPA_M_LN2;
          if (smooth_lse)
            lse += scale_orig * qkm[row];
          const int global_row =
              q_start_row + Br_base + cute::get<0>(tScS_rc(row, 0));
          if (global_row < Nq)
            softmax_lse[lse_base + global_row] = lse;
        }
      }
    }

    if (Br_base + kBr <= Nq - q_start_row)
      tma_store_wait<0>();

    gK += work_chunks_qk;
    gV += work_chunks_v;

    CtaBarrier::arrive(&epilogue_done);
  }
#endif  // defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
}

}  // namespace ffpa_fp4

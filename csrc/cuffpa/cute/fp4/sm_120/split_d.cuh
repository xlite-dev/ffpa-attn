// NVFP4 split-D forward kernel for sm_120, headdims in (256, 768)
// (64-multiples): the regime where the persist-D smem plan (full [kBc, D]
// K/V^T tiles per stage) no longer fits the 99KB opt-in budget.
//
// This is the persist_d kernel's pipeline with one structural change, not a
// port of the fp8 split_d kernel: K and V^T stream through smem in
// 64-element D chunks while everything persist-d validated is kept -
//   * persistent work loop (grid-strided works, barriers never re-init,
//     dense grid = min(total_work, SMs), causal = one CTA per work);
//   * SFQ loaded once per work into a register fragment (extent-aware SFA
//     partitioning covers all D chunks); Q is smem-resident with its
//     64-wide data slice copied to registers per D chunk - only the B
//     operands stream from gmem. O staging aliases q_base behind
//     epilogue_done;
//   * delta_s rank-1 preload into the QK accumulator (DS rides the kv
//     tile's first K chunk barrier);
//   * lazy rescale (warp-vote scores_scale!=1 skip) and register P
//     quantization (M8N1: each warp owns full kBc columns, so P never
//     touches smem);
//   * ftz exp2 / group-absmax fused softmax (fp4_pscale.cuh, shared).
// The warp specialization is dropped on purpose: non-WS 256T with tid==0
// issuing TMA inline (fp8 split_d's issue pattern). The O accumulator
// alone is kBr*D/256 = D/2 f32 regs/thread - already past the 255 wall at
// D=512 - so a 128T producer split and its setmaxnreg 232 cap would only
// add spill; fp8's split-d family made the same call.
//
// Split-D mechanics (fp8 split_d.cuh is the reference for this part only):
// D never enters the grid. Stage/phase come from per-type GLOBAL chunk
// counters (gK, gV) that advance across works: chunk_index =
// work_base + kv_tile*kDChunks + chunk; stage = chunk_index % kStages;
// phase = (chunk_index / kStages) & 1. tid==0 keeps a kStages-deep
// lookahead: after the block consumes chunk G it issues G+kStages (bound
// by the work's chunk count), so during a work's epilogue NO K/V TMA is
// in flight and the O epilogue may stage over the whole freed smem. The
// next work's Q TMA + initial fills wait on epilogue_done (WAR on q_base).
//
// Same subbyte / masking / lse contracts as persist_d.cuh: e2m1 smem via
// make_smem_ptr<Element>(void*), causal/kv-tail masks evaluate token
// positions through kv_perm32, lse = (m*L + log2(row_sum) + log2(1/2688))
// * ln2 + scale*qkm with the qkm dot read back from the resident Q smem
// before O staging overwrites it.
#pragma once

#include <cuda_fp8.h>

#include <cute/tensor.hpp>
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
#include "../attn_traits.cuh"
#include "../cute_ext.h"
#include "../fp4_gemm.cuh"
#include "../fp4_pscale.cuh"

namespace ffpa_fp4 {

using TmaBarrier = cutlass::arch::ClusterTransactionBarrier;
using CtaBarrier = cutlass::arch::ClusterBarrier;

template <typename Traits, typename ElementO, typename TmaQ, typename TmaK,
          typename TmaV, typename TmaO, typename TmaSFQ, typename TmaSFK,
          typename TmaSFVt, typename TmaDS, int kHasAttnBias = 0>
__global__ void __launch_bounds__(Traits::kNumThreads, 1)
    split_d_fwd_cute_fp4_sm120(
        CUTLASS_GRID_CONSTANT TmaQ const tma_q,
        CUTLASS_GRID_CONSTANT TmaK const tma_k,
        CUTLASS_GRID_CONSTANT TmaV const tma_v,
        CUTLASS_GRID_CONSTANT TmaO const tma_o,
        CUTLASS_GRID_CONSTANT TmaSFQ const tma_sfq,
        CUTLASS_GRID_CONSTANT TmaSFK const tma_sfk,
        CUTLASS_GRID_CONSTANT TmaSFVt const tma_sfvt,
        CUTLASS_GRID_CONSTANT TmaDS const tma_ds, ElementO* __restrict__ O,
        float* __restrict__ softmax_lse, const float* __restrict__ km,
        const float* __restrict__ qm, const float* __restrict__ vm, int Nq,
        int Nkv, int Nq_pad, int Nkv_pad, int Nh, int Nh_kv, float scale,
        int Tc, int causal, int total_q_rows, int Nb, int q_start_row = 0,
        bool nhd_out = false, const void* __restrict__ attn_bias = nullptr,
        int attn_bias_dtype = 0, long long attn_bias_stride_b = 0,
        long long attn_bias_stride_h = 0, long long attn_bias_stride_m = 0,
        long long attn_bias_stride_n = 0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  using namespace cute;
  using cute::tma_store_arrive;
  using cute::tma_store_wait;
  using Element = typename Traits::Element;
  using ElementSF = typename Traits::ElementSF;
  using ElementPV = typename Traits::ElementPV;
  using ElementSFV = typename Traits::ElementSFV;
  using SmemLayoutQ = typename Traits::SmemLayoutQ;
  using SmemLayoutK = typename Traits::SmemLayoutK;
  using SmemLayoutVt = typename Traits::SmemLayoutVt;
  using SmemLayoutSFQ = typename Traits::SmemLayoutSFQ;
  using SmemLayoutSFK = typename Traits::SmemLayoutSFK;
  using SmemLayoutSFVt = typename Traits::SmemLayoutSFVt;
  using SmemLayoutDS = typename Traits::SmemLayoutDS;
  using SmemLayoutO = typename Traits::SmemLayoutO;
  using TiledMmaQK = typename Traits::TiledMmaQK;
  using TiledMmaPV = typename Traits::TiledMmaPV;
  using SmemCopyAtomQ = typename Traits::SmemCopyAtomQ;
  using SmemCopyAtomKV = typename Traits::SmemCopyAtomKV;
  using SmemCopyAtomV = typename Traits::SmemCopyAtomV;
  using SmemCopyAtomSF = typename Traits::SmemCopyAtomSF;
  using SmemCopyAtomSFV = typename Traits::SmemCopyAtomSFV;
  using BlkScaledConfig = typename Traits::BlkScaledConfig;
  using BlkScaledConfigV = typename Traits::BlkScaledConfigV;
  constexpr bool kPvMxfp8 = Traits::kPvMxfp8;

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
  (void)kOffSFVt;

  const int group_size = Nh / Nh_kv;
  const int tid = threadIdx.x;

  // Work decomposition: Mb tiles per (b, h), grid-strided over all works
  // (same contract as persist_d; the grid alone picks persistent vs
  // block-per-work).
  const int MB = (Nq - q_start_row + kBr - 1) / kBr;
  const int total_work = MB * Nb * Nh;

  extern __shared__ __align__(1024) char shm[];

  // Barrier inventory (initialized once, never re-init):
  //   q_full        TMA tx barrier, resident Q+SFQ of the current work
  //   k_full[s]     TMA tx barrier, K+SFK chunk stage s (+DS on the kv
  //                 tile's first chunk)
  //   v_full[s]     TMA tx barrier, V^T+SFVt chunk stage s
  //   k/v_empty[s]  256-arrival "stage consumed" (the gemm tails)
  //   epilogue_done WAR fence: O staging aliases q_base, the next work's
  //                 Q TMA waits for the previous epilogue to retire
  __shared__ uint64_t q_full;
  __shared__ uint64_t k_full[kStagesQK];
  __shared__ uint64_t k_empty[kStagesQK];
  __shared__ uint64_t v_full[kStagesPV];
  __shared__ uint64_t v_empty[kStagesPV];
  __shared__ uint64_t epilogue_done;

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
  }
  __syncthreads();

  // Work-independent gmem base tensors (descriptor spaces mirror the
  // launcher): Q/K are (Nb*H*_pad, D) row planes, V^T is
  // (Nb*Hkv*D, Nkv_pad), SF uses the BlockScaledConfig atom layouts.
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
  auto layout_SFVt = BlkScaledConfigV::tile_atom_to_shape_SFVt(
      make_shape(Int<kHeadDim>{}, Nkv_pad, Nh_kv, Nb));
  auto layout_DS = tile_to_shape(typename Traits::SmemLayoutAtomDS{},
                                 make_shape(Nq_pad, Nkv_pad, Nh, Nb),
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
  auto sV = make_tensor(make_smem_ptr<ElementPV>(shm + kOffV), SmemLayoutVt{});
  auto sSFVt =
      make_tensor(make_smem_ptr<ElementSFV>(shm + kOffSFVt), SmemLayoutSFVt{});

  auto tQsQ = q_slice.partition_D(sQ);
  auto tQsSFQ = sfq_slice.partition_D(sSFQ);

  // TMA issue helpers (tid==0 only). Per-chunk gmem views are rebuilt per
  // call from the base tensors - cheap layout arithmetic, and it keeps the
  // work-loop state down to the two global chunk counters. Q+SFQ share the
  // q_full tx barrier; DS rides the K chunk-0 barrier (its expect_tx adds
  // kTxBytesDS on top of kTxBytesK and its TMA copy is issued together
  // with the chunk-0 K load).
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
      auto gDS = local_tile(mDS(_, _, Nh_id, b), Shape<Int<kBr>, Int<kBc>>{},
                            make_coord(q_tile_abs, kv_tile));
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
        make_smem_ptr<ElementPV>(shm + kOffV + stage * Traits::kVBytesStage),
        typename Traits::SmemLayoutVtStage{});
    auto sSFVt_st =
        make_tensor(make_smem_ptr<ElementSFV>(shm + kOffSFVt +
                                              stage * Traits::kSFVtBytesStage),
                    typename Traits::SmemLayoutSFVtStage{});
    cutlass::arch::fence_view_async_shared();
    TmaBarrier::arrive_and_expect_tx(&v_full[stage], Traits::kTxBytesV);
    copy(tma_v.with(v_full[stage]), v_slice.partition_S(gV),
         v_slice.partition_D(sV_st));
    copy(tma_sfvt.with(v_full[stage]), sfvt_slice.partition_S(gSFVt),
         sfvt_slice.partition_D(sSFVt_st));
  };

  // Consumer-side setup (all threads).
  for (int s = 0; s < kStagesQK; ++s)
    CtaBarrier::arrive(&k_empty[s]);
  for (int s = 0; s < kStagesPV; ++s)
    CtaBarrier::arrive(&v_empty[s]);

  TiledMmaQK tiled_mma_qk;
  TiledMmaPV tiled_mma_pv;
  auto thread_mma_qk = tiled_mma_qk.get_thread_slice(tid);
  auto thread_mma_pv = tiled_mma_pv.get_thread_slice(tid);

  // Register fragments for both mmas: the chunk TiledMmaQK (Tile-K=64)
  // partitions A over the tile extent only, so Q's data half is copied
  // per chunk from the resident smem inside the kv loop (a work-constant
  // register preload only exists for the SF half, whose SFA partition is
  // tensor-extent-aware). tOrP/tOrSFP are built on LayoutP/LayoutSFP
  // (traits), B fragments are per-chunk.
  Tensor tSrK = thread_mma_qk.partition_fragment_B(sK(_, _, Int<0>{}));
  Tensor tOrVt = thread_mma_pv.partition_fragment_B(sV(_, _, Int<0>{}));
  Tensor tSrSFQ = partition_fragment_SFA(sSFQ, thread_mma_qk);
  Tensor tSrSFK = partition_fragment_SFB(sSFK(_, _, Int<0>{}), thread_mma_qk);
  Tensor tOrSFVt = partition_fragment_SFB(sSFVt(_, _, Int<0>{}), thread_mma_pv);
  Tensor tOrP = make_tensor_like<ElementPV>(typename Traits::LayoutP{});
  Tensor tOrSFP = make_tensor<ElementSFV>(typename Traits::LayoutSFP{});

  auto smem_tiled_copy_Q = make_tiled_copy_A(SmemCopyAtomQ{}, tiled_mma_qk);
  auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tid);
  Tensor tSsQ =
      smem_thr_copy_Q.partition_S(as_position_independent_swizzle_tensor(sQ));

  auto smem_tiled_copy_K = make_tiled_copy_B(SmemCopyAtomKV{}, tiled_mma_qk);
  auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tid);
  Tensor tSsK =
      smem_thr_copy_K.partition_S(as_position_independent_swizzle_tensor(sK));

  auto smem_tiled_copy_V = make_tiled_copy_B(SmemCopyAtomV{}, tiled_mma_pv);
  auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tid);
  Tensor tOsVt =
      smem_thr_copy_V.partition_S(as_position_independent_swizzle_tensor(sV));

  auto tile_shape_mnk = tile_shape(tiled_mma_qk);
  auto smem_tiled_copy_SFQ = make_tiled_copy_impl(
      SmemCopyAtomSF{}, get_layoutSFA_TV(tiled_mma_qk),
      make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk)));
  auto smem_thr_copy_SFQ = smem_tiled_copy_SFQ.get_thread_slice(tid);
  Tensor tSsSFQ = smem_thr_copy_SFQ.partition_S(
      as_position_independent_swizzle_tensor(sSFQ));
  Tensor tSrSFQ_copy_view = smem_thr_copy_SFQ.retile_D(tSrSFQ);

  auto smem_tiled_copy_SFK = make_tiled_copy_impl(
      SmemCopyAtomSF{}, get_layoutSFB_TV(tiled_mma_qk),
      make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
  auto smem_thr_copy_SFK = smem_tiled_copy_SFK.get_thread_slice(tid);
  Tensor tSsSFK = smem_thr_copy_SFK.partition_S(
      as_position_independent_swizzle_tensor(sSFK));

  auto smem_tiled_copy_SFV =
      make_tiled_copy_impl(SmemCopyAtomSFV{}, get_layoutSFB_TV(tiled_mma_pv),
                           make_shape(size<1>(tile_shape(tiled_mma_pv)),
                                      size<2>(tile_shape(tiled_mma_pv))));
  auto smem_thr_copy_SFV = smem_tiled_copy_SFV.get_thread_slice(tid);
  Tensor tOsSFVt = smem_thr_copy_SFV.partition_S(
      as_position_independent_swizzle_tensor(sSFVt));

  // The QK accumulator fragment views (raw / conversion / reduction) - the
  // S tile is [kBr, kBc] regardless of D chunking, so all persist_d
  // softmax/masking/packing code carries over verbatim.
  Tensor tSrS = partition_fragment_C(tiled_mma_qk, Shape<Int<kBr>, Int<kBc>>{});
  Tensor tSrS_conversion_view =
      make_tensor(tSrS.data(), convert_to_conversion_layout(tSrS.layout()));
  Tensor tSrS_reduction_view =
      make_tensor(tSrS.data(), convert_to_reduction_layout(tSrS.layout()));
  // Per-token-group absmax of the P-domain scores; softmax fills it, the
  // packer turns it into the SFP operand (persist_d verbatim): 16-token
  // groups from the conversion view (NVFP4) or 32-token groups = one
  // mma-k32 block from the reduction view (MXFP8).
  auto AbsMaxP = [&]() {
    if constexpr (kPvMxfp8)
      return make_tensor_like<float>(make_layout(make_shape(
          size<0>(tSrS_reduction_view), size<1, 1>(tSrS_reduction_view))));
    else
      return make_tensor_like<float>(make_layout(shape(group<1, 4>(
          flatten(tSrS_conversion_view.layout()(make_coord(_0{}, _), _, _))))));
  }();

  auto cS = make_identity_tensor(Shape<Int<kBr>, Int<kBc>>{});
  auto tScS = thread_mma_qk.partition_C(cS);
  auto tScS_rc =
      make_tensor(tScS.data(), convert_to_reduction_layout(tScS.layout()));
  constexpr int kSRows = decltype(size<0>(tScS_rc))::value;
  constexpr int kSCols = decltype(size<1>(tScS_rc))::value;

  // O accumulator: per-v_chunk resident fragments (D/2 f32 regs/thread).
  using OFragType = decltype(partition_fragment_C(
      tiled_mma_pv, Shape<Int<kBr>, Int<kVDChunk>>{}));
  using OFragLayout = typename OFragType::layout_type;
  constexpr int kOElemsPerFrag = decltype(size(OFragType{}))::value;
  float o_acc_storage[kDChunksV][kOElemsPerFrag];

  constexpr int kSoftmaxRows = 2 * (2 * kBr / kNumThreads);
  static_assert(kSoftmaxRows == kSRows, "softmax/O row mismatch");
  std::conditional_t<kPvMxfp8, SoftmaxFusedMxfp8<kSoftmaxRows>,
                     SoftmaxFused<kSoftmaxRows>>
      softmax_fused;
  const float scale_orig = scale;
  const float softmax_scale_log2 = scale * FFPA_M_LOG2E;

  // delta_s rank-1 preload (SA3 float4 slot math); the assign doubles as
  // the per-tile acc clear.
  auto add_delta_s = [&](auto& acc, int stage) {
    auto tSsDS_stage = recast<float4>(sDS(_, _, stage));
    auto acc_float4 = recast<float4>(acc);
    int quad_id = (threadIdx.x % 4) * 2;
    for (int i = 0; i < 4; i++) {
      auto num = quad_id + i * 8;
      float4 delta_s_0 =
          tSsDS_stage(make_coord(_0{}, _0{}), make_coord(num, _0{}));
      float4 delta_s_1 =
          tSsDS_stage(make_coord(_0{}, _0{}), make_coord(num + 1, _0{}));
      acc_float4(make_coord(make_coord(_0{}, _0{}), _0{}), _0{}, i) = delta_s_0;
      acc_float4(make_coord(make_coord(_0{}, _1{}), _0{}), _0{}, i) = delta_s_0;
      acc_float4(make_coord(make_coord(_0{}, _0{}), _1{}), _0{}, i) = delta_s_1;
      acc_float4(make_coord(make_coord(_0{}, _1{}), _1{}), _0{}, i) = delta_s_1;
    }
  };

  // Initial fill for work 0 (later works fill at their loop head, after
  // the previous epilogue's epilogue_done).
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
  }
  TmaBarrier::wait(&q_full, 0);
  cutlass::arch::fence_view_async_shared();

  int gK = 0;
  int gV = 0;
  int w = 0;
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
        // WAR on q_base (and the K/V stage regions the epilogue staged O
        // over): wait for the previous epilogue to retire, then issue this
        // work's Q and the initial K/V chunk fills.
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
      }
      TmaBarrier::wait(&q_full, w & 1);
      cutlass::arch::fence_view_async_shared();
    }

    // SFQ is a per-work register constant (tensor-extent-aware SFA
    // partition spans all D chunks); the Q data half copies per chunk in
    // the kv loop. Without the explicit copy the SFA asm operands stay
    // uninitialized and cicc folds them to 0 (the persist_d trap).
    copy(smem_tiled_copy_SFQ, tSsSFQ, tSrSFQ_copy_view);

#pragma unroll
    for (int v = 0; v < kDChunksV; ++v)
#pragma unroll
      for (int i = 0; i < kOElemsPerFrag; ++i)
        o_acc_storage[v][i] = 0.0f;

    const int g0k = gK;
    const int g0v = gV;

#pragma unroll 1
    for (int kv_tile = 0; kv_tile < Tc_eff; ++kv_tile) {
      // Phase 1: QK over D chunks, accumulating one [kBr, kBc] S tile.
      // Unrolled: kDChunksQK is compile-time and the register fragments
      // (tSrQ chunk slices) need static indexing to stay out of local.
#pragma unroll
      for (int d_chunk = 0; d_chunk < kDChunksQK; ++d_chunk) {
        const int seq = g0k + kv_tile * kDChunksQK + d_chunk;
        const int k_stg = seq % kStagesQK;
        const int k_phase = (seq / kStagesQK) & 1;
        TmaBarrier::wait(&k_full[k_stg], k_phase);
        cutlass::arch::fence_view_async_shared();

        if (d_chunk == 0)
          add_delta_s(tSrS, k_stg);
        // A chunk: copy Q's 64-wide slice from the resident smem (no
        // barrier - Q is work-constant) and pair it with the matching SFQ
        // register slice; B chunk streams from the stage. src/dst both
        // partition the chunk tensor so their modes stay congruent.
        auto sQ_chunk = local_tile(sQ, Shape<Int<kBr>, Int<kQKDChunk>>{},
                                   make_coord(_0{}, d_chunk));
        Tensor tSrQ_c = thread_mma_qk.partition_fragment_A(sQ_chunk);
        Tensor tSrQ_c_view = smem_thr_copy_Q.retile_D(tSrQ_c);
        Tensor tSsQ_chunk = smem_thr_copy_Q.partition_S(
            as_position_independent_swizzle_tensor(sQ_chunk));
        copy(smem_tiled_copy_Q, tSsQ_chunk, tSrQ_c_view);
        auto tSrSFQ_c = tSrSFQ(_, _, d_chunk);
        gemm_ss_chunk_fp4(tSrS, tSrQ_c, tSrSFQ_c, tSrK, tSrSFK,
                          tSsK(_, _, _0{}, k_stg), tSsSFK(_, _, _0{}, k_stg),
                          tiled_mma_qk, smem_tiled_copy_K, smem_thr_copy_K,
                          smem_tiled_copy_SFK, smem_thr_copy_SFK, k_empty,
                          k_stg);

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

      // Masking: kv-tail + causal, token positions through kv_perm32
      // (persist_d verbatim).
      {
        auto scores = make_tensor(tSrS.data(),
                                  convert_to_reduction_layout(tSrS.layout()));
        // Additive attn bias in the dequantized score domain; the fused
        // softmax below applies softmax_scale_log2, so bias/scale_orig
        // lands as +bias in softmax-input units. The -INFINITY assignments
        // in the masking below simply override it.
        if constexpr (kHasAttnBias) {
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

      // Online softmax fused with P-quant prep (persist_d verbatim).
      if (kv_tile == 0)
        softmax_fused.template online_softmax_with_quant</*FirstTile=*/true,
                                                         /*InfCheck=*/true>(
            tSrS, AbsMaxP, softmax_scale_log2);
      else
        softmax_fused.template online_softmax_with_quant</*FirstTile=*/false,
                                                         /*InfCheck=*/true>(
            tSrS, AbsMaxP, softmax_scale_log2);

      // Lazy rescale: scores_scale == 1.0f exactly when the row max did
      // not move this tile (~96% of dense tiles). Pre-scale the resident O
      // chunks in place; the gemm then accumulates on top.
      const bool need_rescale =
          kv_tile > 0 &&
          __any_sync(0xffffffff, softmax_fused.scores_scale[0] != 1.0f ||
                                     softmax_fused.scores_scale[1] != 1.0f);
      if (need_rescale) {
#pragma unroll
        for (int v = 0; v < kDChunksV; ++v) {
          auto tCrO =
              make_tensor(make_rmem_ptr(&o_acc_storage[v][0]), OFragLayout{});
          softmax_fused.rescale_acc(tCrO);
        }
      }

      // Phase 3: PV over the D chunks of V^T (unrolled, same reason as
      // the QK chunk loop).
#pragma unroll
      for (int v_chunk = 0; v_chunk < kDChunksV; ++v_chunk) {
        const int seq = g0v + kv_tile * kDChunksV + v_chunk;
        const int v_stg = seq % kStagesPV;
        const int v_phase = (seq / kStagesPV) & 1;
        TmaBarrier::wait(&v_full[v_stg], v_phase);
        cutlass::arch::fence_view_async_shared();

        auto tCrO = make_tensor(make_rmem_ptr(&o_acc_storage[v_chunk][0]),
                                OFragLayout{});
        if constexpr (kPvMxfp8)
          gemm_rs_mxfp8(tCrO, tOrP, tOrSFP, tOrVt, tOrSFVt, tOsVt, tOsSFVt,
                        tiled_mma_pv, smem_tiled_copy_V, smem_thr_copy_V,
                        smem_tiled_copy_SFV, smem_thr_copy_SFV, AbsMaxP,
                        tSrS_reduction_view, v_empty, v_stg, tid & 31);
        else
          gemm_rs_fp4(tCrO, tOrP, tOrSFP, tOrVt, tOrSFVt, tOsVt, tOsSFVt,
                      tiled_mma_pv, smem_tiled_copy_V, smem_thr_copy_V,
                      smem_tiled_copy_SFV, smem_thr_copy_SFV, AbsMaxP,
                      tSrS_conversion_view, v_empty, v_stg);

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

    // Fold row_sum on chunk 0 (finalize) and scale the remaining chunks.
    {
      auto tCrO0 =
          make_tensor(make_rmem_ptr(&o_acc_storage[0][0]), OFragLayout{});
      softmax_fused.finalize(tCrO0);
#pragma unroll
      for (int v = 1; v < kDChunksV; ++v) {
        auto tCrO =
            make_tensor(make_rmem_ptr(&o_acc_storage[v][0]), OFragLayout{});
        softmax_fused.scale_o(tCrO);
      }
    }

    // smooth_v epilogue: add the per-(b, hkv) V column mean back (the
    // persist_d derivation; vm factors out of the column sum and cancels
    // against the row normalization above). Per v_chunk the [kBr,
    // kVDChunk] identity partition maps each C-fragment slot onto its d
    // column within the chunk.
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

    // Epilogue (persist_d ordering): qkm dot from the resident Q smem
    // BEFORE O staging overwrites it, then batched R->S(STSM)->TMA over
    // the freed smem (no K/V TMA is in flight during the epilogue - the
    // lookahead is work-bounded), tail tiles store R->G with a row guard.
    float qkm[kSRows];
    const bool smooth_lse =
        (softmax_lse != nullptr) && (km != nullptr) && (qm != nullptr);
    {
      __syncthreads();

      if (smooth_lse) {
        const float* km_bh = km + static_cast<long>(kv_bh) * kHeadDim;
        const long qm_mb = Nq_pad / kBr;
        const float* qm_blk =
            qm + (static_cast<long>(q_bh) * qm_mb + q_tile_abs) * kHeadDim;
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
        // Tail tile: rows past Nq would alias the next head in the
        // flattened [total_q_rows, D] TMA space, so store R->G per chunk.
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

      if (softmax_lse != nullptr) {
        const int lse_base = Nb_id * Nh * Nq + Nh_id * Nq;
        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < kSRows; ++row) {
          // row_sum lives in a scaled P domain: lse = scale*m +
          // ln(row_sum / domain_scale). NVFP4: P*2688 (fp8_scalexfp4_
          // scale_log2 = log2(1/2688)); MXFP8: P*448 (row_sum = sum of
          // q*SF with q in the e4m3 domain, see SoftmaxFusedMxfp8), so
          // the correction is log2(1/448) = -e4m3_full_log2.
          float lse =
              (softmax_fused.row_max[row] * softmax_scale_log2 +
               log2f(softmax_fused.row_sum[row]) +
               (kPvMxfp8
                    ? -SoftmaxFusedMxfp8<kSoftmaxRows>::e4m3_full_log2
                    : SoftmaxFused<kSoftmaxRows>::fp8_scalexfp4_scale_log2)) *
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

    // Release q_base (and the staged O smem) for the next work's Q TMA.
    CtaBarrier::arrive(&epilogue_done);
  }
#endif  // defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
}

}  // namespace ffpa_fp4

// NVFP4 (e2m1 data + ue4m3 block scale) persist-D warp-specialized kernel for
// sm_120, ported from SageAttention3's NVFP4 data path onto the ffpa fp8
// persist_d producer/consumer skeleton (128T producer + 256T consumer,
// hand-rolled TMA barriers, consumer-side epilogue).
//
// Math chain (per (b,h), Q block of 128 rows):
//   qm  = mean(q, 128-row group)   Qhat = q - qm   (quantized, sub_qm)
//   km  = mean(k)                  Khat = k - km   (quantized, sub_km,
//                                    rows permuted)
//   S   = Qhat @ Khat^T + delta_s,  delta_s[b,h,mb,n] = qm @ (k - km)^T:
//       S = (q - qm)(k - km)^T + qm(k - km)^T = q(k - km)^T
//   P   = softmax(S * scale)              O = P @ v
// Smoothing K leaves O unchanged (softmax shift invariance); the lse must add
// back scale * dot(q_row, km) = scale * (dot(Qhat_row, km) + dot(qm, km)).
//
// Column alignment (empirically locked against sageattn3 in
// .tmp/fp4-persist-d/test_ds_align.py): K/V^T workspaces store tokens with
// the 32-row interleave permutation, so the QK C-fragment's logical column j
// carries the score of original token kv_perm32(j). The SA3 fragment
// adapters (add_delta_s slot arithmetic, LayoutP/LayoutSFP packing, the V^T
// trans storage) all compensate consistently - copied verbatim they form a
// self-consistent system (dense e2e matches the dequantized simulation).
// Only the masking code must be perm-aware: the causal/kv-tail predicates
// evaluate the token position as kv_tile*kBc + kv_perm32(col). Upstream SA3
// masks on the raw column index, which breaks causal attention (max_abs 3.3
// vs SDPA at N=512); this kernel fixes that. kv_perm32 is a bijection on
// every 32-column window, so tile-level mask skipping (mask_start_tile,
// Tc_eff) keeps the unpermuted formulas.
//
// P quantization is two-level (fp4_pscale.cuh): the 1/(448*6) global constant
// is folded into the exp2 shift, the per-16-column group scale SFP (ue4m3)
// is consumed by the blockscaled PV mma. Fully-masked groups degenerate to
// P=0 with SFP=0 (the absmax clamp avoids a 0/0 NaN), contributing nothing.
//
// O epilogue: SM90_U32x2_STSM_N into SW128 smem staged over the freed Q/K
// smem, then one TMA store (SA3 layout, not fp8's U32x4 - the blockscaled
// PV C-fragment differs). Tail Q tiles store R->G with a row guard.
//
// Subbyte pitfall (caused OOB TMA writes at stage >= 1): e2m1 smem tensors
// must be built via make_smem_ptr<Element>(void*) - that overload wraps a
// subbyte_iterator so tensor slicing advances in bits. Wrapping a raw
// reinterpret_cast<Element*> scales offsets by sizeof==1B (2x for 4-bit
// elements) and walks off the smem window. fp8/fp16 paths never hit this
// because their elements are >= 1 byte.
//
// Reference (NVFP4 data path):
// https://github.com/thu-ml/SageAttention/tree/main/sageattention3_blackwell/sageattn3/blackwell
//   (kernel_ws.h / mainloop_tma_ws.h / epilogue_tma_ws.h: the warp-specialized
//    NVFP4 kernel whose fragment adapters this port copies verbatim)
#pragma once

#include <cuda_fp8.h>

#include <cute/tensor.hpp>
#include <cute/tensor_zip.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
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

// Traits live in ../attn_traits.cuh (FFPAAttnCuTePersistDFP4Traits, renamed
// from Fp4PersistDTraits to match the fp8/fp16 naming family).

// kv_perm32 and lse_qkm_dot live in ../fp4_gemm.cuh and ../fp4_pscale.cuh.

// NVFP4 persist-D forward. Grid-scheduling contract (ONE kernel, ONE code
// path): the body is a strided work loop
//     for (work_id = blockIdx.x; work_id < total_work; work_id += gridDim.x)
// over total_work = Mb * Nb * Nh works (bh-outer / Q-tile-inner), so the
// runtime grid alone selects the execution style - there is no separate
// persistent vs non-persistent kernel variant:
//   * persistent:   gridDim.x = min(total_work, num_SMs). Each CTA stays
//     resident on its SM and iterates the loop ~total_work/gridDim.x times.
//     The producer can prefetch the next work's K/V while the consumer runs
//     the current epilogue, so pipeline fill/drain amortize once per CTA
//     instead of once per work. Best for dense shapes (every work runs the
//     full Tc KV tiles; the per-work epilogue_done -> Q TMA round trip is
//     hidden behind a long KV loop).
//   * non-persistent (classic block-per-work): gridDim.x = total_work. The
//     loop runs exactly one iteration per CTA, which is the classic
//     warp-specialized shape - HW scheduler load-balances, and short works
//     finish early to free SMs. Chosen for causal shapes where most works
//     have Tc_eff << Tc: the fixed per-work cost (Q TMA wait on
//     epilogue_done + epilogue store drain) would dominate under a
//     persistent grid.
// The barrier protocol below is valid for ANY gridDim.x: a per-CTA global
// kv-tile counter drives every mbarrier's stage/phase across works (never
// re-initialized - re-init on a live mbarrier is UB, PTX ISA 9.7.13.15.9),
// so the non-persistent launch is just the degenerate case where each
// barrier flips only its first phases. The grid choice lives in
// cute/launch.cuh (causal ? total_work : min(total_work, SMs)).
//
// Workspaces are 128-padded along seqlen; TMA descriptors are built on the
// padded flat row spaces (Q/K/V^T) and on the SF atom-layout tensors
// (SFQ/SFK/SFVt) and the (B,H,Mb,Nkv_pad) delta_s tensor (DS). lse (natural
// log, with the smooth-K correction) is written when softmax_lse != nullptr;
// km/qm may be null to skip the correction.
template <typename Traits, typename ElementO, typename TmaQ, typename TmaK,
          typename TmaV, typename TmaO, typename TmaSFQ, typename TmaSFK,
          typename TmaSFVt, typename TmaDS>
__global__ void __launch_bounds__(384, 1) persist_d_ws_fwd_cute_fp4_sm120(
    CUTLASS_GRID_CONSTANT TmaQ const tma_q,
    CUTLASS_GRID_CONSTANT TmaK const tma_k,
    CUTLASS_GRID_CONSTANT TmaV const tma_v,
    CUTLASS_GRID_CONSTANT TmaO const tma_o,
    CUTLASS_GRID_CONSTANT TmaSFQ const tma_sfq,
    CUTLASS_GRID_CONSTANT TmaSFK const tma_sfk,
    CUTLASS_GRID_CONSTANT TmaSFVt const tma_sfvt,
    CUTLASS_GRID_CONSTANT TmaDS const tma_ds, ElementO* __restrict__ O,
    float* __restrict__ softmax_lse, const float* __restrict__ km,
    const float* __restrict__ qm, int Nq, int Nkv, int Nq_pad, int Nkv_pad,
    int Nh, int Nh_kv, float scale, int Tc, int causal, int total_q_rows,
    int Nb, int q_start_row = 0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  using namespace cute;
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
  constexpr int kStages = Traits::kStages;
  constexpr int kProducerThreads = 128;
  constexpr int kConsumerThreads = 256;
  constexpr int kOffQ = Traits::kOffQ;
  constexpr int kOffSFQ = Traits::kOffSFQ;
  constexpr int kOffK = Traits::kOffK;
  constexpr int kOffSFK = Traits::kOffSFK;
  constexpr int kOffDS = Traits::kOffDS;
  constexpr int kOffV = Traits::kOffV;
  constexpr int kOffSFVt = Traits::kOffSFVt;
  constexpr int kSmemBytes = Traits::kSmemBytes;
  (void)kSmemBytes;

  const int group_size = Nh / Nh_kv;
  const int tid = threadIdx.x;
  const bool is_producer = tid < kProducerThreads;
  const int wg_tid = is_producer ? tid : tid - kProducerThreads;

  // Work decomposition: Mb tiles per (b, h), grid-strided over all works.
  const int MB = (Nq - q_start_row + kBr - 1) / kBr;
  const int total_work = MB * Nb * Nh;

  extern __shared__ __align__(1024) char shm[];
  Element* q_base = reinterpret_cast<Element*>(shm + kOffQ);

  // Barrier inventory (all initialized once, never re-init - see the
  // grid-scheduling contract above):
  //   q_full       TMA tx barrier, Q+SFQ of the current work (tx = kTxBytesQ)
  //   k_full[s]    TMA tx barrier, K+SFK+DS of kv-tile stage s (kTxBytesK)
  //   k_empty[s]   consumer->producer "stage s consumed", 256 arrivals
  //   v_full[s]    TMA tx barrier, V^T+SFVt of kv-tile stage s (kTxBytesV)
  //   v_empty[s]   consumer->producer, 256 arrivals (the gemm_rs_fp4 tail)
  //   epilogue_done consumer->producer WAR fence: the O staging tile
  //                aliases q_base, so the next work's Q TMA must wait for
  //                the previous epilogue (r2s + TMA store + lse readback)
  //                to retire fully.
  __shared__ uint64_t q_full;
  __shared__ uint64_t k_full[kStages];
  __shared__ uint64_t k_empty[kStages];
  __shared__ uint64_t v_full[kStages];
  __shared__ uint64_t v_empty[kStages];
  __shared__ uint64_t epilogue_done;

  if (tid == 0) {
    TmaBarrier::init(&q_full, 1);
    for (int s = 0; s < kStages; ++s) {
      TmaBarrier::init(&k_full[s], 1);
      CtaBarrier::init(&k_empty[s], kConsumerThreads);
      TmaBarrier::init(&v_full[s], 1);
      CtaBarrier::init(&v_empty[s], kConsumerThreads);
    }
    CtaBarrier::init(&epilogue_done, kConsumerThreads);
  }
  __syncthreads();

  if (is_producer) {
    cutlass::arch::warpgroup_reg_dealloc<32>();
    if (wg_tid == 0) {
      // Work-independent gmem base tensors: flat descriptor spaces mirror
      // the launcher: Q/K are (Nb*H*_pad, D) row planes, V^T is
      // (Nb*Hkv*D, Nkv_pad).
      auto mQ = tma_q.get_tma_tensor(
          make_shape((long)Nb * Nh * Nq_pad, Int<kHeadDim>{}));
      auto mK = tma_k.get_tma_tensor(
          make_shape((long)Nb * Nh_kv * Nkv_pad, Int<kHeadDim>{}));
      auto mV = tma_v.get_tma_tensor(
          make_shape((long)Nb * Nh_kv * kHeadDim, Nkv_pad));
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
      auto sDS =
          make_tensor(make_smem_ptr<float>(shm + kOffDS), SmemLayoutDS{});
      auto sV =
          make_tensor(make_smem_ptr<ElementPV>(shm + kOffV), SmemLayoutVt{});
      auto sSFVt = make_tensor(make_smem_ptr<ElementSFV>(shm + kOffSFVt),
                               SmemLayoutSFVt{});

      auto tQsQ = q_slice.partition_D(sQ);
      auto tQsSFQ = sfq_slice.partition_D(sSFQ);
      auto tKsK = group_modes<0, 3>(k_slice.partition_D(sK));
      auto tKsSFK = group_modes<0, 3>(sfk_slice.partition_D(sSFK));
      auto tVsV = group_modes<0, 3>(v_slice.partition_D(sV));
      auto tVsSFVt = group_modes<0, 3>(sfvt_slice.partition_D(sSFVt));
      auto tDSsDS = group_modes<0, 3>(ds_slice.partition_D(sDS));

      // Global kv-tile counter: stage/phase across works come from it, so
      // the SW barriers are never re-initialized (mbarrier re-init on a
      // live barrier is UB, PTX ISA 9.7.13.15.9).
      int g = 0;
      int w = 0;
      for (int work_id = blockIdx.x; work_id < total_work;
           work_id += gridDim.x, ++w) {
        // work_id -> (b, h, Q_tile): bh-outer / Q-tile-inner, so consecutive
        // work ids share one (b,h) and stream its Q tiles; a CTA's works are
        // strided by gridDim.x (grid-stride loop, see header contract).
        const int kv_offset = Nkv - Nq;
        const int bh = work_id / MB;
        const int Q_tile_id = work_id % MB;
        const int b = bh / Nh;
        const int Nh_id = bh % Nh;
        const int kv_head_idx = Nh_id / group_size;
        const int q_tile_abs = Q_tile_id + q_start_row / kBr;
        const int q_bh = bh;
        const int kv_bh = b * Nh_kv + kv_head_idx;
        // Flat row-space offsets into the padded descriptor planes: Q rows
        // live at bh*Nq_pad (+q_start_row for hybrid fp16/fp4 split), K rows
        // at kv_bh*Nkv_pad, V^T planes at kv_bh*kHeadDim (D x Nkv_pad).
        const int q_row_offset = q_bh * Nq_pad + q_start_row;
        const int kv_row_offset = kv_bh * Nkv_pad;
        const int v_row_base = kv_bh * kHeadDim;

        auto gQ = local_tile(domain_offset(make_coord(q_row_offset, _0{}), mQ),
                             Shape<Int<kBr>, Int<kHeadDim>>{},
                             make_coord(Q_tile_id, _0{}));
        auto gK =
            local_tile(domain_offset(make_coord(kv_row_offset, _0{}), mK),
                       Shape<Int<kBc>, Int<kHeadDim>>{}, make_coord(_, _0{}));
        auto gV =
            local_tile(domain_offset(make_coord(v_row_base, _0{}), mV),
                       Shape<Int<kHeadDim>, Int<kBc>>{}, make_coord(_0{}, _));
        auto gSFQ =
            local_tile(mSFQ(_, _, Nh_id, b), Shape<Int<kBr>, Int<kHeadDim>>{},
                       make_coord(q_tile_abs, _0{}));
        auto gSFK =
            local_tile(mSFK(_, _, kv_head_idx, b),
                       Shape<Int<kBc>, Int<kHeadDim>>{}, make_coord(_, _0{}));
        auto gSFVt =
            local_tile(mSFVt(_, _, kv_head_idx, b),
                       Shape<Int<kHeadDim>, Int<kBc>>{}, make_coord(_0{}, _));
        auto gDS = local_tile(mDS(_, _, Nh_id, b), Shape<Int<kBr>, Int<kBc>>{},
                              make_coord(q_tile_abs, _));

        auto tQgQ = q_slice.partition_S(gQ);
        auto tQgSFQ = sfq_slice.partition_S(gSFQ);
        auto tKgK = group_modes<0, 3>(k_slice.partition_S(gK));
        auto tKgSFK = group_modes<0, 3>(sfk_slice.partition_S(gSFK));
        auto tVgV = group_modes<0, 3>(v_slice.partition_S(gV));
        auto tVgSFVt = group_modes<0, 3>(sfvt_slice.partition_S(gSFVt));
        auto tDSgDS = group_modes<0, 3>(ds_slice.partition_S(gDS));

        const int Tc_eff = causal ? min(Tc, ((q_start_row + Q_tile_id * kBr +
                                              kBr - 1 + kv_offset) /
                                             kBc) +
                                                1)
                                  : Tc;

        // O staging aliases q_base: the previous work's epilogue (r2s +
        // O TMA store + lse readback of sQ) must be fully retired first.
        if (w > 0)
          CtaBarrier::wait(&epilogue_done, (w - 1) & 1);
        TmaBarrier::arrive_and_expect_tx(&q_full, Traits::kTxBytesQ);
        copy(tma_q.with(q_full), tQgQ, tQsQ);
        copy(tma_sfq.with(q_full), tQgSFQ, tQsSFQ);

        // K and V of tile n share the smem stage (g0 + n) % kStages: both
        // barriers are driven by the SAME tile sequence (consumer waits
        // k_full/v_full of one stage per kv_tile), so the counters must
        // not interleave.
        const int g0 = g;
        for (int s = 0; s < kStages - 1; ++s) {
          if (s < Tc_eff) {
            const int seq = g0 + s;
            const int stage = seq % kStages;
            const int phase = (seq / kStages) & 1;
            CtaBarrier::wait(&k_empty[stage], phase);
            TmaBarrier::arrive_and_expect_tx(&k_full[stage], Traits::kTxBytesK);
            copy(tma_k.with(k_full[stage]), tKgK(_, s), tKsK(_, stage));
            copy(tma_sfk.with(k_full[stage]), tKgSFK(_, s), tKsSFK(_, stage));
            copy(tma_ds.with(k_full[stage]), tDSgDS(_, s), tDSsDS(_, stage));
          }
        }
        for (int s = 0; s < kStages - 1; ++s) {
          if (s < Tc_eff) {
            const int seq = g0 + s;
            const int stage = seq % kStages;
            const int phase = (seq / kStages) & 1;
            CtaBarrier::wait(&v_empty[stage], phase);
            TmaBarrier::arrive_and_expect_tx(&v_full[stage], Traits::kTxBytesV);
            copy(tma_v.with(v_full[stage]), tVgV(_, s), tVsV(_, stage));
            copy(tma_sfvt.with(v_full[stage]), tVgSFVt(_, s),
                 tVsSFVt(_, stage));
          }
        }
        for (int tile = 0; tile < Tc_eff; ++tile) {
          {
            const int v_tile = tile + kStages - 1;
            if (v_tile < Tc_eff) {
              const int seq = g0 + v_tile;
              const int stage = seq % kStages;
              const int phase = (seq / kStages) & 1;
              CtaBarrier::wait(&v_empty[stage], phase);
              TmaBarrier::arrive_and_expect_tx(&v_full[stage],
                                               Traits::kTxBytesV);
              copy(tma_v.with(v_full[stage]), tVgV(_, v_tile), tVsV(_, stage));
              copy(tma_sfvt.with(v_full[stage]), tVgSFVt(_, v_tile),
                   tVsSFVt(_, stage));
            }
          }
          {
            const int k_tile = tile + kStages - 1;
            if (k_tile < Tc_eff) {
              const int seq = g0 + k_tile;
              const int stage = seq % kStages;
              const int phase = (seq / kStages) & 1;
              CtaBarrier::wait(&k_empty[stage], phase);
              TmaBarrier::arrive_and_expect_tx(&k_full[stage],
                                               Traits::kTxBytesK);
              copy(tma_k.with(k_full[stage]), tKgK(_, k_tile), tKsK(_, stage));
              copy(tma_sfk.with(k_full[stage]), tKgSFK(_, k_tile),
                   tKsSFK(_, stage));
              copy(tma_ds.with(k_full[stage]), tDSgDS(_, k_tile),
                   tDSsDS(_, stage));
            }
          }
        }
        g += Tc_eff;
      }
    }
    return;
  }

  // Consumer
  cutlass::arch::warpgroup_reg_alloc<232>();
  // Pre-drain: mark every stage empty so the producer's first kStages-1
  // prefetches find their k_empty/v_empty barriers armed.
  for (int s = 0; s < kStages; ++s) {
    CtaBarrier::arrive(&k_empty[s]);
    CtaBarrier::arrive(&v_empty[s]);
  }

  TmaBarrier::wait(&q_full, 0);
  cutlass::arch::fence_view_async_shared();

  TiledMmaQK tiled_mma_qk;
  TiledMmaPV tiled_mma_pv;
  auto thread_mma_qk = tiled_mma_qk.get_thread_slice(wg_tid);
  auto thread_mma_pv = tiled_mma_pv.get_thread_slice(wg_tid);

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

  // Register fragments for both mmas. partition_fragment_{A,B} mirror the
  // blockscaled operand convention: each operand is a (data, SF) pair, so
  // A comes as tSrQ/tSrSFQ (Q) or tOrP/tOrSFP (P, quantized in-flight), B as
  // tSrK/tSrSFK and tOrVt/tOrSFVt. tOrP/tOrSFP are built on LayoutP/LayoutSFP
  // (traits) rather than partition_fragment_A because they must ADAPT the QK
  // C-fragment slots onto the PV A-operand slots - see quantize_and_pack_p.
  Tensor tSrQ = thread_mma_qk.partition_fragment_A(sQ);
  Tensor tSrK = thread_mma_qk.partition_fragment_B(sK(_, _, Int<0>{}));
  Tensor tOrVt = thread_mma_pv.partition_fragment_B(sV(_, _, Int<0>{}));
  Tensor tSrSFQ = partition_fragment_SFA(sSFQ, thread_mma_qk);
  Tensor tSrSFK = partition_fragment_SFB(sSFK(_, _, Int<0>{}), thread_mma_qk);
  Tensor tOrSFVt = partition_fragment_SFB(sSFVt(_, _, Int<0>{}), thread_mma_pv);
  Tensor tOrP = make_tensor_like<ElementPV>(typename Traits::LayoutP{});
  Tensor tOrSFP = make_tensor<ElementSFV>(typename Traits::LayoutSFP{});

  auto smem_tiled_copy_Q = make_tiled_copy_A(SmemCopyAtomQ{}, tiled_mma_qk);
  auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(wg_tid);
  Tensor tSsQ =
      smem_thr_copy_Q.partition_S(as_position_independent_swizzle_tensor(sQ));
  Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);

  auto smem_tiled_copy_K = make_tiled_copy_B(SmemCopyAtomKV{}, tiled_mma_qk);
  auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(wg_tid);
  Tensor tSsK =
      smem_thr_copy_K.partition_S(as_position_independent_swizzle_tensor(sK));

  auto smem_tiled_copy_V = make_tiled_copy_B(SmemCopyAtomV{}, tiled_mma_pv);
  auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(wg_tid);
  Tensor tOsVt =
      smem_thr_copy_V.partition_S(as_position_independent_swizzle_tensor(sV));

  auto tile_shape_mnk = tile_shape(tiled_mma_qk);
  auto smem_tiled_copy_SFQ = make_tiled_copy_impl(
      SmemCopyAtomSF{}, get_layoutSFA_TV(tiled_mma_qk),
      make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk)));
  auto smem_thr_copy_SFQ = smem_tiled_copy_SFQ.get_thread_slice(wg_tid);
  Tensor tSsSFQ = smem_thr_copy_SFQ.partition_S(
      as_position_independent_swizzle_tensor(sSFQ));
  Tensor tSrSFQ_copy_view = smem_thr_copy_SFQ.retile_D(tSrSFQ);

  auto smem_tiled_copy_SFK = make_tiled_copy_impl(
      SmemCopyAtomSF{}, get_layoutSFB_TV(tiled_mma_qk),
      make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
  auto smem_thr_copy_SFK = smem_tiled_copy_SFK.get_thread_slice(wg_tid);
  Tensor tSsSFK = smem_thr_copy_SFK.partition_S(
      as_position_independent_swizzle_tensor(sSFK));

  auto smem_tiled_copy_SFV =
      make_tiled_copy_impl(SmemCopyAtomSFV{}, get_layoutSFB_TV(tiled_mma_pv),
                           make_shape(size<1>(tile_shape(tiled_mma_pv)),
                                      size<2>(tile_shape(tiled_mma_pv))));
  auto smem_thr_copy_SFV = smem_tiled_copy_SFV.get_thread_slice(wg_tid);
  Tensor tOsSFVt = smem_thr_copy_SFV.partition_S(
      as_position_independent_swizzle_tensor(sSFVt));

  // The QK accumulator fragment is viewed through THREE layouts, each
  // matching one consumer:
  //   raw tSrS          - mma C-fragment order: what gemm_ss_fp4 accumulates
  //                       into and what online_softmax_with_quant scans,
  //   conversion_view   - order the e2m1 packer reads (8 floats -> uint32),
  //                       consumed by quantize_and_pack_p,
  //   reduction_view    - (row, col) addressing for masking, built per tile.
  Tensor tSrS = partition_fragment_C(tiled_mma_qk, Shape<Int<kBr>, Int<kBc>>{});
  Tensor tSrS_conversion_view =
      make_tensor(tSrS.data(), convert_to_conversion_layout(tSrS.layout()));
  Tensor tSrS_reduction_view =
      make_tensor(tSrS.data(), convert_to_reduction_layout(tSrS.layout()));
  // Per-token-group absmax of the P-domain scores; softmax fills it, the
  // packer turns it into the SFP operand. NVFP4: 16-token groups derived
  // from the conversion view; MXFP8: 32-token groups (= one mma-k32 block,
  // the reduction view's MmaN extent).
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

  Tensor tOrO_store =
      partition_fragment_C(tiled_mma_pv, Shape<Int<kBr>, Int<kHeadDim>>{});

  constexpr int kSoftmaxRows = 2 * (2 * kBr / kConsumerThreads);
  std::conditional_t<kPvMxfp8, SoftmaxFusedMxfp8<kSoftmaxRows>,
                     SoftmaxFused<kSoftmaxRows>>
      softmax_fused;
  const float scale_orig = scale;
  const float softmax_scale_log2 = scale * FFPA_M_LOG2E;

  // Preload the rank-1 delta_s term (qm @ K^T, kept in fp32 smem as a
  // stride-(0,1) row broadcast) into the QK accumulator BEFORE the mma
  // chain, so S = Qhat@Khat^T + qm@K^T falls out as one accumulate. The
  // float4 slot arithmetic below maps each thread's C-fragment slots (quad
  // pairs x 4 f32) onto the matching DS broadcast columns - SA3 verbatim,
  // consistent with the permuted K storage (header "Column alignment").
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
      acc_float4(make_coord(make_coord(_0{}, _0{}), _1{}), _0{}, i) = delta_s_0;
      acc_float4(make_coord(make_coord(_0{}, _1{}), _0{}), _0{}, i) = delta_s_1;
      acc_float4(make_coord(make_coord(_0{}, _1{}), _1{}), _0{}, i) = delta_s_1;
    }
  };

  // QK/PV gemm loops live in ../fp4_gemm.cuh (gemm_ss_fp4 / gemm_rs_fp4,
  // the fp4 counterpart of ffpa_cute::gemm_ss/gemm_rs); quantize_and_pack_p
  // lives in ../fp4_pscale.cuh.

  int g = 0;
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

    if (w > 0) {
      TmaBarrier::wait(&q_full, w & 1);
      cutlass::arch::fence_view_async_shared();
    }
    // Q/SFQ are per-work constants: load the mma fragments once here, not
    // inside the kv_tile loop. Without this the A/SFA asm operands are
    // uninitialized and cicc folds them to 0: QK degenerates to delta_s
    // (rank-1 mean attention), which the probe tolerances masked.
    copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    copy(smem_tiled_copy_SFQ, tSsSFQ, tSrSFQ_copy_view);

    clear(tOrO_store);

#pragma unroll 1
    for (int kv_tile = 0; kv_tile < Tc_eff; ++kv_tile, ++g) {
      const int k_stg = g % kStages;
      const int k_phase = (g / kStages) & 1;
      const int v_stg = k_stg;
      const int v_phase = k_phase;

      TmaBarrier::wait(&k_full[k_stg], k_phase);
      cutlass::arch::fence_view_async_shared();

      // delta_s preloads the rank-1 qm@K^T term into the C accumulator
      // before the QK mma chain adds on top (writes tSrS regs only - no
      // overlap with gemm_ss_fp4's tSrK/tSrSFK operand loads).
      add_delta_s(tSrS, k_stg);
      gemm_ss_fp4(tSrS, tSrQ, tSrSFQ, tSrK, tSrSFK, tSsK, tSsSFK, tiled_mma_qk,
                  smem_tiled_copy_K, smem_thr_copy_K, smem_tiled_copy_SFK,
                  smem_thr_copy_SFK, k_empty, k_stg);

      // Masking: kv-tail (padded columns) + causal (bottom-right). The
      // logical column indexes the PERMUTED storage order, so the token
      // position goes through kv_perm32; the -inf assignment overwrites any
      // delta_s garbage in masked slots. Softmax InfCheck handles rows whose
      // valid columns all land outside this tile.
      {
        auto scores = make_tensor(tSrS.data(),
                                  convert_to_reduction_layout(tSrS.layout()));
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

      // Online softmax fused with the P quantization prep: updates the
      // running row max/sum (rescaling O lazily via scores_scale below),
      // shifts scores into the P2 = P*2688 domain inside exp2 (the 1/(448*6)
      // global scale folded in, see fp4_pscale.cuh), and records the per-
      // 16-token-group absmax into AbsMaxP for quantize_and_pack_p.
      // InfCheck flushes rows whose valid columns all fall outside this
      // tile (causal top-left / fully masked), keeping row_sum finite.
      if (kv_tile == 0)
        softmax_fused.template online_softmax_with_quant</*FirstTile=*/true,
                                                         /*InfCheck=*/true>(
            tSrS, AbsMaxP, softmax_scale_log2);
      else
        softmax_fused.template online_softmax_with_quant</*FirstTile=*/false,
                                                         /*InfCheck=*/true>(
            tSrS, AbsMaxP, softmax_scale_log2);

      // V is loaded after the softmax so the QK math and the V TMA overlap;
      // k_stg == v_stg by construction (same tile sequence drives both).
      TmaBarrier::wait(&v_full[v_stg], v_phase);
      cutlass::arch::fence_view_async_shared();

      auto gemm_rs_pv = [&](auto& tgt) {
        if constexpr (kPvMxfp8)
          gemm_rs_mxfp8(tgt, tOrP, tOrSFP, tOrVt, tOrSFVt, tOsVt, tOsSFVt,
                        tiled_mma_pv, smem_tiled_copy_V, smem_thr_copy_V,
                        smem_tiled_copy_SFV, smem_thr_copy_SFV, AbsMaxP,
                        tSrS_reduction_view, v_empty, v_stg, wg_tid & 31);
        else
          gemm_rs_fp4(tgt, tOrP, tOrSFP, tOrVt, tOrSFVt, tOsVt, tOsSFVt,
                      tiled_mma_pv, smem_tiled_copy_V, smem_thr_copy_V,
                      smem_tiled_copy_SFV, smem_thr_copy_SFV, AbsMaxP,
                      tSrS_conversion_view, v_empty, v_stg);
      };

      if (kv_tile == 0) {
        gemm_rs_pv(tOrO_store);
      } else {
        // scores_scale == 1.0f exactly when the row max did not move this
        // tile (~96% of dense tiles): O = O*1 + O_new needs no rescale at
        // all. Warp-vote keeps both fragments on one uniform path.
        const bool need_rescale = softmax_fused.scores_scale[0] != 1.0f ||
                                  softmax_fused.scores_scale[1] != 1.0f;
        if (__any_sync(0xffffffff, need_rescale)) {
          Tensor tOrO = make_fragment_like(tOrO_store);
          clear(tOrO);
          gemm_rs_pv(tOrO);
          softmax_fused.rescale_o(tOrO_store, tOrO);
        } else {
          gemm_rs_pv(tOrO_store);
        }
      }
    }

    softmax_fused.finalize(tOrO_store);

    // Epilogue, four ordered steps (the O staging tile aliases the Q smem,
    // which drives the ordering):
    //   1. lse correction (optional): lse_qkm_dot reads sQ/sSFQ back from
    //      smem - must run BEFORE O staging overwrites q_base.
    //   2. f32 -> ElementO convert of the PV accumulator.
    //   3a. full Q tile: STSM r2s into the staged O tile + one TMA store
    //       (coalesced, swizzle-matched descriptor).
    //   3b. tail Q tile (Br_base+kBr > Nq): the flattened [total_q_rows, D]
    //       TMA space would alias the next head's rows, so store R->G with a
    //       row guard instead.
    //   4. lse write (P2-domain formula + the smooth-K correction).
    float qkm[kSRows];
    const bool smooth_lse =
        (softmax_lse != nullptr) && (km != nullptr) && (qm != nullptr);
    {
      cutlass::arch::NamedBarrier::sync(kConsumerThreads, 0);

      if (smooth_lse) {
        const float* km_bh = km + static_cast<long>(kv_bh) * kHeadDim;
        const long qm_mb = Nq_pad / kBr;
        const float* qm_blk =
            qm + (static_cast<long>(q_bh) * qm_mb + q_tile_abs) * kHeadDim;
        lse_qkm_dot<kHeadDim, kSRows>(sQ, sSFQ, tScS_rc, km_bh, qm_blk, qkm);
        // lse_qkm_dot reads sQ/sSFQ; the O staging below overwrites that smem.
        cutlass::arch::NamedBarrier::sync(kConsumerThreads, 0);
      }

      auto tCrOHalf = ffpa_cute::convert_type<ElementO>(tOrO_store);

      if (Br_base + kBr <= Nq - q_start_row) {
        auto sO = as_position_independent_swizzle_tensor(make_tensor(
            make_smem_ptr(reinterpret_cast<ElementO*>(q_base)), SmemLayoutO{}));
        auto r2s_copy = make_tiled_copy_C(
            Copy_Atom<SM90_U32x2_STSM_N, ElementO>{}, tiled_mma_pv);
        auto r2s_thr = r2s_copy.get_thread_slice(wg_tid);
        auto tCrOHalf_src = r2s_thr.retile_S(tCrOHalf);
        auto tCsO_dst = r2s_thr.partition_D(sO);
        copy(r2s_copy, tCrOHalf_src, tCsO_dst);
        cutlass::arch::fence_view_async_shared();
        cutlass::arch::NamedBarrier::sync(kConsumerThreads, 0);

        auto mO_tma = domain_offset(make_coord(O_row_offset, 0),
                                    tma_o.get_tma_tensor(make_shape(
                                        (long)total_q_rows, Int<kHeadDim>{})));
        auto o_slice = tma_o.get_slice(_0{});
        auto gO_tma = local_tile(mO_tma, Shape<Int<kBr>, Int<kHeadDim>>{},
                                 make_coord(Q_tile_id, _0{}));
        auto tCgO_tma = o_slice.partition_D(gO_tma);
        auto tOsO = o_slice.partition_S(sO);
        if (wg_tid == 0)
          copy(tma_o, tOsO, tCgO_tma);
        tma_store_arrive();
        tma_store_wait<0>();
      } else {
        // Tail tile: rows past Nq would alias the next head in the flattened
        // [total_q_rows, D] TMA space, so store R->G with a row guard.
        const int O_gmem_offset = (q_bh)*Nq * kHeadDim + q_start_row * kHeadDim;
        auto mO = make_tensor(make_gmem_ptr(O + O_gmem_offset),
                              make_shape(Nq - q_start_row, Int<kHeadDim>{}),
                              make_stride(Int<kHeadDim>{}, _1{}));
        auto gO = local_tile(mO, Shape<Int<kBr>, Int<kHeadDim>>{},
                             make_coord(Q_tile_id, _0{}));
        auto tCgO = thread_mma_pv.partition_C(gO);
        auto cO = make_identity_tensor(Shape<Int<kBr>, Int<kHeadDim>>{});
        auto tOcO = thread_mma_pv.partition_C(cO);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tCrOHalf); ++i) {
          const int global_row = Br_base + cute::get<0>(tOcO(i));
          if (global_row < Nq - q_start_row)
            tCgO(i) = tCrOHalf(i);
        }
      }

      if (softmax_lse != nullptr) {
        const int lse_base = Nb_id * Nh * Nq + Nh_id * Nq;
        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < kSRows; ++row) {
          // row_sum lives in the P2 = P*2688 domain: lse = scale*m +
          // ln(row_sum / 2688); fp8_scalexfp4_scale_log2 is log2(1/2688) < 0.
          float lse = (softmax_fused.row_max[row] * softmax_scale_log2 +
                       log2f(softmax_fused.row_sum[row]) +
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

    // Release q_base for the next work's Q TMA (O staging aliases it).
    CtaBarrier::arrive(&epilogue_done);
  }  // persistent work loop
#endif  // defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
}

}  // namespace ffpa_fp4

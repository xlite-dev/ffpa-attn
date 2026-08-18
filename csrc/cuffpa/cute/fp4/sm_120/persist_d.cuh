// NVFP4 (e2m1 data + ue4m3 block scale) persist-D warp-specialized kernel for
// sm_120, ported from SageAttention3's NVFP4 data path onto the ffpa fp8
// persist_d producer/consumer skeleton (128T producer + 256T consumer,
// hand-rolled TMA barriers, consumer-side epilogue).
//
// Math chain (per (b,h), Q block of 128 rows):
//   qm  = mean(q, 128-row group)          Qhat = q - qm   (quantized, sub_qm)
//   km  = mean(k)                         Khat = k - km   (quantized, sub_km,
//   rows permuted) S   = Qhat @ Khat^T + delta_s,  delta_s[b,h,mb,n] = qm @ (k
//   - km)^T
//       = (q - qm)(k - km)^T + qm(k - km)^T = q(k - km)^T
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
// is consumed by the blockscaled PV mma. Fully-masked groups quantize to
// P=NaN (0/0) with SFP=0, and the mma's scale multiply flushes them to zero.
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
#include "../cute_ext.h"
#include "../fp4_pscale.cuh"

namespace ffpa_fp4 {

using TmaBarrier = cutlass::arch::ClusterTransactionBarrier;
using CtaBarrier = cutlass::arch::ClusterBarrier;

// NVFP4 persist-D traits, D=128 only: SM120 blockscaled 16x32x64 mma
// (4x mma.sync m16n8k64 kind::mxf4nvf4 ue4m3 scale_vec::4X) tiled 8x1x1 over
// a (128, 32, 128) tile for both QK and PV. Q/K/V^T smem share the
// sm120_rr K-major swizzle atom; SF smem uses the BlockScaledConfig atom
// layouts; DS (delta_s) is a stride-(0,1) 128-float broadcast tile.
template <typename ElementO_>
struct Fp4PersistDTraits {
  static constexpr int kBr = 128;
  static constexpr int kBc = 128;
  static constexpr int kHeadDim = 128;
  static constexpr int kStages = 3;

  using Element = cutlass::float_e2m1_t;
  using ElementSF = cutlass::float_ue4m3_t;
  using ElementO = ElementO_;

  using TileShape_MNK = Shape<_128, _128, _128>;
  using MMAAtom =
      MMA_Atom<cute::SM120::BLOCKSCALED::SM120_16x32x64_TN_VS_NVFP4>;
  using AtomLayoutMNK = Layout<Shape<_8, _1, _1>>;
  using TiledMmaQK = decltype(make_tiled_mma(MMAAtom{}, AtomLayoutMNK{},
                                             Tile<_128, _32, _128>{}));
  using TiledMmaPV = TiledMmaQK;

  using SmemLayoutAtomQKV =
      decltype(cutlass::gemm::collective::detail::sm120_rr_smem_selector<
               Element, Int<kHeadDim>>());
  using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQKV{},
                                             Shape<Int<kBr>, Int<kHeadDim>>{}));
  using SmemLayoutK = decltype(tile_to_shape(
      SmemLayoutAtomQKV{},
      make_shape(Int<kBc>{}, Int<kHeadDim>{}, Int<kStages>{})));
  using SmemLayoutVt = decltype(tile_to_shape(
      SmemLayoutAtomQKV{},
      make_shape(Int<kHeadDim>{}, Int<kBc>{}, Int<kStages>{})));

  using BlkScaledConfig = BlockScaledConfig<16>;
  using SmemLayoutAtomSFQ = decltype(BlkScaledConfig::deduce_smem_layoutSFQ(
      TiledMmaQK{}, TileShape_MNK{}));
  using SmemLayoutAtomSFK = decltype(BlkScaledConfig::deduce_smem_layoutSFKV(
      TiledMmaQK{}, TileShape_MNK{}));
  using SmemLayoutAtomSFVt = decltype(BlkScaledConfig::deduce_smem_layoutSFVt(
      TiledMmaPV{}, Shape<Int<kBr>, Int<kHeadDim>, Int<kBc>>{}));
  using SmemLayoutSFQ = decltype(make_layout(shape(SmemLayoutAtomSFQ{}),
                                             stride(SmemLayoutAtomSFQ{})));
  using SmemLayoutSFK =
      decltype(make_layout(append(shape(SmemLayoutAtomSFK{}), Int<kStages>{}),
                           append(stride(SmemLayoutAtomSFK{}),
                                  size(filter_zeros(SmemLayoutAtomSFK{})))));
  using SmemLayoutSFVt =
      decltype(make_layout(append(shape(SmemLayoutAtomSFVt{}), Int<kStages>{}),
                           append(stride(SmemLayoutAtomSFVt{}),
                                  size(filter_zeros(SmemLayoutAtomSFVt{})))));

  using SmemLayoutAtomDS = Layout<Shape<_128, _128>, Stride<_0, _1>>;
  using SmemLayoutDS = decltype(tile_to_shape(
      SmemLayoutAtomDS{}, make_shape(Int<kBr>{}, Int<kBc>{}, Int<kStages>{})));

  // P / SFP rmem fragment layouts: adapter from the QK C-fragment slots to
  // the PV A-operand (k = token) mapping. SA3 verbatim.
  using LayoutP = decltype(make_layout(
      make_shape(make_shape(_8{}, _2{}, _2{}), _1{}, Int<kBc / 64>{}),
      make_stride(make_stride(_1{}, _8{}, _16{}), _0{}, _32{})));
  using LayoutSFP = decltype(make_layout(
      make_shape(make_shape(_16{}, _4{}), _1{}, Int<kBc / 64>{}),
      make_stride(make_stride(_0{}, _1{}), _0{}, _4{})));

  using SmemLayoutAtomO =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K, ElementO, Int<kBr>, Int<kHeadDim>>());
  using SmemLayoutO = decltype(tile_to_shape(
      SmemLayoutAtomO{}, Shape<Int<kBr>, Int<kHeadDim>>{}, Step<_1, _2>{}));

  using SmemCopyAtomQ = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
  using SmemCopyAtomKV = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
  // NOTE: SF smem->reg copies stay byte-granular: the SFA/SFB TV layouts
  // are not 4-value contiguous, so a 32-bit copy atom fails cute's
  // vectorization static assert (tried, falsified).
  using SmemCopyAtomSF = Copy_Atom<UniversalCopy<ElementSF>, ElementSF>;

  // 1 TMA barrier arrival per stage; tx bytes include data + SF (+ DS).
  static constexpr uint32_t kTxBytesQ =
      static_cast<uint32_t>(cute::bits_to_bytes(cosize(SmemLayoutSFQ{}) * 8)) +
      static_cast<uint32_t>(cute::bits_to_bytes(size(SmemLayoutQ{}) * 4));
  static constexpr uint32_t kTxBytesK =
      static_cast<uint32_t>(
          cute::bits_to_bytes(cosize(take<0, 2>(SmemLayoutSFK{})) * 8)) +
      static_cast<uint32_t>(
          cute::bits_to_bytes(cosize(take<0, 2>(SmemLayoutDS{})) * 32)) +
      static_cast<uint32_t>(
          cute::bits_to_bytes(size(take<0, 2>(SmemLayoutK{})) * 4));
  static constexpr uint32_t kTxBytesV =
      static_cast<uint32_t>(
          cute::bits_to_bytes(cosize(take<0, 2>(SmemLayoutSFVt{})) * 8)) +
      static_cast<uint32_t>(
          cute::bits_to_bytes(size(take<0, 2>(SmemLayoutVt{})) * 4));

  // SMEM plan: [Q | SFQ | K*s | SFK*s | DS*s | V^T*s | SFVt*s], regions
  // padded to 1024B so the SW128 TMA destinations stay swizzle-span aligned.
  // The O staging tile (kBr*kHeadDim*2B = 32KB) aliases q_base in the
  // epilogue, after the KV loop has consumed everything below kSmemBytes.
  static constexpr int kQBytes =
      int(cute::bits_to_bytes(size(SmemLayoutQ{}) * 4));
  static constexpr int kSFQBytes =
      int(cute::bits_to_bytes(cosize(SmemLayoutSFQ{}) * 8));
  static constexpr int kKBytesStage =
      int(cute::bits_to_bytes(size(take<0, 2>(SmemLayoutK{})) * 4));
  static constexpr int kSFKBytesStage =
      int(cute::bits_to_bytes(cosize(take<0, 2>(SmemLayoutSFK{})) * 8));
  static constexpr int kDSBytesStage =
      int(cute::bits_to_bytes(cosize(take<0, 2>(SmemLayoutDS{})) * 32));
  static constexpr int kVBytesStage =
      int(cute::bits_to_bytes(size(take<0, 2>(SmemLayoutVt{})) * 4));
  static constexpr int kSFVtBytesStage =
      int(cute::bits_to_bytes(cosize(take<0, 2>(SmemLayoutSFVt{})) * 8));
  static constexpr int kOffQ = 0;
  static constexpr int kOffSFQ = kOffQ + kQBytes;
  static constexpr int kOffK = kOffSFQ + kSFQBytes;
  static constexpr int kOffSFK = kOffK + kStages * kKBytesStage;
  static constexpr int kOffDS = kOffSFK + kStages * kSFKBytesStage;
  static constexpr int kOffV0 = kOffDS + kStages * kDSBytesStage;
  static constexpr int kOffV = (kOffV0 + 1023) / 1024 * 1024;
  static constexpr int kOffSFVt = kOffV + kStages * kVBytesStage;
  static constexpr int kSmemBytes = kOffSFVt + kStages * kSFVtBytesStage;
  static_assert(kOffK % 1024 == 0 && kOffV % 1024 == 0, "SW128 smem alignment");
  // sm_120 (GeForce/PRO Blackwell) opt-in smem per block is 101,376B —
  // NOT the 227KB of datacenter parts. kStages > 4 exceeds it and fails
  // silently (verified: score collapses to zero past 99KB).
  static_assert(kSmemBytes <= 101376, "smem budget");
  static_assert(kBr * kHeadDim * 2 <= kSmemBytes,
                "O staging must fit the freed smem");
};

// K/V^T storage column j -> original token index (the quantize kernels'
// 32-row interleave; bijection inside every 32-window, identity across).
CUTE_DEVICE int kv_perm32(int j) {
  const int loc = j & 31;
  return (j & ~31) + (loc / 8) * 2 + ((loc % 8) / 2) * 8 + (loc % 8) % 2;
}

// lse smooth-K correction: qkm[row] = dot(Qhat_row_dequant, km) +
// dot(qm_block, km). Qhat is read back from smem (e2m1 x SF), quad-strided
// like fp8's smooth_k_qk_dot; the qm term is CTA-constant per Q tile.
template <int kHeadDim, int kRows, typename SmemQTensor, typename SfQTensor,
          typename CoordTensor>
CUTE_DEVICE void lse_qkm_dot(const SmemQTensor& sQ, const SfQTensor& sSFQ,
                             const CoordTensor& tScS_rc,
                             const float* __restrict__ km_bh,
                             const float* __restrict__ qm_blk, float* qkm) {
  constexpr int kQuad = 4;
  constexpr int kIters = kHeadDim / (kQuad * 4);
  const int qlane = threadIdx.x & 3;
#pragma unroll
  for (int row = 0; row < kRows; ++row) {
    const int r = cute::get<0>(tScS_rc(row, 0));
    float acc = 0.0f;
#pragma unroll
    for (int it = 0; it < kIters; ++it) {
      const int col = (qlane + it * kQuad) * 4;
      const float sf = static_cast<float>(sSFQ(r, col));
#pragma unroll
      for (int d = 0; d < 4; ++d)
        acc += static_cast<float>(sQ(r, col + d).get()) * sf * km_bh[col + d];
    }
    qkm[row] = acc;
  }
  float c = 0.0f;
#pragma unroll
  for (int it = 0; it < kIters; ++it) {
    const int col = (qlane + it * kQuad) * 4;
#pragma unroll
    for (int d = 0; d < 4; ++d)
      c += qm_blk[col + d] * km_bh[col + d];
  }
#pragma unroll
  for (int row = 0; row < kRows; ++row) {
    qkm[row] += __shfl_xor_sync(0xffffffff, qkm[row], 1);
    qkm[row] += __shfl_xor_sync(0xffffffff, qkm[row], 2);
    qkm[row] += c;
  }
  c += __shfl_xor_sync(0xffffffff, c, 1);
  c += __shfl_xor_sync(0xffffffff, c, 2);
#pragma unroll
  for (int row = 0; row < kRows; ++row)
    qkm[row] += c;
}

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
  using SmemCopyAtomSF = typename Traits::SmemCopyAtomSF;
  using BlkScaledConfig = typename Traits::BlkScaledConfig;

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

  __shared__ uint64_t q_full;
  __shared__ uint64_t k_full[kStages];
  __shared__ uint64_t k_empty[kStages];
  __shared__ uint64_t v_full[kStages];
  __shared__ uint64_t v_empty[kStages];
  // Consumers arrive after each epilogue; the producer waits before the
  // next Q TMA (the O staging tile aliases q_base: WAR hazard).
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
      auto layout_SFVt = BlkScaledConfig::tile_atom_to_shape_SFVt(
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
          make_tensor(make_smem_ptr<Element>(shm + kOffV), SmemLayoutVt{});
      auto sSFVt = make_tensor(make_smem_ptr<ElementSF>(shm + kOffSFVt),
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
        const int kv_offset = Nkv - Nq;
        const int bh = work_id / MB;
        const int Q_tile_id = work_id % MB;
        const int b = bh / Nh;
        const int Nh_id = bh % Nh;
        const int kv_head_idx = Nh_id / group_size;
        const int q_tile_abs = Q_tile_id + q_start_row / kBr;
        const int q_bh = bh;
        const int kv_bh = b * Nh_kv + kv_head_idx;
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
  auto sV = make_tensor(make_smem_ptr<Element>(shm + kOffV), SmemLayoutVt{});
  auto sSFVt =
      make_tensor(make_smem_ptr<ElementSF>(shm + kOffSFVt), SmemLayoutSFVt{});

  Tensor tSrQ = thread_mma_qk.partition_fragment_A(sQ);
  Tensor tSrK = thread_mma_qk.partition_fragment_B(sK(_, _, Int<0>{}));
  Tensor tOrVt = thread_mma_pv.partition_fragment_B(sV(_, _, Int<0>{}));
  Tensor tSrSFQ = partition_fragment_SFA(sSFQ, thread_mma_qk);
  Tensor tSrSFK = partition_fragment_SFB(sSFK(_, _, Int<0>{}), thread_mma_qk);
  Tensor tOrSFVt = partition_fragment_SFB(sSFVt(_, _, Int<0>{}), thread_mma_pv);
  Tensor tOrP = make_tensor_like<Element>(typename Traits::LayoutP{});
  Tensor tOrSFP = make_tensor<ElementSF>(typename Traits::LayoutSFP{});

  auto smem_tiled_copy_Q = make_tiled_copy_A(SmemCopyAtomQ{}, tiled_mma_qk);
  auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(wg_tid);
  Tensor tSsQ =
      smem_thr_copy_Q.partition_S(as_position_independent_swizzle_tensor(sQ));
  Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);

  auto smem_tiled_copy_K = make_tiled_copy_B(SmemCopyAtomKV{}, tiled_mma_qk);
  auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(wg_tid);
  Tensor tSsK =
      smem_thr_copy_K.partition_S(as_position_independent_swizzle_tensor(sK));
  Tensor tSrK_copy_view = smem_thr_copy_K.retile_D(tSrK);

  auto smem_tiled_copy_V = make_tiled_copy_B(SmemCopyAtomKV{}, tiled_mma_pv);
  auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(wg_tid);
  Tensor tOsVt =
      smem_thr_copy_V.partition_S(as_position_independent_swizzle_tensor(sV));
  Tensor tOrVt_copy_view = smem_thr_copy_V.retile_D(tOrVt);

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
  Tensor tSrSFK_copy_view = smem_thr_copy_SFK.retile_D(tSrSFK);

  auto smem_tiled_copy_SFV = make_tiled_copy_impl(
      SmemCopyAtomSF{}, get_layoutSFB_TV(tiled_mma_pv),
      make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
  auto smem_thr_copy_SFV = smem_tiled_copy_SFV.get_thread_slice(wg_tid);
  Tensor tOsSFVt = smem_thr_copy_SFV.partition_S(
      as_position_independent_swizzle_tensor(sSFVt));
  Tensor tOrSFVt_copy_view = smem_thr_copy_SFV.retile_D(tOrSFVt);

  Tensor tSrS = partition_fragment_C(tiled_mma_qk, Shape<Int<kBr>, Int<kBc>>{});
  Tensor tSrS_conversion_view =
      make_tensor(tSrS.data(), convert_to_conversion_layout(tSrS.layout()));
  Tensor AbsMaxP = make_tensor_like<float>(make_layout(shape(group<1, 4>(
      flatten(tSrS_conversion_view.layout()(make_coord(_0{}, _), _, _))))));

  auto cS = make_identity_tensor(Shape<Int<kBr>, Int<kBc>>{});
  auto tScS = thread_mma_qk.partition_C(cS);
  auto tScS_rc =
      make_tensor(tScS.data(), convert_to_reduction_layout(tScS.layout()));
  constexpr int kSRows = decltype(size<0>(tScS_rc))::value;
  constexpr int kSCols = decltype(size<1>(tScS_rc))::value;

  Tensor tOrO_store =
      partition_fragment_C(tiled_mma_pv, Shape<Int<kBr>, Int<kHeadDim>>{});

  constexpr int kSoftmaxRows = 2 * (2 * kBr / kConsumerThreads);
  SoftmaxFused<kSoftmaxRows> softmax_fused;
  const float scale_orig = scale;
  const float softmax_scale_log2 = scale * FFPA_M_LOG2E;

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

  auto copy_k_block = [&](auto block_id, int stage) {
    auto tSsK_stage = tSsK(_, _, _, stage);
    auto tSsSFK_stage = tSsSFK(_, _, _, stage);
    copy(smem_tiled_copy_K, tSsK_stage(_, _, block_id),
         tSrK_copy_view(_, _, block_id));
    copy(smem_tiled_copy_SFK, tSsSFK_stage(_, _, block_id),
         tSrSFK_copy_view(_, _, block_id));
  };
  auto copy_v_block = [&](auto block_id, int stage) {
    auto tOsVt_stage = tOsVt(_, _, _, stage);
    auto tOsSFVt_stage = tOsSFVt(_, _, _, stage);
    copy(smem_tiled_copy_V, tOsVt_stage(_, _, block_id),
         tOrVt_copy_view(_, _, block_id));
    copy(smem_tiled_copy_SFV, tOsSFVt_stage(_, _, block_id),
         tOrSFVt_copy_view(_, _, block_id));
  };

  auto quantize = [&](auto mma_k, auto& acc_conversion_view) {
    Tensor AbsMaxP_stagek = AbsMaxP(_, make_coord(_, _, mma_k));
    Tensor acc_conversion_stagek = acc_conversion_view(_, _, mma_k);
    Tensor SFP =
        make_tensor_like<cutlass::float_ue4m3_t>(AbsMaxP_stagek.layout());
    Tensor SFP_uint32_view = recast<uint32_t>(SFP);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(AbsMaxP_stagek); i += 4) {
      uint32_t& tmp = SFP_uint32_view(i / 4);
      packed_float_to_ue4m3(AbsMaxP_stagek(i), AbsMaxP_stagek(i + 1),
                            AbsMaxP_stagek(i + 2), AbsMaxP_stagek(i + 3), tmp);
    }
    int const quad_id = threadIdx.x & 3;
    uint32_t MASK = (0xFF00FF) << ((quad_id & 1) * 8);
    Tensor tOrSFP_uint32_view = recast<uint32_t>(tOrSFP(_, _, mma_k));
    Tensor tOrP_uint32_view = recast<uint32_t>(tOrP(_, _, mma_k));
    CUTLASS_PRAGMA_UNROLL
    for (int mma_m = 0; mma_m < size<1>(tOrP); ++mma_m) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < 4; ++i) {
        packed_float_to_e2m1(acc_conversion_stagek(make_coord(_0{}, i), mma_m),
                             acc_conversion_stagek(make_coord(_1{}, i), mma_m),
                             acc_conversion_stagek(make_coord(_2{}, i), mma_m),
                             acc_conversion_stagek(make_coord(_3{}, i), mma_m),
                             acc_conversion_stagek(make_coord(_4{}, i), mma_m),
                             acc_conversion_stagek(make_coord(_5{}, i), mma_m),
                             acc_conversion_stagek(make_coord(_6{}, i), mma_m),
                             acc_conversion_stagek(make_coord(_7{}, i), mma_m),
                             tOrP_uint32_view(i, mma_m));
      }
      uint32_t local_sfp = SFP_uint32_view(_0{}, _0{}, mma_m);
      uint32_t peer_sfp = __shfl_xor_sync(int32_t(-1), local_sfp, 2);
      if ((quad_id & 1) == 0) {
        uint32_t sfp = (local_sfp & MASK) | ((peer_sfp & MASK) << 8);
        tOrSFP_uint32_view(_0{}, mma_m) = sfp;
      } else {
        uint32_t sfp = (peer_sfp & MASK) | ((local_sfp & MASK) >> 8);
        tOrSFP_uint32_view(_0{}, mma_m) = sfp;
      }
    }
  };

  auto pv_gemm = [&](auto& tgt, int v_stg) {
    copy_v_block(_0{}, v_stg);
    quantize(_0{}, tSrS_conversion_view);
    CUTLASS_PRAGMA_UNROLL
    for (int v_block = 0; v_block < size<2>(tOrP); ++v_block) {
      cute::gemm(tiled_mma_pv,
                 make_zip_tensor(tOrP(_, _, v_block), tOrSFP(_, _, v_block)),
                 make_zip_tensor(tOrVt(_, _, v_block), tOrSFVt(_, _, v_block)),
                 tgt);
      if (v_block < size<2>(tOrP) - 1) {
        copy_v_block(v_block + 1, v_stg);
        quantize(v_block + 1, tSrS_conversion_view);
      } else {
        CtaBarrier::arrive(&v_empty[v_stg]);
      }
    }
  };

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

      copy_k_block(_0{}, k_stg);
      add_delta_s(tSrS, k_stg);
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tSrQ); ++k_block) {
        cute::gemm(tiled_mma_qk,
                   make_zip_tensor(tSrQ(_, _, k_block), tSrSFQ(_, _, k_block)),
                   make_zip_tensor(tSrK(_, _, k_block), tSrSFK(_, _, k_block)),
                   tSrS);
        if (k_block < size<2>(tSrQ) - 1) {
          copy_k_block(k_block + 1, k_stg);
        } else {
          CtaBarrier::arrive(&k_empty[k_stg]);
        }
      }

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

      if (kv_tile == 0)
        softmax_fused.template online_softmax_with_quant</*FirstTile=*/true,
                                                         /*InfCheck=*/true>(
            tSrS, AbsMaxP, softmax_scale_log2);
      else
        softmax_fused.template online_softmax_with_quant</*FirstTile=*/false,
                                                         /*InfCheck=*/true>(
            tSrS, AbsMaxP, softmax_scale_log2);

      TmaBarrier::wait(&v_full[v_stg], v_phase);
      cutlass::arch::fence_view_async_shared();

      if (kv_tile == 0) {
        pv_gemm(tOrO_store, v_stg);
      } else {
        // scores_scale == 1.0f exactly when the row max did not move this
        // tile (~96% of dense tiles): O = O*1 + O_new needs no rescale at
        // all. Warp-vote keeps both fragments on one uniform path.
        const bool need_rescale = softmax_fused.scores_scale[0] != 1.0f ||
                                  softmax_fused.scores_scale[1] != 1.0f;
        if (__any_sync(0xffffffff, need_rescale)) {
          Tensor tOrO = make_fragment_like(tOrO_store);
          clear(tOrO);
          pv_gemm(tOrO, v_stg);
          softmax_fused.rescale_o(tOrO_store, tOrO);
        } else {
          pv_gemm(tOrO_store, v_stg);
        }
      }
    }

    softmax_fused.finalize(tOrO_store);

    // Epilogue. qkm (lse correction) reads sQ back from smem, so it must run
    // before the O staging aliases q_base.
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

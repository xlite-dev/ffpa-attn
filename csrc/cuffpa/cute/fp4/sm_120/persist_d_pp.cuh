// NVFP4 FA-3 style ping-pong persist-D kernel for sm_120: 1x128T producer
// + 2x128T consumer warpgroups (setmaxnreg 24/240/240 = 504 regs), Q tile 64
// rows shared read-only by both consumers, kv tiles 64x64 split by parity
// (cid = tile & 1). The single-consumer kernel's serial chain
// QK -> softmax(MUFU.EX2) -> P quant -> PV has no warp-level cover
// (NCU: wait 1.07 dominant, tensor pipe 51%, XU 33% = MUFU.EX2 ~47% of a
// tile budget); here consumer A's softmax/quant overlaps consumer B's
// OMMA stream. Per-cid K stages (Sk=2) and single-buffered V: the V
// issue->wait window spans QK+softmax+quant of the same tile, K(n+1)'s
// spans softmax+quant+PV+rescale (release-as-soon-as-dead: k_empty after
// the QK LDSMs, v_empty after PV).
//
// The math chain, fragment adapters (LayoutP/LayoutSFP/add_delta_s slot
// arithmetic), masking and lse are inherited verbatim from the
// single-consumer persist_d.cuh; only the scheduling structure differs.
// qm/km stay at 128-row granularity (quant kernels untouched): the DS smem
// atom is a (128 row-broadcast, 64 col) tile so Q tiles 2j/2j+1 share one
// delta_s row block and the gmem layout matches the delta_s kernel.
//
// Epilogue split-KV merge (LeetCUDA flash_attn_3_tma_mma_ws_split_q
// pattern): cid1 scatters its fp32 O + (m0,m1,l0,l1) stats over the whole
// smem window (q_base.., >= 34KB), cid0 merges with exp2 max correction,
// normalizes by the merged row_sum, then does the shared epilogue (r2s +
// O TMA store + lse) alone; epilogue_done (count 256) gates the producer's
// next Q TMA (the merge buffer AND the O staging alias q_base).
//
// Reference: LeetCUDA kernels/interview/flash_attn.cuh
// flash_attn_3_tma_mma_ws_split_q_cute (barrier protocol, phase math,
// merge epilogue).
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
#include "persist_d.cuh"

namespace ffpa_fp4 {

using TmaBarrier = cutlass::arch::ClusterTransactionBarrier;
using CtaBarrier = cutlass::arch::ClusterBarrier;

// Split-M ping-pong traits: SM120 blockscaled 16x32x64 mma tiled 4x1x1
// (M4N1: 4 warps x 16 rows = 64 rows per consumer warpgroup). Each work
// covers a kWorkRows=128 Q tile; the two consumers each own one 64-row
// half and stream the FULL kv range independently (kv is issued twice, once
// per cid buffer — DRAM headroom is large). Works therefore stay at 128-row
// granularity: the earlier 64-row-work split-KV variant doubled per-work
// fixed cost (merge epilogue + q_full/epilogue_done round trips + lse/qkm
// traffic) and lost to the single-consumer kernel; ncu pinned the excess
// on LSU cycles (mbarrier spins + merge smem), not on tensor/XU.
template <typename ElementO_>
struct Fp4PersistDPPTraits {
  static constexpr int kBr = 64;       // mma rows per consumer
  static constexpr int kWorkRows = 128;  // q tile rows per work
  static constexpr int kBc = 128;
  static constexpr int kHeadDim = 128;
  static constexpr int kStagesK = 2;  // per consumer
  // V double buffer: the v_empty arrive follows the (async) tcgen05 smem
  // reads, and with a single buffer the producer's next V TMA can land
  // before those reads retire — intermittent NaN under many-work causal
  // schedules. The second stage widens the window by a full tile, same
  // margin the 3-stage single-consumer kernel enjoys.
  static constexpr int kStagesV = 1;  // per consumer

  using Element = cutlass::float_e2m1_t;
  using ElementSF = cutlass::float_ue4m3_t;
  using ElementO = ElementO_;

  using TileShape_MNK = Shape<_64, _128, _128>;
  using MMAAtom =
      MMA_Atom<cute::SM120::BLOCKSCALED::SM120_16x32x64_TN_VS_NVFP4>;
  using AtomLayoutMNK = Layout<Shape<_4, _1, _1>>;
  using TiledMmaQK = decltype(make_tiled_mma(MMAAtom{}, AtomLayoutMNK{},
                                             Tile<_64, _32, _128>{}));
  using TiledMmaPV = TiledMmaQK;

  using SmemLayoutAtomQK =
      decltype(cutlass::gemm::collective::detail::sm120_rr_smem_selector<
               Element, Int<kHeadDim>>());
  using SmemLayoutAtomVt = SmemLayoutAtomQK;  // kBc == kHeadDim
  using SmemLayoutQ = decltype(tile_to_shape(
      SmemLayoutAtomQK{},
      Shape<Int<kWorkRows>, Int<kHeadDim>>{}));
  // 64-row half of the Q tile (same column-stacked atom layout; the two
  // halves tile the full layout, so a half tensor at the cid byte offset
  // aliases the full tensor's rows [cid*64, cid*64+64)).
  using SmemLayoutQHalf = decltype(tile_to_shape(
      SmemLayoutAtomQK{}, Shape<Int<kBr>, Int<kHeadDim>>{}));
  using SmemLayoutK = decltype(tile_to_shape(
      SmemLayoutAtomQK{},
      make_shape(Int<kBc>{}, Int<kHeadDim>{}, Int<kStagesK>{})));
  using SmemLayoutVt = decltype(tile_to_shape(
      SmemLayoutAtomVt{},
      make_shape(Int<kHeadDim>{}, Int<kBc>{}, Int<kStagesV>{})));

  using BlkScaledConfig = BlockScaledConfig<16>;
  using SmemLayoutAtomSFQ = decltype(BlkScaledConfig::deduce_smem_layoutSFQ(
      TiledMmaQK{}, TileShape_MNK{}));
  using SmemLayoutAtomSFK = decltype(BlkScaledConfig::deduce_smem_layoutSFKV(
      TiledMmaQK{}, TileShape_MNK{}));
  using SmemLayoutAtomSFVt = decltype(BlkScaledConfig::deduce_smem_layoutSFVt(
      TiledMmaPV{}, Shape<Int<kBr>, Int<kHeadDim>, Int<kBc>>{}));
  using SmemLayoutSFQ = decltype(tile_to_shape(
      SmemLayoutAtomSFQ{}, Shape<Int<kWorkRows>, Int<kHeadDim>>{}));
  using SmemLayoutSFQHalf = decltype(tile_to_shape(
      SmemLayoutAtomSFQ{}, Shape<Int<kBr>, Int<kHeadDim>>{}));
  using SmemLayoutSFK =
      decltype(make_layout(append(shape(SmemLayoutAtomSFK{}), Int<kStagesK>{}),
                           append(stride(SmemLayoutAtomSFK{}),
                                  size(filter_zeros(SmemLayoutAtomSFK{})))));
  using SmemLayoutSFVt =
      decltype(make_layout(append(shape(SmemLayoutAtomSFVt{}), Int<kStagesV>{}),
                           append(stride(SmemLayoutAtomSFVt{}),
                                  size(filter_zeros(SmemLayoutAtomSFVt{})))));

  // qm stays 128-row granular: DS rows broadcast over the mb128 block, cols
  // tiled by kBc so the gmem layout matches the (unmodified) delta_s kernel.
  // Same hierarchical mode structure as tile_to_shape(128 x kBc atom) in the
  // single-consumer kernel: add_delta_s indexes each mode with a 2-tuple
  // coord ((tile, atom)), so flat modes would not compile. Only cosize=kBc
  // floats are actually stored (row mode is a stride-0 broadcast).
  using SmemLayoutDS = decltype(make_layout(
      make_shape(make_shape(_1{}, _128{}), make_shape(_1{}, Int<kBc>{}),
                 Int<kStagesK>{}),
      make_stride(make_stride(_0{}, _0{}), make_stride(_0{}, _1{}),
                  Int<kBc>{})));

  // P / SFP rmem fragment layouts: SA3 verbatim (parametric in kBc; the
  // kBc/64 mode is 1 at 64-col kv tiles).
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
  using SmemCopyAtomSF = Copy_Atom<UniversalCopy<ElementSF>, ElementSF>;

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

  // SMEM plan: [Q | SFQ | cid0: K*Sk SFK*Sk DS*Sk V SFVt | cid1: same],
  // regions padded to 1024B for the SW128 TMA destinations. In the
  // epilogue the whole window backs the split-KV merge buffer (fp32 O
  // 64x128 = 32KB + 128 threads x 16B stats) and then the O staging tile
  // (16KB), both aliasing q_base after the KV loop consumed everything.
  static constexpr int kQBytes =
      int(cute::bits_to_bytes(size(SmemLayoutQ{}) * 4));
  static constexpr int kQBytesHalf =
      int(cute::bits_to_bytes(size(SmemLayoutQHalf{}) * 4));
  static constexpr int kSFQBytes =
      int(cute::bits_to_bytes(cosize(SmemLayoutSFQ{}) * 8));
  static constexpr int kSFQBytesHalf =
      int(cute::bits_to_bytes(cosize(SmemLayoutSFQHalf{}) * 8));
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
  static constexpr int kCidBytes =
      kStagesK * (kKBytesStage + kSFKBytesStage + kDSBytesStage) +
      kStagesV * (kVBytesStage + kSFVtBytesStage);
  static constexpr int kOffQ = 0;
  static constexpr int kOffSFQ = kOffQ + kQBytes;
  static constexpr int kOffC0 = (kOffSFQ + kSFQBytes + 1023) / 1024 * 1024;
  static constexpr int kOffC1 = (kOffC0 + kCidBytes + 1023) / 1024 * 1024;
  static constexpr int kSmemBytes = (kOffC1 + kCidBytes + 1023) / 1024 * 1024;
  static_assert(kOffC0 % 1024 == 0 && kOffC1 % 1024 == 0, "SW smem alignment");
  static_assert(kSmemBytes <= 101376, "smem budget");
  static_assert(kWorkRows * kHeadDim * 2 <= kSmemBytes,
                "O staging must fit the freed smem");
};

// NVFP4 ping-pong persist-D forward. Same grid-scheduling contract as the
// single-consumer kernel (strided work loop, barriers never re-initialized;
// per-cid own-tile counters drive stage/phase across works).
template <typename Traits, typename ElementO, typename TmaQ, typename TmaK,
          typename TmaV, typename TmaO, typename TmaSFQ, typename TmaSFK,
          typename TmaSFVt, typename TmaDS>
__global__ void __launch_bounds__(384, 1) persist_d_ws_fwd_cute_fp4_sm120_pp(
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
  constexpr int kWorkRows = Traits::kWorkRows;
  constexpr int kBc = Traits::kBc;
  constexpr int kHeadDim = Traits::kHeadDim;
  constexpr int kStagesK = Traits::kStagesK;
  constexpr int kStagesV = Traits::kStagesV;
  constexpr int kNumConsumers = 2;
  constexpr int kProducerThreads = 128;
  constexpr int kConsumerThreads = 128;  // per warpgroup
  constexpr int kOffQ = Traits::kOffQ;
  constexpr int kOffSFQ = Traits::kOffSFQ;
  constexpr int kCidOffArr[2] = {Traits::kOffC0, Traits::kOffC1};
  auto kCidOff = [&](int cid) { return kCidOffArr[cid]; };
  constexpr int kSmemBytes = Traits::kSmemBytes;
  (void)kSmemBytes;

  const int group_size = Nh / Nh_kv;
  const int tid = threadIdx.x;
  const bool is_producer = tid < kProducerThreads;
  const int consumer_id =
      is_producer ? 0 : (tid - kProducerThreads) / kConsumerThreads;
  const int wg_tid =
      is_producer ? tid : (tid - kProducerThreads) % kConsumerThreads;

  // Work decomposition: 64-row Q tiles.
  const int MB = (Nq - q_start_row + kWorkRows - 1) / kWorkRows;
  const int total_work = MB * Nb * Nh;

  extern __shared__ __align__(1024) char shm[];
  Element* q_base = reinterpret_cast<Element*>(shm + kOffQ);

  __shared__ uint64_t q_full;
  __shared__ uint64_t k_full[kNumConsumers][kStagesK];
  __shared__ uint64_t k_empty[kNumConsumers][kStagesK];
  __shared__ uint64_t v_full[kNumConsumers][kStagesV];
  __shared__ uint64_t v_empty[kNumConsumers][kStagesV];
  __shared__ uint64_t epilogue_done;

  if (tid == 0) {
    TmaBarrier::init(&q_full, 1);
    for (int cid = 0; cid < kNumConsumers; ++cid) {
      for (int s = 0; s < kStagesK; ++s) {
        TmaBarrier::init(&k_full[cid][s], 1);
        CtaBarrier::init(&k_empty[cid][s], kConsumerThreads);
      }
      for (int s = 0; s < kStagesV; ++s) {
        for (int sv = 0; sv < kStagesV; ++sv) {
          TmaBarrier::init(&v_full[cid][sv], 1);
          CtaBarrier::init(&v_empty[cid][sv], kConsumerThreads);
        }
      }
    }
    CtaBarrier::init(&epilogue_done, 2 * kConsumerThreads);
  }
  __syncthreads();

  if (is_producer) {
    cutlass::arch::warpgroup_reg_dealloc<24>();
    if (wg_tid == 0) {
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
      // DS gmem: dense (B,H,Mb128,Nkv_pad) tensor, kv tiles kBc wide.
      auto layout_DS = tile_to_shape(
          Layout<Shape<_128, Int<kBc>>, Stride<_0, _1>>{},
          make_shape(Nq_pad, Nkv_pad, Nh, Nb), Step<_2, _1, _3, _4>{});
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
      auto make_cid_tensors = [&](int cid) {
        auto sK = make_tensor(make_smem_ptr<Element>(shm + kCidOff(cid) + 0),
                              SmemLayoutK{});
        auto sSFK = make_tensor(
            make_smem_ptr<ElementSF>(shm + kCidOff(cid) +
                                     kStagesK * Traits::kKBytesStage),
            SmemLayoutSFK{});
        auto sDS = make_tensor(
            make_smem_ptr<float>(
                shm + kCidOff(cid) +
                kStagesK * (Traits::kKBytesStage + Traits::kSFKBytesStage)),
            SmemLayoutDS{});
        auto sV = make_tensor(
            make_smem_ptr<Element>(shm + kCidOff(cid) +
                                   kStagesK * (Traits::kKBytesStage +
                                               Traits::kSFKBytesStage +
                                               Traits::kDSBytesStage)),
            SmemLayoutVt{});
        auto sSFVt = make_tensor(
            make_smem_ptr<ElementSF>(shm + kCidOff(cid) +
                                     kStagesK * (Traits::kKBytesStage +
                                                 Traits::kSFKBytesStage +
                                                 Traits::kDSBytesStage) +
                                     Traits::kVBytesStage),
            SmemLayoutSFVt{});
        return cute::make_tuple(sK, sSFK, sDS, sV, sSFVt);
      };
      auto [sK0, sSFK0, sDS0, sV0, sSFVt0] = make_cid_tensors(0);
      auto [sK1, sSFK1, sDS1, sV1, sSFVt1] = make_cid_tensors(1);

      auto tQsQ = q_slice.partition_D(sQ);
      auto tQsSFQ = sfq_slice.partition_D(sSFQ);
      auto make_cid_dst = [&](auto& sK, auto& sSFK, auto& sDS, auto& sV,
                              auto& sSFVt) {
        auto tKsK = group_modes<0, 3>(k_slice.partition_D(sK));
        auto tKsSFK = group_modes<0, 3>(sfk_slice.partition_D(sSFK));
        auto tVsV = group_modes<0, 3>(v_slice.partition_D(sV));
        auto tVsSFVt = group_modes<0, 3>(sfvt_slice.partition_D(sSFVt));
        auto tDSsDS = group_modes<0, 3>(ds_slice.partition_D(sDS));
        return cute::make_tuple(tKsK, tKsSFK, tVsV, tVsSFVt, tDSsDS);
      };
      auto [tKsK0, tKsSFK0, tVsV0, tVsSFVt0, tDSsDS0] =
          make_cid_dst(sK0, sSFK0, sDS0, sV0, sSFVt0);
      auto [tKsK1, tKsSFK1, tVsV1, tVsSFVt1, tDSsDS1] =
          make_cid_dst(sK1, sSFK1, sDS1, sV1, sSFVt1);

      // Per-cid own-tile counters drive stage/phase across works (never
      // re-initialized, PTX ISA 9.7.13.15.9).
      // Global issue counters per cid (never reset across works): every
      // mbarrier phase below is derived from them, mirroring the single
      // consumer kernel's global tile counter g.
      int own_seq_k[2] = {0, 0};  // K loads issued per cid
      int own_seq_v[2] = {0, 0};  // V loads issued per cid
      int w = 0;
      for (int work_id = blockIdx.x; work_id < total_work;
           work_id += gridDim.x, ++w) {
        const int kv_offset = Nkv - Nq;
        const int bh = work_id / MB;
        const int Q_tile_id = work_id % MB;
        const int b = bh / Nh;
        const int Nh_id = bh % Nh;
        const int kv_head_idx = Nh_id / group_size;
        const int q_tile_abs = Q_tile_id + q_start_row / kWorkRows;
        const int q_bh = bh;
        const int kv_bh = b * Nh_kv + kv_head_idx;
        const int q_row_offset = q_bh * Nq_pad + q_start_row;
        const int kv_row_offset = kv_bh * Nkv_pad;
        const int v_row_base = kv_bh * kHeadDim;

        auto gQ = local_tile(domain_offset(make_coord(q_row_offset, _0{}), mQ),
                             Shape<Int<kWorkRows>, Int<kHeadDim>>{},
                             make_coord(Q_tile_id, _0{}));
        auto gK =
            local_tile(domain_offset(make_coord(kv_row_offset, _0{}), mK),
                       Shape<Int<kBc>, Int<kHeadDim>>{}, make_coord(_, _0{}));
        auto gV =
            local_tile(domain_offset(make_coord(v_row_base, _0{}), mV),
                       Shape<Int<kHeadDim>, Int<kBc>>{}, make_coord(_0{}, _));
        auto gSFQ = local_tile(mSFQ(_, _, Nh_id, b),
                               Shape<Int<kWorkRows>, Int<kHeadDim>>{},
                               make_coord(q_tile_abs, _0{}));
        auto gSFK =
            local_tile(mSFK(_, _, kv_head_idx, b),
                       Shape<Int<kBc>, Int<kHeadDim>>{}, make_coord(_, _0{}));
        auto gSFVt =
            local_tile(mSFVt(_, _, kv_head_idx, b),
                       Shape<Int<kHeadDim>, Int<kBc>>{}, make_coord(_0{}, _));
        // DS rows follow the mb128 block (qm granularity): one 128-row
        // work tile == one qm block.
        auto gDS = local_tile(mDS(_, _, Nh_id, b), Shape<_128, Int<kBc>>{},
                              make_coord(q_tile_abs, _));

        auto tQgQ = q_slice.partition_S(gQ);
        auto tQgSFQ = sfq_slice.partition_S(gSFQ);
        auto tKgK = group_modes<0, 3>(k_slice.partition_S(gK));
        auto tKgSFK = group_modes<0, 3>(sfk_slice.partition_S(gSFK));
        auto tVgV = group_modes<0, 3>(v_slice.partition_S(gV));
        auto tVgSFVt = group_modes<0, 3>(sfvt_slice.partition_S(gSFVt));
        auto tDSgDS = group_modes<0, 3>(ds_slice.partition_S(gDS));

        const int Tc_eff =
            causal
                ? min(Tc, ((q_start_row + Q_tile_id * kWorkRows + kWorkRows -
                            1 + kv_offset) /
                           kBc) +
                              1)
                : Tc;
        // Both consumers stream the full tile range (split-M): kv tiles
        // are issued twice, once per cid buffer.
        const int own_tiles[2] = {Tc_eff, Tc_eff};

        // O staging/merge buffer aliases q_base: the previous work's
        // epilogue must be fully retired first.
        if (w > 0)
          CtaBarrier::wait(&epilogue_done, (w - 1) & 1);
        TmaBarrier::arrive_and_expect_tx(&q_full, Traits::kTxBytesQ);
        copy(tma_q.with(q_full), tQgQ, tQsQ);
        copy(tma_sfq.with(q_full), tQgSFQ, tQsSFQ);

        // Warmup: this work's first K tile per cid (kv tile 0) into the
        // stage indexed by the global own sequence.
        for (int cid = 0; cid < kNumConsumers; ++cid) {
          if (own_tiles[cid] > 0) {
            const int seq = own_seq_k[cid];
            const int stage = seq % kStagesK;
            const int phase = (seq / kStagesK) & 1;
            CtaBarrier::wait(&k_empty[cid][stage], phase);
            TmaBarrier::arrive_and_expect_tx(&k_full[cid][stage],
                                             Traits::kTxBytesK);
            if (cid == 0) {
              copy(tma_k.with(k_full[0][stage]), tKgK(_, 0), tKsK0(_, stage));
              copy(tma_sfk.with(k_full[0][stage]), tKgSFK(_, 0),
                   tKsSFK0(_, stage));
              copy(tma_ds.with(k_full[0][stage]), tDSgDS(_, 0),
                   tDSsDS0(_, stage));
            } else {
              copy(tma_k.with(k_full[1][stage]), tKgK(_, 0), tKsK1(_, stage));
              copy(tma_sfk.with(k_full[1][stage]), tKgSFK(_, 0),
                   tKsSFK1(_, stage));
              copy(tma_ds.with(k_full[1][stage]), tDSgDS(_, 0),
                   tDSsDS1(_, stage));
            }
            ++own_seq_k[cid];
          }
        }

        // Steady: per kv tile, both cids get K(tile+1) prefetch then V
        // (tile). All phases come from the global issue counters; the
        // prefetch target stays inside this work (same as the baseline's
        // tile + kStages - 1 bound).
        for (int tile = 0; tile < Tc_eff; ++tile) {
          for (int cid = 0; cid < kNumConsumers; ++cid) {
            {
              const int tile_p = tile + 1;
              if (tile_p < own_tiles[cid]) {
                const int seq = own_seq_k[cid];
                const int stage = seq % kStagesK;
                const int phase = (seq / kStagesK) & 1;
                CtaBarrier::wait(&k_empty[cid][stage], phase);
                TmaBarrier::arrive_and_expect_tx(&k_full[cid][stage],
                                                 Traits::kTxBytesK);
                if (cid == 0) {
                  copy(tma_k.with(k_full[0][stage]), tKgK(_, tile_p),
                       tKsK0(_, stage));
                  copy(tma_sfk.with(k_full[0][stage]), tKgSFK(_, tile_p),
                       tKsSFK0(_, stage));
                  copy(tma_ds.with(k_full[0][stage]), tDSgDS(_, tile_p),
                       tDSsDS0(_, stage));
                } else {
                  copy(tma_k.with(k_full[1][stage]), tKgK(_, tile_p),
                       tKsK1(_, stage));
                  copy(tma_sfk.with(k_full[1][stage]), tKgSFK(_, tile_p),
                       tKsSFK1(_, stage));
                  copy(tma_ds.with(k_full[1][stage]), tDSgDS(_, tile_p),
                       tDSsDS1(_, stage));
                }
                ++own_seq_k[cid];
              }
            }
            {
              const int vstage = own_seq_v[cid] % kStagesV;
              const int phase = (own_seq_v[cid] / kStagesV) & 1;
              CtaBarrier::wait(&v_empty[cid][vstage], phase);
              TmaBarrier::arrive_and_expect_tx(&v_full[cid][vstage],
                                               Traits::kTxBytesV);
              if (cid == 0) {
                copy(tma_v.with(v_full[0][vstage]), tVgV(_, tile),
                     tVsV0(_, vstage));
                copy(tma_sfvt.with(v_full[0][vstage]), tVgSFVt(_, tile),
                     tVsSFVt0(_, vstage));
              } else {
                copy(tma_v.with(v_full[1][vstage]), tVgV(_, tile),
                     tVsV1(_, vstage));
                copy(tma_sfvt.with(v_full[1][vstage]), tVgSFVt(_, tile),
                     tVsSFVt1(_, vstage));
              }
              ++own_seq_v[cid];
            }
          }
        }
      }
    }
    return;
  }

  // Consumers
  cutlass::arch::warpgroup_reg_alloc<240>();
  const int cid = consumer_id;
  for (int s = 0; s < kStagesK; ++s)
    CtaBarrier::arrive(&k_empty[cid][s]);
  for (int s = 0; s < kStagesV; ++s)
    CtaBarrier::arrive(&v_empty[cid][s]);

  TiledMmaQK tiled_mma_qk;
  TiledMmaPV tiled_mma_pv;
  auto thread_mma_qk = tiled_mma_qk.get_thread_slice(wg_tid);
  auto thread_mma_pv = tiled_mma_pv.get_thread_slice(wg_tid);

  // Full-tile (128-row) Q/SFQ feed the TMA; each consumer works on its
  // 64-row half — a half-layout tensor at the cid byte offset (the halves
  // tile the full column-stacked layout).
  auto sQ = make_tensor(
      make_smem_ptr<Element>(q_base + cid * Traits::kQBytesHalf),
      typename Traits::SmemLayoutQHalf{});
  auto sSFQ = make_tensor(
      make_smem_ptr<ElementSF>(
          reinterpret_cast<ElementSF*>(shm + kOffSFQ) + cid * Traits::kSFQBytesHalf),
      typename Traits::SmemLayoutSFQHalf{});
  auto sK = make_tensor(make_smem_ptr<Element>(shm + kCidOff(cid) + 0),
                        SmemLayoutK{});
  auto sSFK =
      make_tensor(make_smem_ptr<ElementSF>(shm + kCidOff(cid) +
                                           kStagesK * Traits::kKBytesStage),
                  SmemLayoutSFK{});
  auto sDS =
      make_tensor(make_smem_ptr<float>(shm + kCidOff(cid) +
                                       kStagesK * (Traits::kKBytesStage +
                                                   Traits::kSFKBytesStage)),
                  SmemLayoutDS{});
  auto sV =
      make_tensor(make_smem_ptr<Element>(shm + kCidOff(cid) +
                                         kStagesK * (Traits::kKBytesStage +
                                                     Traits::kSFKBytesStage +
                                                     Traits::kDSBytesStage)),
                  SmemLayoutVt{});
  auto sSFVt =
      make_tensor(make_smem_ptr<ElementSF>(shm + kCidOff(cid) +
                                           kStagesK * (Traits::kKBytesStage +
                                                       Traits::kSFKBytesStage +
                                                       Traits::kDSBytesStage) +
                                           Traits::kVBytesStage),
                  SmemLayoutSFVt{});

  Tensor tSrQ = thread_mma_qk.partition_fragment_A(sQ);
  Tensor tSrK = thread_mma_qk.partition_fragment_B(sK(_, _, Int<0>{}));
  Tensor tOrVt = thread_mma_pv.partition_fragment_B(sV(_, _, Int<0>{}));
  Tensor tSrSFQ = partition_fragment_SFA(sSFQ, thread_mma_qk);
  Tensor tSrSFK = partition_fragment_SFB(sSFK(_, _, Int<0>{}), thread_mma_qk);
  Tensor tOrSFVt = partition_fragment_SFB(sSFVt(_, _, Int<0>{}), thread_mma_pv);
  Tensor tOrP = make_tensor_like<Element>(typename Traits::LayoutP{});
  Tensor tOrSFP = make_tensor<ElementSF>(typename Traits::LayoutSFP{});

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

  // delta_s broadcast add: kBc=128 kv columns = 32 float4 slots; the SA3
  // verbatim quad-pair addressing (identical to the single-consumer kernel).
  auto add_delta_s = [&](auto& acc, int stage) {
    auto tSsDS_stage = recast<float4>(sDS(_, _, stage));
    auto acc_float4 = recast<float4>(acc);
    int quad_id = (wg_tid % 4) * 2;
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
    int const quad_id = wg_tid & 3;
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

  // PV accumulates straight into o_store (no O_tmp): the rescale pass runs
  // right after softmax, before PV. In the dual-consumer schedule the
  // other warpgroup's OMMA stream covers the rescale FMULs, which is what
  // made direct accumulation a small regression in the single-consumer
  // kernel. Saves the 64-reg temporary.
  auto pv_gemm = [&](auto& tgt, int stage) {
    copy_v_block(_0{}, stage);
    quantize(_0{}, tSrS_conversion_view);
    CUTLASS_PRAGMA_UNROLL
    for (int v_block = 0; v_block < size<2>(tOrP); ++v_block) {
      cute::gemm(tiled_mma_pv,
                 make_zip_tensor(tOrP(_, _, v_block), tOrSFP(_, _, v_block)),
                 make_zip_tensor(tOrVt(_, _, v_block), tOrSFVt(_, _, v_block)),
                 tgt);
      if (v_block < size<2>(tOrP) - 1) {
        copy_v_block(v_block + 1, stage);
        quantize(v_block + 1, tSrS_conversion_view);
      } else {
        CtaBarrier::arrive(&v_empty[cid][stage]);
      }
    }
  };

  auto rescale_o_store = [&]() {
    Tensor o_store_reduction_view = make_tensor(
        tOrO_store.data(), convert_to_reduction_layout(tOrO_store.layout()));
    CUTLASS_PRAGMA_UNROLL
    for (int mi = 0; mi < kSoftmaxRows; ++mi)
      CUTLASS_PRAGMA_UNROLL
    for (int ni = 0; ni < size<1>(o_store_reduction_view); ni++)
      o_store_reduction_view(mi, ni) *= softmax_fused.scores_scale(mi);
  };

  int own = 0;  // own-tile counter (drives stage/phase across works)
  int w = 0;
  for (int work_id = blockIdx.x; work_id < total_work;
       work_id += gridDim.x, ++w) {
    const int kv_offset = Nkv - Nq;
    const int bh = work_id / MB;
    const int Q_tile_id = work_id % MB;
    const int Nb_id = bh / Nh;
    const int Nh_id = bh % Nh;
    const int kv_head_idx = Nh_id / group_size;
    // This WG's 64-row window inside the 128-row work tile.
    const int Br_base = Q_tile_id * kWorkRows + cid * kBr;
    const int causal_thresh_row0 = q_start_row + Br_base + kv_offset;
    const int Tc_eff =
        causal
            ? min(Tc, ((q_start_row + Q_tile_id * kWorkRows + kWorkRows - 1 +
                        kv_offset) /
                       kBc) +
                          1)
            : Tc;
    const int mask_start_tile =
        causal ? max(0, (causal_thresh_row0 + 1) / kBc) : INT_MAX;
    const int q_bh = bh;
    const int kv_bh = Nb_id * Nh_kv + kv_head_idx;
    const int q_tile_abs = Q_tile_id + q_start_row / kWorkRows;
    const int O_row_offset = q_bh * Nq + q_start_row;
    const int own_tiles = Tc_eff;  // split-M: full range per consumer

    if (w > 0) {
      TmaBarrier::wait(&q_full, w & 1);
      cutlass::arch::fence_view_async_shared();
    }

    clear(tOrO_store);

#pragma unroll 1
    for (int L = 0; L < own_tiles; ++L, ++own) {
      const int kv_tile = L;
      const int k_stg = own % kStagesK;
      const int k_phase = (own / kStagesK) & 1;
      const int v_stg = own % kStagesV;
      const int v_phase = (own / kStagesV) & 1;

      TmaBarrier::wait(&k_full[cid][k_stg], k_phase);
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
          CtaBarrier::arrive(&k_empty[cid][k_stg]);
        }
      }

      // Masking: kv-tail + causal, perm-aware (kv_perm32), identical to
      // the single-consumer kernel but keyed on this WG's own tile index.
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

      if (L == 0)
        softmax_fused.template online_softmax_with_quant</*FirstTile=*/true,
                                                         /*InfCheck=*/true>(
            tSrS, AbsMaxP, softmax_scale_log2);
      else
        softmax_fused.template online_softmax_with_quant</*FirstTile=*/false,
                                                         /*InfCheck=*/true>(
            tSrS, AbsMaxP, softmax_scale_log2);

      if (L > 0)
        rescale_o_store();

      TmaBarrier::wait(&v_full[cid][v_stg], v_phase);
      cutlass::arch::fence_view_async_shared();

      pv_gemm(tOrO_store, v_stg);
    }

    // Split-M epilogue: rows are consumer-private over the full kv range,
    // so no cross-WG merge — finalize() reduces the per-thread partial
    // row_sum over the quad and normalizes o_store.
    softmax_fused.finalize(tOrO_store);

    float qkm[kSRows];
    const bool smooth_lse =
        (softmax_lse != nullptr) && (km != nullptr) && (qm != nullptr);
    if (smooth_lse) {
      const float* km_bh = km + static_cast<long>(kv_bh) * kHeadDim;
      const long qm_mb = Nq_pad / 128;  // qm stays 128-row granular
      const float* qm_blk =
          qm + (static_cast<long>(q_bh) * qm_mb + q_tile_abs) * kHeadDim;
      lse_qkm_dot<kHeadDim, kSRows>(sQ, sSFQ, tScS_rc, km_bh, qm_blk, qkm);
    }
    if (softmax_lse != nullptr) {
      const int lse_base = Nb_id * Nh * Nq + Nh_id * Nq;
      CUTLASS_PRAGMA_UNROLL
      for (int row = 0; row < kSRows; ++row) {
        float lse = (softmax_fused.row_max[row] * softmax_scale_log2 +
                     log2f(softmax_fused.row_sum[row]) +
                     SoftmaxFused<kSoftmaxRows>::fp8_scalexfp4_scale_log2) *
                    FFPA_M_LN2;
        if (smooth_lse)
          lse += scale_orig * qkm[row];
        const int global_row =
            q_start_row + Br_base + cute::get<0>(tScS_rc(row, 0));
        // one lane per quad writes; the 4 quad lanes hold identical sums
        if (global_row < Nq && (wg_tid & 3) == 0)
          softmax_lse[lse_base + global_row] = lse;
      }
    }

    // Epilogue ordering: the two WGs run unordered, and any 16KB staging
    // window overlaps SOME live K/V smem of the peer (its kv loop may
    // still be draining). One 256-thread named barrier after qkm retires
    // every K/V/SF read (mma.sync + LDSM are warp-synchronous, and qkm is
    // the last sQ/sSFQ reader); from here on the whole window is dead
    // until the producer's next TMA (epilogue_done).
    cutlass::arch::NamedBarrier::sync(2 * kConsumerThreads, 0);

    // O staging: cid-partitioned 16KB windows at q_base.
    auto tCrOHalf = ffpa_cute::convert_type<ElementO>(tOrO_store);
    if (Br_base + kBr <= Nq - q_start_row) {
      auto sO = as_position_independent_swizzle_tensor(make_tensor(
          make_smem_ptr(reinterpret_cast<ElementO*>(q_base) +
                        cid * (kBr * kHeadDim)),
          SmemLayoutO{}));
      auto r2s_copy = make_tiled_copy_C(
          Copy_Atom<SM90_U32x2_STSM_N, ElementO>{}, tiled_mma_pv);
      auto r2s_thr = r2s_copy.get_thread_slice(wg_tid);
      auto tCrOHalf_src = r2s_thr.retile_S(tCrOHalf);
      auto tCsO_dst = r2s_thr.partition_D(sO);
      copy(r2s_copy, tCrOHalf_src, tCsO_dst);
      cutlass::arch::fence_view_async_shared();
      cutlass::arch::NamedBarrier::sync(kConsumerThreads, 1);

      auto mO_tma = domain_offset(
          make_coord(O_row_offset, 0),
          tma_o.get_tma_tensor(
              make_shape((long)total_q_rows, Int<kHeadDim>{})));
      auto o_slice = tma_o.get_slice(_0{});
      auto gO_tma = local_tile(mO_tma, Shape<Int<kBr>, Int<kHeadDim>>{},
                               make_coord(Q_tile_id * 2 + cid, _0{}));
      auto tCgO_tma = o_slice.partition_D(gO_tma);
      auto tOsO = o_slice.partition_S(sO);
      if (wg_tid == 0)
        copy(tma_o, tOsO, tCgO_tma);
      tma_store_arrive();
      tma_store_wait<0>();
    } else {
      // Tail tile: rows past Nq would alias the next head in the flattened
      // [total_q_rows, D] TMA space, so store R->G with a row guard.
      const int O_gmem_offset =
          (q_bh)*Nq * kHeadDim + q_start_row * kHeadDim;
      auto mO = make_tensor(make_gmem_ptr(O + O_gmem_offset),
                            make_shape(Nq - q_start_row, Int<kHeadDim>{}),
                            make_stride(Int<kHeadDim>{}, _1{}));
      auto gO = local_tile(mO, Shape<Int<kBr>, Int<kHeadDim>>{},
                           make_coord(Q_tile_id * 2 + cid, _0{}));
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

    // Release q_base for the next work's Q TMA (O staging aliases it).
    // Both warpgroups arrive.
    CtaBarrier::arrive(&epilogue_done);
  }
#endif  // defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
}

}  // namespace ffpa_fp4

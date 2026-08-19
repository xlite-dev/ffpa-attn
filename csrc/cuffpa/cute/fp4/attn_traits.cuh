// NVFP4 attention kernel traits, split out of cute/fp4/sm_120/persist_d.cuh
// and renamed to match the fp8/fp16 naming (FFPAAttnCuTePersistDFP8Traits in
// cute/fp8/attn_traits.cuh). Lives at the fp4/ level so a future split-D fp4
// family can share it. The NVFP4 blockscaled MMA atom and the SF smem
// layouts come from cute_ext.h (same directory).
#pragma once

#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/collective/collective_builder.hpp>

#include "../attn_traits.cuh"
#include "cute_ext.h"

namespace ffpa_fp4 {

// NVFP4 persist-D traits, D in {64,128,192,256} (64-multiples only: the
// blockscaled SF atom is 64-wide along both MN and K): SM120 blockscaled
// 16x32x64 mma
// (4x mma.sync m16n8k64 kind::mxf4nvf4 ue4m3 scale_vec::4X) tiled 8x1x1 over
// a (128, 32, kHeadDim) tile for both QK and PV. Q/K smem use the
// sm120_rr swizzle atom selected by kHeadDim and V^T a separate one
// selected by kBc; SF smem uses the BlockScaledConfig atom
// layouts; DS (delta_s) is a stride-(0,1) 128-float broadcast tile.
template <typename ElementO_, int kHeadDim_>
struct FFPAAttnCuTePersistDFP4Traits {
  static constexpr int kBr = 128;
  static constexpr int kBc = 128;
  static constexpr int kHeadDim = kHeadDim_;
  // D=256 at 3 stages needs 130,560B, past the 99KB sm_120 opt-in budget.
  static constexpr int kStages = (kHeadDim <= 192) ? 3 : 2;
  static_assert(kHeadDim % 64 == 0 && kHeadDim >= 64 && kHeadDim <= 256,
                "fp4 persist_d supports D in {64,128,192,256}");

  using Element = cutlass::float_e2m1_t;
  using ElementSF = cutlass::float_ue4m3_t;
  using ElementO = ElementO_;

  using TileShape_MNK = Shape<_128, _128, Int<kHeadDim>>;
  using MMAAtom =
      MMA_Atom<cute::SM120::BLOCKSCALED::SM120_16x32x64_TN_VS_NVFP4>;
  using AtomLayoutMNK = Layout<Shape<_8, _1, _1>>;
  // Tile-K must equal the GEMM's K extent: kHeadDim for QK, kBc (KV
  // tokens) for PV. SA3 hardcodes PermTileK=kHeadDim for both, which only
  // holds because all its configs have kBc==kHeadDim==128 (its D=64 branch
  // never compiled). A mismatched Tile-K breaks partition_fragment_B's
  // logical_divide (192 % 128 != 0 at D=192).
  using TiledMmaQK = decltype(make_tiled_mma(MMAAtom{}, AtomLayoutMNK{},
                                             Tile<_128, _32, Int<kHeadDim>>{}));
  using TiledMmaPV = decltype(make_tiled_mma(MMAAtom{}, AtomLayoutMNK{},
                                             Tile<_128, _32, Int<kBc>>{}));

  // Q/K contiguous extent is kHeadDim elements, V^T's is kBc (SA3 splits
  // the selector the same way); at D=128 both select SW64, so the D=128
  // layout is bit-identical to the pre-templated version.
  using SmemLayoutAtomQK =
      decltype(cutlass::gemm::collective::detail::sm120_rr_smem_selector<
               Element, Int<kHeadDim>>());
  using SmemLayoutAtomVt =
      decltype(cutlass::gemm::collective::detail::sm120_rr_smem_selector<
               Element, Int<kBc>>());
  using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQK{},
                                             Shape<Int<kBr>, Int<kHeadDim>>{}));
  using SmemLayoutK = decltype(tile_to_shape(
      SmemLayoutAtomQK{},
      make_shape(Int<kBc>{}, Int<kHeadDim>{}, Int<kStages>{})));
  using SmemLayoutVt = decltype(tile_to_shape(
      SmemLayoutAtomVt{},
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

  // SMEM plan: [Q | SFQ | K*s | SFK*s | DS*s | V^T*s | SFVt*s], every region
  // start padded to 1024B so all TMA destinations stay swizzle-span aligned
  // at any D (kOffK is only naturally aligned at D in {128,256}). At D=128
  // the padding is a no-op: offsets match the pre-templated layout exactly.
  // The O staging tile (kBr*kHeadDim*2B) aliases q_base in the epilogue,
  // after the KV loop has consumed everything below kSmemBytes.
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
  static constexpr int kOffSFQ = (kQBytes + 1023) / 1024 * 1024;
  static constexpr int kOffK = (kOffSFQ + kSFQBytes + 1023) / 1024 * 1024;
  static constexpr int kOffSFK =
      (kOffK + kStages * kKBytesStage + 1023) / 1024 * 1024;
  static constexpr int kOffDS =
      (kOffSFK + kStages * kSFKBytesStage + 1023) / 1024 * 1024;
  static constexpr int kOffV =
      (kOffDS + kStages * kDSBytesStage + 1023) / 1024 * 1024;
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

// NVFP4 split-D traits: headdims in (256, 768) (64-multiples), the regime
// where the persist-D smem plan (full [kBc, D] K/V^T tiles) no longer fits
// the 99KB opt-in budget. Same blockscaled 16x32x64 atom and kBr/kBc as
// persist-D, but K and V^T stream through smem in 64-element D chunks
// (kQKDChunk = the atom K extent; kVDChunk = 2 atom-N) while Q/SFQ stay
// resident for the whole work. The kernel is non-WS 256T (tid==0 issues
// TMA inline): the O accumulator alone is kBr*D/256 = D/2 f32 regs/thread,
// already past the 255 wall at D=512, so a producer warp split (and its
// setmaxnreg 232 cap) only adds spill - fp8's split_d made the same call.
// O epilogue stages over the whole freed smem in kVChunksPerBatch batches.
template <typename ElementO_, int kHeadDim_, int kBr_ = 128, int kBc_ = 128,
          int kQKDChunk_ = 64, int kVDChunk_ = 64, int kStagesQK_ = 3,
          int kStagesPV_ = 3>
struct FFPAAttnCuTeSplitDFP4Traits {
  static_assert(kHeadDim_ % 64 == 0 && kHeadDim_ > 256 && kHeadDim_ < 768,
                "fp4 split_d supports 64-multiple D in (256,768)");
  static_assert(kQKDChunk_ == 64, "kQKDChunk must equal the blockscale atom K");
  static_assert(kVDChunk_ % 32 == 0, "kVDChunk must be a multiple of atom N");

  static constexpr int kHeadDim = kHeadDim_;
  static constexpr int kBr = kBr_;
  static constexpr int kBc = kBc_;
  static constexpr int kQKDChunk = kQKDChunk_;
  static constexpr int kVDChunk = kVDChunk_;
  static constexpr int kDChunksQK = kHeadDim / kQKDChunk;
  static constexpr int kDChunksV = kHeadDim / kVDChunk;
  static constexpr int kStagesQK = kStagesQK_;
  static constexpr int kStagesPV = kStagesPV_;
  static constexpr int kNumWarps = kBr / 16;
  static constexpr int kNumThreads = kNumWarps * 32;

  using Element = cutlass::float_e2m1_t;
  using ElementSF = cutlass::float_ue4m3_t;
  using ElementO = ElementO_;

  // Tile-K per gemm call: 64 (one QK d_chunk) for QK, kBc (token extent,
  // unchanged by D chunking) for PV - the SA3 Tile-K == K-extent discipline.
  using MMAAtom =
      MMA_Atom<cute::SM120::BLOCKSCALED::SM120_16x32x64_TN_VS_NVFP4>;
  using AtomLayoutMNK = Layout<Shape<_8, _1, _1>>;
  using TiledMmaQK = decltype(make_tiled_mma(
      MMAAtom{}, AtomLayoutMNK{}, Tile<_128, _32, Int<kQKDChunk>>{}));
  using TiledMmaPV = decltype(make_tiled_mma(MMAAtom{}, AtomLayoutMNK{},
                                             Tile<_128, _32, Int<kBc>>{}));

  // Q/K chunk rows are 64 e2m1 elements (32B) -> SW32; V^T rows are kBc=128
  // elements (64B) -> SW64 (persist_d picks the same V^T atom). The 64-elem
  // Q atom keeps every d_chunk a self-contained swizzle span.
  using SmemLayoutAtomQK =
      decltype(cutlass::gemm::collective::detail::sm120_rr_smem_selector<
               Element, Int<kQKDChunk>>());
  using SmemLayoutAtomVt =
      decltype(cutlass::gemm::collective::detail::sm120_rr_smem_selector<
               Element, Int<kBc>>());
  using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQK{},
                                             Shape<Int<kBr>, Int<kHeadDim>>{}));
  using SmemLayoutK = decltype(tile_to_shape(
      SmemLayoutAtomQK{},
      make_shape(Int<kBc>{}, Int<kQKDChunk>{}, Int<kStagesQK>{})));
  using SmemLayoutVt = decltype(tile_to_shape(
      SmemLayoutAtomVt{},
      make_shape(Int<kVDChunk>{}, Int<kBc>{}, Int<kStagesPV>{})));
  // Per-stage rank-2 views (byte-identical to the stage slices of the 3D
  // layouts above; the TMA issue path addresses them by byte offset).
  using SmemLayoutKStage = decltype(tile_to_shape(
      SmemLayoutAtomQK{}, Shape<Int<kBc>, Int<kQKDChunk>>{}));
  using SmemLayoutVtStage = decltype(tile_to_shape(
      SmemLayoutAtomVt{}, Shape<Int<kVDChunk>, Int<kBc>>{}));

  using BlkScaledConfig = BlockScaledConfig<16>;
  // SFQ covers the full-D resident tile (deduce reads M and K modes);
  // SFK/SFVt are per-chunk stage atoms.
  using SmemLayoutAtomSFQ = decltype(BlkScaledConfig::deduce_smem_layoutSFQ(
      TiledMmaQK{}, Shape<_128, _32, Int<kHeadDim>>{}));
  using SmemLayoutAtomSFK = decltype(BlkScaledConfig::deduce_smem_layoutSFKV(
      TiledMmaQK{}, Shape<_128, _128, Int<kQKDChunk>>{}));
  using SmemLayoutAtomSFVt = decltype(BlkScaledConfig::deduce_smem_layoutSFVt(
      TiledMmaPV{}, Shape<_128, Int<kVDChunk>, Int<kBc>>{}));
  using SmemLayoutSFQ = decltype(make_layout(shape(SmemLayoutAtomSFQ{}),
                                             stride(SmemLayoutAtomSFQ{})));
  using SmemLayoutSFK =
      decltype(make_layout(append(shape(SmemLayoutAtomSFK{}), Int<kStagesQK>{}),
                           append(stride(SmemLayoutAtomSFK{}),
                                  size(filter_zeros(SmemLayoutAtomSFK{})))));
  using SmemLayoutSFVt = decltype(make_layout(
      append(shape(SmemLayoutAtomSFVt{}), Int<kStagesPV>{}),
      append(stride(SmemLayoutAtomSFVt{}),
             size(filter_zeros(SmemLayoutAtomSFVt{})))));

  using SmemLayoutAtomDS = Layout<Shape<_128, _128>, Stride<_0, _1>>;
  using SmemLayoutDS = decltype(tile_to_shape(
      SmemLayoutAtomDS{},
      make_shape(Int<kBr>{}, Int<kBc>{}, Int<kStagesQK>{})));
  using SmemLayoutSFKStage = decltype(make_layout(shape(SmemLayoutAtomSFK{}),
                                                  stride(SmemLayoutAtomSFK{})));
  using SmemLayoutSFVtStage = decltype(make_layout(
      shape(SmemLayoutAtomSFVt{}), stride(SmemLayoutAtomSFVt{})));
  using SmemLayoutDSStage =
      decltype(tile_to_shape(SmemLayoutAtomDS{}, Shape<Int<kBr>, Int<kBc>>{}));

  // P / SFP register adapters: identical to persist-D (kBc unchanged).
  using LayoutP = decltype(make_layout(
      make_shape(make_shape(_8{}, _2{}, _2{}), _1{}, Int<kBc / 64>{}),
      make_stride(make_stride(_1{}, _8{}, _16{}), _0{}, _32{})));
  using LayoutSFP = decltype(make_layout(
      make_shape(make_shape(_16{}, _4{}), _1{}, Int<kBc / 64>{}),
      make_stride(make_stride(_0{}, _1{}), _0{}, _4{})));

  // O staging per v_chunk: [kBr, kVDChunk] ElementO, 128B rows -> SW128.
  using SmemLayoutO = decltype(tile_to_shape(
      GMMA::Layout_K_SW128_Atom<ElementO>{}, Shape<Int<kBr>, Int<kVDChunk>>{}));

  using SmemCopyAtomQ = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
  using SmemCopyAtomKV = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
  using SmemCopyAtomSF = Copy_Atom<UniversalCopy<ElementSF>, ElementSF>;

  // TMA tx bytes: Q/SFQ once per work (resident); K/SFK/DS and V/SFVt per
  // chunk stage. DS rides with the kv_tile's first chunk's barrier.
  static constexpr uint32_t kTxBytesQ =
      static_cast<uint32_t>(cute::bits_to_bytes(cosize(SmemLayoutSFQ{}) * 8)) +
      static_cast<uint32_t>(cute::bits_to_bytes(size(SmemLayoutQ{}) * 4));
  static constexpr uint32_t kTxBytesK =
      static_cast<uint32_t>(
          cute::bits_to_bytes(cosize(take<0, 2>(SmemLayoutSFK{})) * 8)) +
      static_cast<uint32_t>(
          cute::bits_to_bytes(size(take<0, 2>(SmemLayoutK{})) * 4));
  static constexpr uint32_t kTxBytesDS = static_cast<uint32_t>(
      cute::bits_to_bytes(cosize(take<0, 2>(SmemLayoutDS{})) * 32));
  static constexpr uint32_t kTxBytesV =
      static_cast<uint32_t>(
          cute::bits_to_bytes(cosize(take<0, 2>(SmemLayoutSFVt{})) * 8)) +
      static_cast<uint32_t>(
          cute::bits_to_bytes(size(take<0, 2>(SmemLayoutVt{})) * 4));

  // SMEM plan: [Q | SFQ | K*s | SFK*s | DS*s | V^T*s | SFVt*s] - persist_d's
  // ordering with Q/SFQ resident (no stage dim) and the rest chunk-staged;
  // every region start 1024B-aligned.
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
  static constexpr int kOffSFQ = (kQBytes + 1023) / 1024 * 1024;
  static constexpr int kOffK = (kOffSFQ + kSFQBytes + 1023) / 1024 * 1024;
  static constexpr int kOffSFK =
      (kOffK + kStagesQK * kKBytesStage + 1023) / 1024 * 1024;
  static constexpr int kOffDS =
      (kOffSFK + kStagesQK * kSFKBytesStage + 1023) / 1024 * 1024;
  static constexpr int kOffV =
      (kOffDS + kStagesQK * kDSBytesStage + 1023) / 1024 * 1024;
  static constexpr int kOffSFVt = kOffV + kStagesPV * kVBytesStage;
  static constexpr int kSmemBytes = kOffSFVt + kStagesPV * kSFVtBytesStage;
  static_assert(kOffK % 1024 == 0 && kOffV % 1024 == 0, "SW128 smem alignment");
  // sm_120 opt-in budget (see persist_d traits; exceeding fails silently).
  static_assert(kSmemBytes <= 101376, "smem budget");

  // O epilogue batches over the whole freed smem (during the epilogue no
  // K/V chunk TMA is in flight: the lookahead is work-bounded and the next
  // work's fills wait on epilogue_done).
  static constexpr int kVChunksPerBatch = ffpa_cute::compute_vchunks_per_batch(
      kDChunksV, kHeadDim, kBr, kSmemBytes / int(sizeof(ElementO)));
  static constexpr int kNBatches = kDChunksV / kVChunksPerBatch;
  static_assert(kVChunksPerBatch * cosize(SmemLayoutO{}) * sizeof(ElementO) <=
                    kSmemBytes,
                "TMA-O: batched O staging must fit the reused smem");
};

}  // namespace ffpa_fp4

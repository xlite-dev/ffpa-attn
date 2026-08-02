#pragma once

#include "common.cuh"
#include "gemm.cuh"

#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cutlass/arch/barrier.h>

namespace ffpa_cute {
using namespace cute;

// Largest v_chunks-per-batch for the TMA-O epilogue: pick the smallest
// n_batches (divisor of kDChunksV) such that [kBr, kHeadDim/n_batches] fits in
// kSmemElems.
constexpr int compute_vchunks_per_batch(int kDChunksV, int kHeadDim, int kBr,
                                        int kSmemElems) {
  for (int n = 1; n <= kDChunksV; ++n) {
    if (kDChunksV % n != 0)
      continue;
    if (kBr * (kHeadDim / n) <= kSmemElems)
      return kDChunksV / n;
  }
  return 1;
}

template <int kChunk, typename Element>
struct SelectSmemAtom {
  using type = GMMA::Layout_K_SW128_Atom<Element>;
};

template <typename Element>
struct SelectSmemAtom<32, Element> {
  using type = GMMA::Layout_K_SW64_Atom<Element>;
};

template <typename Element>
struct SelectSmemAtom<16, Element> {
  using type = GMMA::Layout_K_SW32_Atom<Element>;
};

template <int kHeadDim_, int kBr_ = 64, int kBc_ = 64, int kQKDChunk_ = 64,
          int kVDChunk_ = 64, int kStagesQK_ = 2, int kStagesPV_ = 2,
          typename Element_ = cutlass::half_t>
struct FFPAAttnCuTeTraits {
  static_assert(kHeadDim_ % kQKDChunk_ == 0);
  static_assert(kHeadDim_ % kVDChunk_ == 0);
  static_assert(kQKDChunk_ == 16 || kQKDChunk_ == 32 || kQKDChunk_ == 64);
  static_assert(kVDChunk_ == 16 || kVDChunk_ == 32 || kVDChunk_ == 64);

  static constexpr int kHeadDim = kHeadDim_;
  static constexpr int kBr = kBr_;
  static constexpr int kBc = kBc_;
  static constexpr int kQKDChunk = kQKDChunk_;
  static constexpr int kVDChunk = kVDChunk_;
  static constexpr int kDChunksQK = kHeadDim / kQKDChunk;
  static constexpr int kDChunksV = kHeadDim / kVDChunk;
  static constexpr int kNumWarps = kBr / 16;
  static constexpr int kNumThreads = kNumWarps * 32;
  static constexpr int kStagesQK = kStagesQK_;
  static constexpr int kStagesPV = kStagesPV_;
  static constexpr int kSmemElems = kStagesQK * kBr * kQKDChunk +
                                    kStagesQK * kBc * kQKDChunk +
                                    kStagesPV * kBc * kVDChunk;
  static constexpr int kVChunksPerBatch =
      compute_vchunks_per_batch(kDChunksV, kHeadDim, kBr, kSmemElems);
  static constexpr int kNBatches = kDChunksV / kVChunksPerBatch;
  // FA-4 conditional rescaling threshold (log2 domain), shared via macro.
  static constexpr float kRescaleThreshold = FFPA_RESCALE_THRESHOLD;

  using Element = Element_;
  using SmemAtomQK = typename SelectSmemAtom<kQKDChunk, Element>::type;
  using SmemAtomV = typename SelectSmemAtom<kVDChunk, Element>::type;
  using SmemLayoutQ =
      decltype(tile_to_shape(SmemAtomQK{}, Shape<Int<kBr>, Int<kQKDChunk>>{}));
  using SmemLayoutK =
      decltype(tile_to_shape(SmemAtomQK{}, Shape<Int<kBc>, Int<kQKDChunk>>{}));
  using SmemLayoutV =
      decltype(tile_to_shape(SmemAtomV{}, Shape<Int<kBc>, Int<kVDChunk>>{}));
  using SmemLayoutVt = decltype(composition(
      SmemLayoutV{},
      make_layout(Shape<Int<kVDChunk>, Int<kBc>>{}, GenRowMajor{})));
  // O output staging buffer: same K_SW128 atom as V, tiled to O's
  // [kBr,kVDChunk]. Reuses v_base smem in the epilogue (V is free after the
  // last PV GEMM).
  using SmemLayoutO =
      decltype(tile_to_shape(SmemAtomV{}, Shape<Int<kBr>, Int<kVDChunk>>{}));

  using MmaAtom =
      std::conditional_t<std::is_same<Element, cutlass::half_t>::value,
                         MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
                         MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>>;

  using TiledMmaQK = decltype(make_tiled_mma(
      MmaAtom{}, Layout<Shape<Int<kNumWarps>, _1, _1>>{},
      Tile<Int<kBr>, Int<kBc>, _16>{}));

  using TiledMmaPV = decltype(make_tiled_mma(
      MmaAtom{}, Layout<Shape<Int<kNumWarps>, _1, _1>>{},
      Tile<Int<kBr>, Int<kVDChunk>, _16>{}));

  using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
  using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, Element>;

  static constexpr int kQTileBytes = kBr * kQKDChunk * sizeof(Element);
  static constexpr int kKTileBytes = kBc * kQKDChunk * sizeof(Element);
  static constexpr int kVTileBytes = kBc * kVDChunk * sizeof(Element);
};

// Persist-D traits: full-D TMA + full-D GEMM (no D-chunking).
// Q persisted in smem; K/V pipelined with independent stages.
// TiledMma uses large N tiles: QK=[kBr,kBc], PV=[kBr,kHeadDim].
template <int kHeadDim_, int kBr_ = 128, int kBc_ = 64, int kStagesK_ = 2,
          int kStagesV_ = 2, typename Element_ = cutlass::half_t>
struct FFPAAttnCuTePersistDTraits {
  static constexpr int kHeadDim = kHeadDim_;
  static constexpr int kBr = kBr_;
  static constexpr int kBc = kBc_;
  static constexpr int kNumWarps = kBr / 16;
  static constexpr int kNumThreads = kNumWarps * 32;
  static constexpr int kStagesK = kStagesK_;
  static constexpr int kStagesV = kStagesV_;
  static constexpr float kRescaleThreshold = FFPA_RESCALE_THRESHOLD;

  static constexpr int kSmemElems =
      kBr * kHeadDim + kStagesK * kBc * kHeadDim + kStagesV * kBc * kHeadDim;

  using Element = Element_;
  using SmemAtom = GMMA::Layout_K_SW128_Atom<Element>;
  using SmemLayoutQ =
      decltype(tile_to_shape(SmemAtom{}, Shape<Int<kBr>, Int<kHeadDim>>{}));
  using SmemLayoutKV =
      decltype(tile_to_shape(SmemAtom{}, Shape<Int<kBc>, Int<kHeadDim>>{}));
  using SmemLayoutKVt = decltype(composition(
      SmemLayoutKV{},
      make_layout(Shape<Int<kHeadDim>, Int<kBc>>{}, GenRowMajor{})));
  using SmemLayoutO =
      decltype(tile_to_shape(SmemAtom{}, Shape<Int<kBr>, Int<kHeadDim>>{}));

  using MmaAtom =
      std::conditional_t<std::is_same<Element, cutlass::half_t>::value,
                         MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
                         MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>>;

  using TiledMmaQK = decltype(make_tiled_mma(
      MmaAtom{}, Layout<Shape<Int<kNumWarps>, _1, _1>>{},
      Tile<Int<kBr>, Int<kBc>, _16>{}));

  using TiledMmaPV = decltype(make_tiled_mma(
      MmaAtom{}, Layout<Shape<Int<kNumWarps>, _1, _1>>{},
      Tile<Int<kBr>, Int<kHeadDim>, _16>{}));

  using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
  using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, Element>;
};

// WS split-D reuses FFPAAttnCuTeTraits directly (FA-2 split-Q M8N1, same as
// non-WS split-D): the WS layer is a pure producer/consumer split on top of
// the unchanged consumer MMA layout.

// Split-D + M4N2 traits: kBr=64, 8 warps (4M×2N), O regs = D/4 per thread.
// SMEM is O(1) w.r.t. D (split-D chunking); register pressure halved vs M8N1.
// P requires SMEM roundtrip (stmatrix→LDSM_N) since each N-warp holds only
// half the Bc columns; softmax requires cross-N-warp reduction.
//
// Scalability limit: D=1024 → O=256 regs/thread (hard ceiling).
// For D>1024 with lossless fp32 acc storage, use M?N4 (e.g. M2N4,
// atom_layout=(2,4,1), kBr=32) to halve O to D/8 per thread.
template <int kHeadDim_, int kBr_ = 64, int kBc_ = 64, int kQKDChunk_ = 64,
          int kVDChunk_ = 64, int kStagesQK_ = 2, int kStagesPV_ = 2,
          typename Element_ = cutlass::half_t>
struct FFPAAttnCuTeSplitDM4N2Traits {
  static_assert(kHeadDim_ % kQKDChunk_ == 0);
  static_assert(kHeadDim_ % kVDChunk_ == 0);
  static_assert(kQKDChunk_ == 16 || kQKDChunk_ == 32 || kQKDChunk_ == 64);
  static_assert(kVDChunk_ == 16 || kVDChunk_ == 32 || kVDChunk_ == 64);
  static_assert(kBr_ == 64, "M4N2 requires kBr=64 (4 warps x 16 rows)");
  static_assert(kBc_ == 64,
                "M4N2 requires kBc=64 (2 N-warps x 8 cols x 4 iters)");

  static constexpr int kHeadDim = kHeadDim_;
  static constexpr int kBr = kBr_;
  static constexpr int kBc = kBc_;
  static constexpr int kQKDChunk = kQKDChunk_;
  static constexpr int kVDChunk = kVDChunk_;
  static constexpr int kDChunksQK = kHeadDim / kQKDChunk;
  static constexpr int kDChunksV = kHeadDim / kVDChunk;
  static constexpr int kNumWarps = 8;
  static constexpr int kNumThreads = 256;
  static constexpr int kStagesQK = kStagesQK_;
  static constexpr int kStagesPV = kStagesPV_;
  static constexpr float kRescaleThreshold = FFPA_RESCALE_THRESHOLD;

  // SMEM: Q stages + K stages + V stages + P staging + softmax exchange
  static constexpr int kSmemQK =
      kStagesQK * kBr * kQKDChunk + kStagesQK * kBc * kQKDChunk;
  static constexpr int kSmemV = kStagesPV * kBc * kVDChunk;
  static constexpr int kSmemP = kBr * kBc;
  // Softmax exchange: separate max and sum buffers, each 8 warps x 16 rows
  // floats. Stored as Element count to align with the dynamic smem allocation.
  static constexpr int kSmemExchange =
      2 * kNumWarps * 16 * sizeof(float) / sizeof(Element_);
  static constexpr int kSmemElems = kSmemQK + kSmemV + kSmemP + kSmemExchange;

  static constexpr int kVChunksPerBatch = compute_vchunks_per_batch(
      kDChunksV, kHeadDim, kBr, kStagesPV* kBc* kVDChunk);
  static constexpr int kNBatches = kDChunksV / kVChunksPerBatch;

  using Element = Element_;
  using SmemAtomQK = typename SelectSmemAtom<kQKDChunk, Element>::type;
  using SmemAtomV = typename SelectSmemAtom<kVDChunk, Element>::type;
  using SmemLayoutQ =
      decltype(tile_to_shape(SmemAtomQK{}, Shape<Int<kBr>, Int<kQKDChunk>>{}));
  using SmemLayoutK =
      decltype(tile_to_shape(SmemAtomQK{}, Shape<Int<kBc>, Int<kQKDChunk>>{}));
  using SmemLayoutV =
      decltype(tile_to_shape(SmemAtomV{}, Shape<Int<kBc>, Int<kVDChunk>>{}));
  using SmemLayoutVt = decltype(composition(
      SmemLayoutV{},
      make_layout(Shape<Int<kVDChunk>, Int<kBc>>{}, GenRowMajor{})));
  using SmemLayoutO =
      decltype(tile_to_shape(SmemAtomV{}, Shape<Int<kBr>, Int<kVDChunk>>{}));
  // P staging: [kBr, kBc], same swizzle atom as QK so stmatrix (QK C-fragment)
  // and ldmatrix (PV A-fragment) hit the same SMEM layout.
  using SmemLayoutP =
      decltype(tile_to_shape(SmemAtomQK{}, Shape<Int<kBr>, Int<kBc>>{}));

  using MmaAtom =
      std::conditional_t<std::is_same<Element, cutlass::half_t>::value,
                         MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
                         MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>>;

  // M4N2: 4 warps along M (4x16=64=kBr), 2 warps along N (2x8=16 per K-step)
  using AtomLayoutMN = Layout<Shape<_4, _2, _1>>;
  using TiledMmaQK = decltype(make_tiled_mma(MmaAtom{}, AtomLayoutMN{},
                                             Tile<Int<kBr>, Int<kBc>, _16>{}));
  using TiledMmaPV = decltype(make_tiled_mma(
      MmaAtom{}, AtomLayoutMN{}, Tile<Int<kBr>, Int<kVDChunk>, _16>{}));

  using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
  using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, Element>;

  static constexpr int kQTileBytes = kBr * kQKDChunk * sizeof(Element);
  static constexpr int kKTileBytes = kBc * kQKDChunk * sizeof(Element);
  static constexpr int kVTileBytes = kBc * kVDChunk * sizeof(Element);
};

}  // namespace ffpa_cute

#pragma once

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
  // FA-4 conditional rescaling threshold (log2 domain). 0.0 = disabled.
  // FP16/BF16: 8.0 = log2(256); FP8 would use 4.0 = log2(16).
  static constexpr float kRescaleThreshold = 8.0f;

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

}  // namespace ffpa_cute

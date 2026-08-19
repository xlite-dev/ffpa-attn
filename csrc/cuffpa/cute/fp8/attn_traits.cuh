// FP8 attention kernel traits (persist-D / split-D / split-D M4N2), split
// out of cute/attn_traits.cuh so each precision owns its traits file. The
// fp16 traits and the shared compute_vchunks_per_batch helper stay in
// cute/attn_traits.cuh; that file is included here for the helper (and its
// common/gemm/cute header chain).
#pragma once

#include "../attn_traits.cuh"

namespace ffpa_cute {
using namespace cute;

// Persist-D FP8 traits: fp8 e4m3 Q/K/V via SM89 m16n8k32 mma.sync.
// V is pre-transposed by the quantize pre-kernel and stored (D x N), so the PV
// B-operand loads with the same non-transposed LDSM_N atom as Q/K; there is no
// SmemCopyAtomTransposed for fp8. TiledMma K-tile is 32 (atom K), not 16.
// kQKInt8: Q/K become symmetric int8 and QK runs SM80 m16n8k32 s8xs8->s32
// (A/B operand layouts identical to the fp8 atom; acc is s32, cast in-kernel);
// V/PV stay fp8 e4m3.
template <int kHeadDim_, typename ElementO_, int kBr_ = 128, int kBc_ = 64,
          int kStagesK_ = 2, int kStagesV_ = 2, bool kQKInt8_ = false>
struct FFPAAttnCuTePersistDFP8Traits {
  static constexpr int kHeadDim = kHeadDim_;
  static constexpr int kBr = kBr_;
  static constexpr int kBc = kBc_;
  static constexpr int kNumWarps = kBr / 16;
  static constexpr int kNumThreads = kNumWarps * 32;
  static constexpr int kStagesK = kStagesK_;
  static constexpr int kStagesV = kStagesV_;
  static constexpr bool kQKInt8 = kQKInt8_;
  // fp8 P operand saturates at 448, so use the FA-4 fp8 rescale threshold.
  static constexpr float kRescaleThreshold = FFPA_RESCALE_THRESHOLD_FP8;

  static constexpr int kSmemElems =
      kBr * kHeadDim + kStagesK * kBc * kHeadDim + kStagesV * kBc * kHeadDim;

  using Element = cutlass::float_e4m3_t;  // V / P (PV side, always fp8)
  using ElementQK = std::conditional_t<kQKInt8, int8_t, cutlass::float_e4m3_t>;
  using ElementO = ElementO_;
  // 1B elems: a 128B SW128 row needs 128 elems; D=64 only has 64 -> use SW64.
  // Swizzle atom picked by the K-dim (contiguous, last) byte count so the
  // swizzle-L divides the leading stride: Q/K/O have K=D, V^T has K=kBc.
  // SW128/SW64/SW32 = 128/64/32-byte swizzle rows; L must divide K*sz and
  // K*sz>=L. SW32 fallback covers all 32-multiple D (e.g. D=32/96/160/224).
  template <typename Elem, int kKBytes>
  using SmemAtomByK = std::conditional_t<
      kKBytes % 128 == 0, GMMA::Layout_K_SW128_Atom<Elem>,
      std::conditional_t<kKBytes % 64 == 0, GMMA::Layout_K_SW64_Atom<Elem>,
                         GMMA::Layout_K_SW32_Atom<Elem>>>;
  using SmemAtomQK =
      SmemAtomByK<ElementQK, kHeadDim* static_cast<int>(sizeof(ElementQK))>;
  using SmemAtomV =
      SmemAtomByK<Element, kBc* static_cast<int>(sizeof(Element))>;
  using SmemLayoutQ =
      decltype(tile_to_shape(SmemAtomQK{}, Shape<Int<kBr>, Int<kHeadDim>>{}));
  using SmemLayoutK =
      decltype(tile_to_shape(SmemAtomQK{}, Shape<Int<kBc>, Int<kHeadDim>>{}));
  // V^T view: (kHeadDim x kBc) row-major, loaded as PV B-operand.
  using SmemLayoutV =
      decltype(tile_to_shape(SmemAtomV{}, Shape<Int<kHeadDim>, Int<kBc>>{}));
  // O staging in the epilogue (ElementO fp16/bf16, NOT the fp8 SmemAtom):
  // after the KV loop the freed Q/K/V smem is aliased to stage O via STSM
  // before the coalesced TMA store; swizzle must match the TMA descriptor.
  // D=32 fp16/bf16 O row is only 64B -> SW64; same rule as SmemAtomByK.
  using SmemAtomO =
      SmemAtomByK<ElementO, kHeadDim* static_cast<int>(sizeof(ElementO))>;
  using SmemLayoutO =
      decltype(tile_to_shape(SmemAtomO{}, Shape<Int<kBr>, Int<kHeadDim>>{}));

  // s8 atom acc is s32; A/B layouts match the fp8 atom exactly (both
  // 32dp32b m16n8k32), so smem/copy/TMA plumbing is shared.
  using MmaAtomQK =
      std::conditional_t<kQKInt8, MMA_Atom<SM80_16x8x32_S32S8S8S32_TN>,
                         MMA_Atom<SM89_16x8x32_F32E4M3E4M3F32_TN>>;
  using MmaAtom = MMA_Atom<SM89_16x8x32_F32E4M3E4M3F32_TN>;

  using TiledMmaQK = decltype(make_tiled_mma(
      MmaAtomQK{}, Layout<Shape<Int<kNumWarps>, _1, _1>>{},
      Tile<Int<kBr>, Int<kBc>, _32>{}));

  using TiledMmaPV = decltype(make_tiled_mma(
      MmaAtom{}, Layout<Shape<Int<kNumWarps>, _1, _1>>{},
      Tile<Int<kBr>, Int<kHeadDim>, _32>{}));
  // f8f8f16 PV atom for the fp16-accumulator inst_buf path (persist_d.cuh):
  // same m16n8k32 shape, A/B e4m3, but C/D half. Its CLayout matches the f32
  // atom (both inherit SM80_16x8_Row), so the half inst_buf absorbs into the
  // float o_acc element-wise via CUDA-core FADD each kv_tile.
  using MmaAtomPVf16 =
      MMA_Atom<SM120_16x8x32_TN<Element, Element, cutlass::half_t>>;
  using TiledMmaPVf16 = decltype(make_tiled_mma(
      MmaAtomPVf16{}, Layout<Shape<Int<kNumWarps>, _1, _1>>{},
      Tile<Int<kBr>, Int<kHeadDim>, _32>{}));

  using SmemCopyAtomQK = Copy_Atom<SM75_U32x4_LDSM_N, ElementQK>;
  using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;

  static constexpr int kQTileBytes = kBr * kHeadDim * sizeof(ElementQK);
  static constexpr int kKTileBytes = kBc * kHeadDim * sizeof(ElementQK);
  static constexpr int kVTileBytes = kBc * kHeadDim * sizeof(Element);
};

// Split-D FP8 traits: non-WS M8N1 like FFPAAttnCuTeSplitDTraits but with
// fp8 e4m3 Q/K/V (SM89 m16n8k32 mma.sync; kQKInt8: SM80 s8 atom with s32 acc
// cast to f32 in-kernel, PV always fp8). V is pre-transposed (D x N) by the
// quantize pre-kernel, so PV B loads with the same non-transposed LDSM_N
// atom as Q/K. kSmemElems counts BYTES (1B/elem); the O-staging budget for
// compute_vchunks_per_batch must therefore be expressed in ElementO elems.
template <int kHeadDim_, typename ElementO_, int kBr_ = 128, int kBc_ = 128,
          int kQKDChunk_ = 32, int kVDChunk_ = 64, int kStagesQK_ = 2,
          int kStagesPV_ = 2, bool kQKInt8_ = false>
struct FFPAAttnCuTeSplitDFP8Traits {
  static_assert(kHeadDim_ % kQKDChunk_ == 0);
  static_assert(kHeadDim_ % kVDChunk_ == 0);
  static_assert(kQKDChunk_ == 32 || kQKDChunk_ == 64);
  static_assert(kVDChunk_ == 32 || kVDChunk_ == 64);

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
  static constexpr bool kQKInt8 = kQKInt8_;
  // fp8 P operand saturates at 448, so use the FA-4 fp8 rescale threshold.
  static constexpr float kRescaleThreshold = FFPA_RESCALE_THRESHOLD_FP8;

  using Element = cutlass::float_e4m3_t;  // V / P (PV side, always fp8)
  using ElementQK = std::conditional_t<kQKInt8, int8_t, cutlass::float_e4m3_t>;
  using ElementO = ElementO_;

  static constexpr int kSmemElems = kStagesQK * kBr * kQKDChunk +
                                    kStagesQK * kBc * kQKDChunk +
                                    kStagesPV * kBc * kVDChunk;
  static constexpr int kVChunksPerBatch = compute_vchunks_per_batch(
      kDChunksV, kHeadDim, kBr, kSmemElems / sizeof(ElementO_));
  static constexpr int kNBatches = kDChunksV / kVChunksPerBatch;

  // Swizzle atom picked by ROW BYTES (1B elems): kBc=64 e4m3 = 64B -> SW64.
  template <typename Elem, int kRowBytes>
  using SmemAtomByBytes = std::conditional_t<
      kRowBytes <= 32, GMMA::Layout_K_SW32_Atom<Elem>,
      std::conditional_t<kRowBytes <= 64, GMMA::Layout_K_SW64_Atom<Elem>,
                         GMMA::Layout_K_SW128_Atom<Elem>>>;
  using SmemAtomQK =
      SmemAtomByBytes<ElementQK,
                      kQKDChunk* static_cast<int>(sizeof(ElementQK))>;
  using SmemAtomV =
      SmemAtomByBytes<Element, kBc* static_cast<int>(sizeof(Element))>;
  using SmemLayoutQ =
      decltype(tile_to_shape(SmemAtomQK{}, Shape<Int<kBr>, Int<kQKDChunk>>{}));
  using SmemLayoutK =
      decltype(tile_to_shape(SmemAtomQK{}, Shape<Int<kBc>, Int<kQKDChunk>>{}));
  // V^T tile: (kVDChunk x kBc) row-major, loaded as PV B-operand.
  using SmemLayoutV =
      decltype(tile_to_shape(SmemAtomV{}, Shape<Int<kVDChunk>, Int<kBc>>{}));
  // O staging (ElementO fp16/bf16): reuses freed V/QK smem in the epilogue.
  using SmemLayoutO = decltype(tile_to_shape(
      GMMA::Layout_K_SW128_Atom<ElementO>{}, Shape<Int<kBr>, Int<kVDChunk>>{}));

  // s8 atom acc is s32; A/B layouts match the fp8 atom exactly (both
  // 32dp32b m16n8k32), so smem/copy/TMA plumbing is shared.
  using MmaAtomQK =
      std::conditional_t<kQKInt8, MMA_Atom<SM80_16x8x32_S32S8S8S32_TN>,
                         MMA_Atom<SM89_16x8x32_F32E4M3E4M3F32_TN>>;
  using MmaAtom = MMA_Atom<SM89_16x8x32_F32E4M3E4M3F32_TN>;

  using TiledMmaQK = decltype(make_tiled_mma(
      MmaAtomQK{}, Layout<Shape<Int<kNumWarps>, _1, _1>>{},
      Tile<Int<kBr>, Int<kBc>, _32>{}));
  using TiledMmaPV = decltype(make_tiled_mma(
      MmaAtom{}, Layout<Shape<Int<kNumWarps>, _1, _1>>{},
      Tile<Int<kBr>, Int<kVDChunk>, _32>{}));
  // f8f8f16 PV atom for the fp16-accumulator inst_buf path (split_d.cuh):
  // same m16n8k32 shape, A/B e4m3, C/D half. CLayout matches the f32 atom so
  // the half inst_buf absorbs into float o_acc via CUDA-core FADD per
  // kv_tile. N dim is kVDChunk (split-D chunk, not full kHeadDim).
  using MmaAtomPVf16 =
      MMA_Atom<SM120_16x8x32_TN<Element, Element, cutlass::half_t>>;
  using TiledMmaPVf16 = decltype(make_tiled_mma(
      MmaAtomPVf16{}, Layout<Shape<Int<kNumWarps>, _1, _1>>{},
      Tile<Int<kBr>, Int<kVDChunk>, _32>{}));

  using SmemCopyAtomQK = Copy_Atom<SM75_U32x4_LDSM_N, ElementQK>;
  using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;

  static constexpr int kQTileBytes = kBr * kQKDChunk * sizeof(ElementQK);
  static constexpr int kKTileBytes = kBc * kQKDChunk * sizeof(ElementQK);
  static constexpr int kVTileBytes = kBc * kVDChunk * sizeof(Element);
};

// Split-D M4N2 FP8 traits: m4n2 atom layout (4,2,1) + fp8 e4m3 Q/K/V via
// SM89 m16n8k32 mma.sync. Combines FFPAAttnCuTeSplitDM4N2Traits (m4n2 geometry,
// 8 warps, cross-N-warp softmax, P SMEM roundtrip) with FFPAAttnCuTeSplitDFP8
// (fp8 elements, V^T pre-transposed, kRescaleThreshold=4.0, K-tile=32).
// kBr=64/kBc=64 fixed (m4n2); O regs = D/4 per thread (D=1024 -> 256 regs).
// SmemLayoutP uses SW64 (kBc=64 e4m3 = 64B row), matching both stmatrix write
// (QK C-fragment) and LDSM_N read (PV A-fragment).
template <int kHeadDim_, typename ElementO_, int kBr_ = 64, int kBc_ = 64,
          int kQKDChunk_ = 64, int kVDChunk_ = 64, int kStagesQK_ = 2,
          int kStagesPV_ = 2, bool kQKInt8_ = false>
struct FFPAAttnCuTeSplitDM4N2FP8Traits {
  static_assert(kHeadDim_ % kQKDChunk_ == 0);
  static_assert(kHeadDim_ % kVDChunk_ == 0);
  static_assert(kQKDChunk_ == 32 || kQKDChunk_ == 64);
  static_assert(kVDChunk_ == 32 || kVDChunk_ == 64);
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
  static constexpr bool kQKInt8 = kQKInt8_;
  static constexpr float kRescaleThreshold = FFPA_RESCALE_THRESHOLD_FP8;

  using Element = cutlass::float_e4m3_t;  // V / P (PV side, always fp8)
  using ElementQK = std::conditional_t<kQKInt8, int8_t, cutlass::float_e4m3_t>;
  using ElementO = ElementO_;

  // SMEM: Q stages + K stages + V stages + P staging + softmax exchange
  static constexpr int kSmemQK =
      kStagesQK * kBr * kQKDChunk + kStagesQK * kBc * kQKDChunk;
  static constexpr int kSmemV = kStagesPV * kBc * kVDChunk;
  static constexpr int kSmemP = kBr * kBc;
  static constexpr int kSmemExchange =
      2 * kNumWarps * 16 * sizeof(float) / sizeof(Element);
  static constexpr int kSmemElems = kSmemQK + kSmemV + kSmemP + kSmemExchange;

  static constexpr int kVChunksPerBatch = compute_vchunks_per_batch(
      kDChunksV, kHeadDim, kBr, kSmemElems / sizeof(ElementO_));
  static constexpr int kNBatches = kDChunksV / kVChunksPerBatch;

  // Swizzle atom picked by ROW BYTES (1B elems): kBc=64 e4m3 = 64B -> SW64.
  template <typename Elem, int kRowBytes>
  using SmemAtomByBytes = std::conditional_t<
      kRowBytes <= 32, GMMA::Layout_K_SW32_Atom<Elem>,
      std::conditional_t<kRowBytes <= 64, GMMA::Layout_K_SW64_Atom<Elem>,
                         GMMA::Layout_K_SW128_Atom<Elem>>>;
  using SmemAtomQK =
      SmemAtomByBytes<ElementQK,
                      kQKDChunk* static_cast<int>(sizeof(ElementQK))>;
  using SmemAtomV =
      SmemAtomByBytes<Element, kBc* static_cast<int>(sizeof(Element))>;
  using SmemLayoutQ =
      decltype(tile_to_shape(SmemAtomQK{}, Shape<Int<kBr>, Int<kQKDChunk>>{}));
  using SmemLayoutK =
      decltype(tile_to_shape(SmemAtomQK{}, Shape<Int<kBc>, Int<kQKDChunk>>{}));
  // V^T tile: (kVDChunk x kBc) row-major, loaded as PV B-operand.
  using SmemLayoutV =
      decltype(tile_to_shape(SmemAtomV{}, Shape<Int<kVDChunk>, Int<kBc>>{}));
  // O staging (ElementO fp16/bf16): reuses freed V/QK smem in the epilogue.
  using SmemLayoutO = decltype(tile_to_shape(
      GMMA::Layout_K_SW128_Atom<ElementO>{}, Shape<Int<kBr>, Int<kVDChunk>>{}));
  // P staging: [kBr, kBc]. SW64 (kBc=64 e4m3 = 64B row). stmatrix cannot
  // write 1B e4m3 (it's a b16 op needing SW128 for 16B vectorization, but
  // SW128's 128-elem atom doesn't divide the 64-col P tile). DefaultCopy
  // (vectorized stores) is used instead; see kernel comment.
  using SmemLayoutP =
      decltype(tile_to_shape(SmemAtomV{}, Shape<Int<kBr>, Int<kBc>>{}));

  using MmaAtomQK =
      std::conditional_t<kQKInt8, MMA_Atom<SM80_16x8x32_S32S8S8S32_TN>,
                         MMA_Atom<SM89_16x8x32_F32E4M3E4M3F32_TN>>;
  using MmaAtom = MMA_Atom<SM89_16x8x32_F32E4M3E4M3F32_TN>;

  // M4N2: 4 warps along M (4x16=64=kBr), 2 warps along N (2x8=16 per K-step)
  using AtomLayoutMN = Layout<Shape<_4, _2, _1>>;
  using TiledMmaQK = decltype(make_tiled_mma(MmaAtomQK{}, AtomLayoutMN{},
                                             Tile<Int<kBr>, Int<kBc>, _32>{}));
  using TiledMmaPV = decltype(make_tiled_mma(
      MmaAtom{}, AtomLayoutMN{}, Tile<Int<kBr>, Int<kVDChunk>, _32>{}));
  // f8f8f16 PV atom for the fp16-accumulator inst_buf path (m4n2.cuh):
  // same m16n8k32 shape, A/B e4m3, C/D half; uses m4n2 AtomLayoutMN so the
  // half inst_buf layout matches the f32 PV atom. N dim is kVDChunk.
  using MmaAtomPVf16 =
      MMA_Atom<SM120_16x8x32_TN<Element, Element, cutlass::half_t>>;
  using TiledMmaPVf16 = decltype(make_tiled_mma(
      MmaAtomPVf16{}, AtomLayoutMN{}, Tile<Int<kBr>, Int<kVDChunk>, _32>{}));

  using SmemCopyAtomQK = Copy_Atom<SM75_U32x4_LDSM_N, ElementQK>;
  using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;

  static constexpr int kQTileBytes = kBr * kQKDChunk * sizeof(ElementQK);
  static constexpr int kKTileBytes = kBc * kQKDChunk * sizeof(ElementQK);
  static constexpr int kVTileBytes = kBc * kVDChunk * sizeof(Element);
};

}  // namespace ffpa_cute

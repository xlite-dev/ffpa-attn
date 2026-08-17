// NVFP4 blockscaled MMA atom + SF (ue4m3) layouts for sm_120, ported from
// SageAttention3 (sageattn3/blackwell/{cute_extension.h,blockscaled_layout.h}).
// The atom issues 4x mma.sync.m16n8k64 kind::mxf4nvf4 (scale_vec::4X, ue4m3
// scale) and presents a fused 16x32x64 shape; SF fragments are 1x uint32 per
// thread holding 4 packed ue4m3 scales.
// Reference:
// https://github.com/thu-ml/SageAttention/tree/main/sageattention3_blackwell/sageattn3/blackwell/cute_extension.h
//            https://github.com/thu-ml/SageAttention/tree/main/sageattention3_blackwell/sageattn3/blackwell/blockscaled_layout.h
#pragma once

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/float8.h>
#include <cutlass/float_subbyte.h>

namespace ffpa_fp4 {

using namespace cute;

// SF tensor layout config: 1x16-element groups, scales stored as ue4m3.
// SfAtom treats the SF tensor as ((MN),(K)) where each 64x4 block of (MN, K/16)
// holds the scales in the exact order the MMA's SFA/SFB operand expects.
template <int SFVecSize_>
struct BlockScaledConfig {
  static constexpr int SFVecSize = SFVecSize_;
  static constexpr int MMA_NSF = 4;
  using Blk_MN = _64;
  using Blk_SF = _4;
  using mnBasicBlockShape = Shape<_16, _4>;
  using mnBasicBlockStride = Stride<_16, _4>;
  using kBasicBlockShape = Shape<Int<SFVecSize>, Int<MMA_NSF>>;
  using kBasicBlockStride = Stride<_0, _1>;
  using SfAtom = Layout<Shape<mnBasicBlockShape, kBasicBlockShape>,
                        Stride<mnBasicBlockStride, kBasicBlockStride>>;

  using LayoutSF = decltype(blocked_product(
      SfAtom{},
      make_layout(make_shape(int32_t(0), int32_t(0), int32_t(0), int32_t(0)),
                  make_stride(int32_t(0), _1{}, int32_t(0), int32_t(0)))));
  // One indivisible block holds 4 scale factors of 64 rows (cols); 4 keeps a
  // 32-bit word's scales within a single row so the packed SF word loads whole.
  using Blk_Elems = decltype(Blk_MN{} * Blk_SF{});
  using sSF_strideMN = decltype(prepend(Blk_Elems{}, mnBasicBlockStride{}));

  // gmem SF layout for Q/K: problem (Seqlen, Dim/16, Head, Batch).
  template <class ProblemShape>
  CUTE_HOST_DEVICE static constexpr auto tile_atom_to_shape_SFQKV(
      ProblemShape problem_shape) {
    auto [Seqlen, Dim, HeadNum, Batch] = problem_shape;
    return tile_to_shape(SfAtom{}, make_shape(Seqlen, Dim, HeadNum, Batch),
                         Step<_2, _1, _3, _4>{});
  }

  // gmem SF layout for V^T: problem (Dim, Seqlen/16, Head, Batch).
  template <class ProblemShape>
  CUTE_HOST_DEVICE static constexpr auto tile_atom_to_shape_SFVt(
      ProblemShape problem_shape) {
    auto [Dim, Seqlen, HeadNum, Batch] = problem_shape;
    return tile_to_shape(SfAtom{}, make_shape(Dim, Seqlen, HeadNum, Batch),
                         Step<_2, _1, _3, _4>{});
  }

  // smem SF layouts matching the MMA SF operand access pattern. M variant is
  // for SFQ (A operand of QK); N variants for SFK (B of QK, N=KV tokens) and
  // SFVt (B of PV, N=head dim when given the PV-permuted tile shape).
  template <class TiledMma, class TileShape_MNK>
  CUTE_HOST_DEVICE static constexpr auto deduce_smem_layoutSFQ(
      TiledMma tiled_mma, TileShape_MNK tileshape_mnk) {
    using sSFQ_shapeK = decltype(prepend(
        make_shape(Blk_SF{} / Int<MMA_NSF>{},
                   size<2>(TileShape_MNK{}) / Int<SFVecSize>{} / Blk_SF{}),
        kBasicBlockShape{}));
    using sSFQ_shapeM = decltype(prepend(size<0>(TileShape_MNK{}) / Blk_MN{},
                                         mnBasicBlockShape{}));
    using sSFQ_strideM = sSF_strideMN;
    using sSFQ_strideK = decltype(prepend(
        make_stride(Int<MMA_NSF>{},
                    size<0>(TileShape_MNK{}) / Blk_MN{} * Blk_Elems{}),
        kBasicBlockStride{}));
    using sSFQ_shape = decltype(make_shape(sSFQ_shapeM{}, sSFQ_shapeK{}));
    using sSFQ_stride = decltype(make_stride(sSFQ_strideM{}, sSFQ_strideK{}));
    return make_layout(sSFQ_shape{}, sSFQ_stride{});
  }

  template <class TiledMma, class TileShape_MNK>
  CUTE_HOST_DEVICE static constexpr auto deduce_smem_layoutSFKV(
      TiledMma tiled_mma, TileShape_MNK tileshape_mnk) {
    using sSFK_shapeK = decltype(prepend(
        make_shape(Blk_SF{} / Int<MMA_NSF>{},
                   size<2>(TileShape_MNK{}) / Int<SFVecSize>{} / Blk_SF{}),
        kBasicBlockShape{}));
    using sSFK_shapeN = decltype(prepend(size<1>(TileShape_MNK{}) / Blk_MN{},
                                         mnBasicBlockShape{}));
    using sSFK_strideN = sSF_strideMN;
    using sSFK_strideK = decltype(prepend(
        make_stride(Int<MMA_NSF>{},
                    size<1>(TileShape_MNK{}) / Blk_MN{} * Blk_Elems{}),
        kBasicBlockStride{}));
    using sSFK_shape = decltype(make_shape(sSFK_shapeN{}, sSFK_shapeK{}));
    using sSFK_stride = decltype(make_stride(sSFK_strideN{}, sSFK_strideK{}));
    return make_layout(sSFK_shape{}, sSFK_stride{});
  }

  // Same atom structure as SFKV; kept separate because the PV tile shape maps
  // head dim into mode 1 (N) and KV tokens into mode 2 (K).
  template <class TiledMma, class TileShape_MNK>
  CUTE_HOST_DEVICE static constexpr auto deduce_smem_layoutSFVt(
      TiledMma tiled_mma, TileShape_MNK tileshape_mnk) {
    using sSFVt_shapeK = decltype(prepend(
        make_shape(Blk_SF{} / Int<MMA_NSF>{},
                   size<2>(TileShape_MNK{}) / Int<SFVecSize>{} / Blk_SF{}),
        kBasicBlockShape{}));
    using sSFVt_shapeN = decltype(prepend(size<1>(TileShape_MNK{}) / Blk_MN{},
                                          mnBasicBlockShape{}));
    using sSFVt_strideN = sSF_strideMN;
    using sSFVt_strideK = decltype(prepend(
        make_stride(Int<MMA_NSF>{},
                    size<1>(TileShape_MNK{}) / Blk_MN{} * Blk_Elems{}),
        kBasicBlockStride{}));
    using sSFVt_shape = decltype(make_shape(sSFVt_shapeN{}, sSFVt_shapeK{}));
    using sSFVt_stride =
        decltype(make_stride(sSFVt_strideN{}, sSFVt_strideK{}));
    return make_layout(sSFVt_shape{}, sSFVt_stride{});
  }
};

}  // namespace ffpa_fp4

namespace cute::SM120::BLOCKSCALED {

using cutlass::float_e2m1_t;
using cutlass::float_ue4m3_t;

// MMA.SF 16x32x64 TN E2M1 x E2M1 with SF E4M3 (NVFP4).
struct SM120_16x32x64_TN_VS_NVFP4 {
  using DRegisters = float[16];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[8];
  using CRegisters = float[16];

  static constexpr int SFBits = 32;
  using RegTypeSF = cute::uint_bit_t<SFBits>;

  using SFARegisters = RegTypeSF[1];
  using SFBRegisters = RegTypeSF[1];

  CUTE_HOST_DEVICE static void fma(
      float& d0, float& d1, float& d2, float& d3, float& d4, float& d5,
      float& d6, float& d7, float& d8, float& d9, float& d10, float& d11,
      float& d12, float& d13, float& d14, float& d15, uint32_t const& a0,
      uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1, uint32_t const& b2,
      uint32_t const& b3, uint32_t const& b4, uint32_t const& b5,
      uint32_t const& b6, uint32_t const& b7, float const& c0, float const& c1,
      float const& c2, float const& c3, float const& c4, float const& c5,
      float const& c6, float const& c7, float const& c8, float const& c9,
      float const& c10, float const& c11, float const& c12, float const& c13,
      float const& c14, float const& c15, RegTypeSF const& sfa0,
      RegTypeSF const& sfb0) {
    static constexpr uint16_t tidA = 0;
    static constexpr uint16_t bidA = 0;
    static constexpr uint16_t bidB = 0;
    static constexpr uint16_t tidB0 = 0;
    static constexpr uint16_t tidB1 = 1;
    static constexpr uint16_t tidB2 = 2;
    static constexpr uint16_t tidB3 = 3;
#if defined(CUTE_ARCH_MXF4NVF4_4X_UE4M3_MMA_ENABLED)
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64."
        "row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13},"
        "{%14},"
        "{%15, %16},"
        "{%17},"
        "{%18, %19};\n"
        : "=f"(d0), "=f"(d1), "=f"(d8), "=f"(d9)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0),
          "f"(c1), "f"(c8), "f"(c9), "r"(uint32_t(sfa0)), "h"(bidA), "h"(tidA),
          "r"(uint32_t(sfb0)), "h"(bidB), "h"(tidB0));
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64."
        "row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13},"
        "{%14},"
        "{%15, %16},"
        "{%17},"
        "{%18, %19};\n"
        : "=f"(d2), "=f"(d3), "=f"(d10), "=f"(d11)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b2), "r"(b3), "f"(c2),
          "f"(c3), "f"(c10), "f"(c11), "r"(uint32_t(sfa0)), "h"(bidA),
          "h"(tidA), "r"(uint32_t(sfb0)), "h"(bidB), "h"(tidB1));
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64."
        "row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13},"
        "{%14},"
        "{%15, %16},"
        "{%17},"
        "{%18, %19};\n"
        : "=f"(d4), "=f"(d5), "=f"(d12), "=f"(d13)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b4), "r"(b5), "f"(c4),
          "f"(c5), "f"(c12), "f"(c13), "r"(uint32_t(sfa0)), "h"(bidA),
          "h"(tidA), "r"(uint32_t(sfb0)), "h"(bidB), "h"(tidB2));
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64."
        "row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13},"
        "{%14},"
        "{%15, %16},"
        "{%17},"
        "{%18, %19};\n"
        : "=f"(d6), "=f"(d7), "=f"(d14), "=f"(d15)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b6), "r"(b7), "f"(c6),
          "f"(c7), "f"(c14), "f"(c15), "r"(uint32_t(sfa0)), "h"(bidA),
          "h"(tidA), "r"(uint32_t(sfb0)), "h"(bidB), "h"(tidB3));
#else
    CUTE_INVALID_CONTROL_PATH(
        "SM120_16x32x64_TN_VS_NVFP4 requires "
        "CUTE_ARCH_MXF4NVF4_4X_UE4M3_MMA_ENABLED");
#endif
  }
};

}  // namespace cute::SM120::BLOCKSCALED

namespace cute {

// MMA NVFP4 16x32x64 TN: A = e2m1 (M,K) row, B = e2m1 (N,K) col, SF = ue4m3
// with 16-element groups. A/B fragment values are 4-bit regardless of decl.
template <>
struct MMA_Traits<SM120::BLOCKSCALED::SM120_16x32x64_TN_VS_NVFP4> {
  using ValTypeA = uint4_t;
  using ValTypeB = uint4_t;
  using ValTypeD = float;
  using ValTypeC = float;
  using ValTypeSF = cutlass::float_ue4m3_t;
  constexpr static int SFVecSize = 16;

  using Shape_MNK = Shape<_16, _32, _64>;
  using ThrID = Layout<_32>;

  // (T32,V32) -> (M16,K64)
  using ALayout = Layout<Shape<Shape<_4, _8>, Shape<_8, _2, _2>>,
                         Stride<Stride<_128, _1>, Stride<_16, _8, _512>>>;
  // (T32,V64) -> (N32,K64)
  using BLayout = Layout<Shape<Shape<_4, _8>, Shape<_8, _2, _4>>,
                         Stride<Stride<_256, _1>, Stride<_32, _1024, _8>>>;
  // (T32,V64) -> (M16,K64)
  using SFALayout =
      Layout<Shape<Shape<_2, _2, _8>, _64>, Stride<Stride<_8, _0, _1>, _16>>;
  // (T32,V64) -> (N32,K64)
  using SFBLayout =
      Layout<Shape<Shape<_4, _8>, _64>, Stride<Stride<_8, _1>, _32>>;
  // (T32,V16) -> (M16,N32)
  using CLayout =
      Layout<Shape<Shape<_4, _8>, Shape<Shape<_2, _4>, _2>>,
             Stride<Stride<_32, _1>, Stride<Stride<_16, _128>, _8>>>;
};

// Slice the SF tensor into a per-thread fragment, honoring the TiledMma's
// permutation tiles (K-row permutation for QK, P-column permutation for PV).
template <class SFATensor, class Atom, class TiledThr, class TiledPerm>
CUTE_HOST_DEVICE constexpr auto thrfrg_SFA(
    SFATensor&& sfatensor, TiledMMA<Atom, TiledThr, TiledPerm>& mma) {
  CUTE_STATIC_ASSERT_V(rank(sfatensor) >= Int<2>{});

  using AtomShape_MNK = typename Atom::Shape_MNK;
  using AtomLayoutSFA_TV = typename Atom::Traits::SFALayout;

  auto permutation_mnk = TiledPerm{};
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

  auto t_tile = make_tile(get<0>(permutation_mnk), get<2>(permutation_mnk));
  auto t_tensor = logical_divide(sfatensor, t_tile);  // (PermM,PermK)

  auto a_tile = make_tile(make_layout(size<0>(AtomShape_MNK{})),
                          make_layout(size<2>(AtomShape_MNK{})));
  auto a_tensor =
      zipped_divide(t_tensor, a_tile);  // ((AtomM,AtomK),(RestM,RestK))

  auto tv_tensor =
      a_tensor.compose(AtomLayoutSFA_TV{}, _);  // ((ThrV,FrgV),(RestM,RestK))

  auto thr_tile =
      make_tile(_, make_tile(make_layout(size<1>(thr_layout_vmnk)),
                             make_layout(size<3>(thr_layout_vmnk))));
  auto thr_tensor = zipped_divide(
      tv_tensor, thr_tile);  // ((ThrV,(ThrM,ThrK)),(FrgV,(RestM,RestK)))

  return thr_tensor;
}

template <class SFBTensor, class Atom, class TiledThr, class TiledPerm>
CUTE_HOST_DEVICE constexpr auto thrfrg_SFB(
    SFBTensor&& sfbtensor, TiledMMA<Atom, TiledThr, TiledPerm>& mma) {
  CUTE_STATIC_ASSERT_V(rank(sfbtensor) >= Int<2>{});

  using AtomShape_MNK = typename Atom::Shape_MNK;
  using AtomLayoutSFB_TV = typename Atom::Traits::SFBLayout;

  auto permutation_mnk = TiledPerm{};
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

  auto t_tile = make_tile(get<1>(permutation_mnk), get<2>(permutation_mnk));
  auto t_tensor = logical_divide(sfbtensor, t_tile);  // (PermN,PermK)

  auto a_tile = make_tile(make_layout(size<1>(AtomShape_MNK{})),
                          make_layout(size<2>(AtomShape_MNK{})));
  auto a_tensor =
      zipped_divide(t_tensor, a_tile);  // ((AtomN,AtomK),(RestN,RestK))

  auto tv_tensor =
      a_tensor.compose(AtomLayoutSFB_TV{}, _);  // ((ThrV,FrgV),(RestN,RestK))

  auto thr_tile =
      make_tile(_, make_tile(make_layout(size<2>(thr_layout_vmnk)),
                             make_layout(size<3>(thr_layout_vmnk))));
  auto thr_tensor = zipped_divide(
      tv_tensor, thr_tile);  // ((ThrV,(ThrN,ThrK)),(FrgV,(RestN,RestK)))
  return thr_tensor;
}

template <class SFATensor, class ThrMma>
CUTE_HOST_DEVICE constexpr auto partition_SFA(SFATensor&& sfatensor,
                                              ThrMma& thread_mma) {
  auto thr_tensor = make_tensor(static_cast<SFATensor&&>(sfatensor).data(),
                                thrfrg_SFA(sfatensor.layout(), thread_mma));
  auto thr_vmnk = thread_mma.thr_vmnk_;
  auto thr_vmk = make_coord(get<0>(thr_vmnk),
                            make_coord(get<1>(thr_vmnk), get<3>(thr_vmnk)));
  return thr_tensor(thr_vmk, make_coord(_, repeat<rank<1, 1>(thr_tensor)>(_)));
}

template <class SFATensor, class ThrMma>
CUTE_HOST_DEVICE constexpr auto partition_fragment_SFA(SFATensor&& sfatensor,
                                                       ThrMma& thread_mma) {
  using ValTypeSF = typename ThrMma::Atom::Traits::ValTypeSF;
  return make_fragment_like<ValTypeSF>(partition_SFA(sfatensor, thread_mma));
}

template <class SFBTensor, class ThrMma>
CUTE_HOST_DEVICE constexpr auto partition_SFB(SFBTensor&& sfbtensor,
                                              ThrMma& thread_mma) {
  auto thr_tensor = make_tensor(static_cast<SFBTensor&&>(sfbtensor).data(),
                                thrfrg_SFB(sfbtensor.layout(), thread_mma));
  auto thr_vmnk = thread_mma.thr_vmnk_;
  auto thr_vnk = make_coord(get<0>(thr_vmnk),
                            make_coord(get<2>(thr_vmnk), get<3>(thr_vmnk)));
  return thr_tensor(thr_vnk, make_coord(_, repeat<rank<1, 1>(thr_tensor)>(_)));
}

template <class SFBTensor, class ThrMma>
CUTE_HOST_DEVICE constexpr auto partition_fragment_SFB(SFBTensor&& sfbtensor,
                                                       ThrMma& thread_mma) {
  using ValTypeSF = typename ThrMma::Atom::Traits::ValTypeSF;
  return make_fragment_like<ValTypeSF>(partition_SFB(sfbtensor, thread_mma));
}

// (thr_idx,val) -> (M,K) copy layout for smem->rmem SF loads (SFA operand).
template <class TiledMma>
CUTE_HOST_DEVICE constexpr auto get_layoutSFA_TV(TiledMma& mma) {
  auto tile_shape_mnk = tile_shape(mma);
  auto ref_A =
      make_layout(make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk)));
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

  auto atile =
      make_tile(_, make_tile(make_layout(make_shape(size<1>(thr_layout_vmnk),
                                                    size<2>(thr_layout_vmnk)),
                                         make_stride(Int<1>{}, Int<0>{})),
                             _));

  auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
  return thrfrg_SFA(ref_A, mma).compose(atile, _).compose(thridx_2_thrid, _);
}

// (thr_idx,val) -> (N,K) copy layout for smem->rmem SF loads (SFB operand).
template <class TiledMma>
CUTE_HOST_DEVICE constexpr auto get_layoutSFB_TV(TiledMma& mma) {
  auto tile_shape_mnk = tile_shape(mma);
  auto ref_B =
      make_layout(make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

  auto btile =
      make_tile(_, make_tile(make_layout(make_shape(size<1>(thr_layout_vmnk),
                                                    size<2>(thr_layout_vmnk)),
                                         make_stride(Int<0>{}, Int<1>{})),
                             _));

  auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
  return thrfrg_SFB(ref_B, mma).compose(btile, _).compose(thridx_2_thrid, _);
}

}  // namespace cute

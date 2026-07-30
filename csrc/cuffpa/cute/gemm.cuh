#pragma once

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cutlass/numeric_conversion.h>

namespace ffpa_cute {
using namespace cute;

template <typename Layout>
CUTE_DEVICE auto convert_layout_acc_rowcol(Layout acc_layout) {
  auto divided = logical_divide(acc_layout, Shape<_2>{});
  return make_layout(make_layout(get<0, 1>(divided), get<1>(divided)),
                     make_layout(get<0, 0>(divided), get<2>(divided)));
}

template <typename TiledMma, typename Layout>
CUTE_DEVICE auto convert_layout_acc_Aregs(Layout acc_layout) {
  using X = Underscore;
  auto divided = logical_divide(acc_layout, Shape<X, X, _2>{});
  return make_layout(make_layout(get<0>(divided), get<2, 0>(divided)),
                     get<1>(divided), get<2, 1>(divided));
}

template <typename To, typename Engine, typename Layout>
CUTE_DEVICE auto convert_type(Tensor<Engine, Layout> const& tensor) {
  using From = typename Engine::value_type;
  constexpr int kElements = decltype(size(tensor))::value;
  cutlass::NumericArrayConverter<To, From, kElements> convert;
  auto fragment = convert(
      *reinterpret_cast<cutlass::Array<From, kElements> const*>(tensor.data()));
  return make_tensor(make_rmem_ptr<To>(&fragment), tensor.layout());
}

template <typename TensorC, typename TensorA, typename TensorB,
          typename TensorSA, typename TensorSB, typename TiledMma,
          typename TiledCopyA, typename TiledCopyB, typename ThreadCopyA,
          typename ThreadCopyB>
CUTE_DEVICE void gemm_ss(TensorC& acc, TensorA& fragment_a, TensorB& fragment_b,
                         TensorSA const& shared_a, TensorSB const& shared_b,
                         TiledMma tiled_mma, TiledCopyA tiled_copy_a,
                         TiledCopyB tiled_copy_b, ThreadCopyA thread_copy_a,
                         ThreadCopyB thread_copy_b) {
  auto copy_view_a = thread_copy_a.retile_D(fragment_a);
  auto copy_view_b = thread_copy_b.retile_D(fragment_b);
  copy(tiled_copy_a, shared_a(_, _, _0{}), copy_view_a(_, _, _0{}));
  copy(tiled_copy_b, shared_b(_, _, _0{}), copy_view_b(_, _, _0{}));
#pragma unroll
  for (int tile_k = 0; tile_k < size<2>(fragment_a); ++tile_k) {
    if (tile_k + 1 < size<2>(fragment_a)) {
      copy(tiled_copy_a, shared_a(_, _, tile_k + 1),
           copy_view_a(_, _, tile_k + 1));
      copy(tiled_copy_b, shared_b(_, _, tile_k + 1),
           copy_view_b(_, _, tile_k + 1));
    }
    gemm(tiled_mma, fragment_a(_, _, tile_k), fragment_b(_, _, tile_k), acc);
  }
}

template <typename TensorC, typename TensorA, typename TensorB,
          typename TensorSB, typename TiledMma, typename TiledCopyB,
          typename ThreadCopyB>
CUTE_DEVICE void gemm_rs(TensorC& acc, TensorA& fragment_a, TensorB& fragment_b,
                         TensorSB const& shared_b, TiledMma tiled_mma,
                         TiledCopyB tiled_copy_b, ThreadCopyB thread_copy_b) {
  auto copy_view_b = thread_copy_b.retile_D(fragment_b);
  copy(tiled_copy_b, shared_b(_, _, _0{}), copy_view_b(_, _, _0{}));
#pragma unroll
  for (int tile_k = 0; tile_k < size<2>(fragment_a); ++tile_k) {
    if (tile_k + 1 < size<2>(fragment_a)) {
      copy(tiled_copy_b, shared_b(_, _, tile_k + 1),
           copy_view_b(_, _, tile_k + 1));
    }
    gemm(tiled_mma, fragment_a(_, _, tile_k), fragment_b(_, _, tile_k), acc);
  }
}

// No-prefetch variants: load→MMA serial per tile_k, lower register pressure.
template <typename TensorC, typename TensorA, typename TensorB,
          typename TensorSA, typename TensorSB, typename TiledMma,
          typename TiledCopyA, typename TiledCopyB, typename ThreadCopyA,
          typename ThreadCopyB>
CUTE_DEVICE void gemm_ss_nobuf(TensorC& acc, TensorA& fragment_a,
                               TensorB& fragment_b, TensorSA const& shared_a,
                               TensorSB const& shared_b, TiledMma tiled_mma,
                               TiledCopyA tiled_copy_a, TiledCopyB tiled_copy_b,
                               ThreadCopyA thread_copy_a,
                               ThreadCopyB thread_copy_b) {
  auto copy_view_a = thread_copy_a.retile_D(fragment_a);
  auto copy_view_b = thread_copy_b.retile_D(fragment_b);
#pragma unroll
  for (int tile_k = 0; tile_k < size<2>(fragment_a); ++tile_k) {
    copy(tiled_copy_a, shared_a(_, _, tile_k), copy_view_a(_, _, tile_k));
    copy(tiled_copy_b, shared_b(_, _, tile_k), copy_view_b(_, _, tile_k));
    gemm(tiled_mma, fragment_a(_, _, tile_k), fragment_b(_, _, tile_k), acc);
  }
}

template <typename TensorC, typename TensorA, typename TensorB,
          typename TensorSB, typename TiledMma, typename TiledCopyB,
          typename ThreadCopyB>
CUTE_DEVICE void gemm_rs_nobuf(TensorC& acc, TensorA& fragment_a,
                               TensorB& fragment_b, TensorSB const& shared_b,
                               TiledMma tiled_mma, TiledCopyB tiled_copy_b,
                               ThreadCopyB thread_copy_b) {
  auto copy_view_b = thread_copy_b.retile_D(fragment_b);
#pragma unroll
  for (int tile_k = 0; tile_k < size<2>(fragment_a); ++tile_k) {
    copy(tiled_copy_b, shared_b(_, _, tile_k), copy_view_b(_, _, tile_k));
    gemm(tiled_mma, fragment_a(_, _, tile_k), fragment_b(_, _, tile_k), acc);
  }
}

}  // namespace ffpa_cute

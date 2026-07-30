#pragma once

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cutlass/numeric_conversion.h>

namespace ffpa_cute {
using namespace cute;

// convert_layout_acc_rowcol: reshape an MMA C-fragment (MMA=4, MMA_M, MMA_N)
// into ((2, MMA_M), (2, MMA_N)) = (nrow, ncol) so online softmax can scan a
// row. m16n8k16 scatters one logical row across 4 lanes in MMA modes 0/1;
// logical_divide(_, _2) regroups so all columns of a row land in one thread,
// enabling 4-lane __shfl_xor<1>/<2> row max/sum. NOT a general property --
// tied to the m16n8k16 fragment convention (attn_traits.cuh MmaAtom).
// Ref: flash-attention/csrc/flash_attn/src/utils.h:188, softmax.h:139.
template <typename Layout>
CUTE_DEVICE auto convert_layout_acc_rowcol(Layout acc_layout) {
  auto divided = logical_divide(acc_layout, Shape<_2>{});
  return make_layout(make_layout(get<0, 1>(divided), get<1>(divided)),
                     make_layout(get<0, 0>(divided), get<2>(divided)));
}

// convert_layout_acc_Aregs: reshape an MMA C-fragment into A-operand register
// layout (MMA', MMA_M, MMA_N/2). QK and PV share one TiledMma, but C's 4 regs
// pack as (2 row, 2 col) while A needs (2 row, 2 K-slice); the 2-col pair is
// repacked into the K direction by logical_divide(_, _, _2). This lets
// softmax-P feed PV's MMA-A directly without writing back to smem -- the FA
// "register reuse" trick. Depends on m16n8k16 fragment convention, NOT general.
// Ref: flash-attention/csrc/flash_attn/src/utils.h:200, fwd_kernel.h:365.
template <typename TiledMma, typename Layout>
CUTE_DEVICE auto convert_layout_acc_Aregs(Layout acc_layout) {
  using X = Underscore;
  auto divided = logical_divide(acc_layout, Shape<X, X, _2>{});
  return make_layout(make_layout(get<0>(divided), get<2, 0>(divided)),
                     get<1>(divided), get<2, 1>(divided));
}

// convert_type: in-register dtype conversion via NumericArrayConverter.
// Returns a tensor over the SAME memory (make_rmem_ptr<To>), zero copy.
// Needed because the MMA accumulator is f32 (for softmax exp/sum precision) but
// PV's A-operand and the final O store require f16.
// Ref: flash-attention/csrc/flash_attn/src/epilogue/epilogue.hpp.
template <typename To, typename Engine, typename Layout>
CUTE_DEVICE auto convert_type(Tensor<Engine, Layout> const& tensor) {
  using From = typename Engine::value_type;
  constexpr int kElements = decltype(size(tensor))::value;
  cutlass::NumericArrayConverter<To, From, kElements> convert;
  auto fragment = convert(
      *reinterpret_cast<cutlass::Array<From, kElements> const*>(tensor.data()));
  return make_tensor(make_rmem_ptr<To>(&fragment), tensor.layout());
}

// gemm_ss: Shared-Shared GEMM -- A (Q) and B (K) are both ldmatrix'd from smem.
// Software pipeline: retile_D aligns the reg fragment to the TiledCopy source
// view so copy() writes the regs the next MMA consumes; preload tile_k=0, then
// each iter loads tile_k+1 overlapping the current gemm() to hide S->R latency.
// Used by the QK step (S = Q @ K^T).
// Ref: flash-attention/csrc/flash_attn/src/utils.h:166,
// FlashMLA/sm90/helpers.h:97.
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

// gemm_rs: Register-Shared GEMM -- A (P) is already in regs after softmax, only
// B (V) is ldmatrix'd from smem. Symmetric pipeline but only B is preloaded;
// this is the payoff of convert_layout_acc_Aregs: softmax-P never touches smem
// again. Used by the PV step (O = P @ V). Ref:
// flash-attention/csrc/flash_attn/src/utils.h:166, fwd_kernel.h:367.
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

// No-prefetch variants: load->MMA serial per tile_k, lower register pressure
// (no live next-tile regs). Trade-off: no S->R / MMA overlap, slower per tile;
// pick these when RF is tight (large head_dim / many d-chunks). See gemm_ss.
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

// gemm_rs_nobuf: serial load->MMA, A (P) in regs, only B (V) preloaded per
// tile_k. Same RF/throughput trade-off as gemm_ss_nobuf. See gemm_rs.
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

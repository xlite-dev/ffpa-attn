// NVFP4 P two-level quantization fused with online softmax, ported from
// SageAttention3 softmax_fused.h/utils.h. Under online softmax rowmax(P)=1,
// so the first-level per-row scale degenerates to the global constant
// 1/(448*6) and is folded into the exp2 shift; sP2 (ue4m1-group absmax/6
// stored as ue4m3) is what the MMA consumes, and the constant cancels
// exactly between O and row_sum.
// Reference:
// https://github.com/thu-ml/SageAttention/tree/main/sageattention3_blackwell/sageattn3/blackwell/softmax_fused.h
//            https://github.com/thu-ml/SageAttention/tree/main/sageattention3_blackwell/sageattn3/blackwell/utils.h
#pragma once

#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>

namespace ffpa_fp4 {

using namespace cute;

CUTE_DEVICE float ptx_exp2(float x) {
  float y;
  // .ftz keeps this a single MUFU.EX2: the non-ftz form makes ptxas wrap
  // every call in range glue (FSETP -126/-INF, FSEL, FMUL 0.5 chain).
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// 4 floats -> 4 ue4m3 packed into one uint32
CUTE_DEVICE void packed_float_to_ue4m3(float const& f0, float const& f1,
                                       float const& f2, float const& f3,
                                       uint32_t& out) {
  asm volatile(
      "{\n"
      ".reg .b16 lo;\n"
      ".reg .b16 hi;\n"
      "cvt.rn.satfinite.e4m3x2.f32   lo, %2, %1;\n"
      "cvt.rn.satfinite.e4m3x2.f32   hi, %4, %3;\n"
      "mov.b32 %0, {lo, hi};\n"
      "}"
      : "=r"(out)
      : "f"(f0), "f"(f1), "f"(f2), "f"(f3));
}

// 8 floats -> 8 e2m1 packed into one uint32. cvt e2m1x2 requires sm_120+;
// callers live under __CUDA_ARCH__ >= 1200 guards (persist_d kernel body).
CUTE_DEVICE void packed_float_to_e2m1(float const& f0, float const& f1,
                                      float const& f2, float const& f3,
                                      float const& f4, float const& f5,
                                      float const& f6, float const& f7,
                                      uint32_t& out) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
      "}"
      : "=r"(out)
      : "f"(f0), "f"(f1), "f"(f2), "f"(f3), "f"(f4), "f"(f5), "f"(f6), "f"(f7));
#else
  (void)f0;
  (void)f1;
  (void)f2;
  (void)f3;
  (void)f4;
  (void)f5;
  (void)f6;
  (void)f7;
  out = 0u;
#endif
}

// (MmaAtom, MmaM, MmaN) accumulator layout -> (rows, (AtomN, MmaN)) view for
// per-row reductions (row_max / row_sum).
template <class Layout>
CUTE_DEVICE constexpr auto convert_to_reduction_layout(Layout mma_layout) {
  static_assert(rank(mma_layout) == 3,
                "Mma Layout should be (MmaAtom, MmaM, MmaN)");
  static_assert(rank(get<0>(shape(mma_layout))) == 2,
                "MmaAtom should be (AtomN, AtomM)");

  return make_layout(make_layout(get<0, 1>(mma_layout), get<1>(mma_layout)),
                     make_layout(get<0, 0>(mma_layout), get<2>(mma_layout)));
}

// (MmaAtom, MmaM, MmaN) -> ((8-float conversion groups), MmaM, MmaN/2) view:
// consecutive 8 floats of a group map to one packed e2m1 uint32. Requires
// MmaAtomN==8 and MmaAtomM==2 (the 16x32x64 blockscaled atom C fragment).
template <class Layout>
CUTE_DEVICE constexpr auto convert_to_conversion_layout(Layout mma_layout) {
  static_assert(rank(mma_layout) == 3,
                "Mma Layout should be (MmaAtom, MmaM, MmaN)");
  static_assert(rank(get<0>(shape(mma_layout))) == 2,
                "MmaAtom should be (AtomN, AtomM)");

  constexpr int MmaAtomN = size<0, 0>(mma_layout);
  constexpr int MmaAtomM = size<0, 1>(mma_layout);
  constexpr int MmaM = size<1>(mma_layout);
  constexpr int MmaN = size<2>(mma_layout);

  static_assert(MmaAtomN == 8, "MmaAtomN should be 8.");
  static_assert(MmaAtomM == 2, "MmaAtomM should be 2.");
  static_assert(MmaN % 2 == 0, "MmaN should be multiple of 2.");

  auto mma_n_division = zipped_divide(layout<2>(mma_layout), make_tile(_2{}));
  return make_layout(make_layout(layout<0, 0>(mma_layout),
                                 make_layout(layout<0, 1>(mma_layout),
                                             layout<0>(mma_n_division))),
                     layout<1>(mma_layout), layout<1>(mma_n_division));
}

// Online softmax emitting P already scaled into the second-level quant domain:
//   P2 = exp2(S*scale_log2 - m*scale_log2 + log2(1/(448*6)))  in [0, 2688]
//   sP2 = exp2(absmax(scale_log2-domain) - max_scaled + log2(1/6)) in (0, 448]
// The same shuffle chain produces both the 16-element group absmax (sP2) and
// the row max, halving the redundant max reductions.
template <int Rows>
struct SoftmaxFused {
  using TensorT = decltype(make_fragment_like<float>(Shape<Int<Rows>>{}));
  TensorT row_sum, row_max, scores_scale;
  static constexpr float fp8_scalexfp4_scale = 1.f / (448 * 6);
  static constexpr float fp8_scalexfp4_scale_log2 = -11.392317422778762f;
  static constexpr float fp4_scale_log2 = -2.584962500721156f;
  static constexpr int RowReductionThr = 4;

  CUTE_DEVICE SoftmaxFused() {}

  template <bool FirstTile, bool InfCheck = false, typename TensorAcc,
            typename TensorMax>
  CUTE_DEVICE auto online_softmax_with_quant(TensorAcc& acc, TensorMax& AbsMaxP,
                                             const float softmax_scale_log2) {
    Tensor acc_reduction_view =
        make_tensor(acc.data(), convert_to_reduction_layout(acc.layout()));
    Tensor acc_conversion_view =
        make_tensor(acc.data(), convert_to_conversion_layout(acc.layout()));

    if constexpr (FirstTile) {
      fill(row_max, -INFINITY);
      clear(row_sum);
      fill(scores_scale, 1.f);

      CUTE_UNROLL
      for (int mi = 0; mi < size<0>(acc_reduction_view); mi++) {
        CUTE_UNROLL
        for (int ni = 0; ni < size<1, 1>(acc_reduction_view); ni++) {
          // AbsMaxP is a register fragment reused across kv tiles (and,
          // in the persistent kernel, across works): start the reduction
          // from -inf, not from its stale contents.
          AbsMaxP(mi, ni) = -INFINITY;
          CUTE_UNROLL
          for (int ei = 0; ei < size<1, 0>(acc_reduction_view); ei++) {
            AbsMaxP(mi, ni) = fmaxf(AbsMaxP(mi, ni),
                                    acc_reduction_view(mi, make_coord(ei, ni)));
          }
          // merge the neighbour thread's 8-element half-group into 16
          float max_recv = __shfl_xor_sync(0xFFFFFFFFu, AbsMaxP(mi, ni), 1);
          AbsMaxP(mi, ni) = fmaxf(AbsMaxP(mi, ni), max_recv);
          row_max(mi) = fmaxf(row_max(mi), AbsMaxP(mi, ni));
        }
        // finish the row max across the quad
        float max_recv = __shfl_xor_sync(0xFFFFFFFFu, row_max(mi), 2);
        row_max(mi) = fmaxf(row_max(mi), max_recv);

        const float max_scaled =
            InfCheck
                ? (row_max(mi) == -INFINITY
                       ? 0.f
                       : (row_max(mi) * softmax_scale_log2 +
                          fp8_scalexfp4_scale_log2))
                : (row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2);
        CUTE_UNROLL
        for (int g = 0; g < size<1, 1>(acc_reduction_view); g++) {
          const float a = AbsMaxP(mi, g);
          const float sfp2 =
              ptx_exp2(a * softmax_scale_log2 - max_scaled + fp4_scale_log2);
          AbsMaxP(mi, g) = sfp2;
          // Fold the 1/absmax normalize into the exp2 argument: q = p/sP2 =
          // exp2((s-a)*L + log2 6) lands directly in the e2m1 (0,6] domain,
          // so the rcp + per-element FMUL pass between softmax and the pack
          // disappears; row_sum absorbs the sP2 factor via FMA. Clamp a for
          // fully-masked groups so q and row_sum degenerate to 0, not NaN.
          const float a_arg = (a == -INFINITY) ? 0.f : a;
          const float c = fmaf(-a_arg, softmax_scale_log2, -fp4_scale_log2);
          CUTE_UNROLL
          for (int ei = 0; ei < size<1, 0>(acc_reduction_view); ei++) {
            const float q =
                ptx_exp2(fmaf(acc_reduction_view(mi, make_coord(ei, g)),
                              softmax_scale_log2, c));
            acc_reduction_view(mi, make_coord(ei, g)) = q;
            row_sum(mi) = fmaf(q, sfp2, row_sum(mi));
          }
        }
      }
    } else {
      Tensor scores_max_prev = make_fragment_like(row_max);
      cute::copy(row_max, scores_max_prev);
      CUTE_UNROLL
      for (int mi = 0; mi < size<0>(acc_reduction_view); mi++) {
        CUTE_UNROLL
        for (int ni = 0; ni < size<1, 1>(acc_reduction_view); ni++) {
          float local_max = -INFINITY;
          CUTE_UNROLL
          for (int ei = 0; ei < size<1, 0>(acc_reduction_view); ei++) {
            local_max =
                fmaxf(local_max, acc_reduction_view(mi, make_coord(ei, ni)));
          }
          float max_recv = __shfl_xor_sync(0xFFFFFFFFu, local_max, 1);
          AbsMaxP(mi, ni) = fmaxf(local_max, max_recv);
          row_max(mi) = fmaxf(row_max(mi), AbsMaxP(mi, ni));
        }
        float max_recv = __shfl_xor_sync(0xFFFFFFFFu, row_max(mi), 2);
        row_max(mi) = fmaxf(row_max(mi), max_recv);

        float scores_max_cur =
            !InfCheck ? row_max(mi)
                      : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
        scores_scale(mi) = ptx_exp2((scores_max_prev(mi) - scores_max_cur) *
                                    softmax_scale_log2);

        const float max_scaled =
            InfCheck
                ? (row_max(mi) == -INFINITY
                       ? 0.f
                       : (row_max(mi) * softmax_scale_log2 +
                          fp8_scalexfp4_scale_log2))
                : (row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2);
        row_sum(mi) = row_sum(mi) * scores_scale(mi);
        CUTE_UNROLL
        for (int g = 0; g < size<1, 1>(acc_reduction_view); g++) {
          const float a = AbsMaxP(mi, g);
          const float sfp2 =
              ptx_exp2(a * softmax_scale_log2 - max_scaled + fp4_scale_log2);
          AbsMaxP(mi, g) = sfp2;
          const float a_arg = (a == -INFINITY) ? 0.f : a;
          const float c = fmaf(-a_arg, softmax_scale_log2, -fp4_scale_log2);
          CUTE_UNROLL
          for (int ei = 0; ei < size<1, 0>(acc_reduction_view); ei++) {
            const float q =
                ptx_exp2(fmaf(acc_reduction_view(mi, make_coord(ei, g)),
                              softmax_scale_log2, c));
            acc_reduction_view(mi, make_coord(ei, g)) = q;
            row_sum(mi) = fmaf(q, sfp2, row_sum(mi));
          }
        }
      }
    }
    // The group normalize is folded into the exp2 argument above (q already
    // in the e2m1 (0,6] domain); only the pack consumes acc now.
  }

  template <typename TensorAcc>
  CUTE_DEVICE void finalize(TensorAcc& o_store) {
    Tensor o_store_reduction_view = make_tensor(
        o_store.data(), convert_to_reduction_layout(o_store.layout()));
    CUTE_UNROLL
    for (int mi = 0; mi < size(row_max); ++mi) {
      CUTE_UNROLL
      for (int i = 1; i < RowReductionThr; i <<= 1) {
        float sum_recv = __shfl_xor_sync(0xFFFFFFFFu, row_sum(mi), i);
        row_sum(mi) += sum_recv;
      }
      float sum = row_sum(mi);
      float inv_sum = (sum == 0.f || sum != sum) ? 0.f : 1 / sum;
      CUTE_UNROLL
      for (int ni = 0; ni < size<1>(o_store_reduction_view); ni++) {
        o_store_reduction_view(mi, ni) *= inv_sum;
      }
    }
  }

  template <typename TensorAcc>
  CUTE_DEVICE void rescale_o(TensorAcc& o_store, TensorAcc const& o_tmp) {
    Tensor o_store_reduction_view = make_tensor(
        o_store.data(), convert_to_reduction_layout(o_store.layout()));
    Tensor o_tmp_reduction_view =
        make_tensor(o_tmp.data(), convert_to_reduction_layout(o_tmp.layout()));
    CUTE_UNROLL
    for (int mi = 0; mi < size(row_max); ++mi) {
      CUTE_UNROLL
      for (int ni = 0; ni < size<1>(o_store_reduction_view); ni++) {
        o_store_reduction_view(mi, ni) =
            o_store_reduction_view(mi, ni) * scores_scale(mi) +
            o_tmp_reduction_view(mi, ni);
      }
    }
  }
};

// Quantize one mma_k slice of the online-softmax P fragment into the packed
// (e2m1 data + ue4m3 SFP) register pair the blockscaled PV mma consumes as
// its A operand (zip(tOrP, tOrSFP)). mma_k is either a compile-time Int<>
// (_0{}) or a runtime int (v_block+1 prefetch), hence the class template
// parameter. See the persist_d kernel for the fragment-layout details.
template <class MmaK, typename AbsMaxTensor, typename AccConvTensor,
          typename PFragment, typename SfpFragment>
CUTE_DEVICE void quantize_and_pack_p(MmaK mma_k, AbsMaxTensor& AbsMaxP,
                                     AccConvTensor& acc_conversion_view,
                                     PFragment& tOrP, SfpFragment& tOrSFP) {
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
    uint32_t peer_sfp = __shfl_xor_sync(0xFFFFFFFFu, local_sfp, 2);
    if ((quad_id & 1) == 0) {
      uint32_t sfp = (local_sfp & MASK) | ((peer_sfp & MASK) << 8);
      tOrSFP_uint32_view(_0{}, mma_m) = sfp;
    } else {
      uint32_t sfp = (peer_sfp & MASK) | ((local_sfp & MASK) >> 8);
      tOrSFP_uint32_view(_0{}, mma_m) = sfp;
    }
  }
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

}  // namespace ffpa_fp4

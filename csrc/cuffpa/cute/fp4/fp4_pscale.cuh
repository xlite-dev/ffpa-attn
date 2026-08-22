// NVFP4 P two-level quantization fused with online softmax, ported from
// SageAttention3 softmax_fused.h/utils.h. Under online softmax rowmax(P)=1,
// so the first-level per-row scale degenerates to the global constant
// 1/(448*6) and is folded into the exp2 shift; sP2 (per-16-column-group
// absmax/6, stored as ue4m3) is what the MMA consumes, and the constant
// cancels exactly between O and row_sum.
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

  // split-d companion to finalize(): O lives in per-D-chunk fragments, so
  // finalize() folds row_sum on the first chunk and this scales the rest
  // with the already-folded row_sum.
  template <typename TensorAcc>
  CUTE_DEVICE void scale_o(TensorAcc& o_store) {
    Tensor o_store_reduction_view = make_tensor(
        o_store.data(), convert_to_reduction_layout(o_store.layout()));
    CUTE_UNROLL
    for (int mi = 0; mi < size(row_max); ++mi) {
      const float sum = row_sum(mi);
      const float inv_sum = (sum == 0.f || sum != sum) ? 0.f : 1 / sum;
      CUTE_UNROLL
      for (int ni = 0; ni < size<1>(o_store_reduction_view); ni++) {
        o_store_reduction_view(mi, ni) *= inv_sum;
      }
    }
  }

  // split-d lazy rescale: multiply an O chunk in place by scores_scale
  // (the gemm then accumulates the new tile on top).
  template <typename TensorAcc>
  CUTE_DEVICE void rescale_acc(TensorAcc& o_store) {
    Tensor o_store_reduction_view = make_tensor(
        o_store.data(), convert_to_reduction_layout(o_store.layout()));
    CUTE_UNROLL
    for (int mi = 0; mi < size(row_max); ++mi) {
      CUTE_UNROLL
      for (int ni = 0; ni < size<1>(o_store_reduction_view); ni++) {
        o_store_reduction_view(mi, ni) *= scores_scale(mi);
      }
    }
  }
};

// Quantize one mma_k slice of the online-softmax P fragment into the packed
// (e2m1 data + ue4m3 SFP) register pair the blockscaled PV mma consumes as
// its A operand. Why the pack exists: the SM120 blockscaled mma
// (SM120_16x32x64_TN_VS_NVFP4) takes each operand as zip(data, SF) REGISTER
// pairs - there is no smem path for P - so the f32 scores produced by
// softmax must be re-encoded in-register into the exact operand layout:
//   * data: e2m1 has 16 encodings total (2 sign x 8 magnitudes: 0, .5, 1,
//     1.5, 2, 3, 4, 6); P >= 0 so only the 8 non-negative values ever
//     appear - i.e. 3 effective mantissa bits per element. packed_float_to_
//     e2m1 packs 8 floats (8 x 4 bit) into one uint32; recast<uint32_t>
//     exposes exactly that packing granularity.
//   * SFP: one ue4m3 scale per 16-token group (AbsMaxP, produced by
//     online_softmax_with_quant). Two-level math: exp2 already shifted P
//     into the P2 = P*2688 domain (2688 = 448 ue4m3-max x 6 e2m1-max, folded
//     into the exp2 shift by SoftmaxFused), so SFP = group_absmax/6 uses the
//     full e2m1 range {0..6} x SFP <= 448 and the product reconstructs P*2688
//     exactly in the mma's scale multiply; the 2688 cancels between O and
//     row_sum (see the lse formula). packed_float_to_ue4m3 packs 4 scales
//     into one uint32.
//   * Fragment mapping: acc_conversion_view (the QK C-fragment re-laid-out
//     by convert_to_conversion_layout) lists the scores in exactly the order
//     packed_float_to_e2m1 wants; tOrP/tOrSFP carry Traits::LayoutP/LayoutSFP,
//     the SA3 adapter that maps QK C-fragment slots onto the PV A-operand
//     (k = token) register slots, so the quantized registers land where the
//     mma reads them with no shuffles for the data half.
//   * The SFP half DOES need a lane fixup: adjacent quads hold different
//     16-token groups, and the SFA operand wants the pair of group scales
//     interleaved in one register - hence the __shfl_xor(2) byte swap that
//     merges local+peer scales into tOrSFP.
// mma_k is either a compile-time Int<> (_0{}) or a runtime int (v_block+1
// prefetch), hence the class template parameter.
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
// ===== MXFP8 PV path (QK stays NVFP4; PV = e4m3 data + ue8m0 per-32 SF) ===
// Numerical contract (the NVFP4 two-level scheme at 32 granularity):
//   SF = 2^ceil((a32 - m) * L)                ue8m0 power, <= 1
//   q  = exp2((s - m) * L + log2(448) - e)    in (0, 448], e4m3 domain
// where a32 is the per-32-token group absmax and m the row max (both in the
// raw score domain). The ceil at worst halves the q headroom (peak lands in
// [224, 448]) and q never saturates; row_sum absorbs SF via FMA so the 448
// factor cancels exactly between O and row_sum.

// 4 floats -> 4 e4m3 packed into one uint32 (byte i = f_i).
CUTE_DEVICE void packed_float_to_e4m3(float const& f0, float const& f1,
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

// Exact exponent of a positive power of two (SF values are powers of two
// by construction), biased into the ue8m0 byte encoding.
CUTE_DEVICE uint8_t pow2_to_ue8m0(float sf) {
  int e = (__float_as_int(sf) >> 23) - 127;
  return uint8_t(e + 127);
}

// Online softmax for the MXFP8 PV domain. Group granularity 32 = one
// mma-k32 block (local 8 + shfl 1 + shfl 2), so AbsMaxP is indexed by the
// same (row, group) pair the PV k loop walks. finalize/rescale_* come from
// the base class (granularity-independent). AbsMaxP ends holding the group
// SF as an f32 power of two for quantize_and_pack_p_mxfp8.
template <int Rows>
struct SoftmaxFusedMxfp8 : SoftmaxFused<Rows> {
  using Base = SoftmaxFused<Rows>;
  static constexpr float e4m3_full_log2 = 8.807354922057604f;  // log2(448)

  template <bool FirstTile, bool InfCheck = false, typename TensorAcc,
            typename TensorMax>
  CUTE_DEVICE void online_softmax_with_quant(TensorAcc& acc, TensorMax& AbsMaxP,
                                             const float L) {
    Tensor rv =
        make_tensor(acc.data(), convert_to_reduction_layout(acc.layout()));
    Tensor prev_max = make_fragment_like(Base::row_max);
    cute::copy(Base::row_max, prev_max);
    if constexpr (FirstTile) {
      fill(Base::row_max, -INFINITY);
      clear(Base::row_sum);
      fill(Base::scores_scale, 1.f);
    }

    // 32-group absmax in the raw score domain
    CUTE_UNROLL
    for (int mi = 0; mi < size<0>(rv); mi++) {
      CUTE_UNROLL
      for (int g = 0; g < size<1, 1>(rv); g++) {
        float m = -INFINITY;
        CUTE_UNROLL
        for (int ei = 0; ei < size<1, 0>(rv); ei++)
          m = fmaxf(m, rv(mi, make_coord(ei, g)));
        float m16 = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, 1));
        AbsMaxP(mi, g) = fmaxf(m16, __shfl_xor_sync(0xFFFFFFFFu, m16, 2));
      }
      float rm = -INFINITY;
      CUTE_UNROLL
      for (int g = 0; g < size<1, 1>(rv); g++)
        rm = fmaxf(rm, AbsMaxP(mi, g));
      Base::row_max(mi) = fmaxf(prev_max(mi), rm);
      const float cur = (InfCheck && Base::row_max(mi) == -INFINITY)
                            ? 0.f
                            : Base::row_max(mi);
      if constexpr (!FirstTile) {
        Base::scores_scale(mi) = ptx_exp2((prev_max(mi) - cur) * L);
        Base::row_sum(mi) *= Base::scores_scale(mi);
      }
    }

    // Emit q in the e4m3 domain and fold row_sum with the group SF.
    CUTE_UNROLL
    for (int mi = 0; mi < size<0>(rv); mi++) {
      const float rm = Base::row_max(mi);
      CUTE_UNROLL
      for (int g = 0; g < size<1, 1>(rv); g++) {
        const float a = AbsMaxP(mi, g);
        const float a_arg = (a == -INFINITY) ? 0.f : a;
        float shift = (a_arg - rm) * L;
        int e = shift > -127.f ? int(ceilf(shift)) : -127;
        if (e > 128)
          e = 128;
        const float sf = ldexpf(1.f, e);
        AbsMaxP(mi, g) = sf;
        const float c = fmaf(-rm, L, e4m3_full_log2 - float(e));
        CUTE_UNROLL
        for (int ei = 0; ei < size<1, 0>(rv); ei++) {
          const float s = rv(mi, make_coord(ei, g));
          const float q = (s == -INFINITY) ? 0.f : ptx_exp2(fmaf(s, L, c));
          rv(mi, make_coord(ei, g)) = q;
          Base::row_sum(mi) = fmaf(q, sf, Base::row_sum(mi));
        }
      }
    }
  }
};

// Quantize one mma-k32 slice of the softmax-emitted scores into the MXFP8
// PV A operand (e4m3 data + ue8m0 SFA). Unlike the NVFP4 packer the A
// fragment's (m,k) slot map differs from the QK C fragment's (m,n): within
// a quad, value (vm, blk, j) at lane s is needed by lane 2*(blk&1)+(s>>1),
// resolved with three uniform 8-wide shuffle rounds (one dead round per
// lane keeps the warp path lockstep). SFA placement follows the T4 probe:
// Pack the softmaxed scores of one kv tile into the K-fused MXFP8 A
// fragment (64 e4m3 = 16 u32 per thread) plus its ue8m0 SFA bytes.
//
// Fragment contract (SM80-family 16x8 mma, cf. CUTLASS
// SM80_16x8x32_S32S8S8S32_TN ALayout, which this atom extends 4x along K):
// with lane t, quad position q = t&3, row group g = t>>4:
//   A (u32 reg i, byte b): rows m = g + 8*(i&1),
//                           k    = 4q + b + 16*((i>>1)&1) + 32*(i>>2)
//   slot (= 4i + b) = b + 4*m1 + 8*m2 + 16*seg
//     (m1 = i&1, m2 = (i>>1)&1, seg = i>>2): k bits {2,3} pick the lane's
//     quad position, all other k bits pick the slot.
//   SFA byte s of lane t scales row m = g + 8*(t&1), k segment s (the
//     SFALayout thread modes ((2,2,8))/((8,0,1)): t&1 selects the row half,
//     (t>>1)&1 is a redundant broadcast).
//
// The QK NVFP4 C fragment (CLayout ((4,8),((2,4),2))/((32,1),((16,128),8)))
// holds rows m = g + 8*m1 and, at reduction-view index (m1,(n8,mn)), the
// LOGICAL column N = 32*mn + 2q + (n8&1) + 8*(n8>>1). K rows are stored
// under the kv_perm32 interleave, so logical column N carries the score of
// TOKEN t = kv_perm32(N); V^T storage is natural, so the A slot for token
// t is t. The permutation reorders the 5 low bits as (b, j0, j1, q0, q1)
// (b = n8&1, j* = n8>>1 bits, q* = quad bits): the A slot of a value is
// then quad = j1 + 2*q0, u32 i = m1 + 2*q1 + 4*mn, byte = b + 2*j0 - the
// byte and the (m1, mn) pair never change, only the owning lane and the
// u32 bit 1 do. So: pack locally into u32 grouped by j1 (i0 = m1 + 2*j1 +
// 4*mn, one j1 per u32), then each lane PULLS its 16 final u32 with one
// directed __shfl_sync each (source quad = l1 + 2*q1s, source idx = m1 +
// 2*l0 + 4*seg). 16 shuffles and 16 u32 of scratch replace the earlier
// 256-shuffle float-domain routing (which spilled and cost ~15x).
template <typename AbsMaxTensor, typename AccRedTensor, typename PFragment,
          typename SfaFragment>
CUTE_DEVICE void quantize_and_pack_p_mxfp8(int mma_k, AbsMaxTensor& AbsMaxP,
                                           AccRedTensor& acc_reduction_view,
                                           PFragment& tOrP, SfaFragment& tOrSFA,
                                           int lane) {
  uint32_t packed[16];
  CUTLASS_PRAGMA_UNROLL
  for (int m1 = 0; m1 < 2; ++m1)
    CUTLASS_PRAGMA_UNROLL
  for (int mn = 0; mn < 4; ++mn)
    CUTLASS_PRAGMA_UNROLL
  for (int j1 = 0; j1 < 2; ++j1)
    packed_float_to_e4m3(acc_reduction_view(m1, make_coord(4 * j1 + 0, mn)),
                         acc_reduction_view(m1, make_coord(4 * j1 + 1, mn)),
                         acc_reduction_view(m1, make_coord(4 * j1 + 2, mn)),
                         acc_reduction_view(m1, make_coord(4 * j1 + 3, mn)),
                         packed[m1 + 2 * j1 + 4 * mn]);

  const int l0 = lane & 1;
  const int l1 = (lane >> 1) & 1;
  const int quad_base = lane & ~3;
  Tensor tOrP_u32 = recast<uint32_t>(tOrP(_, _, mma_k));
  CUTLASS_PRAGMA_UNROLL
  for (int m1 = 0; m1 < 2; ++m1)
    CUTLASS_PRAGMA_UNROLL
  for (int q1s = 0; q1s < 2; ++q1s)
    CUTLASS_PRAGMA_UNROLL
  for (int seg = 0; seg < 4; ++seg)
    tOrP_u32(m1 + 2 * q1s + 4 * seg) = __shfl_sync(
        0xFFFFFFFFu, packed[m1 + 2 * l0 + 4 * seg], quad_base + l1 + 2 * q1s);

  // SFA: each lane scales its SFALayout row g + 8*(lane&1); the lane's
  // AbsMaxP half for that row is lane&1.
  const int row_sel = lane & 1;
  CUTLASS_PRAGMA_UNROLL
  for (int seg = 0; seg < 4; ++seg)
    tOrSFA(make_coord(_0{}, seg), _0{}, mma_k) =
        cutlass::float_ue8m0_t::bitcast(pow2_to_ue8m0(AbsMaxP(row_sel, seg)));

#ifdef FFPA_PACKER_DUMP
  if (blockIdx.x == 0 && threadIdx.x / 32 == 11 && mma_k == 0) {
    unsigned char sfb[4];
    CUTLASS_PRAGMA_UNROLL
    for (int seg = 0; seg < 4; ++seg) {
      const auto sf = tOrSFA(make_coord(_0{}, seg), _0{}, mma_k);
      memcpy(&sfb[seg], &sf, 1);
    }
    // One printf call = one atomic FIFO record: multi-call segments from
    // different lanes interleave and shred the line.
    printf(
        "PD %d %d %d %d"
        " %08x %08x %08x %08x %08x %08x %08x %08x"
        " %08x %08x %08x %08x %08x %08x %08x %08x"
        " %02x %02x %02x %02x\n",
        (int)blockIdx.x, (int)(threadIdx.x / 32), mma_k, lane, tOrP_u32(0),
        tOrP_u32(1), tOrP_u32(2), tOrP_u32(3), tOrP_u32(4), tOrP_u32(5),
        tOrP_u32(6), tOrP_u32(7), tOrP_u32(8), tOrP_u32(9), tOrP_u32(10),
        tOrP_u32(11), tOrP_u32(12), tOrP_u32(13), tOrP_u32(14), tOrP_u32(15),
        (unsigned)sfb[0], (unsigned)sfb[1], (unsigned)sfb[2], (unsigned)sfb[3]);
  }
#endif
}

// lse smooth-K correction: qkm[row] = dot(Qhat_row_dequant, km) +
// dot(qm_block, km). Qhat is read back from smem (e2m1 x SF), quad-strided
// like fp8's smooth_k_qk_dot: the 4 lanes sharing a row each own one
// quarter of D. Both partial sums must be FULLY quad-reduced before being
// combined - otherwise the 4 lanes of a row disagree and their lse gmem
// stores race (and a lane-local quarter of the qm term leaks in).
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
  }
  c += __shfl_xor_sync(0xffffffff, c, 1);
  c += __shfl_xor_sync(0xffffffff, c, 2);
#pragma unroll
  for (int row = 0; row < kRows; ++row)
    qkm[row] += c;
}

// ---------------------------------------------------------------------------
// M4N2 variants (split_d_m4n2): each N-warp owns half the kBc columns, so
// row max/sum cross N-warps via the smem exchange and P quantization moves
// to the smem-roundtrip readback side.
// ---------------------------------------------------------------------------

// Cross-N-warp online softmax emitting P2-domain f32 scores (persist-D's
// SoftmaxFused math): P2 = P*2688 via the +log2(2688) exp2 shift, so the
// group SF = absmax/6 lands in ue4m3's full-precision range - without the
// shift P-domain group scales flush to ue4m3 subnormals (or zero, which
// drops whole 16-k groups via the inv=0 guard) and O degrades uniformly.
// The 2688 cancels between O and row_sum and is folded back into the lse.
// Protocol mirrors ffpa_cute's online_softmax_fp8_fixed_m4n2: one exchange
// barrier for the max, the sum half is folded later by
// ffpa_cute::finalize_row_sum_m4n2 which reuses the caller's P-roundtrip
// __syncthreads() as its publication barrier.
// row_max stays in the log2 domain (tile_max applies *scale before the
// exchange); row_sum stays in the P2 domain.
// Rescale is eager (threshold 0): the exp2 uses the CURRENT row max, so
// P2 <= 2688 exactly and SF <= 448 never saturates; a lazy threshold would
// let stale-max tiles push P2 to 2^thr * 2688 past ue4m3's 448 ceiling.
template <typename ScoresTensor, typename CoordTensor, int kRows,
          int kNumWarps = 8>
CUTE_DEVICE void online_softmax_p2_m4n2(ScoresTensor& scores,
                                        const CoordTensor& tScS_rc, float scale,
                                        float* row_max, float* row_sum,
                                        float* row_scale, float* smem_exchange,
                                        int warp_id, int lane_id,
                                        float rescale_threshold = 0.0f) {
  const int peer_warp = warp_id ^ 4;
  const bool is_writer = (lane_id % 4 == 0);
  const int row_base = lane_id / 4;
  constexpr int kMaxSlots = kNumWarps * 16;

#pragma unroll
  for (int row = 0; row < kRows; ++row) {
    float tile_max = -INFINITY;
#pragma unroll
    for (int col = 0; col < cute::size<1>(scores); ++col)
      tile_max = fmaxf(tile_max, scores(row, col) * scale);
    tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 1));
    tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 2));
    const int row_local = row_base + row * 8;
    if (is_writer)
      smem_exchange[warp_id * 16 + row_local] = tile_max;
  }
  __syncthreads();

#pragma unroll
  for (int row = 0; row < kRows; ++row) {
    const int row_local = row_base + row * 8;
    const float tile_max = smem_exchange[warp_id * 16 + row_local];
    const float peer_max = smem_exchange[peer_warp * 16 + row_local];
    const float global_tile_max = fmaxf(tile_max, peer_max);

    // row_max lives in the log2 domain throughout (tile_max above already
    // applied *scale), so log2_diff exponentiates directly.
    const float next_max = fmaxf(row_max[row], global_tile_max);
    const float log2_diff = row_max[row] - next_max;
    if (log2_diff >= -rescale_threshold) {
      row_scale[row] = 1.0f;
    } else {
      row_scale[row] = exp2f(log2_diff);
      row_max[row] = next_max;
    }
    // NOTE: row_sum rescale is NOT applied here - finalize_row_sum_m4n2
    // folds it (row_sum*row_scale + local + peer) after the P barrier.

    // exp2 shifted by the current row max plus the P2-domain factor
    // 2688 = 448*6: P2 <= 2688 keeps the group SF = absmax/6 <= 448 (the
    // ue4m3 ceiling) while small-probability groups stay well above the
    // subnormal floor; fully-masked rows clamp the shift to avoid NaN.
    constexpr float kP2ShiftLog2 = 11.392317422778762f;  // log2(448*6)
    const float rm = (row_max[row] == -INFINITY) ? -kP2ShiftLog2 : row_max[row];
    float tile_sum = 0.0f;
#pragma unroll
    for (int col = 0; col < cute::size<1>(scores); ++col) {
      const float p = ptx_exp2(scores(row, col) * scale - rm + kP2ShiftLog2);
      scores(row, col) = p;
      tile_sum += p;
    }
    tile_sum += __shfl_xor_sync(0xffffffff, tile_sum, 1);
    tile_sum += __shfl_xor_sync(0xffffffff, tile_sum, 2);
    if (is_writer)
      smem_exchange[kMaxSlots + warp_id * 16 + row_local] = tile_sum;
  }
}

// Quantize the P2-domain f32 A-fragment (read back from the P smem staging
// tile) into the packed (e2m1 data + ue4m3 SF) register pair the PV mma
// consumes. One-level math here (the SoftmaxFused two-level fold needs the
// full row max before the group SF, unavailable per-N-warp): per 16-k
// group, SF = absmax/6 rounded to ue4m3, elements scaled by 1/SF land in
// the e2m1 [0,6] domain, and the mma's scale multiply reconstructs P2
// exactly (modulo ue4m3 rounding).
//
// Fragment contract (measured via an identity-tensor partition probe, see
// .tmp/fp4-splitd/probe_layout.cu): the 32 readback slots are four 8-elem
// packs laid out flat as [m0,k0-7][m8,k0-7][m0,k+32..39][m8,k+32..39] with
// k-base 8*(lane%4); the quad k-peer lane (shfl_xor 1) holds the matching
// other half of each 16-k group. The scale_vec::4X hardware consumes the SF
// bytes in the same broadcast-interleaved form as quantize_and_pack_p: each
// quad lane must carry all four 16-k group scales with the 0xFF00FF
// shfl_xor(2) weave (see .tmp/fp4-splitd/probe_v3/v5/v12: a plain
// contiguous byte pack leaves 1/8 of the P weights contributing to O).
template <typename Pf32Tensor, typename PAFragment, typename PASFFragment>
CUTE_DEVICE void quantize_pack_a_fp4(Pf32Tensor& tPf32, PAFragment& tPA,
                                     PASFFragment& tPASF) {
  Tensor tPf32_flat = flatten(tPf32);
  Tensor tPA_flat = flatten(tPA);
  Tensor tPA_u32 = recast<uint32_t>(tPA_flat);
  static_assert(decltype(size(tPf32_flat))::value == 32,
                "m4n2 A fragment is 32 elems (16x64/32)");
  static_assert(decltype(size(tPA_flat))::value == 32,
                "m4n2 A fragment is 32 elems (16x64/32)");

  uint8_t sf_bytes[4];
  CUTLASS_PRAGMA_UNROLL
  for (int g = 0; g < 4; ++g) {
    float amax = 0.0f;
    CUTLASS_PRAGMA_UNROLL
    for (int e = 0; e < 8; ++e)
      amax = fmaxf(amax, fabsf(tPf32_flat(g * 8 + e)));
    // The peer lane (quad k-neighbor) holds the group's other 8 elems.
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 1));
    float sf = amax / 6.0f;
    reinterpret_cast<__nv_fp8_e4m3&>(sf_bytes[g]) = __nv_fp8_e4m3(sf);
    sf = float(reinterpret_cast<__nv_fp8_e4m3&>(sf_bytes[g]));
    const float inv = (sf == 0.0f) ? 0.0f : 1.0f / sf;
    uint32_t packed;
    packed_float_to_e2m1(
        tPf32_flat(g * 8 + 0) * inv, tPf32_flat(g * 8 + 1) * inv,
        tPf32_flat(g * 8 + 2) * inv, tPf32_flat(g * 8 + 3) * inv,
        tPf32_flat(g * 8 + 4) * inv, tPf32_flat(g * 8 + 5) * inv,
        tPf32_flat(g * 8 + 6) * inv, tPf32_flat(g * 8 + 7) * inv, packed);
    tPA_u32(g) = packed;
  }
  // SF weave. Hardware contract: SFA providers are quad lanes
  // q=0 (row m=gid) and q=1 (row m=gid+8); lanes q=2/3 never provide.
  // Every thread's local bytes {0,2} hold row-gid group scales (reg0/reg2
  // of its k window) and bytes {1,3} row-gid+8 (reg1/reg3) - independent
  // of q. A provider lane must hold its row's full [g0,g1,g2,g3] vector:
  // group g0/g2 come from its own bytes (base, base+2) and g1/g3 from the
  // q^2 lane (which covers the other 16-k half of the same rows), so
  // byte pairs assemble as [F, S, F, S] at byte offsets base and base+2.
  uint32_t local_sf = uint32_t(sf_bytes[0]) | (uint32_t(sf_bytes[1]) << 8) |
                      (uint32_t(sf_bytes[2]) << 16) |
                      (uint32_t(sf_bytes[3]) << 24);
  uint32_t peer_sf = __shfl_xor_sync(0xFFFFFFFFu, local_sf, 2);
  int const quad_id = threadIdx.x & 3;
  int const base = quad_id & 1;  // 0: row-gid scales in bytes {0,2},
                                 // 1: row-gid+8 scales in bytes {1,3}
  uint32_t const& F = (quad_id & 2) == 0 ? local_sf : peer_sf;
  uint32_t const& S = (quad_id & 2) == 0 ? peer_sf : local_sf;
  uint32_t sfp = ((F >> (8 * base)) & 0xFFu) |
                 (((S >> (8 * base)) & 0xFFu) << 8) |
                 (((F >> (8 * (base + 2))) & 0xFFu) << 16) |
                 (((S >> (8 * (base + 2))) & 0xFFu) << 24);
  reinterpret_cast<uint32_t&>(*tPASF.data()) = sfp;
}

}  // namespace ffpa_fp4

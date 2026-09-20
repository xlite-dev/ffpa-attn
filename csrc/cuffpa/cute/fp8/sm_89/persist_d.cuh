#pragma once

// FP8 persist-D Flash Attention forward (cp.async, sm_89+).
//
// Single-kernel merge of the Sage2-replica experiments: the validated v1
// compute layer (int8/fp8 QK MMA with Q resident in registers via gemm_rs,
// fixed p_scale softmax, two-level PV accumulator) on the Sage2
// single-stage pipeline:
//   - per-thread gmem/smem chunk addressing derived once before the loop;
//     issuing tile t only adds constant strides (K: kBc*kHeadDim bytes,
//     V^T: kBc bytes) that preserve the SW128 swizzle pattern,
//   - single-buffer K/V stages with no alternation: K[t+1] issues right
//     after QK (overlaps softmax + PV), V[t+1] after PV (overlaps the
//     next tile's QK + softmax), and the LDSM address chains become
//     loop-invariant,
//   - the Q tile shares the K stage storage whenever the two tiles have
//     the same shape (D <= 128: both SW128 (128,128) 1B tiles -> 32KB
//     smem total, more L1 left on the SM; D > 128 keeps a separate Q
//     buffer),
//   - a mask-free main loop with the masked variant in the tail.
//
// Two-level PV accumulator (the Sage2 structure): the PV MMA lands in a
// per-tile f16 inst_buf whose only consumer is the float running
// accumulator,
//     RO = RO * row_scale + inst      (one fused fmaf per element)
// so no f16 register ever accumulates across KV tiles -- the f16 overflow
// domain that plagued the persistent-o16 kernels is structurally gone
// (same trade Sage2 makes; adversarial inputs can still saturate a single
// tile's inst_buf: worst case |inst| ~ 448 * sum(P_tile) * amax(V), so a
// fully flat 64-row tile with amax(V) > ~2.3 (448*64*2.3 > f16 max) or
// concentrated scores with larger amax can still inf -- the documented
// FA2-unnormalized-P contract, matching randn-domain safety).
//
// v1 scope: per-block Q/K/V quant, additive attn bias in the raw score
// domain (gmem-direct, mode 0), q_start_row=0, no dropout/smooth_v/hybrid.
// D % 64 == 0; kBc=128 for D <= 128 (matches the 128-col quant blocks, so
// ks/vs index kv_tile directly) and 64 above.

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>

#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cutlass/cutlass.h>

#include "../../gemm.cuh"
#include "../attn_traits.cuh"
#include "../../attn_bias.cuh"
#include "../../softmax.cuh"
#include "../fp8_pscale.cuh"
#include "../reg2reg_8b.cuh"
#include "../smooth_k.cuh"

namespace ffpa_fp8 {

template <typename Traits, typename ElementO, int kHasAttnBias = 0>
__global__ void __launch_bounds__(Traits::kNumThreads, 1)
    persist_d_fwd_cute_fp8_sm89(
        typename Traits::ElementQK* __restrict__ Q,
        typename Traits::ElementQK* __restrict__ K,
        typename Traits::Element* __restrict__ V,  // VT (D, Nkv_pad)
        ElementO* __restrict__ O, float* __restrict__ softmax_lse,
        const float* __restrict__ q_scale, const float* __restrict__ k_scale,
        const float* __restrict__ v_scale, int Nq, int Nkv, int Nh, int Nh_kv,
        int n_rb_q, int n_rb_kv, float scale, int Tc, int causal, int Nkv_pad,
        const float* __restrict__ km = nullptr,
        const float* __restrict__ vm = nullptr,
        const void* __restrict__ attn_bias = nullptr, int attn_bias_dtype = 0,
        long long attn_bias_stride_b = 0, long long attn_bias_stride_h = 0,
        long long attn_bias_stride_m = 0, long long attn_bias_stride_n = 0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  using namespace cute;
  using Element = typename Traits::Element;      // float_e4m3_t (V / P)
  using ElementQK = typename Traits::ElementQK;  // int8 (kQKInt8) or e4m3
  using SmemLayoutQ = typename Traits::SmemLayoutQ;
  using SmemLayoutK = typename Traits::SmemLayoutK;
  using SmemLayoutV = typename Traits::SmemLayoutV;
  using TiledMmaQK = typename Traits::TiledMmaQK;

  constexpr int kBr = Traits::kBr;  // 128
  constexpr int kBc = Traits::kBc;  // D <= 128 ? 128 : 64
  constexpr int kHeadDim = Traits::kHeadDim;
  constexpr int kNumThreads = Traits::kNumThreads;  // 256
  static_assert(Traits::kStagesK == 1 && Traits::kStagesV == 1,
                "single-stage Sage2 pipeline");
  // Same-shape Q/K tiles (D <= 128) share one SW128 buffer: Q's g2s and
  // both smem readers drain before K[0] overwrites it. D > 128 keeps a
  // separate (larger) Q buffer next to the K/V stages.
  constexpr bool kQSharesK = cosize(SmemLayoutQ{}) == cosize(SmemLayoutK{});

  // f16 PV atom (Ada-only m16n8k32 f16-acc), 8 warps over the 128 rows.
  using MmaAtomPVf16 = MMA_Atom<SM89_16x8x32_F16E4M3E4M3F16_TN>;
  using TiledMmaPVf16 = decltype(make_tiled_mma(
      MmaAtomPVf16{}, Layout<Shape<Int<kNumThreads / 32>, _1, _1>>{},
      Tile<Int<kBr>, Int<kHeadDim>, _32>{}));

  constexpr int kQTileElements = cosize(SmemLayoutQ{});
  constexpr int kKTileElements = cosize(SmemLayoutK{});

  const int Nb_id = blockIdx.y / Nh;
  const int Nh_id = blockIdx.y % Nh;
  const int Q_tile_id = blockIdx.x;
  const int group_size = Nh / Nh_kv;
  const int kv_head_idx = Nh_id / group_size;
  const int Br_base = Q_tile_id * kBr;
  const int tid = threadIdx.x;

  if (Br_base >= Nq)
    return;

  const int kv_offset = Nkv - Nq;
  const int causal_thresh_row0 = Br_base + kv_offset;
  const int Tc_eff =
      causal ? min(Tc, ((Br_base + kBr - 1 + kv_offset) / kBc) + 1) : Tc;
  const int mask_start_tile =
      causal ? max(0, (causal_thresh_row0 + 1) / kBc) : INT_MAX;

  const int q_row_offset = (Nb_id * Nh + Nh_id) * Nq;
  const int kv_row_offset = (Nb_id * Nh_kv + kv_head_idx) * Nkv;
  const int q_bh = Nb_id * Nh + Nh_id;
  const int kv_bh = Nb_id * Nh_kv + kv_head_idx;

  // SMEM carve. kQSharesK: [K stage (Q transient) | V stage] = 32KB for
  // D=128. Otherwise [Q | K stage | V stage].
  extern __shared__ __align__(1024) char shm[];
  ElementQK* q_base = reinterpret_cast<ElementQK*>(shm);
  ElementQK* k_base = kQSharesK ? q_base : q_base + kQTileElements;
  Element* v_base = reinterpret_cast<Element*>(k_base + kKTileElements);

  // G2S TiledCopy: 16B cp.async over [rows, 64] segments (one swizzle
  // atom wide, keeps the thread tiling integral for every D%64==0).
  using G2SCopyOp = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
  using G2SCopyAtom = Copy_Atom<Copy_Traits<G2SCopyOp>, Element>;
  constexpr int kSegCols = 64;
  constexpr int kG2SThrN = kSegCols / 16;
  constexpr int kG2SThrM = kNumThreads / kG2SThrN;
  using G2SCopy = decltype(make_tiled_copy(
      G2SCopyAtom{},
      make_layout(make_shape(Int<kG2SThrM>{}, Int<kG2SThrN>{}),
                  make_stride(Int<kG2SThrN>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<16>{}))));
  G2SCopy g2s_copy;
  auto g2s_thr = g2s_copy.get_slice(tid);

  auto mQ = make_tensor(make_gmem_ptr(Q + q_row_offset * kHeadDim),
                        make_shape(Nq, Int<kHeadDim>{}),
                        make_stride(Int<kHeadDim>{}, _1{}));
  auto mK = make_tensor(make_gmem_ptr(K + kv_row_offset * kHeadDim),
                        make_shape(Nkv, Int<kHeadDim>{}),
                        make_stride(Int<kHeadDim>{}, _1{}));
  // VT is flat [B*Nh_kv*kHeadDim, Nkv] with a 16B-aligned row stride; the
  // per-(b,h) base advances by kHeadDim D-rows, i.e. kHeadDim*Nkv_pad bytes.
  auto mV = make_tensor(
      make_gmem_ptr(V + static_cast<long>(kv_bh) * kHeadDim * Nkv_pad),
      make_shape(kHeadDim, Nkv), make_stride(Nkv_pad, _1{}));

  auto g2s_load_q = [&]() {
    auto gQ = local_tile(mQ, Shape<Int<kBr>, Int<kHeadDim>>{},
                         make_coord(Q_tile_id, _0{}));
    auto sQ = make_tensor(make_smem_ptr(q_base), SmemLayoutQ{});
    CUTLASS_PRAGMA_UNROLL
    for (int seg = 0; seg < kHeadDim / kSegCols; ++seg) {
      auto gSeg = local_tile(gQ, Shape<Int<kBr>, Int<kSegCols>>{},
                             make_coord(_0{}, seg));
      auto sSeg = local_tile(sQ, Shape<Int<kBr>, Int<kSegCols>>{},
                             make_coord(_0{}, seg));
      copy(g2s_copy, g2s_thr.partition_S(gSeg), g2s_thr.partition_D(sSeg));
    }
  };

  // G2S destination tensors for the K/V stages (single stage each).
  auto sK_g2s = make_tensor(make_smem_ptr(k_base), SmemLayoutK{});
  auto sV_g2s = make_tensor(make_smem_ptr(v_base), SmemLayoutV{});
  // G2S issue for one K/V tile, pure cute: the gmem slice is an affine
  // function of t (K: +kBc*kHeadDim elements, V^T: +kBc elements), the
  // smem stage tensor is loop-invariant in the single-stage schedule, so
  // nvcc folds the copy addressing into a constant-stride pointer walk --
  // the same algebra v1 recomputed as fresh 64-bit layout slices every
  // tile.
  auto issue_k = [&](int t) {
    auto gK =
        local_tile(mK, Shape<Int<kBc>, Int<kHeadDim>>{}, make_coord(t, _0{}));
    CUTLASS_PRAGMA_UNROLL
    for (int seg = 0; seg < kHeadDim / kSegCols; ++seg) {
      auto gSeg = local_tile(gK, Shape<Int<kBc>, Int<kSegCols>>{},
                             make_coord(_0{}, seg));
      auto sSeg = local_tile(sK_g2s, Shape<Int<kBc>, Int<kSegCols>>{},
                             make_coord(_0{}, seg));
      copy(g2s_copy, g2s_thr.partition_S(gSeg), g2s_thr.partition_D(sSeg));
    }
  };
  auto issue_v = [&](int t) {
    auto gV =
        local_tile(mV, Shape<Int<kHeadDim>, Int<kBc>>{}, make_coord(_0{}, t));
    CUTLASS_PRAGMA_UNROLL
    for (int seg = 0; seg < kBc / kSegCols; ++seg) {
      auto gSeg = local_tile(gV, Shape<Int<kHeadDim>, Int<kSegCols>>{},
                             make_coord(_0{}, seg));
      auto sSeg = local_tile(sV_g2s, Shape<Int<kHeadDim>, Int<kSegCols>>{},
                             make_coord(_0{}, seg));
      copy(g2s_copy, g2s_thr.partition_S(gSeg), g2s_thr.partition_D(sSeg));
    }
  };

  TiledMmaQK tiled_mma_qk;
  TiledMmaPVf16 tiled_mma_pv_f16;
  auto thr_mma_qk = tiled_mma_qk.get_thread_slice(tid);
  auto thr_mma_pv_f16 = tiled_mma_pv_f16.get_thread_slice(tid);

  auto s2r_copy_q =
      make_tiled_copy_A(typename Traits::SmemCopyAtomQK{}, tiled_mma_qk);
  auto s2r_copy_k =
      make_tiled_copy_B(typename Traits::SmemCopyAtomQK{}, tiled_mma_qk);
  auto s2r_copy_v =
      make_tiled_copy_B(typename Traits::SmemCopyAtom{}, tiled_mma_pv_f16);
  auto s2r_thr_q = s2r_copy_q.get_thread_slice(tid);
  auto s2r_thr_k = s2r_copy_k.get_thread_slice(tid);
  auto s2r_thr_v = s2r_copy_v.get_thread_slice(tid);

  // s32 QK score fragment: 2 rows x (kBc/4) cols per thread.
  using ScoreFrag =
      decltype(partition_fragment_C(tiled_mma_qk, Shape<Int<kBr>, Int<kBc>>{}));
  using ScoreRCLayout =
      decltype(ffpa_cute::convert_layout_acc_rowcol(ScoreFrag{}.layout()));
  constexpr int kSCols = decltype(cute::size<1>(
      make_tensor((float*)nullptr, ScoreRCLayout{})))::value;
  constexpr int kSRows = decltype(cute::size<0>(
      make_tensor((float*)nullptr, ScoreRCLayout{})))::value;
  static_assert(kSRows == 2, "");

  // f16 PV inst_buf fragment over the full (kBr, D) tile; the float
  // running accumulator keeps the same layout.
  using OFrag16 = decltype(partition_fragment_C(
      tiled_mma_pv_f16, Shape<Int<kBr>, Int<kHeadDim>>{}));
  using ORC16Layout =
      decltype(ffpa_cute::convert_layout_acc_rowcol(OFrag16{}.layout()));
  constexpr int kORows = decltype(cute::size<0>(
      make_tensor((float*)nullptr, ORC16Layout{})))::value;
  constexpr int kOCols = decltype(cute::size<1>(
      make_tensor((float*)nullptr, ORC16Layout{})))::value;
  constexpr int kOElems = decltype(cute::size(OFrag16{}))::value;
  static_assert(kORows == 2, "");

  auto sQ = make_tensor(make_smem_ptr(q_base), SmemLayoutQ{});
  // Single-stage K/V smem tensors and s2r partitions, built once. With no
  // stage alternation the LDSM address chains are loop-invariant, so nvcc
  // hoists them out of the KV loop.
  auto sK_s0 = make_tensor(make_smem_ptr(k_base), SmemLayoutK{});
  auto sV_s0 = make_tensor(make_smem_ptr(v_base), SmemLayoutV{});
  auto tKsK0 = s2r_thr_k.partition_S(sK_s0);
  auto tVsV0 = s2r_thr_v.partition_S(sV_s0);

  // Coordinate tensor for softmax indexing.
  auto cS = make_identity_tensor(Shape<Int<kBr>, Int<kBc>>{});
  auto tScS = thr_mma_qk.partition_C(cS);
  auto tScS_rc = make_tensor(
      tScS.data(), ffpa_cute::convert_layout_acc_rowcol(tScS.layout()));

  const float scale_orig = scale;
  scale *= FFPA_M_LOG2E;

  const float qs = q_scale[static_cast<long>(q_bh) * n_rb_q + Q_tile_id];

  float row_max[kORows];
  float row_sum[kORows];
  float qkm[kORows];
  float row_scale[kORows];
#pragma unroll
  for (int r = 0; r < kORows; ++r) {
    row_max[r] = -INFINITY;
    row_sum[r] = 0.0f;
    qkm[r] = 0.0f;
    row_scale[r] = 1.0f;
  }

  // Float running accumulator (the upper level of the two-level RO).
  float ro[kOElems];
#pragma unroll
  for (int i = 0; i < kOElems; ++i)
    ro[i] = 0.0f;

  const bool smooth_lse = (softmax_lse != nullptr) && (km != nullptr);

  // Prologue, Sage2 style. Commit order Q(1), K[0](2), V[0](3), then per
  // tile K[t] = group 2+2t, V[t] = group 3+2t, so the steady-state waits
  // are constant: wait<1> settles K[t], wait<0> settles V[t]. When Q
  // shares the K storage, both of Q's smem readers (smooth-K dot, Q s2r)
  // drain CTA-wide before K[0] issues.
  auto tCrQ = thr_mma_qk.partition_fragment_A(sQ);
  auto tQsQ_s2r = s2r_thr_q.partition_S(sQ);
  {
    g2s_load_q();
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();  // Q settle, CTA-visible

    // Smooth-K dot correction (lse += scale_orig * qs * qkm[row]); reads
    // the smem Q tile before the A-fragment is moved to regs.
    if (smooth_lse)
      smooth_k_qk_dot<kHeadDim, kORows>(
          sQ, tScS_rc, km + static_cast<long>(kv_bh) * kHeadDim, qkm);

    // Q s2r once: the A fragment is loop-invariant (persist-D), so every
    // QK step below runs as gemm_rs (K-only smem loads).
    auto tXrQ = s2r_thr_q.retile_D(tCrQ);
#pragma unroll
    for (int tile_k = 0; tile_k < size<2>(tCrQ); ++tile_k)
      copy(s2r_copy_q, tQsQ_s2r(_, _, tile_k), tXrQ(_, _, tile_k));

    __syncthreads();  // Q storage drained CTA-wide -> K[0] may overwrite
    if (Tc_eff > 0) {
      issue_k(0);
      cp_async_fence();
      issue_v(0);
      cp_async_fence();
    }
  }

  PackC8bitToA8bitPermVT perm_pack;
  using PLayer = Layout<Shape<Shape<_4, _2, _2>, _1, Int<kBc / 32>>>;

  // One KV tile. `masked == false` is a compile-time constant at the
  // main-loop call site, so the mask code only exists in the tail copy.
  auto process_tile = [&](int kv_tile, bool masked) {
    // kBc == the quant block width for D <= 128; one scale per tile.
    const float ks = k_scale[static_cast<long>(kv_bh) * n_rb_kv + kv_tile];
    const float vs = v_scale[static_cast<long>(kv_bh) * n_rb_kv + kv_tile];
    const float p_quant_scale = vs * kE4m3Max;

    // ---- QK: K-only smem loads feed the resident Q A-fragment ----
    cp_async_wait<1>();  // settles K[t] = group 2+2t
    __syncthreads();

    auto tCrK = thr_mma_qk.partition_fragment_B(sK_s0);
    ScoreFrag tCrS;
    clear(tCrS);
    ffpa_cute::gemm_rs(tCrS, tCrQ, tCrK, tKsK0, tiled_mma_qk, s2r_copy_k,
                       s2r_thr_k);

    // int8 QK: cast the s32 acc to f32 in place (identity view on the
    // e4m3 path, whose accumulator is already f32).
    auto tCrSf =
        make_tensor(reinterpret_cast<float*>(tCrS.data()), tCrS.layout());
    if constexpr (Traits::kQKInt8) {
#pragma unroll
      for (int i = 0; i < size(tCrS); ++i)
        tCrSf(i) = static_cast<float>(tCrS(i));
    }
    auto scores = make_tensor(
        tCrSf.data(), ffpa_cute::convert_layout_acc_rowcol(tCrS.layout()));

    // Additive attn bias in the RAW score domain (gmem-direct, mode 0).
    if constexpr (kHasAttnBias) {
      float bias_inv[kSRows];
#pragma unroll
      for (int row = 0; row < kSRows; ++row)
        bias_inv[row] = 1.0f / (qs * ks * scale_orig);
      const int bias_q_valid = min(kBr, Nq - Br_base);
      const int bias_kv_valid = min(kBc, Nkv - kv_tile * kBc);
      const bool full_tile = bias_q_valid >= kBr && bias_kv_valid >= kBc;
      if (__builtin_expect(full_tile, 1))
        ffpa_cute::apply_attn_bias_quant_rowcol<
            decltype(scores), decltype(tScS_rc), kSRows, kSCols, false>(
            scores, tScS_rc, attn_bias, attn_bias_dtype, attn_bias_stride_b,
            attn_bias_stride_h, attn_bias_stride_m, attn_bias_stride_n, Nb_id,
            Nh_id, Br_base, kv_tile, kBc, bias_inv, bias_q_valid,
            bias_kv_valid);
      else
        ffpa_cute::apply_attn_bias_quant_rowcol<
            decltype(scores), decltype(tScS_rc), kSRows, kSCols, true>(
            scores, tScS_rc, attn_bias, attn_bias_dtype, attn_bias_stride_b,
            attn_bias_stride_h, attn_bias_stride_m, attn_bias_stride_n, Nb_id,
            Nh_id, Br_base, kv_tile, kBc, bias_inv, bias_q_valid,
            bias_kv_valid);
    }

    // Boundary masking (kv_valid / causal) in the raw score domain.
    const int kv_valid = Nkv - kv_tile * kBc;
    bool tile_needs_mask = false;
    if (masked) {
      tile_needs_mask = (kv_valid < kBc) || (kv_tile >= mask_start_tile);
      if (tile_needs_mask) {
#pragma unroll
        for (int row = 0; row < kSRows; ++row) {
          const int q_pos = Br_base + get<0>(tScS_rc(row, 0)) + kv_offset;
#pragma unroll
          for (int col = 0; col < kSCols; ++col) {
            float s = scores(row, col) * qs * ks * scale;
            if (get<1>(tScS_rc(row, col)) >= kv_valid)
              s = -INFINITY;
            if (kv_tile >= mask_start_tile) {
              const int k_pos = kv_tile * kBc + get<1>(tScS_rc(row, col));
              if (k_pos > q_pos)
                s = -INFINITY;
            }
            scores(row, col) = s;
          }
        }
      }
    }

    // Fixed P quant scale: per-block V folds vs into P (P8 = P*vs*448),
    // so vs cancels in the PV MMA and RO lives in one fixed domain.
    // kMaxScaleAfter holds only for the int8-QK + f16-inst combo.
    online_softmax_fp8_fixed<true, decltype(scores), decltype(tScS_rc), kORows,
                             Traits::kQKInt8>(
        scores, tScS_rc, tile_needs_mask ? 1.0f : qs * ks * scale, row_max,
        row_sum, row_scale, log2f(p_quant_scale), 1.0f / p_quant_scale,
        Traits::kRescaleThreshold);

    // f32 score storage -> packed e4m3 PV A operand (perm pack binds the
    // kVTPerm V^T from the quantize pre-kernel).
    auto tCrP = make_tensor(reinterpret_cast<Element*>(tCrSf.data()), PLayer{});
    quantize_p_frag_prescaled(tCrSf, perm_pack);

    // ---- V settle, then PV: per-tile f16 inst_buf -> RO absorb ----
    cp_async_wait<0>();  // settles V[t] = group 3+2t
    __syncthreads();

    // K storage drained (QK read it before this sync): release it now so
    // the K[t+1] load overlaps softmax + PV (Sage2 single-buffer schedule).
    const int kv_next = kv_tile + 1;
    if (kv_next < Tc_eff)
      issue_k(kv_next);
    cp_async_fence();  // empty group on drain keeps the depths exact

    auto tCrV = thr_mma_pv_f16.partition_fragment_B(sV_s0);

    pscale_rowsum_mma(tCrP, row_sum, 1.0f / p_quant_scale);

    OFrag16 inst;
    clear(inst);
    ffpa_cute::gemm_rs(inst, tCrP, tCrV, tVsV0, tiled_mma_pv_f16, s2r_copy_v,
                       s2r_thr_v);

    // RO = RO*rs + inst, one fused fmaf per element; rs is skipped on
    // the first tile (RO is zero anyway).
    {
      auto ro_rc = make_tensor(make_rmem_ptr(ro), ORC16Layout{});
      auto inst_rc = make_tensor(inst.data(), ORC16Layout{});
#pragma unroll
      for (int row = 0; row < kORows; ++row) {
        const float rs =
            (kv_tile > 0 && row_scale[row] < 1.0f) ? row_scale[row] : 1.0f;
#pragma unroll
        for (int col = 0; col < kOCols; ++col)
          ro_rc(row, col) = fmaf(ro_rc(row, col), rs, float(inst_rc(row, col)));
      }
    }

    // ---- release V and issue V[t+1] ----
    // V[t+1] then overlaps the next tile's K wait + QK + softmax. Drain
    // tiles commit an empty group so the depth-limited waits stay exact.
    __syncthreads();
    if (kv_next < Tc_eff)
      issue_v(kv_next);
    cp_async_fence();
  };

  // tail_start: first tile needing any per-element mask (causal diagonal
  // or the out-of-bounds tail); tiles before it run a mask-free body.
  const int oob_start = (Nkv % kBc == 0) ? Tc : (Nkv - 1) / kBc;
  const int tail_start = min(mask_start_tile, oob_start);

#pragma unroll 1
  for (int kv_tile = 0; kv_tile < tail_start && kv_tile < Tc_eff; ++kv_tile)
    process_tile(kv_tile, false);
#pragma unroll 1
  for (int kv_tile = tail_start; kv_tile < Tc_eff; ++kv_tile)
    process_tile(kv_tile, true);
  cp_async_wait<0>();

  // ---- Epilogue: dequant, normalize, store ----
  {
    auto mO = make_tensor(
        make_gmem_ptr(O + (Nb_id * Nh * Nq * kHeadDim) + Nh_id * Nq * kHeadDim),
        make_shape(Nq, Int<kHeadDim>{}), make_stride(Int<kHeadDim>{}, _1{}));
    auto gO = local_tile(mO, Shape<Int<kBr>, Int<kHeadDim>>{},
                         make_coord(Q_tile_id, _0{}));
    auto tCgO = thr_mma_pv_f16.partition_C(gO);
    auto cO = make_identity_tensor(Shape<Int<kBr>, Int<kHeadDim>>{});
    auto tOcO = thr_mma_pv_f16.partition_C(cO);

    auto ro_rc = make_tensor(make_rmem_ptr(ro), ORC16Layout{});
    auto tOHalf = ffpa_cute::convert_type<ElementO>(OFrag16{});
    auto tOH_rc = make_tensor(
        tOHalf.data(), ffpa_cute::convert_layout_acc_rowcol(tOHalf.layout()));
#pragma unroll
    for (int row = 0; row < kORows; ++row) {
      const float inv_sum = (row_sum[row] == 0.0f) ? 1.0f : 1.0f / row_sum[row];
      const float mul = inv_sum * kFP8FixedPScale;
#pragma unroll
      for (int col = 0; col < kOCols; ++col)
        tOH_rc(row, col) = ElementO(ro_rc(row, col) * mul);
    }

    if (Br_base + kBr <= Nq) {
      copy(tOHalf, tCgO);
    } else {
#pragma unroll
      for (int i = 0; i < size(tOHalf); ++i) {
        const int global_row = Br_base + get<0>(tOcO(i));
        if (global_row < Nq)
          tCgO(i) = tOHalf(i);
      }
    }

    if (softmax_lse != nullptr) {
      const int lse_base = Nb_id * Nh * Nq + Nh_id * Nq;
#pragma unroll
      for (int row = 0; row < kORows; ++row) {
        float lse = (row_max[row] + log2f(row_sum[row])) * FFPA_M_LN2;
        if (smooth_lse)
          lse += scale_orig * qs * qkm[row];
        const int global_row = Br_base + get<0>(tScS_rc(row, 0));
        if (global_row < Nq)
          softmax_lse[lse_base + global_row] = lse;
      }
    }
  }
#endif  // defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
}

}  // namespace ffpa_fp8

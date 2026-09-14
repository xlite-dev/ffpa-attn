#pragma once

// FP8 persist-D Flash Attention forward (cp.async, sm_89+).
// Ada port of the fp8 persist-D geometry: no TMA / mbarrier / WS on sm_89,
// so the sm120 fp8 persist-D compute layer (int8 QK MMA, fixed p_scale
// softmax, fp8 f16-acc PV) rides the sm80 non-WS cp.async loader: 256T
// fully synchronous, K/V tiles committed as per-tile cp.async group pairs
// and waited by FIFO group counting (QK needs group 1+2t, PV group 2+2t of
// the per-thread sequence Q,K0,V0,K1,V1,...).
// Q is s2r'd once and every QK step runs gemm_rs (K-only smem loads). V is
// pre-transposed (D x Nkv, kVTPerm column permutation) by the quantize
// pre-kernel, so the PV B operand loads with the non-transposed LDSM atom
// and the P pack is the reorg-free perm variant.
// Divergences from the sm120 kernel: bias is gmem-direct only (mode 0; the
// TMA tile modes 1-3 need mbarrier handoff); dropout is unsupported (same
// contract as the sm120 fp8 path); q_start_row/hybrid is out of scope (v1).

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

template <typename Traits, typename ElementO, int kHasAttnBias = 0,
          bool kPVAccF16 = true, bool kVPerChannel = false,
          bool kQKPerThread = false, bool kReorgFree = true>
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
  using TiledMmaPV = typename Traits::TiledMmaPV;
  using SmemCopyAtomQK = typename Traits::SmemCopyAtomQK;
  using SmemCopyAtom = typename Traits::SmemCopyAtom;

  constexpr int kBr = Traits::kBr;
  constexpr int kBc = Traits::kBc;
  constexpr int kHeadDim = Traits::kHeadDim;
  constexpr int kStages = Traits::kStagesK;
  static_assert(Traits::kStagesK == Traits::kStagesV,
                "the single kStages pool partition assumes K/V stage parity");
  constexpr int kNumThreads = Traits::kNumThreads;
  constexpr int kNumWarps = Traits::kNumWarps;

  // Traits::TiledMmaPVf16 uses the Blackwell SM120 atom; sm89 needs its own
  // f16-acc atom. Same m16n8k32 shape and A/B layouts as the f32 atom, so
  // the B-operand smem plumbing and the P A-operand packing are shared.
  using MmaAtomPVf16 = MMA_Atom<SM89_16x8x32_F16E4M3E4M3F16_TN>;
  using TiledMmaPVf16 = decltype(make_tiled_mma(
      MmaAtomPVf16{}, Layout<Shape<Int<kNumWarps>, _1, _1>>{},
      Tile<Int<kBr>, Int<kHeadDim>, _32>{}));

  constexpr int kQTileElements = cosize(SmemLayoutQ{});
  constexpr int kKTileElements = cosize(SmemLayoutK{});
  constexpr int kVTileElements = cosize(SmemLayoutV{});

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
  const int q_tile_abs = Q_tile_id;

  // SMEM: [Q persist | K stages | V stages], 1B per elem (int8 or e4m3).
  extern __shared__ __align__(1024) char shm[];
  ElementQK* q_base = reinterpret_cast<ElementQK*>(shm);
  ElementQK* k_base = q_base + kQTileElements;
  Element* v_base =
      reinterpret_cast<Element*>(k_base + kStages * kKTileElements);

  // G2S TiledCopy: 16B cp.async over [rows, 64] segments (one swizzle
  // atom wide, keeps the thread tiling integral for every D%64==0).
  // 1B elems: 16 elements per 16B copy; kSegCols=64 -> 4 copies/row.
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

  // Gmem tensors. Q/K stay (rows, D) row-major int8/fp8; V is the
  // pre-transposed VT (D, Nkv_pad) e4m3 output of the quantize kernel.
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

  // G2S helpers: Q once (persist), K/V per kv tile into their stage.
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
  auto g2s_load_k = [&](int kv_tile_idx, int stage) {
    auto gK = local_tile(mK, Shape<Int<kBc>, Int<kHeadDim>>{},
                         make_coord(kv_tile_idx, _0{}));
    auto sK = make_tensor(make_smem_ptr(k_base + stage * kKTileElements),
                          SmemLayoutK{});
    CUTLASS_PRAGMA_UNROLL
    for (int seg = 0; seg < kHeadDim / kSegCols; ++seg) {
      auto gSeg = local_tile(gK, Shape<Int<kBc>, Int<kSegCols>>{},
                             make_coord(_0{}, seg));
      auto sSeg = local_tile(sK, Shape<Int<kBc>, Int<kSegCols>>{},
                             make_coord(_0{}, seg));
      copy(g2s_copy, g2s_thr.partition_S(gSeg), g2s_thr.partition_D(sSeg));
    }
  };
  auto g2s_load_v = [&](int kv_tile_idx, int stage) {
    // VT is [D, Nkv] row-major (kv contiguous), so the tile [kHeadDim, kBc]
    // loads as 64-column segments along the kv direction, same G2S shape as
    // Q/K (rows x 64).
    auto gV = local_tile(mV, Shape<Int<kHeadDim>, Int<kBc>>{},
                         make_coord(_0{}, kv_tile_idx));
    auto sV = make_tensor(make_smem_ptr(v_base + stage * kVTileElements),
                          SmemLayoutV{});
    CUTLASS_PRAGMA_UNROLL
    for (int seg = 0; seg < kBc / kSegCols; ++seg) {
      auto gSeg = local_tile(gV, Shape<Int<kHeadDim>, Int<kSegCols>>{},
                             make_coord(_0{}, seg));
      auto sSeg = local_tile(sV, Shape<Int<kHeadDim>, Int<kSegCols>>{},
                             make_coord(_0{}, seg));
      copy(g2s_copy, g2s_thr.partition_S(gSeg), g2s_thr.partition_D(sSeg));
    }
  };

  // Dual TiledMma (fp8/int8 QK + fp8 PV; the f16-acc PV shares the f32
  // B-operand layout, so one partition set serves both).
  TiledMmaQK tiled_mma_qk;
  TiledMmaPV tiled_mma_pv;
  TiledMmaPVf16 tiled_mma_pv_f16;
  auto thr_mma_qk = tiled_mma_qk.get_thread_slice(tid);
  auto thr_mma_pv = tiled_mma_pv.get_thread_slice(tid);
  auto thr_mma_pv_f16 = tiled_mma_pv_f16.get_thread_slice(tid);

  // S2R copy atoms.
  auto s2r_copy_q = make_tiled_copy_A(SmemCopyAtomQK{}, tiled_mma_qk);
  auto s2r_copy_k = make_tiled_copy_B(SmemCopyAtomQK{}, tiled_mma_qk);
  auto s2r_copy_v = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma_pv);
  auto s2r_copy_v_f16 = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma_pv_f16);
  auto s2r_thr_q = s2r_copy_q.get_thread_slice(tid);
  auto s2r_thr_k = s2r_copy_k.get_thread_slice(tid);
  auto s2r_thr_v = s2r_copy_v.get_thread_slice(tid);
  auto s2r_thr_v_f16 = s2r_copy_v_f16.get_thread_slice(tid);

  // O fragment layout (full-D persist accumulator).
  using OFragType = decltype(partition_fragment_C(
      tiled_mma_pv, Shape<Int<kBr>, Int<kHeadDim>>{}));
  using OFragLayout = typename OFragType::layout_type;
  constexpr int kOElemsPerFrag = decltype(size(OFragType{}))::value;
  constexpr int kORows = decltype(size<0>(
      make_tensor((float*)nullptr,
                  ffpa_cute::convert_layout_acc_rowcol(OFragLayout{}))))::value;
  constexpr int kOCols = decltype(size<1>(
      make_tensor((float*)nullptr,
                  ffpa_cute::convert_layout_acc_rowcol(OFragLayout{}))))::value;

  // Coordinate tensor for softmax indexing.
  auto cS = make_identity_tensor(Shape<Int<kBr>, Int<kBc>>{});
  auto tScS = thr_mma_qk.partition_C(cS);
  auto tScS_rc = make_tensor(
      tScS.data(), ffpa_cute::convert_layout_acc_rowcol(tScS.layout()));
  constexpr int kSRows = decltype(size<0>(tScS_rc))::value;
  constexpr int kSCols = decltype(size<1>(tScS_rc))::value;

  const float scale_orig = scale;
  scale *= FFPA_M_LOG2E;

  // Per-block Q/K dequant scales (kQKPerThread=false: one scalar per
  // kBr-row Q block / kBc-col K block). Per-thread mode keeps qs_arr
  // fragment-aligned; v1 pins per-block (fp8_q_quant_method=0).
  const float qs = q_scale[static_cast<long>(q_bh) * n_rb_q + q_tile_abs];

  float row_max[kORows];
  float row_sum[kORows];
  float qkm[kORows];
#pragma unroll
  for (int r = 0; r < kORows; ++r) {
    row_max[r] = -INFINITY;
    row_sum[r] = 0.0f;
    qkm[r] = 0.0f;
  }

  float o_acc[kOElemsPerFrag];
#pragma unroll
  for (int i = 0; i < kOElemsPerFrag; ++i)
    o_acc[i] = 0.0f;

  // Initial loads: Q group, then K/V[0..S-1] as per-tile pairs (same FIFO
  // invariant as the sm80 fp16 persist-D: at kv tile t K[t] is group 1+2t,
  // V[t] group 2+2t -> wait<2S-1> settles K[t], wait<2S-2> settles V[t]).
  {
    g2s_load_q();
    cp_async_fence();
#pragma unroll
    for (int s = 0; s < kStages; ++s) {
      if (s < Tc_eff) {
        g2s_load_k(s, s);
        cp_async_fence();
        g2s_load_v(s, s);
        cp_async_fence();
      }
    }
    // Prologue wait: settle only Q + K[0] (the t=0 QK operands); the newer
    // K[1..S-1]/V[0..S-1] groups stay in flight and land under the t=0
    // compute. Short-Tc grids submit fewer than 2S+1 groups, which would
    // make the depth-limited wait pass immediately, so they settle
    // everything (the in-loop FIFO waits then pass trivially).
    if (Tc_eff >= kStages)
      cp_async_wait<kStages * 2 - 1>();
    else
      cp_async_wait<0>();
    __syncthreads();
  }

  // Smooth-K dot correction (lse += scale_orig * qs * qkm[row]); reads the
  // smem Q tile before the A-fragment is moved to regs.
  const bool smooth_lse = (softmax_lse != nullptr) && (km != nullptr);
  if (smooth_lse) {
    auto sQ = make_tensor(make_smem_ptr(q_base), SmemLayoutQ{});
    smooth_k_qk_dot<kHeadDim, kORows>(
        sQ, tScS_rc, km + static_cast<long>(kv_bh) * kHeadDim, qkm);
  }

  // Q s2r once: the A fragment is loop-invariant (persist-D), so every QK
  // step below runs as gemm_rs (K-only smem loads).
  auto sQ = make_tensor(make_smem_ptr(q_base), SmemLayoutQ{});
  auto tCrQ = thr_mma_qk.partition_fragment_A(sQ);
  auto tQsQ_s2r = s2r_thr_q.partition_S(sQ);
  {
    auto tXrQ = s2r_thr_q.retile_D(tCrQ);
#pragma unroll
    for (int tile_k = 0; tile_k < size<2>(tCrQ); ++tile_k)
      copy(s2r_copy_q, tQsQ_s2r(_, _, tile_k), tXrQ(_, _, tile_k));
  }

  ReorgC8bitToA8bit reorg;
  PackC8bitToA8bitPermVT perm_pack;

#pragma unroll 1
  for (int kv_tile = 0; kv_tile < Tc_eff; ++kv_tile) {
    const int k_stg = kv_tile % kStages;
    const int v_stg = kv_tile % kStages;

    // K scale: per-block (1 per kBc-col block).
    const float ks = k_scale[static_cast<long>(kv_bh) * n_rb_kv + kv_tile];
    // V scale: per-block (P uses fixed 448 scale; epilogue dequants).
    const float vs = v_scale[static_cast<long>(kv_bh) * n_rb_kv + kv_tile];

    // QK GEMM: gemm_rs with the loop-invariant Q A-fragment in regs.
    cp_async_wait<kStages * 2 - 1>();
    __syncthreads();

    auto sK = make_tensor(make_smem_ptr(k_base + k_stg * kKTileElements),
                          SmemLayoutK{});
    auto tCrK = thr_mma_qk.partition_fragment_B(sK);
    auto tKsK_s2r = s2r_thr_k.partition_S(sK);

    auto tCrS = partition_fragment_C(tiled_mma_qk, Shape<Int<kBr>, Int<kBc>>{});
    clear(tCrS);
    ffpa_cute::gemm_rs(tCrS, tCrQ, tCrK, tKsK_s2r, tiled_mma_qk, s2r_copy_k,
                       s2r_thr_k);

    // int8 QK: cast the s32 acc to f32 in place (identity on the e4m3
    // path); S enters the log2 domain with qs*ks folded in below.
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
    const bool tile_needs_mask =
        (kv_valid < kBc) || (kv_tile >= mask_start_tile);
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

    float row_scale[kORows];
    // Fixed P quant scale: per-block V folds vs into P (P8 = P*vs*448), so
    // vs cancels in the PV MMA and o_acc lives in one fixed domain.
    const float p_quant_scale = vs * kE4m3Max;
    // Fixed mode (kPQuantPerRow=false): fold the P quant scale into the
    // exp2 offset; row_sum is recovered from a tensor-core row-sum over the
    // quantized P (pscale_rowsum_mma), so kRowSumViaMma=true. kMaxScaleAfter
    // mirrors the sm120 gate (only the int8 QK + f16-acc PV combo defers the
    // max pass scaling).
    constexpr bool kMaxScaleAfter = Traits::kQKInt8 && kPVAccF16;
    const float s_dequant = qs * ks;
    const float softmax_scale_eff = tile_needs_mask ? 1.0f : s_dequant * scale;
    online_softmax_fp8_fixed<true, decltype(scores), decltype(tScS_rc), kORows,
                             kMaxScaleAfter>(
        scores, tScS_rc, softmax_scale_eff, row_max, row_sum, row_scale,
        log2f(p_quant_scale), 1.0f / p_quant_scale, Traits::kRescaleThreshold);

    // Rescale o_acc (online softmax, thread-private per-row decision).
    constexpr bool kFuseRescaleAbsorb = kPVAccF16;
    if (kv_tile > 0 && !kFuseRescaleAbsorb) {
      auto tCrO = make_tensor(make_rmem_ptr(o_acc), OFragLayout{});
      auto tCrO_rc = make_tensor(
          tCrO.data(), ffpa_cute::convert_layout_acc_rowcol(tCrO.layout()));
#pragma unroll
      for (int row = 0; row < kORows; ++row) {
        if (row_scale[row] < 1.0f) {
#pragma unroll
          for (int col = 0; col < kOCols; ++col)
            tCrO_rc(row, col) *= row_scale[row];
        }
      }
    }

    // P -> e4m3 A operand (pre-scaled by the fixed-mode softmax; the
    // reorg-free perm pack pairs with the kVTPerm V^T from pre-kernel).
    auto tCrP =
        make_tensor(reinterpret_cast<Element*>(tCrSf.data()),
                    Layout<Shape<Shape<_4, _2, _2>, _1, Int<kBc / 32>>>{});
    if constexpr (kReorgFree) {
      quantize_p_frag_prescaled(tCrSf, perm_pack);
    } else {
      quantize_p_frag_prescaled(tCrSf, reorg);
    }

    // PV GEMM. Tensor-core row sum over the quantized P regs, then the fp8
    // PV MMA accumulates into o_acc (f16-acc inst_buf absorbs the rescale).
    cp_async_wait<kStages * 2 - 2>();
    __syncthreads();

    auto sV = make_tensor(make_smem_ptr(v_base + v_stg * kVTileElements),
                          SmemLayoutV{});
    auto tCrV = thr_mma_pv.partition_fragment_B(sV);
    auto tVsV_s2r = s2r_thr_v.partition_S(sV);

    pscale_rowsum_mma(tCrP, row_sum, 1.0f / p_quant_scale);
    if constexpr (kPVAccF16) {
      auto s2r_thr_pv_f16 = s2r_thr_v_f16;
      auto tCrV_f16 = thr_mma_pv_f16.partition_fragment_B(sV);
      auto tVsV_s2r_f16 = s2r_thr_pv_f16.partition_S(sV);
      auto tCrInst = partition_fragment_C(tiled_mma_pv_f16,
                                          Shape<Int<kBr>, Int<kHeadDim>>{});
      clear(tCrInst);
      ffpa_cute::gemm_rs(tCrInst, tCrP, tCrV_f16, tVsV_s2r_f16,
                         tiled_mma_pv_f16, s2r_copy_v_f16, s2r_thr_v_f16);
      auto tCrO = make_tensor(make_rmem_ptr(o_acc), OFragLayout{});
      auto tCrO_rc = make_tensor(
          tCrO.data(), ffpa_cute::convert_layout_acc_rowcol(tCrO.layout()));
      auto tCrInst_rc =
          make_tensor(tCrInst.data(),
                      ffpa_cute::convert_layout_acc_rowcol(tCrInst.layout()));
#pragma unroll
      for (int row = 0; row < kORows; ++row) {
        const float rs =
            (kv_tile > 0 && row_scale[row] < 1.0f) ? row_scale[row] : 1.0f;
#pragma unroll
        for (int col = 0; col < kOCols; ++col)
          tCrO_rc(row, col) =
              fmaf(tCrO_rc(row, col), rs, float(tCrInst_rc(row, col)));
      }
    } else {
      auto tCrO = make_tensor(make_rmem_ptr(o_acc), OFragLayout{});
      ffpa_cute::gemm_rs(tCrO, tCrP, tCrV, tVsV_s2r, tiled_mma_pv, s2r_copy_v,
                         s2r_thr_v);
    }

    // All threads finished reading K[t]/V[t] stages: safe to reissue the
    // stage slots for tile t+S (the loop-tail commits of the group FIFO).
    __syncthreads();
    {
      const int kv_next = kv_tile + kStages;
      if (kv_next < Tc_eff) {
        g2s_load_k(kv_next, k_stg);
        cp_async_fence();
        g2s_load_v(kv_next, v_stg);
        cp_async_fence();
      }
    }
  }
  cp_async_wait<0>();

  // Phase 4: Epilogue. Dequant o_acc (fixed mode keeps the single
  // 1/p_quant_scale domain), normalize, convert, store R->G.
  {
    const int O_gmem_offset =
        (Nb_id * Nh * Nq * kHeadDim) + (Nh_id * Nq * kHeadDim);
    auto mO = make_tensor(make_gmem_ptr(O + O_gmem_offset),
                          make_shape(Nq, Int<kHeadDim>{}),
                          make_stride(Int<kHeadDim>{}, _1{}));
    auto gO = local_tile(mO, Shape<Int<kBr>, Int<kHeadDim>>{},
                         make_coord(Q_tile_id, _0{}));
    auto tCgO = thr_mma_pv.partition_C(gO);
    auto cO = make_identity_tensor(Shape<Int<kBr>, Int<kHeadDim>>{});
    auto tOcO = thr_mma_pv.partition_C(cO);

    auto tCrO = make_tensor(make_rmem_ptr(o_acc), OFragLayout{});
    auto tCrO_rc = make_tensor(
        tCrO.data(), ffpa_cute::convert_layout_acc_rowcol(tCrO.layout()));
#pragma unroll
    for (int row = 0; row < kORows; ++row) {
      const float inv_sum = (row_sum[row] == 0.0f) ? 1.0f : 1.0f / row_sum[row];
      const float mul = inv_sum * kFP8FixedPScale;
#pragma unroll
      for (int col = 0; col < kOCols; ++col)
        tCrO_rc(row, col) *= mul;
    }
    auto tCrOHalf = ffpa_cute::convert_type<ElementO>(tCrO);
    if (Br_base + kBr <= Nq) {
      copy(tCrOHalf, tCgO);
    } else {
#pragma unroll
      for (int i = 0; i < size(tCrOHalf); ++i) {
        const int global_row = Br_base + get<0>(tOcO(i));
        if (global_row < Nq)
          tCgO(i) = tCrOHalf(i);
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

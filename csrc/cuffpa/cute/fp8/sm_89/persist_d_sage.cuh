#pragma once

// FP8 persist-D Flash Attention forward, sm_89 "Sage2 replica" accumulator.
//
// The SageAttention-2 two-level output accumulator, re-cut for the sm_89
// register file: the PV MMA lands in a per-tile f16 inst_buf whose only
// consumer is the float running accumulator,
//     RO = RO * row_scale + inst      (one fused fmaf per element)
// so no f16 register ever accumulates across KV tiles -- the f16 overflow
// domain that plagued the persistent-o16 dual kernel is structurally gone
// (same trade Sage2 makes; adversarial inputs can still saturate a single
// tile's inst_buf: worst case |inst| ~ 448 * sum(P_tile) * amax(V), so a
// fully flat 64-row tile with amax(V) > ~2.3 (448*64*2.3 > f16 max) or
// concentrated scores with larger amax can still inf -- the documented
// FA2-unnormalized-P contract, matching randn-domain safety).
//
// The literal Sage2 shape (128T, 128 f32 RO/thread) spills 304B of stack
// on this compute layer and runs 4x slower; at 256T/CTA each thread holds
// half a row pair (64 f32 RO + 32 f16 inst) inside the 255-reg budget,
// which is the v1 geometry. What this kernel adds over v1 is the
// dual-kernel loop-body discipline:
//   - per-thread gmem/smem chunk addresses derived once before the loop;
//     issuing tile t only adds constant strides (K: kBc*kHeadDim bytes,
//     V^T: kBc bytes) and a whole-tile stage shift that preserves the
//     SW128 swizzle pattern (v1 rebuilds the full cute layout algebra
//     every tile),
//   - K/V smem tensors and s2r partitions precomputed per stage,
//   - a mask-free main loop with the masked variant in the tail.
//
// The compute layer is the one already validated on sm_120/sm_89: int8 QK
// (m16n8k32 s8, Q resident in registers via gemm_rs), fixed p_scale
// softmax with the vs*448 quant scale folded into exp2, max-scale-after,
// rescale absorbed into the RO fmaf, reorg-free perm pack pairing the
// kVTPerm V^T, tensor-core P row-sum, prescaled e4m3x2 cvt. 3 CTA
// barriers per KV tile.
//
// v1 scope: per-block Q/K/V, int8 QK + f16 PV acc, no bias/dropout/
// hybrid/q_start_row/smooth_v, D=128 only. kBc=128 matches the 128-col
// quant blocks, so ks/vs index kv_tile directly.

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>

#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cutlass/cutlass.h>

#include "../../gemm.cuh"
#include "../attn_traits.cuh"
#include "../../softmax.cuh"
#include "../fp8_pscale.cuh"
#include "../reg2reg_8b.cuh"
#include "../smooth_k.cuh"

namespace ffpa_fp8 {

CUTE_DEVICE void cp_async_cg16_sage(uint32_t smem_addr, const void* gptr) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr),
               "l"(gptr));
}

// Generic pointer -> 32-bit shared window address (setup-only helper).
CUTE_DEVICE uint32_t smem_generic_to_u32_sage(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

template <typename Traits, typename ElementO>
__global__ void __launch_bounds__(Traits::kNumThreads, 1)
    persist_d_fwd_cute_fp8_sm89_sage(
        typename Traits::ElementQK* __restrict__ Q,
        typename Traits::ElementQK* __restrict__ K,
        typename Traits::Element* __restrict__ V,  // VT (D, Nkv_pad)
        ElementO* __restrict__ O, float* __restrict__ softmax_lse,
        const float* __restrict__ q_scale, const float* __restrict__ k_scale,
        const float* __restrict__ v_scale, int Nq, int Nkv, int Nh, int Nh_kv,
        int n_rb_q, int n_rb_kv, float scale, int Tc, int causal, int Nkv_pad,
        const float* __restrict__ km = nullptr,
        const float* __restrict__ vm = nullptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  using namespace cute;
  using Element = typename Traits::Element;      // float_e4m3_t (V / P)
  using ElementQK = typename Traits::ElementQK;  // int8
  using SmemLayoutQ = typename Traits::SmemLayoutQ;
  using SmemLayoutK = typename Traits::SmemLayoutK;
  using SmemLayoutV = typename Traits::SmemLayoutV;
  using TiledMmaQK = typename Traits::TiledMmaQK;

  constexpr int kBr = Traits::kBr;            // 128
  constexpr int kBc = Traits::kBc;            // 128
  constexpr int kHeadDim = Traits::kHeadDim;  // 128
  constexpr int kStages = Traits::kStagesK;
  static_assert(kHeadDim == 128 && kBr == 128 && kBc == 128,
                "sage kernel: D=128, kBr=128, kBc=128");
  static_assert(Traits::kStagesK == Traits::kStagesV, "");
  constexpr int kNumThreads = Traits::kNumThreads;  // 256

  // f16 PV atom (Ada-only m16n8k32 f16-acc), 8 warps over the 128 rows.
  using MmaAtomPVf16 = MMA_Atom<SM89_16x8x32_F16E4M3E4M3F16_TN>;
  using TiledMmaPVf16 = decltype(make_tiled_mma(
      MmaAtomPVf16{}, Layout<Shape<Int<kNumThreads / 32>, _1, _1>>{},
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

  // SMEM: [Q persist | K stages | V stages], 1B per elem.
  extern __shared__ __align__(1024) char shm[];
  ElementQK* q_base = reinterpret_cast<ElementQK*>(shm);
  ElementQK* k_base = q_base + kQTileElements;
  Element* v_base =
      reinterpret_cast<Element*>(k_base + kStages * kKTileElements);

  // G2S TiledCopy: 16B cp.async over [rows, 64] segments.
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

  // Per-thread g2s chunk addresses, derived once before the loop. 4 x 16B
  // chunks per thread for K and V each; issuing tile t adds K: kBc*kHeadDim
  // bytes / V^T: kBc bytes, and the whole-tile stage shift preserves the
  // swizzle pattern (this removes v1's per-tile 64-bit layout algebra).
  constexpr int kKChunks = 4;
  constexpr int kVChunks = 4;
  constexpr int kChunkElems = 16;
  constexpr int kKTileBytes = kBc * kHeadDim;
  uint32_t k_saddr0[kKChunks], v_saddr0[kVChunks];
  const char* k_gptr[kKChunks];
  const char* v_gptr[kVChunks];
  {
    auto sK0 = make_tensor(make_smem_ptr(k_base), SmemLayoutK{});
    auto sV0 = make_tensor(make_smem_ptr(v_base), SmemLayoutV{});
    int ci = 0;
    CUTLASS_PRAGMA_UNROLL
    for (int seg = 0; seg < kHeadDim / kSegCols; ++seg) {
      auto gSeg = local_tile(mK, Shape<Int<kBc>, Int<kSegCols>>{},
                             make_coord(_0{}, seg));
      auto sSeg = local_tile(sK0, Shape<Int<kBc>, Int<kSegCols>>{},
                             make_coord(_0{}, seg));
      auto gp = g2s_thr.partition_S(gSeg);
      auto sp = g2s_thr.partition_D(sSeg);
      static_assert(decltype(size(sp))::value == 2 * kChunkElems, "");
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < 2; ++i, ++ci) {
        k_gptr[ci] = reinterpret_cast<const char*>(&gp(i * kChunkElems));
        k_saddr0[ci] = smem_generic_to_u32_sage(&sp(i * kChunkElems));
      }
    }
    ci = 0;
    CUTLASS_PRAGMA_UNROLL
    for (int seg = 0; seg < kBc / kSegCols; ++seg) {
      auto gSeg = local_tile(mV, Shape<Int<kHeadDim>, Int<kSegCols>>{},
                             make_coord(_0{}, seg));
      auto sSeg = local_tile(sV0, Shape<Int<kHeadDim>, Int<kSegCols>>{},
                             make_coord(_0{}, seg));
      auto gp = g2s_thr.partition_S(gSeg);
      auto sp = g2s_thr.partition_D(sSeg);
      static_assert(decltype(size(sp))::value == 2 * kChunkElems, "");
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < 2; ++i, ++ci) {
        v_gptr[ci] = reinterpret_cast<const char*>(&gp(i * kChunkElems));
        v_saddr0[ci] = smem_generic_to_u32_sage(&sp(i * kChunkElems));
      }
    }
  }
  auto issue_k = [&](int t, int stage) {
    const uint32_t sbase = stage * static_cast<uint32_t>(kKTileElements);
    const int64_t goff = static_cast<int64_t>(t) * kKTileBytes;
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < kKChunks; ++c)
      cp_async_cg16_sage(k_saddr0[c] + sbase, k_gptr[c] + goff);
  };
  auto issue_v = [&](int t, int stage) {
    const uint32_t sbase = stage * static_cast<uint32_t>(kVTileElements);
    const int64_t goff = static_cast<int64_t>(t) * kBc;
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < kVChunks; ++c)
      cp_async_cg16_sage(v_saddr0[c] + sbase, v_gptr[c] + goff);
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

  // s32 QK score fragment: 64 elems/thread = 2 rows x 32 cols (kBc=128).
  using ScoreFrag =
      decltype(partition_fragment_C(tiled_mma_qk, Shape<Int<kBr>, Int<kBc>>{}));
  using ScoreRCLayout =
      decltype(ffpa_cute::convert_layout_acc_rowcol(ScoreFrag{}.layout()));
  constexpr int kSCols = decltype(cute::size<1>(
      make_tensor((float*)nullptr, ScoreRCLayout{})))::value;
  constexpr int kSRows = decltype(cute::size<0>(
      make_tensor((float*)nullptr, ScoreRCLayout{})))::value;
  static_assert(kSCols == 32 && kSRows == 2, "");

  // f16 PV inst_buf fragment: 64 elems/thread = 2 rows x 32 cols. The
  // float running accumulator (64 f32/thread) keeps the same layout.
  using OFrag16 = decltype(partition_fragment_C(
      tiled_mma_pv_f16, Shape<Int<kBr>, Int<kHeadDim>>{}));
  using ORC16Layout =
      decltype(ffpa_cute::convert_layout_acc_rowcol(OFrag16{}.layout()));
  constexpr int kOCols = decltype(cute::size<1>(
      make_tensor((__half*)nullptr, ORC16Layout{})))::value;
  constexpr int kOElems = decltype(cute::size(OFrag16{}))::value;
  static_assert(kOCols == 32 && kOElems == 64, "");

  auto sQ = make_tensor(make_smem_ptr(q_base), SmemLayoutQ{});
  // Per-stage K/V smem tensors and s2r partitions, precomputed once (v1
  // rebuilds them from layout algebra inside every tile).
  auto sK_s0 = make_tensor(make_smem_ptr(k_base), SmemLayoutK{});
  auto sK_s1 =
      make_tensor(make_smem_ptr(k_base + kKTileElements), SmemLayoutK{});
  auto sV_s0 = make_tensor(make_smem_ptr(v_base), SmemLayoutV{});
  auto sV_s1 =
      make_tensor(make_smem_ptr(v_base + kVTileElements), SmemLayoutV{});
  auto tKsK0 = s2r_thr_k.partition_S(sK_s0);
  auto tKsK1 = s2r_thr_k.partition_S(sK_s1);
  auto tVsV0 = s2r_thr_v.partition_S(sV_s0);
  auto tVsV1 = s2r_thr_v.partition_S(sV_s1);

  const float scale_orig = scale;
  scale *= FFPA_M_LOG2E;

  const float qs = q_scale[static_cast<long>(q_bh) * n_rb_q + Q_tile_id];

  float row_max[kSRows];
  float row_sum[kSRows];
  float qkm[kSRows];
  float row_scale[kSRows];
#pragma unroll
  for (int r = 0; r < kSRows; ++r) {
    row_max[r] = -INFINITY;
    row_sum[r] = 0.0f;
    qkm[r] = 0.0f;
    row_scale[r] = 1.0f;
  }

  // Two-level accumulator (the Sage2 structure): per-tile f16 inst_buf
  // above (inside process_tile), cross-tile float running accumulator
  // here. inst never outlives its tile; RO lives in the fixed 448*(P@V)
  // domain and cannot overflow for any realistic sequence length.
  float ro[kOElems];
#pragma unroll
  for (int i = 0; i < kOElems; ++i)
    ro[i] = 0.0f;

  // Coordinate tensor for softmax indexing.
  auto cS = make_identity_tensor(Shape<Int<kBr>, Int<kBc>>{});
  auto tScS = thr_mma_qk.partition_C(cS);
  auto tScS_rc = make_tensor(
      tScS.data(), ffpa_cute::convert_layout_acc_rowcol(tScS.layout()));

  // Prologue: Q group, then K/V[0..S-1] pair groups (FIFO invariant:
  // K[t] = group 1+2t, V[t] = group 2+2t).
  {
    g2s_load_q();
    cp_async_fence();
#pragma unroll
    for (int s = 0; s < kStages; ++s) {
      if (s < Tc_eff) {
        issue_k(s, s);
        cp_async_fence();
        issue_v(s, s);
        cp_async_fence();
      }
    }
    if (Tc_eff >= kStages)
      cp_async_wait<kStages * 2 - 1>();
    else
      cp_async_wait<0>();
    __syncthreads();
  }

  // Smooth-K dot correction (lse += scale_orig * qs * qkm[row]); reads the
  // smem Q tile before the A-fragment is moved to regs.
  const bool smooth_lse = (softmax_lse != nullptr) && (km != nullptr);
  if (smooth_lse)
    smooth_k_qk_dot<kHeadDim, kSRows>(
        sQ, tScS_rc, km + static_cast<long>(kv_bh) * kHeadDim, qkm);

  // Q s2r once: the A fragment is loop-invariant (persist-D), so every QK
  // step below runs as gemm_rs (K-only smem loads).
  auto tCrQ = thr_mma_qk.partition_fragment_A(sQ);
  auto tQsQ_s2r = s2r_thr_q.partition_S(sQ);
  {
    auto tXrQ = s2r_thr_q.retile_D(tCrQ);
#pragma unroll
    for (int tile_k = 0; tile_k < size<2>(tCrQ); ++tile_k)
      copy(s2r_copy_q, tQsQ_s2r(_, _, tile_k), tXrQ(_, _, tile_k));
  }

  PackC8bitToA8bitPermVT perm_pack;
  using PLayer = Layout<Shape<Shape<_4, _2, _2>, _1, Int<kBc / 32>>>;

  // One KV tile. `masked == false` is a compile-time constant at the
  // main-loop call site, so the mask code only exists in the tail copy.
  auto process_tile = [&](int kv_tile, bool masked) {
    const int stg = kv_tile % kStages;
    // kBc == the 128-col quant block, one scale per tile.
    const float ks = k_scale[static_cast<long>(kv_bh) * n_rb_kv + kv_tile];
    const float vs = v_scale[static_cast<long>(kv_bh) * n_rb_kv + kv_tile];
    const float p_quant_scale = vs * kE4m3Max;

    // ---- QK: K-only smem loads feed the resident Q A-fragment ----
    cp_async_wait<kStages * 2 - 1>();
    __syncthreads();

    auto& tKsK = stg ? tKsK1 : tKsK0;
    auto tCrK = thr_mma_qk.partition_fragment_B(sK_s0);
    ScoreFrag tCrS;
    clear(tCrS);
    ffpa_cute::gemm_rs(tCrS, tCrQ, tCrK, tKsK, tiled_mma_qk, s2r_copy_k,
                       s2r_thr_k);

    // int8 QK: cast the s32 acc to f32 in place.
    auto tCrSf =
        make_tensor(reinterpret_cast<float*>(tCrS.data()), tCrS.layout());
#pragma unroll
    for (int i = 0; i < size(tCrS); ++i)
      tCrSf(i) = static_cast<float>(tCrS(i));
    auto scores = make_tensor(
        tCrSf.data(), ffpa_cute::convert_layout_acc_rowcol(tCrS.layout()));

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
    online_softmax_fp8_fixed<true, decltype(scores), decltype(tScS_rc), kSRows,
                             true>(
        scores, tScS_rc, tile_needs_mask ? 1.0f : qs * ks * scale, row_max,
        row_sum, row_scale, log2f(p_quant_scale), 1.0f / p_quant_scale,
        Traits::kRescaleThreshold);

    // f32 score storage -> packed e4m3 PV A operand (perm pack binds the
    // kVTPerm V^T from the quantize pre-kernel).
    auto tCrP = make_tensor(reinterpret_cast<Element*>(tCrSf.data()), PLayer{});
    quantize_p_frag_prescaled(tCrSf, perm_pack);

    // ---- V settle, then PV: per-tile f16 inst_buf -> RO absorb ----
    cp_async_wait<kStages * 2 - 2>();
    __syncthreads();

    auto& tVsV = stg ? tVsV1 : tVsV0;
    auto tCrV = thr_mma_pv_f16.partition_fragment_B(sV_s0);

    pscale_rowsum_mma(tCrP, row_sum, 1.0f / p_quant_scale);

    OFrag16 inst;
    clear(inst);
    ffpa_cute::gemm_rs(inst, tCrP, tCrV, tVsV, tiled_mma_pv_f16, s2r_copy_v,
                       s2r_thr_v);

    // Sage2-style absorption: RO = RO*rs + inst, one fused fmaf per
    // element; rs is skipped on the first tile (RO is zero anyway).
    {
      auto ro_rc = make_tensor(make_rmem_ptr(ro), ORC16Layout{});
      auto inst_rc = make_tensor(inst.data(), ORC16Layout{});
#pragma unroll
      for (int row = 0; row < kSRows; ++row) {
        const float rs =
            (kv_tile > 0 && row_scale[row] < 1.0f) ? row_scale[row] : 1.0f;
#pragma unroll
        for (int col = 0; col < kOCols; ++col)
          ro_rc(row, col) = fmaf(ro_rc(row, col), rs, float(inst_rc(row, col)));
      }
    }

    // ---- release the V stage and issue tile t+kStages ----
    // Drain tiles commit two empty groups so the depth-limited waits keep
    // settling exactly.
    __syncthreads();
    const int kv_next = kv_tile + kStages;
    if (kv_next < Tc_eff) {
      issue_k(kv_next, stg);
      cp_async_fence();
      issue_v(kv_next, stg);
      cp_async_fence();
    } else {
      cp_async_fence();
      cp_async_fence();
    }
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
    for (int row = 0; row < kSRows; ++row) {
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
      for (int row = 0; row < kSRows; ++row) {
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

#pragma once

// FP8 persist-D Flash Attention forward, sm_89 dual sub-tile geometry.
//
// One 128-thread CTA owns 128 Q rows but computes them as TWO independent
// 64-row sub-tiles (4 warps, MMA_M=1 each) sharing a single K/V smem stage:
// the QK/PV k-loops issue one K/V LDSM that feeds two MMAs, so doubling the
// number of serial dependency chains (to fill math-pipe stalls) costs no
// extra K/V traffic or per-tile barriers. Q (128 rows) stays in smem and
// both QK GEMMs reload their A-fragments from it each tile. 48KB smem
// (Q 16K + 2x(K 8K + V 8K)) and __launch_bounds__(128, 2) target two
// resident CTAs per SM.
//
// Loop-body discipline (the single-tile kernel lost ~25% of its issue slots
// to per-tile cute layout algebra and cold-path bloat, so this kernel):
//   - the per-thread LDGSTS chunk addresses (gmem pointers + swizzled smem
//     u32) are derived once before the loop; issuing tile t only adds
//     constant strides (K: kBc*kHeadDim bytes, V^T: kBc bytes) and a whole
//     -tile stage shift that preserves the SW128 swizzle pattern,
//   - K/V s2r partitions are precomputed per stage (kStages == 2),
//   - the masked softmax variant lives in a separate tail loop, the main
//     loop body is mask-free,
//   - the overflow-guard decision is prepared before the V wait and routed
//     with a single __syncthreads_count that doubles as the V-settle
//     barrier (3 CTA barriers per tile, same as SageAttention),
//   - the rare guard path scales P in the float domain (branch-free) and
//     stays out of the fast-path instruction stream.
//
// O is a persistent f16 fragment (f32 does not fit 128 rows at 128T). An
// overflow guard keeps it in range: on the rare saturated/adversarial
// tile the sub-tile enters a persistent g=4^-nq domain shared by o16 and
// row_sum (ratio = attention output stays exact), compensated in lse.
//
// v1 scope (same as persist_d.cuh): per-block Q/K/V, int8 QK + f16 PV acc,
// no bias/dropout/hybrid/q_start_row/smooth_v, D=128 only.
// K/V scales are produced on 128-row quant blocks while this kernel tiles
// KV by 64, so ks/vs index kv_tile/2 (the two 64 halves share one block).

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

// Row sums of the quantized P fragment as RAW e4m3 code sums (d0/d1 are
// the two rows this thread owns). Same all-ones-B mma trick as
// pscale_rowsum_mma, which folds in inv_exp_factor; the caller also needs
// the raw d to bound the f16 O domain (|inst| <= 448*d, V8 code <= 448).
template <typename P8Tensor>
CUTE_DEVICE void pscale_rowsum_raw(const P8Tensor& p8, float& d0, float& d1) {
  constexpr int kKSteps = decltype(cute::size<2>(p8))::value;
  d0 = d1 = 0.0f;
  const uint32_t* p = reinterpret_cast<const uint32_t*>(p8.data());
#pragma unroll
  for (int k = 0; k < kKSteps; ++k) {
    const uint32_t* s = p + k * 4;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, _, %1, _}, {%2, %3, %4, %5}, {%6, %7}, {%0, 0., %1, 0.};\n"
        : "+f"(d0), "+f"(d1)
        : "r"(s[0]), "r"(s[1]), "r"(s[2]), "r"(s[3]), "r"(0x38383838u),
          "r"(0x38383838u));
  }
}

// Scale a packed e4m3 P fragment by an exact power of two in the float
// domain (rare overflow-guard path; branch-free and code-size friendly).
template <int kNElem>
CUTE_DEVICE void p8_scale_pow2(cutlass::float_e4m3_t* p, float s) {
#pragma unroll
  for (int i = 0; i < kNElem; ++i) {
    const float x = static_cast<float>(p[i]);
    p[i] = cutlass::float_e4m3_t(x * s);
  }
}

CUTE_DEVICE void cp_async_cg16(uint32_t smem_addr, const void* gptr) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr),
               "l"(gptr));
}

// Generic pointer -> 32-bit shared window address (setup-only helper).
CUTE_DEVICE uint32_t smem_generic_to_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

template <typename Traits, typename ElementO>
__global__ void __launch_bounds__(Traits::kNumThreads, 2)
    persist_d_fwd_cute_fp8_sm89_dual(
        typename Traits::ElementQK* __restrict__ Q,
        typename Traits::ElementQK* __restrict__ K,
        typename Traits::Element* __restrict__ V,  // VT (D, Nkv_pad)
        ElementO* __restrict__ O, float* __restrict__ softmax_lse,
        const float* __restrict__ q_scale, const float* __restrict__ k_scale,
        const float* __restrict__ v_scale, int Nq, int Nkv, int Nh, int Nh_kv,
        int n_rb_q, int n_rb_kv, float scale, int Tc, int causal, int Nkv_pad,
        int obound_on, const float* __restrict__ km = nullptr,
        const float* __restrict__ vm = nullptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  using namespace cute;
  using Element = typename Traits::Element;      // float_e4m3_t (V / P)
  using ElementQK = typename Traits::ElementQK;  // int8
  using SmemLayoutK = typename Traits::SmemLayoutK;
  using SmemLayoutV = typename Traits::SmemLayoutV;
  using TiledMmaQK = typename Traits::TiledMmaQK;
  using TiledMmaPV = typename Traits::TiledMmaPV;

  constexpr int kBr = 128;
  constexpr int kBrSub = 64;
  constexpr int kBc = Traits::kBc;            // 64
  constexpr int kHeadDim = Traits::kHeadDim;  // 128
  constexpr int kStages = Traits::kStagesK;
  static_assert(kHeadDim == 128 && kBc == 64,
                "dual kernel: D=128, kBc=64 only");
  static_assert(Traits::kStagesK == Traits::kStagesV, "");
  constexpr int kNumThreads = Traits::kNumThreads;  // 128
  static_assert(kNumThreads == 128, "");
  constexpr int kNumSub = kBr / kBrSub;  // 2

  // f16 PV atom (Ada-only m16n8k32 f16-acc), 4 warps over a 64-row sub.
  using MmaAtomPVf16 = MMA_Atom<SM89_16x8x32_F16E4M3E4M3F16_TN>;
  using TiledMmaPVf16 =
      decltype(make_tiled_mma(MmaAtomPVf16{}, Layout<Shape<_4, _1, _1>>{},
                              Tile<Int<kBrSub>, Int<kHeadDim>, _32>{}));

  constexpr int kQTileElements = kBr * kHeadDim;
  constexpr int kKTileElements = cosize(SmemLayoutK{});
  constexpr int kVTileElements = cosize(SmemLayoutV{});

  using SmemLayoutQ =
      decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<ElementQK>{},
                             Shape<Int<kBr>, Int<kHeadDim>>{}));

  // f16-O overflow-guard domain gates (kOAbsMax = f16 max with headroom,
  // kSafeMax keeps two f16 summands inside range).
  constexpr float kOAbsMax = 49152.0f;
  constexpr float kSafeMax = 24576.0f;

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
  // Grid-level tile count = union of both sub-tiles (sub B's last row).
  const int last_row = Br_base + kBr - 1;
  const int Tc_eff = causal ? min(Tc, ((last_row + kv_offset) / kBc) + 1) : Tc;
  // First per-element-masked tile = the sub's FIRST row diagonal tile
  // floor((R+kv_offset)/kBc), R = Br_base + s*64 (conservative +1 form).
  const int mask_start_0 =
      causal ? max(0, (Br_base + kv_offset + 1) / kBc) : INT_MAX;
  const int mask_start_1 =
      causal ? max(0, (Br_base + kBrSub + kv_offset + 1) / kBc) : INT_MAX;
  const int mask_start[kNumSub] = {mask_start_0, mask_start_1};

  const int q_row_offset = (Nb_id * Nh + Nh_id) * Nq;
  const int kv_row_offset = (Nb_id * Nh_kv + kv_head_idx) * Nkv;
  const int q_bh = Nb_id * Nh + Nh_id;
  const int kv_bh = Nb_id * Nh_kv + kv_head_idx;

  // SMEM: [Q 128x128 | K stages | V stages], 1B per elem.
  extern __shared__ __align__(1024) char shm[];
  ElementQK* q_base = reinterpret_cast<ElementQK*>(shm);
  ElementQK* k_base = q_base + kQTileElements;
  Element* v_base =
      reinterpret_cast<Element*>(k_base + kStages * kKTileElements);
  // 8-float CTA reduce scratch past the V stages (rare overflow path).
  float* red_base = reinterpret_cast<float*>(v_base + kStages * kVTileElements);

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
  // swizzle pattern (this removes the per-tile 64-bit layout algebra).
  // The partition tensors hold 16 single-byte elements per 16B chunk, so
  // chunk starts sit at linear indices i*kChunkElems.
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
        k_saddr0[ci] = smem_generic_to_u32(&sp(i * kChunkElems));
      }
    }
    auto gSeg = local_tile(mV, Shape<Int<kHeadDim>, Int<kBc>>{},
                           make_coord(_0{}, _0{}));
    auto sSeg = local_tile(sV0, Shape<Int<kHeadDim>, Int<kBc>>{},
                           make_coord(_0{}, _0{}));
    auto gp = g2s_thr.partition_S(gSeg);
    auto sp = g2s_thr.partition_D(sSeg);
    static_assert(decltype(size(sp))::value == kVChunks * kChunkElems, "");
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kVChunks; ++i) {
      v_gptr[i] = reinterpret_cast<const char*>(&gp(i * kChunkElems));
      v_saddr0[i] = smem_generic_to_u32(&sp(i * kChunkElems));
    }
  }
  auto issue_k = [&](int t, int stage) {
    const uint32_t sbase = stage * static_cast<uint32_t>(kKTileElements);
    const int64_t goff = static_cast<int64_t>(t) * kKTileBytes;
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < kKChunks; ++c)
      cp_async_cg16(k_saddr0[c] + sbase, k_gptr[c] + goff);
  };
  auto issue_v = [&](int t, int stage) {
    const uint32_t sbase = stage * static_cast<uint32_t>(kVTileElements);
    const int64_t goff = static_cast<int64_t>(t) * kBc;
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < kVChunks; ++c)
      cp_async_cg16(v_saddr0[c] + sbase, v_gptr[c] + goff);
  };

  TiledMmaQK tiled_mma_qk;
  TiledMmaPV tiled_mma_pv;
  TiledMmaPVf16 tiled_mma_pv_f16;
  auto thr_mma_qk = tiled_mma_qk.get_thread_slice(tid);
  auto thr_mma_pv = tiled_mma_pv.get_thread_slice(tid);
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

  // s32 QK score fragment (one sub-tile): 32 elems/thread = 2 rows x 16.
  using ScoreFrag = decltype(partition_fragment_C(
      tiled_mma_qk, Shape<Int<kBrSub>, Int<kBc>>{}));
  using ScoreRCLayout =
      decltype(ffpa_cute::convert_layout_acc_rowcol(ScoreFrag{}.layout()));
  constexpr int kSCols = decltype(cute::size<1>(
      make_tensor((float*)nullptr, ScoreRCLayout{})))::value;
  static_assert(kSCols == 16, "");

  using OFrag16 = decltype(partition_fragment_C(
      tiled_mma_pv_f16, Shape<Int<kBrSub>, Int<kHeadDim>>{}));
  using ORC16Layout =
      decltype(ffpa_cute::convert_layout_acc_rowcol(OFrag16{}.layout()));
  constexpr int kO16Cols = decltype(cute::size<1>(
      make_tensor((__half*)nullptr, ORC16Layout{})))::value;
  static_assert(kO16Cols == 32, "");

  auto sQ_full = make_tensor(make_smem_ptr(q_base), SmemLayoutQ{});
  auto sQ0 = local_tile(sQ_full, Shape<Int<kBrSub>, Int<kHeadDim>>{},
                        make_coord(_0{}, _0{}));
  auto sQ1 = local_tile(sQ_full, Shape<Int<kBrSub>, Int<kHeadDim>>{},
                        make_coord(_1{}, _0{}));
  auto tCrQ0 = thr_mma_qk.partition_fragment_A(sQ0);
  auto tCrQ1 = thr_mma_qk.partition_fragment_A(sQ1);
  auto tQsQ0 = s2r_thr_q.partition_S(sQ0);
  auto tQsQ1 = s2r_thr_q.partition_S(sQ1);

  // Per-stage K/V smem tensors and s2r partitions, precomputed once (the
  // per-tile make_tensor/partition reconstruction was pure loop overhead).
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

  // One qs per 128-row quant block: both sub-tiles share it.
  const float qs = q_scale[static_cast<long>(q_bh) * n_rb_q + Q_tile_id];

  float row_max[kNumSub][2];
  float row_sum[kNumSub][2];
  float qkm[kNumSub][2];
  float row_scale[kNumSub][2];
  // Persistent f16-O overflow-guard domain: once a sub-tile ever enters the
  // g=4^-o_sg scaled domain, every later tile is merged at that scale.
  // o_loose is a cancellation-free worst-case bound (updated with 2 scalar
  // FMAs/tile); only when it crosses the gate does the kernel pay for a
  // warp-wide actual-max scan, after which the tight value resets it.
  int o_sg[kNumSub];
  float o_loose[kNumSub][2];
  bool guarded[kNumSub];
#pragma unroll
  for (int s = 0; s < kNumSub; ++s) {
    guarded[s] = false;
    o_sg[s] = 0;
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      row_max[s][r] = -INFINITY;
      row_sum[s][r] = 0.0f;
      qkm[s][r] = 0.0f;
      row_scale[s][r] = 1.0f;
      o_loose[s][r] = 0.0f;
    }
  }

  OFrag16 o16[kNumSub];
#pragma unroll
  for (int s = 0; s < kNumSub; ++s)
    clear(o16[s]);

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

  const bool smooth_lse = (softmax_lse != nullptr) && (km != nullptr);
  if (smooth_lse) {
    auto cS0 = make_identity_tensor(Shape<Int<kBrSub>, Int<kBc>>{});
    auto rc0 = make_tensor(thr_mma_qk.partition_C(cS0).data(),
                           ffpa_cute::convert_layout_acc_rowcol(
                               thr_mma_qk.partition_C(cS0).layout()));
    smooth_k_qk_dot<kHeadDim, 2>(
        sQ0, rc0, km + static_cast<long>(kv_bh) * kHeadDim, qkm[0]);
    auto cS1 = make_identity_tensor(Shape<Int<kBrSub>, Int<kBc>>{});
    auto rc1 = make_tensor(thr_mma_qk.partition_C(cS1).data(),
                           ffpa_cute::convert_layout_acc_rowcol(
                               thr_mma_qk.partition_C(cS1).layout()));
    smooth_k_qk_dot<kHeadDim, 2>(
        sQ1, rc1, km + static_cast<long>(kv_bh) * kHeadDim, qkm[1]);
  }

  PackC8bitToA8bitPermVT perm_pack[2];
  using PLayer = Layout<Shape<Shape<_4, _2, _2>, _1, Int<kBc / 32>>>;
  constexpr int kQKKSteps = kHeadDim / 32;
  constexpr int kPVKSteps = kBc / 32;

  // One KV tile: K/V scales are 128-row blocks; the two 64 halves share
  // one. `masked == false` is a compile-time constant at the main-loop call
  // site, so the mask code only exists in the tail loop's copy.
  auto process_tile = [&](int kv_tile, bool masked) {
    const int stg = kv_tile % kStages;
    const int kv_block128 = kv_tile / 2;
    const float ks = k_scale[static_cast<long>(kv_bh) * n_rb_kv + kv_block128];
    const float vs = v_scale[static_cast<long>(kv_bh) * n_rb_kv + kv_block128];
    const float p_quant_scale = vs * kE4m3Max;

    // ---- QK: one shared K B-fragment feeds both sub-tiles ----
    cp_async_wait<kStages * 2 - 1>();
    __syncthreads();

    auto& tKsK = stg ? tKsK1 : tKsK0;
    auto tCrK = thr_mma_qk.partition_fragment_B(sK_s0);
    ScoreFrag tCrS[2];
    clear(tCrS[0]);
    clear(tCrS[1]);

    {
      auto kb = s2r_thr_k.retile_D(tCrK);
      auto qa0 = s2r_thr_q.retile_D(tCrQ0);
      auto qa1 = s2r_thr_q.retile_D(tCrQ1);
      copy(s2r_copy_k, tKsK(_, _, _0{}), kb(_, _, _0{}));
      copy(s2r_copy_q, tQsQ0(_, _, _0{}), qa0(_, _, _0{}));
      copy(s2r_copy_q, tQsQ1(_, _, _0{}), qa1(_, _, _0{}));
#pragma unroll
      for (int kk = 0; kk < kQKKSteps; ++kk) {
        if (kk + 1 < kQKKSteps) {
          copy(s2r_copy_k, tKsK(_, _, kk + 1), kb(_, _, kk + 1));
          copy(s2r_copy_q, tQsQ0(_, _, kk + 1), qa0(_, _, kk + 1));
          copy(s2r_copy_q, tQsQ1(_, _, kk + 1), qa1(_, _, kk + 1));
        }
        gemm(tiled_mma_qk, tCrQ0(_, _, kk), tCrK(_, _, kk), tCrS[0]);
        gemm(tiled_mma_qk, tCrQ1(_, _, kk), tCrK(_, _, kk), tCrS[1]);
      }
    }

    const int kv_valid = Nkv - kv_tile * kBc;

    // s32 -> f32, mask (tail loop only), log2 softmax, P quant per
    // sub-tile. rmem tensors must be born in place (no post-construction
    // assignment), hence locals.
#pragma unroll
    for (int s = 0; s < kNumSub; ++s) {
      auto sf = make_tensor(reinterpret_cast<float*>(tCrS[s].data()),
                            tCrS[s].layout());
#pragma unroll
      for (int i = 0; i < size(sf); ++i)
        sf(i) = static_cast<float>(tCrS[s](i));
      auto scores = make_tensor(
          sf.data(), ffpa_cute::convert_layout_acc_rowcol(sf.layout()));

      auto cS_s = make_identity_tensor(Shape<Int<kBrSub>, Int<kBc>>{});
      auto tScS = thr_mma_qk.partition_C(cS_s);
      auto tScS_rc = make_tensor(
          tScS.data(), ffpa_cute::convert_layout_acc_rowcol(tScS.layout()));

      bool tile_needs_mask = false;
      if (masked) {
        tile_needs_mask = (kv_valid < kBc) || (kv_tile >= mask_start[s]);
        if (tile_needs_mask) {
#pragma unroll
          for (int row = 0; row < 2; ++row) {
            const int q_pos =
                Br_base + s * kBrSub + get<0>(tScS_rc(row, 0)) + kv_offset;
#pragma unroll
            for (int col = 0; col < kSCols; ++col) {
              float sv = scores(row, col) * qs * ks * scale;
              if (get<1>(tScS_rc(row, col)) >= kv_valid)
                sv = -INFINITY;
              if (kv_tile >= mask_start[s]) {
                const int k_pos = kv_tile * kBc + get<1>(tScS_rc(row, col));
                if (k_pos > q_pos)
                  sv = -INFINITY;
              }
              scores(row, col) = sv;
            }
          }
        }
      }

      online_softmax_fp8_fixed<true, decltype(scores), decltype(tScS_rc), 2,
                               /*kMaxScaleAfter=*/true>(
          scores, tScS_rc, tile_needs_mask ? 1.0f : qs * ks * scale, row_max[s],
          row_sum[s], row_scale[s], log2f(p_quant_scale), 1.0f / p_quant_scale,
          Traits::kRescaleThreshold);

      // f32 score storage -> packed e4m3 PV A operand (same fragment,
      // perm pack binds the kVTPerm V^T from the quantize pre-kernel).
      quantize_p_frag_prescaled(sf, perm_pack[s]);
    }

    // ---- guard prep (fills the V-wait latency) ----
    // dd feeds row_sum on every path; the loose bound / npre only gate the
    // f16-O overflow domain (see the rare path below).
    float dd[kNumSub][2];
    float loose[kNumSub][2];
    int npre[kNumSub];  // single saturated tile: P8 /4 steps before MMA
    bool rare_lane = false;
    {
      auto tCrP0 =
          make_tensor(reinterpret_cast<Element*>(tCrS[0].data()), PLayer{});
      auto tCrP1 =
          make_tensor(reinterpret_cast<Element*>(tCrS[1].data()), PLayer{});
      pscale_rowsum_raw(tCrP0, dd[0][0], dd[0][1]);
      pscale_rowsum_raw(tCrP1, dd[1][0], dd[1][1]);
#pragma unroll
      for (int s = 0; s < kNumSub; ++s) {
        npre[s] = 0;
        if (obound_on) {
          if (guarded[s]) {
            // Persistent-domain tile always merges via the scratch path.
            rare_lane = true;
          } else {
            const float bmax = fmaxf(dd[s][0], dd[s][1]);
            int n = 0;
            float b = kE4m3Max * bmax;
            while (b > kOAbsMax && n < 8) {
              b *= 0.25f;
              ++n;
            }
            npre[s] = n;
            if (n > 0)
              rare_lane = true;  // saturated tile must MMA via scratch
#pragma unroll
            for (int row = 0; row < 2; ++row) {
              const float rs = (kv_tile > 0) ? row_scale[s][row] : 1.0f;
              loose[s][row] = rs * o_loose[s][row] + kE4m3Max * bmax;
              if (loose[s][row] > kOAbsMax)
                rare_lane = true;
            }
          }
        }
      }
    }

    // ---- V settle + guard routing in ONE barrier ----
    cp_async_wait<kStages * 2 - 2>();
    const bool rare = __syncthreads_count(rare_lane ? 1 : 0) > 0;

    auto& tVsV = stg ? tVsV1 : tVsV0;
    auto tCrV = thr_mma_pv_f16.partition_fragment_B(sV_s0);
    auto tCrP0 =
        make_tensor(reinterpret_cast<Element*>(tCrS[0].data()), PLayer{});
    auto tCrP1 =
        make_tensor(reinterpret_cast<Element*>(tCrS[1].data()), PLayer{});

    if (!rare) {
      // Fast path (also used when the guard is disabled): direct PV MMA
      // into o16, numerically identical to an unguarded accumulator.
#pragma unroll
      for (int s = 0; s < kNumSub; ++s) {
        row_sum[s][0] += dd[s][0] / p_quant_scale;
        row_sum[s][1] += dd[s][1] / p_quant_scale;
        if (obound_on) {
          o_loose[s][0] = loose[s][0];
          o_loose[s][1] = loose[s][1];
        }
        auto orc =
            make_tensor(o16[s].data(),
                        ffpa_cute::convert_layout_acc_rowcol(o16[s].layout()));
#pragma unroll
        for (int row = 0; row < 2; ++row) {
          const float rs = (kv_tile > 0) ? row_scale[s][row] : 1.0f;
          if (rs < 1.0f) {
            const __half2 f2 = __floats2half2_rn(rs, rs);
#pragma unroll
            for (int col = 0; col < kO16Cols; col += 2) {
              __half2& o2 = *reinterpret_cast<__half2*>(&orc(row, col));
              o2 = __hmul2(o2, f2);
            }
          }
        }
      }
      auto vb = s2r_thr_v.retile_D(tCrV);
      copy(s2r_copy_v, tVsV(_, _, _0{}), vb(_, _, _0{}));
#pragma unroll
      for (int kk = 0; kk < kPVKSteps; ++kk) {
        if (kk + 1 < kPVKSteps)
          copy(s2r_copy_v, tVsV(_, _, kk + 1), vb(_, _, kk + 1));
        gemm(tiled_mma_pv_f16, tCrP0(_, _, kk), tCrV(_, _, kk), o16[0]);
        gemm(tiled_mma_pv_f16, tCrP1(_, _, kk), tCrV(_, _, kk), o16[1]);
      }
    } else {
      // Rare path (saturated/adversarial tiles). The P operand is scaled
      // ONLY for the single saturated tile whose internal k32 MMA sum
      // could exceed f16 (exact power of two in the float domain); normal
      // aligned tiles (e.g. constant V) keep P untouched because their
      // e4m3 codes are 0/1 and /4 would round them away. Each sub-tile
      // MMAs into a branch-local scratch fragment, and the persistent
      // g=4^-o_sg domain is applied to the f16 MMA result (f16 keeps full
      // relative precision on small values). nq decisions use the CTA-wide
      // actual max (a row's 128 columns are spread across the sub-tile's
      // 4 warps).
#pragma unroll
      for (int s = 0; s < kNumSub; ++s) {
        if (!guarded[s] && npre[s] > 0) {
          auto& tCrP = (s == 0) ? tCrP0 : tCrP1;
          p8_scale_pow2<32>(tCrP.data(), ldexpf(1.0f, -2 * npre[s]));
        }
      }
      OFrag16 inst[2];
      clear(inst[0]);
      clear(inst[1]);
      auto vb = s2r_thr_v.retile_D(tCrV);
      copy(s2r_copy_v, tVsV(_, _, _0{}), vb(_, _, _0{}));
#pragma unroll
      for (int kk = 0; kk < kPVKSteps; ++kk) {
        if (kk + 1 < kPVKSteps)
          copy(s2r_copy_v, tVsV(_, _, kk + 1), vb(_, _, kk + 1));
        gemm(tiled_mma_pv_f16, tCrP0(_, _, kk), tCrV(_, _, kk), inst[0]);
        gemm(tiled_mma_pv_f16, tCrP1(_, _, kk), tCrV(_, _, kk), inst[1]);
      }
#pragma unroll
      for (int s = 0; s < kNumSub; ++s) {
        auto irc =
            make_tensor(inst[s].data(),
                        ffpa_cute::convert_layout_acc_rowcol(inst[s].layout()));
        auto orc =
            make_tensor(o16[s].data(),
                        ffpa_cute::convert_layout_acc_rowcol(o16[s].layout()));
        float old_lmax = 0.0f;
        float new_lmax = 0.0f;
#pragma unroll
        for (int i = 0; i < cute::size(orc); ++i) {
          old_lmax = fmaxf(old_lmax, fabsf(float(orc(i))));
          new_lmax = fmaxf(new_lmax, fabsf(float(irc(i))));
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
          old_lmax =
              fmaxf(old_lmax, __shfl_xor_sync(0xffffffff, old_lmax, off));
          new_lmax =
              fmaxf(new_lmax, __shfl_xor_sync(0xffffffff, new_lmax, off));
        }
        const int wid = (tid >> 5) & 3;  // warp within this sub-tile
        if ((tid & 31) == 0) {
          red_base[s * 4 + wid] = old_lmax;
          red_base[s * 4 + 4 + wid] = new_lmax;
        }
        __syncthreads();
        float old_max = fmaxf(fmaxf(red_base[s * 4], red_base[s * 4 + 1]),
                              fmaxf(red_base[s * 4 + 2], red_base[s * 4 + 3]));
        float new_max = fmaxf(fmaxf(red_base[s * 4 + 4], red_base[s * 4 + 5]),
                              fmaxf(red_base[s * 4 + 6], red_base[s * 4 + 7]));

        int nq;  // total domain exponent this tile
        float qa, qb;
        if (guarded[s]) {
          // scratch holds inst at full scale (npre == 0 here). The old
          // accumulator already sits in the g_old domain (|.| <= old_max),
          // so only the new summand g_new*inst sets the required depth --
          // PLUS the old accumulator itself: o16 grows by qb*inst every
          // tile, so the domain must deepen once rs*old_max re-approaches
          // kSafeMax (otherwise ~500-2500/tile accumulates to f16 inf
          // over a few dozen tiles; this was the const-V N>=8192 inf).
          int dn = 0;
          for (;;) {
            const float g = ldexpf(1.0f, -2 * (o_sg[s] + dn));
            const float g_old = ldexpf(1.0f, -2 * dn);
            if ((g * new_max <= kSafeMax && g_old * old_max <= kSafeMax) ||
                o_sg[s] + dn >= 60)
              break;
            ++dn;
          }
          nq = o_sg[s] + dn;
          qa = ldexpf(1.0f, -2 * dn);
          qb = ldexpf(1.0f, -2 * nq);
        } else {
          // scratch holds 4^-npre*inst; the old accumulator is full scale.
          nq = npre[s];
          for (;;) {
            float worst = 0.0f;
#pragma unroll
            for (int row = 0; row < 2; ++row) {
              const float rs = (kv_tile > 0) ? row_scale[s][row] : 1.0f;
              worst =
                  fmaxf(worst, rs * ldexpf(1.0f, -2 * nq) * old_max +
                                   ldexpf(1.0f, -2 * (nq - npre[s])) * new_max);
            }
            if (worst <= kSafeMax || nq >= 8)
              break;
            ++nq;
          }
          qa = ldexpf(1.0f, -2 * nq);
          qb = ldexpf(1.0f, -2 * (nq - npre[s]));
        }
        const __half2 a2 = __floats2half2_rn(qa, qa);
        const __half2 b2 = __floats2half2_rn(qb, qb);
#pragma unroll
        for (int row = 0; row < 2; ++row) {
          const float rs = (kv_tile > 0) ? row_scale[s][row] : 1.0f;
          const __half2 rs2 = __floats2half2_rn(rs, rs);
#pragma unroll
          for (int col = 0; col < kO16Cols; col += 2) {
            __half2& o2 = *reinterpret_cast<__half2*>(&orc(row, col));
            __half2 iv = *reinterpret_cast<__half2*>(&irc(row, col));
            o2 = __hadd2(__hmul2(o2, __hmul2(rs2, a2)), __hmul2(iv, b2));
          }
          // row_sum always stays in the true probability domain (it is a
          // mass, <= ~1, so float never underflows); online_softmax already
          // rescaled the old sum by rs. Only o16 lives in the g domain.
          row_sum[s][row] += dd[s][row] / p_quant_scale;
          o_loose[s][row] = kSafeMax * 0.5f;
        }
        if (nq > 0) {
          guarded[s] = true;
          o_sg[s] = nq;
        }
        __syncthreads();  // release red_base slots before the next sub
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
  // of either sub-tile or the out-of-bounds tail); tiles before it run a
  // mask-free body.
  const int oob_start = (Nkv % kBc == 0) ? Tc : (Nkv - 1) / kBc;
  const int tail_start = min(min(mask_start[0], mask_start[1]), oob_start);

#pragma unroll 1
  for (int kv_tile = 0; kv_tile < tail_start && kv_tile < Tc_eff; ++kv_tile)
    process_tile(kv_tile, false);
#pragma unroll 1
  for (int kv_tile = tail_start; kv_tile < Tc_eff; ++kv_tile)
    process_tile(kv_tile, true);
  cp_async_wait<0>();

  // ---- Epilogue: dequant, normalize, store per sub-tile ----
  {
    auto mO = make_tensor(
        make_gmem_ptr(O + (Nb_id * Nh * Nq * kHeadDim) + Nh_id * Nq * kHeadDim),
        make_shape(Nq, Int<kHeadDim>{}), make_stride(Int<kHeadDim>{}, _1{}));
    auto cO = make_identity_tensor(Shape<Int<kBrSub>, Int<kHeadDim>>{});

#pragma unroll
    for (int s = 0; s < kNumSub; ++s) {
      auto gO = local_tile(mO, Shape<Int<kBrSub>, Int<kHeadDim>>{},
                           make_coord(Q_tile_id * kNumSub + s, _0{}));
      auto tCgO = thr_mma_pv.partition_C(gO);
      auto tOcO = thr_mma_pv.partition_C(cO);

      auto o_rc = make_tensor(
          o16[s].data(), ffpa_cute::convert_layout_acc_rowcol(o16[s].layout()));
      // ElementO fragment over the same C partition (epilogue-only; the
      // f32 mirror registers do not coexist with the main-loop state).
      auto tOf32 = partition_fragment_C(tiled_mma_pv,
                                        Shape<Int<kBrSub>, Int<kHeadDim>>{});
      auto tOHalf = ffpa_cute::convert_type<ElementO>(tOf32);
      auto tOH_rc = make_tensor(
          tOHalf.data(), ffpa_cute::convert_layout_acc_rowcol(tOHalf.layout()));
#pragma unroll
      for (int row = 0; row < 2; ++row) {
        const float inv_sum =
            (row_sum[s][row] == 0.0f) ? 1.0f : 1.0f / row_sum[s][row];
        // o16 lives in the g=4^-o_sg domain while row_sum is in the true
        // probability domain, so restore g in this multiplier (f32, outside
        // the f16 accumulator).
        const float mul = ldexpf(inv_sum * kFP8FixedPScale, 2 * o_sg[s]);
#pragma unroll
        for (int col = 0; col < kO16Cols; ++col)
          tOH_rc(row, col) = ElementO(float(o_rc(row, col)) * mul);
      }

      if (Br_base + s * kBrSub + kBrSub <= Nq) {
        copy(tOHalf, tCgO);
      } else {
#pragma unroll
        for (int i = 0; i < size(tOHalf); ++i) {
          const int global_row = Br_base + s * kBrSub + get<0>(tOcO(i));
          if (global_row < Nq)
            tCgO(i) = tOHalf(i);
        }
      }

      if (softmax_lse != nullptr) {
        auto cS_s = make_identity_tensor(Shape<Int<kBrSub>, Int<kBc>>{});
        auto tScS = thr_mma_qk.partition_C(cS_s);
        auto tScS_rc = make_tensor(
            tScS.data(), ffpa_cute::convert_layout_acc_rowcol(tScS.layout()));
        const int lse_base = Nb_id * Nh * Nq + Nh_id * Nq;
#pragma unroll
        for (int row = 0; row < 2; ++row) {
          float lse = (row_max[s][row] + log2f(row_sum[s][row])) * FFPA_M_LN2;
          if (smooth_lse)
            lse += scale_orig * qs * qkm[s][row];
          const int global_row = Br_base + s * kBrSub + get<0>(tScS_rc(row, 0));
          if (global_row < Nq)
            softmax_lse[lse_base + global_row] = lse;
        }
      }
    }
  }
#endif  // defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
}

}  // namespace ffpa_fp8

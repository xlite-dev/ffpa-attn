#pragma once

// tensor.hpp MUST precede any cute/atom/* header (see split_d.cuh).
#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cutlass/cutlass.h>

#include "../gemm.cuh"
#include "../attn_traits.cuh"
#include "../attn_bias.cuh"
#include "../dropout.cuh"
#include "../softmax.cuh"

// Persist-D Flash Attention forward (cp.async, sm_80+).
// sm_80 port of the sm120 persist-D geometry: kBr=128, kBc scaled with D
// (128/64/32), Q persisted in smem (A-fragment s2r'd once, gemm_rs QK),
// K/V in independent stage pools. sm_80 has no TMA/mbarrier, so the
// producer/consumer WS split becomes a fully-synchronous 256T non-WS
// loop: K/V tiles ride 16B cp.async committed as per-tile K/V group
// pairs and are waited by FIFO group counting (QK needs group 2t, PV
// group 2t+1 of the per-thread sequence Q,K0,V0,K1,V1,...).
// Loads run over 64-column segments (= one swizzle atom wide), so the
// thread tiling stays integral for every supported D.
// Divergence from the sm120 kernel: bias is gmem-direct only (tile modes
// reuse the Q area under mbarrier handoff, which does not exist here).
template <typename Traits, int kHasAttnBias = 0, int kHasDropout = 0>
__global__ void __launch_bounds__(Traits::kNumThreads, 1)
    persist_d_fwd_cute_sm80(
        typename Traits::Element* __restrict__ Q,
        typename Traits::Element* __restrict__ K,
        typename Traits::Element* __restrict__ V,
        typename Traits::Element* __restrict__ O,
        float* __restrict__ softmax_lse, int Nq, int Nkv, int Nh, int Nh_kv,
        float scale, int Tc, int causal,
        const void* __restrict__ attn_bias = nullptr, int attn_bias_dtype = 0,
        long long attn_bias_stride_b = 0, long long attn_bias_stride_h = 0,
        long long attn_bias_stride_m = 0, long long attn_bias_stride_n = 0,
        float dropout_p = 0.0f, unsigned long long philox_seed = 0,
        unsigned long long philox_offset = 0, int dropout_bitmap_on = 0) {
  using namespace cute;
  using Element = typename Traits::Element;
  using SmemLayoutQ = typename Traits::SmemLayoutQ;
  using SmemLayoutKV = typename Traits::SmemLayoutKV;
  using SmemLayoutKVt = typename Traits::SmemLayoutKVt;
  using TiledMmaQK = typename Traits::TiledMmaQK;
  using TiledMmaPV = typename Traits::TiledMmaPV;
  using SmemCopyAtom = typename Traits::SmemCopyAtom;
  using SmemCopyAtomTransposed = typename Traits::SmemCopyAtomTransposed;

  constexpr int kBr = Traits::kBr;
  constexpr int kBc = Traits::kBc;
  constexpr int kHeadDim = Traits::kHeadDim;
  constexpr int kStages = Traits::kStagesK;
  static_assert(Traits::kStagesK == Traits::kStagesV,
                "the single kStages pool partition assumes K/V stage parity");
  constexpr int kNumThreads = Traits::kNumThreads;

  constexpr int kQTileElements = cosize(SmemLayoutQ{});
  constexpr int kKVTileElements = cosize(SmemLayoutKV{});

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

  // SMEM: [Q persist | K stages | V stages | dropout bitmap]
  extern __shared__ __align__(1024) Element shm[];
  Element* q_base = shm;
  Element* k_base = q_base + kQTileElements;
  Element* v_base = k_base + kStages * kKVTileElements;
  // PC-14 dropout keep-bitmap: [kBr,kBc] bits x2 stages past the V pool
  // (launcher budgets it). Same half-row scheme as the sm80 split_d;
  // kBc<64 (D=192/256 geometry) stays on inline Philox.
  constexpr int kBitmapU32PerStage = kBr * kBc / 32;
  constexpr bool kBitmapCapable = kBc % 64 == 0 && kNumThreads == 2 * kBr;
  uint32_t* bitmap_base =
      reinterpret_cast<uint32_t*>(v_base + kStages * kKVTileElements);

  // G2S TiledCopy: 16B cp.async over [rows, 64] segments (one swizzle
  // atom wide, keeps the thread tiling integral for every D%64==0).
  using G2SCopyOp = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
  using G2SCopyAtom = Copy_Atom<Copy_Traits<G2SCopyOp>, Element>;
  constexpr int kSegCols = 64;
  constexpr int kG2SThrN = kSegCols / 8;
  constexpr int kG2SThrM = kNumThreads / kG2SThrN;
  using G2SCopy = decltype(make_tiled_copy(
      G2SCopyAtom{},
      make_layout(make_shape(Int<kG2SThrM>{}, Int<kG2SThrN>{}),
                  make_stride(Int<kG2SThrN>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<8>{}))));
  G2SCopy g2s_copy;
  auto g2s_thr = g2s_copy.get_slice(tid);

  // Gmem tensors: [total_rows, kHeadDim] row-major.
  auto mQ = make_tensor(make_gmem_ptr(Q + q_row_offset * kHeadDim),
                        make_shape(Nq, Int<kHeadDim>{}),
                        make_stride(Int<kHeadDim>{}, _1{}));
  auto mK = make_tensor(make_gmem_ptr(K + kv_row_offset * kHeadDim),
                        make_shape(Nkv, Int<kHeadDim>{}),
                        make_stride(Int<kHeadDim>{}, _1{}));
  auto mV = make_tensor(make_gmem_ptr(V + kv_row_offset * kHeadDim),
                        make_shape(Nkv, Int<kHeadDim>{}),
                        make_stride(Int<kHeadDim>{}, _1{}));

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
    auto sK = make_tensor(make_smem_ptr(k_base + stage * kKVTileElements),
                          SmemLayoutKV{});
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
    auto gV = local_tile(mV, Shape<Int<kBc>, Int<kHeadDim>>{},
                         make_coord(kv_tile_idx, _0{}));
    auto sV = make_tensor(make_smem_ptr(v_base + stage * kKVTileElements),
                          SmemLayoutKV{});
    CUTLASS_PRAGMA_UNROLL
    for (int seg = 0; seg < kHeadDim / kSegCols; ++seg) {
      auto gSeg = local_tile(gV, Shape<Int<kBc>, Int<kSegCols>>{},
                             make_coord(_0{}, seg));
      auto sSeg = local_tile(sV, Shape<Int<kBc>, Int<kSegCols>>{},
                             make_coord(_0{}, seg));
      copy(g2s_copy, g2s_thr.partition_S(gSeg), g2s_thr.partition_D(sSeg));
    }
  };

  // Dual TiledMma (same layout as the sm120 persist-D).
  TiledMmaQK tiled_mma_qk;
  TiledMmaPV tiled_mma_pv;
  auto thr_mma_qk = tiled_mma_qk.get_thread_slice(tid);
  auto thr_mma_pv = tiled_mma_pv.get_thread_slice(tid);

  // S2R copy atoms.
  auto s2r_copy_q = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma_qk);
  auto s2r_copy_k = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma_qk);
  auto s2r_copy_v = make_tiled_copy_B(SmemCopyAtomTransposed{}, tiled_mma_pv);
  auto s2r_thr_q = s2r_copy_q.get_thread_slice(tid);
  auto s2r_thr_k = s2r_copy_k.get_thread_slice(tid);
  auto s2r_thr_v = s2r_copy_v.get_thread_slice(tid);

  // V fragment layout for gemm_rs.
  auto sV0 = make_tensor(make_smem_ptr(v_base), SmemLayoutKV{});
  auto sVt0_ns =
      make_tensor(sV0.data(), get_nonswizzle_portion(SmemLayoutKVt{}));
  auto tCrV_layout = thr_mma_pv.partition_fragment_B(sVt0_ns).layout();

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

  const float inv_scale = 1.0f / scale;
  scale *= FFPA_M_LOG2E;

  float row_max[kORows];
  float row_sum[kORows];
#pragma unroll
  for (int r = 0; r < kORows; ++r) {
    row_max[r] = -INFINITY;
    row_sum[r] = 0.0f;
  }

  float o_acc[kOElemsPerFrag];
#pragma unroll
  for (int i = 0; i < kOElemsPerFrag; ++i)
    o_acc[i] = 0.0f;

  // Initial loads: Q group, then K/V[0..S-1] as per-tile pairs. The
  // per-thread commit sequence is Q, K0, V0, K1, V1, ... so at kv tile t
  // (before the loop-tail commits) K[t] is group 2t+1 and V[t] group
  // 2t+2 of 1 + 2*min(S + t, Tc_eff) in flight -> wait<2S-1> settles K[t],
  // wait<2S-2> settles V[t].
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
    cp_async_wait<0>();
    __syncthreads();
  }

  // Q s2r once: the A fragment is loop-invariant (persist-D), so every
  // QK step below runs as gemm_rs (K-only smem loads).
  auto sQ = make_tensor(make_smem_ptr(q_base), SmemLayoutQ{});
  auto tCrQ = thr_mma_qk.partition_fragment_A(sQ);
  auto tQsQ_s2r = s2r_thr_q.partition_S(sQ);
  {
    auto tXrQ = s2r_thr_q.retile_D(tCrQ);
#pragma unroll
    for (int tile_k = 0; tile_k < size<2>(tCrQ); ++tile_k)
      copy(s2r_copy_q, tQsQ_s2r(_, _, tile_k), tXrQ(_, _, tile_k));
  }

  // PC-14 dropout bitmap: stage(0) into buffer 0 before the kv loop.
  const bool bitmap_on =
      kBitmapCapable && kHasDropout && dropout_bitmap_on != 0;
  const unsigned long long dropout_head_base =
      (static_cast<unsigned long long>(Nb_id) * Nh + Nh_id) * Nq;
  if constexpr (kBitmapCapable) {
    if (bitmap_on && Tc_eff > 0) {
      ffpa_cute::generate_dropout_bitmap_halfrow<kBc>(
          bitmap_base, tid >> 1, tid & 1, Br_base + (tid >> 1), 0, dropout_p,
          philox_seed, philox_offset, dropout_head_base, Nkv);
      __syncthreads();
    }
  }

#pragma unroll 1
  for (int kv_tile = 0; kv_tile < Tc_eff; ++kv_tile) {
    const int k_stg = kv_tile % kStages;
    const int v_stg = kv_tile % kStages;

    // Bitmap for the next tile: before the K wait so it fills the
    // cp.async flight window instead of the softmax->PV critical path.
    if constexpr (kBitmapCapable) {
      if (bitmap_on && kv_tile + 1 < Tc_eff)
        ffpa_cute::generate_dropout_bitmap_halfrow<kBc>(
            bitmap_base + ((kv_tile + 1) & 1) * kBitmapU32PerStage, tid >> 1,
            tid & 1, Br_base + (tid >> 1), kv_tile + 1, dropout_p, philox_seed,
            philox_offset, dropout_head_base, Nkv);
    }

    // QK GEMM: gemm_rs with the loop-invariant Q A-fragment in regs,
    // full-D Q x full-D K (K-only smem loads).
    cp_async_wait<kStages * 2 - 1>();
    __syncthreads();

    auto sK = make_tensor(make_smem_ptr(k_base + k_stg * kKVTileElements),
                          SmemLayoutKV{});
    auto tCrK = thr_mma_qk.partition_fragment_B(sK);
    auto tKsK_s2r = s2r_thr_k.partition_S(sK);

    auto tCrS = partition_fragment_C(tiled_mma_qk, Shape<Int<kBr>, Int<kBc>>{});
    clear(tCrS);
    ffpa_cute::gemm_rs(tCrS, tCrQ, tCrK, tKsK_s2r, tiled_mma_qk, s2r_copy_k,
                       s2r_thr_k);

    // Online softmax
    auto scores = make_tensor(
        tCrS.data(), ffpa_cute::convert_layout_acc_rowcol(tCrS.layout()));
    float row_scale[kORows];

    {
      const int kv_valid = Nkv - kv_tile * kBc;
      if (kv_valid < kBc) {
#pragma unroll
        for (int row = 0; row < kSRows; ++row)
#pragma unroll
          for (int col = 0; col < kSCols; ++col) {
            if (get<1>(tScS_rc(row, col)) >= kv_valid)
              scores(row, col) = -INFINITY;
          }
      }
    }

    if (kv_tile >= mask_start_tile) {
#pragma unroll
      for (int row = 0; row < kSRows; ++row) {
        const int q_pos = Br_base + get<0>(tScS_rc(row, 0)) + kv_offset;
#pragma unroll
        for (int col = 0; col < kSCols; ++col) {
          const int k_pos = kv_tile * kBc + get<1>(tScS_rc(row, col));
          if (k_pos > q_pos)
            scores(row, col) = -INFINITY;
        }
      }
    }

    // Additive attention bias (pre-softmax, gmem-direct FC-4 path).
    if constexpr (kHasAttnBias) {
      const int bias_q_valid = min(kBr, Nq - Br_base);
      const int bias_kv_valid = min(kBc, Nkv - kv_tile * kBc);
      const bool full_tile = bias_q_valid >= kBr && bias_kv_valid >= kBc;
      if (__builtin_expect(full_tile, 1))
        ffpa_cute::apply_attn_bias_rowcol<decltype(scores), decltype(tScS_rc),
                                          kSRows, kSCols, false>(
            scores, tScS_rc, attn_bias, attn_bias_dtype, attn_bias_stride_b,
            attn_bias_stride_h, attn_bias_stride_m, attn_bias_stride_n, Nb_id,
            Nh_id, Br_base, kv_tile, kBc, inv_scale, bias_q_valid,
            bias_kv_valid);
      else
        ffpa_cute::apply_attn_bias_rowcol<decltype(scores), decltype(tScS_rc),
                                          kSRows, kSCols, true>(
            scores, tScS_rc, attn_bias, attn_bias_dtype, attn_bias_stride_b,
            attn_bias_stride_h, attn_bias_stride_m, attn_bias_stride_n, Nb_id,
            Nh_id, Br_base, kv_tile, kBc, inv_scale, bias_q_valid,
            bias_kv_valid);
    }

    ffpa_cute::online_safe_softmax<decltype(scores), decltype(tScS_rc), kORows>(
        scores, tScS_rc, scale, row_max, row_sum, row_scale,
        Traits::kRescaleThreshold);

    bool local_need_rescale = false;
#pragma unroll
    for (int r = 0; r < kORows; ++r)
      local_need_rescale = local_need_rescale || (row_scale[r] < 1.0f);
    const bool need_rescale = __any_sync(0xffffffff, local_need_rescale);

    // Dropout on P (post-softmax, pre-PV).
    if constexpr (kHasDropout) {
      if (bitmap_on) {
        ffpa_cute::apply_dropout_bitmap_rowcol<
            decltype(scores), decltype(tScS_rc), kSRows, kSCols, kBc>(
            scores, tScS_rc, bitmap_base + (kv_tile & 1) * kBitmapU32PerStage,
            1.0f / (1.0f - dropout_p));
        __syncthreads();
      } else {
        ffpa_cute::apply_dropout_rowcol<decltype(scores), decltype(tScS_rc),
                                        kORows, kSCols>(
            scores, tScS_rc, dropout_p, philox_seed, philox_offset, Nb_id, Nh,
            Nh_id, Nq, Nkv, Br_base, kv_tile, kBc);
      }
    }

    // Rescale O accumulator
    if (kv_tile > 0 && need_rescale) {
      auto tCrO = make_tensor(make_rmem_ptr(o_acc), OFragLayout{});
      auto tCrO_rc = make_tensor(
          tCrO.data(), ffpa_cute::convert_layout_acc_rowcol(tCrO.layout()));
#pragma unroll
      for (int row = 0; row < kORows; ++row)
#pragma unroll
        for (int col = 0; col < kOCols; ++col)
          tCrO_rc(row, col) *= row_scale[row];
    }

    // PV GEMM: single gemm_rs, full-D P x full-D V.
    cp_async_wait<kStages * 2 - 2>();
    __syncthreads();

    auto sV = make_tensor(make_smem_ptr(v_base + v_stg * kKVTileElements),
                          SmemLayoutKV{});
    auto sVt = make_tensor(sV.data(), SmemLayoutKVt{});
    auto tCrVStorage = thr_mma_pv.partition_fragment_B(sV);
    auto tCrV = make_tensor(tCrVStorage.data(), tCrV_layout);
    auto tVsVt_s2r = s2r_thr_v.partition_S(sVt);

    auto tCrP = ffpa_cute::convert_type<Element>(tCrS);
    auto tCrPv = make_tensor(
        tCrP.data(),
        ffpa_cute::convert_layout_acc_Aregs<TiledMmaPV>(tCrP.layout()));
    auto tCrO = make_tensor(make_rmem_ptr(o_acc), OFragLayout{});
    ffpa_cute::gemm_rs(tCrO, tCrPv, tCrV, tVsVt_s2r, tiled_mma_pv, s2r_copy_v,
                       s2r_thr_v);

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

  // Phase 4: Epilogue. Normalize O, convert, store.
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
      const float inv_sum = 1.0f / row_sum[row];
#pragma unroll
      for (int col = 0; col < kOCols; ++col)
        tCrO_rc(row, col) *= inv_sum;
    }
    auto tCrOHalf = ffpa_cute::convert_type<Element>(tCrO);
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
        const float lse = (row_max[row] + log2f(row_sum[row])) * FFPA_M_LN2;
        const int global_row = Br_base + get<0>(tScS_rc(row, 0));
        if (global_row < Nq)
          softmax_lse[lse_base + global_row] = lse;
      }
    }
  }
}

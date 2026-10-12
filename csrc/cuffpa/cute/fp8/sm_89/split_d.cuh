#pragma once

// FP8 split-D Flash Attention forward (cp.async, sm_89+).
//
// Split-D sibling of the sm_89 persist-D kernel: the head dim is sharded
// into kQKDChunk (QK) / kVDChunk (PV) chunks so 224 < D <= 512 fits the
// register budget (O accumulators are D/2 fp32 per thread at 256T, the same
// geometry the sm120 fp8 split-D runs at D=512). Q is re-loaded per kv_tile
// (no register residency -- the split-D trade), K/V are staged through the
// kStagesQK/kStagesPV cp.async pipeline.
//
// Three-way merge of the existing family code (no new modes):
//   - load layer: the PC-12-validated cp.async skeleton of the fp16 sm_80
//     split_d.cuh (per-thread gmem/smem chunk addressing derived once, 16B
//     cp.async per thread, group-FIFO with the d_chunk/v_chunk double
//     buffers, the __syncthreads() before the next-tile prefetch that
//     guards a stage reissue while lagging threads still read it),
//   - D-chunk structure and quant math: the fp8 sm_120 split_d.cuh
//     (gemm_ss QK accumulation over d_chunks, per-v_chunk O accumulators,
//     per-thread QK / per-channel V index math, q_start_row coordinate
//     re-bases, mode-0 gmem-direct bias),
//   - fp8 compute layer: the sm_89 persist_d.cuh (fixed p_scale softmax,
//     prescaled P -> e4m3 pack, pscale_rowsum_mma, two-level f16 inst_buf
//     accumulator, kFP8FixedPScale epilogue, smooth-K lse correction).
//
// The PV MMA atom is built locally as SM89_16x8x32_F16E4M3E4M3F16_TN: the
// traits' MmaAtomPVf16 is SM120 (Blackwell-only) and must not be referenced
// from an sm_89 TU. Its C layout matches the f32 atom (both m16n8k32 with
// the SM80_16x8_Row C convention), so the f16 inst_buf absorbs into the
// float o_acc element-wise.
//
// Quant scope: per-block/per-thread QK x per-block/per-channel V (the four
// combos the launcher instantiates), smooth_k lse correction, smooth_v mean
// add-back, additive attn_bias in the raw score domain (mode 0 only), and
// q_start_row for the fp8-hybrid stage-2. P packing is the shuffle-free
// perm pack (PackC8bitToA8bitPermVT), paired with the permuted V^T the
// quantize pre-kernel writes (launcher reorg_free=true; the pairing is
// load-bearing -- never run the perm pack against an unpermuted V^T).
// v1 scope: no dropout, BHND O only (the launcher rejects NHD views), f16
// PV accumulator only, D % 64 == 0 (kQKDChunk=32 / kVDChunk=64).
//
// Inherited family debt (same as persist_d sm89 / sm_80 fp16 split_d): the
// cp.async G2S loader is unpredicated, so tail tiles read past Nq/Nkv rows
// and the Nkv-pad of the V^T plane; the values are masked out by kv_valid /
// the row guards and the reads stay inside the quantized-buffer allocation
// slack. Predication is tracked as a separate family item.

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

template <typename Traits, typename ElementO, int kHasAttnBias = 0,
          bool kQKPerThread = false, bool kVPerChannel = false>
__global__ void __launch_bounds__(Traits::kNumThreads, 1)
    split_d_fwd_cute_fp8_sm89(
        typename Traits::ElementQK* __restrict__ Q,
        typename Traits::ElementQK* __restrict__ K,
        typename Traits::Element* __restrict__ V,  // VT (D, Nkv_pad)
        ElementO* __restrict__ O, float* __restrict__ softmax_lse,
        const float* __restrict__ q_scale, const float* __restrict__ k_scale,
        const float* __restrict__ v_scale, int Nq, int Nkv, int Nh, int Nh_kv,
        int n_rb_q, int n_rb_kv, float scale, int Tc, int causal, int Nkv_pad,
        int q_start_row = 0, const float* __restrict__ km = nullptr,
        const float* __restrict__ vm = nullptr,
        const void* __restrict__ attn_bias = nullptr, int attn_bias_dtype = 0,
        long long attn_bias_stride_b = 0, long long attn_bias_stride_h = 0,
        long long attn_bias_stride_m = 0, long long attn_bias_stride_n = 0) {
  // Body-level arch guard (see sm_89/persist_d.cuh): mixed -gencode builds
  // compile the sm_89 device pass into a stub; the dispatch only routes this
  // kernel to sm_89 hardware (or the force-sm89 dev-box path).
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  using namespace cute;
  using Element = typename Traits::Element;      // float_e4m3_t (V / P)
  using ElementQK = typename Traits::ElementQK;  // int8 (kQKInt8) or e4m3
  using SmemLayoutQ = typename Traits::SmemLayoutQ;
  using SmemLayoutK = typename Traits::SmemLayoutK;
  using SmemLayoutV = typename Traits::SmemLayoutV;
  using TiledMmaQK = typename Traits::TiledMmaQK;
  using SmemCopyAtom = typename Traits::SmemCopyAtom;
  using SmemCopyAtomQK = typename Traits::SmemCopyAtomQK;

  constexpr int kBr = Traits::kBr;              // 128 (256T non-WS, 1 CTA/SM)
  constexpr int kBc = Traits::kBc;              // 128
  constexpr int kQKDChunk = Traits::kQKDChunk;  // 32
  constexpr int kVDChunk = Traits::kVDChunk;    // 64
  constexpr int kHeadDim = Traits::kHeadDim;
  constexpr int kDChunksQK = Traits::kDChunksQK;
  constexpr int kDChunksV = Traits::kDChunksV;
  constexpr int kNumThreads = Traits::kNumThreads;  // 256
  constexpr int kStagesQK = Traits::kStagesQK;
  constexpr int kStagesPV = Traits::kStagesPV;

  constexpr int kQChunkElements = cosize(SmemLayoutQ{});
  constexpr int kKChunkElements = cosize(SmemLayoutK{});
  constexpr int kVChunkElements = cosize(SmemLayoutV{});

  // Per-channel-V P quant scale (compile-time: the epilogue dequant divides
  // it back out). One tile's inst_buf worst case is kBc*448*v_r(2.25), so
  // kBc=128 must narrow P to 224 to stay inside the f16 range.
  constexpr float kPQuantScalePerCh =
      (kBc * kE4m3Max * 2.25f <= 65504.0f) ? kE4m3Max : 224.0f;

  // f16 PV atom (Ada-only m16n8k32 f16-acc), warps over the kBr rows, N tile
  // is the kVDChunk split-D chunk (not the full kHeadDim). Built locally:
  // the traits' PVf16 atom is SM120 (Blackwell) and unusable on sm_89.
  using MmaAtomPVf16 = MMA_Atom<SM89_16x8x32_F16E4M3E4M3F16_TN>;
  using TiledMmaPVf16 = decltype(make_tiled_mma(
      MmaAtomPVf16{}, Layout<Shape<Int<kNumThreads / 32>, _1, _1>>{},
      Tile<Int<kBr>, Int<kVDChunk>, _32>{}));

  const int Nb_id = blockIdx.y / Nh;
  const int Nh_id = blockIdx.y % Nh;
  const int Q_tile_id = blockIdx.x;
  const int group_size = Nh / Nh_kv;
  const int kv_head_idx = Nh_id / group_size;
  const int Br_base = Q_tile_id * kBr;
  const int tid = threadIdx.x;

  // Hybrid stage-2: rows [q_start_row, Nq); grid.x covers the remainder and
  // q_tile_abs re-bases the quant-scale indexing on absolute tiles.
  if (Br_base >= Nq - q_start_row)
    return;

  const int kv_offset = Nkv - Nq;
  const int causal_thresh_row0 = q_start_row + Br_base + kv_offset;
  const int Tc_eff =
      causal
          ? min(Tc, ((q_start_row + Br_base + kBr - 1 + kv_offset) / kBc) + 1)
          : Tc;
  const int mask_start_tile =
      causal ? max(0, (causal_thresh_row0 + 1) / kBc) : INT_MAX;

  const int q_row_offset = (Nb_id * Nh + Nh_id) * Nq + q_start_row;
  const int kv_row_offset = (Nb_id * Nh_kv + kv_head_idx) * Nkv;
  const int q_bh = Nb_id * Nh + Nh_id;
  const int kv_bh = Nb_id * Nh_kv + kv_head_idx;
  const int q_tile_abs = Q_tile_id + q_start_row / kBr;

  // SMEM carve: [Q stages | K stages | V stages], 1B per elem (int8/e4m3).
  // Stage s of each pool sits at its base plus s * cosize(2D chunk tile).
  extern __shared__ __align__(1024) char shm[];
  ElementQK* q_base = reinterpret_cast<ElementQK*>(shm);
  ElementQK* k_base = q_base + kStagesQK * kQChunkElements;
  Element* v_base =
      reinterpret_cast<Element*>(k_base + kStagesQK * kKChunkElements);

  // G2S TiledCopy: 256 threads, 128-bit cp.async. Separate copies for the
  // QK (kQKDChunk-wide) and V (kVDChunk-wide) chunk tiles; the 16B value
  // layout keeps every copy inside one swizzle row (SW32 for QK chunks,
  // SW128 for V chunks -- traits pick the atoms by chunk row bytes).
  using G2SCopyOp = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
  using G2SCopyAtom = Copy_Atom<Copy_Traits<G2SCopyOp>, Element>;
  constexpr int kG2SThrN_QK = kQKDChunk / 16;
  constexpr int kG2SThrM_QK = kNumThreads / kG2SThrN_QK;
  constexpr int kG2SThrN_V = kVDChunk / 16;
  constexpr int kG2SThrM_V = kNumThreads / kG2SThrN_V;
  using G2SCopyQK = decltype(make_tiled_copy(
      G2SCopyAtom{},
      make_layout(make_shape(Int<kG2SThrM_QK>{}, Int<kG2SThrN_QK>{}),
                  make_stride(Int<kG2SThrN_QK>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<16>{}))));
  using G2SCopyV = decltype(make_tiled_copy(
      G2SCopyAtom{},
      make_layout(make_shape(Int<kG2SThrM_V>{}, Int<kG2SThrN_V>{}),
                  make_stride(Int<kG2SThrN_V>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<16>{}))));
  G2SCopyQK g2s_copy_qk;
  G2SCopyV g2s_copy_v;
  auto g2s_thr_qk = g2s_copy_qk.get_slice(tid);
  auto g2s_thr_v = g2s_copy_v.get_slice(tid);

  // Gmem tensors: Q/K are [rows, kHeadDim] row-major over the flattened
  // quantized buffers; V^T is the flat [B*Nh_kv*kHeadDim, Nkv_pad] plane
  // stack (per-(b,h) base advances by kHeadDim D-rows, i.e.
  // kHeadDim*Nkv_pad elements -- the FC-13 h>=1 offset bug).
  auto mQ = make_tensor(make_gmem_ptr(Q + q_row_offset * kHeadDim),
                        make_shape(Nq, Int<kHeadDim>{}),
                        make_stride(Int<kHeadDim>{}, _1{}));
  auto mK = make_tensor(make_gmem_ptr(K + kv_row_offset * kHeadDim),
                        make_shape(Nkv, Int<kHeadDim>{}),
                        make_stride(Int<kHeadDim>{}, _1{}));
  auto mV = make_tensor(
      make_gmem_ptr(V + static_cast<long>(kv_bh) * kHeadDim * Nkv_pad),
      make_shape(kHeadDim, Nkv), make_stride(Nkv_pad, _1{}));

  auto g2s_load_q = [&](int d_chunk, int stage) {
    auto gQ = local_tile(mQ, Shape<Int<kBr>, Int<kQKDChunk>>{},
                         make_coord(Q_tile_id, d_chunk));
    auto sQ = make_tensor(make_smem_ptr(q_base + stage * kQChunkElements),
                          SmemLayoutQ{});
    copy(g2s_copy_qk, g2s_thr_qk.partition_S(gQ), g2s_thr_qk.partition_D(sQ));
  };

  auto g2s_load_k = [&](int kv_tile_idx, int d_chunk, int stage) {
    auto gK = local_tile(mK, Shape<Int<kBc>, Int<kQKDChunk>>{},
                         make_coord(kv_tile_idx, d_chunk));
    auto sK = make_tensor(make_smem_ptr(k_base + stage * kKChunkElements),
                          SmemLayoutK{});
    copy(g2s_copy_qk, g2s_thr_qk.partition_S(gK), g2s_thr_qk.partition_D(sK));
  };

  // V^T chunk: rows [v_chunk*kVDChunk, +kVDChunk) of the D x Nkv plane,
  // cols [kv_tile*kBc, +kBc); loaded as the PV B-operand (D x N layout).
  auto g2s_load_v = [&](int kv_tile_idx, int v_chunk, int stage) {
    auto gV = local_tile(mV, Shape<Int<kVDChunk>, Int<kBc>>{},
                         make_coord(v_chunk, kv_tile_idx));
    auto sV = make_tensor(make_smem_ptr(v_base + stage * kVChunkElements),
                          SmemLayoutV{});
    copy(g2s_copy_v, g2s_thr_v.partition_S(gV), g2s_thr_v.partition_D(sV));
  };

  TiledMmaQK tiled_mma_qk;
  TiledMmaPVf16 tiled_mma_pv_f16;
  auto thr_mma_qk = tiled_mma_qk.get_thread_slice(tid);
  auto thr_mma_pv_f16 = tiled_mma_pv_f16.get_thread_slice(tid);

  auto s2r_copy_q = make_tiled_copy_A(SmemCopyAtomQK{}, tiled_mma_qk);
  auto s2r_copy_k = make_tiled_copy_B(SmemCopyAtomQK{}, tiled_mma_qk);
  // The B-side (V fragment + smem copy) must derive from the f16 TiledMma,
  // else CuTe gemm silently no-ops (same trap as persist_d fp8).
  auto s2r_copy_v = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma_pv_f16);
  auto s2r_thr_q = s2r_copy_q.get_thread_slice(tid);
  auto s2r_thr_k = s2r_copy_k.get_thread_slice(tid);
  auto s2r_thr_v = s2r_copy_v.get_thread_slice(tid);

  // s32 (int8 QK) or f32 (e4m3 QK) score fragment: 2 rows x (kBc/4) cols.
  using ScoreFrag =
      decltype(partition_fragment_C(tiled_mma_qk, Shape<Int<kBr>, Int<kBc>>{}));
  using ScoreRCLayout =
      decltype(ffpa_cute::convert_layout_acc_rowcol(ScoreFrag{}.layout()));
  constexpr int kSCols = decltype(cute::size<1>(
      make_tensor((float*)nullptr, ScoreRCLayout{})))::value;
  constexpr int kSRows = decltype(cute::size<0>(
      make_tensor((float*)nullptr, ScoreRCLayout{})))::value;
  static_assert(kSRows == 2, "");

  // O fragment layout per v_chunk (f16 PV atom; C layout matches the f32
  // atom so the half inst_buf absorbs into the float o_acc element-wise).
  using OFragType = decltype(partition_fragment_C(
      tiled_mma_pv_f16, Shape<Int<kBr>, Int<kVDChunk>>{}));
  using OFragLayout = typename OFragType::layout_type;
  constexpr int kOElemsPerFrag = decltype(size(OFragType{}))::value;
  constexpr int kORows = decltype(cute::size<0>(
      make_tensor((float*)nullptr,
                  ffpa_cute::convert_layout_acc_rowcol(OFragLayout{}))))::value;
  constexpr int kOCols = decltype(cute::size<1>(
      make_tensor((float*)nullptr,
                  ffpa_cute::convert_layout_acc_rowcol(OFragLayout{}))))::value;
  static_assert(kORows == 2, "");

  // Coordinate tensor for softmax indexing.
  auto cS = make_identity_tensor(Shape<Int<kBr>, Int<kBc>>{});
  auto tScS = thr_mma_qk.partition_C(cS);
  auto tScS_rc = make_tensor(
      tScS.data(), ffpa_cute::convert_layout_acc_rowcol(tScS.layout()));

  const float scale_orig = scale;
  scale *= FFPA_M_LOG2E;

  // Per-row Q dequant scales (kSRows == kORows == 2: the QK C-frag row pair
  // and the PV C-frag rows index the same Q rows). Per-thread mode maps each
  // row into the 128-row quant block: g=(q_row/16)*8+q_row%8 (kBr=128 tiles
  // align with the 128-row quant blocks, so q_tile_abs indexes directly).
  float qs_arr[kSRows];
  if constexpr (kQKPerThread) {
#pragma unroll
    for (int row = 0; row < kSRows; ++row) {
      const int q_row = get<0>(tScS_rc(row, 0));
      const int g = (q_row / 16) * 8 + q_row % 8;
      qs_arr[row] = q_scale[static_cast<long>(q_bh) * (n_rb_q * 64) +
                            q_tile_abs * 64 + g];
    }
  } else {
    const float qs = q_scale[static_cast<long>(q_bh) * n_rb_q + q_tile_abs];
#pragma unroll
    for (int row = 0; row < kSRows; ++row)
      qs_arr[row] = qs;
  }

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

  // Persistent O accumulators across the KV-tile loop (see the sm_120
  // split_d register-budget analysis): D/2 fp32 regs per thread.
  float o_acc_storage[kDChunksV][kOElemsPerFrag];
#pragma unroll
  for (int v = 0; v < kDChunksV; ++v)
#pragma unroll
    for (int i = 0; i < kOElemsPerFrag; ++i)
      o_acc_storage[v][i] = 0.0f;

  // Smooth-K lse correction partials: dot(Q8_row, km) accumulated per D
  // chunk during kv_tile 0 (Q is re-loaded every kv_tile, so only the first
  // pass needs to compute it).
  const bool smooth_lse = (softmax_lse != nullptr) && (km != nullptr);

  PackC8bitToA8bitPermVT perm_pack;
  using PLayer = Layout<Shape<Shape<_4, _2, _2>, _1, Int<kBc / 32>>>;

  // Initial V prefetch for kv_tile 0.
  {
    int v_write = 0;
#pragma unroll
    for (int v = 0; v < kStagesPV - 1 && v < kDChunksV; ++v) {
      g2s_load_v(0, v, v_write);
      cp_async_fence();
      v_write = (v_write + 1) % kStagesPV;
    }
    if constexpr (kStagesPV > 1) {
      cp_async_wait<kStagesPV - 2>();
      __syncthreads();
    }
  }

  // Initial QK prefetch for kv_tile 0.
  {
    int qk_write = 0;
#pragma unroll
    for (int d = 0; d < kStagesQK - 1 && d < kDChunksQK; ++d) {
      g2s_load_q(d, qk_write);
      g2s_load_k(0, d, qk_write);
      cp_async_fence();
      qk_write = (qk_write + 1) % kStagesQK;
    }
    if constexpr (kStagesQK > 1) {
      cp_async_wait<kStagesQK - 2>();
      __syncthreads();
    } else {
      __syncthreads();
    }
  }

#pragma unroll 1
  for (int kv_tile = 0; kv_tile < Tc_eff; ++kv_tile) {
    // V prefetch for kv_tile > 0.
    if (kv_tile > 0) {
      int v_write = 0;
#pragma unroll
      for (int v = 0; v < kStagesPV - 1 && v < kDChunksV; ++v) {
        g2s_load_v(kv_tile, v, v_write);
        cp_async_fence();
        v_write = (v_write + 1) % kStagesPV;
      }
      if constexpr (kStagesPV > 1) {
        cp_async_wait<kStagesPV - 2>();
        __syncthreads();
      }
    }

    // Phase 1: QK GEMM with split-D accumulation.
    auto tCrS = partition_fragment_C(tiled_mma_qk, Shape<Int<kBr>, Int<kBc>>{});
    clear(tCrS);

    int qk_read = 0;
    int qk_write = (kStagesQK > 1) ? (kStagesQK - 1) : 0;

#pragma unroll
    for (int d_chunk = 0; d_chunk < kDChunksQK; ++d_chunk) {
      const int d_next = d_chunk + kStagesQK - 1;

      if constexpr (kStagesQK == 1) {
        // stages=1: prefetch and gemm share the same stage. Sync before
        // prefetch to ensure the previous gemm's smem reads are done.
        if (d_chunk > 0)
          __syncthreads();
        g2s_load_q(d_chunk, 0);
        g2s_load_k(kv_tile, d_chunk, 0);
        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();
      } else {
        // stages>=2: issue prefetch (non-blocking, overlaps gemm below).
        if (d_next < kDChunksQK) {
          g2s_load_q(d_next, qk_write);
          g2s_load_k(kv_tile, d_next, qk_write);
          cp_async_fence();
          qk_write = (qk_write + 1) % kStagesQK;
        }
      }

      auto sQ = make_tensor(make_smem_ptr(q_base + qk_read * kQChunkElements),
                            SmemLayoutQ{});
      auto sK = make_tensor(make_smem_ptr(k_base + qk_read * kKChunkElements),
                            SmemLayoutK{});
      auto tCrQ = thr_mma_qk.partition_fragment_A(sQ);
      auto tCrK = thr_mma_qk.partition_fragment_B(sK);
      auto tQsQ = s2r_thr_q.partition_S(sQ);
      auto tKsK = s2r_thr_k.partition_S(sK);

      ffpa_cute::gemm_ss(tCrS, tCrQ, tCrK, tQsQ, tKsK, tiled_mma_qk, s2r_copy_q,
                         s2r_copy_k, s2r_thr_q, s2r_thr_k);

      if (kv_tile == 0 && smooth_lse)
        smooth_k_qk_dot<kQKDChunk, kORows, true>(
            sQ, tScS_rc,
            km + static_cast<long>(kv_bh) * kHeadDim + d_chunk * kQKDChunk,
            qkm);

      qk_read = (qk_read + 1) % kStagesQK;

      // stages>=2: post-wait ensures next iteration's data is ready.
      if constexpr (kStagesQK > 1) {
        if (d_chunk < kDChunksQK - 1) {
          if (d_next < kDChunksQK) {
            cp_async_wait<kStagesQK - 2>();
          } else {
            cp_async_wait<0>();
          }
          __syncthreads();
        }
      }
    }

    // Prefetch next kv_tile's initial QK chunks (overlaps softmax + PV).
    if (kv_tile < Tc_eff - 1) {
      // The last d_chunk skips the post-wait above; sync so lagging threads
      // finish reading its stage before this reissues stage 0.
      __syncthreads();
      int qk_write_next = 0;
#pragma unroll
      for (int d = 0; d < kStagesQK - 1 && d < kDChunksQK; ++d) {
        g2s_load_q(d, qk_write_next);
        g2s_load_k(kv_tile + 1, d, qk_write_next);
        cp_async_fence();
        qk_write_next = (qk_write_next + 1) % kStagesQK;
      }
      if constexpr (kStagesQK > 1) {
        cp_async_wait<kStagesQK - 2>();
        __syncthreads();
      }
    }

    // int8 QK: cast the s32 acc to f32 in place over the same 4B regs;
    // identity view on the e4m3 path (accumulator is already f32).
    auto tCrSf =
        make_tensor(reinterpret_cast<float*>(tCrS.data()), tCrS.layout());
    if constexpr (Traits::kQKInt8) {
#pragma unroll
      for (int i = 0; i < size(tCrS); ++i)
        tCrSf(i) = static_cast<float>(tCrS(i));
    }

    // Phase 2: fixed-P-scale softmax (fp8_pscale.cuh). Masked tiles pay an
    // explicit scale pass so -inf clamps happen before exp2; unmasked tiles
    // defer scale into the softmax (one multiply per exp).
    auto scores = make_tensor(
        tCrSf.data(), ffpa_cute::convert_layout_acc_rowcol(tCrS.layout()));

    // K scale: per-block (1 per tile) or per-thread (4 per tile, the lane's
    // C-frag column group lane%4 covers cols {2*(lane%4)+8n, +1}).
    const float ks =
        kQKPerThread ? k_scale[static_cast<long>(kv_bh) * (n_rb_kv * 4) +
                               kv_tile * 4 + (tid % 32) % 4]
                     : k_scale[static_cast<long>(kv_bh) * n_rb_kv + kv_tile];
    // Per-channel V: v_scale is (bh, D); vs is unused in-tile (P uses the
    // fixed compile-time scale, the epilogue dequants per D column).
    const float vs =
        kVPerChannel ? 1.0f
                     : v_scale[static_cast<long>(kv_bh) * n_rb_kv + kv_tile];
    const float p_quant_scale =
        kVPerChannel ? kPQuantScalePerCh : (vs * kE4m3Max);

    // Additive attn bias in the RAW score domain (before any dequant-scale
    // path): every path below scales raw scores by qs_arr[row]*ks*scale, so
    // bias/(qs_arr[row]*ks*scale_orig) lands as +bias in softmax-input units
    // on masked and unmasked tiles alike; the -INFINITY assignments in the
    // masking block below simply override it.
    // Bias tail guards: rows/cols of this tile inside the real row window
    // [q_start_row, Nq) x [0, Nkv); the injector clamps pad rows/cols to
    // these bounds (the masking block below overrides their scores anyway).
    const int bias_q_valid = min(kBr, Nq - q_start_row - Br_base);
    const int bias_kv_valid = min(kBc, Nkv - kv_tile * kBc);
    if constexpr (kHasAttnBias) {
      float bias_inv[kSRows];
#pragma unroll
      for (int row = 0; row < kSRows; ++row)
        bias_inv[row] = 1.0f / (qs_arr[row] * ks * scale_orig);
      const bool full_tile = bias_q_valid >= kBr && bias_kv_valid >= kBc;
      if (__builtin_expect(full_tile, 1))
        ffpa_cute::apply_attn_bias_quant_rowcol<
            decltype(scores), decltype(tScS_rc), kSRows, kSCols, false>(
            scores, tScS_rc, attn_bias, attn_bias_dtype, attn_bias_stride_b,
            attn_bias_stride_h, attn_bias_stride_m, attn_bias_stride_n, Nb_id,
            Nh_id, q_start_row + Br_base, kv_tile, kBc, bias_inv, bias_q_valid,
            bias_kv_valid);
      else
        ffpa_cute::apply_attn_bias_quant_rowcol<
            decltype(scores), decltype(tScS_rc), kSRows, kSCols, true>(
            scores, tScS_rc, attn_bias, attn_bias_dtype, attn_bias_stride_b,
            attn_bias_stride_h, attn_bias_stride_m, attn_bias_stride_n, Nb_id,
            Nh_id, q_start_row + Br_base, kv_tile, kBc, bias_inv, bias_q_valid,
            bias_kv_valid);
    }

    const int kv_valid = Nkv - kv_tile * kBc;
    const bool tile_needs_mask =
        (kv_valid < kBc) || (kv_tile >= mask_start_tile);
    if (tile_needs_mask) {
#pragma unroll
      for (int row = 0; row < kSRows; ++row) {
        const int q_pos =
            q_start_row + Br_base + get<0>(tScS_rc(row, 0)) + kv_offset;
#pragma unroll
        for (int col = 0; col < kSCols; ++col) {
          float s = scores(row, col) * qs_arr[row] * ks * scale;
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

    if constexpr (kQKPerThread) {
      // Per-thread QK: pre-dequant scores per-row, then softmax with 'scale'.
      if (!tile_needs_mask) {
#pragma unroll
        for (int row = 0; row < kSRows; ++row) {
          const float sd = qs_arr[row] * ks;
#pragma unroll
          for (int col = 0; col < kSCols; ++col)
            scores(row, col) *= sd;
        }
      }
      const float softmax_scale_eff = tile_needs_mask ? 1.0f : scale;
      online_softmax_fp8_fixed<true, decltype(scores), decltype(tScS_rc),
                               kORows, Traits::kQKInt8>(
          scores, tScS_rc, softmax_scale_eff, row_max, row_sum, row_scale,
          log2f(p_quant_scale), 1.0f / p_quant_scale,
          Traits::kRescaleThreshold);
    } else {
      const float s_dequant = qs_arr[0] * ks;
      const float softmax_scale_eff =
          tile_needs_mask ? 1.0f : s_dequant * scale;
      online_softmax_fp8_fixed<true, decltype(scores), decltype(tScS_rc),
                               kORows, Traits::kQKInt8>(
          scores, tScS_rc, softmax_scale_eff, row_max, row_sum, row_scale,
          log2f(p_quant_scale), 1.0f / p_quant_scale,
          Traits::kRescaleThreshold);
    }

    // P -> e4m3 A operand (fixed mode: softmax already emitted P*vs*448).
    // The perm pack binds the kVTPerm V^T from the quantize pre-kernel
    // (launcher reorg_free=true; the pairing is load-bearing).
    quantize_p_frag_prescaled(tCrSf, perm_pack);
    auto tCrP = make_tensor(reinterpret_cast<Element*>(tCrSf.data()), PLayer{});
    // Tensor-core row sum over the quantized P regs (exact w.r.t. the P the
    // PV MMA consumes); folds vs*448 back out of the probability domain.
    pscale_rowsum_mma(tCrP, row_sum, 1.0f / p_quant_scale);

    // Phase 3: PV GEMM over the split-D V chunks.
    int v_read = 0;
    int v_write_pv = (kStagesPV > 1) ? (kStagesPV - 1) : 0;

#pragma unroll
    for (int v_chunk = 0; v_chunk < kDChunksV; ++v_chunk) {
      const int v_next = v_chunk + kStagesPV - 1;

      if constexpr (kStagesPV == 1) {
        // stages=1: prefetch and gemm share the same stage.
        if (v_chunk > 0)
          __syncthreads();
        g2s_load_v(kv_tile, v_chunk, 0);
        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();
      } else {
        // stages>=2: issue prefetch (non-blocking, overlaps gemm below).
        if (v_next < kDChunksV) {
          g2s_load_v(kv_tile, v_next, v_write_pv);
          cp_async_fence();
          v_write_pv = (v_write_pv + 1) % kStagesPV;
        }
      }

      auto sV = make_tensor(make_smem_ptr(v_base + v_read * kVChunkElements),
                            SmemLayoutV{});

      // f8f8f16 PV: fp16 accumulator avoids 22-bit f8f8f32 loss on causal
      // early rows; absorb to the float o_acc via one fused fmaf per element
      // (rs is skipped on the first tile: o_acc is zero anyway).
      auto tCrV_f16 = thr_mma_pv_f16.partition_fragment_B(sV);
      auto tVsV_f16 = s2r_thr_v.partition_S(sV);
      auto tCrInst = partition_fragment_C(tiled_mma_pv_f16,
                                          Shape<Int<kBr>, Int<kVDChunk>>{});
      clear(tCrInst);
      ffpa_cute::gemm_rs(tCrInst, tCrP, tCrV_f16, tVsV_f16, tiled_mma_pv_f16,
                         s2r_copy_v, s2r_thr_v);

      auto tCrO =
          make_tensor(make_rmem_ptr(&o_acc_storage[v_chunk][0]), OFragLayout{});
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

      v_read = (v_read + 1) % kStagesPV;

      // stages>=2: post-wait ensures next iteration's data is ready.
      if constexpr (kStagesPV > 1) {
        if (v_chunk < kDChunksV - 1) {
          if (v_next < kDChunksV) {
            cp_async_wait<kStagesPV - 2>();
          } else {
            cp_async_wait<0>();
          }
          __syncthreads();
        }
      }
    }
    __syncthreads();
  }
  cp_async_wait<0>();

  // Phase 4: epilogue. O = o_acc * (1/448) / row_sum (fixed-mode domain),
  // per-v_chunk dequant + R->G store on BHND; the tail tile guards rows.
  {
    auto mO = make_tensor(
        make_gmem_ptr(O + (Nb_id * Nh * Nq * kHeadDim) + Nh_id * Nq * kHeadDim +
                      static_cast<long>(q_start_row) * kHeadDim),
        make_shape(Nq - q_start_row, Int<kHeadDim>{}),
        make_stride(Int<kHeadDim>{}, _1{}));
    auto cO = make_identity_tensor(Shape<Int<kBr>, Int<kVDChunk>>{});
    auto tOcO = thr_mma_pv_f16.partition_C(cO);
    // Per-channel V: D-column coords via the PV C-fragment (chunk-local
    // [0, kVDChunk)); the v_chunk base offset is added at load time below.
    auto cD = make_identity_tensor(Shape<Int<kBr>, Int<kVDChunk>>{});
    auto tScD = thr_mma_pv_f16.partition_C(cD);
    auto tScD_rc = make_tensor(
        tScD.data(), ffpa_cute::convert_layout_acc_rowcol(tScD.layout()));
    const float* vs_d_base =
        kVPerChannel ? (v_scale + static_cast<long>(kv_bh) * kHeadDim)
                     : nullptr;
    const float* vm_base = (kVPerChannel && vm)
                               ? (vm + static_cast<long>(kv_bh) * kHeadDim)
                               : nullptr;

#pragma unroll
    for (int v_chunk = 0; v_chunk < kDChunksV; ++v_chunk) {
      auto tCrO =
          make_tensor(make_rmem_ptr(&o_acc_storage[v_chunk][0]), OFragLayout{});
      auto tCrO_rc = make_tensor(
          tCrO.data(), ffpa_cute::convert_layout_acc_rowcol(tCrO.layout()));
      float vs_d_col[kVPerChannel ? kOCols : 1];
      float vm_d_col[kVPerChannel ? kOCols : 1];
      if constexpr (kVPerChannel) {
        const float* vs_d_v = vs_d_base + v_chunk * kVDChunk;
        const float* vm_d_v =
            vm_base ? (vm_base + v_chunk * kVDChunk) : nullptr;
#pragma unroll
        for (int col = 0; col < kOCols; ++col) {
          const int d_idx = get<1>(tScD_rc(0, col));
          vs_d_col[col] = vs_d_v[d_idx];
          if (vm_d_v)
            vm_d_col[col] = vm_d_v[d_idx];
        }
      }
#pragma unroll
      for (int row = 0; row < kORows; ++row) {
        const float inv_sum =
            (row_sum[row] == 0.0f) ? 1.0f : 1.0f / row_sum[row];
#pragma unroll
        for (int col = 0; col < kOCols; ++col) {
          // Per-channel balanced narrowing dequant: divide p_quant_scale.
          const float mul = kVPerChannel
                                ? (inv_sum * vs_d_col[col] / kPQuantScalePerCh)
                                : (inv_sum * kFP8FixedPScale);
          tCrO_rc(row, col) *= mul;
          if (vm_base)
            tCrO_rc(row, col) += vm_d_col[col];
        }
      }
      auto tCrOHalf = ffpa_cute::convert_type<ElementO>(tCrO);
      auto gO = local_tile(mO, Shape<Int<kBr>, Int<kVDChunk>>{},
                           make_coord(Q_tile_id, v_chunk));
      auto tCgO = thr_mma_pv_f16.partition_C(gO);
      if (Br_base + kBr <= Nq - q_start_row) {
        copy(tCrOHalf, tCgO);
      } else {
#pragma unroll
        for (int i = 0; i < size(tCrOHalf); ++i) {
          const int global_row = q_start_row + Br_base + get<0>(tOcO(i));
          if (global_row < Nq)
            tCgO(i) = tCrOHalf(i);
        }
      }
    }

    if (softmax_lse != nullptr) {
      const int lse_base = Nb_id * Nh * Nq + Nh_id * Nq;
#pragma unroll
      for (int row = 0; row < kORows; ++row) {
        float lse = (row_max[row] + log2f(row_sum[row])) * FFPA_M_LN2;
        if (smooth_lse)
          lse += scale_orig * qs_arr[row] * qkm[row];
        const int global_row = q_start_row + Br_base + get<0>(tScS_rc(row, 0));
        if (global_row < Nq)
          softmax_lse[lse_base + global_row] = lse;
      }
    }
  }
#endif  // defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
}

}  // namespace ffpa_fp8

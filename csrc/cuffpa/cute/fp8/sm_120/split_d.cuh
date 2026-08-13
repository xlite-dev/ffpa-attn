#pragma once

#include <cuda_fp8.h>

#include <cute/tensor.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cutlass.h>
#include <cutlass/device_kernel.h>

#include "../../gemm.cuh"
#include "../../attn_traits.cuh"
#include "../fp8_pscale.cuh"
#include "../reg2reg_8b.cuh"
#include "../smooth_k.cuh"

// FP8 causal accuracy note: same ESS-rooted early-row amplitude effect as
// persist_d.cuh (see the detailed math comment there). split_d only supports
// per-block V / f32 PV acc, so the per-channel-V mitigation is unavailable
// here; causal early-row abs error is intrinsic to fp8 + small ESS.

namespace ffpa_fp8 {

using TmaBarrier = cutlass::arch::ClusterTransactionBarrier;
using CtaBarrier = cutlass::arch::ClusterBarrier;

// Split-D FP8 forward (non-WS, CuTe TMA). Name keeps the persist_d "ws"
// convention; the implementation is the non-WS M8N1 design of
// split_d_fwd_cute_sm120 (all threads do MMA, tid=0 issues TMA inline).
//
// fp8 e4m3 Q/K/V (kQKInt8: Q/K symmetric int8 with s32 QK MMA cast to f32;
// PV always fp8). V is pre-transposed (D x N) by the quantize pre-kernel.
// Blockwise scales (per kBr/kBc row block): s_dequant = qs*ks folded into
// the log2-domain softmax; fixed P scale 1/448 via exp_offset = log2(vs*448)
// so the PV MMA domain cancels vs: (P*vs*448) @ (V/vs) = 448*(P@V), and the
// epilogue dequants with (1/448)/row_sum. See fp8_pscale.cuh for the math.
// attn_bias/dropout are not supported on this path.
//
// Known accuracy limit (shared with persist_d fp8, unresolved): causal
// early rows attend few keys, so per-element QK/P quant errors are not
// averaged out -- O error stays < 1e-1 but lse error can reach ~6e-2
// (e4m3 P rounding of near-1 probabilities), vs ~4e-3 for dense.
// kQKPerThread: per-block (false, 1 scale/block) vs per-thread (true,
//   Q=64/block, K=4/block, fragment-aligned). See persist_d.cuh for details.
template <typename Traits, typename ElementO, typename TmaQ, typename TmaK,
          typename TmaV, typename TmaO, bool kPVAccF16 = false,
          bool kVPerChannel = false, bool kQKPerThread = false>
__global__ void __launch_bounds__(Traits::kNumThreads, 1)
    split_d_fwd_cute_fp8_sm120(
        CUTLASS_GRID_CONSTANT TmaQ const tma_q,
        CUTLASS_GRID_CONSTANT TmaK const tma_k,
        CUTLASS_GRID_CONSTANT TmaV const tma_v,
        CUTLASS_GRID_CONSTANT TmaO const tma_o, ElementO* __restrict__ O,
        float* __restrict__ softmax_lse, const float* __restrict__ q_scale,
        const float* __restrict__ k_scale, const float* __restrict__ v_scale,
        int Nq, int Nkv, int Nh, int Nh_kv, float scale, int Tc, int causal,
        int total_q_rows, int total_kv_rows, int n_rb_q, int n_rb_kv,
        int q_start_row = 0, const float* __restrict__ km = nullptr,
        const float* __restrict__ vm = nullptr) {
  // Body-level arch guard (see sm_120/split_d.cuh): mixed -gencode builds
  // compile the sm_89 device pass into a stub; launch.cuh only dispatches
  // this kernel on sm>=90 devices.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  using namespace cute;
  using cute::tma_store_arrive;
  using cute::tma_store_wait;
  using Element = typename Traits::Element;      // float_e4m3_t (V/PV)
  using ElementQK = typename Traits::ElementQK;  // int8 (kQKInt8) or e4m3
  using SmemLayoutQ = typename Traits::SmemLayoutQ;
  using SmemLayoutK = typename Traits::SmemLayoutK;
  using SmemLayoutV = typename Traits::SmemLayoutV;
  using SmemLayoutO = typename Traits::SmemLayoutO;
  using TiledMmaQK = typename Traits::TiledMmaQK;
  using TiledMmaPV = typename Traits::TiledMmaPV;
  using TiledMmaPVf16 = typename Traits::TiledMmaPVf16;
  using SmemCopyAtom = typename Traits::SmemCopyAtom;
  using SmemCopyAtomQK = typename Traits::SmemCopyAtomQK;

  constexpr int kBr = Traits::kBr;
  constexpr int kBc = Traits::kBc;
  constexpr int kQKDChunk = Traits::kQKDChunk;
  constexpr int kVDChunk = Traits::kVDChunk;
  constexpr int kHeadDim = Traits::kHeadDim;
  constexpr int kDChunksQK = Traits::kDChunksQK;
  constexpr int kDChunksV = Traits::kDChunksV;
  constexpr int kNumThreads = Traits::kNumThreads;
  constexpr int kStagesQK = Traits::kStagesQK;
  constexpr int kStagesPV = Traits::kStagesPV;
  constexpr int kSmemBytes = Traits::kSmemElems;

  constexpr int kQChunkElements = cosize(SmemLayoutQ{});
  constexpr int kKChunkElements = cosize(SmemLayoutK{});
  constexpr int kVChunkElements = cosize(SmemLayoutV{});

  // O staging reuses the whole freed smem from its base; guard the batched
  // staging budget in bytes (O is ElementO 2B, QK/V smem counts 1B elems).
  static_assert(
      Traits::kVChunksPerBatch * cosize(SmemLayoutO{}) * sizeof(ElementO) <=
          kSmemBytes,
      "TMA-O: batched O staging must fit the reused smem");

  const int Nb_id = blockIdx.y / Nh;
  const int Nh_id = blockIdx.y % Nh;
  const int Q_tile_id = blockIdx.x;
  const int group_size = Nh / Nh_kv;
  const int kv_head_idx = Nh_id / group_size;
  const int Br_base = Q_tile_id * kBr;
  const int tid = threadIdx.x;

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

  // SMEM: [Q stages | K stages | V stages], 1B per elem (int8 or e4m3).
  extern __shared__ __align__(1024) char shm[];
  ElementQK* q_base = reinterpret_cast<ElementQK*>(shm);
  ElementQK* k_base = q_base + kStagesQK * kQChunkElements;
  Element* v_base =
      reinterpret_cast<Element*>(k_base + kStagesQK * kKChunkElements);

  __shared__ uint64_t qk_full[kStagesQK];
  __shared__ uint64_t qk_empty[kStagesQK];
  __shared__ uint64_t v_full[kStagesPV];
  __shared__ uint64_t v_empty[kStagesPV];

  if (tid == 0) {
    for (int s = 0; s < kStagesQK; ++s) {
      TmaBarrier::init(&qk_full[s], 1);
      CtaBarrier::init(&qk_empty[s], kNumThreads);
    }
    for (int s = 0; s < kStagesPV; ++s) {
      TmaBarrier::init(&v_full[s], 1);
      CtaBarrier::init(&v_empty[s], kNumThreads);
    }
  }
  __syncthreads();

  // TMA views: Q/K over the flattened [total_rows, kHeadDim] quantized
  // buffers; V^T is a flat [B*Nh_kv*D, Nkv] plane stack, offset by the KV
  // head's D plane (persist_d_fp8 pattern).
  auto mQ = domain_offset(
      make_coord(q_row_offset, 0),
      tma_q.get_tma_tensor(make_shape(total_q_rows, Int<kHeadDim>{})));
  auto mK = domain_offset(
      make_coord(kv_row_offset, 0),
      tma_k.get_tma_tensor(make_shape(total_kv_rows, Int<kHeadDim>{})));
  const int v_row_base = kv_bh * kHeadDim;
  const int d_total = (total_kv_rows / Nkv) * kHeadDim;
  auto mV = domain_offset(make_coord(v_row_base, _0{}),
                          tma_v.get_tma_tensor(make_shape(d_total, Nkv)));
  auto q_slice = tma_q.get_slice(_0{});
  auto k_slice = tma_k.get_slice(_0{});
  auto v_slice = tma_v.get_slice(_0{});

  TiledMmaQK tiled_mma_qk;
  TiledMmaPV tiled_mma_pv;
  [[maybe_unused]] TiledMmaPVf16 tiled_mma_pv_f16;
  auto thr_mma_qk = tiled_mma_qk.get_thread_slice(tid);
  auto thr_mma_pv = tiled_mma_pv.get_thread_slice(tid);

  auto s2r_copy_q = make_tiled_copy_A(SmemCopyAtomQK{}, tiled_mma_qk);
  auto s2r_copy_k = make_tiled_copy_B(SmemCopyAtomQK{}, tiled_mma_qk);
  auto s2r_copy_v = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma_pv);
  auto s2r_thr_q = s2r_copy_q.get_thread_slice(tid);
  auto s2r_thr_k = s2r_copy_k.get_thread_slice(tid);
  auto s2r_thr_v = s2r_copy_v.get_thread_slice(tid);
  // f16 PV path: B-side (V fragment + smem copy) must derive from the f16
  // TiledMma, else CuTe gemm silently no-ops (same trap as persist_d fp8).
  [[maybe_unused]] auto thr_mma_pv_f16 = tiled_mma_pv_f16.get_thread_slice(tid);
  [[maybe_unused]] auto s2r_copy_v_f16 =
      make_tiled_copy_B(SmemCopyAtom{}, tiled_mma_pv_f16);
  [[maybe_unused]] auto s2r_thr_v_f16 = s2r_copy_v_f16.get_thread_slice(tid);

  using OFragType = decltype(partition_fragment_C(
      tiled_mma_pv, Shape<Int<kBr>, Int<kVDChunk>>{}));
  using OFragLayout = typename OFragType::layout_type;
  constexpr int kOElemsPerFrag = decltype(size(OFragType{}))::value;
  constexpr int kORows = decltype(size<0>(
      make_tensor((float*)nullptr,
                  ffpa_cute::convert_layout_acc_rowcol(OFragLayout{}))))::value;
  constexpr int kOCols = decltype(size<1>(
      make_tensor((float*)nullptr,
                  ffpa_cute::convert_layout_acc_rowcol(OFragLayout{}))))::value;

  auto cS = make_identity_tensor(Shape<Int<kBr>, Int<kBc>>{});
  auto tScS = thr_mma_qk.partition_C(cS);
  auto tScS_rc = make_tensor(
      tScS.data(), ffpa_cute::convert_layout_acc_rowcol(tScS.layout()));
  constexpr int kSRows = decltype(size<0>(tScS_rc))::value;
  constexpr int kSCols = decltype(size<1>(tScS_rc))::value;

  // Per-row Q dequant scales (one per scores-fragment row). Per-thread mode
  // looks up the group for each row via tScS_rc coords: g=(q_row/16)*8+q_row%8
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

  const float scale_orig = scale;
  scale *= FFPA_M_LOG2E;

  float row_max[kORows];
  float row_sum[kORows];
#pragma unroll
  for (int r = 0; r < kORows; ++r) {
    row_max[r] = -INFINITY;
    row_sum[r] = 0.0f;
  }

  // Persistent O accumulators across the KV-tile loop (see split_d.cuh for
  // the split-D register-budget analysis): D/2 fp32 regs per thread.
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
  float qkm[kORows];
#pragma unroll
  for (int r = 0; r < kORows; ++r)
    qkm[r] = 0.0f;

  ReorgC8bitToA8bit reorg;

  for (int s = 0; s < kStagesQK; ++s)
    CtaBarrier::arrive(&qk_empty[s]);
  for (int s = 0; s < kStagesPV; ++s)
    CtaBarrier::arrive(&v_empty[s]);

  auto issue_qk_tma = [&](int d_chunk, int stage, int kv_tile_idx) {
    cutlass::arch::fence_view_async_shared();
    auto sQ = make_tensor(make_smem_ptr(q_base + stage * kQChunkElements),
                          SmemLayoutQ{});
    auto sK = make_tensor(make_smem_ptr(k_base + stage * kKChunkElements),
                          SmemLayoutK{});
    auto gQ = local_tile(mQ, Shape<Int<kBr>, Int<kQKDChunk>>{},
                         make_coord(Q_tile_id, d_chunk));
    auto gK = local_tile(mK, Shape<Int<kBc>, Int<kQKDChunk>>{},
                         make_coord(kv_tile_idx, d_chunk));
    auto tQgQ = q_slice.partition_S(gQ);
    auto tQsQ = q_slice.partition_D(sQ);
    auto tKgK = k_slice.partition_S(gK);
    auto tKsK = k_slice.partition_D(sK);
    TmaBarrier::arrive_and_expect_tx(&qk_full[stage],
                                     sizeof(ElementQK) * (size(sQ) + size(sK)));
    copy(tma_q.with(qk_full[stage]), tQgQ, tQsQ);
    copy(tma_k.with(qk_full[stage]), tKgK, tKsK);
  };

  auto issue_v_tma = [&](int v_chunk, int stage, int kv_tile_idx) {
    cutlass::arch::fence_view_async_shared();
    auto sV = make_tensor(make_smem_ptr(v_base + stage * kVChunkElements),
                          SmemLayoutV{});
    auto gV = local_tile(mV, Shape<Int<kVDChunk>, Int<kBc>>{},
                         make_coord(v_chunk, kv_tile_idx));
    auto tVgV = v_slice.partition_S(gV);
    auto tVsV = v_slice.partition_D(sV);
    TmaBarrier::arrive_and_expect_tx(&v_full[stage],
                                     sizeof(Element) * size(sV));
    copy(tma_v.with(v_full[stage]), tVgV, tVsV);
  };

  if (tid == 0) {
    for (int d = 0; d < kStagesQK && d < kDChunksQK; ++d) {
      CtaBarrier::wait(&qk_empty[d], 0);
      issue_qk_tma(d, d, 0);
    }
  }
  if (tid == 0) {
    for (int v = 0; v < kStagesPV && v < kDChunksV; ++v) {
      const int chunk_index = v;  // kv_tile == 0
      const int v_stage = chunk_index % kStagesPV;
      const int v_phase = (chunk_index / kStagesPV) & 1;
      CtaBarrier::wait(&v_empty[v_stage], v_phase);
      issue_v_tma(v, v_stage, 0);
    }
  }

#pragma unroll 1
  for (int kv_tile = 0; kv_tile < Tc_eff; ++kv_tile) {
    if (kv_tile > 0 && tid == 0) {
      for (int v = 0; v < kStagesPV && v < kDChunksV; ++v) {
        const int chunk_index = kv_tile * kDChunksV + v;
        const int v_stage = chunk_index % kStagesPV;
        const int v_phase = (chunk_index / kStagesPV) & 1;
        CtaBarrier::wait(&v_empty[v_stage], v_phase);
        issue_v_tma(v, v_stage, kv_tile);
      }
    }

    // K scale: per-block (1 per kBc-col block) or per-thread (4 per block,
    // group lane%4 matches SM89_16x8x32 C-fragment cols {2*(lane%4)+8n, +1};
    // group is valid for all warps since K quant groups repeat every 8 rows).
    const float ks =
        kQKPerThread ? k_scale[static_cast<long>(kv_bh) * (n_rb_kv * 4) +
                               kv_tile * 4 + (tid % 32) % 4]
                     : k_scale[static_cast<long>(kv_bh) * n_rb_kv + kv_tile];
    // Per-channel V: v_scale is (bh, D); skip the per-kv-tile slot (wrong
    // shape). vs=1 placeholder; epilogue dequants per-D instead.
    const float vs =
        kVPerChannel ? 1.0f
                     : v_scale[static_cast<long>(kv_bh) * n_rb_kv + kv_tile];
    // P's fp8 quantization multiplier (P8 = softmax * p_quant_scale).
    // Per-block: vs*448 = amax_block, vs cancels in PV MMA -> unified 448
    //   domain, single epilogue dequant (1/448). Per-channel: 1.0 so P8=
    //   softmax lands in e4m3's [0,1] range; amax_d is global so o_acc's
    //   per-D scale is uniform across tiles -> single epilogue dequant per-D.
    const float p_quant_scale = kVPerChannel ? 1.0f : (vs * kE4m3Max);

    // Phase 1: QK GEMM with split-D accumulation. fp8 accumulates in f32;
    // int8 accumulates in s32 across chunks (exact, <<2^31) and casts once.
    auto tCrS = partition_fragment_C(tiled_mma_qk, Shape<Int<kBr>, Int<kBc>>{});
    clear(tCrS);

#pragma unroll
    for (int d_chunk = 0; d_chunk < kDChunksQK; ++d_chunk) {
      const int chunk_index = kv_tile * kDChunksQK + d_chunk;
      const int stage = chunk_index % kStagesQK;
      const int phase = (chunk_index / kStagesQK) & 1;
      TmaBarrier::wait(&qk_full[stage], phase);
      cutlass::arch::fence_view_async_shared();

      auto sQ = make_tensor(make_smem_ptr(q_base + stage * kQChunkElements),
                            SmemLayoutQ{});
      auto sK = make_tensor(make_smem_ptr(k_base + stage * kKChunkElements),
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

      CtaBarrier::arrive(&qk_empty[stage]);

      if (tid == 0) {
        const int d_next = d_chunk + kStagesQK;
        if (d_next < kDChunksQK) {
          const int next_index = kv_tile * kDChunksQK + d_next;
          const int s_next = next_index % kStagesQK;
          const int phase_next = (next_index / kStagesQK) & 1;
          CtaBarrier::wait(&qk_empty[s_next], phase_next);
          issue_qk_tma(d_next, s_next, kv_tile);
        }
      }
    }

    if (kv_tile < Tc_eff - 1 && tid == 0) {
      for (int d = 0; d < kStagesQK && d < kDChunksQK; ++d) {
        const int chunk_index = (kv_tile + 1) * kDChunksQK + d;
        const int stage = chunk_index % kStagesQK;
        const int phase = (chunk_index / kStagesQK) & 1;
        CtaBarrier::wait(&qk_empty[stage], phase);
        issue_qk_tma(d, stage, kv_tile + 1);
      }
    }

    // int8 QK: cast the s32 acc to f32 in place over the same 4B regs;
    // identity view on the fp8 path.
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

    float row_scale[kORows];
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
                               kORows>(
          scores, tScS_rc, softmax_scale_eff, row_max, row_sum, row_scale,
          log2f(p_quant_scale), 1.0f / p_quant_scale,
          Traits::kRescaleThreshold);
    } else {
      const float s_dequant = qs_arr[0] * ks;
      const float softmax_scale_eff =
          tile_needs_mask ? 1.0f : s_dequant * scale;
      online_softmax_fp8_fixed<true, decltype(scores), decltype(tScS_rc),
                               kORows>(
          scores, tScS_rc, softmax_scale_eff, row_max, row_sum, row_scale,
          log2f(p_quant_scale), 1.0f / p_quant_scale,
          Traits::kRescaleThreshold);
    }

    bool local_need_rescale = false;
#pragma unroll
    for (int r = 0; r < kORows; ++r)
      local_need_rescale = local_need_rescale || (row_scale[r] < 1.0f);
    const bool need_rescale = __any_sync(0xffffffff, local_need_rescale);

    // P -> e4m3 A operand (fixed mode: softmax already emitted P*vs*448).
    quantize_p_frag_prescaled(tCrSf, reorg);
    auto tCrP =
        make_tensor(reinterpret_cast<Element*>(tCrSf.data()),
                    Layout<Shape<Shape<_4, _2, _2>, _1, Int<kBc / 32>>>{});
    // Tensor-core row sum over the quantized P regs (exact w.r.t. the P the
    // PV MMA consumes); folds vs*448 back out of the probability domain.
    pscale_rowsum_mma(tCrP, row_sum, 1.0f / p_quant_scale);

    // Phase 3: PV GEMM over the split-D V chunks.
#pragma unroll
    for (int v_chunk = 0; v_chunk < kDChunksV; ++v_chunk) {
      const int chunk_index = kv_tile * kDChunksV + v_chunk;
      const int v_stage = chunk_index % kStagesPV;
      const int v_phase = (chunk_index / kStagesPV) & 1;
      TmaBarrier::wait(&v_full[v_stage], v_phase);
      cutlass::arch::fence_view_async_shared();

      auto sV = make_tensor(make_smem_ptr(v_base + v_stage * kVChunkElements),
                            SmemLayoutV{});

      auto tCrO =
          make_tensor(make_rmem_ptr(&o_acc_storage[v_chunk][0]), OFragLayout{});
      if (kv_tile > 0 && need_rescale) {
        auto tCrO_rc = make_tensor(
            tCrO.data(), ffpa_cute::convert_layout_acc_rowcol(tCrO.layout()));
#pragma unroll
        for (int row = 0; row < kORows; ++row)
#pragma unroll
          for (int col = 0; col < kOCols; ++col)
            tCrO_rc(row, col) *= row_scale[row];
      }

      if constexpr (kPVAccF16) {
        // f8f8f16 PV: fp16 accumulator avoids 22-bit f8f8f32 loss on causal
        // early rows; absorb to float o_acc via CUDA-core FADD per kv_tile.
        // inst_buf reused across v_chunks (sequential PV).
        auto tCrV_f16 = thr_mma_pv_f16.partition_fragment_B(sV);
        auto tVsV_f16 = s2r_thr_v_f16.partition_S(sV);
        auto tCrInst = partition_fragment_C(tiled_mma_pv_f16,
                                            Shape<Int<kBr>, Int<kVDChunk>>{});
        clear(tCrInst);
        ffpa_cute::gemm_rs(tCrInst, tCrP, tCrV_f16, tVsV_f16, tiled_mma_pv_f16,
                           s2r_copy_v_f16, s2r_thr_v_f16);
        auto tCrO_rc = make_tensor(
            tCrO.data(), ffpa_cute::convert_layout_acc_rowcol(tCrO.layout()));
        auto tCrInst_rc =
            make_tensor(tCrInst.data(),
                        ffpa_cute::convert_layout_acc_rowcol(tCrInst.layout()));
#pragma unroll
        for (int row = 0; row < kORows; ++row)
#pragma unroll
          for (int col = 0; col < kOCols; ++col)
            tCrO_rc(row, col) += float(tCrInst_rc(row, col));
      } else {
        auto tCrV = thr_mma_pv.partition_fragment_B(sV);
        auto tVsV = s2r_thr_v.partition_S(sV);
        ffpa_cute::gemm_rs(tCrO, tCrP, tCrV, tVsV, tiled_mma_pv, s2r_copy_v,
                           s2r_thr_v);
      }

      CtaBarrier::arrive(&v_empty[v_stage]);

      if (tid == 0) {
        const int v_next = v_chunk + kStagesPV;
        if (v_next < kDChunksV) {
          const int next_index = kv_tile * kDChunksV + v_next;
          const int s_next = next_index % kStagesPV;
          const int phase_next = (next_index / kStagesPV) & 1;
          CtaBarrier::wait(&v_empty[s_next], phase_next);
          issue_v_tma(v_next, s_next, kv_tile);
        }
      }
    }
  }

  // Phase 4: epilogue. O = o_acc * (1/448) / row_sum (fixed-mode domain),
  // batched R->S(STSM)->TMA store on aligned tiles, R->G on the tail.
  {
    constexpr int kVChunksPerBatch = Traits::kVChunksPerBatch;
    constexpr int kNBatches = Traits::kNBatches;
    constexpr int kOTileElems = cosize(SmemLayoutO{});

    __syncthreads();  // V smem reads done before R->S overwrites shm

    auto mO_tma = domain_offset(
        make_coord(q_row_offset, 0),
        tma_o.get_tma_tensor(make_shape(total_q_rows, Int<kHeadDim>{})));
    auto o_slice = tma_o.get_slice(_0{});

    auto r2s_copy = make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N, ElementO>{},
                                      tiled_mma_pv);
    auto r2s_thr = r2s_copy.get_slice(tid);

    const int O_gmem_offset = (Nb_id * Nh * Nq * kHeadDim) +
                              (Nh_id * Nq * kHeadDim) + q_start_row * kHeadDim;
    auto mO = make_tensor(make_gmem_ptr(O + O_gmem_offset),
                          make_shape(Nq - q_start_row, Int<kHeadDim>{}),
                          make_stride(Int<kHeadDim>{}, _1{}));
    auto cO = make_identity_tensor(Shape<Int<kBr>, Int<kVDChunk>>{});
    auto tOcO = thr_mma_pv.partition_C(cO);
    // Per-channel V: D-column coords via PV C-fragment (chunk-local [0,
    // kVDChunk)); the v_chunk base offset is added at load time below.
    auto cD = make_identity_tensor(Shape<Int<kBr>, Int<kVDChunk>>{});
    auto tScD = thr_mma_pv.partition_C(cD);
    auto tScD_rc = make_tensor(
        tScD.data(), ffpa_cute::convert_layout_acc_rowcol(tScD.layout()));
    const float* vs_d_base =
        kVPerChannel ? (v_scale + static_cast<long>(kv_bh) * kHeadDim)
                     : nullptr;
    const float* vm_base = (kVPerChannel && vm)
                               ? (vm + static_cast<long>(kv_bh) * kHeadDim)
                               : nullptr;

    if (Br_base + kBr <= Nq - q_start_row) {
#pragma unroll
      for (int batch = 0; batch < kNBatches; ++batch) {
#pragma unroll
        for (int v_in = 0; v_in < kVChunksPerBatch; ++v_in) {
          int v_chunk = batch * kVChunksPerBatch + v_in;
          auto tCrO = make_tensor(make_rmem_ptr(&o_acc_storage[v_chunk][0]),
                                  OFragLayout{});
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
              const float mul = kVPerChannel ? (inv_sum * vs_d_col[col])
                                             : (inv_sum * kFP8FixedPScale);
              tCrO_rc(row, col) *= mul;
              if (vm_base)
                tCrO_rc(row, col) += vm_d_col[col];
            }
          }
          auto tCrOHalf = ffpa_cute::convert_type<ElementO>(tCrO);
          auto sO_v =
              make_tensor(make_smem_ptr(reinterpret_cast<ElementO*>(shm) +
                                        v_in * kOTileElems),
                          SmemLayoutO{});
          auto tCrOHalf_src = r2s_thr.retile_S(tCrOHalf);
          auto tCsO_dst = r2s_thr.partition_D(sO_v);
          copy(r2s_copy, tCrOHalf_src, tCsO_dst);
        }
        cutlass::arch::fence_view_async_shared();
        __syncthreads();
#pragma unroll
        for (int v_in = 0; v_in < kVChunksPerBatch; ++v_in) {
          int v_chunk = batch * kVChunksPerBatch + v_in;
          auto sO_v =
              make_tensor(make_smem_ptr(reinterpret_cast<ElementO*>(shm) +
                                        v_in * kOTileElems),
                          SmemLayoutO{});
          auto gO_tma = local_tile(mO_tma, Shape<Int<kBr>, Int<kVDChunk>>{},
                                   make_coord(Q_tile_id, v_chunk));
          auto tCgO_tma = o_slice.partition_D(gO_tma);
          auto tOsO = o_slice.partition_S(sO_v);
          if (tid == 0) {
            copy(tma_o, tOsO, tCgO_tma);
          }
        }
        tma_store_arrive();
        if (batch < kNBatches - 1) {
          tma_store_wait<0>();  // drain for shm reuse
          __syncthreads();
        }
      }
    } else {
#pragma unroll
      for (int v_chunk = 0; v_chunk < kDChunksV; ++v_chunk) {
        auto tCrO = make_tensor(make_rmem_ptr(&o_acc_storage[v_chunk][0]),
                                OFragLayout{});
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
            const float mul = kVPerChannel ? (inv_sum * vs_d_col[col])
                                           : (inv_sum * kFP8FixedPScale);
            tCrO_rc(row, col) *= mul;
            if (vm_base)
              tCrO_rc(row, col) += vm_d_col[col];
          }
        }
        auto tCrOHalf = ffpa_cute::convert_type<ElementO>(tCrO);
        auto gO = local_tile(mO, Shape<Int<kBr>, Int<kVDChunk>>{},
                             make_coord(Q_tile_id, v_chunk));
        auto tCgO = thr_mma_pv.partition_C(gO);
#pragma unroll
        for (int i = 0; i < size(tCrOHalf); ++i) {
          const int global_row = Br_base + get<0>(tOcO(i));
          if (global_row < Nq - q_start_row)
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

    if (Br_base + kBr <= Nq - q_start_row)
      tma_store_wait<0>();
  }
#endif  // defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
}

}  // namespace ffpa_fp8

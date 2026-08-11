#pragma once

#include <cuda_fp8.h>

// tensor.hpp MUST precede any cute/atom/* header (see sm_80/split_d.cuh).
#include <cute/tensor.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cutlass.h>
#include <cutlass/device_kernel.h>

#include "../../gemm.cuh"
#include "../../attn_traits.cuh"
#include "../fp8_pscale.cuh"
#include "../smooth_k.cuh"

namespace ffpa_fp8 {

using TmaBarrier = cutlass::arch::ClusterTransactionBarrier;
using CtaBarrier = cutlass::arch::ClusterBarrier;

// Split-D M4N2 FP8 forward (non-WS, CuTe TMA). Atom layout (4,2,1): 2
// N-warps split Bc/kVDChunk columns; P goes through SMEM roundtrip (stmatrix
// -> syncthreads -> LDSM_N) because each N-warp holds only half the Bc
// columns. Cross-N-warp softmax reduction via SMEM exchange.
//
// fp8 e4m3 Q/K/V (kQKInt8: Q/K symmetric int8 with s32 QK MMA cast to f32;
// PV always fp8). V is pre-transposed (D x N) by the quantize pre-kernel.
// Blockwise scales: s_dequant = qs*ks folded into log2-domain softmax; fixed
// P scale 1/448 via exp_offset = log2(vs*448) so PV MMA cancels vs. Row sum
// uses fp32 tile_sum (方案 C, single barrier, same structure as fp16 m4n2).
// attn_bias/dropout are not supported.
//
// O regs = D/4 per thread (D=1024 -> 256 regs). See split_d_m4n2.cuh for the
// M4N2 register-budget analysis and cross-point vs M8N1.
// kQKPerThread: per-block (false, 1 scale/block) vs per-thread (true,
//   Q=64/128-row block, K=4/kBc-col block, fragment-aligned). See
//   persist_d.cuh for details. m4n2 note: Q attention tile (kBr=64) is
//   smaller than the 128-row quant block; quant_offset handles the mapping.
template <typename Traits, typename ElementO, typename TmaQ, typename TmaK,
          typename TmaV, typename TmaO, bool kPVAccF16 = false,
          bool kVPerChannel = false, bool kQKPerThread = false>
__global__ void __launch_bounds__(Traits::kNumThreads, 1)
    split_d_m4n2_fwd_cute_fp8_sm120(
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
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  using namespace cute;
  using cute::tma_store_arrive;
  using cute::tma_store_wait;
  using Element = typename Traits::Element;
  using ElementQK = typename Traits::ElementQK;
  using SmemLayoutQ = typename Traits::SmemLayoutQ;
  using SmemLayoutK = typename Traits::SmemLayoutK;
  using SmemLayoutV = typename Traits::SmemLayoutV;
  using SmemLayoutO = typename Traits::SmemLayoutO;
  using SmemLayoutP = typename Traits::SmemLayoutP;
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

  constexpr int kQChunkElements = cosize(SmemLayoutQ{});
  constexpr int kKChunkElements = cosize(SmemLayoutK{});
  constexpr int kVChunkElements = cosize(SmemLayoutV{});
  constexpr int kPElements = cosize(SmemLayoutP{});

  static_assert(cosize(SmemLayoutO{}) <= kStagesPV * cosize(SmemLayoutV{}),
                "TMA-O: O staging buffer must fit in reused V-stage smem");

  const int Nb_id = blockIdx.y / Nh;
  const int Nh_id = blockIdx.y % Nh;
  const int Q_tile_id = blockIdx.x;
  const int group_size = Nh / Nh_kv;
  const int kv_head_idx = Nh_id / group_size;
  const int Br_base = Q_tile_id * kBr;
  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;
  const int n_warp = warp_id / 4;

  if (Br_base >= Nq - q_start_row)
    return;

  // Causal mask is lower-triangular (j <= i), matching PyTorch SDPA; see
  // persist_d.cuh for the rationale behind dropping the Nkv-Nq offset.
  const int causal_thresh_row0 = q_start_row + Br_base;
  const int Tc_eff =
      causal ? min(Tc, ((q_start_row + Br_base + kBr - 1) / kBc) + 1) : Tc;
  const int mask_start_tile =
      causal ? max(0, (causal_thresh_row0 + 1) / kBc) : INT_MAX;

  const int q_row_offset = (Nb_id * Nh + Nh_id) * Nq + q_start_row;
  const int kv_row_offset = (Nb_id * Nh_kv + kv_head_idx) * Nkv;
  const int q_bh = Nb_id * Nh + Nh_id;
  const int kv_bh = Nb_id * Nh_kv + kv_head_idx;
  // Q quantize always uses 128-row blocks; m4n2 attention tiles are 64 rows,
  // so 2 attention tiles map to 1 quantize block.
  const int quant_rb = (q_start_row + Q_tile_id * kBr) / 128;

  // SMEM: [Q stages | K stages | V stages | P staging | exchange]
  extern __shared__ __align__(1024) char shm[];
  ElementQK* q_base = reinterpret_cast<ElementQK*>(shm);
  ElementQK* k_base = q_base + kStagesQK * kQChunkElements;
  Element* v_base =
      reinterpret_cast<Element*>(k_base + kStagesQK * kKChunkElements);
  Element* p_base = v_base + kStagesPV * kVChunkElements;
  float* smem_exchange = reinterpret_cast<float*>(p_base + kPElements);

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

  // TMA views: Q/K over flattened [total_rows, kHeadDim]; V^T is flat
  // [B*Nh_kv*D, Nkv] plane stack, offset by the KV head's D plane.
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
  auto s2r_copy_p = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma_pv);
  auto s2r_copy_v = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma_pv);
  auto s2r_thr_q = s2r_copy_q.get_thread_slice(tid);
  auto s2r_thr_k = s2r_copy_k.get_thread_slice(tid);
  auto s2r_thr_p = s2r_copy_p.get_thread_slice(tid);
  auto s2r_thr_v = s2r_copy_v.get_thread_slice(tid);
  // f16 PV path: B-side (V) must derive from the f16 TiledMma, else CuTe
  // gemm silently no-ops. A-side P reuses tCrPv_storage (layouts match).
  [[maybe_unused]] auto thr_mma_pv_f16 = tiled_mma_pv_f16.get_thread_slice(tid);
  [[maybe_unused]] auto s2r_copy_v_f16 =
      make_tiled_copy_B(SmemCopyAtom{}, tiled_mma_pv_f16);
  [[maybe_unused]] auto s2r_thr_v_f16 = s2r_copy_v_f16.get_thread_slice(tid);

  auto sV0 = make_tensor(make_smem_ptr(v_base), SmemLayoutV{});
  auto tCrV_layout = thr_mma_pv.partition_fragment_B(sV0).layout();

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

  // Per-row Q dequant scales. Per-thread mode: 64 groups per 128-row quantize
  // block; group g=(q_row/16)*8+q_row%8 (SM89_16x8 C-fragment layout).
  // m4n2 kBr=64 < 128-row quant block: odd tiles offset q_row by 64.
  float qs_arr[kSRows];
  if constexpr (kQKPerThread) {
    const int n_rb_q_quant = (Nq + 127) / 128;
    const int quant_offset = (q_start_row + Q_tile_id * kBr) % 128;
#pragma unroll
    for (int row = 0; row < kSRows; ++row) {
      const int q_row = get<0>(tScS_rc(row, 0)) + quant_offset;
      const int g = (q_row / 16) * 8 + q_row % 8;
      qs_arr[row] = q_scale[static_cast<long>(q_bh) * (n_rb_q_quant * 64) +
                            quant_rb * 64 + g];
    }
  } else {
    const float qs = q_scale[static_cast<long>(q_bh) * n_rb_q + Q_tile_id +
                             q_start_row / kBr];
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

  // O accumulators: D/4 fp32 regs per thread (m4n2 halves M8N1's D/2).
  float o_acc_storage[kDChunksV][kOElemsPerFrag];
#pragma unroll
  for (int v = 0; v < kDChunksV; ++v)
#pragma unroll
    for (int i = 0; i < kOElemsPerFrag; ++i)
      o_acc_storage[v][i] = 0.0f;

  const bool smooth_lse = (softmax_lse != nullptr) && (km != nullptr);
  float qkm[kORows];
#pragma unroll
  for (int r = 0; r < kORows; ++r)
    qkm[r] = 0.0f;

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
      const int chunk_index = v;
      const int v_stage = chunk_index % kStagesPV;
      const int v_phase = (chunk_index / kStagesPV) & 1;
      CtaBarrier::wait(&v_empty[v_stage], v_phase);
      issue_v_tma(v, v_stage, 0);
    }
  }

#pragma unroll 1
  for (int kv_tile = 0; kv_tile < Tc_eff; ++kv_tile) {
    if (kv_tile > 0)
      __syncthreads();
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
    // valid for both N-warps since groups repeat every 8 N_kv rows).
    const float ks =
        kQKPerThread ? k_scale[static_cast<long>(kv_bh) * (n_rb_kv * 4) +
                               kv_tile * 4 + (tid % 32) % 4]
                     : k_scale[static_cast<long>(kv_bh) * n_rb_kv + kv_tile];
    // Per-channel V: v_scale is (bh, D); skip the per-kv-tile slot (wrong
    // shape). vs=1 placeholder; epilogue dequants per-D instead.
    const float vs =
        kVPerChannel ? 1.0f
                     : v_scale[static_cast<long>(kv_bh) * n_rb_kv + kv_tile];
    // Per-block: vs*448, vs cancels in PV MMA -> unified 448 domain, single
    //   epilogue dequant (1/448). Per-channel: 1.0 so P8=softmax lands in
    //   e4m3's [0,1] range; amax_d global -> uniform per-D epilogue dequant.
    const float p_quant_scale = kVPerChannel ? 1.0f : (vs * kE4m3Max);

    // Phase 1: QK GEMM with split-D accumulation.
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

    // int8 QK: cast s32 acc to f32; fp8 path: identity view.
    auto tCrSf =
        make_tensor(reinterpret_cast<float*>(tCrS.data()), tCrS.layout());
    if constexpr (Traits::kQKInt8) {
#pragma unroll
      for (int i = 0; i < size(tCrS); ++i)
        tCrSf(i) = static_cast<float>(tCrS(i));
    }

    // Phase 2: fixed-P-scale softmax with cross-N-warp reduction.
    auto scores = make_tensor(
        tCrSf.data(), ffpa_cute::convert_layout_acc_rowcol(tCrS.layout()));
    const int kv_valid = Nkv - kv_tile * kBc;
    const bool tile_needs_mask =
        (kv_valid < kBc) || (kv_tile >= mask_start_tile);
    if (tile_needs_mask) {
#pragma unroll
      for (int row = 0; row < kSRows; ++row) {
        const int q_pos = q_start_row + Br_base + get<0>(tScS_rc(row, 0));
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
      ffpa_cute::online_softmax_fp8_fixed_m4n2<decltype(scores),
                                               decltype(tScS_rc), kORows>(
          scores, tScS_rc, softmax_scale_eff, row_max, row_sum, row_scale,
          smem_exchange, warp_id, lane_id, log2f(p_quant_scale),
          1.0f / p_quant_scale, Traits::kRescaleThreshold);
    } else {
      const float s_dequant = qs_arr[0] * ks;
      const float softmax_scale_eff =
          tile_needs_mask ? 1.0f : s_dequant * scale;
      ffpa_cute::online_softmax_fp8_fixed_m4n2<decltype(scores),
                                               decltype(tScS_rc), kORows>(
          scores, tScS_rc, softmax_scale_eff, row_max, row_sum, row_scale,
          smem_exchange, warp_id, lane_id, log2f(p_quant_scale),
          1.0f / p_quant_scale, Traits::kRescaleThreshold);
    }

    bool local_need_rescale = false;
#pragma unroll
    for (int r = 0; r < kORows; ++r)
      local_need_rescale = local_need_rescale || (row_scale[r] < 1.0f);
    const bool need_rescale = __any_sync(0xffffffff, local_need_rescale);

    // Phase 3: P -> e4m3 -> SMEM roundtrip. stmatrix is a b16 operation that
    // needs SW128 for 16B vectorization, but SW128's 128-elem atom doesn't
    // divide the 64-col P tile; DefaultCopy (vectorized stores) is used
    // instead. Single barrier structure matches fp16 m4n2.
    auto tCrP = ffpa_cute::convert_type<Element>(tCrSf);

    auto sP = make_tensor(make_smem_ptr(p_base), SmemLayoutP{});
    auto r2s_copy_p =
        make_tiled_copy_C(Copy_Atom<DefaultCopy, Element>{}, tiled_mma_qk);
    auto r2s_thr_p = r2s_copy_p.get_thread_slice(tid);
    auto tCrP_src = r2s_thr_p.retile_S(tCrP);
    auto tCsP_dst = r2s_thr_p.partition_D(sP);
    copy(r2s_copy_p, tCrP_src, tCsP_dst);
    cutlass::arch::fence_view_async_shared();
    __syncthreads();
    // finalize_row_sum folds peer tile sums (written by the m4n2 softmax
    // above) into row_sum, rescaled by row_scale.
    ffpa_cute::finalize_row_sum_m4n2<kORows>(row_sum, row_scale, smem_exchange,
                                             warp_id, lane_id);

    auto tCrPv_storage = thr_mma_pv.partition_fragment_A(sP);
    auto tPsP = s2r_thr_p.partition_S(sP);
    copy(s2r_copy_p, tPsP, tCrPv_storage);

    // Phase 4: PV GEMM with split-D accumulation over V^T chunks.
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
        // A-side P stays tCrPv_storage (SMEM roundtrip); only B-side (V)
        // derives from the f16 TiledMma.
        auto tCrV_f16 = thr_mma_pv_f16.partition_fragment_B(sV);
        auto tVsV_f16 = s2r_thr_v_f16.partition_S(sV);
        auto tCrInst = partition_fragment_C(tiled_mma_pv_f16,
                                            Shape<Int<kBr>, Int<kVDChunk>>{});
        clear(tCrInst);
        ffpa_cute::gemm_rs(tCrInst, tCrPv_storage, tCrV_f16, tVsV_f16,
                           tiled_mma_pv_f16, s2r_copy_v_f16, s2r_thr_v_f16);
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
        auto tCrVStorage = thr_mma_pv.partition_fragment_B(sV);
        auto tCrV = make_tensor(tCrVStorage.data(), tCrV_layout);
        auto tVsV = s2r_thr_v.partition_S(sV);
        ffpa_cute::gemm_rs(tCrO, tCrPv_storage, tCrV, tVsV, tiled_mma_pv,
                           s2r_copy_v, s2r_thr_v);
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

  // Phase 5: epilogue. O = o_acc * (1/448) / row_sum, batched R->S->TMA
  // store. Only n_warp==0 writes LSE (both N-warps share same Q rows).
  {
    constexpr int kVChunksPerBatch = Traits::kVChunksPerBatch;
    constexpr int kNBatches = Traits::kNBatches;
    constexpr int kOTileElems = cosize(SmemLayoutO{});

    __syncthreads();

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
          if (tid == 0)
            copy(tma_o, tOsO, tCgO_tma);
        }
        tma_store_arrive();
        if (batch < kNBatches - 1) {
          tma_store_wait<0>();
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

    // LSE: only n_warp==0 writes (both N-warps share same Q rows).
    if (softmax_lse != nullptr && n_warp == 0) {
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

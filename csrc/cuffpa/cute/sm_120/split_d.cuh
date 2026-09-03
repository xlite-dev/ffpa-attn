#pragma once

// tensor.hpp MUST precede any cute/atom/* header (see sm_80/split_d.cuh).
#include <cute/tensor.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/cutlass.h>
#include <cutlass/device_kernel.h>

// namespace ffpa_cute
#include "../gemm.cuh"
#include "../attn_traits.cuh"
#include "../attn_bias.cuh"
#include "../dropout.cuh"
#include "../softmax.cuh"

using TmaBarrier = cutlass::arch::ClusterTransactionBarrier;
using CtaBarrier = cutlass::arch::ClusterBarrier;

// kNhdQ / kNhdKV: Q (resp. K/V) arrive as an NHD (diffusers BNHD) permute
// view, read as flat (B*N, H*D) TMA rows with the head as a column tile
// (the (head * kHeadDim + d_chunk) second coord) and the batch via
// domain_offset. BHND keeps the per-head row-offset domain_offset + plain
// d_chunk column tile. O stays BHND-packed (caller allocates it packed).
template <typename Traits, typename TmaQ, typename TmaK, typename TmaV,
          typename TmaO, typename TmaBias, int kBiasMode = 0, int kBias4B = 0,
          int kHasAttnBias = 0, int kHasDropout = 0, bool kNhdQ = false,
          bool kNhdKV = false>
__global__ void __launch_bounds__(Traits::kNumThreads, 1)
    split_d_fwd_cute_sm120(
        CUTLASS_GRID_CONSTANT TmaQ const tma_q,
        CUTLASS_GRID_CONSTANT TmaK const tma_k,
        CUTLASS_GRID_CONSTANT TmaV const tma_v,
        CUTLASS_GRID_CONSTANT TmaO const tma_o,
        CUTLASS_GRID_CONSTANT TmaBias const tma_bias,
        typename Traits::Element* __restrict__ O,
        float* __restrict__ softmax_lse, int Nq, int Nkv, int Nh, int Nh_kv,
        float scale, int Tc, int causal, int total_q_rows, int total_kv_rows,
        const void* __restrict__ attn_bias = nullptr, int attn_bias_dtype = 0,
        long long attn_bias_stride_b = 0, long long attn_bias_stride_h = 0,
        long long attn_bias_stride_m = 0, long long attn_bias_stride_n = 0,
        float dropout_p = 0.0f, unsigned long long philox_seed = 0,
        unsigned long long philox_offset = 0, bool nhd_out = false,
        long long attn_bias_plane_m_total = 0, int dropout_bitmap_on = 0) {
  // Body-level arch guard: TMA/stmatrix need sm>=90, but in mixed -gencode
  // builds the sm_89 device pass still compiles this TU; the guard compiles
  // the body into a no-op stub there. Body-level (not file-level) is required
  // because the host launcher references this kernel via <<<>>> and nvcc must
  // see its declaration in every device pass; hiding it file-level fails with
  // "identifier undefined". Runtime safety: launch.cuh dispatches TMA kernels
  // only when prop->major >= 9, so pre-90 devices never execute the stub.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  // Split-D Flash Attention forward (non-WS, CuTe TMA).
  //
  // Algorithm per KV tile:
  //   1. QK GEMM: S[Br,Bc] += Q[Br,kQKDChunk] @ K[Bc,kQKDChunk]^T
  //      accumulated over kDChunksQK = kHeadDim/kQKDChunk split-D chunks.
  //   2. Online softmax: row-max, exp2, row-sum with rescale factor.
  //   3. PV GEMM: O[Br,kVDChunk] += P[Br,Bc] @ V[Bc,kVDChunk]
  //      accumulated over kDChunksV = kHeadDim/kVDChunk split-D chunks.
  //      O is rescaled by row_scale before each kv_tile > 0.
  //   4. Epilogue: O /= row_sum, convert to half, store to gmem.
  //
  // TMA pipeline: tid=0 issues TMA loads inline (non-WS). All threads
  // participate in MMA. Barriers: qk_full (TmaBarrier, init=1) signals
  // data ready; qk_empty (CtaBarrier, init=kNumThreads) signals stage
  // consumed. Phase tracking: chunk_index = kv_tile*kDChunks + d_chunk,
  // phase = (chunk_index / kStages) & 1.
  //
  // Layout transforms (all defined in gemm.cuh; see those headers for the
  // m16n8k16-fragment rationale and upstream references):
  //   convert_layout_acc_rowcol: MMA C-fragment → [rows, cols] for softmax
  //     row-max/exp/sum (each row's columns land in one thread so __shfl_xor
  //     reduces across the 4 lanes sharing a row).
  //   convert_layout_acc_Aregs:  MMA C-fragment → A-operand regs for PV
  //     (reuses P registers as MMA-A without writing back to smem).
  //   convert_type:              f32 acc → f16 P/O in-register, zero copy.
  //   gemm_ss / gemm_rs:         software-pipelined ldmatrix + mma.sync
  //     (gemm_rs only preloads B=V since A=P is already in regs).
  //   SmemLayoutVt: transposed V layout for gemm_rs B-operand (LDSM_T).
  // Why NOT WS? Please check ../fwd_sm120.cuh for more details.

  using namespace cute;
  using cute::tma_store_arrive;
  using cute::tma_store_wait;
  using Element = typename Traits::Element;
  using SmemLayoutQ = typename Traits::SmemLayoutQ;
  using SmemLayoutK = typename Traits::SmemLayoutK;
  using SmemLayoutV = typename Traits::SmemLayoutV;
  using SmemLayoutVt = typename Traits::SmemLayoutVt;
  using SmemLayoutO = typename Traits::SmemLayoutO;
  using TiledMmaQK = typename Traits::TiledMmaQK;
  using TiledMmaPV = typename Traits::TiledMmaPV;
  using SmemCopyAtom = typename Traits::SmemCopyAtom;
  using SmemCopyAtomTransposed = typename Traits::SmemCopyAtomTransposed;

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

  // TMA-O epilogue reuses v_base smem as the O staging buffer; guard that it
  // fits. The "no in-flight V TMA at epilogue entry" invariant holds for any
  // kDChunksV/kStagesPV: every v_chunk's V is consumed via TmaBarrier::wait
  // (v_full) inside the PV loop, so by loop exit all V loads are drained and
  // v_base is safe to overwrite after the epilogue's __syncthreads().
  static_assert(cosize(SmemLayoutO{}) <= kStagesPV * cosize(SmemLayoutV{}),
                "TMA-O: O staging buffer must fit in reused V-stage smem");

  const int Nb_id = blockIdx.y / Nh;
  const int Nh_id = blockIdx.y % Nh;
  const int Q_tile_id = blockIdx.x;
  const int group_size = Nh / Nh_kv;
  const int kv_head_idx = Nh_id / group_size;
  const int Br_base = Q_tile_id * kBr;
  const int tid = threadIdx.x;

  if (Br_base >= Nq)
    return;

  // Per-head global row origins for the BHND path (NHD folds the head into
  // the column tile, so it needs only the batch offset via domain_offset).
  // Using the true per-head row count (Nq / Nkv) rather than ceil(N/kBr)*kBr
  // keeps the TMA row coordinate correct when N % kBr != 0 (non-aligned).
  const int q_row_offset = (Nb_id * Nh + Nh_id) * Nq;
  const int kv_row_offset = (Nb_id * Nh_kv + kv_head_idx) * Nkv;

  const int kv_offset = Nkv - Nq;
  const int causal_thresh_row0 = Br_base + kv_offset;
  const int Tc_eff =
      causal ? min(Tc, ((Br_base + kBr - 1 + kv_offset) / kBc) + 1) : Tc;
  const int mask_start_tile =
      causal ? max(0, (causal_thresh_row0 + 1) / kBc) : INT_MAX;

  // NHD (diffusers BNHD): flat (B*N, H*D) TMA rows with the head as a
  // kHeadDim-wide column tile; batch rides domain_offset. BHND: per-head row
  // offset into a plain (B*H*N, D) row-major tensor.
  const long nb = gridDim.y / Nh;
  auto mQ = [&] {
    if constexpr (kNhdQ)
      return domain_offset(
          make_coord(static_cast<long>(Nb_id) * Nq, 0),
          tma_q.get_tma_tensor(make_shape(static_cast<long>(nb) * Nq,
                                          static_cast<long>(Nh) * kHeadDim)));
    else
      return domain_offset(
          make_coord(q_row_offset, 0),
          tma_q.get_tma_tensor(make_shape(total_q_rows, Int<kHeadDim>{})));
  }();
  auto mK = [&] {
    if constexpr (kNhdKV)
      return domain_offset(make_coord(static_cast<long>(Nb_id) * Nkv, 0),
                           tma_k.get_tma_tensor(make_shape(
                               static_cast<long>(nb) * Nkv,
                               static_cast<long>(Nh_kv) * kHeadDim)));
    else
      return domain_offset(
          make_coord(kv_row_offset, 0),
          tma_k.get_tma_tensor(make_shape(total_kv_rows, Int<kHeadDim>{})));
  }();
  auto mV = [&] {
    if constexpr (kNhdKV)
      return domain_offset(make_coord(static_cast<long>(Nb_id) * Nkv, 0),
                           tma_v.get_tma_tensor(make_shape(
                               static_cast<long>(nb) * Nkv,
                               static_cast<long>(Nh_kv) * kHeadDim)));
    else
      return domain_offset(
          make_coord(kv_row_offset, 0),
          tma_v.get_tma_tensor(make_shape(total_kv_rows, Int<kHeadDim>{})));
  }();
  // SMEM layout: [q_base | k_base | v_base], each with kStages copies.
  extern __shared__ __align__(1024) Element shm[];
  Element* q_base = shm;
  Element* k_base = q_base + kStagesQK * kQChunkElements;
  Element* v_base = k_base + kStagesQK * kKChunkElements;

  __shared__ uint64_t qk_full[kStagesQK];
  __shared__ uint64_t qk_empty[kStagesQK];
  __shared__ uint64_t v_full[kStagesPV];
  __shared__ uint64_t v_empty[kStagesPV];
  // Bias tile (PC-0): [kBr,kBc] (dense) or [1,kBc] (row-broadcast). Dense
  // runs single-buffered (halves the footprint so the tile fits the smem
  // budget next to the QK/V buffers; the reissue happens right after the
  // consumer's injection, see below). Row-broadcast keeps a 512B
  // double-buffer: the tiny tile would otherwise expose the full TMA latency
  // on every kv tile (NCU: long_scoreboard/sleeping dominate the bias gap).
  // Tile mode 0 keeps the gmem-direct FC-4 path.
  constexpr int kBiasStages = (kBiasMode == 2) ? 2 : 1;
  __shared__ uint64_t bias_full[kBiasStages];
  __shared__ uint64_t bias_empty[kBiasStages];
  // 16B-aligned past the QK/V buffers (mode 3's plain vector stores); the
  // launcher sizes the dynamic smem with the same rounding.
  uint16_t* bias_base = reinterpret_cast<uint16_t*>(
      reinterpret_cast<char*>(shm) +
      ((Traits::kSmemElems * sizeof(Element) + 15) & ~15));
  // PC-14 dropout keep-bitmap: [kBr, kBc] bits x2 stages past the bias
  // area (the launcher budgets both with the same layout). The 256
  // threads (2 per row) generate half-rows one kv tile ahead at the top
  // of each iteration — off the softmax->PV critical path — and apply
  // after softmax as register bit-tests; one __syncthreads per tile
  // orders apply-vs-regen of the ping-pong buffers (same protocol as
  // persist_d). The bitmap is dead before the R->S epilogue reuses shm.
  constexpr int kBitmapU32PerStage = kBr * kBc / 32;
  static_assert(!kHasDropout || (kBc % 64 == 0 && kNumThreads == 2 * kBr),
                "bitmap generation needs kBc%64==0 and 2 threads/row");
  const int bias_area_u16 =
      (kBiasMode == 3)
          ? (((int)Nkv * ((attn_bias_dtype == 3) ? 2 : 1)) + 7) & ~7
      : (kBiasMode == 2) ? kBiasStages * kBc * ((attn_bias_dtype == 3) ? 2 : 1)
      : (kBiasMode == 1) ? kBr * kBc * ((attn_bias_dtype == 3) ? 2 : 1)
                         : 0;
  uint32_t* bitmap_base =
      reinterpret_cast<uint32_t*>(bias_base + bias_area_u16);

  // Barrier roles:
  //   *_full  (TmaBarrier, init=1):   producer→consumer. The `1` is the single
  //     TMA-issuing thread (tid=0) that arrives via arrive_and_expect_tx(bytes)
  //     once its TMA writes land; consumers block on wait(*_full, phase).
  //   *_empty (CtaBarrier, init=kNumThreads): consumer→producer. The
  //     kNumThreads arrivals = every consumer thread has finished reading the
  //     stage; the next producer TMA blocks on wait(*_empty, phase) so it can
  //     safely overwrite that stage's smem.
  //   wait(bar, phase): phase is a 1-bit flip counter for ping-pong reuse of
  //     the SAME stage slot across passes; producer and consumer must pass
  //     matching phases so a stale arrival from pass N-1 cannot release
  //     pass N early. phase = (chunk_index / kStages) & 1 flips every kStages
  //     chunks (0,0,...,0 | 1,1,...,1 | 0,...).
  if (tid == 0) {
    for (int s = 0; s < kStagesQK; ++s) {
      TmaBarrier::init(&qk_full[s], 1);
      CtaBarrier::init(&qk_empty[s], kNumThreads);
    }
    for (int s = 0; s < kStagesPV; ++s) {
      TmaBarrier::init(&v_full[s], 1);
      CtaBarrier::init(&v_empty[s], kNumThreads);
    }
    if constexpr (kHasAttnBias) {
      for (int s = 0; s < kBiasStages; ++s) {
        TmaBarrier::init(&bias_full[s], 1);
        CtaBarrier::init(&bias_empty[s], kNumThreads);
      }
    }
  }
  __syncthreads();

  auto q_slice = tma_q.get_slice(_0{});
  auto k_slice = tma_k.get_slice(_0{});
  auto v_slice = tma_v.get_slice(_0{});

  // Dual TiledMma: QK uses Tile<kBr,kBc,16> (full S tile in one MMA),
  // PV uses Tile<kBr,kVDChunk,16> (output d-direction is N of MMA).
  TiledMmaQK tiled_mma_qk;
  TiledMmaPV tiled_mma_pv;
  auto thr_mma_qk = tiled_mma_qk.get_thread_slice(tid);
  auto thr_mma_pv = tiled_mma_pv.get_thread_slice(tid);

  // S2R copy atoms: LDSM_N for Q/K (A/B operands of QK GEMM),
  // LDSM_T for V (transposed B operand of PV GEMM via SmemLayoutVt).
  auto s2r_copy_q = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma_qk);
  auto s2r_copy_k = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma_qk);
  auto s2r_copy_v = make_tiled_copy_B(SmemCopyAtomTransposed{}, tiled_mma_pv);
  auto s2r_thr_q = s2r_copy_q.get_thread_slice(tid);
  auto s2r_thr_k = s2r_copy_k.get_thread_slice(tid);
  auto s2r_thr_v = s2r_copy_v.get_thread_slice(tid);

  // V fragment layout: precompute the register layout for PV B-operand
  // so we can reinterpret raw LDSM_T data without extra copies.
  // sVt0_ns uses get_nonswizzle_portion ONLY to derive which register slots
  // each thread holds (partition_fragment_B's thread↔data map); the register
  // layout is swizzle-independent. V's smem bank conflicts are handled by the
  // TMA write (SmemLayoutV has swizzle) + the ldmatrix read (partition_S on
  // the swizzled sVt inside the PV loop applies swizzle). Doing
  // partition_fragment_B on the swizzled sVt0 would conflict the LDSM_T
  // thread mapping with the swizzle composition.
  // Ref: flash-attention/csrc/flash_attn/src/kernel_traits.h
  //      SmemLayoutVtransposedNoSwizzle (same trick).
  auto sV0 = make_tensor(make_smem_ptr(v_base), SmemLayoutV{});
  auto sVt0_ns =
      make_tensor(sV0.data(), get_nonswizzle_portion(SmemLayoutVt{}));
  auto tCrV_layout = thr_mma_pv.partition_fragment_B(sVt0_ns).layout();

  // OFragType/OFragLayout are compile-time aliases over partition_fragment_C
  // used ONLY to size the o_acc_storage scratch (kOElemsPerFrag) and to derive
  // kORows/kOCols for the rowcol reshape; the runtime O fragment is rebuilt
  // fresh each iteration from o_acc_storage[v_chunk] (see the PV loop).
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

  // Online softmax below uses exp2f, which requires the scale in log2 domain:
  // exp(x) == exp2(x * log2(e)). The caller passes the linear-domain scale
  // (1/sqrt(D)); convert it once here so exp2f(scores*scale - max) is correct.
  // (This was the accuracy bug: without log2(e) the P and row_scale were
  // 2^(...) instead of e^(...), compounding across kv_tiles.)
  const float inv_scale = 1.0f / scale;
  scale *= FFPA_M_LOG2E;

  float row_max[kORows];
  float row_sum[kORows];
#pragma unroll
  for (int r = 0; r < kORows; ++r) {
    row_max[r] = -INFINITY;
    row_sum[r] = 0.0f;
  }

  // Persistent O accumulators across the whole KV-tile loop -- the root
  // cause of register pressure and the hardest part of large-D FA.
  // PV reduces over Bc (P[Br,Bc] @ V[Bc,D]); split-D tiles the D axis into
  // kDChunksV disjoint [Bc,kVDChunk] slices, so each v_chunk's O slice must
  // stay live across ALL kv_tiles for online partial rescaling (tCrO *=
  // row_scale each kv_tile, see PV loop): FA2's O_i = diag(e^{m_{i-1}-m_i})
  // O_{i-1} + P_i V_i. Equivalent to R_D in the sm80 large-d kernel (sm80
  // splits a transient R_O from the persistent R_D; here gemm_rs writes the
  // MMA-C operand directly into o_acc_storage, no transient acc).
  // Unlike SMEM -- which large-D FA keeps O(1) by tiling D into chunks --
  // this register footprint is O(D/kVDChunk) per-thread and is a structural
  // cost of single-pass online softmax: rescale needs all kDChunksV slices
  // resident every kv_tile, so none can be streamed out early. Only raising
  // kVDChunk (fewer slices, SMEM/MMA limited) or a two-pass algorithm
  // (finalize m before PV -> no rescale -> serial v_chunk, but QK redone
  // kDChunksVx) can lower it.
  // Register budget: kOElemsPerFrag = kBr*kVDChunk/kNumThreads = 128*64/256 =
  // 32 fp32/thread, so o_acc_storage = kDChunksV*32 = (D/kVDChunk)*32 regs.
  // D=512/kVDChunk=64 -> 256 regs, saturating the 255-reg/thread ceiling (the
  // rest of the kernel -- QK acc, P, V frags, softmax state -- spills to
  // local mem); D>512 -> o_acc_storage alone >256, heavy spill. This is the
  // hard ceiling on head-D for single-pass split-D FA.
  float o_acc_storage[kDChunksV][kOElemsPerFrag];
#pragma unroll
  for (int v = 0; v < kDChunksV; ++v)
#pragma unroll
    for (int i = 0; i < kOElemsPerFrag; ++i)
      o_acc_storage[v][i] = 0.0f;

  // Signal all stages empty so tid=0 can issue initial TMA prefetch.
  for (int s = 0; s < kStagesQK; ++s)
    CtaBarrier::arrive(&qk_empty[s]);
  for (int s = 0; s < kStagesPV; ++s)
    CtaBarrier::arrive(&v_empty[s]);
  if constexpr (kHasAttnBias) {
    for (int s = 0; s < kBiasStages; ++s)
      CtaBarrier::arrive(&bias_empty[s]);
  }

  // Bias tile TMA (PC-0): u16 plane. Dense folds (b,h) into the linear row
  // domain; row-broadcast reads the [m_total,Nkv] plane ((b,h) folds to one
  // row, host-validated) with a static 1-row box.
  // The TMA box must be fully static (vectorization inference rejects
  // dynamic modes) and must match the host descriptor, so the mode is a
  // template parameter.
  auto b_slice = tma_bias.get_slice(_0{});
  constexpr int bias_cols = kBc * (kBias4B ? 2 : 1);
  auto mBias = [&] {
    if constexpr (kBiasMode == 1)
      // TMA-tensor coords are in rows; the (b,h) strides are element
      // counts, so divide by stride_m (exact: plan validated stride_h
      // == Nq*stride_m, stride_b == h_eff*Nq*stride_m).
      return domain_offset(
          make_coord(((long long)Nb_id * attn_bias_stride_b +
                      (long long)Nh_id * attn_bias_stride_h) /
                             attn_bias_stride_m +
                         (long long)Q_tile_id * kBr,
                     0LL),
          tma_bias.get_tma_tensor(make_shape(
              attn_bias_plane_m_total, (long long)Nkv * bias_cols / kBc)));
    else
      // Row-broadcast rows are Nkv elements wide, so the folded (b,h)
      // element offset divides exactly (stride_h==Nkv, stride_b==
      // h_eff*Nkv, host-validated).
      return domain_offset(
          make_coord(((long long)Nb_id * attn_bias_stride_b +
                      (long long)Nh_id * attn_bias_stride_h) /
                         (long long)Nkv,
                     0LL),
          tma_bias.get_tma_tensor(make_shape(
              attn_bias_plane_m_total, (long long)Nkv * bias_cols / kBc)));
  }();
  auto issue_bias_tma = [&](int tile) {
    cutlass::arch::fence_view_async_shared();
    const int stage = tile % kBiasStages;
    const int phase = (tile / kBiasStages) & 1;
    CtaBarrier::wait(&bias_empty[stage], phase);
    if constexpr (kBiasMode == 1) {
      auto sB = make_tensor(make_smem_ptr(bias_base + stage * kBr * bias_cols),
                            Layout<Shape<Int<kBr>, Int<bias_cols>>,
                                   Stride<Int<bias_cols>, _1>>{});
      auto gB = local_tile(mBias, Shape<Int<kBr>, Int<bias_cols>>{},
                           make_coord(_0{}, tile));
      TmaBarrier::arrive_and_expect_tx(&bias_full[stage],
                                       sizeof(uint16_t) * kBr * bias_cols);
      copy(tma_bias.with(bias_full[stage]), b_slice.partition_S(gB),
           b_slice.partition_D(sB));
    } else {
      auto sB = make_tensor(
          make_smem_ptr(bias_base + stage * bias_cols),
          Layout<Shape<_1, Int<bias_cols>>, Stride<Int<bias_cols>, _1>>{});
      auto gB = local_tile(mBias, Shape<_1, Int<bias_cols>>{},
                           make_coord(_0{}, tile));
      TmaBarrier::arrive_and_expect_tx(&bias_full[stage],
                                       sizeof(uint16_t) * bias_cols);
      copy(tma_bias.with(bias_full[stage]), b_slice.partition_S(gB),
           b_slice.partition_D(sB));
    }
  };

  // TMA load helpers: tid=0 issues Q+K (or V) TMA copies with
  // arrive_and_expect_tx on the full barrier for the target stage.
  auto issue_qk_tma = [&](int d_chunk, int stage, int kv_tile_idx) {
    cutlass::arch::fence_view_async_shared();
    auto sQ = make_tensor(make_smem_ptr(q_base + stage * kQChunkElements),
                          SmemLayoutQ{});
    auto sK = make_tensor(make_smem_ptr(k_base + stage * kKChunkElements),
                          SmemLayoutK{});
    // NHD: head rides the column tile (head * kHeadDim + d_chunk); BHND: the
    // d_chunk column tile of a per-head row offset tensor.
    auto gQ = [&] {
      if constexpr (kNhdQ)
        return local_tile(mQ, Shape<Int<kBr>, Int<kQKDChunk>>{},
                          make_coord(Q_tile_id, Nh_id * kDChunksQK + d_chunk));
      else
        return local_tile(mQ, Shape<Int<kBr>, Int<kQKDChunk>>{},
                          make_coord(Q_tile_id, d_chunk));
    }();
    auto gK = [&] {
      if constexpr (kNhdKV)
        return local_tile(
            mK, Shape<Int<kBc>, Int<kQKDChunk>>{},
            make_coord(kv_tile_idx, kv_head_idx * kDChunksQK + d_chunk));
      else
        return local_tile(mK, Shape<Int<kBc>, Int<kQKDChunk>>{},
                          make_coord(kv_tile_idx, d_chunk));
    }();
    auto tQgQ = q_slice.partition_S(gQ);
    auto tQsQ = q_slice.partition_D(sQ);
    auto tKgK = k_slice.partition_S(gK);
    auto tKsK = k_slice.partition_D(sK);
    TmaBarrier::arrive_and_expect_tx(&qk_full[stage],
                                     sizeof(Element) * (size(sQ) + size(sK)));
    copy(tma_q.with(qk_full[stage]), tQgQ, tQsQ);
    copy(tma_k.with(qk_full[stage]), tKgK, tKsK);
  };

  auto issue_v_tma = [&](int v_chunk, int stage, int kv_tile_idx) {
    cutlass::arch::fence_view_async_shared();
    auto sV = make_tensor(make_smem_ptr(v_base + stage * kVChunkElements),
                          SmemLayoutV{});
    auto gV = [&] {
      if constexpr (kNhdKV)
        return local_tile(
            mV, Shape<Int<kBc>, Int<kVDChunk>>{},
            make_coord(kv_tile_idx, kv_head_idx * kDChunksV + v_chunk));
      else
        return local_tile(mV, Shape<Int<kBc>, Int<kVDChunk>>{},
                          make_coord(kv_tile_idx, v_chunk));
    }();
    auto tVgV = v_slice.partition_S(gV);
    auto tVsV = v_slice.partition_D(sV);
    TmaBarrier::arrive_and_expect_tx(&v_full[stage],
                                     sizeof(Element) * size(sV));
    copy(tma_v.with(v_full[stage]), tVgV, tVsV);
  };

  // Initial QK prefetch: fill pipeline with first kStagesQK chunks.
  if (tid == 0) {
    for (int d = 0; d < kStagesQK && d < kDChunksQK; ++d) {
      CtaBarrier::wait(&qk_empty[d], 0);
      issue_qk_tma(d, d, 0);
    }
  }

  // Initial V prefetch for kv_tile 0: issue first kStagesPV V chunks so the
  // V TMA overlaps the entire first QK GEMM + softmax window (V is independent
  // of QK, so it can be launched before the QK loop).
  // v_stage = chunk_index % kStagesPV (the smem slot), v_phase flips 0→1→0
  //   every kStagesPV chunks PER slot so a slot's pass N arrival can't be
  //   mistaken for its pass N-1 arrival. Example kStagesPV=2, kDChunksV=2:
  //   chunk 0→slot0 phase0, 1→slot1 phase0, 2→slot0 phase1, 3→slot1 phase1,
  //   4→slot0 phase0, ... (phase flips 0,0,1,1,0,0 across chunk_index).
  if (tid == 0) {
    for (int v = 0; v < kStagesPV && v < kDChunksV; ++v) {
      const int chunk_index = v;  // kv_tile == 0
      const int v_stage = chunk_index % kStagesPV;
      const int v_phase = (chunk_index / kStagesPV) & 1;
      CtaBarrier::wait(&v_empty[v_stage], v_phase);
      issue_v_tma(v, v_stage, 0);
    }
  }

  // Bias tile(0) prefetch (depth-1 ahead, like the QK pipeline). Mode 3
  // instead loads the resident [1,Nkv] row-broadcast vector once (plain
  // vector loads, host-guaranteed 16B alignment): no TMA and no per-tile
  // bias barrier anywhere in the kv loop.
  if constexpr (kHasAttnBias && kBiasMode == 3) {
    // The resident vector is this (b,h)'s row: rows are Nkv elements wide
    // (stride_h==Nkv, stride_b==h_eff*Nkv, host-validated).
    const uint16_t* src = reinterpret_cast<const uint16_t*>(attn_bias) +
                          ((long long)Nb_id * attn_bias_stride_b +
                           (long long)Nh_id * attn_bias_stride_h) *
                              ((attn_bias_dtype == 3) ? 2 : 1);
    const int n_u16 = (int)Nkv * ((attn_bias_dtype == 3) ? 2 : 1);
    const int vec_end = n_u16 & ~7;
    for (int i = tid * 8; i < vec_end; i += kNumThreads * 8)
      *reinterpret_cast<uint4*>(bias_base + i) =
          *reinterpret_cast<const uint4*>(src + i);
    for (int i = vec_end + tid; i < n_u16; i += kNumThreads)
      bias_base[i] = src[i];
    __syncthreads();
  } else if constexpr (kHasAttnBias && kBiasMode != 0) {
    if (tid == 0 && Tc_eff > 0)
      issue_bias_tma(0);
  }

  // PC-14 dropout bitmap: stage(0) into buffer 0, then the per-tile
  // generate-ahead protocol (see the kv loop).
  const bool bitmap_on = kHasDropout && dropout_bitmap_on != 0;
  const unsigned long long dropout_head_base =
      (static_cast<unsigned long long>(Nb_id) * Nh + Nh_id) * Nq;
  if (bitmap_on && Tc_eff > 0) {
    ffpa_cute::generate_dropout_bitmap_halfrow<kBc>(
        bitmap_base, tid >> 1, tid & 1, Br_base + (tid >> 1), 0, dropout_p,
        philox_seed, philox_offset, dropout_head_base, Nkv);
    __syncthreads();
  }

#pragma unroll 1
  for (int kv_tile = 0; kv_tile < Tc_eff; ++kv_tile) {
    // V prefetch for kv_tile > 0: issue first kStagesPV V chunks before the
    // QK loop so the V TMA overlaps QK GEMM + softmax. (kv_tile 0's V initial
    // was issued before the loop; subsequent kv_tiles' QK initial is issued at
    // the end of the previous kv_tile's QK loop, see below.)
    if (kv_tile > 0 && tid == 0) {
      for (int v = 0; v < kStagesPV && v < kDChunksV; ++v) {
        const int chunk_index = kv_tile * kDChunksV + v;
        const int v_stage = chunk_index % kStagesPV;
        const int v_phase = (chunk_index / kStagesPV) & 1;
        CtaBarrier::wait(&v_empty[v_stage], v_phase);
        issue_v_tma(v, v_stage, kv_tile);
      }
    }

    // 2-stage bias prefetch: issue (t+1) before this tile's QK/softmax so the
    // TMA hides behind them; empty-wait(t+1) needs only the previous tile's
    // injection arrive, which finished last iteration (no self-deadlock).
    if constexpr (kHasAttnBias && kBiasMode != 0) {
      if (kBiasStages == 2 && kv_tile + 1 < Tc_eff && tid == 0)
        issue_bias_tma(kv_tile + 1);
    }

    // Bitmap for the next tile: after tid0's issue points, before the QK
    // TMA wait — fills the wait window instead of the critical path.
    if (bitmap_on && kv_tile + 1 < Tc_eff)
      ffpa_cute::generate_dropout_bitmap_halfrow<kBc>(
          bitmap_base + ((kv_tile + 1) & 1) * kBitmapU32PerStage, tid >> 1,
          tid & 1, Br_base + (tid >> 1), kv_tile + 1, dropout_p, philox_seed,
          philox_offset, dropout_head_base, Nkv);

    // Phase 1: QK GEMM with split-D accumulation.
    // S[Br,Bc] = sum_{d=0}^{kDChunksQK-1} Q_d @ K_d^T
    auto tCrS = partition_fragment_C(tiled_mma_qk, Shape<Int<kBr>, Int<kBc>>{});
    clear(tCrS);

#pragma unroll
    for (int d_chunk = 0; d_chunk < kDChunksQK; ++d_chunk) {
      // Wait for TMA data, fence, then gemm_ss (smem→regs→MMA).
      // TmaBarrier::wait(qk_full[stage], phase): consumers block until tid=0's
      // arrive_and_expect_tx for this stage's Q+K TMA lands.
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

      // Signal stage consumed; tid=0 prefetches next chunk if available.
      // CtaBarrier::arrive(qk_empty[stage]): each consumer thread arrives once
      // it has finished reading stage's smem; once all kNumThreads arrive, the
      // producer's wait(qk_empty[stage], phase_next) unblocks to overwrite it.
      CtaBarrier::arrive(&qk_empty[stage]);

      // Cannot move this prefetch before gemm_ss: s_next == stage and
      // phase_next == 1-phase, so the wait below gates on THIS iter's
      // arrive(qk_empty[stage]) — moving earlier deadlocks (tid=0 would
      // stall waiting for an arrive that needs tid=0 inside gemm_ss).
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

    // Prefetch next kv_tile's QK initial chunks so the QK TMA overlaps this
    // kv_tile's softmax + PV loop. The QK barriers are disjoint from the
    // softmax/PV barriers, so placing this here is safe (zero-deadlock by the
    // disjoint-barrier-set invariant). kv_tile 0's QK initial was issued
    // before the loop; this replaces the old "kv_tile > 0 top" QK prefetch.
    if (kv_tile < Tc_eff - 1 && tid == 0) {
      for (int d = 0; d < kStagesQK && d < kDChunksQK; ++d) {
        const int chunk_index = (kv_tile + 1) * kDChunksQK + d;
        const int stage = chunk_index % kStagesQK;
        const int phase = (chunk_index / kStagesQK) & 1;
        CtaBarrier::wait(&qk_empty[stage], phase);
        issue_qk_tma(d, stage, kv_tile + 1);
      }
    }

    // Phase 2: Online softmax.
    // Layout transform: MMA C-fragment → [kORows, kSCols] rowcol view.
    {
      auto scores = make_tensor(
          tCrS.data(), ffpa_cute::convert_layout_acc_rowcol(tCrS.layout()));
      float row_scale[kORows];

      // Boundary masking: -inf for OOB KV positions.
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

      // Causal masking: -inf where k_pos > q_pos.
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

      // Additive attention bias (pre-softmax, separate pass).
      // NOTE: attn_bias/dropout on CuTe kernel is ~3x slower than the non-WS
      // TMA (../fwd_sm120.cuh) template kernel due to 1 block/SM occupancy (8
      // warps cannot hide scalar gmem load / Philox RNG latency). The launcher
      // should prefer the non-WS TMA fallback when bias/dropout is active;
      // these constexpr paths exist for correctness and future optimization
      // (e.g. vectorized bias load via TMA).
      if constexpr (kHasAttnBias && kBiasMode != 0) {
        const int b_stg = kv_tile % kBiasStages;
        const int b_phase = (kv_tile / kBiasStages) & 1;
        if constexpr (kBiasMode != 3) {
          TmaBarrier::wait(&bias_full[b_stg], b_phase);
          cutlass::arch::fence_view_async_shared();
        }
        const int b_slot_u16 = ((kBiasMode == 1) ? kBr * kBc : kBc) *
                               ((attn_bias_dtype == 3) ? 2 : 1);
        // mode 3: the resident vector's tile-t segment sits at t*kBc.
        const uint16_t* b_slot =
            bias_base + (kBiasMode == 3 ? (long long)kv_tile * kBc *
                                              ((attn_bias_dtype == 3) ? 2 : 1)
                                        : (long long)b_stg * b_slot_u16);
        constexpr int s_row = (kBiasMode == 1) ? kBc : 0;
        if (attn_bias_dtype == 3)
          ffpa_cute::apply_attn_bias_rowcol_smem<
              float, decltype(scores), decltype(tScS_rc), kSRows, kSCols>(
              scores, tScS_rc, reinterpret_cast<const float*>(b_slot), s_row, 1,
              inv_scale);
        else if (attn_bias_dtype == 2)
          ffpa_cute::apply_attn_bias_rowcol_smem<
              cutlass::bfloat16_t, decltype(scores), decltype(tScS_rc), kSRows,
              kSCols>(scores, tScS_rc,
                      reinterpret_cast<const cutlass::bfloat16_t*>(b_slot),
                      s_row, 1, inv_scale);
        else
          ffpa_cute::apply_attn_bias_rowcol_smem<
              cutlass::half_t, decltype(scores), decltype(tScS_rc), kSRows,
              kSCols>(scores, tScS_rc,
                      reinterpret_cast<const cutlass::half_t*>(b_slot), s_row,
                      1, inv_scale);
        if constexpr (kBiasMode != 3) {
          CtaBarrier::arrive(&bias_empty[b_stg]);
          // 1-stage slot: reissue (t+1) only after the injection of t has
          // released the buffer. tid 0 joins the injection, so waiting here
          // (instead of in the prefetch block) is required to avoid a
          // self-deadlock; the TMA then overlaps this tile's softmax + PV.
          if (kBiasStages == 1 && kv_tile + 1 < Tc_eff && tid == 0)
            issue_bias_tma(kv_tile + 1);
        }
      } else if constexpr (kHasAttnBias) {
        ffpa_cute::apply_attn_bias_rowcol<decltype(scores), decltype(tScS_rc),
                                          kSRows, kSCols>(
            scores, tScS_rc, attn_bias, attn_bias_dtype, attn_bias_stride_b,
            attn_bias_stride_h, attn_bias_stride_m, attn_bias_stride_n, Nb_id,
            Nh_id, Br_base, kv_tile, kBc, inv_scale);
      }

      // Row-max + exp2 + row-sum (warp-level reduction via shfl_xor).
      ffpa_cute::online_safe_softmax<decltype(scores), decltype(tScS_rc),
                                     kORows>(scores, tScS_rc, scale, row_max,
                                             row_sum, row_scale,
                                             Traits::kRescaleThreshold);

      // FA-4 conditional rescaling: warp-uniform vote so the O-rescale loop
      // below is skipped without divergence when every row's scale stayed 1.0.
      bool local_need_rescale = false;
#pragma unroll
      for (int r = 0; r < kORows; ++r)  // exp(<0) -> scale < 1.0
        local_need_rescale = local_need_rescale || (row_scale[r] < 1.0f);
      const bool need_rescale = __any_sync(0xffffffff, local_need_rescale);

      // Dropout on P (post-softmax, pre-PV, separate pass).
      if constexpr (kHasDropout) {
        if (dropout_bitmap_on) {
          ffpa_cute::apply_dropout_bitmap_rowcol<
              decltype(scores), decltype(tScS_rc), kSRows, kSCols, kBc>(
              scores, tScS_rc, bitmap_base + (kv_tile & 1) * kBitmapU32PerStage,
              1.0f / (1.0f - dropout_p));
          // Orders this tile's bitmap reads against the next iteration's
          // regen of the same (ping-pong) buffer by any other thread.
          __syncthreads();
        } else {
          ffpa_cute::apply_dropout_rowcol<decltype(scores), decltype(tScS_rc),
                                          kORows, kSCols>(
              scores, tScS_rc, dropout_p, philox_seed, philox_offset, Nb_id, Nh,
              Nh_id, Nq, Nkv, Br_base, kv_tile, kBc);
        }
      }

      // P fragment: convert fp32 scores → Element, then reinterpret
      // C-layout as A-operand registers for PV GEMM (zero-copy reuse).
      auto tCrP = ffpa_cute::convert_type<Element>(tCrS);
      auto tCrPv = make_tensor(
          tCrP.data(),
          ffpa_cute::convert_layout_acc_Aregs<TiledMmaPV>(tCrP.layout()));

      // (V initial prefetch moved to the top of the kv_tile loop so it overlaps
      // QK GEMM + softmax; see the kv_tile > 0 block and the pre-loop block.)

#pragma unroll
      for (int v_chunk = 0; v_chunk < kDChunksV; ++v_chunk) {
        // Wait for V TMA data, fence, then prepare V smem view.
        const int chunk_index = kv_tile * kDChunksV + v_chunk;
        const int v_stage = chunk_index % kStagesPV;
        const int v_phase = (chunk_index / kStagesPV) & 1;
        TmaBarrier::wait(&v_full[v_stage], v_phase);
        cutlass::arch::fence_view_async_shared();

        auto sV = make_tensor(make_smem_ptr(v_base + v_stage * kVChunkElements),
                              SmemLayoutV{});
        auto sVt = make_tensor(sV.data(), SmemLayoutVt{});
        auto tCrVStorage = thr_mma_pv.partition_fragment_B(sV);
        auto tCrV = make_tensor(tCrVStorage.data(), tCrV_layout);
        auto tVsVt = s2r_thr_v.partition_S(sVt);

        // O rescaling: multiply accumulated O by row_scale (kv_tile > 0).
        auto tCrO = make_tensor(make_rmem_ptr(&o_acc_storage[v_chunk][0]),
                                OFragLayout{});
        // Partial online rescaling for current v_chunk in this kv_tile.
        if (kv_tile > 0 && need_rescale) {
          auto tCrO_rc = make_tensor(
              tCrO.data(), ffpa_cute::convert_layout_acc_rowcol(tCrO.layout()));
#pragma unroll
          for (int row = 0; row < kORows; ++row)
#pragma unroll
            for (int col = 0; col < kOCols; ++col)
              tCrO_rc(row, col) *= row_scale[row];
        }

        // gemm_rs: P (register A) @ V (smem B via LDSM_T) → O (register C).
        ffpa_cute::gemm_rs(tCrO, tCrPv, tCrV, tVsVt, tiled_mma_pv, s2r_copy_v,
                           s2r_thr_v);

        // Signal stage consumed; tid=0 prefetches next chunk if available.
        CtaBarrier::arrive(&v_empty[v_stage]);

        // Cannot move this prefetch before gemm_rs: s_next == v_stage and
        // phase_next == 1-v_phase, so the wait below gates on THIS iter's
        // arrive(v_empty[v_stage]) — moving earlier deadlocks (tid=0 would
        // stall waiting for an arrive that needs tid=0 inside gemm_rs).
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
  }

  // Phase 4: Epilogue. Normalize O by 1/row_sum, convert to Element, store.
  //   aligned tile: batched R->S(stmatrix)->swizzled smem->TMA store.
  //   kVChunksPerBatch
  //     v_chunks staged in shm (reusing freed QKV smem), TMA stores batched
  //     into one bulk group (one arrive + one wait per batch), reducing wait
  //     count from kDChunksV to kNBatches. LSE write deferred to overlap last
  //     drain.
  //   tail tile: per-element predicated R->G (unchanged, zero risk).
  //   TMA-store drain race (fixed): only tid=0 issues the store, so
  //     tma_store_wait<0>() is a no-op for every other thread. Without a CTA
  //     barrier the next batch's R->S would overwrite shm the in-flight TMA
  //     store is still reading -> deterministic O corruption whenever
  //     kNBatches >= 2 (D=320 stages=2 -> kNBatches=5 fails; stages=3 ->
  //     kNBatches=1 passes, which is why stages=3 looked correct). The
  //     __syncthreads() after tma_store_wait below gates all threads on the
  //     drain; the batch condition is CTA-uniform so it cannot deadlock.
  {
    constexpr int kVChunksPerBatch = Traits::kVChunksPerBatch;
    constexpr int kNBatches = Traits::kNBatches;
    constexpr int kOTileElems = cosize(SmemLayoutO{});

    __syncthreads();  // V smem reads done before R->S overwrites shm

    // NHD (diffusers BNHD packed) O: rows interleave heads (row stride
    // Nh*kHeadDim); the nhd_out branch only picks coordinates, the batched
    // R->S->TMA copy path is shared. Column tiles fold the head in (the
    // v_chunk walk stays chunk-local), mirroring the NHD Q load.
    const int nb = total_q_rows / (Nh * Nq);
    const int o_row_base = nhd_out ? (Nb_id * Nq) : q_row_offset;
    const int o_rows = nhd_out ? (nb * Nq) : total_q_rows;
    const int o_cols = nhd_out ? (Nh * kHeadDim) : kHeadDim;
    const int o_col_tile = nhd_out ? (Nh_id * kDChunksV) : 0;
    auto mO_tma =
        domain_offset(make_coord(o_row_base, 0),
                      tma_o.get_tma_tensor(make_shape(o_rows, o_cols)));
    auto o_slice = tma_o.get_slice(_0{});

    auto r2s_copy = make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N, Element>{},
                                      tiled_mma_pv);
    auto r2s_thr = r2s_copy.get_slice(tid);

    const int O_gmem_offset =
        nhd_out ? (Nb_id * Nq * Nh + Nh_id) * kHeadDim
                : (Nb_id * Nh * Nq * kHeadDim) + (Nh_id * Nq * kHeadDim);
    const int o_row_stride = nhd_out ? Nh * kHeadDim : kHeadDim;
    auto mO = make_tensor(make_gmem_ptr(O + O_gmem_offset),
                          make_shape(Nq, Int<kHeadDim>{}),
                          make_stride(o_row_stride, _1{}));
    auto cO = make_identity_tensor(Shape<Int<kBr>, Int<kVDChunk>>{});
    auto tOcO = thr_mma_pv.partition_C(cO);

    if (Br_base + kBr <= Nq) {
      // aligned: batched R->S->G via TMA store
#pragma unroll
      for (int batch = 0; batch < kNBatches; ++batch) {
        // R->S: stage kVChunksPerBatch v_chunks into disjoint shm regions
#pragma unroll
        for (int v_in = 0; v_in < kVChunksPerBatch; ++v_in) {
          int v_chunk = batch * kVChunksPerBatch + v_in;
          auto tCrO = make_tensor(make_rmem_ptr(&o_acc_storage[v_chunk][0]),
                                  OFragLayout{});
          auto tCrO_rc = make_tensor(
              tCrO.data(), ffpa_cute::convert_layout_acc_rowcol(tCrO.layout()));
          // Final O rescaling: multiply accumulated O by 1/row_sum (last
          // kv_tile).
#pragma unroll
          for (int row = 0; row < kORows; ++row) {
            const float inv_sum = 1.0f / row_sum[row];
#pragma unroll
            for (int col = 0; col < kOCols; ++col)
              tCrO_rc(row, col) *= inv_sum;
          }
          auto tCrOHalf = ffpa_cute::convert_type<Element>(tCrO);
          auto sO_v = make_tensor(make_smem_ptr(shm + v_in * kOTileElems),
                                  SmemLayoutO{});
          auto tCrOHalf_src = r2s_thr.retile_S(tCrOHalf);
          auto tCsO_dst = r2s_thr.partition_D(sO_v);
          copy(r2s_copy, tCrOHalf_src, tCsO_dst);
        }
        cutlass::arch::fence_view_async_shared();
        __syncthreads();
        // TMA stores: issue all v_chunks in this batch into one bulk group
#pragma unroll
        for (int v_in = 0; v_in < kVChunksPerBatch; ++v_in) {
          int v_chunk = batch * kVChunksPerBatch + v_in;
          auto sO_v = make_tensor(make_smem_ptr(shm + v_in * kOTileElems),
                                  SmemLayoutO{});
          auto gO_tma = local_tile(mO_tma, Shape<Int<kBr>, Int<kVDChunk>>{},
                                   make_coord(Q_tile_id, o_col_tile + v_chunk));
          auto tCgO_tma = o_slice.partition_D(gO_tma);
          auto tOsO = o_slice.partition_S(sO_v);
          if (tid == 0) {
            copy(tma_o, tOsO, tCgO_tma);
          }
        }
        tma_store_arrive();
        if (batch < kNBatches - 1) {
          tma_store_wait<0>();  // drain for shm reuse
          __syncthreads();  // all threads wait (tma_store_wait is tid=0-only)
        }
      }
    } else {
      // tail: per-element predicated R->G (unchanged)
#pragma unroll
      for (int v_chunk = 0; v_chunk < kDChunksV; ++v_chunk) {
        auto tCrO = make_tensor(make_rmem_ptr(&o_acc_storage[v_chunk][0]),
                                OFragLayout{});
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
        auto gO = local_tile(mO, Shape<Int<kBr>, Int<kVDChunk>>{},
                             make_coord(Q_tile_id, v_chunk));
        auto tCgO = thr_mma_pv.partition_C(gO);
#pragma unroll
        for (int i = 0; i < size(tCrOHalf); ++i) {
          const int global_row = Br_base + get<0>(tOcO(i));
          if (global_row < Nq)
            tCgO(i) = tCrOHalf(i);
        }
      }
    }

    // LSE write: overlaps last batch's TMA drain (aligned) or serial (tail).
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

    // Final drain: only if TMA stores were issued (aligned path).
    if (Br_base + kBr <= Nq)
      tma_store_wait<0>();
  }
#endif  // defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
}

// Fused delta_s preprocess for the NVFP4 persist_d path:
//   delta_s[b,h,m,n] = qm[b,h,m,:] @ (K[b,hkv,n,:] - km)
//                    = qm @ K^T - qkm   with qkm[b,h,m] = qm . km
// (the kernel's S = Qhat@Khat^T + delta_s then equals q(k-km)^T, so the
// lse correction needs exactly scale * dot(q_row, km); any extra row
// constant here would shift the lse without changing O). One wmma tile
// kernel replaces the host-side bmm + transpose copy + epilogue cast
// (~1 ms at N=16384).
// Tail columns n >= Nkv are zero-filled (masked -inf in the attn kernel).
// Plain wmma on fp16 operands: the kernel is memory-bound (SM ~10%), so
// the block-scaled NVFP4 CuTe atoms of the attention kernel buy nothing
// here. If MMA ever becomes the limit, a CuTe port (LDSM/swizzled smem,
// tiled HMMA) is the upgrade path.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

// tensor.hpp MUST precede any cute/atom/* header (see sm_120/split_d.cuh).
#include <cute/tensor.hpp>
#include <cute/atom/copy_traits_sm75.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cutlass.h>
#include <cutlass/device_kernel.h>

#include "../fp8/input_layout.cuh"
#include "../gemm.cuh"

namespace ffpa_fp4 {

using namespace nvcuda;

// CTA: one (m_tile 128, n_tile 128) output tile of one (b, h). 8 warps,
// warp w owns rows [16w, 16w+16), full 128 cols via 8 wmma n-tiles.
// The K reduction (kHeadDim) is staged in 64-element chunks: full A/B
// staging would need 128*(D+8)*2*2 bytes and breaks the 99KB sm_120 budget
// at D>=192, so A/B land in a 2*128*72*2 = 36KB window that is reused as
// the f32 accumulator staging tile (128x136 floats = 68KB) for the
// epilogue. Rows m >= Mb and n >= Nkv are zero-filled on load: no OOB gmem
// reads and garbage-free accumulators (out-of-range rows are skipped in the
// epilogue anyway).
template <typename T, int kHeadDim>
__global__ void delta_s_wmma_kernel(
    const T* __restrict__ qm,   // [Nb, Nh, Mb, kHeadDim] row-major
    const T* __restrict__ k,    // (B,H,N,D) via Lkv (BHND- or NHD-packed)
    const T* __restrict__ qkm,  // [Nb, Nh, Mb] row-major
    float* __restrict__ ds,     // [Nb, Nh, Mb, Nkv_pad] row-major
    int Nh, int Nh_kv, int Mb, int Nkv, int Nkv_pad, int d_og,
    ffpa_fp8::Fp8InputLayout Lkv) {
  namespace w = nvcuda::wmma;
  constexpr int kChunk = 64;   // K elements staged per pass
  constexpr int kLdAB = 72;    // kChunk + 8 halves (ldm multiple of 8)
  constexpr int kLdAcc = 136;  // multiple of 4 floats / 16B rows
  static_assert(kHeadDim % kChunk == 0, "delta_s requires 64-multiple D");
  const int m_tiles = (Mb + 127) / 128;
  const int bh = blockIdx.y / m_tiles;
  const int tile_m = blockIdx.y % m_tiles;
  const int b = bh / Nh;
  const int h = bh % Nh;
  const int n0 = blockIdx.x * 128;
  const int warp = threadIdx.x / 32;
  const int m0 = tile_m * 128 + warp * 16;
  // Mb can be < 128 (short Nq): out-of-range warps contribute nothing.
  const bool m_ok = m0 < Mb;

  extern __shared__ float tile[];
  T* sA = reinterpret_cast<T*>(tile);
  T* sB = sA + 128 * kLdAB;

  const T* gA = qm + (long)(bh * Mb + tile_m * 128) * kHeadDim;
  const int bh_kv = b * Nh_kv + h / (Nh / Nh_kv);
  const T* gB = k + fp8_plane_base(Lkv, bh_kv) + (long)n0 * Lkv.s_row;
  constexpr int kVec4PerRow = kChunk / 8;  // 16B loads per staged row

  w::fragment<w::accumulator, 16, 16, 16, float> acc[8];
#pragma unroll
  for (int n = 0; n < 8; ++n)
    w::fill_fragment(acc[n], 0.0f);

  for (int kc = 0; kc < kHeadDim; kc += kChunk) {
    // Cooperative A/B chunk load: 128 rows x 64 halves (8 uint4) each.
#pragma unroll
    for (int i = threadIdx.x; i < 128 * kVec4PerRow; i += 256) {
      const int row = i / kVec4PerRow;
      const int col = (i % kVec4PerRow) * 8;
      if (tile_m * 128 + row < Mb) {
        *reinterpret_cast<uint4*>(sA + row * kLdAB + col) =
            *reinterpret_cast<const uint4*>(gA + row * kHeadDim + kc + col);
      } else {
        *reinterpret_cast<uint4*>(sA + row * kLdAB + col) = uint4{0, 0, 0, 0};
      }
      // K rows are only d_og wide (d_og%8==0: 8-elem uint4 loads stay
      // whole); guarded loads keep pad cols zero. qm is a padded
      // kHeadDim-wide buffer whose pad cols are already 0.
      if (n0 + row < Nkv && kc + col < d_og) {
        *reinterpret_cast<uint4*>(sB + row * kLdAB + col) =
            *reinterpret_cast<const uint4*>(gB + row * Lkv.s_row + kc + col);
      } else {
        *reinterpret_cast<uint4*>(sB + row * kLdAB + col) = uint4{0, 0, 0, 0};
      }
    }
    __syncthreads();
    if (m_ok) {
#pragma unroll
      for (int kk = 0; kk < kChunk / 16; ++kk) {
        w::fragment<w::matrix_a, 16, 16, 16, T, w::row_major> fa;
        // sA holds one 128-row tile: index by the in-tile row (warp * 16),
        // not the global m0 (tile_m * 128 + warp * 16).
        w::load_matrix_sync(fa, sA + warp * 16 * kLdAB + kk * 16, kLdAB);
#pragma unroll
        for (int n = 0; n < 8; ++n) {
          w::fragment<w::matrix_b, 16, 16, 16, T, w::col_major> fb;
          w::load_matrix_sync(fb, sB + n * 16 * kLdAB + kk * 16, kLdAB);
          w::mma_sync(acc[n], fa, fb, acc[n]);
        }
      }
    }
    // A/B smem is dead for every warp only after all warps finish the mma
    // loop (warps read B rows owned by other warps), so the next chunk load
    // must wait.
    __syncthreads();
  }

  // The A/B staging is dead; the same storage now backs the f32 accumulator
  // tile written below.
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    w::store_matrix_sync(tile + warp * 16 * kLdAcc + n * 16, acc[n], kLdAcc,
                         w::mem_row_major);
  }
  __syncthreads();

  const int row = threadIdx.x / 2;
  const int c4 = (threadIdx.x % 2) * 64;  // 16 float4 per thread
  const int m_global = tile_m * 128 + row;
  if (m_global >= Mb)
    return;
  const float qk = static_cast<float>(qkm[(long)bh * Mb + m_global]);
  const float* src = tile + row * kLdAcc + c4;
  float* out = ds + ((long)(bh * Mb) + m_global) * Nkv_pad + n0 + c4;
  const int n_valid = Nkv - n0 - c4;  // valid cols in this thread's span
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const float* s = src + i * 4;
    // qm @ (k - km)^T = (qm @ k^T) - qm.km: subtract ONE qkm.
    float4 v = {s[0] - qk, s[1] - qk, s[2] - qk, s[3] - qk};
    if (i * 4 + 3 < n_valid) {
      reinterpret_cast<float4*>(out)[i] = v;
    } else {  // tail tile: column-wise valid/zero mix
      float* o = out + i * 4;
      o[0] = i * 4 + 0 < n_valid ? v.x : 0.0f;
      o[1] = i * 4 + 1 < n_valid ? v.y : 0.0f;
      o[2] = i * 4 + 2 < n_valid ? v.z : 0.0f;
      o[3] = i * 4 + 3 < n_valid ? v.w : 0.0f;
    }
  }
}

// CuTe + TMA rewrite of delta_s_wmma_kernel (shape-dispatched in
// launch_fp4_delta_s_sm120).
// Same math/tiling (128x128 tile, 64-wide K chunks, f32 acc, qkm subtract,
// tail zero-fill) but TMA loads feed a 3-stage pipeline with the fp16/bf16
// HMMA m16n8k16 atom (SM80_*F32*_TN, the wmma 16x16x16 lowering), and the
// epilogue keeps the float4 union-staged store. A/B loops iterate the full
// kHeadDim/64 chunk count symmetrically: K chunks beyond d_og are all-OOB
// TMA boxes and zero-fill inside the TMA unit (no DRAM traffic), matching
// the old guarded loads. A uses a 3D (Mb, D, Nb*Nh) descriptor so tail rows
// of a partial m-tile zero-fill per (b,h) exactly like the old guards.
template <typename T, int kHeadDim, int kStages, typename TmaA, typename TmaB>
__global__ void __launch_bounds__(256, 1)
    delta_s_cute_kernel(CUTLASS_GRID_CONSTANT TmaA const tma_a,
                        CUTLASS_GRID_CONSTANT TmaB const tma_b,
                        const T* __restrict__ qkm, float* __restrict__ ds,
                        int Nh, int Nh_kv, int Mb, int Nkv, int Nkv_pad,
                        int d_og, int Nb) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  using namespace cute;
  using Element = std::conditional_t<std::is_same<T, __half>::value,
                                     cutlass::half_t, cutlass::bfloat16_t>;
  using TmaBarrier = cutlass::arch::ClusterTransactionBarrier;
  using CtaBarrier = cutlass::arch::ClusterBarrier;
  constexpr int kChunk = 64;
  constexpr int kDChunks = kHeadDim / kChunk;
  constexpr int kNumThreads = 256;
  constexpr int kLdAcc = 136;
  constexpr int kStageElems = 2 * 128 * kChunk;  // A + B per stage
  constexpr int kSmemAB = kStages * kStageElems * (int)sizeof(T);
  constexpr int kSmemAcc = 128 * kLdAcc * (int)sizeof(float);
  constexpr int kSmemBytes = kSmemAB > kSmemAcc ? kSmemAB : kSmemAcc;
  static_assert(kSmemBytes <= 101376, "delta_s cute exceeds sm_120 optin");

  const int m_tiles = (Mb + 127) / 128;
  const int bh = blockIdx.y / m_tiles;
  const int tile_m = blockIdx.y % m_tiles;
  const int b = bh / Nh;
  const int h = bh % Nh;
  const int h_kv = h / (Nh / Nh_kv);
  const int n_tile = blockIdx.x;
  const int tid = threadIdx.x;

  extern __shared__ __align__(1024) char smem_raw[];
  Element* ab_base = reinterpret_cast<Element*>(smem_raw);

  using MmaAtom = std::conditional_t<std::is_same<T, __half>::value,
                                     MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
                                     MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>>;
  auto tiled_mma = make_tiled_mma(MmaAtom{}, Layout<Shape<_8, _1, _1>>{},
                                  Tile<Int<128>, Int<128>, _16>{});
  auto thr_mma = tiled_mma.get_thread_slice(tid);
  using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
  auto s2r_copy_a = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
  auto s2r_copy_b = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
  auto s2r_thr_a = s2r_copy_a.get_thread_slice(tid);
  auto s2r_thr_b = s2r_copy_b.get_thread_slice(tid);

  // A: (Mb, kHeadDim) per (b,h); B: (Nkv, d_og) per (b,h_kv), tiled
  // (128, 64) per chunk. gmem views are built per issue inside the lambda
  // (the split_d issue_qk_tma pattern) so the TMA partition has no rest mode.
  auto mA = tma_a.get_tma_tensor(make_shape(Mb, kHeadDim, gridDim.y / m_tiles));
  auto mB = tma_b.get_tma_tensor(make_shape(Nkv, d_og, Nh_kv, Nb));
  auto a_slice = tma_a.get_slice(_0{});
  auto b_slice = tma_b.get_slice(_0{});

  // 16-bit rows: 64 elements x 2B = 128B -> K_SW128 swizzle atom, tiled to
  // the (128, 64) chunk (the sm_120 QK smem pattern, cute/attn_traits.cuh).
  using SmemAtomAB = GMMA::Layout_K_SW128_Atom<Element>;
  using SmemLayoutAB =
      decltype(tile_to_shape(SmemAtomAB{}, Shape<Int<128>, Int<kChunk>>{}));
  auto sA0 = make_tensor(make_smem_ptr(ab_base), SmemLayoutAB{});
  auto sB0 = make_tensor(make_smem_ptr(ab_base + 128 * kChunk), SmemLayoutAB{});

  // One stage's A+B TMA issue on `bar` (caller sets the expect_tx count).
  auto issue_tma = [&](int chunk, int stage, uint64_t& bar) {
    cutlass::arch::fence_view_async_shared();
    auto sA = make_tensor(make_smem_ptr(ab_base + stage * kStageElems),
                          SmemLayoutAB{});
    auto sB =
        make_tensor(make_smem_ptr(ab_base + stage * kStageElems + 128 * kChunk),
                    SmemLayoutAB{});
    auto gA = local_tile(mA(_, _, bh), Shape<Int<128>, Int<kChunk>>{},
                         make_coord(tile_m, chunk));
    auto gB = local_tile(mB(_, _, h_kv, b), Shape<Int<128>, Int<kChunk>>{},
                         make_coord(n_tile, chunk));
    copy(tma_a.with(bar), a_slice.partition_S(gA), a_slice.partition_D(sA));
    copy(tma_b.with(bar), b_slice.partition_S(gB), b_slice.partition_D(sB));
  };

  auto tCrA = thr_mma.partition_fragment_A(sA0);
  auto tCrB = thr_mma.partition_fragment_B(sB0);
  auto tCrC = partition_fragment_C(tiled_mma, Shape<Int<128>, Int<128>>{});
  clear(tCrC);

  auto gemm_stage = [&](int stage) {
    auto sA = make_tensor(make_smem_ptr(ab_base + stage * kStageElems),
                          SmemLayoutAB{});
    auto sB =
        make_tensor(make_smem_ptr(ab_base + stage * kStageElems + 128 * kChunk),
                    SmemLayoutAB{});
    auto tSsA = s2r_thr_a.partition_S(sA);
    auto tSsB = s2r_thr_b.partition_S(sB);
    ffpa_cute::gemm_ss(tCrC, tCrA, tCrB, tSsA, tSsB, tiled_mma, s2r_copy_a,
                       s2r_copy_b, s2r_thr_a, s2r_thr_b);
  };

  if constexpr (kDChunks <= kStages) {
    // Every chunk lands in its own stage: no stage reuse, so a single
    // barrier and zero empty-barrier handshake (the small-work fast path;
    // the full pipeline below costs more than it hides when D <= 192).
    __shared__ uint64_t full0;
    if (tid == 0)
      TmaBarrier::init(&full0, 1);
    __syncthreads();
    if (tid == 0) {
      TmaBarrier::arrive_and_expect_tx(
          &full0, kDChunks * 2 * 128 * kChunk * (int)sizeof(T));
#pragma unroll
      for (int c = 0; c < kDChunks; ++c)
        issue_tma(c, c, full0);
    }
    TmaBarrier::wait(&full0, 0);
    cutlass::arch::fence_view_async_shared();
#pragma unroll
    for (int c = 0; c < kDChunks; ++c)
      gemm_stage(c);
  } else {
    __shared__ uint64_t full[kStages];
    __shared__ uint64_t empty[kStages];
    if (tid == 0) {
      for (int s = 0; s < kStages; ++s) {
        TmaBarrier::init(&full[s], 1);
        CtaBarrier::init(&empty[s], kNumThreads);
      }
    }
    __syncthreads();

    // Signal all stages empty so tid=0's initial prefetch can land.
    for (int s = 0; s < kStages; ++s)
      CtaBarrier::arrive(&empty[s]);
    if (tid == 0) {
      for (int c = 0; c < kStages && c < kDChunks; ++c) {
        CtaBarrier::wait(&empty[c], 0);
        TmaBarrier::arrive_and_expect_tx(&full[c],
                                         2 * 128 * kChunk * (int)sizeof(T));
        issue_tma(c, c, full[c]);
      }
    }

#pragma unroll 1
    for (int c = 0; c < kDChunks; ++c) {
      const int stage = c % kStages;
      const int phase = (c / kStages) & 1;
      TmaBarrier::wait(&full[stage], phase);
      cutlass::arch::fence_view_async_shared();
      gemm_stage(stage);
      CtaBarrier::arrive(&empty[stage]);
      // Same-stage next-chunk prefetch must stay after the arrive above (see
      // the split_d QK loop deadlock note).
      if (tid == 0) {
        const int c_next = c + kStages;
        if (c_next < kDChunks) {
          const int s_next = c_next % kStages;
          const int phase_next = (c_next / kStages) & 1;
          CtaBarrier::wait(&empty[s_next], phase_next);
          TmaBarrier::arrive_and_expect_tx(&full[s_next],
                                           2 * 128 * kChunk * (int)sizeof(T));
          issue_tma(c_next, s_next, full[s_next]);
        }
      }
    }
  }

  // Epilogue: the pipeline smem is dead; union it into the 128x136 f32
  // staging tile and keep the wmma kernel's float4 store (acc fragments
  // hold only 2 contiguous f32 per row, so a direct 16B store is impossible).
  __syncthreads();
  auto acc_layout = make_layout(make_shape(Int<128>{}, Int<128>{}),
                                make_stride(Int<kLdAcc>{}, _1{}));
  auto sAcc = make_tensor(make_smem_ptr(reinterpret_cast<float*>(smem_raw)),
                          acc_layout);
  auto tCsAcc = thr_mma.partition_C(sAcc);
  copy(tCrC, tCsAcc);
  __syncthreads();

  const int row = tid / 2;
  const int c4 = (tid % 2) * 64;  // 16 float4 per thread
  const int m_global = tile_m * 128 + row;
  if (m_global >= Mb)
    return;
  const float qk = static_cast<float>(qkm[(long)bh * Mb + m_global]);
  const float* src =
      reinterpret_cast<const float*>(smem_raw) + row * kLdAcc + c4;
  float* out = ds + ((long)(bh * Mb) + m_global) * Nkv_pad + n_tile * 128 + c4;
  const int n_valid = Nkv - n_tile * 128 - c4;  // valid cols in this span
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const float* s = src + i * 4;
    float4 v = {s[0] - qk, s[1] - qk, s[2] - qk, s[3] - qk};
    if (i * 4 + 3 < n_valid) {
      reinterpret_cast<float4*>(out)[i] = v;
    } else {  // tail tile: column-wise valid/zero mix
      float* o = out + i * 4;
      o[0] = i * 4 + 0 < n_valid ? v.x : 0.0f;
      o[1] = i * 4 + 1 < n_valid ? v.y : 0.0f;
      o[2] = i * 4 + 2 < n_valid ? v.z : 0.0f;
      o[3] = i * 4 + 3 < n_valid ? v.w : 0.0f;
    }
  }
#endif
}

// TMA launcher for the CuTe variant. s_batch is only stored for NHD inputs,
// so derive the BHND batch stride (Nh_kv planes of Nkv*d_og) here.
template <typename T, int kHeadDim>
inline void launch_fp4_delta_s_cute_sm120(const T* qm, const T* k, const T* qkm,
                                          float* ds, int Nb, int Nh, int Nh_kv,
                                          int Mb, int Nkv, int Nkv_pad,
                                          int d_og, cudaStream_t stream,
                                          const ffpa_fp8::Fp8InputLayout& L) {
  using namespace cute;
  constexpr int kChunk = 64;
  constexpr int kStages = 3;
  using Element = std::conditional_t<std::is_same<T, __half>::value,
                                     cutlass::half_t, cutlass::bfloat16_t>;
  using SmemAtomAB = GMMA::Layout_K_SW128_Atom<Element>;
  using SmemLayoutAB =
      decltype(tile_to_shape(SmemAtomAB{}, Shape<Int<128>, Int<kChunk>>{}));

  auto gA = make_tensor(
      make_gmem_ptr(const_cast<Element*>(reinterpret_cast<const Element*>(qm))),
      make_shape(Mb, kHeadDim, (long)Nh * Nb),
      make_stride((long)kHeadDim, _1{}, (long)Mb * kHeadDim));
  auto tma_a = make_tma_copy(SM90_TMA_LOAD{}, gA, SmemLayoutAB{},
                             Shape<Int<128>, Int<kChunk>>{}, _1{});
  const long s_batch = L.nhd ? L.s_batch : (long)Nh_kv * L.s_head;
  auto gB = make_tensor(
      make_gmem_ptr(const_cast<Element*>(reinterpret_cast<const Element*>(k))),
      make_shape(Nkv, d_og, Nh_kv, Nb),
      make_stride(L.s_row, _1{}, L.s_head, s_batch));
  auto tma_b = make_tma_copy(SM90_TMA_LOAD{}, gB, SmemLayoutAB{},
                             Shape<Int<128>, Int<kChunk>>{}, _1{});

  constexpr int kSmemAB = kStages * 2 * 128 * kChunk * (int)sizeof(T);
  constexpr int kSmemAcc = 128 * 136 * 4;
  constexpr int smem = kSmemAB > kSmemAcc ? kSmemAB : kSmemAcc;
  auto kernel = delta_s_cute_kernel<T, kHeadDim, kStages, decltype(tma_a),
                                    decltype(tma_b)>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       smem);
  const int tc = Nkv_pad / 128;
  const int m_tiles = (Mb + 127) / 128;
  dim3 grid(tc, Nb * Nh * m_tiles);
  kernel<<<grid, 256, smem, stream>>>(tma_a, tma_b, qkm, ds, Nh, Nh_kv, Mb, Nkv,
                                      Nkv_pad, d_og, Nb);
}

template <typename T, int kHeadDim>
inline void launch_fp4_delta_s_sm120(
    const T* qm, const T* k, const T* qkm, float* ds, int Nb, int Nh, int Nh_kv,
    int Mb, int Nkv, int Nkv_pad, int d_og, cudaStream_t stream,
    const ffpa_fp8::Fp8InputLayout* Lkv = nullptr) {
  // nullptr keeps the historical BHND addressing (bh*Nkv*d_og plane base).
  const ffpa_fp8::Fp8InputLayout L =
      Lkv ? *Lkv
          : ffpa_fp8::Fp8InputLayout{false, 0, 0, (long)Nkv * d_og, d_og};
  // Shape dispatch: TMA wins once enough CTAs hide its issue latency;
  // small-Mb cross shapes stay on wmma.
  constexpr int kDChunks = kHeadDim / 64;
  if (kDChunks >= 4 || Mb >= 32 || (Mb >= 16 && Nkv_pad >= 4096)) {
    launch_fp4_delta_s_cute_sm120<T, kHeadDim>(
        qm, k, qkm, ds, Nb, Nh, Nh_kv, Mb, Nkv, Nkv_pad, d_og, stream, L);
    return;
  }
  const int tc = Nkv_pad / 128;
  const int m_tiles = (Mb + 127) / 128;
  dim3 grid(tc, Nb * Nh * m_tiles);
  // Union of the A/B chunk staging (2*128*72*sizeof(T)) and the f32 acc
  // tile (128*136*4); the acc side dominates for fp16/bf16.
  constexpr int kSmemAB = 2 * 128 * 72 * (int)sizeof(T);
  constexpr int kSmemAcc = 128 * 136 * 4;
  constexpr int smem = kSmemAB > kSmemAcc ? kSmemAB : kSmemAcc;
  cudaFuncSetAttribute(delta_s_wmma_kernel<T, kHeadDim>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  delta_s_wmma_kernel<T, kHeadDim><<<grid, 256, smem, stream>>>(
      qm, k, qkm, ds, Nh, Nh_kv, Mb, Nkv, Nkv_pad, d_og, L);
}

}  // namespace ffpa_fp4

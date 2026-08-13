#pragma once

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>

namespace ffpa_fp8 {

// ============================================================================
// Background: why a C->A register reorganization is needed at all
// ============================================================================
// FFPA's fp8 attention runs two m16n8k32 MMAs per KV tile:
//   QK GEMM:  S(16x32 per atom) = Q @ K^T, accumulator in the C layout
//   PV GEMM:  O += P @ V, with P fed as the A operand from registers
// The QK C-fragment and the PV A-fragment distribute a 16x32 tile over the
// warp DIFFERENTLY (PTX ISA, "Matrix Fragments for mma.m16n8k32"; with
// groupID g = lane>>2 and threadID_in_group t = lane&3):
//
//   C fragment (4 regs c0..c3, one value each; 2 cols per row):
//     c0,c1: row g,   cols {2t, 2t+1}
//     c2,c3: row g+8, cols {2t, 2t+1}
//     => per n8 tile a thread holds 2 bytes of row g and 2 of row g+8, at
//        column pair {2t, 2t+1}; consecutive n8 tiles along N are contiguous
//        in register order.
//
//   A fragment (4 regs a0..a3, four 8-bit elems each; 4 k-cols per row):
//     a0: row g,   k = 4t..4t+3
//     a1: row g+8, k = 4t..4t+3
//     a2: row g,   k = 16+4t..16+4t+3
//     a3: row g+8, k = 16+4t..16+4t+3
//     => thread t needs k-columns {4t..4t+3} which, expressed in the C-tile
//        column pairs {2t',2t'+1}, belong to OTHER lanes (e.g. t=1 needs the
//        bytes owned by t=2 and t=3). A purely in-thread permutation cannot
//        build the natural A layout (proven in .tmp/fp8_persist_d/
//        verify_layouts.py), hence the cross-lane ReorgC8bitToA8bit below.

// 8-bit (fp8 e4m3 or symmetric int8; identical A/B operand layouts) m16n8k32:
// reorganize the (fp32->8bit) C-fragment of the QK GEMM into the A-operand
// register layout of the PV GEMM. Rule table derived numerically
// with pycute from MMA_Traits<SM89_16x8x32_F32E4M3E4M3F32_TN>
// (.tmp/fp8_persist_d/verify_layouts.py, V1b): per K=32 group (4 m16n8
// C-tiles c0..c3), A reg r gathers tiles (r%2, r%2+2) from quad-partner lane
// (t%4<2 -> self, else lane-2 within the quad), then __byte_perm picks bytes
// (v0,v2)/(v1,v3): sel 0x6240 for even lanes, 0x7351 for odd lanes.
struct ReorgC8bitToA8bit {
  int selectorEx0;
  int selectorEx1;
  int selectorEx4;
  int selectorEx5;
  // Scalar peer lane ids (was upper_map/lower_map[lane%4]: dynamic local-array
  // indexing generated LDL on every call -> 1GB local traffic per kernel).
  int upper_peer;
  int lower_peer;

  CUTLASS_DEVICE ReorgC8bitToA8bit() {
    int laneId = cutlass::canonical_lane_idx();
    if (laneId % 4 == 0 || laneId % 4 == 3) {
      selectorEx0 = 0x3210;
      selectorEx1 = 0x7654;
      selectorEx4 = 0x5410;
      selectorEx5 = 0x7632;
    } else {
      selectorEx0 = 0x7654;
      selectorEx1 = 0x3210;
      selectorEx4 = 0x1054;
      selectorEx5 = 0x3276;
    }
    // Peer maps as 2-bit fields (LSB-first) instead of constexpr arrays:
    // runtime lane%4 indexing into local arrays emitted LDL/STL on the stack.
    // kUpperMap = {0, 3, 1, 2} -> 0b10_01_11_00; kLowerMap = {1, 2, 0, 3}
    // -> 0b11_00_10_01.
    const int l4 = laneId & 3;
    upper_peer = (0x9C >> (2 * l4)) & 3;
    lower_peer = (0xC9 >> (2 * l4)) & 3;
  }

  // data: per-thread fp8 values ordered as contiguous m16n8 C-tiles along N
  // (4 values per tile). Rewritten in place into A-operand order.
  template <typename Fragment>
  CUTLASS_DEVICE void operator()(Fragment& accum) const {
    auto* data = accum.data();
    const int total = decltype(cute::size(accum))::value;
    for (int n = 0; n < total; n += 8) {
      uint32_t upper = *reinterpret_cast<uint32_t*>(&data[n]);
      uint32_t lower = *reinterpret_cast<uint32_t*>(&data[n + 4]);
      uint32_t upper0 = __byte_perm(upper, lower, selectorEx0);
      uint32_t lower0 = __byte_perm(upper, lower, selectorEx1);
      upper0 = __shfl_sync(0xffffffff, upper0, upper_peer, 4);
      lower0 = __shfl_sync(0xffffffff, lower0, lower_peer, 4);
      uint32_t* d32 = reinterpret_cast<uint32_t*>(&data[n]);
      d32[0] = __byte_perm(upper0, lower0, selectorEx4);
      d32[1] = __byte_perm(upper0, lower0, selectorEx5);
    }
  }
};

// ============================================================================
// PackC8bitToA8bitPermVT: SHUFFLE-FREE C->A pack + V^T column permutation
// ============================================================================
// ReorgC8bitToA8bit costs 16 SHFL + 32 PRMT per thread per 128-col tile, all
// on the QK->softmax->PV critical path. This pack eliminates the SHFLs by
// exploiting the MMA's permutation invariance along the reduction (k) axis:
//
//   sum_k P[m,k]*V[k,n] == sum_s P[m,pi(s)]*V[pi(s),n]   (pi any bijection)
//
// Instead of moving P bytes across lanes to reach the NATURAL A layout, each
// thread packs its LOCALLY owned C-fragment bytes into A registers and lets
// the A operand carry a PERMUTED k-indexing; the V^T operand is stored with
// the matching column permutation so slot s of both operands meets the same
// true kv position pi(s). The PV MMA and the all-ones-B row-sum MMA both sum
// over all slots, so they stay exact (verified in .tmp/int8-f16-opt/
// verify_perm.py).
//
// Pack rule (per K=32 group = 4 contiguous n8 C-tiles w0..w3, 16 bytes).
// Tile j's 4 bytes are the C-fragment values (c0..c3) of kv-cols 8j..8j+7:
//   w_j = [ P[g, 8j+2t], P[g, 8j+2t+1], P[g+8, 8j+2t], P[g+8, 8j+2t+1] ]
// Concatenating pairs (w0,w1) / (w2,w3) gives __byte_perm inputs with byte
// indices 0..7 / 8..15; the four A registers are:
//   a0 = row g   bytes {0,1,4,5}   -> sel 0x5410  (tiles 0,1)
//   a1 = row g+8 bytes {2,3,6,7}   -> sel 0x7632  (tiles 0,1)
//   a2 = row g   bytes {8,9,12,13} -> sel 0xDC98  (tiles 2,3)
//   a3 = row g+8 bytes {10,11,14,15}-> sel 0xFEBA (tiles 2,3)
// The selectors are LANE-INDEPENDENT (unlike ReorgC8bitToA8bit): 4 PRMT per
// K=32 group, zero SHFL, no per-lane constructor state.
//
// Induced permutation pi: A-slot s (relative to the K=32 group) holds the P
// byte of true kv column
//   pi(4t+r)    = 8*(r>>1) + 2t + (r&1)          for s = 4t+r   < 16
//   pi(16+4t+r) = 16 + 8*(r>>1) + 2t + (r&1)     for s = 16+4t+r
// i.e. pi = [0,1,8,9, 2,3,10,11, 4,5,12,13, 6,7,14,15] (+16 in the upper
// half). pi repeats every 32 columns across the whole kBc-wide tile.
//
// CONTRACT: V^T must be stored permuted, V^T_perm[d, s] = V^T[d, pi(s)]
// (equivalently the quantize pre-kernel writes true kv row j into column
// pi^-1(j) = 4*((j>>1)&3) + 2*((j>>3)&1) + (j&1), per 32-col group; see
// VTPermInv32 in quantize_fp8.cuh). Pairing is enforced by the launcher
// (reorg_free gate in launch.cuh, on by default for every persist_d fp8
// config); this pack must NEVER run against an unpermuted V^T.
// ReorgC8bitToA8bit above stays compiled as the fallback/contrast path and
// is still used by the split_d family.
struct PackC8bitToA8bitPermVT {
  template <typename Fragment>
  CUTLASS_DEVICE void operator()(Fragment& accum) const {
    auto* data = accum.data();
    const int total = decltype(cute::size(accum))::value;
    static_assert(decltype(cute::size(accum))::value % 16 == 0,
                  "perm pack works on K=32 groups (16 bytes each)");
#pragma unroll
    for (int n = 0; n < total; n += 16) {
      const uint32_t* src = reinterpret_cast<const uint32_t*>(&data[n]);
      const uint32_t w0 = src[0];
      const uint32_t w1 = src[1];
      const uint32_t w2 = src[2];
      const uint32_t w3 = src[3];
      uint32_t* d32 = reinterpret_cast<uint32_t*>(&data[n]);
      d32[0] = __byte_perm(w0, w1, 0x5410);  // a0: row g,   tiles 0,1
      d32[1] = __byte_perm(w0, w1, 0x7632);  // a1: row g+8, tiles 0,1
      d32[2] = __byte_perm(w2, w3, 0xDC98);  // a2: row g,   tiles 2,3
      d32[3] = __byte_perm(w2, w3, 0xFEBA);  // a3: row g+8, tiles 2,3
    }
  }
};

}  // namespace ffpa_fp8

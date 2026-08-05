#pragma once

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>

namespace ffpa_cute {

// FP8 m16n8k32: reorganize the (fp32->e4m3) C-fragment of the QK GEMM into the
// A-operand register layout of the PV GEMM. Rule table derived numerically
// with pycute from MMA_Traits<SM89_16x8x32_F32E4M3E4M3F32_TN>
// (.tmp/fp8_persist_d/verify_layouts.py, V1b): per K=32 group (4 m16n8
// C-tiles c0..c3), A reg r gathers tiles (r%2, r%2+2) from quad-partner lane
// (t%4<2 -> self, else lane-2 within the quad), then __byte_perm picks bytes
// (v0,v2)/(v1,v3): sel 0x6240 for even lanes, 0x7351 for odd lanes.
struct ReorgCFp8toAFp8 {
  int selectorEx0;
  int selectorEx1;
  int selectorEx4;
  int selectorEx5;
  // Scalar peer lane ids (was upper_map/lower_map[lane%4]: dynamic local-array
  // indexing generated LDL on every call -> 1GB local traffic per kernel).
  int upper_peer;
  int lower_peer;

  CUTLASS_DEVICE ReorgCFp8toAFp8() {
    int laneId = cutlass::canonical_lane_idx();
    constexpr int kUpperMap[4] = {0, 3, 1, 2};
    constexpr int kLowerMap[4] = {1, 2, 0, 3};
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
    upper_peer = kUpperMap[laneId % 4];
    lower_peer = kLowerMap[laneId % 4];
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

}  // namespace ffpa_cute

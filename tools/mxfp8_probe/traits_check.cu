// Compile-time check for the SM120_16x32x32_TN_VS_MXFP8 fused atom: the
// cute layout algebra (TV layouts, tiled mma construction, SF partitioning)
// must be self-consistent before the kernel work starts. Host-only build.
// Build: nvcc -gencode arch=compute_120f,code=sm_120f -std=c++20 -I<cutlass>
//        -I../../csrc/cuffpa -c traits_check.cu -o /dev/null
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>

#include "cute/fp4/cute_ext.h"

int main() {
  using namespace cute;
  using ffpa_fp4::MMA_Atom;
  using Atom = MMA_Atom<SM120::BLOCKSCALED::SM120_16x32x32_TN_VS_MXFP8>;
  auto tiled_pv = make_tiled_mma(Atom{}, Layout<Shape<_8, _1, _1>>{},
                                 Tile<_128, _32, _128>{});
  static_assert(size<2>(tile_shape(tiled_pv)) == 128, "PV tile K");
  // SF partition smoke: (N32 rows, K128/32 groups) logical SF tensor.
  static uint8_t buf[512];
  auto sf = make_tensor(&buf[0], make_layout(make_shape(_128{}, _4{})));
  auto thr = tiled_pv.get_thread_slice(0);
  auto part = partition_fragment_SFB(sf, thr);
  (void)part;
  return 0;
}

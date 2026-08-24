// Shared gmem layout descriptor for the fp8/fp4 quantize pre-kernels.
// Activation tensors are shaped [B, H, N, D] but may be backed by either a
// BHND-packed tensor (historical) or an NHD-packed [B, N, H, D] tensor viewed
// as BHND (diffusers convention, zero-copy permute view). The pre-kernels
// read the ORIGINAL gmem with these strides; quantized outputs stay BHND.
#pragma once

namespace ffpa_fp8 {

struct Fp8InputLayout {
  bool nhd;
  int nh;        // NHD only: heads per batch (bh -> b*Nh + h decomposition)
  long s_batch;  // NHD only: element stride of the batch dim
  long s_head;   // BHND: bh plane stride (N*D_og); NHD: head stride (D_og)
  long s_row;    // row stride: BHND D_og; NHD H*D_og
};

__host__ __device__ __forceinline__ long fp8_plane_base(const Fp8InputLayout& L,
                                                        int bh) {
  return L.nhd ? static_cast<long>(bh / L.nh) * L.s_batch +
                     static_cast<long>(bh % L.nh) * L.s_head
               : static_cast<long>(bh) * L.s_head;
}

}  // namespace ffpa_fp8

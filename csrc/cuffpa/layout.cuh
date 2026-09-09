// Layout-family predicates shared by the routing layer (launch/router.cuh),
// the cute launchers (launch/cute_*.cuh) and the hybrid stage-1 prep
// (dispatch/cute_hybrid.cuh). Moved verbatim out of the old monolithic
// cute/launch.cuh so the dispatcher TU never needs the CUTLASS/kernel
// headers. Unguarded on purpose: the router's NHD output check uses these
// outside the CUTE/TMA extension guards.
#pragma once
#include <torch/types.h>

// NHD (diffusers BNHD) input detection: a [B, H, N, D]-shaped tensor that is
// a zero-copy permute view of a packed [B, N, H, D] tensor, i.e.
// strides (N*H*D, D, H*D, 1). Returns false for BHND-packed tensors.
// B == 1 makes stride(0) an ignored leftover (CP comm primitives leave the
// pre-permute value), so it is exempt from the exact check.
static inline bool ffpa_is_nhd_view(const torch::Tensor& X) {
  const long B = X.size(0), H = X.size(1), N = X.size(2), D = X.size(3);
  return X.dim() == 4 && X.stride(3) == 1 && X.stride(2) == H * D &&
         X.stride(1) == D && (B <= 1 || X.stride(0) == N * H * D);
}

// BHND-packed detection: contiguous [B, H, N, D] strides.
static inline bool ffpa_is_bhnd_packed(const torch::Tensor& X) {
  const long B = X.size(0), H = X.size(1), N = X.size(2), D = X.size(3);
  return X.dim() == 4 && X.stride(3) == 1 && X.stride(2) == D &&
         X.stride(1) == N * D && X.stride(0) == H * N * D;
}

// Strided-NHD predicate: an NHD-family [B, H, N, D] view whose row stride
// exceeds H*D (fused-QKV interleaved chunk layouts) — neither BHND-packed
// nor a packed-NHD permute view. stride(2) >= H*D excludes negative strides
// and head-overlapping rows; stride(0) > 0 excludes reversed batches
// (negative strides can still be %16-aligned).
static inline bool ffpa_is_strided_nhd(const torch::Tensor& X) {
  const long B = X.size(0), H = X.size(1), N = X.size(2), D = X.size(3);
  return X.dim() == 4 && X.stride(3) == 1 && X.stride(1) == D &&
         X.stride(2) >= H * D && !ffpa_is_nhd_view(X) &&
         (B <= 1 || (X.stride(0) == X.stride(2) * N && X.stride(0) > 0));
}

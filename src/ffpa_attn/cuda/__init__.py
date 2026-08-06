"""CUDA FFPA attention forward/backward implementations for large-D (D > 256)."""
import enum

import torch

try:
  from .. import _C as _cuda_ext

  _ffpa_attn_fwd_cuda = _cuda_ext.ffpa_attn_forward
  CUDA_FWD_AVAILABLE = bool(getattr(_cuda_ext, "CUDA_FWD_AVAILABLE", False))
  F16_ACC_AVAILABLE = bool(getattr(_cuda_ext, "F16_ACC_AVAILABLE", False))
  CUDA_TMA_AVAILABLE = bool(getattr(_cuda_ext, "CUDA_TMA_AVAILABLE", False))
  CUDA_CUTE_TMA_AVAILABLE = bool(
    getattr(_cuda_ext, "CUDA_CUTE_TMA_AVAILABLE", False)
  )
  CUDA_BWD_AVAILABLE = False
  _CUDA_IMPORT_ERROR = None
except Exception as exc:
  _ffpa_attn_fwd_cuda = None
  CUDA_FWD_AVAILABLE = False
  F16_ACC_AVAILABLE = False
  CUDA_TMA_AVAILABLE = False
  CUDA_CUTE_TMA_AVAILABLE = False
  CUDA_BWD_AVAILABLE = False
  _CUDA_IMPORT_ERROR = exc


class CudaBackendImpl(enum.IntEnum):
  AUTO = 0
  NATIVE = 1
  TMA = 2
  CUTE = 3
  CUTE_TMA = 4
  CUTE_TMA_W8A8 = 5


def set_cuda_backend_impl(impl: CudaBackendImpl) -> None:
  """Set the CUDA backend implementation hint for kernel dispatch."""
  if _cuda_ext is not None:
    _cuda_ext.set_cuda_backend_impl(int(impl))


def get_cuda_backend_impl() -> CudaBackendImpl:
  """Get the current CUDA backend implementation hint."""
  if _cuda_ext is not None:
    return CudaBackendImpl(_cuda_ext.get_cuda_backend_impl())
  return CudaBackendImpl.AUTO


from ._ffpa_bwd import _ffpa_attn_backward_cuda
from ._ffpa_fwd import _ffpa_attn_forward_cuda

_OP_NAMESPACE = "ffpa_attn"

# ffpa_attn::_fwd_cuda
torch.library.define(
  f"{_OP_NAMESPACE}::_fwd_cuda",
  "(Tensor q, Tensor k, Tensor v, Tensor attn_bias, int stages, int acc, int causal, "
  "float softmax_scale, float dropout_p, int philox_seed, int philox_offset, "
  "bool smooth_k) -> (Tensor o, Tensor softmax_lse)",
)


@torch.library.impl(f"{_OP_NAMESPACE}::_fwd_cuda", "CUDA")
def _fwd_cuda_torch_op(
  Q: torch.Tensor,
  K: torch.Tensor,
  V: torch.Tensor,
  attn_bias: torch.Tensor,
  stages: int,
  acc: int,
  causal: int,
  softmax_scale: float,
  dropout_p: float,
  philox_seed: int,
  philox_offset: int,
  smooth_k: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
  if _ffpa_attn_fwd_cuda is None:
    raise RuntimeError(
      "ffpa_attn forward CUDA backend is unavailable. "
      "Rebuild with ENABLE_FFPA_CUDA_IMPL=1 to enable it. "
      f"Original import error: {_CUDA_IMPORT_ERROR}"
    )
  O = torch.empty_like(Q)  # noqa: E741
  seqlen_q = Q.size(2)
  # NOTE: allocate lse with the exact seqlen (no rounding). CUDA kernels index
  # the lse buffer flat as [B, Nh, Nq] (stride Nq per head); passing a padded
  # storage shifted head h's rows by h and leaves the last rows unwritten when
  # seqlen_q % 8 != 0.
  softmax_lse = torch.empty(
    Q.size(0),
    Q.size(1),
    seqlen_q,
    dtype=torch.float32,
    device=Q.device,
  )
  _ffpa_attn_fwd_cuda(
    Q,
    K,
    V,
    attn_bias,
    O,
    softmax_lse,
    stages,
    acc,
    causal,
    softmax_scale,
    dropout_p,
    philox_seed,
    philox_offset,
    smooth_k,
  )
  return O, softmax_lse


@torch.library.register_fake(f"{_OP_NAMESPACE}::_fwd_cuda")
def _fwd_cuda_fake(
  Q: torch.Tensor,
  K: torch.Tensor,
  V: torch.Tensor,
  attn_bias: torch.Tensor,
  stages: int,
  acc: int,
  causal: int,
  softmax_scale: float,
  dropout_p: float,
  philox_seed: int,
  philox_offset: int,
  smooth_k: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
  O = torch.empty_like(Q)  # noqa: E741
  softmax_lse = Q.new_empty(
    Q.size(0), Q.size(1), Q.size(2), dtype=torch.float32
  )
  return O, softmax_lse


__all__ = [
  "_ffpa_attn_forward_cuda",
  "_ffpa_attn_backward_cuda",
  "CUDA_FWD_AVAILABLE",
  "CUDA_TMA_AVAILABLE",
  "CUDA_CUTE_TMA_AVAILABLE",
  "CUDA_BWD_AVAILABLE",
]

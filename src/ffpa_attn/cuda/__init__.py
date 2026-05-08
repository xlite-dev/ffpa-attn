"""CUDA FFPA attention forward/backward implementations for large-D (D > 256)."""
import torch

try:
  from .. import _C as _cuda_ext

  _ffpa_attn_fwd_cuda = _cuda_ext.ffpa_attn_forward
  _ffpa_attn_bwd_cuda = _cuda_ext.ffpa_attn_backward
  CUDA_FWD_AVAILABLE = bool(getattr(_cuda_ext, "CUDA_FWD_AVAILABLE", True))
  CUDA_BWD_AVAILABLE = bool(getattr(_cuda_ext, "CUDA_BWD_AVAILABLE", True))
  _CUDA_IMPORT_ERROR = None
except Exception as exc:
  _ffpa_attn_fwd_cuda = None
  _ffpa_attn_bwd_cuda = None
  CUDA_FWD_AVAILABLE = False
  CUDA_BWD_AVAILABLE = False
  _CUDA_IMPORT_ERROR = exc

from ._ffpa_bwd import _ffpa_attn_backward_cuda
from ._ffpa_fwd import _ffpa_attn_forward_cuda

_OP_NAMESPACE = "ffpa_attn"

# ffpa_attn::_fwd_cuda
torch.library.define(
  f"{_OP_NAMESPACE}::_fwd_cuda",
  "(Tensor q, Tensor k, Tensor v, int stages, int acc, int causal, "
  "float softmax_scale, int tma) -> (Tensor o, Tensor softmax_lse)",
)


@torch.library.impl(f"{_OP_NAMESPACE}::_fwd_cuda", "CUDA")
def _fwd_cuda_torch_op(
  Q: torch.Tensor,
  K: torch.Tensor,
  V: torch.Tensor,
  stages: int,
  acc: int,
  causal: int,
  softmax_scale: float,
  tma: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  if _ffpa_attn_fwd_cuda is None:
    raise RuntimeError(
      "ffpa_attn forward CUDA backend is unavailable. "
      "Rebuild with ENABLE_FFPA_FWD_CUDA_IMPL=1 to enable it. "
      f"Original import error: {_CUDA_IMPORT_ERROR}"
    )
  O = torch.empty_like(Q)  # noqa: E741
  seqlen_q = Q.size(2)
  seqlen_q_aligned = ((seqlen_q + 7) // 8) * 8
  softmax_lse = torch.empty(
    Q.size(0),
    Q.size(1),
    seqlen_q_aligned,
    dtype=torch.float32,
    device=Q.device,
  )
  _ffpa_attn_fwd_cuda(
    Q,
    K,
    V,
    O,
    softmax_lse[..., :seqlen_q],
    stages,
    acc,
    causal,
    softmax_scale,
    tma,
  )
  return O, softmax_lse


@torch.library.register_fake(f"{_OP_NAMESPACE}::_fwd_cuda")
def _fwd_cuda_fake(
  Q: torch.Tensor,
  K: torch.Tensor,
  V: torch.Tensor,
  stages: int,
  acc: int,
  causal: int,
  softmax_scale: float,
  tma: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  seqlen_q_aligned = ((Q.size(2) + 7) // 8) * 8
  O = torch.empty_like(Q)  # noqa: E741
  softmax_lse = Q.new_empty(Q.size(0), Q.size(1), seqlen_q_aligned, dtype=torch.float32)
  return O, softmax_lse


# ffpa_attn::_bwd_cuda
torch.library.define(
  f"{_OP_NAMESPACE}::_bwd_cuda",
  "(Tensor q, Tensor k, Tensor v, Tensor o, Tensor softmax_lse, Tensor dO, "
  "int stages, int causal, float softmax_scale) -> (Tensor dq, Tensor dk, Tensor dv)",
)


@torch.library.impl(f"{_OP_NAMESPACE}::_bwd_cuda", "CUDA")
def _bwd_cuda_torch_op(
  Q: torch.Tensor,
  K: torch.Tensor,
  V: torch.Tensor,
  O: torch.Tensor,
  softmax_lse: torch.Tensor,
  dO: torch.Tensor,
  stages: int,
  causal: int,
  softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  if _ffpa_attn_bwd_cuda is None:
    raise RuntimeError(
      "ffpa_attn backward CUDA backend is unavailable. "
      "Rebuild with ENABLE_FFPA_FWD_CUDA_IMPL=1 and ENABLE_FFPA_BWD_CUDA_IMPL=1 to enable it. "
      f"Original import error: {_CUDA_IMPORT_ERROR}"
    )
  dQ = torch.zeros_like(Q)
  dK = torch.zeros_like(K)
  dV = torch.zeros_like(V)
  _ffpa_attn_bwd_cuda(Q, K, V, O, softmax_lse, dO, dQ, dK, dV, stages, causal, softmax_scale)
  return dQ, dK, dV


@torch.library.register_fake(f"{_OP_NAMESPACE}::_bwd_cuda")
def _bwd_cuda_fake(
  Q: torch.Tensor,
  K: torch.Tensor,
  V: torch.Tensor,
  O: torch.Tensor,
  softmax_lse: torch.Tensor,
  dO: torch.Tensor,
  stages: int,
  causal: int,
  softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  return torch.empty_like(Q), torch.empty_like(K), torch.empty_like(V)


__all__ = [
  "_ffpa_attn_forward_cuda",
  "_ffpa_attn_backward_cuda",
  "CUDA_FWD_AVAILABLE",
  "CUDA_BWD_AVAILABLE",
]

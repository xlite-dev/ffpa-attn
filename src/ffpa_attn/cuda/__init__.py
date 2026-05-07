"""CUDA FFPA attention forward/backward implementations for large-D (D > 256)."""
import torch

from ._ffpa_bwd import _ffpa_attn_backward_cuda
from ._ffpa_fwd import _ffpa_attn_forward_cuda, _ffpa_attn_fwd_cuda

_OP_NAMESPACE = "ffpa_attn"
_OP_NAME = "attn"
_OP_QUALNAME = f"{_OP_NAMESPACE}::{_OP_NAME}"

# The op mutates ``O`` and ``softmax_lse`` in place and returns O for
# convenience. The ``(a!)`` alias annotations tell torch.library the
# buffers are written, required for correct alias/functionalization under
# torch.compile.
torch.library.define(
  _OP_QUALNAME,
  "(Tensor Q, Tensor K, Tensor V, Tensor(a!) O, Tensor(b!) softmax_lse, int stages, int acc, "
  "int causal, float softmax_scale, int tma) -> Tensor(a!)",
)


@torch.library.impl(_OP_QUALNAME, "CUDA")
def _ffpa_attn_impl_cuda(
  Q: torch.Tensor,
  K: torch.Tensor,
  V: torch.Tensor,
  O: torch.Tensor,
  softmax_lse: torch.Tensor,
  stages: int,
  acc: int,
  causal: int,
  softmax_scale: float,
  tma: int,
) -> torch.Tensor:
  _ffpa_attn_fwd_cuda(Q, K, V, O, softmax_lse, stages, acc, causal, softmax_scale, tma)
  return O


@torch.library.register_fake(_OP_QUALNAME)
def _ffpa_attn_impl_fake(
  Q: torch.Tensor,
  K: torch.Tensor,
  V: torch.Tensor,
  O: torch.Tensor,
  softmax_lse: torch.Tensor,
  stages: int,
  acc: int,
  causal: int,
  softmax_scale: float,
  tma: int,
) -> torch.Tensor:
  del Q, K, V, stages, acc, causal, softmax_scale, tma
  return O


__all__ = ["_ffpa_attn_forward_cuda", "_ffpa_attn_backward_cuda"]

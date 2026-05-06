from __future__ import annotations

import torch

from .._C import ffpa_attn_backward as _ffpa_attn_bwd_cuda


def _ffpa_attn_backward_cuda(
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
  """Call FFPA CUDA backward, returning ``(dQ, dK, dV)``.

  :param Q: Query tensor saved from forward.
  :param K: Key tensor saved from forward.
  :param V: Value tensor saved from forward.
  :param O: Output tensor saved from forward.
  :param softmax_lse: Softmax log-sum-exp tensor saved from forward.
  :param dO: Output gradient tensor.
  :param stages: CUDA backward pipeline stages.
  :param causal: Whether lower-right causal masking was applied.
  :param softmax_scale: Scale applied to ``QK^T``.
  :returns: Gradients for ``Q``, ``K``, and ``V``.
  """
  dQ = torch.zeros_like(Q)
  dK = torch.zeros_like(K)
  dV = torch.zeros_like(V)
  _ffpa_attn_bwd_cuda(
    Q,
    K,
    V,
    O,
    softmax_lse,
    dO,
    dQ,
    dK,
    dV,
    stages,
    causal,
    softmax_scale,
  )
  return dQ, dK, dV

from __future__ import annotations

import torch


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
  """Call FFPA CUDA backward via registered torch op, returning ``(dQ, dK, dV)``.

    :returns: Gradients for ``Q``, ``K``, and ``V``.
    """
  return torch.ops.ffpa_attn._bwd_cuda(
    Q,
    K,
    V,
    O,
    softmax_lse,
    dO,
    stages,
    causal,
    softmax_scale,
  )

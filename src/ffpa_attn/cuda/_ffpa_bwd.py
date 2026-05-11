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
  """Raise for the removed native CUDA backward backend.

  The function signature is kept for compatibility with older imports.
  Active backward development is on the Triton backend.
  """
  del Q, K, V, O, softmax_lse, dO, stages, causal, softmax_scale
  raise RuntimeError(
    "ffpa_attn CUDA backward has been removed from the active backend. "
    "Use backward_backend='triton' or backward_backend='sdpa'."
  )

from __future__ import annotations

import torch


def _ffpa_attn_forward_cuda(
  Q: torch.Tensor,
  K: torch.Tensor,
  V: torch.Tensor,
  O: torch.Tensor | None = None,
  attn_bias: torch.Tensor | None = None,
  stages: int = 2,
  acc: int = 1,
  causal: int = 0,
  softmax_scale: float = 0.0,
  tma: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Call FFPA CUDA forward via registered torch op, returning ``(O, softmax_lse)``.

  The ``O`` parameter is accepted for API compatibility but ignored — the
  registered op always allocates a fresh output buffer.
  The ``tma`` parameter is also accepted for API compatibility; legacy CUDA
  TMA dispatch has been removed from the active backend and is forced off.

  :returns: Output tensor and softmax LSE sliced to visible shape ``[B, H, Nq]``.
  """
  del O
  del tma
  if attn_bias is None:
    attn_bias = Q.new_empty((0, ))
  O_storage, softmax_lse_storage = torch.ops.ffpa_attn._fwd_cuda(
    Q,
    K,
    V,
    attn_bias,
    stages,
    acc,
    causal,
    softmax_scale,
    0,
  )
  return O_storage, softmax_lse_storage[..., :Q.size(2)]

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
  dropout_p: float = 0.0,
  philox_seed: int = 0,
  philox_offset: int = 0,
  smooth_k: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Call FFPA CUDA forward via registered torch op, returning ``(O, softmax_lse)``.

  :returns: Output tensor and softmax LSE sliced to visible shape ``[B, H, Nq]``.
  """
  del O
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
    dropout_p,
    philox_seed,
    philox_offset,
    smooth_k,
  )
  return O_storage, softmax_lse_storage[..., :Q.size(2)]

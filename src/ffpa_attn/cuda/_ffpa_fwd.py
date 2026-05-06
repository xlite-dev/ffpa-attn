from __future__ import annotations

import torch

from .._C import ffpa_attn_forward as _ffpa_attn_fwd_cuda


def _ffpa_attn_forward_cuda(
  Q: torch.Tensor,
  K: torch.Tensor,
  V: torch.Tensor,
  O: torch.Tensor | None = None,
  stages: int = 2,
  acc: int = 1,
  causal: int = 0,
  softmax_scale: float = 0.0,
  tma: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Call FFPA CUDA forward, returning ``(O, softmax_lse)``.

  :param Q: Query tensor in ``[B, Hq, Nq, D]`` layout.
  :param K: Key tensor in ``[B, Hkv, Nkv, D]`` layout.
  :param V: Value tensor in ``[B, Hkv, Nkv, D]`` layout.
  :param O: Optional output buffer. If ``None``, a zeroed buffer is allocated.
  :param stages: CUDA forward pipeline stages.
  :param acc: Native CUDA accumulator code.
  :param causal: Whether to apply lower-right causal masking.
  :param softmax_scale: Scale applied to ``QK^T``.
  :param tma: Whether to request the CUDA TMA path.
  :returns: Output tensor and visible softmax LSE view.
  """
  if O is None:
    O = torch.zeros_like(Q)  # noqa: E741
  seqlen_q = Q.size(2)
  seqlen_q_aligned = ((seqlen_q + 7) // 8) * 8
  softmax_lse_storage = torch.empty(Q.size(0), Q.size(1), seqlen_q_aligned, dtype=torch.float32, device=Q.device)
  softmax_lse = softmax_lse_storage[..., :seqlen_q]
  _ffpa_attn_fwd_cuda(Q, K, V, O, softmax_lse, stages, acc, causal, softmax_scale, tma)
  return O, softmax_lse

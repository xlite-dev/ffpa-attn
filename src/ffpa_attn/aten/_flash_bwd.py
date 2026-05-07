"""Small-D flash-attention backward wrapper using PyTorch's SDPA backward op."""

from __future__ import annotations

import torch


def _aten_flash_attn_backward(
  grad_out: torch.Tensor,
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  o: torch.Tensor,
  lse: torch.Tensor,
  causal: bool,
  rng_state: torch.Tensor,
  unused: torch.Tensor,
  softmax_scale: float,
  dropout_p: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Run the small-D path through PyTorch's flash-attention backward wrapper.

  Delegates to ``torch.ops.aten._scaled_dot_product_flash_attention_backward``
  which is the backward counterpart of ``_flash_attention_forward`` used in the
  small-D forward path.

  :param grad_out: Upstream gradient ``[B, Nh_q, Nq, D]``.
  :param q: Query tensor ``[B, Nh_q, Nq, D]``.
  :param k: Key tensor ``[B, Nh_kv, Nkv, D]``.
  :param v: Value tensor ``[B, Nh_kv, Nkv, D]``.
  :param o: Attention output ``[B, Nh_q, Nq, D]``.
  :param lse: Log-sum-exp from the forward pass.
  :param causal: Whether causal masking was applied.
  :param rng_state: RNG state tensor from the forward pass.
  :param unused: Unused placeholder tensor from the forward pass.
  :param softmax_scale: Pre-softmax scaling factor.
  :param dropout_p: Dropout probability (default 0.0). Must match forward.

  :returns: Tuple ``(dq, dk, dv)``, gradients w.r.t. ``q``, ``k``, ``v``.
  """
  return torch.ops.aten._scaled_dot_product_flash_attention_backward.default(
    grad_out.contiguous(),
    q.contiguous(),
    k.contiguous(),
    v.contiguous(),
    o.contiguous(),
    lse.contiguous(),
    None,
    None,
    q.size(2),
    k.size(2),
    dropout_p,
    causal,
    rng_state.contiguous(),
    unused.contiguous(),
    scale=softmax_scale,
  )

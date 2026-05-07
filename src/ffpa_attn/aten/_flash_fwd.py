"""Small-D flash-attention forward wrapper using the exact aten op."""

from __future__ import annotations

import torch


def _aten_flash_attn_forward(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  o: torch.Tensor | None,
  causal: bool,
  softmax_scale: float,
  dropout_p: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Run the small-D path through the exact aten flash-attention forward op.

  The aten op expects BNHD layout, so inputs are transposed from BHND before
  calling and the output is transposed back.

  :param q: Query tensor, ``[B, Nh_q, Nq, D]``.
  :param k: Key tensor, ``[B, Nh_kv, Nkv, D]``.
  :param v: Value tensor, ``[B, Nh_kv, Nkv, D]``.
  :param o: Optional pre-allocated output tensor, ``[B, Nh_q, Nq, D]``.
  :param causal: Apply causal masking.
  :param softmax_scale: Pre-softmax scaling factor.
  :param dropout_p: Dropout probability (default 0.0). Only effective when > 0.

  :returns: Tuple ``(out, lse, rng_state, unused)`` where ``out`` is
      ``[B, Nh_q, Nq, D]`` in BHND layout.
  """
  q_bnhd = q.transpose(1, 2).contiguous()
  k_bnhd = k.transpose(1, 2).contiguous()
  v_bnhd = v.transpose(1, 2).contiguous()
  out_bnhd, lse, rng_state, unused, _ = torch.ops.aten._flash_attention_forward(
    q_bnhd,
    k_bnhd,
    v_bnhd,
    None,
    None,
    q.size(2),
    k.size(2),
    dropout_p,
    causal,
    False,
    scale=softmax_scale,
  )
  out = out_bnhd.transpose(1, 2).contiguous()
  if o is None:
    return out, lse, rng_state, unused

  o.copy_(out)
  return o, lse, rng_state, unused

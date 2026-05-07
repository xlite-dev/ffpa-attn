"""Large-D efficient-attention backward wrapper using PyTorch's SDPA op.

This module backs ``FFPAAttnFunc.backward(backward_backend="sdpa")`` for the
large-D path. It hides the layout, alignment, precision, and GQA plumbing
required by ``torch.ops.aten._scaled_dot_product_efficient_attention_backward``
so the autograd Function can stay dispatch-only.
"""

from __future__ import annotations

import torch


def _reduce_expanded_kv_grads(
  grad_k: torch.Tensor,
  grad_v: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Reduce expanded GQA/MQA gradients back to the original KV head layout.

  :param grad_k: Expanded key gradient with query-head layout.
  :param grad_v: Expanded value gradient with query-head layout.
  :param k: Original key tensor whose shape and dtype define the target layout.
  :param v: Original value tensor whose shape and dtype define the target layout.
  :param group_size: Query-heads per KV head.
  :returns: ``(dk, dv)`` reduced to the original KV head layout and dtype.
  """
  if group_size == 1:
    return grad_k.to(k.dtype), grad_v.to(v.dtype)

  reduced_k = grad_k.reshape(
    k.size(0),
    k.size(1),
    group_size,
    k.size(2),
    k.size(3),
  ).sum(dim=2)
  reduced_v = grad_v.reshape(
    v.size(0),
    v.size(1),
    group_size,
    v.size(2),
    v.size(3),
  ).sum(dim=2)
  return reduced_k.to(k.dtype), reduced_v.to(v.dtype)


def _aten_efficient_attn_backward(
  grad_out: torch.Tensor,
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  o: torch.Tensor,
  lse: torch.Tensor,
  causal: bool,
  softmax_scale: float,
  high_precision_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Run large-D backward through PyTorch's efficient-attention backward op.

  This wrapper owns the FFPA-specific tensor preparation required by
  ``torch.ops.aten._scaled_dot_product_efficient_attention_backward.default``:
  output-stride repair, LSE alignment, optional fp32 upcast for
  ``high_precision_grad``, and GQA/MQA expand-reduce handling.

  :param grad_out: Upstream output gradient ``[B, Nh_q, Nq, D]``.
  :param q: Query tensor saved from forward.
  :param k: Key tensor saved from forward.
  :param v: Value tensor saved from forward.
  :param o: Forward output tensor saved from the FFPA forward path.
  :param lse: Forward log-sum-exp tensor saved from the FFPA forward path.
  :param causal: Whether lower-right causal masking was used in forward.
  :param softmax_scale: Scale applied to ``QK^T``.
  :param high_precision_grad: Whether to upcast inputs and intermediates to
    fp32 before calling the aten efficient backward op.
  :returns: ``(dq, dk, dv)`` with the original ``q`` / ``k`` / ``v`` dtypes and
    head layouts.
  """
  group_size = q.size(1) // k.size(1)
  attn_bias_mask = None
  causal_for_op = causal

  if causal and k.size(2) != q.size(2):
    kv_offset = k.size(2) - q.size(2)
    row_idx = torch.arange(q.size(2), device=q.device).view(-1, 1)
    col_idx = torch.arange(k.size(2), device=k.device).view(1, -1)
    attn_bias_mask = col_idx <= (row_idx + kv_offset)
    causal_for_op = False

  o = o.transpose(1, 2).contiguous().transpose(1, 2)
  if lse.size(1) > 1 and (lse.stride(1) % 8) != 0:
    seqlen_q_aligned = ((lse.size(-1) + 7) // 8) * 8
    lse_padded = torch.empty(
      *lse.shape[:-1],
      seqlen_q_aligned,
      dtype=lse.dtype,
      device=lse.device,
    )
    lse_padded[..., :lse.size(-1)] = lse
    lse = lse_padded

  if high_precision_grad:
    q_in = q.float()
    k_in = k.float()
    v_in = v.float()
    o_in = o.float()
    lse_in = lse.float()
    grad_out_in = grad_out.float()
  else:
    q_in, k_in, v_in = q, k, v
    o_in, lse_in, grad_out_in = o, lse, grad_out

  attn_bias = None
  if attn_bias_mask is not None:
    attn_bias = torch.zeros(
      q.size(0),
      q_in.size(1),
      q.size(2),
      k.size(2),
      dtype=q_in.dtype,
      device=q.device,
    )
    attn_bias.masked_fill_(~attn_bias_mask.view(1, 1, q.size(2), k.size(2)), float("-inf"))

  if group_size > 1:
    k_in = k_in.repeat_interleave(group_size, dim=1).contiguous()
    v_in = v_in.repeat_interleave(group_size, dim=1).contiguous()

  zero_u64 = torch.zeros(2, dtype=torch.uint64, device=q.device)
  philox_seed = zero_u64[0].unsqueeze(0)
  philox_offset = zero_u64[1].unsqueeze(0)
  dq, dk_expanded, dv_expanded, _ = (
    torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(
      grad_out_in,
      q_in,
      k_in,
      v_in,
      attn_bias,
      o_in,
      lse_in,
      philox_seed,
      philox_offset,
      0.0,
      (True, True, True, False),
      causal_for_op,
      scale=softmax_scale,
    )
  )
  dk, dv = _reduce_expanded_kv_grads(dk_expanded, dv_expanded, k, v, group_size)
  return dq.to(q.dtype), dk, dv


__all__ = ["_aten_efficient_attn_backward"]

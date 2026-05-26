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


def _efficient_attn_backward_aten(
  grad_out: torch.Tensor,
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  o: torch.Tensor,
  lse: torch.Tensor,
  causal: bool,
  softmax_scale: float,
  high_precision_grad: bool = False,
  attn_bias: torch.Tensor | None = None,
  return_attn_bias_grad: bool = False,
  dropout_p: float = 0.0,
  philox_seed: int = 0,
  philox_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
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
  :param attn_bias: Optional additive attention bias broadcast to
    ``[B, Nh_q, Nq, Nkv]``. Boolean masks must already be converted to
    additive bias before entering this wrapper.
  :param return_attn_bias_grad: Whether to request the additive-bias gradient
    from the aten op.
  :param dropout_p: Dropout probability used by the forward pass.
  :param philox_seed: Philox seed saved from the forward pass.
  :param philox_offset: Philox offset saved from the forward pass.
  :returns: ``(dq, dk, dv, d_attn_bias)`` with the original ``q`` / ``k`` /
    ``v`` dtypes and head layouts. ``d_attn_bias`` is ``None`` unless
    requested for an explicit additive bias.
  """
  group_size = q.size(1) // k.size(1)
  attn_bias_mask = None
  causal_for_op = causal
  original_attn_bias = attn_bias

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

  if attn_bias_mask is not None:
    attn_bias = torch.zeros(
      q.size(0),
      q_in.size(1),
      q.size(2),
      k.size(2),
      dtype=q_in.dtype,
      device=q.device,
    )
    attn_bias.masked_fill_(
      ~attn_bias_mask.view(1, 1, q.size(2), k.size(2)), float("-inf")
    )
  elif attn_bias is not None:
    attn_bias = attn_bias.to(dtype=q_in.dtype)
    bias_shape = (q.size(0), q_in.size(1), q.size(2), k.size(2))
    if attn_bias.shape != bias_shape:
      # Keep broadcast dimensions as zero-stride views. Materializing here can
      # turn compact masks such as [1, 1, 1, N] into [B, H, N, N].
      attn_bias = attn_bias.expand(bias_shape)

  if group_size > 1:
    k_in = k_in.repeat_interleave(group_size, dim=1).contiguous()
    v_in = v_in.repeat_interleave(group_size, dim=1).contiguous()

  if dropout_p > 0.0:
    philox_seed_t = torch.tensor([philox_seed], dtype=torch.int64)
    philox_offset_t = torch.tensor([philox_offset], dtype=torch.int64)
  else:
    zero_i64 = torch.zeros(2, dtype=torch.int64)
    philox_seed_t = zero_i64[0].unsqueeze(0)
    philox_offset_t = zero_i64[1].unsqueeze(0)
  dq, dk_expanded, dv_expanded, grad_attn_bias = (
    torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(
      grad_out_in,
      q_in,
      k_in,
      v_in,
      attn_bias,
      o_in,
      lse_in,
      philox_seed_t,
      philox_offset_t,
      dropout_p,
      (
        True, True, True,
        bool(return_attn_bias_grad and original_attn_bias is not None)
      ),
      causal_for_op,
      scale=softmax_scale,
    )
  )
  dk, dv = _reduce_expanded_kv_grads(dk_expanded, dv_expanded, k, v, group_size)
  if not return_attn_bias_grad or original_attn_bias is None:
    grad_attn_bias = None
  else:
    grad_attn_bias = grad_attn_bias.sum_to_size(original_attn_bias.shape
                                                ).to(original_attn_bias.dtype)
  return dq.to(q.dtype), dk, dv, grad_attn_bias


__all__ = ["_efficient_attn_backward_aten"]

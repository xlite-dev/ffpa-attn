"""FFPA CuTeDSL backward pass scaffold for SM80/SM89 Split-D kernels."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

from ..triton import _ffpa_attn_backward_triton
from ._utils import (
  maybe_contiguous,
  _resolve_causal_local_window,
  _validate_max_seqlen_for_cu_seqlens,
  _validate_qkv_common,
  _validate_sm80_arch,
  _validate_sm80_head_dims,
  _validate_tensor,
)


def _ffpa_attn_backward_sm80(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  out: torch.Tensor,
  dout: torch.Tensor,
  lse: torch.Tensor,
  softmax_scale: Optional[float] = None,
  causal: bool = False,
  softcap: float = 0.0,
  window_size_left: Optional[int] = None,
  window_size_right: Optional[int] = None,
  cu_seqlens_q: Optional[torch.Tensor] = None,
  cu_seqlens_k: Optional[torch.Tensor] = None,
  max_seqlen_q: Optional[int] = None,
  max_seqlen_k: Optional[int] = None,
  dq: Optional[torch.Tensor] = None,
  dk: Optional[torch.Tensor] = None,
  dv: Optional[torch.Tensor] = None,
  dlse: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """SM80/SM89 Split-D backward launcher.

  Dense tensors are adapted to the existing Triton Split-D backward kernel while
  the native SM80 CuTeDSL backward kernel is developed. Packed varlen tensors use
  a per-segment PyTorch SDPA backward fallback to preserve API compatibility.

  :param q: Query tensor saved from forward.
  :param k: Key tensor saved from forward.
  :param v: Value tensor saved from forward.
  :param out: Forward output tensor.
  :param dout: Gradient with respect to ``out``.
  :param lse: Forward log-sum-exp tensor.
  :param softmax_scale: Attention scale.
  :param causal: Whether lower-right causal masking is applied.
  :param softcap: Unsupported for the SM80 Split-D path.
  :param window_size_left: Unsupported local-attention left window.
  :param window_size_right: Unsupported local-attention right window.
  :param cu_seqlens_q: Optional packed-query sequence offsets.
  :param cu_seqlens_k: Optional packed-key sequence offsets.
  :param max_seqlen_q: Maximum query sequence length for varlen inputs.
  :param max_seqlen_k: Maximum key sequence length for varlen inputs.
  :param dq: Optional preallocated query gradient.
  :param dk: Optional preallocated key gradient.
  :param dv: Optional preallocated value gradient.
  :param dlse: Optional gradient with respect to LSE.
  :returns: ``(dq, dk, dv)`` matching the SM90 launcher contract.
  """
  _validate_sm80_arch()
  if softcap != 0.0:
    raise NotImplementedError("SM80/SM89 backward does not support softcap")
  if dlse is not None:
    raise NotImplementedError("SM80/SM89 backward does not support dlse yet")
  causal, local, window_size_left, window_size_right = _resolve_causal_local_window(
    causal, window_size_left, window_size_right
  )
  if local:
    raise NotImplementedError(
      "SM80/SM89 backward does not support local/window attention yet"
    )

  q, k, v, out, dout, lse, cu_seqlens_q, cu_seqlens_k = [
    maybe_contiguous(t)
    for t in (q, k, v, out, dout, lse, cu_seqlens_q, cu_seqlens_k)
  ]
  (
    batch_size,
    seqlen_q,
    total_q,
    seqlen_k,
    num_head,
    _num_head_kv,
    head_dim,
    head_dim_v,
  ) = _validate_qkv_common(
    q,
    k,
    v,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_k=cu_seqlens_k,
    validate_head_dims=_validate_sm80_head_dims,
  )
  _validate_max_seqlen_for_cu_seqlens(
    cu_seqlens_q, "cu_seqlens_q", max_seqlen_q, "max_seqlen_q"
  )
  _validate_max_seqlen_for_cu_seqlens(
    cu_seqlens_k, "cu_seqlens_k", max_seqlen_k, "max_seqlen_k"
  )
  if softmax_scale is None:
    softmax_scale = 1.0 / math.sqrt(head_dim)

  if dq is None:
    dq = torch.empty_like(q)
  if dk is None:
    dk = torch.empty_like(k)
  if dv is None:
    dv = torch.empty_like(v)

  if cu_seqlens_q is not None or cu_seqlens_k is not None:
    if cu_seqlens_q is None or cu_seqlens_k is None:
      raise ValueError(
        "SM80/SM89 varlen backward requires both cu_seqlens_q and cu_seqlens_k"
      )
    _validate_tensor(
      out, "out", (total_q, num_head, head_dim_v), q.dtype, q.device
    )
    _validate_tensor(
      dout, "dout", (total_q, num_head, head_dim_v), q.dtype, q.device
    )
    _validate_tensor(lse, "lse", (num_head, total_q), torch.float32, q.device)
    dq.zero_()
    dk.zero_()
    dv.zero_()
    for batch_idx in range(cu_seqlens_q.numel() - 1):
      q_start = int(cu_seqlens_q[batch_idx].item())
      q_end = int(cu_seqlens_q[batch_idx + 1].item())
      k_start = int(cu_seqlens_k[batch_idx].item())
      k_end = int(cu_seqlens_k[batch_idx + 1].item())
      if q_end == q_start:
        continue
      if k_end == k_start:
        dq[q_start:q_end].zero_()
        continue
      dq_bhnd, dk_bhnd, dv_bhnd, _ = _ffpa_attn_backward_triton(
        dout[q_start:q_end].transpose(0, 1).unsqueeze(0).contiguous(),
        q[q_start:q_end].transpose(0, 1).unsqueeze(0).contiguous(),
        k[k_start:k_end].transpose(0, 1).unsqueeze(0).contiguous(),
        v[k_start:k_end].transpose(0, 1).unsqueeze(0).contiguous(),
        out[q_start:q_end].transpose(0, 1).unsqueeze(0).contiguous(),
        lse[:, q_start:q_end].unsqueeze(0).contiguous(),
        causal=causal,
        softmax_scale=softmax_scale,
        autotune=False,
        autotune_mode="fast",
      )
      dq[q_start:q_end].copy_(dq_bhnd.squeeze(0).transpose(0, 1))
      dk[k_start:k_end].copy_(dk_bhnd.squeeze(0).transpose(0, 1))
      dv[k_start:k_end].copy_(dv_bhnd.squeeze(0).transpose(0, 1))
    return dq, dk, dv

  q_batch_seqlen_shape = (batch_size, seqlen_q)
  _validate_tensor(
    out, "out", (*q_batch_seqlen_shape, num_head, head_dim_v), q.dtype, q.device
  )
  _validate_tensor(
    dout, "dout", (*q_batch_seqlen_shape, num_head, head_dim_v), q.dtype,
    q.device
  )
  _validate_tensor(
    lse, "lse", (batch_size, num_head, seqlen_q), torch.float32, q.device
  )

  dq_bhnd, dk_bhnd, dv_bhnd, _ = _ffpa_attn_backward_triton(
    dout.transpose(1, 2).contiguous(),
    q.transpose(1, 2).contiguous(),
    k.transpose(1, 2).contiguous(),
    v.transpose(1, 2).contiguous(),
    out.transpose(1, 2).contiguous(),
    lse,
    causal=causal,
    softmax_scale=softmax_scale,
    autotune=False,
    autotune_mode="fast",
  )
  dq.copy_(dq_bhnd.transpose(1, 2))
  dk.copy_(dk_bhnd.transpose(1, 2))
  dv.copy_(dv_bhnd.transpose(1, 2))
  return dq, dk, dv


__all__ = ["_ffpa_attn_backward_sm80"]

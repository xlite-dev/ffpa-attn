"""CuTeDSL FFPA backend: SM90 + D=512 specialised forward/backward.

Exposes the dense and varlen CuTeDSL entry shims used by
:mod:`ffpa_attn.ffpa_attn_interface` and :mod:`ffpa_attn.functional`.

Dense-path ops ``ffpa_attn::_fwd_cutedsl`` / ``ffpa_attn::_bwd_cutedsl``
and varlen-path ops ``ffpa_attn::_varlen_fwd_cutedsl`` /
``ffpa_attn::_varlen_bwd_cutedsl`` are registered below so both paths
are ``torch.compile``-compatible.
"""

import math
from typing import Optional, Tuple, Callable

import torch

from ._interface import (
  _decode_custom_op_window,
  _encode_optional_int_for_custom_op,
  _ffpa_attn_backward_sm90,
  _ffpa_attn_forward_sm90,
  _validate_max_seqlen_for_cu_seqlens,
  _validate_qkv_common,
  _validate_tensor,
  _validate_training_dtype,
  _validate_varlen_custom_bwd_features,
  _validate_varlen_custom_fwd_features,
  is_fake_mode,
)
from ._wrappers import (
  _ffpa_attn_forward_cutedsl,
  _ffpa_attn_backward_cutedsl,
  _ffpa_attn_varlen_cutedsl,
  _require_cutedsl_supported,
  cutedsl_backward_available,
  cutedsl_forward_available,
)

__all__ = [
  "_ffpa_attn_forward_cutedsl",
  "_ffpa_attn_backward_cutedsl",
  "_ffpa_attn_varlen_cutedsl",
  "ffpa_attn_splitd_varlen_func",
  "_require_cutedsl_supported",
  "cutedsl_forward_available",
  "cutedsl_backward_available",
]

# ---------------------------------------------------------------------------
# Dense torch custom ops — torch.compile-compatible entry for the CuTeDSL
# SplitD D=512 forward/backward.  Follows the ``triton/__init__.py``
# ``torch.library.define`` + ``@impl("CUDA")`` + ``@register_fake``
# pattern.  No ``register_autograd``: backward is managed by
# ``_FFPAAttnFunc`` in ``functional.py``.
# ---------------------------------------------------------------------------

torch.library.define(
  "ffpa_attn::_fwd_cutedsl",
  "(Tensor q, Tensor k, Tensor v, float softmax_scale, int causal, int return_lse) "
  "-> (Tensor o, Tensor lse)",
)


@torch.library.impl("ffpa_attn::_fwd_cutedsl", "CUDA")
def _fwd_cutedsl_torch_op(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  softmax_scale: float,
  causal: int,
  return_lse: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  from ._interface import _ffpa_attn_forward_sm90

  batch, seqlen_q, num_head, head_dim_v = q.shape
  o = torch.empty(batch, seqlen_q, num_head, head_dim_v, dtype=q.dtype, device=q.device)
  need_lse = bool(return_lse)
  lse = (
    torch.empty(batch, num_head, seqlen_q, dtype=torch.float32, device=q.device)
    if need_lse else torch.empty(0, device=q.device)
  )
  _ffpa_attn_forward_sm90(
    q,
    k,
    v,
    softmax_scale=softmax_scale,
    causal=bool(causal),
    return_lse=need_lse,
    out=o,
    lse=lse if need_lse else None,
  )
  return o, lse


@torch.library.register_fake("ffpa_attn::_fwd_cutedsl")
def _fwd_cutedsl_fake(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  softmax_scale: float,
  causal: int,
  return_lse: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  o = torch.empty_like(q)
  lse = (q.new_empty(q.size(0), q.size(-2), q.size(-3), dtype=torch.float32) if return_lse else q.new_empty(0))
  return o, lse


torch.library.define(
  "ffpa_attn::_bwd_cutedsl",
  "(Tensor dout, Tensor q, Tensor k, Tensor v, Tensor out, Tensor lse, "
  "float softmax_scale, int causal) -> (Tensor dq, Tensor dk, Tensor dv)",
)


@torch.library.impl("ffpa_attn::_bwd_cutedsl", "CUDA")
def _bwd_cutedsl_torch_op(
  dout: torch.Tensor,
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  out: torch.Tensor,
  lse: torch.Tensor,
  softmax_scale: float,
  causal: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  from ._interface import _ffpa_attn_backward_sm90

  dq = torch.empty_like(q)
  dk = torch.empty_like(k)
  dv = torch.empty_like(v)
  _ffpa_attn_backward_sm90(
    q,
    k,
    v,
    out,
    dout,
    lse,
    softmax_scale=softmax_scale,
    causal=bool(causal),
    dq=dq,
    dk=dk,
    dv=dv,
  )
  return dq, dk, dv


@torch.library.register_fake("ffpa_attn::_bwd_cutedsl")
def _bwd_cutedsl_fake(
  dout: torch.Tensor,
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  out: torch.Tensor,
  lse: torch.Tensor,
  softmax_scale: float,
  causal: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


# ---------------------------------------------------------------------------
# Varlen torch custom ops — ``@custom_op`` + ``@register_fake`` +
# ``register_autograd`` pattern.  The varlen path owns its own autograd
# boundary (unlike dense, which delegates to ``_FFPAAttnFunc``).
# ---------------------------------------------------------------------------


def _trim_trailing_empty_varlen_segments(
  cu_seqlens_q: torch.Tensor,
  cu_seqlens_k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Drop trailing segments where both q and k have zero length."""
  if cu_seqlens_q.numel() != cu_seqlens_k.numel():
    return cu_seqlens_q, cu_seqlens_k
  if cu_seqlens_q.numel() <= 1 or is_fake_mode():
    return cu_seqlens_q, cu_seqlens_k

  q_lengths = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
  k_lengths = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
  active = (q_lengths != 0) | (k_lengths != 0)
  if bool(active.all().item()):
    return cu_seqlens_q, cu_seqlens_k
  if not bool(active.any().item()):
    return cu_seqlens_q[:1], cu_seqlens_k[:1]

  last_active_segment = int(active.nonzero()[-1].item())
  keep_numel = last_active_segment + 2
  if keep_numel == cu_seqlens_q.numel():
    return cu_seqlens_q, cu_seqlens_k
  return cu_seqlens_q[:keep_numel], cu_seqlens_k[:keep_numel]


@torch.library.custom_op("ffpa_attn::_varlen_fwd_cutedsl", mutates_args=())
def _varlen_fwd_custom(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  cu_seqlens_q: torch.Tensor,
  cu_seqlens_k: torch.Tensor,
  max_seqlen_q: int,
  max_seqlen_k: int,
  softmax_scale: float,
  causal: bool,
  window_size_left: int,
  window_size_right: int,
  softcap: float,
  pack_gqa: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
  cu_seqlens_q, cu_seqlens_k = _trim_trailing_empty_varlen_segments(cu_seqlens_q, cu_seqlens_k)
  window_size_left_opt, window_size_right_opt = _decode_custom_op_window(window_size_left, window_size_right)
  return _ffpa_attn_forward_sm90(
    q,
    k,
    v,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_k=cu_seqlens_k,
    max_seqlen_q=max_seqlen_q,
    max_seqlen_k=max_seqlen_k,
    softmax_scale=softmax_scale,
    causal=causal,
    window_size_left=window_size_left_opt,
    window_size_right=window_size_right_opt,
    softcap=softcap,
    pack_gqa=pack_gqa,
    return_lse=True,
  )


@torch.library.register_fake("ffpa_attn::_varlen_fwd_cutedsl")
def _varlen_fwd_fake(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  cu_seqlens_q: torch.Tensor,
  cu_seqlens_k: torch.Tensor,
  max_seqlen_q: int,
  max_seqlen_k: int,
  softmax_scale: float,
  causal: bool,
  window_size_left: int,
  window_size_right: int,
  softcap: float,
  pack_gqa: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
  (
    _batch_size,
    _seqlen_q,
    total_q,
    _seqlen_k,
    num_head,
    _num_head_kv,
    _head_dim,
    head_dim_v,
  ) = _validate_qkv_common(q, k, v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k)
  _validate_training_dtype(q, k, v, q.requires_grad or k.requires_grad or v.requires_grad)
  _validate_max_seqlen_for_cu_seqlens(cu_seqlens_q, "cu_seqlens_q", max_seqlen_q, "max_seqlen_q")
  _validate_max_seqlen_for_cu_seqlens(cu_seqlens_k, "cu_seqlens_k", max_seqlen_k, "max_seqlen_k")
  _validate_varlen_custom_fwd_features(q, k, v, causal, window_size_left, window_size_right, softcap)
  out = q.new_empty((total_q, num_head, head_dim_v))
  lse = q.new_empty((num_head, total_q), dtype=torch.float32)
  return out, lse


@torch.library.custom_op("ffpa_attn::_varlen_bwd_cutedsl", mutates_args=())
def _varlen_bwd_custom(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  out: torch.Tensor,
  dout: torch.Tensor,
  lse: torch.Tensor,
  cu_seqlens_q: torch.Tensor,
  cu_seqlens_k: torch.Tensor,
  max_seqlen_q: int,
  max_seqlen_k: int,
  softmax_scale: float,
  causal: bool,
  window_size_left: int,
  window_size_right: int,
  softcap: float,
  dlse: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  cu_seqlens_q, cu_seqlens_k = _trim_trailing_empty_varlen_segments(cu_seqlens_q, cu_seqlens_k)
  window_size_left_opt, window_size_right_opt = _decode_custom_op_window(window_size_left, window_size_right)
  return _ffpa_attn_backward_sm90(
    q,
    k,
    v,
    out,
    dout,
    lse,
    softmax_scale=softmax_scale,
    causal=causal,
    softcap=softcap,
    window_size_left=window_size_left_opt,
    window_size_right=window_size_right_opt,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_k=cu_seqlens_k,
    max_seqlen_q=max_seqlen_q,
    max_seqlen_k=max_seqlen_k,
    dlse=dlse,
  )


@torch.library.register_fake("ffpa_attn::_varlen_bwd_cutedsl")
def _varlen_bwd_fake(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  out: torch.Tensor,
  dout: torch.Tensor,
  lse: torch.Tensor,
  cu_seqlens_q: torch.Tensor,
  cu_seqlens_k: torch.Tensor,
  max_seqlen_q: int,
  max_seqlen_k: int,
  softmax_scale: float,
  causal: bool,
  window_size_left: int,
  window_size_right: int,
  softcap: float,
  dlse: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  (
    _batch_size,
    _seqlen_q,
    total_q,
    _seqlen_k,
    num_head,
    _num_head_kv,
    _head_dim,
    head_dim_v,
  ) = _validate_qkv_common(q, k, v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k)
  _validate_max_seqlen_for_cu_seqlens(cu_seqlens_q, "cu_seqlens_q", max_seqlen_q, "max_seqlen_q")
  _validate_max_seqlen_for_cu_seqlens(cu_seqlens_k, "cu_seqlens_k", max_seqlen_k, "max_seqlen_k")
  if q.dtype == torch.float16:
    raise NotImplementedError(
      "SplitD backward currently supports bfloat16 only; the fp16 dQ path has a known launch failure."
    )
  _validate_varlen_custom_bwd_features(causal, window_size_left, window_size_right, softcap)
  device = q.device
  _validate_tensor(out, "out", (total_q, num_head, head_dim_v), q.dtype, device)
  _validate_tensor(dout, "dout", (total_q, num_head, head_dim_v), q.dtype, device)
  _validate_tensor(lse, "lse", (num_head, total_q), torch.float32, device)
  if dlse is not None:
    _validate_tensor(dlse, "dlse", (num_head, total_q), torch.float32, device)
  return torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)


def _varlen_fwd_setup_context(ctx, inputs, output) -> None:
  q, k, v, cu_seqlens_q, cu_seqlens_k = inputs[:5]
  max_seqlen_q, max_seqlen_k, softmax_scale, causal = inputs[5:9]
  window_size_left, window_size_right, softcap = inputs[9:12]
  out, lse = output
  ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k)
  ctx.max_seqlen_q = max_seqlen_q
  ctx.max_seqlen_k = max_seqlen_k
  ctx.softmax_scale = softmax_scale
  ctx.causal = causal
  ctx.window_size_left = window_size_left
  ctx.window_size_right = window_size_right
  ctx.softcap = softcap
  ctx.set_materialize_grads(False)


def _varlen_fwd_backward(ctx, dout, dlse):
  q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
  if dout is None:
    dout = torch.zeros_like(out)
  dq, dk, dv = torch.ops.ffpa_attn._varlen_bwd_cutedsl(
    q,
    k,
    v,
    out,
    dout,
    lse,
    cu_seqlens_q,
    cu_seqlens_k,
    ctx.max_seqlen_q,
    ctx.max_seqlen_k,
    ctx.softmax_scale,
    ctx.causal,
    ctx.window_size_left,
    ctx.window_size_right,
    ctx.softcap,
    dlse,
  )
  return dq, dk, dv, *((None, ) * 10)


torch.library.register_autograd(
  "ffpa_attn::_varlen_fwd_cutedsl",
  _varlen_fwd_backward,
  setup_context=_varlen_fwd_setup_context,
)


def _normalize_varlen_custom_op_inputs(
  q: torch.Tensor,
  k: torch.Tensor,
  cu_seqlens_q: Optional[torch.Tensor],
  cu_seqlens_k: Optional[torch.Tensor],
  max_seqlen_q: Optional[int],
  max_seqlen_k: Optional[int],
  softmax_scale: Optional[float],
  window_size: Tuple[Optional[int], Optional[int]],
  pack_gqa: Optional[bool],
  score_mod: Optional[Callable],
  aux_tensors: Optional[list],
) -> tuple[torch.Tensor, torch.Tensor, int, int, float, int, int, bool]:
  if cu_seqlens_q is None or cu_seqlens_k is None:
    raise ValueError("ffpa_attn_splitd_varlen_func custom op path requires cu_seqlens_q and cu_seqlens_k")
  if max_seqlen_q is None:
    raise ValueError("max_seqlen_q must be provided when cu_seqlens_q is provided")
  if max_seqlen_k is None:
    raise ValueError("max_seqlen_k must be provided when cu_seqlens_k is provided")
  if score_mod is not None:
    raise NotImplementedError("score_mod is not supported by the SplitD varlen custom op schema")
  if aux_tensors is not None:
    raise NotImplementedError("aux_tensors is not supported by the SplitD varlen custom op schema")
  if not isinstance(window_size, tuple) or len(window_size) != 2:
    raise TypeError("window_size must be a tuple of (left, right)")

  if softmax_scale is None:
    softmax_scale = 1.0 / math.sqrt(q.shape[-1])
  if pack_gqa is None:
    if q.ndim < 3 or k.ndim < 3:
      raise ValueError("q and k must be rank-3 packed varlen tensors")
    pack_gqa = q.shape[-2] > k.shape[-2]

  return (
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    float(softmax_scale),
    _encode_optional_int_for_custom_op(window_size[0]),
    _encode_optional_int_for_custom_op(window_size[1]),
    bool(pack_gqa),
  )


def ffpa_attn_splitd_varlen_func(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  cu_seqlens_q: Optional[torch.Tensor] = None,
  cu_seqlens_k: Optional[torch.Tensor] = None,
  max_seqlen_q: Optional[int] = None,
  max_seqlen_k: Optional[int] = None,
  softmax_scale: Optional[float] = None,
  causal: bool = False,
  window_size: Tuple[Optional[int], Optional[int]] = (None, None),
  softcap: float = 0.0,
  pack_gqa: Optional[bool] = None,
  score_mod: Optional[Callable] = None,
  aux_tensors: Optional[list] = None,
  return_lse: bool = False,
):
  """Varlen SplitD FFPA attention for D=512 on SM90.

  q/k/v must be packed as (total_tokens, heads, 512). Training requires bf16
  q/k/v, valid CUDA int32 cu_seqlens, and explicit max_seqlen_q/k whenever
  the corresponding cu_seqlens tensor is provided. If return_lse=False, LSE is
  still computed internally when needed for backward.
  """
  (
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale,
    window_size_left,
    window_size_right,
    pack_gqa,
  ) = _normalize_varlen_custom_op_inputs(
    q,
    k,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale,
    window_size,
    pack_gqa,
    score_mod,
    aux_tensors,
  )
  out, lse = _varlen_fwd_custom(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size_left,
    window_size_right,
    softcap,
    pack_gqa,
  )
  return (out, lse) if return_lse else out

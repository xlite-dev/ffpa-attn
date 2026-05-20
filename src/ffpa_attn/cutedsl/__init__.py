"""CuTeDSL FFPA backend: SM90 + D=512 specialised forward/backward.

Exposes the dense and varlen CuTeDSL entry shims used by
:mod:`ffpa_attn.ffpa_attn_interface` and :mod:`ffpa_attn.functional`,
and triggers the side-effectful import of :mod:`._interface` so the
``ffpa_attn::splitd_fwd_sm90`` / ``splitd_bwd_sm90`` ``torch.library``
ops (used by :func:`ffpa_attn.ffpa_attn_varlen_func`'s CuTeDSL fast-path)
are registered.

Dense-path ops ``ffpa_attn::_fwd_cutedsl`` / ``ffpa_attn::_bwd_cutedsl``
are registered below following the same ``torch.library.define`` +
``@impl`` + ``@register_fake`` pattern as :mod:`ffpa_attn.triton`, so the
dense CuTeDSL path is ``torch.compile``-compatible.
"""

import torch

from . import _interface  # noqa: F401  # register torch ops (varlen splitd_*)
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

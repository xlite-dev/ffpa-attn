"""CuTeDSL FFPA backend: SM90 + D=512 specialised forward/backward.

Exposes the dense and varlen CuTeDSL entry shims used by
:mod:`ffpa_attn.ffpa_attn_interface` and :mod:`ffpa_attn.functional`,
and triggers the side-effectful import of :mod:`._interface` so the
``ffpa_attn::splitd_fwd_sm90`` / ``splitd_bwd_sm90`` ``torch.library``
ops (used by :func:`ffpa_attn.ffpa_attn_varlen_func`'s CuTeDSL fast-path)
are registered.
"""

from . import _interface  # noqa: F401  # register torch ops
from ._wrappers import (
  _ffpa_attn_cutedsl_forward,
  _ffpa_attn_cutedsl_backward,
  _ffpa_attn_varlen_cutedsl,
  _require_cutedsl_supported,
  cutedsl_backward_available,
  cutedsl_forward_available,
)

__all__ = [
  "_ffpa_attn_cutedsl_forward",
  "_ffpa_attn_cutedsl_backward",
  "_ffpa_attn_varlen_cutedsl",
  "_require_cutedsl_supported",
  "cutedsl_forward_available",
  "cutedsl_backward_available",
]

"""CuTeDSL FFPA backend: SM90 + D=512 specialised forward/backward.

Exposes layout-adapting wrappers consumed by
:mod:`ffpa_attn.functional` and triggers the side-effectful import of
:mod:`.interface` so the ``splitd_flash_attn::varlen_fwd`` /
``varlen_bwd`` ``torch.library`` ops (used by
:func:`ffpa_attn.ffpa_attn_varlen_func`'s CuTeDSL fast-path) are registered.
"""

from . import interface as _interface  # noqa: F401  # register torch ops
from ._wrappers import (
  _ffpa_attn_backward_cutedsl,
  _ffpa_attn_forward_cutedsl,
  _require_cutedsl_supported,
  cutedsl_backward_available,
  cutedsl_forward_available,
)

__all__ = [
  "_ffpa_attn_forward_cutedsl",
  "_ffpa_attn_backward_cutedsl",
  "_require_cutedsl_supported",
  "cutedsl_forward_available",
  "cutedsl_backward_available",
]

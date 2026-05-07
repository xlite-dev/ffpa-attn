"""Aten flash-attention forward/backward wrappers for small-D (D <= 256)."""

from ._flash_bwd import _aten_flash_attn_backward
from ._flash_fwd import _aten_flash_attn_forward

__all__ = ["_aten_flash_attn_forward", "_aten_flash_attn_backward"]

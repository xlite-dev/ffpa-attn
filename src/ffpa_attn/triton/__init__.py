"""Triton FFPA attention forward/backward implementations for large-D
(D > 256, but also works for D <= 256).
"""
from ._ffpa_fwd import _ffpa_attn_forward_triton
from ._ffpa_bwd import _ffpa_attn_backward_triton

__all__ = ["_ffpa_attn_forward_triton", "_ffpa_attn_backward_triton"]

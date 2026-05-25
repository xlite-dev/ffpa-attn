"""Generic SM90 CuTeDSL forward kernel wrapper for dense large head dimensions.

This module intentionally leaves the D=512-specialized implementation intact.
The generic dense path reuses that SplitD pipeline while the public tensors keep
their logical head dimension. TMA OOB handling zero-fills padded loads and drops
padded stores for D below the selected physical tile.
"""

from typing import Optional, Type

import cutlass

from ._fwd_d512_sm90 import FFPAAttnFwdSm90SplitD
from ._utils import SM90_SUPPORTED_HEAD_DIM, MIN_SUPPORTED_HEAD_DIM


def _generic_tile_head_dim(head_dim: int, head_dim_v: int) -> int:
  if head_dim != head_dim_v:
    raise ValueError(
      f"generic CuTeDSL dense path requires q/k head_dim == v head_dim_v, "
      f"got {head_dim} and {head_dim_v}"
    )
  if head_dim < MIN_SUPPORTED_HEAD_DIM or head_dim > SM90_SUPPORTED_HEAD_DIM:
    raise ValueError(
      f"generic CuTeDSL dense path supports {MIN_SUPPORTED_HEAD_DIM} <= head_dim <= "
      f"{SM90_SUPPORTED_HEAD_DIM}, got {head_dim}"
    )
  return SM90_SUPPORTED_HEAD_DIM


class FFPAAttnFwdSm90SplitDGeneric(FFPAAttnFwdSm90SplitD):
  """Dense large-D forward path using a 512-wide physical SplitD tile."""

  def __init__(
    self,
    dtype: Type[cutlass.Numeric],
    head_dim: int,
    head_dim_v: Optional[int] = None,
    qhead_per_kvhead: int = 1,
    is_causal: bool = False,
    is_local: bool = False,
    pack_gqa: bool = True,
    tile_m: int = 64,
    tile_n: int = 128,
    score_mod: Optional[cutlass.Constexpr] = None,
    mask_mod: Optional[cutlass.Constexpr] = None,
    has_aux_tensors: bool = False,
    kv_same: bool = False,
  ):
    logical_head_dim_v = head_dim if head_dim_v is None else head_dim_v
    tile_head_dim = _generic_tile_head_dim(head_dim, logical_head_dim_v)
    self.logical_head_dim = head_dim
    self.logical_head_dim_v = logical_head_dim_v
    super().__init__(
      dtype,
      tile_head_dim,
      head_dim_v=tile_head_dim,
      qhead_per_kvhead=qhead_per_kvhead,
      is_causal=is_causal,
      is_local=is_local,
      pack_gqa=pack_gqa,
      tile_m=tile_m,
      tile_n=tile_n,
      score_mod=score_mod,
      mask_mod=mask_mod,
      has_aux_tensors=has_aux_tensors,
      kv_same=kv_same,
    )

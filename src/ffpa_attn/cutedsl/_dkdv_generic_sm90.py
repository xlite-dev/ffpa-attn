"""Generic SM90 CuTeDSL dK/dV kernel wrapper for dense large head dimensions."""

from typing import Optional, Type

import cutlass

from ._dkdv_d512_sm90 import FFPAAttnBwdDKDVSm90SplitD
from ._fwd_generic_sm90 import _generic_tile_head_dim


class FFPAAttnBwdDKDVSm90SplitDGeneric(FFPAAttnBwdDKDVSm90SplitD):
  """Dense large-D dK/dV path using a 512-wide physical SplitD tile."""

  def __init__(
    self,
    dtype: Type[cutlass.Numeric],
    head_dim: int,
    head_dim_v: Optional[int] = None,
    is_causal: bool = False,
    qhead_per_kvhead: int = 1,
    tile_m: int = 64,
    tile_n: int = 64,
  ):
    logical_head_dim_v = head_dim if head_dim_v is None else head_dim_v
    tile_head_dim = _generic_tile_head_dim(head_dim, logical_head_dim_v)
    self.logical_head_dim = head_dim
    self.logical_head_dim_v = logical_head_dim_v
    super().__init__(
      dtype,
      tile_head_dim,
      head_dim_v=tile_head_dim,
      is_causal=is_causal,
      qhead_per_kvhead=qhead_per_kvhead,
      tile_m=tile_m,
      tile_n=tile_n,
    )

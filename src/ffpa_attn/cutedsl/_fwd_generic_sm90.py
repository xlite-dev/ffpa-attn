"""Generic SM90 CuTeDSL forward kernel wrapper for dense large head dimensions.

This module intentionally leaves the D=512-specialized implementation intact.
The generic dense path reuses that SplitD pipeline while the public tensors keep
their logical head dimension. TMA OOB handling zero-fills padded loads and drops
padded stores for D below the selected physical tile.
"""

from typing import Optional, Type

import cutlass

from ._fwd_d512_sm90 import FFPAAttnFwdSm90SplitD
from ._utils import SUPPORTED_HEAD_DIM, MIN_GENERIC_HEAD_DIM

D384_AWARE_HEAD_DIM = 384


def _generic_tile_head_dim(head_dim: int, head_dim_v: int) -> int:
  if head_dim != head_dim_v:
    raise ValueError(
      f"generic CuTeDSL dense path requires q/k head_dim == v head_dim_v, "
      f"got {head_dim} and {head_dim_v}"
    )
  if head_dim <= MIN_GENERIC_HEAD_DIM or head_dim > SUPPORTED_HEAD_DIM:
    raise ValueError(
      f"generic CuTeDSL dense path supports {MIN_GENERIC_HEAD_DIM} < head_dim <= "
      f"{SUPPORTED_HEAD_DIM}, got {head_dim}"
    )
  return SUPPORTED_HEAD_DIM


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


class FFPAAttnFwdSm90SplitDD384Aware(FFPAAttnFwdSm90SplitDGeneric):
  """Dense forward path using a 384-wide physical SplitD tile for D<=384."""

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
    if head_dim != logical_head_dim_v:
      raise ValueError(
        f"D384-aware CuTeDSL dense path requires q/k head_dim == v head_dim_v, "
        f"got {head_dim} and {logical_head_dim_v}"
      )
    if not MIN_GENERIC_HEAD_DIM < head_dim <= D384_AWARE_HEAD_DIM:
      raise ValueError(
        f"D384-aware CuTeDSL dense path supports {MIN_GENERIC_HEAD_DIM} < head_dim <= "
        f"{D384_AWARE_HEAD_DIM}, got {head_dim}"
      )
    super().__init__(
      dtype,
      head_dim,
      head_dim_v=logical_head_dim_v,
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
    self.tile_hdim = D384_AWARE_HEAD_DIM
    self.tile_hdimv = D384_AWARE_HEAD_DIM
    self.tile_hdimv_half = D384_AWARE_HEAD_DIM // 2
    self.tile_hdim_full = D384_AWARE_HEAD_DIM
    self.check_hdim_oob = head_dim != self.tile_hdim
    self.check_hdim_v_oob = logical_head_dim_v != self.tile_hdimv

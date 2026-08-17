# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.
# Adapted from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/copy_utils.py

import math
from typing import Type, Callable

import cutlass
import cutlass.cute as cute
from cutlass import const_expr
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import dsl_user_op


@dsl_user_op
def get_copy_atom(
  dtype: Type[cutlass.Numeric],
  num_copy_elems: int,
  is_async: bool = False,
  *,
  loc=None,
  ip=None
) -> cute.CopyAtom:
  num_copy_bits = const_expr(min(128, num_copy_elems * dtype.width))
  copy_op = cpasync.CopyG2SOp() if is_async else cute.nvgpu.CopyUniversalOp()
  return cute.make_copy_atom(copy_op, dtype, num_bits_per_copy=num_copy_bits)


def tiled_copy_2d(
  dtype: Type[cutlass.Numeric],
  major_mode_size: int,
  num_threads: int,
  is_async: bool = False
) -> cute.TiledCopy:
  num_copy_bits = math.gcd(major_mode_size, 128 // dtype.width) * dtype.width
  copy_elems = num_copy_bits // dtype.width
  copy_op = cpasync.CopyG2SOp() if is_async else cute.nvgpu.CopyUniversalOp()
  copy_atom = cute.make_copy_atom(
    copy_op, dtype, num_bits_per_copy=num_copy_bits
  )
  gmem_threads_per_row = major_mode_size // copy_elems
  assert num_threads % gmem_threads_per_row == 0
  thr_layout = cute.make_ordered_layout(
    (num_threads // gmem_threads_per_row, gmem_threads_per_row),
    order=(1, 0),
  )
  val_layout = cute.make_layout((1, copy_elems))
  return cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)


def tma_get_copy_fn(
  atom: cute.CopyAtom,
  cta_coord: cute.Coord,
  cta_layout: cute.Layout,
  src_tensor: cute.Tensor,
  dst_tensor: cute.Tensor,
  filter_zeros: bool = False,
  single_stage: bool = False,
  **kwargs,
) -> Callable:
  src_is_smem = const_expr(
    isinstance(src_tensor.iterator, cute.Pointer)
    and src_tensor.memspace == cute.AddressSpace.smem
  )
  smem_tensor, gmem_tensor = (src_tensor, dst_tensor
                              ) if src_is_smem else (dst_tensor, src_tensor)
  group_rank_smem = const_expr(
    cute.rank(smem_tensor) - (1 if not single_stage else 0)
  )
  group_rank_gmem = const_expr(
    cute.rank(gmem_tensor) - (1 if not single_stage else 0)
  )
  # ((atom_v, rest_v), STAGE), ((atom_v, rest_v), RestK)
  s, g = cpasync.tma_partition(
    atom,
    cta_coord,
    cta_layout,
    cute.group_modes(smem_tensor, 0, group_rank_smem),
    cute.group_modes(gmem_tensor, 0, group_rank_gmem),
  )
  if const_expr(filter_zeros):
    s = cute.filter_zeros(s)
    g = cute.filter_zeros(g)
  src, dst = (s, g) if src_is_smem else (g, s)

  def copy_tma(src_idx, dst_idx, **new_kwargs):
    cute.copy(
      atom, src[None, src_idx], dst[None, dst_idx], **new_kwargs, **kwargs
    )

  def copy_tma_single_stage(**new_kwargs):
    cute.copy(atom, src, dst, **new_kwargs, **kwargs)

  return (
    copy_tma if const_expr(not single_stage) else copy_tma_single_stage
  ), s, g

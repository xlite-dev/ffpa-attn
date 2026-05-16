# This file is copied from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/tile_scheduler.py
# Copyright (c) 2025, Tri Dao.
# SM90-only trimmed version of flash_attn/cute/tile_scheduler.py
#
# Removed (not used by SM90 fwd/bwd training pipeline):
#   - SchedulingMode enum          (SM90 always uses STATIC, never CLC/DYNAMIC)
#   - ClcState                     (CLC hardware scheduling — SM100+ only)
#   - TileSchedulerProtocol        (abstract protocol, documentation only)
#   - StaticPersistentTileScheduler (SM100+ persistent kernel scheduling)
#
# Simplified in TileSchedulerArguments:
#   - Removed cluster_shape_mn     (SM90 always (1,1))
#   - Removed is_persistent        (SM90 always False)
#   - Removed is_split_kv          (SM90 training never uses split-KV)
#   - Removed use_cluster_idx      (SM90 always False)
#
# Simplified in SingleTileScheduler:
#   - Removed is_split_kv / num_splits_divmod / cluster_shape_mn / use_cluster_idx
#   - get_current_work: removed split_kv divmod branch
#   - get_grid_shape: removed cluster rounding
#   - create: removed cluster_idx branch, removed clc parameter
#
# Simplified in SingleTileLPTScheduler:
#   - Removed ALL CLC code paths (clc_work_to_coords, clc_problem_shape,
#     _clc_grid_shape, and CLC branches in create/get_current_work/
#     initial_work_tile_info/advance_to_next_work/prefetch/producer_tail/
#     __extract/__new_from_mlir)
#   - Removed is_split_kv / num_splits / num_splits_divmod
#   - Removed cluster_shape_m / use_cluster_idx
#   - Removed scheduling_mode field (always STATIC)
#   - Removed clc parameter from __init__ and create
#
# Simplified in SingleTileLPTBwdScheduler:
#   - Removed cluster_shape_mn from Params
#   - Removed cluster_idx division and bidx_in_cluster branch in get_current_work
#   - Simplified total_blocks computation (no cluster ceil_div)
#
# Simplified in SingleTileVarlenScheduler:
#   - Removed ALL CLC code paths (clc_problem_shape, CLC branches in create/
#     get_current_work/initial_work_tile_info/advance/prefetch/producer_tail/
#     __extract/__new_from_mlir)
#   - Removed is_split_kv / cluster_shape_m / scheduling_mode
#   - Removed clc parameter from __init__ and create
#   - _get_num_m_blocks: removed cluster ceil_div (cluster_shape_m=1)
#   - _varlen_coord_map: removed cluster division and bidx_in_cluster branch,
#     removed split_kv return path

from typing import Optional, Tuple
from dataclasses import dataclass

try:
  from typing import override
except ImportError:  # Python < 3.12
  from typing_extensions import override

import cutlass
from cutlass._mlir import ir
import cutlass.cute as cute
from cutlass import Int32, const_expr
from cutlass.cute import FastDivmodDivisor

from quack.cute_dsl_utils import ParamsBase

from . import clz, warp_prefix_sum

# ---------------------------------------------------------------------------
# WorkTileInfo — 4-axis: (block, head, batch, split)
# ---------------------------------------------------------------------------


class WorkTileInfo(cutlass.utils.WorkTileInfo):
  """Altered WorkTileInfo which includes four axes: (block, head, batch, split)"""

  @override
  def __new_from_mlir_values__(self, values: list[ir.Value]) -> "WorkTileInfo":
    assert len(values) == 5
    new_tile_idx = cutlass.new_from_mlir_values(self._tile_idx, values[:-1])
    new_is_valid_tile = cutlass.new_from_mlir_values(self._is_valid_tile, [values[-1]])
    return WorkTileInfo(new_tile_idx, new_is_valid_tile)


# ---------------------------------------------------------------------------
# TileSchedulerArguments — shared argument dataclass (SM90-only)
# ---------------------------------------------------------------------------


@dataclass
class TileSchedulerArguments(ParamsBase):
  num_block: Int32
  num_head: Int32
  num_batch: Int32
  num_splits: Int32  # always 1 for SM90 training
  seqlen_k: Int32
  headdim: Int32
  headdim_v: Int32
  total_q: Int32
  tile_shape_mn: cutlass.Constexpr[Tuple[int, int]]
  mCuSeqlensQ: Optional[cute.Tensor] = None
  mSeqUsedQ: Optional[cute.Tensor] = None
  qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1
  element_size: cutlass.Constexpr[int] = 2
  lpt: cutlass.Constexpr[bool] = False
  head_swizzle: cutlass.Constexpr[bool] = False


# ---------------------------------------------------------------------------
# SingleTileScheduler — non-causal dense fwd, non-deterministic bwd, pre/postprocess
# ---------------------------------------------------------------------------


class SingleTileScheduler:

  @dataclass
  class Params(ParamsBase):
    num_block: Int32
    num_head: Int32
    num_batch: Int32

    @staticmethod
    def create(args: TileSchedulerArguments, *, loc=None, ip=None) -> "SingleTileScheduler.Params":
      return SingleTileScheduler.Params(
        args.num_block,
        args.num_head,
        args.num_batch,
      )

  def __init__(self, params: Params, blk_coord: cute.Coord, *, loc=None, ip=None):
    self.params = params
    self._blk_coord = blk_coord
    self._is_first_block = True
    self._loc = loc
    self._ip = ip

  @staticmethod
  def to_underlying_arguments(
    args: TileSchedulerArguments,
    *,
    loc=None,
    ip=None,
  ) -> Params:
    return SingleTileScheduler.Params.create(args, loc=loc, ip=ip)

  @staticmethod
  def create(params: Params, *, loc=None, ip=None) -> "SingleTileScheduler":
    blk_coord = cute.arch.block_idx()
    return SingleTileScheduler(params, blk_coord, loc=loc, ip=ip)

  @staticmethod
  def get_grid_shape(
    params: Params,
    *,
    loc=None,
    ip=None,
  ) -> Tuple[Int32, Int32, Int32]:
    return (params.num_block, params.num_head, params.num_batch)

  def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
    block_idx, head_idx, batch_idx = self._blk_coord
    return WorkTileInfo(
      (block_idx, head_idx, batch_idx, Int32(0)),
      self._is_first_block,
    )

  def initial_work_tile_info(self, *, loc=None, ip=None):
    return self.get_current_work(loc=loc, ip=ip)

  def prefetch_next_work(self, *, loc=None, ip=None):
    pass

  def advance_to_next_work(self, *, loc=None, ip=None):
    self._is_first_block = False
    return self.get_current_work()

  def producer_tail(self, *, loc=None, ip=None):
    pass

  def __extract_mlir_values__(self):
    values, self._values_pos = [], []
    for obj in [self.params, self._blk_coord]:
      obj_values = cutlass.extract_mlir_values(obj)
      values += obj_values
      self._values_pos.append(len(obj_values))
    return values

  def __new_from_mlir_values__(self, values):
    obj_list = []
    for obj, n_items in zip([self.params, self._blk_coord], self._values_pos):
      obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
      values = values[n_items:]
    return SingleTileScheduler(*(tuple(obj_list)), loc=self._loc)


# ---------------------------------------------------------------------------
# SingleTileLPTScheduler — causal/local dense fwd (L2 swizzle + LPT)
#
# STATIC scheduling only.  CLC / split-KV / cluster paths removed.
# ---------------------------------------------------------------------------


class SingleTileLPTScheduler:

  @dataclass
  class Params(ParamsBase):
    total_blocks: Int32
    num_block: Int32
    num_head: Int32
    num_batch: Int32
    l2_minor: Int32
    num_head_divmod: FastDivmodDivisor
    l2_minor_divmod: FastDivmodDivisor
    l2_major_divmod: FastDivmodDivisor
    l2_minor_residual_divmod: FastDivmodDivisor
    num_hb_quotient: Int32
    lpt: cutlass.Constexpr[bool] = True

    @staticmethod
    @cute.jit
    def create(
      args: TileSchedulerArguments,
      *,
      loc=None,
      ip=None,
    ) -> "SingleTileLPTScheduler.Params":
      size_one_kv_head = args.seqlen_k * (args.headdim + args.headdim_v) * args.element_size
      size_one_head = size_one_kv_head
      size_l2 = 50 * 1024 * 1024  # 40 MB for K & V
      # Swizzle is the size of each "section". Round swizzle to a power of 2
      # swizzle is how many heads can fit in L2
      log2_floor = lambda n: 31 - clz(n)
      swizzle = 1 if size_l2 < size_one_head else (1 << log2_floor(size_l2 // size_one_head))
      # If we're in the last section (called residual), we don't want to divide by
      # swizzle. Instead we want to divide by the remainder.
      num_hb_quotient = (args.num_head * args.num_batch) // swizzle
      num_hb_remainder = (args.num_head * args.num_batch) % swizzle
      return SingleTileLPTScheduler.Params(
        total_blocks=args.num_block * args.num_head * args.num_batch,
        num_block=args.num_block,
        num_head=args.num_head,
        num_batch=args.num_batch,
        l2_minor=Int32(swizzle),
        num_head_divmod=FastDivmodDivisor(args.num_head),
        l2_minor_divmod=FastDivmodDivisor(swizzle),
        l2_major_divmod=FastDivmodDivisor(swizzle * args.num_block),
        l2_minor_residual_divmod=FastDivmodDivisor(max(num_hb_remainder, 1)),
        num_hb_quotient=Int32(num_hb_quotient),
        lpt=args.lpt,
      )

  def __init__(
    self,
    params: Params,
    tile_idx: Int32,
    split_idx: Int32,
    *,
    loc=None,
    ip=None,
  ):
    self.params = params
    self._tile_idx = tile_idx
    self._split_idx = split_idx
    self._loc = loc
    self._ip = ip

  @staticmethod
  def to_underlying_arguments(
    args: TileSchedulerArguments,
    *,
    loc=None,
    ip=None,
  ) -> Params:
    return SingleTileLPTScheduler.Params.create(args, loc=loc, ip=ip)

  @staticmethod
  @cute.jit
  def create(params: Params, *, loc=None, ip=None) -> "SingleTileLPTScheduler":
    tile_idx, split_idx, _ = cute.arch.block_idx()
    return SingleTileLPTScheduler(params, tile_idx, split_idx, loc=loc, ip=ip)

  @staticmethod
  def get_grid_shape(
    params: Params,
    *,
    loc=None,
    ip=None,
  ) -> Tuple[Int32, Int32, Int32]:
    return (params.total_blocks, Int32(1), Int32(1))

  @cute.jit
  def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
    params = self.params
    # L2-swizzled coordinate mapping
    bidhb, l2_mod = divmod(self._tile_idx, params.l2_major_divmod)
    # If we're in the last section (called residual), we don't want to divide by
    # swizzle. Instead we want to divide by the remainder.
    block, bidhb_residual = 0, 0
    if bidhb < params.num_hb_quotient:
      block, bidhb_residual = divmod(l2_mod, params.l2_minor_divmod)
    else:
      block, bidhb_residual = divmod(l2_mod, params.l2_minor_residual_divmod)
    bidhb_actual = bidhb * params.l2_minor + bidhb_residual
    batch_idx, head_idx = divmod(bidhb_actual, params.num_head_divmod)
    # Longest-processing-time-first
    if const_expr(params.lpt):
      block = params.num_block - 1 - block
    is_valid = self._tile_idx < params.total_blocks
    return WorkTileInfo((Int32(block), Int32(head_idx), Int32(batch_idx), Int32(self._split_idx)), is_valid)

  @cute.jit
  def initial_work_tile_info(self, *, loc=None, ip=None):
    return self.get_current_work(loc=loc, ip=ip)

  def prefetch_next_work(self, *, loc=None, ip=None):
    pass

  def advance_to_next_work(self, *, loc=None, ip=None):
    # Single tile scheduler — set to invalid tile_idx to indicate no more work
    self._tile_idx = self.params.total_blocks
    return self.get_current_work()

  def producer_tail(self, *, loc=None, ip=None):
    pass

  def __extract_mlir_values__(self):
    values, self._values_pos = [], []
    for obj in [self.params, self._tile_idx, self._split_idx]:
      obj_values = cutlass.extract_mlir_values(obj)
      values += obj_values
      self._values_pos.append(len(obj_values))
    return values

  def __new_from_mlir_values__(self, values):
    obj_list = []
    for obj, n_items in zip(
      [self.params, self._tile_idx, self._split_idx],
      self._values_pos,
    ):
      obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
      values = values[n_items:]
    return self.__class__(*obj_list, loc=self._loc)


# ---------------------------------------------------------------------------
# SingleTileLPTBwdScheduler — deterministic backward (SPT + L2 swizzle)
# ---------------------------------------------------------------------------


class SingleTileLPTBwdScheduler:

  @dataclass
  class Params(ParamsBase):
    total_blocks: Int32
    num_block: Int32
    l2_minor: Int32
    num_head_divmod: FastDivmodDivisor
    l2_minor_divmod: FastDivmodDivisor
    l2_major_divmod: FastDivmodDivisor
    l2_minor_residual_divmod: FastDivmodDivisor
    num_hb_quotient: Int32
    spt: cutlass.Constexpr[bool] = True

    @staticmethod
    @cute.jit
    def create(
      args: TileSchedulerArguments,
      *,
      loc=None,
      ip=None,
    ) -> "SingleTileLPTBwdScheduler.Params":
      size_l2 = 50 * 1024 * 1024
      size_one_qdo_head = args.seqlen_k * (args.headdim + args.headdim_v) * args.element_size
      size_one_dqaccum_head = args.seqlen_k * (args.headdim) * 4
      size_one_head = size_one_qdo_head + size_one_dqaccum_head
      log2_floor = lambda n: 31 - clz(n)
      swizzle = 1 if size_l2 < size_one_head else (1 << log2_floor(size_l2 // size_one_head))
      # If we're in the last section (called residual), we don't want to divide by
      # swizzle. Instead we want to divide by the remainder.
      num_hb_quotient = (args.num_head * args.num_batch) // swizzle
      num_hb_remainder = (args.num_head * args.num_batch) % swizzle
      return SingleTileLPTBwdScheduler.Params(
        total_blocks=args.num_block * args.num_head * args.num_batch,
        num_block=args.num_block,
        l2_minor=Int32(swizzle),
        num_head_divmod=FastDivmodDivisor(args.num_head),
        l2_minor_divmod=FastDivmodDivisor(swizzle),
        l2_major_divmod=FastDivmodDivisor(swizzle * args.num_block),
        l2_minor_residual_divmod=FastDivmodDivisor(max(num_hb_remainder, 1)),  # don't divide by 0
        num_hb_quotient=Int32(num_hb_quotient),
        spt=args.lpt,
      )

  def __init__(self, params: Params, tile_idx: Int32, *, loc=None, ip=None):
    self.params = params
    self._tile_idx = tile_idx
    self._loc = loc
    self._ip = ip

  @staticmethod
  def to_underlying_arguments(
    args: TileSchedulerArguments,
    *,
    loc=None,
    ip=None,
  ) -> Params:
    return SingleTileLPTBwdScheduler.Params.create(args, loc=loc, ip=ip)

  @staticmethod
  @cute.jit
  def create(params: Params, *, loc=None, ip=None) -> "SingleTileLPTBwdScheduler":
    tile_idx = cute.arch.block_idx()[0]
    return SingleTileLPTBwdScheduler(params, tile_idx, loc=loc, ip=ip)

  # called by host
  @staticmethod
  def get_grid_shape(
    params: Params,
    *,
    loc=None,
    ip=None,
  ) -> Tuple[Int32, Int32, Int32]:
    return (params.total_blocks, Int32(1), Int32(1))

  @cute.jit
  def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
    params = self.params
    # L2-swizzled coordinate mapping
    bidhb, l2_mod = divmod(self._tile_idx, params.l2_major_divmod)
    block, bidhb_residual = 0, 0
    if bidhb < params.num_hb_quotient:
      block, bidhb_residual = divmod(l2_mod, params.l2_minor_divmod)
    else:
      block, bidhb_residual = divmod(l2_mod, params.l2_minor_residual_divmod)
    bidhb_actual = bidhb * params.l2_minor + bidhb_residual
    batch_idx, head_idx = divmod(bidhb_actual, params.num_head_divmod)
    # Shortest-processing-time-first (reverse block order for bwd)
    if cutlass.const_expr(params.spt):
      block = params.num_block - 1 - block
    is_valid = self._tile_idx < params.total_blocks
    return WorkTileInfo((Int32(block), Int32(head_idx), Int32(batch_idx), Int32(0)), is_valid)

  def initial_work_tile_info(self, *, loc=None, ip=None):
    return self.get_current_work(loc=loc, ip=ip)

  def prefetch_next_work(self, *, loc=None, ip=None):
    pass

  def advance_to_next_work(self, *, loc=None, ip=None):
    # Single tile scheduler — set to invalid tile_idx to indicate no more work
    self._tile_idx = self.params.total_blocks
    return self.get_current_work()

  def producer_tail(self, *, loc=None, ip=None):
    pass

  def __extract_mlir_values__(self):
    values, self._values_pos = [], []
    for obj in [self.params, self._tile_idx]:
      obj_values = cutlass.extract_mlir_values(obj)
      values += obj_values
      self._values_pos.append(len(obj_values))
    return values

  def __new_from_mlir_values__(self, values):
    obj_list = []
    for obj, n_items in zip([self.params, self._tile_idx], self._values_pos):
      obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
      values = values[n_items:]
    return self.__class__(*(tuple(obj_list)), loc=self._loc)


# ---------------------------------------------------------------------------
# SingleTileVarlenScheduler — varlen fwd & bwd (warp prefix-sum tile mapping)
#
# STATIC scheduling only.  CLC / split-KV / cluster paths removed.
# Preserves: LPT block reversal, head_swizzle for deterministic bwd.
# ---------------------------------------------------------------------------


class SingleTileVarlenScheduler:

  @dataclass
  class Params(ParamsBase):
    num_head: Int32
    num_batch: Int32
    total_q: Int32
    max_kvblock_in_l2: Int32
    tile_shape_mn: cutlass.Constexpr[Tuple[int, int]]
    mCuSeqlensQ: Optional[cute.Tensor] = None
    mSeqUsedQ: Optional[cute.Tensor] = None
    qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1
    lpt: cutlass.Constexpr[bool] = False
    head_swizzle: cutlass.Constexpr[bool] = False

    @staticmethod
    @cute.jit
    def create(
      args: TileSchedulerArguments,
      *,
      loc=None,
      ip=None,
    ) -> "SingleTileVarlenScheduler.Params":
      size_l2 = 50 * 1024 * 1024  # 50 MB for K & V
      # if backward, this is qdo block size
      kv_block_size = (args.headdim + args.headdim_v) * args.element_size * args.tile_shape_mn[1]
      # if backward, add dqaccum block size to calculate swizzle
      if args.head_swizzle:
        kv_block_size += args.headdim * 4 * args.tile_shape_mn[1]
      max_kvblock_in_l2 = size_l2 // kv_block_size
      assert args.mCuSeqlensQ is not None or args.mSeqUsedQ is not None, (
        "At least one of mCuSeqlensQ or mSeqUsedQ must be provided"
      )
      return SingleTileVarlenScheduler.Params(
        num_head=args.num_head,
        num_batch=args.num_batch,
        total_q=args.total_q,
        max_kvblock_in_l2=max_kvblock_in_l2,
        tile_shape_mn=args.tile_shape_mn,
        mCuSeqlensQ=args.mCuSeqlensQ,
        mSeqUsedQ=args.mSeqUsedQ,
        qhead_per_kvhead_packgqa=args.qhead_per_kvhead_packgqa,
        lpt=args.lpt,
        head_swizzle=args.head_swizzle,
      )

  def __init__(
    self,
    params: Params,
    tile_idx: Int32,
    split_idx: Int32,
    *,
    loc=None,
    ip=None,
  ):
    self.params = params
    self._tile_idx = tile_idx
    self._split_idx = split_idx
    self._is_first_block = True
    self._loc = loc
    self._ip = ip

  @staticmethod
  def to_underlying_arguments(
    args: TileSchedulerArguments,
    *,
    loc=None,
    ip=None,
  ) -> Params:
    return SingleTileVarlenScheduler.Params.create(args, loc=loc, ip=ip)

  @staticmethod
  @cute.jit
  def create(params: Params, *, loc=None, ip=None) -> "SingleTileVarlenScheduler":
    tile_idx, split_idx, _ = cute.arch.block_idx()
    return SingleTileVarlenScheduler(params, tile_idx, split_idx, loc=loc, ip=ip)

  # called by host
  @staticmethod
  def get_grid_shape(
    params: Params,
    *,
    loc=None,
    ip=None,
  ) -> Tuple[Int32, Int32, Int32]:
    # cluster_shape_m=1 for SM90, so no cluster rounding needed
    total_blocks_max = (params.total_q + params.num_batch * (params.tile_shape_mn[0] - 1)) // params.tile_shape_mn[0]
    return (total_blocks_max * params.num_head, Int32(1), Int32(1))

  @cute.jit
  def _get_num_m_blocks(self, lane: Int32, bidb_start: Int32) -> Int32:
    params = self.params
    batch_idx = lane + bidb_start
    if cutlass.const_expr(params.mSeqUsedQ is not None):
      seqlen = Int32(0)
      if batch_idx < params.num_batch:
        seqlen = params.mSeqUsedQ[batch_idx]
    else:
      assert params.mCuSeqlensQ is not None
      cur_cu_seqlen = Int32(0)
      if batch_idx <= params.num_batch:
        cur_cu_seqlen = params.mCuSeqlensQ[batch_idx]
      next_cu_seqlen = cute.arch.shuffle_sync_down(cur_cu_seqlen, offset=1)
      seqlen = next_cu_seqlen - cur_cu_seqlen
    if cutlass.const_expr(params.qhead_per_kvhead_packgqa > 1):
      seqlen *= params.qhead_per_kvhead_packgqa
    # cluster_shape_m=1 for SM90, so no outer ceil_div by cluster
    return (
      cute.ceil_div(seqlen, params.tile_shape_mn[0])
      if batch_idx < params.num_batch and lane < cute.arch.WARP_SIZE - 1 else Int32(0)
    )

  @cute.jit
  def _varlen_coord_map(self) -> WorkTileInfo:
    """Map self._tile_idx to (block, head, batch) via warp-level prefix sums."""
    params = self.params
    lane_idx = cute.arch.lane_idx()
    num_m_blocks = self._get_num_m_blocks(lane_idx, bidb_start=0)
    num_m_blocks_cumulative = warp_prefix_sum(num_m_blocks, lane_idx)
    # Total number of blocks for the next 31 batches
    m_blocks_in_group = cute.arch.shuffle_sync(num_m_blocks_cumulative, cute.arch.WARP_SIZE - 1)
    # Same for all lanes
    group_end_tile = m_blocks_in_group * params.num_head
    block, head_idx, batch_idx = Int32(0), Int32(0), Int32(0)
    # cluster_shape_m=1 for SM90, so next_tile_idx == self._tile_idx
    next_tile_idx = self._tile_idx
    while group_end_tile <= next_tile_idx:
      batch_idx += cute.arch.WARP_SIZE - 1
      if batch_idx >= params.num_batch:
        batch_idx = Int32(params.num_batch)
        group_end_tile = next_tile_idx + 1
      else:
        num_m_blocks = self._get_num_m_blocks(lane_idx, bidb_start=batch_idx)
        num_m_blocks_cumulative = warp_prefix_sum(num_m_blocks, lane_idx)
        m_blocks_in_group = cute.arch.shuffle_sync(num_m_blocks_cumulative, cute.arch.WARP_SIZE - 1)
        group_end_tile += m_blocks_in_group * params.num_head
    is_valid = False
    if batch_idx >= params.num_batch:
      block, head_idx, batch_idx = Int32(0), Int32(0), Int32(params.num_batch)
    else:
      group_start_tile = group_end_tile - m_blocks_in_group * params.num_head
      # The next problem to process is the first one that does not have ending tile position
      # that is greater than or equal to tile index.
      batch_idx_in_group = cute.arch.popc(
        cute.arch.vote_ballot_sync(group_start_tile + num_m_blocks_cumulative * params.num_head <= next_tile_idx)
      )
      batch_idx += batch_idx_in_group
      num_m_blocks_prev_lane = (
        0 if batch_idx_in_group == 0 else cute.arch.shuffle_sync(num_m_blocks_cumulative, batch_idx_in_group - 1)
      )
      num_m_blocks = cute.arch.shuffle_sync(num_m_blocks, batch_idx_in_group)
      mh_block = next_tile_idx - group_start_tile - num_m_blocks_prev_lane * params.num_head
      if cutlass.const_expr(params.lpt or params.head_swizzle):
        # L2-aware LPT scheduling within variable-length batches.
        # cluster_shape_m=1 for SM90, simplifies num_n_blocks calculation.
        num_n_blocks = (
          num_m_blocks * params.tile_shape_mn[0] // params.qhead_per_kvhead_packgqa // params.tile_shape_mn[1]
        )
        # Seems faster to have this be a power of 2
        nheads_in_l2 = (
          16 if num_n_blocks * 16 <= params.max_kvblock_in_l2 else (
            8 if num_n_blocks * 8 <= params.max_kvblock_in_l2 else (
              4 if num_n_blocks * 4 <= params.max_kvblock_in_l2 else
              (2 if num_n_blocks * 2 <= params.max_kvblock_in_l2 else 1)
            )
          )
        )
        nheads_in_l2 = min(nheads_in_l2, params.num_head)
        mh_in_l2 = nheads_in_l2 * num_m_blocks
        section_idx = mh_block // mh_in_l2
        l2_mod = mh_block - section_idx * mh_in_l2
        # Deal with tail section
        nheads_in_this_section = (
          nheads_in_l2 if nheads_in_l2 * (section_idx + 1) <= params.num_head else params.num_head -
          section_idx * nheads_in_l2
        )
        block = l2_mod // nheads_in_this_section
        head_idx_residual = l2_mod - block * nheads_in_this_section
        head_idx = section_idx * nheads_in_l2 + head_idx_residual
        if cutlass.const_expr(params.lpt):
          block = num_m_blocks - 1 - block
      else:
        head_idx = mh_block // num_m_blocks
        block = mh_block - head_idx * num_m_blocks
      is_valid = self._is_first_block and batch_idx < params.num_batch
      # cluster_shape_m=1 for SM90, no bidx_in_cluster adjustment needed
    return WorkTileInfo((Int32(block), Int32(head_idx), Int32(batch_idx), Int32(0)), is_valid)

  @cute.jit
  def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
    return self._varlen_coord_map()

  @cute.jit
  def initial_work_tile_info(self, *, loc=None, ip=None):
    return self._varlen_coord_map()

  def prefetch_next_work(self, *, loc=None, ip=None):
    pass

  def advance_to_next_work(self, *, loc=None, ip=None):
    self._is_first_block = False
    return self.get_current_work()

  def producer_tail(self, *, loc=None, ip=None):
    pass

  def __extract_mlir_values__(self):
    values, self._values_pos = [], []
    for obj in [self.params, self._tile_idx, self._split_idx]:
      obj_values = cutlass.extract_mlir_values(obj)
      values += obj_values
      self._values_pos.append(len(obj_values))
    return values

  def __new_from_mlir_values__(self, values):
    obj_list = []
    for obj, n_items in zip(
      [self.params, self._tile_idx, self._split_idx],
      self._values_pos,
    ):
      obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
      values = values[n_items:]
    return self.__class__(*obj_list, loc=self._loc)

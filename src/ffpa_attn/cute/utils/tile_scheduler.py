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
#   - Removed is_split_kv / scheduling_mode
#   - Removed clc parameter from __init__ and create
#   - _varlen_coord_map: removed the split_kv return path
#
# Restored in SingleTileVarlenScheduler for the SM100 D512 2-CTA forward, each behind a const_expr test against the SM90 default so the six SM90/SM80 call sites lower unchanged: cluster_shape_m / use_cluster_idx, and the compact cluster-persistent walk (varlen_static_persistent) with its resumable 31-batch group cursor.

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
    new_is_valid_tile = cutlass.new_from_mlir_values(
      self._is_valid_tile, [values[-1]]
    )
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
  # Reserved: enables L2-aware head-major scheduling inside SingleTileVarlenScheduler's
  # LPT branch (deterministic bwd). No current call site sets this True.
  head_swizzle: cutlass.Constexpr[bool] = False
  # Cluster-aware compact varlen scheduling (SM100 D512 2-CTA forward); each field below defaults to the SM90 shape and is consumed only under a const_expr test against that default.
  #: Cluster extent the M axis is enumerated in; ``> 1`` makes one work item a whole cluster's supertile rather than one CTA tile.
  cluster_shape_mn: cutlass.Constexpr[Tuple[int, int]] = (1, 1)
  #: Decode from ``cluster_idx`` instead of a physical CTA index, so cluster peers share one logical work id and the spatial rank stays outside it.
  use_cluster_idx: cutlass.Constexpr[bool] = False
  #: Compact cluster-persistent varlen walk: cap the grid at the SM count and stride each cluster through the packed work domain, instead of launching one CTA per possible tile and retiring most of them.
  varlen_static_persistent: cutlass.Constexpr[bool] = False


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
    def create(
      args: TileSchedulerArguments,
      *,
      loc=None,
      ip=None
    ) -> "SingleTileScheduler.Params":
      return SingleTileScheduler.Params(
        args.num_block,
        args.num_head,
        args.num_batch,
      )

  def __init__(
    self, params: Params, blk_coord: cute.Coord, *, loc=None, ip=None
  ):
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
      size_one_kv_head = args.seqlen_k * (
        args.headdim + args.headdim_v
      ) * args.element_size
      size_one_head = size_one_kv_head
      size_l2 = 50 * 1024 * 1024  # 40 MB for K & V
      # Swizzle is the size of each "section". Round swizzle to a power of 2
      # swizzle is how many heads can fit in L2
      log2_floor = lambda n: 31 - clz(n)
      swizzle = 1 if size_l2 < size_one_head else (
        1 << log2_floor(size_l2 // size_one_head)
      )
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
    return WorkTileInfo(
      (Int32(block), Int32(head_idx), Int32(batch_idx), Int32(self._split_idx)),
      is_valid
    )

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
# Reserved: SM90 deterministic backward scheduler (SPT + L2 swizzle).
# Currently no call site imports this; bwd kernels (_ffpa_dq, _ffpa_dkdv) default
# to SingleTileScheduler / SingleTileVarlenScheduler. Wire this in when the
# deterministic-backward implementation lands.


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
      size_one_qdo_head = args.seqlen_k * (
        args.headdim + args.headdim_v
      ) * args.element_size
      size_one_dqaccum_head = args.seqlen_k * (args.headdim) * 4
      size_one_head = size_one_qdo_head + size_one_dqaccum_head
      log2_floor = lambda n: 31 - clz(n)
      swizzle = 1 if size_l2 < size_one_head else (
        1 << log2_floor(size_l2 // size_one_head)
      )
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
        l2_minor_residual_divmod=FastDivmodDivisor(max(num_hb_remainder,
                                                       1)),  # don't divide by 0
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
  def create(
    params: Params, *, loc=None, ip=None
  ) -> "SingleTileLPTBwdScheduler":
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
    return WorkTileInfo(
      (Int32(block), Int32(head_idx), Int32(batch_idx), Int32(0)), is_valid
    )

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
    cluster_shape_m: cutlass.Constexpr[int] = 1
    use_cluster_idx: cutlass.Constexpr[bool] = False
    static_persistent: cutlass.Constexpr[bool] = False

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
      kv_block_size = (args.headdim + args.headdim_v
                       ) * args.element_size * args.tile_shape_mn[1]
      # if backward, add dqaccum block size to calculate swizzle
      if args.head_swizzle:
        kv_block_size += args.headdim * 4 * args.tile_shape_mn[1]
      max_kvblock_in_l2 = size_l2 // kv_block_size
      assert args.mCuSeqlensQ is not None or args.mSeqUsedQ is not None, (
        "At least one of mCuSeqlensQ or mSeqUsedQ must be provided"
      )
      assert args.cluster_shape_mn[1] == 1, (
        "the varlen work domain is enumerated along M; cluster N must be 1"
      )
      # The compact walk decodes from a cluster index; without one each CTA of a pair would resolve its own logical work id and the pair would split.
      assert not args.varlen_static_persistent or args.use_cluster_idx, (
        "compact static-persistent varlen scheduling requires "
        "use_cluster_idx=True"
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
        cluster_shape_m=args.cluster_shape_mn[0],
        use_cluster_idx=args.use_cluster_idx,
        static_persistent=args.varlen_static_persistent,
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
    # Resumable cursor over the 31-batch prefix-sum groups, read only by the static-persistent walk; :meth:`create` primes it so no decode observes the zeroed value.
    self._grp_base = Int32(0)
    self._grp_start = Int32(0)
    self._grp_nmb = Int32(0)
    self._grp_nmb_cum = Int32(0)

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
  def create(
    params: Params, *, loc=None, ip=None
  ) -> "SingleTileVarlenScheduler":
    if const_expr(params.static_persistent):
      # One logical work id per cluster, so peers cannot resolve different tiles; the spatial rank is applied by the caller.
      sched = SingleTileVarlenScheduler(
        params, cute.arch.cluster_idx()[0], Int32(0), loc=loc, ip=ip
      )
      sched._prime_group_cursor()
      return sched
    tile_idx, split_idx, _ = cute.arch.block_idx()
    return SingleTileVarlenScheduler(
      params, tile_idx, split_idx, loc=loc, ip=ip
    )

  # called by host
  @staticmethod
  def get_grid_shape(
    params: Params,
    *,
    loc=None,
    ip=None,
  ) -> Tuple[Int32, Int32, Int32]:
    if const_expr(params.cluster_shape_m == 1):
      # SM90 shape: one CTA per possible tile, no cluster rounding.
      total_blocks_max = (
        params.total_q + params.num_batch * (params.tile_shape_mn[0] - 1)
      ) // params.tile_shape_mn[0]
      return (total_blocks_max * params.num_head, Int32(1), Int32(1))
    # Cluster-aware: count whole clusters, rounding *down* since the odd excess is padding and a partial cluster has no peer to pair with.
    total_blocks_max = (
      params.total_q + params.num_batch *
      (params.cluster_shape_m * params.tile_shape_mn[0] - 1)
    ) // params.tile_shape_mn[0]
    total_blocks_max = (
      total_blocks_max // params.cluster_shape_m * params.cluster_shape_m
    )
    total_blocks_max *= params.num_head
    if const_expr(params.static_persistent):
      # Compact walk: cap the grid at the machine and stride through the packed work domain instead of launching one CTA per possible tile.
      sm_count = (
        cutlass.utils.HardwareInfo().get_device_multiprocessor_count()
      )
      max_ctas = sm_count // params.cluster_shape_m * params.cluster_shape_m
      return (cutlass.min(max_ctas, total_blocks_max), Int32(1), Int32(1))
    return (total_blocks_max, Int32(1), Int32(1))

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
    num_m = cute.ceil_div(seqlen, params.tile_shape_mn[0])
    if cutlass.const_expr(params.cluster_shape_m > 1):
      # Count whole clusters: one work item is one cluster-wide supertile.
      num_m = cute.ceil_div(num_m, params.cluster_shape_m)
    return (
      num_m if batch_idx < params.num_batch and lane < cute.arch.WARP_SIZE - 1
      else Int32(0)
    )

  @cute.jit
  def _prime_group_cursor(self) -> None:
    """Seed the resumable cursor with group 0; an all-zero cursor has ``group_end_tile == 0 <= t`` for every ``t``, so the first decode would advance the base past group 0 before resolving anything."""
    lane_idx = cute.arch.lane_idx()
    num_m_blocks = self._get_num_m_blocks(lane_idx, bidb_start=0)
    self._grp_base = Int32(0)
    self._grp_start = Int32(0)
    self._grp_nmb = num_m_blocks
    self._grp_nmb_cum = warp_prefix_sum(num_m_blocks, lane_idx)

  @cute.jit
  def _locate_group(self, next_tile_idx: Int32, lane_idx: Int32):
    """Find the 31-batch group whose flat tile range contains ``next_tile_idx``.

    Returns ``(group_base_batch, group_start_tile, num_m_blocks, num_m_blocks_cumulative)``, the last two per-lane; ``group_base_batch == num_batch`` means the work id is past the end.

    One-shot mode rescans from batch 0 every call.  A static-persistent cluster visits a strictly increasing sequence of work ids, so it resumes from the cursor the previous decode left behind, turning a per-item ``O(num_batch/31)`` rescan into one for the whole walk; correctness needs only ``grp_start <= t``, and the ``grp_start > next_tile_idx`` test falls back to the from-scratch scan for any cursor that could have overshot.
    """
    params = self.params
    grp_start = Int32(0)
    if cutlass.const_expr(params.static_persistent):
      grp_base = self._grp_base
      grp_start = self._grp_start
      num_m_blocks = self._grp_nmb
      num_m_blocks_cumulative = self._grp_nmb_cum
      if grp_start > next_tile_idx:  # cursor could have overshot -> rescan
        grp_base = Int32(0)
        grp_start = Int32(0)
        num_m_blocks = self._get_num_m_blocks(lane_idx, bidb_start=0)
        num_m_blocks_cumulative = warp_prefix_sum(num_m_blocks, lane_idx)
      m_blocks_in_group = cute.arch.shuffle_sync(
        num_m_blocks_cumulative, cute.arch.WARP_SIZE - 1
      )
      group_end_tile = grp_start + m_blocks_in_group * params.num_head
    else:
      grp_base = Int32(0)
      num_m_blocks = self._get_num_m_blocks(lane_idx, bidb_start=0)
      num_m_blocks_cumulative = warp_prefix_sum(num_m_blocks, lane_idx)
      # Total number of blocks for the next 31 batches; same for all lanes.
      m_blocks_in_group = cute.arch.shuffle_sync(
        num_m_blocks_cumulative, cute.arch.WARP_SIZE - 1
      )
      group_end_tile = m_blocks_in_group * params.num_head
    while group_end_tile <= next_tile_idx:
      grp_base += cute.arch.WARP_SIZE - 1
      if grp_base >= params.num_batch:
        grp_base = Int32(params.num_batch)
        group_end_tile = next_tile_idx + 1
      else:
        if cutlass.const_expr(params.static_persistent):
          grp_start = group_end_tile
        num_m_blocks = self._get_num_m_blocks(lane_idx, bidb_start=grp_base)
        num_m_blocks_cumulative = warp_prefix_sum(num_m_blocks, lane_idx)
        m_blocks_in_group = cute.arch.shuffle_sync(
          num_m_blocks_cumulative, cute.arch.WARP_SIZE - 1
        )
        group_end_tile += m_blocks_in_group * params.num_head
    if cutlass.const_expr(params.static_persistent):
      # Exhaustion leaves grp_start on the last real group, still a valid resume point since it can only be <= any later work id; deriving it from group_end_tile as one-shot mode does would not be, because on exhaustion that value is clamped to next_tile_idx + 1 and this cursor has to survive to the next decode.
      self._grp_base = grp_base
      self._grp_start = grp_start
      self._grp_nmb = num_m_blocks
      self._grp_nmb_cum = num_m_blocks_cumulative
    else:
      grp_start = group_end_tile - m_blocks_in_group * params.num_head
    return grp_base, grp_start, num_m_blocks, num_m_blocks_cumulative

  @cute.jit
  def _varlen_coord_map(self) -> WorkTileInfo:
    """Map self._tile_idx to (block, head, batch) via warp-level prefix sums."""
    params = self.params
    lane_idx = cute.arch.lane_idx()
    block, head_idx = Int32(0), Int32(0)
    # One-shot mode starts from a physical CTA index and folds cluster peers into one logical work id; the static-persistent walk already starts from a cluster index, so dividing again would drop every other item.
    next_tile_idx = self._tile_idx
    if cutlass.const_expr(
      params.cluster_shape_m > 1 and not params.static_persistent
    ):
      next_tile_idx = self._tile_idx // params.cluster_shape_m
    (batch_idx, group_start_tile, num_m_blocks,
     num_m_blocks_cumulative) = self._locate_group(next_tile_idx, lane_idx)
    is_valid = False
    if batch_idx >= params.num_batch:
      block, head_idx, batch_idx = Int32(0), Int32(0), Int32(params.num_batch)
    else:
      # The next problem to process is the first one that does not have ending tile position
      # that is greater than or equal to tile index.
      batch_idx_in_group = cute.arch.popc(
        cute.arch.vote_ballot_sync(
          group_start_tile +
          num_m_blocks_cumulative * params.num_head <= next_tile_idx
        )
      )
      batch_idx += batch_idx_in_group
      num_m_blocks_prev_lane = (
        0 if batch_idx_in_group == 0 else
        cute.arch.shuffle_sync(num_m_blocks_cumulative, batch_idx_in_group - 1)
      )
      num_m_blocks = cute.arch.shuffle_sync(num_m_blocks, batch_idx_in_group)
      mh_block = next_tile_idx - group_start_tile - num_m_blocks_prev_lane * params.num_head
      if cutlass.const_expr(params.lpt or params.head_swizzle):
        # L2-aware LPT scheduling within variable-length batches.  With a
        # cluster the M count is in cluster units, so scale back to rows.
        # The two constants are folded before the multiply so that at
        # ``cluster_shape_m == 1`` this is literally the SM90 expression.
        num_n_blocks = (
          num_m_blocks * (params.tile_shape_mn[0] * params.cluster_shape_m) //
          params.qhead_per_kvhead_packgqa // params.tile_shape_mn[1]
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
          nheads_in_l2 if nheads_in_l2 *
          (section_idx + 1) <= params.num_head else params.num_head -
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
      if cutlass.const_expr(params.static_persistent):
        # The walk is the loop; ``_is_first_block`` would retire it after one item, so exhaustion is signalled by batch_idx above instead.
        is_valid = batch_idx < params.num_batch
      else:
        is_valid = self._is_first_block and batch_idx < params.num_batch
      if cutlass.const_expr(
        params.cluster_shape_m > 1 and not params.use_cluster_idx
      ):
        # Physical-index decode: expand the cluster-unit block back to this CTA's own tile; cluster-indexed callers keep the spatial rank outside the work id and apply it themselves.
        bidx_in_cluster = cute.arch.block_in_cluster_idx()
        block = block * params.cluster_shape_m + bidx_in_cluster[0]
    return WorkTileInfo(
      (Int32(block), Int32(head_idx), Int32(batch_idx), Int32(0)), is_valid
    )

  @cute.jit
  def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
    return self._varlen_coord_map()

  @cute.jit
  def initial_work_tile_info(self, *, loc=None, ip=None):
    return self._varlen_coord_map()

  def prefetch_next_work(self, *, loc=None, ip=None):
    pass

  def advance_to_next_work(self, *, loc=None, ip=None):
    if const_expr(self.params.static_persistent):
      # Both CTA peers observe the same logical cluster stride, so a pair never splits across work items.
      self._tile_idx += cute.arch.cluster_dim()[0]
      return self.get_current_work()
    self._is_first_block = False
    return self.get_current_work()

  def producer_tail(self, *, loc=None, ip=None):
    pass

  def _carried_objs(self):
    """Values that must survive a staged-region boundary with this object."""
    objs = [self.params, self._tile_idx, self._split_idx]
    if const_expr(self.params.static_persistent):
      # The group cursor is live across the role loops, and omitting it is silent rather than fatal: ``__new_from_mlir_values__`` rebuilds through ``__init__``, which zeroes the cursor, and a zeroed cursor resolves a group the work id does not belong to.
      objs += [
        self._grp_base, self._grp_start, self._grp_nmb, self._grp_nmb_cum
      ]
    return objs

  def __extract_mlir_values__(self):
    values, self._values_pos = [], []
    for obj in self._carried_objs():
      obj_values = cutlass.extract_mlir_values(obj)
      values += obj_values
      self._values_pos.append(len(obj_values))
    return values

  def __new_from_mlir_values__(self, values):
    obj_list = []
    for obj, n_items in zip(self._carried_objs(), self._values_pos):
      obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
      values = values[n_items:]
    if const_expr(self.params.static_persistent):
      *ctor_args, grp_base, grp_start, grp_nmb, grp_nmb_cum = obj_list
      new = self.__class__(*ctor_args, loc=self._loc)
      new._grp_base = grp_base
      new._grp_start = grp_start
      new._grp_nmb = grp_nmb
      new._grp_nmb_cum = grp_nmb_cum
      return new
    return self.__class__(*obj_list, loc=self._loc)


# SM100 FMHA static persistent tile scheduler, for the D512 2-CTA forward.  The SM90 schedulers above have no (M, H, B) persistent stride and no notion of a CTA pair that must stay together, which this kernel needs because one CtaGroup.TWO cluster owns one 128-row query supertile; the x axis counts per-CTA 64-row tiles padded up to whole cluster pairs, so the persistent walk (start = bidx, stride = grid size) always hands peers (2q, 2q+1) the same work item with ranks 0/1.


class Sm100FmhaStaticTileSchedulerParams:
  """Persistent (M, H, B) work-space description; ``problem_shape_mbh`` is a work-item count per axis, not a tensor shape."""

  def __init__(
    self,
    is_persistent: bool,
    problem_shape_mbh: cute.Shape,
    *,
    loc=None,
    ip=None,
  ):
    self.is_persistent = is_persistent
    self.problem_shape_mbh = problem_shape_mbh
    self._loc = loc
    self._ip = ip

  def __extract_mlir_values__(self):
    values, self._values_pos = [], []
    for obj in [self.problem_shape_mbh]:
      obj_values = cutlass.extract_mlir_values(obj)
      values += obj_values
      self._values_pos.append(len(obj_values))
    return values

  def __new_from_mlir_values__(self, values):
    obj_list = []
    for obj, n_items in zip([self.problem_shape_mbh], self._values_pos):
      obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
      values = values[n_items:]
    return Sm100FmhaStaticTileSchedulerParams(
      self.is_persistent, *(tuple(obj_list)), loc=self._loc
    )


class Sm100FmhaStaticTileScheduler:
  """Persistent strided walk over an (M, H, B) work space; every role loop builds its own instance and advances it identically, so load, MMA, softmax, correction and epilogue agree on the tile without a broadcast."""

  def __init__(
    self,
    params: Sm100FmhaStaticTileSchedulerParams,
    current_work_linear_idx: Int32,
    blk_coord: cute.Coord,
    grid_shape: cute.Shape,
    *,
    loc=None,
    ip=None,
  ):
    self._params = params
    self._blk_coord = blk_coord
    self._grid_shape = grid_shape
    self._is_persistent = params.is_persistent
    self._current_work_linear_idx = current_work_linear_idx
    self._problem_shape_mbh = cute.make_layout(
      params.problem_shape_mbh, loc=loc, ip=ip
    )
    self._num_blocks = cute.size(self._problem_shape_mbh, loc=loc, ip=ip)
    self._is_first_block = True
    self.num_persistent_sm = cute.size(grid_shape, loc=loc, ip=ip)
    self._loc = loc
    self._ip = ip

  @staticmethod
  def get_grid_shape(
    params: Sm100FmhaStaticTileSchedulerParams,
    *,
    loc=None,
    ip=None,
  ) -> cute.Shape:
    """Grid for the persistent walk: min(SM count, work items) on x."""
    if params.is_persistent:
      hardware_info = cutlass.utils.HardwareInfo()
      sm_count = hardware_info.get_device_multiprocessor_count()
      return (
        cutlass.min(
          sm_count, cute.size(params.problem_shape_mbh, loc=loc, ip=ip)
        ),
        1,
        1,
      )
    return params.problem_shape_mbh

  @staticmethod
  def check_valid_work_for_seqlen_q(
    q_tiler: int,
    current_idx: Int32,
    seqlen_q: Int32,
  ) -> cutlass.Boolean:
    """Whether a padded x index still lands inside the query sequence: the last cluster pair of a short sequence decodes a tile that is scheduled but not valid, and every role must agree on which."""
    return current_idx * q_tiler < seqlen_q

  def get_current_work(
    self, *, loc=None, ip=None
  ) -> cutlass.utils.WorkTileInfo:
    is_valid = (
      self._current_work_linear_idx < self._num_blocks
      if self._is_persistent else self._is_first_block
    )

    blk_coord = (0, 0, 0)
    if self._is_persistent:
      blk_coord = self._problem_shape_mbh.get_hier_coord(
        self._current_work_linear_idx, loc=loc, ip=ip
      )
    else:
      blk_coord = self._blk_coord

    # The kernel role ABI is (m, 0, (head, batch)).
    cur_tile_coord = (blk_coord[0], 0, (blk_coord[1], blk_coord[2]))
    return cutlass.utils.WorkTileInfo(cur_tile_coord, is_valid)

  def initial_work_tile_info(self, *, loc=None, ip=None):
    return self.get_current_work(loc=loc, ip=ip)

  def advance_to_next_work(self, *, advance_count=1, loc=None, ip=None):
    if self._is_persistent:
      self._current_work_linear_idx += advance_count * self.num_persistent_sm
    self._is_first_block = False
    return self.get_current_work()

  def prefetch_next_work(self, *, loc=None, ip=None):
    """No-op: a static schedule has nothing to fetch ahead."""
    pass

  def producer_tail(self, *, loc=None, ip=None):
    """No-op: a static schedule owns no pipeline to drain."""
    pass

  def __extract_mlir_values__(self):
    values = cutlass.extract_mlir_values(self._params)
    values.extend(cutlass.extract_mlir_values(self._current_work_linear_idx))
    values.extend(cutlass.extract_mlir_values(self._blk_coord))
    values.extend(cutlass.extract_mlir_values(self._grid_shape))
    return values

  def __new_from_mlir_values__(self, values):
    assert len(values) == 10
    new_params = cutlass.new_from_mlir_values(self._params, values[0:3])
    new_current_work_linear_idx = cutlass.new_from_mlir_values(
      self._current_work_linear_idx, [values[3]]
    )
    new_blk_coord = cutlass.new_from_mlir_values(self._blk_coord, values[4:7])
    new_grid_shape = cutlass.new_from_mlir_values(self._grid_shape, values[7:])
    return Sm100FmhaStaticTileScheduler(
      new_params,
      new_current_work_linear_idx,
      new_blk_coord,
      new_grid_shape,
    )

  # Reached by the SM100 D512 forward, dK and dQ kernels; no SM90/SM80 path constructs this class.
  @staticmethod
  def to_underlying_arguments(
    o_shape: cute.Shape,
    cta_tiler: tuple[int, int, int],
    is_persistent: bool,
  ) -> Sm100FmhaStaticTileSchedulerParams:
    """Scheduler params from an ``(s, d, ((h_r, h_k), b))`` output shape; despite the field name the problem shape is laid out (M-blocks, H, B), matching the (x, y, z) launch-grid axes."""
    return Sm100FmhaStaticTileSchedulerParams(
      is_persistent,
      (
        cute.ceil_div(cute.size(o_shape[0]), cta_tiler[0]),
        cute.size(o_shape[2][0]),
        cute.size(o_shape[2][1]),
      ),
    )


# Sm100FmhaLptTileScheduler: longest-processing-time-first ordering over the same (M, H, B) work space as Sm100FmhaStaticTileScheduler, so the makespan is not set by a late-scheduled full-width causal tile.  A sibling of the static scheduler, not a replacement; reached by the D512 dK and dQ kernels through their tile_scheduler_cls selection, and nothing on the SM90/SM80 paths reaches it.
class Sm100FmhaLptTileScheduler:
  """Blocked-LPT reordering of a non-persistent (M, H, B) FMHA dispatch.

    A drop-in alternative to ``Sm100FmhaStaticTileScheduler`` on the same grid: same construction signature, same work-tile protocol, and exactly the same ten MLIR values of runtime state, with the reorder folded into the coordinates it hands out.  A linear dispatch position is decoded into (block, stream) through a "section" of neighbouring KV heads, and inside a section the block index is reversed so the longest causal tiles go first; ``remap_tile_coord`` is that decode as a pure function of (coord, problem shape).

    Sectioning rather than a global reversal is what bounds the concurrent KV working set at ``head_group * (K + V)``, and the L2 hit rate it protects is what keeps the operand rings latency-covered.

    ``cost_rises_with_block`` is the one thing the reorder cannot know by itself -- which end of the block axis carries the heavy tiles -- and the two D512 backward kernels sit on opposite ends of it: dQ's tile m runs m+1 KV steps, dK's K tile pair p sweeps Q blocks [2p, nQ).  A wrong setting is invisible to every coverage and bijection check, because it is still a bijection; it just reinstates the anti-LPT closing wave.  Default True is the dQ geometry.

    Three deliberate departures from upstream, each measured on this kernel: the reorder owns no runtime state (the three constants are trace-time Python ints, so a region crossing carries exactly the static scheduler's values and the decode stays a rematerialisable pure function); the section width is a constant rather than derived from an L2 budget, which at d=512 saturates before it discriminates; and no ``FastDivmodDivisor``, since one tile per CTA means the decode runs once and the divisions are free while the params carrying them are not.

    ``max_pairs`` is the wave-count gate: the reorder wins only where the closing wave is a large fraction of the makespan, and gated off is the identity map, so an oversized problem keeps exactly the static scheduler's dispatch.  It counts cluster-padded pairs, so a problem on the boundary reads one pair heavier and gates off -- the conservative direction.

    The domain is the cluster-padded pair domain, which is a correctness property: dividing the *logical* CTA tile count by the cluster is short by one whenever that count is odd, which drops every stream's last tile, duplicates a pair per stream boundary, and runs the last physical pair off the end of the section domain.

    Persistent mode is rejected at construction: the gate is a wave-count criterion and the win is the closing wave, both properties of the non-persistent dispatch.  A persistent variant is a different decode, not a flag on this one.
    """

  def __init__(
    self,
    params: Sm100FmhaStaticTileSchedulerParams,
    current_work_linear_idx: Int32,
    blk_coord: cute.Coord,
    grid_shape: cute.Shape,
    *,
    cluster_shape_m: int,
    head_group: int,
    max_pairs: int,
    cost_rises_with_block: bool = True,
    num_groups: int = 0,
    fold_into_grid: bool = False,
    loc=None,
    ip=None,
  ):
    """Same positional signature as ``Sm100FmhaStaticTileScheduler``, all four stored and round-tripped so the region-crossing value set is identical; the keyword constants are the section geometry, trace-time only."""
    assert not params.is_persistent, (
      "the blocked-LPT reorder is defined on the non-persistent dispatch "
      "(see class docstring); a persistent kernel takes "
      "Sm100FmhaStaticTileScheduler"
    )
    self._params = params
    self._blk_coord = blk_coord
    self._grid_shape = grid_shape
    self._current_work_linear_idx = current_work_linear_idx
    self._is_first_block = True
    self._cluster_shape_m = cluster_shape_m
    self._head_group = head_group
    self._max_pairs = max_pairs
    self._cost_rises_with_block = cost_rises_with_block
    self._num_groups = num_groups
    self._fold_into_grid = fold_into_grid
    self._loc = loc
    self._ip = ip

  # called by host
  @staticmethod
  def to_underlying_arguments(
    o_shape: cute.Shape,
    cta_tiler: tuple[int, int, int],
    is_persistent: bool,
  ) -> Sm100FmhaStaticTileSchedulerParams:
    """The reorder is a bijection on the static dispatch, so its params are the static scheduler's; the M recorded here is the logical CTA tile count, and ``remap_tile_coord`` is the only place that may re-derive the padded pair count by dividing by ``cluster_shape_m``."""
    return Sm100FmhaStaticTileScheduler.to_underlying_arguments(
      o_shape, cta_tiler, is_persistent
    )

  # called by host
  @staticmethod
  def get_grid_shape(
    params: Sm100FmhaStaticTileSchedulerParams,
    *,
    cluster_shape_m: int = 1,
    head_group: int = 0,
    num_groups: int = 0,
    fold_into_grid: bool = False,
    loc=None,
    ip=None,
  ) -> cute.Shape:
    """The dispatch domain, in whichever of the two realizations was asked for.

        ARITHMETIC (default): a bijection reorders the grid, it does not resize it; the launch still rounds up to the cluster and the reorder is defined on that rounded domain.

        FOLDED (``fold_into_grid``): the same permutation as a grid shape -- x carries ``cluster_shape_m`` CTAs of each of ``head_group`` streams, y the block pairs, z the remaining stream groups times the batch -- so walking it x-fastest *is* the section decode and the device side needs only a shift and a mask.
        """
    if fold_into_grid:
      m_pairs = cute.ceil_div(params.problem_shape_mbh[0], cluster_shape_m)
      return (
        cluster_shape_m * head_group,
        m_pairs,
        num_groups * params.problem_shape_mbh[2],
      )
    return Sm100FmhaStaticTileScheduler.get_grid_shape(params, loc=loc, ip=ip)

  @staticmethod
  def check_valid_work_for_seqlen_q(
    q_tiler: int,
    current_idx: Int32,
    seqlen_q: Int32,
  ) -> cutlass.Boolean:
    """Validity is a property of the coordinate, not of the dispatch order."""
    return Sm100FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
      q_tiler, current_idx, seqlen_q
    )

  @staticmethod
  @cute.jit
  def remap_tile_coord(
    coord: cute.Coord,
    problem_shape_mbh: cute.Shape,
    *,
    cluster_shape_m: int,
    head_group: int,
    max_pairs: int,
    cost_rises_with_block: bool = True,
    num_groups: int = 0,
    fold_into_grid: bool = False,
    loc=None,
    ip=None,
  ) -> cute.Coord:
    """Reorder ``(m, 0, (head, batch))`` in place on the (M, H, B) grid.

        Both CTAs of a cluster pair share one dispatch position and their m differ only in the low bits, so they stay a pair across the remap and m stays CTA-granular, which keeps varlen validity checks and every downstream TMA coordinate unaffected.

        The domain permuted is the *cluster-padded* one the launch dispatches, ``[0, round_up(M, cluster_shape_m))``, not the logical ``problem_shape_mbh[0]``.  The contract is a bijection on that padded domain, not that padding stays put: which physical CTA carries the padded tail row is free, since validity is decided from the returned coordinate and its stores are killed by the ``q < seqlen_q`` epilogue predicate.

        Postcondition every consumer relies on: the returned ``m`` is non-negative and its low ``log2(cluster_shape_m)`` bits equal those of ``coord[0]``.  The second half is load-bearing -- the kernel derives its rank in the 2-CTA MMA from the raw ``blockIdx.x`` low bit, so a remap that dropped it would put a CTA's MMA half and its row coordinate out of agreement -- and the first is why the dQ store predicates need no ``q >= 0`` lower bound.
        """
    num_head = problem_shape_mbh[1]
    # Ceil, not floor: the launch rounds the x-grid up to the cluster, so the physical pair count this decode spans is ceil(M / cluster_shape_m).  Under floor, odd M makes the flatten stride one pair short of the per-stream extent and the map stops being a bijection -- every stream's last tile loses its owner, adjacent streams collide on a shared pair, and the final physical pair decodes to block -1 -- and both divisors below can reach zero, where ceil keeps m_pairs >= 1 and hence section_width >= 1.
    if const_expr(fold_into_grid):
      # FOLDED realization: get_grid_shape already dispatched (cluster_shape_m * head_group, m_pairs, num_groups * B), so the hardware's x-fastest walk has performed the section decode and nothing is left but to read the coordinate apart -- x -> (rank in the pair, stream within the group), y -> the block pair, z -> (which group of streams, which batch).
      # Same permutation as the arithmetic arm below, position for position; what differs is the price, since the arithmetic decode runs ahead of the warp-specialised dispatch and every warp pays it.
      # The price of free: this arm PARTITIONS the stream axis instead of permuting it, so it needs head_group * num_groups == num_head exactly -- a partial last group would never be dispatched, and no store predicate can see a tile that was never launched.  Not checkable here (the head count is a dynamic Int32 at trace time), so it belongs to whoever picks the grouping; the D512 dK kernel enforces it in choose_lpt_grouping.  When it cannot be guaranteed, take the arithmetic arm, which has no divisibility requirement at all.
      # coord is (bidx, 0, (bidy, bidz)) as get_current_work assembles it.  Names here are deliberately disjoint from the arithmetic arm's: the DSL's AST pass records assignments from BOTH arms before pruning the const_expr fork, so a name the arithmetic arm assigns inside its *dynamic* `if` would read as "None before the if, Int32 inside" and raise DSLRuntimeError.
      folded_stream = coord[0] // cluster_shape_m
      folded_rank = coord[0] - folded_stream * cluster_shape_m
      # The pair index is y; re-attach this CTA's rank last, so both CTAs of a cluster keep consecutive m and the low bits of blockIdx.x -- the postcondition the 2-CTA MMA role depends on.
      folded_m = coord[2][0] * cluster_shape_m + folded_rank
      if const_expr(num_groups == 1):
        return (folded_m, coord[1], (folded_stream, coord[2][1]))
      folded_group = coord[2][1] % num_groups
      return (
        folded_m,
        coord[1],
        (
          folded_group * head_group + folded_stream,
          coord[2][1] // num_groups,
        ),
      )

    m_pairs = (problem_shape_mbh[0] + (cluster_shape_m - 1)) // cluster_shape_m
    num_stream = num_head * problem_shape_mbh[2]
    m_out = coord[0]
    head_out = coord[2][0]
    batch_out = coord[2][1]
    if m_pairs * num_stream <= Int32(max_pairs):
      stream_in = coord[2][0] + num_head * coord[2][1]
      # The section index is stream_in // head_group, not pair_idx // (head_group * m_pairs); they are the same number, since with pair_idx = a + m_pairs * stream_in and a < m_pairs, writing stream_in = head_group * q + r gives a + m_pairs * r <= head_group * m_pairs - 1, so the remainder never carries into the quotient.  Written this way the divisor is a trace-time int, hence a shift when it is a power of two, and pair_idx and section stop being needed at all.
      bidhb = stream_in // head_group
      section_base = bidhb * head_group
      # Upstream branches to a separate residual divisor for the last, short section; the min is the same thing without the second divisor, since a full section always has head_group streams left.
      section_width = cutlass.min(Int32(head_group), num_stream - section_base)
      offset = coord[
        0] // cluster_shape_m + m_pairs * (stream_in - section_base)
      # Which end of the block axis is heavy belongs to the consumer's causal geometry, not to the reorder: dQ's tile m runs m+1 KV steps, so cost RISES with the index and heaviest-first is descending, while dK's K tile pair p sweeps Q blocks [2p, nQ), so cost FALLS and heaviest-first is ascending.  The wrong end is not a correctness fault -- the map is a bijection either way and every coverage check still passes -- it silently inverts the reorder into the anti-LPT closing wave it exists to remove.
      # Trace-time fork; the arm not taken is not emitted.
      rank = offset // section_width
      if const_expr(cost_rises_with_block):
        block = (m_pairs - 1) - rank
      else:
        block = rank
      # ``offset - rank * section_width`` rather than ``offset % section_width``: both operands are non-negative here, so the remainder is the quotient's leftover and the division already computed for ``rank`` is reused, as the two lines below do for num_head.
      stream = section_base + (offset - rank * section_width)
      batch_out = stream // num_head
      head_out = stream - num_head * batch_out
      m_out = block * cluster_shape_m + coord[0] % cluster_shape_m
    return (m_out, coord[1], (head_out, batch_out))

  def get_current_work(
    self, *, loc=None, ip=None
  ) -> cutlass.utils.WorkTileInfo:
    """The static scheduler's work tile with the reorder already applied.  Call it from inside each warp role's branch, where the decode's inputs are rematerialisable reads and its outputs stay region-local; fetching once before the warp split instead pins the division results as block arguments and costs spill."""
    cur_tile_coord = self.remap_tile_coord(
      (self._blk_coord[0], 0, (self._blk_coord[1], self._blk_coord[2])),
      self._params.problem_shape_mbh,
      cluster_shape_m=self._cluster_shape_m,
      head_group=self._head_group,
      max_pairs=self._max_pairs,
      cost_rises_with_block=self._cost_rises_with_block,
      num_groups=self._num_groups,
      fold_into_grid=self._fold_into_grid,
      loc=loc,
      ip=ip,
    )
    return cutlass.utils.WorkTileInfo(cur_tile_coord, self._is_first_block)

  def initial_work_tile_info(self, *, loc=None, ip=None):
    return self.get_current_work(loc=loc, ip=ip)

  def advance_to_next_work(self, *, advance_count=1, loc=None, ip=None):
    """One tile per CTA: after the first tile there is no more work."""
    self._is_first_block = False
    return self.get_current_work()

  def prefetch_next_work(self, *, loc=None, ip=None):
    """No-op, as in the static scheduler."""

  def producer_tail(self, *, loc=None, ip=None):
    """No-op, as in the static scheduler."""

  def __extract_mlir_values__(self):
    values = cutlass.extract_mlir_values(self._params)
    values.extend(cutlass.extract_mlir_values(self._current_work_linear_idx))
    values.extend(cutlass.extract_mlir_values(self._blk_coord))
    values.extend(cutlass.extract_mlir_values(self._grid_shape))
    return values

  def __new_from_mlir_values__(self, values):
    assert len(values) == 10
    new_params = cutlass.new_from_mlir_values(self._params, values[0:3])
    new_current_work_linear_idx = cutlass.new_from_mlir_values(
      self._current_work_linear_idx, [values[3]]
    )
    new_blk_coord = cutlass.new_from_mlir_values(self._blk_coord, values[4:7])
    new_grid_shape = cutlass.new_from_mlir_values(self._grid_shape, values[7:])
    return Sm100FmhaLptTileScheduler(
      new_params,
      new_current_work_linear_idx,
      new_blk_coord,
      new_grid_shape,
      cluster_shape_m=self._cluster_shape_m,
      head_group=self._head_group,
      max_pairs=self._max_pairs,
      cost_rises_with_block=self._cost_rises_with_block,
      num_groups=self._num_groups,
      fold_into_grid=self._fold_into_grid,
    )

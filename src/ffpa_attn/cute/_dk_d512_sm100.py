# Copyright (c) DefTruth, qyjdef@163.com
# Copyright (c) Butterfingrz，13524387014@163.com
#
# Adapted from the SM100 head-dim 256 specialized implementation
# in https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/sm100_hd256_2cta_fmha_backward_dkdvkernel.py
#
# SM100 (Blackwell) backward dK for FFPA attention — only head_dim=512.
#
# Produces dK alone: dK = (dS^T @ Q^T) * scale with dS = P * (dP - dpsum), P
# from the forward LSE, S^T = K @ Q^T and dP^T = V @ dO^T.  The dV half of the
# fused donor is gone, but its V / dP / dpsum / dS / Q^T chain stays because dK
# closes over it.  Like dV the kernel is KV-stationary, GQA folds into the
# sweep rather than the grid (trip count = Q blocks x h_r), and dK reduces
# entirely in TMEM with the scale delayed to one fp32 multiply in the epilogue.
#
# Design (12 warps / 384 threads, per-CTA tile (64, 64, 512)):
#   - LOAD (warp 9) TMAs K + Q, cp.asyncs LSE and owns the TMEM allocation; dO
#     LOAD (warp 10) takes V + dO + sum(O*dO) and Q^T LOAD (warp 11) the
#     transposed-Q stream the dSQ edge consumes — one issuer per stream.
#   - MMA (warp 8): S^T = K @ Q^T and dP^T = V @ dO^T on a (128, 64, 128)
#     generation tile, four generations per KV block; dK += dS^T @ Q^T is
#     (128, 256, 64), two N slices because tcgen05 caps N at 256.  K parks as
#     a TMEM-A operand, non-causally V too.
#   - COMPUTE (warps 0-7) forms P and dS, then drains dK, which needs all eight.
#   - TMEM: dK [0,256) as 2 x 128 columns, S [256,288), dP [288,320), parked
#     K/V generations out to 512.  The dS^T ring and the epilogue arena both
#     lease sK, each lifetime-disjoint from the generation it covers.
#   - Causal and non-causal are different codegen, not one branch: register
#     split, ring depths, TMEM parking and issue order fork on the same seam.
#
# Constraints:
#   - cta_tiler == (64, 64, 512), and split_head is required
#   - cluster (2, 1, 1)
#   - lse_log2 is log2-domain; batch must agree across Q, K and V
#   - No persistent or CLC scheduling
#   - Sliding window: ctor-accepted but unreachable (the wrapper passes None)

import math
from functools import partial
from typing import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils_basic
from cutlass import const_expr, cute, pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.typing import Int32
from cutlass.pipeline import (
  Agent,
  CooperativeGroup,
  pipeline_init_arrive,
  pipeline_init_wait,
)

from .utils import copy_utils
from . import utils
from .utils.cute_dsl_utils import assume_tensor_aligned
from .utils.blackwell_helpers import (
  SM100_SMEM_CAPACITY_BYTES,
  SM100_TMEM_CAPACITY_COLUMNS,
  gemm_ptx_w_idx,
)
from .utils.block_info import BlockInfo
# No tmem_offset import: every TMEM tensor here sits at lane 0 (dQ needs it).
from .utils.hd512_helpers import (
  check_tmem_intervals,
  reg_to_smem_mma128x128_2cta,
  split_wg,
)
from .utils.named_barrier import NamedBarrierBwdDKSm100Hd512
from .utils.seqlen_info import SeqlenInfoQK
from .utils.tile_scheduler import (
  Sm100FmhaLptTileScheduler as FmhaLptTileScheduler,
  Sm100FmhaStaticTileScheduler as FmhaStaticTileScheduler,
  Sm100FmhaStaticTileSchedulerParams as FmhaStaticTileSchedulerParams,
)


class FFPAAttnBwdDKSm100D512:
  """SM100 D512 backward dK: KV-stationary, dK = dS^T @ Q alone (no dV/dQ).

  The fused donor's V / dP / dpsum / dS / Q^T chain stays because dK closes
  over it; only the dV half is gone (see the module header).
  """
  arch = 100

  # Declared topology; a K tile of 128 would need 640 of 512 TMEM columns.
  TARGET_HEAD_DIM = 512
  TARGET_CTA_TILER = (64, 64, 512)
  TARGET_CLUSTER_SHAPE_MNK = (2, 1, 1)

  # Generation tile: K mode = D gen width; an unsliced arena does not fit SMEM.
  TARGET_MMA_SLICE = (128, 64, 128)
  TARGET_D_GENERATIONS = 4

  # dK: TARGET_DK_SLICES regions of SLICE_STRIDE cols from 0; S, dP follow.
  TARGET_DK_SLICES = 2
  TARGET_TMEM_DK_BASE = 0
  TARGET_TMEM_DK_SLICE_STRIDE = 128
  TARGET_TMEM_S_BASE = 256
  TARGET_TMEM_DP_BASE = 288

  # Order matters: LPT_MAX_PAIRS reads these at class-body eval; 148 SMs/2 = 74.
  SM_PAIRS = 74
  # Past this many waves greedy dispatch balances the tail; reorder is marginal.
  LPT_WAVE_GATE = 8.0
  # Device-side re-test (counts padded pairs); one number so they cannot drift.
  LPT_MAX_PAIRS = int(LPT_WAVE_GATE * SM_PAIRS) - 1
  # Section width in cluster slots; MEASURED by sweep, not from capacity.
  TARGET_SECTION_SLOTS = 64

  def __init__(
    self,
    acc_dtype: type[cutlass.Numeric],
    cta_tiler: tuple[int, int, int],
    is_causal: bool,
    window_size_left: int | None,
    window_size_right: int | None,
    is_persistent: bool = False,
    split_head: bool = False,
    use_clc_scheduler: bool = False,
    lpt_head_group: int = 0,
    lpt_num_groups: int = 0,
  ):
    """Configure the D512 dK kernel; schedule decisions are trace-time."""
    # is_persistent / split_head keep the call shape; asserted, never stored.
    self.acc_dtype = acc_dtype
    # Placeholders; __call__ sets them from the traced tensors.
    self.q_dtype = None
    self.k_dtype = None
    self.v_dtype = None
    self.do_dtype = None
    self.ds_dtype = None
    assert tuple(cta_tiler) == self.TARGET_CTA_TILER, (
      "SM100 backward with head_dim=512 requires the declared CTA tile "
      f"{self.TARGET_CTA_TILER}, got {tuple(cta_tiler)}. The TMEM ledger, "
      "the dK slice stride and the generation arithmetic are all derived "
      "from it, so a different tile silently invalidates the ledger rather "
      "than failing loudly."
    )
    assert cta_tiler[2] == self.TARGET_HEAD_DIM
    assert cta_tiler[2] % self.TARGET_DK_SLICES == 0, (
      f"head_dim {cta_tiler[2]} does not divide into {self.TARGET_DK_SLICES} dK slices"
    )
    assert split_head, (
      "SM100 backward with head_dim=512 requires split_head=True. It is the "
      "four-generation D reduction: the donor's single-generation form "
      "keeps every operand's whole head_dim resident and does not fit the "
      "SMEM ceiling. Passing False would select a "
      "shape that cannot launch, so it is refused rather than silently "
      "reinterpreted."
    )
    assert cta_tiler[2] % self.TARGET_MMA_SLICE[2] == 0
    self.d_generations = cta_tiler[2] // self.TARGET_MMA_SLICE[2]
    assert self.d_generations == self.TARGET_D_GENERATIONS, (
      f"head_dim {cta_tiler[2]} over a {self.TARGET_MMA_SLICE[2]}-wide "
      f"generation is {self.d_generations} iterations, not the declared "
      f"{self.TARGET_D_GENERATIONS}"
    )
    assert not use_clc_scheduler, (
      "SM100 backward with head_dim=512 does not support the CLC/persistent "
      "scheduler; the path was removed in port round seven after the dQ "
      "campaign measured persistent codegen 2-3x slower"
    )
    assert not is_persistent, (
      "SM100 backward with head_dim=512 does not support persistent "
      "scheduling: is_persistent only ever reached the static tile "
      "scheduler's dead parameter, and the kernel launches one block per tile"
    )
    self.cta_tiler = cta_tiler
    self.tile_m = cta_tiler[0]
    self.tile_n = cta_tiler[1]
    self.tile_hdim = cta_tiler[2]
    # For S, one D generation: the reduction cut is TMEM-free and buys SMEM.
    self.KQ_mma_tiler = (
      cta_tiler[1] * 2,
      cta_tiler[0],
      self.TARGET_MMA_SLICE[2],
    )
    # For dP.  Same generation tile, same reduction.
    self.VdO_mma_tiler = (
      cta_tiler[1] * 2,
      cta_tiler[0],
      self.TARGET_MMA_SLICE[2],
    )
    assert self.KQ_mma_tiler == self.TARGET_MMA_SLICE, (
      f"declared generation tile {self.TARGET_MMA_SLICE} is not the tile "
      f"the constructor builds, {self.KQ_mma_tiler}"
    )
    # N must be sliced: tcgen05 caps N <= 256; the donor's N=512 raises OpError.
    self.dSQ_mma_tiler = (
      cta_tiler[1] * 2,
      cta_tiler[2] // self.TARGET_DK_SLICES,
      cta_tiler[0],
    )
    # No dSK edge (no dQ here): dS publishes only transposed, for the dSQ edge.
    self.cluster_shape_mn = self.TARGET_CLUSTER_SHAPE_MNK[:2]
    self.cluster_shape_mnk = (
      *self.cluster_shape_mn, 1
    )  # type: ignore[assignment]
    self.is_causal = is_causal
    # Trace-time ints, else divmod in 4 prologues; grid z = groups*B, 0 = none.
    self.lpt_head_group = int(lpt_head_group) if is_causal else 0
    self.lpt_num_groups = int(lpt_num_groups) if is_causal else 0
    # Scheduler pick: the kernel holds a class, roles see coords; (0,0) = stock.
    if self.lpt_head_group:
      self.tile_scheduler_cls = FmhaLptTileScheduler
    else:
      self.tile_scheduler_cls = FmhaStaticTileScheduler
    self.use_lpt_scheduler = self.tile_scheduler_cls is FmhaLptTileScheduler
    self.window_size_left: int = -1 if window_size_left is None else window_size_left
    self.window_size_right: int = -1 if window_size_right is None else window_size_right
    self.has_sliding_window = False
    if self.window_size_left > 0 or self.window_size_right > 0:
      self.has_sliding_window = True
    if self.is_causal:
      self.window_size_right = 0

    self.compute_warp_ids = (0, 1, 2, 3, 4, 5, 6, 7)
    self.mma_warp_id = 8
    self.load_warp_id = 9
    # One TMA issuer per stream: warp 9 K+Q+LSE, 10 V+dO+sum_OdO, 11 QT.
    self.load_do_warp_id = 10
    self.load_qt_warp_id = 11
    # All 12 warps hold a role: dispatch is a closed elif chain, no empty warp.
    self.num_compute_warps = len(self.compute_warp_ids)

    self.tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

    self.threads_per_warp = cute.arch.WARP_SIZE
    self.threads_per_cta = self.threads_per_warp * len((
      *self.compute_warp_ids,
      self.mma_warp_id,
      self.load_warp_id,
      self.load_do_warp_id,
      self.load_qt_warp_id,
    ))

    self.cta_sync_bar_id = int(NamedBarrierBwdDKSm100Hd512.CtaSync)
    self.tmem_alloc_sync_bar_id = int(NamedBarrierBwdDKSm100Hd512.TmemPtr)
    self.compute_sync_bar_id = int(NamedBarrierBwdDKSm100Hd512.Compute)
    self.epilogue_sync_bar_id = int(NamedBarrierBwdDKSm100Hd512.Epilogue)
    self.epilogue_arena_bar_id = int(NamedBarrierBwdDKSm100Hd512.EpilogueArena)

    # Re-derived: the donor's expressions land S@512 / dP@544, past 512 columns.
    self.tmem_dK_offset = self.TARGET_TMEM_DK_BASE
    self.tmem_dP_offset = self.TARGET_TMEM_DP_BASE
    self.tmem_S_offset = self.TARGET_TMEM_S_BASE
    # K parked as TMEM-A, 64 cols/gen, 3 fill the 192 spare; sK stays (utccp).
    self.ktmem_gens = 3
    # Measured: causal 3, non-causal 1; floor 1, the dS lease needs gen 0 dead.
    self.ktmem_use_gens = self.ktmem_gens if self.is_causal else 1
    self.tmem_K_base = self.TARGET_TMEM_DP_BASE + 32
    self.tmem_K_cols_per_gen = self.TARGET_MMA_SLICE[2] * 16 // 32
    assert (
      self.tmem_K_base + self.ktmem_gens * self.tmem_K_cols_per_gen
      <= SM100_TMEM_CAPACITY_COLUMNS
    ), "the TMEM K region runs past the 512-column capacity"
    # V parked on KQ's TMEM-A contract: non-causal 2 gens [384,512), causal 0.
    self.vtmem_gens = 2
    self.vtmem_use_gens = 0 if self.is_causal else 2
    self.tmem_V_base = (
      self.tmem_K_base + self.ktmem_use_gens * self.tmem_K_cols_per_gen
      if self.vtmem_use_gens > 0 else self.tmem_K_base
    )
    # Bounds BUILT faces (*_gens), not issued: causal (K2,V1) would need 576.
    assert (
      self.tmem_V_base + self.vtmem_gens * self.tmem_K_cols_per_gen
      <= SM100_TMEM_CAPACITY_COLUMNS
    ), "the TMEM V region runs past the 512-column capacity"

    # The only disjointness check; the asserts above bound single regions only.
    check_tmem_intervals(
      self.tmem_region_intervals(self.TARGET_TMEM_DK_SLICE_STRIDE)
    )

    # Non-causal only: uniform-192 measured a causal loss, a non-causal win.
    self.despill_wg2 = not self.is_causal
    self.num_regs_compute = 128
    # 192 unspills the mma warp (dense LDL/STL at 128); wg2 setmaxnreg uniform.
    self.num_regs_mma = 192 if self.despill_wg2 else 128
    self.num_regs_load = 96

    # Per-warpgroup budget from one 64K file: the three sum <= 512 (448 / 384).
    assert (
      self.num_regs_compute * (self.num_compute_warps // 4) +
      max(self.num_regs_mma, self.num_regs_load) <= 512
    )
    # Issuers must form a warpgroup: setmaxnreg on a lone warp is ILLEGAL.
    issuer_warp_ids = (
      self.mma_warp_id,
      self.load_warp_id,
      self.load_do_warp_id,
      self.load_qt_warp_id,
    )
    assert self.num_compute_warps % 4 == 0
    assert issuer_warp_ids == tuple(
      range(self.num_compute_warps, self.num_compute_warps + 4)
    )

    self.buffer_align_bytes = 128
    # Arena in dead sK: K/V released before the last dK MMA, then the dK token.
    # Always True; else arm + 1-elt sdK_epi stub keep the donor SharedStorage.
    self.alias_epilogue_onto_sK = True

  def _get_tiled_mma(self):
    """The 3 MMA edges; all K-major but dSQ's B (QT, MN); TMEM-A at its site."""
    # S.T = K @ Q.T
    kq = sm100_utils_basic.make_trivial_tiled_mma(
      self.k_dtype,
      tcgen05.OperandMajorMode.K,
      tcgen05.OperandMajorMode.K,
      self.acc_dtype,
      self.cta_group,
      self.KQ_mma_tiler[:2],
    )
    # dP.T = V @ dO.T
    vdo = sm100_utils_basic.make_trivial_tiled_mma(
      self.v_dtype,
      tcgen05.OperandMajorMode.K,
      tcgen05.OperandMajorMode.K,
      self.acc_dtype,
      self.cta_group,
      self.VdO_mma_tiler[:2],
    )
    # dK += dS.T @ Q.T
    dsq = sm100_utils_basic.make_trivial_tiled_mma(
      self.ds_dtype,
      tcgen05.OperandMajorMode.K,
      tcgen05.OperandMajorMode.MN,
      self.acc_dtype,
      self.cta_group,
      self.dSQ_mma_tiler[:2],
    )
    self.KQ_tiled_mma = kq
    self.VdO_tiled_mma = vdo
    self.dSQ_tiled_mma = dsq
    return kq, vdo, dsq

  def tmem_region_intervals(self, region_columns):
    """Column ledger; region_columns = declared dK slice stride (not S/dP)."""
    s_columns = self.KQ_mma_tiler[1] // self.TARGET_CLUSTER_SHAPE_MNK[0]
    dp_columns = self.VdO_mma_tiler[1] // self.TARGET_CLUSTER_SHAPE_MNK[0]
    intervals = {
      "S": (self.TARGET_TMEM_S_BASE, self.TARGET_TMEM_S_BASE + s_columns),
      "dP": (self.TARGET_TMEM_DP_BASE, self.TARGET_TMEM_DP_BASE + dp_columns),
    }
    for slice_index in range(self.TARGET_DK_SLICES):
      start = self.TARGET_TMEM_DK_BASE + slice_index * self.TARGET_TMEM_DK_SLICE_STRIDE
      intervals[f"dK{slice_index}"] = (start, start + region_columns)
    # Issued builds only: K and V faces overlap; only one side is ever issued.
    for gen_index in range(self.ktmem_use_gens):
      start = self.tmem_K_base + gen_index * self.tmem_K_cols_per_gen
      intervals[f"K{gen_index}"] = (start, start + self.tmem_K_cols_per_gen)
    for gen_index in range(self.vtmem_use_gens):
      start = self.tmem_V_base + gen_index * self.tmem_K_cols_per_gen
      intervals[f"V{gen_index}"] = (start, start + self.tmem_K_cols_per_gen)
    return intervals

  def dk_epilogue_tiling(self, element_width_bits):
    """Drain geometry for one dK slice, derived once (donor did it twice)."""
    num_wgs = self.num_compute_warps // 4
    slice_columns = self.cta_tiler[2] // self.TARGET_DK_SLICES
    epi_columns = math.gcd(
      128 // (element_width_bits // 8), slice_columns // num_wgs
    )
    stages_per_wg = (slice_columns // num_wgs) // epi_columns
    return (
      num_wgs,
      slice_columns,
      epi_columns,
      (self.cta_tiler[1], epi_columns),
      num_wgs * stages_per_wg,
    )

  def _setup_attributes(self):
    """Schedule decisions are trace-time: each forks a kernel, not a branch."""
    # Q ring forks depth and granularity: non-causal 2 tok x 2 gens, causal 3x1.
    self.load_mma_Q_stage = 3 if self.is_causal else 2
    self.q_gens_per_token = 1 if self.is_causal else 2
    self.load_mma_K_stage = 1
    self.load_mma_V_stage = 1
    self.load_mma_QT_stage = 1
    # dO ring 3 causal / 4 non-causal: the dP burst outruns non-causal refill.
    self.load_mma_dO_stage = 3 if self.is_causal else 4
    self.load_compute_LSE_stage = 1
    self.load_compute_sum_OdO_stage = 1
    self.mma_compute_S_stage = 1
    self.mma_compute_dP_stage = 1
    self.compute_mma_dS_stage = 2
    # dS ring leases sK's front (its own field would not fit); gen 0 dead there.
    self.alias_dst_onto_sK = True
    # Measured: [S,dP,dK] pays only on causal + the 2-deep dS ring; else donor.
    self.dp_before_dk = self.is_causal
    # Fused on causal: unfused, each 3-deep ring gets one short reuse window.
    self.interleave_s_dp = self.is_causal
    # Python bools, not Int32: the assert and const_expr need trace values.
    # TODO: the unfused [S, dP, dK] schedule (dp_before_dk) was never built.
    assert not self.dp_before_dk or self.interleave_s_dp, (
      "dp_before_dk implies interleave_s_dp: the unfused [S, dP, dK] "
      "schedule has never been compiled or measured and mma does "
      "not emit it (port-log entry DK-12b)"
    )
    # AccumulatorHandoff: one dK token (tile-final dSQ commit), payload tdKtdK.
    self.mma_compute_dK_stage = 1

    self.cta_group = tcgen05.CtaGroup.TWO

  @classmethod
  def choose_lpt_grouping(cls, seq_k, heads_kv, batch, block_k, head_dim):
    """Host-side blocked-LPT choice (G, H // G); (0, 0) means the stock grid."""
    if not seq_k or heads_kv < 2 or batch < 1:
      return 0, 0
    # Ceil twice: negating the result of // floors (1 at n_k=3, even n_k hides).
    n_k = -(-seq_k // block_k)
    clusters = -(-n_k // 2)
    if clusters < 1:
      return 0, 0
    if clusters * heads_kv * batch >= cls.LPT_WAVE_GATE * cls.SM_PAIRS:
      return 0, 0
    # A section WIDTH, not a floor: 6 of 7 sweep-measured optima lie at/below.
    group, best_err = heads_kv, None
    g = heads_kv
    while g >= 2:
      err = abs(g * clusters - cls.TARGET_SECTION_SLOTS)
      if best_err is None or err < best_err:
        group, best_err = g, err
      if g % 2:
        break
      g //= 2
    assert heads_kv % group == 0, (
      f"blocked-LPT head_group {group} must divide heads_kv {heads_kv}: "
      "the folded grid partitions the head axis and would drop the "
      f"last {heads_kv % group} KV heads"
    )
    return group, heads_kv // group

  def make_tile_scheduler(self, tile_sched_params):
    """Trace-time scheduler fork, one arm; not @cute.jit so it inlines."""
    blk_idx = cute.arch.block_idx()
    if const_expr(self.use_lpt_scheduler):
      return FmhaLptTileScheduler(
        tile_sched_params,
        blk_idx[0],
        blk_idx,
        cute.arch.grid_dim(),
        cluster_shape_m=self.cluster_shape_mnk[0],
        head_group=self.lpt_head_group,
        max_pairs=self.LPT_MAX_PAIRS,
        # dK cost FALLS with block index; a reversed order still passes.
        cost_rises_with_block=False,
        num_groups=self.lpt_num_groups,
        fold_into_grid=True,
      )
    return FmhaStaticTileScheduler(
      tile_sched_params, blk_idx[0], blk_idx, cute.arch.grid_dim()
    )

  @cute.jit
  def __call__(
    self,
    Q: cute.Tensor,
    K: cute.Tensor,
    V: cute.Tensor,
    dO: cute.Tensor,
    lse_log2: cute.Tensor,
    dpsum: cute.Tensor,
    dK: cute.Tensor,
    scale_softmax: cutlass.Float32,
    cumulative_s_q: cute.Tensor | None,
    cumulative_s_k: cute.Tensor | None,
    stream: cuda.CUstream = None,
  ):
    """Trace entry: build layouts, TMA atoms and SharedStorage, then launch."""
    assert (cumulative_s_q is None) == (cumulative_s_k is None), (
      "varlen dK requires both cumulative_s_q and cumulative_s_k"
    )
    # Normalize: rank-3/4 in -> (B,S,H_k,H_r,D) and (S,((H_r,H_k),B)) views.
    varlen = cumulative_s_q is not None
    q_rank = cute.rank(Q.layout)
    # Rank 3 = packed (total,H,D); rank 4 = dense or (1,total,H,D) packed.
    assert q_rank == 3 or q_rank == 4, (
      "SM100 backward with head_dim=512 expects rank-3 packed or rank-4 dense "
      "operands"
    )
    if const_expr(q_rank == 3):
      h_q_in = Q.shape[1]
      h_k_in = K.shape[1]
    else:
      h_q_in = Q.shape[2]
      h_k_in = K.shape[2]
    h_r_in = h_q_in // h_k_in
    if const_expr(cumulative_s_q is not None):
      b_stats = cumulative_s_q.shape[0] - 1
    else:
      b_stats = Q.shape[0]
    Q, K, V, dK, dO = [assume_tensor_aligned(t) for t in (Q, K, V, dK, dO)]
    Q = utils.as_bshkrd_tensor(Q, h_k_in, h_r_in, varlen)
    K = utils.as_bshkrd_tensor(K, h_k_in, 1, varlen)
    V = utils.as_bshkrd_tensor(V, h_k_in, 1, varlen)
    dK = utils.as_bshkrd_tensor(dK, h_k_in, 1, varlen)
    dO = utils.as_bshkrd_tensor(dO, h_k_in, h_r_in, varlen)
    scaled_LSE = utils.as_shhb_tensor(lse_log2, h_k_in, h_r_in, b_stats, varlen)
    sum_OdO = utils.as_shhb_tensor(dpsum, h_k_in, h_r_in, b_stats, varlen)
    h_r = Q.shape[3]
    h_k = Q.shape[2]
    if const_expr(cumulative_s_q is not None):
      b = cumulative_s_q.shape[0] - 1
    elif const_expr(cumulative_s_k is not None):
      b = cumulative_s_k.shape[0] - 1
    else:
      b = Q.shape[0]
    hb = ((h_r, h_k), b)
    problem_shape = (
      Q.shape[1],
      K.shape[1],
      Q.shape[4],
      hb,
    )
    # (b, s, h_k, h_r, d) -> (s, d, ((h_r, h_k), b))
    Q = cute.make_tensor(
      Q.iterator,
      cute.make_layout(
        (Q.shape[1], Q.shape[4], hb),
        stride=(
          cute.assume(Q.stride[1], divby=64),
          Q.stride[4],
          (
            (Q.stride[3], Q.stride[2]),
            0 if cumulative_s_q is not None else
            cute.assume(Q.stride[0], divby=64),
          ),
        ),
      ),
    )
    # (b, s, h_k, 1, d) -> (s, d, ((1, h_k), b))
    K = cute.make_tensor(
      K.iterator,
      cute.make_layout(
        (K.shape[1], K.shape[4], hb),
        stride=(
          cute.assume(K.stride[1], divby=64),
          K.stride[4],
          (
            (0, K.stride[2]),
            0 if cumulative_s_k is not None else
            cute.assume(K.stride[0], divby=64),
          ),
        ),
      ),
    )
    # (b, s, h_k, 1, d) -> (s, d, ((1, h_k), b))
    V = cute.make_tensor(
      V.iterator,
      cute.make_layout(
        (V.shape[1], V.shape[4], hb),
        stride=(
          cute.assume(V.stride[1], divby=64),
          V.stride[4],
          (
            (0, V.stride[2]),
            0 if cumulative_s_k is not None else
            cute.assume(V.stride[0], divby=64),
          ),
        ),
      ),
    )
    # (s, d, ((h_r, h_k), b)) -> (d, s, ((h_r, h_k), b))
    QT = cute.make_tensor(
      Q.iterator,
      cute.make_layout(
        (Q.shape[1], Q.shape[0], Q.shape[2]),
        stride=(
          Q.stride[1],
          Q.stride[0],
          Q.stride[2],
        ),
      ),
    )
    dK = cute.make_tensor(
      dK.iterator,
      cute.make_layout(
        (dK.shape[1], dK.shape[4], hb),
        stride=(
          cute.assume(dK.stride[1], divby=64),
          dK.stride[4],
          (
            (0, dK.stride[2]),
            0 if cumulative_s_k is not None else
            cute.assume(dK.stride[0], divby=64),
          ),
        ),
      ),
    )
    # (s, d, ((h_r, h_k), b))
    dO = cute.make_tensor(
      dO.iterator,
      cute.make_layout(
        (dO.shape[1], dO.shape[4], hb),
        stride=(
          cute.assume(dO.stride[1], divby=64),
          dO.stride[4],
          (
            (dO.stride[3], dO.stride[2]),
            0 if cumulative_s_q is not None else
            cute.assume(dO.stride[0], divby=64),
          ),
        ),
      ),
    )

    # Trace-time only: dtypes, major modes and layouts come from traced tensors.
    # dS is quantized to Q's dtype before publish: it is the dSQ edge's A dtype.
    self.q_dtype = Q.element_type
    self.k_dtype = K.element_type
    self.v_dtype = V.element_type
    self.do_dtype = dO.element_type
    self.ds_dtype = self.q_dtype

    self.q_major_mode = cutlass.utils.LayoutEnum.from_tensor(Q).mma_major_mode()
    self.k_major_mode = cutlass.utils.LayoutEnum.from_tensor(K).mma_major_mode()
    self.dk_major_mode = cutlass.utils.LayoutEnum.from_tensor(dK
                                                              ).mma_major_mode()
    self.v_major_mode = cutlass.utils.LayoutEnum.from_tensor(V).mma_major_mode()
    self.do_major_mode = cutlass.utils.LayoutEnum.from_tensor(dO
                                                              ).mma_major_mode()

    if const_expr(self.q_major_mode != tcgen05.OperandMajorMode.K):
      raise RuntimeError(
        f"The layout of q is not supported: {self.q_major_mode}"
      )
    if const_expr(self.k_major_mode != tcgen05.OperandMajorMode.K):
      raise RuntimeError("The layout of k is not supported")
    if const_expr(self.dk_major_mode != tcgen05.OperandMajorMode.K):
      raise RuntimeError("The layout of dk is not supported")
    if const_expr(self.v_major_mode != tcgen05.OperandMajorMode.K):
      raise RuntimeError("The layout of v is not supported")
    if const_expr(self.do_major_mode != tcgen05.OperandMajorMode.K):
      raise RuntimeError("The layout of do is not supported")

    self._setup_attributes()

    # S = KQ, dP = VdO, dK = dSQ.  One construction, in _get_tiled_mma.
    KQ_tiled_mma, VdO_tiled_mma, dSQ_tiled_mma = self._get_tiled_mma()
    # KQ's TMEM-A face: only a_source differs; sK's stages are the utccp source.
    KQ_tmem_tiled_mma = sm100_utils_basic.make_trivial_tiled_mma(
      self.k_dtype,
      tcgen05.OperandMajorMode.K,
      tcgen05.OperandMajorMode.K,
      self.acc_dtype,
      self.cta_group,
      self.KQ_mma_tiler[:2],
      tcgen05.OperandSource.TMEM,
    )
    self.KQ_tmem_tiled_mma = KQ_tmem_tiled_mma
    # Layouts from their consuming MMA edges; KT_gen_layout is one gen (utccp).
    self.KT_gen_layout = sm100_utils_basic.make_smem_layout_a(
      self.KQ_tmem_tiled_mma,
      self.KQ_mma_tiler,
      self.k_dtype,
      1,
    )
    # K/V resident: buffer count is generations, not depth; pipelines stay 1.
    self.sK_layout = sm100_utils_basic.make_smem_layout_a(
      self.KQ_tiled_mma,
      self.KQ_mma_tiler,
      self.k_dtype,
      self.d_generations,
    )
    self.sQ_layout = sm100_utils_basic.make_smem_layout_b(
      self.KQ_tiled_mma,
      self.KQ_mma_tiler,
      self.q_dtype,
      self.load_mma_Q_stage * self.q_gens_per_token,
    )
    self.sV_layout = sm100_utils_basic.make_smem_layout_a(
      self.VdO_tiled_mma,
      self.VdO_mma_tiler,
      self.v_dtype,
      self.d_generations,
    )
    self.sdO_layout = sm100_utils_basic.make_smem_layout_b(
      self.VdO_tiled_mma,
      self.VdO_mma_tiler,
      self.do_dtype,
      self.load_mma_dO_stage,
    )
    self.sdSt_layout = sm100_utils_basic.make_smem_layout_a(
      self.dSQ_tiled_mma,
      self.dSQ_mma_tiler,
      self.ds_dtype,
      self.compute_mma_dS_stage,
    )
    # QT staged by dK slice: slice j's B is QT rows [j*N, (j+1)*N), one token.
    self.sQt_layout = sm100_utils_basic.make_smem_layout_b(
      self.dSQ_tiled_mma,
      self.dSQ_mma_tiler,
      self.q_dtype,
      self.load_mma_QT_stage * self.TARGET_DK_SLICES,
    )
    self.LSE_smem_layout = cute.make_layout(
      (self.cta_tiler[0], self.load_compute_LSE_stage)
    )
    self.sum_OdO_smem_layout = cute.make_layout(
      (self.cta_tiler[0], self.load_compute_sum_OdO_stage)
    )

    atom_thr_size = cute.size(KQ_tiled_mma.thr_id.shape)
    self.cluster_layout_vmnk = cute.tiled_divide(
      cute.make_layout(self.cluster_shape_mnk),
      (atom_thr_size, ),
    )

    tma_load_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)

    K_smem_layout = cute.select(self.sK_layout, mode=[0, 1, 2])
    tma_atom_K, tma_tensor_K = cute.nvgpu.make_tiled_tma_atom_A(
      tma_load_op,
      K,
      K_smem_layout,
      self.KQ_mma_tiler,
      KQ_tiled_mma,
      self.cluster_layout_vmnk.shape,
    )

    V_smem_layout = cute.select(self.sV_layout, mode=[0, 1, 2])
    tma_atom_V, tma_tensor_V = cute.nvgpu.make_tiled_tma_atom_A(
      tma_load_op,
      V,
      V_smem_layout,
      self.VdO_mma_tiler,
      VdO_tiled_mma,
      self.cluster_layout_vmnk.shape,
    )

    Q_smem_layout = cute.select(self.sQ_layout, mode=[0, 1, 2])
    tma_atom_Q, tma_tensor_Q = cute.nvgpu.make_tiled_tma_atom_B(
      tma_load_op,
      Q,
      Q_smem_layout,
      self.KQ_mma_tiler,
      KQ_tiled_mma,
      self.cluster_layout_vmnk.shape,
    )
    QT_smem_layout = cute.select(self.sQt_layout, mode=[0, 1, 2])
    tma_atom_QT, tma_tensor_QT = cute.nvgpu.make_tiled_tma_atom_B(
      tma_load_op,
      QT,
      QT_smem_layout,
      self.dSQ_mma_tiler,
      dSQ_tiled_mma,
      self.cluster_layout_vmnk.shape,
    )

    dO_smem_layout = cute.select(self.sdO_layout, mode=[0, 1, 2])
    tma_atom_dO, tma_tensor_dO = cute.nvgpu.make_tiled_tma_atom_B(
      tma_load_op,
      dO,
      dO_smem_layout,
      self.VdO_mma_tiler,
      VdO_tiled_mma,
      self.cluster_layout_vmnk.shape,
    )

    # tx = bytes x buffers per barrier: QT counts slices, K/V generations.
    self.tma_copy_Q_bytes = (
      cute.size_in_bytes(Q.element_type, Q_smem_layout) * atom_thr_size *
      self.q_gens_per_token
    )
    self.tma_copy_QT_bytes = (
      cute.size_in_bytes(Q.element_type, QT_smem_layout) * atom_thr_size *
      self.TARGET_DK_SLICES
    )
    self.tma_copy_K_bytes = (
      cute.size_in_bytes(K.element_type, K_smem_layout) * atom_thr_size *
      self.d_generations
    )
    self.tma_copy_V_bytes = (
      cute.size_in_bytes(V.element_type, V_smem_layout) * atom_thr_size *
      self.d_generations
    )
    self.tma_copy_dO_bytes = cute.size_in_bytes(
      dO.element_type, dO_smem_layout
    ) * atom_thr_size

    # dK's S2G atom; one (64,512) arena tile: both slices, one leader drain.
    tma_store_op = cpasync.CopyBulkTensorTileS2GOp()
    (
      _num_compute_wgs,
      _epi_slice_cols_dK,
      _epi_cols_dK,
      epi_tile_dK,
      total_epi_stages,
    ) = self.dk_epilogue_tiling(dK.element_type.width)
    dK_layout_enum = cutlass.utils.LayoutEnum.from_tensor(dK)
    sdK_epi_layout = sm100_utils_basic.make_smem_layout_epi(
      dK.element_type,
      dK_layout_enum,
      epi_tile_dK,
      total_epi_stages * self.TARGET_DK_SLICES,
    )
    # Arena and sK scale with different dtypes; only this assert stops fp32 dK.
    if const_expr(self.alias_epilogue_onto_sK):
      assert (
        cute.cosize(sdK_epi_layout.outer) * dK.element_type.width
        <= cute.cosize(self.sK_layout) * K.element_type.width
      ), "the dK epilogue arena does not fit inside its sK donor"
    tma_atom_dK, tma_tensor_dK = cpasync.make_tiled_tma_atom(
      tma_store_op,
      dK,
      cute.select(sdK_epi_layout, mode=[0, 1]),
      epi_tile_dK,
    )

    @cute.struct
    class SharedStorage:
      load_mma_Q_mbar_ptr: cute.struct.MemRange[cutlass.Int64,
                                                self.load_mma_Q_stage * 2]
      load_mma_K_mbar_ptr: cute.struct.MemRange[cutlass.Int64,
                                                self.load_mma_K_stage * 2]
      load_mma_V_mbar_ptr: cute.struct.MemRange[cutlass.Int64,
                                                self.load_mma_V_stage * 2]
      load_mma_QT_mbar_ptr: cute.struct.MemRange[cutlass.Int64,
                                                 self.load_mma_QT_stage * 2]
      load_mma_dO_mbar_ptr: cute.struct.MemRange[cutlass.Int64,
                                                 self.load_mma_dO_stage * 2]
      load_compute_lse_mbar_ptr: cute.struct.MemRange[
        cutlass.Int64, self.load_compute_LSE_stage * 2]
      load_compute_sum_OdO_mbar_ptr: cute.struct.MemRange[
        cutlass.Int64, self.load_compute_sum_OdO_stage * 2]
      mma_compute_S_mbar_ptr: cute.struct.MemRange[cutlass.Int64,
                                                   self.mma_compute_S_stage * 2]
      mma_compute_dP_mbar_ptr: cute.struct.MemRange[cutlass.Int64,
                                                    self.mma_compute_dP_stage *
                                                    2]
      compute_mma_dS_mbar_ptr: cute.struct.MemRange[cutlass.Int64,
                                                    self.compute_mma_dS_stage *
                                                    2]
      mma_compute_dK_mbar_ptr: cute.struct.MemRange[cutlass.Int64,
                                                    self.mma_compute_dK_stage *
                                                    2]
      tmem_holding_buf: cutlass.Int32
      tmem_dealloc_mbar: cutlass.Int64
      sK: cute.struct.Align[
        cute.struct.MemRange[K.element_type,
                             cute.cosize(self.sK_layout)],
        self.buffer_align_bytes,
      ]
      sV: cute.struct.Align[
        cute.struct.MemRange[V.element_type,
                             cute.cosize(self.sV_layout)],
        self.buffer_align_bytes,
      ]
      sQ: cute.struct.Align[
        cute.struct.MemRange[Q.element_type,
                             cute.cosize(self.sQ_layout)],
        self.buffer_align_bytes,
      ]
      sQT: cute.struct.Align[
        cute.struct.MemRange[Q.element_type,
                             cute.cosize(self.sQt_layout)],
        self.buffer_align_bytes,
      ]
      sdO: cute.struct.Align[
        cute.struct.MemRange[dO.element_type,
                             cute.cosize(self.sdO_layout)],
        self.buffer_align_bytes,
      ]
      # 1-elt placeholder when aliased; donor's sP+sdST host is 16384 of 32768.
      sdK_epi: cute.struct.Align[
        cute.struct.MemRange[
          dK.element_type,
          1 if self.alias_epilogue_onto_sK else cute.cosize(sdK_epi_layout),
        ],
        self.buffer_align_bytes,
      ]
      sdST: cute.struct.Align[
        cute.struct.MemRange[
          Q.element_type,
          1 if self.alias_dst_onto_sK else cute.cosize(self.sdSt_layout),
        ],
        self.buffer_align_bytes,
      ]

      sLSE: cute.struct.Align[
        cute.struct.MemRange[self.acc_dtype,
                             cute.cosize(self.LSE_smem_layout)],
        self.buffer_align_bytes,
      ]
      sSum_OdO: cute.struct.Align[
        cute.struct.MemRange[self.acc_dtype,
                             cute.cosize(self.sum_OdO_smem_layout)],
        self.buffer_align_bytes,
      ]

    self.shared_storage = SharedStorage
    # The one executable SMEM-budget check; header prose is not a witness.
    assert SharedStorage.size_in_bytes() <= SM100_SMEM_CAPACITY_BYTES, (
      f"SharedStorage {SharedStorage.size_in_bytes()} B > SM100 opt-in "
      f"ceiling {SM100_SMEM_CAPACITY_BYTES} B"
    )

    # H_K alone: dK nests h_r at stride 0; dK.shape launches h_r redundant CTAs.
    o_shape_for_grid = (
      problem_shape[1],
      problem_shape[2],
      ((1, problem_shape[3][0][1]), problem_shape[3][1]),
    )
    self.tile_sched_params = self.tile_scheduler_cls.to_underlying_arguments(
      o_shape_for_grid,
      # Only tiler[0] is read, the dispatch x tile: for dK that is K, so tile_n.
      (self.tile_n, *self.cta_tiler[1:]),
      False,
    )
    # Fold the reorder into the grid: index math pays a decode in EVERY warp.
    if const_expr(self.use_lpt_scheduler):
      bwd_grid = self.tile_scheduler_cls.get_grid_shape(
        self.tile_sched_params,
        cluster_shape_m=self.cluster_shape_mnk[0],
        head_group=self.lpt_head_group,
        num_groups=self.lpt_num_groups,
        fold_into_grid=True,
      )
    else:
      # The static get_grid_shape rejects the LPT-only keywords.
      bwd_grid = self.tile_scheduler_cls.get_grid_shape(self.tile_sched_params)
    bwd_grid = cute.round_up(bwd_grid, self.cluster_shape_mnk)

    self.kernel(
      KQ_tiled_mma,
      VdO_tiled_mma,
      dSQ_tiled_mma,
      KQ_tmem_tiled_mma,
      self.KT_gen_layout,
      tma_atom_K,
      tma_tensor_K,
      K,
      tma_atom_V,
      tma_tensor_V,
      tma_atom_Q,
      tma_tensor_Q,
      Q,
      tma_atom_QT,
      tma_tensor_QT,
      tma_atom_dO,
      tma_tensor_dO,
      dK,
      tma_atom_dK,
      tma_tensor_dK,
      scaled_LSE,
      scale_softmax,
      sum_OdO,
      problem_shape,
      cumulative_s_q,
      cumulative_s_k,
      self.cluster_layout_vmnk,
      self.sK_layout,
      self.sQ_layout,
      self.sV_layout,
      self.sdO_layout,
      self.sdSt_layout,
      self.sQt_layout,
      self.LSE_smem_layout,
      self.sum_OdO_smem_layout,
      sdK_epi_layout,
      self.tile_sched_params,
    ).launch(
      grid=bwd_grid,
      block=[self.threads_per_cta, 1, 1],
      cluster=self.cluster_shape_mnk,
      smem=self.shared_storage.size_in_bytes(),  # type: ignore [attr-defined]
      stream=stream,
      min_blocks_per_mp=1,
    )

  @cute.kernel
  def kernel(
    self,
    KQ_tiled_mma: cute.TiledMma,
    VdO_tiled_mma: cute.TiledMma,
    dSQ_tiled_mma: cute.TiledMma,
    KQ_tmem_tiled_mma: cute.TiledMma,
    KT_gen_layout: cute.ComposedLayout,
    tma_atom_K: cute.CopyAtom,
    mK: cute.Tensor,
    # *_ref: the raw (non-TMA) tensor, read for shape/dtype only.
    mK_ref: cute.Tensor,
    tma_atom_V: cute.CopyAtom,
    mV: cute.Tensor,
    tma_atom_Q: cute.CopyAtom,
    mQ: cute.Tensor,
    mQ_ref: cute.Tensor,
    tma_atom_QT: cute.CopyAtom,
    mQT: cute.Tensor,
    tma_atom_dO: cute.CopyAtom,
    mdO: cute.Tensor,
    mdK: cute.Tensor,
    tma_atom_dK: cute.CopyAtom,
    mdK_tma: cute.Tensor,
    mLSE: cute.Tensor,
    scale_softmax: cutlass.Float32,
    mSumOdO: cute.Tensor,
    problem_shape: tuple[Int32, Int32, Int32, tuple[tuple[Int32, Int32],
                                                    Int32]],
    cumulative_s_q: cute.Tensor | None,
    cumulative_s_k: cute.Tensor | None,
    cluster_layout_vmnk: cute.Layout,
    sK_layout: cute.ComposedLayout,
    sQ_layout: cute.ComposedLayout,
    sV_layout: cute.ComposedLayout,
    sdO_layout: cute.ComposedLayout,
    sdSt_layout: cute.ComposedLayout,
    sQt_layout: cute.ComposedLayout,
    LSE_smem_layout: cute.Layout,
    sum_OdO_smem_layout: cute.Layout,
    sdK_epi_layout: cute.ComposedLayout,
    tile_sched_params: FmhaStaticTileSchedulerParams,
  ):
    """Kernel body: role dispatch by warp index."""
    # One fetch before the split; dQ's per-role rule does not transfer here.
    bidx, _, (bidy, bidz) = (
      self.make_tile_scheduler(tile_sched_params
                               ).initial_work_tile_info().tile_idx
    )
    blk_coord_k, blk_coord_h_k, blk_coord_b = bidx, bidy, bidz
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    # Dispatch bounds of the multi-warp role.
    compute_lo = self.compute_warp_ids[0]
    compute_hi = self.compute_warp_ids[-1]
    varlen = cumulative_s_q is not None or cumulative_s_k is not None

    if warp_idx == self.load_warp_id:
      cpasync.prefetch_descriptor(tma_atom_K)
      cpasync.prefetch_descriptor(tma_atom_Q)
    if warp_idx == self.load_do_warp_id:
      cpasync.prefetch_descriptor(tma_atom_V)
      cpasync.prefetch_descriptor(tma_atom_dO)
    if warp_idx == self.load_qt_warp_id:
      cpasync.prefetch_descriptor(tma_atom_QT)

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(self.shared_storage)

    # One group per participant set; TMA/UMMA arrive elected, one thread each.
    tma_producer_group = CooperativeGroup(Agent.Thread, 1)
    mma_group = CooperativeGroup(Agent.Thread, 1)
    # cp.async has no elected form: the whole warp arrives, warpgroups wait.
    cpasync_producer_group = CooperativeGroup(
      Agent.Thread, self.threads_per_warp
    )
    cpasync_consumer_group = CooperativeGroup(
      Agent.Thread, self.threads_per_warp * self.num_compute_warps
    )
    # 2-CTA UMMA: both CTAs' compute warps read it, so the set spans the pair.
    compute_group_2cta = CooperativeGroup(
      Agent.Thread,
      self.num_compute_warps * self.threads_per_warp *
      cluster_layout_vmnk.shape[0][0],
    )

    load_mma_Q_producer, load_mma_Q_consumer = pipeline.PipelineTmaUmma.create(
      num_stages=self.load_mma_Q_stage,
      producer_group=tma_producer_group,
      consumer_group=mma_group,
      tx_count=self.tma_copy_Q_bytes,
      barrier_storage=storage.load_mma_Q_mbar_ptr.data_ptr(),
      cta_layout_vmnk=cluster_layout_vmnk,
      defer_sync=True,
    ).make_participants()
    load_mma_K_producer, load_mma_K_consumer = pipeline.PipelineTmaUmma.create(
      num_stages=self.load_mma_K_stage,
      producer_group=tma_producer_group,
      consumer_group=mma_group,
      tx_count=self.tma_copy_K_bytes,
      barrier_storage=storage.load_mma_K_mbar_ptr.data_ptr(),
      cta_layout_vmnk=cluster_layout_vmnk,
      defer_sync=True,
    ).make_participants()
    load_mma_V_producer, load_mma_V_consumer = pipeline.PipelineTmaUmma.create(
      num_stages=self.load_mma_V_stage,
      producer_group=tma_producer_group,
      consumer_group=mma_group,
      tx_count=self.tma_copy_V_bytes,
      barrier_storage=storage.load_mma_V_mbar_ptr.data_ptr(),
      cta_layout_vmnk=cluster_layout_vmnk,
      defer_sync=True,
    ).make_participants()
    load_mma_QT_producer, load_mma_QT_consumer = pipeline.PipelineTmaUmma.create(
      num_stages=self.load_mma_QT_stage,
      producer_group=tma_producer_group,
      consumer_group=mma_group,
      # QT's own count: Q's undercounts (QT is DK_SLICES buffers) and HANGS.
      tx_count=self.tma_copy_QT_bytes,
      barrier_storage=storage.load_mma_QT_mbar_ptr.data_ptr(),
      cta_layout_vmnk=cluster_layout_vmnk,
      defer_sync=True,
    ).make_participants()
    load_mma_dO_producer, load_mma_dO_consumer = pipeline.PipelineTmaUmma.create(
      num_stages=self.load_mma_dO_stage,
      producer_group=tma_producer_group,
      consumer_group=mma_group,
      tx_count=self.tma_copy_dO_bytes,
      barrier_storage=storage.load_mma_dO_mbar_ptr.data_ptr(),
      cta_layout_vmnk=cluster_layout_vmnk,
      defer_sync=True,
    ).make_participants()
    load_compute_LSE_producer, load_compute_LSE_consumer = pipeline.PipelineCpAsync.create(
      num_stages=self.load_compute_LSE_stage,
      producer_group=cpasync_producer_group,
      consumer_group=cpasync_consumer_group,
      barrier_storage=storage.load_compute_lse_mbar_ptr.data_ptr(),
    ).make_participants()
    load_compute_sum_OdO_producer, load_compute_sum_OdO_consumer = (
      pipeline.PipelineCpAsync.create(
        num_stages=self.load_compute_sum_OdO_stage,
        producer_group=cpasync_producer_group,
        consumer_group=cpasync_consumer_group,
        barrier_storage=storage.load_compute_sum_OdO_mbar_ptr.data_ptr(),
      ).make_participants()
    )
    mma_compute_S_producer, mma_compute_S_consumer = pipeline.PipelineUmmaAsync.create(
      num_stages=self.mma_compute_S_stage,
      producer_group=mma_group,
      consumer_group=compute_group_2cta,
      barrier_storage=storage.mma_compute_S_mbar_ptr.data_ptr(),
      cta_layout_vmnk=cluster_layout_vmnk,
      defer_sync=True,
    ).make_participants()
    mma_compute_dP_producer, mma_compute_dP_consumer = pipeline.PipelineUmmaAsync.create(
      num_stages=self.mma_compute_dP_stage,
      producer_group=mma_group,
      consumer_group=compute_group_2cta,
      barrier_storage=storage.mma_compute_dP_mbar_ptr.data_ptr(),
      cta_layout_vmnk=cluster_layout_vmnk,
      defer_sync=True,
    ).make_participants()
    compute_mma_dS_producer, compute_mma_dS_consumer = pipeline.PipelineAsyncUmma.create(
      num_stages=self.compute_mma_dS_stage,
      producer_group=compute_group_2cta,
      consumer_group=mma_group,
      barrier_storage=storage.compute_mma_dS_mbar_ptr.data_ptr(),
      cta_layout_vmnk=cluster_layout_vmnk,
      defer_sync=True,
    ).make_participants()
    mma_compute_dK_producer, mma_compute_dK_consumer = pipeline.PipelineUmmaAsync.create(
      num_stages=self.mma_compute_dK_stage,
      producer_group=mma_group,
      consumer_group=compute_group_2cta,
      barrier_storage=storage.mma_compute_dK_mbar_ptr.data_ptr(),
      cta_layout_vmnk=cluster_layout_vmnk,
      defer_sync=True,
    ).make_participants()

    cute.arch.barrier(
      barrier_id=self.cta_sync_bar_id, number_of_threads=self.threads_per_cta
    )

    sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
    sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
    sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
    sdO = storage.sdO.get_tensor(sdO_layout.outer, swizzle=sdO_layout.inner)
    sLSE = storage.sLSE.get_tensor(LSE_smem_layout)
    sSum_OdO = storage.sSum_OdO.get_tensor(sum_OdO_smem_layout)

    sQT = storage.sQT.get_tensor(sQt_layout.outer, swizzle=sQt_layout.inner)
    if const_expr(self.alias_dst_onto_sK):
      # sK gen 0 died at the prologue utccp; dtype mdK.element_type = Q's bits.
      sdST = cute.make_tensor(
        cute.recast_ptr(sK.iterator, sdSt_layout.inner, mdK.element_type),
        sdSt_layout.outer,
      )
    else:
      sdST = storage.sdST.get_tensor(
        sdSt_layout.outer, swizzle=sdSt_layout.inner
      )
    # dK epilogue arena: its whole lifetime (STS then TMA) is inside epilogue().
    if const_expr(self.alias_epilogue_onto_sK):
      # sK dead: last MMA read -> release -> dK token; exact fit, asserted.
      sdK_epi = cute.make_tensor(
        cute.recast_ptr(sK.iterator, sdK_epi_layout.inner, mdK.element_type),
        sdK_epi_layout.outer,
      )
    else:
      sdK_epi = storage.sdK_epi.get_tensor(
        sdK_epi_layout.outer, swizzle=sdK_epi_layout.inner
      )

    # tSTrK shape : (MMA, MMA_M, MMA_K, STAGE)
    tSTrK = KQ_tiled_mma.make_fragment_A(sK)
    # tSTrQ shape : (MMA, MMA_N, MMA_K, STAGE)
    tSTrQ = KQ_tiled_mma.make_fragment_B(sQ)

    # tdPTrV shape : (MMA, MMA_M, MMA_K, STAGE)
    tdPTrV = VdO_tiled_mma.make_fragment_A(sV)
    # tdPTrdO shape : (MMA, MMA_N, MMA_K, STAGE)
    tdPTrdO = VdO_tiled_mma.make_fragment_B(sdO)

    # tdKrdST shape: (MMA, MMA_M, MMA_K, STAGE)
    tdKrdST = dSQ_tiled_mma.make_fragment_A(sdST)
    # tdKrQT shape : (MMA, MMA_N, MMA_K, STAGE)
    tdKrQT = dSQ_tiled_mma.make_fragment_B(sQT)

    tmem_alloc_barrier = pipeline.NamedBarrier(
      barrier_id=self.tmem_alloc_sync_bar_id,
      num_threads=self.threads_per_cta,
    )

    tmem = cutlass.utils.TmemAllocator(
      storage.tmem_holding_buf.ptr,
      barrier_for_retrieve=tmem_alloc_barrier,
      allocator_warp_id=self.load_warp_id,
      is_two_cta=True,
      two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
    )

    tmem.allocate(self.tmem_alloc_cols)

    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

    # TMEM-A: tA_addr picks the gen (A_idx=0 always); cute.gemm/swap_AB hit 0.
    k_dtype = mK_ref.element_type
    tKT_gens = tuple(
      cute.make_tensor(
        cute.recast_ptr(
          tmem_ptr + self.tmem_K_base + gen_index * self.tmem_K_cols_per_gen,
          dtype=k_dtype,
        ),
        KT_gen_layout.outer,
      ) for gen_index in range(self.ktmem_gens)
    )
    # _tmem names the TMEM-A edge, not storage: *_tmem B fragments stay SMEM.
    tSTrK_tmem_gens = tuple(
      KQ_tmem_tiled_mma.make_fragment_A(tKT_gens[gen_index])
      for gen_index in range(self.ktmem_gens)
    )
    tKT_addr_gens = tuple(
      tKT_gens[gen_index].iterator.toint()
      for gen_index in range(self.ktmem_gens)
    )
    tSTrQ_tmem = KQ_tmem_tiled_mma.make_fragment_B(sQ)
    # utccp: each CTA reads its LOCAL sK generation; row r on lanes r%64, +64.
    s2t_atom = cute.make_copy_atom(
      tcgen05.copy.Cp2x64x128b0213Op(tcgen05.CtaGroup.TWO), k_dtype
    )
    tiled_s2t = tcgen05.make_s2t_copy(
      s2t_atom, cute.filter_zeros(tSTrK_tmem_gens[0])
    )
    thr_s2t = tiled_s2t.get_slice(0)
    # 128-bit view of one sK/sV gen stage; gen g at offset g * gen_elems.
    sw128_row_elems = 128 // (k_dtype.width // 8)
    panel_elems = self.tile_n * sw128_row_elems
    k_groups = self.KQ_mma_tiler[2] // sw128_row_elems
    gen_elems = self.tile_n * self.KQ_mma_tiler[2]
    # Modes: ((row, elem in 16 B), 1:0, (chunk pair, quad, 64-col panel), 1:0).
    s2t_src_layout = cute.make_layout(
      ((64, 8), 1, (2, 4, k_groups), 1),
      stride=((sw128_row_elems, 1), 0, (8, 16, panel_elems), 0),
    )
    sK_gen_128b = tuple(
      cute.make_tensor(sK.iterator + gen_index * gen_elems, s2t_src_layout)
      for gen_index in range(self.ktmem_gens)
    )
    tKsK_s2t_gens = tuple(
      tcgen05.get_s2t_smem_desc_tensor(
        tiled_s2t,
        thr_s2t.partition_S(cute.filter_zeros(sK_gen_128b[gen_index])),
      ) for gen_index in range(self.ktmem_gens)
    )
    tKtK_s2t_gens = tuple(
      cute.make_tensor(
        tKT_gens[gen_index].iterator,
        thr_s2t.partition_D(cute.filter_zeros(tSTrK_tmem_gens[gen_index])
                            ).layout,
      ) for gen_index in range(self.ktmem_gens)
    )

    # V faces reuse KQ_tmem_tiled_mma (same tiler/dtype/K-major); base differs.
    assert sV.element_type == k_dtype, "V/K dtype mismatch breaks the shared TMEM-A tiled mma"
    tVT_gens = tuple(
      cute.make_tensor(
        cute.recast_ptr(
          tmem_ptr + self.tmem_V_base + gen_index * self.tmem_K_cols_per_gen,
          dtype=k_dtype,
        ),
        KT_gen_layout.outer,
      ) for gen_index in range(self.vtmem_gens)
    )
    tdPTrV_tmem_gens = tuple(
      KQ_tmem_tiled_mma.make_fragment_A(tVT_gens[gen_index])
      for gen_index in range(self.vtmem_gens)
    )
    tVT_addr_gens = tuple(
      tVT_gens[gen_index].iterator.toint()
      for gen_index in range(self.vtmem_gens)
    )
    tdPTrdO_tmem = KQ_tmem_tiled_mma.make_fragment_B(sdO)
    sV_gen_128b = tuple(
      cute.make_tensor(sV.iterator + gen_index * gen_elems, s2t_src_layout)
      for gen_index in range(self.vtmem_gens)
    )
    tVsV_s2t_gens = tuple(
      tcgen05.get_s2t_smem_desc_tensor(
        tiled_s2t,
        thr_s2t.partition_S(cute.filter_zeros(sV_gen_128b[gen_index])),
      ) for gen_index in range(self.vtmem_gens)
    )
    tVtV_s2t_gens = tuple(
      cute.make_tensor(
        tVT_gens[gen_index].iterator,
        thr_s2t.partition_D(cute.filter_zeros(tdPTrV_tmem_gens[gen_index])
                            ).layout,
      ) for gen_index in range(self.vtmem_gens)
    )

    # Cluster arrive after barrier init; is_relaxed=False keeps consistency.
    pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=False)

    tSTtST_shape = KQ_tiled_mma.partition_shape_C(
      cute.select(self.KQ_mma_tiler, mode=[0, 1])
    )
    tSTtST = KQ_tiled_mma.make_fragment_C(tSTtST_shape)
    # tSTtST shape : (MMA, MMA_M, MMA_N)
    tSTtST = cute.make_tensor(tmem_ptr + self.tmem_S_offset, tSTtST.layout)

    tdPTtdPT_shape = VdO_tiled_mma.partition_shape_C(
      cute.select(self.VdO_mma_tiler, mode=[0, 1])
    )
    tdPTtdPT = VdO_tiled_mma.make_fragment_C(tdPTtdPT_shape)
    # tdPTtdPT shape : (MMA, MMA_M, MMA_N)
    tdPTtdPT = cute.make_tensor(tmem_ptr + self.tmem_dP_offset, tdPTtdPT.layout)

    # One accumulator per dK slice; a slice is an output cut, not a reduction.
    tdKtdK_shape = dSQ_tiled_mma.partition_shape_C(
      cute.select(self.dSQ_mma_tiler, mode=[0, 1])
    )
    tdKtdK_frag = dSQ_tiled_mma.make_fragment_C(tdKtdK_shape)
    # each entry's shape : (MMA, MMA_M, MMA_N)
    tdKtdK = tuple(
      cute.make_tensor(
        tmem_ptr + self.tmem_dK_offset +
        slice_index * self.TARGET_TMEM_DK_SLICE_STRIDE,
        tdKtdK_frag.layout,
      ) for slice_index in range(self.TARGET_DK_SLICES)
    )
    blk_coord = (Int32(0), bidx, Int32(0), ((Int32(0), bidy), bidz))
    # Trace-time factory: each role builds its own; threading spills 8-48 B/thd.
    SeqlenInfoCls = partial(
      SeqlenInfoQK.create,
      seqlen_q_static=mQ_ref.shape[0],
      seqlen_k_static=mK_ref.shape[0],
      mCuSeqlensQ=cumulative_s_q,
      mCuSeqlensK=cumulative_s_k,
      tile_m=self.tile_m,
      tile_n=self.tile_n * self.cluster_shape_mnk[0],
    )
    seqlen = SeqlenInfoCls(bidz)
    seqlen_q_cur_batch = seqlen.seqlen_q
    seqlen_k_cur_batch = seqlen.seqlen_k

    iter_start, iter_end = self.get_Q_block_min_max(seqlen, blk_coord[1])

    pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

    iter_count = (iter_end - iter_start) * problem_shape[3][0][0]
    problem_shape_cur_batch = (
      seqlen_q_cur_batch,
      seqlen_k_cur_batch,
      problem_shape[2],
      problem_shape[3],
    )
    if iter_count <= 0:
      if bidx * self.tile_n < seqlen_k_cur_batch:
        self.epilogue_clear(
          blk_coord,
          SeqlenInfoCls,
          problem_shape_cur_batch,
          mdK,
        )
    # ///  LOAD  ///
    elif warp_idx == self.load_warp_id:
      # setmaxnreg is wg2-uniform (warps 8-11): one instruction, one value.
      if const_expr(self.despill_wg2):
        cute.arch.setmaxregister_increase(self.num_regs_mma)
      else:
        cute.arch.setmaxregister_decrease(self.num_regs_load)

      self.load_q(
        mK,
        mQ,
        mLSE,
        sK,
        sQ,
        sLSE,
        KQ_tiled_mma,
        tma_atom_K,
        tma_atom_Q,
        SeqlenInfoCls,
        problem_shape_cur_batch,
        iter_count,
        iter_start,
        iter_end,
        load_mma_Q_producer,
        load_mma_K_producer,
        load_compute_LSE_producer,
        blk_coord_k,
        blk_coord_h_k,
        blk_coord_b,
      )

    elif warp_idx == self.load_do_warp_id:
      if const_expr(self.despill_wg2):
        cute.arch.setmaxregister_increase(self.num_regs_mma)
      else:
        cute.arch.setmaxregister_decrease(self.num_regs_load)

      self.load_do(
        mV,
        mdO,
        mSumOdO,
        sV,
        sdO,
        sSum_OdO,
        KQ_tiled_mma,
        VdO_tiled_mma,
        tma_atom_V,
        tma_atom_dO,
        SeqlenInfoCls,
        problem_shape_cur_batch,
        iter_count,
        iter_start,
        iter_end,
        load_mma_V_producer,
        load_mma_dO_producer,
        load_compute_sum_OdO_producer,
        blk_coord_k,
        blk_coord_h_k,
        blk_coord_b,
      )

    elif warp_idx == self.load_qt_warp_id:
      if const_expr(self.despill_wg2):
        cute.arch.setmaxregister_increase(self.num_regs_mma)
      else:
        cute.arch.setmaxregister_decrease(self.num_regs_load)

      self.load_qt(
        mQT,
        sQT,
        KQ_tiled_mma,
        dSQ_tiled_mma,
        tma_atom_QT,
        SeqlenInfoCls,
        iter_count,
        iter_start,
        iter_end,
        load_mma_QT_producer,
        blk_coord_k,
        blk_coord_h_k,
        blk_coord_b,
      )

    # ///  MMA  ///
    elif warp_idx == self.mma_warp_id:
      cute.arch.setmaxregister_increase(self.num_regs_mma)

      self.mma(
        KQ_tiled_mma,
        VdO_tiled_mma,
        dSQ_tiled_mma,
        KQ_tmem_tiled_mma,
        tSTrK_tmem_gens,
        tSTrQ_tmem,
        tKT_addr_gens,
        tiled_s2t,
        tKsK_s2t_gens,
        tKtK_s2t_gens,
        tdPTrV_tmem_gens,
        tdPTrdO_tmem,
        tVT_addr_gens,
        tVsV_s2t_gens,
        tVtV_s2t_gens,
        sQ,
        sdO,
        tSTtST,
        tSTrQ,
        tSTrK,
        tdPTtdPT,
        tdPTrV,
        tdPTrdO,
        tdKrdST,
        tdKtdK,
        tdKrQT,
        iter_count,
        load_mma_Q_consumer,
        load_mma_K_consumer,
        load_mma_V_consumer,
        mma_compute_S_producer,
        load_mma_dO_consumer,
        mma_compute_dP_producer,
        compute_mma_dS_consumer,
        load_mma_QT_consumer,
        mma_compute_dK_producer,
      )

    # ///  Compute  ///
    elif warp_idx >= compute_lo and warp_idx <= compute_hi:
      # Static alloc is 168: this must be a dec; an inc downward is UB.
      if const_expr(self.despill_wg2):
        cute.arch.setmaxregister_decrease(self.num_regs_compute)
      else:
        cute.arch.setmaxregister_increase(self.num_regs_compute)

      self.compute_loop(
        tSTtST,
        tdPTtdPT,
        sdK_epi,
        sLSE,
        sdST,
        sSum_OdO,
        mdK,
        tdKtdK,
        blk_coord,
        SeqlenInfoCls,
        problem_shape_cur_batch,
        iter_count,
        iter_start,
        iter_end,
        scale_softmax,
        mma_compute_S_consumer,
        load_compute_LSE_consumer,
        load_compute_sum_OdO_consumer,
        mma_compute_dP_consumer,
        compute_mma_dS_producer,
        mma_compute_dK_consumer,
        varlen,
        seqlen_k_cur_batch,
        tma_atom_dK,
        mdK_tma,
      )

      cute.arch.barrier(
        barrier_id=self.epilogue_sync_bar_id,
        number_of_threads=self.num_compute_warps * self.threads_per_warp,
      )

    cute.arch.cluster_arrive()
    cute.arch.cluster_wait()
    tmem.relinquish_alloc_permit()
    tmem.free(tmem_ptr)
    # Drain last: the CTA must not exit with async arena reads still in flight.
    cute.arch.cp_async_bulk_wait_group(0, read=True)

  @cute.jit
  def get_Q_block_min_max(
    self,
    seqlen: SeqlenInfoQK,
    n_block: Int32,
  ):
    """Q-block trip range of this pair tile (BlockInfo)."""
    # 2-CTA rounding lives in the pair tile: unequal trip counts HANG.
    block_info = BlockInfo(
      self.tile_m,
      self.tile_n * self.cluster_shape_mnk[0],
      self.is_causal,
      self.has_sliding_window,
      self.window_size_left,
      self.window_size_right,
      qhead_per_kvhead_packgqa=1,
    )
    # Do not re-add m_block_min -= m_block_min % 2; the exact gate subsumed it.
    return block_info.get_m_block_min_max(
      seqlen, n_block // self.cluster_shape_mnk[0]
    )

  @cute.jit
  def load_q(
    self,
    mK: cute.Tensor,
    mQ: cute.Tensor,
    mLSE: cute.Tensor,
    sK: cute.Tensor,
    sQ: cute.Tensor,
    sLSE: cute.Tensor,
    KQ_tiled_mma: cute.TiledMma,
    tma_atom_K: cute.CopyAtom,
    tma_atom_Q: cute.CopyAtom,
    SeqlenInfoCls: Callable,
    problem_shape: tuple[Int32, Int32, Int32, tuple[tuple[Int32, Int32],
                                                    Int32]],
    iter_count: Int32,
    iter_start: Int32,
    iter_end: Int32,
    load_mma_Q_producer,
    load_mma_K_producer,
    load_compute_LSE_producer,
    blk_coord_k: Int32,
    blk_coord_h_k: Int32,
    blk_coord_b: Int32,
  ):
    """Issuer warp 9: the resident K set, the Q generations, LSE."""
    tidx, _, _ = cute.arch.thread_idx()
    blk_coord_h_r = Int32(0)
    blk_coord_h = (blk_coord_h_r, blk_coord_h_k)
    iter_index = iter_start
    mma_tile_coord_v = blk_coord_k % cute.size(KQ_tiled_mma.thr_id.shape)
    mma_tile_coord_m = blk_coord_k // cute.size(KQ_tiled_mma.thr_id.shape)

    cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)
    cta_layout_vmnk = cute.tiled_divide(cta_layout_mnk, (KQ_tiled_mma.thr_id, ))
    cta_in_cluster_coord_vmnk = cta_layout_vmnk.get_flat_coord(
      cute.arch.block_idx_in_cluster()
    )

    seqlen = SeqlenInfoCls(blk_coord_b)
    hb_origin = ((Int32(0), Int32(0)), Int32(0))
    K = cute.domain_offset((seqlen.offset_k, Int32(0), hb_origin), mK)
    Q = cute.domain_offset((seqlen.offset_q, Int32(0), hb_origin), mQ)
    # LSE is stored per padded Q tile, so it takes the padded offset.
    LSE = cute.domain_offset((seqlen.padded_offset_q, hb_origin), mLSE)

    gK = cute.local_tile(
      K, cute.select(self.KQ_mma_tiler, mode=[0, 2]), (None, None, None)
    )
    gQ = cute.local_tile(
      Q, cute.select(self.KQ_mma_tiler, mode=[1, 2]), (None, None, None)
    )
    KQ_thr_mma = KQ_tiled_mma.get_slice(mma_tile_coord_v)
    tSTgK = KQ_thr_mma.partition_A(gK)
    tSTgQ = KQ_thr_mma.partition_B(gQ)

    # Raw tma_partition: the helper groups at rank-1, needs a participant bar.
    tKsK, tKgK_mkl = cute.nvgpu.cpasync.tma_partition(
      tma_atom_K,
      cta_in_cluster_coord_vmnk[2],
      cute.make_layout(cute.size(cta_layout_vmnk, mode=[2])),
      cute.group_modes(sK, 0, 3),
      cute.group_modes(tSTgK, 0, 3),
    )
    tQsQ, tQgQ_mkl = cute.nvgpu.cpasync.tma_partition(
      tma_atom_Q,
      cta_in_cluster_coord_vmnk[1],
      cute.make_layout(cute.size(cta_layout_vmnk, mode=[1])),
      cute.group_modes(sQ, 0, 3),
      cute.group_modes(tSTgQ, 0, 3),
    )

    # K is resident: one arrival token covers all generations.
    k_handle = load_mma_K_producer.acquire_and_advance()
    for gen in cutlass.range_constexpr(self.d_generations):
      cute.copy(
        tma_atom_K,
        tKgK_mkl[(None, mma_tile_coord_m, gen, (blk_coord_h, blk_coord_b))],
        tKsK[None, gen],
        tma_bar_ptr=k_handle.barrier,
      )

    thread_idx = tidx % self.threads_per_warp
    async_copy_num_elts = sLSE.shape[0] // self.threads_per_warp
    atom_async_copy = cute.make_copy_atom(
      cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
      self.acc_dtype,
      num_bits_per_copy=self.acc_dtype.width,
    )

    for tok in cutlass.range_constexpr(
      self.d_generations // self.q_gens_per_token
    ):
      q_handle = load_mma_Q_producer.acquire_and_advance()
      for sub in cutlass.range_constexpr(self.q_gens_per_token):
        if const_expr(self.q_gens_per_token == 1):
          q_slot = q_handle.index
        else:
          q_slot = q_handle.index * self.q_gens_per_token + sub
        cute.copy(
          tma_atom_Q,
          tQgQ_mkl[(
            None,
            iter_index,
            tok * self.q_gens_per_token + sub,
            (blk_coord_h, blk_coord_b),
          )],
          tQsQ[None, q_slot],
          tma_bar_ptr=q_handle.barrier,
        )

    lse_handle = load_compute_LSE_producer.acquire_and_advance()
    sLSE_for_copy = cute.flat_divide(sLSE, (1, ))
    LSE_for_copy = cute.flat_divide(LSE, (1, ))
    for i in cutlass.range_constexpr(async_copy_num_elts):
      LSE_idx = self.tile_m * iter_index + thread_idx + i * self.threads_per_warp
      sLSE_idx = thread_idx + i * self.threads_per_warp
      if cute.elem_less(LSE_idx, problem_shape[0]):
        cute.copy(
          atom_async_copy,
          LSE_for_copy[None, LSE_idx, (blk_coord_h, blk_coord_b)],
          sLSE_for_copy[None, sLSE_idx, lse_handle.index],
        )
      else:
        sLSE_for_copy[None, sLSE_idx, lse_handle.index].fill(0.0)
    lse_handle.commit()

    iter_count -= 1
    iter_index += 1

    while iter_count > 0:
      if iter_index == iter_end:
        iter_index = iter_start
        blk_coord_h_r += 1
        blk_coord_h = (blk_coord_h_r, blk_coord_h_k)

      for tok in cutlass.range_constexpr(
        self.d_generations // self.q_gens_per_token
      ):
        q_handle = load_mma_Q_producer.acquire_and_advance()
        for sub in cutlass.range_constexpr(self.q_gens_per_token):
          if const_expr(self.q_gens_per_token == 1):
            q_slot = q_handle.index
          else:
            q_slot = q_handle.index * self.q_gens_per_token + sub
          cute.copy(
            tma_atom_Q,
            tQgQ_mkl[(
              None,
              iter_index,
              tok * self.q_gens_per_token + sub,
              (blk_coord_h, blk_coord_b),
            )],
            tQsQ[None, q_slot],
            tma_bar_ptr=q_handle.barrier,
          )

      lse_handle = load_compute_LSE_producer.acquire_and_advance()
      sLSE_for_copy = cute.flat_divide(sLSE, (1, ))
      LSE_for_copy = cute.flat_divide(LSE, (1, ))
      for i in cutlass.range_constexpr(async_copy_num_elts):
        LSE_idx = self.tile_m * iter_index + thread_idx + i * self.threads_per_warp
        sLSE_idx = thread_idx + i * self.threads_per_warp
        if cute.elem_less(LSE_idx, problem_shape[0]):
          cute.copy(
            atom_async_copy,
            LSE_for_copy[None, LSE_idx, (blk_coord_h, blk_coord_b)],
            sLSE_for_copy[None, sLSE_idx, lse_handle.index],
          )
        else:
          sLSE_for_copy[None, sLSE_idx, lse_handle.index].fill(0.0)
      lse_handle.commit()

      iter_count -= 1
      iter_index += 1

    load_mma_K_producer.tail()
    load_mma_Q_producer.tail()
    load_compute_LSE_producer.tail()

  @cute.jit
  def load_do(
    self,
    mV: cute.Tensor,
    mdO: cute.Tensor,
    mSumOdO: cute.Tensor,
    sV: cute.Tensor,
    sdO: cute.Tensor,
    sSum_OdO: cute.Tensor,
    KQ_tiled_mma: cute.TiledMma,
    VdO_tiled_mma: cute.TiledMma,
    tma_atom_V: cute.CopyAtom,
    tma_atom_dO: cute.CopyAtom,
    SeqlenInfoCls: Callable,
    problem_shape: tuple[Int32, Int32, Int32, tuple[tuple[Int32, Int32],
                                                    Int32]],
    iter_count: Int32,
    iter_start: Int32,
    iter_end: Int32,
    load_mma_V_producer,
    load_mma_dO_producer,
    load_compute_sum_OdO_producer,
    blk_coord_k: Int32,
    blk_coord_h_k: Int32,
    blk_coord_b: Int32,
  ):
    """Issuer warp 10: the resident V set, the dO generations, sum_OdO."""
    tidx, _, _ = cute.arch.thread_idx()
    blk_coord_h_r = Int32(0)
    blk_coord_h = (blk_coord_h_r, blk_coord_h_k)
    iter_index = iter_start
    mma_tile_coord_v = blk_coord_k % cute.size(KQ_tiled_mma.thr_id.shape)
    mma_tile_coord_m = blk_coord_k // cute.size(KQ_tiled_mma.thr_id.shape)

    cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)
    cta_layout_vmnk = cute.tiled_divide(cta_layout_mnk, (KQ_tiled_mma.thr_id, ))
    cta_in_cluster_coord_vmnk = cta_layout_vmnk.get_flat_coord(
      cute.arch.block_idx_in_cluster()
    )

    seqlen = SeqlenInfoCls(blk_coord_b)
    hb_origin = ((Int32(0), Int32(0)), Int32(0))
    V = cute.domain_offset((seqlen.offset_k, Int32(0), hb_origin), mV)
    dO = cute.domain_offset((seqlen.offset_q, Int32(0), hb_origin), mdO)
    # sum_OdO is stored per padded Q tile, exactly as LSE is.
    sum_OdO = cute.domain_offset((seqlen.padded_offset_q, hb_origin), mSumOdO)

    gV = cute.local_tile(
      V, cute.select(self.VdO_mma_tiler, mode=[0, 2]), (None, None, None)
    )
    gdO = cute.local_tile(
      dO, cute.select(self.VdO_mma_tiler, mode=[1, 2]), (None, None, None)
    )
    VdO_thr_mma = VdO_tiled_mma.get_slice(mma_tile_coord_v)
    tdPTgV = VdO_thr_mma.partition_A(gV)
    tdPTgdO = VdO_thr_mma.partition_B(gdO)

    tVsV, tVgV_mkl = cute.nvgpu.cpasync.tma_partition(
      tma_atom_V,
      cta_in_cluster_coord_vmnk[2],
      cute.make_layout(cute.size(cta_layout_vmnk, mode=[2])),
      cute.group_modes(sV, 0, 3),
      cute.group_modes(tdPTgV, 0, 3),
    )
    tdOsdO, tdOgdO_mkl = cute.nvgpu.cpasync.tma_partition(
      tma_atom_dO,
      cta_in_cluster_coord_vmnk[1],
      cute.make_layout(cute.size(cta_layout_vmnk, mode=[1])),
      cute.group_modes(sdO, 0, 3),
      cute.group_modes(tdPTgdO, 0, 3),
    )

    # V is resident on the same terms as K.
    v_handle = load_mma_V_producer.acquire_and_advance()
    for gen in cutlass.range_constexpr(self.d_generations):
      cute.copy(
        tma_atom_V,
        tVgV_mkl[(None, mma_tile_coord_m, gen, (blk_coord_h, blk_coord_b))],
        tVsV[(None, gen)],
        tma_bar_ptr=v_handle.barrier,
      )

    thread_idx = tidx % self.threads_per_warp
    async_copy_num_elts = sSum_OdO.shape[0] // self.threads_per_warp
    atom_async_copy = cute.make_copy_atom(
      cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
      self.acc_dtype,
      num_bits_per_copy=self.acc_dtype.width,
    )

    for gen in cutlass.range_constexpr(self.d_generations):
      do_handle = load_mma_dO_producer.acquire_and_advance()
      cute.copy(
        tma_atom_dO,
        tdOgdO_mkl[(None, iter_index, gen, (blk_coord_h, blk_coord_b))],
        tdOsdO[(None, do_handle.index)],
        tma_bar_ptr=do_handle.barrier,
      )

    sum_odo_handle = load_compute_sum_OdO_producer.acquire_and_advance()
    sSum_OdO_for_copy = cute.flat_divide(sSum_OdO, (1, ))
    sum_OdO_for_copy = cute.flat_divide(sum_OdO, (1, ))
    for i in cutlass.range_constexpr(async_copy_num_elts):
      sum_OdO_idx = self.tile_m * iter_index + thread_idx + i * self.threads_per_warp
      sSum_OdO_idx = thread_idx + i * self.threads_per_warp
      if cute.elem_less(sum_OdO_idx, problem_shape[0]):
        cute.copy(
          atom_async_copy,
          sum_OdO_for_copy[None, sum_OdO_idx, (blk_coord_h, blk_coord_b)],
          sSum_OdO_for_copy[None, sSum_OdO_idx, sum_odo_handle.index],
        )
      else:
        sSum_OdO_for_copy[None, sSum_OdO_idx, sum_odo_handle.index].fill(0.0)
    sum_odo_handle.commit()

    iter_count -= 1
    iter_index += 1

    while iter_count > 0:
      if iter_index == iter_end:
        iter_index = iter_start
        blk_coord_h_r += 1
        blk_coord_h = (blk_coord_h_r, blk_coord_h_k)

      for gen in cutlass.range_constexpr(self.d_generations):
        do_handle = load_mma_dO_producer.acquire_and_advance()
        cute.copy(
          tma_atom_dO,
          tdOgdO_mkl[(None, iter_index, gen, (blk_coord_h, blk_coord_b))],
          tdOsdO[None, do_handle.index],
          tma_bar_ptr=do_handle.barrier,
        )

      sum_odo_handle = load_compute_sum_OdO_producer.acquire_and_advance()
      sSum_OdO_for_copy = cute.flat_divide(sSum_OdO, (1, ))
      sum_OdO_for_copy = cute.flat_divide(sum_OdO, (1, ))
      for i in cutlass.range_constexpr(async_copy_num_elts):
        sum_OdO_idx = (
          self.tile_m * iter_index + thread_idx + i * self.threads_per_warp
        )
        sSum_OdO_idx = thread_idx + i * self.threads_per_warp
        if cute.elem_less(sum_OdO_idx, problem_shape[0]):
          cute.copy(
            atom_async_copy,
            sum_OdO_for_copy[None, sum_OdO_idx, (blk_coord_h, blk_coord_b)],
            sSum_OdO_for_copy[None, sSum_OdO_idx, sum_odo_handle.index],
          )
        else:
          sSum_OdO_for_copy[None, sSum_OdO_idx, sum_odo_handle.index].fill(0.0)
      sum_odo_handle.commit()

      iter_count -= 1
      iter_index += 1

    load_mma_V_producer.tail()
    load_mma_dO_producer.tail()
    load_compute_sum_OdO_producer.tail()

  @cute.jit
  def load_qt(
    self,
    mQT: cute.Tensor,
    sQT: cute.Tensor,
    KQ_tiled_mma: cute.TiledMma,
    dSQ_tiled_mma: cute.TiledMma,
    tma_atom_QT: cute.CopyAtom,
    SeqlenInfoCls: Callable,
    iter_count: Int32,
    iter_start: Int32,
    iter_end: Int32,
    load_mma_QT_producer,
    blk_coord_k: Int32,
    blk_coord_h_k: Int32,
    blk_coord_b: Int32,
  ):
    """Issuer warp 11: the QT slices (dK's B operand), one token per iter."""
    blk_coord_h_r = Int32(0)
    blk_coord_h = (blk_coord_h_r, blk_coord_h_k)
    iter_index = iter_start
    # V half only: dSQ's M mode is this CTA's K tile, so warp 11 needs no M.
    mma_tile_coord_v = blk_coord_k % cute.size(KQ_tiled_mma.thr_id.shape)

    cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)
    cta_layout_vmnk = cute.tiled_divide(cta_layout_mnk, (KQ_tiled_mma.thr_id, ))
    cta_in_cluster_coord_vmnk = cta_layout_vmnk.get_flat_coord(
      cute.arch.block_idx_in_cluster()
    )

    seqlen = SeqlenInfoCls(blk_coord_b)
    # QT is the transposed operand: its sequence mode is 1, not 0.
    QT = cute.domain_offset(
      (Int32(0), seqlen.offset_q, ((Int32(0), Int32(0)), Int32(0))), mQT
    )
    gQT = cute.local_tile(
      QT, cute.select(self.dSQ_mma_tiler, mode=[1, 2]), (None, None, None)
    )
    dSQ_thr_mma = dSQ_tiled_mma.get_slice(mma_tile_coord_v)
    tdKgQT = dSQ_thr_mma.partition_B(gQT)

    tQTsQT, tQTgQT_mkl = cute.nvgpu.cpasync.tma_partition(
      tma_atom_QT,
      cta_in_cluster_coord_vmnk[1],
      cute.make_layout(cute.size(cta_layout_vmnk, mode=[1])),
      cute.group_modes(sQT, 0, 3),
      cute.group_modes(tdKgQT, 0, 3),
    )

    qt_handle = load_mma_QT_producer.acquire_and_advance()
    for slice_index in cutlass.range_constexpr(self.TARGET_DK_SLICES):
      cute.copy(
        tma_atom_QT,
        tQTgQT_mkl[(None, slice_index, iter_index, (blk_coord_h, blk_coord_b))],
        tQTsQT[None, qt_handle.index * self.TARGET_DK_SLICES + slice_index],
        tma_bar_ptr=qt_handle.barrier,
      )

    iter_count -= 1
    iter_index += 1

    while iter_count > 0:
      if iter_index == iter_end:
        iter_index = iter_start
        blk_coord_h_r += 1
        blk_coord_h = (blk_coord_h_r, blk_coord_h_k)

      qt_handle = load_mma_QT_producer.acquire_and_advance()
      for slice_index in cutlass.range_constexpr(self.TARGET_DK_SLICES):
        cute.copy(
          tma_atom_QT,
          tQTgQT_mkl[
            (None, slice_index, iter_index, (blk_coord_h, blk_coord_b))],
          tQTsQT[None, qt_handle.index * self.TARGET_DK_SLICES + slice_index],
          tma_bar_ptr=qt_handle.barrier,
        )

      iter_count -= 1
      iter_index += 1

    load_mma_QT_producer.tail()

  @cute.jit
  def mma(
    self,
    KQ_tiled_mma: cute.TiledMma,
    VdO_tiled_mma: cute.TiledMma,
    dSQ_tiled_mma: cute.TiledMma,
    KQ_tmem_tiled_mma: cute.TiledMma,
    tSTrK_tmem_gens: tuple,
    tSTrQ_tmem: cute.Tensor,
    tKT_addr_gens: tuple,
    tiled_s2t: cute.TiledCopy,
    tKsK_s2t_gens: tuple,
    tKtK_s2t_gens: tuple,
    tdPTrV_tmem_gens: tuple,
    tdPTrdO_tmem: cute.Tensor,
    tVT_addr_gens: tuple,
    tVsV_s2t_gens: tuple,
    tVtV_s2t_gens: tuple,
    sQ: cute.Tensor,
    sdO: cute.Tensor,
    tSTtST: cute.Tensor,
    tSTrQ: cute.Tensor,
    tSTrK: cute.Tensor,
    tdPTtdPT: cute.Tensor,
    tdPTrV: cute.Tensor,
    tdPTrdO: cute.Tensor,
    tdKrdST: cute.Tensor,
    tdKtdK: tuple,
    tdKrQT: cute.Tensor,
    iter_count: Int32,
    load_mma_Q_consumer,
    load_mma_K_consumer,
    load_mma_V_consumer,
    mma_compute_S_producer,
    load_mma_dO_consumer,
    mma_compute_dP_producer,
    compute_mma_dS_consumer,
    load_mma_QT_consumer,
    mma_compute_dK_producer,
  ):
    """Warp 8: the sole issuer of every UMMA and every utccp in the kernel."""
    load_mma_K_releaser = load_mma_K_consumer.clone()
    load_mma_V_releaser = load_mma_V_consumer.clone()

    cta_rank_in_cluster = cute.arch.make_warp_uniform(
      cute.arch.block_idx_in_cluster()
    )
    is_leader_cta = cta_rank_in_cluster % 2 == 0

    # One S^T generation; local def, ZERO captures: closure_check bans them.
    def issue_S_generation(
      gen,
      q_slot,
      ktmem_use_gens,
      KQ_tiled_mma,
      KQ_tmem_tiled_mma,
      tSTtST,
      tSTrK_tmem_gens,
      tSTrQ_tmem,
      tKT_addr_gens,
      tSTrK,
      tSTrQ,
      sQ,
    ):
      # Gens below ktmem_use_gens read A from TMEM (gen 0 seeds); SMEM tail.
      if const_expr(gen < ktmem_use_gens):
        gemm_ptx_w_idx(
          KQ_tmem_tiled_mma,
          tSTtST,
          tSTrK_tmem_gens[gen],
          tSTrQ_tmem,
          None,
          sQ,
          A_idx=0,
          B_idx=q_slot,
          zero_init=(gen == 0),
          cta_group=2,
          tA_addr=tKT_addr_gens[gen],
        )
      else:
        # Not a bare True: on the use_gens = 0 build gen 0 must still seed.
        for k_block in cutlass.range(
          0, cute.size(tSTrQ, mode=[2]), unroll_full=True
        ):
          KQ_tiled_mma.set(tcgen05.Field.ACCUMULATE, gen != 0 or k_block != 0)
          cute.gemm(
            KQ_tiled_mma,
            tSTtST,
            tSTrK[None, None, k_block, gen],
            tSTrQ[None, None, k_block, q_slot],
            tSTtST,
          )
      # Rebind everywhere: set(ACCUMULATE) exits scf by NAME, so renames break.
      return KQ_tiled_mma

    # Mirrors issue_S_generation; the prologue scf.if yields BOTH tiled mmas.
    def issue_dP_generation(
      gen,
      do_slot,
      vtmem_use_gens,
      VdO_tiled_mma,
      KQ_tmem_tiled_mma,
      tdPTtdPT,
      tdPTrV_tmem_gens,
      tdPTrdO_tmem,
      tVT_addr_gens,
      tdPTrV,
      tdPTrdO,
      sdO,
    ):
      # Gens below vtmem_use_gens read A from TMEM (gen 0 seeds); SMEM tail.
      if const_expr(gen < vtmem_use_gens):
        gemm_ptx_w_idx(
          KQ_tmem_tiled_mma,
          tdPTtdPT,
          tdPTrV_tmem_gens[gen],
          tdPTrdO_tmem,
          None,
          sdO,
          A_idx=0,
          B_idx=do_slot,
          zero_init=(gen == 0),
          cta_group=2,
          tA_addr=tVT_addr_gens[gen],
        )
      else:
        # Not a bare True: on the use_gens = 0 build gen 0 must still seed.
        for k_block in cutlass.range(
          0, cute.size(tdPTrV, mode=[2]), unroll_full=True
        ):
          VdO_tiled_mma.set(tcgen05.Field.ACCUMULATE, gen != 0 or k_block != 0)
          cute.gemm(
            VdO_tiled_mma,
            tdPTtdPT,
            tdPTrV[None, None, k_block, gen],
            tdPTrdO[None, None, k_block, do_slot],
            tdPTtdPT,
          )
      return VdO_tiled_mma

    # Gemm only: owning k_block rebinds a carrying ACCUMULATE; measured slower.
    def issue_dK_slice(
      dSQ_tiled_mma,
      tdKtdK,
      tdKrdST,
      tdKrQT,
      k_block,
      slice_index,
      ds_slot,
      qt_slot,
      dk_slices,
    ):
      cute.gemm(
        dSQ_tiled_mma,
        tdKtdK[slice_index],
        tdKrdST[None, None, k_block, ds_slot],
        tdKrQT[None, None, k_block, qt_slot * dk_slices + slice_index],
        tdKtdK[slice_index],
      )

    # ---  prologue  ---
    if is_leader_cta:
      if const_expr(self.interleave_s_dp):
        # The prologue issues the loop's 8 gens; dP acquires after S gen 0.
        s_handle = mma_compute_S_producer.acquire_and_advance()
        load_mma_K_consumer.wait_and_advance()
        for gen in cutlass.range_constexpr(self.ktmem_use_gens):
          cute.copy(tiled_s2t, tKsK_s2t_gens[gen], tKtK_s2t_gens[gen])
        load_mma_V_consumer.wait_and_advance()
        for gen in cutlass.range_constexpr(self.vtmem_use_gens):
          cute.copy(tiled_s2t, tVsV_s2t_gens[gen], tVtV_s2t_gens[gen])
        for gen in cutlass.range_constexpr(self.d_generations):
          q_handle = load_mma_Q_consumer.wait_and_advance()
          if const_expr(self.q_gens_per_token == 1):
            q_slot = q_handle.index
          else:
            q_slot = q_handle.index * self.q_gens_per_token
          KQ_tiled_mma = issue_S_generation(
            gen,
            q_slot,
            self.ktmem_use_gens,
            KQ_tiled_mma,
            KQ_tmem_tiled_mma,
            tSTtST,
            tSTrK_tmem_gens,
            tSTrQ_tmem,
            tKT_addr_gens,
            tSTrK,
            tSTrQ,
            sQ,
          )
          q_handle.release()
          if const_expr(gen == 0):
            dp_handle = mma_compute_dP_producer.acquire_and_advance()
          if const_expr(gen + 1 == self.d_generations):
            cute.arch.fence_view_async_tmem_store()
            s_handle.commit()

          do_handle = load_mma_dO_consumer.wait_and_advance()
          VdO_tiled_mma = issue_dP_generation(
            gen,
            do_handle.index,
            self.vtmem_use_gens,
            VdO_tiled_mma,
            KQ_tmem_tiled_mma,
            tdPTtdPT,
            tdPTrV_tmem_gens,
            tdPTrdO_tmem,
            tVT_addr_gens,
            tdPTrV,
            tdPTrdO,
            sdO,
          )
          do_handle.release()

        dp_handle.commit()
      else:
        s_handle = mma_compute_S_producer.acquire_and_advance()
        # K resident: one token covers all gens; wait once, index directly.
        load_mma_K_consumer.wait_and_advance()

        # Park K in TMEM once: tcgen05.cp and .mma issue in order, no fence.
        for gen in cutlass.range_constexpr(self.ktmem_use_gens):
          cute.copy(tiled_s2t, tKsK_s2t_gens[gen], tKtK_s2t_gens[gen])

        # S = K * Q; one Q token covers q_gens_per_token gens (2 on this arm).
        for tok in cutlass.range_constexpr(
          self.d_generations // self.q_gens_per_token
        ):
          q_handle = load_mma_Q_consumer.wait_and_advance()
          for sub in cutlass.range_constexpr(self.q_gens_per_token):
            gen = tok * self.q_gens_per_token + sub
            if const_expr(self.q_gens_per_token == 1):
              q_slot = q_handle.index
            else:
              q_slot = q_handle.index * self.q_gens_per_token + sub
            KQ_tiled_mma = issue_S_generation(
              gen,
              q_slot,
              self.ktmem_use_gens,
              KQ_tiled_mma,
              KQ_tmem_tiled_mma,
              tSTtST,
              tSTrK_tmem_gens,
              tSTrQ_tmem,
              tKT_addr_gens,
              tSTrK,
              tSTrQ,
              sQ,
            )
          q_handle.release()

        cute.arch.fence_view_async_tmem_store()
        s_handle.commit()

        # V is resident on the same terms as K.
        load_mma_V_consumer.wait_and_advance()

        # Park V gens 0..vtmem_use_gens-1 in TMEM, on K's issue-order guarantee.
        for gen in cutlass.range_constexpr(self.vtmem_use_gens):
          cute.copy(tiled_s2t, tVsV_s2t_gens[gen], tVtV_s2t_gens[gen])

        dp_handle = mma_compute_dP_producer.acquire_and_advance()

        # Compute dP = V * dO, over the same generations.
        for gen in cutlass.range_constexpr(self.d_generations):
          do_handle = load_mma_dO_consumer.wait_and_advance()
          VdO_tiled_mma = issue_dP_generation(
            gen,
            do_handle.index,
            self.vtmem_use_gens,
            VdO_tiled_mma,
            KQ_tmem_tiled_mma,
            tdPTtdPT,
            tdPTrV_tmem_gens,
            tdPTrdO_tmem,
            tVT_addr_gens,
            tdPTrV,
            tdPTrdO,
            sdO,
          )
          do_handle.release()

        dp_handle.commit()
      # V produced once: the slot is held to the end, released via the releaser.

    iter_count -= 1

    # ---  steady  ---
    # One shared field: sites stay k_block-outer / slice-inner, so each seeds.
    dSQ_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
    while iter_count > 0:
      if const_expr(not self.interleave_s_dp):
        if is_leader_cta:
          s_handle = mma_compute_S_producer.acquire_and_advance()

          # S = K * Q.  TMEM-A generations first, SMEM tail generation last.
          for tok in cutlass.range_constexpr(
            self.d_generations // self.q_gens_per_token
          ):
            q_handle = load_mma_Q_consumer.wait_and_advance()
            for sub in cutlass.range_constexpr(self.q_gens_per_token):
              gen = tok * self.q_gens_per_token + sub
              if const_expr(self.q_gens_per_token == 1):
                q_slot = q_handle.index
              else:
                q_slot = q_handle.index * self.q_gens_per_token + sub
              KQ_tiled_mma = issue_S_generation(
                gen,
                q_slot,
                self.ktmem_use_gens,
                KQ_tiled_mma,
                KQ_tmem_tiled_mma,
                tSTtST,
                tSTrK_tmem_gens,
                tSTrQ_tmem,
                tKT_addr_gens,
                tSTrK,
                tSTrQ,
                sQ,
              )
            q_handle.release()
          s_handle.commit()

      if const_expr(self.dp_before_dk):
        if is_leader_cta:
          # S then dP x4: Q takes slots 0,2,4,6 and dO 1,3,5,7 -- equal refill.
          s_handle = mma_compute_S_producer.acquire_and_advance()
          for gen in cutlass.range_constexpr(self.d_generations):
            q_handle = load_mma_Q_consumer.wait_and_advance()
            if const_expr(self.q_gens_per_token == 1):
              q_slot = q_handle.index
            else:
              q_slot = q_handle.index * self.q_gens_per_token
            KQ_tiled_mma = issue_S_generation(
              gen,
              q_slot,
              self.ktmem_use_gens,
              KQ_tiled_mma,
              KQ_tmem_tiled_mma,
              tSTtST,
              tSTrK_tmem_gens,
              tSTrQ_tmem,
              tKT_addr_gens,
              tSTrK,
              tSTrQ,
              sQ,
            )
            q_handle.release()
            # S commits before the LAST dP gen: one gen late, not a block.
            if const_expr(gen + 1 == self.d_generations):
              s_handle.commit()

            if const_expr(gen == 0):
              # Not hoisted: at one stage the window is commit -> acquire.
              dp_handle = mma_compute_dP_producer.acquire_and_advance()
            do_handle = load_mma_dO_consumer.wait_and_advance()
            VdO_tiled_mma = issue_dP_generation(
              gen,
              do_handle.index,
              self.vtmem_use_gens,
              VdO_tiled_mma,
              KQ_tmem_tiled_mma,
              tdPTtdPT,
              tdPTrV_tmem_gens,
              tdPTrdO_tmem,
              tVT_addr_gens,
              tdPTrV,
              tdPTrdO,
              sdO,
            )
            do_handle.release()

          dp_handle.commit()

        if is_leader_cta:
          qt_handle = load_mma_QT_consumer.wait_and_advance()
          ds_handle = compute_mma_dS_consumer.wait_and_advance()

          for k_block in cutlass.range(
            0, cute.size(tdKrdST, mode=[2]), unroll_full=True
          ):
            for slice_index in cutlass.range_constexpr(self.TARGET_DK_SLICES):
              issue_dK_slice(
                dSQ_tiled_mma,
                tdKtdK,
                tdKrdST,
                tdKrQT,
                k_block,
                slice_index,
                ds_handle.index,
                qt_handle.index,
                self.TARGET_DK_SLICES,
              )
            dSQ_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
          qt_handle.release()
          ds_handle.release()
      else:
        if is_leader_cta:
          qt_handle = load_mma_QT_consumer.wait_and_advance()
          ds_handle = compute_mma_dS_consumer.wait_and_advance()

          for k_block in cutlass.range(
            0, cute.size(tdKrdST, mode=[2]), unroll_full=True
          ):
            for slice_index in cutlass.range_constexpr(self.TARGET_DK_SLICES):
              issue_dK_slice(
                dSQ_tiled_mma,
                tdKtdK,
                tdKrdST,
                tdKrQT,
                k_block,
                slice_index,
                ds_handle.index,
                qt_handle.index,
                self.TARGET_DK_SLICES,
              )
            dSQ_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
          qt_handle.release()
          ds_handle.release()

        if is_leader_cta:
          dp_handle = mma_compute_dP_producer.acquire_and_advance()
          # V produced once; reuse the gens -- TMEM-A first, SMEM tail last.
          for gen in cutlass.range_constexpr(self.d_generations):
            do_handle = load_mma_dO_consumer.wait_and_advance()
            VdO_tiled_mma = issue_dP_generation(
              gen,
              do_handle.index,
              self.vtmem_use_gens,
              VdO_tiled_mma,
              KQ_tmem_tiled_mma,
              tdPTtdPT,
              tdPTrV_tmem_gens,
              tdPTrdO_tmem,
              tVT_addr_gens,
              tdPTrV,
              tdPTrdO,
              sdO,
            )
            do_handle.release()

          dp_handle.commit()

      iter_count -= 1

    # ---  drain  ---
    if is_leader_cta:
      # Release point: earlier retires a live sK/sV, later races the dK arena.
      load_mma_K_releaser.release()
      load_mma_K_releaser.advance()
      load_mma_V_releaser.release()
      load_mma_V_releaser.advance()

    if is_leader_cta:
      dk_handle = mma_compute_dK_producer.acquire_and_advance()

      ds_handle = compute_mma_dS_consumer.wait_and_advance()
      qt_handle = load_mma_QT_consumer.wait_and_advance()

      # Final Q block, same body; at iter_count == 1 the pre-loop clear seeds.
      for k_block in cutlass.range(
        0, cute.size(tdKrdST, mode=[2]), unroll_full=True
      ):
        for slice_index in cutlass.range_constexpr(self.TARGET_DK_SLICES):
          issue_dK_slice(
            dSQ_tiled_mma,
            tdKtdK,
            tdKrdST,
            tdKrQT,
            k_block,
            slice_index,
            ds_handle.index,
            qt_handle.index,
            self.TARGET_DK_SLICES,
          )
        dSQ_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

      dk_handle.commit()
      qt_handle.release()
      ds_handle.release()

    mma_compute_S_producer.tail()
    mma_compute_dP_producer.tail()
    mma_compute_dK_producer.tail()

  @cute.jit
  def compute_loop(
    self,
    tSTtST: cute.Tensor,
    tdPTtdPT: cute.Tensor,
    # The dK staging arena, forwarded untouched to epilogue() (donor: P buffer).
    sdK_epi: cute.Tensor,
    sLSE: cute.Tensor,
    sdST: cute.Tensor,
    sSum_OdO: cute.Tensor,
    mdK: cute.Tensor,
    tdKtdK: tuple,
    blk_coord: cute.Coord,
    # Not read here; forwarded to the epilogue this role calls. Do not prune.
    SeqlenInfoCls: Callable,
    problem_shape: tuple[Int32, Int32, Int32, tuple[tuple[Int32, Int32],
                                                    Int32]],
    iter_count: Int32,
    iter_start: Int32,
    iter_end: Int32,
    scale_softmax: cutlass.Float32,
    mma_compute_S_consumer,
    load_compute_LSE_consumer,
    load_compute_sum_OdO_consumer,
    mma_compute_dP_consumer,
    compute_mma_dS_producer,
    mma_compute_dK_consumer,
    varlen: bool,
    problem_shape_k_cur_batch: Int32,
    tma_atom_dK: cute.CopyAtom,
    mdK_tma: cute.Tensor,
  ):
    """Warps 0-7: P from S/LSE, dS from dP/dpsum, publish dS, then epilogue."""
    tidx, _, _ = cute.arch.thread_idx()
    seqlen_q, seqlen_k, _, _ = problem_shape
    _, blk_coord_k, _, _ = blk_coord

    iter_index = iter_start

    tmem_load_op = tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(16))
    tmem_load_atom = cute.make_copy_atom(
      tmem_load_op,
      self.acc_dtype,
    )

    tSTtST = tSTtST[(None, None), 0, 0]
    tdPTtdPT = tdPTtdPT[(None, None), 0, 0]

    cST = cute.make_identity_tensor(cute.select(self.cta_tiler, mode=[1, 0]))
    cdPT = cute.make_identity_tensor(cute.select(self.cta_tiler, mode=[1, 0]))

    num_warp_groups = self.num_compute_warps // 4
    dp_idx = tidx % 128
    wg_idx = (tidx % (self.num_compute_warps * self.threads_per_warp)) // 128
    tiled_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tSTtST)
    thr_t2r = tiled_t2r.get_slice(dp_idx)

    tTR_cST = thr_t2r.partition_D(cST)
    tTR_cST = split_wg(tTR_cST, num_warp_groups, wg_idx)
    tTR_rST = cute.make_rmem_tensor(tTR_cST.shape, self.acc_dtype)

    tTR_tST = thr_t2r.partition_S(tSTtST)
    tTR_tST = split_wg(tTR_tST, num_warp_groups, wg_idx)

    tTR_cdPT_p = thr_t2r.partition_D(cdPT)
    tTR_cdPT = split_wg(tTR_cdPT_p, num_warp_groups, wg_idx)
    tTR_rdPT = cute.make_rmem_tensor(tTR_cdPT.shape, self.acc_dtype)

    tTR_tdPT = thr_t2r.partition_S(tdPTtdPT)
    tTR_tdPT = split_wg(tTR_tdPT, num_warp_groups, wg_idx)

    is_residual_k = blk_coord_k * self.tile_n + self.tile_n > seqlen_k
    last_iter = iter_end - 1
    log2_e = cutlass.Float32(math.log2(math.e))
    softmax_scale_log2_e = scale_softmax * log2_e

    while iter_count > 0:
      s_handle = mma_compute_S_consumer.wait_and_advance()
      lse_handle = load_compute_LSE_consumer.wait_and_advance()

      leading_causal_masking = cutlass.Boolean(False)
      if const_expr(self.is_causal):
        # Exact diagonal gate; no pipeline ops, so the CTA pair may diverge.
        leading_causal_masking = (
          iter_index * self.tile_m + seqlen_k - seqlen_q
          < blk_coord_k * self.tile_n + self.tile_n - 1
        )
        leading_causal_masking = cute.arch.shuffle_sync(
          leading_causal_masking, 0
        )

      trailing_residual_masking = iter_index == last_iter or is_residual_k
      trailing_residual_masking = cute.arch.shuffle_sync(
        trailing_residual_masking, 0
      )

      # Interior tiles -- the vast majority at long S -- skip the predicates.
      is_masked_tile = (
        leading_causal_masking or trailing_residual_masking
        or self.has_sliding_window
      )

      # Compute P = softmax(S, LSE)
      cute.copy(tiled_t2r, tTR_tST, tTR_rST)

      if is_masked_tile:
        for i in cutlass.range(cute.size(tTR_rST), unroll_full=True):
          c_transpose = tTR_cST[i]
          pos = (
            cute.get(c_transpose, mode=[1]) + iter_index * self.tile_m,
            cute.get(c_transpose, mode=[0]) + blk_coord_k * self.tile_n,
          )
          if const_expr(self.has_sliding_window):
            if const_expr(self.window_size_left < 0):
              tTR_rST[i] = (
                -cutlass.Float32.inf if pos[1] > pos[0] + seqlen_k - seqlen_q +
                self.window_size_right else tTR_rST[i]
              )
            else:
              max_K_index = min(
                pos[0] + seqlen_k - seqlen_q + self.window_size_right, seqlen_k
              )
              min_K_index = max(
                0, pos[0] + seqlen_k - seqlen_q - self.window_size_left
              )
              tTR_rST[i] = (
                -cutlass.Float32.inf
                if pos[1] > max_K_index or pos[1] < min_K_index else tTR_rST[i]
              )
          if const_expr(self.is_causal) and (
            pos[0] + seqlen_k - seqlen_q < pos[1]
            or not cute.elem_less(pos, (seqlen_q, seqlen_k))
          ):
            tTR_rST[i] = -cutlass.Float32.inf
          if not cute.elem_less(pos, (seqlen_q, seqlen_k)):
            tTR_rST[i] = -cutlass.Float32.inf

      for i in cutlass.range(0, cute.size(tTR_rST), 2, unroll_full=True):
        lse = (
          -sLSE[
            cute.get(tTR_cST[i], mode=[1]),
            lse_handle.index,
          ],
          -sLSE[
            cute.get(tTR_cST[i + 1], mode=[1]),
            lse_handle.index,
          ],
        )
        tTR_rST[i], tTR_rST[i + 1] = cute.arch.fma_packed_f32x2(
          (tTR_rST[i], tTR_rST[i + 1]),
          (softmax_scale_log2_e, softmax_scale_log2_e),
          lse,
        )
        tTR_rST[i] = cute.math.exp2(tTR_rST[i], fastmath=True)
        tTR_rST[i + 1] = cute.math.exp2(tTR_rST[i + 1], fastmath=True)

      # The barrier proves S loads ISSUED; only this fence proves they landed.
      cute.arch.fence_view_async_tmem_load()
      cute.arch.barrier(
        barrier_id=self.compute_sync_bar_id,
        number_of_threads=self.num_compute_warps * self.threads_per_warp,
      )

      s_handle.release()
      lse_handle.release()

      sum_odo_handle = load_compute_sum_OdO_consumer.wait_and_advance()
      dp_handle = mma_compute_dP_consumer.wait_and_advance()
      ds_handle = compute_mma_dS_producer.acquire_and_advance()

      # Compute dS = dsoftmax(P, dP, sum_OdO)
      cute.copy(tiled_t2r, tTR_tdPT, tTR_rdPT)

      # dP dies at this t2r; interleave acquires at slot 1 (mma_acq_dp stall).
      if const_expr(self.interleave_s_dp):
        cute.arch.fence_view_async_tmem_load()
        dp_handle.release()

      for i in cutlass.range(0, cute.size(tTR_rdPT), 2, unroll_full=True):
        dpsum_0 = -sSum_OdO[
          cute.get(tTR_cdPT[i], mode=[1]),
          sum_odo_handle.index,
        ]
        dpsum_1 = -sSum_OdO[
          cute.get(tTR_cdPT[i + 1], mode=[1]),
          sum_odo_handle.index,
        ]
        if const_expr(varlen):
          if not cute.elem_less(cute.get(tTR_cdPT[i], mode=[1]), seqlen_q):
            dpsum_0 = 0.0
          if not cute.elem_less(cute.get(tTR_cdPT[i + 1], mode=[1]), seqlen_q):
            dpsum_1 = 0.0
        tTR_rdPT[i], tTR_rdPT[i + 1] = cute.arch.add_packed_f32x2(
          (tTR_rdPT[i], tTR_rdPT[i + 1]),
          (dpsum_0, dpsum_1),
        )
        tTR_rdPT[i], tTR_rdPT[i + 1] = cute.arch.mul_packed_f32x2(
          (tTR_rdPT[i], tTR_rdPT[i + 1]), (tTR_rST[i], tTR_rST[i + 1])
        )
      # Zero dS at masked (q,k) or dK is wrong; the SIM102 nesting is required.
      if const_expr(self.is_causal):  # noqa: SIM102
        if is_masked_tile:
          for i in cutlass.range(cute.size(tTR_rdPT), unroll_full=True):
            c_transpose = tTR_cdPT[i]
            pos = (
              cute.get(c_transpose, mode=[1]) + iter_index * self.tile_m,
              cute.get(c_transpose, mode=[0]) + blk_coord_k * self.tile_n,
            )
            if (
              pos[0] + seqlen_k - seqlen_q < pos[1]
              or not cute.elem_less(pos, (seqlen_q, seqlen_k))
            ):
              tTR_rdPT[i] = cutlass.Float32(0.0)

      # A one-key row has analytically zero dS; fp disagrees ~1 ulp, past tol.
      single_key_tile = cutlass.Boolean(False)
      if const_expr(self.is_causal):
        single_key_tile = iter_index * self.tile_m <= seqlen_q - seqlen_k
        single_key_tile = cute.arch.shuffle_sync(single_key_tile, 0)
      # Rows: seqlen_k <= 1 or causal q <= seqlen_q - seqlen_k: diagonal key.
      if single_key_tile or seqlen_k <= 1:
        for i in cutlass.range(cute.size(tTR_rdPT), unroll_full=True):
          if const_expr(self.is_causal):
            q_pos = cute.get(tTR_cdPT[i], mode=[1]) + iter_index * self.tile_m
            if q_pos <= seqlen_q - seqlen_k:
              tTR_rdPT[i] = cutlass.Float32(0.0)
          if seqlen_k <= 1:
            tTR_rdPT[i] = cutlass.Float32(0.0)
      # NOT cvt_f16: .to() packs F2FP.BF16.F32.PACK_AB; cvt_f16's asm blocks.
      tTR_rdST = self.quantize(tTR_rdPT, mdK.element_type)

      if const_expr(not self.interleave_s_dp):
        # The non-fused arm keeps the donor point: mma_acq_dp already at floor.
        cute.arch.fence_view_async_tmem_load()
        dp_handle.release()

      reg_to_smem_mma128x128_2cta(
        tTR_rdST,
        sdST,
        ds_handle.index,
        (self.tile_n, self.tile_m),
        dp_idx,
        wg_idx,
      )
      cute.arch.fence_view_async_shared()
      cute.arch.barrier(
        barrier_id=self.compute_sync_bar_id,
        number_of_threads=self.num_compute_warps * self.threads_per_warp,
      )

      ds_handle.commit()
      sum_odo_handle.release()

      iter_count -= 1
      iter_index += 1
      if iter_index == iter_end:
        iter_index = iter_start

    # In-role: split_wg(.., 2) spans both groups; hoisting moves the dS tail().
    self.epilogue(
      blk_coord,
      SeqlenInfoCls,
      problem_shape,
      mdK,
      tdKtdK,
      scale_softmax,
      mma_compute_dK_consumer,
      problem_shape_k_cur_batch,
      tma_atom_dK,
      mdK_tma,
      varlen,
      sdK_epi,
    )

    compute_mma_dS_producer.tail()

  @cute.jit
  def quantize(
    self,
    input_t: cute.Tensor,
    element_dtype: type[cutlass.Numeric],
  ) -> cute.Tensor:
    """Register-tensor cast to element_dtype."""
    # Deliberately not utils.cvt_f16; the measured reason sits at the dS site.
    output = cute.make_rmem_tensor(input_t.shape, element_dtype)
    output.store(input_t.load().to(element_dtype))
    return output

  @cute.jit
  def store_dK(
    self,
    gmem: cute.Tensor,
    regs: cute.Tensor,
    coord: cute.Tensor,
    tensor_shape: cute.Shape,
  ):
    """Varlen path: predicated reg -> global (the TMA desc is dense-only)."""
    for i in cutlass.range(cute.size(coord, mode=[2]), unroll_full=True):
      if cute.elem_less(coord[None, 0, i][0], tensor_shape):
        # TODO: not lane-map forced; donor is 128-bit (a numerics edit, open).
        gmem[None, 0, i].store(regs[None, 0, i].load())

  @cute.jit
  def epilogue_clear(
    self,
    blk_coord: cute.Coord,
    SeqlenInfoCls: Callable,
    problem_shape: tuple[Int32, Int32, Int32, tuple[tuple[Int32, Int32],
                                                    Int32]],
    mdK: cute.Tensor,
  ):
    """Zero the dK tile of a KV block that no Q block visits."""
    # No visible Q block still writes: the kernel contract assumes no pre-zero.
    tidx, _, _ = cute.arch.thread_idx()
    _, seqlen_k, _, HB = problem_shape
    _, blk_coord_k, _, blk_coord_batch = blk_coord

    # Last member is the ((h_r,h_k),b) nest: index it or SeqlenInfoQK is rank-2.
    seqlen = SeqlenInfoCls(blk_coord_batch[1])
    mdK_offset = cute.assume(seqlen.offset_k * mdK.stride[0], divby=64)
    mdK = cute.make_tensor(
      mdK.iterator + mdK_offset,
      cute.make_layout((seqlen_k, self.tile_hdim, HB), stride=mdK.stride),
    )
    gdK = cute.local_tile(
      mdK, (self.cta_tiler[1], self.cta_tiler[2]), (None, None, None)
    )
    gdK = gdK[None, None, blk_coord_k, 0, blk_coord_batch]
    cdK = cute.domain_offset(
      (blk_coord_k * self.tile_n, 0),
      cute.make_identity_tensor((self.cta_tiler[1], self.cta_tiler[2])),
    )

    num_zero_epi_threads = self.num_compute_warps * self.threads_per_warp

    tiled_copy_r2g = copy_utils.tiled_copy_2d(
      mdK.element_type, self.cta_tiler[2], num_zero_epi_threads
    )

    thr_copy_r2g = tiled_copy_r2g.get_slice(tidx)

    tRG_gdK = thr_copy_r2g.partition_D(gdK)
    tRG_cdK = thr_copy_r2g.partition_D(cdK)

    zero_frg = cute.make_rmem_tensor_like(tRG_gdK[None, 0, None])
    zero_frg.fill(mdK.element_type(0.0))

    if tidx < num_zero_epi_threads:
      for n in cutlass.range(cute.size(tRG_gdK.shape[1]), unroll_full=True):
        if cute.elem_less(tRG_cdK[0, n, 0][0], problem_shape[1]):
          cute.copy(tiled_copy_r2g, zero_frg, tRG_gdK[None, n, None])

  @cute.jit
  def epilogue(
    self,
    blk_coord: cute.Coord,
    SeqlenInfoCls: Callable,
    problem_shape: tuple[Int32, Int32, Int32, tuple[tuple[Int32, Int32],
                                                    Int32]],
    mdK: cute.Tensor,
    tdKtdK: tuple,
    scale_softmax: cutlass.Float32,
    mma_compute_dK_consumer,
    problem_shape_k_cur_batch: Int32,
    tma_atom_dK: cute.CopyAtom,
    mdK_tma: cute.Tensor,
    varlen: bool,
    sdK_epi: cute.Tensor,
  ):
    """Drain dK (AccumulatorHandoff consumer): TMEM -> regs -> arena -> TMA."""
    tidx, _, _ = cute.arch.thread_idx()
    _, seqlen_k, head_dim, HB = problem_shape
    _, blk_coord_k, _, blk_coord_batch = blk_coord

    tmem_copy_op = tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32))
    load_op = cute.make_copy_atom(
      tmem_copy_op,
      self.acc_dtype,
    )

    # As in epilogue_clear: the batch index is inside the head/batch nest.
    seqlen = SeqlenInfoCls(blk_coord_batch[1])
    mdK_offset = cute.assume(seqlen.offset_k * mdK.stride[0], divby=64)
    mdK = cute.make_tensor(
      mdK.iterator + mdK_offset,
      cute.make_layout((seqlen_k, self.tile_hdim, HB), stride=mdK.stride),
    )

    (
      num_warp_groups,
      slice_columns,
      _epi_cols_dK,
      epi_tile_dK,
      total_epi_stages,
    ) = self.dk_epilogue_tiling(mdK.element_type.width)
    dp_idx = tidx % 128
    wg_idx = (tidx % (self.num_compute_warps * self.threads_per_warp)) // 128
    leader_warp = (cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4) == 0
    cta_threads = self.num_compute_warps * self.threads_per_warp

    # Slices tile identically; views differ only in the head_dim tile below.
    gdK_slices = cute.local_tile(
      mdK, (self.cta_tiler[1], slice_columns), (None, None, None)
    )
    if const_expr(not varlen):
      mdK_tma_3d = cute.make_tensor(
        mdK_tma.iterator,
        cute.make_layout((seqlen_k, self.cta_tiler[2], HB),
                         stride=mdK_tma.stride),
      )
      mdK_tma_cur = mdK_tma_3d[None, None, blk_coord_batch]

    # One token, both slices; bind outside the bounds branch (MLIR dominance).
    dk_handle = mma_compute_dK_consumer.wait_and_advance()
    for slice_index in cutlass.range_constexpr(self.TARGET_DK_SLICES):
      if blk_coord_k * self.tile_n < problem_shape_k_cur_batch:
        tdKtdK_slice = tdKtdK[slice_index][(None, None), 0, 0]
        slice_base = slice_index * slice_columns

        gdK = gdK_slices[None, None, blk_coord_k, slice_index, blk_coord_batch]
        # The coord carries K block AND slice base, or varlen checks wrong end.
        cdK = cute.domain_offset(
          (blk_coord_k * self.tile_n, slice_base),
          cute.make_identity_tensor((self.cta_tiler[1], slice_columns)),
        )
        tiled_t2r_dK = tcgen05.make_tmem_copy(load_op, tdKtdK_slice)
        thread_t2r_dK = tiled_t2r_dK.get_slice(dp_idx)

        tTR_cdK = thread_t2r_dK.partition_D(cdK)
        tTR_cdK = split_wg(tTR_cdK, num_warp_groups, wg_idx)
        tTR_gdK = thread_t2r_dK.partition_D(gdK)
        tTR_gdK = split_wg(tTR_gdK, num_warp_groups, wg_idx)
        tTR_rdK = cute.make_rmem_tensor(tTR_cdK.shape, self.acc_dtype)
        tTR_tdK = thread_t2r_dK.partition_S(tdKtdK_slice)
        tTR_tdK = split_wg(tTR_tdK, num_warp_groups, wg_idx)

        cute.copy(tiled_t2r_dK, tTR_tdK, tTR_rdK)

        for i in cutlass.range(cute.size(tTR_rdK), unroll_full=True):
          tTR_rdK[i] = scale_softmax * tTR_rdK[i]

        tTR_rdK_cast = self.quantize(tTR_rdK, mdK.element_type)

        # The varlen fork sits HERE: T2R/scale/quantize shared above.
        if const_expr(not varlen):
          # Staging map n -> (n%epi_cols, n//epi_cols) as a layout, not a loop.
          s_epi_2d = cute.group_modes(sdK_epi, 1, 3)
          s_epi_slice = cute.local_tile(
            s_epi_2d,
            (self.cta_tiler[1], slice_columns),
            (0, slice_index),
          )
          # T2R's own thread slice: reg i == smem i; 128 STS.U16 -> 16 STS.128.
          tTR_sdK = split_wg(
            thread_t2r_dK.partition_D(s_epi_slice),
            num_warp_groups,
            wg_idx,
          )
          cute.autovec_copy(tTR_rdK_cast, tTR_sdK)
        else:
          self.store_dK(tTR_gdK, tTR_rdK_cast, tTR_cdK, (seqlen_k, head_dim))

    # Read-back and rendezvous run once for both slices, outside the slice loop.
    if blk_coord_k * self.tile_n < problem_shape_k_cur_batch:
      if const_expr(not varlen):
        gdK_tma = cute.local_tile(
          mdK_tma_cur,
          (self.cta_tiler[1], self.cta_tiler[2]),
          (blk_coord_k, 0),
        )
        gdK_tma_epi = cute.local_tile(gdK_tma, epi_tile_dK, (0, None))
        cute.arch.fence_view_async_shared()
        # Both warp-groups finish writing before the leader's TMA reads back.
        cute.arch.barrier(
          barrier_id=self.epilogue_arena_bar_id,
          number_of_threads=cta_threads,
        )
        if leader_warp and wg_idx == 0:
          for _stage in cutlass.range_constexpr(
            total_epi_stages * self.TARGET_DK_SLICES
          ):
            sdK_stage = sdK_epi[None, None, _stage]
            gdK_stage = gdK_tma_epi[None, None, _stage]
            # The one tma_get_copy_fn fit: single_stage groups at rank.
            store_dK_stage, _, _ = copy_utils.tma_get_copy_fn(
              tma_atom_dK,
              0,
              cute.make_layout(1),
              sdK_stage,
              gdK_stage,
              single_stage=True,
            )
            store_dK_stage()
            cute.arch.cp_async_bulk_commit_group()
        # No wait here: sK has no later reader; the drain hides at CTA exit.

    # Fence the TMEM loads, then hand the dK token back to the MMA warp.
    cute.arch.fence_view_async_tmem_load()
    dk_handle.release()

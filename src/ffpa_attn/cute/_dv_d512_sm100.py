# Copyright (c) DefTruth, qyjdef@163.com
# Copyright (c) Butterfingrz，13524387014@163.com
#
# Adapted from the SM100 head-dim 256 specialized implementation
# in https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/sm100_hd256_2cta_fmha_backward_dkdvkernel.py
# via flash-attention-512-dev/flash_attn/cute/sm100_hd512_2cta_fmha_backward_dvkernel.py ("the donor" below).
#
# SM100 (Blackwell) backward dV for FFPA attention — only head_dim=512.
#
# Produces dV alone: dV = P^T @ dO with P = exp2(S * scale * log2e - lse) and
# S^T = K @ Q^T; V, dpsum, dP and dS feed dK only.  The kernel is KV-stationary
# — grid (ceil(Sk/64), H_kv, B), one K tile per CTA pair for its whole sweep —
# so dV reduces entirely in TMEM with no workspace, atomics or postprocess, and
# GQA folds into that sweep (trip count = Q blocks x h_r).
#
# Design (12 warps / 384 threads, per-CTA tile (64, 64, 512)):
#   - LOAD (warp 9) TMAs K (stationary across the tile) and Q, cp.asyncs LSE
#     and owns the TMEM allocation; under causal it also issues dO^T.
#   - dO^T LOAD (warp 10, non-causal only) keeps dO^T's ring acquire from
#     queueing Q(j) behind PdO(j-1).  Warp 11 is empty.
#   - MMA (warp 8): S^T = K @ Q^T pair-wide on (128, 64, 512), then dV += P^T
#     @ dO on (128, 256, 64) — the head dim is two slices because the bf16 atom
#     caps that N mode at 256.  A K^T prefix parks in TMEM to drop the sK read.
#   - COMPUTE (warps 0-7) masks and exponentiates S^T against the LSE, re-zeros
#     the NaNs an empty row's -inf would make, publishes P^T into sP, drains dV.
#   - TMEM: dV [0,256) as 2 x 128 columns, S from 256 as (1 + mma_skew) x 32,
#     the parked K^T prefix filling the tail — both forks end at column 512.
#   - Epilogue stages through the dead sK; the scale belongs to dK, not here.
#   - Causal and non-causal are two trace-time builds of one entry, forked on
#     const_expr(is_causal): MMA skew, dO^T issuer, and where views are built.
#
# Constraints:
#   - cta_tiler == (64, 64, 512), and split_head is required
#   - cluster (2, 1, 1); pair-wide MMA M must stay <= 128 so the 2-CTA
#     accumulator keeps folding N
#   - lse_log2 is log2-domain; batch must agree across Q, K and dO
#   - No persistent or CLC scheduling, and no causal + sliding window
#   - Sliding window: ctor-accepted but unreachable (the wrapper passes None)

import math
from functools import partial
from typing import Callable

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.utils import LayoutEnum
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils_basic
from cutlass.pipeline import Agent, CooperativeGroup, pipeline_init_arrive, pipeline_init_wait

from .utils import copy_utils
from . import utils
from .utils.cute_dsl_utils import assume_tensor_aligned
from .utils.block_info import BlockInfo
from .utils.named_barrier import NamedBarrierBwdDVSm100Hd512
from .utils.mask import AttentionMask
from .utils.seqlen_info import SeqlenInfoQK
# TMEM-A bases at a dummy 0x0: without tA_addr a gemm silently reads col 0.
from .utils.blackwell_helpers import (
  SM100_SMEM_CAPACITY_BYTES,
  SM100_TMEM_CAPACITY_COLUMNS,
  gemm_ptx_w_idx,
)
# tmem_offset stays unimported: every TMEM tensor is lane 0, plain col offset.
from .utils.hd512_helpers import (
  check_tmem_intervals,
  reg_to_smem_mma128x128_2cta,
  split_wg,
)


class FFPAAttnBwdDVSm100D512:
  """SM100 D512 backward dV: dV = P^T @ dO alone (dK/dQ live in the
  FFPAAttnBwdDK/DQSm100D512 siblings).

  S^T = K @ Q^T is recomputed per Q tile; V, dpsum, dP and dS never appear
  here (see the module header).
  """
  arch = 100
  # At pair-wide M <= 128 the 2-CTA accumulator folds N (half the columns).
  SM100_MMA_MAX_FOLDING_M = 128

  # Witness declarations: class literals on purpose, never derived values.
  TARGET_HEAD_DIM = 512
  TARGET_CTA_TILER = (64, 64, 512)
  TARGET_CLUSTER_SHAPE_MNK = (2, 1, 1)
  # PdO edge (pair M, dV slice N, Q tile); N at the atom cap => 2 slices.
  TARGET_MMA_SLICE = (128, 256, 64)
  TARGET_DV_SLICES = 2
  # dV regions from 0; stride dv_slice_n // 2: a 2-CTA accumulator splits N.
  TARGET_TMEM_DV_BASE = 0
  TARGET_TMEM_DV_SLICE_STRIDE = 128
  # S starts where dV ends: one region per S stage, kq_s per-CTA face wide.
  TARGET_TMEM_S_BASE = 256
  TARGET_TMEM_S_SLICE_STRIDE = 32

  def __init__(
    self,
    acc_dtype: type[cutlass.Numeric],
    cta_tiler: tuple[int, int, int],
    is_causal: bool,
    window_size_left: int | None,
    window_size_right: int | None,
    is_persistent: bool,
    split_head: bool,
    use_clc_scheduler: bool = False,
  ):
    """Configure the D512 dV kernel."""
    # The epilogue stages dV through sK; safe only with no later sK producer.
    assert not is_persistent, (
      "SM100 backward with head_dim=512 does not support persistent scheduling"
    )
    assert not use_clc_scheduler, (
      "SM100 backward with head_dim=512 does not support the CLC/persistent "
      "scheduler"
    )
    # Un-split would need one N=512 accumulator, over the atom's N=256 cap.
    assert split_head, (
      "SM100 backward with head_dim=512 requires split_head=True"
    )
    self.acc_dtype = acc_dtype
    self.cta_tiler = cta_tiler
    self.tile_shape_Q = cta_tiler[0]
    self.tile_shape_K = cta_tiler[1]
    self.tile_shape_dV_dO = cta_tiler[2]
    # Slice width saturates the atom's N=256 cap: fewest PdO MMAs per tile.
    self.dv_slice_n = min(cta_tiler[2], 256)
    # Extent matches sK by construction; liveness is proved at the alias site.
    # Always True; else arm + 1-elt s_epi_dV stub keep the donor SharedStorage.
    self.alias_epilogue_onto_sK = True
    self.dv_slices = cta_tiler[2] // self.dv_slice_n
    self.KQ_mma_tiler = (
      cta_tiler[1] * 2,
      cta_tiler[0],
      cta_tiler[2],
    )
    # For dV -- one slice of the head dimension per MMA
    self.PdO_mma_tiler = (
      cta_tiler[1] * 2,
      self.dv_slice_n,
      cta_tiler[0],
    )
    self.cluster_shape_mn = (2, 1)
    self.cluster_shape_mnk = (
      *self.cluster_shape_mn, 1
    )  # type: ignore[assignment]
    self.is_causal = is_causal
    self.window_size_left: int = -1 if window_size_left is None else window_size_left
    self.window_size_right: int = -1 if window_size_right is None else window_size_right
    self.has_sliding_window = False
    if self.window_size_left > 0 or self.window_size_right > 0:
      self.has_sliding_window = True
    if self.is_causal:
      self.window_size_right = 0
    # mask.py asserts causal and local exclusive; no suite covers the combo.
    assert not (self.is_causal and self.has_sliding_window), (
      "the d512 dV kernel does not implement a causal sliding window: pass "
      "window_size_left/right only with is_causal=False"
    )

    self.compute_warp_id = (0, 1, 2, 3, 4, 5, 6, 7)
    self.mma_warp_id = 8
    self.load_warp_id = 9
    # Separate issuer: dOT's ring acquire would queue Q(j) behind PdO(j-1).
    self.dot_load_warp_id = 10
    # Not dispatched on: the kernel's final `else` arm is warp 11's role.
    self.empty_warp_id = 11

    self.num_compute_warps = 8

    self.tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

    self.threads_per_warp = 32
    self.threads_per_cta = self.threads_per_warp * len((
      *self.compute_warp_id,
      self.mma_warp_id,
      self.load_warp_id,
      self.dot_load_warp_id,
      self.empty_warp_id,
    ))

    self.cta_sync_bar_id = int(NamedBarrierBwdDVSm100Hd512.CtaSync)
    self.tmem_alloc_sync_bar_id = int(NamedBarrierBwdDVSm100Hd512.TmemPtr)
    self.compute_sync_bar_id = int(NamedBarrierBwdDVSm100Hd512.Compute)
    self.epilogue_sync_bar_id = int(NamedBarrierBwdDVSm100Hd512.Epilogue)
    # Arena barrier: both WGs finish writing before the leader warp TMA-reads.
    self.epilogue_arena_bar_id = int(NamedBarrierBwdDVSm100Hd512.EpilogueArena)

    # Depths settle first: the ledger check and tests read them sans __call__.
    self._setup_attributes()

    # S derives from the dV end: donor's cta_tiler[2] base overflows at D512.
    self.tmem_dv_columns_per_slice = self.dv_slice_n // self.cluster_shape_mn[0]
    self.tmem_dV_base = 0
    self.tmem_dV_slice_stride = self.tmem_dv_columns_per_slice
    self.tmem_S_base = self.tmem_dV_base + self.dv_slices * self.tmem_dV_slice_stride
    # One physical S region per stage: deepening barriers alone silently races.
    self.tmem_S_slice_stride = self.KQ_mma_tiler[1] // self.cluster_shape_mn[0]
    # TMEM KT prefix removes the dominant sK re-read: 64 KiB of a 216 KiB step.
    self.kq_tmem_k = 384 if self.is_causal else 320
    # 16-bit elements pack two per 32-bit TMEM word.
    self.tmem_KT_cols = self.kq_tmem_k * 16 // 32
    # Both forks end at column 512: 352 + 160 non-causal, 320 + 192 causal.
    self.tmem_KT_base = (
      self.tmem_S_base + self.mma_compute_S_stage * self.tmem_S_slice_stride
    )

    # The one non-literal TARGET_*: S stages = 1 + mma_skew (2 causal, 3 non).
    self.TARGET_TMEM_S_STAGES = self.mma_compute_S_stage

    assert cta_tiler[2] == self.TARGET_HEAD_DIM, (
      f"this candidate is the head-dim-{self.TARGET_HEAD_DIM} kernel, got {cta_tiler[2]}"
    )
    assert tuple(cta_tiler) == self.TARGET_CTA_TILER, (
      f"CTA tiler {tuple(cta_tiler)} is not the declared d512 geometry "
      f"{self.TARGET_CTA_TILER}"
    )
    assert (*self.cluster_shape_mn, 1) == self.TARGET_CLUSTER_SHAPE_MNK, (
      f"cluster {(*self.cluster_shape_mn, 1)} is not the declared "
      f"{self.TARGET_CLUSTER_SHAPE_MNK}"
    )
    assert self.dv_slices == self.TARGET_DV_SLICES, (
      f"{self.dv_slices} dV slices were constructed, {self.TARGET_DV_SLICES} declared"
    )
    assert (
      self.tmem_dV_base,
      self.tmem_dV_slice_stride,
      self.tmem_S_base,
      self.tmem_S_slice_stride,
    ) == (
      self.TARGET_TMEM_DV_BASE,
      self.TARGET_TMEM_DV_SLICE_STRIDE,
      self.TARGET_TMEM_S_BASE,
      self.TARGET_TMEM_S_SLICE_STRIDE,
    ), (
      f"the derived TMEM ledger (dV {self.tmem_dV_base}+{self.tmem_dV_slice_stride}, "
      f"S {self.tmem_S_base}+{self.tmem_S_slice_stride}) does not match the "
      f"declared one (dV {self.TARGET_TMEM_DV_BASE}+{self.TARGET_TMEM_DV_SLICE_STRIDE}, "
      f"S {self.TARGET_TMEM_S_BASE}+{self.TARGET_TMEM_S_SLICE_STRIDE})"
    )
    # Above the fold limit the ledger under-counts 2x, silently.
    pair_mma_m = cta_tiler[1] * self.cluster_shape_mn[0]
    assert pair_mma_m <= self.SM100_MMA_MAX_FOLDING_M, (
      f"pair-wide MMA M is {pair_mma_m}, above "
      f"the {self.SM100_MMA_MAX_FOLDING_M} fold limit; the 2-CTA accumulator "
      f"would stop folding N and the TMEM ledger would need "
      f"{self.dv_slices * self.dv_slice_n + self.KQ_mma_tiler[1]} columns"
    )
    assert self.TARGET_MMA_SLICE == self.PdO_mma_tiler, (
      "declared MMA slice does not match the PdO tiler actually constructed"
    )
    assert self.dv_slices * self.dv_slice_n == self.TARGET_HEAD_DIM, (
      f"{self.dv_slices} slices of {self.dv_slice_n} do not cover "
      f"head dim {self.TARGET_HEAD_DIM}"
    )
    assert 16 <= self.dv_slice_n <= 256 and self.dv_slice_n % 16 == 0, (
      f"dV slice N {self.dv_slice_n} is outside what the tcgen05 bf16 "
      f"atom can construct (16..256, multiple of 16)"
    )
    assert self.TARGET_CLUSTER_SHAPE_MNK[0] == 2, (
      "declared cluster M must equal the 2-CTA tcgen05 group"
    )
    # check_tmem_intervals rejects overlap, not gaps; no spare columns here.
    assert (
      self.TARGET_TMEM_DV_BASE + self.TARGET_DV_SLICES *
      self.TARGET_TMEM_DV_SLICE_STRIDE == self.TARGET_TMEM_S_BASE
    ), "S must start where the last dV slice ends"
    # Zero S stages emit no interval, so the shared gate cannot reject it.
    assert self.TARGET_TMEM_S_STAGES >= 1, (
      f"the S pipeline needs at least one physical region, got "
      f"{self.TARGET_TMEM_S_STAGES}"
    )
    # Not column-range facts; the capacity half lives in check_tmem_intervals.
    kq_tmem_k = self.kq_tmem_k
    assert kq_tmem_k % 16 == 0 and 0 < kq_tmem_k <= self.cta_tiler[2], (
      f"TMEM K split {self.kq_tmem_k} must cover whole 16-wide k-blocks "
      f"inside head dim {self.cta_tiler[2]}"
    )
    # Ledger gate (no fragments yet): disjoint regions, one S region per stage.
    check_tmem_intervals(self.tmem_region_intervals(self.tmem_dV_slice_stride))

    self.num_regs_compute = 128
    self.num_regs_mma = 128
    self.num_regs_empty = 96
    self.num_regs_load = 96

    self.buffer_align_bytes = 128

  def _setup_attributes(self):
    """Ring depths and agent topology (runs in __init__; idempotent)."""
    # Q/dOT/LSE deepen for overlap; S/P/dV are rendezvous: depth widens races.
    self.load_mma_Q_stage = 2
    # K = 1 -- stationary across the tile's whole Q x h_r loop.
    self.load_mma_K_stage = 1
    self.load_mma_dOT_stage = 2
    # LSE acquire is the residual stall: 4 stages (+512 B); causal needs 2.
    self.load_compute_LSE_stage = 2 if self.is_causal else 4
    # Skew: KQ(i+skew) issues under softmax(i); causal measured 1 (half steps).
    self.mma_skew = 1 if self.is_causal else 2
    # Topology: non-causal gives dO^T its own issuer warp; causal does not.
    self.split_dot_issuer = not self.is_causal
    # Non-causal's third region buys two S issues of lead for one 32-col stage.
    self.mma_compute_S_stage = 1 + self.mma_skew
    # Two sP slots (+8 KiB of 26) overlap softmax(i)/PdO(i-1); index by stage.
    self.compute_mma_P_stage = 2
    # One publish/wait per tile: depth 1 is exact (the donor's 2nd was dK's).
    self.mma_compute_dV_stage = 1

  def _get_tiled_mma(self):
    """S^T = K @ Q^T and dV += P^T @ dO (KQ TMEM-A face differs in a_source)."""
    cta_group = tcgen05.CtaGroup.TWO
    KQ_tiled_mma = sm100_utils_basic.make_trivial_tiled_mma(
      self.k_dtype,
      self.k_major_mode,
      self.q_major_mode,
      self.acc_dtype,
      cta_group,
      self.KQ_mma_tiler[:2],
    )
    # A-in-TMEM edge: first kq_tmem_k/16 k-blocks; the SMEM edge does the rest.
    KQ_tmem_tiled_mma = sm100_utils_basic.make_trivial_tiled_mma(
      self.k_dtype,
      self.k_major_mode,
      self.q_major_mode,
      self.acc_dtype,
      cta_group,
      self.KQ_mma_tiler[:2],
      tcgen05.OperandSource.TMEM,
    )
    PdO_tiled_mma = sm100_utils_basic.make_trivial_tiled_mma(
      self.do_dtype,
      self.p_major_mode,
      tcgen05.OperandMajorMode.MN,
      self.acc_dtype,
      cta_group,
      self.PdO_mma_tiler[:2],
      tcgen05.OperandSource.SMEM,
    )
    self.KQ_tiled_mma = KQ_tiled_mma
    self.KQ_tmem_tiled_mma = KQ_tmem_tiled_mma
    self.PdO_tiled_mma = PdO_tiled_mma
    return KQ_tiled_mma, KQ_tmem_tiled_mma, PdO_tiled_mma

  def tmem_region_intervals(self, region_columns):
    """Column ledger; region_columns is the declared dV slice width."""
    # S width is kq_s's per-CTA face; a stride mismatch overlaps S regions.
    intervals = {}
    for index in range(self.dv_slices):
      start = self.tmem_dV_base + index * self.tmem_dV_slice_stride
      intervals[f"dV{index}"] = (start, start + region_columns)
    s_columns = self.KQ_mma_tiler[1] // self.cluster_shape_mn[0]
    for index in range(self.mma_compute_S_stage):
      start = self.tmem_S_base + index * self.tmem_S_slice_stride
      intervals[f"S{index}"] = (start, start + s_columns)
    intervals["KT"] = (self.tmem_KT_base, self.tmem_KT_base + self.tmem_KT_cols)
    return intervals

  def dv_epilogue_tiling(self, element_width_bits):
    """Single source for arena, TMA box, loop bound; unit: one WG's D share."""
    num_wgs = self.num_compute_warps // 4
    columns_per_wg = self.cta_tiler[2] // num_wgs
    epi_columns = math.gcd(128 // (element_width_bits // 8), columns_per_wg)
    stages_per_wg = columns_per_wg // epi_columns
    return (
      num_wgs,
      epi_columns,
      (self.cta_tiler[1], epi_columns),
      num_wgs * stages_per_wg,
    )

  @staticmethod
  def _compute_bwd_grid(
    problem_shape: tuple[Int32, Int32, Int32, tuple[tuple[Int32, Int32],
                                                    Int32]],
    block_k: int,
  ) -> tuple[Int32, Int32, Int32]:
    """Launch grid for dV (dK's comes from the scheduler: head_group, LPT)."""
    seqlen_k = problem_shape[1]
    _, H_K = problem_shape[3][0]
    B = problem_shape[3][1]
    return (cute.ceil_div(seqlen_k, block_k), cute.size(H_K), cute.size(B))

  @cute.jit
  def __call__(
    self,
    Q: cute.Tensor,
    K: cute.Tensor,
    dO: cute.Tensor,
    lse_log2: cute.Tensor,
    dV: cute.Tensor,
    scale_softmax: cutlass.Float32,
    cumulative_s_q: cute.Tensor | None,
    cumulative_s_k: cute.Tensor | None,
    stream: cuda.CUstream = None,
  ):
    """Trace entry: layouts, TMA atoms, SharedStorage; launch the build."""
    # Both or neither: SeqlenInfoQK would silently accept one packed side.
    assert (cumulative_s_q is None) == (cumulative_s_k is None), (
      "varlen dV requires both cumulative_s_q and cumulative_s_k"
    )
    # Rank-4 dense / rank-3 packed in; builds (B,S,H_k,H_r,D) and seq views.
    varlen = cumulative_s_q is not None
    q_rank = cute.rank(Q.layout)
    # Rank 3 = packed; rank 4 = dense or the harness's (1, total, H, D) view.
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
    Q, K, dV, dO = [assume_tensor_aligned(t) for t in (Q, K, dV, dO)]
    Q = utils.as_bshkrd_tensor(Q, h_k_in, h_r_in, varlen)
    K = utils.as_bshkrd_tensor(K, h_k_in, 1, varlen)
    dV = utils.as_bshkrd_tensor(dV, h_k_in, 1, varlen)
    dO = utils.as_bshkrd_tensor(dO, h_k_in, h_r_in, varlen)
    scaled_LSE = utils.as_shhb_tensor(lse_log2, h_k_in, h_r_in, b_stats, varlen)
    h_r = Q.shape[3]
    h_k = Q.shape[2]
    if const_expr(cumulative_s_q is not None):
      b = cumulative_s_q.shape[0] - 1
    elif const_expr(cumulative_s_k is not None):
      b = cumulative_s_k.shape[0] - 1
    else:
      b = Q.shape[0]
    problem_shape = (
      Q.shape[1],
      K.shape[1],
      Q.shape[4],
      ((h_r, h_k), b),
    )
    hb = ((h_r, h_k), b)
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
    dV = cute.make_tensor(
      dV.iterator,
      cute.make_layout(
        (dV.shape[1], dV.shape[4], hb),
        stride=(
          cute.assume(dV.stride[1], divby=64),
          dV.stride[4],
          (
            (0, dV.stride[2]),
            0 if cumulative_s_k is not None else
            cute.assume(dV.stride[0], divby=64),
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

    # (s, d, ((h_r, h_k), b)) -> (d, s, ((h_r, h_k), b))
    dOT = cute.make_tensor(
      dO.iterator,
      cute.make_layout(
        (dO.shape[1], dO.shape[0], dO.shape[2]),
        stride=(
          dO.stride[1],
          dO.stride[0],
          dO.stride[2],
        ),
      ),
    )

    # Trace-time only: dtypes, major modes and layouts come from traced tensors.
    # Lowercase: these are the production attributes, not a copy.
    self.q_dtype = Q.element_type
    self.k_dtype = K.element_type
    self.do_dtype = dO.element_type
    self.q_major_mode = LayoutEnum.from_tensor(Q).mma_major_mode()
    self.k_major_mode = LayoutEnum.from_tensor(K).mma_major_mode()
    self.dv_major_mode = LayoutEnum.from_tensor(dV).mma_major_mode()
    self.do_major_mode = LayoutEnum.from_tensor(dO).mma_major_mode()
    # P's major mode and quantization are kernel properties, not input-derived.
    self.p_major_mode = tcgen05.OperandMajorMode.K

    if const_expr(self.q_major_mode != tcgen05.OperandMajorMode.K):
      raise RuntimeError(
        f"The layout of q is not supported: {self.q_major_mode}"
      )
    if const_expr(self.k_major_mode != tcgen05.OperandMajorMode.K):
      raise RuntimeError("The layout of k is not supported")
    if const_expr(self.dv_major_mode != tcgen05.OperandMajorMode.K):
      raise RuntimeError("The layout of dv is not supported")
    if const_expr(self.do_major_mode != tcgen05.OperandMajorMode.K):
      raise RuntimeError("The layout of do is not supported")

    # _setup_attributes ran in __init__ (the ledger check needs it); idempotent.
    KQ_tiled_mma, KQ_tmem_tiled_mma, PdO_tiled_mma = self._get_tiled_mma()

    atom_thr_size = cute.size(KQ_tiled_mma.thr_id.shape)
    self.cluster_layout_vmnk = cute.tiled_divide(
      cute.make_layout(self.cluster_shape_mnk),
      (atom_thr_size, ),
    )

    self.sK_layout = sm100_utils_basic.make_smem_layout_a(
      self.KQ_tiled_mma,
      self.KQ_mma_tiler,
      self.k_dtype,
      1,
    )
    self.sQ_layout = sm100_utils_basic.make_smem_layout_b(
      self.KQ_tiled_mma,
      self.KQ_mma_tiler,
      self.q_dtype,
      self.load_mma_Q_stage,
    )
    self.sP_layout = sm100_utils_basic.make_smem_layout_a(
      self.PdO_tiled_mma,
      self.PdO_mma_tiler,
      self.q_dtype,
      self.compute_mma_P_stage,
    )
    self.sdOt_layout = sm100_utils_basic.make_smem_layout_b(
      self.PdO_tiled_mma,
      self.PdO_mma_tiler,
      self.do_dtype,
      self.load_mma_dOT_stage * self.dv_slices,
    )
    self.LSE_smem_layout = cute.make_layout(
      (self.cta_tiler[0], self.load_compute_LSE_stage)
    )

    self.tma_load_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.TWO)

    # sK's first kq_tmem_k columns: swizzled s2t source, TMEM logical stripped.
    self.sKt_half_layout = sm100_utils_basic.make_smem_layout_a(
      self.KQ_tmem_tiled_mma,
      (self.KQ_mma_tiler[0], self.KQ_mma_tiler[1], self.kq_tmem_k),
      self.k_dtype,
      1,
    )
    sK_layout = self.sK_layout
    sQ_layout = self.sQ_layout
    sP_layout = self.sP_layout
    sdOt_layout = self.sdOt_layout
    LSE_smem_layout = self.LSE_smem_layout
    tma_load_op = self.tma_load_op
    sKt_half_layout = self.sKt_half_layout

    K_smem_layout = cute.select(sK_layout, mode=[0, 1, 2])
    tma_atom_K, tma_tensor_K = cute.nvgpu.make_tiled_tma_atom_A(
      tma_load_op,
      K,
      K_smem_layout,
      self.KQ_mma_tiler,
      KQ_tiled_mma,
      self.cluster_layout_vmnk.shape,
    )

    Q_smem_layout = cute.select(sQ_layout, mode=[0, 1, 2])
    tma_atom_Q, tma_tensor_Q = cute.nvgpu.make_tiled_tma_atom_B(
      tma_load_op,
      Q,
      Q_smem_layout,
      self.KQ_mma_tiler,
      KQ_tiled_mma,
      self.cluster_layout_vmnk.shape,
    )
    dOT_smem_layout = cute.select(sdOt_layout, mode=[0, 1, 2])
    tma_atom_dOT, tma_tensor_dOT = cute.nvgpu.make_tiled_tma_atom_B(
      tma_load_op,
      dOT,
      dOT_smem_layout,
      self.PdO_mma_tiler,
      PdO_tiled_mma,
      self.cluster_layout_vmnk.shape,
    )

    self.tma_copy_Q_bytes = cute.size_in_bytes(
      Q.element_type, Q_smem_layout
    ) * atom_thr_size
    self.tma_copy_K_bytes = cute.size_in_bytes(
      K.element_type, K_smem_layout
    ) * atom_thr_size
    # All dv_slices land on one barrier: tx must count every slice's bytes.
    self.tma_copy_dOT_bytes = (
      cute.size_in_bytes(dO.element_type, dOT_smem_layout) * atom_thr_size *
      self.dv_slices
    )

    tma_store_op = cpasync.CopyBulkTensorTileS2GOp()
    # Read from self in-kernel too: the same four numbers, not equal ones.
    (
      self.epi_num_warp_groups,
      self.epi_cols_dV,
      self.epi_tile_dV,
      self.epi_stages_dV,
    ) = self.dv_epilogue_tiling(dV.element_type.width)
    epi_tile_dV = self.epi_tile_dV
    total_epi_stages = self.epi_stages_dV
    dV_layout_enum = LayoutEnum.from_tensor(dV)
    sdV_epi_layout = sm100_utils_basic.make_smem_layout_epi(
      dV.element_type,
      dV_layout_enum,
      epi_tile_dV,
      total_epi_stages,
    )
    tma_atom_dV, tma_tensor_dV = cpasync.make_tiled_tma_atom(
      tma_store_op,
      dV,
      cute.select(sdV_epi_layout, mode=[0, 1]),
      epi_tile_dV,
    )

    @cute.struct
    class SharedStorage:
      load_mma_Q_mbar_ptr: cute.struct.MemRange[cutlass.Int64,
                                                self.load_mma_Q_stage * 2]
      load_mma_K_mbar_ptr: cute.struct.MemRange[cutlass.Int64,
                                                self.load_mma_K_stage * 2]
      load_mma_dOT_mbar_ptr: cute.struct.MemRange[cutlass.Int64,
                                                  self.load_mma_dOT_stage * 2]
      load_compute_lse_mbar_ptr: cute.struct.MemRange[
        cutlass.Int64, self.load_compute_LSE_stage * 2]
      mma_compute_S_mbar_ptr: cute.struct.MemRange[cutlass.Int64,
                                                   self.mma_compute_S_stage * 2]
      compute_mma_P_mbar_ptr: cute.struct.MemRange[cutlass.Int64,
                                                   self.compute_mma_P_stage * 2]
      mma_compute_dV_mbar_ptr: cute.struct.MemRange[cutlass.Int64,
                                                    self.mma_compute_dV_stage *
                                                    2]
      tmem_holding_buf: cutlass.Int32
      tmem_dealloc_mbar: cutlass.Int64
      sK: cute.struct.Align[
        cute.struct.MemRange[K.element_type,
                             cute.cosize(sK_layout)],
        self.buffer_align_bytes,
      ]
      sQ: cute.struct.Align[
        cute.struct.MemRange[Q.element_type,
                             cute.cosize(sQ_layout)],
        self.buffer_align_bytes,
      ]
      # PdO's B operand: its own buffer; the K-major dO face died with dP.
      sdOT: cute.struct.Align[
        cute.struct.MemRange[dO.element_type,
                             cute.cosize(sdOt_layout)],
        self.buffer_align_bytes,
      ]
      # The A operand of the PdO MMA: P quantized to the dV element type.
      sP: cute.struct.Align[
        cute.struct.MemRange[Q.element_type,
                             cute.cosize(sP_layout)],
        self.buffer_align_bytes,
      ]
      # Donor staged onto sdOT: short at D512, so sK hosts the staging.
      s_epi_dV: cute.struct.Align[
        cute.struct.MemRange[
          dV.element_type,
          1 if self.alias_epilogue_onto_sK else cute.cosize(sdV_epi_layout),
        ],
        self.buffer_align_bytes,
      ]
      sLSE: cute.struct.Align[
        cute.struct.MemRange[self.acc_dtype,
                             cute.cosize(LSE_smem_layout)],
        self.buffer_align_bytes,
      ]

    self.shared_storage = SharedStorage
    # The one executable SMEM-budget check; header prose is not a witness.
    assert SharedStorage.size_in_bytes() <= SM100_SMEM_CAPACITY_BYTES, (
      f"SharedStorage {SharedStorage.size_in_bytes()} B > SM100 opt-in "
      f"ceiling {SM100_SMEM_CAPACITY_BYTES} B"
    )

    bwd_grid = self._compute_bwd_grid(problem_shape, self.cta_tiler[1])
    bwd_grid = cute.round_up(bwd_grid, self.cluster_shape_mnk)

    # One entry: a host `if self.is_causal:` here traced BOTH builds per module.
    self.kernel(
      KQ_tiled_mma,
      KQ_tmem_tiled_mma,
      PdO_tiled_mma,
      tma_atom_K,
      tma_tensor_K,
      K,
      tma_atom_Q,
      tma_tensor_Q,
      Q,
      tma_atom_dOT,
      tma_tensor_dOT,
      dV,
      tma_atom_dV,
      tma_tensor_dV,
      scaled_LSE,
      scale_softmax,
      problem_shape,
      cumulative_s_q,
      cumulative_s_k,
      self.cluster_layout_vmnk,
      sK_layout,
      sKt_half_layout,
      sQ_layout,
      sdOt_layout,
      sP_layout,
      LSE_smem_layout,
      sdV_epi_layout,
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
    KQ_tmem_tiled_mma: cute.TiledMma,
    PdO_tiled_mma: cute.TiledMma,
    tma_atom_K: cute.CopyAtom,
    mK: cute.Tensor,
    # *_ref: the raw (non-TMA) tensor, read for shape/dtype only.
    mK_ref: cute.Tensor,
    tma_atom_Q: cute.CopyAtom,
    mQ: cute.Tensor,
    mQ_ref: cute.Tensor,
    tma_atom_dOT: cute.CopyAtom,
    mdOT: cute.Tensor,
    mdV: cute.Tensor,
    tma_atom_dV: cute.CopyAtom,
    mdV_tma: cute.Tensor,
    mLSE: cute.Tensor,
    scale_softmax: cutlass.Float32,
    problem_shape: tuple[Int32, Int32, Int32, tuple[tuple[Int32, Int32],
                                                    Int32]],
    cumulative_s_q: cute.Tensor | None,
    cumulative_s_k: cute.Tensor | None,
    cluster_layout_vmnk: cute.Layout,
    sK_layout: cute.ComposedLayout,
    sKt_half_layout: cute.ComposedLayout,
    sQ_layout: cute.ComposedLayout,
    sdOt_layout: cute.ComposedLayout,
    sP_layout: cute.ComposedLayout,
    LSE_smem_layout: cute.Layout,
    sdV_epi_layout: cute.ComposedLayout,
  ):
    """Kernel body: role dispatch by warp index."""
    bidx, bidy, bidz = cute.arch.block_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    # Dispatch bounds of the multi-warp role.
    compute_lo = self.compute_warp_id[0]
    compute_hi = self.compute_warp_id[-1]
    varlen = cumulative_s_q is not None or cumulative_s_k is not None

    if warp_idx == self.load_warp_id:
      cpasync.prefetch_descriptor(tma_atom_K)
      cpasync.prefetch_descriptor(tma_atom_Q)
      if const_expr(not self.split_dot_issuer):
        cpasync.prefetch_descriptor(tma_atom_dOT)
    if warp_idx == self.dot_load_warp_id:  # noqa: SIM102
      # Nested on purpose: folding into `and` traces the body on every warp.
      if const_expr(self.split_dot_issuer):
        cpasync.prefetch_descriptor(tma_atom_dOT)

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(self.shared_storage)

    # TMEM alloc first hides the write-back; neither call moves a sync.
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

    (
      load_mma_Q_producer,
      load_mma_Q_consumer,
      load_mma_K_producer,
      load_mma_K_consumer,
      load_mma_dOT_producer,
      load_mma_dOT_consumer,
      load_compute_LSE_producer,
      load_compute_LSE_consumer,
      mma_compute_S_producer,
      mma_compute_S_consumer,
      compute_mma_P_producer,
      compute_mma_P_consumer,
      mma_compute_dV_producer,
      mma_compute_dV_consumer,
    ) = self.make_pipelines(storage, cluster_layout_vmnk)

    cute.arch.barrier(
      barrier_id=self.cta_sync_bar_id, number_of_threads=self.threads_per_cta
    )

    if const_expr(self.is_causal):
      # Causal builds every consumer view once here; non-causal per role
      # (measured).  c_* names: the role arms rebind them without a yield.
      c_sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
      c_sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
      c_sLSE = storage.sLSE.get_tensor(LSE_smem_layout)
      c_s_epi_dV = self.make_epilogue_staging(
        storage, c_sK, sdV_epi_layout, mdV.element_type
      )

      c_sP = storage.sP.get_tensor(sP_layout.outer, swizzle=sP_layout.inner)

      c_sdOT = storage.sdOT.get_tensor(
        sdOt_layout.outer, swizzle=sdOt_layout.inner
      )

      c_tSTrK = KQ_tiled_mma.make_fragment_A(c_sK)
      c_tSTrQ = KQ_tiled_mma.make_fragment_B(c_sQ)

      tmem.wait_for_alloc()
      tmem_ptr_common = tmem.retrieve_ptr(self.acc_dtype)

      c_tSTrK_tmem, c_tSTrQ_tmem, c_tiled_s2t, c_tKsK_s2t, c_tKtK_s2t, (
        c_tKT_addr
      ) = self.make_KT_operand(
        storage, tmem_ptr_common, c_sQ, sKt_half_layout, KQ_tmem_tiled_mma
      )

    # Cluster arrive after barrier init; is_relaxed=False keeps consistency.
    # Non-causal arrives before any view (the KT chain delayed K's TMA).
    pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=False)

    if const_expr(self.is_causal):
      c_tSTtST_stages = self.make_S_stages(tmem_ptr_common, KQ_tiled_mma)

      c_tdVrP = PdO_tiled_mma.make_fragment_A(c_sP)
      c_tdVrdOT = PdO_tiled_mma.make_fragment_B(c_sdOT)

      c_tdVtdV_slices = self.make_dV_slices(tmem_ptr_common, PdO_tiled_mma)

    blk_coord = (Int32(0), bidx, Int32(0), ((Int32(0), bidy), bidz))
    # One SeqlenInfo per role (a shared one spills); tile_m divides LSE offsets.
    SeqlenInfoCls = partial(
      SeqlenInfoQK.create,
      seqlen_q_static=mQ_ref.shape[0],
      seqlen_k_static=mK_ref.shape[0],
      mCuSeqlensQ=cumulative_s_q,
      mCuSeqlensK=cumulative_s_k,
      tile_m=self.tile_shape_Q,
      tile_n=self.tile_shape_K * self.cluster_shape_mnk[0],
    )
    seqlen = SeqlenInfoCls(bidz)

    iter_start, iter_end = self.get_Q_block_min_max(seqlen, blk_coord[1])

    pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

    iter_count = (iter_end - iter_start) * problem_shape[3][0][0]
    problem_shape_cur_batch = (
      seqlen.seqlen_q,
      seqlen.seqlen_k,
      problem_shape[2],
      problem_shape[3],
    )
    if iter_count <= 0:
      if bidx * self.tile_shape_K < seqlen.seqlen_k:
        self.epilogue_clear(
          blk_coord,
          SeqlenInfoCls,
          problem_shape_cur_batch,
          mdV,
        )
    # ///  LOAD  ///
    elif warp_idx == self.load_warp_id:
      cute.arch.setmaxregister_decrease(self.num_regs_load)
      if const_expr(self.is_causal):
        sK, sQ, sdOT, sLSE = c_sK, c_sQ, c_sdOT, c_sLSE
      else:
        # Arrive only: the alloc was issued at the frame top; no wait needed.
        tmem_alloc_barrier.arrive()
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sdOT = storage.sdOT.get_tensor(
          sdOt_layout.outer, swizzle=sdOt_layout.inner
        )
        sLSE = storage.sLSE.get_tensor(LSE_smem_layout)

      # Load-bearing const_expr: merged forms trip MLIR dominance on dOT state.
      if const_expr(not self.split_dot_issuer):
        self.load(
          mK,
          mQ,
          mdOT,
          mLSE,
          sK,
          sQ,
          sdOT,
          sLSE,
          KQ_tiled_mma,
          PdO_tiled_mma,
          tma_atom_K,
          tma_atom_Q,
          tma_atom_dOT,
          SeqlenInfoCls,
          problem_shape_cur_batch,
          iter_count,
          iter_start,
          iter_end,
          load_mma_Q_producer,
          load_mma_K_producer,
          load_compute_LSE_producer,
          load_mma_dOT_producer,
        )
      else:
        self.load_kq_lse(
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
        )

    # ///  dOT LOAD (separate issuer)  ///
    elif warp_idx == self.dot_load_warp_id:
      if const_expr(self.split_dot_issuer):
        cute.arch.setmaxregister_decrease(self.num_regs_load)
        # Only the per-role build arrives; the causal one waited above.
        if const_expr(not self.is_causal):
          tmem_alloc_barrier.arrive()
          sdOT = storage.sdOT.get_tensor(
            sdOt_layout.outer,
            swizzle=sdOt_layout.inner,
          )
        else:
          sdOT = c_sdOT

        self.load_dOT(
          mdOT,
          sdOT,
          PdO_tiled_mma,
          tma_atom_dOT,
          SeqlenInfoCls,
          iter_count,
          iter_start,
          iter_end,
          load_mma_dOT_producer,
        )
      else:
        cute.arch.setmaxregister_decrease(self.num_regs_empty)
        if const_expr(not self.is_causal):
          tmem_alloc_barrier.arrive()

    # ///  MMA  ///
    elif warp_idx == self.mma_warp_id:
      cute.arch.setmaxregister_increase(self.num_regs_mma)

      if const_expr(self.is_causal):
        sQ, tSTrK, tSTrQ = c_sQ, c_tSTrK, c_tSTrQ
        tSTrK_tmem, tSTrQ_tmem, tiled_s2t, tKsK_s2t, tKtK_s2t, tKT_addr = (
          c_tSTrK_tmem,
          c_tSTrQ_tmem,
          c_tiled_s2t,
          c_tKsK_s2t,
          c_tKtK_s2t,
          c_tKT_addr,
        )
        tSTtST_stages, tdVrP, tdVrdOT, tdVtdV_slices = (
          c_tSTtST_stages,
          c_tdVrP,
          c_tdVrdOT,
          c_tdVtdV_slices,
        )
      else:
        # Consumer-side setup, moved out of the common path.
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sP = storage.sP.get_tensor(sP_layout.outer, swizzle=sP_layout.inner)
        sdOT = storage.sdOT.get_tensor(
          sdOt_layout.outer, swizzle=sdOt_layout.inner
        )
        tSTrK = KQ_tiled_mma.make_fragment_A(sK)
        tSTrQ = KQ_tiled_mma.make_fragment_B(sQ)
        tSTrK_tmem, tSTrQ_tmem, tiled_s2t, tKsK_s2t, tKtK_s2t, tKT_addr = (
          self.make_KT_operand(
            storage, tmem_ptr, sQ, sKt_half_layout, KQ_tmem_tiled_mma
          )
        )
        tSTtST_stages = self.make_S_stages(tmem_ptr, KQ_tiled_mma)
        tdVrP = PdO_tiled_mma.make_fragment_A(sP)
        tdVrdOT = PdO_tiled_mma.make_fragment_B(sdOT)
        tdVtdV_slices = self.make_dV_slices(tmem_ptr, PdO_tiled_mma)

      self.mma(
        KQ_tiled_mma,
        KQ_tmem_tiled_mma,
        PdO_tiled_mma,
        tSTtST_stages,
        tSTrQ,
        tSTrK,
        tSTrQ_tmem,
        tSTrK_tmem,
        tiled_s2t,
        tKsK_s2t,
        tKtK_s2t,
        tKT_addr,
        sQ,
        tdVtdV_slices,
        tdVrP,
        tdVrdOT,
        iter_count,
        load_mma_Q_consumer,
        load_mma_K_consumer,
        mma_compute_S_producer,
        load_mma_dOT_consumer,
        compute_mma_P_consumer,
        mma_compute_dV_producer,
      )

    # ///  Compute  ///
    elif warp_idx >= compute_lo and warp_idx <= compute_hi:
      cute.arch.setmaxregister_increase(self.num_regs_compute)

      if const_expr(self.is_causal):
        tSTtST_stages, sP, sLSE, tdVtdV_slices, s_epi_dV = (
          c_tSTtST_stages,
          c_sP,
          c_sLSE,
          c_tdVtdV_slices,
          c_s_epi_dV,
        )
      else:
        # Consumer-side setup, moved out of the common path.
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        s_epi_dV = self.make_epilogue_staging(
          storage, sK, sdV_epi_layout, mdV.element_type
        )
        sP = storage.sP.get_tensor(sP_layout.outer, swizzle=sP_layout.inner)
        sLSE = storage.sLSE.get_tensor(LSE_smem_layout)
        tSTtST_stages = self.make_S_stages(tmem_ptr, KQ_tiled_mma)
        tdVrP = PdO_tiled_mma.make_fragment_A(sP)
        tdVtdV_slices = self.make_dV_slices(tmem_ptr, PdO_tiled_mma)

      self.compute_loop(
        tSTtST_stages,
        sP,
        sLSE,
        mdV,
        tdVtdV_slices,
        blk_coord,
        SeqlenInfoCls,
        seqlen,
        problem_shape_cur_batch,
        iter_count,
        iter_start,
        iter_end,
        scale_softmax,
        mma_compute_S_consumer,
        compute_mma_P_producer,
        load_compute_LSE_consumer,
        mma_compute_dV_consumer,
        varlen,
        seqlen.seqlen_k,
        tma_atom_dV,
        mdV_tma,
        s_epi_dV,
      )

      cute.arch.barrier(
        barrier_id=self.epilogue_sync_bar_id,
        number_of_threads=self.num_compute_warps * self.threads_per_warp,
      )

    else:
      cute.arch.setmaxregister_decrease(self.num_regs_empty)
      # Same asymmetry as the dOT arm: only the per-role build arrives.
      if const_expr(not self.is_causal):
        tmem_alloc_barrier.arrive()

    cute.arch.cluster_arrive()
    cute.arch.cluster_wait()
    if const_expr(self.is_causal):
      tmem_ptr_exit = tmem_ptr_common
    else:
      # tmem_ptr was branch-local: re-retrieve; CTA + pair barriers order it.
      tmem_ptr_exit = tmem.retrieve_ptr(self.acc_dtype)
    tmem.relinquish_alloc_permit()
    tmem.free(tmem_ptr_exit)
    # Last: the CTA must not exit before async reads of the sK staging finish.
    cute.arch.cp_async_bulk_wait_group(0, read=True)

  @cute.jit
  def make_KT_operand(
    self, storage, tmem_ptr, sQ, sKt_half_layout, KQ_tmem_tiled_mma
  ):
    """TMEM-A view of K^T plus the s2t copy that parks K there."""
    tKT = cute.make_tensor(
      cute.recast_ptr(
        tmem_ptr + self.tmem_KT_base,
        dtype=self.k_dtype,
      ),
      sKt_half_layout.outer,
    )
    # _tmem names the TMEM-A edge, not storage: *_tmem B fragments stay SMEM.
    tSTrK_tmem = KQ_tmem_tiled_mma.make_fragment_A(tKT)
    tSTrQ_tmem = KQ_tmem_tiled_mma.make_fragment_B(sQ)
    sK_lo = storage.sK.get_tensor(
      sKt_half_layout.outer,
      swizzle=sKt_half_layout.inner,
    )
    # Dst must carry physical TMEM addressing: the MMA fragment serves both.
    tKT_compact = cute.filter_zeros(tSTrK_tmem)
    # Only 0213: row r -> lane r%64 twice; 128x256b / 0123 fake a cos ~ 0.98.
    s2t_atom = cute.make_copy_atom(
      tcgen05.copy.Cp2x64x128b0213Op(tcgen05.CtaGroup.TWO), self.k_dtype
    )
    tiled_s2t = tcgen05.make_s2t_copy(s2t_atom, tKT_compact)
    thr_s2t = tiled_s2t.get_slice(0)
    sw128_row_elems = 128 // (self.k_dtype.width // 8)
    panel_elems = self.tile_shape_K * sw128_row_elems
    # Modes: ((row, elem in 16 B), 1:0, (chunk pair, quad, 64-col panel), 1:0).
    sK_lo_128b = cute.make_tensor(
      sK_lo.iterator,
      cute.make_layout(
        ((64, 8), 1, (2, 4, self.kq_tmem_k // sw128_row_elems), 1),
        stride=((sw128_row_elems, 1), 0, (8, 16, panel_elems), 0),
      ),
    )
    tKsK_s2t = tcgen05.get_s2t_smem_desc_tensor(
      tiled_s2t, thr_s2t.partition_S(cute.filter_zeros(sK_lo_128b))
    )
    tKtK_s2t = thr_s2t.partition_D(tKT_compact)
    # Copy twin: rebuild dst AND tKT_addr; a miss reads f32 as f16 at step 2-3.
    tKtK_s2t = cute.make_tensor(tKT.iterator, tKtK_s2t.layout)
    tKT_addr = tKT.iterator.toint()
    return tSTrK_tmem, tSTrQ_tmem, tiled_s2t, tKsK_s2t, tKtK_s2t, tKT_addr

  @cute.jit
  def make_S_stages(self, tmem_ptr, KQ_tiled_mma: cute.TiledMma):
    """One region per stage (race note in __init__); stage index is runtime."""
    tSTtST_shape = KQ_tiled_mma.partition_shape_C(
      cute.select(self.KQ_mma_tiler, mode=[0, 1])
    )
    tSTtST_layout = KQ_tiled_mma.make_fragment_C(tSTtST_shape).layout
    return [
      cute.make_tensor(
        tmem_ptr + self.tmem_S_base + index * self.tmem_S_slice_stride,
        tSTtST_layout,
      ) for index in range(self.mma_compute_S_stage)
    ]

  @cute.jit
  def make_dV_slices(self, tmem_ptr, PdO_tiled_mma: cute.TiledMma):
    """One accumulator per slice; wider geometry changes count not structure."""
    tdVtdV_shape = PdO_tiled_mma.partition_shape_C(
      cute.select(self.PdO_mma_tiler, mode=[0, 1])
    )
    tdVtdV_layout = PdO_tiled_mma.make_fragment_C(tdVtdV_shape).layout
    return [
      cute.make_tensor(
        tmem_ptr + self.tmem_dV_base + index * self.tmem_dV_slice_stride,
        tdVtdV_layout,
      ) for index in range(self.dv_slices)
    ]

  @cute.jit
  def make_epilogue_staging(self, storage, sK, sdV_epi_layout, epi_dtype):
    """The dV epilogue arena, aliased onto sK."""
    # sK passed in: the alias starts at exactly the caller's iterator.
    if const_expr(self.alias_epilogue_onto_sK):
      # sK is dead here: one K issue per tile, K released before the dV token.
      return cute.make_tensor(
        cute.recast_ptr(sK.iterator, sdV_epi_layout.inner, epi_dtype),
        sdV_epi_layout.outer,
      )
    else:
      return storage.s_epi_dV.get_tensor(
        sdV_epi_layout.outer, swizzle=sdV_epi_layout.inner
      )

  @cute.jit
  def make_pipelines(self, storage, cluster_layout_vmnk: cute.Layout):
    """Construct every mbarrier pipeline of the kernel."""
    # No alloc, no sync, no handle mutation: a mutation leaves the carried set.
    # TMA issuers and the MMA warp arrive through one elected thread each.
    tma_producer_group = CooperativeGroup(Agent.Thread, 1)
    mma_group = CooperativeGroup(Agent.Thread, 1)
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
    load_mma_dOT_producer, load_mma_dOT_consumer = pipeline.PipelineTmaUmma.create(
      num_stages=self.load_mma_dOT_stage,
      producer_group=tma_producer_group,
      consumer_group=mma_group,
      tx_count=self.tma_copy_dOT_bytes,
      barrier_storage=storage.load_mma_dOT_mbar_ptr.data_ptr(),
      cta_layout_vmnk=cluster_layout_vmnk,
      defer_sync=True,
    ).make_participants()
    load_compute_LSE_producer, load_compute_LSE_consumer = pipeline.PipelineCpAsync.create(
      num_stages=self.load_compute_LSE_stage,
      producer_group=CooperativeGroup(Agent.Thread, self.threads_per_warp),
      consumer_group=CooperativeGroup(
        Agent.Thread, self.threads_per_warp * self.num_compute_warps
      ),
      barrier_storage=storage.load_compute_lse_mbar_ptr.data_ptr(),
    ).make_participants()
    mma_compute_S_producer, mma_compute_S_consumer = pipeline.PipelineUmmaAsync.create(
      num_stages=self.mma_compute_S_stage,
      producer_group=mma_group,
      consumer_group=CooperativeGroup(
        Agent.Thread, self.num_compute_warps * self.threads_per_warp *
        cluster_layout_vmnk.shape[0][0]
      ),
      barrier_storage=storage.mma_compute_S_mbar_ptr.data_ptr(),
      cta_layout_vmnk=cluster_layout_vmnk,
      defer_sync=True,
    ).make_participants()
    compute_mma_P_producer, compute_mma_P_consumer = pipeline.PipelineAsyncUmma.create(
      num_stages=self.compute_mma_P_stage,
      producer_group=CooperativeGroup(
        Agent.Thread, self.num_compute_warps * self.threads_per_warp *
        cluster_layout_vmnk.shape[0][0]
      ),
      consumer_group=mma_group,
      barrier_storage=storage.compute_mma_P_mbar_ptr.data_ptr(),
      cta_layout_vmnk=cluster_layout_vmnk,
      defer_sync=True,
    ).make_participants()
    mma_compute_dV_producer, mma_compute_dV_consumer = pipeline.PipelineUmmaAsync.create(
      num_stages=self.mma_compute_dV_stage,
      producer_group=mma_group,
      consumer_group=CooperativeGroup(
        Agent.Thread, self.num_compute_warps * self.threads_per_warp *
        cluster_layout_vmnk.shape[0][0]
      ),
      barrier_storage=storage.mma_compute_dV_mbar_ptr.data_ptr(),
      cta_layout_vmnk=cluster_layout_vmnk,
      defer_sync=True,
    ).make_participants()
    return (
      load_mma_Q_producer,
      load_mma_Q_consumer,
      load_mma_K_producer,
      load_mma_K_consumer,
      load_mma_dOT_producer,
      load_mma_dOT_consumer,
      load_compute_LSE_producer,
      load_compute_LSE_consumer,
      mma_compute_S_producer,
      mma_compute_S_consumer,
      compute_mma_P_producer,
      compute_mma_P_consumer,
      mma_compute_dV_producer,
      mma_compute_dV_consumer,
    )

  @cute.jit
  def get_Q_block_min_max(
    self,
    seqlen: SeqlenInfoQK,
    blk_coord_k: Int32,
  ):
    """Q-block trip range of this pair tile (BlockInfo)."""
    # tile_n covers the pair and blk_coord_k//2 is uniform: unequal iters hang.
    block_info = BlockInfo(
      self.tile_shape_Q,
      self.tile_shape_K * self.cluster_shape_mnk[0],
      self.is_causal,
      self.has_sliding_window,
      self.window_size_left,
      self.window_size_right,
      qhead_per_kvhead_packgqa=1,
    )
    # No Q_block_min parity rounding: the per-tile diagonal test replaced it.
    return block_info.get_m_block_min_max(
      seqlen, blk_coord_k // self.cluster_shape_mnk[0]
    )

  @cute.jit
  def load_lse_stage(
    self,
    LSE_for_copy: cute.Tensor,
    sLSE_for_copy: cute.Tensor,
    lse_handle,
    iter_index: Int32,
    blk_coord_h,
    blk_coord_b: Int32,
    seqlen_q: Int32,
    thread_idx: Int32,
    async_copy_num_elts: int,
    atom_async_copy: cute.CopyAtom,
  ):
    """One LSE stage by cp.async; rows past seqlen_q are zero-filled."""
    # Warp-coalesced: lane T reads index T + i*W, stride-1 across the warp.
    for i in cutlass.range_constexpr(async_copy_num_elts):
      LSE_idx = (
        self.tile_shape_Q * iter_index + thread_idx + i * self.threads_per_warp
      )
      sLSE_idx = thread_idx + i * self.threads_per_warp
      if cute.elem_less(LSE_idx, seqlen_q):
        cute.copy(
          atom_async_copy,
          LSE_for_copy[None, LSE_idx, (blk_coord_h, blk_coord_b)],
          sLSE_for_copy[None, sLSE_idx, lse_handle.index],
        )
      else:
        sLSE_for_copy[None, sLSE_idx, lse_handle.index].fill(0.0)

  @cute.jit
  def load_kq_lse(
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
  ):
    """Load warp when dO^T has its own issuer (non-causal build): K, Q, LSE."""
    tidx, _, _ = cute.arch.thread_idx()
    blk_coord_k, blk_coord_h_k, blk_coord_b = cute.arch.block_idx()
    blk_coord_h_r = Int32(0)
    blk_coord_h = (blk_coord_h_r, blk_coord_h_k)
    iter_index = iter_start
    mma_tile_coord_v = blk_coord_k % cute.size(KQ_tiled_mma.thr_id.shape)
    mma_tile_coord_m = blk_coord_k // cute.size(KQ_tiled_mma.thr_id.shape)

    # K first: only K's partition precedes its TMA; Q/LSE run under the flight.
    seqlen = SeqlenInfoCls(blk_coord_b)
    hb_origin = ((Int32(0), Int32(0)), Int32(0))
    K = cute.domain_offset((seqlen.offset_k, Int32(0), hb_origin), mK)
    gK = cute.local_tile(
      K, cute.select(self.KQ_mma_tiler, mode=[0, 2]), (None, None, None)
    )
    KQ_thr_mma = KQ_tiled_mma.get_slice(mma_tile_coord_v)
    tSTgK = KQ_thr_mma.partition_A(gK)
    cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)
    cta_layout_vmnk = cute.tiled_divide(cta_layout_mnk, (KQ_tiled_mma.thr_id, ))
    cta_in_cluster_coord_vmnk = cta_layout_vmnk.get_flat_coord(
      cute.arch.block_idx_in_cluster()
    )
    tKsK, tKgK_mkl = cute.nvgpu.cpasync.tma_partition(
      tma_atom_K,
      cta_in_cluster_coord_vmnk[2],
      cute.make_layout(cute.size(cta_layout_vmnk, mode=[2])),
      cute.group_modes(sK, 0, 3),
      cute.group_modes(tSTgK, 0, 3),
    )
    k_handle = load_mma_K_producer.acquire_and_advance()
    cute.copy(
      tma_atom_K,
      tKgK_mkl[(None, mma_tile_coord_m, 0, (blk_coord_h, blk_coord_b))],
      tKsK[None, 0],
      tma_bar_ptr=k_handle.barrier,
    )

    Q = cute.domain_offset((seqlen.offset_q, Int32(0), hb_origin), mQ)
    # LSE is stored per padded Q tile, so it takes the padded offset.
    LSE = cute.domain_offset((seqlen.padded_offset_q, hb_origin), mLSE)

    gQ = cute.local_tile(
      Q, cute.select(self.KQ_mma_tiler, mode=[1, 2]), (None, None, None)
    )
    tSTgQ = KQ_thr_mma.partition_B(gQ)
    tQsQ, tQgQ_mkl = cute.nvgpu.cpasync.tma_partition(
      tma_atom_Q,
      cta_in_cluster_coord_vmnk[1],
      cute.make_layout(cute.size(cta_layout_vmnk, mode=[1])),
      cute.group_modes(sQ, 0, 3),
      cute.group_modes(tSTgQ, 0, 3),
    )

    q_handle = load_mma_Q_producer.acquire_and_advance()
    cute.copy(
      tma_atom_Q,
      tQgQ_mkl[(None, iter_index, 0, (blk_coord_h, blk_coord_b))],
      tQsQ[None, q_handle.index],
      tma_bar_ptr=q_handle.barrier,
    )

    lse_handle = load_compute_LSE_producer.acquire_and_advance()
    thread_idx = tidx % self.threads_per_warp
    async_copy_num_elts = sLSE.shape[0] // self.threads_per_warp
    atom_async_copy = cute.make_copy_atom(
      cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
      self.acc_dtype,
      num_bits_per_copy=self.acc_dtype.width,
    )
    sLSE_for_copy = cute.flat_divide(sLSE, (1, ))
    LSE_for_copy = cute.flat_divide(LSE, (1, ))
    self.load_lse_stage(
      LSE_for_copy,
      sLSE_for_copy,
      lse_handle,
      iter_index,
      blk_coord_h,
      blk_coord_b,
      problem_shape[0],
      thread_idx,
      async_copy_num_elts,
      atom_async_copy,
    )
    lse_handle.commit()

    iter_count -= 1
    iter_index += 1

    while iter_count > 0:
      if iter_index == iter_end:
        iter_index = iter_start
        blk_coord_h_r += 1
        blk_coord_h = (blk_coord_h_r, blk_coord_h_k)

      q_handle = load_mma_Q_producer.acquire_and_advance()
      cute.copy(
        tma_atom_Q,
        tQgQ_mkl[(None, iter_index, 0, (blk_coord_h, blk_coord_b))],
        tQsQ[None, q_handle.index],
        tma_bar_ptr=q_handle.barrier,
      )

      lse_handle = load_compute_LSE_producer.acquire_and_advance()
      self.load_lse_stage(
        LSE_for_copy,
        sLSE_for_copy,
        lse_handle,
        iter_index,
        blk_coord_h,
        blk_coord_b,
        problem_shape[0],
        thread_idx,
        async_copy_num_elts,
        atom_async_copy,
      )
      lse_handle.commit()

      iter_count -= 1
      iter_index += 1

    # Every surviving producer needs its tail, else its consumer deadlocks.
    load_mma_K_producer.tail()
    load_mma_Q_producer.tail()
    load_compute_LSE_producer.tail()

  @cute.jit
  def load(
    self,
    mK: cute.Tensor,
    mQ: cute.Tensor,
    mdOT: cute.Tensor,
    mLSE: cute.Tensor,
    sK: cute.Tensor,
    sQ: cute.Tensor,
    sdOT: cute.Tensor,
    sLSE: cute.Tensor,
    KQ_tiled_mma: cute.TiledMma,
    PdO_tiled_mma: cute.TiledMma,
    tma_atom_K: cute.CopyAtom,
    tma_atom_Q: cute.CopyAtom,
    tma_atom_dOT: cute.CopyAtom,
    SeqlenInfoCls: Callable,
    problem_shape: tuple[Int32, Int32, Int32, tuple[tuple[Int32, Int32],
                                                    Int32]],
    iter_count: Int32,
    iter_start: Int32,
    iter_end: Int32,
    load_mma_Q_producer,
    load_mma_K_producer,
    load_compute_LSE_producer,
    load_mma_dOT_producer,
  ):
    """Single load warp (causal build): K, Q, LSE and dO^T."""
    tidx, _, _ = cute.arch.thread_idx()
    blk_coord_k, blk_coord_h_k, blk_coord_b = cute.arch.block_idx()
    blk_coord_h_r = Int32(0)
    blk_coord_h = (blk_coord_h_r, blk_coord_h_k)
    iter_index = iter_start
    mma_tile_coord_v = blk_coord_k % cute.size(KQ_tiled_mma.thr_id.shape)
    mma_tile_coord_m = blk_coord_k // cute.size(KQ_tiled_mma.thr_id.shape)

    seqlen = SeqlenInfoCls(blk_coord_b)
    hb_origin = ((Int32(0), Int32(0)), Int32(0))
    K = cute.domain_offset((seqlen.offset_k, Int32(0), hb_origin), mK)
    Q = cute.domain_offset((seqlen.offset_q, Int32(0), hb_origin), mQ)
    # dOT is the transposed operand: its sequence mode is 1, not 0.
    dOT = cute.domain_offset((Int32(0), seqlen.offset_q, hb_origin), mdOT)
    # LSE is stored per padded Q tile, so it takes the padded offset.
    LSE = cute.domain_offset((seqlen.padded_offset_q, hb_origin), mLSE)

    gK = cute.local_tile(
      K, cute.select(self.KQ_mma_tiler, mode=[0, 2]), (None, None, None)
    )
    gQ = cute.local_tile(
      Q, cute.select(self.KQ_mma_tiler, mode=[1, 2]), (None, None, None)
    )
    gdOT = cute.local_tile(
      dOT, cute.select(self.PdO_mma_tiler, mode=[1, 2]), (None, None, None)
    )

    KQ_thr_mma = KQ_tiled_mma.get_slice(mma_tile_coord_v)
    PdO_thr_mma = PdO_tiled_mma.get_slice(mma_tile_coord_v)

    tSTgK = KQ_thr_mma.partition_A(gK)
    tSTgQ = KQ_thr_mma.partition_B(gQ)
    tdVgdOT = PdO_thr_mma.partition_B(gdOT)

    cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)
    cta_layout_vmnk = cute.tiled_divide(cta_layout_mnk, (KQ_tiled_mma.thr_id, ))
    cta_in_cluster_coord_vmnk = cta_layout_vmnk.get_flat_coord(
      cute.arch.block_idx_in_cluster()
    )

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
    tdOTsdOT, tdOTgdOT_mkl = cute.nvgpu.cpasync.tma_partition(
      tma_atom_dOT,
      cta_in_cluster_coord_vmnk[1],
      cute.make_layout(cute.size(cta_layout_vmnk, mode=[1])),
      cute.group_modes(sdOT, 0, 3),
      cute.group_modes(tdVgdOT, 0, 3),
    )

    k_handle = load_mma_K_producer.acquire_and_advance()
    cute.copy(
      tma_atom_K,
      tKgK_mkl[(None, mma_tile_coord_m, 0, (blk_coord_h, blk_coord_b))],
      tKsK[None, 0],
      tma_bar_ptr=k_handle.barrier,
    )

    q_handle = load_mma_Q_producer.acquire_and_advance()
    cute.copy(
      tma_atom_Q,
      tQgQ_mkl[(None, iter_index, 0, (blk_coord_h, blk_coord_b))],
      tQsQ[None, q_handle.index],
      tma_bar_ptr=q_handle.barrier,
    )

    lse_handle = load_compute_LSE_producer.acquire_and_advance()
    thread_idx = tidx % self.threads_per_warp
    async_copy_num_elts = sLSE.shape[0] // self.threads_per_warp
    atom_async_copy = cute.make_copy_atom(
      cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
      self.acc_dtype,
      num_bits_per_copy=self.acc_dtype.width,
    )
    sLSE_for_copy = cute.flat_divide(sLSE, (1, ))
    LSE_for_copy = cute.flat_divide(LSE, (1, ))
    self.load_lse_stage(
      LSE_for_copy,
      sLSE_for_copy,
      lse_handle,
      iter_index,
      blk_coord_h,
      blk_coord_b,
      problem_shape[0],
      thread_idx,
      async_copy_num_elts,
      atom_async_copy,
    )
    lse_handle.commit()

    dot_handle = load_mma_dOT_producer.acquire_and_advance()
    for slice_index in cutlass.range_constexpr(self.dv_slices):
      cute.copy(
        tma_atom_dOT,
        tdOTgdOT_mkl[
          (None, slice_index, iter_index, (blk_coord_h, blk_coord_b))],
        tdOTsdOT[None, dot_handle.index * self.dv_slices + slice_index],
        tma_bar_ptr=dot_handle.barrier,
      )

    iter_count -= 1
    iter_index += 1

    while iter_count > 0:
      if iter_index == iter_end:
        iter_index = iter_start
        blk_coord_h_r += 1
        blk_coord_h = (blk_coord_h_r, blk_coord_h_k)

      q_handle = load_mma_Q_producer.acquire_and_advance()
      cute.copy(
        tma_atom_Q,
        tQgQ_mkl[(None, iter_index, 0, (blk_coord_h, blk_coord_b))],
        tQsQ[None, q_handle.index],
        tma_bar_ptr=q_handle.barrier,
      )

      lse_handle = load_compute_LSE_producer.acquire_and_advance()
      self.load_lse_stage(
        LSE_for_copy,
        sLSE_for_copy,
        lse_handle,
        iter_index,
        blk_coord_h,
        blk_coord_b,
        problem_shape[0],
        thread_idx,
        async_copy_num_elts,
        atom_async_copy,
      )
      lse_handle.commit()

      dot_handle = load_mma_dOT_producer.acquire_and_advance()
      for slice_index in cutlass.range_constexpr(self.dv_slices):
        cute.copy(
          tma_atom_dOT,
          tdOTgdOT_mkl[
            (None, slice_index, iter_index, (blk_coord_h, blk_coord_b))],
          tdOTsdOT[None, dot_handle.index * self.dv_slices + slice_index],
          tma_bar_ptr=dot_handle.barrier,
        )

      iter_count -= 1
      iter_index += 1

    # Every surviving producer needs its tail, else its consumer deadlocks.
    load_mma_K_producer.tail()
    load_mma_Q_producer.tail()
    load_compute_LSE_producer.tail()
    load_mma_dOT_producer.tail()

  @cute.jit
  def load_dOT(
    self,
    mdOT: cute.Tensor,
    sdOT: cute.Tensor,
    PdO_tiled_mma: cute.TiledMma,
    tma_atom_dOT: cute.CopyAtom,
    SeqlenInfoCls: Callable,
    iter_count: Int32,
    iter_start: Int32,
    iter_end: Int32,
    load_mma_dOT_producer,
  ):
    """The dOT issuer role: keeps dOT's ring acquire out of the Q/LSE path."""
    blk_coord_k, blk_coord_h_k, blk_coord_b = cute.arch.block_idx()
    blk_coord_h_r = Int32(0)
    blk_coord_h = (blk_coord_h_r, blk_coord_h_k)
    iter_index = iter_start
    mma_tile_coord_v = blk_coord_k % cute.size(PdO_tiled_mma.thr_id.shape)

    seqlen = SeqlenInfoCls(blk_coord_b)
    # dOT is the transposed operand: its sequence mode is 1, not 0.
    dOT = cute.domain_offset(
      (Int32(0), seqlen.offset_q, ((Int32(0), Int32(0)), Int32(0))), mdOT
    )
    gdOT = cute.local_tile(
      dOT, cute.select(self.PdO_mma_tiler, mode=[1, 2]), (None, None, None)
    )
    PdO_thr_mma = PdO_tiled_mma.get_slice(mma_tile_coord_v)
    tdVgdOT = PdO_thr_mma.partition_B(gdOT)

    cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)
    cta_layout_vmnk = cute.tiled_divide(
      cta_layout_mnk, (PdO_tiled_mma.thr_id, )
    )
    cta_in_cluster_coord_vmnk = cta_layout_vmnk.get_flat_coord(
      cute.arch.block_idx_in_cluster()
    )

    tdOTsdOT, tdOTgdOT_mkl = cute.nvgpu.cpasync.tma_partition(
      tma_atom_dOT,
      cta_in_cluster_coord_vmnk[1],
      cute.make_layout(cute.size(cta_layout_vmnk, mode=[1])),
      cute.group_modes(sdOT, 0, 3),
      cute.group_modes(tdVgdOT, 0, 3),
    )

    while iter_count > 0:
      if iter_index == iter_end:
        iter_index = iter_start
        blk_coord_h_r += 1
        blk_coord_h = (blk_coord_h_r, blk_coord_h_k)

      dot_handle = load_mma_dOT_producer.acquire_and_advance()
      for slice_index in cutlass.range_constexpr(self.dv_slices):
        cute.copy(
          tma_atom_dOT,
          tdOTgdOT_mkl[
            (None, slice_index, iter_index, (blk_coord_h, blk_coord_b))],
          tdOTsdOT[None, dot_handle.index * self.dv_slices + slice_index],
          tma_bar_ptr=dot_handle.barrier,
        )

      iter_count -= 1
      iter_index += 1

    load_mma_dOT_producer.tail()

  @cute.jit
  def mma(
    self,
    KQ_tiled_mma: cute.TiledMma,
    KQ_tmem_tiled_mma: cute.TiledMma,
    PdO_tiled_mma: cute.TiledMma,
    tSTtST_stages,
    tSTrQ: cute.Tensor,
    tSTrK: cute.Tensor,
    tSTrQ_tmem: cute.Tensor,
    tSTrK_tmem: cute.Tensor,
    tiled_s2t,
    tKsK_s2t: cute.Tensor,
    tKtK_s2t: cute.Tensor,
    tKT_addr: Int32,
    sQ: cute.Tensor,
    tdVtdV_slices,
    tdVrP: cute.Tensor,
    tdVrdOT: cute.Tensor,
    iter_count: Int32,
    load_mma_Q_consumer,
    load_mma_K_consumer,
    mma_compute_S_producer,
    load_mma_dOT_consumer,
    compute_mma_P_consumer,
    mma_compute_dV_producer,
  ):
    """MMA warp: S^T then dV issues over the Q sweep (single issuer)."""
    # Issue order != source order; the S region count and this order are one.
    load_mma_K_releaser = load_mma_K_consumer.clone()

    cta_rank_in_cluster = cute.arch.make_warp_uniform(
      cute.arch.block_idx_in_cluster()
    )
    is_leader_cta = cta_rank_in_cluster % 2 == 0

    # One S^T issue: local def, ZERO captures (closure_check bans them).
    def issue_S_generation(
      q_slot,
      s_slot,
      KQ_tiled_mma,
      KQ_tmem_tiled_mma,
      tSTtST_base,
      tmem_S_slice_stride,
      tSTrK_tmem,
      tSTrQ_tmem,
      tKT_addr,
      tSTrK,
      tSTrQ,
      sQ,
    ):
      tSTtST = cute.make_tensor(
        tSTtST_base.iterator + s_slot * tmem_S_slice_stride,
        tSTtST_base.layout,
      )
      gemm_ptx_w_idx(
        KQ_tmem_tiled_mma,
        tSTtST,
        tSTrK_tmem,
        tSTrQ_tmem,
        None,
        sQ,
        A_idx=0,
        B_idx=q_slot,
        zero_init=True,
        cta_group=2,
        tA_addr=tKT_addr,
      )
      KQ_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
      for k_block in cutlass.range(
        cute.size(tSTrK_tmem, mode=[2]),
        cute.size(tSTrQ, mode=[2]),
        1,
        unroll_full=True,
      ):
        cute.gemm(
          KQ_tiled_mma,
          tSTtST,
          tSTrK[None, None, k_block, 0],
          tSTrQ[None, None, k_block, q_slot],
          tSTtST,
        )
      # Rebind by NAME at the call: set(ACCUMULATE) exits scf by name.
      return KQ_tiled_mma

    # One P^T @ dO issue over every dV slice; same zero-capture rule.
    def issue_PdO_generation(
      p_slot,
      dot_slot,
      dv_accumulate,
      PdO_tiled_mma,
      tdVtdV_slices,
      tdVrP,
      tdVrdOT,
      dv_slices,
    ):
      for slice_index in cutlass.range_constexpr(len(tdVtdV_slices)):
        PdO_tiled_mma.set(tcgen05.Field.ACCUMULATE, dv_accumulate)
        for k_block in cutlass.range(
          0, cute.size(tdVrP, mode=[2]), unroll_full=True
        ):
          cute.gemm(
            PdO_tiled_mma,
            tdVtdV_slices[slice_index],
            tdVrP[None, None, k_block, p_slot],
            tdVrdOT[None, None, k_block, dot_slot * dv_slices + slice_index],
            tdVtdV_slices[slice_index],
          )
          PdO_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
      return PdO_tiled_mma

    # Advance the TMEM pointer: branching on stage clones the 32-issue KQ loop.
    tSTtST_base = tSTtST_stages[0]

    if is_leader_cta:
      load_mma_K_consumer.wait_and_advance()
      # cp and mma issue in order, no fence; the K slot covers the copy's read.
      cute.copy(tiled_s2t, tKsK_s2t, tKtK_s2t)

      if const_expr(self.mma_skew == 2):
        dv_accumulate = cutlass.Boolean(False)

        if iter_count > 1:
          # ---  prologue  ---
          # Warm-up: two S issues, so steady state pairs KQ(i+2) with PdO(i).
          for warm in cutlass.range_constexpr(2):
            q_handle = load_mma_Q_consumer.wait_and_advance()
            s_handle = mma_compute_S_producer.acquire_and_advance()
            KQ_tiled_mma = issue_S_generation(
              q_handle.index,
              s_handle.index,
              KQ_tiled_mma,
              KQ_tmem_tiled_mma,
              tSTtST_base,
              self.tmem_S_slice_stride,
              tSTrK_tmem,
              tSTrQ_tmem,
              tKT_addr,
              tSTrK,
              tSTrQ,
              sQ,
            )
            q_handle.release()
            cute.arch.fence_view_async_tmem_store()
            s_handle.commit()

          # ---  steady  ---
          issues_left = iter_count - 2
          while issues_left > 0:
            # S for tile i+2 first: this issue covers the P wait below.
            q_handle = load_mma_Q_consumer.wait_and_advance()
            s_handle = mma_compute_S_producer.acquire_and_advance()
            KQ_tiled_mma = issue_S_generation(
              q_handle.index,
              s_handle.index,
              KQ_tiled_mma,
              KQ_tmem_tiled_mma,
              tSTtST_base,
              self.tmem_S_slice_stride,
              tSTrK_tmem,
              tSTrQ_tmem,
              tKT_addr,
              tSTrK,
              tSTrQ,
              sQ,
            )
            q_handle.release()
            cute.arch.fence_view_async_tmem_store()
            s_handle.commit()

            # Consume the P of the tile two S issues back.
            p_handle = compute_mma_P_consumer.wait_and_advance()
            dot_handle = load_mma_dOT_consumer.wait_and_advance()

            PdO_tiled_mma = issue_PdO_generation(
              p_handle.index,
              dot_handle.index,
              dv_accumulate,
              PdO_tiled_mma,
              tdVtdV_slices,
              tdVrP,
              tdVrdOT,
              self.dv_slices,
            )

            p_handle.release()
            dot_handle.release()
            dv_accumulate = cutlass.Boolean(True)

            issues_left -= 1

          # ---  drain  ---
          # Drain: the two S issues of the warm-up still owe their dV.
          for drain in cutlass.range_constexpr(2):
            p_handle = compute_mma_P_consumer.wait_and_advance()
            dot_handle = load_mma_dOT_consumer.wait_and_advance()

            PdO_tiled_mma = issue_PdO_generation(
              p_handle.index,
              dot_handle.index,
              dv_accumulate,
              PdO_tiled_mma,
              tdVtdV_slices,
              tdVrP,
              tdVrdOT,
              self.dv_slices,
            )

            p_handle.release()
            dot_handle.release()
            dv_accumulate = cutlass.Boolean(True)
        else:
          # ---  single tile: prologue and drain in one  ---
          # Single Q tile: no skew constructible; plain issue-then-drain.
          q_handle = load_mma_Q_consumer.wait_and_advance()
          s_handle = mma_compute_S_producer.acquire_and_advance()
          KQ_tiled_mma = issue_S_generation(
            q_handle.index,
            s_handle.index,
            KQ_tiled_mma,
            KQ_tmem_tiled_mma,
            tSTtST_base,
            self.tmem_S_slice_stride,
            tSTrK_tmem,
            tSTrQ_tmem,
            tKT_addr,
            tSTrK,
            tSTrQ,
            sQ,
          )
          q_handle.release()
          cute.arch.fence_view_async_tmem_store()
          s_handle.commit()

          p_handle = compute_mma_P_consumer.wait_and_advance()
          dot_handle = load_mma_dOT_consumer.wait_and_advance()

          PdO_tiled_mma = issue_PdO_generation(
            p_handle.index,
            dot_handle.index,
            dv_accumulate,
            PdO_tiled_mma,
            tdVtdV_slices,
            tdVrP,
            tdVrdOT,
            self.dv_slices,
          )

          p_handle.release()
          dot_handle.release()
      else:
        dv_accumulate = cutlass.Boolean(False)

        # ---  prologue  ---
        # Prologue: issue S for the first tile; the warp stays one S ahead.
        q_handle = load_mma_Q_consumer.wait_and_advance()
        s_handle = mma_compute_S_producer.acquire_and_advance()
        KQ_tiled_mma = issue_S_generation(
          q_handle.index,
          s_handle.index,
          KQ_tiled_mma,
          KQ_tmem_tiled_mma,
          tSTtST_base,
          self.tmem_S_slice_stride,
          tSTrK_tmem,
          tSTrQ_tmem,
          tKT_addr,
          tSTrK,
          tSTrQ,
          sQ,
        )
        q_handle.release()
        cute.arch.fence_view_async_tmem_store()
        s_handle.commit()

        # ---  steady  ---
        issues_left = iter_count - 1
        while issues_left > 0:
          q_handle = load_mma_Q_consumer.wait_and_advance()
          s_handle = mma_compute_S_producer.acquire_and_advance()
          KQ_tiled_mma = issue_S_generation(
            q_handle.index,
            s_handle.index,
            KQ_tiled_mma,
            KQ_tmem_tiled_mma,
            tSTtST_base,
            self.tmem_S_slice_stride,
            tSTrK_tmem,
            tSTrQ_tmem,
            tKT_addr,
            tSTrK,
            tSTrQ,
            sQ,
          )
          q_handle.release()
          cute.arch.fence_view_async_tmem_store()
          s_handle.commit()

          p_handle = compute_mma_P_consumer.wait_and_advance()
          dot_handle = load_mma_dOT_consumer.wait_and_advance()

          PdO_tiled_mma = issue_PdO_generation(
            p_handle.index,
            dot_handle.index,
            dv_accumulate,
            PdO_tiled_mma,
            tdVtdV_slices,
            tdVrP,
            tdVrdOT,
            self.dv_slices,
          )

          p_handle.release()
          dot_handle.release()
          dv_accumulate = cutlass.Boolean(True)

          issues_left -= 1

        # ---  drain  ---
        # Drain: the S issued last still owes its dV.
        p_handle = compute_mma_P_consumer.wait_and_advance()
        dot_handle = load_mma_dOT_consumer.wait_and_advance()

        PdO_tiled_mma = issue_PdO_generation(
          p_handle.index,
          dot_handle.index,
          dv_accumulate,
          PdO_tiled_mma,
          tdVtdV_slices,
          tdVrP,
          tdVrdOT,
          self.dv_slices,
        )

        p_handle.release()
        dot_handle.release()

    if is_leader_cta:
      # All h_r/slices done: release K, then publish; the alias needs K dead.
      load_mma_K_releaser.release()
      load_mma_K_releaser.advance()

      dv_handle = mma_compute_dV_producer.acquire_and_advance()
      dv_handle.commit()

    mma_compute_S_producer.tail()
    mma_compute_dV_producer.tail()

  @cute.jit
  def compute_loop(
    self,
    tSTtST_stages,
    sP: cute.Tensor,
    sLSE: cute.Tensor,
    mdV: cute.Tensor,
    tdVtdV_slices,
    blk_coord: cute.Coord,
    SeqlenInfoCls: Callable,
    seqlen: SeqlenInfoQK,
    problem_shape: tuple[Int32, Int32, Int32, tuple[tuple[Int32, Int32],
                                                    Int32]],
    iter_count: Int32,
    iter_start: Int32,
    iter_end: Int32,
    scale_softmax: cutlass.Float32,
    mma_compute_S_consumer,
    compute_mma_P_producer,
    load_compute_LSE_consumer,
    mma_compute_dV_consumer,
    varlen: bool,
    problem_shape_k_cur_batch: Int32,
    tma_atom_dV: cute.CopyAtom,
    mdV_tma: cute.Tensor,
    s_epi_dV: cute.Tensor,
  ):
    """Recomputes softmax to publish P^T, then drains dV via epilogue()."""
    tidx, _, _ = cute.arch.thread_idx()
    seqlen_q, seqlen_k, _, _ = problem_shape
    _, blk_coord_k, _, _ = blk_coord

    iter_index = iter_start

    # Repetition = S^T width / 8, not a constant: donor's 16 writes OOB here.
    tmem_load_op = tcgen05.copy.Ld32x32bOp(
      tcgen05.copy.Repetition(self.tile_shape_Q // 8)
    )
    tmem_load_atom = cute.make_copy_atom(
      tmem_load_op,
      self.acc_dtype,
    )

    # Built off stage 0; the stage applies to the TMEM pointer, as in mma().
    tSTtST = tSTtST_stages[0][(None, None), 0, 0]

    cST = cute.make_identity_tensor(cute.select(self.cta_tiler, mode=[1, 0]))

    num_warp_groups = self.num_compute_warps // 4
    dp_idx = tidx % 128
    wg_idx = (tidx % (self.num_compute_warps * self.threads_per_warp)) // 128
    tiled_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tSTtST)
    thr_t2r = tiled_t2r.get_slice(dp_idx)

    tTR_cST = thr_t2r.partition_D(cST)
    tTR_cST = split_wg(tTR_cST, num_warp_groups, wg_idx)
    tTR_rST = cute.make_rmem_tensor(tTR_cST.shape, self.acc_dtype)

    t0TR_cST = split_wg(
      tiled_t2r.get_slice(0).partition_D(cST), num_warp_groups, wg_idx
    )
    attention_mask = AttentionMask(
      tile_m=self.tile_shape_Q,
      tile_n=self.tile_shape_K,
      seqlen_info=seqlen,
      window_size_left=self.window_size_left
      if self.has_sliding_window else None,
      window_size_right=self.window_size_right
      if self.has_sliding_window else None,
      swap_AB=True,
    )

    tTR_tST_base = thr_t2r.partition_S(tSTtST)
    tTR_tST_base = split_wg(tTR_tST_base, num_warp_groups, wg_idx)

    is_residual_k = blk_coord_k * self.tile_shape_K + self.tile_shape_K > seqlen_k
    last_iter = iter_end - 1
    log2_e = cutlass.Float32(math.log2(math.e))
    softmax_scale_log2_e = scale_softmax * log2_e

    while iter_count > 0:
      s_handle = mma_compute_S_consumer.wait_and_advance()
      p_handle = compute_mma_P_producer.acquire_and_advance()
      lse_handle = load_compute_LSE_consumer.wait_and_advance()

      leading_causal_masking = cutlass.Boolean(False)
      if const_expr(self.is_causal):
        # Exact tile gate; it also covers the NaN re-mask (always-on: 2 passes).
        leading_causal_masking = (
          iter_index * self.tile_shape_Q + seqlen_k - seqlen_q
          < blk_coord_k * self.tile_shape_K + self.tile_shape_K
        )
        leading_causal_masking = cute.arch.shuffle_sync(
          leading_causal_masking, 0
        )

      trailing_residual_masking = iter_index == last_iter or is_residual_k
      trailing_residual_masking = cute.arch.shuffle_sync(
        trailing_residual_masking, 0
      )

      is_masked_tile = (
        leading_causal_masking or trailing_residual_masking
        or self.has_sliding_window
      )

      # Compute P = softmax(S, LSE)
      tTR_tST = cute.make_tensor(
        tTR_tST_base.iterator + s_handle.index * self.tmem_S_slice_stride,
        tTR_tST_base.layout,
      )
      cute.copy(tiled_t2r, tTR_tST, tTR_rST)

      if is_masked_tile:
        attention_mask.apply_mask_sm100_transposed(
          tTR_rST,
          tTR_cST,
          t0TR_cST,
          m_block=iter_index,
          n_block=blk_coord_k,
          mask_seqlen=True,
          mask_causal=self.is_causal,
          mask_local=self.has_sliding_window,
        )

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

      # LSE = -inf gives exp2 NaN; store 0.  Reachable: q129/k1, bottom-right.
      if is_masked_tile:
        for i in cutlass.range(cute.size(tTR_rST), unroll_full=True):
          c_transpose = tTR_cST[i]
          pos = (
            cute.get(c_transpose, mode=[1]) + iter_index * self.tile_shape_Q,
            cute.get(c_transpose, mode=[0]) + blk_coord_k * self.tile_shape_K,
          )
          lane_invalid = not cute.elem_less(pos, (seqlen_q, seqlen_k))
          if const_expr(self.has_sliding_window):
            if const_expr(self.window_size_left < 0):
              lane_invalid = lane_invalid or (
                pos[1] > pos[0] + seqlen_k - seqlen_q + self.window_size_right
              )
            else:
              max_K_index = min(
                pos[0] + seqlen_k - seqlen_q + self.window_size_right, seqlen_k
              )
              min_K_index = max(
                0, pos[0] + seqlen_k - seqlen_q - self.window_size_left
              )
              lane_invalid = lane_invalid or (
                pos[1] > max_K_index or pos[1] < min_K_index
              )
          if const_expr(self.is_causal):
            lane_invalid = lane_invalid or (
              pos[0] + seqlen_k - seqlen_q < pos[1]
            )
          if lane_invalid:
            tTR_rST[i] = cutlass.Float32(0.0)

      # fp32 P -> the dV element type, the A operand of the PdO MMA.
      tTR_rPT = utils.cvt_f16(tTR_rST, mdV.element_type)
      reg_to_smem_mma128x128_2cta(
        tTR_rPT,
        sP,
        p_handle.index,
        (self.tile_shape_K, self.tile_shape_Q),
        dp_idx,
        wg_idx,
      )
      cute.arch.fence_view_async_shared()
      cute.arch.barrier(
        barrier_id=self.compute_sync_bar_id,
        number_of_threads=self.num_compute_warps * self.threads_per_warp,
      )

      p_handle.commit()

      s_handle.release()
      lse_handle.release()

      iter_count -= 1
      iter_index += 1
      if iter_index == iter_end:
        iter_index = iter_start

    # Epilogue: the consumer is mutated in place and this is its last user.
    self.epilogue(
      blk_coord,
      SeqlenInfoCls,
      problem_shape,
      mdV,
      tdVtdV_slices,
      mma_compute_dV_consumer,
      problem_shape_k_cur_batch,
      tma_atom_dV,
      mdV_tma,
      varlen,
      s_epi_dV,
    )

    compute_mma_P_producer.tail()

  @cute.jit
  def store_dV(
    self,
    gmem: cute.Tensor,
    regs: cute.Tensor,
    coord: cute.Tensor,
    tensor_shape: cute.Shape,
  ):
    """Varlen only: the dV TMA descriptor is dense, so predicated stores."""
    for i in cutlass.range(cute.size(coord, mode=[2]), unroll_full=True):
      # TODO: not lane-map forced; a 128-bit CopyUniversalOp here stays open.
      if cute.elem_less(coord[None, 0, i][0], tensor_shape):
        # ``regs`` arrives cast: one conversion, two arms.
        gmem[None, 0, i].store(regs[None, 0, i].load())

  @cute.jit
  def epilogue_clear(
    self,
    blk_coord: cute.Coord,
    SeqlenInfoCls: Callable,
    problem_shape: tuple[Int32, Int32, Int32, tuple[tuple[Int32, Int32],
                                                    Int32]],
    mdV: cute.Tensor,
  ):
    """Early stopping needs to clear dV."""
    tidx, _, _ = cute.arch.thread_idx()
    _, seqlen_k, _, HB = problem_shape
    _, blk_coord_k, _, blk_coord_batch = blk_coord

    # blk_coord[-1] is the ((h_r, h_k), b) nest; batch is its second element.
    seqlen = SeqlenInfoCls(blk_coord_batch[1])
    mdV_offset = cute.assume(seqlen.offset_k * mdV.stride[0], divby=64)
    mdV = cute.make_tensor(
      mdV.iterator + mdV_offset,
      cute.make_layout((seqlen_k, self.tile_shape_dV_dO, HB),
                       stride=mdV.stride),
    )
    gdV = cute.local_tile(
      mdV, (self.cta_tiler[1], self.cta_tiler[2]), (None, None, None)
    )
    gdV = gdV[None, None, blk_coord_k, 0, blk_coord_batch]
    cdV = cute.domain_offset(
      (blk_coord_k * self.tile_shape_K, 0),
      cute.make_identity_tensor((self.cta_tiler[1], self.cta_tiler[2])),
    )

    # Compute-warp thread count; all 384 threads enter, only tidx < it store.
    num_zero_epi_threads = self.num_compute_warps * self.threads_per_warp

    tiled_copy_r2g = copy_utils.tiled_copy_2d(
      mdV.element_type, self.cta_tiler[2], num_zero_epi_threads
    )

    thr_copy_r2g = tiled_copy_r2g.get_slice(tidx)

    tRG_gdV = thr_copy_r2g.partition_D(gdV)
    tRG_cdV = thr_copy_r2g.partition_D(cdV)

    zero_frg = cute.make_rmem_tensor_like(tRG_gdV[None, 0, None])
    zero_frg.fill(mdV.element_type(0.0))

    if tidx < num_zero_epi_threads:
      for n in cutlass.range(cute.size(tRG_gdV.shape[1]), unroll_full=True):
        if cute.elem_less(tRG_cdV[0, n, 0][0], problem_shape[1]):
          cute.copy(tiled_copy_r2g, zero_frg, tRG_gdV[None, n, None])

  @cute.jit
  def epilogue(
    self,
    blk_coord: cute.Coord,
    SeqlenInfoCls: Callable,
    problem_shape: tuple[Int32, Int32, Int32, tuple[tuple[Int32, Int32],
                                                    Int32]],
    mdV: cute.Tensor,
    tdVtdV_slices,
    mma_compute_dV_consumer,
    problem_shape_k_cur_batch: Int32,
    tma_atom_dV: cute.CopyAtom,
    mdV_tma: cute.Tensor,
    varlen: bool,
    s_epi_dV: cute.Tensor,
  ):
    """Publish dV via sK-aliased SMEM staging and TMA; varlen stores direct."""
    tidx, _, _ = cute.arch.thread_idx()
    _, seqlen_k, head_dim, HB = problem_shape
    _, blk_coord_k, _, blk_coord_batch = blk_coord

    # Again a width formula (one slice / 8 threads), not the donor's literal.
    tmem_copy_op = tcgen05.copy.Ld32x32bOp(
      tcgen05.copy.Repetition(self.dv_slice_n // 8)
    )
    load_op = cute.make_copy_atom(tmem_copy_op, self.acc_dtype)

    dp_idx = tidx % 128
    wg_idx = (tidx % (self.num_compute_warps * self.threads_per_warp)) // 128
    leader_warp = (cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4) == 0

    # Read from self, never re-derived: the loop must walk the dV TMA box.
    num_warp_groups = self.epi_num_warp_groups
    epi_tile_dV = self.epi_tile_dV
    total_epi_stages = self.epi_stages_dV

    mdV_in = cute.make_tensor(
      mdV.iterator,
      cute.make_layout((seqlen_k, self.cta_tiler[2], HB), stride=mdV.stride)
    )
    # As in epilogue_clear: the batch index is the nest's second element.
    seqlen = SeqlenInfoCls(blk_coord_batch[1])
    offset_mdV = cute.assume(seqlen.offset_k * mdV_in.stride[0], divby=64)
    mdV = cute.make_tensor(mdV_in.iterator + offset_mdV, mdV_in.layout)
    gdV = cute.local_tile(
      mdV, (self.cta_tiler[1], self.cta_tiler[2]), (None, None, None)
    )
    gdV = gdV[None, None, blk_coord_k, 0, blk_coord_batch]

    # Per-slice coords: a full-tile tensor overfeeds the T2R copy's N extent.
    cdV_slice_shape = (self.cta_tiler[1], self.dv_slice_n)

    if const_expr(not varlen):
      mdV_tma_3d = cute.make_tensor(
        mdV_tma.iterator,
        cute.make_layout((seqlen_k, self.cta_tiler[2], HB),
                         stride=mdV_tma.stride),
      )
      mdV_tma_cur = mdV_tma_3d[None, None, blk_coord_batch]
      gdV_tma = cute.local_tile(
        mdV_tma_cur, (self.cta_tiler[1], self.cta_tiler[2]), (blk_coord_k, 0)
      )
      gdV_tma_epi = cute.local_tile(gdV_tma, epi_tile_dV, (0, None))

    cta_threads = self.num_compute_warps * self.threads_per_warp

    # One wait for the one output-ready token the MMA warp publishes.
    dv_handle = mma_compute_dV_consumer.wait_and_advance()

    if blk_coord_k * self.tile_shape_K < problem_shape_k_cur_batch:
      for slice_index in cutlass.range_constexpr(len(tdVtdV_slices)):
        tdVtdV = tdVtdV_slices[slice_index][(None, None), 0, 0]
        tiled_t2r_dV = tcgen05.make_tmem_copy(load_op, tdVtdV)
        thread_t2r_dV = tiled_t2r_dV.get_slice(dp_idx)

        slice_n_origin = slice_index * self.dv_slice_n
        cdV = cute.domain_offset(
          (blk_coord_k * self.cta_tiler[1], slice_n_origin),
          cute.make_identity_tensor(cdV_slice_shape),
        )
        gdV_slice = cute.local_tile(gdV, cdV_slice_shape, (0, slice_index))

        tTR_cdV = thread_t2r_dV.partition_D(cdV)
        tTR_cdV = split_wg(tTR_cdV, num_warp_groups, wg_idx)
        tTR_gdV = thread_t2r_dV.partition_D(gdV_slice)
        tTR_gdV = split_wg(tTR_gdV, num_warp_groups, wg_idx)
        tTR_rdV = cute.make_rmem_tensor(tTR_cdV.shape, self.acc_dtype)
        tTR_tdV = thread_t2r_dV.partition_S(tdVtdV)
        tTR_tdV = split_wg(tTR_tdV, num_warp_groups, wg_idx)

        cute.copy(tiled_t2r_dV, tTR_tdV, tTR_rdV)
        # No scale: the donor's scale_softmax post-multiply belongs to dK.
        tTR_rdV_cast = cute.make_rmem_tensor(tTR_rdV.shape, mdV.element_type)
        tTR_rdV_cast.store(tTR_rdV.load().to(mdV.element_type))

        if const_expr(not varlen):
          # The n -> (col, stage) map is a layout -- group modes -- not a loop.
          s_epi_2d = cute.group_modes(s_epi_dV, 1, 3)
          s_epi_slice = cute.local_tile(
            s_epi_2d, cdV_slice_shape, (0, slice_index)
          )
          # Same T2R slice on dst: reg i pairs SMEM i, so 4 STS.128 not 32 U16.
          tTR_sdV = split_wg(
            thread_t2r_dV.partition_D(s_epi_slice),
            num_warp_groups,
            wg_idx,
          )
          cute.autovec_copy(tTR_rdV_cast, tTR_sdV)
        else:
          self.store_dV(tTR_gdV, tTR_rdV_cast, tTR_cdV, (seqlen_k, head_dim))

      if const_expr(not varlen):
        cute.arch.fence_view_async_shared()
        # Both WGs finish writing before the leader warp TMA-reads the buffer.
        cute.arch.barrier(
          barrier_id=self.epilogue_arena_bar_id, number_of_threads=cta_threads
        )
        if leader_warp and wg_idx == 0:
          for _stage in cutlass.range_constexpr(total_epi_stages):
            sdV_stage = s_epi_dV[None, None, _stage]
            gdV_stage = gdV_tma_epi[None, None, _stage]
            td_sdV, td_gdV = cpasync.tma_partition(
              tma_atom_dV,
              0,
              cute.make_layout(1),
              cute.group_modes(sdV_stage, 0, 2),
              cute.group_modes(gdV_stage, 0, 2),
            )
            cute.copy(tma_atom_dV, td_sdV, td_gdV)
            cute.arch.cp_async_bulk_commit_group()
        # Wait sunk to exit: pair sync + dealloc hide the TMA drain (measured).

    cute.arch.fence_view_async_tmem_load()
    dv_handle.release()

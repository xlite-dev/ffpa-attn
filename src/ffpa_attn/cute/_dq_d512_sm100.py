# Copyright (c) DefTruth, qyjdef@163.com
# Copyright (c) Butterfingrz，13524387014@163.com
#
# Adapted from the SM100 head-dim 256 specialized implementation
# in https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/sm100_hd256_2cta_fmha_backward_dqkernel.py
#
# SM100 (Blackwell) backward dQ for FFPA attention — only head_dim=512.
#
# Produces dQ alone: dQ = dS @ K with dS = P * (dP - dpsum) * scale, P =
# exp2(S * scale * log2e - lse), S = Q @ K^T and dP = dO @ V^T; lse (log2
# domain) and dpsum are caller-supplied.  Splitting the backward into three
# single-output kernels is forced, not stylistic — one (64, 128, 512) dQ tile
# already claims the whole SMEM ceiling and all 512 TMEM columns.  In exchange
# dQ iterates query-block outer / KV-block inner and reduces entirely in TMEM,
# so the fused backward's fp32 workspace, cp.reduce drain, determinism
# semaphore and postprocess convert never appear here.
#
# Design (12 warps as three 128-thread warpgroups, cluster (2, 1, 1)):
#   - LOAD (warp 9) TMAs Q, dO and K and cp.asyncs lse/dpsum; V LOAD (warp 10)
#     and K^T LOAD (warp 11) are dedicated issuers, because the three MMA edges
#     interleave and no single-issuer order feeds them without deadlocking.
#   - MMA (warp 8): S = Q @ K^T, dP = dO @ V^T and dQ += dS @ K^T, each on a
#     (128, 128, 128) tiler over four D slices, interleaved at gap six.
#   - COMPUTE (warps 0-3) runs softmax and dsoftmax out of TMEM, writing dS
#     back over dP in place; the dSK edge then takes dS as a TMEM A-operand,
#     reached by one SMEM half-row swap since dOV's row owner is not dS's.
#   - EPILOGUE (warps 4-7) drains dQ while compute runs on, staging through
#     SMEM aliased onto the dead sdO; warp 4 owns the TMEM allocation.
#   - TMEM is exactly full: S [0,64), dP [64,256), dQ [256,512).
#
# Constraints:
#   - mma_tiler == (64, 128, 512), and split_head is required
#   - cluster (2, 1, 1)
#   - lse_log2 is log2-domain
#   - No persistent or CLC scheduling

from typing import Callable, Type, Tuple, Optional
from functools import partial

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils_basic
from cutlass import const_expr, cute, pipeline
from cutlass.pipeline import Agent, CooperativeGroup
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.typing import Int32, Int64, Float32

from .utils import copy_utils
from . import utils
from .utils.cute_dsl_utils import assume_tensor_aligned
from .utils.blackwell_helpers import (
  SM100_SMEM_CAPACITY_BYTES,
  SM100_TMEM_CAPACITY_COLUMNS,
  gemm_ptx_w_idx,
)
from .utils.block_info import BlockInfo
from .utils.hd512_helpers import (
  check_tmem_intervals,
  tmem_offset,
)
from .utils.mask import Sm100FusedMask as FusedMask
from .utils.named_barrier import NamedBarrierBwdDQSm100Hd512
from .utils.seqlen_info import SeqlenInfoQK
from .utils.tile_scheduler import (
  Sm100FmhaLptTileScheduler as FmhaLptTileScheduler,
  Sm100FmhaStaticTileScheduler as FmhaStaticTileScheduler,
  Sm100FmhaStaticTileSchedulerParams as FmhaStaticTileSchedulerParams,
)


def issue_mma_slice(tiled_mma, acc, a_frag, b_frag, num_kphases):
  # Plain-Python loop; caller rebinds. Steady QK/dOV use gemm_ptx_w_idx (mma).
  for kphase_idx in range(num_kphases):
    kphase_coord = (None, None, kphase_idx)
    cute.gemm(tiled_mma, acc, a_frag[kphase_coord], b_frag[kphase_coord], acc)
    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
  return tiled_mma


class FFPAAttnBwdDQSm100D512:
  """SM100 D512 backward dQ: dQ = dS @ K alone, from caller-supplied lse/dpsum.

  S = Q @ K^T and dP = dO @ V^T are recomputed per KV block; dS overwrites dP
  in TMEM and feeds the dSK edge as a TMEM A operand (see the module header).
  """
  # Declared D512 topology: tile/TMEM constants derive from it; rings do not.
  TARGET_HEAD_DIM = 512
  TARGET_CTA_TILER = (64, 128, 512)
  TARGET_CLUSTER_SHAPE_MNK = (2, 1, 1)
  TARGET_MMA_SLICE = (128, 128, 128)
  TARGET_D_SLICES = 4
  # TMEM map, exactly full: 64 S + 3*64 dP + 256 dQ = 512 (ledger-checked).
  TARGET_TMEM_S_BASE = 0
  TARGET_TMEM_DP_BASE = 64
  TARGET_TMEM_DQ_BASE = 256
  TARGET_TMEM_DQ_SLICE_STRIDE = 64
  # K/V L2 bound at 148 SMs (74 pairs), fail-safe below: 2 underfills, 8 loses.
  TARGET_LPT_HEAD_GROUP = 4
  # Past this the closing anti-LPT wave is negligible; reorder gates off.
  TARGET_LPT_MAX_PAIRS = 640
  # S/dP trade depth only with each other; 1 holds because compute frees S fast.
  TARGET_QK_ACC_STAGE = 1
  # Three generations carry the two-step skew's liveness; fewer is a race.
  TARGET_DOV_ACC_STAGE = 3
  DS_PACKED_WORDS_PER_HALF = 32
  # All 32 words at once puts every phase at bank 0: an eight-way conflict.
  DS_EXCHANGE_PASS_WORDS = 4
  # 16 B pitch: each phase touches every bank once; the buffer is 2 KiB.
  DS_EXCHANGE_PITCH_WORDS = DS_EXCHANGE_PASS_WORDS
  # Two barriers per pass; affordable while compute hides behind the MMA.
  DS_EXCHANGE_PASSES = DS_PACKED_WORDS_PER_HALF // DS_EXCHANGE_PASS_WORDS

  def __init__(
    self,
    acc_dtype: Type[cutlass.Numeric],
    mma_tiler: Tuple[int, int, int],
    is_causal: bool,
    window_size_left: int | None,
    window_size_right: int | None,
    is_persistent: bool,
    split_head: bool,
    use_clc_scheduler: bool = False,
  ):
    """Configure the D512 dQ kernel; tile/D-slice topology is asserted."""
    self.acc_dtype = acc_dtype
    self.is_causal = is_causal
    window_size_left = (
      None if (window_size_left is None or window_size_left < 0) else
      cutlass.Int32(window_size_left)
    )
    window_size_right = (
      None if (window_size_right is None or window_size_right < 0) else
      cutlass.Int32(window_size_right)
    )
    self.window_size_left = None if self.is_causal else window_size_left
    self.window_size_right = cutlass.Int32(
      0
    ) if self.is_causal else window_size_right
    # Unreachable through the public wrapper, which rejects local attention.
    self.is_local = (not self.is_causal) and (
      self.window_size_left is not None or self.window_size_right is not None
    )
    assert mma_tiler == self.TARGET_CTA_TILER, (
      f"D512 dQ requires per-CTA tile {self.TARGET_CTA_TILER} "
      f"(head dimension {self.TARGET_HEAD_DIM})"
    )
    assert split_head, "D512 dQ requires four 128-wide head slices"
    # Tiler legend, (M, N, K): M = Q rows, N = KV rows, K = reduced head dim.
    self.cta_tiler = tuple(mma_tiler)
    self.qk_mma_tiler = self.TARGET_MMA_SLICE
    self.dov_mma_tiler = self.qk_mma_tiler
    self.dsk_mma_tiler = self.TARGET_MMA_SLICE
    self.ds_participant_rows = (
      self.dov_mma_tiler[0] // self.TARGET_CLUSTER_SHAPE_MNK[0]
    )

    self.dsk_block_tiler = (
      self.dsk_mma_tiler[0] // 2,
      self.dsk_mma_tiler[1],
      self.dsk_mma_tiler[2],
    )
    # D slices on the QK / dOV MMA K axes; D output slices on the dSK N axis.
    self.iterations_qk = self.cta_tiler[2] // self.qk_mma_tiler[2]
    self.iterations_dov = self.cta_tiler[2] // self.dov_mma_tiler[2]
    self.iterations_dsk = self.cta_tiler[2] // self.dsk_mma_tiler[1]
    assert (
      self.iterations_qk == self.iterations_dov == self.iterations_dsk ==
      self.TARGET_D_SLICES
    )
    self.cluster_shape_mnk = self.TARGET_CLUSTER_SHAPE_MNK
    # Cluster M drives both the tcgen05 CTA group and the TMA multicast group.
    cta_group_by_cluster_m = {
      1: tcgen05.CtaGroup.ONE,
      2: tcgen05.CtaGroup.TWO,
    }
    assert self.cluster_shape_mnk[0] in cta_group_by_cluster_m, (
      f"no tcgen05 CTA group for cluster M {self.cluster_shape_mnk[0]}"
    )
    self.cta_group = cta_group_by_cluster_m[self.cluster_shape_mnk[0]]
    # Rejected: the arena/sdO alias needs one tile per CTA per launch.
    assert not is_persistent, (
      "SM100 backward with head_dim=512 does not support persistent scheduling"
    )
    # Only the causal stream has a tail worth reordering; the rest are static.
    if self.is_causal and not self.is_local:
      self.tile_scheduler_cls = FmhaLptTileScheduler
    else:
      self.tile_scheduler_cls = FmhaStaticTileScheduler
    self.use_lpt_scheduler = self.tile_scheduler_cls is FmhaLptTileScheduler
    self.use_semantic_trip_range = self.is_causal or self.is_local

    self.compute_warp_ids = (0, 1, 2, 3)
    self.epilogue_warp_ids = (4, 5, 6, 7)
    self.mma_warp_id = 8
    self.load_warp_id = 9
    # Deadlock-freedom: no single-issuer order matches the interleaved consumer.
    self.v_load_warp_id = 10
    # The K-transpose TMA stream gets its own issuer.
    self.kt_load_warp_id = 11
    # No idle warp: both spare warps are dedicated TMA issuers at 32 registers.
    self.aux_load_warp_ids = (self.v_load_warp_id, self.kt_load_warp_id)
    # Kept for call-shape parity with the dK/dV siblings; the CLC code is gone.
    assert not use_clc_scheduler, (
      "the interleaved MMA schedule needs a dedicated issuer for each of "
      "K, V and K-transpose, and the CLC scheduler warp claims one of the "
      "two spare warps"
    )
    # Full capacity forces column 0; shrinking silently moves every TMEM offset.
    self.tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS
    self.num_compute_warps = len(self.compute_warp_ids)

    # Ids map in named_barrier.py; no whole-CTA rendezvous, hence no id 0.
    self.tmem_alloc_sync_bar_id = int(NamedBarrierBwdDQSm100Hd512.TmemPtr)
    self.compute_sync_bar_id = int(NamedBarrierBwdDQSm100Hd512.Compute)
    self.epilogue_arena_bar_id = int(NamedBarrierBwdDQSm100Hd512.EpilogueArena)

    self.threads_per_warp = cute.arch.WARP_SIZE
    self.epilogue_arena_threads = self.threads_per_warp * len(
      self.epilogue_warp_ids
    )
    self.threads_per_cta = self.threads_per_warp * len((
      *self.compute_warp_ids,  # all 12 warps: 3 x 128 threads
      *self.epilogue_warp_ids,
      self.mma_warp_id,
      self.load_warp_id,
      *self.aux_load_warp_ids,
    ))

    self.tmem_alloc_barrier = pipeline.NamedBarrier(
      barrier_id=self.tmem_alloc_sync_bar_id,
      num_threads=self.threads_per_cta,
    )
    self.compute_pair_barrier = pipeline.NamedBarrier(
      barrier_id=self.compute_sync_bar_id,
      num_threads=self.threads_per_warp * len(self.compute_warp_ids),
    )

    self.tmem_s_offset = self.TARGET_TMEM_S_BASE
    self.tmem_dp_offset = self.TARGET_TMEM_DP_BASE
    self.tmem_dq_offset = self.TARGET_TMEM_DQ_BASE
    # Early check: no fragment exists yet, so the declared stride is the width.
    check_tmem_intervals(
      self.tmem_region_intervals(self.TARGET_TMEM_DQ_SLICE_STRIDE)
    )

    self.num_regs_compute = 256
    self.num_regs_epilogue = 160
    # Load-bearing: at 96 the compute TRY_ALLOC hangs the device (measured).
    self.num_regs_other = 32
    assert (
      len(self.compute_warp_ids) == 4 and len(self.epilogue_warp_ids) == 4
      and self.threads_per_cta == 3 * 128
    ), "the 256/160/32 split assumes three 128-thread warpgroups"
    # Not asserted: setmaxnreg redistributes at run time, so 256 > the 168 cap.
    assert (
      128 *
      (self.num_regs_compute + self.num_regs_epilogue + self.num_regs_other)
      <= 65536
    ), (
      f"register budget {128 * (self.num_regs_compute + self.num_regs_epilogue + self.num_regs_other)} "
      f"exceeds the 65536 per-CTA register file"
    )

    # Zero Constexpr annotations, on purpose: no bare scalar crosses @cute.jit.

    self.buffer_align_bytes = 128

  def _get_tiled_mma(self):
    """Single factory for the three MMA edges (S = QK, dP = dOV, dQ = dSK)."""
    ds_source = tcgen05.OperandSource.TMEM
    ds_major_mode = tcgen05.OperandMajorMode.K
    k_trans_major_mode = tcgen05.OperandMajorMode.MN
    qk_tiled_mma = sm100_utils_basic.make_trivial_tiled_mma(
      self.q_dtype,
      self.q_major_mode,
      self.k_major_mode,
      self.acc_dtype,
      self.cta_group,
      self.qk_mma_tiler[:2],
    )
    dov_tiled_mma = sm100_utils_basic.make_trivial_tiled_mma(
      self.do_dtype,
      self.do_major_mode,
      self.v_major_mode,
      self.acc_dtype,
      self.cta_group,
      self.dov_mma_tiler[:2],
    )
    dsk_tiled_mma = sm100_utils_basic.make_trivial_tiled_mma(
      self.q_dtype,
      ds_major_mode,
      k_trans_major_mode,
      self.acc_dtype,
      self.cta_group,
      self.dsk_mma_tiler[:2],
      ds_source,
    )
    self.qk_tiled_mma = qk_tiled_mma
    self.dov_tiled_mma = dov_tiled_mma
    self.dsk_tiled_mma = dsk_tiled_mma
    return qk_tiled_mma, dov_tiled_mma, dsk_tiled_mma

  def tmem_region_intervals(self, region_columns):
    """Column ledger; region_columns = declared dQ slice stride (S/dP too)."""
    # Width is a parameter so declared and live-fragment ledgers compare.
    intervals = {
      "S": (
        self.tmem_s_offset,
        self.tmem_s_offset + self.TARGET_QK_ACC_STAGE * region_columns,
      ),
      "dP": (
        self.tmem_dp_offset,
        self.tmem_dp_offset + self.TARGET_DOV_ACC_STAGE * region_columns,
      ),
    }
    for slice_index in range(self.TARGET_D_SLICES):
      start = self.tmem_dq_offset + slice_index * self.TARGET_TMEM_DQ_SLICE_STRIDE
      intervals[f"dQ{slice_index}"] = (start, start + region_columns)
    return intervals

  def _setup_attributes(self):
    """Ring and accumulator stage depths (trace-time)."""
    # Q and dO stay resident: four stages hold a whole 64 x 512 operand per CTA.
    self.q_stage = self.iterations_qk
    self.do_stage = self.iterations_dov
    # Depth 2 fills SMEM exactly to the ceiling (trace-time assert); 4 does not.
    self.k_stage = 2
    self.v_stage = 2
    # One K^T slot starved the load warp (pre-interleave); the exchange funds 2.
    self.kt_stage = 2
    self.qk_acc_stage = self.TARGET_QK_ACC_STAGE
    self.dov_acc_stage = self.TARGET_DOV_ACC_STAGE
    # dS overwrites dP in place: one depth from both ends, never independent.
    self.dsk_acc_stage = self.dov_acc_stage
    self.mma_dq_stage = 1
    # Held once per tile across the KV sweep; a per-step acquire deadlocks.
    self.load_compute_LSE_stage = 1
    self.load_compute_dpsum_stage = 1
    # Derived once: the arena, the TMA box and the loop bound must agree.
    self.epi_tile = self.dsk_block_tiler[:2]
    self.epi_cols_dQ = math.gcd(
      128 // (self.dq_dtype.width // 8), self.epi_tile[1]
    )
    self.num_epi_stages_dQ = self.epi_tile[1] // self.epi_cols_dQ
    self.epi_tile_dQ = (self.epi_tile[0], self.epi_cols_dQ)

  @cute.jit
  def __call__(
    self,
    Q: cute.Tensor,
    K: cute.Tensor,
    V: cute.Tensor,
    dO: cute.Tensor,
    lse_log2: cute.Tensor,
    dpsum: cute.Tensor,
    dQ: cute.Tensor,
    scale_softmax: cutlass.Float32,
    cumulative_s_q: Optional[cute.Tensor],
    cumulative_s_k: Optional[cute.Tensor],
    stream: cuda.CUstream = None,
  ):
    """Trace entry: build layouts, TMA atoms and SharedStorage, then launch."""
    assert (cumulative_s_q is None) == (cumulative_s_k is None), (
      "varlen dQ requires both cumulative_s_q and cumulative_s_k"
    )
    # Rank-3/4 in, 5D (B, S, H_k, H_r, D) and (S, ((H_r, H_k), B)) views out.
    varlen = cumulative_s_q is not None
    q_rank = cute.rank(Q.layout)
    # Rank 3 = packed varlen; rank 4 = dense or a packed (1, total, H, D) view.
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
    Q, K, V, dQ, dO = [assume_tensor_aligned(t) for t in (Q, K, V, dQ, dO)]
    mQ = utils.as_bshkrd_tensor(Q, h_k_in, h_r_in, varlen)
    mK = utils.as_bshkrd_tensor(K, h_k_in, 1, varlen)
    mV = utils.as_bshkrd_tensor(V, h_k_in, 1, varlen)
    mdQ = utils.as_bshkrd_tensor(dQ, h_k_in, h_r_in, varlen)
    mdO = utils.as_bshkrd_tensor(dO, h_k_in, h_r_in, varlen)
    mLSE = utils.as_shhb_tensor(lse_log2, h_k_in, h_r_in, b_stats, varlen)
    mdPsum = utils.as_shhb_tensor(dpsum, h_k_in, h_r_in, b_stats, varlen)
    mCuSeqlensQ = cumulative_s_q
    mCuSeqlensK = cumulative_s_k
    s_q = mQ.shape[1]
    s_k = mK.shape[1]
    d = mQ.shape[4]
    h_k = mQ.shape[2]
    h_r = mQ.shape[3]
    if const_expr(mCuSeqlensQ is not None):
      b = mCuSeqlensQ.shape[0] - 1
    elif const_expr(mCuSeqlensK is not None):
      b = mCuSeqlensK.shape[0] - 1
    else:
      b = mQ.shape[0]
    # The caller pads the stats seq dim; its leading dim fixes batch strides.
    s_lse = mLSE.shape[0]
    s_q64 = Int64(s_q)
    s_k64 = Int64(s_k)
    s_lse64 = Int64(s_lse)
    h_r64 = Int64(h_r)
    h_k64 = Int64(h_k)
    b64 = Int64(b)
    # Packed varlen keeps the physical extent: cuseqlen offsets stay in-domain.
    s_q_total = mQ.shape[1] if mCuSeqlensQ is not None else s_q64
    s_k_total = mK.shape[1] if mCuSeqlensK is not None else s_k64
    b_lse = b64 if mCuSeqlensQ is None else 1
    stride_b_lse = h_r64 * h_k64 * s_lse64 if mCuSeqlensQ is None else 0

    q_layout = cute.make_layout(
      (s_q_total, d, ((h_r, h_k), b)),
      stride=(
        cute.assume(mQ.stride[1], divby=64),
        mQ.stride[4],
        (
          (mQ.stride[3], mQ.stride[2]),
          0 if mCuSeqlensQ is not None else cute.assume(mQ.stride[0], divby=64),
        ),
      ),
    )
    q = cute.make_tensor(mQ.iterator, q_layout)
    do_layout = cute.make_layout(
      (s_q_total, d, ((h_r, h_k), b)),
      stride=(
        cute.assume(mdO.stride[1], divby=64),
        mdO.stride[4],
        (
          (mdO.stride[3], mdO.stride[2]),
          0
          if mCuSeqlensQ is not None else cute.assume(mdO.stride[0], divby=64),
        ),
      ),
    )
    do = cute.make_tensor(mdO.iterator, do_layout)
    # (s, d, ((h_r, h_k), b)), 0-stride for h_r to broadcast
    k_layout = cute.make_layout(
      (s_k_total, d, ((h_r, h_k), b)),
      stride=(
        cute.assume(mK.stride[1], divby=64),
        mK.stride[4],
        (
          (0, mK.stride[2]),
          0 if mCuSeqlensK is not None else cute.assume(mK.stride[0], divby=64),
        ),
      ),
    )
    k = cute.make_tensor(mK.iterator, k_layout)
    # (d, s, ((h_r, h_k), b)), 0-stride for h_r to broadcast
    kt_layout = cute.make_layout(
      (d, s_k_total, ((h_r, h_k), b)),
      stride=(
        mK.stride[4],
        cute.assume(mK.stride[1], divby=64),
        (
          (0, mK.stride[2]),
          0 if mCuSeqlensK is not None else cute.assume(mK.stride[0], divby=64),
        ),
      ),
    )
    kt = cute.make_tensor(mK.iterator, kt_layout)
    # (s, d, ((h_r, h_k), b)), 0-stride for h_r to broadcast
    v_layout = cute.make_layout(
      (s_k_total, d, ((h_r, h_k), b)),
      stride=(
        cute.assume(mV.stride[1], divby=64),
        mV.stride[4],
        (
          (0, mV.stride[2]),
          0 if mCuSeqlensK is not None else cute.assume(mV.stride[0], divby=64),
        ),
      ),
    )
    v = cute.make_tensor(mV.iterator, v_layout)
    # One layout for both statistics keeps the address contracts from drifting.
    stats_layout = cute.make_layout(
      (s_lse64, ((h_r, h_k), b_lse)),
      stride=(1, ((s_lse64, h_r64 * s_lse64), stride_b_lse)),
    )
    lse = cute.make_tensor(mLSE.iterator, stats_layout)
    dpsum = cute.make_tensor(mdPsum.iterator, stats_layout)
    dq_layout = cute.make_layout(
      (s_q_total, d, ((h_r, h_k), b)),
      stride=(
        cute.assume(mdQ.stride[1], divby=64),
        mdQ.stride[4],
        (
          (mdQ.stride[3], mdQ.stride[2]),
          0
          if mCuSeqlensQ is not None else cute.assume(mdQ.stride[0], divby=64),
        ),
      ),
    )
    dq = cute.make_tensor(mdQ.iterator, dq_layout)

    # Trace-time only: dtypes, major modes and layouts come from traced tensors.
    self.q_dtype = q.element_type
    self.k_dtype = k.element_type
    self.v_dtype = v.element_type
    self.do_dtype = do.element_type
    self.dq_dtype = dq.element_type

    o_shape_for_grid = ((s_q, dq.shape[1],
                         dq.shape[2]) if mCuSeqlensQ is not None else dq.shape)
    self.tile_sched_params = self.tile_scheduler_cls.to_underlying_arguments(
      o_shape_for_grid,
      self.cta_tiler,
      False,  # is_persistent: scheduler API field; rejected in __init__
    )
    grid = self.tile_scheduler_cls.get_grid_shape(self.tile_sched_params)

    self.q_major_mode = cutlass.utils.LayoutEnum.from_tensor(q).mma_major_mode()
    self.do_major_mode = cutlass.utils.LayoutEnum.from_tensor(do
                                                              ).mma_major_mode()
    self.k_major_mode = cutlass.utils.LayoutEnum.from_tensor(k).mma_major_mode()
    self.v_major_mode = cutlass.utils.LayoutEnum.from_tensor(v).mma_major_mode()
    self.dq_layout = cutlass.utils.LayoutEnum.from_tensor(dq)

    if const_expr(self.q_major_mode != tcgen05.OperandMajorMode.K):
      raise RuntimeError("The layout of q is not supported")
    if const_expr(self.k_major_mode != tcgen05.OperandMajorMode.K):
      raise RuntimeError("The layout of k is not supported")
    if const_expr(self.v_major_mode != tcgen05.OperandMajorMode.K):
      raise RuntimeError("The layout of v is not supported")
    if const_expr(self.do_major_mode != tcgen05.OperandMajorMode.K):
      raise RuntimeError("The layout of do is not supported")

    if const_expr(self.q_dtype != self.k_dtype):
      raise TypeError(f"Type mismatch: {self.q_dtype} != {self.k_dtype}")
    if const_expr(self.q_dtype != self.v_dtype):
      raise TypeError(f"Type mismatch: {self.q_dtype} != {self.v_dtype}")
    if const_expr(self.q_dtype != self.do_dtype):
      raise TypeError(f"Type mismatch: {self.q_dtype} != {self.do_dtype}")

    self._setup_attributes()

    qk_tiled_mma, dov_tiled_mma, dsk_tiled_mma = self._get_tiled_mma()

    self.cluster_layout_vmnk = cute.tiled_divide(
      cute.make_layout(self.cluster_shape_mnk),
      (qk_tiled_mma.thr_id.shape, ),
    )

    # Every SMEM layout the kernel is launched with, in allocation order.
    self.sQ_layout = sm100_utils_basic.make_smem_layout_a(
      self.qk_tiled_mma,
      self.qk_mma_tiler,
      self.q_dtype,
      self.q_stage,
    )
    self.sK_layout = sm100_utils_basic.make_smem_layout_b(
      self.qk_tiled_mma,
      self.qk_mma_tiler,
      self.k_dtype,
      self.k_stage,
    )
    self.sdO_layout = sm100_utils_basic.make_smem_layout_a(
      self.dov_tiled_mma,
      self.dov_mma_tiler,
      self.do_dtype,
      self.do_stage,
    )
    self.sV_layout = sm100_utils_basic.make_smem_layout_b(
      self.dov_tiled_mma,
      self.dov_mma_tiler,
      self.v_dtype,
      self.v_stage,
    )
    # Stage pinned to 1: S/dP depth must not reach the dS operand layout.
    tdS_layout_staged = sm100_utils_basic.make_smem_layout_a(
      self.dsk_tiled_mma,
      self.dsk_mma_tiler,
      self.q_dtype,
      1,
    )
    self.ds_tmem_layout = cute.select(tdS_layout_staged, mode=[0, 1, 2])
    self.sKt_layout = sm100_utils_basic.make_smem_layout_b(
      self.dsk_tiled_mma,
      self.dsk_mma_tiler,
      self.k_dtype,
      self.kt_stage,
    )
    self.lse_smem_layout = cute.make_layout(
      (self.cta_tiler[0], self.load_compute_LSE_stage)
    )
    self.dpsum_smem_layout = cute.make_layout(
      (self.cta_tiler[0], self.load_compute_dpsum_stage)
    )
    self.sdQ_epi_layout = sm100_utils_basic.make_smem_layout_epi(
      self.dq_dtype,
      self.dq_layout,
      self.epi_tile_dQ,
      1,
    )
    # tma_copy_bytes (tx_count) takes the 2-CTA both-halves factor exactly once.
    q_smem_layout = cute.select(self.sQ_layout, mode=[0, 1, 2])
    k_smem_layout = cute.select(self.sK_layout, mode=[0, 1, 2])
    v_smem_layout = cute.select(self.sV_layout, mode=[0, 1, 2])
    do_smem_layout = cute.select(self.sdO_layout, mode=[0, 1, 2])
    kt_smem_layout = cute.select(self.sKt_layout, mode=[0, 1, 2])

    tma_load_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)

    tma_atom_q, tma_tensor_q = cute.nvgpu.make_tiled_tma_atom_A(
      tma_load_op,
      q,
      q_smem_layout,
      self.qk_mma_tiler,
      qk_tiled_mma,
      self.cluster_layout_vmnk.shape,
    )
    tma_atom_k, tma_tensor_k = cute.nvgpu.make_tiled_tma_atom_B(
      tma_load_op,
      k,
      k_smem_layout,
      self.qk_mma_tiler,
      qk_tiled_mma,
      self.cluster_layout_vmnk.shape,
    )
    tma_atom_do, tma_tensor_do = cute.nvgpu.make_tiled_tma_atom_A(
      tma_load_op,
      do,
      do_smem_layout,
      self.dov_mma_tiler,
      dov_tiled_mma,
      self.cluster_layout_vmnk.shape,
    )
    tma_atom_v, tma_tensor_v = cute.nvgpu.make_tiled_tma_atom_B(
      tma_load_op,
      v,
      v_smem_layout,
      self.dov_mma_tiler,
      dov_tiled_mma,
      self.cluster_layout_vmnk.shape,
    )
    # K^T -- operand B of dQ = dS @ K, the one MN-major operand
    tma_atom_kt, tma_tensor_kt = cute.nvgpu.make_tiled_tma_atom_B(
      tma_load_op,
      kt,
      kt_smem_layout,
      self.dsk_mma_tiler,
      dsk_tiled_mma,
      self.cluster_layout_vmnk.shape,
    )
    tma_atom_dQ, tma_tensor_dQ = cpasync.make_tiled_tma_atom(
      cpasync.CopyBulkTensorTileS2GOp(),
      dq,
      cute.select(self.sdQ_epi_layout, mode=[0, 1]),
      self.epi_tile_dQ,
    )

    cta_group_size = cute.size(qk_tiled_mma.thr_id.shape)
    self.tma_copy_bytes = {
      name: cta_group_size * cute.size_in_bytes(dtype, layout)
      for name, dtype, layout in [
        ("Q", self.q_dtype, q_smem_layout),
        ("K", self.k_dtype, k_smem_layout),
        ("V", self.v_dtype, v_smem_layout),
        ("dO", self.do_dtype, do_smem_layout),
        ("KT", self.k_dtype, kt_smem_layout),
      ]
    }

    @cute.struct
    class SharedStorage:
      # TMA G2S load barriers: LOAD warp (producer) -> MMA warp (consumer)
      load_q_mbar_ptr: cute.struct.MemRange[Int64, self.q_stage * 2
                                            ]  # load_q_{producer,consumer}
      load_do_mbar_ptr: cute.struct.MemRange[Int64, self.do_stage * 2
                                             ]  # load_do_{producer,consumer}
      load_k_mbar_ptr: cute.struct.MemRange[Int64, self.k_stage * 2
                                            ]  # load_k_{producer,consumer}
      load_kt_mbar_ptr: cute.struct.MemRange[Int64, self.kt_stage * 2
                                             ]  # load_kt_{producer,consumer}
      load_v_mbar_ptr: cute.struct.MemRange[Int64, self.v_stage * 2
                                            ]  # load_v_{producer,consumer}
      mma_s_mbar_ptr: cute.struct.MemRange[Int64, self.qk_acc_stage * 2]
      mma_dp_mbar_ptr: cute.struct.MemRange[Int64, self.dov_acc_stage * 2]
      mma_dq_mbar_ptr: cute.struct.MemRange[Int64, self.mma_dq_stage * 2]
      ds_mma_mbar_ptr: cute.struct.MemRange[Int64, self.dsk_acc_stage * 2]
      lse_mbar_ptr: cute.struct.MemRange[cutlass.Int64,
                                         self.load_compute_LSE_stage * 2]
      dpsum_mbar_ptr: cute.struct.MemRange[cutlass.Int64,
                                           self.load_compute_dpsum_stage * 2]
      # 2-CTA TMEM lifetime mbar: both CTAs finish before the dealloc.
      tmem_dealloc_mbar: Int64
      tmem_holding_buf: Int32
      sQ: cute.struct.Align[
        cute.struct.MemRange[self.q_dtype,
                             cute.cosize(self.sQ_layout)],
        self.buffer_align_bytes,
      ]
      sK: cute.struct.Align[
        cute.struct.MemRange[self.k_dtype,
                             cute.cosize(self.sK_layout)],
        self.buffer_align_bytes,
      ]
      sV: cute.struct.Align[
        cute.struct.MemRange[self.v_dtype,
                             cute.cosize(self.sV_layout)],
        self.buffer_align_bytes,
      ]
      # Also hosts the dQ epilogue arena, by recast at the use site (sdO dead).
      sdO: cute.struct.Align[
        cute.struct.MemRange[self.do_dtype,
                             cute.cosize(self.sdO_layout)],
        self.buffer_align_bytes,
      ]
      sKT: cute.struct.Align[
        cute.struct.MemRange[self.k_dtype,
                             cute.cosize(self.sKt_layout)],
        self.buffer_align_bytes,
      ]
      sLSE: cute.struct.Align[
        cute.struct.MemRange[self.acc_dtype,
                             cute.cosize(self.lse_smem_layout)],
        self.buffer_align_bytes,
      ]
      sdPsum: cute.struct.Align[
        cute.struct.MemRange[self.acc_dtype,
                             cute.cosize(self.dpsum_smem_layout)],
        self.buffer_align_bytes,
      ]
      # dP->dS half-row exchange; extent = measured pitch x rows x both ends.
      sdS_xchg: cute.struct.Align[
        cute.struct.MemRange[
          self.acc_dtype,
          cute.cosize(
            cute.make_layout(
              2 * self.ds_participant_rows * self.DS_EXCHANGE_PITCH_WORDS
            )
          ),
        ],
        self.buffer_align_bytes,
      ]

    self.shared_storage = SharedStorage
    # The one executable SMEM-budget check; header prose is not a witness.
    assert SharedStorage.size_in_bytes() <= SM100_SMEM_CAPACITY_BYTES, (
      f"SharedStorage {SharedStorage.size_in_bytes()} B > SM100 opt-in "
      f"ceiling {SM100_SMEM_CAPACITY_BYTES} B"
    )

    grid = cute.round_up(grid, self.cluster_shape_mnk)
    self.kernel(
      qk_tiled_mma,
      dov_tiled_mma,
      dsk_tiled_mma,
      tma_atom_q,
      tma_tensor_q,
      tma_atom_k,
      tma_tensor_k,
      tma_atom_v,
      tma_tensor_v,
      tma_atom_do,
      tma_tensor_do,
      tma_atom_kt,
      tma_tensor_kt,
      tma_atom_dQ,
      tma_tensor_dQ,
      lse,
      dpsum,
      dq,
      mCuSeqlensQ,
      mCuSeqlensK,
      scale_softmax,
      self.window_size_left,
      self.window_size_right,
      self.cluster_layout_vmnk,
      self.sQ_layout,
      self.sK_layout,
      self.sV_layout,
      self.sdO_layout,
      self.sKt_layout,
      self.ds_tmem_layout,
      self.sdQ_epi_layout,
      self.lse_smem_layout,
      self.dpsum_smem_layout,
      self.tile_sched_params,
    ).launch(
      grid=grid,
      block=[self.threads_per_cta, 1, 1],
      cluster=self.cluster_shape_mnk,
      smem=self.shared_storage.size_in_bytes(),  # type: ignore [attr-defined]
      stream=stream,
      min_blocks_per_mp=1,
    )

  @cute.kernel
  def kernel(
    self,
    qk_tiled_mma: cute.TiledMma,
    dov_tiled_mma: cute.TiledMma,
    dsk_tiled_mma: cute.TiledMma,
    tma_atom_q: cute.CopyAtom,
    mQ_qdl: cute.Tensor,
    tma_atom_k: cute.CopyAtom,
    mK_kdl: cute.Tensor,
    tma_atom_v: cute.CopyAtom,
    mV_dkl: cute.Tensor,
    tma_atom_do: cute.CopyAtom,
    mdO_qdl: cute.Tensor,
    tma_atom_kt: cute.CopyAtom,
    mKt_dkl: cute.Tensor,
    tma_atom_dQ: cute.CopyAtom,
    mdQ_tma: cute.Tensor,
    mLSE: cute.Tensor,
    mdPsum: cute.Tensor,
    mdQ_qdl: cute.Tensor,
    mCuSeqlensQ: Optional[cute.Tensor],
    mCuSeqlensK: Optional[cute.Tensor],
    scale_softmax: Float32,
    window_size_left: Optional[Int32],
    window_size_right: Optional[Int32],
    cluster_layout_vmnk: cute.Layout,
    sQ_layout: cute.ComposedLayout,
    sK_layout: cute.ComposedLayout,
    sV_layout: cute.ComposedLayout,
    sdO_layout: cute.ComposedLayout,
    sKt_layout: cute.ComposedLayout,
    ds_tmem_layout: cute.ComposedLayout,
    sdQ_epi_layout: cute.ComposedLayout,
    lse_smem_layout: cute.Layout,
    dpsum_smem_layout: cute.Layout,
    tile_sched_params: FmhaStaticTileSchedulerParams,
  ):
    """Kernel body: role dispatch by warp index."""
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    # Dispatch bounds of the multi-warp roles.
    compute_lo = self.compute_warp_ids[0]
    compute_hi = self.compute_warp_ids[-1]
    epilogue_lo = self.epilogue_warp_ids[0]
    epilogue_hi = self.epilogue_warp_ids[-1]
    if warp_idx == self.load_warp_id:
      cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
      cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
      cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)
      cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_do)
      cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_kt)
      cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_dQ)

    bidx, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    mma_tile_coord_v = bidx % cute.size(qk_tiled_mma.thr_id.shape)
    cta_rank_in_cluster = cute.arch.make_warp_uniform(
      cute.arch.block_idx_in_cluster()
    )
    block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
      cta_rank_in_cluster
    )

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(self.shared_storage)

    # TMA issuers and the MMA warp arrive through one elected thread each;
    # UmmaAsync consumer groups span both CTAs; the factor counts the peer.
    load_producer_group = CooperativeGroup(Agent.Thread, 1)
    mma_group = CooperativeGroup(Agent.Thread, 1)
    compute_group = CooperativeGroup(
      Agent.Thread,
      len(self.compute_warp_ids) * self.threads_per_warp *
      self.cluster_shape_mnk[0],
    )
    epilogue_group = CooperativeGroup(
      Agent.Thread,
      len(self.epilogue_warp_ids) * self.threads_per_warp *
      self.cluster_shape_mnk[0],
    )
    stats_producer_group = CooperativeGroup(Agent.Thread, self.threads_per_warp)
    stats_consumer_group = CooperativeGroup(
      Agent.Thread, self.threads_per_warp * self.num_compute_warps
    )

    load_q_producer, load_q_consumer = pipeline.PipelineTmaUmma.create(
      num_stages=self.q_stage,
      producer_group=load_producer_group,
      consumer_group=mma_group,
      tx_count=self.tma_copy_bytes["Q"],
      barrier_storage=storage.load_q_mbar_ptr.data_ptr(),
      cta_layout_vmnk=cluster_layout_vmnk,
      defer_sync=True,
    ).make_participants()
    load_k_producer, load_k_consumer = pipeline.PipelineTmaUmma.create(
      num_stages=self.k_stage,
      producer_group=load_producer_group,
      consumer_group=mma_group,
      tx_count=self.tma_copy_bytes["K"],
      barrier_storage=storage.load_k_mbar_ptr.data_ptr(),
      cta_layout_vmnk=cluster_layout_vmnk,
      defer_sync=True,
    ).make_participants()
    load_v_producer, load_v_consumer = pipeline.PipelineTmaUmma.create(
      num_stages=self.v_stage,
      producer_group=load_producer_group,
      consumer_group=mma_group,
      tx_count=self.tma_copy_bytes["V"],
      barrier_storage=storage.load_v_mbar_ptr.data_ptr(),
      cta_layout_vmnk=cluster_layout_vmnk,
      defer_sync=True,
    ).make_participants()
    load_do_producer, load_do_consumer = pipeline.PipelineTmaUmma.create(
      num_stages=self.do_stage,
      producer_group=load_producer_group,
      consumer_group=mma_group,
      tx_count=self.tma_copy_bytes["dO"],
      barrier_storage=storage.load_do_mbar_ptr.data_ptr(),
      cta_layout_vmnk=cluster_layout_vmnk,
      defer_sync=True,
    ).make_participants()
    load_kt_producer, load_kt_consumer = pipeline.PipelineTmaUmma.create(
      num_stages=self.kt_stage,
      producer_group=load_producer_group,
      consumer_group=mma_group,
      tx_count=self.tma_copy_bytes["KT"],
      barrier_storage=storage.load_kt_mbar_ptr.data_ptr(),
      cta_layout_vmnk=cluster_layout_vmnk,
      defer_sync=True,
    ).make_participants()
    mma_s_producer, mma_s_consumer = pipeline.PipelineUmmaAsync.create(
      num_stages=self.qk_acc_stage,
      producer_group=mma_group,
      consumer_group=compute_group,
      barrier_storage=storage.mma_s_mbar_ptr.data_ptr(),
      cta_layout_vmnk=cluster_layout_vmnk,
      defer_sync=True,
    ).make_participants()
    mma_dp_producer, mma_dp_consumer = pipeline.PipelineUmmaAsync.create(
      num_stages=self.dov_acc_stage,
      producer_group=mma_group,
      consumer_group=compute_group,
      barrier_storage=storage.mma_dp_mbar_ptr.data_ptr(),
      cta_layout_vmnk=cluster_layout_vmnk,
      defer_sync=True,
    ).make_participants()
    ds_mma_producer, ds_mma_consumer = pipeline.PipelineAsyncUmma.create(
      num_stages=self.dsk_acc_stage,
      producer_group=compute_group,
      consumer_group=mma_group,
      barrier_storage=storage.ds_mma_mbar_ptr.data_ptr(),
      cta_layout_vmnk=cluster_layout_vmnk,
      defer_sync=True,
    ).make_participants()
    mma_dq_producer, mma_dq_consumer = pipeline.PipelineUmmaAsync.create(
      num_stages=self.mma_dq_stage,
      producer_group=mma_group,
      consumer_group=epilogue_group,
      barrier_storage=storage.mma_dq_mbar_ptr.data_ptr(),
      cta_layout_vmnk=cluster_layout_vmnk,
      defer_sync=True,
    ).make_participants()

    load_lse_producer, load_lse_consumer = pipeline.PipelineCpAsync.create(
      num_stages=self.load_compute_LSE_stage,
      producer_group=stats_producer_group,
      consumer_group=stats_consumer_group,
      barrier_storage=storage.lse_mbar_ptr.data_ptr(),
    ).make_participants()
    load_dpsum_producer, load_dpsum_consumer = pipeline.PipelineCpAsync.create(
      num_stages=self.load_compute_dpsum_stage,
      producer_group=stats_producer_group,
      consumer_group=stats_consumer_group,
      barrier_storage=storage.dpsum_mbar_ptr.data_ptr(),
    ).make_participants()

    tmem = cutlass.utils.TmemAllocator(
      storage.tmem_holding_buf.ptr,
      barrier_for_retrieve=self.tmem_alloc_barrier,
      allocator_warp_id=self.epilogue_warp_ids[0],
      is_two_cta=True,
      two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
    )
    tmem.allocate(self.tmem_alloc_cols)
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

    pipeline.pipeline_init_arrive(
      cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True
    )

    sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
    sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
    sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
    sdO = storage.sdO.get_tensor(sdO_layout.outer, swizzle=sdO_layout.inner)
    sKT = storage.sKT.get_tensor(sKt_layout.outer, swizzle=sKt_layout.inner)
    sLSE = storage.sLSE.get_tensor(lse_smem_layout)
    sdPsum = storage.sdPsum.get_tensor(dpsum_smem_layout)
    # Two participants each publish a packed half-row; compute_step joins them.
    sdS_xchg = storage.sdS_xchg.get_tensor(
      cute.make_layout(
        2 * self.ds_participant_rows * self.DS_EXCHANGE_PITCH_WORDS
      )
    )
    # Aliases sdO, dead once dQ is ready; the next tile's fill would race.
    s_epi_dQ = cute.make_tensor(
      cute.recast_ptr(sdO.iterator, sdQ_epi_layout.inner, self.dq_dtype),
      sdQ_epi_layout.outer,
    )
    qk_thr_mma = qk_tiled_mma.get_slice(mma_tile_coord_v)
    dov_thr_mma = dov_tiled_mma.get_slice(mma_tile_coord_v)
    dsk_thr_mma = dsk_tiled_mma.get_slice(mma_tile_coord_v)
    tSrQ = qk_thr_mma.make_fragment_A(sQ)
    tSrK = qk_thr_mma.make_fragment_B(sK)
    tdPrdO = dov_thr_mma.make_fragment_A(sdO)
    tdPrV = dov_thr_mma.make_fragment_B(sV)
    tdQrKT = dsk_thr_mma.make_fragment_B(sKT)
    qk_acc_shape = qk_thr_mma.partition_shape_C(
      (self.qk_mma_tiler[0], self.qk_mma_tiler[1])
    )
    tStS = qk_thr_mma.make_fragment_C(
      cute.append(qk_acc_shape, self.qk_acc_stage)
    )
    dov_acc_shape = dov_thr_mma.partition_shape_C(
      (self.dov_mma_tiler[0], self.dov_mma_tiler[1])
    )
    tdPtdP = dov_thr_mma.make_fragment_C(
      cute.append(dov_acc_shape, self.dov_acc_stage)
    )
    dsk_acc_shape = dsk_thr_mma.partition_shape_C(
      (self.dsk_mma_tiler[0], self.dsk_mma_tiler[1])
    )
    tdQtdQ = dsk_thr_mma.make_fragment_C(dsk_acc_shape)
    tdQtdQ_layout = cute.append(
      tdQtdQ.layout,
      cute.make_layout(
        self.iterations_dsk,
        stride=self.TARGET_TMEM_DQ_SLICE_STRIDE,
      ),
    )
    # Full-capacity rebase precondition: no cheap gate; dK/dV are base-robust.
    tStS = cute.make_tensor(tStS.iterator + self.tmem_s_offset, tStS.layout)
    tdPtdP = cute.make_tensor(
      tdPtdP.iterator + self.tmem_dp_offset, tdPtdP.layout
    )
    tdQtdQ_staged = cute.make_tensor(
      tdQtdQ.iterator + self.tmem_dq_offset, tdQtdQ_layout
    )

    # A region fence for ptxas, not a redundancy: deleting it measured a loss.
    for _i in cutlass.range_constexpr(len(self.aux_load_warp_ids)):
      if warp_idx == self.aux_load_warp_ids[_i]:
        cute.arch.setmaxregister_decrease(self.num_regs_other)

    blk_idx = cute.arch.block_idx()
    if const_expr(self.use_lpt_scheduler):
      tile_sched = FmhaLptTileScheduler(
        tile_sched_params,
        blk_idx[0],
        blk_idx,
        cute.arch.grid_dim(),
        cluster_shape_m=self.cluster_shape_mnk[0],
        head_group=self.TARGET_LPT_HEAD_GROUP,
        max_pairs=self.TARGET_LPT_MAX_PAIRS,
      )
    else:
      tile_sched = FmhaStaticTileScheduler(
        tile_sched_params, blk_idx[0], blk_idx, cute.arch.grid_dim()
      )
    # No WorkTileInfo above the split: it pins divisions, +8 B/thread spill.

    # A factory: an object held across the warp split costs 8-48 B/thread.
    SeqlenInfoCls = partial(
      SeqlenInfoQK.create,
      seqlen_q_static=mQ_qdl.shape[0],
      seqlen_k_static=mK_kdl.shape[0],
      mCuSeqlensQ=mCuSeqlensQ,
      mCuSeqlensK=mCuSeqlensK,
      # The padded LSE/dPsum offset divides by 64 per CTA, not the pair's 128.
      tile_m=self.cta_tiler[0],
      tile_n=self.qk_mma_tiler[1],
    )
    # The object itself: compile-time plus two Int32s the roles already carry.
    block_info = BlockInfo(
      # The pair tiler, not the per-CTA one: disagreement hangs the cluster.
      self.qk_mma_tiler[0],
      self.qk_mma_tiler[1],
      self.is_causal,
      self.is_local and not self.is_causal,
      window_size_left,
      window_size_right,
      qhead_per_kvhead_packgqa=1,
    )

    pipeline.pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

    # ///  LOAD  ///
    if warp_idx == self.load_warp_id:
      cute.arch.setmaxregister_decrease(self.num_regs_other)
      self.load(
        qk_tiled_mma,
        qk_thr_mma,
        dov_thr_mma,
        mQ_qdl,
        mK_kdl,
        mdO_qdl,
        mLSE,
        mdPsum,
        sQ,
        sK,
        sdO,
        sLSE,
        sdPsum,
        tma_atom_q,
        tma_atom_k,
        tma_atom_do,
        load_q_producer,
        load_k_producer,
        load_do_producer,
        load_lse_producer,
        load_dpsum_producer,
        SeqlenInfoCls,
        block_info,
        cluster_layout_vmnk,
        block_in_cluster_coord_vmnk,
        tidx,
        tile_sched,
      )

    # ///  LOAD (V)  ///
    if warp_idx == self.v_load_warp_id:
      cute.arch.setmaxregister_decrease(self.num_regs_other)
      self.load_v(
        qk_tiled_mma,
        dov_thr_mma,
        mV_dkl,
        sV,
        tma_atom_v,
        load_v_producer,
        SeqlenInfoCls,
        block_info,
        cluster_layout_vmnk,
        block_in_cluster_coord_vmnk,
        tile_sched,
      )

    # ///  LOAD (K-transpose)  ///
    if warp_idx == self.kt_load_warp_id:
      cute.arch.setmaxregister_decrease(self.num_regs_other)
      self.load_kt(
        qk_tiled_mma,
        dsk_thr_mma,
        mKt_dkl,
        sKT,
        tma_atom_kt,
        load_kt_producer,
        SeqlenInfoCls,
        block_info,
        cluster_layout_vmnk,
        block_in_cluster_coord_vmnk,
        tile_sched,
      )

    # ///  MMA  ///
    if warp_idx == self.mma_warp_id:
      cute.arch.setmaxregister_decrease(self.num_regs_other)
      self.mma(
        qk_tiled_mma,
        dov_tiled_mma,
        dsk_tiled_mma,
        dsk_thr_mma,
        sQ,
        sK,
        sdO,
        sV,
        tSrQ,
        tSrK,
        tdPrdO,
        tdPrV,
        tdQrKT,
        tStS,
        tdPtdP,
        tdQtdQ_staged,
        ds_tmem_layout,
        load_q_consumer,
        load_k_consumer,
        load_v_consumer,
        load_do_consumer,
        load_kt_consumer,
        mma_s_producer,
        mma_dp_producer,
        ds_mma_consumer,
        mma_dq_producer,
        SeqlenInfoCls,
        block_info,
        tile_sched,
      )

    # ///  COMPUTE (softmax / dsoftmax)  ///
    if warp_idx >= compute_lo and warp_idx <= compute_hi:
      cute.arch.setmaxregister_increase(self.num_regs_compute)
      self.compute_loop(
        qk_tiled_mma,
        qk_thr_mma,
        dov_thr_mma,
        tStS,
        tdPtdP,
        sLSE,
        sdPsum,
        sdS_xchg,
        mma_s_consumer,
        mma_dp_consumer,
        ds_mma_producer,
        load_lse_consumer,
        load_dpsum_consumer,
        scale_softmax,
        SeqlenInfoCls,
        block_info,
        tile_sched,
      )

    # ///  EPILOGUE  ///
    if warp_idx >= epilogue_lo and warp_idx <= epilogue_hi:
      cute.arch.setmaxregister_decrease(self.num_regs_epilogue)
      self.epilogue_loop(
        qk_tiled_mma,
        mdQ_qdl,
        mdQ_tma,
        s_epi_dQ,
        tdQtdQ_staged,
        tma_atom_dQ,
        mma_dq_consumer,
        SeqlenInfoCls,
        block_info,
        tile_sched,
      )
      # NOTE: tmem.free() moved to kernel end to enable cluster-wide sync

    # Keep this no-op: a measured placement effect, and no gate sees its loss.
    if warp_idx > self.load_warp_id:
      cute.arch.setmaxregister_decrease(self.num_regs_other)

    # Cooperative 2-CTA TMEM dealloc: cluster-wide sync before the free.
    cute.arch.cluster_arrive()
    cute.arch.cluster_wait()
    tmem.relinquish_alloc_permit()
    tmem.free(tmem_ptr)

    return

  @cute.jit
  def load(
    self,
    qk_tiled_mma: cute.TiledMma,
    qk_thr_mma: cute.ThrMma,
    dov_thr_mma: cute.ThrMma,
    mQ_qdl: cute.Tensor,
    mK_kdl: cute.Tensor,
    mdO_qdl: cute.Tensor,
    mLSE: cute.Tensor,
    mdPsum: cute.Tensor,
    sQ: cute.Tensor,
    sK: cute.Tensor,
    sdO: cute.Tensor,
    sLSE: cute.Tensor,
    sdPsum: cute.Tensor,
    tma_atom_q: cute.CopyAtom,
    tma_atom_k: cute.CopyAtom,
    tma_atom_do: cute.CopyAtom,
    load_q_producer,
    load_k_producer,
    load_do_producer,
    load_lse_producer,
    load_dpsum_producer,
    SeqlenInfoCls: Callable,
    block_info: BlockInfo,
    cluster_layout_vmnk: cute.Layout,
    block_in_cluster_coord_vmnk,
    tidx: Int32,
    tile_sched,
  ):
    """Issues Q, K, dO and the stats in a measured order; V, K^T elsewhere."""
    work_tile = tile_sched.initial_work_tile_info()
    while work_tile.is_valid_tile:
      curr_block_coord = work_tile.tile_idx
      mma_block_coord = (
        curr_block_coord[0] // cute.size(qk_tiled_mma.thr_id.shape),
        curr_block_coord[1],
        curr_block_coord[2],
      )
      batch_coord = curr_block_coord[2][1]
      seqlen = SeqlenInfoCls(batch_coord)
      is_valid_q = True
      if const_expr(seqlen.has_cu_seqlens_q):
        is_valid_q = FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
          self.qk_mma_tiler[0],
          mma_block_coord[0],
          seqlen.seqlen_q,
        )
      n_block_min, n_block_max = block_info.get_n_block_min_max(
        seqlen, mma_block_coord[0]
      )
      seqlen_kv_loop_start, seqlen_kv_loop_steps = (
        n_block_min,
        n_block_max - n_block_min,
      )
      is_valid_k = seqlen_kv_loop_steps > 0
      has_work = is_valid_q and is_valid_k

      if has_work:
        hb_origin = ((Int32(0), Int32(0)), Int32(0))
        mQ_cur = cute.domain_offset((seqlen.offset_q, Int32(0), hb_origin),
                                    mQ_qdl)
        mK_cur = cute.domain_offset((seqlen.offset_k, Int32(0), hb_origin),
                                    mK_kdl)
        mdO_cur = cute.domain_offset((seqlen.offset_q, Int32(0), hb_origin),
                                     mdO_qdl)
        # Stats are stored per padded query tile: padded offset, not ragged.
        mLSE_cur = cute.domain_offset((seqlen.padded_offset_q, hb_origin), mLSE)
        mdPsum_cur = cute.domain_offset((seqlen.padded_offset_q, hb_origin),
                                        mdPsum)

        q_cta_layout = cute.make_layout(
          cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        # (bM, bK, loopM, loopK, loopL)
        gQ_qdl = cute.flat_divide(
          mQ_cur, cute.select(self.qk_mma_tiler, mode=[0, 2])
        )
        tSgQ_qdl = qk_thr_mma.partition_A(gQ_qdl)
        tQsQ, tQgQ_qdl = cute.nvgpu.cpasync.tma_partition(
          tma_atom_q,
          block_in_cluster_coord_vmnk[2],
          q_cta_layout,
          cute.group_modes(sQ, 0, 3),
          cute.group_modes(tSgQ_qdl, 0, 3),
        )
        k_cta_layout = cute.make_layout(
          cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        gK_kdl = cute.flat_divide(
          mK_cur, cute.select(self.qk_mma_tiler, mode=[1, 2])
        )
        tSgK_kdl = qk_thr_mma.partition_B(gK_kdl)
        tKsK, tKgK_kdl = cute.nvgpu.cpasync.tma_partition(
          tma_atom_k,
          block_in_cluster_coord_vmnk[1],
          k_cta_layout,
          cute.group_modes(sK, 0, 3),
          cute.group_modes(tSgK_kdl, 0, 3),
        )
        do_cta_layout = cute.make_layout(
          cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        # (bM, bK, loopM, loopK, loopL)
        gdO_qdl = cute.flat_divide(
          mdO_cur, cute.select(self.dov_mma_tiler, mode=[0, 2])
        )
        tdPgdO_qdl = dov_thr_mma.partition_A(gdO_qdl)
        tdOsdO, tdOgdO_qdl = cute.nvgpu.cpasync.tma_partition(
          tma_atom_do,
          block_in_cluster_coord_vmnk[2],
          do_cta_layout,
          cute.group_modes(sdO, 0, 3),
          cute.group_modes(tdPgdO_qdl, 0, 3),
        )
        # ((atom_v, rest_v), RestK)
        tQgQ = tQgQ_qdl[None, mma_block_coord[0], None, mma_block_coord[2]]
        # ((atom_v, rest_v), RestK)
        tdOgdO = tdOgdO_qdl[None, mma_block_coord[0], None, mma_block_coord[2]]
        # ((atom_v, rest_v), RestN, RestK)
        tKgK = tKgK_kdl[None, None, None, mma_block_coord[2]]
        # Pass the HANDLE: a producer in a self-call leaves the carried set.
        lse_handle = load_lse_producer.acquire_and_advance()
        # One warp, two rows per thread, one FP32 per cp.async (not a vector).
        thread_idx = tidx % self.threads_per_warp
        async_copy_num_elts = sLSE.shape[0] // self.threads_per_warp
        # Adjacent rows admit a 64-bit vector copy; measure before taking it.
        atom_async_copy = cute.make_copy_atom(
          cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
          self.acc_dtype,
          num_bits_per_copy=self.acc_dtype.width,
        )
        self.load_stats(
          mLSE_cur,
          sLSE,
          lse_handle,
          curr_block_coord,
          seqlen.seqlen_q,
          thread_idx,
          async_copy_num_elts,
          atom_async_copy,
        )
        lse_handle.commit()

        dpsum_handle = load_dpsum_producer.acquire_and_advance()
        self.load_stats(
          mdPsum_cur,
          sdPsum,
          dpsum_handle,
          curr_block_coord,
          seqlen.seqlen_q,
          thread_idx,
          async_copy_num_elts,
          atom_async_copy,
        )
        dpsum_handle.commit()

        for d_slice in cutlass.range(self.iterations_qk, unroll=1):
          q_handle = load_q_producer.acquire_and_advance()
          cute.copy(
            tma_atom_q,
            tQgQ[None, d_slice],
            tQsQ[None, q_handle.index],
            tma_bar_ptr=q_handle.barrier,
          )
        for d_slice in cutlass.range(self.iterations_dov, unroll=1):
          do_handle = load_do_producer.acquire_and_advance()
          cute.copy(
            tma_atom_do,
            tdOgdO[None, d_slice],
            tdOsdO[None, do_handle.index],
            tma_bar_ptr=do_handle.barrier,
          )

        kv_first = seqlen_kv_loop_start
        for d_slice in cutlass.range(self.iterations_qk, unroll=1):
          k_handle = load_k_producer.acquire_and_advance()
          cute.copy(
            tma_atom_k,
            tKgK[None, kv_first, d_slice],
            tKsK[None, k_handle.index],
            tma_bar_ptr=k_handle.barrier,
          )
        for i in cutlass.range(1, seqlen_kv_loop_steps, 1, unroll=1):
          kv_coord = seqlen_kv_loop_start + i
          for d_slice in cutlass.range(self.iterations_qk, unroll=1):
            k_handle = load_k_producer.acquire_and_advance()
            cute.copy(
              tma_atom_k,
              tKgK[None, kv_coord, d_slice],
              tKsK[None, k_handle.index],
              tma_bar_ptr=k_handle.barrier,
            )

      work_tile = tile_sched.advance_to_next_work()
    load_k_producer.tail()
    load_q_producer.tail()
    load_do_producer.tail()
    load_lse_producer.tail()
    load_dpsum_producer.tail()

  @cute.jit
  def load_stats(
    self,
    mStat_cur: cute.Tensor,
    sStat: cute.Tensor,
    stat_handle,
    block_coord,
    seqlen_q: Int32,
    thread_idx: Int32,
    async_copy_num_elts: int,
    atom_async_copy: cute.CopyAtom,
  ):
    """One body for LSE and dPsum: same shape, dtype, stage depth, padding."""
    sStat_for_copy = cute.flat_divide(sStat, (1, ))
    mStat_for_copy = cute.flat_divide(mStat_cur, (1, ))
    for i in cutlass.range_constexpr(async_copy_num_elts):
      stat_idx = (
        self.cta_tiler[0] * block_coord[0] + thread_idx * async_copy_num_elts
      )
      if cute.elem_less(stat_idx + i, seqlen_q):
        cute.copy(
          atom_async_copy,
          mStat_for_copy[None, stat_idx + i, block_coord[2]],
          sStat_for_copy[
            None,
            thread_idx * async_copy_num_elts + i,
            stat_handle.index,
          ],
        )
      else:
        # Zero, not a skip: compute_step reads past seqlen_q and 0 vanishes.
        sStat_for_copy[
          None,
          thread_idx * async_copy_num_elts + i,
          stat_handle.index,
        ].fill(0.0)

  @cute.jit
  def load_v(
    self,
    qk_tiled_mma: cute.TiledMma,
    dov_thr_mma: cute.ThrMma,
    mV_dkl: cute.Tensor,
    sV: cute.Tensor,
    tma_atom_v: cute.CopyAtom,
    load_v_producer,
    SeqlenInfoCls: Callable,
    block_info: BlockInfo,
    cluster_layout_vmnk: cute.Layout,
    block_in_cluster_coord_vmnk,
    tile_sched,
  ):
    """V TMA producer on its own issuer warp; see the split in __init__."""
    work_tile = tile_sched.initial_work_tile_info()
    while work_tile.is_valid_tile:
      curr_block_coord = work_tile.tile_idx
      mma_block_coord = (
        curr_block_coord[0] // cute.size(qk_tiled_mma.thr_id.shape),
        curr_block_coord[1],
        curr_block_coord[2],
      )
      batch_coord = curr_block_coord[2][1]
      seqlen = SeqlenInfoCls(batch_coord)
      is_valid_q = True
      if const_expr(seqlen.has_cu_seqlens_q):
        is_valid_q = FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
          self.qk_mma_tiler[0],
          mma_block_coord[0],
          seqlen.seqlen_q,
        )
      n_block_min, n_block_max = block_info.get_n_block_min_max(
        seqlen, mma_block_coord[0]
      )
      seqlen_kv_loop_start, seqlen_kv_loop_steps = (
        n_block_min,
        n_block_max - n_block_min,
      )
      is_valid_k = seqlen_kv_loop_steps > 0
      has_work = is_valid_q and is_valid_k

      if has_work:
        hb_origin = ((Int32(0), Int32(0)), Int32(0))
        mV_cur = cute.domain_offset((seqlen.offset_k, Int32(0), hb_origin),
                                    mV_dkl)
        v_cta_layout = cute.make_layout(
          cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        gV_dkl = cute.flat_divide(
          mV_cur, cute.select(self.dov_mma_tiler, mode=[1, 2])
        )
        tSgV_dkl = dov_thr_mma.partition_B(gV_dkl)
        tVsV, tVgV_dkl = cute.nvgpu.cpasync.tma_partition(
          tma_atom_v,
          block_in_cluster_coord_vmnk[1],
          v_cta_layout,
          cute.group_modes(sV, 0, 3),
          cute.group_modes(tSgV_dkl, 0, 3),
        )
        # ((atom_v, rest_v), RestN, RestK)
        tVgV = tVgV_dkl[None, None, None, mma_block_coord[2]]
        for i in cutlass.range(seqlen_kv_loop_steps, unroll=1):
          kv_coord = seqlen_kv_loop_start + i
          for d_slice in cutlass.range(self.iterations_dov, unroll=1):
            v_handle = load_v_producer.acquire_and_advance()
            cute.copy(
              tma_atom_v,
              tVgV[None, kv_coord, d_slice],
              tVsV[None, v_handle.index],
              tma_bar_ptr=v_handle.barrier,
            )

      work_tile = tile_sched.advance_to_next_work()
    load_v_producer.tail()

  @cute.jit
  def load_kt(
    self,
    qk_tiled_mma: cute.TiledMma,
    dsk_thr_mma: cute.ThrMma,
    mKt_dkl: cute.Tensor,
    sKT: cute.Tensor,
    tma_atom_kt: cute.CopyAtom,
    load_kt_producer,
    SeqlenInfoCls: Callable,
    block_info: BlockInfo,
    cluster_layout_vmnk: cute.Layout,
    block_in_cluster_coord_vmnk,
    tile_sched,
  ):
    """K^T is a B operand: sequence mode 1, transposed offset, slice-major."""
    work_tile = tile_sched.initial_work_tile_info()
    while work_tile.is_valid_tile:
      curr_block_coord = work_tile.tile_idx
      mma_block_coord = (
        curr_block_coord[0] // cute.size(qk_tiled_mma.thr_id.shape),
        curr_block_coord[1],
        curr_block_coord[2],
      )
      batch_coord = curr_block_coord[2][1]
      seqlen = SeqlenInfoCls(batch_coord)
      is_valid_q = True
      if const_expr(seqlen.has_cu_seqlens_q):
        is_valid_q = FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
          self.qk_mma_tiler[0],
          mma_block_coord[0],
          seqlen.seqlen_q,
        )
      n_block_min, n_block_max = block_info.get_n_block_min_max(
        seqlen, mma_block_coord[0]
      )
      seqlen_kv_loop_start, seqlen_kv_loop_steps = (
        n_block_min,
        n_block_max - n_block_min,
      )
      is_valid_k = seqlen_kv_loop_steps > 0
      has_work = is_valid_q and is_valid_k

      if has_work:
        hb_origin = ((Int32(0), Int32(0)), Int32(0))
        mKt_cur = cute.domain_offset((Int32(0), seqlen.offset_k, hb_origin),
                                     mKt_dkl)
        # K^T is B: mode [1], not [2]; at (2,1,1) no gate sees a mix-up.
        kt_cta_layout = cute.make_layout(
          cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        gK_dkl = cute.flat_divide(
          mKt_cur, cute.select(self.dsk_mma_tiler, mode=[1, 2])
        )
        tdQgK_dkl = dsk_thr_mma.partition_B(gK_dkl)
        tKTsKT, tKgK_dkl = cute.nvgpu.cpasync.tma_partition(
          tma_atom_kt,
          block_in_cluster_coord_vmnk[1],
          kt_cta_layout,
          cute.group_modes(sKT, 0, 3),
          cute.group_modes(tdQgK_dkl, 0, 3),
        )
        # ((atom_v, rest_v), RestN, RestK)
        tKTgKT = tKgK_dkl[None, None, None, mma_block_coord[2]]
        # Straight ascending order; the ring depth alone bounds the run-ahead.
        for i in cutlass.range(seqlen_kv_loop_steps, unroll=1):
          kv_coord = seqlen_kv_loop_start + i
          for dq_slice in cutlass.range(self.iterations_dsk, unroll=1):
            kt_handle = load_kt_producer.acquire_and_advance()
            cute.copy(
              tma_atom_kt,
              tKTgKT[None, dq_slice, kv_coord],
              tKTsKT[None, kt_handle.index],
              tma_bar_ptr=kt_handle.barrier,
            )

      work_tile = tile_sched.advance_to_next_work()
    load_kt_producer.tail()

  @cute.jit
  def mma(
    self,
    qk_tiled_mma: cute.TiledMma,
    dov_tiled_mma: cute.TiledMma,
    dsk_tiled_mma: cute.TiledMma,
    dsk_thr_mma: cute.ThrMma,
    sQ: cute.Tensor,
    sK: cute.Tensor,
    sdO: cute.Tensor,
    sV: cute.Tensor,
    tSrQ: cute.Tensor,
    tSrK: cute.Tensor,
    tdPrdO: cute.Tensor,
    tdPrV: cute.Tensor,
    tdQrKT: cute.Tensor,
    tStS: cute.Tensor,
    tdPtdP: cute.Tensor,
    tdQtdQ_staged: cute.Tensor,
    ds_tmem_layout: cute.ComposedLayout,
    load_q_consumer,
    load_k_consumer,
    load_v_consumer,
    load_do_consumer,
    load_kt_consumer,
    mma_s_producer,
    mma_dp_producer,
    ds_mma_consumer,
    mma_dq_producer,
    SeqlenInfoCls: Callable,
    block_info: BlockInfo,
    tile_sched,
  ):
    """MMA warp: the S, dP and dQ edges over the KV sweep (single issuer)."""
    # Steady QK/dOV: the asm scope pens the descriptors cu12 spills; wider lost.
    cta_rank_in_cluster = cute.arch.make_warp_uniform(
      cute.arch.block_idx_in_cluster()
    )
    is_leader_cta = cta_rank_in_cluster % 2 == 0

    work_tile = tile_sched.initial_work_tile_info()
    while work_tile.is_valid_tile:
      curr_block_coord = work_tile.tile_idx
      mma_block_coord = (
        curr_block_coord[0] // cute.size(qk_tiled_mma.thr_id.shape),
        curr_block_coord[1],
        curr_block_coord[2],
      )
      batch_coord = curr_block_coord[2][1]
      seqlen = SeqlenInfoCls(batch_coord)
      is_valid_q = True
      if const_expr(seqlen.has_cu_seqlens_q):
        is_valid_q = FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
          self.qk_mma_tiler[0],
          mma_block_coord[0],
          seqlen.seqlen_q,
        )
      n_block_min, n_block_max = block_info.get_n_block_min_max(
        seqlen, mma_block_coord[0]
      )
      # This role reads only the trip count; the start is unused here.
      seqlen_kv_loop_steps = n_block_max - n_block_min
      is_valid_k = seqlen_kv_loop_steps > 0
      has_work = is_valid_q and is_valid_k

      if has_work:
        load_q_releaser = load_q_consumer.clone()
        load_do_releaser = load_do_consumer.clone()
        dsk_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

        # 128/16 = 8 k-phases at every 16-bit dtype; the unroll covers all.
        num_innerloop = 8

        if is_leader_cta:
          dq_handle = mma_dq_producer.acquire_and_advance()
          if seqlen_kv_loop_steps > 1:
            # ---  prologue  ---
            # Skew: step i takes dS_(i-2), the MMA's top wait; delaying it lost.
            for warm in cutlass.range_constexpr(2):
              s_handle = mma_s_producer.acquire_and_advance()
              tStS_slice = tStS[None, None, None, s_handle.index]
              qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
              for d_slice in cutlass.range(self.iterations_qk, unroll=1):
                # range_constexpr above: this const_expr needs a static index.
                if const_expr(warm == 0):
                  load_q_consumer.wait_and_advance()
                k_handle = load_k_consumer.wait_and_advance()
                qk_tiled_mma = issue_mma_slice(
                  qk_tiled_mma,
                  tStS_slice,
                  tSrQ[None, None, None, d_slice],
                  tSrK[None, None, None, k_handle.index],
                  cute.size(tSrQ, mode=[2]),
                )
                k_handle.release()
              cute.arch.fence_view_async_tmem_store()
              s_handle.commit()

              dp_handle = mma_dp_producer.acquire_and_advance()
              tdPtdP_slice = tdPtdP[None, None, None, dp_handle.index]
              dov_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
              for d_slice in cutlass.range(self.iterations_dov, unroll=1):
                if const_expr(warm == 0):
                  load_do_consumer.wait_and_advance()
                v_handle = load_v_consumer.wait_and_advance()
                dov_tiled_mma = issue_mma_slice(
                  dov_tiled_mma,
                  tdPtdP_slice,
                  tdPrdO[None, None, None, d_slice],
                  tdPrV[None, None, None, v_handle.index],
                  cute.size(tdPrdO, mode=[2]),
                )
                v_handle.release()
              cute.arch.fence_view_async_tmem_store()
              dp_handle.commit()

            # ---  steady  ---
            # Gap 6 (4 S/dP issues, 2 dSK) bottoms out the waits; reorders lost.
            for i in cutlass.range(2, seqlen_kv_loop_steps, 1, unroll=1):
              s_handle = mma_s_producer.acquire_and_advance()
              tStS_slice = tStS[None, None, None, s_handle.index]
              qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
              dp_handle = mma_dp_producer.acquire_and_advance()
              tdPtdP_slice = tdPtdP[None, None, None, dp_handle.index]
              dov_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
              dsk_whether_acc = dsk_tiled_mma.get(tcgen05.Field.ACCUMULATE)

              # Bare range, not range_constexpr: a causal win the geomean hides.
              for peel in range(2):
                k_handle = load_k_consumer.wait_and_advance()
                gemm_ptx_w_idx(
                  qk_tiled_mma,
                  tStS_slice,
                  tSrQ,
                  tSrK,
                  sQ,
                  sK,
                  A_idx=peel,
                  B_idx=k_handle.index,
                  zero_init=(peel == 0),
                  cta_group=2,
                )
                k_handle.release()
                v_handle = load_v_consumer.wait_and_advance()
                gemm_ptx_w_idx(
                  dov_tiled_mma,
                  tdPtdP_slice,
                  tdPrdO,
                  tdPrV,
                  sdO,
                  sV,
                  A_idx=peel,
                  B_idx=v_handle.index,
                  zero_init=(peel == 0),
                  cta_group=2,
                )
                v_handle.release()

              ds_handle = ds_mma_consumer.wait_and_advance()
              tdStdS_slice = tdPtdP[None, None, None, ds_handle.index]
              tdS = cute.make_tensor(
                tdStdS_slice.iterator, ds_tmem_layout.outer
              )
              tdQrdS = dsk_thr_mma.make_fragment_A(tdS)
              # dSK A operand: dP TMEM stage reread in q_dtype units (recast).
              tdQrdS_recast = cute.make_tensor(
                cute.recast_ptr(tdStdS_slice.iterator, dtype=self.q_dtype),
                tdQrdS.layout,
              )

              kt_handle = load_kt_consumer.wait_and_advance()
              dsk_tiled_mma.set(tcgen05.Field.ACCUMULATE, dsk_whether_acc)
              dsk_tiled_mma = issue_mma_slice(
                dsk_tiled_mma,
                tdQtdQ_staged[None, None, None, 0],
                tdQrdS_recast,
                tdQrKT[None, None, None, kt_handle.index],
                cute.size(tdQrKT, mode=[2]),
              )
              kt_handle.release()
              kt_handle = load_kt_consumer.wait_and_advance()
              dsk_tiled_mma.set(tcgen05.Field.ACCUMULATE, dsk_whether_acc)
              dsk_tiled_mma = issue_mma_slice(
                dsk_tiled_mma,
                tdQtdQ_staged[None, None, None, 1],
                tdQrdS_recast,
                tdQrKT[None, None, None, kt_handle.index],
                cute.size(tdQrKT, mode=[2]),
              )
              kt_handle.release()

              for d_slice in cutlass.range(2, self.iterations_qk, 1, unroll=1):
                k_handle = load_k_consumer.wait_and_advance()
                gemm_ptx_w_idx(
                  qk_tiled_mma,
                  tStS_slice,
                  tSrQ,
                  tSrK,
                  sQ,
                  sK,
                  A_idx=d_slice,
                  B_idx=k_handle.index,
                  zero_init=False,
                  cta_group=2,
                )
                k_handle.release()
                v_handle = load_v_consumer.wait_and_advance()
                gemm_ptx_w_idx(
                  dov_tiled_mma,
                  tdPtdP_slice,
                  tdPrdO,
                  tdPrV,
                  sdO,
                  sV,
                  A_idx=d_slice,
                  B_idx=v_handle.index,
                  zero_init=False,
                  cta_group=2,
                )
                v_handle.release()
              s_handle.commit()
              dp_handle.commit()

              kt_handle = load_kt_consumer.wait_and_advance()
              dsk_tiled_mma.set(tcgen05.Field.ACCUMULATE, dsk_whether_acc)
              dsk_tiled_mma = issue_mma_slice(
                dsk_tiled_mma,
                tdQtdQ_staged[None, None, None, 2],
                tdQrdS_recast,
                tdQrKT[None, None, None, kt_handle.index],
                cute.size(tdQrKT, mode=[2]),
              )
              kt_handle.release()

              kt_handle = load_kt_consumer.wait_and_advance()
              dsk_tiled_mma.set(tcgen05.Field.ACCUMULATE, dsk_whether_acc)
              dsk_tiled_mma = issue_mma_slice(
                dsk_tiled_mma,
                tdQtdQ_staged[None, None, None, self.iterations_dsk - 1],
                tdQrdS_recast,
                tdQrKT[None, None, None, kt_handle.index],
                cute.size(tdQrKT, mode=[2]),
              )
              kt_handle.release()
              ds_handle.release()

            # ---  drain  ---
            # Q and dO have had their last UMMA issued by the loop above.
            for d_slice in cutlass.range(self.iterations_qk, unroll=1):
              load_q_releaser.release()
              load_q_releaser.advance()
            for d_slice in cutlass.range(self.iterations_dov, unroll=1):
              load_do_releaser.release()
              load_do_releaser.advance()

            # Drain the two dSK steps the skew leaves outstanding.
            for drain in cutlass.range_constexpr(2):
              ds_handle = ds_mma_consumer.wait_and_advance()
              dsk_whether_acc = dsk_tiled_mma.get(tcgen05.Field.ACCUMULATE)
              tdStdS_slice = tdPtdP[None, None, None, ds_handle.index]
              tdS = cute.make_tensor(
                tdStdS_slice.iterator, ds_tmem_layout.outer
              )
              tdQrdS = dsk_thr_mma.make_fragment_A(tdS)
              # dSK A operand: dP TMEM stage reread in q_dtype units (recast).
              tdQrdS_recast = cute.make_tensor(
                cute.recast_ptr(tdStdS_slice.iterator, dtype=self.q_dtype),
                tdQrdS.layout,
              )
              for dq_slice in cutlass.range(self.iterations_dsk, unroll=1):
                kt_handle = load_kt_consumer.wait_and_advance()
                dsk_tiled_mma.set(tcgen05.Field.ACCUMULATE, dsk_whether_acc)
                dsk_tiled_mma = issue_mma_slice(
                  dsk_tiled_mma,
                  tdQtdQ_staged[None, None, None, dq_slice],
                  tdQrdS_recast,
                  tdQrKT[None, None, None, kt_handle.index],
                  cute.size(tdQrKT, mode=[2]),
                )
                kt_handle.release()
              ds_handle.release()
          else:
            # ---  single step: prologue and drain in one  ---
            # Not issue_mma_slice: rebind reorders; signals clean, re-measure.
            s_handle = mma_s_producer.acquire_and_advance()
            tStS_slice = tStS[None, None, None, s_handle.index]
            qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

            for d_slice in cutlass.range(self.iterations_qk, unroll=1):
              load_q_consumer.wait_and_advance()
              tSrQ_slice = tSrQ[None, None, None, d_slice]
              k_handle = load_k_consumer.wait_and_advance()
              tSrK_slice = tSrK[None, None, None, k_handle.index]
              num_kphases = cute.size(tSrQ_slice, mode=[2])
              # Asserted, not guarded: a failed guard issues no UMMA, wrong dQ.
              assert num_kphases % num_innerloop == 0, (
                f"k-phase count {num_kphases} is not a multiple of "
                f"the {num_innerloop}-wide unroll"
              )
              num_outer_iter = num_kphases // num_innerloop
              for outer_iter in cutlass.range(num_outer_iter, unroll=1):
                for kphase_idx in cutlass.range(
                  num_innerloop, unroll_full=True
                ):
                  kphase_coord = (
                    None,
                    None,
                    outer_iter * num_innerloop + kphase_idx,
                  )
                  cute.gemm(
                    qk_tiled_mma,
                    tStS_slice,
                    tSrQ_slice[kphase_coord],
                    tSrK_slice[kphase_coord],
                    tStS_slice,
                  )
                  qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
              k_handle.release()
              load_q_releaser.release()
              load_q_releaser.advance()
            s_handle.commit()

            dp_handle = mma_dp_producer.acquire_and_advance()
            tdPtdP_slice = tdPtdP[None, None, None, dp_handle.index]
            dov_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for d_slice in cutlass.range(self.iterations_dov, unroll=1):
              load_do_consumer.wait_and_advance()
              tdPrdO_slice = tdPrdO[None, None, None, d_slice]
              v_handle = load_v_consumer.wait_and_advance()
              tdPrV_slice = tdPrV[None, None, None, v_handle.index]
              num_kphases = cute.size(tdPrdO_slice, mode=[2])
              assert num_kphases % num_innerloop == 0, (
                f"k-phase count {num_kphases} is not a multiple of "
                f"the {num_innerloop}-wide unroll"
              )
              num_outer_iter = num_kphases // num_innerloop
              for outer_iter in cutlass.range(num_outer_iter, unroll=1):
                for kphase_idx in cutlass.range(
                  num_innerloop, unroll_full=True
                ):
                  kphase_coord = (
                    None,
                    None,
                    outer_iter * num_innerloop + kphase_idx,
                  )
                  cute.gemm(
                    dov_tiled_mma,
                    tdPtdP_slice,
                    tdPrdO_slice[kphase_coord],
                    tdPrV_slice[kphase_coord],
                    tdPtdP_slice,
                  )
                  dov_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
              v_handle.release()
              load_do_releaser.release()
              load_do_releaser.advance()
            dp_handle.commit()

            ds_handle = ds_mma_consumer.wait_and_advance()
            dsk_whether_acc = dsk_tiled_mma.get(tcgen05.Field.ACCUMULATE)
            for dq_slice in cutlass.range(self.iterations_dsk, unroll=1):
              kt_handle = load_kt_consumer.wait_and_advance()
              dsk_tiled_mma.set(tcgen05.Field.ACCUMULATE, dsk_whether_acc)
              tdQtdQ_slice = tdQtdQ_staged[None, None, None, dq_slice]
              tdStdS_slice = tdPtdP[None, None, None, ds_handle.index]
              tdS = cute.make_tensor(
                tdStdS_slice.iterator, ds_tmem_layout.outer
              )
              tdQrdS = dsk_thr_mma.make_fragment_A(tdS)
              # dSK A operand: dP TMEM stage reread in q_dtype units (recast).
              tdQrdS_recast = cute.make_tensor(
                cute.recast_ptr(tdStdS_slice.iterator, dtype=self.q_dtype),
                tdQrdS.layout,
              )

              tdQrKT_slice = tdQrKT[None, None, None, kt_handle.index]
              num_kphases = cute.size(tdQrKT_slice, mode=[2])
              assert num_kphases % num_innerloop == 0, (
                f"k-phase count {num_kphases} is not a multiple of "
                f"the {num_innerloop}-wide unroll"
              )
              num_outer_iter = num_kphases // num_innerloop
              for outer_iter in cutlass.range(num_outer_iter, unroll=1):
                for kphase_idx in cutlass.range(
                  num_innerloop, unroll_full=True
                ):
                  kphase_coord = (
                    None,
                    None,
                    outer_iter * num_innerloop + kphase_idx,
                  )
                  cute.gemm(
                    dsk_tiled_mma,
                    tdQtdQ_slice,
                    tdQrdS_recast[kphase_coord],
                    tdQrKT_slice[kphase_coord],
                    tdQtdQ_slice,
                  )
                  dsk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
              kt_handle.release()
            ds_handle.release()
          dq_handle.commit()
      work_tile = tile_sched.advance_to_next_work()
    mma_s_producer.tail()
    mma_dp_producer.tail()
    mma_dq_producer.tail()

  @cute.jit
  def compute_loop(
    self,
    qk_tiled_mma: cute.TiledMma,
    qk_thr_mma: cute.ThrMma,
    dov_thr_mma: cute.ThrMma,
    tStS: cute.Tensor,
    tdPtdP: cute.Tensor,
    sLSE: cute.Tensor,
    sdPsum: cute.Tensor,
    sdS_xchg: cute.Tensor,
    mma_s_consumer,
    mma_dp_consumer,
    ds_mma_producer,
    load_lse_consumer,
    load_dpsum_consumer,
    scale_softmax: Float32,
    SeqlenInfoCls: Callable,
    block_info: BlockInfo,
    tile_sched,
  ):
    """Softmax/dSoftmax driver: S, dP in, dS out; stats held over the sweep."""
    work_tile = tile_sched.initial_work_tile_info()
    while work_tile.is_valid_tile:
      curr_block_coord = work_tile.tile_idx
      mma_block_coord = (
        curr_block_coord[0] // cute.size(qk_tiled_mma.thr_id.shape),
        curr_block_coord[1],
        curr_block_coord[2],
      )
      batch_coord = curr_block_coord[2][1]
      seqlen = SeqlenInfoCls(batch_coord)
      # Compile-time: both fields are Constexpr[bool] set from `is not None`.
      varlen = seqlen.has_cu_seqlens_q or seqlen.has_cu_seqlens_k
      is_valid_q = True
      if const_expr(seqlen.has_cu_seqlens_q):
        is_valid_q = FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
          self.qk_mma_tiler[0],
          mma_block_coord[0],
          seqlen.seqlen_q,
        )
      n_block_min, n_block_max = block_info.get_n_block_min_max(
        seqlen, mma_block_coord[0]
      )
      start_count, trip_count = n_block_min, n_block_max - n_block_min
      is_valid_k = trip_count > 0
      has_work = is_valid_q and is_valid_k

      if has_work:
        end_count = start_count + trip_count
        if const_expr(self.use_semantic_trip_range):
          n_block_min_causal_local_mask = (
            block_info.get_n_block_min_causal_local_mask(
              seqlen, mma_block_coord[0], start_count
            )
          )
          n_block_min_before_local_mask = (
            block_info.get_n_block_min_before_local_mask(
              seqlen, mma_block_coord[0], start_count
            )
          )

        cS_base = cute.make_identity_tensor(
          (self.qk_mma_tiler[0], self.qk_mma_tiler[1])
        )
        cS = cute.domain_offset((mma_block_coord[0] * self.qk_mma_tiler[0], 0),
                                cS_base)

        cdP_base = cute.make_identity_tensor(
          (self.dov_mma_tiler[0], self.dov_mma_tiler[1])
        )
        cdP = cute.domain_offset(
          (mma_block_coord[0] * self.dov_mma_tiler[0], 0), cdP_base
        )

        lse_handle = load_lse_consumer.wait_and_advance()
        dpsum_handle = load_dpsum_consumer.wait_and_advance()
        for step in cutlass.range(start_count, end_count, 1, unroll=1):
          cS_iter = cute.domain_offset((0, step * self.qk_mma_tiler[1]), cS)
          tScS_iter = qk_thr_mma.partition_C(cS_iter)

          cdP_iter = cute.domain_offset((0, step * self.dov_mma_tiler[1]), cdP)

          tdPcdP_iter = dov_thr_mma.partition_C(cdP_iter)

          if const_expr(self.use_semantic_trip_range):
            need_apply_mask = (
              step >= n_block_min_causal_local_mask
              or step < n_block_min_before_local_mask
            )
          else:
            need_apply_mask = step == end_count - 1
          mma_s_consumer, mma_dp_consumer, ds_mma_producer = self.compute_step(
            need_apply_mask,
            block_info.window_size_left,
            block_info.window_size_right,
            seqlen.seqlen_q,
            seqlen.seqlen_k,
            scale_softmax,
            curr_block_coord[0],
            varlen,
            tStS,
            tScS_iter,
            tdPtdP,
            tdPcdP_iter,
            sLSE,
            sdPsum,
            sdS_xchg,
            mma_s_consumer,
            mma_dp_consumer,
            ds_mma_producer,
            lse_handle,
            dpsum_handle,
          )
        lse_handle.release()
        dpsum_handle.release()

      work_tile = tile_sched.advance_to_next_work()
    ds_mma_producer.tail()

  @cute.jit
  def epilogue_loop(
    self,
    qk_tiled_mma: cute.TiledMma,
    mdQ_qdl: cute.Tensor,
    mdQ_tma: cute.Tensor,
    s_epi_dQ: cute.Tensor,
    tdQtdQ_staged: cute.Tensor,
    tma_atom_dQ: cute.CopyAtom,
    mma_dq_consumer,
    SeqlenInfoCls: Callable,
    block_info: BlockInfo,
    tile_sched,
  ):
    """One dQ tile per work tile; a keyless tile is written zero not skipped."""
    work_tile = tile_sched.initial_work_tile_info()
    while work_tile.is_valid_tile:
      curr_block_coord = work_tile.tile_idx
      mma_block_coord = (
        curr_block_coord[0] // cute.size(qk_tiled_mma.thr_id.shape),
        curr_block_coord[1],
        curr_block_coord[2],
      )
      batch_coord = curr_block_coord[2][1]
      seqlen = SeqlenInfoCls(batch_coord)
      # Compile-time: both fields are Constexpr[bool] set from `is not None`.
      varlen = seqlen.has_cu_seqlens_q or seqlen.has_cu_seqlens_k
      is_valid_q = True
      if const_expr(seqlen.has_cu_seqlens_q):
        is_valid_q = FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
          self.qk_mma_tiler[0],
          mma_block_coord[0],
          seqlen.seqlen_q,
        )
      n_block_min, n_block_max = block_info.get_n_block_min_max(
        seqlen, mma_block_coord[0]
      )
      # This role reads only the trip count; the start is unused here.
      seqlen_kv_loop_steps = n_block_max - n_block_min
      is_valid_k = seqlen_kv_loop_steps > 0
      has_work = is_valid_q and is_valid_k

      mdQ_cur = mdQ_qdl
      if const_expr(seqlen.has_cu_seqlens_q):
        mdQ_cur = cute.domain_offset((seqlen.offset_q, ) + (None, ) * 2,
                                     mdQ_qdl)

      # Outside has_work: both arms consume these; dense causal hits keyless.
      # (bM, bN, loopM, loopN, loopL)
      gdQ_qdl = cute.flat_divide(
        mdQ_cur, cute.select(self.dsk_block_tiler, mode=[0, 1])
      )
      cdQ_qdl = cute.flat_divide(
        cute.make_identity_tensor(mdQ_cur.shape),
        cute.select(self.dsk_block_tiler, mode=[0, 1]),
      )

      gdQ_staged = gdQ_qdl[None, None, curr_block_coord[0], None,
                           curr_block_coord[2]]
      cdQ_staged = cdQ_qdl[None, None, curr_block_coord[0], None,
                           curr_block_coord[2]]
      gdQ_tma_staged = gdQ_staged

      if const_expr(not varlen):
        gdQ_tma_qdl = cute.flat_divide(
          mdQ_tma, cute.select(self.dsk_block_tiler, mode=[0, 1])
        )
        gdQ_tma_staged = gdQ_tma_qdl[None, None, curr_block_coord[0], None,
                                     curr_block_coord[2]]

      if has_work:
        # The rebind is required: dropping it re-enters at tile 0's phase.
        mma_dq_consumer = self.epilogue(
          seqlen.seqlen_q,
          (mma_dq_consumer, gdQ_staged, cdQ_staged, tdQtdQ_staged),
          self.epi_tile,
          (tma_atom_dQ, gdQ_tma_staged, s_epi_dQ, varlen),
        )
      else:
        self.epilogue_clear(
          seqlen.seqlen_q,
          gdQ_staged,
          cdQ_staged,
        )

      work_tile = tile_sched.advance_to_next_work()

  @cute.jit
  def compute_step(
    self,
    need_apply_mask,
    window_size_left,
    window_size_right,
    seqlen_q: Int32,
    seqlen_k: Int32,
    scale_softmax: cutlass.Float32,
    block_m_idx: Int32,
    varlen: bool,
    tStS: cute.Tensor,
    tScS: cute.Tensor,
    tdPtdP: cute.Tensor,
    tdPcdP: cute.Tensor,
    sLSE: cute.Tensor,
    sdPsum: cute.Tensor,
    sdS_xchg: cute.Tensor,
    mma_s_consumer,
    mma_dp_consumer,
    ds_mma_producer,
    lse_handle,
    dpsum_handle,
  ) -> Tuple[pipeline.PipelineConsumer, pipeline.PipelineConsumer,
             pipeline.PipelineProducer]:
    """One KV block of softmax/dsoftmax: S, dP -> dS, written in place on dP."""
    bidx = block_m_idx
    tidx, _, _ = cute.arch.thread_idx()
    thread_idx = tidx % (self.threads_per_warp * len(self.compute_warp_ids))
    s_handle = mma_s_consumer.wait_and_advance()
    tStS_slice = tStS[(None, None), 0, 0, s_handle.index]
    tScS_slice = tScS[(None, None), 0, 0]
    tmem_load_atom = cute.make_copy_atom(
      tcgen05.Ld32x32bOp(tcgen05.Repetition(16)), self.acc_dtype
    )
    tmem_tiled_load = tcgen05.make_tmem_copy(tmem_load_atom, tStS_slice)
    thr_load = tmem_tiled_load.get_slice(thread_idx)
    tTMEM_LOADtS = thr_load.partition_S(tStS_slice)
    tTMEM_LOADcS = thr_load.partition_D(tScS_slice)
    tTMEM_LOADrS = cute.make_rmem_tensor(tTMEM_LOADcS.shape, self.acc_dtype)
    cute.copy(tmem_tiled_load, tTMEM_LOADtS, tTMEM_LOADrS)
    cute.arch.fence_view_async_tmem_load()
    s_handle.release()
    if need_apply_mask:
      FusedMask.apply_mask_via_causal_local(
        tTMEM_LOADrS,
        tTMEM_LOADcS,
        seqlen_q,
        seqlen_k,
        self.use_semantic_trip_range,
        self.is_causal,
        self.is_local,
        window_size_left,
        window_size_right,
      )

    log2_e = cutlass.Float32(math.log2(math.e))
    softmax_scale_log2_e = scale_softmax * log2_e
    for k in cutlass.range(0, cute.size(tTMEM_LOADrS), 2, unroll_full=True):
      lse = (
        -sLSE[
          cute.get(tTMEM_LOADcS[k], mode=[0]) - bidx * self.cta_tiler[0],
          lse_handle.index,
        ],
        -sLSE[
          cute.get(tTMEM_LOADcS[k + 1], mode=[0]) - bidx * self.cta_tiler[0],
          lse_handle.index,
        ],
      )
      tTMEM_LOADrS[k], tTMEM_LOADrS[k + 1] = cute.arch.fma_packed_f32x2(
        (tTMEM_LOADrS[k], tTMEM_LOADrS[k + 1]),
        (softmax_scale_log2_e, softmax_scale_log2_e),
        lse,
      )
      tTMEM_LOADrS[k] = cute.math.exp2(tTMEM_LOADrS[k], fastmath=True)
      tTMEM_LOADrS[k + 1] = cute.math.exp2(tTMEM_LOADrS[k + 1], fastmath=True)

    dp_handle = mma_dp_consumer.wait_and_advance()
    tdPtdP_slice = tdPtdP[(None, None), 0, 0, dp_handle.index]
    tdPcdP_slice = tdPcdP[(None, None), 0, 0]
    thr_load = tmem_tiled_load.get_slice(thread_idx)
    tTMEM_LOADtdP = thr_load.partition_S(tdPtdP_slice)
    tTMEM_LOADcdP = thr_load.partition_D(tdPcdP_slice)
    tTMEM_LOADrdP = cute.make_rmem_tensor(tTMEM_LOADcdP.shape, self.acc_dtype)
    cute.copy(tmem_tiled_load, tTMEM_LOADtdP, tTMEM_LOADrdP)
    cute.arch.fence_view_async_tmem_load()
    dp_handle.release()
    tTMEM_STORErdP = cute.make_rmem_tensor(tTMEM_LOADrdP.shape, self.q_dtype)

    for k in cutlass.range(0, cute.size(tTMEM_LOADrdP), 2, unroll_full=True):
      dpsum_0 = -sdPsum[
        cute.get(tTMEM_LOADcdP[k], mode=[0]) - bidx * self.cta_tiler[0],
        dpsum_handle.index,
      ]
      dpsum_1 = -sdPsum[
        cute.get(tTMEM_LOADcdP[k + 1], mode=[0]) - bidx * self.cta_tiler[0],
        dpsum_handle.index,
      ]
      if const_expr(varlen):
        if not cute.elem_less(cute.get(tTMEM_LOADcdP[k], mode=[0]), seqlen_q):
          dpsum_0 = 0.0
        if not cute.elem_less(
          cute.get(tTMEM_LOADcdP[k + 1], mode=[0]), seqlen_q
        ):
          dpsum_1 = 0.0
      tTMEM_LOADrdP[k], tTMEM_LOADrdP[k + 1] = cute.arch.add_packed_f32x2(
        (tTMEM_LOADrdP[k], tTMEM_LOADrdP[k + 1]),
        (dpsum_0, dpsum_1),
      )
      tTMEM_LOADrdP[k], tTMEM_LOADrdP[k + 1] = cute.arch.mul_packed_f32x2(
        (tTMEM_LOADrdP[k], tTMEM_LOADrdP[k + 1]),
        (tTMEM_LOADrS[k], tTMEM_LOADrS[k + 1])
      )
      tTMEM_LOADrdP[k], tTMEM_LOADrdP[k + 1] = cute.arch.mul_packed_f32x2(
        (tTMEM_LOADrdP[k], tTMEM_LOADrdP[k + 1]),
        (scale_softmax, scale_softmax)
      )
      # <=1 key: the analytic 0 is unreachable; NaN at 0 keys, residue at 1.
      row_0 = cute.get(tTMEM_LOADcdP[k], mode=[0])
      row_1 = cute.get(tTMEM_LOADcdP[k + 1], mode=[0])
      if const_expr(self.is_causal):
        # Bottom-right alignment: row r sees keys up to r - (Sq - Sk).
        first_multi_key_row = seqlen_q - seqlen_k + 1
        if cute.elem_less(row_0, first_multi_key_row):
          tTMEM_LOADrdP[k] = 0.0
        if cute.elem_less(row_1, first_multi_key_row):
          tTMEM_LOADrdP[k + 1] = 0.0
      else:
        if cute.elem_less(seqlen_k, 2):
          tTMEM_LOADrdP[k] = 0.0
          tTMEM_LOADrdP[k + 1] = 0.0
    dp_vec = tTMEM_LOADrdP.load()
    tTMEM_STORErdP.store(dp_vec.to(self.q_dtype))

    ds_handle = ds_mma_producer.acquire_and_advance()
    # Thread t holds half t//64 of row t%64; dSK lanes need the full 128 keys.
    rdS_half = cute.make_tensor(
      cute.recast_ptr(tTMEM_STORErdP.iterator, dtype=self.acc_dtype),
      cute.make_layout(self.DS_PACKED_WORDS_PER_HALF),
    )
    participant_row = thread_idx % self.ds_participant_rows
    rdS_full = cute.make_rmem_tensor(
      2 * self.DS_PACKED_WORDS_PER_HALF, self.acc_dtype
    )
    for chunk in cutlass.range_constexpr(self.DS_EXCHANGE_PASSES):
      base = chunk * self.DS_EXCHANGE_PASS_WORDS
      for word in cutlass.range_constexpr(self.DS_EXCHANGE_PASS_WORDS):
        sdS_xchg[thread_idx * self.DS_EXCHANGE_PITCH_WORDS +
                 word] = rdS_half[base + word]
      self.compute_pair_barrier.arrive_and_wait()
      # Both reads always: a select makes the dest index dynamic and spills.
      for word in cutlass.range_constexpr(self.DS_EXCHANGE_PASS_WORDS):
        rdS_full[base + word] = sdS_xchg[participant_row *
                                         self.DS_EXCHANGE_PITCH_WORDS + word]
        rdS_full[self.DS_PACKED_WORDS_PER_HALF + base +
                 word] = sdS_xchg[(participant_row + self.ds_participant_rows) *
                                  self.DS_EXCHANGE_PITCH_WORDS + word]
      # All reads land before the next pass (or KV block) overwrites them.
      self.compute_pair_barrier.arrive_and_wait()

    tmem_store_atom = cute.make_copy_atom(
      tcgen05.St32x32bOp(tcgen05.Repetition(32)), self.acc_dtype
    )
    tdStdS = cute.make_tensor(
      tdPtdP_slice.iterator,
      cute.make_layout(
        (self.dov_mma_tiler[0], 2 * self.DS_PACKED_WORDS_PER_HALF),
        # One row per TMEM lane, one column per packed word.
        stride=(tmem_offset(1, 0), tmem_offset(0, 1)),
      ),
    )
    tdScdS = cute.make_identity_tensor(
      (self.dov_mma_tiler[0], 2 * self.DS_PACKED_WORDS_PER_HALF)
    )
    tmem_tiled_store = tcgen05.make_tmem_copy(tmem_store_atom, tdStdS)

    thr_store = tmem_tiled_store.get_slice(thread_idx)
    tTMEM_STOREtdS = thr_store.partition_D(tdStdS)
    tTMEM_STOREcdP = thr_store.partition_S(tdScdS)
    tTMEM_STORErdS_ = cute.make_tensor(rdS_full.iterator, tTMEM_STOREcdP.shape)
    cute.copy(tmem_tiled_store, tTMEM_STORErdS_, tTMEM_STOREtdS)
    cute.arch.fence_view_async_tmem_store()
    ds_handle.commit()
    return mma_s_consumer, mma_dp_consumer, ds_mma_producer

  @cute.jit
  def epilogue(
    self,
    seqlen_q: Int32,
    dq_args: Tuple,
    epi_tile: cute.Tile,
    tma_args: Tuple,
  ) -> pipeline.PipelineConsumer:
    """Drain one dQ tile: TMEM -> registers -> SMEM arena -> TMA store."""
    (mma_dq_consumer, gdQ_staged, cdQ_staged, tdQtdQ_staged) = dq_args
    tma_atom_dQ, gdQ_tma_staged, s_epi_dQ, varlen = tma_args
    dq_handle = mma_dq_consumer.wait_and_advance()
    cute.arch.fence_view_async_shared()

    # From _setup_attributes, never recomputed: must match the TMA descriptor.
    epi_cols_dQ = self.epi_cols_dQ
    num_epi_stages_dQ = self.num_epi_stages_dQ
    epi_tile_dQ = self.epi_tile_dQ
    leader_warp = (cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4) == 0

    for dq_slice in cutlass.range(self.iterations_dsk):
      gdQ = gdQ_staged[None, None, dq_slice]
      cdQ = cdQ_staged[None, None, dq_slice]
      tdQtdQ = tdQtdQ_staged[(None, None), 0, 0, dq_slice]
      tdQtdQ_epi = cute.zipped_divide(tdQtdQ, epi_tile)
      cdQ_epi = cute.zipped_divide(cdQ, epi_tile)
      gdQ_epi = cute.zipped_divide(gdQ, epi_tile)
      cdQ_local = cute.make_identity_tensor(epi_tile)
      cdQ_local_epi = cute.zipped_divide(cdQ_local, epi_tile)
      tidx, _, _ = cute.arch.thread_idx()
      thread_idx = tidx % (self.threads_per_warp * len(self.epilogue_warp_ids))
      tmem_copy_atom = cute.make_copy_atom(
        tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype
      )
      tiled_tmem_load = tcgen05.make_tmem_copy(tmem_copy_atom, tdQtdQ_epi)
      thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)
      tTMEM_LOADtdQ = thr_tmem_load.partition_S(tdQtdQ_epi)
      tTMEM_LOADgdQ = thr_tmem_load.partition_D(gdQ_epi)
      tTMEM_LOADcdQ = thr_tmem_load.partition_D(cdQ_epi)
      tTMEM_LOADcdQ_local = thr_tmem_load.partition_D(cdQ_local_epi)

      if const_expr(not varlen):
        gdQ_tma = gdQ_tma_staged[None, None, dq_slice]
        gdQ_tma_epi = cute.local_tile(gdQ_tma, epi_tile_dQ, (0, None))
        sdQ_stage = s_epi_dQ[None, None, 0]

        for stage_k in cutlass.range_constexpr(num_epi_stages_dQ):
          for i in cutlass.range(
            cute.size(tTMEM_LOADtdQ, mode=[1]), unroll_full=True
          ):
            tTMEM_LOADtdQ_i = tTMEM_LOADtdQ[None, i, 0]
            tTMEM_LOADcdQ_i_local = tTMEM_LOADcdQ_local[None, i, 0]
            tTMrdQ = cute.make_rmem_tensor(
              tTMEM_LOADcdQ_i_local.shape, self.acc_dtype
            )
            cute.copy(tiled_tmem_load, tTMEM_LOADtdQ_i, tTMrdQ)
            tSMrdQ = cute.make_rmem_tensor(tTMrdQ.shape, self.q_dtype)
            dq_vec = tTMrdQ.load()
            tSMrdQ.store(dq_vec.to(self.q_dtype))
            for j in cutlass.range_constexpr(cute.size(tTMEM_LOADcdQ_i_local)):
              c = tTMEM_LOADcdQ_i_local[j]
              m_pos = c[0]
              n_pos = c[1]
              if n_pos // epi_cols_dQ == stage_k:
                s_epi_dQ[m_pos, n_pos % epi_cols_dQ, 0] = tSMrdQ[j]

          cute.arch.fence_view_async_shared()
          cute.arch.barrier(
            barrier_id=self.epilogue_arena_bar_id,
            number_of_threads=self.epilogue_arena_threads,
          )

          if leader_warp:
            gdQ_stage = gdQ_tma_epi[None, None, stage_k]
            tdQsdQ, tdQgdQ = cpasync.tma_partition(
              tma_atom_dQ,
              0,
              cute.make_layout(1),
              cute.group_modes(sdQ_stage, 0, 2),
              cute.group_modes(gdQ_stage, 0, 2),
            )
            cute.copy(tma_atom_dQ, tdQsdQ, tdQgdQ)
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0, read=True)
          # No stage rotation until the leader warp's TMA read has completed.
          cute.arch.barrier(
            barrier_id=self.epilogue_arena_bar_id,
            number_of_threads=self.epilogue_arena_threads,
          )
      else:
        for i in cutlass.range(
          cute.size(tTMEM_LOADtdQ, mode=[1]), unroll_full=True
        ):
          tTMEM_LOADtdQ_i = tTMEM_LOADtdQ[None, i, 0]
          tTMEM_LOADgdQ_i = tTMEM_LOADgdQ[None, i, 0]
          tTMEM_LOADcdQ_i = tTMEM_LOADcdQ[None, i, 0]
          tTMrdQ = cute.make_rmem_tensor(
            tTMEM_LOADcdQ[None, 0, i].shape, self.acc_dtype
          )
          cute.copy(tiled_tmem_load, tTMEM_LOADtdQ_i, tTMrdQ)
          tSMrdQ = cute.make_rmem_tensor(tTMrdQ.shape, self.q_dtype)
          dq_vec = tTMrdQ.load()
          tSMrdQ.store(dq_vec.to(self.q_dtype))
          if cute.elem_less(tTMEM_LOADcdQ_i[0][0], seqlen_q):
            cute.autovec_copy(tSMrdQ, tTMEM_LOADgdQ_i)
    dq_handle.release()
    return mma_dq_consumer

  @cute.jit
  def epilogue_clear(
    self,
    seqlen_q: Int32,
    gdQ_staged,
    cdQ_staged,
  ):
    """Write a zero dQ tile for a keyless work tile."""
    num_epi_threads = self.threads_per_warp * len(self.epilogue_warp_ids)
    tidx = cute.arch.thread_idx()[0] % num_epi_threads

    tiled_copy_r2g = copy_utils.tiled_copy_2d(
      self.dq_dtype, cute.size(gdQ_staged.shape[1]), num_epi_threads
    )

    thr_copy_r2g = tiled_copy_r2g.get_slice(tidx)
    tdQgdQ_staged = thr_copy_r2g.partition_D(gdQ_staged)
    tdQcdQ_staged = thr_copy_r2g.partition_D(cdQ_staged)

    tdQrdQ = cute.make_rmem_tensor_like(tdQgdQ_staged[None, 0, None, 0])
    tdQrdQ.fill(self.dq_dtype(0.0))

    for dq_slice in cutlass.range(self.iterations_dsk, unroll_full=True):
      tdQgdQ = tdQgdQ_staged[None, None, None, dq_slice]
      tdQcdQ = tdQcdQ_staged[None, None, None, dq_slice]
      for m in cutlass.range(cute.size(tdQgdQ.shape[1]), unroll_full=True):
        if cute.elem_less(tdQcdQ[0, m, 0][0], seqlen_q):
          cute.copy(tiled_copy_r2g, tdQrdQ, tdQgdQ[None, m, None])

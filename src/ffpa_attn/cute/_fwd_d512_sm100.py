# Copyright (c) DefTruth, qyjdef@163.com
# Copyright (c) Butterfingrz，13524387014@163.com
#
# Adapted from the SM100 head-dim 256 specialized implementation
# in https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/sm100_hd256_2cta_fmha_forward.py
#
# SM100 (Blackwell) forward pass for FFPA attention — only head_dim=512.
#
# The pair is the unit of work: a 2-CTA cluster runs one group-M128 tile, each
# CTA owning 64 of the query rows.  QK and PV are both m128xn128xk128 tcgen05
# atoms, so D=512 is four MMA generations per edge, and all four O slices stay
# resident in TMEM for the whole KV sweep — no SplitKV combine, no O workspace.
#
# Design (12 warps / 384 threads, one role each):
#   - LOAD (warp 9) TMAs Q once per work item and K per KV block; V LOAD
#     (warp 10) is a second issuer, so V paces independently of K.
#   - MMA (warp 8, issued by the leader CTA) runs QK two KV blocks ahead of PV.
#   - SOFTMAX (warps 0-3): online max/sum, P_hat into SMEM sP, the rescale
#     factor into sStats, and the natural-log LSE store.
#   - CORRECTION (warps 4-7) rescales O (skipped by warp vote when softmax
#     freezes the row max), owns the TMEM allocation and runs the epilogue.
#   - The P bridge is SMEM, not the donor's S2T copy, so S ([0,128) as 2 x 64
#     columns) and O ([256,512) as 4 x 64) are disjoint TMEM regions.
#   - Epilogue: T2R, scale by 1/row_sum, store to GMEM at 128 bits/thread.
#
# Constraints:
#   - head_dim == head_dim_v == 512, tile_m == tile_n == 128
#   - cluster (2, 1, 1); the 2-CTA group is not optional
#   - Always persistent; a zero-trip tile still runs one masked block-0 trip
#   - No sO arena and no TMA S2G store — SMEM has no room to restore them
#   - No local attention, SplitKV, paged KV, pack_gqa, score_mod/mask_mod,
#     aux tensors, or Q/KV subtiling

import math
from functools import partial
from typing import Callable, Literal, NamedTuple, Tuple, Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils_basic
from cutlass import const_expr, pipeline
from cutlass.pipeline import Agent, CooperativeGroup
from cutlass.cute.typing import Int32, Int64, Float32

from .utils.tile_scheduler import (
  SingleTileVarlenScheduler,
  TileSchedulerArguments,
  Sm100FmhaStaticTileScheduler as FmhaStaticTileScheduler,
  Sm100FmhaStaticTileSchedulerParams as FmhaStaticTileSchedulerParams,
)
from .utils.mask import (
  Sm100FusedMask as FusedMask,
)
from .utils import copy_utils
from .utils.block_info import BlockInfo
from .utils.blackwell_helpers import (  # both re-exported for the SM100 tests
  SM100_SMEM_CAPACITY_BYTES,
  SM100_TMEM_CAPACITY_COLUMNS,
)
from .utils.hd512_helpers import IketTraceChannel, check_tmem_intervals
from .utils.named_barrier import NamedBarrierFwdSm100Hd512
from .utils.seqlen_info import SeqlenInfoQK
from .utils.cute_dsl_utils import assume_tensor_aligned
from . import utils
from .utils import as_bshkrd_tensor, AuxData
from .utils.softmax import ex2_emulation_2

_LOG2_DTYPE_MAX = {
  cutlass.Float16: math.log2(65504.0),
  cutlass.BFloat16: math.log2(3.3895313892515355e38),
}


# Parity-only: fp8 descale is unsupported; ``descale_tensors`` must be None.
class DescaleTensors(NamedTuple):
  q_descale: Optional[cute.Tensor] = None
  k_descale: Optional[cute.Tensor] = None
  v_descale: Optional[cute.Tensor] = None

  def __new_from_mlir_values__(self, values):
    return DescaleTensors(*((*values, None, None, None)[:3]))


# Keys: (is_2cta, is_causal, head_dim, is_fp8); always 2-CTA here, never fp8.
_TUNING_CONFIG = {
  (True, False, 512, False): {
    "ex2_emu_freq": 14,
    "ex2_emu_res": 6,
    "ex2_emu_start_frg": 0,
    "num_regs_softmax": 256,
    "num_regs_correction": 160
  },
  (True, True, 512, False): {
    "ex2_emu_freq": 14,
    "ex2_emu_res": 6,
    "ex2_emu_start_frg": 0,
    "num_regs_softmax": 256,
    "num_regs_correction": 160
  },
}


# Emitters no-op unless `iket_stamps` is set pre-JIT; setting it forks the key.
class FFPAAttnFwdSm100D512(IketTraceChannel):
  """SM100 D512 forward: a 2-CTA cluster runs one group-M128 tile.

  Each CTA owns 64 of the 128 query rows; QK and PV are m128n128k128 tcgen05
  atoms, so D=512 is four MMA slices per edge, and all four O slices stay
  resident in TMEM for the whole KV sweep (see the module header).
  """

  def __init__(
    self,
    head_dim: int,
    head_dim_v: Optional[int] = None,
    qhead_per_kvhead: int = 1,  # parity-only: GQA is a zero-stride view here
    is_causal: bool = False,
    is_local: bool = False,
    is_split_kv: bool = False,
    pack_gqa: bool = False,
    q_subtile_factor: int = 1,
    kv_subtile_factor: int = 1,
    m_block_size: int = 128,
    n_block_size: int = 128,
    q_stage: int = 2,  # parity-only: Q ring depth is set in _setup_attributes
    is_persistent: bool = True,  # parity-only: always persistent
    score_mod=None,
    mask_mod=None,
    has_aux_tensors: bool = False,
    paged_kv_non_tma: bool = False,
    is_varlen_q: bool = False,
    use_2cta_instrs: bool = False,  # parity-only: this kernel is always 2-CTA
    use_clc_scheduler: bool = False,  # rejected below
    # phase / even / e1 are decoded in _setup_attributes.
    dense_causal_sched: Literal["phase", "even", "e1"] = "phase",
  ):
    """Configure the D512 forward; parity-only knobs are accepted, ignored."""
    head_dim_v = head_dim if head_dim_v is None else head_dim_v
    assert head_dim == 512 and head_dim_v == 512, (
      "SM100 forward with head_dim=512 requires (head_dim, head_dim_v) = (512, 512)"
    )
    assert score_mod is None, "SM100 forward with head_dim=512 does not support score_mod"
    assert mask_mod is None, "SM100 forward with head_dim=512 does not support mask_mod"
    assert not has_aux_tensors, "SM100 forward with head_dim=512 does not support aux tensors"
    assert not paged_kv_non_tma, (
      "SM100 forward with head_dim=512 does not support paged KV"
    )
    assert not pack_gqa, "SM100 forward with head_dim=512 does not support pack_gqa"
    assert not is_split_kv, "SM100 forward with head_dim=512 does not support SplitKV"
    assert q_subtile_factor == 1, (
      "SM100 forward with head_dim=512 does not support q_subtile_factor"
    )
    assert kv_subtile_factor == 1, (
      "SM100 forward with head_dim=512 does not support kv_subtile_factor"
    )
    assert m_block_size == 128 and n_block_size == 128, (
      "SM100 forward with head_dim=512 requires tile_m=128 and tile_n=128"
    )
    assert not is_local, (
      "SM100 forward with head_dim=512 does not support local attention"
    )
    assert not use_clc_scheduler, (
      "SM100 forward with head_dim=512 does not support the CLC scheduler"
    )
    # Parity-only knobs are marked in the signature: accepted and ignored.

    qk_acc_dtype = cutlass.Float32
    pv_acc_dtype = cutlass.Float32
    mma_tiler = (64, 128, head_dim)
    self.qk_acc_dtype = qk_acc_dtype
    self.pv_acc_dtype = pv_acc_dtype
    assert mma_tiler[0] == 64 and mma_tiler[1] == 128, (
      "M128 impl: per-CTA tile is 64x128"
    )
    assert mma_tiler[2] == 512, "The full-Dv tile requires D=512"
    # Tiler legend, (M, N, K): M = Q rows, N = KV rows, K = reduced head dim.
    self.cta_tiler = mma_tiler  # per-CTA (64 Q rows, 128 KV rows, full D)
    # QK MMA: the pair's 128 Q rows x 128 KV rows over one 128-wide D slice.
    self.qk_mma_tiler = (
      2 * mma_tiler[0],
      mma_tiler[1],
      min(self.cta_tiler[2], 128),
    )
    # PV MMA: 128 Q rows x one 128-wide Dv output slice over 128 KV rows.
    self.pv_mma_tiler = (self.qk_mma_tiler[0], 128, 128)
    # This CTA's half of the PV MMA tile.
    self.pv_block_tiler = (
      self.pv_mma_tiler[0] // 2,
      self.pv_mma_tiler[1],
      self.pv_mma_tiler[2],
    )
    # D slices along the QK MMA K axis (D / 128).
    self.iterations_qk = self.cta_tiler[2] // self.qk_mma_tiler[2]
    # Dv output slices along the PV MMA N axis (Dv / 128).
    self.iterations_pv = self.cta_tiler[2] // self.pv_mma_tiler[1]
    self.cta_group_size = 2
    self.cluster_shape_mn = (2, 1)
    # Note [M128 Lane Fold]: lane m%64 + 64*h owns row m's N-half h.
    self.tmem_o_slice_stride = self.qk_mma_tiler[1] // self.cta_group_size
    # Note [AccumulatorHandoff]: 128 b/thread; 256 STG.E.U16 -> 32 STG.E.128.
    self.r2g_bits_per_copy = 128
    # Always persistent; the causal order relies on the persistent strided walk.
    self.is_persistent = True
    self.is_causal = is_causal
    # Varlen: pair-safe compact scheduler; causal adds sequence-local LPT.
    self.use_varlen_compact_scheduler = is_varlen_q
    # Dense-causal order, constexpr-exclusive with the packed varlen remap.
    sched = dense_causal_sched
    assert sched in ("phase", "even", "e1"), (
      f"dense_causal_sched must be one of phase/even/e1, got {sched!r}"
    )
    # phase: serpentine + cohort phase; even: serpentine, phase 0; e1: identity.
    # Serpentine is head-local: contiguous K/V prefixes; cohorts alternate ends.
    self.head_serpentine = is_causal and not is_varlen_q and sched != "e1"
    self.head_serpentine_phase = self.head_serpentine and sched == "phase"
    self.varlen_lpt = is_causal and is_varlen_q
    self.is_local = is_local
    self.use_semantic_trip_range = is_causal or is_local
    # A dedicated V warp paces K and V independently; dense KV is host-enforced.
    self.v_load_warp_id = 10  # also in reg_trim_warp_ids: runs at 32 registers

    self.iket_stamps = bool(type(self).iket_stamps)

    self.softmax_warp_ids = (0, 1, 2, 3)
    self.correction_warp_ids = (4, 5, 6, 7)
    self.mma_warp_id = 8
    self.load_warp_id = 9
    self.empty_warp_id = 11
    # Pre-dispatch 32-reg trim set: warp 10 (V load) and the idle warp 11.
    self.reg_trim_warp_ids = (self.v_load_warp_id, self.empty_warp_id)
    self.tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

    # 128 = 64 rows x 2 N-halves; indexes every paired-softmax SMEM scratch.
    self.threads_per_softmax_group = cute.arch.WARP_SIZE * len(
      self.softmax_warp_ids
    )
    self.threads_per_cta = cute.arch.WARP_SIZE * len((
      *self.softmax_warp_ids,
      *self.correction_warp_ids,
      self.mma_warp_id,
      self.load_warp_id,
      self.v_load_warp_id,
      self.empty_warp_id,
    ))

    self.tmem_alloc_barrier = pipeline.NamedBarrier(
      barrier_id=int(NamedBarrierFwdSm100Hd512.TmemPtr),
      num_threads=self.threads_per_cta,
    )
    self.softmax_pair_barrier = pipeline.NamedBarrier(
      barrier_id=int(NamedBarrierFwdSm100Hd512.SoftmaxPair),
      num_threads=self.threads_per_softmax_group,
    )

    self.tmem_s_offset = 0
    # Measured O map [256,512); the gap [128,256) is unowned, off the ledger.
    self.tmem_o_offset = 256

    _tune_key = (True, is_causal, head_dim, False)  # (is_2cta, causal, D, fp8)
    _tune = _TUNING_CONFIG[_tune_key]
    self.num_regs_softmax = _tune["num_regs_softmax"]
    self.num_regs_correction = _tune["num_regs_correction"]
    # Not from _TUNING_CONFIG: fixed at 32, as in hd256 2CTA.
    self.num_regs_other = 32
    # ex2_emu_*: emulated exp2 (freq 0 = all hw exp2), measured neutral.
    self.ex2_emu_freq = _tune["ex2_emu_freq"]
    self.ex2_emu_res = _tune["ex2_emu_res"]
    self.ex2_emu_start_frg = _tune["ex2_emu_start_frg"]

    # Not the donor's 1024: a 1024-B sQ alignment would overflow the ceiling.
    self.buffer_align_bytes = 128

  def causal_serpentine_decode(
    self,
    cluster_m,
    serp_hb,
    serp_num_supertiles,
    serp_num_heads,
    serp_half_grid,
  ):
    """One pair-order seam for all five roles; a bijection -- order only."""
    if const_expr(self.head_serpentine):
      # `serp_hb` untouched: a global remap measured 6.79x DRAM read.
      cohort_parity = (serp_hb[0] ^ ((serp_num_heads & 1) & serp_hb[1])) & 1
      phase = Int32(0)
      if const_expr(self.head_serpentine_phase):
        # 1 iff consecutive cohorts overlap the pair window (`2*NS > P`).
        phase = cutlass.min(
          Int32(1),
          cutlass.max(Int32(0), 2 * serp_num_supertiles - serp_half_grid),
        )
      # Branchless `m -> NS-1-m` on the selected class; operands non-negative.
      cluster_m = cluster_m + (1 - (cohort_parity ^ phase)
                               ) * (serp_num_supertiles - 1 - 2 * cluster_m)
    return cluster_m, serp_hb

  def tmem_region_intervals(self, qk_acc_stage: int):
    """Column ledger of the S stages and O slices for check_tmem_intervals."""
    # Widths are derived: per Note [M128 Lane Fold] a 128 slice is 64 columns.
    columns_per_slice = self.qk_mma_tiler[1] // self.cta_group_size
    intervals = {}
    # S per stage: a deeper-but-not-wider ring would lower, keep SMEM, and race.
    for stage in range(qk_acc_stage):
      start = self.tmem_s_offset + stage * columns_per_slice
      intervals[f"S{stage}"] = (start, start + columns_per_slice)
    for dv_slice in range(self.iterations_pv):
      start = self.tmem_o_offset + dv_slice * self.tmem_o_slice_stride
      intervals[f"O{dv_slice}"] = (start, start + columns_per_slice)
    return intervals

  def _setup_attributes(self):
    """Ring depths and derived constants (trace-time; needs no tensor)."""
    # Q/K/V ring depth = D-slice count (not the ignored q_stage ctor parameter).
    self.q_stage = self.iterations_qk
    self.k_stage = self.iterations_qk
    self.v_stage = self.iterations_pv
    self.qk_acc_stage = 2
    # One O token per item: a slice-granularity token measured a regression.
    self.mma_corr_stage = 1
    self.sum_stage = 1
    self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
    # AccumulatorHandoff: drain rounds derive from the producer-compatible grid.
    self.rescale_tile = self.pv_block_tiler[:2]
    # (M64,N128) is physically (M64,(N64,2)); a (64,64) round fixes the N-half.
    self.epi_tile = (64, 64)
    self.epi_warp_shape_mn = (2, 2)
    check_tmem_intervals(self.tmem_region_intervals(self.qk_acc_stage))

  def _get_tiled_mma(self):
    """S = Q @ K^T and O += P @ V; P is a K-major SMEM operand."""
    cta_group = tcgen05.CtaGroup.TWO
    p_source = tcgen05.OperandSource.SMEM
    p_major_mode = tcgen05.OperandMajorMode.K
    tiled_mma_qk = sm100_utils_basic.make_trivial_tiled_mma(
      self.q_dtype,
      self.q_major_mode,
      self.k_major_mode,
      self.qk_acc_dtype,
      cta_group,
      self.qk_mma_tiler[:2],
    )
    tiled_mma_pv = sm100_utils_basic.make_trivial_tiled_mma(
      self.v_dtype,
      p_major_mode,
      self.v_major_mode,
      self.pv_acc_dtype,
      cta_group,
      self.pv_mma_tiler[:2],
      p_source,
    )
    return tiled_mma_qk, tiled_mma_pv

  @cute.jit
  def __call__(
    self,
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mO: cute.Tensor,
    mLSE: Optional[cute.Tensor],
    softmax_scale: Float32,
    mCuSeqlensQ: Optional[cute.Tensor] = None,
    mCuSeqlensK: Optional[cute.Tensor] = None,
    mSeqUsedQ: Optional[cute.Tensor] = None,
    mSeqUsedK: Optional[cute.Tensor] = None,
    mPageTable: Optional[cute.Tensor] = None,
    window_size_left: Int32 | int | None = None,
    window_size_right: Int32 | int | None = None,
    learnable_sink: Optional[cute.Tensor] = None,
    descale_tensors: Optional[DescaleTensors] = None,
    blocksparse_tensors: Optional[cute.Tensor] = None,
    aux_data: Optional[AuxData] = None,
    stream: cuda.CUstream = None,
  ):
    """Trace entry; mirrors FlashAttentionForwardSm100.__call__'s interface."""
    assert mSeqUsedQ is None and mSeqUsedK is None, (
      "SM100 forward with head_dim=512 does not support seqused_q/seqused_k"
    )
    assert learnable_sink is None, (
      "SM100 forward with head_dim=512 does not support learnable_sink"
    )
    assert blocksparse_tensors is None, (
      "SM100 forward with head_dim=512 does not support block sparse tensors"
    )
    assert aux_data is None or aux_data.tensors is None, (
      "SM100 forward with head_dim=512 does not support aux_tensors"
    )
    assert aux_data is None or aux_data.scalars is None, (
      "SM100 forward with head_dim=512 does not support aux_scalars"
    )
    assert not self.is_local, (
      "SM100 forward with head_dim=512 does not support local attention"
    )
    assert window_size_left is None and window_size_right is None, (
      "SM100 forward with head_dim=512 requires full window bounds"
    )
    assert descale_tensors is None, (
      "SM100 forward with head_dim=512 does not support descale_tensors"
    )
    # Scheduler owns logical enumeration only; offsets/masks stay in every role.
    assert (mCuSeqlensQ is None) == (mCuSeqlensK is None), (
      "SM100 forward with head_dim=512 varlen requires both Q and K prefixes"
    )
    assert mPageTable is None, "SM100 forward with head_dim=512 does not support paged KV"
    # A mismatch runs the dense walk over packed tensors: a silent wrong answer.
    assert (mCuSeqlensQ is not None) == self.use_varlen_compact_scheduler, (
      "SM100 forward with head_dim=512: whether cu_seqlens is passed must match "
      "the is_varlen_q the kernel was constructed with"
    )

    q_tensor, k_tensor, v_tensor, o_tensor = mQ, mK, mV, mO
    lse_tensor = mLSE

    q_rank = len(mQ.shape)
    k_rank = len(mK.shape)
    if const_expr(mCuSeqlensQ is not None):
      # Varlen path accepts either legacy 5D tensors or standard 3D tensors.
      if const_expr(q_rank == 5):
        s_q = mQ.shape[1]
        h_q = mQ.shape[2] * mQ.shape[3]
        d = mQ.shape[4]
      elif const_expr(q_rank == 3):
        s_q = mQ.shape[0]
        h_q = mQ.shape[1]
        d = mQ.shape[2]
      else:
        raise RuntimeError(
          f"SM100 forward with head_dim=512 varlen expects q rank 3 or 5, got rank {q_rank}"
        )
    else:
      # Non-varlen path accepts either legacy 5D tensors or standard 4D tensors.
      if const_expr(q_rank == 5):
        s_q = mQ.shape[1]
        h_q = mQ.shape[2] * mQ.shape[3]
        d = mQ.shape[4]
      elif const_expr(q_rank == 4):
        s_q = mQ.shape[1]
        h_q = mQ.shape[2]
        d = mQ.shape[3]
      else:
        raise RuntimeError(
          f"SM100 forward with head_dim=512 non-varlen expects q rank 4 or 5, got rank {q_rank}"
        )

    if const_expr(mCuSeqlensK is not None):
      if const_expr(k_rank == 5):
        s_k = mK.shape[1]
        h_k = mK.shape[2]
      elif const_expr(k_rank == 3):
        s_k = mK.shape[0]
        h_k = mK.shape[1]
      else:
        raise RuntimeError(
          f"SM100 forward with head_dim=512 varlen expects k rank 3 or 5, got rank {k_rank}"
        )
    else:
      if const_expr(k_rank == 5):
        s_k = mK.shape[1]
        h_k = mK.shape[2]
      elif const_expr(k_rank == 4):
        s_k = mK.shape[1]
        h_k = mK.shape[2]
      else:
        raise RuntimeError(
          f"SM100 forward with head_dim=512 non-varlen expects k rank 4 or 5, got rank {k_rank}"
        )
    if const_expr(mCuSeqlensQ is not None):
      b = mCuSeqlensQ.shape[0] - 1
    elif const_expr(mCuSeqlensK is not None):
      b = mCuSeqlensK.shape[0] - 1
    else:
      b = mQ.shape[0]

    scale_softmax = softmax_scale
    scale_softmax_log2 = softmax_scale * math.log2(math.exp(1.0))
    s_lse = s_q
    h_r = h_q // h_k
    s_q64 = Int64(s_q)
    s_k64 = Int64(s_k)
    s_lse64 = Int64(s_lse)
    h_r64 = Int64(h_r)
    h_k64 = Int64(h_k)
    b64 = Int64(b)
    s_q_total = (
      q_tensor.shape[1] if mCuSeqlensQ is not None and q_rank == 5 else
      (q_tensor.shape[0] if mCuSeqlensQ is not None else s_q64)
    )
    s_k_total = (
      k_tensor.shape[1] if mCuSeqlensK is not None and k_rank == 5 else
      (k_tensor.shape[0] if mCuSeqlensK is not None else s_k64)
    )
    b_lse = b64 if mCuSeqlensQ is None else 1
    stride_b_lse = h_r64 * h_k64 * s_lse64 if mCuSeqlensQ is None else 0

    varlen_q = mCuSeqlensQ is not None
    varlen_k = mCuSeqlensK is not None
    q_norm = as_bshkrd_tensor(q_tensor, h_k, h_r, varlen_q)
    o_norm = as_bshkrd_tensor(o_tensor, h_k, h_r, varlen_q)

    # (s, d, ((h_r, h_k), b)); canonical strides 1=S, 4=D, 3=H_r, 2=H_k, 0=B.
    q = cute.make_tensor(
      q_norm.iterator,
      cute.make_layout(
        (s_q_total, d, ((h_r, h_k), b)),
        stride=(
          q_norm.stride[1],
          q_norm.stride[4],
          ((q_norm.stride[3], q_norm.stride[2]), q_norm.stride[0]),
        ),
      ),
    )
    # h_r extent 1 in the normalizer; the zero h_r stride broadcasts heads.
    k_norm = as_bshkrd_tensor(k_tensor, h_k, 1, varlen_k)
    v_norm = as_bshkrd_tensor(v_tensor, h_k, 1, varlen_k)
    # (s, d, ((h_r, h_k), b)), 0-stride for h_r to broadcast
    k = cute.make_tensor(
      k_norm.iterator,
      cute.make_layout(
        (s_k_total, d, ((h_r, h_k), b)),
        stride=(
          k_norm.stride[1],
          k_norm.stride[4],
          ((0, k_norm.stride[2]), k_norm.stride[0]),
        ),
      ),
    )
    # (d, s, ((h_r, h_k), b)), 0-stride for h_r to broadcast
    v = cute.make_tensor(
      v_norm.iterator,
      cute.make_layout(
        (d, s_k_total, ((h_r, h_k), b)),
        stride=(
          v_norm.stride[4],
          v_norm.stride[1],
          ((0, v_norm.stride[2]), v_norm.stride[0]),
        ),
      ),
    )
    # (s, d, ((h_r, h_k), b))
    o = cute.make_tensor(
      o_norm.iterator,
      cute.make_layout(
        (s_q_total, d, ((h_r, h_k), b)),
        stride=(
          o_norm.stride[1],
          o_norm.stride[4],
          ((o_norm.stride[3], o_norm.stride[2]), o_norm.stride[0]),
        ),
      ),
    )
    o = assume_tensor_aligned(o)
    if const_expr(lse_tensor is not None):
      # (s, ((h_r, h_k), b))
      lse_layout = cute.make_layout(
        (s_lse64, ((h_r, h_k), b_lse)),
        stride=(1, ((s_lse64, h_r64 * s_lse64), stride_b_lse)),
      )
      lse = cute.make_tensor(lse_tensor.iterator, lse_layout)
    else:
      lse = None

    # Trace-time only: dtypes, major modes and layouts come from traced tensors.
    self.q_dtype = q.element_type
    self.k_dtype = k.element_type
    self.v_dtype = v.element_type
    self.o_dtype = o.element_type
    # Note [Low Precision Scaling]: the max may lag; drives `rescale_o`'s skip.
    self.rescale_threshold = 8.0 if const_expr(
      self.q_dtype.width == 16
    ) else 0.0
    assert self.rescale_threshold < _LOG2_DTYPE_MAX[self.q_dtype], (
      f"rescale_threshold ({self.rescale_threshold}) must stay below "
      f"log2(max({self.q_dtype})) or the top probabilities saturate while "
      "the FP32 denominator still counts them in full"
    )
    # No "tilePlikeFP32" scalar: on the folded fragment it silently covers half.

    if const_expr(self.use_varlen_compact_scheduler):
      # One work = one 128-row supertile: ceil(ceil(Lq/64)/2) == ceil(Lq/128).
      tile_sched_args = TileSchedulerArguments(
        num_block=cute.ceil_div(cute.size(q.shape[0]), self.qk_mma_tiler[0]),
        num_head=cute.size(o.shape[2][0]),
        num_batch=cute.size(mCuSeqlensQ.shape[0] - 1),
        num_splits=1,
        seqlen_k=cute.size(k.shape[0]),
        headdim=cute.size(q.shape[1]),
        headdim_v=cute.size(v.shape[0]),
        total_q=cute.size(q.shape[0]),
        tile_shape_mn=self.cta_tiler[:2],
        mCuSeqlensQ=mCuSeqlensQ,
        qhead_per_kvhead_packgqa=1,
        element_size=self.k_dtype.width // 8,
        # LPT flips only the sequence-local Q axis; the predicate is local too.
        lpt=self.varlen_lpt,
        cluster_shape_mn=self.cluster_shape_mn,
        use_cluster_idx=True,
        varlen_static_persistent=True,
      )
      self.tile_sched_params = SingleTileVarlenScheduler.to_underlying_arguments(
        tile_sched_args
      )
      grid = SingleTileVarlenScheduler.get_grid_shape(self.tile_sched_params)
    else:
      # M128: x counts per-CTA 64-row tiles; even m_x/grid keeps peers paired.
      shape_for_grid = ((s_q, o.shape[1],
                         o.shape[2]) if mCuSeqlensQ is not None else o.shape)
      m_tiles_64 = cute.ceil_div(
        cute.size(shape_for_grid[0]), self.cta_tiler[0]
      )
      m_x = (
        cute.ceil_div(m_tiles_64, self.cta_group_size) * self.cta_group_size
      )
      self.tile_sched_params = FmhaStaticTileSchedulerParams(
        self.is_persistent,
        (
          m_x,
          cute.size(shape_for_grid[2][0]),
          cute.size(shape_for_grid[2][1]),
        ),
      )
      grid = FmhaStaticTileScheduler.get_grid_shape(self.tile_sched_params)

    self.q_major_mode = cutlass.utils.LayoutEnum.from_tensor(q).mma_major_mode()
    self.k_major_mode = cutlass.utils.LayoutEnum.from_tensor(k).mma_major_mode()
    self.v_major_mode = cutlass.utils.LayoutEnum.from_tensor(v).mma_major_mode()
    self.o_layout = cutlass.utils.LayoutEnum.from_tensor(o)

    if const_expr(self.q_major_mode != tcgen05.OperandMajorMode.K):
      raise RuntimeError("The layout of q is not supported")
    if const_expr(self.k_major_mode != tcgen05.OperandMajorMode.K):
      raise RuntimeError("The layout of k is not supported")
    if const_expr(self.v_major_mode != tcgen05.OperandMajorMode.MN):
      raise RuntimeError("The layout of v is not supported")

    if const_expr(self.q_dtype != self.k_dtype):
      raise TypeError(f"Type mismatch: {self.q_dtype} != {self.k_dtype}")
    if const_expr(self.q_dtype != self.v_dtype):
      raise TypeError(f"Type mismatch: {self.q_dtype} != {self.v_dtype}")
    self._setup_attributes()

    cta_group = tcgen05.CtaGroup.TWO
    tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()
    self.cta_layout_vmnk = cute.tiled_divide(
      cute.make_layout(self.cluster_shape_mnk),
      (tiled_mma_qk.thr_id.shape, ),
    )

    sQ_layout = sm100_utils_basic.make_smem_layout_a(
      tiled_mma_qk,
      self.qk_mma_tiler,
      self.q_dtype,
      self.q_stage,
    )
    sK_layout = sm100_utils_basic.make_smem_layout_b(
      tiled_mma_qk,
      self.qk_mma_tiler,
      self.k_dtype,
      self.k_stage,
    )
    # Force K_INTER (the heuristic picks SW128): byte-identical to sP_layout.
    tP_a_shape = tiled_mma_pv.partition_shape_A(
      cute.dice(self.pv_mma_tiler, (1, None, 1))
    )
    tP_layout = sm100_utils_basic.tile_to_mma_shape(
      tcgen05.make_smem_layout_atom(
        tcgen05.SmemLayoutAtomKind.K_INTER, self.q_dtype
      ),
      cute.append(tP_a_shape, self.qk_acc_stage),
      order=(1, 2, 3),
    )
    sV_layout = sm100_utils_basic.make_smem_layout_b(
      tiled_mma_pv,
      self.pv_mma_tiler,
      self.v_dtype,
      self.v_stage,
    )
    # One K_INTER buffer (64x128 16-bit) per slot; PV reads these bytes as A.
    sP_layout = cute.tile_to_shape(
      tcgen05.make_smem_layout_atom(
        tcgen05.SmemLayoutAtomKind.K_INTER, self.q_dtype
      ),
      (64, 128, self.qk_acc_stage),
      (0, 1, 2),
    )
    tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(cta_group)

    q_smem_layout = cute.select(sQ_layout, mode=[0, 1, 2])
    tma_atom_q, tma_tensor_q = cute.nvgpu.make_tiled_tma_atom_A(
      tma_load_op,
      q,
      q_smem_layout,
      self.qk_mma_tiler,
      tiled_mma_qk,
      self.cta_layout_vmnk.shape,
    )

    k_smem_layout = cute.select(sK_layout, mode=[0, 1, 2])
    tma_atom_k, tma_tensor_k = cute.nvgpu.make_tiled_tma_atom_B(
      tma_load_op,
      k,
      k_smem_layout,
      self.qk_mma_tiler,
      tiled_mma_qk,
      self.cta_layout_vmnk.shape,
    )
    v_smem_layout = cute.select(sV_layout, mode=[0, 1, 2])
    tma_atom_v, tma_tensor_v = cute.nvgpu.make_tiled_tma_atom_B(
      tma_load_op,
      v,
      v_smem_layout,
      self.pv_mma_tiler,
      tiled_mma_pv,
      self.cta_layout_vmnk.shape,
    )

    q_copy_size = cute.size_in_bytes(self.q_dtype, q_smem_layout)
    k_copy_size = cute.size_in_bytes(self.k_dtype, k_smem_layout)
    v_copy_size = cute.size_in_bytes(self.v_dtype, v_smem_layout)
    self.tma_copy_q_bytes = q_copy_size * self.cta_group_size
    self.tma_copy_k_bytes = k_copy_size * self.cta_group_size
    self.tma_copy_v_bytes = v_copy_size * self.cta_group_size

    @cute.struct
    class SharedStorage:
      # TMA G2S load barriers: LOAD warp (producer) -> MMA warp (consumer)
      load_q_mbar_ptr: cute.struct.MemRange[Int64, self.q_stage * 2
                                            ]  # load_q_{producer,consumer}
      load_k_mbar_ptr: cute.struct.MemRange[Int64, self.k_stage * 2
                                            ]  # load_k_{producer,consumer}
      load_v_mbar_ptr: cute.struct.MemRange[Int64, self.v_stage * 2
                                            ]  # load_v_{producer,consumer}
      mma_s_mbar_ptr: cute.struct.MemRange[Int64, self.qk_acc_stage * 2]
      p_mma_mbar_ptr: cute.struct.MemRange[Int64, self.qk_acc_stage * 2]
      # Softmax -> Correction signaling (sStats carries the factor r_i)
      s_corr_mbar_ptr: cute.struct.MemRange[Int64, self.qk_acc_stage * 2
                                            ]  # s_corr_{producer,consumer}
      sum_mbar_ptr: cute.struct.MemRange[Int64, self.sum_stage * 2]
      # MMA -> Correction ownership for O_partial tokens (rescale/finalize)
      mma_corr_mbar_ptr: cute.struct.MemRange[Int64, self.mma_corr_stage * 2
                                              ]  # mma_corr_{producer,consumer}
      # Cluster-wide TMEM lifetime mbar: both CTAs finish before dealloc.
      tmem_dealloc_mbar: Int64
      tmem_holding_buf: Int32
      # Declaration order IS the address order, and it is the measured one.
      sQ: cute.struct.Align[
        cute.struct.MemRange[self.q_dtype,
                             cute.cosize(sQ_layout.outer)],
        self.buffer_align_bytes,
      ]
      sK: cute.struct.Align[
        cute.struct.MemRange[self.k_dtype,
                             cute.cosize(sK_layout.outer)],
        self.buffer_align_bytes,
      ]
      sV: cute.struct.Align[
        cute.struct.MemRange[self.v_dtype,
                             cute.cosize(sV_layout.outer)],
        self.buffer_align_bytes,
      ]
      # Per-thread partial row sum, index t = row + 64*half.
      sSum: cute.struct.Align[
        cute.struct.MemRange[self.qk_acc_dtype, self.threads_per_softmax_group],
        self.buffer_align_bytes,
      ]
      # Partial row max, same index; single buffer, reuse gated per KV block.
      sPartialMax: cute.struct.Align[
        cute.struct.MemRange[self.qk_acc_dtype, self.threads_per_softmax_group],
        self.buffer_align_bytes,
      ]
      # Ring (s_corr depth) of r_i itself, not maxima: exp2 once on 64 threads.
      sStats: cute.struct.Align[
        cute.struct.MemRange[self.qk_acc_dtype, self.qk_acc_stage * 64],
        self.buffer_align_bytes,
      ]
      # The P bridge is unconditional; sPexch exchange measured a regression.
      sP: cute.struct.Align[
        cute.struct.MemRange[self.q_dtype,
                             cute.cosize(sP_layout.outer)],
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
      tiled_mma_qk,
      tiled_mma_pv,
      tma_atom_q,
      tma_tensor_q,
      tma_atom_k,
      tma_tensor_k,
      tma_atom_v,
      tma_tensor_v,
      o,
      mCuSeqlensQ,
      mCuSeqlensK,
      lse,
      scale_softmax_log2,
      scale_softmax,
      window_size_left,
      window_size_right,
      self.cta_layout_vmnk,
      sQ_layout,
      sK_layout,
      tP_layout,
      sV_layout,
      sP_layout,
      self.tile_sched_params,
    ).launch(
      grid=grid,
      block=[self.threads_per_cta, 1, 1],
      cluster=self.cluster_shape_mnk,
      stream=stream,
      min_blocks_per_mp=1,
    )

  @cute.jit
  def scheduler_work_to_physical_coord(
    self,
    scheduler_coord: cute.Coord,
    pair_rank: Int32,
  ):
    """Compact-scheduler coord -> role ABI; rank stays outside the work id."""
    if const_expr(self.use_varlen_compact_scheduler):
      block_coord, head_coord, batch_coord, _ = scheduler_coord
      return (
        block_coord * self.cta_group_size + pair_rank,
        Int32(0),
        (head_coord, batch_coord),
      )
    return scheduler_coord

  @cute.jit
  def kv_block_info(
    self,
    window_size_left: Optional[Int32],
    window_size_right: Optional[Int32],
  ) -> BlockInfo:
    """Direct BlockInfo: `*_via_block_info` fakes a 2nd seqlen authority."""
    return BlockInfo(
      tile_m=self.qk_mma_tiler[0],
      tile_n=self.qk_mma_tiler[1],
      is_causal=self.is_causal,
      is_local=self.is_local and not self.is_causal,
      window_size_left=window_size_left,
      window_size_right=window_size_right,
    )

  @cute.jit
  def get_kv_trip_start_count(
    self,
    mma_block_coord: cute.Coord,
    seqlen: SeqlenInfoQK,
    window_size_left: Optional[Int32],
    window_size_right: Optional[Int32],
  ) -> Tuple[Int32, Int32]:
    """One trip interval shared by all ring-advancing roles; <=0 clamps to 1."""
    block_info = self.kv_block_info(window_size_left, window_size_right)
    n_block_min, n_block_max = block_info.get_n_block_min_max(
      seqlen, mma_block_coord[0]
    )
    start, count = n_block_min, n_block_max - n_block_min
    if count <= 0:
      # Clamp also gives zero-trip work a valid block-0 V coordinate.
      start = Int32(0)
      count = Int32(1)
    return start, count

  @cute.jit
  def get_kv_mask_bounds(
    self,
    mma_block_coord: cute.Coord,
    seqlen: SeqlenInfoQK,
    window_size_left: Optional[Int32],
    window_size_right: Optional[Int32],
  ) -> Tuple[Int32, Int32]:
    """Dense-walk mask bounds: masked-right start, unmasked-middle start."""
    block_info = self.kv_block_info(window_size_left, window_size_right)
    n_block_min, _ = block_info.get_n_block_min_max(seqlen, mma_block_coord[0])
    return (
      block_info.get_n_block_min_causal_local_mask(
        seqlen, mma_block_coord[0], n_block_min
      ),
      block_info.get_n_block_min_before_local_mask(
        seqlen, mma_block_coord[0], n_block_min
      ),
    )

  @cute.kernel
  def kernel(
    self,
    tiled_mma_qk: cute.TiledMma,
    tiled_mma_pv: cute.TiledMma,
    tma_atom_q: cute.CopyAtom,
    mQ_qdl: cute.Tensor,
    tma_atom_k: cute.CopyAtom,
    mK_kdl: cute.Tensor,
    tma_atom_v: cute.CopyAtom,
    mV_dkl: cute.Tensor,
    mO_qdl: cute.Tensor,
    mCuSeqlensQ: Optional[cute.Tensor],
    mCuSeqlensK: Optional[cute.Tensor],
    mLSE: Optional[cute.Tensor],
    scale_softmax_log2: Float32,
    scale_softmax: Float32,
    window_size_left: Optional[Int32],
    window_size_right: Optional[Int32],
    cta_layout_vmnk: cute.Layout,
    sQ_layout: cute.ComposedLayout,
    sK_layout: cute.ComposedLayout,
    tP_layout: cute.ComposedLayout,
    sV_layout: cute.ComposedLayout,
    sP_layout: cute.ComposedLayout,
    tile_sched_params: FmhaStaticTileSchedulerParams
    | SingleTileVarlenScheduler.Params,
  ):
    """Persistent 2-CTA kernel body: role dispatch by warp index."""
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    # One shared range: all 12 warps run the same setup before role dispatch.
    prologue_range = self._range_start("prologue")

    if warp_idx == self.load_warp_id:
      cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
      cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
    # One prefetch per descriptor, in the warp that uses it: V is below.
    if warp_idx == self.v_load_warp_id:
      cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)

    cta_rank_in_cluster = cute.arch.make_warp_uniform(
      cute.arch.block_idx_in_cluster()
    )
    # The cluster rank is the 64-row shard owner inside the group-M128 pair.
    pair_rank = cta_rank_in_cluster & 1
    mma_tile_coord_v = pair_rank
    block_in_cluster_coord_vmnk = cta_layout_vmnk.get_flat_coord(
      cta_rank_in_cluster
    )

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(self.shared_storage)

    # TMA issuers and the MMA warp arrive through one elected thread each.
    tma_producer_group = CooperativeGroup(Agent.Thread, 1)
    mma_group = CooperativeGroup(Agent.Thread, 1)
    load_q_producer, load_q_consumer = pipeline.PipelineTmaUmma.create(
      num_stages=self.q_stage,
      producer_group=tma_producer_group,
      consumer_group=mma_group,
      tx_count=self.tma_copy_q_bytes,
      barrier_storage=storage.load_q_mbar_ptr.data_ptr(),
      cta_layout_vmnk=cta_layout_vmnk,
      mcast_mode_mn=(1, 0),
      defer_sync=True,
    ).make_participants()
    load_k_producer, load_k_consumer = pipeline.PipelineTmaUmma.create(
      num_stages=self.k_stage,
      producer_group=tma_producer_group,
      consumer_group=mma_group,
      tx_count=self.tma_copy_k_bytes,
      barrier_storage=storage.load_k_mbar_ptr.data_ptr(),
      cta_layout_vmnk=cta_layout_vmnk,
      mcast_mode_mn=(1, 0),
      defer_sync=True,
    ).make_participants()
    load_v_producer, load_v_consumer = pipeline.PipelineTmaUmma.create(
      num_stages=self.v_stage,
      producer_group=tma_producer_group,
      consumer_group=mma_group,
      tx_count=self.tma_copy_v_bytes,
      barrier_storage=storage.load_v_mbar_ptr.data_ptr(),
      cta_layout_vmnk=cta_layout_vmnk,
      mcast_mode_mn=(1, 0),
      defer_sync=True,
    ).make_participants()
    mma_s_producer, mma_s_consumer = pipeline.PipelineUmmaAsync.create(
      num_stages=self.qk_acc_stage,
      producer_group=mma_group,
      consumer_group=CooperativeGroup(
        Agent.Thread,
        self.threads_per_softmax_group * self.cta_group_size,
      ),
      barrier_storage=storage.mma_s_mbar_ptr.data_ptr(),
      cta_layout_vmnk=cta_layout_vmnk,
      defer_sync=True,
    ).make_participants()
    p_mma_producer, p_mma_consumer = pipeline.PipelineAsyncUmma.create(
      num_stages=self.qk_acc_stage,
      producer_group=CooperativeGroup(
        Agent.Thread,
        self.threads_per_softmax_group * self.cta_group_size,
      ),
      consumer_group=mma_group,
      barrier_storage=storage.p_mma_mbar_ptr.data_ptr(),
      cta_layout_vmnk=cta_layout_vmnk,
      defer_sync=True,
    ).make_participants()
    s_corr_producer, s_corr_consumer = pipeline.PipelineAsync.create(
      num_stages=self.qk_acc_stage,
      producer_group=CooperativeGroup(
        Agent.Thread, self.threads_per_softmax_group
      ),
      consumer_group=CooperativeGroup(
        Agent.Thread, cute.arch.WARP_SIZE * len(self.correction_warp_ids)
      ),
      barrier_storage=storage.s_corr_mbar_ptr.data_ptr(),
      defer_sync=True,
    ).make_participants()
    sum_producer, sum_consumer = pipeline.PipelineAsync.create(
      num_stages=self.sum_stage,
      producer_group=CooperativeGroup(
        Agent.Thread, self.threads_per_softmax_group
      ),
      consumer_group=CooperativeGroup(
        Agent.Thread, cute.arch.WARP_SIZE * len(self.correction_warp_ids)
      ),
      barrier_storage=storage.sum_mbar_ptr.data_ptr(),
      defer_sync=True,
    ).make_participants()
    mma_corr_producer, mma_corr_consumer = pipeline.PipelineUmmaAsync.create(
      num_stages=self.mma_corr_stage,
      producer_group=mma_group,
      consumer_group=CooperativeGroup(
        Agent.Thread,
        cute.arch.WARP_SIZE * len(self.correction_warp_ids) *
        self.cta_group_size,
      ),
      barrier_storage=storage.mma_corr_mbar_ptr.data_ptr(),
      cta_layout_vmnk=cta_layout_vmnk,
      defer_sync=True,
    ).make_participants()
    tmem = cutlass.utils.TmemAllocator(
      storage.tmem_holding_buf.ptr,
      barrier_for_retrieve=self.tmem_alloc_barrier,
      allocator_warp_id=self.correction_warp_ids[0],
      is_two_cta=True,
      two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
    )
    tmem.allocate(self.tmem_alloc_cols)
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)

    pipeline.pipeline_init_arrive(
      cluster_shape_mn=cta_layout_vmnk, is_relaxed=True
    )

    sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
    sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
    sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
    sSum = storage.sSum.get_tensor(
      cute.make_layout(self.threads_per_softmax_group)
    )
    sPartialMax = storage.sPartialMax.get_tensor(
      cute.make_layout(self.threads_per_softmax_group)
    )
    sStats = storage.sStats.get_tensor(
      cute.make_layout((self.qk_acc_stage, 64))
    )
    sP = storage.sP.get_tensor(sP_layout.outer, swizzle=sP_layout.inner)
    thr_mma_qk = tiled_mma_qk.get_slice(mma_tile_coord_v)
    thr_mma_pv = tiled_mma_pv.get_slice(mma_tile_coord_v)
    tSrQ = thr_mma_qk.make_fragment_A(sQ)
    tSrK = thr_mma_qk.make_fragment_B(sK)
    tOrV = thr_mma_pv.make_fragment_B(sV)
    # sP: softmax store view (sP_layout); sP_mma: same bytes as PV A (tP).
    # A fragment over the bytes softmax publishes; replaces the ~250-cycle S2T.
    sP_mma = storage.sP.get_tensor(tP_layout.outer, swizzle=tP_layout.inner)
    tOrP = thr_mma_pv.make_fragment_A(sP_mma)
    qk_acc_shape = thr_mma_qk.partition_shape_C(
      (self.qk_mma_tiler[0], self.qk_mma_tiler[1])
    )
    tStS = thr_mma_qk.make_fragment_C(
      cute.append(qk_acc_shape, self.qk_acc_stage)
    )
    pv_acc_shape = thr_mma_pv.partition_shape_C(
      (self.pv_mma_tiler[0], self.pv_mma_tiler[1])
    )
    tOtO = thr_mma_pv.make_fragment_C(pv_acc_shape)
    tOtO_layout = cute.append(
      tOtO.layout,
      cute.make_layout(
        self.iterations_pv,
        stride=self.tmem_o_slice_stride,
      ),
    )
    # make_fragment_C bases at a dummy 0x0: only a full alloc really lands at 0.
    tStS = cute.make_tensor(tStS.iterator + self.tmem_s_offset, tStS.layout)
    tOtO_staged = cute.make_tensor(
      tOtO.iterator + self.tmem_o_offset, tOtO_layout
    )

    # ///  REG TRIM  ///
    for _i in cutlass.range_constexpr(len(self.reg_trim_warp_ids)):
      if warp_idx == self.reg_trim_warp_ids[_i]:
        cute.arch.setmaxregister_decrease(self.num_regs_other)

    if const_expr(self.use_varlen_compact_scheduler):
      tile_sched = SingleTileVarlenScheduler.create(tile_sched_params)
    else:
      blk_idx = cute.arch.block_idx()
      tile_sched = FmhaStaticTileScheduler(
        tile_sched_params, blk_idx[0], blk_idx, cute.arch.grid_dim()
      )
    work_tile = tile_sched.initial_work_tile_info()
    # m_x is padded to cluster pairs (even), so NS = m_x // 2 is exact.
    if const_expr(self.head_serpentine):
      serp_num_supertiles = (
        tile_sched_params.problem_shape_mbh[0] // self.cta_group_size
      )
      serp_num_heads = tile_sched_params.problem_shape_mbh[1]
      # Persistent CTA pairs: the modulus P of the serpentine phase test.
      serp_half_grid = (tile_sched.num_persistent_sm // self.cta_group_size)
      serp_consts = (
        serp_num_supertiles,
        serp_num_heads,
        serp_half_grid,
      )
    else:
      serp_consts = None

    # One seqlen authority as a partial: eager loads would land on all 12 warps.
    SeqlenInfoCls = partial(
      SeqlenInfoQK.create,
      seqlen_q_static=mQ_qdl.shape[0],
      seqlen_k_static=mK_kdl.shape[0],
      mCuSeqlensQ=mCuSeqlensQ,
      mCuSeqlensK=mCuSeqlensK,
    )

    pipeline.pipeline_init_wait(cluster_shape_mn=cta_layout_vmnk)
    self._range_end(prologue_range)

    # ///  LOAD  ///
    if warp_idx == self.load_warp_id:
      cute.arch.setmaxregister_decrease(self.num_regs_other)
      self.load(
        tile_sched,
        work_tile,
        pair_rank,
        serp_consts,
        SeqlenInfoCls,
        thr_mma_qk,
        mQ_qdl,
        mK_kdl,
        sQ,
        sK,
        tma_atom_q,
        tma_atom_k,
        cta_layout_vmnk,
        block_in_cluster_coord_vmnk,
        load_q_producer,
        load_k_producer,
        window_size_left,
        window_size_right,
      )

    # ///  V LOAD  ///
    if warp_idx == self.v_load_warp_id:
      self.load_v(
        tile_sched,
        work_tile,
        pair_rank,
        serp_consts,
        SeqlenInfoCls,
        thr_mma_pv,
        mV_dkl,
        sV,
        tma_atom_v,
        cta_layout_vmnk,
        block_in_cluster_coord_vmnk,
        load_v_producer,
        window_size_left,
        window_size_right,
      )

    # ///  MMA  ///
    # Inline by measurement: extraction demotes a UNIFORM reg to a spill slot.
    if warp_idx == self.mma_warp_id:
      cute.arch.setmaxregister_decrease(self.num_regs_other)

      cta_rank_in_cluster = cute.arch.make_warp_uniform(
        cute.arch.block_idx_in_cluster()
      )
      is_leader_cta = cta_rank_in_cluster % 2 == 0

      self._range_push("mma_e2e")
      if const_expr(self.is_causal):
        qk0_hoisted = Int32(0)
        next_work = work_tile
      while work_tile.is_valid_tile:
        # Before the valid-work guard so zero-trip tiles still show as slices.
        self._range_push("mma_tile")
        self._stamp("mma.tile_begin")
        # All five roles must decode the same coordinate and trip range.
        physical_block_coord = self.scheduler_work_to_physical_coord(
          work_tile.tile_idx, pair_rank
        )
        cluster_m = physical_block_coord[0] // self.cluster_shape_mnk[0]
        serp_hb = physical_block_coord[2]
        if const_expr(self.head_serpentine):
          cluster_m, serp_hb = self.causal_serpentine_decode(
            cluster_m,
            serp_hb,
            serp_num_supertiles,
            serp_num_heads,
            serp_half_grid,
          )
        curr_block_coord = (
          cluster_m * self.cta_group_size + pair_rank,
          physical_block_coord[1],
          serp_hb,
        )
        mma_block_coord = (
          curr_block_coord[0] // self.cta_group_size,
          curr_block_coord[1],
          curr_block_coord[2],
        )
        continue_cond = False
        batch_coord = curr_block_coord[2][1]
        seqlen = SeqlenInfoCls(batch_coord)
        seqlen_q = seqlen.seqlen_q
        if const_expr(seqlen.has_cu_seqlens_q):
          continue_cond = not FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
            self.qk_mma_tiler[0],
            mma_block_coord[0],
            seqlen_q,
          )

        if not continue_cond:
          kv_trip_start, kv_trip_count = (
            self.get_kv_trip_start_count(
              mma_block_coord,
              seqlen,
              window_size_left,
              window_size_right,
            )
          )

          load_q_releaser = load_q_consumer.clone()
          tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, False)
          self._stamp("mma.first_qk_pv")
          if kv_trip_count > 1:
            # ---  prologue  ---
            # QK0
            _qk0_gate = is_leader_cta
            if const_expr(self.is_causal):
              _qk0_gate = is_leader_cta and qk0_hoisted == 0
            if _qk0_gate:
              s_handle = mma_s_producer.acquire_and_advance()
              tStS_slice = tStS[None, None, None, s_handle.index]
              tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, False)
              for d_slice in cutlass.range(self.iterations_qk, unroll=1):
                load_q_consumer.wait_and_advance()
                tSrQ_slice = tSrQ[None, None, None, d_slice]
                k_handle = load_k_consumer.wait_and_advance()
                tSrK_slice = tSrK[None, None, None, k_handle.index]
                num_kphases = cute.size(tSrQ_slice, mode=[2])
                for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                  kphase_coord = (None, None, kphase_idx)
                  cute.gemm(
                    tiled_mma_qk,
                    tStS_slice,
                    tSrQ_slice[kphase_coord],
                    tSrK_slice[kphase_coord],
                    tStS_slice,
                  )
                  tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, True)
                k_handle.release()
              s_handle.commit()
            # Ordering: QK runs 2 ahead; P is SMEM-A, so no TMEM S alias.
            if kv_trip_count > 2:
              if is_leader_cta:
                s_handle = mma_s_producer.acquire_and_advance()
                tStS_slice = tStS[None, None, None, s_handle.index]
                tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, False)
                for d_slice in cutlass.range(self.iterations_qk, unroll=1):
                  tSrQ_slice = tSrQ[None, None, None, d_slice]
                  k_handle = load_k_consumer.wait_and_advance()
                  tSrK_slice = tSrK[None, None, None, k_handle.index]
                  num_kphases = cute.size(tSrQ_slice, mode=[2])
                  for kphase_idx in cutlass.range(
                    num_kphases, unroll_full=True
                  ):
                    kphase_coord = (None, None, kphase_idx)
                    cute.gemm(
                      tiled_mma_qk,
                      tStS_slice,
                      tSrQ_slice[kphase_coord],
                      tSrK_slice[kphase_coord],
                      tStS_slice,
                    )
                    tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, True)
                  k_handle.release()
                s_handle.commit()
            # ---  steady  ---
            # Keep OUTSIDE is_leader_cta: a guard demotes the uniform datapath.
            self._range_push("mma_kv_steady")
            for i in cutlass.range(2, kv_trip_count - 1, 1, unroll=1):
              # QKi
              if is_leader_cta:
                s_handle = mma_s_producer.acquire_and_advance()
                tStS_slice = tStS[None, None, None, s_handle.index]
                tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, False)
                for d_slice in cutlass.range(self.iterations_qk, unroll=1):
                  tSrQ_slice = tSrQ[None, None, None, d_slice]
                  k_handle = load_k_consumer.wait_and_advance()
                  tSrK_slice = tSrK[None, None, None, k_handle.index]
                  num_kphases = cute.size(tSrQ_slice, mode=[2])
                  for kphase_idx in cutlass.range(
                    num_kphases, unroll_full=True
                  ):
                    kphase_coord = (None, None, kphase_idx)
                    cute.gemm(
                      tiled_mma_qk,
                      tStS_slice,
                      tSrQ_slice[kphase_coord],
                      tSrK_slice[kphase_coord],
                      tStS_slice,
                    )
                    tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, True)
                  k_handle.release()
                s_handle.commit()

                # PVi-2
                p_handle = p_mma_consumer.wait_and_advance()
                # O acquire wait: depth 1 blocks on the previous epilogue.
                self._range_push("mma_o_acquire")
                o_handle = mma_corr_producer.acquire_and_advance()
                self._range_pop()
                pv_whether_acc = tiled_mma_pv.get(tcgen05.Field.ACCUMULATE)
                for dv_slice in cutlass.range(self.iterations_pv, unroll=1):
                  v_handle = load_v_consumer.wait_and_advance()
                  tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, pv_whether_acc)
                  tOtO_slice = tOtO_staged[None, None, None, dv_slice]
                  tOrP_slice = tOrP[None, None, None, p_handle.index]
                  tOrV_slice = tOrV[None, None, None, v_handle.index]
                  num_kphases = cute.size(tOrV_slice, mode=[2])
                  for kphase_idx in cutlass.range(
                    num_kphases, unroll_full=True
                  ):
                    kphase_coord = (None, None, kphase_idx)
                    cute.gemm(
                      tiled_mma_pv,
                      tOtO_slice,
                      tOrP_slice[kphase_coord],
                      tOrV_slice[kphase_coord],
                      tOtO_slice,
                    )
                    tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, True)
                  v_handle.release()
                o_handle.commit()
                p_handle.release()
            self._range_pop()  # mma_kv_steady
            # ---  drain  ---
            # Drain the extra in-flight P; without it p_mma deadlocks.
            if kv_trip_count > 2:
              if is_leader_cta:
                p_handle = p_mma_consumer.wait_and_advance()
                o_handle = mma_corr_producer.acquire_and_advance()
                pv_whether_acc = tiled_mma_pv.get(tcgen05.Field.ACCUMULATE)
                for dv_slice in cutlass.range(self.iterations_pv, unroll=1):
                  v_handle = load_v_consumer.wait_and_advance()
                  tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, pv_whether_acc)
                  tOtO_slice = tOtO_staged[None, None, None, dv_slice]
                  tOrP_slice = tOrP[None, None, None, p_handle.index]
                  tOrV_slice = tOrV[None, None, None, v_handle.index]
                  num_kphases = cute.size(tOrV_slice, mode=[2])
                  for kphase_idx in cutlass.range(
                    num_kphases, unroll_full=True
                  ):
                    kphase_coord = (None, None, kphase_idx)
                    cute.gemm(
                      tiled_mma_pv,
                      tOtO_slice,
                      tOrP_slice[kphase_coord],
                      tOrV_slice[kphase_coord],
                      tOtO_slice,
                    )
                    tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, True)
                  v_handle.release()
                o_handle.commit()
                p_handle.release()
            if is_leader_cta:
              # QKend
              s_handle = mma_s_producer.acquire_and_advance()
              tStS_slice = tStS[None, None, None, s_handle.index]
              tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, False)
              for d_slice in cutlass.range(self.iterations_qk, unroll=1):
                tSrQ_slice = tSrQ[None, None, None, d_slice]
                k_handle = load_k_consumer.wait_and_advance()
                tSrK_slice = tSrK[None, None, None, k_handle.index]
                num_kphases = cute.size(tSrQ_slice, mode=[2])
                for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                  kphase_coord = (None, None, kphase_idx)
                  cute.gemm(
                    tiled_mma_qk,
                    tStS_slice,
                    tSrQ_slice[kphase_coord],
                    tSrK_slice[kphase_coord],
                    tStS_slice,
                  )
                  tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, True)
                k_handle.release()
                load_q_releaser.release()
                load_q_releaser.advance()
              s_handle.commit()

              # PVend-1
              p_handle = p_mma_consumer.wait_and_advance()
              o_handle = mma_corr_producer.acquire_and_advance()
              pv_whether_acc = tiled_mma_pv.get(tcgen05.Field.ACCUMULATE)
              for dv_slice in cutlass.range(self.iterations_pv, unroll=1):
                v_handle = load_v_consumer.wait_and_advance()
                tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, pv_whether_acc)
                tOtO_slice = tOtO_staged[None, None, None, dv_slice]
                tOrP_slice = tOrP[None, None, None, p_handle.index]
                tOrV_slice = tOrV[None, None, None, v_handle.index]
                num_kphases = cute.size(tOrV_slice, mode=[2])
                for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                  kphase_coord = (None, None, kphase_idx)
                  cute.gemm(
                    tiled_mma_pv,
                    tOtO_slice,
                    tOrP_slice[kphase_coord],
                    tOrV_slice[kphase_coord],
                    tOtO_slice,
                  )
                  tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, True)
                v_handle.release()
              o_handle.commit()
              p_handle.release()
          else:
            # ---  single trip: prologue and drain in one  ---
            if const_expr(self.is_causal):
              if is_leader_cta and qk0_hoisted != 0:
                for d_slice in cutlass.range(self.iterations_qk, unroll=1):
                  load_q_releaser.release()
                  load_q_releaser.advance()
            _qk0_gate = is_leader_cta
            if const_expr(self.is_causal):
              _qk0_gate = is_leader_cta and qk0_hoisted == 0
            if _qk0_gate:
              # QK0
              s_handle = mma_s_producer.acquire_and_advance()
              tStS_slice = tStS[None, None, None, s_handle.index]
              tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, False)
              for d_slice in cutlass.range(self.iterations_qk, unroll=1):
                load_q_consumer.wait_and_advance()
                tSrQ_slice = tSrQ[None, None, None, d_slice]
                k_handle = load_k_consumer.wait_and_advance()
                tSrK_slice = tSrK[None, None, None, k_handle.index]
                num_kphases = cute.size(tSrQ_slice, mode=[2])
                for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                  kphase_coord = (None, None, kphase_idx)
                  cute.gemm(
                    tiled_mma_qk,
                    tStS_slice,
                    tSrQ_slice[kphase_coord],
                    tSrK_slice[kphase_coord],
                    tStS_slice,
                  )
                  tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, True)
                k_handle.release()
                load_q_releaser.release()
                load_q_releaser.advance()
              s_handle.commit()

          if const_expr(self.is_causal):
            # Coordinate-free QK0 hoist; non-causal measured a net cost.
            next_work = tile_sched.advance_to_next_work()
            qk0_hoisted = Int32(0)
            if next_work.is_valid_tile:
              n_ok = Int32(1)
              if const_expr(seqlen.has_cu_seqlens_q):
                n_pbc = self.scheduler_work_to_physical_coord(
                  next_work.tile_idx, pair_rank
                )
                n_cluster_m = n_pbc[0] // self.cluster_shape_mnk[0]
                n_curr0 = n_cluster_m * self.cta_group_size + pair_rank
                n_mma0 = n_curr0 // self.cta_group_size
                n_seqlen = SeqlenInfoCls(n_pbc[2][1])
                if not FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                  self.qk_mma_tiler[0], n_mma0, n_seqlen.seqlen_q
                ):
                  n_ok = Int32(0)
              if n_ok != 0:
                if is_leader_cta:
                  # QK0 of the next item, verbatim.
                  s_handle = mma_s_producer.acquire_and_advance()
                  tStS_slice = tStS[None, None, None, s_handle.index]
                  tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, False)
                  for d_slice in cutlass.range(self.iterations_qk, unroll=1):
                    load_q_consumer.wait_and_advance()
                    tSrQ_slice = tSrQ[None, None, None, d_slice]
                    k_handle = load_k_consumer.wait_and_advance()
                    tSrK_slice = tSrK[None, None, None, k_handle.index]
                    num_kphases = cute.size(tSrQ_slice, mode=[2])
                    for kphase_idx in cutlass.range(
                      num_kphases, unroll_full=True
                    ):
                      kphase_coord = (None, None, kphase_idx)
                      cute.gemm(
                        tiled_mma_qk,
                        tStS_slice,
                        tSrQ_slice[kphase_coord],
                        tSrK_slice[kphase_coord],
                        tStS_slice,
                      )
                      tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, True)
                    k_handle.release()
                  s_handle.commit()
                qk0_hoisted = Int32(1)
          if is_leader_cta:
            # O before P: sync-equivalent orders, measured neutral -- keep.
            o_handle = mma_corr_producer.acquire_and_advance()
            p_handle = p_mma_consumer.wait_and_advance()
            pv_whether_acc = tiled_mma_pv.get(tcgen05.Field.ACCUMULATE)
            for dv_slice in cutlass.range(self.iterations_pv, unroll=1):
              v_handle = load_v_consumer.wait_and_advance()
              tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, pv_whether_acc)
              tOtO_slice = tOtO_staged[None, None, None, dv_slice]
              tOrP_slice = tOrP[None, None, None, p_handle.index]
              tOrV_slice = tOrV[None, None, None, v_handle.index]
              num_kphases = cute.size(tOrV_slice, mode=[2])
              for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                kphase_coord = (None, None, kphase_idx)
                cute.gemm(
                  tiled_mma_pv,
                  tOtO_slice,
                  tOrP_slice[kphase_coord],
                  tOrV_slice[kphase_coord],
                  tOtO_slice,
                )
                tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, True)
              v_handle.release()
            p_handle.release()
            o_handle.commit()
        self._range_pop()  # mma_tile
        if const_expr(self.is_causal):
          if continue_cond:
            next_work = tile_sched.advance_to_next_work()
            qk0_hoisted = Int32(0)
          work_tile = next_work
        else:
          work_tile = tile_sched.advance_to_next_work()
      mma_s_producer.tail()
      mma_corr_producer.tail()
      self._range_pop()  # mma_e2e

    # ///  Softmax  ///
    if warp_idx >= self.softmax_warp_ids[
      0] and warp_idx <= self.softmax_warp_ids[-1]:
      cute.arch.setmaxregister_increase(self.num_regs_softmax)
      self.softmax_loop(
        tile_sched,
        work_tile,
        pair_rank,
        serp_consts,
        SeqlenInfoCls,
        thr_mma_qk,
        mLSE,
        tStS,
        sSum,
        sStats,
        sPartialMax,
        sP,
        mma_s_consumer,
        p_mma_producer,
        s_corr_producer,
        sum_producer,
        mCuSeqlensQ,
        scale_softmax_log2,
        scale_softmax,
        window_size_left,
        window_size_right,
      )

    # ///  Correction  ///
    if warp_idx >= self.correction_warp_ids[
      0] and warp_idx < self.correction_warp_ids[-1] + 1:
      cute.arch.setmaxregister_decrease(self.num_regs_correction)
      self.correction_loop(
        tile_sched,
        work_tile,
        pair_rank,
        serp_consts,
        SeqlenInfoCls,
        mO_qdl,
        tOtO_staged,
        sSum,
        sStats,
        mma_corr_consumer,
        s_corr_consumer,
        sum_consumer,
        window_size_left,
        window_size_right,
      )

    # ///  Warps 10-11 reg re-trim  ///
    # `>` spans v_load (10) and empty (11): both re-trim once load_v returns.
    if warp_idx > self.load_warp_id:
      cute.arch.setmaxregister_decrease(self.num_regs_other)

    # ///  Cooperative TMEM Dealloc (2CTA)  ///
    cute.arch.cluster_arrive()
    cute.arch.cluster_wait()
    tmem.relinquish_alloc_permit()
    tmem.free(tmem_ptr)

    return

  @cute.jit
  def load(
    self,
    tile_sched,
    work_tile,
    pair_rank: Int32,
    serp_consts,
    SeqlenInfoCls: Callable,
    thr_mma_qk: cute.ThrMma,
    mQ_qdl: cute.Tensor,
    mK_kdl: cute.Tensor,
    sQ: cute.Tensor,
    sK: cute.Tensor,
    tma_atom_q: cute.CopyAtom,
    tma_atom_k: cute.CopyAtom,
    cta_layout_vmnk: cute.Layout,
    block_in_cluster_coord_vmnk,
    load_q_producer,
    load_k_producer,
    window_size_left: Optional[Int32],
    window_size_right: Optional[Int32],
  ):
    """Q + K TMA producer; V has its own producer warp, `load_v`."""
    # Serpentine fork only: the varlen scheduler lacks `num_persistent_sm`.
    if const_expr(self.head_serpentine):
      (
        serp_num_supertiles,
        serp_num_heads,
        serp_half_grid,
      ) = serp_consts
    self._range_push("load_e2e")
    while work_tile.is_valid_tile:
      self._range_push("load_tile")
      # All five roles must decode the same coordinate and trip range.
      physical_block_coord = self.scheduler_work_to_physical_coord(
        work_tile.tile_idx, pair_rank
      )
      cluster_m = physical_block_coord[0] // self.cluster_shape_mnk[0]
      serp_hb = physical_block_coord[2]
      if const_expr(self.head_serpentine):
        cluster_m, serp_hb = self.causal_serpentine_decode(
          cluster_m,
          serp_hb,
          serp_num_supertiles,
          serp_num_heads,
          serp_half_grid,
        )
      curr_block_coord = (
        cluster_m * self.cta_group_size + pair_rank,
        physical_block_coord[1],
        serp_hb,
      )  # (per-CTA 64-row tile, 0, (head_idx, batch_idx))
      mma_block_coord = (
        curr_block_coord[0] // self.cta_group_size,
        curr_block_coord[1],
        curr_block_coord[2],
      )
      continue_cond = False
      batch_coord = curr_block_coord[2][1]
      seqlen = SeqlenInfoCls(batch_coord)
      seqlen_q = seqlen.seqlen_q
      # Packed offsets rebase descriptor origins only; Q rides mode 0, K mode 1.
      block_offset = (
        Int32(seqlen.offset_q),
        Int32(seqlen.offset_k),
        Int32(0),
        ((Int32(0), Int32(0)), Int32(0)),
      )
      if const_expr(seqlen.has_cu_seqlens_q):
        continue_cond = not FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
          self.qk_mma_tiler[0],
          mma_block_coord[0],
          seqlen_q,
        )
      if not continue_cond:
        mQ_qdl_ = cute.domain_offset(
          cute.select(block_offset, mode=[0, 2, 3]), mQ_qdl
        )
        q_cta_layout = cute.make_layout(
          cute.slice_(cta_layout_vmnk, (0, 0, None, 0)).shape
        )
        # (bM, bK, loopM, loopK, loopL)
        gQ_qdl = cute.flat_divide(
          mQ_qdl_, cute.select(self.qk_mma_tiler, mode=[0, 2])
        )
        tSgQ_qdl = thr_mma_qk.partition_A(gQ_qdl)
        tQsQ, tQgQ_qdl = cute.nvgpu.cpasync.tma_partition(
          tma_atom_q,
          block_in_cluster_coord_vmnk[2],
          q_cta_layout,
          cute.group_modes(sQ, 0, 3),
          cute.group_modes(tSgQ_qdl, 0, 3),
        )
        kv_cta_layout = cute.make_layout(
          cute.slice_(cta_layout_vmnk, (0, None, 0, 0)).shape
        )
        mK_kdl_ = cute.domain_offset(
          cute.select(block_offset, mode=[1, 2, 3]), mK_kdl
        )
        gK_kdl = cute.flat_divide(
          mK_kdl_, cute.select(self.qk_mma_tiler, mode=[1, 2])
        )
        tSgK_kdl = thr_mma_qk.partition_B(gK_kdl)
        tKsK, tKgK_kdl = cute.nvgpu.cpasync.tma_partition(
          tma_atom_k,
          block_in_cluster_coord_vmnk[1],
          kv_cta_layout,
          cute.group_modes(sK, 0, 3),
          cute.group_modes(tSgK_kdl, 0, 3),
        )
        # ((atom_v, rest_v), RestN, RestK)
        tKgK = tKgK_kdl[None, None, None, mma_block_coord[2]]
        # ((atom_v, rest_v), RestK)
        tQgQ = tQgQ_qdl[None, mma_block_coord[0], None, mma_block_coord[2]]

        kv_trip_start, kv_trip_count = (
          self.get_kv_trip_start_count(
            mma_block_coord,
            seqlen,
            window_size_left,
            window_size_right,
          )
        )
        # Q + K0: the duration is the ring back-pressure at the tile seam.
        self._range_push("load_q_k0")
        for d_slice in cutlass.range(self.iterations_qk, unroll=1):
          q_handle = load_q_producer.acquire_and_advance()
          cute.copy(
            tma_atom_q,
            tQgQ[None, d_slice],
            tQsQ[None, q_handle.index],
            tma_bar_ptr=q_handle.barrier,
          )

        # K0
        kv_coord = kv_trip_start
        for d_slice in cutlass.range(self.iterations_qk, unroll=1):
          k_handle = load_k_producer.acquire_and_advance()
          cute.copy(
            tma_atom_k,
            tKgK[None, kv_coord, d_slice],
            tKsK[None, k_handle.index],
            tma_bar_ptr=k_handle.barrier,
          )
        kv_coord += 1
        self._range_pop()  # load_q_k0

        self._range_push("load_k_stream")
        for i in cutlass.range(1, kv_trip_count, 1, unroll=1):
          # Ki
          for d_slice in cutlass.range(self.iterations_qk, unroll=1):
            k_handle = load_k_producer.acquire_and_advance()
            cute.copy(
              tma_atom_k,
              tKgK[None, kv_coord, d_slice],
              tKsK[None, k_handle.index],
              tma_bar_ptr=k_handle.barrier,
            )
          kv_coord += 1
        self._range_pop()  # load_k_stream

      self._range_pop()  # load_tile
      work_tile = tile_sched.advance_to_next_work()
    load_k_producer.tail()
    load_q_producer.tail()
    self._range_pop()  # load_e2e

  @cute.jit
  def load_v(
    self,
    tile_sched,
    work_tile,
    pair_rank: Int32,
    serp_consts,
    SeqlenInfoCls: Callable,
    thr_mma_pv: cute.ThrMma,
    mV_dkl: cute.Tensor,
    sV: cute.Tensor,
    tma_atom_v: cute.CopyAtom,
    cta_layout_vmnk: cute.Layout,
    block_in_cluster_coord_vmnk,
    load_v_producer,
    window_size_left: Optional[Int32],
    window_size_right: Optional[Int32],
  ):
    """Dedicated V TMA producer (warp 10); a measured win, zero regressions."""
    # See load(): these three exist only under the head_serpentine fork.
    if const_expr(self.head_serpentine):
      (
        serp_num_supertiles,
        serp_num_heads,
        serp_half_grid,
      ) = serp_consts
    self._range_push("vload_e2e")
    while work_tile.is_valid_tile:
      self._range_push("vload_tile")
      # All five roles must decode the same coordinate and trip range.
      physical_block_coord = self.scheduler_work_to_physical_coord(
        work_tile.tile_idx, pair_rank
      )
      cluster_m = physical_block_coord[0] // self.cluster_shape_mnk[0]
      serp_hb = physical_block_coord[2]
      if const_expr(self.head_serpentine):
        cluster_m, serp_hb = self.causal_serpentine_decode(
          cluster_m,
          serp_hb,
          serp_num_supertiles,
          serp_num_heads,
          serp_half_grid,
        )
      curr_block_coord = (
        cluster_m * self.cta_group_size + pair_rank,
        physical_block_coord[1],
        serp_hb,
      )
      mma_block_coord = (
        curr_block_coord[0] // self.cta_group_size,
        curr_block_coord[1],
        curr_block_coord[2],
      )
      continue_cond = False
      batch_coord = curr_block_coord[2][1]
      seqlen = SeqlenInfoCls(batch_coord)
      seqlen_q = seqlen.seqlen_q
      # V is (d, k, l): the K origin rides mode 1 here.
      v_block_offset = (
        Int32(0),
        Int32(seqlen.offset_k),
        ((Int32(0), Int32(0)), Int32(0)),
      )
      if const_expr(seqlen.has_cu_seqlens_q):
        continue_cond = not FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
          self.qk_mma_tiler[0],
          mma_block_coord[0],
          seqlen_q,
        )
      if not continue_cond:
        kv_cta_layout = cute.make_layout(
          cute.slice_(cta_layout_vmnk, (0, None, 0, 0)).shape
        )
        mV_dkl_ = cute.domain_offset(v_block_offset, mV_dkl)
        gV_dkl = cute.flat_divide(
          mV_dkl_, cute.select(self.pv_mma_tiler, mode=[1, 2])
        )
        tSgV_dkl = thr_mma_pv.partition_B(gV_dkl)
        tVsV, tVgV_dkl = cute.nvgpu.cpasync.tma_partition(
          tma_atom_v,
          block_in_cluster_coord_vmnk[1],
          kv_cta_layout,
          cute.group_modes(sV, 0, 3),
          cute.group_modes(tSgV_dkl, 0, 3),
        )
        tVgV = tVgV_dkl[None, None, None, mma_block_coord[2]]
        kv_trip_start, kv_trip_count = (
          self.get_kv_trip_start_count(
            mma_block_coord,
            seqlen,
            window_size_left,
            window_size_right,
          )
        )
        kv_trip_end = kv_trip_start + kv_trip_count
        self._range_push("vload_v_stream")
        for kv_coord in cutlass.range(kv_trip_start, kv_trip_end, 1, unroll=1):
          for dv_slice in cutlass.range(self.iterations_pv, unroll=1):
            v_handle = load_v_producer.acquire_and_advance()
            cute.copy(
              tma_atom_v,
              tVgV[None, dv_slice, kv_coord],
              tVsV[None, v_handle.index],
              tma_bar_ptr=v_handle.barrier,
            )
        self._range_pop()  # vload_v_stream
      self._range_pop()  # vload_tile
      work_tile = tile_sched.advance_to_next_work()
    load_v_producer.tail()
    self._range_pop()  # vload_e2e

  @cute.jit
  def correction_loop(
    self,
    tile_sched,
    work_tile,
    pair_rank: Int32,
    serp_consts,
    SeqlenInfoCls: Callable,
    mO_qdl: cute.Tensor,
    tOtO_staged: cute.Tensor,
    sSum: cute.Tensor,
    sStats: cute.Tensor,
    mma_corr_consumer,
    s_corr_consumer,
    sum_consumer,
    window_size_left: Optional[Int32],
    window_size_right: Optional[Int32],
  ):
    """O rescale + the WHOLE O store (warps 4-7); splitting measured ~0.98x."""
    # See load(): these three exist only under the head_serpentine fork.
    if const_expr(self.head_serpentine):
      (
        serp_num_supertiles,
        serp_num_heads,
        serp_half_grid,
      ) = serp_consts
    self._range_push("corr_e2e")
    while work_tile.is_valid_tile:
      self._range_push("corr_tile")
      # All five roles must decode the same coordinate and trip range.
      physical_block_coord = self.scheduler_work_to_physical_coord(
        work_tile.tile_idx, pair_rank
      )
      cluster_m = physical_block_coord[0] // self.cluster_shape_mnk[0]
      serp_hb = physical_block_coord[2]
      if const_expr(self.head_serpentine):
        cluster_m, serp_hb = self.causal_serpentine_decode(
          cluster_m,
          serp_hb,
          serp_num_supertiles,
          serp_num_heads,
          serp_half_grid,
        )
      curr_block_coord = (
        cluster_m * self.cta_group_size + pair_rank,
        physical_block_coord[1],
        serp_hb,
      )
      mma_block_coord = (
        curr_block_coord[0] // self.cta_group_size,
        curr_block_coord[1],
        curr_block_coord[2],
      )
      continue_cond = False
      batch_coord = curr_block_coord[2][1]
      seqlen = SeqlenInfoCls(batch_coord)
      seqlen_q = seqlen.seqlen_q
      cuseqlen_q = Int32(seqlen.offset_q)
      if const_expr(seqlen.has_cu_seqlens_q):
        continue_cond = not FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
          self.qk_mma_tiler[0],
          mma_block_coord[0],
          seqlen_q,
        )

      if not continue_cond:
        mO_qdl_eff = mO_qdl
        if const_expr(seqlen.has_cu_seqlens_q):
          block_offset_o = (
            cuseqlen_q,
            Int32(0),
            Int32(0),
            ((Int32(0), Int32(0)), Int32(0)),
          )
          mO_qdl_eff = cute.domain_offset(
            cute.select(block_offset_o, mode=[0, 2, 3]), mO_qdl
          )

        # (bM, bN, loopM, loopN, loopL)
        gO_qdl = cute.flat_divide(
          mO_qdl_eff, cute.select(self.pv_block_tiler, mode=[0, 1])
        )
        cO_qdl = cute.flat_divide(
          cute.make_identity_tensor(mO_qdl_eff.shape),
          cute.select(self.pv_block_tiler, mode=[0, 1]),
        )

        _, kv_trip_count = self.get_kv_trip_start_count(
          mma_block_coord,
          seqlen,
          window_size_left,
          window_size_right,
        )
        gO_staged = gO_qdl[None, None, curr_block_coord[0], None,
                           curr_block_coord[2]]
        cO_staged = cO_qdl[None, None, curr_block_coord[0], None,
                           curr_block_coord[2]]

        # The first empty step skips correction.
        stats_handle = s_corr_consumer.wait_and_advance()
        stats_handle.release()
        self._range_push("corr_rescale")
        for step in cutlass.range(1, kv_trip_count, 1, unroll=1):
          # Oi-1 -> Oi
          mma_corr_consumer, s_corr_consumer = (
            self.correction_rescale(
              s_corr_consumer,
              sStats,
              mma_corr_consumer,
              tOtO_staged,
              cO_staged,
              self.rescale_tile,
            )
          )
        self._range_pop()  # corr_rescale
        # O_partial -> O_final
        self._range_push("corr_epilog")
        mma_corr_consumer, sum_consumer = self.correction_epilog(
          seqlen_q,
          sum_consumer,
          sSum,
          mma_corr_consumer,
          gO_staged,
          cO_staged,
          tOtO_staged,
          self.epi_tile,
        )
        self._range_pop()  # corr_epilog
      self._range_pop()  # corr_tile
      work_tile = tile_sched.advance_to_next_work()
    self._range_pop()  # corr_e2e
    # tmem.free() happens at kernel end, under the cluster-wide sync.

  @cute.jit
  def softmax_loop(
    self,
    tile_sched,
    work_tile,
    pair_rank: Int32,
    serp_consts,
    SeqlenInfoCls: Callable,
    thr_mma_qk: cute.ThrMma,
    mLSE: Optional[cute.Tensor],
    tStS: cute.Tensor,
    sSum: cute.Tensor,
    sStats: cute.Tensor,
    sPartialMax: cute.Tensor,
    sP: cute.Tensor,
    mma_s_consumer,
    p_mma_producer,
    s_corr_producer,
    sum_producer,
    mCuSeqlensQ: Optional[cute.Tensor],
    scale_softmax_log2: Float32,
    scale_softmax: Float32,
    window_size_left: Optional[Int32],
    window_size_right: Optional[Int32],
  ):
    """Online softmax (warps 0-3): P_hat, r_i, row sums, LSE; consumes S."""
    # See load(): these three exist only under the head_serpentine fork.
    if const_expr(self.head_serpentine):
      (
        serp_num_supertiles,
        serp_num_heads,
        serp_half_grid,
      ) = serp_consts
    self._range_push("softmax_e2e")
    while work_tile.is_valid_tile:
      self._range_push("softmax_tile")
      # All five roles must decode the same coordinate and trip range.
      physical_block_coord = self.scheduler_work_to_physical_coord(
        work_tile.tile_idx, pair_rank
      )
      cluster_m = physical_block_coord[0] // self.cluster_shape_mnk[0]
      serp_hb = physical_block_coord[2]
      if const_expr(self.head_serpentine):
        cluster_m, serp_hb = self.causal_serpentine_decode(
          cluster_m,
          serp_hb,
          serp_num_supertiles,
          serp_num_heads,
          serp_half_grid,
        )
      curr_block_coord = (
        cluster_m * self.cta_group_size + pair_rank,
        physical_block_coord[1],
        serp_hb,
      )
      mma_block_coord = (
        curr_block_coord[0] // self.cta_group_size,
        curr_block_coord[1],
        curr_block_coord[2],
      )
      continue_cond = False
      batch_coord = curr_block_coord[2][1]
      seqlen = SeqlenInfoCls(batch_coord)
      seqlen_q = seqlen.seqlen_q
      seqlen_k = seqlen.seqlen_k
      cuseqlen_q = Int32(seqlen.offset_q)
      if const_expr(seqlen.has_cu_seqlens_q):
        continue_cond = not FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
          self.qk_mma_tiler[0],
          mma_block_coord[0],
          seqlen_q,
        )
      if not continue_cond:
        row_max = -Float32.inf
        row_max_prev = -Float32.inf
        row_sum = 0.0

        kv_trip_start, kv_trip_count = self.get_kv_trip_start_count(
          mma_block_coord,
          seqlen,
          window_size_left,
          window_size_right,
        )
        kv_trip_end = kv_trip_start + kv_trip_count
        if const_expr(self.use_semantic_trip_range):
          n_block_min_causal_local_mask, n_block_min_before_local_mask = (
            self.get_kv_mask_bounds(
              mma_block_coord,
              seqlen,
              window_size_left,
              window_size_right,
            )
          )
        cS_base = cute.make_identity_tensor(
          (self.qk_mma_tiler[0], self.qk_mma_tiler[1])
        )
        cS = cute.domain_offset((mma_block_coord[0] * self.qk_mma_tiler[0], 0),
                                cS_base)

        self._range_push("softmax_kv")
        for step in cutlass.range(kv_trip_start, kv_trip_end, 1, unroll=1):
          cS_iter = cute.domain_offset((0, step * self.qk_mma_tiler[1]), cS)
          tScS_iter = thr_mma_qk.partition_C(cS_iter)
          if const_expr(self.use_semantic_trip_range):
            need_apply_mask = (
              step >= n_block_min_causal_local_mask
              or step < n_block_min_before_local_mask or step == kv_trip_end - 1
            )
          else:
            # Residual path only needs seqlen masking on the last K tile.
            need_apply_mask = step == kv_trip_end - 1
          (
            row_max,
            row_sum,
            mma_s_consumer,
            p_mma_producer,
            s_corr_producer,
          ) = self.softmax_step(
            need_apply_mask,
            window_size_left,
            window_size_right,
            row_max_prev,
            row_sum,
            seqlen_q,
            seqlen_k,
            scale_softmax_log2,
            mma_s_consumer,
            p_mma_producer,
            s_corr_producer,
            tStS,
            tScS_iter,
            sPartialMax,
            sStats,
            sP,
          )
          row_max_prev = row_max
        self._range_pop()  # softmax_kv
        self._range_push("softmax_lse")
        sum_producer = self.store_row_sum_and_lse(
          row_max,
          mLSE,
          row_sum,
          sSum,
          sum_producer,
          curr_block_coord,
          seqlen_q,
          mCuSeqlensQ,
          cuseqlen_q,
          scale_softmax,
        )
        self._range_pop()  # softmax_lse
      self._range_pop()  # softmax_tile
      work_tile = tile_sched.advance_to_next_work()
    p_mma_producer.tail()
    s_corr_producer.tail()
    self._range_pop()  # softmax_e2e

  @cute.jit
  def softmax_step(
    self,
    need_apply_mask,
    window_size_left: Optional[Int32],
    window_size_right: Optional[Int32],
    row_max: Float32,
    row_sum: Float32,
    seqlen_q: Int32,
    seqlen_k: Int32,
    scale_softmax_log2: Float32,
    mma_s_consumer: pipeline.PipelineConsumer,
    p_mma_producer: pipeline.PipelineProducer,
    s_corr_producer: pipeline.PipelineProducer,
    tStS: cute.Tensor,
    tScS: cute.Tensor,
    sPartialMax: cute.Tensor,
    sStats: cute.Tensor,
    sP: cute.Tensor,
  ) -> Tuple[Float32, Float32, pipeline.PipelineConsumer,
             pipeline.PipelineProducer, pipeline.PipelineProducer]:
    """One KV block of online softmax for this thread's row: S (TMEM) -> sP."""
    tidx, _, _ = cute.arch.thread_idx()
    thread_idx = tidx % self.threads_per_softmax_group
    s_handle = mma_s_consumer.wait_and_advance()
    tStS_slice = tStS[(None, None), 0, 0, s_handle.index]
    tScS_slice = tScS[(None, None), 0, 0]
    tmem_load_atom = cute.make_copy_atom(
      tcgen05.Ld32x32bOp(tcgen05.Repetition(32)), self.qk_acc_dtype
    )
    tmem_tiled_load = tcgen05.make_tmem_copy(tmem_load_atom, tStS_slice)
    thr_load = tmem_tiled_load.get_slice(thread_idx)
    tTMEM_LOADtS = thr_load.partition_S(tStS_slice)
    tTMEM_LOADcS = thr_load.partition_D(tScS_slice)
    tTMEM_LOADrS = cute.make_rmem_tensor(tTMEM_LOADcS.shape, self.qk_acc_dtype)
    cute.copy(tmem_tiled_load, tTMEM_LOADtS, tTMEM_LOADrS)

    cute.arch.fence_view_async_tmem_load()
    s_handle.release()
    if need_apply_mask:
      # R2P thermometer: per Note [M128 Lane Fold], one row per thread.
      FusedMask.apply_mask_via_causal_local_r2p(
        tTMEM_LOADrS,
        tTMEM_LOADcS,
        seqlen_q,
        seqlen_k,
        self.use_semantic_trip_range,
        self.is_causal,
        window_size_right,
      )
    old_row_max = row_max
    # Paired max meets in SMEM; fmax_reduce, not TensorSSA.reduce (fewer FMAX).
    local_max = utils.fmax_reduce(tTMEM_LOADrS.load(), row_max, arch=100)
    sPartialMax[thread_idx] = local_max
    self.softmax_pair_barrier.arrive_and_wait()
    pair_row = thread_idx % 64
    peer_max = sPartialMax[pair_row + 64]
    # Selects, not `if`s; `fmax` matches `>`: operands finite or -inf, no NaN.
    row_max = utils.fmax(sPartialMax[pair_row], peer_max)
    # This second barrier guards single-buffer reuse next KV block.
    self.softmax_pair_barrier.arrive_and_wait()
    row_max_safe = 0.0 if row_max == -cutlass.Float32.inf else row_max

    # r_i = exp2((m_{i-1}-m_i)*scale_log2); all 128 compute, N-half 0 publishes.
    corr_scale_arg = scale_softmax_log2 * (old_row_max - row_max_safe)
    corr_scale = cute.math.exp2(corr_scale_arg, fastmath=True)
    if const_expr(self.rescale_threshold > 0.0):
      # Freeze to exactly 1.0 so rescale_o skips; both halves freeze together.
      if corr_scale_arg >= -self.rescale_threshold:
        row_max = old_row_max
        row_max_safe = old_row_max
        corr_scale = 1.0
    # N-half 0 writes; both correction column-halves read the same factor.
    stats_handle = s_corr_producer.acquire_and_advance()
    if thread_idx < 64:
      sStats[stats_handle.index, thread_idx] = corr_scale
    cute.arch.fence_view_async_shared()
    stats_handle.commit()

    scale = scale_softmax_log2
    minus_row_max_scale = (0.0 - row_max_safe) * scale
    # Acquire P write slot early — overlaps any pipeline stall with exp2 compute
    p_handle = p_mma_producer.acquire_and_advance()
    # FMA + exp2 + bf16 conversion; polynomial emulation trades SFU for FMA.
    ex2_frg_tile = 32
    ex2_frg_cnt = cute.size(tTMEM_LOADrS) // ex2_frg_tile
    tTMEM_LOADrS_ex2 = cute.logical_divide(
      tTMEM_LOADrS, cute.make_layout(ex2_frg_tile)
    )
    tTMEM_STORErP = cute.make_rmem_tensor(tTMEM_LOADrS.shape, self.q_dtype)
    tTMEM_STORErP_ex2 = cute.logical_divide(
      tTMEM_STORErP, cute.make_layout(ex2_frg_tile)
    )
    for j in cutlass.range_constexpr(ex2_frg_cnt):
      for k in cutlass.range_constexpr(0, ex2_frg_tile, 2):
        tTMEM_LOADrS_ex2[k, j], tTMEM_LOADrS_ex2[
          k + 1, j] = cute.arch.fma_packed_f32x2(
            (tTMEM_LOADrS_ex2[k, j], tTMEM_LOADrS_ex2[k + 1, j]),
            (scale, scale),
            (minus_row_max_scale, minus_row_max_scale),
          )
        if const_expr(self.ex2_emu_freq == 0):
          tTMEM_LOADrS_ex2[
            k, j] = cute.math.exp2(tTMEM_LOADrS_ex2[k, j], fastmath=True)
          tTMEM_LOADrS_ex2[
            k + 1,
            j] = cute.math.exp2(tTMEM_LOADrS_ex2[k + 1, j], fastmath=True)
        else:
          if const_expr(
            k % self.ex2_emu_freq < self.ex2_emu_freq - self.ex2_emu_res
            or j >= ex2_frg_cnt - 1 or j < self.ex2_emu_start_frg
          ):
            tTMEM_LOADrS_ex2[
              k, j] = cute.math.exp2(tTMEM_LOADrS_ex2[k, j], fastmath=True)
            tTMEM_LOADrS_ex2[
              k + 1,
              j] = cute.math.exp2(tTMEM_LOADrS_ex2[k + 1, j], fastmath=True)
          else:
            tTMEM_LOADrS_ex2[k,
                             j], tTMEM_LOADrS_ex2[k + 1, j] = ex2_emulation_2(
                               tTMEM_LOADrS_ex2[k, j], tTMEM_LOADrS_ex2[k + 1,
                                                                        j]
                             )
      tTMEM_STORErP_ex2[None, j].store(
        tTMEM_LOADrS_ex2[None, j].load().to(self.q_dtype)
      )
    # P bridge: tcgen05.st is lane-locked, but PV needs a lane's full K=128.
    # K_INTER in i64 units: 8x16 B core rows, 8-key blocks, one 64x128 P slot.
    i64_per_core_row = 2
    i64_per_core = 8 * i64_per_core_row
    i64_per_key_block = (self.cta_tiler[0] // 8) * i64_per_core
    i64_per_slot = i64_per_key_block * (self.qk_mma_tiler[1] // 8)
    i64_per_half = self.qk_mma_tiler[1] // 2 // 4
    p_row = thread_idx % 64
    sPd = cute.make_tensor(
      cute.recast_ptr(sP.iterator, dtype=Int64),
      cute.make_layout(i64_per_slot * self.qk_acc_stage),
    )
    rPd = cute.make_tensor(
      cute.recast_ptr(tTMEM_STORErP.iterator, dtype=Int64),
      cute.make_layout(i64_per_half),
    )
    _dslot = p_handle.index * i64_per_slot
    _dbase = (p_row % 8) * i64_per_core_row + (p_row // 8) * i64_per_core
    _dhalf = (thread_idx // 64) * i64_per_half
    for _j in cutlass.range_constexpr(i64_per_half):
      _wj = _dhalf + _j
      # One 64-bit store per 4 bf16; these dominate the measured wavefronts.
      sPd[_dslot + _dbase + (_wj % i64_per_core_row) +
          (_wj // i64_per_core_row) * i64_per_key_block] = rPd[_j]
    cute.arch.fence_view_async_shared()
    # Commit after the fence wakes PV; the UMMA p release guards sP reuse.
    p_handle.commit()

    # The *0.5 offsets the (row_sum, row_sum) seed; they must change together.
    acc_scale = corr_scale * 0.5
    row_sum *= acc_scale
    local_row_sum_0 = (row_sum, row_sum)
    local_row_sum_1 = (0.0, 0.0)
    local_row_sum_2 = (0.0, 0.0)
    local_row_sum_3 = (0.0, 0.0)
    reduction_unroll = 4
    frg_tile = cute.size(tTMEM_LOADrS) // reduction_unroll
    tTMEM_LOADrS_frg = cute.logical_divide(
      tTMEM_LOADrS, cute.make_layout(frg_tile)
    )
    for j in cutlass.range_constexpr(
      0, cute.size(tTMEM_LOADrS_frg, mode=[0]), 2
    ):
      local_row_sum_0 = cute.arch.add_packed_f32x2(
        local_row_sum_0, (tTMEM_LOADrS_frg[j, 0], tTMEM_LOADrS_frg[j + 1, 0])
      )
      local_row_sum_1 = cute.arch.add_packed_f32x2(
        local_row_sum_1, (tTMEM_LOADrS_frg[j, 1], tTMEM_LOADrS_frg[j + 1, 1])
      )
      local_row_sum_2 = cute.arch.add_packed_f32x2(
        local_row_sum_2, (tTMEM_LOADrS_frg[j, 2], tTMEM_LOADrS_frg[j + 1, 2])
      )
      local_row_sum_3 = cute.arch.add_packed_f32x2(
        local_row_sum_3, (tTMEM_LOADrS_frg[j, 3], tTMEM_LOADrS_frg[j + 1, 3])
      )
    local_row_sum_0 = cute.arch.add_packed_f32x2(
      local_row_sum_0, local_row_sum_1
    )
    local_row_sum_2 = cute.arch.add_packed_f32x2(
      local_row_sum_2, local_row_sum_3
    )
    local_row_sum_0 = cute.arch.add_packed_f32x2(
      local_row_sum_0, local_row_sum_2
    )
    row_sum = local_row_sum_0[0] + local_row_sum_0[1]
    return (
      row_max,
      row_sum,
      mma_s_consumer,
      p_mma_producer,
      s_corr_producer,
    )

  @cute.jit
  def correction_rescale(
    self,
    s_corr_consumer: pipeline.PipelineConsumer,
    sStats: cute.Tensor,
    mma_o_consumer: pipeline.PipelineConsumer,
    tOtO_staged: cute.Tensor,
    cO_staged: cute.Tensor,
    epi_tile: cute.Tile,
  ) -> Tuple[pipeline.PipelineConsumer, pipeline.PipelineConsumer]:
    """Correction warps: rescale resident O by the row factor from sStats."""
    tidx, _, _ = cute.arch.thread_idx()
    thread_idx = tidx % self.threads_per_softmax_group

    stats_handle = s_corr_consumer.wait_and_advance()
    # M128: thread t owns O row t%64, column-half t//64.
    pair_row = thread_idx % 64
    scale = sStats[stats_handle.index, pair_row]
    stats_handle.release()
    mma_o_consumer = self.rescale_o(
      scale,
      mma_o_consumer,
      tOtO_staged,
      cO_staged,
      epi_tile,
    )
    return mma_o_consumer, s_corr_consumer

  @cute.jit
  def rescale_o(
    self,
    scale: Float32,
    mma_o_consumer: pipeline.PipelineConsumer,
    tOtO_staged: cute.Tensor,
    cO_staged: cute.Tensor,
    epi_tile: cute.Tile,
  ) -> pipeline.PipelineConsumer:
    """Scale every resident O slice of this row by `scale` in TMEM."""
    tidx, _, _ = cute.arch.thread_idx()
    thread_idx = tidx % self.threads_per_softmax_group
    # o_handle turns outside the guard: the MMA ring cannot see the skip.
    o_handle = mma_o_consumer.wait_and_advance()
    # x*1.0 is bitwise identity; the vote is warp-wide as tcgen05.ld/st are.
    if cute.arch.vote_any_sync(scale != Float32(1.0)):
      # One copy for all 4 slices: a TiledCopy has no address until partition_*.
      tOtO_epi_proto = cute.zipped_divide(
        tOtO_staged[(None, None), 0, 0, 0], epi_tile
      )
      tmem_load_atom = cute.make_copy_atom(
        tcgen05.Ld32x32bOp(tcgen05.Repetition(16)),
        self.pv_acc_dtype,
      )
      tmem_tiled_load = tcgen05.make_tmem_copy(tmem_load_atom, tOtO_epi_proto)
      thr_load = tmem_tiled_load.get_slice(thread_idx)
      tmem_store_atom = cute.make_copy_atom(
        tcgen05.St32x32bOp(tcgen05.Repetition(16)),
        self.pv_acc_dtype,
      )
      tmem_store_atom = tcgen05.make_tmem_copy(tmem_store_atom, tOtO_epi_proto)
      thr_store = tmem_store_atom.get_slice(thread_idx)
      for dv_slice in cutlass.range(self.iterations_pv, unroll_full=True):
        tOtO = tOtO_staged[(None, None), 0, 0, dv_slice]
        cO = cO_staged[None, None, dv_slice]
        tOtO_epi = cute.zipped_divide(tOtO, epi_tile)
        cO_epi = cute.zipped_divide(cO, epi_tile)
        tTMEM_LOADtO = thr_load.partition_S(tOtO_epi)
        tTMEM_LOADcO = thr_load.partition_D(cO_epi)
        tTMEM_STOREtO = thr_store.partition_D(tOtO_epi)
        tTMrO = cute.make_rmem_tensor_like(
          cute.append(
            cute.make_layout(tTMEM_LOADcO[None, 0, 0].shape),
            cute.make_layout(
              2, stride=cute.size(tTMEM_LOADcO[None, 0, 0].shape)
            ),
          ),
          self.pv_acc_dtype,
        )
        tTMEM_LOADtO_0 = tTMEM_LOADtO[None, 0, 0]
        cute.copy(tmem_tiled_load, tTMEM_LOADtO_0, tTMrO[None, 0])
        iter_num = cute.size(tTMEM_LOADtO, mode=[1])
        for i in cutlass.range(1, iter_num, unroll_full=True):
          tTMEM_LOADtO_i = tTMEM_LOADtO[None, i, 0]
          cute.copy(tmem_tiled_load, tTMEM_LOADtO_i, tTMrO[None, i % 2])
          for j in cutlass.range(
            0, cute.size(tTMrO, mode=[0]), 2, unroll_full=True
          ):
            tTMrO[j, (i - 1) %
                  2], tTMrO[j + 1, (i - 1) % 2] = cute.arch.mul_packed_f32x2(
                    (tTMrO[j, (i - 1) % 2], tTMrO[j + 1, (i - 1) % 2]),
                    (scale, scale),
                  )
          tTMEM_STOREtO_prev_i = tTMEM_STOREtO[None, i - 1, 0]
          cute.copy(
            tmem_store_atom, tTMrO[None, (i - 1) % 2], tTMEM_STOREtO_prev_i
          )

        for j in cutlass.range(
          0, cute.size(tTMrO, mode=[0]), 2, unroll_full=True
        ):
          tTMrO[j, (iter_num - 1) % 2], tTMrO[j + 1, (iter_num - 1) % 2] = (
            cute.arch.mul_packed_f32x2(
              (
                tTMrO[j, (iter_num - 1) % 2],
                tTMrO[j + 1, (iter_num - 1) % 2],
              ),
              (scale, scale),
            )
          )
        cute.copy(
          tmem_store_atom,
          tTMrO[None, (iter_num - 1) % 2],
          tTMEM_STOREtO[None, iter_num - 1, 0],
        )
      cute.arch.fence_view_async_tmem_store()
    o_handle.release()
    return mma_o_consumer

  @cute.jit
  def correction_epilog(
    self,
    seqlen_q: Int32,
    sum_consumer: pipeline.PipelineConsumer,
    sSum: cute.Tensor,
    mma_o_consumer: pipeline.PipelineConsumer,
    gO_staged: cute.Tensor,
    cO_staged: cute.Tensor,
    tOtO_staged: cute.Tensor,
    epi_tile: cute.Tile,
  ) -> Tuple[pipeline.PipelineConsumer, pipeline.PipelineConsumer]:
    """Correction warps: finish O with the row sum and store it (and LSE)."""
    tidx, _, _ = cute.arch.thread_idx()
    thread_idx = tidx % self.threads_per_softmax_group
    sum_handle = sum_consumer.wait_and_advance()
    row_sum = sSum[thread_idx % 64] + sSum[thread_idx % 64 + 64]
    cute.arch.fence_view_async_shared()
    sum_handle.release()
    # Two ranges: the O wait is other agents' latency, the store is ours.
    self._range_push("corr_o_wait")
    o_handle = mma_o_consumer.wait_and_advance()
    self._range_pop()
    self._range_push("corr_store_o")
    self.store_O_to_gmem(
      row_sum,
      seqlen_q,
      gO_staged,
      cO_staged,
      tOtO_staged,
      epi_tile,
    )
    self._range_pop()
    o_handle.release()
    self._stamp("correction.epilogue_end")
    return mma_o_consumer, sum_consumer

  @cute.jit
  def store_O_to_gmem(
    self,
    row_sum: Float32,
    seqlen_q: Int32,
    gO_staged: cute.Tensor,
    cO_staged: cute.Tensor,
    tOtO_staged: cute.Tensor,
    epi_tile: cute.Tile,
  ):
    """Normalise one O tile by row_sum and store it to global memory."""
    row_sum_is_zero_or_nan = row_sum == 0.0 or row_sum != row_sum
    # The O scale is the softmax normalizer only; no runtime output scale.
    scale = 1.0 / row_sum if not row_sum_is_zero_or_nan else 0.0
    tidx, _, _ = cute.arch.thread_idx()
    thread_idx = tidx % self.threads_per_softmax_group
    # Built once from slice 0; per-slice addresses enter at `partition_S`.
    tOtO_epi_proto = cute.zipped_divide(
      tOtO_staged[(None, None), 0, 0, 0], epi_tile
    )
    tmem_copy_atom = sm100_utils_basic.get_tmem_load_op(
      self.pv_block_tiler[:2],
      self.o_layout,
      self.o_dtype,
      self.pv_acc_dtype,
      epi_tile,
      True,
      tmem_warp_shape_mn=self.epi_warp_shape_mn,
    )
    tiled_tmem_load = tcgen05.make_tmem_copy(tmem_copy_atom, tOtO_epi_proto)
    thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)
    # Per Note [AccumulatorHandoff]: bit count converted, never element-chosen.
    r2g_copy_atom = copy_utils.get_copy_atom(
      self.o_dtype, self.r2g_bits_per_copy // self.o_dtype.width
    )
    # The handoff: the same thread keeps the O it read; `tiled_copy_2d` cannot.
    tiled_r2g = cute.make_tiled_copy_D(r2g_copy_atom, tiled_tmem_load)
    thr_r2g = tiled_r2g.get_slice(thread_idx)
    for dv_slice in cutlass.range_constexpr(self.iterations_pv):
      gO = gO_staged[None, None, dv_slice]
      cO = cO_staged[None, None, dv_slice]
      tOtO = tOtO_staged[(None, None), 0, 0, dv_slice]
      # assumed_align=16 is a hard precondition: D ptr alignment 16 < 128 fails.
      gO = cute.make_tensor(
        cute.make_ptr(
          self.o_dtype,
          gO.iterator.toint(),
          cute.AddressSpace.gmem,
          assumed_align=16,
        ),
        gO.layout,
      )
      tOtO_epi = cute.zipped_divide(tOtO, epi_tile)
      cO_epi = cute.zipped_divide(cO, epi_tile)
      gO_epi = cute.zipped_divide(gO, epi_tile)
      tR2GcO = thr_r2g.partition_S(cO_epi)
      tR2GcO_t2r = thr_tmem_load.partition_D(tR2GcO[(None, None), 0, 0])
      tTMEM_LOADtO = thr_tmem_load.partition_S(tOtO_epi)
      tTMEM_LOADcO = thr_tmem_load.partition_D(cO_epi)
      tR2GgO = thr_r2g.partition_D(gO_epi)
      tTMrO = cute.make_rmem_tensor(tR2GcO_t2r.shape, self.pv_acc_dtype)
      cute.copy(tiled_tmem_load, tTMEM_LOADtO, tTMrO)
      # Zero-trip work may read a dummy V holding NaN/Inf; NaN * 0 is still NaN.
      if row_sum_is_zero_or_nan:
        tTMrO.fill(0.0)
      tR2GrO_f32 = thr_r2g.retile(tTMrO)
      tR2GrO = cute.make_rmem_tensor_like(tR2GrO_f32, self.o_dtype)
      tR2GrO.store((tR2GrO_f32.load() * scale).to(self.o_dtype))
      if cute.elem_less(tTMEM_LOADcO[0][0], seqlen_q):
        cute.copy(thr_r2g, tR2GrO, tR2GgO)

  @cute.jit
  def store_row_sum_and_lse(
    self,
    row_max,
    mLSE,
    row_sum,
    sSum,
    sum_producer,
    current_block_coord,
    seqlen_q,
    mCuSeqlensQ,
    cuseqlen_q,
    scale_softmax,
  ):
    """Publish the row sum to correction and write LSE for this row."""
    tidx, _, _ = cute.arch.thread_idx()
    thread_idx = tidx % self.threads_per_softmax_group
    sum_handle = sum_producer.acquire_and_advance()
    sSum[thread_idx] = row_sum
    cute.arch.fence_view_async_shared()
    sum_handle.commit()

    if const_expr(mLSE is not None):
      # The pair barrier orders the peer's sSum write before our read.
      self.softmax_pair_barrier.arrive_and_wait()
      row = thread_idx % 64
      total_sum = sSum[row] + sSum[row + 64]
      total_is_zero_or_nan = total_sum == 0.0 or total_sum != total_sum
      q_idx = current_block_coord[0] * self.cta_tiler[0] + row
      hb_idx = ((current_block_coord[2][0], Int32(0)) if
                const_expr(mCuSeqlensQ is not None) else current_block_coord[2])
      lse_value = (
        scale_softmax * row_max + cute.math.log(total_sum, fastmath=True)
        if not total_is_zero_or_nan else -Float32.inf
      )
      # Both halves hold the same max; N-half 0 writes LSE once per row.
      if thread_idx < 64 and cute.elem_less(q_idx, seqlen_q):
        global_q_idx = (
          q_idx + cuseqlen_q if const_expr(mCuSeqlensQ is not None) else q_idx
        )
        mLSE[global_q_idx, hb_idx] = lse_value
    # Whether the LSE tail extends past the rest of the item's boundary.
    self._stamp("softmax.lse_tail")
    return sum_producer


# ffpa-attn addition: configuration witness asserted by the SM100 tests.
def compile_key_fields(kernel: "FFPAAttnFwdSm100D512") -> tuple:
  """__init__-decided config; later constants derive from it (+wrapper axes)."""
  return (
    type(kernel).__name__,
    kernel.qk_acc_dtype,
    kernel.pv_acc_dtype,
    kernel.cta_tiler,
    kernel.qk_mma_tiler,
    kernel.pv_mma_tiler,
    kernel.pv_block_tiler,
    kernel.iterations_qk,
    kernel.iterations_pv,
    kernel.cta_group_size,
    kernel.cluster_shape_mn,
    kernel.tmem_o_slice_stride,
    kernel.r2g_bits_per_copy,
    kernel.is_persistent,
    kernel.is_causal,
    kernel.is_local,
    kernel.use_varlen_compact_scheduler,
    kernel.head_serpentine,
    kernel.head_serpentine_phase,
    kernel.varlen_lpt,
    kernel.use_semantic_trip_range,
    kernel.v_load_warp_id,
    kernel.iket_stamps,
    kernel.softmax_warp_ids,
    kernel.correction_warp_ids,
    kernel.mma_warp_id,
    kernel.load_warp_id,
    kernel.reg_trim_warp_ids,
    kernel.tmem_alloc_cols,
    kernel.threads_per_softmax_group,
    kernel.threads_per_cta,
    (
      kernel.tmem_alloc_barrier.barrier_id,
      kernel.tmem_alloc_barrier.num_threads
    ),
    (
      kernel.softmax_pair_barrier.barrier_id,
      kernel.softmax_pair_barrier.num_threads
    ),
    kernel.tmem_s_offset,
    kernel.tmem_o_offset,
    kernel.num_regs_softmax,
    kernel.num_regs_correction,
    kernel.num_regs_other,
    kernel.ex2_emu_freq,
    kernel.ex2_emu_res,
    kernel.ex2_emu_start_frg,
    kernel.buffer_align_bytes,
  )

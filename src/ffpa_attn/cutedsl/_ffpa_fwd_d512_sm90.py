# Copyright (c) DefTruth, qyjdef@163.com
# Copyright (c) Butterfingrz，13524387014@163.com
# SM90 (Hopper) forward pass for flash attention — only SplitD for head_dim=512.
#
# Design:
#   - Producer WG: warp-0 issues TMA for Q (one-shot per tile), K and V
#     (per n_block, K/V double-staged for TMA-WGMMA overlap).
#   - WG1: QK-GEMM (full-D in one WGMMA) → online softmax → PV-front (atom 0,
#     gO cols 0:256) + LSE.
#   - WG2: PV-back only (atom 1, gO cols 256:512). Cross-WG sP/sScale handoff
#     via NamedBarrierFwd.{PFull, PEmpty, ScaleReady}.
#   - Epilogue: dual-slot sO (sO[0]=WG1, sO[1]=WG2), each WG fires its own
#     TMA store_O concurrently. sV ↔ sO live in a cute.union (mainloop /
#     epilogue mutually exclusive lifetimes).
#
# Constraints:
#   - tile_m == 64 (single M-atom WGMMA)
#   - tile_n == 32 (so K can carry a true 2-stage pipeline at hdim=512)
#   - tile_hdimv % 256 == 0 (PV (1,2,1) split)
#   - kv_same must be False (requires an independent V buffer)
#   - No paged KV / learnable_sink / seqused / block_sparsity

from typing import Callable, Optional, Type
from functools import partial
import math

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Boolean, const_expr
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.utils import LayoutEnum
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass import pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.base_dsl.arch import Arch
from cutlass.base_dsl import BaseDSL

from quack import copy_utils
from quack import layout_utils
from quack import sm90_utils

from .utils.cute_dsl_utils import assume_tensor_aligned
from . import utils
from .utils.mask import AttentionMask
from .utils.softmax import (
  Softmax,
  apply_score_mod_inner,
  store_scale_to_smem,
  load_scale_from_smem,
  rescale_O_with_external_scale,
)
from .utils.seqlen_info import SeqlenInfoQK
from .utils.block_info import BlockInfo
from .utils import pipeline as pipeline_custom
from .utils.pack_gqa import PackGQA, pack_gqa_layout, make_packgqa_tiled_tma_atom
from .utils.named_barrier import NamedBarrierFwd
from quack.cute_dsl_utils import ParamsBase
from .utils.tile_scheduler import (
  TileSchedulerArguments,
  SingleTileScheduler,
  SingleTileLPTScheduler,
  SingleTileVarlenScheduler,
)
from cutlass.cute import FastDivmodDivisor


class FlashAttentionForwardSm90TrainOnly:
  """SM90 forward kernel for head_dim=512 (kv_same must be False for now).

    3-role pipeline:
      - Producer WG: independent TMA loader (warp-0 elect_one).
      - WG1: QK + softmax + PV-front + LSE + epilogue (gO cols 0:hdimv/2).
      - WG2: PV-back + epilogue (gO cols hdimv/2:hdimv).

    Full-D Q/K WGMMA (no D-chunk loop), tile_n=32 with 2-stage K and 2-stage V
    pipelines. sV ↔ sO share storage via cute.union (mainloop / epilogue
    mutually exclusive). Cross-WG sP/sScale handoff via named barriers.
    """

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
    self.dtype = dtype
    hdim_multiple_of = 16
    self.tile_hdim = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
    head_dim_v = head_dim_v if head_dim_v is not None else head_dim
    self.tile_hdimv = int(math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of)
    self.check_hdim_oob = head_dim != self.tile_hdim
    self.check_hdim_v_oob = head_dim_v != self.tile_hdimv
    self.qhead_per_kvhead = qhead_per_kvhead
    self.is_causal = is_causal
    self.is_local = is_local
    self.pack_gqa = pack_gqa
    self.tile_m = tile_m
    self.tile_n = tile_n
    self.score_mod = score_mod
    self.mask_mod = mask_mod
    self.qk_acc_dtype = Float32
    self.vec_size: cutlass.Constexpr = getattr(
      score_mod, "__vec_size__", 1 if cutlass.const_expr(has_aux_tensors) else 2
    )
    self.arch = BaseDSL._get_dsl().get_arch_enum()

    # ── kv_same not support for yet  ──
    self.kv_same = kv_same
    assert not kv_same, (
      "_ffpa_fwd_d512_sm90 only supports kv_same=False. "
      "Use the V1 kv_same=True forward path if you need that variant."
    )

    # ── SM90 SplitD specific ──
    # SS-mode PV (sP in SMEM): required for cross-WG handoff
    self.mma_pv_is_rs = False
    # 128 matches TMA cache-line and satisfies all swizzle constraints
    # 8 buffers near the 228KB ceiling).
    self.buffer_align_bytes = 128
    self.use_tma_KV = True
    self.cluster_shape_mn = (1, 1)
    assert self.arch >= Arch.sm_90 and self.arch <= Arch.sm_90a, "Only SM 9.x is supported"
    assert not self.pack_gqa or self.tile_m % self.qhead_per_kvhead == 0, (
      f"SplitD requires tile_m ({self.tile_m}) divisible by qhead_per_kvhead "
      f"({self.qhead_per_kvhead}) when pack_gqa=True. "
      f"Use pack_gqa=False for this head configuration."
    )

    # ════════════════════════════════════════════════════════════════════
    # 3-role pipeline — 1 TMA producer WG + 2 MMA WGs
    # ════════════════════════════════════════════════════════════════════
    self.tile_n = 32
    self.tile_hdimv_half = self.tile_hdimv // 2  # 256
    self.tile_hdim_full = self.tile_hdim  # 512

    self.num_stages_q = 1
    self.num_stages_k = 2  # double-stage K
    self.num_stages_v = 2  # multi-stage V (PV pipeline)
    # sScale double-stage: required when stages_k>=2 because WG1 iter
    # k+2 store_scales(s_{k+1}) may overlap WG2 iter k+1 load_scales(s_k);
    # double buffer lets them physically separate
    self.num_stages_p = 2
    # Defense: sScale must be at least as deeply staged as K to avoid
    # the race described above.
    assert self.num_stages_p >= self.num_stages_k, (
      f"num_stages_p ({self.num_stages_p}) must be >= num_stages_k "
      f"({self.num_stages_k}) to prevent sScale write/read race"
    )
    self.num_stages_sP_buf = 3
    self.num_producer_regs = 24  # producer WG (FA3 baseline)
    self.num_mma_regs_wg1 = 240  # 4×32×240 = 30720
    self.num_mma_regs_wg2 = 240  # 4×32×240 = 30720
    assert self.tile_m == 64, "SplitD requires tile_m == 64"
    assert self.tile_hdimv % 256 == 0, "PV (1,2,1) requires tile_hdimv % 256 == 0"

  def _get_tiled_mma(self):
    # QK-GEMM: S[tile_m, tile_n] = Q[tile_m, full_D] @ K[full_D, tile_n]
    tiled_mma_qk = sm90_utils_basic.make_trivial_tiled_mma(
      self.dtype,
      self.dtype,
      warpgroup.OperandMajorMode.K,
      warpgroup.OperandMajorMode.K,
      Float32,
      atom_layout_mnk=(self.tile_m // 64, 1, 1),
      tiler_mn=(64, self.tile_n),
    )
    # 2-WG asymmetric PV along N
    tiled_mma_pv = sm90_utils_basic.make_trivial_tiled_mma(
      self.dtype,
      self.dtype,
      warpgroup.OperandMajorMode.K,
      warpgroup.OperandMajorMode.MN,
      Float32,
      atom_layout_mnk=(1, 2, 1),
      tiler_mn=(64, self.tile_hdimv_half),
      a_source=warpgroup.OperandSource.SMEM,
    )
    assert tiled_mma_pv.size == 256, "PV TiledMma must lower to 256 threads"
    # RS-mode tiled_mma for WG1's PV-front only.
    tiled_mma_pv_wg1 = sm90_utils_basic.make_trivial_tiled_mma(
      self.dtype,
      self.dtype,
      warpgroup.OperandMajorMode.K,
      warpgroup.OperandMajorMode.MN,
      Float32,
      atom_layout_mnk=(1, 2, 1),
      tiler_mn=(64, self.tile_hdimv_half),
      a_source=warpgroup.OperandSource.RMEM,
    )
    assert tiled_mma_pv_wg1.size == 256, "WG1 RS-mode tiled_mma_pv shape must match SS-mode"
    # Epilogue mini tiled MMA: per-WG single-atom (64, hdimv_half) shape.
    tiled_mma_pv_epi = sm90_utils_basic.make_trivial_tiled_mma(
      self.dtype,
      self.dtype,
      warpgroup.OperandMajorMode.K,
      warpgroup.OperandMajorMode.MN,
      Float32,
      atom_layout_mnk=(1, 1, 1),
      tiler_mn=(64, self.tile_hdimv_half),
      a_source=warpgroup.OperandSource.SMEM,
    )
    assert tiled_mma_pv_epi.size == 128
    return tiled_mma_qk, tiled_mma_pv, tiled_mma_pv_wg1, tiled_mma_pv_epi

  def _get_shared_storage_cls(self):
    """(2-WG async PV) shared storage: sQ/sK + sV/sO union + sP/sScale.

        sV (mainloop) and sO (epilogue) have mutually-exclusive lifetimes —
        sV is released before epilogue starts; sO is only written during
        epilogue. Use cute.union so they share the same physical SMEM
        """
    sQ_struct, sK_struct, sV_struct, sP_struct, sO_struct = [
      cute.struct.Align[
        cute.struct.MemRange[self.dtype, cute.cosize(layout)],
        self.buffer_align_bytes,
      ] for layout in (
        self.sQ_layout,
        self.sK_layout,
        self.sV_layout,
        self.sP_layout,
        self.sO_layout,
      )
    ]
    # sScale: per-row fp32 (tile_m × num_stages_p)
    sScale_struct = cute.struct.Align[
      cute.struct.MemRange[Float32, cute.cosize(self.sScale_layout)],
      self.buffer_align_bytes,
    ]
    mbar_ptr_Q_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages_q * 2]
    mbar_ptr_K_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages_k * 2]
    mbar_ptr_V_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages_v * 2]

    # pipeline_P mbarrier backing (2 stages × 2 phases = 4 Int64 = 32 B).
    mbar_ptr_P_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages_sP_buf * 2]

    # pipeline_Scale mbarrier backing (num_stages_p stages × 2 = 4 Int64).
    mbar_ptr_Scale_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages_p * 2]

    @cute.union
    class SmemVO_t:
      sV: sV_struct  # mainloop: PV-front + PV-back B operand
      sO: sO_struct  # epilogue: per-WG half-width staging (sO[wg_idx])

    @cute.struct
    class SharedStorage:
      mbar_ptr_Q: mbar_ptr_Q_struct
      mbar_ptr_K: mbar_ptr_K_struct
      mbar_ptr_V: mbar_ptr_V_struct
      mbar_ptr_P: mbar_ptr_P_struct  # pipeline_P mbarrier storage
      mbar_ptr_Scale: mbar_ptr_Scale_struct  # pipeline_Scale mbarrier storage
      sQ: sQ_struct
      sK: sK_struct
      sVO: SmemVO_t  # union: sV (mainloop) | sO (epilogue)
      sP: sP_struct  # softmax-out → PV A operand (2-stage, 5.5b)
      sScale: sScale_struct  # row_scale handoff WG1→WG2

    return SharedStorage

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
    window_size_left: Int32 | int | None = None,
    window_size_right: Int32 | int | None = None,
    aux_tensors: Optional[list] = None,
    stream: cuda.CUstream = None,
  ):
    # Type check (inlined, SplitD has no mSeqUsedQ/K)
    if const_expr(not (mQ.element_type == mK.element_type == mV.element_type == mO.element_type)):
      raise TypeError("All tensors must have the same data type")
    if const_expr(mQ.element_type not in [cutlass.Float16, cutlass.BFloat16]):
      raise TypeError("Only Float16 or BFloat16 is supported")
    if const_expr(mLSE is not None and mLSE.element_type != Float32):
      raise TypeError("LSE tensor must be Float32")
    if const_expr(mCuSeqlensQ is not None and mCuSeqlensQ.element_type != Int32):
      raise TypeError("cu_seqlens_q tensor must be Int32")
    if const_expr(mCuSeqlensK is not None and mCuSeqlensK.element_type != Int32):
      raise TypeError("cu_seqlens_k tensor must be Int32")
    assert mQ.element_type == self.dtype
    self.varlen_q = mCuSeqlensQ is not None

    mQ, mK, mV, mO = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]
    QO_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
    mQ, mO = [layout_utils.select(t, QO_layout_transpose) for t in (mQ, mO)]
    KV_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
    mK, mV = [layout_utils.select(t, KV_layout_transpose) for t in (mK, mV)]
    LSE_layout_transpose = [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
    mLSE = layout_utils.select(mLSE, LSE_layout_transpose) if const_expr(mLSE is not None) else None

    tiled_mma_qk, tiled_mma_pv, tiled_mma_pv_wg1, tiled_mma_pv_epi = self._get_tiled_mma()
    self.num_threads_per_warp_group = 128

    # 1 producer WG + 2 MMA WGs. Total = 384 threads.
    self.num_mma_threads = tiled_mma_pv.size  # 256 (WG1+WG2)
    self.num_wg_mma = self.num_mma_threads // self.num_threads_per_warp_group  # 2
    assert self.num_wg_mma == 2, "SplitD requires num_wg_mma == 2"

    # Producer WG occupies warp_group_idx==0 (full WG of 128, only warp-0 issues TMAs).
    self.num_producer_threads = self.num_threads_per_warp_group  # 128
    self.num_threads = self.num_threads_per_warp_group * (self.num_wg_mma + 1)  # 384
    self.num_Q_load_threads = self.num_threads_per_warp_group

    # Each WG epilogues its own half-width sO slot independently.
    self.num_epilogue_threads = self.num_threads_per_warp_group
    self.num_mma_regs = self.num_mma_regs_wg1  # display-only; real values are set per WG

    self.use_tma_Q = self.arch >= Arch.sm_90 and not (self.pack_gqa and self.tile_m % self.qhead_per_kvhead != 0)
    self.use_tma_O = self.use_tma_Q

    # ═══ SMEM layouts (full-D Q/K, multi-stage K/V pipeline) ═══
    # sQ: full-D (64×512), single stage. QK becomes one full-D WGMMA.
    self.sQ_layout = sm90_utils.make_smem_layout(
      mQ.element_type,
      LayoutEnum.ROW_MAJOR,
      (self.tile_m, self.tile_hdim_full),
      stage=self.num_stages_q,
    )
    # sK: full-D (tile_n=32 × 512), num_stages_k=2 stages — true K-stage pipeline
    self.sK_layout = sm90_utils.make_smem_layout(
      mK.element_type,
      LayoutEnum.ROW_MAJOR,
      (self.tile_n, self.tile_hdim_full),
      stage=self.num_stages_k,
    )
    # sV: (tile_n=32, tile_hdimv=512), num_stages_v=2 stages
    # PV B operand wants MN-major (N=tile_hdimv contiguous) → use ROW_MAJOR.
    self.sV_layout = sm90_utils.make_smem_layout(
      mV.element_type,
      LayoutEnum.ROW_MAJOR,
      (self.tile_n, self.tile_hdimv),
      stage=self.num_stages_v,
    )
    # sP: K-major (A operand), (tile_m=64, tile_n=32) × stage=num_stages_sP_buf
    # WG1 writes sP[k%N] while WG2 reads
    # sP[(k-1)%N]; sP[(k-2)%N] is empty (released last iter).
    # Decouples STSM(sP) from LDS(sP) to hide long_scoreboard.
    self.sP_layout = sm90_utils.make_smem_layout(
      self.dtype,
      LayoutEnum.ROW_MAJOR,
      (self.tile_m, self.tile_n),
      stage=self.num_stages_sP_buf,
    )
    # sScale: fp32, (tile_m, kStages=2). Double-buffered for store_scales(prev)
    # vs load_scales(current) staggering (online softmax across K-stages).
    self.sScale_layout = cute.make_layout(
      (self.tile_m, self.num_stages_p),
      stride=(self.num_stages_p + 1, 1),
    )
    # sO: half-width double-staged (64 KB total) — one slot per MMA WG.
    self.sO_layout = sm90_utils.make_smem_layout(
      mO.element_type,
      LayoutEnum.ROW_MAJOR,
      (self.tile_m, self.tile_hdimv_half),
      stage=2,
    )

    SharedStorage = self._get_shared_storage_cls()

    mQ_og, mO_og = mQ, mO
    if const_expr(self.pack_gqa):
      nheads_kv = mK.shape[2]
      mQ = pack_gqa_layout(mQ, self.qhead_per_kvhead, nheads_kv, head_idx=2)
      mO = pack_gqa_layout(mO, self.qhead_per_kvhead, nheads_kv, head_idx=2)
      if const_expr(mLSE is not None):
        mLSE = pack_gqa_layout(mLSE, self.qhead_per_kvhead, nheads_kv, head_idx=1)

    gmem_tiled_copy_Q = cpasync.CopyBulkTensorTileG2SOp()
    gmem_tiled_copy_KV = cpasync.CopyBulkTensorTileG2SOp()
    gmem_tiled_copy_O = cpasync.CopyBulkTensorTileS2GOp()

    # full-D Q/K + full-hdimv V per K-stage
    self.tma_copy_bytes = {
      "Q_full": cute.size_in_bytes(mQ.element_type, cute.select(self.sQ_layout, mode=[0, 1])),
      "K_full": cute.size_in_bytes(mK.element_type, cute.select(self.sK_layout, mode=[0, 1])),
      "V_full": cute.size_in_bytes(mV.element_type, cute.select(self.sV_layout, mode=[0, 1])),
    }

    make_tiled_tma_atom_fn = (
      partial(make_packgqa_tiled_tma_atom, qhead_per_kvhead=self.qhead_per_kvhead, head_idx=2)
      if const_expr(self.pack_gqa) else cpasync.make_tiled_tma_atom
    )

    # loads Q/K full-D in one TMA per work-tile / K-stage
    qk_box_d = self.tile_hdim_full
    tma_atom_Q, tma_tensor_Q = None, None
    if const_expr(self.use_tma_Q):
      tma_atom_Q, tma_tensor_Q = make_tiled_tma_atom_fn(
        gmem_tiled_copy_Q,
        mQ_og if const_expr(self.pack_gqa) else mQ,
        cute.select(self.sQ_layout, mode=[0, 1]),
        (self.tile_m, qk_box_d),
      )
    tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
      gmem_tiled_copy_KV,
      mK,
      cute.select(self.sK_layout, mode=[0, 1]),
      (self.tile_n, qk_box_d),
      1,
    )
    # V TMA atom: loads full hdimv into sV.
    tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
      gmem_tiled_copy_KV,
      mV,
      cute.select(self.sV_layout, mode=[0, 1]),  # full hdimv, MN-major
      (self.tile_n, self.tile_hdimv),
      1,
    )
    tma_atom_O, tma_tensor_O = None, None
    if const_expr(self.use_tma_O):
      mO_tma = mO_og if const_expr(self.pack_gqa) else mO
      if const_expr(self.varlen_q):
        mO_tma = copy_utils.create_ragged_tensor_for_tma(mO_tma, ragged_dim=0, ptr_shift=True)
      # each WG stores its own half-width tile (tile_m, tile_hdimv_half)
      tma_atom_O, tma_tensor_O = make_tiled_tma_atom_fn(
        gmem_tiled_copy_O,
        mO_tma,
        cute.select(self.sO_layout, mode=[0, 1]),
        (self.tile_m, self.tile_hdimv_half),
      )

    if const_expr(mCuSeqlensQ is not None):
      TileScheduler = SingleTileVarlenScheduler
    else:
      TileScheduler = (
        SingleTileScheduler if const_expr(not self.is_causal or self.is_local) else SingleTileLPTScheduler
      )
    tile_sched_args = TileSchedulerArguments(
      cute.ceil_div(cute.size(mQ.shape[0]), self.tile_m),
      cute.size(mQ.shape[2]),
      cute.size(mQ.shape[3]) if const_expr(mCuSeqlensQ is None) else cute.size(mCuSeqlensQ.shape[0] - 1),
      1,
      cute.size(mK.shape[0]),
      mQ.shape[1],
      mV.shape[1],
      total_q=cute.size(mQ.shape[0]) if const_expr(mCuSeqlensQ is not None) else cute.size(mQ.shape[0]) *
      cute.size(mQ.shape[3]),
      tile_shape_mn=(self.tile_m, self.tile_n),
      mCuSeqlensQ=mCuSeqlensQ,
      qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
      element_size=self.dtype.width // 8,
      lpt=self.is_causal or self.is_local,
    )
    tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
    grid_dim = TileScheduler.get_grid_shape(tile_sched_params)
    softmax_scale_log2, softmax_scale = utils.compute_softmax_scale_log2(softmax_scale, self.score_mod)
    window_size_left = Int32(window_size_left) if window_size_left is not None else None
    window_size_right = Int32(window_size_right) if window_size_right is not None else None
    fastdiv_mods = utils.compute_fastdiv_mods(mQ, mK, self.qhead_per_kvhead, self.pack_gqa, aux_tensors, None)

    self.kernel(
      tma_tensor_Q if const_expr(self.use_tma_Q) else mQ,
      tma_tensor_K,
      tma_tensor_V,
      tma_tensor_O if const_expr(self.use_tma_O) else mO,
      mLSE,
      mCuSeqlensQ,
      mCuSeqlensK,
      tma_atom_Q,
      tma_atom_K,
      tma_atom_V,
      tma_atom_O,
      softmax_scale_log2,
      softmax_scale,
      window_size_left,
      window_size_right,
      self.sQ_layout,
      self.sK_layout,
      self.sV_layout,
      self.sP_layout,
      self.sScale_layout,
      self.sO_layout,
      tiled_mma_qk,
      tiled_mma_pv,
      tiled_mma_pv_wg1,  # RS-mode tiled_mma for WG1 PV-front
      tiled_mma_pv_epi,
      tile_sched_params,
      TileScheduler,
      SharedStorage,
      aux_tensors,
      fastdiv_mods,
    ).launch(
      grid=grid_dim,
      block=[self.num_threads, 1, 1],
      stream=stream,
      min_blocks_per_mp=1,
    )

  # ════════════════════════════════════════════════════════════════════════
  # 2-WG asymmetric PV + folded TMA producer
  # ════════════════════════════════════════════════════════════════════════
  @cute.kernel
  def kernel(
    self,
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mO: cute.Tensor,
    mLSE: Optional[cute.Tensor],
    mCuSeqlensQ: Optional[cute.Tensor],
    mCuSeqlensK: Optional[cute.Tensor],
    tma_atom_Q: Optional[cute.CopyAtom],
    tma_atom_K: cute.CopyAtom,
    tma_atom_V: cute.CopyAtom,
    tma_atom_O: Optional[cute.CopyAtom],
    softmax_scale_log2: Float32,
    softmax_scale: Optional[Float32],
    window_size_left: Optional[Int32],
    window_size_right: Optional[Int32],
    sQ_layout: cute.ComposedLayout,
    sK_layout: cute.ComposedLayout,
    sV_layout: cute.ComposedLayout,
    sP_layout: cute.ComposedLayout,
    sScale_layout: cute.Layout,
    sO_layout: cute.ComposedLayout,
    tiled_mma_qk: cute.TiledMma,
    tiled_mma_pv: cute.TiledMma,
    tiled_mma_pv_wg1: cute.TiledMma,  # RS-mode for WG1 PV-front
    tiled_mma_pv_epi: cute.TiledMma,
    tile_sched_params: ParamsBase,
    TileScheduler: cutlass.Constexpr[Callable],
    SharedStorage: cutlass.Constexpr[Callable],
    aux_tensors=Optional[list[cute.Tensor]],
    fastdiv_mods=None,
  ):
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    if warp_idx == 0:
      for tma_atom in (tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_O):
        if const_expr(tma_atom is not None):
          cpasync.prefetch_descriptor(tma_atom)

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)

    ThreadCooperativeGroup = partial(pipeline.CooperativeGroup, pipeline.Agent.Thread)
    # Producer: independent producer WG, warp-0 via elect_one → 1 thread per TMA
    tma_producer_group = ThreadCooperativeGroup(1)

    # Q/K consumer: only WG1 (4 warps = 128 threads).
    # cg.size = #warps because each warp does 1 elect_one release per cycle.
    qk_consumer_group = ThreadCooperativeGroup(self.num_threads_per_warp_group // cute.arch.WARP_SIZE)

    # V consumer: WG1 + WG2 = 2 MMA warpgroups = 8 warps. CRITICAL: use
    # CRITICAL: use num_mma_threads (256, MMA-only), NOT num_threads
    v_consumer_group = ThreadCooperativeGroup(self.num_mma_threads // cute.arch.WARP_SIZE)

    pipeline_q = pipeline_custom.PipelineTmaAsync.create(
      barrier_storage=storage.mbar_ptr_Q.data_ptr(),
      num_stages=self.num_stages_q,
      producer_group=tma_producer_group,
      consumer_group=qk_consumer_group,
      tx_count=self.tma_copy_bytes["Q_full"],
      defer_sync=True,
    )
    pipeline_k = pipeline_custom.PipelineTmaAsync.create(
      barrier_storage=storage.mbar_ptr_K.data_ptr(),
      num_stages=self.num_stages_k,
      producer_group=tma_producer_group,
      consumer_group=qk_consumer_group,
      tx_count=self.tma_copy_bytes["K_full"],
      defer_sync=True,
    )
    pipeline_v = pipeline_custom.PipelineTmaAsync.create(
      barrier_storage=storage.mbar_ptr_V.data_ptr(),
      num_stages=self.num_stages_v,
      producer_group=tma_producer_group,
      consumer_group=v_consumer_group,
      tx_count=self.tma_copy_bytes["V_full"],
      defer_sync=True,
    )

    # cross-WG sP handshake via mbarrier (replaces PFull/PEmpty named
    # barrier). Producer = WG1 (128 threads, writes sP[k]); consumer = WG2
    # (128 threads, reads sP[k-1]).
    sP_producer_group = ThreadCooperativeGroup(self.num_threads_per_warp_group // cute.arch.WARP_SIZE)
    sP_consumer_group = ThreadCooperativeGroup(self.num_threads_per_warp_group // cute.arch.WARP_SIZE)
    pipeline_P = pipeline_custom.PipelineAsync.create(
      barrier_storage=storage.mbar_ptr_P.data_ptr(),
      num_stages=self.num_stages_sP_buf,
      producer_group=sP_producer_group,
      consumer_group=sP_consumer_group,
      # WG1 already issues fence_view_async_shared + sync_warp before
      # commit; PipelineAsync.create's own syncwarp would be redundant.
      elect_one_commit=True,
      syncwarp_before_commit=False,
      elect_one_release=True,
      syncwarp_before_release=False,
    )

    # pipeline_Scale replaces NamedBarrierFwd.ScaleReady in mainloop.
    # Same template as pipeline_P (WG1 producer, WG2 consumer, both 128
    # threads = 4 warps). num_stages = num_stages_p (2) matches sScale layout.
    sScale_producer_group = ThreadCooperativeGroup(self.num_threads_per_warp_group // cute.arch.WARP_SIZE)
    sScale_consumer_group = ThreadCooperativeGroup(self.num_threads_per_warp_group // cute.arch.WARP_SIZE)
    pipeline_Scale = pipeline_custom.PipelineAsync.create(
      barrier_storage=storage.mbar_ptr_Scale.data_ptr(),
      num_stages=self.num_stages_p,
      producer_group=sScale_producer_group,
      consumer_group=sScale_consumer_group,
      elect_one_commit=True,
      syncwarp_before_commit=False,
      elect_one_release=True,
      syncwarp_before_release=False,
    )

    pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

    # ═══ SMEM tensors ═══
    sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
    sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)

    # sV and sO share storage.sVO (cute.union)
    sV = storage.sVO.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
    # sVt is the transpose-view of sV with shape
    sVt = layout_utils.transpose_view(sV)
    sP = storage.sP.get_tensor(sP_layout.outer, swizzle=sP_layout.inner)
    sScale = storage.sScale.get_tensor(sScale_layout, dtype=Float32)
    sO = storage.sVO.sO.get_tensor(sO_layout.outer, swizzle=sO_layout.inner, dtype=self.dtype)

    block_info = BlockInfo(
      self.tile_m,
      self.tile_n,
      self.is_causal,
      self.is_local,
      False,
      window_size_left,
      window_size_right,
      qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
    )
    SeqlenInfoCls = partial(
      SeqlenInfoQK.create,
      seqlen_q_static=mQ.shape[0] if const_expr(not self.pack_gqa) else mQ.shape[0][1],
      seqlen_k_static=mK.shape[0],
      mCuSeqlensQ=mCuSeqlensQ,
      mCuSeqlensK=mCuSeqlensK,
    )
    AttentionMaskCls = partial(
      AttentionMask,
      self.tile_m,
      self.tile_n,
      window_size_left=window_size_left,
      window_size_right=window_size_right,
      qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
    )
    TileSchedulerCls = partial(TileScheduler.create, tile_sched_params)

    pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

    tidx, _, _ = cute.arch.thread_idx()
    warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)

    if warp_group_idx == 0:
      # Producer WG: drops register count so MMA WGs can take more.
      cute.arch.setmaxregister_decrease(self.num_producer_regs)
      self.producer(
        mQ,
        mK,
        mV,
        sQ,
        sK,
        sV,
        tma_atom_Q,
        tma_atom_K,
        tma_atom_V,
        pipeline_q,
        pipeline_k,
        pipeline_v,
        block_info,
        SeqlenInfoCls,
        TileSchedulerCls,
      )
    elif warp_group_idx == 1:
      # WG1 (QK + softmax + PV-front): tidx ∈ [128, 256), pass tidx_in_wg ∈ [0, 128)
      cute.arch.setmaxregister_increase(self.num_mma_regs_wg1)
      tidx_in_wg = tidx - self.num_threads_per_warp_group
      self.mma_wg1(
        tiled_mma_qk,
        tiled_mma_pv,
        tiled_mma_pv_wg1,  # RS-mode tiled_mma for WG1 PV-front
        tiled_mma_pv_epi,
        mO,
        mLSE,
        sQ,
        sK,
        sVt,  # transpose-view for PV partition_B
        sP,
        sScale,
        sO,
        pipeline_q,
        pipeline_k,
        pipeline_v,
        pipeline_P,  # sP cross-WG mbarrier
        pipeline_Scale,  # sScale cross-WG mbarrier
        tma_atom_O,
        tidx_in_wg,
        softmax_scale_log2,
        softmax_scale,
        block_info,
        SeqlenInfoCls,
        AttentionMaskCls,
        TileSchedulerCls,
        aux_tensors,
        fastdiv_mods,
      )
    else:
      # WG2 (PV-back only — no producer role). tidx ∈ [256, 384).
      cute.arch.setmaxregister_increase(self.num_mma_regs_wg2)
      tidx_in_wg = tidx - 2 * self.num_threads_per_warp_group
      self.mma_wg2(
        tiled_mma_pv,
        tiled_mma_pv_epi,
        mO,
        sVt,  # for PV partition_B
        sP,
        sScale,
        sO,
        tma_atom_O,
        pipeline_v,
        pipeline_P,  # sP cross-WG mbarrier
        pipeline_Scale,  # sScale cross-WG mbarrier
        tidx_in_wg,
        block_info,
        SeqlenInfoCls,
        TileSchedulerCls,
      )

  @cute.jit
  def producer(
    self,
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    sQ: cute.Tensor,
    sK: cute.Tensor,
    sV: cute.Tensor,
    tma_atom_Q: Optional[cute.CopyAtom],
    tma_atom_K: cute.CopyAtom,
    tma_atom_V: cute.CopyAtom,
    pipeline_q: pipeline.PipelineAsync,
    pipeline_k: pipeline.PipelineAsync,
    pipeline_v: pipeline.PipelineAsync,
    block_info: BlockInfo,
    SeqlenInfoCls: Callable,
    TileSchedulerCls: Callable,
  ):
    """Independent TMA producer warp group (warp_group_idx==0).

        Only warp-0 (elect_one) issues TMA loads; the remaining 96 threads idle
        with reduced register budget. Q is loaded once per work-tile (full-D
        64×512); K and V are loaded per K-stage along the n_block sweep.

        Synchronization: consumer-side pipelines (pipeline_q/k/v) carry an
        empty-mbarrier credit cycle, so producer_acquire on each load naturally
        waits for WG1/WG2 to release the previous slot. This subsumes the
        QueryEmpty / barrier_Q semantics under CUTe DSL's PipelineTmaAsync.
        """
    warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
    is_tma_warp = warp_idx_in_wg == 0

    if is_tma_warp:
      q_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_stages_q)
      k_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_stages_k)
      v_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_stages_v)

      did_produce_q = Boolean(False)
      did_produce_k = Boolean(False)
      did_produce_v = Boolean(False)

      tile_scheduler = TileSchedulerCls()
      work_tile = tile_scheduler.initial_work_tile_info()

      while work_tile.is_valid_tile:
        m_block, head_idx, batch_idx, _ = work_tile.tile_idx
        seqlen = SeqlenInfoCls(batch_idx)
        mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
        head_idx_kv = (head_idx // self.qhead_per_kvhead if const_expr(not self.pack_gqa) else head_idx)
        mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[None, None, head_idx_kv]
        mV_cur = seqlen.offset_batch_K(mV, batch_idx, dim=3)[None, None, head_idx_kv]

        n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)
        n_block_count = n_block_max - n_block_min
        if n_block_count < 0:
          n_block_count = Int32(0)

        if n_block_count > 0:
          # ── Q load (full-D, one-shot per work-tile) ──
          gQ = cute.local_tile(mQ_cur, (self.tile_m, self.tile_hdim_full), (None, 0))
          load_Q, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_Q,
            0,
            cute.make_layout(1),
            gQ,
            sQ,
          )
          pipeline_q.producer_acquire(q_producer_state)
          load_Q(
            src_idx=m_block,
            dst_idx=q_producer_state.index,
            tma_bar_ptr=pipeline_q.producer_get_barrier(q_producer_state),
          )
          pipeline_q.producer_commit(q_producer_state)
          did_produce_q = Boolean(True)
          q_producer_state.advance()

          # ── K/V loop along n_blocks (reverse order matches consumer) ──
          for i_n in cutlass.range(n_block_count, unroll=1):
            n_block = n_block_max - 1 - i_n
            # K full-D
            gK = cute.local_tile(mK_cur, (self.tile_n, self.tile_hdim_full), (None, 0))
            load_K, _, _ = copy_utils.tma_get_copy_fn(
              tma_atom_K,
              0,
              cute.make_layout(1),
              gK,
              sK,
            )
            pipeline_k.producer_acquire(k_producer_state)
            load_K(
              src_idx=n_block,
              dst_idx=k_producer_state.index,
              tma_bar_ptr=pipeline_k.producer_get_barrier(k_producer_state),
            )
            pipeline_k.producer_commit(k_producer_state)
            did_produce_k = Boolean(True)
            k_producer_state.advance()
            # V full-hdimv (per K-stage)
            gV = cute.local_tile(mV_cur, (self.tile_n, self.tile_hdimv), (None, 0))
            load_V, _, _ = copy_utils.tma_get_copy_fn(
              tma_atom_V,
              0,
              cute.make_layout(1),
              gV,
              sV,
            )
            pipeline_v.producer_acquire(v_producer_state)
            load_V(
              src_idx=n_block,
              dst_idx=v_producer_state.index,
              tma_bar_ptr=pipeline_v.producer_get_barrier(v_producer_state),
            )
            pipeline_v.producer_commit(v_producer_state)
            did_produce_v = Boolean(True)
            v_producer_state.advance()

        tile_scheduler.advance_to_next_work()
        work_tile = tile_scheduler.get_current_work()

      # Producer tail — flush remaining producer states
      if did_produce_q:
        pipeline_q.producer_tail(q_producer_state)
      if did_produce_k:
        pipeline_k.producer_tail(k_producer_state)
      if did_produce_v:
        pipeline_v.producer_tail(v_producer_state)

  @cute.jit
  def mma_wg1(
    self,
    tiled_mma_qk: cute.TiledMma,
    tiled_mma_pv: cute.TiledMma,
    tiled_mma_pv_wg1: cute.TiledMma,  # RS-mode for WG1 PV-front
    tiled_mma_pv_epi: cute.TiledMma,
    mO: cute.Tensor,
    mLSE: Optional[cute.Tensor],
    sQ: cute.Tensor,
    sK: cute.Tensor,
    sVt: cute.Tensor,  # transpose-view of sV; MN-major B operand for PV
    sP: cute.Tensor,
    sScale: cute.Tensor,
    sO: cute.Tensor,
    pipeline_q: pipeline.PipelineAsync,
    pipeline_k: pipeline.PipelineAsync,
    pipeline_v: pipeline.PipelineAsync,
    pipeline_P: pipeline.PipelineAsync,  # cross-WG sP handshake
    pipeline_Scale: pipeline.PipelineAsync,  # cross-WG sScale handshake
    tma_atom_O: Optional[cute.CopyAtom],
    tidx: Int32,
    softmax_scale_log2: Float32,
    softmax_scale: Optional[Float32],
    block_info: BlockInfo,
    SeqlenInfoCls: Callable,
    AttentionMaskCls: Callable,
    TileSchedulerCls: Callable,
    aux_tensors: Optional[list],
    fastdiv_mods=None,
  ):
    """WG1: QK (full-D, single WGMMA) + online softmax + sP/sScale handoff + PV-front + epilogue. tidx ∈ [0, 128) (WG-local).
        """
    warp_group_thread_layout = cute.make_layout(self.num_wg_mma, stride=self.num_threads_per_warp_group)
    thr_mma_qk = tiled_mma_qk.get_slice(tidx)
    wg_mma_qk = tiled_mma_qk.get_slice(warp_group_thread_layout(0))  # WG1 is atom 0
    wg_mma_pv = tiled_mma_pv.get_slice(warp_group_thread_layout(0))  # WG1 covers PV atom 0

    # WG1 PV-front uses RS-mode tiled_mma; same atom_layout (1,2,1),
    # slice 0 selects WG1's N half. partition_fragment_ABC with sA=None
    # signals RS-mode (no SMEM A operand, A comes from tOrP rmem at WGMMA issue).
    wg_mma_pv_wg1 = tiled_mma_pv_wg1.get_slice(warp_group_thread_layout(0))

    # QK fragments — full-D Q/K (one WGMMA per n_block, no D-chunk loop)
    _, tSrQ, tSrK = sm90_utils.partition_fragment_ABC(
      wg_mma_qk,
      (self.tile_m, self.tile_n, self.tile_hdim_full),
      sQ,
      sK,
    )

    # PV-front fragments (SS-mode A from sP, SS-mode B from sVt; B auto-sliced by WG)
    _, tOrP_smem, tOrV = sm90_utils.partition_fragment_ABC(
      wg_mma_pv,
      (self.tile_m, self.tile_hdimv, self.tile_n),
      sP,
      sVt,
    )
    # RS-mode PV-front fragments (B from sVt, A from tOrP rmem at issue)
    _, _tOrP_rmem_template, tOrV_wg1 = sm90_utils.partition_fragment_ABC(
      wg_mma_pv_wg1,
      (self.tile_m, self.tile_hdimv, self.tile_n),
      None,
      sVt,
    )

    # acc_O_front: full (tile_m, tile_hdimv)
    acc_O_front = cute.make_rmem_tensor(
      tiled_mma_pv.partition_shape_C((self.tile_m, self.tile_hdimv)),
      Float32,
    )

    # StMatrix copy atom for writing tOrP (acc_S → bf16) into sP (SMEM)
    # We use the same atom that V1's epilogue uses for bf16 SMEM stores.
    smem_copy_atom_P = utils.get_smem_store_atom(self.arch.major * 10 + self.arch.minor, self.dtype)
    smem_thr_copy_P = cute.make_tiled_copy_C(smem_copy_atom_P, tiled_mma_qk).get_slice(tidx)

    q_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_stages_q)
    k_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_stages_k)
    v_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_stages_v)

    # WG1 is producer on pipeline_P. Initial phase = num_stages
    pipeline_state_P_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_stages_sP_buf)
    # WG1 is producer on pipeline_Scale (same template as pipeline_P).
    pipeline_state_Scale_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_stages_p)

    # softmax row count
    qk_acc_shape = tiled_mma_qk.partition_shape_C((self.tile_m, self.tile_n))
    softmax_num_rows = cute.size(qk_acc_shape[0][0]) * cute.size(qk_acc_shape[1])

    tile_scheduler = TileSchedulerCls()
    work_tile = tile_scheduler.initial_work_tile_info()

    while work_tile.is_valid_tile:
      m_block, head_idx, batch_idx, _ = work_tile.tile_idx
      seqlen = SeqlenInfoCls(batch_idx)

      if const_expr(fastdiv_mods is not None):
        recompute_q = const_expr(aux_tensors is not None and seqlen.has_cu_seqlens_q)
        recompute_k = const_expr(aux_tensors is not None and seqlen.has_cu_seqlens_k)
        seqlen_q_divmod, seqlen_k_divmod = fastdiv_mods
        fastdiv_mods = (
          seqlen_q_divmod if not recompute_q else FastDivmodDivisor(seqlen.seqlen_q),
          seqlen_k_divmod if not recompute_k else FastDivmodDivisor(seqlen.seqlen_k),
        )

      mask = AttentionMaskCls(seqlen)
      mask_fn = partial(
        mask.apply_mask,
        batch_idx=batch_idx,
        head_idx=head_idx,
        m_block=m_block,
        thr_mma=thr_mma_qk,
        mask_causal=self.is_causal,
        mask_local=self.is_local,
        aux_tensors=aux_tensors,
        fastdiv_mods=fastdiv_mods,
      )
      score_mod_fn = None
      if const_expr(self.score_mod is not None):
        score_mod_fn = partial(
          self.apply_score_mod,
          thr_mma_qk,
          batch_idx,
          head_idx,
          m_block,
          softmax_scale=softmax_scale,
          aux_tensors=aux_tensors,
          fastdiv_mods=fastdiv_mods,
        )

      n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)
      n_block_count = n_block_max - n_block_min
      if n_block_count < 0:
        n_block_count = Int32(0)

      # acc_S double buffer. QK[k+1] async-writes into the "nxt" slot
      # while softmax[k] reads the "cur" slot. Caller alternates cur/nxt
      # buffer roles every n_block via the pair-loop structure below.
      # acc_S_a holds QK[0] from the pre-loop pre-issue; iter 0's body
      # writes QK[1] into acc_S_b; etc.
      acc_S_a = cute.make_rmem_tensor(
        tiled_mma_qk.partition_shape_C((self.tile_m, self.tile_n)),
        Float32,
      )
      acc_S_b = cute.make_rmem_tensor(
        tiled_mma_qk.partition_shape_C((self.tile_m, self.tile_n)),
        Float32,
      )
      softmax = Softmax.create(
        softmax_scale_log2,
        num_rows=softmax_num_rows,
        softmax_scale=softmax_scale,
      )

      if n_block_count > 0:
        # ─── Wait Q (one-shot per work-tile, full-D) ───
        pipeline_q.consumer_wait(q_consumer_state, pipeline_q.consumer_try_wait(q_consumer_state))

        # ─── pre-issue QK[0] async into acc_S_a (overlaps with
        # iter 0 body's later work). The iter-0 body picks this up via
        # wait_group(0) at step A. Don't release K[0] here — iter 0's
        # step B handles release after the wait_group drains.
        pipeline_k.consumer_wait(k_consumer_state, pipeline_k.consumer_try_wait(k_consumer_state))
        sm90_utils.gemm_w_idx(
          tiled_mma_qk,
          acc_S_a,
          tSrQ,
          tSrK,
          zero_init=True,
          A_idx=q_consumer_state.index,
          B_idx=k_consumer_state.index,
          wg_wait=-1,
        )

        mma_one_n_block = partial(
          self._mma_wg1_one_n_block,
          acc_O_front=acc_O_front,
          sP=sP,
          sScale=sScale,
          tSrQ=tSrQ,
          tSrK=tSrK,
          tOrP_smem=tOrP_smem,
          tOrV=tOrV,
          tOrV_wg1=tOrV_wg1,  # WG1 RS-mode B operand
          smem_copy_atom_P=smem_copy_atom_P,
          smem_thr_copy_P=smem_thr_copy_P,
          tiled_mma_qk=tiled_mma_qk,
          tiled_mma_pv=tiled_mma_pv,
          tiled_mma_pv_wg1=tiled_mma_pv_wg1,  # WG1 RS-mode tiled_mma
          pipeline_k=pipeline_k,
          pipeline_v=pipeline_v,
          pipeline_P=pipeline_P,
          pipeline_Scale=pipeline_Scale,
          q_consumer_state=q_consumer_state,
          softmax=softmax,
          seqlen=seqlen,
          tidx=tidx,
          score_mod_fn=score_mod_fn,
          mask_fn=mask_fn,
        )

        # ─── first iter — cur=acc_S_a (holds QK[0] from pre-issue),
        # nxt=acc_S_b (will receive QK[1] in step C, if not is_last).
        k_consumer_state, v_consumer_state, pipeline_state_P_producer, pipeline_state_Scale_producer = mma_one_n_block(
          n_block=n_block_max - 1,
          acc_S_cur=acc_S_a,
          acc_S_nxt=acc_S_b,
          k_consumer_state=k_consumer_state,
          v_consumer_state=v_consumer_state,
          pipeline_state_P_producer=pipeline_state_P_producer,
          pipeline_state_Scale_producer=pipeline_state_Scale_producer,
          mask_seqlen=True,
          is_first_n_block=True,
          is_last_n_block=(n_block_count == 1),
        )

        # ─── remaining iters processed in PAIRS. Each pair runs 2
        # n_blocks with cur/nxt swapped so the Python-level tensor refs
        # alternate without runtime dispatch.
        # Iter alternation (post-first):
        #   iter 1: cur=b, nxt=a (b holds QK[1] from first iter's step C)
        #   iter 2: cur=a, nxt=b
        #   iter 3: cur=b, nxt=a
        #   ...
        remaining = n_block_count - 1
        pair_count = remaining // 2
        has_tail = (remaining % 2) == 1

        for i_pair in cutlass.range(pair_count, unroll=1):
          # Pair iter A (overall iter index = 2*i_pair + 1): cur=b, nxt=a
          n_block_A = n_block_max - 2 - 2 * i_pair
          k_consumer_state, v_consumer_state, pipeline_state_P_producer, pipeline_state_Scale_producer = mma_one_n_block(
            n_block=n_block_A,
            acc_S_cur=acc_S_b,
            acc_S_nxt=acc_S_a,
            k_consumer_state=k_consumer_state,
            v_consumer_state=v_consumer_state,
            pipeline_state_P_producer=pipeline_state_P_producer,
            pipeline_state_Scale_producer=pipeline_state_Scale_producer,
            mask_seqlen=False,
            is_first_n_block=False,
            is_last_n_block=False,  # A never last (B follows; or tail handles)
          )
          # Pair iter B (overall iter index = 2*i_pair + 2): cur=a, nxt=b
          n_block_B = n_block_max - 3 - 2 * i_pair
          # B is last iff there is no tail AND this is the final pair
          is_last_B = (not has_tail) and (i_pair == pair_count - 1)
          k_consumer_state, v_consumer_state, pipeline_state_P_producer, pipeline_state_Scale_producer = mma_one_n_block(
            n_block=n_block_B,
            acc_S_cur=acc_S_a,
            acc_S_nxt=acc_S_b,
            k_consumer_state=k_consumer_state,
            v_consumer_state=v_consumer_state,
            pipeline_state_P_producer=pipeline_state_P_producer,
            pipeline_state_Scale_producer=pipeline_state_Scale_producer,
            mask_seqlen=False,
            is_first_n_block=False,
            is_last_n_block=is_last_B,
          )

        # ─── tail iter (if remaining is odd): cur=b, nxt=a, is_last=True
        if has_tail:
          n_block_tail = n_block_max - 2 - 2 * pair_count
          k_consumer_state, v_consumer_state, pipeline_state_P_producer, pipeline_state_Scale_producer = mma_one_n_block(
            n_block=n_block_tail,
            acc_S_cur=acc_S_b,
            acc_S_nxt=acc_S_a,
            k_consumer_state=k_consumer_state,
            v_consumer_state=v_consumer_state,
            pipeline_state_P_producer=pipeline_state_P_producer,
            pipeline_state_Scale_producer=pipeline_state_Scale_producer,
            mask_seqlen=False,
            is_first_n_block=False,
            is_last_n_block=True,
          )

        # ─── drain the last PV-front WGMMA ───
        cute.nvgpu.warpgroup.wait_group(0)
        with cute.arch.elect_one():
          pipeline_v.consumer_release(v_consumer_state)

        # ─── Finalize: write final row_scale to sScale[current] for WG2 ───
        final_stage_idx = Int32(k_consumer_state.index)
        row_scale_final = softmax.finalize(sink_val=None)

        # PFull/PEmpty bridge two warpgroups in producer/consumer style
        cute.arch.barrier(
          barrier_id=int(NamedBarrierFwd.PEmpty),
          number_of_threads=self.num_mma_threads,
        )
        store_scale_to_smem(
          row_scale_final,
          sScale,
          final_stage_idx,
          tiled_mma_qk,
          tidx,
          self.tile_m,
          self.tile_n,
        )
        # fence + sync_warp before final PFull arrive
        cute.arch.fence_view_async_shared()
        cute.arch.sync_warp()
        cute.arch.barrier_arrive(
          barrier_id=int(NamedBarrierFwd.PFull),
          number_of_threads=self.num_mma_threads,
        )
        softmax.rescale_O(acc_O_front, row_scale_final)

        # ─── Release Q so producer can load next work-tile's Q ───
        with cute.arch.elect_one():
          pipeline_q.consumer_release(q_consumer_state)
        q_consumer_state.advance()

        # ─── Epilogue: write O[:, 0:256] + LSE ───
        self.epilogue_wg(
          acc_O_front,
          softmax.row_sum,
          mO,
          mLSE,
          sO,
          seqlen,
          tma_atom_O,
          tiled_mma_pv_epi,
          tidx,
          m_block,
          head_idx,
          batch_idx,
          wg_idx=0,
          write_lse=True,
        )

        # ─── Close the tile's K/V pipeline phase ───
        k_consumer_state.advance()
        v_consumer_state.advance()
      # else: empty tile — interface.py pre-inits O=0, LSE=-inf

      tile_scheduler.advance_to_next_work()
      work_tile = tile_scheduler.get_current_work()

  @cute.jit
  def _mma_wg1_one_n_block(
    self,
    n_block: Int32,
    acc_S_cur: cute.Tensor,  # holds QK[n_block] from prev iter step C (or pre-loop pre-issue on iter 0)
    acc_S_nxt: cute.Tensor,  # receives QK[n_block+1] in step C (if not is_last)
    acc_O_front: cute.Tensor,
    sP: cute.Tensor,
    sScale: cute.Tensor,
    tSrQ: cute.Tensor,
    tSrK: cute.Tensor,
    tOrP_smem: cute.Tensor,
    tOrV: cute.Tensor,
    tOrV_wg1: cute.Tensor,  # WG1 RS-mode B (same as tOrV but typed for RS path)
    smem_copy_atom_P: cute.CopyAtom,
    smem_thr_copy_P: cute.TiledCopy,
    tiled_mma_qk: cute.TiledMma,
    tiled_mma_pv: cute.TiledMma,
    tiled_mma_pv_wg1: cute.TiledMma,  # 5.5f: WG1 RS-mode tiled_mma
    pipeline_k: pipeline.PipelineAsync,
    pipeline_v: pipeline.PipelineAsync,
    pipeline_P: pipeline.PipelineAsync,  # WG1 producer side
    pipeline_Scale: pipeline.PipelineAsync,  # WG1 producer on sScale
    q_consumer_state: pipeline.PipelineState,
    k_consumer_state: pipeline.PipelineState,
    v_consumer_state: pipeline.PipelineState,
    pipeline_state_P_producer,  # producer state (advances each iter)
    pipeline_state_Scale_producer,  # sScale producer state
    softmax: Softmax,
    seqlen: SeqlenInfoQK,
    tidx: Int32,
    is_last_n_block: bool,  # runtime: skip step C pre-issue at end of tile
    score_mod_fn: Optional[Callable] = None,
    mask_fn: Optional[Callable] = None,
    mask_seqlen: cutlass.Constexpr[bool] = False,
    is_first_n_block: cutlass.Constexpr[bool] = False,
  ):
    """async pipeline iteration of WG1.

        Caller alternates acc_S_cur/acc_S_nxt buffer roles every iter via
        explicit kwargs (see mma_wg1's pair-loop) so no runtime slot dispatch
        is needed inside the body.
        """
    p_stage = Int32(pipeline_state_P_producer.index)

    # ═══ step A: drain pre-issued QK[n_block] (and prev iter's PV-front) ═══
    cute.nvgpu.warpgroup.wait_group(0)

    # ═══ step B: release K[n_block]; release V[n_block-1] if not first ═══
    cur_k_stage = Int32(k_consumer_state.index)
    prev_stage_idx = (cur_k_stage + Int32(self.num_stages_k - 1)) % Int32(self.num_stages_k)
    with cute.arch.elect_one():
      pipeline_k.consumer_release(k_consumer_state)
    if const_expr(not is_first_n_block):
      with cute.arch.elect_one():
        pipeline_v.consumer_release(v_consumer_state)

    # ═══ step C: pre-issue QK[n_block+1] async into acc_S_nxt ═══
    if not is_last_n_block:
      k_consumer_state.advance()
      pipeline_k.consumer_wait(k_consumer_state, pipeline_k.consumer_try_wait(k_consumer_state))
      sm90_utils.gemm_w_idx(
        tiled_mma_qk,
        acc_S_nxt,
        tSrQ,
        tSrK,
        zero_init=True,
        A_idx=q_consumer_state.index,  # sQ stage = 0 (single Q buf per tile)
        B_idx=k_consumer_state.index,  # K[n_block+1] stage (post-advance)
        wg_wait=-1,  # async — drained at next iter's step A
      )
      # K[n_block+1] is held; next iter's step B (or finalize) releases.

    # ═══ step D: softmax on acc_S_cur (overlaps with step C's WGMMA on TC) ═══
    if const_expr(score_mod_fn is not None):
      score_mod_fn(acc_S_cur, n_block=n_block, seqlen=seqlen)
    partial(mask_fn, mask_mod=self.mask_mod, mask_seqlen=mask_seqlen)(
      acc_S=acc_S_cur,
      n_block=n_block,
    )
    row_scale = softmax.online_softmax(
      acc_S_cur,
      is_first=is_first_n_block,
      check_inf=True,
    )

    # store_scale + pipeline_Scale
    if const_expr(not is_first_n_block):
      pipeline_Scale.producer_acquire(pipeline_state_Scale_producer)
      store_scale_to_smem(
        row_scale,
        sScale,
        prev_stage_idx,
        tiled_mma_qk,
        tidx,
        self.tile_m,
        self.tile_n,
      )
      # sScale store is generic-proxy (regular SMEM store, not STSM/TMA);
      cute.arch.sync_warp()
      pipeline_Scale.producer_commit(pipeline_state_Scale_producer)
      pipeline_state_Scale_producer.advance()

    # ═══ step E: convert acc_S_cur → tOrP (bf16 in registers) ═══
    tOrP_acc = layout_utils.reshape_acc_to_frgA(acc_S_cur)
    tOrP = cute.make_fragment_like(tOrP_acc, self.dtype)
    tOrP.store(tOrP_acc.load().to(self.dtype))

    # ═══ step F pipeline_P producer_acquire → STSM(sP) → rescale_O →
    # fence → pipeline_P producer_commit
    pipeline_P.producer_acquire(pipeline_state_P_producer)
    cute.copy(
      smem_copy_atom_P,
      smem_thr_copy_P.retile(tOrP),
      smem_thr_copy_P.partition_D(sP[None, None, p_stage]),
    )
    if const_expr(not is_first_n_block):
      softmax.rescale_O(acc_O_front, row_scale)
    cute.arch.fence_view_async_shared()
    cute.arch.sync_warp()
    pipeline_P.producer_commit(pipeline_state_P_producer)
    pipeline_state_P_producer.advance()

    # ═══ step G: advance V (skip on first), wait V[n_block], PV-front async ═══
    if const_expr(not is_first_n_block):
      v_consumer_state.advance()
    pipeline_v.consumer_wait(v_consumer_state, pipeline_v.consumer_try_wait(v_consumer_state))
    # RS-mode PV-front. A operand = tOrP (rmem); WG2 still uses SS-mode
    # via tiled_mma_pv. Eliminates SS-mode SMEM descriptor read for WG1's
    # own PV-front WGMMA — modest TC pipeline startup latency saving (+3
    # TFLOPS empirically).
    sm90_utils.gemm_w_idx(
      tiled_mma_pv_wg1,
      acc_O_front,
      tOrP,
      tOrV_wg1,
      zero_init=is_first_n_block,
      A_idx=None,
      B_idx=v_consumer_state.index,
      wg_wait=-1,  # async — drained at next iter's step A or finalize
    )
    # V[n_block] is held; next iter's step B (or finalize) releases.
    return k_consumer_state, v_consumer_state, pipeline_state_P_producer, pipeline_state_Scale_producer

  @cute.jit
  def mma_wg2(
    self,
    tiled_mma_pv: cute.TiledMma,
    tiled_mma_pv_epi: cute.TiledMma,
    mO: cute.Tensor,
    sVt: cute.Tensor,  # transpose-view for PV partition_B
    sP: cute.Tensor,
    sScale: cute.Tensor,
    sO: cute.Tensor,
    tma_atom_O: Optional[cute.CopyAtom],
    pipeline_v: pipeline.PipelineAsync,
    pipeline_P: pipeline.PipelineAsync,  # cross-WG sP handshake
    pipeline_Scale: pipeline.PipelineAsync,  # cross-WG sScale handshake
    tidx_in_wg: Int32,
    block_info: BlockInfo,
    SeqlenInfoCls: Callable,
    TileSchedulerCls: Callable,
  ):
    """WG2 (style): PV-back consumer (atom 1) + WG2-half epilogue. No producer role.

        tidx_in_wg ∈ [0, 128) is WG-local (CTA-absolute thread is in [256, 384) because
        WG2 sits at warp_group_idx==2). For tiled_mma_pv.get_slice() and load_scale_from_smem
        we need WG2's index *within tiled_mma_pv* which has only 256 threads (2 atoms × 128):
        atom 0 = 0..127 (WG1), atom 1 = 128..255 (WG2). Hence tidx_global = tidx_in_wg + 128,
        not + 256. Passing the CTA-absolute id would index out of the tiled_mma_pv slice
        range and deadlock the PV consumer.
        """
    tidx_global = tidx_in_wg + self.num_threads_per_warp_group  # ∈ [128, 256), WG2 = atom 1 slice

    warp_group_thread_layout = cute.make_layout(self.num_wg_mma, stride=self.num_threads_per_warp_group)
    # WG2 covers PV atom 1; use the WG-level slice for fragments
    wg_mma_pv = tiled_mma_pv.get_slice(warp_group_thread_layout(1))

    # PV-back fragments (sVt has N=tile_hdimv contiguous as required by MN-major B)
    _, tOrP_smem, tOrV = sm90_utils.partition_fragment_ABC(
      wg_mma_pv,
      (self.tile_m, self.tile_hdimv, self.tile_n),
      sP,
      sVt,
    )
    # acc_O_back: per-thread fragment (WG2's slice → atom-1 / cols 256:512)
    acc_O_back = cute.make_rmem_tensor(
      tiled_mma_pv.partition_shape_C((self.tile_m, self.tile_hdimv)),
      Float32,
    )

    # ── Startup: WG1's main-loop PFull/PEmpty are migrated to pipeline_P
    cute.arch.barrier_arrive(
      barrier_id=int(NamedBarrierFwd.PEmpty),
      number_of_threads=self.num_mma_threads,
    )

    v_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_stages_v)
    # WG2 is consumer on pipeline_P. Initial phase=0 — first wait
    # blocks until WG1's first producer_commit arrives.
    pipeline_state_P_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_stages_sP_buf)
    # WG2 is consumer on pipeline_Scale. Same Consumer init pattern.
    pipeline_state_Scale_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_stages_p)

    tile_scheduler = TileSchedulerCls()
    work_tile = tile_scheduler.initial_work_tile_info()

    while work_tile.is_valid_tile:
      m_block, head_idx, batch_idx, _ = work_tile.tile_idx
      seqlen = SeqlenInfoCls(batch_idx)

      n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)
      n_block_count = n_block_max - n_block_min
      if n_block_count < 0:
        n_block_count = Int32(0)

      if n_block_count > 0:
        mma_one_n_block = partial(
          self._mma_wg2_one_n_block,
          acc_O_back=acc_O_back,
          sP=sP,
          sScale=sScale,
          tOrP_smem=tOrP_smem,
          tOrV=tOrV,
          tiled_mma_pv=tiled_mma_pv,
          pipeline_v=pipeline_v,
          pipeline_P=pipeline_P,
          pipeline_Scale=pipeline_Scale,
          tidx_global=tidx_global,
        )
        # First n_block: is_first=True. No stage advance; consume stage 0.
        v_consumer_state, pipeline_state_P_consumer, pipeline_state_Scale_consumer = mma_one_n_block(
          n_block=n_block_max - 1,
          v_consumer_state=v_consumer_state,
          pipeline_state_P_consumer=pipeline_state_P_consumer,
          pipeline_state_Scale_consumer=pipeline_state_Scale_consumer,
          is_first_n_block=True,
        )
        # Remaining n_blocks (in reverse). Each one advances V stage.
        for i_n in cutlass.range(n_block_count - 1, unroll=1):
          n_block = n_block_max - 2 - i_n
          v_consumer_state, pipeline_state_P_consumer, pipeline_state_Scale_consumer = mma_one_n_block(
            n_block=n_block,
            v_consumer_state=v_consumer_state,
            pipeline_state_P_consumer=pipeline_state_P_consumer,
            pipeline_state_Scale_consumer=pipeline_state_Scale_consumer,
            is_first_n_block=False,
          )

        # ─── Finalize: receive final row_scale from WG1 ───
        # Stage index alignment:
        cute.arch.barrier(
          barrier_id=int(NamedBarrierFwd.PFull),
          number_of_threads=self.num_mma_threads,
        )
        row_scale_final = load_scale_from_smem(
          sScale,
          Int32(v_consumer_state.index),
          tiled_mma_pv,
          tidx_global,
          self.tile_m,
          self.tile_hdimv_half,
          num_rows=2,
        )
        rescale_O_with_external_scale(acc_O_back, row_scale_final)
        # Polite-close arrive on PEmpty
        # normalizes barrier counter for persistent-kernel buffer reuse
        cute.arch.barrier_arrive(
          barrier_id=int(NamedBarrierFwd.PEmpty),
          number_of_threads=self.num_mma_threads,
        )
        # ─── Epilogue: write O[:, 256:512] ───
        self.epilogue_wg(
          acc_O_back,
          None,
          mO,
          None,
          sO,
          seqlen,
          tma_atom_O,
          tiled_mma_pv_epi,
          tidx_in_wg,
          m_block,
          head_idx,
          batch_idx,
          wg_idx=1,
          write_lse=False,
        )

        # ─── Close the tile's V pipeline phase ───
        v_consumer_state.advance()
      # else: empty tile; interface.py pre-inits O=0

      tile_scheduler.advance_to_next_work()
      work_tile = tile_scheduler.get_current_work()

  @cute.jit
  def _mma_wg2_one_n_block(
    self,
    n_block: Int32,
    acc_O_back: cute.Tensor,
    sP: cute.Tensor,
    sScale: cute.Tensor,
    tOrP_smem: cute.Tensor,
    tOrV: cute.Tensor,
    tiled_mma_pv: cute.TiledMma,
    pipeline_v: pipeline.PipelineAsync,
    pipeline_P: pipeline.PipelineAsync,  # WG2 consumer side
    pipeline_Scale: pipeline.PipelineAsync,  # WG2 consumer on sScale
    v_consumer_state: pipeline.PipelineState,
    pipeline_state_P_consumer,  # consumer state (advances each iter)
    pipeline_state_Scale_consumer,  # sScale consumer state
    tidx_global: Int32,
    is_first_n_block: cutlass.Constexpr[bool] = False,
  ):
    """One K-stage iteration of WG2: PV-back (atom 1) + cross-WG sP/sScale handoff.

        Stage semantics:
          - sync(PFull) — wait WG1 to publish sP[0] + sScale[prev].
          - load_scales(current) — read sScale[v_consumer_state.index] AFTER
            barrier; WG1 already wrote into the slot indexed by the *previous*
            iter (in WG1's view), which equals WG2's *current* index.
          - Then advance V stage (non-first) and consume sV[v.index] for PV-back.
        """
    p_stage = Int32(pipeline_state_P_consumer.index)

    if const_expr(not is_first_n_block):
      # pipeline_Scale.consumer_wait replaces barrier(ScaleReady).
      # WG2 waits for WG1's producer_commit on the sScale slot it's about
      # to read. Slot index is v_consumer_state.index (= stage_of_V[k-1]),
      # same physical slot WG1 wrote via prev_stage_idx.
      pipeline_Scale.consumer_wait(pipeline_state_Scale_consumer)
      row_scale_back = load_scale_from_smem(
        sScale,
        Int32(v_consumer_state.index),
        tiled_mma_pv,
        tidx_global,
        self.tile_m,
        self.tile_hdimv_half,
        num_rows=2,
      )
      pipeline_Scale.consumer_release(pipeline_state_Scale_consumer)
      pipeline_state_Scale_consumer.advance()
      v_consumer_state.advance()
      v_phase = pipeline_v.consumer_try_wait(v_consumer_state)
      rescale_O_with_external_scale(acc_O_back, row_scale_back)
    else:
      # First iter: WG1 didn't write sScale; just pre-issue V try_wait.
      v_phase = pipeline_v.consumer_try_wait(v_consumer_state)

    # ─── Wait sP ready (5.5b: pipeline_P.consumer_wait replaces PFull) ───
    pipeline_P.consumer_wait(pipeline_state_P_consumer)
    pipeline_v.consumer_wait(v_consumer_state, v_phase)
    sm90_utils.gemm_w_idx(
      tiled_mma_pv,
      acc_O_back,
      tOrP_smem,
      tOrV,
      zero_init=is_first_n_block,
      A_idx=p_stage,
      B_idx=v_consumer_state.index,
      wg_wait=0,
    )
    # Release V (each WG contributes 4 elect_one arrivals; total 8 = cg.size).
    with cute.arch.elect_one():
      pipeline_v.consumer_release(v_consumer_state)
    # Note: don't advance v_consumer_state here — next iter does
    # snapshot+advance via the returned state; symmetric with WG1's pattern.

    # ─── 5.5b: consumer_release on pipeline_P (replaces PEmpty arrive) ───
    pipeline_P.consumer_release(pipeline_state_P_consumer)
    pipeline_state_P_consumer.advance()
    return v_consumer_state, pipeline_state_P_consumer, pipeline_state_Scale_consumer

  @cute.jit
  def epilogue_wg(
    self,
    acc_O: cute.Tensor,
    lse: Optional[cute.Tensor],
    mO: cute.Tensor,
    mLSE: Optional[cute.Tensor],
    sO: cute.Tensor,
    seqlen: SeqlenInfoQK,
    tma_atom_O: cute.CopyAtom,
    tiled_mma_pv: cute.TiledMma,
    tidx: Int32,
    m_block: Int32,
    head_idx: Int32,
    batch_idx: Int32,
    wg_idx: cutlass.Constexpr[int],
    write_lse: cutlass.Constexpr[bool] = False,
  ):
    """per-WG epilogue with concurrent sO writes (Opt-3, dual-slot sO).

        sO has 2 stages — WG1 owns sO[0], WG2 owns sO[1]. Both WGs STSM into
        their own slot and fire TMA store_O concurrently. Only 2 Epilogue
        barriers remain:
          1) sync at start (sV → sO union-slot lifetime transition)
          2) final sync (TMA stores complete before next tile reuses SMEM)

        LSE writes only happen in WG1 and use gmem directly (no sO conflict),
        so they happen before the sO write region.
        """
    # Convert acc_O (fp32) → rO (bf16) in registers — per-thread, no SMEM.
    rO = cute.make_fragment_like(acc_O, self.dtype)
    rO.store(acc_O.load().to(self.dtype))

    # SMEM/TMA fragments (compile-time setup; no SMEM access yet)
    smem_copy_atom_O = utils.get_smem_store_atom(self.arch.major * 10 + self.arch.minor, self.dtype)
    smem_thr_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O, tiled_mma_pv).get_slice(tidx)
    taccOrO = smem_thr_copy_O.retile(rO)
    taccOsO = smem_thr_copy_O.partition_D(sO[None, None, wg_idx])

    ragged = seqlen.has_cu_seqlens_q
    mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3, ragged=ragged)[None, None, head_idx]
    gO = cute.local_tile(mO_cur, (self.tile_m, self.tile_hdimv_half), (m_block, wg_idx))
    store_O, _, _ = copy_utils.tma_get_copy_fn(
      tma_atom_O,
      0,
      cute.make_layout(1),
      sO[None, None, wg_idx],
      gO,
      single_stage=True,
    )
    warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4

    # ── LSE: WG1 only, gmem-only write (no sO interaction) ──
    if const_expr(write_lse and mLSE is not None):
      cO = cute.make_identity_tensor((self.tile_m, self.tile_hdimv_half))
      mLSE_cur = seqlen.offset_batch_Q(mLSE, batch_idx, dim=2)[None, head_idx]
      if const_expr(not self.pack_gqa):
        gLSE = cute.local_tile(mLSE_cur, (self.tile_m, ), (m_block, ))
        gLSE_expanded_layout = cute.append(gLSE.layout, cute.make_layout((self.tile_hdimv_half, ), stride=(0, )))
        gLSE_expanded = cute.make_tensor(gLSE.iterator, gLSE_expanded_layout)
        thr_mma = tiled_mma_pv.get_slice(tidx)
        taccOgLSE = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(gLSE_expanded))
        taccOcO = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(cO))
        t0accOcO = layout_utils.reshape_acc_to_mn(tiled_mma_pv.get_slice(0).partition_C(cO))
        if taccOcO[0][1] == 0:
          for m in cutlass.range(cute.size(taccOgLSE.shape[1]), unroll_full=True):
            if t0accOcO[m, 0][0] < seqlen.seqlen_q - m_block * self.tile_m - taccOcO[0][0]:
              taccOgLSE[m, 0] = lse[m]
      else:
        pack_gqa = PackGQA(
          self.tile_m,
          self.tile_hdimv_half,
          self.check_hdim_v_oob,
          self.qhead_per_kvhead,
        )
        pack_gqa.store_LSE(mLSE_cur, lse, tiled_mma_pv, tidx, m_block, seqlen.seqlen_q)

    # ── Concurrent epilogue: WG1→sO[0], WG2→sO[1] (dual-slot sO) ──
    cute.arch.fence_view_async_shared()
    cute.arch.barrier(
      barrier_id=int(NamedBarrierFwd.Epilogue),
      number_of_threads=self.num_mma_threads,
    )
    cute.copy(smem_copy_atom_O, taccOrO, taccOsO)
    cute.arch.fence_view_async_shared()
    if warp_idx_in_wg == 0:
      store_O()
      cute.arch.cp_async_bulk_commit_group()
      cute.arch.cp_async_bulk_wait_group(0, read=True)
    # Final barrier: both WGs' TMA store_O have committed and the SMEM
    # union slot is free again — required so the next work-tile can safely
    # reuse the slot for sV loads.
    cute.arch.barrier(
      barrier_id=int(NamedBarrierFwd.Epilogue),
      number_of_threads=self.num_mma_threads,
    )

  @cute.jit
  def apply_score_mod(
    self,
    thr_mma_qk,
    batch_idx,
    head_idx,
    m_block,
    acc_S,
    n_block,
    softmax_scale,
    seqlen,
    aux_tensors: Optional[list] = None,
    fastdiv_mods=None,
  ):
    cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
    cS = cute.domain_offset((m_block * self.tile_m, n_block * self.tile_n), cS)
    tScS = thr_mma_qk.partition_C(cS)
    apply_score_mod_inner(
      acc_S,
      tScS,
      self.score_mod,
      batch_idx,
      head_idx,
      softmax_scale,
      self.vec_size,
      self.qk_acc_dtype,
      aux_tensors,
      fastdiv_mods,
      seqlen_info=seqlen,
      constant_q_idx=None,
      qhead_per_kvhead=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
    )

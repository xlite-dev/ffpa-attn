# Copyright (c) DefTruth, qyjdef@163.com
# Copyright (c) Butterfingrz，13524387014@163.com
#
# The idea of splitting the backward pass into a separate dQ kernel is
# inspired by
# https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/sm100_hd256_2cta_fmha_backward_dqkernel.py
# The core implementation below is written from scratch for SM90 and follows
# the SplitD design from the ffpa-attn repo.
#
# SM90 Backward dQ Kernel  : dual asymmetric MMA WG + d_chunk=256
#                          + loop nest reversed (n_block outer, d_pass inner)
#                          + cooperative ① N-axis split for Phase E.
#
# Architecture:
#   - 3 WGs: 1 TMA producer (warp 0..3 worker) + WG1 (S/softmax/dQ_front) + WG2 (dP/dS/dQ_back)
#   - tile_m=64, tile_n=64, d_chunk=256, num_d_passes=2, num_d_inner=2, dQ_n_half=128
#   - Loop nest: for work_tile: for n_block (outer, reversed): {Phase A/B/C/D once} +
#                                  for d_pass (inner): cooperative Phase E (½ wgmma per WG)
#     → Eliminates v2's S/dP recompute across d_passes (40% wgmma reduction).
#   - Per n_block, 5 phases under dual-WG asymmetric protocol:
#       Phase A (WG1): S = ΣQ_d @ K_d^T  (num_d_inner=2 WGMMA, K from sB)
#       Phase B (WG1): mask + softmax → P fp32; STSM → sP_fp32; arrive(PFull, 256);
#                      sync(dSFull, 256) before entering Phase E
#       Phase C (WG2): dP = ΣdO_d @ V_d^T  (num_d_inner=2 WGMMA, V from sB)
#       Phase D (WG2): sync(PFull); s2r sP_fp32; dS = P*(dP-dPsum)*scale;
#                      sync(dSEmpty); STSM → sdS; sync(dSLocal); arrive(PEmpty);
#                      arrive(dSFull)
#       Phase E (inner d_pass loop, cooperative ½ WGMMA per WG):
#                      WG1: acc_dQ_pass[d_pass]_front += sdS @ sKt_front  (M=64,N=128,K=64)
#                      WG2: acc_dQ_pass[d_pass]_back  += sdS @ sKt_back   (M=64,N=128,K=64)
#                      After inner loop: WG1 arrives(dSEmpty, 256); WG2 does NOT arrive
#                      (intra-WG sdS write→read ordering covered by dSLocal; a WG2 arrive
#                      would double-count to 384 and break the 256-counter cycle).
#   - acc_dQ_pass[0..num_d_passes-1]_{front,back} persist in rmem across all n_blocks of the
#     work_tile; per-work_tile epilogue writes each (d_pass, half) slice via sEpi_{front,back}.
#   - 3 pipelines:
#       pipeline_QdO (A_stage=2, 8-warp shared): Q+dO streaming per n_block
#       pipeline_B   (B_stage=2, 8-warp shared): K+V streaming per n_block
#       pipeline_Kt  (Kt_stage=2, 8-warp shared): each (d_pass) ferries Kt_front+Kt_back
#                                                  via single mbar (tx_count = full Kt bytes).
#   - cute.union(sKt_front, sEpi_front) and cute.union(sKt_back, sEpi_back) — mainloop vs
#     epilogue lifetime disjoint
#   - Cross-WG handshakes: PFull/PEmpty (sP_fp32) + dSFull/dSEmpty (sdS) + dSLocal (WG2-internal)
#     + Epilogue (256, per work_tile) + WSWG1 (288, per work_tile).

import math
from typing import Callable, Optional, Type
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass.cute.nvgpu import OperandMajorMode, cpasync
from cutlass.cute import FastDivmodDivisor
from cutlass import Float32, Int32, Boolean, const_expr
from cutlass.utils import LayoutEnum

from quack import copy_utils
from quack import layout_utils
from quack import sm90_utils
from quack.sm90_utils import gemm_w_idx

from .utils.cute_dsl_utils import assume_tensor_aligned
from . import utils
from .utils.mask import AttentionMask
from .utils.seqlen_info import SeqlenInfoQK
from .utils.block_info import BlockInfo
from .utils import pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from quack.cute_dsl_utils import ParamsBase
from .utils.tile_scheduler import (
  TileSchedulerArguments,
  SingleTileScheduler,
  SingleTileVarlenScheduler,
)
from .utils.named_barrier import NamedBarrierBwd


class FFPAAttnBwdDQSm90SplitD:
  """SM90 backward dQ kernel (dQ: cooperative dual WG + loop reversal).

    Computes only dQ (no dK/dV). dK/dV is handled by FFPAAttnBwdDKDVSm90SplitD.
    Q-stationary at the work_tile level (each CTA owns one m_block × head × batch),
    Q/dO/K/V streamed per n_block via shared pipelines.

    """

  arch = 90

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
    self.dtype = dtype
    hdim_multiple_of = 16
    self.tile_hdim = int(
      math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of
    )
    head_dim_v = head_dim_v if head_dim_v is not None else head_dim
    self.tile_hdimv = int(
      math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of
    )
    self.check_hdim_oob = head_dim != self.tile_hdim
    self.check_hdim_v_oob = head_dim_v != self.tile_hdimv

    self.qhead_per_kvhead = qhead_per_kvhead
    self.is_causal = is_causal
    self.tile_m = tile_m
    self.tile_n = tile_n
    self.use_pdl = False
    self.qk_acc_dtype = Float32
    self.buffer_align_bytes = 1024

    # ── SplitD parameters (dQ: d_chunk=256, num_d_passes=2, num_d_inner=2) ──
    self.d_chunk = 256
    self.num_d_passes = self.tile_hdim // self.d_chunk  # = 2
    self.num_d_inner = self.tile_hdim // self.d_chunk  # = 2 (Phase A/C reduction along D)

    # Cooperative N-axis half size for Phase E (dQ += dS @ K_d_pass).
    # Each WG handles d_chunk//2 columns of the dQ output tile.
    self.dQ_n_half = self.d_chunk // 2  # = 128
    assert self.tile_hdim % self.d_chunk == 0
    assert self.tile_hdimv % self.d_chunk == 0
    assert self.d_chunk % 2 == 0, "cooperative ① requires d_chunk divisible by 2"

    # ── MMA WG configuration (dual asymmetric + cooperative on dQ) ──
    self.num_wg_mma_SdP = 1
    self.num_wg_mma_dQ = 1
    self.num_wg_mma = 1  # legacy alias for _get_tiled_mma / _setup_attributes
    # num_threads = 128 producer WG (warp 0 worker + warp 1..3 idle) + 128 WG1 + 128 WG2 = 384
    self.num_threads = 384
    self.num_threads_per_warp_group = 128
    self.num_producer_threads = 32  # actual producer worker threads (warp 0)
    self.num_mma_threads = 256  # WG1 + WG2

    self.num_mma_regs_wg1 = 216
    self.num_mma_regs_wg2 = 216
    self.num_producer_regs = 56

    # ── Pipeline stages (dQ) ──
    # pipeline_QdO (sA): 2 stages cycling Q0,Q1,dO0,dO1 per n_block (was per (d_pass, n_block)).
    self.A_stage = 2
    # pipeline_B (sB): 2 stages cycling K0,K1,V0,V1 per n_block.
    self.B_stage = 2
    self.Kt_stage = 2
    # sP_fp32 / sdS single-buffered. PFull/PEmpty (256) serializes sP_fp32;
    self.PdS_stage = 1

    # ── MMA layout configuration ──
    self.SdP_swapAB = False
    self.AtomLayoutMSdP = 1
    self.AtomLayoutMdQ = 1

  def _setup_attributes(self):
    # sA: (tile_m, d_chunk) — shared Q/dO streaming buffer (Q+dO interleaved per n_block)
    self.sA_layout = sm90_utils.make_smem_layout(
      self.dtype,
      LayoutEnum.ROW_MAJOR,
      (self.tile_m, self.d_chunk),
      stage=self.A_stage,
      major_mode_size=self.d_chunk,
    )
    # sB: (tile_n, d_chunk) — holds K_chunk / V_chunk streaming per n_block
    self.sB_layout = sm90_utils.make_smem_layout(
      self.dtype,
      LayoutEnum.ROW_MAJOR,
      (self.tile_n, self.d_chunk),
      stage=self.B_stage,
    )
    # sKt is physically split into front and back halves (each tile_n × dQ_n_half).
    # WG1 reads sKt_front_layout[stage] for its half-WGMMA; WG2 reads sKt_back_layout[stage].
    self.sKt_half_layout = sm90_utils.make_smem_layout(
      self.dtype,
      LayoutEnum.ROW_MAJOR,
      (self.tile_n, self.dQ_n_half),
      stage=self.Kt_stage,
      major_mode_size=self.dQ_n_half,  # MN_MAJOR for dQ GEMM B operand
    )
    # sEpi half (per d_pass, per WG) — (tile_m, dQ_n_half).
    # Cooperative ① splits the dQ output along its N axis: WG1 writes dQ[:, 0:128] for each
    # d_pass, WG2 writes dQ[:, 128:256] for each d_pass. Each half slice is 16KB at bf16.
    self.sEpi_half_layout = cute.select(
      sm90_utils.make_smem_layout(
        self.dtype,
        LayoutEnum.ROW_MAJOR,
        (self.tile_m, self.dQ_n_half),
        stage=1,
        major_mode_size=self.dQ_n_half,
      ),
      mode=[0, 1],
    )
    # sdS: (tile_m, tile_n) — WG2 stores dS here; both WGs read it for cooperative Phase E.
    wg_n_SdP = self.num_wg_mma // self.AtomLayoutMSdP
    self.sdS_layout = sm90_utils.make_smem_layout(
      self.dtype,
      LayoutEnum.ROW_MAJOR,
      (self.tile_m, self.tile_n),
      stage=self.PdS_stage,
      major_mode_size=self.tile_n // wg_n_SdP,
    )
    # Bf16-roundtrip-free P channel: WG1 writes fp32 acc_S → sP_fp32;
    # WG2 reads fp32 from sP_fp32 (via copy_utils.load_s2r) to compute dS in fp32.
    self.sP_fp32_layout = sm90_utils.make_smem_layout(
      Float32,
      LayoutEnum.ROW_MAJOR,
      (self.tile_m, self.tile_n),
      stage=self.PdS_stage,
      major_mode_size=self.tile_n // wg_n_SdP,
    )

  def _get_tiled_mma(self):
    """Create tiled MMA objects for SdP and dQ GEMMs (dQ cooperative ①)."""
    # ── SdP: S = Q @ K^T (WG1), dP = dO @ V^T (WG2) ──
    # shape_mnk: (tile_m, tile_n, d_chunk) = (64, 64, 256). Both WGs independently
    atom_layout_SdP = (
      self.AtomLayoutMSdP, self.num_wg_mma // self.AtomLayoutMSdP, 1
    )
    tiler_mn_SdP = (
      self.tile_m // atom_layout_SdP[0], self.tile_n // atom_layout_SdP[1]
    )
    tiled_mma_SdP = sm90_utils_basic.make_trivial_tiled_mma(
      self.dtype,
      self.dtype,
      OperandMajorMode.K,
      OperandMajorMode.K,
      self.qk_acc_dtype,
      atom_layout_mnk=atom_layout_SdP,
      tiler_mn=tiler_mn_SdP,
    )

    # ── dQ_half: dQ_half = dS @ K_d_pass_half (cooperative ①) ──
    # shape_mnk: (tile_m, dQ_n_half, tile_n) = (64, 128, 64)
    atom_layout_dQ = (
      self.AtomLayoutMdQ, self.num_wg_mma // self.AtomLayoutMdQ, 1
    )
    tiler_mn_dQ = (
      self.tile_m // atom_layout_dQ[0], self.dQ_n_half // atom_layout_dQ[1]
    )
    tiled_mma_dQ = sm90_utils_basic.make_trivial_tiled_mma(
      self.dtype,
      self.dtype,
      OperandMajorMode.K,
      OperandMajorMode.MN,
      Float32,
      atom_layout_mnk=atom_layout_dQ,
      tiler_mn=tiler_mn_dQ,
    )
    return tiled_mma_SdP, tiled_mma_dQ

  def _get_shared_storage_cls(self):
    sA_struct = cute.struct.Align[
      cute.struct.MemRange[self.dtype, cute.cosize(self.sA_layout)],
      self.buffer_align_bytes,
    ]
    sB_struct = cute.struct.Align[
      cute.struct.MemRange[self.dtype, cute.cosize(self.sB_layout)],
      self.buffer_align_bytes,
    ]
    # sKt is physically split into front+back halves; each (tile_n, dQ_n_half)
    sKt_half_struct = cute.struct.Align[
      cute.struct.MemRange[self.dtype,
                           cute.cosize(self.sKt_half_layout)],
      self.buffer_align_bytes,
    ]
    # sEpi half (single-stage per WG-side, used only during epilogue).
    sEpi_half_struct = cute.struct.Align[
      cute.struct.MemRange[self.dtype,
                           cute.cosize(self.sEpi_half_layout)],
      self.buffer_align_bytes,
    ]

    # dQ: union per-side sKt with sEpi.
    @cute.union
    class SmemKtEpi_front_t:
      sKt_front: sKt_half_struct  # mainloop: K_d_pass front half (cols 0..127)
      sEpi_front: sEpi_half_struct  # epilogue: dQ front-half R2S → TMA store

    @cute.union
    class SmemKtEpi_back_t:
      sKt_back: sKt_half_struct  # mainloop: K_d_pass back half (cols 128..255)
      sEpi_back: sEpi_half_struct  # epilogue: dQ back-half R2S → TMA store

    @cute.struct
    class SharedStorageDQ:
      mbar_ptr_A: cute.struct.MemRange[cutlass.Int64, self.A_stage * 2]
      mbar_ptr_B: cute.struct.MemRange[cutlass.Int64, self.B_stage * 2]
      mbar_ptr_Kt: cute.struct.MemRange[cutlass.Int64, self.Kt_stage * 2]
      sLSE: cute.struct.MemRange[
        Float32,
        cute.round_up(self.tile_m, 64) * self.A_stage,
      ]
      sdPsum: cute.struct.MemRange[
        Float32,
        cute.round_up(self.tile_m, 64) * self.A_stage,
      ]
      sA: sA_struct  # Q+dO streaming
      sB: sB_struct  # K+V streaming
      sKtEpi_front: SmemKtEpi_front_t  # union(sKt_front, sEpi_front)
      sKtEpi_back: SmemKtEpi_back_t  # union(sKt_back, sEpi_back)
      sP_fp32: cute.struct.Align[
        cute.struct.MemRange[Float32, cute.cosize(self.sP_fp32_layout)],
        1024,
      ]
      sdS: cute.struct.Align[
        cute.struct.MemRange[self.dtype,
                             cute.cosize(self.sdS_layout)],
        1024,
      ]

    return SharedStorageDQ

  @cute.jit
  def __call__(
    self,
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mdO: cute.Tensor,
    mLSE: cute.Tensor,
    mdPsum: cute.Tensor,
    mdQ: cute.Tensor,
    softmax_scale: Float32,
    mCuSeqlensQ: Optional[cute.Tensor] = None,
    mCuSeqlensK: Optional[cute.Tensor] = None,
    stream: cuda.CUstream = None,
  ):
    mQ, mK, mV, mdO, mdQ = [
      assume_tensor_aligned(t) for t in (mQ, mK, mV, mdO, mdQ)
    ]
    mLSE, mdPsum = [assume_tensor_aligned(t) for t in (mLSE, mdPsum)]

    # Transpose: (b, s, n, h) -> (s, h, n, b)
    def _qkv_transpose(t):
      return layout_utils.select(
        t, [1, 3, 2, 0] if cute.rank(t.shape) == 4 else [0, 2, 1]
      )

    mQ, mK, mV, mdO, mdQ = [_qkv_transpose(t) for t in (mQ, mK, mV, mdO, mdQ)]
    # Stats: (b, n, s) -> (s, n, b)
    LSE_transpose = [2, 1, 0] if cute.rank(mLSE.shape) == 3 else [1, 0]
    mLSE = layout_utils.select(mLSE, LSE_transpose)
    mdPsum = layout_utils.select(mdPsum, LSE_transpose)

    tiled_mma_SdP, tiled_mma_dQ = self._get_tiled_mma()
    # num_mma_threads = 256 (WG1+WG2) is set in __init__; do NOT overwrite from
    # tiled_mma_SdP.size (= 128 per WG with num_wg_mma=1).
    self.num_mma_threads_per_wg = int(tiled_mma_SdP.size)  # = 128
    self._setup_attributes()
    SharedStorage = self._get_shared_storage_cls()

    sA_layout_sel = cute.select(self.sA_layout, mode=[0, 1])
    sB_layout_sel = cute.select(self.sB_layout, mode=[0, 1])
    sKt_half_layout_sel = cute.select(self.sKt_half_layout, mode=[0, 1])
    sEpi_half_layout_sel = self.sEpi_half_layout
    # tx_count for Kt mbar covers BOTH halves (front+back) in one acquire.
    # Producer issues 2 TMA per d_pass (one per half); both finish before commit.
    kt_half_bytes = cute.size_in_bytes(mK.element_type, sKt_half_layout_sel)
    self.tma_copy_bytes = {
      "A": cute.size_in_bytes(mQ.element_type,
                              sA_layout_sel),  # Q or dO chunk: 32KB
      "B": cute.size_in_bytes(mK.element_type,
                              sB_layout_sel),  # K or V chunk:  32KB
      "Kt_half": kt_half_bytes,  # one Kt half:   16KB
      "Kt": 2 * kt_half_bytes,  # both halves:   32KB
    }
    self.tma_copy_bytes["LSE"] = self.tile_m * Float32.width // 8
    self.tma_copy_bytes["dPsum"] = self.tile_m * Float32.width // 8

    gmem_tiled_copy_g2s = cpasync.CopyBulkTensorTileG2SOp()
    # Q: (tile_m, d_chunk) = (64, 256) → sA
    tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
      gmem_tiled_copy_g2s,
      mQ,
      sA_layout_sel,
      (self.tile_m, self.d_chunk),
    )
    # dO: (tile_m, d_chunk) → sA
    tma_atom_dO, tma_tensor_dO = cpasync.make_tiled_tma_atom(
      gmem_tiled_copy_g2s,
      mdO,
      sA_layout_sel,
      (self.tile_m, self.d_chunk),
    )
    # K: (tile_n, d_chunk) K_MAJOR → sB
    tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
      gmem_tiled_copy_g2s,
      mK,
      sB_layout_sel,
      (self.tile_n, self.d_chunk),
    )
    # V: (tile_n, d_chunk) → sB
    tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
      gmem_tiled_copy_g2s,
      mV,
      sB_layout_sel,
      (self.tile_n, self.d_chunk),
    )
    # Kt half — (tile_n, dQ_n_half) MN_MAJOR. Two TMA per d_pass:
    # one with K offset (n_block, d_pass*d_chunk + 0) → sKt_front,
    # one with K offset (n_block, d_pass*d_chunk + dQ_n_half) → sKt_back.
    tma_atom_Kt_half, tma_tensor_Kt_half = cpasync.make_tiled_tma_atom(
      gmem_tiled_copy_g2s,
      mK,
      sKt_half_layout_sel,
      (self.tile_n, self.dQ_n_half),
    )
    # dQ half store atom — (tile_m, dQ_n_half) cooperative ①.
    # Each WG TMA-stores its own half slice independently to gmem.
    self.varlen_q = mCuSeqlensQ is not None
    self.is_varlen_k = mCuSeqlensK is not None
    gmem_tiled_copy_s2g = cpasync.CopyBulkTensorTileS2GOp()
    mdQ_tma = copy_utils.create_ragged_tensor_for_tma(
      mdQ, ragged_dim=0, ptr_shift=True
    ) if self.varlen_q else mdQ
    tma_atom_dQ_half, tma_tensor_dQ_half = cpasync.make_tiled_tma_atom(
      gmem_tiled_copy_s2g,
      mdQ_tma,
      sEpi_half_layout_sel,
      (self.tile_m, self.dQ_n_half),
    )

    if const_expr(mCuSeqlensQ is not None):
      TileScheduler = SingleTileVarlenScheduler
    else:
      TileScheduler = SingleTileScheduler
    num_m_blocks = cute.ceil_div(cute.size(mQ.shape[0]), self.tile_m)
    num_batch = cute.size(mQ.shape[3]) if cute.rank(
      mQ.shape
    ) == 4 else cute.size(mCuSeqlensQ.shape[0] - 1)
    tile_sched_args = TileSchedulerArguments(
      num_m_blocks,
      cute.size(mQ.shape[2]),  # num_heads
      num_batch,
      1,  # cluster_size
      cute.size(mK.shape[0]),  # seqlen_k for n_block range
      mQ.shape[1],  # head_dim_qk
      mV.shape[1],  # head_dim_v
      total_q=cute.size(mQ.shape[0]),
      tile_shape_mn=(self.tile_m, self.tile_n),
      mCuSeqlensQ=mCuSeqlensQ,
      qhead_per_kvhead_packgqa=1,
      element_size=self.dtype.width // 8,
      lpt=self.is_causal,
    )
    tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
    grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

    softmax_scale_log2 = softmax_scale * math.log2(math.e)

    # GQA: FastDivmodDivisor for head_idx → head_idx_kv mapping
    qhead_per_kvhead_divmod = None
    if const_expr(self.qhead_per_kvhead > 1):
      qhead_per_kvhead_divmod = FastDivmodDivisor(self.qhead_per_kvhead)

    self.kernel(
      tma_tensor_Q,
      tma_tensor_K,
      tma_tensor_V,
      tma_tensor_dO,
      tma_tensor_Kt_half,  # half-N TMA tensor for Kt
      tma_tensor_dQ_half,  # half-N TMA tensor for dQ store
      tma_atom_Q,
      tma_atom_dO,
      tma_atom_K,
      tma_atom_V,
      tma_atom_Kt_half,  # half-N TMA atom for Kt
      tma_atom_dQ_half,  # half-N TMA atom for dQ store
      mLSE,
      mdPsum,
      mCuSeqlensQ,
      mCuSeqlensK,
      softmax_scale_log2,
      softmax_scale,
      self.sA_layout,
      self.sB_layout,
      self.sKt_half_layout,
      self.sEpi_half_layout,
      self.sdS_layout,
      self.sP_fp32_layout,
      tiled_mma_SdP,
      tiled_mma_dQ,  # half-N tiled_mma
      tile_sched_params,
      TileScheduler,
      SharedStorage,
      qhead_per_kvhead_divmod,
    ).launch(
      grid=grid_dim,
      block=[self.num_threads, 1, 1],
      stream=stream,
      min_blocks_per_mp=1,
      use_pdl=self.use_pdl,
    )

  @cute.kernel
  def kernel(
    self,
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mdO: cute.Tensor,
    mKt_half: cute.Tensor,  # half-N (tile_n, dQ_n_half) TMA view of K
    mdQ_half: cute.Tensor,  # half-N (tile_m, dQ_n_half) TMA view of dQ
    tma_atom_Q: cute.CopyAtom,
    tma_atom_dO: cute.CopyAtom,
    tma_atom_K: cute.CopyAtom,
    tma_atom_V: cute.CopyAtom,
    tma_atom_Kt_half: cute.CopyAtom,
    tma_atom_dQ_half: cute.CopyAtom,
    mLSE: cute.Tensor,
    mdPsum: cute.Tensor,
    mCuSeqlensQ: Optional[cute.Tensor],
    mCuSeqlensK: Optional[cute.Tensor],
    softmax_scale_log2: Float32,
    softmax_scale: Float32,
    sA_layout: cute.ComposedLayout,
    sB_layout: cute.ComposedLayout,
    sKt_half_layout: cute.ComposedLayout,
    sEpi_half_layout: cute.
    ComposedLayout,  # fix: kernel-region SSA def for sEpi half tensor
    sdS_layout: cute.ComposedLayout,
    sP_fp32_layout: cute.ComposedLayout,
    tiled_mma_SdP: cute.TiledMma,
    tiled_mma_dQ: cute.TiledMma,  # half-N tiled_mma
    tile_sched_params: ParamsBase,
    TileScheduler: cutlass.Constexpr[Callable],
    SharedStorage: cutlass.Constexpr[Callable],
    qhead_per_kvhead_divmod: Optional[FastDivmodDivisor] = None,
  ):
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    if warp_idx == 0:
      for atom in [
        tma_atom_Q, tma_atom_dO, tma_atom_K, tma_atom_V, tma_atom_Kt_half,
        tma_atom_dQ_half
      ]:
        cpasync.prefetch_descriptor(atom)

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)

    pipeline_producer_group = cutlass.pipeline.CooperativeGroup(
      cutlass.pipeline.Agent.Thread
    )
    # pipeline_QdO / pipeline_B / pipeline_Kt: 8-warp shared consumer (WG1+WG2).
    pipeline_consumer_8warp = cutlass.pipeline.CooperativeGroup(
      cutlass.pipeline.Agent.Thread,
      self.num_mma_threads // cute.arch.WARP_SIZE,  # = 8 warps (WG1+WG2)
    )

    pipeline_QdO = pipeline.PipelineTmaAsync.create(
      barrier_storage=storage.mbar_ptr_A.data_ptr(),
      num_stages=self.A_stage,
      producer_group=pipeline_producer_group,
      consumer_group=pipeline_consumer_8warp,
      tx_count=self.tma_copy_bytes["A"],
      defer_sync=True,
    )
    pipeline_B = pipeline.PipelineTmaAsync.create(
      barrier_storage=storage.mbar_ptr_B.data_ptr(),
      num_stages=self.B_stage,
      producer_group=pipeline_producer_group,
      consumer_group=pipeline_consumer_8warp,
      tx_count=self.tma_copy_bytes["B"],
      defer_sync=True,
    )
    # pipeline_Kt is 8-warp shared (both WGs cooperative-consume the two
    # halves of each Kt push)
    pipeline_Kt = pipeline.PipelineTmaAsync.create(
      barrier_storage=storage.mbar_ptr_Kt.data_ptr(),
      num_stages=self.Kt_stage,
      producer_group=pipeline_producer_group,
      consumer_group=pipeline_consumer_8warp,
      tx_count=self.tma_copy_bytes["Kt"],  # = 2 × half = full Kt bytes
      defer_sync=True,
    )

    # CTA-level init-sync  make mbarrier_init visible
    # to async-proxy before any consumer can reach try_wait.
    pipeline_init_arrive(cluster_shape_mn=(1, 1), is_relaxed=True)
    pipeline_init_wait(cluster_shape_mn=(1, 1))

    # sKt / sEpi share physical SMEM via union per side.
    # Mainloop uses sKt_{front,back} views; epilogue uses sEpi_{front,back} views.
    sA = storage.sA.get_tensor(sA_layout.outer, swizzle=sA_layout.inner)
    sB = storage.sB.get_tensor(sB_layout.outer, swizzle=sB_layout.inner)
    sKt_front = storage.sKtEpi_front.sKt_front.get_tensor(
      sKt_half_layout.outer,
      swizzle=sKt_half_layout.inner,
    )
    sKt_back = storage.sKtEpi_back.sKt_back.get_tensor(
      sKt_half_layout.outer,
      swizzle=sKt_half_layout.inner,
    )
    sdS = storage.sdS.get_tensor(sdS_layout.outer, swizzle=sdS_layout.inner)
    sP_fp32 = storage.sP_fp32.get_tensor(
      sP_fp32_layout.outer, swizzle=sP_fp32_layout.inner
    )

    # Single-stage SMEM views for half epilogue TMA stores (alias sKt halves' physical slots).
    sEpi_front = storage.sKtEpi_front.sEpi_front.get_tensor(
      sEpi_half_layout.outer,
      swizzle=sEpi_half_layout.inner,
    )
    sEpi_back = storage.sKtEpi_back.sEpi_back.get_tensor(
      sEpi_half_layout.outer,
      swizzle=sEpi_half_layout.inner,
    )
    sLSE = storage.sLSE.get_tensor(
      cute.make_layout(
        (self.tile_m, self.A_stage),
        stride=(1, cute.round_up(self.tile_m, 64)),
      )
    )
    sdPsum = storage.sdPsum.get_tensor(
      cute.make_layout(
        (self.tile_m, self.A_stage),
        stride=(1, cute.round_up(self.tile_m, 64)),
      )
    )

    block_info = BlockInfo(
      self.tile_m,
      self.tile_n,
      self.is_causal,
      False,  # is_local
    )
    SeqlenInfoCls = partial(
      SeqlenInfoQK.create,
      seqlen_q_static=mQ.shape[0],
      seqlen_k_static=mK.shape[0],
      mCuSeqlensQ=mCuSeqlensQ,
      mCuSeqlensK=mCuSeqlensK,
      tile_m=self.tile_m,
      tile_n=self.tile_n,
    )
    TileSchedulerCls = partial(TileScheduler.create, tile_sched_params)

    # three-role dispatch — producer WG (warp 0..3) / WG1 (warp 4..7) / WG2 (warp 8..11)
    # WG1 now owns acc_dQ_pass[d]_front (cooperative ① N-front halves);
    # WG2 owns acc_dQ_pass[d]_back. Both participate in cooperative Phase E.
    tidx, _, _ = cute.arch.thread_idx()
    warp_group_idx = cute.arch.make_warp_uniform(
      tidx // self.num_threads_per_warp_group
    )

    if warp_group_idx == 0:
      cute.arch.setmaxregister_decrease(self.num_producer_regs)
      if warp_idx == 0:
        self.load(
          mQ,
          mK,
          mV,
          mdO,
          mKt_half,
          mLSE,
          mdPsum,
          sA,
          sB,
          sKt_front,
          sKt_back,
          sLSE,
          sdPsum,
          tma_atom_Q,
          tma_atom_dO,
          tma_atom_K,
          tma_atom_V,
          tma_atom_Kt_half,
          pipeline_QdO,
          pipeline_B,
          pipeline_Kt,
          block_info,
          SeqlenInfoCls,
          TileSchedulerCls,
          qhead_per_kvhead_divmod,
        )
    elif warp_group_idx == 1:
      # WG1: Phase A (S) + Phase B (softmax → sP_fp32) + cooperative Phase E_front (dQ front halves).
      cute.arch.setmaxregister_increase(self.num_mma_regs_wg1)
      tidx_in_wg = tidx - self.num_threads_per_warp_group
      self.mma_wg1(
        tiled_mma_SdP,
        tiled_mma_dQ,
        mdQ_half,
        sA,
        sB,
        sKt_front,
        sP_fp32,
        sdS,
        sLSE,
        sEpi_front,
        pipeline_QdO,
        pipeline_B,
        pipeline_Kt,
        tidx_in_wg,
        tma_atom_dQ_half,
        softmax_scale_log2,
        softmax_scale,
        block_info,
        SeqlenInfoCls,
        TileSchedulerCls,
        qhead_per_kvhead_divmod,
      )
    else:
      # WG2: Phase C (dP) + Phase D (dS → sdS) + cooperative Phase E_back (dQ back halves) + epilogue.
      cute.arch.setmaxregister_increase(self.num_mma_regs_wg2)
      tidx_in_wg = tidx - 2 * self.num_threads_per_warp_group
      self.mma_wg2(
        tiled_mma_SdP,
        tiled_mma_dQ,
        mdQ_half,
        sA,
        sB,
        sKt_back,
        sP_fp32,
        sdS,
        sdPsum,
        sEpi_back,
        pipeline_QdO,
        pipeline_B,
        pipeline_Kt,
        tidx_in_wg,
        tma_atom_dQ_half,
        softmax_scale_log2,
        softmax_scale,
        block_info,
        SeqlenInfoCls,
        TileSchedulerCls,
        qhead_per_kvhead_divmod,
      )

  @cute.jit
  def load(
    self,
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mdO: cute.Tensor,
    mKt_half: cute.Tensor,  # half-N TMA tensor for Kt
    mLSE: cute.Tensor,
    mdPsum: cute.Tensor,
    sA: cute.Tensor,
    sB: cute.Tensor,
    sKt_front: cute.Tensor,
    sKt_back: cute.Tensor,
    sLSE: cute.Tensor,
    sdPsum: cute.Tensor,
    tma_atom_Q: cute.CopyAtom,
    tma_atom_dO: cute.CopyAtom,
    tma_atom_K: cute.CopyAtom,
    tma_atom_V: cute.CopyAtom,
    tma_atom_Kt_half: cute.CopyAtom,
    pipeline_QdO: pipeline.PipelineAsync,
    pipeline_B: pipeline.PipelineAsync,
    pipeline_Kt: pipeline.PipelineAsync,
    block_info: BlockInfo,
    SeqlenInfoCls: Callable,
    TileSchedulerCls: Callable,
    qhead_per_kvhead_divmod: Optional[FastDivmodDivisor] = None,
  ):
    producer_state_QdO = cutlass.pipeline.make_pipeline_state(
      cutlass.pipeline.PipelineUserType.Producer, self.A_stage
    )
    producer_state_B = cutlass.pipeline.make_pipeline_state(
      cutlass.pipeline.PipelineUserType.Producer, self.B_stage
    )
    producer_state_Kt = cutlass.pipeline.make_pipeline_state(
      cutlass.pipeline.PipelineUserType.Producer, self.Kt_stage
    )

    tile_scheduler = TileSchedulerCls()
    work_tile = tile_scheduler.initial_work_tile_info()
    did_produce_A = Boolean(False)
    did_produce_B = Boolean(False)
    did_produce_Kt = Boolean(False)

    while work_tile.is_valid_tile:
      m_block, head_idx, batch_idx, _ = work_tile.tile_idx
      seqlen = SeqlenInfoCls(batch_idx)

      # GQA: map Q head index to KV head index for K/V loading
      head_idx_kv = head_idx if const_expr(
        self.qhead_per_kvhead == 1
      ) else head_idx // qhead_per_kvhead_divmod

      mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
      mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[None, None,
                                                           head_idx_kv]
      mV_cur = seqlen.offset_batch_K(mV, batch_idx, dim=3)[None, None,
                                                           head_idx_kv]
      mdO_cur = seqlen.offset_batch_Q(mdO, batch_idx, dim=3)[None, None,
                                                             head_idx]
      mKt_half_cur = seqlen.offset_batch_K(mKt_half, batch_idx,
                                           dim=3)[None, None, head_idx_kv]
      mLSE_cur = seqlen.offset_batch_Q(mLSE, batch_idx, dim=2,
                                       padded=True)[None, head_idx]
      mdPsum_cur = seqlen.offset_batch_Q(mdPsum, batch_idx, dim=2,
                                         padded=True)[None, head_idx]

      gLSE = cute.local_tile(mLSE_cur, (self.tile_m, ), (None, ))
      gdPsum = cute.local_tile(mdPsum_cur, (self.tile_m, ), (None, ))
      if const_expr(self.use_pdl):
        cute.arch.griddepcontrol_wait()
      load_LSE = copy_utils.cpasync_bulk_get_copy_fn(gLSE, sLSE)
      load_dPsum = copy_utils.cpasync_bulk_get_copy_fn(gdPsum, sdPsum)

      n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)

      # ═══ loop nest reversed — n_block outer, d_pass inner ═══
      #
      # Per n_block: push Q+K, dO+V (interleaved pairs to avoid common_err_kernel.md #7
      # producer-vs-consumer circular wait), then for each d_pass push Kt_front + Kt_back
      # gated by a single pipeline_Kt mbar (tx_count covers BOTH halves).
      for i_n in cutlass.range(n_block_max - n_block_min, unroll=1):
        n_block = n_block_max - 1 - i_n  # inverse order for causal

        # ── Push (Q_{d_inner}, K_{d_inner}) pairs for WG1 Phase A ──
        # LSE piggybacks on last Q chunk (extra_tx_count on the corresponding acquire).
        for d_inner in cutlass.range_constexpr(self.num_d_inner):
          gQ_d = cute.local_tile(
            mQ_cur, (self.tile_m, self.d_chunk), (None, d_inner)
          )
          load_Q_d, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_Q,
            0,
            cute.make_layout(1),
            gQ_d,
            sA,
          )
          if cutlass.const_expr(d_inner == self.num_d_inner - 1):
            pipeline_QdO.producer_acquire(
              producer_state_QdO,
              extra_tx_count=self.tma_copy_bytes["LSE"],
            )
          else:
            pipeline_QdO.producer_acquire(producer_state_QdO)
          load_Q_d(
            src_idx=m_block,
            dst_idx=producer_state_QdO.index,
            tma_bar_ptr=pipeline_QdO.producer_get_barrier(producer_state_QdO),
          )
          if cutlass.const_expr(d_inner == self.num_d_inner - 1):
            load_LSE(
              src_idx=m_block,
              dst_idx=producer_state_QdO.index,
              tma_bar_ptr=pipeline_QdO.producer_get_barrier(producer_state_QdO),
            )
          pipeline_QdO.producer_commit(producer_state_QdO)
          did_produce_A = Boolean(True)
          producer_state_QdO.advance()

          gK_d = cute.local_tile(
            mK_cur, (self.tile_n, self.d_chunk), (None, d_inner)
          )
          load_K_d, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_K,
            0,
            cute.make_layout(1),
            gK_d,
            sB,
          )
          pipeline_B.producer_acquire(producer_state_B)
          load_K_d(
            src_idx=n_block,
            dst_idx=producer_state_B.index,
            tma_bar_ptr=pipeline_B.producer_get_barrier(producer_state_B),
          )
          pipeline_B.producer_commit(producer_state_B)
          did_produce_B = Boolean(True)
          producer_state_B.advance()

        # ── Push (dO_{d_inner}, V_{d_inner}) pairs for WG2 Phase C ──
        # dPsum piggybacks on last dO chunk.
        for d_inner in cutlass.range_constexpr(self.num_d_inner):
          gdO_d = cute.local_tile(
            mdO_cur, (self.tile_m, self.d_chunk), (None, d_inner)
          )
          load_dO_d, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_dO,
            0,
            cute.make_layout(1),
            gdO_d,
            sA,
          )
          if cutlass.const_expr(d_inner == self.num_d_inner - 1):
            pipeline_QdO.producer_acquire(
              producer_state_QdO,
              extra_tx_count=self.tma_copy_bytes["dPsum"],
            )
          else:
            pipeline_QdO.producer_acquire(producer_state_QdO)
          load_dO_d(
            src_idx=m_block,
            dst_idx=producer_state_QdO.index,
            tma_bar_ptr=pipeline_QdO.producer_get_barrier(producer_state_QdO),
          )
          if cutlass.const_expr(d_inner == self.num_d_inner - 1):
            load_dPsum(
              src_idx=m_block,
              dst_idx=producer_state_QdO.index,
              tma_bar_ptr=pipeline_QdO.producer_get_barrier(producer_state_QdO),
            )
          pipeline_QdO.producer_commit(producer_state_QdO)
          did_produce_A = Boolean(True)
          producer_state_QdO.advance()

          gV_d = cute.local_tile(
            mV_cur, (self.tile_n, self.d_chunk), (None, d_inner)
          )
          load_V_d, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_V,
            0,
            cute.make_layout(1),
            gV_d,
            sB,
          )
          pipeline_B.producer_acquire(producer_state_B)
          load_V_d(
            src_idx=n_block,
            dst_idx=producer_state_B.index,
            tma_bar_ptr=pipeline_B.producer_get_barrier(producer_state_B),
          )
          pipeline_B.producer_commit(producer_state_B)
          did_produce_B = Boolean(True)
          producer_state_B.advance()

        # ── Inner d_pass loop: push Kt_front + Kt_back per d_pass ──
        # mKt_half has been TMA-built with tile (tile_n, dQ_n_half); tiling indexed
        # by (n_block, half_idx) where half_idx = 2*d_pass + {0=front, 1=back}.
        # Single mbar gates both halves (tx_count = full Kt bytes). 8-warp shared
        # consumer group: WG1 reads sKt_front[stage] for half-WGMMA, WG2 reads sKt_back.
        for d_pass in cutlass.range_constexpr(self.num_d_passes):
          half_idx_front = 2 * d_pass + 0
          half_idx_back = 2 * d_pass + 1
          gKt_front = cute.local_tile(
            mKt_half_cur,
            (self.tile_n, self.dQ_n_half),
            (None, half_idx_front),
          )
          gKt_back = cute.local_tile(
            mKt_half_cur,
            (self.tile_n, self.dQ_n_half),
            (None, half_idx_back),
          )
          load_Kt_front, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_Kt_half,
            0,
            cute.make_layout(1),
            gKt_front,
            sKt_front,
          )
          load_Kt_back, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_Kt_half,
            0,
            cute.make_layout(1),
            gKt_back,
            sKt_back,
          )
          pipeline_Kt.producer_acquire(producer_state_Kt)
          load_Kt_front(
            src_idx=n_block,
            dst_idx=producer_state_Kt.index,
            tma_bar_ptr=pipeline_Kt.producer_get_barrier(producer_state_Kt),
          )
          load_Kt_back(
            src_idx=n_block,
            dst_idx=producer_state_Kt.index,
            tma_bar_ptr=pipeline_Kt.producer_get_barrier(producer_state_Kt),
          )
          pipeline_Kt.producer_commit(producer_state_Kt)
          did_produce_Kt = Boolean(True)
          producer_state_Kt.advance()

      # Per-work_tile WSWG1 sync (288 = 32 producer + 256 MMA).
      # Moved out of d_pass loop — epilogue is now per work_tile, so we only need
      # to gate producer→consumer once per work_tile (between mainloop done and
      # next work_tile's TMA bursts; sKt↔sEpi union flip is also per work_tile now).
      cute.arch.barrier(
        barrier_id=int(NamedBarrierBwd.WarpSchedulerWG1),
        number_of_threads=self.num_producer_threads + self.num_mma_threads,
      )

      tile_scheduler.advance_to_next_work()
      work_tile = tile_scheduler.get_current_work()

    if did_produce_A:
      pipeline_QdO.producer_tail(producer_state_QdO)
    if did_produce_B:
      pipeline_B.producer_tail(producer_state_B)
    if did_produce_Kt:
      pipeline_Kt.producer_tail(producer_state_Kt)

  # ════════════════════════════════════════════════════════════════════
  # dQ: WG1 = Phase A (S=ΣQ@K^T) + Phase B (softmax → sP_fp32)
  #          + cooperative Phase E_front (acc_dQ_pass[d]_front += sdS @ sKt_front).
  # acc_dQ_pass[0..num_d_passes-1]_front persist across all n_blocks of the work_tile;
  # per-work_tile epilogue writes them as dQ[:, d_pass*d_chunk+0 : d_pass*d_chunk+dQ_n_half].
  # ════════════════════════════════════════════════════════════════════
  @cute.jit
  def mma_wg1(
    self,
    tiled_mma_SdP: cute.TiledMma,
    tiled_mma_dQ: cute.TiledMma,  # half-N tiled_mma for cooperative Phase E
    mdQ_half: cute.Tensor,  # half-N TMA view of dQ output
    sA: cute.Tensor,
    sB: cute.Tensor,
    sKt_front: cute.Tensor,  # WG1's half of Kt
    sP_fp32: cute.Tensor,  # fp32 P buffer (WG1 writes; WG2 reads in P4)
    sdS: cute.Tensor,  # WG1 reads sdS for Phase E_front
    sLSE: cute.Tensor,
    sEpi_front: cute.Tensor,  # WG1 epilogue staging (front half)
    pipeline_QdO: pipeline.PipelineAsync,
    pipeline_B: pipeline.PipelineAsync,
    pipeline_Kt: pipeline.PipelineAsync,
    tidx: Int32,  # WG-local ∈ [0, 128)
    tma_atom_dQ_half: cute.CopyAtom,
    softmax_scale_log2: Float32,
    softmax_scale: Float32,
    block_info: BlockInfo,
    SeqlenInfoCls: Callable,
    TileSchedulerCls: Callable,
    qhead_per_kvhead_divmod: Optional[FastDivmodDivisor] = None,
  ):
    thr_mma_SdP = tiled_mma_SdP.get_slice(tidx)
    wg_mma_SdP = tiled_mma_SdP.get_slice(0)
    wg_mma_dQ = tiled_mma_dQ.get_slice(0)

    # ── SdP fragments (WG1 uses Q from sA, K from sB) ──
    shape_mnk_SdP = (self.tile_m, self.tile_n, self.d_chunk)
    _, tSrA, tSrB = sm90_utils.partition_fragment_ABC(
      wg_mma_SdP, shape_mnk_SdP, sA, sB, swap_AB=False
    )

    # ── dQ_half fragments (cooperative ①, WG1 front half) ──
    # A = sdS (K_MAJOR), B = sKt_front (MN_MAJOR after transpose). Shape (64, 128, 64).
    sKt_front_t = layout_utils.transpose_view(sKt_front)
    shape_mnk_dQ_half = (self.tile_m, self.dQ_n_half, self.tile_n)
    acc_dQ_pass_0_front = cute.make_rmem_tensor(
      tiled_mma_dQ.partition_shape_C(shape_mnk_dQ_half[:2]), Float32
    )
    acc_dQ_pass_1_front = cute.make_rmem_tensor(
      tiled_mma_dQ.partition_shape_C(shape_mnk_dQ_half[:2]), Float32
    )
    _, tDQrA, tDQrB_front = sm90_utils.partition_fragment_ABC(
      wg_mma_dQ, shape_mnk_dQ_half, sdS, sKt_front_t, swap_AB=False
    )

    # ── sP_fp32 R2S writer: fp32 acc_S → sP_fp32 in C-operand layout ──
    copy_P_fp32_r2s, _, _ = copy_utils.get_smem_store_C(
      tiled_mma_SdP,
      sP_fp32,
      tidx,
      transpose=False,
      position_independent=False,
    )

    # ── LSE per-thread row mapping ──
    tLSEsLSE = layout_utils.mma_partition_C_vec(
      sLSE, thr_mma_SdP, expand_shape=self.tile_n, is_colvec=True
    )

    consumer_state_QdO = cutlass.pipeline.make_pipeline_state(
      cutlass.pipeline.PipelineUserType.Consumer, self.A_stage
    )
    consumer_state_B = cutlass.pipeline.make_pipeline_state(
      cutlass.pipeline.PipelineUserType.Consumer, self.B_stage
    )
    consumer_state_Kt = cutlass.pipeline.make_pipeline_state(
      cutlass.pipeline.PipelineUserType.Consumer, self.Kt_stage
    )

    # startup credit (common_err_kernel.md #2):
    cute.arch.barrier_arrive(
      barrier_id=int(NamedBarrierBwd.dSEmpty),
      number_of_threads=self.num_mma_threads,
    )

    tile_scheduler = TileSchedulerCls()
    work_tile = tile_scheduler.initial_work_tile_info()

    while work_tile.is_valid_tile:
      m_block, head_idx, batch_idx, _ = work_tile.tile_idx
      seqlen = SeqlenInfoCls(batch_idx)
      n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)
      process_tile = const_expr(
        not self.is_varlen_k
      ) or n_block_min < n_block_max

      mask = AttentionMask(self.tile_m, self.tile_n, seqlen)

      # dQ_accumulate is single-bool for whole work_tile. First n_block uses
      # zero_init=True for BOTH d_pass front halves; subsequent n_blocks accumulate.
      dQ_accumulate = Boolean(False)

      for i_n in cutlass.range(n_block_max - n_block_min, unroll=1):
        n_block = n_block_max - 1 - i_n
        mask_fn = partial(
          mask.apply_mask,
          batch_idx=batch_idx,
          head_idx=head_idx,
          m_block=m_block,
          n_block=n_block,
          thr_mma=thr_mma_SdP,
          mask_seqlen=True,
          mask_causal=self.is_causal,
        )
        consumer_state_QdO, consumer_state_B, consumer_state_Kt = self._mma_wg1_one_n_block(
          consumer_state_QdO,
          consumer_state_B,
          consumer_state_Kt,
          tiled_mma_SdP,
          tiled_mma_dQ,
          tSrA,
          tSrB,
          tDQrA,
          tDQrB_front,
          shape_mnk_SdP,
          acc_dQ_pass_0_front,
          acc_dQ_pass_1_front,
          copy_P_fp32_r2s,
          pipeline_QdO,
          pipeline_B,
          pipeline_Kt,
          tLSEsLSE,
          softmax_scale_log2,
          n_block,
          tidx,
          mask_fn,
          dQ_accumulate=dQ_accumulate,
        )
        dQ_accumulate = Boolean(True)

      # ── Per work_tile epilogue: write both d_pass front halves ──
      if process_tile:
        self.epilogue_dQ_half_slice(
          acc_dQ_pass_0_front,
          acc_dQ_pass_1_front,
          mdQ_half,
          sEpi_front,
          seqlen,
          tma_atom_dQ_half,
          tiled_mma_dQ,
          tidx,
          m_block,
          head_idx,
          batch_idx,
          is_front_wg=True,
        )
      else:
        # Even with no n_blocks: keep WG1↔WG2 lockstep on Epilogue handshake
        # (each WG calls 2 barriers per work_tile in epilogue path).
        cute.arch.barrier(
          barrier_id=int(NamedBarrierBwd.Epilogue),
          number_of_threads=self.num_mma_threads,
        )
        cute.arch.barrier(
          barrier_id=int(NamedBarrierBwd.Epilogue),
          number_of_threads=self.num_mma_threads,
        )

      # Per-work_tile WSWG1 sync (288): producer waits before next work_tile.
      cute.arch.barrier(
        barrier_id=int(NamedBarrierBwd.WarpSchedulerWG1),
        number_of_threads=self.num_producer_threads + self.num_mma_threads,
      )

      tile_scheduler.advance_to_next_work()
      work_tile = tile_scheduler.get_current_work()

  @cute.jit
  def _mma_wg1_one_n_block(
    self,
    consumer_state_QdO: cutlass.pipeline.PipelineState,
    consumer_state_B: cutlass.pipeline.PipelineState,
    consumer_state_Kt: cutlass.pipeline.PipelineState,
    tiled_mma_SdP: cute.TiledMma,
    tiled_mma_dQ: cute.TiledMma,
    tSrA: cute.Tensor,
    tSrB: cute.Tensor,
    tDQrA: cute.Tensor,
    tDQrB_front: cute.Tensor,
    shape_mnk_SdP: cute.Shape,
    acc_dQ_pass_0_front: cute.Tensor,
    acc_dQ_pass_1_front: cute.Tensor,
    copy_P_fp32_r2s: Callable,
    pipeline_QdO: pipeline.PipelineAsync,
    pipeline_B: pipeline.PipelineAsync,
    pipeline_Kt: pipeline.PipelineAsync,
    tLSEsLSE: cute.Tensor,
    softmax_scale_log2: Float32,
    n_block: Int32,
    tidx: Int32,
    mask_fn: Callable,
    dQ_accumulate: Boolean = True,
  ):
    p_stage = Int32(0)  # sP_fp32 / sdS single-buffered

    acc_S = cute.make_rmem_tensor(
      tiled_mma_SdP.partition_shape_C(shape_mnk_SdP[:2]), Float32
    )
    # ═══ Phase A: S = ΣQ_d @ K_d^T (num_d_inner WGMMA, real consume Q + K) ═══
    for d_inner in cutlass.range_constexpr(self.num_d_inner):
      pipeline_QdO.consumer_wait(
        consumer_state_QdO, pipeline_QdO.consumer_try_wait(consumer_state_QdO)
      )
      pipeline_B.consumer_wait(
        consumer_state_B, pipeline_B.consumer_try_wait(consumer_state_B)
      )
      gemm_w_idx(
        tiled_mma_SdP,
        acc_S,
        tSrA,
        tSrB,
        zero_init=(d_inner == 0),
        A_idx=consumer_state_QdO.index,
        B_idx=consumer_state_B.index,
        wg_wait=0,
      )
      if cutlass.const_expr(d_inner == self.num_d_inner - 1):
        tLSErLSE = copy_utils.load_s2r(tLSEsLSE[None, consumer_state_QdO.index])
      with cute.arch.elect_one():
        pipeline_QdO.consumer_release(consumer_state_QdO)
      with cute.arch.elect_one():
        pipeline_B.consumer_release(consumer_state_B)
      consumer_state_QdO.advance()
      consumer_state_B.advance()

    # ═══ Empty release: dO0..dO_{num_d_inner-1} (consumed by WG2 Phase C) ═══
    for _ in cutlass.range_constexpr(self.num_d_inner):
      pipeline_QdO.consumer_wait(
        consumer_state_QdO, pipeline_QdO.consumer_try_wait(consumer_state_QdO)
      )
      with cute.arch.elect_one():
        pipeline_QdO.consumer_release(consumer_state_QdO)
      consumer_state_QdO.advance()

    # ═══ Empty release: V0..V_{num_d_inner-1} (consumed by WG2 Phase C) ═══
    for _ in cutlass.range_constexpr(self.num_d_inner):
      pipeline_B.consumer_wait(
        consumer_state_B, pipeline_B.consumer_try_wait(consumer_state_B)
      )
      with cute.arch.elect_one():
        pipeline_B.consumer_release(consumer_state_B)
      consumer_state_B.advance()

    # ═══ Phase B: mask + softmax → P fp32 in rmem (in-place on acc_S) ═══
    mask_fn(acc_S)
    acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S, transpose=False)
    for r in cutlass.range_constexpr(cute.size(acc_S_mn, mode=[0])):
      lse_val = tLSErLSE[r]
      for c in cutlass.range(cute.size(acc_S_mn, mode=[1]), unroll_full=True):
        acc_S_mn[r, c] = cute.math.exp2(
          acc_S_mn[r, c] * softmax_scale_log2 - lse_val, fastmath=True
        )

    # ═══ Cross-WG handshake: wait WG2 done with prior sP_fp32, then write ═══
    cute.arch.barrier(
      barrier_id=int(NamedBarrierBwd.PEmpty),
      number_of_threads=self.num_mma_threads,  # 256
    )
    copy_P_fp32_r2s(acc_S, dst_idx=p_stage)
    cute.arch.fence_view_async_shared()
    cute.arch.sync_warp()
    cute.arch.barrier_arrive(
      barrier_id=int(NamedBarrierBwd.PFull),
      number_of_threads=self.num_mma_threads,
    )

    # ═══ Wait WG2 to publish sdS via dSFull (256-thread cross-WG handshake) ═══
    # WG2 will arrive(dSFull, 256) after its Phase D STSM(sdS). WG1's sync contributes
    # its own 128; combined with WG2's 128 arrive = 256 → fires. (common_err_kernel #2)
    cute.arch.barrier(
      barrier_id=int(NamedBarrierBwd.dSFull),
      number_of_threads=self.num_mma_threads,
    )

    # ═══ Inner d_pass loop: cooperative Phase E_front ═══
    # WG1 owns acc_dQ_pass[d]_front; reads sdS (full tile) + sKt_front[stage] (half).
    # Each d_pass consumes one pipeline_Kt stage; both WGs cooperatively release each
    # stage (8-warp consumer group, elect_one per WG).
    for d_pass in cutlass.range_constexpr(self.num_d_passes):
      pipeline_Kt.consumer_wait(
        consumer_state_Kt, pipeline_Kt.consumer_try_wait(consumer_state_Kt)
      )
      if cutlass.const_expr(d_pass == 0):
        gemm_w_idx(
          tiled_mma_dQ,
          acc_dQ_pass_0_front,
          tDQrA,
          tDQrB_front,
          zero_init=not dQ_accumulate,
          A_idx=p_stage,
          B_idx=consumer_state_Kt.index,
          wg_wait=0,
        )
      else:
        gemm_w_idx(
          tiled_mma_dQ,
          acc_dQ_pass_1_front,
          tDQrA,
          tDQrB_front,
          zero_init=not dQ_accumulate,
          A_idx=p_stage,
          B_idx=consumer_state_Kt.index,
          wg_wait=0,
        )
      with cute.arch.elect_one():
        pipeline_Kt.consumer_release(consumer_state_Kt)
      consumer_state_Kt.advance()

    # ═══ Tell WG2 we're done reading sdS (256-thread cross-WG, paired with WG2's sync) ═══
    # WG2 will sync(dSEmpty, 256) before its next n_block's Phase D STSM.
    cute.arch.barrier_arrive(
      barrier_id=int(NamedBarrierBwd.dSEmpty),
      number_of_threads=self.num_mma_threads,
    )

    return consumer_state_QdO, consumer_state_B, consumer_state_Kt

  # ════════════════════════════════════════════════════════════════════
  # dQ: WG2 = Phase C (dP=ΣdO@V^T) + Phase D (dS → sdS, arrive PEmpty+dSFull)
  #          + cooperative Phase E_back (acc_dQ_pass[d]_back += sdS @ sKt_back).
  # acc_dQ_pass[0..num_d_passes-1]_back persist in rmem across all n_blocks; per-work_tile
  # epilogue writes dQ[:, d_pass*d_chunk + dQ_n_half : d_pass*d_chunk + d_chunk] slices.
  # ════════════════════════════════════════════════════════════════════
  @cute.jit
  def mma_wg2(
    self,
    tiled_mma_SdP: cute.TiledMma,
    tiled_mma_dQ: cute.TiledMma,  # half-N tiled_mma (cooperative ①)
    mdQ_half: cute.Tensor,  # half-N TMA view of dQ output
    sA: cute.Tensor,
    sB: cute.Tensor,
    sKt_back: cute.Tensor,  # WG2's half of Kt
    sP_fp32: cute.Tensor,
    sdS: cute.Tensor,
    sdPsum: cute.Tensor,
    sEpi_back: cute.Tensor,  # WG2 epilogue staging (back half)
    pipeline_QdO: pipeline.PipelineAsync,
    pipeline_B: pipeline.PipelineAsync,
    pipeline_Kt: pipeline.PipelineAsync,
    tidx: Int32,  # WG-local ∈ [0, 128)
    tma_atom_dQ_half: cute.CopyAtom,
    softmax_scale_log2: Float32,
    softmax_scale: Float32,
    block_info: BlockInfo,
    SeqlenInfoCls: Callable,
    TileSchedulerCls: Callable,
    qhead_per_kvhead_divmod: Optional[FastDivmodDivisor] = None,
  ):
    thr_mma_SdP = tiled_mma_SdP.get_slice(tidx)
    wg_mma_SdP = tiled_mma_SdP.get_slice(0)
    wg_mma_dQ = tiled_mma_dQ.get_slice(0)

    # ── SdP fragments (WG2 uses dO from sA, V from sB) ──
    shape_mnk_SdP = (self.tile_m, self.tile_n, self.d_chunk)
    _, tSrA, tSrB = sm90_utils.partition_fragment_ABC(
      wg_mma_SdP, shape_mnk_SdP, sA, sB, swap_AB=False
    )

    # ── dQ_half fragments (cooperative ①, WG2 back half) ──
    # A = sdS (K_MAJOR), B = sKt_back (MN_MAJOR after transpose). Shape (64, 128, 64).
    sKt_back_t = layout_utils.transpose_view(sKt_back)
    shape_mnk_dQ_half = (self.tile_m, self.dQ_n_half, self.tile_n)
    acc_dQ_pass_0_back = cute.make_rmem_tensor(
      tiled_mma_dQ.partition_shape_C(shape_mnk_dQ_half[:2]), Float32
    )
    acc_dQ_pass_1_back = cute.make_rmem_tensor(
      tiled_mma_dQ.partition_shape_C(shape_mnk_dQ_half[:2]), Float32
    )
    _, tDQrA, tDQrB_back = sm90_utils.partition_fragment_ABC(
      wg_mma_dQ, shape_mnk_dQ_half, sdS, sKt_back_t, swap_AB=False
    )

    # ── dS R2S writer (WG2 STSM rmem dS → sdS[0]) ──
    copy_dS_r2s, _, _ = copy_utils.get_smem_store_C(
      tiled_mma_SdP,
      sdS,
      tidx,
      transpose=False,
      position_independent=True,
    )

    # ── sP_fp32 s2r partition (WG2 reads fp32 P from sP_fp32) ──
    tSsP_fp32_partition = thr_mma_SdP.partition_C(sP_fp32)

    # ── dPsum per-thread row mapping ──
    tLSEsdPsum = layout_utils.mma_partition_C_vec(
      sdPsum, thr_mma_SdP, expand_shape=self.tile_n, is_colvec=True
    )

    # startup: pre-arrive PEmpty so WG1's first barrier(PEmpty, 256) finds
    # 128 arrivals from WG2 and only waits for WG1's own 128. (common_err_kernel #2)
    cute.arch.barrier_arrive(
      barrier_id=int(NamedBarrierBwd.PEmpty),
      number_of_threads=self.num_mma_threads,  # 256
    )

    consumer_state_QdO = cutlass.pipeline.make_pipeline_state(
      cutlass.pipeline.PipelineUserType.Consumer, self.A_stage
    )
    consumer_state_B = cutlass.pipeline.make_pipeline_state(
      cutlass.pipeline.PipelineUserType.Consumer, self.B_stage
    )
    consumer_state_Kt = cutlass.pipeline.make_pipeline_state(
      cutlass.pipeline.PipelineUserType.Consumer, self.Kt_stage
    )

    tile_scheduler = TileSchedulerCls()
    work_tile = tile_scheduler.initial_work_tile_info()

    while work_tile.is_valid_tile:
      m_block, head_idx, batch_idx, _ = work_tile.tile_idx
      seqlen = SeqlenInfoCls(batch_idx)
      n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)

      process_tile = const_expr(
        not self.is_varlen_k
      ) or n_block_min < n_block_max

      mask = AttentionMask(self.tile_m, self.tile_n, seqlen)

      # dQ_accumulate is single-bool for whole work_tile (first n_block
      # zero_inits BOTH d_pass back halves; subsequent n_blocks accumulate).
      dQ_accumulate = Boolean(False)

      for i_n in cutlass.range(n_block_max - n_block_min, unroll=1):
        n_block = n_block_max - 1 - i_n
        mask_fn = partial(
          mask.apply_mask,
          batch_idx=batch_idx,
          head_idx=head_idx,
          m_block=m_block,
          n_block=n_block,
          thr_mma=thr_mma_SdP,
          mask_seqlen=True,
          mask_causal=self.is_causal,
        )

        consumer_state_QdO, consumer_state_B, consumer_state_Kt = self._mma_wg2_one_n_block(
          consumer_state_QdO,
          consumer_state_B,
          consumer_state_Kt,
          tiled_mma_SdP,
          tiled_mma_dQ,
          tSrA,
          tSrB,
          tDQrA,
          tDQrB_back,
          shape_mnk_SdP,
          acc_dQ_pass_0_back,
          acc_dQ_pass_1_back,
          copy_dS_r2s,
          tSsP_fp32_partition,
          sB,
          sKt_back,
          pipeline_QdO,
          pipeline_B,
          pipeline_Kt,
          tLSEsdPsum,
          softmax_scale,
          seqlen,
          n_block,
          tidx,
          mask_fn,
          dQ_accumulate=dQ_accumulate,
        )
        dQ_accumulate = Boolean(True)

      # ── Per work_tile epilogue: write both d_pass back halves ──
      if process_tile:
        self.epilogue_dQ_half_slice(
          acc_dQ_pass_0_back,
          acc_dQ_pass_1_back,
          mdQ_half,
          sEpi_back,
          seqlen,
          tma_atom_dQ_half,
          tiled_mma_dQ,
          tidx,
          m_block,
          head_idx,
          batch_idx,
          is_front_wg=False,
        )
      else:
        # Even with no n_blocks: keep WG1↔WG2 lockstep on Epilogue handshake.
        cute.arch.barrier(
          barrier_id=int(NamedBarrierBwd.Epilogue),
          number_of_threads=self.num_mma_threads,
        )
        cute.arch.barrier(
          barrier_id=int(NamedBarrierBwd.Epilogue),
          number_of_threads=self.num_mma_threads,
        )

      # Per-work_tile WSWG1 sync (288): mirror mma_wg1; producer waits.
      cute.arch.barrier(
        barrier_id=int(NamedBarrierBwd.WarpSchedulerWG1),
        number_of_threads=self.num_producer_threads + self.num_mma_threads,
      )

      tile_scheduler.advance_to_next_work()
      work_tile = tile_scheduler.get_current_work()

    # Flush in-flight dQ TMA stores at kernel exit
    warp_idx_final = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    if warp_idx_final == 8:  # WG2's TMA-issuing warp (back-half stores)
      cute.arch.cp_async_bulk_wait_group(0)

  @cute.jit
  def _mma_wg2_one_n_block(
    self,
    consumer_state_QdO: cutlass.pipeline.PipelineState,
    consumer_state_B: cutlass.pipeline.PipelineState,
    consumer_state_Kt: cutlass.pipeline.PipelineState,
    tiled_mma_SdP: cute.TiledMma,
    tiled_mma_dQ: cute.TiledMma,
    tSrA: cute.Tensor,
    tSrB: cute.Tensor,
    tDQrA: cute.Tensor,
    tDQrB_back: cute.Tensor,
    shape_mnk_SdP: cute.Shape,
    acc_dQ_pass_0_back: cute.Tensor,
    acc_dQ_pass_1_back: cute.Tensor,
    copy_dS_r2s: Callable,
    tSsP_fp32_partition: cute.Tensor,
    sB: cute.Tensor,
    sKt_back: cute.Tensor,
    pipeline_QdO: pipeline.PipelineAsync,
    pipeline_B: pipeline.PipelineAsync,
    pipeline_Kt: pipeline.PipelineAsync,
    tLSEsdPsum: cute.Tensor,
    softmax_scale: Float32,
    seqlen: SeqlenInfoQK,
    n_block: Int32,
    tidx: Int32,
    mask_fn: Callable,
    dQ_accumulate: Boolean = True,
  ):
    p_stage = Int32(0)  # sP_fp32 / sdS single-buffered

    # ═══ Empty release: Q0..Q_{num_d_inner-1} (consumed by WG1 Phase A) ═══
    for _ in cutlass.range_constexpr(self.num_d_inner):
      pipeline_QdO.consumer_wait(
        consumer_state_QdO, pipeline_QdO.consumer_try_wait(consumer_state_QdO)
      )
      with cute.arch.elect_one():
        pipeline_QdO.consumer_release(consumer_state_QdO)
      consumer_state_QdO.advance()

    # ═══ Empty release: K0..K_{num_d_inner-1} (consumed by WG1 Phase A) ═══
    for _ in cutlass.range_constexpr(self.num_d_inner):
      pipeline_B.consumer_wait(
        consumer_state_B, pipeline_B.consumer_try_wait(consumer_state_B)
      )
      with cute.arch.elect_one():
        pipeline_B.consumer_release(consumer_state_B)
      consumer_state_B.advance()

    # ═══ Phase C: dP = ΣdO_d @ V_d^T (num_d_inner WGMMA, real consume dO + V) ═══
    acc_dP = cute.make_rmem_tensor(
      tiled_mma_SdP.partition_shape_C(shape_mnk_SdP[:2]), Float32
    )
    for d_inner in cutlass.range_constexpr(self.num_d_inner):
      pipeline_QdO.consumer_wait(
        consumer_state_QdO, pipeline_QdO.consumer_try_wait(consumer_state_QdO)
      )
      pipeline_B.consumer_wait(
        consumer_state_B, pipeline_B.consumer_try_wait(consumer_state_B)
      )
      self.zero_kv_tail_smem_wg2(
        smem=sB,
        stage_idx=consumer_state_B.index,
        seqlen=seqlen,
        n_block=n_block,
        tidx=tidx,
        d_chunk=self.d_chunk,
      )
      gemm_w_idx(
        tiled_mma_SdP,
        acc_dP,
        tSrA,
        tSrB,
        zero_init=(d_inner == 0),
        A_idx=consumer_state_QdO.index,
        B_idx=consumer_state_B.index,
        wg_wait=0,
      )
      if cutlass.const_expr(d_inner == self.num_d_inner - 1):
        tLSErdPsum = copy_utils.load_s2r(
          tLSEsdPsum[None, consumer_state_QdO.index]
        )
      with cute.arch.elect_one():
        pipeline_QdO.consumer_release(consumer_state_QdO)
      with cute.arch.elect_one():
        pipeline_B.consumer_release(consumer_state_B)
      consumer_state_QdO.advance()
      consumer_state_B.advance()

    # ═══ Phase D: wait WG1 sP (PFull) → s2r → dS = P*(dP-dPsum)*scale ═══
    cute.arch.barrier(
      barrier_id=int(NamedBarrierBwd.PFull),
      number_of_threads=self.num_mma_threads,  # 256
    )
    tdSrP_fp32 = copy_utils.load_s2r(
      tSsP_fp32_partition[None, None, None, p_stage]
    )
    tdSrP_mn = layout_utils.reshape_acc_to_mn(tdSrP_fp32, transpose=False)
    acc_dP_mn = layout_utils.reshape_acc_to_mn(acc_dP, transpose=False)
    for r in cutlass.range_constexpr(cute.size(acc_dP_mn, mode=[0])):
      dpsum_val = tLSErdPsum[r]
      for c in cutlass.range(cute.size(acc_dP_mn, mode=[1]), unroll_full=True):
        acc_dP_mn[
          r,
          c] = (tdSrP_mn[r, c] * (acc_dP_mn[r, c] - dpsum_val) * softmax_scale)

    # ═══ Wait sdS-empty (from WG1 + WG2 prior round acks). For first n_block, WG1's
    # startup pre-arrive provides the missing 128 credit so this sync doesn't deadlock.
    cute.arch.barrier(
      barrier_id=int(NamedBarrierBwd.dSEmpty),
      number_of_threads=self.num_mma_threads,
    )

    # Convert dS (in acc_dP rmem, fp32) → fp16 frgA → STSM into sdS[0].
    tdQrdS = utils.cvt_f16(layout_utils.reshape_acc_to_frgA(acc_dP), self.dtype)
    copy_dS_r2s(tdQrdS, dst_idx=p_stage)
    cute.arch.fence_view_async_shared()
    # WG2-internal sync (128) — ensure all WG2 STSM commits before WG2 reads sdS itself.
    cute.arch.barrier(
      barrier_id=int(NamedBarrierBwd.dSLocal),
      number_of_threads=self.num_threads_per_warp_group,
    )
    # Tell WG1 the sP_fp32 slot is empty so it may write next round.
    cute.arch.barrier_arrive(
      barrier_id=int(NamedBarrierBwd.PEmpty),
      number_of_threads=self.num_mma_threads,
    )
    # Tell WG1 sdS is published — paired with WG1's sync(dSFull, 256).
    cute.arch.barrier_arrive(
      barrier_id=int(NamedBarrierBwd.dSFull),
      number_of_threads=self.num_mma_threads,
    )

    # ═══ Inner d_pass loop: cooperative Phase E_back ═══
    # WG2 owns acc_dQ_pass[d]_back; reads sdS (full tile) + sKt_back[stage] (half).
    for d_pass in cutlass.range_constexpr(self.num_d_passes):
      pipeline_Kt.consumer_wait(
        consumer_state_Kt, pipeline_Kt.consumer_try_wait(consumer_state_Kt)
      )
      # Zero Kt back-half tail rows for masked seqlen — only need once per Kt push,
      # but doing it per-stage keeps logic simple and is idempotent.
      self.zero_kv_tail_smem_wg2(
        smem=sKt_back,
        stage_idx=consumer_state_Kt.index,
        seqlen=seqlen,
        n_block=n_block,
        tidx=tidx,
        d_chunk=self.dQ_n_half,
      )
      if cutlass.const_expr(d_pass == 0):
        gemm_w_idx(
          tiled_mma_dQ,
          acc_dQ_pass_0_back,
          tDQrA,
          tDQrB_back,
          zero_init=not dQ_accumulate,
          A_idx=p_stage,
          B_idx=consumer_state_Kt.index,
          wg_wait=0,
        )
      else:
        gemm_w_idx(
          tiled_mma_dQ,
          acc_dQ_pass_1_back,
          tDQrA,
          tDQrB_back,
          zero_init=not dQ_accumulate,
          A_idx=p_stage,
          B_idx=consumer_state_Kt.index,
          wg_wait=0,
        )
      with cute.arch.elect_one():
        pipeline_Kt.consumer_release(consumer_state_Kt)
      consumer_state_Kt.advance()

    # NOTE: dSEmpty is signalled by WG1 only (one arrive per n_block, paired with
    # WG2's sync at the start of next n_block's Phase D STSM). WG2 does NOT arrive
    # dSEmpty — within WG2 the sdS write→read ordering is guaranteed by program order +
    # dSLocal (128) intra-WG fence; the cross-WG concern is purely WG1's prior read.
    # Adding a WG2 arrive here would double-count (counter goes 384 not 256) and break
    # the cycle

    return consumer_state_QdO, consumer_state_B, consumer_state_Kt

  @cute.jit
  def zero_kv_tail_smem_wg2(
    self,
    smem: cute.Tensor,
    stage_idx: Int32,
    seqlen: SeqlenInfoQK,
    n_block: Int32,
    tidx: Int32,  # WG-local ∈ [0, 128)
    d_chunk: cutlass.Constexpr[int] = None,  # allow half d_chunk for sKt_back
  ):
    """WG2-internal V/Kt tail-row zeroing (128 thread).

        d_chunk controls the column-stride of the SMEM buffer being zeroed. For sB (V chunks)
        and full sKt, this is self.d_chunk. For sKt_back (half-N), pass self.dQ_n_half.
        """
    d_chunk_eff = const_expr(self.d_chunk) if d_chunk is None else d_chunk
    valid_rows = seqlen.seqlen_k - n_block * self.tile_n
    if valid_rows < self.tile_n:
      tail_elems = (self.tile_n - valid_rows) * d_chunk_eff
      for linear_idx in cutlass.range(
        tidx, tail_elems, self.num_threads_per_warp_group, unroll=1
      ):
        row_offset = linear_idx // d_chunk_eff
        col = linear_idx - row_offset * d_chunk_eff
        smem[valid_rows + row_offset, col, stage_idx] = smem.element_type(0.0)
      cute.arch.fence_view_async_shared()
      cute.arch.barrier(
        barrier_id=int(NamedBarrierBwd.VTailZero),
        number_of_threads=self.num_threads_per_warp_group,  # 128 (WG2 only)
      )

  # ════════════════════════════════════════════════════════════════════
  # dQ: Per-work_tile epilogue — cooperative ①.
  # WG1 writes dQ[:, d_pass*d_chunk + 0          : d_pass*d_chunk + dQ_n_half] for each d_pass
  # WG2 writes dQ[:, d_pass*d_chunk + dQ_n_half : d_pass*d_chunk + d_chunk]    for each d_pass
  # mdQ_half is tiled at (tile_m, dQ_n_half), so half_tile_idx along the D axis is
  #     2 * d_pass + (0 = front, 1 = back).
  # sEpi_{front,back} alias sKt_{front,back} physical SMEM (union); we enter the epilogue
  # AFTER all n_blocks of the work_tile are done, so sKt has been fully consumer-released.
  # The Epilogue NamedBarrier (256) handshake serializes the union-flip transition.
  # ════════════════════════════════════════════════════════════════════
  @cute.jit
  def epilogue_dQ_half_slice(
    self,
    acc_dQ_pass_0_half: cute.Tensor,
    acc_dQ_pass_1_half: cute.Tensor,
    mdQ_half: cute.Tensor,
    sEpi_half: cute.Tensor,
    seqlen: SeqlenInfoQK,
    tma_atom_dQ_half: cute.CopyAtom,
    tiled_mma_dQ: cute.TiledMma,
    tidx: Int32,  # WG-local ∈ [0, 128)
    m_block: Int32,
    head_idx: Int32,
    batch_idx: Int32,
    is_front_wg: cutlass.Constexpr[bool],
  ):
    """Called by both WGs at per-work_tile epilogue.

        Iterates over d_pass, writing one half slice per d_pass:
          - WG1 (is_front_wg=True):  dQ[:, d_pass*d_chunk + 0           : d_pass*d_chunk + dQ_n_half]
          - WG2 (is_front_wg=False): dQ[:, d_pass*d_chunk + dQ_n_half  : d_pass*d_chunk + d_chunk]
        """
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    # CTA-absolute warp positions: producer=warp 0..3, WG1=warp 4..7, WG2=warp 8..11.
    tma_warp = const_expr(4 if is_front_wg else 8)

    mdQ_cur = seqlen.offset_batch_Q(
      mdQ_half, batch_idx, dim=3, ragged=self.varlen_q
    )[None, None, head_idx]

    copy_dQ_r2s, _, _ = copy_utils.get_smem_store_C(
      tiled_mma_dQ,
      sEpi_half,
      tidx,
      transpose=False,
      position_independent=True,
    )

    # ═══ SYNC A (entry, 256-thread cross-WG): sKt fully released → safe to use sEpi alias ═══
    # fence_view_async_shared makes all async-proxy writes from Phase E WGMMA visible
    # to generic proxy before we overwrite the same SMEM slot.
    cute.arch.fence_view_async_shared()
    cute.arch.barrier(
      barrier_id=int(NamedBarrierBwd.Epilogue),
      number_of_threads=self.num_mma_threads,
    )

    # ── Iterate d_pass: R2S + TMA store for each accumulator ──
    for d_pass in cutlass.range_constexpr(self.num_d_passes):
      half_tile_idx = 2 * d_pass + const_expr(0 if is_front_wg else 1)
      gdQ_half = cute.local_tile(
        mdQ_cur,
        (self.tile_m, self.dQ_n_half),
        (m_block, half_tile_idx),
      )
      store_dQ_half, _, _ = copy_utils.tma_get_copy_fn(
        tma_atom_dQ_half,
        0,
        cute.make_layout(1),
        sEpi_half,
        gdQ_half,
        single_stage=True,
      )

      # Select the d_pass accumulator at compile time (constexpr branch).
      if cutlass.const_expr(d_pass == 0):
        copy_dQ_r2s(acc_dQ_pass_0_half, dst_idx=None)
      else:
        copy_dQ_r2s(acc_dQ_pass_1_half, dst_idx=None)
      cute.arch.fence_view_async_shared()
      # WG-internal 128-thread fence (different barrier IDs per WG so they don't collide).
      cute.arch.barrier(
        barrier_id=int(
          NamedBarrierBwd.WarpSchedulerWG2 if is_front_wg else NamedBarrierBwd.
          WarpSchedulerWG3
        ),
        number_of_threads=self.num_threads_per_warp_group,
      )
      if warp_idx == tma_warp:
        store_dQ_half()
        cute.arch.cp_async_bulk_commit_group()
      # Wait for THIS d_pass's TMA before the next d_pass's R2S overwrites sEpi_half.
      cute.arch.cp_async_bulk_wait_group(0, read=True)
      cute.arch.barrier(
        barrier_id=int(
          NamedBarrierBwd.WarpSchedulerWG2 if is_front_wg else NamedBarrierBwd.
          WarpSchedulerWG3
        ),
        number_of_threads=self.num_threads_per_warp_group,
      )

    # ═══ SYNC B (exit, 256-thread cross-WG): both WGs done with sEpi → next work_tile ok ═══
    cute.arch.barrier(
      barrier_id=int(NamedBarrierBwd.Epilogue),
      number_of_threads=self.num_mma_threads,
    )

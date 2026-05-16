# Copyright (c) DefTruth, qyjdef@163.com
# Copyright (c) Butterfingrz，13524387014@163.com
# SM90 Backward dKdV Kernel with 4-Pass D-Split for head_dim=512.
#
# Architecture:
#   - 2 WGs: 1 producer (TMA, 128 threads), 1 consumer (MMA, 128 threads)
#   - tile_m=64, tile_n=64, d_chunk=128, num_d_passes=4
#   - K/V persistent SMEM: pre-loaded once per d_pass (4 K + 4 V chunks)
#   - Q/dO streaming via pipeline_A (3 stages)
#   - Per d_pass, 6 phases per (n_block, m_block):
#       Phase 1: S = Q @ K^T (4 × d_inner=128 reduction, K from persistent SMEM)
#       Phase 2: P = exp2(S * scale_log2 - LSE)
#       Phase 3: dP = dO @ V^T (4 × d_inner=128 reduction, V from persistent SMEM)
#       Phase 4: dS = P * (dP - dPsum)
#       Phase 5: dV += P^T @ dO_d_pass  (via SMEM)
#       Phase 6: dK += dS^T @ Q_d_pass  (via SMEM)
#   - mma_dkv_is_rs = False (P/dS through SMEM for simplicity)

import math
from typing import Callable, Optional, Type
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass.cute.nvgpu import cpasync, warpgroup
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


class FlashBwdDKDV_SplitD_Sm90:
  """SM90 backward dKdV kernel (dual asymmetric MMA WG + d_chunk=256 + K/V persistence).

    Computes only dK and dV (no dQ). dQ is handled by a separate kernel.
    Each CTA has 3 warp groups: 1 TMA producer + WG1 (S/softmax/dV) + WG2 (dP/dS/dK).
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
    self.tile_hdim = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
    head_dim_v = head_dim_v if head_dim_v is not None else head_dim
    self.tile_hdimv = int(math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of)
    self.check_hdim_oob = head_dim != self.tile_hdim
    self.check_hdim_v_oob = head_dim_v != self.tile_hdimv

    self.is_causal = is_causal
    self.qhead_per_kvhead = qhead_per_kvhead
    self.tile_m = tile_m
    self.tile_n = tile_n
    self.use_pdl = False
    self.qk_acc_dtype = Float32
    self.buffer_align_bytes = 1024

    # ── SplitD parameters (d_chunk=256, num_d_passes=2, num_d_inner=2) ──
    self.d_chunk = 256  # output slice width for dK/dV
    self.num_d_passes = self.tile_hdim // self.d_chunk  # = 2
    self.num_d_inner = self.tile_hdim // self.d_chunk  # = 2
    assert self.tile_hdim % self.d_chunk == 0
    assert self.tile_hdimv % self.d_chunk == 0

    # ── MMA WG configuration──
    # SdP MMA stays num_wg_mma=1 (each WG independently holds acc_S or acc_dP).
    # dV/dK MMA stays num_wg_mma=1 (only WG1 or WG2 partitions, never both).
    self.num_wg_mma_SdP = 1
    self.num_wg_mma_dKV = 1
    self.num_wg_mma = 1
    self.num_threads = 384
    self.num_threads_per_warp_group = 128
    self.num_producer_threads = 32  # actual producer worker threads (warp 0 elect_one)
    self.num_mma_threads = 256  # WG1 + WG2
    self.num_mma_regs_wg1 = 224
    self.num_mma_regs_wg2 = 224
    self.num_mma_regs = 256  # legacy; not used in MVP-2' dispatch
    self.num_producer_regs = 56

    # ── Pipeline stages──
    # A_stage 3→2: d_chunk=256 doubles per-stage size; drop one stage to fit SMEM ≤ 228KB.
    self.A_stage = 2
    # sP/sdS single-buffered; PFull/PEmpty (256-thread named barrier) serializes cross-WG handoff.
    self.PdS_stage = 1

    # ── K/V persistence : preload ONCE per work_tile, reused across d_pass ──
    self.K_persist_chunks = self.num_d_inner  # = 2 chunks of d_chunk=256
    self.V_persist_chunks = self.num_d_inner  # = 2

    # ── MMA layout configuration ──
    self.SdP_swapAB = False
    self.dKV_swapAB = False
    self.AtomLayoutMSdP = 1
    self.AtomLayoutNdKV = 1
    self.mma_dkv_is_rs = False

  def _setup_attributes(self):
    # sA: (tile_m, d_chunk) — holds Q_chunk, dO_chunk, dO_d_pass, Q_d_pass
    self.sA_layout = sm90_utils.make_smem_layout(
      self.dtype,
      LayoutEnum.ROW_MAJOR,
      (self.tile_m, self.d_chunk),
      stage=self.A_stage,
      major_mode_size=self.d_chunk,  # accommodate sA^T
    )
    # sK_persist: (tile_n, d_chunk) × num_d_inner — persistent K across m_blocks
    self.sK_persist_layout = sm90_utils.make_smem_layout(
      self.dtype,
      LayoutEnum.ROW_MAJOR,
      (self.tile_n, self.d_chunk),
      stage=self.K_persist_chunks,
    )
    # sV_persist: (tile_n, d_chunk) × num_d_inner — persistent V across m_blocks
    self.sV_persist_layout = sm90_utils.make_smem_layout(
      self.dtype,
      LayoutEnum.ROW_MAJOR,
      (self.tile_n, self.d_chunk),
      stage=self.V_persist_chunks,
    )
    # sB_epi_layout: per-chunk layout for epilogue TMA store (same as K/V per-chunk)
    self.sB_epi_layout = cute.select(self.sK_persist_layout, mode=[0, 1])
    # sP: (tile_m, tile_n) — P for dV GEMM A operand
    wg_n_SdP = self.num_wg_mma // self.AtomLayoutMSdP
    wg_n_dKV = self.AtomLayoutNdKV
    self.sPdS_layout = sm90_utils.make_smem_layout(
      self.dtype,
      LayoutEnum.ROW_MAJOR,
      (self.tile_m, self.tile_n),
      stage=self.PdS_stage,
      major_mode_size=math.gcd(self.tile_n // wg_n_SdP, self.tile_n // wg_n_dKV),
    )
    # bf16-roundtrip-free P channel:
    # sP_fp32 holds the SAME P values as sP (bf16) but in fp32. WG1 writes both;
    # WG1 dV WGMMA still reads bf16 sP (WGMMA requires bf16/fp16 A operand).
    # WG2 reads sP_fp32 for dS = P*(dP-dPsum)*scale → no fp32→bf16→fp32 precision loss.
    self.sP_fp32_layout = sm90_utils.make_smem_layout(
      Float32,
      LayoutEnum.ROW_MAJOR,
      (self.tile_m, self.tile_n),
      stage=self.PdS_stage,
      major_mode_size=math.gcd(self.tile_n // wg_n_SdP, self.tile_n // wg_n_dKV),
    )

  def _get_tiled_mma(self):
    # ── SdP: S = Q @ K^T, dP = dO @ V^T ──
    # shape_mnk: (tile_m, tile_n, d_chunk) = (64, 64, 128)
    atom_layout_SdP = (self.AtomLayoutMSdP, self.num_wg_mma // self.AtomLayoutMSdP, 1)
    tiler_mn_SdP = (self.tile_m // atom_layout_SdP[0], self.tile_n // atom_layout_SdP[1])
    tiled_mma_SdP = sm90_utils_basic.make_trivial_tiled_mma(
      self.dtype,
      self.dtype,
      warpgroup.OperandMajorMode.K,
      warpgroup.OperandMajorMode.K,
      self.qk_acc_dtype,
      atom_layout_mnk=atom_layout_SdP,
      tiler_mn=tiler_mn_SdP,
    )

    # ── dKV: dV = P^T @ dO_d, dK = dS^T @ Q_d ──
    # shape_mnk: (tile_n, d_chunk, tile_m) = (64, 128, 64)
    atom_layout_dKV = (self.AtomLayoutNdKV, self.num_wg_mma // self.AtomLayoutNdKV, 1)
    # dV: M=tile_n, N=d_chunk
    tiler_mn_dV = (self.tile_n // atom_layout_dKV[0], self.d_chunk // atom_layout_dKV[1])
    tiled_mma_dV = sm90_utils_basic.make_trivial_tiled_mma(
      self.dtype,
      self.dtype,
      warpgroup.OperandMajorMode.MN,
      warpgroup.OperandMajorMode.MN,
      Float32,
      atom_layout_mnk=atom_layout_dKV,
      tiler_mn=tiler_mn_dV,
    )
    # dK: same shape as dV
    tiler_mn_dK = (self.tile_n // atom_layout_dKV[0], self.d_chunk // atom_layout_dKV[1])
    tiled_mma_dK = sm90_utils_basic.make_trivial_tiled_mma(
      self.dtype,
      self.dtype,
      warpgroup.OperandMajorMode.MN,
      warpgroup.OperandMajorMode.MN,
      Float32,
      atom_layout_mnk=atom_layout_dKV,
      tiler_mn=tiler_mn_dK,
    )
    return tiled_mma_SdP, tiled_mma_dK, tiled_mma_dV

  def _get_shared_storage_cls(self):
    sA_struct = cute.struct.Align[
      cute.struct.MemRange[self.dtype, cute.cosize(self.sA_layout)],
      self.buffer_align_bytes,
    ]
    sK_persist_struct = cute.struct.Align[
      cute.struct.MemRange[self.dtype, cute.cosize(self.sK_persist_layout)],
      self.buffer_align_bytes,
    ]
    sV_persist_struct = cute.struct.Align[
      cute.struct.MemRange[self.dtype, cute.cosize(self.sV_persist_layout)],
      self.buffer_align_bytes,
    ]
    # Dedicated single-stage epilogue buffer for TMA dK/dV store.
    sEpi_struct = cute.struct.Align[
      cute.struct.MemRange[self.dtype, cute.cosize(self.sB_epi_layout)],
      self.buffer_align_bytes,
    ]

    # Physical size = max(sA = A_stage × d_chunk-stage size, sEpi = tile_n × d_chunk).
    @cute.union
    class SmemAEpi_t:
      sA: sA_struct  # mainloop: pipeline_A items (Q/dO/dO_d_pass/Q_d_pass)
      sEpi: sEpi_struct  # epilogue: dV (WG1) → dK (WG2) staging via TMA store

    @cute.struct
    class SharedStorageDKDV:
      mbar_ptr_A: cute.struct.MemRange[cutlass.Int64, self.A_stage * 2]
      mbar_ptr_K: cute.struct.MemRange[cutlass.Int64, self.K_persist_chunks * 2]
      mbar_ptr_V: cute.struct.MemRange[cutlass.Int64, self.V_persist_chunks * 2]
      sLSE: cute.struct.MemRange[
        Float32,
        cute.round_up(self.tile_m, 64) * self.A_stage,
      ]
      sdPsum: cute.struct.MemRange[
        Float32,
        cute.round_up(self.tile_m, 64) * self.A_stage,
      ]
      sAEpi: SmemAEpi_t  # sA (mainloop) | sEpi (epilogue)
      sK_persist: sK_persist_struct
      sV_persist: sV_persist_struct
      sP: cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(self.sPdS_layout)], 1024]
      # Bf16-roundtrip-free P channel: WG1 writes fp32 P here in P2,
      # WG2 reads it in P4 for dS computation. Avoids fp32→bf16→fp32 precision loss.
      sP_fp32: cute.struct.Align[cute.struct.MemRange[Float32, cute.cosize(self.sP_fp32_layout)], 1024]
      sdS: cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(self.sPdS_layout)], 1024]

    return SharedStorageDKDV

  @cute.jit
  def __call__(
    self,
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mdO: cute.Tensor,
    mLSE: cute.Tensor,
    mdPsum: cute.Tensor,
    mdK: cute.Tensor,
    mdV: cute.Tensor,
    softmax_scale: Float32,
    mCuSeqlensQ: Optional[cute.Tensor] = None,
    mCuSeqlensK: Optional[cute.Tensor] = None,
    stream: cuda.CUstream = None,
  ):
    mQ, mK, mV, mdO, mdK, mdV = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mdO, mdK, mdV)]
    mLSE, mdPsum = [assume_tensor_aligned(t) for t in (mLSE, mdPsum)]

    # Transpose: (b, s, n, h) → (s, h, n, b)
    def _qkv_transpose(t):
      return layout_utils.select(t, [1, 3, 2, 0] if cute.rank(t.shape) == 4 else [0, 2, 1])

    mQ, mK, mV, mdO, mdK, mdV = [_qkv_transpose(t) for t in (mQ, mK, mV, mdO, mdK, mdV)]
    # Stats: (b, n, s) → (s, n, b)
    LSE_transpose = [2, 1, 0] if cute.rank(mLSE.shape) == 3 else [1, 0]
    mLSE = layout_utils.select(mLSE, LSE_transpose)
    mdPsum = layout_utils.select(mdPsum, LSE_transpose)

    tiled_mma_SdP, tiled_mma_dK, tiled_mma_dV = self._get_tiled_mma()
    # num_mma_threads = 256 (WG1+WG2) is set in __init__; do NOT
    # overwrite from tiled_mma_SdP.size (= 128 per WG with num_wg_mma=1).
    # The per-WG WGMMA thread count is tracked separately if ever needed.
    self.num_mma_threads_per_wg = int(tiled_mma_SdP.size)  # = 128
    self._setup_attributes()
    SharedStorage = self._get_shared_storage_cls()

    sK_layout_sel = cute.select(self.sK_persist_layout, mode=[0, 1])
    self.tma_copy_bytes = {
      name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1]))
      for name, mX, layout in [
        ("A", mQ, self.sA_layout),
        ("KV", mK, self.sK_persist_layout),
      ]
    }
    self.tma_copy_bytes["LSE"] = self.tile_m * Float32.width // 8
    self.tma_copy_bytes["dPsum"] = self.tile_m * Float32.width // 8

    sA_layout_sel = cute.select(self.sA_layout, mode=[0, 1])
    gmem_tiled_copy_g2s = cpasync.CopyBulkTensorTileG2SOp()
    # Q: tile shape (tile_m, d_chunk) = (64, 128)
    tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
      gmem_tiled_copy_g2s,
      mQ,
      sA_layout_sel,
      (self.tile_m, self.d_chunk),
    )
    # dO: tile shape (tile_m, d_chunk) = (64, 128)
    tma_atom_dO, tma_tensor_dO = cpasync.make_tiled_tma_atom(
      gmem_tiled_copy_g2s,
      mdO,
      sA_layout_sel,
      (self.tile_m, self.d_chunk),
    )
    # K: tile shape (tile_n, d_chunk) = (64, 128), persistent in SMEM
    tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
      gmem_tiled_copy_g2s,
      mK,
      sK_layout_sel,
      (self.tile_n, self.d_chunk),
    )
    # V: tile shape (tile_n, d_chunk) = (64, 128), persistent in SMEM
    tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
      gmem_tiled_copy_g2s,
      mV,
      sK_layout_sel,
      (self.tile_n, self.d_chunk),
    )
    # dK/dV: store atoms + TMA tensors.
    self.varlen_k = mCuSeqlensK is not None
    self.is_varlen_q = mCuSeqlensQ is not None
    gmem_tiled_copy_s2g = cpasync.CopyBulkTensorTileS2GOp()
    sB_epi_sel = cute.select(self.sK_persist_layout, mode=[0, 1])
    # Varlen K: create ragged TMA tensors for dK/dV output writes
    mdK_tma = copy_utils.create_ragged_tensor_for_tma(mdK, ragged_dim=0, ptr_shift=True) if self.varlen_k else mdK
    mdV_tma = copy_utils.create_ragged_tensor_for_tma(mdV, ragged_dim=0, ptr_shift=True) if self.varlen_k else mdV
    tma_atom_dK, tma_tensor_dK = cpasync.make_tiled_tma_atom(
      gmem_tiled_copy_s2g,
      mdK_tma,
      sB_epi_sel,
      (self.tile_n, self.d_chunk),
    )
    tma_atom_dV, tma_tensor_dV = cpasync.make_tiled_tma_atom(
      gmem_tiled_copy_s2g,
      mdV_tma,
      sB_epi_sel,
      (self.tile_n, self.d_chunk),
    )

    # ── Tile scheduler ──
    if const_expr(mCuSeqlensK is not None):
      TileScheduler = SingleTileVarlenScheduler
    else:
      TileScheduler = SingleTileScheduler
    num_n_blocks = cute.ceil_div(cute.size(mK.shape[0]), self.tile_n)
    num_batch = cute.size(mK.shape[3]) if cute.rank(mK.shape) == 4 else cute.size(mCuSeqlensK.shape[0] - 1)
    tile_sched_args = TileSchedulerArguments(
      num_n_blocks,  # num_m_blocks (but backward swaps M/N)
      cute.size(mK.shape[2]),  # num_heads
      num_batch,
      1,  # cluster_size
      cute.size(mQ.shape[0]),  # seqlen_k → actually seqlen_q for m_block range
      mQ.shape[1],  # head_dim_qk
      mV.shape[1],  # head_dim_v
      total_q=cute.size(mK.shape[0]),
      tile_shape_mn=(self.tile_n, self.tile_m),  # swapped for backward
      mCuSeqlensQ=mCuSeqlensK,
      qhead_per_kvhead_packgqa=1,
      element_size=self.dtype.width // 8,
      lpt=self.is_causal,
    )
    tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
    grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

    softmax_scale_log2 = softmax_scale * math.log2(math.e)

    qhead_per_kvhead_divmod = None
    if const_expr(self.qhead_per_kvhead > 1):
      qhead_per_kvhead_divmod = FastDivmodDivisor(self.qhead_per_kvhead)

    self.kernel(
      tma_tensor_Q,
      tma_tensor_K,
      tma_tensor_V,
      tma_tensor_dO,
      tma_tensor_dK,
      tma_tensor_dV,
      tma_atom_Q,
      tma_atom_dO,
      tma_atom_K,
      tma_atom_V,
      tma_atom_dK,
      tma_atom_dV,
      mLSE,
      mdPsum,
      mCuSeqlensQ,
      mCuSeqlensK,
      softmax_scale_log2,
      softmax_scale,
      self.sA_layout,
      self.sK_persist_layout,
      self.sV_persist_layout,
      self.sPdS_layout,
      self.sP_fp32_layout,
      tiled_mma_SdP,
      tiled_mma_dK,
      tiled_mma_dV,
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
    mdK: cute.Tensor,
    mdV: cute.Tensor,
    tma_atom_Q: cute.CopyAtom,
    tma_atom_dO: cute.CopyAtom,
    tma_atom_K: cute.CopyAtom,
    tma_atom_V: cute.CopyAtom,
    tma_atom_dK: cute.CopyAtom,
    tma_atom_dV: cute.CopyAtom,
    mLSE: cute.Tensor,
    mdPsum: cute.Tensor,
    mCuSeqlensQ: Optional[cute.Tensor],
    mCuSeqlensK: Optional[cute.Tensor],
    softmax_scale_log2: Float32,
    softmax_scale: Float32,
    sA_layout: cute.ComposedLayout,
    sK_persist_layout: cute.ComposedLayout,
    sV_persist_layout: cute.ComposedLayout,
    sPdS_layout: cute.ComposedLayout,
    sP_fp32_layout: cute.ComposedLayout,
    tiled_mma_SdP: cute.TiledMma,
    tiled_mma_dK: cute.TiledMma,
    tiled_mma_dV: cute.TiledMma,
    tile_sched_params: ParamsBase,
    TileScheduler: cutlass.Constexpr[Callable],
    SharedStorage: cutlass.Constexpr[Callable],
    qhead_per_kvhead_divmod: Optional[FastDivmodDivisor] = None,
  ):
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    if warp_idx == 0:
      for atom in [tma_atom_Q, tma_atom_dO, tma_atom_K, tma_atom_V, tma_atom_dK, tma_atom_dV]:
        cpasync.prefetch_descriptor(atom)

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)

    pipeline_producer_group = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread)
    # pipeline_K consumed only by WG1, pipeline_V only by WG2.
    # Each WG = 4 warps → consumer_group size = 4 (1 elect_one release per warp).
    # pipeline_A consumed by both WG1+WG2 (each m_block, real consumer + empty release) → 8 warps.
    pipeline_consumer_wg1 = cutlass.pipeline.CooperativeGroup(
      cutlass.pipeline.Agent.Thread,
      self.num_threads_per_warp_group // cute.arch.WARP_SIZE,  # = 4 warps
    )
    pipeline_consumer_wg2 = cutlass.pipeline.CooperativeGroup(
      cutlass.pipeline.Agent.Thread,
      self.num_threads_per_warp_group // cute.arch.WARP_SIZE,  # = 4 warps
    )
    pipeline_consumer_a = cutlass.pipeline.CooperativeGroup(
      cutlass.pipeline.Agent.Thread,
      self.num_mma_threads // cute.arch.WARP_SIZE,  # = 8 warps (WG1+WG2)
    )
    # pipeline_K: K_persist_chunks (=2) chunks of d_chunk=256 each; WG1 only.
    pipeline_K = pipeline.PipelineTmaAsync.create(
      barrier_storage=storage.mbar_ptr_K.data_ptr(),
      num_stages=self.K_persist_chunks,
      producer_group=pipeline_producer_group,
      consumer_group=pipeline_consumer_wg1,
      tx_count=self.tma_copy_bytes["KV"],
      defer_sync=True,
    )
    # pipeline_V: V_persist_chunks (=2) chunks; WG2 only.
    pipeline_V = pipeline.PipelineTmaAsync.create(
      barrier_storage=storage.mbar_ptr_V.data_ptr(),
      num_stages=self.V_persist_chunks,
      producer_group=pipeline_producer_group,
      consumer_group=pipeline_consumer_wg2,
      tx_count=self.tma_copy_bytes["KV"],
      defer_sync=True,
    )
    # pipeline_A: Q/dO streaming. Base tx_count covers sA only; LSE/dPsum
    # added via extra_tx_count on the last Q/dO chunk.
    pipeline_A = pipeline.PipelineTmaAsync.create(
      barrier_storage=storage.mbar_ptr_A.data_ptr(),
      num_stages=self.A_stage,
      producer_group=pipeline_producer_group,
      consumer_group=pipeline_consumer_a,
      tx_count=self.tma_copy_bytes["A"],
      defer_sync=True,
    )

    pipeline_init_arrive(cluster_shape_mn=(1, 1), is_relaxed=True)
    pipeline_init_wait(cluster_shape_mn=(1, 1))

    # sA / sEpi share physical SMEM via union; mainloop uses sA view,
    # epilogue uses sEpi view (with explicit fence + Epilogue barrier at transition).
    sA = storage.sAEpi.sA.get_tensor(sA_layout.outer, swizzle=sA_layout.inner)
    sK_persist = storage.sK_persist.get_tensor(sK_persist_layout.outer, swizzle=sK_persist_layout.inner)
    sV_persist = storage.sV_persist.get_tensor(sV_persist_layout.outer, swizzle=sV_persist_layout.inner)
    sP = storage.sP.get_tensor(sPdS_layout.outer, swizzle=sPdS_layout.inner)

    # fp32 P buffer: WG1 writes here (alongside bf16 sP); WG2 reads here for dS computation.
    sP_fp32 = storage.sP_fp32.get_tensor(sP_fp32_layout.outer, swizzle=sP_fp32_layout.inner)
    sdS = storage.sdS.get_tensor(sPdS_layout.outer, swizzle=sPdS_layout.inner)

    # Single-stage SMEM view for epilogue TMA store (alias of sA's physical slot).
    sB_epi_layout_sel = cute.select(sK_persist_layout, mode=[0, 1])
    sB_epi = storage.sAEpi.sEpi.get_tensor(sB_epi_layout_sel.outer, swizzle=sB_epi_layout_sel.inner)
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
      False,  # is_split_kv
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
    tidx, _, _ = cute.arch.thread_idx()
    warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)

    if warp_group_idx == 0:
      # Producer WG: only warp 0 issues TMA; others idle (held by setmaxregister_decrease).
      cute.arch.setmaxregister_decrease(self.num_producer_regs)
      if warp_idx == 0:
        self.load(
          mQ,
          mK,
          mV,
          mdO,
          mLSE,
          mdPsum,
          sA,
          sK_persist,
          sV_persist,
          sLSE,
          sdPsum,
          tma_atom_Q,
          tma_atom_dO,
          tma_atom_K,
          tma_atom_V,
          pipeline_A,
          pipeline_K,
          pipeline_V,
          block_info,
          SeqlenInfoCls,
          TileSchedulerCls,
          qhead_per_kvhead_divmod,
        )
    elif warp_group_idx == 1:
      # WG1: S + P + dV. tidx_in_wg ∈ [0, 128).
      cute.arch.setmaxregister_increase(self.num_mma_regs_wg1)
      tidx_in_wg = tidx - self.num_threads_per_warp_group
      self.mma_wg1(
        tiled_mma_SdP,
        tiled_mma_dV,
        mdV,
        sA,
        sK_persist,
        sP,
        sP_fp32,
        sLSE,
        sB_epi,
        pipeline_A,
        pipeline_K,
        tidx_in_wg,
        tma_atom_dV,
        softmax_scale_log2,
        softmax_scale,
        block_info,
        SeqlenInfoCls,
        TileSchedulerCls,
        qhead_per_kvhead_divmod,
      )
    else:
      # WG2: dP + dS + dK. tidx_in_wg ∈ [0, 128).
      cute.arch.setmaxregister_increase(self.num_mma_regs_wg2)
      tidx_in_wg = tidx - 2 * self.num_threads_per_warp_group
      self.mma_wg2(
        tiled_mma_SdP,
        tiled_mma_dK,
        mdK,
        sA,
        sV_persist,
        sP_fp32,
        sdS,
        sdPsum,
        sB_epi,
        pipeline_A,
        pipeline_V,
        tidx_in_wg,
        tma_atom_dK,
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
    mLSE: cute.Tensor,
    mdPsum: cute.Tensor,
    sA: cute.Tensor,
    sK_persist: cute.Tensor,
    sV_persist: cute.Tensor,
    sLSE: cute.Tensor,
    sdPsum: cute.Tensor,
    tma_atom_Q: cute.CopyAtom,
    tma_atom_dO: cute.CopyAtom,
    tma_atom_K: cute.CopyAtom,
    tma_atom_V: cute.CopyAtom,
    pipeline_A: pipeline.PipelineAsync,
    pipeline_K: pipeline.PipelineAsync,
    pipeline_V: pipeline.PipelineAsync,
    block_info: BlockInfo,
    SeqlenInfoCls: Callable,
    TileSchedulerCls: Callable,
    qhead_per_kvhead_divmod: Optional[FastDivmodDivisor] = None,
  ):
    producer_state_A = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, self.A_stage)
    producer_state_K = cutlass.pipeline.make_pipeline_state(
      cutlass.pipeline.PipelineUserType.Producer, self.K_persist_chunks
    )
    producer_state_V = cutlass.pipeline.make_pipeline_state(
      cutlass.pipeline.PipelineUserType.Producer, self.V_persist_chunks
    )

    tile_scheduler = TileSchedulerCls()
    work_tile = tile_scheduler.initial_work_tile_info()
    did_produce_A = Boolean(False)
    did_produce_K = Boolean(False)
    did_produce_V = Boolean(False)

    while work_tile.is_valid_tile:
      n_block, head_idx_kv, batch_idx, _ = work_tile.tile_idx
      seqlen = SeqlenInfoCls(batch_idx)

      # K/V slicing — invariant across Q heads in GQA group
      mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[None, None, head_idx_kv]
      mV_cur = seqlen.offset_batch_K(mV, batch_idx, dim=3)[None, None, head_idx_kv]

      m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)
      process_tile = const_expr(not self.is_varlen_q) or m_block_min < m_block_max

      if process_tile:
        # ═══ K/V preload ONCE per work_tile (Cross-pass reuse) ═══
        # K (= num_d_inner chunks of d_chunk=256) → sK_persist via pipeline_K
        for d_inner in cutlass.range_constexpr(self.num_d_inner):
          gK_d = cute.local_tile(mK_cur, (self.tile_n, self.d_chunk), (None, d_inner))
          load_K_d, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_K,
            0,
            cute.make_layout(1),
            gK_d,
            sK_persist,
          )
          pipeline_K.producer_acquire(producer_state_K)
          load_K_d(
            src_idx=n_block,
            dst_idx=d_inner,
            tma_bar_ptr=pipeline_K.producer_get_barrier(producer_state_K),
          )
          pipeline_K.producer_commit(producer_state_K)
          did_produce_K = Boolean(True)
          producer_state_K.advance()

        # V (= num_d_inner chunks) → sV_persist via pipeline_V
        for d_inner in cutlass.range_constexpr(self.num_d_inner):
          gV_d = cute.local_tile(mV_cur, (self.tile_n, self.d_chunk), (None, d_inner))
          load_V_d, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_V,
            0,
            cute.make_layout(1),
            gV_d,
            sV_persist,
          )
          pipeline_V.producer_acquire(producer_state_V)
          load_V_d(
            src_idx=n_block,
            dst_idx=d_inner,
            tma_bar_ptr=pipeline_V.producer_get_barrier(producer_state_V),
          )
          pipeline_V.producer_commit(producer_state_V)
          did_produce_V = Boolean(True)
          producer_state_V.advance()

        # ── Outer d_pass loop (after K/V preload, inside work_tile) ──
        for d_pass in cutlass.range_constexpr(self.num_d_passes):
          # ── GQA Q-head loop: iterate over all Q heads in this KV group ──
          for q_head_offset in cutlass.range(self.qhead_per_kvhead, unroll=1):
            head_idx_q = head_idx_kv * self.qhead_per_kvhead + q_head_offset

            # Q/dO/LSE/dPsum slicing — per Q head
            mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx_q]
            mdO_cur = seqlen.offset_batch_Q(mdO, batch_idx, dim=3)[None, None, head_idx_q]
            mLSE_cur = seqlen.offset_batch_Q(mLSE, batch_idx, dim=2, padded=True)[None, head_idx_q]
            mdPsum_cur = seqlen.offset_batch_Q(mdPsum, batch_idx, dim=2, padded=True)[None, head_idx_q]

            gLSE = cute.local_tile(mLSE_cur, (self.tile_m, ), (None, ))
            gdPsum = cute.local_tile(mdPsum_cur, (self.tile_m, ), (None, ))
            if const_expr(self.use_pdl):
              cute.arch.griddepcontrol_wait()
            load_LSE = copy_utils.cpasync_bulk_get_copy_fn(gLSE, sLSE)
            load_dPsum = copy_utils.cpasync_bulk_get_copy_fn(gdPsum, sdPsum)

            for m_block in cutlass.range(m_block_min, m_block_max, unroll=1):
              # ═══ Phase 1: Q0..Q{num_d_inner-1} → sA. LSE piggyback on last Q. ═══
              for d_inner in cutlass.range_constexpr(self.num_d_inner):
                gQ_d = cute.local_tile(mQ_cur, (self.tile_m, self.d_chunk), (None, d_inner))
                load_Q_d, _, _ = copy_utils.tma_get_copy_fn(
                  tma_atom_Q,
                  0,
                  cute.make_layout(1),
                  gQ_d,
                  sA,
                )
                if cutlass.const_expr(d_inner == self.num_d_inner - 1):
                  pipeline_A.producer_acquire(
                    producer_state_A,
                    extra_tx_count=self.tma_copy_bytes["LSE"],
                  )
                else:
                  pipeline_A.producer_acquire(producer_state_A)
                load_Q_d(
                  src_idx=m_block,
                  dst_idx=producer_state_A.index,
                  tma_bar_ptr=pipeline_A.producer_get_barrier(producer_state_A),
                )
                if cutlass.const_expr(d_inner == self.num_d_inner - 1):
                  load_LSE(
                    src_idx=m_block,
                    dst_idx=producer_state_A.index,
                    tma_bar_ptr=pipeline_A.producer_get_barrier(producer_state_A),
                  )
                pipeline_A.producer_commit(producer_state_A)
                did_produce_A = Boolean(True)
                producer_state_A.advance()

              # ═══ Phase 3: dO0..dO{num_d_inner-1} → sA. dPsum on last dO. ═══
              for d_inner in cutlass.range_constexpr(self.num_d_inner):
                gdO_d = cute.local_tile(mdO_cur, (self.tile_m, self.d_chunk), (None, d_inner))
                load_dO_d, _, _ = copy_utils.tma_get_copy_fn(
                  tma_atom_dO,
                  0,
                  cute.make_layout(1),
                  gdO_d,
                  sA,
                )
                if cutlass.const_expr(d_inner == self.num_d_inner - 1):
                  pipeline_A.producer_acquire(
                    producer_state_A,
                    extra_tx_count=self.tma_copy_bytes["dPsum"],
                  )
                else:
                  pipeline_A.producer_acquire(producer_state_A)
                load_dO_d(
                  src_idx=m_block,
                  dst_idx=producer_state_A.index,
                  tma_bar_ptr=pipeline_A.producer_get_barrier(producer_state_A),
                )
                if cutlass.const_expr(d_inner == self.num_d_inner - 1):
                  load_dPsum(
                    src_idx=m_block,
                    dst_idx=producer_state_A.index,
                    tma_bar_ptr=pipeline_A.producer_get_barrier(producer_state_A),
                  )
                pipeline_A.producer_commit(producer_state_A)
                did_produce_A = Boolean(True)
                producer_state_A.advance()

              # ═══ Phase 5: dO_d_pass → sA (for WG1 dV GEMM) ═══
              gdO_pass = cute.local_tile(mdO_cur, (self.tile_m, self.d_chunk), (None, d_pass))
              load_dO_pass, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_dO,
                0,
                cute.make_layout(1),
                gdO_pass,
                sA,
              )
              pipeline_A.producer_acquire(producer_state_A)
              load_dO_pass(
                src_idx=m_block,
                dst_idx=producer_state_A.index,
                tma_bar_ptr=pipeline_A.producer_get_barrier(producer_state_A),
              )
              pipeline_A.producer_commit(producer_state_A)
              did_produce_A = Boolean(True)
              producer_state_A.advance()

              # ═══ Phase 6: Q_d_pass → sA (for WG2 dK GEMM) ═══
              gQ_pass = cute.local_tile(mQ_cur, (self.tile_m, self.d_chunk), (None, d_pass))
              load_Q_pass, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Q,
                0,
                cute.make_layout(1),
                gQ_pass,
                sA,
              )
              pipeline_A.producer_acquire(producer_state_A)
              load_Q_pass(
                src_idx=m_block,
                dst_idx=producer_state_A.index,
                tma_bar_ptr=pipeline_A.producer_get_barrier(producer_state_A),
              )
              pipeline_A.producer_commit(producer_state_A)
              did_produce_A = Boolean(True)
              producer_state_A.advance()

          # WarpSchedulerWG1 barrier per d_pass
          # (NOT work_tile end). sA ↔ sEpi union forces producer to wait
          # until consumer epilogue done before loading next d_pass into sA.
          cute.arch.barrier(
            barrier_id=int(NamedBarrierBwd.WarpSchedulerWG1),
            number_of_threads=self.num_producer_threads + self.num_mma_threads,
          )

      tile_scheduler.advance_to_next_work()
      work_tile = tile_scheduler.get_current_work()

    if did_produce_A:
      pipeline_A.producer_tail(producer_state_A)
    if did_produce_K:
      pipeline_K.producer_tail(producer_state_K)
    if did_produce_V:
      pipeline_V.producer_tail(producer_state_V)

  # ════════════════════════════════════════════════════════════════════
  # MVP-2': WG1 = S (P1) + softmax→sP→arrive PFull (P2) + dV (P5)
  # ════════════════════════════════════════════════════════════════════
  @cute.jit
  def mma_wg1(
    self,
    tiled_mma_SdP: cute.TiledMma,
    tiled_mma_dV: cute.TiledMma,
    mdV: cute.Tensor,
    sA: cute.Tensor,
    sK_persist: cute.Tensor,
    sP: cute.Tensor,
    sP_fp32: cute.Tensor,  # fp32 P buffer for WG2 to read (no precision loss)
    sLSE: cute.Tensor,
    sB_epi: cute.Tensor,
    pipeline_A: pipeline.PipelineAsync,
    pipeline_K: pipeline.PipelineAsync,
    tidx: Int32,
    tma_atom_dV: cute.CopyAtom,
    softmax_scale_log2: Float32,
    softmax_scale: Float32,
    block_info: BlockInfo,
    SeqlenInfoCls: Callable,
    TileSchedulerCls: Callable,
    qhead_per_kvhead_divmod: Optional[FastDivmodDivisor] = None,
  ):
    thr_mma_SdP = tiled_mma_SdP.get_slice(tidx)
    wg_mma_SdP = tiled_mma_SdP.get_slice(0)
    wg_mma_dV = tiled_mma_dV.get_slice(0)

    # ── SdP fragments (WG1 needs only K, not V) ──
    shape_mnk_SdP = (self.tile_m, self.tile_n, self.d_chunk)
    _, tSrA, tSrB_K = sm90_utils.partition_fragment_ABC(wg_mma_SdP, shape_mnk_SdP, sA, sK_persist, swap_AB=False)

    # ── dV GEMM fragments: dV = P^T @ dO_d_pass ──
    sPt = layout_utils.transpose_view(sP)
    sAt = layout_utils.transpose_view(sA)
    shape_mnk_dV = (self.tile_n, self.d_chunk, self.tile_m)
    acc_dV, tdVrPt, tdVrdOt = sm90_utils.partition_fragment_ABC(wg_mma_dV, shape_mnk_dV, sPt, sAt, swap_AB=False)
    mma_dV_fn = partial(gemm_w_idx, tiled_mma_dV, acc_dV, tdVrPt, tdVrdOt, swap_AB=False)

    # ── P R2S copies: ──
    #   (a) bf16 sP for WG1 dV WGMMA SS-mode (must be fp16/bf16 for WGMMA A operand)
    copy_P_r2s, _, _ = copy_utils.get_smem_store_C(
      tiled_mma_SdP,
      sP,
      tidx,
      self.arch,
      transpose=False,
      position_independent=True,
    )
    #   (b) fp32 sP store via get_smem_store_C → CopyUniversalOp atom in C-operand
    #       layout (matches partition_C(sP_fp32) addressing)
    copy_P_fp32_r2s, _, _ = copy_utils.get_smem_store_C(
      tiled_mma_SdP,
      sP_fp32,
      tidx,
      self.arch,
      transpose=False,
      position_independent=False,
    )

    # ── LSE partitioning (per-thread row mapping via MMA C partition) ──
    tLSEsLSE = layout_utils.mma_partition_C_vec(sLSE, thr_mma_SdP, expand_shape=self.tile_n, is_colvec=True)

    consumer_state_A = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, self.A_stage)
    consumer_state_K = cutlass.pipeline.make_pipeline_state(
      cutlass.pipeline.PipelineUserType.Consumer, self.K_persist_chunks
    )

    tile_scheduler = TileSchedulerCls()
    work_tile = tile_scheduler.initial_work_tile_info()

    while work_tile.is_valid_tile:
      n_block, head_idx_kv, batch_idx, _ = work_tile.tile_idx
      seqlen = SeqlenInfoCls(batch_idx)
      m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)
      process_tile = const_expr(not self.is_varlen_q) or m_block_min < m_block_max

      if process_tile:
        # K preload wait & release ONCE per work_tile (Cross-pass reuse)
        for _k in cutlass.range_constexpr(self.K_persist_chunks):
          pipeline_K.consumer_wait(consumer_state_K, pipeline_K.consumer_try_wait(consumer_state_K))
          with cute.arch.elect_one():
            pipeline_K.consumer_release(consumer_state_K)
          consumer_state_K.advance()

        for d_pass in cutlass.range_constexpr(self.num_d_passes):
          dKV_accumulate = Boolean(False)

          for q_head_offset in cutlass.range(self.qhead_per_kvhead, unroll=1):
            head_idx_q = head_idx_kv * self.qhead_per_kvhead + q_head_offset

            mask = AttentionMask(self.tile_m, self.tile_n, seqlen)
            mask_fn = partial(
              mask.apply_mask,
              batch_idx=batch_idx,
              head_idx=head_idx_q,
              n_block=n_block,
              thr_mma=thr_mma_SdP,
              mask_seqlen=True,
              mask_causal=self.is_causal,
            )

            for m_block in cutlass.range(m_block_min, m_block_max, unroll=1):
              consumer_state_A = self._mma_wg1_one_m_block(
                m_block,
                consumer_state_A,
                tiled_mma_SdP,
                tSrA,
                tSrB_K,
                shape_mnk_SdP,
                mma_dV_fn,
                copy_P_r2s,
                copy_P_fp32_r2s,
                pipeline_A,
                tLSEsLSE,
                softmax_scale_log2,
                mask_fn,
                dKV_accumulate=dKV_accumulate,
              )
              dKV_accumulate = Boolean(True)

          # Per-d_pass epilogue: WG1 writes dV[:, d_pass*d_chunk:(d_pass+1)*d_chunk]
          self.epilogue_dV_slice(
            acc_dV,
            mdV,
            sB_epi,
            seqlen,
            tma_atom_dV,
            tiled_mma_dV,
            tidx,
            n_block,
            head_idx_kv,
            batch_idx,
            d_pass,
          )

          # Per-d_pass WSWG1 sync (288): producer waits before loading
          # next d_pass's pipeline_A items into sA (union with sEpi).
          cute.arch.barrier(
            barrier_id=int(NamedBarrierBwd.WarpSchedulerWG1),
            number_of_threads=self.num_producer_threads + self.num_mma_threads,
          )

      tile_scheduler.advance_to_next_work()
      work_tile = tile_scheduler.get_current_work()

    warp_idx_final = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    if warp_idx_final == 4:  # WG1's TMA-issuing warp (dV stores)
      # Full wait (no `read=True`): wait for GMEM visibility, not just SMEM-read.
      cute.arch.cp_async_bulk_wait_group(0)

  @cute.jit
  def _mma_wg1_one_m_block(
    self,
    m_block: Int32,
    consumer_state_A: cutlass.pipeline.PipelineState,
    tiled_mma_SdP: cute.TiledMma,
    tSrA: cute.Tensor,
    tSrB_K: cute.Tensor,
    shape_mnk_SdP: cute.Shape,
    mma_dV_fn: Callable,
    copy_P_r2s: Callable,
    copy_P_fp32_r2s: Callable,  # fp32 sP_fp32 store via CopyUniversalOp atom in C-operand layout
    pipeline_A: pipeline.PipelineAsync,
    tLSEsLSE: cute.Tensor,
    softmax_scale_log2: Float32,
    mask_fn: Callable,
    dKV_accumulate: Boolean = True,
  ):
    p_stage = Int32(0)  # sP single-buffered

    # ═══ Phase 1: S = ΣQ_d @ K_d^T (num_d_inner WGMMA, real consume Q) ═══
    acc_S = cute.make_rmem_tensor(tiled_mma_SdP.partition_shape_C(shape_mnk_SdP[:2]), Float32)
    for d_inner in cutlass.range_constexpr(self.num_d_inner):
      pipeline_A.consumer_wait(consumer_state_A, pipeline_A.consumer_try_wait(consumer_state_A))
      gemm_w_idx(
        tiled_mma_SdP,
        acc_S,
        tSrA,
        tSrB_K,
        zero_init=(d_inner == 0),
        A_idx=consumer_state_A.index,
        B_idx=d_inner,
        wg_wait=0,
      )
      if cutlass.const_expr(d_inner == self.num_d_inner - 1):
        # LSE piggybacked on last Q chunk — read before release.
        tLSErLSE = copy_utils.load_s2r(tLSEsLSE[None, consumer_state_A.index])
      with cute.arch.elect_one():
        pipeline_A.consumer_release(consumer_state_A)
      consumer_state_A.advance()

    # ═══ Empty release: dO0..dO_{num_d_inner-1} (consumed by WG2) ═══
    for _ in cutlass.range_constexpr(self.num_d_inner):
      pipeline_A.consumer_wait(consumer_state_A, pipeline_A.consumer_try_wait(consumer_state_A))
      with cute.arch.elect_one():
        pipeline_A.consumer_release(consumer_state_A)
      consumer_state_A.advance()

    # ═══ Phase 2: mask + softmax → P (fp32 in rmem, in-place on acc_S) ═══
    mask_fn(acc_S, m_block=m_block)
    acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S, transpose=False)
    for r in cutlass.range_constexpr(cute.size(acc_S_mn, mode=[0])):
      lse_val = tLSErLSE[r]
      for c in cutlass.range(cute.size(acc_S_mn, mode=[1]), unroll_full=True):
        acc_S_mn[r, c] = cute.math.exp2(acc_S_mn[r, c] * softmax_scale_log2 - lse_val, fastmath=True)

    # bf16 P (frgA layout) for STSM into sP (WG1's dV WGMMA A operand).
    tdVrP = utils.cvt_f16(layout_utils.reshape_acc_to_frgA(acc_S), self.dtype)

    # ═══ Cross-WG handshake: wait WG2 done with prior sP, then write both sP buffers ═══
    cute.arch.barrier(
      barrier_id=int(NamedBarrierBwd.PEmpty),
      number_of_threads=self.num_mma_threads,  # 256
    )
    # (a) STSM bf16 P → sP (for WG1's own dV WGMMA SS-mode read)
    copy_P_r2s(tdVrP, dst_idx=p_stage)
    # (b) Universal-op store fp32 acc_S → sP_fp32 in C-operand layout (Step 14).
    copy_P_fp32_r2s(acc_S, dst_idx=p_stage)
    cute.arch.fence_view_async_shared()  # orders both STSM and direct stores
    cute.arch.sync_warp()
    cute.arch.barrier_arrive(
      barrier_id=int(NamedBarrierBwd.PFull),
      number_of_threads=self.num_mma_threads,
    )

    # ═══ Phase 5: dV += P^T @ dO_d_pass (sP self-read via SMEM) ═══
    pipeline_A.consumer_wait(consumer_state_A, pipeline_A.consumer_try_wait(consumer_state_A))
    smem_idx_dO_pass = consumer_state_A.index
    mma_dV_fn(
      A_idx=p_stage,
      B_idx=smem_idx_dO_pass,
      zero_init=not dKV_accumulate,
      wg_wait=0,
    )
    with cute.arch.elect_one():
      pipeline_A.consumer_release(consumer_state_A)
    consumer_state_A.advance()

    # ═══ Empty release Q_d_pass (consumed by WG2 P6) ═══
    pipeline_A.consumer_wait(consumer_state_A, pipeline_A.consumer_try_wait(consumer_state_A))
    with cute.arch.elect_one():
      pipeline_A.consumer_release(consumer_state_A)
    consumer_state_A.advance()

    return consumer_state_A

  # ════════════════════════════════════════════════════════════════════
  # WG2 = dP (P3) + sync PFull → dS → sdS → arrive PEmpty (P4) + dK (P6)
  # ════════════════════════════════════════════════════════════════════
  @cute.jit
  def mma_wg2(
    self,
    tiled_mma_SdP: cute.TiledMma,
    tiled_mma_dK: cute.TiledMma,
    mdK: cute.Tensor,
    sA: cute.Tensor,
    sV_persist: cute.Tensor,
    sP_fp32: cute.Tensor,  # fp32 P buffer (replaces bf16 sP; WG2 no longer needs bf16)
    sdS: cute.Tensor,
    sdPsum: cute.Tensor,
    sB_epi: cute.Tensor,
    pipeline_A: pipeline.PipelineAsync,
    pipeline_V: pipeline.PipelineAsync,
    tidx: Int32,
    tma_atom_dK: cute.CopyAtom,
    softmax_scale_log2: Float32,
    softmax_scale: Float32,
    block_info: BlockInfo,
    SeqlenInfoCls: Callable,
    TileSchedulerCls: Callable,
    qhead_per_kvhead_divmod: Optional[FastDivmodDivisor] = None,
  ):
    thr_mma_SdP = tiled_mma_SdP.get_slice(tidx)
    wg_mma_SdP = tiled_mma_SdP.get_slice(0)
    wg_mma_dK = tiled_mma_dK.get_slice(0)

    # ── SdP fragments (WG2 needs only V, not K) ──
    shape_mnk_SdP = (self.tile_m, self.tile_n, self.d_chunk)
    _, tSrA, tSrB_V = sm90_utils.partition_fragment_ABC(wg_mma_SdP, shape_mnk_SdP, sA, sV_persist, swap_AB=False)

    # ── dK GEMM fragments: dK = dS^T @ Q_d_pass ──
    sdSt = layout_utils.transpose_view(sdS)
    sAt = layout_utils.transpose_view(sA)
    shape_mnk_dK = (self.tile_n, self.d_chunk, self.tile_m)
    acc_dK, tdKrdSt, tdKrQt = sm90_utils.partition_fragment_ABC(wg_mma_dK, shape_mnk_dK, sdSt, sAt, swap_AB=False)
    mma_dK_fn = partial(gemm_w_idx, tiled_mma_dK, acc_dK, tdKrdSt, tdKrQt, swap_AB=False)

    # ── dS R2S copy (WG2-internal: STSM rmem dS → sdS[0]) ──
    copy_dS_r2s, _, _ = copy_utils.get_smem_store_C(
      tiled_mma_SdP,
      sdS,
      tidx,
      self.arch,
      transpose=False,
      position_independent=True,
    )

    # ── sP_fp32 s2r partition (WG2 reads fp32 P from sP_fp32, NOT bf16 sP).
    tSsP_fp32_partition = thr_mma_SdP.partition_C(sP_fp32)  # ((atom), MMA_M, MMA_N, stage)

    # ── dPsum partitioning (per-thread row mapping) ──
    tLSEsdPsum = layout_utils.mma_partition_C_vec(sdPsum, thr_mma_SdP, expand_shape=self.tile_n, is_colvec=True)

    #   pre-arrive PEmpty so WG1's first barrier(PEmpty, 256) finds
    #   128 arrivals from WG2 and only waits for WG1's own 128.
    cute.arch.barrier_arrive(
      barrier_id=int(NamedBarrierBwd.PEmpty),
      number_of_threads=self.num_mma_threads,  # 256
    )

    consumer_state_A = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, self.A_stage)
    consumer_state_V = cutlass.pipeline.make_pipeline_state(
      cutlass.pipeline.PipelineUserType.Consumer, self.V_persist_chunks
    )

    tile_scheduler = TileSchedulerCls()
    work_tile = tile_scheduler.initial_work_tile_info()

    while work_tile.is_valid_tile:
      n_block, head_idx_kv, batch_idx, _ = work_tile.tile_idx
      seqlen = SeqlenInfoCls(batch_idx)
      m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)
      process_tile = const_expr(not self.is_varlen_q) or m_block_min < m_block_max

      if process_tile:
        # ★ V preload wait & release ONCE per work_tile (跨 d_pass 复用)
        for _v in cutlass.range_constexpr(self.V_persist_chunks):
          pipeline_V.consumer_wait(consumer_state_V, pipeline_V.consumer_try_wait(consumer_state_V))
          with cute.arch.elect_one():
            pipeline_V.consumer_release(consumer_state_V)
          consumer_state_V.advance()

        # ★ V tail zero (WG2 internal, 128 thread; ONCE per work_tile)
        self.zero_v_tail_smem_wg2(sV_persist=sV_persist, seqlen=seqlen, n_block=n_block, tidx=tidx)

        for d_pass in cutlass.range_constexpr(self.num_d_passes):
          dKV_accumulate = Boolean(False)

          for q_head_offset in cutlass.range(self.qhead_per_kvhead, unroll=1):
            head_idx_q = head_idx_kv * self.qhead_per_kvhead + q_head_offset

            mask = AttentionMask(self.tile_m, self.tile_n, seqlen)
            mask_fn = partial(
              mask.apply_mask,
              batch_idx=batch_idx,
              head_idx=head_idx_q,
              n_block=n_block,
              thr_mma=thr_mma_SdP,
              mask_seqlen=True,
              mask_causal=self.is_causal,
            )

            for m_block in cutlass.range(m_block_min, m_block_max, unroll=1):
              consumer_state_A = self._mma_wg2_one_m_block(
                m_block,
                consumer_state_A,
                tiled_mma_SdP,
                tSrA,
                tSrB_V,
                shape_mnk_SdP,
                mma_dK_fn,
                copy_dS_r2s,
                tSsP_fp32_partition,
                pipeline_A,
                tLSEsdPsum,
                softmax_scale,
                mask_fn,
                dKV_accumulate=dKV_accumulate,
              )
              dKV_accumulate = Boolean(True)

          # Per-d_pass epilogue: WG2 writes dK[:, d_pass*d_chunk:(d_pass+1)*d_chunk]
          # (serialized with WG1 dV via Epilogue NamedBarrier inside epi function).
          self.epilogue_dK_slice(
            acc_dK,
            mdK,
            sB_epi,
            seqlen,
            tma_atom_dK,
            tiled_mma_dK,
            tidx,
            n_block,
            head_idx_kv,
            batch_idx,
            d_pass,
          )

          # ★ Per-d_pass WSWG1 sync (288): mirror mma_wg1; producer waits.
          cute.arch.barrier(
            barrier_id=int(NamedBarrierBwd.WarpSchedulerWG1),
            number_of_threads=self.num_producer_threads + self.num_mma_threads,
          )

        # No polite-close on PEmpty: per-m_block (WG1 barrier + WG2 arrive = 256)

      tile_scheduler.advance_to_next_work()
      work_tile = tile_scheduler.get_current_work()

    warp_idx_final = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    if warp_idx_final == 8:  # WG2's TMA-issuing warp (dK stores)
      cute.arch.cp_async_bulk_wait_group(0)

  @cute.jit
  def _mma_wg2_one_m_block(
    self,
    m_block: Int32,
    consumer_state_A: cutlass.pipeline.PipelineState,
    tiled_mma_SdP: cute.TiledMma,
    tSrA: cute.Tensor,
    tSrB_V: cute.Tensor,
    shape_mnk_SdP: cute.Shape,
    mma_dK_fn: Callable,
    copy_dS_r2s: Callable,
    tSsP_fp32_partition: cute.Tensor,  # thr_mma_SdP.partition_C(sP_fp32) — fp32 per-thread C-acc view
    pipeline_A: pipeline.PipelineAsync,
    tLSEsdPsum: cute.Tensor,
    softmax_scale: Float32,
    mask_fn: Callable,
    dKV_accumulate: Boolean = True,
  ):
    p_stage = Int32(0)  # sP/sdS single-buffered

    # ═══ Empty release: Q0..Q_{num_d_inner-1} (consumed by WG1) ═══
    for _ in cutlass.range_constexpr(self.num_d_inner):
      pipeline_A.consumer_wait(consumer_state_A, pipeline_A.consumer_try_wait(consumer_state_A))
      with cute.arch.elect_one():
        pipeline_A.consumer_release(consumer_state_A)
      consumer_state_A.advance()

    # ═══ Phase 3  all d_inner iters sync; release immediately ═══
    acc_dP = cute.make_rmem_tensor(tiled_mma_SdP.partition_shape_C(shape_mnk_SdP[:2]), Float32)
    for d_inner in cutlass.range_constexpr(self.num_d_inner):
      pipeline_A.consumer_wait(consumer_state_A, pipeline_A.consumer_try_wait(consumer_state_A))
      gemm_w_idx(
        tiled_mma_SdP,
        acc_dP,
        tSrA,
        tSrB_V,
        zero_init=(d_inner == 0),
        A_idx=consumer_state_A.index,
        B_idx=d_inner,
        wg_wait=0,
      )
      if cutlass.const_expr(d_inner == self.num_d_inner - 1):
        # dPsum piggybacked on last dO chunk — read before release.
        tLSErdPsum = copy_utils.load_s2r(tLSEsdPsum[None, consumer_state_A.index])
      with cute.arch.elect_one():
        pipeline_A.consumer_release(consumer_state_A)
      consumer_state_A.advance()

    # ═══ Phase 4: barrier(PFull) → load fp32 P → dS compute ═══
    cute.arch.barrier(
      barrier_id=int(NamedBarrierBwd.PFull),
      number_of_threads=self.num_mma_threads,  # 256
    )
    # Step 14: read fp32 P from sP_fp32 — no precision loss.
    tdSrP_fp32 = copy_utils.load_s2r(tSsP_fp32_partition[None, None, None, p_stage])

    # dS = P * (dP - dpsum) * scale — acc_dP from sync WGMMAs above, P from LDS.
    tdSrP_mn = layout_utils.reshape_acc_to_mn(tdSrP_fp32, transpose=False)
    acc_dP_mn = layout_utils.reshape_acc_to_mn(acc_dP, transpose=False)
    for r in cutlass.range_constexpr(cute.size(acc_dP_mn, mode=[0])):
      dpsum_val = tLSErdPsum[r]
      for c in cutlass.range(cute.size(acc_dP_mn, mode=[1]), unroll_full=True):
        acc_dP_mn[r, c] = (tdSrP_mn[r, c] * (acc_dP_mn[r, c] - dpsum_val) * softmax_scale)

    # Convert dS (now in acc_dP rmem, fp32) → fp16 frgA → STSM into sdS[0].
    tdKrdS = utils.cvt_f16(layout_utils.reshape_acc_to_frgA(acc_dP), self.dtype)
    copy_dS_r2s(tdKrdS, dst_idx=p_stage)
    cute.arch.fence_view_async_shared()
    # WG2-internal sync (128-thread): ensure all WG2 STSM commits before P6 reads sdS.
    cute.arch.barrier(
      barrier_id=int(NamedBarrierBwd.dSLocal),
      number_of_threads=self.num_threads_per_warp_group,  # 128 (WG2 only)
    )
    # Tell WG1 the sP slot is empty (P5 inside WG1 already self-read it).
    cute.arch.barrier_arrive(
      barrier_id=int(NamedBarrierBwd.PEmpty),
      number_of_threads=self.num_mma_threads,
    )

    # ═══ Empty release: dO_d_pass (consumed by WG1 P5) ═══
    pipeline_A.consumer_wait(consumer_state_A, pipeline_A.consumer_try_wait(consumer_state_A))
    with cute.arch.elect_one():
      pipeline_A.consumer_release(consumer_state_A)
    consumer_state_A.advance()

    # ═══ Phase 6: dK += dS^T @ Q_d_pass ═══
    pipeline_A.consumer_wait(consumer_state_A, pipeline_A.consumer_try_wait(consumer_state_A))
    smem_idx_Q_pass = consumer_state_A.index
    mma_dK_fn(
      A_idx=p_stage,
      B_idx=smem_idx_Q_pass,
      zero_init=not dKV_accumulate,
      wg_wait=0,
    )
    with cute.arch.elect_one():
      pipeline_A.consumer_release(consumer_state_A)
    consumer_state_A.advance()

    return consumer_state_A

  @cute.jit
  def zero_v_tail_smem_wg2(
    self,
    sV_persist: cute.Tensor,
    seqlen: SeqlenInfoQK,
    n_block: Int32,
    tidx: Int32,  # WG-local tidx ∈ [0, 128)
  ):
    valid_rows = seqlen.seqlen_k - n_block * self.tile_n
    if valid_rows < self.tile_n:
      tail_elems = (self.tile_n - valid_rows) * self.d_chunk
      for d_inner in cutlass.range_constexpr(self.num_d_inner):
        for linear_idx in cutlass.range(tidx, tail_elems, self.num_threads_per_warp_group, unroll=1):
          row_offset = linear_idx // self.d_chunk
          col = linear_idx - row_offset * self.d_chunk
          sV_persist[valid_rows + row_offset, col, d_inner] = sV_persist.element_type(0.0)
      cute.arch.fence_view_async_shared()
      cute.arch.barrier(
        barrier_id=int(NamedBarrierBwd.VTailZero),
        number_of_threads=self.num_threads_per_warp_group,  # 128 (WG2 only)
      )

  # ════════════════════════════════════════════════════════════════════
  # Epilogue split into dV (WG1) and dK (WG2), serialized via
  # Epilogue NamedBarrier 3-step handshake (SYNC A → SYNC B → SYNC C).
  # Shared sEpi physical SMEM is unioned with sA; lifetime switch is
  # protected by entry SYNC A (256-thread) after fence_view_async_shared.
  # ════════════════════════════════════════════════════════════════════
  @cute.jit
  def epilogue_dV_slice(
    self,
    acc_dV: cute.Tensor,
    mdV: cute.Tensor,
    sEpi: cute.Tensor,
    seqlen: SeqlenInfoQK,
    tma_atom_dV: cute.CopyAtom,
    tiled_mma_dV: cute.TiledMma,
    tidx: Int32,  # WG-local ∈ [0, 128)
    n_block: Int32,
    head_idx: Int32,
    batch_idx: Int32,
    d_pass: Int32,
  ):
    """Called by WG1 only. Writes dV[:, d_pass*d_chunk:(d_pass+1)*d_chunk]."""
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    # WG1 occupies warp 4..7 (CTA-absolute). warp_idx == 4 issues the TMA store.

    # Build dV gmem slice for this (n_block, d_pass).
    mdV_cur = seqlen.offset_batch_K(mdV, batch_idx, dim=3, ragged=self.varlen_k)[None, None, head_idx]
    gdV = cute.local_tile(mdV_cur, (self.tile_n, self.d_chunk), (n_block, d_pass))
    store_dV, _, _ = copy_utils.tma_get_copy_fn(tma_atom_dV, 0, cute.make_layout(1), sEpi, gdV, single_stage=True)
    copy_dV_r2s, _, _ = copy_utils.get_smem_store_C(
      tiled_mma_dV,
      sEpi,
      tidx,
      self.arch,
      transpose=False,
      position_independent=True,
    )

    # ═══ SYNC A (entry): 256-thread cross-WG barrier ═══
    # After this, sA mainloop is done; safe to overwrite sEpi (= sA union slot).
    # fence_view_async_shared ensures all prior async-shared writes (pipeline_A
    # consumer releases) are globally visible before sEpi STSM.
    cute.arch.fence_view_async_shared()
    cute.arch.barrier(
      barrier_id=int(NamedBarrierBwd.Epilogue),
      number_of_threads=self.num_mma_threads,
    )

    # WG1 R2S: acc_dV (fp32) → sEpi (fp16) via STSM.
    copy_dV_r2s(acc_dV, dst_idx=None)
    cute.arch.fence_view_async_shared()

    # WG1-internal 128-thread fence (reuse WarpSchedulerWG2 id=3 for this).
    cute.arch.barrier(
      barrier_id=int(NamedBarrierBwd.WarpSchedulerWG2),
      number_of_threads=self.num_threads_per_warp_group,
    )
    if warp_idx == 4:
      store_dV()
      cute.arch.cp_async_bulk_commit_group()
      cute.arch.cp_async_bulk_wait_group(0, read=True)

    # ═══ SYNC B (WG1 dV TMA done): WG2 may now overwrite sEpi for dK ═══
    cute.arch.barrier(
      barrier_id=int(NamedBarrierBwd.Epilogue),
      number_of_threads=self.num_mma_threads,
    )

    # ═══ SYNC C (WG2 dK done): WG1 waits for WG2 to finish before exiting ═══
    cute.arch.barrier(
      barrier_id=int(NamedBarrierBwd.Epilogue),
      number_of_threads=self.num_mma_threads,
    )

  @cute.jit
  def epilogue_dK_slice(
    self,
    acc_dK: cute.Tensor,
    mdK: cute.Tensor,
    sEpi: cute.Tensor,
    seqlen: SeqlenInfoQK,
    tma_atom_dK: cute.CopyAtom,
    tiled_mma_dK: cute.TiledMma,
    tidx: Int32,  # WG-local ∈ [0, 128)
    n_block: Int32,
    head_idx: Int32,
    batch_idx: Int32,
    d_pass: Int32,
  ):
    """Called by WG2 only. Writes dK[:, d_pass*d_chunk:(d_pass+1)*d_chunk]."""
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    # WG2 occupies warp 8..11 (CTA-absolute). warp_idx == 8 issues TMA store.

    mdK_cur = seqlen.offset_batch_K(mdK, batch_idx, dim=3, ragged=self.varlen_k)[None, None, head_idx]
    gdK = cute.local_tile(mdK_cur, (self.tile_n, self.d_chunk), (n_block, d_pass))
    store_dK, _, _ = copy_utils.tma_get_copy_fn(tma_atom_dK, 0, cute.make_layout(1), sEpi, gdK, single_stage=True)
    copy_dK_r2s, _, _ = copy_utils.get_smem_store_C(
      tiled_mma_dK,
      sEpi,
      tidx,
      self.arch,
      transpose=False,
      position_independent=True,
    )

    # ═══ SYNC A (entry): symmetric with WG1 ═══
    cute.arch.fence_view_async_shared()
    cute.arch.barrier(
      barrier_id=int(NamedBarrierBwd.Epilogue),
      number_of_threads=self.num_mma_threads,
    )

    # ═══ SYNC B: wait for WG1 dV TMA done (sEpi freed) ═══
    cute.arch.barrier(
      barrier_id=int(NamedBarrierBwd.Epilogue),
      number_of_threads=self.num_mma_threads,
    )

    # WG2 R2S: acc_dK (fp32) → sEpi (fp16).
    copy_dK_r2s(acc_dK, dst_idx=None)
    cute.arch.fence_view_async_shared()

    # WG2-internal 128-thread fence (reuse WarpSchedulerWG3 id=4).
    cute.arch.barrier(
      barrier_id=int(NamedBarrierBwd.WarpSchedulerWG3),
      number_of_threads=self.num_threads_per_warp_group,
    )
    if warp_idx == 8:
      store_dK()
      cute.arch.cp_async_bulk_commit_group()
      cute.arch.cp_async_bulk_wait_group(0, read=True)

    # ═══ SYNC C: signal WG1 that WG2's dK TMA is complete ═══
    cute.arch.barrier(
      barrier_id=int(NamedBarrierBwd.Epilogue),
      number_of_threads=self.num_mma_threads,
    )

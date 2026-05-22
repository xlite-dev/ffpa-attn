# Copyright (c) DefTruth, qyjdef@163.com
# Copyright (c) Butterfingrz，13524387014@163.com
#
# The idea of splitting the backward pass into a separate dKdV kernel is
# inspired by
# https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/sm100_hd256_2cta_fmha_backward_dkdvkernel.py
# The core implementation below is written from scratch for SM90 and follows
# the SplitD design from the ffpa-attn repo.
#
# SM90 Backward dKdV Kernel with 2-pass D-split for dense D<=384.
#
# Architecture:
#   - 3 WGs: 1 producer WG (warp 0 issues TMA), WG1 (S/softmax/dV), WG2 (dP/dS/dK)
#   - tile_m=64, tile_n=64, d_chunk=256, d_chunk_tail=128, num_d_passes=2
#   - K/V persistent SMEM: full 0:256 plus true-tail 256:384 pre-loaded once per work_tile
#   - Q/dO streaming via pipeline_A (2 stages)
#   - Per d_pass, 6 phases per (n_block, m_block). Pass 0 stores D 0:256;
#     pass 1 uses 128-wide tail MMA/TMA and stores only logical D 256:384.
#       Phase 1: S = Q_full @ K_full^T + Q_tail @ K_tail^T
#       Phase 2: P = exp2(S * scale_log2 - LSE)
#       Phase 3: dP = dO_full @ V_full^T + dO_tail @ V_tail^T
#       Phase 4: dS = P * (dP - dPsum)
#       Phase 5: dV += P^T @ dO_d_pass  (via SMEM)
#       Phase 6: dK += dS^T @ Q_d_pass  (via SMEM)
#   - mma_dkv_is_rs = False (P/dS through SMEM for simplicity)
#
# D384 true-tail design notes:
#   - The original D384-aware path used two 256-wide chunks, so pass 1 covered
#     logical D 256:384 plus padded D 384:512. That wasted tail TMA traffic and
#     WGMMA work. The current path keeps pass 0 as a full 256-wide tile and makes
#     pass 1 a real 128-wide tail tile.
#   - TMA atoms and WGMMA fragments are shape-specialized in CuTeDSL, so the full
#     and tail paths need separate atoms, layouts, MMA objects, accumulators, and
#     epilogue store views. The tail path still uses gmem tile coordinate 2 because
#     128-wide TMA tiles address D offsets in 128-wide units; coordinate 2 is D=256.
#   - Pipeline tx_count uses the smaller tail transfer as the base count. Full
#     transfers add the extra 128-wide bytes via extra_tx_count so the same pipeline
#     state can carry both full and tail slots without over-counting tail arrivals.
#   - dV/dK epilogues issue TMA store from a single warp, but the read wait must be
#     executed by the whole warpgroup. Keeping cp_async_bulk_wait_group inside only
#     the issuing warp can leave other lanes racing the cross-WG Epilogue barrier
#     and was observed to hang backward on H800.

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


class FFPAAttnBwdDKDVSm90SplitDD384:
  """SM90 backward dK/dV kernel for dense D<=384.

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
    head_dim_v = head_dim_v if head_dim_v is not None else head_dim
    if head_dim != head_dim_v or not 256 < head_dim <= 384:
      raise ValueError(
        f"D384 dK/dV kernel requires q/k head_dim == v head_dim_v and "
        f"256 < head_dim <= 384, got {head_dim} and {head_dim_v}"
      )
    self.tile_hdim = 384
    self.tile_hdimv = 384
    self.check_hdim_oob = head_dim != self.tile_hdim
    self.check_hdim_v_oob = head_dim_v != self.tile_hdimv

    self.is_causal = is_causal
    self.qhead_per_kvhead = qhead_per_kvhead
    self.tile_m = tile_m
    self.tile_n = tile_n
    self.use_pdl = False
    self.qk_acc_dtype = Float32
    self.buffer_align_bytes = 1024

    # SplitD parameters for physical D=384.
    self.d_chunk = 256  # pass 0 output slice width for dK/dV
    self.d_chunk_tail = 128  # pass 1 true-tail width, D 256:384
    self.num_d_passes = 2  # pass 0 handles 0:256, pass 1 handles 256:384
    self.num_d_inner = 2  # full + tail reduction chunks
    assert self.d_chunk <= 256

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
    self.num_producer_regs = 56

    # ── Pipeline stages──
    # A_stage 3→2: d_chunk=256 doubles per-stage size; drop one stage to fit SMEM ≤ 228KB.
    self.A_stage = 2
    # sP/sdS single-buffered; PFull/PEmpty (256-thread named barrier) serializes cross-WG handoff.
    self.PdS_stage = 1

    # ── K/V persistence: preload both physical D chunks once per work_tile ──
    self.K_persist_chunks = self.num_d_inner
    self.V_persist_chunks = self.num_d_inner

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
    sA_tail_base_layout = sm90_utils.make_smem_layout(
      self.dtype,
      LayoutEnum.ROW_MAJOR,
      (self.tile_m, self.d_chunk_tail),
      stage=self.A_stage,
      major_mode_size=self.d_chunk_tail,
    )
    self.sA_tail_layout = cute.make_composed_layout(
      sA_tail_base_layout.inner,
      sA_tail_base_layout.offset,
      cute.make_layout(
        sA_tail_base_layout.outer.shape,
        stride=(
          sA_tail_base_layout.outer.stride[0],
          sA_tail_base_layout.outer.stride[1],
          (0, self.tile_m * self.d_chunk),
        ),
      ),
    )
    # Tail views reuse the full MemRange. Keeping the full-stage stride avoids
    # overlap between pipeline stages while exposing a 128-wide TMA/MMA tile.
    # sK_persist: (tile_n, d_chunk) × num_d_inner — persistent K across m_blocks
    self.sK_persist_layout = sm90_utils.make_smem_layout(
      self.dtype,
      LayoutEnum.ROW_MAJOR,
      (self.tile_n, self.d_chunk),
      stage=self.K_persist_chunks,
    )
    sK_persist_tail_base_layout = sm90_utils.make_smem_layout(
      self.dtype,
      LayoutEnum.ROW_MAJOR,
      (self.tile_n, self.d_chunk_tail),
      stage=self.K_persist_chunks,
      major_mode_size=self.d_chunk_tail,
    )
    self.sK_persist_tail_layout = cute.make_composed_layout(
      sK_persist_tail_base_layout.inner,
      sK_persist_tail_base_layout.offset,
      cute.make_layout(
        sK_persist_tail_base_layout.outer.shape,
        stride=(
          sK_persist_tail_base_layout.outer.stride[0],
          sK_persist_tail_base_layout.outer.stride[1],
          (0, self.tile_n * self.d_chunk),
        ),
      ),
    )
    # K/V tail layouts map stage 1 onto the physical bytes after the full 0:256
    # chunk, so persistent preload can share the original two-stage storage.
    # sV_persist: (tile_n, d_chunk) × num_d_inner — persistent V across m_blocks
    self.sV_persist_layout = sm90_utils.make_smem_layout(
      self.dtype,
      LayoutEnum.ROW_MAJOR,
      (self.tile_n, self.d_chunk),
      stage=self.V_persist_chunks,
    )
    sV_persist_tail_base_layout = sm90_utils.make_smem_layout(
      self.dtype,
      LayoutEnum.ROW_MAJOR,
      (self.tile_n, self.d_chunk_tail),
      stage=self.V_persist_chunks,
      major_mode_size=self.d_chunk_tail,
    )
    self.sV_persist_tail_layout = cute.make_composed_layout(
      sV_persist_tail_base_layout.inner,
      sV_persist_tail_base_layout.offset,
      cute.make_layout(
        sV_persist_tail_base_layout.outer.shape,
        stride=(
          sV_persist_tail_base_layout.outer.stride[0],
          sV_persist_tail_base_layout.outer.stride[1],
          (0, self.tile_n * self.d_chunk),
        ),
      ),
    )
    # sB_epi_layout: per-chunk layout for epilogue TMA store (same as K/V per-chunk)
    self.sB_epi_layout = cute.select(self.sK_persist_layout, mode=[0, 1])
    self.sB_epi_tail_layout = cute.select(
      self.sK_persist_tail_layout, mode=[0, 1]
    )
    # sP: (tile_m, tile_n) — P for dV GEMM A operand
    wg_n_SdP = self.num_wg_mma // self.AtomLayoutMSdP
    wg_n_dKV = self.AtomLayoutNdKV
    self.sPdS_layout = sm90_utils.make_smem_layout(
      self.dtype,
      LayoutEnum.ROW_MAJOR,
      (self.tile_m, self.tile_n),
      stage=self.PdS_stage,
      major_mode_size=math.gcd(
        self.tile_n // wg_n_SdP, self.tile_n // wg_n_dKV
      ),
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
      major_mode_size=math.gcd(
        self.tile_n // wg_n_SdP, self.tile_n // wg_n_dKV
      ),
    )

  def _get_tiled_mma(self):
    # ── SdP: S = Q @ K^T, dP = dO @ V^T ──
    # shape_mnk: (tile_m, tile_n, d_chunk) = (64, 64, 256)
    atom_layout_SdP = (
      self.AtomLayoutMSdP, self.num_wg_mma // self.AtomLayoutMSdP, 1
    )
    tiler_mn_SdP = (
      self.tile_m // atom_layout_SdP[0], self.tile_n // atom_layout_SdP[1]
    )
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
    # shape_mnk: (tile_n, d_chunk, tile_m) = (64, 256, 64)
    atom_layout_dKV = (
      self.AtomLayoutNdKV, self.num_wg_mma // self.AtomLayoutNdKV, 1
    )
    # dV: M=tile_n, N=d_chunk
    tiler_mn_dV = (
      self.tile_n // atom_layout_dKV[0], self.d_chunk // atom_layout_dKV[1]
    )
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
    tiler_mn_dK = (
      self.tile_n // atom_layout_dKV[0], self.d_chunk // atom_layout_dKV[1]
    )
    tiled_mma_dK = sm90_utils_basic.make_trivial_tiled_mma(
      self.dtype,
      self.dtype,
      warpgroup.OperandMajorMode.MN,
      warpgroup.OperandMajorMode.MN,
      Float32,
      atom_layout_mnk=atom_layout_dKV,
      tiler_mn=tiler_mn_dK,
    )
    tiler_mn_dV_tail = (
      self.tile_n // atom_layout_dKV[0], self.d_chunk_tail // atom_layout_dKV[1]
    )
    tiled_mma_dV_tail = sm90_utils_basic.make_trivial_tiled_mma(
      self.dtype,
      self.dtype,
      warpgroup.OperandMajorMode.MN,
      warpgroup.OperandMajorMode.MN,
      Float32,
      atom_layout_mnk=atom_layout_dKV,
      tiler_mn=tiler_mn_dV_tail,
    )
    tiler_mn_dK_tail = (
      self.tile_n // atom_layout_dKV[0], self.d_chunk_tail // atom_layout_dKV[1]
    )
    tiled_mma_dK_tail = sm90_utils_basic.make_trivial_tiled_mma(
      self.dtype,
      self.dtype,
      warpgroup.OperandMajorMode.MN,
      warpgroup.OperandMajorMode.MN,
      Float32,
      atom_layout_mnk=atom_layout_dKV,
      tiler_mn=tiler_mn_dK_tail,
    )
    return tiled_mma_SdP, tiled_mma_dK, tiled_mma_dV, tiled_mma_dK_tail, tiled_mma_dV_tail

  def _get_shared_storage_cls(self):
    sA_struct = cute.struct.Align[
      cute.struct.MemRange[self.dtype, cute.cosize(self.sA_layout)],
      self.buffer_align_bytes,
    ]
    sK_persist_struct = cute.struct.Align[
      cute.struct.MemRange[self.dtype,
                           cute.cosize(self.sK_persist_layout)],
      self.buffer_align_bytes,
    ]
    sV_persist_struct = cute.struct.Align[
      cute.struct.MemRange[self.dtype,
                           cute.cosize(self.sV_persist_layout)],
      self.buffer_align_bytes,
    ]
    # Dedicated single-stage epilogue buffer for TMA dK/dV store.
    sEpi_struct = cute.struct.Align[
      cute.struct.MemRange[self.dtype,
                           cute.cosize(self.sB_epi_layout)],
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
      sP: cute.struct.Align[cute.struct.MemRange[self.dtype,
                                                 cute.cosize(self.sPdS_layout)],
                            1024]
      # Bf16-roundtrip-free P channel: WG1 writes fp32 P here in P2,
      # WG2 reads it in P4 for dS computation. Avoids fp32→bf16→fp32 precision loss.
      sP_fp32: cute.struct.Align[cute.struct.MemRange[
        Float32, cute.cosize(self.sP_fp32_layout)], 1024]
      sdS: cute.struct.Align[cute.struct.MemRange[
        self.dtype, cute.cosize(self.sPdS_layout)], 1024]

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
    mQ, mK, mV, mdO, mdK, mdV = [
      assume_tensor_aligned(t) for t in (mQ, mK, mV, mdO, mdK, mdV)
    ]
    mLSE, mdPsum = [assume_tensor_aligned(t) for t in (mLSE, mdPsum)]

    # Transpose: (b, s, n, h) → (s, h, n, b)
    def _qkv_transpose(t):
      return layout_utils.select(
        t, [1, 3, 2, 0] if cute.rank(t.shape) == 4 else [0, 2, 1]
      )

    mQ, mK, mV, mdO, mdK, mdV = [
      _qkv_transpose(t) for t in (mQ, mK, mV, mdO, mdK, mdV)
    ]
    # Stats: (b, n, s) → (s, n, b)
    LSE_transpose = [2, 1, 0] if cute.rank(mLSE.shape) == 3 else [1, 0]
    mLSE = layout_utils.select(mLSE, LSE_transpose)
    mdPsum = layout_utils.select(mdPsum, LSE_transpose)

    (
      tiled_mma_SdP,
      tiled_mma_dK,
      tiled_mma_dV,
      tiled_mma_dK_tail,
      tiled_mma_dV_tail,
    ) = self._get_tiled_mma()
    # num_mma_threads = 256 (WG1+WG2) is set in __init__; do NOT
    # overwrite from tiled_mma_SdP.size (= 128 per WG with num_wg_mma=1).
    # The per-WG WGMMA thread count is tracked separately if ever needed.
    self.num_mma_threads_per_wg = int(tiled_mma_SdP.size)  # = 128
    self._setup_attributes()
    SharedStorage = self._get_shared_storage_cls()

    sK_layout_sel = cute.select(self.sK_persist_layout, mode=[0, 1])
    sA_tail_layout_sel = cute.select(self.sA_tail_layout, mode=[0, 1])
    sK_tail_layout_sel = cute.select(self.sK_persist_tail_layout, mode=[0, 1])
    self.tma_copy_bytes = {
      name:
      cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1]))
      for name, mX, layout in [
        ("A", mQ, self.sA_layout),
        ("KV", mK, self.sK_persist_layout),
      ]
    }
    self.tma_copy_bytes["A_tail"] = cute.size_in_bytes(
      mQ.element_type, sA_tail_layout_sel
    )
    self.tma_copy_bytes["KV_tail"] = cute.size_in_bytes(
      mK.element_type, sK_tail_layout_sel
    )
    self.tma_copy_bytes["A_full_extra"] = (
      self.tma_copy_bytes["A"] - self.tma_copy_bytes["A_tail"]
    )
    self.tma_copy_bytes["KV_full_extra"] = (
      self.tma_copy_bytes["KV"] - self.tma_copy_bytes["KV_tail"]
    )
    # Pipelines are initialized with tail-sized tx_count; full chunks add these
    # deltas at acquire time. This keeps tail barriers from waiting for bytes
    # that are never transferred.
    self.tma_copy_bytes["LSE"] = self.tile_m * Float32.width // 8
    self.tma_copy_bytes["dPsum"] = self.tile_m * Float32.width // 8

    sA_layout_sel = cute.select(self.sA_layout, mode=[0, 1])
    gmem_tiled_copy_g2s = cpasync.CopyBulkTensorTileG2SOp()
    # Q: tile shape (tile_m, d_chunk) = (64, 256)
    tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
      gmem_tiled_copy_g2s,
      mQ,
      sA_layout_sel,
      (self.tile_m, self.d_chunk),
    )
    tma_atom_Q_tail, tma_tensor_Q_tail = cpasync.make_tiled_tma_atom(
      gmem_tiled_copy_g2s,
      mQ,
      sA_tail_layout_sel,
      (self.tile_m, self.d_chunk_tail),
    )
    # Tail TMA atoms are separate because the descriptor encodes the 128-wide
    # CTA tile shape; reusing the 256-wide atom would reintroduce padded traffic.
    # dO: tile shape (tile_m, d_chunk) = (64, 256)
    tma_atom_dO, tma_tensor_dO = cpasync.make_tiled_tma_atom(
      gmem_tiled_copy_g2s,
      mdO,
      sA_layout_sel,
      (self.tile_m, self.d_chunk),
    )
    tma_atom_dO_tail, tma_tensor_dO_tail = cpasync.make_tiled_tma_atom(
      gmem_tiled_copy_g2s,
      mdO,
      sA_tail_layout_sel,
      (self.tile_m, self.d_chunk_tail),
    )
    # K: tile shape (tile_n, d_chunk) = (64, 256), persistent in SMEM
    tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
      gmem_tiled_copy_g2s,
      mK,
      sK_layout_sel,
      (self.tile_n, self.d_chunk),
    )
    tma_atom_K_tail, tma_tensor_K_tail = cpasync.make_tiled_tma_atom(
      gmem_tiled_copy_g2s,
      mK,
      sK_tail_layout_sel,
      (self.tile_n, self.d_chunk_tail),
    )
    # V: tile shape (tile_n, d_chunk) = (64, 256), persistent in SMEM
    tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
      gmem_tiled_copy_g2s,
      mV,
      sK_layout_sel,
      (self.tile_n, self.d_chunk),
    )
    tma_atom_V_tail, tma_tensor_V_tail = cpasync.make_tiled_tma_atom(
      gmem_tiled_copy_g2s,
      mV,
      sK_tail_layout_sel,
      (self.tile_n, self.d_chunk_tail),
    )
    # dK/dV: store atoms + TMA tensors.
    self.varlen_k = mCuSeqlensK is not None
    self.is_varlen_q = mCuSeqlensQ is not None
    gmem_tiled_copy_s2g = cpasync.CopyBulkTensorTileS2GOp()
    sB_epi_sel = cute.select(self.sK_persist_layout, mode=[0, 1])
    # Varlen K: create ragged TMA tensors for dK/dV output writes
    mdK_tma = copy_utils.create_ragged_tensor_for_tma(
      mdK, ragged_dim=0, ptr_shift=True
    ) if self.varlen_k else mdK
    mdV_tma = copy_utils.create_ragged_tensor_for_tma(
      mdV, ragged_dim=0, ptr_shift=True
    ) if self.varlen_k else mdV
    tma_atom_dK, tma_tensor_dK = cpasync.make_tiled_tma_atom(
      gmem_tiled_copy_s2g,
      mdK_tma,
      sB_epi_sel,
      (self.tile_n, self.d_chunk),
    )
    tma_atom_dK_tail, tma_tensor_dK_tail = cpasync.make_tiled_tma_atom(
      gmem_tiled_copy_s2g,
      mdK_tma,
      self.sB_epi_tail_layout,
      (self.tile_n, self.d_chunk_tail),
    )
    tma_atom_dV, tma_tensor_dV = cpasync.make_tiled_tma_atom(
      gmem_tiled_copy_s2g,
      mdV_tma,
      sB_epi_sel,
      (self.tile_n, self.d_chunk),
    )
    tma_atom_dV_tail, tma_tensor_dV_tail = cpasync.make_tiled_tma_atom(
      gmem_tiled_copy_s2g,
      mdV_tma,
      self.sB_epi_tail_layout,
      (self.tile_n, self.d_chunk_tail),
    )

    # ── Tile scheduler ──
    if const_expr(mCuSeqlensK is not None):
      TileScheduler = SingleTileVarlenScheduler
    else:
      TileScheduler = SingleTileScheduler
    num_n_blocks = cute.ceil_div(cute.size(mK.shape[0]), self.tile_n)
    num_batch = cute.size(mK.shape[3]) if cute.rank(
      mK.shape
    ) == 4 else cute.size(mCuSeqlensK.shape[0] - 1)
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
      tma_tensor_Q_tail,
      tma_tensor_K,
      tma_tensor_K_tail,
      tma_tensor_V,
      tma_tensor_V_tail,
      tma_tensor_dO,
      tma_tensor_dO_tail,
      tma_tensor_dK,
      tma_tensor_dK_tail,
      tma_tensor_dV,
      tma_tensor_dV_tail,
      tma_atom_Q,
      tma_atom_Q_tail,
      tma_atom_dO,
      tma_atom_dO_tail,
      tma_atom_K,
      tma_atom_K_tail,
      tma_atom_V,
      tma_atom_V_tail,
      tma_atom_dK,
      tma_atom_dK_tail,
      tma_atom_dV,
      tma_atom_dV_tail,
      mLSE,
      mdPsum,
      mCuSeqlensQ,
      mCuSeqlensK,
      softmax_scale_log2,
      softmax_scale,
      self.sA_layout,
      self.sA_tail_layout,
      self.sK_persist_layout,
      self.sK_persist_tail_layout,
      self.sV_persist_layout,
      self.sV_persist_tail_layout,
      self.sB_epi_tail_layout,
      self.sPdS_layout,
      self.sP_fp32_layout,
      tiled_mma_SdP,
      tiled_mma_dK,
      tiled_mma_dV,
      tiled_mma_dK_tail,
      tiled_mma_dV_tail,
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
    mQ_tail: cute.Tensor,
    mK: cute.Tensor,
    mK_tail: cute.Tensor,
    mV: cute.Tensor,
    mV_tail: cute.Tensor,
    mdO: cute.Tensor,
    mdO_tail: cute.Tensor,
    mdK: cute.Tensor,
    mdK_tail: cute.Tensor,
    mdV: cute.Tensor,
    mdV_tail: cute.Tensor,
    tma_atom_Q: cute.CopyAtom,
    tma_atom_Q_tail: cute.CopyAtom,
    tma_atom_dO: cute.CopyAtom,
    tma_atom_dO_tail: cute.CopyAtom,
    tma_atom_K: cute.CopyAtom,
    tma_atom_K_tail: cute.CopyAtom,
    tma_atom_V: cute.CopyAtom,
    tma_atom_V_tail: cute.CopyAtom,
    tma_atom_dK: cute.CopyAtom,
    tma_atom_dK_tail: cute.CopyAtom,
    tma_atom_dV: cute.CopyAtom,
    tma_atom_dV_tail: cute.CopyAtom,
    mLSE: cute.Tensor,
    mdPsum: cute.Tensor,
    mCuSeqlensQ: Optional[cute.Tensor],
    mCuSeqlensK: Optional[cute.Tensor],
    softmax_scale_log2: Float32,
    softmax_scale: Float32,
    sA_layout: cute.ComposedLayout,
    sA_tail_layout: cute.ComposedLayout,
    sK_persist_layout: cute.ComposedLayout,
    sK_persist_tail_layout: cute.ComposedLayout,
    sV_persist_layout: cute.ComposedLayout,
    sV_persist_tail_layout: cute.ComposedLayout,
    sB_epi_tail_layout: cute.ComposedLayout,
    sPdS_layout: cute.ComposedLayout,
    sP_fp32_layout: cute.ComposedLayout,
    tiled_mma_SdP: cute.TiledMma,
    tiled_mma_dK: cute.TiledMma,
    tiled_mma_dV: cute.TiledMma,
    tiled_mma_dK_tail: cute.TiledMma,
    tiled_mma_dV_tail: cute.TiledMma,
    tile_sched_params: ParamsBase,
    TileScheduler: cutlass.Constexpr[Callable],
    SharedStorage: cutlass.Constexpr[Callable],
    qhead_per_kvhead_divmod: Optional[FastDivmodDivisor] = None,
  ):
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    if warp_idx == 0:
      for atom in [
        tma_atom_Q, tma_atom_Q_tail, tma_atom_dO, tma_atom_dO_tail, tma_atom_K,
        tma_atom_K_tail, tma_atom_V, tma_atom_V_tail, tma_atom_dK,
        tma_atom_dK_tail, tma_atom_dV, tma_atom_dV_tail
      ]:
        cpasync.prefetch_descriptor(atom)

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)

    pipeline_producer_group = cutlass.pipeline.CooperativeGroup(
      cutlass.pipeline.Agent.Thread
    )
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
    # pipeline_K: full 256 chunk + true-tail 128 chunk; WG1 only.
    pipeline_K = pipeline.PipelineTmaAsync.create(
      barrier_storage=storage.mbar_ptr_K.data_ptr(),
      num_stages=self.K_persist_chunks,
      producer_group=pipeline_producer_group,
      consumer_group=pipeline_consumer_wg1,
      tx_count=self.tma_copy_bytes["KV_tail"],
      defer_sync=True,
    )
    # pipeline_V: full 256 chunk + true-tail 128 chunk; WG2 only.
    pipeline_V = pipeline.PipelineTmaAsync.create(
      barrier_storage=storage.mbar_ptr_V.data_ptr(),
      num_stages=self.V_persist_chunks,
      producer_group=pipeline_producer_group,
      consumer_group=pipeline_consumer_wg2,
      tx_count=self.tma_copy_bytes["KV_tail"],
      defer_sync=True,
    )
    # pipeline_A: Q/dO streaming. Base tx_count covers tail; full chunks and
    # LSE/dPsum piggybacks add extra_tx_count.
    pipeline_A = pipeline.PipelineTmaAsync.create(
      barrier_storage=storage.mbar_ptr_A.data_ptr(),
      num_stages=self.A_stage,
      producer_group=pipeline_producer_group,
      consumer_group=pipeline_consumer_a,
      tx_count=self.tma_copy_bytes["A_tail"],
      defer_sync=True,
    )

    pipeline_init_arrive(cluster_shape_mn=(1, 1), is_relaxed=True)
    pipeline_init_wait(cluster_shape_mn=(1, 1))

    # sA / sEpi share physical SMEM via union; mainloop uses sA view,
    # epilogue uses sEpi view (with explicit fence + Epilogue barrier at transition).
    sA = storage.sAEpi.sA.get_tensor(sA_layout.outer, swizzle=sA_layout.inner)
    sA_tail = storage.sAEpi.sA.get_tensor(
      sA_tail_layout.outer, swizzle=sA_tail_layout.inner
    )
    sK_persist = storage.sK_persist.get_tensor(
      sK_persist_layout.outer, swizzle=sK_persist_layout.inner
    )
    sK_persist_tail = storage.sK_persist.get_tensor(
      sK_persist_tail_layout.outer, swizzle=sK_persist_tail_layout.inner
    )
    sV_persist = storage.sV_persist.get_tensor(
      sV_persist_layout.outer, swizzle=sV_persist_layout.inner
    )
    sV_persist_tail = storage.sV_persist.get_tensor(
      sV_persist_tail_layout.outer, swizzle=sV_persist_tail_layout.inner
    )
    sP = storage.sP.get_tensor(sPdS_layout.outer, swizzle=sPdS_layout.inner)

    # fp32 P buffer: WG1 writes here (alongside bf16 sP); WG2 reads here for dS computation.
    sP_fp32 = storage.sP_fp32.get_tensor(
      sP_fp32_layout.outer, swizzle=sP_fp32_layout.inner
    )
    sdS = storage.sdS.get_tensor(sPdS_layout.outer, swizzle=sPdS_layout.inner)

    # Single-stage SMEM view for epilogue TMA store (alias of sA's physical slot).
    sB_epi_layout_sel = cute.select(sK_persist_layout, mode=[0, 1])
    sB_epi = storage.sAEpi.sEpi.get_tensor(
      sB_epi_layout_sel.outer, swizzle=sB_epi_layout_sel.inner
    )
    sB_epi_tail = storage.sAEpi.sEpi.get_tensor(
      sB_epi_tail_layout.outer, swizzle=sB_epi_tail_layout.inner
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
    tidx, _, _ = cute.arch.thread_idx()
    warp_group_idx = cute.arch.make_warp_uniform(
      tidx // self.num_threads_per_warp_group
    )

    if warp_group_idx == 0:
      # Producer WG: only warp 0 issues TMA; others idle (held by setmaxregister_decrease).
      cute.arch.setmaxregister_decrease(self.num_producer_regs)
      if warp_idx == 0:
        self.load(
          mQ,
          mQ_tail,
          mK,
          mK_tail,
          mV,
          mV_tail,
          mdO,
          mdO_tail,
          mLSE,
          mdPsum,
          sA,
          sA_tail,
          sK_persist,
          sK_persist_tail,
          sV_persist,
          sV_persist_tail,
          sLSE,
          sdPsum,
          tma_atom_Q,
          tma_atom_Q_tail,
          tma_atom_dO,
          tma_atom_dO_tail,
          tma_atom_K,
          tma_atom_K_tail,
          tma_atom_V,
          tma_atom_V_tail,
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
        tiled_mma_dV_tail,
        mdV,
        mdV_tail,
        sA,
        sA_tail,
        sK_persist,
        sK_persist_tail,
        sP,
        sP_fp32,
        sLSE,
        sB_epi,
        sB_epi_tail,
        pipeline_A,
        pipeline_K,
        tidx_in_wg,
        tma_atom_dV,
        tma_atom_dV_tail,
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
        tiled_mma_dK_tail,
        mdK,
        mdK_tail,
        sA,
        sA_tail,
        sV_persist,
        sV_persist_tail,
        sP_fp32,
        sdS,
        sdPsum,
        sB_epi,
        sB_epi_tail,
        pipeline_A,
        pipeline_V,
        tidx_in_wg,
        tma_atom_dK,
        tma_atom_dK_tail,
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
    mQ_tail: cute.Tensor,
    mK: cute.Tensor,
    mK_tail: cute.Tensor,
    mV: cute.Tensor,
    mV_tail: cute.Tensor,
    mdO: cute.Tensor,
    mdO_tail: cute.Tensor,
    mLSE: cute.Tensor,
    mdPsum: cute.Tensor,
    sA: cute.Tensor,
    sA_tail: cute.Tensor,
    sK_persist: cute.Tensor,
    sK_persist_tail: cute.Tensor,
    sV_persist: cute.Tensor,
    sV_persist_tail: cute.Tensor,
    sLSE: cute.Tensor,
    sdPsum: cute.Tensor,
    tma_atom_Q: cute.CopyAtom,
    tma_atom_Q_tail: cute.CopyAtom,
    tma_atom_dO: cute.CopyAtom,
    tma_atom_dO_tail: cute.CopyAtom,
    tma_atom_K: cute.CopyAtom,
    tma_atom_K_tail: cute.CopyAtom,
    tma_atom_V: cute.CopyAtom,
    tma_atom_V_tail: cute.CopyAtom,
    pipeline_A: pipeline.PipelineAsync,
    pipeline_K: pipeline.PipelineAsync,
    pipeline_V: pipeline.PipelineAsync,
    block_info: BlockInfo,
    SeqlenInfoCls: Callable,
    TileSchedulerCls: Callable,
    qhead_per_kvhead_divmod: Optional[FastDivmodDivisor] = None,
  ):
    producer_state_A = cutlass.pipeline.make_pipeline_state(
      cutlass.pipeline.PipelineUserType.Producer, self.A_stage
    )
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
      mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[None, None,
                                                           head_idx_kv]
      mK_tail_cur = seqlen.offset_batch_K(mK_tail, batch_idx,
                                          dim=3)[None, None, head_idx_kv]
      mV_cur = seqlen.offset_batch_K(mV, batch_idx, dim=3)[None, None,
                                                           head_idx_kv]
      mV_tail_cur = seqlen.offset_batch_K(mV_tail, batch_idx,
                                          dim=3)[None, None, head_idx_kv]

      m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)
      process_tile = const_expr(
        not self.is_varlen_q
      ) or m_block_min < m_block_max

      if process_tile:
        # ═══ K/V preload ONCE per work_tile (Cross-pass reuse) ═══
        # K full 0:256 → sK_persist[0]
        gK_full = cute.local_tile(
          mK_cur, (self.tile_n, self.d_chunk), (None, 0)
        )
        load_K_full, _, _ = copy_utils.tma_get_copy_fn(
          tma_atom_K,
          0,
          cute.make_layout(1),
          gK_full,
          sK_persist,
        )
        pipeline_K.producer_acquire(
          producer_state_K, extra_tx_count=self.tma_copy_bytes["KV_full_extra"]
        )
        load_K_full(
          src_idx=n_block,
          dst_idx=producer_state_K.index,
          tma_bar_ptr=pipeline_K.producer_get_barrier(producer_state_K),
        )
        pipeline_K.producer_commit(producer_state_K)
        did_produce_K = Boolean(True)
        producer_state_K.advance()

        # K true-tail 256:384 → sK_persist_tail[1]
        gK_tail = cute.local_tile(
          mK_tail_cur, (self.tile_n, self.d_chunk_tail), (None, 2)
        )
        load_K_tail, _, _ = copy_utils.tma_get_copy_fn(
          tma_atom_K_tail,
          0,
          cute.make_layout(1),
          gK_tail,
          sK_persist_tail,
        )
        pipeline_K.producer_acquire(producer_state_K)
        load_K_tail(
          src_idx=n_block,
          dst_idx=producer_state_K.index,
          tma_bar_ptr=pipeline_K.producer_get_barrier(producer_state_K),
        )
        pipeline_K.producer_commit(producer_state_K)
        did_produce_K = Boolean(True)
        producer_state_K.advance()

        # V full 0:256 → sV_persist[0]
        gV_full = cute.local_tile(
          mV_cur, (self.tile_n, self.d_chunk), (None, 0)
        )
        load_V_full, _, _ = copy_utils.tma_get_copy_fn(
          tma_atom_V,
          0,
          cute.make_layout(1),
          gV_full,
          sV_persist,
        )
        pipeline_V.producer_acquire(
          producer_state_V, extra_tx_count=self.tma_copy_bytes["KV_full_extra"]
        )
        load_V_full(
          src_idx=n_block,
          dst_idx=producer_state_V.index,
          tma_bar_ptr=pipeline_V.producer_get_barrier(producer_state_V),
        )
        pipeline_V.producer_commit(producer_state_V)
        did_produce_V = Boolean(True)
        producer_state_V.advance()

        # V true-tail 256:384 → sV_persist_tail[1]
        gV_tail = cute.local_tile(
          mV_tail_cur, (self.tile_n, self.d_chunk_tail), (None, 2)
        )
        load_V_tail, _, _ = copy_utils.tma_get_copy_fn(
          tma_atom_V_tail,
          0,
          cute.make_layout(1),
          gV_tail,
          sV_persist_tail,
        )
        pipeline_V.producer_acquire(producer_state_V)
        load_V_tail(
          src_idx=n_block,
          dst_idx=producer_state_V.index,
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
            mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None,
                                                                 head_idx_q]
            mQ_tail_cur = seqlen.offset_batch_Q(mQ_tail, batch_idx,
                                                dim=3)[None, None, head_idx_q]
            mdO_cur = seqlen.offset_batch_Q(mdO, batch_idx, dim=3)[None, None,
                                                                   head_idx_q]
            mdO_tail_cur = seqlen.offset_batch_Q(mdO_tail, batch_idx,
                                                 dim=3)[None, None, head_idx_q]
            mLSE_cur = seqlen.offset_batch_Q(
              mLSE, batch_idx, dim=2, padded=True
            )[None, head_idx_q]
            mdPsum_cur = seqlen.offset_batch_Q(
              mdPsum, batch_idx, dim=2, padded=True
            )[None, head_idx_q]

            gLSE = cute.local_tile(mLSE_cur, (self.tile_m, ), (None, ))
            gdPsum = cute.local_tile(mdPsum_cur, (self.tile_m, ), (None, ))
            if const_expr(self.use_pdl):
              cute.arch.griddepcontrol_wait()
            load_LSE = copy_utils.cpasync_bulk_get_copy_fn(gLSE, sLSE)
            load_dPsum = copy_utils.cpasync_bulk_get_copy_fn(gdPsum, sdPsum)

            for m_block in cutlass.range(m_block_min, m_block_max, unroll=1):
              # ═══ Phase 1: Q full + true-tail → sA. LSE piggybacks on tail. ═══
              gQ_full = cute.local_tile(
                mQ_cur, (self.tile_m, self.d_chunk), (None, 0)
              )
              load_Q_full, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Q,
                0,
                cute.make_layout(1),
                gQ_full,
                sA,
              )
              pipeline_A.producer_acquire(
                producer_state_A,
                extra_tx_count=self.tma_copy_bytes["A_full_extra"],
              )
              load_Q_full(
                src_idx=m_block,
                dst_idx=producer_state_A.index,
                tma_bar_ptr=pipeline_A.producer_get_barrier(producer_state_A),
              )
              pipeline_A.producer_commit(producer_state_A)
              did_produce_A = Boolean(True)
              producer_state_A.advance()

              gQ_tail = cute.local_tile(
                mQ_tail_cur, (self.tile_m, self.d_chunk_tail), (None, 2)
              )
              load_Q_tail, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Q_tail,
                0,
                cute.make_layout(1),
                gQ_tail,
                sA_tail,
              )
              pipeline_A.producer_acquire(
                producer_state_A, extra_tx_count=self.tma_copy_bytes["LSE"]
              )
              load_Q_tail(
                src_idx=m_block,
                dst_idx=producer_state_A.index,
                tma_bar_ptr=pipeline_A.producer_get_barrier(producer_state_A),
              )
              load_LSE(
                src_idx=m_block,
                dst_idx=producer_state_A.index,
                tma_bar_ptr=pipeline_A.producer_get_barrier(producer_state_A),
              )
              pipeline_A.producer_commit(producer_state_A)
              did_produce_A = Boolean(True)
              producer_state_A.advance()

              # ═══ Phase 3: dO full + true-tail → sA. dPsum piggybacks on tail. ═══
              gdO_full = cute.local_tile(
                mdO_cur, (self.tile_m, self.d_chunk), (None, 0)
              )
              load_dO_full, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_dO,
                0,
                cute.make_layout(1),
                gdO_full,
                sA,
              )
              pipeline_A.producer_acquire(
                producer_state_A,
                extra_tx_count=self.tma_copy_bytes["A_full_extra"],
              )
              load_dO_full(
                src_idx=m_block,
                dst_idx=producer_state_A.index,
                tma_bar_ptr=pipeline_A.producer_get_barrier(producer_state_A),
              )
              pipeline_A.producer_commit(producer_state_A)
              did_produce_A = Boolean(True)
              producer_state_A.advance()

              gdO_tail = cute.local_tile(
                mdO_tail_cur, (self.tile_m, self.d_chunk_tail), (None, 2)
              )
              load_dO_tail, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_dO_tail,
                0,
                cute.make_layout(1),
                gdO_tail,
                sA_tail,
              )
              pipeline_A.producer_acquire(
                producer_state_A, extra_tx_count=self.tma_copy_bytes["dPsum"]
              )
              load_dO_tail(
                src_idx=m_block,
                dst_idx=producer_state_A.index,
                tma_bar_ptr=pipeline_A.producer_get_barrier(producer_state_A),
              )
              load_dPsum(
                src_idx=m_block,
                dst_idx=producer_state_A.index,
                tma_bar_ptr=pipeline_A.producer_get_barrier(producer_state_A),
              )
              pipeline_A.producer_commit(producer_state_A)
              did_produce_A = Boolean(True)
              producer_state_A.advance()

              # ═══ Phase 5: dO_d_pass → sA (for WG1 dV GEMM) ═══
              if cutlass.const_expr(d_pass == 0):
                gdO_pass = cute.local_tile(
                  mdO_cur, (self.tile_m, self.d_chunk), (None, 0)
                )
                load_dO_pass, _, _ = copy_utils.tma_get_copy_fn(
                  tma_atom_dO,
                  0,
                  cute.make_layout(1),
                  gdO_pass,
                  sA,
                )
                pipeline_A.producer_acquire(
                  producer_state_A,
                  extra_tx_count=self.tma_copy_bytes["A_full_extra"],
                )
                load_dO_pass(
                  src_idx=m_block,
                  dst_idx=producer_state_A.index,
                  tma_bar_ptr=pipeline_A.producer_get_barrier(producer_state_A),
                )
              else:
                # pass1 consumes only D 256:384; loading a 128-wide tile avoids
                # pulling the padded 384:512 half into SMEM for dV.
                gdO_pass_tail = cute.local_tile(
                  mdO_tail_cur, (self.tile_m, self.d_chunk_tail), (None, 2)
                )
                load_dO_pass_tail, _, _ = copy_utils.tma_get_copy_fn(
                  tma_atom_dO_tail,
                  0,
                  cute.make_layout(1),
                  gdO_pass_tail,
                  sA_tail,
                )
                pipeline_A.producer_acquire(producer_state_A)
                load_dO_pass_tail(
                  src_idx=m_block,
                  dst_idx=producer_state_A.index,
                  tma_bar_ptr=pipeline_A.producer_get_barrier(producer_state_A),
                )
              pipeline_A.producer_commit(producer_state_A)
              did_produce_A = Boolean(True)
              producer_state_A.advance()

              # ═══ Phase 6: Q_d_pass → sA (for WG2 dK GEMM) ═══
              if cutlass.const_expr(d_pass == 0):
                gQ_pass = cute.local_tile(
                  mQ_cur, (self.tile_m, self.d_chunk), (None, 0)
                )
                load_Q_pass, _, _ = copy_utils.tma_get_copy_fn(
                  tma_atom_Q,
                  0,
                  cute.make_layout(1),
                  gQ_pass,
                  sA,
                )
                pipeline_A.producer_acquire(
                  producer_state_A,
                  extra_tx_count=self.tma_copy_bytes["A_full_extra"],
                )
                load_Q_pass(
                  src_idx=m_block,
                  dst_idx=producer_state_A.index,
                  tma_bar_ptr=pipeline_A.producer_get_barrier(producer_state_A),
                )
              else:
                # pass1 dK mirrors dV: the Q operand is a 128-wide true-tail tile.
                gQ_pass_tail = cute.local_tile(
                  mQ_tail_cur, (self.tile_m, self.d_chunk_tail), (None, 2)
                )
                load_Q_pass_tail, _, _ = copy_utils.tma_get_copy_fn(
                  tma_atom_Q_tail,
                  0,
                  cute.make_layout(1),
                  gQ_pass_tail,
                  sA_tail,
                )
                pipeline_A.producer_acquire(producer_state_A)
                load_Q_pass_tail(
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
  # WG1 = S (P1) + softmax→sP→arrive PFull (P2) + dV (P5)
  # ════════════════════════════════════════════════════════════════════
  @cute.jit
  def mma_wg1(
    self,
    tiled_mma_SdP: cute.TiledMma,
    tiled_mma_dV: cute.TiledMma,
    tiled_mma_dV_tail: cute.TiledMma,
    mdV: cute.Tensor,
    mdV_tail: cute.Tensor,
    sA: cute.Tensor,
    sA_tail: cute.Tensor,
    sK_persist: cute.Tensor,
    sK_persist_tail: cute.Tensor,
    sP: cute.Tensor,
    sP_fp32: cute.Tensor,  # fp32 P buffer for WG2 to read (no precision loss)
    sLSE: cute.Tensor,
    sB_epi: cute.Tensor,
    sB_epi_tail: cute.Tensor,
    pipeline_A: pipeline.PipelineAsync,
    pipeline_K: pipeline.PipelineAsync,
    tidx: Int32,
    tma_atom_dV: cute.CopyAtom,
    tma_atom_dV_tail: cute.CopyAtom,
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
    wg_mma_dV_tail = tiled_mma_dV_tail.get_slice(0)

    # ── SdP fragments (WG1 needs only K, not V) ──
    shape_mnk_SdP = (self.tile_m, self.tile_n, self.d_chunk)
    _, tSrA, tSrB_K = sm90_utils.partition_fragment_ABC(
      wg_mma_SdP, shape_mnk_SdP, sA, sK_persist, swap_AB=False
    )
    shape_mnk_SdP_tail = (self.tile_m, self.tile_n, self.d_chunk_tail)
    _, tSrA_tail, tSrB_K_tail = sm90_utils.partition_fragment_ABC(
      wg_mma_SdP,
      shape_mnk_SdP_tail,
      sA_tail,
      sK_persist_tail,
      swap_AB=False,
    )
    # S still accumulates into the full (tile_m, tile_n) accumulator; only the K
    # dimension changes from the full chunk to the 128-wide tail chunk.

    # ── dV GEMM fragments: dV = P^T @ dO_d_pass ──
    sPt = layout_utils.transpose_view(sP)
    sAt = layout_utils.transpose_view(sA)
    shape_mnk_dV = (self.tile_n, self.d_chunk, self.tile_m)
    acc_dV, tdVrPt, tdVrdOt = sm90_utils.partition_fragment_ABC(
      wg_mma_dV, shape_mnk_dV, sPt, sAt, swap_AB=False
    )
    mma_dV_fn = partial(
      gemm_w_idx, tiled_mma_dV, acc_dV, tdVrPt, tdVrdOt, swap_AB=False
    )
    sAt_tail = layout_utils.transpose_view(sA_tail)
    shape_mnk_dV_tail = (self.tile_n, self.d_chunk_tail, self.tile_m)
    acc_dV_tail, tdVrPt_tail, tdVrdOt_tail = sm90_utils.partition_fragment_ABC(
      wg_mma_dV_tail, shape_mnk_dV_tail, sPt, sAt_tail, swap_AB=False
    )
    # dV tail has a smaller N dimension, so it needs an independent accumulator
    # and epilogue store path instead of slicing the full 256-wide accumulator.
    mma_dV_tail_fn = partial(
      gemm_w_idx,
      tiled_mma_dV_tail,
      acc_dV_tail,
      tdVrPt_tail,
      tdVrdOt_tail,
      swap_AB=False,
    )

    # ── P R2S copies: ──
    #   (a) bf16 sP for WG1 dV WGMMA SS-mode (must be fp16/bf16 for WGMMA A operand)
    copy_P_r2s, _, _ = copy_utils.get_smem_store_C(
      tiled_mma_SdP,
      sP,
      tidx,
      transpose=False,
      position_independent=True,
    )
    #   (b) fp32 sP store via get_smem_store_C → CopyUniversalOp atom in C-operand
    #       layout (matches partition_C(sP_fp32) addressing)
    copy_P_fp32_r2s, _, _ = copy_utils.get_smem_store_C(
      tiled_mma_SdP,
      sP_fp32,
      tidx,
      transpose=False,
      position_independent=False,
    )

    # ── LSE partitioning (per-thread row mapping via MMA C partition) ──
    tLSEsLSE = layout_utils.mma_partition_C_vec(
      sLSE, thr_mma_SdP, expand_shape=self.tile_n, is_colvec=True
    )

    consumer_state_A = cutlass.pipeline.make_pipeline_state(
      cutlass.pipeline.PipelineUserType.Consumer, self.A_stage
    )
    consumer_state_K = cutlass.pipeline.make_pipeline_state(
      cutlass.pipeline.PipelineUserType.Consumer, self.K_persist_chunks
    )

    tile_scheduler = TileSchedulerCls()
    work_tile = tile_scheduler.initial_work_tile_info()

    while work_tile.is_valid_tile:
      n_block, head_idx_kv, batch_idx, _ = work_tile.tile_idx
      seqlen = SeqlenInfoCls(batch_idx)
      m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)
      process_tile = const_expr(
        not self.is_varlen_q
      ) or m_block_min < m_block_max

      if process_tile:
        # K preload wait & release ONCE per work_tile (Cross-pass reuse)
        for _k in cutlass.range_constexpr(self.K_persist_chunks):
          pipeline_K.consumer_wait(
            consumer_state_K, pipeline_K.consumer_try_wait(consumer_state_K)
          )
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
                tSrA_tail,
                tSrB_K_tail,
                shape_mnk_SdP,
                mma_dV_fn,
                mma_dV_tail_fn,
                copy_P_r2s,
                copy_P_fp32_r2s,
                pipeline_A,
                tLSEsLSE,
                softmax_scale_log2,
                mask_fn,
                d_pass,
                dKV_accumulate=dKV_accumulate,
              )
              dKV_accumulate = Boolean(True)

          # Per-d_pass epilogue: WG1 writes dV[:, d_pass*d_chunk:(d_pass+1)*d_chunk]
          if cutlass.const_expr(d_pass == 0):
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
          else:
            self.epilogue_dV_tail_slice(
              acc_dV_tail,
              mdV_tail,
              sB_epi_tail,
              seqlen,
              tma_atom_dV_tail,
              tiled_mma_dV_tail,
              tidx,
              n_block,
              head_idx_kv,
              batch_idx,
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
    tSrA_tail: cute.Tensor,
    tSrB_K_tail: cute.Tensor,
    shape_mnk_SdP: cute.Shape,
    mma_dV_fn: Callable,
    mma_dV_tail_fn: Callable,
    copy_P_r2s: Callable,
    copy_P_fp32_r2s:
    Callable,  # fp32 sP_fp32 store via CopyUniversalOp atom in C-operand layout
    pipeline_A: pipeline.PipelineAsync,
    tLSEsLSE: cute.Tensor,
    softmax_scale_log2: Float32,
    mask_fn: Callable,
    d_pass: cutlass.Constexpr[int],
    dKV_accumulate: Boolean = True,
  ):
    p_stage = Int32(0)  # sP single-buffered

    # ═══ Phase 1: S = Q_full @ K_full^T + Q_tail @ K_tail^T ═══
    acc_S = cute.make_rmem_tensor(
      tiled_mma_SdP.partition_shape_C(shape_mnk_SdP[:2]), Float32
    )
    pipeline_A.consumer_wait(
      consumer_state_A, pipeline_A.consumer_try_wait(consumer_state_A)
    )
    gemm_w_idx(
      tiled_mma_SdP,
      acc_S,
      tSrA,
      tSrB_K,
      zero_init=True,
      A_idx=consumer_state_A.index,
      B_idx=0,
      wg_wait=0,
    )
    with cute.arch.elect_one():
      pipeline_A.consumer_release(consumer_state_A)
    consumer_state_A.advance()

    pipeline_A.consumer_wait(
      consumer_state_A, pipeline_A.consumer_try_wait(consumer_state_A)
    )
    gemm_w_idx(
      tiled_mma_SdP,
      acc_S,
      tSrA_tail,
      tSrB_K_tail,
      zero_init=False,
      A_idx=consumer_state_A.index,
      B_idx=1,
      wg_wait=0,
    )
    # LSE piggybacked on the Q tail chunk — read before release.
    tLSErLSE = copy_utils.load_s2r(tLSEsLSE[None, consumer_state_A.index])
    with cute.arch.elect_one():
      pipeline_A.consumer_release(consumer_state_A)
    consumer_state_A.advance()

    # ═══ Empty release: dO0..dO_{num_d_inner-1} (consumed by WG2) ═══
    for _ in cutlass.range_constexpr(self.num_d_inner):
      pipeline_A.consumer_wait(
        consumer_state_A, pipeline_A.consumer_try_wait(consumer_state_A)
      )
      with cute.arch.elect_one():
        pipeline_A.consumer_release(consumer_state_A)
      consumer_state_A.advance()

    # ═══ Phase 2: mask + softmax → P (fp32 in rmem, in-place on acc_S) ═══
    mask_fn(acc_S, m_block=m_block)
    acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S, transpose=False)
    for r in cutlass.range_constexpr(cute.size(acc_S_mn, mode=[0])):
      lse_val = tLSErLSE[r]
      for c in cutlass.range(cute.size(acc_S_mn, mode=[1]), unroll_full=True):
        acc_S_mn[r, c] = cute.math.exp2(
          acc_S_mn[r, c] * softmax_scale_log2 - lse_val, fastmath=True
        )

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
    pipeline_A.consumer_wait(
      consumer_state_A, pipeline_A.consumer_try_wait(consumer_state_A)
    )
    smem_idx_dO_pass = consumer_state_A.index
    if cutlass.const_expr(d_pass == 0):
      mma_dV_fn(
        A_idx=p_stage,
        B_idx=smem_idx_dO_pass,
        zero_init=not dKV_accumulate,
        wg_wait=0,
      )
    else:
      mma_dV_tail_fn(
        A_idx=p_stage,
        B_idx=smem_idx_dO_pass,
        zero_init=not dKV_accumulate,
        wg_wait=0,
      )
    with cute.arch.elect_one():
      pipeline_A.consumer_release(consumer_state_A)
    consumer_state_A.advance()

    # ═══ Empty release Q_d_pass (consumed by WG2 P6) ═══
    pipeline_A.consumer_wait(
      consumer_state_A, pipeline_A.consumer_try_wait(consumer_state_A)
    )
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
    tiled_mma_dK_tail: cute.TiledMma,
    mdK: cute.Tensor,
    mdK_tail: cute.Tensor,
    sA: cute.Tensor,
    sA_tail: cute.Tensor,
    sV_persist: cute.Tensor,
    sV_persist_tail: cute.Tensor,
    sP_fp32: cute.
    Tensor,  # fp32 P buffer (replaces bf16 sP; WG2 no longer needs bf16)
    sdS: cute.Tensor,
    sdPsum: cute.Tensor,
    sB_epi: cute.Tensor,
    sB_epi_tail: cute.Tensor,
    pipeline_A: pipeline.PipelineAsync,
    pipeline_V: pipeline.PipelineAsync,
    tidx: Int32,
    tma_atom_dK: cute.CopyAtom,
    tma_atom_dK_tail: cute.CopyAtom,
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
    wg_mma_dK_tail = tiled_mma_dK_tail.get_slice(0)

    # ── SdP fragments (WG2 needs only V, not K) ──
    shape_mnk_SdP = (self.tile_m, self.tile_n, self.d_chunk)
    _, tSrA, tSrB_V = sm90_utils.partition_fragment_ABC(
      wg_mma_SdP, shape_mnk_SdP, sA, sV_persist, swap_AB=False
    )
    shape_mnk_SdP_tail = (self.tile_m, self.tile_n, self.d_chunk_tail)
    _, tSrA_tail, tSrB_V_tail = sm90_utils.partition_fragment_ABC(
      wg_mma_SdP,
      shape_mnk_SdP_tail,
      sA_tail,
      sV_persist_tail,
      swap_AB=False,
    )
    # dP reduction must also be true-tail; otherwise dS would include padded V
    # columns that are outside logical D384.

    # ── dK GEMM fragments: dK = dS^T @ Q_d_pass ──
    sdSt = layout_utils.transpose_view(sdS)
    sAt = layout_utils.transpose_view(sA)
    shape_mnk_dK = (self.tile_n, self.d_chunk, self.tile_m)
    acc_dK, tdKrdSt, tdKrQt = sm90_utils.partition_fragment_ABC(
      wg_mma_dK, shape_mnk_dK, sdSt, sAt, swap_AB=False
    )
    mma_dK_fn = partial(
      gemm_w_idx, tiled_mma_dK, acc_dK, tdKrdSt, tdKrQt, swap_AB=False
    )
    sAt_tail = layout_utils.transpose_view(sA_tail)
    shape_mnk_dK_tail = (self.tile_n, self.d_chunk_tail, self.tile_m)
    acc_dK_tail, tdKrdSt_tail, tdKrQt_tail = sm90_utils.partition_fragment_ABC(
      wg_mma_dK_tail, shape_mnk_dK_tail, sdSt, sAt_tail, swap_AB=False
    )
    # dK tail follows the same split as dV tail: separate accumulator and store
    # avoid both padded MMA work and padded global writes.
    mma_dK_tail_fn = partial(
      gemm_w_idx,
      tiled_mma_dK_tail,
      acc_dK_tail,
      tdKrdSt_tail,
      tdKrQt_tail,
      swap_AB=False,
    )

    # ── dS R2S copy (WG2-internal: STSM rmem dS → sdS[0]) ──
    copy_dS_r2s, _, _ = copy_utils.get_smem_store_C(
      tiled_mma_SdP,
      sdS,
      tidx,
      transpose=False,
      position_independent=True,
    )

    # ── sP_fp32 s2r partition (WG2 reads fp32 P from sP_fp32, NOT bf16 sP).
    tSsP_fp32_partition = thr_mma_SdP.partition_C(
      sP_fp32
    )  # ((atom), MMA_M, MMA_N, stage)

    # ── dPsum partitioning (per-thread row mapping) ──
    tLSEsdPsum = layout_utils.mma_partition_C_vec(
      sdPsum, thr_mma_SdP, expand_shape=self.tile_n, is_colvec=True
    )

    #   pre-arrive PEmpty so WG1's first barrier(PEmpty, 256) finds
    #   128 arrivals from WG2 and only waits for WG1's own 128.
    cute.arch.barrier_arrive(
      barrier_id=int(NamedBarrierBwd.PEmpty),
      number_of_threads=self.num_mma_threads,  # 256
    )

    consumer_state_A = cutlass.pipeline.make_pipeline_state(
      cutlass.pipeline.PipelineUserType.Consumer, self.A_stage
    )
    consumer_state_V = cutlass.pipeline.make_pipeline_state(
      cutlass.pipeline.PipelineUserType.Consumer, self.V_persist_chunks
    )

    tile_scheduler = TileSchedulerCls()
    work_tile = tile_scheduler.initial_work_tile_info()

    while work_tile.is_valid_tile:
      n_block, head_idx_kv, batch_idx, _ = work_tile.tile_idx
      seqlen = SeqlenInfoCls(batch_idx)
      m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)
      process_tile = const_expr(
        not self.is_varlen_q
      ) or m_block_min < m_block_max

      if process_tile:
        # ★ V preload wait & release ONCE per work_tile (跨 d_pass 复用)
        for _v in cutlass.range_constexpr(self.V_persist_chunks):
          pipeline_V.consumer_wait(
            consumer_state_V, pipeline_V.consumer_try_wait(consumer_state_V)
          )
          with cute.arch.elect_one():
            pipeline_V.consumer_release(consumer_state_V)
          consumer_state_V.advance()

        # ★ V tail zero (WG2 internal, 128 thread; ONCE per work_tile)
        self.zero_v_tail_smem_wg2(
          sV_persist=sV_persist,
          sV_persist_tail=sV_persist_tail,
          seqlen=seqlen,
          n_block=n_block,
          tidx=tidx,
        )

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
                tSrA_tail,
                tSrB_V_tail,
                shape_mnk_SdP,
                mma_dK_fn,
                mma_dK_tail_fn,
                copy_dS_r2s,
                tSsP_fp32_partition,
                pipeline_A,
                tLSEsdPsum,
                softmax_scale,
                mask_fn,
                d_pass,
                dKV_accumulate=dKV_accumulate,
              )
              dKV_accumulate = Boolean(True)

          # Per-d_pass epilogue: WG2 writes dK[:, d_pass*d_chunk:(d_pass+1)*d_chunk]
          # (serialized with WG1 dV via Epilogue NamedBarrier inside epi function).
          if cutlass.const_expr(d_pass == 0):
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
          else:
            self.epilogue_dK_tail_slice(
              acc_dK_tail,
              mdK_tail,
              sB_epi_tail,
              seqlen,
              tma_atom_dK_tail,
              tiled_mma_dK_tail,
              tidx,
              n_block,
              head_idx_kv,
              batch_idx,
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
    tSrA_tail: cute.Tensor,
    tSrB_V_tail: cute.Tensor,
    shape_mnk_SdP: cute.Shape,
    mma_dK_fn: Callable,
    mma_dK_tail_fn: Callable,
    copy_dS_r2s: Callable,
    tSsP_fp32_partition: cute.
    Tensor,  # thr_mma_SdP.partition_C(sP_fp32) — fp32 per-thread C-acc view
    pipeline_A: pipeline.PipelineAsync,
    tLSEsdPsum: cute.Tensor,
    softmax_scale: Float32,
    mask_fn: Callable,
    d_pass: cutlass.Constexpr[int],
    dKV_accumulate: Boolean = True,
  ):
    p_stage = Int32(0)  # sP/sdS single-buffered

    # ═══ Empty release: Q0..Q_{num_d_inner-1} (consumed by WG1) ═══
    for _ in cutlass.range_constexpr(self.num_d_inner):
      pipeline_A.consumer_wait(
        consumer_state_A, pipeline_A.consumer_try_wait(consumer_state_A)
      )
      with cute.arch.elect_one():
        pipeline_A.consumer_release(consumer_state_A)
      consumer_state_A.advance()

    # ═══ Phase 3: dP = dO_full @ V_full^T + dO_tail @ V_tail^T ═══
    acc_dP = cute.make_rmem_tensor(
      tiled_mma_SdP.partition_shape_C(shape_mnk_SdP[:2]), Float32
    )
    pipeline_A.consumer_wait(
      consumer_state_A, pipeline_A.consumer_try_wait(consumer_state_A)
    )
    gemm_w_idx(
      tiled_mma_SdP,
      acc_dP,
      tSrA,
      tSrB_V,
      zero_init=True,
      A_idx=consumer_state_A.index,
      B_idx=0,
      wg_wait=0,
    )
    with cute.arch.elect_one():
      pipeline_A.consumer_release(consumer_state_A)
    consumer_state_A.advance()

    pipeline_A.consumer_wait(
      consumer_state_A, pipeline_A.consumer_try_wait(consumer_state_A)
    )
    gemm_w_idx(
      tiled_mma_SdP,
      acc_dP,
      tSrA_tail,
      tSrB_V_tail,
      zero_init=False,
      A_idx=consumer_state_A.index,
      B_idx=1,
      wg_wait=0,
    )
    # dPsum piggybacked on the dO tail chunk — read before release.
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
    tdSrP_fp32 = copy_utils.load_s2r(
      tSsP_fp32_partition[None, None, None, p_stage]
    )

    # dS = P * (dP - dpsum) * scale — acc_dP from sync WGMMAs above, P from LDS.
    tdSrP_mn = layout_utils.reshape_acc_to_mn(tdSrP_fp32, transpose=False)
    acc_dP_mn = layout_utils.reshape_acc_to_mn(acc_dP, transpose=False)
    for r in cutlass.range_constexpr(cute.size(acc_dP_mn, mode=[0])):
      dpsum_val = tLSErdPsum[r]
      for c in cutlass.range(cute.size(acc_dP_mn, mode=[1]), unroll_full=True):
        acc_dP_mn[
          r,
          c] = (tdSrP_mn[r, c] * (acc_dP_mn[r, c] - dpsum_val) * softmax_scale)

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
    pipeline_A.consumer_wait(
      consumer_state_A, pipeline_A.consumer_try_wait(consumer_state_A)
    )
    with cute.arch.elect_one():
      pipeline_A.consumer_release(consumer_state_A)
    consumer_state_A.advance()

    # ═══ Phase 6: dK += dS^T @ Q_d_pass ═══
    pipeline_A.consumer_wait(
      consumer_state_A, pipeline_A.consumer_try_wait(consumer_state_A)
    )
    smem_idx_Q_pass = consumer_state_A.index
    if cutlass.const_expr(d_pass == 0):
      mma_dK_fn(
        A_idx=p_stage,
        B_idx=smem_idx_Q_pass,
        zero_init=not dKV_accumulate,
        wg_wait=0,
      )
    else:
      mma_dK_tail_fn(
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
    sV_persist_tail: cute.Tensor,
    seqlen: SeqlenInfoQK,
    n_block: Int32,
    tidx: Int32,  # WG-local tidx ∈ [0, 128)
  ):
    valid_rows = seqlen.seqlen_k - n_block * self.tile_n
    if valid_rows < self.tile_n:
      tail_elems = (self.tile_n - valid_rows) * self.d_chunk
      for d_inner in cutlass.range_constexpr(self.num_d_inner):
        for linear_idx in cutlass.range(
          tidx, tail_elems, self.num_threads_per_warp_group, unroll=1
        ):
          row_offset = linear_idx // self.d_chunk
          col = linear_idx - row_offset * self.d_chunk
          sV_persist[valid_rows + row_offset, col,
                     d_inner] = sV_persist.element_type(0.0)
      tail_elems_tail = (self.tile_n - valid_rows) * self.d_chunk_tail
      # Non-aligned N blocks need the row tail cleared in both persistent V
      # views; dP reads the tail view independently during the second reduction.
      for linear_idx in cutlass.range(
        tidx, tail_elems_tail, self.num_threads_per_warp_group, unroll=1
      ):
        row_offset = linear_idx // self.d_chunk_tail
        col = linear_idx - row_offset * self.d_chunk_tail
        sV_persist_tail[valid_rows + row_offset, col,
                        1] = sV_persist_tail.element_type(0.0)
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

    mdV_cur = seqlen.offset_batch_K(
      mdV, batch_idx, dim=3, ragged=self.varlen_k
    )[None, None, head_idx]
    gdV = cute.local_tile(
      mdV_cur, (self.tile_n, self.d_chunk), (n_block, d_pass)
    )
    store_dV, _, _ = copy_utils.tma_get_copy_fn(
      tma_atom_dV, 0, cute.make_layout(1), sEpi, gdV, single_stage=True
    )
    copy_dV_r2s, _, _ = copy_utils.get_smem_store_C(
      tiled_mma_dV,
      sEpi,
      tidx,
      transpose=False,
      position_independent=True,
    )

    cute.arch.fence_view_async_shared()
    cute.arch.barrier(
      barrier_id=int(NamedBarrierBwd.Epilogue),
      number_of_threads=self.num_mma_threads,
    )

    copy_dV_r2s(acc_dV, dst_idx=None)
    cute.arch.fence_view_async_shared()
    cute.arch.barrier(
      barrier_id=int(NamedBarrierBwd.WarpSchedulerWG2),
      number_of_threads=self.num_threads_per_warp_group,
    )
    if warp_idx == 4:
      store_dV()
      cute.arch.cp_async_bulk_commit_group()
    # The store is issued by warp 4, but the whole WG waits before the cross-WG
    # Epilogue barrier so no lane observes sEpi as reusable too early.
    cute.arch.cp_async_bulk_wait_group(0, read=True)

    cute.arch.barrier(
      barrier_id=int(NamedBarrierBwd.Epilogue),
      number_of_threads=self.num_mma_threads,
    )
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

    mdK_cur = seqlen.offset_batch_K(
      mdK, batch_idx, dim=3, ragged=self.varlen_k
    )[None, None, head_idx]
    gdK = cute.local_tile(
      mdK_cur, (self.tile_n, self.d_chunk), (n_block, d_pass)
    )
    store_dK, _, _ = copy_utils.tma_get_copy_fn(
      tma_atom_dK, 0, cute.make_layout(1), sEpi, gdK, single_stage=True
    )
    copy_dK_r2s, _, _ = copy_utils.get_smem_store_C(
      tiled_mma_dK,
      sEpi,
      tidx,
      transpose=False,
      position_independent=True,
    )

    cute.arch.fence_view_async_shared()
    cute.arch.barrier(
      barrier_id=int(NamedBarrierBwd.Epilogue),
      number_of_threads=self.num_mma_threads,
    )
    cute.arch.barrier(
      barrier_id=int(NamedBarrierBwd.Epilogue),
      number_of_threads=self.num_mma_threads,
    )

    copy_dK_r2s(acc_dK, dst_idx=None)
    cute.arch.fence_view_async_shared()
    cute.arch.barrier(
      barrier_id=int(NamedBarrierBwd.WarpSchedulerWG3),
      number_of_threads=self.num_threads_per_warp_group,
    )
    if warp_idx == 8:
      store_dK()
      cute.arch.cp_async_bulk_commit_group()
    # Keep this wait outside the issuing-warp branch for the same reason as dV.
    cute.arch.cp_async_bulk_wait_group(0, read=True)

    cute.arch.barrier(
      barrier_id=int(NamedBarrierBwd.Epilogue),
      number_of_threads=self.num_mma_threads,
    )

  @cute.jit
  def epilogue_dV_tail_slice(
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
  ):
    """Called by WG1 only. Writes the true-tail dV[:, 256:384] slice."""
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    mdV_cur = seqlen.offset_batch_K(
      mdV, batch_idx, dim=3, ragged=self.varlen_k
    )[None, None, head_idx]
    gdV = cute.local_tile(
      mdV_cur, (self.tile_n, self.d_chunk_tail), (n_block, 2)
    )
    # For 128-wide output tiles, D tile coordinate 2 corresponds to offset 256.
    store_dV, _, _ = copy_utils.tma_get_copy_fn(
      tma_atom_dV, 0, cute.make_layout(1), sEpi, gdV, single_stage=True
    )
    copy_dV_r2s, _, _ = copy_utils.get_smem_store_C(
      tiled_mma_dV,
      sEpi,
      tidx,
      transpose=False,
      position_independent=True,
    )

    cute.arch.fence_view_async_shared()
    cute.arch.barrier(
      barrier_id=int(NamedBarrierBwd.Epilogue),
      number_of_threads=self.num_mma_threads,
    )

    copy_dV_r2s(acc_dV, dst_idx=None)
    cute.arch.fence_view_async_shared()
    cute.arch.barrier(
      barrier_id=int(NamedBarrierBwd.WarpSchedulerWG2),
      number_of_threads=self.num_threads_per_warp_group,
    )
    if warp_idx == 4:
      store_dV()
      cute.arch.cp_async_bulk_commit_group()
    # Tail store uses the same WG-wide wait rule as the full store.
    cute.arch.cp_async_bulk_wait_group(0, read=True)

    cute.arch.barrier(
      barrier_id=int(NamedBarrierBwd.Epilogue),
      number_of_threads=self.num_mma_threads,
    )
    cute.arch.barrier(
      barrier_id=int(NamedBarrierBwd.Epilogue),
      number_of_threads=self.num_mma_threads,
    )

  @cute.jit
  def epilogue_dK_tail_slice(
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
  ):
    """Called by WG2 only. Writes the true-tail dK[:, 256:384] slice."""
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    mdK_cur = seqlen.offset_batch_K(
      mdK, batch_idx, dim=3, ragged=self.varlen_k
    )[None, None, head_idx]
    gdK = cute.local_tile(
      mdK_cur, (self.tile_n, self.d_chunk_tail), (n_block, 2)
    )
    # For 128-wide output tiles, D tile coordinate 2 corresponds to offset 256.
    store_dK, _, _ = copy_utils.tma_get_copy_fn(
      tma_atom_dK, 0, cute.make_layout(1), sEpi, gdK, single_stage=True
    )
    copy_dK_r2s, _, _ = copy_utils.get_smem_store_C(
      tiled_mma_dK,
      sEpi,
      tidx,
      transpose=False,
      position_independent=True,
    )

    cute.arch.fence_view_async_shared()
    cute.arch.barrier(
      barrier_id=int(NamedBarrierBwd.Epilogue),
      number_of_threads=self.num_mma_threads,
    )
    cute.arch.barrier(
      barrier_id=int(NamedBarrierBwd.Epilogue),
      number_of_threads=self.num_mma_threads,
    )

    copy_dK_r2s(acc_dK, dst_idx=None)
    cute.arch.fence_view_async_shared()
    cute.arch.barrier(
      barrier_id=int(NamedBarrierBwd.WarpSchedulerWG3),
      number_of_threads=self.num_threads_per_warp_group,
    )
    if warp_idx == 8:
      store_dK()
      cute.arch.cp_async_bulk_commit_group()
    # Tail store uses the same WG-wide wait rule as the full store.
    cute.arch.cp_async_bulk_wait_group(0, read=True)

    cute.arch.barrier(
      barrier_id=int(NamedBarrierBwd.Epilogue),
      number_of_threads=self.num_mma_threads,
    )

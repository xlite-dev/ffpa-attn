# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# Adapted from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/flash_bwd.py
# FFPA adaptation: SM80/SM89 atomic-free Split-D dK/dV backward with 64-wide D chunks.

from typing import Optional, Type

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils_basic
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync, warp

from quack import layout_utils

from . import utils
from ._utils import SM80_BWD_SPLIT_D_CHUNK
from .utils import ampere_helpers as sm80_utils
from .utils.cute_dsl_utils import assume_tensor_aligned


class FFPAAttnBwdDKDVSm80SplitDGeneric:
  """SM80/SM89 atomic-free dK/dV kernel for the large-D Split-D path.

  One CTA owns one ``(batch, kv_head, n_block)`` tile and streams all query
  blocks. For each score tile it reconstructs ``S`` and ``dP`` once across all
  D chunks, then walks the D chunks to update ``dk`` and ``dv``. This matches
  the Triton Split-D loop ordering and avoids multiplying score recomputation by
  the number of D chunks.
  """

  def __init__(
    self,
    dtype: Type[cutlass.Numeric],
    head_dim: int,
    head_dim_v: Optional[int] = None,
    qhead_per_kvhead: int = 1,
    is_causal: bool = False,
    tile_m: int = 64,
    tile_n: int = 64,
    num_stages_Q: int = 1,
    num_stages_dO: int = 1,
    num_threads: int = 128,
    d_chunk: Optional[int] = None,
    dkdv_storage_dtype: Optional[Type[cutlass.Numeric]] = None,
  ):
    self.dtype = dtype
    # HBM storage dtype for dK/dV. None falls back to ``dtype`` (today's
    # behaviour). Float32 keeps the cross-tile accumulation precision when
    # the q/k/v activation dtype is bf16 / fp16 (mirrors Triton's
    # ``grad_kv_storage_dtype`` option).
    self.dkdv_storage_dtype = dtype if dkdv_storage_dtype is None else dkdv_storage_dtype
    self.head_dim = head_dim
    self.head_dim_v = head_dim if head_dim_v is None else head_dim_v
    self.d_chunk = SM80_BWD_SPLIT_D_CHUNK if d_chunk is None else d_chunk
    self.num_d_chunks = head_dim // self.d_chunk
    self.qhead_per_kvhead = qhead_per_kvhead
    self.is_causal = is_causal
    self.tile_m = tile_m
    self.tile_n = tile_n
    self.num_stages_Q = num_stages_Q
    self.num_stages_dO = num_stages_dO
    self.num_threads = num_threads
    self.tile_hdim = self.d_chunk

  @staticmethod
  def can_implement(
    dtype,
    head_dim,
    head_dim_v,
    tile_m,
    tile_n,
    num_stages_Q,
    num_stages_dO,
    num_threads,
    is_causal,
    smem_capacity_arch: str = "sm_80",
    d_chunk: Optional[int] = None,
    dkdv_storage_dtype: Optional[Type[cutlass.Numeric]] = None,
  ) -> bool:
    """Check whether the SM80 Split-D dK/dV configuration fits resources."""
    del is_causal
    if head_dim_v is None:
      head_dim_v = head_dim
    if d_chunk is None:
      d_chunk = SM80_BWD_SPLIT_D_CHUNK
    if dtype not in [cutlass.Float16, cutlass.BFloat16]:
      return False
    if dkdv_storage_dtype is not None and dkdv_storage_dtype not in (
      dtype, cutlass.Float32
    ):
      return False
    if head_dim != head_dim_v:
      return False
    if head_dim <= 256 or head_dim > 1024 or head_dim % d_chunk != 0:
      return False
    if tile_m % 16 != 0 or tile_n % 16 != 0:
      return False
    if num_threads % 32 != 0:
      return False
    if num_stages_Q < 1 or num_stages_dO < 1:
      return False
    elem_bytes = 2
    # sQ/sdO/sK/sV are physically separated and each carries a stage axis so
    # the d_chunk loop can prefetch stage+1 while MMA consumes the current
    # stage. P/dS kept in registers via SdP swapAB + reshape_acc_to_frgA.
    smem_usage = (
      tile_m * d_chunk * num_stages_Q + tile_n * d_chunk * num_stages_Q +
      tile_m * d_chunk * num_stages_dO + tile_n * d_chunk * num_stages_dO
    ) * elem_bytes
    smem_capacity = utils_basic.get_smem_capacity_in_bytes(smem_capacity_arch)
    return smem_usage <= smem_capacity

  def _check_type(
    self,
    mQ_type: Type[cutlass.Numeric],
    mK_type: Type[cutlass.Numeric],
    mV_type: Type[cutlass.Numeric],
    mdO_type: Type[cutlass.Numeric],
    mLSE_type: Type[cutlass.Numeric],
    mD_type: Type[cutlass.Numeric],
    mdK_type: Type[cutlass.Numeric],
    mdV_type: Type[cutlass.Numeric],
  ) -> None:
    if const_expr(not (mQ_type == mK_type == mV_type == mdO_type)):
      raise TypeError("q, k, v, and dout must have the same dtype")
    if const_expr(mQ_type not in [cutlass.Float16, cutlass.BFloat16]):
      raise TypeError("Only Float16 and BFloat16 are supported")
    if const_expr(mLSE_type is not Float32 or mD_type is not Float32):
      raise TypeError("lse_log2 and dpsum must be Float32")
    if const_expr(
      mdK_type != self.dkdv_storage_dtype or mdV_type != self.dkdv_storage_dtype
    ):
      raise TypeError(
        "dk and dv dtype must match dkdv_storage_dtype "
        "(defaults to the q/k/v dtype when not overridden)"
      )
    assert mQ_type == self.dtype

  def _get_shared_storage_cls(self):

    @cute.struct
    class SharedStorage:
      sQ: cute.struct.Align[cute.struct.MemRange[self.dtype,
                                                 cute.cosize(self.sQ_layout)],
                            1024]
      sdO: cute.struct.Align[cute.struct.MemRange[self.dtype,
                                                  cute.cosize(self.sdO_layout)],
                             1024]
      sK: cute.struct.Align[cute.struct.MemRange[self.dtype,
                                                 cute.cosize(self.sK_layout)],
                            1024]
      sV: cute.struct.Align[cute.struct.MemRange[self.dtype,
                                                 cute.cosize(self.sV_layout)],
                            1024]

    return SharedStorage

  def _setup_attributes(self) -> None:
    sQ_layout_atom = sm80_utils.get_smem_layout_atom(self.dtype, self.tile_hdim)
    sK_layout_atom = sQ_layout_atom
    sV_layout_atom = sm80_utils.get_smem_layout_atom(self.dtype, self.tile_hdim)
    # Stage axis is the slowest-varying dim so per-stage slice keeps the
    # canonical (rows, d_chunk) layout that ldmatrix / mma_make_fragment expect.
    self.sQ_layout = cute.tile_to_shape(
      sQ_layout_atom,
      (self.tile_m, self.tile_hdim, self.num_stages_Q),
      (0, 1, 2),
    )
    self.sdO_layout = cute.tile_to_shape(
      sQ_layout_atom,
      (self.tile_m, self.tile_hdim, self.num_stages_dO),
      (0, 1, 2),
    )
    self.sK_layout = cute.tile_to_shape(
      sK_layout_atom,
      (self.tile_n, self.tile_hdim, self.num_stages_Q),
      (0, 1, 2),
    )
    self.sV_layout = cute.tile_to_shape(
      sV_layout_atom,
      (self.tile_n, self.tile_hdim, self.num_stages_dO),
      (0, 1, 2),
    )

    universal_copy_bits = 128
    async_copy_elems = universal_copy_bits // self.dtype.width
    atom_async_copy = cute.make_copy_atom(
      cpasync.CopyG2SOp(cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL),
      self.dtype,
      num_bits_per_copy=universal_copy_bits,
    )
    tQK_shape_dim_1 = sQ_layout_atom.outer.shape[1] // async_copy_elems
    tQK_layout = cute.make_ordered_layout(
      (self.num_threads // tQK_shape_dim_1, tQK_shape_dim_1), order=(1, 0)
    )
    tVdO_shape_dim_1 = sV_layout_atom.outer.shape[1] // async_copy_elems
    tVdO_layout = cute.make_ordered_layout(
      (self.num_threads // tVdO_shape_dim_1, tVdO_shape_dim_1), order=(1, 0)
    )
    value_layout = cute.make_layout((1, async_copy_elems))
    self.gmem_tiled_copy_QK = cute.make_tiled_copy_tv(
      atom_async_copy, tQK_layout, value_layout
    )
    self.gmem_tiled_copy_VdO = cute.make_tiled_copy_tv(
      atom_async_copy, tVdO_layout, value_layout
    )

  def _get_tiled_mma(self):
    num_mma_warps = self.num_threads // cute.arch.WARP_SIZE
    atom_layout = (num_mma_warps, 1, 1)
    permutation_mnk = (num_mma_warps * 16, 16, 16)
    tiled_mma_sdp = cute.make_tiled_mma(
      warp.MmaF16BF16Op(self.dtype, Float32, (16, 8, 16)),
      atom_layout,
      permutation_mnk=permutation_mnk,
    )
    tiled_mma_dkv = cute.make_tiled_mma(
      warp.MmaF16BF16Op(self.dtype, Float32, (16, 8, 16)),
      atom_layout,
      permutation_mnk=permutation_mnk,
    )
    return tiled_mma_sdp, tiled_mma_dkv

  @cute.jit
  def __call__(
    self,
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mdO: cute.Tensor,
    mLSElog2: cute.Tensor,
    mD: cute.Tensor,
    mdK: cute.Tensor,
    mdV: cute.Tensor,
    softmax_scale: Float32,
    stream: cuda.CUstream = None,
  ):
    """Configure and launch the dense SM80 dK/dV kernel."""
    self._check_type(
      *(t.element_type for t in (mQ, mK, mV, mdO, mLSElog2, mD, mdK, mdV))
    )
    mQ, mK, mV, mdO, mLSElog2, mD, mdK, mdV = [
      assume_tensor_aligned(t)
      for t in (mQ, mK, mV, mdO, mLSElog2, mD, mdK, mdV)
    ]
    self._setup_attributes()
    tiled_mma_sdp, tiled_mma_dkv = self._get_tiled_mma()
    SharedStorage = self._get_shared_storage_cls()
    batch = mQ.shape[0]
    num_head_kv = mK.shape[2]
    seqlen_k = mK.shape[1]
    grid = [
      cute.ceil_div(seqlen_k, self.tile_n),
      batch * num_head_kv,
      1,
    ]
    self.kernel(
      mQ,
      mK,
      mV,
      mdO,
      mLSElog2,
      mD,
      mdK,
      mdV,
      softmax_scale,
      self.sQ_layout,
      self.sdO_layout,
      self.sK_layout,
      self.sV_layout,
      self.gmem_tiled_copy_QK,
      self.gmem_tiled_copy_VdO,
      tiled_mma_sdp,
      tiled_mma_dkv,
      SharedStorage,
    ).launch(
      grid=grid,
      block=[self.num_threads, 1, 1],
      smem=SharedStorage.size_in_bytes(),
      stream=stream,
    )

  @cute.kernel
  def kernel(
    self,
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mdO: cute.Tensor,
    mLSElog2: cute.Tensor,
    mD: cute.Tensor,
    mdK: cute.Tensor,
    mdV: cute.Tensor,
    softmax_scale: Float32,
    sQ_layout: cute.ComposedLayout,
    sdO_layout: cute.ComposedLayout,
    sK_layout: cute.ComposedLayout,
    sV_layout: cute.ComposedLayout,
    gmem_tiled_copy_QK: cute.TiledCopy,
    gmem_tiled_copy_VdO: cute.TiledCopy,
    tiled_mma_sdp: cute.TiledMma,
    tiled_mma_dkv: cute.TiledMma,
    SharedStorage: cutlass.Constexpr,
  ):
    tidx, _, _ = cute.arch.thread_idx()
    block_n, block_hb, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    batch_idx = block_hb // mK.shape[2]
    kv_head_idx = block_hb - batch_idx * mK.shape[2]
    seqlen_q = mQ.shape[1]
    seqlen_k = mK.shape[1]
    num_m_blocks = cute.ceil_div(seqlen_q, self.tile_m)
    softmax_scale_log2 = softmax_scale * 1.4426950408889634

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sQ_all = storage.sQ.get_tensor(sQ_layout)
    sdO_all = storage.sdO.get_tensor(sdO_layout)
    sK_all = storage.sK.get_tensor(sK_layout)
    sV_all = storage.sV.get_tensor(sV_layout)

    ns_Q = const_expr(self.num_stages_Q)
    ns_dO = const_expr(self.num_stages_dO)
    ND = const_expr(self.num_d_chunks)
    # Per-ring wait counts. Each steady iter issues two separate commit_group()
    # calls (one per ring). For a ring with ns=1 the just-issued load is the
    # data consumed this iter (inline), so the wait after it must drain it
    # (== 0). For ns>=2 the just-issued load is a future prefetch and can stay
    # in flight together with the other ring's prefetch, giving total
    # (ns_Q-1)+(ns_dO-1) pending groups allowed.
    wait_count_q_active = const_expr((ns_Q - 1) +
                                     (ns_dO - 1) if ns_Q >= 2 else 0)
    wait_count_do_active = const_expr((ns_Q - 1) +
                                      (ns_dO - 1) if ns_dO >= 2 else 0)

    # Per-stage smem slices; ldmatrix / mma fragments share atom across stages.
    sQ_stage = [sQ_all[None, None, s] for s in range(self.num_stages_Q)]
    sdO_stage = [sdO_all[None, None, s] for s in range(self.num_stages_dO)]
    sK_stage = [sK_all[None, None, s] for s in range(self.num_stages_Q)]
    sV_stage = [sV_all[None, None, s] for s in range(self.num_stages_dO)]
    sQt_stage = [layout_utils.transpose_view(t) for t in sQ_stage]
    sdOt_stage = [layout_utils.transpose_view(t) for t in sdO_stage]

    # SdP uses swapAB so S^T / dP^T accumulators have per-warp rows aligned to
    # tile_n, matching dKV's A-operand layout for in-register P / dS transfer.
    thr_mma_sdp = tiled_mma_sdp.get_slice(tidx)
    thr_mma_dkv = tiled_mma_dkv.get_slice(tidx)
    tSrQ = [
      utils.mma_make_fragment_A(t, thr_mma_sdp, swapAB=True) for t in sQ_stage
    ]
    tSrK = [
      utils.mma_make_fragment_B(t, thr_mma_sdp, swapAB=True) for t in sK_stage
    ]
    tdPrdO = [
      utils.mma_make_fragment_A(t, thr_mma_sdp, swapAB=True) for t in sdO_stage
    ]
    tdPrV = [
      utils.mma_make_fragment_B(t, thr_mma_sdp, swapAB=True) for t in sV_stage
    ]
    tdVrdO = [utils.mma_make_fragment_B(t, thr_mma_dkv) for t in sdOt_stage]
    tdKrQ = [utils.mma_make_fragment_B(t, thr_mma_dkv) for t in sQt_stage]
    acc_shape_dK = thr_mma_dkv.partition_shape_C((self.tile_n, self.d_chunk))

    smem_copy_atom = cute.make_copy_atom(
      warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype
    )
    smem_copy_atom_t = cute.make_copy_atom(
      warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self.dtype
    )
    smem_thr_copy_Q = utils.make_tiled_copy_A(
      smem_copy_atom, tiled_mma_sdp, swapAB=True
    ).get_slice(tidx)
    smem_thr_copy_K = utils.make_tiled_copy_B(
      smem_copy_atom, tiled_mma_sdp, swapAB=True
    ).get_slice(tidx)
    smem_thr_copy_dO = utils.make_tiled_copy_A(
      smem_copy_atom, tiled_mma_sdp, swapAB=True
    ).get_slice(tidx)
    smem_thr_copy_V = utils.make_tiled_copy_B(
      smem_copy_atom, tiled_mma_sdp, swapAB=True
    ).get_slice(tidx)
    smem_thr_copy_dOt = utils.make_tiled_copy_B(
      smem_copy_atom_t, tiled_mma_dkv
    ).get_slice(tidx)
    smem_thr_copy_Qt = utils.make_tiled_copy_B(smem_copy_atom_t,
                                               tiled_mma_dkv).get_slice(tidx)
    tSsQ = [smem_thr_copy_Q.partition_S(t) for t in sQ_stage]
    tSsK = [smem_thr_copy_K.partition_S(t) for t in sK_stage]
    tdPsdO = [smem_thr_copy_dO.partition_S(t) for t in sdO_stage]
    tdPsV = [smem_thr_copy_V.partition_S(t) for t in sV_stage]
    tdVsdOt = [smem_thr_copy_dOt.partition_S(t) for t in sdOt_stage]
    tdKsQt = [smem_thr_copy_Qt.partition_S(t) for t in sQt_stage]
    # reg->gmem direct STG: atom width is capped at the MMA C partition's
    # per-thread contiguous extent. For SM80 m16n8k16 with fp16/bf16 that is
    # only 2 elements (= 32 bit), so STG.32 here is the maximum achievable on
    # this path. Upgrading to STG.128 would require a reg->smem(swizzled)->
    # gmem epilogue (see ``_fwd_generic_sm80.store_O_and_lse`` for the
    # reference pattern); tracked as a separate optimisation.
    # Alt approach: ``sync_store_o_r2g`` in ``csrc/cuffpa/prefill.cuh`` uses
    # ``__shfl_sync(..., width=4)`` to gather the 4 lanes of an MMA quad's
    # 32-bit fragments into ``lane%4==0`` and emit one ``st.global.v4``. The
    # CuTeDSL equivalent (``cute.arch.shuffle_sync`` / ``utils.shuffle_sync``
    # + 128-bit CopyUniversalOp atom guarded by a lane predicate) saves smem
    # and one barrier vs the staged epilogue, but the expected speedup over
    # an already smem-staged STG.128 is small.
    gmem_copy_atom_DKV = cute.make_copy_atom(
      cute.nvgpu.CopyUniversalOp(),
      self.dkdv_storage_dtype,
      num_bits_per_copy=2 * self.dkdv_storage_dtype.width
    )
    acc2g_thr_copy_DKV = cute.make_tiled_copy_C(
      gmem_copy_atom_DKV, tiled_mma_dkv
    ).get_slice(tidx)

    for q_head_group in cutlass.range(self.qhead_per_kvhead, unroll=1):
      q_head_idx = kv_head_idx * self.qhead_per_kvhead + q_head_group
      for m_block in cutlass.range(num_m_blocks, unroll=1):
        if const_expr(not self.is_causal
                      ) or (seqlen_q == seqlen_k and m_block >= block_n):
          acc_shape_S = thr_mma_sdp.partition_shape_C(
            (self.tile_n, self.tile_m)
          )
          acc_S = cute.make_rmem_tensor(acc_shape_S, Float32)
          acc_dP = cute.make_rmem_tensor(acc_shape_S, Float32)
          acc_S.fill(0.0)
          acc_dP.fill(0.0)

          # Phase A prologue: prefetch ns-1 QK pairs and ns-1 dOV pairs so the
          # steady-state inner loop can overlap load(d+ns-1) with mma(d).
          for s in cutlass.range_constexpr(ns_Q - 1):
            if s < ND:
              gQ = cute.local_tile(
                mQ[batch_idx, None, q_head_idx, None],
                (self.tile_m, self.d_chunk),
                (m_block, s),
              )
              gK = cute.local_tile(
                mK[batch_idx, None, kv_head_idx, None],
                (self.tile_n, self.d_chunk),
                (block_n, s),
              )
              self.load_m_tile(
                gmem_tiled_copy_QK.get_slice(tidx), gQ, sQ_stage[s], m_block,
                seqlen_q
              )
              self.load_n_tile(
                gmem_tiled_copy_QK.get_slice(tidx), gK, sK_stage[s], block_n,
                seqlen_k
              )
            cute.arch.cp_async_commit_group()
          for s in cutlass.range_constexpr(ns_dO - 1):
            if s < ND:
              gdO = cute.local_tile(
                mdO[batch_idx, None, q_head_idx, None],
                (self.tile_m, self.d_chunk),
                (m_block, s),
              )
              gV = cute.local_tile(
                mV[batch_idx, None, kv_head_idx, None],
                (self.tile_n, self.d_chunk),
                (block_n, s),
              )
              self.load_m_tile(
                gmem_tiled_copy_VdO.get_slice(tidx), gdO, sdO_stage[s], m_block,
                seqlen_q
              )
              self.load_n_tile(
                gmem_tiled_copy_VdO.get_slice(tidx), gV, sV_stage[s], block_n,
                seqlen_k
              )
            cute.arch.cp_async_commit_group()

          for score_d_block in cutlass.range_constexpr(self.num_d_chunks):
            stage_q = score_d_block % ns_Q
            stage_do = score_d_block % ns_dO
            # QK prefetch for (score_d_block + ns_Q - 1) into ring slot
            # (score_d_block + ns_Q - 1) % ns_Q; commit even if no load.
            prefetch_d_qk = score_d_block + ns_Q - 1
            if prefetch_d_qk < ND:
              prefetch_slot_qk = prefetch_d_qk % ns_Q
              gQp = cute.local_tile(
                mQ[batch_idx, None, q_head_idx, None],
                (self.tile_m, self.d_chunk),
                (m_block, prefetch_d_qk),
              )
              gKp = cute.local_tile(
                mK[batch_idx, None, kv_head_idx, None],
                (self.tile_n, self.d_chunk),
                (block_n, prefetch_d_qk),
              )
              self.load_m_tile(
                gmem_tiled_copy_QK.get_slice(tidx), gQp,
                sQ_stage[prefetch_slot_qk], m_block, seqlen_q
              )
              self.load_n_tile(
                gmem_tiled_copy_QK.get_slice(tidx), gKp,
                sK_stage[prefetch_slot_qk], block_n, seqlen_k
              )
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(wait_count_q_active)
            cute.arch.barrier()
            self.zero_m_tail(sQ_stage[stage_q], m_block, seqlen_q, tidx)
            self.zero_n_tail(sK_stage[stage_q], block_n, seqlen_k, tidx)
            cute.arch.barrier()
            sm80_utils.gemm(
              thr_mma_sdp,
              acc_S,
              tSrQ[stage_q],
              tSrK[stage_q],
              tSsQ[stage_q],
              tSsK[stage_q],
              smem_thr_copy_Q,
              smem_thr_copy_K,
              swap_AB=True
            )
            cute.arch.barrier()

            # dOV prefetch for (score_d_block + ns_dO - 1); commit always.
            prefetch_d_dov = score_d_block + ns_dO - 1
            if prefetch_d_dov < ND:
              prefetch_slot_dov = prefetch_d_dov % ns_dO
              gdOp = cute.local_tile(
                mdO[batch_idx, None, q_head_idx, None],
                (self.tile_m, self.d_chunk),
                (m_block, prefetch_d_dov),
              )
              gVp = cute.local_tile(
                mV[batch_idx, None, kv_head_idx, None],
                (self.tile_n, self.d_chunk),
                (block_n, prefetch_d_dov),
              )
              self.load_m_tile(
                gmem_tiled_copy_VdO.get_slice(tidx), gdOp,
                sdO_stage[prefetch_slot_dov], m_block, seqlen_q
              )
              self.load_n_tile(
                gmem_tiled_copy_VdO.get_slice(tidx), gVp,
                sV_stage[prefetch_slot_dov], block_n, seqlen_k
              )
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(wait_count_do_active)
            cute.arch.barrier()
            self.zero_m_tail(sdO_stage[stage_do], m_block, seqlen_q, tidx)
            self.zero_n_tail(sV_stage[stage_do], block_n, seqlen_k, tidx)
            cute.arch.barrier()
            sm80_utils.gemm(
              thr_mma_sdp,
              acc_dP,
              tdPrdO[stage_do],
              tdPrV[stage_do],
              tdPsdO[stage_do],
              tdPsV[stage_do],
              smem_thr_copy_dO,
              smem_thr_copy_V,
              swap_AB=True
            )
            cute.arch.barrier()

          # Drain any QK / dOV groups still in flight before reusing rings for
          # Phase B.
          cute.arch.cp_async_wait_group(0)
          cute.arch.barrier()

          self.make_p_and_ds(
            acc_S,
            acc_dP,
            thr_mma_sdp,
            mLSElog2,
            mD,
            softmax_scale,
            softmax_scale_log2,
            batch_idx,
            q_head_idx,
            m_block,
            block_n,
            seqlen_q,
            seqlen_k,
          )
          rP = cute.make_fragment_like(acc_S, self.dtype)
          rdS = cute.make_fragment_like(acc_dP, self.dtype)
          rP.store(acc_S.load().to(self.dtype))
          rdS.store(acc_dP.load().to(self.dtype))
          tdVrP = layout_utils.reshape_acc_to_frgA(rP)
          tdKrdS = layout_utils.reshape_acc_to_frgA(rdS)

          # Phase B prologue: prefetch dO_out and Q_out ns-1 ahead. Rings were
          # drained by Phase A so all stages are free for reuse.
          for s in cutlass.range_constexpr(ns_dO - 1):
            if s < ND:
              gdO_pre = cute.local_tile(
                mdO[batch_idx, None, q_head_idx, None],
                (self.tile_m, self.d_chunk),
                (m_block, s),
              )
              self.load_m_tile(
                gmem_tiled_copy_VdO.get_slice(tidx), gdO_pre, sdO_stage[s],
                m_block, seqlen_q
              )
            cute.arch.cp_async_commit_group()
          for s in cutlass.range_constexpr(ns_Q - 1):
            if s < ND:
              gQ_pre = cute.local_tile(
                mQ[batch_idx, None, q_head_idx, None],
                (self.tile_m, self.d_chunk),
                (m_block, s),
              )
              self.load_m_tile(
                gmem_tiled_copy_QK.get_slice(tidx), gQ_pre, sQ_stage[s],
                m_block, seqlen_q
              )
            cute.arch.cp_async_commit_group()

          for d_block in cutlass.range_constexpr(self.num_d_chunks):
            stage_q = d_block % ns_Q
            stage_do = d_block % ns_dO
            acc_dK = cute.make_rmem_tensor(acc_shape_dK, Float32)
            acc_dV = cute.make_rmem_tensor(acc_shape_dK, Float32)
            acc_dK.fill(0.0)
            acc_dV.fill(0.0)
            # dV side: prefetch dO_out for d_block + ns_dO - 1.
            prefetch_d_do = d_block + ns_dO - 1
            if prefetch_d_do < ND:
              prefetch_slot_do = prefetch_d_do % ns_dO
              gdOp = cute.local_tile(
                mdO[batch_idx, None, q_head_idx, None],
                (self.tile_m, self.d_chunk),
                (m_block, prefetch_d_do),
              )
              self.load_m_tile(
                gmem_tiled_copy_VdO.get_slice(tidx), gdOp,
                sdO_stage[prefetch_slot_do], m_block, seqlen_q
              )
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(wait_count_do_active)
            cute.arch.barrier()
            self.zero_m_tail(sdO_stage[stage_do], m_block, seqlen_q, tidx)
            cute.arch.barrier()

            sm80_utils.gemm_rs(
              thr_mma_dkv,
              acc_dV,
              tdVrP,
              tdVrdO[stage_do],
              tdVsdOt[stage_do],
              smem_thr_copy_dOt,
            )
            cute.arch.barrier()

            # dK side: prefetch Q_out for d_block + ns_Q - 1.
            prefetch_d_q = d_block + ns_Q - 1
            if prefetch_d_q < ND:
              prefetch_slot_q = prefetch_d_q % ns_Q
              gQp = cute.local_tile(
                mQ[batch_idx, None, q_head_idx, None],
                (self.tile_m, self.d_chunk),
                (m_block, prefetch_d_q),
              )
              self.load_m_tile(
                gmem_tiled_copy_QK.get_slice(tidx), gQp,
                sQ_stage[prefetch_slot_q], m_block, seqlen_q
              )
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(wait_count_q_active)
            cute.arch.barrier()
            self.zero_m_tail(sQ_stage[stage_q], m_block, seqlen_q, tidx)
            cute.arch.barrier()
            sm80_utils.gemm_rs(
              thr_mma_dkv,
              acc_dK,
              tdKrdS,
              tdKrQ[stage_q],
              tdKsQt[stage_q],
              smem_thr_copy_Qt,
            )
            cute.arch.barrier()
            self.store_dkdv_tile(
              acc_dK,
              acc_dV,
              mdK,
              mdV,
              acc2g_thr_copy_DKV,
              gmem_copy_atom_DKV,
              batch_idx,
              kv_head_idx,
              block_n,
              d_block,
              seqlen_k,
              True,
            )
            cute.arch.barrier()

          # Drain Phase B rings before the next m_block reuses stages.
          cute.arch.cp_async_wait_group(0)
          cute.arch.barrier()

  @cute.jit
  def make_p_and_ds(
    self,
    acc_S: cute.Tensor,
    acc_dP: cute.Tensor,
    thr_mma: cute.TiledMma,
    mLSElog2: cute.Tensor,
    mD: cute.Tensor,
    softmax_scale: Float32,
    softmax_scale_log2: Float32,
    batch_idx: Int32,
    q_head_idx: Int32,
    m_block: Int32,
    n_block: Int32,
    seqlen_q: Int32,
    seqlen_k: Int32,
  ) -> None:
    # acc_S / acc_dP were produced with SdP swapAB, so pass transpose=True to
    # recover (M, N) row/col indexing matching the identity tile below.
    acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S, transpose=True)
    acc_dP_mn = layout_utils.reshape_acc_to_mn(acc_dP, transpose=True)
    cS = cute.make_identity_tensor((self.tile_n, self.tile_m))
    tScS_mn = layout_utils.reshape_acc_to_mn(
      thr_mma.partition_C(cS), transpose=True
    )
    kv_offset = seqlen_k - seqlen_q
    tile_in_bounds = ((m_block + 1) * self.tile_m
                      <= seqlen_q) and ((n_block + 1) * self.tile_n <= seqlen_k)
    if const_expr(not self.is_causal):
      if tile_in_bounds:
        for row in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
          q_idx = m_block * self.tile_m + tScS_mn[row, 0][1]
          p_row = cute.math.exp2(
            acc_S_mn[row, None].load() * softmax_scale_log2 -
            mLSElog2[batch_idx, q_head_idx, q_idx],
            fastmath=True,
          )
          acc_S_mn[row, None].store(p_row)
          acc_dP_mn[row, None].store(
            p_row *
            (acc_dP_mn[row, None].load() - mD[batch_idx, q_head_idx, q_idx]) *
            softmax_scale
          )
      else:
        for row in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
          for col in cutlass.range(
            cute.size(tScS_mn.shape[1]), unroll_full=True
          ):
            q_idx = m_block * self.tile_m + tScS_mn[row, col][1]
            k_idx = n_block * self.tile_n + tScS_mn[row, col][0]
            valid = (q_idx < seqlen_q) and (k_idx < seqlen_k)
            if valid:
              p = cute.math.exp2(
                acc_S_mn[row, col] * softmax_scale_log2 -
                mLSElog2[batch_idx, q_head_idx, q_idx],
                fastmath=True,
              )
              acc_S_mn[row, col] = p
              acc_dP_mn[row, col] = p * (
                acc_dP_mn[row, col] - mD[batch_idx, q_head_idx, q_idx]
              ) * softmax_scale
            else:
              acc_S_mn[row, col] = 0.0
              acc_dP_mn[row, col] = 0.0
    else:
      for row in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
        for col in cutlass.range(cute.size(tScS_mn.shape[1]), unroll_full=True):
          q_idx = m_block * self.tile_m + tScS_mn[row, col][1]
          k_idx = n_block * self.tile_n + tScS_mn[row, col][0]
          valid = (q_idx < seqlen_q) and (k_idx < seqlen_k)
          valid = valid and (k_idx <= q_idx + kv_offset)
          if valid:
            p = cute.math.exp2(
              acc_S_mn[row, col] * softmax_scale_log2 -
              mLSElog2[batch_idx, q_head_idx, q_idx],
              fastmath=True,
            )
            acc_S_mn[row, col] = p
            acc_dP_mn[row, col] = p * (
              acc_dP_mn[row, col] - mD[batch_idx, q_head_idx, q_idx]
            ) * softmax_scale
          else:
            acc_S_mn[row, col] = 0.0
            acc_dP_mn[row, col] = 0.0

  @cute.jit
  def load_m_tile(
    self,
    gmem_thr_copy: cute.TiledCopy,
    gmem_tile: cute.Tensor,
    smem_tile: cute.Tensor,
    block: Int32,
    seqlen: Int32,
  ) -> None:
    tGsG = gmem_thr_copy.partition_S(gmem_tile)
    tSsS = gmem_thr_copy.partition_D(smem_tile)
    cG = cute.make_identity_tensor((self.tile_m, self.d_chunk))
    tGcG = gmem_thr_copy.partition_S(cG)
    t0GcG = gmem_thr_copy.get_slice(0).partition_S(cG)
    for m in cutlass.range_constexpr(cute.size(tSsS.shape[1])):
      if t0GcG[0, m, 0][0] < seqlen - block * self.tile_m - tGcG[0][0]:
        cute.copy(gmem_thr_copy, tGsG[None, m, None], tSsS[None, m, None])

  @cute.jit
  def load_n_tile(
    self,
    gmem_thr_copy: cute.TiledCopy,
    gmem_tile: cute.Tensor,
    smem_tile: cute.Tensor,
    block: Int32,
    seqlen: Int32,
  ) -> None:
    tGsG = gmem_thr_copy.partition_S(gmem_tile)
    tSsS = gmem_thr_copy.partition_D(smem_tile)
    cG = cute.make_identity_tensor((self.tile_n, self.d_chunk))
    tGcG = gmem_thr_copy.partition_S(cG)
    t0GcG = gmem_thr_copy.get_slice(0).partition_S(cG)
    for n in cutlass.range_constexpr(cute.size(tSsS.shape[1])):
      if t0GcG[0, n, 0][0] < seqlen - block * self.tile_n - tGcG[0][0]:
        cute.copy(gmem_thr_copy, tGsG[None, n, None], tSsS[None, n, None])

  @cute.jit
  def zero_m_tail(
    self,
    smem_tile: cute.Tensor,
    block: Int32,
    seqlen: Int32,
    tidx: Int32,
  ) -> None:
    valid_rows = seqlen - block * self.tile_m
    if valid_rows < self.tile_m:
      tail_rows = self.tile_m - valid_rows
      for row_offset in cutlass.range(
        tidx, tail_rows, self.num_threads, unroll=1
      ):
        smem_tile[valid_rows + row_offset, None].fill(0.0)

  @cute.jit
  def zero_n_tail(
    self,
    smem_tile: cute.Tensor,
    block: Int32,
    seqlen: Int32,
    tidx: Int32,
  ) -> None:
    valid_rows = seqlen - block * self.tile_n
    if valid_rows < self.tile_n:
      tail_rows = self.tile_n - valid_rows
      for row_offset in cutlass.range(
        tidx, tail_rows, self.num_threads, unroll=1
      ):
        smem_tile[valid_rows + row_offset, None].fill(0.0)

  @cute.jit
  def store_dkdv_tile(
    self,
    acc_dK: cute.Tensor,
    acc_dV: cute.Tensor,
    mdK: cute.Tensor,
    mdV: cute.Tensor,
    acc2g_thr_copy_DKV: cute.TiledCopy,
    gmem_copy_atom_DKV: cute.CopyAtom,
    batch_idx: Int32,
    kv_head_idx: Int32,
    n_block: Int32,
    d_block: Int32,
    seqlen_k: Int32,
    add_to_existing: cutlass.Boolean,
  ) -> None:
    gDK = cute.local_tile(
      mdK[batch_idx, None, kv_head_idx, None],
      (self.tile_n, self.d_chunk),
      (n_block, d_block),
    )
    gDV = cute.local_tile(
      mdV[batch_idx, None, kv_head_idx, None],
      (self.tile_n, self.d_chunk),
      (n_block, d_block),
    )
    tDgDK = acc2g_thr_copy_DKV.partition_D(gDK)
    tDgDV = acc2g_thr_copy_DKV.partition_D(gDV)
    tRgDK = acc2g_thr_copy_DKV.retile(acc_dK)
    tRgDV = acc2g_thr_copy_DKV.retile(acc_dV)
    # When ``dkdv_storage_dtype`` is fp32 the in-register store fragment is
    # already fp32, so the trailing ``.to(self.dtype)`` cast becomes a no-op
    # and we skip rounding to keep cross-tile accumulation precision.
    rDK = cute.make_fragment_like(tRgDK, self.dkdv_storage_dtype)
    rDV = cute.make_fragment_like(tRgDV, self.dkdv_storage_dtype)
    if add_to_existing:
      rOldDK = cute.make_fragment_like(tRgDK, self.dkdv_storage_dtype)
      rOldDV = cute.make_fragment_like(tRgDV, self.dkdv_storage_dtype)
      cute.copy(gmem_copy_atom_DKV, tDgDK, rOldDK)
      cute.copy(gmem_copy_atom_DKV, tDgDV, rOldDV)
      rDK.store(
        (tRgDK.load() + rOldDK.load().to(Float32)).to(self.dkdv_storage_dtype)
      )
      rDV.store(
        (tRgDV.load() + rOldDV.load().to(Float32)).to(self.dkdv_storage_dtype)
      )
    else:
      rDK.store(tRgDK.load().to(self.dkdv_storage_dtype))
      rDV.store(tRgDV.load().to(self.dkdv_storage_dtype))
    cute.copy(gmem_copy_atom_DKV, rDK, tDgDK)
    cute.copy(gmem_copy_atom_DKV, rDV, tDgDV)


__all__ = ["FFPAAttnBwdDKDVSm80SplitDGeneric"]

# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# Adapted from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/flash_fwd.py
# FFPA adaptation: SM80/SM89 large-head-dim Split-D forward with 64-wide D chunks.

from typing import Callable, Optional, Type

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cutlass_dsl import BaseDSL
from cutlass.cute.nvgpu import cpasync, warp

from quack import layout_utils

from . import utils
from ._utils import SM80_SPLIT_D_CHUNK
from .utils import ampere_helpers as sm80_utils
from .utils.block_info import BlockInfo
from .utils.cute_dsl_utils import assume_tensor_aligned
from .utils.seqlen_info import SeqlenInfoQK
from .utils.softmax import Softmax
from .utils.tile_scheduler import (
  SingleTileScheduler,
  SingleTileVarlenScheduler,
  TileSchedulerArguments,
)


class FFPAAttnFwdSm80SplitD:
  """SM80/SM89 FFPA forward kernel with a 64-wide Split-D inner loop.

  One CTA owns one Q-row block for one query head. For each KV tile it first
  reconstructs the full QK score tile by reducing over 64-wide Q/K chunks,
  then reuses the same softmax tile for every 64-wide V slice. This mirrors
  the Triton FFPA forward path: ``R_D[j] = alpha * R_D[j] + P @ V_j``.
  """

  def __init__(
    self,
    dtype: Type[cutlass.Numeric],
    head_dim: int,
    head_dim_v: Optional[int] = None,
    qhead_per_kvhead: int = 1,
    is_causal: bool = False,
    pack_gqa: bool = False,
    tile_m: int = 64,
    tile_n: int = 128,
    num_threads: int = 128,
  ):
    self.dtype = dtype
    self.head_dim = head_dim
    self.head_dim_v = head_dim if head_dim_v is None else head_dim_v
    self.d_chunk = SM80_SPLIT_D_CHUNK
    self.num_d_chunks = head_dim // self.d_chunk
    self.num_v_chunks = self.head_dim_v // self.d_chunk
    self.tile_hdim = self.d_chunk
    self.tile_hdimv = self.d_chunk
    self.check_hdim_oob = False
    self.check_hdim_v_oob = False
    self.qhead_per_kvhead = qhead_per_kvhead
    self.is_causal = is_causal
    self.is_local = False
    self.pack_gqa = pack_gqa
    self.tile_m = tile_m
    self.tile_n = tile_n
    self.num_threads = num_threads
    self.num_stages = 1
    self.arch = BaseDSL._get_dsl().get_arch_enum()

  def _check_type(
    self,
    mQ_type: Type[cutlass.Numeric],
    mK_type: Type[cutlass.Numeric],
    mV_type: Type[cutlass.Numeric],
    mO_type: Type[cutlass.Numeric],
    mLSE_type: Type[cutlass.Numeric] | None,
    mCuSeqlensQ_type: Type[cutlass.Numeric] | None,
    mCuSeqlensK_type: Type[cutlass.Numeric] | None,
  ) -> None:
    if const_expr(not (mQ_type == mK_type == mV_type == mO_type)):
      raise TypeError("All tensors must have the same data type")
    if const_expr(mQ_type not in [cutlass.Float16, cutlass.BFloat16]):
      raise TypeError("Only Float16 or BFloat16 is supported")
    if const_expr(mLSE_type not in [None, Float32]):
      raise TypeError("LSE tensor must be Float32")
    if const_expr(mCuSeqlensQ_type not in [None, cutlass.Int32]):
      raise TypeError("cu_seqlens_q tensor must be Int32")
    if const_expr(mCuSeqlensK_type not in [None, cutlass.Int32]):
      raise TypeError("cu_seqlens_k tensor must be Int32")
    assert mQ_type == self.dtype

  def _setup_attributes(self) -> None:
    sQ_layout_atom = sm80_utils.get_smem_layout_atom(self.dtype, self.tile_hdim)
    sK_layout_atom = sQ_layout_atom
    sV_layout_atom = sm80_utils.get_smem_layout_atom(
      self.dtype, self.tile_hdimv
    )
    self.sQ_layout = cute.tile_to_shape(
      sQ_layout_atom, (self.tile_m, self.tile_hdim), (0, 1)
    )
    self.sK_layout = cute.tile_to_shape(
      sK_layout_atom, (self.tile_n, self.tile_hdim), (0, 1)
    )
    self.sV_layout = cute.tile_to_shape(
      sV_layout_atom, (self.tile_n, self.tile_hdimv), (0, 1)
    )
    self.sO_layout = cute.tile_to_shape(
      sV_layout_atom, (self.tile_m, self.tile_hdimv), (0, 1)
    )

    universal_copy_bits = 128
    async_copy_elems = universal_copy_bits // self.dtype.width
    atom_async_copy = cute.make_copy_atom(
      cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
      self.dtype,
      num_bits_per_copy=universal_copy_bits,
    )
    atom_universal_copy = cute.make_copy_atom(
      cute.nvgpu.CopyUniversalOp(),
      self.dtype,
      num_bits_per_copy=universal_copy_bits
    )
    tQK_shape_dim_1 = sQ_layout_atom.outer.shape[1] // async_copy_elems
    tQ_layout = cute.make_ordered_layout(
      (self.num_threads // tQK_shape_dim_1, tQK_shape_dim_1), order=(1, 0)
    )
    tK_layout = tQ_layout
    tV_shape_dim_1 = sV_layout_atom.outer.shape[1] // async_copy_elems
    tV_layout = cute.make_ordered_layout(
      (self.num_threads // tV_shape_dim_1, tV_shape_dim_1), order=(1, 0)
    )
    tO_layout = tV_layout
    vQKV_layout = cute.make_layout((1, async_copy_elems))
    self.gmem_tiled_copy_Q = cute.make_tiled_copy_tv(
      atom_async_copy, tQ_layout, vQKV_layout
    )
    self.gmem_tiled_copy_K = cute.make_tiled_copy_tv(
      atom_async_copy, tK_layout, vQKV_layout
    )
    self.gmem_tiled_copy_V = cute.make_tiled_copy_tv(
      atom_async_copy, tV_layout, vQKV_layout
    )
    self.gmem_tiled_copy_O = cute.make_tiled_copy_tv(
      atom_universal_copy, tO_layout, vQKV_layout
    )

  def _get_tiled_mma(self):
    tiled_mma_qk = cute.make_tiled_mma(
      warp.MmaF16BF16Op(self.dtype, Float32, (16, 8, 16)),
      (self.num_threads // 32, 1, 1),
      permutation_mnk=(self.num_threads // 32 * 16, 16, 16),
    )
    tiled_mma_pv = cute.make_tiled_mma(
      warp.MmaF16BF16Op(self.dtype, Float32, (16, 8, 16)),
      (self.num_threads // 32, 1, 1),
      permutation_mnk=(self.num_threads // 32 * 16, 16, 16),
    )
    return tiled_mma_qk, tiled_mma_pv

  def _get_shared_storage_cls(self):
    sQ_struct, sK_struct, sV_struct = [
      cute.struct.Align[cute.struct.MemRange[self.dtype,
                                             cute.cosize(layout)], 1024]
      for layout in (self.sQ_layout, self.sK_layout, self.sV_layout)
    ]
    cosize_sQV = max(cute.cosize(self.sQ_layout), cute.cosize(self.sV_layout))
    sQV_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sQV],
                                   1024]

    @cute.struct
    class SharedStorage:
      sQV: sQV_struct
      sK: sK_struct

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
    stream: cuda.CUstream = None,
  ):
    """Configure and launch the SM80 Split-D forward kernel."""
    self._check_type(
      *(
        t.element_type if t is not None else None
        for t in (mQ, mK, mV, mO, mLSE, mCuSeqlensQ, mCuSeqlensK)
      )
    )
    tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()
    self.num_producer_threads = self.num_threads
    self.num_Q_load_threads = self.num_threads
    self.num_epilogue_threads = self.num_threads
    self._setup_attributes()
    SharedStorage = self._get_shared_storage_cls()
    mQ, mK, mV, mO = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]

    qo_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None
                                              ) else [0, 2, 1]
    kv_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensK is None
                                              ) else [0, 2, 1]
    mQ, mO = [
      cute.make_tensor(t.iterator, cute.select(t.layout, mode=qo_transpose))
      for t in (mQ, mO)
    ]
    mK, mV = [
      cute.make_tensor(t.iterator, cute.select(t.layout, mode=kv_transpose))
      for t in (mK, mV)
    ]
    if const_expr(mLSE is not None):
      lse_transpose = [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
      mLSE = cute.make_tensor(
        mLSE.iterator, cute.select(mLSE.layout, mode=lse_transpose)
      )

    TileScheduler = SingleTileVarlenScheduler if const_expr(
      mCuSeqlensQ is not None
    ) else SingleTileScheduler
    num_batch = mCuSeqlensQ.shape[0] - 1 if const_expr(mCuSeqlensQ is not None
                                                       ) else mQ.shape[3]
    tile_sched_args = TileSchedulerArguments(
      num_block=cute.ceil_div(mQ.shape[0], self.tile_m),
      num_head=cute.size(mQ.shape[2]),
      num_batch=num_batch,
      num_splits=1,
      seqlen_k=cute.size(mK.shape[0]),
      headdim=mQ.shape[1],
      headdim_v=mV.shape[1],
      total_q=cute.size(mQ.shape[0]) if const_expr(mCuSeqlensQ is not None) else
      cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
      tile_shape_mn=(self.tile_m, self.tile_n),
      qhead_per_kvhead_packgqa=1,
      mCuSeqlensQ=mCuSeqlensQ,
    )
    tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
    grid_dim = TileScheduler.get_grid_shape(tile_sched_params)
    softmax_scale_log2, softmax_scale = utils.compute_softmax_scale_log2(
      softmax_scale, None
    )

    self.kernel(
      mQ,
      mK,
      mV,
      mO,
      mLSE,
      mCuSeqlensQ,
      mCuSeqlensK,
      softmax_scale_log2,
      softmax_scale,
      self.sQ_layout,
      self.sK_layout,
      self.sV_layout,
      self.sO_layout,
      self.gmem_tiled_copy_Q,
      self.gmem_tiled_copy_K,
      self.gmem_tiled_copy_V,
      self.gmem_tiled_copy_O,
      tiled_mma_qk,
      tiled_mma_pv,
      SharedStorage,
      tile_sched_params,
      TileScheduler,
    ).launch(
      grid=grid_dim,
      block=[self.num_threads, 1, 1],
      smem=SharedStorage.size_in_bytes(),
      stream=stream
    )

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
    softmax_scale_log2: Float32,
    softmax_scale: Optional[Float32],
    sQ_layout: cute.ComposedLayout,
    sK_layout: cute.ComposedLayout,
    sV_layout: cute.ComposedLayout,
    sO_layout: cute.ComposedLayout,
    gmem_tiled_copy_Q: cute.TiledCopy,
    gmem_tiled_copy_K: cute.TiledCopy,
    gmem_tiled_copy_V: cute.TiledCopy,
    gmem_tiled_copy_O: cute.TiledCopy,
    tiled_mma_qk: cute.TiledMma,
    tiled_mma_pv: cute.TiledMma,
    SharedStorage: cutlass.Constexpr,
    tile_sched_params,
    TileScheduler: cutlass.Constexpr[Callable],
  ):
    tidx, _, _ = cute.arch.thread_idx()
    tile_scheduler = TileScheduler.create(tile_sched_params)
    work_tile = tile_scheduler.initial_work_tile_info()
    m_block, head_idx, batch_idx, _ = work_tile.tile_idx

    seqlen = SeqlenInfoQK.create(
      batch_idx=batch_idx,
      seqlen_q_static=mQ.shape[0],
      seqlen_k_static=mK.shape[0],
      mCuSeqlensQ=mCuSeqlensQ,
      mCuSeqlensK=mCuSeqlensK,
      tile_m=self.tile_m,
      tile_n=self.tile_n,
    )
    block_info = BlockInfo(
      self.tile_m,
      self.tile_n,
      self.is_causal,
      False,
      qhead_per_kvhead_packgqa=1,
    )
    _n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)

    num_head_kv = head_idx // self.qhead_per_kvhead
    if const_expr(not seqlen.has_cu_seqlens_q):
      mQ_cur = mQ[None, None, head_idx, batch_idx]
      mO_cur = mO[None, None, head_idx, batch_idx]
    else:
      mQ_cur = cute.domain_offset((seqlen.offset_q, 0), mQ[None, None,
                                                           head_idx])
      mO_cur = cute.domain_offset((seqlen.offset_q, 0), mO[None, None,
                                                           head_idx])
    if const_expr(not seqlen.has_cu_seqlens_k):
      mK_cur = mK[None, None, num_head_kv, batch_idx]
      mV_cur = mV[None, None, num_head_kv, batch_idx]
    else:
      mK_cur = cute.domain_offset((seqlen.offset_k, 0), mK[None, None,
                                                           num_head_kv])
      mV_cur = cute.domain_offset((seqlen.offset_k, 0), mV[None, None,
                                                           num_head_kv])

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sQ = storage.sQV.get_tensor(sQ_layout)
    sK = storage.sK.get_tensor(sK_layout)
    sV = cute.make_tensor(
      cute.recast_ptr(sQ.iterator, dtype=self.dtype), sV_layout
    )
    sVt = layout_utils.transpose_view(sV)

    thr_mma_qk = tiled_mma_qk.get_slice(tidx)
    thr_mma_pv = tiled_mma_pv.get_slice(tidx)
    tSrQ = thr_mma_qk.make_fragment_A(thr_mma_qk.partition_A(sQ))
    tSrK = thr_mma_qk.make_fragment_B(thr_mma_qk.partition_B(sK))
    tOrVt = thr_mma_pv.make_fragment_B(thr_mma_pv.partition_B(sVt))
    acc_shape_O = thr_mma_pv.partition_shape_C((self.tile_m, self.tile_hdimv))
    acc_Os = tuple(
      cute.make_fragment(acc_shape_O, Float32)
      for _ in range(self.num_v_chunks)
    )
    for v_group in cutlass.range_constexpr(self.num_v_chunks):
      acc_Os[v_group].fill(0.0)

    smem_copy_atom_QK = cute.make_copy_atom(
      warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype
    )
    smem_copy_atom_V = cute.make_copy_atom(
      warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self.dtype
    )
    smem_thr_copy_Q = utils.make_tiled_copy_A(smem_copy_atom_QK,
                                              tiled_mma_qk).get_slice(tidx)
    smem_thr_copy_K = utils.make_tiled_copy_B(smem_copy_atom_QK,
                                              tiled_mma_qk).get_slice(tidx)
    smem_thr_copy_V = utils.make_tiled_copy_B(smem_copy_atom_V,
                                              tiled_mma_pv).get_slice(tidx)
    tSsQ = smem_thr_copy_Q.partition_S(sQ)
    tSsK = smem_thr_copy_K.partition_S(sK)
    tOsVt = smem_thr_copy_V.partition_S(sVt)

    softmax = Softmax.create(
      softmax_scale_log2,
      num_rows=acc_Os[0].shape[0][0] * acc_Os[0].shape[1],
      softmax_scale=softmax_scale,
    )
    softmax.reset()

    if n_block_max > 0:
      n_block = Int32(0)
      acc_shape_S = thr_mma_qk.partition_shape_C((self.tile_m, self.tile_n))
      acc_S = cute.make_fragment(acc_shape_S, Float32)
      acc_S.fill(0.0)
      for d_group in cutlass.range_constexpr(self.num_d_chunks):
        gQ = cute.local_tile(
          mQ_cur, (self.tile_m, self.tile_hdim), (m_block, d_group)
        )
        gK = cute.local_tile(
          mK_cur, (self.tile_n, self.tile_hdim), (n_block, d_group)
        )
        self.load_Q(
          gmem_tiled_copy_Q.get_slice(tidx), gQ, sQ, m_block, seqlen.seqlen_q
        )
        self.load_K(
          gmem_tiled_copy_K.get_slice(tidx), gK, sK, n_block, seqlen.seqlen_k
        )
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()
        sm80_utils.gemm(
          thr_mma_qk,
          acc_S,
          tSrQ,
          tSrK,
          tSsQ,
          tSsK,
          smem_thr_copy_Q,
          smem_thr_copy_K,
        )
        cute.arch.barrier()

      self.apply_mask(acc_S, thr_mma_qk, m_block, n_block, seqlen)
      row_scale = softmax.online_softmax(acc_S, is_first=True)
      rP = cute.make_fragment_like(acc_S, self.dtype)
      rP.store(acc_S.load().to(self.dtype))
      tOrP = layout_utils.reshape_acc_to_frgA(rP)
      for v_group in cutlass.range_constexpr(self.num_v_chunks):
        softmax.rescale_O(acc_Os[v_group], row_scale)
        gV = cute.local_tile(
          mV_cur, (self.tile_n, self.tile_hdimv), (n_block, v_group)
        )
        self.load_V(
          gmem_tiled_copy_V.get_slice(tidx), gV, sV, n_block, seqlen.seqlen_k
        )
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        self.zero_V_tail(sV, n_block, seqlen.seqlen_k, tidx)
        sm80_utils.gemm_rs(
          thr_mma_pv, acc_Os[v_group], tOrP, tOrVt, tOsVt, smem_thr_copy_V
        )
        cute.arch.barrier()

      for n_iter in cutlass.range(n_block_max - 1, unroll=1):
        n_block = n_iter + 1
        acc_shape_S = thr_mma_qk.partition_shape_C((self.tile_m, self.tile_n))
        acc_S = cute.make_fragment(acc_shape_S, Float32)
        acc_S.fill(0.0)
        for d_group in cutlass.range_constexpr(self.num_d_chunks):
          gQ = cute.local_tile(
            mQ_cur, (self.tile_m, self.tile_hdim), (m_block, d_group)
          )
          gK = cute.local_tile(
            mK_cur, (self.tile_n, self.tile_hdim), (n_block, d_group)
          )
          self.load_Q(
            gmem_tiled_copy_Q.get_slice(tidx), gQ, sQ, m_block, seqlen.seqlen_q
          )
          self.load_K(
            gmem_tiled_copy_K.get_slice(tidx), gK, sK, n_block, seqlen.seqlen_k
          )
          cute.arch.cp_async_commit_group()
          cute.arch.cp_async_wait_group(0)
          cute.arch.barrier()
          sm80_utils.gemm(
            thr_mma_qk,
            acc_S,
            tSrQ,
            tSrK,
            tSsQ,
            tSsK,
            smem_thr_copy_Q,
            smem_thr_copy_K,
          )
          cute.arch.barrier()

        self.apply_mask(acc_S, thr_mma_qk, m_block, n_block, seqlen)
        row_scale = softmax.online_softmax(acc_S, is_first=False)
        rP = cute.make_fragment_like(acc_S, self.dtype)
        rP.store(acc_S.load().to(self.dtype))
        tOrP = layout_utils.reshape_acc_to_frgA(rP)
        for v_group in cutlass.range_constexpr(self.num_v_chunks):
          softmax.rescale_O(acc_Os[v_group], row_scale)
          gV = cute.local_tile(
            mV_cur, (self.tile_n, self.tile_hdimv), (n_block, v_group)
          )
          self.load_V(
            gmem_tiled_copy_V.get_slice(tidx), gV, sV, n_block, seqlen.seqlen_k
          )
          cute.arch.cp_async_commit_group()
          cute.arch.cp_async_wait_group(0)
          self.zero_V_tail(sV, n_block, seqlen.seqlen_k, tidx)
          sm80_utils.gemm_rs(
            thr_mma_pv, acc_Os[v_group], tOrP, tOrVt, tOsVt, smem_thr_copy_V
          )
          cute.arch.barrier()

    row_scale_final = softmax.finalize()
    sO = cute.make_tensor(sQ.iterator, sO_layout)
    for v_group in cutlass.range_constexpr(self.num_v_chunks):
      softmax.rescale_O(acc_Os[v_group], row_scale_final)
      self.store_O_and_lse(
        acc_Os[v_group],
        softmax.row_sum,
        mO_cur,
        mLSE,
        sO,
        seqlen,
        gmem_tiled_copy_O,
        tiled_mma_pv,
        tidx,
        m_block,
        head_idx,
        batch_idx,
        v_group,
        write_lse=v_group == 0,
      )

  @cute.jit
  def apply_mask(
    self,
    acc_S: cute.Tensor,
    thr_mma: cute.TiledMma,
    m_block: Int32,
    n_block: Int32,
    seqlen: SeqlenInfoQK,
  ) -> None:
    acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S)
    cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
    tScS_mn = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(cS))
    kv_offset = seqlen.seqlen_k - seqlen.seqlen_q
    for row in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
      for col in cutlass.range(cute.size(tScS_mn.shape[1]), unroll_full=True):
        q_idx = m_block * self.tile_m + tScS_mn[row, col][0]
        k_idx = n_block * self.tile_n + tScS_mn[row, col][1]
        valid = (q_idx < seqlen.seqlen_q) and (k_idx < seqlen.seqlen_k)
        if const_expr(self.is_causal):
          valid = valid and (k_idx <= q_idx + kv_offset)
        acc_S_mn[row, col] = acc_S_mn[row, col] if valid else -Float32.inf

  @cute.jit
  def load_Q(
    self,
    gmem_thr_copy: cute.TiledCopy,
    gQ: cute.Tensor,
    sQ: cute.Tensor,
    block: Int32,
    seqlen: Int32,
  ) -> None:
    tQsQ, tQgQ = gmem_thr_copy.partition_D(sQ), gmem_thr_copy.partition_S(gQ)
    cQ = cute.make_identity_tensor((self.tile_m, self.tile_hdim))
    tQcQ = gmem_thr_copy.partition_S(cQ)
    t0QcQ = gmem_thr_copy.get_slice(0).partition_S(cQ)
    for m in cutlass.range_constexpr(cute.size(tQsQ.shape[1])):
      if t0QcQ[0, m, 0][0] < seqlen - block * self.tile_m - tQcQ[0][0]:
        cute.copy(gmem_thr_copy, tQgQ[None, m, None], tQsQ[None, m, None])

  @cute.jit
  def load_K(
    self,
    gmem_thr_copy: cute.TiledCopy,
    gK: cute.Tensor,
    sK: cute.Tensor,
    block: Int32,
    seqlen: Int32,
  ) -> None:
    tKsK, tKgK = gmem_thr_copy.partition_D(sK), gmem_thr_copy.partition_S(gK)
    cK = cute.make_identity_tensor((self.tile_n, self.tile_hdim))
    tKcK = gmem_thr_copy.partition_S(cK)
    t0KcK = gmem_thr_copy.get_slice(0).partition_S(cK)
    for n in cutlass.range_constexpr(cute.size(tKsK.shape[1])):
      if t0KcK[0, n, 0][0] < seqlen - block * self.tile_n - tKcK[0][0]:
        cute.copy(gmem_thr_copy, tKgK[None, n, None], tKsK[None, n, None])

  @cute.jit
  def load_V(
    self,
    gmem_thr_copy: cute.TiledCopy,
    gV: cute.Tensor,
    sV: cute.Tensor,
    block: Int32,
    seqlen: Int32,
  ) -> None:
    tVsV, tVgV = gmem_thr_copy.partition_D(sV), gmem_thr_copy.partition_S(gV)
    cV = cute.make_identity_tensor((self.tile_n, self.tile_hdimv))
    tVcV = gmem_thr_copy.partition_S(cV)
    t0VcV = gmem_thr_copy.get_slice(0).partition_S(cV)
    for n in cutlass.range_constexpr(cute.size(tVsV.shape[1])):
      if t0VcV[0, n, 0][0] < seqlen - block * self.tile_n - tVcV[0][0]:
        cute.copy(gmem_thr_copy, tVgV[None, n, None], tVsV[None, n, None])

  @cute.jit
  def zero_V_tail(
    self,
    sV: cute.Tensor,
    block: Int32,
    seqlen: Int32,
    tidx: Int32,
  ) -> None:
    valid_rows = seqlen - block * self.tile_n
    if valid_rows < self.tile_n:
      tail_elems = (self.tile_n - valid_rows) * self.tile_hdimv
      for linear_idx in cutlass.range(
        tidx, tail_elems, self.num_threads, unroll=1
      ):
        row_offset = linear_idx // self.tile_hdimv
        col = linear_idx - row_offset * self.tile_hdimv
        sV[valid_rows + row_offset, col] = sV.element_type(0.0)
    cute.arch.barrier()

  @cute.jit
  def store_O_and_lse(
    self,
    acc_O: cute.Tensor,
    lse: cute.Tensor,
    mO_cur: cute.Tensor,
    mLSE: Optional[cute.Tensor],
    sO: cute.Tensor,
    seqlen: SeqlenInfoQK,
    gmem_tiled_copy_O: cute.TiledCopy,
    tiled_mma: cute.TiledMma,
    tidx: Int32,
    m_block: Int32,
    head_idx: Int32,
    batch_idx: Int32,
    v_group: cutlass.Constexpr[int],
    write_lse: cutlass.Constexpr[bool],
  ) -> None:
    rO = cute.make_fragment_like(acc_O, self.dtype)
    rO.store(acc_O.load().to(self.dtype))
    cute.arch.barrier()
    smem_copy_atom_O = cute.make_copy_atom(
      cute.nvgpu.CopyUniversalOp(),
      self.dtype,
      num_bits_per_copy=2 * self.dtype.width
    )
    smem_thr_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O,
                                             tiled_mma).get_slice(tidx)
    taccOrO = smem_thr_copy_O.retile(rO)
    taccOsO = smem_thr_copy_O.partition_D(sO)
    cute.copy(smem_copy_atom_O, taccOrO, taccOsO)

    if const_expr(write_lse and mLSE is not None):
      mLSE_cur = seqlen.offset_batch_Q(mLSE, batch_idx, dim=2)[None, head_idx]
      gLSE = cute.local_tile(mLSE_cur, (self.tile_m, ), (m_block, ))
      gLSE_expanded_layout = cute.append(
        gLSE.layout, cute.make_layout((self.tile_hdimv, ), stride=(0, ))
      )
      gLSE_expanded = cute.make_tensor(gLSE.iterator, gLSE_expanded_layout)
      thr_mma = tiled_mma.get_slice(tidx)
      taccOgLSE = layout_utils.reshape_acc_to_mn(
        thr_mma.partition_C(gLSE_expanded)
      )
      taccOcO = layout_utils.reshape_acc_to_mn(
        thr_mma.partition_C(
          cute.make_identity_tensor((self.tile_m, self.tile_hdimv))
        )
      )
      t0accOcO = layout_utils.reshape_acc_to_mn(
        thr_mma.get_slice(0).partition_C(
          cute.make_identity_tensor((self.tile_m, self.tile_hdimv))
        )
      )
      if taccOcO[0][1] == 0:
        for row in cutlass.range(
          cute.size(taccOgLSE.shape[1]), unroll_full=True
        ):
          if t0accOcO[
            row,
            0][0] < seqlen.seqlen_q - m_block * self.tile_m - taccOcO[0][0]:
            taccOgLSE[row, 0] = lse[row]

    cute.arch.barrier()
    gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
    tOsO = gmem_thr_copy_O.partition_S(sO)
    tOrO = cute.make_fragment_like(tOsO, self.dtype)
    cute.autovec_copy(tOsO, tOrO)
    gO = cute.local_tile(
      mO_cur, (self.tile_m, self.tile_hdimv), (m_block, v_group)
    )
    tOgO = gmem_thr_copy_O.partition_D(gO)
    cO = cute.make_identity_tensor((self.tile_m, self.tile_hdimv))
    tOcO = gmem_thr_copy_O.partition_S(cO)
    t0OcO = gmem_tiled_copy_O.get_slice(0).partition_S(cO)
    for rest_m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
      if t0OcO[0, rest_m,
               0][0] < seqlen.seqlen_q - m_block * self.tile_m - tOcO[0][0]:
        cute.copy(
          gmem_tiled_copy_O,
          tOrO[None, rest_m, None],
          tOgO[None, rest_m, None],
        )


__all__ = ["FFPAAttnFwdSm80SplitD"]

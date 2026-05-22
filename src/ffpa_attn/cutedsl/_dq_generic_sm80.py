# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# Adapted from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/flash_bwd.py
# FFPA adaptation: SM80/SM89 atomic-free Split-D dQ backward with 64-wide D chunks.

from typing import Optional, Type

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils_basic
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync, warp

from quack import layout_utils

from . import utils
from ._utils import SM80_SPLIT_D_CHUNK
from .utils import ampere_helpers as sm80_utils
from .utils.cute_dsl_utils import assume_tensor_aligned


class FFPAAttnBwdDQSm80SplitDGeneric:
  """SM80/SM89 atomic-free dQ kernel for the large-D Split-D path.

  One CTA owns one ``(batch, query_head, m_block)`` tile and streams every KV
  tile. For each score tile it reconstructs ``S`` and ``dP`` once across all D
  chunks, then walks the D chunks to update ``dq``. This follows the Triton
  Split-D loop ordering instead of recomputing scores once per output chunk.
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
    num_stages_Q: int = 2,
    num_stages_dO: int = 2,
    num_threads: int = 128,
  ):
    self.dtype = dtype
    self.head_dim = head_dim
    self.head_dim_v = head_dim if head_dim_v is None else head_dim_v
    self.d_chunk = SM80_SPLIT_D_CHUNK
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
  ) -> bool:
    """Check whether the SM80 Split-D dQ configuration fits resources."""
    del is_causal
    if head_dim_v is None:
      head_dim_v = head_dim
    d_chunk = SM80_SPLIT_D_CHUNK
    if dtype not in [cutlass.Float16, cutlass.BFloat16]:
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
    smem_usage = (
      tile_m * d_chunk * num_stages_Q * elem_bytes +
      tile_n * d_chunk * elem_bytes + tile_n * d_chunk * elem_bytes +
      tile_m * d_chunk * num_stages_dO * elem_bytes + tile_m * d_chunk * 4
    )
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
    mdQ_type: Type[cutlass.Numeric],
  ) -> None:
    if const_expr(not (mQ_type == mK_type == mV_type == mdO_type)):
      raise TypeError("q, k, v, and dout must have the same dtype")
    if const_expr(mQ_type not in [cutlass.Float16, cutlass.BFloat16]):
      raise TypeError("Only Float16 and BFloat16 are supported")
    if const_expr(mLSE_type is not Float32 or mD_type is not Float32):
      raise TypeError("lse_log2 and dpsum must be Float32")
    if const_expr(mdQ_type != mQ_type):
      raise TypeError("dq must match the input dtype")
    assert mQ_type == self.dtype

  def _get_shared_storage_cls(self):
    out_layout = cute.make_layout((self.tile_m, self.d_chunk),
                                  stride=(self.d_chunk, 1))

    @cute.struct
    class SharedStorage:
      sQ: cute.struct.Align[cute.struct.MemRange[self.dtype,
                                                 cute.cosize(self.sQ_layout)],
                            1024]
      sK: cute.struct.Align[cute.struct.MemRange[self.dtype,
                                                 cute.cosize(self.sK_layout)],
                            1024]
      sV: cute.struct.Align[cute.struct.MemRange[self.dtype,
                                                 cute.cosize(self.sV_layout)],
                            1024]
      sdO: cute.struct.Align[cute.struct.MemRange[self.dtype,
                                                  cute.cosize(self.sdO_layout)],
                             1024]
      sdQ: cute.struct.Align[cute.struct.MemRange[Float32,
                                                  cute.cosize(out_layout)], 128]

    return SharedStorage, out_layout

  def _setup_attributes(self) -> None:
    sQ_layout_atom = sm80_utils.get_smem_layout_atom(self.dtype, self.tile_hdim)
    sK_layout_atom = sQ_layout_atom
    sV_layout_atom = sm80_utils.get_smem_layout_atom(self.dtype, self.tile_hdim)
    self.sQ_layout = cute.tile_to_shape(
      sQ_layout_atom, (self.tile_m, self.tile_hdim, self.num_stages_Q),
      (0, 1, 2)
    )
    self.sK_layout = cute.tile_to_shape(
      sK_layout_atom, (self.tile_n, self.tile_hdim), (0, 1)
    )
    self.sV_layout = cute.tile_to_shape(
      sV_layout_atom, (self.tile_n, self.tile_hdim), (0, 1)
    )
    self.sdO_layout = cute.tile_to_shape(
      sV_layout_atom, (self.tile_m, self.tile_hdim, self.num_stages_dO),
      (0, 1, 2)
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
    tiled_mma_dq = cute.make_tiled_mma(
      warp.MmaF16BF16Op(self.dtype, Float32, (16, 8, 16)),
      atom_layout,
      permutation_mnk=permutation_mnk,
    )
    return tiled_mma_sdp, tiled_mma_dq

  @cute.jit
  def __call__(
    self,
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mdO: cute.Tensor,
    mLSElog2: cute.Tensor,
    mD: cute.Tensor,
    mdQ: cute.Tensor,
    softmax_scale: Float32,
    stream: cuda.CUstream = None,
  ):
    """Configure and launch the dense SM80 dQ kernel."""
    self._check_type(
      *(t.element_type for t in (mQ, mK, mV, mdO, mLSElog2, mD, mdQ))
    )
    mQ, mK, mV, mdO, mLSElog2, mD, mdQ = [
      assume_tensor_aligned(t) for t in (mQ, mK, mV, mdO, mLSElog2, mD, mdQ)
    ]
    self._setup_attributes()
    tiled_mma_sdp, tiled_mma_dq = self._get_tiled_mma()
    SharedStorage, out_layout = self._get_shared_storage_cls()
    batch = mQ.shape[0]
    num_head = mQ.shape[2]
    seqlen_q = mQ.shape[1]
    grid = [
      cute.ceil_div(seqlen_q, self.tile_m),
      batch * num_head,
      1,
    ]
    self.kernel(
      mQ,
      mK,
      mV,
      mdO,
      mLSElog2,
      mD,
      mdQ,
      softmax_scale,
      out_layout,
      self.sQ_layout,
      self.sK_layout,
      self.sV_layout,
      self.sdO_layout,
      self.gmem_tiled_copy_QK,
      self.gmem_tiled_copy_VdO,
      tiled_mma_sdp,
      tiled_mma_dq,
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
    mdQ: cute.Tensor,
    softmax_scale: Float32,
    out_layout: cute.Layout,
    sQ_layout: cute.ComposedLayout,
    sK_layout: cute.ComposedLayout,
    sV_layout: cute.ComposedLayout,
    sdO_layout: cute.ComposedLayout,
    gmem_tiled_copy_QK: cute.TiledCopy,
    gmem_tiled_copy_VdO: cute.TiledCopy,
    tiled_mma_sdp: cute.TiledMma,
    tiled_mma_dq: cute.TiledMma,
    SharedStorage: cutlass.Constexpr,
  ):
    tidx, _, _ = cute.arch.thread_idx()
    block_m, block_hb, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    batch_idx = block_hb // mQ.shape[2]
    q_head_idx = block_hb - batch_idx * mQ.shape[2]
    kv_head_idx = q_head_idx // self.qhead_per_kvhead
    start_m = block_m * self.tile_m
    seqlen_q = mQ.shape[1]
    seqlen_k = mK.shape[1]
    num_n_blocks = cute.ceil_div(seqlen_k, self.tile_n)
    softmax_scale_log2 = softmax_scale * 1.4426950408889634

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sQ = storage.sQ.get_tensor(sQ_layout)
    sK = storage.sK.get_tensor(sK_layout)
    sV = storage.sV.get_tensor(sV_layout)
    sdO = storage.sdO.get_tensor(sdO_layout)
    sdQ = storage.sdQ.get_tensor(out_layout)

    sKt = layout_utils.transpose_view(sK)

    thr_mma_sdp = tiled_mma_sdp.get_slice(tidx)
    thr_mma_dq = tiled_mma_dq.get_slice(tidx)
    tSrQ = utils.mma_make_fragment_A(sQ[None, None, 0], thr_mma_sdp)
    tSrK = utils.mma_make_fragment_B(sK, thr_mma_sdp)
    tdPrdO = utils.mma_make_fragment_A(sdO[None, None, 0], thr_mma_sdp)
    tdPrV = utils.mma_make_fragment_B(sV, thr_mma_sdp)
    tdQrK = utils.mma_make_fragment_B(sKt, thr_mma_dq)
    acc_shape_dQ = thr_mma_dq.partition_shape_C((self.tile_m, self.d_chunk))

    smem_copy_atom = cute.make_copy_atom(
      warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype
    )
    smem_copy_atom_t = cute.make_copy_atom(
      warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self.dtype
    )
    smem_thr_copy_Q = utils.make_tiled_copy_A(smem_copy_atom,
                                              tiled_mma_sdp).get_slice(tidx)
    smem_thr_copy_K = utils.make_tiled_copy_B(smem_copy_atom,
                                              tiled_mma_sdp).get_slice(tidx)
    smem_thr_copy_dO = utils.make_tiled_copy_A(smem_copy_atom,
                                               tiled_mma_sdp).get_slice(tidx)
    smem_thr_copy_V = utils.make_tiled_copy_B(smem_copy_atom,
                                              tiled_mma_sdp).get_slice(tidx)
    smem_thr_copy_Kt = utils.make_tiled_copy_B(smem_copy_atom_t,
                                               tiled_mma_dq).get_slice(tidx)
    tSsQ = smem_thr_copy_Q.partition_S(sQ)
    tSsK = smem_thr_copy_K.partition_S(sK)
    tdPsdO = smem_thr_copy_dO.partition_S(sdO)
    tdPsV = smem_thr_copy_V.partition_S(sV)
    tdQsKt = smem_thr_copy_Kt.partition_S(sKt)
    acc2s_thr_copy = cute.make_tiled_copy_C(
      cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=Float32.width
      ),
      tiled_mma_dq,
    ).get_slice(tidx)
    tAccSdQ = acc2s_thr_copy.partition_D(sdQ)

    for n_block in cutlass.range(num_n_blocks, unroll=1):
      start_n = n_block * self.tile_n
      acc_shape_S = thr_mma_sdp.partition_shape_C((self.tile_m, self.tile_n))
      acc_S = cute.make_rmem_tensor(acc_shape_S, Float32)
      acc_dP = cute.make_rmem_tensor(acc_shape_S, Float32)
      acc_S.fill(0.0)
      acc_dP.fill(0.0)
      gQ0 = cute.local_tile(
        mQ[batch_idx, None, q_head_idx, None],
        (self.tile_m, self.d_chunk),
        (block_m, 0),
      )
      gdO0 = cute.local_tile(
        mdO[batch_idx, None, q_head_idx, None],
        (self.tile_m, self.d_chunk),
        (block_m, 0),
      )
      self.load_m_tile(
        gmem_tiled_copy_QK.get_slice(tidx), gQ0, sQ[None, None, 0], block_m,
        seqlen_q
      )
      self.load_m_tile(
        gmem_tiled_copy_VdO.get_slice(tidx), gdO0, sdO[None, None, 0], block_m,
        seqlen_q
      )
      cute.arch.cp_async_commit_group()
      for score_d_block in cutlass.range_constexpr(self.num_d_chunks):
        q_stage = score_d_block % self.num_stages_Q
        do_stage = score_d_block % self.num_stages_dO
        gK = cute.local_tile(
          mK[batch_idx, None, kv_head_idx, None],
          (self.tile_n, self.d_chunk),
          (n_block, score_d_block),
        )
        gV = cute.local_tile(
          mV[batch_idx, None, kv_head_idx, None],
          (self.tile_n, self.d_chunk),
          (n_block, score_d_block),
        )
        self.load_n_tile(
          gmem_tiled_copy_QK.get_slice(tidx), gK, sK, n_block, seqlen_k
        )
        self.load_n_tile(
          gmem_tiled_copy_VdO.get_slice(tidx), gV, sV, n_block, seqlen_k
        )
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()
        self.zero_m_tail(sQ[None, None, q_stage], block_m, seqlen_q, tidx)
        self.zero_m_tail(sdO[None, None, do_stage], block_m, seqlen_q, tidx)
        self.zero_n_tail(sK, n_block, seqlen_k, tidx)
        self.zero_n_tail(sV, n_block, seqlen_k, tidx)
        if const_expr(score_d_block + 1 < self.num_d_chunks):
          next_q_stage = (score_d_block + 1) % self.num_stages_Q
          next_do_stage = (score_d_block + 1) % self.num_stages_dO
          gQ_next = cute.local_tile(
            mQ[batch_idx, None, q_head_idx, None],
            (self.tile_m, self.d_chunk),
            (block_m, score_d_block + 1),
          )
          gdO_next = cute.local_tile(
            mdO[batch_idx, None, q_head_idx, None],
            (self.tile_m, self.d_chunk),
            (block_m, score_d_block + 1),
          )
          self.load_m_tile(
            gmem_tiled_copy_QK.get_slice(tidx), gQ_next,
            sQ[None, None, next_q_stage], block_m, seqlen_q
          )
          self.load_m_tile(
            gmem_tiled_copy_VdO.get_slice(tidx), gdO_next,
            sdO[None, None, next_do_stage], block_m, seqlen_q
          )
          cute.arch.cp_async_commit_group()
        sm80_utils.gemm(
          thr_mma_sdp, acc_S, tSrQ, tSrK, tSsQ[None, None, None, q_stage], tSsK,
          smem_thr_copy_Q, smem_thr_copy_K
        )
        sm80_utils.gemm(
          thr_mma_sdp, acc_dP, tdPrdO, tdPrV, tdPsdO[None, None, None,
                                                     do_stage], tdPsV,
          smem_thr_copy_dO, smem_thr_copy_V
        )
        cute.arch.barrier()

      self.make_ds(
        acc_S,
        acc_dP,
        thr_mma_sdp,
        mLSElog2,
        mD,
        softmax_scale,
        softmax_scale_log2,
        batch_idx,
        q_head_idx,
        block_m,
        n_block,
        seqlen_q,
        seqlen_k,
      )
      rdS = cute.make_fragment_like(acc_dP, self.dtype)
      rdS.store(acc_dP.load().to(self.dtype))
      tdQrdS = layout_utils.reshape_acc_to_frgA(rdS)

      for d_block in cutlass.range_constexpr(self.num_d_chunks):
        acc_dQ = cute.make_rmem_tensor(acc_shape_dQ, Float32)
        acc_dQ.fill(0.0)
        gK_out = cute.local_tile(
          mK[batch_idx, None, kv_head_idx, None],
          (self.tile_n, self.d_chunk),
          (n_block, d_block),
        )
        self.load_n_tile(
          gmem_tiled_copy_QK.get_slice(tidx), gK_out, sK, n_block, seqlen_k
        )
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()
        self.zero_n_tail(sK, n_block, seqlen_k, tidx)
        sm80_utils.gemm_rs(
          thr_mma_dq, acc_dQ, tdQrdS, tdQrK, tdQsKt, smem_thr_copy_Kt
        )
        cute.arch.barrier()
        cute.copy(acc2s_thr_copy, acc2s_thr_copy.retile(acc_dQ), tAccSdQ)
        cute.arch.barrier()
        self.store_dq_tile(
          sdQ,
          mdQ,
          batch_idx,
          q_head_idx,
          block_m,
          d_block,
          seqlen_q,
          n_block != 0,
          tidx,
          bdim,
        )
        cute.arch.barrier()

  @cute.jit
  def make_ds(
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
    acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S)
    acc_dP_mn = layout_utils.reshape_acc_to_mn(acc_dP)
    cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
    tScS_mn = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(cS))
    kv_offset = seqlen_k - seqlen_q
    for row in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
      for col in cutlass.range(cute.size(tScS_mn.shape[1]), unroll_full=True):
        q_idx = m_block * self.tile_m + tScS_mn[row, col][0]
        k_idx = n_block * self.tile_n + tScS_mn[row, col][1]
        valid = (q_idx < seqlen_q) and (k_idx < seqlen_k)
        if const_expr(self.is_causal):
          valid = valid and (k_idx <= q_idx + kv_offset)
        if valid:
          p = cute.math.exp2(
            acc_S_mn[row, col] * softmax_scale_log2 -
            mLSElog2[batch_idx, q_head_idx, q_idx],
            fastmath=True,
          )
          acc_dP_mn[row, col] = p * (
            acc_dP_mn[row, col] - mD[batch_idx, q_head_idx, q_idx]
          ) * softmax_scale
        else:
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
      tail_elems = (self.tile_m - valid_rows) * self.d_chunk
      for linear_idx in cutlass.range(
        tidx, tail_elems, self.num_threads, unroll=1
      ):
        row_offset = linear_idx // self.d_chunk
        col = linear_idx - row_offset * self.d_chunk
        smem_tile[valid_rows + row_offset, col] = smem_tile.element_type(0.0)
    cute.arch.barrier()

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
      tail_elems = (self.tile_n - valid_rows) * self.d_chunk
      for linear_idx in cutlass.range(
        tidx, tail_elems, self.num_threads, unroll=1
      ):
        row_offset = linear_idx // self.d_chunk
        col = linear_idx - row_offset * self.d_chunk
        smem_tile[valid_rows + row_offset, col] = smem_tile.element_type(0.0)
    cute.arch.barrier()

  @cute.jit
  def store_dq_tile(
    self,
    sdQ: cute.Tensor,
    mdQ: cute.Tensor,
    batch_idx: Int32,
    q_head_idx: Int32,
    m_block: Int32,
    d_block: Int32,
    seqlen_q: Int32,
    add_to_existing: cutlass.Boolean,
    tidx: Int32,
    bdim: Int32,
  ) -> None:
    d_start = d_block * self.d_chunk
    start_m = m_block * self.tile_m
    for linear in cutlass.range(tidx, self.tile_m * self.d_chunk, bdim):
      m_local = linear // self.d_chunk
      d_local = linear - m_local * self.d_chunk
      q_idx = start_m + m_local
      d_idx = d_start + d_local
      if q_idx < seqlen_q and d_idx < self.head_dim:
        dq_val = sdQ[m_local, d_local]
        if add_to_existing:
          dq_val = dq_val + mdQ[batch_idx, q_idx, q_head_idx, d_idx].to(Float32)
        mdQ[batch_idx, q_idx, q_head_idx, d_idx] = dq_val.to(self.dtype)


__all__ = ["FFPAAttnBwdDQSm80SplitDGeneric"]

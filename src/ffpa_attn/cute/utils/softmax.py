# This file is copied from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/softmax.py
# Copyright (c) 2025, Tri Dao.

import math
import operator
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Float32

from quack import layout_utils
from . import fmax_reduce, fadd_reduce, warp_reduce, scalar_to_ssa
from quack.cute_dsl_utils import ParamsBase
from .seqlen_info import SeqlenInfoQK


@dataclass
class Softmax(ParamsBase):
  scale_log2: Float32
  num_rows: cutlass.Constexpr[int]
  row_max: cute.Tensor
  row_sum: cute.Tensor
  arch: cutlass.Constexpr[int] = 80
  softmax_scale: Float32 | None = None

  @staticmethod
  def create(
    scale_log2: Float32,
    num_rows: cutlass.Constexpr[int],
    arch: cutlass.Constexpr[int] = 80,
    softmax_scale: Float32 | None = None,
  ):
    row_max = cute.make_rmem_tensor(num_rows, Float32)
    row_sum = cute.make_rmem_tensor(num_rows, Float32)
    return Softmax(scale_log2, num_rows, row_max, row_sum, arch, softmax_scale)

  def reset(self) -> None:
    self.row_max.fill(-Float32.inf)
    self.row_sum.fill(0.0)

  def _compute_row_max(
    self,
    acc_S_row: cute.TensorSSA,
    init_val: float | Float32 | None = None
  ) -> Float32:
    return fmax_reduce(acc_S_row, init_val, arch=self.arch)

  def _compute_row_sum(
    self,
    acc_S_row_exp: cute.TensorSSA,
    init_val: float | Float32 | None = None
  ) -> Float32:
    return fadd_reduce(acc_S_row_exp, init_val, arch=self.arch)

  @cute.jit
  def online_softmax(
    self,
    acc_S: cute.Tensor,
    is_first: cutlass.Constexpr[bool] = False,
    check_inf: cutlass.Constexpr[bool] = True,
  ) -> cute.Tensor:
    """Apply online softmax and return the row_scale to rescale O.

        :param acc_S: acc_S tensor
        :type acc_S: cute.Tensor
        :param is_first: is first n_block
        :type is_first: cutlass.Constexpr
        """
    # Change acc_S to M,N layout view.
    acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S)
    row_scale = cute.make_fragment_like(self.row_max, Float32)

    row_max = self.row_max
    row_sum = self.row_sum
    scale_log2 = self.scale_log2
    arch = self.arch

    # Each iteration processes one row of acc_S
    for r in cutlass.range(cute.size(row_max), unroll_full=True):
      acc_S_row = acc_S_mn[r, None].load()  # (n_block_size)

      row_max_cur = fmax_reduce(
        acc_S_row,
        init_val=row_max[r] if cutlass.const_expr(not is_first) else None,
        arch=arch,
      )

      row_max_cur = cute.arch.warp_reduction_max(
        row_max_cur, threads_in_group=4
      )
      # Update row_max before changing row_max_cur to safe value for -inf
      row_max_prev = row_max[r]
      row_max[r] = row_max_cur

      if cutlass.const_expr(check_inf):  # apply mask
        row_max_cur = 0.0 if row_max_cur == -Float32.inf else row_max_cur

      if cutlass.const_expr(is_first):
        row_max_cur_scaled = row_max_cur * scale_log2
        acc_S_row_exp = cute.math.exp2(
          acc_S_row * scale_log2 - row_max_cur_scaled, fastmath=True
        )
        acc_S_row_sum = fadd_reduce(acc_S_row_exp, init_val=None, arch=arch)
        row_scale[r] = 1.0
      else:
        row_max_cur_scaled = row_max_cur * scale_log2
        acc_S_row_exp = cute.math.exp2(
          acc_S_row * scale_log2 - row_max_cur_scaled, fastmath=True
        )
        # row_scale[r] = cute.math.exp2(row_max_prev * self.scale_log2 - row_max_cur_scaled)
        row_scale[r] = cute.math.exp2((row_max_prev - row_max_cur) * scale_log2,
                                      fastmath=True)
        acc_S_row_sum = fadd_reduce(
          acc_S_row_exp, init_val=row_sum[r] * row_scale[r], arch=arch
        )

      row_sum[r] = acc_S_row_sum
      acc_S_mn[r, None].store(acc_S_row_exp)

    return row_scale

  @cute.jit
  def finalize(
    self,
    final_scale: Float32 = 1.0,
    sink_val: Float32 | cute.Tensor | None = None
  ) -> cute.Tensor:
    """Finalize the online softmax by computing the scale and logsumexp."""
    if cutlass.const_expr(
      sink_val is not None and isinstance(sink_val, cute.Tensor)
    ):
      assert cute.size(sink_val) == cute.size(self.row_sum)
    row_sum = self.row_sum
    row_max = self.row_max
    scale_log2 = self.scale_log2

    # quad reduction for row_sum as we didn't do it during each iteration of online softmax
    row_sum.store(warp_reduce(row_sum.load(), operator.add, width=4))
    row_scale = cute.make_fragment_like(row_max, Float32)

    for r in cutlass.range(cute.size(row_sum), unroll_full=True):
      if cutlass.const_expr(sink_val is not None):
        sink_val_cur = sink_val if not isinstance(sink_val,
                                                  cute.Tensor) else sink_val[r]
        LOG2_E = math.log2(math.e)
        row_sum[r] += cute.math.exp2(
          sink_val_cur * LOG2_E - row_max[r] * scale_log2, fastmath=True
        )

      # if row_sum is zero or nan, set acc_O_mn_row to 1.0
      acc_O_mn_row_is_zero_or_nan = row_sum[r] == 0.0 or row_sum[r] != row_sum[r]
      row_scale[r] = (
        cute.arch.
        rcp_approx(row_sum[r] if not acc_O_mn_row_is_zero_or_nan else 1.0)
      ) * final_scale
      row_sum_cur = row_sum[r]
      LN2 = math.log(2.0)
      row_sum[r] = (
        (row_max[r] * scale_log2 + cute.math.log2(row_sum_cur, fastmath=True)) *
        LN2 if not acc_O_mn_row_is_zero_or_nan else -Float32.inf
      )
    return row_scale

  @cute.jit
  def rescale_O(self, acc_O: cute.Tensor, row_scale: cute.Tensor) -> None:
    """Scale each row of acc_O by the given scale tensor.
        :param acc_O: input tensor
        :type acc_O: cute.Tensor
        :param row_scale: row_scale tensor
        :type row_scale: cute.Tensor
        """
    acc_O_mn = layout_utils.reshape_acc_to_mn(acc_O)
    assert cute.size(row_scale) == cute.size(acc_O_mn, mode=[0])
    for r in cutlass.range(cute.size(row_scale), unroll_full=True):
      acc_O_mn[r, None].store(acc_O_mn[r, None].load() * row_scale[r])


# ════════════════════════════════════════════════════════════════════════════
# V2 (2-WG asymmetric PV) cross-WG row-scale handoff helpers
# ════════════════════════════════════════════════════════════════════════════
# WG1 owns softmax and computes row_scale; WG2 must apply the same row_scale
# to its acc_O_back. We hand it off through `sScale` shaped (tile_m, num_stages_p).
#
# Layout convention: tiled_mma_qk and tiled_mma_pv both use atom_M=64 (single
# WGMMA M-mode atom), so per-thread M-partition is identical between them. Each
# row of M is owned by a 4-thread quad: only `lane % 4 == 0` writes; all 4
# threads in the quad read (and within each WG, all threads with that row
# coordinate read).


@cute.jit
def store_scale_to_smem(
  row_scale: cute.Tensor,
  sScale: cute.Tensor,
  stage,
  tiled_mma: cute.TiledMma,
  tidx,
  tile_m: cutlass.Constexpr[int],
  tile_n: cutlass.Constexpr[int],
):
  """WG1 writes its per-thread row_scale fragment into sScale[row_global, stage].

    Only the leftmost-quad thread (col offset 0) of each row writes. Per-thread
    fragment length must equal the WGMMA C-frag's num_rows_per_thread (typically 2
    for atom_M=64).
    """
  thr_mma = tiled_mma.get_slice(tidx)
  cS = cute.make_identity_tensor((tile_m, tile_n))
  tScS = thr_mma.partition_C(cS)
  tScS_mn = layout_utils.reshape_acc_to_mn(tScS)
  # tScS_mn[r, c] = (row_global, col_global) tuple
  if tScS_mn[0, 0][1] == 0:
    for r in cutlass.range(cute.size(row_scale), unroll_full=True):
      r_global = tScS_mn[r, 0][0]
      sScale[r_global, stage] = row_scale[r]


@cute.jit
def load_scale_from_smem(
  sScale: cute.Tensor,
  stage,
  tiled_mma: cute.TiledMma,
  tidx,
  tile_m: cutlass.Constexpr[int],
  tile_n: cutlass.Constexpr[int],
  num_rows: cutlass.Constexpr[int],
) -> cute.Tensor:
  """Each thread loads its owned rows of row_scale from sScale[row_global, stage]."""
  row_scale = cute.make_rmem_tensor(num_rows, Float32)
  thr_mma = tiled_mma.get_slice(tidx)
  cS = cute.make_identity_tensor((tile_m, tile_n))
  tScS = thr_mma.partition_C(cS)
  tScS_mn = layout_utils.reshape_acc_to_mn(tScS)
  for r in cutlass.range(num_rows, unroll_full=True):
    r_global = tScS_mn[r, 0][0]
    row_scale[r] = sScale[r_global, stage]
  return row_scale


@cute.jit
def rescale_O_with_external_scale(
  acc_O: cute.Tensor,
  row_scale: cute.Tensor,
):
  """Free-function rescale_O (no Softmax instance) used by WG2."""
  acc_O_mn = layout_utils.reshape_acc_to_mn(acc_O)
  assert cute.size(row_scale) == cute.size(acc_O_mn, mode=[0])
  for r in cutlass.range(cute.size(row_scale), unroll_full=True):
    acc_O_mn[r, None].store(acc_O_mn[r, None].load() * row_scale[r])


@cute.jit
def floor_if_packed(
  q_idx,
  qhead_per_kvhead: cutlass.Constexpr[int],
) -> cute.Tensor:
  """Convert q_idx to packed format for Pack-GQA."""
  if cutlass.const_expr(qhead_per_kvhead == 1):
    return q_idx
  return q_idx // qhead_per_kvhead


@cute.jit
def apply_score_mod_inner(
  score_tensor,
  index_tensor,
  score_mod: cutlass.Constexpr,
  batch_idx,
  head_idx,
  softmax_scale,
  vec_size: cutlass.Constexpr,
  qk_acc_dtype: cutlass.Constexpr,
  aux_tensors,
  fastdiv_mods,
  seqlen_info: SeqlenInfoQK,
  constant_q_idx: cutlass.Constexpr,
  qhead_per_kvhead: cutlass.Constexpr[int] = 1,
  transpose_indices: cutlass.Constexpr[bool] = False,
):
  """Shared implementation for applying score modification.

    Args:
        score_tensor: The scores to modify (acc_S for flash_fwd, tSrS_t2r for sm100)
        index_tensor: Index positions (tScS for flash_fwd, tScS_t2r for sm100)
        score_mod: The score modification function to apply
        batch_idx: Batch index
        head_idx: Head index
        softmax_scale: Scale to apply
        vec_size: Vector size for processing elements
        qk_acc_dtype: Data type for accumulator
        aux_tensors: Optional aux_tensors for FlexAttention
        fastdiv_mods: Tuple of (seqlen_q_divmod, seqlen_k_divmod) for wrapping
        seqlen_info: Sequence length info
        constant_q_idx: If provided, use this constant for all q_idx values
                        If None, compute q_idx per-element
        qhead_per_kvhead_packgqa: Pack-GQA replication factor. Divide q_idx by this
                                  when greater than 1 so score mods see logical heads.
        transpose_indices: If True, swap q_idx/kv_idx in index_tensor (for bwd kernel where S is transposed)
    """
  # Index positions in the index_tensor tuple
  # Forward: index_tensor[...][0] = q_idx, index_tensor[...][1] = kv_idx
  # Backward (transposed): index_tensor[...][0] = kv_idx, index_tensor[...][1] = q_idx
  if cutlass.const_expr(transpose_indices):
    q_idx_pos = cutlass.const_expr(1)
    kv_idx_pos = cutlass.const_expr(0)
  else:
    q_idx_pos = cutlass.const_expr(0)
    kv_idx_pos = cutlass.const_expr(1)

  n_vals = cutlass.const_expr(cute.size(score_tensor.shape))
  score_vec = cute.make_rmem_tensor(vec_size, qk_acc_dtype)
  kv_idx_vec = cute.make_rmem_tensor(vec_size, cutlass.Int32)

  # SSA values for batch (constant across all elements)
  batch_idx_ssa = scalar_to_ssa(batch_idx,
                                cutlass.Int32).broadcast_to((vec_size, ))

  # Handle q_idx based on whether it's constant
  q_idx_vec = cute.make_rmem_tensor(vec_size, cutlass.Int32)

  # For Pack-GQA with non-constant q_idx, we need per-element head indices
  # since a thread my process multiple query head indices
  if cutlass.const_expr(qhead_per_kvhead > 1 and constant_q_idx is None):
    head_idx_vec = cute.make_rmem_tensor(vec_size, cutlass.Int32)

  for i in cutlass.range(0, n_vals, vec_size, unroll_full=True):
    for j in cutlass.range(vec_size, unroll_full=True):
      score_vec[j] = score_tensor[i + j] * softmax_scale

      # Extract head offset from packed q_idx for Pack-GQA
      if cutlass.const_expr(qhead_per_kvhead > 1 and constant_q_idx is None):
        q_idx_packed = index_tensor[i + j][q_idx_pos]
        # Building up the logical q_head idx: final_q_head = kv_head * qhead_per_kvhead + (q_physical % qhead_per_kvhead)
        q_idx_logical = q_idx_packed // qhead_per_kvhead
        head_offset = q_idx_packed - q_idx_logical * qhead_per_kvhead
        head_idx_vec[j] = head_idx * qhead_per_kvhead + head_offset

      # If we will do loads we mod, in order to not read OOB
      if cutlass.const_expr(
        aux_tensors is not None and fastdiv_mods is not None
      ):
        if cutlass.const_expr(constant_q_idx is None):
          seqlen_q_divmod, seqlen_k_divmod = fastdiv_mods
          q_idx_floored = floor_if_packed(
            index_tensor[i + j][q_idx_pos], qhead_per_kvhead
          )
          _, q_idx_wrapped = divmod(q_idx_floored, seqlen_q_divmod)
          q_idx_vec[j] = q_idx_wrapped
        else:
          _, seqlen_k_divmod = fastdiv_mods

        _, kv_idx_wrapped = divmod(
          index_tensor[i + j][kv_idx_pos], seqlen_k_divmod
        )
        kv_idx_vec[j] = kv_idx_wrapped
      else:
        # No bounds checking - direct indexing
        if constant_q_idx is None:
          q_idx_vec[j] = floor_if_packed(
            index_tensor[i + j][q_idx_pos], qhead_per_kvhead
          )
        kv_idx_vec[j] = index_tensor[i + j][kv_idx_pos]

    # Convert to SSA for score_mod call
    score_ssa = score_vec.load()
    kv_idx_ssa = kv_idx_vec.load()
    if cutlass.const_expr(constant_q_idx is None):
      q_idx_ssa = q_idx_vec.load()
    else:
      # NB we do not apply Pack-GQA division here, as constant_q_idx is assumed to already be logical
      q_idx_const = constant_q_idx
      q_idx_ssa = scalar_to_ssa(q_idx_const,
                                cutlass.Int32).broadcast_to((vec_size, ))

    # Compute head_idx_ssa: per-element for Pack-GQA with non-constant q_idx, constant otherwise
    if cutlass.const_expr(qhead_per_kvhead > 1 and constant_q_idx is None):
      head_idx_ssa = head_idx_vec.load()
    else:
      head_idx_ssa = scalar_to_ssa(head_idx,
                                   cutlass.Int32).broadcast_to((vec_size, ))

    aux_args = []
    if cutlass.const_expr(aux_tensors is not None):
      aux_args = aux_tensors

    post_mod_scores = score_mod(
      score_ssa,
      batch_idx_ssa,
      head_idx_ssa,
      q_idx=q_idx_ssa,
      kv_idx=kv_idx_ssa,
      seqlen_info=seqlen_info,
      aux_tensors=aux_args,
    )

    # Write back modified scores
    score_vec.store(post_mod_scores)
    for j in cutlass.range(vec_size, unroll_full=True):
      score_tensor[i + j] = score_vec[j]


# cal backward for score mode in forward
@cute.jit
def apply_score_mod_bwd_inner(
  grad_tensor,
  score_tensor,
  index_tensor,
  score_mod_bwd: cutlass.Constexpr,
  batch_idx,
  head_idx,
  softmax_scale,
  vec_size: cutlass.Constexpr,
  qk_acc_dtype: cutlass.Constexpr,
  aux_tensors,
  fastdiv_mods,
  seqlen_info,
  constant_q_idx: cutlass.Constexpr,
  qhead_per_kvhead: cutlass.Constexpr[int] = 1,
  transpose_indices: cutlass.Constexpr[bool] = False,
):
  """Apply backward score modification (joint graph).

    Args:
        grad_tensor: in/out: dlogits rewritten in-place with d(scaled_scores)
        score_tensor: pre-mod scores (unscaled QK tile), scaled by softmax_scale internally
        index_tensor: Index positions (same as forward)
        score_mod_bwd: The backward score modification function (joint graph)
        batch_idx: Batch index
        head_idx: Head index
        softmax_scale: Scale to apply to score_tensor
        vec_size: Vector size for processing elements
        qk_acc_dtype: Data type for accumulator
        aux_tensors: Optional aux_tensors for FlexAttention
        fastdiv_mods: Tuple of (seqlen_q_divmod, seqlen_k_divmod) for wrapping
        seqlen_info: Sequence length info
        constant_q_idx: If provided, use this constant for all q_idx values
        qhead_per_kvhead: Pack-GQA replication factor
        transpose_indices: If True, swap q_idx/kv_idx in index_tensor
    """
  # Index positions in the index_tensor tuple
  # Forward: index_tensor[...][0] = q_idx, index_tensor[...][1] = kv_idx
  # Backward (transposed): index_tensor[...][0] = kv_idx, index_tensor[...][1] = q_idx
  if cutlass.const_expr(transpose_indices):
    q_idx_pos = cutlass.const_expr(1)
    kv_idx_pos = cutlass.const_expr(0)
  else:
    q_idx_pos = cutlass.const_expr(0)
    kv_idx_pos = cutlass.const_expr(1)
  n_vals = cutlass.const_expr(cute.size(grad_tensor.shape))
  grad_vec = cute.make_rmem_tensor(vec_size, qk_acc_dtype)
  score_vec = cute.make_rmem_tensor(vec_size, qk_acc_dtype)
  kv_idx_vec = cute.make_rmem_tensor(vec_size, cutlass.Int32)
  batch_idx_ssa = scalar_to_ssa(batch_idx,
                                cutlass.Int32).broadcast_to((vec_size, ))
  q_idx_vec = cute.make_rmem_tensor(vec_size, cutlass.Int32)

  # For Pack-GQA with non-constant q_idx, we need per-element head indices
  if cutlass.const_expr(qhead_per_kvhead > 1 and constant_q_idx is None):
    head_idx_vec = cute.make_rmem_tensor(vec_size, cutlass.Int32)

  for i in cutlass.range(0, n_vals, vec_size, unroll_full=True):
    for j in cutlass.range(vec_size, unroll_full=True):
      grad_vec[j] = grad_tensor[i + j]
      # Scale score so joint graph sees same value as forward score_mod
      score_vec[j] = score_tensor[i + j] * softmax_scale

      if cutlass.const_expr(qhead_per_kvhead > 1 and constant_q_idx is None):
        q_idx_packed = index_tensor[i + j][q_idx_pos]
        q_idx_logical = q_idx_packed // qhead_per_kvhead
        head_offset = q_idx_packed - q_idx_logical * qhead_per_kvhead
        head_idx_vec[j] = head_idx * qhead_per_kvhead + head_offset

      if cutlass.const_expr(
        aux_tensors is not None and fastdiv_mods is not None
      ):
        if cutlass.const_expr(constant_q_idx is None):
          seqlen_q_divmod, seqlen_k_divmod = fastdiv_mods
          q_idx_floored = floor_if_packed(
            index_tensor[i + j][q_idx_pos], qhead_per_kvhead
          )
          _, q_idx_wrapped = divmod(q_idx_floored, seqlen_q_divmod)
          q_idx_vec[j] = q_idx_wrapped
        else:
          _, seqlen_k_divmod = fastdiv_mods

        _, kv_idx_wrapped = divmod(
          index_tensor[i + j][kv_idx_pos], seqlen_k_divmod
        )
        kv_idx_vec[j] = kv_idx_wrapped
      else:
        # No bounds checking - direct indexing
        if constant_q_idx is None:
          q_idx_vec[j] = floor_if_packed(
            index_tensor[i + j][q_idx_pos], qhead_per_kvhead
          )
        kv_idx_vec[j] = index_tensor[i + j][kv_idx_pos]

    grad_ssa = grad_vec.load()
    score_ssa = score_vec.load()
    kv_idx_ssa = kv_idx_vec.load()

    if cutlass.const_expr(constant_q_idx is None):
      q_idx_ssa = q_idx_vec.load()
    else:
      q_idx_ssa = scalar_to_ssa(constant_q_idx,
                                cutlass.Int32).broadcast_to((vec_size, ))

    if cutlass.const_expr(qhead_per_kvhead > 1 and constant_q_idx is None):
      head_idx_ssa = head_idx_vec.load()
    else:
      head_idx_ssa = scalar_to_ssa(head_idx,
                                   cutlass.Int32).broadcast_to((vec_size, ))

    aux_args = []
    if cutlass.const_expr(aux_tensors is not None):
      aux_args = aux_tensors

    grad_out_ssa = score_mod_bwd(
      grad_ssa,
      score_ssa,
      batch_idx_ssa,
      head_idx_ssa,
      q_idx=q_idx_ssa,
      kv_idx=kv_idx_ssa,
      seqlen_info=seqlen_info,
      aux_tensors=aux_args,
    )

    grad_vec.store(grad_out_ssa)
    for j in cutlass.range(vec_size, unroll_full=True):
      grad_tensor[i + j] = grad_vec[j]

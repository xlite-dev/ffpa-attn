# This file is adapted from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/mask.py
# Copyright (c) 2025, Tri Dao.
# SM90-only trimmed version of mask.py for the Hopper training port.
#
# Removed vs upstream:
#   - apply_mask_sm100 method (SM100 forward, uses thr_tmem_load / rBitmask / head_divmod)
#   - apply_mask_sm100_transposed method (SM100 backward, uses thr_tmem_load / row_to_r2p_idx)
#   - row_to_r2p_idx helper function (only used by apply_mask_sm100_transposed)
#
# Retained primitives (all referenced inside apply_mask):
#   - r2p_bitmask_below / r2p_bitmask_above (R2P bitmask primitives)
#   - mask_r2p_lambda (R2P masking kernel)
#   - sm90_col_to_r2p_idx (SM90 MMA column-to-R2P coordinate transform)
#
# AttentionMask.apply_mask call sites in this repo:
#   - _ffpa_fwd_d512_sm90.py:982   (seqlen-only / causal / local / mask_mod / PackGQA)
#   - _ffpa_dq_d512_sm90.py:1027, 1330   (seqlen-only / causal)
#   - _ffpa_dkdv_d512_sm90.py:982, 1233  (seqlen-only / causal)
#
# swap_AB=True branch of apply_mask is reserved for a future SM90 bwd SdP swap_AB path
# (no current call sites). See `swap_AB` field below and the swap_AB arm inside apply_mask.
#
# Dependencies:
#   - quack.layout_utils.reshape_acc_to_mn (MMA accumulator reshape)
#   - flash_attn.cute.utils: shr_u32, shl_u32, scalar_to_ssa, ssa_to_scalar, shuffle_sync
#   - flash_attn.cute.seqlen_info: SeqlenInfoQK (held as member)

from typing import Optional, Callable, TypeAlias
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint32, const_expr

from quack import layout_utils
from . import shr_u32, shl_u32, shuffle_sync, scalar_to_ssa, ssa_to_scalar
from .seqlen_info import SeqlenInfoQK

MaskGenFn: TypeAlias = Callable[[int], Uint32]
MASK_R2P_CHUNK_SIZE: int = 32


@cute.jit
def r2p_bitmask_below(limit: Int32, s: int) -> Uint32:
  """32-bit R2P bitmask keeping positions < limit (exclusive upper bound).

    Positions 0..limit-1 in chunk `s` get bit=1 (keep), the rest bit=0 (mask).
    Uses inline PTX to avoid shift-by-type-width UB.
    """
  m = max((s + 1) * MASK_R2P_CHUNK_SIZE - limit, 0)
  return shr_u32(Uint32(0xFFFFFFFF), Uint32(m))


@cute.jit
def r2p_bitmask_above(limit: Int32, s: int) -> Uint32:
  """32-bit R2P bitmask keeping positions >= limit (inclusive lower bound).

    Positions limit..31 in chunk `s` get bit=1 (keep), the rest bit=0 (mask).
    Uses inline PTX to avoid shift-by-type-width UB.
    """
  n = max(limit - s * MASK_R2P_CHUNK_SIZE, 0)
  return shl_u32(Uint32(0xFFFFFFFF), Uint32(n))


@cute.jit
def mask_r2p_lambda(
  X: cute.Tensor,
  mask_gen_fn: cutlass.Constexpr[MaskGenFn],
  rank1: bool = False,
) -> None:
  """Apply R2P masking with a custom bitmask generator.

    mask_gen_fn(chunk_idx: constexpr int) -> Uint32:
        Returns a 32-bit bitmask for the chunk. Bit i set means column
        chunk_idx * chunk_size + i is KEPT; bit i clear means masked to -inf.
    """
  ncol = const_expr(cute.size(X.shape[cute.rank(X) - 1]) if not rank1 else cute.size(X.shape))
  # 32-column chunks. The mask_gen_fn returns a Uint32 bitmask (1=keep).
  CHUNK_SIZE = MASK_R2P_CHUNK_SIZE
  for s in cutlass.range_constexpr(cute.ceil_div(ncol, CHUNK_SIZE)):
    mask = mask_gen_fn(s)
    # This needs to be range_constexpr, o/w the compiler can't generate the R2P instruction
    for i in cutlass.range_constexpr(min(CHUNK_SIZE, ncol - s * CHUNK_SIZE)):
      in_bound = cutlass.Boolean(mask & (Uint32(1) << i))
      c = s * CHUNK_SIZE + i
      if const_expr(rank1):
        X[c] = X[c] if in_bound else -Float32.inf
      else:
        for r in cutlass.range_constexpr(cute.size(X.shape[0])):
          X[r, c] = X[r, c] if in_bound else -Float32.inf


@cute.jit
def sm90_col_to_r2p_idx(col_limit: Int32) -> Int32:
  """Transform SM90 MMA column coordinate to R2P element index.

    SM90 MMA accumulator column indices are non-contiguous: 0, 1, 8, 9, 16, 17, ...
    Element indices are contiguous: 0, 1, 2, 3, 4, 5, ...
    This converts a column-space threshold to element-space for r2p_bitmask_below/above.
    """
  return col_limit // 8 * 2 + min(col_limit % 8, 2)


@dataclass(frozen=True)
class AttentionMask:
  tile_m: cutlass.Constexpr[int]
  tile_n: cutlass.Constexpr[int]
  seqlen_info: SeqlenInfoQK
  window_size_left: Optional[Int32] = None
  window_size_right: Optional[Int32] = None
  qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1  # only pass in if we're doing PackGQA
  # Reserved: enables the bwd SdP swap_AB path of apply_mask (currently no SM90 call site sets this True).
  swap_AB: cutlass.Constexpr[bool] = False

  @property
  def seqlen_q(self) -> Int32:
    return self.seqlen_info.seqlen_q

  @property
  def seqlen_k(self) -> Int32:
    return self.seqlen_info.seqlen_k

  @cute.jit
  def apply_mask(
    self,
    acc_S: cute.Tensor,
    batch_idx: cutlass.Int32,
    head_idx: cutlass.Int32,
    m_block: cutlass.Int32,
    n_block: cutlass.Int32,
    thr_mma: cute.TiledMma,
    mask_seqlen: cutlass.Constexpr[bool],
    mask_causal: cutlass.Constexpr[bool],
    mask_local: cutlass.Constexpr[bool] = False,
    mask_mod: cutlass.Constexpr[Optional[Callable]] = None,
    aux_tensors: Optional[list] = None,
    fastdiv_mods=(None, None),
  ) -> None:
    assert not (mask_causal and mask_local), "mask_causal and mask_local cannot be both True"
    acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S, transpose=self.swap_AB)
    acc_shape = (self.tile_m, self.tile_n)
    cS = cute.make_identity_tensor(acc_shape if not self.swap_AB else acc_shape[::-1])
    tScS_mn = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(cS), transpose=self.swap_AB)
    # We use t0ScS as these indices are known at compile time. We then must subtract the
    # column limit by the thread column offset.
    t0ScS_mn = layout_utils.reshape_acc_to_mn(thr_mma.get_slice(0).partition_C(cS), transpose=self.swap_AB)
    ROW = 0 if const_expr(not self.swap_AB) else 1
    COL = 1 if const_expr(not self.swap_AB) else 0
    thr_col_offset = tScS_mn[0][COL]
    # To handle edge cases of completely masked out rows where n_block_max = 0,
    # we treat negative n_blocks as 0th n_block
    # TODO: find more transparent solution
    if n_block < 0:
      n_block = 0
    seqlenk_col_limit = self.seqlen_k - n_block * self.tile_n - thr_col_offset
    if const_expr(not mask_causal and not mask_local and mask_mod is None):
      if const_expr(mask_seqlen):
        r2p = const_expr(not self.swap_AB)
        if const_expr(not r2p):
          # traverse column index.
          for c in cutlass.range(cute.size(tScS_mn.shape[1]), unroll_full=True):
            oob = t0ScS_mn[0, c][COL] >= seqlenk_col_limit
            for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
              acc_S_mn[r, c] = -Float32.inf if oob else acc_S_mn[r, c]
        else:
          seqlenk_col_limit_r2p = sm90_col_to_r2p_idx(seqlenk_col_limit)
          mask_r2p_lambda(acc_S_mn, lambda s: r2p_bitmask_below(seqlenk_col_limit_r2p, s))

    elif const_expr(not mask_causal and not mask_local and mask_mod is not None):  # FlexAttention mask mod
      nrow = const_expr(cute.size(tScS_mn.shape[0]))
      ncol = const_expr(cute.size(tScS_mn.shape[1]))
      has_fastdiv = const_expr(fastdiv_mods is not None and fastdiv_mods[0] is not None and fastdiv_mods[1] is not None)
      wrap_aux_indices = const_expr(has_fastdiv and mask_seqlen and const_expr(aux_tensors is not None))

      for r in cutlass.range_constexpr(nrow):
        # Respect swap_AB: ROW/COL determine which coordinate component corresponds to Q/KV.
        local_row = tScS_mn[r, 0][ROW]
        global_row_idx = local_row + m_block * self.tile_m
        row_for_mod = global_row_idx
        head_idx_for_mod = head_idx
        if const_expr(self.qhead_per_kvhead_packgqa != 1):
          head_offset = global_row_idx % self.qhead_per_kvhead_packgqa
          head_idx_for_mod = head_idx * self.qhead_per_kvhead_packgqa + head_offset
          row_for_mod = global_row_idx // self.qhead_per_kvhead_packgqa
        row_for_seqlen = row_for_mod
        if const_expr(wrap_aux_indices):
          _, row_for_mod = divmod(row_for_mod, fastdiv_mods[0])

        for col in cutlass.range_constexpr(ncol):
          col_idx_local = t0ScS_mn[0, col][COL]
          # Convert to absolute column index
          global_col_idx = thr_col_offset + col_idx_local + n_block * self.tile_n
          col_for_mod = global_col_idx
          if const_expr(wrap_aux_indices):
            _, col_for_mod = divmod(global_col_idx, fastdiv_mods[1])

          batch_idx_ssa = scalar_to_ssa(batch_idx, cutlass.Int32)
          head_idx_ssa = scalar_to_ssa(head_idx_for_mod, cutlass.Int32)
          q_idx_ssa = scalar_to_ssa(row_for_mod, cutlass.Int32)
          kv_idx_ssa = scalar_to_ssa(col_for_mod, cutlass.Int32)
          mask_value = mask_mod(
            batch_idx_ssa,
            head_idx_ssa,
            q_idx_ssa,
            kv_idx_ssa,
            self.seqlen_info,
            aux_tensors,
          )
          cond = cutlass.Boolean(ssa_to_scalar(mask_value))
          if const_expr(mask_seqlen):
            out_of_bounds = (row_for_seqlen >= self.seqlen_q) or (global_col_idx >= self.seqlen_k)
            if out_of_bounds:
              acc_S_mn[r, col] = -cutlass.Float32.inf
            else:
              acc_S_mn[r, col] = acc_S_mn[r, col] if cond else -cutlass.Float32.inf
          else:
            acc_S_mn[r, col] = acc_S_mn[r, col] if cond else -cutlass.Float32.inf

    else:  # Causal or local
      if const_expr(not self.swap_AB):
        # If PackGQA, we split the work of compute divmod among threads in the same row
        threads_per_row = thr_mma.tv_layout_C.shape[0][0]
        mma_m_idx = None
        if const_expr(self.qhead_per_kvhead_packgqa != 1):
          assert not self.swap_AB, "swap_AB with PackGQA not supported yet"
          assert cute.arch.WARP_SIZE % threads_per_row == 0, "threads_per_row must divide WARP_SIZE"
          assert cute.size(acc_S_mn.shape[0]) <= threads_per_row
          tidx = thr_mma.thr_idx
          mma_m_idx = (m_block * self.tile_m + tScS_mn[tidx % threads_per_row, 0][0]) // self.qhead_per_kvhead_packgqa
        causal_row_offset = 1 + self.seqlen_k - n_block * self.tile_n - self.seqlen_q - thr_col_offset
        if const_expr(mask_causal):
          r2p = const_expr(not self.swap_AB)  # R2P trick, see apply_mask_sm100
          for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
            # get the column index limit based on current row. Only consider the row index, so the column index sets to 0.
            if const_expr(self.qhead_per_kvhead_packgqa == 1):
              row_idx = tScS_mn[r, 0][0] + m_block * self.tile_m
            else:
              row_idx = shuffle_sync(mma_m_idx, r % threads_per_row, width=threads_per_row)
            col_limit_right = row_idx + causal_row_offset
            if const_expr(mask_seqlen):
              col_limit_right = cutlass.min(col_limit_right, seqlenk_col_limit)
            if const_expr(not r2p):
              # traverse column index.
              for c in cutlass.range(cute.size(tScS_mn.shape[1]), unroll_full=True):
                acc_S_mn[r, c] = (-Float32.inf if t0ScS_mn[0, c][1] >= col_limit_right else acc_S_mn[r, c])
            else:
              col_limit_r2p = sm90_col_to_r2p_idx(col_limit_right)
              mask_r2p_lambda(
                acc_S_mn[r, None],
                lambda s: r2p_bitmask_below(col_limit_r2p, s),
                rank1=True,
              )
        else:  # Local
          local_row_offset_right = (
            causal_row_offset + self.window_size_right if const_expr(self.window_size_right is not None) else None
          )
          local_row_offset_left = (
            causal_row_offset - 1 - self.window_size_left if const_expr(self.window_size_left is not None) else None
          )
          r2p_local = const_expr(not self.swap_AB)
          for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
            if const_expr(self.qhead_per_kvhead_packgqa == 1):
              row_idx = tScS_mn[r, 0][0] + m_block * self.tile_m
            else:
              row_idx = shuffle_sync(mma_m_idx, r % threads_per_row, width=threads_per_row)
            if const_expr(self.window_size_right is not None):
              col_limit_right = row_idx + local_row_offset_right
            else:
              col_limit_right = self.tile_n
            if const_expr(mask_seqlen):
              col_limit_right = cutlass.min(col_limit_right, seqlenk_col_limit)
            col_limit_left = (row_idx + local_row_offset_left if const_expr(self.window_size_left is not None) else 0)
            if const_expr(not r2p_local):
              # traverse column index.
              for c in cutlass.range(cute.size(tScS_mn.shape[1]), unroll_full=True):
                col_idx = t0ScS_mn[0, c][1]
                if col_idx >= col_limit_right or col_idx < col_limit_left:
                  acc_S_mn[r, c] = -Float32.inf
            else:
              col_limit_right_r2p = sm90_col_to_r2p_idx(col_limit_right)
              col_limit_left_r2p = sm90_col_to_r2p_idx(col_limit_left)

              def mask_gen_fn(s: int) -> Uint32:
                return r2p_bitmask_below(col_limit_right_r2p, s) & r2p_bitmask_above(col_limit_left_r2p, s)

              mask_r2p_lambda(acc_S_mn[r, None], mask_gen_fn, rank1=True)
      # Reserved: future SM90 bwd SdP swap_AB path; not reached under current call sites.
      else:  # swap_AB (backward SdP path)
        assert self.qhead_per_kvhead_packgqa == 1
        thr_row_offset = tScS_mn[0][ROW]
        causal_row_offset = seqlenk_col_limit - self.seqlen_q + m_block * self.tile_m + thr_row_offset
        if const_expr(mask_causal):
          for c in cutlass.range(cute.size(tScS_mn.shape[1]), unroll_full=True):
            col0 = t0ScS_mn[0, c][COL]
            # If col0 is beyond the column limit, we want to mask out the entire
            # column, by setting row limit to be self.tile_m.
            row_limit_top = (self.tile_m if col0 >= seqlenk_col_limit and mask_seqlen else col0 - causal_row_offset)
            for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
              acc_S_mn[r, c] = -Float32.inf if t0ScS_mn[r, 0][ROW] < row_limit_top else acc_S_mn[r, c]
        else:
          for c in cutlass.range(cute.size(tScS_mn.shape[1]), unroll_full=True):
            col0 = t0ScS_mn[0, c][COL]
            # If col0 is beyond the column limit, we want to mask out the entire
            # column, by setting row limit to be self.tile_m.
            row_limit_top = (
              self.tile_m if col0 >= seqlenk_col_limit and mask_seqlen else (
                col0 - causal_row_offset -
                self.window_size_right if const_expr(self.window_size_right is not None) else 0
              )
            )
            row_limit_bot = (
              col0 - causal_row_offset +
              self.window_size_left if const_expr(self.window_size_left is not None) else self.tile_m
            )
            for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
              row_idx = t0ScS_mn[r, 0][ROW]
              acc_S_mn[r, c] = (-Float32.inf if row_idx < row_limit_top or row_idx > row_limit_bot else acc_S_mn[r, c])

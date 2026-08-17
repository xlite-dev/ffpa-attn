# This file is adapted from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/mask.py
# Copyright (c) 2025, Tri Dao.
# Was an SM90-only trim; the SM100 pieces it had removed were restored for the SM100 D512 port from donor revision b3012e4a539 -- apply_mask_sm100_transposed, row_to_r2p_idx / call_mask_mod, Sm100MaskEnum / Sm100FusedMask -- and are marked with their own banners below.
# Withdrawn again as unreachable, restorable from that revision: apply_mask_sm100_fragment, apply_mask_sm100 with the apply_mask_mod_sm100_* / apply_packed_mask_chunk cluster it alone reached, and Sm100FusedMask's nine trip-count methods (with the BlockInfo and typing.Tuple imports that half alone needed).
#
# Retained primitives (all referenced inside apply_mask):
#   - r2p_bitmask_below / r2p_bitmask_above (R2P bitmask primitives)
#   - mask_r2p_lambda (R2P masking kernel)
#   - sm90_col_to_r2p_idx (SM90 MMA column-to-R2P coordinate transform)
#
# AttentionMask.apply_mask call sites in this repo:
#   - _fwd_d512_sm90.py:982   (seqlen-only / causal / local / mask_mod / PackGQA)
#   - _dq_d512_sm90.py:1027, 1330   (seqlen-only / causal)
#   - _dkdv_d512_sm90.py:982, 1233  (seqlen-only / causal)
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
import enum

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint32, const_expr
from cutlass.cutlass_dsl import min as dsl_min

from quack import layout_utils
from . import (
  AuxData,
  shr_u32,
  shl_u32,
  shuffle_sync,
  scalar_to_ssa,
  ssa_to_scalar,
)
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
  ncol = const_expr(
    cute.size(X.shape[cute.rank(X) - 1]) if not rank1 else cute.size(X.shape)
  )
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
  qhead_per_kvhead_packgqa: cutlass.Constexpr[
    int] = 1  # only pass in if we're doing PackGQA
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
    assert not (
      mask_causal and mask_local
    ), "mask_causal and mask_local cannot be both True"
    acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S, transpose=self.swap_AB)
    acc_shape = (self.tile_m, self.tile_n)
    cS = cute.make_identity_tensor(
      acc_shape if not self.swap_AB else acc_shape[::-1]
    )
    tScS_mn = layout_utils.reshape_acc_to_mn(
      thr_mma.partition_C(cS), transpose=self.swap_AB
    )
    # We use t0ScS as these indices are known at compile time. We then must subtract the
    # column limit by the thread column offset.
    t0ScS_mn = layout_utils.reshape_acc_to_mn(
      thr_mma.get_slice(0).partition_C(cS), transpose=self.swap_AB
    )
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
            for r in cutlass.range(
              cute.size(tScS_mn.shape[0]), unroll_full=True
            ):
              acc_S_mn[r, c] = -Float32.inf if oob else acc_S_mn[r, c]
        else:
          seqlenk_col_limit_r2p = sm90_col_to_r2p_idx(seqlenk_col_limit)
          mask_r2p_lambda(
            acc_S_mn, lambda s: r2p_bitmask_below(seqlenk_col_limit_r2p, s)
          )

    elif const_expr(
      not mask_causal and not mask_local and mask_mod is not None
    ):  # FlexAttention mask mod
      nrow = const_expr(cute.size(tScS_mn.shape[0]))
      ncol = const_expr(cute.size(tScS_mn.shape[1]))
      has_fastdiv = const_expr(
        fastdiv_mods is not None and fastdiv_mods[0] is not None
        and fastdiv_mods[1] is not None
      )
      wrap_aux_indices = const_expr(
        has_fastdiv and mask_seqlen and const_expr(aux_tensors is not None)
      )

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
            out_of_bounds = (row_for_seqlen >= self.seqlen_q
                             ) or (global_col_idx >= self.seqlen_k)
            if out_of_bounds:
              acc_S_mn[r, col] = -cutlass.Float32.inf
            else:
              acc_S_mn[r, col] = acc_S_mn[r,
                                          col] if cond else -cutlass.Float32.inf
          else:
            acc_S_mn[r, col] = acc_S_mn[r,
                                        col] if cond else -cutlass.Float32.inf

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
          mma_m_idx = (
            m_block * self.tile_m + tScS_mn[tidx % threads_per_row, 0][0]
          ) // self.qhead_per_kvhead_packgqa
        causal_row_offset = 1 + self.seqlen_k - n_block * self.tile_n - self.seqlen_q - thr_col_offset
        if const_expr(mask_causal):
          r2p = const_expr(not self.swap_AB)  # R2P trick, see apply_mask_sm100
          for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
            # get the column index limit based on current row. Only consider the row index, so the column index sets to 0.
            if const_expr(self.qhead_per_kvhead_packgqa == 1):
              row_idx = tScS_mn[r, 0][0] + m_block * self.tile_m
            else:
              row_idx = shuffle_sync(
                mma_m_idx, r % threads_per_row, width=threads_per_row
              )
            col_limit_right = row_idx + causal_row_offset
            if const_expr(mask_seqlen):
              col_limit_right = cutlass.min(col_limit_right, seqlenk_col_limit)
            if const_expr(not r2p):
              # traverse column index.
              for c in cutlass.range(
                cute.size(tScS_mn.shape[1]), unroll_full=True
              ):
                acc_S_mn[r, c] = (
                  -Float32.inf
                  if t0ScS_mn[0, c][1] >= col_limit_right else acc_S_mn[r, c]
                )
            else:
              col_limit_r2p = sm90_col_to_r2p_idx(col_limit_right)
              mask_r2p_lambda(
                acc_S_mn[r, None],
                lambda s: r2p_bitmask_below(col_limit_r2p, s),
                rank1=True,
              )
        else:  # Local
          local_row_offset_right = (
            causal_row_offset + self.window_size_right
            if const_expr(self.window_size_right is not None) else None
          )
          local_row_offset_left = (
            causal_row_offset - 1 - self.window_size_left
            if const_expr(self.window_size_left is not None) else None
          )
          r2p_local = const_expr(not self.swap_AB)
          for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
            if const_expr(self.qhead_per_kvhead_packgqa == 1):
              row_idx = tScS_mn[r, 0][0] + m_block * self.tile_m
            else:
              row_idx = shuffle_sync(
                mma_m_idx, r % threads_per_row, width=threads_per_row
              )
            if const_expr(self.window_size_right is not None):
              col_limit_right = row_idx + local_row_offset_right
            else:
              col_limit_right = self.tile_n
            if const_expr(mask_seqlen):
              col_limit_right = cutlass.min(col_limit_right, seqlenk_col_limit)
            col_limit_left = (
              row_idx + local_row_offset_left
              if const_expr(self.window_size_left is not None) else 0
            )
            if const_expr(not r2p_local):
              # traverse column index.
              for c in cutlass.range(
                cute.size(tScS_mn.shape[1]), unroll_full=True
              ):
                col_idx = t0ScS_mn[0, c][1]
                if col_idx >= col_limit_right or col_idx < col_limit_left:
                  acc_S_mn[r, c] = -Float32.inf
            else:
              col_limit_right_r2p = sm90_col_to_r2p_idx(col_limit_right)
              col_limit_left_r2p = sm90_col_to_r2p_idx(col_limit_left)

              def mask_gen_fn(s: int) -> Uint32:
                return r2p_bitmask_below(
                  col_limit_right_r2p, s
                ) & r2p_bitmask_above(col_limit_left_r2p, s)

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
            row_limit_top = (
              self.tile_m if col0 >= seqlenk_col_limit and mask_seqlen else
              col0 - causal_row_offset
            )
            for r in cutlass.range(
              cute.size(tScS_mn.shape[0]), unroll_full=True
            ):
              acc_S_mn[r, c] = -Float32.inf if t0ScS_mn[
                r, 0][ROW] < row_limit_top else acc_S_mn[r, c]
        else:
          for c in cutlass.range(cute.size(tScS_mn.shape[1]), unroll_full=True):
            col0 = t0ScS_mn[0, c][COL]
            # If col0 is beyond the column limit, we want to mask out the entire
            # column, by setting row limit to be self.tile_m.
            row_limit_top = (
              self.tile_m if col0 >= seqlenk_col_limit and mask_seqlen else (
                col0 - causal_row_offset - self.window_size_right
                if const_expr(self.window_size_right is not None) else 0
              )
            )
            row_limit_bot = (
              col0 - causal_row_offset + self.window_size_left
              if const_expr(self.window_size_left is not None) else self.tile_m
            )
            for r in cutlass.range(
              cute.size(tScS_mn.shape[0]), unroll_full=True
            ):
              row_idx = t0ScS_mn[r, 0][ROW]
              acc_S_mn[r, c] = (
                -Float32.inf if row_idx < row_limit_top
                or row_idx > row_limit_bot else acc_S_mn[r, c]
              )

  # SM100 (Blackwell) entry point, reached by the D512 dV kernel.  It masks a different object from apply_mask above -- the register fragment a TMEM load produced, paired with the identity coordinate tensor from the same partitioning -- so equal element counts do not make the two interchangeable.

  @cute.jit
  def apply_mask_sm100_transposed(
    self,
    acc_S: cute.Tensor,
    tScS_t2r: cute.Tensor,
    t0ScS_t2r: cute.Tensor,
    m_block: cutlass.Int32,
    n_block: cutlass.Int32,
    mask_seqlen: cutlass.Constexpr,
    mask_causal: cutlass.Constexpr,
    mask_local: cutlass.Constexpr,
    mask_mod: cutlass.Constexpr[Optional[Callable]] = None,
    batch_idx: Int32 = None,
    head_idx: Int32 = None,
    aux_data: AuxData = AuxData(),
    fastdiv_mods=(None, None),
    is_full_block: bool = False,
    check_m_boundary: bool = True,
  ) -> None:
    """
        Backward pass: mask S = K @ Q.T where n_block tiles seqlen_k and m_block tiles seqlen_q.

        Coordinate convention:
        - ROW corresponds to Q (m_block)
        - COL corresponds to KV (n_block)

        is_full_block: If True, skip mask_mod (all elements valid). Only apply seqlen masking.
        check_m_boundary: If False, skip seqlen_q boundary check (optimization for non-boundary m_blocks).
                          When iterating m_blocks in forward order, only the last m_block may be partial.
        """
    assert not (
      mask_causal and mask_local
    ), "mask_causal and mask_local cannot be both True"
    ROW = 0 if const_expr(not self.swap_AB) else 1
    COL = 1 if const_expr(not self.swap_AB) else 0
    # assert t0ScS_t2r[0][COL] == 0, "col0 == 0" # tmp comment for 2-cta bwd
    thr_col_offset = tScS_t2r[0][COL]
    seqlenk_col_limit = self.seqlen_k - n_block * self.tile_n - thr_col_offset

    if const_expr(not mask_causal and not mask_local and mask_mod is not None):
      # Block sparse case with mask_mod (backward)
      #
      # Coordinate convention: ROW → Q (m_block), COL → KV (n_block).
      # These already account for swap_AB.
      #
      # FULL blocks: mask_mod returns True for all elements, so skip it.
      #   Still need seqlen bounds check (elements may be OOB on last m_block).
      # PARTIAL blocks: apply mask_mod element-wise, then seqlen bounds.
      if is_full_block:
        if const_expr(mask_seqlen):
          if seqlenk_col_limit <= 0:
            # Entire tile is OOB for K
            for i in cutlass.range(cute.size(acc_S.shape), unroll_full=True):
              acc_S[i] = -cutlass.Float32.inf
          elif check_m_boundary:
            # Last m_block: check Q and K boundaries
            ncol = const_expr(cute.size(tScS_t2r.shape))
            for i in cutlass.range_constexpr(ncol):
              row_coord = tScS_t2r[i][ROW]
              col_coord = tScS_t2r[i][COL]
              global_q = row_coord + m_block * self.tile_m
              global_kv = col_coord + n_block * self.tile_n
              q_out_of_bounds = global_q >= self.seqlen_q
              kv_out_of_bounds = global_kv >= self.seqlen_k
              out_of_bounds = q_out_of_bounds or kv_out_of_bounds
              acc_S[i] = -cutlass.Float32.inf if out_of_bounds else acc_S[i]
      else:
        # Partial block
        has_fastdiv = const_expr(
          fastdiv_mods is not None and fastdiv_mods[0] is not None
          and fastdiv_mods[1] is not None
        )
        wrap_aux_indices = const_expr(
          has_fastdiv and mask_seqlen
          and const_expr(aux_data.tensors is not None)
        )
        batch_idx_ssa = scalar_to_ssa(batch_idx, cutlass.Int32)
        head_idx_ssa = scalar_to_ssa(head_idx, cutlass.Int32)

        ncol = const_expr(cute.size(tScS_t2r.shape))
        for i in cutlass.range_constexpr(ncol):
          row_coord = tScS_t2r[i][ROW]
          col_coord = tScS_t2r[i][COL]
          global_q = row_coord + m_block * self.tile_m
          global_kv = col_coord + n_block * self.tile_n

          q_idx_for_mod = global_q
          kv_idx_for_mod = global_kv
          if const_expr(wrap_aux_indices):
            _, q_idx_for_mod = divmod(global_q, fastdiv_mods[0])
            _, kv_idx_for_mod = divmod(global_kv, fastdiv_mods[1])

          q_idx_ssa = scalar_to_ssa(q_idx_for_mod, cutlass.Int32)
          kv_idx_ssa = scalar_to_ssa(kv_idx_for_mod, cutlass.Int32)

          mask_value = call_mask_mod(
            mask_mod,
            batch_idx_ssa,
            head_idx_ssa,
            q_idx_ssa,
            kv_idx_ssa,
            self.seqlen_info,
            aux_data,
          )
          cond = cutlass.Boolean(ssa_to_scalar(mask_value))
          acc_S[i] = acc_S[i] if cond else -cutlass.Float32.inf

          if const_expr(mask_seqlen):
            # check_m_boundary=False skips q check for non-boundary m_blocks
            q_out_of_bounds = check_m_boundary and (global_q >= self.seqlen_q)
            kv_out_of_bounds = global_kv >= self.seqlen_k
            out_of_bounds = q_out_of_bounds or kv_out_of_bounds
            acc_S[i] = -cutlass.Float32.inf if out_of_bounds else acc_S[i]

    elif const_expr(not mask_causal and not mask_local):
      if const_expr(mask_seqlen):
        if seqlenk_col_limit <= 0:
          for i in cutlass.range(cute.size(acc_S.shape), unroll_full=True):
            acc_S[i] = -cutlass.Float32.inf
    else:  # Causal or local
      thr_row_offset = tScS_t2r[0][ROW]
      seqlenq_row_limit = self.seqlen_q - m_block * self.tile_m - thr_row_offset
      causal_offset = seqlenq_row_limit - seqlenk_col_limit
      if const_expr(mask_causal):
        # tidx = cute.arch.thread_idx()[0] % 256
        # if tidx < 32:
        #     cute.printf("tidx = {}, {} {}, {} {}", tidx, tScS_t2r[0][0], tScS_t2r[0][1], tScS_t2r[1][0], tScS_t2r[1][1])
        row_limit_top = causal_offset
        if const_expr(mask_seqlen):
          # If col is beyond the column limit, we want to mask out the entire
          # column, by setting row limit to be self.tile_m.
          if seqlenk_col_limit <= 0:
            row_limit_top = self.tile_m
        r2p = True
        if const_expr(not r2p):
          for i in cutlass.range(cute.size(acc_S.shape), unroll_full=True):
            acc_S[i] = (
              -cutlass.Float32.inf
              if t0ScS_t2r[i][ROW] < row_limit_top else acc_S[i]
            )
        else:
          num_rep = cute.size(tScS_t2r, mode=[0])  # 16 or 32
          num_wg = 2
          row_limit = row_to_r2p_idx(row_limit_top, num_rep, num_wg)
          mask_r2p_lambda(
            acc_S,
            lambda s: r2p_bitmask_above(row_limit, s),
            rank1=True,
          )
      else:
        if const_expr(self.window_size_right is not None):
          row_limit_top = causal_offset - self.window_size_right
        else:
          row_limit_top = 0
        if const_expr(self.window_size_left is not None):
          row_limit_bot = causal_offset + self.window_size_left
        if const_expr(mask_seqlen):
          if seqlenk_col_limit <= 0:
            row_limit_top = self.tile_m
        r2p = True
        if const_expr(not r2p):
          for i in cutlass.range(cute.size(acc_S.shape), unroll_full=True):
            row_idx = t0ScS_t2r[i][ROW]
            local_mask = row_idx < row_limit_top
            if const_expr(self.window_size_left is not None):
              local_mask |= row_idx > row_limit_bot
            acc_S[i] = -cutlass.Float32.inf if local_mask else acc_S[i]
        else:

          def mask_gen_fn(s: int) -> Uint32:
            num_rep = cute.size(tScS_t2r, mode=[0])
            num_wg = 2

            row_limit = row_to_r2p_idx(row_limit_top, num_rep, num_wg)
            mask = r2p_bitmask_above(row_limit, s)

            if const_expr(self.window_size_left is not None):
              row_limit_bottom = row_to_r2p_idx(
                row_limit_bot + 1, num_rep, num_wg
              )
              mask = mask & r2p_bitmask_below(row_limit_bottom, s)

            return mask

          mask_r2p_lambda(
            acc_S,
            mask_gen_fn,
            rank1=True,
          )


# SM100 (Blackwell) free helpers and the fused-mask trip-count bridge.  Sm100FusedMask is a scheduling object, not a masking one: it maps a work tile to the KV trip interval and the leading/trailing masked-block counts so every role in a warp-specialised kernel derives one shared physical trip count.
@cute.jit
def call_mask_mod(
  mask_mod: cutlass.Constexpr,
  batch_idx,
  head_idx,
  q_idx,
  kv_idx,
  seqlen_info,
  aux_data: AuxData,
):
  # Compatibility shim for pre-aux_scalars mask_mod callables.
  if const_expr(aux_data.scalars is not None):
    return mask_mod(
      batch_idx,
      head_idx,
      q_idx,
      kv_idx,
      seqlen_info,
      aux_data.tensors,
      aux_data.scalars,
    )
  return mask_mod(
    batch_idx,
    head_idx,
    q_idx,
    kv_idx,
    seqlen_info,
    aux_data.tensors,
  )


@cute.jit
def row_to_r2p_idx(x: Int32, num_rep: int, num_wg: int) -> Int32:
  """Row coordinate -> R2P element index in the warp-group interleaved layout, clamping unowned rows to the boundary index (safe because R2P thresholds are monotonic)."""
  return x // (num_rep *
               num_wg) * num_rep + min(x % (num_rep * num_wg), num_rep)


class Sm100MaskEnum(enum.Enum):
  """Mask types for FMHA operations; the ``*_INFERENCE`` variants additionally require the end of q to align with the end of k."""

  NO_MASK = enum.auto()
  RESIDUAL_MASK = enum.auto()
  CAUSAL_MASK = enum.auto()
  WINDOW_MASK = enum.auto()
  WINDOW_MASK_INFERENCE = enum.auto()
  # Deprecated the following types
  WINDOW_MASK_BWD = enum.auto()
  WINDOW_MASK_BWD_INFERENCE = enum.auto()
  RESIDUAL_MASK_BWD = enum.auto()


class Sm100FusedMask:
  """No mask, residual mask for variable sequence lengths, and causal mask, applied to attention scores.  The donor's other half -- the trip-count and trip-bound scheduling calculators -- was withdrawn as unreachable."""

  # The donor class carries nine more methods (get_trip_count, get_trip_start, get_trip_start_count_via_block_info, get_trip_mask_bounds_via_block_info, get_unmasked_trip_count, get_masked_leading_count, get_masked_trailing_count, get_leading_mask_id, get_trailing_mask_id), all withdrawn: the D512 kernels enter at apply_mask_via_causal_local (dQ) and apply_mask_via_causal_local_r2p (forward) and drive their own trip counts from BlockInfo.
  # The two *_via_block_info entries are bypassed on purpose: they fabricate a zero-offset SeqlenInfoQK around two scalar lengths, standing up a second seqlen authority beside the one the kernel already carries.
  # apply_mask below has no SM100 D512 caller either, but the SM90 backward kernels call mask.apply_mask(...) on a local whose type no static rule here can pin down, and AttentionMask has a method of the same name -- unresolvable receiver, so assume live.

  @cute.jit
  def apply_mask(
    mask_type: Sm100MaskEnum,
    acc_qk: cute.Tensor,
    index_qk: cute.Tensor,
    seqlen_q: Int32,
    seqlen_k: Int32,
    window_size_left: Optional[int] = None,
    window_size_right: Optional[int] = None,
    index_transform: cutlass.Constexpr = lambda index_q, index_k: (
      index_q,
      index_k,
    ),
  ):
    """Modify ``acc_qk`` in place according to ``mask_type`` and the positions in ``index_qk``."""
    offset = 0
    # Causal here end-aligns Q with K when seqlen_k != seqlen_q, as the reference does -- keep iff k_index <= q_index + (seqlen_k - seqlen_q) + window_right -- and is spelled (window_left is None, window_right is not None).
    if cutlass.const_expr(
      window_size_left is None and window_size_right is not None
    ):
      offset = seqlen_k - seqlen_q
    elif cutlass.const_expr(
      mask_type is Sm100MaskEnum.WINDOW_MASK_INFERENCE
      or mask_type is Sm100MaskEnum.WINDOW_MASK_BWD_INFERENCE
    ):
      offset = seqlen_k - seqlen_q
    for i in cutlass.range_constexpr(cute.size(acc_qk), unroll_full=True):
      index_q, index_k = index_transform(*index_qk[i])
      if cutlass.const_expr(
        window_size_left is not None or window_size_right is not None
      ):
        if cutlass.const_expr(window_size_left is None):
          if index_q + offset + window_size_right < index_k:
            acc_qk[i] = -Float32.inf
          if index_k >= seqlen_k or index_q >= seqlen_q:  # residual mask
            acc_qk[i] = -Float32.inf
        elif cutlass.const_expr(window_size_right is None):
          if index_q + offset - window_size_left > index_k:
            acc_qk[i] = -Float32.inf
          if index_k >= seqlen_k or index_q >= seqlen_q:  # residual mask
            acc_qk[i] = -Float32.inf
        else:
          max_K_index = dsl_min(index_q + offset + window_size_right, seqlen_k)
          min_K_index = max(0, index_q + offset - window_size_left)
          if index_k > max_K_index or index_k < min_K_index:
            acc_qk[i] = -Float32.inf
          if index_k >= seqlen_k or index_q >= seqlen_q:  # residual mask
            acc_qk[i] = -Float32.inf

      if cutlass.const_expr(
        mask_type == Sm100MaskEnum.RESIDUAL_MASK
        or mask_type == Sm100MaskEnum.RESIDUAL_MASK_BWD
      ):
        if index_k >= seqlen_k or index_q >= seqlen_q:
          acc_qk[i] = -Float32.inf

  @cute.jit
  def apply_mask_via_causal_local(
    acc_qk: cute.Tensor,
    index_qk: cute.Tensor,
    seqlen_q: Int32,
    seqlen_k: Int32,
    apply_semantic_window: cutlass.Constexpr[bool] = True,
    is_causal: cutlass.Constexpr[bool] = False,
    is_local: cutlass.Constexpr[bool] = False,
    window_size_left: Optional[int] = None,
    window_size_right: Optional[int] = None,
    index_transform: cutlass.Constexpr = lambda index_q, index_k: (
      index_q,
      index_k,
    ),
  ):
    """Forward mask without a ``mask_type``: causal/local window constraints when ``apply_semantic_window``, residual OOB masking always."""
    offset = 0
    if cutlass.const_expr(apply_semantic_window):
      # Match WINDOW_MASK_INFERENCE semantics: end-align Q/K when lengths differ.
      offset = seqlen_k - seqlen_q
    for i in cutlass.range_constexpr(cute.size(acc_qk), unroll_full=True):
      index_q, index_k = index_transform(*index_qk[i])
      if cutlass.const_expr(apply_semantic_window):
        if cutlass.const_expr(is_causal and not is_local):
          # Pure causal; tolerate both external forms, (None, None) from the interface and (None, 0) from fused-mask-style callers.
          right = 0 if const_expr(
            window_size_right is None
          ) else window_size_right
          if index_q + offset + right < index_k:
            acc_qk[i] = -Float32.inf
        elif cutlass.const_expr(
          is_local or window_size_left is not None
          or window_size_right is not None
        ):
          if cutlass.const_expr(window_size_left is None):
            if index_q + offset + window_size_right < index_k:
              acc_qk[i] = -Float32.inf
          elif cutlass.const_expr(window_size_right is None):
            if index_q + offset - window_size_left > index_k:
              acc_qk[i] = -Float32.inf
          else:
            max_K_index = dsl_min(
              index_q + offset + window_size_right, seqlen_k
            )
            min_K_index = max(0, index_q + offset - window_size_left)
            if index_k > max_K_index or index_k < min_K_index:
              acc_qk[i] = -Float32.inf
      # Residual mask is always needed for boundary protection.
      if index_k >= seqlen_k or index_q >= seqlen_q:
        acc_qk[i] = -Float32.inf

  @cute.jit
  def apply_mask_via_causal_local_r2p(
    acc_qk: cute.Tensor,
    index_qk: cute.Tensor,
    seqlen_q: Int32,
    seqlen_k: Int32,
    apply_semantic_window: cutlass.Constexpr[bool] = True,
    is_causal: cutlass.Constexpr[bool] = False,
    window_size_right: Optional[int] = None,
  ) -> None:
    """`apply_mask_via_causal_local` as one R2P thermometer bitmap, bitwise identical output.

        PRECONDITION, and the whole reason the form is legal: every element of `acc_qk` belongs to ONE query row and element `i` is key column `index_qk[0][1] + i`, which the SM100 D512 forward's M128 lane fold guarantees.  A fragment spanning several rows, or in the WGMMA `0,1,8,9,...` column order, must use `apply_mask_via_causal_local` or convert the limit with `sm90_col_to_r2p_idx` first.
        """
    row_idx = index_qk[0][0]
    col_base = index_qk[0][1]
    limit = seqlen_k
    if const_expr(apply_semantic_window):
      # One upper bound only; a left window would need `r2p_bitmask_above` as a second, lower bound.
      assert is_causal, "the R2P form here implements the causal upper bound only"
      # Bottom-right alignment, matching `apply_mask_via_causal_local`: keep iff index_k <= index_q + (seqlen_k - seqlen_q) + right.
      right = 0 if const_expr(window_size_right is None) else window_size_right
      limit = dsl_min(row_idx + (seqlen_k - seqlen_q) + right + 1, limit)
    # Thread-uniform, so it is one select for the whole fragment.
    limit = Int32(0) if row_idx >= seqlen_q else limit
    # r2p_bitmask_below counts in element space from column `col_base`; out-of-range limits are safe through the UB-safe shift in `shr_u32`, <= 0 masking everything and >= 64 keeping everything.
    limit_elem = limit - col_base
    mask_r2p_lambda(
      acc_qk, lambda s: r2p_bitmask_below(limit_elem, s), rank1=True
    )

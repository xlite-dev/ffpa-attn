# Copyright (c) 2025, Tri Dao.
# Adapted from https://github.com/Dao-AILab/flash-attention,
# with reference to the d256 specialized implementation

# Shared by the SM100 HD512 2-CTA kernels (check_tmem_intervals: all four;
# tmem_offset: dQ only; the trace channel: forward only); every name here
# encodes an HD512 assumption, so HD512-independent hardware facts live in
# blackwell_helpers.py instead.  Everything above the trace-channel banner at
# the end is donor code that regenerates from the donor (see setup.cfg on the
# vendored copies); the channel below it does not.

import cutlass
import cutlass.cute as cute
from cutlass import const_expr
from cutlass.cute.typing import Int32

from .blackwell_helpers import SM100_TMEM_CAPACITY_COLUMNS


def tmem_offset(lane, col):
  # (lane, column) packed into one 32-bit word; also the spelling for a TMEM stride.
  return (lane << 16) + col


def check_tmem_intervals(intervals):
  # Reject a half-open column ledger that overlaps or leaves the addressable range; the adjacent-pair overlap test is sound only because emptiness is rejected first.
  for name, (start, stop) in intervals.items():
    if stop <= start:
      raise AssertionError(f"TMEM region {name} is empty: [{start}, {stop})")
    if start < 0 or stop > SM100_TMEM_CAPACITY_COLUMNS:
      raise AssertionError(
        f"TMEM region {name} spans [{start}, {stop}), outside "
        f"[0, {SM100_TMEM_CAPACITY_COLUMNS})"
      )
  ordered = sorted(intervals.items(), key=lambda item: item[1][0])
  for (a_name, (_, a_stop)), (b_name, (b_start,
                                       _)) in zip(ordered, ordered[1:]):
    if a_stop > b_start:
      raise AssertionError(
        f"TMEM regions {a_name} and {b_name} overlap at column {b_start}"
      )
  return intervals


# One warp group's share of a partitioned tensor's last mode: the INTERLEAVED set {w, w + num_wg, ...}, not the contiguous-block split of the same name elsewhere -- the epilogue and dS-publication geometry were derived against it, so swapping bodies repartitions TMEM under the same call sites.
@cute.jit
def split_wg(
  t: cute.Tensor,
  num_warp_groups: Int32,
  wg_idx: Int32,
) -> cute.Tensor:
  ret = None
  if const_expr(cute.rank(t.layout) == 3):
    # The T2R fragment's (copy, rest_m, rest_n) face.
    p = cute.composition(
      t,
      cute.make_layout((
        t.shape[0],
        t.shape[1],
        (num_warp_groups, cute.size(t, mode=[2]) // num_warp_groups),
      )),
    )
    ret = p[None, None, (wg_idx, None)]
  else:
    # Rank 4 adds the slice mode.
    p = cute.composition(
      t,
      cute.make_layout((
        t.shape[0],
        t.shape[1],
        t.shape[2],
        (num_warp_groups, cute.size(t, mode=[3]) // num_warp_groups),
      )),
    )
    ret = p[None, None, None, (wg_idx, None)]
  return ret


# (shape, stride) addressing the SMEM field for one 2-CTA M128 fragment from a (lane, warp_row) x (register, warp_group, block, warp_column) coordinate; the warp column continues the run instead of stepping a 64-column plane when the per-cell run is short, and using the plane stride there addresses past the field.
def ds_publisher_map(registers, block_loops):
  run_before_warp_col = registers * 2 * block_loops
  warp_col_stride = run_before_warp_col if run_before_warp_col < 64 else 64 * 64
  shape = ((32, 2), (registers, 2, block_loops, 2))
  stride = ((64, 32 * 64), (1, registers, registers * 2, warp_col_stride))
  return shape, stride


@cute.jit
def reg_to_smem_mma128x128_2cta(
  regs: cute.Tensor,
  smem: cute.Tensor,
  index: Int32,
  tiler_mn: tuple[Int32, Int32],
  dp_idx: Int32,
  wg_idx: Int32,
):
  # Publish one 2-CTA M128 accumulator fragment into a staged SMEM field: dS^T into the dS ring for dK, P^T into sP for dV.
  smem_slice = smem[None, None, None, index]
  thread_layout = cute.make_ordered_layout(
    # (tileN, tileM)
    tiler_mn,
    (0, 1),
  )
  smem_slice_tmp = cute.composition(smem_slice, thread_layout)

  warp_idx = dp_idx // 32
  warp_row_idx = warp_idx % 2
  warp_col_idx = warp_idx // 2  # the second 64 columns of the field
  lane_idx = dp_idx % 32
  # ((16,1),1,2) at a 128-wide M, ((16,1),1,1) at a 64-wide M, ((8,1),1,2) at the dV kernel's 64x64 tiler.
  reg_shape = regs.shape
  block_loops = reg_shape[2]
  registers = cute.size(reg_shape[0])

  tmp_shape, tmp_stride = ds_publisher_map(registers, block_loops)
  # `make_tensor(iterator, layout)` never consults `smem_slice`'s layout, so nothing downstream clamps a map that addresses past the field; refuse at lowering instead.
  addressed = 1 + sum((extent - 1) * step
                      for extent, step in zip((*tmp_shape[0], *tmp_shape[1]),
                                              (*tmp_stride[0], *tmp_stride[1])))
  available = cute.cosize(smem_slice.layout)
  if const_expr(addressed > available):
    raise ValueError(
      f"publication layout addresses {addressed} elements of a "
      f"{available}-element field (tiler_mn={tiler_mn})"
    )
  smem_copy = cute.make_tensor(
    smem_slice_tmp.iterator, cute.make_layout(tmp_shape, stride=tmp_stride)
  )

  for ib in cutlass.range(block_loops):
    regs_copy = regs[(None, 0), 0, ib]
    smem_copy_slice = smem_copy[(lane_idx, warp_row_idx),
                                (None, wg_idx, ib, warp_col_idx)]
    cute.autovec_copy(regs_copy, smem_copy_slice)


def accumulator_edge_witness(tiled_mma, mma_tiler_mn):
  # One MMA edge's observed accumulator geometry as JSON-safe fields, read off the fragment so a gate can compare it against the declared TARGET_* constants.
  thr = tiled_mma.get_slice(0)
  acc_shape = thr.partition_shape_C(mma_tiler_mn)
  frag = thr.make_fragment_C(cute.append(acc_shape, 1))
  sl = frag[(None, None), 0, 0, 0]
  group = int(cute.size(tiled_mma.thr_id.shape))
  return {
    "cta_group":
    group,
    "thr_id_shape":
    str(tiled_mma.thr_id.shape),
    "per_cta_slices": [
      str(
        tiled_mma.get_slice(rank).make_fragment_C(cute.append(acc_shape,
                                                              1))[(None, None),
                                                                  0, 0,
                                                                  0].layout
      ) for rank in range(group)
    ],
    "mma_tiler_mn":
    list(mma_tiler_mn),
    "partition_shape_c":
    str(acc_shape),
    "accumulator_shape":
    str(sl.shape),
    "accumulator_layout":
    str(sl.layout),
    "rows_per_cta":
    int(cute.size(sl, mode=[0])),
    "n_extent":
    int(cute.size(sl, mode=[1])),
    # A two-CTA MMA divides N across the pair, so this is what the declared slice stride has to cover.
    "columns_per_cta":
    int(cute.size(sl, mode=[1])) // group,
    "n_mode_is_hierarchical":
    isinstance(sl.shape[1], tuple),
  }


# Note [Trace channel]: non-donor IKET helpers.
class IketTraceChannel:
  """Base of a kernel class: the emitters of Note [Trace channel].

    Plain Python, not a `cute` construct -- each method is traced inline into
    whichever `@cute.jit`/`@cute.kernel` body calls it, so the guard is the
    calling kernel's own compile-time constant."""

  # IKET selector, a compile-time constant under `const_expr` (release builds
  # omit every mark and range); set on the KERNEL class -- not on this base --
  # before its first JIT, and a compile-key term there.
  iket_stamps: bool = False

  def _stamp(self, name: str, payload=None):
    """Emit one instant trace event, or nothing: under `const_expr` a release
        build carries no instruction.  Payload = item ordinal."""
    if const_expr(self.iket_stamps):
      if const_expr(payload is None):
        cute.experimental.iket.mark(name)
      else:
        cute.experimental.iket.mark(name, payload)

  def _range_push(self, name: str, payload=None):
    """Open a nested trace range on this warp's stack, or nothing.  Pairs with
        `_range_pop`; see Note [Trace channel] for the balance rule."""
    if const_expr(self.iket_stamps):
      if const_expr(payload is None):
        cute.experimental.iket.range_push(name)
      else:
        cute.experimental.iket.range_push(name, payload)

  def _range_pop(self):
    """Close the innermost `_range_push`, or nothing."""
    if const_expr(self.iket_stamps):
      cute.experimental.iket.range_pop()

  def _range_start(self, name: str):
    """Open a range that gets a track of its own instead of stacking; returns
        the token `_range_end` consumes, or `None` in a release build."""
    if const_expr(self.iket_stamps):
      return cute.experimental.iket.range_start(name)
    return None

  def _range_end(self, token):
    """Close a `_range_start`, or nothing."""
    if const_expr(self.iket_stamps):
      cute.experimental.iket.range_end(token)

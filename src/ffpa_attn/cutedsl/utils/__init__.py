# This file is copied from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/utils.py
# Copyright (c) 2025, Tri Dao.
# SM90-only trimmed version of flash_attn/cute/utils.py
#
# Removed (not used by SM90 fwd/bwd pipeline):
#   - import quack.activation  (only used by ex2_emulation_2, which is SM100+)
#   - POLY_EX2 dict            (ex2 polynomial coefficients, SM100+ only)
#   - LOG2_E module constant   (each SM90 kernel computes math.log2(math.e) locally)
#   - _fa_clc_enabled / _fa_disable_2cta_enabled / _fa_disable_2cta_cuda12
#   - _is_cuda_12() / _get_use_clc_scheduler_default() / _get_disable_2cta_default()
#   - _compute_base_hash       (kept as internal impl of hash_callable, not removed)
#   - convert_from_dlpack      (SM90 only uses convert_from_dlpack_leading_static)
#   - smid()                   (SM90 pipeline doesn't use it)
#   - domain_offset_aligned()  (SM90 pipeline doesn't use it)
#   - evaluate_polynomial / evaluate_polynomial_2
#   - add_round_down / combine_int_frac_ex2
#   - ex2_emulation / ex2_emulation_2 / e2e_asm2
#
# Simplified:
#   - fmax: removed 3-input (c) parameter (SM100+ only)
#   - fmax_reduce: removed arch >= 100 branch (3-input fmax)
#   - fadd_reduce: removed arch >= 100 branch (packed f32x2 add)
#   - get_smem_store_atom: removed arch < 90 branch

import math
import hashlib
import inspect
from typing import Type, Callable, Optional, Tuple, overload

import cutlass
import cutlass.cute as cute

from cutlass import Float32, Int32, const_expr
from cutlass.cute import FastDivmodDivisor
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import nvvm, llvm
from cutlass.cute.runtime import from_dlpack

# ---------------------------------------------------------------------------
# Callable hashing  (used by interface_sm90.py for compile keys)
# ---------------------------------------------------------------------------
_MIXER_ATTRS = ("__vec_size__", )


def _compute_base_hash(func: Callable) -> str:
  """Compute hash from source code or bytecode and closure values."""
  try:
    data = inspect.getsource(func).encode()
  except (OSError, TypeError):
    if hasattr(func, "__code__") and func.__code__ is not None:
      data = func.__code__.co_code
    else:
      data = repr(func).encode()

  hasher = hashlib.sha256(data)

  if hasattr(func, "__closure__") and func.__closure__ is not None:
    for cell in func.__closure__:
      hasher.update(repr(cell.cell_contents).encode())

  return hasher.hexdigest()


def hash_callable(
  func: Callable,
  mixer_attrs: Tuple[str] = _MIXER_ATTRS,
  set_cute_hash: bool = True
) -> str:
  """Hash a callable based on the source code or bytecode and closure values.
    Fast-path: if the callable (or its __wrapped__ base) has a ``__cute_hash__``
    attribute, that value is returned immediately as the base hash, then
    metadata dunders are mixed in to produce the final dict-key hash.
    set_cute_hash: whether or not to set func.__cute_hash__
    """
  # Resolve base hash
  if hasattr(func, "__cute_hash__"):
    base_hash = func.__cute_hash__
  else:
    # Unwrap decorated functions (e.g., cute.jit wrappers).
    base_func = getattr(func, "__wrapped__", func)

    if hasattr(base_func, "__cute_hash__"):
      base_hash = base_func.__cute_hash__
    else:
      base_hash = _compute_base_hash(base_func)

      if set_cute_hash:
        base_func.__cute_hash__ = base_hash

  # Mix in mutable metadata dunders
  mixer_values = tuple(getattr(func, attr, None) for attr in mixer_attrs)

  if all(v is None for v in mixer_values):
    return base_hash

  hasher = hashlib.sha256(base_hash.encode())

  for attr, val in zip(_MIXER_ATTRS, mixer_values):
    hasher.update(f"{attr}={val!r}".encode())

  return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Softcap score_mod helpers  (used by interface_sm90.py)
# ---------------------------------------------------------------------------
def create_softcap_scoremod(softcap_val):

  @cute.jit
  def scoremod_premask_fn(
    acc_S_SSA, batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors
  ):
    scores = acc_S_SSA / softcap_val
    return softcap_val * cute.math.tanh(scores, fastmath=True)

  return scoremod_premask_fn


def create_softcap_scoremod_bwd(softcap_val):

  @cute.jit
  def scoremod_bwd_fn(
    grad_out_SSA, score_SSA, batch_idx, head_idx, q_idx, kv_idx, seqlen_info,
    aux_tensors
  ):
    scores = score_SSA / softcap_val
    tanh_scores = cute.math.tanh(scores, fastmath=True)
    return grad_out_SSA * (1.0 - tanh_scores * tanh_scores)

  return scoremod_bwd_fn


# ---------------------------------------------------------------------------
# Softmax scale helpers  (used by fwd/bwd kernels)
# ---------------------------------------------------------------------------
def compute_softmax_scale_log2(softmax_scale, score_mod):
  """Compute softmax_scale_log2 and adjusted softmax_scale based on whether score_mod is used.

    When score_mod is None, fold the log2(e) factor into softmax_scale_log2 and set softmax_scale
    to None. When score_mod is present, keep softmax_scale separate so it can be applied before
    the score_mod, and set softmax_scale_log2 to just the change-of-base constant.

    Returns (softmax_scale_log2, softmax_scale).
    """
  _LOG2_E = math.log2(math.e)
  if const_expr(score_mod is None):
    return softmax_scale * _LOG2_E, None
  else:
    return _LOG2_E, softmax_scale


def compute_fastdiv_mods(
  mQ, mK, qhead_per_kvhead, pack_gqa, aux_tensors, mPageTable=None
):
  """Compute FastDivmodDivisor pairs for aux_tensors index computation.

    Returns a (seqlen_q_divmod, seqlen_k_divmod) tuple, or None if aux_tensors is None.
    """
  if const_expr(aux_tensors is None):
    return None
  seqlen_q = cute.size(mQ.shape[0]
                       ) // (qhead_per_kvhead if const_expr(pack_gqa) else 1)
  seqlen_k = cute.size(mK.shape[0]) if const_expr(
    mPageTable is None
  ) else mK.shape[0] * mPageTable.shape[1]
  return (FastDivmodDivisor(seqlen_q), FastDivmodDivisor(seqlen_k))


# ---------------------------------------------------------------------------
# DLPack conversion  (used by interface_sm90.py for semaphore tensors)
# ---------------------------------------------------------------------------
def convert_from_dlpack_leading_static(
  x,
  leading_dim,
  alignment=16,
  static_modes=None,
  stride_order=None
) -> cute.Tensor:
  if stride_order is None:
    stride_order = x.dim_order()
  x_ = from_dlpack(x, assumed_align=alignment)
  for i in range(x.ndim):
    if i != leading_dim and (static_modes is None or i not in static_modes):
      x_ = x_.mark_compact_shape_dynamic(mode=i, stride_order=stride_order)
  return x_


# ---------------------------------------------------------------------------
# Tiled copy / MMA fragment helpers  (used by flash_fwd.py / flash_bwd.py base classes)
# ---------------------------------------------------------------------------
def make_tiled_copy_A(
  copy_atom: cute.CopyAtom,
  tiled_mma: cute.TiledMma,
  swapAB: cutlass.Constexpr[bool] = False
) -> cute.TiledCopy:
  if const_expr(swapAB):
    return cute.make_tiled_copy_B(copy_atom, tiled_mma)
  else:
    return cute.make_tiled_copy_A(copy_atom, tiled_mma)


def make_tiled_copy_B(
  copy_atom: cute.CopyAtom,
  tiled_mma: cute.TiledMma,
  swapAB: cutlass.Constexpr[bool] = False
) -> cute.TiledCopy:
  if const_expr(swapAB):
    return cute.make_tiled_copy_A(copy_atom, tiled_mma)
  else:
    return cute.make_tiled_copy_B(copy_atom, tiled_mma)


def mma_make_fragment_A(
  smem: cute.Tensor,
  thr_mma: cute.core.ThrMma,
  swapAB: cutlass.Constexpr[bool] = False
) -> cute.Tensor:
  if const_expr(swapAB):
    return mma_make_fragment_B(smem, thr_mma)
  else:
    return thr_mma.make_fragment_A(thr_mma.partition_A(smem))


def mma_make_fragment_B(
  smem: cute.Tensor,
  thr_mma: cute.core.ThrMma,
  swapAB: cutlass.Constexpr[bool] = False
) -> cute.Tensor:
  if const_expr(swapAB):
    return mma_make_fragment_A(smem, thr_mma)
  else:
    return thr_mma.make_fragment_B(thr_mma.partition_B(smem))


# ---------------------------------------------------------------------------
# SMEM store atom  (SM90-only: always use StMatrix for 16-bit types)
# ---------------------------------------------------------------------------
def get_smem_store_atom(
  arch: cutlass.Constexpr[int],
  element_type: Type[cute.Numeric],
  transpose: bool = False
) -> cute.CopyAtom:
  # SM90 with 16-bit element types always uses StMatrix.
  # The arch < 90 branch (CopyUniversalOp) is removed for SM90-only builds.
  # Signature kept compatible: arch parameter is accepted but ignored.
  if const_expr(arch < 90 or element_type.width != 16):
    # Fallback for non-16-bit types (e.g., fp32 accumulators) — still needed
    return cute.make_copy_atom(
      cute.nvgpu.CopyUniversalOp(),
      element_type,
      num_bits_per_copy=2 * element_type.width,
    )
  else:
    return cute.make_copy_atom(
      cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=transpose, num_matrices=4),
      element_type,
    )


# ---------------------------------------------------------------------------
# Warp-level reductions  (used by softmax.py, flash_bwd_preprocess.py)
# ---------------------------------------------------------------------------
@cute.jit
def warp_reduce(
  val: cute.TensorSSA | cute.Numeric,
  op: Callable,
  width: cutlass.Constexpr[int] = cute.arch.WARP_SIZE,
) -> cute.TensorSSA | cute.Numeric:
  if const_expr(isinstance(val, cute.TensorSSA)):
    res = cute.make_fragment(val.shape, val.dtype)
    res.store(val)
    for i in cutlass.range_constexpr(cute.size(val.shape)):
      res[i] = warp_reduce(res[i], op, width)
    return res.load()
  else:
    for i in cutlass.range_constexpr(int(math.log2(width))):
      val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
  return val


# ---------------------------------------------------------------------------
# fmax / fmax_reduce / fadd_reduce  (SM90-only: arch < 100 paths only)
# ---------------------------------------------------------------------------
@dsl_user_op
def fmax(
  a: float | Float32, b: float | Float32, *, loc=None, ip=None
) -> Float32:
  """2-input fmax for SM90. The 3-input (c) variant is SM100+ only and removed."""
  from cutlass import CUDA_VERSION

  if CUDA_VERSION.major == 12 and CUDA_VERSION.minor == 9:
    return Float32(
      nvvm.fmax(
        T.f32(),
        Float32(a).ir_value(loc=loc, ip=ip),
        Float32(b).ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
      )
    )
  else:
    return Float32(
      nvvm.fmax(
        Float32(a).ir_value(loc=loc, ip=ip),
        Float32(b).ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
      )
    )


@cute.jit
def fmax_reduce(
  x: cute.TensorSSA,
  init_val: float | Float32 | None = None,
  arch: cutlass.Constexpr[int] = 80
) -> Float32:
  """Row-max reduction. SM90 only uses the arch < 100 path (4-accumulator loop)."""
  res = cute.make_fragment(x.shape, Float32)
  res.store(x)
  local_max = [res[0], res[1], res[2], res[3]]
  for i in cutlass.range_constexpr(4, cute.size(x.shape), 4):
    local_max[0] = fmax(local_max[0], res[i + 0])
    local_max[1] = fmax(local_max[1], res[i + 1])
    local_max[2] = fmax(local_max[2], res[i + 2])
    local_max[3] = fmax(local_max[3], res[i + 3])
  local_max[0] = fmax(local_max[0], local_max[1])
  local_max[2] = fmax(local_max[2], local_max[3])
  local_max[0] = fmax(local_max[0], local_max[2])
  return local_max[0] if const_expr(init_val is None
                                    ) else fmax(local_max[0], init_val)


@cute.jit
def fadd_reduce(
  x: cute.TensorSSA,
  init_val: float | Float32 | None = None,
  arch: cutlass.Constexpr[int] = 80
) -> Float32:
  """Row-sum reduction. SM90 only uses the arch < 100 path (cute reduce ADD)."""
  if const_expr(init_val is None):
    init_val = Float32.zero
  return x.reduce(cute.ReductionOp.ADD, init_val, 0)


# ---------------------------------------------------------------------------
# Atomic add  (used by flash_bwd.py for dQ accumulation)
# ---------------------------------------------------------------------------
@dsl_user_op
def atomic_add_fp32(
  a: float | Float32, gmem_ptr: cute.Pointer, *, loc=None, ip=None
) -> None:
  nvvm.atomicrmw(
    res=T.f32(),
    op=nvvm.AtomicOpKind.FADD,
    ptr=gmem_ptr.llvm_ptr,
    a=Float32(a).ir_value()
  )


@dsl_user_op
def elem_pointer(
  x: cute.Tensor, coord: cute.Coord, *, loc=None, ip=None
) -> cute.Pointer:
  return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)


# ---------------------------------------------------------------------------
# Predicate helpers  (used by flash_fwd.py, flash_bwd.py, flash_bwd_postprocess.py)
# ---------------------------------------------------------------------------
@cute.jit
def predicate_k(tAcA: cute.Tensor, limit: cutlass.Int32) -> cute.Tensor:
  # Only compute predicates for the "k" dimension. For the mn dimension, we will use "if"
  tApA = cute.make_fragment(
    cute.make_layout(
      (
        cute.size(tAcA, mode=[0, 1]), cute.size(tAcA, mode=[1]),
        cute.size(tAcA, mode=[2])
      ),
      stride=(cute.size(tAcA, mode=[2]), 0, 1),
    ),
    cutlass.Boolean,
  )
  for rest_v in cutlass.range_constexpr(tApA.shape[0]):
    for rest_k in cutlass.range_constexpr(tApA.shape[2]):
      tApA[rest_v, 0,
           rest_k] = cute.elem_less(tAcA[(0, rest_v), 0, rest_k][1], limit)
  return tApA


# ---------------------------------------------------------------------------
# Warp group index  (used by flash_fwd_sm90.py)
# ---------------------------------------------------------------------------
def canonical_warp_group_idx(sync: bool = True) -> cutlass.Int32:
  warp_group_idx = cute.arch.thread_idx()[0] // 128
  if const_expr(sync):
    warp_group_idx = cute.arch.make_warp_uniform(warp_group_idx)
  return warp_group_idx


# ---------------------------------------------------------------------------
# Shuffle sync  (used by flash_bwd_sm90.py, mask.py, pack_gqa.py)
# ---------------------------------------------------------------------------
@cute.jit
def shuffle_sync(
  value: cute.Numeric,
  offset: cute.typing.Int,
  width: cutlass.Constexpr[int] = cute.arch.WARP_SIZE,
) -> cute.Numeric:
  assert value.width % 32 == 0, "value type must be a multiple of 32 bits"
  # 1 -> 0b11111, 2 -> 0b11110, 4 -> 0b11100, 8 -> 0b11000, 16 -> 0b10000, 32 -> 0b00000
  mask = cute.arch.WARP_SIZE - width
  clamp = cute.arch.WARP_SIZE - 1
  mask_and_clamp = mask << 8 | clamp
  # important: need stride 1 and not 0 for recast_tensor to work
  val = cute.make_rmem_tensor(
    cute.make_layout((1, ), stride=(1, )), type(value)
  )
  val[0] = value
  val_i32 = cute.recast_tensor(val, cutlass.Int32)
  for i in cutlass.range_constexpr(cute.size(val_i32)):
    val_i32[i] = cute.arch.shuffle_sync(
      val_i32[i], offset, mask_and_clamp=mask_and_clamp
    )
  return val[0]


# ---------------------------------------------------------------------------
# Bit-shift ops  (used by mask.py for R2P bitmask generation)
# ---------------------------------------------------------------------------
@dsl_user_op
def shl_u32(
  val: cutlass.Uint32,
  shift: cutlass.Uint32,
  *,
  loc=None,
  ip=None
) -> cutlass.Uint32:
  """
    Left-shift val by shift bits using PTX shl.b32 (sign-agnostic).

    Uses inline PTX to avoid shift-by-type-width UB in LLVM IR.
    See original docstring in utils.py for full explanation.
    """
  return cutlass.Uint32(
    llvm.inline_asm(
      T.i32(),
      [
        cutlass.Uint32(val).ir_value(loc=loc, ip=ip),
        cutlass.Uint32(shift).ir_value(loc=loc, ip=ip),
      ],
      "shl.b32 $0, $1, $2;",
      "=r,r,r",
      has_side_effects=False,
      is_align_stack=False,
      asm_dialect=llvm.AsmDialect.AD_ATT,
    )
  )


@dsl_user_op
def shr_u32(
  val: cutlass.Uint32,
  shift: cutlass.Uint32,
  *,
  loc=None,
  ip=None
) -> cutlass.Uint32:
  """
    Unsigned right-shift val by shift bits using PTX shr.u32 (zero-fills).

    Uses inline PTX to avoid shift-by-type-width UB in LLVM IR.
    """
  return cutlass.Uint32(
    llvm.inline_asm(
      T.i32(),
      [
        cutlass.Uint32(val).ir_value(loc=loc, ip=ip),
        cutlass.Uint32(shift).ir_value(loc=loc, ip=ip),
      ],
      "shr.u32 $0, $1, $2;",
      "=r,r,r",
      has_side_effects=False,
      is_align_stack=False,
      asm_dialect=llvm.AsmDialect.AD_ATT,
    )
  )


@cute.jit
def clz(x: Int32) -> Int32:
  """Count leading zeros of a 32-bit integer."""
  # Early exit is not supported yet
  res = Int32(32)
  done = False
  for i in cutlass.range(32):
    if ((1 << (31 - i)) & x) and not done:
      res = Int32(i)
      done = True
  return res


# ---------------------------------------------------------------------------
# Warp prefix sum  (used by tile_scheduler.py)
# ---------------------------------------------------------------------------
@cute.jit
def warp_prefix_sum(
  val: cutlass.Int32, lane: Optional[cutlass.Int32] = None
) -> cutlass.Int32:
  if const_expr(lane is None):
    lane = cute.arch.lane_idx()
  for i in cutlass.range_constexpr(int(math.log2(cute.arch.WARP_SIZE))):
    offset = 1 << i
    # Very important that we set mask_and_clamp to 0
    partial_sum = cute.arch.shuffle_sync_up(
      val, offset=offset, mask_and_clamp=0
    )
    if lane >= offset:
      val += partial_sum
  return val


# ---------------------------------------------------------------------------
# f32 → f16/bf16 conversion  (used by flash_fwd_sm90.py, flash_bwd_sm90.py)
# ---------------------------------------------------------------------------
@dsl_user_op
def cvt_f16x2_f32(
  a: float | Float32,
  b: float | Float32,
  to_dtype: Type,
  *,
  loc=None,
  ip=None
) -> cutlass.Int32:
  assert to_dtype in [
    cutlass.BFloat16, cutlass.Float16
  ], "to_dtype must be BFloat16 or Float16"
  return cutlass.Int32(
    llvm.inline_asm(
      T.i32(),
      [
        Float32(a).ir_value(loc=loc, ip=ip),
        Float32(b).ir_value(loc=loc, ip=ip)
      ],
      f"cvt.rn.{'bf16x2' if to_dtype is cutlass.BFloat16 else 'f16x2'}.f32 $0, $2, $1;",
      "=r,f,f",
      has_side_effects=False,
      is_align_stack=False,
      asm_dialect=llvm.AsmDialect.AD_ATT,
    )
  )


@overload
def cvt_f16(src: cute.Tensor, dst: cute.Tensor) -> None:
  ...


@overload
def cvt_f16(src: cute.Tensor, dtype: Type[cute.Numeric]) -> cute.Tensor:
  ...


@cute.jit
def cvt_f16(src: cute.Tensor, dst_or_dtype):
  """Convert Float32 tensor to Float16/BFloat16.

    Args:
        src: Source tensor with Float32 element type
        dst_or_dtype: Either a destination tensor or a dtype (Float16/BFloat16)

    Returns:
        None if dst is a tensor, or a new tensor if dtype is provided
    """
  if const_expr(isinstance(dst_or_dtype, type)):
    # dtype variant: create new tensor and call the tensor variant
    dtype = dst_or_dtype
    dst = cute.make_fragment(src.shape, dtype)
    cvt_f16(src, dst)
    return dst
  else:
    # tensor variant: write to dst
    dst = dst_or_dtype
    assert cute.size(dst.shape) == cute.size(
      src.shape
    ), "dst and src must have the same size"
    assert cute.size(
      src.shape
    ) % 2 == 0, "src must have an even number of elements"
    assert dst.element_type in [
      cutlass.BFloat16, cutlass.Float16
    ], "dst must be BFloat16 or Float16"
    assert src.element_type is Float32, "src must be Float32"
    dst_i32 = cute.recast_tensor(dst, cutlass.Int32)
    assert cute.size(dst_i32.shape) * 2 == cute.size(src.shape)
    for i in cutlass.range_constexpr(cute.size(dst_i32)):
      dst_i32[i] = cvt_f16x2_f32(src[2 * i], src[2 * i + 1], dst.element_type)


# ---------------------------------------------------------------------------
# SSA ↔ scalar helpers  (used by softmax.py, mask.py for score_mod/mask_mod)
# ---------------------------------------------------------------------------
@cute.jit
def scalar_to_ssa(a: cute.Numeric, dtype) -> cute.TensorSSA:
  """Convert a scalar to a cute TensorSSA of shape (1,) and given dtype"""
  vec = cute.make_fragment(1, dtype)
  vec[0] = a
  return vec.load()


def ssa_to_scalar(val):
  """Could inline but nice for reflecting the above api"""
  return val[0]

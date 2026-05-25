"""Shared utilities for FFPA cutedsl SplitD forward and backward paths.

Constants, validation helpers, tensor utilities, and optional-int encoding used
by both SM90 and SM80/SM89 CuTeDSL paths (and also imported by
:mod:`cutedsl.__init__` for the torch custom op entry points).
"""

import os
from functools import lru_cache
from typing import Optional, Tuple, Callable

import torch
from torch._guards import active_fake_mode
import tvm_ffi

import cutlass
import cutlass.utils as utils_basic
from cutlass.base_dsl import BaseDSL
from cutlass.base_dsl.arch import Arch

MIN_SUPPORTED_HEAD_DIM = 320

# SM90 ENVIRONMENT VARIABLES
SM90_SUPPORTED_HEAD_DIM = 512
SM90_FWD_TILE_M = 64
SM90_FWD_TILE_N = 128
SM90_BWD_TILE_M = 64
SM90_BWD_TILE_N = 64

# SM80/SM89 ENVIRONMENT VARIABLES
SM80_SUPPORTED_HEAD_DIM = 1024

SM80_FWD_TILE_M = 64
SM80_FWD_TILE_N = 128
SM80_FWD_NUM_STAGES = 2
SM80_FWD_NUM_THREADS = 128
SM80_FWD_SPLIT_D_CHUNK = 32

SM80_BWD_DKDV_TILE_M = 64
SM80_BWD_DKDV_TILE_N = 64
# DKDV kernel supports multi-stage prefetch over the d_chunk dim.
# Each ring uses an independent commit_group; per-ring wait counts let
# asymmetric (ns_Q, ns_dO) be correct.
SM80_BWD_DKDV_NUM_STAGES_Q = 1
SM80_BWD_DKDV_NUM_STAGES_DO = 2
SM80_BWD_DKDV_NUM_THREADS = 128

SM80_BWD_DQ_TILE_M = 64
SM80_BWD_DQ_TILE_N = 64
# DQ kernel multi-stage prefetch (per-ring commit groups, per-ring wait counts).
SM80_BWD_DQ_NUM_STAGES_Q = 1
SM80_BWD_DQ_NUM_STAGES_DO = 1
SM80_BWD_DQ_NUM_THREADS = 128

SM80_BWD_SPLIT_D_CHUNK = 64

# Wider Split-D candidates probed by ``_pick_split_d_chunk`` before falling back
# to the per-kernel default. Larger chunks shorten the d-loop and amortise epilogue
# cost but proportionally grow shared-memory usage; ``can_implement`` decides per
# arch whether each candidate fits the SMEM_CAPACITY_MAP entry for that target.
_SPLIT_D_CHUNK_CANDIDATES = (256, 128, 64)

# Wider chunks are only worthwhile on archs with abundant SMEM (A100 ~164KB,
# Hopper / server Blackwell ~228KB). On sm_89/sm_86 and sm_120/sm_121 the
# per-SM SMEM is ~100KB, where a wider chunk eats into occupancy and measured
# slower than the conservative default in microbenchmarks (D=640 on L20).
_WIDE_SPLIT_D_MIN_SMEM_BYTES = 160 * 1024

_VARLEN_CUSTOM_OP_NONE_INT = -(2**31)

torch2cute_dtype_map = {
  torch.float16: cutlass.Float16,
  torch.bfloat16: cutlass.BFloat16,
}


def is_fake_mode() -> bool:
  return active_fake_mode() is not None


def _parse_arch_str(arch_str):
  """Parse arch string (e.g. 'sm_90a', '90') to int (e.g. 90)."""
  import re

  match = re.match(r"^(?:sm_?|SM_?)?(\d+)(\d)([af]?)$", arch_str)
  if not match:
    raise ValueError(f"Invalid arch format: {arch_str}")
  major, minor, _ = match.groups()
  return int(major) * 10 + int(minor)


@lru_cache(maxsize=None)
def _get_device_arch():
  """Cached device arch check. Override with FLASH_ATTENTION_ARCH env var."""
  arch_override = os.environ.get("FLASH_ATTENTION_ARCH", None)
  if arch_override is not None:
    return _parse_arch_str(arch_override)
  major, minor = torch.cuda.get_device_capability()
  return major * 10 + int(minor)


def _validate_head_dims(head_dim: int, head_dim_v: int) -> None:
  """Validate the SM90 specialised cutedsl Split-D head dimension constraints.

  Used only when the dispatcher routes to the SM90 Hopper specialised
  forward/backward kernels (``major == 9`` and ``head_dim <= 512``);
  every other cutedsl path goes through :func:`_validate_sm80_head_dims`.
  """
  if head_dim != head_dim_v or not (
    MIN_SUPPORTED_HEAD_DIM <= head_dim <= SM90_SUPPORTED_HEAD_DIM
  ):
    raise ValueError(
      f"(head_dim, head_dim_v)=({head_dim}, {head_dim_v}) is not supported. "
      f"This dense SplitD interface requires q/k head_dim == v head_dim_v and "
      f"{MIN_SUPPORTED_HEAD_DIM} <= head_dim <= {SM90_SUPPORTED_HEAD_DIM}."
    )


def _validate_sm80_head_dims(head_dim: int, head_dim_v: int) -> None:
  """Validate the SM80 Ampere Split-D head dimension constraints.

  Used for the SM80 Split-D fallback path, which now covers every
  non-SM90 architecture (SM80/SM89, SM100/SM103/SM120, ...) and any
  ``head_dim > 512`` on SM90. Requires symmetric q/k/v head_dim in
  ``[MIN_SUPPORTED_HEAD_DIM, SM80_SUPPORTED_HEAD_DIM]`` and
  ``head_dim % SM80_FWD_SPLIT_D_CHUNK == 0``.
  """
  if head_dim != head_dim_v or not (
    MIN_SUPPORTED_HEAD_DIM <= head_dim <= SM80_SUPPORTED_HEAD_DIM
  ) or head_dim % SM80_FWD_SPLIT_D_CHUNK != 0:
    raise ValueError(
      f"(head_dim, head_dim_v)=({head_dim}, {head_dim_v}) is not supported. "
      f"The SM80/SM89 Split-D interface requires q/k head_dim == v "
      f"head_dim_v, {MIN_SUPPORTED_HEAD_DIM} <= head_dim <= "
      f"{SM80_SUPPORTED_HEAD_DIM}, and head_dim % {SM80_FWD_SPLIT_D_CHUNK} == 0."
    )


def maybe_contiguous(x):
  return x.contiguous() if x is not None and not x.is_contiguous() else x


def _call_with_tvm_ffi_current_stream(fn, *args, device: torch.device):
  """Run a TVM FFI launch on PyTorch's current CUDA stream for the tensor device."""
  if is_fake_mode() or device.type != "cuda":
    return fn(*args)
  stream = torch.cuda.current_stream(device=device)
  with tvm_ffi.use_torch_stream(torch.cuda.stream(stream)):
    return fn(*args)


def _encode_optional_int_for_custom_op(value: Optional[int]) -> int:
  return _VARLEN_CUSTOM_OP_NONE_INT if value is None else int(value)


def _decode_optional_int_from_custom_op(value: int) -> Optional[int]:
  return None if value == _VARLEN_CUSTOM_OP_NONE_INT else value


def _decode_custom_op_window(
  window_size_left: int,
  window_size_right: int,
) -> Tuple[Optional[int], Optional[int]]:
  return (
    _decode_optional_int_from_custom_op(window_size_left),
    _decode_optional_int_from_custom_op(window_size_right),
  )


def _validate_tensor(
  tensor, name, expected_shape, expected_dtype, expected_device
):
  if tensor is None:
    raise ValueError(f"{name} must not be None")
  if tensor.shape != expected_shape:
    raise ValueError(
      f"{name} has shape {tensor.shape}, expected {expected_shape}"
    )
  if tensor.dtype != expected_dtype:
    raise TypeError(
      f"{name} has dtype {tensor.dtype}, expected {expected_dtype}"
    )
  if tensor.device != expected_device:
    raise RuntimeError(
      f"{name} is on {tensor.device}, expected {expected_device}"
    )


def _cute_arch_cache_key(arch: Arch) -> str:
  return getattr(arch, "name", str(arch))


def _validate_sm90_arch() -> tuple[int, str]:
  arch = _get_device_arch()
  if arch // 10 != 9:
    raise RuntimeError(
      f"This SM90-only SplitD interface requires Hopper (SM 9.x), got compute capability {arch}. "
      "Falls back to the generic FFPA dispatcher for other architectures."
    )
  cute_arch = BaseDSL._get_dsl().get_arch_enum()
  if cute_arch < Arch.sm_90a:
    raise RuntimeError(
      "This SplitD D=512 path emits Hopper SM90a instructions such as WGMMA, TMA, "
      f"and setmaxnreg. CuTeDSL selected {cute_arch}, but Arch.sm_90a or newer is required."
    )
  return arch, _cute_arch_cache_key(cute_arch)


def _validate_sm80_arch() -> tuple[int, str]:
  """Validate that the active CuTeDSL target is SM80 or newer.

  The Split-D path emits Ampere-class warp MMA + cp.async, which is
  forward-compatible on SM89/SM90/SM120. The dispatcher routes Hopper-only
  D<=512 to the specialised SM90 kernel; everything else (including D>512
  on Hopper and all kernels on Blackwell) lands here.
  """
  arch = _get_device_arch()
  if arch < 80:
    raise RuntimeError(
      f"This Split-D interface requires compute capability >= 8.0, got {arch}."
    )
  cute_arch = BaseDSL._get_dsl().get_arch_enum()
  if cute_arch < Arch.sm_80:
    raise RuntimeError(
      "This Split-D path emits Ampere-class warp-level MMA and cp.async code. "
      f"CuTeDSL selected {cute_arch}, but Arch.sm_80 or newer is required."
    )
  return arch, _cute_arch_cache_key(cute_arch)


def _pick_split_d_chunk(
  can_implement_fn: Callable[..., bool],
  default_chunk: int,
  wide_min_smem_bytes: int = 0,
  **can_implement_kwargs,
) -> int:
  """Pick the largest Split-D chunk that divides ``head_dim`` and fits in SMEM.

  Candidates are probed from widest to narrowest (256 → 128 → 64) before
  falling back to ``default_chunk``. The kernel's own ``can_implement`` is the
  source of truth for whether a given chunk fits the target arch's SMEM
  capacity (see :data:`cutlass.cutlass_dsl.SMEM_CAPACITY_MAP`).

  ``wide_min_smem_bytes`` gates whether wide candidates are tried at all:

  - The forward path passes :data:`_WIDE_SPLIT_D_MIN_SMEM_BYTES` (160KB) so
    only Hopper / server Blackwell (sm_90/sm_100/sm_103/sm_110 with ~228KB)
    upgrade the chunk; SMEM-tight archs (sm_89/sm_86/sm_120/sm_121 ≈ 100KB)
    keep the conservative default that micro-benchmarks favoured.
  - The backward path passes ``0`` to always probe the wider candidates,
    because dK/dV and dQ default to a larger ``d_chunk=64`` whose smem
    footprint is small enough that wider variants often still fit even on
    SMEM-tight archs and just shorten the d-loop.

  :param can_implement_fn: Bound ``can_implement`` of the target kernel class.
  :param default_chunk: Conservative chunk used as the final fallback.
  :param wide_min_smem_bytes: Minimum arch SMEM capacity (bytes) required
      before the wide candidates are considered. ``0`` always probes; set to
      :data:`_WIDE_SPLIT_D_MIN_SMEM_BYTES` to restrict to large-SMEM archs.
  :param can_implement_kwargs: Forwarded to ``can_implement`` for each probe;
      must include ``head_dim`` so chunk divisibility can be checked, and
      ``smem_capacity_arch`` so the wide-chunk gate can be evaluated.
  :returns: The selected chunk width.
  :raises RuntimeError: If no candidate (including ``default_chunk``) fits.
  """
  head_dim = can_implement_kwargs["head_dim"]
  smem_capacity_arch = can_implement_kwargs.get("smem_capacity_arch", "sm_80")
  smem_capacity = utils_basic.get_smem_capacity_in_bytes(smem_capacity_arch)
  wide_candidates = (
    _SPLIT_D_CHUNK_CANDIDATES if smem_capacity >= wide_min_smem_bytes else ()
  )
  seen = set()
  ordered: list[int] = []
  for cand in (*wide_candidates, default_chunk):
    if cand in seen:
      continue
    seen.add(cand)
    if head_dim % cand != 0:
      continue
    ordered.append(cand)
  for cand in ordered:
    if can_implement_fn(d_chunk=cand, **can_implement_kwargs):
      return cand
  raise RuntimeError(
    f"No Split-D chunk in {ordered or list(seen)} fits the kernel resource "
    f"limits for head_dim={head_dim}; default_chunk={default_chunk}."
  )


def _validate_training_dtype(
  q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, requires_grad: bool
) -> None:
  """Validate training dtype constraints shared by SplitD fwd/bwd paths.

  The common q/k/v validator already enforces fp16 or bf16 and matching dtypes.
  Training follows the same dtype support matrix; this helper remains as the
  single call site for any future training-only dtype restriction.
  """
  del k, v, requires_grad
  if q.dtype not in torch2cute_dtype_map:
    raise TypeError(
      "SplitD training inputs must be torch.float16 or torch.bfloat16"
    )


def _validate_cu_seqlens(
  tensor,
  name,
  batch_size: Optional[int] = None,
  total_tokens: Optional[int] = None,
) -> None:
  if tensor is None:
    return
  if tensor.ndim != 1:
    raise ValueError(f"{name} must be a 1D tensor, got rank {tensor.ndim}")
  if tensor.numel() == 0:
    raise ValueError(f"{name} must have at least one element")
  if batch_size is not None and tensor.shape != (batch_size + 1, ):
    raise ValueError(
      f"{name} must have shape ({batch_size + 1},), got {tensor.shape}"
    )
  if tensor.dtype != torch.int32:
    raise TypeError(f"{name} must have dtype torch.int32, got {tensor.dtype}")
  if tensor.stride(0) != 1:
    raise ValueError(f"{name} must be contiguous")
  if is_fake_mode():
    return
  if not tensor.is_cuda:
    raise RuntimeError(f"{name} must be on a CUDA device, got {tensor.device}")

  first = int(tensor[0].item())
  if first != 0:
    raise ValueError(f"{name}[0] must be 0, got {first}")
  if total_tokens is not None:
    last = int(tensor[-1].item())
    if last != total_tokens:
      raise ValueError(
        f"{name}[-1] must equal total tokens ({total_tokens}), got {last}"
      )
  if tensor.numel() > 1 and bool(torch.any(tensor[1:] < tensor[:-1]).item()):
    raise ValueError(f"{name} must be monotonically non-decreasing")


def _validate_max_seqlen_for_cu_seqlens(
  tensor, name, max_seqlen, max_name
) -> None:
  if tensor is None:
    return
  if max_seqlen is None:
    raise ValueError(f"{max_name} must be provided when {name} is provided")
  if isinstance(max_seqlen, bool) or not isinstance(max_seqlen, int):
    raise TypeError(
      f"{max_name} must be an int, got {type(max_seqlen).__name__}"
    )
  if max_seqlen < 0:
    raise ValueError(f"{max_name} must be non-negative, got {max_seqlen}")
  if is_fake_mode():
    return

  lengths = tensor[1:] - tensor[:-1]
  actual_max = int(lengths.max().item()) if lengths.numel() > 0 else 0
  if max_seqlen < actual_max:
    raise ValueError(
      f"{max_name} ({max_seqlen}) must be >= max sequence length from {name} ({actual_max})"
    )


def _ensure_cuda_tensors(*named_tensors) -> None:
  if is_fake_mode():
    return
  for name, tensor in named_tensors:
    if tensor is not None and not tensor.is_cuda:
      raise RuntimeError(
        f"{name} must be on a CUDA device, got {tensor.device}"
      )


def _validate_qkv_common(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  cu_seqlens_q: Optional[torch.Tensor] = None,
  cu_seqlens_k: Optional[torch.Tensor] = None,
  validate_head_dims: Callable[[int, int], None] = _validate_head_dims,
):
  q_rank = 3 if cu_seqlens_q is not None else 4
  kv_rank = 3 if cu_seqlens_k is not None else 4
  if q.ndim != q_rank:
    raise ValueError(f"q must have rank {q_rank}, got rank {q.ndim}")
  if k.ndim != kv_rank:
    raise ValueError(f"k must have rank {kv_rank}, got rank {k.ndim}")
  if v.ndim != kv_rank:
    raise ValueError(f"v must have rank {kv_rank}, got rank {v.ndim}")

  num_head, head_dim = q.shape[-2:]
  seqlen_k = k.shape[-3]
  num_head_kv = k.shape[-2]
  head_dim_v = v.shape[-1]

  if cu_seqlens_q is None:
    batch_size, seqlen_q = q.shape[:2]
    total_q = batch_size * seqlen_q
  else:
    batch_size = cu_seqlens_q.numel() - 1
    seqlen_q = None
    total_q = q.shape[0]
    _validate_cu_seqlens(
      cu_seqlens_q, "cu_seqlens_q", batch_size, total_tokens=total_q
    )

  if k.shape[-1] != head_dim:
    raise ValueError(f"k head_dim is {k.shape[-1]}, expected {head_dim}")
  if v.shape[-3] != seqlen_k or v.shape[-2] != num_head_kv:
    raise ValueError(
      f"v has shape {v.shape}, expected matching seqlen/head dims (*, {seqlen_k}, {num_head_kv}, {head_dim_v})"
    )

  if cu_seqlens_k is None:
    expected_k_shape = (batch_size, seqlen_k, num_head_kv, head_dim)
    expected_v_shape = (batch_size, seqlen_k, num_head_kv, head_dim_v)
  else:
    _validate_cu_seqlens(
      cu_seqlens_k, "cu_seqlens_k", batch_size, total_tokens=seqlen_k
    )
    expected_k_shape = (seqlen_k, num_head_kv, head_dim)
    expected_v_shape = (seqlen_k, num_head_kv, head_dim_v)
  if k.shape != expected_k_shape:
    raise ValueError(f"k has shape {k.shape}, expected {expected_k_shape}")
  if v.shape != expected_v_shape:
    raise ValueError(f"v has shape {v.shape}, expected {expected_v_shape}")
  if q.dtype not in torch2cute_dtype_map:
    raise TypeError("SM90 CuTe inputs must be torch.float16 or torch.bfloat16")
  if q.dtype != k.dtype or q.dtype != v.dtype:
    raise TypeError(
      f"q, k, and v must have the same dtype, got {q.dtype}, {k.dtype}, {v.dtype}"
    )
  _ensure_cuda_tensors(
    ("q", q),
    ("k", k),
    ("v", v),
    ("cu_seqlens_q", cu_seqlens_q),
    ("cu_seqlens_k", cu_seqlens_k),
  )
  if num_head % num_head_kv != 0:
    raise ValueError(
      f"num_head ({num_head}) must be divisible by num_head_kv ({num_head_kv})"
    )
  validate_head_dims(head_dim, head_dim_v)
  return batch_size, seqlen_q, total_q, seqlen_k, num_head, num_head_kv, head_dim, head_dim_v


def _unsupported_training_features(
  requires_grad: bool,
  softcap: Optional[float],
  local: bool,
  score_mod: Optional[Callable],
  mask_mod: Optional[Callable],
  aux_tensors: Optional[list[torch.Tensor]],
):
  if not requires_grad:
    return
  unsupported = []
  if softcap is not None:
    unsupported.append("softcap")
  if local:
    unsupported.append("local/window attention")
  if score_mod is not None:
    unsupported.append("score_mod")
  if mask_mod is not None:
    unsupported.append("mask_mod")
  if aux_tensors is not None:
    unsupported.append("aux_tensors")
  if unsupported:
    raise NotImplementedError(
      "SplitD backward does not support training with " +
      ", ".join(unsupported) + "."
    )


def _resolve_causal_local_window(
  causal, window_size_left, window_size_right, mask_mod=None
):
  local = False
  if window_size_left is not None or window_size_right is not None:
    if causal:
      raise ValueError("causal and window_size are mutually exclusive")
    if mask_mod is not None:
      raise ValueError("mask_mod and window_size are mutually exclusive")
    if window_size_left is not None and window_size_right is not None:
      if window_size_left < 0 and window_size_right < 0:
        causal, local = False, False
        window_size_left, window_size_right = None, None
      elif window_size_right == 0 and window_size_left < 0:
        causal = True
        local = False
        window_size_left, window_size_right = None, None
      else:
        causal, local = False, True
    else:
      causal, local = False, True
  else:
    local = False
  return causal, local, window_size_left, window_size_right


def _validate_varlen_custom_fwd_features(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  causal: bool,
  window_size_left: int,
  window_size_right: int,
  softcap: float,
) -> None:
  window_size_left_opt, window_size_right_opt = _decode_custom_op_window(
    window_size_left, window_size_right
  )
  softcap_opt = None if softcap == 0.0 else softcap
  _, local, _, _ = _resolve_causal_local_window(
    causal, window_size_left_opt, window_size_right_opt
  )
  _unsupported_training_features(
    q.requires_grad or k.requires_grad or v.requires_grad,
    softcap_opt,
    local,
    None,
    None,
    None,
  )


def _validate_varlen_custom_bwd_features(
  causal: bool,
  window_size_left: int,
  window_size_right: int,
  softcap: float,
) -> None:
  if softcap != 0.0:
    raise NotImplementedError("SplitD backward does not support softcap yet")
  window_size_left_opt, window_size_right_opt = _decode_custom_op_window(
    window_size_left, window_size_right
  )
  _, local, _, _ = _resolve_causal_local_window(
    causal, window_size_left_opt, window_size_right_opt
  )
  if local:
    raise NotImplementedError(
      "SplitD backward does not support local/window attention yet"
    )

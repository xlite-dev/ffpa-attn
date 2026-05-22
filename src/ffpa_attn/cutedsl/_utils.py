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
from cutlass.base_dsl import BaseDSL
from cutlass.base_dsl.arch import Arch

MIN_GENERIC_HEAD_DIM = 256
SUPPORTED_HEAD_DIM = 512
SM80_SUPPORTED_HEAD_DIM = 1024
SM80_SPLIT_D_CHUNK = 64
FWD_TILE_M = 64
FWD_TILE_N = 128
BWD_TILE_M = 64
BWD_TILE_N = 64
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
  """Validate dense SM90 SplitD head dimension constraints."""
  if head_dim != head_dim_v or not (
    MIN_GENERIC_HEAD_DIM < head_dim <= SUPPORTED_HEAD_DIM
  ):
    raise ValueError(
      f"(head_dim, head_dim_v)=({head_dim}, {head_dim_v}) is not supported. "
      f"This dense SplitD interface requires q/k head_dim == v head_dim_v and "
      f"{MIN_GENERIC_HEAD_DIM} < head_dim <= {SUPPORTED_HEAD_DIM}."
    )


def _validate_sm80_head_dims(head_dim: int, head_dim_v: int) -> None:
  """Validate dense SM80/SM89 Split-D head dimension constraints."""
  if head_dim != head_dim_v or not (
    MIN_GENERIC_HEAD_DIM < head_dim <= SM80_SUPPORTED_HEAD_DIM
  ) or head_dim % SM80_SPLIT_D_CHUNK != 0:
    raise ValueError(
      f"(head_dim, head_dim_v)=({head_dim}, {head_dim_v}) is not supported. "
      f"The SM80/SM89 Split-D interface requires q/k head_dim == v "
      f"head_dim_v, {MIN_GENERIC_HEAD_DIM} < head_dim <= "
      f"{SM80_SUPPORTED_HEAD_DIM}, and head_dim % {SM80_SPLIT_D_CHUNK} == 0."
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
  """Validate that the active CuTeDSL target is Ampere/Ada SM80-SM89."""
  arch = _get_device_arch()
  if arch // 10 != 8:
    raise RuntimeError(
      f"This SM80/SM89 Split-D interface requires compute capability 8.x, got {arch}."
    )
  cute_arch = BaseDSL._get_dsl().get_arch_enum()
  if cute_arch < Arch.sm_80 or cute_arch >= Arch.sm_90a:
    raise RuntimeError(
      "This Split-D path emits Ampere/Ada warp-level MMA and cp.async code. "
      f"CuTeDSL selected {cute_arch}, but an SM80-SM89 target is required."
    )
  return arch, _cute_arch_cache_key(cute_arch)


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

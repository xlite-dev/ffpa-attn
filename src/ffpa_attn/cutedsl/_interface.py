# FFPA SplitD attention interface for head_dim == 512 on SM90 (Hopper).
# Based on https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/interface.py
# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
#
# This interface is specialized to the D=512 SplitD kernels.
# All non-SplitD code paths (head_dim <= 256) have been removed.
# Training-only build: page_table, learnable_sink, seqused_q/k, block_sparsity removed.

import os
import math
from functools import lru_cache
from typing import Optional, Tuple, Callable

import torch
from torch._guards import active_fake_mode
import tvm_ffi

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32
from cutlass.base_dsl import BaseDSL
from cutlass.base_dsl.arch import Arch
from quack.compile_utils import make_fake_tensor as fake_tensor
from .utils.cache_utils import get_jit_cache

if os.environ.get("CUTE_DSL_PTXAS_PATH", None) is not None:
  from .utils import cute_dsl_ptxas  # noqa: F401

  cute_dsl_ptxas.patch()

from . import utils
from .utils import fa_logging
from .utils.cute_dsl_utils import (
  to_cute_tensor,
  to_cute_aux_tensor,
  get_aux_tensor_metadata,
)
from ._ffpa_fwd_d512_sm90 import FFPAAttnFwdSm90SplitD
from ._ffpa_bwd_preprocess import FFPAAttnBwdPreprocess
from ._ffpa_dkdv_d512_sm90 import FFPAAttnBwdDKDVSm90SplitD
from ._ffpa_dq_d512_sm90 import FFPAAttnBwdDQSm90SplitD

SUPPORTED_HEAD_DIM = 512
FWD_TILE_M = 64
FWD_TILE_N = 128
BWD_TILE_M = 64
BWD_TILE_N = 64
_VARLEN_CUSTOM_OP_NONE_INT = -(2**31)


def is_fake_mode() -> bool:
  return active_fake_mode() is not None


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


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
  """Validate SplitD head dimension constraints: head_dim == head_dim_v == 512."""
  if head_dim != SUPPORTED_HEAD_DIM or head_dim_v != SUPPORTED_HEAD_DIM:
    raise ValueError(
      f"(head_dim, head_dim_v)=({head_dim}, {head_dim_v}) is not supported. "
      f"This SplitD interface requires q/k head_dim == {SUPPORTED_HEAD_DIM} and "
      f"v head_dim_v == {SUPPORTED_HEAD_DIM}, matching the kernel's fixed "
      "8x64 D-slice layout."
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


def _validate_tensor(tensor, name, expected_shape, expected_dtype, expected_device):
  if tensor is None:
    raise ValueError(f"{name} must not be None")
  if tensor.shape != expected_shape:
    raise ValueError(f"{name} has shape {tensor.shape}, expected {expected_shape}")
  if tensor.dtype != expected_dtype:
    raise TypeError(f"{name} has dtype {tensor.dtype}, expected {expected_dtype}")
  if tensor.device != expected_device:
    raise RuntimeError(f"{name} is on {tensor.device}, expected {expected_device}")


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


def _validate_training_dtype(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, requires_grad: bool) -> None:
  if requires_grad and q.dtype != torch.bfloat16:
    raise NotImplementedError(
      "SplitD training currently supports torch.bfloat16 only. "
      f"Got q/k/v dtype {q.dtype}; use bfloat16 inputs or a different attention backend."
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
    raise ValueError(f"{name} must have shape ({batch_size + 1},), got {tensor.shape}")
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
      raise ValueError(f"{name}[-1] must equal total tokens ({total_tokens}), got {last}")
  if tensor.numel() > 1 and bool(torch.any(tensor[1:] < tensor[:-1]).item()):
    raise ValueError(f"{name} must be monotonically non-decreasing")


def _validate_max_seqlen_for_cu_seqlens(tensor, name, max_seqlen, max_name) -> None:
  if tensor is None:
    return
  if max_seqlen is None:
    raise ValueError(f"{max_name} must be provided when {name} is provided")
  if isinstance(max_seqlen, bool) or not isinstance(max_seqlen, int):
    raise TypeError(f"{max_name} must be an int, got {type(max_seqlen).__name__}")
  if max_seqlen < 0:
    raise ValueError(f"{max_name} must be non-negative, got {max_seqlen}")
  if is_fake_mode():
    return

  lengths = tensor[1:] - tensor[:-1]
  actual_max = int(lengths.max().item()) if lengths.numel() > 0 else 0
  if max_seqlen < actual_max:
    raise ValueError(f"{max_name} ({max_seqlen}) must be >= max sequence length from {name} ({actual_max})")


def _ensure_cuda_tensors(*named_tensors) -> None:
  if is_fake_mode():
    return
  for name, tensor in named_tensors:
    if tensor is not None and not tensor.is_cuda:
      raise RuntimeError(f"{name} must be on a CUDA device, got {tensor.device}")


def _validate_qkv_common(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  cu_seqlens_q: Optional[torch.Tensor] = None,
  cu_seqlens_k: Optional[torch.Tensor] = None,
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
    _validate_cu_seqlens(cu_seqlens_q, "cu_seqlens_q", batch_size, total_tokens=total_q)

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
    _validate_cu_seqlens(cu_seqlens_k, "cu_seqlens_k", batch_size, total_tokens=seqlen_k)
    expected_k_shape = (seqlen_k, num_head_kv, head_dim)
    expected_v_shape = (seqlen_k, num_head_kv, head_dim_v)
  if k.shape != expected_k_shape:
    raise ValueError(f"k has shape {k.shape}, expected {expected_k_shape}")
  if v.shape != expected_v_shape:
    raise ValueError(f"v has shape {v.shape}, expected {expected_v_shape}")
  if q.dtype not in torch2cute_dtype_map:
    raise TypeError("SM90 CuTe inputs must be torch.float16 or torch.bfloat16")
  if q.dtype != k.dtype or q.dtype != v.dtype:
    raise TypeError(f"q, k, and v must have the same dtype, got {q.dtype}, {k.dtype}, {v.dtype}")
  _ensure_cuda_tensors(
    ("q", q),
    ("k", k),
    ("v", v),
    ("cu_seqlens_q", cu_seqlens_q),
    ("cu_seqlens_k", cu_seqlens_k),
  )
  if num_head % num_head_kv != 0:
    raise ValueError(f"num_head ({num_head}) must be divisible by num_head_kv ({num_head_kv})")
  _validate_head_dims(head_dim, head_dim_v)
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
    raise NotImplementedError("SplitD backward does not support training with " + ", ".join(unsupported) + ".")


torch2cute_dtype_map = {
  torch.float16: cutlass.Float16,
  torch.bfloat16: cutlass.BFloat16,
}


def _resolve_causal_local_window(causal, window_size_left, window_size_right, mask_mod=None):
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
  window_size_left_opt, window_size_right_opt = _decode_custom_op_window(window_size_left, window_size_right)
  softcap_opt = None if softcap == 0.0 else softcap
  _, local, _, _ = _resolve_causal_local_window(causal, window_size_left_opt, window_size_right_opt)
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
  window_size_left_opt, window_size_right_opt = _decode_custom_op_window(window_size_left, window_size_right)
  _, local, _, _ = _resolve_causal_local_window(causal, window_size_left_opt, window_size_right_opt)
  if local:
    raise NotImplementedError("SplitD backward does not support local/window attention yet")


# ---------------------------------------------------------------------------
# Forward pass — SplitD SM90 (training, head_dim == 512)
# ---------------------------------------------------------------------------


def _ffpa_attn_forward_sm90(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  cu_seqlens_q: Optional[torch.Tensor] = None,
  cu_seqlens_k: Optional[torch.Tensor] = None,
  max_seqlen_q: Optional[int] = None,
  max_seqlen_k: Optional[int] = None,
  softmax_scale: Optional[float] = None,
  causal: bool = False,
  softcap: Optional[float] = None,
  window_size_left: Optional[int] = None,
  window_size_right: Optional[int] = None,
  pack_gqa: Optional[bool] = None,
  score_mod: Optional[Callable] = None,
  mask_mod: Optional[Callable] = None,
  return_lse: bool = False,
  out: Optional[torch.Tensor] = None,
  lse: Optional[torch.Tensor] = None,
  aux_tensors: Optional[list[torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
  """SplitD SM90 forward pass for FFPA attention (head_dim == 512)."""
  q, k, v = [maybe_contiguous(t) for t in (q, k, v)]
  (
    batch_size,
    seqlen_q,
    total_q,
    seqlen_k,
    num_head,
    num_head_kv,
    head_dim,
    head_dim_v,
  ) = _validate_qkv_common(q, k, v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k)

  requires_grad = q.requires_grad or k.requires_grad or v.requires_grad
  _validate_training_dtype(q, k, v, requires_grad)
  _validate_max_seqlen_for_cu_seqlens(cu_seqlens_q, "cu_seqlens_q", max_seqlen_q, "max_seqlen_q")
  _validate_max_seqlen_for_cu_seqlens(cu_seqlens_k, "cu_seqlens_k", max_seqlen_k, "max_seqlen_k")

  device_arch, cute_arch_key = _validate_sm90_arch()
  if softmax_scale is None:
    softmax_scale = 1.0 / math.sqrt(head_dim)
  if softcap == 0.0:
    softcap = None
  qhead_per_kvhead = num_head // num_head_kv
  if pack_gqa is None:
    pack_gqa = qhead_per_kvhead > 1

  device = q.device
  out_torch_dtype = q.dtype
  q_batch_seqlen_shape = (batch_size, seqlen_q) if cu_seqlens_q is None else (total_q, )
  lse_shape = (batch_size, num_head, seqlen_q) if cu_seqlens_q is None else (num_head, total_q)

  if out is None:
    out = torch.empty(*q_batch_seqlen_shape, num_head, head_dim_v, dtype=out_torch_dtype, device=device)
  else:
    _validate_tensor(out, "out", (*q_batch_seqlen_shape, num_head, head_dim_v), out_torch_dtype, device)

  if lse is None:
    lse = torch.empty(lse_shape, dtype=torch.float32, device=device) if requires_grad or return_lse else None
  elif lse is not None:
    _validate_tensor(lse, "lse", lse_shape, torch.float32, device)

  dtype = torch2cute_dtype_map[q.dtype]

  causal, local, window_size_left, window_size_right = _resolve_causal_local_window(
    causal, window_size_left, window_size_right, mask_mod
  )
  _unsupported_training_features(requires_grad, softcap, local, score_mod, mask_mod, aux_tensors)

  current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

  # SplitD tile sizes (hardcoded)
  tile_m = FWD_TILE_M  # tile_m=64 required by num_wg_mma==1 for register headroom
  tile_n = FWD_TILE_N  # tile_n=128 with sO_spill for register pressure management

  # Auto-detect K=V: same data pointer means same tensor
  kv_same = k is v if is_fake_mode() else k.data_ptr() == v.data_ptr()

  if max_seqlen_q is None:
    max_seqlen_q = seqlen_q if cu_seqlens_q is None else total_q
  if max_seqlen_k is None:
    max_seqlen_k = seqlen_k

  if softcap is not None:
    if score_mod is not None:
      raise ValueError("softcap and score_mod cannot be used together")
    score_mod = utils.create_softcap_scoremod(softcap)

  score_mod_hash = utils.hash_callable(score_mod) if score_mod is not None else False
  mask_mod_hash = utils.hash_callable(mask_mod) if mask_mod is not None else False

  is_varlen = cu_seqlens_q is not None or cu_seqlens_k is not None

  if mask_mod is not None and is_varlen:
    raise NotImplementedError("mask_mod with aux_tensors is not yet supported for varlen sequences.")

  if aux_tensors is not None:
    aux_tensor_metadata = get_aux_tensor_metadata(aux_tensors)
  else:
    aux_tensor_metadata = None

  # forward kernel skips those tiles; prefill their mathematical result here.
  if (is_varlen or causal or local) and not is_fake_mode():
    out.zero_()
    if lse is not None:
      lse.fill_(-float("inf"))

  if total_q == 0 or seqlen_k == 0:
    if not is_fake_mode():
      out.zero_()
      if lse is not None:
        lse.fill_(-float("inf"))
    return out, lse

  compile_key = (
    dtype,
    head_dim,
    head_dim_v,
    qhead_per_kvhead,
    causal,
    score_mod_hash,
    mask_mod_hash,
    aux_tensor_metadata,
    lse is None,
    cu_seqlens_q is None,
    cu_seqlens_k is None,
    window_size_left is not None,
    window_size_right is not None,
    tile_m,
    tile_n,
    pack_gqa,
    device_arch,
    cute_arch_key,
    kv_same,
    fa_logging.get_fa_log_level(),
  )
  if compile_key not in _ffpa_attn_forward_sm90.compile_cache:
    cu_seqlens_q_tensor, cu_seqlens_k_tensor = [
      to_cute_tensor(t, assumed_align=4, leading_dim=0) if t is not None else None
      for t in (cu_seqlens_q, cu_seqlens_k)
    ]
    q_tensor, k_tensor, v_tensor, o_tensor = [to_cute_tensor(t) for t in (q, k, v, out)]
    if lse is not None:
      lse_tensor = to_cute_tensor(lse, assumed_align=4)
    else:
      lse_tensor = None

    cute_aux_tensors = None
    if aux_tensors is not None:
      cute_aux_tensors = [to_cute_aux_tensor(buf) for buf in aux_tensors]

    ffpa_fwd = FFPAAttnFwdSm90SplitD(
      dtype,
      head_dim,
      head_dim_v,
      qhead_per_kvhead,
      is_causal=causal,
      is_local=local,
      pack_gqa=pack_gqa,
      tile_m=tile_m,
      tile_n=tile_n,
      kv_same=kv_same,
      mask_mod=mask_mod,
      score_mod=score_mod,
      has_aux_tensors=aux_tensors is not None,
    )

    # Positional args must match FFPAAttnFwdSm90SplitD.__call__ signature:
    # mQ, mK, mV, mO, mLSE, scale, cuseqlens_q, cuseqlens_k, wsl, wsr, aux, stream
    compile_args = [
      ffpa_fwd,
      q_tensor,
      k_tensor,
      v_tensor,
      o_tensor,
      lse_tensor,
      softmax_scale,
      cu_seqlens_q_tensor,
      cu_seqlens_k_tensor,
      window_size_left,
      window_size_right,
      cute_aux_tensors,
      current_stream,
    ]
    _ffpa_attn_forward_sm90.compile_cache[compile_key] = cute.compile(
      *compile_args,
      options=("--enable-tvm-ffi --ptxas-options '--verbose --warn-on-spills --warn-on-local-memory-usage'"),
    )

  if not is_fake_mode():
    q_call, k_call, v_call = q.detach(), k.detach(), v.detach()
    call_args = [
      q_call,
      k_call,
      v_call,
      out.detach(),
      lse,
      softmax_scale,
      cu_seqlens_q,
      cu_seqlens_k,
      window_size_left,
      window_size_right,
      aux_tensors,
    ]
    _call_with_tvm_ffi_current_stream(
      _ffpa_attn_forward_sm90.compile_cache[compile_key],
      *call_args,
      device=device,
    )
  return out, lse


_ffpa_attn_forward_sm90.compile_cache = get_jit_cache("fwd_sm90")

# ---------------------------------------------------------------------------
# Backward helpers
# ---------------------------------------------------------------------------


def _make_fake_bwd_preprocess_tensors(dtype, varlen_q):
  sym = cute.sym_int
  div = 128 // dtype.width  # 8 for fp16/bf16
  b, seqlen_q, h_q, d_v = sym(), sym(), sym(), sym()
  seqlen_q_rounded = sym()
  total_q, total_q_rounded = sym(), sym()
  b_seqlenq = (b, seqlen_q) if not varlen_q else (total_q, )
  mO = fake_tensor(dtype, (*b_seqlenq, h_q, d_v), divisibility=div)
  mdO = fake_tensor(dtype, (*b_seqlenq, h_q, d_v), divisibility=div)
  if not varlen_q:
    mLSE = fake_tensor(Float32, (b, h_q, seqlen_q), divisibility=1)
    mLSElog2 = fake_tensor(Float32, (b, h_q, seqlen_q_rounded), divisibility=4)
    mPdPsum = fake_tensor(Float32, (b, h_q, seqlen_q_rounded), divisibility=4)
  else:
    mLSE = fake_tensor(Float32, (h_q, total_q), divisibility=1)
    mLSElog2 = fake_tensor(Float32, (h_q, total_q_rounded), divisibility=4)
    mPdPsum = fake_tensor(Float32, (h_q, total_q_rounded), divisibility=4)
  return mO, mdO, mLSE, mLSElog2, mPdPsum


def _compile_bwd_preprocess(
  dtype,
  head_dim,
  head_dim_v,
  m_block_size,
  has_cuseqlens_q,
  has_dlse,
  device_arch,
  cute_arch_key,
):
  """Compile bwd preprocess kernel using cute fake tensors."""
  batchp1 = cute.sym_int()
  mO, mdO, mLSE, mLSElog2, mPdPsum = _make_fake_bwd_preprocess_tensors(dtype, varlen_q=has_cuseqlens_q)
  mCuSeqlensQ = fake_tensor(Int32, (batchp1, ), divisibility=1) if has_cuseqlens_q else None
  mdLSE = fake_tensor(Float32, mLSE.shape, divisibility=1) if has_dlse else None
  ffpa_bwd_pre = FFPAAttnBwdPreprocess(dtype, head_dim, head_dim_v, m_block_size)
  return cute.compile(
    ffpa_bwd_pre,
    mO,
    mdO,
    mPdPsum,
    mLSE,
    mLSElog2,
    None,
    mCuSeqlensQ,
    None,
    mdLSE,
    cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    options="--enable-tvm-ffi",
  )


def _bwd_preprocess(
  out,
  dout,
  dpsum,
  lse,
  lse_log2,
  cu_seqlens_q,
  dlse,
  dtype,
  head_dim,
  head_dim_v,
  m_block_size,
  device_arch,
  cute_arch_key,
):
  """Backward preprocess: compute (o * dout).sum(dim=-1) - dLSE, and lse * log2_e."""
  is_varlen = cu_seqlens_q is not None
  compile_key = (
    dtype,
    head_dim,
    head_dim_v,
    m_block_size,
    is_varlen,
    dlse is not None,
    device_arch,
    cute_arch_key,
  )
  if compile_key not in _bwd_preprocess.compile_cache:
    _bwd_preprocess.compile_cache[compile_key] = _compile_bwd_preprocess(*compile_key)
  if not is_fake_mode():
    _call_with_tvm_ffi_current_stream(
      _bwd_preprocess.compile_cache[compile_key],
      out,
      dout,
      dpsum,
      lse,
      lse_log2,
      None,
      cu_seqlens_q,
      None,
      dlse,
      device=out.device,
    )


_bwd_preprocess.compile_cache = get_jit_cache("bwd_pre_sm90")

# ---------------------------------------------------------------------------
# Backward pass — SplitD SM90 (training, head_dim == 512)
# ---------------------------------------------------------------------------


def _ffpa_attn_backward_sm90(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  out: torch.Tensor,
  dout: torch.Tensor,
  lse: torch.Tensor,
  softmax_scale: Optional[float] = None,
  causal: bool = False,
  softcap: float = 0.0,
  window_size_left: Optional[int] = None,
  window_size_right: Optional[int] = None,
  cu_seqlens_q: Optional[torch.Tensor] = None,
  cu_seqlens_k: Optional[torch.Tensor] = None,
  max_seqlen_q: Optional[int] = None,
  max_seqlen_k: Optional[int] = None,
  dq: Optional[torch.Tensor] = None,
  dk: Optional[torch.Tensor] = None,
  dv: Optional[torch.Tensor] = None,
  dlse: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """SplitD SM90 backward pass (training only, head_dim == 512)."""
  device_arch, cute_arch_key = _validate_sm90_arch()

  if softcap != 0.0:
    raise NotImplementedError("SplitD backward does not support softcap yet")

  causal, local, window_size_left, window_size_right = _resolve_causal_local_window(
    causal, window_size_left, window_size_right
  )
  if local:
    raise NotImplementedError("SplitD backward does not support local/window attention yet")

  # SplitD tile sizes (hardcoded)
  m_block_size = BWD_TILE_M
  n_block_size = BWD_TILE_N

  q, k, v, out, dout, lse, cu_seqlens_q, cu_seqlens_k = [
    maybe_contiguous(t) for t in (q, k, v, out, dout, lse, cu_seqlens_q, cu_seqlens_k)
  ]
  (
    batch_size,
    seqlen_q,
    total_q,
    seqlen_k,
    num_head,
    num_head_kv,
    head_dim,
    head_dim_v,
  ) = _validate_qkv_common(q, k, v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k)
  _validate_max_seqlen_for_cu_seqlens(cu_seqlens_q, "cu_seqlens_q", max_seqlen_q, "max_seqlen_q")
  _validate_max_seqlen_for_cu_seqlens(cu_seqlens_k, "cu_seqlens_k", max_seqlen_k, "max_seqlen_k")
  if q.dtype == torch.float16:
    raise NotImplementedError(
      "SplitD backward currently supports bfloat16 only; the fp16 dQ path has a known launch failure."
    )
  if cu_seqlens_q is None:
    seqlen_q_for_rounding = seqlen_q
  else:
    seqlen_q_for_rounding = max_seqlen_q if max_seqlen_q is not None else total_q

  seqlen_q_rounded = (seqlen_q_for_rounding + m_block_size - 1) // m_block_size * m_block_size
  device = q.device
  out_torch_dtype = q.dtype
  if cu_seqlens_q is not None:
    out_shape = (total_q, num_head, head_dim_v)
    lse_shape = (num_head, total_q)
  else:
    out_shape = (batch_size, seqlen_q, num_head, head_dim_v)
    lse_shape = (batch_size, num_head, seqlen_q)
  _validate_tensor(out, "out", out_shape, out_torch_dtype, device)
  _validate_tensor(dout, "dout", out_shape, out_torch_dtype, device)
  _validate_tensor(lse, "lse", lse_shape, torch.float32, device)
  if dlse is not None:
    dlse = maybe_contiguous(dlse)
    _validate_tensor(dlse, "dlse", lse_shape, torch.float32, device)
  if softmax_scale is None:
    softmax_scale = 1.0 / math.sqrt(head_dim)
  qhead_per_kvhead = num_head // num_head_kv

  if dq is None:
    dq = torch.zeros_like(q)
  else:
    _validate_tensor(dq, "dq", q.shape, out_torch_dtype, device)
    if not is_fake_mode():
      dq.zero_()

  if dk is None:
    dk = torch.zeros_like(k)
  else:
    _validate_tensor(dk, "dk", k.shape, out_torch_dtype, device)
    if not is_fake_mode():
      dk.zero_()

  if dv is None:
    dv = torch.zeros_like(v)
  else:
    _validate_tensor(dv, "dv", v.shape, out_torch_dtype, device)
    if not is_fake_mode():
      dv.zero_()

  if total_q == 0 or seqlen_k == 0:
    return dq, dk, dv

  if cu_seqlens_q is None:
    dpsum = torch.empty(batch_size, num_head, seqlen_q_rounded, dtype=torch.float32, device=device)
    lse_log2 = torch.empty(batch_size, num_head, seqlen_q_rounded, dtype=torch.float32, device=device)
  else:
    total_q_rounded_padded = (total_q + cu_seqlens_q.shape[0] * m_block_size - 1) // m_block_size * m_block_size
    dpsum = torch.empty(num_head, total_q_rounded_padded, dtype=torch.float32, device=device)
    lse_log2 = torch.empty(num_head, total_q_rounded_padded, dtype=torch.float32, device=device)

  dtype = torch2cute_dtype_map[q.dtype]
  current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

  # (1) Preprocess dpsum and lse_log2 for the SplitD backward kernels.
  _bwd_preprocess(
    out,
    dout,
    dpsum,
    lse,
    lse_log2,
    cu_seqlens_q,
    dlse,
    dtype,
    head_dim,
    head_dim_v,
    m_block_size,
    device_arch,
    cute_arch_key,
  )

  # (2) Compile and execute SplitD dKdV and dQ kernels
  bwd_key = (
    dtype,
    head_dim,
    head_dim_v,
    causal,
    m_block_size,
    n_block_size,
    cu_seqlens_q is not None,
    cu_seqlens_k is not None,
    qhead_per_kvhead,
    device_arch,
    cute_arch_key,
  )
  if bwd_key not in _ffpa_attn_backward_sm90.compile_cache_dkdv:
    q_t, k_t, v_t, do_t = [to_cute_tensor(t) for t in (q, k, v, dout)]
    dk_t, dv_t = [to_cute_tensor(t) for t in (dk, dv)]
    lse_log2_t = to_cute_tensor(lse_log2, assumed_align=4)
    dpsum_t = to_cute_tensor(dpsum, assumed_align=4)
    cu_seqlens_q_t = (
      to_cute_tensor(cu_seqlens_q, assumed_align=4, leading_dim=0) if cu_seqlens_q is not None else None
    )
    cu_seqlens_k_t = (
      to_cute_tensor(cu_seqlens_k, assumed_align=4, leading_dim=0) if cu_seqlens_k is not None else None
    )

    ffpa_dkdv = FFPAAttnBwdDKDVSm90SplitD(
      dtype,
      head_dim,
      head_dim_v=head_dim_v,
      is_causal=causal,
      qhead_per_kvhead=qhead_per_kvhead,
      tile_m=m_block_size,
      tile_n=n_block_size,
    )
    _ffpa_attn_backward_sm90.compile_cache_dkdv[bwd_key] = cute.compile(
      ffpa_dkdv,
      q_t,
      k_t,
      v_t,
      do_t,
      lse_log2_t,
      dpsum_t,
      dk_t,
      dv_t,
      softmax_scale,
      cu_seqlens_q_t,
      cu_seqlens_k_t,
      current_stream,
      options=("--enable-tvm-ffi --ptxas-options '--verbose --warn-on-spills --warn-on-local-memory-usage'"),
    )

  if bwd_key not in _ffpa_attn_backward_sm90.compile_cache_dq:
    q_t2, k_t2, v_t2, do_t2 = [to_cute_tensor(t) for t in (q, k, v, dout)]
    dq_t = to_cute_tensor(dq)
    lse_log2_t2 = to_cute_tensor(lse_log2, assumed_align=4)
    dpsum_t2 = to_cute_tensor(dpsum, assumed_align=4)
    cu_seqlens_q_t2 = (
      to_cute_tensor(cu_seqlens_q, assumed_align=4, leading_dim=0) if cu_seqlens_q is not None else None
    )
    cu_seqlens_k_t2 = (
      to_cute_tensor(cu_seqlens_k, assumed_align=4, leading_dim=0) if cu_seqlens_k is not None else None
    )

    ffpa_dq = FFPAAttnBwdDQSm90SplitD(
      dtype,
      head_dim,
      head_dim_v=head_dim_v,
      is_causal=causal,
      qhead_per_kvhead=qhead_per_kvhead,
      tile_m=m_block_size,
      tile_n=n_block_size,
    )
    _ffpa_attn_backward_sm90.compile_cache_dq[bwd_key] = cute.compile(
      ffpa_dq,
      q_t2,
      k_t2,
      v_t2,
      do_t2,
      lse_log2_t2,
      dpsum_t2,
      dq_t,
      softmax_scale,
      cu_seqlens_q_t2,
      cu_seqlens_k_t2,
      current_stream,
      options=("--enable-tvm-ffi --ptxas-options '--verbose --warn-on-spills --warn-on-local-memory-usage'"),
    )

  # Execute dKdV and dQ kernels
  if not is_fake_mode():
    _call_with_tvm_ffi_current_stream(
      _ffpa_attn_backward_sm90.compile_cache_dkdv[bwd_key],
      q.detach(),
      k.detach(),
      v.detach(),
      dout,
      lse_log2,
      dpsum,
      dk,
      dv,
      softmax_scale,
      cu_seqlens_q,
      cu_seqlens_k,
      device=device,
    )
    _call_with_tvm_ffi_current_stream(
      _ffpa_attn_backward_sm90.compile_cache_dq[bwd_key],
      q.detach(),
      k.detach(),
      v.detach(),
      dout,
      lse_log2,
      dpsum,
      dq,
      softmax_scale,
      cu_seqlens_q,
      cu_seqlens_k,
      device=device,
    )

  return dq, dk, dv


_ffpa_attn_backward_sm90.compile_cache_dkdv = get_jit_cache("bwd_splitd_dkdv_sm90")
_ffpa_attn_backward_sm90.compile_cache_dq = get_jit_cache("bwd_splitd_dq_sm90")

# ---------------------------------------------------------------------------
# PyTorch custom ops — varlen SplitD SM90
# ---------------------------------------------------------------------------

# These dispatcher ops are intentionally return-oriented and functional from
# PyTorch's point of view: ``mutates_args=()`` only says input arguments are not
# mutated.  The CUTE kernels still write output storage directly, but that
# storage belongs to tensors allocated inside ``_ffpa_attn_forward_sm90`` /
# ``_ffpa_attn_backward_sm90`` and returned to the caller.  Do not add Python
# ``copy_`` here; buffer-oriented legacy APIs should call the lower-level
# wrappers with ``out`` / ``dq`` / ``dk`` / ``dv`` if they need caller-owned
# storage identity.


@torch.library.custom_op("ffpa_attn::splitd_fwd_sm90", mutates_args=())
def _varlen_fwd_custom(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  cu_seqlens_q: torch.Tensor,
  cu_seqlens_k: torch.Tensor,
  max_seqlen_q: int,
  max_seqlen_k: int,
  softmax_scale: float,
  causal: bool,
  window_size_left: int,
  window_size_right: int,
  softcap: float,
  pack_gqa: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
  cu_seqlens_q, cu_seqlens_k = _trim_trailing_empty_varlen_segments(cu_seqlens_q, cu_seqlens_k)
  window_size_left_opt, window_size_right_opt = _decode_custom_op_window(window_size_left, window_size_right)
  return _ffpa_attn_forward_sm90(
    q,
    k,
    v,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_k=cu_seqlens_k,
    max_seqlen_q=max_seqlen_q,
    max_seqlen_k=max_seqlen_k,
    softmax_scale=softmax_scale,
    causal=causal,
    window_size_left=window_size_left_opt,
    window_size_right=window_size_right_opt,
    softcap=softcap,
    pack_gqa=pack_gqa,
    return_lse=True,
  )


@torch.library.register_fake("ffpa_attn::splitd_fwd_sm90")
def _varlen_fwd_fake(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  cu_seqlens_q: torch.Tensor,
  cu_seqlens_k: torch.Tensor,
  max_seqlen_q: int,
  max_seqlen_k: int,
  softmax_scale: float,
  causal: bool,
  window_size_left: int,
  window_size_right: int,
  softcap: float,
  pack_gqa: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
  # Runtime custom op wrappers trim trailing empty cu_seqlens before launching kernels.
  # Fake kernels intentionally do not trim: cu_seqlens values are unavailable here,
  # and output metadata depends only on q/k/v shapes.
  (
    _batch_size,
    _seqlen_q,
    total_q,
    _seqlen_k,
    num_head,
    _num_head_kv,
    _head_dim,
    head_dim_v,
  ) = _validate_qkv_common(q, k, v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k)
  _validate_training_dtype(q, k, v, q.requires_grad or k.requires_grad or v.requires_grad)
  _validate_max_seqlen_for_cu_seqlens(cu_seqlens_q, "cu_seqlens_q", max_seqlen_q, "max_seqlen_q")
  _validate_max_seqlen_for_cu_seqlens(cu_seqlens_k, "cu_seqlens_k", max_seqlen_k, "max_seqlen_k")
  _validate_varlen_custom_fwd_features(q, k, v, causal, window_size_left, window_size_right, softcap)
  out = q.new_empty((total_q, num_head, head_dim_v))
  lse = q.new_empty((num_head, total_q), dtype=torch.float32)
  return out, lse


@torch.library.custom_op("ffpa_attn::splitd_bwd_sm90", mutates_args=())
def _varlen_bwd_custom(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  out: torch.Tensor,
  dout: torch.Tensor,
  lse: torch.Tensor,
  cu_seqlens_q: torch.Tensor,
  cu_seqlens_k: torch.Tensor,
  max_seqlen_q: int,
  max_seqlen_k: int,
  softmax_scale: float,
  causal: bool,
  window_size_left: int,
  window_size_right: int,
  softcap: float,
  dlse: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  cu_seqlens_q, cu_seqlens_k = _trim_trailing_empty_varlen_segments(cu_seqlens_q, cu_seqlens_k)
  window_size_left_opt, window_size_right_opt = _decode_custom_op_window(window_size_left, window_size_right)
  return _ffpa_attn_backward_sm90(
    q,
    k,
    v,
    out,
    dout,
    lse,
    softmax_scale=softmax_scale,
    causal=causal,
    softcap=softcap,
    window_size_left=window_size_left_opt,
    window_size_right=window_size_right_opt,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_k=cu_seqlens_k,
    max_seqlen_q=max_seqlen_q,
    max_seqlen_k=max_seqlen_k,
    dlse=dlse,
  )


@torch.library.register_fake("ffpa_attn::splitd_bwd_sm90")
def _varlen_bwd_fake(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  out: torch.Tensor,
  dout: torch.Tensor,
  lse: torch.Tensor,
  cu_seqlens_q: torch.Tensor,
  cu_seqlens_k: torch.Tensor,
  max_seqlen_q: int,
  max_seqlen_k: int,
  softmax_scale: float,
  causal: bool,
  window_size_left: int,
  window_size_right: int,
  softcap: float,
  dlse: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  # Runtime custom op wrappers trim trailing empty cu_seqlens before launching kernels.
  # Fake kernels intentionally do not trim: cu_seqlens values are unavailable here,
  # and output metadata depends only on q/k/v shapes.
  (
    _batch_size,
    _seqlen_q,
    total_q,
    _seqlen_k,
    num_head,
    _num_head_kv,
    _head_dim,
    head_dim_v,
  ) = _validate_qkv_common(q, k, v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k)
  _validate_max_seqlen_for_cu_seqlens(cu_seqlens_q, "cu_seqlens_q", max_seqlen_q, "max_seqlen_q")
  _validate_max_seqlen_for_cu_seqlens(cu_seqlens_k, "cu_seqlens_k", max_seqlen_k, "max_seqlen_k")
  if q.dtype == torch.float16:
    raise NotImplementedError(
      "SplitD backward currently supports bfloat16 only; the fp16 dQ path has a known launch failure."
    )
  _validate_varlen_custom_bwd_features(causal, window_size_left, window_size_right, softcap)
  device = q.device
  _validate_tensor(out, "out", (total_q, num_head, head_dim_v), q.dtype, device)
  _validate_tensor(dout, "dout", (total_q, num_head, head_dim_v), q.dtype, device)
  _validate_tensor(lse, "lse", (num_head, total_q), torch.float32, device)
  if dlse is not None:
    _validate_tensor(dlse, "dlse", (num_head, total_q), torch.float32, device)
  return torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)


def _varlen_fwd_setup_context(ctx, inputs, output) -> None:
  q, k, v, cu_seqlens_q, cu_seqlens_k = inputs[:5]
  max_seqlen_q, max_seqlen_k, softmax_scale, causal = inputs[5:9]
  window_size_left, window_size_right, softcap = inputs[9:12]
  out, lse = output
  ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k)
  ctx.max_seqlen_q = max_seqlen_q
  ctx.max_seqlen_k = max_seqlen_k
  ctx.softmax_scale = softmax_scale
  ctx.causal = causal
  ctx.window_size_left = window_size_left
  ctx.window_size_right = window_size_right
  ctx.softcap = softcap
  ctx.set_materialize_grads(False)


def _varlen_fwd_backward(ctx, dout, dlse):
  q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
  if dout is None:
    dout = torch.zeros_like(out)
  dq, dk, dv = torch.ops.ffpa_attn.splitd_bwd_sm90(
    q,
    k,
    v,
    out,
    dout,
    lse,
    cu_seqlens_q,
    cu_seqlens_k,
    ctx.max_seqlen_q,
    ctx.max_seqlen_k,
    ctx.softmax_scale,
    ctx.causal,
    ctx.window_size_left,
    ctx.window_size_right,
    ctx.softcap,
    dlse,
  )
  return dq, dk, dv, *((None, ) * 10)


torch.library.register_autograd(
  "ffpa_attn::splitd_fwd_sm90",
  _varlen_fwd_backward,
  setup_context=_varlen_fwd_setup_context,
)


def _normalize_varlen_custom_op_inputs(
  q: torch.Tensor,
  k: torch.Tensor,
  cu_seqlens_q: Optional[torch.Tensor],
  cu_seqlens_k: Optional[torch.Tensor],
  max_seqlen_q: Optional[int],
  max_seqlen_k: Optional[int],
  softmax_scale: Optional[float],
  window_size: Tuple[Optional[int], Optional[int]],
  pack_gqa: Optional[bool],
  score_mod: Optional[Callable],
  aux_tensors: Optional[list],
) -> tuple[torch.Tensor, torch.Tensor, int, int, float, int, int, bool]:
  if cu_seqlens_q is None or cu_seqlens_k is None:
    raise ValueError("ffpa_attn_splitd_varlen_func custom op path requires cu_seqlens_q and cu_seqlens_k")
  if max_seqlen_q is None:
    raise ValueError("max_seqlen_q must be provided when cu_seqlens_q is provided")
  if max_seqlen_k is None:
    raise ValueError("max_seqlen_k must be provided when cu_seqlens_k is provided")
  if score_mod is not None:
    raise NotImplementedError("score_mod is not supported by the SplitD varlen custom op schema")
  if aux_tensors is not None:
    raise NotImplementedError("aux_tensors is not supported by the SplitD varlen custom op schema")
  if not isinstance(window_size, tuple) or len(window_size) != 2:
    raise TypeError("window_size must be a tuple of (left, right)")

  if softmax_scale is None:
    softmax_scale = 1.0 / math.sqrt(q.shape[-1])
  if pack_gqa is None:
    if q.ndim < 3 or k.ndim < 3:
      raise ValueError("q and k must be rank-3 packed varlen tensors")
    pack_gqa = q.shape[-2] > k.shape[-2]

  return (
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    float(softmax_scale),
    _encode_optional_int_for_custom_op(window_size[0]),
    _encode_optional_int_for_custom_op(window_size[1]),
    bool(pack_gqa),
  )


def _trim_trailing_empty_varlen_segments(
  cu_seqlens_q: torch.Tensor,
  cu_seqlens_k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Drop trailing segments where both q and k have zero length.

    A fixed-shape padded cu_seqlens such as ``[..., total, total, ...]`` is
    mathematically equivalent to its trimmed prefix, but the kernel scheduler
    uses ``cu_seqlens.shape[0] - 1`` as the segment count.  Canonicalizing the
    metadata here keeps the public packed-varlen API tolerant of fixed-shape
    padding without making zero-length tail segments visible to the kernels.
    """
  if cu_seqlens_q.numel() != cu_seqlens_k.numel():
    return cu_seqlens_q, cu_seqlens_k
  if cu_seqlens_q.numel() <= 1 or is_fake_mode():
    return cu_seqlens_q, cu_seqlens_k

  q_lengths = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
  k_lengths = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
  active = (q_lengths != 0) | (k_lengths != 0)
  if bool(active.all().item()):
    return cu_seqlens_q, cu_seqlens_k
  if not bool(active.any().item()):
    return cu_seqlens_q[:1], cu_seqlens_k[:1]

  last_active_segment = int(active.nonzero()[-1].item())
  keep_numel = last_active_segment + 2
  if keep_numel == cu_seqlens_q.numel():
    return cu_seqlens_q, cu_seqlens_k
  return cu_seqlens_q[:keep_numel], cu_seqlens_k[:keep_numel]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ffpa_attn_splitd_varlen_func(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  cu_seqlens_q: Optional[torch.Tensor] = None,
  cu_seqlens_k: Optional[torch.Tensor] = None,
  max_seqlen_q: Optional[int] = None,
  max_seqlen_k: Optional[int] = None,
  softmax_scale: Optional[float] = None,
  causal: bool = False,
  window_size: Tuple[Optional[int], Optional[int]] = (None, None),
  softcap: float = 0.0,
  pack_gqa: Optional[bool] = None,
  score_mod: Optional[Callable] = None,
  aux_tensors: Optional[list] = None,
  return_lse: bool = False,
):
  """Varlen SplitD FFPA attention for D=512 on SM90.

    q/k/v must be packed as (total_tokens, heads, 512). Training requires bf16
    q/k/v, valid CUDA int32 cu_seqlens, and explicit max_seqlen_q/k whenever
    the corresponding cu_seqlens tensor is provided. If return_lse=False, LSE is
    still computed internally when needed for backward.
    """
  (
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale,
    window_size_left,
    window_size_right,
    pack_gqa,
  ) = _normalize_varlen_custom_op_inputs(
    q,
    k,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale,
    window_size,
    pack_gqa,
    score_mod,
    aux_tensors,
  )
  out, lse = _varlen_fwd_custom(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size_left,
    window_size_right,
    softcap,
    pack_gqa,
  )
  return (out, lse) if return_lse else out


__all__ = [
  "ffpa_attn_splitd_varlen_func",
]

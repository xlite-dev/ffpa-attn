"""CuTeDSL FFPA backend with SM8x Split-D and SM90 specialised paths.

Exposes the dense and varlen CuTeDSL entry shims used by
:mod:`ffpa_attn.ffpa_attn_interface` and :mod:`ffpa_attn.functional`.

The CuTeDSL kernels in :mod:`ffpa_attn.cute._ffpa_fwd_sm80`,
:mod:`ffpa_attn.cute._ffpa_fwd_sm90`, and their backward launchers operate on the
``[B, N, H, D]`` (or packed ``[T, H, D]``) layout. The SDPA-style
``[B, H, N, D]`` wrappers (:func:`_ffpa_attn_forward_cute`,
:func:`_ffpa_attn_backward_cute`, :func:`_ffpa_attn_varlen_cute`)
handle layout conversion and dispatch through the registered torch ops below.

Dense-path ops ``ffpa_attn::_fwd_cute`` / ``ffpa_attn::_bwd_cute``
and varlen-path ops ``ffpa_attn::_varlen_fwd_cute`` /
``ffpa_attn::_varlen_bwd_cute`` are registered below so both paths
are ``torch.compile``-compatible.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Callable

import torch

from ._utils import (
  MIN_SUPPORTED_HEAD_DIM,
  dense_min_supported_head_dim,
  SM80_FWD_SPLIT_D_CHUNK,
  SM80_SUPPORTED_HEAD_DIM,
  SM90_SUPPORTED_HEAD_DIM,
  _decode_custom_op_window,
  _encode_optional_int_for_custom_op,
  _validate_max_seqlen_for_cu_seqlens,
  _validate_qkv_common,
  _validate_sm80_head_dims,
  _validate_tensor,
  _validate_training_dtype,
  _validate_varlen_custom_bwd_features,
  _validate_varlen_custom_fwd_features,
  is_fake_mode,
)
from ._ffpa_fwd_sm80 import _ffpa_attn_forward_sm80
from ._ffpa_fwd_sm90 import _ffpa_attn_forward_sm90
from ._ffpa_bwd_sm80 import _ffpa_attn_backward_sm80
from ._ffpa_bwd_sm90 import _ffpa_attn_backward_sm90

__all__ = [
  "_ffpa_attn_forward_cute",
  "_ffpa_attn_backward_cute",
  "_ffpa_attn_varlen_cute",
  "_ffpa_attn_varlen_impl",
  "_require_cute_supported",
  "cute_forward_available",
  "cute_backward_available",
  "cute_max_supported_head_dim",
]

# ---------------------------------------------------------------------------
# Entry shims — user-facing validation, layout conversion, and dispatch
# into the registered torch ops.  Formerly lived in _wrappers.py.
# ---------------------------------------------------------------------------


def _check_supported_options(
  *,
  source: str,
  dropout_p: float = 0.0,
  window_size: object = None,
  sink: torch.Tensor | None = None,
  attention_mask: torch.Tensor | None = None,
  block_mask: object | None = None,
  softcap: float | None = None,
  score_mod: object | None = None,
  aux_tensors: list[torch.Tensor] | None = None,
  seqused_k: torch.Tensor | None = None,
  block_table: torch.Tensor | None = None,
  num_splits: int | None = None,
  alibi_slopes: torch.Tensor | None = None,
) -> None:
  """Raise ``NotImplementedError`` for any non-default cutedsl-unsupported option.

  The cutedsl SplitD kernels (``_ffpa_attn_varlen_impl``,
  ``_ffpa_attn_forward_cute``, ``_ffpa_attn_backward_cute``) only
  honor dense / varlen attention with optional causal masking.
  Every other option commonly exposed by attention APIs (mask tensors,
  sliding window, softcap, score_mod, aux tensors, FlashAttention varlen
  extensions, dropout) has no kernel-side implementation and is rejected
  up front so callers see one actionable error rather than a deep kernel
  crash or silent semantic divergence.

  ``source`` is embedded in the error so the caller can tell which
  public-API surface produced the message.
  """
  unsupported: list[str] = []
  if dropout_p not in (None, 0.0):
    unsupported.append("dropout_p")
  if window_size is not None and window_size != (None, None):
    unsupported.append("window_size")
  if sink is not None:
    unsupported.append("sink")
  if attention_mask is not None:
    unsupported.append("attention_mask")
  if block_mask is not None:
    unsupported.append("block_mask")
  if softcap not in (None, 0.0):
    unsupported.append("softcap")
  if score_mod is not None:
    unsupported.append("score_mod")
  if aux_tensors is not None:
    unsupported.append("aux_tensors")
  if seqused_k is not None:
    unsupported.append("seqused_k")
  if block_table is not None:
    unsupported.append("block_table")
  if num_splits is not None:
    unsupported.append("num_splits")
  if alibi_slopes is not None:
    unsupported.append("alibi_slopes")
  if unsupported:
    raise NotImplementedError(
      f"{source} only supports dense/varlen attention with optional "
      f"causal masking; unsupported options: {', '.join(unsupported)}. "
      f"Use forward_backend='triton' when these options are required."
    )


def cute_forward_available(device: Optional[torch.device] = None) -> bool:
  """Return whether the CuTeDSL forward kernel can run on ``device``.

  CuTeDSL forward supports SM90 through the existing Hopper specialised path
  for ``head_dim <= 512`` and falls back to the SM80 Ampere Split-D path for
  every other supported architecture (SM80/SM89, SM100/SM103/SM120, ...) and
  for any ``head_dim > 512``. Other backend constraints (head_dim ceiling,
  dtype, no mask/dropout) are enforced per-call by
  :func:`_require_cute_supported`; this only checks the device-level
  prerequisite so callers can pre-select a backend before allocating tensors.
  """
  if not torch.cuda.is_available():
    return False
  if device is None:
    device = torch.device("cuda", torch.cuda.current_device())
  if device.type != "cuda":
    return False
  major, _ = torch.cuda.get_device_capability(device)
  return major >= 8


def cute_max_supported_head_dim(device: Optional[torch.device] = None) -> int:
  """Return the current CuTeDSL dense head-dim ceiling for ``device``.

  The SM80 Ampere Split-D implementation acts as a cross-architecture
  fallback and supports up to ``SM80_SUPPORTED_HEAD_DIM`` on every
  supported architecture. SM90 keeps the existing specialised path only
  in the ``head_dim <= SM90_SUPPORTED_HEAD_DIM`` range, but the SM80
  fallback covers the rest, so the effective ceiling is the SM80 value
  everywhere. ``device`` is kept for API compatibility but is unused.
  """
  del device
  return SM80_SUPPORTED_HEAD_DIM


def cute_backward_available(device: Optional[torch.device] = None) -> bool:
  """Whether the CuTeDSL backward kernel can run on ``device``.

  Mirrors :func:`cute_forward_available`: SM90 keeps the existing Hopper
  specialised backward for ``head_dim <= 512``; every other supported
  architecture (SM80/SM89, SM100/SM103/SM120, ...) and every
  ``head_dim > 512`` uses the SM80 Ampere Split-D backward as a fallback.
  """
  if not torch.cuda.is_available():
    return False
  if device is None:
    device = torch.device("cuda", torch.cuda.current_device())
  if device.type != "cuda":
    return False
  major, _ = torch.cuda.get_device_capability(device)
  return major >= 8


def _cute_device_major(device: torch.device) -> int:
  if device.type == "cuda" and torch.cuda.is_available():
    major, _ = torch.cuda.get_device_capability(device)
    return major
  return 9


def _use_sm90_specialized(major: int, head_dim: int, head_dim_v: int) -> bool:
  """Return ``True`` when the SM90 specialised cutedsl path should be used.

  Single routing predicate shared by forward/backward dispatch and by the
  fake-mode varlen validator selection. The SM90 specialised kernels only
  cover Hopper with symmetric ``head_dim <= SM90_SUPPORTED_HEAD_DIM``;
  every other case (other archs, or ``head_dim > 512``, or asymmetric q/v
  head_dim) falls back to the SM80 Ampere Split-D path.
  """
  return (
    major == 9 and MIN_SUPPORTED_HEAD_DIM <= head_dim <= SM90_SUPPORTED_HEAD_DIM
    and MIN_SUPPORTED_HEAD_DIM <= head_dim_v <= SM90_SUPPORTED_HEAD_DIM
  )


def _forward_impl_for_device(
  device: torch.device, head_dim: int, head_dim_v: int
):
  major = _cute_device_major(device)
  if major < 8:
    raise NotImplementedError(
      f"cutedsl forward requires compute capability >= 8.0; got {major}.x"
    )
  if _use_sm90_specialized(major, head_dim, head_dim_v):
    return _ffpa_attn_forward_sm90
  return _ffpa_attn_forward_sm80


def _backward_impl_for_device(
  device: torch.device, head_dim: int, head_dim_v: int
):
  major = _cute_device_major(device)
  if major < 8:
    raise NotImplementedError(
      f"cutedsl backward requires compute capability >= 8.0; got {major}.x"
    )
  if _use_sm90_specialized(major, head_dim, head_dim_v):
    return _ffpa_attn_backward_sm90
  return _ffpa_attn_backward_sm80


def _require_cute_supported(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  *,
  requires_grad: bool,
) -> None:
  """Validate tensor-level constraints for the cutedsl backend.

  Checks device, supported CUDA architecture, dense large head_dim, and q/k/v dtype. Kwarg-level
  functional compatibility (``dropout_p``,
  ``attn_mask``, FlashAttention-extension kwargs) is **not** the
  responsibility of this function; that lives in
  :func:`_check_supported_options`, applied by the entry shims
  (:func:`_ffpa_attn_forward_cute`, :func:`_ffpa_attn_varlen_cute`).

  Raises ``NotImplementedError`` / ``RuntimeError`` / ``TypeError`` for
  any tensor-level violation so users who pass ``forward_backend='cutedsl'``
  see an actionable error rather than a deep kernel crash.
  """
  if q.device.type != "cuda":
    raise RuntimeError(
      f"cutedsl backend requires CUDA tensors, got device {q.device}"
    )
  if not torch.cuda.is_available():
    raise RuntimeError(
      "cutedsl backend requires a CUDA-capable build of PyTorch"
    )
  major, _ = torch.cuda.get_device_capability(q.device)
  if major < 8:
    raise NotImplementedError(
      f"cutedsl backend requires compute capability >= 8.0; got {major}.x"
    )
  head_dim_q = q.size(-1)
  head_dim_v = v.size(-1)
  use_sm90 = _use_sm90_specialized(major, head_dim_q, head_dim_v)
  min_head_dim = MIN_SUPPORTED_HEAD_DIM if use_sm90 else dense_min_supported_head_dim(
  )
  max_head_dim = (
    SM90_SUPPORTED_HEAD_DIM if use_sm90 else SM80_SUPPORTED_HEAD_DIM
  )
  if not (min_head_dim <= head_dim_q <= max_head_dim):
    raise NotImplementedError(
      f"cutedsl backend only supports dense head_dim in "
      f"[{min_head_dim}, {max_head_dim}]; got {head_dim_q}"
    )
  if not use_sm90 and head_dim_q % SM80_FWD_SPLIT_D_CHUNK != 0:
    raise NotImplementedError(
      f"cutedsl SM80 Split-D fallback requires head_dim divisible by "
      f"{SM80_FWD_SPLIT_D_CHUNK}; got {head_dim_q}"
    )
  if q.dtype not in (torch.float16, torch.bfloat16):
    raise TypeError(
      f"cutedsl backend requires torch.float16 or torch.bfloat16, got {q.dtype}"
    )
  del requires_grad
  if k.size(-1) != q.size(-1) or v.size(-1) != q.size(-1):
    raise NotImplementedError(
      f"cutedsl backend requires q/k/v to share head_dim={q.size(-1)}; "
      f"got k={k.size(-1)} v={v.size(-1)}"
    )


def _bhnd_to_bnhd(t: torch.Tensor) -> torch.Tensor:
  """Reshape ``[B, H, N, D]`` (SDPA) to the CuTeDSL-native ``[B, N, H, D]`` (FA)."""
  return t.transpose(1, 2).contiguous()


def _bnhd_to_bhnd(t: torch.Tensor) -> torch.Tensor:
  """Reverse of :func:`_bhnd_to_bnhd`: FA ``[B, N, H, D]`` → SDPA ``[B, H, N, D]``."""
  return t.transpose(1, 2).contiguous()


def _ffpa_attn_forward_cute(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  softmax_scale: float,
  causal: bool,
  *,
  return_lse: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
  """CuTeDSL Split-D forward with SDPA-layout in/out.

  Accepts ``[B, H, N, D]`` (SDPA) layout, transposes to the CuTeDSL-native
  ``[B, N, H, D]`` (FA) layout, calls the registered torch op
  ``torch.ops.ffpa_attn._fwd_cute``, and transposes the output back.
  ``lse`` is always in ``[B, H, N]`` shape and does not require a transpose.

  Called from :meth:`_FFPAAttnFunc.forward` when the dispatch selects
  ``CuTeDSLBackend`` — the autograd boundary is owned by
  :class:`ffpa_attn.functional.FFPAAttnFunc`, not by this function.

  :param q: Query tensor ``[B, H_q, N_q, D]``.
  :param k: Key tensor ``[B, H_kv, N_kv, D]``.
  :param v: Value tensor ``[B, H_kv, N_kv, D]``.
  :param softmax_scale: Pre-softmax scaling factor (already resolved, never None).
  :param causal: Whether causal masking is applied.
  :param return_lse: Always ``True`` when called from the training path so lse
      is saved for backward.
  :returns: ``(out, lse)`` where ``out`` is ``[B, H_q, N_q, D]`` and
      ``lse`` is ``[B, H_q, N_q]`` float32.
  """
  requires_grad = any(t.requires_grad for t in (q, k, v))
  _require_cute_supported(q, k, v, requires_grad=requires_grad)

  q_nhd, k_nhd, v_nhd = (_bhnd_to_bnhd(t) for t in (q, k, v))
  out_nhd, lse = torch.ops.ffpa_attn._fwd_cute(
    q_nhd,
    k_nhd,
    v_nhd,
    softmax_scale,
    int(causal),
    int(return_lse),
  )
  out_bhnd = _bnhd_to_bhnd(out_nhd)
  return out_bhnd, lse


def _ffpa_attn_backward_cute(
  grad_out: torch.Tensor,
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  out: torch.Tensor,
  lse: torch.Tensor,
  softmax_scale: float,
  causal: bool,
  *,
  grad_kv_storage_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """CuTeDSL backward with SDPA-layout in/out.

  Accepts all tensors in ``[B, H, N, D]`` (SDPA) layout, transposes to
  ``[B, N, H, D]`` (FA) for the registered torch op
  ``torch.ops.ffpa_attn._bwd_cute``, and transposes the gradient outputs
  back to SDPA layout.

  Called from :meth:`_FFPAAttnFunc.backward` when the dispatch selects
  ``CuTeDSLBackend``.

  :param grad_out: Gradient w.r.t. output ``[B, H_q, N_q, D]``.
  :param q: Query tensor ``[B, H_q, N_q, D]`` (saved from forward).
  :param k: Key tensor ``[B, H_kv, N_kv, D]`` (saved from forward).
  :param v: Value tensor ``[B, H_kv, N_kv, D]`` (saved from forward).
  :param out: Output tensor ``[B, H_q, N_q, D]`` (saved from forward).
  :param lse: Log-sum-exp ``[B, H_q, N_q]`` float32 (saved from forward).
  :param softmax_scale: Pre-softmax scaling factor.
  :param causal: Whether causal masking was applied.
  :param grad_kv_storage_dtype: Optional ``torch.float32`` / ``torch.float16``
      storage dtype for the internal dK/dV HBM buffer; final returned
      gradients are always cast back to ``k.dtype`` / ``v.dtype``. ``None``
      keeps today's behaviour (buffer dtype = activation dtype).
  :returns: ``(dq, dk, dv)`` all in ``[B, H, N, D]`` SDPA layout.
  """
  q_nhd, k_nhd, v_nhd, out_nhd, dout_nhd = (
    _bhnd_to_bnhd(t) for t in (q, k, v, out, grad_out)
  )
  dq_nhd, dk_nhd, dv_nhd = torch.ops.ffpa_attn._bwd_cute(
    dout_nhd,
    q_nhd,
    k_nhd,
    v_nhd,
    out_nhd,
    lse,
    softmax_scale,
    int(causal),
    _grad_kv_storage_dtype_to_code(grad_kv_storage_dtype),
  )
  dq, dk, dv = (_bnhd_to_bhnd(t) for t in (dq_nhd, dk_nhd, dv_nhd))
  return dq, dk, dv


def _ffpa_attn_varlen_cute(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  cu_seqlens_q: torch.Tensor,
  cu_seqlens_k: torch.Tensor | None,
  max_seqlen_q: int,
  max_seqlen_k: int,
  *,
  dropout_p: float,
  softmax_scale: float | None,
  causal: bool,
  enable_gqa: bool,
  return_lse: bool,
  kwargs: dict,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
  """Packed THD cutedsl entry. The varlen path bypasses
  :class:`ffpa_attn.functional.FFPAAttnFunc` and is autograd-registered via
  the ``ffpa_attn::_varlen_fwd_cute`` torch op directly.

  CuTeDSL varlen forward called from
  :func:`ffpa_attn.ffpa_attn_interface.ffpa_attn_varlen_func`. The CuTeDSL
  kernel consumes packed ``[T, H, D]`` layout natively — no transpose, no
  per-sequence loop.
  """
  _check_supported_options(
    source="ffpa_attn_varlen_func",
    dropout_p=dropout_p,
    window_size=kwargs.get("window_size"),
    sink=kwargs.get("sink"),
    attention_mask=kwargs.get("attention_mask", kwargs.get("attn_mask")),
    block_mask=kwargs.get("block_mask"),
    softcap=kwargs.get("softcap"),
    score_mod=kwargs.get("score_mod"),
    aux_tensors=kwargs.get("aux_tensors"),
    seqused_k=kwargs.get("seqused_k"),
    block_table=kwargs.get("block_table"),
    num_splits=kwargs.get("num_splits"),
    alibi_slopes=kwargs.get("alibi_slopes"),
  )

  if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
    raise ValueError(
      f"ffpa_attn_varlen_func: q/k/v must be 3-D packed [T, H, D], "
      f"got ranks q={q.dim()} k={k.dim()} v={v.dim()}"
    )
  if k.shape != v.shape:
    raise ValueError(
      f"ffpa_attn_varlen_func: k/v must share shape, "
      f"got k={tuple(k.shape)} v={tuple(v.shape)}"
    )
  if q.dtype not in (torch.float16, torch.bfloat16):
    raise TypeError(
      f"ffpa_attn_varlen_func: q/k/v must be fp16/bf16, got {q.dtype}"
    )
  if k.dtype != q.dtype or v.dtype != q.dtype:
    raise TypeError(
      f"ffpa_attn_varlen_func: q/k/v must share dtype, got "
      f"q={q.dtype} k={k.dtype} v={v.dtype}"
    )

  if cu_seqlens_k is None:
    cu_seqlens_k = cu_seqlens_q
  if cu_seqlens_q.dtype != torch.int32 or cu_seqlens_k.dtype != torch.int32:
    raise TypeError(
      "ffpa_attn_varlen_func: cu_seqlens_q/cu_seqlens_k must be int32"
    )
  if cu_seqlens_q.numel() != cu_seqlens_k.numel() or cu_seqlens_q.numel() < 2:
    raise ValueError(
      "ffpa_attn_varlen_func: cu_seqlens_q and cu_seqlens_k must share length >= 2"
    )

  if not enable_gqa and q.size(-2) != k.size(-2):
    raise ValueError(
      f"ffpa_attn_varlen_func: enable_gqa=False but query num_heads "
      f"({q.size(-2)}) != key/value num_heads ({k.size(-2)}). "
      f"Set enable_gqa=True or use matching head counts."
    )
  if q.size(-2) % k.size(-2) != 0:
    raise ValueError(
      f"ffpa_attn_varlen_func: query num_heads ({q.size(-2)}) must be an "
      f"integer multiple of key/value num_heads ({k.size(-2)}) for GQA/MQA."
    )

  requires_grad = any(t.requires_grad for t in (q, k, v))
  max_head_dim = cute_max_supported_head_dim(q.device)
  if not (MIN_SUPPORTED_HEAD_DIM <= q.size(-1) <= max_head_dim):
    raise NotImplementedError(
      f"ffpa_attn_varlen_func cutedsl supports head_dim in [{MIN_SUPPORTED_HEAD_DIM}, {max_head_dim}]; "
      f"got {q.size(-1)}"
    )
  _require_cute_supported(q, k, v, requires_grad=requires_grad)

  # _ffpa_attn_varlen_impl is defined below in this module.
  return _ffpa_attn_varlen_impl(
    q,
    k,
    v,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_k=cu_seqlens_k,
    max_seqlen_q=max_seqlen_q,
    max_seqlen_k=max_seqlen_k,
    softmax_scale=softmax_scale,
    causal=causal,
    return_lse=return_lse,
  )


# ---------------------------------------------------------------------------
# Dense torch custom ops — torch.compile-compatible entry for the CuTeDSL
# SplitD D=512 forward/backward.  Follows the ``triton/__init__.py``
# ``torch.library.define`` + ``@impl("CUDA")`` + ``@register_fake``
# pattern.  No ``register_autograd``: backward is managed by
# ``_FFPAAttnFunc`` in ``functional.py``.
# ---------------------------------------------------------------------------

torch.library.define(
  "ffpa_attn::_fwd_cute",
  "(Tensor q, Tensor k, Tensor v, float softmax_scale, int causal, int return_lse) "
  "-> (Tensor o, Tensor lse)",
)


@torch.library.impl("ffpa_attn::_fwd_cute", "CUDA")
def _fwd_cute_torch_op(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  softmax_scale: float,
  causal: int,
  return_lse: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  batch, seqlen_q, num_head, head_dim_v = q.shape
  o = torch.empty(
    batch, seqlen_q, num_head, head_dim_v, dtype=q.dtype, device=q.device
  )
  need_lse = bool(return_lse)
  lse = (
    torch.empty(
      batch, num_head, seqlen_q, dtype=torch.float32, device=q.device
    ) if need_lse else torch.empty(0, device=q.device)
  )
  _forward_impl_for_device(q.device, q.size(-1), v.size(-1))(
    q,
    k,
    v,
    softmax_scale=softmax_scale,
    causal=bool(causal),
    return_lse=need_lse,
    out=o,
    lse=lse if need_lse else None,
  )
  return o, lse


@torch.library.register_fake("ffpa_attn::_fwd_cute")
def _fwd_cute_fake(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  softmax_scale: float,
  causal: int,
  return_lse: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  o = torch.empty_like(q)
  lse = (
    q.new_empty(q.size(0), q.size(-2), q.size(-3), dtype=torch.float32)
    if return_lse else q.new_empty(0)
  )
  return o, lse


torch.library.define(
  "ffpa_attn::_bwd_cute",
  "(Tensor dout, Tensor q, Tensor k, Tensor v, Tensor out, Tensor lse, "
  "float softmax_scale, int causal, int grad_kv_storage_dtype_code) "
  "-> (Tensor dq, Tensor dk, Tensor dv)",
)


def _grad_kv_storage_dtype_to_code(dtype: torch.dtype | None, ) -> int:
  """Encode the dK/dV storage dtype selector for the torch op boundary.

  Matches Triton's encoding: ``0=None`` (use activation dtype),
  ``1=fp32``, ``2=fp16``.
  """
  if dtype is None:
    return 0
  if dtype == torch.float32:
    return 1
  if dtype == torch.float16:
    return 2
  raise ValueError(
    f"Unsupported grad_kv_storage_dtype {dtype!r}; expected None, "
    "torch.float32, or torch.float16."
  )


def _grad_kv_storage_dtype_from_code(code: int) -> torch.dtype | None:
  """Decode the dK/dV storage dtype selector at the torch op boundary."""
  if code == 0:
    return None
  if code == 1:
    return torch.float32
  if code == 2:
    return torch.float16
  raise ValueError(
    f"Unsupported grad_kv_storage_dtype code {code}; expected 0, 1, or 2."
  )


@torch.library.impl("ffpa_attn::_bwd_cute", "CUDA")
def _bwd_cute_torch_op(
  dout: torch.Tensor,
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  out: torch.Tensor,
  lse: torch.Tensor,
  softmax_scale: float,
  causal: int,
  grad_kv_storage_dtype_code: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  storage_dtype = _grad_kv_storage_dtype_from_code(grad_kv_storage_dtype_code)
  backward_impl = _backward_impl_for_device(q.device, q.size(-1), v.size(-1))
  if storage_dtype is not None and backward_impl is not _ffpa_attn_backward_sm80:
    raise NotImplementedError(
      "grad_kv_storage_dtype is only supported by the generic SM80 CuTeDSL "
      "dKdV backward path; the SM90-specialised kernel does not accept a "
      f"dK/dV storage-dtype override (head_dim={q.size(-1)}, "
      f"head_dim_v={v.size(-1)}, device_major="
      f"{torch.cuda.get_device_capability(q.device)[0]})."
    )
  dq = torch.empty_like(q)
  if storage_dtype is None:
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
  else:
    dk = torch.empty_like(k, dtype=storage_dtype)
    dv = torch.empty_like(v, dtype=storage_dtype)
  backward_impl(
    q,
    k,
    v,
    out,
    dout,
    lse,
    softmax_scale=softmax_scale,
    causal=bool(causal),
    dq=dq,
    dk=dk,
    dv=dv,
  )
  if storage_dtype is not None and storage_dtype != k.dtype:
    dk = dk.to(k.dtype)
    dv = dv.to(v.dtype)
  return dq, dk, dv


@torch.library.register_fake("ffpa_attn::_bwd_cute")
def _bwd_cute_fake(
  dout: torch.Tensor,
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  out: torch.Tensor,
  lse: torch.Tensor,
  softmax_scale: float,
  causal: int,
  grad_kv_storage_dtype_code: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


# ---------------------------------------------------------------------------
# Varlen torch custom ops — ``@custom_op`` + ``@register_fake`` +
# ``register_autograd`` pattern.  The varlen path owns its own autograd
# boundary (unlike dense, which delegates to ``_FFPAAttnFunc``).
# ---------------------------------------------------------------------------


def _trim_trailing_empty_varlen_segments(
  cu_seqlens_q: torch.Tensor,
  cu_seqlens_k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Drop trailing segments where both q and k have zero length."""
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


@torch.library.custom_op("ffpa_attn::_varlen_fwd_cute", mutates_args=())
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
  cu_seqlens_q, cu_seqlens_k = _trim_trailing_empty_varlen_segments(
    cu_seqlens_q, cu_seqlens_k
  )
  window_size_left_opt, window_size_right_opt = _decode_custom_op_window(
    window_size_left, window_size_right
  )
  return _forward_impl_for_device(q.device, q.size(-1), v.size(-1))(
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


@torch.library.register_fake("ffpa_attn::_varlen_fwd_cute")
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
  validate_kwargs = {}
  if not _use_sm90_specialized(
    _cute_device_major(q.device), q.size(-1), v.size(-1)
  ):
    validate_kwargs["validate_head_dims"] = _validate_sm80_head_dims
  (
    _batch_size,
    _seqlen_q,
    total_q,
    _seqlen_k,
    num_head,
    _num_head_kv,
    _head_dim,
    head_dim_v,
  ) = _validate_qkv_common(
    q,
    k,
    v,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_k=cu_seqlens_k,
    **validate_kwargs,
  )
  _validate_training_dtype(
    q, k, v, q.requires_grad or k.requires_grad or v.requires_grad
  )
  _validate_max_seqlen_for_cu_seqlens(
    cu_seqlens_q, "cu_seqlens_q", max_seqlen_q, "max_seqlen_q"
  )
  _validate_max_seqlen_for_cu_seqlens(
    cu_seqlens_k, "cu_seqlens_k", max_seqlen_k, "max_seqlen_k"
  )
  _validate_varlen_custom_fwd_features(
    q, k, v, causal, window_size_left, window_size_right, softcap
  )
  out = q.new_empty((total_q, num_head, head_dim_v))
  lse = q.new_empty((num_head, total_q), dtype=torch.float32)
  return out, lse


@torch.library.custom_op("ffpa_attn::_varlen_bwd_cute", mutates_args=())
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
  cu_seqlens_q, cu_seqlens_k = _trim_trailing_empty_varlen_segments(
    cu_seqlens_q, cu_seqlens_k
  )
  window_size_left_opt, window_size_right_opt = _decode_custom_op_window(
    window_size_left, window_size_right
  )
  return _backward_impl_for_device(q.device, q.size(-1), v.size(-1))(
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


@torch.library.register_fake("ffpa_attn::_varlen_bwd_cute")
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
  (
    _batch_size,
    _seqlen_q,
    total_q,
    _seqlen_k,
    num_head,
    _num_head_kv,
    _head_dim,
    head_dim_v,
  ) = _validate_qkv_common(
    q, k, v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k
  )
  _validate_max_seqlen_for_cu_seqlens(
    cu_seqlens_q, "cu_seqlens_q", max_seqlen_q, "max_seqlen_q"
  )
  _validate_max_seqlen_for_cu_seqlens(
    cu_seqlens_k, "cu_seqlens_k", max_seqlen_k, "max_seqlen_k"
  )
  _validate_varlen_custom_bwd_features(
    causal, window_size_left, window_size_right, softcap
  )
  device = q.device
  _validate_tensor(out, "out", (total_q, num_head, head_dim_v), q.dtype, device)
  _validate_tensor(
    dout, "dout", (total_q, num_head, head_dim_v), q.dtype, device
  )
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
  dq, dk, dv = torch.ops.ffpa_attn._varlen_bwd_cute(
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
  "ffpa_attn::_varlen_fwd_cute",
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
    raise ValueError(
      "_ffpa_attn_varlen_impl custom op path requires cu_seqlens_q and cu_seqlens_k"
    )
  if max_seqlen_q is None:
    raise ValueError(
      "max_seqlen_q must be provided when cu_seqlens_q is provided"
    )
  if max_seqlen_k is None:
    raise ValueError(
      "max_seqlen_k must be provided when cu_seqlens_k is provided"
    )
  if score_mod is not None:
    raise NotImplementedError(
      "score_mod is not supported by the SplitD varlen custom op schema"
    )
  if aux_tensors is not None:
    raise NotImplementedError(
      "aux_tensors is not supported by the SplitD varlen custom op schema"
    )
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


def _ffpa_attn_varlen_impl(
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

  q/k/v must be packed as (total_tokens, heads, 512). Training supports fp16
  and bf16 q/k/v, valid CUDA int32 cu_seqlens, and explicit max_seqlen_q/k
  whenever the corresponding cu_seqlens tensor is provided. If return_lse=False,
  LSE is still computed internally when needed for backward.
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

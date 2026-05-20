"""Layout-adapting FFPA entry shims for the CuTeDSL D=512 SM90 kernels.

The CuTeDSL kernels in :mod:`ffpa_attn.cutedsl._interface` operate on the
``[B, N, H, D]`` (or packed ``[T, H, D]``) layout (the upstream Dao-AILab
flash-attention layout convention reused here). The public FFPA APIs
(:func:`ffpa_attn.ffpa_attn_func`, :func:`ffpa_attn.ffpa_attn_varlen_func`)
present the SDPA-style ``[B, H, N, D]`` / FA-style ``[T, H, D]`` surface and
route ``forward_backend='cutedsl'`` through the unified
:class:`ffpa_attn.functional.FFPAAttnFunc` autograd boundary.
:func:`_ffpa_attn_forward_cutedsl` and :func:`_ffpa_attn_backward_cutedsl`
transpose between SDPA and FA layouts and dispatch through the registered
torch ops ``ffpa_attn::_fwd_cutedsl`` / ``ffpa_attn::_bwd_cutedsl``.

The module also centralises the **tensor-level** SM90 / D=512 / dtype gating
via :func:`_require_cutedsl_supported` and the **kwarg-level** compatibility
gating via :func:`_check_supported_options`: any non-default unsupported
option (``dropout_p``, ``attn_mask``, FlashAttention extensions) raises
``NotImplementedError`` with a single consolidated message.
"""

from __future__ import annotations

from typing import Optional

import torch

from ._interface import SUPPORTED_HEAD_DIM


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

  The cutedsl SplitD D=512 kernels (``_ffpa_attn_varlen_impl``,
  ``_ffpa_attn_forward_cutedsl``, ``_ffpa_attn_backward_cutedsl``) only
  honor dense / varlen D=512 attention with optional causal masking. Every other option commonly
  exposed by attention APIs (mask tensors, sliding window, softcap,
  score_mod, aux tensors, FlashAttention varlen extensions, dropout)
  has no kernel-side implementation and is rejected up front so callers
  see one actionable error rather than a deep kernel crash or silent
  semantic divergence.

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
      f"{source} only supports dense/varlen D=512 attention with optional "
      f"causal masking; unsupported options: {', '.join(unsupported)}. "
      f"Use forward_backend='triton' when these options are required."
    )


def cutedsl_forward_available(device: Optional[torch.device] = None) -> bool:
  """Return whether the CuTeDSL forward kernel can run on ``device``.

  CuTeDSL requires a Hopper (SM 9.x) CUDA device. Other backend constraints
  (head_dim, dtype, no mask/dropout) are enforced per-call by
  :func:`_require_cutedsl_supported`; this only checks the device-level
  prerequisite so callers can pre-select a backend before allocating tensors.
  """
  if not torch.cuda.is_available():
    return False
  if device is None:
    device = torch.device("cuda", torch.cuda.current_device())
  if device.type != "cuda":
    return False
  major, _ = torch.cuda.get_device_capability(device)
  return major == 9


def cutedsl_backward_available(device: Optional[torch.device] = None) -> bool:
  """Whether the CuTeDSL backward kernel can run on ``device``.

  Identical hardware requirement to forward; the backward kernel additionally
  requires bf16 inputs which is validated per-call.
  """
  return cutedsl_forward_available(device)


def _require_cutedsl_supported(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  *,
  requires_grad: bool,
) -> None:
  """Validate tensor-level constraints for the cutedsl backend.

  Checks device, SM90, head_dim==512, q/k/v dtype, and the bf16-only rule
  for training. Kwarg-level functional compatibility (``dropout_p``,
  ``attn_mask``, FlashAttention-extension kwargs) is **not** the
  responsibility of this function; that lives in
  :func:`_check_supported_options`, applied by the entry shims
  (:func:`_ffpa_attn_forward_cutedsl`, :func:`_ffpa_attn_varlen_cutedsl`).

  Raises ``NotImplementedError`` / ``RuntimeError`` / ``TypeError`` for
  any tensor-level violation so users who pass ``forward_backend='cutedsl'``
  see an actionable error rather than a deep kernel crash.
  """
  if q.device.type != "cuda":
    raise RuntimeError(f"cutedsl backend requires CUDA tensors, got device {q.device}")
  # Defensive: a CUDA tensor cannot exist unless CUDA was available at
  # allocation time, but the runtime can be poisoned (e.g. env var flips)
  # after tensors were created — keep this as a final guard.
  if not torch.cuda.is_available():
    raise RuntimeError("cutedsl backend requires a CUDA-capable build of PyTorch")
  major, _ = torch.cuda.get_device_capability(q.device)
  if major != 9:
    raise NotImplementedError(f"cutedsl backend only supports SM90 (Hopper); got compute capability {major}.x")
  if q.size(-1) != SUPPORTED_HEAD_DIM:
    raise NotImplementedError(f"cutedsl backend only supports head_dim={SUPPORTED_HEAD_DIM}; got {q.size(-1)}")
  if q.dtype not in (torch.float16, torch.bfloat16):
    raise TypeError(f"cutedsl backend requires torch.float16 or torch.bfloat16, got {q.dtype}")
  if requires_grad and q.dtype != torch.bfloat16:
    raise NotImplementedError("cutedsl backward currently supports torch.bfloat16 only; use bf16 inputs for training")
  if k.size(-1) != SUPPORTED_HEAD_DIM or v.size(-1) != SUPPORTED_HEAD_DIM:
    raise NotImplementedError(
      f"cutedsl backend requires k/v head_dim={SUPPORTED_HEAD_DIM}; "
      f"got k={k.size(-1)} v={v.size(-1)}"
    )


def _bhnd_to_bnhd(t: torch.Tensor) -> torch.Tensor:
  """Reshape ``[B, H, N, D]`` (SDPA) to the CuTeDSL-native ``[B, N, H, D]`` (FA).

  ``transpose(1, 2)`` always produces a non-contiguous view; the trailing
  ``.contiguous()`` materializes one copy. The downstream
  :func:`_ffpa_attn_forward_sm90` re-runs ``maybe_contiguous`` defensively,
  but materializing at the layout-boundary here keeps the "kernel input is
  always contiguous" invariant local and robust to future kernel-side changes.
  """
  return t.transpose(1, 2).contiguous()


def _bnhd_to_bhnd(t: torch.Tensor) -> torch.Tensor:
  """Reverse of :func:`_bhnd_to_bnhd`: FA ``[B, N, H, D]`` → SDPA ``[B, H, N, D]``."""
  return t.transpose(1, 2).contiguous()


def _ffpa_attn_forward_cutedsl(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  softmax_scale: float,
  causal: bool,
  *,
  return_lse: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
  """CuTeDSL SplitD forward for D=512 on SM90 with SDPA-layout in/out.

  Accepts ``[B, H, N, D]`` (SDPA) layout, transposes to the CuTeDSL-native
  ``[B, N, H, D]`` (FA) layout, calls the registered torch op
  ``torch.ops.ffpa_attn._fwd_cutedsl``, and transposes the output back.
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
  _require_cutedsl_supported(q, k, v, requires_grad=requires_grad)

  q_nhd, k_nhd, v_nhd = (_bhnd_to_bnhd(t) for t in (q, k, v))
  out_nhd, lse = torch.ops.ffpa_attn._fwd_cutedsl(
    q_nhd,
    k_nhd,
    v_nhd,
    softmax_scale,
    int(causal),
    int(return_lse),
  )
  out_bhnd = _bnhd_to_bhnd(out_nhd)
  return out_bhnd, lse


def _ffpa_attn_backward_cutedsl(
  grad_out: torch.Tensor,
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  out: torch.Tensor,
  lse: torch.Tensor,
  softmax_scale: float,
  causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """CuTeDSL SplitD backward for D=512 on SM90 with SDPA-layout in/out.

  Accepts all tensors in ``[B, H, N, D]`` (SDPA) layout, transposes to
  ``[B, N, H, D]`` (FA) for the registered torch op
  ``torch.ops.ffpa_attn._bwd_cutedsl``, and transposes the gradient outputs
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
  :returns: ``(dq, dk, dv)`` all in ``[B, H, N, D]`` SDPA layout.
  """
  q_nhd, k_nhd, v_nhd, out_nhd, dout_nhd = (_bhnd_to_bnhd(t) for t in (q, k, v, out, grad_out))
  dq_nhd, dk_nhd, dv_nhd = torch.ops.ffpa_attn._bwd_cutedsl(
    dout_nhd,
    q_nhd,
    k_nhd,
    v_nhd,
    out_nhd,
    lse,
    softmax_scale,
    int(causal),
  )
  dq, dk, dv = (_bnhd_to_bhnd(t) for t in (dq_nhd, dk_nhd, dv_nhd))
  return dq, dk, dv


def _ffpa_attn_varlen_cutedsl(
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
  the ``ffpa_attn::_varlen_fwd_cutedsl`` torch op directly.

  CuTeDSL varlen forward called from
  :func:`ffpa_attn.ffpa_attn_interface.ffpa_attn_varlen_func`. The CuTeDSL
  kernel consumes packed ``[T, H, D]`` layout natively — no transpose, no
  per-sequence loop.

  The kernel only consumes ``softmax_scale``, ``causal``, GQA-pack, and the
  cu_seqlens / max_seqlen tuple. Backend selection is already fixed by the
  public varlen API before reaching this wrapper, so this entry only enforces
  tensor and kwarg compatibility for the CuTeDSL path itself.

  All API-level guard-rail checks (unsupported FA-extension kwargs,
  dropout_p, q/k/v shape/dtype, cu_seqlens validity) are performed here.
  Kwarg compatibility is enforced by
  :func:`_check_supported_options`: any non-default value of ``dropout_p``
  or any of the FlashAttention-extension / mask / softcap / score_mod /
  aux_tensors / sink / block_mask kwargs raises ``NotImplementedError``
  with a single consolidated message.
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
    raise ValueError(f"ffpa_attn_varlen_func: k/v must share shape, got k={tuple(k.shape)} v={tuple(v.shape)}")
  if q.dtype not in (torch.float16, torch.bfloat16):
    raise TypeError(f"ffpa_attn_varlen_func: q/k/v must be fp16/bf16, got {q.dtype}")
  if k.dtype != q.dtype or v.dtype != q.dtype:
    raise TypeError(f"ffpa_attn_varlen_func: q/k/v must share dtype, got "
                    f"q={q.dtype} k={k.dtype} v={v.dtype}")

  if cu_seqlens_k is None:
    cu_seqlens_k = cu_seqlens_q
  if cu_seqlens_q.dtype != torch.int32 or cu_seqlens_k.dtype != torch.int32:
    raise TypeError("ffpa_attn_varlen_func: cu_seqlens_q/cu_seqlens_k must be int32")
  if cu_seqlens_q.numel() != cu_seqlens_k.numel() or cu_seqlens_q.numel() < 2:
    raise ValueError("ffpa_attn_varlen_func: cu_seqlens_q and cu_seqlens_k must share length >= 2")

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

  # dtype was already validated above with a user-facing
  # "ffpa_attn_varlen_func: ..." prefix; _require_cutedsl_supported re-checks
  # it as the tensor-level single source of truth.
  requires_grad = any(t.requires_grad for t in (q, k, v))
  _require_cutedsl_supported(q, k, v, requires_grad=requires_grad)

  from . import _ffpa_attn_varlen_impl

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


__all__ = [
  "_ffpa_attn_forward_cutedsl",
  "_ffpa_attn_backward_cutedsl",
  "_ffpa_attn_varlen_cutedsl",
  "cutedsl_forward_available",
  "cutedsl_backward_available",
  "_require_cutedsl_supported",
]

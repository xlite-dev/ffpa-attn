"""Layout-adapting FFPA entry shims for the CuTeDSL D=512 SM90 kernels.

The CuTeDSL kernels in :mod:`ffpa_attn.cutedsl._interface` operate on the
``[B, N, H, D]`` (or packed ``[T, H, D]``) layout (the upstream Dao-AILab
flash-attention layout convention reused here). The public FFPA APIs
(:func:`ffpa_attn.ffpa_attn_func`, :func:`ffpa_attn.ffpa_attn_varlen_func`)
present the SDPA-style ``[B, H, N, D]`` / FA-style ``[T, H, D]`` surface and
route ``forward_backend='cutedsl'`` directly into :func:`_ffpa_attn_cutedsl`
and :func:`_ffpa_attn_varlen_cutedsl` defined here, which transpose and
dispatch into :func:`ffpa_attn_splitd_func` /
:func:`ffpa_attn_splitd_varlen_func`. Autograd is owned by
:class:`ffpa_attn.cutedsl._interface.FFPAAttnSplitDFunc`, not by
:class:`ffpa_attn.functional.FFPAAttnFunc`.

The module also centralises the **tensor-level** SM90 / D=512 / dtype gating
via :func:`_require_cutedsl_supported` and the **kwarg-level** compatibility
gating via :func:`_check_supported_options`: any non-default unsupported
option (``dropout_p``, ``attn_mask``, FlashAttention extensions) raises
``NotImplementedError`` with a single consolidated message.
"""

from __future__ import annotations

import math
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

  The cutedsl SplitD D=512 kernels (``ffpa_attn_splitd_func``,
  ``ffpa_attn_splitd_varlen_func``) only honor dense / varlen D=512
  attention with optional causal masking. Every other option commonly
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
  (:func:`_ffpa_attn_cutedsl`, :func:`_ffpa_attn_varlen_cutedsl`).

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


def _ffpa_attn_cutedsl(
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  *,
  attn_mask: torch.Tensor | None,
  dropout_p: float,
  is_causal: bool,
  scale: float | None,
  enable_gqa: bool,
) -> torch.Tensor:
  """Dense ``[B, H, N, D]`` cutedsl entry called from
  :func:`ffpa_attn.ffpa_attn_interface.ffpa_attn_func` whenever
  ``forward_backend == 'cutedsl'``. Sibling of
  :func:`_ffpa_attn_varlen_cutedsl`. Routes through
  :func:`ffpa_attn.cutedsl._interface.ffpa_attn_splitd_func`, which wraps
  ``FFPAAttnSplitDFunc.apply(...)`` for autograd — the cutedsl backend owns
  its own autograd boundary and never traverses
  :class:`ffpa_attn.functional.FFPAAttnFunc`.

  Layout conversion: SDPA ``[B, Nh_q, Nq, D]`` is transposed to FA
  ``[B, Nq, Nh_q, D]`` on the way in and back on the way out.

  ``attn_mask`` and ``dropout_p`` are accepted for public-API uniformity
  but the cutedsl SplitD kernel does not implement either: any non-default
  value raises ``NotImplementedError`` from :func:`_check_supported_options`.
  Use ``forward_backend='triton'`` if dropout / mask are functionally
  required.
  """
  from ._interface import ffpa_attn_splitd_func

  _check_supported_options(
    source="ffpa_attn_func(forward_backend='cutedsl')",
    dropout_p=dropout_p,
    attention_mask=attn_mask,
  )

  requires_grad = any(t.requires_grad for t in (query, key, value))
  _require_cutedsl_supported(query, key, value, requires_grad=requires_grad)
  if not enable_gqa and query.size(1) != key.size(1):
    raise ValueError(
      f"ffpa_attn_func: enable_gqa=False but query num_heads ({query.size(1)}) "
      f"!= key/value num_heads ({key.size(1)}); set enable_gqa=True or match head counts."
    )

  q_nhd, k_nhd, v_nhd = (_bhnd_to_bnhd(t) for t in (query, key, value))

  softmax_scale = scale if scale is not None else (1.0 / math.sqrt(query.size(-1)))
  # pack_gqa: omitted; _ffpa_attn_forward_sm90 auto-detects via qhead_per_kvhead > 1.
  out_nhd = ffpa_attn_splitd_func(
    q_nhd,
    k_nhd,
    v_nhd,
    softmax_scale=softmax_scale,
    causal=is_causal,
    return_lse=False,
  )
  return _bnhd_to_bhnd(out_nhd)


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
  the ``ffpa_attn::splitd_fwd_sm90`` torch op directly.

  CuTeDSL varlen forward called from
  :func:`ffpa_attn.ffpa_attn_interface.ffpa_attn_varlen_func`. The CuTeDSL
  kernel consumes packed ``[T, H, D]`` layout natively — no transpose, no
  per-sequence loop.

  ``kwargs`` is forwarded to :meth:`FFPAAttnMeta.from_kwargs` only for
  forward/backward backend pair-binding and unknown-key validation; the
  kernel only consumes ``softmax_scale``, ``causal``, GQA-pack, and the
  cu_seqlens / max_seqlen tuple. ``forward_backend`` / ``backward_backend``
  default to ``"cutedsl"`` here so callers do not need to plumb them through
  every time — the varlen API is cutedsl-only by construction.

  All API-level guard-rail checks (unsupported FA-extension kwargs,
  dropout_p, forward_backend, q/k/v shape/dtype, cu_seqlens validity) are
  performed here so :func:`ffpa_attn.ffpa_attn_varlen_func` can remain a
  thin shim. Kwarg compatibility is enforced by
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

  forward_backend = kwargs.get("forward_backend", "cutedsl")
  if forward_backend != "cutedsl":
    raise NotImplementedError(
      f"ffpa_attn_varlen_func: only forward_backend='cutedsl' is supported, "
      f"got {forward_backend!r}. Unpack the batch and call ffpa_attn_func "
      f"per sequence for other backends."
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

  from ..functional import FFPAAttnMeta

  meta_kwargs = dict(kwargs)
  meta_kwargs.setdefault("forward_backend", "cutedsl")
  meta_kwargs.setdefault("backward_backend", "cutedsl")
  meta = FFPAAttnMeta.from_kwargs(**meta_kwargs)
  if meta.forward_backend != "cutedsl" or meta.backward_backend != "cutedsl":
    raise ValueError(
      f"ffpa_attn_varlen_func: backends must both be 'cutedsl'; got "
      f"forward={meta.forward_backend!r} backward={meta.backward_backend!r}"
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

  # dtype was already validated above with a user-facing
  # "ffpa_attn_varlen_func: ..." prefix; _require_cutedsl_supported re-checks
  # it as the tensor-level single source of truth.
  requires_grad = any(t.requires_grad for t in (q, k, v))
  _require_cutedsl_supported(q, k, v, requires_grad=requires_grad)

  from ._interface import ffpa_attn_splitd_varlen_func

  return ffpa_attn_splitd_varlen_func(
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
  "_ffpa_attn_cutedsl",
  "_ffpa_attn_varlen_cutedsl",
  "cutedsl_forward_available",
  "cutedsl_backward_available",
  "_require_cutedsl_supported",
]

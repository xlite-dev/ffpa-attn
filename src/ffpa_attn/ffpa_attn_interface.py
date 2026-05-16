"""Public Python interface for FFPA prefill attention.

The CUDA and Triton backend packages register forward kernels as
``torch.library`` operators under ``torch.ops.ffpa_attn`` so that
``torch.compile`` can trace through the forward path as proper custom ops.

Backward pass delegates to Triton FFPA backward or PyTorch SDPA backward
functions, routing by headdim and user-selected backend.
Small-D directly delegates to ``torch.nn.functional.scaled_dot_product_attention``;
large-D forward continues to use the FFPA Triton kernel by default, with a
legacy optional CUDA forward backend available only when compiled in.

When the caller opts into the CuTeDSL backend on D=512 / SM90 inputs
(``forward_backend='cutedsl'``) ``ffpa_attn_func`` short-circuits the FFPA
multi-backend dispatcher and routes straight to
:func:`ffpa_attn.cutedsl.interface.split_flash_attn_func`, skipping
``FFPAAttnMeta`` normalization and the ``FFPAAttnFunc`` autograd boundary.
Ineligible cutedsl requests fall through to the standard dispatcher so
error messages stay consistent.

Variable-length (packed THD) attention is exposed via ``ffpa_attn_varlen_func``,
which mirrors the FlashAttention varlen surface (``q, k, v, cu_seqlens_q,
cu_seqlens_k, max_seqlen_q, max_seqlen_k, ...``) and dispatches directly to the
CuTeDSL ``splitd_flash_attn::varlen_fwd`` autograd-registered torch op. The
varlen API is currently CuTeDSL-only (SM90, D=512); other shapes / backends
raise ``NotImplementedError``.
"""

from __future__ import annotations

import math

import torch
from .functional import FFPAAttnFunc, FFPAAttnMeta


def _should_fallback_to_sdpa(
  query: torch.Tensor,
  key: torch.Tensor,
  attn_mask: torch.Tensor | None,
  dropout_p: float,
  forward_backend: str,
) -> bool:
  """Return whether the public API should delegate to SDPA directly.

  For now, as FFPA is mainly designed for prefill and may not outperform SDPA
  for short sequences. While Nq == 1 is a common case for decode attention and
  FFPA does support it by flash-decoding algorithm, the speedup over SDPA is may
  not be significant (~10% speedup on Ada). FFPA currently falls back to SDPA for
  the following cases:

  * ``head_dim <= 256``
  * ``head_dim > 1024``
  * ``dropout_p > 0.0`` when the large-D forward backend is not Triton
  * ``attn_mask is not None`` when the large-D forward backend is not Triton
  * ``8 <= Nq < 512``
  * ``Nk < 512``

  As FFPA grows support for these cases, remove the corresponding condition
  here instead of scattering dispatch checks throughout ``ffpa_attn_func``.
  """
  assert query.dim() == 4, "Expected query shape [B, Nh_q, Nq, D]"
  assert key.dim() == 4, "Expected key shape [B, Nh_kv, Nkv, D]"
  B, Nh_q, Nq, D = query.shape  # noqa: F841
  _, Nh_kv, Nkv, D_k = key.shape
  assert D == D_k, "Query and key must have the same head dimension"
  # CuTeDSL backend targets only D == 512 on SM90 but covers arbitrary Nq/Nkv;
  # delegate D coverage decisions to functional.normalize (it raises on
  # D != 512) and only fall back here when D is entirely outside FFPA's
  # large-D range, so SDPA still handles small/oversized cases uniformly.
  if forward_backend == "cutedsl":
    return D <= 256 or D > 1024
  _fallback = any([
    D <= 256,
    D > 1024,
    # dropout is only supported by triton backend for now.
    dropout_p > 0.0 and forward_backend != "triton",
    # attn_mask is only supported by triton backend for now.
    attn_mask is not None and forward_backend != "triton",
    (8 <= Nq < 512),
    Nkv < 512,
  ])
  return _fallback


def _should_take_cutedsl_fast_path(
  query: torch.Tensor,
  attn_mask: torch.Tensor | None,
  dropout_p: float,
  forward_backend: str,
) -> bool:
  """Return whether ``ffpa_attn_func`` can short-circuit to the D=512 CuTeDSL path.

  Lightweight pre-check; final hard validation happens via
  :func:`_require_cutedsl_supported` inside the fast-path body. Ineligible
  cutedsl requests fall through to the standard ``FFPAAttnMeta`` route so
  the error path is unchanged. The leading ``forward_backend`` early-return
  keeps the ``cutedsl`` package import lazy for non-cutedsl callers.
  """
  if forward_backend != "cutedsl":
    return False
  from .cutedsl._wrappers import cutedsl_forward_available
  return (query.size(-1) == 512 and attn_mask is None and dropout_p == 0.0 and cutedsl_forward_available(query.device))


def _ffpa_attn_func_cutedsl_fast_path(
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  is_causal: bool,
  scale: float | None,
  enable_gqa: bool,
) -> torch.Tensor:
  """Direct D=512 CuTeDSL forward — bypasses ``FFPAAttnMeta`` and ``FFPAAttnFunc``.

  Routes through :func:`ffpa_attn.cutedsl.interface.split_flash_attn_func`,
  which transparently wraps ``FlashAttnFunc.apply(...)`` for autograd. Layout
  conversion mirrors
  :func:`ffpa_attn.cutedsl._wrappers._ffpa_attn_forward_cutedsl`: SDPA
  ``[B, Nh_q, Nq, D]`` is transposed to FA ``[B, Nq, Nh_q, D]`` on the way in
  and back to SDPA layout on the way out.

  Hard validation is delegated to
  :func:`ffpa_attn.cutedsl._wrappers._require_cutedsl_supported` so error
  messages are identical to the slow-path dispatch.
  """
  from .cutedsl._wrappers import _require_cutedsl_supported
  from .cutedsl.interface import split_flash_attn_func

  requires_grad = any(t.requires_grad for t in (query, key, value))
  _require_cutedsl_supported(
    query,
    key,
    value,
    is_causal=is_causal,
    dropout_p=0.0,
    attn_bias=None,
    enable_gqa_user=enable_gqa,
    requires_grad=requires_grad,
  )
  if not enable_gqa and query.size(1) != key.size(1):
    raise ValueError(
      f"ffpa_attn_func: enable_gqa=False but query num_heads ({query.size(1)}) "
      f"!= key/value num_heads ({key.size(1)}); set enable_gqa=True or match head counts."
    )

  # SDPA [B, H, N, D]  →  FA [B, N, H, D]
  q_nhd, k_nhd, v_nhd = (t.transpose(1, 2).contiguous() for t in (query, key, value))

  softmax_scale = scale if scale is not None else (1.0 / math.sqrt(query.size(-1)))
  # pack_gqa: omitted; _flash_attn_fwd_sm90 auto-detects via qhead_per_kvhead > 1.
  out_nhd = split_flash_attn_func(
    q_nhd,
    k_nhd,
    v_nhd,
    softmax_scale=softmax_scale,
    causal=is_causal,
    return_lse=False,
  )
  # FA [B, N, H, D]  →  SDPA [B, H, N, D]
  return out_nhd.transpose(1, 2).contiguous()


def ffpa_attn_func(
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  attn_mask: torch.Tensor | None = None,
  dropout_p: float = 0.0,
  is_causal: bool = False,
  scale: float | None = None,
  enable_gqa: bool = False,
  **kwargs: object,
) -> torch.Tensor:
  """FFPA: Faster Flash Prefill Attention for large headdims (D > 256).

  Signature aligned with ``torch.nn.functional.scaled_dot_product_attention``.
  Dispatches by ``query.dtype`` (fp16 / bf16) and ``acc`` through a single
  registered torch op (``torch.ops.ffpa_attn.attn``), keeping the
  Python layer minimal and fully compatible with ``torch.compile``.

  Supports cross-attention where ``query`` seqlen (``Nq``) differs from
  ``key``/``value`` seqlen (``Nkv``) and grouped-query attention where
  ``query`` has more heads than ``key``/``value`` (MQA is the
  ``Nh_kv == 1`` special case). ``key`` and ``value`` must share the same
  ``Nh_kv`` and the same ``Nkv``. Causal masking is supported via
  ``is_causal=True`` with queries aligned to the tail of the KV sequence
  (``Nkv >= Nq`` required).

  Backward pass is supported via :class:`FFPAAttnFunc`. The public API falls
  back to SDPA for cases FFPA does not currently support directly (small-D,
  ``D > 1024``, and non-Triton large-D dropout), and otherwise keeps the
  existing FFPA forward plus SDPA/FFPA backward routing. Large-D Triton forward
  and backward support explicit additive ``attn_mask`` gradients.
  ``forward_backend`` only affects the large-D path.

  :param query: Query tensor with layout ``[B, Nh_q, Nq, D]``; dtype must be
      ``torch.float16`` or ``torch.bfloat16`` and match ``key`` / ``value``.
  :param key: Key tensor with layout ``[B, Nh_kv, Nkv, D]``; same dtype as
      ``query``. ``Nh_q`` must be an integer multiple of ``Nh_kv``
      (``group_size = Nh_q / Nh_kv``).
  :param value: Value tensor with layout ``[B, Nh_kv, Nkv, D]``; same dtype
      as ``query``. ``key`` and ``value`` must share the same ``Nh_kv`` and
      ``Nkv``.
  :param attn_mask: Optional attention mask broadcastable to
      ``[B, Nh_q, Nq, Nkv]``. Boolean masks follow SDPA semantics where
      ``True`` means the element participates in attention; floating masks are
      additive attention bias. Large-D Triton supports additive mask gradients.
  :param dropout_p: Dropout probability. Large-D Triton implements SDPA-style
      attention dropout; non-Triton large-D dropout routes to SDPA.
  :param is_causal: When ``True``, apply a causal attention mask so that
      query row ``r`` only attends to KV positions ``k <= r + (Nkv - Nq)``
      (standard ``queries aligned to KV tail`` convention). Requires
      ``Nkv >= Nq``. Non-causal tiles pay only one compare-and-branch
      per KV tile; diagonal tiles apply a per-fragment -inf mask.
  :param scale: Pre-softmax scaling factor applied to ``QK^T``.
      Defaults to ``1 / sqrt(D)`` (standard attention scale) when ``None``.
  :param enable_gqa: Grouped-query attention mode. Defaults to ``False`` to
      match SDPA exactly. When ``False``, the large-D FFPA path requires
      ``query`` and ``key``/``value`` to have the same number of heads. Pass
      ``True`` to opt into GQA/MQA semantics explicitly.
  :param kwargs: Implementation-specific options for experimentation.
      Supported keys are ``stages``, ``acc``, ``enable_forward_tma``,
      ``enable_backward_tma``, ``enable_forward_ws``,
      ``enable_backward_ws``, ``enable_tma``, ``enable_ws``,
      ``high_precision_grad``, ``forward_backend``,
      ``triton_autotune``, ``triton_autotune_mode``,
      ``backward_backend``,
      ``triton_backward_preprocess_d_chunk``, and
      ``triton_backward_grad_v_storage_dtype``. ``forward_backend`` only affects ``D > 256``.
      ``enable_forward_tma`` and ``enable_backward_tma`` independently opt
      into the SM90+ Triton descriptor/TMA forward and backward paths when
      supported. ``enable_forward_ws`` and ``enable_backward_ws`` request
      warp-specialized TMA configs for the matching direction. ``enable_tma``
      and ``enable_ws`` are compatibility aliases that set both directions.
      ``backward_backend`` supports ``"triton"`` and ``"sdpa"``.
      ``triton_backward_grad_v_storage_dtype`` defaults to ``None`` and
      currently only accepts ``torch.float32`` as an override for Triton
      backward's internal ``DV`` storage dtype. These options do not change
      the autograd contract; unknown keys raise ``TypeError``.

  :returns: Output tensor ``O`` with layout ``[B, Nh_q, Nq, D]``,
      filled with the attention output ``softmax(scale * QK^T) V``.

  :raises TypeError: if ``query.dtype`` is neither fp16 nor bf16.
  :raises ValueError: if ``acc`` is not one of ``{'f16', 'f32'}``, if
      bf16 activations are combined with ``acc='f16'`` (no bf16-acc mma
      PTX instruction exists on supported architectures), or if
      ``is_causal=True`` is combined with ``Nkv < Nq``.
  :raises NotImplementedError: propagated from SDPA or FFPA backends for
      unsupported backend-specific combinations.
  """
  meta = FFPAAttnMeta.from_kwargs(**kwargs)
  if _should_fallback_to_sdpa(query, key, attn_mask, dropout_p, meta.forward_backend):
    # Fallback intentionally delegates to SDPA exactly as the user called it.
    # Do not synthesize masks or reinterpret GQA semantics here.
    # HACK: Use the native SDPA op directly to avoid recursive calls to this function
    # if the user has monkey-patched torch.nn.functional.scaled_dot_product_attention
    # to point to this function (e.g., for benchmarking). For example:
    # >>> import torch.nn.functional as F
    # >>> from ffpa_attn import ffpa_attn_func
    # >>> F.scaled_dot_product_attention = ffpa_attn_func
    return torch._C._nn.scaled_dot_product_attention(
      query,
      key,
      value,
      attn_mask=attn_mask,
      dropout_p=dropout_p,
      is_causal=is_causal,
      scale=scale,
      enable_gqa=enable_gqa,
    )

  # D=512 SM90 CuTeDSL fast-path — opt-in via forward_backend='cutedsl'.
  # Bypasses FFPAAttnMeta normalization and the FFPAAttnFunc autograd boundary,
  # dispatching directly to cutedsl.interface.split_flash_attn_func (autograd
  # via FlashAttnFunc). Ineligible cutedsl requests fall through so the slow
  # path raises the canonical NotImplementedError.
  if _should_take_cutedsl_fast_path(query, attn_mask, dropout_p, meta.forward_backend):
    return _ffpa_attn_func_cutedsl_fast_path(
      query,
      key,
      value,
      is_causal=is_causal,
      scale=scale,
      enable_gqa=enable_gqa,
    )

  meta, query, key, value, attn_bias = meta.normalize_inputs(
    query,
    key,
    value,
    attn_mask,
    dropout_p,
    is_causal,
    scale,
    enable_gqa,
  )

  return FFPAAttnFunc.apply(query, key, value, attn_bias, meta)


def _ffpa_attn_varlen_cutedsl(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  cu_seqlens_q: torch.Tensor,
  cu_seqlens_k: torch.Tensor,
  max_seqlen_q: int,
  max_seqlen_k: int,
  *,
  softmax_scale: float | None,
  causal: bool,
  enable_gqa: bool,
  return_lse: bool,
  kwargs: dict,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
  """CuTeDSL varlen shim: SM90 D=512 packed THD attention.

  Dispatches directly to the autograd-registered
  ``splitd_flash_attn::varlen_fwd`` torch op. The CuTeDSL kernel consumes
  packed ``[T, H, D]`` layout natively — no transpose, no per-sequence loop.

  ``kwargs`` is forwarded to :meth:`FFPAAttnMeta.from_kwargs` only for
  forward/backward backend pair-binding and unknown-key validation; the
  kernel only consumes ``softmax_scale``, ``causal``, GQA-pack, and the
  cu_seqlens / max_seqlen tuple. ``forward_backend`` / ``backward_backend``
  default to ``"cutedsl"`` here so that callers do not need to plumb them
  through every time — the varlen API is cutedsl-only by construction.
  """
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

  # Side-effect import: registers splitd_flash_attn::varlen_fwd / varlen_bwd
  # and surfaces the cutedsl support gate.
  from . import cutedsl  # noqa: F401
  from .cutedsl._wrappers import _require_cutedsl_supported
  from .cutedsl.interface import _encode_optional_int_for_custom_op

  requires_grad = any(t.requires_grad for t in (q, k, v))
  _require_cutedsl_supported(
    q,
    k,
    v,
    is_causal=causal,
    dropout_p=0.0,
    attn_bias=None,
    enable_gqa_user=enable_gqa,
    requires_grad=requires_grad,
  )

  D = q.size(-1)
  scale = float(softmax_scale) if softmax_scale is not None else 1.0 / math.sqrt(D)
  # pack_gqa mirrors interface.py: enable when Nh_q > Nh_kv.
  pack_gqa = q.size(-2) > k.size(-2)

  out_packed, lse_packed = torch.ops.splitd_flash_attn.varlen_fwd(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    int(max_seqlen_q),
    int(max_seqlen_k),
    scale,
    bool(causal),
    _encode_optional_int_for_custom_op(None),  # window_size_left
    _encode_optional_int_for_custom_op(None),  # window_size_right
    0.0,  # softcap
    bool(pack_gqa),
  )

  if not return_lse:
    return out_packed
  return out_packed, lse_packed


_UNSUPPORTED_VARLEN_KWARGS = ("seqused_k", "block_table", "num_splits", "window_size", "alibi_slopes", "softcap")


def ffpa_attn_varlen_func(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  cu_seqlens_q: torch.Tensor,
  cu_seqlens_k: torch.Tensor | None,
  max_seqlen_q: int,
  max_seqlen_k: int,
  *,
  dropout_p: float = 0.0,
  softmax_scale: float | None = None,
  causal: bool = False,
  enable_gqa: bool = False,
  return_lse: bool = False,
  **kwargs: object,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
  """FFPA variable-length attention (packed THD, FlashAttention-style).

  Signature aligned with Dao-AILab ``flash_attn_varlen_func``. Inputs are
  packed THD: ``q`` is ``[T_q, H_q, D]`` and ``k`` / ``v`` are
  ``[T_k, H_kv, D]``. Sequence boundaries are described by ``cu_seqlens_q``
  and ``cu_seqlens_k`` (int32 CUDA tensors of length ``B+1`` starting at 0).
  When ``cu_seqlens_k is None`` it defaults to ``cu_seqlens_q`` (self-attention).

  Only the CuTeDSL backend is supported: SM90 Hopper, ``D == 512``, fp16 /
  bf16 (bf16 required for training). Any unsupported case raises an
  actionable error immediately — there is no silent fallback to dense /
  per-sequence paths. Callers needing other shapes / backends should
  unpack the batch and call :func:`ffpa_attn_func` per sequence.

  :param q: Query tensor of shape ``[T_q, H_q, D]``.
  :param k: Key tensor of shape ``[T_k, H_kv, D]``.
  :param v: Value tensor of shape ``[T_k, H_kv, D]``.
  :param cu_seqlens_q: ``[B+1]`` int32 CUDA tensor; ``cu_seqlens_q[0] == 0``
      and ``cu_seqlens_q[-1] == T_q``.
  :param cu_seqlens_k: Same convention for keys; defaults to
      ``cu_seqlens_q`` if ``None``.
  :param max_seqlen_q: Maximum per-sequence query length across the batch.
  :param max_seqlen_k: Maximum per-sequence key length across the batch.
  :param dropout_p: FlashAttention-compat; must be ``0.0`` (CuTeDSL has no
      dropout support).
  :param softmax_scale: Pre-softmax scaling factor; defaults to
      ``1 / sqrt(D)``.
  :param causal: Apply a lower-right (tail-aligned) causal mask.
  :param enable_gqa: Opt-in to GQA/MQA (``H_q != H_kv``). When ``False``,
      ``H_q`` must equal ``H_kv``.
  :param return_lse: When ``True``, also return the log-sum-exp tensor of
      shape ``[H_q, T_q]`` in fp32 (CUDA convention).
  :param kwargs: Implementation hooks consumed by ``FFPAAttnMeta`` for
      pair-binding (only ``forward_backend`` / ``backward_backend`` are
      meaningful; both must be ``"cutedsl"`` or unset).

  :returns: ``out`` of shape ``[T_q, H_q, D]`` if ``return_lse=False``,
      otherwise ``(out, lse)``.

  :raises NotImplementedError: for ``D != 512``, non-SM90 hardware,
      ``dropout_p > 0``, non-CuTeDSL ``forward_backend``, or any unsupported
      FlashAttention extension kwarg (``seqused_k``, ``block_table``,
      ``num_splits``, ``window_size``, ``alibi_slopes``, ``softcap``).
  :raises TypeError: if ``cu_seqlens_*`` is not int32, or if dtype is not
      fp16/bf16.
  :raises ValueError: for shape mismatches between ``q``/``k``/``v``,
      malformed ``cu_seqlens_*``, or ``enable_gqa=False`` with
      ``H_q != H_kv``.
  """
  for unsupported in _UNSUPPORTED_VARLEN_KWARGS:
    if unsupported in kwargs:
      raise NotImplementedError(f"ffpa_attn_varlen_func: '{unsupported}' is not supported")
  if dropout_p != 0.0:
    raise NotImplementedError(f"ffpa_attn_varlen_func: dropout_p must be 0.0 (CuTeDSL has no dropout), got {dropout_p}")
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

  return _ffpa_attn_varlen_cutedsl(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale=softmax_scale,
    causal=causal,
    enable_gqa=enable_gqa,
    return_lse=return_lse,
    kwargs=kwargs,
  )


__all__ = ["ffpa_attn_func", "ffpa_attn_varlen_func"]

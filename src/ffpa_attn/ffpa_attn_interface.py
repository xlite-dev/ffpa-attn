"""Public Python interface for FFPA prefill attention.

The CUDA and Triton backend packages register forward kernels as
``torch.library`` operators under ``torch.ops.ffpa_attn`` so that
``torch.compile`` can trace through the forward path as proper custom ops.

Backward pass delegates to Triton FFPA backward or PyTorch SDPA backward
functions, routing by headdim and user-selected backend.
Small-D directly delegates to ``torch.nn.functional.scaled_dot_product_attention``;
large-D forward continues to use the FFPA Triton kernel by default, with a
legacy optional CUDA forward backend available only when compiled in.
"""

from __future__ import annotations

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
      Supported keys are ``stages``, ``acc``, ``enable_tma``,
      ``high_precision_grad``, ``forward_backend``,
      ``triton_autotune``, ``triton_autotune_mode``,
      ``backward_backend``,
      ``triton_backward_preprocess_d_chunk``, and
      ``triton_backward_grad_v_storage_dtype``. ``forward_backend`` only affects ``D > 256``.
      ``enable_tma`` is reserved for future Triton kernels and is currently a
      no-op. ``backward_backend`` supports ``"triton"`` and ``"sdpa"``.
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


__all__ = ["ffpa_attn_func"]

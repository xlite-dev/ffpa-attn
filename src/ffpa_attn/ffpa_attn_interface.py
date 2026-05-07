"""Public Python interface for FFPA prefill attention.

The CUDA and Triton backend packages register forward / backward kernels as
``torch.library`` operators under ``torch.ops.ffpa_attn`` so that
``torch.compile`` can trace through the forward path as proper custom ops.

Backward pass delegates to the registered FFPA backward ops or PyTorch SDPA
backward functions, routing by headdim and user-selected backend.
Small-D directly delegates to ``torch.nn.functional.scaled_dot_product_attention``;
large-D forward continues to use the FFPA CUDA or Triton kernels.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: F401
from .functional import FFPAAttnFunc, FFPAAttnMeta


def _should_fallback_to_sdpa(
  query: torch.Tensor,
  attn_mask: torch.Tensor | None,
  dropout_p: float,
) -> bool:
  """Return whether the public API should delegate to SDPA directly.

  FFPA currently falls back to SDPA for the following cases:

  * ``head_dim <= 256``
  * ``head_dim > 1024``
  * ``attn_mask is not None``
  * ``dropout_p > 0.0``

  As FFPA grows support for these cases, remove the corresponding condition
  here instead of scattering dispatch checks throughout ``ffpa_attn_func``.
  """
  assert query.dim() == 4, "Expected query shape [B, Nh_q, Nq, D]"
  head_dim = query.size(3)
  return head_dim <= 256 or head_dim > 1024 or attn_mask is not None or dropout_p > 0.0


def _normalize_inputs(
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  attn_mask: torch.Tensor | None,
  dropout_p: float,
  is_causal: bool,
  scale: float | None,
  enable_gqa: bool,
  **kwargs: object,
) -> FFPAAttnMeta:
  """Validate and normalise all public-API inputs, returning a meta object."""
  return FFPAAttnMeta.from_kwargs(**kwargs).normalize(
    query,
    key,
    value,
    attn_mask,
    dropout_p,
    is_causal,
    scale,
    enable_gqa,
  )


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
  ``D > 1024``, explicit ``attn_mask``, and dropout), and otherwise keeps the
  existing FFPA forward plus SDPA/FFPA backward routing.
  ``forward_backend`` only affects the large-D path.

  :param query: Query tensor with layout ``[B, Nh_q, Nq, D]``; dtype must be
      ``torch.float16`` or ``torch.bfloat16`` and match ``key`` / ``value``.
  :param key: Key tensor with layout ``[B, Nh_kv, Nkv, D]``; same dtype as
      ``query``. ``Nh_q`` must be an integer multiple of ``Nh_kv``
      (``group_size = Nh_q / Nh_kv``).
  :param value: Value tensor with layout ``[B, Nh_kv, Nkv, D]``; same dtype
      as ``query``. ``key`` and ``value`` must share the same ``Nh_kv`` and
      ``Nkv``.
  :param attn_mask: Optional attention mask (``[Nq, Nkv]`` bool or float).
      Passing a non-``None`` value currently routes the call to SDPA.
  :param dropout_p: Dropout probability. Passing ``dropout_p > 0`` currently
      routes the call to SDPA because the FFPA native kernels do not yet
      implement dropout.
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
      ``triton_forward_autotune``, ``backward_backend``,
      ``triton_backward_autotune``, ``triton_backward_version``, and
      ``triton_backward_preprocess_d_chunk``. ``forward_backend`` only
      affects ``D > 256``. These options do not change the autograd
      contract; unknown keys raise ``TypeError``.

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
  if _should_fallback_to_sdpa(query, attn_mask, dropout_p):
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

  _meta = _normalize_inputs(
    query,
    key,
    value,
    attn_mask,
    dropout_p,
    is_causal,
    scale,
    enable_gqa,
    **kwargs,
  )

  return FFPAAttnFunc.apply(query, key, value, _meta)


__all__ = ["ffpa_attn_func"]

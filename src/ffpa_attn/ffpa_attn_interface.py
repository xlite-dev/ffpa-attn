"""Public Python interface for FFPA prefill attention.

The CUDA backend package registers the native forward kernel as a
``torch.library`` operator so callers (including ``torch.compile`` graphs)
can reach it through ``torch.ops.ffpa_attn.attn`` instead of calling the
C-extension symbol directly.

Backward pass delegates to PyTorch SDPA backward functions, routing by
headdim: flash_attention_backward for D <= 256, efficient_attention_backward
for D > 256. Small-D forward/backward use PyTorch's aten flash-attention
operator pair; large-D forward continues to use the FFPA CUDA or Triton
kernels.
"""

from __future__ import annotations

import torch

from .functional import FFPAAttnFunc, FFPAAttnMeta


def _normalize_inputs(
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  attn_mask: torch.Tensor | None,
  dropout_p: float,
  is_causal: bool,
  scale: float | None,
  enable_gqa: bool | None,
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
  enable_gqa: bool | None = None,
  **kwargs: object,
) -> torch.Tensor:
  """Unified FFPA prefill attention entry.

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

  Backward pass is supported via :class:`FFPAAttnFunc`. For ``D <= 256`` it
  uses PyTorch's flash-attention forward/backward pair; for ``D > 256`` it
  keeps the existing FFPA forward plus SDPA/FFPA backward routing.
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
      Currently **not supported** — passing a non-``None`` value raises
      ``NotImplementedError``. Reserved for future use.
  :param dropout_p: Dropout probability. Only supported for ``D <= 256``
      (the PyTorch flash-attention path). When ``dropout_p > 0`` and
      ``D > 256``, a ``NotImplementedError`` is raised because the FFPA
      native kernels do not implement dropout.
  :param is_causal: When ``True``, apply a causal attention mask so that
      query row ``r`` only attends to KV positions ``k <= r + (Nkv - Nq)``
      (standard ``queries aligned to KV tail`` convention). Requires
      ``Nkv >= Nq``. Non-causal tiles pay only one compare-and-branch
      per KV tile; diagonal tiles apply a per-fragment -inf mask.
  :param scale: Pre-softmax scaling factor applied to ``QK^T``.
      Defaults to ``1 / sqrt(D)`` (standard attention scale) when ``None``.
  :param enable_gqa: Grouped-query attention mode. ``None`` (default)
      auto-detects GQA when ``query`` and ``key``/``value`` have different
      numbers of heads. ``True`` forces GQA expansion (broadcast K/V heads
      even when head counts match). ``False`` disables GQA (raises
      ``ValueError`` if head counts differ).
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
  :raises NotImplementedError: if ``attn_mask`` is not ``None``, or if
      ``dropout_p > 0`` with ``D > 256``.
  """
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

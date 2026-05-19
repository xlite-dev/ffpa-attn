"""Public Python interface for FFPA prefill attention.

The CUDA and Triton backend packages register forward kernels as
``torch.library`` operators under ``torch.ops.ffpa_attn`` so that
``torch.compile`` can trace through the forward path as proper custom ops.

Backward pass delegates to Triton FFPA backward or PyTorch SDPA backward
functions, routing by headdim and user-selected backend.
Small-D directly delegates to ``torch.nn.functional.scaled_dot_product_attention``;
large-D forward continues to use the FFPA Triton kernel by default, with a
legacy optional CUDA forward backend available only when compiled in.

When the caller opts into the CuTeDSL backend (``forward_backend='cutedsl'``)
``ffpa_attn_func`` routes straight to
:func:`ffpa_attn.cutedsl._wrappers._ffpa_attn_cutedsl` (the dense Layer-2
entry, sibling of the varlen one below), which wraps
``FFPAAttnSplitDFunc.apply(...)`` and skips ``FFPAAttnMeta`` normalization
and the ``FFPAAttnFunc`` autograd boundary. CuTeDSL compatibility is split
in two:

- **Hard (tensor-level)**: ``head_dim != 512``, non-SM90 device, wrong
  dtype, fp16 training all raise ``NotImplementedError`` / ``TypeError``
  from :func:`ffpa_attn.cutedsl._wrappers._require_cutedsl_supported`
  inside the dense entry.

- **Soft (kwarg-level)**: enforced by
  :func:`ffpa_attn.cutedsl._wrappers._check_supported_options` at each
  entry shim. The dense ``ffpa_attn_func(forward_backend='cutedsl')``
  path only forwards ``dropout_p`` and ``attn_mask`` to the helper;
  other unsupported kwargs reach the dense path via ``**kwargs`` and
  are rejected later by :meth:`FFPAAttnMeta.from_kwargs`'s unknown-key
  ``TypeError``. The varlen ``ffpa_attn_varlen_func`` path additionally
  forwards ``window_size``, ``softcap``, ``sink``, ``block_mask``,
  ``score_mod``, ``aux_tensors``, ``seqused_k``, ``block_table``,
  ``num_splits``, ``alibi_slopes`` to the helper. Any non-default
  value raises ``NotImplementedError`` with a single consolidated
  message naming every offending option — no silent strip-to-default.
  Use ``forward_backend='triton'`` when these options are required.

For the dense entry, the two pure hardware mismatches —
``head_dim != 512`` and a non-SM90 device — fall back to SDPA with a
``warning_once`` log on the ``FFPA.ffpa_attn.ffpa_attn_interface``
logger, since neither is fixable at the call site. Every other
constraint (dtype, fp16 training, ``dropout_p > 0``, explicit
``attn_mask``, all FA-extension kwargs above, and the entire varlen
path) continues to raise ``NotImplementedError`` / ``TypeError`` /
``ValueError``; there is no silent fallback for those.

Variable-length (packed THD) attention is exposed via ``ffpa_attn_varlen_func``,
which mirrors the FlashAttention varlen surface (``q, k, v, cu_seqlens_q,
cu_seqlens_k, max_seqlen_q, max_seqlen_k, ...``) and delegates to
:func:`ffpa_attn.cutedsl._wrappers._ffpa_attn_varlen_cutedsl`, which
dispatches to the CuTeDSL ``ffpa_attn::splitd_fwd_sm90`` autograd-registered
torch op. The varlen API is currently CuTeDSL-only (SM90, D=512); other
shapes / backends raise ``NotImplementedError``.
"""

from __future__ import annotations

import torch
from .functional import FFPAAttnFunc, FFPAAttnMeta
from .logger import init_logger

logger = init_logger(__name__)


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
  * ``dropout_p > 0.0`` when the large-D forward backend cannot support it
  * ``attn_mask is not None`` when the large-D forward backend cannot support it
  * ``8 <= Nq < 512``
  * ``Nk < 512``
  * ``forward_backend == 'cutedsl'`` and ``head_dim != 512`` (hardware
    mismatch — cutedsl is specialised to D=512). Emits a ``warning_once``
    before returning ``True``.
  * ``forward_backend == 'cutedsl'`` and the device is not SM90 (Hopper).
    Emits a ``warning_once`` before returning ``True``. Other cutedsl
    constraints (dtype, fp16 training, ``dropout_p > 0``, ``attn_mask``,
    FA-extension kwargs) continue to raise from the cutedsl wrappers
    rather than fall back here.

  As FFPA grows support for these cases, remove the corresponding condition
  here instead of scattering dispatch checks throughout ``ffpa_attn_func``.
  """
  assert query.dim() == 4, "Expected query shape [B, Nh_q, Nq, D]"
  assert key.dim() == 4, "Expected key shape [B, Nh_kv, Nkv, D]"
  B, Nh_q, Nq, D = query.shape  # noqa: F841
  _, Nh_kv, Nkv, D_k = key.shape
  assert D == D_k, "Query and key must have the same head dimension"
  # cutedsl is opt-in: the only fallback we apply is the pure hardware
  # mismatch (head_dim != 512 or non-SM90), with a one-shot warning. All
  # other cutedsl constraints (dtype, fp16 training, dropout_p > 0, explicit
  # attn_mask, FA-extension kwargs) must keep raising from
  # _require_cutedsl_supported / _check_supported_options, so cutedsl
  # bypasses the legacy any([...]) heuristics below.
  if forward_backend == "cutedsl":
    from .cutedsl._wrappers import cutedsl_forward_available
    cutedsl_hw_unsupported = D != 512 or not cutedsl_forward_available(query.device)
    if cutedsl_hw_unsupported:
      logger.warning_once(
        "forward_backend='cutedsl' falling back to SDPA: head_dim=%d, device=%s "
        "(cutedsl requires head_dim=512 on SM90 Hopper).",
        D,
        query.device,
      )
    return cutedsl_hw_unsupported

  return any([
    D <= 256,
    D > 1024,
    # attn_mask and dropout only supported in triton backend for now.
    attn_mask is not None and forward_backend == "cutedsl",
    dropout_p > 0.0 and forward_backend == "cutedsl",
    (8 <= Nq < 512),
    Nkv < 512,
  ])


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
  ``D > 1024``, and unsupported large-D dropout), and otherwise keeps the
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
      ``forward_backend='cutedsl'`` rejects any non-``None`` ``attn_mask``
      with ``NotImplementedError`` (no silent fallback).
    :param dropout_p: Dropout probability. Large-D CUDA and Triton implement
      SDPA-style attention dropout; unsupported backends route to SDPA, except
      that ``forward_backend='cutedsl'`` raises ``NotImplementedError`` for
      ``dropout_p > 0`` instead of falling back.
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
      ``triton_backward_preprocess_d_chunk``,
      ``triton_backward_enable_persist_dkdv``, and
      ``triton_backward_grad_kv_storage_dtype``. ``forward_backend`` only affects ``D > 256``.
      ``enable_forward_tma`` and ``enable_backward_tma`` independently opt
      into the SM90+ Triton descriptor/TMA forward and backward paths when
      supported. ``enable_forward_ws`` and ``enable_backward_ws`` request
      warp-specialized TMA configs for the matching direction. ``enable_tma``
      and ``enable_ws`` are compatibility aliases that set both directions.
      ``triton_backward_enable_persist_dkdv`` enables an experimental SM90 TMA
      backward path that keeps dK/dV accumulators in fp32 registers across Q
      blocks and requires ``enable_backward_tma=True``.
      ``backward_backend`` supports ``"triton"`` and ``"sdpa"``.
      ``triton_backward_grad_kv_storage_dtype`` defaults to ``None`` and
      currently accepts ``torch.float16`` or ``torch.float32`` as overrides for Triton
      backward's internal ``DK`` / ``DV`` storage dtype. These options do not change
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

  # CuTeDSL backend — opt-in via forward_backend='cutedsl'.
  # Bypasses FFPAAttnMeta normalization and the FFPAAttnFunc autograd
  # boundary, dispatching directly to ffpa_attn_splitd_func (autograd via
  # FFPAAttnSplitDFunc). _should_fallback_to_sdpa above returns False for
  # the cutedsl branch by construction; unsupported cases raise inside
  # _ffpa_attn_cutedsl: tensor-level (head_dim != 512, dtype, non-SM90
  # device) via _require_cutedsl_supported, kwarg-level (dropout_p > 0,
  # attn_mask is not None) via _check_supported_options. The lazy import
  # keeps the cutedsl package off the hot path for non-cutedsl callers.
  if meta.forward_backend == "cutedsl":
    from .cutedsl import _ffpa_attn_cutedsl
    return _ffpa_attn_cutedsl(
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
  :param kwargs: Most kwargs are recognized-and-rejected by
      :func:`ffpa_attn.cutedsl._wrappers._check_supported_options` — passing a
      non-default value for ``window_size``, ``softcap``, ``sink``,
      ``attention_mask`` / ``attn_mask``, ``block_mask``, ``score_mod``,
      ``aux_tensors``, ``seqused_k``, ``block_table``, ``num_splits``, or
      ``alibi_slopes`` raises ``NotImplementedError`` (see ``:raises:``).
      Only ``forward_backend`` / ``backward_backend`` are forwarded to
      :meth:`FFPAAttnMeta.from_kwargs` for pair-binding; both must be
      ``"cutedsl"`` or unset.

  :returns: ``out`` of shape ``[T_q, H_q, D]`` if ``return_lse=False``,
      otherwise ``(out, lse)``.

  :raises NotImplementedError: for ``D != 512``, non-SM90 hardware,
      ``dropout_p > 0``, non-CuTeDSL ``forward_backend``, or any non-default
      unsupported kwarg: ``window_size``, ``softcap``, ``sink``,
      ``attention_mask`` / ``attn_mask``, ``block_mask``, ``score_mod``,
      ``aux_tensors``, ``seqused_k``, ``block_table``, ``num_splits``,
      ``alibi_slopes``.
  :raises TypeError: if ``cu_seqlens_*`` is not int32, or if dtype is not
      fp16/bf16.
  :raises ValueError: for shape mismatches between ``q``/``k``/``v``,
      malformed ``cu_seqlens_*``, or ``enable_gqa=False`` with
      ``H_q != H_kv``.
  """
  from .cutedsl import _ffpa_attn_varlen_cutedsl
  return _ffpa_attn_varlen_cutedsl(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=dropout_p,
    softmax_scale=softmax_scale,
    causal=causal,
    enable_gqa=enable_gqa,
    return_lse=return_lse,
    kwargs=kwargs,
  )


__all__ = ["ffpa_attn_func", "ffpa_attn_varlen_func"]

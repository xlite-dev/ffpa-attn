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

from dataclasses import dataclass
import math
import warnings

import torch

from .cuda import _ffpa_attn_forward_cuda
from .cuda import _ffpa_attn_backward_cuda
from .triton import _ffpa_attn_forward_triton
from .triton import _ffpa_attn_backward_triton

# The SM90 TMA large-d kernel only widens the K box to 64 fp16 cols
# (SWIZZLE_128B) when the head dim satisfies these constraints; outside
# this set the C++ ``ExperimentalTmaLargeDConfig::kCanAttempt`` predicate
# is false and the SM90 TMA kernel template is never instantiated, so
# requesting ``tma=True`` for an unsupported head dim cannot dispatch to
# a TMA kernel anyway. Keep this check in sync with that predicate.
_TMA_MIN_HEADDIM = 128
_TMA_HEADDIM_ALIGN = 64

# acc encoding kept in sync with csrc/pybind/ffpa_attn_api.cc::ffpa_attn.
_ACC_F16 = 0
_ACC_F32 = 1

_FFPA_ATTN_IMPL_DEFAULTS: dict[str, object] = {
  "stages": 2,
  "acc": "f32",
  "tma": False,
  "high_precision_grad": False,
  "forward_backend": "cuda",
  "triton_forward_autotune": False,
  "backward_backend": "triton",
  "triton_backward_autotune": False,
  "triton_backward_version": "v2",
  "triton_backward_preprocess_d_chunk": False,
}


def _aten_flash_attn_forward(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  o: torch.Tensor | None,
  causal: bool,
  softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Run the small-D path through the exact aten flash-attention forward op."""

  q_bnhd = q.transpose(1, 2).contiguous()
  k_bnhd = k.transpose(1, 2).contiguous()
  v_bnhd = v.transpose(1, 2).contiguous()
  out_bnhd, lse, rng_state, unused, _ = torch.ops.aten._flash_attention_forward(
    q_bnhd,
    k_bnhd,
    v_bnhd,
    None,
    None,
    q.size(2),
    k.size(2),
    0.0,
    causal,
    False,
    scale=softmax_scale,
  )
  out = out_bnhd.transpose(1, 2).contiguous()
  if o is None:
    return out, lse, rng_state, unused

  o.copy_(out)
  return o, lse, rng_state, unused


def _aten_flash_attn_backward(
  grad_out: torch.Tensor,
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  o: torch.Tensor,
  lse: torch.Tensor,
  causal: bool,
  rng_state: torch.Tensor,
  unused: torch.Tensor,
  softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Run the small-D path through PyTorch's flash-attention backward wrapper."""

  return torch.ops.aten._scaled_dot_product_flash_attention_backward.default(
    grad_out.contiguous(),
    q.contiguous(),
    k.contiguous(),
    v.contiguous(),
    o.contiguous(),
    lse.contiguous(),
    None,
    None,
    q.size(2),
    k.size(2),
    0.0,
    causal,
    rng_state.contiguous(),
    unused.contiguous(),
    scale=softmax_scale,
  )


@dataclass(frozen=True)
class FFPAAttnMeta:
  """Non-tensor FFPA options passed through the autograd Function.

  :param causal: Whether to apply lower-right causal masking.
  :param softmax_scale: Scale applied to ``QK^T``.
  :param stages: CUDA forward pipeline stages.
  :param acc: Native CUDA accumulator code.
  :param tma: Whether to request the CUDA TMA path.
  :param is_grad_enabled: Grad-mode state captured at the public API.
  :param high_precision_grad: Whether SDPA backward should upcast.
  :param forward_backend: Forward backend name, ``"cuda"`` or ``"triton"``.
  :param triton_forward_autotune: Whether to enable Triton forward autotune.
  :param backward_backend: Backward backend name. ``"sdpa"``, ``"cuda"``, or ``"triton"``.
  :param triton_backward_autotune: Whether to enable Triton backward autotune.
  :param triton_backward_version: Triton backward kernel version.
  :param triton_backward_preprocess_d_chunk: Whether Triton backward should
    compute delta with the split-D preprocess kernel.
  """

  causal: bool
  softmax_scale: float
  stages: int
  acc: int
  tma: int
  is_grad_enabled: bool
  high_precision_grad: bool
  forward_backend: str
  triton_forward_autotune: bool
  backward_backend: str
  triton_backward_autotune: bool
  triton_backward_version: str
  triton_backward_preprocess_d_chunk: bool


# ---------------------------------------------------------------------------
# Autograd Function
# ---------------------------------------------------------------------------
# Small-D uses PyTorch's flash-attention forward/backward pair; large-D uses
# FFPA forward plus the existing FFPA / SDPA backward routing. When any input
# requires gradients and grad mode is enabled, the selected path saves the
# intermediates its matching backward consumes.
# ---------------------------------------------------------------------------


class FFPAAttnFunc(torch.autograd.Function):
  """FFPA attention with autograd support.

    Forward routes by headdim. ``D <= 256`` uses PyTorch's flash-attention
    forward/backward pair. ``D > 256`` continues to use the FFPA CUDA or
    Triton kernels. When any input requires gradients and grad mode is
    enabled, the intermediate tensors needed by the selected backward path
    are saved on the context.

    Dropout is not supported (always 0.0).
  """

  @staticmethod
  def forward(
    ctx,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor | None,
    meta: FFPAAttnMeta,
  ) -> torch.Tensor:
    is_grad = meta.is_grad_enabled and any(x.requires_grad for x in [q, k, v])
    head_dim = q.size(-1)

    if head_dim <= 256:
      O, lse, rng_state, unused = _aten_flash_attn_forward(
        q,
        k,
        v,
        o,
        bool(meta.causal),
        meta.softmax_scale,
      )
    elif meta.forward_backend == "cuda":
      O, lse = _ffpa_attn_forward_cuda(
        q,
        k,
        v,
        o,
        meta.stages,
        meta.acc,
        int(bool(meta.causal)),
        meta.softmax_scale,
        meta.tma,
      )
    elif meta.forward_backend == "triton":
      O, lse = _ffpa_attn_forward_triton(
        q,
        k,
        v,
        o,
        bool(meta.causal),
        meta.softmax_scale,
        meta.triton_forward_autotune,
      )
    else:
      raise ValueError(f"Unsupported forward_backend={meta.forward_backend!r};")

    if head_dim > 256:
      rng_state = torch.empty(0, dtype=torch.uint8, device=q.device)
      unused = torch.empty(0, dtype=torch.uint8, device=q.device)

    if is_grad:
      ctx.save_for_backward(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        O.contiguous(),
        lse,
        rng_state,
        unused,
      )
      ctx.meta = meta

    return O

  @staticmethod
  def backward(ctx, grad_out: torch.Tensor):
    q, k, v, O, lse, rng_state, unused = ctx.saved_tensors
    meta = ctx.meta
    D = q.size(-1)
    group_size = q.size(1) // k.size(1)

    zero_u64 = torch.zeros(2, dtype=torch.uint64, device=q.device)
    philox_seed = zero_u64[0].unsqueeze(0)
    philox_offset = zero_u64[1].unsqueeze(0)

    if D > 256:
      if meta.backward_backend == "cuda":
        dq, dk, dv = _ffpa_attn_backward_cuda(
          q.contiguous(),
          k.contiguous(),
          v.contiguous(),
          O.contiguous(),
          lse.contiguous(),
          grad_out.contiguous(),
          meta.stages,
          int(bool(meta.causal)),
          meta.softmax_scale,
        )
      elif meta.backward_backend == "triton":
        # Pad LSE to seqlen_q_rounded (Triton kernel needs padded stride
        # for safe masked loads).
        seqlen_q = q.size(2)
        seqlen_q_rounded = ((seqlen_q + 127) // 128) * 128
        if lse.size(-1) < seqlen_q_rounded:
          lse_padded = torch.empty(
            *lse.shape[:-1],
            seqlen_q_rounded,
            dtype=torch.float32,
            device=lse.device,
          )
          lse_padded[..., :lse.size(-1)] = lse
          lse = lse_padded

        if group_size > 1:
          k_in = k.repeat_interleave(group_size, dim=1).contiguous()
          v_in = v.repeat_interleave(group_size, dim=1).contiguous()
        else:
          k_in, v_in = k, v

        # Allocate gradient buffers.
        dq = torch.empty_like(q)
        dk_gqa = torch.empty_like(k_in)
        dv_gqa = torch.empty_like(v_in)

        _ffpa_attn_backward_triton(
          do=grad_out.contiguous(),
          q=q.contiguous(),
          k=k_in.contiguous(),
          v=v_in.contiguous(),
          o=O.contiguous(),
          lse=lse,
          dq=dq,
          dk=dk_gqa,
          dv=dv_gqa,
          causal=meta.causal,
          softmax_scale=meta.softmax_scale,
          autotune=meta.triton_backward_autotune,
          kernel_version=meta.triton_backward_version,
          preprocess_d_chunk=meta.triton_backward_preprocess_d_chunk,
        )

        if group_size > 1:
          dk = dk_gqa.reshape(
            k.size(0),
            k.size(1),
            group_size,
            k.size(2),
            k.size(3),
          ).sum(dim=2).to(k.dtype)
          dv = dv_gqa.reshape(
            v.size(0),
            v.size(1),
            group_size,
            v.size(2),
            v.size(3),
          ).sum(dim=2).to(v.dtype)
        else:
          dk, dv = dk_gqa.to(k.dtype), dv_gqa.to(v.dtype)
        dq = dq.to(q.dtype)
      else:
        # ---- SDPA delegation (original path) ----
        # The CUTLASS kernel inside efficient_attention_backward
        # expects O in the stride layout produced by SDPA forward:
        # a BNHD→BHND transposed view (stride H*D == D, stride N*D
        # == H*D).  FFPA produces contiguous BHND O which, after the
        # internal transpose(1,2), yields non-standard strides and
        # triggers illegal memory accesses.  We reshape FFPA's O to
        # match SDPA's stride pattern. The mem-efficient backward also
        # requires lse.stride(1) % 8 == 0 when num_heads > 1, so pad the
        # sequence dimension for odd tail lengths before calling into it.
        O = O.transpose(1, 2).contiguous().transpose(1, 2)  # noqa: E741
        if lse.size(1) > 1 and (lse.stride(1) % 8) != 0:
          seqlen_q_aligned = ((lse.size(-1) + 7) // 8) * 8
          lse_padded = torch.empty(
            *lse.shape[:-1],
            seqlen_q_aligned,
            dtype=lse.dtype,
            device=lse.device,
          )
          lse_padded[..., :lse.size(-1)] = lse
          lse = lse_padded

        if meta.high_precision_grad:
          _q = q.float()
          _k = k.float()
          _v = v.float()
          _O = O.float()
          _lse = lse.float()
          _grad_out = grad_out.float()
        else:
          _q, _k, _v = q, k, v
          _O, _lse, _grad_out = O, lse, grad_out

        if group_size > 1:
          k_in = _k.repeat_interleave(group_size, dim=1).contiguous()
          v_in = _v.repeat_interleave(group_size, dim=1).contiguous()
        else:
          k_in, v_in = _k, _v

        dq, dk_e, dv_e, _ = \
            torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(
                _grad_out, _q, k_in, v_in, None,
                _O, _lse,
                philox_seed, philox_offset,
                0.0,
                (True, True, True, False),
                meta.causal,
                scale=meta.softmax_scale,
            )
        if group_size > 1:
          dk = dk_e.reshape(
            k.size(0),
            k.size(1),
            group_size,
            k.size(2),
            k.size(3),
          ).sum(dim=2).to(k.dtype)
          dv = dv_e.reshape(
            v.size(0),
            v.size(1),
            group_size,
            v.size(2),
            v.size(3),
          ).sum(dim=2).to(v.dtype)
        else:
          dk, dv = dk_e.to(k.dtype), dv_e.to(v.dtype)
        dq = dq.to(q.dtype)
    else:
      dq, dk, dv = _aten_flash_attn_backward(
        grad_out,
        q,
        k,
        v,
        O,
        lse,
        meta.causal,
        rng_state,
        unused,
        meta.softmax_scale,
      )

    # Gradients for: q, k, v, o, meta.
    return dq, dk, dv, None, None


def ffpa_attn_func(
  Q: torch.Tensor,
  K: torch.Tensor,
  V: torch.Tensor,
  O: torch.Tensor | None = None,
  causal: bool = False,
  softmax_scale: float | None = None,
  **kwargs: object,
) -> torch.Tensor:
  """Unified FFPA prefill attention entry.

  Dispatches by ``Q.dtype`` (fp16 / bf16) and ``acc`` through a single
  registered torch op (``torch.ops.ffpa_attn.attn``), keeping the
  Python layer minimal and fully compatible with ``torch.compile``.

  Supports cross-attention where ``Q`` seqlen (``Nq``) differs from ``K``/``V``
  seqlen (``Nkv``) and grouped-query attention where ``Q`` has more heads
  than ``K``/``V`` (MQA is the ``Nh_kv == 1`` special case). ``K`` and ``V``
  must share the same ``Nh_kv`` and the same ``Nkv``. Causal masking is
  supported via ``causal=True`` with queries aligned to the tail of the
  KV sequence (``Nkv >= Nq`` required).

  Backward pass is supported via :class:`FFPAAttnFunc`. For ``D <= 256`` it
  uses PyTorch's flash-attention forward/backward pair; for ``D > 256`` it
  keeps the existing FFPA forward plus SDPA/FFPA backward routing.
  ``forward_backend`` only affects the large-D path.

  :param Q: Query tensor with layout ``[B, Nh_q, Nq, D]``; dtype must be
      ``torch.float16`` or ``torch.bfloat16`` and match ``K`` / ``V`` / ``O``.
  :param K: Key tensor with layout ``[B, Nh_kv, Nkv, D]``; same dtype as
      ``Q``. ``Nh_q`` must be an integer multiple of ``Nh_kv``
      (``group_size = Nh_q / Nh_kv``).
  :param V: Value tensor with layout ``[B, Nh_kv, Nkv, D]``; same dtype as
      ``Q``. ``K`` and ``V`` must share the same ``Nh_kv`` and ``Nkv``.
  :param O: Optional output tensor with layout ``[B, Nh_q, Nq, D]`` and
      same dtype as ``Q``; allocated via ``torch.zeros_like(Q)`` when
      ``None``.
  :param causal: When ``True``, apply a causal attention mask so that
      query row ``r`` only attends to KV positions ``k <= r + (Nkv - Nq)``
      (standard ``queries aligned to KV tail`` convention). Requires
      ``Nkv >= Nq``. Non-causal tiles pay only one compare-and-branch
      per KV tile; diagonal tiles apply a per-fragment -inf mask.
  :param softmax_scale: Pre-softmax scaling factor applied to
      ``QK^T``. Defaults to ``1 / sqrt(D)`` (standard attention scale)
      when ``None``. Named ``softmax_scale`` to match the
      ``flash-attn`` convention.
  :param kwargs: Implementation-specific options for experimentation.  Supported
      keys are ``stages``, ``acc``, ``tma``, ``high_precision_grad``,
      ``forward_backend``, ``triton_forward_autotune``, ``backward_backend``,
      ``triton_backward_autotune``, ``triton_backward_version``, and
      ``triton_backward_preprocess_d_chunk``. ``forward_backend`` only affects
      ``D > 256``. These options do not change the autograd contract; unknown
      keys raise ``TypeError``.

  :returns: ``O``, filled with the attention output
      ``softmax(softmax_scale * QK^T) V``.

  :raises TypeError: if ``Q.dtype`` is neither fp16 nor bf16.
  :raises ValueError: if ``acc`` is not one of ``{'f16', 'f32'}``, if
      bf16 activations are combined with ``acc='f16'`` (no bf16-acc mma
      PTX instruction exists on supported architectures), or if
      ``causal=True`` is combined with ``Nkv < Nq``.
  """
  unknown_kwargs = sorted(set(kwargs) - set(_FFPA_ATTN_IMPL_DEFAULTS))
  if unknown_kwargs:
    unknown = ", ".join(unknown_kwargs)
    raise TypeError(f"ffpa_attn_func got unexpected keyword argument(s): {unknown}")

  impl_options = {**_FFPA_ATTN_IMPL_DEFAULTS, **kwargs}
  stages = int(impl_options["stages"])
  acc = impl_options["acc"]
  tma = bool(impl_options["tma"])
  high_precision_grad = bool(impl_options["high_precision_grad"])
  forward_backend = str(impl_options["forward_backend"])
  triton_forward_autotune = bool(impl_options["triton_forward_autotune"])
  backward_backend = str(impl_options["backward_backend"])
  triton_backward_autotune = bool(impl_options["triton_backward_autotune"])
  triton_backward_version = str(impl_options["triton_backward_version"])
  triton_backward_preprocess_d_chunk = bool(impl_options["triton_backward_preprocess_d_chunk"])

  assert forward_backend in ("cuda", "triton"), \
    f"Unsupported forward_backend={forward_backend!r}; choose 'cuda' or 'triton'."
  assert backward_backend in ("sdpa", "triton", "cuda"), \
    f"Unsupported backward_backend={backward_backend!r}; choose 'sdpa', 'triton', or 'cuda'."
  if acc == "f32":
    acc_code = _ACC_F32
  elif acc == "f16":
    acc_code = _ACC_F16
  else:
    raise ValueError(f"acc must be 'f16' or 'f32', got {acc!r}")

  if Q.dtype == torch.bfloat16 and acc_code == _ACC_F16:
    raise ValueError("bf16 activations require acc='f32'; no bf16-acc mma PTX exists.")
  if Q.dtype not in (torch.float16, torch.bfloat16):
    raise TypeError(f"ffpa_attn_func only supports fp16/bf16, got {Q.dtype}")

  if Q.dim() != 4 or K.dim() != 4 or V.dim() != 4:
    raise ValueError("Q/K/V must be 4-D [B, H, N, D] tensors")
  if Q.size(0) != K.size(0) or Q.size(0) != V.size(0):
    raise ValueError("Q/K/V must share the same batch size")
  if K.size(1) != V.size(1):
    raise ValueError(f"K and V must share the same num_heads, got Nh_k={K.size(1)}, Nh_v={V.size(1)}")
  if Q.size(1) % K.size(1) != 0:
    raise ValueError(
      f"Q num_heads must be an integer multiple of K/V num_heads (GQA/MQA), "
      f"got Nh_q={Q.size(1)}, Nh_kv={K.size(1)}"
    )
  if K.size(2) != V.size(2):
    raise ValueError(f"K and V must share the same seqlen, got Nk={K.size(2)}, Nv={V.size(2)}")
  if Q.size(3) != K.size(3) or Q.size(3) != V.size(3):
    raise ValueError("Q/K/V must share the same head dim")

  # Pre-check tma eligibility on the head-dim axis. The SM90 TMA kernel
  # only widens the K box to 64 fp16 cols when D >= 128 and D % 64 == 0;
  # outside that set the C++ template is not instantiated, so dispatch
  # would silently fall back. Coerce tma to False here so the user gets
  # a single explicit warning instead of confusing 'TMA enabled but no
  # speedup' behavior.
  if tma:
    head_dim = Q.size(3)
    if head_dim < _TMA_MIN_HEADDIM or (head_dim % _TMA_HEADDIM_ALIGN) != 0:
      warnings.warn(
        f"ffpa_attn_func: tma=True is only supported for head_dim >= {_TMA_MIN_HEADDIM} "
        f"and divisible by {_TMA_HEADDIM_ALIGN}, got head_dim={head_dim}; "
        f"falling back to the cp.async kernel.",
        RuntimeWarning,
        stacklevel=2,
      )
      tma = False

  if causal and K.size(2) < Q.size(2):
    raise ValueError(
      f"causal=True requires Nkv >= Nq (queries are aligned to the KV tail), "
      f"got Nq={Q.size(2)}, Nkv={K.size(2)}"
    )

  if O is None:
    O = torch.zeros_like(Q)  # noqa: E741

  if softmax_scale is None:
    softmax_scale = 1.0 / math.sqrt(Q.size(-1))

  _meta = FFPAAttnMeta(
    causal=bool(causal),
    softmax_scale=float(softmax_scale),
    stages=int(stages),
    acc=acc_code,
    tma=int(bool(tma)),
    is_grad_enabled=torch.is_grad_enabled(),
    high_precision_grad=bool(high_precision_grad),
    forward_backend=str(forward_backend),
    triton_forward_autotune=bool(triton_forward_autotune),
    backward_backend=str(backward_backend),
    triton_backward_autotune=bool(triton_backward_autotune),
    triton_backward_version=str(triton_backward_version),
    triton_backward_preprocess_d_chunk=bool(triton_backward_preprocess_d_chunk),
  )

  return FFPAAttnFunc.apply(
    Q,
    K,
    V,
    O,
    _meta,
  )

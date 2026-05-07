"""Autograd Function and metadata for FFPA attention.

Houses the ``FFPAAttnMeta`` dataclass and ``FFPAAttnFunc`` autograd Function
that routes forward/backward across the CUDA, Triton, and aten flash-attention
backends. Imported by ``ffpa_attn_interface.py`` and other callers that need
to access the low-level dispatch layer.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .cuda import _ffpa_attn_forward_cuda, _ffpa_attn_backward_cuda  # D > 256
from .triton import _ffpa_attn_forward_triton, _ffpa_attn_backward_triton  # D > 256
from .aten import _aten_flash_attn_forward, _aten_flash_attn_backward, _aten_efficient_attn_backward  # D <= 256

if TYPE_CHECKING:
  from typing import Tuple, Union, Optional  # noqa: F401

# The SM>=90 TMA large-d kernel only widens the K box to 64 fp16 cols
# (SWIZZLE_128B) when the head dim satisfies these constraints; outside
# this set the C++ ``ExperimentalTmaLargeDConfig::kCanAttempt`` predicate
# is false and the SM90 TMA kernel template is never instantiated, so
# requesting ``enable_tma=True`` for an unsupported head dim cannot dispatch to
# a TMA kernel anyway. Keep this check in sync with that predicate.
_TMA_MIN_HEADDIM = 128
_TMA_HEADDIM_ALIGN = 64

# MMA Acc encoding kept in sync with csrc/pybind/ffpa_attn_api.cc::ffpa_attn.
_ACC_F16 = 0
_ACC_F32 = 1

_FFPA_ATTN_IMPL_DEFAULTS: dict[str, object] = {
  "stages": 2,
  "acc": "f32",
  "enable_tma": False,
  "high_precision_grad": False,
  "forward_backend": "cuda",
  "triton_forward_autotune": False,
  "backward_backend": "triton",
  "triton_backward_autotune": False,
  "triton_backward_version": "v2",
  "triton_backward_preprocess_d_chunk": False,
}


@dataclass
class FFPAAttnMeta:
  """Non-tensor FFPA options passed through the autograd Function.

  :param is_causal: Whether to apply lower-right causal masking.
  :param scale: Scale applied to ``QK^T``.
  :param stages: CUDA forward pipeline stages.
  :param acc: Native CUDA accumulator code.
  :param enable_tma: Whether to request the CUDA TMA path.
  :param dropout_p: Dropout probability (default 0.0).
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

  is_causal: bool
  scale: float
  stages: int
  acc: int
  enable_tma: int
  dropout_p: float
  is_grad_enabled: bool
  high_precision_grad: bool
  forward_backend: str
  triton_forward_autotune: bool
  backward_backend: str
  triton_backward_autotune: bool
  triton_backward_version: str
  triton_backward_preprocess_d_chunk: bool

  @classmethod
  def from_kwargs(cls, **kwargs: object) -> FFPAAttnMeta:
    """Build a meta from the impl-specific ``**kwargs``.

    Merges the given kwargs with :data:`_FFPA_ATTN_IMPL_DEFAULTS`, validates
    unknown keys, backends and ``acc``, and returns a fully populated meta
    instance.  User-facing fields (``is_causal``, ``scale``, ``dropout_p``,
    ``is_grad_enabled``) are set to safe defaults; call :meth:`normalize`
    to fill and validate them from the public-API inputs.
    """
    unknown = sorted(set(kwargs) - set(_FFPA_ATTN_IMPL_DEFAULTS))
    if unknown:
      keys = ", ".join(unknown)
      raise TypeError(f"ffpa_attn_func got unexpected keyword argument(s): {keys}")

    impl_options = {**_FFPA_ATTN_IMPL_DEFAULTS, **kwargs}

    stages = int(impl_options["stages"])
    acc_str = impl_options["acc"]
    enable_tma = int(bool(impl_options["enable_tma"]))
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

    if acc_str == "f32":
      acc = _ACC_F32
    elif acc_str == "f16":
      acc = _ACC_F16
    else:
      raise ValueError(f"acc must be 'f16' or 'f32', got {acc_str!r}")

    return cls(
      # NOTE: Some of these fields maybe updated later by normalize() based on
      # the public API inputs, but we need to set them to some value here to create
      # the instance.
      is_causal=False,
      scale=0.0,
      dropout_p=0.0,
      is_grad_enabled=False,
      acc=acc,
      stages=stages,
      enable_tma=enable_tma,
      high_precision_grad=high_precision_grad,
      forward_backend=forward_backend,
      triton_forward_autotune=triton_forward_autotune,
      backward_backend=backward_backend,
      triton_backward_autotune=triton_backward_autotune,
      triton_backward_version=triton_backward_version,
      triton_backward_preprocess_d_chunk=triton_backward_preprocess_d_chunk,
    )

  def normalize(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None,
    dropout_p: float,
    is_causal: bool,
    scale: float | None,
    enable_gqa: bool,
  ) -> FFPAAttnMeta:
    """Fill user-facing fields and validate all inputs in place.

    Call this right after :meth:`from_kwargs` to get a fully validated meta::

        meta = FFPAAttnMeta.from_kwargs(**kwargs).normalize(
            query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa,
        )

    Raises ``TypeError``, ``ValueError``, or ``NotImplementedError`` for
    invalid or unsupported combinations.
    """
    if attn_mask is not None:
      raise NotImplementedError(
        "ffpa_attn_func: attn_mask is not yet supported. "
        "Pass attn_mask=None (the default) to use the built-in causal mask via is_causal=True."
      )
    if dropout_p > 0.0 and query.size(-1) > 256:
      raise NotImplementedError(
        "ffpa_attn_func: dropout_p > 0 is only supported for head_dim <= 256 "
        "(the PyTorch flash-attention path). For head_dim > 256, pass dropout_p=0.0."
      )

    # Fill in user-facing fields.
    self.is_causal = is_causal
    self.dropout_p = float(dropout_p)
    self.is_grad_enabled = torch.is_grad_enabled()

    # Validate that acc-code is compatible with activation dtype.
    if query.dtype == torch.bfloat16 and self.acc == _ACC_F16:
      raise ValueError("bf16 activations require acc='f32'; no bf16-acc mma PTX exists.")
    if query.dtype not in (torch.float16, torch.bfloat16):
      raise TypeError(f"ffpa_attn_func only supports fp16/bf16, got {query.dtype}")

    # Validate tensor shapes.
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
      raise ValueError("query/key/value must be 4-D [B, H, N, D] tensors")
    if query.size(0) != key.size(0) or query.size(0) != value.size(0):
      raise ValueError("query/key/value must share the same batch size")
    if key.size(1) != value.size(1):
      raise ValueError(
        f"key and value must share the same num_heads, "
        f"got Nh_k={key.size(1)}, Nh_v={value.size(1)}"
      )
    if query.size(1) % key.size(1) != 0:
      raise ValueError(
        f"query num_heads must be an integer multiple of key/value num_heads (GQA/MQA), "
        f"got Nh_q={query.size(1)}, Nh_kv={key.size(1)}"
      )
    if key.size(2) != value.size(2):
      raise ValueError(f"key and value must share the same seqlen, got Nk={key.size(2)}, Nv={value.size(2)}")
    if query.size(3) != key.size(3) or query.size(3) != value.size(3):
      raise ValueError("query/key/value must share the same head dim")

    if not enable_gqa and query.size(1) != key.size(1):
      raise ValueError(
        f"enable_gqa=False but query num_heads ({query.size(1)}) != "
        f"key/value num_heads ({key.size(1)}). "
        f"Set enable_gqa=True or use matching head counts."
      )

    # TMA eligibility check — mutate self.enable_tma if needed.
    if self.enable_tma:
      head_dim = query.size(3)
      if head_dim < _TMA_MIN_HEADDIM or (head_dim % _TMA_HEADDIM_ALIGN) != 0:
        warnings.warn(
          f"ffpa_attn_func: enable_tma=True is only supported for "
          f"head_dim >= {_TMA_MIN_HEADDIM} and divisible by {_TMA_HEADDIM_ALIGN}, "
          f"got head_dim={head_dim}; falling back to the cp.async kernel.",
          RuntimeWarning,
          stacklevel=3,
        )
        self.enable_tma = 0

    if is_causal and key.size(2) < query.size(2):
      raise ValueError(
        f"is_causal=True requires Nkv >= Nq (queries are aligned to the KV tail), "
        f"got Nq={query.size(2)}, Nkv={key.size(2)}"
      )

    if scale is None:
      self.scale = 1.0 / math.sqrt(query.size(-1))
    else:
      self.scale = float(scale)

    return self


class FFPAAttnFunc(torch.autograd.Function):
  """FFPA attention with autograd support.

    Forward routes by headdim. ``D <= 256`` uses PyTorch's flash-attention
    forward/backward pair. ``D > 256`` continues to use the FFPA CUDA or
    Triton kernels. When any input requires gradients and grad mode is
    enabled, the intermediate tensors needed by the selected backward path
    are saved on the context.

    Backward is intentionally dispatch-only: backend-specific tensor
    preparation and result restoration live in the backend wrappers under
    ``ffpa_attn.aten`` / ``ffpa_attn.triton`` / ``ffpa_attn.cuda`` rather than
    inside :meth:`backward` itself.

    Dropout is not supported for D > 256 now (always 0.0).
  """

  @staticmethod
  def forward(
    ctx,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    meta: FFPAAttnMeta,
  ) -> torch.Tensor:
    is_grad = meta.is_grad_enabled and any(x.requires_grad for x in [q, k, v])
    head_dim = q.size(-1)
    O = torch.empty_like(q)  # noqa: E741

    if head_dim <= 256:
      O, lse, rng_state, unused = _aten_flash_attn_forward(
        q,
        k,
        v,
        O,
        meta.is_causal,
        meta.scale,
        meta.dropout_p,
      )
    elif meta.forward_backend == "cuda":
      O, lse = _ffpa_attn_forward_cuda(
        q,
        k,
        v,
        O,
        meta.stages,
        meta.acc,
        int(meta.is_causal),
        meta.scale,
        meta.enable_tma,
      )
    elif meta.forward_backend == "triton":
      O, lse = _ffpa_attn_forward_triton(
        q,
        k,
        v,
        O,
        meta.is_causal,
        meta.scale,
        meta.triton_forward_autotune,
      )
    else:
      raise ValueError(f"Unsupported forward_backend={meta.forward_backend!r};")

    # NO rng_state or unused output from the FFPA CUDA forward / backward kernels
    # due to no dropout, but we need to return something to keep the autograd contract
    # consistent across backends. Return empty tensors on large-D paths since the
    # small-D path's backward expects tensors to be returned and saved.
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
    if TYPE_CHECKING:
      q: torch.Tensor
      k: torch.Tensor
      v: torch.Tensor
      O: torch.Tensor
      lse: torch.Tensor
      rng_state: torch.Tensor
      unused: torch.Tensor
      dq: torch.Tensor
      dk: torch.Tensor
      dv: torch.Tensor

    q, k, v, O, lse, rng_state, unused = ctx.saved_tensors
    meta: FFPAAttnMeta = ctx.meta
    D = q.size(-1)

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
          int(meta.is_causal),
          meta.scale,
        )
      elif meta.backward_backend == "triton":
        dq, dk, dv = _ffpa_attn_backward_triton(
          grad_out=grad_out,
          q=q,
          k=k,
          v=v,
          o=O,
          lse=lse,
          causal=meta.is_causal,
          softmax_scale=meta.scale,
          autotune=meta.triton_backward_autotune,
          kernel_version=meta.triton_backward_version,
          preprocess_d_chunk=meta.triton_backward_preprocess_d_chunk,
        )
      else:
        dq, dk, dv = _aten_efficient_attn_backward(
          grad_out=grad_out,
          q=q,
          k=k,
          v=v,
          o=O,
          lse=lse,
          causal=meta.is_causal,
          softmax_scale=meta.scale,
          high_precision_grad=meta.high_precision_grad,
        )
    else:
      # Aten flash-attention backward for D <= 256, which also supports dropout gradients
      # (currently always 0.0 since dropout is not supported).
      dq, dk, dv = _aten_flash_attn_backward(
        grad_out,
        q,
        k,
        v,
        O,
        lse,
        meta.is_causal,
        rng_state,
        unused,
        meta.scale,
        meta.dropout_p,
      )

    # Gradients for: q, k, v, o, meta.
    return dq, dk, dv, None, None


__all__ = ["FFPAAttnMeta", "FFPAAttnFunc"]

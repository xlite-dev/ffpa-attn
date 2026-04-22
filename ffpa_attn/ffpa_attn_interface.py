"""Torch custom op registration for FFPA prefill attention.

Wraps the single pybind entry ``ffpa_attn._C.ffpa_attn`` as a real
``torch.library`` operator so callers (including ``torch.compile`` graphs)
can reach the kernel through ``torch.ops.ffpa_attn.attn`` instead of
calling the C-extension symbol directly.
"""

from __future__ import annotations

import math
import warnings

import torch

from ._C import ffpa_attn as _ffpa_attn_cuda

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

_OP_NAMESPACE = "ffpa_attn"
_OP_NAME = "attn"
_OP_QUALNAME = f"{_OP_NAMESPACE}::{_OP_NAME}"

# The op mutates ``O`` in place and returns it for convenience. The
# ``(a!)`` alias annotation tells torch.library the buffer is written,
# which is required for correct alias/functionalization behavior under
# torch.compile.
torch.library.define(
  _OP_QUALNAME,
  "(Tensor Q, Tensor K, Tensor V, Tensor(a!) O, int stages, int acc, int causal, "
  "float softmax_scale, int tma) -> Tensor(a!)",
)


@torch.library.impl(_OP_QUALNAME, "CUDA")
def _ffpa_attn_impl_cuda(
  Q: torch.Tensor,
  K: torch.Tensor,
  V: torch.Tensor,
  O: torch.Tensor,
  stages: int,
  acc: int,
  causal: int,
  softmax_scale: float,
  tma: int,
) -> torch.Tensor:
  _ffpa_attn_cuda(Q, K, V, O, stages, acc, causal, softmax_scale, tma)
  return O


@torch.library.register_fake(_OP_QUALNAME)
def _ffpa_attn_impl_fake(
  Q: torch.Tensor,
  K: torch.Tensor,
  V: torch.Tensor,
  O: torch.Tensor,
  stages: int,
  acc: int,
  causal: int,
  softmax_scale: float,
  tma: int,
) -> torch.Tensor:
  del Q, K, V, stages, acc, causal, softmax_scale, tma
  return O


def ffpa_attn_func(
  Q: torch.Tensor,
  K: torch.Tensor,
  V: torch.Tensor,
  O: torch.Tensor | None = None,
  causal: bool = False,
  softmax_scale: float | None = None,
  stages: int = 2,
  acc: str = "f32",
  tma: bool = False,
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
  :param stages: Pipeline stages forwarded to the underlying CUDA kernel.
  :param acc: MMA accumulator dtype. ``'f32'`` selects the fp32-acc kernel
      (required for bf16 activations); ``'f16'`` selects the fp16-acc
      kernel and is only valid for fp16 activations.
  :param tma: When ``True``, opt in to the experimental SM90+ TMA path
      for the large-d kernel. Defaults to ``False`` because the TMA path
      is not currently faster than the cp.async fallback on any
      architecture (it is kept as an opt-in for experimentation and as a
      foundation for future warp-specialised producer/consumer work).
      Honored only when:

        * the head dim satisfies ``D >= 128 and D % 64 == 0`` (otherwise
          the K TMA box cannot be widened to 64 fp16 cols and the SM90
          TMA kernel template is not instantiated); requesting ``tma=True``
          on an unsupported head dim emits a warning and silently falls
          back to the cp.async kernel,
        * the runtime device has compute capability >= 9.0,
        * the environment variable ``ENABLE_FFPA_EXPERIMENTAL_TMA=1`` is
          set,
        * the C++ compile-time eligibility check
          (``ExperimentalTmaLargeDConfig::kCanAttempt``: zero K/V
          padding, multi-stage support) passes.

      When any of these is not met, the launcher transparently uses the
      architecture-agnostic ``cp.async``-based kernel.

  :returns: ``O``, filled with the attention output
      ``softmax(softmax_scale * QK^T) V``.

  :raises TypeError: if ``Q.dtype`` is neither fp16 nor bf16.
  :raises ValueError: if ``acc`` is not one of ``{'f16', 'f32'}``, if
      bf16 activations are combined with ``acc='f16'`` (no bf16-acc mma
      PTX instruction exists on supported architectures), or if
      ``causal=True`` is combined with ``Nkv < Nq``.
  """
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

  return torch.ops.ffpa_attn.attn(
    Q, K, V, O, int(stages), acc_code, int(bool(causal)), float(softmax_scale), int(bool(tma))
  )

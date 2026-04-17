"""Torch custom op registration for FFPA prefill attention.

Wraps the single pybind entry ``ffpa_attn._C.ffpa_attn`` as a real
``torch.library`` operator so callers (including ``torch.compile`` graphs)
can reach the kernel through ``torch.ops.ffpa_attn.attn`` instead of
calling the C-extension symbol directly.
"""

from __future__ import annotations

import torch

from ._C import ffpa_attn as _ffpa_attn_cuda

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
  "(Tensor Q, Tensor K, Tensor V, Tensor(a!) O, int stages, int acc) -> Tensor(a!)",
)


@torch.library.impl(_OP_QUALNAME, "CUDA")
def _ffpa_attn_impl_cuda(
  Q: torch.Tensor,
  K: torch.Tensor,
  V: torch.Tensor,
  O: torch.Tensor,
  stages: int,
  acc: int,
) -> torch.Tensor:
  _ffpa_attn_cuda(Q, K, V, O, stages, acc)
  return O


@torch.library.register_fake(_OP_QUALNAME)
def _ffpa_attn_impl_fake(
  Q: torch.Tensor,
  K: torch.Tensor,
  V: torch.Tensor,
  O: torch.Tensor,
  stages: int,
  acc: int,
) -> torch.Tensor:
  del Q, K, V, stages, acc
  return O


def ffpa_attn_func(
  Q: torch.Tensor,
  K: torch.Tensor,
  V: torch.Tensor,
  O: torch.Tensor | None = None,
  stages: int = 2,
  acc: str = "f32",
) -> torch.Tensor:
  """Unified FFPA prefill attention entry.

  Dispatches by ``Q.dtype`` (fp16 / bf16) and ``acc`` through a single
  registered torch op (``torch.ops.ffpa_attn.attn``), keeping the
  Python layer minimal and fully compatible with ``torch.compile``.

  :param Q: Query tensor with layout ``[B, H, N, D]``; dtype must be
      ``torch.float16`` or ``torch.bfloat16`` and match ``K`` / ``V`` / ``O``.
  :param K: Key tensor with the same layout and dtype as ``Q``.
  :param V: Value tensor with the same layout and dtype as ``Q``.
  :param O: Optional output tensor with the same layout and dtype as
      ``Q``; allocated via ``torch.zeros_like(Q)`` when ``None``.
  :param stages: Pipeline stages forwarded to the underlying CUDA kernel.
  :param acc: MMA accumulator dtype. ``'f32'`` selects the fp32-acc kernel
      (required for bf16 activations); ``'f16'`` selects the fp16-acc
      kernel and is only valid for fp16 activations.

  :returns: ``O``, filled with the attention output
      ``softmax(QK^T / sqrt(D)) V``.

  :raises TypeError: if ``Q.dtype`` is neither fp16 nor bf16.
  :raises ValueError: if ``acc`` is not one of ``{'f16', 'f32'}``, or if
      bf16 activations are combined with ``acc='f16'`` (no bf16-acc mma
      PTX instruction exists on supported architectures).
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

  if O is None:
    O = torch.zeros_like(Q)  # noqa: E741

  return torch.ops.ffpa_attn.attn(Q, K, V, O, int(stages), acc_code)

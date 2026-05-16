"""Layout-adapting FFPA wrappers around the CuTeDSL D=512 SM90 kernels.

The CuTeDSL kernels in :mod:`ffpa_attn.cutedsl.interface` operate on the
``[B, N, H, D]`` (or packed ``[T, H, D]``) layout used by Dao-AILab
flash-attention. The FFPA dispatch pipeline in
:mod:`ffpa_attn.functional` operates on the SDPA-style ``[B, H, N, D]``
layout. This module bridges the two with a minimal transpose-and-call shim
and centralises the SM90 / D=512 / dtype / unsupported-feature gating that
:class:`ffpa_attn.functional.FFPAAttnMeta` invokes via
:func:`_require_cutedsl_supported`.
"""

from __future__ import annotations

from typing import Optional

import torch

from .interface import (
  SUPPORTED_HEAD_DIM,
  _flash_attn_bwd_sm90,
  _flash_attn_fwd_sm90,
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
  is_causal: bool,
  dropout_p: float,
  attn_bias: Optional[torch.Tensor],
  enable_gqa_user: bool,
  requires_grad: bool,
) -> None:
  """Validate that the FFPA inputs are compatible with the CuTeDSL backend.

  Raises ``NotImplementedError`` / ``RuntimeError`` / ``TypeError`` for any
  case the SM90 D=512 kernel cannot handle. Called explicitly by
  :meth:`FFPAAttnMeta.normalize` so that users who pass
  ``forward_backend='cutedsl'`` get an actionable error instead of a silent
  fallback or a deep kernel crash.
  """
  del is_causal, enable_gqa_user  # both natively supported by the CuTeDSL kernel
  if q.device.type != "cuda":
    raise RuntimeError(f"cutedsl backend requires CUDA tensors, got device {q.device}")
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
  if dropout_p > 0.0:
    raise NotImplementedError("cutedsl backend does not support dropout (dropout_p must be 0.0)")
  if attn_bias is not None:
    raise NotImplementedError("cutedsl backend does not support attn_mask / additive attn_bias")
  if k.size(-1) != SUPPORTED_HEAD_DIM or v.size(-1) != SUPPORTED_HEAD_DIM:
    raise NotImplementedError(
      f"cutedsl backend requires k/v head_dim={SUPPORTED_HEAD_DIM}; "
      f"got k={k.size(-1)} v={v.size(-1)}"
    )


def _bhnd_to_bnhd(t: torch.Tensor) -> torch.Tensor:
  """Reshape ``[B, H, N, D]`` to the CuTeDSL-native ``[B, N, H, D]``."""
  return t.transpose(1, 2).contiguous()


def _ffpa_attn_forward_cutedsl(
  Q: torch.Tensor,
  K: torch.Tensor,
  V: torch.Tensor,
  O: Optional[torch.Tensor] = None,
  causal: bool = False,
  softmax_scale: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
  """CuTeDSL FFPA forward dispatch shim.

  :param Q: ``[B, Nh_q, Nq, D=512]`` query tensor.
  :param K: ``[B, Nh_kv, Nkv, D=512]`` key tensor.
  :param V: ``[B, Nh_kv, Nkv, D=512]`` value tensor.
  :param O: Ignored. Output is allocated fresh by the CuTeDSL kernel.
  :param causal: Lower-right (tail-aligned) causal mask flag.
  :param softmax_scale: Pre-softmax scale applied to ``QK^T``.
  :returns: ``(O, lse)`` in FFPA layout — ``O`` is ``[B, Nh_q, Nq, D]`` and
    ``lse`` is ``[B, Nh_q, Nq]`` in fp32.
  """
  del O  # CuTeDSL allocates output internally; FFPA layout differs anyway.
  q_nhd, k_nhd, v_nhd = (_bhnd_to_bnhd(t) for t in (Q, K, V))
  out_nhd, lse = _flash_attn_fwd_sm90(
    q_nhd,
    k_nhd,
    v_nhd,
    softmax_scale=softmax_scale,
    causal=causal,
    return_lse=True,
  )
  # out_nhd: [B, Nq, Nh_q, D]; bring back to FFPA's [B, Nh_q, Nq, D].
  out_bhnd = out_nhd.transpose(1, 2).contiguous()
  return out_bhnd, lse


def _ffpa_attn_backward_cutedsl(
  *,
  grad_out: torch.Tensor,
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  o: torch.Tensor,
  lse: torch.Tensor,
  causal: bool,
  softmax_scale: float,
  attn_bias: Optional[torch.Tensor] = None,
  return_attn_bias_grad: bool = False,
  dropout_p: float = 0.0,
  philox_seed: int = 0,
  philox_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
  """CuTeDSL FFPA backward dispatch shim.

  Mirrors the keyword surface of :func:`_ffpa_attn_backward_triton` so the
  ``FFPAAttnFunc.backward`` dispatcher can swap backends without per-backend
  branching for the call site. Any keyword that CuTeDSL cannot honor must be
  at its default value, otherwise we raise instead of silently dropping it.
  """
  del philox_seed, philox_offset  # not used: dropout_p must be 0 here
  if attn_bias is not None or return_attn_bias_grad:
    raise NotImplementedError("cutedsl backward does not support attn_bias gradients")
  if dropout_p != 0.0:
    raise NotImplementedError("cutedsl backward does not support dropout")

  q_nhd, k_nhd, v_nhd, o_nhd, dout_nhd = (_bhnd_to_bnhd(t) for t in (q, k, v, o, grad_out))
  dq_nhd, dk_nhd, dv_nhd = _flash_attn_bwd_sm90(
    q_nhd,
    k_nhd,
    v_nhd,
    o_nhd,
    dout_nhd,
    lse,
    softmax_scale=softmax_scale,
    causal=causal,
  )
  dq, dk, dv = (t.transpose(1, 2).contiguous() for t in (dq_nhd, dk_nhd, dv_nhd))
  return dq, dk, dv, None


__all__ = [
  "_ffpa_attn_forward_cutedsl",
  "_ffpa_attn_backward_cutedsl",
  "cutedsl_forward_available",
  "cutedsl_backward_available",
  "_require_cutedsl_supported",
]

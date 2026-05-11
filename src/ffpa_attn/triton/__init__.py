"""Triton FFPA attention forward/backward implementations for large-D
(D > 256, but also works for D <= 256).
"""
import torch

from ._ffpa_fwd import _ffpa_attn_forward_triton
from ._ffpa_bwd import _ffpa_attn_backward_triton

_OP_NAMESPACE = "ffpa_attn"


def _attn_bias_grad_needs_reduction(attn_bias: torch.Tensor | None, q: torch.Tensor, k: torch.Tensor) -> bool:
  """Return whether compact bias gradients are reduced across broadcast dimensions."""
  if attn_bias is None:
    return False
  return any([
    attn_bias.size(0) == 1 and q.size(0) > 1,
    attn_bias.size(1) == 1 and q.size(1) > 1,
    attn_bias.size(2) == 1 and q.size(2) > 1,
    attn_bias.size(3) == 1 and k.size(2) > 1,
  ])


def _attn_bias_grad_dtype(attn_bias: torch.Tensor, q: torch.Tensor, k: torch.Tensor) -> torch.dtype:
  """Return the internal accumulation dtype for Triton bias gradients.

  BF16 additive-bias gradients are numerically sensitive even when the logical
  mask shape does not require broadcast reduction. Keep the Triton accumulation
  buffer in fp32 and cast back to the user dtype at the Python wrapper
  boundary so the kernel only rounds once.
  """
  if attn_bias.dtype == torch.bfloat16 or _attn_bias_grad_needs_reduction(attn_bias, q, k):
    return torch.float32
  return attn_bias.dtype


def _triton_bwd_grad_tensor_like(tensor: torch.Tensor) -> torch.Tensor:
  """Allocate the internal Triton backward output buffer for one gradient.

  Keep the Triton-owned backward gradient workspace in fp32 and cast back in
  the higher-level Python wrapper once all kernel-side accumulation is
  finished. This makes fp32 accumulation the default for both fp16 and bf16
  backward kernels.
  """
  return torch.empty_like(tensor, dtype=torch.float32)


torch.library.define(
  f"{_OP_NAMESPACE}::_fwd_triton",
  "(Tensor q, Tensor k, Tensor v, Tensor? attn_bias, float softmax_scale, "
  "int causal, int autotune, int autotune_mode_is_max, float dropout_p, int philox_seed, int philox_offset) "
  "-> (Tensor o, Tensor softmax_lse)",
)


@torch.library.impl(f"{_OP_NAMESPACE}::_fwd_triton", "CUDA")
def _fwd_triton_torch_op(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  attn_bias: torch.Tensor | None,
  softmax_scale: float,
  causal: int,
  autotune: int,
  autotune_mode_is_max: int,
  dropout_p: float,
  philox_seed: int,
  philox_offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  from ._ffpa_fwd import _ffpa_attn_forward_impl as _triton_fwd_kernel

  if q.stride(-1) != 1:
    q = q.contiguous()
  if k.stride(-1) != 1:
    k = k.contiguous()
  if v.stride(-1) != 1:
    v = v.contiguous()

  o = torch.empty_like(q)
  seqlen_q = q.size(2)
  seqlen_q_aligned = ((seqlen_q + 127) // 128) * 128
  softmax_lse = torch.empty(
    q.size(0),
    q.size(1),
    seqlen_q_aligned,
    dtype=torch.float32,
    device=q.device,
  )
  _triton_fwd_kernel(
    q,
    k,
    v,
    o,
    softmax_lse,
    attn_bias=attn_bias,
    causal=bool(causal),
    softmax_scale=softmax_scale,
    autotune=bool(autotune),
    autotune_mode="max" if autotune_mode_is_max else "fast",
    dropout_p=dropout_p,
    philox_seed=philox_seed,
    philox_offset=philox_offset,
  )
  return o, softmax_lse


@torch.library.register_fake(f"{_OP_NAMESPACE}::_fwd_triton")
def _fwd_triton_fake(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  attn_bias: torch.Tensor | None,
  softmax_scale: float,
  causal: int,
  autotune: int,
  autotune_mode_is_max: int,
  dropout_p: float,
  philox_seed: int,
  philox_offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  seqlen_q_aligned = ((q.size(2) + 127) // 128) * 128
  o = torch.empty_like(q)
  softmax_lse = q.new_empty(q.size(0), q.size(1), seqlen_q_aligned, dtype=torch.float32)
  return o, softmax_lse


torch.library.define(
  f"{_OP_NAMESPACE}::_bwd_triton",
  "(Tensor dO, Tensor q, Tensor k, Tensor v, Tensor o, Tensor lse, Tensor? attn_bias, "
  "float softmax_scale, int causal, int autotune, "
  "int autotune_mode_is_max, int preprocess_d_chunk, int return_attn_bias_grad, "
  "float dropout_p, int philox_seed, int philox_offset) "
  "-> (Tensor dq, Tensor dk, Tensor dv, Tensor grad_attn_bias)",
)


@torch.library.impl(f"{_OP_NAMESPACE}::_bwd_triton", "CUDA")
def _bwd_triton_torch_op(
  do: torch.Tensor,
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  o: torch.Tensor,
  lse: torch.Tensor,
  attn_bias: torch.Tensor | None,
  softmax_scale: float,
  causal: int,
  autotune: int,
  autotune_mode_is_max: int,
  preprocess_d_chunk: int,
  return_attn_bias_grad: int,
  dropout_p: float,
  philox_seed: int,
  philox_offset: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  from ._ffpa_bwd import _ffpa_attn_backward_triton_impl as _triton_bwd_kernel

  dq = _triton_bwd_grad_tensor_like(q)
  dk = _triton_bwd_grad_tensor_like(k)
  dv = _triton_bwd_grad_tensor_like(v)
  if attn_bias is not None and return_attn_bias_grad:
    grad_dtype = _attn_bias_grad_dtype(attn_bias, q, k)
    grad_attn_bias = torch.empty_like(attn_bias, dtype=grad_dtype)
  else:
    grad_attn_bias = q.new_empty(0)
  if grad_attn_bias.numel() > 0:
    grad_attn_bias.zero_()

  _triton_bwd_kernel(
    do=do,
    q=q,
    k=k,
    v=v,
    o=o,
    lse=lse,
    attn_bias=attn_bias,
    dq=dq,
    dk=dk,
    dv=dv,
    grad_attn_bias=grad_attn_bias if grad_attn_bias.numel() > 0 else None,
    causal=bool(causal),
    softmax_scale=softmax_scale,
    autotune=bool(autotune),
    autotune_mode="max" if autotune_mode_is_max else "fast",
    preprocess_d_chunk=bool(preprocess_d_chunk),
    dropout_p=dropout_p,
    philox_seed=philox_seed,
    philox_offset=philox_offset,
  )
  return dq, dk, dv, grad_attn_bias


@torch.library.register_fake(f"{_OP_NAMESPACE}::_bwd_triton")
def _bwd_triton_fake(
  do: torch.Tensor,
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  o: torch.Tensor,
  lse: torch.Tensor,
  attn_bias: torch.Tensor | None,
  softmax_scale: float,
  causal: int,
  autotune: int,
  autotune_mode_is_max: int,
  preprocess_d_chunk: int,
  return_attn_bias_grad: int,
  dropout_p: float,
  philox_seed: int,
  philox_offset: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  del (
    softmax_scale,
    causal,
    autotune,
    autotune_mode_is_max,
    preprocess_d_chunk,
    dropout_p,
    philox_seed,
    philox_offset,
  )
  if attn_bias is not None and return_attn_bias_grad:
    grad_dtype = _attn_bias_grad_dtype(attn_bias, q, k)
    grad_attn_bias = torch.empty_like(attn_bias, dtype=grad_dtype)
  else:
    grad_attn_bias = q.new_empty(0)
  return (
    _triton_bwd_grad_tensor_like(q),
    _triton_bwd_grad_tensor_like(k),
    _triton_bwd_grad_tensor_like(v),
    grad_attn_bias,
  )


__all__ = ["_ffpa_attn_forward_triton", "_ffpa_attn_backward_triton"]

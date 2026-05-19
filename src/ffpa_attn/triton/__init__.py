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


def _triton_bwd_grad_tensor_like(
  tensor: torch.Tensor,
  grad_storage_dtype: torch.dtype | None = None,
) -> torch.Tensor:
  """Allocate the internal Triton backward output buffer for one gradient.

  This allocation also determines the dtype used by the Triton kernel's global
  ``tl.load`` / ``tl.store`` traffic on ``DQ`` / ``DK`` / ``DV``. For example,
  when ``_triton_bwd_grad_tensor_like(k)`` returns bf16 storage, the backward
  kernel's ``tl.load(dk_ptrs)`` / ``tl.store(dk_ptrs, ...)`` sites operate on
  bf16 values. Returning fp16/fp32 here therefore changes the kernel's cross-tile
  global accumulation dtype, not just the Python-visible output tensor dtype.

  Memory note:
  The fp32 override is expensive for large tensors, and the Triton wrapper
  passes already-expanded K/V tensors for GQA/MQA. One fp32 buffer costs

  ``tensor.numel() * 4`` bytes,

  so a typical large-D self-attention or causal shape
  ``B=1, Hq=32, Nq=Nkv=8192, D=512`` allocates

  ``1 * 32 * 8192 * 512 * 4 = 536870912`` bytes per buffer = ``512 MiB``,

  Because K/V storage follows the expanded query-head layout, this fp32 cost
  also applies to GQA/MQA after head expansion, even if the original KV tensors
  had fewer heads.

  Recommendation:
  keep fp32 storage targeted to gradients that need higher cross-tile
  accumulation precision. For the current Triton backward path this is dK/dV.

  :param tensor: Reference tensor that provides shape, device, and default
    dtype.
  :param grad_storage_dtype: Optional storage dtype for this internal gradient
    buffer. ``None`` keeps the user-visible activation dtype; ``torch.float16``
    or ``torch.float32`` overrides cross-tile global accumulation storage.
  :return: Newly allocated gradient buffer.
  """
  if grad_storage_dtype is None:
    return torch.empty_like(tensor)
  return torch.empty_like(tensor, dtype=grad_storage_dtype)


def _grad_kv_storage_dtype_from_code(code: int) -> torch.dtype | None:
  """Decode the internal dK/dV storage dtype selector."""
  if code == 0:
    return None
  if code == 1:
    return torch.float32
  if code == 2:
    return torch.float16
  raise ValueError(f"Unsupported grad_kv_storage_dtype code {code}; expected 0, 1, or 2.")


torch.library.define(
  f"{_OP_NAMESPACE}::_fwd_triton",
  "(Tensor q, Tensor k, Tensor v, Tensor? attn_bias, float softmax_scale, "
  "int causal, int autotune, int autotune_mode_is_max, float dropout_p, int philox_seed, int philox_offset, "
  "int enable_tma, int enable_ws) "
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
  enable_tma: int,
  enable_ws: int,
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
    enable_tma=bool(enable_tma),
    enable_ws=bool(enable_ws),
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
  enable_tma: int,
  enable_ws: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  seqlen_q_aligned = ((q.size(2) + 127) // 128) * 128
  o = torch.empty_like(q)
  softmax_lse = q.new_empty(q.size(0), q.size(1), seqlen_q_aligned, dtype=torch.float32)
  return o, softmax_lse


torch.library.define(
  f"{_OP_NAMESPACE}::_bwd_triton",
  "(Tensor dO, Tensor q, Tensor k, Tensor v, Tensor o, Tensor lse, Tensor? attn_bias, "
  "float softmax_scale, int causal, int autotune, "
  "int autotune_mode_is_max, int preprocess_d_chunk, int return_attn_bias_grad, int grad_kv_storage_dtype_code, "
  "int original_nheads_kv, float dropout_p, int philox_seed, int philox_offset, int enable_tma, int enable_ws, "
  "int enable_persist_dkdv) "
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
  grad_kv_storage_dtype_code: int,
  original_nheads_kv: int,
  dropout_p: float,
  philox_seed: int,
  philox_offset: int,
  enable_tma: int,
  enable_ws: int,
  enable_persist_dkdv: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  from ._ffpa_bwd import _ffpa_attn_backward_triton_impl as _triton_bwd_kernel

  grad_kv_storage_dtype = _grad_kv_storage_dtype_from_code(grad_kv_storage_dtype_code)
  dq = _triton_bwd_grad_tensor_like(q)
  dk = _triton_bwd_grad_tensor_like(k, grad_kv_storage_dtype)
  dv = _triton_bwd_grad_tensor_like(v, grad_kv_storage_dtype)
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
    original_nheads_kv=original_nheads_kv,
    dropout_p=dropout_p,
    philox_seed=philox_seed,
    philox_offset=philox_offset,
    enable_tma=bool(enable_tma),
    enable_ws=bool(enable_ws),
    enable_persist_dkdv=bool(enable_persist_dkdv),
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
  grad_kv_storage_dtype_code: int,
  original_nheads_kv: int,
  dropout_p: float,
  philox_seed: int,
  philox_offset: int,
  enable_tma: int,
  enable_ws: int,
  enable_persist_dkdv: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  del (
    softmax_scale,
    causal,
    autotune,
    autotune_mode_is_max,
    preprocess_d_chunk,
    original_nheads_kv,
    dropout_p,
    philox_seed,
    philox_offset,
    enable_tma,
    enable_ws,
    enable_persist_dkdv,
  )
  grad_kv_storage_dtype = _grad_kv_storage_dtype_from_code(grad_kv_storage_dtype_code)
  if attn_bias is not None and return_attn_bias_grad:
    grad_dtype = _attn_bias_grad_dtype(attn_bias, q, k)
    grad_attn_bias = torch.empty_like(attn_bias, dtype=grad_dtype)
  else:
    grad_attn_bias = q.new_empty(0)
  return (
    _triton_bwd_grad_tensor_like(q),
    _triton_bwd_grad_tensor_like(k, grad_kv_storage_dtype),
    _triton_bwd_grad_tensor_like(v, grad_kv_storage_dtype),
    grad_attn_bias,
  )


__all__ = ["_ffpa_attn_forward_triton", "_ffpa_attn_backward_triton"]

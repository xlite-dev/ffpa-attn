"""Triton FFPA attention forward/backward implementations for large-D
(D > 256, but also works for D <= 256).
"""
import torch

from ._ffpa_fwd import _ffpa_attn_forward_triton
from ._ffpa_bwd import _ffpa_attn_backward_triton

_OP_NAMESPACE = "ffpa_attn"

# ---------------------------------------------------------------------------
# _fwd_triton  --  ffpa_attn::_fwd_triton
# ---------------------------------------------------------------------------
torch.library.define(
  f"{_OP_NAMESPACE}::_fwd_triton",
  "(Tensor q, Tensor k, Tensor v, float softmax_scale, "
  "int causal, int autotune) -> (Tensor o, Tensor softmax_lse)",
)


@torch.library.impl(f"{_OP_NAMESPACE}::_fwd_triton", "CUDA")
def _fwd_triton_torch_op(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  softmax_scale: float,
  causal: int,
  autotune: int,
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
    causal=bool(causal),
    softmax_scale=softmax_scale,
    autotune=bool(autotune),
  )
  return o, softmax_lse


@torch.library.register_fake(f"{_OP_NAMESPACE}::_fwd_triton")
def _fwd_triton_fake(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  softmax_scale: float,
  causal: int,
  autotune: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  seqlen_q_aligned = ((q.size(2) + 127) // 128) * 128
  o = torch.empty_like(q)
  softmax_lse = q.new_empty(q.size(0), q.size(1), seqlen_q_aligned, dtype=torch.float32)
  return o, softmax_lse


# ---------------------------------------------------------------------------
# _bwd_triton  --  ffpa_attn::_bwd_triton
# ---------------------------------------------------------------------------
torch.library.define(
  f"{_OP_NAMESPACE}::_bwd_triton",
  "(Tensor dO, Tensor q, Tensor k, Tensor v, Tensor o, Tensor lse, "
  "float softmax_scale, int causal, int autotune, "
  "int kernel_version_is_v2, int preprocess_d_chunk) "
  "-> (Tensor dq, Tensor dk, Tensor dv)",
)


@torch.library.impl(f"{_OP_NAMESPACE}::_bwd_triton", "CUDA")
def _bwd_triton_torch_op(
  do: torch.Tensor,
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  o: torch.Tensor,
  lse: torch.Tensor,
  softmax_scale: float,
  causal: int,
  autotune: int,
  kernel_version_is_v2: int,
  preprocess_d_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  from ._ffpa_bwd import _ffpa_attn_backward_triton_impl as _triton_bwd_kernel

  dq = torch.empty_like(q)
  dk = torch.empty_like(k)
  dv = torch.empty_like(v)

  _triton_bwd_kernel(
    do=do,
    q=q,
    k=k,
    v=v,
    o=o,
    lse=lse,
    dq=dq,
    dk=dk,
    dv=dv,
    causal=bool(causal),
    softmax_scale=softmax_scale,
    autotune=bool(autotune),
    kernel_version="v2" if kernel_version_is_v2 else "v1",
    preprocess_d_chunk=bool(preprocess_d_chunk),
  )
  return dq, dk, dv


@torch.library.register_fake(f"{_OP_NAMESPACE}::_bwd_triton")
def _bwd_triton_fake(
  do: torch.Tensor,
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  o: torch.Tensor,
  lse: torch.Tensor,
  softmax_scale: float,
  causal: int,
  autotune: int,
  kernel_version_is_v2: int,
  preprocess_d_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


__all__ = ["_ffpa_attn_forward_triton", "_ffpa_attn_backward_triton"]

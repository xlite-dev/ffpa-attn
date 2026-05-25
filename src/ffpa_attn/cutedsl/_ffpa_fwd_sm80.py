"""FFPA CuTeDSL forward pass scaffold for SM80/SM89 Split-D kernels.

This module owns the SM80/SM89 orchestration surface. The public signature is
kept aligned with :func:`ffpa_attn.cutedsl._ffpa_fwd_sm90._ffpa_attn_forward_sm90`
so the existing CuTeDSL custom ops can dispatch by architecture without adding
new torch op schemas.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import torch
import cutlass.cute as cute

from ._fwd_generic_sm80 import FFPAAttnFwdSm80SplitD
from ._utils import (
  SM80_FWD_TILE_M,
  SM80_FWD_TILE_N,
  SM80_FWD_NUM_STAGES,
  SM80_FWD_NUM_THREADS,
  SM80_FWD_SPLIT_D_CHUNK,
  is_fake_mode,
  maybe_contiguous,
  _call_with_tvm_ffi_current_stream,
  _pick_split_d_chunk,
  _resolve_causal_local_window,
  _unsupported_training_features,
  _validate_max_seqlen_for_cu_seqlens,
  _validate_qkv_common,
  _validate_sm80_arch,
  _validate_sm80_head_dims,
  _validate_tensor,
  _validate_training_dtype,
  torch2cute_dtype_map,
)
from .utils.cache_utils import get_jit_cache
from .utils import fa_logging
from .utils.cute_dsl_utils import to_cute_tensor


def _ffpa_attn_forward_sm80(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  cu_seqlens_q: Optional[torch.Tensor] = None,
  cu_seqlens_k: Optional[torch.Tensor] = None,
  max_seqlen_q: Optional[int] = None,
  max_seqlen_k: Optional[int] = None,
  softmax_scale: Optional[float] = None,
  causal: bool = False,
  softcap: Optional[float] = None,
  window_size_left: Optional[int] = None,
  window_size_right: Optional[int] = None,
  pack_gqa: Optional[bool] = None,
  score_mod: Optional[Callable] = None,
  mask_mod: Optional[Callable] = None,
  return_lse: bool = False,
  out: Optional[torch.Tensor] = None,
  lse: Optional[torch.Tensor] = None,
  aux_tensors: Optional[list[torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
  """SM80/SM89 Split-D forward launcher.

  :param q: Query tensor in dense ``[B, N, H, D]`` or packed ``[T, H, D]`` layout.
  :param k: Key tensor with the same head dimension as ``q``.
  :param v: Value tensor with the same head dimension as ``q``.
  :param cu_seqlens_q: Optional packed-query sequence offsets.
  :param cu_seqlens_k: Optional packed-key sequence offsets.
  :param max_seqlen_q: Maximum query sequence length for varlen inputs.
  :param max_seqlen_k: Maximum key sequence length for varlen inputs.
  :param softmax_scale: Attention scale. Resolved by the caller when ``None``.
  :param causal: Whether lower-right causal masking is applied.
  :param softcap: Unsupported for the SM80 Split-D path.
  :param window_size_left: Unsupported local-attention left window.
  :param window_size_right: Unsupported local-attention right window.
  :param pack_gqa: Whether query heads are packed for GQA/MQA.
  :param score_mod: Unsupported score modifier.
  :param mask_mod: Unsupported mask modifier.
  :param return_lse: Whether the caller needs LSE returned.
  :param out: Optional preallocated output tensor.
  :param lse: Optional preallocated LSE tensor.
  :param aux_tensors: Unsupported auxiliary tensors.
  :returns: ``(out, lse)`` matching the SM90 launcher contract.
  """
  # This SM80 forward launcher handles varlen natively (no per-segment
  # Python fallback) because the kernel itself is built on the same
  # TileScheduler + SeqlenInfoQK abstraction as the SM90 forward path.
  # The SM80 backward kernels (dK/dV, dQ) lack this infrastructure and
  # therefore rely on per-segment dense calls in their launcher.
  q, k, v = [maybe_contiguous(t) for t in (q, k, v)]
  (
    batch_size,
    seqlen_q,
    total_q,
    seqlen_k,
    num_head,
    num_head_kv,
    head_dim,
    head_dim_v,
  ) = _validate_qkv_common(
    q,
    k,
    v,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_k=cu_seqlens_k,
    validate_head_dims=_validate_sm80_head_dims,
  )

  requires_grad = q.requires_grad or k.requires_grad or v.requires_grad
  _validate_training_dtype(q, k, v, requires_grad)
  _validate_max_seqlen_for_cu_seqlens(
    cu_seqlens_q, "cu_seqlens_q", max_seqlen_q, "max_seqlen_q"
  )
  _validate_max_seqlen_for_cu_seqlens(
    cu_seqlens_k, "cu_seqlens_k", max_seqlen_k, "max_seqlen_k"
  )

  device_arch, cute_arch_key = _validate_sm80_arch()
  if softmax_scale is None:
    softmax_scale = 1.0 / math.sqrt(head_dim)
  if softcap == 0.0:
    softcap = None

  causal, local, window_size_left, window_size_right = _resolve_causal_local_window(
    causal, window_size_left, window_size_right, mask_mod
  )
  _unsupported_training_features(
    requires_grad, softcap, local, score_mod, mask_mod, aux_tensors
  )
  unsupported = []
  if softcap is not None:
    unsupported.append("softcap")
  if local:
    unsupported.append("local/window attention")
  if score_mod is not None:
    unsupported.append("score_mod")
  if mask_mod is not None:
    unsupported.append("mask_mod")
  if aux_tensors is not None:
    unsupported.append("aux_tensors")
  if unsupported:
    raise NotImplementedError(
      "SM80/SM89 CuTeDSL forward currently supports dense/varlen attention "
      f"with optional causal masking only; unsupported options: {', '.join(unsupported)}."
    )

  qhead_per_kvhead = num_head // num_head_kv
  if pack_gqa:
    raise NotImplementedError(
      "SM80/SM89 CuTeDSL forward does not use the SM90 pack_gqa layout; "
      "leave pack_gqa unset and use the built-in query-head to KV-head mapping."
    )

  device = q.device
  out_torch_dtype = q.dtype
  q_batch_seqlen_shape = (batch_size,
                          seqlen_q) if cu_seqlens_q is None else (total_q, )
  lse_shape = (batch_size, num_head,
               seqlen_q) if cu_seqlens_q is None else (num_head, total_q)

  if out is None:
    out = torch.empty(
      *q_batch_seqlen_shape,
      num_head,
      head_dim_v,
      dtype=out_torch_dtype,
      device=device,
    )
  else:
    _validate_tensor(
      out, "out", (*q_batch_seqlen_shape, num_head, head_dim_v),
      out_torch_dtype, device
    )

  if lse is None:
    lse = torch.empty(
      lse_shape, dtype=torch.float32, device=device
    ) if requires_grad or return_lse else None
  else:
    _validate_tensor(lse, "lse", lse_shape, torch.float32, device)

  dtype = torch2cute_dtype_map[q.dtype]
  current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
  is_varlen = cu_seqlens_q is not None or cu_seqlens_k is not None
  smem_capacity_arch = f"sm_{device_arch // 10}{device_arch % 10}"
  fwd_d_chunk = _pick_split_d_chunk(
    FFPAAttnFwdSm80SplitD.can_implement,
    SM80_FWD_SPLIT_D_CHUNK,
    dtype=dtype,
    head_dim=head_dim,
    head_dim_v=head_dim_v,
    tile_m=SM80_FWD_TILE_M,
    tile_n=SM80_FWD_TILE_N,
    num_stages=SM80_FWD_NUM_STAGES,
    num_threads=SM80_FWD_NUM_THREADS,
    is_causal=causal,
    smem_capacity_arch=smem_capacity_arch,
  )
  if not FFPAAttnFwdSm80SplitD.can_implement(
    dtype,
    head_dim,
    head_dim_v,
    SM80_FWD_TILE_M,
    SM80_FWD_TILE_N,
    SM80_FWD_NUM_STAGES,
    SM80_FWD_NUM_THREADS,
    causal,
    smem_capacity_arch=smem_capacity_arch,
    d_chunk=fwd_d_chunk,
  ):
    raise RuntimeError(
      "SM80/SM89 CuTeDSL forward configuration exceeds kernel resource limits: "
      f"head_dim={head_dim}, tile=({SM80_FWD_TILE_M}, {SM80_FWD_TILE_N}), "
      f"num_stages={SM80_FWD_NUM_STAGES}, arch={smem_capacity_arch}, "
      f"d_chunk={fwd_d_chunk}."
    )

  if (is_varlen or causal) and not is_fake_mode():
    out.zero_()
    if lse is not None:
      lse.fill_(-float("inf"))

  if total_q == 0 or seqlen_k == 0:
    if not is_fake_mode():
      out.zero_()
      if lse is not None:
        lse.fill_(-float("inf"))
    return out, lse

  compile_key = (
    "sm80_fwd_generic",
    dtype,
    head_dim,
    head_dim_v,
    qhead_per_kvhead,
    causal,
    SM80_FWD_NUM_STAGES,
    lse is None,
    cu_seqlens_q is None,
    cu_seqlens_k is None,
    SM80_FWD_TILE_M,
    SM80_FWD_TILE_N,
    device_arch,
    cute_arch_key,
    fwd_d_chunk,
    fa_logging.get_fa_log_level(),
  )
  if compile_key not in _ffpa_attn_forward_sm80.compile_cache:
    cu_seqlens_q_tensor, cu_seqlens_k_tensor = [
      to_cute_tensor(t, assumed_align=4, leading_dim=0)
      if t is not None else None for t in (cu_seqlens_q, cu_seqlens_k)
    ]
    q_tensor, k_tensor, v_tensor, o_tensor = [
      to_cute_tensor(t) for t in (q, k, v, out)
    ]
    lse_tensor = to_cute_tensor(
      lse, assumed_align=4
    ) if lse is not None else None

    ffpa_fwd = FFPAAttnFwdSm80SplitD(
      dtype,
      head_dim,
      head_dim_v,
      qhead_per_kvhead,
      is_causal=causal,
      pack_gqa=False,
      tile_m=SM80_FWD_TILE_M,
      tile_n=SM80_FWD_TILE_N,
      num_stages=SM80_FWD_NUM_STAGES,
      d_chunk=fwd_d_chunk,
    )
    compile_args = [
      ffpa_fwd,
      q_tensor,
      k_tensor,
      v_tensor,
      o_tensor,
      lse_tensor,
      softmax_scale,
      cu_seqlens_q_tensor,
      cu_seqlens_k_tensor,
      current_stream,
    ]
    _ffpa_attn_forward_sm80.compile_cache[compile_key] = cute.compile(
      *compile_args,
      options=(
        "--enable-tvm-ffi --ptxas-options '--verbose --warn-on-spills --warn-on-local-memory-usage'"
      ),
    )

  if not is_fake_mode():
    _call_with_tvm_ffi_current_stream(
      _ffpa_attn_forward_sm80.compile_cache[compile_key],
      q.detach(),
      k.detach(),
      v.detach(),
      out.detach(),
      lse,
      softmax_scale,
      cu_seqlens_q,
      cu_seqlens_k,
      device=device,
    )
  return out, lse


_ffpa_attn_forward_sm80.compile_cache = get_jit_cache("fwd_sm80")

__all__ = ["_ffpa_attn_forward_sm80"]

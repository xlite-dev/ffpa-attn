"""SM100 D512 forward wrapper; support contract in cute/README.md."""

import math
from typing import Callable, Optional, Tuple

import torch
import cutlass.cute as cute

from ._utils import (
  SM100_D512_HEAD_DIM,
  is_fake_mode,
  torch2cute_dtype_map,
  _call_with_tvm_ffi_current_stream,
  _needs_dense_copy,
  _resolve_causal_local_window,
  _validate_max_seqlen_for_cu_seqlens,
  _validate_qkv_common,
  _validate_sm100_arch,
  _validate_tensor,
  _validate_training_dtype,
)
from ._fwd_d512_sm100 import FFPAAttnFwdSm100D512, compile_key_fields
from .utils import AuxData, fa_logging
from .utils.cache_utils import get_jit_cache
from .utils.cute_dsl_utils import to_cute_tensor
from .utils.fa_logging import fa_log

# Kill switch, read by _use_sm100_d512_specialized: it moves the forward *and*
# the backward to the SM80 fallback, so flipping it cannot split the pair.
SM100_D512_KERNEL_LIVE = True

# Diagnostics only (--warn-on-spills reports nothing here); patch() runs in bwd.
_PTXAS = (
  "--enable-tvm-ffi --ptxas-options "
  "'--verbose --warn-on-spills --warn-on-local-memory-usage'"
)


def _sm100_d512_unsupported_reason(
  *,
  cu_seqlens_q: Optional[torch.Tensor] = None,
  cu_seqlens_k: Optional[torch.Tensor] = None,
  local: bool = False,
  softcap: Optional[float] = None,
  score_mod: Optional[Callable] = None,
  mask_mod: Optional[Callable] = None,
  aux_tensors: Optional[list] = None,
) -> Optional[str]:
  """The frozen-contract rule this call breaks, named, or ``None``."""
  # Each prefix rebases a different origin; one alone skews the trip counts.
  if (cu_seqlens_q is None) != (cu_seqlens_k is None):
    return "varlen-one-sided"
  # Unsupported in the ported mainloop; the name says which rule fired.
  if local:
    return "local-window"
  if softcap:
    return "softcap"
  if score_mod is not None:
    return "score_mod"
  if mask_mod is not None:
    return "mask_mod"
  if aux_tensors:
    return "aux_tensors"
  return None


def _ffpa_attn_forward_sm100(
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
  """SM100 D512 forward launcher, signature-compatible with SM90/SM80."""
  device_arch, cute_arch_key = _validate_sm100_arch()
  if softcap == 0.0:
    softcap = None
  causal, local, window_size_left, window_size_right = (
    _resolve_causal_local_window(
      causal, window_size_left, window_size_right, mask_mod
    )
  )
  # Named before the shape validation, so a half-paired varlen call reports the
  # rule it broke rather than a rank-4 symptom.
  reason = _sm100_d512_unsupported_reason(
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_k=cu_seqlens_k,
    local=local,
    softcap=softcap,
    score_mod=score_mod,
    mask_mod=mask_mod,
    aux_tensors=aux_tensors,
  )
  if reason is not None:
    raise NotImplementedError(
      f"The SM100 dedicated D512 forward supports dense or packed-varlen "
      f"attention with optional causal masking only; unsupported: {reason}."
    )

  # Copy only per _needs_dense_copy; full contiguity is pure extra HBM traffic.
  q, k, v = [t.contiguous() if _needs_dense_copy(t) else t for t in (q, k, v)]
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
    q, k, v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k
  )
  if head_dim != SM100_D512_HEAD_DIM or head_dim_v != SM100_D512_HEAD_DIM:
    raise NotImplementedError(
      f"The SM100 dedicated forward is an exact specialisation for "
      f"head_dim == head_dim_v == {SM100_D512_HEAD_DIM}; got "
      f"{head_dim} / {head_dim_v}."
    )
  requires_grad = (q.requires_grad or k.requires_grad
                   or v.requires_grad) and not is_fake_mode()
  _validate_training_dtype(q, k, v, requires_grad)
  _validate_max_seqlen_for_cu_seqlens(
    cu_seqlens_q, "cu_seqlens_q", max_seqlen_q, "max_seqlen_q"
  )
  _validate_max_seqlen_for_cu_seqlens(
    cu_seqlens_k, "cu_seqlens_k", max_seqlen_k, "max_seqlen_k"
  )
  if softmax_scale is None:
    softmax_scale = 1.0 / math.sqrt(head_dim)
  qhead_per_kvhead = num_head // num_head_kv
  device = q.device
  is_varlen = cu_seqlens_q is not None
  # GQA is a zero-stride broadcast, so ignoring pack_gqa is correctness-safe.
  if pack_gqa and is_varlen:
    fa_log(
      1, "SM100 D512 forward ignoring pack_gqa=True: GQA is a zero-stride "
      "broadcast on this path"
    )

  out_shape = ((total_q, num_head, head_dim_v) if is_varlen else
               (batch_size, seqlen_q, num_head, head_dim_v))
  if out is None:
    if is_varlen:
      out = torch.empty(out_shape, dtype=q.dtype, device=device)
    else:
      # [B, H, N, D]-backed O saves the SDPA caller an exit copy (same rate).
      out = torch.empty(
        batch_size,
        num_head,
        seqlen_q,
        head_dim_v,
        dtype=q.dtype,
        device=device,
      ).transpose(1, 2)
  else:
    _validate_tensor(out, "out", out_shape, q.dtype, device)
    # An output cannot be copied into shape; a violation is silently wrong.
    if _needs_dense_copy(out):
      raise ValueError(
        f"out must have a contiguous trailing dimension and 128-bit aligned "
        f"leading strides; got shape {tuple(out.shape)} stride {out.stride()}"
      )

  # Varlen LSE is one flat (head, total_q) plane; the kernel rebases the row.
  lse_shape = ((num_head, total_q) if is_varlen else
               (batch_size, num_head, seqlen_q))
  if lse is None:
    lse = (
      torch.empty(lse_shape, dtype=torch.float32, device=device)
      if requires_grad or return_lse else None
    )
  else:
    _validate_tensor(lse, "lse", lse_shape, torch.float32, device)

  # Kernel stores empty-row O = 0 / LSE = -inf; only size-0 inputs land here.
  if total_q == 0 or seqlen_k == 0:
    if not is_fake_mode():
      out.zero_()
      if lse is not None:
        lse.fill_(-float("inf"))
    return out, lse

  dtype = torch2cute_dtype_map[q.dtype]
  ffpa_fwd = FFPAAttnFwdSm100D512(
    head_dim,
    head_dim_v,
    qhead_per_kvhead=qhead_per_kvhead,
    is_causal=causal,
    is_varlen_q=is_varlen,
  )

  # Every choice that reaches codegen must fork the compile key.
  compile_key = compile_key_fields(ffpa_fwd) + (
    dtype,
    head_dim,
    head_dim_v,
    qhead_per_kvhead,
    is_varlen,
    lse is None,
    device_arch,
    cute_arch_key,
    fa_logging.get_fa_log_level(),
  )
  if compile_key not in _ffpa_attn_forward_sm100.compile_cache:
    q_tensor, k_tensor, v_tensor, o_tensor = [
      to_cute_tensor(t) for t in (q, k, v, out)
    ]
    cu_q_tensor, cu_k_tensor = [
      to_cute_tensor(t, assumed_align=4, leading_dim=0)
      if t is not None else None for t in (cu_seqlens_q, cu_seqlens_k)
    ]
    lse_tensor = to_cute_tensor(
      lse, assumed_align=4
    ) if lse is not None else None
    current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    # None slots are constexpr-pruned features; the tags pin the donor order.
    _ffpa_attn_forward_sm100.compile_cache[compile_key] = cute.compile(
      ffpa_fwd,
      q_tensor,
      k_tensor,
      v_tensor,
      o_tensor,
      lse_tensor,
      softmax_scale,
      cu_q_tensor,
      cu_k_tensor,
      None,  # mSeqUsedQ
      None,  # mSeqUsedK
      None,  # mPageTable
      None,  # window_size_left
      None,  # window_size_right
      None,  # learnable_sink
      None,  # descale_tensors
      None,  # blocksparse_tensors
      AuxData(),
      current_stream,
      options=_PTXAS,
    )

  if not is_fake_mode():
    _call_with_tvm_ffi_current_stream(
      _ffpa_attn_forward_sm100.compile_cache[compile_key],
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


_ffpa_attn_forward_sm100.compile_cache = get_jit_cache("fwd_sm100")

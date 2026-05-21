"""FFPA cutedsl forward pass — SplitD SM90 for head_dim == 512.

Exposes :func:`_ffpa_attn_forward_sm90` and its compile cache, imported by
:mod:`cutedsl.__init__` for the ``ffpa_attn::_fwd_cutedsl`` torch custom op
and the varlen forward path.
"""

import os
import math
from typing import Optional, Tuple, Callable

import torch
import cutlass.cute as cute

from ._utils import (
  FWD_TILE_M,
  FWD_TILE_N,
  is_fake_mode,
  maybe_contiguous,
  _call_with_tvm_ffi_current_stream,
  _validate_tensor,
  _validate_sm90_arch,
  _validate_training_dtype,
  _validate_max_seqlen_for_cu_seqlens,
  _validate_qkv_common,
  _unsupported_training_features,
  _resolve_causal_local_window,
  torch2cute_dtype_map,
)
from ._fwd_d512_sm90 import FFPAAttnFwdSm90SplitD
from ._fwd_generic_sm90 import FFPAAttnFwdSm90SplitDGeneric
from .utils.cache_utils import get_jit_cache
from . import utils
from .utils import fa_logging
from .utils.cute_dsl_utils import (
  to_cute_tensor,
  to_cute_aux_tensor,
  get_aux_tensor_metadata,
)

if os.environ.get("CUTE_DSL_PTXAS_PATH", None) is not None:
  from .utils import cute_dsl_ptxas  # noqa: F401

  cute_dsl_ptxas.patch()


def _ffpa_attn_forward_sm90(
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
  """SplitD SM90 forward pass for FFPA attention (head_dim == 512)."""
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
    q, k, v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k
  )

  requires_grad = q.requires_grad or k.requires_grad or v.requires_grad
  _validate_training_dtype(q, k, v, requires_grad)
  _validate_max_seqlen_for_cu_seqlens(
    cu_seqlens_q, "cu_seqlens_q", max_seqlen_q, "max_seqlen_q"
  )
  _validate_max_seqlen_for_cu_seqlens(
    cu_seqlens_k, "cu_seqlens_k", max_seqlen_k, "max_seqlen_k"
  )

  device_arch, cute_arch_key = _validate_sm90_arch()
  if softmax_scale is None:
    softmax_scale = 1.0 / math.sqrt(head_dim)
  if softcap == 0.0:
    softcap = None
  qhead_per_kvhead = num_head // num_head_kv
  if pack_gqa is None:
    pack_gqa = qhead_per_kvhead > 1

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
      device=device
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
  elif lse is not None:
    _validate_tensor(lse, "lse", lse_shape, torch.float32, device)

  dtype = torch2cute_dtype_map[q.dtype]

  causal, local, window_size_left, window_size_right = _resolve_causal_local_window(
    causal, window_size_left, window_size_right, mask_mod
  )
  _unsupported_training_features(
    requires_grad, softcap, local, score_mod, mask_mod, aux_tensors
  )

  current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

  # SplitD tile sizes (hardcoded)
  tile_m = FWD_TILE_M  # tile_m=64 required by num_wg_mma==1 for register headroom
  tile_n = FWD_TILE_N  # tile_n=128 with sO_spill for register pressure management

  # Auto-detect K=V: same data pointer means same tensor
  kv_same = k is v if is_fake_mode() else k.data_ptr() == v.data_ptr()

  if max_seqlen_q is None:
    max_seqlen_q = seqlen_q if cu_seqlens_q is None else total_q
  if max_seqlen_k is None:
    max_seqlen_k = seqlen_k

  if softcap is not None:
    if score_mod is not None:
      raise ValueError("softcap and score_mod cannot be used together")
    score_mod = utils.create_softcap_scoremod(softcap)

  score_mod_hash = utils.hash_callable(
    score_mod
  ) if score_mod is not None else False
  mask_mod_hash = utils.hash_callable(
    mask_mod
  ) if mask_mod is not None else False

  is_varlen = cu_seqlens_q is not None or cu_seqlens_k is not None

  if mask_mod is not None and is_varlen:
    raise NotImplementedError(
      "mask_mod with aux_tensors is not yet supported for varlen sequences."
    )

  if aux_tensors is not None:
    aux_tensor_metadata = get_aux_tensor_metadata(aux_tensors)
  else:
    aux_tensor_metadata = None

  # forward kernel skips those tiles; prefill their mathematical result here.
  if (is_varlen or causal or local) and not is_fake_mode():
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
    dtype,
    head_dim,
    head_dim_v,
    qhead_per_kvhead,
    causal,
    score_mod_hash,
    mask_mod_hash,
    aux_tensor_metadata,
    lse is None,
    cu_seqlens_q is None,
    cu_seqlens_k is None,
    window_size_left is not None,
    window_size_right is not None,
    tile_m,
    tile_n,
    pack_gqa,
    device_arch,
    cute_arch_key,
    kv_same,
    fa_logging.get_fa_log_level(),
  )
  if compile_key not in _ffpa_attn_forward_sm90.compile_cache:
    cu_seqlens_q_tensor, cu_seqlens_k_tensor = [
      to_cute_tensor(t, assumed_align=4, leading_dim=0)
      if t is not None else None for t in (cu_seqlens_q, cu_seqlens_k)
    ]
    q_tensor, k_tensor, v_tensor, o_tensor = [
      to_cute_tensor(t) for t in (q, k, v, out)
    ]
    if lse is not None:
      lse_tensor = to_cute_tensor(lse, assumed_align=4)
    else:
      lse_tensor = None

    cute_aux_tensors = None
    if aux_tensors is not None:
      cute_aux_tensors = [to_cute_aux_tensor(buf) for buf in aux_tensors]

    fwd_kernel_cls = (
      FFPAAttnFwdSm90SplitD
      if head_dim == 512 and head_dim_v == 512 else FFPAAttnFwdSm90SplitDGeneric
    )
    ffpa_fwd = fwd_kernel_cls(
      dtype,
      head_dim,
      head_dim_v,
      qhead_per_kvhead,
      is_causal=causal,
      is_local=local,
      pack_gqa=pack_gqa,
      tile_m=tile_m,
      tile_n=tile_n,
      kv_same=kv_same,
      mask_mod=mask_mod,
      score_mod=score_mod,
      has_aux_tensors=aux_tensors is not None,
    )

    # Positional args must match FFPAAttnFwdSm90SplitD.__call__ signature:
    # mQ, mK, mV, mO, mLSE, scale, cuseqlens_q, cuseqlens_k, wsl, wsr, aux, stream
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
      window_size_left,
      window_size_right,
      cute_aux_tensors,
      current_stream,
    ]
    _ffpa_attn_forward_sm90.compile_cache[compile_key] = cute.compile(
      *compile_args,
      options=(
        "--enable-tvm-ffi --ptxas-options '--verbose --warn-on-spills --warn-on-local-memory-usage'"
      ),
    )

  if not is_fake_mode():
    q_call, k_call, v_call = q.detach(), k.detach(), v.detach()
    call_args = [
      q_call,
      k_call,
      v_call,
      out.detach(),
      lse,
      softmax_scale,
      cu_seqlens_q,
      cu_seqlens_k,
      window_size_left,
      window_size_right,
      aux_tensors,
    ]
    _call_with_tvm_ffi_current_stream(
      _ffpa_attn_forward_sm90.compile_cache[compile_key],
      *call_args,
      device=device,
    )
  return out, lse


_ffpa_attn_forward_sm90.compile_cache = get_jit_cache("fwd_sm90")

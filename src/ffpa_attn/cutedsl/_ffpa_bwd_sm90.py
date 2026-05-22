"""FFPA cutedsl backward pass — SplitD SM90 for head_dim == 512.

Exposes :func:`_ffpa_attn_backward_sm90` and its compile caches, together with
the bwd preprocess kernel (:func:`_bwd_preprocess`), imported by
:mod:`cutedsl.__init__` for the ``ffpa_attn::_bwd_cutedsl`` torch custom op.
"""

import os
import math
from typing import Optional, Tuple

import torch
import cutlass.cute as cute
from cutlass import Int32, Float32
from quack.compile_utils import make_fake_tensor as fake_tensor

from ._utils import (
  BWD_TILE_M,
  BWD_TILE_N,
  is_fake_mode,
  maybe_contiguous,
  _call_with_tvm_ffi_current_stream,
  _validate_tensor,
  _validate_sm90_arch,
  _validate_max_seqlen_for_cu_seqlens,
  _validate_qkv_common,
  _resolve_causal_local_window,
  torch2cute_dtype_map,
)
from ._bwd_preprocess import FFPAAttnBwdPreprocess
from ._dkdv_d512_sm90 import FFPAAttnBwdDKDVSm90SplitD
from ._dq_d512_sm90 import FFPAAttnBwdDQSm90SplitD
from ._dkdv_d384_sm90 import FFPAAttnBwdDKDVSm90SplitDD384
from ._dq_d384_sm90 import FFPAAttnBwdDQSm90SplitDD384
from ._dkdv_generic_sm90 import FFPAAttnBwdDKDVSm90SplitDGeneric
from ._dq_generic_sm90 import FFPAAttnBwdDQSm90SplitDGeneric
from .utils.cache_utils import get_jit_cache
from .utils.cute_dsl_utils import (
  to_cute_tensor,
)

if os.environ.get("CUTE_DSL_PTXAS_PATH", None) is not None:
  from .utils import cute_dsl_ptxas  # noqa: F401

  cute_dsl_ptxas.patch()


def _make_fake_bwd_preprocess_tensors(dtype, varlen_q):
  sym = cute.sym_int
  div = 128 // dtype.width  # 8 for fp16/bf16
  b, seqlen_q, h_q, d_v = sym(), sym(), sym(), sym()
  seqlen_q_rounded = sym()
  total_q, total_q_rounded = sym(), sym()
  b_seqlenq = (b, seqlen_q) if not varlen_q else (total_q, )
  mO = fake_tensor(dtype, (*b_seqlenq, h_q, d_v), divisibility=div)
  mdO = fake_tensor(dtype, (*b_seqlenq, h_q, d_v), divisibility=div)
  if not varlen_q:
    mLSE = fake_tensor(Float32, (b, h_q, seqlen_q), divisibility=1)
    mLSElog2 = fake_tensor(Float32, (b, h_q, seqlen_q_rounded), divisibility=4)
    mPdPsum = fake_tensor(Float32, (b, h_q, seqlen_q_rounded), divisibility=4)
  else:
    mLSE = fake_tensor(Float32, (h_q, total_q), divisibility=1)
    mLSElog2 = fake_tensor(Float32, (h_q, total_q_rounded), divisibility=4)
    mPdPsum = fake_tensor(Float32, (h_q, total_q_rounded), divisibility=4)
  return mO, mdO, mLSE, mLSElog2, mPdPsum


def _compile_bwd_preprocess(
  dtype,
  head_dim,
  head_dim_v,
  m_block_size,
  has_cuseqlens_q,
  has_dlse,
  device_arch,
  cute_arch_key,
):
  """Compile bwd preprocess kernel using cute fake tensors."""
  batchp1 = cute.sym_int()
  mO, mdO, mLSE, mLSElog2, mPdPsum = _make_fake_bwd_preprocess_tensors(
    dtype, varlen_q=has_cuseqlens_q
  )
  mCuSeqlensQ = fake_tensor(
    Int32, (batchp1, ), divisibility=1
  ) if has_cuseqlens_q else None
  mdLSE = fake_tensor(Float32, mLSE.shape, divisibility=1) if has_dlse else None
  ffpa_bwd_pre = FFPAAttnBwdPreprocess(
    dtype, head_dim, head_dim_v, m_block_size
  )
  return cute.compile(
    ffpa_bwd_pre,
    mO,
    mdO,
    mPdPsum,
    mLSE,
    mLSElog2,
    None,
    mCuSeqlensQ,
    None,
    mdLSE,
    cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    options="--enable-tvm-ffi",
  )


def _bwd_preprocess(
  out,
  dout,
  dpsum,
  lse,
  lse_log2,
  cu_seqlens_q,
  dlse,
  dtype,
  head_dim,
  head_dim_v,
  m_block_size,
  device_arch,
  cute_arch_key,
):
  """Backward preprocess: compute (o * dout).sum(dim=-1) - dLSE, and lse * log2_e."""
  is_varlen = cu_seqlens_q is not None
  compile_key = (
    dtype,
    head_dim,
    head_dim_v,
    m_block_size,
    is_varlen,
    dlse is not None,
    device_arch,
    cute_arch_key,
  )
  if compile_key not in _bwd_preprocess.compile_cache:
    _bwd_preprocess.compile_cache[compile_key] = _compile_bwd_preprocess(
      *compile_key
    )
  if not is_fake_mode():
    _call_with_tvm_ffi_current_stream(
      _bwd_preprocess.compile_cache[compile_key],
      out,
      dout,
      dpsum,
      lse,
      lse_log2,
      None,
      cu_seqlens_q,
      None,
      dlse,
      device=out.device,
    )


_bwd_preprocess.compile_cache = get_jit_cache("bwd_pre_sm90")


def _ffpa_attn_backward_sm90(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  out: torch.Tensor,
  dout: torch.Tensor,
  lse: torch.Tensor,
  softmax_scale: Optional[float] = None,
  causal: bool = False,
  softcap: float = 0.0,
  window_size_left: Optional[int] = None,
  window_size_right: Optional[int] = None,
  cu_seqlens_q: Optional[torch.Tensor] = None,
  cu_seqlens_k: Optional[torch.Tensor] = None,
  max_seqlen_q: Optional[int] = None,
  max_seqlen_k: Optional[int] = None,
  dq: Optional[torch.Tensor] = None,
  dk: Optional[torch.Tensor] = None,
  dv: Optional[torch.Tensor] = None,
  dlse: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """SplitD SM90 backward pass (training only, head_dim == 512)."""
  device_arch, cute_arch_key = _validate_sm90_arch()

  if softcap != 0.0:
    raise NotImplementedError("SplitD backward does not support softcap yet")

  causal, local, window_size_left, window_size_right = _resolve_causal_local_window(
    causal, window_size_left, window_size_right
  )
  if local:
    raise NotImplementedError(
      "SplitD backward does not support local/window attention yet"
    )

  # SplitD tile sizes (hardcoded)
  m_block_size = BWD_TILE_M
  n_block_size = BWD_TILE_N

  q, k, v, out, dout, lse, cu_seqlens_q, cu_seqlens_k = [
    maybe_contiguous(t)
    for t in (q, k, v, out, dout, lse, cu_seqlens_q, cu_seqlens_k)
  ]
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
  _validate_max_seqlen_for_cu_seqlens(
    cu_seqlens_q, "cu_seqlens_q", max_seqlen_q, "max_seqlen_q"
  )
  _validate_max_seqlen_for_cu_seqlens(
    cu_seqlens_k, "cu_seqlens_k", max_seqlen_k, "max_seqlen_k"
  )
  if cu_seqlens_q is None:
    seqlen_q_for_rounding = seqlen_q
  else:
    seqlen_q_for_rounding = max_seqlen_q if max_seqlen_q is not None else total_q

  seqlen_q_rounded = (
    seqlen_q_for_rounding + m_block_size - 1
  ) // m_block_size * m_block_size
  device = q.device
  out_torch_dtype = q.dtype
  if cu_seqlens_q is not None:
    out_shape = (total_q, num_head, head_dim_v)
    lse_shape = (num_head, total_q)
  else:
    out_shape = (batch_size, seqlen_q, num_head, head_dim_v)
    lse_shape = (batch_size, num_head, seqlen_q)
  _validate_tensor(out, "out", out_shape, out_torch_dtype, device)
  _validate_tensor(dout, "dout", out_shape, out_torch_dtype, device)
  _validate_tensor(lse, "lse", lse_shape, torch.float32, device)
  if dlse is not None:
    dlse = maybe_contiguous(dlse)
    _validate_tensor(dlse, "dlse", lse_shape, torch.float32, device)
  if softmax_scale is None:
    softmax_scale = 1.0 / math.sqrt(head_dim)
  qhead_per_kvhead = num_head // num_head_kv

  if dq is None:
    dq = torch.zeros_like(q)
  else:
    _validate_tensor(dq, "dq", q.shape, out_torch_dtype, device)
    if not is_fake_mode():
      dq.zero_()

  if dk is None:
    dk = torch.zeros_like(k)
  else:
    _validate_tensor(dk, "dk", k.shape, out_torch_dtype, device)
    if not is_fake_mode():
      dk.zero_()

  if dv is None:
    dv = torch.zeros_like(v)
  else:
    _validate_tensor(dv, "dv", v.shape, out_torch_dtype, device)
    if not is_fake_mode():
      dv.zero_()

  if total_q == 0 or seqlen_k == 0:
    return dq, dk, dv

  if cu_seqlens_q is None:
    dpsum = torch.empty(
      batch_size,
      num_head,
      seqlen_q_rounded,
      dtype=torch.float32,
      device=device
    )
    lse_log2 = torch.empty(
      batch_size,
      num_head,
      seqlen_q_rounded,
      dtype=torch.float32,
      device=device
    )
  else:
    total_q_rounded_padded = (
      total_q + cu_seqlens_q.shape[0] * m_block_size - 1
    ) // m_block_size * m_block_size
    dpsum = torch.empty(
      num_head, total_q_rounded_padded, dtype=torch.float32, device=device
    )
    lse_log2 = torch.empty(
      num_head, total_q_rounded_padded, dtype=torch.float32, device=device
    )

  dtype = torch2cute_dtype_map[q.dtype]
  current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

  # (1) Preprocess dpsum and lse_log2 for the SplitD backward kernels.
  _bwd_preprocess(
    out,
    dout,
    dpsum,
    lse,
    lse_log2,
    cu_seqlens_q,
    dlse,
    dtype,
    head_dim,
    head_dim_v,
    m_block_size,
    device_arch,
    cute_arch_key,
  )

  # (2) Compile and execute SplitD dKdV and dQ kernels
  if head_dim == 512 and head_dim_v == 512:
    bwd_kernel_kind = "d512"
  elif head_dim <= 384 and head_dim_v <= 384:
    bwd_kernel_kind = "d384"
  else:
    bwd_kernel_kind = "d512_generic"

  bwd_key = (
    bwd_kernel_kind,
    dtype,
    head_dim,
    head_dim_v,
    causal,
    m_block_size,
    n_block_size,
    cu_seqlens_q is not None,
    cu_seqlens_k is not None,
    qhead_per_kvhead,
    device_arch,
    cute_arch_key,
  )
  if bwd_key not in _ffpa_attn_backward_sm90.compile_cache_dkdv:
    q_t, k_t, v_t, do_t = [to_cute_tensor(t) for t in (q, k, v, dout)]
    dk_t, dv_t = [to_cute_tensor(t) for t in (dk, dv)]
    lse_log2_t = to_cute_tensor(lse_log2, assumed_align=4)
    dpsum_t = to_cute_tensor(dpsum, assumed_align=4)
    cu_seqlens_q_t = (
      to_cute_tensor(cu_seqlens_q, assumed_align=4, leading_dim=0)
      if cu_seqlens_q is not None else None
    )
    cu_seqlens_k_t = (
      to_cute_tensor(cu_seqlens_k, assumed_align=4, leading_dim=0)
      if cu_seqlens_k is not None else None
    )

    if bwd_kernel_kind == "d512":
      dkdv_kernel_cls = FFPAAttnBwdDKDVSm90SplitD
    elif bwd_kernel_kind == "d384":
      dkdv_kernel_cls = FFPAAttnBwdDKDVSm90SplitDD384
    else:
      dkdv_kernel_cls = FFPAAttnBwdDKDVSm90SplitDGeneric
    ffpa_dkdv = dkdv_kernel_cls(
      dtype,
      head_dim,
      head_dim_v=head_dim_v,
      is_causal=causal,
      qhead_per_kvhead=qhead_per_kvhead,
      tile_m=m_block_size,
      tile_n=n_block_size,
    )
    _ffpa_attn_backward_sm90.compile_cache_dkdv[bwd_key] = cute.compile(
      ffpa_dkdv,
      q_t,
      k_t,
      v_t,
      do_t,
      lse_log2_t,
      dpsum_t,
      dk_t,
      dv_t,
      softmax_scale,
      cu_seqlens_q_t,
      cu_seqlens_k_t,
      current_stream,
      options=(
        "--enable-tvm-ffi --ptxas-options '--verbose --warn-on-spills --warn-on-local-memory-usage'"
      ),
    )

  if bwd_key not in _ffpa_attn_backward_sm90.compile_cache_dq:
    q_t2, k_t2, v_t2, do_t2 = [to_cute_tensor(t) for t in (q, k, v, dout)]
    dq_t = to_cute_tensor(dq)
    lse_log2_t2 = to_cute_tensor(lse_log2, assumed_align=4)
    dpsum_t2 = to_cute_tensor(dpsum, assumed_align=4)
    cu_seqlens_q_t2 = (
      to_cute_tensor(cu_seqlens_q, assumed_align=4, leading_dim=0)
      if cu_seqlens_q is not None else None
    )
    cu_seqlens_k_t2 = (
      to_cute_tensor(cu_seqlens_k, assumed_align=4, leading_dim=0)
      if cu_seqlens_k is not None else None
    )

    if bwd_kernel_kind == "d512":
      dq_kernel_cls = FFPAAttnBwdDQSm90SplitD
    elif bwd_kernel_kind == "d384":
      dq_kernel_cls = FFPAAttnBwdDQSm90SplitDD384
    else:
      dq_kernel_cls = FFPAAttnBwdDQSm90SplitDGeneric
    ffpa_dq = dq_kernel_cls(
      dtype,
      head_dim,
      head_dim_v=head_dim_v,
      is_causal=causal,
      qhead_per_kvhead=qhead_per_kvhead,
      tile_m=m_block_size,
      tile_n=n_block_size,
    )
    _ffpa_attn_backward_sm90.compile_cache_dq[bwd_key] = cute.compile(
      ffpa_dq,
      q_t2,
      k_t2,
      v_t2,
      do_t2,
      lse_log2_t2,
      dpsum_t2,
      dq_t,
      softmax_scale,
      cu_seqlens_q_t2,
      cu_seqlens_k_t2,
      current_stream,
      options=(
        "--enable-tvm-ffi --ptxas-options '--verbose --warn-on-spills --warn-on-local-memory-usage'"
      ),
    )

  # Execute dKdV and dQ kernels
  if not is_fake_mode():
    _call_with_tvm_ffi_current_stream(
      _ffpa_attn_backward_sm90.compile_cache_dkdv[bwd_key],
      q.detach(),
      k.detach(),
      v.detach(),
      dout,
      lse_log2,
      dpsum,
      dk,
      dv,
      softmax_scale,
      cu_seqlens_q,
      cu_seqlens_k,
      device=device,
    )
    _call_with_tvm_ffi_current_stream(
      _ffpa_attn_backward_sm90.compile_cache_dq[bwd_key],
      q.detach(),
      k.detach(),
      v.detach(),
      dout,
      lse_log2,
      dpsum,
      dq,
      softmax_scale,
      cu_seqlens_q,
      cu_seqlens_k,
      device=device,
    )

  return dq, dk, dv


_ffpa_attn_backward_sm90.compile_cache_dkdv = get_jit_cache(
  "bwd_splitd_dkdv_sm90"
)
_ffpa_attn_backward_sm90.compile_cache_dq = get_jit_cache("bwd_splitd_dq_sm90")

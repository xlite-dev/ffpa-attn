"""SM100 D512 backward wrapper; support contract in cute/README.md.
Four launches (preprocess, then dV, dK, dQ each alone): one gradient saturates
SMEM/TMEM; an unsplit pair cannot fit."""

import math
import os
from typing import Optional, Tuple

import torch
import cutlass.cute as cute
from cutlass import Int32, Float32
from quack.compile_utils import make_fake_tensor as fake_tensor

from ._utils import (
  SM100_BWD_TILE_M,
  SM100_BWD_TILE_N_DKDV,
  SM100_BWD_TILE_N_DQ,
  SM100_D512_HEAD_DIM,
  is_fake_mode,
  maybe_contiguous,
  torch2cute_dtype_map,
  _call_with_tvm_ffi_current_stream,
  _needs_dense_copy,
  _resolve_causal_local_window,
  _validate_max_seqlen_for_cu_seqlens,
  _validate_qkv_common,
  _validate_sm100_arch,
  _validate_tensor,
)
from ._bwd_preprocess import FFPAAttnBwdPreprocess
from ._dv_d512_sm100 import FFPAAttnBwdDVSm100D512
from ._dk_d512_sm100 import FFPAAttnBwdDKSm100D512
from ._dq_d512_sm100 import FFPAAttnBwdDQSm100D512
from .utils.cache_utils import get_jit_cache
from .utils.cute_dsl_utils import to_cute_tensor

if os.environ.get("CUTE_DSL_PTXAS_PATH", None) is not None:
  from .utils import cute_dsl_ptxas  # noqa: F401

  cute_dsl_ptxas.patch()

# Liveness witness: read only by the SM100 tests; this wrapper never reads it.
SM100_D512_BWD_KERNEL_LIVE = True

# Duplicates _ffpa_fwd_sm100._PTXAS so the two wrappers stay independent.
_PTXAS = (
  "--enable-tvm-ffi --ptxas-options "
  "'--verbose --warn-on-spills --warn-on-local-memory-usage'"
)


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
  """Compile the bwd preprocess: SM90 code; only the cache namespace differs."""
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
  """Backward preprocess: (o * dout).sum(-1) - dLSE and lse * log2(e)."""
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


_bwd_preprocess.compile_cache = get_jit_cache("bwd_pre_sm100")


def _ffpa_attn_backward_sm100(
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
  """SM100 dedicated D512 backward pass (training, head_dim == 512)."""
  device_arch, cute_arch_key = _validate_sm100_arch()

  if softcap != 0.0:
    raise NotImplementedError("SM100 D512 backward does not support softcap")

  causal, local, window_size_left, window_size_right = (
    _resolve_causal_local_window(causal, window_size_left, window_size_right)
  )
  if local:
    raise NotImplementedError(
      "SM100 D512 backward does not support local/window attention"
    )

  # Half a pairing reads across sequences; caught before the rank-4 symptom.
  if (cu_seqlens_q is None) != (cu_seqlens_k is None):
    raise NotImplementedError(
      "SM100 D512 backward varlen requires both cu_seqlens_q and cu_seqlens_k"
    )

  m_block_size = SM100_BWD_TILE_M

  # Forward's copy rule (_ffpa_fwd_sm100): only per _needs_dense_copy.
  q, k, v, out, dout = [
    t.contiguous() if t is not None and _needs_dense_copy(t) else t
    for t in (q, k, v, out, dout)
  ]
  # Preprocess holds these at divisibility=1; _needs_dense_copy would misreport.
  lse, cu_seqlens_q, cu_seqlens_k = [
    maybe_contiguous(t) for t in (lse, cu_seqlens_q, cu_seqlens_k)
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
  if head_dim != SM100_D512_HEAD_DIM or head_dim_v != SM100_D512_HEAD_DIM:
    raise NotImplementedError(
      f"The SM100 dedicated backward is an exact specialisation for "
      f"head_dim == head_dim_v == {SM100_D512_HEAD_DIM}; got "
      f"{head_dim} / {head_dim_v}."
    )
  _validate_max_seqlen_for_cu_seqlens(
    cu_seqlens_q, "cu_seqlens_q", max_seqlen_q, "max_seqlen_q"
  )
  _validate_max_seqlen_for_cu_seqlens(
    cu_seqlens_k, "cu_seqlens_k", max_seqlen_k, "max_seqlen_k"
  )
  device = q.device
  out_torch_dtype = q.dtype
  is_varlen = cu_seqlens_q is not None
  if is_varlen:
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

  # An output cannot be copied into shape; the epilogues read its strides.
  def _reject_dense_violation(t, name):
    if _needs_dense_copy(t):
      raise ValueError(
        f"{name} must have a contiguous trailing dimension, 128-bit aligned "
        f"leading strides and a 16-byte aligned base; got shape "
        f"{tuple(t.shape)} stride {t.stride()}"
      )

  # Zero-init is correctness: tiles no work item covers are never written.
  if dq is None:
    dq = torch.zeros_like(q)
  else:
    _validate_tensor(dq, "dq", q.shape, out_torch_dtype, device)
    _reject_dense_violation(dq, "dq")
    if not is_fake_mode():
      dq.zero_()
  if dk is None:
    dk = torch.zeros_like(k)
  else:
    _validate_tensor(dk, "dk", k.shape, out_torch_dtype, device)
    _reject_dense_violation(dk, "dk")
    if not is_fake_mode():
      dk.zero_()
  if dv is None:
    dv = torch.zeros_like(v)
  else:
    _validate_tensor(dv, "dv", v.shape, out_torch_dtype, device)
    _reject_dense_violation(dv, "dv")
    if not is_fake_mode():
      dv.zero_()

  if total_q == 0 or seqlen_k == 0:
    return dq, dk, dv

  # padded_offset_q is cumulative: one padded tile per boundary, else OOB.
  if not is_varlen:
    seqlen_q_rounded = (
      seqlen_q + m_block_size - 1
    ) // m_block_size * m_block_size
    stats_shape = (batch_size, num_head, seqlen_q_rounded)
  else:
    total_q_rounded_padded = (
      total_q + cu_seqlens_q.shape[0] * m_block_size - 1
    ) // m_block_size * m_block_size
    stats_shape = (num_head, total_q_rounded_padded)
  dpsum = torch.empty(stats_shape, dtype=torch.float32, device=device)
  lse_log2 = torch.empty(stats_shape, dtype=torch.float32, device=device)

  dtype = torch2cute_dtype_map[q.dtype]
  current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

  # (1) lse_log2 and dpsum, in exactly the layout the three kernels read.
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

  # (2) LPT grouping keys on the decision, so equal groupings share artifacts.
  lpt_head_group, lpt_num_groups = 0, 0
  if causal:
    lpt_head_group, lpt_num_groups = FFPAAttnBwdDKSm100D512.choose_lpt_grouping(
      int(seqlen_k),
      int(num_head_kv),
      1 if is_varlen else int(batch_size),
      SM100_BWD_TILE_N_DKDV,
      SM100_D512_HEAD_DIM,
    )

  # One key serves all three caches: every static axis this wrapper can reach.
  bwd_key = (
    dtype,
    head_dim,
    head_dim_v,
    causal,
    m_block_size,
    (SM100_BWD_TILE_N_DKDV, SM100_BWD_TILE_N_DQ),
    is_varlen,
    qhead_per_kvhead,
    (lpt_head_group, lpt_num_groups),
    device_arch,
    cute_arch_key,
  )

  def _cute_inputs():
    q_t, k_t, v_t, do_t = [to_cute_tensor(t) for t in (q, k, v, dout)]
    lse_log2_t = to_cute_tensor(lse_log2, assumed_align=4)
    dpsum_t = to_cute_tensor(dpsum, assumed_align=4)
    cu_q_t, cu_k_t = [
      to_cute_tensor(t, assumed_align=4, leading_dim=0)
      if t is not None else None for t in (cu_seqlens_q, cu_seqlens_k)
    ]
    return q_t, k_t, v_t, do_t, lse_log2_t, dpsum_t, cu_q_t, cu_k_t

  if bwd_key not in _ffpa_attn_backward_sm100.compile_cache_dv:
    q_t, k_t, _, do_t, lse_log2_t, _, cu_q_t, cu_k_t = _cute_inputs()
    _ffpa_attn_backward_sm100.compile_cache_dv[bwd_key] = cute.compile(
      FFPAAttnBwdDVSm100D512(
        acc_dtype=Float32,
        cta_tiler=(
          SM100_BWD_TILE_M, SM100_BWD_TILE_N_DKDV, SM100_D512_HEAD_DIM
        ),
        is_causal=causal,
        window_size_left=None,  # local attention is rejected above
        window_size_right=None,
        is_persistent=False,
        split_head=True,  # two 256-wide dV slices
      ),
      q_t,
      k_t,
      do_t,
      lse_log2_t,
      to_cute_tensor(dv),
      softmax_scale,
      cu_q_t,
      cu_k_t,
      current_stream,
      options=_PTXAS,
    )

  if bwd_key not in _ffpa_attn_backward_sm100.compile_cache_dk:
    q_t, k_t, v_t, do_t, lse_log2_t, dpsum_t, cu_q_t, cu_k_t = _cute_inputs()
    _ffpa_attn_backward_sm100.compile_cache_dk[bwd_key] = cute.compile(
      FFPAAttnBwdDKSm100D512(
        acc_dtype=Float32,
        cta_tiler=(
          SM100_BWD_TILE_M, SM100_BWD_TILE_N_DKDV, SM100_D512_HEAD_DIM
        ),
        is_causal=causal,
        window_size_left=None,  # local attention is rejected above
        window_size_right=None,
        is_persistent=False,
        split_head=True,  # the four-generation D reduction
        lpt_head_group=lpt_head_group,
        lpt_num_groups=lpt_num_groups,
      ),
      q_t,
      k_t,
      v_t,
      do_t,
      lse_log2_t,
      dpsum_t,
      to_cute_tensor(dk),
      softmax_scale,
      cu_q_t,
      cu_k_t,
      current_stream,
      options=_PTXAS,
    )

  if bwd_key not in _ffpa_attn_backward_sm100.compile_cache_dq:
    q_t, k_t, v_t, do_t, lse_log2_t, dpsum_t, cu_q_t, cu_k_t = _cute_inputs()
    _ffpa_attn_backward_sm100.compile_cache_dq[bwd_key] = cute.compile(
      FFPAAttnBwdDQSm100D512(
        acc_dtype=Float32,
        mma_tiler=(SM100_BWD_TILE_M, SM100_BWD_TILE_N_DQ, SM100_D512_HEAD_DIM),
        is_causal=causal,
        window_size_left=None,  # local attention is rejected above
        window_size_right=None,
        is_persistent=False,
        split_head=True,  # four 128-wide head slices
      ),
      q_t,
      k_t,
      v_t,
      do_t,
      lse_log2_t,
      dpsum_t,
      to_cute_tensor(dq),
      softmax_scale,
      cu_q_t,
      cu_k_t,
      current_stream,
      options=_PTXAS,
    )

  # (3) Three independent resident single-output launches.
  if not is_fake_mode():
    q_d, k_d, v_d = q.detach(), k.detach(), v.detach()
    _call_with_tvm_ffi_current_stream(
      _ffpa_attn_backward_sm100.compile_cache_dv[bwd_key],
      q_d,
      k_d,
      dout,
      lse_log2,
      dv,
      softmax_scale,
      cu_seqlens_q,
      cu_seqlens_k,
      device=device,
    )
    _call_with_tvm_ffi_current_stream(
      _ffpa_attn_backward_sm100.compile_cache_dk[bwd_key],
      q_d,
      k_d,
      v_d,
      dout,
      lse_log2,
      dpsum,
      dk,
      softmax_scale,
      cu_seqlens_q,
      cu_seqlens_k,
      device=device,
    )
    _call_with_tvm_ffi_current_stream(
      _ffpa_attn_backward_sm100.compile_cache_dq[bwd_key],
      q_d,
      k_d,
      v_d,
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


_ffpa_attn_backward_sm100.compile_cache_dv = get_jit_cache("bwd_dv_sm100")
_ffpa_attn_backward_sm100.compile_cache_dk = get_jit_cache("bwd_dk_sm100")
_ffpa_attn_backward_sm100.compile_cache_dq = get_jit_cache("bwd_dq_sm100")

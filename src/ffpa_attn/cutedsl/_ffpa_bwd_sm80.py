"""FFPA CuTeDSL backward pass for SM80/SM89 Split-D kernels."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import cutlass.cute as cute
from cutlass import Float32, Int32
from quack.compile_utils import make_fake_tensor as fake_tensor

from ._bwd_preprocess import FFPAAttnBwdPreprocess
from ._dkdv_generic_sm80 import FFPAAttnBwdDKDVSm80SplitDGeneric
from ._dq_generic_sm80 import FFPAAttnBwdDQSm80SplitDGeneric
from ._utils import (
  SM80_BWD_SPLIT_D_CHUNK,
  is_fake_mode,
  maybe_contiguous,
  _call_with_tvm_ffi_current_stream,
  _resolve_causal_local_window,
  _validate_max_seqlen_for_cu_seqlens,
  _validate_qkv_common,
  _validate_sm80_arch,
  _validate_sm80_head_dims,
  _validate_tensor,
  torch2cute_dtype_map,
)
from .utils.cache_utils import get_jit_cache
from .utils.cute_dsl_utils import to_cute_tensor

SM80_BWD_DKDV_TILE_M = 64
SM80_BWD_DKDV_TILE_N = 64
SM80_BWD_DQ_TILE_M = 64
SM80_BWD_DQ_TILE_N = 64
SM80_BWD_NUM_STAGES_Q = 1
SM80_BWD_NUM_STAGES_DO = 1
SM80_BWD_DKDV_NUM_THREADS = 128
SM80_BWD_DQ_NUM_THREADS = 128


def _make_fake_bwd_preprocess_tensors(dtype, varlen_q):
  sym = cute.sym_int
  div = 128 // dtype.width
  batch, seqlen_q, num_head, head_dim_v = sym(), sym(), sym(), sym()
  seqlen_q_rounded = sym()
  total_q, total_q_rounded = sym(), sym()
  q_shape = (batch, seqlen_q) if not varlen_q else (total_q, )
  mO = fake_tensor(dtype, (*q_shape, num_head, head_dim_v), divisibility=div)
  mdO = fake_tensor(dtype, (*q_shape, num_head, head_dim_v), divisibility=div)
  if not varlen_q:
    mLSE = fake_tensor(Float32, (batch, num_head, seqlen_q), divisibility=1)
    mLSElog2 = fake_tensor(
      Float32, (batch, num_head, seqlen_q_rounded), divisibility=4
    )
    mD = fake_tensor(
      Float32, (batch, num_head, seqlen_q_rounded), divisibility=4
    )
  else:
    mLSE = fake_tensor(Float32, (num_head, total_q), divisibility=1)
    mLSElog2 = fake_tensor(Float32, (num_head, total_q_rounded), divisibility=4)
    mD = fake_tensor(Float32, (num_head, total_q_rounded), divisibility=4)
  return mO, mdO, mLSE, mLSElog2, mD


def _compile_bwd_preprocess(
  dtype,
  head_dim,
  head_dim_v,
  tile_m,
  has_cuseqlens_q,
  has_dlse,
  device_arch,
  cute_arch_key,
):
  """Compile the CuTeDSL backward preprocess kernel for SM80."""
  del device_arch, cute_arch_key
  batchp1 = cute.sym_int()
  mO, mdO, mLSE, mLSElog2, mD = _make_fake_bwd_preprocess_tensors(
    dtype, varlen_q=has_cuseqlens_q
  )
  mCuSeqlensQ = fake_tensor(
    Int32, (batchp1, ), divisibility=1
  ) if has_cuseqlens_q else None
  mdLSE = fake_tensor(Float32, mLSE.shape, divisibility=1) if has_dlse else None
  ffpa_bwd_pre = FFPAAttnBwdPreprocess(dtype, head_dim, head_dim_v, tile_m)
  return cute.compile(
    ffpa_bwd_pre,
    mO,
    mdO,
    mD,
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
  tile_m,
  device_arch,
  cute_arch_key,
):
  """Compute ``D = rowsum(out * dout) - dlse`` and ``lse * log2(e)``."""
  is_varlen = cu_seqlens_q is not None
  compile_key = (
    dtype,
    head_dim,
    head_dim_v,
    tile_m,
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


_bwd_preprocess.compile_cache = get_jit_cache("bwd_pre_sm80")


def _ffpa_attn_backward_sm80_dense(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  out: torch.Tensor,
  dout: torch.Tensor,
  lse: torch.Tensor,
  softmax_scale: float,
  causal: bool,
  dq: torch.Tensor,
  dk: torch.Tensor,
  dv: torch.Tensor,
  dlse: Optional[torch.Tensor],
  device_arch: int,
  cute_arch_key: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Run the native dense SM80 backward kernels."""
  batch_size, seqlen_q, num_head, head_dim = q.shape
  seqlen_k = k.shape[1]
  num_head_kv = k.shape[2]
  head_dim_v = v.shape[-1]
  qhead_per_kvhead = num_head // num_head_kv
  dkdv_tile_m = SM80_BWD_DKDV_TILE_M
  dkdv_tile_n = SM80_BWD_DKDV_TILE_N
  dq_tile_m = SM80_BWD_DQ_TILE_M
  dq_tile_n = SM80_BWD_DQ_TILE_N
  pre_tile_m = dq_tile_m
  dtype = torch2cute_dtype_map[q.dtype]
  smem_capacity_arch = f"sm_{device_arch // 10}{device_arch % 10}"
  if not FFPAAttnBwdDKDVSm80SplitDGeneric.can_implement(
    dtype,
    head_dim,
    head_dim_v,
    dkdv_tile_m,
    dkdv_tile_n,
    SM80_BWD_NUM_STAGES_Q,
    SM80_BWD_NUM_STAGES_DO,
    SM80_BWD_DKDV_NUM_THREADS,
    causal,
    smem_capacity_arch=smem_capacity_arch,
  ):
    raise RuntimeError(
      "SM80/SM89 CuTeDSL dK/dV configuration exceeds kernel resource limits: "
      f"head_dim={head_dim}, tile=({dkdv_tile_m}, {dkdv_tile_n}), "
      f"num_stages_Q={SM80_BWD_NUM_STAGES_Q}, "
      f"num_stages_dO={SM80_BWD_NUM_STAGES_DO}, arch={smem_capacity_arch}."
    )
  if not FFPAAttnBwdDQSm80SplitDGeneric.can_implement(
    dtype,
    head_dim,
    head_dim_v,
    dq_tile_m,
    dq_tile_n,
    SM80_BWD_NUM_STAGES_Q,
    SM80_BWD_NUM_STAGES_DO,
    SM80_BWD_DQ_NUM_THREADS,
    causal,
    smem_capacity_arch=smem_capacity_arch,
  ):
    raise RuntimeError(
      "SM80/SM89 CuTeDSL dQ configuration exceeds kernel resource limits: "
      f"head_dim={head_dim}, tile=({dq_tile_m}, {dq_tile_n}), "
      f"num_stages_Q={SM80_BWD_NUM_STAGES_Q}, "
      f"num_stages_dO={SM80_BWD_NUM_STAGES_DO}, arch={smem_capacity_arch}."
    )
  seqlen_q_rounded = (seqlen_q + pre_tile_m - 1) // pre_tile_m * pre_tile_m

  dpsum = torch.empty(
    batch_size,
    num_head,
    seqlen_q_rounded,
    dtype=torch.float32,
    device=q.device
  )
  lse_log2 = torch.empty_like(dpsum)
  current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

  _bwd_preprocess(
    out,
    dout,
    dpsum,
    lse,
    lse_log2,
    None,
    dlse,
    dtype,
    head_dim,
    head_dim_v,
    pre_tile_m,
    device_arch,
    cute_arch_key,
  )

  bwd_key = (
    "sm80_bwd_generic",
    dtype,
    head_dim,
    head_dim_v,
    causal,
    dkdv_tile_m,
    dkdv_tile_n,
    dq_tile_m,
    dq_tile_n,
    SM80_BWD_NUM_STAGES_Q,
    SM80_BWD_NUM_STAGES_DO,
    SM80_BWD_SPLIT_D_CHUNK,
    SM80_BWD_DKDV_NUM_THREADS,
    SM80_BWD_DQ_NUM_THREADS,
    seqlen_q,
    seqlen_k,
    qhead_per_kvhead,
    device_arch,
    cute_arch_key,
  )
  if bwd_key not in _ffpa_attn_backward_sm80_dense.compile_cache_dkdv:
    q_t, k_t, v_t, do_t = [to_cute_tensor(t) for t in (q, k, v, dout)]
    lse_log2_t = to_cute_tensor(lse_log2, assumed_align=4)
    dpsum_t = to_cute_tensor(dpsum, assumed_align=4)
    dk_t, dv_t = [to_cute_tensor(t) for t in (dk, dv)]
    ffpa_dkdv = FFPAAttnBwdDKDVSm80SplitDGeneric(
      dtype,
      head_dim,
      head_dim_v=head_dim_v,
      qhead_per_kvhead=qhead_per_kvhead,
      is_causal=causal,
      tile_m=dkdv_tile_m,
      tile_n=dkdv_tile_n,
      num_stages_Q=SM80_BWD_NUM_STAGES_Q,
      num_stages_dO=SM80_BWD_NUM_STAGES_DO,
      num_threads=SM80_BWD_DKDV_NUM_THREADS,
    )
    _ffpa_attn_backward_sm80_dense.compile_cache_dkdv[bwd_key] = cute.compile(
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
      current_stream,
      options="--enable-tvm-ffi",
    )

  if bwd_key not in _ffpa_attn_backward_sm80_dense.compile_cache_dq:
    q_t, k_t, v_t, do_t = [to_cute_tensor(t) for t in (q, k, v, dout)]
    lse_log2_t = to_cute_tensor(lse_log2, assumed_align=4)
    dpsum_t = to_cute_tensor(dpsum, assumed_align=4)
    dq_t = to_cute_tensor(dq)
    ffpa_dq = FFPAAttnBwdDQSm80SplitDGeneric(
      dtype,
      head_dim,
      head_dim_v=head_dim_v,
      qhead_per_kvhead=qhead_per_kvhead,
      is_causal=causal,
      tile_m=dq_tile_m,
      tile_n=dq_tile_n,
      num_stages_Q=SM80_BWD_NUM_STAGES_Q,
      num_stages_dO=SM80_BWD_NUM_STAGES_DO,
      num_threads=SM80_BWD_DQ_NUM_THREADS,
    )
    _ffpa_attn_backward_sm80_dense.compile_cache_dq[bwd_key] = cute.compile(
      ffpa_dq,
      q_t,
      k_t,
      v_t,
      do_t,
      lse_log2_t,
      dpsum_t,
      dq_t,
      softmax_scale,
      current_stream,
      options="--enable-tvm-ffi",
    )

  if not is_fake_mode():
    _call_with_tvm_ffi_current_stream(
      _ffpa_attn_backward_sm80_dense.compile_cache_dkdv[bwd_key],
      q.detach(),
      k.detach(),
      v.detach(),
      dout,
      lse_log2,
      dpsum,
      dk,
      dv,
      softmax_scale,
      device=q.device,
    )
    _call_with_tvm_ffi_current_stream(
      _ffpa_attn_backward_sm80_dense.compile_cache_dq[bwd_key],
      q.detach(),
      k.detach(),
      v.detach(),
      dout,
      lse_log2,
      dpsum,
      dq,
      softmax_scale,
      device=q.device,
    )
  return dq, dk, dv


_ffpa_attn_backward_sm80_dense.compile_cache_dkdv = get_jit_cache(
  "bwd_dkdv_sm80"
)
_ffpa_attn_backward_sm80_dense.compile_cache_dq = get_jit_cache("bwd_dq_sm80")


def _ffpa_attn_backward_sm80(
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
  """SM80/SM89 atomic-free Split-D backward launcher.

  :param q: Query tensor saved from forward.
  :param k: Key tensor saved from forward.
  :param v: Value tensor saved from forward.
  :param out: Forward output tensor.
  :param dout: Gradient with respect to ``out``.
  :param lse: Forward log-sum-exp tensor.
  :param softmax_scale: Attention scale.
  :param causal: Whether lower-right causal masking is applied.
  :param softcap: Unsupported for the SM80 Split-D path.
  :param window_size_left: Unsupported local-attention left window.
  :param window_size_right: Unsupported local-attention right window.
  :param cu_seqlens_q: Optional packed-query sequence offsets.
  :param cu_seqlens_k: Optional packed-key sequence offsets.
  :param max_seqlen_q: Maximum query sequence length for varlen inputs.
  :param max_seqlen_k: Maximum key sequence length for varlen inputs.
  :param dq: Optional preallocated query gradient.
  :param dk: Optional preallocated key gradient.
  :param dv: Optional preallocated value gradient.
  :param dlse: Optional gradient with respect to LSE.
  :returns: ``(dq, dk, dv)``.
  """
  device_arch, cute_arch_key = _validate_sm80_arch()
  if softcap != 0.0:
    raise NotImplementedError("SM80/SM89 backward does not support softcap")
  causal, local, window_size_left, window_size_right = _resolve_causal_local_window(
    causal, window_size_left, window_size_right
  )
  if local:
    raise NotImplementedError(
      "SM80/SM89 backward does not support local/window attention yet"
    )

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
    _num_head_kv,
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
  _validate_max_seqlen_for_cu_seqlens(
    cu_seqlens_q, "cu_seqlens_q", max_seqlen_q, "max_seqlen_q"
  )
  _validate_max_seqlen_for_cu_seqlens(
    cu_seqlens_k, "cu_seqlens_k", max_seqlen_k, "max_seqlen_k"
  )
  if softmax_scale is None:
    softmax_scale = 1.0 / math.sqrt(head_dim)

  if dq is None:
    dq = torch.zeros_like(q)
  else:
    _validate_tensor(dq, "dq", q.shape, q.dtype, q.device)
    if not is_fake_mode():
      dq.zero_()
  if dk is None:
    dk = torch.zeros_like(k)
  else:
    _validate_tensor(dk, "dk", k.shape, k.dtype, k.device)
    if not is_fake_mode():
      dk.zero_()
  if dv is None:
    dv = torch.zeros_like(v)
  else:
    _validate_tensor(dv, "dv", v.shape, v.dtype, v.device)
    if not is_fake_mode():
      dv.zero_()

  if cu_seqlens_q is not None or cu_seqlens_k is not None:
    if cu_seqlens_q is None or cu_seqlens_k is None:
      raise ValueError(
        "SM80/SM89 varlen backward requires both cu_seqlens_q and cu_seqlens_k"
      )
    _validate_tensor(
      out, "out", (total_q, num_head, head_dim_v), q.dtype, q.device
    )
    _validate_tensor(
      dout, "dout", (total_q, num_head, head_dim_v), q.dtype, q.device
    )
    _validate_tensor(lse, "lse", (num_head, total_q), torch.float32, q.device)
    if dlse is not None:
      dlse = maybe_contiguous(dlse)
      _validate_tensor(
        dlse, "dlse", (num_head, total_q), torch.float32, q.device
      )
    if not is_fake_mode():
      dq.zero_()
      dk.zero_()
      dv.zero_()
    # The SM80 dK/dV and dQ kernels use a dense grid (grid dimensions
    # derived from batch * seqlen) and a fixed tile loop inside the CTA,
    # without the TileScheduler / cu_seqlens awareness that the SM90 path
    # has.  For varlen inputs we therefore decompose into per-segment
    # dense calls, padding each segment to a tile multiple and copying
    # only the valid rows back.  This avoids reworking the kernel itself
    # for a workload that is rare on Ampere/Ada inference silicon.
    # TODO: Implement a native varlen path for SM80 if there is demand,
    # potentially reusing some of the SM90 infrastructure.
    for batch_idx in range(cu_seqlens_q.numel() - 1):
      q_start = int(cu_seqlens_q[batch_idx].item())
      q_end = int(cu_seqlens_q[batch_idx + 1].item())
      k_start = int(cu_seqlens_k[batch_idx].item())
      k_end = int(cu_seqlens_k[batch_idx + 1].item())
      if q_end == q_start:
        continue
      if k_end == k_start:
        dq[q_start:q_end].zero_()
        continue
      q_seg = q[q_start:q_end].unsqueeze(0).contiguous()
      k_seg = k[k_start:k_end].unsqueeze(0).contiguous()
      v_seg = v[k_start:k_end].unsqueeze(0).contiguous()
      out_seg = out[q_start:q_end].unsqueeze(0).contiguous()
      dout_seg = dout[q_start:q_end].unsqueeze(0).contiguous()
      lse_seg = lse[:, q_start:q_end].unsqueeze(0).contiguous()
      dlse_seg = dlse[:, q_start:q_end].unsqueeze(0).contiguous(
      ) if dlse is not None else None
      q_len = q_end - q_start
      k_len = k_end - k_start
      q_len_rounded = (
        q_len + SM80_BWD_DQ_TILE_M - 1
      ) // SM80_BWD_DQ_TILE_M * SM80_BWD_DQ_TILE_M
      k_len_rounded = (
        k_len + SM80_BWD_DKDV_TILE_N - 1
      ) // SM80_BWD_DKDV_TILE_N * SM80_BWD_DKDV_TILE_N
      dq_seg = q_seg.new_zeros((1, q_len_rounded, num_head, head_dim))
      dk_seg = k_seg.new_zeros((1, k_len_rounded, _num_head_kv, head_dim))
      dv_seg = v_seg.new_zeros((1, k_len_rounded, _num_head_kv, head_dim_v))
      _ffpa_attn_backward_sm80_dense(
        q_seg,
        k_seg,
        v_seg,
        out_seg,
        dout_seg,
        lse_seg,
        softmax_scale,
        causal,
        dq_seg,
        dk_seg,
        dv_seg,
        dlse_seg,
        device_arch,
        cute_arch_key,
      )
      dq[q_start:q_end].copy_(dq_seg[:, :q_len].squeeze(0))
      dk[k_start:k_end].copy_(dk_seg[:, :k_len].squeeze(0))
      dv[k_start:k_end].copy_(dv_seg[:, :k_len].squeeze(0))
    return dq, dk, dv

  q_batch_seqlen_shape = (batch_size, seqlen_q)
  _validate_tensor(
    out, "out", (*q_batch_seqlen_shape, num_head, head_dim_v), q.dtype, q.device
  )
  _validate_tensor(
    dout,
    "dout",
    (*q_batch_seqlen_shape, num_head, head_dim_v),
    q.dtype,
    q.device,
  )
  _validate_tensor(
    lse, "lse", (batch_size, num_head, seqlen_q), torch.float32, q.device
  )
  if dlse is not None:
    dlse = maybe_contiguous(dlse)
    _validate_tensor(
      dlse, "dlse", (batch_size, num_head, seqlen_q), torch.float32, q.device
    )

  return _ffpa_attn_backward_sm80_dense(
    q,
    k,
    v,
    out,
    dout,
    lse,
    softmax_scale,
    causal,
    dq,
    dk,
    dv,
    dlse,
    device_arch,
    cute_arch_key,
  )


__all__ = ["_ffpa_attn_backward_sm80"]

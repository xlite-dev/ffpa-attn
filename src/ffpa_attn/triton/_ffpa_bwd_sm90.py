"""SM90+ Triton backward entry points for experimental TMA kernels.

This module provides an SM90-specialized backward main path that replaces the
large Q / K / V / dO loads with TensorDescriptor loads. Small tensors and
irregular/broadcasted bias paths stay on raw pointers, and gradient outputs use
raw masked stores in this first TMA phase.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from ._autotune_utils import autotune_seqlen_key
from ._ffpa_bwd import (
  _attn_bias_broadcast_strides,
  _attn_bias_grad_is_key_bias,
  _attn_bias_grad_needs_reduction,
  _attn_bias_grad_reduces_query,
  _dropout_multiplier,
  _ffpa_bwd_key_bias_grad_reduce_kernel,
  _ffpa_bwd_pre,
  _get_pre_autotune,
  _normalize_bwd_pre_config,
)
from ._persistent_autotune import (
  PersistentConfigRequest,
  dtype_name,
  lookup_persistent_config,
)

_SM90_BWD_HEURISTICS = {
  "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
  "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
}


def _sm90_bwd_host_descriptor_pre_hook(nargs):
  """Set per-descriptor block shapes before a TMA backward launch."""
  if not isinstance(nargs.get("desc_q"), TensorDescriptor):
    return
  BLOCK_M = nargs["BLOCK_M"]
  BLOCK_N = nargs["BLOCK_N"]
  BLOCK_HEADDIM = nargs["BLOCK_HEADDIM"]
  nargs["desc_q"].block_shape = [BLOCK_M, BLOCK_HEADDIM]
  nargs["desc_k"].block_shape = [BLOCK_N, BLOCK_HEADDIM]
  nargs["desc_v"].block_shape = [BLOCK_N, BLOCK_HEADDIM]
  nargs["desc_do"].block_shape = [BLOCK_M, BLOCK_HEADDIM]


@triton.jit
def _ffpa_bwd_dkdv_sm90_q_block(
  desc_q: tl.tensor_descriptor,
  desc_k: tl.tensor_descriptor,
  desc_v: tl.tensor_descriptor,
  desc_do: tl.tensor_descriptor,
  DK: torch.Tensor,
  DV: torch.Tensor,
  LSE: torch.Tensor,
  D: torch.Tensor,
  AttnBias: torch.Tensor,
  GradAttnBias: torch.Tensor,
  softmax_scale: float,
  stride_dkn: int,
  stride_dvn: int,
  stride_bm: int,
  stride_bn: int,
  stride_gbm: int,
  stride_gbn: int,
  q_base_y: int,
  k_offset_y: int,
  start_m: int,
  begin_m: int,
  off_hb: int,
  offs_m: torch.Tensor,
  offs_n: torch.Tensor,
  offs_d: torch.Tensor,
  seqlen_q: int,
  seqlen_k: int,
  headdim: int,
  dropout_p: float,
  philox_offset: int,
  IS_CAUSAL: tl.constexpr,
  HAS_ATTN_BIAS: tl.constexpr,
  HAS_DROPOUT: tl.constexpr,
  PHILOX_SEED: tl.constexpr,
  BIAS_REQUIRES_GRAD: tl.constexpr,
  GRAD_BIAS_NEEDS_REDUCTION: tl.constexpr,
  GRAD_BIAS_REDUCES_M: tl.constexpr,
  GRAD_BIAS_STORE_PARTIAL: tl.constexpr,
  BLOCK_HEADDIM: tl.constexpr,
  DTYPE: tl.constexpr,
  EVEN_M: tl.constexpr,
  EVEN_N: tl.constexpr,
  BLOCK_M: tl.constexpr,
  BLOCK_N: tl.constexpr,
  num_d_chunks: tl.constexpr,
) -> None:
  offs_qm = start_m + offs_m
  q_offset_y = q_base_y + start_m

  S = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
  dP = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

  for d_chunk in range(num_d_chunks):
    d_start = d_chunk * BLOCK_HEADDIM
    q = desc_q.load([q_offset_y, d_start])
    k = desc_k.load([k_offset_y, d_start])
    v = desc_v.load([k_offset_y, d_start])
    do = desc_do.load([q_offset_y, d_start])
    S = tl.dot(q, tl.trans(k), acc=S)
    dP = tl.dot(do, tl.trans(v), acc=dP)

  if not EVEN_N:
    S = tl.where(offs_n[None, :] < seqlen_k, S, float("-inf"))
  if not EVEN_M:
    m_mask = offs_qm < seqlen_q
    S = tl.where(m_mask[:, None], S, float("-inf"))
  if IS_CAUSAL:
    S = tl.where(offs_qm[:, None] >= (offs_n[None, :]), S, float("-inf"))
  S = S * softmax_scale
  if HAS_ATTN_BIAS:
    bias = tl.load(
      AttnBias + offs_qm[:, None] * stride_bm + offs_n[None, :] * stride_bn,
      mask=(offs_qm[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k),
      other=0.0,
    )
    S += bias
  if EVEN_M:
    lse_i = tl.load(LSE + offs_qm)
  else:
    lse_i = tl.load(LSE + offs_qm, mask=offs_qm < seqlen_q, other=0.0)
  P = tl.exp(S - lse_i[:, None])
  dropout_mult = _dropout_multiplier(
    off_hb,
    offs_qm,
    offs_n,
    seqlen_q,
    seqlen_k,
    dropout_p,
    PHILOX_SEED,
    philox_offset,
    HAS_DROPOUT,
  )
  dP = dP * dropout_mult
  P_drop = P * dropout_mult
  if EVEN_M:
    Di = tl.load(D + offs_qm)
  else:
    Di = tl.load(D + offs_qm, mask=offs_qm < seqlen_q, other=0.0)
  if BIAS_REQUIRES_GRAD:
    dBias = P * (dP - Di[:, None])
    grad_bias_mask = (offs_qm[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k)
    if GRAD_BIAS_REDUCES_M:
      m_block = start_m // BLOCK_M
      grad_bias_ptrs = GradAttnBias + m_block * stride_gbm + offs_n * stride_gbn
      grad_bias = tl.sum(tl.where(grad_bias_mask, dBias, 0.0), axis=0)
      if GRAD_BIAS_STORE_PARTIAL:
        tl.store(grad_bias_ptrs, grad_bias, mask=offs_n < seqlen_k)
      else:
        tl.atomic_add(grad_bias_ptrs, grad_bias, sem="relaxed", mask=offs_n < seqlen_k)
    elif GRAD_BIAS_NEEDS_REDUCTION:
      grad_bias_ptrs = GradAttnBias + offs_qm[:, None] * stride_gbm + offs_n[None, :] * stride_gbn
      tl.atomic_add(grad_bias_ptrs, dBias, sem="relaxed", mask=grad_bias_mask)
    else:
      grad_bias_ptrs = GradAttnBias + offs_qm[:, None] * stride_gbm + offs_n[None, :] * stride_gbn
      tl.store(grad_bias_ptrs, dBias, mask=grad_bias_mask)
    dS = (dBias * softmax_scale).to(DTYPE)
  else:
    dS = (P * (dP - Di[:, None]) * softmax_scale).to(DTYPE)
  if not EVEN_M:
    dS = tl.where(m_mask[:, None], dS, 0.0)
    P_drop = tl.where(m_mask[:, None], P_drop, 0.0)

  for d_chunk in range(num_d_chunks):
    d_start = d_chunk * BLOCK_HEADDIM
    d_offs = d_start + offs_d
    q = desc_q.load([q_offset_y, d_start])
    do = desc_do.load([q_offset_y, d_start])
    dk_ptrs = DK + offs_n[:, None] * stride_dkn + d_offs[None, :]
    dv_ptrs = DV + offs_n[:, None] * stride_dvn + d_offs[None, :]
    grad_mask = (offs_n[:, None] < seqlen_k) & (d_offs[None, :] < headdim)

    if start_m == begin_m:
      dk_d = tl.trans(tl.dot(tl.trans(q), dS, out_dtype=tl.float32))
      tl.store(dk_ptrs, dk_d, mask=grad_mask, eviction_policy="evict_last")
      dv_d = tl.trans(tl.dot(tl.trans(do), P_drop.to(DTYPE), out_dtype=tl.float32))
      tl.store(dv_ptrs, dv_d, mask=grad_mask, eviction_policy="evict_last")
    else:
      dk_val = tl.load(dk_ptrs, mask=grad_mask, other=0., eviction_policy="evict_last")
      dk_d = tl.trans(tl.dot(tl.trans(q), dS, out_dtype=tl.float32))
      tl.store(dk_ptrs, dk_val + dk_d, mask=grad_mask, eviction_policy="evict_last")
      dv_val = tl.load(dv_ptrs, mask=grad_mask, other=0., eviction_policy="evict_last")
      dv_d = tl.trans(tl.dot(tl.trans(do), P_drop.to(DTYPE), out_dtype=tl.float32))
      tl.store(dv_ptrs, dv_val + dv_d, mask=grad_mask, eviction_policy="evict_last")


@triton.jit
def _ffpa_bwd_dkdv_sm90(
  desc_q: tl.tensor_descriptor,
  desc_k: tl.tensor_descriptor,
  desc_v: tl.tensor_descriptor,
  desc_do: tl.tensor_descriptor,
  DK: torch.Tensor,
  DV: torch.Tensor,
  LSE: torch.Tensor,
  D: torch.Tensor,
  AttnBias: torch.Tensor,
  GradAttnBias: torch.Tensor,
  softmax_scale: float,
  stride_dkb: int,
  stride_dkh: int,
  stride_dkn: int,
  stride_dvb: int,
  stride_dvh: int,
  stride_dvn: int,
  stride_bb: int,
  stride_bh: int,
  stride_bm: int,
  stride_bn: int,
  stride_gbb: int,
  stride_gbh: int,
  stride_gbm: int,
  stride_gbn: int,
  nheads: int,
  seqlen_q: int,
  seqlen_k: int,
  seqlen_q_rounded: int,
  headdim: int,
  dropout_p: float,
  philox_offset: int,
  IS_CAUSAL: tl.constexpr,
  HAS_ATTN_BIAS: tl.constexpr,
  HAS_DROPOUT: tl.constexpr,
  PHILOX_SEED: tl.constexpr,
  BIAS_REQUIRES_GRAD: tl.constexpr,
  GRAD_BIAS_NEEDS_REDUCTION: tl.constexpr,
  GRAD_BIAS_REDUCES_M: tl.constexpr,
  GRAD_BIAS_STORE_PARTIAL: tl.constexpr,
  BLOCK_HEADDIM: tl.constexpr,
  DTYPE: tl.constexpr,
  EVEN_M: tl.constexpr,
  EVEN_N: tl.constexpr,
  BLOCK_M: tl.constexpr,
  BLOCK_N: tl.constexpr,
  warp_specialize: tl.constexpr,
) -> None:
  """dK/dV half of the SM90 TMA Split-D backward kernel.

  This helper keeps the original cross-Q global load/add/store accumulation
  pattern. It is called from ``_ffpa_bwd_sm90_kernel_impl`` so the public SM90
  backward path remains a single Triton launch.
  """
  pid = tl.program_id(0)
  off_hb = tl.program_id(2)
  off_b = off_hb // nheads
  off_h = off_hb % nheads

  q_base_y = (off_b * nheads + off_h) * seqlen_q
  kv_base_y = (off_b * nheads + off_h) * seqlen_k

  DK += off_b * stride_dkb + off_h * stride_dkh
  DV += off_b * stride_dvb + off_h * stride_dvh
  D += off_hb * seqlen_q_rounded
  LSE += off_hb * seqlen_q_rounded

  if HAS_ATTN_BIAS:
    AttnBias += off_b * stride_bb + off_h * stride_bh
  if BIAS_REQUIRES_GRAD:
    GradAttnBias += off_b * stride_gbb + off_h * stride_gbh

  num_d_chunks = tl.cdiv(headdim, BLOCK_HEADDIM)

  # Part 1: dK / dV, pid as K-column block index.
  start_n = pid * BLOCK_N
  if start_n < seqlen_k:
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    num_block_m = tl.cdiv(seqlen_q, BLOCK_M)
    begin_m = 0 if not IS_CAUSAL else start_n // BLOCK_M * BLOCK_M
    k_offset_y = kv_base_y + start_n

    if warp_specialize:
      for start_m in tl.range(
        begin_m,
        num_block_m * BLOCK_M,
        BLOCK_M,
        disallow_acc_multi_buffer=True,
        flatten=True,
        warp_specialize=True,
      ):
        _ffpa_bwd_dkdv_sm90_q_block(
          desc_q, desc_k, desc_v, desc_do, DK, DV, LSE, D, AttnBias, GradAttnBias, softmax_scale, stride_dkn,
          stride_dvn, stride_bm, stride_bn, stride_gbm, stride_gbn, q_base_y, k_offset_y, start_m, begin_m, off_hb,
          offs_m, offs_n, offs_d, seqlen_q, seqlen_k, headdim, dropout_p, philox_offset, IS_CAUSAL, HAS_ATTN_BIAS,
          HAS_DROPOUT, PHILOX_SEED, BIAS_REQUIRES_GRAD, GRAD_BIAS_NEEDS_REDUCTION, GRAD_BIAS_REDUCES_M,
          GRAD_BIAS_STORE_PARTIAL, BLOCK_HEADDIM, DTYPE, EVEN_M, EVEN_N, BLOCK_M, BLOCK_N, num_d_chunks
        )
    else:
      for start_m in range(begin_m, num_block_m * BLOCK_M, BLOCK_M):
        _ffpa_bwd_dkdv_sm90_q_block(
          desc_q, desc_k, desc_v, desc_do, DK, DV, LSE, D, AttnBias, GradAttnBias, softmax_scale, stride_dkn,
          stride_dvn, stride_bm, stride_bn, stride_gbm, stride_gbn, q_base_y, k_offset_y, start_m, begin_m, off_hb,
          offs_m, offs_n, offs_d, seqlen_q, seqlen_k, headdim, dropout_p, philox_offset, IS_CAUSAL, HAS_ATTN_BIAS,
          HAS_DROPOUT, PHILOX_SEED, BIAS_REQUIRES_GRAD, GRAD_BIAS_NEEDS_REDUCTION, GRAD_BIAS_REDUCES_M,
          GRAD_BIAS_STORE_PARTIAL, BLOCK_HEADDIM, DTYPE, EVEN_M, EVEN_N, BLOCK_M, BLOCK_N, num_d_chunks
        )


@triton.jit
def _ffpa_bwd_dkdv_persist_sm90(
  desc_q: tl.tensor_descriptor,
  desc_k: tl.tensor_descriptor,
  desc_v: tl.tensor_descriptor,
  desc_do: tl.tensor_descriptor,
  DK: torch.Tensor,
  DV: torch.Tensor,
  LSE: torch.Tensor,
  D: torch.Tensor,
  AttnBias: torch.Tensor,
  GradAttnBias: torch.Tensor,
  softmax_scale: float,
  stride_dkb: int,
  stride_dkh: int,
  stride_dkn: int,
  stride_dvb: int,
  stride_dvh: int,
  stride_dvn: int,
  stride_bb: int,
  stride_bh: int,
  stride_bm: int,
  stride_bn: int,
  stride_gbb: int,
  stride_gbh: int,
  stride_gbm: int,
  stride_gbn: int,
  nheads: int,
  seqlen_q: int,
  seqlen_k: int,
  seqlen_q_rounded: int,
  headdim: int,
  dropout_p: float,
  philox_offset: int,
  IS_CAUSAL: tl.constexpr,
  HAS_ATTN_BIAS: tl.constexpr,
  HAS_DROPOUT: tl.constexpr,
  PHILOX_SEED: tl.constexpr,
  BIAS_REQUIRES_GRAD: tl.constexpr,
  GRAD_BIAS_NEEDS_REDUCTION: tl.constexpr,
  GRAD_BIAS_REDUCES_M: tl.constexpr,
  GRAD_BIAS_STORE_PARTIAL: tl.constexpr,
  BLOCK_HEADDIM: tl.constexpr,
  DTYPE: tl.constexpr,
  EVEN_M: tl.constexpr,
  EVEN_N: tl.constexpr,
  BLOCK_M: tl.constexpr,
  BLOCK_N: tl.constexpr,
  warp_specialize: tl.constexpr,
) -> None:
  """dK/dV half with fp32 register accumulation across Q blocks."""
  pid = tl.program_id(0)
  off_hb = tl.program_id(2)
  off_b = off_hb // nheads
  off_h = off_hb % nheads

  q_base_y = (off_b * nheads + off_h) * seqlen_q
  kv_base_y = (off_b * nheads + off_h) * seqlen_k

  DK += off_b * stride_dkb + off_h * stride_dkh
  DV += off_b * stride_dvb + off_h * stride_dvh
  D += off_hb * seqlen_q_rounded
  LSE += off_hb * seqlen_q_rounded

  if HAS_ATTN_BIAS:
    AttnBias += off_b * stride_bb + off_h * stride_bh
  if BIAS_REQUIRES_GRAD:
    GradAttnBias += off_b * stride_gbb + off_h * stride_gbh

  num_d_chunks = tl.cdiv(headdim, BLOCK_HEADDIM)
  start_n = pid * BLOCK_N
  if start_n < seqlen_k:
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    num_block_m = tl.cdiv(seqlen_q, BLOCK_M)
    begin_m = 0 if not IS_CAUSAL else start_n // BLOCK_M * BLOCK_M
    k_offset_y = kv_base_y + start_n

    for out_d_chunk in range(num_d_chunks):
      d_start_out = out_d_chunk * BLOCK_HEADDIM
      d_offs = d_start_out + offs_d
      # Persistent accumulators for dK/dV across all M blocks for this N block.
      # Avoid fp32->bf16/fp16->fp32 round-trips until the final dK/dV store at
      # the end of the N loop. The roud-trips will cause significant precision
      # loss when the IS_CAUSAL is True. This workaround is only suitable for
      # devices with large computation throughput like SM90+; SM<90, like Ada
      # and Ampere should use fp32 HBM storage for dK/dV to get good performance.
      dk_acc = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
      dv_acc = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)

      for start_m in tl.range(
        begin_m,
        num_block_m * BLOCK_M,
        BLOCK_M,
        disallow_acc_multi_buffer=warp_specialize,
        flatten=warp_specialize,
        warp_specialize=warp_specialize,
      ):
        offs_qm = start_m + offs_m
        q_offset_y = q_base_y + start_m

        S = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dP = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        for in_d_chunk in range(num_d_chunks):
          d_start = in_d_chunk * BLOCK_HEADDIM
          q = desc_q.load([q_offset_y, d_start])
          k = desc_k.load([k_offset_y, d_start])
          v = desc_v.load([k_offset_y, d_start])
          do = desc_do.load([q_offset_y, d_start])
          S = tl.dot(q, tl.trans(k), acc=S)
          dP = tl.dot(do, tl.trans(v), acc=dP)

        if not EVEN_N:
          S = tl.where(offs_n[None, :] < seqlen_k, S, float("-inf"))
        if not EVEN_M:
          m_mask = offs_qm < seqlen_q
          S = tl.where(m_mask[:, None], S, float("-inf"))
        if IS_CAUSAL:
          S = tl.where(offs_qm[:, None] >= (offs_n[None, :]), S, float("-inf"))
        S = S * softmax_scale
        if HAS_ATTN_BIAS:
          bias = tl.load(
            AttnBias + offs_qm[:, None] * stride_bm + offs_n[None, :] * stride_bn,
            mask=(offs_qm[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k),
            other=0.0,
          )
          S += bias
        if EVEN_M:
          lse_i = tl.load(LSE + offs_qm)
        else:
          lse_i = tl.load(LSE + offs_qm, mask=offs_qm < seqlen_q, other=0.0)
        P = tl.exp(S - lse_i[:, None])
        dropout_mult = _dropout_multiplier(
          off_hb,
          offs_qm,
          offs_n,
          seqlen_q,
          seqlen_k,
          dropout_p,
          PHILOX_SEED,
          philox_offset,
          HAS_DROPOUT,
        )
        dP = dP * dropout_mult
        P_drop = P * dropout_mult
        if EVEN_M:
          Di = tl.load(D + offs_qm)
        else:
          Di = tl.load(D + offs_qm, mask=offs_qm < seqlen_q, other=0.0)
        if BIAS_REQUIRES_GRAD:
          dBias = P * (dP - Di[:, None])
          grad_bias_mask = (offs_qm[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k)
          if out_d_chunk == 0:
            if GRAD_BIAS_REDUCES_M:
              m_block = start_m // BLOCK_M
              grad_bias_ptrs = GradAttnBias + m_block * stride_gbm + offs_n * stride_gbn
              grad_bias = tl.sum(tl.where(grad_bias_mask, dBias, 0.0), axis=0)
              if GRAD_BIAS_STORE_PARTIAL:
                tl.store(grad_bias_ptrs, grad_bias, mask=offs_n < seqlen_k)
              else:
                tl.atomic_add(grad_bias_ptrs, grad_bias, sem="relaxed", mask=offs_n < seqlen_k)
            elif GRAD_BIAS_NEEDS_REDUCTION:
              grad_bias_ptrs = GradAttnBias + offs_qm[:, None] * stride_gbm + offs_n[None, :] * stride_gbn
              tl.atomic_add(grad_bias_ptrs, dBias, sem="relaxed", mask=grad_bias_mask)
            else:
              grad_bias_ptrs = GradAttnBias + offs_qm[:, None] * stride_gbm + offs_n[None, :] * stride_gbn
              tl.store(grad_bias_ptrs, dBias, mask=grad_bias_mask)
          dS = (dBias * softmax_scale).to(DTYPE)
        else:
          dS = (P * (dP - Di[:, None]) * softmax_scale).to(DTYPE)
        if not EVEN_M:
          dS = tl.where(m_mask[:, None], dS, 0.0)
          P_drop = tl.where(m_mask[:, None], P_drop, 0.0)

        q = desc_q.load([q_offset_y, d_start_out])
        do = desc_do.load([q_offset_y, d_start_out])
        dk_acc += tl.trans(tl.dot(tl.trans(q), dS, out_dtype=tl.float32))
        dv_acc += tl.trans(tl.dot(tl.trans(do), P_drop.to(DTYPE), out_dtype=tl.float32))

      grad_mask = (offs_n[:, None] < seqlen_k) & (d_offs[None, :] < headdim)
      dk_ptrs = DK + offs_n[:, None] * stride_dkn + d_offs[None, :]
      dv_ptrs = DV + offs_n[:, None] * stride_dvn + d_offs[None, :]
      tl.store(dk_ptrs, dk_acc, mask=grad_mask, eviction_policy="evict_last")
      tl.store(dv_ptrs, dv_acc, mask=grad_mask, eviction_policy="evict_last")


@triton.jit
def _ffpa_bwd_dq_sm90(
  desc_q: tl.tensor_descriptor,
  desc_k: tl.tensor_descriptor,
  desc_v: tl.tensor_descriptor,
  desc_do: tl.tensor_descriptor,
  DQ: torch.Tensor,
  LSE: torch.Tensor,
  D: torch.Tensor,
  AttnBias: torch.Tensor,
  softmax_scale: float,
  stride_dqb: int,
  stride_dqh: int,
  stride_dqm: int,
  stride_bb: int,
  stride_bh: int,
  stride_bm: int,
  stride_bn: int,
  nheads: int,
  seqlen_q: int,
  seqlen_k: int,
  seqlen_q_rounded: int,
  headdim: int,
  dropout_p: float,
  philox_offset: int,
  IS_CAUSAL: tl.constexpr,
  HAS_ATTN_BIAS: tl.constexpr,
  HAS_DROPOUT: tl.constexpr,
  PHILOX_SEED: tl.constexpr,
  BLOCK_HEADDIM: tl.constexpr,
  DTYPE: tl.constexpr,
  EVEN_N: tl.constexpr,
  BLOCK_M: tl.constexpr,
  BLOCK_N: tl.constexpr,
  warp_specialize: tl.constexpr,
) -> None:
  """dQ half of the SM90 TMA Split-D backward kernel."""
  pid = tl.program_id(0)
  off_hb = tl.program_id(2)
  off_b = off_hb // nheads
  off_h = off_hb % nheads

  q_base_y = (off_b * nheads + off_h) * seqlen_q
  kv_base_y = (off_b * nheads + off_h) * seqlen_k

  DQ += off_b * stride_dqb + off_h * stride_dqh
  D += off_hb * seqlen_q_rounded
  LSE += off_hb * seqlen_q_rounded

  if HAS_ATTN_BIAS:
    AttnBias += off_b * stride_bb + off_h * stride_bh

  num_d_chunks = tl.cdiv(headdim, BLOCK_HEADDIM)

  start_m = pid * BLOCK_M
  if start_m < seqlen_q:
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    q_offset_y = q_base_y + start_m

    num_block_n = tl.cdiv(seqlen_k, BLOCK_N)
    end_n_k = start_m + BLOCK_M if IS_CAUSAL else num_block_n * BLOCK_N

    for start_n_k in tl.range(
      0,
      end_n_k,
      BLOCK_N,
      flatten=warp_specialize,
      warp_specialize=warp_specialize,
    ):
      offs_nk = start_n_k + offs_n
      k_offset_y = kv_base_y + start_n_k

      S_qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
      dP_qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

      for d_chunk in range(num_d_chunks):
        d_start = d_chunk * BLOCK_HEADDIM
        q = desc_q.load([q_offset_y, d_start])
        k = desc_k.load([k_offset_y, d_start])
        v = desc_v.load([k_offset_y, d_start])
        do = desc_do.load([q_offset_y, d_start])
        S_qk = tl.dot(q, tl.trans(k), acc=S_qk)
        dP_qk = tl.dot(do, tl.trans(v), acc=dP_qk)

      if not EVEN_N:
        S_qk = tl.where(offs_nk[None, :] < seqlen_k, S_qk, float("-inf"))
      if IS_CAUSAL:
        S_qk = tl.where(offs_m[:, None] >= (offs_nk[None, :]), S_qk, float("-inf"))
      S_qk = S_qk * softmax_scale
      if HAS_ATTN_BIAS:
        bias = tl.load(
          AttnBias + offs_m[:, None] * stride_bm + offs_nk[None, :] * stride_bn,
          mask=(offs_m[:, None] < seqlen_q) & (offs_nk[None, :] < seqlen_k),
          other=0.0,
        )
        S_qk += bias
      lse_i = tl.load(LSE + offs_m)
      P_qk = tl.exp(S_qk - lse_i[:, None])
      dropout_mult_qk = _dropout_multiplier(
        off_hb,
        offs_m,
        offs_nk,
        seqlen_q,
        seqlen_k,
        dropout_p,
        PHILOX_SEED,
        philox_offset,
        HAS_DROPOUT,
      )
      dP_qk = dP_qk * dropout_mult_qk
      Di = tl.load(D + offs_m)
      dS_qk = (P_qk * (dP_qk - Di[:, None]) * softmax_scale).to(DTYPE)

      for d_chunk in range(num_d_chunks):
        d_start = d_chunk * BLOCK_HEADDIM
        d_offs = d_start + offs_d
        k = desc_k.load([k_offset_y, d_start])
        dq_ptrs = DQ + offs_m[:, None] * stride_dqm + d_offs[None, :]
        dq_mask = (offs_m[:, None] < seqlen_q) & (d_offs[None, :] < headdim)
        if start_n_k == 0:
          dq_d = tl.dot(dS_qk, k, out_dtype=tl.float32)
          tl.store(dq_ptrs, dq_d, mask=dq_mask, eviction_policy="evict_last")
        else:
          dq_val = tl.load(dq_ptrs, mask=dq_mask, other=0., eviction_policy="evict_last")
          dq_d = tl.dot(dS_qk, k, out_dtype=tl.float32)
          tl.store(dq_ptrs, dq_val + dq_d, mask=dq_mask, eviction_policy="evict_last")


@triton.heuristics(_SM90_BWD_HEURISTICS)
@triton.jit
def _ffpa_bwd_sm90_kernel_impl(
  desc_q: tl.tensor_descriptor,
  desc_k: tl.tensor_descriptor,
  desc_v: tl.tensor_descriptor,
  desc_do: tl.tensor_descriptor,
  DQ: torch.Tensor,
  DK: torch.Tensor,
  DV: torch.Tensor,
  LSE: torch.Tensor,
  D: torch.Tensor,
  AttnBias: torch.Tensor,
  GradAttnBias: torch.Tensor,
  softmax_scale: float,
  stride_dqb: int,
  stride_dqh: int,
  stride_dqm: int,
  stride_dkb: int,
  stride_dkh: int,
  stride_dkn: int,
  stride_dvb: int,
  stride_dvh: int,
  stride_dvn: int,
  stride_bb: int,
  stride_bh: int,
  stride_bm: int,
  stride_bn: int,
  stride_gbb: int,
  stride_gbh: int,
  stride_gbm: int,
  stride_gbn: int,
  nheads: int,
  seqlen_q: int,
  seqlen_k: int,
  autotune_seqlen_q_bucket: int,
  autotune_seqlen_k_bucket: int,
  autotune_causal_key: int,
  autotune_dtype_key: int,
  seqlen_q_rounded: int,
  headdim: int,
  dropout_p: float,
  philox_offset: int,
  IS_CAUSAL: tl.constexpr,
  HAS_ATTN_BIAS: tl.constexpr,
  HAS_DROPOUT: tl.constexpr,
  PHILOX_SEED: tl.constexpr,
  BIAS_REQUIRES_GRAD: tl.constexpr,
  GRAD_BIAS_NEEDS_REDUCTION: tl.constexpr,
  GRAD_BIAS_REDUCES_M: tl.constexpr,
  GRAD_BIAS_STORE_PARTIAL: tl.constexpr,
  PERSIST_DKDV_ACC: tl.constexpr,
  BLOCK_HEADDIM: tl.constexpr,
  DTYPE: tl.constexpr,
  EVEN_M: tl.constexpr,
  EVEN_N: tl.constexpr,
  BLOCK_M: tl.constexpr,
  BLOCK_N: tl.constexpr,
  warp_specialize: tl.constexpr,
) -> None:
  """TMA-descriptor variant of the main Split-D FFPA backward kernel.

  Keep this kernel at Triton's default ``num_ctas=1``. The fused backward
  body contains multiple ``tt.dot`` ops across the dK / dV and dQ phases.
  This is a kernel / Triton ``FuncOp`` level limitation, not a loop-local
  limitation: the current PlanCTA planner can only see one ``DotOp`` per
  kernel when ``num_ctas=2``. Triton 3.6/3.7 runs the NVIDIA CTA planning
  pass (``TritonGPUPlanCTAPass``, pipeline name
  ``triton-nvidia-gpu-plan-cta``) and the second ``DotOp`` hits the
  PlanCTA.cpp assertion ``!tiled && "CTA tiling is already determined"``.
  """
  # Keys for autotune and heuristics lookups.
  _ = autotune_seqlen_q_bucket
  _ = autotune_seqlen_k_bucket
  _ = autotune_causal_key
  _ = autotune_dtype_key

  if PERSIST_DKDV_ACC:
    _ffpa_bwd_dkdv_persist_sm90(
      desc_q, desc_k, desc_v, desc_do, DK, DV, LSE, D, AttnBias, GradAttnBias, softmax_scale, stride_dkb, stride_dkh,
      stride_dkn, stride_dvb, stride_dvh, stride_dvn, stride_bb, stride_bh, stride_bm, stride_bn, stride_gbb,
      stride_gbh, stride_gbm, stride_gbn, nheads, seqlen_q, seqlen_k, seqlen_q_rounded, headdim, dropout_p,
      philox_offset, IS_CAUSAL, HAS_ATTN_BIAS, HAS_DROPOUT, PHILOX_SEED, BIAS_REQUIRES_GRAD, GRAD_BIAS_NEEDS_REDUCTION,
      GRAD_BIAS_REDUCES_M, GRAD_BIAS_STORE_PARTIAL, BLOCK_HEADDIM, DTYPE, EVEN_M, EVEN_N, BLOCK_M, BLOCK_N,
      warp_specialize
    )
  else:
    _ffpa_bwd_dkdv_sm90(
      desc_q, desc_k, desc_v, desc_do, DK, DV, LSE, D, AttnBias, GradAttnBias, softmax_scale, stride_dkb, stride_dkh,
      stride_dkn, stride_dvb, stride_dvh, stride_dvn, stride_bb, stride_bh, stride_bm, stride_bn, stride_gbb,
      stride_gbh, stride_gbm, stride_gbn, nheads, seqlen_q, seqlen_k, seqlen_q_rounded, headdim, dropout_p,
      philox_offset, IS_CAUSAL, HAS_ATTN_BIAS, HAS_DROPOUT, PHILOX_SEED, BIAS_REQUIRES_GRAD, GRAD_BIAS_NEEDS_REDUCTION,
      GRAD_BIAS_REDUCES_M, GRAD_BIAS_STORE_PARTIAL, BLOCK_HEADDIM, DTYPE, EVEN_M, EVEN_N, BLOCK_M, BLOCK_N,
      warp_specialize
    )
  _ffpa_bwd_dq_sm90(
    desc_q, desc_k, desc_v, desc_do, DQ, LSE, D, AttnBias, softmax_scale, stride_dqb, stride_dqh, stride_dqm, stride_bb,
    stride_bh, stride_bm, stride_bn, nheads, seqlen_q, seqlen_k, seqlen_q_rounded, headdim, dropout_p, philox_offset,
    IS_CAUSAL, HAS_ATTN_BIAS, HAS_DROPOUT, PHILOX_SEED, BLOCK_HEADDIM, DTYPE, EVEN_N, BLOCK_M, BLOCK_N, warp_specialize
  )


_SM90_BWD_DEFAULT_CONFIG = {
  "BLOCK_M": 128,
  "BLOCK_N": 64,
  "BLOCK_HEADDIM": 64,
  "warp_specialize": False,
  "num_warps": 8,
  "num_stages": 2,
}

_SM90_BWD_PERSIST_DKDV_DEFAULT_CONFIG = {
  "BLOCK_M": 128,
  "BLOCK_N": 64,
  "BLOCK_HEADDIM": 128,
  "warp_specialize": False,
  "num_warps": 8,
  "num_stages": 2,
}


def _default_bwd_sm90_config(enable_persist_dkdv: bool) -> dict:
  """Return the fixed SM90 backward launch config for the selected dKdV mode."""
  if enable_persist_dkdv:
    return dict(_SM90_BWD_PERSIST_DKDV_DEFAULT_CONFIG)
  return dict(_SM90_BWD_DEFAULT_CONFIG)


def _gen_bwd_sm90_autotune_configs(
  headdim: int = 512,
  autotune_mode: str = "max",
  enable_ws: bool = False,
  enable_persist_dkdv: bool = False,
) -> list[triton.Config]:
  """Generate autotune configs for the SM90 TMA backward main kernel."""
  del headdim
  # fast: 2*1*2*1*1 = 4 configs; max: 2*2*2*2*2 = 32 configs
  configs = []
  if enable_persist_dkdv:
    block_headdim_candidates = [128] if autotune_mode == "fast" else [64, 128]
  else:
    block_headdim_candidates = [64] if autotune_mode == "fast" else [64, 128]

  for block_m in [64, 128]:
    for block_n in ([64] if autotune_mode == "fast" else [64, 128]):
      for block_headdim in block_headdim_candidates:
        for num_warps in ([4] if autotune_mode == "fast" else [4, 8]):
          for num_stages in ([2] if autotune_mode == "fast" else [2, 3]):
            configs.append(
              triton.Config(
                {
                  "BLOCK_M": block_m,
                  "BLOCK_N": block_n,
                  "BLOCK_HEADDIM": block_headdim,
                  "warp_specialize": enable_ws,
                },
                num_warps=num_warps,
                num_stages=num_stages,
                pre_hook=_sm90_bwd_host_descriptor_pre_hook,
              )
            )
  return configs


_ffpa_bwd_sm90_autotune_cache: dict[tuple[int, str, str, bool, bool, bool], callable] = {}


def _get_bwd_sm90_autotune(
  headdim: int,
  autotune_mode: str,
  dtype: str,
  bias_requires_grad: bool,
  enable_ws: bool = False,
  enable_persist_dkdv: bool = False,
):
  """Return an autotune wrapper for the SM90 TMA backward main kernel."""
  cache_key = (headdim, autotune_mode, dtype, bias_requires_grad, enable_ws, enable_persist_dkdv)
  if cache_key not in _ffpa_bwd_sm90_autotune_cache:
    reset_args = []
    if bias_requires_grad:
      reset_args.append("GradAttnBias")
    _ffpa_bwd_sm90_autotune_cache[cache_key] = triton.autotune(
      configs=_gen_bwd_sm90_autotune_configs(
        headdim=headdim,
        autotune_mode=autotune_mode,
        enable_ws=enable_ws,
        enable_persist_dkdv=enable_persist_dkdv,
      ),
      key=[
        "autotune_seqlen_q_bucket",
        "autotune_seqlen_k_bucket",
        "headdim",
        "autotune_causal_key",
        "autotune_dtype_key",
      ],
      reset_to_zero=reset_args,
      cache_results=True,
    )(_ffpa_bwd_sm90_kernel_impl)
  return _ffpa_bwd_sm90_autotune_cache[cache_key]


def is_sm90_tma_backward_supported(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  do: torch.Tensor,
  dq: torch.Tensor,
  dk: torch.Tensor,
  dv: torch.Tensor,
  *,
  seqlen_q: int,
) -> bool:
  """Return whether the experimental SM90 TMA backward main path may run."""
  if seqlen_q < 8:
    return False
  if not q.is_cuda:
    return False
  if torch.cuda.get_device_capability(q.device)[0] < 9:
    return False
  if q.dtype not in (torch.float16, torch.bfloat16):
    return False
  return all(tensor.stride(-1) == 1 for tensor in (q, k, v, do, dq, dk, dv))


def _ffpa_bwd_sm90_make_descs(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  do: torch.Tensor,
) -> tuple[TensorDescriptor, TensorDescriptor, TensorDescriptor, TensorDescriptor]:
  """Create flattened TMA descriptors for backward input tensors."""
  batch, nheads, seqlen_q, headdim = q.shape
  _, _, seqlen_k, _ = k.shape
  dummy_block = [1, 1]

  def _make_tensor_desc(x: torch.Tensor, shape: list[int]) -> TensorDescriptor:
    return TensorDescriptor(x, shape=shape, strides=[shape[1], 1], block_shape=dummy_block)

  y_dim_q = batch * nheads * seqlen_q
  y_dim_kv = batch * nheads * seqlen_k
  return (
    _make_tensor_desc(q, [y_dim_q, headdim]),
    _make_tensor_desc(k, [y_dim_kv, headdim]),
    _make_tensor_desc(v, [y_dim_kv, headdim]),
    _make_tensor_desc(do, [y_dim_q, headdim]),
  )


def _ffpa_bwd_sm90_prepare_descs(
  desc_q: TensorDescriptor,
  desc_k: TensorDescriptor,
  desc_v: TensorDescriptor,
  desc_do: TensorDescriptor,
  launch_config: dict,
) -> None:
  """Set descriptor block shapes for a fixed-config launch."""
  desc_q.block_shape = [launch_config["BLOCK_M"], launch_config["BLOCK_HEADDIM"]]
  desc_k.block_shape = [launch_config["BLOCK_N"], launch_config["BLOCK_HEADDIM"]]
  desc_v.block_shape = [launch_config["BLOCK_N"], launch_config["BLOCK_HEADDIM"]]
  desc_do.block_shape = [launch_config["BLOCK_M"], launch_config["BLOCK_HEADDIM"]]


def lookup_bwd_sm90_persistent_config(
  *,
  q: torch.Tensor,
  seqlen_q: int,
  seqlen_k: int,
  headdim: int,
  autotune_mode: str,
  causal: bool,
  bias_grad: bool,
  grad_kv_storage_dtype: str | None,
  has_attn_bias: bool,
  has_dropout: bool,
  nheads_q: int,
  nheads_kv: int,
  enable_ws: bool,
  enable_persist_dkdv: bool = False,
) -> dict | None:
  """Lookup a persisted SM90 TMA backward config."""
  return lookup_persistent_config(
    PersistentConfigRequest(
      direction="backward",
      kernel="bwd_sm90_generic_persist_dkdv" if enable_persist_dkdv else "bwd_sm90_generic",
      autotune_mode=autotune_mode,
      dtype=dtype_name(q.dtype),
      headdim=headdim,
      seqlen_q=seqlen_q,
      seqlen_k=seqlen_k,
      causal=causal,
      bias_grad=bias_grad,
      grad_kv_storage_dtype=grad_kv_storage_dtype,
      has_attn_bias=has_attn_bias,
      has_dropout=has_dropout,
      enable_tma=True,
      enable_ws=enable_ws,
      nheads_q=nheads_q,
      nheads_kv=nheads_kv,
      device_index=q.device.index,
    )
  )


def _ffpa_attn_backward_sm90_impl(
  do: torch.Tensor,
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  o: torch.Tensor,
  lse: torch.Tensor,
  attn_bias: torch.Tensor | None,
  dq: torch.Tensor,
  dk: torch.Tensor,
  dv: torch.Tensor,
  grad_attn_bias: torch.Tensor | None,
  causal: bool = False,
  softmax_scale: float | None = None,
  autotune: bool = False,
  autotune_mode: str = "fast",
  preprocess_d_chunk: bool = False,
  original_nheads_kv: int | None = None,
  dropout_p: float = 0.0,
  philox_seed: int = 0,
  philox_offset: int = 0,
  enable_ws: bool = False,
  enable_persist_dkdv: bool = False,
) -> None:
  """Run the SM90+ TMA backward main path.

  This is the backward counterpart of ``_ffpa_attn_forward_sm90_tma_impl``:
  callers hand it the same high-level tensors and metadata as the generic
  backward implementation, while descriptor creation, TMA allocator setup,
  persistent config lookup, and SM90 kernel launch details stay in this module.
  """
  if do.stride(-1) != 1:
    do = do.contiguous()
  batch, nheads, seqlen_q, headdim = q.shape
  _, _, seqlen_k, _ = k.shape
  original_nheads_kv = original_nheads_kv or nheads
  softmax_scale = softmax_scale or (1.0 / (headdim**0.5))
  seqlen_q_rounded = lse.shape[-1]
  autotune_seqlen_q_bucket = autotune_seqlen_key(seqlen_q, autotune_mode)
  autotune_seqlen_k_bucket = autotune_seqlen_key(seqlen_k, autotune_mode)
  autotune_causal_key = int(causal)
  has_attn_bias = attn_bias is not None
  attn_bias_in = attn_bias if attn_bias is not None else q
  bias_strides = _attn_bias_broadcast_strides(attn_bias, batch, nheads, seqlen_q, seqlen_k)
  bias_requires_grad = grad_attn_bias is not None
  grad_bias_needs_reduction = _attn_bias_grad_needs_reduction(
    grad_attn_bias,
    batch,
    nheads,
    seqlen_q,
    seqlen_k,
  )
  grad_bias_reduces_m = _attn_bias_grad_reduces_query(grad_attn_bias, seqlen_q)
  use_key_bias_grad_reduction = _attn_bias_grad_is_key_bias(grad_attn_bias, seqlen_q, seqlen_k)
  has_dropout = dropout_p > 0.0
  runtime_dtype = dtype_name(q.dtype)
  if dk.dtype == torch.float32 or dv.dtype == torch.float32:
    grad_kv_storage_dtype = "fp32"
  elif (dk.dtype == torch.float16 or dv.dtype == torch.float16) and q.dtype != torch.float16:
    grad_kv_storage_dtype = "fp16"
  else:
    grad_kv_storage_dtype = None
  DTYPE = tl.float16 if q.dtype == torch.float16 else tl.bfloat16
  autotune_dtype_key = 1 if q.dtype == torch.bfloat16 else 0

  if use_key_bias_grad_reduction:
    num_m_blocks_for_bias = triton.cdiv(seqlen_q, 64)
    partial_grad_bias = torch.zeros(
      (batch * nheads, num_m_blocks_for_bias, seqlen_k),
      dtype=torch.float32,
      device=q.device,
    )
    grad_attn_bias_in = partial_grad_bias
    grad_bias_strides = (
      nheads * partial_grad_bias.stride(0),
      partial_grad_bias.stride(0),
      partial_grad_bias.stride(1),
      partial_grad_bias.stride(2),
    )
  else:
    partial_grad_bias = None
    grad_attn_bias_in = grad_attn_bias if grad_attn_bias is not None else q
    grad_bias_strides = _attn_bias_broadcast_strides(
      grad_attn_bias,
      batch,
      nheads,
      seqlen_q,
      seqlen_k,
    )

  block_headdim_delta = 64 if preprocess_d_chunk else max(64, triton.next_power_of_2(headdim))
  delta = torch.empty_like(lse)

  def pre_grid(meta: dict) -> tuple[int, int]:
    return (triton.cdiv(seqlen_q, meta["BLOCK_M"]), batch * nheads)

  pre_args = (
    o,
    do,
    delta,
    o.stride(0),
    o.stride(1),
    o.stride(2),
    do.stride(0),
    do.stride(1),
    do.stride(2),
    nheads,
    seqlen_q,
    autotune_seqlen_q_bucket,
    seqlen_q_rounded,
    headdim,
  )
  if autotune:
    _get_pre_autotune(preprocess_d_chunk, autotune_mode, runtime_dtype)[pre_grid](*pre_args)
  else:
    persisted_pre_config = lookup_persistent_config(
      PersistentConfigRequest(
        direction="backward",
        kernel="bwd_preproc",
        autotune_mode=autotune_mode,
        dtype=runtime_dtype,
        headdim=headdim,
        seqlen_q=seqlen_q,
        preprocess_d_chunk=preprocess_d_chunk,
        device_index=q.device.index,
      )
    )
    pre_config = _normalize_bwd_pre_config(
      persisted_pre_config,
      preprocess_d_chunk=preprocess_d_chunk,
      block_headdim_delta=block_headdim_delta,
    )
    _ffpa_bwd_pre[(triton.cdiv(seqlen_q, pre_config["BLOCK_M"]), batch * nheads)](
      *pre_args,
      **pre_config,
    )

  def grid(meta: dict) -> tuple[int, ...]:
    return (
      max(triton.cdiv(seqlen_k, meta["BLOCK_N"]), triton.cdiv(seqlen_q, meta["BLOCK_M"])),
      1,
      batch * nheads,
    )

  desc_q, desc_k, desc_v, desc_do = _ffpa_bwd_sm90_make_descs(q, k, v, do)

  def _tma_alloc_fn(size: int, align: int, _):
    return torch.empty(size, dtype=torch.int8, device=q.device)

  triton.set_allocator(_tma_alloc_fn)

  kernel_args = (
    desc_q,
    desc_k,
    desc_v,
    desc_do,
    dq,
    dk,
    dv,
    lse,
    delta,
    attn_bias_in,
    grad_attn_bias_in,
    softmax_scale,
    dq.stride(0),
    dq.stride(1),
    dq.stride(2),
    dk.stride(0),
    dk.stride(1),
    dk.stride(2),
    dv.stride(0),
    dv.stride(1),
    dv.stride(2),
    bias_strides[0],
    bias_strides[1],
    bias_strides[2],
    bias_strides[3],
    grad_bias_strides[0],
    grad_bias_strides[1],
    grad_bias_strides[2],
    grad_bias_strides[3],
    nheads,
    seqlen_q,
    seqlen_k,
    autotune_seqlen_q_bucket,
    autotune_seqlen_k_bucket,
    autotune_causal_key,
    autotune_dtype_key,
    seqlen_q_rounded,
    headdim,
    dropout_p,
    philox_offset,
  )
  kernel_meta = dict(
    IS_CAUSAL=causal,
    HAS_ATTN_BIAS=has_attn_bias,
    HAS_DROPOUT=has_dropout,
    PHILOX_SEED=philox_seed,
    BIAS_REQUIRES_GRAD=bias_requires_grad,
    GRAD_BIAS_NEEDS_REDUCTION=grad_bias_needs_reduction,
    GRAD_BIAS_REDUCES_M=grad_bias_reduces_m,
    GRAD_BIAS_STORE_PARTIAL=use_key_bias_grad_reduction,
    PERSIST_DKDV_ACC=enable_persist_dkdv,
    DTYPE=DTYPE,
  )

  if autotune:
    _get_bwd_sm90_autotune(
      headdim,
      autotune_mode,
      runtime_dtype,
      bias_requires_grad,
      enable_ws=enable_ws,
      enable_persist_dkdv=enable_persist_dkdv,
    )[grid](*kernel_args, **kernel_meta)
  else:
    launch_config = _default_bwd_sm90_config(enable_persist_dkdv)
    persisted_config = lookup_bwd_sm90_persistent_config(
      q=q,
      seqlen_q=seqlen_q,
      seqlen_k=seqlen_k,
      headdim=headdim,
      autotune_mode=autotune_mode,
      causal=causal,
      bias_grad=bias_requires_grad,
      grad_kv_storage_dtype=grad_kv_storage_dtype,
      has_attn_bias=has_attn_bias,
      has_dropout=has_dropout,
      nheads_q=nheads,
      nheads_kv=original_nheads_kv,
      enable_ws=enable_ws,
      enable_persist_dkdv=enable_persist_dkdv,
    )
    if persisted_config is not None:
      launch_config.update(persisted_config)
    launch_config["warp_specialize"] = bool(enable_ws)
    if enable_ws:
      launch_config["num_stages"] = 2
    _ffpa_bwd_sm90_prepare_descs(desc_q, desc_k, desc_v, desc_do, launch_config)
    _ffpa_bwd_sm90_kernel_impl[grid](*kernel_args, **kernel_meta, **launch_config)

  if use_key_bias_grad_reduction:
    key_bias_block_n = 64
    _ffpa_bwd_key_bias_grad_reduce_kernel[(triton.cdiv(seqlen_k, key_bias_block_n), )](
      partial_grad_bias,
      grad_attn_bias,
      seqlen_k,
      partial_grad_bias.numel() // seqlen_k,
      grad_attn_bias.stride(3),
      BLOCK_N=key_bias_block_n,
      BLOCK_R=64,
      num_warps=8,
      num_stages=2,
    )

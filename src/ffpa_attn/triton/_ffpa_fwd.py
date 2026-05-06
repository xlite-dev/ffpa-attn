"""FFPA Attention Forward (Split-D) — Triton implementation.

This module provides a single Triton forward kernel for large-head-dim FFPA
prefill attention.  The program mapping follows FlashAttention v2: one program
owns a Q-row block for one batch/query-head pair.  Inside that program, the
large head dimension is processed in chunks so D=320/512 can be handled without
materialising the full attention matrix.

The saved LSE uses the natural logarithm convention expected by the existing
Triton backward kernel: ``lse = log(sum(exp(score)))`` where
``score = softmax_scale * (Q @ K.T)`` after masking.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


def _gen_fwd_autotune_configs() -> list[triton.Config]:
  """Generate autotune configs for the single FFPA Triton forward kernel.

  The search space mirrors the backward module style: tune Q/K tile sizes,
  D-chunk sizes, warp count, and pipeline depth.  ``BLOCK_HEADDIM_O`` controls
  how much of the output head dimension is accumulated at once inside the one
  forward kernel; it is not a kernel-version selector.
  """
  configs = []
  for block_m in [16, 32, 64]:
    for block_n in [32, 64, 128]:
      for block_headdim_qk in [64, 128, 256]:
        for block_headdim_o in [64, 128]:
          for num_d_acc in [1, 2, 4]:
            for num_warps in [4, 8]:
              for num_stages in [2, 3]:
                if block_m * block_headdim_o * num_d_acc > 8192:
                  continue
                configs.append(
                  triton.Config(
                    {
                      "BLOCK_M": block_m,
                      "BLOCK_N": block_n,
                      "BLOCK_HEADDIM_QK": block_headdim_qk,
                      "BLOCK_HEADDIM_O": block_headdim_o,
                      "NUM_D_ACC": num_d_acc,
                    },
                    num_warps=num_warps,
                    num_stages=num_stages,
                  )
                )
  return configs


_FFPA_FWD_AUTOTUNE_CONFIGS = _gen_fwd_autotune_configs()
_FFPA_FWD_HEURISTICS = {
  "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
  "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
}


@triton.heuristics(_FFPA_FWD_HEURISTICS)
@triton.jit
def _ffpa_fwd_kernel_impl(
  Q: torch.Tensor,
  K: torch.Tensor,
  V: torch.Tensor,
  O: torch.Tensor,
  LSE: torch.Tensor,
  softmax_scale: float,
  stride_qb: int,
  stride_qh: int,
  stride_qm: int,
  stride_kb: int,
  stride_kh: int,
  stride_kn: int,
  stride_vb: int,
  stride_vh: int,
  stride_vn: int,
  stride_ob: int,
  stride_oh: int,
  stride_om: int,
  nheads_q: int,
  nheads_kv: int,
  seqlen_q: int,
  seqlen_k: int,
  seqlen_q_rounded: int,
  headdim: int,
  IS_CAUSAL: tl.constexpr,
  DTYPE: tl.constexpr,
  EVEN_M: tl.constexpr,
  EVEN_N: tl.constexpr,
  BLOCK_M: tl.constexpr,
  BLOCK_N: tl.constexpr,
  BLOCK_HEADDIM_QK: tl.constexpr,
  BLOCK_HEADDIM_O: tl.constexpr,
  NUM_D_ACC: tl.constexpr,
) -> None:
  """Single-kernel Split-D FFPA forward."""
  start_m = tl.program_id(0)
  off_hb = tl.program_id(1)
  off_b = off_hb // nheads_q
  off_hq = off_hb % nheads_q
  group_size = nheads_q // nheads_kv
  off_hkv = off_hq // group_size

  Q += off_b * stride_qb + off_hq * stride_qh
  K += off_b * stride_kb + off_hkv * stride_kh
  V += off_b * stride_vb + off_hkv * stride_vh
  O += off_b * stride_ob + off_hq * stride_oh
  LSE += off_hb * seqlen_q_rounded

  offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
  offs_n = tl.arange(0, BLOCK_N)
  offs_d_qk = tl.arange(0, BLOCK_HEADDIM_QK)
  offs_d_o = tl.arange(0, BLOCK_HEADDIM_O)

  num_qk_d_chunks = tl.cdiv(headdim, BLOCK_HEADDIM_QK)
  num_o_d_chunks = tl.cdiv(headdim, BLOCK_HEADDIM_O)
  num_o_groups = tl.cdiv(num_o_d_chunks, NUM_D_ACC)
  kv_offset = seqlen_k - seqlen_q

  for o_group in range(num_o_groups):
    o_chunk0 = o_group * NUM_D_ACC
    o_d0 = o_chunk0 * BLOCK_HEADDIM_O + offs_d_o
    o_d1 = (o_chunk0 + 1) * BLOCK_HEADDIM_O + offs_d_o
    o_d2 = (o_chunk0 + 2) * BLOCK_HEADDIM_O + offs_d_o
    o_d3 = (o_chunk0 + 3) * BLOCK_HEADDIM_O + offs_d_o

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc0 = tl.zeros([BLOCK_M, BLOCK_HEADDIM_O], dtype=tl.float32)
    acc1 = tl.zeros([BLOCK_M, BLOCK_HEADDIM_O], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK_M, BLOCK_HEADDIM_O], dtype=tl.float32)
    acc3 = tl.zeros([BLOCK_M, BLOCK_HEADDIM_O], dtype=tl.float32)

    for start_n in range(0, seqlen_k, BLOCK_N):
      start_n = tl.multiple_of(start_n, BLOCK_N)
      offs_kv = start_n + offs_n

      scores = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
      for qk_d_chunk in range(num_qk_d_chunks):
        qk_d_start = qk_d_chunk * BLOCK_HEADDIM_QK
        qk_d = qk_d_start + offs_d_qk
        q = tl.load(
          Q + offs_m[:, None] * stride_qm + qk_d[None, :],
          mask=(offs_m[:, None] < seqlen_q) & (qk_d[None, :] < headdim),
          other=0.0,
        )
        k = tl.load(
          K + offs_kv[:, None] * stride_kn + qk_d[None, :],
          mask=(offs_kv[:, None] < seqlen_k) & (qk_d[None, :] < headdim),
          other=0.0,
        )
        scores = tl.dot(q, tl.trans(k), acc=scores)

      scores = scores * softmax_scale
      if not EVEN_N:
        scores = tl.where(offs_kv[None, :] < seqlen_k, scores, -float("inf"))
      if IS_CAUSAL:
        causal_mask = offs_kv[None, :] <= (offs_m[:, None] + kv_offset)
        scores = tl.where(causal_mask, scores, -float("inf"))

      m_new = tl.maximum(m_i, tl.max(scores, axis=1))
      alpha = tl.exp(m_i - m_new)
      p = tl.exp(scores - m_new[:, None])
      l_new = l_i * alpha + tl.sum(p, axis=1)

      p_cast = p.to(DTYPE)
      v0 = tl.load(
        V + offs_kv[:, None] * stride_vn + o_d0[None, :],
        mask=(offs_kv[:, None] < seqlen_k) & (o_d0[None, :] < headdim),
        other=0.0,
      )
      acc0 = acc0 * alpha[:, None] + tl.dot(p_cast, v0)
      if NUM_D_ACC >= 2:
        v1 = tl.load(
          V + offs_kv[:, None] * stride_vn + o_d1[None, :],
          mask=(offs_kv[:, None] < seqlen_k) & (o_d1[None, :] < headdim),
          other=0.0,
        )
        acc1 = acc1 * alpha[:, None] + tl.dot(p_cast, v1)
      if NUM_D_ACC >= 3:
        v2 = tl.load(
          V + offs_kv[:, None] * stride_vn + o_d2[None, :],
          mask=(offs_kv[:, None] < seqlen_k) & (o_d2[None, :] < headdim),
          other=0.0,
        )
        acc2 = acc2 * alpha[:, None] + tl.dot(p_cast, v2)
      if NUM_D_ACC >= 4:
        v3 = tl.load(
          V + offs_kv[:, None] * stride_vn + o_d3[None, :],
          mask=(offs_kv[:, None] < seqlen_k) & (o_d3[None, :] < headdim),
          other=0.0,
        )
        acc3 = acc3 * alpha[:, None] + tl.dot(p_cast, v3)
      m_i = m_new
      l_i = l_new

    out0 = acc0 / (l_i[:, None] + 1.0e-10)
    tl.store(
      O + offs_m[:, None] * stride_om + o_d0[None, :],
      out0.to(DTYPE),
      mask=(offs_m[:, None] < seqlen_q) & (o_d0[None, :] < headdim),
    )
    if NUM_D_ACC >= 2:
      out1 = acc1 / (l_i[:, None] + 1.0e-10)
      tl.store(
        O + offs_m[:, None] * stride_om + o_d1[None, :],
        out1.to(DTYPE),
        mask=(offs_m[:, None] < seqlen_q) & (o_d1[None, :] < headdim),
      )
    if NUM_D_ACC >= 3:
      out2 = acc2 / (l_i[:, None] + 1.0e-10)
      tl.store(
        O + offs_m[:, None] * stride_om + o_d2[None, :],
        out2.to(DTYPE),
        mask=(offs_m[:, None] < seqlen_q) & (o_d2[None, :] < headdim),
      )
    if NUM_D_ACC >= 4:
      out3 = acc3 / (l_i[:, None] + 1.0e-10)
      tl.store(
        O + offs_m[:, None] * stride_om + o_d3[None, :],
        out3.to(DTYPE),
        mask=(offs_m[:, None] < seqlen_q) & (o_d3[None, :] < headdim),
      )
    tl.store(LSE + offs_m, m_i + tl.log(l_i), mask=offs_m < seqlen_q)


_ffpa_fwd_autotune = triton.autotune(
  configs=_FFPA_FWD_AUTOTUNE_CONFIGS,
  key=["seqlen_q", "seqlen_k", "headdim"],
  cache_results=True,
)(_ffpa_fwd_kernel_impl)

_ffpa_fwd = _ffpa_fwd_kernel_impl


def _ffpa_attn_forward_impl(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  o: torch.Tensor,
  lse: torch.Tensor,
  causal: bool = False,
  softmax_scale: float | None = None,
  autotune: bool = False,
) -> None:
  """Run the Triton FFPA Split-D forward kernel.

  :param q: Query tensor in ``[B, Hq, Nq, D]`` layout.
  :param k: Key tensor in ``[B, Hkv, Nkv, D]`` layout.
  :param v: Value tensor in ``[B, Hkv, Nkv, D]`` layout.
  :param o: Output tensor in ``[B, Hq, Nq, D]`` layout, written in place.
  :param lse: Float32 LSE tensor with shape ``[B, Hq, Nq_aligned]`` or a
      view whose last dimension is the visible query length.
  :param causal: Whether to apply lower-right causal masking.
  :param softmax_scale: Scale applied to ``Q @ K.T``. Defaults to
      ``1 / sqrt(D)``.
  :param autotune: Whether to run Triton's autotuner for this shape.
  """
  batch, nheads_q, seqlen_q, headdim = q.shape
  _, nheads_kv, seqlen_k, _ = k.shape
  softmax_scale = softmax_scale or (1.0 / math.sqrt(headdim))
  seqlen_q_rounded = lse.shape[-1]

  assert q.dtype == k.dtype == v.dtype == o.dtype
  assert q.dtype in (torch.float16, torch.bfloat16)
  assert lse.dtype == torch.float32
  assert q.stride(-1) == k.stride(-1) == v.stride(-1) == o.stride(-1) == 1

  DTYPE = tl.float16 if q.dtype == torch.float16 else tl.bfloat16

  def grid(meta: dict) -> tuple[int, int]:
    return (triton.cdiv(seqlen_q, meta["BLOCK_M"]), batch * nheads_q)

  if autotune:
    _ffpa_fwd_autotune[grid](
      q,
      k,
      v,
      o,
      lse,
      softmax_scale,
      q.stride(0),
      q.stride(1),
      q.stride(2),
      k.stride(0),
      k.stride(1),
      k.stride(2),
      v.stride(0),
      v.stride(1),
      v.stride(2),
      o.stride(0),
      o.stride(1),
      o.stride(2),
      nheads_q,
      nheads_kv,
      seqlen_q,
      seqlen_k,
      seqlen_q_rounded,
      headdim,
      IS_CAUSAL=causal,
      DTYPE=DTYPE,
    )
  else:
    _ffpa_fwd[grid](
      q,
      k,
      v,
      o,
      lse,
      softmax_scale,
      q.stride(0),
      q.stride(1),
      q.stride(2),
      k.stride(0),
      k.stride(1),
      k.stride(2),
      v.stride(0),
      v.stride(1),
      v.stride(2),
      o.stride(0),
      o.stride(1),
      o.stride(2),
      nheads_q,
      nheads_kv,
      seqlen_q,
      seqlen_k,
      seqlen_q_rounded,
      headdim,
      IS_CAUSAL=causal,
      DTYPE=DTYPE,
      BLOCK_M=16,
      BLOCK_N=64,
      BLOCK_HEADDIM_QK=64,
      BLOCK_HEADDIM_O=128,
      NUM_D_ACC=4,
      num_warps=4,
      num_stages=2,
    )


def _ffpa_attn_forward(
  Q: torch.Tensor,
  K: torch.Tensor,
  V: torch.Tensor,
  O: torch.Tensor | None = None,
  causal: bool = False,
  softmax_scale: float = 0.0,
  autotune: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Call the Triton FFPA forward, returning ``(O, softmax_lse)``.

  :param Q: Query tensor with layout ``[B, Nh_q, Nq, D]``.
  :param K: Key tensor with layout ``[B, Nh_kv, Nkv, D]``.
  :param V: Value tensor with layout ``[B, Nh_kv, Nkv, D]``.
  :param O: Optional output tensor with layout ``[B, Nh_q, Nq, D]``.
  :param causal: Whether to apply lower-right causal masking.
  :param softmax_scale: Scale applied to ``Q @ K.T``.
  :param autotune: Whether to enable Triton forward autotuning.
  :return: Tuple ``(O, softmax_lse)``.  ``softmax_lse`` stores natural-log
      LSE values with visible shape ``[B, Nh_q, Nq]``.
  """
  if Q.stride(-1) != 1:
    Q = Q.contiguous()
  if K.stride(-1) != 1:
    K = K.contiguous()
  if V.stride(-1) != 1:
    V = V.contiguous()
  if O is None:
    O = torch.empty_like(Q)  # noqa: E741
  if O.stride(-1) != 1:
    raise ValueError("Triton forward requires O.stride(-1) == 1")

  seqlen_q = Q.size(2)
  seqlen_q_aligned = ((seqlen_q + 127) // 128) * 128
  softmax_lse_storage = torch.empty(Q.size(0), Q.size(1), seqlen_q_aligned, dtype=torch.float32, device=Q.device)
  softmax_lse = softmax_lse_storage[..., :seqlen_q]
  _ffpa_attn_forward_impl(
    Q,
    K,
    V,
    O,
    softmax_lse,
    causal=causal,
    softmax_scale=softmax_scale,
    autotune=autotune,
  )
  return O, softmax_lse

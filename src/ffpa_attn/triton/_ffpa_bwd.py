"""
Split-D FFPA Attention Backward — Triton Implementation.

Triton >= 3.x compatible (uses ``tl.trans`` instead of ``trans_b``).
Supports headdim up to 1024 via Split-D tiling.

Phase 1 (per Q-block, D-chunk outer → accumulate S/dP across D):
    S   = sum_d Q_d @ K_d^T           [BLOCK_M, BLOCK_N]  fp32
    dP  = sum_d dO_d @ V_d^T           [BLOCK_M, BLOCK_N]  fp32

Phase 1b/1c (per Q-block):
    P   = exp(S * scale - LSE)        [BLOCK_M, BLOCK_N]  fp32
    dS  = P * (dP - delta) * scale    [BLOCK_M, BLOCK_N]  fp32→DTYPE

Phase 2 (per Q-block, D-chunk):
    dQ_d = dS @ K_d                   [BLOCK_M, BLOCK_HEADDIM]
    dK_d = (Q_d^T @ dS)^T             [BLOCK_N, BLOCK_HEADDIM]  (atomicAdd)
    dV_d = (dO_d^T @ P)^T             [BLOCK_N, BLOCK_HEADDIM]  (atomicAdd)

delta = rowsum(dO * O) is precomputed.
"""

import math

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Preprocess: delta = rowsum(dO * O)
# ---------------------------------------------------------------------------


@triton.jit
def _bwd_preprocess_do_o_dot(
  Out,
  DO,
  Delta,
  stride_ob,
  stride_oh,
  stride_om,
  stride_dob,
  stride_doh,
  stride_dom,
  nheads,
  seqlen_q,
  seqlen_q_rounded,
  headdim,
  BLOCK_M: tl.constexpr,
  BLOCK_HEADDIM: tl.constexpr,
):
  start_m = tl.program_id(0)
  off_hb = tl.program_id(1)
  off_b = off_hb // nheads
  off_h = off_hb % nheads
  offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
  offs_d = tl.arange(0, BLOCK_HEADDIM)
  o = tl.load(
    Out + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :],
    mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
    other=0.0,
  ).to(tl.float32)
  do = tl.load(
    DO + off_b * stride_dob + off_h * stride_doh + offs_m[:, None] * stride_dom + offs_d[None, :],
    mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
    other=0.0,
  ).to(tl.float32)
  delta = tl.sum(o * do, axis=1)
  tl.store(Delta + off_hb * seqlen_q_rounded + offs_m, delta)


# ---------------------------------------------------------------------------
# Split-D backward — one K/V column block
# ---------------------------------------------------------------------------


@triton.jit
def _ffpa_bwd_kernel_one_col_block(
  start_n,
  Q,
  K,
  V,
  DO,
  DQ,
  DK,
  DV,
  LSE,
  D,
  softmax_scale,
  stride_qm,
  stride_kn,
  stride_vn,
  stride_dom,
  stride_dqm,
  stride_dkn,
  stride_dvn,
  seqlen_q,
  seqlen_k,
  headdim,
  ATOMIC_ADD: tl.constexpr,
  IS_CAUSAL: tl.constexpr,
  BLOCK_HEADDIM: tl.constexpr,
  DTYPE: tl.constexpr,
  EVEN_M: tl.constexpr,
  EVEN_N: tl.constexpr,
  BLOCK_M: tl.constexpr,
  BLOCK_N: tl.constexpr,
):
  begin_m = 0 if not IS_CAUSAL else ((start_n * BLOCK_N) // BLOCK_M) * BLOCK_M
  offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
  offs_m = tl.arange(0, BLOCK_M)
  offs_d = tl.arange(0, BLOCK_HEADDIM)

  if begin_m >= seqlen_q:
    return

  num_d_chunks = tl.cdiv(headdim, BLOCK_HEADDIM)
  num_block_m = tl.cdiv(seqlen_q, BLOCK_M)

  for start_m in range(begin_m, num_block_m * BLOCK_M, BLOCK_M):
    start_m = tl.multiple_of(start_m, BLOCK_M)
    offs_qm = start_m + offs_m  # Q row indices for this Q block
    offs_m_curr = offs_qm  # same as offs_qm

    # ---- Phase 1: S = sum_d Q_d @ K_d^T,  dP = sum_d dO_d @ V_d^T ----
    S = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    dP = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for d_chunk in range(num_d_chunks):
      d_start = d_chunk * BLOCK_HEADDIM
      d_offs = d_start + offs_d

      # Load Q_d.
      q = tl.load(
        Q + offs_qm[:, None] * stride_qm + d_offs[None, :],
        mask=(offs_m_curr[:, None] < seqlen_q) & (d_offs[None, :] < headdim),
        other=0.0
      )

      # Load K_d.
      k = tl.load(
        K + offs_n[:, None] * stride_kn + d_offs[None, :],
        mask=(offs_n[:, None] < seqlen_k) & (d_offs[None, :] < headdim),
        other=0.0
      )

      # Load V_d.
      v = tl.load(
        V + offs_n[:, None] * stride_vn + d_offs[None, :],
        mask=(offs_n[:, None] < seqlen_k) & (d_offs[None, :] < headdim),
        other=0.0
      )

      # Load dO_d.
      do = tl.load(
        DO + offs_qm[:, None] * stride_dom + d_offs[None, :],
        mask=(offs_m_curr[:, None] < seqlen_q) & (d_offs[None, :] < headdim),
        other=0.0
      )

      # Accumulate across D chunks.
      S = tl.dot(q, tl.trans(k), acc=S)
      dP = tl.dot(do, tl.trans(v), acc=dP)

    # ---- Phase 1b: softmax reconstruction ----
    if not EVEN_N:
      S = tl.where(offs_n[None, :] < seqlen_k, S, float("-inf"))
    if IS_CAUSAL:
      S = tl.where(offs_m_curr[:, None] >= (offs_n[None, :]), S, float("-inf"))

    lse_i = tl.load(LSE + offs_m_curr)
    P = tl.exp(S * softmax_scale - lse_i[:, None])

    # ---- Phase 1c: dS = P * (dP - delta) * scale ----
    Di = tl.load(D + offs_m_curr)
    dS = (P * (dP - Di[:, None]) * softmax_scale).to(DTYPE)

    # ---- Phase 2: dQ, dK, dV per D chunk ----
    for d_chunk in range(num_d_chunks):
      d_start = d_chunk * BLOCK_HEADDIM
      d_offs = d_start + offs_d

      # --- dQ_d = dS @ K_d ---
      k = tl.load(
        K + offs_n[:, None] * stride_kn + d_offs[None, :],
        mask=(offs_n[:, None] < seqlen_k) & (d_offs[None, :] < headdim),
        other=0.0
      )
      dq_d = tl.dot(dS, k).to(DTYPE)
      dq_ptrs = DQ + offs_qm[:, None] * stride_dqm + d_offs[None, :]
      if not ATOMIC_ADD:
        dq = tl.load(
          dq_ptrs,
          mask=(offs_m_curr[:, None] < seqlen_q) & (d_offs[None, :] < headdim),
          other=0.0,
          eviction_policy="evict_last"
        )
        dq += dq_d
        tl.store(
          dq_ptrs,
          dq,
          mask=(offs_m_curr[:, None] < seqlen_q) & (d_offs[None, :] < headdim),
          eviction_policy="evict_last"
        )
      else:
        tl.atomic_add(dq_ptrs, dq_d, mask=(offs_m_curr[:, None] < seqlen_q) & (d_offs[None, :] < headdim))

      # --- dK_d = (Q_d^T @ dS)^T ---
      q = tl.load(
        Q + offs_qm[:, None] * stride_qm + d_offs[None, :],
        mask=(offs_m_curr[:, None] < seqlen_q) & (d_offs[None, :] < headdim),
        other=0.0
      )
      dk_d = tl.trans(tl.dot(tl.trans(q), dS)).to(DTYPE)
      dk_ptrs = DK + offs_n[:, None] * stride_dkn + d_offs[None, :]
      tl.atomic_add(dk_ptrs, dk_d, mask=(offs_n[:, None] < seqlen_k) & (d_offs[None, :] < headdim))

      # --- dV_d = (dO_d^T @ P)^T ---
      do = tl.load(
        DO + offs_qm[:, None] * stride_dom + d_offs[None, :],
        mask=(offs_m_curr[:, None] < seqlen_q) & (d_offs[None, :] < headdim),
        other=0.0
      )
      dv_d = tl.trans(tl.dot(tl.trans(do).to(tl.float32), P)).to(DTYPE)
      dv_ptrs = DV + offs_n[:, None] * stride_dvn + d_offs[None, :]
      tl.atomic_add(dv_ptrs, dv_d, mask=(offs_n[:, None] < seqlen_k) & (d_offs[None, :] < headdim))


# ---------------------------------------------------------------------------
# Main backward kernel
# ---------------------------------------------------------------------------


@triton.heuristics({
  "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
  "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
})
@triton.jit
def _ffpa_bwd_kernel(
  Q,
  K,
  V,
  DO,
  DQ,
  DK,
  DV,
  LSE,
  D,
  softmax_scale,
  stride_qb,
  stride_qh,
  stride_qm,
  stride_kb,
  stride_kh,
  stride_kn,
  stride_vb,
  stride_vh,
  stride_vn,
  stride_dob,
  stride_doh,
  stride_dom,
  stride_dqb,
  stride_dqh,
  stride_dqm,
  stride_dkb,
  stride_dkh,
  stride_dkn,
  stride_dvb,
  stride_dvh,
  stride_dvn,
  nheads,
  seqlen_q,
  seqlen_k,
  seqlen_q_rounded,
  headdim,
  IS_CAUSAL: tl.constexpr,
  BLOCK_HEADDIM: tl.constexpr,
  DTYPE: tl.constexpr,
  SEQUENCE_PARALLEL: tl.constexpr,
  EVEN_M: tl.constexpr,
  EVEN_N: tl.constexpr,
  BLOCK_M: tl.constexpr,
  BLOCK_N: tl.constexpr,
):
  off_hb = tl.program_id(1)
  off_b = off_hb // nheads
  off_h = off_hb % nheads

  Q += off_b * stride_qb + off_h * stride_qh
  K += off_b * stride_kb + off_h * stride_kh
  V += off_b * stride_vb + off_h * stride_vh
  DO += off_b * stride_dob + off_h * stride_doh
  DQ += off_b * stride_dqb + off_h * stride_dqh
  DK += off_b * stride_dkb + off_h * stride_dkh
  DV += off_b * stride_dvb + off_h * stride_dvh
  D += off_hb * seqlen_q_rounded
  LSE += off_hb * seqlen_q_rounded

  if not SEQUENCE_PARALLEL:
    num_block_n = tl.cdiv(seqlen_k, BLOCK_N)
    for start_n in range(0, num_block_n):
      _ffpa_bwd_kernel_one_col_block(
        start_n,
        Q,
        K,
        V,
        DO,
        DQ,
        DK,
        DV,
        LSE,
        D,
        softmax_scale,
        stride_qm,
        stride_kn,
        stride_vn,
        stride_dom,
        stride_dqm,
        stride_dkn,
        stride_dvn,
        seqlen_q,
        seqlen_k,
        headdim,
        ATOMIC_ADD=False,
        IS_CAUSAL=IS_CAUSAL,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
        DTYPE=DTYPE,
        EVEN_M=EVEN_M,
        EVEN_N=EVEN_N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
      )
  else:
    start_n = tl.program_id(0)
    _ffpa_bwd_kernel_one_col_block(
      start_n,
      Q,
      K,
      V,
      DO,
      DQ,
      DK,
      DV,
      LSE,
      D,
      softmax_scale,
      stride_qm,
      stride_kn,
      stride_vn,
      stride_dom,
      stride_dqm,
      stride_dkn,
      stride_dvn,
      seqlen_q,
      seqlen_k,
      headdim,
      ATOMIC_ADD=True,
      IS_CAUSAL=IS_CAUSAL,
      BLOCK_HEADDIM=BLOCK_HEADDIM,
      DTYPE=DTYPE,
      EVEN_M=EVEN_M,
      EVEN_N=EVEN_N,
      BLOCK_M=BLOCK_M,
      BLOCK_N=BLOCK_N,
    )


# ---------------------------------------------------------------------------
# Python entry point
# ---------------------------------------------------------------------------


def _ffpa_attn_backward(do, q, k, v, o, lse, dq, dk, dv, causal=False, softmax_scale=None):
  """
    FFPA backward entry point.

    The kernel uses three strides per tensor:
        stride_qb  — batch stride   (distance between consecutive batch elements)
        stride_qh  — head stride    (distance between consecutive heads)
        stride_qm  — row stride     (distance between consecutive seqlen rows)

    For FFPA layout [B, Nh, Nq, D] these are naturally:
        stride_qb = q.stride(0) = Nh * Nq * D
        stride_qh = q.stride(1) = Nq * D
        stride_qm = q.stride(2) = D

    LSE and delta are indexed linearly: offset = (batch * Nh + head) * Nq_rounded + row.
    """
  if do.stride(-1) != 1:
    do = do.contiguous()
  batch, nheads, seqlen_q, headdim = q.shape
  _, _, seqlen_k, _ = k.shape
  softmax_scale = softmax_scale or (1.0 / math.sqrt(headdim))
  seqlen_q_rounded = lse.shape[-1]

  assert q.dtype == k.dtype == v.dtype == o.dtype == do.dtype
  assert q.dtype in (torch.float16, torch.bfloat16)

  DTYPE = tl.float16 if q.dtype == torch.float16 else tl.bfloat16

  BLOCK_HEADDIM_DELTA = max(triton.next_power_of_2(headdim), 16)
  delta = torch.empty_like(lse)
  _bwd_preprocess_do_o_dot[(triton.cdiv(seqlen_q, 128), batch * nheads)](
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
    seqlen_q_rounded,
    headdim,
    BLOCK_M=128,
    BLOCK_HEADDIM=BLOCK_HEADDIM_DELTA,
  )

  if headdim >= 128:
    BLOCK_HEADDIM_MAIN = 128
    BLOCK_M = 32
    BLOCK_N = 32
  else:
    BLOCK_HEADDIM_MAIN = max(triton.next_power_of_2(headdim), 16)
    BLOCK_M = 64
    BLOCK_N = 64
  SEQUENCE_PARALLEL = False

  dq.zero_()
  dk.zero_()
  dv.zero_()

  grid = (triton.cdiv(seqlen_k, BLOCK_N) if SEQUENCE_PARALLEL else 1, batch * nheads)

  _ffpa_bwd_kernel[grid](
    q,
    k,
    v,
    do,
    dq,
    dk,
    dv,
    lse,
    delta,
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
    do.stride(0),
    do.stride(1),
    do.stride(2),
    dq.stride(0),
    dq.stride(1),
    dq.stride(2),
    dk.stride(0),
    dk.stride(1),
    dk.stride(2),
    dv.stride(0),
    dv.stride(1),
    dv.stride(2),
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    IS_CAUSAL=causal,
    BLOCK_HEADDIM=BLOCK_HEADDIM_MAIN,
    DTYPE=DTYPE,
    SEQUENCE_PARALLEL=SEQUENCE_PARALLEL,
    BLOCK_M=BLOCK_M,
    BLOCK_N=BLOCK_N,
  )

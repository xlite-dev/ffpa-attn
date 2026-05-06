"""
FFPA Attention Backward (Split-D) — Triton Implementation.

FFPA v1 Backward Kernel was adapted from:
  https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_triton.py
FFPA v2 Backward Kernel was adapted from:
  https://triton-lang.org/main/_downloads/54a35f6ec55f9746935b9566fb6bb1df/06-fused-attention.py

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

Known Limitations & Future Optimizations
-----------------------------------------
1. **Q / dO / K repeated HBM reads across D-chunks.**  Phase 1 and Phase 2
   both iterate over D-chunks independently.  For D=512 with BLOCK_HEADDIM=128
   this means 4 chunks x 2 phases = 8 HBM loads each for Q, dO and K.  A
   future optimisation should cache these tiles in shared memory (requires
   >= 64 KB SMEM, i.e. Ada or extended Ampere) or split the kernel into
   Phase-1-only and Phase-2-only kernels to eliminate the re-reads entirely.
"""

import math

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Preprocess: delta = rowsum(dO * O)
# ---------------------------------------------------------------------------


@triton.jit
def _ffpa_bwd_pre_impl(
  Out: torch.Tensor,
  DO: torch.Tensor,
  Delta: torch.Tensor,
  stride_ob: int,
  stride_oh: int,
  stride_om: int,
  stride_dob: int,
  stride_doh: int,
  stride_dom: int,
  nheads: int,
  seqlen_q: int,
  seqlen_q_rounded: int,
  headdim: int,
  BLOCK_M: tl.constexpr,
  BLOCK_HEADDIM: tl.constexpr,
) -> None:
  """Preprocess kernel to compute delta = rowsum(dO * O) for the backward pass."""
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


def _gen_pre_autotune_configs() -> list[triton.Config]:
  """Generate autotune configs for the preprocess delta kernel."""
  configs = []
  for block_m in [64, 128, 256]:
    for block_headdim in [64, 128]:
      for num_warps in [4, 8]:
        configs.append(
          triton.Config(
            {
              "BLOCK_M": block_m,
              "BLOCK_HEADDIM": block_headdim
            },
            num_warps=num_warps,
            num_stages=3,
          )
        )
  return configs


# Autotuned variant.
_ffpa_bwd_pre_autotune = triton.autotune(
  configs=_gen_pre_autotune_configs(),
  key=["seqlen_q", "headdim"],
  reset_to_zero=["Delta"],
  cache_results=True,
)(_ffpa_bwd_pre_impl)

# Non-autotuned variant.
_ffpa_bwd_pre = _ffpa_bwd_pre_impl

# ---------------------------------------------------------------------------
# Split-D backward v1 kernel — one K/V column block
# ---------------------------------------------------------------------------


@triton.jit
def ffpa_bwd_v1_kernel(
  start_n: int,
  Q: torch.Tensor,
  K: torch.Tensor,
  V: torch.Tensor,
  DO: torch.Tensor,
  DQ: torch.Tensor,
  DK: torch.Tensor,
  DV: torch.Tensor,
  LSE: torch.Tensor,
  D: torch.Tensor,
  softmax_scale: float,
  stride_qm: int,
  stride_kn: int,
  stride_vn: int,
  stride_dom: int,
  stride_dqm: int,
  stride_dkn: int,
  stride_dvn: int,
  seqlen_q: int,
  seqlen_k: int,
  headdim: int,
  ATOMIC_ADD: tl.constexpr,
  IS_CAUSAL: tl.constexpr,
  BLOCK_HEADDIM: tl.constexpr,
  DTYPE: tl.constexpr,
  EVEN_M: tl.constexpr,
  EVEN_N: tl.constexpr,
  BLOCK_M: tl.constexpr,
  BLOCK_N: tl.constexpr,
) -> None:
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
      # NOTE: ATOMIC_ADD is always True because the persistent kernel
      # (SEQUENCE_PARALLEL=True) launches one program per K-column block,
      # and all programs write to the same Q-row positions in dQ.  Non-
      # atomic load+add+store would produce data races.  The non-atomic
      # branch is only reachable when SEQUENCE_PARALLEL=False, which is
      # no longer used in the current implementation.
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
#
# Two entry points share the same jit implementation:
#
#   _ffpa_bwd_v1_autotune  — wraps the kernel with @triton.autotune for
#                                 automatic tile-size / warp search.  First
#                                 call at each shape benchmarks all configs
#                                 (~4-6s) then caches the best.
#
#   _ffpa_bwd_v1           — direct call without autotune.  Uses the
#                                 known-best config (BLOCK_M=128, BLOCK_N=32,
#                                 BLOCK_HEADDIM=128, num_warps=8,
#                                 num_stages=2) discovered on Ampere.
#                                 for D=512.  Suitable for production use
#                                 where predictable launch latency matters.
# ---------------------------------------------------------------------------


def _gen_bwd_autotune_configs() -> list[triton.Config]:
  """Generate autotune configs over BLOCK_M, BLOCK_N, BLOCK_HEADDIM, num_warps, num_stages.

    NOTE: ATOMIC_ADD is intentionally excluded.  With SEQUENCE_PARALLEL=True
    (persistent kernel, always enabled) every column-block program writes to
    the same dQ positions, so atomic-add is required for correctness.
    ATOMIC_ADD=False would produce data-raced dQ gradients.
  """
  # BLOCK_M: larger = fewer Q-block iterations (good), more register pressure.
  # BLOCK_N:
  #   64  — fewer column blocks, halves atomic contention for v1.
  #   128 — 4× fewer blocks, minimal atomic contention for v1.
  # BLOCK_HEADDIM (gated by available shared memory):
  #   64, 128 — classic D-chunk split, low register pressure, widely compatible.
  #   256     — 2 chunks for D=512, halves HBM reloads.  Requires BLOCK_M ≤ 64
  #             to fit registers; 1.3x slower on Ampere, may win on Ada+.
  #   512     — full-D single chunk, eliminates D-chunk loop entirely.
  #             Needs >= 128 KB SMEM; only included on Ada (128 KB) or Hopper
  #             (228 KB).  Skipped on Ampere (99 KB limit).
  # TODO: Optimize the autotune time by saving the best config per shape
  # (device-shape/headdim) in a file and loading it at the start of autotune.
  try:
    _max_smem = torch.cuda.get_device_properties(0).shared_memory_per_block_optin
  except Exception:
    _max_smem = 48 * 1024  # safe fallback: default SMEM
  _headdim_candidates = [64, 128, 256]
  if _max_smem >= 128 * 1024:
    _headdim_candidates.append(512)

  configs = []
  for block_m in [64, 128]:
    for block_n in [64, 128]:
      for block_headdim in _headdim_candidates:
        for num_warps in [4, 8]:
          for num_stages in [2, 3]:
            configs.append(
              triton.Config(
                {
                  "BLOCK_M": block_m,
                  "BLOCK_N": block_n,
                  "BLOCK_HEADDIM": block_headdim
                },
                num_warps=num_warps,
                num_stages=num_stages,
              )
            )
  return configs


_FFPA_BWD_AUTOTUNE_CONFIGS = _gen_bwd_autotune_configs()
_FFPA_BWD_HEURISTICS = {
  "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
  "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
}


@triton.heuristics(_FFPA_BWD_HEURISTICS)
@triton.jit
def _ffpa_bwd_v1_kernel_impl(
  Q: torch.Tensor,
  K: torch.Tensor,
  V: torch.Tensor,
  DO: torch.Tensor,
  DQ: torch.Tensor,
  DK: torch.Tensor,
  DV: torch.Tensor,
  LSE: torch.Tensor,
  D: torch.Tensor,
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
  stride_dob: int,
  stride_doh: int,
  stride_dom: int,
  stride_dqb: int,
  stride_dqh: int,
  stride_dqm: int,
  stride_dkb: int,
  stride_dkh: int,
  stride_dkn: int,
  stride_dvb: int,
  stride_dvh: int,
  stride_dvn: int,
  nheads: int,
  seqlen_q: int,
  seqlen_k: int,
  seqlen_q_rounded: int,
  headdim: int,
  IS_CAUSAL: tl.constexpr,
  SEQUENCE_PARALLEL: tl.constexpr,
  BLOCK_HEADDIM: tl.constexpr,
  DTYPE: tl.constexpr,
  EVEN_M: tl.constexpr,
  EVEN_N: tl.constexpr,
  BLOCK_M: tl.constexpr,
  BLOCK_N: tl.constexpr,
) -> None:
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
    # Serial path: one program per (batch, head) iterates over all K-column
    # blocks in a loop.  Low SM utilisation — for B=1, H=8 only 8 of 48 SM
    # are occupied. Kept only for code clarity / reference.
    num_block_n = tl.cdiv(seqlen_k, BLOCK_N)
    for start_n in range(0, num_block_n):
      ffpa_bwd_v1_kernel(
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
    # Persistent / column-parallel path: one program per (K-column block,
    # batch-head).  All programs independently process their own column
    # block and write dQ via atomic-add (other programs may update the
    # same Q rows).  This fully utilises the GPU's SM count.
    start_n = tl.program_id(0)
    ffpa_bwd_v1_kernel(
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


# Autotuned variant — wraps the impl with autotune; do NOT pass
# BLOCK_M / BLOCK_N / BLOCK_HEADDIM when calling this variant.
_ffpa_bwd_v1_autotune = triton.autotune(
  configs=_FFPA_BWD_AUTOTUNE_CONFIGS,
  key=["seqlen_q", "seqlen_k", "headdim"],
  reset_to_zero=["DQ", "DK", "DV"],
  cache_results=True,
)(_ffpa_bwd_v1_kernel_impl)

# Non-autotuned variant — same impl, called with the best known config.
_ffpa_bwd_v1 = _ffpa_bwd_v1_kernel_impl

# ====================================================================
# v2 kernel — shared-pid split-D backward (no dQ atomic_add)
#
# Inspired by flash-attention v2 _attn_bwd: one program_id serves as
# both the K-column block index and the Q-row block index.
#
# Grid: (max(cdiv(Nk, BLOCK_N), cdiv(Nq, BLOCK_M)), 1, B*Nh)
# Each program:
#   1. Computes dK/dV for its K-col block (if pid*BLOCK_N < Nk).
#   2. Computes dQ for its Q-row block (if pid*BLOCK_M < Nq).
#
# Because each program owns a unique Q-row block, dQ can be written
# non-atomically, removing the main v1 bottleneck at long seqlen.
# ====================================================================


@triton.heuristics(_FFPA_BWD_HEURISTICS)
@triton.jit
def _ffpa_bwd_v2_kernel_impl(
  Q: torch.Tensor,
  K: torch.Tensor,
  V: torch.Tensor,
  DO: torch.Tensor,
  DQ: torch.Tensor,
  DK: torch.Tensor,
  DV: torch.Tensor,
  LSE: torch.Tensor,
  D: torch.Tensor,
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
  stride_dob: int,
  stride_doh: int,
  stride_dom: int,
  stride_dqb: int,
  stride_dqh: int,
  stride_dqm: int,
  stride_dkb: int,
  stride_dkh: int,
  stride_dkn: int,
  stride_dvb: int,
  stride_dvh: int,
  stride_dvn: int,
  nheads: int,
  seqlen_q: int,
  seqlen_k: int,
  seqlen_q_rounded: int,
  headdim: int,
  IS_CAUSAL: tl.constexpr,
  BLOCK_HEADDIM: tl.constexpr,
  DTYPE: tl.constexpr,
  EVEN_M: tl.constexpr,
  EVEN_N: tl.constexpr,
  BLOCK_M: tl.constexpr,
  BLOCK_N: tl.constexpr,
) -> None:
  pid = tl.program_id(0)
  off_hb = tl.program_id(2)
  off_b = off_hb // nheads
  off_h = off_hb % nheads

  # ---- base pointers ----
  Q += off_b * stride_qb + off_h * stride_qh
  K += off_b * stride_kb + off_h * stride_kh
  V += off_b * stride_vb + off_h * stride_vh
  DO += off_b * stride_dob + off_h * stride_doh
  DQ += off_b * stride_dqb + off_h * stride_dqh
  DK += off_b * stride_dkb + off_h * stride_dkh
  DV += off_b * stride_dvb + off_h * stride_dvh
  D += off_hb * seqlen_q_rounded
  LSE += off_hb * seqlen_q_rounded

  num_d_chunks = tl.cdiv(headdim, BLOCK_HEADDIM)

  # ================================================================
  # Part 1: dK / dV — pid as K-column block index
  # ================================================================
  start_n = pid * BLOCK_N
  if start_n < seqlen_k:
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    num_block_m = tl.cdiv(seqlen_q, BLOCK_M)

    for start_m in range(0, num_block_m * BLOCK_M, BLOCK_M):
      offs_qm = start_m + offs_m

      # --- Phase 1: S = sum_d Q_d @ K_d^T, dP = sum_d dO_d @ V_d^T ---
      S = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
      dP = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

      for d_chunk in range(num_d_chunks):
        d_offs = d_chunk * BLOCK_HEADDIM + offs_d
        q = tl.load(
          Q + offs_qm[:, None] * stride_qm + d_offs[None, :],
          mask=(offs_qm[:, None] < seqlen_q) & (d_offs[None, :] < headdim),
          other=0.
        )
        k = tl.load(
          K + offs_n[:, None] * stride_kn + d_offs[None, :],
          mask=(offs_n[:, None] < seqlen_k) & (d_offs[None, :] < headdim),
          other=0.
        )
        v = tl.load(
          V + offs_n[:, None] * stride_vn + d_offs[None, :],
          mask=(offs_n[:, None] < seqlen_k) & (d_offs[None, :] < headdim),
          other=0.
        )
        do = tl.load(
          DO + offs_qm[:, None] * stride_dom + d_offs[None, :],
          mask=(offs_qm[:, None] < seqlen_q) & (d_offs[None, :] < headdim),
          other=0.
        )
        S = tl.dot(q, tl.trans(k), acc=S)
        dP = tl.dot(do, tl.trans(v), acc=dP)

      # --- Phase 1b/1c: softmax + dS ---
      if not EVEN_N:
        S = tl.where(offs_n[None, :] < seqlen_k, S, float("-inf"))
      if IS_CAUSAL:
        S = tl.where(offs_qm[:, None] >= (offs_n[None, :]), S, float("-inf"))
      lse_i = tl.load(LSE + offs_qm)
      P = tl.exp(S * softmax_scale - lse_i[:, None])
      Di = tl.load(D + offs_qm)
      dS = (P * (dP - Di[:, None]) * softmax_scale).to(DTYPE)

      # --- Phase 2 for dK/dV ---
      for d_chunk in range(num_d_chunks):
        d_offs = d_chunk * BLOCK_HEADDIM + offs_d
        q = tl.load(
          Q + offs_qm[:, None] * stride_qm + d_offs[None, :],
          mask=(offs_qm[:, None] < seqlen_q) & (d_offs[None, :] < headdim),
          other=0.
        )
        do = tl.load(
          DO + offs_qm[:, None] * stride_dom + d_offs[None, :],
          mask=(offs_qm[:, None] < seqlen_q) & (d_offs[None, :] < headdim),
          other=0.
        )
        dk_d = tl.trans(tl.dot(tl.trans(q), dS)).to(DTYPE)
        dk_ptrs = DK + offs_n[:, None] * stride_dkn + d_offs[None, :]
        dk_val = tl.load(dk_ptrs, mask=(offs_n[:, None] < seqlen_k) & (d_offs[None, :] < headdim), other=0.)
        dk_val += dk_d
        tl.store(dk_ptrs, dk_val, mask=(offs_n[:, None] < seqlen_k) & (d_offs[None, :] < headdim))
        dv_d = tl.trans(tl.dot(tl.trans(do).to(tl.float32), P)).to(DTYPE)
        dv_ptrs = DV + offs_n[:, None] * stride_dvn + d_offs[None, :]
        dv_val = tl.load(dv_ptrs, mask=(offs_n[:, None] < seqlen_k) & (d_offs[None, :] < headdim), other=0.)
        dv_val += dv_d
        tl.store(dv_ptrs, dv_val, mask=(offs_n[:, None] < seqlen_k) & (d_offs[None, :] < headdim))

  # ================================================================
  # Part 2: dQ — pid as Q-row block index (NON-ATOMIC!)
  # ================================================================
  start_m = pid * BLOCK_M
  if start_m < seqlen_q:
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    num_block_n = tl.cdiv(seqlen_k, BLOCK_N)

    for start_n_k in range(0, num_block_n * BLOCK_N, BLOCK_N):
      offs_nk = start_n_k + offs_n

      # --- Phase 1: S, dP for this Q-block × K-block ---
      S_qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
      dP_qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

      for d_chunk in range(num_d_chunks):
        d_offs = d_chunk * BLOCK_HEADDIM + offs_d
        q = tl.load(
          Q + offs_m[:, None] * stride_qm + d_offs[None, :],
          mask=(offs_m[:, None] < seqlen_q) & (d_offs[None, :] < headdim),
          other=0.
        )
        k = tl.load(
          K + offs_nk[:, None] * stride_kn + d_offs[None, :],
          mask=(offs_nk[:, None] < seqlen_k) & (d_offs[None, :] < headdim),
          other=0.
        )
        v = tl.load(
          V + offs_nk[:, None] * stride_vn + d_offs[None, :],
          mask=(offs_nk[:, None] < seqlen_k) & (d_offs[None, :] < headdim),
          other=0.
        )
        do = tl.load(
          DO + offs_m[:, None] * stride_dom + d_offs[None, :],
          mask=(offs_m[:, None] < seqlen_q) & (d_offs[None, :] < headdim),
          other=0.
        )
        S_qk = tl.dot(q, tl.trans(k), acc=S_qk)
        dP_qk = tl.dot(do, tl.trans(v), acc=dP_qk)

      # --- Phase 1b/1c ---
      if not EVEN_N:
        S_qk = tl.where(offs_nk[None, :] < seqlen_k, S_qk, float("-inf"))
      if IS_CAUSAL:
        S_qk = tl.where(offs_m[:, None] >= (offs_nk[None, :]), S_qk, float("-inf"))
      lse_i = tl.load(LSE + offs_m)
      P_qk = tl.exp(S_qk * softmax_scale - lse_i[:, None])
      Di = tl.load(D + offs_m)
      dS_qk = (P_qk * (dP_qk - Di[:, None]) * softmax_scale).to(DTYPE)

      # --- Phase 2 for dQ ---
      for d_chunk in range(num_d_chunks):
        d_offs = d_chunk * BLOCK_HEADDIM + offs_d
        k = tl.load(
          K + offs_nk[:, None] * stride_kn + d_offs[None, :],
          mask=(offs_nk[:, None] < seqlen_k) & (d_offs[None, :] < headdim),
          other=0.
        )
        dq_d = tl.dot(dS_qk, k).to(DTYPE)
        dq_ptrs = DQ + offs_m[:, None] * stride_dqm + d_offs[None, :]
        dq_val = tl.load(dq_ptrs, mask=(offs_m[:, None] < seqlen_q) & (d_offs[None, :] < headdim), other=0.)
        dq_val += dq_d
        # NOTE: dQ is written non-atomically — each program owns a unique Q-row block.
        tl.store(dq_ptrs, dq_val, mask=(offs_m[:, None] < seqlen_q) & (d_offs[None, :] < headdim))


# Autotuned v2 variant.
_ffpa_bwd_v2_autotune = triton.autotune(
  configs=_FFPA_BWD_AUTOTUNE_CONFIGS,
  key=["seqlen_q", "seqlen_k", "headdim"],
  reset_to_zero=["DQ", "DK", "DV"],
  cache_results=True,
)(_ffpa_bwd_v2_kernel_impl)

# Non-autotuned v2 variant.
_ffpa_bwd_v2 = _ffpa_bwd_v2_kernel_impl

# ---------------------------------------------------------------------------
# Python entry point
# ---------------------------------------------------------------------------


def _ffpa_attn_backward(
  do: torch.Tensor,
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  o: torch.Tensor,
  lse: torch.Tensor,
  dq: torch.Tensor,
  dk: torch.Tensor,
  dv: torch.Tensor,
  causal: bool = False,
  softmax_scale: float | None = None,
  autotune: bool = False,
  kernel_version: str = "v2",
) -> None:
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
  if autotune:

    def pre_grid(meta: dict) -> tuple[int, int]:
      return (triton.cdiv(seqlen_q, meta["BLOCK_M"]), batch * nheads)

    _ffpa_bwd_pre_autotune[pre_grid](
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
    )
  else:
    _ffpa_bwd_pre[(triton.cdiv(seqlen_q, 128), batch * nheads)](
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

  # Grid and kernel dispatch.
  dq.zero_()
  dk.zero_()
  dv.zero_()

  if kernel_version == "v2":
    # v2: shared-pid split-D, grid = (max(K-blocks, Q-blocks), 1, B*Nh).
    # pid serves as both K-col block index and Q-row block index.
    def grid(meta: dict) -> tuple[int, ...]:
      return (
        max(triton.cdiv(seqlen_k, meta["BLOCK_N"]), triton.cdiv(seqlen_q, meta["BLOCK_M"])),
        1,
        batch * nheads,
      )

    if autotune:
      _ffpa_bwd_v2_autotune[grid](
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
        DTYPE=DTYPE,
      )
    else:
      _ffpa_bwd_v2[grid](
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
        DTYPE=DTYPE,
        BLOCK_M=128,
        BLOCK_N=64,
        BLOCK_HEADDIM=128,
        num_warps=8,
        num_stages=2,
      )
  else:
    # v1: split-kv (current), grid = (K-column blocks, B*Nh).
    def grid(meta: dict) -> tuple[int, ...]:
      return (triton.cdiv(seqlen_k, meta["BLOCK_N"]), batch * nheads)

    if autotune:
      _ffpa_bwd_v1_autotune[grid](
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
        SEQUENCE_PARALLEL=True,
        DTYPE=DTYPE,
      )
    else:
      _ffpa_bwd_v1[grid](
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
        SEQUENCE_PARALLEL=True,
        DTYPE=DTYPE,
        BLOCK_M=128,
        BLOCK_N=64,
        BLOCK_HEADDIM=128,
        num_warps=8,
        num_stages=2,
      )

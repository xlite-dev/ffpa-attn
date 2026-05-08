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

Known Limitations & Future Optimizations:
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

# Preprocess: delta = rowsum(dO * O)
# In full-D mode BLOCK_HEADDIM must cover the whole head dimension.  In
# D_CHUNK mode the launcher/autotuner supplies the chunk size explicitly.
_FFPA_BWD_PRE_HEURISTICS = {
  "BLOCK_HEADDIM":
  lambda args: args["BLOCK_HEADDIM"] if args["D_CHUNK"] else max(64, triton.next_power_of_2(args["headdim"])),
}


@triton.heuristics(_FFPA_BWD_PRE_HEURISTICS)
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
  D_CHUNK: tl.constexpr,
) -> None:
  """Preprocess kernel to compute delta = rowsum(dO * O) for the backward pass."""
  start_m = tl.program_id(0)
  off_hb = tl.program_id(1)
  off_b = off_hb // nheads
  off_h = off_hb % nheads
  offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
  offs_d = tl.arange(0, BLOCK_HEADDIM)

  if D_CHUNK:
    delta = tl.zeros([BLOCK_M], dtype=tl.float32)
    num_d_chunks = tl.cdiv(headdim, BLOCK_HEADDIM)
    for d_chunk in range(num_d_chunks):
      d_offs = d_chunk * BLOCK_HEADDIM + offs_d
      o = tl.load(
        Out + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + d_offs[None, :],
        mask=(offs_m[:, None] < seqlen_q) & (d_offs[None, :] < headdim),
        other=0.0,
      ).to(tl.float32)
      do = tl.load(
        DO + off_b * stride_dob + off_h * stride_doh + offs_m[:, None] * stride_dom + d_offs[None, :],
        mask=(offs_m[:, None] < seqlen_q) & (d_offs[None, :] < headdim),
        other=0.0,
      ).to(tl.float32)
      delta += tl.sum(o * do, axis=1)
  else:
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

  tl.store(Delta + off_hb * seqlen_q_rounded + offs_m, delta, mask=offs_m < seqlen_q)


def _gen_pre_autotune_configs(d_chunk: bool) -> list[triton.Config]:
  """Generate autotune configs for the preprocess delta kernel.

  ``BLOCK_HEADDIM`` participates in autotune only for D_CHUNK mode.  Full-D
  mode keeps the historical runtime heuristic so invalid narrow configs are
  never benchmarked for large head dimensions.

  :param d_chunk: Whether generated configs should enable D_CHUNK mode.
  :return: Triton autotune configurations for the delta preprocess kernel.
  """
  configs = []
  for block_m in [64, 128, 256]:
    if not d_chunk:
      for num_warps in [2, 4, 8]:
        configs.append(triton.Config(
          {
            "BLOCK_M": block_m,
            "D_CHUNK": False
          },
          num_warps=num_warps,
        ))
      continue

    for block_headdim in [64, 128, 256]:
      for num_warps in [2, 4, 8]:
        configs.append(
          triton.Config(
            {
              "BLOCK_M": block_m,
              "BLOCK_HEADDIM": block_headdim,
              "D_CHUNK": True
            },
            num_warps=num_warps,
          )
        )
  return configs


# Autotuned variant.
_ffpa_bwd_pre_autotune = triton.autotune(
  configs=_gen_pre_autotune_configs(d_chunk=False),
  key=["seqlen_q", "headdim"],
  reset_to_zero=["Delta"],
  cache_results=True,
)(_ffpa_bwd_pre_impl)

_ffpa_bwd_pre_d_chunk_autotune = triton.autotune(
  configs=_gen_pre_autotune_configs(d_chunk=True),
  key=["seqlen_q", "headdim"],
  reset_to_zero=["Delta"],
  cache_results=True,
)(_ffpa_bwd_pre_impl)

# Non-autotuned variant.
_ffpa_bwd_pre = _ffpa_bwd_pre_impl


# Split-D backward v1 kernel — one K/V column block
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


# Main backward kernel
#
# Two entry points share the same jit implementation:
#
#   _ffpa_bwd_v1_autotune  — wraps the kernel with @triton.autotune for
#                                 automatic tile-size / warp search.  First
#                                 call at each shape benchmarks all configs
#                                 (~4-6s) then caches the best.
#
#   _ffpa_bwd_v1           — direct call without autotune. The Python launcher
#                                 supplies the fixed fallback tile config.
def _gen_bwd_autotune_configs(block_n_values: tuple[int, ...], headdim: int = 512) -> list[triton.Config]:
  """Generate autotune configs over BLOCK_M, BLOCK_N, BLOCK_HEADDIM, num_warps, num_stages.

  ``ATOMIC_ADD`` is intentionally excluded from autotune. In the v1
  column-parallel path, every K-column-block program can update the same dQ
  rows, so dQ atomic-add is required for correctness.

  :param block_n_values: Candidate ``BLOCK_N`` values for the target backward
      kernel variant.
  :param headdim: Full-D ``BLOCK_HEADDIM`` candidate for architectures with
      enough shared memory.  When the actual runtime headdim matches this
      value the kernel skips the D-chunk loop entirely.
  :return: Triton autotune configurations for one backward kernel variant.
  """
  # BLOCK_M: larger = fewer Q-block iterations (good), more register pressure.
  # BLOCK_N:
  #   64  — fewer column blocks, halves atomic contention for v1.
  #   128 — 4× fewer blocks, minimal atomic contention for v1.
  # BLOCK_HEADDIM (gated by available shared memory):
  #   64, 128 — classic D-chunk split, low register pressure, widely compatible.
  #   256     — 2 chunks for D=512, halves HBM reloads.  Requires BLOCK_M ≤ 64
  #             to fit registers; 1.3x slower on Ampere, may win on Ada+.
  #   headdim — full-D single chunk, eliminates D-chunk loop entirely.
  #             Needs >= 128 KB SMEM; only included on Ada (128 KB) or Hopper
  #             (228 KB).  Skipped on Ampere (99 KB limit).
  # TODO: Optimize the autotune time by saving the best config per shape
  # (device-shape/headdim) in a file and loading it at the start of autotune.
  try:
    _max_smem = torch.cuda.get_device_properties(0).shared_memory_per_block_optin
  except Exception:
    _max_smem = 48 * 1024  # safe fallback: default SMEM
  _headdim_candidates = [64, 128, 256]
  # Use triton.next_power_of_2(headdim) as a near-full-D single-chunk block size:
  #   - power-of-2 headdims (512, 1024): next_pow2 == headdim → single D chunk.
  #   - non-power-of-2 headdims (320→512, 640→1024): next_pow2 pads to the next
  #     power-of-2.  The kernel's load/store masks (d_offs < headdim) zero out the
  #     padding columns, so correctness is preserved.
  # tl.arange requires a power-of-2 range, so next_power_of_2 always produces a
  # valid block size. Only included on high-SMEM devices (Ada/Hopper, >= 128 KB);
  # skip when next_pow2 is already in [64, 128, 256] (dedup).
  _next_pow2 = triton.next_power_of_2(headdim)
  if _max_smem >= 128 * 1024 and _next_pow2 not in _headdim_candidates:  # 128 KB
    _headdim_candidates.append(_next_pow2)

  configs = []
  for block_m in [64, 128]:
    for block_n in block_n_values:
      for block_headdim in _headdim_candidates:
        for num_warps in [4, 8]:
          for num_stages in [2, 3, 4]:
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


# Non-autotuned variant — called with the best known config.
_ffpa_bwd_v1 = _ffpa_bwd_v1_kernel_impl

_ffpa_bwd_v1_autotune_cache: dict[int, callable] = {}  # headdim -> callable


def _get_v1_autotune(headdim: int):
  """Return a headdim-specific autotune wrapper for the v1 backward kernel."""
  if headdim not in _ffpa_bwd_v1_autotune_cache:
    configs = _gen_bwd_autotune_configs(block_n_values=(64, 128), headdim=headdim)
    _ffpa_bwd_v1_autotune_cache[headdim] = triton.autotune(
      configs=configs,
      key=["seqlen_q", "seqlen_k", "headdim"],
      reset_to_zero=["DQ", "DK", "DV"],
      cache_results=True,
    )(_ffpa_bwd_v1_kernel_impl)
  return _ffpa_bwd_v1_autotune_cache[headdim]


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

  # Part 1: dK / dV — pid as K-column block index
  start_n = pid * BLOCK_N
  if start_n < seqlen_k:
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    num_block_m = tl.cdiv(seqlen_q, BLOCK_M)
    begin_m = 0 if not IS_CAUSAL else start_n // BLOCK_M * BLOCK_M

    for start_m in range(begin_m, num_block_m * BLOCK_M, BLOCK_M):
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

  # Part 2: dQ — pid as Q-row block index (NON-ATOMIC!)
  start_m = pid * BLOCK_M
  if start_m < seqlen_q:
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    num_block_n = tl.cdiv(seqlen_k, BLOCK_N)
    end_n_k = start_m + BLOCK_M if IS_CAUSAL else num_block_n * BLOCK_N

    for start_n_k in range(0, end_n_k, BLOCK_N):
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


# Non-autotuned v2 variant.
_ffpa_bwd_v2 = _ffpa_bwd_v2_kernel_impl

_ffpa_bwd_v2_autotune_cache: dict[int, callable] = {}  # headdim -> callable


def _get_v2_autotune(headdim: int):
  """Return a headdim-specific autotune wrapper for the v2 backward kernel."""
  if headdim not in _ffpa_bwd_v2_autotune_cache:
    configs = _gen_bwd_autotune_configs(block_n_values=(64, ), headdim=headdim)
    _ffpa_bwd_v2_autotune_cache[headdim] = triton.autotune(
      configs=configs,
      key=["seqlen_q", "seqlen_k", "headdim"],
      reset_to_zero=["DQ", "DK", "DV"],
      cache_results=True,
    )(_ffpa_bwd_v2_kernel_impl)
  return _ffpa_bwd_v2_autotune_cache[headdim]


def _ffpa_attn_backward_triton_impl(
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
  preprocess_d_chunk: bool = False,
) -> None:
  """Run the Triton FFPA Split-D backward kernels in place.

  This is the low-level Triton implementation entrypoint used by the public
  wrapper below. Callers are expected to perform all FFPA-specific tensor
  preparation before entering here:

  * ``lse`` must already expose the padded last-dimension storage required by
    masked Triton loads
  * any GQA/MQA expansion of ``k`` and ``v`` must already be done
  * ``dq``, ``dk``, and ``dv`` must already be allocated with the expanded
    head layout expected by the selected kernel

  The function only computes delta, dispatches the chosen Triton backward
  kernel, and writes gradients into the provided output buffers.

  :param do: Upstream output gradient with layout ``[B, Nh, Nq, D]``.
  :param q: Query tensor saved from forward, layout ``[B, Nh, Nq, D]``.
  :param k: Key tensor saved from forward, layout ``[B, Nh, Nk, D]``.
  :param v: Value tensor saved from forward, layout ``[B, Nh, Nk, D]``.
  :param o: Forward output tensor, layout ``[B, Nh, Nq, D]``.
  :param lse: Forward softmax log-sum-exp tensor with visible layout
    ``[B, Nh, Nq]`` and storage rounded on the last dimension.
  :param dq: Query-gradient output tensor, written in place.
  :param dk: Key-gradient output tensor, written in place.
  :param dv: Value-gradient output tensor, written in place.
  :param causal: Whether the forward pass used lower-triangular causal
    masking.
  :param softmax_scale: Scale applied to ``Q @ K.T``. Defaults to
    ``1 / sqrt(D)`` when ``None``.
  :param autotune: Whether to run Triton's autotuner for the preprocess and
    selected backward kernel.
  :param kernel_version: Backward kernel variant to launch. ``"v2"`` uses the
    shared-pid split-D kernel without dQ atomics; any other value selects the
    v1 column-parallel kernel.
  :param preprocess_d_chunk: Whether the delta preprocess kernel should split
    the head dimension into ``BLOCK_HEADDIM`` chunks instead of processing the
    full head dimension in one program.
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

    pre_kernel = _ffpa_bwd_pre_d_chunk_autotune if preprocess_d_chunk else _ffpa_bwd_pre_autotune
    pre_kernel[pre_grid](
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
    block_headdim_delta = 64 if preprocess_d_chunk else BLOCK_HEADDIM_DELTA
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
      BLOCK_HEADDIM=block_headdim_delta,
      D_CHUNK=preprocess_d_chunk,
      num_warps=4,
    )

  # Grid and kernel dispatch.
  dq.zero_()
  dk.zero_()
  dv.zero_()

  # TODO: May force use v1 for short seqlen where atomics are not a
  # bottleneck and v2 overhead would dominate.
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
      _get_v2_autotune(headdim)[grid](
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
        BLOCK_HEADDIM=64,
        num_warps=8,
        num_stages=2,
      )
  else:
    # v1: column-parallel split-D, grid = (K-column blocks, B*Nh).
    def grid(meta: dict) -> tuple[int, ...]:
      return (triton.cdiv(seqlen_k, meta["BLOCK_N"]), batch * nheads)

    if autotune:
      _get_v1_autotune(headdim)[grid](
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
        BLOCK_HEADDIM=64,
        num_warps=8,
        num_stages=2,
      )


def _ffpa_attn_backward_triton(
  grad_out: torch.Tensor,
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  o: torch.Tensor,
  lse: torch.Tensor,
  causal: bool = False,
  softmax_scale: float | None = None,
  autotune: bool = False,
  kernel_version: str = "v2",
  preprocess_d_chunk: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Run the Triton FFPA backward path and return ``(dq, dk, dv)``.

  This is the backend-facing wrapper used by
  ``FFPAAttnFunc.backward(backward_backend="triton")``. It owns the
  FFPA-specific tensor preparation that should not live in the autograd
  dispatch layer:

  * pad ``lse`` to the rounded sequence length required by the Triton kernels
  * expand ``k`` / ``v`` for GQA or MQA when ``Nh_q > Nh_kv``
  * allocate the expanded ``dq`` / ``dk`` / ``dv`` buffers
  * call :func:`_ffpa_attn_backward_triton_impl`
  * reduce expanded ``dk`` / ``dv`` back to the original KV head layout
  * cast the returned gradients back to the original input dtypes

  :param grad_out: Upstream output gradient with shape ``[B, Nh_q, Nq, D]``.
  :param q: Query tensor saved from forward with shape ``[B, Nh_q, Nq, D]``.
  :param k: Key tensor saved from forward with shape ``[B, Nh_kv, Nkv, D]``.
  :param v: Value tensor saved from forward with shape ``[B, Nh_kv, Nkv, D]``.
  :param o: Forward output tensor saved on the autograd context with shape
    ``[B, Nh_q, Nq, D]``.
  :param lse: Forward log-sum-exp tensor saved on the autograd context with
    visible shape ``[B, Nh_q, Nq]``. The wrapper may pad its storage to
    ``[B, Nh_q, ceil_div(Nq, 128) * 128]`` before calling the Triton kernel.
  :param causal: Whether lower-right causal masking was used in forward.
  :param softmax_scale: Scale applied to ``QK^T``.
  :param autotune: Whether to use the headdim-specific Triton autotuned entry.
  :param kernel_version: Triton backward kernel variant to dispatch.
  :param preprocess_d_chunk: Whether to split the preprocess delta reduction
    across head-dim chunks.
  :returns: ``(dq, dk, dv)`` where ``dq`` has shape ``[B, Nh_q, Nq, D]`` and
    ``dk`` / ``dv`` have shape ``[B, Nh_kv, Nkv, D]``. Returned tensors use
    the original ``q`` / ``k`` / ``v`` dtypes and head layouts.
  """
  seqlen_q = q.size(2)
  seqlen_q_rounded = ((seqlen_q + 127) // 128) * 128
  if lse.size(-1) < seqlen_q_rounded:
    lse_padded = torch.empty(
      *lse.shape[:-1],
      seqlen_q_rounded,
      dtype=lse.dtype,
      device=lse.device,
    )
    lse_padded[..., :lse.size(-1)] = lse
    lse = lse_padded

  group_size = q.size(1) // k.size(1)
  if group_size > 1:
    k_in = k.repeat_interleave(group_size, dim=1).contiguous()
    v_in = v.repeat_interleave(group_size, dim=1).contiguous()
  else:
    k_in, v_in = k, v

  dq, dk_expanded, dv_expanded = torch.ops.ffpa_attn._bwd_triton(
    grad_out.contiguous(),
    q.contiguous(),
    k_in.contiguous(),
    v_in.contiguous(),
    o.contiguous(),
    lse,
    softmax_scale or (1.0 / math.sqrt(q.size(-1))),
    int(causal),
    int(autotune),
    int(kernel_version == "v2"),
    int(preprocess_d_chunk),
  )

  if group_size > 1:
    dk = dk_expanded.reshape(
      k.size(0),
      k.size(1),
      group_size,
      k.size(2),
      k.size(3),
    ).sum(dim=2).to(k.dtype)
    dv = dv_expanded.reshape(
      v.size(0),
      v.size(1),
      group_size,
      v.size(2),
      v.size(3),
    ).sum(dim=2).to(v.dtype)
  else:
    dk = dk_expanded.to(k.dtype)
    dv = dv_expanded.to(v.dtype)
  return dq.to(q.dtype), dk, dv

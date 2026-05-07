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

_MAX_HEADDIM = 1024


@triton.jit
def _update_o_accs(o_accs, v_group: tl.constexpr, o_acc):
  return o_accs[:v_group] + (o_acc, ) + o_accs[v_group + 1:]


def _gen_fwd_autotune_configs(headdim: int = 256) -> list[triton.Config]:
  """Generate autotune configs for the single FFPA Triton forward kernel.

  The search space is compact: tune Q-block size, QK/V D-chunk size, warp
  count, and pipeline depth.  ``BLOCK_N`` is fixed at 64.  ``BLOCK_HEADDIM_QK``
  and ``BLOCK_HEADDIM_V`` move in lockstep.

  When the device has >= 128 KB SMEM per block (Ada / Hopper), the actual
  ``headdim`` is appended to the candidates so a full-D single-chunk config
  is benchmarked.  When ``headdim`` equals a chunk boundary (64/128/256) it
  is deduplicated automatically because it would match an existing candidate.

  :param headdim: The actual head dimension for the target shape.  Used to
      optionally include a full-D ``BLOCK_HEADDIM`` config on high-SMEM
      architectures.  When the full-D config wins, ``NUM_V_GROUPS == 1`` and
      the V-group loop is unrolled to a single iteration.
  :return: Triton autotune configurations for the forward kernel.
  """
  try:
    _max_smem = torch.cuda.get_device_properties(0).shared_memory_per_block_optin
  except Exception:
    _max_smem = 48 * 1024  # safe fallback: default SMEM

  _headdim_candidates = [64, 128, 256]
  # Use triton.next_power_of_2(headdim) as a near-full-D single-chunk block size:
  #   - power-of-2 headdims (512, 1024): next_pow2 == headdim → NUM_V_GROUPS=1,
  #     eliminates the D-chunk loop entirely.
  #   - non-power-of-2 headdims (320→512, 640→1024): next_pow2 pads to the next
  #     power-of-2.  The kernel's load/store masks (qk_d < HEADDIM, o_d < HEADDIM)
  #     zero out the padding columns, so correctness is preserved.
  # tl.arange requires a power-of-2 range, so next_power_of_2 always produces a
  # valid block size.  Only included on high-SMEM devices to keep register pressure
  # manageable; skip when next_pow2 is already in [64, 128, 256] (dedup).
  _next_pow2 = triton.next_power_of_2(headdim)
  if _max_smem >= 96 * 1024 and _next_pow2 not in _headdim_candidates:  # 96 KB
    _headdim_candidates.append(_next_pow2)

  configs = []
  for block_m in [64, 128]:
    for block_headdim in _headdim_candidates:
      for num_warps in [4, 8]:
        for num_stages in [2, 3, 4]:
          configs.append(
            triton.Config(
              {
                "BLOCK_M": block_m,
                "BLOCK_N": 64,
                "BLOCK_HEADDIM_QK": block_headdim,
                "BLOCK_HEADDIM_V": block_headdim,
              },
              num_warps=num_warps,
              num_stages=num_stages,
            )
          )
  return configs


_FFPA_FWD_HEURISTICS = {
  "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
  "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
  "NUM_V_GROUPS": lambda args: triton.cdiv(args["HEADDIM"], args["BLOCK_HEADDIM_V"]),
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
  IS_CAUSAL: tl.constexpr,
  DTYPE: tl.constexpr,
  HEADDIM: tl.constexpr,
  EVEN_M: tl.constexpr,
  EVEN_N: tl.constexpr,
  BLOCK_M: tl.constexpr,
  BLOCK_N: tl.constexpr,
  BLOCK_HEADDIM_QK: tl.constexpr,
  BLOCK_HEADDIM_V: tl.constexpr,
  NUM_V_GROUPS: tl.constexpr,
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
  offs_d_v = tl.arange(0, BLOCK_HEADDIM_V)

  num_qk_d_chunks = tl.cdiv(HEADDIM, BLOCK_HEADDIM_QK)
  kv_offset = seqlen_k - seqlen_q

  m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
  l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
  zero_acc = tl.zeros([BLOCK_M, BLOCK_HEADDIM_V], dtype=tl.float32)
  # Mirrors CUDA fwd's R_D registers: one O accumulator per V head-dim slice.
  # Each accumulator is rescaled by the online-softmax alpha before adding
  # the current P @ V_slice contribution.
  o_accs = (zero_acc, ) * NUM_V_GROUPS

  end_n = seqlen_k
  if IS_CAUSAL:
    # Skip KV blocks that are fully masked by lower-right causal attention.
    end_n = tl.minimum(seqlen_k, (start_m + 1) * BLOCK_M + kv_offset)

  for start_n in range(0, end_n, BLOCK_N):
    start_n = tl.multiple_of(start_n, BLOCK_N)
    offs_kv = start_n + offs_n

    scores = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for qk_d_chunk in range(num_qk_d_chunks):
      qk_d_start = qk_d_chunk * BLOCK_HEADDIM_QK
      qk_d = qk_d_start + offs_d_qk
      q = tl.load(
        Q + offs_m[:, None] * stride_qm + qk_d[None, :],
        mask=(offs_m[:, None] < seqlen_q) & (qk_d[None, :] < HEADDIM),
        other=0.0,
      )
      k = tl.load(
        K + offs_kv[:, None] * stride_kn + qk_d[None, :],
        mask=(offs_kv[:, None] < seqlen_k) & (qk_d[None, :] < HEADDIM),
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
    p = p.to(DTYPE)

    # Reuse the same softmax tile for all V slices, matching FFPA CUDA fwd:
    # R_D[j] = alpha * R_D[j] + P @ V_j. This avoids recomputing QK/softmax
    # per output head-dim group while keeping Split-D register pressure bounded.
    for v_group in tl.static_range(0, NUM_V_GROUPS):
      o_d = BLOCK_HEADDIM_V * v_group + offs_d_v
      v = tl.load(
        V + offs_kv[:, None] * stride_vn + o_d[None, :],
        mask=(offs_kv[:, None] < seqlen_k) & (o_d[None, :] < HEADDIM),
        other=0.0,
      )
      o_acc = o_accs[v_group] * alpha[:, None] + tl.dot(p, v)
      o_accs = _update_o_accs(o_accs, v_group, o_acc)
    m_i = m_new
    l_i = l_new

  for v_group in tl.static_range(0, NUM_V_GROUPS):
    o_d = BLOCK_HEADDIM_V * v_group + offs_d_v
    out = o_accs[v_group] / (l_i[:, None] + 1.0e-10)
    tl.store(
      O + offs_m[:, None] * stride_om + o_d[None, :],
      out.to(DTYPE),
      mask=(offs_m[:, None] < seqlen_q) & (o_d[None, :] < HEADDIM),
    )
  tl.store(LSE + offs_m, m_i + tl.log(l_i), mask=offs_m < seqlen_q)


_ffpa_fwd = _ffpa_fwd_kernel_impl

_ffpa_fwd_autotune_cache: dict[int, callable] = {}  # headdim -> callable


def _get_fwd_autotune(headdim: int):
  """Return a headdim-specific autotune wrapper for the forward kernel.

  Results are cached by headdim so the autotune overhead is paid at most once
  per shape on a given process.  The search space is generated by
  :func:`_gen_fwd_autotune_configs` which gates a full-D chunk config on
  device SMEM capacity.

  :param headdim: The actual head dimension for the target forward call.
  :return: A ``triton.autotune``-wrapped version of ``_ffpa_fwd_kernel_impl``
      tuned for ``headdim``.
  """
  if headdim not in _ffpa_fwd_autotune_cache:
    configs = _gen_fwd_autotune_configs(headdim)
    _ffpa_fwd_autotune_cache[headdim] = triton.autotune(
      configs=configs,
      key=["seqlen_q", "seqlen_k", "HEADDIM"],
      cache_results=True,
    )(_ffpa_fwd_kernel_impl)
  return _ffpa_fwd_autotune_cache[headdim]


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
  if headdim > _MAX_HEADDIM:
    raise ValueError(f"Triton forward supports headdim <= {_MAX_HEADDIM}, got {headdim}")

  DTYPE = tl.float16 if q.dtype == torch.float16 else tl.bfloat16

  def grid(meta: dict) -> tuple[int, int]:
    return (triton.cdiv(seqlen_q, meta["BLOCK_M"]), batch * nheads_q)

  if autotune:
    _get_fwd_autotune(headdim)[grid](
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
      IS_CAUSAL=causal,
      DTYPE=DTYPE,
      HEADDIM=headdim,
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
      IS_CAUSAL=causal,
      DTYPE=DTYPE,
      HEADDIM=headdim,
      BLOCK_M=128,
      BLOCK_N=64,
      BLOCK_HEADDIM_QK=64,
      BLOCK_HEADDIM_V=64,
      num_warps=8,
      num_stages=3,
    )


def _ffpa_attn_forward_triton(
  Q: torch.Tensor,
  K: torch.Tensor,
  V: torch.Tensor,
  O: torch.Tensor | None = None,
  causal: bool = False,
  softmax_scale: float = 0.0,
  autotune: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Call the Triton FFPA forward via registered torch op, returning ``(O, softmax_lse)``.

  The ``O`` parameter is accepted for API compatibility but ignored — the
  registered op always allocates a fresh output buffer.

  :returns: Output tensor and softmax LSE sliced to visible shape ``[B, Nh_q, Nq]``.
  """
  if Q.stride(-1) != 1:
    Q = Q.contiguous()
  if K.stride(-1) != 1:
    K = K.contiguous()
  if V.stride(-1) != 1:
    V = V.contiguous()
  del O

  seqlen_q = Q.size(2)
  O_storage, softmax_lse_storage = torch.ops.ffpa_attn._fwd_triton(
    Q,
    K,
    V,
    softmax_scale,
    int(causal),
    int(autotune),
  )
  return O_storage, softmax_lse_storage[..., :seqlen_q]

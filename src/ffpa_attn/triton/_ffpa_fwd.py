"""FFPA Attention Forward (Split-D) — Triton implementation.

This module provides a single Triton forward kernel for large-head-dim FFPA
prefill attention.  The program mapping follows FlashAttention v2: one program
owns a Q-row block for one batch/query-head pair.  Inside that program, the
large head dimension is processed in chunks so D=320/512 can be handled without
materialising the full attention matrix.

There are two execution paths:

* Generic path: one kernel streams the full KV sequence for each Q block,
  performs online softmax, and writes O/LSE directly.
* Decode path: used when split-kv improves occupancy for small query windows.
  Stage1 computes one partial O and chunk-local LSE per KV chunk; stage2 merges
  those partials with the log-sum-exp merge formula.

GQA/MQA is handled inside the Triton kernels by mapping each query head to a KV
head with ``off_hkv = off_hq // group_size``. Additive ``attn_bias`` follows
SDPA's logical score shape ``[B, Hq, Nq, Nkv]`` and may use stride-0 broadcast
dimensions, so compact masks such as ``[1, 1, 1, Nkv]`` never need to be
materialized. Dropout is replay-compatible with SDPA: each logical score element
uses one Philox output in row-major ``[B, Hq, Nq, Nkv]`` order.

The saved LSE uses the natural logarithm convention expected by the existing
Triton backward kernel: ``lse = log(sum(exp(score)))`` where
``score = softmax_scale * (Q @ K.T)`` after masking.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from ._autotune_utils import bucket_autotune_seqlen


def _attn_bias_broadcast_strides(
  attn_bias: torch.Tensor | None,
  batch: int,
  nheads: int,
  seqlen_q: int,
  seqlen_k: int,
) -> tuple[int, int, int, int]:
  """Return logical-score strides for a broadcastable 4-D attention bias.

  The Triton kernels always form ``AttnBias + m * stride_bm + n * stride_bn``
  as if the mask had full logical shape ``[B, Hq, Nq, Nkv]``. A stride of zero
  marks a user dimension of size 1 that broadcasts over the logical scores.

  :param attn_bias: Optional additive bias tensor already normalized to a 4-D
      broadcastable shape.
  :param batch: Runtime batch size of the logical score tensor.
  :param nheads: Runtime query-head count of the logical score tensor.
  :param seqlen_q: Runtime query sequence length.
  :param seqlen_k: Runtime KV sequence length.
  :return: ``(stride_b, stride_h, stride_m, stride_n)`` for logical indexing.
  """
  if attn_bias is None:
    return (0, 0, 0, 0)
  return (
    0 if attn_bias.size(0) == 1 and batch > 1 else attn_bias.stride(0),
    0 if attn_bias.size(1) == 1 and nheads > 1 else attn_bias.stride(1),
    0 if attn_bias.size(2) == 1 and seqlen_q > 1 else attn_bias.stride(2),
    0 if attn_bias.size(3) == 1 and seqlen_k > 1 else attn_bias.stride(3),
  )


_MAX_HEADDIM = 1024


@triton.jit
def _update_o_accs(o_accs, v_group: tl.constexpr, o_acc):
  return o_accs[:v_group] + (o_acc, ) + o_accs[v_group + 1:]


@triton.jit
def _curand_uniform_from_element_offset(seed: tl.constexpr, element_offset):
  # PyTorch mem-efficient attention maps each logical score element in
  # [B, H, Nq, Nkv] to one Philox output.  Use the same uint32 -> uniform
  # conversion as curand_uniform4; going through signed int32 can introduce a
  # one-ulp mismatch for high-bit values, which is enough to break exact mask
  # parity at the dropout threshold.
  quad_offset = element_offset // 4
  lane = element_offset - quad_offset * 4
  r0, r1, r2, r3 = tl.randint4x(seed, quad_offset)
  r = tl.where(lane == 0, r0, tl.where(lane == 1, r1, tl.where(lane == 2, r2, r3)))
  r_u32 = r.to(tl.uint32, bitcast=True)
  return (r_u32.to(tl.float32) + 1.0) * 2.3283064365386963e-10


@triton.jit
def _apply_dropout_to_p(
  p,
  off_hb,
  offs_m,
  offs_n,
  seqlen_q: int,
  seqlen_k: int,
  dropout_p: float,
  philox_seed: tl.constexpr,
  philox_offset: int,
  HAS_DROPOUT: tl.constexpr,
):
  # Forward and backward both depend on the exact same dropout RNG mapping.
  # ``offs_n`` must be global KV positions, including split-kv chunk offsets,
  # so the element_offset matches SDPA's logical [B, H, Nq, Nkv] score layout.
  if HAS_DROPOUT:
    # Keep this in SDPA's logical score order.  In split-kv decode, offs_n is
    # passed as the global KV index, so no extra chunk offset should be added.
    linear = off_hb * seqlen_q * seqlen_k + offs_m[:, None] * seqlen_k + offs_n[None, :]
    rand = _curand_uniform_from_element_offset(philox_seed, philox_offset + linear)
    keep = rand > dropout_p
    p = p * keep * (1.0 / (1.0 - dropout_p))
  return p


def _gen_fwd_autotune_configs(headdim: int = 256, autotune_mode: str = "max") -> list[triton.Config]:
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
  # 96 KB is the minimum SMEM for a full-D block with FFPA's current memory layout
  if all([
    _max_smem >= 96 * 1024,
    _next_pow2 > 256,
    _next_pow2 <= _MAX_HEADDIM,
    autotune_mode == "max",
  ]):
    _headdim_candidates.append(_next_pow2)

  if autotune_mode == "fast":
    _headdim_candidates = [c for c in _headdim_candidates if c <= 128]

  configs = []
  for block_m in [64, 128]:
    for block_headdim in _headdim_candidates:
      num_warps_candidates = [8] if autotune_mode == "fast" else [4, 8]
      for num_warps in num_warps_candidates:
        for num_stages in ([2, 3] if autotune_mode == "fast" else [2, 3, 4]):
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


def _gen_decode_fwd_stage1_autotune_configs(
  headdim: int = 256,
  use_gemv: bool = False,
  autotune_mode: str = "max",
) -> list[triton.Config]:
  """Generate headdim-specific autotune configs for decode stage1.

  The decode stage1 search space is intentionally compact. ``CHUNK_SIZE`` is
  owned by the launcher via ``num_splits`` and is passed at runtime instead of
  being autotuned here. GEMV and multi-row MMA paths are tuned separately so
  full-D tiles are only explored for the single-row path.

  :param headdim: The actual head dimension for the target decode call.
  :param use_gemv: Whether configs are generated for the ``seqlen_q == 1``
      GEMV path.
  :return: Triton autotune configurations for decode stage1.
  """
  try:
    _max_smem = torch.cuda.get_device_properties(0).shared_memory_per_block_optin
  except Exception:
    _max_smem = 48 * 1024

  _headdim_candidates = [64, 128]
  _next_pow2 = triton.next_power_of_2(headdim)
  if all([
    use_gemv,
    _max_smem >= 96 * 1024,
    _next_pow2 > 128,
    _next_pow2 <= _MAX_HEADDIM,
    autotune_mode == "max",
  ]):
    _headdim_candidates.append(_next_pow2)

  if use_gemv:
    block_m_candidates = [8]
    block_n_candidates = [64, 128]
    if autotune_mode == "max":
      block_n_candidates.append(256)
  else:
    block_m_candidates = [8, 16, 32] if autotune_mode == "max" else [8, 16]
    block_n_candidates = [64, 128]
    if autotune_mode == "max":
      block_n_candidates.append(256)

  if autotune_mode == "fast":
    _headdim_candidates = [c for c in _headdim_candidates if c <= 128]

  configs = []
  for block_n in block_n_candidates:
    for block_m in block_m_candidates:
      for block_headdim in _headdim_candidates:
        num_warps_candidates = [8]
        if autotune_mode == "max" and not use_gemv and block_m >= 32 and block_n >= 128:
          num_warps_candidates.append(4)
        for num_warps in num_warps_candidates:
          for num_stages in ([2] if autotune_mode == "fast" else [2, 3]):
            configs.append(
              triton.Config(
                {
                  "BLOCK_M": block_m,
                  "BLOCK_N": block_n,
                  "BLOCK_HEADDIM_QK": block_headdim,
                  "BLOCK_HEADDIM_V": block_headdim,
                },
                num_warps=num_warps,
                num_stages=num_stages,
              )
            )
  return configs


def _decode_num_splits_heuristic(
  batch_nheads_mblocks: int,
  num_sms: int,
  num_n_blocks: int,
  max_splits: int,
) -> int:
  """Mirror FlashAttention's split-kv occupancy heuristic in Python.

  :param batch_nheads_mblocks: Number of independent query row blocks.
  :param num_sms: Effective SM count available to the kernel launch.
  :param num_n_blocks: Number of KV tiles along sequence length.
  :param max_splits: Upper bound for the split search space.
  :return: The chosen split count.
  """
  if batch_nheads_mblocks >= 0.8 * num_sms:
    return 1

  max_splits = max(1, min(max_splits, num_sms, num_n_blocks))
  max_efficiency = 0.0
  efficiencies: list[float] = []

  def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b

  def _is_split_eligible(num_splits: int) -> bool:
    return num_splits == 1 or _ceil_div(num_n_blocks, num_splits) != _ceil_div(
      num_n_blocks,
      num_splits - 1,
    )

  for num_splits in range(1, max_splits + 1):
    if not _is_split_eligible(num_splits):
      efficiencies.append(0.0)
      continue
    n_waves = float(batch_nheads_mblocks * num_splits) / float(num_sms)
    efficiency = n_waves / math.ceil(n_waves)
    max_efficiency = max(max_efficiency, efficiency)
    efficiencies.append(efficiency)

  for num_splits in range(1, max_splits + 1):
    if not _is_split_eligible(num_splits):
      continue
    if efficiencies[num_splits - 1] >= 0.85 * max_efficiency:
      return num_splits

  return 1


def _get_decode_num_splits(
  seqlen_q: int,
  seqlen_k: int,
  headdim: int,
  batch: int,
  nheads_q: int,
  device: torch.device,
) -> int:
  """Choose a decode split count using FlashAttention's split-kv heuristic."""
  num_sms = max(1, torch.cuda.get_device_properties(device).multi_processor_count * 2)
  block_n = 256 if headdim <= 64 else (128 if headdim <= 128 else 64)
  num_n_blocks = triton.cdiv(seqlen_k, block_n)
  num_m_blocks = triton.cdiv(seqlen_q, 64)
  num_splits = _decode_num_splits_heuristic(
    batch * nheads_q * num_m_blocks,
    num_sms,
    num_n_blocks,
    max_splits=128,
  )
  return num_splits


_FFPA_FWD_HEURISTICS = {
  "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
  "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
  "NUM_V_GROUPS": lambda args: triton.cdiv(args["HEADDIM"], args["BLOCK_HEADDIM_V"]),
}

_FFPA_DECODE_FWD_HEURISTICS = {
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
  AttnBias: torch.Tensor,
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
  stride_bb: int,
  stride_bh: int,
  stride_bm: int,
  stride_bn: int,
  nheads_q: int,
  nheads_kv: int,
  seqlen_q: int,
  seqlen_k: int,
  # Autotune buckets are passed explicitly to avoid redundant autotune
  # runs for shapes that differ only in seqlen but fall in the same bucket.
  # The kernel itself only uses the bucketed values.
  seqlen_q_bucket: int,
  seqlen_k_bucket: int,
  seqlen_q_rounded: int,
  dropout_p: float,
  philox_offset: int,
  IS_CAUSAL: tl.constexpr,
  HAS_ATTN_BIAS: tl.constexpr,
  HAS_DROPOUT: tl.constexpr,
  PHILOX_SEED: tl.constexpr,
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
  """Run the generic single-kernel Split-D FFPA forward path.

  One program owns one Q block for one logical query head. It streams KV blocks,
  reconstructs QK over head-dim chunks, applies masking/bias/dropout, and keeps
  an online-softmax accumulator for each V head-dim slice.
  """
  start_m = tl.program_id(0)
  off_hb = tl.program_id(1)
  off_b = off_hb // nheads_q
  off_hq = off_hb % nheads_q
  # GQA/MQA: multiple query heads can share one KV head. The output and LSE are
  # query-head indexed; K/V loads are mapped back to the owning KV head.
  group_size = nheads_q // nheads_kv
  off_hkv = off_hq // group_size

  Q += off_b * stride_qb + off_hq * stride_qh
  K += off_b * stride_kb + off_hkv * stride_kh
  V += off_b * stride_vb + off_hkv * stride_vh
  O += off_b * stride_ob + off_hq * stride_oh
  LSE += off_hb * seqlen_q_rounded
  if HAS_ATTN_BIAS:
    AttnBias += off_b * stride_bb + off_hq * stride_bh

  offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
  offs_n = tl.arange(0, BLOCK_N)
  offs_d_qk = tl.arange(0, BLOCK_HEADDIM_QK)
  offs_d_v = tl.arange(0, BLOCK_HEADDIM_V)

  num_qk_d_chunks = tl.cdiv(HEADDIM, BLOCK_HEADDIM_QK)
  kv_offset = seqlen_k - seqlen_q

  m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
  l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
  # Always use fp32 accumulators for O to reduce numerical instability;
  # the final output is downcast to the input dtype at store time.
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
    if HAS_ATTN_BIAS:
      # Broadcasted mask dimensions have stride 0, so this covers full masks
      # and compact masks with the same pointer expression.
      bias = tl.load(
        AttnBias + offs_m[:, None] * stride_bm + offs_kv[None, :] * stride_bn,
        mask=(offs_m[:, None] < seqlen_q) & (offs_kv[None, :] < seqlen_k),
        other=0.0,
      )
      scores += bias
    if not EVEN_N:
      scores = tl.where(offs_kv[None, :] < seqlen_k, scores, -float("inf"))
    if IS_CAUSAL:
      # Lower-right causal semantics for Nq <= Nkv. Query row m can attend to
      # key positions <= m + (Nkv - Nq), matching PyTorch SDPA.
      causal_mask = offs_kv[None, :] <= (offs_m[:, None] + kv_offset)
      scores = tl.where(causal_mask, scores, -float("inf"))

    # Online softmax merge for the next KV block. ``m_i`` and ``l_i`` track the
    # row max and denominator before dropout; ``p`` is dropped only for the O
    # accumulation, while LSE stays the undropped softmax normalizer.
    m_new = tl.maximum(m_i, tl.max(scores, axis=1))
    alpha = tl.exp(m_i - m_new)
    p = tl.exp(scores - m_new[:, None])
    l_new = l_i * alpha + tl.sum(p, axis=1)
    p = _apply_dropout_to_p(
      p,
      off_hb,
      offs_m,
      offs_kv,
      seqlen_q,
      seqlen_k,
      dropout_p,
      PHILOX_SEED,
      philox_offset,
      HAS_DROPOUT,
    )
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


@triton.heuristics(_FFPA_DECODE_FWD_HEURISTICS)
@triton.jit
def _ffpa_decode_fwd_stage1_kernel(
  Q: torch.Tensor,
  K: torch.Tensor,
  V: torch.Tensor,
  PartialOut: torch.Tensor,
  ChunkLSE: torch.Tensor,
  AttnBias: torch.Tensor,
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
  stride_pb: int,
  stride_ph: int,
  stride_pc: int,
  stride_pm: int,
  stride_lb: int,
  stride_lh: int,
  stride_lc: int,
  stride_lm: int,
  stride_bb: int,
  stride_bh: int,
  stride_bm: int,
  stride_bn: int,
  nheads_q: int,
  nheads_kv: int,
  seqlen_q: int,
  seqlen_k: int,
  # Autotune buckets are passed explicitly to avoid redundant autotune
  # runs for shapes that differ only in seqlen but fall in the same bucket.
  # The kernel itself only uses the bucketed values.
  seqlen_q_bucket: int,
  seqlen_k_bucket: int,
  dropout_p: float,
  philox_offset: int,
  IS_CAUSAL: tl.constexpr,
  HAS_ATTN_BIAS: tl.constexpr,
  HAS_DROPOUT: tl.constexpr,
  PHILOX_SEED: tl.constexpr,
  USE_GEMV: tl.constexpr,
  DTYPE: tl.constexpr,
  HEADDIM: tl.constexpr,
  BLOCK_M: tl.constexpr,
  CHUNK_SIZE: tl.constexpr,
  BLOCK_N: tl.constexpr,
  BLOCK_HEADDIM_QK: tl.constexpr,
  BLOCK_HEADDIM_V: tl.constexpr,
  NUM_V_GROUPS: tl.constexpr,
) -> None:
  # Split-kv decode stage1. Each program owns (KV chunk, B/Hq, Q block) and
  # writes a partial output plus chunk-local LSE. Stage2 merges chunks using
  # their LSEs, so stage1 never writes the final O/LSE directly.
  chunk_idx = tl.program_id(0)
  off_hb = tl.program_id(1)
  q_block = tl.program_id(2)
  off_b = off_hb // nheads_q
  off_hq = off_hb % nheads_q
  # Same GQA/MQA contract as the generic path: query heads select their KV head
  # through integer grouping, without materializing expanded K/V tensors.
  group_size = nheads_q // nheads_kv
  off_hkv = off_hq // group_size

  Q += off_b * stride_qb + off_hq * stride_qh
  K += off_b * stride_kb + off_hkv * stride_kh
  V += off_b * stride_vb + off_hkv * stride_vh
  PartialOut += off_b * stride_pb + off_hq * stride_ph + chunk_idx * stride_pc
  ChunkLSE += off_b * stride_lb + off_hq * stride_lh + chunk_idx * stride_lc
  if HAS_ATTN_BIAS:
    AttnBias += off_b * stride_bb + off_hq * stride_bh

  chunk_start = chunk_idx * CHUNK_SIZE
  chunk_end = tl.minimum(seqlen_k, chunk_start + CHUNK_SIZE)
  kv_offset = seqlen_k - seqlen_q
  offs_m = q_block * BLOCK_M + tl.arange(0, BLOCK_M)
  mask_m = offs_m < seqlen_q
  offs_n = tl.arange(0, BLOCK_N)
  offs_d_qk = tl.arange(0, BLOCK_HEADDIM_QK)
  offs_d_v = tl.arange(0, BLOCK_HEADDIM_V)
  num_qk_d_chunks = tl.cdiv(HEADDIM, BLOCK_HEADDIM_QK)

  if USE_GEMV:  # gemv
    # Single-query decode path. Use vector reductions instead of MMA tiles so a
    # one-row query does not pay matrix-tile overhead.
    m_i_single = -float("inf")
    l_i_single = 0.0
    zero_acc_single = tl.zeros([BLOCK_HEADDIM_V], dtype=tl.float32)
    o_accs_single = (zero_acc_single, ) * NUM_V_GROUPS

    for start_n in range(0, CHUNK_SIZE, BLOCK_N):
      start_n = tl.multiple_of(start_n, BLOCK_N)
      offs_kv = chunk_start + start_n + offs_n
      mask_n = offs_kv < chunk_end

      scores = tl.zeros([1, BLOCK_N], dtype=tl.float32)
      for qk_d_chunk in range(num_qk_d_chunks):
        qk_d_start = qk_d_chunk * BLOCK_HEADDIM_QK
        qk_d = qk_d_start + offs_d_qk
        q = tl.load(
          Q + qk_d,
          mask=qk_d < HEADDIM,
          other=0.0,
        )
        k = tl.load(
          K + offs_kv[:, None] * stride_kn + qk_d[None, :],
          mask=mask_n[:, None] & (qk_d[None, :] < HEADDIM),
          other=0.0,
        )
        scores += tl.sum(k * q[None, :], axis=1)[None, :]

      scores = tl.sum(scores, axis=0)
      scores = scores * softmax_scale
      if HAS_ATTN_BIAS:
        # Nq == 1, so the query offset is exactly q_block * BLOCK_M. KV offsets
        # are global positions inside the full sequence, not chunk-local ids.
        bias = tl.load(
          AttnBias + (q_block * BLOCK_M) * stride_bm + offs_kv * stride_bn,
          mask=mask_n,
          other=0.0,
        )
        scores += bias
      scores = tl.where(mask_n, scores, -float("inf"))
      if IS_CAUSAL:
        causal_mask = offs_kv <= (seqlen_k - 1)
        scores = tl.where(causal_mask, scores, -float("inf"))

      m_new = tl.maximum(m_i_single, tl.max(scores, axis=0))
      alpha = tl.exp(m_i_single - m_new)
      p = tl.exp(scores - m_new)
      l_new = l_i_single * alpha + tl.sum(p, axis=0)
      if HAS_DROPOUT:
        # offs_kv already includes chunk_start, matching the global Nkv axis in
        # SDPA's [B, H, Nq, Nkv] dropout RNG layout.
        linear = off_hb * seqlen_q * seqlen_k + (q_block * BLOCK_M) * seqlen_k + offs_kv
        rand = _curand_uniform_from_element_offset(PHILOX_SEED, philox_offset + linear)
        keep = rand > dropout_p
        p = p * keep * (1.0 / (1.0 - dropout_p))
      p = p.to(DTYPE)

      for v_group in tl.static_range(0, NUM_V_GROUPS):
        o_d = BLOCK_HEADDIM_V * v_group + offs_d_v
        v = tl.load(
          V + offs_kv[:, None] * stride_vn + o_d[None, :],
          mask=mask_n[:, None] & (o_d[None, :] < HEADDIM),
          other=0.0,
        )
        o_acc = o_accs_single[v_group] * alpha + tl.sum(v * p[:, None], axis=0)
        o_accs_single = _update_o_accs(o_accs_single, v_group, o_acc)

      m_i_single = m_new
      l_i_single = l_new

    for v_group in tl.static_range(0, NUM_V_GROUPS):
      o_d = BLOCK_HEADDIM_V * v_group + offs_d_v
      out = o_accs_single[v_group] / (l_i_single + 1.0e-10)
      tl.store(
        PartialOut + o_d,
        out,
        mask=o_d < HEADDIM,
      )
    tl.store(ChunkLSE, m_i_single + tl.log(l_i_single))

  else:  # MMA gemm

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    zero_acc = tl.zeros([BLOCK_M, BLOCK_HEADDIM_V], dtype=tl.float32)
    o_accs = (zero_acc, ) * NUM_V_GROUPS

    for start_n in range(0, CHUNK_SIZE, BLOCK_N):
      start_n = tl.multiple_of(start_n, BLOCK_N)
      offs_kv = chunk_start + start_n + offs_n
      mask_n = offs_kv < chunk_end

      scores = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
      for qk_d_chunk in range(num_qk_d_chunks):
        qk_d_start = qk_d_chunk * BLOCK_HEADDIM_QK
        qk_d = qk_d_start + offs_d_qk
        q = tl.load(
          Q + offs_m[:, None] * stride_qm + qk_d[None, :],
          mask=mask_m[:, None] & (qk_d[None, :] < HEADDIM),
          other=0.0,
        )
        k = tl.load(
          K + offs_kv[:, None] * stride_kn + qk_d[None, :],
          mask=mask_n[:, None] & (qk_d[None, :] < HEADDIM),
          other=0.0,
        )
        scores = tl.dot(q, tl.trans(k), acc=scores)

      scores = scores * softmax_scale
      if HAS_ATTN_BIAS:
        # ``offs_kv`` already includes chunk_start. This keeps additive masks
        # and dropout RNG aligned with the global SDPA score matrix.
        bias = tl.load(
          AttnBias + offs_m[:, None] * stride_bm + offs_kv[None, :] * stride_bn,
          mask=mask_m[:, None] & mask_n[None, :],
          other=0.0,
        )
        scores += bias
      scores = tl.where(mask_n[None, :], scores, -float("inf"))
      if IS_CAUSAL:
        # Tail-aligned causal mask for decode windows. Rows outside the visible
        # query length are masked later by ``mask_m``.
        causal_mask = offs_kv[None, :] <= (offs_m[:, None] + kv_offset)
        scores = tl.where(causal_mask, scores, -float("inf"))
      scores = tl.where(mask_m[:, None], scores, -float("inf"))

      m_prev = tl.where(mask_m, m_i, 0.0)
      m_new = tl.where(mask_m, tl.maximum(m_i, tl.max(scores, axis=1)), 0.0)
      alpha = tl.exp(m_prev - m_new)
      p = tl.exp(scores - m_new[:, None])
      l_new = l_i * alpha + tl.sum(p, axis=1)
      p = _apply_dropout_to_p(
        p,
        off_hb,
        offs_m,
        offs_kv,
        seqlen_q,
        seqlen_k,
        dropout_p,
        PHILOX_SEED,
        philox_offset,
        HAS_DROPOUT,
      )
      p = p.to(DTYPE)

      for v_group in tl.static_range(0, NUM_V_GROUPS):
        o_d = BLOCK_HEADDIM_V * v_group + offs_d_v
        v = tl.load(
          V + offs_kv[:, None] * stride_vn + o_d[None, :],
          mask=mask_n[:, None] & (o_d[None, :] < HEADDIM),
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
        PartialOut + offs_m[:, None] * stride_pm + o_d[None, :],
        out,
        mask=mask_m[:, None] & (o_d[None, :] < HEADDIM),
      )
    tl.store(ChunkLSE + offs_m * stride_lm, m_i + tl.log(l_i), mask=mask_m)


@triton.jit
def _ffpa_decode_fwd_stage2_kernel(
  PartialOut: torch.Tensor,
  ChunkLSE: torch.Tensor,
  O: torch.Tensor,
  LSE: torch.Tensor,
  stride_pb: int,
  stride_ph: int,
  stride_pc: int,
  stride_pm: int,
  stride_lb: int,
  stride_lh: int,
  stride_lc: int,
  stride_lm: int,
  stride_ob: int,
  stride_oh: int,
  stride_om: int,
  stride_lseb: int,
  stride_lseh: int,
  stride_lsem: int,
  nheads_q: int,
  seqlen_q: int,
  n_chunks: int,
  DTYPE: tl.constexpr,
  HEADDIM: tl.constexpr,
  BLOCK_HEADDIM_V: tl.constexpr,
  BLOCK_CHUNKS: tl.constexpr,
) -> None:
  # Split-kv stage2. Each program merges one (B, Hq, query row, V slice) across
  # all chunk partials. The numerically stable merge is:
  #   O = sum_c exp(LSE_c - LSE) * O_c,  LSE = logsumexp_c(LSE_c)
  # where O_c was normalized within its chunk by stage1.
  off_hbm = tl.program_id(0)
  v_group = tl.program_id(1)
  off_hb = off_hbm // seqlen_q
  off_m = off_hbm % seqlen_q
  off_b = off_hb // nheads_q
  off_hq = off_hb % nheads_q

  PartialOut += off_b * stride_pb + off_hq * stride_ph + off_m * stride_pm
  ChunkLSE += off_b * stride_lb + off_hq * stride_lh + off_m * stride_lm
  O += off_b * stride_ob + off_hq * stride_oh + off_m * stride_om
  LSE += off_b * stride_lseb + off_hq * stride_lseh + off_m * stride_lsem

  offs_c = tl.arange(0, BLOCK_CHUNKS)
  mask_c = offs_c < n_chunks
  chunk_lse = tl.load(
    ChunkLSE + offs_c * stride_lc,
    mask=mask_c,
    other=-float("inf"),
  )
  valid_c = mask_c & (chunk_lse > -float("inf"))
  max_lse = tl.max(chunk_lse, axis=0)
  weights = tl.where(valid_c, tl.exp(chunk_lse - max_lse), 0.0)
  denom = tl.sum(weights, axis=0)

  offs_d = v_group * BLOCK_HEADDIM_V + tl.arange(0, BLOCK_HEADDIM_V)
  partial = tl.load(
    PartialOut + offs_c[:, None] * stride_pc + offs_d[None, :],
    mask=valid_c[:, None] & (offs_d[None, :] < HEADDIM),
    other=0.0,
  )
  out = tl.sum(weights[:, None] * partial, axis=0) / denom
  tl.store(
    O + offs_d,
    out.to(DTYPE),
    mask=offs_d < HEADDIM,
  )
  if v_group == 0:
    tl.store(LSE, max_lse + tl.log(denom))


_ffpa_decode_fwd_stage1 = _ffpa_decode_fwd_stage1_kernel
_ffpa_decode_fwd_stage1_autotune_cache: dict[tuple[int, bool, str], callable] = {}


def _get_decode_fwd_stage1_autotune(headdim: int, use_gemv: bool, autotune_mode: str):
  """Return a shape-class-specific autotune wrapper for decode stage1."""
  cache_key = (headdim, use_gemv, autotune_mode)
  if cache_key not in _ffpa_decode_fwd_stage1_autotune_cache:
    configs = _gen_decode_fwd_stage1_autotune_configs(
      headdim,
      use_gemv=use_gemv,
      autotune_mode=autotune_mode,
    )
    _ffpa_decode_fwd_stage1_autotune_cache[cache_key] = triton.autotune(
      configs=configs,
      key=["seqlen_q_bucket", "seqlen_k_bucket", "HEADDIM"],
      cache_results=True,
    )(_ffpa_decode_fwd_stage1_kernel)
  return _ffpa_decode_fwd_stage1_autotune_cache[cache_key]


def _ffpa_attn_forward_generic_impl(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  o: torch.Tensor,
  lse: torch.Tensor,
  attn_bias: torch.Tensor | None = None,
  causal: bool = False,
  softmax_scale: float | None = None,
  autotune: bool = False,
  autotune_mode: str = "fast",
  dropout_p: float = 0.0,
  philox_seed: int = 0,
  philox_offset: int = 0,
) -> None:
  """Launch the generic Triton forward kernel without split-kv scratch.

  This path is selected when the split-kv occupancy heuristic returns one
  split. The kernel streams all KV blocks in one program per Q block and writes
  final ``o`` and natural-log ``lse`` in place.

  :param q: Query tensor in ``[B, Hq, Nq, D]`` layout.
  :param k: Key tensor in ``[B, Hkv, Nkv, D]`` layout.
  :param v: Value tensor in ``[B, Hkv, Nkv, D]`` layout.
  :param o: Output tensor in ``[B, Hq, Nq, D]`` layout, written in place.
  :param lse: Float32 LSE tensor with rounded last-dimension storage.
  :param attn_bias: Optional additive mask broadcastable to
    ``[B, Hq, Nq, Nkv]``.
  :param causal: Whether to apply lower-right causal masking.
  :param softmax_scale: Scale applied to ``Q @ K.T``. Defaults to
    ``1 / sqrt(D)`` when ``None``.
  :param autotune: Whether to use the Triton autotuned entry for this shape.
  :param autotune_mode: Triton autotune search-space mode, ``"fast"`` or
    ``"max"``.
  :param dropout_p: Forward dropout probability. The same Philox state must be
    saved for backward replay.
  :param philox_seed: Philox seed used for dropout. Ignored when
    ``dropout_p == 0``.
  :param philox_offset: Philox element offset used for dropout replay parity
    with SDPA.
  """
  batch, nheads_q, seqlen_q, headdim = q.shape
  _, nheads_kv, seqlen_k, _ = k.shape
  softmax_scale = softmax_scale or (1.0 / math.sqrt(headdim))
  seqlen_q_rounded = lse.shape[-1]
  seqlen_q_bucket = bucket_autotune_seqlen(seqlen_q)
  seqlen_k_bucket = bucket_autotune_seqlen(seqlen_k)
  DTYPE = tl.float16 if q.dtype == torch.float16 else tl.bfloat16
  has_attn_bias = attn_bias is not None
  has_dropout = dropout_p > 0.0
  attn_bias_in = attn_bias if attn_bias is not None else q
  bias_strides = _attn_bias_broadcast_strides(attn_bias, batch, nheads_q, seqlen_q, seqlen_k)

  def grid(meta: dict) -> tuple[int, int]:
    return (triton.cdiv(seqlen_q, meta["BLOCK_M"]), batch * nheads_q)

  if autotune:
    _get_fwd_autotune(headdim, autotune_mode)[grid](
      q,
      k,
      v,
      o,
      lse,
      attn_bias_in,
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
      bias_strides[0],
      bias_strides[1],
      bias_strides[2],
      bias_strides[3],
      nheads_q,
      nheads_kv,
      seqlen_q,
      seqlen_k,
      seqlen_q_bucket,
      seqlen_k_bucket,
      seqlen_q_rounded,
      dropout_p,
      philox_offset,
      IS_CAUSAL=causal,
      HAS_ATTN_BIAS=has_attn_bias,
      HAS_DROPOUT=has_dropout,
      PHILOX_SEED=philox_seed,
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
      attn_bias_in,
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
      bias_strides[0],
      bias_strides[1],
      bias_strides[2],
      bias_strides[3],
      nheads_q,
      nheads_kv,
      seqlen_q,
      seqlen_k,
      seqlen_q_bucket,
      seqlen_k_bucket,
      seqlen_q_rounded,
      dropout_p,
      philox_offset,
      IS_CAUSAL=causal,
      HAS_ATTN_BIAS=has_attn_bias,
      HAS_DROPOUT=has_dropout,
      PHILOX_SEED=philox_seed,
      DTYPE=DTYPE,
      HEADDIM=headdim,
      BLOCK_M=128,
      BLOCK_N=64,
      BLOCK_HEADDIM_QK=64,
      BLOCK_HEADDIM_V=64,
      num_warps=8,
      num_stages=3,
    )


def _ffpa_attn_forward_decode_impl(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  o: torch.Tensor,
  lse: torch.Tensor,
  attn_bias: torch.Tensor | None = None,
  causal: bool = False,
  softmax_scale: float | None = None,
  autotune: bool = False,
  autotune_mode: str = "fast",
  num_splits: int | None = None,
  dropout_p: float = 0.0,
  philox_seed: int = 0,
  philox_offset: int = 0,
) -> None:
  """Run the split-kv Triton forward path used for decode-like shapes.

  Stage1 splits the KV sequence into ``num_splits`` chunks and writes fp32
  scratch tensors ``partial_out`` and ``chunk_lse``. Stage2 merges those chunks
  into the final output using a log-sum-exp weighted sum. This path is usually
  selected for small ``Nq`` and long ``Nkv`` where splitting improves SM
  occupancy.

  :param q: Query tensor in ``[B, Hq, Nq, D]`` layout.
  :param k: Key tensor in ``[B, Hkv, Nkv, D]`` layout.
  :param v: Value tensor in ``[B, Hkv, Nkv, D]`` layout.
  :param o: Output tensor in ``[B, Hq, Nq, D]`` layout, written in place.
  :param lse: Float32 visible LSE tensor in ``[B, Hq, Nq]`` layout.
  :param attn_bias: Optional additive mask broadcastable to
    ``[B, Hq, Nq, Nkv]``.
  :param causal: Whether to apply lower-right causal masking.
  :param softmax_scale: Scale applied to ``Q @ K.T``. Defaults to
    ``1 / sqrt(D)`` when ``None``.
  :param autotune: Whether to use the Triton autotuned stage1 entry.
  :param autotune_mode: Triton autotune search-space mode, ``"fast"`` or
    ``"max"``.
  :param num_splits: Optional explicit split count. When ``None``, a
    FlashAttention-style occupancy heuristic chooses it.
  :param dropout_p: Forward dropout probability.
  :param philox_seed: Philox seed used for dropout. Ignored when
    ``dropout_p == 0``.
  :param philox_offset: Philox element offset used for SDPA-compatible dropout
    RNG layout.
  """
  batch, nheads_q, seqlen_q, headdim = q.shape
  _, nheads_kv, seqlen_k, _ = k.shape
  softmax_scale = softmax_scale or (1.0 / math.sqrt(headdim))
  DTYPE = tl.float16 if q.dtype == torch.float16 else tl.bfloat16
  use_gemv = seqlen_q == 1
  seqlen_q_bucket = bucket_autotune_seqlen(seqlen_q)
  seqlen_k_bucket = bucket_autotune_seqlen(seqlen_k)
  if num_splits is None:
    num_splits = _get_decode_num_splits(seqlen_q, seqlen_k, headdim, batch, nheads_q, q.device)
  has_attn_bias = attn_bias is not None
  has_dropout = dropout_p > 0.0
  attn_bias_in = attn_bias if attn_bias is not None else q
  bias_strides = _attn_bias_broadcast_strides(attn_bias, batch, nheads_q, seqlen_q, seqlen_k)

  n_chunks = num_splits
  chunk_size = triton.cdiv(seqlen_k, n_chunks)
  block_m = 8 if use_gemv else min(64, max(8, triton.next_power_of_2(seqlen_q)))
  block_headdim = triton.next_power_of_2(headdim) if use_gemv else 64

  partial_out = torch.empty(
    (batch, nheads_q, n_chunks, seqlen_q, headdim),
    device=q.device,
    dtype=torch.float32,
  )
  chunk_lse = torch.empty(
    (batch, nheads_q, n_chunks, seqlen_q),
    device=q.device,
    dtype=torch.float32,
  )
  if autotune:
    chunk_lse.fill_(-float("inf"))

  def stage1_grid(meta: dict) -> tuple[int, int, int]:
    return (
      triton.cdiv(seqlen_k, meta["CHUNK_SIZE"]),
      batch * nheads_q,
      triton.cdiv(seqlen_q, meta["BLOCK_M"]),
    )

  if autotune:
    _get_decode_fwd_stage1_autotune(headdim, use_gemv, autotune_mode)[stage1_grid](
      q,
      k,
      v,
      partial_out,
      chunk_lse,
      attn_bias_in,
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
      partial_out.stride(0),
      partial_out.stride(1),
      partial_out.stride(2),
      partial_out.stride(3),
      chunk_lse.stride(0),
      chunk_lse.stride(1),
      chunk_lse.stride(2),
      chunk_lse.stride(3),
      bias_strides[0],
      bias_strides[1],
      bias_strides[2],
      bias_strides[3],
      nheads_q,
      nheads_kv,
      seqlen_q,
      seqlen_k,
      seqlen_q_bucket,
      seqlen_k_bucket,
      dropout_p,
      philox_offset,
      IS_CAUSAL=causal,
      HAS_ATTN_BIAS=has_attn_bias,
      HAS_DROPOUT=has_dropout,
      PHILOX_SEED=philox_seed,
      USE_GEMV=use_gemv,
      DTYPE=DTYPE,
      HEADDIM=headdim,
      CHUNK_SIZE=chunk_size,
    )
  else:
    _ffpa_decode_fwd_stage1[stage1_grid](
      q,
      k,
      v,
      partial_out,
      chunk_lse,
      attn_bias_in,
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
      partial_out.stride(0),
      partial_out.stride(1),
      partial_out.stride(2),
      partial_out.stride(3),
      chunk_lse.stride(0),
      chunk_lse.stride(1),
      chunk_lse.stride(2),
      chunk_lse.stride(3),
      bias_strides[0],
      bias_strides[1],
      bias_strides[2],
      bias_strides[3],
      nheads_q,
      nheads_kv,
      seqlen_q,
      seqlen_k,
      seqlen_q_bucket,
      seqlen_k_bucket,
      dropout_p,
      philox_offset,
      IS_CAUSAL=causal,
      HAS_ATTN_BIAS=has_attn_bias,
      HAS_DROPOUT=has_dropout,
      PHILOX_SEED=philox_seed,
      USE_GEMV=use_gemv,
      DTYPE=DTYPE,
      HEADDIM=headdim,
      BLOCK_M=block_m,
      CHUNK_SIZE=chunk_size,
      BLOCK_N=128,
      BLOCK_HEADDIM_QK=block_headdim,
      BLOCK_HEADDIM_V=block_headdim,
      num_warps=8,
      num_stages=2,
    )

  stage2_block_headdim_v = triton.next_power_of_2(headdim) if headdim <= 512 else 128
  block_chunks = triton.next_power_of_2(n_chunks)

  def stage2_grid(meta: dict) -> tuple[int, int]:
    num_v_groups = triton.cdiv(o.size(-1), meta["BLOCK_HEADDIM_V"])
    return (o.size(0) * o.size(1) * o.size(2), num_v_groups)

  _ffpa_decode_fwd_stage2_kernel[stage2_grid](
    partial_out,
    chunk_lse,
    o,
    lse,
    partial_out.stride(0),
    partial_out.stride(1),
    partial_out.stride(2),
    partial_out.stride(3),
    chunk_lse.stride(0),
    chunk_lse.stride(1),
    chunk_lse.stride(2),
    chunk_lse.stride(3),
    o.stride(0),
    o.stride(1),
    o.stride(2),
    lse.stride(0),
    lse.stride(1),
    lse.stride(2),
    o.size(1),
    o.size(2),
    n_chunks,
    DTYPE=DTYPE,
    HEADDIM=o.size(-1),
    BLOCK_HEADDIM_V=stage2_block_headdim_v,
    BLOCK_CHUNKS=block_chunks,
    num_warps=4,
  )


_ffpa_fwd = _ffpa_fwd_kernel_impl
_ffpa_fwd_autotune_cache: dict[tuple[int, str], callable] = {}  # (headdim, mode) -> callable


def _get_fwd_autotune(headdim: int, autotune_mode: str):
  """Return a headdim-specific autotune wrapper for the forward kernel.

  Results are cached by headdim so the autotune overhead is paid at most once
  per shape on a given process.  The search space is generated by
  :func:`_gen_fwd_autotune_configs` which gates a full-D chunk config on
  device SMEM capacity.

  :param headdim: The actual head dimension for the target forward call.
  :return: A ``triton.autotune``-wrapped version of ``_ffpa_fwd_kernel_impl``
      tuned for ``headdim``.
  """
  cache_key = (headdim, autotune_mode)
  if cache_key not in _ffpa_fwd_autotune_cache:
    configs = _gen_fwd_autotune_configs(headdim, autotune_mode=autotune_mode)
    _ffpa_fwd_autotune_cache[cache_key] = triton.autotune(
      configs=configs,
      key=["seqlen_q_bucket", "seqlen_k_bucket", "HEADDIM"],
      cache_results=True,
    )(_ffpa_fwd_kernel_impl)
  return _ffpa_fwd_autotune_cache[cache_key]


def _ffpa_attn_forward_impl(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  o: torch.Tensor,
  lse: torch.Tensor,
  attn_bias: torch.Tensor | None = None,
  causal: bool = False,
  softmax_scale: float | None = None,
  autotune: bool = False,
  autotune_mode: str = "fast",
  dropout_p: float = 0.0,
  philox_seed: int = 0,
  philox_offset: int = 0,
) -> None:
  """Run the Triton FFPA Split-D forward kernel.

  This is the low-level implementation entry used by the registered torch op.
  It validates tensor layout/dtypes, chooses between the generic and split-kv
  decode paths, and forwards the saved dropout RNG state to the selected kernel.

  :param q: Query tensor in ``[B, Hq, Nq, D]`` layout.
  :param k: Key tensor in ``[B, Hkv, Nkv, D]`` layout.
  :param v: Value tensor in ``[B, Hkv, Nkv, D]`` layout.
  :param o: Output tensor in ``[B, Hq, Nq, D]`` layout, written in place.
  :param lse: Float32 LSE tensor with shape ``[B, Hq, Nq_aligned]`` or a
      view whose last dimension is the visible query length.
  :param attn_bias: Optional additive mask broadcastable to
      ``[B, Hq, Nq, Nkv]``.
  :param causal: Whether to apply lower-right causal masking.
  :param softmax_scale: Scale applied to ``Q @ K.T``. Defaults to
      ``1 / sqrt(D)``.
  :param autotune: Whether to run Triton's autotuner for this shape.
  :param autotune_mode: Triton autotune search-space mode, ``"fast"`` or
      ``"max"``.
  :param dropout_p: Forward dropout probability.
  :param philox_seed: Philox seed used for dropout. Ignored when
      ``dropout_p == 0``.
  :param philox_offset: Philox element offset used for SDPA-compatible dropout
      RNG layout.
  """
  batch, nheads_q, seqlen_q, headdim = q.shape
  _, _, seqlen_k, _ = k.shape
  softmax_scale = softmax_scale or (1.0 / math.sqrt(headdim))

  assert q.dtype == k.dtype == v.dtype == o.dtype
  assert q.dtype in (torch.float16, torch.bfloat16)
  assert lse.dtype == torch.float32
  assert q.stride(-1) == k.stride(-1) == v.stride(-1) == o.stride(-1) == 1
  if headdim > _MAX_HEADDIM:
    raise ValueError(f"Triton forward supports headdim <= {_MAX_HEADDIM}, got {headdim}")

  num_splits = _get_decode_num_splits(seqlen_q, seqlen_k, headdim, batch, nheads_q, q.device)

  if num_splits == 1:
    _ffpa_attn_forward_generic_impl(
      q,
      k,
      v,
      o,
      lse,
      attn_bias=attn_bias,
      causal=causal,
      softmax_scale=softmax_scale,
      autotune=autotune,
      autotune_mode=autotune_mode,
      dropout_p=dropout_p,
      philox_seed=philox_seed,
      philox_offset=philox_offset,
    )
    return

  _ffpa_attn_forward_decode_impl(
    q,
    k,
    v,
    o,
    lse,
    attn_bias=attn_bias,
    causal=causal,
    softmax_scale=softmax_scale,
    autotune=autotune,
    autotune_mode=autotune_mode,
    num_splits=num_splits,
    dropout_p=dropout_p,
    philox_seed=philox_seed,
    philox_offset=philox_offset,
  )


def _ffpa_attn_forward_triton(
  Q: torch.Tensor,
  K: torch.Tensor,
  V: torch.Tensor,
  O: torch.Tensor | None = None,
  causal: bool = False,
  softmax_scale: float = 0.0,
  autotune: bool = False,
  autotune_mode: str = "fast",
  attn_bias: torch.Tensor | None = None,
  dropout_p: float = 0.0,
  philox_seed: int = 0,
  philox_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Call the Triton FFPA forward via registered torch op, returning ``(O, softmax_lse)``.

  The ``O`` parameter is accepted for API compatibility but ignored — the
  registered op always allocates a fresh output buffer.

  :param Q: Query tensor in ``[B, Hq, Nq, D]`` layout.
  :param K: Key tensor in ``[B, Hkv, Nkv, D]`` layout.
  :param V: Value tensor in ``[B, Hkv, Nkv, D]`` layout.
  :param O: Ignored compatibility parameter.
  :param causal: Whether to apply lower-right causal masking.
  :param softmax_scale: Scale applied to ``Q @ K.T``. ``0.0`` means the op will
    use its default scale.
  :param autotune: Whether to use Triton's autotuner for the selected path.
  :param autotune_mode: Triton autotune search-space mode, ``"fast"`` or
    ``"max"``.
  :param attn_bias: Optional additive mask broadcastable to
    ``[B, Hq, Nq, Nkv]``.
  :param dropout_p: Forward dropout probability.
  :param philox_seed: Philox seed used for dropout. Ignored when
    ``dropout_p == 0``.
  :param philox_offset: Philox element offset used for SDPA-compatible dropout
    RNG layout.
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
    attn_bias,
    softmax_scale,
    int(causal),
    int(autotune),
    int(autotune_mode == "max"),
    dropout_p,
    philox_seed,
    philox_offset,
  )
  return O_storage, softmax_lse_storage[..., :seqlen_q]

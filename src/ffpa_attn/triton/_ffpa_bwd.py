"""
FFPA Attention Backward (Split-D) — Triton Implementation. Adapted from:
- https://triton-lang.org/main/_downloads/54a35f6ec55f9746935b9566fb6bb1df/06-fused-attention.py

Triton >= 3.x compatible (uses ``tl.trans`` instead of ``trans_b``).
Supports headdim up to 1024 via Split-D tiling.

The backward implementation has two execution paths:

* Main path for Nq >= 8: a shared-pid split-D kernel computes dK/dV and dQ
  in the same launch. The pid is reused as either a K-block id or a Q-block id,
  which keeps dQ non-atomic because each pid owns one Q tile.
* Decode path for Nq < 8: stage1 computes dK/dV and per-K-block partial dQ;
  a second kernel reduces partial dQ across K blocks. Nq == 1 uses a GEMV-style
  specialization to avoid forming tiny matrix tiles.

Additive ``attn_bias`` follows SDPA's logical score shape
``[B, Hq, Nq, Nkv]``. Compact masks are represented by stride-0 dimensions in
the Triton kernels, and their gradients are reduced back to the compact user
shape. Dropout is replayed from the forward Philox seed/offset using the same
logical score element order as PyTorch SDPA. GQA/MQA is handled outside these
kernels by expanding K/V to query-head layout and reducing dK/dV back after the
kernel returns.

Main path (Nq >= 8):
  The single shared-pid kernel has two independent roles. When pid maps to a
  K tile, it streams all relevant Q tiles, reconstructs scores and dP across
  head-dim chunks, applies causal/additive-bias/dropout state, derives dS,
  and accumulates one DK/DV tile. When the same pid maps to a Q tile, it
  streams K tiles, reconstructs the same local dS, and accumulates DQ for the
  owned Q rows. DQ is a plain store because Q tiles are uniquely owned; DK/DV
  are also written by one K-tile owner after local accumulation over Q tiles.

  Performance note:
    The current main-path implementation does not keep one full DK/DV/DQ tile
    in registers until completion. Instead, it repeatedly performs global
    ``load old grad -> add current tile contribution -> store updated grad``
    on ``DQ`` / ``DK`` / ``DV`` as it walks K or Q tiles. This has two direct
    consequences:

    1. Memory bandwidth pressure: each partial update pays an extra global
       read and global write, so backward traffic scales with the number of
       contributing tiles rather than one final write per output tile.
    2. Accumulation dtype follows output storage: these global round-trips use
       the actual storage dtype of ``DQ`` / ``DK`` / ``DV``. If the wrapper
       allocates fp32 buffers, cross-tile accumulation stays fp32. If it
       allocates bf16/fp16 buffers, every partial update is rounded at store
       time and reloaded at that lower precision on the next iteration.

    Because of this, the repeated DQ/DK/DV load/store pattern is both a major
    performance bottleneck and the reason output-buffer dtype materially
    affects backward accuracy. A future rewrite should prefer register or
    local-scratch fp32 accumulation with one final cast/store per output tile.

Decode path (Nq < 8):
  The stage1 kernel is split by K tile because a tiny query window would
  underutilize the main matrix-tile path. Each stage1 program computes DK/DV
  for its K tile and writes a PartialDQ contribution. A second reduce kernel
  sums PartialDQ across K tiles. Nq == 1 uses a GEMV-style specialization;
  Nq in [2, 7] uses a small matrix tile. Causal masking is tail-aligned with
  SDPA, so query row m can attend to key positions <= m + (Nkv - Nq).

``delta = rowsum(dO * O)`` is precomputed once and reused by both paths.
"""

import math

import torch
import triton
import triton.language as tl

from ._autotune_utils import autotune_seqlen_key
from ._persistent_autotune import PersistentConfigRequest, dtype_name, lookup_persistent_config


def _attn_bias_broadcast_strides(
  attn_bias: torch.Tensor | None,
  batch: int,
  nheads: int,
  seqlen_q: int,
  seqlen_k: int,
) -> tuple[int, int, int, int]:
  """Return strides for broadcasting a compact 4-D attention bias.

  The kernels always index ``AttnBias`` as if it had logical shape
  ``[B, Hq, Nq, Nkv]``. A stride of zero means the corresponding user dimension
  was size 1 and should be reused for every logical score element. This avoids
  materializing common masks such as ``[1, 1, 1, Nkv]``.
  """
  if attn_bias is None:
    return (0, 0, 0, 0)
  return (
    0 if attn_bias.size(0) == 1 and batch > 1 else attn_bias.stride(0),
    0 if attn_bias.size(1) == 1 and nheads > 1 else attn_bias.stride(1),
    0 if attn_bias.size(2) == 1 and seqlen_q > 1 else attn_bias.stride(2),
    0 if attn_bias.size(3) == 1 and seqlen_k > 1 else attn_bias.stride(3),
  )


def _attn_bias_grad_needs_reduction(
  grad_attn_bias: torch.Tensor | None,
  batch: int,
  nheads: int,
  seqlen_q: int,
  seqlen_k: int,
) -> bool:
  """Return whether broadcasted score gradients must be accumulated.

  SDPA accepts broadcastable ``attn_mask`` shapes such as ``[Nq, Nkv]``,
  ``[1, 1, Nq, Nkv]``, and ``[B, 1, Nq, Nkv]`` for logical scores with
  shape ``[B, Hq, Nq, Nkv]``. When one of those compact dimensions broadcasts
  to more than one score dimension, multiple ``dBias`` elements map back to
  the same user mask element. The Triton kernel must therefore accumulate with
  ``tl.atomic_add`` instead of using a plain ``tl.store``. A full
  ``[B, Hq, Nq, Nkv]`` mask does not need this reduction and uses ``store``.
  """
  if grad_attn_bias is None:
    return False
  return any([
    grad_attn_bias.size(0) == 1 and batch > 1,
    grad_attn_bias.size(1) == 1 and nheads > 1,
    grad_attn_bias.size(2) == 1 and seqlen_q > 1,
    grad_attn_bias.size(3) == 1 and seqlen_k > 1,
  ])


def _attn_bias_grad_reduces_query(
  grad_attn_bias: torch.Tensor | None,
  seqlen_q: int,
) -> bool:
  """Return whether compact bias gradients reduce the query dimension.

  ``[1, 1, 1, Nkv]`` key-position masks need a sum over all query rows. The
  main kernel either atomically accumulates this reduction directly or writes
  per-Q-block partials for the dedicated fp32 reducer below.
  """
  return grad_attn_bias is not None and grad_attn_bias.size(2) == 1 and seqlen_q > 1


def _attn_bias_grad_is_key_bias(
  grad_attn_bias: torch.Tensor | None,
  seqlen_q: int,
  seqlen_k: int,
) -> bool:
  """Return whether grad is for a compact key-position bias [1, 1, 1, Nkv].

  The special path is limited to the main kernel path (``seqlen_q >= 8``). It
  keeps the high-volume ``sum over B*Hq*Nq`` in fp32 partial buffers and then
  reduces once, which is more accurate and avoids many atomics for long Nq.
  """
  if grad_attn_bias is None or seqlen_q < 8:
    return False
  return all([
    grad_attn_bias.size(0) == 1,
    grad_attn_bias.size(1) == 1,
    grad_attn_bias.size(2) == 1,
    grad_attn_bias.size(3) == seqlen_k,
  ])


@triton.jit
def _curand_uniform_from_element_offset(seed: tl.constexpr, element_offset):
  # Must match forward and PyTorch mem-efficient attention exactly: one Philox
  # output per logical [B, H, Nq, Nkv] score element, converted through uint32
  # like curand_uniform4.  Signed conversion can differ by one ulp.
  quad_offset = element_offset // 4
  lane = element_offset - quad_offset * 4
  r0, r1, r2, r3 = tl.randint4x(seed, quad_offset)
  r = tl.where(lane == 0, r0, tl.where(lane == 1, r1, tl.where(lane == 2, r2, r3)))
  r_u32 = r.to(tl.uint32, bitcast=True)
  return (r_u32.to(tl.float32) + 1.0) * 2.3283064365386963e-10


@triton.jit
def _dropout_multiplier(
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
  # Forward stores one dropout decision for every logical score element in
  # row-major [B, H, Nq, Nkv] order. Backward must regenerate the same decisions
  # before both dP and P are used: dP sees the mask through dO @ V^T, while dV
  # sees the dropped probability through P_drop.
  mult = tl.full([offs_m.shape[0], offs_n.shape[0]], 1.0, dtype=tl.float32)
  if HAS_DROPOUT:
    # Replay the exact forward dropout mask using SDPA's logical score layout.
    linear = off_hb * seqlen_q * seqlen_k + offs_m[:, None] * seqlen_k + offs_n[None, :]
    rand = _curand_uniform_from_element_offset(philox_seed, philox_offset + linear)
    keep = rand > dropout_p
    mult = keep * (1.0 / (1.0 - dropout_p))
  return mult


# Preprocess: delta = rowsum(dO * O)
# In full-D mode BLOCK_HEADDIM must cover the whole head dimension.  In
# D_CHUNK mode the launcher/autotuner supplies the chunk size explicitly.
_FFPA_BWD_PRE_HEURISTICS = {
  "BLOCK_HEADDIM":
  lambda args: args["BLOCK_HEADDIM"] if args["D_CHUNK"] else max(
    64,
    triton.next_power_of_2(args["headdim"]),
  ),
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
  autotune_seqlen_q_bucket: int,
  seqlen_q_rounded: int,
  headdim: int,
  BLOCK_M: tl.constexpr,
  BLOCK_HEADDIM: tl.constexpr,
  D_CHUNK: tl.constexpr,
) -> None:
  """Preprocess kernel to compute delta = rowsum(dO * O) for the backward pass."""
  _ = autotune_seqlen_q_bucket
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


def _gen_pre_autotune_configs(d_chunk: bool, autotune_mode: str = "max") -> list[triton.Config]:
  """Generate autotune configs for the preprocess delta kernel.

  ``BLOCK_HEADDIM`` participates in autotune only for D_CHUNK mode.  Full-D
  mode keeps the historical runtime heuristic so invalid narrow configs are
  never benchmarked for large head dimensions.

  :param d_chunk: Whether generated configs should enable D_CHUNK mode.
  :param autotune_mode: Accepted for API symmetry; backward preprocess uses the
      same bounded search space for fast and max because only bwd_main gets an
      expanded max search.
  :return: Triton autotune configurations for the delta preprocess kernel.
  """
  del autotune_mode
  configs = []
  for block_m in [64, 128, 256]:
    if not d_chunk:
      for num_warps in [4, 8]:
        configs.append(triton.Config(
          {
            "BLOCK_M": block_m,
            "D_CHUNK": False
          },
          num_warps=num_warps,
        ))
      continue

    for block_headdim in [64, 128]:
      for num_warps in [4, 8]:
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


_ffpa_bwd_pre_autotune_cache: dict[tuple[bool, str, str], callable] = {}


def _get_pre_autotune(d_chunk: bool, autotune_mode: str, dtype: str):
  cache_key = (d_chunk, autotune_mode, dtype)
  if cache_key not in _ffpa_bwd_pre_autotune_cache:
    _ffpa_bwd_pre_autotune_cache[cache_key] = triton.autotune(
      configs=_gen_pre_autotune_configs(d_chunk=d_chunk, autotune_mode=autotune_mode),
      key=["autotune_seqlen_q_bucket", "headdim"],
      reset_to_zero=["Delta"],
      cache_results=True,
    )(_ffpa_bwd_pre_impl)
  return _ffpa_bwd_pre_autotune_cache[cache_key]


# Non-autotuned variant.
_ffpa_bwd_pre = _ffpa_bwd_pre_impl


def _normalize_bwd_pre_config(
  config: dict[str, object] | None,
  *,
  preprocess_d_chunk: bool,
  block_headdim_delta: int,
) -> dict[str, object]:
  """Return a complete launch config for the backward preprocess kernel.

  Older persistent ``bwd_preproc`` entries were generated from full-D autotune
  configs that relied on the Triton heuristic for ``BLOCK_HEADDIM`` and thus
  did not serialize that key.  The non-autotuned persistent launcher passes an
  explicit config dict, so fill the runtime value here before launching.

  :param config: Persisted launch config, or ``None`` when no entry matched.
  :param preprocess_d_chunk: Whether the preprocess path uses D-chunk mode.
  :param block_headdim_delta: Full-D ``BLOCK_HEADDIM`` value for this runtime
      head dimension, or the D-chunk block size when ``preprocess_d_chunk`` is
      enabled.
  :return: Complete kernel launch meta dict.
  """
  normalized = {
    "BLOCK_M": 128,
    "BLOCK_HEADDIM": block_headdim_delta,
    "D_CHUNK": preprocess_d_chunk,
    "num_warps": 4,
  }
  if config is not None:
    normalized.update(config)
  return normalized


def _supports_bwd_main_max_extra_configs() -> bool:
  """Return whether the current GPU should try extra bwd_main max configs."""
  try:
    major, _minor = torch.cuda.get_device_capability()
  except Exception:
    return False
  return major >= 9


def _gen_bwd_autotune_configs(
  block_n_values: tuple[int, ...],
  autotune_mode: str = "max",
) -> list[triton.Config]:
  """Generate autotune configs over BLOCK_M, BLOCK_N, BLOCK_HEADDIM, num_warps, num_stages.

  :param block_n_values: Candidate ``BLOCK_N`` values for the target backward
      kernel variant.
  :param autotune_mode: Search-space mode, ``"fast"`` or ``"max"``.
  :return: Triton autotune configurations for one backward kernel variant.
  """
  # BLOCK_M: larger = fewer Q-block iterations (good), more register pressure.
  # BLOCK_N: controls K/V tile size for dK/dV and dQ recomputation.
  # BLOCK_HEADDIM:
  #   64, 128 — classic D-chunk split, low register pressure, widely compatible.
  #   256     — SM90+ max-only wider split-D candidate.  On SM < 90 measured
  #             performance is effectively identical, so max intentionally uses
  #             the fast search space.  Full-D dynamic candidates and stage-4
  #             pipelines are still pruned because they make offline persistent
  #             tuning much slower without a clear default win.
  # TODO: Optimize the autotune time by saving the best config per shape
  # (device-shape/headdim) in a file and loading it at the start of autotune.
  use_extra_max_configs = autotune_mode == "max" and _supports_bwd_main_max_extra_configs()
  if autotune_mode == "max" and not use_extra_max_configs:
    block_n_values = (64, )
  _headdim_candidates = [64, 128, 256] if use_extra_max_configs else [64, 128]
  _num_stages_candidates = [2, 3]

  configs = []
  for block_m in [64, 128]:
    for block_n in block_n_values:
      for block_headdim in _headdim_candidates:
        for num_warps in [4, 8]:
          for num_stages in _num_stages_candidates:
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


# Shared-pid split-D backward kernel (no dQ atomic_add)
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
# non-atomically.
#
# The kernel sees K/V already expanded to query-head layout for GQA/MQA. That
# keeps the Triton code head-local: off_h always indexes a query head, and the
# wrapper folds expanded dK/dV back to the original KV heads afterwards.
@triton.heuristics(_FFPA_BWD_HEURISTICS)
@triton.jit
def _ffpa_bwd_kernel_impl(
  Q: torch.Tensor,
  K: torch.Tensor,
  V: torch.Tensor,
  DO: torch.Tensor,
  DQ: torch.Tensor,
  DK: torch.Tensor,
  DV: torch.Tensor,
  LSE: torch.Tensor,
  D: torch.Tensor,
  AttnBias: torch.Tensor,
  GradAttnBias: torch.Tensor,
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
  BLOCK_HEADDIM: tl.constexpr,
  DTYPE: tl.constexpr,
  EVEN_M: tl.constexpr,
  EVEN_N: tl.constexpr,
  BLOCK_M: tl.constexpr,
  BLOCK_N: tl.constexpr,
) -> None:
  # Keys for autotuning heuristics and persistent autotune lookup.
  _ = autotune_seqlen_q_bucket
  _ = autotune_seqlen_k_bucket
  _ = autotune_causal_key
  _ = autotune_dtype_key

  pid = tl.program_id(0)
  off_hb = tl.program_id(2)
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

  if HAS_ATTN_BIAS:
    AttnBias += off_b * stride_bb + off_h * stride_bh
  if BIAS_REQUIRES_GRAD:
    GradAttnBias += off_b * stride_gbb + off_h * stride_gbh

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

      # Reconstruct local scores and dP for this K tile by streaming D chunks.
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

      # Convert scores to P/dS under the same masking and dropout state as fwd.
      if not EVEN_N:
        S = tl.where(offs_n[None, :] < seqlen_k, S, float("-inf"))
      if IS_CAUSAL:
        S = tl.where(offs_qm[:, None] >= (offs_n[None, :]), S, float("-inf"))
      S = S * softmax_scale
      if HAS_ATTN_BIAS:
        # AttnBias strides may be zero for broadcast dimensions. The pointer
        # math therefore covers full masks and compact masks with the same load.
        bias = tl.load(
          AttnBias + offs_qm[:, None] * stride_bm + offs_n[None, :] * stride_bn,
          mask=(offs_qm[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k),
          other=0.0,
        )
        S += bias
      lse_i = tl.load(LSE + offs_qm)
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
      Di = tl.load(D + offs_qm)
      if BIAS_REQUIRES_GRAD:
        # dBias is the gradient wrt the additive score bias before multiplying
        # by softmax_scale. The write strategy depends on how the user mask was
        # broadcast to logical [B, Hq, Nq, Nkv] scores:
        #   1. GRAD_BIAS_REDUCES_M: the query dimension was broadcast, e.g.
        #      [1, 1, 1, Nkv]. Sum the BLOCK_M rows first because all rows in
        #      this tile alias the same key-position element. When
        #      GRAD_BIAS_STORE_PARTIAL is true, the target is a fp32 partial
        #      buffer indexed by Q-block, so a plain store is correct. Otherwise
        #      multiple Q blocks alias the final compact mask and need atomics.
        #   2. GRAD_BIAS_NEEDS_REDUCTION: some non-query dimension broadcasts,
        #      such as batch or head. Keep the [M, N] layout but atomic-add each
        #      score because different programs can target the same compact
        #      batch/head mask element.
        #   3. Full mask: every logical score owns a unique output element, so a
        #      normal store is both correct and faster.
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

      # Accumulate this K tile's DK/DV over Q tiles, one D chunk at a time.
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
        dk_ptrs = DK + offs_n[:, None] * stride_dkn + d_offs[None, :]
        dv_ptrs = DV + offs_n[:, None] * stride_dvn + d_offs[None, :]
        grad_mask = (offs_n[:, None] < seqlen_k) & (d_offs[None, :] < headdim)

        if start_m == begin_m:
          dk_d = tl.trans(tl.dot(tl.trans(q), dS, out_dtype=tl.float32))
          tl.store(dk_ptrs, dk_d, mask=grad_mask, eviction_policy="evict_last")
          dv_d = tl.trans(tl.dot(tl.trans(do), P_drop.to(DTYPE), out_dtype=tl.float32))
          tl.store(dv_ptrs, dv_d, mask=grad_mask, eviction_policy="evict_last")
        else:
          # NOTE: These global load/store ops use each gradient buffer's storage
          # dtype, which is chosen by _triton_bwd_grad_tensor_like() in
          # triton/__init__.py. dV can optionally use fp32 internal storage;
          # otherwise each later update round-trips through bf16/fp16 storage.
          dk_val = tl.load(dk_ptrs, mask=grad_mask, other=0., eviction_policy="evict_last")
          dk_d = tl.trans(tl.dot(tl.trans(q), dS, out_dtype=tl.float32))
          tl.store(dk_ptrs, dk_val + dk_d, mask=grad_mask, eviction_policy="evict_last")
          dv_val = tl.load(dv_ptrs, mask=grad_mask, other=0., eviction_policy="evict_last")
          dv_d = tl.trans(tl.dot(tl.trans(do), P_drop.to(DTYPE), out_dtype=tl.float32))
          tl.store(dv_ptrs, dv_val + dv_d, mask=grad_mask, eviction_policy="evict_last")

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

      # Reconstruct local scores and dP for this Q tile by streaming D chunks.
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

      # Convert scores to dS for the owned Q tile. dQ uses this dS only; dBias
      # was already emitted on the K-tile side to avoid duplicate mask writes.
      if not EVEN_N:
        S_qk = tl.where(offs_nk[None, :] < seqlen_k, S_qk, float("-inf"))
      if IS_CAUSAL:
        S_qk = tl.where(offs_m[:, None] >= (offs_nk[None, :]), S_qk, float("-inf"))
      S_qk = S_qk * softmax_scale
      if HAS_ATTN_BIAS:
        # Same logical [B, Hq, Nq, Nkv] bias addressing as the dK/dV side. When
        # a dimension was broadcast by the user, the corresponding stride is 0.
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
      # dQ does not write dBias. It still must replay dropout in dP so the
      # softmax backward matches the forward dropped probability matrix.
      dS_qk = (P_qk * (dP_qk - Di[:, None]) * softmax_scale).to(DTYPE)

      # Accumulate the owned Q tile's DQ over K tiles, one D chunk at a time.
      for d_chunk in range(num_d_chunks):
        d_offs = d_chunk * BLOCK_HEADDIM + offs_d
        k = tl.load(
          K + offs_nk[:, None] * stride_kn + d_offs[None, :],
          mask=(offs_nk[:, None] < seqlen_k) & (d_offs[None, :] < headdim),
          other=0.
        )
        dq_ptrs = DQ + offs_m[:, None] * stride_dqm + d_offs[None, :]
        dq_mask = (offs_m[:, None] < seqlen_q) & (d_offs[None, :] < headdim)
        if start_n_k == 0:
          dq_d = tl.dot(dS_qk, k, out_dtype=tl.float32)
          tl.store(dq_ptrs, dq_d, mask=dq_mask, eviction_policy="evict_last")
        else:
          # Same storage-dtype rule as DK/DV above: later DQ accumulation
          # precision follows the buffer allocated by _triton_bwd_grad_tensor_like().
          dq_val = tl.load(dq_ptrs, mask=dq_mask, other=0., eviction_policy="evict_last")
          dq_d = tl.dot(dS_qk, k, out_dtype=tl.float32)
          # NOTE: dQ is written non-atomically — each program owns a unique Q-row block.
          tl.store(dq_ptrs, dq_val + dq_d, mask=dq_mask, eviction_policy="evict_last")


# Non-autotuned main backward variant.
_ffpa_bwd = _ffpa_bwd_kernel_impl
# (headdim, mode, bias_grad) -> callable
_ffpa_bwd_autotune_cache: dict[tuple[int, str, bool], callable] = {}


def _get_bwd_autotune(headdim: int, autotune_mode: str, bias_requires_grad: bool):
  """Return an autotune wrapper for the main backward kernel.

  :param headdim: Runtime head dimension used by the config generator.
  :param autotune_mode: Search-space mode, ``"fast"`` or ``"max"``.
  :param bias_requires_grad: Whether GradAttnBias may need reset between
      autotune candidate runs.
  :return: Triton autotune wrapper for ``_ffpa_bwd_kernel_impl``.
  """
  cache_key = (headdim, autotune_mode, bias_requires_grad)
  if cache_key not in _ffpa_bwd_autotune_cache:
    configs = _gen_bwd_autotune_configs(
      block_n_values=(64, ) if autotune_mode == "fast" else (64, 128),
      autotune_mode=autotune_mode,
    )
    reset_args = []
    if bias_requires_grad:
      reset_args.append("GradAttnBias")
    _ffpa_bwd_autotune_cache[cache_key] = triton.autotune(
      configs=configs,
      key=[
        "autotune_seqlen_q_bucket",
        "autotune_seqlen_k_bucket",
        "headdim",
        "autotune_causal_key",
        "autotune_dtype_key",
      ],
      reset_to_zero=reset_args,
      cache_results=True,
    )(_ffpa_bwd_kernel_impl)
  return _ffpa_bwd_autotune_cache[cache_key]


def _gen_decode_bwd_stage1_autotune_configs(
  headdim: int = 512,
  use_gemv: bool = False,
  autotune_mode: str = "max",
) -> list[triton.Config]:
  """Generate decode-backward stage1 autotune configs.

  :param headdim: Runtime head dimension.  Kept in the signature for caller
      symmetry; decode backward uses the same bounded search space for fast and
      max because only bwd_main gets an expanded max search.
  :param use_gemv: Whether the target shape is the single-query GEMV path.
  :param autotune_mode: Accepted search-space mode, ``"fast"`` or ``"max"``.
  :return: Triton autotune configs for the decode backward stage1 kernel.
  """
  del headdim, autotune_mode

  headdim_candidates = [64, 128]
  block_n_candidates = [64, 128]
  block_m_candidates = [8] if use_gemv else [16]

  configs = []
  for block_n in block_n_candidates:
    for block_m in block_m_candidates:
      for block_headdim in headdim_candidates:
        for num_stages in [2]:
          configs.append(
            triton.Config(
              {
                "BLOCK_M": block_m,
                "BLOCK_N": block_n,
                "BLOCK_HEADDIM": block_headdim,
              },
              num_warps=8,
              num_stages=num_stages,
            )
          )
  return configs


_ffpa_bwd_decode_stage1_autotune_cache: dict[tuple[int, bool, str, bool], callable] = {}


def _get_decode_bwd_stage1_autotune(
  headdim: int,
  use_gemv: bool,
  autotune_mode: str,
  bias_requires_grad: bool,
):
  """Return an autotune wrapper for the decode backward stage1 kernel."""
  cache_key = (headdim, use_gemv, autotune_mode, bias_requires_grad)
  if cache_key not in _ffpa_bwd_decode_stage1_autotune_cache:
    reset_args = ["DK", "DV", "PartialDQ"]
    if bias_requires_grad:
      # Decode stage1 can also accumulate or alias compact mask gradients
      # across autotune candidate runs, so GradAttnBias must be reset too.
      reset_args.append("GradAttnBias")
    _ffpa_bwd_decode_stage1_autotune_cache[cache_key] = triton.autotune(
      configs=_gen_decode_bwd_stage1_autotune_configs(
        headdim=headdim,
        use_gemv=use_gemv,
        autotune_mode=autotune_mode,
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
    )(_ffpa_bwd_decode_stage1_kernel)
  return _ffpa_bwd_decode_stage1_autotune_cache[cache_key]


@triton.jit
def _ffpa_bwd_key_bias_grad_reduce_kernel(
  PartialGradBias: torch.Tensor,
  GradAttnBias: torch.Tensor,
  seqlen_k: int,
  total_rows: int,
  stride_gbn: int,
  BLOCK_N: tl.constexpr,
  BLOCK_R: tl.constexpr,
) -> None:
  start_n = tl.program_id(0) * BLOCK_N
  offs_n = start_n + tl.arange(0, BLOCK_N)
  offs_r = tl.arange(0, BLOCK_R)
  mask_n = offs_n < seqlen_k
  acc = tl.zeros([BLOCK_R, BLOCK_N], dtype=tl.float32)
  for start_r in range(0, total_rows, BLOCK_R):
    rows = start_r + offs_r
    partial = tl.load(
      PartialGradBias + rows[:, None] * seqlen_k + offs_n[None, :],
      mask=(rows[:, None] < total_rows) & mask_n[None, :],
      other=0.0,
    )
    acc += partial
  grad = tl.sum(acc, axis=0)
  tl.store(GradAttnBias + offs_n * stride_gbn, grad, mask=mask_n)


@triton.jit
def _ffpa_bwd_decode_stage1_kernel(
  Q: torch.Tensor,
  K: torch.Tensor,
  V: torch.Tensor,
  DO: torch.Tensor,
  DK: torch.Tensor,
  DV: torch.Tensor,
  PartialDQ: torch.Tensor,
  WrittenKBlocks: torch.Tensor,
  LSE: torch.Tensor,
  D: torch.Tensor,
  AttnBias: torch.Tensor,
  GradAttnBias: torch.Tensor,
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
  stride_dkb: int,
  stride_dkh: int,
  stride_dkn: int,
  stride_dvb: int,
  stride_dvh: int,
  stride_dvn: int,
  stride_pqb: int,
  stride_pqh: int,
  stride_pqk: int,
  stride_pqm: int,
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
  DTYPE: tl.constexpr,
  USE_GEMV: tl.constexpr,
  BLOCK_M: tl.constexpr,
  BLOCK_N: tl.constexpr,
  BLOCK_HEADDIM: tl.constexpr,
) -> None:
  # Keys for autotuning heuristics and persistent autotune lookup.
  _ = autotune_seqlen_q_bucket
  _ = autotune_seqlen_k_bucket
  _ = autotune_causal_key
  _ = autotune_dtype_key

  # Decode backward splits work by K block. DK/DV are independent per K block,
  # but dQ is a sum over all K blocks, so stage1 writes PartialDQ and the reduce
  # kernel below performs the cross-K accumulation.
  start_n_block = tl.program_id(0)
  off_hb = tl.program_id(1)
  off_b = off_hb // nheads
  off_h = off_hb % nheads

  if start_n_block == 0:
    tl.store(WrittenKBlocks + off_hb, tl.num_programs(0))

  Q += off_b * stride_qb + off_h * stride_qh
  K += off_b * stride_kb + off_h * stride_kh
  V += off_b * stride_vb + off_h * stride_vh
  DO += off_b * stride_dob + off_h * stride_doh
  DK += off_b * stride_dkb + off_h * stride_dkh
  DV += off_b * stride_dvb + off_h * stride_dvh
  PartialDQ += off_b * stride_pqb + off_h * stride_pqh + start_n_block * stride_pqk
  LSE += off_hb * seqlen_q_rounded
  D += off_hb * seqlen_q_rounded
  if HAS_ATTN_BIAS:
    AttnBias += off_b * stride_bb + off_h * stride_bh
  if BIAS_REQUIRES_GRAD:
    GradAttnBias += off_b * stride_gbb + off_h * stride_gbh

  offs_n = start_n_block * BLOCK_N + tl.arange(0, BLOCK_N)
  offs_m = tl.arange(0, BLOCK_M)
  offs_d = tl.arange(0, BLOCK_HEADDIM)
  mask_n = offs_n < seqlen_k
  mask_m = offs_m < seqlen_q
  num_d_chunks = tl.cdiv(headdim, BLOCK_HEADDIM)

  if USE_GEMV:
    scores = tl.zeros([BLOCK_N], dtype=tl.float32)
    dP = tl.zeros([BLOCK_N], dtype=tl.float32)
    for d_chunk in range(num_d_chunks):
      d_offs = d_chunk * BLOCK_HEADDIM + offs_d
      q = tl.load(Q + d_offs, mask=d_offs < headdim, other=0.0).to(tl.float32)
      do = tl.load(DO + d_offs, mask=d_offs < headdim, other=0.0).to(tl.float32)
      k = tl.load(
        K + offs_n[:, None] * stride_kn + d_offs[None, :],
        mask=mask_n[:, None] & (d_offs[None, :] < headdim),
        other=0.0,
      ).to(tl.float32)
      v = tl.load(
        V + offs_n[:, None] * stride_vn + d_offs[None, :],
        mask=mask_n[:, None] & (d_offs[None, :] < headdim),
        other=0.0,
      ).to(tl.float32)
      scores += tl.sum(k * q[None, :], axis=1)
      dP += tl.sum(v * do[None, :], axis=1)

    scores = scores * softmax_scale
    if HAS_ATTN_BIAS:
      # USE_GEMV is only used for Nq == 1, so the bias pointer has no query
      # offset; broadcasted batch/head/key strides were already encoded by the
      # launcher.
      bias = tl.load(AttnBias + offs_n * stride_bn, mask=mask_n, other=0.0)
      scores += bias
    if IS_CAUSAL:
      # With a single tail-aligned query, all real K positions are legal. The
      # mask still guards padded BLOCK_N lanes.
      scores = tl.where(offs_n <= (seqlen_k - 1), scores, -float("inf"))
    scores = tl.where(mask_n, scores, -float("inf"))

    lse_i = tl.load(LSE)
    P = tl.exp(scores - lse_i)
    dropout_mult = tl.full([BLOCK_N], 1.0, dtype=tl.float32)
    if HAS_DROPOUT:
      linear = off_hb * seqlen_q * seqlen_k + offs_n
      rand = _curand_uniform_from_element_offset(PHILOX_SEED, philox_offset + linear)
      keep = rand > dropout_p
      dropout_mult = keep * (1.0 / (1.0 - dropout_p))
    dP = dP * dropout_mult
    P_drop = P * dropout_mult
    delta_i = tl.load(D)
    dBias = P * (dP - delta_i)
    dS = dBias * softmax_scale

    if BIAS_REQUIRES_GRAD:
      grad_bias_ptrs = GradAttnBias + offs_n * stride_gbn
      if GRAD_BIAS_NEEDS_REDUCTION:
        tl.atomic_add(grad_bias_ptrs, dBias, sem="relaxed", mask=mask_n)
      else:
        tl.store(grad_bias_ptrs, dBias, mask=mask_n)

    for d_chunk in range(num_d_chunks):
      d_offs = d_chunk * BLOCK_HEADDIM + offs_d
      q = tl.load(Q + d_offs, mask=d_offs < headdim, other=0.0).to(tl.float32)
      do = tl.load(DO + d_offs, mask=d_offs < headdim, other=0.0).to(tl.float32)
      k = tl.load(
        K + offs_n[:, None] * stride_kn + d_offs[None, :],
        mask=mask_n[:, None] & (d_offs[None, :] < headdim),
        other=0.0,
      ).to(tl.float32)
      dk = dS[:, None] * q[None, :]
      dv = P_drop[:, None] * do[None, :]
      tl.store(
        DK + offs_n[:, None] * stride_dkn + d_offs[None, :],
        dk,
        mask=mask_n[:, None] & (d_offs[None, :] < headdim),
      )
      tl.store(
        DV + offs_n[:, None] * stride_dvn + d_offs[None, :],
        dv,
        mask=mask_n[:, None] & (d_offs[None, :] < headdim),
      )
      partial_dq = tl.sum(dS[:, None] * k, axis=0)
      tl.store(PartialDQ + d_offs, partial_dq, mask=d_offs < headdim)
  else:
    scores = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    dP = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for d_chunk in range(num_d_chunks):
      d_offs = d_chunk * BLOCK_HEADDIM + offs_d
      q = tl.load(
        Q + offs_m[:, None] * stride_qm + d_offs[None, :],
        mask=mask_m[:, None] & (d_offs[None, :] < headdim),
        other=0.0,
      )
      do = tl.load(
        DO + offs_m[:, None] * stride_dom + d_offs[None, :],
        mask=mask_m[:, None] & (d_offs[None, :] < headdim),
        other=0.0,
      )
      k = tl.load(
        K + offs_n[:, None] * stride_kn + d_offs[None, :],
        mask=mask_n[:, None] & (d_offs[None, :] < headdim),
        other=0.0,
      )
      v = tl.load(
        V + offs_n[:, None] * stride_vn + d_offs[None, :],
        mask=mask_n[:, None] & (d_offs[None, :] < headdim),
        other=0.0,
      )
      scores = tl.dot(q, tl.trans(k), acc=scores)
      dP = tl.dot(do, tl.trans(v), acc=dP)

    scores = scores * softmax_scale
    if HAS_ATTN_BIAS:
      bias = tl.load(
        AttnBias + offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn,
        mask=mask_m[:, None] & mask_n[None, :],
        other=0.0,
      )
      scores += bias
    scores = tl.where(mask_n[None, :], scores, -float("inf"))
    if IS_CAUSAL:
      kv_offset = seqlen_k - seqlen_q
      # Tail-aligned lower-right causal mask for decode windows where Nq can be
      # smaller than Nkv. A query row m sees keys up to m + (Nkv - Nq).
      causal_mask = offs_n[None, :] <= (offs_m[:, None] + kv_offset)
      scores = tl.where(causal_mask, scores, -float("inf"))
    scores = tl.where(mask_m[:, None], scores, -float("inf"))

    lse_i = tl.load(LSE + offs_m, mask=mask_m, other=-float("inf"))
    P = tl.exp(scores - lse_i[:, None])
    score_mask = mask_m[:, None] & mask_n[None, :]
    P = tl.where(score_mask, P, 0.0)
    dropout_mult = _dropout_multiplier(
      off_hb,
      offs_m,
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
    delta_i = tl.load(D + offs_m, mask=mask_m, other=0.0)
    dBias = P * (dP - delta_i[:, None])
    dBias = tl.where(score_mask, dBias, 0.0)
    dS = (dBias * softmax_scale).to(DTYPE)

    if BIAS_REQUIRES_GRAD:
      grad_bias_mask = mask_m[:, None] & mask_n[None, :]
      # Decode can also receive compact masks. Query-broadcast masks reduce the
      # M dimension inside the tile first; other broadcast dimensions use
      # atomic adds to merge aliases across programs.
      if GRAD_BIAS_REDUCES_M:
        grad_bias_ptrs = GradAttnBias + offs_n * stride_gbn
        grad_bias = tl.sum(tl.where(grad_bias_mask, dBias, 0.0), axis=0)
        tl.atomic_add(grad_bias_ptrs, grad_bias, sem="relaxed", mask=mask_n)
      elif GRAD_BIAS_NEEDS_REDUCTION:
        grad_bias_ptrs = GradAttnBias + offs_m[:, None] * stride_gbm + offs_n[None, :] * stride_gbn
        tl.atomic_add(grad_bias_ptrs, dBias, sem="relaxed", mask=grad_bias_mask)
      else:
        grad_bias_ptrs = GradAttnBias + offs_m[:, None] * stride_gbm + offs_n[None, :] * stride_gbn
        tl.store(grad_bias_ptrs, dBias, mask=grad_bias_mask)

    for d_chunk in range(num_d_chunks):
      d_offs = d_chunk * BLOCK_HEADDIM + offs_d
      q = tl.load(
        Q + offs_m[:, None] * stride_qm + d_offs[None, :],
        mask=mask_m[:, None] & (d_offs[None, :] < headdim),
        other=0.0,
      )
      do = tl.load(
        DO + offs_m[:, None] * stride_dom + d_offs[None, :],
        mask=mask_m[:, None] & (d_offs[None, :] < headdim),
        other=0.0,
      )
      k = tl.load(
        K + offs_n[:, None] * stride_kn + d_offs[None, :],
        mask=mask_n[:, None] & (d_offs[None, :] < headdim),
        other=0.0,
      )
      dk = tl.dot(tl.trans(dS), q, out_dtype=tl.float32)
      dv = tl.dot(tl.trans(P_drop.to(DTYPE)), do, out_dtype=tl.float32)
      partial_dq = tl.dot(dS, k, out_dtype=tl.float32)
      tl.store(
        DK + offs_n[:, None] * stride_dkn + d_offs[None, :],
        dk,
        mask=mask_n[:, None] & (d_offs[None, :] < headdim),
      )
      tl.store(
        DV + offs_n[:, None] * stride_dvn + d_offs[None, :],
        dv,
        mask=mask_n[:, None] & (d_offs[None, :] < headdim),
      )
      tl.store(
        PartialDQ + offs_m[:, None] * stride_pqm + d_offs[None, :],
        partial_dq,
        mask=mask_m[:, None] & (d_offs[None, :] < headdim),
      )


@triton.jit
def _ffpa_bwd_decode_dq_reduce_kernel(
  PartialDQ: torch.Tensor,
  WrittenKBlocks: torch.Tensor,
  DQ: torch.Tensor,
  stride_pqb: int,
  stride_pqh: int,
  stride_pqk: int,
  stride_pqm: int,
  stride_dqb: int,
  stride_dqh: int,
  stride_dqm: int,
  nheads: int,
  seqlen_q: int,
  headdim: int,
  BLOCK_K: tl.constexpr,
  BLOCK_M: tl.constexpr,
  BLOCK_HEADDIM: tl.constexpr,
) -> None:
  d_block = tl.program_id(0)
  q_block = tl.program_id(1)
  off_hb = tl.program_id(2)
  off_b = off_hb // nheads
  off_h = off_hb % nheads

  offs_k = tl.arange(0, BLOCK_K)
  offs_m = q_block * BLOCK_M + tl.arange(0, BLOCK_M)
  offs_d = d_block * BLOCK_HEADDIM + tl.arange(0, BLOCK_HEADDIM)
  mask_m = offs_m < seqlen_q
  acc = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
  written_k_blocks = tl.load(WrittenKBlocks + off_hb)
  PartialDQ += off_b * stride_pqb + off_h * stride_pqh
  for start_k in range(0, written_k_blocks, BLOCK_K):
    k_blocks = start_k + offs_k
    partial = tl.load(
      PartialDQ + k_blocks[:, None, None] * stride_pqk + offs_m[None, :, None] * stride_pqm + offs_d[None, None, :],
      mask=(k_blocks[:, None, None] < written_k_blocks) & mask_m[None, :, None] & (offs_d[None, None, :] < headdim),
      other=0.0,
    )
    acc += tl.sum(partial, axis=0)
  DQ += off_b * stride_dqb + off_h * stride_dqh
  tl.store(
    DQ + offs_m[:, None] * stride_dqm + offs_d[None, :],
    acc,
    mask=mask_m[:, None] & (offs_d[None, :] < headdim),
  )


def _ffpa_attn_backward_triton_impl(
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
  enable_tma: bool = False,
  enable_ws: bool = False,
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
    main backward kernel.
  :param preprocess_d_chunk: Whether the delta preprocess kernel should split
    the head dimension into ``BLOCK_HEADDIM`` chunks instead of processing the
    full head dimension in one program.
  :param original_nheads_kv: Original KV-head count before GQA/MQA expansion.
  """
  if do.stride(-1) != 1:
    do = do.contiguous()
  batch, nheads, seqlen_q, headdim = q.shape
  _, _, seqlen_k, _ = k.shape
  original_nheads_kv = original_nheads_kv or nheads
  softmax_scale = softmax_scale or (1.0 / math.sqrt(headdim))

  if enable_tma and seqlen_q >= 8:
    from ._ffpa_bwd_sm90 import _ffpa_attn_backward_sm90_impl as _bwd_sm90_impl
    from ._ffpa_bwd_sm90 import is_sm90_tma_backward_supported as _sm90_supported

    if _sm90_supported(q, k, v, do, dq, dk, dv, seqlen_q=seqlen_q):
      _bwd_sm90_impl(
        do,
        q,
        k,
        v,
        o,
        lse,
        attn_bias,
        dq,
        dk,
        dv,
        grad_attn_bias,
        causal=causal,
        softmax_scale=softmax_scale,
        autotune=autotune,
        autotune_mode=autotune_mode,
        preprocess_d_chunk=preprocess_d_chunk,
        original_nheads_kv=original_nheads_kv,
        dropout_p=dropout_p,
        philox_seed=philox_seed,
        philox_offset=philox_offset,
        enable_ws=enable_ws,
      )
      return

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
  # The [1, 1, 1, Nkv] key-position mask is common in examples and avoids
  # materializing [B, Hq, Nq, Nkv]. Route its gradient through the same fp32
  # partial-buffer path in both autotune and non-autotune modes so compact
  # key-bias dMask keeps one reduction semantic instead of switching to
  # score-level atomics only because autotune is enabled.
  use_key_bias_grad_reduction = _attn_bias_grad_is_key_bias(grad_attn_bias, seqlen_q, seqlen_k)
  main_bias_requires_grad = bias_requires_grad
  grad_bias_store_partial = use_key_bias_grad_reduction
  has_dropout = dropout_p > 0.0
  runtime_dtype = dtype_name(q.dtype)
  grad_v_storage_dtype = "fp32" if dv.dtype == torch.float32 else None
  persisted_main_config = None
  if not autotune and seqlen_q >= 8:
    persisted_main_config = lookup_persistent_config(
      PersistentConfigRequest(
        direction="backward",
        kernel="bwd_generic",
        autotune_mode=autotune_mode,
        dtype=runtime_dtype,
        headdim=headdim,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        causal=causal,
        bias_grad=main_bias_requires_grad,
        grad_v_storage_dtype=grad_v_storage_dtype,
        has_attn_bias=has_attn_bias,
        has_dropout=has_dropout,
        nheads_q=nheads,
        nheads_kv=original_nheads_kv,
        device_index=q.device.index,
      )
    )
  partial_grad_bias = None
  if use_key_bias_grad_reduction:
    min_block_m_for_bias = 64 if autotune else int((persisted_main_config or {"BLOCK_M": 128})["BLOCK_M"])
    num_m_blocks_for_bias = triton.cdiv(seqlen_q, min_block_m_for_bias)
    # Main-path autotune can switch between BLOCK_M=64 and BLOCK_M=128. Size the
    # partial key-bias buffer for the smallest candidate so BLOCK_M=64 has room
    # for every Q block. Persistent mode must use the selected BLOCK_M; otherwise
    # a persisted BLOCK_M=64 kernel would write more rows than a fallback-sized
    # BLOCK_M=128 buffer provides. Zero-init so larger BLOCK_M configs safely
    # leave the extra rows unused.
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
    grad_attn_bias_in = grad_attn_bias if grad_attn_bias is not None else q
    grad_bias_strides = _attn_bias_broadcast_strides(
      grad_attn_bias,
      batch,
      nheads,
      seqlen_q,
      seqlen_k,
    )

  assert q.dtype == k.dtype == v.dtype == o.dtype == do.dtype
  assert q.dtype in (torch.float16, torch.bfloat16)

  if q.dtype == torch.float16:
    DTYPE = tl.float16
  else:
    DTYPE = tl.bfloat16
  autotune_dtype_key = 1 if q.dtype == torch.bfloat16 else 0

  BLOCK_HEADDIM_DELTA = max(64, triton.next_power_of_2(headdim))
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
    pre_kernel = _get_pre_autotune(preprocess_d_chunk, autotune_mode, runtime_dtype)
    pre_kernel[pre_grid](*pre_args)
  else:
    block_headdim_delta = 64 if preprocess_d_chunk else BLOCK_HEADDIM_DELTA
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

  # Very short query lengths are decode-like: many K tiles contribute to one or
  # a few query rows. The dedicated path keeps DK/DV per K block and reduces dQ
  # explicitly, which is faster than launching the shared-pid matrix kernel for
  # tiny Nq. The causal mask in this path is tail-aligned to SDPA semantics.
  if seqlen_q < 8:
    dq.zero_()
    dk.zero_()
    dv.zero_()
    use_gemv = seqlen_q == 1
    block_m_decode = 8 if use_gemv else 16
    block_n_decode = 64 if use_gemv else 128
    block_headdim_decode = 64
    reduce_block_headdim_decode = block_headdim_decode
    decode_persisted_config = None
    if not autotune:
      decode_persisted_config = lookup_persistent_config(
        PersistentConfigRequest(
          direction="backward",
          kernel="decode_bwd_stage1",
          autotune_mode=autotune_mode,
          dtype=runtime_dtype,
          headdim=headdim,
          seqlen_q=seqlen_q,
          seqlen_k=seqlen_k,
          causal=causal,
          bias_grad=main_bias_requires_grad,
          grad_v_storage_dtype=grad_v_storage_dtype,
          use_gemv=use_gemv,
          has_attn_bias=has_attn_bias,
          has_dropout=has_dropout,
          nheads_q=nheads,
          nheads_kv=original_nheads_kv,
          device_index=q.device.index,
        )
      )
      if decode_persisted_config is not None:
        block_m_decode = int(decode_persisted_config["BLOCK_M"])
        block_n_decode = int(decode_persisted_config["BLOCK_N"])
        block_headdim_decode = int(decode_persisted_config["BLOCK_HEADDIM"])
    min_block_n_decode = 64 if autotune else block_n_decode
    num_k_blocks = triton.cdiv(seqlen_k, min_block_n_decode)
    # Allocate for the smallest candidate BLOCK_N so every autotune config has
    # enough slots. Stage1 records the actual launched K-block count per head,
    # and the reducer below sums only that written prefix, so no full-buffer
    # zero-init is needed here.
    partial_dq = torch.empty(
      (batch, nheads, num_k_blocks, block_m_decode, headdim),
      dtype=torch.float32,
      device=q.device,
    )
    written_k_blocks = torch.empty((batch * nheads, ), dtype=torch.int32, device=q.device)

    def decode_grid(meta: dict) -> tuple[int, int]:
      return (triton.cdiv(seqlen_k, meta["BLOCK_N"]), batch * nheads)

    decode_stage1_args = (
      q,
      k,
      v,
      do,
      dk,
      dv,
      partial_dq,
      written_k_blocks,
      lse,
      delta,
      attn_bias_in,
      grad_attn_bias_in,
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
      dk.stride(0),
      dk.stride(1),
      dk.stride(2),
      dv.stride(0),
      dv.stride(1),
      dv.stride(2),
      partial_dq.stride(0),
      partial_dq.stride(1),
      partial_dq.stride(2),
      partial_dq.stride(3),
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
    decode_stage1_meta = dict(
      IS_CAUSAL=causal,
      HAS_ATTN_BIAS=has_attn_bias,
      HAS_DROPOUT=has_dropout,
      PHILOX_SEED=philox_seed,
      BIAS_REQUIRES_GRAD=main_bias_requires_grad,
      GRAD_BIAS_NEEDS_REDUCTION=grad_bias_needs_reduction,
      GRAD_BIAS_REDUCES_M=grad_bias_reduces_m,
      DTYPE=DTYPE,
      USE_GEMV=use_gemv,
    )
    if autotune:
      _get_decode_bwd_stage1_autotune(
        headdim,
        use_gemv,
        autotune_mode,
        main_bias_requires_grad,
      )[decode_grid](
        *decode_stage1_args,
        **decode_stage1_meta,
      )
    else:
      decode_launch_config = decode_persisted_config or {
        "BLOCK_M": block_m_decode,
        "BLOCK_N": block_n_decode,
        "BLOCK_HEADDIM": block_headdim_decode,
        "num_warps": 8,
        "num_stages": 2,
      }
      _ffpa_bwd_decode_stage1_kernel[decode_grid](
        *decode_stage1_args,
        **decode_stage1_meta,
        **decode_launch_config,
      )
    _ffpa_bwd_decode_dq_reduce_kernel[(
      triton.cdiv(headdim, reduce_block_headdim_decode),
      triton.cdiv(seqlen_q, block_m_decode),
      batch * nheads,
    )](
      partial_dq,
      written_k_blocks,
      dq,
      partial_dq.stride(0),
      partial_dq.stride(1),
      partial_dq.stride(2),
      partial_dq.stride(3),
      dq.stride(0),
      dq.stride(1),
      dq.stride(2),
      nheads,
      seqlen_q,
      headdim,
      BLOCK_K=64,
      BLOCK_M=block_m_decode,
      BLOCK_HEADDIM=reduce_block_headdim_decode,
      num_warps=8,
      num_stages=2,
    )
    return

  def grid(meta: dict) -> tuple[int, ...]:
    return (
      max(triton.cdiv(seqlen_k, meta["BLOCK_N"]), triton.cdiv(seqlen_q, meta["BLOCK_M"])),
      1,
      batch * nheads,
    )

  if autotune:
    _get_bwd_autotune(headdim, autotune_mode, main_bias_requires_grad)[grid](
      q,
      k,
      v,
      do,
      dq,
      dk,
      dv,
      lse,
      delta,
      attn_bias_in,
      grad_attn_bias_in,
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
      IS_CAUSAL=causal,
      HAS_ATTN_BIAS=has_attn_bias,
      HAS_DROPOUT=has_dropout,
      PHILOX_SEED=philox_seed,
      BIAS_REQUIRES_GRAD=main_bias_requires_grad,
      GRAD_BIAS_NEEDS_REDUCTION=grad_bias_needs_reduction,
      GRAD_BIAS_REDUCES_M=grad_bias_reduces_m,
      GRAD_BIAS_STORE_PARTIAL=grad_bias_store_partial,
      DTYPE=DTYPE,
    )
  else:
    main_config = persisted_main_config or {
      "BLOCK_M": 128,
      "BLOCK_N": 64,
      "BLOCK_HEADDIM": 64,
      "num_warps": 8,
      "num_stages": 2,
    }
    _ffpa_bwd[grid](
      q,
      k,
      v,
      do,
      dq,
      dk,
      dv,
      lse,
      delta,
      attn_bias_in,
      grad_attn_bias_in,
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
      IS_CAUSAL=causal,
      HAS_ATTN_BIAS=has_attn_bias,
      HAS_DROPOUT=has_dropout,
      PHILOX_SEED=philox_seed,
      BIAS_REQUIRES_GRAD=main_bias_requires_grad,
      GRAD_BIAS_NEEDS_REDUCTION=grad_bias_needs_reduction,
      GRAD_BIAS_REDUCES_M=grad_bias_reduces_m,
      GRAD_BIAS_STORE_PARTIAL=grad_bias_store_partial,
      DTYPE=DTYPE,
      **main_config,
    )

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
  autotune_mode: str = "fast",
  preprocess_d_chunk: bool = False,
  attn_bias: torch.Tensor | None = None,
  return_attn_bias_grad: bool = False,
  grad_v_storage_dtype: torch.dtype | None = None,
  dropout_p: float = 0.0,
  philox_seed: int = 0,
  philox_offset: int = 0,
  enable_tma: bool = False,
  enable_ws: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
  """Run the Triton FFPA backward path and return ``(dq, dk, dv, d_attn_bias)``.

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
  :param autotune_mode: Triton autotune search-space mode, ``"fast"`` or
    ``"max"``.
  :param preprocess_d_chunk: Whether to split the preprocess delta reduction
    across head-dim chunks.
  :param attn_bias: Optional additive attention bias broadcast to
    ``[B, Nh_q, Nq, Nkv]``.
  :param return_attn_bias_grad: Whether to materialize the expanded additive
    mask gradient for autograd.
  :param grad_v_storage_dtype: Optional storage dtype for the internal Triton
    ``DV`` buffer. ``None`` keeps v's dtype; ``torch.float32`` upgrades dV's
    cross-tile global accumulation storage while leaving dQ/dK storage in the
    original q/k dtype.
  :returns: ``(dq, dk, dv, d_attn_bias)`` where ``dq`` has shape ``[B, Nh_q, Nq, D]`` and
    ``dk`` / ``dv`` have shape ``[B, Nh_kv, Nkv, D]``. Returned tensors use
    the original ``q`` / ``k`` / ``v`` dtypes and head layouts. ``d_attn_bias``
    is the expanded bias gradient when requested, otherwise ``None``.
  """
  if grad_v_storage_dtype not in (None, torch.float32):
    raise ValueError(
      "_ffpa_attn_backward_triton: grad_v_storage_dtype must be None or torch.float32, "
      f"got {grad_v_storage_dtype!r}"
    )
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

  original_nheads_kv = k.size(1)
  group_size = q.size(1) // original_nheads_kv
  if group_size > 1:
    # GQA/MQA contract: kernels operate on expanded query-head layout. Gradients
    # for repeated KV heads are summed back into the original KV head dimension
    # after the Triton op returns.
    k_in = k.repeat_interleave(group_size, dim=1).contiguous()
    v_in = v.repeat_interleave(group_size, dim=1).contiguous()
  else:
    k_in, v_in = k, v

  dq, dk_expanded, dv_expanded, grad_attn_bias = torch.ops.ffpa_attn._bwd_triton(
    grad_out.contiguous(),
    q.contiguous(),
    k_in.contiguous(),
    v_in.contiguous(),
    o.contiguous(),
    lse,
    attn_bias,
    softmax_scale or (1.0 / math.sqrt(q.size(-1))),
    int(causal),
    int(autotune),
    int(autotune_mode == "max"),
    int(preprocess_d_chunk),
    int(return_attn_bias_grad and attn_bias is not None),
    int(grad_v_storage_dtype == torch.float32),
    original_nheads_kv,
    dropout_p,
    philox_seed,
    philox_offset,
    int(enable_tma),
    int(enable_ws),
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
  if grad_attn_bias.numel() == 0:
    grad_attn_bias_out = None
  else:
    grad_attn_bias_out = grad_attn_bias.to(attn_bias.dtype) if attn_bias is not None else grad_attn_bias
  return dq.to(q.dtype), dk, dv, grad_attn_bias_out

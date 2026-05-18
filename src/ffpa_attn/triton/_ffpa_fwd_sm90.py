"""SM90+ Triton forward entry points for experimental TMA kernels.

This module provides an SM90-specialized forward path that replaces raw-pointer
memory access with TMA descriptor loads/stores and can enable Triton's warp
specialized TMA pipeline on supported configs.

Design
------
* Q / K / V / O are passed as ``TensorDescriptor`` objects flattened to
  ``[B*H*N, D]`` so the kernel addresses them with simple ``(y, x)`` offsets.
* LSE and ``attn_bias`` remain raw pointers — LSE is too small to benefit from
  TMA, and ``attn_bias`` may carry stride-0 broadcast dimensions that are
  incompatible with descriptor semantics.
* The Split-D structure (head-dim chunks for QK, V accumulator groups) is
  preserved from the original generic kernel.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from ._ffpa_fwd import (
  _apply_dropout_to_p,
  _attn_bias_broadcast_strides,
  _update_o_accs,
)
from ._persistent_autotune import (
  PersistentConfigRequest,
  dtype_name,
  lookup_persistent_config,
)
from ._autotune_utils import autotune_seqlen_key


def _sm90_num_v_groups(args):
  return triton.cdiv(args["HEADDIM"], args["BLOCK_HEADDIM_V"])


_SM90_FWD_HEURISTICS = {
  "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
  "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
  "NUM_V_GROUPS": _sm90_num_v_groups,
}


@triton.jit
def _ffpa_fwd_sm90_process_kv_block(
  desc_q: tl.tensor_descriptor,
  desc_k: tl.tensor_descriptor,
  desc_v: tl.tensor_descriptor,
  AttnBias: torch.Tensor,
  q_offset_y: int,
  kv_base_y: int,
  start_n: int,
  off_hb: int,
  offs_m: int,
  offs_n: int,
  o_accs: tl.tensor,
  m_i: tl.tensor,
  l_i: tl.tensor,
  softmax_scale: float,
  stride_bm: int,
  stride_bn: int,
  seqlen_q: tl.constexpr,
  seqlen_k: tl.constexpr,
  kv_offset: tl.constexpr,
  dropout_p: float,
  philox_offset: int,
  IS_CAUSAL: tl.constexpr,
  HAS_ATTN_BIAS: tl.constexpr,
  HAS_DROPOUT: tl.constexpr,
  PHILOX_SEED: tl.constexpr,
  DTYPE: tl.constexpr,
  EVEN_N: tl.constexpr,
  BLOCK_M: tl.constexpr,
  BLOCK_N: tl.constexpr,
  BLOCK_HEADDIM_QK: tl.constexpr,
  BLOCK_HEADDIM_V: tl.constexpr,
  NUM_QK_D_CHUNKS: tl.constexpr,
  NUM_V_GROUPS: tl.constexpr,
):
  start_n = tl.multiple_of(start_n, BLOCK_N)
  offs_kv = start_n + offs_n
  k_offset_y = kv_base_y + start_n
  v_offset_y = kv_base_y + start_n

  scores = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
  # Phase 1: QK with Split-D reduction structure.
  for qk_d_chunk in range(NUM_QK_D_CHUNKS):
    qk_d_start = qk_d_chunk * BLOCK_HEADDIM_QK
    # TMA descriptor loads — OOB elements return 0 automatically.
    q = desc_q.load([q_offset_y, qk_d_start])
    k = desc_k.load([k_offset_y, qk_d_start])
    scores = tl.dot(q, tl.trans(k), acc=scores)

  scores = scores * softmax_scale
  if HAS_ATTN_BIAS:
    # attn_bias stays on raw pointer — may be stride-0 broadcast.
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

  # Phase 2: Online softmax.
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

  # Phase 3: PV with Split-D V accumulation.
  # Reuse the same softmax tile for all V slices, matching FFPA CUDA fwd:
  # R_D[j] = alpha * R_D[j] + P @ V_j. This avoids recomputing QK/softmax
  # per output head-dim group while keeping Split-D register pressure bounded.
  for v_group in tl.static_range(0, NUM_V_GROUPS):
    o_d_start = BLOCK_HEADDIM_V * v_group
    v = desc_v.load([v_offset_y, o_d_start])
    o_acc = o_accs[v_group] * alpha[:, None] + tl.dot(p, v)
    o_accs = _update_o_accs(o_accs, v_group, o_acc)
  return o_accs, m_new, l_new


def _sm90_host_descriptor_pre_hook(nargs):
  """Set per-descriptor block shapes before a TMA kernel launch.

  Called as a ``pre_hook`` on :class:`triton.Config` so that each autotune
  candidate updates the block shape to match its compile-time tile sizes.
  """
  if not isinstance(nargs.get("desc_q"), TensorDescriptor):
    return
  BLOCK_M = nargs["BLOCK_M"]
  BLOCK_N = nargs["BLOCK_N"]
  BLOCK_HEADDIM_QK = nargs["BLOCK_HEADDIM_QK"]
  BLOCK_HEADDIM_V = nargs["BLOCK_HEADDIM_V"]
  nargs["desc_q"].block_shape = [BLOCK_M, BLOCK_HEADDIM_QK]
  nargs["desc_k"].block_shape = [BLOCK_N, BLOCK_HEADDIM_QK]
  nargs["desc_v"].block_shape = [BLOCK_N, BLOCK_HEADDIM_V]
  nargs["desc_o"].block_shape = [BLOCK_M, BLOCK_HEADDIM_V]


@triton.heuristics(_SM90_FWD_HEURISTICS)
@triton.jit
def _ffpa_fwd_sm90_kernel_impl(
  desc_q: tl.tensor_descriptor,
  desc_k: tl.tensor_descriptor,
  desc_v: tl.tensor_descriptor,
  desc_o: tl.tensor_descriptor,
  LSE: torch.Tensor,
  AttnBias: torch.Tensor,
  O: torch.Tensor,
  stride_ob: int,
  stride_oh: int,
  stride_om: int,
  softmax_scale: float,
  stride_bb: int,
  stride_bh: int,
  stride_bm: int,
  stride_bn: int,
  nheads_q: int,
  nheads_kv: int,
  seqlen_q: tl.constexpr,
  seqlen_k: tl.constexpr,
  autotune_seqlen_q_bucket: int,
  autotune_seqlen_k_bucket: int,
  autotune_causal_key: int,
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
  warp_specialize: tl.constexpr,
) -> None:
  """TMA-descriptor variant of the Split-D FFPA generic forward kernel.

  Identical algorithm to ``_ffpa_fwd_kernel_impl`` with Q / K / V / O
  pointer arithmetic replaced by ``desc.load`` / ``desc.store`` calls.
  LSE and attn_bias stay on raw pointers.

  Keep this kernel at Triton's default ``num_ctas=1``. The fused Split-D
  attention body contains multiple ``tt.dot`` ops (QK plus one or more PV
  dots) in the same kernel. This is a kernel / Triton ``FuncOp`` level
  limitation, not a loop-local limitation: the current PlanCTA planner can
  only see one ``DotOp`` per kernel when ``num_ctas=2``. Triton 3.6/3.7 runs
  the NVIDIA CTA planning pass (``TritonGPUPlanCTAPass``, pipeline name
  ``triton-nvidia-gpu-plan-cta``) and the second ``DotOp`` hits the
  PlanCTA.cpp assertion ``!tiled && "CTA tiling is already determined"``.
  """
  # Keys for autotuning heuristics and persistent autotune lookup.
  _ = autotune_seqlen_q_bucket
  _ = autotune_seqlen_k_bucket
  _ = autotune_causal_key

  start_m = tl.program_id(0)
  off_hb = tl.program_id(1)
  off_b = off_hb // nheads_q
  off_hq = off_hb % nheads_q
  group_size = nheads_q // nheads_kv
  off_hkv = off_hq // group_size

  # per-program offsets
  q_base_y = (off_b * nheads_q + off_hq) * seqlen_q
  kv_base_y = (off_b * nheads_kv + off_hkv) * seqlen_k
  q_offset_y = q_base_y + start_m * BLOCK_M
  o_offset_y = q_offset_y  # O follows Q layout

  LSE += off_hb * seqlen_q_rounded
  O += off_b * stride_ob + off_hq * stride_oh
  if HAS_ATTN_BIAS:
    AttnBias += off_b * stride_bb + off_hq * stride_bh

  # arange helpers
  offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
  offs_n = tl.arange(0, BLOCK_N)

  num_qk_d_chunks = tl.cdiv(HEADDIM, BLOCK_HEADDIM_QK)
  kv_offset = seqlen_k - seqlen_q

  # Online softmax state
  m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
  l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
  zero_acc = tl.zeros([BLOCK_M, BLOCK_HEADDIM_V], dtype=tl.float32)
  # Reuse the same softmax tile for all V slices, matching FFPA CUDA fwd:
  # R_D[j] = alpha * R_D[j] + P @ V_j. This avoids recomputing QK/softmax
  # per output head-dim group while keeping Split-D register pressure bounded.
  o_accs = (zero_acc, ) * NUM_V_GROUPS

  end_n = seqlen_k
  if IS_CAUSAL:
    end_n = tl.minimum(seqlen_k, (start_m + 1) * BLOCK_M + kv_offset)

  if warp_specialize:
    # KV loop / warp-specialization boundary.
    # disallow_acc_multi_buffer=True is needed for the Split-D accumulator shape.
    # The total fp32 state is comparable to a FA2-style [BLOCK_M, HEAD_DIM] acc,
    # but here Triton sees multiple loop-carried dot accumulators in o_accs plus
    # QK split-D dot sites. Letting it multi-buffer those independent accumulators
    # makes WS partitioning/resource use much less stable on the validated tiles.
    # flatten=True gives Triton's WS partitioner a single flattened loop body.
    for start_n in tl.range(
      0,
      end_n,
      BLOCK_N,
      disallow_acc_multi_buffer=True,
      flatten=True,
      warp_specialize=True,
    ):
      o_accs, m_i, l_i = _ffpa_fwd_sm90_process_kv_block(
        desc_q,
        desc_k,
        desc_v,
        AttnBias,
        q_offset_y,
        kv_base_y,
        start_n,
        off_hb,
        offs_m,
        offs_n,
        o_accs,
        m_i,
        l_i,
        softmax_scale,
        stride_bm,
        stride_bn,
        seqlen_q,
        seqlen_k,
        kv_offset,
        dropout_p,
        philox_offset,
        IS_CAUSAL,
        HAS_ATTN_BIAS,
        HAS_DROPOUT,
        PHILOX_SEED,
        DTYPE,
        EVEN_N,
        BLOCK_M,
        BLOCK_N,
        BLOCK_HEADDIM_QK,
        BLOCK_HEADDIM_V,
        num_qk_d_chunks,
        NUM_V_GROUPS,
      )
  else:
    # Keep the non-WS TMA path on the original plain range. A tl.range with
    # WS-only attrs still changes non-WS codegen and caps the stage3/4 TMA
    # candidates that were the fast path before WS was added.
    for start_n in range(0, end_n, BLOCK_N):
      o_accs, m_i, l_i = _ffpa_fwd_sm90_process_kv_block(
        desc_q,
        desc_k,
        desc_v,
        AttnBias,
        q_offset_y,
        kv_base_y,
        start_n,
        off_hb,
        offs_m,
        offs_n,
        o_accs,
        m_i,
        l_i,
        softmax_scale,
        stride_bm,
        stride_bn,
        seqlen_q,
        seqlen_k,
        kv_offset,
        dropout_p,
        philox_offset,
        IS_CAUSAL,
        HAS_ATTN_BIAS,
        HAS_DROPOUT,
        PHILOX_SEED,
        DTYPE,
        EVEN_N,
        BLOCK_M,
        BLOCK_N,
        BLOCK_HEADDIM_QK,
        BLOCK_HEADDIM_V,
        num_qk_d_chunks,
        NUM_V_GROUPS,
      )

  # Phase 4: Epilogue - final scale O and write O/LSE to global memory.
  # Write O via descriptor when aligned, raw pointer otherwise.
  # TensorDescriptor.store ignores offsets outside the descriptor's global
  # [B*H*N, D] bounds, but it cannot see the per-head seqlen_q boundary. On a
  # partial final M block, rows with offs_m >= seqlen_q may still be inside the
  # flattened descriptor and alias the next head/batch rows, so non-aligned
  # seqlen_q still needs an explicit raw-pointer mask.
  if EVEN_M:
    for v_group in tl.static_range(0, NUM_V_GROUPS):
      o_d_start = BLOCK_HEADDIM_V * v_group
      out = o_accs[v_group] / (l_i[:, None] + 1.0e-10)
      desc_o.store([o_offset_y, o_d_start], out.to(DTYPE))
  else:
    offs_d_v = tl.arange(0, BLOCK_HEADDIM_V)
    for v_group in tl.static_range(0, NUM_V_GROUPS):
      o_d = BLOCK_HEADDIM_V * v_group + offs_d_v
      out = o_accs[v_group] / (l_i[:, None] + 1.0e-10)
      tl.store(
        O + offs_m[:, None] * stride_om + o_d[None, :],
        out.to(DTYPE),
        mask=(offs_m[:, None] < seqlen_q) & (o_d[None, :] < HEADDIM),
      )
  tl.store(LSE + offs_m, m_i + tl.log(l_i), mask=offs_m < seqlen_q)


_SM90_DEFAULT_CONFIG = {
  "BLOCK_M": 64,
  "BLOCK_N": 128,
  "BLOCK_HEADDIM_QK": 64,
  "BLOCK_HEADDIM_V": 64,
  # Fixed-launch fallback is non-WS by default, so launch num_stages=3 does
  # not hit the WS resource path that forced warp-specialized configs to 2.
  "warp_specialize": False,
  "num_warps": 4,
  "num_stages": 3,
}


def _gen_fwd_sm90_autotune_configs(
  autotune_mode: str = "max",
  enable_ws: bool = True,
) -> list[triton.Config]:
  """Generate autotune configs for the SM90 TMA forward kernel.

  The search space is compact: tune BLOCK_M, head-dim chunk size, warp
  count, and pipeline depth.  Every config carries
  :func:`_sm90_host_descriptor_pre_hook` so the descriptor block shapes
  are updated before each trial.

  :param headdim: The actual head dimension for the target shape.
  :param autotune_mode: Triton autotune search-space mode, ``"fast"`` or
      ``"max"``.
  :param enable_ws: Whether to generate only warp-specialized configs.
  :return: Triton autotune configurations for the SM90 TMA forward kernel.
  """
  # fast: 2*2*2*1 = 8 configs; max: 2*2*2*2 = 16 configs
  configs = []
  for block_m in [64, 128]:
    for block_n in [64, 128]:
      for block_headdim in [64, 128]:
        for num_warps in [4] if autotune_mode == "fast" else [4, 8]:
          configs.append(
            triton.Config(
              {
                "BLOCK_M": block_m,
                "BLOCK_N": block_n,
                "BLOCK_HEADDIM_QK": block_headdim,
                "BLOCK_HEADDIM_V": block_headdim,
                "warp_specialize": enable_ws,
              },
              num_warps=num_warps,
              num_stages=3,  # 3 is almost always the better choice
              pre_hook=_sm90_host_descriptor_pre_hook,
            )
          )
  return configs


_ffpa_fwd_sm90_autotune_cache: dict[tuple[int, str, str, bool], callable] = {}


def _get_fwd_sm90_autotune(headdim: int, autotune_mode: str, dtype: str, enable_ws: bool = True):
  """Return a headdim-specific autotune wrapper for the SM90 TMA kernel.

  Results are cached by headdim so the autotune overhead is paid at most
  once per shape on a given process.

  :param headdim: The actual head dimension for the target forward call.
  :param autotune_mode: Triton autotune search-space mode, ``"fast"`` or
      ``"max"``.
  :param dtype: Dtype name matching :func:`dtype_name`.
  :param enable_ws: Whether to use only warp-specialized configs.
  :return: A ``triton.autotune``-wrapped version of
      ``_ffpa_fwd_sm90_kernel_impl`` tuned for ``headdim``.
  """
  cache_key = (headdim, autotune_mode, dtype, enable_ws)
  if cache_key not in _ffpa_fwd_sm90_autotune_cache:
    configs = _gen_fwd_sm90_autotune_configs(
      autotune_mode=autotune_mode,
      enable_ws=enable_ws,
    )
    _ffpa_fwd_sm90_autotune_cache[cache_key] = triton.autotune(
      configs=configs,
      key=[
        "autotune_seqlen_q_bucket",
        "autotune_seqlen_k_bucket",
        "autotune_causal_key",
        "HEADDIM",
      ],
      cache_results=True,
    )(_ffpa_fwd_sm90_kernel_impl)
  return _ffpa_fwd_sm90_autotune_cache[cache_key]


def _ffpa_attn_forward_sm90_generic_impl(
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
  enable_ws: bool = False,
) -> None:
  """Launch the SM90 TMA forward kernel (generic prefill path).

  This is the TMA counterpart of ``_ffpa_attn_forward_generic_impl``.
  Phase 1 uses a fixed launch config; autotune integration is deferred
  to a later phase.
  """
  batch, nheads_q, seqlen_q, headdim = q.shape
  _, nheads_kv, seqlen_k, _ = k.shape
  softmax_scale = softmax_scale or (1.0 / math.sqrt(headdim))
  seqlen_q_rounded = lse.shape[-1]
  DTYPE = tl.float16 if q.dtype == torch.float16 else tl.bfloat16
  has_attn_bias = attn_bias is not None
  has_dropout = dropout_p > 0.0
  attn_bias_in = attn_bias if attn_bias is not None else q
  bias_strides = _attn_bias_broadcast_strides(attn_bias, batch, nheads_q, seqlen_q, seqlen_k)

  launch_config = dict(_SM90_DEFAULT_CONFIG)
  if enable_ws:
    launch_config["warp_specialize"] = True
    launch_config["num_stages"] = 2

  # When autotune is requested, the autotune wrapper (with pre_hook) manages
  # block_shape updates; the launcher only needs to pick a fixed-path config.
  if not autotune:
    persisted = lookup_persistent_config(
      PersistentConfigRequest(
        direction="forward",
        kernel="fwd_sm90_generic",
        autotune_mode=autotune_mode,
        dtype=dtype_name(q.dtype),
        headdim=headdim,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        causal=causal,
        has_attn_bias=has_attn_bias,
        has_dropout=has_dropout,
        enable_tma=True,
        enable_ws=enable_ws,
        nheads_q=nheads_q,
        nheads_kv=nheads_kv,
        device_index=q.device.index,
      )
    )
    if persisted is not None:
      launch_config = persisted

  y_dim_q = batch * nheads_q * seqlen_q
  y_dim_kv = batch * nheads_kv * seqlen_k
  dummy_block = [1, 1]

  def _make_tensor_desc(x: torch.Tensor, shape: list[int]) -> TensorDescriptor:
    # The TMA path uses TensorDescriptors for Q/K/V/O with a simple [B*H*N, D] layout
    # shape = [B*H*N, D], strides = [D, 1] so that the kernel can index with (y, x) offsets.
    return TensorDescriptor(x, shape=shape, strides=[shape[1], 1], block_shape=dummy_block)

  desc_q = _make_tensor_desc(q, [y_dim_q, headdim])
  desc_k = _make_tensor_desc(k, [y_dim_kv, headdim])
  desc_v = _make_tensor_desc(v, [y_dim_kv, headdim])
  desc_o = _make_tensor_desc(o, [y_dim_q, headdim])

  # For the fixed-config path we set block_shape explicitly.  The autotune
  # path leaves block_shape as dummy because the pre_hook on each Config
  # updates them before every trial and final invocation.
  if not autotune:
    desc_q.block_shape = [launch_config["BLOCK_M"], launch_config["BLOCK_HEADDIM_QK"]]
    desc_k.block_shape = [launch_config["BLOCK_N"], launch_config["BLOCK_HEADDIM_QK"]]
    desc_v.block_shape = [launch_config["BLOCK_N"], launch_config["BLOCK_HEADDIM_V"]]
    desc_o.block_shape = [launch_config["BLOCK_M"], launch_config["BLOCK_HEADDIM_V"]]

  # TMA allocator (required for descriptor path)
  def _tma_alloc_fn(size: int, align: int, _):
    return torch.empty(size, dtype=torch.int8, device=q.device)

  triton.set_allocator(_tma_alloc_fn)

  # bucket keys (used by autotune cache key)
  autotune_seqlen_q_bucket = autotune_seqlen_key(seqlen_q, autotune_mode)
  autotune_seqlen_k_bucket = autotune_seqlen_key(seqlen_k, autotune_mode)
  autotune_causal_key = int(causal)

  if autotune:

    def grid(meta):
      return (triton.cdiv(seqlen_q, meta["BLOCK_M"]), batch * nheads_q)

    _get_fwd_sm90_autotune(
      headdim,
      autotune_mode,
      dtype_name(q.dtype),
      enable_ws=enable_ws,
    )[grid](
      desc_q,
      desc_k,
      desc_v,
      desc_o,
      lse,
      attn_bias_in,
      o,
      o.stride(0),
      o.stride(1),
      o.stride(2),
      softmax_scale,
      bias_strides[0],
      bias_strides[1],
      bias_strides[2],
      bias_strides[3],
      nheads_q,
      nheads_kv,
      seqlen_q,
      seqlen_k,
      autotune_seqlen_q_bucket,
      autotune_seqlen_k_bucket,
      autotune_causal_key,
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
    return

  def grid(meta):
    return (triton.cdiv(seqlen_q, meta["BLOCK_M"]), batch * nheads_q)

  _ffpa_fwd_sm90_kernel_impl[grid](
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    lse,
    attn_bias_in,
    o,
    o.stride(0),
    o.stride(1),
    o.stride(2),
    softmax_scale,
    bias_strides[0],
    bias_strides[1],
    bias_strides[2],
    bias_strides[3],
    nheads_q,
    nheads_kv,
    seqlen_q,
    seqlen_k,
    autotune_seqlen_q_bucket,
    autotune_seqlen_k_bucket,
    autotune_causal_key,
    seqlen_q_rounded,
    dropout_p,
    philox_offset,
    IS_CAUSAL=causal,
    HAS_ATTN_BIAS=has_attn_bias,
    HAS_DROPOUT=has_dropout,
    PHILOX_SEED=philox_seed,
    DTYPE=DTYPE,
    HEADDIM=headdim,
    **launch_config,
  )


def is_sm90_tma_forward_supported(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  o: torch.Tensor,
  *,
  num_splits: int,
) -> bool:
  """Return whether the experimental SM90 TMA forward path may run.

  The caller must also pass ``enable_tma=True``; this function only checks
  hardware / shape / dtype preconditions.  On SM < 90 (including Ada L20)
  it returns ``False`` so the existing generic path is used silently.

  :param q: Query tensor in ``[B, Hq, Nq, D]`` layout.
  :param k: Key tensor in ``[B, Hkv, Nkv, D]`` layout.
  :param v: Value tensor in ``[B, Hkv, Nkv, D]`` layout.
  :param o: Output tensor in ``[B, Hq, Nq, D]`` layout.
  :param num_splits: Decode split count selected by the generic dispatcher.
  :return: ``True`` when the call is eligible for the SM90 generic prefill
    path; otherwise ``False`` so the caller can use the existing fallback.
  """
  if num_splits != 1:
    return False
  if not q.is_cuda:
    return False
  if torch.cuda.get_device_capability(q.device)[0] < 9:
    return False
  if q.dtype not in (torch.float16, torch.bfloat16):
    return False
  return q.stride(-1) == k.stride(-1) == v.stride(-1) == o.stride(-1) == 1


def _ffpa_attn_forward_sm90_tma_impl(
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
  enable_ws: bool = False,
) -> None:
  """Run the SM90 TMA forward implementation.

  This is the integration scaffold for the descriptor/TMA kernel.  Phase 1
  implements a non-warp-specialized TMA kernel that replaces raw-pointer
  memory access with descriptor loads/stores while preserving the Split-D
  algorithm structure.

  :param q: Query tensor in ``[B, Hq, Nq, D]`` layout.
  :param k: Key tensor in ``[B, Hkv, Nkv, D]`` layout.
  :param v: Value tensor in ``[B, Hkv, Nkv, D]`` layout.
  :param o: Output tensor in ``[B, Hq, Nq, D]`` layout, written in place.
  :param lse: Float32 LSE tensor with rounded last-dimension storage.
  :param attn_bias: Optional additive mask broadcastable to
    ``[B, Hq, Nq, Nkv]``.
  :param causal: Whether to apply lower-right causal masking.
  :param softmax_scale: Scale applied to ``Q @ K.T``.
  :param autotune: Whether to use the Triton autotuner for the SM90 TMA path.
  :param autotune_mode: Triton autotune search-space mode, ``"fast"`` or
    ``"max"``.
  :param dropout_p: Forward dropout probability.
  :param philox_seed: Philox seed used for dropout.
  :param philox_offset: Philox element offset used for dropout replay parity
    with SDPA.
  """
  _ffpa_attn_forward_sm90_generic_impl(
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
    enable_ws=enable_ws,
  )

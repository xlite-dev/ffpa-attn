"""Triton FFPA attention forward/backward implementations for large-D
(D > 256, but also works for D <= 256).

.. todo:: Triton varlen (packed-sequence) support plan

  **Current state**: the Triton fwd/bwd kernels require dense ``[B, N, H, D]``
  tensors with uniform ``seqlen_q`` / ``seqlen_k`` across the batch dimension.
  The CuTeDSL backend (both SM80 and SM90 paths) already supports varlen via
  packed ``[total_q, H, D]`` tensors and ``cu_seqlens_q`` / ``cu_seqlens_k``
  cumulative-offset tensors.  This plan describes how to add the same varlen
  contract to the Triton backend without regressing dense-path performance.

  **Why this works without performance loss**

  The overhead of varlen support is negligible because:

  * **cu_seqlens reads**: each program loads 2–4 int32 scalars from
    ``cu_seqlens_q`` / ``cu_seqlens_k`` once at the top of the kernel.  These
    are coalesced loads served from L2 (the offsets tensor is tiny, typically a
    few hundred bytes) and cost ~10 cycles per program — invisible in a
    1024-program grid.
  * **Pointer-offset arithmetic**: the existing dense path computes
    ``off_b * stride_qb + off_h * stride_qh`` for every Q/K/V load.  The
    varlen path replaces ``off_b * stride_qb`` with
    ``cu_seqlens_q[seq_idx] * nheads * headdim`` (a single integer multiply-add
    that was already being done for the stride-based address).  No new
    arithmetic in the innermost loops.
  * **seqlen from scalar → variable**: in the dense kernel ``seqlen_q`` /
    ``seqlen_k`` are already runtime values read from a kernel argument
    register.  Reading them from the ``cu_seqlens`` buffers instead of a
    uniform argument costs nothing extra – it just changes *which* register
    holds the value.
  * **One launch, not N**: unlike the SM80 CuTeDSL *backward* kernels that
    require a Python-side per-segment decomposition, the Triton kernels natively
    map ``program_id`` to ``(tile, batch, head)``.  Adding varlen only requires
    remapping ``off_hb`` to ``(seq_idx, head)`` — the total number of programs
    remains ``num_seqs * nheads``, launched in a single ``triton.grid``.

  **Areas requiring care**

  1. **BWD shared-pid grid dim 0 padding**.  The backward kernel uses a
     shared-program-id design: ``pid`` simultaneously indexes a K-column block
     (dK/dV) and a Q-row block (dQ).  Grid dim 0 is
     ``max(cdiv(Nk, BLOCK_N), cdiv(Nq, BLOCK_M))``.  For varlen, different
     sequences have different ``Nq`` and ``Nk``, so grid dim 0 must be the
     *maximum* over all sequences.  Programs whose ``pid`` exceeds the current
     sequence's tile count will early-return at the top of the kernel (a
     single ``if pid >= num_tiles: return``).  The extra "empty" programs
     are quickly skipped by the SM scheduler and add at most ~20 % launch
     overhead in the worst case (highly variable sequence lengths).

  2. **BWD preprocess Delta layout**.  The preprocess kernel computes
     ``D = rowsum(dO * O)``, stored as ``[batch * nheads, seqlen_q_rounded]``.
     For varlen, Delta must switch to a head-grouped layout:
     ``[nheads, total_q_rounded]`` where ``total_q_rounded`` is the sum over
     all sequences of ``ceil(seqlen_q_i / BLOCK_M) * BLOCK_M``.  Each program
     needs to compute its sequence's start row in this packed Delta tensor.
     The LSE tensor follows the same layout convention.

  3. **Autotune key for varlen**.  The fwd kernel already uses ``fast`` mode
     by default, bucketing by ``(headdim, BLOCK_M, BLOCK_N)`` without
     seqlen-dependent keys.  Varlen adds no new autotune dimension — the same
     tuned config works for any sequence length.  The ``causal`` key remains
     identical.  For ``"max"`` autotune mode, the ``autotune_seqlen_q_bucket``
     / ``autotune_seqlen_k_bucket`` keys should use a conservative bucket
     (e.g. ``max_seqlen`` rounded up to the next power of two) to avoid cache
     explosion from per-sequence-length tuning.

  4. **dropout Philox offset**.  Dropout uses the logical ``[B, Hq, Nq, Nkv]``
     element offset.  For varlen, replace the batch-index part with a
     ``cu_seqlens_q``-based cumulative offset so sequences do not share Philox
     states.

  5. **GQA/MQA with varlen**.  The current GQA mapping is
     ``off_hkv = off_hq // group_size``.  This is unchanged for varlen — the
     head index is independent of sequence packing.  The only change is that
     ``off_b`` is derived from ``cu_seqlens`` rather than from ``off_hb``
     division.

  **Implementation steps**

  **Step 1 — Add ``CuSeqlensQ`` / ``CuSeqlensK`` kernel arguments**

  Both fwd and bwd kernels gain two new optional ``torch.Tensor`` arguments
  pointing to int32 ``cu_seqlens_q`` / ``cu_seqlens_k`` (matching the CuTeDSL
  convention).  At the top of each kernel, map ``off_hb`` to
  ``(seq_idx, head_idx)``:

  .. code-block:: python

     seq_idx = off_hb // nheads_q
     off_hq = off_hb % nheads_q

     q_start = tl.load(CuSeqlensQ + seq_idx)
     q_end   = tl.load(CuSeqlensQ + seq_idx + 1)
     seqlen_q = q_end - q_start

     k_start = tl.load(CuSeqlensK + seq_idx)
     k_end   = tl.load(CuSeqlensK + seq_idx + 1)
     seqlen_k = k_end - k_start

  ``seqlen_q`` and ``seqlen_k`` replace the uniform scalar kernel args for the
  varlen path.  A ``constexpr`` / heuristic boolean guards the dense code path
  so non-varlen kernels see zero overhead (the loads are dead-code-eliminated
  by Triton's compiler when the dense path is taken).

  **Step 2 — Remap tensor base addresses**

  Replace the dense batch-stride pointer arithmetic:

  .. code-block:: python

     # Dense:
     Q += off_b * stride_qb + off_hq * stride_qh
     # Varlen:
     Q += q_start * nheads_q * headdim + off_hq * headdim

  The ``stride_qm``, ``stride_kn``, ``stride_vn`` strides (inner dimensions)
  are unchanged — they still point to consecutive elements in head-dimension
  order.  The same transformation applies to ``K``, ``V``, ``O``, ``DO``,
  ``DQ``, ``DK``, ``DV``.

  **Step 3 — Grid adaptation**

  * FWD grid: ``(cdiv(total_q // num_seqs, BLOCK_M), num_seqs * nheads_q)``
    or conservatively ``(cdiv(max_seqlen_q, BLOCK_M), num_seqs * nheads_q)``
    with out-of-range rows silently masked to zero.
  * BWD grid (shared-pid): dim 0 = ``max over seqs of
    max(cdiv(Nk_seq, BLOCK_N), cdiv(Nq_seq, BLOCK_M))``.  Each program
    early-returns if ``pid >= max_tiles(seq_idx)``.
  * BWD pre grid: ``(cdiv(max_seqlen_q, BLOCK_M), num_seqs * nheads)``.
    Programs for rows beyond the actual sequence length write nothing.

  **Step 4 — LSE / Delta layout for varlen**

  For dense, LSE and Delta are ``[batch * nheads, seqlen_q_rounded]``.  For
  varlen, switch to a head-grouped packed layout:

  * Shape: ``[nheads, total_q_rounded]`` where
    ``total_q_rounded = sum_i ceil(seqlen_q_i / BLOCK_M) * BLOCK_M``.
  * A program with ``(seq_idx, head)`` computes:
    ``base_row = sum_{s < seq_idx} ceil(seqlen_q_s / BLOCK_M) * BLOCK_M``.
  * Read / write LSE / Delta at ``base_row + offs_m``.

  This avoids materializing a per-sequence padding stride and keeps the
  storage dense.

  **Step 5 — Autotune & host wrapper**

  * Keep the existing ``fast`` autotune key unchanged — it already does not
    depend on seqlen.
  * For ``max`` mode, bucket sequences by their *individual* seqlen
    (not the pack total) using the existing ``autotune_seqlen_key`` helper.
  * Host wrapper: the public ``_ffpa_attn_forward_triton`` /
    ``_ffpa_attn_backward_triton`` functions gain optional
    ``cu_seqlens_q`` / ``cu_seqlens_k`` parameters.  When ``None``, the
    existing dense fast-path is taken (zero overhead).  When provided, the
    varlen path described above is activated via a ``constexpr``-style
    boolean flag baked into the JIT specialization.

  **Expected outcome**

  * Dense path: zero regression (varlen logic is dead-code-eliminated when
    ``cu_seqlens`` is ``None``).
  * Varlen fwd/bwd: < 5 % overhead vs an ideal dense kernel with the same
    total number of tokens, because every varlen overhead is either a
    program-entry-time load or a pointer-offset recomputation that replaces
    an already-existing stride computation.
"""
import torch

from ._ffpa_fwd import _ffpa_attn_forward_triton
from ._ffpa_bwd import _ffpa_attn_backward_triton

_OP_NAMESPACE = "ffpa_attn"


def _attn_bias_grad_needs_reduction(
  attn_bias: torch.Tensor | None, q: torch.Tensor, k: torch.Tensor
) -> bool:
  """Return whether compact bias gradients are reduced across broadcast dimensions."""
  if attn_bias is None:
    return False
  return any([
    attn_bias.size(0) == 1 and q.size(0) > 1,
    attn_bias.size(1) == 1 and q.size(1) > 1,
    attn_bias.size(2) == 1 and q.size(2) > 1,
    attn_bias.size(3) == 1 and k.size(2) > 1,
  ])


def _attn_bias_grad_dtype(
  attn_bias: torch.Tensor, q: torch.Tensor, k: torch.Tensor
) -> torch.dtype:
  """Return the internal accumulation dtype for Triton bias gradients.

  BF16 additive-bias gradients are numerically sensitive even when the logical
  mask shape does not require broadcast reduction. Keep the Triton accumulation
  buffer in fp32 and cast back to the user dtype at the Python wrapper
  boundary so the kernel only rounds once.
  """
  if attn_bias.dtype == torch.bfloat16 or _attn_bias_grad_needs_reduction(
    attn_bias, q, k
  ):
    return torch.float32
  return attn_bias.dtype


def _triton_bwd_grad_tensor_like(
  tensor: torch.Tensor,
  grad_storage_dtype: torch.dtype | None = None,
) -> torch.Tensor:
  """Allocate the internal Triton backward output buffer for one gradient.

  This allocation also determines the dtype used by the Triton kernel's global
  ``tl.load`` / ``tl.store`` traffic on ``DQ`` / ``DK`` / ``DV``. For example,
  when ``_triton_bwd_grad_tensor_like(k)`` returns bf16 storage, the backward
  kernel's ``tl.load(dk_ptrs)`` / ``tl.store(dk_ptrs, ...)`` sites operate on
  bf16 values. Returning fp16/fp32 here therefore changes the kernel's cross-tile
  global accumulation dtype, not just the Python-visible output tensor dtype.

  Memory note:
  The fp32 override is expensive for large tensors, and the Triton wrapper
  passes already-expanded K/V tensors for GQA/MQA. One fp32 buffer costs

  ``tensor.numel() * 4`` bytes,

  so a typical large-D self-attention or causal shape
  ``B=1, Hq=32, Nq=Nkv=8192, D=512`` allocates

  ``1 * 32 * 8192 * 512 * 4 = 536870912`` bytes per buffer = ``512 MiB``,

  Because K/V storage follows the expanded query-head layout, this fp32 cost
  also applies to GQA/MQA after head expansion, even if the original KV tensors
  had fewer heads.

  Recommendation:
  keep fp32 storage targeted to gradients that need higher cross-tile
  accumulation precision. For the current Triton backward path this is dK/dV.

  :param tensor: Reference tensor that provides shape, device, and default
    dtype.
  :param grad_storage_dtype: Optional storage dtype for this internal gradient
    buffer. ``None`` keeps the user-visible activation dtype; ``torch.float16``
    or ``torch.float32`` overrides cross-tile global accumulation storage.
  :return: Newly allocated gradient buffer.
  """
  if grad_storage_dtype is None:
    return torch.empty_like(tensor)
  return torch.empty_like(tensor, dtype=grad_storage_dtype)


def _grad_kv_storage_dtype_from_code(code: int) -> torch.dtype | None:
  """Decode the internal dK/dV storage dtype selector."""
  if code == 0:
    return None
  if code == 1:
    return torch.float32
  if code == 2:
    return torch.float16
  raise ValueError(
    f"Unsupported grad_kv_storage_dtype code {code}; expected 0, 1, or 2."
  )


torch.library.define(
  f"{_OP_NAMESPACE}::_fwd_triton",
  "(Tensor q, Tensor k, Tensor v, Tensor? attn_bias, float softmax_scale, "
  "int causal, int autotune, int autotune_mode_is_max, float dropout_p, int philox_seed, int philox_offset, "
  "int enable_tma, int enable_ws) "
  "-> (Tensor o, Tensor softmax_lse)",
)


@torch.library.impl(f"{_OP_NAMESPACE}::_fwd_triton", "CUDA")
def _fwd_triton_torch_op(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  attn_bias: torch.Tensor | None,
  softmax_scale: float,
  causal: int,
  autotune: int,
  autotune_mode_is_max: int,
  dropout_p: float,
  philox_seed: int,
  philox_offset: int,
  enable_tma: int,
  enable_ws: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  from ._ffpa_fwd import _ffpa_attn_forward_impl as _triton_fwd_kernel

  if q.stride(-1) != 1:
    q = q.contiguous()
  if k.stride(-1) != 1:
    k = k.contiguous()
  if v.stride(-1) != 1:
    v = v.contiguous()

  o = torch.empty_like(q)
  seqlen_q = q.size(2)
  seqlen_q_aligned = ((seqlen_q + 127) // 128) * 128
  softmax_lse = torch.empty(
    q.size(0),
    q.size(1),
    seqlen_q_aligned,
    dtype=torch.float32,
    device=q.device,
  )
  _triton_fwd_kernel(
    q,
    k,
    v,
    o,
    softmax_lse,
    attn_bias=attn_bias,
    causal=bool(causal),
    softmax_scale=softmax_scale,
    autotune=bool(autotune),
    autotune_mode="max" if autotune_mode_is_max else "fast",
    dropout_p=dropout_p,
    philox_seed=philox_seed,
    philox_offset=philox_offset,
    enable_tma=bool(enable_tma),
    enable_ws=bool(enable_ws),
  )
  return o, softmax_lse


@torch.library.register_fake(f"{_OP_NAMESPACE}::_fwd_triton")
def _fwd_triton_fake(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  attn_bias: torch.Tensor | None,
  softmax_scale: float,
  causal: int,
  autotune: int,
  autotune_mode_is_max: int,
  dropout_p: float,
  philox_seed: int,
  philox_offset: int,
  enable_tma: int,
  enable_ws: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  seqlen_q_aligned = ((q.size(2) + 127) // 128) * 128
  o = torch.empty_like(q)
  softmax_lse = q.new_empty(
    q.size(0), q.size(1), seqlen_q_aligned, dtype=torch.float32
  )
  return o, softmax_lse


torch.library.define(
  f"{_OP_NAMESPACE}::_bwd_triton",
  "(Tensor dO, Tensor q, Tensor k, Tensor v, Tensor o, Tensor lse, Tensor? attn_bias, "
  "float softmax_scale, int causal, int autotune, "
  "int autotune_mode_is_max, int preprocess_d_chunk, int return_attn_bias_grad, int grad_kv_storage_dtype_code, "
  "int original_nheads_kv, float dropout_p, int philox_seed, int philox_offset, int enable_tma, int enable_ws, "
  "int enable_persist_dkdv, int enable_split_launch) "
  "-> (Tensor dq, Tensor dk, Tensor dv, Tensor grad_attn_bias)",
)


@torch.library.impl(f"{_OP_NAMESPACE}::_bwd_triton", "CUDA")
def _bwd_triton_torch_op(
  do: torch.Tensor,
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  o: torch.Tensor,
  lse: torch.Tensor,
  attn_bias: torch.Tensor | None,
  softmax_scale: float,
  causal: int,
  autotune: int,
  autotune_mode_is_max: int,
  preprocess_d_chunk: int,
  return_attn_bias_grad: int,
  grad_kv_storage_dtype_code: int,
  original_nheads_kv: int,
  dropout_p: float,
  philox_seed: int,
  philox_offset: int,
  enable_tma: int,
  enable_ws: int,
  enable_persist_dkdv: int,
  enable_split_launch: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  from ._ffpa_bwd import _ffpa_attn_backward_triton_impl as _triton_bwd_kernel

  grad_kv_storage_dtype = _grad_kv_storage_dtype_from_code(
    grad_kv_storage_dtype_code
  )
  dq = _triton_bwd_grad_tensor_like(q)
  dk = _triton_bwd_grad_tensor_like(k, grad_kv_storage_dtype)
  dv = _triton_bwd_grad_tensor_like(v, grad_kv_storage_dtype)
  if attn_bias is not None and return_attn_bias_grad:
    grad_dtype = _attn_bias_grad_dtype(attn_bias, q, k)
    grad_attn_bias = torch.empty_like(attn_bias, dtype=grad_dtype)
  else:
    grad_attn_bias = q.new_empty(0)
  if grad_attn_bias.numel() > 0:
    grad_attn_bias.zero_()

  _triton_bwd_kernel(
    do=do,
    q=q,
    k=k,
    v=v,
    o=o,
    lse=lse,
    attn_bias=attn_bias,
    dq=dq,
    dk=dk,
    dv=dv,
    grad_attn_bias=grad_attn_bias if grad_attn_bias.numel() > 0 else None,
    causal=bool(causal),
    softmax_scale=softmax_scale,
    autotune=bool(autotune),
    autotune_mode="max" if autotune_mode_is_max else "fast",
    preprocess_d_chunk=bool(preprocess_d_chunk),
    original_nheads_kv=original_nheads_kv,
    dropout_p=dropout_p,
    philox_seed=philox_seed,
    philox_offset=philox_offset,
    enable_tma=bool(enable_tma),
    enable_ws=bool(enable_ws),
    enable_persist_dkdv=bool(enable_persist_dkdv),
    enable_split_launch=bool(enable_split_launch),
  )
  return dq, dk, dv, grad_attn_bias


@torch.library.register_fake(f"{_OP_NAMESPACE}::_bwd_triton")
def _bwd_triton_fake(
  do: torch.Tensor,
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  o: torch.Tensor,
  lse: torch.Tensor,
  attn_bias: torch.Tensor | None,
  softmax_scale: float,
  causal: int,
  autotune: int,
  autotune_mode_is_max: int,
  preprocess_d_chunk: int,
  return_attn_bias_grad: int,
  grad_kv_storage_dtype_code: int,
  original_nheads_kv: int,
  dropout_p: float,
  philox_seed: int,
  philox_offset: int,
  enable_tma: int,
  enable_ws: int,
  enable_persist_dkdv: int,
  enable_split_launch: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  del (
    softmax_scale,
    causal,
    autotune,
    autotune_mode_is_max,
    preprocess_d_chunk,
    original_nheads_kv,
    dropout_p,
    philox_seed,
    philox_offset,
    enable_tma,
    enable_ws,
    enable_persist_dkdv,
    enable_split_launch,
  )
  grad_kv_storage_dtype = _grad_kv_storage_dtype_from_code(
    grad_kv_storage_dtype_code
  )
  if attn_bias is not None and return_attn_bias_grad:
    grad_dtype = _attn_bias_grad_dtype(attn_bias, q, k)
    grad_attn_bias = torch.empty_like(attn_bias, dtype=grad_dtype)
  else:
    grad_attn_bias = q.new_empty(0)
  return (
    _triton_bwd_grad_tensor_like(q),
    _triton_bwd_grad_tensor_like(k, grad_kv_storage_dtype),
    _triton_bwd_grad_tensor_like(v, grad_kv_storage_dtype),
    grad_attn_bias,
  )


__all__ = ["_ffpa_attn_forward_triton", "_ffpa_attn_backward_triton"]

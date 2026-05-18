"""FFPA attention backward example.

Runs the same shape regimes as the forward example and compares FFPA against
PyTorch SDPA for backward correctness and backward runtime by default:

1. Self-Attention            -- Nq == Nkv, Nh_q == Nh_kv, aligned seqlen.
2. Cross / Decode Attention  -- Nq != Nkv (short query, long KV).
3. Grouped-Query Attention   -- Nh_q % Nh_kv == 0, Nh_kv < Nh_q (MQA => Nh_kv=1).
4. Causal Attention          -- causal=True, queries aligned to KV tail.
5. Dropout Attention         -- dropout_p > 0, compares against SDPA dropout.
6. Non-aligned Seqlen        -- N=8191 (not a multiple of Bc=64).

Usage::

    CUDA_VISIBLE_DEVICES=0 python examples/ffpa_attn_bwd.py
    CUDA_VISIBLE_DEVICES=0 python examples/ffpa_attn_bwd.py --backward-backend triton --autotune
    CUDA_VISIBLE_DEVICES=0 python examples/ffpa_attn_bwd.py --timing-mode full
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Any

import torch
import torch.nn.functional as F

from ffpa_attn import ffpa_attn_func
from attention_flops import attention_bwd_flops, format_tflops_short, tflops_from_ms

DEFAULT_WARMUP = 2
DEFAULT_ITERS = 10
MAX_MASK_GRAD_SEQLEN = 1024 * 16  # 16K, avoid OOM.
BACKWARD_RESULT = dict[str, Any]


def _parse_grad_v_dtype(arg: str) -> torch.dtype | None:
  """Parse the CLI grad-v-dtype option.

  :param arg: CLI value, ``"none"`` or ``"fp32"``.
  :return: ``None`` or ``torch.float32``.
  """
  if arg == "none":
    return None
  if arg == "fp32":
    return torch.float32
  raise ValueError(f"Unsupported grad-v-dtype={arg!r}; choose 'none' or 'fp32'.")


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="FFPA backward example and SDPA comparison.")
  parser.add_argument(
    "--backward-backend",
    "--backend",
    "--bwd",
    choices=["sdpa", "triton", "cutedsl"],
    default="triton",
    help=(
      "Backward backend passed to ffpa_attn_func. 'cutedsl' auto-pairs the "
      "forward to cutedsl and only runs SM90 + D=512 + bf16, with no "
      "attn_mask/dropout/non-aligned cases (auto-skipped)."
    ),
  )
  parser.add_argument("--B", type=int, default=1, help="Batch size.")
  parser.add_argument("--N", type=int, default=8192, help="Sequence length (non-aligned uses N-1).")
  parser.add_argument("--D", type=int, default=512, help="Head dimension.")
  parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP, help="Warmup iterations used for timing.")
  parser.add_argument("--iters", type=int, default=DEFAULT_ITERS, help="Measured iterations used for timing.")
  parser.add_argument("--dropout-p", type=float, default=0.1, help="Dropout probability for the dropout example case.")
  parser.add_argument("--seed", type=int, default=42, help="Random seed for input tensors.")
  parser.add_argument(
    "--norm",
    action="store_true",
    help="Enable pre-attention LayerNorm on q/k/v for both FFPA and SDPA paths.",
  )
  parser.add_argument(
    "--timing-mode",
    choices=["backward-only", "full"],
    default="backward-only",
    help="Whether to time only backward or end-to-end forward+backward.",
  )
  parser.add_argument(
    "--triton-autotune",
    "--autotune",
    "--tune",
    action="store_true",
    help="Enable Triton autotuning (only effective when --backward-backend=triton).",
  )
  parser.add_argument(
    "--triton-autotune-mode",
    "--autotune-mode",
    "--mode",
    choices=["fast", "max"],
    default="fast",
    help="Triton autotune search-space mode.",
  )
  parser.add_argument(
    "--grad-v-storage-dtype",
    "--grad-v-dtype",
    choices=["none", "fp32"],
    default="none",
    help="Optional Triton backward dV storage dtype forwarded to ffpa_attn_func.",
  )
  parser.add_argument(
    "--enable-tma",
    action="store_true",
    help="Compatibility alias for --enable-bwd-tma.",
  )
  parser.add_argument(
    "--enable-ws",
    action="store_true",
    help="Compatibility alias for --enable-bwd-ws.",
  )
  parser.add_argument(
    "--enable-bwd-tma",
    action="store_true",
    help="Enable the SM90+ Triton descriptor/TMA backward path when supported.",
  )
  parser.add_argument(
    "--enable-bwd-ws",
    action="store_true",
    help="Request warp-specialized SM90+ Triton backward configs when supported.",
  )
  args = parser.parse_args()
  if args.enable_tma:
    args.enable_bwd_tma = True
  if args.enable_ws:
    args.enable_bwd_ws = True
  if args.backward_backend == "cutedsl" and (
    args.triton_autotune or args.enable_bwd_tma or args.enable_bwd_ws or args.grad_v_storage_dtype != "none"
  ):
    print(
      "[warn] --backward-backend=cutedsl ignores --triton-autotune / "
      "--enable-bwd-tma / --enable-bwd-ws / --grad-v-storage-dtype."
    )
  return args


def _validate_timing_args(warmup: int, iters: int) -> None:
  """Validate benchmark timing loop counts.

  :param warmup: Warmup iterations used for timing.
  :param iters: Measured iterations used for timing.
  :raises ValueError: If ``warmup`` is negative or ``iters`` is not positive.
  """
  if warmup < 0:
    raise ValueError(f"warmup must be non-negative, got {warmup}")
  if iters <= 0:
    raise ValueError(f"iters must be positive, got {iters}")


def _make_sdpa_kwargs(causal: bool, nq: int, nkv: int):
  if causal and nq != nkv:
    kv_offset = nkv - nq
    row_idx = torch.arange(nq, device="cuda").view(-1, 1)
    col_idx = torch.arange(nkv, device="cuda").view(1, -1)
    return {"attn_mask": col_idx <= (row_idx + kv_offset)}
  if causal:
    return {"is_causal": True}
  return {}


def _dtype_tag(dtype: torch.dtype) -> str:
  """Return a short dtype tag.

  :param dtype: Torch dtype.
  :return: String form without the ``torch.`` prefix.
  """
  if dtype == torch.float16:
    return "fp16"
  if dtype == torch.bfloat16:
    return "bf16"
  return str(dtype).replace("torch.", "")


def _resolve_gqa_heads(num_heads: int) -> int:
  """Choose the KV head count used by the GQA example.

  :param num_heads: Query head count.
  :return: KV head count that still divides ``num_heads``.
  """
  if num_heads <= 1:
    return 1
  candidate = max(1, num_heads // 4)
  while candidate > 1 and num_heads % candidate != 0:
    candidate -= 1
  return candidate


def _resolve_non_aligned_heads(num_heads: int) -> int:
  """Choose the head count used by the non-aligned case.

  :param num_heads: Base head count.
  :return: Head count used by the non-aligned example.
  """
  if num_heads <= 8:
    return num_heads
  candidate = max(1, num_heads // 4)
  while candidate > 1 and num_heads % candidate != 0:
    candidate -= 1
  return candidate


def _mask_grad_status(
  dmask_ffpa: torch.Tensor | None,
  dmask_ref: torch.Tensor | None,
  mask_grad_skip_reason: str | None,
  attn_mask: torch.Tensor | None,
) -> tuple[float | None, str | None]:
  """Summarize mask-gradient comparison status.

  :param dmask_ffpa: FFPA mask gradient.
  :param dmask_ref: Reference mask gradient.
  :param mask_grad_skip_reason: Reason why mask gradient comparison was skipped.
  :param attn_mask: Original additive mask.
  :return: ``(error, status)`` pair.
  """
  if dmask_ffpa is not None and dmask_ref is not None:
    return (dmask_ffpa.float() - dmask_ref.float()).abs().max().item(), "compared"
  if attn_mask is not None and mask_grad_skip_reason is not None:
    return None, mask_grad_skip_reason
  return None, "no-grad"


def _max_abs_diff(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
  """Return max abs diff after promoting both tensors to fp32.

  :param lhs: First tensor.
  :param rhs: Second tensor.
  :return: Maximum absolute difference in fp32.
  """
  return (lhs.float() - rhs.float()).abs().max().item()


def _tensor_allclose(lhs: torch.Tensor, rhs: torch.Tensor, tol: float) -> bool:
  """Return whether two tensors are close after promoting both to fp32.

  :param lhs: First tensor.
  :param rhs: Second tensor.
  :param tol: Absolute and relative tolerance.
  :return: ``True`` when the promoted tensors satisfy ``torch.allclose``.
  """
  return torch.allclose(lhs.float(), rhs.float(), atol=tol, rtol=tol)


def _prepare_attn_mask(
  attn_mask: torch.Tensor | None,
  dtype: torch.dtype,
  compare_mask_grad: bool,
) -> torch.Tensor | None:
  """Prepare an additive attention mask for the target attention dtype.

  :param attn_mask: Optional additive mask.
  :param dtype: Attention dtype.
  :param compare_mask_grad: Whether mask gradients should be compared.
  :return: Mask ready for FFPA/SDPA execution.
  """
  if attn_mask is None:
    return None
  # Keep additive masks in their original dtype when we compare dMask. The
  # Triton wrapper intentionally allocates bf16-mask gradients in fp32 and only
  # casts back at the autograd boundary; forcing the example mask itself to q's
  # bf16 dtype would turn the reported dMask into a bf16 leaf-gradient test and
  # hide whether the internal fp32 grad_attn_bias path actually helps.
  if compare_mask_grad:
    return attn_mask
  prepared = attn_mask if compare_mask_grad else attn_mask.detach()
  if prepared.dtype != dtype:
    prepared = prepared.to(dtype)
    if compare_mask_grad and attn_mask.requires_grad:
      prepared = prepared.detach().requires_grad_(True)
  return prepared


def _maybe_norm_qkv(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  apply_norm: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Optionally apply per-tensor LayerNorm over the head dimension.

  :param q: Query tensor.
  :param k: Key tensor.
  :param v: Value tensor.
  :param apply_norm: Whether to normalize q/k/v before attention.
  :return: Normalized or original ``(q, k, v)``.
  """
  if not apply_norm:
    return q, k, v
  q = F.layer_norm(q, (q.size(-1), ))
  k = F.layer_norm(k, (k.size(-1), ))
  v = F.layer_norm(v, (v.size(-1), ))
  return q, k, v


def _format_backward_result(result: BACKWARD_RESULT) -> str:
  """Format one backward benchmark result for CLI output.

  :param result: Structured backward result.
  :return: Human-readable one-line summary.
  """
  if result["dmask_err"] is not None:
    dmask_msg = f"dMask_err={result['dmask_err']:.4e}  "
  elif result["dmask_status"] == "skipped-large-logical-mask":
    dmask_msg = "dMask_err=(SKIPPED large logical mask)  "
  elif result["dmask_status"] == "skipped-ffpa-forward-fallback":
    dmask_msg = "dMask_err=(SKIPPED FFPA fwd fallback)  "
  else:
    dmask_msg = "dMask_err=(NO Grad)  "
  return (
    f"[{result['case_name']:<14} {result['dtype']:<8}] "
    f"B={result['B']} Hq={result['Hq']} Hkv={result['Hkv']} "
    f"Nq={result['Nq']} Nkv={result['Nkv']} D={result['D']} "
    f"causal={int(result['causal'])} dropout_p={result['dropout_p']:g}  "
    f"dQ_err={result['dq_err']:.4e}  dK_err={result['dk_err']:.4e}  "
    f"dV_err={result['dv_err']:.4e}  {dmask_msg}"
    f"backend={result['backward_backend']}  "
    f"FFPA={result['ffpa_ms']:.2f} ms  SDPA={result['sdpa_ms']:.2f} ms  "
    f"TFLOPS={format_tflops_short(result['ffpa_tflops'])}/{format_tflops_short(result['sdpa_tflops'])}  "
    f"speedup={result['speedup']:.2f}x"
  )


def _make_broadcast_additive_attn_mask(nq: int, nkv: int, dtype: torch.dtype, seed: int) -> torch.Tensor:
  """Build a differentiable key-position additive attention bias."""
  del dtype
  torch.manual_seed(seed + 1)
  del nq
  return (torch.randn(1, 1, 1, nkv, dtype=torch.float32, device="cuda") * 0.25).requires_grad_(True)


def _is_key_position_bias(attn_mask: torch.Tensor | None, Nkv: int) -> bool:
  """Return whether ``attn_mask`` is a compact [1, 1, 1, Nkv] key bias."""
  return attn_mask is not None and tuple(attn_mask.shape) == (1, 1, 1, Nkv)


def _key_position_bias_grad_ref(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  scale: float,
  attn_mask: torch.Tensor,
  block_m: int = 128,
  block_n: int = 1024,
) -> torch.Tensor:
  """Compute a fp32 reference gradient for compact key-position additive bias."""
  with torch.no_grad():
    group_size = q.size(1) // k.size(1)
    k_ref = k.detach().repeat_interleave(group_size, dim=1) if group_size > 1 else k.detach()
    v_ref = v.detach().repeat_interleave(group_size, dim=1) if group_size > 1 else v.detach()
    q_f = q.detach().float()
    k_f = k_ref.float()
    key_value_sum = v_ref.float().sum(dim=-1)
    key_bias = attn_mask.detach().reshape(-1).float()
    Nq = q.size(2)
    Nkv = k_ref.size(2)
    grad = torch.zeros(Nkv, dtype=torch.float32, device=q.device)

    for m_start in range(0, Nq, block_m):
      q_block = q_f[:, :, m_start:m_start + block_m, :]
      lse = torch.full(q_block.shape[:-1], -torch.inf, dtype=torch.float32, device=q.device)
      for n_start in range(0, Nkv, block_n):
        scores = torch.matmul(q_block, k_f[:, :, n_start:n_start + block_n, :].transpose(-2, -1)) * scale
        scores = scores + key_bias[n_start:n_start + block_n].view(1, 1, 1, -1)
        lse = torch.logaddexp(lse, torch.logsumexp(scores, dim=-1))

      delta = torch.zeros_like(lse)
      for n_start in range(0, Nkv, block_n):
        scores = torch.matmul(q_block, k_f[:, :, n_start:n_start + block_n, :].transpose(-2, -1)) * scale
        scores = scores + key_bias[n_start:n_start + block_n].view(1, 1, 1, -1)
        prob = torch.exp(scores - lse[..., None])
        delta += (prob * key_value_sum[:, :, None, n_start:n_start + block_n]).sum(dim=-1)

      for n_start in range(0, Nkv, block_n):
        scores = torch.matmul(q_block, k_f[:, :, n_start:n_start + block_n, :].transpose(-2, -1)) * scale
        scores = scores + key_bias[n_start:n_start + block_n].view(1, 1, 1, -1)
        prob = torch.exp(scores - lse[..., None])
        d_bias = prob * (key_value_sum[:, :, None, n_start:n_start + block_n] - delta[..., None])
        grad[n_start:n_start + block_n] += d_bias.sum(dim=(0, 1, 2))

    return grad.view(1, 1, 1, Nkv)


def _time_fn(fn, *args, warmup: int = DEFAULT_WARMUP, iters: int = DEFAULT_ITERS, rng_seed: int | None = None) -> float:
  _validate_timing_args(warmup, iters)
  for _ in range(warmup):
    if rng_seed is not None:
      torch.manual_seed(rng_seed)
    fn(*args)
  torch.cuda.synchronize()
  t0 = time.perf_counter()
  for _ in range(iters):
    if rng_seed is not None:
      torch.manual_seed(rng_seed)
    fn(*args)
  torch.cuda.synchronize()
  return (time.perf_counter() - t0) * 1000.0 / iters  # ms


def _time_backward_only(
  fn,
  q,
  k,
  v,
  grad_out,
  warmup: int = DEFAULT_WARMUP,
  iters: int = DEFAULT_ITERS,
  rng_seed: int | None = None,
) -> float:
  _validate_timing_args(warmup, iters)
  start_event = torch.cuda.Event(enable_timing=True)
  end_event = torch.cuda.Event(enable_timing=True)

  for _ in range(warmup):
    q_i = q.detach().clone().requires_grad_(True)
    k_i = k.detach().clone().requires_grad_(True)
    v_i = v.detach().clone().requires_grad_(True)
    if rng_seed is not None:
      torch.manual_seed(rng_seed)
    out = fn(q_i, k_i, v_i)
    torch.cuda.synchronize()
    out.backward(grad_out)
  torch.cuda.synchronize()

  elapsed_ms = 0.0
  for _ in range(iters):
    q_i = q.detach().clone().requires_grad_(True)
    k_i = k.detach().clone().requires_grad_(True)
    v_i = v.detach().clone().requires_grad_(True)
    if rng_seed is not None:
      torch.manual_seed(rng_seed)
    out = fn(q_i, k_i, v_i)
    torch.cuda.synchronize()
    start_event.record()
    out.backward(grad_out)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_ms += start_event.elapsed_time(end_event)

  return elapsed_ms / iters


def _ffpa_forward(
  q_i: torch.Tensor,
  k_i: torch.Tensor,
  v_i: torch.Tensor,
  scale: float,
  backward_backend: str,
  triton_autotune: bool = False,
  triton_autotune_mode: str = "fast",
  causal: bool = False,
  attn_mask: torch.Tensor | None = None,
  dropout_p: float = 0.0,
  triton_backward_grad_v_storage_dtype: torch.dtype | None = None,
  enable_tma: bool = False,
  enable_ws: bool = False,
) -> torch.Tensor:
  return ffpa_attn_func(
    q_i,
    k_i,
    v_i,
    attn_mask=attn_mask,
    is_causal=causal,
    dropout_p=dropout_p,
    scale=scale,
    enable_gqa=q_i.size(1) != k_i.size(1),
    forward_backend="cutedsl" if backward_backend == "cutedsl" else "triton",
    backward_backend=backward_backend,
    triton_autotune=triton_autotune,
    triton_autotune_mode=triton_autotune_mode,
    triton_backward_grad_v_storage_dtype=triton_backward_grad_v_storage_dtype,
    enable_backward_tma=enable_tma,
    enable_backward_ws=enable_ws,
  )


def _sdpa_forward(
  q_i: torch.Tensor,
  k_i: torch.Tensor,
  v_i: torch.Tensor,
  scale: float,
  causal: bool = False,
  attn_mask: torch.Tensor | None = None,
  dropout_p: float = 0.0,
) -> torch.Tensor:
  group_size = q_i.size(1) // k_i.size(1)
  k_in = k_i.repeat_interleave(group_size, dim=1) if group_size > 1 else k_i
  v_in = v_i.repeat_interleave(group_size, dim=1) if group_size > 1 else v_i
  sdpa_mask = attn_mask.to(q_i.dtype) if attn_mask is not None and attn_mask.dtype != q_i.dtype else attn_mask
  kw = {"attn_mask": sdpa_mask} if sdpa_mask is not None else _make_sdpa_kwargs(causal, q_i.size(2), k_i.size(2))
  return F.scaled_dot_product_attention(q_i, k_in, v_in, scale=scale, dropout_p=dropout_p, **kw)


def _run_ffpa_backward(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  scale: float,
  backward_backend: str,
  triton_autotune: bool = False,
  triton_autotune_mode: str = "fast",
  causal: bool = False,
  attn_mask: torch.Tensor | None = None,
  dropout_p: float = 0.0,
  triton_backward_grad_v_storage_dtype: torch.dtype | None = None,
  enable_tma: bool = False,
  enable_ws: bool = False,
) -> None:
  if attn_mask is not None:
    attn_mask.grad = None
  q_i = q.detach().clone().requires_grad_(True)
  k_i = k.detach().clone().requires_grad_(True)
  v_i = v.detach().clone().requires_grad_(True)
  out = _ffpa_forward(
    q_i,
    k_i,
    v_i,
    scale,
    backward_backend=backward_backend,
    triton_autotune=triton_autotune,
    triton_autotune_mode=triton_autotune_mode,
    causal=causal,
    attn_mask=attn_mask,
    dropout_p=dropout_p,
    triton_backward_grad_v_storage_dtype=triton_backward_grad_v_storage_dtype,
    enable_backward_tma=enable_tma,
    enable_backward_ws=enable_ws,
  )
  out.sum().backward()


def _run_sdpa_backward(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  scale: float,
  causal: bool = False,
  attn_mask: torch.Tensor | None = None,
  dropout_p: float = 0.0,
) -> None:
  if attn_mask is not None:
    attn_mask.grad = None
  q_i = q.detach().clone().requires_grad_(True)
  k_i = k.detach().clone().requires_grad_(True)
  v_i = v.detach().clone().requires_grad_(True)

  out = _sdpa_forward(q_i, k_i, v_i, scale, causal=causal, attn_mask=attn_mask, dropout_p=dropout_p)
  out.sum().backward()


def _sdpa_ref_grads(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  scale: float,
  causal: bool = False,
  attn_mask: torch.Tensor | None = None,
  return_mask_grad: bool = True,
  dropout_p: float = 0.0,
  dropout_seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
  q_ref = q.detach().clone().requires_grad_(True)
  k_ref = k.detach().clone().requires_grad_(True)
  v_ref = v.detach().clone().requires_grad_(True)

  group_size = q_ref.size(1) // k_ref.size(1)
  k_in = k_ref.repeat_interleave(group_size, dim=1) if group_size > 1 else k_ref
  v_in = v_ref.repeat_interleave(group_size, dim=1) if group_size > 1 else v_ref
  attn_mask_ref = None
  if attn_mask is not None:
    attn_mask_ref = attn_mask.detach().clone()
    if attn_mask_ref.dtype != q_ref.dtype:
      attn_mask_ref = attn_mask_ref.to(q_ref.dtype)
      return_mask_grad = False
    attn_mask_ref.requires_grad_(return_mask_grad and attn_mask.requires_grad)
  kw = {
    "attn_mask": attn_mask_ref
  } if attn_mask_ref is not None else _make_sdpa_kwargs(causal, q_ref.size(2), k_ref.size(2))
  if dropout_seed is not None:
    torch.manual_seed(dropout_seed)
  out_ref = F.scaled_dot_product_attention(q_ref, k_in, v_in, scale=scale, dropout_p=dropout_p, **kw)
  out_ref.sum().backward()
  return q_ref.grad, k_ref.grad, v_ref.grad, attn_mask_ref.grad if attn_mask_ref is not None else None


def _mask_grad_skip_reason(
  B: int,
  Nh_q: int,
  Nq: int,
  Nkv: int,
  D: int,
  attn_mask: torch.Tensor | None,
) -> str | None:
  """Return why additive mask-gradient comparison should be skipped.

  The backward example keeps additive masks in fp32 only when the call stays on
  FFPA's large-D forward path, because that is the path whose internal fp32
  attention-bias gradient handling we want to validate. When the public API
  falls back to native SDPA forward, passing a fp32 additive bias alongside bf16
  or fp16 queries triggers a dtype mismatch in the native op.

  :return: Skip reason string, or ``None`` when mask-gradient comparison should
    proceed.
  """
  del B, Nh_q
  if attn_mask is None or not attn_mask.requires_grad:
    return "no-grad"
  if Nq > MAX_MASK_GRAD_SEQLEN or Nkv > MAX_MASK_GRAD_SEQLEN:
    return "skipped-large-logical-mask"
  if D <= 256 or D > 1024 or (8 <= Nq < 512) or Nkv < 512:
    return "skipped-ffpa-forward-fallback"
  return None


def _run_case(
  name: str,
  dtype: torch.dtype,
  backward_backend: str,
  triton_autotune: bool,
  triton_autotune_mode: str,
  seed: int,
  B: int,
  Nh_q: int,
  Nh_kv: int,
  Nq: int,
  Nkv: int,
  D: int,
  causal: bool = False,
  attn_mask: torch.Tensor | None = None,
  dropout_p: float = 0.0,
  timing_mode: str = "backward-only",
  triton_backward_grad_v_storage_dtype: torch.dtype | None = None,
  enable_tma: bool = False,
  enable_ws: bool = False,
  apply_norm: bool = False,
  warmup: int = DEFAULT_WARMUP,
  iters: int = DEFAULT_ITERS,
  print_result: bool = True,
) -> BACKWARD_RESULT:
  torch.manual_seed(seed)
  q = torch.randn(B, Nh_q, Nq, D, dtype=dtype, device="cuda")
  k = torch.randn(B, Nh_kv, Nkv, D, dtype=dtype, device="cuda")
  v = torch.randn(B, Nh_kv, Nkv, D, dtype=dtype, device="cuda")
  q, k, v = _maybe_norm_qkv(q, k, v, apply_norm)
  q = q.requires_grad_(True)
  k = k.requires_grad_(True)
  v = v.requires_grad_(True)
  scale = 1.0 / math.sqrt(D)
  dropout_seed = seed + 17
  mask_grad_skip_reason = _mask_grad_skip_reason(B, Nh_q, Nq, Nkv, D, attn_mask)
  compare_mask_grad = mask_grad_skip_reason is None
  active_attn_mask = _prepare_attn_mask(attn_mask, q.dtype, compare_mask_grad)

  torch.manual_seed(dropout_seed)
  out = ffpa_attn_func(
    q,
    k,
    v,
    attn_mask=active_attn_mask,
    is_causal=causal,
    dropout_p=dropout_p,
    scale=scale,
    enable_gqa=Nh_q != Nh_kv,
    forward_backend="cutedsl" if backward_backend == "cutedsl" else "triton",
    backward_backend=backward_backend,
    triton_autotune=triton_autotune,
    triton_autotune_mode=triton_autotune_mode,
    triton_backward_grad_v_storage_dtype=triton_backward_grad_v_storage_dtype,
    enable_backward_tma=enable_tma,
    enable_backward_ws=enable_ws,
  )
  out.sum().backward()

  dq_ffpa = q.grad.detach().clone()
  dk_ffpa = k.grad.detach().clone()
  dv_ffpa = v.grad.detach().clone()
  dmask_ffpa = active_attn_mask.grad.detach().clone(
  ) if active_attn_mask is not None and active_attn_mask.grad is not None else None
  dq_ref, dk_ref, dv_ref, dmask_ref = _sdpa_ref_grads(
    q,
    k,
    v,
    scale,
    causal=causal,
    attn_mask=active_attn_mask,
    return_mask_grad=compare_mask_grad,
    dropout_p=dropout_p,
    dropout_seed=dropout_seed,
  )
  if all([
    compare_mask_grad,
    dropout_p == 0.0,
    not causal,
    _is_key_position_bias(active_attn_mask, Nkv),
  ]):
    # PyTorch SDPA returns a bf16/fp16 mask gradient for compact [1, 1, 1, Nkv]
    # bias, but that broadcast-reduced leaf grad can differ materially from the
    # mathematically reduced reference in low precision. Use the dedicated
    # reference for this compact mask shape so dMask_err reflects FFPA kernel
    # accuracy rather than SDPA's compact-mask autograd behavior.
    dmask_ref = _key_position_bias_grad_ref(q, k, v, scale, active_attn_mask)
  if dmask_ffpa is not None and dmask_ref is not None and dmask_ref.dtype != dmask_ffpa.dtype:
    dmask_ref = dmask_ref.to(dmask_ffpa.dtype)

  if timing_mode == "backward-only":
    grad_out = torch.ones_like(q)
    ms_ffpa = _time_backward_only(
      lambda q_i, k_i, v_i: _ffpa_forward(
        q_i,
        k_i,
        v_i,
        scale,
        backward_backend,
        triton_autotune,
        triton_autotune_mode,
        causal,
        active_attn_mask,
        dropout_p,
        triton_backward_grad_v_storage_dtype,
        enable_tma,
        enable_ws,
      ),
      q,
      k,
      v,
      grad_out,
      warmup=warmup,
      iters=iters,
      rng_seed=dropout_seed if dropout_p > 0.0 else None,
    )
    ms_sdpa = _time_backward_only(
      lambda q_i, k_i, v_i: _sdpa_forward(
        q_i,
        k_i,
        v_i,
        scale,
        causal=causal,
        attn_mask=active_attn_mask,
        dropout_p=dropout_p,
      ),
      q,
      k,
      v,
      grad_out,
      warmup=warmup,
      iters=iters,
      rng_seed=dropout_seed if dropout_p > 0.0 else None,
    )
  else:
    ms_ffpa = _time_fn(
      _run_ffpa_backward,
      q,
      k,
      v,
      scale,
      backward_backend,
      triton_autotune,
      triton_autotune_mode,
      causal,
      active_attn_mask,
      dropout_p,
      triton_backward_grad_v_storage_dtype,
      enable_tma,
      enable_ws,
      warmup=warmup,
      iters=iters,
      rng_seed=dropout_seed if dropout_p > 0.0 else None,
    )
    ms_sdpa = _time_fn(
      _run_sdpa_backward,
      q,
      k,
      v,
      scale,
      causal,
      active_attn_mask,
      dropout_p,
      warmup=warmup,
      iters=iters,
      rng_seed=dropout_seed if dropout_p > 0.0 else None,
    )

  flop_count = attention_bwd_flops(B, Nh_q, Nq, Nkv, D, causal)

  tol = 7.5e-2 if dtype == torch.bfloat16 and causal else 5e-2 if dtype == torch.bfloat16 else 2e-2
  dq_err = _max_abs_diff(dq_ffpa, dq_ref)
  dk_err = _max_abs_diff(dk_ffpa, dk_ref)
  dv_err = _max_abs_diff(dv_ffpa, dv_ref)
  dmask_err, dmask_status = _mask_grad_status(dmask_ffpa, dmask_ref, mask_grad_skip_reason, attn_mask)
  allclose = (
    _tensor_allclose(dq_ffpa, dq_ref, tol) and _tensor_allclose(dk_ffpa, dk_ref, tol)
    and _tensor_allclose(dv_ffpa, dv_ref, tol)
  )
  if dmask_ffpa is not None and dmask_ref is not None:
    allclose = allclose and _tensor_allclose(dmask_ffpa, dmask_ref, tol)

  result: BACKWARD_RESULT = {
    "case_name": name,
    "dtype": _dtype_tag(dtype),
    "backward_backend": backward_backend,
    "timing_mode": timing_mode,
    "B": B,
    "Hq": Nh_q,
    "Hkv": Nh_kv,
    "Nq": Nq,
    "Nkv": Nkv,
    "D": D,
    "causal": causal,
    "dropout_p": dropout_p,
    "dq_err": dq_err,
    "dk_err": dk_err,
    "dv_err": dv_err,
    "dmask_err": dmask_err,
    "dmask_status": dmask_status,
    "compare_mask_grad": compare_mask_grad,
    "allclose": allclose,
    "tolerance": tol,
    "warmup": warmup,
    "iters": iters,
    "ffpa_ms": ms_ffpa,
    "sdpa_ms": ms_sdpa,
    "ffpa_tflops": tflops_from_ms(flop_count, ms_ffpa),
    "sdpa_tflops": tflops_from_ms(flop_count, ms_sdpa),
    "speedup": ms_sdpa / ms_ffpa,
  }
  if print_result:
    print(_format_backward_result(result))
  return result


def run_backward_examples(
  *,
  B: int = 1,
  H: int = 32,
  N: int = 8192,
  D: int = 512,
  dropout_p: float = 0.1,
  seed: int = 42,
  apply_norm: bool = False,
  backward_backend: str = "triton",
  timing_mode: str = "backward-only",
  triton_autotune: bool = False,
  triton_autotune_mode: str = "fast",
  triton_backward_grad_v_storage_dtype: torch.dtype | None = None,
  enable_tma: bool = False,
  enable_ws: bool = False,
  warmup: int = DEFAULT_WARMUP,
  iters: int = DEFAULT_ITERS,
  print_results: bool = True,
  tasks: set[str] | None = None,
  dtypes: tuple[torch.dtype, ...] = (torch.float16, torch.bfloat16),
) -> list[BACKWARD_RESULT]:
  """Run the canonical backward benchmark cases.

  :param B: Batch size.
  :param H: Base query head count used by the examples.
  :param N: Base sequence length.
  :param D: Head dimension.
  :param dropout_p: Dropout probability for the dropout case.
  :param seed: RNG seed.
  :param apply_norm: Whether to normalize q/k/v before attention.
  :param backward_backend: Backward backend passed to ``ffpa_attn_func``.
  :param timing_mode: Benchmark timing mode.
  :param triton_autotune: Whether to enable Triton runtime autotune.
  :param triton_autotune_mode: Triton autotune mode.
  :param triton_backward_grad_v_storage_dtype: Optional Triton backward dV
    storage dtype forwarded to ``ffpa_attn_func``.
  :param enable_tma: Whether to enable the SM90+ Triton descriptor/TMA
    backward path when supported.
  :param enable_ws: Whether to request warp-specialized SM90+ Triton configs
    when supported.
  :param warmup: Warmup iterations used for timing.
  :param iters: Measured iterations used for timing.
  :param print_results: Whether to print each case result.
  :param tasks: Optional case-name filter. ``None`` runs all cases.
  :param dtypes: Activation dtypes iterated for each case.
  :return: One structured result per executed case and dtype.
  """
  _validate_timing_args(warmup, iters)
  results: list[BACKWARD_RESULT] = []
  gqa_heads = _resolve_gqa_heads(H)
  non_aligned_heads = _resolve_non_aligned_heads(H)

  if backward_backend == "cutedsl":
    dtypes = tuple(dt for dt in dtypes if dt == torch.bfloat16)
    if not dtypes:
      raise ValueError(
        "cutedsl backward requires torch.bfloat16; pass dtypes=(torch.bfloat16,) "
        "or remove fp16 from the dtypes tuple."
      )

  print(
    f"\nRunning FFPA backward examples with backward_backend={backward_backend}, "
    f"apply_norm={apply_norm}, "
    f"triton_autotune={triton_autotune}, "
    f"triton_autotune_mode={triton_autotune_mode}, "
    f"triton_backward_grad_v_storage_dtype={triton_backward_grad_v_storage_dtype}, "
    f"enable_bwd_tma={enable_tma}, enable_bwd_ws={enable_ws}, "
    f"timing_mode={timing_mode}, tasks={sorted(tasks) if tasks is not None else 'full'}, "
    f"warmup={warmup}, iters={iters}"
  )
  if backward_backend == "cutedsl":
    print(
      "[cutedsl] backend constraints in effect: D=512 + bf16 + no mask/dropout; "
      "forward auto-paired to cutedsl; triton-* / enable-bwd-tma|ws / grad-v-dtype are ignored."
    )

  mask_dropout_supported = backward_backend != "cutedsl"

  for dtype in dtypes:
    mask_n = max(N, 512)
    case_specs: list[dict[str, Any]] = [
      {
        "name": "self-attn",
        "Nh_q": H,
        "Nh_kv": H,
        "Nq": N,
        "Nkv": N
      },
      {
        "name": "cross-attn",
        "Nh_q": H,
        "Nh_kv": H,
        "Nq": 1024,
        "Nkv": N
      },
      {
        "name": "decode-attn",
        "Nh_q": H,
        "Nh_kv": H,
        "Nq": 1,
        "Nkv": N
      },
      {
        "name": "gqa",
        "Nh_q": H,
        "Nh_kv": gqa_heads,
        "Nq": N,
        "Nkv": N
      },
      {
        "name": "causal",
        "Nh_q": H,
        "Nh_kv": H,
        "Nq": N,
        "Nkv": N,
        "causal": True
      },
    ]
    if mask_dropout_supported:
      case_specs.extend([
        {
          "name": "attn-mask",
          "Nh_q": H,
          "Nh_kv": H,
          "Nq": mask_n,
          "Nkv": mask_n,
          "attn_mask": _make_broadcast_additive_attn_mask(mask_n, mask_n, dtype, seed),
        },
        {
          "name": "dropout",
          "Nh_q": H,
          "Nh_kv": H,
          "Nq": N,
          "Nkv": N,
          "dropout_p": dropout_p,
        },
      ])
    case_specs.append({
      "name": "non-aligned",
      "Nh_q": non_aligned_heads,
      "Nh_kv": non_aligned_heads,
      "Nq": N - 1 if N > 1 else N,
      "Nkv": N - 1 if N > 1 else N,
    })
    if tasks is not None:
      case_specs = [case for case in case_specs if case["name"] in tasks]

    for case in case_specs:
      results.append(
        _run_case(
          case["name"],
          dtype,
          backward_backend,
          triton_autotune,
          triton_autotune_mode,
          seed=seed,
          B=B,
          Nh_q=case["Nh_q"],
          Nh_kv=case["Nh_kv"],
          Nq=case["Nq"],
          Nkv=case["Nkv"],
          D=D,
          causal=case.get("causal", False),
          attn_mask=case.get("attn_mask"),
          dropout_p=case.get("dropout_p", 0.0),
          timing_mode=timing_mode,
          triton_backward_grad_v_storage_dtype=triton_backward_grad_v_storage_dtype,
          enable_tma=enable_tma,
          enable_ws=enable_ws,
          apply_norm=apply_norm,
          warmup=warmup,
          iters=iters,
          print_result=print_results,
        )
      )

  return results


def main() -> None:
  args = _parse_args()
  print(args)

  if not torch.cuda.is_available():
    raise SystemExit("CUDA is required to run this example.")
  grad_v_dtype = _parse_grad_v_dtype(args.grad_v_storage_dtype)
  run_backward_examples(
    B=args.B,
    N=args.N,
    D=args.D,
    dropout_p=args.dropout_p,
    seed=args.seed,
    apply_norm=args.norm,
    backward_backend=args.backward_backend,
    timing_mode=args.timing_mode,
    triton_autotune=args.triton_autotune,
    triton_autotune_mode=args.triton_autotune_mode,
    triton_backward_grad_v_storage_dtype=grad_v_dtype,
    enable_tma=args.enable_bwd_tma,
    enable_ws=args.enable_bwd_ws,
    warmup=args.warmup,
    iters=args.iters,
    print_results=True,
  )


if __name__ == "__main__":
  main()

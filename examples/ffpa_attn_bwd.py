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

import torch
import torch.nn.functional as F

from ffpa_attn import ffpa_attn_func

WARMUP, ITERS = 2, 10
MAX_MASK_GRAD_SEQLEN = 1024 * 16  # 16K, avoid OOM.


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="FFPA backward example and SDPA comparison.")
  parser.add_argument(
    "--backward-backend",
    "--backend",
    "--bwd",
    choices=["sdpa", "triton"],
    default="triton",
    help="Backward backend passed to ffpa_attn_func.",
  )
  parser.add_argument("--B", type=int, default=1, help="Batch size.")
  parser.add_argument("--N", type=int, default=8192, help="Sequence length (non-aligned uses N-1).")
  parser.add_argument("--D", type=int, default=512, help="Head dimension.")
  parser.add_argument("--dropout-p", type=float, default=0.1, help="Dropout probability for the dropout example case.")
  parser.add_argument("--seed", type=int, default=42, help="Random seed for input tensors.")
  parser.add_argument(
    "--timing-mode",
    choices=["backward-only", "full"],
    default="backward-only",
    help="Whether to time only backward or end-to-end forward+backward.",
  )
  parser.add_argument(
    "--triton-backward-autotune",
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
  return parser.parse_args()


def _make_sdpa_kwargs(causal: bool, nq: int, nkv: int):
  if causal and nq != nkv:
    kv_offset = nkv - nq
    row_idx = torch.arange(nq, device="cuda").view(-1, 1)
    col_idx = torch.arange(nkv, device="cuda").view(1, -1)
    return {"attn_mask": col_idx <= (row_idx + kv_offset)}
  if causal:
    return {"is_causal": True}
  return {}


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


def _time_fn(fn, *args, warmup: int = WARMUP, iters: int = ITERS, rng_seed: int | None = None) -> float:
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
  warmup: int = WARMUP,
  iters: int = ITERS,
  rng_seed: int | None = None,
) -> float:
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
  triton_backward_autotune: bool = False,
  triton_autotune_mode: str = "fast",
  causal: bool = False,
  attn_mask: torch.Tensor | None = None,
  dropout_p: float = 0.0,
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
    backward_backend=backward_backend,
    triton_backward_autotune=triton_backward_autotune,
    triton_autotune_mode=triton_autotune_mode,
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
  triton_backward_autotune: bool = False,
  triton_autotune_mode: str = "fast",
  causal: bool = False,
  attn_mask: torch.Tensor | None = None,
  dropout_p: float = 0.0,
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
    triton_backward_autotune=triton_backward_autotune,
    triton_autotune_mode=triton_autotune_mode,
    causal=causal,
    attn_mask=attn_mask,
    dropout_p=dropout_p,
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


def _should_compare_mask_grad(B: int, Nh_q: int, Nq: int, Nkv: int, attn_mask: torch.Tensor | None) -> bool:
  """Return whether the example should ask SDPA for additive-mask gradients."""
  del B, Nh_q
  if attn_mask is None or not attn_mask.requires_grad:
    return False
  return Nq <= MAX_MASK_GRAD_SEQLEN and Nkv <= MAX_MASK_GRAD_SEQLEN


def _run_case(
  name: str,
  dtype: torch.dtype,
  backward_backend: str,
  triton_backward_autotune: bool,
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
) -> None:
  torch.manual_seed(seed)
  q = torch.randn(B, Nh_q, Nq, D, dtype=dtype, device="cuda", requires_grad=True)
  k = torch.randn(B, Nh_kv, Nkv, D, dtype=dtype, device="cuda", requires_grad=True)
  v = torch.randn(B, Nh_kv, Nkv, D, dtype=dtype, device="cuda", requires_grad=True)
  scale = 1.0 / math.sqrt(D)
  dropout_seed = seed + 17
  compare_mask_grad = _should_compare_mask_grad(B, Nh_q, Nq, Nkv, attn_mask)
  active_attn_mask = attn_mask if compare_mask_grad else attn_mask.detach() if attn_mask is not None else None

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
    backward_backend=backward_backend,
    triton_backward_autotune=triton_backward_autotune,
    triton_autotune_mode=triton_autotune_mode,
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
    dmask_ref is None,
    dropout_p == 0.0,
    not causal,
    _is_key_position_bias(active_attn_mask, Nkv),
  ]):
    dmask_ref = _key_position_bias_grad_ref(q, k, v, scale, active_attn_mask)

  if timing_mode == "backward-only":
    grad_out = torch.ones_like(q)
    ms_ffpa = _time_backward_only(
      lambda q_i, k_i, v_i: _ffpa_forward(
        q_i,
        k_i,
        v_i,
        scale,
        backward_backend,
        triton_backward_autotune,
        triton_autotune_mode,
        causal,
        active_attn_mask,
        dropout_p,
      ),
      q,
      k,
      v,
      grad_out,
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
      triton_backward_autotune,
      triton_autotune_mode,
      causal,
      active_attn_mask,
      dropout_p,
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
      rng_seed=dropout_seed if dropout_p > 0.0 else None,
    )

  dt_tag = str(dtype).replace("torch.", "")
  dmask_msg = "dMask_err=(NO Grad)  "
  if dmask_ffpa is not None and dmask_ref is not None:
    dmask_msg = f"dMask_err={(dmask_ffpa - dmask_ref).abs().max().item():.4e}  "
  elif attn_mask is not None and not compare_mask_grad:
    dmask_msg = "dMask_err=(SKIPPED large logical mask)  "
  print(
    f"[{name:<14} {dt_tag:<8}] "
    f"B={B} Hq={Nh_q} Hkv={Nh_kv} Nq={Nq} Nkv={Nkv} D={D} causal={int(causal)} dropout_p={dropout_p:g}  "
    f"dQ_err={(dq_ffpa - dq_ref).abs().max().item():.4e}  "
    f"dK_err={(dk_ffpa - dk_ref).abs().max().item():.4e}  "
    f"dV_err={(dv_ffpa - dv_ref).abs().max().item():.4e}  "
    f"{dmask_msg}"
    f"backend={backward_backend}  "
    f"FFPA={ms_ffpa:.2f} ms  SDPA={ms_sdpa:.2f} ms  speedup={ms_sdpa / ms_ffpa:.2f}x"
  )


def main() -> None:
  args = _parse_args()
  print(args)
  N, D = args.N, args.D

  if not torch.cuda.is_available():
    raise SystemExit("CUDA is required to run this example.")

  for dtype in (torch.float16, torch.bfloat16):
    _run_case(
      "self-attn",
      dtype,
      args.backward_backend,
      args.triton_backward_autotune,
      args.triton_autotune_mode,
      seed=args.seed,
      B=args.B,
      Nh_q=32,
      Nh_kv=32,
      Nq=N,
      Nkv=N,
      D=D,
      timing_mode=args.timing_mode,
    )
    _run_case(
      "cross-attn",
      dtype,
      args.backward_backend,
      args.triton_backward_autotune,
      args.triton_autotune_mode,
      seed=args.seed,
      B=args.B,
      Nh_q=32,
      Nh_kv=32,
      Nq=1024,
      Nkv=N,
      D=D,
      timing_mode=args.timing_mode,
    )
    _run_case(
      "decode-attn",
      dtype,
      args.backward_backend,
      args.triton_backward_autotune,
      args.triton_autotune_mode,
      seed=args.seed,
      B=args.B,
      Nh_q=32,
      Nh_kv=32,
      Nq=1,
      Nkv=N,
      D=D,
      timing_mode=args.timing_mode,
    )
    _run_case(
      "gqa",
      dtype,
      args.backward_backend,
      args.triton_backward_autotune,
      args.triton_autotune_mode,
      seed=args.seed,
      B=args.B,
      Nh_q=32,
      Nh_kv=8,
      Nq=N,
      Nkv=N,
      D=D,
      timing_mode=args.timing_mode,
    )
    _run_case(
      "causal",
      dtype,
      args.backward_backend,
      args.triton_backward_autotune,
      args.triton_autotune_mode,
      seed=args.seed,
      B=args.B,
      Nh_q=32,
      Nh_kv=32,
      Nq=N,
      Nkv=N,
      D=D,
      causal=True,
      timing_mode=args.timing_mode,
    )
    mask_n = max(N, 512)
    _run_case(
      "attn-mask",
      dtype,
      args.backward_backend,
      args.triton_backward_autotune,
      args.triton_autotune_mode,
      seed=args.seed,
      B=args.B,
      Nh_q=32,
      Nh_kv=32,
      Nq=mask_n,
      Nkv=mask_n,
      D=D,
      attn_mask=_make_broadcast_additive_attn_mask(mask_n, mask_n, dtype, args.seed),
      timing_mode=args.timing_mode,
    )
    _run_case(
      "dropout",
      dtype,
      args.backward_backend,
      args.triton_backward_autotune,
      args.triton_autotune_mode,
      seed=args.seed,
      B=args.B,
      Nh_q=32,
      Nh_kv=32,
      Nq=N,
      Nkv=N,
      D=D,
      dropout_p=args.dropout_p,
      timing_mode=args.timing_mode,
    )
    _run_case(
      "non-aligned",
      dtype,
      args.backward_backend,
      args.triton_backward_autotune,
      args.triton_autotune_mode,
      seed=args.seed,
      B=args.B,
      Nh_q=8,
      Nh_kv=8,
      Nq=N - 1 if N > 1 else N,  # avoid zero-dim
      Nkv=N - 1 if N > 1 else N,  # avoid zero-dim
      D=D,
      timing_mode=args.timing_mode,
    )


if __name__ == "__main__":
  main()

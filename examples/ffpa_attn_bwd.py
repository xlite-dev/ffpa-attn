"""FFPA attention backward example.

Runs the same shape regimes as the forward example and compares FFPA against
PyTorch SDPA for backward correctness and backward runtime by default:

1. Self-Attention            -- Nq == Nkv, Nh_q == Nh_kv, aligned seqlen.
2. Cross / Decode Attention  -- Nq != Nkv (short query, long KV).
3. Grouped-Query Attention   -- Nh_q % Nh_kv == 0, Nh_kv < Nh_q (MQA => Nh_kv=1).
4. Causal Attention          -- causal=True, queries aligned to KV tail.
5. Non-aligned Seqlen        -- N=8191 (not a multiple of Bc=64).

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


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="FFPA backward example and SDPA comparison.")
  parser.add_argument(
    "--backward-backend",
    "--backend",
    "--bwd",
    choices=["sdpa", "cuda", "triton"],
    default="triton",
    help="Backward backend passed to ffpa_attn_func.",
  )
  parser.add_argument("--B", type=int, default=1, help="Batch size.")
  parser.add_argument("--N", type=int, default=8192, help="Sequence length (non-aligned uses N-1).")
  parser.add_argument("--D", type=int, default=512, help="Head dimension.")
  parser.add_argument("--seed", type=int, default=0, help="Random seed for input tensors.")
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


def _time_fn(fn, *args, warmup: int = WARMUP, iters: int = ITERS) -> float:
  for _ in range(warmup):
    fn(*args)
  torch.cuda.synchronize()
  t0 = time.perf_counter()
  for _ in range(iters):
    fn(*args)
  torch.cuda.synchronize()
  return (time.perf_counter() - t0) * 1000.0 / iters  # ms


def _time_backward_only(fn, q, k, v, grad_out, warmup: int = WARMUP, iters: int = ITERS) -> float:
  start_event = torch.cuda.Event(enable_timing=True)
  end_event = torch.cuda.Event(enable_timing=True)

  for _ in range(warmup):
    q_i = q.detach().clone().requires_grad_(True)
    k_i = k.detach().clone().requires_grad_(True)
    v_i = v.detach().clone().requires_grad_(True)
    out = fn(q_i, k_i, v_i)
    torch.cuda.synchronize()
    out.backward(grad_out)
  torch.cuda.synchronize()

  elapsed_ms = 0.0
  for _ in range(iters):
    q_i = q.detach().clone().requires_grad_(True)
    k_i = k.detach().clone().requires_grad_(True)
    v_i = v.detach().clone().requires_grad_(True)
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
) -> torch.Tensor:
  return ffpa_attn_func(
    q_i,
    k_i,
    v_i,
    is_causal=causal,
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
) -> torch.Tensor:
  group_size = q_i.size(1) // k_i.size(1)
  k_in = k_i.repeat_interleave(group_size, dim=1) if group_size > 1 else k_i
  v_in = v_i.repeat_interleave(group_size, dim=1) if group_size > 1 else v_i
  kw = _make_sdpa_kwargs(causal, q_i.size(2), k_i.size(2))
  return F.scaled_dot_product_attention(q_i, k_in, v_in, scale=scale, **kw)


def _run_ffpa_backward(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  scale: float,
  backward_backend: str,
  triton_backward_autotune: bool = False,
  triton_autotune_mode: str = "fast",
  causal: bool = False,
) -> None:
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
  )
  out.sum().backward()


def _run_sdpa_backward(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  scale: float,
  causal: bool = False,
) -> None:
  q_i = q.detach().clone().requires_grad_(True)
  k_i = k.detach().clone().requires_grad_(True)
  v_i = v.detach().clone().requires_grad_(True)

  out = _sdpa_forward(q_i, k_i, v_i, scale, causal=causal)
  out.sum().backward()


def _sdpa_ref_grads(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  scale: float,
  causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  q_ref = q.detach().clone().requires_grad_(True)
  k_ref = k.detach().clone().requires_grad_(True)
  v_ref = v.detach().clone().requires_grad_(True)

  group_size = q_ref.size(1) // k_ref.size(1)
  k_in = k_ref.repeat_interleave(group_size, dim=1) if group_size > 1 else k_ref
  v_in = v_ref.repeat_interleave(group_size, dim=1) if group_size > 1 else v_ref
  kw = _make_sdpa_kwargs(causal, q_ref.size(2), k_ref.size(2))
  out_ref = F.scaled_dot_product_attention(q_ref, k_in, v_in, scale=scale, **kw)
  out_ref.sum().backward()
  return q_ref.grad, k_ref.grad, v_ref.grad


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
  timing_mode: str = "backward-only",
) -> None:
  torch.manual_seed(seed)
  q = torch.randn(B, Nh_q, Nq, D, dtype=dtype, device="cuda", requires_grad=True)
  k = torch.randn(B, Nh_kv, Nkv, D, dtype=dtype, device="cuda", requires_grad=True)
  v = torch.randn(B, Nh_kv, Nkv, D, dtype=dtype, device="cuda", requires_grad=True)
  scale = 1.0 / math.sqrt(D)

  out = ffpa_attn_func(
    q,
    k,
    v,
    is_causal=causal,
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
  dq_ref, dk_ref, dv_ref = _sdpa_ref_grads(q, k, v, scale, causal=causal)

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
      ),
      q,
      k,
      v,
      grad_out,
    )
    ms_sdpa = _time_backward_only(
      lambda q_i, k_i, v_i: _sdpa_forward(q_i, k_i, v_i, scale, causal=causal),
      q,
      k,
      v,
      grad_out,
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
    )
    ms_sdpa = _time_fn(_run_sdpa_backward, q, k, v, scale, causal)

  dt_tag = str(dtype).replace("torch.", "")
  print(
    f"[{name:<14} {dt_tag:<8}] "
    f"B={B} Hq={Nh_q} Hkv={Nh_kv} Nq={Nq} Nkv={Nkv} D={D} causal={int(causal)}  "
    f"dQ_err={(dq_ffpa - dq_ref).abs().max().item():.4e}  "
    f"dK_err={(dk_ffpa - dk_ref).abs().max().item():.4e}  "
    f"dV_err={(dv_ffpa - dv_ref).abs().max().item():.4e}  "
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

"""FFPA attention backward example.

Runs the same shape regimes as the forward example and compares FFPA against
PyTorch SDPA for backward correctness and end-to-end forward+backward runtime:

1. Self-Attention            -- Nq == Nkv, Nh_q == Nh_kv, aligned seqlen.
2. Cross / Decode Attention  -- Nq != Nkv (short query, long KV).
3. Grouped-Query Attention   -- Nh_q % Nh_kv == 0, Nh_kv < Nh_q (MQA => Nh_kv=1).
4. Causal Attention          -- causal=True, queries aligned to KV tail.
5. Non-aligned Seqlen        -- N=8191 (not a multiple of Bc=64).

Usage::

    CUDA_VISIBLE_DEVICES=0 python examples/ffpa_attn_bwd.py
"""

from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn.functional as F

from ffpa_attn import ffpa_attn_func

WARMUP, ITERS = 2, 5
D = 512


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="FFPA backward example and SDPA comparison.")
  parser.add_argument(
    "--backward-backend",
    choices=["sdpa", "cuda", "triton"],
    default="triton",
    help="Backward backend passed to ffpa_attn_func.",
  )
  parser.add_argument(
    "--triton-backward-autotune",
    "--autotune",
    action="store_true",
    help="Enable Triton autotuning (only effective when --backward-backend=triton).",
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


def _run_ffpa_backward(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  scale: float,
  backward_backend: str,
  triton_backward_autotune: bool = False,
  causal: bool = False,
) -> None:
  q_i = q.detach().clone().requires_grad_(True)
  k_i = k.detach().clone().requires_grad_(True)
  v_i = v.detach().clone().requires_grad_(True)
  out = ffpa_attn_func(
    q_i,
    k_i,
    v_i,
    causal=causal,
    softmax_scale=scale,
    backward_backend=backward_backend,
    triton_backward_autotune=triton_backward_autotune,
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

  group_size = q_i.size(1) // k_i.size(1)
  k_in = k_i.repeat_interleave(group_size, dim=1) if group_size > 1 else k_i
  v_in = v_i.repeat_interleave(group_size, dim=1) if group_size > 1 else v_i
  kw = _make_sdpa_kwargs(causal, q_i.size(2), k_i.size(2))
  out = F.scaled_dot_product_attention(q_i, k_in, v_in, scale=scale, **kw)
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
  B: int,
  Nh_q: int,
  Nh_kv: int,
  Nq: int,
  Nkv: int,
  causal: bool = False,
) -> None:
  torch.manual_seed(0)
  q = torch.randn(B, Nh_q, Nq, D, dtype=dtype, device="cuda", requires_grad=True)
  k = torch.randn(B, Nh_kv, Nkv, D, dtype=dtype, device="cuda", requires_grad=True)
  v = torch.randn(B, Nh_kv, Nkv, D, dtype=dtype, device="cuda", requires_grad=True)
  scale = 1.0 / math.sqrt(D)

  out = ffpa_attn_func(
    q,
    k,
    v,
    causal=causal,
    softmax_scale=scale,
    backward_backend=backward_backend,
    triton_backward_autotune=triton_backward_autotune,
  )
  out.sum().backward()

  dq_ffpa = q.grad.detach().clone()
  dk_ffpa = k.grad.detach().clone()
  dv_ffpa = v.grad.detach().clone()
  dq_ref, dk_ref, dv_ref = _sdpa_ref_grads(q, k, v, scale, causal=causal)

  ms_ffpa = _time_fn(
    _run_ffpa_backward,
    q,
    k,
    v,
    scale,
    backward_backend,
    triton_backward_autotune,
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

  if not torch.cuda.is_available():
    raise SystemExit("CUDA is required to run this example.")

  for dtype in (torch.float16, torch.bfloat16):
    _run_case(
      "self-attn",
      dtype,
      args.backward_backend,
      args.triton_backward_autotune,
      B=1,
      Nh_q=32,
      Nh_kv=32,
      Nq=8192,
      Nkv=8192
    )
    _run_case(
      "cross-attn",
      dtype,
      args.backward_backend,
      args.triton_backward_autotune,
      B=1,
      Nh_q=32,
      Nh_kv=32,
      Nq=1024,
      Nkv=8192
    )
    _run_case(
      "gqa",
      dtype,
      args.backward_backend,
      args.triton_backward_autotune,
      B=1,
      Nh_q=32,
      Nh_kv=8,
      Nq=8192,
      Nkv=8192,
    )
    _run_case(
      "causal",
      dtype,
      args.backward_backend,
      args.triton_backward_autotune,
      B=1,
      Nh_q=32,
      Nh_kv=32,
      Nq=8192,
      Nkv=8192,
      causal=True
    )
    _run_case(
      "non-aligned",
      dtype,
      args.backward_backend,
      args.triton_backward_autotune,
      B=1,
      Nh_q=8,
      Nh_kv=8,
      Nq=8191,
      Nkv=8191
    )


if __name__ == "__main__":
  main()

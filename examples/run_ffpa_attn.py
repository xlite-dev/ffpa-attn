"""Minimal FFPA attention example.

Runs ``ffpa_attn_func`` on a single (B, H, N, D) = (1, 32, 8192, 512) shape
and compares the output against ``torch.nn.functional.scaled_dot_product_attention``
for both fp16 (MMA acc=f32) and bf16 (MMA acc=f32) activations.

Usage::

    CUDA_VISIBLE_DEVICES=0 python examples/run_ffpa_attn.py
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

from ffpa_attn import ffpa_attn_func

B, H, N, D = 1, 32, 8192, 512
STAGES = 2
WARMUP, ITERS = 2, 5


def _sdpa_ref(q, k, v):
  return F.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(q.size(-1)))


def _time_fn(fn, *args, warmup: int = WARMUP, iters: int = ITERS) -> float:
  for _ in range(warmup):
    fn(*args)
  torch.cuda.synchronize()
  t0 = time.perf_counter()
  for _ in range(iters):
    fn(*args)
  torch.cuda.synchronize()
  return (time.perf_counter() - t0) * 1000.0 / iters  # ms


def run_one(dtype: torch.dtype, acc: str, B: int = B, H: int = H, N: int = N, D: int = D) -> None:
  torch.manual_seed(0)
  q = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
  k = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
  v = torch.randn(B, H, N, D, dtype=dtype, device="cuda")

  out_ffpa = ffpa_attn_func(q, k, v, stages=STAGES, acc=acc)
  out_sdpa = _sdpa_ref(q, k, v)

  diff = (out_ffpa.float() - out_sdpa.float()).abs()
  tol_atol = 5e-2 if dtype == torch.bfloat16 else 2e-2
  all_close = torch.allclose(out_ffpa, out_sdpa, atol=tol_atol, rtol=tol_atol)

  ms_ffpa = _time_fn(lambda q, k, v: ffpa_attn_func(q, k, v, stages=STAGES, acc=acc), q, k, v)
  ms_sdpa = _time_fn(_sdpa_ref, q, k, v)

  tag = f"{str(dtype).replace('torch.', ''):<8} acc={acc}"
  print(
    f"[{tag}] B={B} H={H} N={N} D={D}  "
    f"max|diff|={diff.max().item():.4f}  mean|diff|={diff.mean().item():.5f}  "
    f"allclose(atol={tol_atol})={all_close}  "
    f"FFPA={ms_ffpa:.2f} ms  SDPA={ms_sdpa:.2f} ms  speedup={ms_sdpa / ms_ffpa:.2f}x"
  )


def main() -> None:
  if not torch.cuda.is_available():
    raise SystemExit("CUDA is required to run this example.")
  # 1) aligned seqlen (multiple of tile Bc=64): fast path, no boundary masking.
  run_one(torch.float16, acc="f32")
  run_one(torch.bfloat16, acc="f32")
  # 2) non-multiple-of-8 seqlen: exercises cp.async zero-fill + softmax -inf
  # mask + per-row store predicate on the tail tile. Reduce H to keep memory
  # footprint modest under D=512 + large N.
  run_one(torch.float16, acc="f32", H=8, N=8191)
  run_one(torch.bfloat16, acc="f32", H=8, N=8191)


if __name__ == "__main__":
  main()

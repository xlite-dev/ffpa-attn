import argparse
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ffpa_attn import ffpa_attn_func  # noqa: E402


def parse_args():
  parser = argparse.ArgumentParser(description="Benchmark FFPA native split-D backward against SDPA backward.")
  parser.add_argument("--B", type=int, default=1)
  parser.add_argument("--H", type=int, default=2)
  parser.add_argument("--N", type=int, default=128)
  parser.add_argument("--D", type=int, default=512)
  parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
  parser.add_argument("--causal", action="store_true")
  parser.add_argument("--stages", type=int, default=1)
  parser.add_argument("--warmup", type=int, default=5)
  parser.add_argument("--iters", type=int, default=20)
  parser.add_argument("--seed", type=int, default=0)
  return parser.parse_args()


def make_inputs(args):
  torch.manual_seed(args.seed)
  dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
  q = torch.randn(args.B, args.H, args.N, args.D, device="cuda", dtype=dtype)
  k = torch.randn(args.B, args.H, args.N, args.D, device="cuda", dtype=dtype)
  v = torch.randn(args.B, args.H, args.N, args.D, device="cuda", dtype=dtype)
  dO = torch.randn_like(q)
  return q, k, v, dO


def time_backward(name, fn, q, k, v, dO, warmup, iters):
  for _ in range(warmup):
    q_i = q.detach().clone().requires_grad_(True)
    k_i = k.detach().clone().requires_grad_(True)
    v_i = v.detach().clone().requires_grad_(True)
    fn(q_i, k_i, v_i).backward(dO)
  torch.cuda.synchronize()

  start = time.time()
  for _ in range(iters):
    q_i = q.detach().clone().requires_grad_(True)
    k_i = k.detach().clone().requires_grad_(True)
    v_i = v.detach().clone().requires_grad_(True)
    fn(q_i, k_i, v_i).backward(dO)
  torch.cuda.synchronize()
  elapsed_ms = (time.time() - start) * 1000.0 / iters
  print(f"{name:>8}: {elapsed_ms:.3f} ms")
  return elapsed_ms


def main():
  args = parse_args()
  assert torch.cuda.is_available(), "CUDA is required"
  q, k, v, dO = make_inputs(args)
  scale = 1.0 / math.sqrt(args.D)

  def split_d(q_i, k_i, v_i):
    return ffpa_attn_func(
      q_i,
      k_i,
      v_i,
      causal=args.causal,
      softmax_scale=scale,
      stages=args.stages,
      acc="f32",
      backward_backend="split_d",
    )

  def sdpa(q_i, k_i, v_i):
    return F.scaled_dot_product_attention(q_i, k_i, v_i, is_causal=args.causal, scale=scale)

  print(
    f"shape B={args.B} H={args.H} N={args.N} D={args.D} dtype={args.dtype} "
    f"causal={args.causal} warmup={args.warmup} iters={args.iters}"
  )
  split_ms = time_backward("split_d", split_d, q, k, v, dO, args.warmup, args.iters)
  sdpa_ms = time_backward("sdpa", sdpa, q, k, v, dO, args.warmup, args.iters)
  print(f"speedup: {sdpa_ms / split_ms:.3f}x vs sdpa")


if __name__ == "__main__":
  main()

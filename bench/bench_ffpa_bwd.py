import argparse
import math
import time

import torch
import torch.nn.functional as F

from ffpa_attn import ffpa_attn_func  # noqa: E402


def parse_args():
  parser = argparse.ArgumentParser(description="Benchmark FFPA backward backends against SDPA backward.")
  parser.add_argument("--B", type=int, default=1)
  parser.add_argument("--H", type=int, default=2)
  parser.add_argument("--N", type=int, default=128)
  parser.add_argument("--D", type=int, default=512)
  parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
  parser.add_argument("--causal", action="store_true")
  parser.add_argument("--stages", type=int, default=1)
  parser.add_argument("--compare-stages", action="store_true", help="Run split-D stages 1, 2, and 3 side by side.")
  parser.add_argument(
    "--compare-backends",
    action="store_true",
    help="Run split-D, persistent-KV, Triton, and SDPA backward side by side.",
  )
  parser.add_argument(
    "--backward-backend",
    choices=["split_d", "persistent_kv", "triton"],
    default="split_d",
    help="Native backward backend to benchmark when not using a compare mode.",
  )
  parser.add_argument(
    "--mode",
    choices=["full", "backward-only"],
    default="full",
    help="full measures forward+backward; backward-only times only the backward call after forward is ready.",
  )
  parser.add_argument(
    "--autotune", action="store_true", help="Enable Triton autotuning (only effective for triton backend)."
  )
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


def time_backward_only(name, fn, q, k, v, dO, warmup, iters):
  start_event = torch.cuda.Event(enable_timing=True)
  end_event = torch.cuda.Event(enable_timing=True)

  for _ in range(warmup):
    q_i = q.detach().clone().requires_grad_(True)
    k_i = k.detach().clone().requires_grad_(True)
    v_i = v.detach().clone().requires_grad_(True)
    out = fn(q_i, k_i, v_i)
    torch.cuda.synchronize()
    out.backward(dO)
  torch.cuda.synchronize()

  elapsed_ms = 0.0
  for _ in range(iters):
    q_i = q.detach().clone().requires_grad_(True)
    k_i = k.detach().clone().requires_grad_(True)
    v_i = v.detach().clone().requires_grad_(True)
    out = fn(q_i, k_i, v_i)
    torch.cuda.synchronize()
    start_event.record()
    out.backward(dO)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_ms += start_event.elapsed_time(end_event)

  elapsed_ms /= iters
  print(f"{name:>14}: {elapsed_ms:.3f} ms")
  return elapsed_ms


def try_time_backend(timer, name, fn, q, k, v, dO, warmup, iters):
  try:
    return timer(name, fn, q, k, v, dO, warmup, iters)
  except RuntimeError as exc:
    if ("K/V-resident smem requirement" in str(exc) or "native backward was not compiled" in str(exc)):
      print(f"{name:>14}: unavailable ({exc})")
      return None
    raise


def backend_label(backend, stages):
  if backend == "split_d":
    return f"split_d_s{stages}"
  if backend == "persistent_kv":
    return f"persist_s{stages}"
  return backend


def main():
  args = parse_args()
  assert torch.cuda.is_available(), "CUDA is required"
  q, k, v, dO = make_inputs(args)
  scale = 1.0 / math.sqrt(args.D)

  def make_native(backend, stages):

    def native(q_i, k_i, v_i):
      return ffpa_attn_func(
        q_i,
        k_i,
        v_i,
        causal=args.causal,
        softmax_scale=scale,
        stages=stages,
        acc="f32",
        backward_backend=backend,
        autotune=args.autotune,
      )

    return native

  def make_split_d(stages):
    return make_native("split_d", stages)

  def split_d(q_i, k_i, v_i):
    return ffpa_attn_func(
      q_i,
      k_i,
      v_i,
      causal=args.causal,
      softmax_scale=scale,
      stages=args.stages,
      acc="f32",
      backward_backend=args.backward_backend,
      autotune=args.autotune,
    )

  def sdpa(q_i, k_i, v_i):
    return F.scaled_dot_product_attention(q_i, k_i, v_i, is_causal=args.causal, scale=scale)

  print(
    f"shape B={args.B} H={args.H} N={args.N} D={args.D} dtype={args.dtype} "
    f"causal={args.causal} mode={args.mode} autotune={args.autotune} "
    f"warmup={args.warmup} iters={args.iters}"
  )
  timer = time_backward_only if args.mode == "backward-only" else time_backward
  if args.compare_backends:
    split1_ms = try_time_backend(timer, "split_d_s1", make_native("split_d", 1), q, k, v, dO, args.warmup, args.iters)
    split2_ms = try_time_backend(timer, "split_d_s2", make_native("split_d", 2), q, k, v, dO, args.warmup, args.iters)
    pkv1_ms = try_time_backend(
      timer, "persist_s1", make_native("persistent_kv", 1), q, k, v, dO, args.warmup, args.iters
    )
    pkv2_ms = try_time_backend(
      timer, "persist_s2", make_native("persistent_kv", 2), q, k, v, dO, args.warmup, args.iters
    )
    triton_ms = try_time_backend(timer, "triton", make_native("triton", 1), q, k, v, dO, args.warmup, args.iters)
    sdpa_ms = timer("sdpa_bwd" if args.mode == "backward-only" else "sdpa", sdpa, q, k, v, dO, args.warmup, args.iters)
    if split1_ms is not None:
      print(f"speedup split_d_s1: {sdpa_ms / split1_ms:.3f}x vs sdpa")
    if split2_ms is not None:
      print(f"speedup split_d_s2: {sdpa_ms / split2_ms:.3f}x vs sdpa")
    if pkv1_ms is not None:
      print(f"speedup persist_s1: {sdpa_ms / pkv1_ms:.3f}x vs sdpa")
      if split1_ms is not None:
        print(f"persist_s1/split_d_s1: {pkv1_ms / split1_ms:.3f}x time")
    if pkv2_ms is not None:
      print(f"speedup persist_s2: {sdpa_ms / pkv2_ms:.3f}x vs sdpa")
      if split2_ms is not None:
        print(f"persist_s2/split_d_s2: {pkv2_ms / split2_ms:.3f}x time")
    if triton_ms is not None:
      print(f"speedup triton: {sdpa_ms / triton_ms:.3f}x vs sdpa")
      if split1_ms is not None:
        print(f"triton/split_d_s1: {triton_ms / split1_ms:.3f}x time")
      if split2_ms is not None:
        print(f"triton/split_d_s2: {triton_ms / split2_ms:.3f}x time")
  elif args.compare_stages:
    split1_ms = timer("split_d_s1", make_split_d(1), q, k, v, dO, args.warmup, args.iters)
    split2_ms = timer("split_d_s2", make_split_d(2), q, k, v, dO, args.warmup, args.iters)
    split3_ms = timer("split_d_s3", make_split_d(3), q, k, v, dO, args.warmup, args.iters)
    sdpa_ms = timer("sdpa_bwd" if args.mode == "backward-only" else "sdpa", sdpa, q, k, v, dO, args.warmup, args.iters)
    print(f"speedup stage1: {sdpa_ms / split1_ms:.3f}x vs sdpa")
    print(f"speedup stage2: {sdpa_ms / split2_ms:.3f}x vs sdpa")
    print(f"speedup stage3: {sdpa_ms / split3_ms:.3f}x vs sdpa")
    print(f"stage2/stage1: {split2_ms / split1_ms:.3f}x time")
    print(f"stage3/stage1: {split3_ms / split1_ms:.3f}x time")
    print(f"stage3/stage2: {split3_ms / split2_ms:.3f}x time")
  else:
    backend_name = backend_label(args.backward_backend, args.stages)
    split_ms = try_time_backend(timer, backend_name, split_d, q, k, v, dO, args.warmup, args.iters)
    sdpa_ms = timer("sdpa_bwd" if args.mode == "backward-only" else "sdpa", sdpa, q, k, v, dO, args.warmup, args.iters)
    if split_ms is not None:
      print(f"speedup {backend_name}: {sdpa_ms / split_ms:.3f}x vs sdpa")


if __name__ == "__main__":
  main()

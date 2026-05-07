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
  parser.add_argument(
    "--compare-stages",
    action="store_true",
    help="Run CUDA stages 1, 2, and 3 side by side.",
  )
  parser.add_argument(
    "--compare-backends",
    action="store_true",
    help="Run CUDA, Triton, and SDPA backward side by side.",
  )
  parser.add_argument(
    "--backward-backend",
    choices=["cuda", "triton"],
    default="triton",
    help="Native backward backend to benchmark when not using a compare mode.",
  )
  parser.add_argument(
    "--mode",
    choices=["full", "backward-only"],
    default="full",
    help="full measures forward+backward or backward-only.",
  )
  parser.add_argument(
    "--triton-backward-autotune",
    "--autotune",
    action="store_true",
    help="Enable Triton FFPA backward autotuning (only effective for triton backend)."
  )
  parser.add_argument(
    "--triton-backward-version",
    "--version",
    type=str,
    default="v2",
    help="Enable Triton FFPA backward versioning (only effective for triton backend)."
  )
  parser.add_argument(
    "--triton-preprocess-d-chunk",
    "--d-chunk",
    action="store_true",
    help="Use split-D delta preprocess for the Triton backward backend."
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


def run_forward_backward_once(fn, q, k, v, dO):
  """Run one forward/backward pass and return output plus input gradients.

  :param fn: Attention callable that consumes ``(q, k, v)`` and returns ``out``.
  :param q: Query tensor.
  :param k: Key tensor.
  :param v: Value tensor.
  :param dO: Upstream gradient tensor matching the output shape.
  :return: Tuple ``(out, dq, dk, dv)`` detached from autograd.
  """
  q_i = q.detach().clone().requires_grad_(True)
  k_i = k.detach().clone().requires_grad_(True)
  v_i = v.detach().clone().requires_grad_(True)
  out = fn(q_i, k_i, v_i)
  out.backward(dO)
  return out.detach(), q_i.grad.detach(), k_i.grad.detach(), v_i.grad.detach()


def tensor_max_abs_err(ref: torch.Tensor, actual: torch.Tensor) -> float:
  """Return the maximum absolute error between two tensors."""
  return (ref.float() - actual.float()).abs().max().item()


def print_max_abs_err(name, fn, ref, q, k, v, dO):
  """Print max-abs-error for one backend against the SDPA reference.

  :param name: Label shown in benchmark output.
  :param fn: Backend callable under test.
  :param ref: Reference tuple ``(out, dq, dk, dv)`` from SDPA.
  :param q: Query tensor.
  :param k: Key tensor.
  :param v: Value tensor.
  :param dO: Upstream gradient tensor.
  """
  out_ref, dq_ref, dk_ref, dv_ref = ref
  out, dq, dk, dv = run_forward_backward_once(fn, q, k, v, dO)
  out_err = tensor_max_abs_err(out_ref, out)
  dq_err = tensor_max_abs_err(dq_ref, dq)
  dk_err = tensor_max_abs_err(dk_ref, dk)
  dv_err = tensor_max_abs_err(dv_ref, dv)
  max_abs_err = max(out_err, dq_err, dk_err, dv_err)
  print(
    f"{name:>14}: max_abs_err={max_abs_err:.6e} "
    f"(out={out_err:.6e}, dq={dq_err:.6e}, dk={dk_err:.6e}, dv={dv_err:.6e})"
  )
  # Per-tensor allclose check with tolerances matching those used in
  # bench_ffpa_fwd.py (1e-2 for fp16, 2e-2 for bf16).
  rtol = 1e-2 if q.dtype == torch.float16 else 2e-2
  atol = rtol
  checks = [
    ("out", out_ref, out),
    ("dq", dq_ref, dq),
    ("dk", dk_ref, dk),
    ("dv", dv_ref, dv),
  ]
  failed = [tag for tag, a, b in checks if not torch.allclose(a.float(), b.float(), rtol=rtol, atol=atol)]
  if failed:
    print(f"  WARNING: {name} FAILED allclose check for: {', '.join(failed)} "
          f"(rtol={rtol}, atol={atol})")


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
  name = f"{name}_bwd"
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
  if backend == "cuda":
    return f"cuda_s{stages}"
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
        is_causal=args.causal,
        scale=scale,
        stages=stages,
        acc="f32",
        backward_backend=backend,
        triton_backward_autotune=args.triton_backward_autotune,
        triton_backward_version=args.triton_backward_version,
        triton_backward_preprocess_d_chunk=args.triton_preprocess_d_chunk,
      )

    return native

  def make_cuda(stages):
    return make_native("cuda", stages)

  def cuda(q_i, k_i, v_i):
    return ffpa_attn_func(
      q_i,
      k_i,
      v_i,
      is_causal=args.causal,
      scale=scale,
      stages=args.stages,
      acc="f32",
      backward_backend=args.backward_backend,
      triton_backward_autotune=args.triton_backward_autotune,
      triton_backward_version=args.triton_backward_version,
      triton_backward_preprocess_d_chunk=args.triton_preprocess_d_chunk,
    )

  def sdpa(q_i, k_i, v_i):
    return F.scaled_dot_product_attention(q_i, k_i, v_i, is_causal=args.causal, scale=scale)

  print(
    f"shape B={args.B} H={args.H} N={args.N} D={args.D} dtype={args.dtype} "
    f"causal={args.causal} mode={args.mode} autotune={args.triton_backward_autotune} "
    f"version={args.triton_backward_version} preprocess_d_chunk={args.triton_preprocess_d_chunk} "
    f"warmup={args.warmup} iters={args.iters}"
  )
  sdpa_ref = run_forward_backward_once(sdpa, q, k, v, dO)
  timer = time_backward_only if args.mode == "backward-only" else time_backward
  if args.compare_backends:
    cuda1_ms = try_time_backend(timer, "cuda_s1", make_native("cuda", 1), q, k, v, dO, args.warmup, args.iters)
    cuda2_ms = try_time_backend(timer, "cuda_s2", make_native("cuda", 2), q, k, v, dO, args.warmup, args.iters)
    triton_ms = try_time_backend(timer, "triton", make_native("triton", 1), q, k, v, dO, args.warmup, args.iters)
    sdpa_ms = timer("sdpa", sdpa, q, k, v, dO, args.warmup, args.iters)
    if cuda1_ms is not None:
      print(f"speedup cuda_s1: {sdpa_ms / cuda1_ms:.3f}x vs sdpa")
    if cuda2_ms is not None:
      print(f"speedup cuda_s2: {sdpa_ms / cuda2_ms:.3f}x vs sdpa")
    if triton_ms is not None:
      print(f"speedup triton: {sdpa_ms / triton_ms:.3f}x vs sdpa")
      if cuda1_ms is not None:
        print(f"triton/cuda_s1: {triton_ms / cuda1_ms:.3f}x time")
      if cuda2_ms is not None:
        print(f"triton/cuda_s2: {triton_ms / cuda2_ms:.3f}x time")
    if cuda1_ms is not None:
      print_max_abs_err("cuda_s1", make_native("cuda", 1), sdpa_ref, q, k, v, dO)
    if cuda2_ms is not None:
      print_max_abs_err("cuda_s2", make_native("cuda", 2), sdpa_ref, q, k, v, dO)
    if triton_ms is not None:
      print_max_abs_err("triton", make_native("triton", 1), sdpa_ref, q, k, v, dO)
  elif args.compare_stages:
    cuda1_ms = timer("cuda_s1", make_cuda(1), q, k, v, dO, args.warmup, args.iters)
    cuda2_ms = timer("cuda_s2", make_cuda(2), q, k, v, dO, args.warmup, args.iters)
    cuda3_ms = timer("cuda_s3", make_cuda(3), q, k, v, dO, args.warmup, args.iters)
    sdpa_ms = timer("sdpa", sdpa, q, k, v, dO, args.warmup, args.iters)
    print(f"speedup stage1: {sdpa_ms / cuda1_ms:.3f}x vs sdpa")
    print(f"speedup stage2: {sdpa_ms / cuda2_ms:.3f}x vs sdpa")
    print(f"speedup stage3: {sdpa_ms / cuda3_ms:.3f}x vs sdpa")
    print(f"stage2/stage1: {cuda2_ms / cuda1_ms:.3f}x time")
    print(f"stage3/stage1: {cuda3_ms / cuda1_ms:.3f}x time")
    print(f"stage3/stage2: {cuda3_ms / cuda2_ms:.3f}x time")
    print_max_abs_err("cuda_s1", make_cuda(1), sdpa_ref, q, k, v, dO)
    print_max_abs_err("cuda_s2", make_cuda(2), sdpa_ref, q, k, v, dO)
    print_max_abs_err("cuda_s3", make_cuda(3), sdpa_ref, q, k, v, dO)
  else:
    backend_name = backend_label(args.backward_backend, args.stages)
    cuda_ms = try_time_backend(timer, backend_name, cuda, q, k, v, dO, args.warmup, args.iters)
    sdpa_ms = timer("sdpa", sdpa, q, k, v, dO, args.warmup, args.iters)
    if cuda_ms is not None:
      print(f"speedup {backend_name}: {sdpa_ms / cuda_ms:.3f}x vs sdpa")
      print_max_abs_err(backend_name, cuda, sdpa_ref, q, k, v, dO)


if __name__ == "__main__":
  main()

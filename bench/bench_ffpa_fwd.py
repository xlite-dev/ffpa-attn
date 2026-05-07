import argparse
import math
import time

import torch
import torch.nn.functional as F

from ffpa_attn import ffpa_attn_func  # noqa: E402


def parse_args():
  parser = argparse.ArgumentParser(description="Benchmark FFPA forward backends against SDPA forward.")
  parser.add_argument("--B", type=int, default=1)
  parser.add_argument("--H", type=int, default=16)
  parser.add_argument("--H-kv", type=int, default=None, help="KV heads for GQA/MQA. Defaults to H.")
  parser.add_argument("--N", type=int, default=1024)
  parser.add_argument("--N-kv", type=int, default=None, help="KV sequence length. Defaults to N.")
  parser.add_argument("--D", type=int, default=320)
  parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
  parser.add_argument("--causal", action="store_true")
  parser.add_argument("--stages", type=int, default=2)
  parser.add_argument(
    "--compare-stages",
    action="store_true",
    help="Run CUDA stages 1, 2, and 3 side by side.",
  )
  parser.add_argument(
    "--compare-backends",
    action="store_true",
    help="Run CUDA, Triton, and SDPA forward side by side.",
  )
  parser.add_argument(
    "--forward-backend",
    choices=["cuda", "triton"],
    default="triton",
    help="Native forward backend to benchmark when not using a compare mode.",
  )
  parser.add_argument(
    "--triton-forward-autotune",
    "--autotune",
    action="store_true",
    help="Enable Triton FFPA forward autotuning (only effective for triton backend).",
  )
  parser.add_argument("--warmup", type=int, default=5)
  parser.add_argument("--iters", type=int, default=20)
  parser.add_argument("--seed", type=int, default=0)
  return parser.parse_args()


def make_inputs(args):
  torch.manual_seed(args.seed)
  dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
  nheads_kv = args.H if args.H_kv is None else args.H_kv
  seqlen_k = args.N if args.N_kv is None else args.N_kv
  q = torch.randn(args.B, args.H, args.N, args.D, device="cuda", dtype=dtype)
  k = torch.randn(args.B, nheads_kv, seqlen_k, args.D, device="cuda", dtype=dtype)
  v = torch.randn(args.B, nheads_kv, seqlen_k, args.D, device="cuda", dtype=dtype)
  return q, k, v


def sdpa_reference(q, k, v, causal, scale):
  group_size = q.size(1) // k.size(1)
  if group_size > 1:
    k = k.repeat_interleave(group_size, dim=1)
    v = v.repeat_interleave(group_size, dim=1)
  if causal and k.size(2) != q.size(2):
    n_q, n_k = q.size(2), k.size(2)
    kv_offset = n_k - n_q
    row_idx = torch.arange(n_q, device=q.device).view(-1, 1)
    col_idx = torch.arange(n_k, device=q.device).view(1, -1)
    attn_mask = (col_idx <= (row_idx + kv_offset))
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=scale)
  return F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)


def tensor_max_abs_err(ref: torch.Tensor, actual: torch.Tensor) -> float:
  """Return the maximum absolute error between two tensors."""
  return (ref.float() - actual.float()).abs().max().item()


def tensor_mean_abs_err(ref: torch.Tensor, actual: torch.Tensor) -> float:
  """Return the mean absolute error between two tensors."""
  return (ref.float() - actual.float()).abs().mean().item()


def print_abs_err(name, fn, ref, q, k, v):
  """Print forward accuracy for one backend against the SDPA reference.

  :param name: Label shown in benchmark output.
  :param fn: Backend callable under test.
  :param ref: SDPA reference output.
  :param q: Query tensor.
  :param k: Key tensor.
  :param v: Value tensor.
  """
  out = fn(q, k, v).detach()
  max_abs_err = tensor_max_abs_err(ref, out)
  mean_abs_err = tensor_mean_abs_err(ref, out)
  rtol = 1e-2 if q.dtype == torch.float16 else 2e-2
  atol = rtol
  allclose = torch.allclose(ref.float(), out.float(), rtol=rtol, atol=atol)
  print(f"{name:>14}: allclose={allclose} max_abs_err={max_abs_err:.6e} mean_abs_err={mean_abs_err:.6e}")


def time_forward(name, fn, q, k, v, warmup, iters):
  for _ in range(warmup):
    fn(q, k, v)
  torch.cuda.synchronize()

  start = time.time()
  for _ in range(iters):
    fn(q, k, v)
  torch.cuda.synchronize()
  elapsed_ms = (time.time() - start) * 1000.0 / iters
  print(f"{name:>14}: {elapsed_ms:.3f} ms")
  return elapsed_ms


def try_time_backend(name, fn, q, k, v, warmup, iters):
  try:
    return time_forward(name, fn, q, k, v, warmup, iters)
  except RuntimeError as exc:
    if "native forward was not compiled" in str(exc):
      print(f"{name:>14}: unavailable ({exc})")
      return None
    raise


def backend_label(backend, stages):
  if backend == "cuda":
    return f"cuda_s{stages}"
  return f"ffpa({backend})"


def main():
  args = parse_args()
  assert torch.cuda.is_available(), "CUDA is required"
  q, k, v = make_inputs(args)
  scale = 1.0 / math.sqrt(args.D)

  def make_native(backend, stages):

    def native(q_i, k_i, v_i):
      enable_gqa = q_i.size(1) != k_i.size(1)
      return ffpa_attn_func(
        q_i,
        k_i,
        v_i,
        is_causal=args.causal,
        scale=scale,
        enable_gqa=enable_gqa,
        stages=stages,
        acc="f32",
        forward_backend=backend,
        triton_forward_autotune=args.triton_forward_autotune,
      )

    return native

  def sdpa(q_i, k_i, v_i):
    return sdpa_reference(q_i, k_i, v_i, args.causal, scale)

  nheads_kv = args.H if args.H_kv is None else args.H_kv
  seqlen_k = args.N if args.N_kv is None else args.N_kv
  print(
    f"shape B={args.B} Hq={args.H} Hkv={nheads_kv} Nq={args.N} Nkv={seqlen_k} "
    f"D={args.D} dtype={args.dtype} causal={args.causal} "
    f"autotune={args.triton_forward_autotune} warmup={args.warmup} iters={args.iters}"
  )
  ref = sdpa(q, k, v).detach()

  if args.compare_backends:
    cuda1 = make_native("cuda", 1)
    cuda2 = make_native("cuda", 2)
    ffpa_triton = make_native("triton", 1)
    cuda1_ms = try_time_backend("cuda_s1", cuda1, q, k, v, args.warmup, args.iters)
    cuda2_ms = try_time_backend("cuda_s2", cuda2, q, k, v, args.warmup, args.iters)
    ffpa_ms = try_time_backend("ffpa(triton)", ffpa_triton, q, k, v, args.warmup, args.iters)
    sdpa_ms = time_forward("sdpa", sdpa, q, k, v, args.warmup, args.iters)
    if cuda1_ms is not None:
      print(f"speedup cuda_s1: {sdpa_ms / cuda1_ms:.3f}x vs sdpa")
    if cuda2_ms is not None:
      print(f"speedup cuda_s2: {sdpa_ms / cuda2_ms:.3f}x vs sdpa")
    if ffpa_ms is not None:
      print(f"speedup ffpa(triton): {sdpa_ms / ffpa_ms:.3f}x vs sdpa")
      if cuda1_ms is not None:
        print(f"ffpa(triton)/cuda_s1: {ffpa_ms / cuda1_ms:.3f}x time")
      if cuda2_ms is not None:
        print(f"ffpa(triton)/cuda_s2: {ffpa_ms / cuda2_ms:.3f}x time")
    if cuda1_ms is not None:
      print_abs_err("cuda_s1", cuda1, ref, q, k, v)
    if cuda2_ms is not None:
      print_abs_err("cuda_s2", cuda2, ref, q, k, v)
    if ffpa_ms is not None:
      print_abs_err("ffpa(triton)", ffpa_triton, ref, q, k, v)
  elif args.compare_stages:
    cuda1 = make_native("cuda", 1)
    cuda2 = make_native("cuda", 2)
    cuda3 = make_native("cuda", 3)
    cuda1_ms = time_forward("cuda_s1", cuda1, q, k, v, args.warmup, args.iters)
    cuda2_ms = time_forward("cuda_s2", cuda2, q, k, v, args.warmup, args.iters)
    cuda3_ms = time_forward("cuda_s3", cuda3, q, k, v, args.warmup, args.iters)
    sdpa_ms = time_forward("sdpa", sdpa, q, k, v, args.warmup, args.iters)
    print(f"speedup cuda_s1: {sdpa_ms / cuda1_ms:.3f}x vs sdpa")
    print(f"speedup cuda_s2: {sdpa_ms / cuda2_ms:.3f}x vs sdpa")
    print(f"speedup cuda_s3: {sdpa_ms / cuda3_ms:.3f}x vs sdpa")
    print_abs_err("cuda_s1", cuda1, ref, q, k, v)
    print_abs_err("cuda_s2", cuda2, ref, q, k, v)
    print_abs_err("cuda_s3", cuda3, ref, q, k, v)
  else:
    native = make_native(args.forward_backend, args.stages)
    backend_name = backend_label(args.forward_backend, args.stages)
    native_ms = try_time_backend(backend_name, native, q, k, v, args.warmup, args.iters)
    sdpa_ms = time_forward("sdpa", sdpa, q, k, v, args.warmup, args.iters)
    if native_ms is not None:
      print(f"speedup {backend_name}: {sdpa_ms / native_ms:.3f}x vs sdpa")
      print_abs_err(backend_name, native, ref, q, k, v)


if __name__ == "__main__":
  main()

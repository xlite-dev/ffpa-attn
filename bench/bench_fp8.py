"""FFPA-FP8 vs SageAttention vs SDPA(FA2) forward benchmark.

Covers self / causal / GQA / GQA-causal / cross-dense / non-aligned cases.
Cross-causal (Nq<Nkv with causal) is excluded: FFPA uses tail-aligned causal
semantics that diverge from SDPA's lower-triangular convention.

SDPA is pinned to FLASH_ATTENTION (FA2 on PyTorch 2.11) via ``sdpa_kernel``.
The accuracy reference is bf16 SDPA-FA2 (FA fp32-accumulates internally; fp32
SDPA would OOM at large N since MATH materializes the full Nq*Nkv matrix).

Examples:
  # default: N=8192,16384 D=128 H=32 Hkv=8 fp16
  python bench/bench_fp8.py
  # single shape
  python bench/bench_fp8.py --N 8192 --D 128 --H 32 --Hkv 8
  # bf16, more iters
  python bench/bench_fp8.py --dtype bf16 --iters 50
  # skip sage
  python bench/bench_fp8.py --no-sage
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from ffpa_attn import ffpa_attn_func
from ffpa_attn.cli._flops import attention_fwd_flops, tflops_from_ms
from ffpa_attn.functional import CUDABackend

SAGE_INSTALLED = importlib.util.find_spec("sageattention") is not None
if SAGE_INSTALLED:
  from sageattention import sageattn


def fp8_backend() -> CUDABackend:
  return CUDABackend(
    backward=False, enable_tma=True, enable_cute=True, enable_fp8=True
  )


def bench_ms(fn, warmup=10, iters=30) -> float:
  for _ in range(warmup):
    fn()
  torch.cuda.synchronize()
  ts = []
  for _ in range(iters):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    ts.append(time.perf_counter() - t0)
  return min(ts) * 1e3


@contextmanager
def fa2():
  # PyTorch 2.11 exposes FLASH_ATTENTION (= FA2); older versions use
  # FLASH_ATTENTION_2. Pick whichever is available.
  backend = getattr(SDPBackend, "FLASH_ATTENTION_2", SDPBackend.FLASH_ATTENTION)
  with sdpa_kernel(backend):
    yield


@dataclass
class Scenario:
  name: str
  B: int
  Hq: int
  Hkv: int
  Nq: int
  Nkv: int
  D: int
  causal: bool
  gqa: bool

  @property
  def flops(self) -> int:
    return attention_fwd_flops(
      batch=self.B,
      num_heads_q=self.Hq,
      nq=self.Nq,
      nkv=self.Nkv,
      headdim=self.D,
      causal=self.causal,
    )


def _mk(B, Hq, Hkv, Nq, Nkv, D, dtype, scale):
  q = torch.randn(B, Hq, Nq, D, dtype=dtype, device="cuda") * scale
  k = torch.randn(B, Hkv, Nkv, D, dtype=dtype, device="cuda") * scale
  v = torch.randn(B, Hkv, Nkv, D, dtype=dtype, device="cuda") * scale
  return q, k, v


def run_ffpa(q, k, v, causal, gqa):
  return ffpa_attn_func(
    q, k, v, is_causal=causal, enable_gqa=gqa, forward_backend=fp8_backend()
  )


def run_sage(q, k, v, causal, gqa):
  if not SAGE_INSTALLED:
    return None
  if gqa and q.size(1) != k.size(1):
    k = k.repeat_interleave(q.size(1) // k.size(1), dim=1)
    v = v.repeat_interleave(q.size(1) // v.size(1), dim=1)
  return sageattn(q, k, v, tensor_layout="HND", is_causal=causal)


def run_sdpa_fa2(q, k, v, causal, gqa):
  with fa2():
    return F.scaled_dot_product_attention(
      q, k, v, is_causal=causal, enable_gqa=gqa
    )


def ref_bf16_fa2(q, k, v, causal, gqa):
  # bf16 SDPA-FA2 as high-precision ref: FA internally fp32-accumulates, and
  # bf16 has more exponent range than fp16. fp32 SDPA would OOM at large N
  # (MATH backend materializes the full Nq*Nkv matrix) and fused kernels
  # do not support fp32.
  with fa2():
    return F.scaled_dot_product_attention(
      q.to(torch.bfloat16),
      k.to(torch.bfloat16),
      v.to(torch.bfloat16),
      is_causal=causal,
      enable_gqa=gqa,
    ).float()


def rel_err(out, ref) -> float:
  if out is None:
    return float("nan")
  return ((out.float() - ref).norm() / ref.norm()).item()


def build_scenarios(N, B, H, Hkv, D, cross_dense=True, non_aligned_pad=15):
  scs = [
    Scenario("self", B, H, H, N, N, D, causal=False, gqa=False),
    Scenario("causal", B, H, H, N, N, D, causal=True, gqa=False),
    Scenario("gqa", B, H, Hkv, N, N, D, causal=False, gqa=True),
    Scenario("gqa-causal", B, H, Hkv, N, N, D, causal=True, gqa=True),
  ]
  # cross-dense (Nkv=2Nq) skipped when 2N would exceed memory budget.
  if cross_dense:
    scs.append(
      Scenario(
        "cross-dense (Nkv=2Nq)", B, H, H, N, N * 2, D, causal=False, gqa=False
      )
    )
  if non_aligned_pad > 0:
    n_off = max(N - non_aligned_pad, 1)
    scs.append(
      Scenario(
        "non-aligned-dense", B, H, H, n_off, n_off, D, causal=False, gqa=False
      )
    )
    scs.append(
      Scenario(
        "non-aligned-causal", B, H, H, n_off, n_off, D, causal=True, gqa=False
      )
    )
  return scs


def run_scenario(sc, dtype, scale, warmup, iters, use_sage):
  q, k, v = _mk(sc.B, sc.Hq, sc.Hkv, sc.Nq, sc.Nkv, sc.D, dtype, scale)
  ref = ref_bf16_fa2(q, k, v, sc.causal, sc.gqa).to(dtype)

  outs = {
    "FFPA-FP8": run_ffpa(q, k, v, sc.causal, sc.gqa),
    "SDPA-FA2": run_sdpa_fa2(q, k, v, sc.causal, sc.gqa),
  }
  if use_sage:
    sage_out = run_sage(q, k, v, sc.causal, sc.gqa)
    if sage_out is not None:
      outs["Sage"] = sage_out
  errs = {name: rel_err(o, ref) for name, o in outs.items()}

  fns = {
    "FFPA-FP8": lambda: run_ffpa(q, k, v, sc.causal, sc.gqa),
    "SDPA-FA2": lambda: run_sdpa_fa2(q, k, v, sc.causal, sc.gqa),
  }
  if use_sage and SAGE_INSTALLED:
    fns["Sage"] = lambda: run_sage(q, k, v, sc.causal, sc.gqa)
  ms = {
    name: bench_ms(fn, warmup=warmup, iters=iters)
    for name, fn in fns.items()
  }

  print(f"### {sc.name}")
  print(
    f"  shape: B{sc.B} Hq{sc.Hq} Hkv{sc.Hkv} "
    f"Nq{sc.Nq} Nkv{sc.Nkv} D{sc.D} "
    f"causal={sc.causal} gqa={sc.gqa}"
  )
  print("| backend | rel_err | min(ms) | TFLOPS | speedup vs SDPA-FA2 |")
  print("|---|---|---|---|---|")
  order = ["FFPA-FP8", "Sage", "SDPA-FA2"]
  for name in order:
    if name not in ms:
      continue
    tf = tflops_from_ms(sc.flops, ms[name])
    sp = ms["SDPA-FA2"] / ms[name] if name != "SDPA-FA2" else 1.0
    err = errs.get(name, float("nan"))
    err_str = f"{err:.4f}" if err == err else "n/a"
    tf_str = f"{tf:.1f}" if tf is not None else "n/a"
    print(f"| {name} | {err_str} | {ms[name]:.3f} | {tf_str} | {sp:.3f}x |")
  print()


def parse_args():
  p = argparse.ArgumentParser(
    description="FFPA-FP8 vs Sage vs SDPA(FA2) forward benchmark"
  )
  p.add_argument(
    "--N",
    type=str,
    default="8192,16384",
    help="Comma-separated seqlens (default: 8192,16384)",
  )
  p.add_argument("--B", type=int, default=1, help="Batch size")
  p.add_argument("--H", type=int, default=32, help="Query heads")
  p.add_argument("--Hkv", type=int, default=8, help="KV heads (GQA)")
  p.add_argument("--D", type=int, default=128, help="Head dim")
  p.add_argument(
    "--dtype",
    type=str,
    default="fp16",
    choices=["fp16", "bf16"],
    help="Activation dtype",
  )
  p.add_argument("--scale", type=float, default=0.5, help="Input randn scale")
  p.add_argument("--warmup", type=int, default=10, help="Warmup iters")
  p.add_argument("--iters", type=int, default=30, help="Bench iters")
  p.add_argument("--no-sage", action="store_true", help="Skip SageAttention")
  p.add_argument(
    "--no-cross-dense",
    action="store_true",
    help="Skip cross-dense (Nkv=2Nq) scenario",
  )
  p.add_argument(
    "--non-aligned-pad",
    type=int,
    default=15,
    help="Non-aligned scenario uses N-pad (0 disables)",
  )
  return p.parse_args()


def main():
  args = parse_args()
  torch.manual_seed(0)
  dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
  use_sage = (not args.no_sage) and SAGE_INSTALLED
  Ns = [int(x) for x in args.N.split(",")]

  print(f"# dtype={dtype}, GPU={torch.cuda.get_device_name()}")
  print(
    f"# sage={'on' if use_sage else 'off'}, "
    f"SDPA=FLASH_ATTENTION(FA2), B={args.B} H={args.H} Hkv={args.Hkv} "
    f"D={args.D} warmup={args.warmup} iters={args.iters}\n"
  )

  for N in Ns:
    scenarios = build_scenarios(
      N,
      args.B,
      args.H,
      args.Hkv,
      args.D,
      cross_dense=not args.no_cross_dense,
      non_aligned_pad=args.non_aligned_pad,
    )
    print(f"## Nq=Nkv={N}\n")
    for sc in scenarios:
      run_scenario(sc, dtype, args.scale, args.warmup, args.iters, use_sage)


if __name__ == "__main__":
  main()

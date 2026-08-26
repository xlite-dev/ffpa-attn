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

import os
import argparse
import importlib.util
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from ffpa_attn import ffpa_attn_func
from ffpa_attn.cli._flops import attention_fwd_flops, tflops_from_ms
from ffpa_attn.functional import CUDABackend

SAGE_INSTALLED = importlib.util.find_spec("sageattention") is not None
if SAGE_INSTALLED:
  from sageattention import sageattn

FFPA_BENCH_FP8_FORCE_QK_INT8 = os.environ.get(
  "FFPA_BENCH_FP8_FORCE_QK_INT8", "0"
) == "1"
if FFPA_BENCH_FP8_FORCE_QK_INT8:
  print(
    "FFPA_BENCH_FP8_FORCE_QK_INT8=1: forcing FP8 Q/K matmul int8, P/V accumulate f16"
  )

# cachedit mirrors cache-dit's ffpa_fp8 backend: int8 QK + f16 PV acc +
# per_thread Q/K + per_channel V + smooth k (hybrid opt-in via --hybrid,
# default off: Sage has no equivalent stage). Same-basis comparison against
# Sage's native per-thread/per-channel kernels.
PRESETS = ("default", "int8", "cachedit", "cache_dit", "cache-dit")


def _is_5090_or_force_int8() -> bool:
  return "5090" in torch.cuda.get_device_name() or FFPA_BENCH_FP8_FORCE_QK_INT8


@lru_cache(maxsize=None)
def fp8_backend(preset: str = "default", hybrid: bool = False) -> CUDABackend:
  if preset == "int8" or (preset == "default" and _is_5090_or_force_int8()):
    # 5090 default / explicit int8 preset: int8 QK matmul, f16 PV acc.
    return CUDABackend(
      backward=False,
      enable_tma=True,
      enable_cute=True,
      enable_fp8=True,
      fp8_qk_mm_type="int8",
      fp8_pv_acc_type="f16",
      fp8_hybrid=hybrid,
    )
  if preset in ("cachedit", "cache_dit", "cache-dit"):
    return CUDABackend(
      backward=False,
      enable_tma=True,
      enable_cute=True,
      enable_fp8=True,
      fp8_qk_mm_type="int8",
      fp8_pv_acc_type="f16",
      fp8_q_quant_method="per_thread",
      fp8_k_quant_method="per_thread",
      fp8_v_quant_method="per_channel",
      fp8_smooth_k=True,
      fp8_smooth_v=False,
      fp8_hybrid=hybrid,
      fp8_hybrid_n_early=256,
    )
  # default: FP8 Q/K matmul fp8, P/V accumulate f32
  return CUDABackend(
    backward=False,
    enable_tma=True,
    enable_cute=True,
    enable_fp8=True,
    fp8_hybrid=hybrid,
  )


def bench_ms(fn, warmup=3, iters=5) -> float:
  # Inference benchmark: no_grad matches deployment and lets the ffpa_attn
  # inference fast path engage (grad-on forces the full meta/autograd chain,
  # ~30us of python per call that is not part of kernel performance).
  with torch.no_grad():
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


# SDPA backend registry. "fa2" resolves the version-dependent symbol
# (FLASH_ATTENTION on PyTorch 2.11, FLASH_ATTENTION_2 before).
SDPA_BACKENDS = {
  "fa2": getattr(SDPBackend, "FLASH_ATTENTION_2", SDPBackend.FLASH_ATTENTION),
  "cudnn": SDPBackend.CUDNN_ATTENTION,
  "math": SDPBackend.MATH,
  "mem_eff": SDPBackend.EFFICIENT_ATTENTION,
}


def resolve_sdpa_backend(name: str, D: int) -> str:
  # auto: FA2 covers D<=256, larger D falls to MATH. cuDNN (D<=128 only) is
  # opt-in via an explicit --sdpa-backend cudnn.
  if name == "auto":
    if D <= 256:
      return "fa2"
    return "math"
  return name


@contextmanager
def sdpa_backend_ctx(name: str):
  with sdpa_kernel(SDPA_BACKENDS[name]):
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


def run_ffpa(
  q, k, v, causal, gqa, preset="default", hybrid=False, with_permute=False
):
  if with_permute:
    # cache-dit E2E path: inputs arrive as diffusers NHD [B,N,H,D] storage;
    # ffpa consumes them natively via a zero-copy permute view (Phase C, no
    # transpose copy). The storage itself was materialized OUTSIDE the timed
    # region in run_scenario.
    q, k, v = (x.permute(0, 2, 1, 3) for x in (q, k, v))
  return ffpa_attn_func(
    q,
    k,
    v,
    is_causal=causal,
    enable_gqa=gqa,
    forward_backend=fp8_backend(preset, hybrid)
  )


def run_sage(q, k, v, causal, gqa, with_permute=False):
  if not SAGE_INSTALLED:
    return None
  # sageattn handles GQA natively (Hq % Hkv == 0); no KV repeat needed.
  # with_permute: q/k/v are already NHD [B,N,H,D] storage (diffusers native);
  # the NHD kernel returns [B,N,H,D], transpose back for BHND comparison.
  if with_permute:
    return sageattn(q, k, v, tensor_layout="NHD",
                    is_causal=causal).permute(0, 2, 1, 3)
  return sageattn(q, k, v, tensor_layout="HND", is_causal=causal)


def run_sdpa(q, k, v, causal, gqa, backend):
  with sdpa_backend_ctx(backend):
    return F.scaled_dot_product_attention(
      q, k, v, is_causal=causal, enable_gqa=gqa
    )


def ref_bf16(q, k, v, causal, gqa, backend):
  # bf16 SDPA as high-precision ref: fused kernels internally fp32-accumulate,
  # and bf16 has more exponent range than fp16. fp32 SDPA would OOM at large N
  # (MATH materializes the full Nq*Nkv matrix) and fused kernels don't do fp32.
  with sdpa_backend_ctx(backend):
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


def run_scenario(
  sc,
  dtype,
  scale,
  warmup,
  iters,
  use_sage,
  sdpa_name,
  preset="default",
  hybrid=False,
  with_permute=False
):
  q, k, v = _mk(sc.B, sc.Hq, sc.Hkv, sc.Nq, sc.Nkv, sc.D, dtype, scale)
  ref = ref_bf16(q, k, v, sc.causal, sc.gqa, sdpa_name).to(dtype)
  sdpa_label = f"SDPA-{sdpa_name.upper()}"
  if with_permute:
    # Materialize diffusers-style NHD [B,N,H,D] storage ONCE, outside every
    # timed region: ffpa consumes it via zero-copy permute views (Phase C),
    # sage natively; this mirrors real pipelines where attention inputs are
    # produced in NHD by upstream ops. SDPA still gets zero-copy BHND views
    # (it has no NHD mode here).
    q, k, v = (x.permute(0, 2, 1, 3).contiguous() for x in (q, k, v))
    sq, sk, sv = (x.permute(0, 2, 1, 3) for x in (q, k, v))
  else:
    sq, sk, sv = q, k, v

  outs = {
    "FFPA-FP8":
    run_ffpa(q, k, v, sc.causal, sc.gqa, preset, hybrid, with_permute),
    sdpa_label: run_sdpa(sq, sk, sv, sc.causal, sc.gqa, sdpa_name),
  }
  if use_sage:
    sage_out = run_sage(q, k, v, sc.causal, sc.gqa, with_permute)
    if sage_out is not None:
      outs["Sage"] = sage_out
  errs = {name: rel_err(o, ref) for name, o in outs.items()}

  # Bench order: SDPA pre-heat (unmeasured) stabilizes GPU clock/power first,
  # then FFPA, then Sage, then SDPA (reference). Measured backends run hot.
  for _ in range(5):
    run_sdpa(sq, sk, sv, sc.causal, sc.gqa, sdpa_name)
  torch.cuda.synchronize()
  fns = {
    "FFPA-FP8":
    lambda: run_ffpa(q, k, v, sc.causal, sc.gqa, preset, hybrid, with_permute),
  }
  if use_sage and SAGE_INSTALLED:
    fns["Sage"] = lambda: run_sage(q, k, v, sc.causal, sc.gqa, with_permute)
  fns[sdpa_label] = lambda: run_sdpa(sq, sk, sv, sc.causal, sc.gqa, sdpa_name)
  ms = {
    name: bench_ms(fn, warmup=warmup, iters=iters)
    for name, fn in fns.items()
  }

  print(
    f"{sc.name}, shape: B{sc.B} Hq{sc.Hq} Hkv{sc.Hkv} "
    f"Nq{sc.Nq} Nkv{sc.Nkv} D{sc.D} "
    f"causal={sc.causal} gqa={sc.gqa}"
  )
  # Markdown table with dynamic column widths so the pipes align in terminal.
  cols = ["backend", "rel_err", "min(ms)", "TFLOPS", f"speedup vs {sdpa_label}"]
  rows = []
  order = ["FFPA-FP8", "Sage", sdpa_label]
  for name in order:
    if name not in ms:
      continue
    tf = tflops_from_ms(sc.flops, ms[name])
    sp = ms[sdpa_label] / ms[name] if name != sdpa_label else 1.0
    err = errs.get(name, float("nan"))
    err_str = f"{err:.4f}" if err == err else "n/a"
    tf_str = f"{tf:.1f}" if tf is not None else "n/a"
    sp_str = f"{sp:.3f}x"
    rows.append([name, err_str, f"{ms[name]:.3f}", tf_str, sp_str])
  widths = [
    max(len(cols[i]), *(len(r[i]) for r in rows)) for i in range(len(cols))
  ]

  def fmt_row(cells):
    return "| " + " | ".join(
      f"{c:<{widths[i]}}" for i, c in enumerate(cells)
    ) + " |"

  print(fmt_row(cols))
  print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
  for r in rows:
    print(fmt_row(r))
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
  p.add_argument("--warmup", type=int, default=3, help="Warmup iters")
  p.add_argument("--iters", type=int, default=5, help="Bench iters")
  p.add_argument("--no-sage", action="store_true", help="Skip SageAttention")
  p.add_argument(
    "--hybrid",
    action="store_true",
    help="Enable FFPA fp8 hybrid path (fp16 early rows). Off by default: "
    "Sage has no equivalent, so hybrid on would bias the comparison.",
  )
  p.add_argument(
    "--preset",
    type=str,
    default="default",
    choices=list(PRESETS),
    help="FFPA fp8 config: default (fp8/f32 acc, per_block), int8 (int8/f16 "
    "acc, per_block), cachedit (int8/f16 acc, per_thread+per_channel+smooth "
    "{k,v}+hybrid 256 — same as cache-dit ffpa_fp8 backend)",
  )
  p.add_argument(
    "--with-permute",
    action="store_true",
    help="Simulate cache-dit E2E layout: FFPA pays NHD->BHND permute+"
    "contiguous per tensor, Sage consumes NHD natively",
  )
  p.add_argument(
    "--sdpa-backend",
    type=str,
    default="auto",
    choices=["auto", "fa2", "cudnn", "math", "mem_eff"],
    help="SDPA reference backend: auto picks FA2 (D<=256), MATH (D>256); "
    "an explicit value (incl. cudnn, D<=128) forces that backend.",
  )
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

  print(f"dtype={dtype}, GPU={torch.cuda.get_device_name()}")
  print(
    f"sage={'on' if use_sage else 'off'}, "
    f"SDPA={args.sdpa_backend}, B={args.B} H={args.H} Hkv={args.Hkv} "
    f"D={args.D} warmup={args.warmup} iters={args.iters} "
    f"preset={args.preset} with_permute={args.with_permute}\n"
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
      sdpa_name = resolve_sdpa_backend(args.sdpa_backend, sc.D)
      run_scenario(
        sc,
        dtype,
        args.scale,
        args.warmup,
        args.iters,
        use_sage,
        sdpa_name,
        preset=args.preset,
        hybrid=args.hybrid,
        with_permute=args.with_permute,
      )


if __name__ == "__main__":
  main()

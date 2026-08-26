"""FFPA-FP4 vs SageAttention3 (Blackwell NVFP4) vs SDPA(FA2) benchmark.

Both NVFP4 paths quantize Q/K/V to e2m1 + ue4m3 block scales and P to
two-level fp4 (SageAttention3 scheme); the comparison is end-to-end (FFPA
includes its fused preprocess kernels, sageattn3 its python preprocess +
clone to protect the caller's K, which its preprocess centers in-place).

sageattn3 has no GQA support and no causal hybrid: GQA rows show n/a for
Sage, and its causal early-row error (measured max ~1.2 at scale 0.5) is
the same e2m1 noise floor FFPA's fp16 hybrid suppresses.

The accuracy reference is bf16 SDPA-FA2 (fused kernels fp32-accumulate
internally; fp32 SDPA would OOM at large N).

Examples:
  python bench/bench_fp4.py
  python bench/bench_fp4.py --N 16384 --dtype bf16
  python bench/bench_fp4.py --no-sage --hybrid
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from ffpa_attn import ffpa_attn_func
from ffpa_attn.cli._flops import (
  attention_fwd_flops,
  format_tflops_short,
  tflops_from_ms,
)
from ffpa_attn.functional import CUDABackend

# Match the bench_fp8.py / ffpa_attn.bench CLI plot style.
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SAGE3_INSTALLED = importlib.util.find_spec("sageattn3") is not None
if SAGE3_INSTALLED:
  from sageattn3 import sageattn3_blackwell

PLOT_OUTPUT_DIR = Path(__file__).resolve().parent / ".tmp"
PLOT_CASE_ORDER = (
  "self",
  "causal",
  "gqa",
  "gqa-causal",
  "cross-dense (Nkv=2Nq)",
  "non-aligned-dense",
  "non-aligned-causal",
)
PLOT_CASE_LABELS = {"cross-dense (Nkv=2Nq)": "cross-dense"}
BACKEND_ORDER = ("FFPA-FP4", "FFPA-FP8", "Sage3", "SDPA")
BACKEND_COLORS = {
  "FFPA-FP4": "#663399",
  "FFPA-FP8": "#fd493c",
  "Sage3": "#2171b5",
  "SDPA": "#b0b0b0",
}


def fp4_backend(hybrid: bool = False, hybrid_n_early: int = 256):
  return CUDABackend(
    backward=False,
    enable_tma=True,
    enable_cute=True,
    enable_fp4=True,
    fp4_hybrid=hybrid,
    fp4_hybrid_n_early=hybrid_n_early,
  )


def fp8_backend() -> CUDABackend:
  return CUDABackend(
    backward=False,
    enable_tma=True,
    enable_cute=True,
    enable_fp8=True,
    fp8_hybrid=None,
  )


def bench_ms(fn, warmup=3, iters=5) -> float:
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


SDPA_BACKENDS = {
  "fa2": getattr(SDPBackend, "FLASH_ATTENTION_2", SDPBackend.FLASH_ATTENTION),
  "cudnn": SDPBackend.CUDNN_ATTENTION,
  "math": SDPBackend.MATH,
  "mem_eff": SDPBackend.EFFICIENT_ATTENTION,
}


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


def run_ffpa(q, k, v, causal, gqa, hybrid=False, hybrid_n_early=256):
  return ffpa_attn_func(
    q,
    k,
    v,
    is_causal=causal,
    enable_gqa=gqa,
    forward_backend=fp4_backend(hybrid, hybrid_n_early),
  )


def run_ffpa8(q, k, v, causal, gqa):
  return ffpa_attn_func(
    q,
    k,
    v,
    is_causal=causal,
    enable_gqa=gqa,
    forward_backend=fp8_backend(),
  )


def run_sage3(q, k, v, causal, gqa):
  if gqa:
    return None  # sageattn3 has no GQA support
  # sageattn3 centers K in place; clone keeps the benchmark inputs stable
  # across warmup + iters (the clone is part of its measured e2e cost).
  return sageattn3_blackwell(q, k.clone(), v, is_causal=causal)


def run_sdpa(q, k, v, causal, gqa, backend):
  with sdpa_backend_ctx(backend):
    return F.scaled_dot_product_attention(
      q, k, v, is_causal=causal, enable_gqa=gqa
    )


def ref_bf16(q, k, v, causal, gqa, backend):
  with sdpa_backend_ctx(backend):
    return F.scaled_dot_product_attention(
      q.to(torch.bfloat16),
      k.to(torch.bfloat16),
      v.to(torch.bfloat16),
      is_causal=causal,
      enable_gqa=gqa,
    ).float()


def build_scenarios(N, B, H, Hkv, D, cross_dense=True, non_aligned_pad=15):
  scs = [
    Scenario("self", B, H, H, N, N, D, causal=False, gqa=False),
    Scenario("causal", B, H, H, N, N, D, causal=True, gqa=False),
    Scenario("gqa", B, H, Hkv, N, N, D, causal=False, gqa=True),
    Scenario("gqa-causal", B, H, Hkv, N, N, D, causal=True, gqa=True),
  ]
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
  use_sage3,
  sdpa_name,
  base_n,
  hybrid=False,
  hybrid_n_early=256
):
  q, k, v = _mk(sc.B, sc.Hq, sc.Hkv, sc.Nq, sc.Nkv, sc.D, dtype, scale)
  ref = ref_bf16(q, k, v, sc.causal, sc.gqa, sdpa_name).to(dtype)
  sdpa_label = f"SDPA-{sdpa_name.upper()}"

  outs = {
    "FFPA-FP4": run_ffpa(q, k, v, sc.causal, sc.gqa, hybrid, hybrid_n_early),
    "FFPA-FP8": run_ffpa8(q, k, v, sc.causal, sc.gqa),
    sdpa_label: run_sdpa(q, k, v, sc.causal, sc.gqa, sdpa_name),
  }
  if use_sage3:
    outs["Sage3"] = run_sage3(q, k, v, sc.causal, sc.gqa)
  errs = {}
  maxerrs = {}
  for name, o in outs.items():
    if o is None:
      errs[name] = float("nan")
      maxerrs[name] = float("nan")
    else:
      d = o.float() - ref.float()
      errs[name] = (d.norm() / ref.float().norm()).item()
      maxerrs[name] = d.abs().max().item()

  # SDPA pre-heat (unmeasured) stabilizes clocks; then FFPA, Sage3, SDPA.
  for _ in range(5):
    run_sdpa(q, k, v, sc.causal, sc.gqa, sdpa_name)
  torch.cuda.synchronize()
  fns = {
    "FFPA-FP4":
    lambda: run_ffpa(q, k, v, sc.causal, sc.gqa, hybrid, hybrid_n_early),
    "FFPA-FP8":
    lambda: run_ffpa8(q, k, v, sc.causal, sc.gqa),
  }
  if use_sage3 and not sc.gqa:
    fns["Sage3"] = lambda: run_sage3(q, k, v, sc.causal, sc.gqa)
  fns[sdpa_label] = lambda: run_sdpa(q, k, v, sc.causal, sc.gqa, sdpa_name)
  ms = {
    name: bench_ms(fn, warmup=warmup, iters=iters)
    for name, fn in fns.items()
  }

  print(
    f"{sc.name}, shape: B{sc.B} Hq{sc.Hq} Hkv{sc.Hkv} "
    f"Nq{sc.Nq} Nkv{sc.Nkv} D{sc.D} "
    f"causal={sc.causal} gqa={sc.gqa}"
  )
  cols = [
    "backend", "rel_err", "max_err", "min(ms)", "TFLOPS",
    f"speedup vs {sdpa_label}", "speedup vs FP8"
  ]
  rows = []
  order = ["FFPA-FP4", "FFPA-FP8", "Sage3", sdpa_label]
  fp8_ms = ms.get("FFPA-FP8")
  for name in order:
    if name not in ms:
      continue
    tf = tflops_from_ms(sc.flops, ms[name])
    sp = ms[sdpa_label] / ms[name] if name != sdpa_label else 1.0
    ratio = (fp8_ms / ms[name]) if (fp8_ms and name != "FFPA-FP8") else None
    ratio_str = f"{ratio:.3f}x" if ratio is not None else "-"
    err = errs.get(name, float("nan"))
    err_str = f"{err:.4f}" if err == err else "n/a"
    merr = maxerrs.get(name, float("nan"))
    merr_str = f"{merr:.4f}" if merr == merr else "n/a"
    tf_str = f"{tf:.1f}" if tf is not None else "n/a"
    rows.append([
      name, err_str, merr_str, f"{ms[name]:.3f}", tf_str, f"{sp:.3f}x",
      ratio_str
    ])
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
  return {
    "case_name": sc.name,
    "base_n": base_n,
    "tflops": {
      name: tflops_from_ms(sc.flops, ms[name])
      for name in ms
    },
  }


def _fmt_n(n: int) -> str:
  """Format a sequence length compactly for labels.

  :param n: Base sequence length.
  :return: ``8192 -> "8K"``; non-1024 multiples stay decimal.
  """
  if n % 1024 == 0:
    return f"{n // 1024}K"
  return str(n)


def _slugify_device_name(device_name: str) -> str:
  """Convert a device name into a filesystem-friendly slug.

  :param device_name: Human-readable device name.
  :return: Lowercase slug safe for filenames.
  """
  slug = re.sub(r"[^0-9A-Za-z]+", "-", device_name.strip().lower())
  slug = re.sub(r"-+", "-", slug).strip("-")
  return slug or "unknown-device"


def _normalize_backend(name: str) -> str:
  """Collapse dynamic SDPA labels (SDPA-FA2, ...) into one plot series.

  :param name: Backend key from the ``ms`` dict.
  :return: Canonical plot backend name.
  """
  return "SDPA" if name.startswith("SDPA") else name


def plot_tflops(
  rows,
  *,
  device_name,
  output_path,
  B,
  H,
  Hkv,
  D,
):
  """Render the forward TFLOPS bar chart across scenarios and seqlens.

  Draws at most the three largest base sequence lengths (tables stay complete
  for every ``--N`` value), mirroring bench_fp8.py. Sage3 lacks GQA support,
  so GQA clusters omit its bar entirely (no reserved gap; bars pack tight).

  :param rows: Result rows returned by :func:`run_scenario`.
  :param device_name: Device name shown in the title.
  :param output_path: Output PNG path.
  :param B: Batch size shown in the title.
  :param H: Base query-head count shown in the title.
  :param Hkv: Base KV-head count (GQA) shown in the title.
  :param D: Head dimension shown in the title.
  :return: Saved PNG path.
  """
  plot_ns = sorted({row["base_n"] for row in rows})[-3:]
  filtered = [row for row in rows if row["base_n"] in plot_ns]
  present_cases = {row["case_name"] for row in filtered}
  cases = [c for c in PLOT_CASE_ORDER if c in present_cases]
  cases += [c for c in present_cases if c not in PLOT_CASE_ORDER]
  values = {}
  for row in filtered:
    for name, tf in row["tflops"].items():
      values[(row["case_name"], row["base_n"], _normalize_backend(name))] = tf
  present_backends = {
    _normalize_backend(name)
    for row in filtered
    for name in row["tflops"]
  }
  backends = [b for b in BACKEND_ORDER if b in present_backends]
  width = 0.24

  # Dense slot packing: clusters skip absent backends (Sage3 has no GQA), so
  # no empty slot is reserved for unsupported case/backend combinations.
  slot_of = {}
  cluster_base = {}
  cluster_slots = {}
  cursor = 0.0
  for case in cases:
    slots = 0
    for n in plot_ns:
      for backend in backends:
        if values.get((case, n, backend)) is not None:
          slot_of[(case, n, backend)] = slots
          slots += 1
    cluster_base[case] = cursor
    cluster_slots[case] = slots
    cursor += slots * width + width  # one-bar-width gap between clusters

  fig, ax = plt.subplots(figsize=(32, 12))
  finite_values = [
    tf for tf in values.values() if tf is not None and np.isfinite(tf)
  ]
  ymax = max(finite_values) if finite_values else 1.0

  def _autolabel(x, h, text) -> None:
    if not np.isfinite(h):
      return
    ax.annotate(
      text,
      # Bars are centered at x in data coordinates.
      xy=(x, h),
      xytext=(0, 5),
      textcoords="offset points",
      ha="center",
      va="bottom",
      fontsize=10,
      fontweight="bold",
    )

  for n in plot_ns:
    for backend in backends:
      positions = []
      heights = []
      for case in cases:
        tf = values.get((case, n, backend))
        if tf is None:
          continue  # Sage3 GQA: no slot reserved, bars pack tight
        pos = cluster_base[case] + slot_of[(case, n, backend)] * width
        positions.append(pos)
        heights.append(tf)
      ax.bar(
        positions,
        heights,
        width,
        label=f"{backend} N={_fmt_n(n)}",
        color=BACKEND_COLORS[backend],
        edgecolor="white",
        linewidth=1,
      )
      for x, h in zip(positions, heights):
        _autolabel(x, h, format_tflops_short(h).removesuffix("T"))

  fig.suptitle(
    f"FFPA-FP4 TFLOPS (FWD) | {device_name} | "
    f"B={B}, H={H}, Hkv={Hkv}, D={D}",
    fontsize=18,
    fontweight="bold",
    y=0.958,
  )
  ax.set_xticks([
    cluster_base[c] + (cluster_slots[c] - 1) * width / 2 for c in cases
  ])
  # One-bar-width left margin; right margin halved (bars crowd the right edge).
  last = cases[-1]
  ax.set_xlim(
    -width, cluster_base[last] + cluster_slots[last] * width - width / 2
  )
  # Tag each case with the plotted seqlens (left-to-right N order in-cluster).
  ns_tag = f"({'/'.join(_fmt_n(n) for n in plot_ns)})"
  ax.set_xticklabels(
    [PLOT_CASE_LABELS.get(c, c) + ns_tag for c in cases],
    rotation=0,
    ha="center",
    fontsize=15,
  )
  ax.set_ylabel("Throughput (TFLOPS)", fontsize=18)
  ax.tick_params(axis="y", labelsize=16)
  ax.set_ylim(0, ymax * 1.10 if ymax > 0 else 1.0)
  ax.legend(
    fontsize=16,
    loc="upper right",
    ncol=len(backends),
    columnspacing=1.5,
    handletextpad=0.6,
    frameon=True,
  )
  ax.grid(axis="y", alpha=0.3)

  fig.tight_layout(rect=(0, 0, 1, 0.965))
  output_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(output_path)
  plt.close(fig)
  return output_path


def parse_args():
  p = argparse.ArgumentParser(
    description="FFPA-FP4 vs SageAttn3 vs SDPA(FA2) forward benchmark"
  )
  p.add_argument(
    "--N",
    type=str,
    default="8192,16384",
    help="Comma-separated seqlens (default: 8192,16384)"
  )
  p.add_argument("--B", type=int, default=1, help="Batch size")
  p.add_argument("--H", type=int, default=32, help="Query heads")
  p.add_argument("--Hkv", type=int, default=8, help="KV heads (GQA)")
  p.add_argument(
    "--D",
    type=int,
    default=128,
    help="Head dim (fp4: any D%8==0 in [8,256], padded up to {64,128,192,256})"
  )
  p.add_argument(
    "--dtype",
    type=str,
    default="bf16",
    choices=["fp16", "bf16"],
    help="Activation dtype"
  )
  p.add_argument("--scale", type=float, default=0.5, help="Input randn scale")
  p.add_argument("--warmup", type=int, default=3, help="Warmup iters")
  p.add_argument("--iters", type=int, default=5, help="Bench iters")
  p.add_argument("--no-sage", action="store_true", help="Skip sageattn3")
  p.add_argument(
    "--hybrid",
    action="store_true",
    help="Enable FFPA causal fp16 hybrid (fp16 early rows). Off by default "
    "(mirrors bench_fp8.py): Sage3 has no equivalent stage."
  )
  p.add_argument(
    "--fp4-hybrid-n-early",
    type=int,
    default=256,
    help="Leading query rows computed in fp16 under the fp4 hybrid "
    "(default 256, must be multiple of 128)"
  )
  p.add_argument(
    "--sdpa-backend",
    type=str,
    default="fa2",
    choices=["fa2", "cudnn", "math", "mem_eff"],
    help="SDPA reference backend"
  )
  p.add_argument(
    "--no-cross-dense",
    action="store_true",
    help="Skip cross-dense (Nkv=2Nq) scenario"
  )
  return p.parse_args()


def main():
  args = parse_args()
  dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
  torch.cuda.init()
  print(f"GPU: {torch.cuda.get_device_name()}")
  print(
    f"dtype={args.dtype} scale={args.scale} "
    f"warmup={args.warmup} iters={args.iters} "
    f"sage3={SAGE3_INSTALLED and not args.no_sage} "
    f"hybrid={args.hybrid}"
  )
  rows = []
  use_sage3 = SAGE3_INSTALLED and not args.no_sage
  for n in [int(x) for x in args.N.split(",")]:
    for sc in build_scenarios(
      n, args.B, args.H, args.Hkv, args.D, cross_dense=not args.no_cross_dense
    ):
      rows.append(
        run_scenario(
          sc,
          dtype,
          args.scale,
          args.warmup,
          args.iters,
          use_sage3,
          args.sdpa_backend,
          base_n=n,
          hybrid=args.hybrid,
          hybrid_n_early=args.fp4_hybrid_n_early
        )
      )

  device_name = torch.cuda.get_device_name()
  output_path = PLOT_OUTPUT_DIR / (
    f"bench_fp4_tflops_{_slugify_device_name(device_name)}"
    f"_B{args.B}_H{args.H}_Hkv{args.Hkv}_D{args.D}_{args.dtype}.png"
  )
  saved = plot_tflops(
    rows,
    device_name=device_name,
    output_path=output_path,
    B=args.B,
    H=args.H,
    Hkv=args.Hkv,
    D=args.D,
  )
  print(f"Saved TFLOPS plot to {saved}")


if __name__ == "__main__":
  main()

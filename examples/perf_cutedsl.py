"""Benchmark the CuTeDSL D=512 SM90 FFPA path and render plot plus Markdown tables.

This script is the CuTeDSL counterpart to ``examples/perf.py``. It exclusively
exercises the ``ffpa_attn_func(..., forward_backend="cutedsl")`` fast-path that
dispatches to ``ffpa_attn.cutedsl.interface.split_flash_attn_func`` and the
native ``_flash_attn_fwd_sm90`` / ``_flash_attn_bwd_sm90`` kernels. Forward and
backward are paired automatically by ``FFPAAttnMeta`` — there is no
``--fwd-backend`` / ``--bwd-backend`` choice to make.

Hard constraints inherited from the CuTeDSL backend (see
``ffpa_attn/cutedsl/_wrappers.py``):

* SM90 (Hopper) only.
* ``head_dim == 512`` for q / k / v.
* dtype is bf16 (training requires bf16; fp16 backward is unsupported).
* No ``attn_mask`` / no ``dropout`` (those cases are dropped from the suite).

Usage::

  CUDA_VISIBLE_DEVICES=0 python examples/perf_cutedsl.py
  CUDA_VISIBLE_DEVICES=0 python examples/perf_cutedsl.py --no-bwd
  CUDA_VISIBLE_DEVICES=0 python examples/perf_cutedsl.py --tasks self-attn,gqa,causal
  CUDA_VISIBLE_DEVICES=0 python examples/perf_cutedsl.py --show-allclose --N 4096
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import re
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
  sys.path.insert(0, str(EXAMPLES_DIR))

from attention_flops import (
  attention_bwd_flops,
  attention_fwd_flops,
  format_tflops_short,
  tflops_from_ms,
)
from ffpa_attn_bwd import (
  _make_sdpa_kwargs,
  _time_backward_only,
)
from ffpa_attn_fwd import (
  DEFAULT_ITERS,
  DEFAULT_WARMUP,
  _dtype_tag,
  _expand_kv,
  _maybe_norm_qkv,
  _max_abs_diff,
  _mean_abs_diff,
  _resolve_gqa_heads,
  _sdpa_ref,
  _tensor_allclose,
  _time_fn,
  _validate_timing_args,
)

from ffpa_attn import ffpa_attn_func

# Keep the exact legacy plotting style from tools/plot.py.
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HEAD_DIM = 512  # CuTeDSL backend is specialized to D=512.
CUTEDSL_DTYPE = torch.bfloat16  # bf16 supports both fwd and bwd; fp16 bwd raises.

PLOT_CASES: list[tuple[str, str]] = [
  ("self-attn", "self-attn(F/B)"),
  ("cross-attn", "cross-attn(F/B)"),
  ("gqa", "gqa(F/B)"),
  ("causal", "causal(F/B)"),
]
CASE_LABELS = dict(PLOT_CASES)
VALID_TASKS = tuple(case_name for case_name, _ in PLOT_CASES)
DTYPE_ORDER = ["bf16"]
DEFAULT_OUTPUT_STEM = "ffpa_speedup_cutedsl"
DEFAULT_OUTPUT_DIR = Path(".tmp")
TFLOPS_FWD_SDPA_COLOR = "#b0b0b0"
TFLOPS_FWD_FFPA_COLOR = "#2171b5"
TFLOPS_BWD_SDPA_COLOR = "#f5a623"
TFLOPS_BWD_FFPA_COLOR = "#fd493c"
SECTION_LABEL = "CuTeDSL (SM90 D=512)"
RESULT_ROW = dict[str, Any]


def _parse_args() -> argparse.Namespace:
  """Parse CLI arguments.

  :return: Parsed CLI namespace.
  """
  parser = argparse.ArgumentParser(
    description="Benchmark the CuTeDSL D=512 SM90 FFPA forward/backward path against SDPA."
  )
  parser.add_argument(
    "--no-forward",
    "--no-fwd",
    dest="forward",
    action="store_false",
    help="Disable forward benchmark cases.",
  )
  parser.add_argument(
    "--no-backward",
    "--no-bwd",
    dest="backward",
    action="store_false",
    help="Disable backward benchmark cases.",
  )
  parser.set_defaults(forward=True, backward=True)
  parser.add_argument(
    "--tasks",
    nargs="*",
    default=None,
    help=(
      "Benchmark cases to run, separated by commas or whitespace, for example self-attn,cross-attn. "
      "Defaults to full; valid cases: " + ",".join(VALID_TASKS)
    ),
  )
  parser.add_argument("--B", type=int, default=1, help="Batch size used by benchmark mode.")
  parser.add_argument("--H", type=int, default=32, help="Base head count used by benchmark mode.")
  parser.add_argument("--N", type=int, default=8192, help="Base sequence length used by benchmark mode.")
  parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP, help="Warmup iterations used for timing.")
  parser.add_argument("--iters", type=int, default=DEFAULT_ITERS, help="Measured iterations used for timing.")
  parser.add_argument("--seed", type=int, default=42, help="RNG seed used by benchmark mode.")
  parser.add_argument(
    "--norm",
    action="store_true",
    help="Enable pre-attention LayerNorm on q/k/v for both FFPA and SDPA paths.",
  )
  parser.add_argument(
    "--show-allclose",
    action="store_true",
    help="Include the allclose column in the generated Markdown tables.",
  )
  parser.add_argument(
    "--save-path",
    type=Path,
    default=None,
    help="Optional output directory used to save the generated PNG and Markdown artifacts. Defaults to ./.tmp.",
  )
  return parser.parse_args()


def _parse_tasks_arg(tasks_arg: list[str] | None) -> set[str] | None:
  """Parse the optional benchmark task filter.

  :param tasks_arg: Raw ``--tasks`` values.
  :return: Selected case names, or ``None`` for the full benchmark suite.
  :raises SystemExit: If an unknown case name is requested.
  """
  if tasks_arg is None:
    return None
  normalized = " ".join(tasks_arg).strip()
  if normalized == "" or normalized.lower() in {"full", "all", "none"}:
    return None
  tasks = {task for task in re.split(r"[\s,]+", normalized) if task}
  if not tasks:
    return None
  invalid = sorted(tasks.difference(VALID_TASKS))
  if invalid:
    valid = ",".join(VALID_TASKS)
    raise SystemExit(f"Unknown --tasks value(s): {','.join(invalid)}. Valid cases: {valid}, or full.")
  return tasks


def _active_plot_cases(tasks: set[str] | None) -> list[tuple[str, str]]:
  """Return plot cases filtered by an optional task set.

  :param tasks: Optional selected case names.
  :return: Ordered plot case list.
  """
  if tasks is None:
    return list(PLOT_CASES)
  return [(case_name, label) for case_name, label in PLOT_CASES if case_name in tasks]


def _device_name() -> str:
  """Return the active CUDA device name when available.

  :return: CUDA device name or a fallback label.
  """
  if torch.cuda.is_available():
    return torch.cuda.get_device_name(torch.cuda.current_device())
  return "CUDA Unavailable"


def _display_device_name(device_name: str) -> str:
  """Return the device name used in plot titles.

  Renames ``H20Z`` to ``H200`` for display so the chart matches the marketed
  product name; the filename slug and Markdown ``Env:`` field keep the raw
  CUDA-reported name to remain consistent with previously published artifacts.
  """
  return re.sub(r"H20Z", "H200", device_name, flags=re.IGNORECASE)


def _slugify_device_name(device_name: str) -> str:
  """Convert a device name into a filesystem-friendly slug.

  :param device_name: Human-readable device name.
  :return: Lowercase slug safe for filenames.
  """
  slug = re.sub(r"[^0-9A-Za-z]+", "-", device_name.strip().lower())
  slug = re.sub(r"-+", "-", slug).strip("-")
  return slug or "unknown-device"


def _output_stem(device_name: str, B: int, H: int, N: int, D: int) -> Path:
  """Build the output stem shared by the PNG and Markdown files.

  :param device_name: Device name used in the run.
  :param B: Batch size.
  :param H: Head count.
  :param N: Sequence length.
  :param D: Head dimension.
  :return: Output stem without extension.
  """
  device_slug = _slugify_device_name(device_name)
  return Path(f"{DEFAULT_OUTPUT_STEM}_{device_slug}_B{B}_H{H}_N{N}_D{D}")


def _resolve_output_stem(save_path: Path | None, device_name: str, B: int, H: int, N: int, D: int) -> Path:
  """Resolve the final output stem, optionally rooted at ``save_path``.

  :param save_path: Optional output directory.
  :param device_name: Device name used in the run.
  :param B: Batch size.
  :param H: Head count.
  :param N: Sequence length.
  :param D: Head dimension.
  :return: Output stem without extension.
  """
  default_stem = _output_stem(device_name, B, H, N, D)
  output_dir = DEFAULT_OUTPUT_DIR if save_path is None else save_path
  output_dir.mkdir(parents=True, exist_ok=True)
  return output_dir / default_stem.name


def _mode_suffix(has_forward: bool, has_backward: bool) -> str:
  """Build the title suffix that matches the legacy title style."""
  if has_forward and has_backward:
    return "FWD & BWD"
  if has_forward:
    return "FWD"
  return "BWD"


def _decorate_rows(direction: str, rows: list[dict[str, Any]]) -> list[RESULT_ROW]:
  """Attach the direction field to benchmark results."""
  return [{"direction": direction, **row} for row in rows]


def _case_specs(N: int, H: int, tasks: set[str] | None) -> list[dict[str, Any]]:
  """Build the cutedsl-compatible case specs.

  Mirrors the spec construction in ``ffpa_attn_fwd.run_forward_examples`` for
  the cases the cutedsl backend can handle: self-attn, cross-attn, gqa, causal.
  """
  gqa_heads = _resolve_gqa_heads(H)
  specs: list[dict[str, Any]] = [
    {
      "name": "self-attn",
      "Nh_q": H,
      "Nh_kv": H,
      "Nq": N,
      "Nkv": N
    },
    {
      "name": "cross-attn",
      "Nh_q": H,
      "Nh_kv": H,
      "Nq": 1024,
      "Nkv": N
    },
    {
      "name": "gqa",
      "Nh_q": H,
      "Nh_kv": gqa_heads,
      "Nq": N,
      "Nkv": N
    },
    {
      "name": "causal",
      "Nh_q": H,
      "Nh_kv": H,
      "Nq": N,
      "Nkv": N,
      "causal": True
    },
  ]
  if tasks is not None:
    specs = [spec for spec in specs if spec["name"] in tasks]
  return specs


def _format_forward_result(row: RESULT_ROW) -> str:
  """One-line forward result summary mirroring the perf.py CLI output."""
  return (
    f"[{row['case_name']:<14} {row['dtype']:<6}] "
    f"B={row['B']} Hq={row['Hq']} Hkv={row['Hkv']} "
    f"Nq={row['Nq']} Nkv={row['Nkv']} D={row['D']} "
    f"causal={int(row['causal'])}  "
    f"max|diff|={row['max_diff']:.4f}  mean|diff|={row['mean_diff']:.5f}  "
    f"allclose(atol={row['tolerance']})={row['allclose']}  "
    f"FFPA={row['ffpa_ms']:.2f} ms  SDPA={row['sdpa_ms']:.2f} ms  "
    f"TFLOPS={format_tflops_short(row['ffpa_tflops'])}/{format_tflops_short(row['sdpa_tflops'])}  "
    f"speedup={row['speedup']:.2f}x"
  )


def _format_backward_result(row: RESULT_ROW) -> str:
  """One-line backward result summary mirroring the perf.py CLI output."""
  return (
    f"[{row['case_name']:<14} {row['dtype']:<6}] "
    f"B={row['B']} Hq={row['Hq']} Hkv={row['Hkv']} "
    f"Nq={row['Nq']} Nkv={row['Nkv']} D={row['D']} "
    f"causal={int(row['causal'])}  "
    f"dQ_err={row['dq_err']:.4e}  dK_err={row['dk_err']:.4e}  dV_err={row['dv_err']:.4e}  "
    f"allclose={row['allclose']}  "
    f"FFPA={row['ffpa_ms']:.2f} ms  SDPA={row['sdpa_ms']:.2f} ms  "
    f"TFLOPS={format_tflops_short(row['ffpa_tflops'])}/{format_tflops_short(row['sdpa_tflops'])}  "
    f"speedup={row['speedup']:.2f}x"
  )


def _run_cutedsl_forward_case(
  name: str,
  *,
  seed: int,
  B: int,
  Nh_q: int,
  Nh_kv: int,
  Nq: int,
  Nkv: int,
  D: int,
  causal: bool,
  apply_norm: bool,
  warmup: int,
  iters: int,
  print_result: bool = True,
) -> RESULT_ROW:
  """Run one forward case through the CuTeDSL fast-path and SDPA, then time both."""
  dtype = CUTEDSL_DTYPE
  torch.manual_seed(seed)
  q = torch.randn(B, Nh_q, Nq, D, dtype=dtype, device="cuda")
  k = torch.randn(B, Nh_kv, Nkv, D, dtype=dtype, device="cuda")
  v = torch.randn(B, Nh_kv, Nkv, D, dtype=dtype, device="cuda")
  q, k, v = _maybe_norm_qkv(q, k, v, apply_norm)

  out_ffpa = ffpa_attn_func(
    q,
    k,
    v,
    is_causal=causal,
    enable_gqa=Nh_q != Nh_kv,
    forward_backend="cutedsl",
  )

  k_ref, v_ref = _expand_kv(k, v, Nh_q)
  out_sdpa = _sdpa_ref(q, k_ref, v_ref, is_causal=causal)

  tol = 5e-2
  ok = _tensor_allclose(out_ffpa, out_sdpa, tol)

  ms_ffpa = _time_fn(
    lambda q, k, v: ffpa_attn_func(
      q,
      k,
      v,
      is_causal=causal,
      enable_gqa=Nh_q != Nh_kv,
      forward_backend="cutedsl",
    ),
    q,
    k,
    v,
    warmup=warmup,
    iters=iters,
  )
  ms_sdpa = _time_fn(
    lambda q, k, v: _sdpa_ref(q, k, v, is_causal=causal),
    q,
    k_ref,
    v_ref,
    warmup=warmup,
    iters=iters,
  )
  flop_count = attention_fwd_flops(B, Nh_q, Nq, Nkv, D, causal)

  row: RESULT_ROW = {
    "case_name": name,
    "dtype": _dtype_tag(dtype),
    "forward_backend": "cutedsl",
    "B": B,
    "Hq": Nh_q,
    "Hkv": Nh_kv,
    "Nq": Nq,
    "Nkv": Nkv,
    "D": D,
    "causal": causal,
    "dropout_p": 0.0,
    "max_diff": _max_abs_diff(out_ffpa, out_sdpa),
    "mean_diff": _mean_abs_diff(out_ffpa, out_sdpa),
    "allclose": ok,
    "tolerance": tol,
    "warmup": warmup,
    "iters": iters,
    "ffpa_ms": ms_ffpa,
    "sdpa_ms": ms_sdpa,
    "ffpa_tflops": tflops_from_ms(flop_count, ms_ffpa),
    "sdpa_tflops": tflops_from_ms(flop_count, ms_sdpa),
    "speedup": ms_sdpa / ms_ffpa,
  }
  if print_result:
    print(_format_forward_result(row))
  return row


def _cutedsl_forward_for_backward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool) -> torch.Tensor:
  """Forward wrapper used by ``_time_backward_only`` for the CuTeDSL path."""
  return ffpa_attn_func(
    q,
    k,
    v,
    is_causal=causal,
    enable_gqa=q.size(1) != k.size(1),
    forward_backend="cutedsl",
  )


def _sdpa_forward_for_backward(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  *,
  causal: bool,
  Nh_q: int,
) -> torch.Tensor:
  """SDPA forward wrapper that matches the FFPA shapes for backward timing."""
  k_in, v_in = _expand_kv(k, v, Nh_q)
  kw = _make_sdpa_kwargs(causal, q.size(2), k_in.size(2))
  scale = 1.0 / math.sqrt(q.size(-1))
  return F.scaled_dot_product_attention(q, k_in, v_in, scale=scale, dropout_p=0.0, **kw)


def _run_cutedsl_backward_case(
  name: str,
  *,
  seed: int,
  B: int,
  Nh_q: int,
  Nh_kv: int,
  Nq: int,
  Nkv: int,
  D: int,
  causal: bool,
  apply_norm: bool,
  warmup: int,
  iters: int,
  print_result: bool = True,
) -> RESULT_ROW:
  """Run one backward case through CuTeDSL and SDPA, then time backward-only."""
  dtype = CUTEDSL_DTYPE
  torch.manual_seed(seed)
  q_raw = torch.randn(B, Nh_q, Nq, D, dtype=dtype, device="cuda")
  k_raw = torch.randn(B, Nh_kv, Nkv, D, dtype=dtype, device="cuda")
  v_raw = torch.randn(B, Nh_kv, Nkv, D, dtype=dtype, device="cuda")
  q_raw, k_raw, v_raw = _maybe_norm_qkv(q_raw, k_raw, v_raw, apply_norm)

  q = q_raw.detach().clone().requires_grad_(True)
  k = k_raw.detach().clone().requires_grad_(True)
  v = v_raw.detach().clone().requires_grad_(True)

  out_ffpa = ffpa_attn_func(
    q,
    k,
    v,
    is_causal=causal,
    enable_gqa=Nh_q != Nh_kv,
    forward_backend="cutedsl",
  )
  out_ffpa.sum().backward()
  dq_ffpa = q.grad.detach().clone()
  dk_ffpa = k.grad.detach().clone()
  dv_ffpa = v.grad.detach().clone()

  q_ref = q_raw.detach().clone().requires_grad_(True)
  k_ref = k_raw.detach().clone().requires_grad_(True)
  v_ref = v_raw.detach().clone().requires_grad_(True)
  k_in, v_in = _expand_kv(k_ref, v_ref, Nh_q)
  kw = _make_sdpa_kwargs(causal, Nq, Nkv)
  scale = 1.0 / math.sqrt(D)
  out_sdpa = F.scaled_dot_product_attention(q_ref, k_in, v_in, scale=scale, dropout_p=0.0, **kw)
  out_sdpa.sum().backward()
  dq_ref = q_ref.grad
  dk_ref = k_ref.grad
  dv_ref = v_ref.grad

  tol = 7.5e-2 if causal else 5e-2
  dq_err = _max_abs_diff(dq_ffpa, dq_ref)
  dk_err = _max_abs_diff(dk_ffpa, dk_ref)
  dv_err = _max_abs_diff(dv_ffpa, dv_ref)
  ok = (
    _tensor_allclose(dq_ffpa, dq_ref, tol) and _tensor_allclose(dk_ffpa, dk_ref, tol)
    and _tensor_allclose(dv_ffpa, dv_ref, tol)
  )

  grad_out = torch.ones_like(q_raw)
  ms_ffpa = _time_backward_only(
    lambda q_i, k_i, v_i: _cutedsl_forward_for_backward(q_i, k_i, v_i, causal=causal),
    q_raw,
    k_raw,
    v_raw,
    grad_out,
    warmup=warmup,
    iters=iters,
  )
  ms_sdpa = _time_backward_only(
    lambda q_i, k_i, v_i: _sdpa_forward_for_backward(q_i, k_i, v_i, causal=causal, Nh_q=Nh_q),
    q_raw,
    k_raw,
    v_raw,
    grad_out,
    warmup=warmup,
    iters=iters,
  )
  flop_count = attention_bwd_flops(B, Nh_q, Nq, Nkv, D, causal)

  row: RESULT_ROW = {
    "case_name": name,
    "dtype": _dtype_tag(dtype),
    "backward_backend": "cutedsl",
    "timing_mode": "backward-only",
    "B": B,
    "Hq": Nh_q,
    "Hkv": Nh_kv,
    "Nq": Nq,
    "Nkv": Nkv,
    "D": D,
    "causal": causal,
    "dropout_p": 0.0,
    "dq_err": dq_err,
    "dk_err": dk_err,
    "dv_err": dv_err,
    "allclose": ok,
    "tolerance": tol,
    "warmup": warmup,
    "iters": iters,
    "ffpa_ms": ms_ffpa,
    "sdpa_ms": ms_sdpa,
    "ffpa_tflops": tflops_from_ms(flop_count, ms_ffpa),
    "sdpa_tflops": tflops_from_ms(flop_count, ms_sdpa),
    "speedup": ms_sdpa / ms_ffpa,
  }
  if print_result:
    print(_format_backward_result(row))
  return row


def _run_cutedsl_examples(
  direction: str,
  *,
  args: argparse.Namespace,
  tasks: set[str] | None,
) -> list[RESULT_ROW]:
  """Iterate the cutedsl-compatible case specs and collect rows for one direction."""
  _validate_timing_args(args.warmup, args.iters)
  specs = _case_specs(args.N, args.H, tasks)
  print(
    f"\nRunning FFPA {direction} examples on the CuTeDSL D=512 SM90 backend, "
    f"dtype={_dtype_tag(CUTEDSL_DTYPE)}, apply_norm={args.norm}, "
    f"tasks={sorted(tasks) if tasks is not None else 'full'}, "
    f"warmup={args.warmup}, iters={args.iters}"
  )
  runner = _run_cutedsl_forward_case if direction == "forward" else _run_cutedsl_backward_case
  rows: list[RESULT_ROW] = []
  for spec in specs:
    rows.append(
      runner(
        spec["name"],
        seed=args.seed,
        B=args.B,
        Nh_q=spec["Nh_q"],
        Nh_kv=spec["Nh_kv"],
        Nq=spec["Nq"],
        Nkv=spec["Nkv"],
        D=HEAD_DIM,
        causal=spec.get("causal", False),
        apply_norm=args.norm,
        warmup=args.warmup,
        iters=args.iters,
      )
    )
  return rows


def _aggregate_speedups(
  rows: list[RESULT_ROW],
  direction: str,
  plot_cases: list[tuple[str, str]] | None = None,
) -> list[float] | None:
  """Aggregate per-dtype rows into the bar heights used by the plot.

  Only bf16 is exercised in this script, so the per-case reduction is a
  pass-through, but the shape is preserved for compatibility with the
  ``perf.py`` plotting layer.
  """
  active_plot_cases = PLOT_CASES if plot_cases is None else plot_cases
  active_case_names = {case_name for case_name, _ in active_plot_cases}
  case_to_speedups: dict[str, list[float]] = {case_name: [] for case_name, _ in active_plot_cases}
  for row in rows:
    if row["direction"] != direction:
      continue
    if row["case_name"] not in active_case_names:
      continue
    case_to_speedups[row["case_name"]].append(float(row["speedup"]))
  if not any(case_to_speedups.values()):
    return None
  values: list[float] = []
  for case_name, _ in active_plot_cases:
    speeds = case_to_speedups[case_name]
    values.append(float(np.amax(speeds)) if speeds else float("nan"))
  return values


def _aggregate_metric(
  rows: list[RESULT_ROW],
  direction: str,
  metric_key: str,
  plot_cases: list[tuple[str, str]] | None = None,
) -> list[float] | None:
  """Aggregate one numeric metric per case for plotting."""
  active_plot_cases = PLOT_CASES if plot_cases is None else plot_cases
  active_case_names = {case_name for case_name, _ in active_plot_cases}
  case_to_values: dict[str, list[float]] = {case_name: [] for case_name, _ in active_plot_cases}
  for row in rows:
    if row["direction"] != direction:
      continue
    if row["case_name"] not in active_case_names:
      continue
    value = row.get(metric_key)
    if value is None:
      continue
    case_to_values[row["case_name"]].append(float(value))
  if not any(case_to_values.values()):
    return None
  values: list[float] = []
  for case_name, _ in active_plot_cases:
    metric_values = case_to_values[case_name]
    values.append(float(np.amax(metric_values)) if metric_values else float("nan"))
  return values


def plot_speedup(
  forward_rows: list[RESULT_ROW],
  backward_rows: list[RESULT_ROW],
  *,
  device_name: str,
  B: int,
  H: int,
  N: int,
  D: int,
  output_path: Path,
  plot_cases: list[tuple[str, str]] | None = None,
) -> Path:
  """Render the speedup bar chart while preserving the legacy look."""
  active_plot_cases = PLOT_CASES if plot_cases is None else plot_cases
  fwd_speedups = _aggregate_speedups(forward_rows, "forward", active_plot_cases)
  bwd_speedups = _aggregate_speedups(backward_rows, "backward", active_plot_cases)
  has_forward = fwd_speedups is not None
  has_backward = bwd_speedups is not None
  if not has_forward and not has_backward:
    raise ValueError("No forward or backward rows were provided for plotting.")

  attn_types = [label for _, label in active_plot_cases]
  sdpa_speedups = [1.0] * len(attn_types)

  fig, ax = plt.subplots(figsize=(32, 12))
  width = 0.20
  x = np.arange(len(attn_types))

  def _autolabel(rects) -> None:
    for rect in rects:
      h = rect.get_height()
      if not np.isfinite(h):
        continue
      offset = 8 if h >= 1 else 20
      va_pos = "bottom" if h >= 1 else "top"
      ax.annotate(
        f"{h:.1f}x",
        xy=(rect.get_x() + rect.get_width() / 2, h),
        xytext=(0, offset),
        textcoords="offset points",
        ha="center",
        va=va_pos,
        fontsize=19,
        fontweight="bold",
      )

  if has_forward and has_backward:
    x_sdpa = x - width
    x_fwd = x
    x_bwd = x + width
  elif has_forward:
    x_sdpa = x - width / 2
    x_fwd = x + width / 2
    x_bwd = None
  else:
    x_sdpa = x - width / 2
    x_fwd = None
    x_bwd = x + width / 2

  rect_sdpa = ax.bar(
    x_sdpa,
    sdpa_speedups,
    width,
    label="SDPA Baseline",
    color="#b0b0b0",
    edgecolor="white",
    linewidth=1,
  )
  _autolabel(rect_sdpa)

  finite_values = [1.0]
  if x_fwd is not None and fwd_speedups is not None:
    rect_fwd = ax.bar(
      x_fwd,
      fwd_speedups,
      width,
      label="FFPA Forward (FWD)",
      color="#2171b5",
      edgecolor="white",
      linewidth=1,
    )
    _autolabel(rect_fwd)
    finite_values.extend(value for value in fwd_speedups if np.isfinite(value))

  if x_bwd is not None and bwd_speedups is not None:
    rect_bwd = ax.bar(
      x_bwd,
      bwd_speedups,
      width,
      label="FFPA Backward (BWD)",
      color="#fd493c",
      edgecolor="white",
      linewidth=1,
    )
    _autolabel(rect_bwd)
    finite_values.extend(value for value in bwd_speedups if np.isfinite(value))

  ax.axhline(y=1, color="#555555", linestyle="--", linewidth=2)
  ax.set_ylabel("Speedup Ratio (FFPA / SDPA)", fontsize=18)
  fig.suptitle(
    f"FFPA CuTeDSL vs SDPA Speedup ({_mode_suffix(has_forward, has_backward)}) | "
    f"{_display_device_name(device_name)} | B={B}, N={N}, H={H}, D={D}",
    fontsize=22,
    fontweight="bold",
    y=0.958,
  )
  ax.set_xticks(x)
  ax.set_xticklabels(attn_types, rotation=0, ha="center", fontsize=22, fontweight="bold")
  ax.tick_params(axis="y", labelsize=16)
  ymax = max(finite_values) if finite_values else 1.0
  ax.set_ylim(0, ymax * 1.17 if ymax > 0 else 1.0)
  ax.legend(
    fontsize=20,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.972),
    ncol=3 if has_forward and has_backward else 2,
    columnspacing=1.5,
    handletextpad=0.6,
    frameon=True,
  )
  ax.grid(axis="y", alpha=0.9)

  fig.tight_layout(rect=(0, 0, 1, 0.955))
  fig.savefig(output_path)
  plt.close(fig)
  return output_path


def plot_tflops(
  forward_rows: list[RESULT_ROW],
  backward_rows: list[RESULT_ROW],
  *,
  device_name: str,
  B: int,
  H: int,
  N: int,
  D: int,
  output_path: Path,
  plot_cases: list[tuple[str, str]] | None = None,
) -> Path | None:
  """Render the TFLOPS comparison bar chart."""
  active_plot_cases = PLOT_CASES if plot_cases is None else plot_cases
  fwd_ffpa_tflops = _aggregate_metric(forward_rows, "forward", "ffpa_tflops", plot_cases=active_plot_cases)
  fwd_sdpa_tflops = _aggregate_metric(forward_rows, "forward", "sdpa_tflops", plot_cases=active_plot_cases)
  bwd_ffpa_tflops = _aggregate_metric(backward_rows, "backward", "ffpa_tflops", plot_cases=active_plot_cases)
  bwd_sdpa_tflops = _aggregate_metric(backward_rows, "backward", "sdpa_tflops", plot_cases=active_plot_cases)
  has_forward = fwd_ffpa_tflops is not None and fwd_sdpa_tflops is not None
  has_backward = bwd_ffpa_tflops is not None and bwd_sdpa_tflops is not None
  if not has_forward and not has_backward:
    return None

  attn_types = [label for _, label in active_plot_cases]
  x = np.arange(len(attn_types))
  fig, ax = plt.subplots(figsize=(16, 12))

  def _autolabel(rects) -> None:
    for rect in rects:
      h = rect.get_height()
      if not np.isfinite(h):
        continue
      ax.annotate(
        format_tflops_short(float(h)),
        xy=(rect.get_x() + rect.get_width() / 2, h),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=19,
        fontweight="bold",
      )

  finite_values: list[float] = []
  if has_forward and has_backward:
    width = 0.18
    x_fwd_sdpa = x - 1.5 * width
    x_fwd_ffpa = x - 0.5 * width
    x_bwd_sdpa = x + 0.5 * width
    x_bwd_ffpa = x + 1.5 * width
  else:
    width = 0.20
    x_fwd_sdpa = x - width / 2 if has_forward else None
    x_fwd_ffpa = x + width / 2 if has_forward else None
    x_bwd_sdpa = x - width / 2 if has_backward else None
    x_bwd_ffpa = x + width / 2 if has_backward else None

  if has_forward and x_fwd_sdpa is not None and x_fwd_ffpa is not None:
    rect_fwd_sdpa = ax.bar(
      x_fwd_sdpa,
      fwd_sdpa_tflops,
      width,
      label="SDPA FWD",
      color=TFLOPS_FWD_SDPA_COLOR,
      edgecolor="white",
      linewidth=1,
    )
    rect_fwd_ffpa = ax.bar(
      x_fwd_ffpa,
      fwd_ffpa_tflops,
      width,
      label="FFPA FWD",
      color=TFLOPS_FWD_FFPA_COLOR,
      edgecolor="white",
      linewidth=1,
    )
    _autolabel(rect_fwd_sdpa)
    _autolabel(rect_fwd_ffpa)
    finite_values.extend(value for value in fwd_sdpa_tflops if np.isfinite(value))
    finite_values.extend(value for value in fwd_ffpa_tflops if np.isfinite(value))

  if has_backward and x_bwd_sdpa is not None and x_bwd_ffpa is not None:
    rect_bwd_sdpa = ax.bar(
      x_bwd_sdpa,
      bwd_sdpa_tflops,
      width,
      label="SDPA BWD",
      color=TFLOPS_BWD_SDPA_COLOR,
      edgecolor="white",
      linewidth=1,
    )
    rect_bwd_ffpa = ax.bar(
      x_bwd_ffpa,
      bwd_ffpa_tflops,
      width,
      label="FFPA BWD",
      color=TFLOPS_BWD_FFPA_COLOR,
      edgecolor="white",
      linewidth=1,
    )
    _autolabel(rect_bwd_sdpa)
    _autolabel(rect_bwd_ffpa)
    finite_values.extend(value for value in bwd_sdpa_tflops if np.isfinite(value))
    finite_values.extend(value for value in bwd_ffpa_tflops if np.isfinite(value))

  ax.set_ylabel("Throughput (TFLOPS)", fontsize=18)
  fig.suptitle(
    f"FFPA CuTeDSL vs SDPA TFLOPS ({_mode_suffix(has_forward, has_backward)}) | "
    f"{_display_device_name(device_name)} | B={B}, N={N}, H={H}, D={D}",
    fontsize=18,
    fontweight="bold",
    y=0.958,
  )
  ax.set_xticks(x)
  ax.set_xticklabels(attn_types, rotation=0, ha="center", fontsize=22, fontweight="bold")
  ax.tick_params(axis="y", labelsize=16)
  ymax = max(finite_values) if finite_values else 1.0
  ax.set_ylim(0, ymax * 1.10 if ymax > 0 else 1.0)
  ax.legend(
    fontsize=18,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.01),
    ncol=4,
    columnspacing=1.5,
    handletextpad=0.6,
    frameon=True,
  )
  ax.grid(axis="y", alpha=0.3)

  fig.tight_layout(rect=(0, 0, 1, 0.965))
  fig.savefig(output_path)
  plt.close(fig)
  return output_path


def _sort_rows(rows: list[RESULT_ROW]) -> list[RESULT_ROW]:
  """Sort rows by case order and dtype order."""
  case_rank = {case_name: index for index, (case_name, _) in enumerate(PLOT_CASES)}
  dtype_rank = {dtype: index for index, dtype in enumerate(DTYPE_ORDER)}
  return sorted(rows, key=lambda row: (case_rank[row["case_name"]], dtype_rank.get(row["dtype"], 999)))


def _allclose_marker(row: RESULT_ROW) -> str:
  """Convert the allclose field into the Markdown marker."""
  if row.get("allclose") is True:
    return "✅"
  if row.get("allclose") is False:
    return "❌"
  return "-"


def _markdown_table_columns(show_allclose: bool) -> tuple[str, str]:
  """Return Markdown header and alignment rows."""
  if show_allclose:
    return (
      "| Case | dtype | Nq/Nkv | allclose | FFPA / SDPA | TFLOPS | speedup |",
      "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    )
  return (
    "| Case | dtype | Nq/Nkv | FFPA / SDPA | TFLOPS | speedup |",
    "|:---:|:---:|:---:|:---:|:---:|:---:|",
  )


def _latency_cell(row: RESULT_ROW) -> str:
  """Format the latency cell for Markdown tables."""
  ffpa_ms = row.get("ffpa_ms")
  sdpa_ms = row.get("sdpa_ms")
  if ffpa_ms is None or sdpa_ms is None:
    return "- / -"
  return f"{ffpa_ms:.2f} / {sdpa_ms:.2f} ms"


def _tflops_cell(row: RESULT_ROW) -> str:
  """Format the TFLOPS cell for Markdown tables."""
  return f"{format_tflops_short(row.get('ffpa_tflops'))} / {format_tflops_short(row.get('sdpa_tflops'))}"


def _render_table(rows: list[RESULT_ROW], show_allclose: bool) -> list[str]:
  """Render one GFM benchmark table."""
  header, align = _markdown_table_columns(show_allclose)
  lines = [header, align]
  for row in _sort_rows(rows):
    if show_allclose:
      lines.append(
        "| "
        f"{row['case_name']} | {row['dtype']} | {row['Nq']}/{row['Nkv']} | {_allclose_marker(row)} | {_latency_cell(row)} | {_tflops_cell(row)} | {row['speedup']:.2f}x |"
      )
    else:
      lines.append(
        "| "
        f"{row['case_name']} | {row['dtype']} | {row['Nq']}/{row['Nkv']} | {_latency_cell(row)} | {_tflops_cell(row)} | {row['speedup']:.2f}x |"
      )
  return lines


def render_speedup_markdown(
  forward_rows: list[RESULT_ROW],
  backward_rows: list[RESULT_ROW],
  *,
  device_name: str,
  B: int,
  H: int,
  N: int,
  D: int,
  show_allclose: bool,
) -> str:
  """Render README-style Markdown benchmark tables."""
  lines = ["## Benchmark", "", f"Env: {device_name}, B={B}, N={N}, H={H}, D={D}."]
  lines.extend([
    "",
    "Backend: CuTeDSL D=512 SM90 fast-path (bf16 only). "
    "TFLOPS reports the theoretical dominant attention GEMM throughput only; "
    "forward and backward are computed separately from the measured latency.",
  ])
  if forward_rows:
    lines.extend(["", f"### Forward Pass ({SECTION_LABEL})", ""])
    lines.extend(_render_table(forward_rows, show_allclose))
  if backward_rows:
    lines.extend(["", f"### Backward Pass ({SECTION_LABEL})", ""])
    lines.extend(_render_table(backward_rows, show_allclose))
  return "\n".join(lines) + "\n"


def _benchmark_rows(args: argparse.Namespace) -> tuple[list[RESULT_ROW], list[RESULT_ROW]]:
  """Collect benchmark rows for the requested directions."""
  if not torch.cuda.is_available():
    raise SystemExit("CUDA is required to run the CuTeDSL benchmark.")

  tasks = _parse_tasks_arg(args.tasks)
  forward_rows: list[RESULT_ROW] = []
  backward_rows: list[RESULT_ROW] = []
  if args.forward:
    forward_rows = _decorate_rows("forward", _run_cutedsl_examples("forward", args=args, tasks=tasks))
  if args.backward:
    backward_rows = _decorate_rows("backward", _run_cutedsl_examples("backward", args=args, tasks=tasks))
  return forward_rows, backward_rows


def _require_sm90() -> None:
  """Fail fast on non-Hopper devices so users get an actionable error."""
  from ffpa_attn.cutedsl._wrappers import cutedsl_forward_available

  if not torch.cuda.is_available():
    raise SystemExit("CUDA is required: the CuTeDSL backend only runs on SM90 (Hopper) GPUs.")
  device = torch.device("cuda", torch.cuda.current_device())
  if not cutedsl_forward_available(device):
    major, minor = torch.cuda.get_device_capability(device)
    raise SystemExit(
      f"CuTeDSL backend requires SM90 (Hopper). Detected device "
      f"'{torch.cuda.get_device_name(device)}' with compute capability {major}.{minor}."
    )


def main() -> None:
  """Run the requested benchmark mode and emit plot plus Markdown outputs."""
  args = _parse_args()
  if not args.forward and not args.backward:
    raise SystemExit("At least one direction must remain enabled. Drop --no-fwd or --no-bwd.")

  _require_sm90()
  device_name = _device_name()
  tasks = _parse_tasks_arg(args.tasks)

  forward_rows, backward_rows = _benchmark_rows(args)

  plot_cases = _active_plot_cases(tasks)

  output_stem = _resolve_output_stem(args.save_path, device_name, args.B, args.H, args.N, HEAD_DIM)
  speedup_png_path = plot_speedup(
    forward_rows,
    backward_rows,
    device_name=device_name,
    B=args.B,
    H=args.H,
    N=args.N,
    D=HEAD_DIM,
    output_path=output_stem.with_name(f"{output_stem.name}.png"),
    plot_cases=plot_cases,
  )
  tflops_png_path = plot_tflops(
    forward_rows,
    backward_rows,
    device_name=device_name,
    B=args.B,
    H=args.H,
    N=args.N,
    D=HEAD_DIM,
    output_path=output_stem.with_name(f"{output_stem.name}_T.png"),
    plot_cases=plot_cases,
  )
  markdown = render_speedup_markdown(
    forward_rows,
    backward_rows,
    device_name=device_name,
    B=args.B,
    H=args.H,
    N=args.N,
    D=HEAD_DIM,
    show_allclose=args.show_allclose,
  )
  md_path = output_stem.with_suffix(".md")
  md_path.write_text(markdown, encoding="utf-8")

  print(f"\n{markdown}\n", end="")
  print(f"Saved speedup plot to {speedup_png_path}")
  if tflops_png_path is not None:
    print(f"Saved TFLOPS plot to {tflops_png_path}")
  print(f"Saved Markdown to {md_path}")


if __name__ == "__main__":
  main()

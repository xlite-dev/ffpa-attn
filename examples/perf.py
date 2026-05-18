"""Benchmark FFPA speedups and render plot plus Markdown tables.

Usage::

  CUDA_VISIBLE_DEVICES=0 python examples/perf.py
  CUDA_VISIBLE_DEVICES=0 python examples/perf.py --no-bwd --fwd-backend triton --tune fast
  CUDA_VISIBLE_DEVICES=0 python examples/perf.py --no-fwd --bwd-backend triton --tune max
  CUDA_VISIBLE_DEVICES=0 python examples/perf.py --fwd-backend triton --bwd-backend triton --tune fast
  CUDA_VISIBLE_DEVICES=0 python examples/perf.py --fwd-backend cutedsl --bwd-backend cutedsl

The cutedsl backend is SM90 (Hopper) only and locks D=512 / bf16. Selecting
``cutedsl`` on either ``--fwd-backend`` or ``--bwd-backend`` auto-pairs the
other side and restricts tasks to the cutedsl-compatible subset
(self-attn, cross-attn, gqa, causal).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
  sys.path.insert(0, str(EXAMPLES_DIR))

from attention_flops import format_tflops_short
from ffpa_attn_bwd import run_backward_examples
from ffpa_attn_fwd import run_forward_examples


def _parse_grad_v_dtype(arg: str) -> torch.dtype | None:
  """Parse the CLI grad-v-dtype option.

  :param arg: CLI value, ``"none"`` or ``"fp32"``.
  :return: ``None`` or ``torch.float32``.
  """
  if arg == "none":
    return None
  if arg == "fp32":
    return torch.float32
  raise ValueError(f"Unsupported grad-v-dtype={arg!r}; choose 'none' or 'fp32'.")


# Keep the exact legacy plotting style from tools/plot.py.
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PLOT_CASES: list[tuple[str, str]] = [
  ("self-attn", "self-attn(F/B)"),
  ("cross-attn", "cross-attn(F/B)"),
  ("decode-attn", "decode(Nq=1,F/B)"),
  ("gqa", "gqa(F/B)"),
  ("causal", "causal(F/B)"),
  ("attn-mask", "attn-mask(F/B)"),
  ("dropout", "dropout(F/B)"),
  ("non-aligned", "non-aligned(F/B)"),
]
SPEEDUP_PLOT_CASES: list[tuple[str, str]] = [
  ("self-attn", "self-attn(F/B)"),
  ("cross-attn", "cross-attn(F/B)"),
  ("decode-attn", "decode(Nq=1,F/B)"),
  ("gqa", "gqa(F/B)"),
  ("causal", "causal(F/B)"),
  ("attn-mask", "attn-mask(F/B)"),
  ("dropout", "dropout(F/B)"),
]
TFLOPS_PLOT_CASES: list[tuple[str, str]] = [
  ("self-attn", "self-attn(F/B)"),
  ("cross-attn", "cross-attn(F/B)"),
  ("gqa", "gqa(F/B)"),
  ("causal", "causal(F/B)"),
]
CASE_LABELS = dict(PLOT_CASES)
VALID_TASKS = tuple(case_name for case_name, _ in PLOT_CASES)
DTYPE_ORDER = ["fp16", "bf16"]
DEFAULT_OUTPUT_STEM = "ffpa_speedup"
DEFAULT_OUTPUT_DIR = Path(".tmp")
FALLBACK_DEVICE_NAME = "NVIDIA RTX 5090 Blackwell"
CUTEDSL_BACKEND = "cutedsl"
CUTEDSL_COMPAT_TASKS = frozenset({"self-attn", "cross-attn", "gqa", "causal", "non-aligned"})
CUTEDSL_DTYPES: tuple[torch.dtype, ...] = (torch.float16, torch.bfloat16)
CUTEDSL_HEAD_DIM = 512
CUTEDSL_OUTPUT_STEM = "ffpa_speedup_cutedsl"
CUTEDSL_SECTION_LABEL = "CuTeDSL (SM90 D=512)"
TFLOPS_FWD_SDPA_COLOR = "#b0b0b0"
TFLOPS_FWD_FFPA_COLOR = "#2171b5"
TFLOPS_BWD_SDPA_COLOR = "#f5a623"
TFLOPS_BWD_FFPA_COLOR = "#fd493c"
FALLBACK_SPEEDUPS: dict[str, dict[str, dict[str, float]]] = {
  "forward": {
    "self-attn": {
      "fp16": 2.06,
      "bf16": 2.08
    },
    "cross-attn": {
      "fp16": 1.86,
      "bf16": 1.87
    },
    "decode-attn": {
      "fp16": 2.86,
      "bf16": 2.85
    },
    "gqa": {
      "fp16": 2.06,
      "bf16": 2.09
    },
    "causal": {
      "fp16": 1.96,
      "bf16": 1.99
    },
    "attn-mask": {
      "fp16": 1.70,
      "bf16": 1.74
    },
    "dropout": {
      "fp16": 1.79,
      "bf16": 1.82
    },
    "non-aligned": {
      "fp16": 1.96,
      "bf16": 1.98
    },
  },
  "backward": {
    "self-attn": {
      "fp16": 2.34,
      "bf16": 2.49
    },
    "cross-attn": {
      "fp16": 2.57,
      "bf16": 2.51
    },
    "decode-attn": {
      "fp16": 2.97,
      "bf16": 3.11
    },
    "gqa": {
      "fp16": 2.32,
      "bf16": 2.46
    },
    "causal": {
      "fp16": 2.22,
      "bf16": 2.56
    },
    "attn-mask": {
      "fp16": 2.06,
      "bf16": 2.17
    },
    "dropout": {
      "fp16": 2.27,
      "bf16": 2.41
    },
    "non-aligned": {
      "fp16": 2.37,
      "bf16": 2.67
    },
  },
}
RESULT_ROW = dict[str, Any]


def _parse_args() -> argparse.Namespace:
  """Parse CLI arguments.

  :return: Parsed CLI namespace.
  """
  parser = argparse.ArgumentParser(
    description="Benchmark FFPA forward/backward cases and generate plot plus Markdown tables."
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
    "--show-fallback",
    action="store_true",
    help="Render the legacy hard-coded fallback plot and Markdown table instead of running real benchmarks.",
  )
  parser.add_argument(
    "--forward-backend",
    "--fwd-backend",
    choices=["cuda", "triton", "cutedsl"],
    default="triton",
    help="Forward backend used when --forward is enabled.",
  )
  parser.add_argument(
    "--backward-backend",
    "--bwd-backend",
    choices=["sdpa", "triton", "cutedsl"],
    default="triton",
    help="Backward backend used when --backward is enabled.",
  )
  parser.add_argument("--tune", choices=["fast", "max"], help="Enable Triton autotune with the selected search mode.")
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
  parser.add_argument("--D", type=int, default=512, help="Head dimension used by benchmark mode.")
  parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations used for timing.")
  parser.add_argument("--iters", type=int, default=10, help="Measured iterations used for timing.")
  parser.add_argument("--seed", type=int, default=42, help="RNG seed used by benchmark mode.")
  parser.add_argument(
    "--dtype",
    choices=["fp16", "bf16", "both"],
    default="both",
    help="Activation dtype to benchmark. 'both' (default) runs fp16+bf16; "
    "fp16/bf16 narrows to that single dtype.",
  )
  parser.add_argument(
    "--norm",
    action="store_true",
    help="Enable pre-attention LayerNorm on q/k/v for both FFPA and SDPA paths.",
  )
  parser.add_argument(
    "--enable-tma",
    action="store_true",
    help="Compatibility alias for --enable-fwd-tma --enable-bwd-tma.",
  )
  parser.add_argument(
    "--enable-ws",
    action="store_true",
    help="Compatibility alias for --enable-fwd-ws --enable-bwd-ws.",
  )
  parser.add_argument(
    "--enable-fwd-tma",
    action="store_true",
    help="Enable experimental SM90+ TMA forward path (silently falls back on unsupported devices).",
  )
  parser.add_argument(
    "--enable-bwd-tma",
    action="store_true",
    help="Enable experimental SM90+ TMA backward path (silently falls back on unsupported devices).",
  )
  parser.add_argument(
    "--enable-fwd-ws",
    action="store_true",
    help="Force warp-specialized configs for the experimental SM90+ TMA forward path.",
  )
  parser.add_argument(
    "--enable-bwd-ws",
    action="store_true",
    help="Force warp-specialized configs for the experimental SM90+ TMA backward path.",
  )
  parser.add_argument(
    "--grad-v-storage-dtype",
    "--grad-v-dtype",
    choices=["none", "fp32"],
    default="none",
    help="Optional Triton backward dV storage dtype forwarded to the example runners.",
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
  return _resolve_directional_cli_flags(parser.parse_args())


def _resolve_directional_cli_flags(args: argparse.Namespace) -> argparse.Namespace:
  """Resolve legacy global TMA/WS flags into directional benchmark flags."""
  if args.enable_tma:
    args.enable_fwd_tma = True
    args.enable_bwd_tma = True
  if args.enable_ws:
    args.enable_fwd_ws = True
    args.enable_bwd_ws = True
  return args


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


def _active_plot_cases(tasks: set[str] | None, *, kind: str = "speedup") -> list[tuple[str, str]]:
  """Return plot cases filtered by an optional task set.

  :param tasks: Optional selected case names.
  :param kind: ``"speedup"`` (bar chart, omits non-aligned), ``"tflops"`` (TFLOPS
      chart), or ``"all"`` (full case list used by the Markdown sort order).
  :return: Ordered plot case list.
  """
  if kind == "speedup":
    source = SPEEDUP_PLOT_CASES
  elif kind == "tflops":
    source = TFLOPS_PLOT_CASES
  elif kind == "all":
    source = PLOT_CASES
  else:
    raise ValueError(f"Unknown plot kind={kind!r}; choose 'speedup', 'tflops', or 'all'.")
  if tasks is None:
    return list(source)
  return [(case_name, label) for case_name, label in source if case_name in tasks]


def _device_name() -> str:
  """Return the active CUDA device name when available.

  :return: CUDA device name or a fallback label.
  """
  if torch.cuda.is_available():
    return torch.cuda.get_device_name(torch.cuda.current_device())
  return "CUDA Unavailable"


def _display_device_name(device_name: str) -> str:
  """Rename H20Z → H200 for cutedsl plot titles only; filenames stay raw."""
  return re.sub(r"H20Z", "H200", device_name, flags=re.IGNORECASE)


def _require_sm90() -> None:
  """Fail fast on non-Hopper devices when the cutedsl backend is selected."""
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


def _resolve_cutedsl_backends(args: argparse.Namespace) -> bool:
  """Auto-pair cutedsl backends; reject mixing cutedsl with a non-cutedsl peer.

  Auto-promotion only fires when the peer side is still at its default
  ("triton"), so an explicit ``--backward-backend sdpa`` alongside
  ``--forward-backend cutedsl`` raises instead of being silently overridden.
  """
  fwd, bwd = args.forward_backend, args.backward_backend
  if CUTEDSL_BACKEND not in {fwd, bwd}:
    return False
  if fwd == CUTEDSL_BACKEND and bwd != CUTEDSL_BACKEND:
    if bwd != "triton":
      raise SystemExit(f"--forward-backend cutedsl requires --backward-backend cutedsl; got {bwd!r}.")
    args.backward_backend = CUTEDSL_BACKEND
  if bwd == CUTEDSL_BACKEND and fwd != CUTEDSL_BACKEND:
    if fwd != "triton":
      raise SystemExit(f"--backward-backend cutedsl requires --forward-backend cutedsl; got {fwd!r}.")
    args.forward_backend = CUTEDSL_BACKEND
  return True


def _slugify_device_name(device_name: str) -> str:
  """Convert a device name into a filesystem-friendly slug.

  :param device_name: Human-readable device name.
  :return: Lowercase slug safe for filenames.
  """
  slug = re.sub(r"[^0-9A-Za-z]+", "-", device_name.strip().lower())
  slug = re.sub(r"-+", "-", slug).strip("-")
  return slug or "unknown-device"


def _output_stem(device_name: str, B: int, H: int, N: int, D: int, *, cutedsl: bool = False) -> Path:
  """Build the output stem shared by the PNG and Markdown files.

  :param device_name: Device name used in the run.
  :param B: Batch size.
  :param H: Head count.
  :param N: Sequence length.
  :param D: Head dimension.
  :param cutedsl: When ``True``, switch the prefix to keep cutedsl artifacts
      from clobbering the standard ones.
  :return: Output stem without extension.
  """
  prefix = CUTEDSL_OUTPUT_STEM if cutedsl else DEFAULT_OUTPUT_STEM
  device_slug = _slugify_device_name(device_name)
  return Path(f"{prefix}_{device_slug}_B{B}_H{H}_N{N}_D{D}")


def _resolve_output_stem(
  save_path: Path | None,
  device_name: str,
  B: int,
  H: int,
  N: int,
  D: int,
  *,
  cutedsl: bool = False,
) -> Path:
  """Resolve the final output stem, optionally rooted at ``save_path``.

  :param save_path: Optional output directory.
  :param device_name: Device name used in the run.
  :param B: Batch size.
  :param H: Head count.
  :param N: Sequence length.
  :param D: Head dimension.
  :param cutedsl: Forwarded to :func:`_output_stem` for prefix selection.
  :return: Output stem without extension.
  """
  default_stem = _output_stem(device_name, B, H, N, D, cutedsl=cutedsl)
  output_dir = DEFAULT_OUTPUT_DIR if save_path is None else save_path
  output_dir.mkdir(parents=True, exist_ok=True)
  return output_dir / default_stem.name


def _case_shape(case_name: str, sequence_length: int) -> tuple[int, int]:
  """Return the benchmark shape shown in Markdown for one case.

  :param case_name: Canonical case name.
  :param sequence_length: Base sequence length.
  :return: ``(Nq, Nkv)`` pair.
  """
  if case_name == "cross-attn":
    return 1024, sequence_length
  if case_name == "decode-attn":
    return 1, sequence_length
  if case_name == "attn-mask":
    mask_n = max(sequence_length, 512)
    return mask_n, mask_n
  if case_name == "non-aligned":
    non_aligned_n = sequence_length - 1 if sequence_length > 1 else sequence_length
    return non_aligned_n, non_aligned_n
  return sequence_length, sequence_length


def _mode_suffix(has_forward: bool, has_backward: bool) -> str:
  """Build the title suffix that matches the legacy title style.

  :param has_forward: Whether forward data is present.
  :param has_backward: Whether backward data is present.
  :return: Title mode suffix.
  """
  if has_forward and has_backward:
    return "FWD & BWD"
  if has_forward:
    return "FWD"
  return "BWD"


def _forward_section_label(backend: str, tune_mode: str | None, fallback: bool) -> str:
  """Describe the forward data source for Markdown headings.

  :param backend: Forward backend.
  :param tune_mode: Triton autotune mode.
  :param fallback: Whether fallback hard-coded data is used.
  :return: Human-readable section label.
  """
  if fallback:
    return "Fallback hard-coded data"
  if backend == "cuda":
    return "Legacy CUDA"
  if backend == CUTEDSL_BACKEND:
    return CUTEDSL_SECTION_LABEL
  if tune_mode is not None:
    return f"Triton w/ autotune ({tune_mode})"
  return "Triton"


def _backward_section_label(backend: str, tune_mode: str | None, fallback: bool) -> str:
  """Describe the backward data source for Markdown headings.

  :param backend: Backward backend.
  :param tune_mode: Triton autotune mode.
  :param fallback: Whether fallback hard-coded data is used.
  :return: Human-readable section label.
  """
  if fallback:
    return "Fallback hard-coded data"
  if backend == "sdpa":
    return "SDPA backward"
  if backend == CUTEDSL_BACKEND:
    return CUTEDSL_SECTION_LABEL
  if tune_mode is not None:
    return f"Triton w/ autotune ({tune_mode})"
  return "Triton"


def _decorate_rows(direction: str, rows: list[dict[str, Any]]) -> list[RESULT_ROW]:
  """Attach the direction field to benchmark results.

  :param direction: ``forward`` or ``backward``.
  :param rows: Raw rows returned by the example helper.
  :return: Decorated result rows.
  """
  return [{"direction": direction, **row} for row in rows]


def _build_fallback_rows(
  B: int,
  H: int,
  N: int,
  D: int,
  tasks: set[str] | None = None,
) -> tuple[list[RESULT_ROW], list[RESULT_ROW]]:
  """Build structured rows for the legacy hard-coded plot data.

  :param B: Batch size shown in metadata.
  :param H: Base head count shown in metadata.
  :param N: Base sequence length shown in metadata.
  :param D: Head dimension shown in metadata.
  :param tasks: Optional case-name filter.
  :return: ``(forward_rows, backward_rows)``.
  """
  forward_rows: list[RESULT_ROW] = []
  backward_rows: list[RESULT_ROW] = []
  for direction, target in (("forward", forward_rows), ("backward", backward_rows)):
    for case_name, _ in PLOT_CASES:
      if tasks is not None and case_name not in tasks:
        continue
      nq, nkv = _case_shape(case_name, N)
      for dtype in DTYPE_ORDER:
        target.append({
          "direction": direction,
          "case_name": case_name,
          "dtype": dtype,
          "B": B,
          "Hq": H,
          "Hkv": H,
          "Nq": nq,
          "Nkv": nkv,
          "D": D,
          "allclose": None,
          "ffpa_ms": None,
          "sdpa_ms": None,
          "speedup": FALLBACK_SPEEDUPS[direction][case_name][dtype],
          "forward_backend": "hard-coded" if direction == "forward" else None,
          "backward_backend": "hard-coded" if direction == "backward" else None,
          "dropout_p": 0.1 if case_name == "dropout" else 0.0,
          "causal": case_name == "causal",
        })
  return forward_rows, backward_rows


def _aggregate_speedups(
  rows: list[RESULT_ROW],
  direction: str,
  plot_cases: list[tuple[str, str]] | None = None,
) -> list[float] | None:
  """Aggregate per-dtype rows into the bar heights used by the plot.

  Prefers the bf16 value when present so the bar chart compares against a
  single canonical dtype; falls back to the max of remaining dtypes when bf16
  is absent (e.g. ``--dtype fp16``).

  :param rows: Structured rows for one or both directions.
  :param direction: ``forward`` or ``backward``.
  :param plot_cases: Ordered cases to aggregate for plotting.
  :return: Aggregated bar heights, or ``None`` when the direction is absent.
  """
  active_plot_cases = PLOT_CASES if plot_cases is None else plot_cases
  active_case_names = {case_name for case_name, _ in active_plot_cases}
  case_to_speedups: dict[str, dict[str, float]] = {case_name: {} for case_name, _ in active_plot_cases}
  for row in rows:
    if row["direction"] != direction:
      continue
    if row["case_name"] not in active_case_names:
      continue
    case_to_speedups[row["case_name"]][row["dtype"]] = float(row["speedup"])
  if not any(case_to_speedups.values()):
    return None
  values: list[float] = []
  for case_name, _ in active_plot_cases:
    dtyped = case_to_speedups[case_name]
    if "bf16" in dtyped:
      values.append(dtyped["bf16"])
    elif dtyped:
      values.append(float(np.amax(list(dtyped.values()))))
    else:
      values.append(float("nan"))
  return values


def _aggregate_metric(
  rows: list[RESULT_ROW],
  direction: str,
  metric_key: str,
  plot_cases: list[tuple[str, str]] | None = None,
) -> list[float] | None:
  """Aggregate one numeric metric per case for plotting.

  Prefers the bf16 value when present (single canonical dtype on the bar
  chart); falls back to the max of remaining dtypes when bf16 is absent
  (e.g. ``--dtype fp16``).

  :param rows: Structured rows for one or both directions.
  :param direction: ``forward`` or ``backward``.
  :param metric_key: Row field containing the numeric metric.
  :param plot_cases: Ordered cases to aggregate for plotting.
  :return: Aggregated bar heights, or ``None`` when the direction is absent.
  """
  active_plot_cases = PLOT_CASES if plot_cases is None else plot_cases
  active_case_names = {case_name for case_name, _ in active_plot_cases}
  case_to_values: dict[str, dict[str, float]] = {case_name: {} for case_name, _ in active_plot_cases}
  for row in rows:
    if row["direction"] != direction:
      continue
    if row["case_name"] not in active_case_names:
      continue
    value = row.get(metric_key)
    if value is None:
      continue
    case_to_values[row["case_name"]][row["dtype"]] = float(value)
  if not any(case_to_values.values()):
    return None
  values: list[float] = []
  for case_name, _ in active_plot_cases:
    dtyped = case_to_values[case_name]
    if "bf16" in dtyped:
      values.append(dtyped["bf16"])
    elif dtyped:
      values.append(float(np.amax(list(dtyped.values()))))
    else:
      values.append(float("nan"))
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
  cutedsl: bool = False,
) -> Path:
  """Render the speedup bar chart while preserving the legacy look.

  :param forward_rows: Forward result rows.
  :param backward_rows: Backward result rows.
  :param device_name: Device name shown in the title.
  :param B: Batch size shown in the title.
  :param H: Head count shown in the title.
  :param N: Sequence length shown in the title.
  :param D: Head dimension shown in the title.
  :param output_path: Output PNG path.
  :param plot_cases: Ordered cases to include in the plot.
  :param cutedsl: When ``True``, swap the title prefix to "FFPA CuTeDSL vs
      SDPA Speedup" and apply the H20Z → H200 display rename.
  :return: Saved PNG path.
  """
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
  title_prefix = "FFPA CuTeDSL vs SDPA Speedup" if cutedsl else "FFPA vs SDPA Speedup"
  title_device = _display_device_name(device_name) if cutedsl else device_name
  fig.suptitle(
    f"{title_prefix} ({_mode_suffix(has_forward, has_backward)}) | {title_device} | B={B}, N={N}, H={H}, D={D}",
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
  cutedsl: bool = False,
) -> Path | None:
  """Render the TFLOPS comparison bar chart.

  :param forward_rows: Forward result rows.
  :param backward_rows: Backward result rows.
  :param device_name: Device name shown in the title.
  :param B: Batch size shown in the title.
  :param H: Head count shown in the title.
  :param N: Sequence length shown in the title.
  :param D: Head dimension shown in the title.
  :param output_path: Output PNG path.
  :param plot_cases: Ordered cases to include in the plot.
  :param cutedsl: When ``True``, swap the title prefix to "FFPA CuTeDSL vs
      SDPA TFLOPS" and apply the H20Z → H200 display rename.
  :return: Saved PNG path, or ``None`` when no TFLOPS data is available.
  """
  active_plot_cases = TFLOPS_PLOT_CASES if plot_cases is None else plot_cases
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
  title_prefix = "FFPA CuTeDSL vs SDPA TFLOPS" if cutedsl else "FFPA vs SDPA TFLOPS"
  title_device = _display_device_name(device_name) if cutedsl else device_name
  fig.suptitle(
    f"{title_prefix} ({_mode_suffix(has_forward, has_backward)}) | {title_device} | B={B}, N={N}, H={H}, D={D}",
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
  """Sort rows by case order and dtype order.

  :param rows: Structured result rows.
  :return: Sorted rows.
  """
  case_rank = {case_name: index for index, (case_name, _) in enumerate(PLOT_CASES)}
  dtype_rank = {dtype: index for index, dtype in enumerate(DTYPE_ORDER)}
  return sorted(rows, key=lambda row: (case_rank[row["case_name"]], dtype_rank.get(row["dtype"], 999)))


def _allclose_marker(row: RESULT_ROW) -> str:
  """Convert the allclose field into the Markdown marker.

  :param row: Structured result row.
  :return: ``✅``, ``❌``, or ``-``.
  """
  if row.get("allclose") is True:
    return "✅"
  if row.get("allclose") is False:
    return "❌"
  return "-"


def _markdown_table_columns(show_allclose: bool) -> tuple[str, str]:
  """Return Markdown header and alignment rows.

  :param show_allclose: Whether to include the allclose column.
  :return: ``(header, align)`` rows.
  """
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
  """Format the latency cell for Markdown tables.

  :param row: Structured result row.
  :return: Markdown cell content.
  """
  ffpa_ms = row.get("ffpa_ms")
  sdpa_ms = row.get("sdpa_ms")
  if ffpa_ms is None or sdpa_ms is None:
    return "- / -"
  return f"{ffpa_ms:.2f} / {sdpa_ms:.2f} ms"


def _tflops_cell(row: RESULT_ROW) -> str:
  """Format the TFLOPS cell for Markdown tables.

  :param row: Structured result row.
  :return: Markdown cell content.
  """
  return f"{format_tflops_short(row.get('ffpa_tflops'))} / {format_tflops_short(row.get('sdpa_tflops'))}"


def _render_table(rows: list[RESULT_ROW], show_allclose: bool) -> list[str]:
  """Render one GFM benchmark table.

  :param rows: Structured result rows for one direction.
  :param show_allclose: Whether to include the allclose column.
  :return: Markdown lines for the table.
  """
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
  forward_backend: str,
  backward_backend: str,
  tune_mode: str | None,
  fallback: bool,
  show_allclose: bool,
  cutedsl: bool = False,
) -> str:
  """Render README-style Markdown benchmark tables.

  :param forward_rows: Forward result rows.
  :param backward_rows: Backward result rows.
  :param device_name: Device name shown in the metadata line.
  :param B: Batch size shown in the metadata line.
  :param H: Head count shown in the metadata line.
  :param N: Sequence length shown in the metadata line.
  :param D: Head dimension shown in the metadata line.
  :param forward_backend: Selected forward backend.
  :param backward_backend: Selected backward backend.
  :param tune_mode: Triton autotune mode.
  :param fallback: Whether fallback hard-coded data is used.
  :param show_allclose: Whether to include the allclose column.
  :return: Markdown document fragment.
  """
  lines = ["## Benchmark", "", f"Env: {device_name}, B={B}, N={N}, H={H}, D={D}."]
  if cutedsl:
    lines.extend([
      "",
      "Backend: CuTeDSL D=512 SM90 fast-path (fp16/bf16 forward, bf16-only backward). "
      "TFLOPS reports the theoretical dominant attention GEMM throughput only; "
      "forward and backward are computed separately from the measured latency.",
    ])
  else:
    lines.extend([
      "",
      "TFLOPS reports the theoretical dominant attention GEMM throughput only; "
      "forward and backward are computed separately from the measured latency.",
    ])
  if fallback:
    lines.extend([
      "",
      "Note: fallback mode reuses hard-coded plot speedups only, so FFPA / SDPA latency, TFLOPS, and allclose are unavailable."
    ])
  if forward_rows:
    lines.extend(["", f"### Forward Pass ({_forward_section_label(forward_backend, tune_mode, fallback)})", ""])
    lines.extend(_render_table(forward_rows, show_allclose))
  if backward_rows:
    lines.extend(["", f"### Backward Pass ({_backward_section_label(backward_backend, tune_mode, fallback)})", ""])
    lines.extend(_render_table(backward_rows, show_allclose))
  return "\n".join(lines) + "\n"


def _filter_cutedsl_tasks(tasks: set[str] | None) -> set[str]:
  """Intersect a user-supplied task set with the cutedsl-compatible cases.

  ``None`` (no ``--tasks``) becomes the full cutedsl-compatible set.
  Unsupported requests are dropped with a stderr note; an empty intersection
  raises ``SystemExit``.
  """
  if tasks is None:
    return set(CUTEDSL_COMPAT_TASKS)
  rejected = sorted(tasks - CUTEDSL_COMPAT_TASKS)
  if rejected:
    print(
      f"[cutedsl] Skipping tasks unsupported by the CuTeDSL backend: {rejected}. "
      f"Supported: {sorted(CUTEDSL_COMPAT_TASKS)}.",
      file=sys.stderr,
    )
  kept = tasks & CUTEDSL_COMPAT_TASKS
  if not kept:
    raise SystemExit("No requested tasks are supported by the CuTeDSL backend.")
  return kept


def _benchmark_rows(
  args: argparse.Namespace,
  *,
  tasks: set[str] | None,
  dtypes: tuple[torch.dtype, ...],
) -> tuple[list[RESULT_ROW], list[RESULT_ROW]]:
  """Collect benchmark rows for the requested directions.

  :param args: Parsed CLI arguments.
  :param tasks: Pre-filtered case-name set (already restricted to backend-compat cases).
  :param dtypes: Activation dtypes to iterate.
  :return: ``(forward_rows, backward_rows)``.
  """
  if not torch.cuda.is_available():
    raise SystemExit("CUDA is required when --forward or --backward is requested.")

  tune_mode = args.tune
  grad_v_dtype = _parse_grad_v_dtype(args.grad_v_storage_dtype)
  forward_rows: list[RESULT_ROW] = []
  backward_rows: list[RESULT_ROW] = []
  if args.forward:
    forward_rows = _decorate_rows(
      "forward",
      run_forward_examples(
        B=args.B,
        H=args.H,
        N=args.N,
        D=args.D,
        seed=args.seed,
        apply_norm=args.norm,
        forward_backend=args.forward_backend,
        triton_autotune=args.forward_backend == "triton" and tune_mode is not None,
        triton_autotune_mode=tune_mode or "fast",
        triton_backward_grad_v_storage_dtype=grad_v_dtype,
        warmup=args.warmup,
        iters=args.iters,
        print_results=True,
        enable_tma=args.enable_fwd_tma,
        enable_ws=args.enable_fwd_ws,
        tasks=tasks,
        dtypes=dtypes,
      ),
    )
  if args.backward:
    bwd_dtypes = dtypes
    if args.backward_backend == CUTEDSL_BACKEND and torch.float16 in bwd_dtypes:
      filtered = tuple(d for d in bwd_dtypes if d != torch.float16)
      if not filtered:
        raise SystemExit(
          "Selected --dtype fp16 with CuTeDSL backward, but CuTeDSL "
          "backward only supports bf16. Use --dtype bf16/both or --no-bwd."
        )
      print(
        "[cutedsl] Skipping fp16 backward: bf16-only (known fp16 dQ launch "
        "failure in src/ffpa_attn/cutedsl/_interface.py).",
        file=sys.stderr,
      )
      bwd_dtypes = filtered
    backward_rows = _decorate_rows(
      "backward",
      run_backward_examples(
        B=args.B,
        H=args.H,
        N=args.N,
        D=args.D,
        seed=args.seed,
        apply_norm=args.norm,
        backward_backend=args.backward_backend,
        timing_mode="backward-only",
        triton_autotune=args.backward_backend == "triton" and tune_mode is not None,
        triton_autotune_mode=tune_mode or "fast",
        triton_backward_grad_v_storage_dtype=grad_v_dtype,
        enable_tma=args.enable_bwd_tma,
        enable_ws=args.enable_bwd_ws,
        warmup=args.warmup,
        iters=args.iters,
        print_results=True,
        tasks=tasks,
        dtypes=bwd_dtypes,
      ),
    )
  return forward_rows, backward_rows


def main() -> None:
  """Run the requested benchmark mode and emit plot plus Markdown outputs."""
  args = _parse_args()
  fallback = args.show_fallback
  is_cutedsl = _resolve_cutedsl_backends(args)
  if is_cutedsl:
    if fallback:
      raise SystemExit("--show-fallback is not compatible with the cutedsl backend.")
    if args.D != CUTEDSL_HEAD_DIM:
      print(
        f"[cutedsl] Forcing --D from {args.D} to {CUTEDSL_HEAD_DIM} (cutedsl is D=512 only).",
        file=sys.stderr,
      )
      args.D = CUTEDSL_HEAD_DIM
    _require_sm90()

  device_name = FALLBACK_DEVICE_NAME if fallback else _device_name()
  tasks = _parse_tasks_arg(args.tasks)
  if is_cutedsl:
    tasks = _filter_cutedsl_tasks(tasks)
  dtypes = CUTEDSL_DTYPES if is_cutedsl else (torch.float16, torch.bfloat16)
  if args.dtype == "fp16":
    dtypes = tuple(d for d in dtypes if d == torch.float16)
  elif args.dtype == "bf16":
    dtypes = tuple(d for d in dtypes if d == torch.bfloat16)

  if not fallback and not args.forward and not args.backward:
    raise SystemExit("At least one direction must remain enabled. Use default settings, --no-fwd, or --no-bwd.")

  if fallback:
    forward_rows, backward_rows = _build_fallback_rows(args.B, args.H, args.N, args.D, tasks=tasks)
  else:
    forward_rows, backward_rows = _benchmark_rows(args, tasks=tasks, dtypes=dtypes)

  speedup_plot_cases = _active_plot_cases(tasks, kind="speedup")
  tflops_plot_cases = _active_plot_cases(tasks, kind="tflops")

  output_stem = _resolve_output_stem(
    args.save_path,
    device_name,
    args.B,
    args.H,
    args.N,
    args.D,
    cutedsl=is_cutedsl,
  )
  speedup_png_path = plot_speedup(
    forward_rows,
    backward_rows,
    device_name=device_name,
    B=args.B,
    H=args.H,
    N=args.N,
    D=args.D,
    output_path=output_stem.with_name(f"{output_stem.name}.png"),  # speedup
    plot_cases=speedup_plot_cases,
    cutedsl=is_cutedsl,
  )
  tflops_png_path = None if fallback else plot_tflops(
    forward_rows,
    backward_rows,
    device_name=device_name,
    B=args.B,
    H=args.H,
    N=args.N,
    D=args.D,
    output_path=output_stem.with_name(f"{output_stem.name}_T.png"),  # tflops
    plot_cases=tflops_plot_cases,
    cutedsl=is_cutedsl,
  )
  markdown = render_speedup_markdown(
    forward_rows,
    backward_rows,
    device_name=device_name,
    B=args.B,
    H=args.H,
    N=args.N,
    D=args.D,
    forward_backend=args.forward_backend,
    backward_backend=args.backward_backend,
    tune_mode=args.tune,
    fallback=fallback,
    show_allclose=args.show_allclose,
    cutedsl=is_cutedsl,
  )
  md_path = output_stem.with_suffix(".md")
  md_path.write_text(markdown, encoding="utf-8")

  print(f"\n{markdown}\n", end="")
  print(f"Saved speedup plot to {speedup_png_path}")
  if tflops_png_path is not None:
    print(f"Saved TFLOPS plot to {tflops_png_path}")
  elif fallback:
    print("Skipped TFLOPS plot in fallback mode because no TFLOPS data is available.")
  print(f"Saved Markdown to {md_path}")


if __name__ == "__main__":
  main()

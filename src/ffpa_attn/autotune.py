"""Generate persistent FFPA Triton tuned configs.

Example:

.. code-block:: bash

   python -m ffpa_attn.autotune --mode fast --B 1 --H 32 --overwrite
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import triton

from ffpa_attn import __version__, ffpa_attn_func
from ffpa_attn.triton._autotune_utils import bucket_autotune_seqlen
from ffpa_attn.triton._ffpa_bwd import _get_bwd_autotune, _get_decode_bwd_stage1_autotune, _get_pre_autotune
from ffpa_attn.triton._ffpa_fwd import _get_decode_fwd_stage1_autotune, _get_decode_num_splits, _get_fwd_autotune
from ffpa_attn.triton._persistent_autotune import (
  DEFAULT_HEADDIMS,
  DEFAULT_SEQLENS,
  SCHEMA_VERSION,
  config_from_triton_config,
  default_config_dir,
  device_config_path,
  max_configs_from_env,
  sanitize_device_name,
  write_config_file,
)


@dataclass(frozen=True)
class TuneTask:
  """One runtime shape to autotune.

  :param direction: ``"forward"`` or ``"backward"``.
  :param dtype: Activation dtype.
  :param headdim: Head dimension.
  :param seqlen_q: Query sequence length.
  :param seqlen_k: KV sequence length.
  :param causal: Whether to apply causal masking.
  """

  direction: str
  dtype: torch.dtype
  headdim: int
  seqlen_q: int
  seqlen_k: int
  causal: bool


def _parse_dtypes(value: str) -> list[torch.dtype]:
  mapping = {"fp16": torch.float16, "bf16": torch.bfloat16}
  dtypes: list[torch.dtype] = []
  for item in value.split(","):
    key = item.strip().lower()
    if key not in mapping:
      raise argparse.ArgumentTypeError(f"Unsupported dtype {item!r}; choose from fp16,bf16")
    if mapping[key] not in dtypes:
      dtypes.append(mapping[key])
  if not dtypes:
    raise argparse.ArgumentTypeError("At least one dtype is required")
  return dtypes


def _dtype_schema_name(dtype: torch.dtype) -> str:
  if dtype == torch.float16:
    return "fp16"
  if dtype == torch.bfloat16:
    return "bf16"
  raise ValueError(f"Unsupported dtype {dtype!r}")


def _available_seqlens() -> list[int]:
  total_memory = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
  if total_memory >= 48 * 1024**3:
    return list(DEFAULT_SEQLENS)
  return [value for value in DEFAULT_SEQLENS if value < 16384]


def _iter_forward_tasks(dtypes: list[torch.dtype], seqlens: list[int]) -> list[TuneTask]:
  tasks: list[TuneTask] = []
  prefill_seqlens = [value for value in seqlens if value >= 512]
  decode_kv_seqlens = prefill_seqlens
  for dtype in dtypes:
    for headdim in DEFAULT_HEADDIMS:
      for causal in (False, True):
        for seqlen_k in decode_kv_seqlens:
          tasks.append(TuneTask("forward", dtype, headdim, 1, seqlen_k, causal))
        for seqlen_q in prefill_seqlens:
          for seqlen_k in prefill_seqlens:
            if causal and seqlen_k < seqlen_q:
              continue
            tasks.append(TuneTask("forward", dtype, headdim, seqlen_q, seqlen_k, causal))
  return tasks


def _iter_backward_tasks(dtypes: list[torch.dtype], seqlens: list[int]) -> list[TuneTask]:
  tasks: list[TuneTask] = []
  prefill_seqlens = [value for value in seqlens if value >= 512]
  decode_query_seqlens = [1, 4]
  for dtype in dtypes:
    for headdim in DEFAULT_HEADDIMS:
      for causal in (False, True):
        for seqlen_q in decode_query_seqlens:
          for seqlen_k in prefill_seqlens:
            tasks.append(TuneTask("backward", dtype, headdim, seqlen_q, seqlen_k, causal))
        for seqlen_q in prefill_seqlens:
          for seqlen_k in prefill_seqlens:
            if causal and seqlen_k < seqlen_q:
              continue
            tasks.append(TuneTask("backward", dtype, headdim, seqlen_q, seqlen_k, causal))
  return tasks


def _limit_tasks(tasks: list[TuneTask]) -> list[TuneTask]:
  limit = max_configs_from_env()
  if limit is None:
    return tasks
  return tasks[:limit]


def _make_tensors(task: TuneTask, batch: int, heads: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  shape_q = (batch, heads, task.seqlen_q, task.headdim)
  shape_kv = (batch, heads, task.seqlen_k, task.headdim)
  q = torch.randn(shape_q, device="cuda", dtype=task.dtype, requires_grad=task.direction == "backward")
  k = torch.randn(shape_kv, device="cuda", dtype=task.dtype, requires_grad=task.direction == "backward")
  v = torch.randn(shape_kv, device="cuda", dtype=task.dtype, requires_grad=task.direction == "backward")
  return q, k, v


def _entry_base(task: TuneTask, mode: str, kernel: str, config: dict[str, Any]) -> dict[str, Any]:
  return {
    "direction": task.direction,
    "kernel": kernel,
    "autotune_mode": mode,
    "causal": task.causal,
    "dtype": _dtype_schema_name(task.dtype),
    "headdim": task.headdim,
    "seqlen_q": task.seqlen_q,
    "seqlen_k": task.seqlen_k,
    "seqlen_q_bucket": bucket_autotune_seqlen(task.seqlen_q, mode),
    "seqlen_k_bucket": bucket_autotune_seqlen(task.seqlen_k, mode),
    "config": config,
  }


def _record_entry(entries: dict[tuple[Any, ...], dict[str, Any]], entry: dict[str, Any]) -> None:
  key = (
    entry["direction"],
    entry["kernel"],
    entry["autotune_mode"],
    entry["causal"],
    entry["dtype"],
    entry["headdim"],
    entry["seqlen_q"],
    entry["seqlen_k"],
    entry.get("preprocess_d_chunk"),
    entry.get("bias_grad"),
    entry.get("grad_v_storage_dtype"),
    entry.get("use_gemv"),
    entry.get("has_dropout"),
  )
  entries[key] = entry


def _format_config(config: dict[str, Any]) -> str:
  """Return a compact JSON-style config string for progress logs.

  :param config: Persisted Triton launch config.
  :return: Human-readable config string.
  """
  return json.dumps(config, sort_keys=True, separators=(",", ":"))


def _format_entry(entry: dict[str, Any], choices_count: int, batch: int, heads: int) -> str:
  """Return one progress-log description for a tuned entry.

  :param entry: Persisted config entry.
  :param choices_count: Number of autotune candidates considered by the wrapper.
  :param batch: Batch size used for tuning.
  :param heads: Number of heads used for tuning.
  :return: Human-readable entry description.
  """
  direction = "FWD" if entry["direction"] == "forward" else "BWD"
  shape = (
    f"{direction}:{entry['kernel']}("
    f"{entry['dtype']},B{batch},H{heads},Q{entry['seqlen_q']},K{entry['seqlen_k']},"
    f"D{entry['headdim']},C{int(bool(entry['causal']))}"
  )
  return f"{shape}) best[{choices_count}]={_format_config(entry['config'])}"


def _tune_forward(
  task: TuneTask,
  batch: int,
  heads: int,
  mode: str,
  entries: dict[tuple[Any, ...], dict[str, Any]],
) -> list[tuple[dict[str, Any], int]]:
  q, k, v = _make_tensors(task, batch, heads)
  out = ffpa_attn_func(
    q,
    k,
    v,
    is_causal=task.causal,
    forward_backend="triton",
    triton_forward_autotune=True,
    triton_autotune_mode=mode,
  )
  del out
  num_splits = _get_decode_num_splits(task.seqlen_q, task.seqlen_k, task.headdim, batch, heads, q.device)
  if num_splits == 1:
    wrapper = _get_fwd_autotune(task.headdim, mode)
    kernel = "fwd_generic"
    entry = _entry_base(task, mode, kernel, config_from_triton_config(wrapper.best_config))
    choices_count = len(wrapper.configs)
  else:
    use_gemv = task.seqlen_q == 1
    wrapper = _get_decode_fwd_stage1_autotune(task.headdim, use_gemv, mode)
    kernel = "decode_fwd_stage1"
    entry = _entry_base(task, mode, kernel, config_from_triton_config(wrapper.best_config))
    entry["use_gemv"] = use_gemv
    choices_count = len(wrapper.configs)
  _record_entry(entries, entry)
  return [(entry, choices_count)]


def _tune_backward(
  task: TuneTask,
  batch: int,
  heads: int,
  mode: str,
  entries: dict[tuple[Any, ...], dict[str, Any]],
) -> list[tuple[dict[str, Any], int]]:
  q, k, v = _make_tensors(task, batch, heads)
  out = ffpa_attn_func(
    q,
    k,
    v,
    is_causal=task.causal,
    forward_backend="triton",
    backward_backend="triton",
    triton_forward_autotune=False,
    triton_backward_autotune=True,
    triton_autotune_mode=mode,
  )
  out.float().sum().backward()

  pre_wrapper = _get_pre_autotune(False, mode)
  pre_entry = _entry_base(task, mode, "bwd_preproc", config_from_triton_config(pre_wrapper.best_config))
  pre_entry["preprocess_d_chunk"] = False
  _record_entry(entries, pre_entry)
  pre_choices_count = len(pre_wrapper.configs)

  if task.seqlen_q < 8:
    use_gemv = task.seqlen_q == 1
    wrapper = _get_decode_bwd_stage1_autotune(task.headdim, use_gemv, mode, False)
    entry = _entry_base(task, mode, "decode_bwd_stage1", config_from_triton_config(wrapper.best_config))
    entry.update({
      "bias_grad": False,
      "grad_v_storage_dtype": None,
      "has_dropout": False,
      "use_gemv": use_gemv,
    })
    choices_count = len(wrapper.configs)
  else:
    wrapper = _get_bwd_autotune(task.headdim, mode, False)
    entry = _entry_base(task, mode, "bwd_generic", config_from_triton_config(wrapper.best_config))
    entry.update({
      "bias_grad": False,
      "grad_v_storage_dtype": None,
      "has_dropout": False,
    })
    choices_count = len(wrapper.configs)
  _record_entry(entries, entry)
  return [(pre_entry, pre_choices_count), (entry, choices_count)]


def _build_payload(entries: list[dict[str, Any]], mode: str, batch: int, heads: int,
                   seqlens: list[int]) -> dict[str, Any]:
  props = torch.cuda.get_device_properties(torch.cuda.current_device())
  return {
    "schema_version": SCHEMA_VERSION,
    "device_name": torch.cuda.get_device_name(torch.cuda.current_device()),
    "device_name_sanitized": sanitize_device_name(torch.cuda.get_device_name(torch.cuda.current_device())),
    "compute_capability": f"{props.major}.{props.minor}",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "ffpa_version": __version__,
    "torch_version": torch.__version__,
    "triton_version": triton.__version__,
    "autotune_mode": mode,
    "B": batch,
    "H": heads,
    "tune_grid": {
      "headdims": DEFAULT_HEADDIMS,
      "seqlens": seqlens,
    },
    "entries": entries,
  }


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Generate persistent FFPA Triton tuned configs.")
  parser.add_argument("--mode", choices=("fast", "max"), default="fast", help="Triton autotune search-space mode.")
  parser.add_argument("--directions", choices=("forward", "backward", "both"), default="both")
  parser.add_argument("--B", type=int, default=1, help="Batch size. First version supports B=1.")
  parser.add_argument("--H", type=int, default=32, help="Number of query/KV heads. First version supports H=32.")
  parser.add_argument(
    "--dtypes", type=_parse_dtypes, default=[torch.bfloat16], help="Comma-separated dtypes: bf16,fp16."
  )
  parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing device config JSON.")
  parser.add_argument("--output-dir", type=Path, default=None, help="Directory for generated device JSON.")
  return parser.parse_args()


def main() -> int:
  """Run the autotune config generation CLI.

  :return: Process exit code.
  """
  args = _parse_args()
  if args.B != 1 or args.H != 32:
    raise SystemExit("First version of ffpa_attn.autotune supports only --B 1 --H 32")
  if not torch.cuda.is_available():
    raise SystemExit("ffpa_attn.autotune requires a CUDA device")

  output_dir = args.output_dir or default_config_dir()
  output_path = device_config_path(output_dir)
  if output_path.exists() and not args.overwrite:
    print(f"Tuned config already exists, not overwriting: {output_path}")
    return 0

  seqlens = _available_seqlens()
  tasks: list[TuneTask] = []
  if args.directions in ("forward", "both"):
    tasks.extend(_iter_forward_tasks(args.dtypes, seqlens))
  if args.directions in ("backward", "both"):
    tasks.extend(_iter_backward_tasks(args.dtypes, seqlens))
  tasks = _limit_tasks(tasks)

  entries: dict[tuple[Any, ...], dict[str, Any]] = {}
  print(f"Generating {len(tasks)} FFPA Triton tuned config task(s) for {output_path}")
  for index, task in enumerate(tasks, start=1):
    try:
      start_time = time.perf_counter()
      if task.direction == "forward":
        tuned_entries = _tune_forward(task, args.B, args.H, args.mode, entries)
      else:
        tuned_entries = _tune_backward(task, args.B, args.H, args.mode, entries)
      torch.cuda.synchronize()
      elapsed = time.perf_counter() - start_time
      for tuned_entry, choices_count in tuned_entries:
        print(
          f"[AUTOTUNED][{index}/{len(tasks)}] {_format_entry(tuned_entry, choices_count, args.B, args.H)}, t={elapsed:.3f}s",
          flush=True,
        )
    except RuntimeError as exc:
      if "out of memory" not in str(exc).lower():
        raise
      elapsed = time.perf_counter() - start_time
      print(f"[AUTOTUNE-SKIPPED][{index}/{len(tasks)}] OOM after {elapsed:.3f}s: {exc}", flush=True)
      torch.cuda.empty_cache()

  ordered_entries = sorted(
    entries.values(),
    key=lambda item: (
      item["direction"],
      item["kernel"],
      item["dtype"],
      item["headdim"],
      item["seqlen_q"],
      item["seqlen_k"],
      item["causal"],
    )
  )
  payload = _build_payload(ordered_entries, args.mode, args.B, args.H, seqlens)
  write_config_file(payload, output_path, overwrite=args.overwrite)
  print(f"Wrote {len(ordered_entries)} tuned config entries to {output_path}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())

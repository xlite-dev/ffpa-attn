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
from ffpa_attn.triton._autotune_utils import (
  autotune_seqlen_key,
  exact_autotune_seqlen_keys,
)
from ffpa_attn.triton._ffpa_bwd import (
  _get_bwd_autotune,
  _get_decode_bwd_stage1_autotune,
  _get_pre_autotune,
)
from ffpa_attn.triton._ffpa_bwd_sm90 import (
  _get_bwd_sm90_autotune,
  is_sm90_tma_backward_supported,
)
from ffpa_attn.triton._ffpa_fwd import (
  _get_decode_fwd_stage1_autotune,
  _get_decode_num_splits,
  _get_fwd_autotune,
)
from ffpa_attn.triton._ffpa_fwd_sm90 import (
  _get_fwd_sm90_autotune,
  is_sm90_tma_forward_supported,
)
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
from .logger import init_logger

_FULL_TASK_DROPOUT_P = 0.1
_TUNE_SEED = 42

logger = init_logger(__name__)


@dataclass(frozen=True)
class TuneTask:
  """One runtime shape to autotune.

  :param direction: ``"forward"`` or ``"backward"``.
  :param dtype: Activation dtype.
  :param headdim: Head dimension.
  :param seqlen_q: Query sequence length.
  :param seqlen_k: KV sequence length.
  :param causal: Whether to apply causal masking.
  :param nheads_q: Query-head count.
  :param nheads_kv: Key/value-head count.
  :param has_attn_bias: Whether to tune with an additive attention bias.
  :param has_dropout: Whether to tune with dropout enabled.
  :param case_name: Human-readable case name used in logs and JSON metadata.
  """

  direction: str
  dtype: torch.dtype
  headdim: int
  seqlen_q: int
  seqlen_k: int
  causal: bool
  nheads_q: int = 32
  nheads_kv: int = 32
  has_attn_bias: bool = False
  has_dropout: bool = False
  case_name: str = "common"


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


def _resolve_directional_cli_flags(args: argparse.Namespace) -> argparse.Namespace:
  """Resolve legacy global TMA/WS CLI flags into directional flags."""
  if args.enable_tma:
    args.enable_fwd_tma = True
    args.enable_bwd_tma = True
  if args.enable_ws:
    args.enable_fwd_ws = True
    args.enable_bwd_ws = True
  return args


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


def _resolve_gqa_heads(num_heads: int) -> int:
  """Choose a KV-head count for the canonical GQA full-task case.

  :param num_heads: Query-head count.
  :return: KV-head count that divides ``num_heads``.
  """
  if num_heads <= 1:
    return 1
  candidate = max(1, num_heads // 4)
  while candidate > 1 and num_heads % candidate != 0:
    candidate -= 1
  return candidate


def _iter_full_variant_tasks(
  direction: str,
  dtype: torch.dtype,
  headdim: int,
  prefill_seqlens: list[int],
  heads: int,
) -> list[TuneTask]:
  """Return canonical full-coverage variants for one dtype/head-dim pair.

  :param direction: ``"forward"`` or ``"backward"``.
  :param dtype: Activation dtype.
  :param headdim: Head dimension.
  :param prefill_seqlens: Sequence lengths used for square prefill cases.
  :param heads: Query-head count.
  :return: Extra mask/dropout/GQA/MQA tuning tasks.
  """
  tasks: list[TuneTask] = []
  gqa_heads = _resolve_gqa_heads(heads)
  for seqlen in prefill_seqlens:
    tasks.extend([
      TuneTask(
        direction,
        dtype,
        headdim,
        seqlen,
        seqlen,
        False,
        heads,
        heads,
        has_attn_bias=True,
        case_name="attn-mask",
      ),
      TuneTask(
        direction,
        dtype,
        headdim,
        seqlen,
        seqlen,
        False,
        heads,
        heads,
        has_dropout=True,
        case_name="dropout",
      ),
    ])
    if 1 < gqa_heads < heads:
      tasks.append(TuneTask(
        direction,
        dtype,
        headdim,
        seqlen,
        seqlen,
        False,
        heads,
        gqa_heads,
        case_name="gqa",
      ))
    if heads > 1:
      tasks.append(TuneTask(
        direction,
        dtype,
        headdim,
        seqlen,
        seqlen,
        False,
        heads,
        1,
        case_name="mqa",
      ))
  return tasks


def _iter_forward_tasks(
  dtypes: list[torch.dtype],
  seqlens: list[int],
  heads: int = 32,
  full_tasks: bool = False,
) -> list[TuneTask]:
  tasks: list[TuneTask] = []
  prefill_seqlens = [value for value in seqlens if value >= 512]
  decode_kv_seqlens = [value for value in prefill_seqlens if value > 1]
  for dtype in dtypes:
    for headdim in DEFAULT_HEADDIMS:
      for causal in (False, True):
        for seqlen_q in prefill_seqlens:
          for seqlen_k in prefill_seqlens:
            if causal and seqlen_k < seqlen_q:
              continue
            tasks.append(TuneTask(
              "forward",
              dtype,
              headdim,
              seqlen_q,
              seqlen_k,
              causal,
              heads,
              heads,
            ))
        for seqlen_k in decode_kv_seqlens:
          tasks.append(
            TuneTask(
              "forward",
              dtype,
              headdim,
              1,
              seqlen_k,
              causal,
              heads,
              heads,
              case_name="decode-attn",
            )
          )
      if full_tasks:
        tasks.extend(_iter_full_variant_tasks(
          "forward",
          dtype,
          headdim,
          prefill_seqlens,
          heads,
        ))
  return tasks


def _iter_backward_tasks(
  dtypes: list[torch.dtype],
  seqlens: list[int],
  heads: int = 32,
  full_tasks: bool = False,
) -> list[TuneTask]:
  tasks: list[TuneTask] = []
  prefill_seqlens = [value for value in seqlens if value >= 512]
  decode_query_seqlens = [1]
  decode_kv_seqlens = [value for value in prefill_seqlens if value > 1]
  for dtype in dtypes:
    for headdim in DEFAULT_HEADDIMS:
      for causal in (False, True):
        for seqlen_q in prefill_seqlens:
          for seqlen_k in prefill_seqlens:
            if causal and seqlen_k < seqlen_q:
              continue
            tasks.append(TuneTask(
              "backward",
              dtype,
              headdim,
              seqlen_q,
              seqlen_k,
              causal,
              heads,
              heads,
            ))
        for seqlen_q in decode_query_seqlens:
          for seqlen_k in decode_kv_seqlens:
            tasks.append(
              TuneTask(
                "backward",
                dtype,
                headdim,
                seqlen_q,
                seqlen_k,
                causal,
                heads,
                heads,
                case_name="decode-attn",
              )
            )
      if full_tasks:
        tasks.extend(_iter_full_variant_tasks(
          "backward",
          dtype,
          headdim,
          prefill_seqlens,
          heads,
        ))
  return tasks


def _limit_tasks(tasks: list[TuneTask]) -> list[TuneTask]:
  limit = max_configs_from_env()
  if limit is None:
    return tasks
  return tasks[:limit]


def _make_tensors(
  task: TuneTask,
  batch: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  shape_q = (batch, task.nheads_q, task.seqlen_q, task.headdim)
  shape_kv = (batch, task.nheads_kv, task.seqlen_k, task.headdim)
  q = torch.randn(
    shape_q,
    device="cuda",
    dtype=task.dtype,
    requires_grad=task.direction == "backward",
  )
  k = torch.randn(
    shape_kv,
    device="cuda",
    dtype=task.dtype,
    requires_grad=task.direction == "backward",
  )
  v = torch.randn(
    shape_kv,
    device="cuda",
    dtype=task.dtype,
    requires_grad=task.direction == "backward",
  )
  return q, k, v


def _make_attn_bias(task: TuneTask) -> torch.Tensor | None:
  """Build the compact additive attention bias for a tuning task.

  :param task: Tuning task metadata.
  :return: Additive attention bias, or ``None`` for no-mask tasks.
  """
  if not task.has_attn_bias:
    return None
  torch.manual_seed(_TUNE_SEED + 1)
  if task.direction == "backward":
    return (torch.randn(
      1,
      1,
      1,
      task.seqlen_k,
      dtype=torch.float32,
      device="cuda",
    ) * 0.25).requires_grad_(True)
  return torch.randn(1, 1, 1, task.seqlen_k, dtype=task.dtype, device="cuda") * 0.25


def _entry_base(
  task: TuneTask,
  mode: str,
  kernel: str,
  config: dict[str, Any],
  enable_tma: bool | None = None,
  enable_ws: bool | None = None,
) -> dict[str, Any]:
  entry = {
    "direction": task.direction,
    "kernel": kernel,
    "causal": task.causal,
    "dtype": _dtype_schema_name(task.dtype),
    "headdim": task.headdim,
    "seqlen_q": task.seqlen_q,
    "seqlen_k": task.seqlen_k,
    "seqlen_q_bucket": autotune_seqlen_key(task.seqlen_q, mode),
    "seqlen_k_bucket": autotune_seqlen_key(task.seqlen_k, mode),
    "nheads_q": task.nheads_q,
    "nheads_kv": task.nheads_kv,
    "has_attn_bias": task.has_attn_bias,
    "has_dropout": task.has_dropout,
    "case_name": task.case_name,
    "config": config,
  }
  if enable_tma is not None:
    entry["enable_tma"] = enable_tma
  if enable_ws is not None:
    entry["enable_ws"] = enable_ws
  return entry


def _record_entry(
  entries: dict[tuple[Any, ...], dict[str, Any]],
  entry: dict[str, Any],
) -> None:
  key = (
    entry["direction"],
    entry["kernel"],
    entry["causal"],
    entry["dtype"],
    entry["headdim"],
    entry["seqlen_q"],
    entry["seqlen_k"],
    entry.get("preprocess_d_chunk"),
    entry.get("bias_grad"),
    entry.get("grad_v_storage_dtype"),
    entry.get("use_gemv"),
    entry.get("enable_tma"),
    entry.get("enable_ws"),
  )
  if entry["kernel"] != "bwd_preproc":
    key += (
      entry.get("nheads_q"),
      entry.get("nheads_kv"),
      entry.get("has_attn_bias"),
      entry.get("has_dropout"),
    )
  entries[key] = entry


def _format_config(config: dict[str, Any]) -> str:
  """Return a compact JSON-style config string for progress logs.

  :param config: Persisted Triton launch config.
  :return: Human-readable config string.
  """
  return json.dumps(config, sort_keys=True, separators=(",", ":"))


def _format_entry(entry: dict[str, Any], choices_count: int, batch: int) -> str:
  """Return one progress-log description for a tuned entry.

  :param entry: Persisted config entry.
  :param choices_count: Number of autotune candidates considered by the wrapper.
  :param batch: Batch size used for tuning.
  :return: Human-readable entry description.
  """
  direction = "FWD" if entry["direction"] == "forward" else "BWD"
  nheads_q = int(entry.get("nheads_q", 0))
  nheads_kv = int(entry.get("nheads_kv", nheads_q))
  shape = (
    f"{direction}:{entry['kernel']}("
    f"case={entry.get('case_name', 'common')},{entry['dtype']},"
    f"B{batch},Hq/Hkv={nheads_q}/{nheads_kv},"
    f"Q{entry['seqlen_q']},K{entry['seqlen_k']},D{entry['headdim']},"
    f"C{int(bool(entry['causal']))},mask={int(bool(entry.get('has_attn_bias', False)))},"
    f"drop={int(bool(entry.get('has_dropout', False)))},gqa={int(nheads_q != nheads_kv)}"
  )
  return f"{shape}) best[{choices_count}]={_format_config(entry['config'])}"


def _tune_forward(
  task: TuneTask,
  batch: int,
  mode: str,
  entries: dict[tuple[Any, ...], dict[str, Any]],
  enable_tma: bool = False,
  enable_ws: bool = False,
) -> list[tuple[dict[str, Any], int]]:
  q, k, v = _make_tensors(task, batch)
  attn_bias = _make_attn_bias(task)
  dtype = _dtype_schema_name(task.dtype)
  num_splits = _get_decode_num_splits(
    task.seqlen_q,
    task.seqlen_k,
    task.headdim,
    batch,
    task.nheads_q,
    q.device,
  )
  use_sm90_tma = enable_tma and is_sm90_tma_forward_supported(
    q,
    k,
    v,
    torch.empty_like(q),
    num_splits=num_splits,
  )
  use_sm90_ws = use_sm90_tma and enable_ws

  def run_forward_tune(run_enable_tma: bool, run_enable_ws: bool) -> None:
    if task.has_dropout:
      torch.manual_seed(_TUNE_SEED + 17)
    out = ffpa_attn_func(
      q,
      k,
      v,
      attn_mask=attn_bias,
      dropout_p=_FULL_TASK_DROPOUT_P if task.has_dropout else 0.0,
      is_causal=task.causal,
      enable_gqa=task.nheads_q != task.nheads_kv,
      forward_backend="triton",
      triton_autotune=True,
      triton_autotune_mode=mode,
      enable_tma=run_enable_tma,
      enable_ws=run_enable_ws,
    )
    del out

  tuned_entries: list[tuple[dict[str, Any], int]] = []

  run_forward_tune(False, False)
  if num_splits == 1:
    wrapper = _get_fwd_autotune(task.headdim, mode, dtype)
    entry = _entry_base(
      task,
      mode,
      "fwd_generic",
      config_from_triton_config(wrapper.best_config),
      enable_tma=False,
      enable_ws=False,
    )
    choices_count = len(wrapper.configs)
  else:
    use_gemv = task.seqlen_q == 1
    wrapper = _get_decode_fwd_stage1_autotune(
      task.headdim,
      use_gemv,
      mode,
      dtype,
    )
    entry = _entry_base(
      task,
      mode,
      "decode_fwd_stage1",
      config_from_triton_config(wrapper.best_config),
      enable_tma=False,
      enable_ws=False,
    )
    entry["use_gemv"] = use_gemv
    choices_count = len(wrapper.configs)
  _record_entry(entries, entry)
  tuned_entries.append((entry, choices_count))

  if use_sm90_tma:
    run_forward_tune(True, use_sm90_ws)
    wrapper = _get_fwd_sm90_autotune(
      task.headdim,
      mode,
      dtype,
      enable_ws=use_sm90_ws,
    )
    entry = _entry_base(
      task,
      mode,
      "fwd_sm90_generic",
      config_from_triton_config(wrapper.best_config),
      enable_tma=True,
      enable_ws=use_sm90_ws,
    )
    choices_count = len(wrapper.configs)
    _record_entry(entries, entry)
    tuned_entries.append((entry, choices_count))

  return tuned_entries


def _tune_backward(
  task: TuneTask,
  batch: int,
  mode: str,
  entries: dict[tuple[Any, ...], dict[str, Any]],
  enable_tma: bool = False,
  enable_ws: bool = False,
) -> list[tuple[dict[str, Any], int]]:
  q, k, v = _make_tensors(task, batch)
  attn_bias = _make_attn_bias(task)
  dtype = _dtype_schema_name(task.dtype)
  use_sm90_tma = enable_tma and is_sm90_tma_backward_supported(
    q,
    k,
    v,
    q,
    q,
    k,
    v,
    seqlen_q=task.seqlen_q,
  )
  use_sm90_ws = use_sm90_tma and enable_ws

  def run_backward_tune(run_enable_tma: bool, run_enable_ws: bool) -> None:
    if task.has_dropout:
      torch.manual_seed(_TUNE_SEED + 17)
    out = ffpa_attn_func(
      q,
      k,
      v,
      attn_mask=attn_bias,
      dropout_p=_FULL_TASK_DROPOUT_P if task.has_dropout else 0.0,
      is_causal=task.causal,
      enable_gqa=task.nheads_q != task.nheads_kv,
      forward_backend="triton",
      backward_backend="triton",
      triton_autotune=True,
      triton_autotune_mode=mode,
      enable_tma=run_enable_tma,
      enable_ws=run_enable_ws,
    )
    out.float().sum().backward()

  run_backward_tune(False, False)

  pre_wrapper = _get_pre_autotune(False, mode, dtype)
  pre_entry = _entry_base(
    task,
    mode,
    "bwd_preproc",
    config_from_triton_config(pre_wrapper.best_config),
  )
  pre_entry["config"].setdefault(
    "BLOCK_HEADDIM",
    max(64, triton.next_power_of_2(task.headdim)),
  )
  pre_entry.update({
    "preprocess_d_chunk": False,
    "has_attn_bias": False,
    "has_dropout": False,
    "nheads_kv": task.nheads_q,
    "case_name": "common",
  })
  _record_entry(entries, pre_entry)
  pre_choices_count = len(pre_wrapper.configs)
  tuned_entries = [(pre_entry, pre_choices_count)]

  if task.seqlen_q < 8:
    use_gemv = task.seqlen_q == 1
    wrapper = _get_decode_bwd_stage1_autotune(
      task.headdim,
      use_gemv,
      mode,
      task.has_attn_bias,
    )
    entry = _entry_base(
      task,
      mode,
      "decode_bwd_stage1",
      config_from_triton_config(wrapper.best_config),
    )
    entry.update({
      "bias_grad": task.has_attn_bias,
      "grad_v_storage_dtype": None,
      "use_gemv": use_gemv,
    })
    choices_count = len(wrapper.configs)
  else:
    wrapper = _get_bwd_autotune(task.headdim, mode, task.has_attn_bias)
    entry = _entry_base(
      task,
      mode,
      "bwd_generic",
      config_from_triton_config(wrapper.best_config),
      enable_tma=False,
      enable_ws=False,
    )
    entry.update({
      "bias_grad": task.has_attn_bias,
      "grad_v_storage_dtype": None,
    })
    choices_count = len(wrapper.configs)
  _record_entry(entries, entry)
  tuned_entries.append((entry, choices_count))

  if task.seqlen_q >= 8 and use_sm90_tma:
    run_backward_tune(True, use_sm90_ws)
    wrapper = _get_bwd_sm90_autotune(
      task.headdim,
      mode,
      dtype,
      task.has_attn_bias,
      enable_ws=use_sm90_ws,
    )
    entry = _entry_base(
      task,
      mode,
      "bwd_sm90_generic",
      config_from_triton_config(wrapper.best_config),
      enable_tma=True,
      enable_ws=use_sm90_ws,
    )
    entry.update({
      "bias_grad": task.has_attn_bias,
      "grad_v_storage_dtype": None,
    })
    choices_count = len(wrapper.configs)
    _record_entry(entries, entry)
    tuned_entries.append((entry, choices_count))

  return tuned_entries


def _build_payload(
  entries: list[dict[str, Any]],
  mode: str,
  batch: int,
  heads: int,
  full_tasks: bool,
  seqlens: list[int],
  enable_forward_tma: bool,
  enable_backward_tma: bool,
  enable_forward_ws: bool,
  enable_backward_ws: bool,
) -> dict[str, Any]:
  device_index = torch.cuda.current_device()
  device_name = torch.cuda.get_device_name(device_index)
  props = torch.cuda.get_device_properties(device_index)
  return {
    "schema_version": SCHEMA_VERSION,
    "device_name": device_name,
    "device_name_sanitized": sanitize_device_name(device_name),
    "compute_capability": f"{props.major}.{props.minor}",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "ffpa_version": __version__,
    "torch_version": torch.__version__,
    "triton_version": triton.__version__,
    "autotune_mode": mode,
    "B": batch,
    "H": heads,
    "full_tasks": full_tasks,
    "hardware_desc": {
      "enable_forward_tma": enable_forward_tma,
      "enable_backward_tma": enable_backward_tma,
      "enable_forward_ws": enable_forward_ws,
      "enable_backward_ws": enable_backward_ws,
    },
    "tune_grid": {
      "headdims": DEFAULT_HEADDIMS,
      "seqlens": seqlens,
    },
    "entries": entries,
  }


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Generate persistent FFPA Triton tuned configs.")
  parser.add_argument(
    "--mode",
    choices=("fast", "max"),
    default="fast",
    help="Triton autotune search-space mode.",
  )
  parser.add_argument(
    "--directions",
    choices=("forward", "backward", "both"),
    default="both",
  )
  parser.add_argument(
    "--B",
    type=int,
    default=1,
    help="Batch size used for tuning.",
  )
  parser.add_argument(
    "--H",
    type=int,
    default=32,
    help="Base query-head count used for tuning.",
  )
  parser.add_argument(
    "--full-tasks",
    action="store_true",
    help="Also tune canonical attn_mask, dropout, GQA, and MQA variants.",
  )
  parser.add_argument(
    "--dtypes",
    type=_parse_dtypes,
    default=[torch.bfloat16],
    help="Comma-separated dtypes: bf16,fp16.",
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
    help="Also generate persistent configs for the SM90+ TMA forward path when supported.",
  )
  parser.add_argument(
    "--enable-bwd-tma",
    action="store_true",
    help="Also generate persistent configs for the SM90+ TMA backward path when supported.",
  )
  parser.add_argument(
    "--enable-fwd-ws",
    action="store_true",
    help="Force warp-specialized SM90+ TMA forward configs when --enable-fwd-tma is set.",
  )
  parser.add_argument(
    "--enable-bwd-ws",
    action="store_true",
    help="Force warp-specialized SM90+ TMA backward configs when --enable-bwd-tma is set.",
  )
  parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Overwrite an existing device config JSON.",
  )
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=None,
    help="Directory for generated device JSON.",
  )
  return _resolve_directional_cli_flags(parser.parse_args())


def main() -> int:
  """Run the autotune config generation CLI.

  :return: Process exit code.
  """
  args = _parse_args()
  if args.B <= 0 or args.H <= 0:
    raise SystemExit("ffpa_attn.autotune requires positive --B and --H values")
  if not torch.cuda.is_available():
    raise SystemExit("ffpa_attn.autotune requires a CUDA device")

  output_dir = args.output_dir or default_config_dir()
  output_path = device_config_path(output_dir)
  if output_path.exists() and not args.overwrite:
    logger.info("Tuned config already exists, not overwriting: %s", output_path)
    return 0

  seqlens = _available_seqlens()
  tasks: list[TuneTask] = []
  if args.directions in ("forward", "both"):
    tasks.extend(_iter_forward_tasks(
      args.dtypes,
      seqlens,
      heads=args.H,
      full_tasks=args.full_tasks,
    ))
  if args.directions in ("backward", "both"):
    tasks.extend(_iter_backward_tasks(
      args.dtypes,
      seqlens,
      heads=args.H,
      full_tasks=args.full_tasks,
    ))
  tasks = _limit_tasks(tasks)

  entries: dict[tuple[Any, ...], dict[str, Any]] = {}
  full_variant_count = sum(task.case_name in {"attn-mask", "dropout", "gqa", "mqa"} for task in tasks)
  logger.info(
    "Generating %d FFPA Triton tuned config task(s) for %s "
    "mode=%s directions=%s B=%d H=%d full_tasks=%s full_variants=%d",
    len(tasks),
    output_path,
    args.mode,
    args.directions,
    args.B,
    args.H,
    args.full_tasks,
    full_variant_count,
  )
  tune_start_time = time.perf_counter()
  # Offline persistent tuning deliberately bypasses runtime seqlen bucketing so
  # target grid points such as 2048 are tuned on their exact shapes instead of
  # being intercepted by a coarser online autotune bucket.
  with exact_autotune_seqlen_keys():
    for index, task in enumerate(tasks, start=1):
      try:
        start_time = time.perf_counter()
        if task.direction == "forward":
          tuned_entries = _tune_forward(
            task,
            args.B,
            args.mode,
            entries,
            enable_tma=args.enable_fwd_tma,
            enable_ws=args.enable_fwd_ws,
          )
        else:
          tuned_entries = _tune_backward(
            task,
            args.B,
            args.mode,
            entries,
            enable_tma=args.enable_bwd_tma,
            enable_ws=args.enable_bwd_ws,
          )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        total_elapsed = time.perf_counter() - tune_start_time
        for tuned_entry, choices_count in tuned_entries:
          logger.info(
            "[AUTOTUNED][%d/%d] %s, t=%.3fs, T=%.3fs",
            index,
            len(tasks),
            _format_entry(tuned_entry, choices_count, args.B),
            elapsed,
            total_elapsed,
          )
      except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
          raise
        elapsed = time.perf_counter() - start_time
        total_elapsed = time.perf_counter() - tune_start_time
        logger.info(
          "[AUTOTUNE-SKIPPED][%d/%d] OOM after t=%.3fs, T=%.3fs: %s",
          index,
          len(tasks),
          elapsed,
          total_elapsed,
          exc,
        )
        torch.cuda.empty_cache()

  ordered_entries = sorted(
    entries.values(),
    key=lambda item: (
      item["direction"],
      item["kernel"],
      item["dtype"],
      item.get("case_name", "common"),
      item["headdim"],
      item["seqlen_q"],
      item["seqlen_k"],
      item["causal"],
      item.get("nheads_q", 0),
      item.get("nheads_kv", 0),
      item.get("has_attn_bias", False),
      item.get("has_dropout", False),
    )
  )
  payload = _build_payload(
    ordered_entries,
    args.mode,
    args.B,
    args.H,
    args.full_tasks,
    seqlens,
    args.enable_fwd_tma,
    args.enable_bwd_tma,
    args.enable_fwd_ws,
    args.enable_bwd_ws,
  )
  write_config_file(payload, output_path, overwrite=args.overwrite)
  logger.info("Wrote %d tuned config entries to %s", len(ordered_entries), output_path)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())

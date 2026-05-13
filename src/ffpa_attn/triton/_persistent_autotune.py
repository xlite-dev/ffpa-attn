"""Persistent Triton autotune config helpers.

This module owns the device-local JSON format used by the autotune CLI and by
the forward/backward Triton launchers when runtime autotune is disabled.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from ..logger import init_logger

SCHEMA_VERSION = 1
CONFIG_ENV_VAR = "FFPA_TUNED_CONFIG_DIR"
MAX_CONFIGS_ENV_VAR = "FFPA_AUTOTUNE_MAX_CONFIGS"

logger = init_logger(__name__)

DEFAULT_HEADDIMS = [320, 512, 640, 768, 1024]
DEFAULT_SEQLENS = [1, 512, 1024, 2048, 4096, 8192, 16384]

_CONFIG_CACHE: dict[tuple[str, str], list[dict[str, Any]]] = {}
_DEVICE_NAME_CACHE: dict[int, str] = {}

_KERNEL_CONFIG_KEYS = {
  "fwd_generic": {
    "BLOCK_M",
    "BLOCK_N",
    "BLOCK_HEADDIM_QK",
    "BLOCK_HEADDIM_V",
    "num_warps",
    "num_stages",
    "num_ctas",
    "maxnreg",
  },
  "decode_fwd_stage1": {
    "BLOCK_M",
    "BLOCK_N",
    "BLOCK_HEADDIM_QK",
    "BLOCK_HEADDIM_V",
    "num_warps",
    "num_stages",
    "num_ctas",
    "maxnreg",
  },
  "bwd_preproc": {
    "BLOCK_M",
    "BLOCK_HEADDIM",
    "D_CHUNK",
    "num_warps",
    "num_stages",
    "num_ctas",
    "maxnreg",
  },
  "bwd_generic": {
    "BLOCK_M",
    "BLOCK_N",
    "BLOCK_HEADDIM",
    "num_warps",
    "num_stages",
    "num_ctas",
    "maxnreg",
  },
  "decode_bwd_stage1": {
    "BLOCK_M",
    "BLOCK_N",
    "BLOCK_HEADDIM",
    "num_warps",
    "num_stages",
    "num_ctas",
    "maxnreg",
  },
}


@dataclass(frozen=True)
class PersistentConfigRequest:
  """Lookup request for a persisted Triton launch config.

  :param direction: Kernel direction, ``"forward"`` or ``"backward"``.
  :param kernel: Persisted kernel name.
  :param autotune_mode: Triton autotune search-space mode.
  :param dtype: Runtime activation dtype name, ``"fp16"`` or ``"bf16"``.
  :param headdim: Runtime head dimension.
  :param seqlen_q: Runtime query sequence length.
  :param seqlen_k: Runtime key/value sequence length.
  :param causal: Whether the attention is causal.
  :param preprocess_d_chunk: Backward preprocess D-chunk mode.
  :param bias_grad: Whether the current backward call writes attention-bias gradients.
  :param grad_v_storage_dtype: Optional Triton backward dV storage dtype name.
  :param use_gemv: Decode backward single-query specialization flag.
  :param has_attn_bias: Whether an additive attention bias is active.
  :param has_dropout: Whether dropout is active.
  :param nheads_q: Optional runtime query-head count, kept for callers that
      want to describe the request. Lookup does not require it to match.
  :param nheads_kv: Optional runtime key/value-head count before backward
      GQA/MQA expansion. Lookup does not require it to match.
  :param device_index: Optional CUDA device index for runtime callers. Passing
      this avoids a CUDA current-device query on repeated cache hits.
  """

  direction: str
  kernel: str
  autotune_mode: str
  dtype: str
  headdim: int
  seqlen_q: int
  seqlen_k: int | None = None
  causal: bool | None = None
  preprocess_d_chunk: bool | None = None
  bias_grad: bool | None = None
  grad_v_storage_dtype: str | None = None
  use_gemv: bool | None = None
  has_attn_bias: bool | None = None
  has_dropout: bool | None = None
  nheads_q: int | None = None
  nheads_kv: int | None = None
  device_index: int | None = None


def default_config_dir() -> Path:
  """Return the package-local tuned config directory.

  :return: Directory where bundled device JSON files live.
  """
  return Path(__file__).resolve().parent / "configs"


def runtime_config_dir() -> Path:
  """Return the runtime config directory, honoring ``FFPA_TUNED_CONFIG_DIR``.

  :return: Runtime directory searched for device JSON files.
  """
  override = os.environ.get(CONFIG_ENV_VAR)
  if override:
    return Path(override).expanduser()
  return default_config_dir()


def sanitize_device_name(device_name: str) -> str:
  """Convert a CUDA device name into a stable JSON file stem.

  :param device_name: Raw value returned by ``torch.cuda.get_device_name``.
  :return: File-name-safe device name stem.
  """
  stem = re.sub(r"[^0-9A-Za-z]+", "_", device_name.strip()).strip("_")
  return stem or "unknown_device"


def device_config_path(config_dir: Path | None = None, device_name: str | None = None) -> Path:
  """Return the tuned config path for a CUDA device.

  :param config_dir: Optional directory override.
  :param device_name: Optional raw CUDA device name.
  :return: Path named ``{sanitized_device_name}.json``.
  """
  root = config_dir or runtime_config_dir()
  name = device_name or _device_name(torch.cuda.current_device())
  return root / f"{sanitize_device_name(name)}.json"


def _device_name(device_index: int) -> str:
  """Return the cached CUDA device name for a device index.

  :param device_index: CUDA device index.
  :return: CUDA device name.
  """
  if device_index not in _DEVICE_NAME_CACHE:
    _DEVICE_NAME_CACHE[device_index] = torch.cuda.get_device_name(device_index)
  return _DEVICE_NAME_CACHE[device_index]


def dtype_name(dtype: torch.dtype) -> str:
  """Return the schema dtype name for a torch dtype.

  :param dtype: Runtime tensor dtype.
  :return: ``"fp16"`` or ``"bf16"``.
  :raises ValueError: If the dtype is unsupported by FFPA Triton kernels.
  """
  if dtype == torch.float16:
    return "fp16"
  if dtype == torch.bfloat16:
    return "bf16"
  raise ValueError(f"Unsupported FFPA Triton dtype {dtype!r}")


def grad_storage_dtype_name(dtype: torch.dtype | None) -> str | None:
  """Return the schema dtype name for optional gradient storage override.

  :param dtype: Optional internal gradient storage dtype.
  :return: ``"fp32"`` or ``None``.
  """
  if dtype is None:
    return None
  if dtype == torch.float32:
    return "fp32"
  raise ValueError(f"Unsupported Triton grad storage dtype {dtype!r}")


def nearest_value(values: list[int], target: int) -> int | None:
  """Return the nearest value to ``target`` from ``values``.

  Ties choose the larger value to match the high-side bucket preference used by
  the tuned config grid.

  :param values: Candidate integer values.
  :param target: Runtime value.
  :return: Nearest candidate, or ``None`` when ``values`` is empty.
  """
  if not values:
    return None
  return min(sorted(values), key=lambda value: (abs(value - target), -value))


def upper_or_max_value(values: list[int], target: int) -> int | None:
  """Return the smallest candidate >= ``target``, or the largest candidate.

  :param values: Candidate sequence-length values.
  :param target: Runtime sequence length.
  :return: Upper candidate, largest candidate, or ``None`` if empty.
  """
  if not values:
    return None
  ordered = sorted(values)
  for value in ordered:
    if value >= target:
      return value
  return ordered[-1]


def max_configs_from_env() -> int | None:
  """Return the optional autotune shape cap from ``FFPA_AUTOTUNE_MAX_CONFIGS``.

  :return: Positive cap value, or ``None`` when unset.
  :raises ValueError: If the environment value is not a positive integer.
  """
  value = os.environ.get(MAX_CONFIGS_ENV_VAR)
  if value in (None, ""):
    return None
  limit = int(value)
  if limit <= 0:
    raise ValueError(f"{MAX_CONFIGS_ENV_VAR} must be positive, got {value!r}")
  return limit


def config_from_triton_config(config: Any) -> dict[str, Any]:
  """Serialize a ``triton.Config``-like object to plain JSON data.

  :param config: Object exposing ``all_kwargs()``.
  :return: JSON-serializable config metadata.
  """
  raw = dict(config.all_kwargs())
  return {key: value for key, value in raw.items() if value is not None and key != "ir_override"}


def sanitize_kernel_config(kernel: str, config: dict[str, Any]) -> dict[str, Any] | None:
  """Validate and filter a persisted config for a kernel.

  :param kernel: Persisted kernel name.
  :param config: Raw config dictionary loaded from JSON.
  :return: Filtered config dict, or ``None`` when no valid meta remains.
  """
  allowed = _KERNEL_CONFIG_KEYS.get(kernel)
  if allowed is None:
    return None
  filtered = {key: value for key, value in config.items() if key in allowed and value is not None}
  return filtered or None


def write_config_file(payload: dict[str, Any], path: Path, overwrite: bool = False) -> None:
  """Write a tuned config JSON file.

  :param payload: JSON-serializable tuned config payload.
  :param path: Destination path.
  :param overwrite: Whether an existing file may be replaced.
  :raises FileExistsError: If ``path`` exists and ``overwrite`` is ``False``.
  """
  if path.exists() and not overwrite:
    raise FileExistsError(f"Tuned config already exists: {path}")
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_config_entries(config_dir: Path | None = None, device_name: str | None = None) -> list[dict[str, Any]]:
  """Load entries for the current device.

  Missing or malformed files return an empty list so runtime launchers can keep
  their built-in defaults.

  :param config_dir: Optional directory override.
  :param device_name: Optional raw CUDA device name.
  :return: Persisted config entries.
  """
  path = device_config_path(config_dir=config_dir, device_name=device_name)
  cache_key = (str(path.parent), path.stem)
  if cache_key in _CONFIG_CACHE:
    return _CONFIG_CACHE[cache_key]

  candidates = [path, path.with_name(f"{path.stem}.config.json")]
  for candidate in candidates:
    if not candidate.exists():
      continue
    try:
      payload = json.loads(candidate.read_text())
      if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
        entries: list[dict[str, Any]] = []
      else:
        loaded = payload.get("entries", [])
        entries = loaded if isinstance(loaded, list) else []
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
      entries = []
    _CONFIG_CACHE[cache_key] = entries
    return entries

  _CONFIG_CACHE[cache_key] = []
  return []


def clear_config_cache() -> None:
  """Clear the in-process JSON entry cache."""
  _CONFIG_CACHE.clear()
  _DEVICE_NAME_CACHE.clear()
  _lookup_persistent_config_cached.cache_clear()


def _head_layout_rank(entry: dict[str, Any], request: PersistentConfigRequest) -> int:
  """Rank head-layout metadata without making it a compatibility filter.

  :param entry: Persisted config entry.
  :param request: Runtime lookup request.
  :return: ``0`` for an exact recorded head-layout match, otherwise ``1``.
  """
  if request.nheads_q is None or request.nheads_kv is None:
    return 1
  try:
    if int(entry.get("nheads_q")) == request.nheads_q and int(entry.get("nheads_kv")) == request.nheads_kv:
      return 0
  except (TypeError, ValueError):
    pass
  return 1


def _lookup_cache_key(request: PersistentConfigRequest) -> tuple[str, int, PersistentConfigRequest]:
  """Return the cache key for one persistent lookup request."""
  device_index = request.device_index if request.device_index is not None else torch.cuda.current_device()
  return (os.environ.get(CONFIG_ENV_VAR, ""), device_index, request)


def _debug_lookup_message(prefix: str, request: PersistentConfigRequest, config: dict[str, Any] | None) -> None:
  """Log one DEBUG persistent lookup event.

  :param prefix: Event prefix describing lookup state.
  :param request: Runtime shape and variant filters.
  :param config: Selected kernel launch meta dict, or ``None`` on fallback.
  """
  logger.debug_once(
    "%s: direction=%s kernel=%s mode=%s dtype=%s D=%d Nq=%d Nkv=%s "
    "causal=%s use_gemv=%s bias_grad=%s has_attn_bias=%s has_dropout=%s "
    "Hq=%s Hkv=%s config=%s",
    prefix,
    request.direction,
    request.kernel,
    request.autotune_mode,
    request.dtype,
    request.headdim,
    request.seqlen_q,
    request.seqlen_k,
    request.causal,
    request.use_gemv,
    request.bias_grad,
    request.has_attn_bias,
    request.has_dropout,
    request.nheads_q,
    request.nheads_kv,
    config,
  )


@lru_cache(maxsize=None)
def _lookup_persistent_config_cached(
  config_dir_key: str,
  device_index: int,
  request: PersistentConfigRequest,
) -> dict[str, Any] | None:
  """Find and cache the best persisted launch config for one request."""
  del config_dir_key
  device_name = _device_name(device_index)

  candidates: list[dict[str, Any]] = []
  for entry in load_config_entries(device_name=device_name):
    if entry.get("direction") != request.direction:
      continue
    if entry.get("kernel") != request.kernel:
      continue
    if entry.get("autotune_mode") != request.autotune_mode:
      continue
    if entry.get("dtype") != request.dtype:
      continue
    if request.causal is not None and bool(entry.get("causal", False)) != request.causal:
      continue
    if request.preprocess_d_chunk is not None and bool(
      entry.get("preprocess_d_chunk", False)
    ) != request.preprocess_d_chunk:
      continue
    if request.bias_grad is not None and bool(entry.get("bias_grad", False)) != request.bias_grad:
      continue
    if request.grad_v_storage_dtype is not None and entry.get("grad_v_storage_dtype") != request.grad_v_storage_dtype:
      continue
    if request.grad_v_storage_dtype is None and entry.get("grad_v_storage_dtype") is not None:
      continue
    if request.use_gemv is not None and bool(entry.get("use_gemv", False)) != request.use_gemv:
      continue
    if request.has_attn_bias is not None and bool(entry.get("has_attn_bias", False)) != request.has_attn_bias:
      continue
    if request.has_dropout is not None and bool(entry.get("has_dropout", False)) != request.has_dropout:
      continue
    if not isinstance(entry.get("config"), dict):
      continue
    try:
      int(entry["headdim"])
      int(entry["seqlen_q"])
      if request.seqlen_k is not None:
        int(entry["seqlen_k"])
    except (KeyError, TypeError, ValueError):
      continue
    candidates.append(entry)

  if not candidates:
    return None

  headdim_target = nearest_value(sorted({int(entry["headdim"]) for entry in candidates}), request.headdim)
  if headdim_target is None:
    return None
  candidates = [entry for entry in candidates if int(entry.get("headdim", -1)) == headdim_target]

  seqlen_q_target = upper_or_max_value(sorted({int(entry["seqlen_q"]) for entry in candidates}), request.seqlen_q)
  if seqlen_q_target is None:
    return None
  candidates = [entry for entry in candidates if int(entry.get("seqlen_q", -1)) == seqlen_q_target]

  if request.seqlen_k is not None:
    seqlen_k_target = upper_or_max_value(sorted({int(entry["seqlen_k"]) for entry in candidates}), request.seqlen_k)
    if seqlen_k_target is None:
      return None
    candidates = [entry for entry in candidates if int(entry.get("seqlen_k", -1)) == seqlen_k_target]

  if not candidates:
    return None
  selected = min(candidates, key=lambda entry: _head_layout_rank(entry, request))
  return sanitize_kernel_config(request.kernel, selected["config"])


def lookup_persistent_config(request: PersistentConfigRequest) -> dict[str, Any] | None:
  """Find the best persisted launch config for a runtime request.

  :param request: Runtime shape and variant filters.
  :return: Kernel launch meta dict, or ``None`` if no compatible entry exists.
  """
  cache_key = _lookup_cache_key(request)
  if not logger.isEnabledFor(logging.DEBUG):
    return _lookup_persistent_config_cached(*cache_key)

  hits_before = _lookup_persistent_config_cached.cache_info().hits
  config = _lookup_persistent_config_cached(*cache_key)
  hits_after = _lookup_persistent_config_cached.cache_info().hits
  if hits_after > hits_before and config is not None:
    _debug_lookup_message("Persistent autotune cache hit", request, config)
  elif config is not None:
    _debug_lookup_message("Persistent autotune selected config", request, config)
  else:
    _debug_lookup_message("Persistent autotune lookup miss", request, None)
  return config

"""Unit tests for persistent FFPA Triton autotune configs."""

import logging

import pytest

from ffpa_attn.triton import _persistent_autotune as persistent
from ffpa_attn.triton._ffpa_bwd import _normalize_bwd_pre_config


def _payload(entries):
  return {
    "schema_version": persistent.SCHEMA_VERSION,
    "device_name": "NVIDIA L20",
    "entries": entries,
  }


def _patch_cuda_device(monkeypatch):
  monkeypatch.setattr(persistent.torch.cuda, "current_device", lambda: 0)
  monkeypatch.setattr(persistent.torch.cuda, "get_device_name", lambda device=0: "NVIDIA L20")


def test_sanitize_device_name_and_nearest_values():
  assert persistent.sanitize_device_name("NVIDIA L20") == "NVIDIA_L20"
  assert persistent.sanitize_device_name("  NVIDIA GeForce RTX 5090  ") == "NVIDIA_GeForce_RTX_5090"
  assert persistent.nearest_value([320, 512, 640], 384) == 320
  assert persistent.nearest_value([320, 512, 640], 448) == 512
  assert persistent.nearest_value([320, 512, 640, 768, 1024], 900) == 1024
  assert persistent.upper_or_max_value([1, 512, 1024, 2048, 4096, 8192], 3000) == 4096
  assert persistent.upper_or_max_value([1, 512, 1024, 2048, 4096, 8192], 32768) == 8192


def test_lookup_forward_uses_shape_grid_nearest(tmp_path, monkeypatch):
  _patch_cuda_device(monkeypatch)
  monkeypatch.setenv(persistent.CONFIG_ENV_VAR, str(tmp_path))
  persistent.clear_config_cache()
  path = persistent.device_config_path(tmp_path, "NVIDIA L20")
  persistent.write_config_file(
    _payload([
      {
        "direction": "forward",
        "kernel": "fwd_generic",
        "autotune_mode": "fast",
        "causal": False,
        "dtype": "bf16",
        "headdim": 320,
        "seqlen_q": 4096,
        "seqlen_k": 8192,
        "config": {
          "BLOCK_M": 64,
          "BLOCK_N": 64,
          "BLOCK_HEADDIM_QK": 128,
          "BLOCK_HEADDIM_V": 128,
          "num_warps": 8
        },
      },
      {
        "direction": "forward",
        "kernel": "fwd_generic",
        "autotune_mode": "fast",
        "causal": False,
        "dtype": "bf16",
        "headdim": 512,
        "seqlen_q": 2048,
        "seqlen_k": 8192,
        "config": {
          "BLOCK_M": 128,
          "BLOCK_N": 64,
          "BLOCK_HEADDIM_QK": 64,
          "BLOCK_HEADDIM_V": 64,
          "num_warps": 8
        },
      },
    ]),
    path,
  )

  config = persistent.lookup_persistent_config(
    persistent.PersistentConfigRequest(
      direction="forward",
      kernel="fwd_generic",
      autotune_mode="fast",
      dtype="bf16",
      headdim=384,
      seqlen_q=3000,
      seqlen_k=32768,
      causal=False,
    )
  )
  assert config == {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_HEADDIM_QK": 128, "BLOCK_HEADDIM_V": 128, "num_warps": 8}


def test_lookup_reuses_cached_request_result(tmp_path, monkeypatch):
  _patch_cuda_device(monkeypatch)
  monkeypatch.setenv(persistent.CONFIG_ENV_VAR, str(tmp_path))
  persistent.clear_config_cache()
  path = persistent.device_config_path(tmp_path, "NVIDIA L20")
  persistent.write_config_file(
    _payload([
      {
        "direction": "forward",
        "kernel": "fwd_generic",
        "autotune_mode": "fast",
        "causal": False,
        "dtype": "bf16",
        "headdim": 512,
        "seqlen_q": 1024,
        "seqlen_k": 8192,
        "config": {
          "BLOCK_M": 128,
          "BLOCK_N": 64,
          "BLOCK_HEADDIM_QK": 64,
          "BLOCK_HEADDIM_V": 64,
          "num_warps": 8
        },
      },
    ]),
    path,
  )
  load_calls = 0
  original_load_config_entries = persistent.load_config_entries

  def counted_load_config_entries(*args, **kwargs):
    nonlocal load_calls
    load_calls += 1
    return original_load_config_entries(*args, **kwargs)

  monkeypatch.setattr(persistent, "load_config_entries", counted_load_config_entries)
  monkeypatch.setattr(
    persistent.torch.cuda, "current_device", lambda: (_ for _ in ()).throw(RuntimeError("unexpected"))
  )
  request = persistent.PersistentConfigRequest(
    direction="forward",
    kernel="fwd_generic",
    autotune_mode="fast",
    dtype="bf16",
    headdim=512,
    seqlen_q=1024,
    seqlen_k=8192,
    causal=False,
    device_index=0,
  )

  assert persistent.lookup_persistent_config(request)["BLOCK_M"] == 128
  assert persistent.lookup_persistent_config(request)["BLOCK_M"] == 128
  assert load_calls == 1


def test_lookup_skip_persistent_tuned_config_env_returns_none(tmp_path, monkeypatch):
  _patch_cuda_device(monkeypatch)
  monkeypatch.setenv(persistent.CONFIG_ENV_VAR, str(tmp_path))
  persistent.clear_config_cache()
  path = persistent.device_config_path(tmp_path, "NVIDIA L20")
  persistent.write_config_file(
    _payload([
      {
        "direction": "forward",
        "kernel": "fwd_generic",
        "autotune_mode": "fast",
        "causal": False,
        "dtype": "bf16",
        "headdim": 512,
        "seqlen_q": 1024,
        "seqlen_k": 8192,
        "config": {
          "BLOCK_M": 128,
          "BLOCK_N": 64,
          "BLOCK_HEADDIM_QK": 64,
          "BLOCK_HEADDIM_V": 64,
          "num_warps": 8
        },
      },
    ]),
    path,
  )
  request = persistent.PersistentConfigRequest(
    direction="forward",
    kernel="fwd_generic",
    autotune_mode="fast",
    dtype="bf16",
    headdim=512,
    seqlen_q=1024,
    seqlen_k=8192,
    causal=False,
    device_index=0,
  )

  assert persistent.lookup_persistent_config(request)["BLOCK_M"] == 128
  monkeypatch.setenv(persistent.SKIP_PERSISTENT_TUNED_CONFIG_ENV_VAR, "1")
  assert persistent.lookup_persistent_config(request) is None


def test_lookup_caches_device_name_by_device_index(tmp_path, monkeypatch):
  monkeypatch.setattr(persistent.torch.cuda, "current_device", lambda: 0)
  monkeypatch.setenv(persistent.CONFIG_ENV_VAR, str(tmp_path))
  persistent.clear_config_cache()
  path = persistent.device_config_path(tmp_path, "NVIDIA L20")
  persistent.write_config_file(
    _payload([
      {
        "direction": "forward",
        "kernel": "fwd_generic",
        "autotune_mode": "fast",
        "causal": False,
        "dtype": "bf16",
        "headdim": 512,
        "seqlen_q": 1024,
        "seqlen_k": 8192,
        "config": {
          "BLOCK_M": 128,
          "BLOCK_N": 64,
          "BLOCK_HEADDIM_QK": 64,
          "BLOCK_HEADDIM_V": 64,
          "num_warps": 8
        },
      },
    ]),
    path,
  )
  device_name_calls = 0

  def counted_get_device_name(device=0):
    nonlocal device_name_calls
    device_name_calls += 1
    assert device == 0
    return "NVIDIA L20"

  monkeypatch.setattr(persistent.torch.cuda, "get_device_name", counted_get_device_name)
  request = persistent.PersistentConfigRequest(
    direction="forward",
    kernel="fwd_generic",
    autotune_mode="fast",
    dtype="bf16",
    headdim=512,
    seqlen_q=1024,
    seqlen_k=8192,
    causal=False,
    device_index=0,
  )

  assert persistent.lookup_persistent_config(request)["BLOCK_M"] == 128
  assert persistent.lookup_persistent_config(request.__class__(**{
    **request.__dict__, "headdim": 384
  }))["BLOCK_M"] == 128
  assert device_name_calls == 1


def test_lookup_debug_logs_selected_config_cached_hit_and_miss(tmp_path, monkeypatch):
  _patch_cuda_device(monkeypatch)
  monkeypatch.setenv(persistent.CONFIG_ENV_VAR, str(tmp_path))
  persistent.clear_config_cache()
  path = persistent.device_config_path(tmp_path, "NVIDIA L20")
  persistent.write_config_file(
    _payload([
      {
        "direction": "forward",
        "kernel": "fwd_generic",
        "autotune_mode": "fast",
        "causal": False,
        "dtype": "bf16",
        "headdim": 512,
        "seqlen_q": 1024,
        "seqlen_k": 8192,
        "config": {
          "BLOCK_M": 128,
          "BLOCK_N": 64,
          "BLOCK_HEADDIM_QK": 64,
          "BLOCK_HEADDIM_V": 64,
          "num_warps": 8
        },
      },
    ]),
    path,
  )
  debug_messages = []

  def debug_once(message, *args, **kwargs):
    del kwargs
    debug_messages.append(message % args)

  monkeypatch.setattr(persistent.logger, "isEnabledFor", lambda level: level == logging.DEBUG)
  monkeypatch.setattr(persistent.logger, "debug_once", debug_once)
  request = persistent.PersistentConfigRequest(
    direction="forward",
    kernel="fwd_generic",
    autotune_mode="fast",
    dtype="bf16",
    headdim=512,
    seqlen_q=1024,
    seqlen_k=8192,
    causal=False,
    device_index=0,
  )

  assert persistent.lookup_persistent_config(request)["BLOCK_M"] == 128
  assert len(debug_messages) == 1
  assert debug_messages[0].startswith("Persistent autotune selected config")
  assert "kernel=fwd_generic" in debug_messages[0]
  assert "config={" in debug_messages[0]
  assert "'BLOCK_M': 128" in debug_messages[0]
  assert persistent.lookup_persistent_config(request)["BLOCK_M"] == 128

  assert len(debug_messages) == 2
  assert debug_messages[1].startswith("Persistent autotune cache hit")
  assert "kernel=fwd_generic" in debug_messages[1]
  assert "config={" in debug_messages[1]
  assert "'BLOCK_M': 128" in debug_messages[1]

  missing_request = request.__class__(**{**request.__dict__, "dtype": "fp16"})
  assert persistent.lookup_persistent_config(missing_request) is None
  assert len(debug_messages) == 3
  assert debug_messages[2].startswith("Persistent autotune lookup miss")
  assert "kernel=fwd_generic" in debug_messages[2]
  assert "config=None" in debug_messages[2]


def test_lookup_backward_filters_variants(tmp_path, monkeypatch):
  _patch_cuda_device(monkeypatch)
  monkeypatch.setenv(persistent.CONFIG_ENV_VAR, str(tmp_path))
  persistent.clear_config_cache()
  path = persistent.device_config_path(tmp_path, "NVIDIA L20")
  persistent.write_config_file(
    _payload([
      {
        "direction": "backward",
        "kernel": "bwd_generic",
        "autotune_mode": "fast",
        "causal": True,
        "dtype": "fp16",
        "headdim": 512,
        "seqlen_q": 1024,
        "seqlen_k": 1024,
        "bias_grad": False,
        "grad_v_storage_dtype": None,
        "has_dropout": False,
        "config": {
          "BLOCK_M": 64,
          "BLOCK_N": 64,
          "BLOCK_HEADDIM": 128,
          "num_warps": 4,
          "num_stages": 2
        },
      },
      {
        "direction": "backward",
        "kernel": "bwd_generic",
        "autotune_mode": "fast",
        "causal": True,
        "dtype": "fp16",
        "headdim": 512,
        "seqlen_q": 1024,
        "seqlen_k": 1024,
        "bias_grad": True,
        "grad_v_storage_dtype": None,
        "has_dropout": False,
        "config": {
          "BLOCK_M": 128,
          "BLOCK_N": 64,
          "BLOCK_HEADDIM": 64,
          "num_warps": 8,
          "num_stages": 2
        },
      },
    ]),
    path,
  )

  request = persistent.PersistentConfigRequest(
    direction="backward",
    kernel="bwd_generic",
    autotune_mode="fast",
    dtype="fp16",
    headdim=512,
    seqlen_q=1024,
    seqlen_k=1024,
    causal=True,
    bias_grad=False,
    grad_v_storage_dtype=None,
    has_dropout=False,
  )
  assert persistent.lookup_persistent_config(request)["BLOCK_HEADDIM"] == 128
  assert persistent.lookup_persistent_config(request.__class__(**{**request.__dict__, "has_dropout": True})) is None


def test_backward_preprocess_config_backfills_block_headdim():
  config = _normalize_bwd_pre_config(
    {
      "BLOCK_M": 128,
      "D_CHUNK": False,
      "num_warps": 4
    },
    preprocess_d_chunk=False,
    block_headdim_delta=512,
  )
  assert config == {"BLOCK_M": 128, "BLOCK_HEADDIM": 512, "D_CHUNK": False, "num_warps": 4}

  d_chunk_config = _normalize_bwd_pre_config(
    {
      "BLOCK_M": 64,
      "BLOCK_HEADDIM": 128,
      "D_CHUNK": True,
      "num_warps": 8
    },
    preprocess_d_chunk=True,
    block_headdim_delta=64,
  )
  assert d_chunk_config["BLOCK_HEADDIM"] == 128


def test_lookup_filters_mask_dropout_but_not_head_layout(tmp_path, monkeypatch):
  _patch_cuda_device(monkeypatch)
  monkeypatch.setenv(persistent.CONFIG_ENV_VAR, str(tmp_path))
  persistent.clear_config_cache()
  path = persistent.device_config_path(tmp_path, "NVIDIA L20")
  persistent.write_config_file(
    _payload([
      {
        "direction": "forward",
        "kernel": "fwd_generic",
        "autotune_mode": "fast",
        "causal": False,
        "dtype": "bf16",
        "headdim": 512,
        "seqlen_q": 1024,
        "seqlen_k": 1024,
        "config": {
          "BLOCK_M": 64,
          "BLOCK_N": 64,
          "BLOCK_HEADDIM_QK": 64,
          "BLOCK_HEADDIM_V": 64,
          "num_warps": 8
        },
      },
      {
        "direction": "forward",
        "kernel": "fwd_generic",
        "autotune_mode": "fast",
        "causal": False,
        "dtype": "bf16",
        "headdim": 512,
        "seqlen_q": 1024,
        "seqlen_k": 1024,
        "nheads_q": 8,
        "nheads_kv": 8,
        "has_attn_bias": True,
        "has_dropout": False,
        "config": {
          "BLOCK_M": 128,
          "BLOCK_N": 64,
          "BLOCK_HEADDIM_QK": 64,
          "BLOCK_HEADDIM_V": 64,
          "num_warps": 8
        },
      },
      {
        "direction": "forward",
        "kernel": "fwd_generic",
        "autotune_mode": "fast",
        "causal": False,
        "dtype": "bf16",
        "headdim": 512,
        "seqlen_q": 1024,
        "seqlen_k": 1024,
        "nheads_q": 8,
        "nheads_kv": 8,
        "has_attn_bias": False,
        "has_dropout": True,
        "config": {
          "BLOCK_M": 64,
          "BLOCK_N": 64,
          "BLOCK_HEADDIM_QK": 128,
          "BLOCK_HEADDIM_V": 128,
          "num_warps": 8
        },
      },
      {
        "direction": "forward",
        "kernel": "fwd_generic",
        "autotune_mode": "fast",
        "causal": False,
        "dtype": "bf16",
        "headdim": 512,
        "seqlen_q": 1024,
        "seqlen_k": 1024,
        "nheads_q": 8,
        "nheads_kv": 2,
        "has_attn_bias": False,
        "has_dropout": False,
        "config": {
          "BLOCK_M": 128,
          "BLOCK_N": 64,
          "BLOCK_HEADDIM_QK": 128,
          "BLOCK_HEADDIM_V": 128,
          "num_warps": 4
        },
      },
      {
        "direction": "forward",
        "kernel": "fwd_generic",
        "autotune_mode": "fast",
        "causal": False,
        "dtype": "bf16",
        "headdim": 512,
        "seqlen_q": 1024,
        "seqlen_k": 1024,
        "nheads_q": 8,
        "nheads_kv": 1,
        "has_attn_bias": False,
        "has_dropout": False,
        "config": {
          "BLOCK_M": 64,
          "BLOCK_N": 64,
          "BLOCK_HEADDIM_QK": 128,
          "BLOCK_HEADDIM_V": 128,
          "num_warps": 4
        },
      },
    ]),
    path,
  )

  request = persistent.PersistentConfigRequest(
    direction="forward",
    kernel="fwd_generic",
    autotune_mode="fast",
    dtype="bf16",
    headdim=512,
    seqlen_q=1024,
    seqlen_k=1024,
    causal=False,
    has_attn_bias=False,
    has_dropout=False,
    nheads_q=8,
    nheads_kv=8,
  )
  assert persistent.lookup_persistent_config(request)["BLOCK_M"] == 64
  assert persistent.lookup_persistent_config(request.__class__(**{
    **request.__dict__, "has_attn_bias": True
  }))["BLOCK_M"] == 128
  assert persistent.lookup_persistent_config(request.__class__(**{
    **request.__dict__, "has_dropout": True
  }))["BLOCK_HEADDIM_QK"] == 128
  assert persistent.lookup_persistent_config(request.__class__(**{
    **request.__dict__, "nheads_kv": 1
  }))["BLOCK_HEADDIM_QK"] == 128
  assert persistent.lookup_persistent_config(request.__class__(**{
    **request.__dict__, "nheads_q": 16,
    "nheads_kv": 4
  }))["BLOCK_M"] == 64


def test_missing_or_wrong_schema_config_falls_back(tmp_path, monkeypatch):
  _patch_cuda_device(monkeypatch)
  monkeypatch.setenv(persistent.CONFIG_ENV_VAR, str(tmp_path))
  persistent.clear_config_cache()
  path = persistent.device_config_path(tmp_path, "NVIDIA L20")
  persistent.write_config_file({"schema_version": -1, "entries": []}, path)
  config = persistent.lookup_persistent_config(
    persistent.PersistentConfigRequest(
      direction="forward",
      kernel="fwd_generic",
      autotune_mode="fast",
      dtype="bf16",
      headdim=320,
      seqlen_q=1024,
      seqlen_k=1024,
      causal=False,
    )
  )
  assert config is None


def test_max_configs_from_env(monkeypatch):
  monkeypatch.setenv(persistent.MAX_CONFIGS_ENV_VAR, "3")
  assert persistent.max_configs_from_env() == 3
  monkeypatch.setenv(persistent.MAX_CONFIGS_ENV_VAR, "0")
  with pytest.raises(ValueError):
    persistent.max_configs_from_env()

"""Unit tests for persistent FFPA Triton autotune configs."""

import pytest

from ffpa_attn.triton import _persistent_autotune as persistent


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

"""Unit tests for Triton FFPA autotune mode plumbing and search-space pruning."""

from types import SimpleNamespace

import pytest
import torch
import triton

import ffpa_attn.autotune as autotune_module
from ffpa_attn.autotune import TuneTask, _build_payload, _iter_backward_tasks, _iter_forward_tasks, _tune_forward
from ffpa_attn.functional import FFPAAttnMeta
from ffpa_attn.triton._autotune_utils import autotune_seqlen_key, bucket_autotune_seqlen, exact_autotune_seqlen_keys
from ffpa_attn.triton._ffpa_bwd import (
  _gen_bwd_autotune_configs,
  _gen_decode_bwd_stage1_autotune_configs,
  _gen_pre_autotune_configs,
  _get_pre_autotune,
)
from ffpa_attn.triton._persistent_autotune import config_from_triton_config
from ffpa_attn.triton._ffpa_fwd import (
  _gen_decode_fwd_stage1_autotune_configs,
  _gen_fwd_autotune_configs,
  _get_decode_fwd_stage1_autotune,
  _get_fwd_autotune,
)
from ffpa_attn.triton._ffpa_fwd_sm90 import (
  _gen_fwd_sm90_autotune_configs,
  _get_fwd_sm90_autotune,
)


def test_triton_autotune_mode_defaults_to_fast():
  meta = FFPAAttnMeta.from_kwargs()
  assert meta.triton_autotune_mode == "fast"


def test_triton_autotune_defaults_to_false():
  meta = FFPAAttnMeta.from_kwargs()
  assert meta.triton_autotune is False


def test_triton_autotune_accepts_true():
  meta = FFPAAttnMeta.from_kwargs(triton_autotune=True)
  assert meta.triton_autotune is True


@pytest.mark.parametrize("old_key", ["triton_forward_autotune", "triton_backward_autotune"])
def test_old_directional_triton_autotune_keys_are_rejected(old_key):
  with pytest.raises(TypeError, match=old_key):
    FFPAAttnMeta.from_kwargs(**{old_key: True})


@pytest.mark.parametrize("mode", ["fast", "max"])
def test_triton_autotune_mode_accepts_valid_values(mode):
  meta = FFPAAttnMeta.from_kwargs(triton_autotune_mode=mode)
  assert meta.triton_autotune_mode == mode


def test_triton_autotune_mode_rejects_invalid_value():
  with pytest.raises(AssertionError, match="triton_autotune_mode"):
    FFPAAttnMeta.from_kwargs(triton_autotune_mode="bad")


@pytest.mark.parametrize(
  ("seqlen", "expected_bucket"),
  [
    (1, 1024),
    (1024, 1024),
    (1025, 2048),
    (8191, 8192),
    (8192, 8192),
    (8193, 8192),
    (16384, 8192),
  ],
)
def test_autotune_seqlen_bucket(seqlen, expected_bucket):
  assert bucket_autotune_seqlen(seqlen) == expected_bucket


def test_autotune_seqlen_key_uses_exact_context():
  assert autotune_seqlen_key(513, "fast") == 1024
  with exact_autotune_seqlen_keys():
    assert autotune_seqlen_key(513, "fast") == 513
    assert autotune_seqlen_key(8193, "fast") == 8193
  assert autotune_seqlen_key(513, "fast") == 1024


@pytest.mark.parametrize("headdim", [320, 512])
def test_fwd_fast_mode_prunes_generic_configs(headdim):
  fast = _gen_fwd_autotune_configs(headdim, autotune_mode="fast")
  max_configs = _gen_fwd_autotune_configs(headdim, autotune_mode="max")
  assert len(fast) < len(max_configs)
  assert all(config.num_warps == 8 for config in fast)


def test_fwd_fast_mode_prunes_decode_configs():
  fast = _gen_decode_fwd_stage1_autotune_configs(320, use_gemv=False, autotune_mode="fast")
  max_configs = _gen_decode_fwd_stage1_autotune_configs(320, use_gemv=False, autotune_mode="max")
  assert len(fast) < len(max_configs)


def test_sm90_fwd_configs_include_warp_specialize():
  fast = _gen_fwd_sm90_autotune_configs(512, autotune_mode="fast", enable_ws=True)
  max_configs = _gen_fwd_sm90_autotune_configs(512, autotune_mode="max", enable_ws=True)
  serialized = [config_from_triton_config(config) for config in fast]
  max_serialized = [config_from_triton_config(config) for config in max_configs]
  ws_configs = [config for config in serialized if config["warp_specialize"]]
  max_ws_configs = [config for config in max_serialized if config["warp_specialize"]]
  assert len(fast) < len(max_configs)
  assert {config["warp_specialize"] for config in serialized} == {False, True}
  assert all(config["BLOCK_HEADDIM_QK"] == config["BLOCK_HEADDIM_V"] for config in serialized)
  expected_ws_configs = {
    (64, 64, 64),
    (64, 64, 128),
    (64, 128, 64),
    (128, 64, 64),
  }
  assert {(config["BLOCK_M"], config["BLOCK_N"], config["BLOCK_HEADDIM_QK"])
          for config in ws_configs} == expected_ws_configs
  assert all(config["BLOCK_HEADDIM_QK"] == config["BLOCK_HEADDIM_V"] for config in ws_configs)
  assert all(config["num_warps"] == 4 for config in ws_configs)
  assert {config["num_stages"] for config in ws_configs} == {2, 3}
  assert {(config["BLOCK_M"], config["BLOCK_N"], config["BLOCK_HEADDIM_QK"])
          for config in max_ws_configs} == expected_ws_configs
  assert {config["num_warps"] for config in max_ws_configs} == {4, 8}
  assert {config["num_stages"] for config in max_ws_configs} == {2, 3}


def test_sm90_fwd_configs_can_disable_warp_specialize():
  configs = _gen_fwd_sm90_autotune_configs(512, autotune_mode="max", enable_ws=False)
  serialized = [config_from_triton_config(config) for config in configs]
  assert {config["warp_specialize"] for config in serialized} == {False}


def test_persistent_payload_records_hardware_desc(monkeypatch):
  monkeypatch.setattr(autotune_module.torch.cuda, "current_device", lambda: 0)
  monkeypatch.setattr(autotune_module.torch.cuda, "get_device_name", lambda device=0: "NVIDIA L20")

  class Props:
    major = 8
    minor = 9

  monkeypatch.setattr(autotune_module.torch.cuda, "get_device_properties", lambda device=0: Props())
  payload = _build_payload([], "fast", 1, 32, False, [512], enable_tma=True, enable_ws=True)

  assert payload["hardware_desc"] == {"enable_tma": True, "enable_ws": True}
  assert "enable_tma" not in payload
  assert "enable_ws" not in payload
  assert "generation_options" not in payload


def test_persistent_tune_forward_records_sm90_tma_config(monkeypatch):
  task = TuneTask("forward", torch.float16, 320, 512, 512, False, 8, 8)
  q = torch.empty(1, 8, 512, 320)
  k = torch.empty_like(q)
  v = torch.empty_like(q)
  ws_config = triton.Config(
    {
      "BLOCK_M": 64,
      "BLOCK_N": 128,
      "BLOCK_HEADDIM_QK": 64,
      "BLOCK_HEADDIM_V": 64,
      "warp_specialize": True,
    },
    num_warps=4,
    num_stages=2,
  )
  wrapper = SimpleNamespace(best_config=ws_config, configs=[ws_config])
  seen_kwargs = {}

  def fake_ffpa_attn_func(*args, **kwargs):
    del args
    seen_kwargs.update(kwargs)
    return torch.empty_like(q)

  monkeypatch.setattr(autotune_module, "_make_tensors", lambda task, batch: (q, k, v))
  monkeypatch.setattr(autotune_module, "_make_attn_bias", lambda task: None)
  monkeypatch.setattr(autotune_module, "_get_decode_num_splits", lambda *args, **kwargs: 1)
  monkeypatch.setattr(autotune_module, "is_sm90_tma_forward_supported", lambda *args, **kwargs: True)
  monkeypatch.setattr(autotune_module, "ffpa_attn_func", fake_ffpa_attn_func)
  monkeypatch.setattr(autotune_module, "_get_fwd_sm90_autotune", lambda *args, **kwargs: wrapper)

  entries = {}
  tuned_entries = _tune_forward(task, 1, "fast", entries, enable_tma=True, enable_ws=True)

  entry, choices_count = tuned_entries[0]
  assert choices_count == 1
  assert entry["kernel"] == "fwd_sm90_generic"
  assert entry["enable_tma"] is True
  assert entry["enable_ws"] is True
  assert entry["config"]["warp_specialize"] is True
  assert seen_kwargs["enable_tma"] is True
  assert seen_kwargs["enable_ws"] is True
  assert list(entries.values()) == [entry]


@pytest.mark.parametrize(
  "task",
  [
    TuneTask("forward", torch.float16, 320, 512, 512, True, 8, 8),
    TuneTask("forward", torch.float16, 320, 512, 512, False, 8, 8, has_attn_bias=True),
    TuneTask("forward", torch.float16, 320, 512, 512, False, 8, 8, has_dropout=True),
  ],
)
def test_persistent_tune_forward_allows_sm90_ws_for_masked_variants(monkeypatch, task):
  q = torch.empty(1, 8, 512, 320)
  k = torch.empty_like(q)
  v = torch.empty_like(q)
  config = triton.Config(
    {
      "BLOCK_M": 64,
      "BLOCK_N": 128,
      "BLOCK_HEADDIM_QK": 64,
      "BLOCK_HEADDIM_V": 64,
      "warp_specialize": False,
    },
    num_warps=4,
    num_stages=2,
  )
  wrapper = SimpleNamespace(best_config=config, configs=[config])
  seen = {"get_ws": None, "call_ws": None}

  def fake_get_sm90(*args, **kwargs):
    del args
    seen["get_ws"] = kwargs["enable_ws"]
    return wrapper

  def fake_ffpa_attn_func(*args, **kwargs):
    del args
    seen["call_ws"] = kwargs["enable_ws"]
    return torch.empty_like(q)

  monkeypatch.setattr(autotune_module, "_make_tensors", lambda task, batch: (q, k, v))
  monkeypatch.setattr(autotune_module, "_make_attn_bias", lambda task: torch.empty(1) if task.has_attn_bias else None)
  monkeypatch.setattr(autotune_module, "_get_decode_num_splits", lambda *args, **kwargs: 1)
  monkeypatch.setattr(autotune_module, "is_sm90_tma_forward_supported", lambda *args, **kwargs: True)
  monkeypatch.setattr(autotune_module, "ffpa_attn_func", fake_ffpa_attn_func)
  monkeypatch.setattr(autotune_module, "_get_fwd_sm90_autotune", fake_get_sm90)

  _tune_forward(task, 1, "fast", {}, enable_tma=True, enable_ws=True)

  assert seen == {"get_ws": True, "call_ws": True}


def test_bwd_non_main_modes_use_same_preprocess_configs():
  fast = _gen_pre_autotune_configs(d_chunk=False, autotune_mode="fast")
  max_configs = _gen_pre_autotune_configs(d_chunk=False, autotune_mode="max")
  assert [config_from_triton_config(config)
          for config in fast] == [config_from_triton_config(config) for config in max_configs]


@pytest.mark.parametrize("use_gemv", [False, True])
def test_bwd_non_main_modes_use_same_decode_configs(use_gemv):
  fast = _gen_decode_bwd_stage1_autotune_configs(512, use_gemv=use_gemv, autotune_mode="fast")
  max_configs = _gen_decode_bwd_stage1_autotune_configs(512, use_gemv=use_gemv, autotune_mode="max")
  assert [config_from_triton_config(config)
          for config in fast] == [config_from_triton_config(config) for config in max_configs]


def test_bwd_fast_mode_prunes_kernel_configs(monkeypatch):
  monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
  fast = _gen_bwd_autotune_configs((64, ), autotune_mode="fast")
  max_configs = _gen_bwd_autotune_configs((64, 128), autotune_mode="max")
  assert len(fast) < len(max_configs)


def test_bwd_max_mode_keeps_main_kernel_search_light_on_sm90(monkeypatch):
  monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
  configs = _gen_bwd_autotune_configs((64, 128), autotune_mode="max")
  serialized = [config_from_triton_config(config) for config in configs]
  assert len(serialized) == 48
  assert {config["BLOCK_HEADDIM"] for config in serialized} == {64, 128, 256}
  assert {config["num_stages"] for config in serialized} == {2, 3}


def test_bwd_max_mode_matches_fast_kernel_search_below_sm90(monkeypatch):
  monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 9))
  fast = _gen_bwd_autotune_configs((64, ), autotune_mode="fast")
  max_configs = _gen_bwd_autotune_configs((64, 128), autotune_mode="max")
  assert [config_from_triton_config(config)
          for config in max_configs] == [config_from_triton_config(config) for config in fast]


def test_forward_autotune_keys_include_causal():
  expected_keys = {"autotune_seqlen_q_bucket", "autotune_seqlen_k_bucket", "autotune_causal_key"}
  assert expected_keys <= set(_get_fwd_autotune(320, "fast", "bf16").keys)
  assert expected_keys <= set(_get_fwd_sm90_autotune(320, "fast", "bf16", enable_ws=False).keys)
  assert expected_keys <= set(_get_decode_fwd_stage1_autotune(320, True, "fast", "bf16").keys)


def test_autotune_wrappers_are_dtype_scoped():
  assert _get_fwd_autotune(320, "fast", "bf16") is not _get_fwd_autotune(320, "fast", "fp16")
  assert _get_fwd_sm90_autotune(320, "fast", "bf16",
                                enable_ws=False) is not _get_fwd_sm90_autotune(320, "fast", "fp16", enable_ws=False)
  assert _get_decode_fwd_stage1_autotune(320, True, "fast",
                                         "bf16") is not _get_decode_fwd_stage1_autotune(320, True, "fast", "fp16")
  assert _get_pre_autotune(False, "fast", "bf16") is not _get_pre_autotune(False, "fast", "fp16")


def test_persistent_autotune_decode_tasks_skip_nkv1_and_nq4():
  dtypes = [torch.bfloat16]
  seqlens = [1, 4, 512]

  forward_decode_tasks = [task for task in _iter_forward_tasks(dtypes, seqlens) if task.seqlen_q == 1]
  backward_decode_tasks = [task for task in _iter_backward_tasks(dtypes, seqlens) if task.seqlen_q < 8]

  assert forward_decode_tasks
  assert backward_decode_tasks
  assert all(task.seqlen_q == 1 and task.seqlen_k > 1 for task in forward_decode_tasks)
  assert all(task.seqlen_q == 1 and task.seqlen_k > 1 for task in backward_decode_tasks)


def test_persistent_autotune_tasks_start_with_common_prefill():
  dtypes = [torch.bfloat16]
  seqlens = [1, 512, 1024]

  forward_tasks = _iter_forward_tasks(dtypes, seqlens, heads=8, full_tasks=True)
  backward_tasks = _iter_backward_tasks(dtypes, seqlens, heads=8, full_tasks=True)

  assert forward_tasks[0].case_name == "common"
  assert forward_tasks[0].seqlen_q >= 512
  assert backward_tasks[0].case_name == "common"
  assert backward_tasks[0].seqlen_q >= 512
  first_forward_full = next(index for index, task in enumerate(forward_tasks) if task.case_name == "attn-mask")
  first_backward_full = next(index for index, task in enumerate(backward_tasks) if task.case_name == "attn-mask")
  assert any(task.case_name == "decode-attn" for task in forward_tasks[:first_forward_full])
  assert any(task.case_name == "decode-attn" for task in backward_tasks[:first_backward_full])


def test_persistent_autotune_default_tasks_keep_baseline_variants():
  dtypes = [torch.bfloat16]
  seqlens = [1, 512]

  tasks = _iter_forward_tasks(dtypes, seqlens, heads=8) + _iter_backward_tasks(dtypes, seqlens, heads=8)

  assert tasks
  assert all(task.nheads_q == 8 and task.nheads_kv == 8 for task in tasks)
  assert all(not task.has_attn_bias for task in tasks)
  assert all(not task.has_dropout for task in tasks)


def test_persistent_autotune_full_tasks_add_mask_dropout_gqa_mqa():
  dtypes = [torch.bfloat16]
  seqlens = [1, 512]

  tasks = _iter_forward_tasks(dtypes, seqlens, heads=8, full_tasks=True)
  cases = {task.case_name for task in tasks}

  assert {"attn-mask", "dropout", "gqa", "mqa"}.issubset(cases)
  assert any(task.case_name == "attn-mask" and task.has_attn_bias for task in tasks)
  assert any(task.case_name == "dropout" and task.has_dropout for task in tasks)
  assert any(task.case_name == "gqa" and task.nheads_q == 8 and 1 < task.nheads_kv < 8 for task in tasks)
  assert any(task.case_name == "mqa" and task.nheads_q == 8 and task.nheads_kv == 1 for task in tasks)


def test_triton_config_serialization_round_trip_shape():
  config = _gen_bwd_autotune_configs((64, ), autotune_mode="fast")[0]
  serialized = config_from_triton_config(config)
  assert serialized["BLOCK_M"] in (64, 128)
  assert serialized["BLOCK_N"] == 64
  assert "num_warps" in serialized

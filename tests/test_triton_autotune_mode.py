"""Unit tests for Triton FFPA autotune mode plumbing and search-space pruning."""

import pytest
import torch

from ffpa_attn.autotune import _iter_backward_tasks, _iter_forward_tasks
from ffpa_attn.functional import FFPAAttnMeta
from ffpa_attn.triton._autotune_utils import autotune_seqlen_key, bucket_autotune_seqlen, exact_autotune_seqlen_keys
from ffpa_attn.triton._ffpa_bwd import _gen_bwd_autotune_configs, _gen_pre_autotune_configs, _get_pre_autotune
from ffpa_attn.triton._persistent_autotune import config_from_triton_config
from ffpa_attn.triton._ffpa_fwd import (
  _gen_decode_fwd_stage1_autotune_configs,
  _gen_fwd_autotune_configs,
  _get_decode_fwd_stage1_autotune,
  _get_fwd_autotune,
)


def test_triton_autotune_mode_defaults_to_fast():
  meta = FFPAAttnMeta.from_kwargs()
  assert meta.triton_autotune_mode == "fast"


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


def test_bwd_fast_mode_prunes_preprocess_configs():
  fast = _gen_pre_autotune_configs(d_chunk=False, autotune_mode="fast")
  max_configs = _gen_pre_autotune_configs(d_chunk=False, autotune_mode="max")
  assert len(fast) < len(max_configs)


def test_bwd_fast_mode_prunes_kernel_configs():
  fast = _gen_bwd_autotune_configs((64, 128), headdim=512, autotune_mode="fast")
  max_configs = _gen_bwd_autotune_configs((64, 128), headdim=512, autotune_mode="max")
  assert len(fast) < len(max_configs)


def test_forward_autotune_keys_include_causal():
  assert "autotune_causal_key" in _get_fwd_autotune(320, "fast", "bf16").keys
  assert "autotune_causal_key" in _get_decode_fwd_stage1_autotune(320, True, "fast", "bf16").keys


def test_autotune_wrappers_are_dtype_scoped():
  assert _get_fwd_autotune(320, "fast", "bf16") is not _get_fwd_autotune(320, "fast", "fp16")
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
  config = _gen_bwd_autotune_configs((64, ), headdim=512, autotune_mode="fast")[0]
  serialized = config_from_triton_config(config)
  assert serialized["BLOCK_M"] in (64, 128)
  assert serialized["BLOCK_N"] == 64
  assert "num_warps" in serialized

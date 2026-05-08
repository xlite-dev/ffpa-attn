"""Unit tests for Triton FFPA autotune mode plumbing and search-space pruning."""

import pytest

from ffpa_attn.functional import FFPAAttnMeta
from ffpa_attn.triton._autotune_utils import bucket_autotune_seqlen
from ffpa_attn.triton._ffpa_bwd import _gen_bwd_autotune_configs, _gen_pre_autotune_configs
from ffpa_attn.triton._ffpa_fwd import _gen_decode_fwd_stage1_autotune_configs, _gen_fwd_autotune_configs


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


@pytest.mark.parametrize("headdim", [320, 512])
def test_fwd_fast_mode_prunes_generic_configs(headdim):
  fast = _gen_fwd_autotune_configs(headdim, autotune_mode="fast")
  max_configs = _gen_fwd_autotune_configs(headdim, autotune_mode="max")
  assert len(fast) < len(max_configs)
  assert all(config.num_warps == 4 for config in fast)


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

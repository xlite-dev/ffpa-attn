"""Unit tests for Triton FFPA autotune mode plumbing and search-space pruning."""

import pytest

from ffpa_attn.functional import FFPAAttnMeta
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


@pytest.mark.parametrize("headdim", [320, 512])
def test_fwd_fast_mode_prunes_generic_configs(headdim):
  fast = _gen_fwd_autotune_configs(headdim, autotune_mode="fast")
  max_configs = _gen_fwd_autotune_configs(headdim, autotune_mode="max")
  assert len(fast) < len(max_configs)


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

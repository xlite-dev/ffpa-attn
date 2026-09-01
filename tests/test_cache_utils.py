"""Host-only contract tests for the compiled-kernel cache factory."""

import importlib
import os

import pytest

import ffpa_attn.cute.utils.cache_utils as cache_utils

_ENV_VARS = (
  "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED",
  "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR",
)


@pytest.fixture()
def fresh_cache_utils():
  """Reload ``cache_utils`` under a controlled environment; restore after."""
  saved = {k: os.environ.get(k) for k in _ENV_VARS}

  def _reload(*, enabled, cache_dir=None):
    os.environ["FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED"] = (
      "1" if enabled else "0"
    )
    if cache_dir is None:
      os.environ.pop("FLASH_ATTENTION_CUTE_DSL_CACHE_DIR", None)
    else:
      os.environ["FLASH_ATTENTION_CUTE_DSL_CACHE_DIR"] = str(cache_dir)
    return importlib.reload(cache_utils)

  try:
    yield _reload
  finally:
    for key, value in saved.items():
      if value is None:
        os.environ.pop(key, None)
      else:
        os.environ[key] = value
    importlib.reload(cache_utils)


def test_disabled_flag_is_silent(fresh_cache_utils, recwarn):
  fresh_cache_utils(enabled=False)
  assert [w for w in recwarn if issubclass(w.category, UserWarning)] == []


def test_enabled_flag_is_an_inert_no_op(fresh_cache_utils, tmp_path):
  cache_dir = tmp_path / "cache"
  with pytest.warns(UserWarning, match="is inert"):
    mod = fresh_cache_utils(enabled=True, cache_dir=cache_dir)
  cache = mod.get_jit_cache("fwd_sm100")
  assert type(cache) is mod.JITCache
  assert not cache_dir.exists()

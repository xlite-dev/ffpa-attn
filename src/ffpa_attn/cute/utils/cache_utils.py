# Derived from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/cache_utils.py
# Compiled-kernel cache: an in-process dict of ``JitCompiledFunction``
# wrappers.  Cross-process persistence is retired: raw ``export_to_c``
# callables expose the fixed parameter ABI and crash
# wrapper-convention call sites (``TypeError: Expects 17 parameters``),
# and the CUTE DSL file cache cannot take over while ``cute.compile``
# hardcodes ``no_cache=True``.
import os
import warnings
from typing import Hashable, TypeAlias

from cutlass.cutlass_dsl import JitCompiledFunction

CompileKeyType: TypeAlias = tuple[Hashable, ...]

# Inert legacy flag; FLASH_ATTENTION_CUTE_DSL_CACHE_DIR is likewise ignored.
# Warn rather than log: setting it is a request for persistence that no
# longer exists, and the operator must hear that at the default verbosity.
CUTE_DSL_CACHE_ENABLED: bool = os.getenv(
  "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED", "0"
) == "1"
if CUTE_DSL_CACHE_ENABLED:
  warnings.warn(
    "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 is inert: cross-process kernel "
    "persistence was removed, so the compiled-kernel cache is in-process only "
    "and FLASH_ATTENTION_CUTE_DSL_CACHE_DIR is ignored.",
    UserWarning,
  )


class JITCache:
  """
    In-memory cache for compiled functions.
    """

  def __init__(self):
    self.cache: dict[CompileKeyType, JitCompiledFunction] = {}

  def __setitem__(self, key: CompileKeyType, fn: JitCompiledFunction) -> None:
    self.cache[key] = fn

  def __getitem__(self, key: CompileKeyType) -> JitCompiledFunction:
    return self.cache[key]

  def __contains__(self, key: CompileKeyType) -> bool:
    return key in self.cache

  def clear(self) -> None:
    """
        Clear in-memory cache of compiled functions
        """
    self.cache.clear()


def get_jit_cache(name: str | None = None) -> JITCache:
  """JIT cache factory; ``name`` (a kernel-family tag) is kept for
  interface stability."""
  del name
  return JITCache()

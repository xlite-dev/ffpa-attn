"""Helpers for Triton autotune key bucketing."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_AUTOTUNE_SEQLEN_BUCKET_SIZE = 1024
_AUTOTUNE_SEQLEN_BUCKET_CAP = 8192
_AUTOTUNE_MAX_SEQLEN_BUCKET_CAP = 16384
_EXACT_AUTOTUNE_SEQLEN_KEYS: ContextVar[bool] = ContextVar("ffpa_exact_autotune_seqlen_keys", default=False)


def _bucket_upper_edge(seqlen: int, bucket_size: int) -> int:
  """Return the upper edge of the bucket containing ``seqlen``.

  :param seqlen: Positive runtime sequence length.
  :param bucket_size: Positive bucket size.
  :return: Upper bucket edge.
  """
  return ((seqlen - 1) // bucket_size + 1) * bucket_size


def bucket_autotune_seqlen(seqlen: int, autotune_mode: str = "fast") -> int:
  """Bucket a sequence length for autotune cache-key reuse.

  Fast mode preserves the current coarse autotune-cache reuse strategy: runtime
  sequence lengths are grouped in 1024-sized bins using the upper bin edge as
  the representative key, and values above 8k are all mapped to the same 8k
  bucket.

  Max mode uses finer-grained piecewise upper-edge buckets:

  * ``<= 512`` uses bucket size ``64``
  * ``(512, 1024]`` uses bucket size ``128``
  * ``(1024, 2048]`` uses bucket size ``256``
  * ``[2048, 8192]`` uses bucket size ``512``
  * ``(8192, 16384]`` uses bucket size ``1024``
  * values above ``16384`` reuse the ``16384`` bucket

  Examples:

  * ``bucket_autotune_seqlen(8191, "fast") -> 8192``
  * ``bucket_autotune_seqlen(8193, "fast") -> 8192``
  * ``bucket_autotune_seqlen(513, "max") -> 640``
  * ``bucket_autotune_seqlen(1025, "max") -> 1280``
  * ``bucket_autotune_seqlen(8193, "max") -> 9216``
  * ``bucket_autotune_seqlen(20000, "max") -> 16384``

  :param seqlen: Runtime sequence length.
  :param autotune_mode: Triton autotune search-space mode, ``"fast"`` or
    ``"max"``.
  :return: Bucketed autotune key value.
  """
  if seqlen <= 0:
    raise ValueError(f"Expected positive sequence length, got {seqlen}")
  if autotune_mode == "fast":
    if seqlen > _AUTOTUNE_SEQLEN_BUCKET_CAP:
      return _AUTOTUNE_SEQLEN_BUCKET_CAP
    return _bucket_upper_edge(seqlen, _AUTOTUNE_SEQLEN_BUCKET_SIZE)
  if autotune_mode == "max":
    if seqlen <= 512:
      return _bucket_upper_edge(seqlen, 64)
    if seqlen <= 1024:
      return _bucket_upper_edge(seqlen, 128)
    if seqlen < 2048:
      return _bucket_upper_edge(seqlen, 256)
    if seqlen <= 8192:
      return _bucket_upper_edge(seqlen, 512)
    if seqlen <= _AUTOTUNE_MAX_SEQLEN_BUCKET_CAP:
      return _bucket_upper_edge(seqlen, 1024)
    return _AUTOTUNE_MAX_SEQLEN_BUCKET_CAP
  raise ValueError(f"Unsupported autotune_mode={autotune_mode!r}; choose 'fast' or 'max'.")


def autotune_seqlen_key(seqlen: int, autotune_mode: str = "fast") -> int:
  """Return the sequence-length key used by Triton autotune wrappers.

  Runtime autotune uses bucketed keys to reduce repeated online tuning. The
  persistent autotune generator switches this helper to exact keys so every
  target grid shape is independently benchmarked before being written to JSON.

  :param seqlen: Runtime sequence length.
  :param autotune_mode: Triton autotune search-space mode.
  :return: Exact or bucketed autotune key.
  """
  if _EXACT_AUTOTUNE_SEQLEN_KEYS.get():
    if seqlen <= 0:
      raise ValueError(f"Expected positive sequence length, got {seqlen}")
    return seqlen
  return bucket_autotune_seqlen(seqlen, autotune_mode)


@contextmanager
def exact_autotune_seqlen_keys() -> Iterator[None]:
  """Use exact sequence lengths as Triton autotune keys inside the context."""
  token = _EXACT_AUTOTUNE_SEQLEN_KEYS.set(True)
  try:
    yield
  finally:
    _EXACT_AUTOTUNE_SEQLEN_KEYS.reset(token)

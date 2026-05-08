"""Helpers for Triton autotune key bucketing."""

from __future__ import annotations

_AUTOTUNE_SEQLEN_BUCKET_SIZE = 1024
_AUTOTUNE_SEQLEN_BUCKET_CAP = 8192


def bucket_autotune_seqlen(seqlen: int) -> int:
  """Bucket a sequence length for autotune cache-key reuse.

  Sequence lengths are grouped in 1024-sized bins using the upper bin edge as
  the representative key. Values above 8k are all mapped to the same 8k bucket.

  Examples:

  * ``8191 -> 8192``
  * ``8192 -> 8192``
  * ``8193 -> 8192``

  :param seqlen: Runtime sequence length.
  :return: Bucketed autotune key value.
  """
  if seqlen <= 0:
    raise ValueError(f"Expected positive sequence length, got {seqlen}")
  if seqlen > _AUTOTUNE_SEQLEN_BUCKET_CAP:
    return _AUTOTUNE_SEQLEN_BUCKET_CAP
  return ((seqlen - 1) // _AUTOTUNE_SEQLEN_BUCKET_SIZE + 1) * _AUTOTUNE_SEQLEN_BUCKET_SIZE

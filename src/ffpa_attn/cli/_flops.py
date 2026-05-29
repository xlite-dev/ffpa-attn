"""Shared attention GEMM FLOPs helpers for benchmark reporting.

These helpers intentionally report only the dominant GEMM work in attention,
not the full kernel instruction count. They are used to derive approximate
forward/backward TFLOPS from measured latency in the example benchmarks.

References: https://github.com/Dao-AILab/flash-attention/blob/main/benchmarks/benchmark_flash_attention.py
"""

from __future__ import annotations

import math


def attention_valid_pairs(nq: int, nkv: int, causal: bool) -> int:
  """Return the logical number of query/key pairs evaluated by attention.

  FFPA's non-square causal semantics align the query rows to the tail of the KV
  sequence. For row ``i`` in ``[0, nq)``, valid columns satisfy
  ``col <= i + (nkv - nq)``. This helper counts those valid pairs exactly.

  :param nq: Query sequence length.
  :param nkv: Key/value sequence length.
  :param causal: Whether causal masking is active.
  :return: Number of valid query/key pairs.
  """
  if not causal:
    return nq * nkv

  total = 0
  kv_offset = nkv - nq
  for row_idx in range(nq):
    total += max(0, min(nkv, row_idx + kv_offset + 1))
  return total


def attention_fwd_flops(
  batch: int, num_heads_q: int, nq: int, nkv: int, headdim: int, causal: bool
) -> int:
  """Return the theoretical forward dominant-GEMM FLOPs.

  Forward attention is approximated as the two GEMMs ``QK^T`` and ``PV``.

  :param batch: Batch size.
  :param num_heads_q: Logical query-head count.
  :param nq: Query sequence length.
  :param nkv: Key/value sequence length.
  :param headdim: Head dimension.
  :param causal: Whether causal masking is active.
  :return: Theoretical forward FLOPs.
  """
  valid_pairs = attention_valid_pairs(nq, nkv, causal)
  return 4 * batch * num_heads_q * headdim * valid_pairs


def attention_bwd_flops(
  batch: int, num_heads_q: int, nq: int, nkv: int, headdim: int, causal: bool
) -> int:
  """Return the theoretical backward dominant-GEMM FLOPs.

  Backward attention is approximated as five dominant GEMMs: one ``QK^T``
  recompute plus the four large backward matrix multiplies. This is ``2.5x``
  the forward dominant-GEMM work.

  :param batch: Batch size.
  :param num_heads_q: Logical query-head count.
  :param nq: Query sequence length.
  :param nkv: Key/value sequence length.
  :param headdim: Head dimension.
  :param causal: Whether causal masking is active.
  :return: Theoretical backward FLOPs.
  """
  return 5 * attention_fwd_flops(
    batch, num_heads_q, nq, nkv, headdim, causal
  ) // 2


def tflops_from_ms(flops: int, latency_ms: float | None) -> float | None:
  """Convert theoretical FLOPs and measured latency into TFLOPS.

  :param flops: Theoretical FLOP count.
  :param latency_ms: Measured latency in milliseconds.
  :return: TFLOPS value, or ``None`` when latency is invalid.
  """
  if latency_ms is None or latency_ms <= 0 or not math.isfinite(latency_ms):
    return None
  return flops / (latency_ms * 1.0e9)


def format_tflops_short(tflops: float | None) -> str:
  """Format one TFLOPS value using the compact ``90T``-style notation.

  :param tflops: TFLOPS value.
  :return: Compact string or ``-`` when unavailable.
  """
  if tflops is None or not math.isfinite(tflops):
    return "-"
  if tflops >= 10.0:
    return f"{tflops:.0f}T"
  if tflops >= 1.0:
    return f"{tflops:.1f}T"
  return f"{tflops:.2f}T"

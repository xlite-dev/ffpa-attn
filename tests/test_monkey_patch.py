"""Tests for monkey-patching ``torch.nn.functional.scaled_dot_product_attention``.

These tests lock in the public usage pattern documented in
``ffpa_attn_interface.py``:

    import torch.nn.functional as F
    from ffpa_attn import ffpa_attn_func
    F.scaled_dot_product_attention = ffpa_attn_func

Large-D cases must still dispatch through FFPA after the patch, while fallback
cases must call the native SDPA op directly instead of recursing through the
patched Python symbol.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from ffpa_attn import ffpa_attn_func

# ROCm detection: dropout mask RNG differs between Triton-AMD and native SDPA
IS_ROCM = hasattr(torch.version, 'hip') and torch.version.hip is not None


def _native_sdpa(*args, **kwargs):
  return torch._C._nn.scaled_dot_product_attention(*args, **kwargs)


def _alloc_qkv(dtype: torch.dtype, headdim: int = 128):
  torch.manual_seed(0)
  q = torch.randn(1, 4, 64, headdim, dtype=dtype, device="cuda")
  k = torch.randn(1, 4, 64, headdim, dtype=dtype, device="cuda")
  v = torch.randn(1, 4, 64, headdim, dtype=dtype, device="cuda")
  return q, k, v


def _alloc_cross_qkv(dtype: torch.dtype, nq: int, nkv: int, headdim: int = 512):
  torch.manual_seed(0)
  q = torch.randn(1, 4, nq, headdim, dtype=dtype, device="cuda")
  k = torch.randn(1, 4, nkv, headdim, dtype=dtype, device="cuda")
  v = torch.randn(1, 4, nkv, headdim, dtype=dtype, device="cuda")
  return q, k, v


def _tail_aligned_causal_mask(nq: int, nkv: int) -> torch.Tensor:
  row_idx = torch.arange(nq, device="cuda").view(-1, 1)
  col_idx = torch.arange(nkv, device="cuda").view(1, -1)
  return col_idx <= (row_idx + (nkv - nq))


def _tolerance(dtype: torch.dtype) -> dict[str, float]:
  return {
    "atol": 2e-2,
    "rtol": 2e-2
  } if dtype == torch.bfloat16 else {
    "atol": 1e-2,
    "rtol": 1e-2
  }


def _block_native_sdpa(*args, **kwargs):
  del args, kwargs
  raise AssertionError(
    "large-D monkey-patched case unexpectedly fell back to native SDPA"
  )


@pytest.fixture(scope="module", autouse=True)
def _require_cuda():
  if not torch.cuda.is_available():
    pytest.skip("CUDA not available")


@pytest.mark.parametrize(
  "dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"]
)
def test_monkey_patched_sdpa_small_d_fallback_matches_native(
  dtype, monkeypatch
):
  q, k, v = _alloc_qkv(dtype, headdim=128)
  ref = _native_sdpa(q, k, v, scale=1.0 / math.sqrt(q.size(-1)))

  monkeypatch.setattr(F, "scaled_dot_product_attention", ffpa_attn_func)
  out = F.scaled_dot_product_attention(
    q, k, v, scale=1.0 / math.sqrt(q.size(-1))
  )

  torch.testing.assert_close(out, ref, **_tolerance(dtype))


@pytest.mark.parametrize(
  "dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"]
)
@pytest.mark.parametrize(
  "case", ["self", "cross", "causal", "dropout", "attn_mask"]
)
def test_monkey_patched_sdpa_large_d_ffpa_paths_match_native(
  dtype, case, monkeypatch
):
  if IS_ROCM and case == "dropout":
    pytest.skip("Dropout mask RNG differs between Triton-AMD and native SDPA")
  if case == "cross":
    q, k, v = _alloc_cross_qkv(dtype, nq=512, nkv=1024)
  else:
    q, k, v = _alloc_cross_qkv(dtype, nq=512, nkv=512)

  scale = 1.0 / math.sqrt(q.size(-1))
  kwargs: dict[str, object] = {"scale": scale}
  if case == "causal":
    kwargs["is_causal"] = True
  elif case == "dropout":
    kwargs["dropout_p"] = 0.2
  elif case == "attn_mask":
    kwargs["attn_mask"] = _tail_aligned_causal_mask(q.size(2), k.size(2))

  if case == "dropout":
    torch.manual_seed(1234)
  ref = _native_sdpa(q, k, v, **kwargs)

  # If this call accidentally routes to fallback, it will hit the blocked
  # native op below.  That keeps this test focused on the monkey-patched FFPA
  # path instead of merely comparing two native SDPA calls.
  monkeypatch.setattr(F, "scaled_dot_product_attention", ffpa_attn_func)
  monkeypatch.setattr(
    torch._C._nn, "scaled_dot_product_attention", _block_native_sdpa
  )
  if case == "dropout":
    torch.manual_seed(1234)
  out = F.scaled_dot_product_attention(q, k, v, **kwargs)

  tol = {"atol": 5e-2, "rtol": 5e-2} if case == "dropout" else _tolerance(dtype)
  torch.testing.assert_close(out, ref, **tol)

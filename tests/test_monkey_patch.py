"""Tests for monkey-patching ``torch.nn.functional.scaled_dot_product_attention``.

These tests lock in the public usage pattern documented in
``ffpa_attn_interface.py``:

    import torch.nn.functional as F
    from ffpa_attn import ffpa_attn_func
    F.scaled_dot_product_attention = ffpa_attn_func

The fallback path inside ``ffpa_attn_func`` must call the native SDPA op
directly instead of recursing through the patched Python symbol.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from ffpa_attn import ffpa_attn_func


def _native_sdpa(*args, **kwargs):
  return torch._C._nn.scaled_dot_product_attention(*args, **kwargs)


def _alloc_qkv(dtype: torch.dtype, headdim: int = 128):
  torch.manual_seed(0)
  q = torch.randn(1, 4, 64, headdim, dtype=dtype, device="cuda")
  k = torch.randn(1, 4, 64, headdim, dtype=dtype, device="cuda")
  v = torch.randn(1, 4, 64, headdim, dtype=dtype, device="cuda")
  return q, k, v


def _tolerance(dtype: torch.dtype) -> dict[str, float]:
  return {"atol": 2e-2, "rtol": 2e-2} if dtype == torch.bfloat16 else {"atol": 1e-2, "rtol": 1e-2}


@pytest.fixture(scope="module", autouse=True)
def _require_cuda():
  if not torch.cuda.is_available():
    pytest.skip("CUDA not available")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_monkey_patched_sdpa_small_d_fallback_matches_native(dtype, monkeypatch):
  q, k, v = _alloc_qkv(dtype, headdim=128)
  ref = _native_sdpa(q, k, v, scale=1.0 / math.sqrt(q.size(-1)))

  monkeypatch.setattr(F, "scaled_dot_product_attention", ffpa_attn_func)
  out = F.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(q.size(-1)))

  torch.testing.assert_close(out, ref, **_tolerance(dtype))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_monkey_patched_sdpa_attn_mask_fallback_matches_native(dtype, monkeypatch):
  q, k, v = _alloc_qkv(dtype, headdim=512)
  row_idx = torch.arange(q.size(2), device=q.device).view(-1, 1)
  col_idx = torch.arange(k.size(2), device=k.device).view(1, -1)
  attn_mask = col_idx <= row_idx
  ref = _native_sdpa(q, k, v, attn_mask=attn_mask, scale=1.0 / math.sqrt(q.size(-1)))

  monkeypatch.setattr(F, "scaled_dot_product_attention", ffpa_attn_func)
  out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=1.0 / math.sqrt(q.size(-1)))

  torch.testing.assert_close(out, ref, **_tolerance(dtype))

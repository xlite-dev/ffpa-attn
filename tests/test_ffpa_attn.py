"""FFPA attention kernel unit tests.

Tests the unified ``ffpa_attn_func`` dispatcher across fp16 / bf16 activations.
Heavy benchmarking lives under ``bench/bench_ffpa_attn.py``.
"""

import itertools
import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ffpa_attn import ffpa_attn_func  # noqa: E402

SEQLENS = [1024, 4096, 8192]
HEADDIMS = [64, 128, 320, 512, 640]
HEADNUMS = [8, 16, 32, 48]
DTYPES = [torch.float16, torch.bfloat16]

# Correctness subset: one representative shape per (dtype, headdim) category
# plus two longer-seqlen spot checks. Fast (<~25 s on L20) while covering
# both small_d (d<=128) and large_d (d>128) kernel paths.
CORRECTNESS_SHAPES = [
  (1, 8, 1024, 64),
  (1, 8, 1024, 128),
  (1, 16, 1024, 320),
  (1, 16, 1024, 512),
  (1, 32, 1024, 640),
  (1, 32, 4096, 128),
  (1, 48, 4096, 320),
]

# Dispatch/smoke: every (H, D) at N=1024 to verify each tile launches and
# produces finite output. Accuracy validated separately above.
DISPATCH_SHAPES = [(1, H, 1024, D) for H, D in itertools.product(HEADNUMS, HEADDIMS)]


def _sdpa_ref(Q, K, V):
  return F.scaled_dot_product_attention(Q, K, V, scale=1.0 / math.sqrt(Q.size(-1)))


def _tolerance(dtype):
  return {"atol": 2e-2, "rtol": 2e-2} if dtype == torch.bfloat16 else {"atol": 1e-2, "rtol": 1e-2}


def _alloc_qkv(B, H, N, D, dtype):
  torch.manual_seed(0)
  q = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
  k = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
  v = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
  return q, k, v


def _acc_for(dtype):
  # bf16 activations require acc='f32' (no bf16-acc mma PTX); fp16 also uses
  # f32 here for accuracy parity with the SDPA reference.
  return "f32"


@pytest.fixture(scope="module", autouse=True)
def _require_cuda():
  if not torch.cuda.is_available():
    pytest.skip("CUDA not available")


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("B,H,N,D", CORRECTNESS_SHAPES)
def test_ffpa_attn_func_matches_sdpa(dtype, B, H, N, D):
  q, k, v = _alloc_qkv(B, H, N, D, dtype)
  out = ffpa_attn_func(q, k, v, stages=2, acc=_acc_for(dtype))
  ref = _sdpa_ref(q, k, v)
  assert out.dtype == dtype
  assert out.shape == ref.shape
  assert torch.isfinite(out).all(), "FFPA output contains NaN/Inf"
  torch.testing.assert_close(out, ref, **_tolerance(dtype))


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("B,H,N,D", DISPATCH_SHAPES)
def test_ffpa_attn_func_dispatch_shapes(dtype, B, H, N, D):
  q, k, v = _alloc_qkv(B, H, N, D, dtype)
  out = ffpa_attn_func(q, k, v, stages=2, acc=_acc_for(dtype))
  assert out.shape == q.shape
  assert out.dtype == dtype
  assert torch.isfinite(out).all()


def test_ffpa_attn_func_rejects_bf16_acc_f16():
  q, k, v = _alloc_qkv(1, 8, 1024, 128, torch.bfloat16)
  with pytest.raises(ValueError, match="bf16"):
    ffpa_attn_func(q, k, v, stages=2, acc="f16")


def test_ffpa_attn_func_rejects_unsupported_dtype():
  q, k, v = _alloc_qkv(1, 8, 1024, 128, torch.float32)
  with pytest.raises(TypeError):
    ffpa_attn_func(q, k, v, stages=2, acc="f32")


def test_ffpa_attn_func_rejects_invalid_acc():
  q, k, v = _alloc_qkv(1, 8, 1024, 128, torch.float16)
  with pytest.raises(ValueError, match="acc"):
    ffpa_attn_func(q, k, v, stages=2, acc="bad")


# Boundary shapes: seqlen not a multiple of Bc=64 (and/or Br=64). Covers
# partial first tile (N<64), single-tile tail, multi-tile + tail, and
# sizes near common power-of-two boundaries. Uses D=128 (small_d kernel)
# and D=256 (large_d kernel) for coverage of both kernel paths.
BOUNDARY_SEQLENS = [1, 17, 33, 63, 65, 100, 127, 129, 200, 1000, 2047, 4095, 5000]
BOUNDARY_HEADDIMS = [128, 256]
BOUNDARY_SHAPES = [(1, 4, N, D) for N, D in itertools.product(BOUNDARY_SEQLENS, BOUNDARY_HEADDIMS)]


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("B,H,N,D", BOUNDARY_SHAPES)
def test_ffpa_attn_func_boundary_seqlen(dtype, B, H, N, D):
  q, k, v = _alloc_qkv(B, H, N, D, dtype)
  out = ffpa_attn_func(q, k, v, stages=2, acc=_acc_for(dtype))
  ref = _sdpa_ref(q, k, v)
  assert out.shape == ref.shape
  assert torch.isfinite(out).all(), f"FFPA output has NaN/Inf at N={N}, D={D}"
  torch.testing.assert_close(out, ref, **_tolerance(dtype))


# Cross-attention: Nq may differ from Nkv; Nk==Nv required. Covers
# short-query / long-KV (decoding-style), long-query / short-KV, and
# non-aligned tail on both sides.
CROSS_SHAPES = [
  # (Nq, Nkv)
  (128, 1024),
  (128, 8192),
  (1024, 128),
  (1024, 8192),
  (8191, 8192),  # non-aligned Nq tail
  (8192, 8191),  # non-aligned Nkv tail
  (1, 4096),  # incremental-decoding-style single query
]
CROSS_HEADDIMS = [128, 256, 512]


def _alloc_cross_qkv(B, H, Nq, Nkv, D, dtype):
  torch.manual_seed(0)
  q = torch.randn(B, H, Nq, D, dtype=dtype, device="cuda")
  k = torch.randn(B, H, Nkv, D, dtype=dtype, device="cuda")
  v = torch.randn(B, H, Nkv, D, dtype=dtype, device="cuda")
  return q, k, v


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("Nq,Nkv", CROSS_SHAPES)
@pytest.mark.parametrize("D", CROSS_HEADDIMS)
def test_ffpa_attn_func_cross_attention(dtype, Nq, Nkv, D):
  B, H = 1, 4
  q, k, v = _alloc_cross_qkv(B, H, Nq, Nkv, D, dtype)
  out = ffpa_attn_func(q, k, v, stages=2, acc=_acc_for(dtype))
  ref = _sdpa_ref(q, k, v)
  assert out.shape == (B, H, Nq, D)
  assert out.dtype == dtype
  assert torch.isfinite(out).all(), f"FFPA output has NaN/Inf at Nq={Nq}, Nkv={Nkv}, D={D}"
  torch.testing.assert_close(out, ref, **_tolerance(dtype))


def test_ffpa_attn_func_rejects_mismatched_kv_seqlen():
  torch.manual_seed(0)
  q = torch.randn(1, 4, 128, 128, dtype=torch.float16, device="cuda")
  k = torch.randn(1, 4, 1024, 128, dtype=torch.float16, device="cuda")
  v = torch.randn(1, 4, 2048, 128, dtype=torch.float16, device="cuda")
  with pytest.raises(ValueError, match="seqlen"):
    ffpa_attn_func(q, k, v, stages=2, acc="f32")

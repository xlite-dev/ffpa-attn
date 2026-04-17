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


# GQA / MQA: Q has Nh_q heads, K/V have Nh_kv heads (Nh_q % Nh_kv == 0).
# group_size = Nh_q / Nh_kv; MHA is group_size=1, MQA is Nh_kv=1.
# Covers standard GQA ratios, MQA, and a mix with cross-attention seqlens.
GQA_HEAD_CONFIGS = [
  # (Nh_q, Nh_kv)
  (8, 1),  # MQA
  (16, 2),  # GQA (typical 8x)
  (32, 4),  # GQA (typical 8x)
  (32, 8),  # GQA (4x, Llama-3 style)
  (16, 16),  # MHA degenerate (group_size=1)
]
GQA_SEQLENS = [
  # (Nq, Nkv)
  (128, 128),
  (1024, 1024),
  (128, 8192),  # short Q, long KV (decoding-style)
  (1024, 4096),  # cross + GQA
]
GQA_HEADDIMS = [64, 128, 256, 512]


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("Nh_q,Nh_kv", GQA_HEAD_CONFIGS)
@pytest.mark.parametrize("Nq,Nkv", GQA_SEQLENS)
@pytest.mark.parametrize("D", GQA_HEADDIMS)
def test_ffpa_attn_func_gqa(dtype, Nh_q, Nh_kv, Nq, Nkv, D):
  B = 1
  torch.manual_seed(0)
  q = torch.randn(B, Nh_q, Nq, D, dtype=dtype, device="cuda")
  k = torch.randn(B, Nh_kv, Nkv, D, dtype=dtype, device="cuda")
  v = torch.randn(B, Nh_kv, Nkv, D, dtype=dtype, device="cuda")
  out = ffpa_attn_func(q, k, v, stages=2, acc=_acc_for(dtype))
  group_size = Nh_q // Nh_kv
  k_ref = k.repeat_interleave(group_size, dim=1)
  v_ref = v.repeat_interleave(group_size, dim=1)
  ref = _sdpa_ref(q, k_ref, v_ref)
  assert out.shape == (B, Nh_q, Nq, D)
  assert out.dtype == dtype
  assert torch.isfinite(out
                        ).all(), (f"FFPA output has NaN/Inf at Nh_q={Nh_q}, Nh_kv={Nh_kv}, Nq={Nq}, Nkv={Nkv}, D={D}")
  torch.testing.assert_close(out, ref, **_tolerance(dtype))


def test_ffpa_attn_func_rejects_indivisible_num_heads():
  torch.manual_seed(0)
  # Nh_q=12, Nh_kv=8 -> 12 % 8 != 0
  q = torch.randn(1, 12, 128, 128, dtype=torch.float16, device="cuda")
  k = torch.randn(1, 8, 128, 128, dtype=torch.float16, device="cuda")
  v = torch.randn(1, 8, 128, 128, dtype=torch.float16, device="cuda")
  with pytest.raises(ValueError, match="num_heads"):
    ffpa_attn_func(q, k, v, stages=2, acc="f32")


def test_ffpa_attn_func_rejects_mismatched_kv_num_heads():
  torch.manual_seed(0)
  q = torch.randn(1, 16, 128, 128, dtype=torch.float16, device="cuda")
  k = torch.randn(1, 4, 128, 128, dtype=torch.float16, device="cuda")
  v = torch.randn(1, 2, 128, 128, dtype=torch.float16, device="cuda")
  with pytest.raises(ValueError, match="num_heads"):
    ffpa_attn_func(q, k, v, stages=2, acc="f32")


# Causal attention: queries are aligned to the tail of the KV sequence so
# Q row ``r`` only attends to ``k <= r + (Nkv - Nq)`` (the standard
# FlashAttention "causal" convention). Covers:
#   - self-attention causal (Nq == Nkv) with aligned and non-aligned seqlens,
#   - decoding-style short-Q / long-KV causal (Nq < Nkv, every Q row sees
#     the full KV prefix beyond the diagonal),
#   - both small-D (<= 128) and large-D kernel paths,
#   - combined with GQA.
CAUSAL_SELF_SHAPES = [
  # (N, D), Nq == Nkv
  (64, 128),
  (128, 64),
  (1024, 128),
  (1024, 256),
  (4096, 128),
  (127, 128),  # non-aligned tail
  (129, 256),  # crosses Br=Bc=64 boundary by one row
  # FFPA targets large headdim (FA-2 caps at 256); cover the D > 256 path too.
  (512, 320),
  (1024, 512),
  (512, 1024),
]
CAUSAL_CROSS_SHAPES = [
  # (Nq, Nkv, D); Nkv >= Nq required
  (1, 8192, 128),  # 1-token decoding
  (128, 1024, 128),  # short prefill
  (128, 8192, 256),  # short prefill, long context
  (1024, 4096, 128),  # chunked prefill
  (129, 2048, 512),  # non-aligned Nq + large D
  (128, 4096, 512),  # long context + D=512 (FFPA's sweet spot)
  (64, 2048, 1024),  # short decoding with maximum D=1024
]


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("N,D", CAUSAL_SELF_SHAPES)
def test_ffpa_attn_func_causal_self_attention(dtype, N, D):
  B, H = 1, 4
  q, k, v = _alloc_qkv(B, H, N, D, dtype)
  out = ffpa_attn_func(q, k, v, stages=2, acc=_acc_for(dtype), causal=True)
  ref = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0 / math.sqrt(D))
  assert out.shape == ref.shape
  assert out.dtype == dtype
  assert torch.isfinite(out).all(), f"FFPA causal output has NaN/Inf at N={N}, D={D}"
  torch.testing.assert_close(out, ref, **_tolerance(dtype))


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("Nq,Nkv,D", CAUSAL_CROSS_SHAPES)
def test_ffpa_attn_func_causal_cross_attention(dtype, Nq, Nkv, D):
  B, H = 1, 4
  q, k, v = _alloc_cross_qkv(B, H, Nq, Nkv, D, dtype)
  out = ffpa_attn_func(q, k, v, stages=2, acc=_acc_for(dtype), causal=True)
  # Reference: build an explicit attn mask where Q row r may attend to
  # KV positions k <= r + (Nkv - Nq). SDPA's is_causal only supports
  # the Nq == Nkv square case, so use attn_mask for the general case.
  kv_offset = Nkv - Nq
  row_idx = torch.arange(Nq, device="cuda").view(-1, 1)
  col_idx = torch.arange(Nkv, device="cuda").view(1, -1)
  attn_mask = (col_idx <= (row_idx + kv_offset))  # [Nq, Nkv] bool
  ref = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=1.0 / math.sqrt(D))
  assert out.shape == (B, H, Nq, D)
  assert out.dtype == dtype
  assert torch.isfinite(out).all(), (f"FFPA causal output has NaN/Inf at Nq={Nq}, Nkv={Nkv}, D={D}")
  torch.testing.assert_close(out, ref, **_tolerance(dtype))


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("Nh_q,Nh_kv", [(8, 1), (32, 4), (32, 8)])
@pytest.mark.parametrize(
  "Nq,Nkv,D",
  [
    (128, 128, 128),
    (1024, 1024, 128),
    (128, 4096, 256),
    # Large-headdim (> 256) causal + GQA — FFPA's core use case.
    (1024, 1024, 512),
    (128, 4096, 512),
    (512, 2048, 1024),
  ],
)
def test_ffpa_attn_func_causal_gqa(dtype, Nh_q, Nh_kv, Nq, Nkv, D):
  B = 1
  torch.manual_seed(0)
  q = torch.randn(B, Nh_q, Nq, D, dtype=dtype, device="cuda")
  k = torch.randn(B, Nh_kv, Nkv, D, dtype=dtype, device="cuda")
  v = torch.randn(B, Nh_kv, Nkv, D, dtype=dtype, device="cuda")
  out = ffpa_attn_func(q, k, v, stages=2, acc=_acc_for(dtype), causal=True)
  group_size = Nh_q // Nh_kv
  k_ref = k.repeat_interleave(group_size, dim=1)
  v_ref = v.repeat_interleave(group_size, dim=1)
  kv_offset = Nkv - Nq
  row_idx = torch.arange(Nq, device="cuda").view(-1, 1)
  col_idx = torch.arange(Nkv, device="cuda").view(1, -1)
  attn_mask = (col_idx <= (row_idx + kv_offset))
  ref = F.scaled_dot_product_attention(q, k_ref, v_ref, attn_mask=attn_mask, scale=1.0 / math.sqrt(D))
  assert out.shape == (B, Nh_q, Nq, D)
  assert out.dtype == dtype
  assert torch.isfinite(out).all()
  torch.testing.assert_close(out, ref, **_tolerance(dtype))


def test_ffpa_attn_func_rejects_causal_with_shorter_kv():
  torch.manual_seed(0)
  # Nq > Nkv + causal is rejected (no valid keys for later Q rows).
  q = torch.randn(1, 4, 256, 128, dtype=torch.float16, device="cuda")
  k = torch.randn(1, 4, 128, 128, dtype=torch.float16, device="cuda")
  v = torch.randn(1, 4, 128, 128, dtype=torch.float16, device="cuda")
  with pytest.raises(ValueError, match="causal"):
    ffpa_attn_func(q, k, v, stages=2, acc="f32", causal=True)

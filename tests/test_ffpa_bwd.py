"""FFPA attention backward pass unit tests.

Tests ``ffpa_attn_func`` backward correctness by comparing dQ/dK/dV
against :func:`torch.nn.functional.scaled_dot_product_attention` across:

* Multiple head dimensions (64, 320, 512 — the subset built for fast iteration)
* Causal and non-causal paths
* GQA / MQA (Nh_q > Nh_kv)
* Cross-attention (Nq != Nkv)
* Inference-only path (torch.no_grad)
"""

import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ffpa_attn import ffpa_attn_func  # noqa: E402

# Build subset for fast iteration: 64 (small-d), 320, 512 (large-d).
HEADDIMS = [64, 320, 512]
DTYPES = [torch.float16, torch.bfloat16]


def _tolerance(dtype):
  return {"atol": 5e-2, "rtol": 5e-2} if dtype == torch.bfloat16 else {"atol": 1e-2, "rtol": 1e-2}


def _make_sdpa_kwargs(causal, Nq, Nkv):
  """Build SDPA keyword args for causal / cross-attention."""
  if causal and Nq != Nkv:
    kv_offset = Nkv - Nq
    row_idx = torch.arange(Nq, device="cuda").view(-1, 1)
    col_idx = torch.arange(Nkv, device="cuda").view(1, -1)
    return {"attn_mask": col_idx <= (row_idx + kv_offset)}
  elif causal:
    return {"is_causal": True}
  return {}


def _sdpa_ref(q, k, v, causal, scale):
  """Run SDPA forward only (no grad) for output comparison."""
  group_size = q.size(1) // k.size(1)
  k2 = k.repeat_interleave(group_size, dim=1) if group_size > 1 else k
  v2 = v.repeat_interleave(group_size, dim=1) if group_size > 1 else v
  kw = _make_sdpa_kwargs(causal, q.size(2), k.size(2))
  return F.scaled_dot_product_attention(q, k2, v2, scale=scale, **kw)


def _sdpa_ref_grads(q, k, v, causal, scale):
  """Run SDPA forward + backward and return (dq, dk, dv) from autograd.

    SDPA supports MQA (Nh_kv == 1) natively, but for general GQA
    (Nh_kv > 1, Nh_q > Nh_kv) we repeat K/V to match Q, then sum-reduce
    the K/V gradients back to the original head count.
    """
  q2 = q.detach().clone().requires_grad_(True)
  k2 = k.detach().clone().requires_grad_(True)
  v2 = v.detach().clone().requires_grad_(True)

  group_size = q.size(1) // k.size(1)
  k_in = k2.repeat_interleave(group_size, dim=1) if group_size > 1 else k2
  v_in = v2.repeat_interleave(group_size, dim=1) if group_size > 1 else v2

  kw = _make_sdpa_kwargs(causal, q.size(2), k.size(2))
  out_ref = F.scaled_dot_product_attention(q2, k_in, v_in, scale=scale, **kw)
  loss_ref = out_ref.sum()
  loss_ref.backward()

  if group_size > 1:
    # k2.grad accumulates from repeat_interleave; shape [B, Nh_kv, N, D].
    dk = k2.grad
    dv = v2.grad
  else:
    dk = k2.grad
    dv = v2.grad

  return q2.grad, dk, dv


# ---------------------------------------------------------------------------
# Basic backward correctness
# ---------------------------------------------------------------------------

BASIC_BWD_SHAPES = [
  (1, 8, 4096, 64),
  (1, 8, 4096, 320),
  (1, 8, 4096, 512),
  (1, 16, 8192, 64),
  (1, 16, 8192, 320),
  (1, 16, 16384, 512),
]


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("B,H,N,D", BASIC_BWD_SHAPES)
def test_ffpa_bwd_basic(dtype, B, H, N, D):
  """dQ/dK/dV from FFPA backward must match SDPA reference."""
  torch.manual_seed(0)
  q = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
  k = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
  v = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)

  scale = 1.0 / math.sqrt(D)
  out = ffpa_attn_func(
    q,
    k,
    v,
    causal=False,
    softmax_scale=scale,
    stages=2,
    acc="f32",
    high_precision_grad=True,
  )
  loss = out.sum()
  loss.backward()

  dq_ref, dk_ref, dv_ref = _sdpa_ref_grads(q, k, v, False, scale)

  tol = _tolerance(dtype)
  torch.testing.assert_close(q.grad, dq_ref, **tol)
  torch.testing.assert_close(k.grad, dk_ref, **tol)
  torch.testing.assert_close(v.grad, dv_ref, **tol)


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("causal", [False, True], ids=["noncausal", "causal"])
def test_ffpa_bwd_split_d_native_hdim512(dtype, causal):
  """Native split-D backward for D=512 must match SDPA on a focused smoke shape."""
  B, H, N, D = 1, 2, 128, 512
  torch.manual_seed(0)
  q = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
  k = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
  v = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)

  scale = 1.0 / math.sqrt(D)
  out = ffpa_attn_func(
    q,
    k,
    v,
    causal=causal,
    softmax_scale=scale,
    stages=1,
    acc="f32",
    high_precision_grad=True,
    backward_backend="split_d",
  )
  out.sum().backward()

  dq_ref, dk_ref, dv_ref = _sdpa_ref_grads(q, k, v, causal, scale)

  tol = _tolerance(dtype)
  torch.testing.assert_close(q.grad, dq_ref, **tol)
  torch.testing.assert_close(k.grad, dk_ref, **tol)
  torch.testing.assert_close(v.grad, dv_ref, **tol)


# ---------------------------------------------------------------------------
# Backward + causal
# ---------------------------------------------------------------------------

CAUSAL_BWD_SHAPES = [
  (4096, 64),
  (4096, 320),
  (8192, 512),
  (16384, 64),
  (16384, 320),
]


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("N,D", CAUSAL_BWD_SHAPES)
def test_ffpa_bwd_causal(dtype, N, D):
  """Causal backward gradients must match SDPA reference."""
  B, H = 1, 8
  torch.manual_seed(0)
  q = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
  k = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
  v = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)

  scale = 1.0 / math.sqrt(D)
  out = ffpa_attn_func(
    q,
    k,
    v,
    causal=True,
    softmax_scale=scale,
    stages=2,
    acc="f32",
    high_precision_grad=True,
  )
  loss = out.sum()
  loss.backward()

  dq_ref, dk_ref, dv_ref = _sdpa_ref_grads(q, k, v, True, scale)

  tol = _tolerance(dtype)
  torch.testing.assert_close(q.grad, dq_ref, **tol)
  torch.testing.assert_close(k.grad, dk_ref, **tol)
  torch.testing.assert_close(v.grad, dv_ref, **tol)


# ---------------------------------------------------------------------------
# Backward + GQA
# ---------------------------------------------------------------------------

GQA_BWD_CONFIGS = [
  (16, 2, 4096, 64),  # 8x GQA
  (32, 4, 8192, 320),  # 8x GQA, large-d
  (32, 8, 16384, 512),  # 4x GQA, large-d
  (8, 1, 4096, 64),  # MQA
  (8, 1, 8192, 320),  # MQA, large-d
]


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("Nh_q,Nh_kv,N,D", GQA_BWD_CONFIGS)
def test_ffpa_bwd_gqa(dtype, Nh_q, Nh_kv, N, D):
  """GQA backward gradients must match SDPA reference."""
  B = 1
  torch.manual_seed(0)
  q = torch.randn(B, Nh_q, N, D, dtype=dtype, device="cuda", requires_grad=True)
  k = torch.randn(B, Nh_kv, N, D, dtype=dtype, device="cuda", requires_grad=True)
  v = torch.randn(B, Nh_kv, N, D, dtype=dtype, device="cuda", requires_grad=True)

  scale = 1.0 / math.sqrt(D)
  out = ffpa_attn_func(
    q,
    k,
    v,
    causal=False,
    softmax_scale=scale,
    stages=2,
    acc="f32",
    high_precision_grad=True,
  )
  loss = out.sum()
  loss.backward()

  dq_ref, dk_ref, dv_ref = _sdpa_ref_grads(q, k, v, False, scale)

  tol = _tolerance(dtype)
  torch.testing.assert_close(q.grad, dq_ref, **tol)
  torch.testing.assert_close(k.grad, dk_ref, **tol)
  torch.testing.assert_close(v.grad, dv_ref, **tol)


# ---------------------------------------------------------------------------
# Backward + causal + GQA
# ---------------------------------------------------------------------------

CAUSAL_GQA_BWD_CONFIGS = [
  (16, 2, 4096, 64),
  (32, 4, 8192, 320),
  (8, 1, 16384, 512),  # MQA + causal + large-d
  (32, 8, 4096, 64),
]


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("Nh_q,Nh_kv,N,D", CAUSAL_GQA_BWD_CONFIGS)
def test_ffpa_bwd_causal_gqa(dtype, Nh_q, Nh_kv, N, D):
  """Causal + GQA backward gradients must match SDPA reference."""
  B = 1
  torch.manual_seed(0)
  q = torch.randn(B, Nh_q, N, D, dtype=dtype, device="cuda", requires_grad=True)
  k = torch.randn(B, Nh_kv, N, D, dtype=dtype, device="cuda", requires_grad=True)
  v = torch.randn(B, Nh_kv, N, D, dtype=dtype, device="cuda", requires_grad=True)

  scale = 1.0 / math.sqrt(D)
  out = ffpa_attn_func(
    q,
    k,
    v,
    causal=True,
    softmax_scale=scale,
    stages=2,
    acc="f32",
    high_precision_grad=True,
  )
  loss = out.sum()
  loss.backward()

  dq_ref, dk_ref, dv_ref = _sdpa_ref_grads(q, k, v, True, scale)

  tol = _tolerance(dtype)
  torch.testing.assert_close(q.grad, dq_ref, **tol)
  torch.testing.assert_close(k.grad, dk_ref, **tol)
  torch.testing.assert_close(v.grad, dv_ref, **tol)


# ---------------------------------------------------------------------------
# Backward + cross-attention
# ---------------------------------------------------------------------------

CROSS_BWD_SHAPES = [
  (256, 4096, 64),
  (256, 8192, 320),
  (256, 16384, 512),
  (1024, 8192, 320),
]


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("Nq,Nkv,D", CROSS_BWD_SHAPES)
def test_ffpa_bwd_cross_attention(dtype, Nq, Nkv, D):
  """Cross-attention backward gradients must match SDPA reference."""
  B, H = 1, 8
  torch.manual_seed(0)
  q = torch.randn(B, H, Nq, D, dtype=dtype, device="cuda", requires_grad=True)
  k = torch.randn(B, H, Nkv, D, dtype=dtype, device="cuda", requires_grad=True)
  v = torch.randn(B, H, Nkv, D, dtype=dtype, device="cuda", requires_grad=True)

  scale = 1.0 / math.sqrt(D)
  out = ffpa_attn_func(
    q,
    k,
    v,
    causal=False,
    softmax_scale=scale,
    stages=2,
    acc="f32",
    high_precision_grad=True,
  )
  loss = out.sum()
  loss.backward()

  dq_ref, dk_ref, dv_ref = _sdpa_ref_grads(q, k, v, False, scale)

  tol = _tolerance(dtype)
  torch.testing.assert_close(q.grad, dq_ref, **tol)
  torch.testing.assert_close(k.grad, dk_ref, **tol)
  torch.testing.assert_close(v.grad, dv_ref, **tol)


# ---------------------------------------------------------------------------
# Inference path (no_grad)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("D", HEADDIMS)
def test_ffpa_bwd_nograd_forward_ok(dtype, D):
  """Under torch.no_grad, forward still works and produces correct output."""
  B, H, N = 1, 8, 512
  torch.manual_seed(0)
  q = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
  k = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
  v = torch.randn(B, H, N, D, dtype=dtype, device="cuda")

  with torch.no_grad():
    out = ffpa_attn_func(
      q,
      k,
      v,
      stages=2,
      acc="f32",
      high_precision_grad=True,
    )

  scale = 1.0 / math.sqrt(D)
  ref = _sdpa_ref(q, k, v, False, scale)
  tol = _tolerance(dtype)
  torch.testing.assert_close(out, ref, **tol)

"""CuTeDSL backend integration tests for ffpa_attn_func.

The CuTeDSL kernels are specialised for SM90 (Hopper) and head_dim == 512.
These tests gate on the device capability and only validate the dispatch
wiring + numerical parity against the existing Triton backend, not the
kernel internals (covered by the standalone CuTeDSL kernel tests).

Varlen coverage now lives in ``tests/test_ffpa_varlen.py``.
"""

import logging

import pytest
import torch
import torch.nn.functional as F

from ffpa_attn import ffpa_attn_func
import ffpa_attn.ffpa_attn_interface as iface


def _sm90_available() -> bool:
  if not torch.cuda.is_available():
    return False
  major, _ = torch.cuda.get_device_capability()
  return major == 9


pytestmark = pytest.mark.skipif(not _sm90_available(), reason="cutedsl backend requires SM90 (Hopper)")


def _tol():
  return {"atol": 2e-2, "rtol": 2e-2}


def _grad_tol():
  return {"atol": 5e-2, "rtol": 5e-2}


@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("Hq,Hkv", [(8, 8), (16, 4)])
def test_cutedsl_forward_matches_triton(is_causal, Hq, Hkv):
  torch.manual_seed(0)
  B, Nq, Nkv, D = 1, 1024, 1024, 512
  q = torch.randn(B, Hq, Nq, D, dtype=torch.bfloat16, device="cuda")
  k = torch.randn(B, Hkv, Nkv, D, dtype=torch.bfloat16, device="cuda")
  v = torch.randn(B, Hkv, Nkv, D, dtype=torch.bfloat16, device="cuda")
  enable_gqa = Hq != Hkv

  out_tri = ffpa_attn_func(q, k, v, is_causal=is_causal, enable_gqa=enable_gqa, forward_backend="triton")
  out_cute = ffpa_attn_func(q, k, v, is_causal=is_causal, enable_gqa=enable_gqa, forward_backend="cutedsl")
  assert out_cute.shape == out_tri.shape
  torch.testing.assert_close(out_cute, out_tri, **_tol())


@pytest.mark.parametrize("is_causal", [False, True])
def test_cutedsl_autograd_matches_triton(is_causal):
  torch.manual_seed(42)
  B, H, N, D = 1, 4, 1024, 512

  def make():
    return (
      torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda", requires_grad=True),
      torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda", requires_grad=True),
      torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda", requires_grad=True),
    )

  torch.manual_seed(42)
  q_t, k_t, v_t = make()
  out_t = ffpa_attn_func(q_t, k_t, v_t, is_causal=is_causal, forward_backend="triton")
  out_t.sum().backward()

  torch.manual_seed(42)
  q_c, k_c, v_c = make()
  out_c = ffpa_attn_func(q_c, k_c, v_c, is_causal=is_causal, forward_backend="cutedsl")
  out_c.sum().backward()

  torch.testing.assert_close(out_c, out_t, **_tol())
  torch.testing.assert_close(q_c.grad, q_t.grad, **_grad_tol())
  torch.testing.assert_close(k_c.grad, k_t.grad, **_grad_tol())
  torch.testing.assert_close(v_c.grad, v_t.grad, **_grad_tol())


@pytest.mark.parametrize("D", [320, 128])
def test_cutedsl_falls_back_for_non_512_head_dim(caplog, monkeypatch, D):
  """D != 512 with cutedsl backend falls back to SDPA with a one-shot warning."""
  monkeypatch.setattr(iface.logger, "propagate", True)
  torch.manual_seed(0)
  q = torch.randn(1, 8, 1024, D, dtype=torch.bfloat16, device="cuda")

  with caplog.at_level(logging.WARNING, logger=iface.logger.name):
    out = ffpa_attn_func(q, q, q, forward_backend="cutedsl")

  ref = F.scaled_dot_product_attention(q, q, q)
  torch.testing.assert_close(out, ref, **_tol())
  assert any(
    "falling back to SDPA" in r.getMessage() and f"head_dim={D}" in r.getMessage() for r in caplog.records
  ), f"expected fallback warning for head_dim={D}, got: {[r.getMessage() for r in caplog.records]}"


def test_cutedsl_rejects_fp16_training():
  q = torch.randn(1, 8, 1024, 512, dtype=torch.float16, device="cuda", requires_grad=True)
  with pytest.raises(NotImplementedError, match="bfloat16"):
    ffpa_attn_func(q, q, q, forward_backend="cutedsl")


def test_cutedsl_raises_on_attn_mask():
  """attn_mask != None with cutedsl: raise NotImplementedError (uniform policy)."""
  q = torch.randn(1, 8, 1024, 512, dtype=torch.bfloat16, device="cuda")
  m = torch.zeros(1024, 1024, dtype=torch.bfloat16, device="cuda")
  with pytest.raises(NotImplementedError, match="attention_mask"):
    ffpa_attn_func(q, q, q, attn_mask=m, forward_backend="cutedsl")


def test_cutedsl_raises_on_dropout():
  """dropout_p > 0 with cutedsl: raise NotImplementedError (uniform policy)."""
  q = torch.randn(1, 8, 1024, 512, dtype=torch.bfloat16, device="cuda")
  with pytest.raises(NotImplementedError, match="dropout_p"):
    ffpa_attn_func(q, q, q, dropout_p=0.1, forward_backend="cutedsl")


def test_cutedsl_rejects_mixed_backward_backend():
  q = torch.randn(1, 8, 1024, 512, dtype=torch.bfloat16, device="cuda")
  with pytest.raises(ValueError, match="forward_backend='cutedsl' requires backward_backend='cutedsl'"):
    ffpa_attn_func(q, q, q, forward_backend="cutedsl", backward_backend="triton")


def test_cutedsl_fast_path_bypasses_ffpaattnfunc(monkeypatch):
  """forward_backend='cutedsl' + D=512 + SM90 must short-circuit FFPAAttnFunc.

  Patches the dispatcher entry point in ffpa_attn_interface and asserts it is
  never invoked: the fast-path in ffpa_attn_func should route directly to
  cutedsl._interface.ffpa_attn_splitd_func.
  """
  import ffpa_attn.ffpa_attn_interface as iface

  call_count = [0]
  orig = iface.FFPAAttnFunc.apply

  def spy(*args, **kwargs):
    call_count[0] += 1
    return orig(*args, **kwargs)

  monkeypatch.setattr(iface.FFPAAttnFunc, "apply", spy)

  q = torch.randn(1, 8, 1024, 512, dtype=torch.bfloat16, device="cuda")
  out = ffpa_attn_func(q, q, q, forward_backend="cutedsl")
  assert call_count[0] == 0, "fast-path should have bypassed FFPAAttnFunc.apply"
  assert out.shape == q.shape and torch.isfinite(out).all()

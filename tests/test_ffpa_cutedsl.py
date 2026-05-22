"""CuTeDSL backend integration tests for ffpa_attn_func.

The CuTeDSL kernels are specialised for SM90 (Hopper) and dense 256<D<=512.
These tests gate on the device capability and only validate the dispatch
wiring + numerical parity against the existing Triton backend, not the
kernel internals (covered by the standalone CuTeDSL kernel tests).

Varlen coverage now lives in ``tests/test_ffpa_varlen.py``.
"""

import pytest
import torch
import torch.nn.functional as F

from ffpa_attn import ffpa_attn_func


def _sm90_available() -> bool:
  if not torch.cuda.is_available():
    return False
  major, _ = torch.cuda.get_device_capability()
  return major == 9


pytestmark = pytest.mark.skipif(
  not _sm90_available(), reason="cutedsl backend requires SM90 (Hopper)"
)


def _tol():
  return {"atol": 2e-2, "rtol": 2e-2}


def _grad_tol():
  return {"atol": 5e-2, "rtol": 5e-2}


@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("Hq,Hkv", [(8, 8), (16, 4)])
@pytest.mark.parametrize("D", [320, 384, 512])
def test_cutedsl_forward_matches_triton(is_causal, Hq, Hkv, D):
  torch.manual_seed(0)
  B, Nq, Nkv = 1, 1024, 1024
  q = torch.randn(B, Hq, Nq, D, dtype=torch.bfloat16, device="cuda")
  k = torch.randn(B, Hkv, Nkv, D, dtype=torch.bfloat16, device="cuda")
  v = torch.randn(B, Hkv, Nkv, D, dtype=torch.bfloat16, device="cuda")
  enable_gqa = Hq != Hkv

  out_tri = ffpa_attn_func(
    q,
    k,
    v,
    is_causal=is_causal,
    enable_gqa=enable_gqa,
    forward_backend="triton"
  )
  out_cute = ffpa_attn_func(
    q,
    k,
    v,
    is_causal=is_causal,
    enable_gqa=enable_gqa,
    forward_backend="cutedsl"
  )
  assert out_cute.shape == out_tri.shape
  torch.testing.assert_close(out_cute, out_tri, **_tol())


@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("D", [320, 384, 512])
@pytest.mark.parametrize(
  "dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"]
)
def test_cutedsl_autograd_matches_triton(is_causal, D, dtype):
  torch.manual_seed(42)
  B, H, N = 1, 4, 1024

  def make():
    return (
      torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True),
      torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True),
      torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True),
    )

  torch.manual_seed(42)
  q_t, k_t, v_t = make()
  out_t = ffpa_attn_func(
    q_t, k_t, v_t, is_causal=is_causal, forward_backend="triton"
  )
  out_t.sum().backward()

  torch.manual_seed(42)
  q_c, k_c, v_c = make()
  out_c = ffpa_attn_func(
    q_c, k_c, v_c, is_causal=is_causal, forward_backend="cutedsl"
  )
  out_c.sum().backward()

  torch.testing.assert_close(out_c, out_t, **_tol())
  torch.testing.assert_close(q_c.grad, q_t.grad, **_grad_tol())
  torch.testing.assert_close(k_c.grad, k_t.grad, **_grad_tol())
  torch.testing.assert_close(v_c.grad, v_t.grad, **_grad_tol())


@pytest.mark.parametrize("D", [128])
def test_cutedsl_falls_back_for_small_head_dim(D):
  """D <= 256 with cutedsl backend falls back to SDPA."""
  torch.manual_seed(0)
  q = torch.randn(1, 8, 1024, D, dtype=torch.bfloat16, device="cuda")

  out = ffpa_attn_func(q, q, q, forward_backend="cutedsl")

  ref = F.scaled_dot_product_attention(q, q, q)
  torch.testing.assert_close(out, ref, **_tol())


def test_cutedsl_accepts_fp16_training():
  q = torch.randn(
    1, 8, 256, 384, dtype=torch.float16, device="cuda", requires_grad=True
  )
  out = ffpa_attn_func(q, q, q, forward_backend="cutedsl")
  out.sum().backward()
  assert out.dtype == torch.float16
  assert q.grad is not None and torch.isfinite(q.grad).all()


def test_cutedsl_raises_on_attn_mask():
  """attn_mask != None with cutedsl: raise NotImplementedError (uniform policy)."""
  q = torch.randn(1, 8, 1024, 512, dtype=torch.bfloat16, device="cuda")
  m = torch.zeros(1024, 1024, dtype=torch.bfloat16, device="cuda")
  with pytest.raises(NotImplementedError, match="attn_mask|attention_mask"):
    ffpa_attn_func(q, q, q, attn_mask=m, forward_backend="cutedsl")


def test_cutedsl_raises_on_dropout():
  """dropout_p > 0 with cutedsl: raise NotImplementedError (uniform policy)."""
  q = torch.randn(1, 8, 1024, 512, dtype=torch.bfloat16, device="cuda")
  with pytest.raises(NotImplementedError, match="dropout"):
    ffpa_attn_func(q, q, q, dropout_p=0.1, forward_backend="cutedsl")


def test_cutedsl_rejects_mixed_backward_backend():
  q = torch.randn(1, 8, 1024, 512, dtype=torch.bfloat16, device="cuda")
  with pytest.raises(
    ValueError,
    match="forward_backend='cutedsl' requires backward_backend='cutedsl'"
  ):
    ffpa_attn_func(
      q, q, q, forward_backend="cutedsl", backward_backend="triton"
    )


@pytest.mark.parametrize("D", [320, 384, 512])
def test_cutedsl_routes_through_ffpaattnfunc(monkeypatch, D):
  """forward_backend='cutedsl' + dense large D + SM90 must go through FFPAAttnFunc.

  After the dispatch refactor, cutedsl follows the same code path as other
  backends: ffpa_attn_func → FFPAAttnMeta.normalize_inputs →
  FFPAAttnFunc.apply → _FFPAAttnFunc.forward (CuTeDSLBackend branch).
  """
  import ffpa_attn.ffpa_attn_interface as iface

  call_count = [0]
  orig = iface.FFPAAttnFunc.apply

  def spy(*args, **kwargs):
    call_count[0] += 1
    return orig(*args, **kwargs)

  monkeypatch.setattr(iface.FFPAAttnFunc, "apply", spy)

  q = torch.randn(1, 8, 1024, D, dtype=torch.bfloat16, device="cuda")
  out = ffpa_attn_func(q, q, q, forward_backend="cutedsl")
  assert call_count[0] == 1, "cutedsl should route through FFPAAttnFunc.apply"
  assert out.shape == q.shape and torch.isfinite(out).all()


# Each (Nq, Nkv, is_causal) row exercises a distinct boundary path in the dense
# cutedsl forward kernel. Oracle is torch SDPA — the documented semantic target
# of ffpa_attn_func — which also keeps the forward and autograd tests in this
# file using the same reference. SDPA's is_causal=True matches FFPA's
# tail-aligned causal only when Nq == Nkv, so cross-attn cases stay non-causal.
NONALIGNED_FORWARD_CASES = [
  # Nq, Nkv, is_causal  — what each row exercises
  (8191, 8192, False
   ),  # Q tail only: ceil_div m_block + LSE row bound + O TMA OOB drop
  (8192, 8191, False),  # KV tail only: R2P column mask on boundary n_block
  (129, 129, False),  # small both-sided tail, non-causal
  (129, 129,
   True),  # small both-sided tail + causal mask x boundary interaction
  (8191, 8191, False),  # large both-sided tail, non-causal
  (8191, 8191, True),  # large both-sided tail + causal
]


@pytest.mark.parametrize("Nq,Nkv,is_causal", NONALIGNED_FORWARD_CASES)
def test_cutedsl_forward_nonaligned_matches_sdpa(Nq, Nkv, is_causal):
  """Forward parity of the cutedsl backend on non-aligned seqlen (tile_m=64, tile_n=128).

  Verifies the four implicit boundary contracts of the dense cutedsl forward:
    1. ceil_div m_block scheduling visits the partial Q tail tile
       (_ffpa_fwd_d512_sm90.py:450)
    2. mask_seqlen=True on the boundary n_block applies an R2P column bitmask
       (utils/mask.py:155-167, _ffpa_fwd_d512_sm90.py:1088)
    3. LSE explicit per-row boundary check on the partial Q tail tile
       (_ffpa_fwd_d512_sm90.py:1654-1657)
    4. O TMA descriptor extent + Hopper OOB store-drop on the partial Q tail
       (_ffpa_fwd_d512_sm90.py:430-441, 1676)
  """
  torch.manual_seed(0)
  B, H, D = 1, 4, 512
  q = torch.randn(B, H, Nq, D, dtype=torch.bfloat16, device="cuda")
  k = torch.randn(B, H, Nkv, D, dtype=torch.bfloat16, device="cuda")
  v = torch.randn(B, H, Nkv, D, dtype=torch.bfloat16, device="cuda")

  out_ref = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
  out_cute = ffpa_attn_func(
    q, k, v, is_causal=is_causal, forward_backend="cutedsl"
  )

  assert out_cute.shape == out_ref.shape == (B, H, Nq, D)
  assert torch.isfinite(out_cute).all(), (
    f"cutedsl output has NaN/Inf at Nq={Nq}, Nkv={Nkv}, is_causal={is_causal}"
  )
  torch.testing.assert_close(out_cute, out_ref, **_tol())


NONALIGNED_AUTOGRAD_SEQLENS = [65, 129, 1023, 8191]


@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("N", NONALIGNED_AUTOGRAD_SEQLENS)
def test_cutedsl_autograd_nonaligned_matches_sdpa(is_causal, N):
  """Autograd parity for cutedsl backend with non-aligned seqlen (Nq == Nkv == N).

  Compared against ``torch.nn.functional.scaled_dot_product_attention`` rather
  than the Triton FFPA backend: at the time of writing, Triton's bwd produces
  NaN gradients for non-aligned causal self-attention at D=512, so using it as
  the oracle masks cutedsl's actual (correct) behavior. SDPA is the documented
  semantic target of ``ffpa_attn_func`` anyway.
  """
  B, H, D = 1, 4, 512

  def make():
    return (
      torch.randn(
        B, H, N, D, dtype=torch.bfloat16, device="cuda", requires_grad=True
      ),
      torch.randn(
        B, H, N, D, dtype=torch.bfloat16, device="cuda", requires_grad=True
      ),
      torch.randn(
        B, H, N, D, dtype=torch.bfloat16, device="cuda", requires_grad=True
      ),
    )

  torch.manual_seed(42)
  q_r, k_r, v_r = make()
  out_r = F.scaled_dot_product_attention(q_r, k_r, v_r, is_causal=is_causal)
  out_r.sum().backward()

  torch.manual_seed(42)
  q_c, k_c, v_c = make()
  out_c = ffpa_attn_func(
    q_c, k_c, v_c, is_causal=is_causal, forward_backend="cutedsl"
  )
  out_c.sum().backward()

  assert torch.isfinite(out_c).all(
  ), f"cutedsl output has NaN/Inf at N={N}, is_causal={is_causal}"
  for name, t in (("q", q_c), ("k", k_c), ("v", v_c)):
    assert torch.isfinite(
      t.grad
    ).all(), f"cutedsl {name}.grad has NaN/Inf at N={N}, is_causal={is_causal}"
  torch.testing.assert_close(out_c, out_r, **_tol())
  torch.testing.assert_close(q_c.grad, q_r.grad, **_grad_tol())
  torch.testing.assert_close(k_c.grad, k_r.grad, **_grad_tol())
  torch.testing.assert_close(v_c.grad, v_r.grad, **_grad_tol())

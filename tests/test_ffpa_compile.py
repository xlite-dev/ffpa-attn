"""Tests for torch.compile compatibility of FFPA attention ops.

Verifies that ``torch.compile(ffpa_attn_func, fullgraph=False)`` produces
correct outputs and gradients across CUDA and Triton backends.
"""

import pytest
import torch

from ffpa_attn import ffpa_attn_func
import ffpa_attn.functional as ffpa_attn_functional

# Parametrized tests produce many shape/dtype/backend variants; allow
# enough recompilations to avoid hitting the default limit of 8.
torch._dynamo.config.recompile_limit = 64


# Fixtures & helpers
@pytest.fixture(scope="module", autouse=True)
def _require_cuda():
  if not torch.cuda.is_available():
    pytest.skip("CUDA not available")


def _tolerance(dtype):
  if dtype == torch.bfloat16:
    return {"atol": 2e-2, "rtol": 2e-2}
  return {"atol": 1e-2, "rtol": 1e-2}


# Representative shapes (kept small to keep test time reasonable)
FWD_SHAPES = [
  (1, 8, 1024, 320),
  (1, 8, 512, 512),
]

BWD_SHAPES = [
  (1, 8, 512, 320),
  (1, 8, 512, 512),
]

DTYPES = [torch.float16, torch.bfloat16]


def _require_cuda_forward_impl() -> None:
  if not ffpa_attn_functional.cuda_forward_available():
    pytest.skip("CUDA forward backend was not compiled")


def _require_cuda_backward_impl() -> None:
  if not ffpa_attn_functional.cuda_backward_available():
    pytest.skip("CUDA backward backend was not compiled")


# Forward-only compile tests
@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("B,H,N,D", FWD_SHAPES)
def test_compile_forward_cuda(dtype, B, H, N, D):
  """torch.compile forward with CUDA backend matches eager reference."""
  _require_cuda_forward_impl()
  torch.manual_seed(0)
  device = "cuda"
  q = torch.randn(B, H, N, D, dtype=dtype, device=device)
  k = torch.randn(B, H, N, D, dtype=dtype, device=device)
  v = torch.randn(B, H, N, D, dtype=dtype, device=device)

  eager = ffpa_attn_func(q, k, v, forward_backend="cuda")

  compiled_fn = torch.compile(ffpa_attn_func, fullgraph=False)
  out = compiled_fn(q, k, v, forward_backend="cuda")

  torch.testing.assert_close(out, eager, **_tolerance(dtype))
  assert torch.isfinite(out).all()


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("B,H,N,D", FWD_SHAPES)
def test_compile_forward_triton(dtype, B, H, N, D):
  """torch.compile forward with Triton backend matches eager reference."""
  torch.manual_seed(0)
  device = "cuda"
  q = torch.randn(B, H, N, D, dtype=dtype, device=device)
  k = torch.randn(B, H, N, D, dtype=dtype, device=device)
  v = torch.randn(B, H, N, D, dtype=dtype, device=device)

  eager = ffpa_attn_func(q, k, v, forward_backend="triton")

  compiled_fn = torch.compile(ffpa_attn_func, fullgraph=False)
  out = compiled_fn(q, k, v, forward_backend="triton")

  torch.testing.assert_close(out, eager, **_tolerance(dtype))
  assert torch.isfinite(out).all()


# Forward + backward compile tests.
# Use fullgraph=False because _ffpa_apply is guarded with
# torch._dynamo.disable to prevent Dynamo from inlining the autograd
# Function and replacing the real backward with an auto-generated template.
# The forward path (including all torch.ops.ffpa_attn.* calls) is still
# fully captured up to the graph break at the Function boundary.

_BACKEND_PAIRS = [
  # (forward_backend, backward_backend)
  ("cuda", "triton"),
  ("triton", "triton"),
  ("cuda", "sdpa"),
]


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("B,H,N,D", BWD_SHAPES)
@pytest.mark.parametrize("fw,bw", _BACKEND_PAIRS, ids=lambda p: f"{p[0]}-{p[1]}")
def test_compile_backward(dtype, B, H, N, D, fw, bw):
  """torch.compile forward+backward matches eager grads across backend pairs."""
  if fw == "cuda":
    _require_cuda_forward_impl()
  if bw == "cuda":
    _require_cuda_backward_impl()
  torch.manual_seed(0)
  device = "cuda"
  q = torch.randn(B, H, N, D, dtype=dtype, device=device, requires_grad=True)
  k = torch.randn(B, H, N, D, dtype=dtype, device=device, requires_grad=True)
  v = torch.randn(B, H, N, D, dtype=dtype, device=device, requires_grad=True)

  # Eager reference
  qr = q.detach().requires_grad_(True)
  kr = k.detach().requires_grad_(True)
  vr = v.detach().requires_grad_(True)
  out_ref = ffpa_attn_func(qr, kr, vr, forward_backend=fw, backward_backend=bw)
  out_ref.sum().backward()
  grads_ref = [qr.grad.clone(), kr.grad.clone(), vr.grad.clone()]

  # Compiled
  q.grad = k.grad = v.grad = None
  compiled_fn = torch.compile(ffpa_attn_func, fullgraph=False)
  out = compiled_fn(q, k, v, forward_backend=fw, backward_backend=bw)
  out.sum().backward()

  torch.testing.assert_close(out, out_ref, **_tolerance(dtype))
  torch.testing.assert_close(q.grad, grads_ref[0], **_tolerance(dtype))
  torch.testing.assert_close(k.grad, grads_ref[1], **_tolerance(dtype))
  torch.testing.assert_close(v.grad, grads_ref[2], **_tolerance(dtype))


# Compile modes
@pytest.mark.parametrize("mode", ["default", "reduce-overhead"])
@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
def test_compile_modes_forward(mode, dtype):
  """torch.compile forward passes with different compile modes."""
  _require_cuda_forward_impl()
  torch.manual_seed(0)
  device = "cuda"
  B, H, N, D = 1, 8, 512, 320
  q = torch.randn(B, H, N, D, dtype=dtype, device=device)
  k = torch.randn(B, H, N, D, dtype=dtype, device=device)
  v = torch.randn(B, H, N, D, dtype=dtype, device=device)

  eager = ffpa_attn_func(q, k, v, forward_backend="cuda")
  compiled_fn = torch.compile(ffpa_attn_func, mode=mode, fullgraph=False)
  out = compiled_fn(q, k, v, forward_backend="cuda")

  torch.testing.assert_close(out, eager, **_tolerance(dtype))


# GQA compile test
@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
def test_compile_gqa(dtype):
  """torch.compile with GQA shapes matches eager across backend pairs."""
  _require_cuda_forward_impl()
  B, Nh_q, Nh_kv, N, D = 1, 16, 4, 512, 320
  torch.manual_seed(0)
  device = "cuda"
  q = torch.randn(B, Nh_q, N, D, dtype=dtype, device=device, requires_grad=True)
  k = torch.randn(B, Nh_kv, N, D, dtype=dtype, device=device, requires_grad=True)
  v = torch.randn(B, Nh_kv, N, D, dtype=dtype, device=device, requires_grad=True)

  # Eager (CUDA fwd + Triton bwd)
  qr = q.detach().requires_grad_(True)
  kr = k.detach().requires_grad_(True)
  vr = v.detach().requires_grad_(True)
  out_ref = ffpa_attn_func(
    qr,
    kr,
    vr,
    enable_gqa=True,
    forward_backend="cuda",
    backward_backend="triton",
  )
  out_ref.sum().backward()
  grads_ref = [qr.grad.clone(), kr.grad.clone(), vr.grad.clone()]

  # Compiled
  q.grad = k.grad = v.grad = None
  compiled_fn = torch.compile(ffpa_attn_func, fullgraph=False)
  out = compiled_fn(
    q,
    k,
    v,
    enable_gqa=True,
    forward_backend="cuda",
    backward_backend="triton",
  )
  out.sum().backward()

  torch.testing.assert_close(out, out_ref, **_tolerance(dtype))
  torch.testing.assert_close(q.grad, grads_ref[0], atol=5e-2, rtol=5e-2)
  torch.testing.assert_close(k.grad, grads_ref[1], atol=5e-2, rtol=5e-2)
  torch.testing.assert_close(v.grad, grads_ref[2], atol=5e-2, rtol=5e-2)


# Causal compile test
@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("fw,bw", [("cuda", "triton"), ("triton", "triton")], ids=["cuda-triton", "triton-triton"])
def test_compile_causal(dtype, fw, bw):
  """torch.compile with causal masking matches eager across backend pairs."""
  if fw == "cuda":
    _require_cuda_forward_impl()
  if bw == "cuda":
    _require_cuda_backward_impl()
  B, H, N, D = 1, 8, 512, 320
  torch.manual_seed(0)
  device = "cuda"
  q = torch.randn(B, H, N, D, dtype=dtype, device=device, requires_grad=True)
  k = torch.randn(B, H, N, D, dtype=dtype, device=device, requires_grad=True)
  v = torch.randn(B, H, N, D, dtype=dtype, device=device, requires_grad=True)

  # Eager
  qr = q.detach().requires_grad_(True)
  kr = k.detach().requires_grad_(True)
  vr = v.detach().requires_grad_(True)
  out_ref = ffpa_attn_func(
    qr,
    kr,
    vr,
    is_causal=True,
    forward_backend=fw,
    backward_backend=bw,
  )
  out_ref.sum().backward()
  grads_ref = [qr.grad.clone(), kr.grad.clone(), vr.grad.clone()]

  # Compiled
  q.grad = k.grad = v.grad = None
  compiled_fn = torch.compile(ffpa_attn_func, fullgraph=False)
  out = compiled_fn(
    q,
    k,
    v,
    is_causal=True,
    forward_backend=fw,
    backward_backend=bw,
  )
  out.sum().backward()

  torch.testing.assert_close(out, out_ref, **_tolerance(dtype))
  torch.testing.assert_close(q.grad, grads_ref[0], **_tolerance(dtype))
  torch.testing.assert_close(k.grad, grads_ref[1], **_tolerance(dtype))
  torch.testing.assert_close(v.grad, grads_ref[2], **_tolerance(dtype))


# Repeated invocation (cache hit) test
def test_compile_repeated_invocation():
  """Multiple calls through the same compiled function produce consistent output."""
  _require_cuda_forward_impl()
  torch.manual_seed(0)
  device = "cuda"
  B, H, N, D = 1, 8, 512, 320
  q = torch.randn(B, H, N, D, dtype=torch.float16, device=device)
  k = torch.randn(B, H, N, D, dtype=torch.float16, device=device)
  v = torch.randn(B, H, N, D, dtype=torch.float16, device=device)

  compiled_fn = torch.compile(ffpa_attn_func, fullgraph=False)
  out1 = compiled_fn(q, k, v, forward_backend="cuda")
  out2 = compiled_fn(q, k, v, forward_backend="cuda")

  torch.testing.assert_close(out1, out2)

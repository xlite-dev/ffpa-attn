"""SM80/SM89 CuTeDSL Split-D tests with SM90 regression coverage."""

import math

import pytest
import torch
import torch.nn.functional as F

from ffpa_attn import ffpa_attn_func, ffpa_attn_varlen_func
from ffpa_attn.cute import _ffpa_attn_forward_cute, _ffpa_attn_varlen_impl
from ffpa_attn.functional import CuTeDSLBackend


def _cute_large_d_available() -> bool:
  if not torch.cuda.is_available():
    return False
  major, _ = torch.cuda.get_device_capability()
  return major >= 8


pytestmark = pytest.mark.skipif(
  not _cute_large_d_available(),
  reason="CuTeDSL large-D tests require compute capability >= 8.0",
)


def _tol(dtype: torch.dtype) -> dict[str, float]:
  if dtype == torch.float16:
    return {"atol": 2e-3, "rtol": 2e-3}
  return {"atol": 3e-3, "rtol": 3e-3}


def _grad_tol(dtype: torch.dtype) -> dict[str, float]:
  if dtype == torch.float16:
    return {"atol": 8e-3, "rtol": 8e-3}
  return {"atol": 1e-2, "rtol": 1e-2}


@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("D", [320, 512])
def test_sm80_cutedsl_forward_matches_sdpa(dtype, causal, D):
  torch.manual_seed(0)
  B, H, N = 1, 2, 96
  q = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
  k = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
  v = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
  scale = 1.0 / math.sqrt(D)

  out_cute, lse = _ffpa_attn_forward_cute(
    q, k, v, scale, causal, return_lse=True
  )
  out_ref = F.scaled_dot_product_attention(
    q, k, v, dropout_p=0.0, is_causal=causal, scale=scale
  )

  assert out_cute.shape == out_ref.shape
  assert lse.shape == (B, H, N)
  torch.testing.assert_close(out_cute, out_ref, **_tol(dtype))


def test_sm80_cutedsl_forward_zeroes_v_tail_smem_after_dirty_kernel():
  torch.manual_seed(2)
  B, H, N, D = 1, 2, 64, 320
  q, k, v = [
    torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda")
    for _ in range(3)
  ]
  q_ref, k_ref, v_ref = [
    t.detach().clone().requires_grad_(True) for t in (q, k, v)
  ]
  F.scaled_dot_product_attention(q_ref, k_ref, v_ref,
                                 dropout_p=0.0).sum().backward()

  scale = 1.0 / math.sqrt(D)
  out_cute, lse = _ffpa_attn_forward_cute(
    q, k, v, scale, causal=False, return_lse=True
  )
  out_ref = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, scale=scale)

  assert torch.isfinite(out_cute).all()
  assert torch.isfinite(lse).all()
  torch.testing.assert_close(out_cute, out_ref, **_tol(torch.bfloat16))


@pytest.mark.parametrize("D", [320, 512])
def test_sm80_cutedsl_dense_autograd_matches_sdpa(D):
  torch.manual_seed(3)
  B, H, N = 1, 2, 64
  scale = 1.0 / math.sqrt(D)

  def make():
    return [
      torch.randn(
        B, H, N, D, dtype=torch.bfloat16, device="cuda", requires_grad=True
      ) for _ in range(3)
    ]

  q_ref, k_ref, v_ref = make()
  q_cute, k_cute, v_cute = [
    t.detach().clone().requires_grad_(True) for t in (q_ref, k_ref, v_ref)
  ]
  dout = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda")

  out_ref = F.scaled_dot_product_attention(
    q_ref, k_ref, v_ref, dropout_p=0.0, scale=scale
  )
  out_ref.backward(dout)

  out_cute = ffpa_attn_func(
    q_cute, k_cute, v_cute, scale=scale, backend="cutedsl"
  )
  out_cute.backward(dout)

  torch.testing.assert_close(out_cute, out_ref, **_tol(torch.bfloat16))
  torch.testing.assert_close(
    q_cute.grad, q_ref.grad, **_grad_tol(torch.bfloat16)
  )
  torch.testing.assert_close(
    k_cute.grad, k_ref.grad, **_grad_tol(torch.bfloat16)
  )
  torch.testing.assert_close(
    v_cute.grad, v_ref.grad, **_grad_tol(torch.bfloat16)
  )


@pytest.mark.parametrize("causal", [False, True])
def test_sm80_cutedsl_dkdv_fp32_buffer_matches_sdpa(causal):
  """CuTeDSL SM80 dKdV with grad_kv_storage_dtype=fp32 must keep public
  gradient dtype = bf16 and improve dK/dV cross-tile precision under
  causal bf16 vs the default activation-dtype buffer."""
  torch.manual_seed(7)
  B, H, N, D = 1, 4, 256, 512
  dtype = torch.bfloat16
  scale = 1.0 / math.sqrt(D)
  q0 = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
  k0 = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
  v0 = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
  dout = torch.randn(B, H, N, D, dtype=dtype, device="cuda")

  def run(storage_dtype):
    q = q0.detach().clone().requires_grad_(True)
    k = k0.detach().clone().requires_grad_(True)
    v = v0.detach().clone().requires_grad_(True)
    out = ffpa_attn_func(
      q,
      k,
      v,
      is_causal=causal,
      scale=scale,
      forward_backend=CuTeDSLBackend(forward=True),
      backward_backend=CuTeDSLBackend(
        backward=True, grad_kv_storage_dtype=storage_dtype
      ),
    )
    out.backward(dout)
    return q.grad, k.grad, v.grad

  q_ref = q0.detach().clone().requires_grad_(True)
  k_ref = k0.detach().clone().requires_grad_(True)
  v_ref = v0.detach().clone().requires_grad_(True)
  F.scaled_dot_product_attention(
    q_ref,
    k_ref,
    v_ref,
    dropout_p=0.0,
    is_causal=causal,
    scale=scale,
  ).backward(dout)

  dq_def, dk_def, dv_def = run(None)
  dq_f32, dk_f32, dv_f32 = run(torch.float32)

  # Public gradient dtype must remain activation dtype.
  for g in (dq_def, dk_def, dv_def, dq_f32, dk_f32, dv_f32):
    assert g.dtype == dtype

  # fp32 buffer matches SDPA within the standard grad tol.
  torch.testing.assert_close(dq_f32, q_ref.grad, **_grad_tol(dtype))
  torch.testing.assert_close(dk_f32, k_ref.grad, **_grad_tol(dtype))
  torch.testing.assert_close(dv_f32, v_ref.grad, **_grad_tol(dtype))

  # fp32 buffer is at least as accurate as the default storage on dK/dV.
  for g_def, g_f32, g_ref in (
    (dk_def, dk_f32, k_ref.grad),
    (dv_def, dv_f32, v_ref.grad),
  ):
    err_def = (g_def - g_ref).abs().max().item()
    err_f32 = (g_f32 - g_ref).abs().max().item()
    assert err_f32 <= err_def + 1e-6, (err_def, err_f32)


def test_sm80_cutedsl_varlen_forward_matches_sdpa():
  torch.manual_seed(1)
  lengths = [32, 17, 64]
  cu_seqlens = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()],
                            device="cuda",
                            dtype=torch.int32)
  total_q = int(cu_seqlens[-1].item())
  num_heads, head_dim = 2, 320
  q = torch.randn(
    total_q, num_heads, head_dim, dtype=torch.bfloat16, device="cuda"
  )
  k = torch.randn_like(q)
  v = torch.randn_like(q)
  scale = 1.0 / math.sqrt(head_dim)

  out_cute, lse = _ffpa_attn_varlen_impl(
    q,
    k,
    v,
    cu_seqlens,
    cu_seqlens,
    max(lengths),
    max(lengths),
    softmax_scale=scale,
    causal=False,
    return_lse=True,
  )

  refs = []
  for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
    q_segment = q[start:end].transpose(0, 1).unsqueeze(0)
    k_segment = k[start:end].transpose(0, 1).unsqueeze(0)
    v_segment = v[start:end].transpose(0, 1).unsqueeze(0)
    refs.append(
      F.scaled_dot_product_attention(
        q_segment,
        k_segment,
        v_segment,
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
      ).squeeze(0).transpose(0, 1)
    )
  out_ref = torch.cat(refs, dim=0)

  assert out_cute.shape == out_ref.shape
  assert lse.shape == (num_heads, total_q)
  torch.testing.assert_close(out_cute, out_ref, **_tol(torch.bfloat16))


def test_sm80_cutedsl_varlen_autograd_matches_sdpa():
  torch.manual_seed(4)
  lengths = [32, 17, 64]
  cu_seqlens = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()],
                            device="cuda",
                            dtype=torch.int32)
  total_q = int(cu_seqlens[-1].item())
  num_heads, head_dim = 2, 320
  scale = 1.0 / math.sqrt(head_dim)
  q_ref = torch.randn(
    total_q,
    num_heads,
    head_dim,
    dtype=torch.bfloat16,
    device="cuda",
    requires_grad=True
  )
  k_ref = torch.randn_like(q_ref, requires_grad=True)
  v_ref = torch.randn_like(q_ref, requires_grad=True)
  q_cute, k_cute, v_cute = [
    t.detach().clone().requires_grad_(True) for t in (q_ref, k_ref, v_ref)
  ]
  dout = torch.randn_like(q_ref)

  out_cute = ffpa_attn_varlen_func(
    q_cute,
    k_cute,
    v_cute,
    cu_seqlens,
    cu_seqlens,
    max(lengths),
    max(lengths),
    softmax_scale=scale,
  )
  out_cute.backward(dout)

  refs = []
  for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
    q_segment = q_ref[start:end].transpose(0, 1).unsqueeze(0)
    k_segment = k_ref[start:end].transpose(0, 1).unsqueeze(0)
    v_segment = v_ref[start:end].transpose(0, 1).unsqueeze(0)
    refs.append(
      F.scaled_dot_product_attention(
        q_segment,
        k_segment,
        v_segment,
        dropout_p=0.0,
        scale=scale,
      ).squeeze(0).transpose(0, 1)
    )
  out_ref = torch.cat(refs, dim=0)
  out_ref.backward(dout)

  torch.testing.assert_close(out_cute, out_ref, **_tol(torch.bfloat16))
  torch.testing.assert_close(
    q_cute.grad, q_ref.grad, **_grad_tol(torch.bfloat16)
  )
  torch.testing.assert_close(
    k_cute.grad, k_ref.grad, **_grad_tol(torch.bfloat16)
  )
  torch.testing.assert_close(
    v_cute.grad, v_ref.grad, **_grad_tol(torch.bfloat16)
  )

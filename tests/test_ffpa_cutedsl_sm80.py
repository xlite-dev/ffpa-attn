"""SM80/SM89 CuTeDSL Split-D tests with SM90 regression coverage."""

import math

import pytest
import torch
import torch.nn.functional as F

from ffpa_attn import ffpa_attn_func, ffpa_attn_varlen_func
from ffpa_attn.cutedsl import _ffpa_attn_forward_cutedsl, _ffpa_attn_varlen_impl


def _cutedsl_large_d_available() -> bool:
  if not torch.cuda.is_available():
    return False
  major, _ = torch.cuda.get_device_capability()
  return major in (8, 9)


pytestmark = pytest.mark.skipif(
  not _cutedsl_large_d_available(),
  reason="CuTeDSL large-D tests require compute capability 8.x or 9.x",
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

  out_cute, lse = _ffpa_attn_forward_cutedsl(
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
  out_cute, lse = _ffpa_attn_forward_cutedsl(
    q, k, v, scale, causal=False, return_lse=True
  )
  out_ref = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, scale=scale)

  assert torch.isfinite(out_cute).all()
  assert torch.isfinite(lse).all()
  torch.testing.assert_close(out_cute, out_ref, **_tol(torch.bfloat16))


def test_sm80_cutedsl_dense_autograd_matches_sdpa():
  torch.manual_seed(3)
  B, H, N, D = 1, 2, 64, 320
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

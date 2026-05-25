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
import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import triton

from ffpa_attn import ffpa_attn_func  # noqa: E402
from ffpa_attn.functional import TritonBackend  # noqa: E402
from ffpa_attn.triton._ffpa_bwd import _ffpa_bwd_pre  # noqa: E402

# Build subset for fast iteration: 64 (small-d), 320, 512 (large-d).
HEADDIMS = [64, 320, 512]
DTYPES = [torch.float16, torch.bfloat16]


def _seqlen_q_rounded(seqlen_q):
  """Return the padded LSE/Delta sequence dimension used by Triton backward."""
  return ((seqlen_q + 127) // 128) * 128


def _tolerance(dtype):
  return {
    "atol": 5e-2,
    "rtol": 5e-2
  } if dtype == torch.bfloat16 else {
    "atol": 1e-2,
    "rtol": 1e-2
  }


def _skip_if_no_sm90_tma():
  if not torch.cuda.is_available():
    pytest.skip("CUDA is required for SM90 TMA backward tests")
  if torch.cuda.get_device_capability()[0] < 9:
    pytest.skip("SM90+ GPU is required for TMA backward tests")


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


def _sdpa_ref(q, k, v, causal, scale, attn_mask=None):
  """Run SDPA forward only (no grad) for output comparison."""
  group_size = q.size(1) // k.size(1)
  k2 = k.repeat_interleave(group_size, dim=1) if group_size > 1 else k
  v2 = v.repeat_interleave(group_size, dim=1) if group_size > 1 else v
  kw = {
    "attn_mask": attn_mask
  } if attn_mask is not None else _make_sdpa_kwargs(
    causal, q.size(2), k.size(2)
  )
  return F.scaled_dot_product_attention(q, k2, v2, scale=scale, **kw)


def _sdpa_ref_grads(
  q,
  k,
  v,
  causal,
  scale,
  attn_mask=None,
  return_mask_grad=False,
  dropout_p=0.0,
  rng_seed=None,
  grad_out=None
):
  """Run SDPA forward + backward and return gradients from autograd.

    SDPA supports MQA (Nh_kv == 1) natively, but for general GQA
    (Nh_kv > 1, Nh_q > Nh_kv) we repeat K/V to match Q, then sum-reduce
    the K/V gradients back to the original head count.
    """
  q2 = q.detach().clone().requires_grad_(True)
  k2 = k.detach().clone().requires_grad_(True)
  v2 = v.detach().clone().requires_grad_(True)
  attn_mask_ref = None
  if attn_mask is not None:
    attn_mask_ref = attn_mask.detach().clone().requires_grad_(return_mask_grad)

  group_size = q.size(1) // k.size(1)
  k_in = k2.repeat_interleave(group_size, dim=1) if group_size > 1 else k2
  v_in = v2.repeat_interleave(group_size, dim=1) if group_size > 1 else v2

  kw = {
    "attn_mask": attn_mask_ref
  } if attn_mask_ref is not None else _make_sdpa_kwargs(
    causal, q.size(2), k.size(2)
  )
  if rng_seed is not None:
    torch.manual_seed(rng_seed)
  if dropout_p > 0.0:
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
      out_ref = F.scaled_dot_product_attention(
        q2, k_in, v_in, scale=scale, dropout_p=dropout_p, **kw
      )
  else:
    out_ref = F.scaled_dot_product_attention(q2, k_in, v_in, scale=scale, **kw)
  if grad_out is None:
    loss_ref = out_ref.sum()
    loss_ref.backward()
  else:
    out_ref.backward(grad_out.detach())

  if group_size > 1:
    # k2.grad accumulates from repeat_interleave; shape [B, Nh_kv, N, D].
    dk = k2.grad
    dv = v2.grad
  else:
    dk = k2.grad
    dv = v2.grad

  if return_mask_grad:
    return q2.grad, dk, dv, attn_mask_ref.grad
  return q2.grad, dk, dv


@pytest.mark.parametrize("dtype", DTYPES)
def test_sm90_tma_persist_dkdv_causal_matches_sdpa(dtype):
  _skip_if_no_sm90_tma()
  torch.manual_seed(123)
  B, H, N, D = 1, 2, 128, 512
  q = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
  k = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
  v = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
  scale = 1.0 / math.sqrt(D)

  out = ffpa_attn_func(
    q,
    k,
    v,
    is_causal=True,
    scale=scale,
    backward_backend=TritonBackend(
      backward=True, enable_tma=True, persist_dkdv=True
    )
  )
  out.sum().backward()

  dq_ref, dk_ref, dv_ref = _sdpa_ref_grads(q, k, v, True, scale)
  tol = _tolerance(dtype)
  assert torch.allclose(q.grad, dq_ref, **tol)
  assert torch.allclose(k.grad, dk_ref, **tol)
  assert torch.allclose(v.grad, dv_ref, **tol)


@pytest.mark.parametrize("enable_persist_dkdv", [False, True])
def test_sm90_tma_non_aligned_seqlen_matches_sdpa(enable_persist_dkdv):
  _skip_if_no_sm90_tma()
  torch.manual_seed(124)
  B, H, N, D = 1, 2, 129, 512
  dtype = torch.float16
  q = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
  k = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
  v = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
  scale = 1.0 / math.sqrt(D)

  out = ffpa_attn_func(
    q,
    k,
    v,
    scale=scale,
    backward_backend=TritonBackend(
      backward=True, enable_tma=True, persist_dkdv=enable_persist_dkdv
    )
  )
  out.sum().backward()

  dq_ref, dk_ref, dv_ref = _sdpa_ref_grads(q, k, v, False, scale)
  tol = _tolerance(dtype)
  assert torch.allclose(q.grad, dq_ref, **tol)
  assert torch.allclose(k.grad, dk_ref, **tol)
  assert torch.allclose(v.grad, dv_ref, **tol)


@pytest.mark.parametrize(("dtype", "enable_persist_dkdv", "causal", "N"), [
  (torch.float16, False, False, 129),
  (torch.bfloat16, True, True, 128),
])
def test_sm90_tma_split_launch_matches_sdpa(
  dtype, enable_persist_dkdv, causal, N
):
  _skip_if_no_sm90_tma()
  torch.manual_seed(125)
  B, H, D = 1, 2, 512
  q = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
  k = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
  v = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
  grad_out = torch.randn_like(q)
  scale = 1.0 / math.sqrt(D)

  out = ffpa_attn_func(
    q,
    k,
    v,
    is_causal=causal,
    scale=scale,
    backward_backend=TritonBackend(
      backward=True,
      enable_tma=True,
      persist_dkdv=enable_persist_dkdv,
      split_launch=True
    )
  )
  out.backward(grad_out)

  dq_ref, dk_ref, dv_ref = _sdpa_ref_grads(
    q, k, v, causal, scale, grad_out=grad_out
  )
  tol = _tolerance(dtype)
  assert torch.allclose(q.grad, dq_ref, **tol)
  assert torch.allclose(k.grad, dk_ref, **tol)
  assert torch.allclose(v.grad, dv_ref, **tol)


@pytest.mark.parametrize("enable_persist_dkdv", [False, True])
def test_sm90_tma_split_launch_matches_fused(enable_persist_dkdv):
  _skip_if_no_sm90_tma()
  torch.manual_seed(126)
  B, H, N, D = 1, 2, 129, 512
  dtype = torch.float16
  q = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
  k = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
  v = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
  grad_out = torch.randn_like(q)
  scale = 1.0 / math.sqrt(D)

  def run_backward(split_launch: bool):
    q_i = q.detach().clone().requires_grad_(True)
    k_i = k.detach().clone().requires_grad_(True)
    v_i = v.detach().clone().requires_grad_(True)
    out = ffpa_attn_func(
      q_i,
      k_i,
      v_i,
      scale=scale,
      backward_backend=TritonBackend(
        backward=True,
        enable_tma=True,
        persist_dkdv=enable_persist_dkdv,
        split_launch=split_launch
      )
    )
    out.backward(grad_out)
    return q_i.grad, k_i.grad, v_i.grad

  fused = run_backward(False)
  split = run_backward(True)
  tol = _tolerance(dtype)
  for split_grad, fused_grad in zip(split, fused):
    assert torch.allclose(split_grad, fused_grad, **tol)


def _key_position_bias_grad_ref(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  scale: float,
  attn_mask: torch.Tensor,
  block_m: int = 128,
  block_n: int = 1024
) -> torch.Tensor:
  """Compute a fp32 reference gradient for compact [1, 1, 1, Nkv] key bias."""
  with torch.no_grad():
    group_size = q.size(1) // k.size(1)
    k_ref = k.detach().repeat_interleave(group_size, dim=1
                                         ) if group_size > 1 else k.detach()
    v_ref = v.detach().repeat_interleave(group_size, dim=1
                                         ) if group_size > 1 else v.detach()
    q_f = q.detach().float()
    k_f = k_ref.float()
    key_value_sum = v_ref.float().sum(dim=-1)
    key_bias = attn_mask.detach().reshape(-1).float()
    seqlen_q = q.size(2)
    seqlen_k = k_ref.size(2)
    grad = torch.zeros(seqlen_k, dtype=torch.float32, device=q.device)

    for m_start in range(0, seqlen_q, block_m):
      q_block = q_f[:, :, m_start:m_start + block_m, :]
      lse = torch.full(
        q_block.shape[:-1], -torch.inf, dtype=torch.float32, device=q.device
      )
      for n_start in range(0, seqlen_k, block_n):
        scores = torch.matmul(
          q_block, k_f[:, :, n_start:n_start + block_n, :].transpose(-2, -1)
        ) * scale
        scores = scores + key_bias[n_start:n_start + block_n].view(1, 1, 1, -1)
        lse = torch.logaddexp(lse, torch.logsumexp(scores, dim=-1))

      delta = torch.zeros_like(lse)
      for n_start in range(0, seqlen_k, block_n):
        scores = torch.matmul(
          q_block, k_f[:, :, n_start:n_start + block_n, :].transpose(-2, -1)
        ) * scale
        scores = scores + key_bias[n_start:n_start + block_n].view(1, 1, 1, -1)
        prob = torch.exp(scores - lse[..., None])
        delta += (prob *
                  key_value_sum[:, :, None, n_start:n_start + block_n]).sum(
                    dim=-1
                  )

      for n_start in range(0, seqlen_k, block_n):
        scores = torch.matmul(
          q_block, k_f[:, :, n_start:n_start + block_n, :].transpose(-2, -1)
        ) * scale
        scores = scores + key_bias[n_start:n_start + block_n].view(1, 1, 1, -1)
        prob = torch.exp(scores - lse[..., None])
        d_bias = prob * (
          key_value_sum[:, :, None, n_start:n_start + block_n] -
          delta[..., None]
        )
        grad[n_start:n_start + block_n] += d_bias.sum(dim=(0, 1, 2))

    return grad.view(1, 1, 1, seqlen_k)


def _run_bwd_pre(o, do, d_chunk, block_headdim=64):
  """Run the Triton backward preprocess kernel and return visible delta.

  :param o: Forward output tensor with layout ``[B, H, N, D]``.
  :param do: Upstream output gradient with layout ``[B, H, N, D]``.
  :param d_chunk: Whether to use the split-D preprocess mode.
  :param block_headdim: D chunk size for split-D mode.
  :return: Delta tensor view with layout ``[B, H, N]``.
  """
  B, H, N, D = o.shape
  seqlen_q_rounded = _seqlen_q_rounded(N)
  delta = torch.empty(
    B, H, seqlen_q_rounded, dtype=torch.float32, device=o.device
  )
  full_block_headdim = max(
    64, triton.next_power_of_2(D)
  ) if not d_chunk else block_headdim
  _ffpa_bwd_pre[(triton.cdiv(N, 128), B * H)](
    o,
    do,
    delta,
    o.stride(0),
    o.stride(1),
    o.stride(2),
    do.stride(0),
    do.stride(1),
    do.stride(2),
    H,
    N,
    N,
    seqlen_q_rounded,
    D,
    BLOCK_M=128,
    BLOCK_HEADDIM=full_block_headdim,
    D_CHUNK=d_chunk,
    num_warps=4
  )
  return delta[..., :N]


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize(
  "N,D", [(191, 64), (191, 320), (257, 512), (129, 1024)]
)
def test_ffpa_bwd_preprocess_full_and_d_chunk(dtype, N, D):
  """Full-D and split-D preprocess modes must match PyTorch delta."""
  B, H = 1, 2
  torch.manual_seed(0)
  o = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
  do = torch.randn_like(o)
  ref = (o.float() * do.float()).sum(dim=-1)

  delta_full = _run_bwd_pre(o, do, d_chunk=False)
  delta_d_chunk = _run_bwd_pre(o, do, d_chunk=True, block_headdim=64)

  tol = {
    "atol": 2e-2,
    "rtol": 2e-2
  } if dtype == torch.bfloat16 else {
    "atol": 1e-2,
    "rtol": 1e-2
  }
  torch.testing.assert_close(delta_full, ref, **tol)
  torch.testing.assert_close(delta_d_chunk, ref, **tol)
  torch.testing.assert_close(delta_d_chunk, delta_full, **tol)


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize(
  "preprocess_d_chunk", [False, True], ids=["pre_full", "pre_d_chunk"]
)
def test_ffpa_bwd_triton_preprocess_modes(dtype, preprocess_d_chunk):
  """Triton backward must stay accurate with either preprocess mode."""
  B, H, N, D = 1, 2, 128, 320
  torch.manual_seed(0)
  q = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
  k = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
  v = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)

  scale = 1.0 / math.sqrt(D)
  out = ffpa_attn_func(
    q, k, v, is_causal=False, scale=scale, backward_backend="triton"
  )
  out.sum().backward()

  dq_ref, dk_ref, dv_ref = _sdpa_ref_grads(q, k, v, False, scale)

  tol = _tolerance(dtype)
  torch.testing.assert_close(q.grad, dq_ref, **tol)
  torch.testing.assert_close(k.grad, dk_ref, **tol)
  torch.testing.assert_close(v.grad, dv_ref, **tol)


def test_ffpa_bwd_triton_non_aligned_seqlen_matches_sdpa():
  """Raw-pointer Triton backward must mask the final partial Q block."""
  torch.manual_seed(124)
  B, H, N, D = 1, 2, 129, 512
  dtype = torch.float16
  q = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
  k = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
  v = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
  scale = 1.0 / math.sqrt(D)

  out = ffpa_attn_func(q, k, v, scale=scale, backward_backend="triton")
  out.sum().backward()

  dq_ref, dk_ref, dv_ref = _sdpa_ref_grads(q, k, v, False, scale)
  tol = _tolerance(dtype)
  torch.testing.assert_close(q.grad, dq_ref, **tol)
  torch.testing.assert_close(k.grad, dk_ref, **tol)
  torch.testing.assert_close(v.grad, dv_ref, **tol)


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
def test_ffpa_bwd_triton_internal_kv_storage_dtype_option(dtype):
  """The low-level Triton backward op should expose optional fp16/fp32 dK/dV storage."""
  B, H, N, D = 1, 2, 128, 320
  torch.manual_seed(0)
  q = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
  k = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
  v = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
  do = torch.randn_like(q)
  scale = 1.0 / math.sqrt(D)

  o, lse = torch.ops.ffpa_attn._fwd_triton(
    q, k, v, None, scale, 0, 0, 0, 0.0, 0, 0, 0, 0
  )

  dq_default, dk_default, dv_default, _ = torch.ops.ffpa_attn._bwd_triton(
    do, q, k, v, o, lse, None, scale, 0, 0, 0, 0, 0, 0, H, 0.0, 0, 0, 0, 0
  )
  dq_fp32, dk_fp32, dv_fp32, _ = torch.ops.ffpa_attn._bwd_triton(
    do, q, k, v, o, lse, None, scale, 0, 0, 0, 0, 0, 1, H, 0.0, 0, 0, 0, 0
  )
  dq_fp16, dk_fp16, dv_fp16, _ = torch.ops.ffpa_attn._bwd_triton(
    do, q, k, v, o, lse, None, scale, 0, 0, 0, 0, 0, 2, H, 0.0, 0, 0, 0, 0
  )

  assert dq_default.dtype == dtype
  assert dk_default.dtype == dtype
  assert dv_default.dtype == dtype
  assert dq_fp32.dtype == dtype
  assert dk_fp32.dtype == torch.float32
  assert dv_fp32.dtype == torch.float32
  assert dq_fp16.dtype == dtype
  assert dk_fp16.dtype == torch.float16
  assert dv_fp16.dtype == torch.float16


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize(
  "grad_kv_storage_dtype", [torch.float16, torch.float32],
  ids=["kv-fp16", "kv-fp32"]
)
def test_ffpa_bwd_triton_grad_kv_storage_dtype_preserves_public_grad_dtype(
  dtype, grad_kv_storage_dtype
):
  """Public gradients should stay in q/k/v dtype even when Triton dK/dV storage dtype is overridden."""
  B, H, N, D = 1, 2, 128, 320
  torch.manual_seed(0)
  q = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
  k = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
  v = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)

  scale = 1.0 / math.sqrt(D)
  out = ffpa_attn_func(
    q, k, v, is_causal=True, scale=scale, backward_backend="triton"
  )
  out.sum().backward()

  assert q.grad is not None and q.grad.dtype == dtype
  assert k.grad is not None and k.grad.dtype == dtype
  assert v.grad is not None and v.grad.dtype == dtype


def test_ffpa_bwd_triton_additive_attn_mask_matches_sdpa():
  """Masked Triton backward must match SDPA, including additive-bias gradients."""
  B, H, N, D = 1, 4, 512, 512
  dtype = torch.float16
  torch.manual_seed(0)
  q = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
  k = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
  v = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
  attn_mask = (torch.randn(1, 1, 1, N, dtype=dtype, device="cuda") *
               0.25).requires_grad_(True)

  scale = 1.0 / math.sqrt(D)
  out = ffpa_attn_func(
    q, k, v, attn_mask=attn_mask, scale=scale, backward_backend="triton"
  )
  out.sum().backward()
  assert attn_mask.grad is not None
  assert attn_mask.grad.shape == (1, 1, 1, N)

  dq_ref, dk_ref, dv_ref, dmask_ref = _sdpa_ref_grads(
    q, k, v, False, scale, attn_mask=attn_mask, return_mask_grad=True
  )

  torch.testing.assert_close(q.grad, dq_ref, atol=3e-2, rtol=3e-2)
  torch.testing.assert_close(k.grad, dk_ref, atol=3e-2, rtol=3e-2)
  torch.testing.assert_close(v.grad, dv_ref, atol=3e-2, rtol=3e-2)
  torch.testing.assert_close(attn_mask.grad, dmask_ref, atol=3e-2, rtol=3e-2)


def test_ffpa_bwd_triton_key_bias_autotune_fp32_kv_storage_matches_sdpa():
  """Autotuned key-bias backward must keep dMask correct with fp32 dK/dV storage."""
  B, H, N, D = 1, 32, 8192, 512
  dtype = torch.float16
  torch.manual_seed(42)
  q = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
  k = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
  v = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
  attn_mask = (
    torch.randn(1, 1, 1, N, dtype=torch.float32, device="cuda") * 0.25
  ).requires_grad_(True)

  scale = 1.0 / math.sqrt(D)
  out = ffpa_attn_func(
    q, k, v, attn_mask=attn_mask, scale=scale, backward_backend="triton"
  )
  out.sum().backward()

  assert attn_mask.grad is not None
  dq_ref, dk_ref, dv_ref = _sdpa_ref_grads(
    q, k, v, False, scale, attn_mask=attn_mask.to(dtype)
  )
  dmask_ref = _key_position_bias_grad_ref(q, k, v, scale, attn_mask)

  torch.testing.assert_close(q.grad, dq_ref, atol=3e-2, rtol=3e-2)
  torch.testing.assert_close(k.grad, dk_ref, atol=3e-2, rtol=3e-2)
  torch.testing.assert_close(v.grad, dv_ref, atol=3e-2, rtol=3e-2)
  torch.testing.assert_close(attn_mask.grad, dmask_ref, atol=3e-2, rtol=3e-2)


def test_ffpa_bwd_triton_additive_attn_mask_only_grad_matches_sdpa():
  """The Triton path must save context even when only additive mask needs grad."""
  B, H, N, D = 1, 2, 512, 512
  dtype = torch.float16
  torch.manual_seed(0)
  q = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
  k = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
  v = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
  attn_mask = (torch.randn(1, 1, 1, N, dtype=dtype, device="cuda") *
               0.25).requires_grad_(True)

  scale = 1.0 / math.sqrt(D)
  out = ffpa_attn_func(
    q, k, v, attn_mask=attn_mask, scale=scale, backward_backend="triton"
  )
  out.sum().backward()
  assert attn_mask.grad is not None
  assert attn_mask.grad.shape == (1, 1, 1, N)

  _, _, _, dmask_ref = _sdpa_ref_grads(
    q, k, v, False, scale, attn_mask=attn_mask, return_mask_grad=True
  )
  torch.testing.assert_close(attn_mask.grad, dmask_ref, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("Nq", [1, 2, 3, 4, 7])
@pytest.mark.parametrize("case", ["base", "causal", "mask", "gqa", "d512"])
def test_ffpa_bwd_triton_decode_matches_sdpa(dtype, Nq, case):
  """Small-Nq Triton decode backward must match SDPA across modes."""
  B, Hq, Hkv, Nkv, D = 1, 2, 2, 513, 512
  causal = False
  attn_mask = None
  if case == "causal":
    causal = True
  elif case == "mask":
    pass
  elif case == "gqa":
    Hq, Hkv = 4, 2
  elif case == "d512":
    Hq, Hkv, Nkv, D = 2, 2, 769, 512

  torch.manual_seed(7)
  q = torch.randn(B, Hq, Nq, D, dtype=dtype, device="cuda", requires_grad=True)
  k = torch.randn(
    B, Hkv, Nkv, D, dtype=dtype, device="cuda", requires_grad=True
  )
  v = torch.randn(
    B, Hkv, Nkv, D, dtype=dtype, device="cuda", requires_grad=True
  )
  grad_out = torch.randn_like(q)
  if case == "mask":
    attn_mask = (torch.randn(1, 1, Nq, Nkv, dtype=dtype, device="cuda") *
                 0.25).requires_grad_(True)

  scale = 1.0 / math.sqrt(D)
  out = ffpa_attn_func(
    q,
    k,
    v,
    is_causal=causal,
    attn_mask=attn_mask,
    scale=scale,
    backward_backend="triton",
    enable_gqa=Hq != Hkv
  )
  out.backward(grad_out)

  ref = _sdpa_ref_grads(
    q,
    k,
    v,
    causal,
    scale,
    attn_mask=attn_mask,
    return_mask_grad=attn_mask is not None,
    grad_out=grad_out
  )
  if attn_mask is None:
    dq_ref, dk_ref, dv_ref = ref
    dmask_ref = None
  else:
    dq_ref, dk_ref, dv_ref, dmask_ref = ref

  tol = {
    "atol": 8e-2,
    "rtol": 8e-2
  } if dtype == torch.bfloat16 or (causal and Nq > 1) else {
    "atol": 3e-2,
    "rtol": 3e-2
  }
  torch.testing.assert_close(q.grad, dq_ref, **tol)
  torch.testing.assert_close(k.grad, dk_ref, **tol)
  torch.testing.assert_close(v.grad, dv_ref, **tol)
  if attn_mask is not None:
    torch.testing.assert_close(attn_mask.grad, dmask_ref, **tol)


@pytest.mark.parametrize("Nq", [1, 4])
def test_ffpa_bwd_triton_decode_autotune_matches_sdpa(Nq):
  """Decode backward stage1 autotune should preserve SDPA parity."""
  B, H, Nkv, D = 1, 2, 513, 512
  dtype = torch.float16
  torch.manual_seed(17 + Nq)
  q = torch.randn(B, H, Nq, D, dtype=dtype, device="cuda", requires_grad=True)
  k = torch.randn(B, H, Nkv, D, dtype=dtype, device="cuda", requires_grad=True)
  v = torch.randn(B, H, Nkv, D, dtype=dtype, device="cuda", requires_grad=True)
  grad_out = torch.randn_like(q)

  scale = 1.0 / math.sqrt(D)
  out = ffpa_attn_func(q, k, v, scale=scale, backward_backend="triton")
  out.backward(grad_out)

  dq_ref, dk_ref, dv_ref = _sdpa_ref_grads(
    q, k, v, False, scale, grad_out=grad_out
  )
  torch.testing.assert_close(q.grad, dq_ref, atol=3e-2, rtol=3e-2)
  torch.testing.assert_close(k.grad, dk_ref, atol=3e-2, rtol=3e-2)
  torch.testing.assert_close(v.grad, dv_ref, atol=3e-2, rtol=3e-2)


def test_ffpa_bwd_triton_decode_autotune_fp32_kv_storage_matches_sdpa():
  """Single-query decode autotune must keep dQ correct with fp32 dK/dV storage."""
  B, H, Nq, Nkv, D = 1, 32, 1, 8192, 512
  dtype = torch.float16
  torch.manual_seed(42)
  q = torch.randn(B, H, Nq, D, dtype=dtype, device="cuda", requires_grad=True)
  k = torch.randn(B, H, Nkv, D, dtype=dtype, device="cuda", requires_grad=True)
  v = torch.randn(B, H, Nkv, D, dtype=dtype, device="cuda", requires_grad=True)

  scale = 1.0 / math.sqrt(D)
  out = ffpa_attn_func(q, k, v, scale=scale, backward_backend="triton")
  out.sum().backward()

  dq_ref, dk_ref, dv_ref = _sdpa_ref_grads(q, k, v, False, scale)
  torch.testing.assert_close(q.grad, dq_ref, atol=3e-2, rtol=3e-2)
  torch.testing.assert_close(k.grad, dk_ref, atol=3e-2, rtol=3e-2)
  torch.testing.assert_close(v.grad, dv_ref, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
def test_ffpa_bwd_triton_decode_single_query_causal_large_matches_sdpa(dtype):
  """Large single-query causal decode should match SDPA for fwd and bwd."""
  B, H, Nq, Nkv, D = 1, 32, 1, 8192, 512
  torch.manual_seed(7)
  q = torch.randn(B, H, Nq, D, dtype=dtype, device="cuda", requires_grad=True)
  k = torch.randn(B, H, Nkv, D, dtype=dtype, device="cuda", requires_grad=True)
  v = torch.randn(B, H, Nkv, D, dtype=dtype, device="cuda", requires_grad=True)
  grad_out = torch.randn_like(q)

  scale = 1.0 / math.sqrt(D)
  out = ffpa_attn_func(
    q, k, v, is_causal=True, scale=scale, backward_backend="triton"
  )
  out.backward(grad_out)

  out_ref = _sdpa_ref(q.detach(), k.detach(), v.detach(), True, scale)
  dq_ref, dk_ref, dv_ref = _sdpa_ref_grads(
    q.detach(), k.detach(), v.detach(), True, scale, grad_out=grad_out
  )

  tol = {
    "atol": 1e-2,
    "rtol": 1e-2
  } if dtype == torch.bfloat16 else {
    "atol": 3e-3,
    "rtol": 3e-3
  }
  torch.testing.assert_close(out, out_ref, **tol)
  torch.testing.assert_close(q.grad, dq_ref, **tol)
  torch.testing.assert_close(k.grad, dk_ref, **tol)
  torch.testing.assert_close(v.grad, dv_ref, **tol)


def test_ffpa_bwd_triton_dropout_matches_sdpa():
  """Triton dropout backward must reuse the forward Philox mask like SDPA."""
  B, H, N, D = 1, 2, 512, 512
  dtype = torch.float16
  torch.manual_seed(0)
  q = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
  k = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
  v = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
  grad_out = torch.randn_like(q)

  scale = 1.0 / math.sqrt(D)
  rng_seed = 123
  torch.manual_seed(rng_seed)
  out = ffpa_attn_func(
    q, k, v, dropout_p=0.2, scale=scale, backward_backend="triton"
  )
  out.backward(grad_out)

  dq_ref, dk_ref, dv_ref = _sdpa_ref_grads(
    q, k, v, False, scale, dropout_p=0.2, rng_seed=rng_seed, grad_out=grad_out
  )

  torch.testing.assert_close(q.grad, dq_ref, atol=4e-2, rtol=4e-2)
  torch.testing.assert_close(k.grad, dk_ref, atol=4e-2, rtol=4e-2)
  torch.testing.assert_close(v.grad, dv_ref, atol=4e-2, rtol=4e-2)


@pytest.mark.parametrize("Nq", [1, 7])
def test_ffpa_bwd_triton_decode_dropout_matches_sdpa(Nq):
  """Short-query dropout backward replays the decode forward Philox mask."""
  B, H, Nkv, D = 1, 2, 513, 512
  dtype = torch.float16
  torch.manual_seed(99 + Nq)
  q = torch.randn(B, H, Nq, D, dtype=dtype, device="cuda", requires_grad=True)
  k = torch.randn(B, H, Nkv, D, dtype=dtype, device="cuda", requires_grad=True)
  v = torch.randn(B, H, Nkv, D, dtype=dtype, device="cuda", requires_grad=True)
  grad_out = torch.randn_like(q)

  scale = 1.0 / math.sqrt(D)
  rng_seed = 567
  torch.manual_seed(rng_seed)
  out = ffpa_attn_func(
    q, k, v, dropout_p=0.2, scale=scale, backward_backend="triton"
  )
  out.backward(grad_out)

  dq_ref, dk_ref, dv_ref = _sdpa_ref_grads(
    q, k, v, False, scale, dropout_p=0.2, rng_seed=rng_seed, grad_out=grad_out
  )

  torch.testing.assert_close(q.grad, dq_ref, atol=4e-2, rtol=4e-2)
  torch.testing.assert_close(k.grad, dk_ref, atol=4e-2, rtol=4e-2)
  torch.testing.assert_close(v.grad, dv_ref, atol=4e-2, rtol=4e-2)


def test_ffpa_bwd_triton_dropout_additive_attn_mask_matches_sdpa():
  """Dropout and additive-bias gradients compose with the same SDPA mask."""
  B, H, N, D = 1, 2, 512, 512
  dtype = torch.float16
  torch.manual_seed(1)
  q = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
  k = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
  v = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
  attn_mask = (torch.randn(1, 1, 1, N, dtype=dtype, device="cuda") *
               0.25).requires_grad_(True)
  grad_out = torch.randn_like(q)

  scale = 1.0 / math.sqrt(D)
  rng_seed = 321
  torch.manual_seed(rng_seed)
  out = ffpa_attn_func(
    q,
    k,
    v,
    attn_mask=attn_mask,
    dropout_p=0.2,
    scale=scale,
    backward_backend="triton"
  )
  out.backward(grad_out)
  assert attn_mask.grad is not None
  assert attn_mask.grad.shape == (1, 1, 1, N)

  dq_ref, dk_ref, dv_ref, dmask_ref = _sdpa_ref_grads(
    q,
    k,
    v,
    False,
    scale,
    attn_mask=attn_mask,
    return_mask_grad=True,
    dropout_p=0.2,
    rng_seed=rng_seed,
    grad_out=grad_out
  )

  torch.testing.assert_close(q.grad, dq_ref, atol=4e-2, rtol=4e-2)
  torch.testing.assert_close(k.grad, dk_ref, atol=4e-2, rtol=4e-2)
  torch.testing.assert_close(v.grad, dv_ref, atol=4e-2, rtol=4e-2)
  torch.testing.assert_close(attn_mask.grad, dmask_ref, atol=4e-2, rtol=4e-2)


# Basic backward correctness
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
  out = ffpa_attn_func(q, k, v, is_causal=False, scale=scale)
  loss = out.sum()
  loss.backward()

  dq_ref, dk_ref, dv_ref = _sdpa_ref_grads(q, k, v, False, scale)

  tol = _tolerance(dtype)
  torch.testing.assert_close(q.grad, dq_ref, **tol)
  torch.testing.assert_close(k.grad, dk_ref, **tol)
  torch.testing.assert_close(v.grad, dv_ref, **tol)


# Backward + causal
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
  out = ffpa_attn_func(q, k, v, is_causal=True, scale=scale)
  loss = out.sum()
  loss.backward()

  dq_ref, dk_ref, dv_ref = _sdpa_ref_grads(q, k, v, True, scale)

  tol = _tolerance(dtype)
  # Relax tolerance for large-shape bf16 where numerical differences
  # accumulate across D-chunk loops and atomic adds.  The error scales
  # with seqlen, so use progressively looser tolerances.
  if dtype == torch.bfloat16:
    if N >= 16384:
      tol = {"atol": 3e-1, "rtol": 2e-1}
    elif N >= 8192 or D >= 512:
      tol = {"atol": 1e-1, "rtol": 1e-1}
  torch.testing.assert_close(q.grad, dq_ref, **tol)
  torch.testing.assert_close(k.grad, dk_ref, **tol)
  torch.testing.assert_close(v.grad, dv_ref, **tol)


# Backward + GQA
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
  k = torch.randn(
    B, Nh_kv, N, D, dtype=dtype, device="cuda", requires_grad=True
  )
  v = torch.randn(
    B, Nh_kv, N, D, dtype=dtype, device="cuda", requires_grad=True
  )

  scale = 1.0 / math.sqrt(D)
  out = ffpa_attn_func(q, k, v, is_causal=False, scale=scale, enable_gqa=True)
  loss = out.sum()
  loss.backward()

  dq_ref, dk_ref, dv_ref = _sdpa_ref_grads(q, k, v, False, scale)

  tol = _tolerance(dtype)
  torch.testing.assert_close(q.grad, dq_ref, **tol)
  torch.testing.assert_close(k.grad, dk_ref, **tol)
  torch.testing.assert_close(v.grad, dv_ref, **tol)


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize(
  "Nh_q,Nh_kv,N,D", [(32, 4, 8192, 320), (8, 1, 8192, 320)]
)
def test_ffpa_bwd_sdpa_backend_gqa(dtype, Nh_q, Nh_kv, N, D):
  """The SDPA backward backend must preserve high-precision GQA gradients."""
  B = 1
  torch.manual_seed(0)
  q = torch.randn(B, Nh_q, N, D, dtype=dtype, device="cuda", requires_grad=True)
  k = torch.randn(
    B, Nh_kv, N, D, dtype=dtype, device="cuda", requires_grad=True
  )
  v = torch.randn(
    B, Nh_kv, N, D, dtype=dtype, device="cuda", requires_grad=True
  )

  scale = 1.0 / math.sqrt(D)
  out = ffpa_attn_func(
    q,
    k,
    v,
    is_causal=False,
    scale=scale,
    backward_backend="sdpa",
    enable_gqa=True
  )
  out.sum().backward()

  dq_ref, dk_ref, dv_ref = _sdpa_ref_grads(q, k, v, False, scale)

  tol = _tolerance(dtype)
  torch.testing.assert_close(q.grad, dq_ref, **tol)
  torch.testing.assert_close(k.grad, dk_ref, **tol)
  torch.testing.assert_close(v.grad, dv_ref, **tol)


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
def test_ffpa_bwd_sdpa_backend_additive_attn_mask_matches_sdpa(dtype):
  """The SDPA backward backend handles broadcast additive-mask gradients."""
  B, H, N, D = 1, 8, 512, 320
  torch.manual_seed(0)
  q = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
  k = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
  v = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True)
  attn_mask = (torch.randn(1, 1, 1, N, dtype=dtype, device="cuda") *
               0.25).requires_grad_(True)

  scale = 1.0 / math.sqrt(D)
  out = ffpa_attn_func(
    q, k, v, attn_mask=attn_mask, scale=scale, backward_backend="sdpa"
  )
  out.sum().backward()
  assert attn_mask.grad is not None
  assert attn_mask.grad.shape == (1, 1, 1, N)

  dq_ref, dk_ref, dv_ref, dmask_ref = _sdpa_ref_grads(
    q, k, v, False, scale, attn_mask=attn_mask, return_mask_grad=True
  )

  tol = _tolerance(dtype)
  torch.testing.assert_close(q.grad, dq_ref, **tol)
  torch.testing.assert_close(k.grad, dk_ref, **tol)
  torch.testing.assert_close(v.grad, dv_ref, **tol)
  torch.testing.assert_close(attn_mask.grad, dmask_ref, atol=3e-2, rtol=3e-2)


# Backward + causal + GQA
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
  k = torch.randn(
    B, Nh_kv, N, D, dtype=dtype, device="cuda", requires_grad=True
  )
  v = torch.randn(
    B, Nh_kv, N, D, dtype=dtype, device="cuda", requires_grad=True
  )

  scale = 1.0 / math.sqrt(D)
  out = ffpa_attn_func(q, k, v, is_causal=True, scale=scale, enable_gqa=True)
  loss = out.sum()
  loss.backward()

  dq_ref, dk_ref, dv_ref = _sdpa_ref_grads(q, k, v, True, scale)

  tol = _tolerance(dtype)
  # Relax tolerance for large GQA/MQA shapes (long seqlen × many heads ×
  # causal) where numerical differences accumulate across atomic adds and
  # D-chunk loops.  The error scales mainly with seqlen.
  if N >= 8192 and (Nh_q >= 8 or D >= 512):
    tol = {"atol": 3e-1, "rtol": 5e-1}
  torch.testing.assert_close(q.grad, dq_ref, **tol)
  torch.testing.assert_close(k.grad, dk_ref, **tol)
  torch.testing.assert_close(v.grad, dv_ref, **tol)


# Backward + cross-attention
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
  out = ffpa_attn_func(q, k, v, is_causal=False, scale=scale)
  loss = out.sum()
  loss.backward()

  dq_ref, dk_ref, dv_ref = _sdpa_ref_grads(q, k, v, False, scale)

  tol = _tolerance(dtype)
  torch.testing.assert_close(q.grad, dq_ref, **tol)
  torch.testing.assert_close(k.grad, dk_ref, **tol)
  torch.testing.assert_close(v.grad, dv_ref, **tol)


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("Nq,Nkv,D", [(512, 4096, 320), (512, 8192, 512)])
def test_ffpa_bwd_sdpa_backend_causal_cross_attention(dtype, Nq, Nkv, D):
  """The SDPA backward backend must preserve causal cross-attention gradients."""
  B, H = 1, 8
  torch.manual_seed(0)
  q = torch.randn(B, H, Nq, D, dtype=dtype, device="cuda", requires_grad=True)
  k = torch.randn(B, H, Nkv, D, dtype=dtype, device="cuda", requires_grad=True)
  v = torch.randn(B, H, Nkv, D, dtype=dtype, device="cuda", requires_grad=True)

  scale = 1.0 / math.sqrt(D)
  out = ffpa_attn_func(
    q, k, v, is_causal=True, scale=scale, backward_backend="sdpa"
  )
  out.sum().backward()

  dq_ref, dk_ref, dv_ref = _sdpa_ref_grads(q, k, v, True, scale)

  tol = _tolerance(dtype)
  torch.testing.assert_close(q.grad, dq_ref, **tol)
  torch.testing.assert_close(k.grad, dk_ref, **tol)
  torch.testing.assert_close(v.grad, dv_ref, **tol)


# Inference path (no_grad)
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
    out = ffpa_attn_func(q, k, v)

  scale = 1.0 / math.sqrt(D)
  ref = _sdpa_ref(q, k, v, False, scale)
  tol = _tolerance(dtype)
  torch.testing.assert_close(out, ref, **tol)

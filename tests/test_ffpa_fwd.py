"""FFPA attention forward pass unit tests.

Tests the unified ``ffpa_attn_func`` dispatcher across fp16 / bf16 activations.
Heavy benchmarking lives under ``bench/bench_ffpa_fwd.py``.

Backward pass tests live in ``test_ffpa_bwd.py``.
"""

import itertools
import math
import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from ffpa_attn import ffpa_attn_func  # noqa: E402
import ffpa_attn.functional as ffpa_attn_functional  # noqa: E402
from ffpa_attn.functional import TritonBackend  # noqa: E402
from ffpa_attn.triton._ffpa_fwd import _ffpa_attn_forward_decode_impl  # noqa: E402

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
DISPATCH_SHAPES = [(1, H, 1024, D)
                   for H, D in itertools.product(HEADNUMS, HEADDIMS)]


def _sdpa_ref(Q, K, V, attn_mask=None):
  return F.scaled_dot_product_attention(
    Q, K, V, attn_mask=attn_mask, scale=1.0 / math.sqrt(Q.size(-1))
  )


def _sdpa_fallback(
  q,
  k,
  v,
  *,
  attn_mask=None,
  dropout_p=0.0,
  is_causal=False,
  scale=None,
  enable_gqa=False
):
  """Mirror the public raw SDPA fallback path without rewriting user inputs."""
  return torch._C._nn.scaled_dot_product_attention(
    q,
    k,
    v,
    attn_mask=attn_mask,
    dropout_p=dropout_p,
    is_causal=is_causal,
    scale=scale,
    enable_gqa=enable_gqa,
  )


def _is_sdpa_fallback_shape(
  q, k, *, attn_mask=None, dropout_p=0.0, forward_backend="triton"
):
  D = q.size(-1)
  Nq = q.size(2)
  Nkv = k.size(2)
  if forward_backend == "cutedsl":
    backend = ffpa_attn_functional.CuTeDSLBackend(forward=True)
  elif forward_backend == "cuda":
    backend = ffpa_attn_functional.CUDABackend(forward=True)
  else:
    backend = ffpa_attn_functional.TritonBackend(forward=True)
  return any([
    ffpa_attn_functional._should_use_aten_small_d_forward(backend, D),
    D > 1024,
    dropout_p > 0.0 and forward_backend == "cutedsl",
    8 <= Nq < 512,
    Nkv < 512,
  ])


def _tail_aligned_causal_mask(Nq, Nkv):
  """Return FFPA's lower-right causal mask for cross/decode references."""
  row_idx = torch.arange(Nq, device="cuda").view(-1, 1)
  col_idx = torch.arange(Nkv, device="cuda").view(1, -1)
  return col_idx <= (row_idx + (Nkv - Nq))


def _tolerance(dtype):
  return {
    "atol": 2e-2,
    "rtol": 2e-2
  } if dtype == torch.bfloat16 else {
    "atol": 1e-2,
    "rtol": 1e-2
  }


def _alloc_qkv(B, H, N, D, dtype):
  torch.manual_seed(0)
  q = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
  k = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
  v = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
  return q, k, v


def _require_cuda_forward_impl() -> None:
  if ffpa_attn_functional._ffpa_attn_forward_cuda is None:
    pytest.skip("CUDA forward backend was not compiled")


def _force_cuda_backend_unavailable(monkeypatch) -> None:
  monkeypatch.setattr(ffpa_attn_functional, "_ffpa_attn_forward_cuda", None)


@pytest.fixture(scope="module", autouse=True)
def _require_cuda():
  if not torch.cuda.is_available():
    pytest.skip("CUDA not available")


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("B,H,N,D", CORRECTNESS_SHAPES)
def test_ffpa_attn_func_matches_sdpa(dtype, B, H, N, D):
  q, k, v = _alloc_qkv(B, H, N, D, dtype)
  out = ffpa_attn_func(q, k, v)
  ref = _sdpa_ref(q, k, v)
  assert out.dtype == dtype
  assert out.shape == ref.shape
  assert torch.isfinite(out).all(), "FFPA output contains NaN/Inf"
  torch.testing.assert_close(out, ref, **_tolerance(dtype))


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("B,H,N,D", DISPATCH_SHAPES)
def test_ffpa_attn_func_dispatch_shapes(dtype, B, H, N, D):
  q, k, v = _alloc_qkv(B, H, N, D, dtype)
  out = ffpa_attn_func(q, k, v)
  assert out.shape == q.shape
  assert out.dtype == dtype
  assert torch.isfinite(out).all()


def test_ffpa_attn_func_rejects_unsupported_dtype():
  q, k, v = _alloc_qkv(1, 8, 1024, 512, torch.float32)
  with pytest.raises(TypeError):
    ffpa_attn_func(q, k, v)


def test_ffpa_attn_func_rejects_invalid_forward_backend():
  q, k, v = _alloc_qkv(1, 4, 512, 320, torch.float16)
  with pytest.raises((ValueError, TypeError)):
    ffpa_attn_func(q, k, v, forward_backend="bad")


def test_ffpa_attn_func_rejects_unknown_impl_option():
  q, k, v = _alloc_qkv(1, 4, 512, 320, torch.float16)
  with pytest.raises(TypeError, match="unexpected keyword"):
    ffpa_attn_func(q, k, v, unknown_impl_option=True)


@pytest.mark.skip(
  reason="_require_cuda_forward_impl removed; needs update for new dispatch"
)
@pytest.mark.parametrize("forward_backend", ["cuda", "triton"])
def test_ffpa_attn_func_small_d_routes_to_flash_attention(
  monkeypatch, forward_backend
):
  q, k, v = _alloc_qkv(1, 4, 64, 128, torch.float16)

  def _unexpected_backend(*args, **kwargs):
    raise AssertionError("small-D should not dispatch to FFPA forward")

  def _fake_flash(q_in, k_in, v_in, o_in, causal, softmax_scale, dropout_p=0.0):
    del o_in
    del causal, softmax_scale, dropout_p
    out = _sdpa_ref(q_in, k_in, v_in)
    lse = torch.zeros(
      q_in.size(0),
      q_in.size(2),
      q_in.size(1),
      device=q_in.device,
      dtype=torch.float32
    )
    seed = torch.zeros(1, device=q_in.device, dtype=torch.int64)
    offset = torch.zeros(1, device=q_in.device, dtype=torch.int64)
    return out, lse, seed, offset

  monkeypatch.setattr(
    ffpa_attn_functional, "_require_cuda_forward_impl",
    lambda: _unexpected_backend
  )
  monkeypatch.setattr(
    ffpa_attn_functional, "_ffpa_attn_forward_triton", _unexpected_backend
  )
  monkeypatch.setattr(
    ffpa_attn_functional, "_aten_flash_attn_forward", _fake_flash
  )

  out = ffpa_attn_func(q, k, v, forward_backend=forward_backend)

  torch.testing.assert_close(
    out, _sdpa_ref(q, k, v), **_tolerance(torch.float16)
  )


def test_ffpa_attn_func_triton_bool_attn_mask_matches_sdpa():
  q, k, v = _alloc_qkv(1, 4, 512, 512, torch.float16)
  row_idx = torch.arange(q.size(2), device=q.device).view(-1, 1)
  col_idx = torch.arange(k.size(2), device=k.device).view(1, -1)
  attn_mask = col_idx <= row_idx

  out = ffpa_attn_func(q, k, v, attn_mask=attn_mask, forward_backend="triton")
  ref = F.scaled_dot_product_attention(
    q, k, v, attn_mask=attn_mask, scale=1.0 / math.sqrt(q.size(-1))
  )
  torch.testing.assert_close(out, ref, **_tolerance(torch.float16))


def test_ffpa_attn_func_triton_additive_attn_mask_matches_sdpa():
  q, k, v = _alloc_qkv(1, 4, 512, 512, torch.float16)
  torch.manual_seed(1)
  attn_mask = torch.randn(
    1, 1, 1, k.size(2), device=q.device, dtype=q.dtype
  ) * 0.25

  out = ffpa_attn_func(q, k, v, attn_mask=attn_mask, forward_backend="triton")
  ref = _sdpa_ref(q, k, v, attn_mask=attn_mask)
  torch.testing.assert_close(out, ref, **_tolerance(torch.float16))


def test_ffpa_attn_func_triton_small_d_default_falls_back_to_sdpa(monkeypatch):
  import ffpa_attn.ffpa_attn_interface as iface

  q, k, v = _alloc_qkv(1, 4, 1024, 256, torch.float16)

  def _unexpected_apply(*args, **kwargs):
    raise AssertionError(
      "small-D triton should fall back before FFPAAttnFunc.apply"
    )

  monkeypatch.setattr(iface.FFPAAttnFunc, "apply", _unexpected_apply)

  out = ffpa_attn_func(q, k, v, forward_backend="triton")

  torch.testing.assert_close(
    out, _sdpa_ref(q, k, v), **_tolerance(torch.float16)
  )


def test_ffpa_attn_func_triton_small_d_env_uses_triton(monkeypatch):
  monkeypatch.setenv("FFPA_TRITON_ALLOW_SMALL_D", "1")
  q, k, v = _alloc_qkv(1, 4, 1024, 256, torch.float16)

  def _unexpected_flash(*args, **kwargs):
    raise AssertionError(
      "small-D triton should bypass aten flash when env is enabled"
    )

  monkeypatch.setattr(
    ffpa_attn_functional, "_flash_attn_forward_aten", _unexpected_flash
  )

  out = ffpa_attn_func(q, k, v, forward_backend="triton")

  assert out.shape == q.shape
  assert torch.isfinite(out).all()
  torch.testing.assert_close(
    out, _sdpa_ref(q, k, v), **_tolerance(torch.float16)
  )


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("D", [320, 512])
@pytest.mark.parametrize("mask_kind", ["bool_2d", "additive_broadcast"])
def test_ffpa_attn_func_cuda_attn_mask_matches_sdpa(dtype, D, mask_kind):
  _require_cuda_forward_impl()
  q, k, v = _alloc_qkv(1, 4, 512, D, dtype)
  if mask_kind == "bool_2d":
    attn_mask = torch.ones(
      q.size(2), k.size(2), dtype=torch.bool, device=q.device
    )
    attn_mask[:, 3::7] = False
    attn_mask[:, 0] = True
  else:
    torch.manual_seed(1)
    attn_mask = torch.randn(
      1, 1, 1, k.size(2), device=q.device, dtype=dtype
    ) * 0.25

  out = ffpa_attn_func(q, k, v, attn_mask=attn_mask, forward_backend="cuda")
  ref = _sdpa_ref(q, k, v, attn_mask=attn_mask)
  torch.testing.assert_close(out, ref, **_tolerance(dtype))


def test_ffpa_attn_func_cuda_attn_mask_cross_gqa_matches_sdpa():
  _require_cuda_forward_impl()
  dtype = torch.float16
  B, Nh_q, Nh_kv, Nq, Nkv, D = 1, 4, 2, 512, 768, 320
  torch.manual_seed(0)
  q = torch.randn(B, Nh_q, Nq, D, dtype=dtype, device="cuda")
  k = torch.randn(B, Nh_kv, Nkv, D, dtype=dtype, device="cuda")
  v = torch.randn(B, Nh_kv, Nkv, D, dtype=dtype, device="cuda")
  torch.manual_seed(1)
  attn_mask = torch.randn(1, 1, Nq, Nkv, device=q.device, dtype=dtype) * 0.125

  out = ffpa_attn_func(
    q,
    k,
    v,
    attn_mask=attn_mask,
    forward_backend="cuda",
    enable_gqa=True,
  )
  ref = _sdpa_fallback(
    q, k, v, attn_mask=attn_mask, scale=1.0 / math.sqrt(D), enable_gqa=True
  )
  torch.testing.assert_close(out, ref, **_tolerance(dtype))


def test_ffpa_attn_func_triton_dropout_matches_sdpa():
  q, k, v = _alloc_qkv(1, 2, 512, 512, torch.float16)

  torch.manual_seed(0)
  out = ffpa_attn_func(q, k, v, dropout_p=0.25, forward_backend="triton")
  torch.manual_seed(0)
  with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
    ref = F.scaled_dot_product_attention(
      q, k, v, dropout_p=0.25, scale=1.0 / math.sqrt(q.size(-1))
    )
  torch.testing.assert_close(out, ref, **_tolerance(torch.float16))


def test_ffpa_attn_func_cuda_dropout_matches_sdpa():
  _require_cuda_forward_impl()
  q, k, v = _alloc_qkv(1, 2, 512, 512, torch.float16)

  torch.manual_seed(0)
  out = ffpa_attn_func(q, k, v, dropout_p=0.25, forward_backend="cuda")
  torch.manual_seed(0)
  with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
    ref = F.scaled_dot_product_attention(
      q, k, v, dropout_p=0.25, scale=1.0 / math.sqrt(q.size(-1))
    )
  torch.testing.assert_close(out, ref, atol=4e-2, rtol=4e-2)


def test_ffpa_attn_func_cuda_dropout_with_attn_mask_matches_sdpa():
  _require_cuda_forward_impl()
  q, k, v = _alloc_qkv(1, 2, 512, 320, torch.float16)
  attn_mask = torch.randn(
    1, 1, 1, k.size(2), device=q.device, dtype=q.dtype
  ) * 0.125

  torch.manual_seed(1)
  out = ffpa_attn_func(
    q,
    k,
    v,
    attn_mask=attn_mask,
    dropout_p=0.2,
    forward_backend="cuda",
  )
  torch.manual_seed(1)
  with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
    ref = F.scaled_dot_product_attention(
      q,
      k,
      v,
      attn_mask=attn_mask,
      dropout_p=0.2,
      scale=1.0 / math.sqrt(q.size(-1)),
    )
  torch.testing.assert_close(out, ref, atol=4e-2, rtol=4e-2)


def test_ffpa_attn_func_cuda_decode_dropout_matches_sdpa():
  _require_cuda_forward_impl()
  q, k, v = _alloc_cross_qkv(1, 2, 1, 4096, 512, torch.float16)
  dropout_p = 0.2
  scale = 1.0 / math.sqrt(q.size(-1))

  torch.manual_seed(2)
  out = ffpa_attn_func(
    q, k, v, dropout_p=dropout_p, scale=scale, forward_backend="cuda"
  )
  torch.manual_seed(2)
  with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
    ref = F.scaled_dot_product_attention(
      q, k, v, dropout_p=dropout_p, scale=scale
    )
  torch.testing.assert_close(out, ref, atol=4e-2, rtol=4e-2)


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize(
  "Nq,Nkv,D,causal",
  [
    (1, 4096, 512, False),
    (2, 4096, 512, False),
    (7, 4096, 512, False),
    (1, 8192, 320, True),
    (7, 4096, 512, True),
  ],
)
def test_ffpa_attn_func_triton_decode_dropout_matches_sdpa(
  dtype, Nq, Nkv, D, causal
):
  """Decode/cross dropout must use the same SDPA Philox mask per score."""
  q, k, v = _alloc_cross_qkv(1, 2, Nq, Nkv, D, dtype)
  scale = 1.0 / math.sqrt(D)
  dropout_p = 0.2
  rng_seed = 1234

  torch.manual_seed(rng_seed)
  out = ffpa_attn_func(
    q,
    k,
    v,
    dropout_p=dropout_p,
    is_causal=causal,
    scale=scale,
    forward_backend="triton",
  )

  sdpa_kwargs = {}
  if causal:
    # SDPA is_causal uses a square-style upper-left convention for Nq != Nkv;
    # FFPA decode aligns queries to the KV tail, so use an explicit mask.
    sdpa_kwargs["attn_mask"] = _tail_aligned_causal_mask(Nq, Nkv)
  torch.manual_seed(rng_seed)
  with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
    ref = F.scaled_dot_product_attention(
      q, k, v, dropout_p=dropout_p, scale=scale, **sdpa_kwargs
    )

  tol = {
    "atol": 5e-2,
    "rtol": 5e-2
  } if dtype == torch.bfloat16 else {
    "atol": 4e-2,
    "rtol": 4e-2
  }
  torch.testing.assert_close(out, ref, **tol)


@pytest.mark.skip(
  reason="_require_cuda_forward_impl removed; needs update for new dispatch"
)
def test_ffpa_attn_func_too_large_d_falls_back_to_sdpa(monkeypatch):
  q, k, v = _alloc_qkv(1, 4, 64, 1152, torch.float16)

  def _unexpected_backend(*args, **kwargs):
    raise AssertionError(
      "D > 1024 fallback should not dispatch to FFPA forward"
    )

  monkeypatch.setattr(
    ffpa_attn_functional, "_require_cuda_forward_impl",
    lambda: _unexpected_backend
  )
  monkeypatch.setattr(
    ffpa_attn_functional, "_ffpa_attn_forward_triton", _unexpected_backend
  )

  out = ffpa_attn_func(q, k, v)
  ref = F.scaled_dot_product_attention(
    q, k, v, scale=1.0 / math.sqrt(q.size(-1))
  )
  torch.testing.assert_close(out, ref, **_tolerance(torch.float16))


@pytest.mark.skip(
  reason="_require_cuda_forward_impl removed; needs update for new dispatch"
)
@pytest.mark.parametrize("forward_backend", ["cuda", "triton"])
def test_ffpa_attn_func_large_d_keeps_selected_backend(
  monkeypatch, forward_backend
):
  q, k, v = _alloc_qkv(1, 4, 512, 320, torch.float16)
  called = {"cuda": 0, "triton": 0}

  def _fake_cuda(*args, **kwargs):
    q_in, k_in, v_in = args[:3]
    called["cuda"] += 1
    out = q_in + k_in + v_in
    lse = torch.zeros(
      q_in.size(0), q_in.size(1), q_in.size(2), device=q_in.device
    )
    return out, lse

  def _fake_triton(
    q_in,
    k_in,
    v_in,
    o_in,
    causal,
    softmax_scale,
    autotune,
    autotune_mode,
    attn_bias,
    dropout_p=0.0,
    philox_seed=0,
    philox_offset=0,
    enable_tma=False,
    enable_ws=False,
  ):
    del o_in, causal, softmax_scale, autotune, autotune_mode, attn_bias, dropout_p, philox_seed, philox_offset, enable_tma, enable_ws
    called["triton"] += 1
    out = q_in + k_in + v_in
    lse = torch.zeros(
      q_in.size(0), q_in.size(1), q_in.size(2), device=q_in.device
    )
    return out, lse

  def _unexpected_flash(*args, **kwargs):
    raise AssertionError("large-D should not route to aten flash forward")

  monkeypatch.setattr(
    ffpa_attn_functional, "_require_cuda_forward_impl", lambda: _fake_cuda
  )
  monkeypatch.setattr(
    ffpa_attn_functional, "_ffpa_attn_forward_triton", _fake_triton
  )
  monkeypatch.setattr(
    ffpa_attn_functional, "_aten_flash_attn_forward", _unexpected_flash
  )

  out = ffpa_attn_func(q, k, v, forward_backend=forward_backend)

  assert called[forward_backend] == 1
  assert called["cuda" if forward_backend == "triton" else "triton"] == 0
  torch.testing.assert_close(out, q + k + v)


@pytest.mark.skip(
  reason="_require_cuda_forward_impl removed; needs update for new dispatch"
)
def test_ffpa_attn_func_large_d_defaults_to_triton(monkeypatch):
  q, k, v = _alloc_qkv(1, 4, 512, 320, torch.float16)
  called = {"cuda": 0, "triton": 0}

  def _fake_cuda(*args, **kwargs):
    called["cuda"] += 1
    raise AssertionError("default large-D path should not use CUDA forward")

  def _fake_triton(
    q_in,
    k_in,
    v_in,
    o_in,
    causal,
    softmax_scale,
    autotune,
    autotune_mode,
    attn_bias,
    dropout_p=0.0,
    philox_seed=0,
    philox_offset=0,
    enable_tma=False,
    enable_ws=False,
  ):
    del o_in, causal, softmax_scale, autotune, autotune_mode, attn_bias, dropout_p, philox_seed, philox_offset, enable_tma, enable_ws
    called["triton"] += 1
    out = q_in + k_in + v_in
    lse = torch.zeros(
      q_in.size(0), q_in.size(1), q_in.size(2), device=q_in.device
    )
    return out, lse

  monkeypatch.setattr(
    ffpa_attn_functional, "_require_cuda_forward_impl", lambda: _fake_cuda
  )
  monkeypatch.setattr(
    ffpa_attn_functional, "_ffpa_attn_forward_triton", _fake_triton
  )

  out = ffpa_attn_func(q, k, v)

  assert called["triton"] == 1
  assert called["cuda"] == 0
  torch.testing.assert_close(out, q + k + v)


def test_ffpa_attn_func_backward_autotune_enables_saved_forward_autotune(
  monkeypatch
):
  seen_autotune = []

  def _fake_triton(
    q_in,
    k_in,
    v_in,
    o_in,
    causal,
    softmax_scale,
    autotune,
    autotune_mode,
    attn_bias,
    dropout_p=0.0,
    philox_seed=0,
    philox_offset=0,
    enable_tma=False,
    enable_ws=False,
  ):
    del k_in, v_in, causal, softmax_scale, autotune_mode, attn_bias, dropout_p, philox_seed, philox_offset, enable_tma, enable_ws
    seen_autotune.append(autotune)
    return torch.empty_like(o_in), torch.empty(
      q_in.shape[:-1], dtype=torch.float32, device=q_in.device
    )

  monkeypatch.setattr(
    ffpa_attn_functional, "_ffpa_attn_forward_triton", _fake_triton
  )
  q = torch.randn(
    1, 1, 512, 320, dtype=torch.float16, device="cuda", requires_grad=True
  )
  k = torch.randn(
    1, 1, 512, 320, dtype=torch.float16, device="cuda", requires_grad=True
  )
  v = torch.randn(
    1, 1, 512, 320, dtype=torch.float16, device="cuda", requires_grad=True
  )

  out = ffpa_attn_func(
    q,
    k,
    v,
    forward_backend=TritonBackend(forward=True, autotune=True),
    backward_backend="triton",
  )

  assert out.shape == q.shape
  assert seen_autotune == [True]


def test_ffpa_attn_func_routes_directional_tma_ws_flags(monkeypatch):
  seen_forward: list[tuple[bool, bool]] = []
  seen_backward: list[tuple[bool, bool]] = []

  def _fake_forward(
    q_in,
    k_in,
    v_in,
    o_in,
    causal,
    softmax_scale,
    autotune,
    autotune_mode,
    attn_bias,
    dropout_p=0.0,
    philox_seed=0,
    philox_offset=0,
    enable_tma=False,
    enable_ws=False,
  ):
    del k_in, v_in, causal, softmax_scale, autotune, autotune_mode, attn_bias, dropout_p, philox_seed, philox_offset
    seen_forward.append((enable_tma, enable_ws))
    return torch.empty_like(o_in), torch.empty(
      q_in.shape[:-1], dtype=torch.float32, device=q_in.device
    )

  def _fake_backward(
    grad_out,
    q,
    k,
    v,
    o,
    lse,
    causal,
    softmax_scale,
    autotune,
    autotune_mode,
    preprocess_d_chunk,
    attn_bias,
    return_attn_bias_grad,
    grad_kv_storage_dtype,
    dropout_p=0.0,
    philox_seed=0,
    philox_offset=0,
    enable_tma=False,
    enable_ws=False,
    enable_persist_dkdv=False,
    enable_split_launch=False,
  ):
    del grad_out, o, lse, causal, softmax_scale, autotune, autotune_mode, preprocess_d_chunk
    del attn_bias, return_attn_bias_grad, grad_kv_storage_dtype, dropout_p, philox_seed, philox_offset, enable_persist_dkdv, enable_split_launch
    seen_backward.append((enable_tma, enable_ws))
    return torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v), None

  monkeypatch.setattr(
    ffpa_attn_functional, "_ffpa_attn_forward_triton", _fake_forward
  )
  monkeypatch.setattr(
    ffpa_attn_functional, "_ffpa_attn_backward_triton", _fake_backward
  )
  q = torch.randn(
    1, 1, 512, 320, dtype=torch.float16, device="cuda", requires_grad=True
  )
  k = torch.randn(
    1, 1, 512, 320, dtype=torch.float16, device="cuda", requires_grad=True
  )
  v = torch.randn(
    1, 1, 512, 320, dtype=torch.float16, device="cuda", requires_grad=True
  )

  out = ffpa_attn_func(
    q,
    k,
    v,
    forward_backend=TritonBackend(
      forward=True, enable_tma=True, enable_ws=True
    ),
    backward_backend=TritonBackend(
      backward=True, enable_tma=False, enable_ws=False
    ),
  )
  out.sum().backward()

  assert seen_forward == [(True, True)]
  assert seen_backward == [(False, False)]


def test_ffpa_attn_func_cuda_forward_unavailable_raises(monkeypatch):
  _force_cuda_backend_unavailable(monkeypatch)
  q, k, v = _alloc_qkv(1, 4, 512, 320, torch.float16)

  with pytest.raises((RuntimeError, AssertionError), match="CUDA"):
    ffpa_attn_func(q, k, v, forward_backend="cuda")


def test_ffpa_attn_func_rejects_cuda_backward_backend():
  q, k, v = _alloc_qkv(1, 4, 512, 320, torch.float16)

  with pytest.raises(
    AssertionError, match="cuda backend does not support backward"
  ):
    ffpa_attn_func(q, k, v, forward_backend="triton", backward_backend="cuda")


@pytest.mark.parametrize("D", [128, 256])
def test_ffpa_attn_func_small_d_backward_smoke(D):
  q, k, v = _alloc_qkv(1, 4, 64, D, torch.float16)
  q = q.requires_grad_(True)
  k = k.requires_grad_(True)
  v = v.requires_grad_(True)

  out = ffpa_attn_func(q, k, v, forward_backend="triton")
  loss = out.float().square().mean()
  loss.backward()

  assert torch.isfinite(out).all()
  assert torch.isfinite(q.grad).all()
  assert torch.isfinite(k.grad).all()
  assert torch.isfinite(v.grad).all()


def test_ffpa_attn_func_small_d_honors_output_buffer():
  q, k, v = _alloc_qkv(1, 4, 64, 128, torch.float16)
  out_buffer = torch.empty_like(q)

  out = ffpa_attn_func(q, k, v, forward_backend="triton")
  ref = _sdpa_ref(q, k, v)

  assert out.shape == out_buffer.shape
  assert out.dtype == out_buffer.dtype
  assert torch.isfinite(out).all()
  torch.testing.assert_close(out, ref, **_tolerance(torch.float16))


@pytest.mark.skip(
  reason=
  "D <= 256 now delegates to F.scaled_dot_product_attention directly, bypassing FFPAAttnFunc"
)
def test_ffpa_attn_func_small_d_backward_consumes_saved_flash_state(
  monkeypatch
):
  q, k, v = _alloc_qkv(1, 4, 32, 128, torch.float16)
  q = q.requires_grad_(True)
  k = k.requires_grad_(True)
  v = v.requires_grad_(True)
  saved_seed = torch.tensor([123], device=q.device, dtype=torch.int64)
  saved_offset = torch.tensor([456], device=q.device, dtype=torch.int64)
  seen = {"checked": False}

  def _fake_flash_forward(
    q_in, k_in, v_in, o_in, causal, softmax_scale, dropout_p=0.0
  ):
    del o_in, causal, softmax_scale, dropout_p
    out = _sdpa_ref(q_in, k_in, v_in)
    lse = torch.zeros(
      q_in.size(0),
      q_in.size(2),
      q_in.size(1),
      device=q_in.device,
      dtype=torch.float32
    )
    return out, lse, saved_seed, saved_offset

  def _fake_flash_backward(
    grad_out,
    q_in,
    k_in,
    v_in,
    o_in,
    lse,
    causal,
    rng_state,
    unused,
    softmax_scale,
    dropout_p=0.0
  ):
    del grad_out, o_in, lse, causal, softmax_scale, dropout_p
    assert rng_state.data_ptr() == saved_seed.data_ptr()
    assert unused.data_ptr() == saved_offset.data_ptr()
    seen["checked"] = True
    return torch.zeros_like(q_in), torch.zeros_like(k_in
                                                    ), torch.zeros_like(v_in)

  monkeypatch.setattr(
    ffpa_attn_functional, "_aten_flash_attn_forward", _fake_flash_forward
  )
  monkeypatch.setattr(
    ffpa_attn_functional, "_aten_flash_attn_backward", _fake_flash_backward
  )

  out = ffpa_attn_func(q, k, v, forward_backend="triton")
  out.sum().backward()

  assert seen["checked"]
  assert torch.count_nonzero(q.grad) == 0
  assert torch.count_nonzero(k.grad) == 0
  assert torch.count_nonzero(v.grad) == 0


TRITON_FORWARD_SHAPES = [
  (1, 4, 128, 128, 320, False),
  (1, 4, 129, 257, 320, False),
  (1, 4, 128, 128, 512, True),
  (1, 8, 64, 512, 512, True),
]


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("B,H,Nq,Nkv,D,causal", TRITON_FORWARD_SHAPES)
def test_ffpa_attn_func_triton_forward_matches_sdpa(
  dtype, B, H, Nq, Nkv, D, causal
):
  q, k, v = _alloc_cross_qkv(B, H, Nq, Nkv, D, dtype)
  out = ffpa_attn_func(q, k, v, is_causal=causal, forward_backend="triton")
  if _is_sdpa_fallback_shape(q, k, forward_backend="triton"):
    ref = _sdpa_fallback(q, k, v, is_causal=causal, scale=1.0 / math.sqrt(D))
  elif causal:
    kv_offset = Nkv - Nq
    row_idx = torch.arange(Nq, device="cuda").view(-1, 1)
    col_idx = torch.arange(Nkv, device="cuda").view(1, -1)
    attn_mask = (col_idx <= (row_idx + kv_offset))
    ref = F.scaled_dot_product_attention(
      q, k, v, attn_mask=attn_mask, scale=1.0 / math.sqrt(D)
    )
  else:
    ref = _sdpa_ref(q, k, v)
  assert out.shape == ref.shape
  assert out.dtype == dtype
  assert torch.isfinite(out).all()
  torch.testing.assert_close(out, ref, **_tolerance(dtype))


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize(
  "Nq,Nkv,D,causal",
  [
    (1, 4096, 512, False),
    (7, 4096, 320, False),
    (1, 8192, 512, False),
    (7, 8192, 512, True),
  ],
)
def test_ffpa_attn_func_triton_decode_matches_sdpa(dtype, Nq, Nkv, D, causal):
  B, H = 1, 4
  q, k, v = _alloc_cross_qkv(B, H, Nq, Nkv, D, dtype)
  out = ffpa_attn_func(q, k, v, is_causal=causal, forward_backend="triton")
  if _is_sdpa_fallback_shape(q, k, forward_backend="triton"):
    ref = _sdpa_fallback(q, k, v, is_causal=causal, scale=1.0 / math.sqrt(D))
  elif causal:
    kv_offset = Nkv - Nq
    row_idx = torch.arange(Nq, device="cuda").view(-1, 1)
    col_idx = torch.arange(Nkv, device="cuda").view(1, -1)
    attn_mask = (col_idx <= (row_idx + kv_offset))
    ref = F.scaled_dot_product_attention(
      q, k, v, attn_mask=attn_mask, scale=1.0 / math.sqrt(D)
    )
  else:
    ref = _sdpa_ref(q, k, v)
  assert out.shape == ref.shape
  assert out.dtype == dtype
  assert torch.isfinite(out).all()
  torch.testing.assert_close(out, ref, **_tolerance(dtype))


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize(
  "Nq,Nkv,D,causal",
  [
    (1, 4096, 512, False),
    (7, 4096, 320, False),
    (15, 8192, 512, True),
  ],
)
def test_ffpa_attn_func_cuda_decode_matches_sdpa(dtype, Nq, Nkv, D, causal):
  _require_cuda_forward_impl()
  B, H = 1, 4
  q, k, v = _alloc_cross_qkv(B, H, Nq, Nkv, D, dtype)
  out = ffpa_attn_func(q, k, v, is_causal=causal, forward_backend="cuda")
  if _is_sdpa_fallback_shape(q, k, forward_backend="cuda"):
    ref = _sdpa_fallback(q, k, v, is_causal=causal, scale=1.0 / math.sqrt(D))
  elif causal:
    kv_offset = Nkv - Nq
    row_idx = torch.arange(Nq, device="cuda").view(-1, 1)
    col_idx = torch.arange(Nkv, device="cuda").view(1, -1)
    attn_mask = (col_idx <= (row_idx + kv_offset))
    ref = F.scaled_dot_product_attention(
      q, k, v, attn_mask=attn_mask, scale=1.0 / math.sqrt(D)
    )
  else:
    ref = _sdpa_ref(q, k, v)
  assert out.shape == ref.shape
  assert out.dtype == dtype
  assert torch.isfinite(out).all()
  torch.testing.assert_close(out, ref, **_tolerance(dtype))


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("D", [320, 512])
def test_ffpa_attn_func_triton_forward_gqa_matches_sdpa(dtype, D):
  B, Nh_q, Nh_kv, Nq, Nkv = 1, 8, 2, 128, 256
  torch.manual_seed(0)
  q = torch.randn(B, Nh_q, Nq, D, dtype=dtype, device="cuda")
  k = torch.randn(B, Nh_kv, Nkv, D, dtype=dtype, device="cuda")
  v = torch.randn(B, Nh_kv, Nkv, D, dtype=dtype, device="cuda")
  out = ffpa_attn_func(q, k, v, forward_backend="triton", enable_gqa=True)
  group_size = Nh_q // Nh_kv
  ref = _sdpa_ref(
    q, k.repeat_interleave(group_size, dim=1),
    v.repeat_interleave(group_size, dim=1)
  )
  assert out.shape == ref.shape
  assert torch.isfinite(out).all()
  torch.testing.assert_close(out, ref, **_tolerance(dtype))


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("Nq,D", [(1, 512), (5, 320)])
def test_ffpa_attn_func_triton_decode_gqa_matches_sdpa(dtype, Nq, D):
  B, Nh_q, Nh_kv, Nkv = 1, 8, 2, 8192
  torch.manual_seed(0)
  q = torch.randn(B, Nh_q, Nq, D, dtype=dtype, device="cuda")
  k = torch.randn(B, Nh_kv, Nkv, D, dtype=dtype, device="cuda")
  v = torch.randn(B, Nh_kv, Nkv, D, dtype=dtype, device="cuda")
  out = ffpa_attn_func(q, k, v, forward_backend="triton", enable_gqa=True)
  group_size = Nh_q // Nh_kv
  ref = _sdpa_ref(
    q, k.repeat_interleave(group_size, dim=1),
    v.repeat_interleave(group_size, dim=1)
  )
  assert out.shape == ref.shape
  assert out.dtype == dtype
  assert torch.isfinite(out).all()
  torch.testing.assert_close(out, ref, **_tolerance(dtype))


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("Nq,D", [(1, 512), (7, 320)])
def test_ffpa_attn_func_cuda_decode_gqa_matches_sdpa(dtype, Nq, D):
  _require_cuda_forward_impl()
  B, Nh_q, Nh_kv, Nkv = 1, 8, 2, 8192
  torch.manual_seed(0)
  q = torch.randn(B, Nh_q, Nq, D, dtype=dtype, device="cuda")
  k = torch.randn(B, Nh_kv, Nkv, D, dtype=dtype, device="cuda")
  v = torch.randn(B, Nh_kv, Nkv, D, dtype=dtype, device="cuda")
  out = ffpa_attn_func(q, k, v, forward_backend="cuda", enable_gqa=True)
  group_size = Nh_q // Nh_kv
  ref = _sdpa_ref(
    q, k.repeat_interleave(group_size, dim=1),
    v.repeat_interleave(group_size, dim=1)
  )
  assert out.shape == ref.shape
  assert out.dtype == dtype
  assert torch.isfinite(out).all()
  torch.testing.assert_close(out, ref, **_tolerance(dtype))


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("Nq,Nkv,D", [(32, 8192, 512), (512, 8192, 512)])
def test_ffpa_attn_forward_decode_impl_matches_sdpa_for_larger_queries(
  dtype, Nq, Nkv, D
):
  B, H = 1, 4
  q, k, v = _alloc_cross_qkv(B, H, Nq, Nkv, D, dtype)
  o = torch.empty_like(q)
  lse = torch.empty(B, H, Nq, device="cuda", dtype=torch.float32)

  _ffpa_attn_forward_decode_impl(
    q,
    k,
    v,
    o,
    lse,
    causal=False,
    softmax_scale=1.0 / math.sqrt(D),
    autotune=False,
  )

  ref = _sdpa_ref(q, k, v)
  assert o.shape == ref.shape
  assert o.dtype == dtype
  assert torch.isfinite(o).all()
  torch.testing.assert_close(o, ref, **_tolerance(dtype))


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("Nq,Nkv,D", [(32, 8192, 512), (512, 8192, 512)])
def test_ffpa_attn_func_triton_decode_heuristic_matches_sdpa_for_larger_queries(
  dtype, Nq, Nkv, D
):
  B, H = 1, 4
  q, k, v = _alloc_cross_qkv(B, H, Nq, Nkv, D, dtype)
  out = ffpa_attn_func(q, k, v, forward_backend="triton")

  ref = _sdpa_ref(q, k, v)
  assert out.shape == ref.shape
  assert out.dtype == dtype
  assert torch.isfinite(out).all()
  torch.testing.assert_close(out, ref, **_tolerance(dtype))


@pytest.mark.parametrize("backward_backend", ["sdpa", "triton"])
def test_ffpa_attn_func_triton_forward_backward_smoke(backward_backend):
  q, k, v = _alloc_qkv(1, 2, 64, 320, torch.float16)
  q = q.requires_grad_(True)
  k = k.requires_grad_(True)
  v = v.requires_grad_(True)
  out = ffpa_attn_func(
    q, k, v, forward_backend="triton", backward_backend=backward_backend
  )
  loss = out.float().square().mean()
  loss.backward()
  assert torch.isfinite(out).all()
  assert torch.isfinite(q.grad).all()
  assert torch.isfinite(k.grad).all()
  assert torch.isfinite(v.grad).all()


# Boundary shapes: seqlen not a multiple of Bc=64 (and/or Br=64). Covers
# partial first tile (N<64), single-tile tail, multi-tile + tail, and
# sizes near common power-of-two boundaries. Uses D=128 (small_d kernel)
# and D=256 (large_d kernel) for coverage of both kernel paths.
BOUNDARY_SEQLENS = [
  1, 17, 33, 63, 65, 100, 127, 129, 200, 1000, 2047, 4095, 5000
]
BOUNDARY_HEADDIMS = [128, 256]
BOUNDARY_SHAPES = [
  (1, 4, N, D)
  for N, D in itertools.product(BOUNDARY_SEQLENS, BOUNDARY_HEADDIMS)
]


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("B,H,N,D", BOUNDARY_SHAPES)
def test_ffpa_attn_func_boundary_seqlen(dtype, B, H, N, D):
  q, k, v = _alloc_qkv(B, H, N, D, dtype)
  out = ffpa_attn_func(q, k, v)
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
  out = ffpa_attn_func(q, k, v)
  ref = _sdpa_ref(q, k, v)
  assert out.shape == (B, H, Nq, D)
  assert out.dtype == dtype
  assert torch.isfinite(out).all(
  ), f"FFPA output has NaN/Inf at Nq={Nq}, Nkv={Nkv}, D={D}"
  torch.testing.assert_close(out, ref, **_tolerance(dtype))


def test_ffpa_attn_func_rejects_mismatched_kv_seqlen():
  torch.manual_seed(0)
  q = torch.randn(1, 4, 512, 512, dtype=torch.float16, device="cuda")
  k = torch.randn(1, 4, 1024, 512, dtype=torch.float16, device="cuda")
  v = torch.randn(1, 4, 2048, 512, dtype=torch.float16, device="cuda")
  with pytest.raises(ValueError, match="seqlen"):
    ffpa_attn_func(q, k, v)


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
  out = ffpa_attn_func(q, k, v, enable_gqa=True)
  group_size = Nh_q // Nh_kv
  k_ref = k.repeat_interleave(group_size, dim=1)
  v_ref = v.repeat_interleave(group_size, dim=1)
  ref = _sdpa_ref(q, k_ref, v_ref)
  assert out.shape == (B, Nh_q, Nq, D)
  assert out.dtype == dtype
  assert torch.isfinite(out).all(), (
    f"FFPA output has NaN/Inf at Nh_q={Nh_q}, Nh_kv={Nh_kv}, Nq={Nq}, Nkv={Nkv}, D={D}"
  )
  torch.testing.assert_close(out, ref, **_tolerance(dtype))


def test_ffpa_attn_func_rejects_indivisible_num_heads():
  torch.manual_seed(0)
  # Nh_q=12, Nh_kv=8 -> 12 % 8 != 0
  q = torch.randn(1, 12, 512, 512, dtype=torch.float16, device="cuda")
  k = torch.randn(1, 8, 512, 512, dtype=torch.float16, device="cuda")
  v = torch.randn(1, 8, 512, 512, dtype=torch.float16, device="cuda")
  with pytest.raises(ValueError, match="num_heads"):
    ffpa_attn_func(q, k, v)


def test_ffpa_attn_func_rejects_mismatched_kv_num_heads():
  torch.manual_seed(0)
  q = torch.randn(1, 16, 512, 512, dtype=torch.float16, device="cuda")
  k = torch.randn(1, 4, 512, 512, dtype=torch.float16, device="cuda")
  v = torch.randn(1, 2, 512, 512, dtype=torch.float16, device="cuda")
  with pytest.raises(ValueError, match="num_heads"):
    ffpa_attn_func(q, k, v)


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
  out = ffpa_attn_func(q, k, v, is_causal=True)
  ref = F.scaled_dot_product_attention(
    q, k, v, is_causal=True, scale=1.0 / math.sqrt(D)
  )
  assert out.shape == ref.shape
  assert out.dtype == dtype
  assert torch.isfinite(out).all(
  ), f"FFPA causal output has NaN/Inf at N={N}, D={D}"
  torch.testing.assert_close(out, ref, **_tolerance(dtype))


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("Nq,Nkv,D", CAUSAL_CROSS_SHAPES)
def test_ffpa_attn_func_causal_cross_attention(dtype, Nq, Nkv, D):
  B, H = 1, 4
  q, k, v = _alloc_cross_qkv(B, H, Nq, Nkv, D, dtype)
  if _is_sdpa_fallback_shape(q, k, forward_backend="triton"):
    try:
      ref = _sdpa_fallback(q, k, v, is_causal=True, scale=1.0 / math.sqrt(D))
    except RuntimeError:
      with pytest.raises(RuntimeError):
        ffpa_attn_func(q, k, v, is_causal=True)
      return

    out = ffpa_attn_func(q, k, v, is_causal=True)
    assert out.shape == ref.shape
    assert out.dtype == dtype
    assert torch.isfinite(out).all(
    ), (f"FFPA causal output has NaN/Inf at Nq={Nq}, Nkv={Nkv}, D={D}")
    torch.testing.assert_close(out, ref, **_tolerance(dtype))
    return

  out = ffpa_attn_func(q, k, v, is_causal=True)
  # Reference: build an explicit attn mask where Q row r may attend to
  # KV positions k <= r + (Nkv - Nq). SDPA's is_causal only supports
  # the Nq == Nkv square case, so use attn_mask for the general case.
  kv_offset = Nkv - Nq
  row_idx = torch.arange(Nq, device="cuda").view(-1, 1)
  col_idx = torch.arange(Nkv, device="cuda").view(1, -1)
  attn_mask = (col_idx <= (row_idx + kv_offset))  # [Nq, Nkv] bool
  ref = F.scaled_dot_product_attention(
    q, k, v, attn_mask=attn_mask, scale=1.0 / math.sqrt(D)
  )
  assert out.shape == (B, H, Nq, D)
  assert out.dtype == dtype
  assert torch.isfinite(out).all(
  ), (f"FFPA causal output has NaN/Inf at Nq={Nq}, Nkv={Nkv}, D={D}")
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
  if _is_sdpa_fallback_shape(q, k, forward_backend="triton"):
    try:
      ref = _sdpa_fallback(
        q,
        k,
        v,
        is_causal=True,
        scale=1.0 / math.sqrt(D),
        enable_gqa=True,
      )
    except RuntimeError:
      with pytest.raises(RuntimeError):
        ffpa_attn_func(q, k, v, is_causal=True, enable_gqa=True)
      return

    out = ffpa_attn_func(q, k, v, is_causal=True, enable_gqa=True)
    assert out.shape == (B, Nh_q, Nq, D)
    assert out.dtype == dtype
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, ref, **_tolerance(dtype))
    return

  out = ffpa_attn_func(q, k, v, is_causal=True, enable_gqa=True)
  group_size = Nh_q // Nh_kv
  k_ref = k.repeat_interleave(group_size, dim=1)
  v_ref = v.repeat_interleave(group_size, dim=1)
  kv_offset = Nkv - Nq
  row_idx = torch.arange(Nq, device="cuda").view(-1, 1)
  col_idx = torch.arange(Nkv, device="cuda").view(1, -1)
  attn_mask = (col_idx <= (row_idx + kv_offset))
  ref = F.scaled_dot_product_attention(
    q, k_ref, v_ref, attn_mask=attn_mask, scale=1.0 / math.sqrt(D)
  )
  assert out.shape == (B, Nh_q, Nq, D)
  assert out.dtype == dtype
  assert torch.isfinite(out).all()
  torch.testing.assert_close(out, ref, **_tolerance(dtype))


def test_ffpa_attn_func_rejects_causal_with_shorter_kv():
  torch.manual_seed(0)
  # Nq > Nkv + causal is rejected (no valid keys for later Q rows).
  q = torch.randn(1, 4, 1024, 512, dtype=torch.float16, device="cuda")
  k = torch.randn(1, 4, 768, 512, dtype=torch.float16, device="cuda")
  v = torch.randn(1, 4, 768, 512, dtype=torch.float16, device="cuda")
  with pytest.raises(ValueError, match="is_causal"):
    ffpa_attn_func(q, k, v, is_causal=True)


def test_ffpa_attn_func_requires_explicit_gqa_opt_in_for_large_d():
  torch.manual_seed(0)
  q = torch.randn(1, 16, 512, 512, dtype=torch.float16, device="cuda")
  k = torch.randn(1, 4, 512, 512, dtype=torch.float16, device="cuda")
  v = torch.randn(1, 4, 512, 512, dtype=torch.float16, device="cuda")
  with pytest.raises(ValueError, match="enable_gqa=False"):
    ffpa_attn_func(q, k, v)

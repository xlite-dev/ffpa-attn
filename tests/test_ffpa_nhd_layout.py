"""NHD (diffusers BNHD) layout tests for the persist-D kernels.

Each family runs the same attention twice — BHND-packed in/out and NHD
packed in/out — and asserts the outputs agree bitwise (same kernel, only
the O store coordinates differ). The NHD tensors are materialized
NHD-packed ([B, N, H, D] contiguous) before the call, the documented
construction that avoids false NHD views. The fp8 section additionally
covers strided-NHD inputs (fused-QKV chunk views with row stride wider
than H*D) consumed zero-copy through the relaxed layout gate.
"""

import pytest
import torch

from ffpa_attn import ffpa_attn_func
from ffpa_attn.functional import CUDABackend

pytestmark = pytest.mark.skipif(
  not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12,
  reason="NHD persist-D store requires an sm_120 GPU",
)


def _backend(**kwargs) -> CUDABackend:
  return CUDABackend(
    backward=False, enable_tma=True, enable_cute=True, **kwargs
  )


def _mk(B, H, Hkv, N, D, dtype=torch.bfloat16):
  q = torch.randn(B, H, N, D, dtype=dtype, device="cuda") * 0.5
  k = torch.randn(B, Hkv, N, D, dtype=dtype, device="cuda") * 0.5
  v = torch.randn(B, Hkv, N, D, dtype=dtype, device="cuda") * 0.5
  return q, k, v


def _nhd(t):
  return t.permute(0, 2, 1, 3).contiguous()


@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
@pytest.mark.parametrize(
  "B,H,Hkv,N", [(1, 24, 24, 2048), (1, 24, 24, 16383), (2, 8, 8, 3000),
                (1, 24, 8, 4096)],
  ids=["full", "tail", "batch", "gqa"]
)
@pytest.mark.parametrize("D", [64, 128], ids=["d64", "d128"])
def test_fp16_nhd_layout_bit_exact(B, H, Hkv, N, D, causal, monkeypatch):
  monkeypatch.setenv("FFPA_CUDA_ALLOW_SMALL_D", "1")
  torch.manual_seed(0)
  q, k, v = _mk(B, H, Hkv, N, D)
  backend = _backend()
  backend_nhd = _backend(tensor_layout="NHD")

  with torch.no_grad():
    out_b = ffpa_attn_func(q, k, v, is_causal=causal, forward_backend=backend)
    out_n = ffpa_attn_func(
      _nhd(q), _nhd(k), _nhd(v), is_causal=causal, forward_backend=backend_nhd
    )
  assert out_n.shape == (B, N, H, D) and out_n.is_contiguous()
  torch.testing.assert_close(out_n, out_b.permute(0, 2, 1, 3))


@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
@pytest.mark.parametrize("pv", ["fp4", "fp8"], ids=["nvfp4", "mxfp8"])
@pytest.mark.parametrize("D", [128, 256], ids=["d128", "d256"])
def test_fp4_nhd_layout_bit_exact(D, pv, causal, monkeypatch):
  monkeypatch.setenv("FFPA_CUDA_ALLOW_SMALL_D", "1")
  if pv == "fp8" and D > 192:
    pytest.skip("fp4_pv_mm_type='fp8' (MXFP8) is smem-limited to D<=192")
  torch.manual_seed(0)
  B, H, Hkv, N = 1, 24, 24, 2048
  q, k, v = _mk(B, H, Hkv, N, D)
  backend = _backend(enable_fp4=True, fp4_pv_mm_type=pv, fp4_hybrid=False)
  backend_nhd = _backend(
    enable_fp4=True,
    fp4_pv_mm_type=pv,
    fp4_hybrid=False,
    tensor_layout="NHD",
  )

  with torch.no_grad():
    out_b = ffpa_attn_func(q, k, v, is_causal=causal, forward_backend=backend)
    out_n = ffpa_attn_func(
      _nhd(q), _nhd(k), _nhd(v), is_causal=causal, forward_backend=backend_nhd
    )
  assert out_n.shape == (B, N, H, D) and out_n.is_contiguous()
  torch.testing.assert_close(out_n, out_b.permute(0, 2, 1, 3))


def test_fp4_nhd_tail_batch_gqa(monkeypatch):
  monkeypatch.setenv("FFPA_CUDA_ALLOW_SMALL_D", "1")
  torch.manual_seed(0)
  B, H, Hkv, N, D = 2, 8, 4, 3000, 128
  q, k, v = _mk(B, H, Hkv, N, D)
  backend = _backend(enable_fp4=True, fp4_hybrid=False)
  backend_nhd = _backend(enable_fp4=True, fp4_hybrid=False, tensor_layout="NHD")

  with torch.no_grad():
    out_b = ffpa_attn_func(q, k, v, forward_backend=backend)
    out_n = ffpa_attn_func(
      _nhd(q), _nhd(k), _nhd(v), forward_backend=backend_nhd
    )
  assert out_n.shape == (B, N, H, D) and out_n.is_contiguous()
  torch.testing.assert_close(out_n, out_b.permute(0, 2, 1, 3))


@pytest.mark.parametrize("family", ["fp8", "fp4"], ids=["fp8", "fp4"])
@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
def test_hybrid_nhd_layout_bit_exact(family, causal, monkeypatch):
  monkeypatch.setenv("FFPA_CUDA_ALLOW_SMALL_D", "1")
  torch.manual_seed(0)
  B, H, Hkv, N, D = 1, 24, 24, 2048, 128
  q, k, v = _mk(B, H, Hkv, N, D)
  kw = (
    dict(enable_fp8=True, fp8_hybrid=True, fp8_hybrid_n_early=256)
    if family == "fp8" else
    dict(enable_fp4=True, fp4_hybrid=True, fp4_hybrid_n_early=256)
  )
  backend = _backend(**kw)
  backend_nhd = _backend(tensor_layout="NHD", **kw)
  with torch.no_grad():
    out_b = ffpa_attn_func(q, k, v, is_causal=causal, forward_backend=backend)
    out_n = ffpa_attn_func(
      _nhd(q), _nhd(k), _nhd(v), is_causal=causal, forward_backend=backend_nhd
    )
  assert out_n.shape == (B, N, H, D) and out_n.is_contiguous()
  torch.testing.assert_close(out_n, out_b.permute(0, 2, 1, 3))


@pytest.mark.parametrize(
  "family,D",
  [("fp8", 128), ("fp8", 120), ("fp4", 128), ("fp4", 192)],
  ids=["fp8-d128", "fp8-d120-pad", "fp4-d128-fused", "fp4-d192-wht"],
)
def test_hadamard_nhd_layout_bit_exact(family, D, monkeypatch):
  # hadamard only rotates Q/K (orthogonal, cancels inside QK^T), so NHD is
  # layout-only: the WHT kernel materializes BHND copies of NHD Q/K/V in
  # the launcher (fp4 pow2-D fuses the WHT into the quantize kernel and
  # stays zero-copy).
  monkeypatch.setenv("FFPA_CUDA_ALLOW_SMALL_D", "1")
  torch.manual_seed(0)
  B, H, Hkv, N = 1, 24, 24, 2048
  q, k, v = _mk(B, H, Hkv, N, D)
  kw = (
    dict(enable_fp8=True, fp8_hadamard=True)
    if family == "fp8" else dict(enable_fp4=True, fp4_hadamard=True)
  )
  backend = _backend(**kw)
  backend_nhd = _backend(tensor_layout="NHD", **kw)
  with torch.no_grad():
    out_b = ffpa_attn_func(q, k, v, forward_backend=backend)
    out_n = ffpa_attn_func(
      _nhd(q), _nhd(k), _nhd(v), forward_backend=backend_nhd
    )
  assert out_n.shape == (B, N, H, D) and out_n.is_contiguous()
  torch.testing.assert_close(out_n, out_b.permute(0, 2, 1, 3))


def test_nhd_layout_rejections(monkeypatch):
  monkeypatch.setenv("FFPA_CUDA_ALLOW_SMALL_D", "1")
  torch.manual_seed(0)
  B, H, N = 1, 24, 4096

  # fp16: %32 multiples stay on the CUTE_TMA fast path across the range
  # (persist-D <=128, split-D (32,64) %64 / (32,32) %32, M4N2 %64>=768);
  # 288 hits the (32,32) variant. Non-multiples beyond the persist-D pad
  # range (e.g. 280) have no kernel: the gate declines with TypeError.
  assert _backend().is_nhd_supported(256)
  assert _backend().is_nhd_supported(288)
  assert not _backend().is_nhd_supported(280)
  with pytest.raises(TypeError, match="NHD"):
    with torch.no_grad():
      q, k, v = _mk(B, H, H, N, 280)
      ffpa_attn_func(
        _nhd(q),
        _nhd(k),
        _nhd(v),
        forward_backend=_backend(tensor_layout="NHD"),
      )
  # Outside the persist-D range fp8/fp4 are now covered by
  # split-D/M4N2 (%64 chunk granularity); 288/280 (not %64) have no
  # kernel. Hybrid and hadamard no longer block (stride-generic
  # writeback / launcher-side BHND copies).
  assert _backend(enable_fp8=True).is_nhd_supported(256)
  assert not _backend(enable_fp8=True).is_nhd_supported(288)
  assert _backend(enable_fp4=True).is_nhd_supported(320)
  assert not _backend(enable_fp4=True).is_nhd_supported(280)
  assert _backend(enable_fp8=True, fp8_hybrid=True).is_nhd_supported(128)
  assert _backend(enable_fp4=True, fp4_hybrid=True).is_nhd_supported(128)
  assert _backend(enable_fp8=True, fp8_hadamard=True).is_nhd_supported(128)
  assert _backend(enable_fp4=True, fp4_hadamard=True).is_nhd_supported(128)
  # fp16 with the CUTE_TMA path opted out (enable_cute=False): every
  # remaining fp16 path stores through a static BHND descriptor.
  with pytest.raises(TypeError, match="NHD"):
    with torch.no_grad():
      q, k, v = _mk(B, H, H, N, 128)
      ffpa_attn_func(
        _nhd(q),
        _nhd(k),
        _nhd(v),
        forward_backend=CUDABackend(
          backward=False,
          enable_tma=True,
          enable_cute=False,
          tensor_layout="NHD",
        ),
      )
  # force_cute (enable_tma=False + enable_cute=True): the python gate
  # declines NHD (needs CUTE_TMA); the C++ backstop keeps rejecting it
  # too, so the sm80 cp.async path never silently writes a packed O.
  with pytest.raises(TypeError, match="NHD"):
    with torch.no_grad():
      q, k, v = _mk(B, H, H, N, 320)
      ffpa_attn_func(
        _nhd(q),
        _nhd(k),
        _nhd(v),
        forward_backend=CUDABackend(
          backward=False,
          enable_tma=False,
          enable_cute=True,
          tensor_layout="NHD",
        ),
      )
  # grad-on: the fast path declines, and the full chain assumes BHND.
  with pytest.raises(TypeError, match="NHD"):
    q, k, v = _mk(B, H, H, N, 128)
    ffpa_attn_func(
      _nhd(q), _nhd(k), _nhd(v), forward_backend=_backend(tensor_layout="NHD")
    )


def _fused_qkv(B, H, Hkv, N, D, mlp=4096, dtype=torch.bfloat16):
  """FLUX.2-style fused-QKV chunk views ([B, N, H, D] with row stride
  ``R > H * D``): Q carved from a qkv+mlp projection row, K/V from a
  kv+mlp row (GQA-safe: K/V carry ``Hkv`` heads)."""
  R = 3 * H * D + mlp
  proj = torch.randn(B, N, R, dtype=dtype, device="cuda") * 0.5
  q = proj[..., 0:H * D].view(B, N, H, D)
  Rkv = 2 * Hkv * D + mlp
  kv = torch.randn(B, N, Rkv, dtype=dtype, device="cuda") * 0.5
  k = kv[..., 0:Hkv * D].view(B, N, Hkv, D)
  v = kv[..., Hkv * D:2 * Hkv * D].view(B, N, Hkv, D)
  return q, k, v


@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
@pytest.mark.parametrize(
  "B,H,Hkv,N",
  [(1, 24, 24, 2048), (1, 24, 24, 3000), (2, 8, 8, 2048), (1, 24, 6, 2048)],
  ids=["full", "tail", "batch", "gqa"],
)
@pytest.mark.parametrize(
  "vq", ["per_channel", "per_block"], ids=["vchan", "vblk"]
)
def test_fp8_strided_nhd_bit_exact(B, H, Hkv, N, vq, causal, monkeypatch):
  """Strided-NHD (fused-QKV chunk) inputs read through the relaxed
  ``ffpa_layout_of`` gate must match the packed-NHD and BHND runs bitwise:
  the quantize pre-kernels are stride-generic, so only the addressing
  differs. ``batch`` pins the batch-stride wiring, ``tail`` a partial
  last tile."""
  monkeypatch.setenv("FFPA_CUDA_ALLOW_SMALL_D", "1")
  torch.manual_seed(0)
  D = 128
  q, k, v = _fused_qkv(B, H, Hkv, N, D)
  assert not q.is_contiguous()
  backend = _backend(enable_fp8=True, fp8_v_quant_method=vq)
  backend_nhd = _backend(
    enable_fp8=True, fp8_v_quant_method=vq, tensor_layout="NHD"
  )
  with torch.no_grad():
    out_b = ffpa_attn_func(
      q.permute(0, 2, 1, 3).contiguous(),
      k.permute(0, 2, 1, 3).contiguous(),
      v.permute(0, 2, 1, 3).contiguous(),
      is_causal=causal,
      forward_backend=backend,
    )
    out_s = ffpa_attn_func(
      q, k, v, is_causal=causal, forward_backend=backend_nhd
    )
    out_n = ffpa_attn_func(
      q.contiguous(),
      k.contiguous(),
      v.contiguous(),
      is_causal=causal,
      forward_backend=backend_nhd,
    )
  assert out_s.shape == (B, N, H, D) and out_s.is_contiguous()
  assert torch.equal(out_s, out_n)
  assert torch.equal(out_s, out_b.permute(0, 2, 1, 3))


@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
@pytest.mark.parametrize(
  "B,H,Hkv,N",
  [(1, 24, 24, 2048), (2, 8, 8, 2048), (1, 24, 6, 2048)],
  ids=["full", "batch", "gqa"],
)
def test_fp16_strided_nhd_bit_exact(B, H, Hkv, N, causal, monkeypatch):
  """fp16 persist-D stride-parameterized TMA descriptors read strided
  fused-QKV views zero-copy; K/V stay in the same NHD family with
  different row strides (the FLUX.2 shape)."""
  monkeypatch.setenv("FFPA_CUDA_ALLOW_SMALL_D", "1")
  torch.manual_seed(0)
  D = 128
  q, k, v = _fused_qkv(B, H, Hkv, N, D)
  with torch.no_grad():
    out_s = ffpa_attn_func(
      q, k, v, is_causal=causal, forward_backend=_backend(tensor_layout="NHD")
    )
    out_n = ffpa_attn_func(
      q.contiguous(),
      k.contiguous(),
      v.contiguous(),
      is_causal=causal,
      forward_backend=_backend(tensor_layout="NHD"),
    )
  assert out_s.shape == (B, N, H, D) and out_s.is_contiguous()
  assert torch.equal(out_s, out_n)


@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
@pytest.mark.parametrize("D", [128, 192], ids=["d128-fused", "d192-sep"])
def test_fp4_strided_nhd_bit_exact(D, causal, monkeypatch):
  """fp4 persist-D relaxed gate: fused (D<=128) and separate (D>128)
  quantize chains both read strided fused-QKV views zero-copy."""
  monkeypatch.setenv("FFPA_CUDA_ALLOW_SMALL_D", "1")
  torch.manual_seed(0)
  B, H, N = 1, 24, 2048
  q, k, v = _fused_qkv(B, H, H, N, D)
  kw = dict(enable_fp4=True, fp4_hybrid=False)
  with torch.no_grad():
    out_s = ffpa_attn_func(
      q,
      k,
      v,
      is_causal=causal,
      forward_backend=_backend(tensor_layout="NHD", **kw)
    )
    out_n = ffpa_attn_func(
      q.contiguous(),
      k.contiguous(),
      v.contiguous(),
      is_causal=causal,
      forward_backend=_backend(tensor_layout="NHD", **kw),
    )
  assert out_s.shape == (B, N, H, D) and out_s.is_contiguous()
  assert torch.equal(out_s, out_n)


def test_fp4_strided_nhd_batch2(monkeypatch):
  """B=2 pins the batch-stride wiring of the fp4 pre-kernels/delta_s."""
  monkeypatch.setenv("FFPA_CUDA_ALLOW_SMALL_D", "1")
  torch.manual_seed(0)
  B, H, N, D = 2, 8, 2048, 128
  q, k, v = _fused_qkv(B, H, H, N, D, mlp=1024)
  kw = dict(enable_fp4=True, fp4_hybrid=False)
  with torch.no_grad():
    out_s = ffpa_attn_func(
      q, k, v, forward_backend=_backend(tensor_layout="NHD", **kw)
    )
    out_n = ffpa_attn_func(
      q.contiguous(),
      k.contiguous(),
      v.contiguous(),
      forward_backend=_backend(tensor_layout="NHD", **kw),
    )
  assert torch.equal(out_s, out_n)


@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
def test_fp8_hybrid_strided_nhd_bit_exact(causal, monkeypatch):
  """Hybrid + strided: stage-1 materializes the strided K/V for the fp16
  kernel (shared ``prepare_hybrid_stage1`` fallback), stage-2 reads the
  originals zero-copy — the result still matches bitwise."""
  monkeypatch.setenv("FFPA_CUDA_ALLOW_SMALL_D", "1")
  torch.manual_seed(0)
  B, H, N, D = 1, 24, 2048, 128
  q, k, v = _fused_qkv(B, H, H, N, D)
  kw = dict(enable_fp8=True, fp8_hybrid=True, fp8_hybrid_n_early=256)
  with torch.no_grad():
    out_s = ffpa_attn_func(
      q,
      k,
      v,
      is_causal=causal,
      forward_backend=_backend(tensor_layout="NHD", **kw)
    )
    out_n = ffpa_attn_func(
      q.contiguous(),
      k.contiguous(),
      v.contiguous(),
      is_causal=causal,
      forward_backend=_backend(tensor_layout="NHD", **kw),
    )
  assert torch.equal(out_s, out_n)


def test_fp8_hadamard_strided_nhd_bit_exact(monkeypatch):
  """Hadamard + strided: the WHT path materializes non-contiguous inputs
  (safe copy path), output stays bitwise-equal to the packed run."""
  monkeypatch.setenv("FFPA_CUDA_ALLOW_SMALL_D", "1")
  torch.manual_seed(0)
  B, H, N, D = 1, 24, 2048, 128
  q, k, v = _fused_qkv(B, H, H, N, D)
  kw = dict(enable_fp8=True, fp8_hadamard=True)
  with torch.no_grad():
    out_s = ffpa_attn_func(
      q, k, v, forward_backend=_backend(tensor_layout="NHD", **kw)
    )
    out_n = ffpa_attn_func(
      q.contiguous(),
      k.contiguous(),
      v.contiguous(),
      forward_backend=_backend(tensor_layout="NHD", **kw),
    )
  assert torch.equal(out_s, out_n)


def test_fp8_strided_nhd_rejections(monkeypatch):
  """Layouts outside the relaxed contract must fail loudly (never
  silently mis-index): unaligned row strides, a broken batch stride at
  B>1, and head-overlapping rows (``stride(row) < H*D``)."""
  from ffpa_attn import is_nhd_zero_copy_input

  monkeypatch.setenv("FFPA_CUDA_ALLOW_SMALL_D", "1")
  torch.manual_seed(0)
  B, N, H, D = 1, 1024, 8, 128

  def _views(t, Bv, Nv):
    return (
      t[..., 0:H * D].view(Bv, Nv, H, D),
      t[..., H * D:2 * H * D].view(Bv, Nv, H, D),
      t[..., 2 * H * D:3 * H * D].view(Bv, Nv, H, D),
    )

  def _reject(q, k, v, match):
    assert not is_nhd_zero_copy_input(q)
    with pytest.raises(RuntimeError, match=match):
      with torch.no_grad():
        ffpa_attn_func(
          q,
          k,
          v,
          forward_backend=_backend(enable_fp8=True, tensor_layout="NHD")
        )

  # Row stride 3*H*D + 5 elements is not 16B-aligned for bf16.
  t = torch.randn(B, N, 3 * H * D + 5, dtype=torch.bfloat16, device="cuda")
  _reject(*_views(t, B, N), "16B-aligned")
  # B=2 with gap rows between batches: batch stride != N * row stride.
  gap = 8
  big = torch.randn(2 * N + gap, 3 * H * D, dtype=torch.bfloat16, device="cuda")
  st = ((N + gap) * 3 * H * D, 3 * H * D, D, 1)
  _reject(
    torch.as_strided(big, (2, N, H, D), st),
    torch.as_strided(big[:, H * D:], (2, N, H, D), st),
    torch.as_strided(big[:, 2 * H * D:], (2, N, H, D), st),
    "permute view",
  )
  # Rows narrower than H*D interleave heads.
  narrow = torch.randn(B, N * H * D, dtype=torch.bfloat16, device="cuda")
  ov = (N * H * D, H * D - 8, D, 1)
  _reject(
    torch.as_strided(narrow, (B, N, H, D), ov),
    torch.as_strided(narrow, (B, N, H, D), ov),
    torch.as_strided(narrow, (B, N, H, D), ov),
    "permute view",
  )


# ---------------------------------------------------------------------------
# split-D / M4N2 (RFC FC-1): strided-NHD reads on the large-D kernels.
# Inputs are [B, H, N, D]-shaped strided fused-QKV chunk views consumed
# zero-copy; O stays BHND-packed (the NHD O store is persist-D only until
# FC-2), so the packed contiguous run is the reference.


def _strided_hnd(q, k, v):
  """[B, N, H, D] fused-QKV chunk views as [B, H, N, D]-shaped strided
  views — what the HND-default call path sees (C++ stride detection is
  the layout ground truth)."""
  return q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)


@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
@pytest.mark.parametrize(
  "B,H,Hkv,N",
  [(1, 24, 24, 2048), (1, 24, 24, 3000), (2, 8, 8, 2048), (1, 24, 6, 2048)],
  ids=["full", "tail", "batch", "gqa"],
)
@pytest.mark.parametrize(
  "vq", ["per_channel", "per_block"], ids=["vchan", "vblk"]
)
def test_fp8_strided_nhd_split_d_bit_exact(B, H, Hkv, N, vq, causal):
  """D=320 lands on the split-D fp8 kernel: strided fused-QKV inputs are
  read zero-copy through the relaxed gate and the independent Lv
  descriptor; only the pre-kernel addressing differs, so the packed run
  must agree bitwise."""
  torch.manual_seed(0)
  D = 320
  q, k, v = _fused_qkv(B, H, Hkv, N, D)
  backend = _backend(enable_fp8=True, fp8_v_quant_method=vq, fp8_hybrid=False)
  qs, ks, vs = _strided_hnd(q, k, v)
  with torch.no_grad():
    out_s = ffpa_attn_func(
      qs, ks, vs, is_causal=causal, forward_backend=backend
    )
    out_c = ffpa_attn_func(
      qs.contiguous(),
      ks.contiguous(),
      vs.contiguous(),
      is_causal=causal,
      forward_backend=backend,
    )
  assert out_s.shape == (B, H, N, D)
  assert torch.equal(out_s, out_c)


@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
@pytest.mark.parametrize(
  "B,H,Hkv,N", [(1, 24, 24, 2048), (1, 24, 6, 2048)], ids=["full", "gqa"]
)
@pytest.mark.parametrize(
  "vq", ["per_channel", "per_block"], ids=["vchan", "vblk"]
)
def test_fp8_strided_nhd_m4n2_bit_exact(B, H, Hkv, N, vq, causal):
  """D=768 lands on the split-D M4N2 fp8 kernel (same FC-1 wiring)."""
  torch.manual_seed(0)
  D = 768
  q, k, v = _fused_qkv(B, H, Hkv, N, D)
  backend = _backend(enable_fp8=True, fp8_v_quant_method=vq, fp8_hybrid=False)
  qs, ks, vs = _strided_hnd(q, k, v)
  with torch.no_grad():
    out_s = ffpa_attn_func(
      qs, ks, vs, is_causal=causal, forward_backend=backend
    )
    out_c = ffpa_attn_func(
      qs.contiguous(),
      ks.contiguous(),
      vs.contiguous(),
      is_causal=causal,
      forward_backend=backend,
    )
  assert out_s.shape == (B, H, N, D)
  assert torch.equal(out_s, out_c)


@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
@pytest.mark.parametrize(
  "B,H,Hkv,N",
  [(1, 24, 24, 2048), (2, 8, 8, 2048), (1, 24, 6, 2048)],
  ids=["full", "batch", "gqa"],
)
def test_fp4_strided_nhd_split_d_bit_exact(B, H, Hkv, N, causal):
  """D=320 lands on the split-D fp4 kernel: the quantize chain is
  tensor-stride native, kv-mean/delta_s read through the relaxed Lkv."""
  torch.manual_seed(0)
  D = 320
  q, k, v = _fused_qkv(B, H, Hkv, N, D)
  backend = _backend(enable_fp4=True, fp4_hybrid=False)
  qs, ks, vs = _strided_hnd(q, k, v)
  with torch.no_grad():
    out_s = ffpa_attn_func(
      qs, ks, vs, is_causal=causal, forward_backend=backend
    )
    out_c = ffpa_attn_func(
      qs.contiguous(),
      ks.contiguous(),
      vs.contiguous(),
      is_causal=causal,
      forward_backend=backend,
    )
  assert out_s.shape == (B, H, N, D)
  assert torch.equal(out_s, out_c)


@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
def test_fp4_strided_nhd_m4n2_bit_exact(causal):
  """D=768 lands on the split-D M4N2 fp4 kernel (same FC-1 wiring)."""
  torch.manual_seed(0)
  B, H, Hkv, N, D = 1, 24, 6, 2048, 768
  q, k, v = _fused_qkv(B, H, Hkv, N, D)
  backend = _backend(enable_fp4=True, fp4_hybrid=False)
  qs, ks, vs = _strided_hnd(q, k, v)
  with torch.no_grad():
    out_s = ffpa_attn_func(
      qs, ks, vs, is_causal=causal, forward_backend=backend
    )
    out_c = ffpa_attn_func(
      qs.contiguous(),
      ks.contiguous(),
      vs.contiguous(),
      is_causal=causal,
      forward_backend=backend,
    )
  assert out_s.shape == (B, H, N, D)
  assert torch.equal(out_s, out_c)


@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
@pytest.mark.parametrize(
  "B,H,Hkv,N",
  [(1, 24, 24, 2048), (2, 8, 8, 2048), (1, 24, 6, 2048)],
  ids=["full", "batch", "gqa"],
)
def test_fp16_strided_nhd_split_d_bit_exact(B, H, Hkv, N, causal):
  """fp16 split-D (D=320): stride-parameterized TMA descriptors read the
  strided views zero-copy, mirroring the persist-D pattern."""
  torch.manual_seed(0)
  D = 320
  q, k, v = _fused_qkv(B, H, Hkv, N, D)
  backend = _backend()
  qs, ks, vs = _strided_hnd(q, k, v)
  with torch.no_grad():
    out_s = ffpa_attn_func(
      qs, ks, vs, is_causal=causal, forward_backend=backend
    )
    out_c = ffpa_attn_func(
      qs.contiguous(),
      ks.contiguous(),
      vs.contiguous(),
      is_causal=causal,
      forward_backend=backend,
    )
  assert out_s.shape == (B, H, N, D)
  assert torch.equal(out_s, out_c)


@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
def test_fp16_strided_nhd_m4n2_bit_exact(causal):
  """fp16 split-D M4N2 (D=768): same stride-parameterized descriptors."""
  torch.manual_seed(0)
  B, H, Hkv, N, D = 1, 24, 6, 2048, 768
  q, k, v = _fused_qkv(B, H, Hkv, N, D)
  backend = _backend()
  qs, ks, vs = _strided_hnd(q, k, v)
  with torch.no_grad():
    out_s = ffpa_attn_func(
      qs, ks, vs, is_causal=causal, forward_backend=backend
    )
    out_c = ffpa_attn_func(
      qs.contiguous(),
      ks.contiguous(),
      vs.contiguous(),
      is_causal=causal,
      forward_backend=backend,
    )
  assert out_s.shape == (B, H, N, D)
  assert torch.equal(out_s, out_c)


def test_fp16_strided_nhd_mixed_q_bhnd():
  """Mixed families are legal per tensor: a BHND-packed Q with strided-NHD
  K/V must match the all-packed run (the launcher dispatches the Q
  descriptor family independently from K/V)."""
  torch.manual_seed(0)
  B, H, Hkv, N, D = 1, 24, 6, 2048, 320
  q, k, v = _fused_qkv(B, H, Hkv, N, D)
  backend = _backend()
  _, ks, vs = _strided_hnd(q, k, v)
  with torch.no_grad():
    out_m = ffpa_attn_func(
      _strided_hnd(q, k, v)[0].contiguous(),
      ks,
      vs,
      forward_backend=backend,
    )
    out_c = ffpa_attn_func(
      _strided_hnd(q, k, v)[0].contiguous(),
      ks.contiguous(),
      vs.contiguous(),
      forward_backend=backend,
    )
  assert torch.equal(out_m, out_c)


def test_strided_nhd_large_d_rejections():
  """Large-D strided combos that stay closed until FC-3 must fail loudly:
  hybrid on split-D/M4N2 ranges (its stage-1 slice path needs the FC-3
  wiring). The NHD O store itself is open since FC-2 and covered
  below."""
  torch.manual_seed(0)
  B, H, N, D = 1, 24, 2048, 320
  q, k, v = _strided_hnd(*_fused_qkv(B, H, H, N, D))
  with pytest.raises(RuntimeError, match="persist-D path"):
    with torch.no_grad():
      ffpa_attn_func(
        q,
        k,
        v,
        forward_backend=_backend(
          enable_fp8=True, fp8_hybrid=True, fp8_hybrid_n_early=256
        ),
      )
  with pytest.raises(RuntimeError, match="persist-D path"):
    with torch.no_grad():
      ffpa_attn_func(
        q,
        k,
        v,
        forward_backend=_backend(
          enable_fp4=True, fp4_hybrid=True, fp4_hybrid_n_early=256
        ),
      )


# ---------------------------------------------------------------------------
# NHD O store (RFC FC-2): the split-D/M4N2 kernels take a runtime
# nhd_out branch (dynamic TMA descriptor + head-folded column tiles).
# Same-kernel NHD vs BHND outputs must agree bitwise. B>=2 and H>=2 are
# mandatory shapes here: with B==1 or H==1 the two packings coincide and
# a wrong head fold would pass silently.


@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
@pytest.mark.parametrize(
  "B,H,Hkv,N",
  [(2, 8, 8, 2048), (2, 8, 8, 3000), (2, 24, 6, 4096)],
  ids=["batch", "tail", "gqa"],
)
@pytest.mark.parametrize(
  "vq", ["per_channel", "per_block"], ids=["vchan", "vblk"]
)
def test_fp8_nhd_out_split_d_bit_exact(B, H, Hkv, N, vq, causal):
  """D=320 lands on the split-D M8N1 fp8 kernel; the tail case pins the
  R->G store's NHD row stride (Nh*kHeadDim) and offset."""
  torch.manual_seed(0)
  D = 320
  q, k, v = _mk(B, H, Hkv, N, D)
  backend = _backend(enable_fp8=True, fp8_v_quant_method=vq, fp8_hybrid=False)
  backend_nhd = _backend(
    enable_fp8=True,
    fp8_v_quant_method=vq,
    fp8_hybrid=False,
    tensor_layout="NHD",
  )
  with torch.no_grad():
    out_b = ffpa_attn_func(q, k, v, is_causal=causal, forward_backend=backend)
    out_n = ffpa_attn_func(
      _nhd(q), _nhd(k), _nhd(v), is_causal=causal, forward_backend=backend_nhd
    )
  assert out_n.shape == (B, N, H, D) and out_n.is_contiguous()
  assert torch.equal(out_n, out_b.permute(0, 2, 1, 3))


@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
@pytest.mark.parametrize(
  "B,H,Hkv,N",
  [(2, 8, 8, 2048), (2, 8, 8, 3000), (2, 24, 6, 4096)],
  ids=["batch", "tail", "gqa"],
)
@pytest.mark.parametrize(
  "vq", ["per_channel", "per_block"], ids=["vchan", "vblk"]
)
def test_fp8_nhd_out_m4n2_bit_exact(B, H, Hkv, N, vq, causal):
  """D=768 lands on the split-D M4N2 fp8 kernel; the tail case pins the
  partial-Nq semantics (kBr=64, N-warp half columns absorbed by the
  shared epilogue path)."""
  torch.manual_seed(0)
  D = 768
  q, k, v = _mk(B, H, Hkv, N, D)
  backend = _backend(enable_fp8=True, fp8_v_quant_method=vq, fp8_hybrid=False)
  backend_nhd = _backend(
    enable_fp8=True,
    fp8_v_quant_method=vq,
    fp8_hybrid=False,
    tensor_layout="NHD",
  )
  with torch.no_grad():
    out_b = ffpa_attn_func(q, k, v, is_causal=causal, forward_backend=backend)
    out_n = ffpa_attn_func(
      _nhd(q), _nhd(k), _nhd(v), is_causal=causal, forward_backend=backend_nhd
    )
  assert out_n.shape == (B, N, H, D) and out_n.is_contiguous()
  assert torch.equal(out_n, out_b.permute(0, 2, 1, 3))


@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
@pytest.mark.parametrize(
  "B,H,Hkv,N",
  [(1, 24, 24, 2048), (2, 8, 8, 2048), (1, 24, 6, 2048)],
  ids=["full", "batch", "gqa"],
)
def test_fp8_strided_nhd_out_bit_exact(B, H, Hkv, N, causal):
  """Strided fused-QKV inputs (FC-1 reads) combined with the NHD O store
  (FC-2 writes). ``_fused_qkv`` already yields [B, N, H, D]-shaped
  strided views — the NHD-call convention; the packed-NHD run is the
  reference, BHND the anchor."""
  torch.manual_seed(0)
  D = 320
  q, k, v = _fused_qkv(B, H, Hkv, N, D)
  backend_nhd = _backend(enable_fp8=True, fp8_hybrid=False, tensor_layout="NHD")
  with torch.no_grad():
    out_s = ffpa_attn_func(
      q, k, v, is_causal=causal, forward_backend=backend_nhd
    )
    out_n = ffpa_attn_func(
      q.contiguous(),
      k.contiguous(),
      v.contiguous(),
      is_causal=causal,
      forward_backend=backend_nhd,
    )
    out_b = ffpa_attn_func(
      q.permute(0, 2, 1, 3).contiguous(),
      k.permute(0, 2, 1, 3).contiguous(),
      v.permute(0, 2, 1, 3).contiguous(),
      is_causal=causal,
      forward_backend=_backend(enable_fp8=True, fp8_hybrid=False),
    )
  assert out_s.shape == (B, N, H, D) and out_s.is_contiguous()
  assert torch.equal(out_s, out_n)
  assert torch.equal(out_s, out_b.permute(0, 2, 1, 3))


@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
@pytest.mark.parametrize(
  "B,H,Hkv,N",
  [(2, 8, 8, 2048), (2, 8, 8, 3000), (2, 24, 6, 4096)],
  ids=["batch", "tail", "gqa"],
)
def test_fp4_nhd_out_split_d_bit_exact(B, H, Hkv, N, causal):
  """D=320 lands on the split-D fp4 kernel (persistent work loop); the
  per-work coordinates carry the NHD fold, the staging protocol is
  untouched."""
  torch.manual_seed(0)
  D = 320
  q, k, v = _mk(B, H, Hkv, N, D)
  backend = _backend(enable_fp4=True, fp4_hybrid=False)
  backend_nhd = _backend(enable_fp4=True, fp4_hybrid=False, tensor_layout="NHD")
  with torch.no_grad():
    out_b = ffpa_attn_func(q, k, v, is_causal=causal, forward_backend=backend)
    out_n = ffpa_attn_func(
      _nhd(q), _nhd(k), _nhd(v), is_causal=causal, forward_backend=backend_nhd
    )
  assert out_n.shape == (B, N, H, D) and out_n.is_contiguous()
  assert torch.equal(out_n, out_b.permute(0, 2, 1, 3))


@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
@pytest.mark.parametrize(
  "B,H,Hkv,N",
  [(2, 8, 8, 2048), (2, 8, 8, 3000), (2, 24, 6, 4096)],
  ids=["batch", "tail", "gqa"],
)
def test_fp4_nhd_out_m4n2_bit_exact(B, H, Hkv, N, causal):
  """D=768 lands on the split-D M4N2 fp4 kernel; the tail case pins the
  partial-Nq R->G store (kBr=64) on the packed layout."""
  torch.manual_seed(0)
  D = 768
  q, k, v = _mk(B, H, Hkv, N, D)
  backend = _backend(enable_fp4=True, fp4_hybrid=False)
  backend_nhd = _backend(enable_fp4=True, fp4_hybrid=False, tensor_layout="NHD")
  with torch.no_grad():
    out_b = ffpa_attn_func(q, k, v, is_causal=causal, forward_backend=backend)
    out_n = ffpa_attn_func(
      _nhd(q), _nhd(k), _nhd(v), is_causal=causal, forward_backend=backend_nhd
    )
  assert out_n.shape == (B, N, H, D) and out_n.is_contiguous()
  assert torch.equal(out_n, out_b.permute(0, 2, 1, 3))


@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
@pytest.mark.parametrize(
  "B,H,Hkv,N",
  [(2, 8, 8, 2048), (2, 8, 8, 3000), (2, 24, 6, 4096)],
  ids=["batch", "tail", "gqa"],
)
@pytest.mark.parametrize("D", [320, 768], ids=["d320", "d768"])
def test_fp16_nhd_out_bit_exact(B, H, Hkv, N, D, causal):
  """fp16 split-D (D=320; the epilogue is shared across the 4x4 template
  grid) and M4N2 (D=768): the runtime nhd_out branch covers every
  variant."""
  torch.manual_seed(0)
  q, k, v = _mk(B, H, Hkv, N, D)
  backend = _backend()
  backend_nhd = _backend(tensor_layout="NHD")
  with torch.no_grad():
    out_b = ffpa_attn_func(q, k, v, is_causal=causal, forward_backend=backend)
    out_n = ffpa_attn_func(
      _nhd(q), _nhd(k), _nhd(v), is_causal=causal, forward_backend=backend_nhd
    )
  assert out_n.shape == (B, N, H, D) and out_n.is_contiguous()
  assert torch.equal(out_n, out_b.permute(0, 2, 1, 3))

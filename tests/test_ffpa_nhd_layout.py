"""NHD (diffusers BNHD) layout tests for the fp16 and fp4 persist-D kernels.

Each family runs the same attention twice — BHND-packed in/out and NHD
packed in/out — and asserts the outputs agree bitwise (same kernel, only
the O store coordinates differ). The NHD tensors are materialized
NHD-packed ([B, N, H, D] contiguous) before the call, the documented
construction that avoids false NHD views.
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

  # fp16 D=256: outside the persist-D D<=128 range -> declined by the
  # python gate (CUDABackend.is_nhd_supported), which raises TypeError
  # (NHD has no path outside the fast path).
  with pytest.raises(TypeError, match="NHD"):
    with torch.no_grad():
      q, k, v = _mk(B, H, H, N, 256)
      ffpa_attn_func(
        _nhd(q),
        _nhd(k),
        _nhd(v),
        forward_backend=_backend(tensor_layout="NHD"),
      )
  # Outside the persist-D range (fp8 D>224, fp4 D>256): the gate declines
  # so NHD keeps the graceful BHND fallback. Hybrid and hadamard no
  # longer block (stride-generic writeback / launcher-side BHND copies).
  assert not _backend(enable_fp8=True).is_nhd_supported(256)
  assert not _backend(enable_fp4=True).is_nhd_supported(320)
  assert _backend(enable_fp8=True, fp8_hybrid=True).is_nhd_supported(128)
  assert _backend(enable_fp4=True, fp4_hybrid=True).is_nhd_supported(128)
  assert _backend(enable_fp8=True, fp8_hadamard=True).is_nhd_supported(128)
  assert _backend(enable_fp4=True, fp4_hadamard=True).is_nhd_supported(128)
  # fp16 with the CUTE_TMA path opted out (enable_cute=False): NHD output
  # packing only exists in the CUTE_TMA persist-D kernel.
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
  # grad-on: the fast path declines, and the full chain assumes BHND.
  with pytest.raises(TypeError, match="NHD"):
    q, k, v = _mk(B, H, H, N, 128)
    ffpa_attn_func(
      _nhd(q), _nhd(k), _nhd(v), forward_backend=_backend(tensor_layout="NHD")
    )

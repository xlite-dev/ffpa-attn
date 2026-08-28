"""FP4 (NVFP4) attention tests: parity vs fp32 SDPA across kernel families.

The FP4 sm120 path quantizes Q/K/V/P to e2m1 + ue4m3/16 SF (blockscaled
MMA); inputs stay fp16/bf16. Coverage mirrors test_ffpa_fp8.py across the
three kernel families — persist-D (D in [64,256] %64), split-D ((256,768))
and split-D M4N2 ([768,1024]) — plus the fp4-only knobs (MXFP8 PV,
smooth_v, hybrid early-row protection, 64-aligned head_dim pad).
Correctness bars follow the documented fp4 quant floor (report §6.5:
dense max_abs ~0.15 / causal ~0.70 at sigma=1; inputs here are sigma=0.5,
so the bars carry ~2x margin)."""

import time

import pytest
import torch
import torch.nn.functional as F

from ffpa_attn import ffpa_attn_func
from ffpa_attn.cli._flops import attention_fwd_flops, tflops_from_ms
from ffpa_attn.functional import CUDABackend

# Documented fp4 quant-error floor (max_abs at sigma=1); sigma=0.5 inputs
# land at roughly half of these.
_TOL_DENSE = dict(atol=1.5e-1, rtol=1.5e-1)
_TOL_CAUSAL = dict(atol=7e-1, rtol=7e-1)


def _fp4_available() -> bool:
  if not torch.cuda.is_available():
    return False
  major, _ = torch.cuda.get_device_capability()
  return major == 12


pytestmark = pytest.mark.skipif(
  not _fp4_available(),
  reason="fp4 path requires an sm_120 GPU",
)


def _fp4_backend(tensor_layout: str = "HND", **kw) -> CUDABackend:
  return CUDABackend(
    backward=False,
    enable_tma=True,
    enable_cute=True,
    enable_fp4=True,
    tensor_layout=tensor_layout,
    **kw,
  )


def _fp4_out_lse(
  q,
  k,
  v,
  causal: bool,
  hybrid: bool = False,
  n_early: int = 256,
  pv: str = "fp4",
  smooth_v: bool = False,
  hadamard: bool = False
):
  from ffpa_attn.cuda import set_cuda_backend_impl, CudaBackendImpl
  from ffpa_attn.cuda._ffpa_fwd import _ffpa_attn_forward_cuda

  set_cuda_backend_impl(CudaBackendImpl.CUTE_TMA_FP4)
  return _ffpa_attn_forward_cuda(
    q,
    k,
    v,
    causal=int(causal),
    softmax_scale=q.size(-1)**-0.5,
    fp4_hybrid=hybrid,
    fp4_hybrid_n_early=n_early,
    fp4_hadamard=hadamard,
    fp4_pv_mm_type=0 if pv == "fp4" else 1,
    fp4_smooth_v=smooth_v,
  )


def _mk(B, H, Hkv, N, D, dtype=torch.float16):
  q = torch.randn(B, H, N, D, dtype=dtype, device="cuda") * 0.5
  k = torch.randn(B, Hkv, N, D, dtype=dtype, device="cuda") * 0.5
  v = torch.randn(B, Hkv, N, D, dtype=dtype, device="cuda") * 0.5
  return q, k, v


# ---------------------------------------------------------------------------
# persist-D: D in [64,256] %64 (small-D routing needs the env gate).


@pytest.mark.parametrize("D", [64, 128, 256])
@pytest.mark.parametrize(
  "dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"]
)
def test_fp4_forward_parity_vs_fp32_sdpa(D, dtype, monkeypatch):
  monkeypatch.setenv("FFPA_CUDA_ALLOW_SMALL_D", "1")
  torch.manual_seed(0)
  B, H, N = 1, 32, 4096
  q, k, v = _mk(B, H, H, N, D, dtype)

  out = ffpa_attn_func(q, k, v, is_causal=False, forward_backend=_fp4_backend())
  ref = F.scaled_dot_product_attention(
    q.float(), k.float(), v.float(), is_causal=False
  ).to(dtype)
  torch.testing.assert_close(out, ref, **_TOL_DENSE)


@pytest.mark.parametrize("D", [64, 128])
def test_fp4_forward_parity_causal(D, monkeypatch):
  monkeypatch.setenv("FFPA_CUDA_ALLOW_SMALL_D", "1")
  torch.manual_seed(0)
  B, H, N = 1, 32, 2048
  q, k, v = _mk(B, H, H, N, D)

  out = ffpa_attn_func(q, k, v, is_causal=True, forward_backend=_fp4_backend())
  ref = F.scaled_dot_product_attention(
    q.float(), k.float(), v.float(), is_causal=True
  ).half()
  # e2m1 P/V quant error is amplified on early causal rows (few attended
  # keys): this is the format floor, not an implementation bug.
  torch.testing.assert_close(out, ref, **_TOL_CAUSAL)


def test_fp4_pv_mm_type_validation():
  assert _fp4_backend().fp4_pv_mm_type == "fp4"
  mxfp8 = CUDABackend(
    backward=False,
    enable_tma=True,
    enable_cute=True,
    enable_fp4=True,
    fp4_pv_mm_type="fp8",
  )
  assert mxfp8.fp4_pv_mm_type_code == 1
  with pytest.raises(AssertionError):
    CUDABackend(
      backward=False,
      enable_tma=True,
      enable_cute=True,
      enable_fp4=True,
      fp4_pv_mm_type="bogus",
    )


@pytest.mark.parametrize("D", [128, 192])
def test_fp4_mxfp8_pv_parity(D, monkeypatch):
  # fp4_pv_mm_type='fp8' (MXFP8 e4m3+ue8m0/32 PV) is smem-limited to D<=192.
  monkeypatch.setenv("FFPA_CUDA_ALLOW_SMALL_D", "1")
  torch.manual_seed(0)
  B, H, N = 1, 32, 4096
  q, k, v = _mk(B, H, H, N, D)

  out = ffpa_attn_func(
    q,
    k,
    v,
    is_causal=False,
    forward_backend=_fp4_backend(fp4_pv_mm_type="fp8")
  )
  ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
  torch.testing.assert_close(out.float(), ref, **_TOL_DENSE)


def test_fp4_mxfp8_pv_rejects_large_d(monkeypatch):
  monkeypatch.setenv("FFPA_CUDA_ALLOW_SMALL_D", "1")
  torch.manual_seed(0)
  B, H, N, D = 1, 32, 2048, 256  # persist-D D=256: MXFP8 PV is smem-limited
  q, k, v = _mk(B, H, H, N, D)
  with pytest.raises(RuntimeError, match="headdim|fp4"):
    with torch.no_grad():
      ffpa_attn_func(
        q, k, v, forward_backend=_fp4_backend(fp4_pv_mm_type="fp8")
      )


@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
def test_fp4_mxfp8_pv_split_d_parity(causal, monkeypatch):
  # FC-6: MXFP8 PV extends past persist-D into split-D ((256,768)); e4m3
  # V + ue8m0/32 PV through the K=128 MXFP8 atom (split-D PV Tile-K=kBc=
  # 128 fits it exactly).
  torch.manual_seed(0)
  B, H, N, D = 1, 32, 4096, 320
  q, k, v = _mk(B, H, H, N, D)

  with torch.no_grad():
    out = ffpa_attn_func(
      q,
      k,
      v,
      is_causal=causal,
      forward_backend=_fp4_backend(fp4_pv_mm_type="fp8"),
    )
  ref = F.scaled_dot_product_attention(
    q.float(), k.float(), v.float(), is_causal=causal
  )
  tol = _TOL_CAUSAL if causal else _TOL_DENSE
  torch.testing.assert_close(out.float(), ref, **tol)
  # MXFP8 PV is strictly more precise than NVFP4 PV (e4m3 vs e2m1 data):
  # never notably worse on the same inputs.
  with torch.no_grad():
    out_fp4pv = ffpa_attn_func(
      q, k, v, is_causal=causal, forward_backend=_fp4_backend()
    )
  err_mxfp8 = (out.float() - ref).abs().mean()
  err_fp4 = (out_fp4pv.float() - ref).abs().mean()
  assert err_mxfp8 <= err_fp4 * 1.5 + 1e-3


def test_fp4_mxfp8_pv_rejects_m4n2():
  # Architectural N/A: the MXFP8 PV atom consumes Tile-K=128 tokens per
  # mma, m4n2 tiles are kBc=64.
  torch.manual_seed(0)
  B, H, N, D = 1, 32, 2048, 768
  q, k, v = _mk(B, H, H, N, D)
  with pytest.raises(RuntimeError, match="persist_d"):
    with torch.no_grad():
      ffpa_attn_func(
        q, k, v, forward_backend=_fp4_backend(fp4_pv_mm_type="fp8")
      )


def test_fp4_smooth_v_parity(monkeypatch):
  # smooth_v subtracts the per-channel V mean (math-lossless for softmax
  # row-sum=1) to widen the e2m1 dynamic range; persist-D only.
  monkeypatch.setenv("FFPA_CUDA_ALLOW_SMALL_D", "1")
  torch.manual_seed(0)
  B, H, N, D = 1, 32, 4096, 128
  q, k, v = _mk(B, H, H, N, D)
  v += 2.0  # per-channel mean offset is what smooth_v removes

  out = ffpa_attn_func(
    q, k, v, is_causal=False, forward_backend=_fp4_backend(fp4_smooth_v=True)
  )
  ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
  torch.testing.assert_close(out.float(), ref, **_TOL_DENSE)


@pytest.mark.parametrize("D", [320, 768], ids=["split_d", "m4n2"])
@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
def test_fp4_smooth_v_large_d_parity(D, causal, monkeypatch):
  # FC-6: smooth_v extends past persist-D into split-D and m4n2 — the V^T
  # quantize kernels always took vm; only the launcher wiring and the
  # per-chunk epilogue add-back were missing. Math-lossless (softmax rows
  # sum to 1); v += 2.0 makes the removed mean the dominant V energy.
  monkeypatch.setenv("FFPA_CUDA_ALLOW_SMALL_D", "1")
  torch.manual_seed(0)
  B, H, N = 1, 32, 2048
  q, k, v = _mk(B, H, H, N, D)
  v += 2.0

  with torch.no_grad():
    out = ffpa_attn_func(
      q,
      k,
      v,
      is_causal=causal,
      forward_backend=_fp4_backend(fp4_smooth_v=True),
    )
    out_unsmoothed = ffpa_attn_func(
      q, k, v, is_causal=causal, forward_backend=_fp4_backend()
    )
  ref = F.scaled_dot_product_attention(
    q.float(), k.float(), v.float(), is_causal=causal
  )
  tol = _TOL_CAUSAL if causal else _TOL_DENSE
  torch.testing.assert_close(out.float(), ref, **tol)
  # The mean offset is what smooth_v removes: smoothing must beat the
  # unsmoothed run, whose e2m1 V blockscale wastes range on the offset.
  err_smooth = (out.float() - ref).abs().mean()
  err_plain = (out_unsmoothed.float() - ref).abs().mean()
  assert err_smooth < err_plain


@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
def test_fp4_hybrid_early_rows(causal, monkeypatch):
  # Hybrid stage-1 runs the fp16 kernel on the first n_early rows: early
  # rows must land at fp16 accuracy, late rows stay bitwise-identical to
  # the non-hybrid fp4 run (stage-2 is deterministic on the same inputs).
  monkeypatch.setenv("FFPA_CUDA_ALLOW_SMALL_D", "1")
  torch.manual_seed(0)
  B, H, N, D, n_early = 1, 32, 2048, 128, 256
  q, k, v = _mk(B, H, H, N, D)

  out_h = ffpa_attn_func(
    q,
    k,
    v,
    is_causal=causal,
    forward_backend=_fp4_backend(fp4_hybrid=True, fp4_hybrid_n_early=n_early),
  )
  out_q = ffpa_attn_func(
    q,
    k,
    v,
    is_causal=causal,
    forward_backend=_fp4_backend(fp4_hybrid=False),
  )
  ref = F.scaled_dot_product_attention(
    q.float(), k.float(), v.float(), is_causal=causal
  )
  early_h = (out_h[:, :, :n_early].float() - ref[:, :, :n_early]).abs().max()
  early_q = (out_q[:, :, :n_early].float() - ref[:, :, :n_early]).abs().max()
  assert early_h < early_q
  assert early_h < 0.05  # stage-1 rows are exact fp16 math
  assert torch.equal(out_h[:, :, n_early:], out_q[:, :, n_early:])


@pytest.mark.parametrize("D", [128, 256])
def test_fp4_lse_parity(D, monkeypatch):
  monkeypatch.setenv("FFPA_CUDA_ALLOW_SMALL_D", "1")
  torch.manual_seed(0)
  B, H, N = 1, 8, 2048
  q, k, v = _mk(B, H, H, N, D)

  out, lse = _fp4_out_lse(q, k, v, causal=False)
  p = (q.float() @ k.float().transpose(-1, -2)) * (D**-0.5)
  ref = torch.softmax(p, dim=-1) @ v.float()
  lse_ref = torch.logsumexp(p, dim=-1)
  torch.testing.assert_close(out.float(), ref, **_TOL_DENSE)
  torch.testing.assert_close(lse.float(), lse_ref, atol=1.5e-1, rtol=2e-2)


@pytest.mark.parametrize("D", [128, 320], ids=["persist_d", "split_d"])
def test_fp4_mxfp8_lse_parity(D, monkeypatch):
  # MXFP8 PV row_sum lives in the P*448 domain (not NVFP4's P*2688); the
  # lse correction must divide by 448 (ln(6) smaller). Guards the domain
  # constant on both persist-D and split-D (FC-6).
  monkeypatch.setenv("FFPA_CUDA_ALLOW_SMALL_D", "1")
  torch.manual_seed(0)
  B, H, N = 1, 8, 2048
  q, k, v = _mk(B, H, H, N, D)

  out, lse = _fp4_out_lse(q, k, v, causal=False, pv="fp8")
  p = (q.float() @ k.float().transpose(-1, -2)) * (D**-0.5)
  lse_ref = torch.logsumexp(p, dim=-1)
  torch.testing.assert_close(lse.float(), lse_ref, atol=5e-2, rtol=2e-2)


# ---------------------------------------------------------------------------
# head_dim pad: D_og % 8 != 0 pads to the next 64 multiple inside the
# quantize kernels (Q/K/V stay D_og-wide; zero-filled pad columns are
# exact: data 0 * SF 0 contributes 0 to every dot product).


def test_fp4_head_dim_pad_persist(monkeypatch):
  monkeypatch.setenv("FFPA_CUDA_ALLOW_SMALL_D", "1")
  torch.manual_seed(0)
  B, H, N, D = 1, 16, 2048, 72  # pads to kHeadDim=128
  q, k, v = _mk(B, H, H, N, D)

  out = ffpa_attn_func(q, k, v, is_causal=False, forward_backend=_fp4_backend())
  ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
  torch.testing.assert_close(out.float(), ref, **_TOL_DENSE)


def test_fp4_head_dim_pad_split_d():
  torch.manual_seed(0)
  B, H, N, D = 1, 16, 2048, 264  # pads to kHeadDim=320 (split-D range)
  q, k, v = _mk(B, H, H, N, D)

  out = ffpa_attn_func(q, k, v, is_causal=False, forward_backend=_fp4_backend())
  ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
  torch.testing.assert_close(out.float(), ref, **_TOL_DENSE)


# ---------------------------------------------------------------------------
# split-D: D in (256,768) %64.


@pytest.mark.parametrize("D", [320, 512])
@pytest.mark.parametrize(
  "dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"]
)
def test_fp4_split_d_forward_parity(D, dtype):
  torch.manual_seed(0)
  B, H, N = 1, 32, 4096
  q, k, v = _mk(B, H, H, N, D, dtype)

  out = ffpa_attn_func(q, k, v, is_causal=False, forward_backend=_fp4_backend())
  ref = F.scaled_dot_product_attention(
    q.float(), k.float(), v.float(), is_causal=False
  ).to(dtype)
  torch.testing.assert_close(out, ref, **_TOL_DENSE)


@pytest.mark.parametrize("D", [320, 512])
def test_fp4_split_d_causal_parity(D):
  torch.manual_seed(0)
  B, H, N = 1, 32, 2048
  q, k, v = _mk(B, H, H, N, D)

  out = ffpa_attn_func(q, k, v, is_causal=True, forward_backend=_fp4_backend())
  ref = F.scaled_dot_product_attention(
    q.float(), k.float(), v.float(), is_causal=True
  ).half()
  torch.testing.assert_close(out, ref, **_TOL_CAUSAL)


@pytest.mark.parametrize("D", [320, 512])
def test_fp4_split_d_gqa_parity(D):
  torch.manual_seed(0)
  B, H, H_kv, N = 1, 32, 8, 4096
  q, k, v = _mk(B, H, H_kv, N, D)

  out = ffpa_attn_func(
    q, k, v, is_causal=False, enable_gqa=True, forward_backend=_fp4_backend()
  )
  k_rep = k.repeat_interleave(H // H_kv, dim=1)
  v_rep = v.repeat_interleave(H // H_kv, dim=1)
  ref = F.scaled_dot_product_attention(q.float(), k_rep.float(), v_rep.float())
  torch.testing.assert_close(out.float(), ref, **_TOL_DENSE)


@pytest.mark.parametrize("D", [320, 512])
def test_fp4_split_d_nkv_unaligned_lse(D):
  # N % 128 != 0: tail tile + partial quantize; also pins the lse flat
  # [B, Nh, Nq] indexing (stride Nq per head).
  torch.manual_seed(0)
  B, H, N = 1, 8, 2120
  q, k, v = _mk(B, H, H, N, D)

  out, lse = _fp4_out_lse(q, k, v, causal=True)
  ref = F.scaled_dot_product_attention(
    q.float(), k.float(), v.float(), is_causal=True
  )
  torch.testing.assert_close(out.float(), ref, **_TOL_CAUSAL)
  p = (q.float() @ k.float().transpose(-1, -2)) * (D**-0.5)
  mask = torch.triu(torch.ones(N, N, device="cuda"), diagonal=1).bool()
  lse_ref = torch.logsumexp(p.masked_fill(mask, float("-inf")), dim=-1)
  torch.testing.assert_close(lse.float(), lse_ref, atol=2e-1, rtol=2e-2)


def test_fp4_nq_not_multiple_of_8_lse():
  # Mirrors the fp8 regression: lse must be indexed with the exact Nq.
  torch.manual_seed(0)
  B, H, N, D = 1, 8, 2047, 320  # 2047 % 8 == 7
  q, k, v = _mk(B, H, H, N, D)

  out, lse = _fp4_out_lse(q, k, v, causal=True)
  ref = F.scaled_dot_product_attention(
    q.float(), k.float(), v.float(), is_causal=True
  )
  torch.testing.assert_close(out.float(), ref, **_TOL_CAUSAL)
  p = (q.float() @ k.float().transpose(-1, -2)) * (D**-0.5)
  mask = torch.triu(torch.ones(N, N, device="cuda"), diagonal=1).bool()
  lse_ref = torch.logsumexp(p.masked_fill(mask, float("-inf")), dim=-1)
  torch.testing.assert_close(lse.float(), lse_ref, atol=2e-1, rtol=2e-2)


def test_fp4_nkv_unaligned_dense_parity():
  # Dense + Nkv % 16 != 0: the V^T quantize pad columns must stay
  # zero-filled (garbage there poisons O through the tail tile).
  torch.manual_seed(0)
  B, H, N, D = 1, 8, 2120, 320
  q, k, v = _mk(B, H, H, N, D)

  out, _ = _fp4_out_lse(q, k, v, causal=False)
  ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
  assert not torch.isnan(out.float()).any()
  torch.testing.assert_close(out.float(), ref, **_TOL_DENSE)


# ---------------------------------------------------------------------------
# split-D M4N2: D in [768,1024] %64 (atom layout (4,2,1), single-level
# per-16 P SF — slightly coarser than persist/split's two-level 2688 domain).


@pytest.mark.parametrize("D", [768, 1024])
@pytest.mark.parametrize(
  "dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"]
)
def test_fp4_split_d_m4n2_forward_parity(D, dtype):
  torch.manual_seed(0)
  B, H, N = 1, 16, 4096
  q, k, v = _mk(B, H, H, N, D, dtype)

  out = ffpa_attn_func(q, k, v, is_causal=False, forward_backend=_fp4_backend())
  ref = F.scaled_dot_product_attention(
    q.float(), k.float(), v.float(), is_causal=False
  ).to(dtype)
  torch.testing.assert_close(out, ref, **_TOL_DENSE)


@pytest.mark.parametrize("D", [768, 1024])
def test_fp4_split_d_m4n2_causal_parity(D):
  torch.manual_seed(0)
  B, H, N = 1, 16, 2048
  q, k, v = _mk(B, H, H, N, D)

  out = ffpa_attn_func(q, k, v, is_causal=True, forward_backend=_fp4_backend())
  ref = F.scaled_dot_product_attention(
    q.float(), k.float(), v.float(), is_causal=True
  ).half()
  torch.testing.assert_close(out, ref, **_TOL_CAUSAL)


@pytest.mark.parametrize("D", [768, 1024])
def test_fp4_split_d_m4n2_gqa_parity(D):
  torch.manual_seed(0)
  B, H, H_kv, N = 1, 16, 4, 4096
  q, k, v = _mk(B, H, H_kv, N, D)

  out = ffpa_attn_func(
    q, k, v, is_causal=False, enable_gqa=True, forward_backend=_fp4_backend()
  )
  k_rep = k.repeat_interleave(H // H_kv, dim=1)
  v_rep = v.repeat_interleave(H // H_kv, dim=1)
  ref = F.scaled_dot_product_attention(q.float(), k_rep.float(), v_rep.float())
  torch.testing.assert_close(out.float(), ref, **_TOL_DENSE)


# ---------------------------------------------------------------------------
# perf: markdown latency/TFLOPS table vs SDPA (split-D shape, no small-D
# env needed).


def _bench_ms(fn, warmup=10, iters=30) -> float:
  for _ in range(warmup):
    fn()
  torch.cuda.synchronize()
  ts = []
  for _ in range(iters):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    ts.append(time.perf_counter() - t0)
  return min(ts) * 1e3


def test_fp4_perf_vs_sdpa(capsys):
  """Markdown perf/TFLOPS table: FFPA FP4 vs SDPA fp16/bf16."""
  torch.manual_seed(0)
  B, H, N, D = 1, 32, 8192, 320
  q, k, v = _mk(B, H, H, N, D)
  flops = attention_fwd_flops(
    batch=B, num_heads_q=H, nq=N, nkv=N, headdim=D, causal=False
  )

  backends = [
    (
      "FFPA FP4",
      lambda:
      ffpa_attn_func(q, k, v, is_causal=False, forward_backend=_fp4_backend()),
    ),
    ("SDPA fp16", lambda: F.scaled_dot_product_attention(q, k, v)),
    (
      "SDPA bf16", lambda: F.
      scaled_dot_product_attention(q.bfloat16(), k.bfloat16(), v.bfloat16())
    ),
  ]
  ms = {name: _bench_ms(fn) for name, fn in backends}

  lines = [
    "| backend | min (ms) | TFLOPS | speedup vs SDPA fp16 |",
    "|---|---|---|---|",
  ]
  for name, _ in backends:
    t = tflops_from_ms(flops, ms[name])
    lines.append(
      f"| {name} | {ms[name]:.3f} | {t:.1f} | "
      f"{ms['SDPA fp16'] / ms[name]:.3f}x |"
    )
  with capsys.disabled():
    print(f"\nshape: B{B} H{H} N{N} D{D} (non-causal)\n")
    print("\n".join(lines))

  assert ms["FFPA FP4"] < ms["SDPA fp16"]


# ---------------------------------------------------------------------------
# attn_bias (FC-4): additive bias / bool mask across the three fp4 families.
# The fp4 kernels store KV permuted (kv_perm32), so the causal+bias cases
# double as the column-mapping check (a wrong map flips masked columns and
# fails parity immediately).


def _mk_attn_bias(q, k, kind):
  Nq, Nkv = q.size(2), k.size(2)
  torch.manual_seed(1)
  if kind == "additive":
    return torch.randn(1, 1, Nq, Nkv, dtype=q.dtype, device=q.device) * 0.25
  if kind == "broadcast":
    return torch.randn(1, 1, 1, Nkv, dtype=q.dtype, device=q.device) * 0.25
  mask = torch.ones(Nq, Nkv, dtype=torch.bool, device=q.device)
  mask[:, 3::7] = False
  mask[:, 0] = True
  return mask


def _sdpa_bias_ref(q, k, v, bias, causal):
  # SDPA takes is_causal XOR attn_mask, so causal+bias composes into one
  # additive mask (this mirrors the kernel: -inf masking overrides bias).
  if bias.dtype == torch.bool:
    score_bias = torch.zeros_like(bias, dtype=torch.float32
                                  ).masked_fill(~bias, float("-inf"))
  else:
    score_bias = bias.float()
  if causal:
    tri = torch.ones(q.size(2), k.size(2), dtype=torch.bool,
                     device=q.device).tril()
    score_bias = score_bias.masked_fill(~tri, float("-inf"))
  return F.scaled_dot_product_attention(
    q.float(), k.float(), v.float(), attn_mask=score_bias
  )


@pytest.mark.parametrize(
  "kind", ["dense", "causal_fused"], ids=["dense", "causal-fused"]
)
@pytest.mark.parametrize(
  "D", [128, 320, 768], ids=["persist_d", "split_d", "split_d_m4n2"]
)
def test_fp4_attn_bias_parity(D, kind, monkeypatch):
  # The public API rejects explicit attn_mask + is_causal (SDPA semantics),
  # so the causal composition runs as one additive mask. This still checks
  # the kv_perm32 column mapping: a wrong map flips masked/bias columns and
  # fails parity immediately.
  monkeypatch.setenv("FFPA_CUDA_ALLOW_SMALL_D", "1")
  torch.manual_seed(0)
  B, H, N = 1, 8, 2048
  q, k, v = _mk(B, H, H, N, D)
  bias = _mk_attn_bias(q, k, "additive")
  if kind == "causal_fused":
    tri = torch.ones(N, N, dtype=torch.bool, device=q.device).tril()
    bias = bias.masked_fill(~tri, float("-inf"))

  out = ffpa_attn_func(q, k, v, attn_mask=bias, forward_backend=_fp4_backend())
  ref = _sdpa_bias_ref(q, k, v, bias, causal=False).half()
  tol = _TOL_CAUSAL if kind == "causal_fused" else _TOL_DENSE
  torch.testing.assert_close(out.float(), ref.float(), **tol)


@pytest.mark.parametrize("kind", ["additive", "broadcast", "bool"])
def test_fp4_attn_mask_forms_match_sdpa(kind, monkeypatch):
  monkeypatch.setenv("FFPA_CUDA_ALLOW_SMALL_D", "1")
  torch.manual_seed(0)
  B, H, N, D = 1, 8, 2048, 128
  q, k, v = _mk(B, H, H, N, D)
  bias = _mk_attn_bias(q, k, kind)

  out = ffpa_attn_func(q, k, v, attn_mask=bias, forward_backend=_fp4_backend())
  ref = _sdpa_bias_ref(q, k, v, bias, causal=False).half()
  torch.testing.assert_close(out.float(), ref.float(), **_TOL_DENSE)


def test_fp4_attn_bias_gqa_parity(monkeypatch):
  monkeypatch.setenv("FFPA_CUDA_ALLOW_SMALL_D", "1")
  torch.manual_seed(0)
  B, H, Hkv, Nq, Nkv, D = 1, 4, 2, 1024, 1536, 128
  q = torch.randn(B, H, Nq, D, dtype=torch.float16, device="cuda") * 0.5
  k = torch.randn(B, Hkv, Nkv, D, dtype=torch.float16, device="cuda") * 0.5
  v = torch.randn(B, Hkv, Nkv, D, dtype=torch.float16, device="cuda") * 0.5
  bias = _mk_attn_bias(q, k, "additive")

  out = ffpa_attn_func(
    q, k, v, attn_mask=bias, enable_gqa=True, forward_backend=_fp4_backend()
  )
  ref = F.scaled_dot_product_attention(
    q.float(),
    k.float(),
    v.float(),
    attn_mask=bias.float(),
    enable_gqa=True,
  ).half()
  torch.testing.assert_close(out.float(), ref.float(), **_TOL_DENSE)


def test_fp4_hybrid_attn_bias_parity(monkeypatch):
  # Hybrid stage-1 (fp16 kernel) takes the [0, n_early) bias rows; stage-2
  # offsets rows via q_start_row against the full bias.
  monkeypatch.setenv("FFPA_CUDA_ALLOW_SMALL_D", "1")
  torch.manual_seed(0)
  B, H, N, D, n_early = 1, 8, 2048, 128, 256
  q, k, v = _mk(B, H, H, N, D)
  bias = _mk_attn_bias(q, k, "additive")

  out = ffpa_attn_func(
    q,
    k,
    v,
    attn_mask=bias,
    forward_backend=_fp4_backend(fp4_hybrid=True, fp4_hybrid_n_early=n_early),
  )
  ref = _sdpa_bias_ref(q, k, v, bias, causal=False).half()
  torch.testing.assert_close(out.float(), ref.float(), **_TOL_DENSE)


def test_fp4_rejects_dropout(monkeypatch):
  monkeypatch.setenv("FFPA_CUDA_ALLOW_SMALL_D", "1")
  torch.manual_seed(0)
  B, H, N, D = 1, 4, 512, 128
  q, k, v = _mk(B, H, H, N, D)
  bias = _mk_attn_bias(q, k, "additive")

  with pytest.raises(RuntimeError, match="does not support dropout"):
    ffpa_attn_func(
      q, k, v, attn_mask=bias, dropout_p=0.1, forward_backend=_fp4_backend()
    )

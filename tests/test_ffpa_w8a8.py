"""W8A8 FP8 attention tests: parity vs fp32 SDPA + perf/TFLOPS vs Sage/SDPA.

The W8A8 persist-D sm120 path quantizes Q/K/V to e4m3 in-flight and runs
fp8 MMA; inputs stay fp16/bf16. Correctness tests assert parity against the
fp32 SDPA reference; the perf test prints a markdown table (latency + TFLOPS)
comparing FFPA FP8, SageAttention (when installed) and SDPA fp16/bf16.
"""

import importlib.util
import time

import pytest
import torch
import torch.nn.functional as F

from ffpa_attn import ffpa_attn_func
from ffpa_attn.cli._flops import attention_fwd_flops, tflops_from_ms
from ffpa_attn.functional import CUDABackend

SAGE_INSTALLED = importlib.util.find_spec("sageattention") is not None


def _w8a8_available() -> bool:
  if not torch.cuda.is_available():
    return False
  major, _ = torch.cuda.get_device_capability()
  return major == 12


pytestmark = pytest.mark.skipif(
  not _w8a8_available(),
  reason="w8a8 fp8 path requires an sm_120 GPU",
)


def _w8a8_backend() -> CUDABackend:
  return CUDABackend(
    backward=False, enable_tma=True, enable_cute=True, enable_w8a8=True
  )


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


@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize(
  "dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"]
)
def test_w8a8_forward_parity_vs_fp32_sdpa(D, dtype):
  torch.manual_seed(0)
  B, H, N = 1, 32, 4096
  q = torch.randn(B, H, N, D, dtype=dtype, device="cuda") * 0.5
  k = torch.randn(B, H, N, D, dtype=dtype, device="cuda") * 0.5
  v = torch.randn(B, H, N, D, dtype=dtype, device="cuda") * 0.5

  out = ffpa_attn_func(
    q, k, v, is_causal=False, forward_backend=_w8a8_backend()
  )
  ref = F.scaled_dot_product_attention(
    q.float(), k.float(), v.float(), is_causal=False
  ).to(dtype)
  torch.testing.assert_close(out, ref, atol=4e-2, rtol=4e-2)


def test_w8a8_forward_parity_causal():
  torch.manual_seed(0)
  B, H, N, D = 1, 32, 2048, 128
  q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.5
  k = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.5
  v = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.5

  out = ffpa_attn_func(q, k, v, is_causal=True, forward_backend=_w8a8_backend())
  ref = F.scaled_dot_product_attention(
    q.float(), k.float(), v.float(), is_causal=True
  ).half()
  # fp8 P-quant error is amplified on early causal rows (few attended keys).
  torch.testing.assert_close(out, ref, atol=1e-1, rtol=1e-1)


def test_w8a8_perf_vs_sage_sdpa(capsys):
  """Markdown perf/TFLOPS table: FFPA FP8 vs Sage vs SDPA fp16/bf16."""
  torch.manual_seed(0)
  B, H, N, D = 1, 32, 8192, 128
  q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.5
  k = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.5
  v = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.5
  flops = attention_fwd_flops(
    batch=B, num_heads_q=H, nq=N, nkv=N, headdim=D, causal=False
  )

  backends = [
    (
      "FFPA FP8 (w8a8)", lambda:
      ffpa_attn_func(q, k, v, is_causal=False, forward_backend=_w8a8_backend())
    ),
    ("SDPA fp16", lambda: F.scaled_dot_product_attention(q, k, v)),
    (
      "SDPA bf16", lambda: F.
      scaled_dot_product_attention(q.bfloat16(), k.bfloat16(), v.bfloat16())
    ),
  ]
  if SAGE_INSTALLED:
    from sageattention import sageattn
    backends.insert(
      1, (
        "SageAttention",
        lambda: sageattn(q, k, v, tensor_layout="HND", is_causal=False)
      )
    )

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

  assert ms["FFPA FP8 (w8a8)"] < ms["SDPA fp16"]


def _set_qk_int8(monkeypatch, enable: bool) -> None:
  if enable:
    monkeypatch.setenv("FFPA_W8A8_QK_INT8", "1")
  else:
    monkeypatch.delenv("FFPA_W8A8_QK_INT8", raising=False)


def _w8a8_out_lse(q, k, v, causal: bool, smooth_k: bool = True):
  from ffpa_attn.cuda import set_cuda_backend_impl, CudaBackendImpl
  from ffpa_attn.cuda._ffpa_fwd import _ffpa_attn_forward_cuda

  set_cuda_backend_impl(CudaBackendImpl.CUTE_TMA_W8A8)
  return _ffpa_attn_forward_cuda(
    q,
    k,
    v,
    causal=int(causal),
    softmax_scale=q.size(-1)**-0.5,
    smooth_k=smooth_k,
  )


@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize(
  "dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"]
)
@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
def test_w8a8_qk_int8_forward_parity(D, dtype, causal, monkeypatch):
  _set_qk_int8(monkeypatch, True)
  torch.manual_seed(0)
  B, H, N = 1, 32, 4096
  q = torch.randn(B, H, N, D, dtype=dtype, device="cuda") * 0.5
  k = torch.randn(B, H, N, D, dtype=dtype, device="cuda") * 0.5
  v = torch.randn(B, H, N, D, dtype=dtype, device="cuda") * 0.5

  out = ffpa_attn_func(
    q, k, v, is_causal=causal, forward_backend=_w8a8_backend()
  )
  ref = F.scaled_dot_product_attention(
    q.float(), k.float(), v.float(), is_causal=causal
  ).to(dtype)
  # fp8 P-quant error is amplified on early causal rows (few attended keys).
  tol = 1e-1 if causal else 4e-2
  torch.testing.assert_close(out, ref, atol=tol, rtol=tol)


def test_w8a8_qk_int8_beats_fp8_high_amp(monkeypatch):
  torch.manual_seed(0)
  B, H, N, D = 1, 32, 4096, 128
  q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 4.0
  k = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 4.0
  v = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 4.0
  ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float())

  _set_qk_int8(monkeypatch, False)
  out_fp8 = ffpa_attn_func(q, k, v, forward_backend=_w8a8_backend())
  _set_qk_int8(monkeypatch, True)
  out_int8 = ffpa_attn_func(q, k, v, forward_backend=_w8a8_backend())

  rel_fp8 = ((out_fp8.float() - ref).norm() / ref.norm()).item()
  rel_int8 = ((out_int8.float() - ref).norm() / ref.norm()).item()
  assert rel_int8 < rel_fp8
  assert rel_int8 < 0.10


@pytest.mark.parametrize("qk_int8", [False, True], ids=["fp8-qk", "int8-qk"])
@pytest.mark.parametrize(
  "smooth_k", [False, True], ids=["smooth-off", "smooth-on"]
)
def test_w8a8_smooth_k_and_lse(qk_int8, smooth_k, monkeypatch):
  _set_qk_int8(monkeypatch, qk_int8)
  torch.manual_seed(0)
  B, H, N, D = 1, 32, 4096, 128
  q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.5
  k = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.5
  v = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.5

  out, lse = _w8a8_out_lse(q, k, v, causal=False, smooth_k=smooth_k)
  p = (q.float() @ k.float().transpose(-1, -2)) * (D**-0.5)
  ref = torch.softmax(p, dim=-1) @ v.float()
  lse_ref = torch.logsumexp(p, dim=-1)
  torch.testing.assert_close(out.float(), ref, atol=4e-2, rtol=4e-2)
  torch.testing.assert_close(lse.float(), lse_ref, atol=5e-2, rtol=1e-3)


@pytest.mark.parametrize("qk_int8", [False, True], ids=["fp8-qk", "int8-qk"])
def test_w8a8_gqa_forward_parity(qk_int8, monkeypatch):
  _set_qk_int8(monkeypatch, qk_int8)
  torch.manual_seed(0)
  B, H, H_kv, N, D = 1, 32, 8, 4096, 128
  q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.5
  k = torch.randn(B, H_kv, N, D, dtype=torch.float16, device="cuda") * 0.5
  v = torch.randn(B, H_kv, N, D, dtype=torch.float16, device="cuda") * 0.5

  out = ffpa_attn_func(
    q, k, v, is_causal=False, enable_gqa=True, forward_backend=_w8a8_backend()
  )
  k_rep = k.repeat_interleave(H // H_kv, dim=1)
  v_rep = v.repeat_interleave(H // H_kv, dim=1)
  ref = F.scaled_dot_product_attention(q.float(), k_rep.float(), v_rep.float())
  torch.testing.assert_close(out.float(), ref, atol=4e-2, rtol=4e-2)


@pytest.mark.parametrize("qk_int8", [False, True], ids=["fp8-qk", "int8-qk"])
def test_w8a8_nkv_unaligned_parity(qk_int8, monkeypatch):
  _set_qk_int8(monkeypatch, qk_int8)
  torch.manual_seed(0)
  B, H, N, D = 1, 32, 2120, 128  # N % 128 != 0: tail tile + partial quantize
  q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.5
  k = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.5
  v = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.5

  out, lse = _w8a8_out_lse(q, k, v, causal=True)
  ref = F.scaled_dot_product_attention(
    q.float(), k.float(), v.float(), is_causal=True
  )
  torch.testing.assert_close(out.float(), ref, atol=1e-1, rtol=1e-1)
  p = (q.float() @ k.float().transpose(-1, -2)) * (D**-0.5)
  mask = torch.triu(torch.ones(N, N, device="cuda"), diagonal=1).bool()
  lse_ref = torch.logsumexp(p.masked_fill(mask, float("-inf")), dim=-1)
  torch.testing.assert_close(lse.float(), lse_ref, atol=5e-2, rtol=1e-3)

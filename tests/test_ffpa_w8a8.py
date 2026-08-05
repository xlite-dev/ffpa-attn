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

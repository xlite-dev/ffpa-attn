"""FP8 attention tests: parity vs fp32 SDPA + perf/TFLOPS vs Sage/SDPA.

The FP8 persist-D sm120 path quantizes Q/K/V to e4m3 in-flight and runs
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


def _fp8_available() -> bool:
  if not torch.cuda.is_available():
    return False
  major, _ = torch.cuda.get_device_capability()
  return major == 12


pytestmark = pytest.mark.skipif(
  not _fp8_available(),
  reason="fp8 path requires an sm_120 GPU",
)


def _fp8_backend(tensor_layout: str = "HND") -> CUDABackend:
  return CUDABackend(
    backward=False,
    enable_tma=True,
    enable_cute=True,
    enable_fp8=True,
    tensor_layout=tensor_layout,
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
def test_fp8_forward_parity_vs_fp32_sdpa(D, dtype):
  torch.manual_seed(0)
  B, H, N = 1, 32, 4096
  q = torch.randn(B, H, N, D, dtype=dtype, device="cuda") * 0.5
  k = torch.randn(B, H, N, D, dtype=dtype, device="cuda") * 0.5
  v = torch.randn(B, H, N, D, dtype=dtype, device="cuda") * 0.5

  out = ffpa_attn_func(q, k, v, is_causal=False, forward_backend=_fp8_backend())
  ref = F.scaled_dot_product_attention(
    q.float(), k.float(), v.float(), is_causal=False
  ).to(dtype)
  torch.testing.assert_close(out, ref, atol=4e-2, rtol=4e-2)


def test_fp8_forward_parity_causal():
  torch.manual_seed(0)
  B, H, N, D = 1, 32, 2048, 128
  q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.5
  k = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.5
  v = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.5

  out = ffpa_attn_func(q, k, v, is_causal=True, forward_backend=_fp8_backend())
  ref = F.scaled_dot_product_attention(
    q.float(), k.float(), v.float(), is_causal=True
  ).half()
  # fp8 P-quant error is amplified on early causal rows (few attended keys).
  torch.testing.assert_close(out, ref, atol=1e-1, rtol=1e-1)


def test_fp8_perf_vs_sage_sdpa(capsys):
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
      "FFPA FP8", lambda:
      ffpa_attn_func(q, k, v, is_causal=False, forward_backend=_fp8_backend())
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

  assert ms["FFPA FP8"] < ms["SDPA fp16"]


def _set_qk_int8(monkeypatch, enable: bool) -> None:
  # The FFPA_FP8_QK_INT8 env was removed; swap the backend factory directly.
  def backend():
    return CUDABackend(
      backward=False,
      enable_tma=True,
      enable_cute=True,
      enable_fp8=True,
      fp8_qk_mm_type="int8" if enable else "fp8",
    )

  monkeypatch.setitem(globals(), "_fp8_backend", backend)


def test_fp8_qk_mm_type_validation():
  # fp8_qk_mm_type replaces the removed qk_int8 param / FFPA_FP8_QK_INT8 env.
  assert _fp8_backend().fp8_qk_mm_type == "fp8"
  int8_backend = CUDABackend(
    backward=False,
    enable_tma=True,
    enable_cute=True,
    enable_fp8=True,
    fp8_qk_mm_type="int8",
  )
  assert int8_backend.fp8_qk_mm_type_code == 1
  with pytest.raises(AssertionError):
    CUDABackend(
      backward=False,
      enable_tma=True,
      enable_cute=True,
      enable_fp8=True,
      fp8_qk_mm_type="bogus",
    )


def _fp8_out_lse(
  q, k, v, causal: bool, smooth_k: bool = True, qk_int8: bool = False
):
  from ffpa_attn.cuda import set_cuda_backend_impl, CudaBackendImpl
  from ffpa_attn.cuda._ffpa_fwd import _ffpa_attn_forward_cuda

  set_cuda_backend_impl(CudaBackendImpl.CUTE_TMA_FP8)
  return _ffpa_attn_forward_cuda(
    q,
    k,
    v,
    causal=int(causal),
    softmax_scale=q.size(-1)**-0.5,
    fp8_smooth_k=smooth_k,
    fp8_qk_mm_type=int(qk_int8),
  )


@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize(
  "dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"]
)
@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
def test_fp8_qk_int8_forward_parity(D, dtype, causal, monkeypatch):
  _set_qk_int8(monkeypatch, True)
  torch.manual_seed(0)
  B, H, N = 1, 32, 4096
  q = torch.randn(B, H, N, D, dtype=dtype, device="cuda") * 0.5
  k = torch.randn(B, H, N, D, dtype=dtype, device="cuda") * 0.5
  v = torch.randn(B, H, N, D, dtype=dtype, device="cuda") * 0.5

  out = ffpa_attn_func(
    q, k, v, is_causal=causal, forward_backend=_fp8_backend()
  )
  ref = F.scaled_dot_product_attention(
    q.float(), k.float(), v.float(), is_causal=causal
  ).to(dtype)
  # Causal early rows (few attended keys) carry fp8 P/V quant error that the
  # fp8 PV path does not correct, so they get the looser 1e-1 bar.
  tol = 1e-1 if causal else 4e-2
  torch.testing.assert_close(out, ref, atol=tol, rtol=tol)


def test_fp8_qk_int8_beats_fp8_high_amp(monkeypatch):
  torch.manual_seed(0)
  B, H, N, D = 1, 32, 4096, 128
  q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 4.0
  k = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 4.0
  v = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 4.0
  ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float())

  _set_qk_int8(monkeypatch, False)
  out_fp8 = ffpa_attn_func(q, k, v, forward_backend=_fp8_backend())
  _set_qk_int8(monkeypatch, True)
  out_int8 = ffpa_attn_func(q, k, v, forward_backend=_fp8_backend())

  rel_fp8 = ((out_fp8.float() - ref).norm() / ref.norm()).item()
  rel_int8 = ((out_int8.float() - ref).norm() / ref.norm()).item()
  assert rel_int8 < rel_fp8
  assert rel_int8 < 0.10


@pytest.mark.parametrize("qk_int8", [False, True], ids=["fp8-qk", "int8-qk"])
@pytest.mark.parametrize(
  "smooth_k", [False, True], ids=["smooth-off", "smooth-on"]
)
def test_fp8_smooth_k_and_lse(qk_int8, smooth_k, monkeypatch):
  _set_qk_int8(monkeypatch, qk_int8)
  torch.manual_seed(0)
  B, H, N, D = 1, 32, 4096, 128
  q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.5
  k = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.5
  v = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.5

  out, lse = _fp8_out_lse(
    q, k, v, causal=False, smooth_k=smooth_k, qk_int8=qk_int8
  )
  p = (q.float() @ k.float().transpose(-1, -2)) * (D**-0.5)
  ref = torch.softmax(p, dim=-1) @ v.float()
  lse_ref = torch.logsumexp(p, dim=-1)
  torch.testing.assert_close(out.float(), ref, atol=4e-2, rtol=4e-2)
  torch.testing.assert_close(lse.float(), lse_ref, atol=5e-2, rtol=1e-3)


@pytest.mark.parametrize("qk_int8", [False, True], ids=["fp8-qk", "int8-qk"])
def test_fp8_gqa_forward_parity(qk_int8, monkeypatch):
  _set_qk_int8(monkeypatch, qk_int8)
  torch.manual_seed(0)
  B, H, H_kv, N, D = 1, 32, 8, 4096, 128
  q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.5
  k = torch.randn(B, H_kv, N, D, dtype=torch.float16, device="cuda") * 0.5
  v = torch.randn(B, H_kv, N, D, dtype=torch.float16, device="cuda") * 0.5

  out = ffpa_attn_func(
    q, k, v, is_causal=False, enable_gqa=True, forward_backend=_fp8_backend()
  )
  k_rep = k.repeat_interleave(H // H_kv, dim=1)
  v_rep = v.repeat_interleave(H // H_kv, dim=1)
  ref = F.scaled_dot_product_attention(q.float(), k_rep.float(), v_rep.float())
  torch.testing.assert_close(out.float(), ref, atol=4e-2, rtol=4e-2)


@pytest.mark.parametrize("qk_int8", [False, True], ids=["fp8-qk", "int8-qk"])
def test_fp8_nkv_unaligned_parity(qk_int8, monkeypatch):
  _set_qk_int8(monkeypatch, qk_int8)
  torch.manual_seed(0)
  B, H, N, D = 1, 32, 2120, 128  # N % 128 != 0: tail tile + partial quantize
  q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.5
  k = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.5
  v = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.5

  out, lse = _fp8_out_lse(q, k, v, causal=True, qk_int8=qk_int8)
  ref = F.scaled_dot_product_attention(
    q.float(), k.float(), v.float(), is_causal=True
  )
  torch.testing.assert_close(out.float(), ref, atol=1e-1, rtol=1e-1)
  p = (q.float() @ k.float().transpose(-1, -2)) * (D**-0.5)
  mask = torch.triu(torch.ones(N, N, device="cuda"), diagonal=1).bool()
  lse_ref = torch.logsumexp(p.masked_fill(mask, float("-inf")), dim=-1)
  torch.testing.assert_close(lse.float(), lse_ref, atol=5e-2, rtol=1e-3)


@pytest.mark.parametrize("qk_int8", [False, True], ids=["fp8-qk", "int8-qk"])
def test_fp8_nq_not_multiple_of_8_lse(qk_int8, monkeypatch):
  # Regression: the torch op used to allocate lse with seqlen rounded up to 8
  # and pass a sliced view, while kernels index lse flat with stride Nq per
  # head; head h's rows shifted by h and the last head's trailing rows were
  # left unwritten whenever Nq % 8 != 0.
  _set_qk_int8(monkeypatch, qk_int8)
  torch.manual_seed(0)
  B, H, N, D = 1, 8, 2047, 128  # 2047 % 8 == 7
  q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.5
  k = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.5
  v = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.5

  out, lse = _fp8_out_lse(q, k, v, causal=True, qk_int8=qk_int8)
  ref = F.scaled_dot_product_attention(
    q.float(), k.float(), v.float(), is_causal=True
  )
  torch.testing.assert_close(out.float(), ref, atol=1e-1, rtol=1e-1)
  p = (q.float() @ k.float().transpose(-1, -2)) * (D**-0.5)
  mask = torch.triu(torch.ones(N, N, device="cuda"), diagonal=1).bool()
  lse_ref = torch.logsumexp(p.masked_fill(mask, float("-inf")), dim=-1)
  torch.testing.assert_close(lse.float(), lse_ref, atol=5e-2, rtol=1e-3)


# Regression (Wan2.1-T2V self-attn, dense Nq=Nkv=14040, black-video bug): the
# V^T quantize kernels left the [Nkv, Nkv_pad) pad columns uninitialized, TMA
# loaded them into the tail tile, and 0 * NaN garbage poisoned O for every
# fp8 config. Dense + Nkv%16!=0 is the trigger; causal tests missed it.
@pytest.mark.parametrize("qk_int8", [False, True], ids=["fp8-qk", "int8-qk"])
def test_fp8_nkv_unaligned_dense_parity(qk_int8, monkeypatch):
  _set_qk_int8(monkeypatch, qk_int8)
  torch.manual_seed(0)
  B, H, N, D = 1, 8, 2120, 128  # N % 128 != 0 and N % 16 != 0, dense
  q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.5
  k = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.5
  v = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.5

  out, lse = _fp8_out_lse(q, k, v, causal=False, qk_int8=qk_int8)
  ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
  torch.testing.assert_close(out.float(), ref, atol=4e-2, rtol=4e-2)
  p = (q.float() @ k.float().transpose(-1, -2)) * (D**-0.5)
  lse_ref = torch.logsumexp(p, dim=-1)
  torch.testing.assert_close(lse.float(), lse_ref, atol=5e-2, rtol=1e-3)


def test_fp8_wan_self_attn_shape_parity():
  # Wan2.1-T2V 480x832x33f self-attention shape (9*30*52 tokens, bf16, dense)
  # with the cache-dit default ffpa_fp8 config (int8 QK + f16 PV acc +
  # per_thread QK + per_channel V): NaN-free and within the dense parity bar.
  torch.manual_seed(0)
  B, H, N, D = 1, 40, 14040, 128
  q = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda") * 0.5
  k = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda") * 0.5
  v = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda") * 0.5

  backend = CUDABackend(
    backward=False,
    enable_tma=True,
    enable_cute=True,
    enable_fp8=True,
    fp8_qk_mm_type="int8",
    fp8_pv_acc_type="f16",
    fp8_q_quant_method="per_thread",
    fp8_k_quant_method="per_thread",
    fp8_v_quant_method="per_channel",
    fp8_smooth_k=True,
    fp8_smooth_v=True,
    fp8_hybrid=True,
    fp8_hybrid_n_early=256,
  )
  out = ffpa_attn_func(q, k, v, is_causal=False, forward_backend=backend)
  ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
  assert not torch.isnan(out.float()).any()
  torch.testing.assert_close(out.float(), ref, atol=4e-2, rtol=4e-2)


# Split-D path: D>128 routes to a separate kernel (split_d.cuh) that
# shards the head dim across d-chunks for register pressure. Same parity
# bars as the persist-D path; causal and lse checks use the looser 1e-1 bar
# that documents the known fp8 early-row quant-error floor (split-D has the
# same limitation as persist-D for causal Nq<8192).
@pytest.mark.parametrize("D", [320, 512])
@pytest.mark.parametrize(
  "dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"]
)
def test_fp8_split_d_forward_parity(D, dtype):
  torch.manual_seed(0)
  B, H, N = 1, 32, 4096
  q = torch.randn(B, H, N, D, dtype=dtype, device="cuda") * 0.5
  k = torch.randn(B, H, N, D, dtype=dtype, device="cuda") * 0.5
  v = torch.randn(B, H, N, D, dtype=dtype, device="cuda") * 0.5

  out = ffpa_attn_func(q, k, v, is_causal=False, forward_backend=_fp8_backend())
  ref = F.scaled_dot_product_attention(
    q.float(), k.float(), v.float(), is_causal=False
  ).to(dtype)
  torch.testing.assert_close(out, ref, atol=4e-2, rtol=4e-2)


@pytest.mark.parametrize("D", [320, 512])
def test_fp8_split_d_causal_parity(D):
  torch.manual_seed(0)
  B, H, N = 1, 32, 2048
  q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.5
  k = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.5
  v = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.5

  out = ffpa_attn_func(q, k, v, is_causal=True, forward_backend=_fp8_backend())
  ref = F.scaled_dot_product_attention(
    q.float(), k.float(), v.float(), is_causal=True
  ).half()
  torch.testing.assert_close(out, ref, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("D", [320, 512])
@pytest.mark.parametrize("qk_int8", [False, True], ids=["fp8-qk", "int8-qk"])
def test_fp8_split_d_gqa_parity(D, qk_int8, monkeypatch):
  _set_qk_int8(monkeypatch, qk_int8)
  torch.manual_seed(0)
  B, H, H_kv, N = 1, 32, 8, 4096
  q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.5
  k = torch.randn(B, H_kv, N, D, dtype=torch.float16, device="cuda") * 0.5
  v = torch.randn(B, H_kv, N, D, dtype=torch.float16, device="cuda") * 0.5

  out = ffpa_attn_func(
    q, k, v, is_causal=False, enable_gqa=True, forward_backend=_fp8_backend()
  )
  k_rep = k.repeat_interleave(H // H_kv, dim=1)
  v_rep = v.repeat_interleave(H // H_kv, dim=1)
  ref = F.scaled_dot_product_attention(q.float(), k_rep.float(), v_rep.float())
  torch.testing.assert_close(out.float(), ref, atol=4e-2, rtol=4e-2)


@pytest.mark.parametrize("D", [320, 512])
@pytest.mark.parametrize("qk_int8", [False, True], ids=["fp8-qk", "int8-qk"])
def test_fp8_split_d_nkv_unaligned_lse(D, qk_int8, monkeypatch):
  _set_qk_int8(monkeypatch, qk_int8)
  torch.manual_seed(0)
  B, H, N = 1, 32, 2120  # N % 128 != 0: tail tile + partial quantize
  q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.5
  k = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.5
  v = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.5

  out, lse = _fp8_out_lse(q, k, v, causal=True, qk_int8=qk_int8)
  ref = F.scaled_dot_product_attention(
    q.float(), k.float(), v.float(), is_causal=True
  )
  torch.testing.assert_close(out.float(), ref, atol=1e-1, rtol=1e-1)
  p = (q.float() @ k.float().transpose(-1, -2)) * (D**-0.5)
  mask = torch.triu(torch.ones(N, N, device="cuda"), diagonal=1).bool()
  lse_ref = torch.logsumexp(p.masked_fill(mask, float("-inf")), dim=-1)
  torch.testing.assert_close(lse.float(), lse_ref, atol=1e-1, rtol=1e-3)


@pytest.mark.parametrize("D", [320, 512])
@pytest.mark.parametrize("qk_int8", [False, True], ids=["fp8-qk", "int8-qk"])
@pytest.mark.parametrize(
  "smooth_k", [False, True], ids=["smooth-off", "smooth-on"]
)
def test_fp8_split_d_smooth_k_and_lse(D, qk_int8, smooth_k, monkeypatch):
  _set_qk_int8(monkeypatch, qk_int8)
  torch.manual_seed(0)
  B, H, N = 1, 32, 4096
  q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.5
  k = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.5
  v = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.5

  out, lse = _fp8_out_lse(
    q, k, v, causal=False, smooth_k=smooth_k, qk_int8=qk_int8
  )
  p = (q.float() @ k.float().transpose(-1, -2)) * (D**-0.5)
  ref = torch.softmax(p, dim=-1) @ v.float()
  lse_ref = torch.logsumexp(p, dim=-1)
  torch.testing.assert_close(out.float(), ref, atol=4e-2, rtol=4e-2)
  torch.testing.assert_close(lse.float(), lse_ref, atol=5e-2, rtol=1e-3)


# Split-D M4N2 FP8 path: D>=768 routes to split_d_m4n2.cuh (M4N2 atom
# layout (4,2,1), P SMEM roundtrip via DefaultCopy since stmatrix is b16-only).
# O regs = D/4 per thread (vs M8N1's D/2 which spills for D>=768).
# Causal early-row accuracy is worse than M8N1 (O_err ~0.17 vs ~0.08) due to
# smaller kBc=64 tiles (more quant blocks, more error accumulation); the 2e-1
# bar documents this known fp8 limitation.
@pytest.mark.parametrize("D", [768, 896, 1024])
@pytest.mark.parametrize(
  "dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"]
)
def test_fp8_split_d_m4n2_forward_parity(D, dtype):
  torch.manual_seed(0)
  B, H, N = 1, 32, 4096
  q = torch.randn(B, H, N, D, dtype=dtype, device="cuda") * 0.5
  k = torch.randn(B, H, N, D, dtype=dtype, device="cuda") * 0.5
  v = torch.randn(B, H, N, D, dtype=dtype, device="cuda") * 0.5

  out = ffpa_attn_func(q, k, v, is_causal=False, forward_backend=_fp8_backend())
  ref = F.scaled_dot_product_attention(
    q.float(), k.float(), v.float(), is_causal=False
  ).to(dtype)
  torch.testing.assert_close(out, ref, atol=5e-3, rtol=5e-3)


@pytest.mark.parametrize("D", [768, 1024])
def test_fp8_split_d_m4n2_causal_parity(D):
  torch.manual_seed(0)
  B, H, N = 1, 32, 2048
  q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.5
  k = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.5
  v = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.5

  out = ffpa_attn_func(q, k, v, is_causal=True, forward_backend=_fp8_backend())
  ref = F.scaled_dot_product_attention(
    q.float(), k.float(), v.float(), is_causal=True
  ).half()
  # fp8 P-quant error on causal early rows is amplified by the smaller kBc=64
  # tiles (vs M8N1's kBc=128); O_err ~0.17 < 2e-1 documents this known limit.
  torch.testing.assert_close(out, ref, atol=2e-1, rtol=2e-1)


@pytest.mark.parametrize("D", [768, 1024])
@pytest.mark.parametrize("qk_int8", [False, True], ids=["fp8-qk", "int8-qk"])
def test_fp8_split_d_m4n2_gqa_parity(D, qk_int8, monkeypatch):
  _set_qk_int8(monkeypatch, qk_int8)
  torch.manual_seed(0)
  B, H, H_kv, N = 1, 32, 8, 4096
  q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.5
  k = torch.randn(B, H_kv, N, D, dtype=torch.float16, device="cuda") * 0.5
  v = torch.randn(B, H_kv, N, D, dtype=torch.float16, device="cuda") * 0.5

  out = ffpa_attn_func(
    q, k, v, is_causal=False, enable_gqa=True, forward_backend=_fp8_backend()
  )
  k_rep = k.repeat_interleave(H // H_kv, dim=1)
  v_rep = v.repeat_interleave(H // H_kv, dim=1)
  ref = F.scaled_dot_product_attention(q.float(), k_rep.float(), v_rep.float())
  torch.testing.assert_close(out.float(), ref, atol=5e-3, rtol=5e-3)


@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
@pytest.mark.parametrize(
  "B,H,Hkv,N",
  [(1, 24, 24, 2048), (1, 24, 24, 16383), (2, 8, 8, 3000), (1, 24, 8, 4096)],
  ids=["full", "tail", "batch", "gqa"],
)
def test_fp8_nhd_layout_bit_exact(B, H, Hkv, N, causal, monkeypatch):
  # NHD (diffusers BNHD) storage in/out vs BHND in/out: same kernel run, so
  # the two layouts must agree bitwise. The NHD tensors are materialized
  # NHD-packed then permuted back to BHND shape (the documented construction
  # that avoids false NHD views). fp8_hybrid=False: the hybrid stage-1 is
  # BHND-only, so causal NHD takes the plain causal persist-D kernel.
  monkeypatch.setenv("FFPA_CUDA_ALLOW_SMALL_D", "1")  # D=128 small-D routing
  torch.manual_seed(0)
  D = 128
  backend = CUDABackend(
    backward=False,
    enable_tma=True,
    enable_cute=True,
    enable_fp8=True,
    fp8_hybrid=False,
  )
  backend_nhd = CUDABackend(
    backward=False,
    enable_tma=True,
    enable_cute=True,
    enable_fp8=True,
    fp8_hybrid=False,
    tensor_layout="NHD",
  )
  q = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda") * 0.5
  k = torch.randn(B, Hkv, N, D, dtype=torch.bfloat16, device="cuda") * 0.5
  v = torch.randn(B, Hkv, N, D, dtype=torch.bfloat16, device="cuda") * 0.5
  # NHD-native storage: [B, N, H(, Hkv), D]
  q3 = q.permute(0, 2, 1, 3).contiguous()
  k3 = k.permute(0, 2, 1, 3).contiguous()
  v3 = v.permute(0, 2, 1, 3).contiguous()

  with torch.no_grad():
    out_b = ffpa_attn_func(q, k, v, is_causal=causal, forward_backend=backend)
    out_n = ffpa_attn_func(
      q3,
      k3,
      v3,
      is_causal=causal,
      forward_backend=backend_nhd,
    )
  assert out_n.shape == (B, N, H, D) and out_n.is_contiguous()
  torch.testing.assert_close(out_n, out_b.permute(0, 2, 1, 3))


def test_fp8_nhd_layout_parity_vs_sdpa(monkeypatch):
  monkeypatch.setenv("FFPA_CUDA_ALLOW_SMALL_D", "1")
  torch.manual_seed(0)
  B, H, N, D = 1, 24, 4096, 128
  q3 = torch.randn(B, N, H, D, dtype=torch.bfloat16, device="cuda") * 0.5
  k3 = torch.randn(B, N, H, D, dtype=torch.bfloat16, device="cuda") * 0.5
  v3 = torch.randn(B, N, H, D, dtype=torch.bfloat16, device="cuda") * 0.5
  with torch.no_grad():
    out = ffpa_attn_func(q3, k3, v3, forward_backend=_fp8_backend("NHD"))
  q, k, v = (t.permute(0, 2, 1, 3) for t in (q3, k3, v3))
  ref = F.scaled_dot_product_attention(q.float(), k.float(),
                                       v.float()).to(torch.bfloat16)
  torch.testing.assert_close(out, ref.permute(0, 2, 1, 3), atol=4e-2, rtol=4e-2)


def test_fp8_nhd_layout_rejections(monkeypatch):
  monkeypatch.setenv("FFPA_CUDA_ALLOW_SMALL_D", "1")
  torch.manual_seed(0)
  B, H, N, D = 1, 24, 4096, 128
  q3 = torch.randn(B, N, H, D, dtype=torch.bfloat16, device="cuda")
  k3 = torch.randn(B, N, H, D, dtype=torch.bfloat16, device="cuda")
  v3 = torch.randn(B, N, H, D, dtype=torch.bfloat16, device="cuda")
  with pytest.raises(TypeError):
    # full chain (no explicit backend) does not accept a tensor_layout kwarg
    with torch.no_grad():
      ffpa_attn_func(q3, k3, v3, tensor_layout="NHD")
  with pytest.raises(TypeError):
    # grad-on declines the fast path, so NHD has no supported path
    ffpa_attn_func(q3, k3, v3, forward_backend=_fp8_backend("NHD"))


# ---------------------------------------------------------------------------
# attn_bias (FC-4): additive bias / bool mask across the three fp8 families.


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
def test_fp8_attn_bias_parity(D, kind):
  # The public API rejects explicit attn_mask + is_causal (SDPA semantics),
  # so the causal composition runs as one additive mask: bias + the causal
  # -inf triangle (this still exercises the kernel bias load on every tile
  # including -inf values).
  torch.manual_seed(0)
  B, H, N = 1, 8, 2048
  q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.5
  k = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.5
  v = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.5
  bias = _mk_attn_bias(q, k, "additive")
  if kind == "causal_fused":
    tri = torch.ones(N, N, dtype=torch.bool, device=q.device).tril()
    bias = bias.masked_fill(~tri, float("-inf"))

  out = ffpa_attn_func(q, k, v, attn_mask=bias, forward_backend=_fp8_backend())
  ref = _sdpa_bias_ref(q, k, v, bias, causal=False).half()
  if kind == "causal_fused":
    # Same early-row quant floor as the causal tests.
    torch.testing.assert_close(out, ref, atol=1e-1, rtol=1e-1)
  else:
    torch.testing.assert_close(out, ref, atol=4e-2, rtol=4e-2)


@pytest.mark.parametrize("kind", ["additive", "broadcast", "bool"])
def test_fp8_attn_mask_forms_match_sdpa(kind):
  torch.manual_seed(0)
  B, H, N, D = 1, 8, 2048, 128
  q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.5
  k = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.5
  v = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.5
  bias = _mk_attn_bias(q, k, kind)

  out = ffpa_attn_func(q, k, v, attn_mask=bias, forward_backend=_fp8_backend())
  ref = _sdpa_bias_ref(q, k, v, bias, causal=False).half()
  torch.testing.assert_close(out, ref, atol=4e-2, rtol=4e-2)


def test_fp8_attn_bias_gqa_parity():
  torch.manual_seed(0)
  B, H, Hkv, Nq, Nkv, D = 1, 4, 2, 1024, 1536, 128
  q = torch.randn(B, H, Nq, D, dtype=torch.float16, device="cuda") * 0.5
  k = torch.randn(B, Hkv, Nkv, D, dtype=torch.float16, device="cuda") * 0.5
  v = torch.randn(B, Hkv, Nkv, D, dtype=torch.float16, device="cuda") * 0.5
  bias = _mk_attn_bias(q, k, "additive")

  out = ffpa_attn_func(
    q, k, v, attn_mask=bias, enable_gqa=True, forward_backend=_fp8_backend()
  )
  ref = F.scaled_dot_product_attention(
    q.float(),
    k.float(),
    v.float(),
    attn_mask=bias.float(),
    enable_gqa=True,
  ).half()
  torch.testing.assert_close(out, ref, atol=4e-2, rtol=4e-2)


def test_fp8_hybrid_attn_bias_parity():
  torch.manual_seed(0)
  B, H, N, D, n_early = 1, 8, 2048, 128, 256
  q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.5
  k = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.5
  v = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.5
  bias = _mk_attn_bias(q, k, "additive")
  backend = CUDABackend(
    backward=False,
    enable_tma=True,
    enable_cute=True,
    enable_fp8=True,
    fp8_hybrid=True,
    fp8_hybrid_n_early=n_early,
  )

  out = ffpa_attn_func(q, k, v, attn_mask=bias, forward_backend=backend)
  ref = _sdpa_bias_ref(q, k, v, bias, causal=False).half()
  torch.testing.assert_close(out, ref, atol=4e-2, rtol=4e-2)


def test_fp8_rejects_dropout():
  torch.manual_seed(0)
  B, H, N, D = 1, 4, 512, 128
  q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.5
  k = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.5
  v = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.5
  bias = _mk_attn_bias(q, k, "additive")

  with pytest.raises(RuntimeError, match="does not support dropout"):
    ffpa_attn_func(
      q, k, v, attn_mask=bias, dropout_p=0.1, forward_backend=_fp8_backend()
    )

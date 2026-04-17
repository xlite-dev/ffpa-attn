"""FFPA attention examples.

Runs ``ffpa_attn_func`` across the supported shape regimes and compares
against ``torch.nn.functional.scaled_dot_product_attention``:

1. Self-Attention            -- Nq == Nkv, Nh_q == Nh_kv, aligned seqlen.
2. Cross / Decode Attention  -- Nq != Nkv (short query, long KV).
3. Grouped-Query Attention   -- Nh_q % Nh_kv == 0, Nh_kv < Nh_q (MQA => Nh_kv=1).
4. Causal Attention          -- causal=True, queries aligned to KV tail.
5. Non-aligned Seqlen        -- N=8191 (not a multiple of Bc=64), exercises
                               cp.async zero-fill + softmax -inf mask +
                               per-row store predicate on the tail tile.

Usage::

    CUDA_VISIBLE_DEVICES=0 python examples/run_ffpa_attn.py
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

from ffpa_attn import ffpa_attn_func

D = 512
STAGES = 2
WARMUP, ITERS = 2, 5


def _sdpa_ref(q, k, v, is_causal: bool = False):
  return F.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(q.size(-1)), is_causal=is_causal)


def _time_fn(fn, *args, warmup: int = WARMUP, iters: int = ITERS) -> float:
  for _ in range(warmup):
    fn(*args)
  torch.cuda.synchronize()
  t0 = time.perf_counter()
  for _ in range(iters):
    fn(*args)
  torch.cuda.synchronize()
  return (time.perf_counter() - t0) * 1000.0 / iters  # ms


def _expand_kv(k: torch.Tensor, v: torch.Tensor, nh_q: int):
  """Repeat K/V heads to match Nh_q for the SDPA reference path."""
  nh_kv = k.size(1)
  if nh_kv == nh_q:
    return k, v
  rep = nh_q // nh_kv
  return k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)


def _run_case(
  name: str,
  dtype: torch.dtype,
  B: int,
  Nh_q: int,
  Nh_kv: int,
  Nq: int,
  Nkv: int,
  causal: bool = False,
  acc: str = "f32",
) -> None:
  torch.manual_seed(0)
  q = torch.randn(B, Nh_q, Nq, D, dtype=dtype, device="cuda")
  k = torch.randn(B, Nh_kv, Nkv, D, dtype=dtype, device="cuda")
  v = torch.randn(B, Nh_kv, Nkv, D, dtype=dtype, device="cuda")

  out_ffpa = ffpa_attn_func(q, k, v, stages=STAGES, acc=acc, causal=causal)
  k_ref, v_ref = _expand_kv(k, v, Nh_q)
  out_sdpa = _sdpa_ref(q, k_ref, v_ref, is_causal=causal)

  diff = (out_ffpa.float() - out_sdpa.float()).abs()
  tol = 5e-2 if dtype == torch.bfloat16 else 2e-2
  ok = torch.allclose(out_ffpa, out_sdpa, atol=tol, rtol=tol)

  ms_ffpa = _time_fn(
    lambda q, k, v: ffpa_attn_func(q, k, v, stages=STAGES, acc=acc, causal=causal),
    q,
    k,
    v,
  )
  ms_sdpa = _time_fn(lambda q, k, v: _sdpa_ref(q, k, v, is_causal=causal), q, k_ref, v_ref)

  dt_tag = str(dtype).replace("torch.", "")
  print(
    f"[{name:<14} {dt_tag:<8} acc={acc}] "
    f"B={B} Hq={Nh_q} Hkv={Nh_kv} Nq={Nq} Nkv={Nkv} D={D} causal={int(causal)}  "
    f"max|diff|={diff.max().item():.4f}  mean|diff|={diff.mean().item():.5f}  "
    f"allclose(atol={tol})={ok}  "
    f"FFPA={ms_ffpa:.2f} ms  SDPA={ms_sdpa:.2f} ms  speedup={ms_sdpa / ms_ffpa:.2f}x"
  )


def main() -> None:
  if not torch.cuda.is_available():
    raise SystemExit("CUDA is required to run this example.")

  for dtype in (torch.float16, torch.bfloat16):
    # 1) Self-Attention: Nq == Nkv, Nh_q == Nh_kv, aligned seqlen.
    _run_case("self-attn", dtype, B=1, Nh_q=32, Nh_kv=32, Nq=8192, Nkv=8192)
    # 2) Cross / Decode Attention: short query against long KV cache.
    _run_case("cross-attn", dtype, B=1, Nh_q=32, Nh_kv=32, Nq=1024, Nkv=8192)
    # 3) Grouped-Query Attention: Nh_q=32, Nh_kv=8 (group size 4).
    _run_case("gqa", dtype, B=1, Nh_q=32, Nh_kv=8, Nq=8192, Nkv=8192)
    # 4) Causal Attention: queries aligned to the tail of the KV sequence.
    _run_case("causal", dtype, B=1, Nh_q=32, Nh_kv=32, Nq=8192, Nkv=8192, causal=True)
    # 5) Non-aligned seqlen (8191 % 64 != 0): exercises tail-tile masking.
    _run_case("non-aligned", dtype, B=1, Nh_q=8, Nh_kv=8, Nq=8191, Nkv=8191)


if __name__ == "__main__":
  main()

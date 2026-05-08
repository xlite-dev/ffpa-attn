"""FFPA attention forward examples.

Runs ``ffpa_attn_func`` across the supported forward shape regimes and compares
against ``torch.nn.functional.scaled_dot_product_attention``:

1. Self-Attention            -- Nq == Nkv, Nh_q == Nh_kv, aligned seqlen.
2. Cross / Decode Attention  -- Nq != Nkv (short query, long KV).
3. Grouped-Query Attention   -- Nh_q % Nh_kv == 0, Nh_kv < Nh_q (MQA => Nh_kv=1).
4. Causal Attention          -- causal=True, queries aligned to KV tail.
5. Non-aligned Seqlen        -- N=8191 (not a multiple of Bc=64), exercises
                               cp.async zero-fill + softmax -inf mask +
                               per-row store predicate on the tail tile.

Usage::

    CUDA_VISIBLE_DEVICES=0 python examples/ffpa_attn_fwd.py --forward-backend cuda
    CUDA_VISIBLE_DEVICES=0 python examples/ffpa_attn_fwd.py --forward-backend triton --autotune
"""

from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn.functional as F

from ffpa_attn import ffpa_attn_func

STAGES = 2
WARMUP, ITERS = 2, 10


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="FFPA forward example and SDPA comparison.")
  parser.add_argument(
    "--forward-backend",
    choices=["cuda", "triton"],
    default="cuda",
    help="Forward backend passed to ffpa_attn_func.",
  )
  parser.add_argument("--N", type=int, default=8192, help="Sequence length (non-aligned uses N-1).")
  parser.add_argument("--D", type=int, default=512, help="Head dimension.")
  parser.add_argument("--seed", type=int, default=0, help="Random seed for input tensors.")
  parser.add_argument(
    "--triton-forward-autotune",
    "--autotune",
    "--tune",
    action="store_true",
    help="Enable Triton forward autotuning (only effective when --forward-backend=triton).",
  )
  return parser.parse_args()


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
  forward_backend: str,
  triton_forward_autotune: bool,
  seed: int,
  B: int,
  Nh_q: int,
  Nh_kv: int,
  Nq: int,
  Nkv: int,
  D: int,
  causal: bool = False,
  acc: str = "f32",
) -> None:
  torch.manual_seed(seed)
  q = torch.randn(B, Nh_q, Nq, D, dtype=dtype, device="cuda")
  k = torch.randn(B, Nh_kv, Nkv, D, dtype=dtype, device="cuda")
  v = torch.randn(B, Nh_kv, Nkv, D, dtype=dtype, device="cuda")

  out_ffpa = ffpa_attn_func(
    q,
    k,
    v,
    stages=STAGES,
    acc=acc,
    is_causal=causal,
    enable_gqa=Nh_q != Nh_kv,
    forward_backend=forward_backend,
    triton_forward_autotune=triton_forward_autotune,
  )
  k_ref, v_ref = _expand_kv(k, v, Nh_q)
  out_sdpa = _sdpa_ref(q, k_ref, v_ref, is_causal=causal)

  diff = (out_ffpa.float() - out_sdpa.float()).abs()
  tol = 5e-2 if dtype == torch.bfloat16 else 2e-2
  ok = torch.allclose(out_ffpa, out_sdpa, atol=tol, rtol=tol)

  ms_ffpa = _time_fn(
    lambda q, k, v: ffpa_attn_func(
      q,
      k,
      v,
      stages=STAGES,
      acc=acc,
      is_causal=causal,
      enable_gqa=Nh_q != Nh_kv,
      forward_backend=forward_backend,
      triton_forward_autotune=triton_forward_autotune,
    ),
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
    f"backend={forward_backend}  "
    f"FFPA={ms_ffpa:.2f} ms  SDPA={ms_sdpa:.2f} ms  speedup={ms_sdpa / ms_ffpa:.2f}x"
  )


def main() -> None:
  args = _parse_args()
  print(args)
  N, D = args.N, args.D

  if not torch.cuda.is_available():
    raise SystemExit("CUDA is required to run this example.")

  for dtype in (torch.float16, torch.bfloat16):
    _run_case(
      "self-attn",
      dtype,
      args.forward_backend,
      args.triton_forward_autotune,
      seed=args.seed,
      B=1,
      Nh_q=32,
      Nh_kv=32,
      Nq=N,
      Nkv=N,
      D=D,
    )
    _run_case(
      "cross-attn",
      dtype,
      args.forward_backend,
      args.triton_forward_autotune,
      seed=args.seed,
      B=1,
      Nh_q=32,
      Nh_kv=32,
      Nq=1024,
      Nkv=N,
      D=D,
    )
    _run_case(
      "decode-attn",
      dtype,
      args.forward_backend,
      args.triton_forward_autotune,
      seed=args.seed,
      B=1,
      Nh_q=32,
      Nh_kv=32,
      Nq=1,
      Nkv=N,
      D=D,
    )
    _run_case(
      "gqa",
      dtype,
      args.forward_backend,
      args.triton_forward_autotune,
      seed=args.seed,
      B=1,
      Nh_q=32,
      Nh_kv=8,
      Nq=N,
      Nkv=N,
      D=D,
    )
    _run_case(
      "causal",
      dtype,
      args.forward_backend,
      args.triton_forward_autotune,
      seed=args.seed,
      B=1,
      Nh_q=32,
      Nh_kv=32,
      Nq=N,
      Nkv=N,
      D=D,
      causal=True,
    )
    _run_case(
      "non-aligned",
      dtype,
      args.forward_backend,
      args.triton_forward_autotune,
      seed=args.seed,
      B=1,
      Nh_q=8,
      Nh_kv=8,
      Nq=N - 1 if N > 1 else N,  # avoid zero-dim
      Nkv=N - 1 if N > 1 else N,  # avoid zero-dim
      D=D,
    )


if __name__ == "__main__":
  main()

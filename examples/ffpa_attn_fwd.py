"""FFPA attention forward examples.

Runs ``ffpa_attn_func`` across the supported forward shape regimes and compares
against ``torch.nn.functional.scaled_dot_product_attention``:

1. Self-Attention            -- Nq == Nkv, Nh_q == Nh_kv, aligned seqlen.
2. Cross / Decode Attention  -- Nq != Nkv (short query, long KV).
3. Grouped-Query Attention   -- Nh_q % Nh_kv == 0, Nh_kv < Nh_q (MQA => Nh_kv=1).
4. Causal Attention          -- causal=True, queries aligned to KV tail.
5. Dropout Attention         -- dropout_p > 0, compares against SDPA dropout.
6. Non-aligned Seqlen        -- N=8191 (not a multiple of Bc=64), exercises
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
from typing import Any

import torch
import torch.nn.functional as F

from ffpa_attn import ffpa_attn_func
from attention_flops import attention_fwd_flops, format_tflops_short, tflops_from_ms

STAGES = 2
DEFAULT_WARMUP = 2
DEFAULT_ITERS = 10
FORWARD_RESULT = dict[str, Any]


def _parse_grad_v_dtype(arg: str) -> torch.dtype | None:
  """Parse the CLI grad-v-dtype option.

  :param arg: CLI value, ``"none"`` or ``"fp32"``.
  :return: ``None`` or ``torch.float32``.
  """
  if arg == "none":
    return None
  if arg == "fp32":
    return torch.float32
  raise ValueError(f"Unsupported grad-v-dtype={arg!r}; choose 'none' or 'fp32'.")


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="FFPA forward example and SDPA comparison.")
  parser.add_argument(
    "--forward-backend",
    "--backend",
    "--fwd",
    choices=["cuda", "triton"],
    default="triton",
    help="Forward backend passed to ffpa_attn_func.",
  )
  parser.add_argument("--B", type=int, default=1, help="Batch size.")
  parser.add_argument("--N", type=int, default=8192, help="Sequence length (non-aligned uses N-1).")
  parser.add_argument("--D", type=int, default=512, help="Head dimension.")
  parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP, help="Warmup iterations used for timing.")
  parser.add_argument("--iters", type=int, default=DEFAULT_ITERS, help="Measured iterations used for timing.")
  parser.add_argument("--dropout-p", type=float, default=0.1, help="Dropout probability for the dropout example case.")
  parser.add_argument("--seed", type=int, default=42, help="Random seed for input tensors.")
  parser.add_argument(
    "--norm",
    action="store_true",
    help="Enable pre-attention LayerNorm on q/k/v for both FFPA and SDPA paths.",
  )
  parser.add_argument(
    "--triton-autotune",
    "--autotune",
    "--tune",
    action="store_true",
    help="Enable Triton runtime autotuning (only effective when --forward-backend=triton).",
  )
  parser.add_argument(
    "--triton-autotune-mode",
    "--autotune-mode",
    "--mode",
    choices=["fast", "max"],
    default="fast",
    help="Triton autotune search-space mode.",
  )
  parser.add_argument(
    "--grad-v-storage-dtype",
    "--grad-v-dtype",
    choices=["none", "fp32"],
    default="none",
    help="Optional Triton backward dV storage dtype forwarded to ffpa_attn_func.",
  )
  parser.add_argument(
    "--enable-tma",
    action="store_true",
    help="Enable experimental SM90+ TMA forward path (silently falls back on unsupported devices).",
  )
  parser.add_argument(
    "--enable-ws",
    action="store_true",
    help="Force warp-specialized configs for the experimental SM90+ TMA forward path.",
  )
  return parser.parse_args()


def _validate_timing_args(warmup: int, iters: int) -> None:
  """Validate benchmark timing loop counts.

  :param warmup: Warmup iterations used for timing.
  :param iters: Measured iterations used for timing.
  :raises ValueError: If ``warmup`` is negative or ``iters`` is not positive.
  """
  if warmup < 0:
    raise ValueError(f"warmup must be non-negative, got {warmup}")
  if iters <= 0:
    raise ValueError(f"iters must be positive, got {iters}")


def _sdpa_ref(q, k, v, is_causal: bool = False, attn_mask: torch.Tensor | None = None, dropout_p: float = 0.0):
  return F.scaled_dot_product_attention(
    q,
    k,
    v,
    attn_mask=attn_mask,
    scale=1.0 / math.sqrt(q.size(-1)),
    is_causal=is_causal,
    dropout_p=dropout_p,
  )


def _make_broadcast_additive_attn_mask(nq: int, nkv: int, dtype: torch.dtype, seed: int) -> torch.Tensor:
  """Build a key-position additive attention bias for SDPA/FFPA."""
  torch.manual_seed(seed + 1)
  del nq
  return torch.randn(1, 1, 1, nkv, dtype=dtype, device="cuda") * 0.25


def _time_fn(fn, *args, warmup: int = DEFAULT_WARMUP, iters: int = DEFAULT_ITERS, rng_seed: int | None = None) -> float:
  _validate_timing_args(warmup, iters)
  for _ in range(warmup):
    if rng_seed is not None:
      torch.manual_seed(rng_seed)
    fn(*args)
  torch.cuda.synchronize()
  t0 = time.perf_counter()
  for _ in range(iters):
    if rng_seed is not None:
      torch.manual_seed(rng_seed)
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


def _dtype_tag(dtype: torch.dtype) -> str:
  """Return a short dtype tag.

  :param dtype: Torch dtype.
  :return: String form without the ``torch.`` prefix.
  """
  if dtype == torch.float16:
    return "fp16"
  if dtype == torch.bfloat16:
    return "bf16"
  return str(dtype).replace("torch.", "")


def _resolve_gqa_heads(num_heads: int) -> int:
  """Choose the KV head count used by the GQA example.

  :param num_heads: Query head count.
  :return: KV head count that still divides ``num_heads``.
  """
  if num_heads <= 1:
    return 1
  candidate = max(1, num_heads // 4)
  while candidate > 1 and num_heads % candidate != 0:
    candidate -= 1
  return candidate


def _resolve_non_aligned_heads(num_heads: int) -> int:
  """Choose the head count used by the non-aligned case.

  :param num_heads: Base head count.
  :return: Head count used by the non-aligned example.
  """
  if num_heads <= 8:
    return num_heads
  candidate = max(1, num_heads // 4)
  while candidate > 1 and num_heads % candidate != 0:
    candidate -= 1
  return candidate


def _maybe_norm_qkv(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  apply_norm: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Optionally apply per-tensor LayerNorm over the head dimension.

  :param q: Query tensor.
  :param k: Key tensor.
  :param v: Value tensor.
  :param apply_norm: Whether to normalize q/k/v before attention.
  :return: Normalized or original ``(q, k, v)``.
  """
  if not apply_norm:
    return q, k, v
  q = F.layer_norm(q, (q.size(-1), ))
  k = F.layer_norm(k, (k.size(-1), ))
  v = F.layer_norm(v, (v.size(-1), ))
  return q, k, v


def _max_abs_diff(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
  """Return max abs diff after promoting both tensors to fp32.

  :param lhs: First tensor.
  :param rhs: Second tensor.
  :return: Maximum absolute difference in fp32.
  """
  return (lhs.float() - rhs.float()).abs().max().item()


def _mean_abs_diff(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
  """Return mean abs diff after promoting both tensors to fp32.

  :param lhs: First tensor.
  :param rhs: Second tensor.
  :return: Mean absolute difference in fp32.
  """
  return (lhs.float() - rhs.float()).abs().mean().item()


def _tensor_allclose(lhs: torch.Tensor, rhs: torch.Tensor, tol: float) -> bool:
  """Return whether two tensors are close after promoting both to fp32.

  :param lhs: First tensor.
  :param rhs: Second tensor.
  :param tol: Absolute and relative tolerance.
  :return: ``True`` when the promoted tensors satisfy ``torch.allclose``.
  """
  return torch.allclose(lhs.float(), rhs.float(), atol=tol, rtol=tol)


def _format_forward_result(result: FORWARD_RESULT) -> str:
  """Format one forward benchmark result for CLI output.

  :param result: Structured forward result.
  :return: Human-readable one-line summary.
  """
  return (
    f"[{result['case_name']:<16} {result['dtype']:<8} acc={result['acc']}] "
    f"B={result['B']} Hq={result['Hq']} Hkv={result['Hkv']} "
    f"Nq={result['Nq']} Nkv={result['Nkv']} D={result['D']} "
    f"causal={int(result['causal'])} dropout_p={result['dropout_p']:g}  "
    f"max|diff|={result['max_diff']:.4f}  mean|diff|={result['mean_diff']:.5f}  "
    f"allclose(atol={result['tolerance']})={result['allclose']}  "
    f"backend={result['forward_backend']}  "
    f"tma={int(result.get('enable_tma', False))}  "
    f"ws={int(result.get('enable_ws', False))}  "
    f"FFPA={result['ffpa_ms']:.2f} ms  SDPA={result['sdpa_ms']:.2f} ms  "
    f"TFLOPS={format_tflops_short(result['ffpa_tflops'])}/{format_tflops_short(result['sdpa_tflops'])}  "
    f"speedup={result['speedup']:.2f}x"
  )


def _run_case(
  name: str,
  dtype: torch.dtype,
  forward_backend: str,
  triton_autotune: bool,
  triton_autotune_mode: str,
  seed: int,
  B: int,
  Nh_q: int,
  Nh_kv: int,
  Nq: int,
  Nkv: int,
  D: int,
  causal: bool = False,
  attn_mask: torch.Tensor | None = None,
  dropout_p: float = 0.0,
  acc: str = "f32",
  triton_backward_grad_v_storage_dtype: torch.dtype | None = None,
  apply_norm: bool = False,
  warmup: int = DEFAULT_WARMUP,
  iters: int = DEFAULT_ITERS,
  print_result: bool = True,
  enable_tma: bool = False,
  enable_ws: bool = False,
) -> FORWARD_RESULT:
  torch.manual_seed(seed)
  q = torch.randn(B, Nh_q, Nq, D, dtype=dtype, device="cuda")
  k = torch.randn(B, Nh_kv, Nkv, D, dtype=dtype, device="cuda")
  v = torch.randn(B, Nh_kv, Nkv, D, dtype=dtype, device="cuda")
  q, k, v = _maybe_norm_qkv(q, k, v, apply_norm)

  torch.manual_seed(seed + 17)
  out_ffpa = ffpa_attn_func(
    q,
    k,
    v,
    stages=STAGES,
    acc=acc,
    attn_mask=attn_mask,
    is_causal=causal,
    dropout_p=dropout_p,
    enable_gqa=Nh_q != Nh_kv,
    forward_backend=forward_backend,
    triton_autotune=triton_autotune,
    triton_autotune_mode=triton_autotune_mode,
    triton_backward_grad_v_storage_dtype=triton_backward_grad_v_storage_dtype,
    enable_tma=enable_tma,
    enable_ws=enable_ws,
  )
  k_ref, v_ref = _expand_kv(k, v, Nh_q)
  torch.manual_seed(seed + 17)
  out_sdpa = _sdpa_ref(q, k_ref, v_ref, is_causal=causal, attn_mask=attn_mask, dropout_p=dropout_p)

  tol = 5e-2 if dtype == torch.bfloat16 else 2e-2
  ok = _tensor_allclose(out_ffpa, out_sdpa, tol)

  ms_ffpa = _time_fn(
    lambda q, k, v: ffpa_attn_func(
      q,
      k,
      v,
      stages=STAGES,
      acc=acc,
      attn_mask=attn_mask,
      is_causal=causal,
      dropout_p=dropout_p,
      enable_gqa=Nh_q != Nh_kv,
      forward_backend=forward_backend,
      triton_autotune=triton_autotune,
      triton_autotune_mode=triton_autotune_mode,
      triton_backward_grad_v_storage_dtype=triton_backward_grad_v_storage_dtype,
      enable_tma=enable_tma,
      enable_ws=enable_ws,
    ),
    q,
    k,
    v,
    warmup=warmup,
    iters=iters,
    rng_seed=seed + 17 if dropout_p > 0.0 else None,
  )
  ms_sdpa = _time_fn(
    lambda q, k, v: _sdpa_ref(q, k, v, is_causal=causal, attn_mask=attn_mask, dropout_p=dropout_p),
    q,
    k_ref,
    v_ref,
    warmup=warmup,
    iters=iters,
    rng_seed=seed + 17 if dropout_p > 0.0 else None,
  )
  flop_count = attention_fwd_flops(B, Nh_q, Nq, Nkv, D, causal)

  result: FORWARD_RESULT = {
    "case_name": name,
    "dtype": _dtype_tag(dtype),
    "forward_backend": forward_backend,
    "B": B,
    "Hq": Nh_q,
    "Hkv": Nh_kv,
    "Nq": Nq,
    "Nkv": Nkv,
    "D": D,
    "causal": causal,
    "dropout_p": dropout_p,
    "acc": acc,
    "max_diff": _max_abs_diff(out_ffpa, out_sdpa),
    "mean_diff": _mean_abs_diff(out_ffpa, out_sdpa),
    "allclose": ok,
    "tolerance": tol,
    "warmup": warmup,
    "iters": iters,
    "ffpa_ms": ms_ffpa,
    "sdpa_ms": ms_sdpa,
    "ffpa_tflops": tflops_from_ms(flop_count, ms_ffpa),
    "sdpa_tflops": tflops_from_ms(flop_count, ms_sdpa),
    "speedup": ms_sdpa / ms_ffpa,
    "enable_tma": enable_tma,
    "enable_ws": enable_ws,
  }
  if print_result:
    print(_format_forward_result(result))
  return result


def run_forward_examples(
  *,
  B: int = 1,
  H: int = 32,
  N: int = 8192,
  D: int = 512,
  dropout_p: float = 0.1,
  seed: int = 42,
  apply_norm: bool = False,
  forward_backend: str = "triton",
  triton_autotune: bool = False,
  triton_autotune_mode: str = "fast",
  triton_backward_grad_v_storage_dtype: torch.dtype | None = None,
  warmup: int = DEFAULT_WARMUP,
  iters: int = DEFAULT_ITERS,
  print_results: bool = True,
  enable_tma: bool = False,
  enable_ws: bool = False,
  tasks: set[str] | None = None,
) -> list[FORWARD_RESULT]:
  """Run the canonical forward benchmark cases.

  :param B: Batch size.
  :param H: Base query head count used by the examples.
  :param N: Base sequence length.
  :param D: Head dimension.
  :param dropout_p: Dropout probability for the dropout case.
  :param seed: RNG seed.
  :param apply_norm: Whether to normalize q/k/v before attention.
  :param forward_backend: Forward backend passed to ``ffpa_attn_func``.
  :param triton_autotune: Whether to enable Triton runtime autotune.
  :param triton_autotune_mode: Triton autotune mode.
  :param triton_backward_grad_v_storage_dtype: Optional Triton backward dV
    storage dtype forwarded to ``ffpa_attn_func``.
  :param warmup: Warmup iterations used for timing.
  :param iters: Measured iterations used for timing.
  :param print_results: Whether to print each case result.
  :param enable_tma: Whether to enable the SM90+ TMA forward path.
  :param enable_ws: Whether to force warp-specialized SM90 TMA configs.
  :param tasks: Optional case-name filter. ``None`` runs all cases.
  :return: One structured result per executed case and dtype.
  """
  _validate_timing_args(warmup, iters)
  results: list[FORWARD_RESULT] = []
  gqa_heads = _resolve_gqa_heads(H)
  non_aligned_heads = _resolve_non_aligned_heads(H)
  print(
    f"\nRunning FFPA forward examples with forward_backend={forward_backend}, "
    f"apply_norm={apply_norm}, "
    f"triton_autotune={triton_autotune}, "
    f"triton_autotune_mode={triton_autotune_mode}, "
    f"triton_backward_grad_v_storage_dtype={triton_backward_grad_v_storage_dtype}, "
    f"enable_tma={enable_tma}, "
    f"enable_ws={enable_ws}, "
    f"tasks={sorted(tasks) if tasks is not None else 'full'}, "
    f"warmup={warmup}, iters={iters}"
  )

  for dtype in (torch.float16, torch.bfloat16):
    case_specs: list[dict[str, Any]] = [
      {
        "name": "self-attn",
        "Nh_q": H,
        "Nh_kv": H,
        "Nq": N,
        "Nkv": N
      },
      {
        "name": "cross-attn",
        "Nh_q": H,
        "Nh_kv": H,
        "Nq": 1024,
        "Nkv": N
      },
      {
        "name": "decode-attn",
        "Nh_q": H,
        "Nh_kv": H,
        "Nq": 1,
        "Nkv": N
      },
      {
        "name": "gqa",
        "Nh_q": H,
        "Nh_kv": gqa_heads,
        "Nq": N,
        "Nkv": N
      },
      {
        "name": "causal",
        "Nh_q": H,
        "Nh_kv": H,
        "Nq": N,
        "Nkv": N,
        "causal": True
      },
    ]
    if forward_backend != "cuda":
      mask_n = max(N, 512)
      case_specs.extend([
        {
          "name": "attn-mask",
          "Nh_q": H,
          "Nh_kv": H,
          "Nq": mask_n,
          "Nkv": mask_n,
          "attn_mask": _make_broadcast_additive_attn_mask(mask_n, mask_n, dtype, seed),
        },
        {
          "name": "dropout",
          "Nh_q": H,
          "Nh_kv": H,
          "Nq": N,
          "Nkv": N,
          "dropout_p": dropout_p,
        },
      ])
    case_specs.append({
      "name": "non-aligned",
      "Nh_q": non_aligned_heads,
      "Nh_kv": non_aligned_heads,
      "Nq": N - 1 if N > 1 else N,
      "Nkv": N - 1 if N > 1 else N,
    })
    if tasks is not None:
      case_specs = [case for case in case_specs if case["name"] in tasks]

    for case in case_specs:
      results.append(
        _run_case(
          case["name"],
          dtype,
          forward_backend,
          triton_autotune,
          triton_autotune_mode,
          seed=seed,
          B=B,
          Nh_q=case["Nh_q"],
          Nh_kv=case["Nh_kv"],
          Nq=case["Nq"],
          Nkv=case["Nkv"],
          D=D,
          causal=case.get("causal", False),
          attn_mask=case.get("attn_mask"),
          dropout_p=case.get("dropout_p", 0.0),
          apply_norm=apply_norm,
          triton_backward_grad_v_storage_dtype=triton_backward_grad_v_storage_dtype,
          warmup=warmup,
          iters=iters,
          print_result=print_results,
          enable_tma=enable_tma,
          enable_ws=enable_ws,
        )
      )

  return results


def main() -> None:
  args = _parse_args()
  print(args)

  if not torch.cuda.is_available():
    raise SystemExit("CUDA is required to run this example.")
  grad_v_dtype = _parse_grad_v_dtype(args.grad_v_storage_dtype)
  run_forward_examples(
    B=args.B,
    N=args.N,
    D=args.D,
    dropout_p=args.dropout_p,
    seed=args.seed,
    apply_norm=args.norm,
    forward_backend=args.forward_backend,
    triton_autotune=args.triton_autotune,
    triton_autotune_mode=args.triton_autotune_mode,
    triton_backward_grad_v_storage_dtype=grad_v_dtype,
    warmup=args.warmup,
    iters=args.iters,
    print_results=True,
    enable_tma=args.enable_tma,
    enable_ws=args.enable_ws,
  )


if __name__ == "__main__":
  main()

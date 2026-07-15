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
"""

from __future__ import annotations

import math
import os
import time
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from ffpa_attn import CUDABackend, CuTeDSLBackend, TritonBackend, ffpa_attn_func
from ._flops import attention_fwd_flops, format_tflops_short, tflops_from_ms

DEFAULT_WARMUP = 2
DEFAULT_ITERS = 10
FORWARD_RESULT = dict[str, Any]
TRITON_SMALL_D_ENV = "FFPA_TRITON_ALLOW_SMALL_D"
CUTEDSL_SMALL_D_ENV = "FFPA_CUTE_ALLOW_SMALL_D"


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


def _sdpa_ref(
  q,
  k,
  v,
  is_causal: bool = False,
  attn_mask: torch.Tensor | None = None,
  dropout_p: float = 0.0
):
  sdpa_kwargs = {
    "attn_mask": attn_mask,
    "scale": 1.0 / math.sqrt(q.size(-1)),
    "is_causal": is_causal,
    "dropout_p": dropout_p,
  }
  if dropout_p > 0.0:
    # Triton dropout parity is intentionally aligned to SDPA's efficient
    # attention backend. The default SDPA dispatcher may pick flash-attention
    # for D<=256, which uses a different RNG stream and breaks apples-to-apples
    # dropout comparisons in the examples/perf benchmark.
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
      return F.scaled_dot_product_attention(q, k, v, **sdpa_kwargs)
  return F.scaled_dot_product_attention(q, k, v, **sdpa_kwargs)


def _make_broadcast_additive_attn_mask(
  nq: int, nkv: int, dtype: torch.dtype, seed: int
) -> torch.Tensor:
  """Build a key-position additive attention bias for SDPA/FFPA."""
  torch.manual_seed(seed + 1)
  del nq
  return torch.randn(1, 1, 1, nkv, dtype=dtype, device="cuda") * 0.25


def _time_fn(
  fn,
  *args,
  warmup: int = DEFAULT_WARMUP,
  iters: int = DEFAULT_ITERS,
  rng_seed: int | None = None
) -> float:
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


def _format_forward_result(
  result: FORWARD_RESULT, *, verbose: bool = False
) -> str:
  """Format one forward benchmark result for CLI output.

  :param result: Structured forward result.
  :param verbose: If True, include accuracy fields (O_err, allclose).
  :return: Human-readable one-line summary.
  """
  ffpa_t = format_tflops_short(result["ffpa_tflops"])
  sdpa_t = format_tflops_short(result["sdpa_tflops"])
  tflops_str = f"{ffpa_t}/{sdpa_t}"
  if not verbose:
    return (
      f"[{result['case_name']:<12} {result['dtype']:>4}] "
      f"B={result['B']:<1} Hq={result['Hq']:<2} Hkv={result['Hkv']:<2} "
      f"Nq={result['Nq']:<5} Nkv={result['Nkv']:<5} D={result['D']:<3} "
      f"FFPA={result['ffpa_ms']:<6.2f}ms SDPA={result['sdpa_ms']:<6.2f}ms "
      f"TFLOPS={tflops_str:<9} "
      f"🎉{result['speedup']:<4.2f}x"
    )
  return (
    f"[{result['case_name']:<12} {result['dtype']:>4}] "
    f"B={result['B']:<1} Hq={result['Hq']:<2} Hkv={result['Hkv']:<2} "
    f"Nq={result['Nq']:<5} Nkv={result['Nkv']:<5} D={result['D']:<3} "
    f"O_err={result['max_diff']:<9.4f} "
    f"allclose(atol={result['tolerance']:.2f})={str(result['allclose']):<5} "
    f"FFPA={result['ffpa_ms']:<6.2f}ms SDPA={result['sdpa_ms']:<6.2f}ms "
    f"TFLOPS={tflops_str:<9} "
    f"🎉{result['speedup']:<4.2f}x"
  )


def _make_forward_backend(
  name: str,
  *,
  acc: str,
  triton_autotune: bool,
  triton_autotune_mode: str,
  enable_tma: bool,
  enable_ws: bool,
):
  if name == "cuda":
    return CUDABackend(forward=True, acc=acc)
  if name == "triton":
    return TritonBackend(
      forward=True,
      autotune=triton_autotune,
      autotune_mode=triton_autotune_mode,
      enable_tma=enable_tma,
      enable_ws=enable_ws,
    )
  if name == "cutedsl":
    return CuTeDSLBackend(forward=True)
  raise ValueError(f"Unsupported forward_backend={name!r}")


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
  apply_norm: bool = False,
  warmup: int = DEFAULT_WARMUP,
  iters: int = DEFAULT_ITERS,
  print_result: bool = True,
  enable_tma: bool = False,
  enable_ws: bool = False,
  verbose: bool = False,
) -> FORWARD_RESULT:
  torch.manual_seed(seed)
  q = torch.randn(B, Nh_q, Nq, D, dtype=dtype, device="cuda")
  k = torch.randn(B, Nh_kv, Nkv, D, dtype=dtype, device="cuda")
  v = torch.randn(B, Nh_kv, Nkv, D, dtype=dtype, device="cuda")
  q, k, v = _maybe_norm_qkv(q, k, v, apply_norm)
  forward_backend = _make_forward_backend(
    forward_backend,
    acc=acc,
    triton_autotune=triton_autotune,
    triton_autotune_mode=triton_autotune_mode,
    enable_tma=enable_tma,
    enable_ws=enable_ws,
  )
  backward_backend = CuTeDSLBackend(
    backward=True
  ) if forward_backend.name == "cutedsl" else None

  torch.manual_seed(seed + 17)
  out_ffpa = ffpa_attn_func(
    q,
    k,
    v,
    attn_mask=attn_mask,
    is_causal=causal,
    dropout_p=dropout_p,
    enable_gqa=Nh_q != Nh_kv,
    forward_backend=forward_backend,
    backward_backend=backward_backend,
  )
  k_ref, v_ref = _expand_kv(k, v, Nh_q)
  torch.manual_seed(seed + 17)
  out_sdpa = _sdpa_ref(
    q, k_ref, v_ref, is_causal=causal, attn_mask=attn_mask, dropout_p=dropout_p
  )

  tol = 5e-2 if dtype == torch.bfloat16 else 2e-2
  ok = _tensor_allclose(out_ffpa, out_sdpa, tol)

  ms_ffpa = _time_fn(
    lambda q, k, v: ffpa_attn_func(
      q,
      k,
      v,
      attn_mask=attn_mask,
      is_causal=causal,
      dropout_p=dropout_p,
      enable_gqa=Nh_q != Nh_kv,
      forward_backend=forward_backend,
      backward_backend=backward_backend,
    ),
    q,
    k,
    v,
    warmup=warmup,
    iters=iters,
    rng_seed=seed + 17 if dropout_p > 0.0 else None,
  )
  ms_sdpa = _time_fn(
    lambda q, k, v: _sdpa_ref(
      q, k, v, is_causal=causal, attn_mask=attn_mask, dropout_p=dropout_p
    ),
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
    "forward_backend": forward_backend.name,
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
    print(_format_forward_result(result, verbose=verbose))
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
  grad_kv_storage_dtype: torch.dtype | None = None,
  warmup: int = DEFAULT_WARMUP,
  iters: int = DEFAULT_ITERS,
  print_results: bool = True,
  enable_tma: bool = False,
  enable_ws: bool = False,
  tasks: set[str] | None = None,
  dtypes: tuple[torch.dtype, ...] = (torch.float16, torch.bfloat16),
  verbose: bool = False,
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
  :param grad_kv_storage_dtype: Optional backward dK/dV (Triton or CuTeDSL)
    storage dtype forwarded to ``ffpa_attn_func``.
  :param warmup: Warmup iterations used for timing.
  :param iters: Measured iterations used for timing.
  :param print_results: Whether to print each case result.
  :param enable_tma: Whether to enable the SM90+ TMA forward path.
  :param enable_ws: Whether to force warp-specialized SM90 TMA configs.
  :param tasks: Optional case-name filter. ``None`` runs all cases.
  :param dtypes: Activation dtypes iterated for each case.
  :return: One structured result per executed case and dtype.
  """
  _validate_timing_args(warmup, iters)
  results: list[FORWARD_RESULT] = []
  gqa_heads = _resolve_gqa_heads(H)
  non_aligned_heads = _resolve_non_aligned_heads(H)
  tasks_str = ",".join(
    sorted(tasks)
  ) if tasks is not None else "self-attn,cross-attn,decode-attn,gqa,causal,attn-mask,dropout,non-aligned"
  config_items: list[tuple[str, str]] = [
    ("backend", forward_backend),
    ("apply_norm", str(apply_norm)),
    ("triton_autotune", str(triton_autotune)),
    ("triton_autotune_mode", triton_autotune_mode),
    ("grad_kv_storage_dtype", str(grad_kv_storage_dtype)),
    ("enable_fwd_tma", str(enable_tma)),
    ("enable_fwd_ws", str(enable_ws)),
    ("tasks", tasks_str),
    ("warmup", str(warmup)),
    ("iters", str(iters)),
  ]
  key_w = max(len(k) for k, _ in config_items)
  val_w = max(len(v) for _, v in config_items)
  _backend_label = {"cuda": "CUDA", "triton": "Triton", "cutedsl": "CuTeDSL"}
  title = f"FFPA Forward ({_backend_label.get(forward_backend, forward_backend)})"
  title_w = max(key_w + val_w + 3, len(title))
  bar = "+" + "=" * (title_w + 2) + "+"
  print(f"\n{bar}")
  print(f"| {title:^{title_w}} |")
  print(bar)
  for key, val in config_items:
    print(f"| {key:<{key_w}} | {val:<{val_w}} |")
  print(bar)
  if forward_backend == "triton" and D <= 256:
    triton_small_d_enabled = bool(int(os.environ.get(TRITON_SMALL_D_ENV, "0")))
    print(
      f"[Triton] D <= 256 {'runs FFPA Triton' if triton_small_d_enabled else 'stays on the SDPA fallback path'} "
      f"when {TRITON_SMALL_D_ENV}={'1' if triton_small_d_enabled else '0'}."
    )
  if forward_backend == "cutedsl":
    cutedsl_small_d_enabled = bool(
      int(os.environ.get(CUTEDSL_SMALL_D_ENV, "0"))
    )
    if D < 320:
      print(
        f"[CuTeDSL] D < 320 {'runs the SM80 Split-D fallback path' if cutedsl_small_d_enabled else 'stays on the SDPA fallback path'} "
        f"when {CUTEDSL_SMALL_D_ENV}={'1' if cutedsl_small_d_enabled else '0'}."
      )
    else:
      print(
        "[CuTeDSL] backend constraints in effect: SM8x/SM90 dense path, no attn_mask/dropout."
      )

  for dtype in dtypes:
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
    if forward_backend != "cutedsl":
      mask_n = max(N, 512)
      case_specs.append({
        "name":
        "attn-mask",
        "Nh_q":
        H,
        "Nh_kv":
        H,
        "Nq":
        mask_n,
        "Nkv":
        mask_n,
        "attn_mask":
        _make_broadcast_additive_attn_mask(mask_n, mask_n, dtype, seed),
      })
    if forward_backend != "cutedsl":
      case_specs.extend([
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
          warmup=warmup,
          iters=iters,
          print_result=print_results,
          enable_tma=enable_tma,
          enable_ws=enable_ws,
          verbose=verbose,
        )
      )

  return results

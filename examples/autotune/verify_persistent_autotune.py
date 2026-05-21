"""Verify persistent tuned configs against online Triton fast autotune.

Usage::

  CUDA_VISIBLE_DEVICES=7 python examples/verify_persistent_autotune.py --case decode-attn --direction backward --dtype fp16
"""

from __future__ import annotations

import argparse
import math
from typing import Any

import torch

from ffpa_attn import TritonBackend, ffpa_attn_func
from ffpa_attn.triton._ffpa_bwd import _get_bwd_autotune, _get_decode_bwd_stage1_autotune, _get_pre_autotune
from ffpa_attn.triton._ffpa_bwd_sm90 import _get_bwd_sm90_autotune, is_sm90_tma_backward_supported
from ffpa_attn.triton._ffpa_fwd import _get_decode_fwd_stage1_autotune, _get_decode_num_splits, _get_fwd_autotune
from ffpa_attn.triton._ffpa_fwd_sm90 import _get_fwd_sm90_autotune, is_sm90_tma_forward_supported
from ffpa_attn.triton._persistent_autotune import (
  PersistentConfigRequest,
  clear_config_cache,
  config_from_triton_config,
  dtype_name,
  lookup_persistent_config,
)


def _parse_args() -> argparse.Namespace:
  """Parse CLI arguments.

  :return: Parsed arguments.
  """
  parser = argparse.ArgumentParser(
    description=
    "Compare online fast autotune config with persistent lookup config."
  )
  parser.add_argument(
    "--direction", choices=("forward", "backward"), default="backward"
  )
  parser.add_argument(
    "--case",
    choices=("self-attn", "cross-attn", "decode-attn", "non-aligned"),
    default="decode-attn"
  )
  parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
  parser.add_argument("--B", type=int, default=1)
  parser.add_argument("--H", type=int, default=32)
  parser.add_argument("--N", type=int, default=8192)
  parser.add_argument("--D", type=int, default=512)
  parser.add_argument("--mode", choices=("fast", "max"), default="fast")
  parser.add_argument(
    "--enable-tma",
    action="store_true",
    help="Compatibility alias for both TMA directions."
  )
  parser.add_argument(
    "--enable-ws",
    action="store_true",
    help="Compatibility alias for both WS directions."
  )
  parser.add_argument(
    "--enable-fwd-tma",
    action="store_true",
    help="Verify the SM90+ TMA forward path."
  )
  parser.add_argument(
    "--enable-bwd-tma",
    action="store_true",
    help="Verify the SM90+ TMA backward path."
  )
  parser.add_argument(
    "--enable-fwd-ws",
    action="store_true",
    help="Verify warp-specialized SM90+ TMA forward configs."
  )
  parser.add_argument(
    "--enable-bwd-ws",
    action="store_true",
    help="Verify warp-specialized SM90+ TMA backward configs."
  )
  args = parser.parse_args()
  if args.enable_tma:
    args.enable_fwd_tma = True
    args.enable_bwd_tma = True
  if args.enable_ws:
    args.enable_fwd_ws = True
    args.enable_bwd_ws = True
  return args


def _case_shape(case_name: str, heads: int,
                seqlen: int) -> tuple[int, int, int, int]:
  """Return ``(Hq, Hkv, Nq, Nkv)`` for one canonical benchmark case.

  :param case_name: Benchmark case name.
  :param heads: Base query-head count.
  :param seqlen: Base sequence length.
  :return: Query/KV head and sequence lengths.
  """
  if case_name == "cross-attn":
    return heads, heads, 1024, seqlen
  if case_name == "decode-attn":
    return heads, heads, 1, seqlen
  if case_name == "non-aligned":
    non_aligned_heads = heads if heads <= 8 else max(1, heads // 4)
    non_aligned_seqlen = seqlen - 1 if seqlen > 1 else seqlen
    return non_aligned_heads, non_aligned_heads, non_aligned_seqlen, non_aligned_seqlen
  return heads, heads, seqlen, seqlen


def _same_config(
  lhs: dict[str, Any] | None, rhs: dict[str, Any] | None
) -> bool:
  """Return whether two config dictionaries match exactly."""
  return lhs == rhs


def _print_comparison(
  name: str, online: dict[str, Any] | None, persistent: dict[str, Any] | None
) -> bool:
  """Print one config comparison and return whether it matched."""
  matched = _same_config(online, persistent)
  print(f"{name}: {'MATCH' if matched else 'MISMATCH'}")
  print(f"  online    = {online}")
  print(f"  persistent= {persistent}")
  return matched


def _run_forward(
  args: argparse.Namespace, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> list[bool]:
  """Run online forward autotune and compare persistent lookup."""
  out = ffpa_attn_func(
    q,
    k,
    v,
    scale=1 / math.sqrt(args.D),
    forward_backend=TritonBackend(
      forward=True,
      autotune=True,
      autotune_mode=args.mode,
      enable_tma=args.enable_fwd_tma,
      enable_ws=args.enable_fwd_ws,
    ),
  )
  del out
  torch.cuda.synchronize()

  dtype = dtype_name(q.dtype)
  _, nheads_q, seqlen_q, headdim = q.shape
  _, nheads_kv, seqlen_k, _ = k.shape
  num_splits = _get_decode_num_splits(
    seqlen_q, seqlen_k, headdim, q.size(0), nheads_q, q.device
  )
  use_sm90_tma = args.enable_fwd_tma and is_sm90_tma_forward_supported(
    q, k, v, torch.empty_like(q), num_splits=num_splits
  )
  if use_sm90_tma:
    kernel = "fwd_sm90_generic"
    online = config_from_triton_config(
      _get_fwd_sm90_autotune(
        headdim, args.mode, dtype, enable_ws=args.enable_fwd_ws
      ).best_config
    )
    use_gemv = None
  elif num_splits == 1:
    kernel = "fwd_generic"
    online = config_from_triton_config(
      _get_fwd_autotune(headdim, args.mode, dtype).best_config
    )
    use_gemv = None
  else:
    kernel = "decode_fwd_stage1"
    use_gemv = seqlen_q == 1
    online = config_from_triton_config(
      _get_decode_fwd_stage1_autotune(headdim, use_gemv, args.mode,
                                      dtype).best_config
    )
  persistent = lookup_persistent_config(
    PersistentConfigRequest(
      direction="forward",
      kernel=kernel,
      autotune_mode=args.mode,
      dtype=dtype,
      headdim=headdim,
      seqlen_q=seqlen_q,
      seqlen_k=seqlen_k,
      causal=False,
      use_gemv=use_gemv,
      has_attn_bias=False,
      has_dropout=False,
      enable_tma=True if use_sm90_tma else False,
      enable_ws=args.enable_fwd_ws if use_sm90_tma else False,
      nheads_q=nheads_q,
      nheads_kv=nheads_kv,
      device_index=q.device.index,
    )
  )
  return [_print_comparison(kernel, online, persistent)]


def _run_backward(
  args: argparse.Namespace, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> list[bool]:
  """Run online backward autotune and compare persistent lookup."""
  out = ffpa_attn_func(
    q,
    k,
    v,
    scale=1 / math.sqrt(args.D),
    forward_backend=TritonBackend(
      forward=True,
      autotune=True,
      autotune_mode=args.mode,
    ),
    backward_backend=TritonBackend(
      forward=False,
      backward=True,
      autotune=True,
      autotune_mode=args.mode,
      enable_tma=args.enable_bwd_tma,
      enable_ws=args.enable_bwd_ws,
    ),
  )
  out.sum().backward()
  torch.cuda.synchronize()

  dtype = dtype_name(q.dtype)
  _, nheads_q, seqlen_q, headdim = q.shape
  _, nheads_kv, seqlen_k, _ = k.shape
  pre_online = config_from_triton_config(
    _get_pre_autotune(False, args.mode, dtype).best_config
  )
  pre_persistent = lookup_persistent_config(
    PersistentConfigRequest(
      direction="backward",
      kernel="bwd_preproc",
      autotune_mode=args.mode,
      dtype=dtype,
      headdim=headdim,
      seqlen_q=seqlen_q,
      preprocess_d_chunk=False,
      device_index=q.device.index,
    )
  )
  use_sm90_tma = args.enable_bwd_tma and seqlen_q >= 8 and is_sm90_tma_backward_supported(
    q,
    k,
    v,
    q,
    q,
    k,
    v,
    seqlen_q=seqlen_q,
  )
  if seqlen_q < 8:
    kernel = "decode_bwd_stage1"
    use_gemv = seqlen_q == 1
    online = config_from_triton_config(
      _get_decode_bwd_stage1_autotune(headdim, use_gemv, args.mode,
                                      False).best_config
    )
  elif use_sm90_tma:
    kernel = "bwd_sm90_generic"
    use_gemv = None
    online = config_from_triton_config(
      _get_bwd_sm90_autotune(
        headdim, args.mode, dtype, False, enable_ws=args.enable_bwd_ws
      ).best_config
    )
  else:
    kernel = "bwd_generic"
    use_gemv = None
    online = config_from_triton_config(
      _get_bwd_autotune(headdim, args.mode, False).best_config
    )
  persistent = lookup_persistent_config(
    PersistentConfigRequest(
      direction="backward",
      kernel=kernel,
      autotune_mode=args.mode,
      dtype=dtype,
      headdim=headdim,
      seqlen_q=seqlen_q,
      seqlen_k=seqlen_k,
      causal=False,
      bias_grad=False,
      grad_kv_storage_dtype=None,
      use_gemv=use_gemv,
      has_attn_bias=False,
      has_dropout=False,
      enable_tma=True if use_sm90_tma else False,
      enable_ws=args.enable_bwd_ws if use_sm90_tma else False,
      nheads_q=nheads_q,
      nheads_kv=nheads_kv,
      device_index=q.device.index,
    )
  )
  return [
    _print_comparison("bwd_preproc", pre_online, pre_persistent),
    _print_comparison(kernel, online, persistent),
  ]


def main() -> int:
  """Run the verifier.

  :return: Process exit code.
  """
  args = _parse_args()
  if not torch.cuda.is_available():
    raise SystemExit("CUDA is required")
  dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
  hq, hkv, nq, nkv = _case_shape(args.case, args.H, args.N)
  torch.manual_seed(42)
  q = torch.randn(
    args.B,
    hq,
    nq,
    args.D,
    device="cuda",
    dtype=dtype,
    requires_grad=args.direction == "backward"
  )
  k = torch.randn(
    args.B,
    hkv,
    nkv,
    args.D,
    device="cuda",
    dtype=dtype,
    requires_grad=args.direction == "backward"
  )
  v = torch.randn(
    args.B,
    hkv,
    nkv,
    args.D,
    device="cuda",
    dtype=dtype,
    requires_grad=args.direction == "backward"
  )
  clear_config_cache()
  results = _run_forward(
    args, q, k, v
  ) if args.direction == "forward" else _run_backward(args, q, k, v)
  return 0 if all(results) else 1


if __name__ == "__main__":
  raise SystemExit(main())

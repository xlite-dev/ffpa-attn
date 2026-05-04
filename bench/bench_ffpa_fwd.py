import argparse
import math
import random
import sys
import time
from typing import Optional

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
try:
  from flash_attn import flash_attn_func
  has_flash_attn = True
except ImportError:
  flash_attn_func = None
  has_flash_attn = False

sys.path.append("../")
from env import ENV, pretty_print_line

torch.set_grad_enabled(False)
torch.set_printoptions(precision=6, threshold=8, edgeitems=3, linewidth=120, sci_mode=False)


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--no-rand-q", "--no-rq", action="store_true")
  parser.add_argument("--no-rand-k", "--no-rk", action="store_true")
  parser.add_argument("--no-rand-v", "--no-rv", action="store_true")
  parser.add_argument("--no-rand-qkv", "--no-rqkv", action="store_true")
  parser.add_argument("--run-torch-unfused", "--torch", action="store_true")
  parser.add_argument("--run-flash-attn", "--flash", action="store_true")
  parser.add_argument("--check", action="store_true")
  parser.add_argument("--check-all", action="store_true")
  parser.add_argument("--show-all", "--show", action="store_true")
  parser.add_argument("--show-less", "--show-l", action="store_true")
  parser.add_argument("--show-matrix", action="store_true")
  parser.add_argument("--only-flops-matmul", "--flops-mm", action="store_true")
  parser.add_argument("--B", type=int, default=None)
  parser.add_argument("--H", type=int, default=None)
  parser.add_argument("--N", type=int, default=None)
  parser.add_argument("--D", type=int, default=None)
  parser.add_argument("--MAX-D", "--MD", type=int, default=1024)
  parser.add_argument("--seed", type=int, default=None)
  parser.add_argument("--sleep", type=float, default=0.05)
  parser.add_argument("--debug", action="store_true")
  parser.add_argument("--verbose", "--v", action="store_true")
  parser.add_argument("--warmup", "--w", type=int, default=1)
  parser.add_argument("--iters", "--i", type=int, default=5)
  parser.add_argument("--tag-hints", "--tags", "--hints", type=str, default=None)
  parser.add_argument("--plot-flops", "--plot", action="store_true", help="Plot TFLOPS")
  parser.add_argument("--save-dir", "--dir", type=str, default="tmp", help="Save dir for plot")
  parser.add_argument("--save-tag", "--tag", type=str, default=None, help="Save name for plot")
  parser.add_argument("--gen-bench-table", "--gen-bench", action="store_true")
  parser.add_argument("--force-build", "--build", action="store_true", help="Force build from sources")
  parser.add_argument(
    "--dtype",
    choices=["fp16", "bf16"],
    default="fp16",
    help="Activation dtype. bf16 forces MMA acc=f32 (no bf16-acc mma PTX)."
  )

  return parser.parse_args()


args = get_args()
ENV.list_ffpa_env()


def set_rand_seed(seed: int = 1):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)


pretty_print_line()
print(args)
pretty_print_line()

# Load the CUDA kernel as a python module
ffpa_attn, use_ffpa_attn_package = ENV.try_load_ffpa_library(force_build=args.force_build, verbose=args.verbose)
if use_ffpa_attn_package:
  import ffpa_attn  # noqa: F401

# The sole public Python API is ``ffpa_attn_func``: it dispatches by
# ``Q.dtype`` + ``acc`` through a registered torch op. Bind two local
# ``partial`` views to keep the call sites concise. Under bf16 there is
# no bf16-acc mma PTX, so the "f16-acc" slot is routed to ``acc='f32'``.
from functools import partial as _partial
from ffpa_attn import ffpa_attn_func

if args.dtype == "bf16":
  ffpa_f16_acc = _partial(ffpa_attn_func, acc="f32")
  ffpa_f32_acc = _partial(ffpa_attn_func, acc="f32")
else:
  ffpa_f16_acc = _partial(ffpa_attn_func, acc="f16")
  ffpa_f32_acc = _partial(ffpa_attn_func, acc="f32")


def get_mha_tflops(B: int, H: int, N: int, D: int, secs: float = 1.0, only_matmul: bool = False):
  flops_qk = B * H * N * N * (2 * D - 1)
  flops_scaling = B * H * N * N
  flops_row_max = B * H * N * (N - 1)
  flops_subtract_max = B * H * N * N
  flops_exp = B * H * N * N
  flops_row_sum = B * H * N * (N - 1)
  flops_normalization = B * H * N * N
  flops_safe_softmax = (flops_row_max + flops_subtract_max + flops_exp + flops_row_sum + flops_normalization)
  flops_pv = B * H * N * D * (2 * N - 1)
  total_flops = flops_qk + flops_scaling + flops_safe_softmax + flops_pv
  if only_matmul:
    total_flops = flops_qk + flops_pv
  tflops = total_flops * 1e-12 / (secs)
  return tflops


MAX_TFLOPS = -1
STATIS_INFO: dict[str, list[float | int] | set] = {}
STATIS_INFO["headdim"] = set()
TOATL_TFLOPS: dict[str, float] = {}
SDPA_TFLOPS = -1


def run_benchmark(
  perf_func: callable,
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  tag: str,
  out: Optional[torch.Tensor] = None,
  s: Optional[torch.Tensor] = None,
  stages: int = -1,
  warmup: int = args.warmup,
  iters: int = args.iters,
  show_matrix: bool = args.show_matrix,
  only_show_improved: bool = not args.show_all,
):

  global MAX_TFLOPS
  global MAX_HEADDIM_CFG
  global SDPA_TFLOPS

  tag_hints: str = args.tag_hints
  if tag_hints:
    tag_hints: list = tag_hints.strip().split(",")
    tag_hints.append("sdpa")
    tag_hints.append("unfused")
    hit_hints = False
    for hint in tag_hints:
      if hint in tag:
        hit_hints = True
    if not hit_hints:
      return None, None

  B, H, N, D = q.size()
  if "flash" in tag:
    B, N, H, D = q.size()

  if "unfused" in tag and (not args.run_torch_unfused):
    return None, None
  if "flash" in tag and ((not args.run_flash_attn) or (not has_flash_attn) or (D > 256)):
    return None, None

  STATIS_INFO["headdim"].add(D)

  max_supported_D = MAX_HEADDIM_CFG.get(tag, None)
  if max_supported_D is not None:
    if D > max_supported_D:
      return None, None

  if out is not None:
    out.fill_(0)
  if s is not None:
    s.fill_(0)
  if out is not None:
    for i in range(warmup):
      if stages >= 1:
        if s is not None:
          perf_func(q, k, v, out, s, stages)
        else:
          perf_func(q, k, v, out, stages=stages)
      else:
        perf_func(q, k, v, out)
  else:
    for i in range(warmup):
      _ = perf_func(q, k, v)

  torch.cuda.synchronize()
  start = time.time()
  if out is not None:
    for i in range(iters):
      if stages >= 1:
        if s is not None:
          perf_func(q, k, v, out, s, stages)
        else:
          perf_func(q, k, v, out, stages=stages)
      else:
        perf_func(q, k, v, out)
  else:
    for i in range(iters):
      out = perf_func(q, k, v)
  torch.cuda.synchronize()
  end = time.time()
  total_secs = end - start
  total_time = (end - start) * 1000
  mean_time = total_time / iters
  mean_secs = total_secs / iters

  TFLOPS = get_mha_tflops(B, H, N, D, mean_secs, only_matmul=args.only_flops_matmul)
  if tag in STATIS_INFO:
    STATIS_INFO[tag].append(int(round(TFLOPS)))
  else:
    STATIS_INFO[tag] = []
    STATIS_INFO[tag].append(int(round(TFLOPS)))

  if "sdpa" in tag:
    SDPA_TFLOPS = TFLOPS
  out_info = f"{tag}"
  out_val_first = out.flatten()[:3].detach().float().cpu().numpy().tolist()
  out_val_last = out.flatten()[-3:].detach().float().cpu().numpy().tolist()
  out_val_first = [round(v, 8) for v in out_val_first]
  out_val_last = [round(v, 8) for v in out_val_last]
  if not args.show_less:
    out_val = out_val_first[:2]
    out_val.append(out_val_last[-1])
  else:
    out_val = out_val_first[:1]
  out_val = [f"{v:<12}" for v in out_val]
  if args.show_less:
    out_val = [v.strip() for v in out_val]

  if SDPA_TFLOPS > 0:
    speedup_sdpa = TFLOPS / SDPA_TFLOPS
  else:
    speedup_sdpa = 1.0

  if not show_matrix:
    if (speedup_sdpa >= MAX_TFLOPS) or (not only_show_improved) or ("sdpa" in tag):
      print(f"{out_info:>65}: {out_val}, time:{mean_time:.6f}ms, TFLOPS:{TFLOPS:<6.2f}(~{speedup_sdpa:.2f}x)")
    else:
      print(f"{out_info:>65}: {' ' * 50}, TFLOPS:{TFLOPS:<6.2f}(~{speedup_sdpa:.2f}x)")
  else:
    print(f"{out_info:>42}: {out_val}, time:{mean_time:.6f}ms, TFLOPS:{TFLOPS:<6.2f}(~{speedup_sdpa:.2f}x)")

  if speedup_sdpa > MAX_TFLOPS:
    MAX_TFLOPS = speedup_sdpa

  return out, mean_time


def get_qkvo(B, H, N, D):
  device = torch.device("cuda")
  torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
  q = torch.randn((B, H, N, D), device=device, dtype=torch_dtype)
  k = torch.randn((B, H, N, D), device=device, dtype=torch_dtype)
  v = torch.randn((B, H, N, D), device=device, dtype=torch_dtype)
  o = torch.zeros((B, H, N, D), device=device, dtype=torch_dtype)

  if args.no_rand_q or args.no_rand_qkv:
    q.fill_(0.1)
  if args.no_rand_k or args.no_rand_qkv:
    k.fill_(0.1)
  if args.no_rand_v or args.no_rand_qkv:
    v.fill_(0.1)
  return q, k, v, o


def sdpa(q: Tensor, k: Tensor, v: Tensor):
  with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
    out: Tensor = F.scaled_dot_product_attention(q, k, v)
    return out


def unfused_standard_attn(q: Tensor, k: Tensor, v: Tensor):
  att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(q.size(-1)))
  att = torch.softmax(att, dim=-1)
  out = att @ v
  return out


MAX_HEADDIM_CFG = {}


def check_all_close(out_flash_or_sdpa: Tensor, out_mma: Tensor, tag: str = "out_mma", check_all: bool = False):
  if args.run_torch_unfused:
    pretty_print_line(m="-")
  else:
    pretty_print_line(m="=")
  print(f"out_flash_or_sdpa: {out_flash_or_sdpa.shape}, {out_flash_or_sdpa.dtype}")
  print(f"out_mma         : {out_mma.shape}, {out_mma.dtype}")
  pretty_print_line(m="-")
  out_flash_or_sdpa = out_flash_or_sdpa.float()
  out_mma = out_mma.float()
  rtol, atol = (1e-2, 1e-2) if args.dtype == "fp16" else (2e-2, 2e-2)
  all_close = torch.allclose(out_flash_or_sdpa, out_mma, rtol=rtol, atol=atol)
  max_diff = (out_flash_or_sdpa - out_mma).abs().max().item()
  mean_diff = (out_flash_or_sdpa - out_mma).abs().mean().item()
  print(f"{tag:<18}: allclose={all_close}, max_diff={max_diff:.8f}, mean_diff={mean_diff:.8f}")
  if check_all:
    diff = (out_flash_or_sdpa - out_mma).abs()
    idx = torch.argmax(diff)
    idx = np.unravel_index(idx.cpu().item(), diff.shape)
    print(f"max_diff index: {idx}, out_flash_or_sdpa={out_flash_or_sdpa[idx]}, out_mma={out_mma[idx]}")
  pretty_print_line(m="=")


def main():
  global MAX_HEADDIM_CFG

  B = args.B or 1
  H = args.H or 48
  N = args.N or 8192
  D = args.D or 320

  seed = args.seed if args.seed is not None else 1
  set_rand_seed(seed)

  q, k, v, o = get_qkvo(B, H, N, D)

  if has_flash_attn:
    q_flash = q.transpose(1, 2).contiguous()
    k_flash = k.transpose(1, 2).contiguous()
    v_flash = v.transpose(1, 2).contiguous()

  if args.check or args.check_all:
    out_sdpa = sdpa(q, k, v)
    out_ffpa_f16 = ffpa_f16_acc(q, k, v)
    out_ffpa_f32 = ffpa_f32_acc(q, k, v)
    check_all_close(out_sdpa, out_ffpa_f16, tag="ffpa_f16_acc", check_all=args.check_all)
    check_all_close(out_sdpa, out_ffpa_f32, tag="ffpa_f32_acc", check_all=args.check_all)
    return

  pretty_print_line()
  print(f"B={B}, H={H}, N={N}, D={D}, dtype={args.dtype}, warmup={args.warmup}, iters={args.iters}")
  pretty_print_line()

  out_sdpa, _ = run_benchmark(sdpa, q, k, v, "sdpa")
  run_benchmark(ffpa_f16_acc, q, k, v, "ffpa-f16-acc")
  run_benchmark(ffpa_f32_acc, q, k, v, "ffpa-f32-acc")

  if args.run_torch_unfused:
    run_benchmark(unfused_standard_attn, q, k, v, "torch-unfused")

  if has_flash_attn and args.run_flash_attn and D <= 256:
    run_benchmark(flash_attn_func, q_flash, k_flash, v_flash, "flash-attn")


if __name__ == "__main__":
  main()

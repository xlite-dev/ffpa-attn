import argparse
import math
import random
import sys
import time
from functools import partial
from typing import Optional

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

sys.path.append("../")
from env import ENV, pretty_print_line

torch.set_grad_enabled(False)
torch.set_printoptions(
    precision=6, threshold=8, edgeitems=3, linewidth=120, sci_mode=False
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-rand-q", "--no-rq", action="store_true")
    parser.add_argument("--no-rand-k", "--no-rk", action="store_true")
    parser.add_argument("--no-rand-v", "--no-rv", action="store_true")
    parser.add_argument("--no-rand-qkv", "--no-rqkv", action="store_true")
    parser.add_argument("--run-torch-unfused", "--torch", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--check-all", action="store_true")
    parser.add_argument("--show-all", "--show", action="store_true")
    parser.add_argument("--show-matrix", action="store_true")
    parser.add_argument("--only-flops-matmul", "--flops-mm", action="store_true")
    parser.add_argument("--B", type=int, default=None)
    parser.add_argument("--H", type=int, default=None)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--D", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--verbose", "--v", action="store_true")
    parser.add_argument("--warmup", "--w", type=int, default=1)
    parser.add_argument("--iters", "--i", type=int, default=5)
    parser.add_argument("--tag-hints", "--tags", "--hints", type=str, default=None)
    parser.add_argument(
        "--force-build", "--build", action="store_true", help="Force build from sources"
    )
    parser.add_argument("--gen-bench-table", "--gen-bench", action="store_true")
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
pyffpa, use_pyffpa_package = ENV.try_load_pyffpa_library(
    force_build=args.force_build, verbose=args.verbose
)
if use_pyffpa_package:
    import pyffpa  # tricks for IDE code search


def get_mha_tflops(
    B: int, H: int, N: int, D: int, secs: float = 1.0, only_matmul: bool = False
):
    # Q @ K^T FLOPs
    flops_qk = B * H * N * N * (2 * D - 1)

    # Scaling FLOPs
    flops_scaling = B * H * N * N

    # Safe_Softmax FLOPs
    flops_row_max = B * H * N * (N - 1)  # row max
    flops_subtract_max = B * H * N * N  # sub max
    flops_exp = B * H * N * N  # pointwise exp
    flops_row_sum = B * H * N * (N - 1)  # row sum
    flops_normalization = B * H * N * N  # normalization

    flops_safe_softmax = (
        flops_row_max
        + flops_subtract_max
        + flops_exp
        + flops_row_sum
        + flops_normalization
    )

    # P @ V FLOPs
    flops_pv = B * H * N * D * (2 * N - 1)

    # Total FLOPs
    total_flops = flops_qk + flops_scaling + flops_safe_softmax + flops_pv
    if only_matmul:
        total_flops = flops_qk + flops_pv

    # Convert to TFLOPS
    # 1 TFLOPS = 10^12 FLOPS
    # ref: https://imgtec.eetrend.com/blog/2021/100062210.html.
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
    s: Optional[torch.Tensor] = None,  # DEBUG
    stages: int = -1,
    warmup: int = args.warmup,
    iters: int = args.iters,
    show_matrix: bool = args.show_matrix,
    only_show_improved: bool = not args.show_all,
):

    global MAX_TFLOPS
    global MAX_HEADDIM_CFG
    global SDPA_TFLOPS

    tag_hints: str = args.tag_hints  # e.g "share-qkv,tiling-kv,swizzle"
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

    if "unfused" in tag and (not args.run_torch_unfused):
        return None, None

    B, H, N, D = q.size()
    if "flash" in tag:
        B, N, H, D = q.size()

    STATIS_INFO["headdim"].add(D)

    max_supported_D = MAX_HEADDIM_CFG.get(tag, None)
    # skip if headdim not supported.
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
                    perf_func(q, k, v, out, stages)
            else:
                perf_func(q, k, v, out)
    else:
        for i in range(warmup):
            _ = perf_func(q, k, v)

    torch.cuda.synchronize()
    start = time.time()
    # iters
    if out is not None:
        for i in range(iters):
            if stages >= 1:
                if s is not None:
                    perf_func(q, k, v, out, s, stages)
                else:
                    perf_func(q, k, v, out, stages)
            else:
                perf_func(q, k, v, out)
    else:
        for i in range(iters):
            out = perf_func(q, k, v)
    torch.cuda.synchronize()
    end = time.time()
    total_secs = end - start
    total_time = (end - start) * 1000  # ms
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
    out_val_first = out.flatten()[:3].detach().cpu().numpy().tolist()
    out_val_last = out.flatten()[-3:].detach().cpu().numpy().tolist()
    out_val_first = [round(v, 8) for v in out_val_first]
    out_val_last = [round(v, 8) for v in out_val_last]
    out_val = out_val_first[:2]
    out_val.append(out_val_last[-1])
    out_val = [f"{v:<12}" for v in out_val]

    if SDPA_TFLOPS > 0:
        speedup_sdpa = TFLOPS / SDPA_TFLOPS
    else:
        speedup_sdpa = 1.0

    # caculate TFLOPS improved.
    if TFLOPS > MAX_TFLOPS:
        if MAX_TFLOPS > 0:
            improve = ((TFLOPS - MAX_TFLOPS) / MAX_TFLOPS) * 100
            improve = round(improve, 2)
        else:
            improve = 0

        MAX_TFLOPS = TFLOPS
        print(
            f"{out_info:>25}: {out_val}, time:{str(mean_time)[:8]}ms, "
            f"TFLOPS:{TFLOPS:<6.2f}(+{improve:<5.2f}%)(~{speedup_sdpa:<4.2f}x)"
        )
    else:
        improve = 0
        if (not only_show_improved) or (("flash" in tag) or ("sdpa" in tag)):
            print(
                f"{out_info:>25}: {out_val}, time:{str(mean_time)[:8]}ms, "
                f"TFLOPS:{TFLOPS:<6.2f}(+{improve:<5.2f}%)(~{speedup_sdpa:<4.2f}x)"
            )

    if show_matrix:
        print(out)
    time.sleep(args.sleep)
    torch.cuda.synchronize()
    return out.clone() if args.check else out, mean_time


def gen_bench_markdown_table():
    global STATIS_INFO
    STATIS_INFO["headdim"] = sorted(list(STATIS_INFO["headdim"]))
    pretty_print_line("FFPA Benchmark Data")
    print(STATIS_INFO)
    pretty_print_line()
    headdims = [str(d) for d in STATIS_INFO["headdim"]]
    num_headdim = len(headdims)
    table_header = "|Algorithm|" + "|".join(headdims) + "|\n"
    table_header += "|:---:|" + ":---:|" * num_headdim
    # calculate improved
    sdpa_tflops = STATIS_INFO["(sdpa)"]
    # TODO: calculate best ffpa tflops across multi-stages.    
    ffpa_l1_f322_tflops = STATIS_INFO["(ffpa+acc+f32+L1+stage2)"]
    ffpa_l1_f322_speedup = [
        round(f / s, 2) for f, s in zip(ffpa_l1_f322_tflops, sdpa_tflops)
    ]
    ffpa_l1_f162_tflops = STATIS_INFO["(ffpa+acc+f16+L1+stage2)"]
    ffpa_l1_f162_speedup = [
        round(f / s, 2) for f, s in zip(ffpa_l1_f162_tflops, sdpa_tflops)
    ]
    # sdpa, ffpa, speedup strings.
    sdpa_str = "|SDPA EA|" + "|".join([str(s) + "T" for s in sdpa_tflops]) + "|"
    ffpa_l1_f32_str = (
        "|FFPA L1*|" + "|".join([str(f) + "T" for f in ffpa_l1_f322_tflops]) + "|"
    )
    ffpa_l1_f32_speedup_str = (
        "|Speedup|" + "|".join([str(fs) + "x" for fs in ffpa_l1_f322_speedup]) + "|"
    )
    ffpa_l1_f16_str = (
        "|FFPA L1^|" + "|".join([str(f) + "T" for f in ffpa_l1_f162_tflops]) + "|"
    )
    ffpa_l1_f16_speedup_str = (
        "|Speedup|" + "|".join([str(fs) + "x" for fs in ffpa_l1_f162_speedup]) + "|"
    )
    pretty_print_line("FFPA Benchmark Markdown Table")
    print(table_header)
    print(sdpa_str)
    print(ffpa_l1_f32_str)
    print(ffpa_l1_f32_speedup_str)
    print(ffpa_l1_f16_str)
    print(ffpa_l1_f16_speedup_str)
    pretty_print_line()


def get_qkvo(B, H, N, D):
    if not (args.no_rand_q or args.no_rand_qkv):
        q = torch.randn((B, H, N, D), dtype=torch.half, device="cuda")
    else:
        q = torch.ones(B, H, N, D, device="cuda", dtype=torch.half).contiguous()
    if not (args.no_rand_k or args.no_rand_qkv):
        k = torch.randn((B, H, N, D), dtype=torch.half, device="cuda")
    else:
        k = torch.ones(B, H, N, D, device="cuda", dtype=torch.half).contiguous()
    if not (args.no_rand_v or args.no_rand_qkv):
        v = torch.randn((B, H, N, D), dtype=torch.half, device="cuda")
    else:
        v = torch.ones(B, H, N, D, device="cuda", dtype=torch.half).contiguous()

    o = torch.zeros(B, H, N, D, device="cuda", dtype=torch.half).contiguous()
    # transpose (H,N) -> (N,H) for FA2.
    fq = q.transpose(1, 2).contiguous()
    fk = k.transpose(1, 2).contiguous()
    fv = v.transpose(1, 2).contiguous()
    # transpose (N,D) -> (D,N) for V smem swizzle.
    tk = k.transpose(-2, -1).contiguous()  # [B,H,N,D] -> [B,H,D,N]
    tv = v.transpose(-2, -1).contiguous()  # [B,H,N,D] -> [B,H,D,N]

    return q, k, v, o, fq, fk, fv, tk, tv


# un-fused naive attn
def unfused_standard_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y


def sdpa(q: Tensor, k: Tensor, v: Tensor, use_flash: bool = False):
    if not use_flash:
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            out: Tensor = F.scaled_dot_product_attention(q, k, v)
    else:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out: Tensor = F.scaled_dot_product_attention(q, k, v)
    return out


def check_all_close(
    out_flash_or_sdpa: torch.Tensor,
    out_mma: torch.Tensor,
    tag: str = "out_mma",
    check_all: bool = False,
    is_flash: bool = False,
):
    if any((out_flash_or_sdpa is None, out_mma is None)):
        return
    if is_flash:
        true_tag = "out_flash"
        out_flash_or_sdpa = out_flash_or_sdpa.transpose(1, 2)
    else:
        true_tag = "out_sdpa"
    if check_all:
        for i in range(int(N / 8)):
            if i < 4:
                pretty_print_line()
                print(f"{true_tag}[:, :,  {(i * 8)}:{(i + 1) * 8}, :]:\n")
                print(out_flash_or_sdpa[:, :, (i * 8) : (i + 1) * 8, :].float())
                print(f"{tag}[:, :, {(i * 8)}:{(i + 1) * 8}, :]:\n")
                print(out_mma[:, :, (i * 8) : (i + 1) * 8, :].float())
        pretty_print_line()
    diff = torch.abs(out_flash_or_sdpa - out_mma)
    all_close = str(torch.allclose(out_flash_or_sdpa, out_mma, atol=1e-2))
    pretty_print_line(
        f"{true_tag} vs {tag:<15}, all close: {all_close:<6}, "
        f"max diff: {diff.max().item():.6f}, min diff: {diff.min().item():.6f}, "
        f"mean diff: {diff.mean().item():.6f}", sep="", mode="left"
    )


Bs = [1] if not args.B else [args.B]
Hs = [48] if not args.H else [args.H]
Ns = [8192] if not args.N else [args.N]
Ds = list(range(320, 1024 + 64, 64)) if not args.D else [args.D]
# batch_size, n_head, seq_len, head_dim (B,H,N,D)
BHNDs = [(B, H, N, D) for B in Bs for H in Hs for N in Ns for D in Ds]
# max headdim supported for different methods. skip if D > max_D.
MAX_HEADDIM_CFG: dict[str, int] = {
    # FFPA, SDPA, Naive MHA.
    "(sdpa)": 4096,  # may no limit
    "(unfused)": 4096,  # may no limit
    "(ffpa+acc+f16+L1+stage1)": 1024,  # may no limit
    "(ffpa+acc+f16+L1+stage2)": 1024,  # may no limit
    "(ffpa+acc+f16+L1+stage3)": 1024,  # may no limit
    "(ffpa+acc+f16+L1+stage4)": 1024,  # may no limit
    "(ffpa+acc+f32+L1+stage1)": 1024,  # may no limit
    "(ffpa+acc+f32+L1+stage2)": 1024,  # may no limit
    "(ffpa+acc+f32+L1+stage3)": 1024,  # may no limit
    "(ffpa+acc+f32+L1+stage4)": 1024,  # may no limit
}

seed = args.seed if args.seed else random.choice(range(10000))
set_rand_seed(seed)
pretty_print_line()
pretty_print_line(
    f"B: batch_size, H: n_head, N: seq_len, D: head_dim, "
    f"seed: {seed}, Warmup: {args.warmup}, Iters: {args.iters}"
)

for (B, H, N, D) in BHNDs:
    MAX_TFLOPS = -1
    SDPA_TFLOPS = -1
    q, k, v, o, fq, fk, fv, tk, tv = get_qkvo(B, H, N, D)
    torch.cuda.synchronize()
    pretty_print_line()
    pretty_print_line(
        f"B={B}, H={H}, N={N}, D={D}, Warmup: {args.warmup}, Iters: {args.iters}"
    )
    if not use_pyffpa_package:
        # Naive MHA, FFPA, SDPA (D > 256)
        out_unfused, _ = run_benchmark(unfused_standard_attn, q, k, v, "(unfused)")
        out_sdpa, _ = run_benchmark(
            partial(sdpa, use_flash=(D <= 256)), q, k, v, "(sdpa)"
        )
        out_ffpa_l1_f321, _ = run_benchmark(
            pyffpa.ffpa_mma_acc_f32_L1, q, k, v, "(ffpa+acc+f32+L1+stage1)", o, stages=1
        )
        out_ffpa_l1_f322, _ = run_benchmark(
            pyffpa.ffpa_mma_acc_f32_L1, q, k, v, "(ffpa+acc+f32+L1+stage2)", o, stages=2
        )
        out_ffpa_l1_f161, _ = run_benchmark(
            pyffpa.ffpa_mma_acc_f16_L1, q, k, v, "(ffpa+acc+f16+L1+stage1)", o, stages=1
        )
        out_ffpa_l1_f162, _ = run_benchmark(
            pyffpa.ffpa_mma_acc_f16_L1, q, k, v, "(ffpa+acc+f16+L1+stage2)", o, stages=2
        )
        if ENV.enable_all_mutistages():
            out_ffpa_l1_f323, _ = run_benchmark(
                pyffpa.ffpa_mma_acc_f32_L1, q, k, v, "(ffpa+acc+f32+L1+stage3)", o, stages=3
            )
            out_ffpa_l1_f324, _ = run_benchmark(
                pyffpa.ffpa_mma_acc_f32_L1, q, k, v, "(ffpa+acc+f32+L1+stage4)", o, stages=4
            )
            out_ffpa_l1_f163, _ = run_benchmark(
                pyffpa.ffpa_mma_acc_f16_L1, q, k, v, "(ffpa+acc+f16+L1+stage3)", o, stages=3
            )
            out_ffpa_l1_f164, _ = run_benchmark(
                pyffpa.ffpa_mma_acc_f16_L1, q, k, v, "(ffpa+acc+f16+L1+stage4)", o, stages=4
            )
    else:
        # Naive MHA, FFPA, SDPA (D > 256)
        out_unfused, _ = run_benchmark(unfused_standard_attn, q, k, v, "(unfused)")
        out_sdpa, _ = run_benchmark(
            partial(sdpa, use_flash=(D <= 256)), q, k, v, "(sdpa)"
        )
        out_ffpa_l1_f321, _ = run_benchmark(
            partial(pyffpa.ffpa, level=pyffpa.L1, acc=pyffpa.FP32),
            q,
            k,
            v,
            "(ffpa+acc+f32+L1+stage1)",
            o,
            stages=1,
        )
        out_ffpa_l1_f322, _ = run_benchmark(
            partial(pyffpa.ffpa, level=pyffpa.L1, acc=pyffpa.FP32),
            q,
            k,
            v,
            "(ffpa+acc+f32+L1+stage2)",
            o,
            stages=2,
        )
        out_ffpa_l1_f161, _ = run_benchmark(
            partial(pyffpa.ffpa, level=pyffpa.L1, acc=pyffpa.FP16),
            q,
            k,
            v,
            "(ffpa+acc+f16+L1+stage1)",
            o,
            stages=1,
        )
        out_ffpa_l1_f162, _ = run_benchmark(
            partial(pyffpa.ffpa, level=pyffpa.L1, acc=pyffpa.FP16),
            q,
            k,
            v,
            "(ffpa+acc+f16+L1+stage2)",
            o,
            stages=2,
        )
        if ENV.enable_all_mutistages():
            out_ffpa_l1_f323, _ = run_benchmark(
                partial(pyffpa.ffpa, level=pyffpa.L1, acc=pyffpa.FP32),
                q,
                k,
                v,
                "(ffpa+acc+f32+L1+stage3)",
                o,
                stages=3,
            )
            out_ffpa_l1_f324, _ = run_benchmark(
                partial(pyffpa.ffpa, level=pyffpa.L1, acc=pyffpa.FP32),
                q,
                k,
                v,
                "(ffpa+acc+f32+L1+stage4)",
                o,
                stages=4,
            )
            out_ffpa_l1_f163, _ = run_benchmark(
                partial(pyffpa.ffpa, level=pyffpa.L1, acc=pyffpa.FP16),
                q,
                k,
                v,
                "(ffpa+acc+f16+L1+stage3)",
                o,
                stages=3,
            )
            out_ffpa_l1_f164, _ = run_benchmark(
                partial(pyffpa.ffpa, level=pyffpa.L1, acc=pyffpa.FP16),
                q,
                k,
                v,
                "(ffpa+acc+f16+L1+stage4)",
                o,
                stages=4,
            )
    pretty_print_line()

    del q; del k; del v; del o; del fq; del fk; del fv; del tk; del tv
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    if args.check:
        check_all_close(out_sdpa, out_ffpa_l1_f321, "out_ffpa_l1_f321", args.check_all)
        check_all_close(out_sdpa, out_ffpa_l1_f322, "out_ffpa_l1_f322", args.check_all)
        check_all_close(out_sdpa, out_ffpa_l1_f161, "out_ffpa_l1_f161", args.check_all)
        check_all_close(out_sdpa, out_ffpa_l1_f162, "out_ffpa_l1_f161", args.check_all)
        if ENV.enable_all_mutistages():
            check_all_close(out_sdpa, out_ffpa_l1_f323, "out_ffpa_l1_f323", args.check_all)
            check_all_close(out_sdpa, out_ffpa_l1_f324, "out_ffpa_l1_f324", args.check_all)
            check_all_close(out_sdpa, out_ffpa_l1_f163, "out_ffpa_l1_f163", args.check_all)
            check_all_close(out_sdpa, out_ffpa_l1_f164, "out_ffpa_l1_f164", args.check_all)
        pretty_print_line()

if args.gen_bench_table:
    gen_bench_markdown_table()

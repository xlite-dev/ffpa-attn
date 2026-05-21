
# FFPA Attention Examples

## Quick Start

```bash
python3 examples/perf.py # default: forward + backward w/o autotuning
python3 examples/perf.py --no-bwd # only forward pass
python3 examples/perf.py --no-fwd # only backward pass
python3 examples/perf.py --fwd-backend triton --bwd-backend triton --tune fast
python3 examples/perf.py --fwd-backend triton --bwd-backend triton --tune max
python3 examples/perf.py --fwd-backend triton --bwd-backend triton --tune max --fwd-tma --bwd-tma # SM>=90
python3 examples/perf.py --fwd-backend cutedsl --bwd-backend cutedsl # SM==90 + dense 256<D<=512
```

The `examples/perf.py` migrated benchmark plotting entrypoint. It preserves the old plot style, can benchmark forward/backward cases on demand, and writes both `ffpa_{device}_speedup.png` and `ffpa_{device}_speedup.md`. The additive-mask example uses a compact `[1, 1, 1, Nkv]` key-position bias by default. Use `[1, 1, Nq, Nkv]` only when per-query bias is required, since it scales as `O(Nq * Nkv)` memory.

## Benchmark

TFLOPS reports the theoretical dominant attention GEMM throughput only; forward and backward are computed separately from the measured latency. Env: NVIDIA L20 (Ada, 119.5 TFLOPS), PyTorch 2.11, CUDA 13.0, Headdim=512 (FA-2 not supported).

<div align='center' markdown="1">

### Forward Pass (Triton)

| Case | dtype | Nq/Nkv | FFPA / SDPA | TFLOPS | speedup |
|:---:|:---:|:---:|:---:|:---:|:---:|
| self-attn | fp16 | 8192/8192 | 50.33 / 75.32 ms | 87T / 58T | 1.50x |
| self-attn | bf16 | 8192/8192 | 47.90 / 75.98 ms | 92T / 58T | 1.59x |
| cross-attn | fp16 | 1024/8192 | 6.42 / 10.05 ms | 86T / 55T | 1.57x |
| cross-attn | bf16 | 1024/8192 | 6.71 / 9.93 ms | 82T / 55T | 1.48x |
| decode-attn | fp16 | 1/8192 | 0.77 / 1.00 ms | 0.70T / 0.54T | 1.30x |
| decode-attn | bf16 | 1/8192 | 0.77 / 0.80 ms | 0.70T / 0.67T | 1.04x |
| gqa | fp16 | 8192/8192 | 50.72 / 75.27 ms | 87T / 58T | 1.48x |
| gqa | bf16 | 8192/8192 | 47.21 / 75.41 ms | 93T / 58T | 1.60x |
| causal | fp16 | 8192/8192 | 26.69 / 38.17 ms | 82T / 58T | 1.43x |
| causal | bf16 | 8192/8192 | 26.40 / 37.53 ms | 83T / 59T | 1.42x |
| attn-mask | fp16 | 8192/8192 | 55.57 / 82.04 ms | 79T / 54T | 1.48x |
| attn-mask | bf16 | 8192/8192 | 52.98 / 81.68 ms | 83T / 54T | 1.54x |
| dropout | fp16 | 8192/8192 | 55.77 / 83.61 ms | 79T / 53T | 1.50x |
| dropout | bf16 | 8192/8192 | 52.60 / 84.28 ms | 84T / 52T | 1.60x |
| non-aligned | fp16 | 8191/8191 | 14.19 / 19.01 ms | 77T / 58T | 1.34x |
| non-aligned | bf16 | 8191/8191 | 12.36 / 19.12 ms | 89T / 58T | 1.55x |

### Backward Pass (Triton)

| Case | dtype | Nq/Nkv | FFPA / SDPA | TFLOPS | speedup |
|:---:|:---:|:---:|:---:|:---:|:---:|
| self-attn | fp16 | 8192/8192 | 192.15 / 353.61 ms | 57T / 31T | 1.84x |
| self-attn | bf16 | 8192/8192 | 196.53 / 353.47 ms | 56T / 31T | 1.80x |
| cross-attn | fp16 | 1024/8192 | 25.98 / 42.82 ms | 53T / 32T | 1.65x |
| cross-attn | bf16 | 1024/8192 | 26.24 / 43.49 ms | 52T / 32T | 1.66x |
| decode-attn | fp16 | 1/8192 | 2.65 / 6.05 ms | 0.51T / 0.22T | 2.28x |
| decode-attn | bf16 | 1/8192 | 2.65 / 6.01 ms | 0.51T / 0.22T | 2.27x |
| gqa | fp16 | 8192/8192 | 193.69 / 352.02 ms | 57T / 31T | 1.82x |
| gqa | bf16 | 8192/8192 | 198.45 / 352.54 ms | 55T / 31T | 1.78x |
| causal | fp16 | 8192/8192 | 96.86 / 199.46 ms | 57T / 28T | 2.06x |
| causal | bf16 | 8192/8192 | 96.95 / 199.67 ms | 57T / 28T | 2.06x |
| attn-mask | fp16 | 8192/8192 | 195.14 / 375.20 ms | 56T / 29T | 1.92x |
| attn-mask | bf16 | 8192/8192 | 197.66 / 377.68 ms | 56T / 29T | 1.91x |
| dropout | fp16 | 8192/8192 | 200.19 / 367.95 ms | 55T / 30T | 1.84x |
| dropout | bf16 | 8192/8192 | 204.05 / 364.58 ms | 54T / 30T | 1.79x |
| non-aligned | fp16 | 8191/8191 | 52.67 / 101.27 ms | 52T / 27T | 1.92x |
| non-aligned | bf16 | 8191/8191 | 52.90 / 101.69 ms | 52T / 27T | 1.92x |

</div>

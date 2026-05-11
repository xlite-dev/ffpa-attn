
# FFPA-Attn Examples

## Quick Start

```bash
python3 examples/ffpa_attn_fwd.py --forward-backend triton
python3 examples/ffpa_attn_fwd.py --forward-backend triton --autotune
ENABLE_FFPA_CUDA_IMPL=1 python3 examples/ffpa_attn_fwd.py --forward-backend cuda
python3 examples/ffpa_attn_bwd.py --backward-backend sdpa
python3 examples/ffpa_attn_bwd.py --backward-backend triton
python3 examples/ffpa_attn_bwd.py --backward-backend triton --autotune
python3 examples/perf.py
python3 examples/perf.py --forward --backward --fwd-backend triton --bwd-backend triton --tune fast
```

- `examples/ffpa_attn_fwd.py`: forward-only examples for self-attn, cross-attn, GQA, causal, and non-aligned seqlen.
- `examples/ffpa_attn_bwd.py`: end2end forward + backward examples for self-attn, cross-attn, GQA, causal, and non-aligned seqlen.
- `examples/perf.py`: migrated benchmark plotting entrypoint. It preserves the old plot style, can benchmark forward/backward cases on demand, and writes both `ffpa_speedup.png` and `ffpa_speedup.md`.
- The additive-mask example uses a compact `[1, 1, 1, Nkv]` key-position bias by default. Use `[1, 1, Nq, Nkv]` only when per-query bias is required, since it scales as `O(Nq * Nkv)` memory.

## Benchmark

Env: NVIDIA L20 (Ada), PyTorch 2.11, CUDA 13.0, Headdim=512 (FA-2 not supported).

### Forward Pass (Legacy CUDA)

| Case | dtype | Nq/Nkv | allclose | FFPA / SDPA | speedup |
|:---:|:---:|:---:|:---:|:---:|:---:|
| self-attn | fp16 | 8192/8192 | ✅ | 46.7 / 74.7 ms | 1.60x |
| cross-attn | fp16 | 1024/8192 | ✅ | 6.32 / 9.94 ms | 1.57x |
| gqa | fp16 | 8192/8192 | ✅ | 46.4 / 74.8 ms | 1.61x |
| causal | fp16 | 8192/8192 | ✅ | 24.3 / 37.4 ms | 1.54x |
| non-aligned | fp16 | 8191/8191 | ✅ | 12.3 / 19.0 ms | 1.55x |
| self-attn | bf16 | 8192/8192 | ✅ | 46.5 / 74.7 ms | 1.61x |
| cross-attn | bf16 | 1024/8192 | ✅ | 6.29 / 9.95 ms | 1.58x |
| gqa | bf16 | 8192/8192 | ✅ | 46.2 / 74.7 ms | 1.62x |
| causal | bf16 | 8192/8192 | ✅ | 24.2 / 37.5 ms | 1.55x |
| non-aligned | bf16 | 8191/8191 | ✅ | 12.3 / 19.0 ms | 1.55x |

### Backward Pass (Triton w/ autotune)

| Case | dtype | Nq/Nkv | allclose | FFPA / SDPA | speedup |
|:---:|:---:|:---:|:---:|:---:|:---:|
| self-attn | fp16 | 8192/8192 | ✅ | 255.9 / 429.2 ms | 1.68x |
| cross-attn | fp16 | 1024/8192 | ✅ | 36.54 / 54.41 ms | 1.49x |
| gqa | fp16 | 8192/8192 | ✅ | 256.4 / 425.3 ms | 1.66x |
| causal | fp16 | 8192/8192 | ✅ | 132.9 / 240.2 ms | 1.81x |
| non-aligned | fp16 | 8191/8191 | ✅ | 68.59 / 118.4 ms | 1.73x |
| self-attn | bf16 | 8192/8192 | ✅ | 255.2 / 425.3 ms | 1.67x |
| cross-attn | bf16 | 1024/8192 | ✅ | 36.34 / 54.50 ms | 1.50x |
| gqa | bf16 | 8192/8192 | ✅ | 255.7 / 425.6 ms | 1.66x |
| causal | bf16 | 8192/8192 | ✅ | 132.6 / 237.7 ms | 1.79x |
| non-aligned | bf16 | 8191/8191 | ✅ | 68.44 / 118.3 ms | 1.73x |

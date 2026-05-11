## Benchmark

Env: NVIDIA L20, B=1, N=8192, H=32, D=512.

Note: fallback mode reuses hard-coded plot speedups only, so FFPA / SDPA latency and allclose are unavailable.

### Forward Pass (Fallback hard-coded data)

| Case | dtype | Nq/Nkv | allclose | FFPA / SDPA | speedup |
|:---:|:---:|:---:|:---:|:---:|:---:|
| self-attn | fp16 | 8192/8192 | - | - / - | 2.06x |
| self-attn | bf16 | 8192/8192 | - | - / - | 2.08x |
| cross-attn | fp16 | 1024/8192 | - | - / - | 1.86x |
| cross-attn | bf16 | 1024/8192 | - | - / - | 1.87x |
| decode-attn | fp16 | 1/8192 | - | - / - | 2.86x |
| decode-attn | bf16 | 1/8192 | - | - / - | 2.85x |
| gqa | fp16 | 8192/8192 | - | - / - | 2.06x |
| gqa | bf16 | 8192/8192 | - | - / - | 2.09x |
| causal | fp16 | 8192/8192 | - | - / - | 1.96x |
| causal | bf16 | 8192/8192 | - | - / - | 1.99x |
| attn-mask | fp16 | 8192/8192 | - | - / - | 1.70x |
| attn-mask | bf16 | 8192/8192 | - | - / - | 1.74x |
| dropout | fp16 | 8192/8192 | - | - / - | 1.79x |
| dropout | bf16 | 8192/8192 | - | - / - | 1.82x |
| non-aligned | fp16 | 8191/8191 | - | - / - | 1.96x |
| non-aligned | bf16 | 8191/8191 | - | - / - | 1.98x |

### Backward Pass (Fallback hard-coded data)

| Case | dtype | Nq/Nkv | allclose | FFPA / SDPA | speedup |
|:---:|:---:|:---:|:---:|:---:|:---:|
| self-attn | fp16 | 8192/8192 | - | - / - | 2.34x |
| self-attn | bf16 | 8192/8192 | - | - / - | 2.49x |
| cross-attn | fp16 | 1024/8192 | - | - / - | 2.57x |
| cross-attn | bf16 | 1024/8192 | - | - / - | 2.51x |
| decode-attn | fp16 | 1/8192 | - | - / - | 2.97x |
| decode-attn | bf16 | 1/8192 | - | - / - | 3.11x |
| gqa | fp16 | 8192/8192 | - | - / - | 2.32x |
| gqa | bf16 | 8192/8192 | - | - / - | 2.46x |
| causal | fp16 | 8192/8192 | - | - / - | 2.22x |
| causal | bf16 | 8192/8192 | - | - / - | 2.56x |
| attn-mask | fp16 | 8192/8192 | - | - / - | 2.06x |
| attn-mask | bf16 | 8192/8192 | - | - / - | 2.17x |
| dropout | fp16 | 8192/8192 | - | - / - | 2.27x |
| dropout | bf16 | 8192/8192 | - | - / - | 2.41x |
| non-aligned | fp16 | 8191/8191 | - | - / - | 2.37x |
| non-aligned | bf16 | 8191/8191 | - | - / - | 2.67x |

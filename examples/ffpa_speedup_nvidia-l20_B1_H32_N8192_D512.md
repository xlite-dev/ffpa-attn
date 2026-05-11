## Benchmark

Env: NVIDIA L20, B=1, N=8192, H=32, D=512.

### Forward Pass (Triton)

| Case | dtype | Nq/Nkv | allclose | FFPA / SDPA | speedup |
|:---:|:---:|:---:|:---:|:---:|:---:|
| self-attn | fp16 | 8192/8192 | ✅ | 51.67 / 74.54 ms | 1.44x |
| self-attn | bf16 | 8192/8192 | ✅ | 51.27 / 75.27 ms | 1.47x |
| cross-attn | fp16 | 1024/8192 | ✅ | 7.09 / 11.02 ms | 1.55x |
| cross-attn | bf16 | 1024/8192 | ✅ | 6.72 / 10.39 ms | 1.55x |
| decode-attn | fp16 | 1/8192 | ✅ | 0.75 / 0.80 ms | 1.07x |
| decode-attn | bf16 | 1/8192 | ✅ | 0.75 / 0.85 ms | 1.14x |
| gqa | fp16 | 8192/8192 | ✅ | 49.95 / 76.04 ms | 1.52x |
| gqa | bf16 | 8192/8192 | ✅ | 49.95 / 75.93 ms | 1.52x |
| causal | fp16 | 8192/8192 | ✅ | 26.64 / 38.09 ms | 1.43x |
| causal | bf16 | 8192/8192 | ✅ | 26.50 / 38.17 ms | 1.44x |
| attn-mask | fp16 | 8192/8192 | ✅ | 55.05 / 81.91 ms | 1.49x |
| attn-mask | bf16 | 8192/8192 | ✅ | 55.07 / 81.37 ms | 1.48x |
| dropout | fp16 | 8192/8192 | ✅ | 56.67 / 83.47 ms | 1.47x |
| dropout | bf16 | 8192/8192 | ✅ | 56.24 / 84.13 ms | 1.50x |
| non-aligned | fp16 | 8191/8191 | ✅ | 14.00 / 19.02 ms | 1.36x |
| non-aligned | bf16 | 8191/8191 | ✅ | 13.49 / 19.06 ms | 1.41x |

### Backward Pass (Triton)

| Case | dtype | Nq/Nkv | allclose | FFPA / SDPA | speedup |
|:---:|:---:|:---:|:---:|:---:|:---:|
| self-attn | fp16 | 8192/8192 | ✅ | 234.94 / 354.44 ms | 1.51x |
| self-attn | bf16 | 8192/8192 | ✅ | 234.88 / 348.82 ms | 1.49x |
| cross-attn | fp16 | 1024/8192 | ✅ | 32.26 / 43.57 ms | 1.35x |
| cross-attn | bf16 | 1024/8192 | ✅ | 32.36 / 42.18 ms | 1.30x |
| decode-attn | fp16 | 1/8192 | ✅ | 3.12 / 5.90 ms | 1.89x |
| decode-attn | bf16 | 1/8192 | ✅ | 3.12 / 5.86 ms | 1.88x |
| gqa | fp16 | 8192/8192 | ✅ | 237.13 / 353.82 ms | 1.49x |
| gqa | bf16 | 8192/8192 | ✅ | 235.09 / 351.14 ms | 1.49x |
| causal | fp16 | 8192/8192 | ✅ | 120.93 / 194.89 ms | 1.61x |
| causal | bf16 | 8192/8192 | ✅ | 119.63 / 193.02 ms | 1.61x |
| attn-mask | fp16 | 8192/8192 | ✅ | 256.13 / 371.43 ms | 1.45x |
| attn-mask | bf16 | 8192/8192 | ✅ | 254.59 / 384.31 ms | 1.51x |
| dropout | fp16 | 8192/8192 | ✅ | 250.74 / 360.23 ms | 1.44x |
| dropout | bf16 | 8192/8192 | ✅ | 248.38 / 358.07 ms | 1.44x |
| non-aligned | fp16 | 8191/8191 | ✅ | 61.98 / 101.42 ms | 1.64x |
| non-aligned | bf16 | 8191/8191 | ✅ | 62.60 / 100.48 ms | 1.61x |

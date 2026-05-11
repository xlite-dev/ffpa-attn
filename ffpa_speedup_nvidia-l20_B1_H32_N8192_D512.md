## Benchmark

Env: NVIDIA L20, B=1, N=8192, H=32, D=512.

### Backward Pass (Triton)

| Case | dtype | Nq/Nkv | allclose | FFPA / SDPA | speedup |
|:---:|:---:|:---:|:---:|:---:|:---:|
| self-attn | fp16 | 8192/8192 | ✅ | 233.12 / 365.69 ms | 1.57x |
| self-attn | bf16 | 8192/8192 | ✅ | 233.30 / 363.68 ms | 1.56x |
| cross-attn | fp16 | 1024/8192 | ✅ | 32.28 / 42.20 ms | 1.31x |
| cross-attn | bf16 | 1024/8192 | ✅ | 32.30 / 42.43 ms | 1.31x |
| decode-attn | fp16 | 1/8192 | ✅ | 3.10 / 5.90 ms | 1.90x |
| decode-attn | bf16 | 1/8192 | ✅ | 3.11 / 5.89 ms | 1.90x |
| gqa | fp16 | 8192/8192 | ✅ | 234.98 / 365.25 ms | 1.55x |
| gqa | bf16 | 8192/8192 | ✅ | 235.07 / 361.02 ms | 1.54x |
| causal | fp16 | 8192/8192 | ✅ | 119.40 / 195.77 ms | 1.64x |
| causal | bf16 | 8192/8192 | ✅ | 119.56 / 194.13 ms | 1.62x |
| attn-mask | fp16 | 8192/8192 | ✅ | 253.75 / 383.69 ms | 1.51x |
| attn-mask | bf16 | 8192/8192 | ✅ | 253.57 / 384.96 ms | 1.52x |
| dropout | fp16 | 8192/8192 | ✅ | 246.82 / 362.32 ms | 1.47x |
| dropout | bf16 | 8192/8192 | ✅ | 247.13 / 362.48 ms | 1.47x |
| non-aligned | fp16 | 8191/8191 | ✅ | 61.68 / 100.33 ms | 1.63x |
| non-aligned | bf16 | 8191/8191 | ✅ | 61.74 / 100.35 ms | 1.63x |

## Benchmark

Env: NVIDIA L20, B=1, N=1024, H=32, D=320.

### Forward Pass (Triton)

| Case | dtype | Nq/Nkv | allclose | FFPA / SDPA | speedup |
|:---:|:---:|:---:|:---:|:---:|:---:|
| self-attn | fp16 | 1024/1024 | ✅ | 0.54 / 0.85 ms | 1.57x |
| self-attn | bf16 | 1024/1024 | ✅ | 0.54 / 0.85 ms | 1.58x |
| cross-attn | fp16 | 1024/1024 | ✅ | 0.54 / 0.85 ms | 1.58x |
| cross-attn | bf16 | 1024/1024 | ✅ | 0.54 / 0.85 ms | 1.57x |
| decode-attn | fp16 | 1/1024 | ✅ | 0.27 / 0.07 ms | 0.26x |
| decode-attn | bf16 | 1/1024 | ✅ | 0.26 / 0.07 ms | 0.27x |
| gqa | fp16 | 1024/1024 | ✅ | 0.54 / 0.85 ms | 1.58x |
| gqa | bf16 | 1024/1024 | ✅ | 0.54 / 0.85 ms | 1.58x |
| causal | fp16 | 1024/1024 | ✅ | 0.40 / 0.48 ms | 1.19x |
| causal | bf16 | 1024/1024 | ✅ | 0.40 / 0.48 ms | 1.19x |
| attn-mask | fp16 | 1024/1024 | ✅ | 0.62 / 0.95 ms | 1.54x |
| attn-mask | bf16 | 1024/1024 | ✅ | 0.62 / 0.95 ms | 1.53x |
| dropout | fp16 | 1024/1024 | ✅ | 0.67 / 1.01 ms | 1.51x |
| dropout | bf16 | 1024/1024 | ✅ | 0.67 / 1.01 ms | 1.50x |
| non-aligned | fp16 | 1023/1023 | ✅ | 0.28 / 0.21 ms | 0.77x |
| non-aligned | bf16 | 1023/1023 | ✅ | 0.27 / 0.21 ms | 0.79x |

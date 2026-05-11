## Benchmark

Env: NVIDIA L20, B=1, N=128, H=32, D=64.

### Forward Pass (Triton)

| Case | dtype | Nq/Nkv | allclose | FFPA / SDPA | speedup |
|:---:|:---:|:---:|:---:|:---:|:---:|
| self-attn | fp16 | 128/128 | ✅ | 0.03 / 0.02 ms | 0.72x |
| self-attn | bf16 | 128/128 | ✅ | 0.03 / 0.02 ms | 0.74x |
| cross-attn | fp16 | 1024/128 | ✅ | 0.03 / 0.02 ms | 0.80x |
| cross-attn | bf16 | 1024/128 | ✅ | 0.03 / 0.02 ms | 0.83x |
| decode-attn | fp16 | 1/128 | ✅ | 0.03 / 0.02 ms | 0.79x |
| decode-attn | bf16 | 1/128 | ✅ | 0.03 / 0.02 ms | 0.79x |
| gqa | fp16 | 128/128 | ✅ | 0.03 / 0.02 ms | 0.78x |
| gqa | bf16 | 128/128 | ✅ | 0.03 / 0.02 ms | 0.78x |
| causal | fp16 | 128/128 | ✅ | 0.03 / 0.02 ms | 0.80x |
| causal | bf16 | 128/128 | ✅ | 0.03 / 0.02 ms | 0.79x |
| attn-mask | fp16 | 512/512 | ✅ | 0.04 / 0.04 ms | 0.98x |
| attn-mask | bf16 | 512/512 | ✅ | 0.04 / 0.04 ms | 0.98x |
| dropout | fp16 | 128/128 | ✅ | 0.19 / 0.18 ms | 0.93x |
| dropout | bf16 | 128/128 | ✅ | 0.19 / 0.18 ms | 0.93x |
| non-aligned | fp16 | 127/127 | ✅ | 0.03 / 0.02 ms | 0.79x |
| non-aligned | bf16 | 127/127 | ✅ | 0.03 / 0.02 ms | 0.77x |

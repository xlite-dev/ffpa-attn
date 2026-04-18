
# FFPA-Attn Examples

```bash
python3 examples/run_ffpa_attn.py
```

Env: NVIDIA L20 (Ada), PyTorch 2.11, CUDA 13.0, Headdim=512 (FA-2 not supported).

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

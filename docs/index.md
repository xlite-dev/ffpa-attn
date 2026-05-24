<div align="center">
  <p align="center">
    <h2>🤖FFPA: Yet another Faster Flash Prefill Attention <br>with O(1)⚡️GPU SRAM complexity for large headdim🐑</h2>
  </p>
  <img src=https://img.shields.io/badge/language-CUDA/Python-brightgreen.svg >
  <a href="https://pepy.tech/projects/ffpa-attn"><img src=https://static.pepy.tech/personalized-badge/ffpa-attn?period=total&units=ABBREVIATION&left_color=GRAY&right_color=BLUE&left_text=downloads/pypi ></a>
  <a href="https://pypi.org/project/ffpa-attn/"><img src=https://img.shields.io/github/release/xlite-dev/ffpa-attn.svg?color=GREEN ></a>
  <img src=https://img.shields.io/github/stars/xlite-dev/ffpa-attn.svg?style=dark >
  <img src="https://img.shields.io/github/license/xlite-dev/ffpa-attn.svg?color=blue"><br>
  <img src="assets/ffpa-api.png" width="700px">
</div>

**FFPA(Split-D)**: Yet another **Faster Flash Prefill Attention** with **Split-D** strategy, achieve **O(1) SRAM complexity** and **O(d/4) register complexity** for large headdim (**> 256**), **1.5~3x** 🎉 faster than SDPA. 📚👇The Core features:

<div align='center' markdown="1">

|[Self Attn](./examples)| [GQA/MQA](./examples) |[Cross Attn](./examples)|[Causal/Mask](./examples)|[Dropout](./examples)|[Headdim](#ffpa-design)|[Fwd/Bwd](./examples)|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|✔️(`Nq=Nkv`)|✔️(`Hq!=Hkv`)|✔️(`Nq!=Nkv`)|✔️(`attn_mask`)|✔️(`p>0`)|**320~1024** |**1.5~3x↑** |

</div>

## 📖 Quick Start

<div id="install"></div>

First, install the prebuilt package from [PyPI](https://pypi.org/project/ffpa-attn/) or build [ffpa-attn](https://github.com/xlite-dev/ffpa-attn) from source:

```bash
# Fisrt, install the prebuilt package from PyPI
pip3 install -U ffpa-attn # (support: sm_{80,...,120})
# Or, build ffpa-attn from source, just follow the cmds
git clone https://github.com/xlite-dev/ffpa-attn.git
# Then, build the wheel package (Triton backend only)
cd ffpa-attn && pip3 install -e . --no-build-isolation
```

Then, try to accelerate the attention for large headdim with just <i><b>one-line</b></i> of code:

```python
>>> import torch.nn.functional as F
>>> from ffpa_attn import ffpa_attn_func
>>> # Monkey-patch SDPA to point to FFPA. Every thing that FFPA
>>> # does not support will auto fallback to SDPA: D <= 256, etc.
>>> F.scaled_dot_product_attention = ffpa_attn_func # one-line code
```

For more advanced features, please refer to our online docs at 📘[ffpa-attn.io](https://ffpa-attn.readthedocs.io/en/latest/).

> [!NOTE]
> FFPA supports **cross-attention** where the query seqlen ``Nq`` may differ from the key/value seqlen ``Nkv``, **GQA / MQA** attention where Q has ``Nh_q`` heads and K/V have ``Nh_kv`` heads (requires ``Nh_q % Nh_kv == 0``; group size = ``Nh_q / Nh_kv``), and **causal attention** (pass ``is_causal=True``; queries are aligned to the KV tail, i.e. Q row ``r`` attends to ``k <= r + (Nkv - Nq)``, which requires ``Nkv >= Nq``). K/V must share the same ``Nh_kv`` and ``Nkv``. `enable_gqa` now defaults to `False` to match SDPA exactly, so GQA/MQA usage must pass `enable_gqa=True` explicitly.

<a id="example-self"></a>

Minimal usage example — **Self-Attention** (B=1, H=32, N=8192, D=512):
```python
import torch
import torch.nn.functional as F
from ffpa_attn import ffpa_attn_func

# D: 32, 64, ..., 320, ..., 1024 (FA-2 <= 256, FFPA supports up to 1024).
B, H, N, D = 1, 32, 8192, 512 # batch_size, num_heads, seq_len, head_dim
q = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda")
k = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda")
v = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda")

# FFPA self attention; layout follows SDPA: (B, H, N, D).
out = ffpa_attn_func(q, k, v)  # -> torch.Tensor of shape (B, H, N, D)
print(out.shape, out.dtype)

ref = F.scaled_dot_product_attention(q, k, v)
print(f"vs SDPA max_abs_err={(out - ref).abs().max().item():.4e}")
```

<a id="example-cross"></a>

**Cross-Attention** or **Decoding-Attention** example (short query, long KV cache; `Nq != Nkv`):
```python
import torch
import torch.nn.functional as F
from ffpa_attn import ffpa_attn_func

# Short-query / long-KV, e.g. incremental decoding or cross-attention:
# Q: [B, H, Nq, D], K/V: [B, H, Nkv, D]; Nq can differ from Nkv but Nk==Nv required.
B, H, D = 1, 8, 512
Nq, Nkv = 128, 8192
q = torch.randn(B, H, Nq,  D, dtype=torch.bfloat16, device="cuda")
k = torch.randn(B, H, Nkv, D, dtype=torch.bfloat16, device="cuda")
v = torch.randn(B, H, Nkv, D, dtype=torch.bfloat16, device="cuda")

out = ffpa_attn_func(q, k, v)  # -> (B, H, Nq, D) = (1, 8, 128, 512)
print(out.shape, out.dtype)

ref = F.scaled_dot_product_attention(q, k, v)
print(f"vs SDPA max_abs_err={(out - ref).abs().max().item():.4e}")
```

<a id="example-gqa"></a>

**Grouped-Query / Multi-Query Attention** example (Q has more heads than K/V):
```python
import torch
import torch.nn.functional as F
from ffpa_attn import ffpa_attn_func

# GQA: Q has Nh_q heads, K/V share Nh_kv heads; group_size = Nh_q / Nh_kv.
# Typical Llama-3-style 32/8 ratio; MQA is the Nh_kv==1 special case.
# FFPA targets large headdim so we use D=512 here (FA-2 tops out at D=256).
# enable_gqa defaults to False, so opt into GQA semantics explicitly.
B, D, Nq, Nkv = 1, 512, 1024, 4096
Nh_q, Nh_kv = 32, 8  # group_size = 4
q = torch.randn(B, Nh_q,  Nq,  D, dtype=torch.bfloat16, device="cuda")
k = torch.randn(B, Nh_kv, Nkv, D, dtype=torch.bfloat16, device="cuda")
v = torch.randn(B, Nh_kv, Nkv, D, dtype=torch.bfloat16, device="cuda")

out = ffpa_attn_func(q, k, v, enable_gqa=True)  # -> (B, Nh_q, Nq, D) = (1, 32, 1024, 512)
print(out.shape, out.dtype)

# Reference: replicate K/V along head dim to match Q's head count.
group_size = Nh_q // Nh_kv
k_ref = k.repeat_interleave(group_size, dim=1)
v_ref = v.repeat_interleave(group_size, dim=1)
ref = F.scaled_dot_product_attention(q, k_ref, v_ref)
print(f"vs SDPA max_abs_err={(out - ref).abs().max().item():.4e}")
```

<a id="example-causal"></a>

**Causal Attention** example (self-attention causal; also supports chunked / decoding prefill with `Nkv > Nq`):
```python
import torch
import torch.nn.functional as F
from ffpa_attn import ffpa_attn_func

# Causal self-attention: Q row r attends to k <= r (standard triangular mask).
# FFPA is tuned for large headdim, so we keep D=512 as in the self-attn example.
B, H, N, D = 1, 8, 4096, 512
q = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda")
k = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda")
v = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda")

out = ffpa_attn_func(q, k, v, is_causal=True)
print(out.shape, out.dtype)

ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
print(f"vs SDPA max_abs_err={(out - ref).abs().max().item():.4e}")

# Chunked / decoding prefill: Nq < Nkv, queries aligned to the KV tail
# so Q row r attends to k <= r + (Nkv - Nq). Requires Nkv >= Nq.
# This example keeps D=512 so it stays on the FFPA large-D path. For D <= 256,
# ffpa_attn_func forwards the inputs directly to SDPA without synthesizing a
# causal cross-attention mask for you.
Nq, Nkv = 128, 8192
q = torch.randn(B, H, Nq,  D, dtype=torch.bfloat16, device="cuda")
k = torch.randn(B, H, Nkv, D, dtype=torch.bfloat16, device="cuda")
v = torch.randn(B, H, Nkv, D, dtype=torch.bfloat16, device="cuda")
out = ffpa_attn_func(q, k, v, is_causal=True)
print(out.shape, out.dtype)  # (1, 8, 128, 512)
```

<a id="example-backward"></a>

**Backward Pass** example (compare dQ / dK / dV against SDPA):
```python
import math
import torch
import torch.nn.functional as F
from ffpa_attn import ffpa_attn_func

# Focus on a large-headdim case where FFPA is typically used.
B, H, N, D = 1, 32, 8192, 512
scale = 1.0 / math.sqrt(D)

q = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
k = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
v = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)

out = ffpa_attn_func(q, k, v, scale=scale)
out.sum().backward()

dq = q.grad.detach().clone()
dk = k.grad.detach().clone()
dv = v.grad.detach().clone()

q_ref = q.detach().clone().requires_grad_(True)
k_ref = k.detach().clone().requires_grad_(True)
v_ref = v.detach().clone().requires_grad_(True)
out_ref = F.scaled_dot_product_attention(q_ref, k_ref, v_ref, scale=scale)
out_ref.sum().backward()

print(f"dQ vs SDPA dQ max_abs_err={(dq - q_ref.grad).abs().max().item():.4e}")
print(f"dK vs SDPA dK max_abs_err={(dk - k_ref.grad).abs().max().item():.4e}")
print(f"dV vs SDPA dV max_abs_err={(dv - v_ref.grad).abs().max().item():.4e}")
```

## 📖 Split-D

<a id="ffpa-design"></a>

We extend FlashAttention to support large headdim ($D>256$) via **fine-grained tiling** at the **MMA** level for $QK^\top$ and $PV$ matrix multiplication, referred to as **Split-D**. This design keeps SRAM usage fixed at $B_r \times 16$ (with $B_r=B_c$) for Q, K and V, yielding constant SRAM complexity $O(B_r \times 16) \approx O(1)$ and register complexity $O(d/4)$.

<div align='center'>
  <img src="./assets/split-d.png" width="700px">
  </p><i>
    <b>FFPA</b> enables headdim <b> > 256</b>, and outperforms standard SDPA by <b>1.5~3x</b>🎉.
  </i></p>
</div>

> [!NOTE]
> FFPA has been tested on `Ampere`, `Ada`, `Hopper`, and `Blackwell` architectures (e.g., A30, L20, 4090, H200, 5090), achieves `1.5~3×↑🎉` speedup over SDPA. FFPA is mainly design for **prefill** and large headdim, and may not be faster than SDPA for 😈 small sequence length (`N<512`) or small headdim (`D<=256`).

## 🎉 Benchmark

Runnable examples are provided under [`examples`](./examples). The performance benchmarks for the NVIDIA L20 (**Ada**), NVIDIA Geforce RTX 5090 (**Blackwell**), NVIDIA H800 PCIE (**Hopper**), NVIDIA H200 SXM (**Hopper**, **CuTeDSL** backend, up to **427** TFLOPS!🎉) with large headdim are shown below:

<div align='center'>
  <img src='./assets/perf/ffpa_speedup_nvidia-l20_B1_H32_N8192_D320_T.png' width='350px'>
  <img src='./assets/perf/ffpa_speedup_nvidia-l20_B1_H32_N8192_D512_T.png' width='350px'><br>
  <img src='./assets/perf/ffpa_speedup_nvidia-geforce-rtx-5090_B1_H32_N8192_D320_T.png' width='350px'>
  <img src='./assets/perf/ffpa_speedup_nvidia-geforce-rtx-5090_B1_H32_N8192_D512_T.png' width='350px'><br>
  <img src='./assets/perf/ffpa_speedup_nvidia-h800-pcie_B1_H32_N8192_D320_T.png' width='350px'>
  <img src='./assets/perf/ffpa_speedup_nvidia-h800-pcie_B1_H32_N8192_D512_T.png' width='350px'><br>
  <img src='./assets/perf/ffpa_speedup_cutedsl_nvidia-h20z_B1_H32_N8192_D512_T.png' width='350px'>
  <img src='./assets/perf/ffpa_speedup_cutedsl_nvidia-h20z_B1_H32_N16384_D512_T.png' width='350px'>
</div>

## 🤖 Backends

FFPA supports multiple backends for the forward and backward pass, including: [`SDPA`](./examples/) (baseline), [`CUDA`](./examples/) (forward only), [`Triton`](./examples/), and [`CuTeDSL`](./examples/). The `CuTeDSL` backend is currently in early stage and has some constraints (e.g., D <= 512), but it can achieve up to `427🎉` TFLOPS on H200! Stay tuned for future updates.

<div align='center' markdown="1">

|Backend|Arch|Fwd|Bwd|Headdim|Autotune|Speedup|Recommend|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|SDPA|sm>=75|✔|✔|All|❌|**1.0x**|sm>=75|
|CUDA|sm>=80|✔|❌|320~1024|❌|**1.5x~3x**🎉|sm80~89|
|Triton|sm>=80|✔|✔|320~1024|✔|**1.5x~3x**🎉|sm>=80|
|CuTeDSL|sm80~89|✔|✔|320~1024|❌|**1.5x~2x**🎉|sm80~89|
|CuTeDSL|sm90|✔|✔|320~512|❌|**3x~6x**🎉|sm90|

<i>Special thanks to [Butterfingrz](https://github.com/Butterfingrz) for contributing to the CuTeDSL backend! Awesome work!🎉</i>

</div>

How to use different backends for your own scenario? Users can simply pass the Backend configs (SDPABackend, CUDABackend, TritonBackend or CuTeDSLBackend) to [ffpa_attn_func](https://ffpa-attn.readthedocs.io/en/latest/api/ffpa_attn/), for example:

```python
>>> from ffpa_attn import ffpa_attn_func, CuTeDSLBackend
>>> # CuTeDSL backend, D=512 scenario, fastest on H200!🎉
>>> o = ffpa_attn_func(q, k, v, backend=CuTeDSLBackend())
```

## ©️License

<a id="License"></a>

Apache License 2.0

## ©️Citations

```BibTeX
@misc{ffpa-attn@2025,
  title={FFPA: Yet another Faster Flash Prefill Attention for large headdim.},
  url={https://github.com/xlite-dev/ffpa-attn.git},
  note={Open-source software available at https://github.com/xlite-dev/ffpa-attn.git},
  author={DefTruth, Butterfingrz},
  year={2025}
}
```

## 📖 References
<a id="ref"></a>

- [flash-attention](https://github.com/Dao-AILab/flash-attention)
- [LeetCUDA](https://github.com/xlite-dev/LeetCUDA)
- [flashinfer](https://github.com/flashinfer-ai/flashinfer)
- [quack](https://github.com/Dao-AILab/quack)
- [cutlass](https://github.com/NVIDIA/cutlass)

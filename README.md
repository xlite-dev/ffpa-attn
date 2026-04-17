<div align="center">
  <p align="center">
    <h2>🤖FFPA(Split-D): Yet another Faster Flash Prefill Attention with O(1)⚡️GPU SRAM complexity for large headdim🐑</h2>
    <a href="https://zhuanlan.zhihu.com/p/13975660308">📚FFPA(Split-D) Blog</a> | <a href="#bench-l20"> 📈L20 ~1.9x↑🎉 </a> | <a href="#bench-a30"> 📈A30 ~1.8x↑🎉 </a> | <a href="#bench-3080"> 📈3080 ~2.9x↑🎉 </a> | <a href="#bench-4090"> 📈4090 ~2.1x↑🎉 </a>
  </p>
  <img src='https://github.com/user-attachments/assets/447e2937-f7c8-47c8-8550-8c0c71b910e6' width="411px">
  <img src='https://github.com/user-attachments/assets/65a8d564-8fa7-4d66-86b9-e238feb86143' width="411px">
</div>

🤖**FFPA(Split-D)**: Yet another **Faster Flash Prefill Attention** with **Split-D** strategy, achieve **O(1) SRAM complexity** and **O(d/4) register complexity** for large headdim (**D > 256**), **1.8x~3x** 🎉 faster than SDPA on many devices.
## ©️Citations🎉🎉

```BibTeX
@misc{ffpa-attn@2025,
  title={FFPA: Yet another Faster Flash Prefill Attention for large headdim.},
  url={https://github.com/xlite-dev/ffpa-attn.git},
  note={Open-source software available at https://github.com/xlite-dev/ffpa-attn.git},
  author={DefTruth},
  year={2025}
}
```

## 📖 Quick Start

<div id="install"></div>

First, clone the repo and build the package from source: (Note: `pip uninstall ffpa-attn -y` if you want to reinstall after code changes)
```bash
git clone https://github.com/xlite-dev/ffpa-attn.git
python3 setup.py bdist_wheel
pip3 install dist/*.whl # pip uninstall ffpa-attn -y
```

> [!NOTE]
> FFPA currently only supports **equal-length Q/K/V self-attention** (i.e. `Nq == Nk == Nv`). **Causal attention (causal mask)** and cross-attention / incremental decoding where KV length differs from Q length are **not supported yet**.

Minimal usage example (B=1, H=32, N=8192, D=512):
```python
import torch
import torch.nn.functional as F
from ffpa_attn import ffpa_attn_func

B, H, N, D = 1, 32, 8192, 512
q = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda")
k = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda")
v = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda")

# FFPA prefill attention; layout follows SDPA: (B, H, N, D).
out = ffpa_attn_func(q, k, v)  # -> torch.Tensor of shape (B, H, N, D)
print(out.shape, out.dtype)

# Accuracy check against PyTorch SDPA (same bf16 dtype).
ref = F.scaled_dot_product_attention(q, k, v)
max_abs = (out - ref).abs().max().item()
mean_abs = (out - ref).abs().mean().item()
print(f"vs SDPA: max_abs_err={max_abs:.4e}, mean_abs_err={mean_abs:.4e}")
```

A runnable end-to-end example (with SDPA accuracy/perf comparison, both aligned N=8192 and non-aligned N=8191 cases) is provided under [`examples/run_ffpa_attn.py`](./examples/run_ffpa_attn.py):

```bash
CUDA_VISIBLE_DEVICES=0 python3 examples/run_ffpa_attn.py
```

## 📖 Contents

- [📖 FFPA Design](#ffpa-design)
- [📈 FFPA: L20 ~1.9x↑🎉](#bench-l20)
- [📈 FFPA: A30 ~1.8x↑🎉](#bench-a30)
- [📈 FFPA: 3080 ~2.9x↑🎉](#bench-3080)
- [📈 FFPA: 4090 ~2.1x↑🎉](#bench-4090)

## 📖 FFPA Design

<div id="ffpa-design"></div>

We have extended FlashAttention for large headdim (D > 256) by implementing **Fine-grained Tiling** at the **MMA level (GEMM style)** for the Q@K^T and P@V matmul. This approach results in a constant SRAM usage of Br * 16 or Bc * 16 (Br = Bc) for Q, K, and V, leading to an overall SRAM complexity of O(2 * Br * 16) ≈ O(1) and a register complexity of O(d/4). Consequently, this method allows us to extend headdim beyond 256 and achieve faster performance compared to SDPA with or without MMA Accumulation F32 (**1.8x~3x** 🎉 faster than SDPA EA).

<div align='center'>
  <img src=https://github.com/user-attachments/assets/ed30185b-2e11-4293-832f-43e9003d6ad9 >
</div>

We have named this new attention tiling technique **FFPA: Faster Flash Prefill Attention**. FFPA does not introduce any additional VRAM requirement, so the HBM memory complexity remains the same as FlashAttention.

By leveraging this approach, we can achieve better performance than SDPA EA for very large headdim (D > 256, `FA-2 not supported`). Approximate SRAM and register complexity analysis for FFPA is as follows: (`d`=headdim, `C,Br,Bc`=Constant, `Br=Bc`, let O(C)≈O(1)) 👇

<div align='center'>

|📚Complexity| 📚FFPA | 📚FA-2 |
|:---:|:---:|:---:|
|SRAM | O(2xBrx16)≈O(1) | ≈O(3xBrxd), d↑ |
|Register | ≈O(d/4), d↑ | ≈O(d/2), d↑ |
|HBM| ≈FA2≈O(Nd), O | ≈O(Nd), O |
|Extra HBM| ≈FA2≈O(N), m,l | ≈O(N), m,l |

</div>

**📚👇Core Features🎉🎉**: FFPA is implemented using pure MMA PTX instructions, which supports many features such as Split-Q, SMEM Swizzle/Padding, QKV Multi-Stages(1~4), Tile MMAs/Warps, Mixed MMA F32/F16 Acc (Q@K^T MMA Acc F32 + P@V MMA Acc F16), Fully Shared QKV SMEM, Prefetch QKV g2s, Persist Q s2r/g2s, **Fully QKV Fine-grained Tiling(GEMM style)**, Collective Store, etc.

<div align='center'>

|📚Feature |📚Feature |📚Feature |📚Feature|
|:---:|:---:|:---:|:---:|
|✔️Tensor Cores |✔️**MMA(m16n8k16)** |✔️Tile Block(Br, Bc) |✔️Tile MMA/Warp |
|✔️**Split Q**(FA-2)|✔️Pack LDST(128 bits)|✔️SMEM **Swizzle/Pad** |✔️Copy Async |
|✔️**Reg Double Buffers** |✔️QKV **Multi-Stages(1~4)** |✔️Collective Store(**Shfl**)|✔️**Prefetch QKV** g2s |
|✔️**QKV Fine-grained Tiling**|✔️**Shared QKV** SMEM|✔️Mixed MMA Acc|✔️**Persist Q** s2r/g2s|

</div>

## 📖 Prerequisites
<div id="prerequisites"></div>

- Python >= 3.12
- PyTorch >= 2.9.0, CUDA >= 12.9
- flash-attention >= 2.6.3 (for test)
- Recommended: PyTorch 2.11.0, CUDA 13.0

## 📖 FFPA Benchmark 🎉🎉

<div id="bench-l20"></div>

O(2xBrx16)≈O(1) SRAM complexity, O(d/4) register complexity, the same GPU HBM memory complexity as FlashAttention. B=1, H=48, N=8192, **D=320-1024(FA2 not supported 👀)**. (Notes, *=MMA Acc F32, ^=MMA Acc F16, Softmax Acc dtype is always be F32, T=TFLOPS, 👇Benchmark)

<div align='center'>

<p>📚 NVIDIA L20 (*=MMA Acc F32, ^=MMA Acc F16, T=TFLOPS, <b>~1.8x↑🎉</b>)</p>

|Algorithm|320|384|448|512|576|640|704|768|832|896|960|1024|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|SDPA EA|56T|63T|58T|58T|55T|56T|54T|55T|54T|55T|54T|56T|
|FFPA*|102T|102T|103T|104T|103T|95T|95T|95T|95T|96T|95T|94T|
|Speedup|1.82x|1.62x|1.78x|1.79x|1.87x|1.7x|1.76x|1.73x|1.76x|1.75x|1.76x|1.68x|
|FFPA^|104T|103T|103T|102T|104T|103T|102T|94T|94T|94T|100T|100T|
|Speedup|1.86x|1.63x|1.78x|1.76x|1.89x|1.84x|1.89x|1.71x|1.74x|1.71x|1.85x|1.79x|

<p>📚 NVIDIA L20 (*=MMA Acc: QK F32 + PV F16, ^=MMA Acc F16, T=TFLOPS, <b>~1.9x↑🎉</b>)</p>

|Algorithm|320|384|448|512|576|640|704|768|832|896|960|1024|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|SDPA EA|56T|64T|58T|58T|55T|56T|54T|55T|54T|55T|54T|56T|
|FFPA*|105T|102T|104T|103T|105T|95T|95T|94T|94T|94T|102T|101T|
|Speedup|1.88x|1.59x|1.79x|1.78x|1.91x|1.7x|1.76x|1.71x|1.74x|1.71x|1.89x|1.8x|
|FFPA^|104T|103T|103T|102T|103T|103T|102T|94T|94T|94T|100T|100T|
|Speedup|1.86x|1.61x|1.78x|1.76x|1.87x|1.84x|1.89x|1.71x|1.74x|1.71x|1.85x|1.79x|

<div align='center'>
  <img src='https://github.com/user-attachments/assets/a4927108-3f97-4209-9b80-bb31ad271e04' width="411px">
  <img src='https://github.com/user-attachments/assets/eeb9943f-919d-45d8-a8a6-e0f8874f4bcd' width="411px">
</div>

</div>

<div id="bench-a30"></div>

<div align='center'>

<p>📚 NVIDIA A30 (*=MMA Acc F32, ^=MMA Acc F16, T=TFLOPS, <b>~1.8x↑🎉</b>)</p>

|Algorithm|320|384|448|512|576|640|704|768|832|896|960|1024|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|SDPA EA|25T|25T|24T|24T|24T|24T|23T|22T|22T|22T|22T|18T|
|FFPA*|45T|44T|44T|43T|43T|38T|37T|37T|37T|36T|33T|32T|
|Speedup|1.8x|1.76x|1.83x|1.79x|1.79x|1.58x|1.61x|1.68x|1.68x|1.64x|1.5x|1.78x|
|FFPA^|48T|46T|45T|43T|44T|44T|44T|38T|37T|36T|40T|34T|
|Speedup|1.92x|1.84x|1.88x|1.79x|1.83x|1.83x|1.91x|1.73x|1.68x|1.64x|1.82x|1.89x|

<p>📚 NVIDIA A30 (*=MMA Acc: QK F32 + PV F16, ^=MMA Acc F16, T=TFLOPS, <b>~1.9x↑🎉</b>)</p>

|Algorithm|320|384|448|512|576|640|704|768|832|896|960|1024|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|SDPA EA|25T|25T|24T|24T|24T|24T|23T|22T|22T|22T|22T|18T|
|FFPA*|48T|46T|46T|43T|44T|38T|38T|38T|37T|36T|40T|34T|
|Speedup|1.92x|1.84x|1.92x|1.79x|1.83x|1.58x|1.65x|1.73x|1.68x|1.64x|1.82x|1.89x|
|FFPA^|48T|46T|45T|43T|44T|44T|44T|38T|37T|36T|39T|34T|
|Speedup|1.92x|1.84x|1.88x|1.79x|1.83x|1.83x|1.91x|1.73x|1.68x|1.64x|1.77x|1.89x|

<div align='center'>
  <img src='https://github.com/user-attachments/assets/7e323005-4445-41af-8e94-6efb62ed2b77' width="411px">
  <img src='https://github.com/user-attachments/assets/e314649e-82b5-414d-85c9-8b6fbf260138' width="411px">
</div>

</div>

<div id="bench-3080"></div>

<div align='center'>

<p>📚 NVIDIA RTX 3080 Laptop (*=MMA Acc F32, ^=MMA Acc F16, T=TFLOPS, <b>~2.5x↑🎉</b>)</p>

|Algorithm|320|384|448|512|576|640|704|768|832|896|960|1024|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|SDPA EA|13T|16T|11T|16T|15T|15T|15T|15T|14T|14T|14T|14T|
|FFPA*|33T|31T|30T|30T|30T|27T|27T|26T|26T|26T|26T|25T|
|Speedup|2.54x|1.94x|2.73x|1.88x|2.0x|1.8x|1.8x|1.73x|1.86x|1.86x|1.86x|1.79x|
|FFPA^|43T|41T|39T|39T|39T|39T|39T|36T|34T|33T|31T|33T|
|Speedup|3.31x|2.56x|3.55x|2.44x|2.6x|2.6x|2.6x|2.4x|2.43x|2.36x|2.21x|2.36x|

<p>📚 NVIDIA RTX 3080 Laptop (*=MMA Acc: QK F32 + PV F16, ^=MMA Acc F16, T=TFLOPS, <b>~2.9x↑🎉</b>)</p>

|Algorithm|320|384|448|512|576|640|704|768|832|896|960|1024|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|SDPA EA|13T|15T|12T|15T|14T|15T|14T|14T|14T|14T|14T|14T|
|FFPA*|38T|36T|34T|35T|34T|31T|32T|31T|30T|28T|27T|27T|
|Speedup|2.92x|2.4x|2.83x|2.33x|2.43x|2.07x|2.29x|2.21x|2.14x|2.0x|1.93x|1.93x|
|FFPA^|44T|41T|39T|39T|38T|39T|39T|36T|34T|32T|31T|33T|
|Speedup|3.38x|2.73x|3.25x|2.6x|2.71x|2.6x|2.79x|2.57x|2.43x|2.29x|2.21x|2.36x|

<div align='center'>
  <img src='https://github.com/user-attachments/assets/d157cd69-4444-4735-a691-edaaff408137' width="411px">
  <img src='https://github.com/user-attachments/assets/3ce47627-e79d-40ee-b753-bdd235603b7d' width="411px">
</div>

</div>

<div id="bench-4090"></div>

<div align='center'>
<p>📚 NVIDIA RTX 4090 (*=MMA Acc F32, ^=MMA Acc F16, T=TFLOPS, <b>~1.8x↑🎉</b>)</p>

|Algorithm|320|384|448|512|576|640|704|768|832|896|960|1024|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|SDPA EA|81T|94T|85T|85T|79T|81T|79T|80T|79T|80T|78T|78T|
|FFPA*|149T|150T|150T|150T|150T|140T|140T|140T|139T|139T|137T|134T|
|Speedup|1.84x|1.6x|1.76x|1.76x|1.9x|1.73x|1.77x|1.75x|1.76x|1.74x|1.76x|1.72x|
|FFPA^|194T|194T|189T|191T|197T|188T|184T|180T|177T|172T|171T|171T|
|Speedup|2.4x|2.06x|2.22x|2.25x|2.49x|2.32x|2.33x|2.25x|2.24x|2.15x|2.19x|2.19x|

<p>📚 NVIDIA RTX 4090 (*=MMA Acc: QK F32 + PV F16, ^=MMA Acc F16, T=TFLOPS, <b>~2.1x↑🎉</b>)</p>

|Algorithm|320|384|448|512|576|640|704|768|832|896|960|1024|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|SDPA EA|82T|92T|85T|84T|78T|81T|79T|80T|78T|79T|77T|78T|
|FFPA*|176T|170T|171T|171T|171T|161T|160T|161T|160T|158T|165T|164T|
|Speedup|2.15x|1.85x|2.01x|2.04x|2.19x|1.99x|2.03x|2.01x|2.05x|2.0x|2.14x|2.1x|
|FFPA^|200T|191T|189T|191T|188T|188T|186T|179T|175T|173T|172T|170T|
|Speedup|2.44x|2.08x|2.22x|2.27x|2.41x|2.32x|2.35x|2.24x|2.24x|2.19x|2.23x|2.18x|

<div align='center'>
  <img src='https://github.com/user-attachments/assets/447e2937-f7c8-47c8-8550-8c0c71b910e6' width="411px">
  <img src='https://github.com/user-attachments/assets/65a8d564-8fa7-4d66-86b9-e238feb86143' width="411px">
</div>
</div>

## ©️License

<div id="License"></div>

GNU General Public License v3.0

## 🎉Contribute

<div id="Contribute"></div>

How to contribute? Wecome to star⭐️ this repo to support me👆🏻 ~

<div align='center'>
<a href="https://star-history.com/#xlite-dev/ffpa-attn&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=xlite-dev/ffpa-attn&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=xlite-dev/ffpa-attn&type=Date" />
   <img img width=450 height=300 alt="Star History Chart" src="https://api.star-history.com/svg?repos=xlite-dev/ffpa-attn&type=Date" />
 </picture>
</a>
</div>

## 📖 References
<div id="ref"></div>

- [flash-attention](https://github.com/Dao-AILab/flash-attention)
- [LeetCUDA](https://github.com/xlite-dev/LeetCUDA)
- [flashinfer](https://github.com/flashinfer-ai/flashinfer)

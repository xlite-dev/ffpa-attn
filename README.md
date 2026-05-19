<div align="center">
  <p align="center">
    <h2>🤖FFPA: Yet another Faster Flash Prefill Attention <br>with O(1)⚡️GPU SRAM complexity for large headdim🐑</h2>
    <img src=https://img.shields.io/badge/language-CUDA/Python-brightgreen.svg >
    <a href="https://pepy.tech/projects/ffpa-attn"><img src=https://static.pepy.tech/personalized-badge/ffpa-attn?period=total&units=ABBREVIATION&left_color=GRAY&right_color=BLUE&left_text=downloads/pypi ></a>
    <a href="https://pypi.org/project/ffpa-attn/"><img src=https://img.shields.io/github/release/xlite-dev/ffpa-attn.svg?color=GREEN ></a>
    <img src=https://img.shields.io/github/stars/xlite-dev/ffpa-attn.svg?style=dark >
    <img src="https://img.shields.io/github/license/xlite-dev/ffpa-attn.svg?color=blue"><br>
    <img src="docs/assets/ffpa-api.png" width="700px">
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

## 📖 Split-D

<a id="ffpa-design"></a>

We extend FlashAttention to support large headdim ($D>256$) via **fine-grained tiling** at the **MMA** level for $QK^\top$ and $PV$ matrix multiplication, referred to as **Split-D**. This design keeps SRAM usage fixed at $B_r \times 16$ (with $B_r=B_c$) for Q, K and V, yielding constant SRAM complexity $O(B_r \times 16) \approx O(1)$ and register complexity $O(d/4)$.

<div align='center'>
  <img src="./docs/assets/split-d.png" width="700px">
  </p><i>
    <b>FFPA</b> enables headdim <b> > 256</b>, and outperforms standard SDPA by <b>1.5~3x</b>🎉.
  </i></p>
</div>

> [!NOTE]
> FFPA has been tested on `Ampere`, `Ada`, `Hopper`, and `Blackwell` architectures (e.g., A30, L20, 4090, H200, 5090), achieves `1.5~3×↑🎉` speedup over SDPA. FFPA is mainly design for **prefill** and large headdim, and may not be faster than SDPA for 😈 small sequence length (`N<512`) or small headdim (`D<=256`).

## 🎉 Benchmark

Runnable examples are provided under [`examples`](./examples). The performance benchmarks for the NVIDIA L20 (**Ada**), NVIDIA Geforce RTX 5090 (**Blackwell**), NVIDIA H800 PCIE (**Hopper**), NVIDIA H200 SXM (**Hopper**, **CuTeDSL** backend, up to **427** TFLOPS!🎉) with large headdim are shown below:

<div align='center'>
  <img src='./docs/assets/perf/ffpa_speedup_nvidia-l20_B1_H32_N8192_D320_T.png' width='400px'>
  <img src='./docs/assets/perf/ffpa_speedup_nvidia-l20_B1_H32_N8192_D512_T.png' width='400px'><br>
  <img src='./docs/assets/perf/ffpa_speedup_nvidia-geforce-rtx-5090_B1_H32_N8192_D320_T.png' width='400px'>
  <img src='./docs/assets/perf/ffpa_speedup_nvidia-geforce-rtx-5090_B1_H32_N8192_D512_T.png' width='400px'><br>
  <img src='./docs/assets/perf/ffpa_speedup_nvidia-h800-pcie_B1_H32_N8192_D320_T.png' width='400px'>
  <img src='./docs/assets/perf/ffpa_speedup_nvidia-h800-pcie_B1_H32_N8192_D512_T.png' width='400px'><br>
  <img src='./docs/assets/perf/ffpa_speedup_cutedsl_nvidia-h20z_B1_H32_N8192_D512_T.png' width='400px'>
  <img src='./docs/assets/perf/ffpa_speedup_cutedsl_nvidia-h20z_B1_H32_N16384_D512_T.png' width='400px'>
</div>

## 🤖 Backends

FFPA supports multiple backends for the forward and backward pass, including: `CUDA` (forward only), `Triton`, and `CuTeDSL`. The CuTeDSL backend is currently in early stage and has some constraints (e.g., D=512 only), but it can achieve up to `427🎉` TFLOPS on H200, which is very exciting! We will continue to optimize the CuTeDSL backend and lift the constraints in the future.

<div align='center' markdown="1">

|Backend|Arch|Fwd.|Bwd.|Autotune|Feat.|Headdim|Recomm.|Speedup|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|CUDA|Ampere+|✔|❌|❌|All|320~1024|Ampere, Ada|**1.5x~3x**🎉|
|Triton|Ampere+|✔|✔|✔|All|320~1024|Ampere+|**1.5x~3x**🎉|
|CuTeDSL|Hopper|✔|✔|❌|Limited|512|Hopper|**3x~5x**🎉|

</div>

## ©️License

<div id="License"></div>

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
<div id="ref"></div>

- [flash-attention](https://github.com/Dao-AILab/flash-attention)
- [LeetCUDA](https://github.com/xlite-dev/LeetCUDA)
- [flashinfer](https://github.com/flashinfer-ai/flashinfer)

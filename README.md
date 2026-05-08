<div align="center">
  <p align="center">
    <h2>🤖FFPA: Yet another Faster Flash Prefill Attention <br>with O(1)⚡️GPU SRAM complexity for large headdim🐑</h2>
    <a href="./bench/README.md#bench-l20"> 📈L20 ~1.9x↑🎉 </a> | <a href="./bench/README.md#bench-a30"> 📈A30 ~1.8x↑🎉 </a> | <a href="./bench/README.md#bench-3080"> 📈3080 ~2.9x↑🎉 </a> | <a href="./bench/README.md#bench-4090"> 📈4090 ~2.1x↑🎉 </a>
  </p>
  <img src="docs/assets/ffpa-api.png" width="700px">
</div>

**FFPA(Split-D)**: Yet another **Faster Flash Prefill Attention** with **Split-D** strategy, achieve **O(1) SRAM complexity** and **O(d/4) register complexity** for large headdim (**> 256**), **1.8x~3x** 🎉 faster than SDPA. 👇Core features:

<div align='center'>

|[Self Attn](./examples)| [GQA/MQA](./examples) |[Cross Attn](./examples)|[Causal Attn](./examples)|[Headdim](#ffpa-design)|[Forward↑](./examples)|[Backward↑](./examples)|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|✔️(`Nq=Nkv`)|✔️(`Hq!=Hkv`)|✔️(`Nq!=Nkv`)|✔️(`causal`)|**320~1024** |**1.8x~3x↑🎉** |**1.5x~2.5x↑🎉** |

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
# Optional: build the whl with Triton and CUDA backends
ENABLE_FFPA_FWD_CUDA_IMPL=1 && MAX_JOBS=32 pip3 install -e .
```

Then, try to accelerate the attention for large headdim with just <i><b>one-line</b></i> of code:

```python
>>> import torch.nn.functional as F
>>> from ffpa_attn import ffpa_attn_func
>>> # Monkey-patch SDPA to point to FFPA attention. Every thing that
>>> # FFPA does not support will automatically fallback to SDPA. For
>>> # example, if the user calls SDPA with headdim <= 256 or > 1024,
>>> # attn_mask not None, dropout_p > 0.0, and N < 512, etc.
>>> F.scaled_dot_product_attention = ffpa_attn_func # one-line code
```

For more advanced features, please refer to our online docs at 📘[ffpa-attn.io](https://ffpa-attn.readthedocs.io/en/latest/).

## 📖 Split-D

<a id="ffpa-design"></a>

We extend FlashAttention to support large headdim ($D>256$) via **fine-grained tiling** at the **MMA** level for $QK^\top$ and $PV$ matrix multiplication, referred to as **Split-D**. This design keeps SRAM usage fixed at $B_r \times 16$ (with $B_r=B_c$) for Q, K and V, yielding constant SRAM complexity $O(B_r \times 16) \approx O(1)$ and register complexity $O(d/4)$.

<div align='center'>
  <img src=https://github.com/user-attachments/assets/ed30185b-2e11-4293-832f-43e9003d6ad9 width="700px">
  </p><i>
    <b>FFPA</b> enables headdim <b> > 256</b>, and outperforms standard SDPA by <b>1.8x~3x</b>🎉.
  </i></p>
</div>

> [!NOTE]
> FFPA has been tested on `Ampere`, `Ada`, `Hopper`, and `Blackwell` architectures (e.g., A30, L20, 4090, H200, 5090), achieves `1.8×~3×↑🎉` forward and `1.5×~2.5×↑🎉` backward padd speedup over SDPA. Currently, FFPA is mainly design for **prefill** (`N>=512`) and large headdim (`D>256`), and may not be faster than SDPA for small sequence length (`N<512`) or small headdim (`D<=256`).

## 🎉 Benchmark

Runnable examples are provided under [`examples`](./examples). The performance benchmark for the 4090 with large headdim (D=320~1024) is shown below. Please refer to our [bench](./bench/README.md) for more details.

<div align='center'>
  <img src='https://github.com/user-attachments/assets/447e2937-f7c8-47c8-8550-8c0c71b910e6' width="370px">
  <img src='https://github.com/user-attachments/assets/65a8d564-8fa7-4d66-86b9-e238feb86143' width="370px">
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
  author={DefTruth},
  year={2025}
}
```

## 📖 References
<div id="ref"></div>

- [flash-attention](https://github.com/Dao-AILab/flash-attention)
- [LeetCUDA](https://github.com/xlite-dev/LeetCUDA)
- [flashinfer](https://github.com/flashinfer-ai/flashinfer)

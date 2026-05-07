<div align="center">
  <p align="center">
    <h2>🤖FFPA: Yet another Faster Flash Prefill Attention <br>with O(1)⚡️GPU SRAM complexity for large headdim🐑</h2>
    <a href="./bench/README.md#bench-l20"> 📈L20 ~1.9x↑🎉 </a> | <a href="./bench/README.md#bench-a30"> 📈A30 ~1.8x↑🎉 </a> | <a href="./bench/README.md#bench-3080"> 📈3080 ~2.9x↑🎉 </a> | <a href="./bench/README.md#bench-4090"> 📈4090 ~2.1x↑🎉 </a>
  </p>
  <img src="docs/assets/ffpa-api.png" width="700px">
</div>

**FFPA(Split-D)**: Yet another **Faster Flash Prefill Attention** with **Split-D** strategy, achieve **O(1) SRAM complexity** and **O(d/4) register complexity** for large headdim (**> 256**), **1.8x~3x** 🎉 faster than SDPA. Currently, FFPA supports self-attention, cross-attention, grouped/multi-query attention, causal attention with large headdim (D=320~1024). While the standard FlashAttention-2 only support headdim <= 256.

<div align='center'>

|[Self Attention](#example-self)| [Cross/Decode Attention](#example-cross)|[GQA/MQA Attention](#example-gqa)|[Causal Attention](#example-causal)|[Headdim](#ffpa-design)|
|:---:|:---:|:---:|:---:|:---:|
|✔️(`Nq = Nkv`)|✔️(`Nq != Nkv`)|✔️(`Nh_q % Nh_kv == 0`)|✔️(`causal mask`)|**32~1024** |

</div>

> [!NOTE]
> FFPA has been tested on `Ampere`, `Ada`, `Hopper`, and `Blackwell` architectures (e.g., A30, L20, 4090, H200, 5090), achieves `1.8×~3×↑🎉` forward (CUDA) and `1.5×~2.5×↑🎉` backward (Triton w/ autotune) speedup over SDPA for headdim `> 256`.

## 📖 Quick Start

<div id="install"></div>

First, install the prebuilt package from [PyPI](https://pypi.org/project/ffpa-attn/) or build [ffpa-attn](https://github.com/xlite-dev/ffpa-attn) from source:

```bash
# Required: PyTorch>=2.11.0, CUDA>=13.0, Ubuntu>=22.04
pip3 install -U ffpa-attn # (support: sm_{80,90,...,120})
# Or, build ffpa-attn from source, just follow the cmds:
git clone https://github.com/xlite-dev/ffpa-attn.git
# Then, build the wheel package and install it with pip
cd ffpa-attn && MAX_JOBS=32 python3 setup.py bdist_wheel
# Optional: build ffpa-attn with ccache for faster rebuilds
apt install ccache && bash tools/build_fast.sh bdist_wheel
# Optional: for editable whl, use `pip install -e .` instead.
pip3 install dist/ffpa_attn-*.whl # pip uninstall ffpa-attn -y
```

Then, try to accelerate your attention computations with just ♥️one line♥️ of code ~

```python
>>> import torch.nn.functional as F
>>> from ffpa_attn import ffpa_attn_func
>>> # Monkey-patch SDPA to point to FFPA attention. Every thing that
>>> # FFPA does not support will automatically fallback to SDPA. For
>>> # example, if the user calls SDPA with headdim <= 256 or > 1024,
>>> # attn_mask not None, and dropout_p > 0.0, etc.
>>> F.scaled_dot_product_attention = ffpa_attn_func
```

For more advanced features, please refer to our online docs at 📘[ffpa-attn.io](https://ffpa-attn.readthedocs.io/en/latest/).

## 📖 Split-D

<a id="ffpa-design"></a>

We have extended FlashAttention for large headdim (D > 256) by implementing **Fine-grained Tiling** at the **MMA level (GEMM style)** for the Q@K^T and P@V matmul (namely, **Split-D**). This approach results in a constant SRAM usage of Br * 16 or Bc * 16 (Br = Bc) for Q, K, and V, leading to an overall SRAM complexity of O(Br * 16) ≈ O(1) and a register complexity of O(d/4). Consequently, this method allows us to extend headdim > 256 and achieve faster performance compared to SDPA with or without MMA Accumulation F32 (**1.8x~3x** 🎉 faster than SDPA EA).

<div align='center'>
  <img src=https://github.com/user-attachments/assets/ed30185b-2e11-4293-832f-43e9003d6ad9 >
</div>

## 🎉 Benchmark

Runnable examples are provided under [`examples`](./examples). The performance benchmark for the 4090 with large headdim (D=320~1024) is shown below. Please refer to our [bench](./bench/README.md) for more details.

<div align='center'>
  <img src='https://github.com/user-attachments/assets/447e2937-f7c8-47c8-8550-8c0c71b910e6' width="411px">
  <img src='https://github.com/user-attachments/assets/65a8d564-8fa7-4d66-86b9-e238feb86143' width="411px">
</div>

<!--
## 🤔 Why not TMA?

<div id="why-not-tma"></div>

FFPA ships an experimental SM90 TMA path (**enable_tma=True**) that replaces the K/V [cp.async](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=cp%2520async#data-movement-and-conversion-instructions-cp-async) global-to-shared transfer with [cp.async.bulk.tensor.2d](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=cp%2520async%2520bulk%2520tensor%25202d#data-movement-and-conversion-instructions-cp-async-bulk-tensor). After tuning (K SWIZZLE_128B, 64-col TMA box) it reaches parity with the cp.async baseline, but does not beat it.

**FFPA's Split-D dataflow is a TMA anti-pattern**. TMA wins when single thread instruction can amortise its dispatch cost over a large box, but split-D gives it narrow **Bc** x **kMmaAtomK** slices. It would require a major redesign (**super-tiled K/V on TMA** + **warp-specialized** [WGMMA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=cp%2520async%2520bulk%2520tensor%25202d#asynchronous-warpgroup-level-matrix-instructions)), rather than a drop-in K/V replacement.
-->

## ©️License

<div id="License"></div>

Apache License 2.0

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

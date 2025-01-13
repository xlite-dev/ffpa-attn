<div align='center'>
  <img src=https://github.com/user-attachments/assets/9f764ccf-3dce-43c2-b2ae-aa068231dea2 >
</div>

<div align='center'>
  <img src=https://img.shields.io/badge/Language-CUDA/Python-brightgreen.svg >
  <img src=https://img.shields.io/github/watchers/DefTruth/faster-prefill-attention?color=9cc >
  <img src=https://img.shields.io/github/forks/DefTruth/faster-prefill-attention.svg?style=social >
  <img src=https://img.shields.io/github/stars/DefTruth/faster-prefill-attention.svg?style=social >
  <img src=https://img.shields.io/badge/Release-v0.0.1-brightgreen.svg >
  <img src=https://img.shields.io/badge/License-GPLv3.0-turquoise.svg >
 </div>

🤖[WIP] **FFPA**: Yet antother **Faster Flash Prefill Attention** with **O(1) SRAM complexity** & **O(d/4) or O(1) register complexity** for large headdim (D > 256), almost **1.5x~2x** 🎉 faster than SDPA EA with or without MMA Acc F32 on many devices: [📈L20 ~1.7x↑🎉](#L1-bench), [📈 A30 ~1.5x↑🎉](#L1-bench), [📈3080 ~2.5x↑🎉](#L1-bench), [📈4090 ~1.8x↑🎉](#L1-bench). 👇Features:🎉🎉

|Tensor Cores|Loop over N/D |Tile Block (Br, Bc) |MMA (m16n8k16)|
|:---:|:---:|:---:|:---:|
|✔️|✔️|✔️|✔️|
|**Split Q** (FA-2)|Pack LDST (128 bits)|SMEM **Swizzle**/Padding |Copy Async |
|✔️|✔️|✔️|✔️|
|Tile MMA (More Threads)|Tile Warp (More Values) |Multi Stages (1~4) |Collective Store (**Shfl**)|
|✔️|✔️|✔️|✔️|
|**QKV Fine-grained Tiling**|**Shared QKV** SMEM|Mixed F32/F16 MMA Acc|**FFPA L1**|
|✔️|✔️|✔️|✔️|
<!--
-->

💡NOTE: This project is still in its early dev stages and now provides some kernels and benchmarks for reference. More features will be added in the future. Welcome to 🌟👆🏻star this repo to support me ~ 🎉🎉

## ©️Citations🎉🎉

```BibTeX
@misc{ffpa-attn@2025,
  title={FFPA: Yet another Faster Flash Prefill Attention for large headdim.},
  url={https://github.com/DefTruth/ffpa-attn-mma.git},
  note={Open-source software available at https://github.com/DefTruth/ffpa-attn-mma.git},
  author={DefTruth etc},
  year={2025}
}
```

## 📖 Contents

- [📖 Installation⚙️](#install)
- [📖 Python Testing👇](#python-test)
- [📖 FFPA L1~L3 Design💡](#ffpa-design)
- [📈 FFPA L1: L20 ~1.7x↑🎉](#L1-bench)
- [📈 FFPA L1: A30 ~1.5x↑🎉](#L1-bench)
- [📈 FFPA L1: 3080 ~2.5x↑🎉](#L1-bench)
- [📈 FFPA L1: 4090 ~1.8x↑🎉](#L1-bench)

## 📖 FFPA L1~L3: FlashAttention + QKV Fine-grained Tiling at MMA level💡
<div id="ffpa-design"></div>

We have extended FlashAttention for large headdim (D > 256) by implementing **Fine-grained Tiling** at the **MMA level (GEMM style)** for the Q@K^T and P@V matmul. This approach results in a constant SRAM usage of Br * 16 or Bc * 16 (Br = Bc) for Q, K, and V, leading to an overall SRAM complexity of O(2 * Br * 16) ≈ O(1) and a register complexity of O(d/4) or O(1). Consequently, this method allows us to extend **headdim > 256** and achieve faster performance compared to SDPA with or without MMA Accumulation F32 (almost **1.5x~2x** 🎉 faster than SDPA EA).

We have named this new attention tiling technique **FFPA: Faster Flash Prefill Attention**. We have designed three `(L1~L3)` levels of FFPA based on SRAM and register complexity considerations. All levels will not introduce any additional VRAM requirements, ensuring that the HBM memory complexity remains same as FlashAttention. 👇

- [x] 📚L1: level 1, O(2xBrx16)≈O(1) SRAM complexity, ≈O(d/4) register complexity.
- [ ] 📚L2: level 2, O(2xBrx16)≈O(1) SRAM complexity, ≈O(1) register complexity + Q@K^T recomputation.
- [ ] 📚L3: level 3, O(2xBrx16)≈O(1) SRAM complexity, ≈O(1) register complexity + scaling O via HBM offloading.

By leveraging this approach, we can achieve better performance for large headdim (D > 256) through a balanced utilization of FlashAttention (which is not designed to support D > 256) and SDPA EA. Approximate SRAM and register complexity analysis for L1~L3 is as follows: (`d`=headdim, `C,Br,Bc`=Constant, `Br=Bc`) 👇

|📚Complexity| 📚FFPA L1 |  📚FFPA L2 |  📚FFPA L3 | 📚FA-2 |
|:---:|:---:|:---:|:---:|:---:|
|SRAM | O(2xBrx16)≈O(1) | O(2xBrx16)≈O(1) | O(2xBrx16)≈O(1) | ≈O(3xBrxd), d↑ |
|Register | ≈O(d/4), d↑ | O((Bc/16)x4+2C)≈O(1)|O((Bc/16)x4+2C)≈O(1)| ≈O(d/2), d↑ |
|HBM| ≈FA2≈O(Nd), O | ≈FA2≈O(Nd), O| ≈FA2≈O(Nd), O | ≈O(Nd), O |
|Extra HBM| ≈FA2≈O(N), m,l | ≈FA2≈O(N), m,l | ≈FA2≈O(N), m,l | ≈O(N), m,l |

## 📖 Prerequisites
<div id="prerequisites"></div>

- Python >= 3.10
- PyTorch >= 2.4.0, CUDA >= 12.4
- Recommended: PyTorch 2.5.1, CUDA 12.5

## 📖 Installation

<div id="install"></div>

The FFPA implemented in this repo can be install as a python library, namely, `ffpa-attn` library (optional).
```bash
git clone https://github.com/DefTruth/ffpa-attn-mma.git
# clone, then, run bash .dev/install.sh directly or run commands:
python3 setup.py bdist_wheel && cd dist && python3 -m pip install *.whl # pip uninstall ffpa-attn -y
```

## 📖 FFPA L1 (Level 1): Benchmark 🎉🎉

<div id="L1-bench"></div>

L1: level 1, O(2xBrx16)≈O(1) SRAM complexity, O(d/4) register complexity, the same GPU HBM memory complexity as FlashAttention. B=1, H=48, N=8192, **D=320-1024(FA2 not supported 👀)**. (Notes, `*`=MMA Acc F32, `^`=MMA Acc F16, Softmax Acc dtype is always be F32, T=TFLOPS, 👇Benchmark)

- 📚 NVIDIA L20 (`*`=MMA Acc F32, `^`=MMA Acc F16, `T`=TFLOPS, **~1.7x↑🎉**)

|Algorithm|320|384|448|512|576|640|704|768|832|896|960|1024|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|SDPA EA|56T|63T|58T|58T|55T|56T|54T|55T|54T|55T|54T|56T|
|FFPA L1*|99T|101T|102T|95T|95T|95T|94T|92T|92T|93T|93T|93T|
|Speedup|1.77x|1.6x|1.76x|1.64x|1.73x|1.7x|1.74x|1.67x|1.7x|1.69x|1.72x|1.66x|
|FFPA L1^|98T|100T|101T|102T|101T|93T|92T|93T|94T|93T|93T|93T|
|Speedup|1.75x|1.59x|1.74x|1.76x|1.84x|1.66x|1.7x|1.69x|1.74x|1.69x|1.72x|1.66x|

- 📚 NVIDIA A30 (`*`=MMA Acc F32, `^`=MMA Acc F16, `T`=TFLOPS, **~1.5x↑🎉**)

|Algorithm|320|384|448|512|576|640|704|768|832|896|960|1024|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|SDPA EA|25T|25T|24T|24T|24T|24T|23T|22T|22T|22T|22T|18T|
|FFPA L1*|36T|37T|35T|36T|35T|36T|35T|34T|34T|32T|34T|31T|
|Speedup|1.44x|1.48x|1.46x|1.5x|1.46x|1.5x|1.52x|1.55x|1.55x|1.45x|1.55x|1.72x|
|FFPA L1^|36T|36T|37T|37T|35T|35T|35T|35T|35T|34T|35T|32T|
|Speedup|1.44x|1.44x|1.54x|1.54x|1.46x|1.46x|1.52x|1.59x|1.59x|1.55x|1.59x|1.78x|

- 📚 NVIDIA RTX 3080 Laptop (`*`=MMA Acc F32, `^`=MMA Acc F16, `T`=TFLOPS, **~2.5x↑🎉**)

|Algorithm|320|384|448|512|576|640|704|768|832|896|960|1024|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|SDPA EA|13T|16T|12T|16T|15T|15T|15T|15T|14T|14T|14T|14T|
|FFPA L1*|30T|30T|30T|28T|29T|28T|27T|27T|25T|25T|24T|24T|
|Speedup|2.31x|1.88x|2.5x|1.75x|1.93x|1.87x|1.8x|1.8x|1.79x|1.79x|1.71x|1.71x|
|FFPA L1^|41T|40T|40T|39T|39T|37T|37T|35T|34T|32T|34T|32T|
|Speedup|3.15x|2.5x|3.33x|2.44x|2.6x|2.47x|2.47x|2.33x|2.43x|2.29x|2.43x|2.29x|

- 📚 NVIDIA RTX 4090 (`*`=MMA Acc F32, `^`=MMA Acc F16, `T`=TFLOPS, **~1.8x↑🎉**)

|Algorithm|320|384|448|512|576|640|704|768|832|896|960|1024|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|SDPA EA|82T|93T|85T|85T|79T|81T|79T|80T|79T|80T|78T|78T|
|FFPA L1*|145T|148T|143T|139T|140T|139T|138T|135T|134T|134T|132T|129T|
|Speedup|1.77x|1.59x|1.68x|1.64x|1.77x|1.72x|1.75x|1.69x|1.7x|1.68x|1.69x|1.65x|
|FFPA L1^|176T|174T|171T|174T|173T|170T|169T|167T|164T|163T|162T|159T|
|Speedup|2.15x|1.87x|2.01x|2.05x|2.19x|2.1x|2.14x|2.09x|2.08x|2.04x|2.08x|2.04x|

## 📖 Python Testing
<div id="python-test"></div>

👇 You can test many custom FFPA kernels via Python and figure out the difference in their performance.
```bash
# You can test on many devices, such as Volta, Ampere, Ada, Hopper, ...
cd tests && python3 test.py --B 1 --H 48 --N 8192 --show-all --D 320
```
- 📚 case: B=1, H=48, N=8192, D=320(`FA2 not supported`), Device=NVIDIA RTX 4090.
```bash
python3 test.py --B 1 --H 48 --N 8192 --show-all --D 320 # NVIDIA RTX 4090
-----------------------------B=1, H=48, N=8192, D=320, Warmup: 1, Iters: 5-----------------------
                   (sdpa): ['-0.01750183 '], time:50.36ms, TFLOPS:82.19 (+0.00 %)(~1.00x)
 (ffpa+acc+f32+L1+stage1): ['-0.01754761 '], time:40.23ms, TFLOPS:102.87(+25.17%)(~1.25x)
 (ffpa+acc+f32+L1+stage2): ['-0.01754761 '], time:30.35ms, TFLOPS:136.34(+32.54%)(~1.66x)
 (ffpa+acc+f16+L1+stage1): ['-0.01747131 '], time:31.03ms, TFLOPS:133.27(+0.00 %)(~1.62x)
 (ffpa+acc+f16+L1+stage2): ['-0.01747131 '], time:26.98ms, TFLOPS:153.41(+12.51%)(~1.87x)
-------------------------------------------------------------------------------------------------
```
- 📚 case: Generate benchmark table on Your own device (Welcome to PR your benchmark table 🎉🎉)
```bash
python3 test.py --gen-bench --show-all # NVIDIA RTX 4090
|Algorithm|320|384|448|512|576|640|704|768|832|896|960|1024|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|SDPA EA|80T|94T|86T|85T|79T|81T|79T|81T|79T|80T|79T|72T|
|FFPA L1*|135T|140T|143T|135T|134T|134T|134T|134T|131T|131T|130T|131T|
|Speedup|1.69x|1.49x|1.66x|1.59x|1.7x|1.65x|1.7x|1.65x|1.66x|1.64x|1.65x|1.82x|
|FFPA L1^|153T|155T|157T|157T|159T|157T|157T|156T|151T|151T|150T|153T|
|Speedup|1.91x|1.65x|1.83x|1.85x|2.01x|1.94x|1.99x|1.93x|1.91x|1.89x|1.9x|2.12x|
```


## ©️License

<div id="License"></div>

GNU General Public License v3.0

## 🎉Contribute

<div id="Contribute"></div>

How to contribute? Wecome to star⭐️ this repo to support me👆🏻 ~

<div align='center'>
<a href="https://star-history.com/#DefTruth/ffpa-attn-mma&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=DefTruth/ffpa-attn-mma&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=DefTruth/ffpa-attn-mma&type=Date" />
   <img img width=450 height=300 alt="Star History Chart" src="https://api.star-history.com/svg?repos=DefTruth/ffpa-attn-mma&type=Date" />
 </picture>
</a>
</div>

## 📖 References
<div id="ref"></div>

- [flash-attention](https://github.com/Dao-AILab/flash-attention)
- [CUDA-Learn-Notes](https://github.com/DefTruth/CUDA-Learn-Notes)
- [flashinfer](https://github.com/flashinfer-ai/flashinfer)

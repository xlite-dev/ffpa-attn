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

🤖[WIP] **FFPA**: Yet antother **Faster Flash Prefill Attention** with **O(1) SRAM complexity** & **O(d/4) or O(1) register complexity** for large headdim (D > 256), almost **1.5x~2x** 🎉 faster than SDPA EA with or without MMA Acc F32 on many devices: [📈L20 ~1.7x↑🎉](#L1-bench), [📈 A30 ~1.5x↑🎉](#L1-bench), [📈3080 ~2.5x↑🎉](#L1-bench), [📈4090 ~1.8x↑🎉](#L1-bench). 

<div align='center'>
  <img src='https://github.com/user-attachments/assets/7dc42fa1-a10e-453c-8e2c-befba6f12719' width="407px">
  <img src='https://github.com/user-attachments/assets/c0443e13-94a4-4d29-8f77-e326e62a668e' width="407px">
</div> 

💡NOTE: This project is still in its early dev stages and now provides some kernels and benchmarks for reference. More features will be added in the future. (Welcome to 🌟👆🏻star this repo to support me ~)

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
- [📈 FFPA L1: L20 ~1.7x↑🎉](#L1-bench-l20)
- [📈 FFPA L1: A30 ~1.5x↑🎉](#L1-bench-a30)
- [📈 FFPA L1: 3080 ~2.5x↑🎉](#L1-bench-3080)
- [📈 FFPA L1: 4090 ~1.8x↑🎉](#L1-bench-4090)

## 📖 FFPA L1~L3: FlashAttention + QKV Fine-grained Tiling at MMA level💡
<div id="ffpa-design"></div>

We have extended FlashAttention for large headdim (D > 256) by implementing **Fine-grained Tiling** at the **MMA level (GEMM style)** for the Q@K^T and P@V matmul. This approach results in a constant SRAM usage of Br * 16 or Bc * 16 (Br = Bc) for Q, K, and V, leading to an overall SRAM complexity of O(2 * Br * 16) ≈ O(1) and a register complexity of O(d/4) or O(1). Consequently, this method allows us to extend **headdim > 256** and achieve faster performance compared to SDPA with or without MMA Accumulation F32 (**1.5x~2x** 🎉 faster than SDPA EA).

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

**📚👇Core Features🎉🎉**: I have implemented **FFPA L1~L3** using pure MMA PTX instructions, which supports many features such as Split-Q, SMEM Swizzle/Padding, QKV Multi-Stages(1~4), Tile MMAs/Warps, Mixed MMA F32/F16 Acc (Q@K^T MMA Acc F32 + P@V MMA Acc F16), Fully Shared QKV SMEM, Prefetch QKV g2s, **Fully QKV Fine-grained Tiling(GEMM style)**, Collective Store, etc.

|📚Feature |📚Feature |📚Feature |📚Feature|
|:---:|:---:|:---:|:---:|
|✔️Tensor Cores|✔️Loop over N/D |✔️Tile Block(Br, Bc) |✔️MMA(m16n8k16)|
|✔️**Split Q**(FA-2)|✔️Pack LDST(128 bits)|✔️SMEM **Swizzle**/Pad |✔️Copy Async |
|✔️Tile MMA/Warp |✔️QKV Multi-Stages(1~4) |✔️Collective Store(Shfl)|✔️**Prefetch QKV** g2s |
|✔️**QKV Fine-grained Tiling**|✔️**Shared QKV** SMEM|✔️Mixed MMA Acc|✔️**FFPA L1 Level**|

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

<div id="L1-bench-l20"></div>

L1: level 1, O(2xBrx16)≈O(1) SRAM complexity, O(d/4) register complexity, the same GPU HBM memory complexity as FlashAttention. B=1, H=48, N=8192, **D=320-1024(FA2 not supported 👀)**. (Notes, `*`=MMA Acc F32, `^`=MMA Acc F16, Softmax Acc dtype is always be F32, T=TFLOPS, 👇Benchmark)

- 📚 NVIDIA L20 (`*`=MMA Acc F32, `^`=MMA Acc F16, `T`=TFLOPS, **~1.8x↑🎉**)

|Algorithm|320|384|448|512|576|640|704|768|832|896|960|1024|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|SDPA EA|56T|63T|58T|58T|55T|56T|54T|55T|54T|55T|54T|56T|
|FFPA L1*|103T|104T|103T|95T|95T|94T|95T|95T|93T|93T|93T|92T|
|Speedup|1.84x|1.65x|1.78x|1.64x|1.73x|1.68x|1.76x|1.73x|1.72x|1.69x|1.72x|1.64x|
|FFPA L1^|104T|105T|105T|104T|102T|95T|94T|93T|93T|94T|92T|93T|
|Speedup|1.86x|1.67x|1.81x|1.79x|1.85x|1.7x|1.74x|1.69x|1.72x|1.71x|1.7x|1.66x|

- 📚 NVIDIA L20 (`*`=MMA Acc: QK F32 + PV F16, `^`=MMA Acc F16, `T`=TFLOPS, **~1.9x↑🎉**)

|Algorithm|320|384|448|512|576|640|704|768|832|896|960|1024|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|SDPA EA|56T|64T|58T|58T|55T|56T|54T|55T|54T|55T|54T|56T|
|FFPA L1*|105T|106T|105T|96T|95T|94T|93T|93T|92T|92T|93T|93T|
|Speedup|1.88x|1.66x|1.81x|1.66x|1.73x|1.68x|1.72x|1.69x|1.7x|1.67x|1.72x|1.66x|
|FFPA L1^|104T|105T|105T|104T|102T|94T|94T|93T|93T|94T|92T|93T|
|Speedup|1.86x|1.64x|1.81x|1.79x|1.85x|1.68x|1.74x|1.69x|1.72x|1.71x|1.7x|1.66x|

<div align='center'>
  <img src='https://github.com/user-attachments/assets/7881c7fc-aeb4-4556-92a0-901b5b25ee1b' width="407px">
  <img src='https://github.com/user-attachments/assets/f530900d-0dff-4986-a7e7-47a47ba15556' width="407px">
</div> 

<div id="L1-bench-a30"></div>

- 📚 NVIDIA A30 (`*`=MMA Acc F32, `^`=MMA Acc F16, `T`=TFLOPS, **~1.5x↑🎉**)

|Algorithm|320|384|448|512|576|640|704|768|832|896|960|1024|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|SDPA EA|25T|25T|24T|24T|24T|24T|23T|22T|22T|22T|22T|18T|
|FFPA L1*|36T|37T|35T|36T|35T|36T|35T|34T|34T|32T|34T|31T|
|Speedup|1.44x|1.48x|1.46x|1.5x|1.46x|1.5x|1.52x|1.55x|1.55x|1.45x|1.55x|1.72x|
|FFPA L1^|36T|36T|37T|37T|35T|35T|35T|35T|35T|34T|35T|32T|
|Speedup|1.64x|1.68x|1.71x|1.67x|1.54x|1.54x|1.57x|1.64x|1.59x|1.59x|1.45x|1.72x|

- 📚 NVIDIA A30 (`*`=MMA Acc: QK F32 + PV F16, `^`=MMA Acc F16, `T`=TFLOPS, **~1.7x↑🎉**)

|Algorithm|320|384|448|512|576|640|704|768|832|896|960|1024|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|SDPA EA|25T|25T|24T|24T|24T|24T|23T|22T|22T|22T|22T|18T|
|FFPA L1*|42T|41T|39T|38T|37T|36T|36T|35T|34T|33T|31T|30T|
|Speedup|1.68x|1.64x|1.62x|1.58x|1.54x|1.5x|1.57x|1.59x|1.55x|1.5x|1.41x|1.67x|
|FFPA L1^|41T|42T|41T|40T|37T|37T|36T|36T|35T|35T|32T|31T|
|Speedup|1.64x|1.68x|1.71x|1.67x|1.54x|1.54x|1.57x|1.64x|1.59x|1.59x|1.45x|1.72x|

<div align='center'>
  <img src='https://github.com/user-attachments/assets/7437341e-207d-4e35-b13f-b5834957591f' width="407px">
  <img src='https://github.com/user-attachments/assets/014df0f8-8283-4270-812e-a43bdf10366f' width="407px">
</div> 

<div id="L1-bench-3080"></div>

- 📚 NVIDIA RTX 3080 Laptop (`*`=MMA Acc F32, `^`=MMA Acc F16, `T`=TFLOPS, **~2.5x↑🎉**)

|Algorithm|320|384|448|512|576|640|704|768|832|896|960|1024|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|SDPA EA|13T|16T|12T|16T|15T|15T|14T|14T|14T|14T|14T|14T|
|FFPA L1*|33T|31T|31T|28T|28T|27T|26T|25T|25T|24T|23T|23T|
|Speedup|2.54x|1.94x|2.58x|1.75x|1.87x|1.8x|1.86x|1.79x|1.79x|1.71x|1.64x|1.64x|
|FFPA L1^|41T|40T|39T|38T|37T|36T|36T|34T|33T|32T|30T|30T|
|Speedup|3.15x|2.5x|3.25x|2.38x|2.47x|2.4x|2.57x|2.43x|2.36x|2.29x|2.14x|2.14x|

- 📚 NVIDIA RTX 3080 Laptop (`*`=MMA Acc: QK F32 + PV F16, `^`=MMA Acc F16, `T`=TFLOPS, **~2.8x↑🎉**)

|Algorithm|320|384|448|512|576|640|704|768|832|896|960|1024|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|SDPA EA|13T|16T|12T|16T|15T|15T|15T|15T|14T|14T|14T|14T|
|FFPA L1*|37T|35T|35T|33T|31T|30T|30T|29T|26T|28T|26T|25T|
|Speedup|2.85x|2.19x|2.92x|2.06x|2.07x|2.0x|2.0x|1.93x|1.86x|2.0x|1.86x|1.79x|
|FFPA L1^|41T|41T|40T|39T|38T|37T|36T|35T|32T|31T|30T|31T|
|Speedup|3.15x|2.56x|3.33x|2.44x|2.53x|2.47x|2.4x|2.33x|2.29x|2.21x|2.14x|2.21x|

<div align='center'>
  <img src='https://github.com/user-attachments/assets/7dc42fa1-a10e-453c-8e2c-befba6f12719' width="407px">
  <img src='https://github.com/user-attachments/assets/c0443e13-94a4-4d29-8f77-e326e62a668e' width="407px">
</div> 

<div id="L1-bench-4090"></div>

- 📚 NVIDIA RTX 4090 (`*`=MMA Acc F32, `^`=MMA Acc F16, `T`=TFLOPS, **~1.8x↑🎉**)

|Algorithm|320|384|448|512|576|640|704|768|832|896|960|1024|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|SDPA EA|82T|93T|85T|85T|79T|81T|79T|80T|79T|80T|78T|78T|
|FFPA L1*|145T|148T|143T|139T|140T|139T|138T|135T|134T|134T|132T|129T|
|Speedup|1.77x|1.59x|1.68x|1.64x|1.77x|1.72x|1.75x|1.69x|1.7x|1.68x|1.69x|1.65x|
|FFPA L1^|176T|174T|171T|174T|173T|170T|169T|167T|164T|163T|162T|159T|
|Speedup|2.15x|1.87x|2.01x|2.05x|2.19x|2.1x|2.14x|2.09x|2.08x|2.04x|2.08x|2.04x|

- 📚 NVIDIA RTX 4090 (`*`=MMA Acc: QK F32 + PV F16, `^`=MMA Acc F16, `T`=TFLOPS, **~2x↑🎉**)

|Algorithm|320|384|448|512|576|640|704|768|832|896|960|1024|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|SDPA EA|82T|93T|85T|85T|79T|81T|79T|80T|79T|80T|78T|78T|
|FFPA L1*|170T|170T|160T|159T|156T|154T|153T|152T|151T|149T|143T|136T|
|Speedup|2.07x|1.83x|1.88x|1.87x|1.97x|1.9x|1.94x|1.9x|1.91x|1.86x|1.83x|1.74x|
|FFPA L1^|175T|173T|171T|174T|171T|170T|169T|167T|164T|164T|162T|157T|
|Speedup|2.13x|1.86x|2.01x|2.05x|2.16x|2.1x|2.14x|2.09x|2.08x|2.05x|2.08x|2.01x|

<div align='center'>
  <img src='https://github.com/user-attachments/assets/5699465b-03b8-460c-8d9e-7b84bad25d85' width="407px">
  <img src='https://github.com/user-attachments/assets/083a3c6c-1afb-4fc5-9622-34ca22129627' width="407px">
</div> 

## 📖 Python Testing
<div id="python-test"></div>

👇 You can test many custom FFPA kernels via Python and figure out the difference in their performance.
```bash
# You can test on many devices, such as Volta, Ampere, Ada, Hopper, ...
cd tests && python3 test.py --B 1 --H 48 --N 8192 --show-all --D 320
```
- 📚 case: B=1, H=48, N=8192, D=320(`FA2 not supported`)
```bash
python3 test.py --B 1 --H 48 --N 8192 --show-all --D 320
```
- 📚 case: Generate benchmark table on Your own device (Welcome to PR your benchmark table 🎉🎉)
```bash
python3 test.py --gen-bench --show-all
```
- 📚 case: Generate benchmark plots on Your own device (Welcome to PR your benchmark plots 🎉🎉)
```bash
python3 test.py --gen-bench --show-all --plot
```

💡NOTE: Please check all configurable environment variables in [env.py](./env.py).

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

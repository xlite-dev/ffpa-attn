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
 
🤖 [WIP] **FFPA**: Yet antother **Faster Flash Prefill Attention** with **O(1) SRAM complexity** & **O(d/4) or O(1) register complexity** for large headdim (D > 256), almost **>1.5x** 🎉 faster than SDPA EA with or without MMA Accumulation F32 on many devices, such as NVIDIA L20, 4090, 3080 Laptop (Experimental 👀~). The FFPA kernels are modified from my repo 📖[CUDA-Learn-Notes](https://github.com/DefTruth/CUDA-Learn-Notes/tree/main/kernels/flash-attn)  ![](https://img.shields.io/github/stars/DefTruth/CUDA-Learn-Notes.svg?style=social).

<!--
|Tensor Cores|Loop over N/D |Tile Block (Br, Bc) |MMA (m16n8k16)|
|:---:|:---:|:---:|:---:|
|✔️|✔️|✔️|✔️|
|Pack LDST (128 bits)|SMEM **Swizzle**/Padding |Copy Async|Tile MMA (More Threads) |
|✔️|✔️|✔️|✔️|
|Tile Warp (More Values) |Multi Stages (1/2) |Collective Store (**Shfl**)|**Split Q**|
|✔️|✔️|✔️|✔️|
|**QKV Fine-grained Tiling**|**Shared QKV** SMEM|**FFPA L1 Level**|**FFPA L2/L3 Level** |
|✔️|✔️|✔️|?|
-->

NOTE: This project is still in its early development stages and currently provides a few experimental kernels and benchmarks for reference. More benchmarks data and features (FFPA **L2/L3** & more devices) will be added over time as the project continues to develop. 

## ©️Citations🎉🎉

```BibTeX
@misc{faster-prefill-attention@2025,
  title={FFPA: Yet another Faster Flash Prefill Attention with O(1) SRAM complexity for large headdim.},
  url={https://github.com/DefTruth/faster-prefill-attention},
  note={Open-source software available at https://github.com/DefTruth/faster-prefill-attention},
  author={DefTruth etc},
  year={2025}
}
```

## 📖 Contents

- [📖 Prerequisites](#prerequisites)
- [📖 Installation](#install)
- [📖 FFPA L1~L3 Design](#ffpa-design)
- [📖 FFPA L1 Benchmark](#L1-bench)
- [📖 FFPA L2 Benchmark](#L1-bench)
- [📖 FFPA L3 Benchmark](#L1-bench)
- [📖 Python Testing](#python-test)
- [📖 References](#ref)

## 📖 FFPA L1~L3: FlashAttention + QKV Fine-grained Tiling at MMA level 🔑️
<div id="ffpa-design"></div>  

We have extended FlashAttention for large headdim (D > 256) by implementing **Fine-grained Tiling** at the **MMA level (GEMM style)** for the Q@K^T and P@V matmul. This approach results in a constant SRAM usage of Br * 16 or Bc * 16 for Q, K, and V, leading to an overall SRAM complexity of O(Br * 16) ≈ O(1) and a register complexity of O(d/4) or O(1). Consequently, this method allows us to extend **headdim > 256** and achieve faster performance compared to SDPA with or without MMA Accumulation F32 (almost **>1.5x** 🎉 faster than SDPA EA). 

We have named this new attention tiling technique **FFPA: Faster Flash Prefill Attention**. We have designed three levels of FFPA based on SRAM and register complexity considerations. All levels will not introduce any additional VRAM requirements, ensuring that the GPU HBM memory complexity remains consistent with FlashAttention. (`d`=headdim,`C`=Constant)

- [x] 📚L1: level 1, O(Brx16)~O(1) SRAM complexity, O(d/4) register complexity.  
- [ ] 📚L2: level 2, O(Brx16)~O(1) SRAM complexity, O(1) register complexity + Q@K^T recomputation.  
- [ ] 📚L3: level 3, O(Brx16)~O(1) SRAM complexity, O(1) register complexity + scaling O via HBM offloading. 

|📚Complexity| 📚FFPA L1 |  📚FFPA L2 |  📚FFPA L3 | 📚FlashAttention | 
|:---:|:---:|:---:|:---:|:---:| 
|SRAM | O(Brx16)~O(1) | O(Brx16)~O(1) | O(Brx16)~O(1) | ~O(3xBrxd), QKV |
|Register | O(d/4) | O((Bc/16)x4+Cx2)~O(1)|O((Bc/16)x4+Cx2)~O(1)| ~O(d/2), SO |

By leveraging this approach, we can achieve improved performance for large headdim (D > 256) through a balanced utilization of FlashAttention (which is not designed to support D > 256) and SDPA EA. This allows us to take advantage of the strengths of both methods while mitigating their limitations. 

## 📖 Prerequisites
<div id="prerequisites"></div>  

- PyTorch >= 2.4.0, CUDA >= 12.4
- Recommended: PyTorch 2.5.1, CUDA 12.5

## 📖 Installation  

<div id="install"></div>  

The FFPA implemented in this repo can be install as a python library, namely, `ffpa-attn` library (optional). 
```bash
git clone https://github.com/DefTruth/faster-prefill-attention.git
# Then, run tests/install.sh directly or run commands as belows:
python3 setup.py bdist_wheel && rm -rf *.egg-info # build ffpa-attn from sources
cd dist && python3 -m pip install ffpa_attn-*-linux_x86_64.whl # pip uninstall ffpa-attn -y 
```

## 📖 FFPA L1 (Level 1): Benchmark 🎉🎉

<div id="L1-bench"></div>  

L1: level 1, O(Brx16)~O(1) SRAM complexity, O(d/4) register complexity, the same GPU HBM memory complexity as FlashAttention. B=1, H=48, N=8192, **D=320-1024(FA2 not supported 👀)**. (Notes, `*`=MMA Acc F32, `^`=MMA Acc F16, Softmax Acc dtype is always be F32, T=TFLOPS, 👇Benchmark)

- 📚 NVIDIA RTX 3080 Laptop (`*`=MMA Acc F32, `^`=MMA Acc F16, `T`=TFLOPS)

|Algorithm|320|384|448|512|576|640|704|768|832|896|960|1024|    
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|  
|SDPA EA|13T|16T|12T|16T|15T|15T|15T|15T|15T|15T|15T|15T|  
|FFPA L1*|32T|30T|30T|28T|28T|27T|26T|25T|25T|25T|25T|24T|   
|Speedup|2.48x|1.88x|2.55x|1.75x|1.90x|1.77x|1.73x|1.67x|1.66x|1.66x|1.66x|1.54x|  
|FFPA L1^|40T|38T|39T|36T|35T|34T|33T|32T|31T|31T|28T|27T|  
|Speedup|3.07x|2.42x|3.33x|2.24x|2.35x|2.19x|2.19x|2.13x|2.03x|2.03x|1.90x|1.74x|

- 📚 NVIDIA RTX 4090 (`*`=MMA Acc F32, `^`=MMA Acc F16, `T`=TFLOPS)

|Algorithm|320|384|448|512|576|640|704|768|832|896|960|1024|    
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|  
|SDPA EA|82T|92T|83T|84T|78T|80T|78T|80T|78T|80T|78T|79T|
|FFPA L1*|136T|135T|135T|132T|133T|133T|132T|131T|130T|125T|123T|93T|
|Speedup|1.64x|1.45x|1.61x|1.57x|1.71x|1.65x|1.68x|1.62x|1.65x|1.56x|1.55x|1.17x| 
|FFPA L1^|154T|161T|160T|157T|156T|155T|157T|154T|149T|150T|145T|100T|
|Speedup|1.85x|1.73x|1.92x|1.87x|1.99x|1.93x|1.99x|1.90x|1.90x|1.88x|1.84x|1.25x|

- 📚 NVIDIA L20 (`*`=MMA Acc F32, `^`=MMA Acc F16, `T`=TFLOPS)

|Algorithm|320|384|448|512|576|640|704|768|832|896|960|1024|    
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|  
|SDPA EA|56T|63T|57T|58T|55T|56T|54T|55T|54T|55T|54T|56T|
|FFPA L1*|99T|95T|95T|93T|94T|92T|92T|90T|89T|90T|90T|89T|
|Speedup|1.77x|1.49x|1.64x|1.58x|1.72x|1.65x|1.68x|1.63x|1.64x|1.63x|1.67x|1.58x| 
|FFPA L1^|96T|99T|100T|92T|93T|92T|93T|91T|90T|90T|88T|91T|
|Speedup|1.71x|1.55x|1.73x|1.56x|1.69x|1.65x|1.71x|1.64x|1.65x|1.63x|1.62x|1.62x|

- 📚 NVIDIA A30 (`*`=MMA Acc F32, `^`=MMA Acc F16, `T`=TFLOPS)

|Algorithm|320|384|448|512|576|640|704|768|832|896|960|1024|    
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|  
|SDPA EA|25T|25T|24T|23T|24T|24T|23T|22T|22T|21T|21T|18T|
|FFPA L1*|33T|33T|32T|31T|32T|32T|30T|28T|25T|24T|24T|24T|
|Speedup|1.33x|1.33x|1.30x|1.31x|1.33x|1.33x|1.32x|1.23x|1.15x|1.11x|1.11x|1.27x|
|FFPA L1^|33T|33T|33T|30T|31T|32T|31T|30T|30T|27T|24T|23T|
|Speedup|1.33x|1.33x|1.36x|1.30x|1.31x|1.33x|1.37x|1.35x|1.35x|1.25x|1.11x|1.25x|

## 📖 Python Testing 
<div id="python-test"></div>  

👇 You can test many custom FFPA kernels via Python and figure out the difference in their performance.
```bash
# You can test Ada or Ampere only, also, Volta, Ampere, Ada, Hopper, ...
export TORCH_CUDA_ARCH_LIST=Ada # for Ada only
export TORCH_CUDA_ARCH_LIST=Ampere # for Ampere only
python3 tests/test.py --B 1 --H 48 --N 8192 --show-all --D 320 
```
- 📚 case: B=1, H=48, N=8192, D=320(FA2 not supported), Device=NVIDIA RTX 4090.
```bash
python3 tests/test.py --B 1 --H 48 --N 8192 --show-all --D 320
-------------------------------------------------------------------------------------------------
-----------------------------B=1, H=48, N=8192, D=320, Warmup: 1, Iters: 5-----------------------
                   (sdpa): ['-0.01750183 '], time:50.36ms, TFLOPS:82.19 (+0.00 %)(~1.00x)
 (ffpa+acc+f32+L1+stage1): ['-0.01754761 '], time:40.23ms, TFLOPS:102.87(+25.17%)(~1.25x)
 (ffpa+acc+f32+L1+stage2): ['-0.01754761 '], time:30.35ms, TFLOPS:136.34(+32.54%)(~1.66x)
 (ffpa+acc+f16+L1+stage1): ['-0.01747131 '], time:31.03ms, TFLOPS:133.27(+0.00 %)(~1.62x)
 (ffpa+acc+f16+L1+stage2): ['-0.01747131 '], time:26.98ms, TFLOPS:153.41(+12.51%)(~1.87x)
-------------------------------------------------------------------------------------------------
```

## ©️License

<div id="License"></div>  

GNU General Public License v3.0

## 🎉Contribute 

<div id="Contribute"></div>  

How to contribute? Wecome to star⭐️ this repo to support me👆🏻 ~

<div align='center'>
<a href="https://star-history.com/#DefTruth/faster-prefill-attention&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=DefTruth/faster-prefill-attention&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=DefTruth/faster-prefill-attention&type=Date" />
   <img img width=450 height=300 alt="Star History Chart" src="https://api.star-history.com/svg?repos=DefTruth/faster-prefill-attention&type=Date" />
 </picture>
</a>
</div>

## 📖 References   
<div id="ref"></div>  

- [flash-attention](https://github.com/Dao-AILab/flash-attention)
- [CUDA-Learn-Notes](https://github.com/DefTruth/CUDA-Learn-Notes)
- [cutlass](https://github.com/NVIDIA/cutlass)

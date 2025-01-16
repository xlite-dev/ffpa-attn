

<div align='center'>
  <img src=https://github.com/user-attachments/assets/1312aba0-4707-4fcb-84dc-59c57347b23f width=250 >
</div>

<div align='center'>
  <img src=https://img.shields.io/badge/Language-CUDA/Python-brightgreen.svg >
  <img src=https://img.shields.io/github/watchers/DefTruth/faster-prefill-attention?color=9cc >
  <img src=https://img.shields.io/github/forks/DefTruth/faster-prefill-attention.svg?style=social >
  <img src=https://img.shields.io/github/stars/DefTruth/faster-prefill-attention.svg?style=social >
  <img src=https://img.shields.io/badge/Release-v0.0.1-brightgreen.svg >
  <img src=https://img.shields.io/badge/License-GPLv3.0-turquoise.svg >
 </div>

🤖[WIP] **FFPA**: Yet antother **Faster Flash Prefill Attention** with **O(1) SRAM complexity** & **O(d/4) or O(1) register complexity** for large headdim (D > 256), almost **1.8x~3x** 🎉 faster than SDPA EA with or without MMA Acc F32 on many devices: [📈L20 ~1.9x↑🎉](#L1-bench-l20), [📈 A30 ~1.8x↑🎉](#L1-bench-a30), [📈3080 ~2.9x↑🎉](#L1-bench-3080), [📈4090 ~2.1x↑🎉](#L1-bench-4090). 


<!--
<div align='left'>
  <img src='https://github.com/user-attachments/assets/447e2937-f7c8-47c8-8550-8c0c71b910e6' width="411px">
  <img src='https://github.com/user-attachments/assets/65a8d564-8fa7-4d66-86b9-e238feb86143' width="411px">
</div> 
<div align='left'>
  <img src='https://github.com/user-attachments/assets/cba2edce-ac0d-412e-823c-7eea2cc63f83' height="170px" width="270px">
  <img src='https://github.com/user-attachments/assets/447e2937-f7c8-47c8-8550-8c0c71b910e6' height="170px" width="270px">
  <img src='https://github.com/user-attachments/assets/65a8d564-8fa7-4d66-86b9-e238feb86143' height="170px" width="270px">
</div> 
<div align='center'>
  <img src=https://github.com/user-attachments/assets/9f764ccf-3dce-43c2-b2ae-aa068231dea2 >
</div>
-->


💡NOTE: This project is still in its early dev stages and now provides some kernels and benchmarks for reference. More features will be added in the future. (Welcome to 🌟👆🏻star this repo to support me ~)

## ©️Citations🎉🎉

```BibTeX
@misc{ffpa-attn-mma@2025,
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
- [📈 FFPA L1: L20 ~1.9x↑🎉](#L1-bench-l20)
- [📈 FFPA L1: A30 ~1.8x↑🎉](#L1-bench-a30)
- [📈 FFPA L1: 3080 ~2.9x↑🎉](#L1-bench-3080)
- [📈 FFPA L1: 4090 ~2.1x↑🎉](#L1-bench-4090)

## 📖 FFPA L1~L3: FlashAttention + QKV Fine-grained Tiling at MMA level💡
<div id="ffpa-design"></div>

We have extended FlashAttention for large headdim (D > 256) by implementing **Fine-grained Tiling** at the **MMA level (GEMM style)** for the Q@K^T and P@V matmul. This approach results in a constant SRAM usage of Br * 16 or Bc * 16 (Br = Bc) for Q, K, and V, leading to an overall SRAM complexity of O(2 * Br * 16) ≈ O(1) and a register complexity of O(d/4) or O(1). Consequently, this method allows us to extend headdim beyond 256 and achieve faster performance compared to SDPA with or without MMA Accumulation F32 (**1.8x~3x** 🎉 faster than SDPA EA).

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
|✔️Tensor Cores|✔️Loop over N/D |✔️Tile Block(Br, Bc) |✔️**MMA(m16n8k16)**|
|✔️**Split Q**(FA-2)|✔️Pack LDST(128 bits)|✔️SMEM **Swizzle/Pad** |✔️Copy Async |
|✔️Tile MMA/Warp |✔️QKV Multi-Stages(1~4) |✔️Collective Store(**Shfl**)|✔️**Prefetch QKV** g2s |
|✔️**QKV Fine-grained Tiling**|✔️**Shared QKV** SMEM|✔️Mixed MMA Acc|✔️**FFPA L1 Level**|

- 📚 case: FFPA `L1` kernel template signature: [ffpa_attn_templates_L1.cuh](csrc/cuffpa/ffpa_attn_templates_L1.cuh)

```CUDA
template<
  const int kHeadDim,              // Headdim, 32~1024     
  const int kMmaAtomM,             // MMA Atom M, 16
  const int kMmaAtomN,             // MMA Atom N, 8
  const int kMmaAtomK,             // MMA Atom K, 16
  const int kMmaTileSeqLenQ,       // 4, more MMA(warp), M=16*4=64, Q@K^T=[Br(M), d(K)]@[d(K),  Bc(N)]  
  const int kMmaTileSeqLenK,       // 1, more MMA(warp), N=8*1 =8,  Q@K^T=[Br(M), d(K)]@[d(K),  Bc(N)]    
  const int kMmaTileSeqLenP,       // 4, more MMA(warp), M=16*4=64, P@V  =[Br(M),Bc(K)]@[Bc(K), d(N) ]
  const int kMmaTileHeadDimV,      // 1, more MMA(warp), N=8*1 =8,  P@V  =[Br(M),Bc(K)]@[Bc(K), d(N) ]       
  const int kWarpTileSeqLenQ,      // 1, more values, M, Br=64*1=64, matmul M 
  const int kWarpTileSeqLenK,      // 8, more values, N, Bc=8*8 =64, matmul N
  const int kWarpTileSeqLenP,      // 1, more values, M, Br=64*1=64, matmul M
  const int kWarpTileHeadDimV,     // 8, more values, N, d=8*(1|2|3|4|...)=8|...|32|64|96|128|...
  const int kMmaAccFloat32QK,      // 0/1, Q@K^T, 0 MMA Acc with fp16, 1 MMA Acc with fp32.
  const int kMmaAccFloat32PV,      // 0/1, P@V, 0 MMA Acc with fp16, 1 MMA Acc with fp32.
  const int kOStorageAccFloat32,   // 0/1, MMA Acc always be f32/f16, but O storage can be fp32 or half.
  const int kPrefetchQK,           // Prefetch QK at the Appropriate Time Point. 
  const int kPrefetchPV,           // Prefetch V at the Appropriate Time Point. 
  const int kShareSmemQKV,         // QKV share the same shared memory, reuse QK smem for V.
  const int kPersistQs2r,          // Persist load Q s2r for headdim < 512, but still keep O(1) SRAM.
  const int kStageQK,              // <= 4, may apply different multi stages policy for QK and V (<=4)
  const int kStagePV,              // <= 4, may apply different multi stages policy for QK and V (<=4)
  const int kPadQ,                 // Pad Q/K/V 0,8; 0 -> smem swizzle, > 0 -> padding
  const int kPadK,                 // Pad Q/K/V 0,8; 0 -> smem swizzle, > 0 -> padding
  const int kPadV                  // Pad Q/K/V 0,8; 0 -> smem swizzle, > 0 -> padding
> __global__ void // Q, K, V, O -> [B, H, N, D]
ffpa_mma_stages_split_q_L1_template(half* Q, half* K, half* V, half* O, ...);
```

## 📖 Prerequisites
<div id="prerequisites"></div>

- Python >= 3.10
- PyTorch >= 2.4.0, CUDA >= 12.4
- Recommended: PyTorch 2.5.1, CUDA 12.5
- Docker: nvcr.io/nvidia/pytorch:24.10-py3

## 📖 Installation

<div id="install"></div>

The FFPA implemented in this repo can be install as a python library, namely, `ffpa-attn` library (optional).
```bash
git clone https://github.com/DefTruth/ffpa-attn-mma.git
# clone, then, run bash .dev/install.sh directly or run commands:
python3 setup.py bdist_wheel && cd dist && python3 -m pip install *.whl # pip uninstall ffpa-attn -y
```

## 📖 FFPA L1 (Level 1): Benchmark 🎉🎉

<!--
![NVIDIA_A30](https://github.com/user-attachments/assets/69be99e4-977f-4a8c-bef5-9d6667241e23)
![NVIDIA_A30_ffpa+acc+f16+L1_Speedup](https://github.com/user-attachments/assets/7e323005-4445-41af-8e94-6efb62ed2b77)
![NVIDIA_A30_ffpa+acc+f32+L1_Speedup](https://github.com/user-attachments/assets/e314649e-82b5-414d-85c9-8b6fbf260138)
![NVIDIA_GeForce_RTX_3080_Laptop_GPU_WSL2](https://github.com/user-attachments/assets/be071842-25a7-4477-acc8-14d6e2ff5a54)
![NVIDIA_GeForce_RTX_3080_Laptop_GPU_WSL2_ffpa+acc+f16+L1_Speedup](https://github.com/user-attachments/assets/d157cd69-4444-4735-a691-edaaff408137)
![NVIDIA_GeForce_RTX_3080_Laptop_GPU_WSL2_ffpa+acc+f32+L1_Speedup](https://github.com/user-attachments/assets/3ce47627-e79d-40ee-b753-bdd235603b7d)
![NVIDIA_GeForce_RTX_4090](https://github.com/user-attachments/assets/cba2edce-ac0d-412e-823c-7eea2cc63f83)
![NVIDIA_GeForce_RTX_4090_ffpa+acc+f16+L1_Speedup](https://github.com/user-attachments/assets/447e2937-f7c8-47c8-8550-8c0c71b910e6)
![NVIDIA_GeForce_RTX_4090_ffpa+acc+f32+L1_Speedup](https://github.com/user-attachments/assets/65a8d564-8fa7-4d66-86b9-e238feb86143)
![NVIDIA_L20](https://github.com/user-attachments/assets/6be1708c-9491-4dc8-92cc-a3d48a335784)
![NVIDIA_L20_ffpa+acc+f16+L1_Speedup](https://github.com/user-attachments/assets/a4927108-3f97-4209-9b80-bb31ad271e04)
![NVIDIA_L20_ffpa+acc+f32+L1_Speedup](https://github.com/user-attachments/assets/eeb9943f-919d-45d8-a8a6-e0f8874f4bcd)
-->

<div id="L1-bench-l20"></div>

L1: level 1, O(2xBrx16)≈O(1) SRAM complexity, O(d/4) register complexity, the same GPU HBM memory complexity as FlashAttention. B=1, H=48, N=8192, **D=320-1024(FA2 not supported 👀)**. (Notes, `*`=MMA Acc F32, `^`=MMA Acc F16, Softmax Acc dtype is always be F32, T=TFLOPS, 👇Benchmark)

- 📚 NVIDIA L20 (`*`=MMA Acc F32, `^`=MMA Acc F16, `T`=TFLOPS, **~1.8x↑🎉**)

|Algorithm|320|384|448|512|576|640|704|768|832|896|960|1024|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|SDPA EA|56T|63T|58T|58T|55T|56T|54T|55T|54T|55T|54T|56T|
|FFPA L1*|102T|102T|103T|104T|103T|95T|95T|95T|95T|96T|95T|94T|
|Speedup|1.82x|1.62x|1.78x|1.79x|1.87x|1.7x|1.76x|1.73x|1.76x|1.75x|1.76x|1.68x|
|FFPA L1^|104T|103T|103T|102T|104T|103T|102T|94T|94T|94T|100T|100T|
|Speedup|1.86x|1.63x|1.78x|1.76x|1.89x|1.84x|1.89x|1.71x|1.74x|1.71x|1.85x|1.79x|

- 📚 NVIDIA L20 (`*`=MMA Acc: QK F32 + PV F16, `^`=MMA Acc F16, `T`=TFLOPS, **~1.9x↑🎉**)

|Algorithm|320|384|448|512|576|640|704|768|832|896|960|1024|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|SDPA EA|56T|64T|58T|58T|55T|56T|54T|55T|54T|55T|54T|56T|
|FFPA L1*|105T|102T|104T|103T|105T|95T|95T|94T|94T|94T|102T|101T|
|Speedup|1.88x|1.59x|1.79x|1.78x|1.91x|1.7x|1.76x|1.71x|1.74x|1.71x|1.89x|1.8x|
|FFPA L1^|104T|103T|103T|102T|103T|103T|102T|94T|94T|94T|100T|100T|
|Speedup|1.86x|1.61x|1.78x|1.76x|1.87x|1.84x|1.89x|1.71x|1.74x|1.71x|1.85x|1.79x|

<div align='left'>
  <img src='https://github.com/user-attachments/assets/a4927108-3f97-4209-9b80-bb31ad271e04' width="411px">
  <img src='https://github.com/user-attachments/assets/eeb9943f-919d-45d8-a8a6-e0f8874f4bcd' width="411px">
</div> 

<div id="L1-bench-a30"></div>

- 📚 NVIDIA A30 (`*`=MMA Acc F32, `^`=MMA Acc F16, `T`=TFLOPS, **~1.8x↑🎉**)

|Algorithm|320|384|448|512|576|640|704|768|832|896|960|1024|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|SDPA EA|25T|25T|24T|24T|24T|24T|23T|22T|22T|22T|22T|18T|
|FFPA L1*|45T|44T|44T|43T|43T|38T|37T|37T|37T|36T|33T|32T|
|Speedup|1.8x|1.76x|1.83x|1.79x|1.79x|1.58x|1.61x|1.68x|1.68x|1.64x|1.5x|1.78x|
|FFPA L1^|48T|46T|45T|43T|44T|44T|44T|38T|37T|36T|40T|34T|
|Speedup|1.92x|1.84x|1.88x|1.79x|1.83x|1.83x|1.91x|1.73x|1.68x|1.64x|1.82x|1.89x|

- 📚 NVIDIA A30 (`*`=MMA Acc: QK F32 + PV F16, `^`=MMA Acc F16, `T`=TFLOPS, **~1.9x↑🎉**)

|Algorithm|320|384|448|512|576|640|704|768|832|896|960|1024|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|SDPA EA|25T|25T|24T|24T|24T|24T|23T|22T|22T|22T|22T|18T|
|FFPA L1*|48T|46T|46T|43T|44T|38T|38T|38T|37T|36T|40T|34T|
|Speedup|1.92x|1.84x|1.92x|1.79x|1.83x|1.58x|1.65x|1.73x|1.68x|1.64x|1.82x|1.89x|
|FFPA L1^|48T|46T|45T|43T|44T|44T|44T|38T|37T|36T|39T|34T|
|Speedup|1.92x|1.84x|1.88x|1.79x|1.83x|1.83x|1.91x|1.73x|1.68x|1.64x|1.77x|1.89x|

<div align='left'>
  <img src='https://github.com/user-attachments/assets/7e323005-4445-41af-8e94-6efb62ed2b77' width="411px">
  <img src='https://github.com/user-attachments/assets/e314649e-82b5-414d-85c9-8b6fbf260138' width="411px">
</div> 

<div id="L1-bench-3080"></div>

- 📚 NVIDIA RTX 3080 Laptop (`*`=MMA Acc F32, `^`=MMA Acc F16, `T`=TFLOPS, **~2.5x↑🎉**)

|Algorithm|320|384|448|512|576|640|704|768|832|896|960|1024|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|SDPA EA|13T|16T|11T|16T|15T|15T|15T|15T|14T|14T|14T|14T|
|FFPA L1*|33T|31T|30T|30T|30T|27T|27T|26T|26T|26T|26T|25T|
|Speedup|2.54x|1.94x|2.73x|1.88x|2.0x|1.8x|1.8x|1.73x|1.86x|1.86x|1.86x|1.79x|
|FFPA L1^|43T|41T|39T|39T|39T|39T|39T|36T|34T|33T|31T|33T|
|Speedup|3.31x|2.56x|3.55x|2.44x|2.6x|2.6x|2.6x|2.4x|2.43x|2.36x|2.21x|2.36x|

- 📚 NVIDIA RTX 3080 Laptop (`*`=MMA Acc: QK F32 + PV F16, `^`=MMA Acc F16, `T`=TFLOPS, **~2.9x↑🎉**)

|Algorithm|320|384|448|512|576|640|704|768|832|896|960|1024|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|SDPA EA|13T|15T|12T|15T|14T|15T|14T|14T|14T|14T|14T|14T|
|FFPA L1*|38T|36T|34T|35T|34T|31T|32T|31T|30T|28T|27T|27T|
|Speedup|2.92x|2.4x|2.83x|2.33x|2.43x|2.07x|2.29x|2.21x|2.14x|2.0x|1.93x|1.93x|
|FFPA L1^|44T|41T|39T|39T|38T|39T|39T|36T|34T|32T|31T|33T|
|Speedup|3.38x|2.73x|3.25x|2.6x|2.71x|2.6x|2.79x|2.57x|2.43x|2.29x|2.21x|2.36x|

<div align='left'>
  <img src='https://github.com/user-attachments/assets/d157cd69-4444-4735-a691-edaaff408137' width="411px">
  <img src='https://github.com/user-attachments/assets/3ce47627-e79d-40ee-b753-bdd235603b7d' width="411px">
</div> 

<div id="L1-bench-4090"></div>

- 📚 NVIDIA RTX 4090 (`*`=MMA Acc F32, `^`=MMA Acc F16, `T`=TFLOPS, **~1.8x↑🎉**)

|Algorithm|320|384|448|512|576|640|704|768|832|896|960|1024|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|SDPA EA|81T|94T|85T|85T|79T|81T|79T|80T|79T|80T|78T|78T|
|FFPA L1*|149T|150T|150T|150T|150T|140T|140T|140T|139T|139T|137T|134T|
|Speedup|1.84x|1.6x|1.76x|1.76x|1.9x|1.73x|1.77x|1.75x|1.76x|1.74x|1.76x|1.72x|
|FFPA L1^|194T|194T|189T|191T|197T|188T|184T|180T|177T|172T|171T|171T|
|Speedup|2.4x|2.06x|2.22x|2.25x|2.49x|2.32x|2.33x|2.25x|2.24x|2.15x|2.19x|2.19x|

- 📚 NVIDIA RTX 4090 (`*`=MMA Acc: QK F32 + PV F16, `^`=MMA Acc F16, `T`=TFLOPS, **~2.1x↑🎉**)

|Algorithm|320|384|448|512|576|640|704|768|832|896|960|1024|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|SDPA EA|82T|92T|85T|84T|78T|81T|79T|80T|78T|79T|77T|78T|
|FFPA L1*|176T|170T|171T|171T|171T|161T|160T|161T|160T|158T|165T|164T|
|Speedup|2.15x|1.85x|2.01x|2.04x|2.19x|1.99x|2.03x|2.01x|2.05x|2.0x|2.14x|2.1x|
|FFPA L1^|200T|191T|189T|191T|188T|188T|186T|179T|175T|173T|172T|170T|
|Speedup|2.44x|2.08x|2.22x|2.27x|2.41x|2.32x|2.35x|2.24x|2.24x|2.19x|2.23x|2.18x|

<div align='left'>
  <img src='https://github.com/user-attachments/assets/447e2937-f7c8-47c8-8550-8c0c71b910e6' width="411px">
  <img src='https://github.com/user-attachments/assets/65a8d564-8fa7-4d66-86b9-e238feb86143' width="411px">
</div> 

## 📖 Python Testing
<div id="python-test"></div>

👇You can test many custom FFPA kernels via Python and figure out the difference in their performance. The `--gen-bench` and `--plot` options help you generate a benchmark table in Markdown style and speedup bar plots on your device. Contributions of your benchmark tables and plots are welcome via a PR 🎉🎉.

- 📚 case: B=1, H=48, N=8192, D=320(`FA2 not supported`)
```bash
# You can test on many devices, such as Volta, Ampere, Ada, Hopper, ...
cd tests && python3 test.py --B 1 --H 48 --N 8192 --show-all --D 320
```
- 📚 case: Generate benchmark table and speedup bar plots on Your device.
```bash
cd tests && pip install matplotlib && python3 test.py --gen-bench --show-all --plot
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

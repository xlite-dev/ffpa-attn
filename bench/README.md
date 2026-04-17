# 📖 FFPA Benchmark 🎉🎉

## Quick Start

```bash
python3 bench/bench_ffpa_attn.py --B 1 --H 48 --N 8192 --show-all --D 320
```

## Benchmark Summary

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

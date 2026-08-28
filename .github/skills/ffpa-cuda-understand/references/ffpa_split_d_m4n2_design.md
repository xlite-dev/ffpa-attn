# Split-D + M4N2 TiledMma: Large Headdim Attention 设计方案

## 1. 问题定义

ffpa-attn 的 CuTe kernel 在处理 large headdim（D≥512）时面临两个独立的 O(D) 瓶颈：

| 瓶颈 | 根因 | 现状（D=512） | 硬件上限 |
|------|------|--------------|---------|
| 寄存器 | PV GEMM 的 N 方向 = D | 256 fp32/thread（略超 255 上限，边缘 spill） | 255 regs/thread |
| SMEM | Persist-D 存储 full-D tile | 192KB（kBr=64、stages=1 最小配置，仍超限） | sm120a: 99KB |

两个瓶颈正交，必须分别解决。本文档论证：**split-D 解决 SMEM 的 O(D)，M4N2 TiledMma 解决寄存器的 O(D) 常数，两者组合是唯一可行方案。**

> **状态**：本方案已实现并验证（正确性 O_err≈1e-4，性能见 §4.6 实测表）。
> 实现：`csrc/cuffpa/cute/attn_traits.cuh` 的 `FFPAAttnCuTeSplitDM4N2Traits`、
> `csrc/cuffpa/cute/fwd_sm120.cuh` 的 `ffpa_attn_split_d_m4n2_fwd_cute_sm120`、
> `csrc/cuffpa/launch.cuh` 的 dispatch（D≥768 走 M4N2）。
> 本文档已按实际实现修正 regs/SMEM 计算，并补充实测数据。

## 2. 根因分析

### 2.1 寄存器 O(D)：PV 的 N 方向就是 D

Attention 的两个 GEMM：

$$S = Q \cdot K^T \quad [Br, Bc] = [Br, D] \times [D, Bc]$$

$$O = P \cdot V \quad [Br, D] = [Br, Bc] \times [Bc, D]$$

MMA m16n8k16 的语义：每 warp（32 threads）产出 [16, 8] 的 C-fragment，每 thread 持有 4 个 fp32 寄存器。

**QK GEMM**：M=Br, N=Bc, K=D。N 方向是 Bc（定长，典型值 32/64/128），不随 D 增长。M8N1 layout 下每 warp 持有 [16, Bc] 的 S，寄存器 = $16 \times Bc / 32 = Bc/2$，与 D 无关。**QK 用 M8N1 没有问题。**

**PV GEMM**：M=Br, N=D, K=Bc。N 方向就是 D（headdim），随模型配置线性增长。M8N1 layout（`atom_layout=(8,1,1)`）下：

- 8 warps 沿 M 方向排列，1 warp 沿 N 方向
- 每 warp 持有 O 的 [16, D] 切片
- 每 thread 寄存器 = $16 \times D / 32 = D/2$ fp32

$$\boxed{\text{regs/thread} = \frac{kBr \times D}{N_{threads}}}$$

这是 CTA 级不变量：总寄存器 = kBr × D / 256，与 warp 排列无关。但**每 warp 的寄存器压力**取决于它持有多少 N 列：

$$\text{regs/thread} = \frac{16 \times \text{cols\_per\_warp}}{32} = \frac{\text{cols\_per\_warp}}{2}$$

**为什么 M4N1（只减 M，不加 N）无效：**

| Layout | kBr | warps_M | warps_N | 每 warp 持有 O 切片 | cols_per_warp | regs/thread |
|--------|-----|---------|---------|-------------------|--------------|-------------|
| M8N1 | 128 | 8 | 1 | [16, D] | D | D/2 |
| **M4N1** | **64** | **4** | **1** | **[16, D]** | **D** | **D/2（不变！）** |
| M4N2 | 64 | 4 | 2 | [16, D/2] | D/2 | D/4 |

M4N1 将 kBr 从 128 降到 64（CTA 处理的 Q 行数减半），但 N=1 意味着**每个 warp 仍然独占完整的 D 列**。每 warp 的 [16, D] 切片大小不变，寄存器压力不变。减少 M 只是让 CTA 少干活（吞吐减半），不减轻单 warp 负担。

**根因是 N=1，不是 M=8。** PV 的 N 方向就是 D（headdim），N=1 时单 warp 必须持有 full-D。只有将 N 从 1 拆到 2（M4N2），每 warp 才只持有 D/2 列，寄存器真正减半。

M4N2（`atom_layout=(4,2,1)`）+ kBr=64：

- 4 warps 沿 M，2 warps 沿 N
- 每 warp 持有 O 的 [16, D/2] 切片
- 每 thread 寄存器 = $16 \times (D/2) / 32 = D/4$ fp32

### 2.2 SMEM O(D)：Persist-D 的 full-D tile

Persist-D 策略将 Q/K/V 的完整 D 维度一次性加载到 SMEM：

$$\text{SMEM} = (kBr + kStages_K \times kBc + kStages_V \times kBc) \times D \times \text{sizeof(Element)}$$

| D | kBr=64, kBc=64, stages=1 | kBr=128, kBc=64, stages=2 |
|---|--------------------------|--------------------------|
| 128 | 48KB | 96KB |
| 256 | 96KB（刚好 fit sm120a） | 192KB |
| 512 | 192KB（超限） | 384KB |
| 1024 | 384KB | 768KB |

SMEM 与 D 严格线性。无论 M8N1 还是 M4N2，只要走 persist-D，SMEM 就是 O(D)。**M4N2 不改变 SMEM 复杂度。**

## 3. 方案排除

### 3.1 Persist-D + M4N2（v2 方案）

- 寄存器：D/4 ✓（D=512 → 128 regs）
- SMEM：O(D) ✗（D=512 → 192KB，超 sm120a 99KB）
- 结论：仅适用于 D≤256 的快速路径，不具备 large headdim 可扩展性

### 3.2 Split-D + M8N1（现有方案）

- SMEM：O(1) ✓（只存 kQKDChunk/kVDChunk 切片，与 D 无关）
- 寄存器：D/2 ✗（D=512 → 256 regs，超 255 上限 1 个，边缘 spill 可容忍；D=768 → 384 regs，spill 加剧；D=1024 → 512 regs，大量 spill 到 local memory，性能崩塌）
- 实测（§4.6）：D≤640 时 M8N1 仍是最优（spill 量小）；D≥768 起 spill 代价超过 M4N2 的 kBr 减半代价
- 结论：SMEM 解决了，但寄存器仍然是 O(D)，D≥768 时必须换 M4N2

### 3.3 混合 MMA layout（QK M8N1 + PV M4N2）

直觉：QK 的 N=Bc 是定长，M8N1 没问题；PV 的 N=D 是变量，用 M4N2 省寄存器。

**不可行，原因有二：**

**(a) kBr 必须一致。** QK 和 PV 共享同一个 Q tile（kBr 行），TiledMma 的 M 维度必须匹配。若 QK 用 M8N1（kBr=128），PV 也必须 kBr=128。此时 PV M4N2 的寄存器 = $128 \times D / 256 = D/2$，**与 M8N1 完全相同，无任何改善**。

**(b) 若强制 kBr=64 + QK M8N1：** M8N1 的 atom_layout M=8 × m16 = 128 行，但 kBr=64 只有 64 行。4 个 warp 有活干，4 个空转。**50% QK 算力浪费。**

### 3.4 半 warp 空转方案（v1 错误思路）

v1 设计让 warp 0-3 做 QK（M4N1），warp 4-7 空闲等待，然后全部 8 warps 做 PV（M4N2）。

性能建模（D=512, Bc=64, Tc=128）：

| 阶段 | M8N1 (baseline) | v1 (4-warp QK) | v2 (8-warp QK) |
|------|-----------------|----------------|----------------|
| QK FLOPs | 2×128×64×512 | 2×64×64×512（半量） | 2×64×64×512 |
| QK warps | 8 | 4 | 8 |
| QK 时间 | T | T（半量但半 warp） | T/2 |
| PV 时间 | T' | T'/2（regs 减半） | T'/2 |

v1 的 QK 阶段：计算量减半但 warp 也减半，时间不变。相比 v2 白白浪费了 50% 的 QK 算力。**v1 在 D=512 时与 M8N1 baseline 打平，无收益。**

## 4. 最终方案：Split-D + M4N2（全 8 warps 参与）

### 4.1 核心思路

- **Split-D** 将 D 维度切成 kQKDChunk/kVDChunk 大小的切片，SMEM 只存切片 → O(1)
- **M4N2** 将 kBr 从 128 降到 64，8 warps 排列为 4M×2N → 寄存器 D/4
- QK 和 PV **共用同一个 M4N2 TiledMma**，8 warps 全程参与，无空转

### 4.2 参数

```
kBr = 64, kBc = 64            // static_assert 固定，M4N2 布局要求
kQKDChunk = 64, kVDChunk = 64
kStagesQK = 2, kStagesPV = 2  // launcher 将 kStage clamp 到 [2, 3]，实测用 2
kNumWarps = 8, kNumThreads = 256
atom_layout = (4, 2, 1)   // 4M × 2N × 1K
```

### 4.3 SMEM 数学推导

Split-D 下 SMEM 只存 D-chunk 切片：

$$\text{SMEM} = \underbrace{kStages_{QK} \times kBr \times kQKDChunk}_{Q} + \underbrace{kStages_{QK} \times kBc \times kQKDChunk}_{K} + \underbrace{kStages_{PV} \times kBc \times kVDChunk}_{V}$$

代入参数（单位：elements，fp16 = 2 bytes）：

$$\text{SMEM} = 2 \times 64 \times 64 + 2 \times 64 \times 64 + 2 \times 64 \times 64 = 24576 \text{ elements} = 48\text{KB}$$

加上 P staging（kBr × kBc × 2B = 8KB）和 softmax cross-warp 交换区（max/sum 两个独立 region，各 8 warps × 16 rows × 4B float = 512B，共 1KB）：

$$\text{Total} = 48 + 8 + 1 = 57\text{KB}$$

即 `FFPAAttnCuTeSplitDM4N2Traits::kSmemElems = 24576 + 4096 + 512 = 29184 elements = 58368 B`。

> 交换区按 [8 warps][16 rows] 而非 [kBr] 分配：M4N2 下每 warp 只覆盖自己 m16 块的
> 16 行，peer N-warp 按 `warp_id ^ 4` 读取；max 与 sum 必须分开存放（两 region 间
> 无 `__syncthreads`，共用会引入 cross-warp RAW hazard）。
> Epilogue 的 O staging 直接复用 V-stage SMEM（KV 循环结束后 V 已 free），
> 由 `static_assert(cosize(SmemLayoutO) <= kStagesPV * cosize(SmemLayoutV))` 保证，
> 不额外占 SMEM。

**与 D 完全无关。** D=512、D=1024、D=2048 均为 57KB，远低于 sm120a 的 99KB 上限。

对比 persist-D：

| D | Persist-D SMEM | Split-D SMEM |
|---|---------------|-------------|
| 128 | 48KB | 57KB |
| 256 | 96KB | 57KB |
| 512 | 192KB ✗ | 57KB ✓ |
| 1024 | 384KB ✗ | 57KB ✓ |

### 4.4 寄存器数学推导

**O accumulator（主要开销）：**

PV GEMM 每 v_chunk 产出 O[Br, kVDChunk] = [64, 64]。M4N2 下每 warp 持有 [16, 32]：

$$\text{regs per v\_chunk per thread} = \frac{16 \times 32}{32} = 16 \text{ fp32}$$

Online softmax 要求所有 v_chunk 的 O 切片在 KV-tile 循环中保持 live（rescale 需要），实现中即 `float o_acc_storage[kDChunksV][kOElemsPerFrag]`（`kOElemsPerFrag = 16`）：

$$\text{total O regs/thread} = kDChunks_V \times 16 = \frac{D}{kVDChunk} \times 16 = \frac{D}{64} \times 16 = \frac{D}{4}$$

| D | M8N1 (kBr=128) | M4N2 (kBr=64) | 255-reg 上限 |
|---|---------------|--------------|-------------|
| 128 | 64 | 32 | ✓ |
| 256 | 128 | 64 | ✓ |
| 512 | 256 ✗ | 128 | ✓ |
| 1024 | 512 ✗ | 256 | ✗（超 1 reg，见下文实测说明） |

> **实测修正**：256 regs 的 M8N1@D=512 与 M4N2@D=1024 都只超 255 上限 1 个寄存器，
> 编译器仅 spill 边缘寄存器，实际均可运行且性能良好（§4.6：M8N1@D=512 194T、
> M4N2@D=1024 154T）。真正崩塌的是 M8N1@D≥896（≥448 regs，大量 local memory spill）。
> M4N2 的意义不是"让 D=512 可行"，而是把崩塌点从 ~D=768 推迟到 D≥2048。

**其他寄存器开销（每 thread）：**

| 项目 | 数量 | 说明 |
|------|------|------|
| QK S accumulator | Bc/4 = 16 fp32 | M4N2 下每 warp [16, Bc/2] |
| Q/K A/B fragments | ~48 regs（fp16） | gemm_ss 全 fragment 驻留：Q [16,64] 16 regs + K [64,32] 32 regs per warp |
| P A-fragment (PV) | 32 fp16 = 16 regs | gemm_rs 要求 P 全 k-tile 驻留寄存器 |
| V B fragment | 64 fp16 = 32 regs | [64,32] per warp，1-ahead 双缓冲 |
| row_max, row_sum, row_scale | 3 × kORows = 6 | **kORows = 2**（m16n8 fragment 每 thread 覆盖 2 行，非 4） |
| P staging (SMEM roundtrip) | 0 | 不占寄存器 |
| 循环变量、地址计算 | ~16 | |

> `kORows/kOCols` 由 `convert_layout_acc_rowcol(OFragLayout)` 决定：m16n8k16 的
> C-fragment 每 thread 持有 2 行 × 8 列（kORows=2, kOCols=8，共 16 元素），
> 4 个 lane（`lane_id/4` 相同）共享同一逻辑行，row 归约用 `__shfl_xor<1>/<2>`。
> QK 与 PV 的 fragment 生命周期不重叠（Q/K frags 在 softmax 前已死，S 转 P
> 复用存储），按 phase 取峰值：QK 阶段 ≈ 128(O)+16(S)+48(Q/K frags)+16 ≈ 208；
> PV 阶段 ≈ 128(O)+16(P A-frag)+32(V frag)+6(softmax)+16 ≈ 198
> （P fp32→fp16 转换瞬间额外 ~8 regs）。D=512 峰值约 210 regs/thread，
> 在 255 上限内，无 spill（fragment 数为 CuTe partition 精确值 + 调度估算）。

### 4.5 与现有方案的核心区别

| 维度 | 现有 Split-D (M8N1) | Persist-D (M8N1) | **Split-D + M4N2（本方案）** |
|------|---------------------|-------------------|---------------------------|
| kBr | 128 | 128 | **64** |
| atom_layout | (8,1,1) | (8,1,1) | **(4,2,1)** |
| SMEM 复杂度 | O(1) | O(D) | **O(1)** |
| O regs/thread | D/2 | D/2 | **D/4** |
| D=512 可行 | ✓（256 regs 边缘 spill，实测最快） | ✗（SMEM 超限） | **✓** |
| 崩塌点 | D≥896（大量 spill） | D≥256（SMEM） | **D≥2048（O regs 512）** |
| P→PV 路径 | 寄存器直转 | 寄存器直转 | **SMEM roundtrip** |
| Softmax | warp 内 shfl | warp 内 shfl | **cross-N-warp SMEM 交换** |
| QK warp 利用率 | 100% | 100% | **100%** |

### 4.6 实测性能与最终 dispatch（实现后回填）

RTX 5090 (SM120)，torch 2.13.0+cu132，self-attn N=8192，stages=2，fp16（bf16 ±2%），
正确性 O_err≈1e-4 两者一致（FFPA time ms / TFLOPS）：

| D | M8N1 (ms/TFLOPS) | M4N2 (ms/TFLOPS) | winner |
|---|------------------|------------------|--------|
| 320 | 13.21 / 208T | 15.35 / 179T | M8N1 +16% |
| 384 | 16.37 / 202T | 18.13 / 182T | M8N1 +11% |
| 448 | 20.33 / 189T | 20.77 / 185T | M8N1 +2% |
| 512 | 22.69 / 194T | 23.73 / 185T | M8N1 +4% |
| 576 | 26.54 / 186T | 26.42 / 187T | M4N2 +0.5% |
| 640 | 29.99 / 183T | 31.28 / 176T | M8N1 +4% |
| 768 | 40.55 / 163T | 37.78 / 175T | M4N2 +7% |
| 896 | 54.03 / 142T | 49.16 / 157T | M4N2 +11% |
| 1024 | 88.37 / 100T | 57.11 / 154T | M4N2 +55% |

关键结论（与前期预期的差异）：

- **M8N1 在 D≤640 仍然占优**。kBr=64 使 CTA 数翻倍、每 CTA 工作量减半，小 D 下
  这一代价大于寄存器收益。交叉点在 D=640~768 之间。
- **最终 dispatch（`launch.cuh`，kHeadDim % 64 == 0 路径）**：
  `D < 768 → M8N1 split-D`，`D >= 768 → M4N2`。
- M8N1@D=1024 崩塌至 100T（o_acc = D/2 = 512 regs 大量 spill 到 local memory）；
  M4N2@D=1024 的 o_acc = D/4 = 256 regs 仅超上限 1 个，保持 154T（+55%）。

## 5. Kernel 设计要点

### 5.1 TiledMma 配置

QK 和 PV 共用同一个 M4N2 布局：

```cpp
using AtomLayoutMN = Layout<Shape<_4, _2, _1>>;

using TiledMmaQK = decltype(make_tiled_mma(
    MmaAtom{}, AtomLayoutMN{}, Tile<Int<kBr>, Int<kBc>, _16>{}));

using TiledMmaPV = decltype(make_tiled_mma(
    MmaAtom{}, AtomLayoutMN{}, Tile<Int<kBr>, Int<kVDChunk>, _16>{}));
```

- QK：Tile<64, 64, 16>，每 warp 产出 [16, 32] 的 S（Bc 的半列）
- PV：Tile<64, 64, 16>，每 warp 产出 [16, 32] 的 O（D-chunk 的半列）

### 5.2 Cross-N-warp Softmax

M4N2 下，同一 M 行的 S 列分布在 2 个 N-warp 中（warp_id 和 warp_id ^ 4 互为 peer）。Online softmax 的 row-max 和 row-sum 必须跨 peer 归约：

```
warp (m, 0): S[16, 0:32]   ──┐
                              ├── row_max = max(local_max, peer_max)
warp (m, 1): S[16, 32:64]  ──┘
```

实现（`online_safe_softmax_m4n2` + `finalize_row_sum_m4n2`）：

- 交换区布局 `[8 warps][16 rows]` float：每 warp 覆盖自己 m16 块的 16 行，
  每行由 4 个 lane 共享（`lane_id%4==0` 的 lane 负责写），warp 内先做
  `__shfl_xor<1>/<2>` 归约。max 和 sum 使用**两个独立 region**（无 sync 间隔时
  共用会引入 cross-warp RAW hazard）。
- **只增加 1 个 CTA barrier**：max 写入后 `__syncthreads()` 读 peer
  （`warp_id ^ 4`）的 max 合并；sum 写入后**不加 barrier**，由 Phase 3 的
  P stmatrix→LDSM_N `__syncthreads()` 顺带发布，`finalize_row_sum_m4n2`
  在该 barrier 之后读 peer sum 并累加。比 3-sync 朴素版每 KV tile 省 1 个 barrier。
- 采用 FA-4 条件 rescale（`FFPA_RESCALE_THRESHOLD = 8.0`，log2 域；= FA-4 论文
  BF16 阈值 $\tau=\log_2 256$，本设计 P 走 f16 发射、无 e4m3 448 天花板，
  故可用 8；若移植到 e4m3-P 变体须按 §报告 11.6(c) 收紧）：
  `row_max - next_max >= -threshold` 时跳过 rescale（row_scale=1，沿用旧 max），
  延迟的缩放最终在 epilogue 的 `O / row_sum` 中抵消。

**无需 P 修正**：两个 N-warp 使用相同的 eff_max 做 exp2，各自持有的 P 列已经是正确的归一化前值。

### 5.3 P → PV 的 SMEM Roundtrip

现有 M8N1 split-D 用 `convert_layout_acc_Aregs` 做 QK C-fragment → PV A-regs 的寄存器直转，因为每 warp 持有完整的 [16, Bc] S。

M4N2 下每 N-warp 只持有 P 的半列 [16, Bc/2]，**无法通过 warp 内 reshuffle 重建完整 P**。必须走 SMEM：

1. S（fp32）先经 `convert_type` 转为 Element（fp16/bf16）
2. 每 warp 通过 stmatrix（`SM90_U32x4_STSM_N`，`make_tiled_copy_C` 按 QK TiledMma 切分）
   将自己的 [16, Bc/2] 写入 `SmemLayoutP`（与 QK 同 swizzle atom）的对应列区域
3. `fence_view_async_shared()` + `__syncthreads()`（同时发布 softmax sum 写入）
4. 所有 warp 通过 LDSM_N（`make_tiled_copy_A`）读取完整 [kBr, kBc] P 的
   本 warp A-fragment（跨两个 N-warp 的列），作为 PV 的 A-operand

代价：每 KV tile 一次 SMEM 写读（kBr × kBc × 2B = 8KB），相比寄存器减半的收益可忽略。

### 5.4 Split-D 循环结构

与现有 split-D 完全一致，仅 TiledMma 和 softmax 不同：

```
for kv_tile in 0..Tc:
    __syncthreads()                    // kv_tile>0：上一 tile 的 P/sum 读与本 tile 的交换区/stmatrix 写的复用屏障
    tid0: issue V TMA × kDChunksV for this kv_tile

    // Phase 1: QK GEMM (split-D accumulation)
    S = 0
    for d_chunk in 0..kDChunksQK:
        wait qk_full[stage]
        S += Q_d @ K_d^T               // gemm_ss, M4N2
        arrive qk_empty[stage]; tid0: prefetch next QK stage
    tid0: prefetch next kv_tile's QK stages

    // Phase 2: Online softmax (cross-N-warp, FA-4 conditional rescale)
    masking (tail/causal) → scores
    cross_warp_row_max + exp2          // 1 SMEM exchange + 1 syncthreads
    row_scale = exp2(old_max - new_max) or 1.0 (skip if within threshold)
    need_rescale = __any_sync(row_scale < 1.0)

    // Phase 3: P → SMEM → PV A-regs
    convert_type<Element>(S) → stmatrix → SMEM
    __syncthreads()                    // 同时发布 softmax sum 写入
    finalize_row_sum (读 peer sum)
    LDSM_N(P) → A-regs

    // Phase 4: PV GEMM (split-D accumulation)
    for v_chunk in 0..kDChunksV:
        wait v_full[v_stage]
        if kv_tile > 0 && need_rescale:
            O[v_chunk] *= row_scale    // 按行条件 rescale
        O[v_chunk] += P @ V_d          // gemm_rs (A=P regs, B=V smem LDSM_T)
        arrive v_empty[v_stage]; tid0: prefetch next V stage

// Epilogue: O /= row_sum, 分批 stmatrix → TMA store（见 §5.6）
```

### 5.5 TMA Pipeline

复用现有 split-D 的双 barrier 结构：

- `qk_full[kStagesQK]`（TmaBarrier）：Q+K TMA 完成信号
- `qk_empty[kStagesQK]`（CtaBarrier）：QK stage 消费完毕信号
- `v_full[kStagesPV]`（TmaBarrier）：V TMA 完成信号
- `v_empty[kStagesPV]`（CtaBarrier）：V stage 消费完毕信号

非 WS 模式（tid=0 发 TMA，全 256 threads 做 MMA），与现有 split-D 一致。

### 5.6 Epilogue：分批 TMA-O store（实现补充）

O staging 不额外分配 SMEM：KV 循环结束后 Q/K/V/P 全部 free，staging tile 从
`shm` 起始处放置（物理上复用已释放区域）。

- `kVChunksPerBatch = compute_vchunks_per_batch(kDChunksV, kHeadDim, kBr,
  kStagesPV*kBc*kVDChunk)`：批大小预算为 **V-stage 区域**（stages=2 时 16KB，
  即 2 个 8KB O tile），取最小的 batch 数（kDChunksV 的约数）使
  `[kBr, kHeadDim/n_batches]` 装进预算。stages=2、kVDChunk=64 时恒有
  **kVChunksPerBatch = 2**，kNBatches = kDChunksV/2（D=512 → 4 批，
  D=1024 → 8 批）。
- 每批：O /= row_sum → `convert_type<Element>` → stmatrix 到 `shm` 起始处 →
  `__syncthreads()` → tid0 发 TMA store × kVChunksPerBatch → `tma_store_arrive()`。
- **批间必须 `tma_store_wait<0>()` + `__syncthreads()`（全 CTA）**：只有 tid0 发
  store，若不 gate 全部线程，下一批的 R→S 会覆盖在途 TMA store 仍在读的 SMEM，
  造成确定性 O 错写（kNBatches≥2 时触发）。batch 条件对 CTA 一致，不会死锁。
- 尾部 tile（`Br_base + kBr > Nq`）走 predicated 寄存器直写 gmem，不经 SMEM。
- LSE 只由 `n_warp == 0` 的 warp 写（两个 N-warp 共享相同 Q 行）。

## 6. 可扩展性分析

| D | O regs/thread | SMEM | dispatch | 备注 |
|---|--------------|------|----------|------|
| 128~640 | 32~160 | 57KB | M8N1 | M8N1 实测更快（§4.6） |
| 768 | 192 | 57KB | M4N2 | dispatch 交叉点 |
| 1024 | 256 | 57KB | M4N2 | o_acc 超 255 上限 1 reg，边缘 spill，实测 154T |
| 2048 | 512 | 57KB | ✗ | 需 M2N4 或 D-tile 外循环 |

SMEM 与 D 无关，瓶颈只剩 O 寄存器。D=1024 时 `o_acc_storage` 单项即 256 regs，
已超 255 硬件上限 1 个：编译器仅 spill 边缘寄存器，实测仍保持 154T（对比 M8N1
同 D 崩塌至 100T），但这是当前布局的实际上限。D≥2048 需要：
- M2N4（`atom_layout=(2,4,1)`，kBr=32）进一步减半到 D/8/thread，或
- D-tile 外循环（分批累加 O 并中途落盘，牺牲 rescale 精度或增加 gmem 往返）

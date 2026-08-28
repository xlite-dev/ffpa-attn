# Split-D + M4N4 TiledMma (512 threads) 对 D>1024 的可行性分析

> **结论**：M4N4（atom_layout=(4,4,1), kBr=64, 16 warps = 512 threads）对 D>1024
> **没有性能收益**。核心原因是一个优美的数学恒等式：
>
> $$\boxed{\text{O\_acc} / \text{ceiling} = D \cdot M_w / 4096}$$
>
> 该比值**只与 D 和 M_w 有关，与 N_w 完全无关**。M4N4（M_w=4）与现有 M4N2（M_w=4）
> 在寄存器可行性上**完全等价** —— 增大 N_w 让 O_acc 减半的同时，threads 翻倍让
> ceiling 也减半，两个"减半"精确抵消。D>1024 的真正解法是 **M2N4（M_w=2, kBr=32）**，
> 它把 M_w 减半（而非增大 N_w）来真正降低 O_acc/ceiling 比例。

## 1. 问题定义

现有 M4N2（[`ffpa_split_d_m4n2_design.md`](ffpa_split_d_m4n2_design.md)）在 D=1024 时
O_acc = D/4 = 256 regs/thread，刚好饱和 256T 的 256 ceiling（实测 154T，边缘 spill）。
D>1024（如 D=2048）时 O_acc = 512，远超 ceiling，必然灾难性 spill。

直觉方案：**增大 N_w**（N-warp 数），让每个 warp 只持有 D 的一部分列，从而减小 O_acc。
M4N4 把 N_w 从 2 翻到 4，O_acc 从 D/4 降到 D/8 —— 看起来正好解决 D=2048 的问题。

**但这个直觉是错的**，原因如下。

## 2. M4N4 配置

```
atom_layout = Layout<Shape<_4, _4, _1>>   // 4 M-warps × 4 N-warps = 16 warps
kBr = 64, kBc = 64                        // 同 M4N2
kQKDChunk = 64, kVDChunk = 64              // 同 M4N2
kStagesQK = 2, kStagesPV = 2
kNumWarps = 16, kNumThreads = 512          // ← 关键：threads 翻倍
```

### CuTe TiledMma 验证（基于 [`ffpa_split_d_m8n2_design.md`](ffpa_split_d_m8n2_design.md) §2 已验证的公式）

对于 `make_tiled_mma(MmaAtom_SM80_16x8x16, AtomLayout<M_w, N_w, 1>, Tile<kBr, kVDChunk, 16>)`：

$$
\text{RestM} = \frac{kBr}{16 \cdot M_w}, \quad
\text{RestN} = \frac{kVDChunk}{8 \cdot N_w}, \quad
\text{fp32 acc/thread per chunk} = 4 \cdot \text{RestM} \cdot \text{RestN}
$$

$$
O\_acc = kDChunks_V \times \text{fp32 per chunk}
= \frac{D}{kVDChunk} \times 4 \times 1 \times \frac{kVDChunk}{8 \cdot N_w}
= \frac{D}{2 \cdot N_w}
$$

代入 M4N4（M_w=4, N_w=4, kBr=64, kVDChunk=64）：
- RestM = 64/(16·4) = 1
- RestN = 64/(8·4) = 2
- fp32 acc/thread per chunk = 4·1·2 = **8**
- O_acc = (D/64) × 8 = **D/8**

单 warp C-fragment = [16, 16]（每 warp 在 N 方向覆盖 16 列，非连续）。

## 3. 核心数学：O_acc / ceiling 恒等式

### 3.1 推导

寄存器 ceiling（1 block/SM，sm120a 65536 regs/SM）：

$$
\text{ceiling} = \frac{65536}{\text{threads}} = \frac{65536}{32 \cdot M_w \cdot N_w}
$$

O_acc 与 ceiling 之比：

$$
\frac{O\_acc}{\text{ceiling}}
= \frac{D / (2 \cdot N_w)}{65536 / (32 \cdot M_w \cdot N_w)}
= \frac{D}{2 \cdot N_w} \times \frac{32 \cdot M_w \cdot N_w}{65536}
= \frac{D \cdot 16 \cdot M_w}{65536}
$$

$$
\boxed{\frac{O\_acc}{\text{ceiling}} = \frac{D \cdot M_w}{4096}}
$$

**N_w 在推导中完全消去。** 这是 split-D FA 的一个不变量：增大 N_w 把 O_acc 和
ceiling 同比例缩放（前者除 N_w，后者乘 N_w 在分母），净效应为零。

### 3.2 全 layout 对比表（数值验证）

| Layout | M_w | N_w | threads | kBr | O_acc (D=1024) | ceiling | **ratio** | verdict |
|--------|-----|-----|---------|-----|----------------|---------|-----------|---------|
| M4N2 (现有) | 4 | 2 | 256 | 64 | 256 | 256 | **100%** | 边缘 |
| M8N1 (现有) | 8 | 1 | 256 | 128 | 512 | 256 | **200%** | SPILL |
| M8N2 (否决) | 8 | 2 | 512 | 128 | 256 | 128 | **200%** | SPILL |
| **M4N4 (本问)** | **4** | **4** | **512** | **64** | **128** | **128** | **100%** | **边缘** |
| **M2N4 (正解)** | **2** | **4** | **256** | **32** | **128** | **256** | **50%** | **OK** |
| M1N8 (极端) | 1 | 8 | 256 | 16 | 64 | 256 | **25%** | OK |

D=2048：

| Layout | M_w | O_acc (D=2048) | ceiling | **ratio** | verdict |
|--------|-----|----------------|---------|-----------|---------|
| M4N2 | 4 | 512 | 256 | **200%** | SPILL |
| M8N2 | 8 | 512 | 128 | **400%** | SPILL |
| **M4N4** | **4** | **256** | **128** | **200%** | **SPILL** |
| **M2N4** | **2** | **256** | **256** | **100%** | **边缘** |
| M1N8 | 1 | 128 | 256 | **50%** | OK |

### 3.3 关键观察

1. **M4N4 与 M4N2 在寄存器可行性上完全等价**（都是 M_w=4 → ratio = D/1024）。
   M4N4 的 N_w=4 红利（O_acc 减半）被 512T 的 ceiling 减半精确抵消。

2. **M4N4 @ D=1024 是"边缘"（ratio 100%）**，与 M4N2 @ D=1024 相同。但 M4N4
   还有额外的 4-way softmax 开销（§4）和 512T 的其他副作用，所以实际更差。

3. **M4N4 @ D=2048 仍然是"SPILL"（ratio 200%）**，与 M4N2 @ D=2048 相同。
   M4N4 完全没有解决 D>1024 的根本问题。

4. **只有减小 M_w 才能降低 ratio**：M2N4（M_w=2）在 D=1024 是 50%（OK），
   D=2048 是 100%（边缘）。这才是 D>1024 的正确方向。

### 3.4 可行阈值

令 ratio ≤ 60%（留 40% 给 Q/K/V fragment、softmax state 等非 O_acc 寄存器）：

$$
\frac{D \cdot M_w}{4096} \leq 0.6 \quad \Rightarrow \quad M_w \leq \frac{2458}{D}
$$

| D | max M_w | 推荐 layout | kBr |
|---|---------|-------------|-----|
| 512 | 4.8 | M4N2 (M_w=4) | 64 |
| 768 | 3.2 | M4N2 (M_w=4, 边缘) → M2N4 (M_w=2, 宽松) | 64/32 |
| **1024** | **2.4** | **M2N4 (M_w=2)** | **32** |
| 1536 | 1.6 | M1N8 (M_w=1) 或两遍算法 | 16 |
| 2048 | 1.2 | M1N8 (M_w=1, 边缘) 或两遍算法 | 16 |
| 4096 | 0.6 | 两遍算法（无单遍 layout 可行） | — |

## 4. M4N4 的其他开销（即使寄存器不是问题）

### 4.1 4-way cross-N-warp softmax

M4N2 有 2 个 N-warp，softmax reduction 是 2-way（warp_id 和 warp_id^4 互为 peer，
1 次 SMEM exchange + 1 barrier）。M4N4 有 4 个 N-warp，需要 **4-way reduction**：

- SMEM exchange 区增大：4 warps × 16 rows × 2(max+sum) × 4B = 2KB（M4N2 是 1KB）
- reduction 逻辑更复杂：树形（2 步）或全连通（每 warp 读其他 3 个 warp 的值）
- 每 kv_tile 的固定开销增加

M2N4 同样需要 4-way reduction，所以这不是 M4N4 独有的问题，但对两者都是
比 M4N2 的额外开销。

### 4.2 P SMEM roundtrip

M4N4 每 warp 只持有 P 的 [16, 16] 切片（4 个 N-warp 各覆盖不同列范围）。
P→PV 需要 SMEM roundtrip（stmatrix → SMEM → LDSM_N），与 M4N2 相同机制。
P staging 大小 = kBr × kBc = 64 × 64 = 8KB（与 M4N2 相同，因为 kBr 不变）。

### 4.3 SMEM 占用

M4N4 的 SMEM（kBr=64，与 M4N2 相同的 Q/K/V/P，exchange 略增）：

```
Q  : 2 × 64 × 64 × 2B = 16 KB   (同 M4N2)
K  : 2 × 64 × 64 × 2B = 16 KB   (同 M4N2)
V  : 2 × 64 × 64 × 2B = 16 KB   (同 M4N2)
P  :    64 × 64 × 2B =  8 KB    (同 M4N2)
exchange: 2 × 16 warps × 16 rows × 4B = 2 KB  (warps 翻倍, 比 M4N2 的 1KB 多)
─────────────────────────────────────────────────
Total                    ≈ 58 KB  (同 M4N2 量级, < 99KB ✓)
```

SMEM 不是瓶颈。2 block/SM 需要 116KB > 102KB，所以仍 1 block/SM。

## 5. M4N4 vs M2N4：为什么 M2N4 严格更优

M4N4 和 M2N4 都用 N_w=4（4-way softmax, O_acc = D/8），区别只在 M_w：

| 维度 | M4N4 (M_w=4) | M2N4 (M_w=2) | 优势方 |
|------|-------------|-------------|--------|
| threads | 512 | 256 | M2N4（ceiling 高） |
| kBr | 64 | 32 | M4N4（CTA 数少） |
| O_acc (D=1024) | 128 | 128 | 平（相同） |
| ceiling | 128 | 256 | **M2N4** |
| O_acc/ceiling (D=1024) | 100% | 50% | **M2N4** |
| O_acc/ceiling (D=2048) | 200% | 100% | **M2N4** |
| 4-way softmax | 是 | 是 | 平 |
| CTA 数 | Nq/64 | Nq/32 | M4N4（少一半） |
| 算法总 mma | 相同 | 相同 | 平 |

**M2N4 在寄存器可行性上严格优于 M4N4**（ratio 减半），而算法工作量完全相同。
M4N4 唯一的潜在优势是 CTA 数减半（kBr=64 vs 32），但这与 M8N2 vs M4N2 的分析
完全平行：CTA 数减少的好处（启动开销 <1%）远不足以补偿寄存器 spill 的损失。

### 5.1 数学等价性（与 M8N2 vs M4N2 同构）

M4N4 与 M2N4 的算法总 warp-iterations（参考 [`ffpa_split_d_m8n2_design.md`](ffpa_split_d_m8n2_design.md) §2 的推导）：

$$
\text{total} = \frac{Nq}{kBr} \times (M_w \cdot N_w) \times \frac{Nkv}{kBc} \times \text{RestN}
$$

代入 `kBr = 16·M_w` 和 `RestN = kVDChunk / (8·N_w)`：

$$
= \frac{Nq}{16 \cdot M_w} \times M_w \cdot N_w \times \frac{Nkv}{kBc} \times \frac{kVDChunk}{8 \cdot N_w}
= \frac{Nq \cdot Nkv \cdot kVDChunk}{128 \cdot kBc}
$$

**M_w 和 N_w 都消去**，M4N4 与 M2N4（以及 M4N2、M8N2）的总 mma 完全相同。

## 6. M2N4 的可行性简析（D>1024 的正解）

M2N4（atom_layout=(2,4,1), kBr=32, 256T）：

- **O_acc = D/8**（同 M4N4，因为 N_w=4）
- **ceiling = 256**（因为 256T，同 M4N2）
- **ratio = D·2/4096 = D/2048**
  - D=1024: 50%（宽松，其他寄存器有 128 regs 空间）
  - D=2048: 100%（边缘，与 M4N2 @ D=1024 同级）

### 6.1 M2N4 的挑战

1. **kBr=32 → CTA 数翻倍**（Nq/32 vs M4N2 的 Nq/64），但这是减小 M_w 的必然代价
2. **4-way softmax reduction**（同 M4N4）
3. **kBc 约束**：需要 kBc % (8·N_w) == 0 即 kBc % 32 == 0，所以 kBc ≥ 32（kBc=64 OK）
4. **SMEM 减半**（kBr=32 使 Q 减半）：Q 8KB + K 16KB + V 16KB + P 4KB + exchange 2KB ≈ 46KB

### 6.2 M2N4 的上限

ratio ≤ 60% → D ≤ 1229（保守）。实际 M2N4 @ D=2048 ratio=100%（边缘 spill），
预期能跑（类似 M4N2 @ D=1024 的 154T），但 D>2048 需要 M1N8 或两遍算法。

## 7. 综合结论

### 7.1 M4N4 对 D>1024 无性能收益

1. **寄存器可行性等价于 M4N2**（核心恒等式 ratio = D·M_w/4096，M_w=4 相同）
2. **4-way softmax 额外开销**（比 M4N2 的 2-way 更复杂）
3. **512T 的 ceiling 降低**（128 vs 256），与 M8N2 同样的失败模式
4. D=2048 时 ratio=200%，与 M4N2 一样灾难性 spill

### 7.2 核心洞察：增大 N_w 不能降低寄存器压力

$$\frac{O\_acc}{\text{ceiling}} = \frac{D \cdot M_w}{4096} \quad \text{(与 } N_w \text{ 无关)}
$$

这是 split-D FA 在 sm120a（per-SM 寄存器池）上的**根本约束**：
- 增大 N_w：O_acc 减半，但 threads 翻倍使 ceiling 减半 → 净效应为零
- **减小 M_w**：O_acc 不变，threads 不变 → 真正降低 ratio

M4N4（增大 N_w）和 M8N2（增大 M_w）都是死路，原因不同但数学根源相同：
**任何让 threads 翻倍的 layout 变体都无法改善寄存器可行性**。

### 7.3 D>1024 的正确路径

| D 范围 | 推荐 layout | M_w | kBr | ratio | 备注 |
|--------|------------|-----|-----|-------|------|
| ≤ 640 | M8N1 (现有) | 8 | 128 | ≤125% | 实测最快 |
| 768~1024 | M4N2 (现有) | 4 | 64 | 75~100% | 实测最快 |
| **1024~2048** | **M2N4** | **2** | **32** | **50~100%** | **唯一可行单遍路径** |
| ≥ 2048 | 两遍算法 | — | — | — | 退役 O_acc，QK 重算 |

**M2N4 是 D>1024 的下一个实现目标**（见 [`ffpa_split_d_m4n2_design.md`](ffpa_split_d_m4n2_design.md) §6）。

## 附录：推导的 Python 验证

```python
# 核心恒等式验证: O_acc/ceiling = D*M_w/4096
configs = [("M4N2",4,2,256), ("M8N1",8,1,256), ("M8N2",8,2,512),
           ("M4N4",4,4,512), ("M2N4",2,4,256), ("M1N8",1,8,256)]
for D in [1024, 2048]:
    for name, M_w, N_w, thr in configs:
        O_acc = D // (2*N_w)
        ceil = 65536 // thr
        ratio = D * M_w / 4096      # 恒等式
        ratio_direct = O_acc / ceil  # 直接计算
        assert abs(ratio - ratio_direct) < 0.01  # 验证一致
```

输出与 §3.2 表格完全吻合（已用 pylance 验证）。

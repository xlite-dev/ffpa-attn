# SageAttention 1/2/2++/3 核心技术原理

> 依据：`references/papers/SageAttention{1,2,2++,3}.txt`。
> 本文梳理每代的核心技术与数学动机，并标注 ffpa-attn CUDA backend 的对应实现
> （配合主文档 §11 数学原理、§5/§6 量化路径）。
> 一句话脉络：**SA1 证明量化 attention 可行（int8 QK + fp16 PV）→ SA2 下探 int4 QK + fp8 PV
> 并系统化 smoothing → SA2++ 换用 2x 快的 fp8-f16acc MMA 并推导值域约束 → SA3 进
> Blackwell FP4（microscaling + 两级 P 量化）**。

---

## SA1（SageAttention，arXiv 2410.02367）

**目标**：第一个端到端无损的量化 attention（plug-and-play PTQ），RTX4090/3090 上
2.1x/2.7x vs FA2/xformers。

**两大挑战与对策**：

1. **(C1) K 的通道级 outlier**：直接量化 K 精度崩塌。对策 **smooth-K**——减去 K 的
   跨 token 通道均值 $\bar{k}$；由 softmax 逐行平移不变性，$O$ 严格不变（只有 lse 需
   补偿 $\text{scale}\cdot q\bar{k}$）。开销 <0.2%。
2. **(C2) P/V 直接 int8 不稳定**：对策 **PV 保持 fp16 但用 fp16 累加器**
   （`mma.f16.f16.f16`，pv_fp16_qk_int8 模式）——速度翻倍且精度无损（fp16 acc 的
   舍入在 softmax 概率域可控）。

**关键选型**：
- **Q/K 用 int8 而非 fp8**：(a) 4090/3090 上 int8 MMA 是 fp16 的 4x、fp8 的 2x；
  (b) int8 均匀步长对 $|S|$ 小的区域误差更小（fp8 指数格式小值区分辨率浪费）。
- Q/K per-block 量化；per-layer 自动选最快且精度达标的实现变体。
- 工程：fused RoPE+quantize kernel、FA 风格 tiling、`mma.u8.u8.s32`。

**ffpa 对应**：`fp8_qk_mm_type='int8'`（int8 QK 变体）、`fp8_smooth_k`（默认开）、
`fp8_pv_acc_type='f16'`；§11.5（int8 均匀误差）/§11.6(a)（s_dequant 折叠）/§11.8(a)。

---

## SA2（SageAttention2，arXiv 2411.10958）

**目标**：int4 QK（更快 MMA）+ fp8 PV，保持端到端无损。

**三大技术**：

1. **per-thread 量化粒度**：per-block 对 int4 不够准，per-token 又引入每线程多
   scale 的 dequant 开销。解法：按 PTX mma 的 **thread↔内存布局映射**分组——同一
   线程持有的元素归一组、每线程单一 scale（Q 64 scale/128 行块、K 4/kBc 列块），
   精度远高于 per-block 且零额外 dequant。
2. **smooth-Q（rank-1 修正 delta_s）**：减 Q 通道均值 $q_m$（每 128 行块）后补
   $\Delta S = q_m K^\top$（rank-1 GEMV 广播），配合 SA1 smooth-K 后 $S$ 只差行常数、
   由 lse 吸收。
3. **FP8 PV 的 FP32 buffer**：发现 `mma.f32.f8.f8.f32` 的"FP32"累加器实为
   **FP22（1+8+13）**——长 KV 累加误差累积。对策：每个 block matmul 后把 FP22
   累加值搬进真 FP32 buffer，误差限制在块内。
4. （可选）**smooth-V**：V 减通道均值、epilogue 加回（softmax 行和为 1，$O$ 不变）。

**其它**：per-channel V scale $\delta_V=\mathrm{colmax}(|V|)/448$；per-thread int4 的
fragment 对齐量化/反量化。

**ffpa 对应**：`fp8_q/k_quant_method='per_thread'`（§11.5 粒度谱系）、
fp4 强制的 `qm/km + delta_s`（`cute/fp4/delta_s.cuh`，§11.8(b)）、
`fp8_smooth_v`（§11.8(c)）、per-channel V（§5.3）。注意：ffpa 未走 int4 QK 路线——
sm_120 无硬件整数 INT4 MMA（无 `tcgen05.mma`；Tensor Core 原生仅
NVFP4/FP8/BF16/FP16/INT8），int4 只能软件解码 + 上转 INT8/FP16 走现有 TC
通路、比原生路径更慢；fp4/fp8 在 sm_120 均不走 int4 QK。int4 思想以
per-thread 粒度形式保留在 fp8 路径；int4 QK 仅对 sm_89（Ada 原生 int4 MMA、
2x int8）有意义，见 [RFC PC-6](rfc-future-optimizations.md)（低优搁置）。

**开源状态**（2026-08-28 核对本地 `SageAttention/` 仓库）：SA2 的 **int4 QK
attention kernel 未开源**。开源仓库仅保留编译期脚手架：`attn_utils.cuh` 的
`kInt4` 枚举与 quant/dequant 分支、sm80/sm89 kernel 模板的 static_assert 允许
kInt4、`mma.cuh` 的两个 int4 MMA wrapper（m16n8k32 / m16n16k64）；但已发布的
7 个实例化文件全是 `sm89_qk_int8_*`，`pybind_sm89.cpp` 只暴露 `qk_int8_*`，
Python 侧零 int4 引用。实现 int4 QK 需自写 kernel 本体。

---

## SA2++（arXiv 2505.21136）

**目标**：SA2 的更快实现版——换用 **`mma.f16.f8.f8.f16`**（fp8 matmul + fp16 累加器），
该指令是 fp16 的 4x（`mma.f32.f8.f8.f32` 只有 2x），精度与 SA2 持平，3.9x vs FA。

**核心数学——fp16 累加器值域约束**（§3.1）：
fp16 累加器上限 65504，m16n8k32 的 K 深度 32，要求

$$|32\cdot p\cdot v|\le65504 \quad\Rightarrow\quad P_r\times V_r\le2047$$

其中 $P_r,V_r$ 是 P/V 发射值域上界。SA2++ 同时收窄两侧（$P_r=224,\ V_r=4.5$，
论文 Table 2 多组配置精度持平）。

**ffpa 对应**：`fp8_pv_acc_type='f16'` 的来源；但 **ffpa 的分配不同**
（`cute/fp8/smooth_v.cuh` 注释）：P 侧保持 448 满量程（$\tilde P=P\cdot v_s\cdot448$），
约束全压 V 侧——per-channel V + f16 acc 时发射 $\hat V\in[-2.25,2.25]$
（$448\times2.25=1008\le2047$ ✓）。见主文档 §11.6(d)。

---

## SA3（SageAttention3，arXiv 2505.11594）

**目标**：Blackwell FP4 attention（RTX5090 1038 TOPS，5x vs FA）+ 首个低比特训练
探索（SageBwd 8-bit，finetune 无损、pretrain 收敛慢）。

**FP4 推理三大技术**：

1. **(C1) FP4 值域仅 15 个码值** → **NVFP4 microscaling**：e2m1 数据 + ue4m3 scale
   factor，量化块 $1\times16$（MXFP4 的 ue8m0/32 块精度更差，弃用）。块内
   $s=\mathrm{amax}(16)/6$，outlier 影响被限制在块内。
2. **(C2) P∈[0,1] 压垮 SF 动态范围** → **两级 P 量化**：先按行归一到 [0, 448×6=2688]
   （fp32 全局 $s_{P1}$，折进 softmax exp2 shift 常数 $\log_2(1/2688)$），再做
   per-16 microscaling（ue4m3 $s_{P2}$ 铺满 e4m3 域）。
3. **Q/K smoothing 强制**（e2m1 ±6 动态范围太窄）+ rank-1 delta_s（承 SA2）；
   **K 列置换**（permutation for K）使 blockscale MMA 的 C 累加器布局直接匹配
   下一 GEMM 的 A 操作数布局，免 cross-lane reshuffle。

**ffpa 对应**（数据路径移植自 SA3 Blackwell kernel，fragment adapter 逐字拷贝）：
§6 全章、§11.10（两级 P 量化与 2688 域）、§11.11(b)（kv_perm32 + **perm-aware
masking**——上游 SA3 对原始列 index 做 mask 导致 causal 错误，ffpa 修复）、
§11.8(b)（delta_s 免物化恒等式）。fp4 lse 复合公式
`(m*L + log2(row_sum) + log2(1/2688))*ln2 + scale*qkm` 逐项对应 SA3 数值链。

---

## 历代"精度责任分配"总览（对齐主文档 §5.4/§11.7）

| 代 | QK | PV | P 量化 | smoothing | 误差主矛盾 |
|---|---|---|---|---|---|
| SA1 | int8 per-block | fp16 + f16 acc | 不量化（fp16） | K | K outlier |
| SA2 | int4 per-thread | fp8 + FP32 buffer | fp8（fp32 buffer 兜底） | Q(+K/V) | FP22 累加器 |
| SA2++ | 同 SA2 | fp8 + **f16 acc** | fp8（值域收窄） | 同 | f16 acc 值域 |
| SA3 | fp4 microscaling | fp4 blockscale | **两级 fp4（2688 域）** | Q/K 强制 | P 的 SF 动态范围 |

> **P 的 online 量化贯穿后三代且逐代加重**（fp8→两级 fp4），是量化 attention
> 精度损失的最主要根源之一；主文档 §5.4/§9.1 方向 9 把"针对 P online 量化的精度
> 优化"列为后续方向。

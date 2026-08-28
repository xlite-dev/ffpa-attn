# FlashAttention 1-4 核心技术原理

> 依据：skill 内论文文本 `references/papers/FlashAttention-{1,2,3,4}.txt`
> （FA-1 arXiv 2205.14135 / FA-2 arXiv 2307.08691 / FA-3 arXiv 2407.08608 /
> FA-4 arXiv 2603.05451）。每代标注 ffpa-attn CUDA backend 的对应实现
> （配合主文档 §11.1-§11.3、§5.7 与附录 A）。
> 一句话脉络：**FA-1 tiling + online softmax 奠基（IO-aware）→ FA-2 更好的并行与
> work partitioning、削减非 matmul FLOPs → FA-3 Hopper 异步化（warp specialization +
> ping-pong）+ FP8 量化 → FA-4 面向数据中心 Blackwell"不对称硬件扩展"的
> 算法-kernel 协同设计（条件 rescale、exp 分流、TMEM/2-CTA、LPT 调度）**。

---

## FA-1（Dao et al., NeurIPS 2022）

**核心思想**：IO-aware 精确 attention——GPU 上瓶颈是 HBM↔SRAM 访存而非 FLOPs；
标准 attention 把 $S$/$P$ 物化到 HBM（$O(N^2)$ 次访问）。

1. **tiling**：Q/K/V 切块载入 SRAM，内核内逐块完成 softmax 与两个 matmul，
   $S$/$P$ 从不物化到 HBM。
2. **online softmax**：运行最大值 $m$ 与归一化和 $l$ 的跨块递推，数学精确、
   非近似——所有现代 attention kernel 的骨架。
3. **IO 复杂度**（定理 1/2）：FlashAttention 为 $O(N^2d)$ FLOPs、$O(N)$ 额外内存，
   HBM 访问 $\Theta(N^2d^2/M)$（$M$ = SRAM 大小，标准实现为 $\Theta(Nd+N^2)$），
   且证明**任何精确 attention 算法都无法渐近优于该下界**。
4. **backward 重算**：只保存 O 与统计量，反向从 Q/K/V 重算 S/P，HBM 访问
   从 $O(N^2)$ 降回 $\Theta(N^2d^2/M)$。
5. causal：对角线以上的块整块跳过。

**论文实测**：BERT-large（512）端到端 +15%（对 MLPerf 1.1 训练纪录）、
GPT-2（1K）3x、Long-Range Arena（1K-4K）2.4x。

**ffpa 对应**：全部 kernel 的 tiling + online softmax 骨架（§11.1，log2 域实现）；
tile 级 causal 剪枝（§11.14 `mask_start_tile`）。ffpa 定位前向推理，无反向重算需求。

---

## FA-2（Dao, 2023）

**出发点**：FA-1 在 A100 仅达理论峰值的约 25-40%；根因是 threadblock/warp 间
**work partitioning 次优**（batch×heads 小时并行度不足、warp 划分引入
shared memory 读写与冗余非 matmul 操作）。

三项改进：

1. **削减非 matmul FLOPs**：A100 matmul 312 TFLOPs/s vs 非 matmul FP32
   19.5 TFLOPs/s（**单 FLOP 贵 16x**）。对策：
   - **推迟 rescale**：维护未归一 $\tilde O$（只做乘性更新），把
     $\mathrm{diag}(l)^{-1}$ 除法留到最末一次——每个 block 省一次除法；
   - 反向只需存 LSE（$L=m+\log l$），不必分别存 $m$ 与 $l$。
   这是后续"延迟 scale / 条件 rescale"一系思想的源头。
2. **序列长度维并行**：batch×head 并行度不足时按 Q 块再切 CTA。forward
   每 CTA 持一个 Q 块、内循环遍历 KV 块（相对 FA-1 减少非 matmul 指令与
   寄存器读写）。
3. **warp 划分**：forward 每 warp 持 Q 的行分片独立算——FA-1 中所有 warp
   需经 shared memory 归约中间结果，FA-2 消掉这一往返；backward 保持按
   序列长度并行，避免 split-K 的 shared memory 写。

**论文实测**：约 2x FA-1，A100 达理论峰值 50-73%（至 225 TFLOPs/s），
接近 GEMM 效率；GPT-3 175B 端到端 +5%（72% 模型 FLOPs 利用率）。

**ffpa 对应**：native sm80/sm120 kernel 的 split-D 结构（§3.3，
`kQKDChunk/kVDChunk` 双 GEMM 累加 + 在线 rescale）承 FA-2 谱系；
max-pass 延迟 scale（`kMaxScaleAfter`，§11.1）与 FA-2"推迟归一"同源；
split-KV decode 两阶段（§11.2）是序列维并行在 Nq=1 的特化。

---

## FA-3（Shah et al., 2024）——Hopper

**动机**：FA-2 在 H100 仅 35% 利用率——同步算法模型吃不满 Hopper 的异步硬件
（Tensor Core / TMA 独立单元）与低精度。三大技术：

1. **warp specialization（producer-consumer 异步）**：producer warpgroup 专职
   TMA 异步加载，consumer warpgroup 专职 WGMMA 计算，多级流水
   （计算与数据搬运全局重叠）。
2. **ping-pong 调度（softmax 藏进 GEMM）**：两个 consumer warpgroup 交替——
   warpgroup A 对 block j 做 softmax 时，warpgroup B 在异步 proxy 上算
   block j+1 的 QK WGMMA；PV 与 rescale 同样重叠。非 matmul 操作被
   塞进 matmul 的等待气泡。
3. **FP8 低精度**：
   - **block quantization**：比 per-tensor 细的块级 scale，限制 outlier 影响；
   - **incoherent processing**：随机正交（Hadamard 类）旋转把离群值能量摊平到
     所有通道，旋转在 $QK^\top$ 内正交消去（2.6x 更低误差 vs per-tensor 基线）；
   - **asymmetric quantization**：利用 $P\ge0$、V 列分布做非对称量化；
   - WGMMA 布局适配：in-kernel transpose（ldmatrix/stmatrix）而非 shuffle。

**论文实测**：FP16 740 TFLOPs（75% 利用率）、FP8 ~1.2 PFLOPs（H100 首次破
PFLOPs，比 FA-2 快 1.5-2.0x）。

**ffpa 对应**：
- fp8/fp4 persist-D 的 **1 producer(128T TMA) + 1 consumer(256T)** WS 结构
  即 (1) 的 sm_120 形态（§5.7）；但 **FA-3 式双 consumer（2×128T）在 sm_120
  已证伪**（附录 A #1：GB202 无 WGMMA，ping-pong 的异步前提不成立）。
- (3) 的 block quantization ↔ per-block/per-thread scale（§11.5）；
  incoherent processing ↔ `fp8_hadamard`/`fp4_hadamard` knob（§11.9）；
  布局适配思想 ↔ reorg-free PV pack 与 kv_perm32（§11.11，数学解法替代搬运）。

---

## FA-4（Zadouri, Hoehnerbach, Shah et al., 2026）——数据中心 Blackwell

**主题：不对称硬件扩展（asymmetric hardware scaling）下的算法-kernel 协同设计**。
B200 tensor core 吞吐翻倍（BF16 2.25 vs Hopper 1 PFLOPS），但 SMEM 带宽
（128 B/clk/SM）、MUFU exp（16 ops/clk/SM）、整数/浮点 ALU 基本不涨。
roofline 分析：典型 attention 上 **SMEM 流量与 exp 运算占主导，超出 MMA 计算
25-60%**——瓶颈从 MMA 转向非 MMA 单元。

**硬件边界（重要）**：FA-4 目标为 sm_100 数据中心 Blackwell（B200/GB200），
依赖 **TMEM（每 SM 256KB tensor memory）、tcgen05 全异步 MMA（128×128 tile、
输出直写 TMEM）、2-CTA MMA 模式**——这些 sm_120（消费级 Blackwell，只有
warp 级 mma.sync）一概没有；但**算法层技术**（条件 rescale、exp 分流、
LPT 调度）与架构无关、可移植。另注：FA-3 无法在 B200 运行（Hopper MMA
指令无前向兼容）。

四大支柱：

1. **流水重组（最大化 overlap）**：延续 FA-3 ping-pong，两个 Q tile（H/L，
   各 128 行）累加器驻 TMEM；两个 softmax warpgroup（128 线程、每线程独占
   一整行 → 免 warp 间 row-max shuffle）；P 经 TMEM 传递（不经寄存器）使
   O rescale 解耦成独立的 **correction warpgroup**、移出关键路径；P 分段
   store（先 3/4 后 1/4）控制寄存器压力；TMEM 分区让 S 与 P 复用同一块、
   流水线开局即可算两个 S tile。
2. **exp 单元瓶颈缓解（双管齐下）**：
   - **软件 exp 仿真**：$2^x = 2^{\lfloor x\rfloor}\cdot 2^{x-\lfloor x\rfloor}$
     （Cody-Waite range reduction）——整数部分走指数位整数运算，小数部分
     （区间 [0,1)）用 FMA 单元上的多项式（Horner）求值，与 MUFU 并行。
     3 次多项式 FP32 max rel err 8.8e-5（约硬件 600x），但**舍入到 BF16 后
     误差与硬件不可区分**（BF16 量化误差 3.9e-3 主导）→ 3 阶足够。
     **部分仿真**：仅 10-25% 元素走 FMA 路径（比例按 MMA/exp 吞吐比调），
     其余走硬件 MUFU.EX2，避免全仿真的寄存器压力反噬。
   - **条件（lazy）softmax rescale**：$m_j - m_{j-1} \le \tau$ 时跳过 rescale
     （**$\tau=\log_2 256 = 8.0$**，即膨胀因子 256 内容忍），exp 继续用旧
     max $m_{j-1}$；末尾用真实 $m_{final}, l_{final}$ 统一归一恢复精确；
     warp 级 vote（任一线程需 rescale 则整 warp 做）避免 warp divergence。
3. **backward SMEM 流量削减**：TMEM 复用（S/P 共享块 0；dP/dS/dQ 共享块 1——
   最多容 4 个 128×128 累加 tile，且 dV/dK 是累加器不可共享）；**2-CTA MMA
   模式**（M=256）：CTA pair 合成大 tile、每 CTA 只 stage 半个 operand B →
   B 的 SMEM 流量约减半；dQ 的归约轴 N 恰好被 pair 拆分 → 用 DSMEM 交换
   半个 dS、每 CTA 持 M/2 行做满 2N 归约，**dQ 全局 atomic 归约减半**。
   确定性 backward：semaphore 串行化 + SPT 排序（causal 下 KV 降序、Q 自对角线
   升序、dQ 归约按 Q 块降序），达非确定性版 75% 速度。
4. **LPT 调度（longest-processing-time-first）**：causal/varlen 天然负载不均。
   causal：batch 最外层、heads 按 L2 容量分段 swizzle、**mblocks 逆序**
   （长 worktile 先做）；MQA/GQA 先遍历同一 KV head 的所有 query head。
   varlen：预处理 kernel 按每 batch 最大 worktile 时间排序（元数据可缓存，
   排序零开销）。**跨架构有效**（H200 实测 +4-8% MHA / +7-14% MQA-8，
   在 FA-3/Hopper 上同样验证过）。

**工程形态**：全 CuTe-DSL（Python embedded）实现、**零 CUDA C++**；JIT 编译
比 FA-3 的 C++ 模板快 20-30x（fwd 55s→2.5s、bwd 45s→1.4s）；FlexAttention /
block-sparse 等变体可在不改核心框架的前提下搭建。

**论文实测**：B200 BF16 至 1613 TFLOPs/s（**71% 利用率**），1.1-1.3x vs
cuDNN 9.13、2.1-2.7x vs Triton；causal 增益更大（LPT 贡献）。

**ffpa 对应与边界**：
- **条件 rescale ↔ ffpa lazy rescale**（§11.1）：思想相同，但参数来源不同——
  FA-4 论文取 $\tau=8$（BF16）；ffpa fp8 取 `FFPA_RESCALE_THRESHOLD_FP8=4`，
  是 e4m3 发射域约束（$2^T\cdot\mathrm{amax}(V)\le448$，§11.6(c)）反推的更紧
  选择，**非论文参数**。ffpa 实现：per-row 判定（fp8 persist-D）与 warp-vote
  （fp4，dense 命中率 96.5%、-4.9%，§6.6），warp-vote 与 FA-4 同构。
  证伪边界：per-row P quant + 重开 lazy rescale 组合已证伪（附录 A #12：
  满量程零 headroom）——条件 rescale 只在固定发射域下成立。
- **exp 多项式仿真 ↔ ffpa 附录 A #17 证伪**：fp4 exp2 多项式替换在 sm_120
  被否决（XU 28% 利用率非瓶颈、串行 FFMA 更糟）。与 FA-4 并不矛盾——
  FA-4 有效的前提是 B200 上 MUFU 被 roofline 证明为瓶颈；**教训：先证明
  exp 是瓶颈（ncu XU/MUFU 利用率）再上仿真**。（B300 已把 exp 吞吐翻倍到
  32 ops/clk/SM，该瓶颈未来会再移动。）
- **TMEM/tcgen05/2-CTA 为 sm_100 专属**，不适用 sm_120（ffpa 目标硬件）。
- **LPT 调度**：架构中立，若未来 ffpa 面向 causal/varlen 负载不均场景，
  是可借鉴的 CTA 调度方向（当前未见瓶颈）。

---

## 技术传承速查（FA 系 → ffpa）

| FA 技术 | 代际 | ffpa 落点 | 报告章节 |
|---|---|---|---|
| tiling + online softmax | FA-1 | 全 kernel 骨架（log2 域） | §11.1 |
| IO 下界 $\Theta(N^2d^2/M)$ | FA-1 | 设计前提（访存瓶颈论） | §11.1 |
| 序列维并行 / split-KV | FA-2 | native decode 两阶段 | §11.2 |
| 推迟归一（延迟 rescale） | FA-2 | `kMaxScaleAfter`、lazy rescale | §11.1 |
| split-D（D 维分块） | FA-2 谱系 | cute 三族大 D 路径 | §11.3 |
| warp specialization | FA-3 | fp8/fp4 persist-D（1P+1C） | §5.7 |
| ping-pong 双 consumer | FA-3 | **sm_120 证伪**（无 WGMMA） | 附录 A #1 |
| block quantization / incoherent processing | FA-3 | per-block/per-thread scale；hadamard knob | §11.5、§11.9 |
| 条件 rescale | FA-4 | T=4（发射域反推）+ warp-vote | §11.1、§6.6 |
| exp 软件仿真 | FA-4 | sm_120 证伪（XU 非瓶颈） | 附录 A #17 |
| LPT causal/varlen 调度 | FA-4 | 未采用（候选方向） | — |

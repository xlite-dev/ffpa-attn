# ffpa-attn CUDA Backend 未来优化 RFC（功能完备性优先）

> 状态：Draft ｜ 日期：2026-08-28 ｜ 关联文档：[SKILL.md](../SKILL.md)（特性现状与数学原理，下称"报告"）
> 本稿按"**功能完备性 > 性能优化**"两条轨道组织。功能完备性轨道解锁"现在做不了/被物化降级/直接报错"的能力；性能轨道在功能不变前提下提速。
> 动手前**必读附录 A（已证伪清单）**；所有性能类收益数字按报告 §2.2 纪律标注卡型/冷热条件。
> **约定**：本文所有代码路径均相对于 **ffpa-attn 仓库根目录**（如 `csrc/cuffpa/cute/launch.cuh`）。实施完成状态统一记录在两处：**完成状态清单**（下方，GitHub 上可直接勾选）与**总览表状态列**；各条目 `Status: Draft` 仅表示设计稿状态。每做完一项，同步勾选清单 + 更新总览表状态列（⬜ 待开始 → 🚧 进行中 → ✅ 已完成）。
> **子项编号**：任一大 FC/PC 可按场景拆挂子项，编号为 `父-子`（如 `PC-0-0`），子项在总览表与完成状态清单中缩进列于父项之下，可独立推进/验收/勾选；父项状态 = 全部子项完成后方可置 ✅（部分完成时父项标 🚧 并括注进度，如 `🚧 1/3`）。

## RFC实现规范（⚠️ 强制约束）

- 1. 每项动手前：先在 plan 模式（Copilot下要切换到plan agent）完成实施规划（改动面 / 注入点 / 验证矩阵），规划好再动手 (自动模式下可以按照规划继续实施操作)。
- 2. 每做完一项：勾选对应条目（`- [ ]` → `- [x]`），并同步更新上方总览表状态列。注意，要同时更新[SKILL.md](../SKILL.md) 中的技术报告和本文档的总览表状态列，确保两处状态一致。
- 3. **验收必须落实到 `ffpa_attn.bench` CLI**：`python -m ffpa_attn.bench` 全链路（`--fwd-backend cuda --cuda-impl <impl> --tasks <相关task>`）跑通并通过 parity。`tests/` 与临时脚本仅是开发阶段验证，**不能替代 CLI 验收**；若该项能力影响 bench task 集合（新增/排除 task），须同步放开/过滤 bench CLI 的 task 并在 CLI 输出中可见。

## 0. 功能全景对比：ffpa fp8/fp4 vs SageAttention-2/3（sm_120）

> 2026-09-02 依据本地源码核对（SageAttention `sageattention/core.py`、`sageattention3_blackwell/sageattn3/api.py` + `blackwell/static_switch.h`；ffpa `ffpa_api.cc`）。sage3 在 sm_120 可用（warp 级 `mma.sync kind::mxf4nvf4`，无 TMEM/tcgen05 依赖）。
> 总括：**ffpa 接口对齐 SDPA（除 dropout），场景/维度覆盖为 sage-2/3 的超集**；sage 系列以场景收窄换 kernel 极简与峰值吞吐。

| 能力 | ffpa fp8 | ffpa fp4 | SageAttention-2 | SageAttention-3 |
|---|---|---|---|---|
| self / cross-attn | ✓ | ✓ | ✓ | ✓ |
| causal | ✓ | ✓ | ✓ | ✓ |
| GQA | ✓ | ✓ | ✓（Hq 需被 Hkv 整除） | **✗**（无任何 Hq≠Hkv 处理） |
| attn_mask（bias 注入） | ✓ 全族（FC-4 + PC-0 tile 化；rowvec/dense/折叠 6 形态） | ✓（同左；m4n2 已知 race 见 PC-0-5，影响受控） | **✗**（cuda kernel 无注入） | **✗**（签名有 `attn_mask` 但函数体完全忽略——伪参数，传入即静默丢 mask） |
| dropout | ✗（量化族不支持；fp16/native ✓） | ✗ | ✗ | ✗ |
| return_lse | ✓ | ✓ | ✓ | ✗ |
| head_dim | **64-1024 任意 D%8==0**（kernel 内零物化 pad 到 64 倍数实例） | 同左（pad 后 ∈[64,1024]） | **≤128**（<64 pad 64；(64,128) pad 128；>128 raise） | **仅 64/128**（`HEADDIM_SWITCH` 有 64/128/256 实例，但入口 `D≥256` fallback SDPA、非 64/128 倍数无处理） |
| 输入布局 | BHND packed / strided fused-QKV / NHD O 写**全零拷贝**（FC-1/2/3） | 同左 | HND/NHD 双布局（要求 `stride(-1)==1`） | 仅 HND 4D |
| QK 精度 | **FP8 或 INT8**（`fp8_qk_mm_type`）+ per-block/per-thread 量化粒度 | NVFP4（1×16 + E4M3 SF） | INT8 per_warp/per_thread（**sm_120 仅 per_warp**——per_thread 是 triton 路径，sm120 不可用；论文 int4 QK 未开源） | NVFP4 固定 |
| PV 精度 | FP8 E4M3，累加器 FP16/FP32 可选 | NVFP4（或 MXFP8 PV） | FP8 E4M3（acc `fp32+fp16`/`fp32+fp32`）或 FP16 PV | NVFP4 固定 |
| smoothing / 数值 knobs | smooth_k / smooth_v / hadamard 独立开关 | smooth_v / hadamard | smooth_k 内置 + smooth_v 可选 | K 减均值 + Q per-block 均值（`per_block_mean`），无其它 knob |
| hybrid（前缀 fp16 + 量化主体，causal 短行精度） | ✓（fp8/fp4_hybrid） | ✓ | ✗ | ✗ |
| backward | ✗（FC-9 设计稿） | ✗ | ✗（SageBwd 论文有、未开源） | ✗ |
| seq len 约束 | 任意（non-aligned task 覆盖 Nkv 非对齐） | 同左 | 任意 | N pad 至 128 倍数（preprocess 内） |

> 读法：sage-2/3 用"场景子集 + 固定量化配方"换 kernel 简洁与峰值 TOPS（sage3 论文 1038 TOPS @RTX5090）；ffpa 走 SDPA 全接口兼容路线——attn_mask/GQA/任意 %8 headdim/布局零拷贝/hybrid 均为量化路径保留，代价是 kernel 家族多（persist_d/split_d/m4n2 × fp8/fp4）与 knobs 维护面。下游（diffusers/cache-dit）替换 SDPA 时 ffpa 可直接物化，sage 需要接口降级或 fallback 混跑。

## 优先级框架

两大轨道，**轨道 F 整体优先于轨道 P**；轨道内按依赖与投入产出排序：

- **轨道 F（功能完备性，最高优先级）**：解除"用户想做但做不了"的硬限制。判定标准：完成后解锁一类**此前不可用、或必须物化降级**的用法。子轨道：
  - **F1 布局零拷贝闭环**（最高）——直接服务下游（diffusers / cache-dit）最高频形态；
  - **F2 量化路径特性对齐**——让低精度家族获得与 fp16 家族同等的输入特性（`attn_bias` 是最大高频需求）；
  - **F3 场景与硬件覆盖**——decode / pad / 多架构，按需。
- **轨道 P（性能优化）**：功能不变下提速——前处理融合（Mega Quantize Kernel）、kernel 内部、配置自适应、graph 友好。

## 功能完备性现状矩阵

`✓` = 已支持 ｜ `✗` = 缺口（列对应缺口编号）｜ `—` = 不适用：

| 能力 | native | fp16 cute | fp8 cute | fp4 cute | 缺口 |
|---|---|---|---|---|---|
| `attn_bias` | ✓ | ✓ | 全族（FC-4） | 全族（FC-4） | FC-4 ✅ |
| `dropout` | ✓ | ✓ | ✗ | ✗ | FC-5 |
| `tensor_layout='NHD'` O 写 | ✗ | persist-D | 全族（FC-2） | 全族（FC-2） | 全族（FC-2） | FC-2 ✅ |
| strided-NHD 读（fused-QKV） | ✗ | persist-D | 全族（FC-1） | 全族（FC-1） | 全族（FC-1） | FC-1 ✅ |
| strided/NHD + hybrid 组合 | — | — | ✓ | ✓ | FC-3 ✅ |
| smooth_v / MXFP8-PV knob | — | — | ✓ | smooth_v 全族；MXFP8-PV persist-D+split-D（m4n2 架构排除） | FC-6 ✅ |
| head_dim pad | ✓（FC-8） | ✓ | ✓ | ✓ | FC-8 ✅ |
| decode / 短 Nq 量化 | ✗（无量化） | ✗ | ✗ | ✗ | FC-7 |
| backward | ✗ | ✗ | ✗ | ✗ | FC-9 |
| sm90 / sm100 | 部分（fp16 TMA） | ✗ | ✗（fp8 限 sm_120） | ✗（fp4 限 sm_120） | FC-10 |

> 读法：列方向看某个量化家族缺什么；行方向看某能力在哪些家族缺口。persist-D 三族功能最全，**所有大 D（超出 persist-D 上限）场景目前被上述 ✗ 卡住**。
>
> **正确性现状（PC-0-5 止血，2026-09-04）**：**native / fp16 全族 / fp8 六族 / fp4 split_d / fp4 persist_d 的 bias 路径全部 bitwise 稳定**（persist_d D=256/D=128 各 0/30 实测，`.tmp/pc5-race/m4n2_persistd.py`）；**唯一 PC-0-5 问题 = fp4 split_d_m4n2 + attn_bias**——2026-09-04 重启深挖证实 pure bias（无 prelude）在 mode 2/3 下 **100% 触发**（推翻 09-02"纯 bias 稳定"定性），指纹恒定为单个 (m-warp, n-warp, v-chunk) PV C tile；**修复 = launcher pin mode 0（gmem 直读）**，pure 序列 10/10 稳定（代价 attn-mask ~5%）；**残留（已接受）**：重负载前置（任意 GPU 工作）下 bias 模板仍低概率不稳（mode 0 亦然，硬件负载时序层，no-bias 模板同负载干净），fp4 m4n2 仅服务 D≥768 fp4、场景少，待 NVIDIA 上报。`FFPA_BIAS_TILE_KEEP=1` 可恢复 mode 2/3。详见完成清单 PC-0-5。**注意区分**：fp4 persist_d 另有一处独立的低概率（3/30）epilogue race（先于 PC-0-1 存在、非 PC-0-5），persist_d 无 mode 0 等价路径，需独立排查；另发现 m4n2 **Nq=64（MB=1）illegal access 独立 bug 待修**。

## RFC 总览（按优先级）

| 编号 | 标题 | 轨道 | 状态 | 依赖 |
|---|---|---|---|---|
| FC-1 | split-D/M4N2 独立 Lv（strided-NHD 读） | F1 | ✅ 已完成（ffpa-attn cc8e8dc/4a49d38/882ee07） | — |
| FC-2 | split-D/M4N2 NHD O 写 | F1 | ✅ 已完成（ffpa-attn df7d572/c4ca38b/2382ca4） | FC-1 热身 |
| FC-3 | split-D/M4N2 + hybrid strided 组合 | F1 | ✅ 已完成（ffpa-attn 9b9dcae） | FC-1 |
| FC-4 | 量化路径 `attn_bias` | F2 | ✅ 已完成 | — |
| FC-5 | 量化路径 `dropout` (**暂不实施，仅保留设计稿**) | F2 | ⬜ 待开始 | FC-4 注入点 |
| FC-6 | fp4 smooth_v/MXFP8-PV 扩展至三族 | F2 | ✅ 已完成（ffpa-attn 76a8bd8） | FC-1/FC-2 |
| FC-7 | 短 Nq/decode 量化路径 (**暂不实施，仅保留设计稿**) | F3 | ⬜ 待开始 | — |
| FC-8 | native head_dim pad | F3 | ✅ 已完成 | — |
| FC-9 | CUDA backward (**暂不实施，仅保留设计稿**) | F3 | ⬜ 待开始 | — |
| FC-10 | sm90/sm100 量化覆盖 (**暂不实施，仅保留设计稿**) | F3 | ⬜ 待开始 | — |
| FC-11 | native 路径 dropout 精度修复（bug，高优） | F3 | ✅ 已完成（ffpa-attn 542f774/e1fe363，根因=torch ref uint32 bug） | — |
| FC-12 | cute sm_80 家族补齐 persist-D / split-D M4N2 (**暂不实施，仅保留设计稿**) | F3 | ⬜ 待开始 | 与 PC-12 同路线（sm_80 cp.async） |
| PC-0 | attn mask 场景性能优化（bias tile IO 重构） | P | ✅ 主体完成（P 轨三子项 PC-0-0/0-1/0-2 落地；PC-0-3 证伪关闭=结构极限定论；PC-0-4/0-5 P3 搁置） | FC-4 注入点 |
| PC-0-0 | ↳ cute/cute_tma 场景（fp16 cute 家族） | P | ✅ 完成（b4a811e + 7ffe765/1e4d9b6 迭代：bench CLI D=128 gap 1.12/89%、D=768 1.07/93% 双达标，原记录 0.99 系测量异常已修正；D=320 1.44/70% 结构极限未达 → **PC-0-3 专项**；2026-08-31 A0 补丁修复 mode 2/3 (b,h) 折叠缺陷 + sm_80 dense 平方 bug；2026-09-02 D=64 dense 拆段 TMA 补强：tile 超出 Q 复用区时按 Q 容量拆多段 TMA（前段 Q 区 + 尾段 extra 区，单 mbarrier expect_tx 总账），fp16 mask 1.34x vs gmem、fp32 超预算自动降级，见完成清单） | — |
| PC-0-1 | ↳ fp8/fp4 场景（量化六族，原 PC-0 主体） | P | ✅ 完成（17ac22f A0 → 16eaea7/39c63ea/f42b12a/f12406f/b194bec/c2fc67d B1-B6 → 7d5ca4c C 阶段：mode 3 全驻留为主力，fp8 D=128 1.85x、fp4 D=320 1.67x、D=768 1.04x；fp8 split_d D≥512 demote mode 0 → **PC-0-4 专项**；先在 race → **PC-0-5**，见完成清单） | FC-4 注入点；PC-0-0 热身 |
| PC-0-6 | ↳ D=768 split-d vs m4n2 A/B + 寄存器模型分析 | P | ✅ 完成（2026-09-02：fp4 split_d D=768 因 O staging 196608B>101376B smem 预算**物理不可行**，m4n2 是唯一可行 kernel；fp8 A/B 实证 m4n2 五场景全胜，attn-mask split_d 崩溃 3.15x 差距；寄存器实证量化路径 O regs/线程 split_d=D/2=384 重 spill vs m4n2=D/4=192 近零 spill，fp4 m4n2 有量化状态中等 spill 528-920B → 量化寄存器模型不等同 fp16，m4n2 在大 D 必需非可选） | — |
| PC-0-2 | ↳ native/native_tma 场景 | P | ✅ 完成（c645ac1：launcher 门控 rowvec 快路径——stride_m==0/Nq==1 + stride_n==1 + pair 对齐时半精度 `half2`/fp32 `float2` 对加载，1 次对加载服务 4 个 fragment 槽（原 2048 loads/warp/tile 冗余 16x）；NCU 定性指令数瓶颈（非带宽）；11/11 bitwise；验收 tma D512 2.00x / native D512 1.89x / native D128 1.67x（vs SDPA），N=16384 无退化；残余为 load-latency 主导，地址提升无增益，预取方案受 255 寄存器预算约束搁置，见完成清单） | — |
| PC-0-3 | ↳ D=320 split_d 注入开销专项（PC-0-0 遗留） | P | ❌ 证伪关闭（gap 1.44=结构极限：杠杆①向量化 +2.7%、杠杆②softmax 融合 87.7 vs 83.6ms 双证伪；加载侧 smem prefetch 本体收益 45ms/35% 已在 PC-0-0 落地，见完成清单） | PC-0-0 |
| PC-0-4 | ↳ fp8 split_d D≥512 attn-mask tile 化专项（PC-0-1 B-3 遗留） | P3 | ⬜ 待开始（当前 D≥512 已 demote mode 0 保底，gap 停留 gmem 基线 1.48x，见完成清单） | PC-0-1 |
| PC-0-5 | ↳ 量化家族先在时序竞争排查（PC-0-1 B-4/B-6 实证，bitwise 断言 flaky 根因） | P3 | 🚧 **止血落地、残留接受**（2026-09-04 重启深挖：**pure bias（无 prelude）100% 触发**（10/10 全 6/6，推翻 C9a"稳定"；918ae3c squash 终态即存在，build41 重建实证）；指纹恒定 = **单个 (m-warp0, n-warp0, v-chunk3) PV C tile**（d 恒 [192,224) 32 列 = 一个 SM120_16x32x64_TN_VS_NVFP4 atom 的 C 宽 = 一个 uint4 B 寄存器载荷），坏行限定每 64 行 tile 的 rows 0-15（warp_id%4==0 行组），lse 稳定、diff ~0.8% 元素 med 0.002/max ~0.039（漏/多一个 tile 贡献项）；维度窗口 = **works×tiles 乘积**（2×32 或 32×2 稳定、32×32 轻触发、128×32 全触发）；协议全文复审 + `mma.sync`（SM120 f8f6f4 为同步 warp mma）语义级闭环，**ptxas -O2 证伪编译调度**、**producer 挪位（tid0→255）证伪 producer×consumer 交织**（race 存活且条纹相位漂移 +12 → 坏块跟随 warp×chunk 相对时序而非绝对 warp id）；**决定性反转：重负载 prelude（任意 matmul×9）下 mode 0 gmem 直读也触发**（6-12/24）且指纹同源 → **历史"mode 0 完全免疫"结论修正**，race = **bias 模板 × 负载时序窗口**的硬件层问题（no-bias 模板同负载 0/24 干净）；**修复**：launcher pin mode 0（堵住 pure 冷启动 100% 入口，10/10 0/6；代价 attn-mask ~5%：397.9→419.1ms，其余 task 零回归）+ `FFPA_BIAS_TILE_KEEP=1` 逃生口 + 测试分类（pure 序列转正必过 gate / prelude 序列 xfail 残留）+ **残留接受**（重负载下 bias 模板低概率不稳，m4n2 仅服务 D≥768 fp4 现实场景少，待 NVIDIA 上报，复现器 `.tmp/pc5-race/m4n2_stat.py` 等 8 脚本）；顺带发现 **Nq=64（MB=1）illegal access 独立 bug 待修**；fp16/fp8/fp4 split_d/persist_d 全路径保持干净 | PC-0-1；PC-11 解耦可独立推进 |
| PC-1 | Mega Quantize Kernel（aux 链大融合） | P | ⬜ 待开始 | — |
| PC-2 | 增量融合（Mega Kernel 步进） | P | ⬜ 待开始 | 被 PC-1 收编 |
| PC-3 | N-crossover 量化配置自适应 | P | ⬜ 待开始 | — |
| PC-4 | fp4 persist-D attn kernel 内部优化 | P | ⬜ 待开始 | — |
| PC-5 | CUDA graph 友好化 (**暂不实施，仅保留设计稿**) | P | ⬜ 待开始 | PC-1 评估 |
| PC-6 | sm_89 fp8 int4 QK (**暂不实施，仅保留设计稿**) | P | ⬜ 低优搁置 | PC-12（cute sm_80 fp16 性能达标 → 迁移 cute/fp8/sm_89 即 fp8 路线复活） |
| PC-7 | fp8 split-D (M8N1) 量化大 D kernel 性能优化 | P | ⬜ 待开始 | — |
| PC-8 | fp8 split-D M4N2 量化大 D kernel 性能优化 | P | ⬜ 待开始 | PC-7（顺序） |
| PC-9 | fp4 split-D (M8N1) 量化大 D kernel 性能优化 | P | ⬜ 待开始 | PC-8（顺序） |
| PC-10 | fp4 split-D M4N2 量化大 D kernel 性能优化 | P | ⬜ 待开始 | PC-9（顺序） |
| PC-11 | warp 级 `__any_sync` lazy-rescale 统一治理（精度治理专项） | P | ⬜ 待开始 | 已与 PC-0-5 解耦可独立推进（vote 非本次 race 根因——force-rescale 实证；治理价值在消除 warp-uniform 分支的调度脆弱性） |
| PC-12 | cute sm_80 fp16 性能优化（cp.async + 多级流水线，fp8/sm_89 路线前置） | P | ⬜ 待开始 | —；被 PC-6 依赖 |
| PC-13 | fp8/fp4 hybrid 路径性能优化（双 attn kernel → 融合 kernel） | P | ⬜ 待开始 | —（与 PC-7~10 协同） |
| PC-14 | fp16 dropout 路径性能优化（RNG bitmap 预计算 + producer/consumer 重排） | P | ✅ 完成（consumer 侧双缓冲 bitmap：persist_d D=64 1.02x / D=128 2.25x，split_d D=320 2.05x，sm_80 split_d 完成（f158eb1），bitwise 全过 + 全 task 套件零回归（fp16 7 tasks×2 dtypes + fp8/fp4 smoke）；producer 方案证伪；**m4n2 证伪不实现**（D=768 bitmap 212.82ms 反慢于 inline 202.31ms，tile 小 + PV/exchange 主导，RNG 非瓶颈；未来有需求再评估）；**persist-D half-row 方案要求 kBc≥64**——D=192/256 的 kBc=32 实例化编译期 `kBitmapCapable` 门控回落 inline Philox（d6a4a1d）；RNG 指令地板结论见 SKILL §11.16——契约下上限约 1.2x） | PC-0 同构（bias tile 协议复用）；FC-5 是量化路径功能项（⏸），与本项无重叠 |

> 未收录项：分卡基准标注（文档规范，随下次 bench 执行）。（原列于此的 cache-dit `_keep_or_pack` 物化兜底移除已于 2026-08-28 完成，cache-dit@4b5c977：三 tensor 直传零拷贝，契约外布局由 C++ layout gate 显式报错。）

## 完成状态清单（备忘录）

> 每项动手前：先在 plan 模式（Copilot下要切换到plan agent）完成实施规划（改动面 / 注入点 / 验证矩阵），规划好再动手 (自动模式下可以按照规划继续实施操作)。
> 每做完一项：勾选对应条目（`- [ ]` → `- [x]`），并同步更新上方总览表状态列。

**轨道 F（功能完备性，最高优先级）**

- [x] FC-1：split-D/M4N2 独立 Lv（strided-NHD 读）—— F1 基建第一步（2026-08-28 完成）
- [x] FC-2：split-D/M4N2 NHD O 写 —— F1 基建第二步（2026-08-28 完成）
- [x] FC-3：split-D/M4N2 + hybrid strided 组合 —— F1 布局闭环收尾（2026-08-28 完成）
- [x] FC-4：量化路径 `attn_bias`（S/P 域注入基建）—— fp8/fp4 六族 kernel raw-S 域注入 + FfpaBiasParams helper（2026-08-28 完成）
- [ ] FC-5：量化路径 `dropout`
- [x] FC-6：fp4 smooth_v/MXFP8-PV 扩展至三族 —— smooth_v 全族；MXFP8-PV 至 split-D（2026-08-28 完成）
- [ ] FC-7：短 Nq/decode 量化路径 ⏸（暂不实施，仅保留设计稿）
- [x] FC-8：native head_dim pad —— kernel 侧 d_og 零物化 pad（cp.async src-size 列守卫 / TMA OOB 零填充），AUTO/NATIVE/TMA 三 hint 64 对齐（2026-08-31 完成）
- [ ] FC-9：CUDA backward（定位评估）
- [ ] FC-10：sm90/sm100 量化覆盖
- [x] FC-11：native 路径 dropout 精度修复 —— 存量 bug（2026-08-31 记录）；已结案（同日）：实为 stale `.so` + Triton int32 回绕 + torch 2.11 mem-eff ref 自身 uint32 回绕（PyTorch main 已修），FFPA 源码本身正确
- [ ] FC-12：cute sm_80 家族补齐 persist-D / split-D M4N2 ⏸（暂不实施，仅保留设计稿）
  - 现状：sm_80 cute 只有 split_d M8N1（`cute/sm_80/split_d.cuh`，cp.async loader，CUTE hint 分发）；对照 sm_120 家族缺两个成员——persist-D（D≤128 小 D，Q 驻留，寄存器/带宽模型最优）与 split-D M4N2（D≥768，kBr=64 + (4,2,1) atom 解 O regs=D/2 撞 255 墙，sm_80 版可复用 sm_120 m4n2 的几何但 loader 全换 cp.async，无 TMA/async proxy）。
  - 价值：无 TMA 硬件（sm_80/89）的 cute 家族 D 维全覆盖（小 D persist-D / 中 D split-D M8N1 / 大 D M4N2）；与 PC-12 同路线，PC-12 的 cp.async 多级流水线经验直接复用。
  - 触发条件：出现真实 sm_80/89 fp16 部署需求，或 PC-12 达标后随 fp8/sm_89 路线一并补齐。

**轨道 P（性能优化）**

- [ ] PC-0：attn mask 场景性能优化 —— bias tile IO 重构（P 轨最高优先，2026-08-31 立项）
  - [x] PC-0-0：cute/cute_tma 场景（fp16 cute 家族 `apply_attn_bias_rowcol` smem tile 化）—— 2026-08-31 完成（b4a811e）：TMA tile 预取 + persist_d Q-s2r 寄存器持久化腾出 Q smem 给 bias tile（fp8 kPersistQs2r 模式移植）+ split_d/m4n2 单缓冲防自锁 + dense 平面行坐标修复（元素 stride 误作行单位，h≥1 bias 被 TMA OOB zero-fill）；99 parity 用例 + bench CLI 验收：D=128 gap 1.12x（TFLOPS 89%）、D=768 gap 1.07x（93%）双达标（原记录 0.99x/101% 系测量异常：2026-08-31 A0 前后同口径复测均为 ~1.07-1.08，gap<1 本身反常）；dense 全量 mask 下 D=128 增量已达 HBM 带宽极限（~2.2TB/s 等效）。后续迭代 7ffe765（rowvec 双缓冲）+ 1e4d9b6（mode 3 全驻留）：D=320 gap 2.21→1.44（TFLOPS 66%→70%）仍未达 1.2x，属 split_d 结构极限 → 移交 PC-0-3 专项。A0 补丁（2026-08-31，PC-0-1 规划审查发现）：mode 2/3 的 (b,h) 未折叠——head-key/batch-key rowvec mask（stride_m 被 size-1 归零而 stride_b/h≠0）全部 CTA 读 (0,0) 行，h≥1 输出错但 3e-2 断言假阴性掩盖（repro 差 0.0097）；修复 = 分类器补 rowvec 平面校验（stride_h∈{0,Nkv}、stride_b∈{0,h_eff·Nkv}、Nkv·elem 16B 对齐）+ TMA 坐标/mode 3 装载 fold + sm_80 loader 统一行坐标公式（顺带修 dense 元素偏移×stride_m 平方 bug，sm_120 GPU 上从未暴露）；新增 15 例 tile-vs-gmem bitwise 对照（storage-offset 视图强制 mode 0，`buf[...,:N]` 左切片不改变 ptr/stride 无效）。**2026-09-02 补验**：CUTE hint（非 TMA，`launch_cute_fwd_split_d_sm80` cp.async loader，CUTE=3）attn-mask 路径此前从未被 bench/parity 覆盖（全部验收走 CUTE_TMA），补测通过——D=128/768 parity abs≤1.3e-5（含 head-key `[1,H,1,N]`/batch-head `[B,H,1,N]` 折叠变体，A0 的 sm_80 loader 行坐标修复实证正确）、bitwise 稳定 0/8、vs SDPA 1.94-2.12x（比 CUTE_TMA 慢 7%@D=768 / 28%@D=128，小 D cp.async loader 开销占比大；复现脚本 `.tmp/pc5-race/fp16_cute_nontma_mask.py`）。**2026-09-02 拆段 TMA 补强（F1 后续）**：D=64 dense tile（fp16 32KB/fp32 64KB）超 Q 复用区（16KB）时不再整体降级 gmem——kernel 按 Q 容量拆多段 TMA（box 高度 `kQPersistU16/bias_cols`，前段落 Q 区、尾段落 K/V 后 extra 区，共享单 mbarrier expect_tx 总账；消费函数加默认尾参 `bias_smem2/split_elems` 分段读，其余 5 调用点零改动）；fp16 mask 1.34x vs gmem（0.229 vs 0.306ms，B1H4N4096）、fp32 超总预算（80+48>99KB）自动降级 gmem；**账目修复**：budget 降级必须同步清零 `bias_extra`（否则 setAttribute 超限 → launch 报 invalid argument，F1 守卫版纯靠先行降级的巧合掩盖此坑）。验收：171 parity + f1 复现器 D=64/128 + A/B 路径生效实证 + bench CLI D=64 无回归（attn-mask 2.13x）
  - [x] PC-0-1：fp8/fp4 场景（量化六族 kernel，原 PC-0 主体设计）—— 2026-09-01 完成（A0 17ac22f 分类器量化族扩展 + 平面校验 → B1-B6 六 kernel 垂直切片 16eaea7/39c63ea/f42b12a/f12406f/b194bec/c2fc67d → C 7d5ca4c parity 扩展 159 用例 + bench 全 D 矩阵验收 + TMA dummy descriptor stride 修复）。
    - 模式总览：**mode 3 全驻留为主力**（fp8 persist_d 1.84x / fp4 split_d D=320 1.67x / fp4 m4n2 1.04x），**occupancy 守卫**（resident 驻留不得降 CTA/SM：fp8 m4n2 3 CTA/SM，32KB 驻留降 1 CTA/SM 反慢 4.9% → 守卫后 mode 2 150.7ms gap 1.029）、**fp8 split_d D≥512 demote mode 0**（B-3 结构性劣化 2.4x → PC-0-4 专项）、fp4 persist_d rowvec-only（dense fp32 超预算全 demote）。
    - 逐 kernel（B=1 H=32 N=16384 rowvec，PRO 5000）：fp8 m4n2 mode 2 1.03x（守卫）；fp8 persist_d mode 3 **1.84x**（14.46 vs 26.61ms）；fp8 split_d D=320 mode 2 **1.20x** 达标 / D≥512 demote 0（1.0x 持平）；fp4 persist_d mode 3（parity 达标，gap 小——fp4 persist_d 本身 attn 占比低）；fp4 split_d D=320 mode 2/3 **1.67x**、D=512 **1.29x**；fp4 m4n2 mode 3 **1.04x**（401.5 vs 419.4ms，D=768）。
    - 关键设计：fp4 注入 `apply_attn_bias_fp4_rowcol_smem`（original-token-ordered tile，kv_perm32 索引，标量读——PC-0-3 向量化证伪结论复用）；persist_d 的 bias barrier 跨 grid-strided work 全局计数（bias_g/bias_gc 分离 producer/consumer）；m4n2 issue_bias_tma rowvec 双缓冲 + kv loop 顶 t+1 预取；fp8 split_d 与 fp16 族 barrier 协议同构。
    - C 阶段修复（7d5ca4c）：bench non-aligned task（Nkv=16383）在 make_tma_bias **dummy** descriptor 构建期 SIGABRT——dummy 行 stride 误用运行时 plane_cols（Nkv%8≠0 违 16B 断言），修为编译期 bias_cols（bias_desc_live ? plane_cols : bias_cols，4 launcher 统一）；parity 扩至 6 broadcast kinds × fp8/fp4 + fp4 D=768 + fp4 tile-vs-gmem 严格对照（storage-offset mode 0），159 用例全绿。
    - 遗留：量化家族两处**先在时序竞争**（fp4 persist_d FC-4 epilogue race + fp4 m4n2 interleaved race，均先于 PC-0-1 存在、stash 复现实证；fp8 六族实证干净）→ **PC-0-5** 专项；验证策略避开 interleaved bitwise 断言。复验脚本 `.tmp/pc1-bias-tile/`。
  - [x] PC-0-5：量化家族先在时序竞争排查（2026-09-01 深度排查 + 2026-09-02 触发面收窄与假说消元，**已收敛定性、专题搁置**）
    - 复现与判定（fp4 m4n2 D=768 bias mode 3，e5d p0 判定器 `.tmp/pc5-race/m4n2_e5d.py`）：同进程 no-bias 模板（Li0E，smem 56576B）前置 ×9 → bias 模板（Li3E，60416B）连续调用 **6/6 bitwise 非确定**（O body 真实错误，64 行对齐条纹，幅度 ~5e-3..3e-2，lse 稳定）；纯 bias 序列无 prelude **稳定**（C9a）。
    - 触发面（mode 矩阵，09-02）：共同条件 = **bias 数据经 smem**——mode 0（gmem 直读，`FFPA_BIAS_TILE_DISABLE=1`）**完全免疫**（p0 0/6 ×2 轮 + C9b 变体 0/6）**（09-04 已修正：该免疫只在冷/轻负载序列成立，重负载前置下 mode 0 亦触发，见下方重启条目）**；mode 2（TMA row-broadcast，`FFPA_BIAS_RESIDENT_DISABLE=1`）**6/6 ×2 轮**（Li2E 模板 profiler 实证）；mode 3（fill STS）6/6。**D 依赖实为 kernel 结构依赖**：D=768→split_d_m4n2、D=320→split_d（headdim 分发）；m4n2 独有 **P 跨 N-warp smem 通信**（atom 布局 (4,2,1) 下 P 无法留寄存器 → softmax 写 [kBr,kBc] smem staging tile、各 N-warp 读回量化，且 row max/sum 跨 N-warp 经 softmax exchange buffer 归约，split_d 无此路径，见 `attn_traits.cuh` M4N2 traits 注释）——D=320（split_d，无 P smem 通信）pad=0 即免疫，仅 D=768（m4n2，有 P smem 通信 + 60416B 大占用）复现 → 触发与 **P smem 通信存在**强相关。
    - **假说消元（09-02）**：①跨模板切换——C9b/C10 同模板（Li3E/Li2E）零 bias prelude → 随机 bias 仍 6/6 → **证伪**；②H1 布局同位叠加——D=768 pad 256/1024/4096B 把 bias_base 移出 no-bias 尾部字节带（profiler 实证仍走 mode 3 未 demote）仍 6/6 → **证伪**；③allocator 紧缩——expandable_segments 仍 6/6 → 排除。叠加 09-01 的 bias 数值（C2a）/写方式（C1）/注入调用（C4）消元 → 与 smem 内容、布局、写协议、allocator 全部无关，**剩下 P smem 通信 × bias smem 写 × 大占用**这一 m4n2 结构组合。
    - 排除项（全部实证）：smem barrier 语义缺失（racecheck 0 hazard 且 race 在 hook 下复现；A2/A3 语义等价注入无效）、插桩伪影链、共享量化链（E5c split_d 稳定）、越界写（launcher printf 56320+4096=60416B 合法）、未初始化读（initcheck 0）。
    - 修复探索：候选 A（kv loop 顶无条件 barrier）修 race 但 mode 3 慢 57%（SASS：spill 822 vs 440/tile），且修复为 spill 时序副产物（A3 等价却不修）→ 拒绝。**已落地**：tail work `tma_store_wait` 无条件化 + `__threadfence`（协议补全，无回退）+ E1 清理 + race 用例沉淀 `tests/test_ffpa_fp4_m4n2_bias_race.py`（xfail）+ `FFPA_BIAS_TILE_DISABLE`/`FFPA_BIAS_RESIDENT_DISABLE` A-B 开关。
    - 定性与搁置决定：**m4n2 P smem 通信 × bias smem 写 × 大占用** 的时序敏感竞争（语言级协议完备、sanitizer 不可见、布局/内容/协议/allocator 全部排除）——疑似 sm_120 TMA/L2 与调度交互的硬件/driver 层问题。**影响面窄（仅 fp4 m4n2 bias，使用人少）→ 专题搁置**。缓解：`FFPA_BIAS_TILE_DISABLE=1` 退 mode 0 完全免疫（牺牲 PC-0-1 attn-mask 收益）。根治待 NVIDIA 上报（最小复现器 + mode 矩阵 + D/结构依赖证据链完整）；fp4 persist_d 3/30 待独立排查（persist_d bias 已实测干净：D=256/D=128 各 0/30，无 P 跨 N-warp staging 不满足 PC-0-5 组合；该 3/30 为独立 epilogue race、非 PC-0-5、persist_d 无 mode 0 等价路径需另解）；fp16/fp8/fp4 split_d 全路径实证干净。
  - [x] PC-0-5 重启深挖与止血（2026-09-04，pure 100% 触发重启 → pin mode 0 落地 → 残留负载敏感定性 → 接受收敛）
    - 重启证据：pure bias（无 prelude）**100% 触发**（10/10 轮全 6/6，8/8 unique；918ae3c squash 终态即存在——build41 用 918ae3c csrc 重建实证 6/6×2，排除近期引入）；**推翻 09-02 C9a"纯 bias 稳定"定性**。
    - 指纹定位（morph2/morph3）：坏行**恒限于每 64 行 tile 的 rows 0-15**（warp_id%4==0 行组，row%64 hist 16/48 分明），坏 d **恒 [192,224) 连续 32 列**（= n_warp0 半区 v_chunk3 = 单个 SM120_16x32x64_TN_VS_NVFP4 atom 的 C 宽 = 一个 uint4 B 寄存器载荷），坏块 = **单个 (m-warp0, n-warp0, v-chunk3) PV C 累计 tile**；行间独立（27 种 bad-d 子集）；diff med 0.0024 / max 0.039（O 值域 ±0.065，~0.8% 元素，"漏/多一个 tile 贡献项"量级）；lse 稳定 → 排除 A(P)/SFV/DS 污染（都会动 lse 或更大影响面）。
    - 维度窗口（scope.py）：**works×tiles 乘积**——2×32 / 32×2 稳定，32×32 轻触发（6 calls 1 次），128×32 全触发（6/6）；32×32 时 grid>works 单 work CTA 也坏 → 排除 persistent work 轮转路径，race 在 kv 循环内。
    - 假设证伪链：①`mma.sync`（SM120 f8f6f4 blockscaled 为**同步 warp mma**，cute/arch/mma_sm120.hpp 实证）→ async-mma/arrive-过早假设排除；②ptxas -O2 单 TU 重编（D1）→ race+指纹不变 → 编译调度排除；③producer 挪位 tid0→255（E-prod，7 处）→ race 存活、条纹相位漂移 +12 → producer×consumer 交织排除，坏块跟随 warp×chunk **相对时序**；④协议全文复审（P roundtrip/exchange/v_empty/v_full/epilogue_done/batch sO/tma_store_wait）语言级自洽（与 09-01/09-02 结论一致）。
    - **决定性反转（m4n2_nobias_load.py）**：重负载 prelude（任意 matmul 8192²×9，非 no-bias 模板特异）下 **mode 0 gmem 直读也触发**（6-12/24，指纹同源）→ **修正 09-02"mode 0 完全免疫"**——免疫只在冷/轻负载序列成立；**no-bias 模板同负载 0/24 干净** → race = **bias 模板（kHasAttnBias=1）× 负载时序窗口**的硬件层问题（时钟/功耗/内存系统状态改变 SM 内相对时序，踩中固定 (warp, chunk) 位置的既有窗口）。
    - 止血修复：launcher（`launch_cute_fwd_split_d_m4n2_fp4_sm120_impl`）pin mode 0（纯 bias 10/10 0/6；attn-mask 397.9→419.1ms ~5%，其余 task 零回归，bench 全绿）+ `FFPA_BIAS_TILE_KEEP=1` 逃生口；测试分类：pure 序列**转正必过 gate**（`test_ffpa_fp4_m4n2_pure_bias_determinism` 新增）+ prelude 序列 xfail(strict=False) 残留标记。
    - 残留接受理由：触发需"bias + 重负载前置 + m4n2（仅 D≥768 fp4）"三重条件，现实场景少；根治需硬件层归因（NVIDIA 上报材料已齐：8 复现脚本 `.tmp/pc5-race/`、指纹/窗口/证伪链完整）。
    - 顺带发现：**Nq=64（MB=1，kBr=64 单 q-tile）illegal memory access 独立 bug**（scope.py 1work/1tile 形态触发）——非 PC-0-5，待独立修复。
  - [x] PC-0-6：D=768 split-d vs m4n2 A/B + 寄存器模型分析（2026-09-02 完成）
    - **fp4 split_d D=768 物理不可行**：split_d fp4 的 O epilogue 用整块释放 smem 做 staging，`static_assert(kBr*kHeadDim*2 <= kSmemBytes)` → kBr=128×768×2B=**196608B ≫ 101376B opt-in**（2 倍超预算，编译期即失败）。m4n2 用 kBr=64 把 O staging 降到 98304B 才可行——**m4n2 是 D≥768 fp4 的唯一可行 kernel**。
    - **fp8 A/B 实证**（`FFPA_FP8_FORCE_KERNEL=split_d|m4n2`，D=768 N=8192 B1 H32，PRO 5000，TFLOPS）：
      | 场景 | split_d | m4n2 | m4n2 优势 |
      |---|---|---|---|
      | self-attn | 139T / 47.59ms | 158T / 41.88ms | 1.14x |
      | cross-attn | 85T / 9.70ms | 102T / 8.11ms | 1.20x |
      | gqa | 141T / 46.69ms | 169T / 39.14ms | 1.20x |
      | causal | 126T / 26.13ms | 149T / 22.20ms | 1.18x |
      | **attn-mask** | **50T / 130.96ms** | **159T / 41.52ms** | **3.15x（split_d 崩溃至 0.81x<SDPA）** |
    - **寄存器压力（cuobjdump -res-usage，D=768 sm_120f）**：fp8 split_d REG:255 + **STACK 1104-1944B**（重 spill）；fp8 m4n2 REG:255 + STACK 0-264B（几乎无 spill）；fp4 m4n2 REG:255 + STACK 528-920B（中等 spill）；fp16 split_d REG:196-254 + STACK ~1536B。
    - **结论（验证用户假设）**：m4n2 的 O regs/线程 = D/4（192），split_d = D/2（384）——**量化路径沿用 fp16 m4n2 的 kBr/kBc 几何收缩思路方向正确，但量化 kernel 的寄存器模型不等同 fp16**：①fp4 多了 SFQ/SFK/SFVt/DS 量化状态寄存器（fp4 m4n2 STACK 528-920B > fp16 m4n2 0-312B）；②attn-mask 场景 split_d 的 spill 与 bias smem 写叠加后**性能崩溃**（3.15x 差距），而 m4n2 把 spill 压到近零保持稳定。→ m4n2 在量化大 D 路径是**必需而非可选**，fp4 m4n2 的中等 spill 是后续 PC-10 的优化面。
  - [x] PC-0-2：native/native_tma 场景（标量 loader 路径）—— 2026-09-03 完成（c645ac1，四文件 +208/−21）：
    - Phase 0 定性（NCU，PRO 5000，B=1 H32 N=4096 D=512/128，tma+native 各两配置）：bias 注入开销**指令数瓶颈而非带宽瓶颈**——bias 路径指令 +1.39G（+22%）、L1 hit 98.4%、sectors/request 1.0、long_scoreboard 仅 +0.64、mio_throttle 不变。标量路径每 warp 每 tile 2048 次 load 只覆盖 128 个独立值（per-lane 2x + 跨 lane 8x = 16x 冗余）。**决策树落分支①（指令开销主导）→ 内联向量化**，不引入独立预取协议（non-WS kernel 无 producer，问题在 load 条数不在单次延迟）。
    - 实现（`prefill.cuh` + `launch.cuh` + 两个 `split_d.cuh`）：`load_attn_bias_pair`（half2/__nv_bfloat162/float2 三分支）+ `sync_apply_attn_bias_rowvec`（pair-base 提升、`k1 < Nkv` 尾部安全、消行界检查、单语句形式保 f32 FFMA 收缩与 f16-acc 表达式形态——禁止预乘 `inv_scale`，舍入不同）。门控：`(stride_m==0 || Nq==1) && stride_n==1 && ptr 对齐 && (b,h) plane stride 偶数`，`FFPA_BIAS_ROWVEC_DISABLE=1` 强制标量路径做 A/B；Nq==1 时行索引恒 0，padding 行读 row-0 值但被 O/LSE 写守卫丢弃。不满足门控（misaligned / dense 4D mask）自动回落原标量路径。
    - 验收（全部 `python -m ffpa_attn.bench` CLI）：11/11 bitwise parity（tma+native，fp16/bf16 query × fp16/bf16/fp32 bias，odd-Nkv 尾 tile、misaligned 自动回落、Nq=1、tail Q rows、GQA、dense fallback）；四形状加速比（vs SDPA）tma D512 1.42→2.00x、native D512 1.89x、native D128 1.67x、tma D128 1.44x；N=16384 抽查无 L2 退化（tma mask 2.02x/self 2.30x，native 1.84x/2.00x）；全 task 套件两 impl 零回归。
    - 残余与搁置：优化后 NCU 显示残余为 **load-latency 主导**（long_scoreboard 2.40 / wait stall），地址提升 v2 实测无增益（编译器已优化寻址）；预取/双缓冲是唯一剩余杠杆但与 255 寄存器预算冲突（spill 墙），内联约束下判为实际上限，搁置。f16-acc 分支（kMmaAccFloat32=0）已实例化但当前构建矩阵不可达（默认 QK/PV f32 acc），保留镜像标量路径表达式形态。复验：`.tmp/pc02-native-mask/`（parity.py + 5 ncu-rep + bench 输出）。
  - [x] PC-0-3：D=320 split_d attn-mask 注入开销专项（PC-0-0 遗留，目标 gap 1.44→≤1.2x）—— 2026-09-04 **证伪关闭**（目标判定不可达，gap 1.44 为 1 CTA/SM 结构极限）
    - 现状（2026-08-31，N=16384 B1 H32，self 57.4ms/191T）：attn-mask 83.5ms/132T，gap 1.44、TFLOPS 70%；迭代路径 mode0 gmem 2.21 → 单缓冲 tile 1.51 → rowvec 双缓冲 1.48 → mode 3 全驻留 1.44；dense 全量 1.59
    - NCU 定性（三轮）：SM 吞吐 71.4%→47.6%（memory 42.8%/occupancy 16.67% 不变）；bias 同步开销已消除（mode 3 后 short_scoreboard 1.47→1.21、bias barrier 全无）；sleeping 1.7 = tid==0 producer 在 qk/v empty-wait 空转（注入拉长 consumer 每 tile → K/V 流水变浅）。**剩余 gap 本质 = 注入计算本身进入 critical path**（1 CTA/SM × 8 warps，无并行 warp 掩盖）
    - 杠杆②（注入×online_softmax row-max 融合）**已证伪（2026-09-04 实测，PRO 5000）**：4 组矩阵（fused×tile，`FFPA_BIAS_FUSED_DISABLE`/`FFPA_BIAS_TILE_DISABLE`）——分离注入+smem tile 83.56ms（基线）vs **fused+smem tile 87.74ms（反而慢 4.2ms）**：单 pass 融合使 softmax 段寄存器压力/ILP 变差，且 kernel 非 softmax 发射受限（gmem 对照组 fused 122.1 vs 分离 128.5 有小收益，但被 tile 路径完全取代）。**bitwise parity 双重代价**：①`__fmaf_rn` 收缩钉法（kContractInj/kContractExp2）在 REG=255 满载 kernel 引发 FFMA 调度压力 → attn-mask 83.54→86.66（+3.1ms，归因实验确凿）；②两侧自然表达式则 nvcc 上下文收缩不一致（12 用例 ulp 差异 parity fail）。加载侧才是收益之王：tile vs gmem 直读 = 83.56 vs 128.52（**prefetch 收益 45ms/35%**，已在 PC-0-0 基线落地）→ bias 路径先测 bandwidth 维度再做 compute 融合。代码已 revert（diff 备份 `.tmp/pc03-fusion/diff_pc03_final_abandoned.patch`，复验脚本同目录）；③ 提高 occupancy 不可行（smem 96KB 已 1 CTA/SM 上限）
    - 相关文件：`csrc/cuffpa/cute/attn_bias.cuh`（注入函数）、`sm_120/split_d.cuh`（主循环）、`.tmp/pc0-bias-tile/`（ab2_bench/ncu_probe 脚本与三轮 ncu-rep）、`.tmp/pc03-fusion/`（4 组矩阵 bench + parity.py + sass_exp2_form.py）
  - [ ] PC-0-4：fp8 split_d D≥512 attn-mask tile 化专项（PC-0-1 B-3 遗留，目标 gap 1.48→≤1.2x）—— 2026-09-01 现状定案（全部实测，PRO 5000，B=1 H=32 N=16384 rowvec fp16 mask，s2 配置）：
    - 现象：D=320（10 QK chunks）tile mode 2 正常达标（50.9 vs gmem 69.6ms，gap 1.203）；**D=512（16 QK chunks）tile 反慢 2.4x**（mode 2 275ms / mode 3 267ms vs gmem 115.5ms），bitwise parity 全对（非正确性问题）。
    - NCU 证据链（`--page source --csv` pcsamp + cuobjdump）：排除 spill（LOCAL:0）、排除 occupancy（两版均 16.67%，**Block Limit Registers=1**——REG:255，smem 守卫在该家族恒空转）、排除协议错（模板尾参与 D=320 逐字同构）。定位：tile 版 **SYNCS.PHASECHK（mbarrier phase 自旋）占 18% 采样（top8 全是 qk_full wait 点）+ sleeping 22.3%**（gmem 版仅 3.5%）——**smem 注入（LDS）进入 consumer critical path 拉长每 tile 时间 → s2 两级 QK 流水断供 → 全线程自旋等 K**。gmem 版的 LDG 注入走 L1 load path 不抢 smem/MIO，QK s2r 不受拖累反而快。
    - 根因定性：**结构性**——1 CTA/SM（REG 限制）无 warp 掩盖 × s2 浅流水 × D=512 的 16-chunk 高 s2r MIO 压力三者叠加；与 PC-0-3 的"注入进入 critical path"同构，但 fp8 split_d 流水更浅所以直接崩。
    - 附带结论：**mode 排序是家族属性不可跨族继承**——mode 3 在 fp16 家族最优（PC-0-0），在 fp8 split_d 全败（D=320：mode3 59.2 vs mode2 50.9；D=512：267 ≈ mode2 275 都烂），B-3 dispatch 已移除 mode 3 分支。
    - 当前处置（commit f42b12a）：launcher 数据驱动 `kHeadDim >= 512 → mode 0`（实测 115.8ms 与 gmem 持平零劣化），D=320 保持 mode 2 达标。
    - 候选方向（动手前先 A/B）：①bias 激活时升 s3/s4 加深 QK 流水（48KB base + 512B tile 仍放得下，直击断供根因）；②`bias·inv_sd` tile 级预乘减半注入 LDS 压力（PC-0-1 设计的可选项）；③注入移出 softmax critical path（与 PV 重叠的分块注入）；④注入向量化**不要做**（PC-0-3 杠杆①已证伪 +2.7%，低 occupancy 依赖链加深）。
    - 复验脚本：`.tmp/pc1-bias-tile/fp8_splitd_check.py`（parity+A/B）、`fp8_mode_of.py`（模板尾参）、`fp8_ncu_run.py`（NCU 采集）。
- [ ] PC-1：Mega Quantize Kernel —— P 轨基建
- [ ] PC-2：增量融合（Mega Kernel 步进，被 PC-1 收编）
- [ ] PC-3：N-crossover 量化配置自适应
- [ ] PC-4：fp4 persist-D attn kernel 内部优化
- [ ] PC-5：CUDA graph 友好化
- [ ] PC-6：sm_89 fp8 int4 QK（低优搁置：sm_120 无原生 int4 MMA，SA2 int4 kernel 未开源）
- [ ] PC-11：warp 级 `__any_sync` lazy-rescale 统一治理（精度治理专项，2026-09-01 立项）
  - 背景：PC-0-5 调查早期曾怀疑 fp4 split_d_m4n2 的 warp-uniform lazy-rescale（`__any_sync(0xffffffff, row_scale != 1.0f ...)`）参与 bias 场景的 bitwise 非确定，**后续消元已证伪 vote 因果**（force-rescale 后仍触发；race 真身为跨模板时序敏感竞争，见 PC-0-5 完成清单）。本专项保留的治理价值：`__any_sync` 投票模式源自 CUTLASS 77_blackwell_fmha 的 **shared-TMEM collective rescale** 场景——那里 warp-uniform 是必须的；本仓库所有 kernel 的 rescale 目标均为 **thread-private 寄存器**，投票既非必需，又把 per-row 决策强行提升为 warp-uniform 分支，徒增调度脆弱性与 warp divergence 面。
  - 参考实现（治理目标形态）：`csrc/cuffpa/cute/fp8/sm_120/persist_d.cuh` 已改为 thread-level per-row——`const float rs = (kv_tile > 0 && row_scale[row] < 1.0f) ? row_scale[row] : 1.0f;` 逐行独立判断（`< 1.0f` 顺带拒绝全 masked 行的 NaN scale），无跨 lane 通信；其注释含完整论证。
  - 治理清单（`grep -r "__any_sync" csrc/cuffpa/cute/` 全量 9 处实际使用 + 1 处已治理注释）：
    - `sm_120/split_d_m4n2.cuh`、`sm_120/split_d.cuh`、`sm_120/persist_d.cuh`（fp16 cute 三族）
    - `fp8/sm_120/split_d_m4n2.cuh`、`fp8/sm_120/split_d.cuh`（fp8 persist_d 已治理）
    - `fp4/sm_120/split_d_m4n2.cuh`（PC-0-5 实证 race，随 PC-0-5 修复先行落地）、`fp4/sm_120/split_d.cuh`、`fp4/sm_120/persist_d.cuh`（fp4 persist_d 3/30 低概率触发待排查）
    - `sm_80/split_d.cuh`
  - 动作与验收：①逐 kernel 转 per-row lazy-rescale（语义等价，多数 tile 无 max 增长时按行跳过乘法，性能预期持平或略优）；②每处 `ffpa_attn.bench` A/B 无回归；③每家族补"no-bias 前置 ×N → bias self-loop bitwise 断言"用例（PC-0-5 复现脚本 `.tmp/pc5-race/` 沉淀为 tests 后纳入）；④PC-0-5 的 fp4 m4n2 修复是本专项第一块拼图，其余 8 处排期跟进。
- [ ] PC-7：fp8 split-D (M8N1) 量化大 D kernel 性能优化
- [ ] PC-8：fp8 split-D M4N2 量化大 D kernel 性能优化
- [ ] PC-9：fp4 split-D (M8N1) 量化大 D kernel 性能优化
- [ ] PC-10：fp4 split-D M4N2 量化大 D kernel 性能优化
- [ ] PC-12：cute sm_80 fp16 性能优化（cp.async + 多级流水线，fp8/sm_89 路线前置，2026-09-02 立项）
  - 背景：cute sm_80 路径（`cute/sm_80/split_d.cuh`，cp.async loader）此前从未做专项性能优化——2026-09-02 补验 attn-mask 时实测 vs CUTE_TMA 慢 7%（D=768）~28%（D=128），cp.async loader 开销与小 D 流水深度是主因。
  - 战略意义（fp8/sm_89 路线的训练场）：**sm_89 不支持 TMA 与 async proxy，只有 cp.async general proxy**——fp8 sm_89 量化路径无法复用 sm_120 的 TMA + WS/non-WS 模式，只能走 **cp.async + 多级流水线**。该技术路线的全部经验（stage 深度/同步开销/barrier 协议/寄存器规划 under cp.async）必须先在 cute sm_80 fp16 上打磨成熟，性能达标后才迁移到 cute/fp8/sm_89 实现 fp8 量化（即 sm_89 fp8 路线复活，解锁 PC-6）。
  - 动作与验收：①NCU 基线（loader 停顿/s2r MIO/occupancy）；②stage 深度与 stage 组合扫描（现状 sm_120 上 cap 2/3，物理 smem 上限内探索）；③cp.async commit-group 分组与多级流水线重构；④`ffpa_attn.bench`（CUTE hint）vs CUTE_TMA gap 收敛到 ≤1.1x 作为"达标"准出（经验才值得迁移）；⑤达标后开 cute/fp8/sm_89 专项（量化链 + kernel 移植，届时与 FC-12 一并评估）。
- [ ] PC-13：fp8/fp4 hybrid 路径性能优化（双 attn kernel → 融合 kernel，2026-09-02 立项）
  - 背景：hybrid（causal 前缀 `n_early` 行走 fp16 保精度、其余行走量化）当前是 **stage-1 fp16 attn kernel + stage-2 量化 attn kernel 两条主 kernel 路径背靠背**（`launch.cuh` 6 处 hybrid 分支：fp8/fp4 × persist_d/split_d/m4n2 全同构）：`prepare_hybrid_stage1` 物化切片 + `O_e`/`lse_e` 临时分配 + `O.slice(2,0,n_early).copy_(O_e)` 拼接拷贝 + K/V 双份加载（fp16 原值给 stage-1、量化值给 stage-2）+ 两条 pre-kernel 链与两次 launch——固定开销显著（n_early 占比越大越亏），且 stage-1 走 fp16 kernel 本身吞吐低于量化 kernel。
  - 融合方向（设计要点）：单 kernel 内按 work 的 Q 行域选精度——前缀 tile 走 fp16 MMA、其余 tile 走 fp8/fp4 MMA，同 grid/同流水/同 epilogue 直写 O（`q_start_row` 行域判定已具备），消除拼接拷贝与双份 K/V IO；难点 = 同一 kernel 内两套 smem 布局/量化状态的条件编译分支对寄存器压力的影响（PC-0-6 教训：量化寄存器模型不等同 fp16）。
  - 动作与验收：①hybrid 现状开销量化（nsys：双 kernel + copy + pre-kernel 链时间占比，`n_early` 扫描）；②融合 kernel 原型（先 persist_d 家族，行域分支最简单）；③hybrid bench A/B + 精度对照（hybrid 本身是精度特性，融合版必须保持 stage-1 fp16 数值语义不变）。
  - 关联：与 PC-7~PC-10（量化大 D kernel 内部优化）协同——融合 kernel 的量化段直接继承其优化成果。
- [x] PC-14：fp16 dropout 路径性能优化（RNG bitmap 预计算 + producer/consumer 重排，2026-09-02 立项，2026-09-03 完成）
  - 背景：D=64 bench 实测 dropout task **0.83x**（FFPA 37.69ms vs SDPA 31.31ms；无 mask self-attn 9.97ms → dropout 3.8 倍耗时，当前量化路径外唯一劣化点）。根因：`apply_dropout_rowcol`（cute/dropout.cuh）在 consumer 的 QK-MMA→softmax 串行链上**逐 element 计算 Philox**（philox4x32_10 = 4×10 round 整数乘加/异或链，每 2 个 score 一次完整调用），算术密度远超 score 本身的 add/mul。
  - 思路（与 PC-0 attn bias tile 同构）：producer warp（128T，目前仅发 TMA 基本空闲）预计算下一 KV tile 的 `[kBr,kBc]` dropout keep bitmap（1 bit/elem）写入 smem 预取窗口——复用 bias tile 已验证的 Q 区/extra 段布局与 `bias_full/empty` mbarrier 协议；consumer 只查 bitmap + 乘 keep_scale。Philox 生成侧按 4 连续列块对齐（一次 philox4x32_10 出 4 个决策），消除 lane0==3 的重算分支。
  - 进阶（评估项）：drop 的 keep_scale 因子后置到 P 域/row_scale 折叠（online softmax 已有 rescale 乘法链，drop=置 0 可与 rescale 合并乘法）。
  - 动作与验收：①NCU 基线（dropout 段指令占比/串行链停顿）；②producer bitmap 原型（先 persist_d 家族）；③`ffpa_attn.bench` dropout task ≥1.0x SDPA（D=64/128 全 dtype）；④**bitwise RNG 语义不变**（philox offset = `philox_offset + row*Nkv + col` 的决策序列与现状一致，保证与 torch 训练对齐；parity 注意 torch 参考 2^32 offset wrap bug 标注）。
  - 进展（2026-09-02，persist_d 家族完成）：producer 预计算方案**已证伪**（philox 集中在 4/12 warps = 3× 指令浓缩成机器瓶颈，15.6ms；且 producer 提寄存器预算在 32/64/96 全档 spill）。落地形态为 **consumer 侧双缓冲 smem bitmap**：256 consumer threads 在 kv 迭代顶部（TMA wait 窗口，脱离 softmax→PV 关键路径）按 offset-quad 预生成下一 tile 的 keep bitmap（1 philox/4 元素，旧 inline 为 1/2，philox 总量减半），softmax 后以寄存器 bit-test 应用，每 tile 一次 NamedBarrier(256)；producer 回归薄 TMA issuer（setmaxnreg 维持 32/232 不动）。验收：D=64 0.82x→**1.02x**、D=128 **2.25x**，15/15 bitwise A/B + 全 task 矩阵无回归。
  - 进展（2026-09-03，split_d 家族完成 / m4n2 证伪）：**split_d 同构移植成功**（half-row 生成，kBr=128/kBc=128，`__syncthreads` 替代 NamedBarrier）：D=320 dropout forward **2.05x** SDPA（inline 1.52x），A/B 22/22 bit-exact（dense/causal/odd-Nkv/row-bias fp16+fp32/GQA/bf16），全 task 矩阵零回归（8842cff）。**m4n2 证伪、该路径不实现 bitmap**：bit-exact 成立（word-per-thread 生成，27/27），但性能**反向**——D=768 N=16384 bitmap 212.82ms(2.16x) 慢于禁用态 inline 202.31ms(2.25x)，-5.2%。根因：m4n2 tile 仅 64×64（4KB bitmap），inline philox 量本小且 kernel 时间由 PV split-D + cross-N-warp softmax exchange 主导，RNG 不在关键路径上；bitmap 的生成 + smem 写读往返是固定净增开销，无 RNG 节省可抵。已回滚全部 m4n2 改动；未来若 m4n2 结构变化（如 PC-8/PC-10 优化后 RNG 占比上升）再重新评估。
  - **理论结论（详见 SKILL §11.16）**：bit-exact Philox 契约存在指令地板（每 4 决策约 150 条线程指令 → PRO 5000 上 N=16384 约 20ms / N=8192 约 5.1ms，实测已贴地板）；加速比上限 $= (T_{\text{self}} + \Delta_{\text{SDPA}}) / (T_{\text{self}} + \text{floor})$，契约不变时结构上到不了 2x（完美实现约 1.2x），与 attn_bias 的 2.1x+ 量级差源于 bias 是 memory-bound 数据问题而 dropout 是 ALU-bound 计算问题。要突破上限只能换非 bit-exact 的便宜 RNG（产品决策）。
  - 进展（2026-09-03，sm_80 完成 / persist-D kBc 门控 / 收尾验收）：**sm_80 split_d 同构移植完成**（half-row 生成 + `__syncthreads`，f158eb1）；**persist-D 增加 kBc≥64 编译期门控**（d6a4a1d）：half-row 方案要求 row 跨偶数个 32-bit word，D=192/256 的 kBc=32 实例化（fp4-hybrid stage-1 调用点触发）经 `kBitmapCapable` 回落 inline Philox，launcher 同步加 `kBc >= 64` 运行期门控跳过 bitmap smem 预算。收尾验收：全 headdim 构建 162/162，fp16 7-task 套件零回归（dropout 1.90x@D512，persist-D parity D64 1.03x / D128 2.13x），fp8/fp4 smoke 通过。PC-14 关闭。

## 实施路线图（基建优先，承上启下）

原则：**通用基础设施先行、优先级最高**——基建做完后，后续每一项的改动面更小、
可独立验证、风险更低。

```
阶段 1（F1 通用基建，全局最高优先级）
  FC-1 独立 Lv 尾参 ──► FC-2 nhd_out 动态 O 描述符
        │（描述符/尾参基建，受益方 ↓）
        ├─► FC-3 strided+hybrid 组合放开（收尾）
        ├─► FC-6 smooth_v/MXFP8-PV 三族扩展 ✓（2026-08-28，76a8bd8）
        └─► cache-dit _keep_or_pack 物化兜底移除 ✓（2026-08-28，cache-dit@4b5c977）
阶段 2（F2 特性对齐）
  FC-4 attn_bias（S/P 域注入基建）──► FC-5 dropout ⏸（暂不实施，复用注入点）
阶段 3（F3 覆盖，按需推进，互相独立）
  FC-7 短 Nq/decode ⏸ ｜ FC-8 native pad ｜ FC-9 backward 评估 ⏸ ｜
  FC-10 sm90/sm100 ⏸ ｜ FC-12 cute sm_80 persist-D/M4N2 ⏸（FC-7/FC-9/FC-10/FC-12 均暂不实施）
阶段 4（轨道 P）
  PC-0 attn mask 场景性能优化（P0：attn-mask 是当前量化路径最大退化点）
        ├─► PC-0-0 cute/cute_tma（fp16 家族，方案 A 热身台阶）
        ├─► PC-0-1 fp8/fp4（量化六族，主体）
        └─► PC-0-2 native/native_tma（✅ c645ac1，rowvec 内联向量化）
  PC-1 Mega Quantize Kernel（先做 cooperative 两阶段原型）
        ├─► 收编 PC-2（增量融合是其落地台阶）
        └─► 联动 PC-5 ⏸（暂不实施；launch 形态定型后才能定 graph 兼容方案）
  PC-3 配置自适应 ｜ PC-4 fp4 persist-D kernel 内部（与上并行，互不依赖）
  PC-7 → PC-8 → PC-9 → PC-10 量化大 D kernel（优化复杂，严格逐个推进：
        fp8 split-D → fp8 M4N2 → fp4 split-D → fp4 M4N2，上一项验收后再启动下一项）
  PC-12 cute sm_80 fp16（cp.async + 多级流水线）──达标──► cute/fp8/sm_89 量化实现
        （sm_89 无 TMA/async proxy，fp8 只能走 cp.async 路线；复活后解锁 PC-6）
  PC-13 fp8/fp4 hybrid 融合 kernel（现状 = fp16 + 量化两条主 attn kernel 背靠背，
        拼接拷贝/双份 K/V IO/双 launch 固定开销显著 → 单 kernel 内按 Q 行域选精度）
  PC-14 fp16 dropout 性能 ✅（consumer 侧双缓冲 bitmap，producer 方案证伪；
        persist_d/split_d/sm_80 完成、m4n2 证伪；D=64 0.83x→1.02x，D=128 2.25x）
  PC-6 sm_89 int4 QK ⏸（暂不实施，低优搁置，不入执行序列；前置 = PC-12 达标）
```

> ⏸ = **暂不实施，仅保留设计稿**：不入执行序列、不排期；未来大概率不做，
> 仅当出现真实需求时重新评估。当前共 7 项：FC-5 / FC-7 / FC-9 / FC-10 /
> FC-12 / PC-5 / PC-6。（FC-7 搁置理由：短 Nq/decode 量化基本没有收益——固定前处理
> 链开销结构性占优，小 Nq 下量化 kernel 的吞吐优势摊不开，且 decode 已由
> native split-KV fp16 路径覆盖。）

承上启下要点：

1. **FC-1 → FC-2 → FC-3（顺序不可颠倒）**：FC-1 只加 `Lv` 尾参（纯调用点扩展、
   默认 `nullptr` 零行为变化、风险最小），先打通新描述符路径；FC-2 在同一批调用点
   照抄 persist-D 已验证的动态 descriptor 模式（改动面大但有范本）；FC-3 在两者就位
   后放开组合，只改判定逻辑。三步各自独立验收、独立回退。
2. **FC-4 → FC-5**：`attn_bias` 建立的 S 域注入基建（per-block bias 载入、P 量化前
   修正、`-inf` 掩码叠加顺序）正是 dropout 乘性掩码所需的注入点，FC-5 复用即可
   （FC-5 ⏸ 暂不实施：FC-4 落地后注入点自然可用，届时再评估）。
3. **PC-1 → PC-2/PC-5**：Mega Kernel 的两阶段 + grid 屏障是 aux 链融合的终局形态；
   PC-2 的相邻对融合是其增量台阶（PC-1 落地即收编），PC-5 的 graph 兼容性必须等
   PC-1 的 cooperative launch 形态定型后才有确定方案（PC-5 ⏸ 暂不实施，届时再评估）。
4. **阶段 3 各项与阶段 1/2 无依赖**，可在基建间隙穿插推进，但不得抢占基建资源
   （实际排期仅 FC-8；FC-7/FC-9/FC-10 ⏸ 暂不实施，不占档期）。

---

## 轨道 F1：布局零拷贝闭环（最高优先级）

> 这三项共同解锁**大 D 场景下的 NHD/strided 全闭环**。persist-D 三族已完备（报告 §7.4），
> 缺口全部集中在 split-D / M4N2（D > fp8 224 / fp4 256 / fp16 128）。下游（diffusers 传
> `tensor_layout='NHD'`、cache-dit 走 fused-QKV）一旦命中大 D 即物化或报错，是最高频的功能痛点。

### FC-1：split-D/M4N2 独立 Lv（strided-NHD 读）

- **Status**: Done（2026-08-28，ffpa-attn cc8e8dc/4a49d38/882ee07） ｜ **Priority**: F1 ｜ **Track**: 功能（布局）
  实施偏离设计稿两点：fp4 不引入 `Lv`（其 V 消费全走 tensor-stride 参数化
  quantize kernel，无需描述符）；fp8 的 `Lq` 一并放宽（fused-QKV 的 Q 同为
  strided chunk）。顺带修复 fp16 split-D/M4N2 的 q_c 绑死 kv_c latent bug
  （BHND Q + NHD K/V 混合布局静默错读，现独立分派）。

#### Motivation

persist-D 已验证独立 `Lv` 模式（报告 §7.4 行 3）：fp8/fp4 persist-D launcher impl
各自经 `ffpa_layout_of(..., /*allow_strided_rows=*/true)` 为 Q/K/V 构建独立
`Fp8InputLayout`（`Lq/Lkv/Lv`），quantize/VT 前处理 kernel 吃 `&Lv` 行 stride，
使 strided fused-QKV（如 FLUX.2 single-stream 的 V）能零拷贝进 CuTe kernel。
split-D / M4N2 的 launcher impl 仍只用严格门禁的 `ffpa_layout_of(K, ...)` 构建
**共享** `Lkv`——strided fused-QKV 输入在 D>224(fp8) / D>256(fp4) 时仍需物化或
拒绝。这是把 persist-D 已验证模式推广到其余两族的最小、最直接动作，也是
FC-2/FC-3 的热身（先打通 split-D 上的新描述符路径）。

#### Design

1. split-D/M4N2 launcher impl（`launch_cute_fwd_split_d_fp8_sm120_impl` /
   `launch_cute_fwd_split_d_fp4_sm120_impl` /
   `launch_cute_fwd_split_d_m4n2_fp8_sm120_impl` /
   `launch_cute_fwd_split_d_m4n2_fp4_sm120_impl`，`csrc/cuffpa/cute/launch.cuh`）
   把 K/V 布局构建改成 persist-D 同款：独立 `Lkv`/`Lv`（`allow_strided_rows=true`），
   quantize/VT 前处理调用点传 `&Lv`（对照现状共享 `Lkv` 的调用点，如 fp8 L1239 起、
   fp4 L2471 起）。
2. fp16 split-D/M4N2（`launch_cute_fwd_split_d_sm120` /
   `launch_cute_fwd_split_d_m4n2_sm120`）的 TMA 描述符同步支持 V 独立行步长
   （参照 fp16 persist-D 的 NHD 寻址模式）。
3. 约束对齐：strided-NHD 输入在 split-D 上仍**不与** hybrid / causal tail 等路径
   叠加（沿用 persist-D 的 TORCH_CHECK 负例矩阵）；hybrid stage-1 调用点
   `prepare_hybrid_stage1`（`csrc/cuffpa/launch.cuh`）本项不动，留给 FC-3。

#### Files & Symbols

- `csrc/cuffpa/cute/launch.cuh`：split-D/M4N2 launcher impl（fp16 L105/L335、
  fp8 L1214/L1549、fp4 L2393/L2648）；`ffpa_layout_of`（L82，`allow_strided_rows`
  尾参）；参照 persist-D fp8 impl（L811 起，独立 `Lq/Lkv/Lv` 构建 L845-850、
  前处理 `&Lv` 传参 L986-1034）。
- `csrc/cuffpa/cute/fp8/input_layout.cuh`：`Fp8InputLayout`（前处理 kernel 的
  `s_row`/`s_batch` 行寻址）。

#### Validation

1. 新用例进 `tests/test_ffpa_nhd_layout.py`：strided fused-QKV × split-D（fp8 D=320）×
   M4N2（fp8 D=768）× fp4 split-D（D=320），bit-exact vs persist-D 同参数参考。
2. 负例矩阵扩一条：strided + hybrid 仍拒绝（在 FC-3 前）。
3. bitwise 8 场景 probe（附录 B）确认 BHND packed 路径零变化。

#### Risks & Rollback

- 布局构建放宽 + 前处理传参扩展，BHND packed 行为不变（满足放宽门禁的等价分支），
  回退 = 恢复严格 `ffpa_layout_of` 调用。
- fp4 split-D 的量化链（`csrc/cuffpa/cute/fp4/quantize_fp4.cuh`，独立于 fp8）须确认
  其 K/V 量化也吃 `Lv` 行 stride（对照 fp8 `quantize_fp8.cuh` 的 quantize/VT kernel
  已经 `Fp8InputLayout` 寻址的模式，报告 §5.6）。

#### Expected Benefit

大 D strided fused-QKV 零拷贝，消灭该场景物化；为下游（cache-dit fused-QKV 大 D）解锁。

#### Dependencies

无（纯调用点扩展）。

---

### FC-2：split-D/M4N2 NHD O 写

- **Status**: Done（2026-08-28，ffpa-attn df7d572/c4ca38b/2382ca4，fp8/fp4/fp16 三族） ｜ **Priority**: F1 ｜ **Track**: 功能（布局）

#### Motivation

`tensor_layout='NHD'` 在 D>224(fp8) / D>256(fp4) / D>128(fp16) 直接不可用：python
gate decline（报告 §4.2）→ TypeError。根因是 split-D/M4N2 的 O store 用静态
BHND TMA descriptor（`tma_o.get_tma_tensor(make_shape(total_q_rows, kHeadDim))`），
没有 persist-D 那套**运行时 `nhd_out` 分支 + 动态 shape descriptor**
（persist-D 按 `nhd_out` 运行时切换 full/partial tile 的 shape/stride/offset，
`csrc/cuffpa/cute/sm_120/persist_d.cuh` L498-522）。memory `ffpa-nhd-native-layout`
记录了完整模式，照抄即可，无新的数学风险。

#### Design

1. split-D/M4N2 各族 launcher（`launch_cute_fwd_split_d_sm120` /
   `launch_cute_fwd_split_d_m4n2_sm120` 及 fp8/fp4 对应 `_impl`，
   `csrc/cuffpa/cute/launch.cuh`）增加 `bool nhd_out` 参数；O TMA descriptor 改为动态构建：
   - BHND：`get_tma_tensor<H*Br, D>(O)`（现状）；
   - NHD：走 `get_tma_tensor` 的运行时 shape 版（full-tile `(Nq, H*D)`、partial-tile
     `(Nq_tail, H*D)`），`o_ptr` 步长按 `H*D`（照抄 persist-D 的 `nhd_out` 运行时
     分支模式，`csrc/cuffpa/cute/sm_120/persist_d.cuh` L498-522）。
2. python gate 放行：`CUDABackend.forward` 与 `ffpa_attn_fwd` 移除/放宽
   `tensor_layout` 对大 D 的 decline，统一由 C++ 侧按 `nhd_out` 分派。
3. hybrid stage-2 的 fp16 split-D 输出（FC-3 前置）同步接 `nhd_out`。

#### Files & Symbols

- `csrc/cuffpa/cute/launch.cuh`：split-D/M4N2 launcher（`nhd_out` 传参）；
  三格式同名文件的 O store：
  `csrc/cuffpa/cute/sm_120/{split_d,split_d_m4n2}.cuh`（fp16）、
  `csrc/cuffpa/cute/fp8/sm_120/`、`csrc/cuffpa/cute/fp4/sm_120/` 下同名文件；
  参照范本 `csrc/cuffpa/cute/sm_120/persist_d.cuh`（`nhd_out` 运行时分支）。
- `src/ffpa_attn/cuda/__init__.py`：gate。

#### Validation

1. `test_ffpa_nhd_layout.py` 扩 split-D/M4N2 NHD × {fp8 320/768, fp4 320, fp16 320}，
   bit-exact vs 物化参考。
2. bitwise probe 确认 BHND O 写不回归。
3. nsys 核对 kernel 名/数量不变（只是 descriptor 参数化）。

#### Risks & Rollback

- 双分支动态 descriptor 有性能/正确性双重风险——persist-D 已验证同款模式，照抄降风险；
  回退 = 恢复静态 BHND + gate decline。
- M4N2 的 partial Nq O 写需核对 persist-D `nhd_out` 分支的尾 tile 处理
  （`o_rows` guard + `O_gmem_offset`）与现有 split-D 的 O tile 语义一致。

#### Expected Benefit

`tensor_layout='NHD'` 全 D 可用（当前最大可用性缺口之一）。

#### Dependencies

FC-1 先行（同文件调用点，先打通描述符尾参路径，降低一次性改动面）。

---

### FC-3：split-D/M4N2 + hybrid strided 组合

- **Status**: Done（2026-08-28，ffpa-attn 9b9dcae） ｜ **Priority**: F1 ｜ **Track**: 功能（布局）
  实施远小于设计稿：fp16 split-D/M4N2 launcher 与 fp8/fp4 stage-2 impl 经
  FC-1/FC-2 已 layout 原生（从 tensor stride 自检，无需显式传 `Lv`/`nhd_out`）；
  kernel 内 `nhd_out + q_start_row` 组合偏移已在 FC-2 实现；量化链恒做
  full-Q 量化，故 `q_start_row` 与输入布局正交。本项实际改动 = 删两组
  dispatch `TORCH_CHECK`（fp8 D>224 拒 `nhd_in||strided_in`、fp4 D>256 拒
  `strided_in`）+ `prepare_hybrid_stage1` 非 causal 分支的 K/V 族不匹配
  物化兜底（BHND K + strided V 等 mixed 输入对 stage-2 合法但 fp16 stage-1
  要求 `k_nhd==v_nhd`），保证 hybrid 能力 ⊇ 非 hybrid。nsys 确认 dense
  模式 K/V 零物化（每次调用仅 Q_e slice copy + O/lse 写回 3 个 copy kernel）。

#### Motivation

fp8 D>224 / fp4 D>256 时，strided/NHD 输入 + hybrid 组合被 `TORCH_CHECK` 显式拒绝
（报告 §8#3，"requires the persist-D path"）。根因：hybrid stage-1 的 fp16 split-D TMA
输入路径（`prepare_hybrid_stage1`）只在非 causal 分支直传，且烧死 BHND。FC-1/FC-2 打通
独立 `Lv` 与 `nhd_out` 后，此项把组合放开，完成大 D 布局闭环的最后一块。

#### Design

1. `prepare_hybrid_stage1` 接 `Lv`（FC-1 已铺）与 `nhd_out`（FC-2 已铺），让
   fp16-split stage-1 在 strided-NHD 输入下用独立 `Lv` 读、按 `nhd_out` 写中间 buffer。
2. 放宽对应 `TORCH_CHECK`（causal / non-causal 两分支），改为按"独立 `Lv` + `nhd_out`
   齐备"判定可支持性。
3. 保持 hybrid 语义：stage-1 输出仍是 BHND 中间 buffer（stage-2 消费），仅**输入**支持
   strided-NHD，避免 stage-2 量化链再引入布局分支。

#### Files & Symbols

- `csrc/cuffpa/launch.cuh`：`prepare_hybrid_stage1`（L30）、hybrid 主入口
  `launch_ffpa_attn_fwd_template`（L66，调用点 L288-542）与组合拒绝
  `TORCH_CHECK`（"requires the persist-D path"，L443-460）。
- `csrc/cuffpa/cute/launch.cuh`：fp16 split-D/M4N2 launcher（stage-1 kernel，
  `launch_cute_fwd_split_d_sm120` / `launch_cute_fwd_split_d_m4n2_sm120`）。

#### Validation

1. 新用例：strided-NHD + hybrid（fp8 D=320 / fp4 D=320，causal & non-causal），
   parity vs persist-D 参考（报告 §8.3 的 hybrid 验证法）。
2. 负例：`Lv`/`nhd_out` 未齐备的旧调用形态仍按现状拒绝。

#### Risks & Rollback

- stage-1/stage-2 布局语义耦合是主要风险——设计上把布局变化**限制在输入侧**，
  stage-1 输出固定 BHND，控制爆炸半径。回退 = 恢复 TORCH_CHECK。

#### Expected Benefit

大 D + fused-QKV + hybrid 组合可用（当前完全拒绝）。

#### Dependencies

FC-1、FC-2（共用描述符/写路径基建）。

---

## 轨道 F2：量化路径特性对齐（与 fp16 家族对齐）

> 量化家族（fp8/fp4）当前只支持"裸 QKV"，`attn_bias` / `dropout` 全部拒绝（报告 §7.2）。
> `attn_bias`（含 padding/causal-mask 变体）是下游最高频的附加输入，缺它等于把低精度挡在
> 大量真实工作负载之外。

### FC-4：量化路径 `attn_bias`

- **Status**: ✅ Done（2026-08-28） ｜ **Priority**: F2（功能轨内最高频需求） ｜ **Track**: 功能

> **完成记录（2026-08-28）**：fp8/fp4 六族 sm_120 kernel 全部落地。实现要点与设计稿
> 的差异：fp8 注入在 **raw-S 域**（QK GEMM 后、任何 dequant 预乘前），注入
> `bias/(qs_arr[row]*ks*scale_orig)`，四条 softmax 路径的 `qs*ks*scale(LOG2E)` 缩放
> 使其落地为 exp 域 `+bias`；fp4 注入在 dequant 域（blockscale MMA 已折 SF），列须
> `kv_perm32(j)`（K/V^T 置换存储）。注入点位于 masking `-INFINITY` 赋值之前，
> `-inf` 掩码覆盖 bias（无 NaN）。共享 `FfpaBiasParams` helper（cute/launch.cuh）：
> 4-D 校验 + size==1 维 stride-0 broadcast + dtype code（1=half/2=bf16/3=float）。
> launcher 以 `std::integral_constant` tag dispatch 双实例化 `kHasAttnBias`，bias=None
> 编译路径零变化。hybrid 两段各自带 bias（stage-1 fp16 取 bias 前 n_early 行，
> stage-2 fp4/fp8 全量传参）。测试：fp8/fp4 各 5 组（dense/causal_fused/mask 形态/
> GQA/hybrid + dropout 拒绝），fp8 85/85、fp4 61/61、cute 49/49 回归全绿。
> **踩坑**：两处模板实参错位（fp8 persist_d 的 kBiasOn 落入 kPersistQs2r 槽位；
> fp8 split_d_m4n2 非 per-thread 4 变体漏传 kQKPerThread 显式 false）——
> 均为"kBiasOn 挤掉中部带默认值参数、kHasAttnBias 吃默认 0"的静默错位，
> 详见 repo memory `ffpa-fc4-attn-bias.md`。

#### Motivation

fp8/fp4 全体（含 persist-D）拒绝 `attn_bias`（报告 §8#6）。数学上可行（§11.11 归约轴置换
不变性保证 `ΔS` 与 `S` 可同构叠加）：`S = QKᵀ/√d + bias`，只要 bias 在 S 域 fp32 化后
再走 P 量化，就不破坏任何已有不变量。工程量大但路径清晰。收益：解锁低精度 + mask/padding
场景（当前只能退回慢的 fp16）。**这是功能完备性中对用户价值最高的一项。**

#### Design

0. **编译期路径隔离（硬约束）**：bias 支持必须通过**模板参数 / `if constexpr`**
   约束编译路径——量化 kernel 按 `kHasAttnBias` 双变体实例化（照抄 fp16 家族
   `kHasAttnBias×kHasDropout` 编译期 4 变体模式，报告 §4.2），dispatch 按
   `attn_bias.numel()>0` 选实例。**无 bias 时编译路径与现状逐指令一致**（无 bias
   载入、无额外分支、零寄存器开销），确保已有场景数值 bitwise 与性能零影响；
   禁止用运行时分支混入主循环（cute bias 运行时路径曾因此慢 ~2x，报告 §4.2 教训）。
1. **S 域注入点**：attn kernel 内 `S = s * (scale_qk_fp32 * sm_scale)` 之后、`row_max`
   之前，加 `ΔS`（fp32/fp16 → fp32）。需 per-block 从 gmem 载入对应 `bias` 块
   （`Shape<Bq,H,Bkv>` tile），载入通道仅在 `kHasAttnBias` 变体编译。
2. **P 量化前置修正**：fp8 的 per-block P quant scale（`p_scale`）基于 `ΔS - m` 计算，
   bias 天然进入；fp4 的 2688 域 P 量化同理（`sfb*sf_v` 不变，只改输入值）。
3. **causal/tail masking 与 bias 叠加顺序**：`-inf` 掩码必须在 `+bias` 之后，
   否则 `-inf + bias = NaN`（报告 §7.2 的 bias 语义）。
4. python gate 放开 `attn_bias.numel() > 0`（仅 CuTe 量化路径，native 不变）。

#### Files & Symbols

- fp8 主循环 `csrc/cuffpa/cute/fp8/sm_120/{persist_d,split_d,split_d_m4n2}.cuh`；
  fp4 主循环 `csrc/cuffpa/cute/fp4/sm_120/{persist_d,split_d,split_d_m4n2}.cuh`；
  fp4 的 `quantize_and_pack_p` 前置（`csrc/cuffpa/cute/fp4/fp4_pscale.cuh`）；
  `launch_ffpa_attn_fwd_template` 的 `use_bias` 传参（`csrc/cuffpa/launch.cuh`）；
  `src/ffpa_attn/cuda/__init__.py` gate。

#### Validation

1. parity：fp8/fp4 + attn_bias vs fp16 同参数（attn_bias 全支持），cos_sim ≥ 0.999；
   causal + bias 叠加顺序专项（§7.2 负例语义）。
2. rejection 矩阵：quant + bias 从"拒绝"转为"支持"，更新
   `test_ffpa_fp8.py` / fp4 用例。
3. bitwiseprobe 确认 bias=None 路径零变化；并核对无 bias 实例与现 kernel 的
   寄存器数/指令数一致（模板隔离生效的硬指标，ptxas -v 或 NCU 对比）。

#### Post-completion Fix (2026-08-31)

FC-4 完成时漏改 bench CLI：`_runner_fwd.py` 的 attn-mask case 仍带
`not enable_fp8 and not enable_fp4` gate，`ffpa_attn.bench --cuda-impl
fp8/fp4` 输出中没有 attn-mask 行。已放开（fp8/fp4 attn-mask task 现随
全量跑，D=128 parity 通过 2.2x/2.2x）；同时按规范 3 把 fp8/fp4 不支持
的 decode-attn/dropout 在 `_bench.py` task-set 级过滤
（`CUDA_QUANT_EXCLUDED_TASKS`），避免 NaN 空档进入 tflops/speedup plots。

#### Risks & Rollback

- 每 block 多一次 gmem bias 载入（带宽 +1 tile）——对长序列占比小；
- P 量化与 bias 叠加的数值验证是主要工作量；
- 回退 = gate 恢复 `attn_bias.numel()==0` 检查。

#### Expected Benefit

解锁低精度 + mask 场景（当前最大功能差距）。

#### Dependencies

无。

---

### FC-5：量化路径 `dropout` (暂不实施，仅保留设计稿)

- **Status**: Draft ｜ **Priority**: F2（低于 FC-4） ｜ **Track**: 功能

#### Motivation

与 `attn_bias` 同源（报告 §7.2，`dropout_p == 0.0` 拒绝）。但 **dropout 是训练特性，
推理几乎恒为 0**（ffpa cuda backend定位 prefill/推理），故优先级低于 `attn_bias`；若未来有
训练/蒸馏场景再启用。数学上 `O = dropout(softmax(S)) V` 只需在 P 上叠乘性掩码
（Philox RNG），同样须在 P 量化前施加。

#### Design

复用 FC-4 的 S/P 域注入基建：在 `P` 计算后、PV GEMM 前叠 `dropout_mask`
（fp32 乘），再走各自 P 量化。Philox RNG state 随 seed 传入。

#### Files & Symbols

- 同 FC-4 注入点；新增 Philox RNG（可参考 flash-attention 的 dropout 实现）。

#### Validation

- parity：fp8/fp4 + dropout vs fp16 同 seed，统计量比对（均值/方差）；
- dropout_p=0 路径零变化（bitwiseprobe）。

#### Risks & Rollback

- RNG 引入的寄存器/指令开销（推理不用时可完全关闭，编译期分支）。
- 回退 = 保持 `dropout_p == 0.0` 检查。

#### Expected Benefit

训练/蒸馏场景量化路径可用（当前拒绝）。**建议推迟到有真实需求再实施。**

#### Dependencies

FC-4（共用 S/P 注入点）。

---

### FC-6：fp4 smooth_v / MXFP8-PV 扩展至三族

- **Status**: Done（2026-08-28，ffpa-attn 76a8bd8）｜ **Priority**: F2 ｜ **Track**: 功能

> 完成范围与设计稿的差异：**smooth_v** 按设计扩展至三族（split-D/m4n2 补 launcher
> 接线 + per-v_chunk epilogue add-back，量化入口本就带 vm 尾参）；**MXFP8-PV** 实际
> 只扩展到 split-D——split-D 的 PV Tile-K = kBc = 128 恰好等于 MXFP8 atom
> （SM120_16x8x128）的 K extent，persist-D 的全套 mxfp8 机制直接复用（smem 最坏
> D=704 约 95KB < 99KB opt-in）；**m4n2 架构性排除**（atom K=128 > kBc=64，除非
> 把两个 kv tile 融进一次 PV 调用，改动面不成比例），wrapper 保留拒绝并注明理由。
> 顺带修复 latent bug：MXFP8 路径 row_sum 处于 P·448 域，但 lse 修正无条件用了
> NVFP4 的 log2(1/2688)（差 ln 6）；两 kernel 均改为按 kPvMxfp8 选域常量。
> Bench（PRO 5000，D=320 N=16384 split-D）：NVFP4-PV 29.5ms/372T（5.39x SDPA），
> MXFP8-PV 44.8ms/245T（3.56x）——精度 knob 代价约 50%，与 persist-D 行为一致。

#### Motivation

fp4 的 `smooth_v`（精度 knob，§11.8）与 `mx_fp8_pv`（速度 knob，§11.5）当前
`TORCH_CHECK` 限定只在 persist-D（报告 §8#4）。大 D（fp4 split-D/M4N2）用户
无法选用这两个 knob，形成"小 D 有精度/速度选择、大 D 没有"的功能不对称。

#### Design

1. **smooth_v 扩展**：fp4 split-D/M4N2 launcher 的量化链（`launch_fp4_quant_vt_t`
   等，`csrc/cuffpa/cute/fp4/quantize_fp4.cuh`）接 `smooth_v` 分支（fp4 smooth 仅
   影响 `vstats` 与 `v_t` 的 scale，量化链已有 `fp4_smooth_v` 参数，只需放开
   persist-D 的 `TORCH_CHECK`（`csrc/cuffpa/launch.cuh` L321/L361）并在 split-D/M4N2
   quantize 路径透传）。
2. **MXFP8-PV 扩展**：`mxfp8_pv` 需要 V 的 MXFP8 量化（1×32 blockscale，
   `launch_mxfp8_quant_vt_t`）+ PV GEMM 用 e4m3。split-D 的 P 打包
   （`quantize_and_pack_p`，`csrc/cuffpa/cute/fp4/fp4_pscale.cuh`）切到
   `quantize_and_pack_p_mxfp8` 模式，traits 需放开 `static_assert D<=192`
   （M4N2 mxfp8 的 D 上限，按 smem 实算）。

#### Files & Symbols

- `csrc/cuffpa/cute/launch.cuh`：fp4 split-D/M4N2 launcher
  （`launch_cute_fwd_split_d_fp4_sm120` L2618、`launch_cute_fwd_split_d_m4n2_fp4_sm120`
  L2850）与 `fp4_smooth_v` 的 `TORCH_CHECK`；
  `csrc/cuffpa/cute/fp4/quantize_fp4.cuh`：`launch_fp4_quant_{q_t,k_t,vt_t}` /
  `launch_mxfp8_quant_vt_t`（smooth_v 与 MXFP8-PV 的量化入口）；
  `csrc/cuffpa/cute/fp4/sm_120/persist_d.cuh`（参照 persist-D 的 smooth_v 实现）。

#### Validation

1. parity：split-D/M4N2 × {smooth_v, mxfp8_pv} vs persist-D 同配置参考。
2. rejection：`D>192` + mxfp8_pv（M4N2）仍拒绝（static_assert 边界）。
3. 数值：smooth_v 打开后 cos_sim 提升（对照 §11.8 的 outlier 收敛效果）。

#### Risks & Rollback

- MXFP8-PV 在 split-D 的 P 打包路径改动面较大（`nvfp4_pack_p` 双模式）；
- 回退 = 恢复 persist-D 限定。

#### Expected Benefit

大 D fp4 用户获得与小 D 同等的精度/速度配置空间。

#### Dependencies

FC-1 / FC-2（同族布局基建先行，减少一次改动面）。

---

## 轨道 F3：场景与硬件覆盖

> 按需推进的功能完备性项：decode / 短序列、head_dim pad、backward、多架构。
> 单项价值低于 F1/F2，但补齐后覆盖面完整。

### FC-7：短 Nq / decode 量化路径

- **Status**: Draft（⏸ **暂不实施，仅保留设计稿**）｜ **Priority**: F3 ｜ **Track**: 功能
  搁置理由（2026-08-28）：短 Nq/decode 量化**基本没有收益**——量化链固定前处理
  开销（fp4 ~1.1ms）结构性占优，小 Nq 下 attn kernel 本身占比小、量化吞吐优势
  摊不开；decode 场景已由 native split-KV fp16 路径覆盖。短期无必要，仅当出现
  真实短序列量化需求时重新评估。

#### Motivation

量化路径对短序列有结构性开销：fp8 per-block quant 固定 1 个
`min(Bq, kv_chunk_rows)` grid + 多个 aux kernel；fp4 恒有 quantize(2)+smooth+vstats+perm
五连发。Nq 很小时前处理时间 ≈ 甚至 > attn 本身；短序列下 python 还因
`min_seqlen_q < 512` 直接走 SDPA（report §3.4/§4.4）。**推理场景大量短/单行请求被排除在
量化路径之外。** 这是量化家族相对 native（split-KV decode）的场景覆盖缺口。

#### Design

1. **fuse 前处理**：per-block quant + vstats 合并为单 kernel（grid 不变，消除一次
   aux launch）；`hadamard=false` 且非 fused-QKV 时跳过独立的 pad/物化前置步骤
   （直接走 quantize kernel 内的 fused pad 路径，报告 §7.5）。
2. **短 Nq 下跳过 smooth/hadamard**：当 `Bq` 极小且用户未启用时，量化链退化到
   最小集合（quantize + perm 必须，其余可跳）。
3. **放宽 `min_seqlen_q` gate**：前处理开销降下来后，短序列也能量化路径受益——
   用 micro-bench 找新的交叉点（可能 <512）。
4. （可选）decode 专用：Nq=1 时 Q 量化退化为 per-channel，可进一步简化。

#### Files & Symbols

- `csrc/cuffpa/cute/launch.cuh` 量化链（fp8/fp4 launcher impl 内的前处理序列）；
  `src/ffpa_attn/functional.py` 短序列 gate（`8 <= nq < 512 or nkv < 512` decline，
  L239 与 meta fallback，报告 §2.2/§2.3）。

#### Validation

1. micro-bench：Nq ∈ {1,8,64,256,512} 各配置的 quant 前处理 + attn 总时间，
   确认短序列收益为正。
2. parity：短 Nq 量化 vs fp16 同配置。
3. 放宽 gate 后 SDPA↔ffpa 选择点的 e2e 精度抽检。

#### Risks & Rollback

- 短序列收益依赖前处理实际开销占比——**先 profile 再定收益上限**（可能不值得）；
- 回退 = 恢复原 gate 与前处理链。

#### Expected Benefit

解锁推理场景大量短/单行请求的量化路径（当前被结构性开销 + gate 排除）。

#### Dependencies

无。

---

### FC-8：native 路径 head_dim pad

- **Status**: ✅ Completed（2026-08-31） ｜ **Priority**: F3 ｜ **Track**: 功能

#### Motivation

native 路径（fp16 TMA/cp.async）不支持 head_dim pad（报告 §8 脚注），
导致某些非 64 倍数 D 无法走 native 家族（而 CuTe 家族支持 pad）。
当用户需要 native 路径（如 sm90/100 上无 CuTe 量化）但 D 不整时受阻。

#### Design

native 路径的 TMA descriptor 加 pad：`D_padded = ceil(D/64)*64`，
K/V smem 按 `D_padded` 分配，gmem load 按真实 D（TMA 越界元素自然为 0，
不影响 softmax——pad 列在 QKᵀ 中贡献 0，等价于 `S` 不受影响）。

#### Files & Symbols

- `csrc/cuffpa/native/launch.cuh`（host 侧 `CUtensorMap` 构建）、
  `csrc/cuffpa/native/sm_120/split_d.cuh`（TMA kernel 主体）、
  `csrc/cuffpa/native/sm_80/split_d.cuh`（cp.async 对照）。

#### Validation

- parity：pad D（如 D=100/120）native vs 物化 pad 参考；
- bitwise：整 D（无 pad）路径零变化。

#### Risks & Rollback

- pad 列引入的 smem 增量需核对各 D 档 smem 上限；回退 = 维持拒绝。

#### Expected Benefit

native 家族覆盖非整 D（补齐与 CuTe 家族的对齐）。

#### Dependencies

无。

#### Completion Record (2026-08-31)

实现与设计稿的差异（零物化方向）：

- **api 层**（`csrc/cuffpa/ffpa_api.cc`）：AUTO/NATIVE/TMA 三 hint 纳入
  `needs_pad`，pad 目标为下一个 **64 倍数**（native 家族编译集
  `range(64,1025,64)`），范围校验 [64,1024]；`head_dim_dispatch` 三路
  （fp4 64 对齐 / native 64 对齐 / 其余 32 对齐）；O 仍由 api 层 pad +
  narrow 回切。TMA hint 仅在 `ENABLE_FFPA_TMA_EXT` 且 sm90+ 时计入
  native 家族（镜像 launcher 分派：pre-sm90 / 无 TMA ext 时回落 CUTE
  sm80 kernel，仍走 32 对齐 torch pad）。
- **launcher**（`csrc/cuffpa/launch.cuh`）：`native_kernel_pad =
  force_native || tma_kernel_active`（后者 `#ifdef ENABLE_FFPA_TMA_EXT` +
  `major>=9` 双重限定，防 TMA-ext 未编译时未物化 QKV 泄漏进 cute sm80
  kernel 的静默数据错乱）；`qkv_padded` 排除 native 家族——Q/K/V 保持
  D_og 宽，不再做 `constant_pad_nd` 物化拷贝。
- **kernel 侧零物化**：
  - sm80 cp.async（`native/prefill.cuh` + `sm_80/split_d.cuh`）：
    `cp_async_qkv_g2s` 增加 `d_og` 行 stride + 16B chunk 列守卫
    （`cp_async_zfill` src-size=0），17 个调用点全部显式传参
    （移除默认参数，编译器强制覆盖）；`split_d_fwd_sm80` O 偏移与
    QKV 偏移解耦（O 恒为 kHeadDim 宽）。
  - sm80 decode split-KV（`sm_80/split_kv.cuh`）：s1 全部 5 处 gmem
    载入改 zfill + `d_og` stride；kStage==1 直读分支加越界零向量守卫。
  - sm90+ TMA（`native/launch.cuh`）：`make_desc` 的
    `minor_dim`/`major_stride_bytes` 改用运行时 `d_og = Q.size(3)`，
    TMA OOB 自动零填充 pad 列（无需 kernel 改动）。
- **约束**：D_og%8==0（16B stride 对齐 + 16B chunk 列守卫粒度）；
  `--headdim all` 全量编译集 {64..1024 step 64} 16 档无空洞。
- **验收**（规范 3，`ffpa_attn.bench` CLI 全链路）：
  - `--cuda-impl native --D 328 --tasks self-attn,cross-attn,decode-attn,gqa,causal,non-aligned` 通过（328→384）；
  - `--cuda-impl native --D 120`（`FFPA_CUDA_ALLOW_SMALL_D=1`，120→128）全 task 通过；
  - `--cuda-impl tma --D 120/328` 通过（TMA OOB 零填充路径，D=328 self-attn 2.14x）；
  - 回归：`test_ffpa_fwd.py` 503 passed（新增 `test_native_head_dim_pad_*`
    40 项全过；`test_ffpa_fp8.py` 85 / `test_ffpa_fp4.py` 61 全过）。
    当时的 `triton_small_d_default_falls_back_to_sdpa` 1 项失败后经查为
    shell env 污染（残留 `FFPA_TRITON_ALLOW_SMALL_D=1`）+ 测试未自封闭，
    非代码缺陷；已加 `monkeypatch.delenv` 修复（06e23fe），全量 504 passed 0 failed。

---

### FC-9：CUDA backward (暂不实施，仅保留设计稿)

- **Status**: Draft ｜ **Priority**: F3（长期/评估） ｜ **Track**: 功能

#### Motivation

CUDA backend 完全无 backward（`CUDA_BWD_AVAILABLE=False`，报告 §6.4）。
本库定位是**推理 / prefill**，训练路径活跃在 Triton backend（`backward=True`）。
但若量化推理需要与训练共享数值路径，或未来有低精度训练需求，backward 是功能完备性的
最终缺口。

#### Design

**建议先做定位评估再决定是否实施**：

1. 调研下游是否有量化训练/蒸馏的真实需求（当前推理为主，可能长期不需要）。
2. 若实施：backward 需要保存 softmax 统计量（LSE）+ 重算，工程量远大于 forward；
   可先支持 fp16 backward（复用 FA-2 的 bwd 结构），量化 backward 更长期。

#### Files & Symbols

- 新增 `csrc/cuffpa/*bwd*`；`src/ffpa_attn/cuda/__init__.py` 的
  `CUDA_BWD_AVAILABLE`。

#### Validation

- parity：CUDA bwd vs Triton bwd vs torch sdpa（数值梯度检查）。

#### Risks & Rollback

- 工程量极大（forward 的 3-5 倍）；**建议仅在确认真实需求后立项**。

#### Expected Benefit

训练/蒸馏场景 CUDA 量化路径（当前完全无）。

#### Dependencies

需求确认（下游调研）。

---

### FC-10：sm90 / sm100 量化覆盖 (暂不实施，仅保留设计稿)

- **Status**: Draft ｜ **Priority**: F3（长期） ｜ **Track**: 功能

#### Motivation

量化路径全部限定 `prop->major == 12`（sm_120）：fp8/fp4 在 sm90/sm100 直接拒绝
（报告 §3.4/§4.4）。fp16 TMA 在 sm90/100 也仅 "Unverified"。本库当前深度绑定
Blackwell 消费/专业卡；若目标硬件扩到 H100/B200，量化路径不可用。

#### Design

1. **sm90（H100/H800）**：CuTe 的 TMA + WGMMA 可用，但 `setmaxnreg` / cp.async.bulk
   细节需适配；`tcgen05` 不用（sm90 无），用 WGMMA。工作量中等。
2. **sm100（B200）**：架构接近，主要是编译宏与 arch 参数扩展。
3. native TMA 路径（fp16）在 sm90/100 实测补齐（报告 §3.4 标注 "Unverified"）。

#### Files & Symbols

- 全量化路径的 `TORCH_CHECK(prop->major == 12, ...)`（launch.cuh 多处）；
  build.sh `--arch` 扩展；`#if __CUDA_ARCH_FAMILY_SPECIFIC__` 条件编译。

#### Validation

- 目标硬件实测（sm90/sm100 机器）：功能正确性 + 性能基线；
- 本库当前无 sm90 测试机，需先确认硬件可得性。

#### Risks & Rollback

- 依赖目标硬件可得性；sm90 WGMMA 与 sm_120 的 MMA 路径差异需要独立调优；
- 回退 = 维持 `major==12` 检查。

#### Expected Benefit

量化路径覆盖 H100/B200（当前仅 sm_120）。

#### Dependencies

目标硬件可得性确认。

---

### FC-11：native 路径 dropout 精度修复

- **Status**: ✅ Done（2026-08-31，ffpa-attn 542f774/e1fe363） ｜ **Priority**: F3（bug 修复，高优） ｜ **Track**: 功能/正确性

#### Motivation

native 家族（AUTO/NATIVE cp.async 与 sm90+ TMA）的 dropout task 在 bench CLI
上 parity 失败（RTX PRO 5000，B=1 H=32 N=16384，dropout_p=0.1）：

| 配置 | O_err | allclose |
|---|---|---|
| `--cuda-impl fp16` D=120 fp16/bf16（atol 0.02/0.05） | 0.0629 / 0.0630 | **False / False** |
| `--cuda-impl fp16` D=128 fp16/bf16 | 0.0779 / 0.0775 | **False / False** |
| `--cuda-impl tma`  D=120 fp16/bf16 | 0.0629 / 0.0630 | **False / False** |
| `--cuda-impl tma`  D=128 fp16/bf16 | 0.0779 / 0.0776 | **False / False** |

同批 run 的 self-attn / causal / attn-mask / non-aligned / decode / gqa /
cross 全部通过（O_err ≤ 0.005）→ 问题**仅限 dropout**。

#### Root Cause（已验证，根因反转）

三个相互独立的因素叠加，最终结论是 **FFPA 源码正确，参照系（本地 torch
2.11.0 mem-efficient SDPA）在 >2^32 score elements 时自身 uint32 回绕**：

1. **bench 实测的 native FAIL 主体是 stale `.so`**：`_C*.so`（03:07 编译）
   落后于 `prefill.cuh`（04:53 `git pull` 拉入 #351 更新），
   `rm -rf build src/ffpa_attn/_C*.so` 全量重编后 native 恢复正确
   （源码 `sync_apply_dropout_to_p` 全链路 u64，#199 起即正确）；
2. **Triton 侧确有真 bug（已修）**：`_apply_dropout_to_p` /
   `_ffpa_decode_fwd_stage1_kernel` / `_dropout_multiplier` 三处的
   `linear = off_hb * seqlen_q * seqlen_k + ...` 在 int32 域计算
   （`program_id`/`offs_*` 均 32 位），B\*H\*Nq\*Nkv > 2^31 即回绕；
   统一 `.to(tl.int64)` 提升后修复（< 2^31 数值逐位不变，
   stash 基线对比证实）；
3. **torch 2.11.0 mem-eff 参照系 bug**：`kernel_forward.h`
   `advance_to_block()` 中 `batch_id * num_heads * num_queries * num_keys`
   的 `num_*` 全是 `int32_t`、`blockIdx` 分量为 32 位 unsigned → 乘法在
   **uint32 域**完成后才赋给 u64。边界矩阵实证：max linear index =
   2^32−1 的三个 shape（B1 H16 N16384、B2 H32 N8192 等）全部 PASS，
   仅 2^33−1（B1 H32 N16384）FAIL，per-head 图显示 sdpa 恰从 head 16
   （base = 2^32）起分歧。PyTorch main 已修（`gemm_kernel_utils::
   dropout_rng_offset` 全 int64_t helper，注释明确 "keeps num_queries *
   num_keys from wrapping at 2^32"）。

修复后五路交叉验证（native / tma / cute / cute_tma / triton，
B1 H32 N16384）：互相 max_err ≤ 1e-4，与 sdpa 一致地 0.0452（仅
head 16+ 分歧，即 ref bug）。bwd 组合（cuda-fwd + triton-bwd）与全
triton 的 dQ/dK/dV 互差 ≤ 1.2e-4 → **fwd/bwd mask replay 自洽**。

#### Design（实际落地）

1. Triton 三处 RNG offset 计算 int64 化（helper 内统一 cast，
   覆盖 sm80/sm90 所有调用点）；
2. bench 侧：runner 检测 `dropout_p > 0 且 B*Hq*Nq*Nkv > 2^32` 时打印
   `[warn]`、行内标注 `[torch-ref-2^32-bug]`、Markdown marker 渲染 ⚠️
   而非 ❌（torch 修复版落地后自动恢复正常判定）；
3. native 侧无需改动（u64 源码正确，stale 二进制问题）。

#### Files & Symbols

- `src/ffpa_attn/triton/_ffpa_fwd.py`（`_apply_dropout_to_p`、decode
  stage1 内联 dropout）
- `src/ffpa_attn/triton/_ffpa_bwd.py`（`_dropout_multiplier`）
- `src/ffpa_attn/cli/_runner_fwd.py`、`_runner_bwd.py`、`_bench.py`
  （ref-bug 检测与标注）
- `tests/test_ffpa_fwd.py`（三个大 N 回归测试）

#### Validation（已通过）

- `pytest -k 'dropout_large_n or cross_consistent'`：3 passed；
  stash 修复后 triton 大 N 测试按预期 FAIL（证明可捕获回归）；
- `ffpa_attn.bench --backend cuda/triton --D 128 --N 8192` 全任务
  allclose=True（triton bwd dropout dQ_err 与修复前基线逐位一致 →
  无回归）；
- `--N 16384 --tasks dropout`（fwd/bwd）：输出 `[torch-ref-2^32-bug]`
  标注；`--N 8192` 无误标。

#### Risks & Rollback

- bench 的 ⚠️ 标注条件精确限定于 dropout + >2^32（该条件下 ref 数学上
  必坏），不会掩盖 FFPA 其它缺陷；回退 = revert 两个 commit。

#### Expected Benefit

dropout RNG 语义与 PyTorch main 对齐；N=16384 dropout 不再误报回归；
大 N 回归防线建立（2^32 边界 SDPA-parity + 2^33 交叉一致性）。

#### Dependencies

无（与 FC-5 量化 dropout 设计稿独立）。附带发现未处理项：
tma+dropout 112ms 性能异常（见上"附带发现"，归属后续性能 RFC）。

---

## 轨道 P：性能优化（功能不变下提速）

> 轨道 F 解决"能不能用"，轨道 P 解决"快不快"。以下各项在功能完备的前提下推进。

### PC-0：attn mask 场景性能优化（bias tile IO 重构）

- **Status**: Draft ｜ **Priority**: P0（性能轨最高优先） ｜ **Track**: 性能

#### 子项分解（场景拆分，可独立推进/验收/回退）

父项按 kernel 家族拆三个子项，共享同一根因诊断（bias 标量 IO 的 cache line
级重复 + 低效 sector 利用），但注入点/基建不同，故分别设计与验收：

- **PC-0-0：cute/cute_tma 场景**（fp16 cute 家族，`apply_attn_bias_rowcol`）——
  方案 A（bias tile smem 预载）的最小验证场：无 raw-S 域反量化、无 fp4
  `kv_perm32` 置换、无 fp8 `inv_sd` 乘法，先在此跑通 smem tile + `tScS_rc`
  坐标读 + 流水预取，作为 PC-0-1 的热身台阶。
- **PC-0-1：fp8/fp4 场景**（量化六族：fp8/fp4 × persist-D/split-D/M4N2）——
  原 PC-0 主体设计（下方 Motivation/Design/Files/Validation 均归属此子项），
  在 PC-0-0 基建上叠加 raw-S 域注入、perm32 列置换、inv_sd 折叠三个量化特化。
- **PC-0-2：native/native_tma 场景**（`prefill.cuh` 标量 loader
  `load_attn_bias_value`，R_S fragment + broadcast-stride 语义）——已完成
  （c645ac1）：NCU 复核确认退化是**指令数瓶颈**（非带宽），故选内联向量化
  而非 tile 化——rowvec 门控（stride_m==0/Nq==1 + stride_n==1 + 对齐）下
  `half2`/`float2` 对加载服务全部 4 个 fragment 槽，消除 16x load 冗余；
  残余为 load-latency 主导（预取受寄存器预算约束搁置），见完成清单。

#### Motivation

FC-4 打开了量化路径的 `attn_bias`，但注入实现（`apply_attn_bias_quant_rowcol` /
`apply_attn_bias_fp4_rowcol`，均调 native 的标量 loader `load_attn_bias_value`）
使 attn-mask 成为量化路径**最大的性能退化点**（RTX PRO 5000，B=1 H=32 N=16384）：

| 场景 | self-attn | attn-mask | 退化 |
|---|---|---|---|
| fp4 D=128 fp16 | 8.16 ms / 539T | 30.55 ms / 144T | **3.74x** |
| fp4 D=128 bf16 | 8.17 ms / 538T | 26.23 ms / 168T | 3.21x |
| fp8 D=320 fp16 | 38.36 ms / 287T | 68.16 ms / 161T | 1.78x |

fp4 D=128 的纯 bias 开销 ≈ 22.4 ms；16384² fp32 mask ≈ 1 GB → 有效带宽仅
~45 GB/s（<3% DRAM 峰值）。fp4 越快（NVFP4 MMA 主循环 539T）标量 bias 注入
占比越大，退化越狠。

#### Root Cause（初步分析，动手前 NCU 复核）

1. **cache line 级重复 IO（主因）**：cute rowcol score fragment 坐标散布，
   同一 128B line 的 bias 元素被不同 thread、不同循环迭代的**独立标量 load**
   分别取出；1 GB mask 远超 L2（96MB）→ line 反复从 DRAM 拉取。
   fp4 的 `kv_perm32` 列置换（j → 0,1,8,9,16,17,...）进一步打散跨迭代局部性。
2. **低效 IO**：逐元素 32-bit 标量 load，warp 内地址不连续 → 无合并，
   32B sector 只取 4B（sector 效率 ~12.5%）；广播 mask（stride_n==0 等）也
   逐元素重复读同一地址。
3. 计算顺带浪费：每元素运行时 dtype 三分支 + `long long` 地址算术 +
   fp8 的 `inv_sd[row]` 逐元素乘（可 tile 级预乘）。

#### Design（重点：消除重复 IO + 高效 IO；仅向量化不够）

**方案 A（主案，数学不变）——cute 专属 bias tile smem 预载**：

- 为 cute 家族**单独实现** bias tile 加载（TMA 2D box / cp.async 向量化），
  **不复用 native/prefill.cuh 的标量函数**（cute 有自己的 TMA/Tensor 基建；
  native 的 R_S fragment + broadcast-stride 语义留在 native 侧）。
- per (Q-tile, KV-tile) 把 `[Br, Bc]` bias 块一次性载入 smem（行连续 →
  sector 满载），与 K/V tile 同流水异步预取（latency 隐藏）；注入函数改为
  按 `tScS_rc` 坐标读 smem Tensor。**每 bias 元素、每 cache line 从 DRAM
  只取一次**：DRAM 流量降到理论最小 `Nq·Nkv·sizeof(dtype)`，line 一次取满。
- 广播特化在 loader 侧：`stride_n==0` 载 `[Br,1]` 单列、`stride_m==0` 载
  `[1,Bc]` 单行、h/b 广播由 grid 索引天然处理——广播 mask 的重复读归零。
- fp4 适配：smem 存原序列，读时按 `kv_perm32(j)` 索引（与既有 masking
  同构）；是否载入侧预置换由实现时 A/B 定。
- fp8 raw-S 域：`inv_sd[row]` 折叠加法不变，仅换 load 来源；可选把
  `bias·inv_sd` 预乘后以 fp16 存 smem（流量减半，精度需验证）。

**方案 B（进阶，可选）——乘子域预折叠（数学等价变换）**：
`softmax 输入 = S_dequant + bias`，则 `exp(S' − m) = exp2((S_dequant − m̃)·c) ·
exp2(bias·c)`。预处理 kernel 一次性产出 `bias_exp` 乘子表（半精度）+ per-row
`max(bias)` 作 online-max 初值上界（防溢出，softmax 平移不变保证等价）；
fp8 的 `qs·ks` 在 softmax 输入域恰好消去 → 乘子与 row 无关，per-(q,k) 常量表。
attn kernel 内 add 变 mul、读表流量减半。风险：online-max 初始化语义变化 +
半精度乘子表精度，A 落地后再评估。乘子表预处理可并入 PC-1 的 aux 融合。

**方案 C（仅根因对照，非交付路径）**：stride_n==1 时 128-bit 向量化 direct
load——按用户判断"仅向量化不够"（line 级重复仍在），只用于 A/B 对照验证
sector 效率诊断，不作为交付方案。

**验证顺序**：NCU 先行（attn-mask kernel 看
`l1tex` sector/request 比与 DRAM 带宽利用率，预测 <15% / <10%）→ A 落地 →
复核指标归位。

#### Files & Symbols

- `csrc/cuffpa/cute/attn_bias.cuh`（新增 cute 原生 bias tile loader；
  `apply_attn_bias_quant_rowcol` 的 gmem 直读路径退役为 fallback）
- `csrc/cuffpa/cute/fp4/fp4_gemm.cuh`（`apply_attn_bias_fp4_rowcol` 改 smem 读 + perm32）
- `csrc/cuffpa/cute/fp8/sm_120/{persist_d,split_d,split_d_m4n2}.cuh`、
  `cute/fp4/sm_120/{persist_d,split_d,split_d_m4n2}.cuh`（注入点接线；
  fp16 cute 家族 `apply_attn_bias_rowcol` 同模式受益，可随做）
- `csrc/cuffpa/cute/launch.cuh`（bias 描述符 / smem 布局 / 流水接线）

#### Validation

- parity：`ffpa_attn.bench --cuda-impl fp8/fp4 --tasks attn-mask` 前后 O_err
  不变（容差同 FC-4：fp8 5e-2 / fp4 0.15）；六族 kernel 全过；
- 性能：attn-mask 与 self-attn 差距 3.74x(fp4 D128) / 1.78x(fp8 D320) 收敛至
  ≤1.2x；attn-mask TFLOPS 回到同 D self-attn 的 80%+；
- NCU：标量 global load 消失，sector/request 与 DRAM 带宽利用率归位；
- 广播 mask 专项（stride 0 各维）与尾 tile（Nq/Nkv 非 tile 倍数）正确性。

#### Risks & Rollback

- smem 预算：+bias tile `[Br,Bc]`（fp32 最坏 ~64KB）——按档位裁剪（fp16 存 /
  从 stage 数腾 / 大 D 档 fallback 直读路径保留）；
- TMA box 形状：Nq/Nkv 非 box 整数倍的边界（OOB bias 读 0 与 kv_mask -inf
  屏蔽交互需验证）；broadcast stride 无法直接 TMA → 单行/单列特化；
- 回退 = `kHasAttnBias` 双实例保留直读路径，逐 kernel 切换可独立回退。

#### Expected Benefit

attn-mask 量化路径从 144T（fp4 D128）回到 ~400T+ 量级；对齐 SageAttention
等竞品在 mask 场景的竞争力（当前 attn-mask 是量化路径唯一 >1.8x 退化点）。

#### Dependencies

FC-4 注入点基建（`kHasAttnBias` 编译期双实例，bias=None 零开销不变）；
与 PC-1 正交（方案 B 的乘子表可并入其 aux 融合）。

---

### PC-1：Mega Quantize Kernel（aux 链大融合）

- **Status**: Draft ｜ **Priority**: P1（aux 链基建） ｜ **Track**: 性能

#### Motivation

fp8/fp4 前处理链当前是 8-13 个串行小 kernel：fp8 `pad→vstats→vt(+pad)→quantize
→(bias)→(smoothQ/K/V)→permute`；fp4 `quantize(2)→smooth→(hadamard)→vstats→vt→permute`。
开销有两类：每 kernel 的 launch/dispatch（CPU dispatch wall-GPU 129µs vs sage 50µs，
报告 §5.3），以及中间结果的 gmem round-trip（每阶段写回 gmem 再读回，带宽受限）。
**multi-stream 并行 aux 链不可行**：依赖图跨分支的并行窗口小，小 kernel 在消费卡上
无法互饱 SM，反而叠加 event 同步与 launch 开销——并行化 launch 省不掉 gmem 往返。
真正可行的方向是**把所有预处理与量化融合进单个巨型 kernel（Mega Quantize Kernel）**，
中间结果留 smem/寄存器，消灭 round-trip 与 dispatch。

#### Design

1. **两阶段 + grid 级屏障**：phase-1 各 threadblock 并行算 vstats/smooth 的 per-channel
   Partial 统计写 workspace；`cooperative groups grid.sync()` 全局完成；phase-2 用定稿
   scale 做 quantize/permute/vt 写出。附录 A #18 证伪的是"无同步单遍融合"（per-channel
   scale 必须先完成），两阶段屏障正是其解法。
2. **smem/寄存器承中间结果**：V tile 一次进 smem，vstats partial、quantize、vt 写出
   复用同一 tile，V 的 gmem 流量从 3+ 遍降到 1 读 1 写。
3. **launch 形态**：cooperative launch 要求 grid ≤ 常驻 CTA 数；aux 链 grid 大时改
   两阶段拆两个 cooperative kernel 或 persistent kernel + tile 级屏障，实测定。
4. **数值硬约束**：scale 计算顺序不变，输出与现有链 bitwise 一致。

#### Files & Symbols

- `csrc/cuffpa/cute/launch.cuh`：fp8/fp4 launcher impl 内的前处理调用序列
  （fp8 自 `launch_cute_fwd_persist_d_fp8_sm120_impl` L811 起：smooth_k →
  Q/K quantize → V quantize/VT；fp4 自 `launch_cute_fwd_split_d_fp4_sm120_impl`
  L2393 起：qm/km → quantize → delta_s）。
- 前处理 kernel 定义：`csrc/cuffpa/cute/fp8/{smooth_k,smooth_v,quantize_fp8}.cuh`、
  `csrc/cuffpa/cute/fp4/{quantize_fp4,delta_s}.cuh`、`csrc/cuffpa/cute/hadamard.cuh`；
  新增 `mega_quantize.cuh`。

#### Validation

1. **bitwise 8 场景 probe**（附录 B）：融合前后输出完全一致（硬约束）。
2. **冷数据轮转 bench**（附录 B）：fp8 D=128/320 × N∈{1024,4608,16384}；
   **必须报告卡型 + 冷热条件**（报告 §2.2）。
3. nsys：kernel 数 13 → ≤3（mega + 不可避免残量），aux 链时间与 dispatch gap 下降。

#### Risks & Rollback

- cooperative launch 的 grid 限制与 grid_sync 开销；短序列若负收益需 per-N 门控。
- smem 预算（V tile + partials）按 D 档核对。
- 回退 = 保留原串行链（feature flag）。

#### Expected Benefit

前处理 gmem 流量与 dispatch 开销大削；长序列总时间 -3~8%（依 aux 链实测占比），
CPU dispatch gap 向 sage 水平收敛。

#### Dependencies

无（两阶段屏障形态需先做 cooperative launch 原型验证）。

---

### PC-2：增量融合（Mega Kernel 步进）

- **Status**: Draft ｜ **Priority**: P2 ｜ **Track**: 性能

#### Motivation

PC-1 的巨型 kernel 改动面大；先落地无全局依赖的相邻融合对，每步独立验收收益，
同时为 mega kernel 积累 smem tile 复用模式。若 PC-1 直接落地，本项被其收编。

#### Design

1. **pad+quantize 融合**：零填充与 scale 计算合并，中间 pad 结果不落 gmem。
2. **smooth_v 后置合并**：smooth 的 scale 应用并入消费它的下一个 kernel。
3. **kv_mean 融进 quantize**：跨块全局依赖，需重新评估原子/屏障成本（报告 §9.1 方向 2）。
4. 保持输出布局/数值不变（融合只改执行方式）。

> vstats+vt 不进增量融合（附录 A #18），留给 PC-1 的两阶段屏障解决。

#### Files & Symbols

- `csrc/cuffpa/cute/launch.cuh` 的前处理调用序列与
  `csrc/cuffpa/cute/fp8/{quantize_fp8,smooth_k,smooth_v}.cuh`、
  `csrc/cuffpa/cute/fp4/{quantize_fp4,delta_s}.cuh` 的 aux kernel 定义，
  需新增融合版。

#### Validation

1. bitwise probe：融合前后输出一致。
2. 冷数据轮转：量化前处理时间分项对比。
3. nsys kernel 数统计（目标：13 → <10 阶梯下降）。

#### Risks & Rollback

- 融合 kernel 的寄存器压力可能上升（两合一）；需核对无 spill。
- 回退 = 保留原独立 kernel 路径。

#### Expected Benefit

每对融合省一次 gmem round-trip + 一次 launch；前处理 -10~20%。

#### Dependencies

被 PC-1 收编（作为其增量落地台阶）。

---

### PC-3：形状/卡型感知的量化配置自适应

- **Status**: Draft ｜ **Priority**: P3 ｜ **Track**: 性能

#### Motivation

kernel 家族 dispatch 目前**纯按 headdim**（D≤224 persist-D / 224<D<768 split-D
M8N1 / D≥768 M4N2，`csrc/cuffpa/launch.cuh`），无 N 维分支。历史上有两类与 N 相关
的配置反转记录，须先甄别再谈自适应：

1. **量化 knob 反转**（报告 §5.3）：`fp8_qk_mm_type/pv_acc_type` 的 crossover
   N∈(4608,8192)。注意默认配置已统一为 QK int8 + PV f16 acc，该反转的前提已变，
   须在新默认下复测。
2. **家族反转**（旧稿记录，**未证实**）："N=1024 时 persist-D 比 split-D 快 3%、
   N=2048 反转"——旧稿称存在静态阈值 `N<4096→persist-D`，但代码中**不存在**该阈值
   （dispatch 纯 D 决定），此记录需重新实测确认；若属实，引入的是**新的**运行时
   N 分支，而非修既有逻辑。

#### Design

1. cold bench 复测：新默认配置（QK int8 + PV f16 acc）下按 (卡型, D, N) 全矩阵跑
   knob 反转与家族反转，确认哪些真实存在（避免在噪声或过时记录上建表）。
2. 确有反转：建实测查找表（带**卡型键**），在 `launch_ffpa_attn_fwd_template`
   （headdim dispatch 处）引入运行时分支按表选择。
3. 默认保守：查不到表项按现状（纯 D dispatch），避免误伤。

#### Files & Symbols

- `csrc/cuffpa/launch.cuh`：`launch_ffpa_attn_fwd_template`（L66）的 headdim
  dispatch 分支（persist-D/split-D/M4N2 选择点）；查找表可用静态数组或编译期常量。
- `csrc/cuffpa/cute/launch.cuh`：各族 launcher 入口（若需家族内 knob 切换）。

#### Validation

1. cold bench 全矩阵复测（附录 B 冷数据轮转），反转区须双卡复现。
2. parity：配置切换不改变数值（同配置下 bitwise 一致）。

#### Risks & Rollback

- 查找表的卡型依赖（5090 vs PRO 5000 可能不同）——表要带卡型键。
- 回退 = 恢复纯 D dispatch。

#### Expected Benefit

若反转实测存在：小 N 场景 ~3%（量级以复测为准）。

#### Dependencies

无（但依赖冷数据轮转基建，附录 B）。

---

### PC-4：fp4 persist-D attn kernel 内部优化

- **Status**: Draft ｜ **Priority**: P3 ｜ **Track**: 性能

#### Motivation

fp4 persist-D attn kernel 当前 `wait` stall 31.1% 为主（报告 §6.5），
tensor pipe 未饱和（`math_pipe_throttle` 仅 7.4%），说明数据供给（Q/K/V smem→reg、
P pack、rescale）是瓶颈而非 MMA 吞吐。潜在方向：减少 P pack 的寄存器往返、
优化 rescale 的 FFMA 依赖链、提升 TMA→smem→reg 的供给速率。**但必须先证明热点**
（附录 B 的 NCU stall 采样），避免在已饱和处做无用功。

#### Design

1. **先决**：`ncu --page source` 定位 fp4 attn kernel 的 stall 热点行
   （`wait` 31.1% 集中在哪条指令链）。
2. 按热点选方向（候选）：
   - P pack（`quantize_and_pack_p`）的寄存器压力优化；
   - rescale 与 MMA 的依赖解耦；
   - smem→reg 的 load 合并。
3. 每次改动用 NCU A/B 验证（附录 B）。

#### Files & Symbols

- `csrc/cuffpa/cute/fp4/sm_120/{persist_d,split_d,split_d_m4n2}.cuh` fp4 主循环；
  `csrc/cuffpa/cute/fp4/fp4_pscale.cuh`（`quantize_and_pack_p` /
  `quantize_and_pack_p_mxfp8` 的 P pack）。

#### Validation

1. NCU stall 画像前后对比（`wait` 占比下降）。
2. 冷数据轮转性能。
3. bitwise probe 数值一致。

#### Risks & Rollback

- fp4 kernel 微优化空间可能已接近上限（参照附录 A 元教训：微优化到顶）；
  **先 NCU 证明有可攻克的热点再投入**。
- 回退 = 逐项保留原路径。

#### Expected Benefit

fp4 attn kernel -3~5%（若热点可攻克）。

#### Dependencies

无（但强依赖 NCU 先决分析）。

---

### PC-7：fp8 split-D (M8N1) 量化大 D kernel 性能优化

- **Status**: Draft ｜ **Priority**: P2 ｜ **Track**: 性能

#### Motivation

fp8 split-D (M8N1, 224<D<768，代表 D=320/512) 是量化大 D attention 的薄弱点
之一：non-WS 结构，大 D 下 `o_acc=D/2` per-thread 寄存器（D=512 → 256 regs，
逼近 255 硬上限，spill 到 local mem），QK→softmax→P 量化→PV 串行依赖链长。
同族 persist-D（D≤224）已被深度打磨（WS / non-WS 双版本 + 多轮局部优化），
其中验证有效的局部优化方案与 split-D 主循环结构无关，可移植。

#### Design

1. **先决（硬）**：`ncu --page source` 拿到 D=320/512 的 stall / roofline /
   occupancy 画像，确认热点结构；**无画像不动手**（同 PC-4 纪律）。
2. 逐项移植 fp8 persist-D 的局部优化菜单：reg reconfig 参数、NamedBarrier
   数量/位置、producer TMA issue 顺序（K/V 先后与 prefetch 深度）、softmax
   MUFU、int8 s32→f32 cast 向量化——每项 ≤ 几十行、不动主循环骨架。
3. persistent work loop **不假设可移植**：fp8 persist-D 已证伪零收益
   （附录 A #3），split-D 上须独立实测后才能定论。
4. 编译期路径隔离，默认配置行为零变化。

#### Files & Symbols

- `csrc/cuffpa/cute/fp8/sm_120/split_d.cuh`；
- 参照范本：`csrc/cuffpa/cute/fp8/sm_120/persist_d.cuh`（WS / non-WS 双版本，
  已落地各优化项）。

#### Validation

1. NCU stall / roofline 画像前后对比（附录 B）。
2. 冷数据轮转（附录 B）：fp8 × D∈{320,512} × N∈{1024,4608,16384}，
   **必须报告卡型 + 冷热条件**。
3. bitwise probe 数值一致（优化只改执行方式、不改结果，硬约束）。
4. 动手前查附录 A 证伪边界（#3 persistent、#4 stages 加深等）。

#### Risks & Rollback

- fp8 可能同为 latency-bound、可挖空间有限（参照
  `ffpa-fp4-125x-ceiling-analysis` 的收益上限定量方法）；
- 结构优化收益不能跨结构外推（附录 A 元教训）——每方案独立 A/B；
- 回退 = 逐项恢复（每方案独立 flag / 分支）。

#### Expected Benefit

按方案实测（参考量级：局部优化单项 0.5~2%）。

#### Dependencies

—（PC-7/8/9/10 四部曲首项，严格逐个推进；强依赖 NCU 先决分析）。

---

### PC-8：fp8 split-D M4N2 量化大 D kernel 性能优化

- **Status**: Draft ｜ **Priority**: P2 ｜ **Track**: 性能

#### Motivation

fp8 split-D M4N2（D≥768，代表 D=768/1024）：kBr=64、atom_layout=(4,2,1)、
`o_acc=D/4`（寄存器压力为 M8N1 的一半，正是 D≥768 的 spill 解法），但 MMA 形状
与 kv 循环结构与 M8N1 不同，热点结构须独立画像，收益不能从 PC-7 直接外推。

#### Design

同 PC-7 方法论（NCU 先决 + 局部优化菜单 + 附录 A 证伪边界核对）。fp8 两 kernel
共享量化链与 softmax 结构，PC-7 验证有效的方案优先复测移植。

#### Files & Symbols

- `csrc/cuffpa/cute/fp8/sm_120/split_d_m4n2.cuh`；
- 参照范本：同 PC-7。

#### Validation

同 PC-7 套件，bench 代表点改 D∈{768,1024}；每方案独立 A/B、独立回退。

#### Risks & Rollback

同 PC-7（M4N2 寄存器压力更低，latency-bound 特征可能更显著，收益上限先定量）。

#### Expected Benefit

按方案实测。

#### Dependencies

PC-7（顺序：复杂优化逐个推进，同族经验复用）。

---

### PC-9：fp4 split-D (M8N1) 量化大 D kernel 性能优化

- **Status**: Draft ｜ **Priority**: P2 ｜ **Track**: 性能

#### Motivation

fp4 split-D (M8N1, 256<D<768，代表 D=320/512)：NVFP4 OMMA.SF MMA，两级 P 量化
（`quantize_and_pack_p` 2688 域），non-WS，大 D 寄存器压力同族同构。fp4
persist-D 沉淀的**两项已验证有效方案**与主循环结构无关，是首选移植对象：

1. **softmax 尾部逐元素除法 → 每 group 一次倒数 + FMUL（fp8 pscale 风格）+
   FirstTile row_sum 融合进 exp2 pass**——fp4 persist-D 实测 self@16k
   9.40→9.34；
2. **persistent work loop**——fp4 persist-D 验证有效（**注意这是家族相关的：
   fp8 同款已证伪零收益，附录 A #3**）；fp4 split-D 已有 per-work 批式
   epilogue（FC-2），主循环对齐成本低。

#### Design

1. **先决（硬）**：`ncu --page source` 拿到 D=320/512 的 stall / roofline /
   occupancy 画像；**无画像不动手**。
2. 按 Motivation 顺序移植两项有效方案，再叠加局部优化菜单（reg reconfig /
   NamedBarrier / TMA issue / MUFU）；每项 ≤ 几十行、不动主循环骨架。
3. 编译期路径隔离，默认配置行为零变化。

#### Files & Symbols

- `csrc/cuffpa/cute/fp4/sm_120/split_d.cuh`、P pack
  `csrc/cuffpa/cute/fp4/fp4_pscale.cuh`；
- 参照范本：`csrc/cuffpa/cute/fp4/sm_120/persist_d.cuh`（两项有效方案的
  落地形态）。

#### Validation

同 PC-7 套件（fp4 × D∈{320,512} × N∈{1024,4608,16384}）；动手前查附录 A
（#14 fp4 Q smem 复用、#15 rescale merge 等 fp4 专属证伪项）。

#### Risks & Rollback

同 PC-7；fp4 专属风险：P pack 与 rescale 的微改动易触碰已证伪方向
（#15/#16/#17），逐项先查边界。

#### Expected Benefit

按方案实测（参考量级：softmax 尾部单项 ~0.6%）。

#### Dependencies

PC-8（顺序：复杂优化逐个推进）。

---

### PC-10：fp4 split-D M4N2 量化大 D kernel 性能优化

- **Status**: Draft ｜ **Priority**: P2 ｜ **Track**: 性能

#### Motivation

fp4 split-D M4N2（D≥768，代表 D=768/1024）：kBr=64、atom_layout=(4,2,1)、
`o_acc=D/4`；与 fp8 M4N2 同为"低寄存器压力 + 长依赖链"形态。fp4 家族两项
已验证有效方案（同 PC-9 Motivation）继续适用，M4N2 结构下须独立画像验证。

#### Design

同 PC-9 方法论（NCU 先决 + 两项有效方案 + 局部菜单 + 证伪边界核对）；
PC-9 验证有效的方案优先复测移植。

#### Files & Symbols

- `csrc/cuffpa/cute/fp4/sm_120/split_d_m4n2.cuh`、P pack
  `csrc/cuffpa/cute/fp4/fp4_pscale.cuh`；
- 参照范本：同 PC-9。

#### Validation

同 PC-7 套件，bench 代表点改 D∈{768,1024}；每方案独立 A/B、独立回退。

#### Risks & Rollback

同 PC-9。

#### Expected Benefit

按方案实测。

#### Dependencies

PC-9（顺序：复杂优化逐个推进，同族经验复用）。

---

### PC-5：CUDA graph 友好化 (暂不实施，仅保留设计稿)

- **Status**: Draft ｜ **Priority**: P3 ｜ **Track**: 性能（部署能力）

#### Motivation

CUDA graph capture 要求 kernel 参数、stream、内存分配在 capture
期间确定。当前量化路径的前处理链有多次 `cudaMalloc`/`cudaFree`（descriptor 构建）、
cooperative 两阶段 launch（PC-1 引入后）、以及依赖运行时形状的动态 launch，可能破坏 graph 捕获。
cache-dit 若启用图模式部署，需要 ffpa 路径 graph-safe。

#### Design

1. **消除 capture 期间的分配**：descriptor/workspace 预分配或用 graph-safe 池。
2. **launch 结构固定**：PC-1 的 cooperative 两阶段 launch 与 workspace 在 capture 前分配，capture 内复用。
3. **动态 shape 处理**：若形状在 capture 时已知，可特化；否则用 graph 外部
   的 shape 绑定。
4. 用 `torch.cuda.graphs.make_graphed_callables` 验证。

#### Files & Symbols

- `csrc/cuffpa/native/launch.cuh`（descriptor 三连 malloc/copy/free）
- 与 PC-1 的 cooperative 两阶段 launch（capture 分支）

#### Validation

1. **捕获正确性**：`torch.cuda.graphs.make_graphed_callables` 或手动 capture/replay × 20 次，
   输出 bitwise 一致（同输入地址复用）；量化路径 + fp16 路径各一。
2. **性能不回退**：非 graph 的常规 bench 逐项持平（池查找开销 <1µs）。
3. **回归**：全套测试（非 capture 路径零变化是硬约束）。

#### Risks & Rollback

- 池的内存占用（每 descriptor 128B × 数量级，可忽略）；地址复用假设在 graph replay
  下成立（capture 固定地址）但**非 capture 的常规路径若启用池，需处理 tensor 地址变更的
  失效**——按 ptr 键自然失效即可。
- 若 cache-dit 图模式不使用 native TMA 路径（fp8/fp4 的 CuTe descriptor 是栈上
  `CUtensorMap` 作 kernel 参数，本就 graph 安全），本 RFC 范围自动缩小到 fp16 TMA hint
  ——先调查实际使用路径再实施。

#### Expected Benefit

解锁 graph 捕获（reduce-overhead / 图模式部署）；非纯性能项，偏部署能力。

#### Dependencies

PC-1 的 launch 形态先定（cooperative launch 的 graph 兼容性联动）；附录 A #2 的证伪边界。

---

### PC-6：sm_89 fp8 int4 QK（低优先级，搁置）(暂不实施，仅保留设计稿)

- **Status**: Draft ｜ **Priority**: P4（低，搁置） ｜ **Track**: 性能（sm_89 专属）

#### Motivation

SA2 证明了 int4 QK 的精度可行性（per-thread int4 量化 + smooth-Q rank-1 修正），
且 sm_89（Ada）有原生 int4 tensor MMA（`mma.m16n8k32.s32.s4.s4`，吞吐 2x int8）。
若 ffpa 的 sm_89 fp8 路线复活（见 Dependencies），QK 从 int8 下探 int4 是潜在性能方向。

**为什么优先级低——指令集区分（2026-08-28 确认）**：

- **SM100/SM103（B100/B200，数据中心 Blackwell）**：拥有完整 `tcgen05`，硬件原生支持
  INT4、NVFP4 block-scaled MMA。
- **SM120（RTX5090 / RTX PRO6000）**：**没有 `tcgen05.mma`**，没有硬件整数-INT4
  矩阵乘指令；Tensor Core 只原生支持 **NVFP4（E2M1 block-scaled FP4）、FP8、BF16、
  FP16、INT8**。
- **INT4 在 SM120 上**（W4A16 / Marlin-INT4 / AWQ-INT4 类）：**软件解码 + 上转成
  INT8/FP16 送入 TensorCore**，属软件实现的 4bit 权重 GEMM，复用现有 INT8/FP16 TC
  通路，不是硬件原生 INT4 MMA。

因此：sm_120 上 fp4/fp8 不走 int4 QK（fp4 已是硬件原生 NVFP4 的低比特上限；int4
只能软件实现、比原生 INT8/FP8 更慢），int4 QK 只对 sm_89 有意义——而 ffpa 当前
主目标是 sm_120。

#### 前置事实（2026-08-28 核对本地 `SageAttention/` 仓库）

- **SA2 的 int4 QK attention kernel 未开源**：`csrc/qattn/` 仅保留编译期脚手架——
  `attn_utils.cuh` 的 `kInt4` 枚举与 quant/dequant 分支、kernel 模板 static_assert
  允许 kInt4、`mma.cuh` 的两个 int4 MMA wrapper（m16n8k32 / m16n16k64）；但已发布的
  7 个实例化文件全是 `sm89_qk_int8_*`，`pybind_sm89.cpp` 只暴露 `qk_int8_*`，
  Python 侧零 int4 引用。**kernel 本体需自写**，可参考的只有论文 §3.3（per-thread
  int4）+ 上述脚手架。
- **ffpa 的 sm_89 fp8 路线当前处于回退状态**（memory `ffpa-fp8-sm89-cpasync`：最佳
  冷数据 1024 vs sage 857（+19.5%），判定无法追平、路线放弃）。根因是精度特性指令
  吃光 issue 带宽、tensor 管线饥饿——int4 QK 不直接解决该矛盾（per-thread int4 的
  quant/dequant 链同样指令密集）。必须先复活 sm_89 路线并确认 int4 QK 有净收益。
- 参考数据点：PRO 5000 上 int8 MMA 对 fp8 无 2x 吞吐（消费卡特性）。sm_89 的
  "int4 = 2x int8" 收益同样需按卡实测（消费 Ada 与 Ada 专业卡可能不同）。

#### Design

1. QK-only int4（PV 保持 fp8；fp32 buffer（SA2）/ f16acc（SA2++）按卡选择）；
2. 量化器：在 ffpa 现有 per-thread 量化基建上做 int4 变体（发射域收窄 ±7、按 mma
   fragment 所有权分组，SA2 §3.3），与既有 smooth-K + delta_s 组合；
3. kernel：复活后的 sm_89 fp8 kernel 上，QK MMA 换 `mma.m16n8k32.s32.s4.s4` +
   int32 acc；int4 fragment 打包/布局可参考 `SageAttention/csrc/mma.cuh` wrapper
   与 `attn_utils.cuh` 的 kInt4 分支；
4. 编译期隔离：int4 QK 作为独立 kernel 变体实例化（模板参数），sm_120 家族与既有
   路径零影响。

#### Files & Symbols

- 参考（ffpa 外，SageAttention 仓库）：`csrc/qattn/attn_utils.cuh`（kInt4 分支）、
  `csrc/mma.cuh`（int4 MMA wrapper）
- ffpa 侧：依赖 sm_89 fp8 路线复活形态（已删除的 `csrc/cuffpa/cute/fp8/sm_89/`
  谱系，结构见 memory `ffpa-fp8-sm89-cpasync`）
- 量化器：`csrc/cuffpa/cute/fp8/` 量化器家族的 per-thread 变体（int4 版新增）

#### Validation

1. 精度：CosSim/rel-L1 对 fp16 参考达到 SA2 论文水平（平均 ≥99.4%、最差 ≥96.7%）；
2. 性能：sm_89（4090 级）先单 kernel QK GEMM 微 bench 确认硬件 2x 收益存在，再
   全 kernel bench；
3. 回归：sm_120 全家族 bitwise 零变化（编译期隔离保证）。

#### Risks & Rollback

- SA2 int4 kernel 未开源 → 实现量大（新 kernel + 新量化器），故低优搁置；
- sm_89 路线未追平 sage 的根因（issue 带宽饥饿）未解决，int4 QK 可能仍整体落后；
- 回退：独立实例化变体，直接移除。

#### Expected Benefit

仅 sm_89 原生 int4 硬件有意义：QK GEMM 段理论 2x；整体收益取决于 QK 段占比。

#### Dependencies

先复活 sm_89 fp8 路线（memory `ffpa-fp8-sm89-cpasync` 的回退原因需重评估）。

---

## 附录 A：已证伪优化清单（动手前必读）

以下实验已完成且**负收益或零收益**，数据与结论详见报告 §5.8/§6.6 及 memory
（`ffpa-fp8-d128-ws-opt` / `ffpa-fp8-persistent-falsified` / `ffpa-fp4-persist-d`）。
重复投入前先查本表。

| # | 实验 | 结果 | 关键数字 |
|---|---|---|---|
| 1 | fp8 WS 双 consumer（FA3 式 2×128T） | 负优化，sm_120 不可行 | — |
| 2 | descriptor-only TMA cache（性能向） | 零收益已回退 | 5090 encode 0.42µs/次；bench 纹丝不动 |
| 3 | fp8 persistent work loop（fp4 方案移植） | 正确但零收益 | +0.04%/-0.25%（噪声） |
| 4 | fp8 stages 加深/不对称（K3V2/K2V3/K3V1） | 负 | K3V2 -1.7%；冷数据复测 -1.0% |
| 5 | fp8 K2+V1 | 严重负 | +12%（V 预取提前量归零） |
| 6 | fp8 v2（128P+128C, Br=64） | 严重负 | +31.6%（4-warp consumer MMA ILP 减半） |
| 7 | fp8 v5（V fragment 全量预载寄存器） | 严重负 | +48%（寄存器爆 spill） |
| 8 | cluster 化（无 DSMEM/multicast 需求） | 负 | 4-CTA 慢 5.7% |
| 9 | CUDA-core row_sum 替代 tensor rowsum MMA | 负 | -4%（rowsum 在 tensor 气泡免费） |
| 10 | fp8 O2 预计算 log2/RCP；O3 bank conflict | 否决 | MUFU 不入 stall top-25；ld conflict 0.39% |
| 11 | fp8 softmax max-pass 原地 prescale | 无效 | 跨 pass 依赖抵消收益（区别于有效的延迟 scale） |
| 12 | fp8 lazy rescale 重开（per-row P quant） | 否决 | 满量程零 headroom；f16acc 已被 FFMA 吸收 |
| 13 | aux vstats 扩容（512→128 rows/chunk） | 零收益已回退 | 带宽受限（~853GB/s） |
| 14 | fp4 Q smem 复用（fp8 模式迁移） | 负已回退（代码保留默认 OFF） | fp4 L1TEX hit 96.75% 饱和，fp8 前提不成立 |
| 15 | fp4 rescale merge→in-place | 负 | +1.7%（FMUL 进 MMA 依赖链） |
| 16 | fp4 q_full wait 延迟 | 中性偏负 | +0.1~0.2% |
| 17 | fp4 exp2 多项式替换 | 否决 | XU 28% 利用率非瓶颈，串行 FFMA 更糟 |
| 18 | vstats+vt 融合 | 算法不可行 | per-channel scale 必须先完成 |
| 19 | flat workspace + reorder（dispatch 侧） | 无收益 | TensorImpl 开销 ≈ 省掉的 empty |

**跨实验元教训**：fp8 attn kernel 微优化已到顶（kernel 级稳定优于 SageAttention +1.1~2.3%）；
结构优化收益是 kernel-结构相关的，不能跨量化格式外推（fp4 persistent 有效、fp8 零收益）；
优化前先查目标 kernel 的 L1 hit / stall 画像（fp4 教训）。

> 注意：#18（vstats+vt 融合）的证伪对象是**无同步的单遍融合**（per-channel scale
> 必须先完成）。PC-1 的 Mega Kernel 用两阶段 + grid 级屏障绕过该约束；PC-2 的增量
> 融合不含 vstats+vt。实施时先原型验证屏障成本，若 grid_sync 开销抵消 round-trip
> 削减，则退回 PC-2 的相邻对融合。

## 附录 B：验证基础设施

| 工具 | 状态 | 用途 |
|---|---|---|
| `tests/test_ffpa_nhd_layout.py` | 正式（63+ 用例） | NHD/strided bit-exact、hybrid、hadamard、负例矩阵 |
| `tests/test_ffpa_fp8.py` / fp4 用例 | 正式 | 量化路径 parity/rejection |
| `bench` CLI（`python -m ffpa_attn.bench`） | 正式 | `--cuda-impl` 全变体、`--pre-heat`、`--no-bwd` |
| `cache-dit-metrics psnr ssim` | 正式（cache-dit） | e2e 精度门禁（PC-3） |
| bitwise 8 场景 probe | **临时脚本**（`.tmp/int8-f16-opt/` 等） | 纯调度类改动的最快闸门——建议收编为 `tests/test_bitwise_probe.py` |
| 冷数据轮转 harness（`cold_bench.py` 模式） | **需重建**（原脚本已随 `.tmp` 清理；模式见 memory `ffpa-fp8-d128-ws-opt`：32 组 tensor 轮转 896MB≫L2 + CUDA event per-call） | aux 链/kernel 级 A/B（PC-1/2）；nsys 数字必须带 N 与冷热条件 |
| paired-window bench + median | 方法论 | 5090 功耗墙下降频的公平基准（单顺序 bench 不可信） |
| nsys NVTX 归因（sqlite join） | 方法论 | e2e 中 per-kernel/per-range 归因；`nsys GPU 总和 ≠ bench wall` |
| NCU stall 采样（`--page source`） | 方法论 | PC-4 的先决条件（先证明热点存在） |

**构建提醒**：`bash build.sh --arch sm_120f --headdim <list> --ext all --jobs 64`；
默认 headdim 集 = 64 倍数 ∈ [320,1024]，**覆盖 64-256 与 split-D 端点必须显式传**；
sm_120f（非 120a）才有 setmaxnreg；small-D 测试需 `FFPA_CUDA_ALLOW_SMALL_D=1`。

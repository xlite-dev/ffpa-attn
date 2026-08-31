# ffpa-attn CUDA Backend 未来优化 RFC（功能完备性优先）

> 状态：Draft ｜ 日期：2026-08-28 ｜ 关联文档：[SKILL.md](../SKILL.md)（特性现状与数学原理，下称"报告"）
> 本稿按"**功能完备性 > 性能优化**"两条轨道组织。功能完备性轨道解锁"现在做不了/被物化降级/直接报错"的能力；性能轨道在功能不变前提下提速。
> 动手前**必读附录 A（已证伪清单）**；所有性能类收益数字按报告 §2.2 纪律标注卡型/冷热条件。
> **约定**：本文所有代码路径均相对于 **ffpa-attn 仓库根目录**（如 `csrc/cuffpa/cute/launch.cuh`）。实施完成状态统一记录在两处：**完成状态清单**（下方，GitHub 上可直接勾选）与**总览表状态列**；各条目 `Status: Draft` 仅表示设计稿状态。每做完一项，同步勾选清单 + 更新总览表状态列（⬜ 待开始 → 🚧 进行中 → ✅ 已完成）。

## RFC实现规范（⚠️ 强制约束）

- 1. 每项动手前：先在 plan 模式（Copilot下要切换到plan agent）完成实施规划（改动面 / 注入点 / 验证矩阵），规划好再动手 (自动模式下可以按照规划继续实施操作)。
- 2. 每做完一项：勾选对应条目（`- [ ]` → `- [x]`），并同步更新上方总览表状态列。注意，要同时更新[SKILL.md](../SKILL.md) 中的技术报告和本文档的总览表状态列，确保两处状态一致。

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
| head_dim pad | ✗ | ✓ | ✓ | ✓ | FC-8 |
| decode / 短 Nq 量化 | ✗（无量化） | ✗ | ✗ | ✗ | FC-7 |
| backward | ✗ | ✗ | ✗ | ✗ | FC-9 |
| sm90 / sm100 | 部分（fp16 TMA） | ✗ | ✗（fp8 限 sm_120） | ✗（fp4 限 sm_120） | FC-10 |

> 读法：列方向看某个量化家族缺什么；行方向看某能力在哪些家族缺口。persist-D 三族功能最全，**所有大 D（超出 persist-D 上限）场景目前被上述 ✗ 卡住**。

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
| FC-8 | native head_dim pad | F3 | ⬜ 待开始 | — |
| FC-9 | CUDA backward (**暂不实施，仅保留设计稿**) | F3 | ⬜ 待开始 | — |
| FC-10 | sm90/sm100 量化覆盖 (**暂不实施，仅保留设计稿**) | F3 | ⬜ 待开始 | — |
| PC-1 | Mega Quantize Kernel（aux 链大融合） | P | ⬜ 待开始 | — |
| PC-2 | 增量融合（Mega Kernel 步进） | P | ⬜ 待开始 | 被 PC-1 收编 |
| PC-3 | N-crossover 量化配置自适应 | P | ⬜ 待开始 | — |
| PC-4 | fp4 persist-D attn kernel 内部优化 | P | ⬜ 待开始 | — |
| PC-5 | CUDA graph 友好化 (**暂不实施，仅保留设计稿**) | P | ⬜ 待开始 | PC-1 评估 |
| PC-6 | sm_89 fp8 int4 QK (**暂不实施，仅保留设计稿**) | P | ⬜ 低优搁置 | sm_89 fp8 路线复活 |
| PC-7 | fp8 split-D (M8N1) 量化大 D kernel 性能优化 | P | ⬜ 待开始 | — |
| PC-8 | fp8 split-D M4N2 量化大 D kernel 性能优化 | P | ⬜ 待开始 | PC-7（顺序） |
| PC-9 | fp4 split-D (M8N1) 量化大 D kernel 性能优化 | P | ⬜ 待开始 | PC-8（顺序） |
| PC-10 | fp4 split-D M4N2 量化大 D kernel 性能优化 | P | ⬜ 待开始 | PC-9（顺序） |

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
- [ ] FC-8：native head_dim pad
- [ ] FC-9：CUDA backward（定位评估）
- [ ] FC-10：sm90/sm100 量化覆盖

**轨道 P（性能优化）**

- [ ] PC-1：Mega Quantize Kernel —— P 轨基建
- [ ] PC-2：增量融合（Mega Kernel 步进，被 PC-1 收编）
- [ ] PC-3：N-crossover 量化配置自适应
- [ ] PC-4：fp4 persist-D attn kernel 内部优化
- [ ] PC-5：CUDA graph 友好化
- [ ] PC-6：sm_89 fp8 int4 QK（低优搁置：sm_120 无原生 int4 MMA，SA2 int4 kernel 未开源）
- [ ] PC-7：fp8 split-D (M8N1) 量化大 D kernel 性能优化
- [ ] PC-8：fp8 split-D M4N2 量化大 D kernel 性能优化
- [ ] PC-9：fp4 split-D (M8N1) 量化大 D kernel 性能优化
- [ ] PC-10：fp4 split-D M4N2 量化大 D kernel 性能优化

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
  FC-10 sm90/sm100 ⏸（FC-7/FC-9/FC-10 均暂不实施）
阶段 4（轨道 P）
  PC-1 Mega Quantize Kernel（先做 cooperative 两阶段原型）
        ├─► 收编 PC-2（增量融合是其落地台阶）
        └─► 联动 PC-5 ⏸（暂不实施；launch 形态定型后才能定 graph 兼容方案）
  PC-3 配置自适应 ｜ PC-4 fp4 persist-D kernel 内部（与上并行，互不依赖）
  PC-7 → PC-8 → PC-9 → PC-10 量化大 D kernel（优化复杂，严格逐个推进：
        fp8 split-D → fp8 M4N2 → fp4 split-D → fp4 M4N2，上一项验收后再启动下一项）
  PC-6 sm_89 int4 QK ⏸（暂不实施，低优搁置，不入执行序列；前置 = sm_89 fp8 路线复活）
```

> ⏸ = **暂不实施，仅保留设计稿**：不入执行序列、不排期；未来大概率不做，
> 仅当出现真实需求时重新评估。当前共 6 项：FC-5 / FC-7 / FC-9 / FC-10 /
> PC-5 / PC-6。（FC-7 搁置理由：短 Nq/decode 量化基本没有收益——固定前处理
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

- **Status**: Draft ｜ **Priority**: F3 ｜ **Track**: 功能

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

## 轨道 P：性能优化（功能不变下提速）

> 轨道 F 解决"能不能用"，轨道 P 解决"快不快"。以下各项在功能完备的前提下推进。

### PC-1：Mega Quantize Kernel（aux 链大融合）

- **Status**: Draft ｜ **Priority**: P1（性能轨最高） ｜ **Track**: 性能

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

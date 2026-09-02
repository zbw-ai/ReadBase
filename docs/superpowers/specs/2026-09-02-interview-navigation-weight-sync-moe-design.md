# 面试手册总览导航、权重同步与 MoE 基础题设计

## 1. 目标

在不推翻现有六个 Part 和 Core 10 的前提下，对面试手册做一次增量收敛：

1. 建立“先整体、后局部”的稳定阅读路径，让读者既能按 Part 学习，也能从顶部直接定位任一道题；
2. 新增 AReaL `XCCL` 与 `disk` 权重同步选型题，说明项目最终选择 XCCL 的条件、收益与边界；
3. 新增 Dense 与 MoE 基础题，补齐 expert routing、expert 粒度、总专家数/激活专家数和 shared expert 等高频概念；
4. 把 X1 项目统一表述为“200B MoE 模型”，删除“约 200B”这种绕口口径；
5. 简化个人能力图中心文字，保留结构信息但增加留白和视觉层级。

本轮不修改简历 DOCX，不虚构 X1 的专家配置，不重画能力图外围六个能力域，也不拆出新的分支文档。

## 2. 修改范围

### 2.1 主入口

- `private_resume/2026-08-llm-infra-interview-prep.md`

### 2.2 详细专题

- `training-infra-roadmap/topics/moe.md`
- `training-infra-roadmap/topics/agentic_rl.md`
- `training-infra-roadmap/topics/rl_framework_selection.md`：仅在需要补充框架间 XCCL 选型项目口径或双向链接时修改，不复制 AReaL 源码细节。

### 2.3 视觉资产

- `private_resume/assets/llm-infra-personal-capability-map.svg`

### 2.4 只读代码证据

- `/Users/zengbw/Codebase/for_agentic_rl/trail_renew_0720`

代码仓只用于核验 AReaL 当前项目分支的实现与支持矩阵，不修改其中任何文件，也不处理其中现有未提交改动。

## 3. 主文档信息架构

### 3.1 总体阅读路径

主文档形成一条明确的四层路径：

```text
能力全景 → Core 10 → 全量问题索引 → 各 Part 答案 / 专题深挖
```

- **能力全景**回答“候选人的能力边界是什么”；
- **Core 10**回答“只剩三小时必须会哪十道”；
- **全量问题索引**回答“某个知识点或题号在哪里”；
- **各 Part 与专题**回答“如何口述，以及原理如何展开”。

顶部不新增第二张装饰性架构图。已有个人能力图承担整体视野，Markdown 导航承担查询效率。

### 3.2 顶部导航区

现有“使用方法”收敛为“整体视野与问题导航”，顶层顺序严格按以下三项组织，随后直接进入各 Part 正文：

1. **能力全景**：先展示现有能力图；紧接着用一个紧凑表格说明六个 Part 分别解决的系统问题、核心关键词和 Part 入口，并在同一区域给出最终题量与 P0/P1/P2 总览。Part 角色表和题量统计都是能力全景的内部元素，不形成插在 Core 10 或全量索引之间的新顶层；
2. **Core 10**：保持现有十题和唯一正文位置不变，继续作为 P0 子集；
3. **全量问题索引**：按 Part 使用折叠块组织，块内再按 P0/P1/P2 列出所有题目的直接 anchor 链接。

因此实际阅读顺序始终是“能力全景（含 Part 角色与题量）→ Core 10 → 全量问题索引 → 各 Part 正文/专题”，不能把 Part 表或题量表再次插入 Core 10 与全量索引之间。

全量索引的要求是“可查、可跳、默认不压正文”：

- 每道唯一问题都必须出现一次直接链接；
- 索引只写 `题号 + 短标题`，不重复答案；
- Core 标签保留在相应 P0 题旁边，不额外制造一套题库；
- 各 Part 正文保留本 Part 的局部导航和追问路线；
- 顶部索引与 Part 局部导航均指向同一个正文 anchor。

### 3.3 六个 Part 不变

1. Part I｜个人定位、Ownership 与职业选择；
2. Part II｜Megatron、MoE、训练后端与长上下文；
3. Part III｜RL 算法、verl 与 Fully Async RLVR；
4. Part IV｜AReaL、Gateway、Agentic RL 与 MOPD；
5. Part V｜通用 Infra 与生产排障；
6. Part VI｜面试应变与查漏补缺。

Part VI 仍是学习和应变方法，不计入唯一问题总数。

## 4. 新增题目与最终计数

### 4.1 `MOE-01`（Part II，P0）

题目固定为：

> **MOE-01｜Dense 和 MoE 的主要区别是什么？expert 如何路由，是大专家还是小专家，有多少专家，是否有 shared expert？（P0，15 分钟）**

位置放在 Part II 的 MoE 基础入口，先于现有 `MEGATRON-06`。两题职责不重复：

- `MOE-01`：模型结构、路由和配置坐标系；
- `MEGATRON-06`：EP、token dispatch、all-to-all、负载不均和通信优化。

### 4.2 `AREAL-11`（Part IV，P0）

题目固定为：

> **AREAL-11｜AReaL 的 XCCL 与 disk 权重同步有什么区别？为什么项目最终选择 XCCL？（P0，15–18 分钟）**

该题与两个已有问题互链：

- `VERL-03`：训练模型如何同步到 rollout；
- `AREAL-06`：权重同步如何保证原子性和 policy version 正确。

`AREAL-11` 重点回答传输路径和工程选型，不复制上述两题的完整转换或原子提交回答。

### 4.3 规范性计数

新增两道唯一 P0 后，主文档最终必须满足：

| Part | 唯一问题数 |
|---|---:|
| Part I | 7 |
| Part II | 22 |
| Part III | 14 |
| Part IV | 17 |
| Part V | 8 |
| 合计 | 68 |

最终优先级分布为：

- P0：38；
- P1：25；
- P2：5；
- Core 10 是 P0 的子集，不额外计数。

Core 10 的成员不变。三天冲刺表、Part 题量、最后一小时清单和时间预算需要与 `68 / 38 / 25 / 5` 同步，不再出现旧的 `66 / 36 / 25 / 5`。

## 5. `MOE-01` 内容契约

### 5.1 主文档的最短准确答案

主文档先给一段可以在 30–60 秒口述的答案：

> Dense 模型中，每个 token 都经过同一套 FFN 参数，执行规则、GEMM 形状和负载比较稳定；MoE 把 FFN 换成多个 expert，由 router 为每个 token 选择 top-k expert，因此总参数可以很大，但单 token 只激活少量参数。代价是多出 router、token 重排、all-to-all、Grouped GEMM、负载均衡和更复杂的 checkpoint/并行映射。总专家数 `E` 和每个 token 激活的 `top-k` 是两个概念；所谓大专家或小专家主要看单个 expert 的 FFN/intermediate width，更多、更窄的专家能细化专业化，但也更容易产生小 GEMM、通信和负载不均。shared expert 是每个 token 都会经过的公共 FFN，用来承载共性能力，routed experts 再负责专业化；它不是所有 MoE 都必有的结构。

回答必须强调：具体模型到底有多少 expert、`top-k` 是多少、expert FFN intermediate size 多大、有没有 shared expert，取决于模型配置，不能从“200B MoE”自动推断。

### 5.2 详细专题内容

在 `topics/moe.md` 的 Parallel Folding 章节之前新增独立锚点和基础章节，避免为了插入一节而机械重编号全文。章节至少包含：

1. Dense 与 MoE 的计算路径和参数激活差异；
2. 常见 token-choice top-k routing：router score、选 expert、dispatch、expert compute、combine；
3. 总专家数 `E`、激活专家数 `k`、总参数和激活参数的区别；
4. large expert 与 fine-grained/small expert 的工程取舍；
5. shared expert 的作用、额外计算和适用边界；
6. load balancing、capacity/dropless、小 GEMM 和 all-to-all 等生产代价；
7. 一张“拿到 MoE 配置先问什么”的检查表。

检查表至少包含：

```text
总专家数 E / top-k / expert FFN intermediate size / shared expert 数量
router 与 balance 策略 / capacity 或 dropless / EP 与 topology 映射
```

可以提 expert-choice 等其他路由作为扩展，但不能喧宾夺主；面试主线以工程中更常见的 token-choice top-k 为主。

### 5.3 项目证据边界

- 项目可确认的是“X1 200B MoE 模型”的适配与性能优化；
- 当前资料没有给出可公开且已核验的 `E / top-k / expert FFN intermediate size / shared expert` 配置；
- 主文档的项目证据卡新增这些字段并保留待本人补齐，实施者不得用论文、相似模型或经验值代填；
- 不把“fine-grained expert 更先进”写成普遍结论，必须同时说明通信、kernel efficiency 和负载均衡代价。

## 6. `AREAL-11` 内容契约

### 6.1 先建立共同语义

两种模式解决的是同一件事：optimizer update 后，把新 actor 权重交给 rollout engine，并且只在传输成功后推进可见的 policy version。它们的差异主要在数据路径，而不是是否需要暂停生成或是否需要版本原子性。

`disk` 在这里是**临时 HF 格式权重传输路径**，不等于长期保存的 recovery checkpoint。主文档和专题都必须显式区分二者。

### 6.2 XCCL 数据路径

基于本地项目分支的实现事实描述：

```text
trainer weights
  → layout/name/dtype 对齐与分桶
  → 参与传输的 trainer sender rank(s) 与 rollout ranks 建立 XCCL process group
  → NCCL/XCCL broadcast 或点到多点的显存直传
  → rollout engine 更新权重
  → 成功后切换 policy version
```

工程特征：

- 跳过 HF 序列化、共享文件系统写入和 rollout 侧重新加载；
- 适合频繁同步大模型权重，项目固定 workload 下通常延迟更低；
- 可通过 bucket/chunk 控制峰值显存和流水；
- 更依赖 rank/process-group、参数名、shape、dtype、并行 layout 和 backend 支持矩阵正确；
- 失败时更容易表现为 collective hang、版本不一致或部分 rank 卡住，排障复杂。

### 6.3 disk 数据路径

```text
trainer weights
  → 保存到 versioned 临时 HF path
  → 通知 rollout server 加载对应路径
  → 各 rollout engine 从存储读入并更新
  → 成功后切换 policy version
  → 清理临时传输文件
```

工程特征：

- producer/consumer 解耦，路径可观察、可重试、容易人工检查；
- 对 backend、部署形态和部分 LoRA 路径的兼容范围更宽；
- 代价是 save + filesystem + load 的额外时延、共享存储带宽和容量压力，以及临时完整权重副本；
- 本地项目分支中的 actor–rollout colocation 显式要求 disk；ref/critic 等其他 colocation 不自动触发该限制。SGLang 的 LoRA XCCL 路径也不受支持，不能把 XCCL 说成所有组合都能用。

### 6.4 选型结论与项目口径

主文档允许使用的项目表述是：

> 我们最终在 verl 和 AReaL 两条项目链路中都选择了 XCCL。原因不是“XCCL 永远更好”，而是在当时固定模型、并行布局、rollout backend 和网络条件下，更新频率较高，XCCL 避开了完整权重落盘和重新加载，实测同步更快；同时相关 backend 路径已经跑通，因此收益大于集成复杂度。若是低频更新、共享存储足够快、跨集群解耦、需要更容易进行传输重试、artifact 检查或故障定位，或者 backend/LoRA/actor–rollout colocation 组合不支持 XCCL，我会重新评估 disk。

没有已核验 benchmark 数字时，不填写具体加速倍数。`verl` 与 `AReaL` 都选择 XCCL 是项目事实；具体底层实现和支持矩阵分别归属于框架，不能因为选型相同就声称两者实现完全一致。

### 6.5 正确性与排障追问

详细专题至少覆盖：

- 为什么同步前需要 pause/admission control，失败后为什么不能推进 version；
- 参数名、shape、dtype、TP/PP/EP layout 不一致会怎样；
- XCCL hang 的 rank 日志、collective sequence、process group 和 timeout 排查；
- disk 模式的半写文件、旧 path、加载失败、存储带宽和清理问题；
- 为什么 weight transfer artifact 与 training recovery checkpoint 是两套生命周期。

## 7. “200B MoE 模型”统一口径

本轮只处理当前面试准备链路与能力图中的 X1 项目表述：

- `X1 约 200B MoE` → `X1 200B MoE 模型`；
- 卡片空间较小时允许写 `X1 · 200B MoE`；
- 不批量改写历史审计文档或论文中本来表达估计值的“约”；
- 不因删除“约”而进一步补充未确认的精确参数、expert 配置或客户敏感信息。

## 8. 能力图中心排版

### 8.1 保持不变

- 外围六个能力域、连线、证据区和整体画布不变；
- 继续使用仓库内 SVG，不改成位图；
- 不增加 framework logo、图标或额外装饰。

### 8.2 中心内容

删除当前中心的“个人能力地图”“大模型训练推理”“Infra”和框架名堆叠，改为三层：

```text
LLM Infra
TRAINING · RL · ROLLOUT
性能 × 正确性 × 交付
```

排版原则：

- 第一行是唯一主标题，字号约 `52–56px`，字重最高；
- 第二行使用约 `18–20px` 的英文小标题和适度 letter spacing；
- 第三行约 `23–25px`，与第二行留出明显行距；
- 框架名不再放中心，外围卡片已经承载 Megatron、verl、AReaL；
- 保持三行都在中心卡片安全区，不碰边框、不与连线重叠；
- 内层描边若保留必须弱化，不能形成“框中再塞一张表”的视觉效果。

本轮目标是减少拥挤和建立层级，不调整外围卡片文案长度；若后续还需要整图出版级重绘，单独立项。

## 9. 链接与知识生命周期

- `MOE-01` 从主文档链接到 `topics/moe.md` 的 Dense-vs-MoE 新锚点；专题回链到 `MOE-01`；
- `AREAL-11` 从主文档链接到 `topics/agentic_rl.md` 的 weight sync 新锚点；专题回链到 `AREAL-11`；
- `AREAL-11` 与 `VERL-03`、`AREAL-06` 互相提供相邻题跳转；
- `topics/rl_framework_selection.md` 只保留“两个项目最终均选 XCCL”的选型摘要和详细专题链接，避免三处维护同一源码说明；
- 现有 README、`KNOWLEDGE_GRAPH.md` 和 `MASTER_READING_LIST.md` 已能到达这些专题时不扩大改动；只有入口不可达才补链。

## 10. 验收标准

### 10.1 内容与结构

1. 主文档形成“能力全景 → Core 10 → 全量问题索引 → Part 正文/专题”的可见路径；
2. 全量索引覆盖 68 道唯一题，所有链接指向题目唯一正文；
3. Part 题量严格为 `7 / 22 / 14 / 17 / 8`；优先级严格为 `38 / 25 / 5`；
4. `MOE-01` 和 `AREAL-11` 都是 P0，题号、标题和显式 anchor 唯一；
5. Core 10 不扩容、不复制，仍是 P0 子集；
6. `MOE-01` 与 `MEGATRON-06`、`AREAL-11` 与 `VERL-03/AREAL-06` 职责清晰且互链；
7. 主文档保持短答案，详细机制进入专题，不产生新的分支专题文件。

### 10.2 技术口径

1. 不混淆总专家数与 active top-k，不把 shared expert 写成所有 MoE 必备；
2. 不虚构 X1 的 expert 数、top-k、expert FFN intermediate size 或 shared expert；
3. 不把 disk weight transfer 等同于 recovery checkpoint；
4. 不把 XCCL 描述成无条件优于 disk，也不声称 verl 与 AReaL 的实现完全相同；
5. 正向说明本地分支中 actor–rollout colocation 和 SGLang LoRA 等支持矩阵边界，且不把 ref/critic colocation 混入该限制；
6. “verl 和 AReaL 最终使用 XCCL，因为项目 workload 下更快”明确标为项目选型结论，缺少 A/B 数字时不写倍数；
7. 面试准备链路和能力图不再出现 `约 200B MoE`，正向出现 `200B MoE 模型`。

### 10.3 视觉与文件

1. SVG 中心只有三行规范文案，标题、英文副标题、中文能力结果形成清晰层级；
2. SVG 可通过 XML 解析，并渲染为 PNG 做人工视觉检查；
3. 中心文字不溢出、不重叠，外围结构和连线无意外位移；
4. 所有 Markdown 相对链接和图片路径可解析；
5. `git diff --check` 无错误，Git diff 不包含简历 DOCX、只读代码仓或无关文件改动。

## 11. 提交与发布

实现完成后使用一个聚焦的文档提交，提交信息建议为：

```text
Improve interview navigation and weight sync coverage
```

提交前获取远端引用并核对本地 diff；推送到当前 `codex/personal-infra-map` 分支。若需要同步主分支，沿用仓库现有安全合并流程，不覆盖远端未包含的用户改动。

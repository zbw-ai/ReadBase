# 个人版大模型训练推理 Infra 能力地图设计

## 1. 背景与目标

用户希望参考一张中心辐射式 AI Infra 思维导图，结合自己的真实项目经历，制作一张适合社招面试使用的个人能力总览图。该图不是通用知识树，也不是简历时间线，而是主面试手册开头的“阅读导航 + 面试定位”视觉入口。

目标有四个：

1. 面试官或候选人能在 20 秒内看懂核心技术主线。
2. 以真实项目证据支撑能力，不把框架知识包装成亲自实现。
3. 把 Megatron、verl、AReaL、推理、通信、显存和规模化交付连成一套系统能力。
4. 从总览图可以顺畅跳转到主文档中的重点题目和详细专题。

## 2. 已确认方案

采用“六域能力地图 + 项目证据带”。

- 中心：`大模型训练推理 Infra`，副标题为 `Megatron · RLVR · Agentic RL · Scale Delivery`。
- 中层：六个能力域，从中心向外辐射。
- 外层：每个能力域的关键机制和框架。
- 底部：五至六张真实项目证据卡，呈现可核验数字与项目边界。
- 视觉语义：实心节点表示项目实战；空心节点表示原理掌握或能力延伸。

不采用双环能力图，因为信息密度过高、在 Markdown 中缩放后阅读困难；不采用分层技术栈，因为不如辐射式结构适合作为个人面试能力总览。

## 3. 信息架构

### 3.1 六个能力域

#### A. 训练系统 / Megatron

- Megatron-Core 与 5D 并行：DP、TP、PP、CP、EP。
- MoE：Parallel Folding、Grouped MatMul、token dispatch、融合算子。
- 长上下文：128K/256K、SP/CP、recompute。
- SFT 数据供给：packing、DataLoader/prefetch 和有效 token 口径。
- 训练状态：Distributed Optimizer、checkpoint/recovery。

#### B. RL 与后训练

- verl：SFT/RLVR、HybridFlow、colocate/disaggregate、Fully Async。
- AReaL：Agentic RL、Gateway、Session、Cohort、staleness。
- Agent/Tool/Sandbox 环境对接，以及 ready-cohort 长尾和 trajectory goodput。
- OPD/MOPD：TILE merge 基线、多 Teacher 蒸馏和能力汇聚。

#### C. 推理与 Rollout

- vLLM、SGLang。
- gen-TP 与多实例资源配比。
- continuous batching、KV Cache、Prefix Cache、CUDA Graph。
- 训练—推理权重同步和 policy version。

#### D. 通信与拓扑

- NCCL/XCCL。
- AllReduce、ReduceScatter、AllGather、AllToAll、P2P。
- NVLink/NVSwitch、IB/RoCE/HCCS。
- 计算通信重叠、exposed communication 和 process group 映射。

#### E. 显存与性能

- Megatron 显存账本：参数、梯度、优化器、activation、logits/KV/workspace。
- recompute、offload、fusion。
- profiling、MFU、吞吐与尾延迟。
- Memory / Communication / Compute Efficiency 三堵墙和瓶颈迁移。

#### F. 正确性与规模化交付

- loss、logprob、mask、precision alignment。
- FUNCTIONAL / NUMERIC / EFFICACY 三层门禁。
- checkpoint、恢复、trajectory lineage。
- 国产卡适配、千卡交付、性能迭代闭环和 Ownership。

### 3.2 项目证据带

项目证据使用比知识节点更深的底色，并保留事实边界：

1. `X1 · 约 200B MoE`：相对性能 `0.16x → 0.95x`，MFU 35%，3K 卡训练保障。
2. `Fully Async RLVR`：`76` 明确标为 async 初始 `3T+1R / gen-TP=4`；`211–255 tokens/s/GPU` 标为 `gen-TP 4→2` 后的代表性稳态窗口；`236–293` 只标为 `2T+2R candidate`。三者都不是全程平均，更不是 sync→async 提升，图中不得出现“相比同步 3x”。
3. `SFT / Long Context`：Qwen3/Qwen3.5，32K–256K。由于 `31s → 9.3s` 的 GPU、sequence、batch、有效 token 等完整口径尚待证据卡补齐，主图不展示该数字，只展示数据供给、recompute 和长上下文交付范围。
4. `AReaL Agentic RL`：Gateway、Session、Cohort、trajectory/version 链路与异步长尾治理。
5. `MOPD`：多个 RL Expert 作为 Teacher、RL 前模型作为 Student、使用原 RL 数据进行 OPD；最终 efficacy 未完全闭环时不得写成已证明提升。
6. `X1 / TX 国产卡交付`：个人负责模型适配与性能达标闭环，即模型跑通 → profile → 定位 → 优化 → 验证 → 扩容；不是整个千卡/万卡平台或集群运维 owner。

## 4. 经验边界表达

图中必须把“实际做过”和“理解/可延伸”分开：

- 实心圆点或实色节点：简历与项目事实底稿能支持的直接经验，包括亲自集成、配置、调优、验证和交付；实心不等于底层算法或 kernel 由本人实现。
- 空心圆点或描边节点：仅掌握机制、尚待补齐项目证据，或今天会评估的新方向。
- 图下注释明确写：`实心 = 项目实战；空心 = 原理掌握/能力延伸`。
- CUDA/Triton kernel 开发、底层 collective 实现、量化算法研发不得画成核心 ownership；可以作为原理掌握节点出现。
- NVIDIA 2026 报告中的 Parallel Folding、DeepEP、FP8/FP4 等新方向不得倒灌成 X1 项目历史事实。

### 4.1 节点级经验与证据映射

实施时按下表逐项设置视觉状态，不能由绘图者临时推断：

| 能力域 | 节点 | 状态 | 主文档证据 | Ownership 限定 |
|---|---|---|---|---|
| 训练系统 | Megatron 5D 配置与调优 | 实心 | `#megatron-01`、`#resume-01a` | 使用、集成与调优，不声称发明并行算法 |
| 训练系统 | Grouped MatMul / 融合算子接入 | 实心 | `#resume-01a` | 接入、A/B 验证和性能闭环；底层 kernel 是否亲写需单独说明 |
| 训练系统 | 128K/256K、SP/CP、recompute | 实心 | `#resume-06`、`#megatron-04` | 长上下文配置、显存和交付经验 |
| 训练系统 | packing / DataLoader / prefetch | 实心 | `#resume-05` | 数据供给和训练 step 优化，数字口径仍需补齐 |
| 训练系统 | Parallel Folding / DeepEP / FP8/FP4 | 空心 | `#resume-01a` 延伸阅读 | NVIDIA 2026 新方向，不能作为 X1 已实施成果 |
| RL 与后训练 | verl SFT/RLVR / HybridFlow | 实心 | `#verl-01`、`#areal-01` | 框架选型、集成、配置和二次开发 |
| RL 与后训练 | Fully Async producer-consumer | 实心 | `#resume-02`、`#verl-04` | 资源配平、queue/staleness 与吞吐优化 |
| RL 与后训练 | AReaL Gateway / Session / Cohort | 实心 | `#resume-08`、`#areal-03` | Agentic RL 服务链集成与改造，不声称独立设计整个 AReaL |
| RL 与后训练 | Agent/Tool/Sandbox 环境对接 | 实心 | `#resume-08` | Tool/Sandbox 属于 Agent/外部环境，不画成 AReaL 固定内部 stage |
| RL 与后训练 | ready-cohort 长尾 / trajectory goodput | 实心 | `#resume-08`、`#areal-04` | 关注真正进入训练的 trajectory，而非只看 generated tokens |
| RL 与后训练 | MOPD 多 Teacher 链路 | 实心 | `#resume-09` | FUNCTIONAL 已验证；NUMERIC/EFFICACY 未闭环时不得宣称最终提升 |
| 推理与 Rollout | vLLM / SGLang rollout | 实心 | `#resume-03`、`VERL-09` | 后端接入、部署和调优，不声称实现引擎核心 |
| 推理与 Rollout | gen-TP / 多实例 / continuous batching | 实心 | `#resume-02`、`#resume-03` | 资源配比与吞吐调优 |
| 推理与 Rollout | KV/Prefix Cache / CUDA Graph | 实心 | `RESUME-13`、`RESUME-14` | 使用、调优和端到端归因；不等于 kernel/graph runtime 作者 |
| 推理与 Rollout | 权重同步 / policy version | 实心 | `#verl-03`、`#areal-02` | 布局转换、版本与 staleness 语义 |
| 通信与拓扑 | AllReduce/RS/AG/A2A/P2P | 实心 | `#infra-04`、`#resume-01a` | 使用、profile、process group 与通信掩盖，不声称实现 collective |
| 通信与拓扑 | NCCL/XCCL 集成与排障 | 实心 | `#infra-03`、`#verl-03` | 集成、诊断和权重同步，不声称实现通信库 |
| 通信与拓扑 | NVLink/NVSwitch、IB/RoCE/HCCS 精细拓扑映射 | 空心 | `INFRA-05`、`#resume-10` | 有相关交付背景，但 X1/TX 的真实 group→物理拓扑细节尚待补齐 |
| 通信与拓扑 | 计算通信 overlap / exposed time | 实心 | `#resume-01a` | profile、依赖分析和方案验证 |
| 显存与性能 | Megatron 显存账本 / OOM | 实心 | `#infra-02`、`#resume-06` | 配置、估算、profile 与问题定位 |
| 显存与性能 | recompute / fusion | 实心 | `#resume-01a`、`#resume-05`、`#resume-06` | 以实际使能和 A/B 实验为边界 |
| 显存与性能 | offload | 空心 | `MEGATRON-09` | 原理掌握；没有明确项目证据时不画成实战 |
| 显存与性能 | profiling / MFU / 瓶颈迁移 | 实心 | `#resume-01a`、`#infra-01` | 性能基线、profile、A/B 与规模回归 |
| 显存与性能 | Three Walls 框架 | 空心 | `#resume-01a` | 用于今天的系统化归纳，不倒灌为当时项目方法名称 |
| 正确性与交付 | loss/logprob/mask / precision alignment | 实心 | `#verl-05`、`RESUME-12` | 功能与数值验证、first divergence 定位 |
| 正确性与交付 | FUNCTIONAL/NUMERIC/EFFICACY | 实心 | `#resume-09`、`AREAL-08` | 三层门禁方法；MOPD 最终效果边界单独标明 |
| 正确性与交付 | checkpoint/recovery / trajectory lineage | 实心 | `#infra-03`、`#areal-04` | 恢复、版本和训练贡献可追踪性 |
| 正确性与交付 | X1/TX 国产卡规模交付闭环 | 实心 | `#resume-10` | 模型适配和性能达标 owner，不是集群平台/运维总 owner |

## 5. 视觉设计

- 格式：仓库原生 SVG，适合 GitHub 与 Markdown 缩放阅读。
- 建议画布：`1600 × 1000`，浅色背景，中心辐射结构。
- 布局：左右各三个能力域；每个域用一条平滑主曲线连接中心，子节点沿水平支线展开。
- 色彩：以青绿色为主色，六域使用低饱和度蓝、青、紫、橙、绿、红作语义区分。
- 字体：优先系统中文字体，不依赖外部字体；为保证 736px 宽度下有效字号不低于约 11px，1600px viewBox 内所有有信息量的文字不小于 24px。
- 节点：领域标题使用圆角矩形；能力节点以短语为主，避免完整句子。
- 项目证据：底部使用六张等宽卡片，数字比描述更醒目。
- 避免：水印、装饰性图标、交叉箭头、超长节点、同层信息重复、把所有节点画成相同权重。
- 可访问性：SVG 包含 `<title>` 和 `<desc>`；Markdown 使用有意义的 alt text，而不是文件名。

## 6. 主文档集成

### 6.1 文件位置

- SVG：`private_resume/assets/llm-infra-personal-capability-map.svg`
- 主文档：`private_resume/2026-08-llm-infra-interview-prep.md`

### 6.2 嵌入位置

放在主文档 `## 0. 先看结论：面试官会如何评估你` 中，“五句法”说明之后、`## 1. 三天冲刺安排` 之前。这样既保留当前优先级结构，也让读者先理解评估标准，再看个人能力地图。

### 6.3 图下导航

图下提供短链接，不依赖 SVG 内部点击；在窄屏下拆成两行，避免形成过密长链接：

- Megatron → `#megatron-01`
- verl → `#verl-01`
- AReaL → `#areal-01`
- 显存 → `#infra-02`
- 通信 → `#infra-04`
- 项目证据 → `#resume-01a`、`#resume-02`、`#resume-08`、`#resume-09`、`#resume-10`

图下再提供一段 20–30 秒口述建议：先报两条主线，再用 X1 MoE 和 Fully Async/AReaL 各给一个证据，最后说明自己的定位是训练系统集成、性能与正确性优化及规模化交付。

## 7. 浏览器预览

为帮助用户在正式落库前检查布局，同时创建一个对话内可视化预览。预览只用于审阅，不作为仓库最终资产；最终仓库以 SVG 和主文档为准。预览需要保留六域、经验边界和项目证据，但可以提供聚焦某一能力域的交互，以便检查信息层级。

## 8. 验收标准

1. SVG 能通过 XML 解析，并包含 `<title>`、`<desc>` 和有意义的 Markdown alt text。
2. 按 §4.1 映射逐项核对：六个能力域、六张证据卡、图例，以及每个节点的实心/空心状态完整且正确。
3. 将 SVG 实际渲染成 1024px 和 736px 宽截图并人工检查，无文字重叠、截断或低于约 11px 的有效字号；证据卡中的边界标签必须可读。
4. 主文档中的图片路径、所有新增相对链接和 `#fragment` anchors 均可解析。
5. Fully Async 数字逐项核对：76 是 async 初始配置；211–255 是 gen-TP 4→2 后代表性稳态窗口；236–293 是 2T+2R candidate；三者不是全程平均，图中不出现 sync→async 3x。
6. SFT 证据卡不展示尚未补齐 workload 口径的 `31s → 9.3s`；MOPD 卡不宣传未闭环 efficacy。
7. 图中不会让读者误以为用户是 CUDA/Triton kernel、NCCL collective、Megatron 核心算法、整个千卡平台或集群运维 owner。
8. 图与现有三天冲刺、P0/P1/P2 结构互补，不复制大段题目答案。

## 9. 非目标

- 不重构整个主面试文档。
- 不新增多份分支学习文档。
- 不把图扩成完整 AI Infra 通用知识树。
- 不公开简历中需要继续脱敏的客户名称或内部实现细节。
- 不对尚未严格闭环的模型效果做确定性宣传。

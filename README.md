# ReadBase

面向训练、推理、RL 与具身智能的 AI Systems 工程学习库：理解机制，形成判断，再用实验和排障验证。

<a id="familiar-entry"></a>
## 原文档快捷入口

原来的 `training-infra-roadmap/` 入口已收编到 **[02 · 训练手册](02-training-infra/README.md#original-library)**。训练正文和学习计划都在这里；原有 RL、论文、PDF、实践及面试资料也能从同一页直接找到，不必重新记目录。

| 想找的内容 | 直接进入 |
|---|---|
| Megatron 与并行训练 | [5D 并行](02-training-infra/topics/distributed_training.md) · [TP](02-training-infra/topics/tensor_parallelism.md) · [MoE / Parallel Folding](02-training-infra/topics/moe.md) · [FSDP/FSDP2/ZeRO](02-training-infra/topics/fsdp.md) |
| RL 与 Agent | [Agentic RL / Gateway](04-rl-infra/topics/agentic_rl.md) · [verl / AReaL 框架选型](04-rl-infra/topics/rl_framework_selection.md) · [MOPD](04-rl-infra/topics/mopd.md) |
| 原论文与学习记录 | [NVIDIA MoE 五份中文 PDF](research/MASTER_READING_LIST.md#megatron-core-moe-2026-zh-pdf) · [论文／报告索引](research/MASTER_READING_LIST.md) · [扫描](research/tracking/README.md) / [月报](research/tracking/monthly_reviews.md) · [30 天](02-training-infra/roadmaps/30_day_plan.md) / [90 天](02-training-infra/roadmaps/90_day_plan.md) / [年度计划](02-training-infra/roadmaps/yearly_plan.md) |
| 项目与面试 | [实验与项目](practice/README.md) · [面试主文档](private_resume/2026-08-llm-infra-interview-prep.md#interview-console) · [Coding](private_resume/2026-09-interview-coding.md#coding-top) |

找主题看下面的 [01–05 技术 Part](#learning-parts)，按步骤学习看[核心路线](#core-roadmaps)。目录编号固定，Topic 的 `1.1、2.1…` 编号只用于各 Part 内导航，正文文件名保持不变。

## 当前学习重点

沿这条线开始，已有的 LLM Training 与 Agentic RL 积累继续保留：

1. [具身六阶段路线](05-embodied-infra/roadmap.md)：模型与动作 → episode 契约 → 数据 I/O → 训练 → 评估 → 部署；先建立模型印象，再复用已有 Infra 正文。
2. [FSDP／FSDP2／ZeRO](02-training-infra/topics/fsdp.md)：从状态账本、通信和参数生命周期入手，继续深化源码与性能取舍。
3. [A100 学习实验](practice/experiments/a100_fsdp_io_lab.md)：先环境画像，再状态分片、通信与多模态 I/O；E00–E08 均未执行。

具体阅读待办只在 [P0／P1 队列](research/reading_queue/README.md)维护；方向调整不把旧条目自动标为读完。

<a id="recent-updates"></a>
## 最近更新

| 文档更新 | 类别 | 值得读的变化 |
|---|---|---|
| 2026-10-09 | 导航与学习路线 | [原训练手册入口](02-training-infra/README.md#original-library)统一收编旧入口；五个技术 Part 采用两层编号；[具身六阶段](05-embodied-infra/roadmap.md)仍为当前重点 |
| 2026-10-09 | MiMo 技术研究 | [V2.6 系统设计](research/tech_reports/mimo_v26.md#3-强化学习基础设施)：统一轨迹、多框架并发、控制与数据面分离、训推概率一致性；区分生产机制与开源交付；提供[飞书 Word 导入版](assets/handbook/mimo_v26/mimo_v26_feishu.docx)，未复现训练 |
| 2026-09-22 | 学习结构／实验设计 | 各技术 Part 根目录并列；新增 [A100 E00–E08 课程](practice/experiments/a100_fsdp_io_lab.md)，覆盖训练、I/O、恢复、推理和 RL，尚未执行 |
| 2026-10-08 | 研究复盘 | [季度／月度总览](research/tracking/monthly_reviews.md)与[九月正式月报](research/tracking/monthly_signal_2026-09.md)：保留云端历史补证及月底增补，原始新信号见[十月扫描](research/tracking/frontier_scan_2026-10-08.md) |
| 2026-09-22 | 实现证据 | [7–9 月 GitHub 历史复盘](research/tracking/github_retrospective_2026-07_to_2026-09.md)：保留事件索引、重点 PR 复核和证据边界 |

这里只记录仓库新增或实质更新，最多 5 条；外部新文章看[研究雷达](research/tracking/README.md)，项目进展看[实践入口](practice/README.md)。

<a id="learning-parts"></a>
## 日常学习：按问题进入

| Part | 入口 | 解决什么问题 |
|---|---|---|
| 01 | [硬件与系统基础](01-systems/README.md) | GPU／CPU／内存／互联如何制约 kernel、通信与数据供给？ |
| 02 | [训练系统](02-training-infra/README.md) | 如何切分模型和状态，优化显存、吞吐、checkpoint 与训练稳定性？ |
| 03 | [推理系统](03-inference-infra/README.md) | 如何兼顾吞吐、时延、缓存和服务质量？目前为局部内容入口 |
| 04 | [RL 与 Agent 系统](04-rl-infra/README.md) | rollout、环境、reward、训练与权重版本怎样形成可靠闭环？ |
| 05 | [具身模型与 Infra](05-embodied-infra/README.md) | MLLM／VLA／World Model 的数据、学习与执行需求怎样落到系统？ |
| 06 | [实验、项目与排障](practice/README.md) | 哪些理解得到验证，遇到故障怎样定位、恢复并记录边界？ |

这些方向并列、相互复用，不必先“学完训练”才开始推理或具身。完整关系见[知识地图](KNOWLEDGE_GRAPH.md)。

<a id="core-roadmaps"></a>
## 核心学习路线：按阶段学，按问题查

路线只管顺序、掌握标准与正文入口，不复制技术章，也不替代研究阅读队列。

| 路线 | 核心主线 |
|---|---|
| [具身 Models & Infra（当前重点）](05-embodied-infra/roadmap.md) | 模型 → episode → I/O → 训练 → 闭环评估 → 部署 |
| [GPU Systems](01-systems/roadmaps/gpu-systems.md) | 执行与内存 → tiling/layout → IO-aware kernel → runtime → 测量 |
| [Distributed Systems](01-systems/roadmaps/distributed-systems.md) | 通信 → 部分失败 → 背压 → 版本协调 → 持久化与恢复 |
| [Inference Infra](03-inference-infra/roadmap.md) | 成本 → KV 状态 → batching → 执行优化 → 服务 SLO |
| [Agentic RL Infra](04-rl-infra/roadmap.md) | 算法契约 → 轨迹 → 调度 → 异步 → 权重同步 → 恢复 |

训练基础沿用 [Part 02](02-training-infra/README.md)，当前重点是 [FSDP/FSDP2/ZeRO](02-training-infra/topics/fsdp.md)；历史 30/90 天计划保留，不再另建一套全库日程。各路线的练习复用 [A100 E00–E08](practice/experiments/a100_fsdp_io_lab.md)，均为待执行设计。

## 按需使用：面试复习

[Part 07 · 面试准备](interview/README.md) → [主文档现场速查](private_resume/2026-08-llm-infra-interview-prep.md#interview-console) · [Coding 题库](private_resume/2026-09-interview-coding.md#coding-top)。平时学习无需经过题库。

## 资料与研究流程

[研究入口](research/README.md) · [材料索引](research/MASTER_READING_LIST.md) · [月度复盘](research/tracking/monthly_reviews.md) · [扫描账本](research/tracking/scan_log.md) · [学习记录](research/learning_log/README.md)

- 技术正文在各 Part 的 `topics/`，同一问题只有一份完整正文，跨方向用链接复用。
- 论文、报告、工程博客与追踪队列统一在 `research/`；不要每个 Part 再建一套雷达。
- `practice/` 区分实验设计、实际结果、项目状态和排障；读过、写过、验证过不是同一状态。
- 各页可返回所属 Part 或首页；返回刚才的精确位置用浏览器“后退”。仓库内链接不依赖本地预览服务。

研究方法见 [philosophy](research/philosophy.md)，维护约束见 [AGENTS](AGENTS.md)。

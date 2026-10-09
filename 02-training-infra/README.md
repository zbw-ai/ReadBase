# Part 02｜训练基础设施

这一部分沿着“容量 → 并行 → 通信 → 稳定运行”理解大规模训练，重点是配置为何成立、瓶颈怎样迁移，以及故障后能否恢复。它是长期学习路线，不是需要一次读完的技术清单；先修入口是 [Part 01](../01-systems/README.md)。

<a id="original-library"></a>
## 原训练手册常用入口

原 `training-infra-roadmap/` 的常用入口集中在这里，直接进入正文或资料，不再经过旧跳转页。跨 Part 正文、研究账本与实践记录仍各自只有一个维护位置。

| 要找的内容 | 直接入口 |
|---|---|
| Megatron / 并行与训练 | [5D 并行总览](topics/distributed_training.md) · [FSDP / ZeRO](topics/fsdp.md) · [TP](topics/tensor_parallelism.md) · [MoE](topics/moe.md) · [长上下文](topics/long_context_training.md) |
| GPU / 通信基础 | [Transformer Engine / GPU / PyTorch](../01-systems/topics/transformer_engine.md) · [NCCL](../01-systems/topics/nccl.md) |
| RL / Agent / Teacher | [Agentic RL](../04-rl-infra/topics/agentic_rl.md) · [Gateway](../04-rl-infra/topics/agentic_rl.md#gateway-streaming-refill) · [框架选型](../04-rl-infra/topics/rl_framework_selection.md) · [MOPD](../04-rl-infra/topics/mopd.md) |
| 具身平台设计 | [Agentic for Embodied](../05-embodied-infra/topics/agentic_for_embodied.md) |
| 原训练学习计划 | [30 天](roadmaps/30_day_plan.md) · [90 天](roadmaps/90_day_plan.md) · [年度计划](roadmaps/yearly_plan.md) |
| 论文、工业报告与 PDF | [论文目录](../research/papers/) · [技术报告目录](../research/tech_reports/) · [阅读总表](../research/MASTER_READING_LIST.md) · [Megatron Core MoE 中文 PDF（5 份）](../research/MASTER_READING_LIST.md#megatron-core-moe-2026-zh-pdf) |
| 扫描、月报与阅读进度 | [研究雷达](../research/tracking/README.md) · [月度复盘 / 月报](../research/tracking/monthly_reviews.md) · [阅读队列](../research/reading_queue/README.md) · [学习日志](../research/learning_log/README.md) |
| 实验、项目与排障 | [实验](../practice/experiments/README.md) · [项目](../practice/projects/) · [Playbooks](../practice/playbooks/README.md) |
| 面试准备与全局地图 | [面试速查控制台](../private_resume/2026-08-llm-infra-interview-prep.md#interview-console) · [Coding 题单](../private_resume/2026-09-interview-coding.md#coding-top) · [知识图谱](../KNOWLEDGE_GRAPH.md) |

## 从哪里开始

| 编号 | 主题入口 | 学习问题与覆盖 |
|---|---|---|
| 2.1 | [分布式训练 / 5D 总览](topics/distributed_training.md) | 先会计算 tensor shape 与显存账本，再回答每个并行维度切什么、引入什么通信 |
| 2.2 | [DP](topics/data_parallelism.md) · [FSDP / ZeRO 与后端选型](topics/fsdp.md) · [ZeRO](topics/zero.md) | 区分模型状态与 activation；FSDP 已展开，独立 ZeRO topic 仍是骨架 |
| 2.3 | [Tensor Parallelism](topics/tensor_parallelism.md) | 以 collective 语义为基础，推导 Linear 前后向的数据布局和通信 |
| 2.4 | [Pipeline Parallelism](topics/pipeline_parallelism.md) · [Sequence Parallelism](topics/sequence_parallelism.md) | 结合 5D 总览理解组合；独立 PP / SP topic 仍是骨架 |
| 2.5 | [长上下文训练](topics/long_context_training.md) · [Context Parallelism](topics/context_parallelism.md) | 追问 activation、重计算、变长负载和 loss 路径的边界 |
| 2.6 | [MoE](topics/moe.md) | 理解专家并行与负载不均；正文已有，排障仍需按 workload 验证 |
| 2.7 | [Checkpointing](topics/checkpointing.md) · [容错](topics/fault_tolerance.md) | 从“能保存”推进到状态一致、可恢复和长期 goodput |

## 问题到验证

| 工程问题 | 正文入口 | 实验 / 排障入口 | 当前状态 |
|---|---|---|---|
| FSDP 后仍 OOM，先查什么？ | [FSDP](topics/fsdp.md) · [ZeRO](topics/zero.md) | [FSDP OOM](../practice/playbooks/fsdp_oom.md) · [A100 实验路线](../practice/experiments/a100_fsdp_io_lab.md) | FSDP 已展开；ZeRO 为骨架；runbook 待补，实验未执行 |
| 扩 TP 还是扩 DP？ | [TP](topics/tensor_parallelism.md) · [DP](topics/data_parallelism.md) | [TP vs DP](../practice/experiments/tensor_parallelism/tp_vs_dp.md) | 正文已有；实验 `NEW`，无结果 |
| 多维并行怎样组合？ | [5D 总览](topics/distributed_training.md) · [PP](topics/pipeline_parallelism.md) · [SP](topics/sequence_parallelism.md) | [Slow Step](../practice/playbooks/slow_step_debug.md) | 总览已有；独立 PP / SP 仍是骨架 |
| MoE 为什么出现 straggler？ | [MoE](topics/moe.md) | [MoE 负载不均](../practice/playbooks/moe_load_imbalance.md) | 正文已有；排障步骤需按 workload 验证 |
| 异步保存是否真的可恢复？ | [Checkpointing](topics/checkpointing.md) · [容错](topics/fault_tolerance.md) | [Async Checkpoint](../practice/experiments/checkpoint/async_checkpoint.md) · [恢复手册](../practice/playbooks/checkpoint_recovery.md) | 正文已有；实验 `NEW`，无结果 |

## 继续深入

按时间安排可沿用训练方向的 [30 天](roadmaps/30_day_plan.md)、[90 天](roadmaps/90_day_plan.md)与[年度计划](roadmaps/yearly_plan.md)；它们不是整个 ReadBase 的统一学习计划。论文和工业报告从[研究阅读总表](../research/MASTER_READING_LIST.md)查找，当前优先级仅在[阅读队列](../research/reading_queue/README.md)维护。

<a id="megatron-core-moe-2026-zh-pdf"></a>
需要视觉导航时，可选看[训练知识分图](../assets/readbase-knowledge-map.svg)；既有的 [Megatron Core MoE 中文翻译五份 PDF](../research/MASTER_READING_LIST.md#megatron-core-moe-2026-zh-pdf)保留为选读入口，术语与关键数字仍需回到原文核对。

[返回总入口](../README.md) · [最近更新](../README.md#recent-updates) · [知识图谱](../KNOWLEDGE_GRAPH.md) · [上一部分：系统基础](../01-systems/README.md) · [下一部分：推理基础设施](../03-inference-infra/README.md)

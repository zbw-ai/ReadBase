# Part II｜训练基础设施

这一部分沿着“容量 → 并行 → 通信 → 稳定运行”理解大规模训练，重点是配置为何成立、瓶颈怎样迁移，以及故障后能否恢复。它是长期学习路线，不是需要一次读完的技术清单；先修入口是 [Part I](../systems/README.md)。

## 从哪里开始

1. [分布式训练总览](topics/distributed_training.md)：先会计算 tensor shape 与显存账本，再回答每个并行维度切什么、引入什么通信。
2. [Data Parallelism](topics/data_parallelism.md) → [FSDP / ZeRO 与后端选型](topics/fsdp.md)：区分模型状态与 activation，说明分片能省什么、不能省什么。
3. [Tensor Parallelism](topics/tensor_parallelism.md)：以前面的 collective 语义为基础，推导 Linear 前后向的数据布局和通信。
4. [长上下文训练](topics/long_context_training.md)：结合 [CP](topics/context_parallelism.md)，追问 activation、重计算、变长负载和 loss 路径的边界。
5. [Checkpointing](topics/checkpointing.md) → [容错](topics/fault_tolerance.md)：从“能保存”推进到状态一致、可恢复和长期 goodput。

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

需要视觉导航时，可选看[训练知识分图](../assets/readbase-knowledge-map.svg)；既有的 [Megatron Core MoE 中文翻译五份 PDF](../research/MASTER_READING_LIST.md#megatron-core-moe-2026-zh-pdf)保留为选读入口，术语与关键数字仍需回到原文核对。

[返回总入口](../README.md) · [最近更新](../README.md#recent-updates) · [知识图谱](../KNOWLEDGE_GRAPH.md) · [上一部分：系统基础](../systems/README.md) · [下一部分：推理基础设施](../inference-infra/README.md)

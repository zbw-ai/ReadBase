# 90 天计划：建立完整训练系统知识图谱

目标：从单篇论文阅读升级到系统设计能力，能回答“为什么这个训练系统这么设计”。

## Phase 1：基础计算图与并行策略（1-30 天）

- Transformer / BERT / GPT-3：理解训练对象。
- Megatron-LM / Megatron 2021：建立 TP + PP + DP 框架。
- GPipe / PipeDream / Alpa：理解 pipeline 调度与自动并行边界。
- 产出：更新 [Tensor Parallelism](../topics/tensor_parallelism.md)、[Pipeline Parallelism](../topics/pipeline_parallelism.md)、[Data Parallelism](../topics/data_parallelism.md)。

## Phase 2：显存、Kernel、MoE（31-60 天）

- ZeRO / FSDP / Offload / Infinity：建立状态分片模型。
- FlashAttention 1/2/3：建立 IO-aware kernel 模型。
- GShard / Switch / DeepSpeed-MoE / Mixtral：建立 Expert Parallel 模型。
- 产出：更新 [Checkpointing](../topics/checkpointing.md)、[FlashAttention](../topics/flashattention.md)、[MoE](../topics/moe.md)。

## Phase 3：超大规模训练系统（61-90 天）

- PaLM / OPT-175B / Llama 3 / DeepSeek-V3 / MegaScale。
- 重点：稳定性、观测、straggler、fault tolerance、distributed checkpoint、NCCL。
- 产出：写一份 `10K GPU Training System Design` 设计笔记，覆盖并行布局、网络、checkpoint、故障恢复和监控。

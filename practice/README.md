# Part VI｜工程实践与生产排障

这一部分把前五部分的机制转成可证伪问题、受控实验和恢复步骤，记录环境、配置、命令、结果及判断变化。实验计划、历史项目证据和生产 runbook 分开维护；有文档不等于执行过，更不等于生产验证。

## 从哪里开始

1. [实验记录规则](experiments/README.md)：先确定问题、baseline、正确性检查和结果口径，再申请资源或运行任务。
2. [A100 / FSDP / IO 实验路线](experiments/a100_fsdp_io_lab.md)：配合 [FSDP 正文](../training-infra/topics/fsdp.md)学习；E00–E08 均未执行，不存在本路线实测收益。
3. [生产排障目录](playbooks/README.md)：有故障时按症状选入口，优先找 first failure 和影响范围，避免先调参数。
4. [长上下文 Agentic RL 项目](projects/2026-q3-long-context-agentic-rl/README.md)：学会区分历史 baseline、诊断 case 与后续实验，最新记录以项目 [STATUS](projects/2026-q3-long-context-agentic-rl/STATUS.md)原日期为准。

## 问题到验证

| 工程问题 | 机制正文 | 实验 / 排障入口 | 当前状态 |
|---|---|---|---|
| 状态分片和 IO 如何影响 step？ | [FSDP](../training-infra/topics/fsdp.md) · [Checkpointing](../training-infra/topics/checkpointing.md) | [A100 实验路线](experiments/a100_fsdp_io_lab.md) | E00–E08 未执行 |
| Attention / 并行配置收益能否复现？ | [FlashAttention](../systems/topics/flashattention.md) · [TP](../training-infra/topics/tensor_parallelism.md) | [Attention benchmark](experiments/flashattention/benchmark.md) · [TP vs DP](experiments/tensor_parallelism/tp_vs_dp.md) | `NEW`；当前只有问题，尚无结果 |
| 故障时如何定位与恢复？ | [容错](../training-infra/topics/fault_tolerance.md) | [NCCL hang](playbooks/tp_nccl_hang.md) · [FSDP OOM](playbooks/fsdp_oom.md) · [checkpoint 恢复](playbooks/checkpoint_recovery.md) | 已有 runbook 结构，部分命令 / 修复细节待补 |
| RL 状态与环境契约怎样验收？ | [Agentic RL](../rl-infra/topics/agentic_rl.md) | [状态边界](experiments/rl_state_boundaries.md) · [环境与 mixer](experiments/mimo_v26_environment_and_mixer.md) | 均为待执行计划 |

## 继续深入

实验改变机制理解时回写对应 Part，形成可复用判断时进入[研究洞察](../research/insights/README.md)，进度与未解问题记入[学习日志](../research/learning_log/README.md)。历史项目保留原测量口径和日期，不能因归档位置改变而被当作本次新结果。

[返回总入口](../README.md) · [最近更新](../README.md#recent-updates) · [知识图谱](../KNOWLEDGE_GRAPH.md) · [上一部分：具身智能](../embodied-infra/README.md) · [按需进入 Part VII：面试](../interview/README.md)

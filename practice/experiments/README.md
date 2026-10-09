# Experiments

[工程实践](../README.md) · [首页](../../README.md) · [知识地图](../../KNOWLEDGE_GRAPH.md)

## 当前学习课程

[A100 E00–E08 实验课程](a100_fsdp_io_lab.md)：环境画像 → FSDP/ZeRO 状态账本 → 通信生命周期 → 多模态 I/O，并保留恢复、跨机、推理与 RL 实验。全部未执行，需先确认环境、预算与授权。

`experiments/` 用来记录实践验证。

很多知识不是读懂的，而是实验懂的。这里记录 benchmark、复现、参数对比、失败案例和工程观察。

## 规则

- 实验可以很小，但必须有问题意识。
- 记录环境、配置、命令、结果和结论。
- 不追求一次成功，失败实验同样有价值。
- 如果实验改变了工程判断，要链接到 `insights/` 或 `topics/`。

## Template

```text
# Experiment Title

## Question

## Environment

## Setup

## Commands

## Results

## Analysis

## Decision Impact

## Follow-up
```

## Current Experiments

- [FlashAttention Benchmark](flashattention/benchmark.md)：验证 IO-aware attention 在不同序列长度、batch size、dtype 下的收益边界。
- [TP vs DP](tensor_parallelism/tp_vs_dp.md)：验证不同 parallelism 配置对 step time 和通信开销的影响。
- [Async Checkpoint](checkpoint/async_checkpoint.md)：验证异步 checkpoint 对 step time spike 和恢复路径的影响。

## 待执行

- [RL 状态边界故障注入](rl_state_boundaries.md)：partial group、checkpoint 发布、refit 后准入；尚未执行。

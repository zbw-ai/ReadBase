# Part 03｜推理基础设施

这一部分关注请求如何转成 GPU 工作，以及延迟、吞吐、显存和调度如何互相约束。目前尚无独立推理 topic，入口复用系统基础和 RL rollout 章节中的相关内容；这些是局部覆盖，不等于完整的 serving 手册。

## 从哪里开始

连续学习用 [Inference 核心路线](roadmap.md)：先成本与 KV，再调度与 SLO；通用 serving 独立于 RL rollout。先修为 [GPU 执行与片上复用](../01-systems/topics/transformer_engine.md#gpu-execution)，下表沿路线五阶段组织已有入口，不代表已有五篇独立 topic。

| 编号 | 主题入口 | 学习问题与覆盖 |
|---|---|---|
| 3.1 | Prefill / decode：[Roofline 与性能测量](../01-systems/topics/transformer_engine.md#roofline) · [decode 工作流](../04-rl-infra/topics/agentic_rl.md#cuda-graph-decode) | 区分计算、搬运与提交成本；独立请求成本模型待补 |
| 3.2 | KV 与前缀复用：[动态 KV 场景](../04-rl-infra/topics/agentic_rl.md#cuda-graph-decode) · [vLLM / SGLang 后端选型](../04-rl-infra/topics/rl_framework_selection.md#vllm-sglang-selection) | 现有内容只覆盖局部机制与 RL 集成；paged KV / prefix cache 正文待补 |
| 3.3 | Batching 与调度：[供给与执行容量](../04-rl-infra/topics/agentic_rl.md#gateway-streaming-refill) | 现有内容仅覆盖 rollout；continuous batching / chunked prefill 正文待补 |
| 3.4 | Kernel、Graph 与量化：[compile / CUDA Graph](../01-systems/topics/transformer_engine.md#torch-compile) · [FP8 验证边界](../01-systems/topics/fp8.md#precision-validation) | 区分 kernel 优化、提交开销与数值约束；推理量化专项待补 |
| 3.5 | SLO 与分布式服务：[NCCL](../01-systems/topics/nccl.md) · [版本准入](../04-rl-infra/topics/agentic_rl.md#rl-state-boundaries) | 现有内容提供底层机制；SLO、容量规划与分布式 serving 正文待补 |

## 问题到验证

| 工程问题 | 当前正文覆盖 | 实验 / 排障入口 | 当前状态 |
|---|---|---|---|
| decode 为什么有 launch 空洞？ | [CUDA Graph decode](../04-rl-infra/topics/agentic_rl.md#cuda-graph-decode) | [A100 实验路线](../practice/experiments/a100_fsdp_io_lab.md) | 有局部机制解释；新路线未执行 |
| 提高并发为什么没有增加有效供给？ | [Gateway 与 rollout 调度](../04-rl-infra/topics/agentic_rl.md#gateway-streaming-refill) | [Rollout Latency](../practice/playbooks/rollout_latency.md) | 仅覆盖 rollout；不直接泛化为在线服务 |
| 后端换得更快，RL 却没更快？ | [后端选型](../04-rl-infra/topics/rl_framework_selection.md#vllm-sglang-selection) | [RL 状态边界实验](../practice/experiments/rl_state_boundaries.md) | 有集成与正确性问题框架；故障注入未执行 |

## 待补边界与继续深入

Prefill / decode 的独立性能模型、KV cache 管理、continuous batching、prefix caching、服务 SLO 与容量规划仍需系统成章。本次只建立导航，不复制现有正文，也不把计划写成已完成章节；后续来源与阅读决策统一进入[研究入口](../research/README.md)。

[返回总入口](../README.md) · [最近更新](../README.md#recent-updates) · [知识图谱](../KNOWLEDGE_GRAPH.md) · [上一部分：训练基础设施](../02-training-infra/README.md) · [下一部分：RL 基础设施](../04-rl-infra/README.md)

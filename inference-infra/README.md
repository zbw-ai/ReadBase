# Part III｜推理基础设施

这一部分关注请求如何转成 GPU 工作，以及延迟、吞吐、显存和调度如何互相约束。目前尚无独立推理 topic，入口复用系统基础和 RL rollout 章节中的相关内容；这些是局部覆盖，不等于完整的 serving 手册。

## 从哪里开始

连续学习用 [Inference 核心路线](roadmap.md)：先成本与 KV，再调度与 SLO；通用 serving 独立于 RL rollout。下方保留按问题查找的正文入口。

1. [GPU 执行与片上复用](../systems/topics/transformer_engine.md#gpu-execution)：先掌握矩阵乘与 tensor layout，理解计算和数据搬运的成本。
2. [Roofline 与性能测量](../systems/topics/transformer_engine.md#roofline)：追问相同 Linear 在不同 batch 下为何会遇到不同瓶颈。
3. [compile 与 CUDA Graph](../systems/topics/transformer_engine.md#torch-compile) → [decode 工作流](../rl-infra/topics/agentic_rl.md#cuda-graph-decode)：区分 kernel 优化与提交开销，检查动态 shape 和内存约束。
4. [vLLM / SGLang rollout 后端选型](../rl-infra/topics/rl_framework_selection.md#vllm-sglang-selection)：以固定 workload 为前提；这里只覆盖 RL 集成视角，不能替代通用 serving 选型。

## 问题到验证

| 工程问题 | 当前正文覆盖 | 实验 / 排障入口 | 当前状态 |
|---|---|---|---|
| decode 为什么有 launch 空洞？ | [CUDA Graph decode](../rl-infra/topics/agentic_rl.md#cuda-graph-decode) | [A100 实验路线](../practice/experiments/a100_fsdp_io_lab.md) | 有局部机制解释；新路线未执行 |
| 提高并发为什么没有增加有效供给？ | [Gateway 与 rollout 调度](../rl-infra/topics/agentic_rl.md#gateway-streaming-refill) | [Rollout Latency](../practice/playbooks/rollout_latency.md) | 仅覆盖 rollout；不直接泛化为在线服务 |
| 后端换得更快，RL 却没更快？ | [后端选型](../rl-infra/topics/rl_framework_selection.md#vllm-sglang-selection) | [RL 状态边界实验](../practice/experiments/rl_state_boundaries.md) | 有集成与正确性问题框架；故障注入未执行 |

## 待补边界与继续深入

Prefill / decode 的独立性能模型、KV cache 管理、continuous batching、prefix caching、服务 SLO 与容量规划仍需系统成章。本次只建立导航，不复制现有正文，也不把计划写成已完成章节；后续来源与阅读决策统一进入[研究入口](../research/README.md)。

[返回总入口](../README.md) · [最近更新](../README.md#recent-updates) · [知识图谱](../KNOWLEDGE_GRAPH.md) · [上一部分：训练基础设施](../training-infra/README.md) · [下一部分：RL 基础设施](../rl-infra/README.md)

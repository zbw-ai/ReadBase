# Part IV｜RL / Agentic RL 基础设施

这一部分把 rollout、reward、training、weight sync 与恢复视为一个持续运行的分布式系统，重点是有效样本供给和状态正确性。先具备 [训练后端](../training-infra/topics/fsdp.md)与[推理执行](../inference-infra/README.md)的基础，再讨论异步是否有收益。

## 从哪里开始

连续学习用 [Agentic RL 六阶段路线](roadmap.md)：先算法／数据契约，再环境、调度、异步、权重和恢复。框架选型作为案例，不另复制一套正文。

1. [Agentic RL 系统地图](topics/agentic_rl.md)：先画出 trajectory 的生产者、消费者和 policy version，回答训练究竟在等什么。
2. [同步、streaming、partial rollout 与 staleness](topics/agentic_rl.md#async-streaming-partial-staleness)：分清执行关系、数据到达方式和样本生命周期。
3. [RL 框架选型](topics/rl_framework_selection.md)：按 workload、placement、后端和二次开发边界比较，不把历史版本判断当成永久排名。
4. [权重同步与状态提交](topics/agentic_rl.md#areal-weight-sync-xccl-disk)：区分传输 artifact、checkpoint 与新版本流量准入。
5. [MOPD](topics/mopd.md)：在理解 rollout / training 数据流后，追问 Teacher 服务、routing 与多领域信号怎样影响系统成本。

## 问题到验证

| 工程问题 | 正文入口 | 实验 / 排障入口 | 当前状态 |
|---|---|---|---|
| Trainer 缺数据，但 GPU 并不全忙？ | [供给与调度](topics/agentic_rl.md#gateway-streaming-refill) | [Rollout Latency](../practice/playbooks/rollout_latency.md) | 已有解释；排障步骤需按真实链路验证 |
| 部分失败后能否安全继续训练？ | [状态提交边界](topics/agentic_rl.md#rl-state-boundaries) | [RL 故障注入](../practice/experiments/rl_state_boundaries.md) | 有不变量与计划；实验未执行 |
| 环境可信，是否就该立即进入训练？ | [环境与消费配比契约](topics/agentic_rl.md#mimo-v26-environment-contract) | [环境与 mixer 验证计划](../practice/experiments/mimo_v26_environment_and_mixer.md) | 有工程判断；验证计划未执行 |
| 多 Teacher 怎样稳定服务 Student？ | [MOPD](topics/mopd.md) | 暂无独立实验记录 | `READING`；原理已展开，代码与性能待验证 |

## 继续深入

[长上下文 Agentic RL 项目](../practice/projects/2026-q3-long-context-agentic-rl/README.md)保存历史证据和实验口径，当前项目状态以其 [STATUS](../practice/projects/2026-q3-long-context-agentic-rl/STATUS.md)日期为准。[研究雷达](../research/tracking/README.md)与[阅读队列](../research/reading_queue/README.md)承接框架变化，不在这里维护另一份更新清单。

[返回总入口](../README.md) · [最近更新](../README.md#recent-updates) · [知识图谱](../KNOWLEDGE_GRAPH.md) · [上一部分：推理基础设施](../inference-infra/README.md) · [下一部分：具身智能](../embodied-infra/README.md)

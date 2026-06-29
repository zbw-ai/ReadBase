# 001: Agentic RL Will Change Training Infrastructure

## 核心判断

Agentic RL 会把训练基础设施从“高效执行大规模矩阵计算”推进到“高效运营一个持续产生经验数据的分布式系统”。

这不是一个小的 post-training 分支，而是训练平台职责边界的扩张。未来的 RL training platform 不仅要训练模型，还要调度 rollout、运行 agent runtime、管理 tool/environment、执行 verifier、控制样本新鲜度、同步权重、追踪 policy lineage，并把这些状态纳入 checkpoint 和 observability。

## 为什么我认为它重要

Pretraining 的系统目标相对清楚：让 GPU 以稳定 batch 消费 token。Agentic RL 的输入不是静态数据集，而是由当前 policy 在环境中生成的 trajectory。这意味着数据生产速度、样本质量、reward 延迟和训练稳定性互相耦合。

从工程角度看，Agentic RL 至少引入四个新的一等问题：

- rollout tail latency：长链推理和工具调用导致 batch 内样本时间差极大；
- sample freshness：样本来自哪个 policy version 会影响训练稳定性；
- runtime observability：prompt、tool call、reward、trace 必须能被复盘；
- train/inference dual state：同一个 actor 在训练布局和推理布局之间反复切换。

这些问题不是 PPO 参数能解决的。它们是平台问题。

## 本周信号

- [AReaL](https://arxiv.org/abs/2505.24298)：用 fully asynchronous RL 解耦 rollout 和 training，把 freshness 和 staleness 变成系统控制项。
- [HybridFlow / verl](https://arxiv.org/abs/2409.19256)：把 RLHF 表达为复杂 dataflow，并正面处理 actor training/generation 之间的 resharding。
- [Agent Lightning](https://arxiv.org/abs/2508.03680)：把 agent execution 和 trainer 解耦，让已有 agent runtime 通过 trace/transition 接入 RL。

这三条信号共同说明：Agentic RL Infra 的主战场不是单个算法，而是 pipeline architecture。

## 对工程决策的影响

如果我要设计下一代训练平台，我会提前预留这些能力：

1. Rollout worker pool 和 training worker pool 分离调度。
2. Trajectory store 支持 policy version、reward version、tokenizer version 和 tool/env version。
3. Reward/verifier 独立扩缩容，并暴露 queue depth 和 p99 latency。
4. Weight sync 有原子切换、版本确认和回滚能力。
5. Checkpoint 不只保存模型状态，还保存 RL pipeline 状态。
6. Observability 覆盖 token/s、request/s、sample freshness、policy idle、verifier backlog。
7. Inference engine 作为训练依赖纳入版本治理，而不是临时服务。

## 需要警惕的误区

- 误区一：把 Agentic RL 当成 SFT 后的一段小训练脚本。
- 误区二：只看训练 GPU utilization，不看 rollout/reward/trainer 的分段等待。
- 误区三：认为异步一定更好，忽略 stale sample 对算法稳定性的影响。
- 误区四：忽略 tokenizer、chat template、tool schema 变化带来的 correctness 问题。
- 误区五：不保存 trace，只保存 reward，导致失败样本无法复盘。

## 下一步

- 精读 AReaL，重点拆 rollout/training 解耦和 staleness 控制。
- 精读 HybridFlow，重点拆 RLHF dataflow 和 actor resharding。
- 阅读 Agent Lightning，重点看 agent trace 如何进入 trainer。
- 用 [Rollout Latency Playbook](../playbooks/rollout_latency.md) 反推一套最小观测指标。

## 结论

Agentic RL 会让 training infra 和 inference infra、agent infra 汇合。谁能把 rollout、verifier、training、checkpoint 和 observability 做成稳定闭环，谁才真正拥有可扩展的 reasoning/agent post-training 能力。

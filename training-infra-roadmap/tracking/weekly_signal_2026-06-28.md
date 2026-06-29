# Weekly Signal Report, 2026-W26

## 本周核心信号

Agentic RL 正在把 post-training 从“训练一个模型”推向“运行一个持续生产 trajectory 的分布式系统”。最重要的变化不是 PPO/GRPO 细节，而是 rollout、reward/verifier、policy update、weight sync、trajectory storage、observability 变成同一条生产流水线里的资源调度问题。

## Top 3: 进入 P0

### AReaL: Large-Scale Asynchronous RL System

- 来源：arXiv / GitHub
- 类型：paper / repo
- 链接：[paper](https://arxiv.org/abs/2505.24298), [repo](https://github.com/areal-project/AReaL)
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：它把 rollout generation 和 policy training 完全解耦，直接命中同步 RL 中 longest rollout 拖慢全局 step 的系统瓶颈。
- Status：READING
- 建议动作：进入 P0
- 预计阅读：2h
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [Distributed Training](../topics/distributed_training.md), [Checkpointing](../topics/checkpointing.md)

AReaL 的核心信号是：LLM RL 的主要瓶颈正在从单次 update 的训练吞吐，转移到 rollout producer 和 training consumer 之间的异步协同。同步系统需要等 batch 中最长输出完成才更新模型，因此 GPU 利用率会被 response length tail 拖垮；AReaL 选择让 rollout workers 持续生成，training workers 收到足够样本就更新。

这对训练 infra 的意义很直接：以后 RL 平台不能只复用 pretraining scheduler。它需要显式管理 sample freshness、policy version、queue depth、staleness bound、reward/verifier latency，并且要能解释“为什么 GPU 忙但训练没进展”。

### verl / HybridFlow

- 来源：arXiv / GitHub
- 类型：paper / repo
- 链接：[paper](https://arxiv.org/abs/2409.19256), [repo](https://github.com/verl-project/verl)
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：它把 RLHF 表达成复杂 dataflow，并把训练态和生成态之间的 actor resharding 作为一等系统问题。
- Status：READING
- 建议动作：进入 P0
- 预计阅读：2h
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [FSDP](../topics/fsdp.md), [Megatron-LM](../papers/megatron_lm.md)

HybridFlow 的价值不只是“又一个 RLHF 框架”。它指出 RLHF dataflow 里的每个节点本身都是一个分布式训练或推理程序，每条边也不是普通 tensor 传递，而是跨 worker、跨模型、跨并行策略的数据依赖。verl 的工程意义在于把 FSDP/Megatron/vLLM/SGLang 等既有 infra 拼成可调度的 RL 数据流。

对工程师最值得读的是 3D-HybridEngine：actor 在 training phase 和 generation phase 的并行策略、显存布局、权重形态不一样，resharding 会成为吞吐和稳定性的关键点。这是 Agentic RL 平台绕不开的“训练-推理双态模型”问题。

### Agent Lightning

- 来源：arXiv / GitHub
- 类型：paper / repo
- 链接：[paper](https://arxiv.org/abs/2508.03680), [repo](https://github.com/microsoft/agent-lightning)
- 影响等级：★★★★
- Decision：Read
- Reason：它把 agent execution 和 RL training 解耦，代表 agent runtime 可能成为训练系统的上游数据平面。
- Status：NEW
- 建议动作：进入 P0
- 预计阅读：1.5h
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [Rollout Latency Playbook](../playbooks/rollout_latency.md)

Agent Lightning 的信号是：Agentic RL 不一定要求重写 agent runtime。它强调 Training-Agent Disaggregation，把现有 agent 框架中的 prompt、tool call、reward、trace 事件转换成可训练 transition。这让 RL infra 从“跑一个固定数据集”变成“接入各种 agent execution runtime”。

这会带来一类新问题：token id drift、trajectory schema、trace observability、tool latency、reward attribution、multi-agent credit assignment 都会进入训练平台边界。它更像是 agent runtime 到 RL trainer 的接口规范，而不只是算法框架。

## Top 10 Signals

### 1. AReaL: Fully Asynchronous RL

- 来源：arXiv / GitHub
- 类型：paper / repo
- 链接：https://arxiv.org/abs/2505.24298
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：异步 rollout/train 解耦是 Agentic RL infra 的关键路线。
- Status：READING
- 建议动作：进入 P0
- 关联主题：rollout / scheduler / freshness / distributed training

一句话价值：把 RL 训练低效问题从算法稳定性扩展为 producer-consumer 系统设计问题。

### 2. verl / HybridFlow

- 来源：arXiv / GitHub
- 类型：paper / repo
- 链接：https://arxiv.org/abs/2409.19256
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：提供 production-ready RL dataflow 抽象和 train/generate resharding 视角。
- Status：READING
- 建议动作：进入 P0
- 关联主题：RLHF dataflow / FSDP / Megatron / vLLM / resharding

一句话价值：说明 RL post-training 是多个分布式程序的编排问题，而不是单个 trainer loop。

### 3. Agent Lightning

- 来源：arXiv / GitHub
- 类型：paper / repo
- 链接：https://arxiv.org/abs/2508.03680
- 影响等级：★★★★
- Decision：Read
- Reason：把 agent runtime 和 trainer 解耦，适合理解真实 agent training 接入层。
- Status：NEW
- 建议动作：进入 P0
- 关联主题：agent runtime / trace / observability / credit assignment

一句话价值：Agentic RL 的输入不再是静态 prompt batch，而是可观测的 agent execution trace。

### 4. SkyRL

- 来源：GitHub
- 类型：repo
- 链接：https://github.com/NovaSky-AI/SkyRL
- 影响等级：★★★★
- Decision：Read
- Reason：全栈 RL library，覆盖 training、agent layer、gym environments，适合跟踪多轮工具调用训练工程化。
- Status：NEW
- 建议动作：进入 P1
- 关联主题：agent training / long-horizon / gym environment / inference-training engine

一句话价值：把 long-horizon tool-use agent training 做成可复用栈，而不是一次性实验脚本。

### 5. DAPO

- 来源：arXiv
- 类型：paper / system recipe
- 链接：https://arxiv.org/abs/2503.14476
- 影响等级：★★★★
- Decision：Read
- Reason：虽然更偏算法 recipe，但它建立在 verl 之上，能帮助理解 reasoning RL 的可复现实验栈。
- Status：NEW
- 建议动作：进入 P1
- 关联主题：GRPO / PPO variants / data filtering / reproducibility

一句话价值：适合作为“算法 recipe 如何压到系统栈上”的案例，不是本周第一优先。

### 6. RLHFless

- 来源：arXiv
- 类型：paper
- 链接：https://arxiv.org/abs/2602.22718
- 影响等级：★★★
- Decision：Observe
- Reason：serverless RLHF 很有想象力，但生产大模型训练是否采用该形态仍需观察。
- Status：NEW
- 建议动作：观察
- 关联主题：elastic resource / serverless / cost-aware scheduling

一句话价值：提示 RL pipeline 的资源需求是动态的，但架构成熟度还需要更多验证。

### 7. AReaL-Hex

- 来源：arXiv
- 类型：paper
- 链接：https://arxiv.org/abs/2511.00796
- 影响等级：★★★
- Decision：Observe
- Reason：异构 GPU 上的 asynchronous RL 调度很贴近成本优化，但应先读 AReaL 主线。
- Status：NEW
- 建议动作：观察
- 关联主题：heterogeneous GPU / scheduler / staleness bound

一句话价值：如果未来训练资源不是同构 H100/H200，异构 rollout/train mapping 会变重要。

### 8. vLLM + OpenRLHF Integration

- 来源：engineering blog
- 类型：blog
- 链接：https://vllm.ai/blog/2025-04-23-openrlhf-vllm
- 影响等级：★★★
- Decision：Read
- Reason：这篇文章把 rollout inference、Ray placement group、vLLM Ray Executor、CUDA IPC/NCCL weight sync 放到同一个 RLHF pipeline 里解释，证据比博客首页更可追溯。
- Status：NEW
- 建议动作：进入 P1
- 关联主题：rollout inference / placement group / weight sync / inference engine

一句话价值：推理引擎已经不是外部 serving 组件，而是 RLHF/Agentic RL 训练闭环的一部分。

### 9. DeepSeek-R1

- 来源：tech report
- 类型：report
- 链接：https://arxiv.org/abs/2501.12948
- 影响等级：★★★★
- Decision：Read
- Reason：R1 是 reasoning RL 需求爆发的起点之一，但本周重点先放系统框架。
- Status：SUMMARIZED
- 建议动作：保持关联
- 关联主题：[DeepSeek-R1](../tech_reports/deepseek_r1.md), RL pipeline / checkpoint lineage

一句话价值：它解释为什么大家开始把 RL reasoning pipeline 当成核心能力，而不是小规模对齐阶段。

### 10. Rollout Latency Playbook

- 来源：internal
- 类型：playbook
- 链接：[rollout_latency.md](../playbooks/rollout_latency.md)
- 影响等级：★★★★
- Decision：Deep Dive
- Reason：读资料必须落到可排障的 runbook，否则无法转化为工程能力。
- Status：DIGESTED
- 建议动作：本周更新
- 关联主题：queue / latency / verifier / policy idle / freshness

一句话价值：把 Agentic RL 的抽象讨论落到“policy update 为什么在等样本”。

## 本周观察

Agentic RL Infra 的主线已经很清楚：pretraining 是同步大规模矩阵计算，Agentic RL 是异步 producer-consumer 系统。前者的核心是 GPU collective、显存和 checkpoint；后者还要额外管理 rollout tail latency、reward/verifier 资源、trajectory schema、policy version、staleness 和 agent runtime observability。

本周不应该贪多。P0 只保留三条：AReaL 看异步系统，verl/HybridFlow 看可编排 RL dataflow，Agent Lightning 看 agent execution 和 trainer 的边界。读完这三条，再进入 OpenRLHF、vLLM + OpenRLHF、SkyRL、DAPO、NeMo RL 这类 P1 工程化材料；RLHFless 和 AReaL-Hex 暂时只观察。

## 下一步动作

- [x] 加入 `reading_queue/P0.md`：AReaL、verl/HybridFlow、Agent Lightning。
- [x] 加入 `reading_queue/P1.md`：OpenRLHF、vLLM + OpenRLHF Integration、SkyRL、DAPO、NVIDIA NeMo RL。
- [ ] 仅索引 / 观察：RLHFless、AReaL-Hex。
- [x] 需要更新的 topic：[Agentic RL](../topics/agentic_rl.md)。
- [x] 需要新增的 insight：[Agentic RL will change training infra](../insights/001_agentic_rl_will_change_training_infra.md)。
- [x] 需要更新的 playbook：[Rollout Latency](../playbooks/rollout_latency.md)。

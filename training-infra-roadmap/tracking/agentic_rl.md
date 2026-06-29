# Agentic RL Tracking

用于追踪 Agentic RL、long-context RL、rollout infra、reward/verifier pipeline、异步采样和训练调度。

这个文件不等同于 RL 算法笔记。这里关注训练基础设施问题：

- rollout 如何调度？
- 长上下文 trajectory 如何存储和切分？
- policy training 与 inference worker 如何解耦？
- reward / verifier 如何成为系统瓶颈？
- PPO / GRPO / DAPO / agentic RL 对 checkpoint lineage 有什么新要求？
- scheduler 如何在 sample efficiency、GPU utilization、freshness 之间取舍？

## 模板

```text
## YYYY-MM-DD

### 标题

- 来源：
- 类型：paper / blog / repo / report
- 链接：
- 影响等级：
- Decision：Ignore / Observe / Read / Deep Dive
- Reason：
- 建议动作：
- Status：
- 关联主题：rollout / verifier / reward / scheduler / long context / checkpoint / distributed training
- 一句话价值：
- 需要追问：
```

## Backlog

## 2026-06-28

### AReaL: A Large-Scale Asynchronous RL System

- 来源：arXiv / GitHub
- 类型：paper / repo
- 链接：https://arxiv.org/abs/2505.24298
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：全异步 rollout/train 解耦直接击中 Agentic RL 的系统瓶颈。
- 建议动作：进入 P0
- Status：READING
- 关联主题：rollout / scheduler / freshness / distributed training / checkpoint
- 一句话价值：把 RL 训练从同步 batch loop 推向 producer-consumer 系统。
- 需要追问：如何设计 staleness bound，既提升吞吐又不破坏训练稳定性？

### HybridFlow / verl

- 来源：arXiv / GitHub
- 类型：paper / repo
- 链接：https://arxiv.org/abs/2409.19256
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：RLHF dataflow 和 actor resharding 是 production RL 平台的基础问题。
- 建议动作：进入 P0
- Status：READING
- 关联主题：RLHF dataflow / FSDP / Megatron / vLLM / SGLang / resharding
- 一句话价值：说明 RL post-training 是多个分布式程序的编排问题。
- 需要追问：training state 和 generation state 之间如何最小成本同步权重？

### Agent Lightning

- 来源：arXiv / GitHub
- 类型：paper / repo
- 链接：https://arxiv.org/abs/2508.03680
- 影响等级：★★★★
- Decision：Read
- Reason：Training-Agent Disaggregation 是 agent runtime 接入 RL trainer 的代表路线。
- 建议动作：进入 P0
- Status：NEW
- 关联主题：agent runtime / trace / observability / credit assignment
- 一句话价值：把 agent execution trace 变成可训练 transition。
- 需要追问：trace schema 如何同时满足训练、排障和复现？

### SkyRL

- 来源：GitHub
- 类型：repo
- 链接：https://github.com/NovaSky-AI/SkyRL
- 影响等级：★★★★
- Decision：Read
- Reason：全栈 RL library 覆盖 training、agent layer、gym environments，但应排在 AReaL/verl 之后。
- 建议动作：进入 P1
- Status：NEW
- 关联主题：long-horizon agent / tool-use / environment / evaluation
- 一句话价值：适合作为多轮工具调用 agent training 的工程化参考。
- 需要追问：SkyRL 的 agent layer 和 rollout/trainer 解耦程度如何？

### RLHFless

- 来源：arXiv
- 类型：paper
- 链接：https://arxiv.org/abs/2602.22718
- 影响等级：★★★
- Decision：Observe
- Reason：serverless RLHF 提醒我们资源需求动态变化，但是否适合大规模生产训练仍需观察。
- 建议动作：观察
- Status：NEW
- 关联主题：elastic resource / cost-aware scheduling / serverless
- 一句话价值：提供一种从资源弹性视角看 RLHF idle time 的路线。
- 需要追问：serverless 形态是否能承受大模型权重加载和 KV cache 状态管理？

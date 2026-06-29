# Historical Backfill

`historical_backfill.md` 用于补录过去几年已经被验证有长期价值、但当前仓库还没有充分吸收的 paper / blog / report / repo / talk / doc。

它不是 Weekly Signal。

| 类型 | 作用 |
|---|---|
| Weekly Signal | 本周新出现、可能改变技术判断的信号 |
| Historical Backfill | 过去已经证明重要、但仓库还没吸收的经典材料 |
| Reading Queue | 从 weekly/backfill 中筛选真正要读的 P0/P1 |
| Topics / Insights / Playbooks | 最终沉淀位置 |

历史材料不要按时间补，要按“它能补哪个工程判断缺口”来补。

## 记录模板

```markdown
## Title

- 原始时间：
- 补录时间：
- 类型：paper / blog / report / repo / talk / doc
- 链接：
- 为什么现在补录：
- 历史影响：
- 今天是否仍有价值：★★★★★ / ★★★★☆ / ★★★☆☆ / ★★☆☆☆ / ★☆☆☆☆
- Decision：Ignore / Observe / Read / Deep Dive
- Reason：
- 建议动作：进入 P0 / P1 / 直接沉淀到 topic / 仅索引
- 关联主题：
- 最终应流向：paper note / engineering blog / topic / insight / playbook / experiment
- 生命周期状态：NEW / READING / SUMMARIZED / DIGESTED / VERIFIED / IMPLEMENTED / OBSOLETE
```

## Agentic RL / Rollout Infra Classics

### AReaL: A Large-Scale Asynchronous RL System

- 原始时间：2025-05
- 补录时间：2026-06
- 类型：paper / repo
- 链接：https://arxiv.org/abs/2505.24298
- 为什么现在补录：当前需要理解 Agentic RL 为什么不能继续依赖同步 rollout -> update loop。
- 历史影响：把 rollout generation 和 policy training 完全解耦，明确把 staleness 当成系统控制项。
- 今天是否仍有价值：★★★★★
- Decision：Deep Dive
- Reason：它补的是“异步 RL 系统如何同时要吞吐和样本新鲜度”的判断缺口。
- 建议动作：进入 P0
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [Rollout Latency](../playbooks/rollout_latency.md)
- 最终应流向：topic / insight / playbook / experiment
- 生命周期状态：READING

### verl / HybridFlow

- 原始时间：2024-09
- 补录时间：2026-06
- 类型：paper / repo
- 链接：https://arxiv.org/abs/2409.19256
- 为什么现在补录：当前需要把 RLHF/Agentic RL 看成多个分布式程序组成的 dataflow，而不是单个 trainer。
- 历史影响：提出 hybrid controller 和 3D-HybridEngine，正面处理 actor training/generation resharding。
- 今天是否仍有价值：★★★★★
- Decision：Deep Dive
- Reason：它补的是“训练态和生成态如何切换、权重如何同步”的判断缺口。
- 建议动作：进入 P0
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [FSDP](../topics/fsdp.md), [Megatron-LM](../papers/megatron_lm.md)
- 最终应流向：topic / experiment / playbook
- 生命周期状态：READING

### Agent Lightning

- 原始时间：2025-08
- 补录时间：2026-06
- 类型：paper / repo
- 链接：https://arxiv.org/abs/2508.03680
- 为什么现在补录：当前需要理解现有 agent runtime 如何接入 RL trainer，而不是为训练重写 agent。
- 历史影响：强调 Training-Agent Disaggregation，把 agent execution trace 转成可训练 transition。
- 今天是否仍有价值：★★★★☆
- Decision：Read
- Reason：它补的是“agent execution 和 training pipeline 的接口边界”判断缺口。
- 建议动作：进入 P0
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [Rollout Latency](../playbooks/rollout_latency.md)
- 最终应流向：insight / playbook
- 生命周期状态：NEW

### SkyRL

- 原始时间：2025
- 补录时间：2026-06
- 类型：repo
- 链接：https://github.com/NovaSky-AI/SkyRL
- 为什么现在补录：AReaL/verl 之外，需要一个 long-horizon tool-use agent training 栈作为横向参照。
- 历史影响：把 training、agent layer、gym environments 和 Tinker-style API 放到同一套 RL library 中。
- 今天是否仍有价值：★★★★☆
- Decision：Read
- Reason：它补的是“多轮工具调用 agent training 如何工程化”的判断缺口。
- 建议动作：进入 P1
- 关联主题：[Agentic RL](../topics/agentic_rl.md), rollout / environment / evaluation
- 最终应流向：engineering blog / topic / experiment
- 生命周期状态：NEW

## RLHF / Post-training Infra Classics

### OpenRLHF

- 原始时间：2024-05
- 补录时间：2026-06
- 类型：paper / repo
- 链接：https://arxiv.org/abs/2405.11143
- 为什么现在补录：它是理解 Ray + vLLM + DeepSpeed 如何拼成 RLHF pipeline 的经典工程材料。
- 历史影响：把 Actor、Critic、Reference、Reward、vLLM engine 等组件显式调度，影响了后续开源 RLHF 栈。
- 今天是否仍有价值：★★★★☆
- Decision：Read
- Reason：它补的是“RLHF 四模型/多组件如何在 GPU 上放置和调度”的判断缺口。
- 建议动作：进入 P1
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [Rollout Latency](../playbooks/rollout_latency.md)
- 最终应流向：engineering blog / topic / playbook
- 生命周期状态：NEW

### DAPO

- 原始时间：2025-03
- 补录时间：2026-06
- 类型：paper / system recipe
- 链接：https://arxiv.org/abs/2503.14476
- 为什么现在补录：它是 reasoning RL 可复现路线的重要材料，且建立在 verl 上。
- 历史影响：把 Decoupled Clip 和 Dynamic Sampling 放进开源大规模 RL 系统，推动 DeepSeek-R1 类训练可复现。
- 今天是否仍有价值：★★★★☆
- Decision：Read
- Reason：它补的是“算法 recipe 如何压到 production RL stack 上”的判断缺口。
- 建议动作：进入 P1
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [DeepSeek-R1](../tech_reports/deepseek_r1.md)
- 最终应流向：paper note / topic
- 生命周期状态：NEW

### RLHFless

- 原始时间：2026-02
- 补录时间：2026-06
- 类型：paper
- 链接：https://arxiv.org/abs/2602.22718
- 为什么现在补录：它从 serverless / elastic resource 角度重新审视 RLHF idle time。
- 历史影响：提示 RLHF pipeline 的资源需求是动态变化的，不同阶段不应被固定资源形态绑死。
- 今天是否仍有价值：★★★☆☆
- Decision：Observe
- Reason：它补的是“RLHF 是否可以更弹性地使用资源”的判断缺口，但生产成熟度需观察。
- 建议动作：仅索引
- 关联主题：[Agentic RL](../topics/agentic_rl.md), scheduler / elastic resource
- 最终应流向：insight / experiment
- 生命周期状态：NEW

### DeepSpeed-Chat

- 原始时间：2023-08
- 补录时间：2026-06
- 类型：paper / repo
- 链接：https://arxiv.org/abs/2308.01320
- 为什么现在补录：它是早期开源端到端 RLHF training pipeline 的工程起点之一。
- 历史影响：把 InstructGPT-style SFT/RM/RLHF pipeline 和 DeepSpeed 优化结合，降低复现门槛。
- 今天是否仍有价值：★★★☆☆
- Decision：Observe
- Reason：它补的是“早期 RLHF 系统如何把训练和推理统一优化”的历史背景。
- 建议动作：仅索引
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [ZeRO](../topics/zero.md)
- 最终应流向：paper note / engineering blog
- 生命周期状态：NEW

### DeepSeek-R1

- 原始时间：2025-01
- 补录时间：2026-06
- 类型：report
- 链接：https://arxiv.org/abs/2501.12948
- 为什么现在补录：它是 reasoning RL 需求爆发的重要背景，解释为什么 rollout infra 突然变成主问题。
- 历史影响：把 GRPO / rule-based reward / distillation 推到行业中心，带动开源 reasoning RL 复现潮。
- 今天是否仍有价值：★★★★★
- Decision：Read
- Reason：它补的是“为什么 Agentic/RL post-training 会改变训练平台需求”的背景缺口。
- 建议动作：直接沉淀到 topic
- 关联主题：[DeepSeek-R1](../tech_reports/deepseek_r1.md), [Agentic RL](../topics/agentic_rl.md)
- 最终应流向：tech report / topic / insight
- 生命周期状态：SUMMARIZED

## Serving Systems Needed by RL Rollout

### vLLM + OpenRLHF Integration

- 原始时间：2025-04
- 补录时间：2026-06
- 类型：blog / doc
- 链接：https://vllm.ai/blog/2025-04-23-openrlhf-vllm
- 为什么现在补录：rollout 生成可能占 RLHF 训练大头，vLLM 如何参与 weight sync 和资源放置是核心工程问题。
- 历史影响：展示 Ray placement group、vLLM Ray Executor、CUDA IPC/NCCL weight sync 在 RLHF 中的用法。
- 今天是否仍有价值：★★★★☆
- Decision：Read
- Reason：它补的是“推理引擎如何成为训练系统一部分”的判断缺口。
- 建议动作：进入 P1
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [Rollout Latency](../playbooks/rollout_latency.md)
- 最终应流向：engineering blog / playbook / experiment
- 生命周期状态：NEW

### Ray RLlib / Ray Train

- 原始时间：2020-2026
- 补录时间：2026-06
- 类型：doc / paper
- 链接：https://docs.ray.io/en/latest/rllib/index.html
- 为什么现在补录：OpenRLHF、AReaL、SkyRL 都大量依赖 Ray-style actor/scheduler 思维。
- 历史影响：RLlib 长期把分布式 RL 视为 dataflow 和 actor 调度问题，Ray Train 则提供通用分布式训练抽象。
- 今天是否仍有价值：★★★☆☆
- Decision：Observe
- Reason：它补的是“RL pipeline 底层调度抽象”的背景缺口，但不应抢占 P0。
- 建议动作：仅索引
- 关联主题：[Distributed Training](../topics/distributed_training.md), [Agentic RL](../topics/agentic_rl.md)
- 最终应流向：engineering blog / topic
- 生命周期状态：NEW

## NVIDIA / Large-scale Training Stack Classics

### NVIDIA NeMo RL

- 原始时间：2026
- 补录时间：2026-06
- 类型：doc / repo
- 链接：https://docs.nvidia.com/nemo/rl/latest/index.html
- 为什么现在补录：NVIDIA 已经把 GRPO、DAPO、reward environment、vLLM rollout、Megatron backend 放进同一个 post-training stack。
- 历史影响：代表厂商训练栈从 pretraining 扩展到 RL post-training、multimodal、long-context 和 cluster deployment。
- 今天是否仍有价值：★★★★☆
- Decision：Read
- Reason：它补的是“NVIDIA training stack 如何进入 RL/post-training”的判断缺口。
- 建议动作：进入 P1
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [Transformer Engine](../topics/transformer_engine.md), [NCCL](../topics/nccl.md)
- 最终应流向：engineering blog / topic / playbook
- 生命周期状态：NEW

### DeepSeek-V3

- 原始时间：2024-12
- 补录时间：2026-06
- 类型：report
- 链接：https://arxiv.org/abs/2412.19437
- 为什么现在补录：V3 不是 Agentic RL 材料，但它是 R1 的 base 和大规模训练工程底座。
- 历史影响：MLA、DeepSeekMoE、FP8、DualPipe、auxiliary-loss-free load balance 都是训练系统重要信号。
- 今天是否仍有价值：★★★★★
- Decision：Read
- Reason：它补的是“reasoning RL 之前的 base model infra 从哪里来”的判断缺口。
- 建议动作：直接沉淀到 topic
- 关联主题：[DeepSeek-V3](../tech_reports/deepseek_v3.md), [MoE](../topics/moe.md), [FP8](../topics/fp8.md)
- 最终应流向：tech report / topic / insight
- 生命周期状态：DIGESTED

## To Be Backfilled Later

- SGLang RL rollout / serving integration：补齐 vLLM 之外的 rollout engine 对照。
- Tinker / Tinker-compatible APIs：补齐“训练服务 API 化”的路线。
- Search-R1 / Tool-use RL recipes：补齐 search environment 和 verifier 设计。
- ReTool / code agent RL recipes：补齐代码类 agent 的 environment、reward 和 sandbox 问题。
- Long-context RL papers：补齐 trajectory 长上下文、KV cache、context truncation 和 CP 的交叉问题。

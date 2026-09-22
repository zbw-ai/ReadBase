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

<a id="mimo-v26-research"></a>

## 2026-09-22：MiMo-V2.6 / CodeMidas 定向研究

用户指定的专题精读，非全量 frontier scan；不改变 [scan_log](scan_log.md) 的覆盖游标。两项都是 9 月新材料，本文完成来源核验和知识沉淀；后续全量 scan 按 Source ID 去重并引用本记录。

| Source ID / First seen | 来源与原始时间 | Impact / Decision / Status | Reason / 一句话价值 / 下一步 |
|---|---|---|---|
| arxiv:2609.22068v1 / 2026-09-22 | [CodeMidas](https://arxiv.org/abs/2609.22068v1)，paper，Bowen Ye、Lei Li、Shicheng Li 等；2026-09-18 17:55:17 UTC | ★★★★★ / Deep Dive / DIGESTED | 从源码构造环境并审核 verifier 误判；补环境供给与奖励可靠性判断；执行环境验收实验 |
| hf:XiaomiMiMo/MiMo-V2.6-Pro-RL:MiMo_V2_6_technical_report.pdf / 2026-09-22 | [MiMo-V2.6 报告](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL/blob/73875d00b30a89ef8cc353a0b60b0e9f9561952d/MiMo_V2_6_technical_report.pdf)，technical report，LLM-Core Xiaomi；随 2026-09-22 发布公开；辅证为[官方直播](https://mimo.xiaomi.com/rl/) | ★★★★★ / Deep Dive / DIGESTED | 披露 grader、混合任务调度、训推一致性和 OOM 恢复边界；补 scale-backed 工业证据；执行 Sample Mixer / 冷启动验证 |

- 核验范围：CodeMidas arXiv title / 19 authors / citation date、PDF 方法与结果；MiMo PDF 标题署名、官方发布日期、直播 status/benchmarks/notices；[来源快照](../assets/mimo_v26/source_snapshot.json)。
- 工程维度：environment、reward、rollout、scheduler、data/trajectory path、training、checkpoint/recovery、inference backend；对 AReaL 可迁移，但未实施。
- 相关主题：[Agentic RL](../topics/agentic_rl.md#mimo-v26-environment-contract)、[MoE](../topics/moe.md)、[Long Context](../topics/long_context_training.md)。
- 输出：[团队分享报告](../tech_reports/mimo_v26.md)、[P1](../reading_queue/P1.md#mimo-v26-reading)、[实验计划](../experiments/mimo_v26_environment_and_mixer.md)。DIGESTED 表示已形成系统判断，不表示实验 VERIFIED。
- 覆盖边界：专题材料精读；没有执行四厂商/HF/RL framework 全量扫描，不补填其 watch 结论。

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

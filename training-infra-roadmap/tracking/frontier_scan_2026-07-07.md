# Frontier Scan, 2026-07-07

- Previous scan：[2026-07-04](frontier_scan_2026-07-04.md)
- Window：2026-07-05 00:00 ~ 2026-07-07 23:59
- Timezone：Asia/Shanghai
- Generated at：2026-07-07
- Report type：flexible frontier scan
- Sources scanned：arXiv cs.LG / cs.AI / cs.DC / cs.CL recent；GitHub / blogs / releases 沿用本轮前次扫描，未在本次论文补扫中新增
- Scan completeness：arXiv 重点方向完成扫描；部分 API 查询受 rate limit 影响，已用 recent list 补查。已补扫 Tue, 7 Jul 2026 latest paper entries。

## 本次核心判断

本次扫描有 2 条 P0 候选：一条是 coding-agent RL 的 rollout infrastructure tax，说明 Agentic RL Infra 的瓶颈正在从 trainer 迁移到 sandbox、execution substrate、trajectory runtime 和 worker-hour 成本；另一条是 CompactionRL，说明 long-horizon agent 的上下文压缩正在进入 RL 训练目标本身，而不是只作为推理阶段的 prompt 工程。另有 verifier、checkpoint/fault tolerance、MoE training stack、long-sequence attention overlap 等 P1 方向值得继续观察。

## Accepted Frontier Signals

### The Rollout Infrastructure Tax in Coding-Agent Reinforcement Learning

- Signal ID：2026-07-07-001
- Source ID：arxiv:2607.01415
- First seen：2026-07-07
- Scan window：2026-07-05 ~ 2026-07-07
- Focus Match：P0 Focus
- 来源：arXiv
- 类型：paper
- 链接：https://arxiv.org/abs/2607.01415
- 发布时间：2026-07
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：它把 coding-agent RL 的主要成本从模型训练拉到 rollout execution substrate，直接对应 sandbox、container、K8s/VM、cold-start、worker-hour、trajectory IO。
- Status：NEW
- 建议动作：进入 P0
- 预计阅读：2h
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [Rollout Latency](../playbooks/rollout_latency.md)

这条信号非常贴近当前主线：RL Infra 不是只优化 policy update，而是要把任务执行环境当成训练系统的一等组件。对代码类 agent 来说，sandbox 冷启动、环境隔离、文件系统、测试执行和轨迹采集会决定训练吞吐和成本。

### CompactionRL: Reinforcement Learning with Context Compaction for Long-Horizon Agents

- Signal ID：2026-07-07-002
- Source ID：arxiv:2607.05378
- First seen：2026-07-07
- Scan window：2026-07-05 ~ 2026-07-07
- Focus Match：P0 Focus
- 来源：arXiv
- 类型：paper
- 链接：https://arxiv.org/abs/2607.05378
- 发布时间：2026-07
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：它把 long-horizon agent 的 context compaction 纳入 RL 训练策略，直接连接长上下文、trajectory、loss normalization、cross-trajectory GAE 和 agentic coding task。
- Status：SUMMARIZED
- 建议动作：已生成 [paper note](../papers/compactionrl.md)，后续沉淀到 topics / playbook / experiment
- 预计阅读：2h
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [Long-context Training](../topics/long_context_training.md), [Rollout Latency](../playbooks/rollout_latency.md)

这条信号很关键：长上下文问题不再只是“模型能不能吃 128k/256k”，而是 long-horizon rollout 训练时如何压缩历史交互状态、如何给 summary generation 分配 credit、如何避免 trajectory 过长把训练吞吐和上下文预算拖垮。它和当前 [Long-context Training](../topics/long_context_training.md) 以及 [Agentic RL](../topics/agentic_rl.md) 两条主线直接交叉。

### LLM-as-a-Verifier: A General-Purpose Verification Framework

- Signal ID：2026-07-07-003
- Source ID：arxiv:2607.05391
- First seen：2026-07-07
- Scan window：2026-07-05 ~ 2026-07-07
- Focus Match：P1 Focus
- 来源：arXiv
- 类型：paper / framework
- 链接：https://arxiv.org/abs/2607.05391
- 发布时间：2026-07
- 影响等级：★★★★☆
- Decision：Read
- Reason：它把 verifier 从简单 LM judge 推向可扩展的连续评分和 criteria decomposition，并明确提到 dense feedback for RL，适合补 reward/verifier pipeline。
- Status：NEW
- 建议动作：进入 P1 候选
- 预计阅读：1h
- 关联主题：[Agentic RL](../topics/agentic_rl.md), verifier / reward pipeline

这条不如 CompactionRL 那么直接改变训练系统边界，但对 Agentic RL Infra 的 verifier/reward 子系统有价值。它提醒我们：未来 rollout infra 不只需要更快生成样本，还需要更稳定、可分解、可重复、可校准的 verifier 作为训练反馈源。

## Observed / Rejected Candidates

| 材料 | Source ID | Focus Match | Decision | 原因 |
|---|---|---|---|---|
| DeadPool: Resilient LLM Training with Hot-Swapping via Zero-Overhead Checkpoint | arxiv:2607.01646 | P0 Focus | Read | checkpoint / fault tolerance 方向强相关，进入 P1 更合适 |
| Mixture-of-Parallelisms: Towards Memory-Efficient Training Stack for Mixture-of-Experts Models | arxiv:2607.01844 | P0 Focus | Read | MoE training stack 与并行组合相关，进入 P1 |
| HCMS: Head-Chunked Multi-Stream Pipeline for Communication-Computation Overlap in Long-Sequence Parallel Attention | arxiv:2607.01817 | P0 Focus | Read | 长序列 attention 通信计算 overlap，与 CP/128k 训练相关，进入 P1 |
| Adaptive Inference Batching using Policy Gradients | arxiv:2607.05272 | P1 Focus | Observe | 推理 batching/routing 与 rollout serving 相关，但目前更像模拟器验证，不进入 P0 |
| Communication-Aware Placement and Pruning for Efficient Mixture-of-Experts Inference | arxiv:2607.05116 | P1 Focus | Observe | MoE inference placement/pruning 相关，可能影响 rollout serving 成本，先观察 |
| Direct Model State Migration for Elastic Training of Large Language Models | arxiv:2607.04749 | P0 Focus | Observe | elastic training / state migration 相关，但本次未能完整读取摘要，暂不提升到 accepted |
| Adaptive Space-efficient Collectives for Dynamic and Unstructured Sparsity on GPU Platforms | arxiv:2607.04676 | P0 Focus | Observe | GPU collectives 与稀疏通信相关，需确认是否能落到 LLM training 主线 |
| Latent Programming Horizons in Coding Agents | arxiv:2607.05188 | P1 Focus | Observe | coding agent 表征分析有趣，但更偏可解释性，不直接改变 infra 判断 |
| SCAPE: Accurate and Efficient LLM Training with Extreme Sparse Communication | arxiv:2607.01678 | P0 Focus | Observe | sparse communication optimizer 方向值得观察，但需要验证规模与代码成熟度 |
| Towards Load-Aware Prefill Deflection for Disaggregated LLM Serving | arxiv:2607.02043 | P1 Focus | Observe | PD disaggregation 和 prefill queue 对 rollout inference 有价值，先观察 |

## Reading Queue Updates

- [ ] 建议加入 `reading_queue/P0.md`：arxiv:2607.01415，待替换当前 P0 时确认
- [x] arxiv:2607.05378 已完成初读并生成 [paper note](../papers/compactionrl.md)
- [ ] 加入 `reading_queue/P1.md`：LLM-as-a-Verifier, DeadPool, Mixture-of-Parallelisms, HCMS
- [ ] 仅观察：Adaptive Inference Batching, Communication-Aware MoE Placement/Pruning, Direct Model State Migration, Adaptive Space-efficient Collectives, Latent Programming Horizons, SCAPE, Load-Aware Prefill Deflection
- [ ] 转入 `tracking/backfill/YYYY-MM.md`：无

## 去重记录

- 本次新增 Source ID：arxiv:2607.01415, arxiv:2607.05378, arxiv:2607.05391
- Follow-up Source ID：无
- 与历史 backfill 重复但未收录：无

## 扫描完整性

- 已扫描来源：arXiv cs.LG / cs.AI / cs.DC / cs.CL recent / 重点关键词
- 未完整扫描来源：部分 arXiv API 查询遇到 rate limit，已用 arXiv recent list 补查
- 已知盲区：官方博客、GitHub release 仍需后续固定来源清单
- 下次优先补扫：rollout infra、long-horizon context compaction、verifier/reward pipeline、checkpoint/fault tolerance、long-context attention overlap

## 下一步动作

- [x] 更新 [Scan Log](scan_log.md)
- [ ] 决定是否将 arxiv:2607.01415 替换进 [P0](../reading_queue/P0.md)
- [x] arxiv:2607.05378 已先完成 paper note，不再等待 P0 替换
- [ ] 阅读 arxiv:2607.01415
- [x] 阅读 arxiv:2607.05378
- [ ] 评估 LLM-as-a-Verifier 是否进入 [P1](../reading_queue/P1.md)
- [ ] 评估 DeadPool 是否进入 checkpointing / fault tolerance topic
- [ ] 评估 HCMS 是否进入 [Long-context Training](../topics/long_context_training.md)

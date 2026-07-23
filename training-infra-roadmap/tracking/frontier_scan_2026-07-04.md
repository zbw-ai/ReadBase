# Frontier Scan, 2026-07-04

- Previous scan：无正式 scan log，按本次作为 2026-07-01 起点回补
- Publication window：2026-07-01 00:00 ~ 2026-07-04 23:59
- Rescanned at：2026-07-08 10:58
- Timezone：Asia/Shanghai
- Report type：historical correction for frontier scan
- Sources scanned：arXiv cs.LG / cs.AI / cs.CL / cs.DC recent pages；重点覆盖 2026-07-01 到 2026-07-04 的 paper 条目
- Scan completeness：arXiv 重点方向完成回补扫描；该文件修正旧版 2026-07-04 记录只扫到部分候选的问题。官方博客源在 [2026-07-08 scan](frontier_scan_2026-07-08.md) 中补扫。

## 本次核心判断

2026-07-01 到 2026-07-04 的高价值信号集中在 **Agentic RL rollout substrate、MoE/long-sequence training stack、fault tolerance 和 long-context serving**。最重要的一条仍是 Rollout Infrastructure Tax：它把 coding-agent RL 的成本中心从 trainer 推到 sandbox / execution substrate / worker-hour。其余信号更适合进入 P1，用来补 MoE 并行、hot-swapping fault tolerance 和 long-sequence attention overlap。

## Accepted Frontier Signals

### The Rollout Infrastructure Tax in Coding-Agent Reinforcement Learning

- Signal ID：2026-07-04-001
- Source ID：arxiv:2607.01415
- First seen：2026-07-04
- Scan window：2026-07-01 ~ 2026-07-04
- Focus Match：P0 Focus
- 来源：arXiv
- 类型：paper
- 链接：https://arxiv.org/abs/2607.01415
- 发布时间：2026-07-01
- Primary-source check：title / authors / date / abstract 已对齐 arXiv abs 页
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：它把 coding-agent RL 的主要成本从 trainer 内部推到 rollout execution substrate，直接对应 sandbox、cold start、worker-hour、trajectory IO。
- Status：NEW
- 建议动作：已进入 [P0](../reading_queue/P0.md)
- 预计阅读：2h
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [Rollout Latency](../playbooks/rollout_latency.md)

这条信号是当前 RL Infra 最贴近生产系统的问题：训练吞吐不只由 GPU step time 决定，也由 rollout 环境启动、执行隔离、测试运行、轨迹采集和失败重试决定。它适合直接流向 rollout latency playbook 和 agentic RL topic。

### Next-Generation Agentic Reinforcement Learning Systems Enable Self-Evolving Agents

- Signal ID：2026-07-04-002
- Source ID：arxiv:2607.01120
- First seen：2026-07-04
- Scan window：2026-07-01 ~ 2026-07-04
- Focus Match：P1 Focus
- 来源：arXiv
- 类型：paper
- 链接：https://arxiv.org/abs/2607.01120
- 发布时间：2026-07
- 影响等级：★★★★☆
- Decision：Read
- Reason：它把 agentic online RL 从算法 recipe 提升为系统架构问题，触达 rollout、trajectory protocol、data proxy 和 control plane。
- Status：NEW
- 建议动作：进入 P1 候选；先读 Rollout Infrastructure Tax
- 预计阅读：1h
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [Rollout Latency](../playbooks/rollout_latency.md)

### Mixture-of-Parallelisms: Towards Memory-Efficient Training Stack for Mixture-of-Experts Models

- Signal ID：2026-07-04-003
- Source ID：arxiv:2607.01844
- First seen：2026-07-04
- Scan window：2026-07-01 ~ 2026-07-04
- Focus Match：P1 Focus
- 来源：arXiv
- 类型：paper
- 链接：https://arxiv.org/abs/2607.01844
- 发布时间：2026-07-02
- Primary-source check：title / authors / date / abstract 已对齐 arXiv abs 页
- 影响等级：★★★★☆
- Decision：Read
- Reason：它关注 MoE training stack 中多种并行方式组合和显存效率，适合后续扩写 MoE / distributed training 主题。
- Status：NEW
- 建议动作：已进入 [P1](../reading_queue/P1.md)
- 预计阅读：1h
- 关联主题：[MoE](../topics/moe.md), [Tensor Parallelism](../topics/tensor_parallelism.md), [Distributed Training](../topics/distributed_training.md)

### PHOENIX: Resilient LLM Training with Hot-Swapping via Zero-Overhead Checkpoint

- Signal ID：2026-07-04-004
- Source ID：arxiv:2607.01646
- First seen：2026-07-04
- Scan window：2026-07-01 ~ 2026-07-04
- Focus Match：P1 Focus
- 来源：arXiv
- 类型：paper
- 链接：https://arxiv.org/abs/2607.01646
- 发布时间：2026-07-02
- Primary-source check：title / authors / date / abstract 已对齐 arXiv abs 页
- 影响等级：★★★★☆
- Decision：Read
- Reason：它面向 LLM 训练故障恢复，用 hot-swapping、in-memory checkpoint 和 communicator reconstruction 降低永久节点失败的恢复代价。
- Status：NEW
- 建议动作：已进入 [P1](../reading_queue/P1.md)
- 预计阅读：1h
- 关联主题：[Checkpointing](../topics/checkpointing.md), [Fault Tolerance](../topics/fault_tolerance.md), [MegaScale](../tech_reports/megascale.md)

### HCMS: Head-Chunked Multi-Stream Pipeline for Communication-Computation Overlap in Long-Sequence Parallel Attention

- Signal ID：2026-07-04-005
- Source ID：arxiv:2607.01817
- First seen：2026-07-04
- Scan window：2026-07-01 ~ 2026-07-04
- Focus Match：P1 Focus
- 来源：arXiv
- 类型：paper
- 链接：https://arxiv.org/abs/2607.01817
- 发布时间：2026-07-02
- Primary-source check：title / authors / date / abstract 已对齐 arXiv abs 页
- 影响等级：★★★★☆
- Decision：Read
- Reason：它直接触达 long-sequence attention 的通信计算 overlap，对 context parallel / 128k 训练性能分析有参考价值。
- Status：NEW
- 建议动作：已进入 [P1](../reading_queue/P1.md)
- 预计阅读：1h
- 关联主题：[Long-context Training](../topics/long_context_training.md), [Context Parallelism](../topics/context_parallelism.md), [FlashAttention](../topics/flashattention.md)

## Observed / Rejected Candidates

| 材料 | Source ID | Focus Match | Decision | 原因 |
|---|---|---|---|---|
| MosaicKV: Serving Long-Context LLM with Dynamic Two-D KV Cache Compression | arxiv:2607.00760 | P1 Focus | Observe | long-context serving / KV cache 相关，但更偏推理 serving，暂不挤占训练主线 |
| Lynx: Progressive Speculative Quantization for accelerating KV Transfer in Long-Context Inference | arxiv:2607.01831 | P1 Focus | Observe | disaggregated inference KV transfer 与 rollout inference 有关，先观察 |
| ELDR: Expert-Locality-Aware Decode Routing for PD-Disaggregated MoE Serving | arxiv:2607.00466 | P1 Focus | Observe | MoE serving / PD disaggregation 相关，等 MoE inference 主线启动后再读 |
| Towards Load-Aware Prefill Deflection for Disaggregated LLM Serving | arxiv:2607.02043 | P1 Focus | Observe | prefill queue / disaggregated serving 对 rollout inference 有价值，但本轮不进入队列 |
| SmoothAgent: Efficient Long-Horizon LLM-Based Agent Serving with Lookahead Context Engineering | arxiv:2607.00151 | P1 Focus | Observe | agent serving 相关，但更像 serving 策略，等 rollout serving playbook 扩写时再补 |
| SCAPE: Accurate and Efficient LLM Training with Extreme Sparse Communication | arxiv:2607.01678 | P1 Focus | Observe | sparse communication optimizer 方向有趣，需要确认规模和代码成熟度 |
| GPUAlert: A Zero-Instrumentation Process-Boundary Monitor for Diagnosing GPU Training-Job Failures | arxiv:2607.01409 | P1 Focus | Observe | 生产排障相关，但需先确认是否适合 training infra playbook |
| Evidence-State Rewards for Long-Context Reasoning | arxiv:2607.02073 | P1 Focus | Observe | reward / long-context 相关，算法信号强于系统信号 |
| Are Performance-Optimization Benchmarks Reliably Measuring Coding Agents? | arxiv:2607.01211 | P1 Focus | Observe | coding agent benchmark 相关，但不直接改变 infra 设计 |

## OpenAI / Anthropic / NVIDIA Watch

| Vendor | Sources checked | Decision | 结果 |
|---|---|---|---|
| OpenAI | Not scanned in this correction file | Deferred | 本文件只修正 2026-07-01 到 2026-07-04 的 arXiv paper 窗口；OpenAI official sources 由 [2026-07-08 scan](frontier_scan_2026-07-08.md) 承担。 |
| Anthropic | Not scanned in this correction file | Deferred | 本文件只修正 arXiv paper 窗口；Anthropic official sources 由 [2026-07-08 scan](frontier_scan_2026-07-08.md) 记录其可用性限制。 |
| NVIDIA | Not scanned in this correction file | Deferred | 本文件不覆盖官方博客；7 月 NVIDIA Nonuniform TP 已在 [2026-07-08 scan](frontier_scan_2026-07-08.md) 进入 accepted。 |

## RL Framework Watch: Historical Audit Backfill

> 回补说明：本节于 2026-07-23 按原扫描窗口复核官方 release 与 merged PR。它补充框架演进证据，不修改当时的 Accepted、P0/P1 或 cursor 记录。

| Framework | Window change | Subsystem | Evidence / state | Decision | 对 AReaL 的参考 |
|---|---|---|---|---|---|
| AReaL | [v2.0.0](https://github.com/areal-project/AReaL/releases/tag/v2.0.0)：训练、推理、Agent 与 weight update 服务化，并提供统一 CLI 和 Hermes RL 示例 | scheduler / rollout / training / weight sync | official release；2026-07-02 00:23（Asia/Shanghai）发布 | Read | 这是 AReaL 自身从单体训练框架走向 RL service architecture 的基线，后续优化应围绕 service boundary、failure domain 和跨服务状态一致性评估 |
| slime | [PR #2089](https://github.com/THUDM/slime/pull/2089)：disaggregated rollout 的 disk-level delta weight sync | weight sync / inference backend | merged；2026-07-02 17:20（Asia/Shanghai） | Read | 对照 AReaL AWEX/disk mode：重点比较 delta 生成成本、全量 fallback、版本校验和远端 rollout 恢复语义 |

## Reading Queue Updates

- [x] 保持 [P0](../reading_queue/P0.md)：arxiv:2607.01415 已进入 P0。
- [x] 保持 [P1](../reading_queue/P1.md)：PHOENIX / Mixture-of-Parallelisms / HCMS 已在 P1。
- [ ] P1 候选：arxiv:2607.01120，等读完 Rollout Infrastructure Tax 后再决定是否加入。
- [ ] 仅观察：MosaicKV, Lynx, ELDR, Load-Aware Prefill Deflection, SmoothAgent, SCAPE, GPUAlert。

## 去重记录

- 本次新增 Source ID：arxiv:2607.01415, arxiv:2607.01120, arxiv:2607.01844, arxiv:2607.01646, arxiv:2607.01817
- Follow-up Source ID：无
- 与历史 scan 重复但归档修正：arxiv:2607.01415 / 2607.01844 / 2607.01646 / 2607.01817 曾出现在旧 2026-07-07 记录中；按发布时间应归入本窗口。

## 扫描完整性

- 已扫描来源：arXiv cs.LG / cs.AI / cs.CL / cs.DC recent pages，分页覆盖 2026-07-01 到 2026-07-04。
- 未完整扫描来源：本文件只修正 paper 窗口；官方博客和技术报告源由 [2026-07-08 scan](frontier_scan_2026-07-08.md) 处理。
- 已知盲区：arXiv 之外的 GitHub repo release 在本窗口没有做全量自动扫描。
- 下次优先补扫：Agentic RL repo / official framework release notes。

## 下一步动作

- [x] 更新 [Scan Log](scan_log.md)
- [ ] 阅读 [Rollout Infrastructure Tax](https://arxiv.org/abs/2607.01415)
- [ ] 读完后更新 [Agentic RL](../topics/agentic_rl.md) 和 [Rollout Latency](../playbooks/rollout_latency.md)
- [ ] 后续评估是否将 arxiv:2607.01120 加入 P1

# Frontier Scan, 2026-07-07

- Previous scan：[2026-07-04](frontier_scan_2026-07-04.md)
- Window：2026-07-05 00:00 ~ 2026-07-07 23:59
- Timezone：Asia/Shanghai
- Generated at：2026-07-07
- Report type：flexible frontier scan
- Sources scanned：arXiv / GitHub / blogs / releases
- Scan completeness：arXiv 重点方向完成扫描；部分 API 查询受 rate limit 影响，已用 recent list 补查

## 本次核心判断

本次扫描有 1 条 P0：coding-agent RL 的 rollout infrastructure tax 已经被单独抽象出来，说明 Agentic RL Infra 的瓶颈正在从 trainer 迁移到 sandbox、execution substrate、trajectory runtime 和 worker-hour 成本。另有 checkpoint/fault tolerance、MoE training stack、long-sequence attention overlap 等 P1 方向值得继续观察。

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

## Observed / Rejected Candidates

| 材料 | Source ID | Focus Match | Decision | 原因 |
|---|---|---|---|---|
| DeadPool: Resilient LLM Training with Hot-Swapping via Zero-Overhead Checkpoint | arxiv:2607.01646 | P0 Focus | Read | checkpoint / fault tolerance 方向强相关，进入 P1 更合适 |
| Mixture-of-Parallelisms: Towards Memory-Efficient Training Stack for Mixture-of-Experts Models | arxiv:2607.01844 | P0 Focus | Read | MoE training stack 与并行组合相关，进入 P1 |
| HCMS: Head-Chunked Multi-Stream Pipeline for Communication-Computation Overlap in Long-Sequence Parallel Attention | arxiv:2607.01817 | P0 Focus | Read | 长序列 attention 通信计算 overlap，与 CP/128k 训练相关，进入 P1 |
| SCAPE: Accurate and Efficient LLM Training with Extreme Sparse Communication | arxiv:2607.01678 | P0 Focus | Observe | sparse communication optimizer 方向值得观察，但需要验证规模与代码成熟度 |
| Towards Load-Aware Prefill Deflection for Disaggregated LLM Serving | arxiv:2607.02043 | P1 Focus | Observe | PD disaggregation 和 prefill queue 对 rollout inference 有价值，先观察 |

## Reading Queue Updates

- [ ] 建议加入 `reading_queue/P0.md`：arxiv:2607.01415，待替换当前 P0 时确认
- [ ] 加入 `reading_queue/P1.md`：DeadPool, Mixture-of-Parallelisms, HCMS
- [ ] 仅观察：SCAPE, Load-Aware Prefill Deflection
- [ ] 转入 `tracking/backfill/YYYY-MM.md`：无

## 去重记录

- 本次新增 Source ID：arxiv:2607.01415
- Follow-up Source ID：无
- 与历史 backfill 重复但未收录：无

## 扫描完整性

- 已扫描来源：arXiv recent / 重点关键词
- 未完整扫描来源：部分 arXiv API 查询遇到 rate limit，已用 arXiv recent list 补查
- 已知盲区：官方博客、GitHub release 仍需后续固定来源清单
- 下次优先补扫：rollout infra、checkpoint/fault tolerance、long-context attention overlap

## 下一步动作

- [x] 更新 [Scan Log](scan_log.md)
- [ ] 决定是否将 arxiv:2607.01415 替换进 [P0](../reading_queue/P0.md)
- [ ] 阅读 arxiv:2607.01415
- [ ] 评估 DeadPool 是否进入 checkpointing / fault tolerance topic
- [ ] 评估 HCMS 是否进入 [Long-context Training](../topics/long_context_training.md)

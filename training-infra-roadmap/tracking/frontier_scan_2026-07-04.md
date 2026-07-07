# Frontier Scan, 2026-07-04

- Previous scan：无正式 scan log，按本次作为 frontier scan 起点
- Window：2026-06-29 00:00 ~ 2026-07-04 23:59
- Timezone：Asia/Shanghai
- Generated at：2026-07-04
- Report type：flexible frontier scan
- Sources scanned：arXiv / GitHub / blogs / releases
- Scan completeness：完成本轮重点方向扫描；作为从 weekly 模型切换到 frontier scan 模型后的第一条记录

## 本次核心判断

本次扫描有 1 条值得进入 P0 的前沿信号：Agentic RL 正在从算法 recipe 走向系统论文，开始正面讨论 trajectory data protocol、data proxy、control plane 和 online RL 系统边界。其他材料多与长上下文 inference、KV cache 或 serving 相关，值得观察，但不应挤占 P0。

## Accepted Frontier Signals

### Next-Generation Agentic Reinforcement Learning Systems Enable Self-Evolving Agents

- Signal ID：2026-07-04-001
- Source ID：arxiv:2607.01120
- First seen：2026-07-04
- Scan window：2026-06-29 ~ 2026-07-04
- Focus Match：P0 Focus
- 来源：arXiv
- 类型：paper
- 链接：https://arxiv.org/abs/2607.01120
- 发布时间：2026-07
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：它把 agentic online RL 从“训练算法”提升为系统架构问题，直接触达 rollout、trajectory protocol、data proxy 和 control plane。
- Status：NEW
- 建议动作：进入 P0
- 预计阅读：2h
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [Rollout Latency](../playbooks/rollout_latency.md)

这条信号的价值在于：它不是又一个 GRPO/DAPO recipe，而是把 Agentic RL 的系统边界讲清楚。未来 RL Infra 很可能围绕 rollout runtime、sample freshness、训练/推理解耦、trajectory store 和 control plane 重构，而不是只围绕 trainer 内部优化。

## Observed / Rejected Candidates

| 材料 | Source ID | Focus Match | Decision | 原因 |
|---|---|---|---|---|
| TraceLab: Characterizing Coding Agent Workloads for LLM Serving | arxiv:2606.30560 | P1 Focus | Read | coding-agent workload trace 对 rollout serving 有参考价值，但更像 P1 背景材料 |
| Lynx: Progressive Speculative Quantization for accelerating KV Transfer in Long-Context Inference | arxiv:2607.01831 | P1 Focus | Read | disaggregated inference KV transfer 与 rollout inference 相关，但不是训练主线 P0 |
| MosaicKV | arxiv:2607.01016 | P1 Focus | Observe | KV cache 优化方向相关，先观察工程可落地性 |
| RaBitQCache | arxiv:2606.31519 | P1 Focus | Observe | KV cache 压缩相关，暂不进入 P0 |
| SeKV | arxiv:2606.31145 | P1 Focus | Observe | 长上下文 KV 方向相关，但系统影响需要继续验证 |
| ReContext | arxiv:2607.02509 | P1 Focus | Observe | 长上下文方向相关，先观察 |
| 3DLS | arxiv:2607.01617 | P1 Focus | Observe | serving / scheduling 相关，但与当前训练 infra 主线距离略远 |

## Reading Queue Updates

- [ ] 建议加入 `reading_queue/P0.md`：arxiv:2607.01120，待替换当前 P0 时确认
- [ ] 加入 `reading_queue/P1.md`：TraceLab, Lynx
- [ ] 仅观察：MosaicKV, RaBitQCache, SeKV, ReContext, 3DLS
- [ ] 转入 `tracking/backfill/YYYY-MM.md`：无

## 去重记录

- 本次新增 Source ID：arxiv:2607.01120
- Follow-up Source ID：无
- 与历史 backfill 重复但未收录：无

## 扫描完整性

- 已扫描来源：arXiv 重点方向、GitHub / blogs / releases 的公开更新线索
- 未完整扫描来源：无正式自动化列表，后续需要固定来源清单
- 已知盲区：厂商博客和 GitHub release 仍依赖人工检索
- 下次优先补扫：Agentic RL / rollout infra / long-context training 方向的新 arXiv 与官方工程博客

## 下一步动作

- [x] 更新 [Scan Log](scan_log.md)
- [ ] 决定是否将 arxiv:2607.01120 替换进 [P0](../reading_queue/P0.md)
- [ ] 阅读 arxiv:2607.01120
- [ ] 判断是否更新 [Agentic RL](../topics/agentic_rl.md)
- [ ] 判断是否更新 [Rollout Latency](../playbooks/rollout_latency.md)

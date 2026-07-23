# 2026 Q3 Long-context Agentic RL

这个项目是 Q3 的工程实战主线：以 AReaL/Trail 和 Qwen3.5-9B-128K R2E-Gym 训练为对象，建立能够解释、复现和优化长上下文 Agentic RL 效率的完整方法。

主要目标不是继续扩充知识条目，而是把一次真实训练拆成可观测的因果链，最终在固定 workload、资源和算法正确性约束下降低 E2E step time。框架设计判断和个人能力成长是这条主线的自然产物。

## Reading Contract

对于项目 owner：

- 默认只读 [Current Status](STATUS.md)。
- 每次实验只审核 scorecard 顶部的 Owner Review；需要追溯时再进入详细证据。
- 只有方向变化、guardrail 失败或需要资源决策时，Agent 才要求阅读长文档。

对于 Agent：

- 每次任务先读 `STATUS.md`，不能依赖聊天历史补全事实。
- 性能比较再读 `dashboard.md` 和对应 run；实验设计再读 metric contract 和 scorecard。
- 任务结束后更新 `STATUS.md` 的 baseline、判断、下一步或 owner attention；详细证据留在冷层文档。

## Canonical Scenario

| 项目 | 约束 |
|---|---|
| Model | Qwen3.5-9B |
| Context | 128K |
| Workload | R2E-Gym / SWE-style Agentic RL |
| Framework | AReaL/Trail |
| Primary bottleneck | rollout，具体归因持续验证 |
| July baseline | [R8b](baselines/R8b.md) |
| Current diagnostic case | [bs2 long-trajectory case](cases/bs2-eqtraj-C1b-v2.md) |
| Primary objective | constrained end-to-end step time |

## Primary Objective

```text
minimize post-warmup end-to-end step time

subject to:
  fixed logical training batch and group semantics
  fixed maximum sequence length
  fixed total budget of 32 A100-80GB GPUs
  no regression in downstream evaluation of trained checkpoints
  algorithm correctness and lineage guardrails pass
```

在当前配置中，固定 logical workload 指每个 step 消费 32 个 cohort、每个 cohort 8 条 trajectory，共 256 条，并固定 `max_seq_len=128K`。Trajectory 实际产生和参与训练的 token 数、turn 数及实际 context length 可以变化；如果它们下降且 reward/训练效果不降低，这正是有效优化，而不是 workload 偷减。

异步 rollout/train 场景中的 E2E step time，定义为连续 optimizer/update 完成点之间的 steady-state wall-clock interval。对于包含 K 个连续 interval 的 post-warmup 窗口，均值等于 `window makespan / K`。不能把 rollout active time 和 train active time直接相加，因为二者可能重叠。

E2E step time 必须和下面三类约束一起看：

- 算法正确性：policy version、interaction/trajectory lineage、GRPO group、logp、mask、importance ratio。
- 行为分布：训练中 reward、turn count、context length、failure origin、no-EOS/truncation。
- 资源与数据量：固定 32 张 A100-80GB 总预算、cohort 数、group size、最大序列长度、task/seed/checkpoint。
- 最终效果：训练所得 checkpoint 的下游任务评测；训练 reward 只作在线参考。

Rollout/train 的资源划分和并行拓扑允许作为单一实验变量，只要总预算仍是 32 张 A100-80GB。每个案例必须同时报告 trainer-consumed cohort/trajectory、full-sequence token、loss-active token 和 gradient-active token；历史 tracing 缺失时明确标记 `UNKNOWN`。Reward 正负用于描述训练数据组成，不用于把实际参与训练的数据判成浪费。

## Navigation

- [Current Status](STATUS.md)：owner 默认只需要阅读的一页状态。
- [Performance Dashboard](dashboard.md)：所有 baseline 和实验的统一对比入口。
- [R8b Baseline](baselines/R8b.md)：当前 canonical baseline 的证据与限制。
- [bs2 Long-trajectory Case](cases/bs2-eqtraj-C1b-v2.md)：小 batch 下的 cache、80-turn 和高成本长尾分析。
- [Trajectory Utilization Ledger](analysis/trajectory-utilization-ledger.md)：从 `2000 generated -> 960 consumed -> 1040 unconsumed` 追踪样本去向、回收机会和验证实验。
- [Cohort Recovery Implementation Plan](plans/2026-07-22-cohort-recovery-implementation-plan.md)：从 lineage、runtime-ready sandbox、reset retry 到 manifest 的落地顺序与验收门槛。
- [Experiment Scorecard Template](templates/experiment_scorecard.md)：每次实验必须填写的统一格式。
- [Metric Contract](instrumentation/metric_contract.md)：字段、统计窗口、join key 和计算口径。
- [Structured Metrics](metrics/)：与 scorecard 同口径的 JSON，供校验和自动 dashboard 使用。
- [Experiments](experiments/README.md)：受控实验记录和状态。

## Working Loop

1. 冻结 baseline manifest：代码、镜像、模型、数据、资源和配置。
2. 先运行 observation-only tracing，不改变 agent 行为和调度语义。
3. 使用固定 checkpoint、task set 和 seed 做 rollout replay。
4. 每次只改变一个主要因素，先做小规模 A/B。
5. 同时比较性能、trajectory 分布和算法正确性。
6. 通过小实验后，再进入连续在线训练验证。
7. 把结果写入 scorecard，并更新 dashboard 和工程判断。

## Current Bottleneck Ranking

| Priority | Bottleneck | Current evidence | Status |
|---|---|---|---|
| P0 | LLM request path 与 context growth | bs2 episode elapsed 与累计 LLM RPC / max prompt 的相关系数为 0.906 / 0.660 | Strong evidence |
| P0 | 8-way cohort straggler 放大长尾 | 第 7 条结束后等待最后一条的时间 p95 约 49.8 min | Strong evidence |
| P0 | Agent horizon 产生高成本长尾 | bs2 的 375 条 80-turn episode 仅 10 条 reward > 0；是否 early-stop 仍需训练效果验证 | Cost evidence strong |
| P1 | Cache 收益在 late turn 退化 | request cache ratio 从 turn 10-19 的 95.44% 降到 turn 20+ 的 69.45% | Measured；causality pending |
| P1 | 高成本长轨迹改变 rollout 供给 | R8b / bs2 非正 reward trajectory 分别占 generation-pool wall time 约 59.9% / 41.7%；是否应 early-stop 必须由训练效果验证 | Cost evidence strong；optimization value pending |
| P2 | Sandbox reliability | 显式异常 release 约 3.16% | Secondary |

当前结论是方向判断，不是提前指定优化方案。任何 early stop、cohort policy 或 routing 修改都需要受控实验确认。

# Long-context Agentic RL Performance Dashboard

这是本项目所有性能实验的统一入口。每个 run 在 `experiments/` 或 `baselines/` 中保留完整 scorecard；本页只展示可直接比较的核心指标。

项目 owner 默认只需阅读 [Current Status](STATUS.md)。本页用于跨实验比较和需要进一步审计时查看。

## Reading Rules

- `Canonical`：当前正式 baseline。
- `Candidate`：待与 baseline 比较。
- `Accepted`：性能收益通过正确性 guardrail。
- `Rejected`：性能无收益或行为/算法分布回归。
- `UNKNOWN` / `N/A`：当前 tracing 无法可靠计算，不能解释为 0。
- 默认比较窗口是去掉 warmup 后的完整 step；其他窗口必须单独标注。
- 主排序指标是固定 logical batch（32x8）、最大序列长度（128K）和 32 张 A100-80GB 总预算下，overlap-aware post-warmup update interval / window makespan。历史 run 暂用 logged trainer phase sum，并明确标注 proxy。
- Trainer-consumed、loss-active、gradient-active 和 correctness 是约束与解释指标，防止通过暗降训练参与量获得虚假加速。
- 训练 reward 是在线参考；最终效果判定来自训练所得 checkpoint 的下游任务评测。
- trajectory、cohort、step 和 vLLM log 如果不能通过 lineage 精确 join，必须分表展示。

## Run Registry

| Run | Role | Date | Steps | Model / context | Rollout / train GPUs | Primary change | Decision |
|---|---|---:|---:|---|---:|---|---|
| [R8b](baselines/R8b.md) | Canonical | 2026-07-06 | 7 uninterrupted | Qwen3.5-9B / 128K | 16 / 16 | Baseline | Keep |
| [bs2 C1b v2](cases/bs2-eqtraj-C1b-v2.md) | Diagnostic | 2026-07-20 | 60 completed | Qwen3.5-9B / 128K | 16 / 16 | Small-batch tracing case；与 R8b 存在多项配置差异 | Analyze only |

## End-to-end Performance

默认窗口：R8b 使用 post-warmup step 1-6。

| Run | Logical batch | Logged phase-sum mean / p95 | Rollout wait mean | Rollout share | Train active | Total allocated GPU-h | Consumed traj / total GPU-h | Loss-active token goodput |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R8b | 32 x 8 | 83.89 / 93.18 min | 73.21 min | 87.27% | 10.46 min | 268.45 | 5.722 | `UNKNOWN` |
| bs2 C1b v2 | 2 x 8 | 15.98 / 32.54 min | 14.99 min | 93.78% | 43.74 sec | 502.83 | 1.877 | `UNKNOWN` |

Overlap-aware update interval 是项目主性能指标。当前两条历史数据使用 `perf/time_per_step`，它是 trainer 记录的 `timeperf/*` phase sum，不是独立重建的连续 update interval；下一次 tracing run 必须用 update timestamps 和 makespan 复核。后台 rollout active time 与 trainer phase timer 不能直接相加。

bs2 的 step mean 比 R8b 低 80.95%，但 logical work 小 16 倍，cohort/rollout GPU-hour 反而低 69.5%。因此它只提供更短的诊断反馈周期，不能进入 canonical 性能排名。

## Rollout Supply And Training Participation

| Run | Generated pool | Trainer consumed | Consumed / generated | Aggregate unknown disposition | Full-sequence tokens, performance window | Loss/gradient-active | Evidence |
|---|---:|---:|---:|---:|---:|---|---|
| R8b | 2645 summaries | 1792 | 67.75% | 853 | `UNKNOWN` exact sum | `UNKNOWN` | Generated -> consumed join missing |
| bs2 C1b v2 | 2000 results | 960 | 48.00% | 1040 | 51015838 | `UNKNOWN` | Count exact；disposition join missing |

`Aggregate unknown disposition` 只是 pool count 减 consumed count，依赖二者来自同一供给池的假设。它不能直接叫作 terminal waste；必须继续拆成 queued-at-end、dropped/stale/cancelled 和 terminal-unconsumed。

## Trajectory Distribution

窗口：R8b 的 2645 份 trajectory summary，可能包含尚未被 7 个训练 step 消费的异步结果。

| Run | Trajectories | Elapsed p50 / p95 / p99 | Context proxy p50 / p95 / p99 | Reward <= 0 share | Reward <= 0 wall time | Max iterations | Loop detector |
|---|---:|---|---|---:|---:|---:|---:|
| R8b | 2645 | 24.4 / 60.0 / 97.0 min | Max request: 252 / 484 / 632 KB | 50.70% | 59.86% | 30.25% | 15.20% |
| bs2 C1b v2 | 1933 joined | 15.3 / 36.3 / 54.2 min | Max prompt: 58.2K / 98.2K / 126.9K tokens | 32.54% | 41.73% | 18.83% | 4.97% |

bs2 trajectory 表覆盖 online generation pool，不等于 trainer 消费的 960 条 trajectory。1933 条中有 375 条到达 80-turn cap，只有 10 条 reward > 0；详细 cache 和 termination 分布见 [case study](cases/bs2-eqtraj-C1b-v2.md)。

## Cohort Tail

窗口：267 个可以完整还原的 8-way group。

| Run | Group wall p50 / p95 | Wait after 7th p50 / p95 | Max / median duration p50 / p95 | Straggler reward <= 0 | Primary straggler origin |
|---|---|---|---|---:|---|
| R8b | 47.0 / 102.4 min | 7.8 / 51.5 min | 1.77x / 3.63x | 74.91% | max_iterations |
| bs2 C1b v2 | N/A | N/A | N/A | N/A | 缺 generated -> consumed cohort lineage |

`Wait after 7th` 的 ctx-timing 口径为 p50 7.7 min、p95 49.8 min；trajectory summary 重建口径为 p50 7.8 min、p95 51.5 min。两者相互验证，但不能混成同一个精确值。

## Inference Engine

窗口：16 个 vLLM server 的 74112 条周期日志；74042 条记录处于 active 状态。

| Run | Generation tok/s p50 / p95 | APC mean | Final APC across engines | KV usage p50 / p95 / max | Running requests p50 | Waiting > 0 |
|---|---|---:|---|---|---:|---:|
| R8b | 177.1 / 277.4 | 58.33% | 30.8%-77.3% | 16.0% / 25.9% / 44.2% | 6 | 5.99% |

R8b 没有显示持续的 vLLM queue 或 KV saturation。APC 是解释 per-turn prefill 成本的中间指标，不是最终优化目标。

bs2 没有汇总同口径的 vLLM 周期日志，不能在本表与 R8b APC 直接比较。request-level `cached_tokens / prompt_tokens` 为 73.56%；turn 10-19 为 95.44%，turn 20+ 降至 69.45%。

## Participation And Correctness Guardrails

| Run | Reward avg range by step | Seq length avg range | No-EOS range | Infra abnormal release | Policy/logp lineage | Distribution verdict |
|---|---|---|---|---:|---|---|
| R8b | -0.040 to 0.315 | 44.4K-53.1K | 0.78%-3.13% | 3.16% | N/A: historical tracing gap | Baseline distribution |
| bs2 C1b v2 | -0.656 to 0.994 | 34.4K-73.1K | Invalid under dynamic padding | N/A | Episode/result joined；trainer consumption 未 join | Diagnostic distribution only |

bs2 每 step 固定消费 16 条，其中 60 step 合计 `reward > 0` 625 条、`reward <= 0` 335 条；两类都属于 trainer-consumed 数据。当前 no-EOS 实现比较 `seq_len == padded_batch_width`，在 `pad_to_maximum=false` 时会把 batch 内最长序列误当作 no-EOS，不能用于 acceptance。

## Experiment Decision Board

| Candidate | Baseline | Single primary change | E2E delta | Participation delta | Downstream eval | Decision |
|---|---|---|---:|---:|---|---|
| Pending | R8b | Observation-only tracing reproduction | N/A | N/A | Not applicable: no training change | Planned |
| bs2 C1b v2 | R8b | No：logical batch 与多项配置不同 | -80.95% logged step mean | Consumed cohort/rollout GPU-h -69.5%；loss-active unknown | N/A | Diagnostic only；不可接受为性能收益 |

## Update Protocol

每次实验完成后：

1. 从 [template](templates/experiment_scorecard.md) 新建 scorecard。
2. 明确 baseline、单一主要变量和统计窗口。
3. 把本页各分表新增一行；不能可靠计算的指标填 `N/A: 原因`。
4. 先判断 correctness guardrail，再判断性能收益。
5. 只有 E2E step time 下降且 guardrail 通过的结果才能标记为 `Accepted`。

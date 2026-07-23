# Experiment Scorecard: `<experiment_id>`

Template version: `v2`

复制本模板创建每个 baseline、diagnostic case 和受控实验记录。标题、章节顺序和表头保持不变；没有可靠数据时填写 `UNKNOWN: 原因`，不能删行或用 `0` 代替缺失值。

可信度标签：

- `EXACT`：原始日志直接记录，或可由同一窗口的完整原始数据确定。
- `DERIVED`：由 `EXACT` 数据按明确公式计算。
- `INFERRED`：依赖尚未被 lineage 证明的假设。
- `UNKNOWN`：现有 tracing 无法回答。

## 0. Owner Review

这是 owner 默认需要阅读的全部内容，控制在一屏内。

| Item | Result | Confidence |
|---|---|---|
| One-line verdict | | |
| Workload | `<cohorts> x <group_size> trajectories`, `<max_seq_len>` | |
| Post-warmup E2E step mean / p95 | | |
| Trainer-consumed trajectories | | |
| Loss-active trajectories / tokens | | |
| Confirmed waste | | |
| Largest exposed wait | | |
| Training-time reward reference | | |
| Downstream checkpoint evaluation | Pass / Fail / Pending / Not applicable | |
| Decision and next action | | |

## 1. Decision Question

用一句可证伪的问题描述实验：

> 在固定 logical batch、最大序列长度、model、task set、seed 和 32 张 A100-80GB 总预算下，`<single change>` 是否降低 overlap-aware post-warmup E2E step time，同时保持 trainer participation、算法正确性和下游 checkpoint 评测不回归？

## 2. Identity And Reproduction

| Field | Value | Confidence |
|---|---|---|
| Experiment ID | | |
| Role / status | Canonical / Candidate / Diagnostic；Planned / Running / Analyzed / Accepted / Rejected | |
| Baseline | | |
| Date | | |
| Owner | | |
| Fuyao job | | |
| Trial | | |
| Code SHA and dirty diff | | |
| Image digest | | |
| Model path and digest | | |
| Dataset/task manifest and digest | | |
| Seed | | |
| Full config artifact | | |
| Log root | | |
| SwanLab run | | |
| Structured metrics | | |

## 3. Workload And Resource Contract

| Item | Baseline | This run | Comparable? |
|---|---|---|---|
| Model/checkpoint | | | |
| Cohorts per trainer step | | | |
| Trajectories per cohort | | | |
| Trajectories per trainer step | | | |
| Maximum sequence length | | | |
| Total GPU count/type | 32 x A100-80GB | | |
| Rollout / train GPUs | | | |
| Parallel topology | | | |
| Task IDs/order and seed | | | |
| Sampling params | | | |
| Session/WIP/concurrency | | | |
| Agent iteration/stop policy | | | |
| Primary variable | | | |

同时改变多个主要变量时，实验只能标记为 `Diagnostic` 或 `Exploratory`，不能声称单一因果结论。

## 4. Measurement Windows

不同窗口必须分开；generation pool 不能冒充 trainer-consumed batch。

| Window | Start/end | Count | Warmup policy | Join completeness | Confidence |
|---|---|---:|---|---:|---|
| Trainer steps, whole run | | | | | |
| Trainer steps, performance | | | | | |
| Trainer-consumed cohorts | | | | | |
| Trainer-consumed trajectories | | | | | |
| Generated result episodes | | | | | |
| Joined trajectory summaries | | | | | |
| Complete cohorts | | | | | |
| LLM turns | | | | | |
| vLLM periodic records | | | | | |
| Sandbox jobs | | | | | |

## 5. Step Critical Path

### 5.1 Timing Identity

先写 trainer 可见的 critical-path 恒等式。异步后台 generation 的 activity time 另记，不能与这些 phase timer 重复相加。

```text
step_time
= rollout_wait_or_prepare_batch
 + compute_advantage
 + train_step
 + update_weights
 + save
 + checkpoint_or_recovery
 + eval_and_cleanup
 + unattributed
```

| Component, seconds | Sum | Mean | p50 | p90 | p95 | Max | Step share | Confidence |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| E2E step wall | | | | | | | 100% | |
| Rollout wait / batch preparation | | | | | | | | |
| Compute advantage | | | | | | | | |
| Train forward/backward/optimizer | | | | | | | | |
| Weight update/sync | | | | | | | | |
| Save | | | | | | | | |
| Checkpoint/recovery | | | | | | | | |
| Eval/cleanup | | | | | | | | |
| Unattributed residual | | | | | | | | |

校验：component sum 与 E2E 差值必须为 `0` 或写明误差来源。

### 5.2 Representative Steps

| Step | Role | Step wall | Rollout wait | Train | Other | Batch trajectories | Full-sequence tokens | Notes |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| | Median-like | | | | | | | |
| | Near-p95 | | | | | | | |
| | Maximum | | | | | | | |

### 5.3 Async Activity And Overlap

| Metric | Value | Confidence |
|---|---:|---|
| Update-completion interval mean / p50 / p95 | | |
| Post-warmup makespan / interval count | | |
| Rollout active time | | |
| Train active time | | |
| Rollout/train overlap time and ratio | | |
| Rollout-only exposed time | | |
| Train-only exposed time | | |
| Coordination/idle time | | |
| Total allocated GPU-hours | | |

`timeperf/rollout` 在 online mode 通常是 trainer 等待 `prepare_batch` 返回的时间，不是 vLLM 纯 generation time。

## 6. Rollout Supply-Demand

### 6.1 Per-step Demand

```text
required_cohorts_per_step = <N>
group_size = <G>
required_trajectories_per_step = N x G
```

| Metric | Mean | p50 | p95 | Max | Confidence |
|---|---:|---:|---:|---:|---|
| Time to first eligible cohort | | | | | |
| Time to final required cohort | | | | | |
| Wait after penultimate required cohort | | | | | |
| Ready queue depth at dequeue | | | | | |
| In-flight trajectories at dequeue | | | | | |

### 6.2 Whole-run Data Funnel

| Stage | Trajectories | Share of generated | Meaning | Confidence |
|---|---:|---:|---|---|
| Attempted | | | Agent execution started | |
| Generated result | | 100% | Result artifact exists | |
| Manager rewarded | | | Reward/session lifecycle completed inside rollout manager | |
| Manager exported | | | Complete cohort handed to workflow | |
| Structurally eligible | | | Tensor/group/lineage checks pass | |
| Trainer consumed | | | Entered `prepare_batch -> compute_advantages -> ppo_update` | |
| Workflow/algorithm filtered | | | Exported but rejected before trainer, for example uniform reward | |
| Manager rejected after reward | | | Stale/partial/incomplete/infra rejection after some members finished | |
| Queued at run end | | | Still reusable after the measurement window | |
| Dropped / stale / cancelled | | | Explicit scheduler disposition | |
| Terminal unconsumed | | | Generated but never consumed before run termination | |
| Unknown disposition | | | Cannot join generated result to trainer | |

只有存在 trajectory-level disposition join 时，下面的等式才允许标记为 `EXACT`：

```text
generated_result
= trainer_consumed
 + workflow_or_algorithm_filtered
 + manager_rejected_after_reward
 + queued_at_end
 + dropped_or_stale_or_cancelled
 + terminal_unconsumed
```

同时输出 cohort-level reject ledger，至少包含：`reason`、cohort 数、expected/admitted/ended/rewarded slot 数、policy drift、cohort age，以及是否整组已完成。Result artifact、manager rewarded、manager exported 和 trainer consumed 是四个不同层级，不能互相替代。

## 7. Training Participation

实际被 trainer 消费的数据都计为有价值训练数据；reward 正负只描述组成，不判断是否浪费。

| Level | Whole run | Performance window | Definition | Confidence |
|---|---:|---:|---|---|
| Trainer-consumed cohorts | | | Cohort entered advantage computation | |
| Trainer-consumed trajectories | | | Trajectory entered trainer batch | |
| Full-sequence tokens processed | | | Sum of consumed sequence lengths | |
| Loss-active trajectories | | | At least one token has active `loss_mask` | |
| Loss-active tokens | | | Tokens with active `loss_mask` | |
| Gradient-active trajectories | | | At least one token has non-zero effective loss weight/advantage | |
| Gradient-active tokens | | | Tokens with non-zero effective contribution | |
| Fully masked / zero-weight trajectories | | | Consumed but contributes zero direct gradient | |

### 7.1 Consumed-data Composition

| Composition | Whole run | Performance window | Interpretation | Confidence |
|---|---:|---:|---|---|
| Reward > 0 | | | Positive outcome composition | |
| Reward <= 0 | | | Negative training examples, not waste by default | |
| Normal completion | | | | |
| Max iterations | | | | |
| Loop detector | | | | |
| Timeout / infra error | | | | |

## 8. Waste And Waiting Ledger

“确认浪费”要求有明确 disposition；昂贵的负样本或长轨迹只能列为成本组成或优化候选。

| Category | Trajectories / events | Wall time | GPU-hours | Share | Status / evidence |
|---|---:|---:|---:|---:|---|
| Terminal unconsumed generation | | | | | |
| Explicit dropped / stale / cancelled | | | | | |
| Retry / duplicate | | | | | |
| Uniform-reward / zero-signal group filter | | | | | |
| Partial/incomplete cohort amplification | | | | | |
| Infra-invalid generation | | | | | |
| Trainer blocked waiting for rollout | | | | | |
| Cohort straggler tail | | | | | |
| Save/checkpoint exposed time | | | | | |
| Unknown disposition | | | | | |

## 9. Trajectory And Cohort Distribution

### 9.1 Trajectory Distribution

| Metric | Count / mean | p50 | p90 | p95 | p99 | Max | Confidence |
|---|---:|---:|---:|---:|---:|---:|---|
| Episode elapsed | | | | | | | |
| Turn count | | | | | | | |
| Max prompt/context tokens | | | | | | | |
| Cumulative LLM RPC | | | | | | | |

| Cost composition | Count | Trajectory share | Episode-wall share | LLM-RPC share | Confidence |
|---|---:|---:|---:|---:|---|
| Reward <= 0 | | | | | |
| Max iterations | | | | | |
| Loop detector | | | | | |
| Context >= configured threshold | | | | | |
| Turn cap reached | | | | | |

### 9.2 Cohort Tail

| Metric | Mean | p50 | p90 | p95 | Max | Confidence |
|---|---:|---:|---:|---:|---:|---|
| Group wall | | | | | | |
| Wait after penultimate trajectory | | | | | | |
| Max / median duration | | | | | | |
| Straggler reward <= 0 share | | | | | | |

## 10. Turn And Engine Metrics

| Metric | Mean | p50 | p90 | p95 | p99 | Max | Confidence |
|---|---:|---:|---:|---:|---:|---:|---|
| LLM RPC latency | | | | | | | |
| TTFT | | | | | | | |
| Decode tok/s | | | | | | | |
| Per-request cache ratio | | | | | | | |
| Queue wait | | | | | | | |
| KV usage | | | | | | | |
| Tool latency | | | | | | | |
| Sandbox latency | | | | | | | |

额外报告 prompt、completion、cached token 总量，以及按 turn range 的 cache shape。

## 11. Correctness And Training Effect

| Guardrail | Baseline | This run | Allowed range | Result |
|---|---:|---:|---:|---|
| Trainer-consumed trajectory count | | | Exact workload | |
| Loss-active token/trajectory count | | | No unexplained regression | |
| Reward distribution | | | Pre-registered | |
| Context-length distribution | | | Pre-registered | |
| Turn/failure distribution | | | Pre-registered | |
| Policy-version lineage | | | 100% valid | |
| Group completeness | | | 100% or algorithm-defined | |
| Logp / importance-ratio diagnostics | | | Pre-registered | |
| True EOS / truncation reason | | | 100% classified | |
| Infra failure rate | | | No regression | |

| Layer | Metric | Role | Result |
|---|---|---|---|
| During training | Reward and outcome distribution | Online reference | |
| Downstream | Pre-registered benchmark on trained checkpoint | Final effect gate | |
| Efficiency | Time/GPU-hours to target downstream quality | Final efficiency result | |

## 12. Causal Analysis

```text
Change
  -> turn/request effect
  -> trajectory cost and outcome effect
  -> cohort supply/tail effect
  -> trainer wait and E2E effect
  -> participation/correctness/downstream effect
```

列出支持证据、反证、替代解释和 tracing 缺口。

## 13. Decision

| Item | Result |
|---|---|
| Performance verdict | |
| Participation verdict | |
| Correctness verdict | |
| Final decision | Accept / Reject / Diagnostic only / Inconclusive |
| Confidence | High / Medium / Low |
| Rollback trigger | |
| Next experiment | |

只有 overlap-aware E2E step time 下降、总资源和 logical workload 可比、trainer participation 没有暗降，并且 correctness 与下游评测 guardrail 通过时，才允许最终 `Accept`。

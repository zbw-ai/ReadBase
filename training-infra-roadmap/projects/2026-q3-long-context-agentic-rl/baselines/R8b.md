# Experiment Scorecard: R8b

Template version: `v2`

## 0. Owner Review

| Item | Result | Confidence |
|---|---|---|
| One-line verdict | 当前 `32x8` canonical historical baseline；post-warmup logged step mean 83.89 min，其中 87.27% 是等待 full rollout batch | `DERIVED` from logged rows |
| Workload | `32 cohorts x 8 trajectories = 256 trajectories/step`，`max_seq_len=131072` | `EXACT` |
| Post-warmup E2E step mean / p95 | 83.89 / 93.18 min，口径为 logged trainer phase sum | `DERIVED` |
| Trainer-consumed trajectories | 全 7 step 为 1792；性能窗口 step 1-6 为 1536 | `DERIVED` |
| Loss-active trajectories / tokens | `UNKNOWN`：历史日志未保存最终 mask/weight count | `UNKNOWN` |
| Confirmed waste | 89 次 explicit abnormal sandbox release；无法 join 到 trainer consumption | Count `EXACT`，training impact `UNKNOWN` |
| Largest exposed wait | post-warmup trainer rollout wait 共 26355.6 sec，对应 117.136 train-partition GPU-h | `DERIVED` |
| Training-time reward reference | 性能窗口 1536 条 consumed 中 753 positive、783 non-positive；两类都参与训练 | `DERIVED` |
| Downstream checkpoint evaluation | `UNKNOWN`：未归档预注册评测 | `UNKNOWN` |
| Decision and next action | 保留为历史 baseline；下一次 `32x8` observation-only reproduction 补齐 identity、participation 和 overlap | `EXACT` |

## 1. Decision Question

> 在 32 张 A100-80GB、Qwen3.5-9B、128K、`32x8` logical workload 下，当前系统完成一个 trainer step 的历史成本和首要瓶颈是什么？

R8b 是历史基线，不是受控优化实验。它用于提供正式 workload 的数量级和瓶颈证据。

## 2. Identity And Reproduction

| Field | Value | Confidence |
|---|---|---|
| Experiment ID | `R8b` | `EXACT` |
| Role / status | Canonical / Historical baseline | `EXACT` |
| Baseline | Self | `EXACT` |
| Date | 2026-07-06 | `EXACT` |
| Owner | zengbw1 | `EXACT` |
| Fuyao job | `bifrost-2026070622090200-zengbw1` | `EXACT` |
| Trial | `trial_oh_bs32_NATIVE_qwen3coder_128k_CP8_d16t1_async_ses128_wip64_ofp2_PROD500_R8` | `EXACT` |
| Code SHA and dirty diff | Historical log 未确认 | `UNKNOWN` |
| Image digest | Historical log 未确认 immutable digest | `UNKNOWN` |
| Model path and digest | Qwen3.5-9B；exact path/digest 待补 | `UNKNOWN` |
| Dataset/task manifest and digest | R2E-Gym；exact task manifest/digest 待补 | `UNKNOWN` |
| Seed | `UNKNOWN` | `UNKNOWN` |
| Full config artifact | Historical log/config，未归档 immutable artifact | Existing fields `EXACT`；immutability `UNKNOWN` |
| Log root | `/dataset_rc_b1/zengbw1/log/areal_r2e_gym_qwen35_9b/128k_bifrost-2026070622090200-zengbw1/experiments_zbw_128k_nativeqwen3coder_d1t2c4p1_bs32_g8_mb131072/20260706_221625` | `EXACT` |
| SwanLab run | [R8b run](https://swanlab.cn/@zengbw1/areal-experiments/runs/1b9mqas2ope3rando41kr) | `EXACT` |
| Structured metrics | [R8b metrics](../metrics/R8b.json) | `EXACT` |

训练在 2026-07-06 22:16:25 拉起。Step 0-6 是 R8 系列第一次、也是唯一连续完成 7 step 且未被打断的窗口。

## 3. Workload And Resource Contract

| Item | Baseline | This run | Comparable? |
|---|---|---|---|
| Model/checkpoint | Qwen3.5-9B | Qwen3.5-9B；digest unknown | Self |
| Cohorts per trainer step | 32 | 32 | Yes |
| Trajectories per cohort | 8 | 8 | Yes |
| Trajectories per trainer step | 256 | 256 | Yes |
| Maximum sequence length | 131072 | 131072 | Yes |
| Total GPU count/type | 32 x A100-80GB | 32 x A100-80GB | Yes |
| Rollout / train GPUs | 16 / 16 | 16 / 16 | Yes |
| Parallel topology | rollout 16 engines TP1；train CP8 | rollout vLLM `d16t1` TP1；train Megatron CP8 | Yes |
| Task IDs/order and seed | Historical baseline | Manifest missing | Unknown |
| Sampling params | Historical baseline | Config artifact not immutable | Unknown |
| Session/WIP/concurrency | session 128 / WIP 64 | session 128 / WIP 64 | Yes |
| Agent iteration/stop policy | Historical baseline | Historical config | Self |
| Primary variable | Baseline | None | N/A |

## 4. Measurement Windows

| Window | Start/end | Count | Warmup policy | Join completeness | Confidence |
|---|---|---:|---|---:|---|
| Trainer steps, whole run | step 0-6 | 7 | Includes warmup | 7/7 | `EXACT` |
| Trainer steps, performance | step 1-6 | 6 | Excludes step 0 | 6/6 | `EXACT` |
| Trainer-consumed cohorts | step 0-6 / step 1-6 | 224 / 192 | Same as step window | Count only | `DERIVED` |
| Trainer-consumed trajectories | step 0-6 / step 1-6 | 1792 / 1536 | Same as step window | Count only | `DERIVED` |
| Generated result episodes | Whole tracing pool | 2645 summaries | Not a step window | Consumption join missing | `EXACT` count |
| Joined trajectory summaries | Whole tracing pool | 2645 | Not a step window | Summary-level only | `EXACT` |
| Complete cohorts | Whole tracing pool | 267 | Not a step window | 8-way groups reconstructed | `EXACT` |
| LLM turns | Whole tracing pool | `UNKNOWN` | N/A | Per-turn tokens not stable | `UNKNOWN` |
| vLLM periodic records | 16 engines | 74112 | N/A | 74042 active records | `EXACT` |
| Sandbox jobs | Whole run | 2819 acquired | N/A | Outcome counts available | `EXACT` |

## 5. Step Critical Path

### 5.1 Timing Identity

历史表保留了 step、rollout 和 train timer。其余 phase 只能用残差合并，不能事后伪造 compute-advantage/save/checkpoint 的分位数。

窗口：step 1-6，单位均为秒；分位数由 6 行历史 logged value 计算，原始行保留 0.1-0.01 sec 精度。

| Component, seconds | Sum | Mean | p50 | p90 | p95 | Max | Step share | Confidence |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| E2E logged phase sum | 30200.10 | 5033.35 | 5070.70 | 5486.80 | 5590.55 | 5694.30 | 100% | `DERIVED` from logged rows |
| Rollout wait / batch preparation | 26355.60 | 4392.60 | 4405.35 | 4847.60 | 4943.10 | 5038.60 | 87.2699% | `DERIVED` |
| Compute advantage | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | Included in residual | `UNKNOWN` |
| Train forward/backward/optimizer | 3766.96 | 627.827 | 636.04 | 655.155 | 661.563 | 667.97 | 12.4733% | `DERIVED` |
| Weight update/sync | `UNKNOWN` distribution；observed range about 1.9-2.4 sec | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | Included in residual | Range `EXACT`；distribution `UNKNOWN` |
| Save | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | Included in residual | `UNKNOWN` |
| Checkpoint/recovery | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | Included in residual | `UNKNOWN` |
| Eval/cleanup | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | Included in residual | `UNKNOWN` |
| Other recorded phases residual | 77.54 | 12.923 | 12.705 | 13.575 | 13.683 | 13.79 | 0.2568% | `DERIVED` from rounded rows |

原始 step 行：

| Step | Rollout sec | Train sec | Step sec | Seq avg | Positive / non-positive |
|---:|---:|---:|---:|---:|---:|
| 0 | 8838.5 | 944.42 | 9804.3 | 50.0K | 114 / 142 |
| 1 | 4258.2 | 634.45 | 4905.2 | 50.2K | 92 / 164 |
| 2 | 3591.5 | 575.66 | 4179.9 | 44.4K | 115 / 141 |
| 3 | 5038.6 | 642.34 | 5694.3 | 47.1K | 126 / 130 |
| 4 | 4656.6 | 608.91 | 5279.3 | 44.4K | 127 / 129 |
| 5 | 4442.5 | 637.63 | 5092.8 | 52.0K | 135 / 121 |
| 6 | 4368.2 | 667.97 | 5048.6 | 53.1K | 158 / 98 |

这里的 Positive/Non-positive 来自 `reward > 0`，不是结构合法性判断。

### 5.2 Representative Steps

| Step | Role | Step wall | Rollout wait | Train | Other | Batch trajectories | Full-sequence tokens | Notes |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 5 | Median-like | 5092.8 sec | 4442.5 sec | 637.63 sec | 12.67 sec | 256 | `UNKNOWN` exact total | 135 positive / 121 non-positive |
| 4 | Closest logged step below p95 | 5279.3 sec | 4656.6 sec | 608.91 sec | 13.79 sec | 256 | `UNKNOWN` exact total | 127 positive / 129 non-positive |
| 3 | Maximum / only step above p95 | 5694.3 sec | 5038.6 sec | 642.34 sec | 13.36 sec | 256 | `UNKNOWN` exact total | 126 positive / 130 non-positive |

### 5.3 Async Activity And Overlap

| Metric | Value | Confidence |
|---|---:|---|
| Update-completion interval mean / p50 / p95 | `UNKNOWN` | `UNKNOWN` |
| Logged phase sum / step count | 30200.10 sec / 6 = 5033.35 sec | `DERIVED` |
| Rollout active time | `UNKNOWN` | `UNKNOWN` |
| Train active time | 3766.96 sec timer sum；16.742 train-GPU-h | `DERIVED` |
| Rollout/train overlap time and ratio | `UNKNOWN` | `UNKNOWN` |
| Rollout-only exposed time | 26355.60 sec trainer-visible wait proxy | `DERIVED` |
| Train-only exposed time | `UNKNOWN` | `UNKNOWN` |
| Coordination/idle time | `UNKNOWN` | `UNKNOWN` |
| Total allocated GPU-hours | 268.445 GPU-h = 30200.10 sec x 32 / 3600 | `DERIVED` |

## 6. Rollout Supply-Demand

### 6.1 Per-step Demand

```text
required_cohorts_per_step = 32
group_size = 8
required_trajectories_per_step = 32 x 8 = 256
```

| Metric | Mean | p50 | p95 | Max | Confidence |
|---|---:|---:|---:|---:|---|
| `prepare_batch` call to full batch return | 4392.60 sec | 4405.35 sec | 4943.10 sec | 5038.60 sec | `DERIVED` |
| Time to first eligible cohort | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | No cohort-ready event |
| Time to final required cohort from generation start | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | Existing queue state unknown |
| Wait after penultimate required cohort | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | Trainer-required cohort order unavailable |
| Ready queue depth at dequeue | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | No queue snapshot |
| In-flight trajectories at dequeue | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | No queue snapshot |

### 6.2 Whole-run Data Funnel

| Stage | Trajectories | Share of generated | Meaning | Confidence |
|---|---:|---:|---|---|
| Attempted | `UNKNOWN` | `UNKNOWN` | Agent execution started | `UNKNOWN` |
| Generated trajectory summary | 2645 | 100% | Summary artifact exists | `EXACT` |
| Structurally eligible | `UNKNOWN` | `UNKNOWN` | Tensor/group/lineage checks pass | `UNKNOWN` |
| Trainer consumed | 1792 | 67.75% | 7 steps x 256 trajectories | Count `DERIVED`，identity join missing |
| Queued at run end | `UNKNOWN` | `UNKNOWN` | Potential future consumption | `UNKNOWN` |
| Dropped / stale / cancelled | `UNKNOWN` | `UNKNOWN` | Scheduler disposition | `UNKNOWN` |
| Terminal unconsumed | `UNKNOWN` | `UNKNOWN` | Generated and never reused | `UNKNOWN` |
| Aggregate unknown disposition | 853 | 32.25% | `2645 - 1792`，假设 consumed 来自同一 summary pool | `INFERRED` |

## 7. Training Participation

| Level | Whole run | Performance window | Definition | Confidence |
|---|---:|---:|---|---|
| Trainer-consumed cohorts | 224 | 192 | Cohort entered advantage computation | `DERIVED` |
| Trainer-consumed trajectories | 1792 | 1536 | Trajectory entered trainer batch | `DERIVED` |
| Full-sequence tokens processed | `UNKNOWN` exact sum | `UNKNOWN` exact sum | Historical table only has rounded per-step average length | `UNKNOWN` |
| Loss-active trajectories | `UNKNOWN` | `UNKNOWN` | At least one final `loss_mask` token | `UNKNOWN` |
| Loss-active tokens | `UNKNOWN` | `UNKNOWN` | Final `loss_mask` count | `UNKNOWN` |
| Gradient-active trajectories | `UNKNOWN` | `UNKNOWN` | At least one non-zero effective contribution | `UNKNOWN` |
| Gradient-active tokens | `UNKNOWN` | `UNKNOWN` | Non-zero effective loss weight/advantage | `UNKNOWN` |
| Fully masked / zero-weight trajectories | `UNKNOWN` | `UNKNOWN` | Consumed but zero direct gradient | `UNKNOWN` |

### 7.1 Consumed-data Composition

| Composition | Whole run | Performance window | Interpretation | Confidence |
|---|---:|---:|---|---|
| Reward > 0 | 867 / 1792 = 48.38% | 753 / 1536 = 49.02% | Positive outcome composition | `DERIVED` |
| Reward <= 0 | 925 / 1792 = 51.62% | 783 / 1536 = 50.98% | 实际参与训练的 negative examples，不是浪费 | `DERIVED` |
| Mean reward | Historical step range -0.040 to 0.315 | Same range | Online reference | `EXACT` range |
| Normal completion | `UNKNOWN` | `UNKNOWN` | Generated pool 有分布，consumed join missing | `UNKNOWN` |
| Max iterations | `UNKNOWN` | `UNKNOWN` | Same | `UNKNOWN` |
| Loop detector | `UNKNOWN` | `UNKNOWN` | Same | `UNKNOWN` |
| Timeout / infra error | `UNKNOWN` | `UNKNOWN` | Same | `UNKNOWN` |

## 8. Waste And Waiting Ledger

| Category | Trajectories / events | Wall time | GPU-hours | Share | Status / evidence |
|---|---:|---:|---:|---:|---|
| Terminal unconsumed generation | `UNKNOWN`，aggregate surplus 候选 853 | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | End-of-run disposition missing |
| Explicit dropped / stale / cancelled | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | Scheduler lineage missing |
| Retry / duplicate | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | Retry lineage missing |
| Explicit abnormal sandbox release | 89 / 2819 | `UNKNOWN` | `UNKNOWN` | 3.16% acquired sandboxes | 74 provision/setup + 15 terminal errors，`EXACT` count |
| Trainer blocked waiting for rollout | 6 step wait events | 26355.60 sec = 7.321 h | 117.136 train-GPU-h | 87.27% step wall；43.63% total allocated GPU-h | `DERIVED` |
| Cohort straggler tail | 267 reconstructed groups | p50 7.8 min；p95 51.5 min | `UNKNOWN` | `UNKNOWN` | Generated-pool cohort tail；consumption join missing |
| Other phase residual | 6 steps | 77.54 sec | 0.345 train-GPU-h | 0.257% step wall | `DERIVED` from rounded rows |
| Unknown generation disposition | 853 aggregate | `UNKNOWN` | `UNKNOWN` | 32.25% generated summaries | `INFERRED` |

## 9. Trajectory And Cohort Distribution

### 9.1 Trajectory Distribution

窗口：2645 个 generated trajectory summaries，不等于 1792 个 trainer-consumed trajectories。

| Metric | Count / mean | p50 | p90 | p95 | p99 | Max | Confidence |
|---|---:|---:|---:|---:|---:|---:|---|
| Episode elapsed, min | 2645 / 28.1 | 24.4 | 48.7 | 60.0 | 97.0 | 180.1 timeout observed | `EXACT` |
| Turn count | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | Historical tracing gap |
| Max request size, KB | `UNKNOWN` | 252 | `UNKNOWN` | 484 | 632 | `UNKNOWN` | `EXACT` available quantiles |
| Cumulative request size | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | Correlation only |

| Cost composition | Count | Trajectory share | Episode-wall share | LLM-RPC share | Confidence |
|---|---:|---:|---:|---:|---|
| Reward <= 0 | 1341 | 50.70% | 59.86% | `UNKNOWN` | `EXACT` generated pool |
| Max iterations | 800 | 30.25% | `UNKNOWN`；mean elapsed 38.0 min | `UNKNOWN` | `EXACT` count |
| Loop detector | 402 | 15.20% | `UNKNOWN`；mean elapsed 23.7 min | `UNKNOWN` | `EXACT` count |
| Zero patch | 255 approx | 9.64% | 10.86% | `UNKNOWN` | `DERIVED` share/count rounded |
| Run timeout | 1 | 0.04% | `UNKNOWN` | `UNKNOWN` | `EXACT` |

Trajectory elapsed 与 max/cumulative request size 的 Pearson `r=0.845/0.805`，与 completion count 的 `r=0.425`。主要问题是 context growth 与 per-turn cost 的乘积，不只是 turn 数。

### 9.2 Cohort Tail

| Metric | Mean | p50 | p90 | p95 | Max | Confidence |
|---|---:|---:|---:|---:|---:|---|
| Group wall, min | `UNKNOWN` | 47.0 | `UNKNOWN` | 102.4 | 180.1 | `EXACT` available quantiles |
| Wait after 7th trajectory, min | `UNKNOWN` | 7.8 | `UNKNOWN` | 51.5 | `UNKNOWN` | `EXACT` available quantiles |
| Max / median duration | `UNKNOWN` | 1.77x | `UNKNOWN` | 3.63x | `UNKNOWN` | `EXACT` available quantiles |
| Straggler reward <= 0 share | 74.91% | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `EXACT` aggregate |

Straggler origin 为 154 `max_iterations`、32 loop、80 none、1 timeout。独立 ctx-timing 重建得到 wait-after-7th p50 7.7 min、p95 49.8 min，与 trajectory-summary 口径相互验证。

## 10. Turn And Engine Metrics

| Metric | Mean | p50 | p90 | p95 | p99 | Max | Confidence |
|---|---:|---:|---:|---:|---:|---:|---|
| LLM RPC latency | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | Historical request-level gap |
| TTFT | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | Not traced |
| Decode tok/s | 169.4 generation tok/s | 177.1 | `UNKNOWN` | 277.4 | `UNKNOWN` | `UNKNOWN` | Engine-periodic, not request-level |
| APC/cache ratio | 58.33% | 60.45% | `UNKNOWN` | 83.0% | `UNKNOWN` | final engines 30.8%-77.3% | Engine-periodic |
| Queue wait | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | Waiting-request fraction only |
| KV usage | 16.14% | 16.0% | `UNKNOWN` | 25.9% | `UNKNOWN` | 44.2% | `EXACT` periodic records |
| Tool latency | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | Not split |
| Sandbox latency | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | Outcome only |

16 个 vLLM engine 共 74112 条周期记录，74042 条处于 active；running requests p50 为 6，waiting requests positive fraction 为 5.99%。没有持续 queue 或 KV saturation 的证据，但缺少 request -> engine -> trajectory join。

## 11. Correctness And Training Effect

| Guardrail | Baseline | This run | Allowed range | Result |
|---|---:|---:|---:|---|
| Trainer-consumed trajectory count | 256/step | 256/step，7 step | Exact workload | Pass |
| Loss-active token/trajectory count | `UNKNOWN` | `UNKNOWN` | No unexplained regression | Pending |
| Reward distribution | Self | Step mean range -0.040 to 0.315 | Baseline distribution | Recorded |
| Context-length distribution | Self | Seq avg 44.4K-53.1K | Baseline distribution | Recorded |
| Turn/failure distribution | Self | Generated pool available，consumed join missing | Baseline distribution | Partial |
| Policy-version lineage | `UNKNOWN` | `UNKNOWN` | 100% valid | Pending |
| Group completeness | 8 | Count implies 8；consumption IDs missing | 100% | Partial |
| Logp / importance-ratio diagnostics | `UNKNOWN` | `UNKNOWN` | Pre-registered | Pending |
| True EOS / truncation reason | `UNKNOWN` | Historical proxy not trusted | 100% classified | Pending |
| Infra failure rate | Self | 3.16% explicit abnormal sandbox release | No regression | Baseline only |

| Layer | Metric | Role | Result |
|---|---|---|---|
| During training | Reward/outcome distribution | Online reference | Recorded |
| Downstream | Pre-registered benchmark | Final effect gate | `UNKNOWN` |
| Efficiency | Time/GPU-hours to target quality | Final efficiency result | `UNKNOWN` |

## 12. Causal Analysis

```text
long context and expensive late turns
  -> trajectory elapsed tail
  -> 8-way cohort waits for a straggler
  -> ready cohort supply falls below 32 cohorts/step demand
  -> trainer prepare_batch waits about 73.21 min/step
  -> logged E2E step reaches 83.89 min
```

支持证据：rollout wait 占 step 87.27%；cohort wait-after-7th p95 为 51.5 min；trajectory elapsed 与 max request size 的相关系数为 0.845；vLLM 没有持续 queue/KV saturation。

限制：generated cohort 不能 join 到 7 个 consumed batches；没有 per-turn token/cache/RPC；没有 aligned active interval，所以无法判断后台 rollout 与 train 的实际 overlap，也不能精确给 853 条 aggregate surplus 分配 disposition 或 GPU-hour。

## 13. Decision

| Item | Result |
|---|---|
| Performance verdict | 正式 `32x8` workload 的 historical step mean 83.89 min；rollout supply 是一阶瓶颈 |
| Participation verdict | 1792 条 trajectory 确认按 logical workload 进入 7 个 trainer batch；loss-active/gradient-active 未观测 |
| Correctness verdict | 历史 lineage、mask、true EOS 和 downstream eval 不完整 |
| Final decision | Keep as canonical historical baseline |
| Confidence | Step/cohort bottleneck High；exact waste and training effect Low |
| Rollback trigger | N/A |
| Next experiment | 同 `32x8` 配置做 observation-only reproduction，记录 update completion、disposition、loss-active token 和 policy lineage |

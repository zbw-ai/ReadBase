# Experiment Scorecard: bs2-eqtraj-C1b-v2

Template version: `v2`

## 0. Owner Review

| Item | Result | Confidence |
|---|---|---|
| One-line verdict | 这是有效的长轨迹诊断 run，但不是性能更优的 baseline：每 step 工作量缩小 16 倍后，logged step mean 为 15.98 min，仍有 93.78% 暴露为等待 full rollout batch | `EXACT` |
| Workload | `2 cohorts x 8 trajectories = 16 trajectories/step`，`max_seq_len=131072` | `EXACT` |
| Post-warmup E2E step mean / p95 | 15.98 / 32.54 min，口径为 logged trainer phase sum，不是独立重建的 update interval | `EXACT` |
| Trainer-consumed trajectories | 全 60 step 为 960；性能窗口 step 1-59 为 944 | `EXACT` |
| Loss-active trajectories / tokens | `UNKNOWN`：没有导出最终 `loss_mask` 和 effective loss weight count | `UNKNOWN` |
| Confirmed waste | 1040 条未消费已做 aggregate closure：376 stale、364 partial-cohort deadline、128 uniform-reward filter、43 incomplete cohort、14 sandbox exception、1 cohort timeout、5 open-at-shutdown rewarded、109 截止时未进入 manager rewarded；仍缺逐 result ID/token/GPU cost join | Aggregate `EXACT`；per-result cost `UNKNOWN` |
| Largest exposed wait | post-warmup trainer 等待 batch 共 53052.90 sec，对应 235.79 train-partition GPU-h | `DERIVED` |
| Training-time reward reference | 全 960 条 consumed 中 625 条 reward > 0、335 条 reward <= 0；二者都属于训练参与数据 | `EXACT` |
| Downstream checkpoint evaluation | Not applicable：diagnostic run，不作为最终训练效果结论 | `EXACT` |
| Decision and next action | 保留为 diagnostic case；优先修 cohort-aware admission 与 retry idempotency，再用 `max_head_offpolicyness=2/3` 做 correctness-gated A/B | `EXACT` |

## 1. Decision Question

> 这个 `2x8` 小 batch run 能否用更短反馈周期定位 Qwen3.5-9B-128K rollout 长尾、prefix cache 和 trainer 等待的关系，并暴露下一次正式 `32x8` 实验必须补齐的 tracing？

它没有固定 R8b 的 logical batch 和多项配置，因此不能回答“性能优化是否让正式训练更快”。

## 2. Identity And Reproduction

| Field | Value | Confidence |
|---|---|---|
| Experiment ID | `bs2-eqtraj-C1b-v2` | `EXACT` |
| Role / status | Diagnostic / Analyzed | `EXACT` |
| Baseline | [R8b](../baselines/R8b.md)，只作参照，不作受控 A/B | `EXACT` |
| Date | 2026-07-20 | `EXACT` |
| Owner | zengbw1 | `EXACT` |
| Fuyao job | `bifrost-2026070622090200-zengbw1` | `EXACT` |
| Trial | `trial_oh_bs2_eqtraj_C1b_v2` | `EXACT` |
| Code SHA and dirty diff | `60b3b548431f146273d96d1f6e38628a14e4d06d`；运行时 working tree dirty，diff 未固化 | SHA `EXACT`；diff `UNKNOWN` |
| Image digest | `UNKNOWN` | `UNKNOWN` |
| Model path and digest | Qwen3.5-9B；exact path/digest 未固化 | `UNKNOWN` |
| Dataset/task manifest and digest | R2E-Gym；exact task manifest/digest 未固化 | `UNKNOWN` |
| Seed | `UNKNOWN` | `UNKNOWN` |
| Full config artifact | run log/config；未归档 immutable artifact | Existing fields `EXACT`；immutability `UNKNOWN` |
| Log root | `/dataset_rc_b1/zengbw1/log/areal_r2e_gym_qwen35_9b/128k_bifrost-2026070622090200-zengbw1/experiments_zbw_128k_nativeqwen3coder_d1t2c4p1_bs32_g8_mb131072/20260720_141521` | `EXACT` |
| SwanLab run | [bs2 run](https://swanlab.cn/@zengbw1/areal-experiments/runs/5jprfoiiydg0e2e4t8ru4/chart) | `EXACT` |
| Structured metrics | [bs2 metrics](../metrics/bs2-eqtraj-C1b-v2.json) | `EXACT` |

## 3. Workload And Resource Contract

| Item | R8b baseline | This run | Comparable? |
|---|---|---|---|
| Model/checkpoint | Qwen3.5-9B；exact digest unknown | Qwen3.5-9B；exact digest unknown | Partial |
| Cohorts per trainer step | 32 | 2 | No |
| Trajectories per cohort | 8 | 8 | Yes |
| Trajectories per trainer step | 256 | 16 | No，少 16 倍 |
| Maximum sequence length | 131072 | 131072 | Yes |
| Total GPU count/type | 32 x A100-80GB | 32 x A100-80GB | Yes |
| Rollout / train GPUs | 16 / 16 | 16 / 16 | Yes |
| Parallel topology | rollout `d16t1`；train CP8 | rollout vLLM `d16t1` TP1；train Megatron `d1t2c8p1` | Partial |
| Task IDs/order and seed | Unknown | Unknown | Unknown |
| Sampling params | Historical config | 未固化 diff | Unknown |
| Session/WIP/concurrency | session 128 / WIP 64 | 80 concurrent rollouts，worker capacity 128，FIFO window 96 | No |
| Agent iteration/stop policy | Historical config | 8 trajectories/cohort，80-turn cap | Partial |
| Primary variable | Canonical workload | 小 logical batch + tracing diagnostic | No single-variable causality |

## 4. Measurement Windows

| Window | Start/end | Count | Warmup policy | Join completeness | Confidence |
|---|---|---:|---|---:|---|
| Trainer steps, whole run | step 0-59 | 60 | Includes warmup | 60/60 | `EXACT` |
| Trainer steps, performance | step 1-59 | 59 | Excludes step 0 | 59/59 | `EXACT` |
| Trainer-consumed cohorts | step 0-59 / step 1-59 | 120 / 118 | Same as step window | Count only，no cohort IDs | `DERIVED` |
| Trainer-consumed trajectories | step 0-59 / step 1-59 | 960 / 944 | Same as step window | Count only，no trajectory IDs | `EXACT` |
| Generated result episodes | Whole generation pool | 2000 | Not a step window | 2000 unique result IDs | `EXACT` |
| Joined trajectory summaries | Whole generation pool | 1933 | Not a step window | 1933/2000 = 96.65% | `EXACT` |
| Cohort terminal ledger | Whole generation pool | 248/250 terminal；136 exported、112 manager-rejected、2 open | N/A | seq 0-249，缺 terminal seq 247/249 | `EXACT` |
| LLM turns | 1933 joined episodes | 101068 | N/A | Joined episodes only | `EXACT` |
| vLLM periodic records | Whole run | `UNKNOWN` | N/A | Same-port records not summarized | `UNKNOWN` |
| Sandbox jobs | Whole run | `UNKNOWN` | N/A | File exists，not joined in current artifact | `UNKNOWN` |

## 5. Step Critical Path

### 5.1 Timing Identity

`perf/time_per_step` 在 exact run code 中是所有 `timeperf/*` 的和。它覆盖 trainer loop 中已记录的 phase，但不等于连续 optimizer/update completion timestamp；`log_stats`、其后的 resume/tracer 等未被独立纳入 update interval。

```text
958.794633 sec mean step
= 899.201643 rollout wait / prepare_batch
 + 0.220930 compute advantage
 + 43.735078 train step
 + 2.010344 update weights
 + 5.625437 save
 + 8.001202 checkpoint/recovery and other recorded residual
```

窗口：step 1-59，单位均为秒。

| Component, seconds | Sum | Mean | p50 | p90 | p95 | Max | Step share | Confidence |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| E2E logged phase sum | 56568.883 | 958.795 | 805.445 | 1705.225 | 1952.628 | 2427.906 | 100% | `EXACT` |
| Rollout wait / batch preparation | 53052.897 | 899.202 | 756.852 | 1642.886 | 1908.432 | 2375.363 | 93.7846% | `EXACT` |
| Compute advantage | 13.035 | 0.221 | 0.183 | 0.342 | 0.369 | 0.830 | 0.0230% | `EXACT` |
| Train forward/backward/optimizer | 2580.370 | 43.735 | 44.319 | 54.792 | 56.846 | 62.987 | 4.5615% | `EXACT` |
| Weight update/sync | 118.610 | 2.010 | 2.001 | 2.276 | 2.302 | 2.717 | 0.2097% | `EXACT` |
| Save | 331.901 | 5.625 | 0.001 | 27.638 | 27.975 | 29.520 | 0.5867% | `EXACT` |
| Checkpoint/recovery + other recorded phases | 472.071 | 8.001 | 8.001 | 8.002 | 8.002 | 8.002 | 0.8345% | `DERIVED` residual |
| Eval/cleanup standalone split | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | Included in residual if logged | `UNKNOWN` |
| Unattributed residual | 0 | 0 | 0 | 0 | 0 | 0 | 0% | Accounting `EXACT` |

大 step 几乎完全由 rollout wait 平移：rollout wait p95 为 31.81 min，而 train p95 只有 56.85 sec。Save 每约 5 step 出现一次约 28 sec 峰值，但不是一阶瓶颈。

### 5.2 Representative Steps

| Step | Role | Step wall | Rollout wait | Train | Other | Batch trajectories | Full-sequence tokens | Notes |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 12 | Exact p50 | 805.445 sec | 749.721 sec | 45.443 sec | 10.282 sec | 16 | 905968 | 7 positive / 9 non-positive |
| 38 | Above p95 threshold | 2050.262 sec | 1992.563 sec | 47.348 sec | 10.351 sec | 16 | 961403 | 14 positive / 2 non-positive；tail 仍由 wait 主导 |
| 1 | Maximum | 2427.906 sec | 2375.363 sec | 42.354 sec | 10.189 sec | 16 | 821331 | 6 positive / 10 non-positive |

Step wall 与当步 full-sequence token 的 Pearson `r=0.158`，说明“batch token 多”不是 trainer wait 长尾的充分解释。缺少 completion order 和 queue lineage，不能把长 step 归因到 batch 内某条 trajectory。

### 5.3 Async Activity And Overlap

| Metric | Value | Confidence |
|---|---:|---|
| Update-completion interval mean / p50 / p95 | `UNKNOWN`：没有连续 update completion 事件 | `UNKNOWN` |
| Logged phase sum / step count | 56568.883 sec / 59 = 958.795 sec | `EXACT` |
| Rollout active time | `UNKNOWN`：只有 trainer wait 和 episode RPC timing | `UNKNOWN` |
| Train active time | 2580.370 sec timer sum；11.468 train-GPU-h | `EXACT` / `DERIVED` |
| Rollout/train overlap time and ratio | `UNKNOWN` | `UNKNOWN` |
| Rollout-only exposed time | 53052.897 sec trainer-visible wait proxy | `EXACT` semantics |
| Train-only exposed time | `UNKNOWN`：background generation overlap 未 join | `UNKNOWN` |
| Coordination/idle time | `UNKNOWN` | `UNKNOWN` |
| Total allocated GPU-hours | 502.835 GPU-h = 56568.883 sec x 32 / 3600 | `DERIVED` |

## 6. Rollout Supply-Demand

### 6.1 Per-step Demand

```text
required_cohorts_per_step = 2
group_size = 8
required_trajectories_per_step = 2 x 8 = 16
```

| Metric | Mean | p50 | p95 | Max | Confidence |
|---|---:|---:|---:|---:|---|
| `prepare_batch` call to full batch return | 899.202 sec | 756.852 sec | 1908.432 sec | 2375.363 sec | `EXACT` |
| Time to first eligible cohort | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | 有 cohort-ready event，但尚未按 step join |
| Time to final required cohort from generation start | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | Existing queue state unknown |
| Wait after penultimate required cohort | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | No cohort completion order |
| Ready queue depth at dequeue | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | FIFO diagnostic 不是 dequeue snapshot；观测到 max ready=0 |
| In-flight trajectories at dequeue | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | FIFO diagnostic max active sessions=58 |

### 6.2 Whole-run Data Funnel

这张表使用全 60 step 和整个 generation pool，不能与 post-warmup 性能窗口混算。

| Stage | Trajectories | Share of generated | Meaning | Confidence |
|---|---:|---:|---|---|
| Expected cohort slots | 2000 | 100% | 250 cohorts x group size 8 | `EXACT` |
| Generated result | 2000 | 100% | `results.jsonl` unique episodes；ID 覆盖 0..2044，不是连续 0..1999 | `EXACT` |
| AReaL manager rewarded | 1891 | 94.55% | 1088 exported + 798 rejected + 5 open-at-shutdown | Aggregate `EXACT` |
| Exported from manager | 1088 | 54.40% | 136 complete 8-way cohorts | `EXACT` |
| Trainer consumed | 960 | 48.00% | 120 accepted cohorts x 8 | `EXACT` |
| Workflow filtered: uniform reward | 128 | 6.40% | 16 complete cohorts，reward std <= 1e-6 | `EXACT` |
| Manager rejected after reward | 798 | 39.90% | Stale/partial/incomplete/infra reasons below | `EXACT` |
| Open-at-shutdown but rewarded | 5 | 0.25% | seq 247；另有 11 active/unrewarded slots | `EXACT` |
| Not manager-rewarded by cutoff | 109 | 5.45% | 92 missing group slots + 6 rejected un/rewarded + 11 open active slots | `DERIVED` |

`960 + 128 + 798 + 5 + 109 = 2000`。这是基于同一 `idx-0..249` cohort ledger 的 aggregate closure；result episode ID 受动态调度影响并不连续，尚不能把每个 `results.jsonl` ID join 到 trainer step，也不能恢复各 disposition 的 token/GPU-hour 成本。

### 6.3 Manager Reject Reasons

| Reason | Cohorts | Rewarded but unused trajectories | Expected slots not rewarded | Mean cohort age at reject | Optimization boundary |
|---|---:|---:|---:|---:|---|
| Staleness drift 3/4/5/7 > 2 | 47 | 376 | 0 | 44.34 min | 先减少 wall-clock/version drift；`ofp=3` 最多可覆盖其中 280 条，但必须过 IS/clip/reward/downstream guardrail |
| Partial cohort deadline | 54 | 364 | 68 | 34.12 min | 15 min deadline 后延迟 sweep；优先 cohort-aware admission/补齐 member，单纯加 deadline 会增加 stale 与 HOL 风险 |
| Incomplete cohort ended before full group | 8 | 43 | 21 | 9.41 min | 缺 member 导致整组原子拒绝；可做同 policy-version replacement |
| `SandboxInternalException` | 2 | 14 | 2 | 39.78 min | 可重试 infra failure，但 replacement 必须保持 task/policy/version 语义 |
| Cohort timeout | 1 | 1 | 7 | 180.08 min | 10800 sec 上限生效；优先修 session hang，不建议继续抬 timeout |

Partial deadline 的 54 个 cohort 只缺少 67 个 admission slot，却连带丢弃 364 条已 reward 结果。`2000 - 1933 = 67` 只是与 joined-summary 缺口数值相等，不能作为 admission lineage；2026-07-22 的逐 episode 审计确认，真正与这 67 个 slot 一一对应的是缺少 `areal_start_session_begin`。它是当前提升 usable sample ratio 的第一优先级，不需要放宽 off-policy correctness。

### 6.4 Missing-member Root Cause Audit

证据链来自同一 run 的 `results.jsonl`、`ctx_timing.jsonl`、`evals.log`、`areal.log` 和当前代码。原始 group/rank 按 `entry_idx = (episode_id - group_id) // 8` 还原；manager 的 `slot_states=[r0...rN]` 不能用于判断缺失的原始 rank，因为 `_reserve_slot_locked()` 使用 admission 顺序重新编号。

| Audit item | Exact result | Interpretation |
|---|---:|---|
| Partial cohorts | 54 | 全部在 manager 侧因 `partial cohort deadline` 拒绝 |
| Admitted / rewarded | 365 / 364 | 另有 1 条 admitted trajectory 发生 `SandboxInternalException` |
| Never started session | 67 | 全部有 result artifact，但没有 `areal_start_session_begin` |
| Missing 1 / 2 / 3 / 4 members | 46 / 4 / 3 / 1 cohorts | 不是固定缺最后一个 member |
| Missing original rank 0..7 | 9 / 9 / 9 / 7 / 9 / 9 / 5 / 10 | 没有明显 tail-rank 偏置 |
| Reject age | min 1002.2 sec；p50 1899.5 sec；max 5465.7 sec | 15 min 是最早 eligibility；有 active confirmed slot 时实现继续等待 |

67 个 pre-session failure 的直接错误分布：

| Failure class | Count | Lifecycle stage |
|---|---:|---|
| `tmux` / sandbox command setup failure | 49 | 42 条明确 HTTP 404；其余为同类 setup/404 变体 |
| OpenHands install/bootstrap failure | 11 | install timeout、network error、SDK disconnect |
| Runtime network / SDK disconnect | 6 | 尚未创建 AReaL session |
| Sandbox internal `get_base_url` error | 1 | 尚未创建 AReaL session |

实际机制不是 deadline 主动“杀掉第 8 条”，而是：

```text
sandbox / OpenHands reset fails
  -> env wrapper cannot produce the first LLMRequest
  -> no areal_start_session_begin
  -> CohortManager only sees 4-7 admitted members
  -> admitted members finish and receive reward
  -> partial deadline becomes eligible and no confirmed member remains active
  -> whole cohort is rejected
```

因此，单纯增大 deadline 不会补出缺失成员，只会增加 policy drift 和 FIFO head-of-line blocking。正确修复点在 deadline 之前：预构建包含 `tmux` 和 OpenHands 的 sandbox image；对 create/setup 的 transient error 做有界重建与重试；由 orchestrator 先登记 `cohort_key + original_group_rank + attempt`，再启动昂贵的环境和 rollout；缺员时按相同 prompt、sampling config 和 behavior-policy version 补采。

另有两个 `incomplete` cohort（idx 119、230）是独立的控制面 race：manager 在分别只有 1 个 admitted slot 时，于 6.8 sec 和 3.1 sec 判断“现有 slot 已全部结束且 group 未满”并关闭 cohort；随后每组 6 个并发 `/rl/start_session` 返回 409，另 1 个原始 member 在 sandbox init 阶段失败。这里需要 atomic cohort manifest/reservation 或 producer-done handshake，不能靠客户端重试同一个已关闭 idempotency key。

公开方案与本次场景的对应关系：

| Scheme | Public evidence | Value here | Correctness boundary |
|---|---|---|---|
| Same-rank replacement | [GLM-5 heartbeat fault tolerance](https://arxiv.org/html/2602.15763#S3.SS6.SSS3) 将 retry 路由到健康 server | 首选；补 67 个未 admission member，加上 1 个 admitted infra-invalid member，可回收 364 条既有 reward trajectory | replacement 必须保持同 prompt/version；否则需明确 off-policy correction |
| Repeat valid members | [GLM-5 noisy-sample handling](https://arxiv.org/html/2602.15763#S4.SS1.SSS2) 在 valid count 超过 group 一半时重复 valid sample，否则丢组 | 本次理论上可保留 53/54 组、360 条 unique valid trajectory | 会改变重复样本权重和 group mean/std；应作为 algorithm-gated fallback，不等同于真实补采 |
| Dynamic sampling | [DAPO](https://arxiv.org/html/2503.14476#S3.SS2) 持续采样直到 non-uniform-reward batch 满 | 解决全 0/全 1 的零梯度 group | 不解决某个既有 GRPO group 的 infra 缺员 |
| Remove group barrier | [SAO](https://arxiv.org/html/2607.07508#S3) 用 single rollout、critic 和 token-level clipping | 从根本上消除 group completion barrier | 算法和系统改造大，不是当前 GRPO infra hotfix |

若短期不能做真实 replacement，次优实现不是直接复制 tensor：只把 duplicate 当 shape padding，group statistics 仅在 unique valid members 上计算，并用 `1 / repeat_count` sample weight 消除重复梯度权重；这仍是 variable-size GRPO，需要单独做 reward、KL/clip、下游评测 A/B。

## 7. Training Participation

| Level | Whole run | Performance window | Definition | Confidence |
|---|---:|---:|---|---|
| Trainer-consumed cohorts | 120 | 118 | Cohort entered advantage computation | `DERIVED` |
| Trainer-consumed trajectories | 960 | 944 | Trajectory entered trainer batch | `EXACT` |
| Full-sequence tokens processed | 51717550 | 51015838 | `perf/total_tokens`，完整 sequence token | `EXACT` |
| Loss-active trajectories | `UNKNOWN` | `UNKNOWN` | At least one final `loss_mask` token | `UNKNOWN` |
| Loss-active tokens | `UNKNOWN` | `UNKNOWN` | Final `loss_mask` count | `UNKNOWN` |
| Gradient-active trajectories | `UNKNOWN` | `UNKNOWN` | At least one non-zero effective contribution | `UNKNOWN` |
| Gradient-active tokens | `UNKNOWN` | `UNKNOWN` | Non-zero effective loss weight/advantage | `UNKNOWN` |
| Fully masked / zero-weight trajectories | `UNKNOWN` | `UNKNOWN` | Consumed but zero direct gradient | `UNKNOWN` |

`perf/total_tokens` 来自 consumed batch 的完整 sequence length，可以描述 trainer 算力负载，但不能冒充 loss-active token。性能窗口平均每条 consumed trajectory 为 54042.20 full-sequence tokens。

### 7.1 Consumed-data Composition

| Composition | Whole run | Performance window | Interpretation | Confidence |
|---|---:|---:|---|---|
| Reward > 0 | 625 / 960 = 65.10% | 619 / 944 = 65.57% | Positive outcome composition | `EXACT` |
| Reward <= 0 | 335 / 960 = 34.90% | 325 / 944 = 34.43% | 实际参与训练的 negative examples，不是浪费 | `EXACT` |
| Mean reward | 0.4720 | 0.4816 | Online reference | `EXACT` |
| Normal completion | `UNKNOWN` | `UNKNOWN` | Generated pool 有分布，但无法 join consumed | `UNKNOWN` |
| Max iterations | `UNKNOWN` | `UNKNOWN` | Same | `UNKNOWN` |
| Loop detector | `UNKNOWN` | `UNKNOWN` | Same | `UNKNOWN` |
| Timeout / infra error | `UNKNOWN` | `UNKNOWN` | Same | `UNKNOWN` |

## 8. Waste And Waiting Ledger

| Category | Trajectories / events | Wall time | GPU-hours | Share | Status / evidence |
|---|---:|---:|---:|---:|---|
| Stale complete cohorts | 47 cohorts / 376 trajectories | `UNKNOWN` | `UNKNOWN` | 18.80% generated | manager reject；全部 8/8 ended+rewarded |
| Partial/incomplete cohort amplification | 62 cohorts / 407 rewarded trajectories | `UNKNOWN` | `UNKNOWN` | 20.35% generated | 54 partial + 8 incomplete；完成样本被 group atomicity 连带丢弃 |
| Uniform-reward algorithm filter | 16 cohorts / 128 trajectories | `UNKNOWN` | `UNKNOWN` | 6.40% generated | 11 all-1、2 all-0、3 same dense reward；当前 group-relative advantage 无信号 |
| Retry / duplicate interactions | 1088 export events / 55397 orphan interactions | `UNKNOWN` | `UNKNOWN` | 54.81% of 101068 logged LLM turns，诊断比率 | export 日志明确标记 client retry 后 first result 未被 client observed；不是 55397 条 trajectory |
| Infra-invalid cohort rejection | 3 cohorts / 15 rewarded trajectories | `UNKNOWN` | `UNKNOWN` | 0.75% generated | 2 sandbox exception + 1 cohort timeout |
| Open-at-shutdown / manager-unrewarded | 5 rewarded + 109 not rewarded by cutoff | `UNKNOWN` | `UNKNOWN` | 5.70% generated | 最终 2 个 open cohort；6 个 controller wait future 被 shutdown cancel，不等于 6 条 trajectory |
| Trainer blocked waiting for rollout | 59 step wait events | 53052.897 sec = 14.737 h | 235.791 train-GPU-h | 93.78% step wall；46.89% total allocated GPU-h | `DERIVED`；16 train GPUs 在 `prepare_batch` 路径阻塞 |
| Cohort straggler tail | 54 partial deadline + 8 incomplete cohorts | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | 有 aggregate disposition，仍无 per-step penultimate-to-last join |
| Save + checkpoint/recovery exposed time | 59 steps | 803.972 sec | 3.573 train-GPU-h | 1.42% step wall | `DERIVED`；residual 含极小其他 timer |

结论：1040 条未消费已能按 aggregate disposition 解释，但各类 token/GPU-hour 仍未知。最值得先修的是 **partial cohort fill 和 retry idempotency**；直接接受 stale 样本虽然能提高表面利用率，却会改变 off-policy correctness。

## 9. Trajectory And Cohort Distribution

### 9.1 Trajectory Distribution

窗口：1933 个 joined generated episodes，不等于 960 个 trainer-consumed trajectories。

| Metric | Count / mean | p50 | p90 | p95 | p99 | Max | Confidence |
|---|---:|---:|---:|---:|---:|---:|---|
| Episode elapsed, min | 1933 / 17.39 | 15.26 | `UNKNOWN` | 36.28 | 54.15 | 180.70 | `EXACT` except p90 |
| Turn count | 52.29 | 50 | 80 | 80 | 80 | 80 | `EXACT` |
| Max prompt tokens | `UNKNOWN` mean | 58.2K | `UNKNOWN` | 98.2K | 126.9K | 131034 | `EXACT` except mean/p90 |
| Cumulative LLM RPC | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | Per-episode aggregate not preserved in scorecard artifact |

| Cost composition | Count | Trajectory share | Episode-wall share | LLM-RPC share | Confidence |
|---|---:|---:|---:|---:|---|
| Reward <= 0 | 629 | 32.54% | 41.73% | 41.95% | `EXACT` generated pool |
| Max iterations | 364 | 18.83% | `UNKNOWN` | `UNKNOWN` | `EXACT` count |
| Loop detector | 96 | 4.97% | `UNKNOWN` | `UNKNOWN` | `EXACT` count |
| Max prompt >= 120K | 32 | 1.66% | `UNKNOWN`；mean elapsed 50.60 min | `UNKNOWN` | `EXACT` count |
| 80-turn cap reached | 375 | 19.40% | `UNKNOWN`；mean elapsed 26.20 min | `UNKNOWN` | `EXACT` count |

375 条 80-turn episode 中只有 10 条 reward > 0，positive rate 为 2.67%，mean score 为 -0.458。它们是优先研究的高成本组成，但不能在没有训练效果实验时直接删掉。

2000 个 result artifact 的终止分布为 1417 completed、484 truncated、99 error；1933 个 ctx-joined episode 中 error 为 32。failure origin 为 364 `max_iterations`、96 `loop_detector`、1 `run_timeout`。Score 与 turn count 的 Pearson `r=-0.433`，与 elapsed 的 `r=-0.301`。

### 9.2 Cohort Tail

| Metric | Mean | p50 | p90 | p95 | Max | Confidence |
|---|---:|---:|---:|---:|---:|---|
| Exported cohort age, minutes | 27.45 | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | 71.38 | 136 exported cohort terminal logs；min 11.28 min |
| Wait after penultimate trajectory | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | No completion order |
| Max / median duration | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | No stable cohort join |
| Straggler reward <= 0 share | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | No stable cohort join |

Run 结束时 seq 247 为 `5/8 rewarded, 3 active`，seq 249 为 `0/8 rewarded, 8 active`。FIFO diagnostic 的 max 为 15 open cohorts、58 active sessions、107 pending claims，ready queue 始终为 0；主要矛盾是 incomplete/open cohort 和供给长尾，不是 ready backlog 堆积。

## 10. Turn And Engine Metrics

| Metric | Mean | p50 | p90 | p95 | p99 | Max | Confidence |
|---|---:|---:|---:|---:|---:|---:|---|
| LLM RPC latency, sec | `UNKNOWN` | 7.09 | `UNKNOWN` | 64.97 | 140.59 | 385.51 | `EXACT` available quantiles |
| TTFT | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | Not traced |
| Decode tok/s | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | Not traced |
| Per-request cache ratio | weighted 73.56% | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `EXACT` weighted aggregate |
| Queue wait | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | Not traced |
| KV usage | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | Not joined |
| Tool latency | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | Not split |
| Sandbox latency | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | Not split |
| Orphan interactions dropped/export | 50.92 | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | 79 | 55397 / 1088；`EXACT` count |

`agent_llm_decode_latency_sec` 实际是 proxy request 到 response 的整次 LLM RPC，包含 queue、prefill、decode、serialization 和 network，不能解释成纯 decode。

| Turn range | Turns | Weighted `cached_tokens / prompt_tokens` |
|---|---:|---:|
| 1-4 | 7592 | 75.76% |
| 5-9 | 9485 | 92.00% |
| 10-19 | 18854 | 95.44% |
| 20+ | 65137 | 69.45% |
| All turns | 101068 | 73.56% |

80-turn episode 的 mean per-episode cache ratio 为 54.47%，未到上限 episode 为 87.00%。Episode elapsed 与累计 LLM RPC 的 `r=0.906`，与 max prompt token 的 `r=0.660`，说明 LLM request path 与 context growth 是 generation cost 的一阶解释。

## 11. Correctness And Training Effect

| Guardrail | R8b baseline | This run | Allowed range | Result |
|---|---:|---:|---:|---|
| Trainer-consumed trajectory count | 256/step | 16/step，60 step 无缺口 | Exact run workload | Pass for diagnostic；not comparable |
| Loss-active token/trajectory count | `UNKNOWN` | `UNKNOWN` | No unexplained regression | Pending |
| Reward distribution | Historical | Whole-run mean 0.4720 | Diagnostic only | Observed |
| Context-length distribution | Historical | consumed seq avg 34.4K-73.1K by step | Diagnostic only | Observed |
| Turn/failure distribution | Historical | Generated pool available，consumed join missing | Diagnostic only | Partial |
| Policy-version lineage | `UNKNOWN` | manager version/disposition 可恢复；result -> consumed ID 仍缺 | 100% valid | Partial |
| Group completeness | 8 | 136 exported 为 8/8；62 partial/incomplete 被拒绝 | 100% at trainer boundary | Pass at consumed boundary |
| Logp / importance-ratio diagnostics | Historical gap | `UNKNOWN` | Pre-registered | Pending |
| True EOS / truncation reason | Historical gap | Existing trainer no-EOS metric invalid | 100% classified | Fail |
| Infra failure rate | 3.16% sandbox abnormal | Consumed-level infra failure unknown | No regression | Pending |

当前 trainer 使用：

```python
no_eos_ratios = (seqlens == attn_mask.shape[-1]).float()
```

同时 `pad_to_maximum=false`，所以它会把当前 dynamic batch 中最长的 sequence 当作 no-EOS。59 个 step 中该指标几乎固定为 `1/16`，这是实现结构导致的，不是真实 EOS 观测。

| Layer | Metric | Role | Result |
|---|---|---|---|
| During training | Reward/outcome distribution | Online reference | Mean reward 0.4720；composition recorded |
| Downstream | Pre-registered benchmark | Final effect gate | Not applicable for this diagnostic |
| Efficiency | Time/GPU-hours to target quality | Final efficiency result | Not evaluated |

## 12. Causal Analysis

当前最值得验证的机制链：

```text
prefix cache improves early/mid-turn latency
  -> more episodes reach late turns
  -> context continues growing
  -> cache reuse falls after turn 20
  -> LLM RPC tail expands
  -> long-running sessions occupy rollout capacity
  -> trainer waits longer for the next full 2-cohort batch
```

支持证据：turn 10-19 cache ratio 95.44%，turn 20+ 降到 69.45%；episode elapsed 与累计 LLM RPC、max prompt 的相关系数为 0.906、0.660；step wall 的 93.78% 是 `prepare_batch` wait。

反证与限制：单个 run 无法证明 cache 导致 trajectory 变长；step wall 与当步 consumed token 的相关性只有 0.158；aggregate cohort disposition 已恢复，但没有 result -> consumed step 和 cohort completion order，所以仍不能确认哪条 trajectory 阻塞了 trainer。

新的直接证据给出三条独立机制：

1. 54 个 partial cohort 的 67 个缺员已全部归因到 pre-session sandbox/OpenHands failure；它们连带丢掉 364 条已 reward 结果。应优先做 sandbox 预构建、cohort manifest 和 same-version replacement，而不是降低 valid predicate 或增加 deadline。
2. 47 个 complete cohort 因 policy drift > 2 丢掉 376 条；`ofp=3` 的理论回收上限是 280 条，但必须检查 importance ratio、clip fraction、reward 和下游评测。
3. 1088 个 exported trajectory 均记录 orphan-filter event，共丢弃 55397 个 retry interaction；需要 request idempotency、response replay 和 timeout/p99 对齐。

必须补齐的 join：

1. `result_id -> cohort_id/rank -> manager disposition -> trainer step -> policy version`，并记录 disposition timestamp/reason。
2. 最终 `loss_mask`、effective loss weight、advantage 非零 count。
3. `request -> engine queue/KV snapshot` 和统一 monotonic timeline。

## 13. Decision

| Item | Result |
|---|---|
| Performance verdict | 相比 R8b step mean 低 80.95%，但 logical work 少 16 倍；cohort/rollout GPU-h 低 69.5%，不能称为训练提速 |
| Participation verdict | 960 条 trajectory 确认进入 trainer；loss-active/gradient-active 未观测 |
| Correctness verdict | consumed boundary 保持 8/8 group；stale 均按 `max=2` 拒绝。no-EOS guardrail 仍无效，per-result consumption lineage 不完整 |
| Final decision | Diagnostic only |
| Confidence | E2E phase timing High；aggregate disposition High；per-result token/GPU cost Low |
| Rollback trigger | N/A |
| Next experiment | 先做 cohort-aware admission + retry idempotency 的 bs2 A/B；另行做 `ofp=2/3` correctness-gated A/B，不能与调度改动混在一个实验 |

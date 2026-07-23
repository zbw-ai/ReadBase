# Metric Contract

Version: `v2.0`

这个 contract 定义长上下文 Agentic RL 性能实验必须输出什么、如何 join、如何聚合。目标是让不同实验可以机械比较，而不是每次重新解释日志。

## Metric Hierarchy

| Layer | Question | Primary output |
|---|---|---|
| End-to-end | 训练多久得到一个 batch？ | step/rollout/train time |
| Supply-demand | 为凑齐 batch 生成、等待和丢弃了多少数据？ | generated/consumed/queued/dropped |
| Participation | 哪些 trajectory/token 实际进入 trainer 和 loss？ | consumed/loss-active/gradient-active |
| Cohort | 最后一条 trajectory 造成多少等待？ | group wall、wait after 7th |
| Trajectory | 时间花在什么 outcome 和 context shape 上？ | elapsed、turn、length、failure |
| Turn | 每轮 LLM、tool、sandbox 分别多慢？ | TTFT、decode、tool latency |
| Engine | 路由、APC、queue、KV 是否解释 turn latency？ | cached tokens、engine、queue/KV |
| Correctness | 性能变化是否破坏训练语义？ | lineage、logp、mask、distribution |

## Primary Optimization Objective

```text
minimize post-warmup end-to-end step time

subject to:
  consumed_cohorts_per_step is fixed
  group_size is fixed
  maximum_sequence_length is fixed
  total resources are fixed at 32 A100-80GB GPUs
  downstream checkpoint evaluation is non-inferior
  trainer participation does not silently decrease
  algorithm correctness guardrails pass
```

Rollout/train allocation 和并行拓扑可以作为单一实验变量，但总资源预算保持不变。对于当前 R8b workload，logical batch 是 32 cohorts x 8 trajectories，最大序列长度是 128K。实际 token、turn 和 context length 是需要观测的结果变量，不要求机械固定；在下游评测不回归时减少这些成本属于有效优化。

## Async Step-time Semantics

训练和 rollout 异步重叠时，主指标按 wall-clock 事件定义：

```text
update_interval_i = update_complete_i - update_complete_(i-1)
steady_state_step_time = mean(post_warmup update_interval_i)
window_step_time = (last_update_complete - first_update_complete) / interval_count
```

包含 K 个连续 update interval 的窗口需要 K+1 个 completion timestamp，避免把 update 数和 interval 数混淆。

同时在统一 timeline 上计算：

```text
makespan
= rollout_only_exposed
 + train_only_exposed
 + rollout_train_overlap
 + coordination_or_idle
```

Rollout active time 与 train active time不能直接相加。框架中的 `timeperf/rollout` 在 online mode 可能表示 trainer 等待/dequeue ready data 的时间，也不能解释为纯 generation time。

## Required Join Keys

每条事件至少携带适用范围内的以下字段：

```text
experiment_id
run_id
policy_version
train_step
cohort_id
task_id
trajectory_id
session_id
interaction_id
turn_idx
engine_id
sandbox_job_id
```

最小可用链路：

```text
train_step
  -> cohort_id
  -> trajectory_id/session_id
  -> interaction_id/turn_idx
  -> engine_id
  -> sandbox_job_id
```

缺失 join key 时，可以保留分布统计，但不能声称 request-level 或 trajectory-level 因果关系。

## End-to-end Metrics

| Metric | Definition |
|---|---|
| `step_time_sec` | 一个 trainer step 的 wall time |
| `rollout_wait_sec` | online trainer 等待满足 batch 的 wall time；不是纯 generation time |
| `train_time_sec` | forward/backward/optimizer 的 wall time |
| `update_interval_sec` | 相邻 update/optimizer completion timestamp 的差 |
| `window_makespan_sec` | post-warmup 窗口首尾 update completion 的 wall time |
| `rollout_train_overlap_sec` | rollout active interval 与 train active interval 的交集 |
| `exposed_rollout_sec` | 未被 train 覆盖且位于 update critical path 的 rollout 时间 |
| `weight_update_sec` | actor 到 rollout engine 的权重同步时间 |
| `total_gpu_hours` | `window_makespan_sec * 32 / 3600` |
| `trainer_consumed_cohort_goodput` | trainer-consumed cohorts / rollout GPU-hour |
| `trainer_consumed_trajectory_goodput` | trainer-consumed trajectories / total或rollout GPU-hour，必须标注分母 |
| `loss_active_token_goodput` | 最终 `loss_mask` 为真的 token / total GPU-hour |
| `gradient_active_token_goodput` | effective loss weight/advantage 非零的 token / total GPU-hour |

`update_interval_sec` / `window_makespan_sec` 是异步场景的主性能指标。`trainer_consumed_cohort_goodput` 用于确认 logical work，`loss_active_token_goodput` 和 `gradient_active_token_goodput` 用于判断进入 loss 和真正产生直接梯度贡献的数据量。历史 run 缺少 loss mask/advantage 统计时，不能拿 full-sequence token 代替 active token。

## Evidence Status

每个聚合值必须携带以下状态之一：

| Status | Meaning |
|---|---|
| `EXACT` | 原始日志直接记录，或可以从同一窗口的完整原始数据确定 |
| `EXACT_AGGREGATE` | 同一窗口的总数能够精确闭合，但缺少逐 trajectory/request ID join；可以使用总量，不能声称逐对象 lineage 已验证 |
| `DERIVED` | 由 `EXACT` 数据按公开公式计算 |
| `DERIVED_UPPER_BOUND` | 由可靠总数和公开公式计算的理想收益上限；依赖明确假设，不代表已经观测到的实际收益 |
| `INFERRED` | 依赖尚未通过 lineage 验证的假设 |
| `UNKNOWN` | 现有 tracing 不能回答 |

`EXACT_AGGREGATE` 不是估算：aggregate 总数准确，缺失的是每个对象的身份映射和对应成本。不能为了让表格完整而把 `INFERRED` 写成 `EXACT`，也不能用 `0` 表示 `UNKNOWN`；`DERIVED_UPPER_BOUND` 不能写成实际收益。

## Rollout Supply-Demand Contract

每个 trainer step 必须先明确需求：

```text
required_trajectories_per_step
= required_cohorts_per_step * group_size
```

每个 rollout result 需要记录最终 disposition：

```text
attempted
generated_result
manager_rewarded
manager_exported
structurally_eligible
trainer_consumed
workflow_filtered_uniform_reward
manager_rejected_staleness
manager_rejected_partial_cohort
manager_rejected_incomplete_cohort
manager_rejected_infra
queued_at_window_end
dropped
stale
cancelled
open_at_shutdown
terminal_unconsumed
```

当且仅当 `trajectory_id` 可以贯穿 generator、queue 和 trainer 时，下面的账目才可以标记为 `EXACT`：

```text
generated_result
= trainer_consumed
 + workflow_or_algorithm_filtered
 + manager_rejected_after_reward
 + queued_at_run_end
 + dropped_or_stale_or_cancelled
 + open_or_not_rewarded_at_shutdown
 + terminal_unconsumed
```

Result artifact、manager rewarded、manager exported、trainer consumed 和 gradient active 是五个不同层级。特别是：

- `results.jsonl` 有记录不代表形成了 AReaL 可导出的 session trajectory。
- partial/incomplete cohort 中已经 reward 的成员仍会因 group atomicity 整组不可训练。
- uniform-reward group 可以完整 export，但在 group-relative advantage 下没有学习信号，必须单列 algorithm filter。
- controller wait future 的 cancel 是内部 task event，不能直接记成 cancelled trajectory。

`queued_at_window_end` 不等于浪费：它可能被后续 step 消费。只有 run 已终止且不再复用时，才转为 `terminal_unconsumed`。允许用 cohort seq、group size、admitted/ended/rewarded slot 做 aggregate closure，但必须标记 `EXACT_AGGREGATE`；只有 result ID/rank 可以贯穿 manager 和 trainer 时，才能标记 per-result `EXACT`。

每个 cohort terminal event 至少输出：

```text
cohort_id, cohort_seq, sample_key, group_size
manager_rollout_version, manager_current_version, manager_head_drift
token_behavior_version_min/p50/p95/max, train_version
token_staleness_min/p50/p95/max, token_version_coverage
expected_slots, admitted_slots, ended_slots, rewarded_slots
status, disposition_reason, created_at, terminal_at, age_sec
trainer_task_id, trainer_step
```

`max_head_offpolicyness` 约束的是 cohort/control-plane head，不应直接解释成 per-token hard threshold。真实训练数据的 off-policy 程度必须由 `versions` 与 `train_version` 逐 token 计算，并与 rejection、importance ratio 和 clip 结果一起报告。

每个 step 还应记录：第一个 eligible cohort 到达时间、最后一个 required cohort 到达时间、penultimate-to-last wait、dequeue 时 ready queue depth 和 in-flight 数。这样才能区分供给不足、cohort straggler 和 scheduler 空转。

## Training Participation Contract

本项目把实际被 trainer 消费的数据都视为有价值训练数据，不使用 reward 正负判断浪费。参与度按四层统计：

| Level | Definition |
|---|---|
| `trainer_consumed` | trajectory 出现在 trainer batch，并进入 `compute_advantages -> ppo_update` 路径 |
| `loss_active` | trajectory 至少一个 token 的最终 `loss_mask` 为真 |
| `policy_gradient_active` | trajectory 至少一个 token 对当前 actor policy logprob 产生直接非零梯度；必须在 hard mask、rejection、weight 和 loss branch 选择后判断 |
| `full_sequence_tokens` | trainer 处理的完整 sequence token；用于算力负载，不等于 active token |

必须同时输出：

```text
trainer_consumed_cohorts
trainer_consumed_trajectories
full_sequence_tokens
loss_active_trajectories
loss_active_tokens
policy_gradient_active_trajectories
policy_gradient_active_tokens
fully_masked_or_zero_weight_trajectories
```

对于当前 PPO path，`policy_gradient_active` 不能只用 `advantage != 0` 近似。P0 的精确定义为：

```text
policy_gradient_active_token
= final_hard_loss_mask
  AND effective_advantage != 0
  AND effective_loss_weight != 0
  AND behave_importance_weight > 0       # rejection sampling enabled 时
  AND policy_ratio_derivative is finite and non-zero
  AND NOT clip_mask
  AND NOT dual_clip_mask
```

其中 `final_hard_loss_mask` 已包含 M2PO 和 rejection `action=mask` 的结果。SAPO、KD 或其他 actor loss 必须分别实现并测试自己的 derivative-active 规则；没有对应实现时标记 `UNKNOWN/unsupported`，不能输出 0，也不能标记 `EXACT`。

推荐的 token 账目：

```text
full_sequence_tokens = prompt_tokens + response_tokens
response_tokens = loss_active_tokens + response_masked_tokens
loss_active_tokens = policy_gradient_active_tokens + zero_gradient_contribution_tokens
```

GRPO 中 `reward <= 0` 可以是必要的 negative example。reward、completion/failure origin、turn 和 context length 属于 consumed-data composition，用于理解学习信号和成本，不直接进入 waste ledger。

## Waste And Waiting Contract

确认浪费只包括有明确证据的资源消耗：

- run 结束后仍未消费且不会复用的 terminal overproduction。
- scheduler 明确 dropped、stale 或 cancelled 的 rollout。
- retry/duplicate 和明确的 infra-invalid execution。
- partial/incomplete cohort 因 group atomicity 连带丢弃的已完成成员。
- 已 export 但被 uniform-reward/zero-signal predicate 过滤的 cohort；这类数据不能靠绕过 predicate 变成 gradient-active。
- trainer partition 因等待 rollout、同步、checkpoint 或调度而空转的 GPU-hours。
- cohort 中等待最后一条 trajectory 的 exposed tail。

昂贵但已被 trainer 消费的数据不能计为浪费。80-turn、max-iteration、loop 或 non-positive reward 可以列为“高成本组成/优化候选”，但删除或 early-stop 前必须通过训练效果 guardrail。

等待 GPU-hours 的默认计算为：

```text
trainer_blocked_gpu_hours
= exposed_trainer_wait_sec * allocated_train_gpus / 3600
```

如果无法证明整个 train partition 在该区间空闲，必须降级为 `INFERRED`。

Retry 需要同时记录 `logical_interaction_id`、`attempt_id`、idempotency key、server completion、client ACK 和最终 retained/dropped disposition。`orphan_interaction_count` 是 request/turn 层的浪费，不能冒充 trajectory count；token/GPU 成本只有在 orphan interaction 能 join 到 engine token timing 后才可计算。

## Valid Predicate

一个 cohort 至少需要满足：

- policy version 和 interaction lineage 可验证。
- group size、advantage/reward 语义符合当前算法配置。
- trajectory tensorization、loss mask、no-EOS/truncation 处理正确。
- 不是 sandbox provision、network、port collision 等 infra failure。
- 正样本和算法定义的 trainable negative 都可计入。
- 因 tracing 缺失而无法判断时标记 unknown，不能默认 valid。

具体 predicate 需要随算法配置版本化，建议字段为 `valid_predicate_version`。

## Trajectory Metrics

每条 trajectory 输出：

- start/end/elapsed time。
- task、cohort、trajectory、session、policy version。
- reward、failure origin、termination reason、trainable/valid classification。
- turn count、max/cumulative prompt tokens、completion tokens。
- max context length、no-EOS、truncation、patch lines/files。
- repeated action、error turn、zero-progress 等 agent behavior 指标。

聚合必须输出 count、mean、p50、p90、p95、p99 和 max；只给平均值不足以描述 Agentic RL 长尾。

### No-EOS Contract

不能用下面的 dynamic-batch proxy 判断真实 no-EOS：

```python
seq_len == attention_mask.shape[-1]
```

当 `pad_to_maximum=false` 时，`attention_mask.shape[-1]` 是当前 batch 的 padded width，至少一条最长序列会被标记。真实 no-EOS 必须来自 rollout metadata，并至少区分：

- terminal EOS observed；
- configured token/context cap；
- max iterations / loop detector；
- environment timeout / infra error；
- normal agent completion。

在新指标完成前，旧 `ppo_actor/no_eos_ratios` 标记为 `INVALID_DYNAMIC_PADDING_PROXY`，不能进入 correctness acceptance。

## Cohort Metrics

对于 group size 8：

```text
group_wall = max(end_i) - min(start_i)
wait_after_7th = max(end_i) - second_max(end_i)
tail_fraction = wait_after_7th / group_wall
straggler_ratio = max(elapsed_i) / median(elapsed_i)
```

同时输出 straggler 的 reward、failure origin、context/turn、engine 和 sandbox 信息。

## Turn And Engine Metrics

每次 LLM call 应输出：

- request/response timestamp、LLM RPC latency。
- prompt、completion、cached tokens。
- per-request cache ratio：`cached_tokens / prompt_tokens`。
- TTFT、decode duration、inter-token latency或 decode tok/s。
- engine ID、queue wait、running/waiting requests、KV usage。
- prompt token LCP 或可解释的 prefix reuse 指标。
- tool name/category、tool execution latency、observation size。
- sandbox RPC latency、retry 和 error type。

注意：LLM RPC latency 不能命名为纯 `decode_latency`，因为它可能包含 queue、prefill、decode、serialization 和 network。

## Statistical Windows

- 默认 E2E 比较排除 step 0 warmup，并报告完整 update interval 数、window makespan、mean、p50 和 p95。
- 所有 overlap 计算必须来自同一 monotonic clock 或经过验证的时钟对齐。
- trajectory/cohort 指标只使用在窗口内完整结束的对象。
- 对异步系统，同时报告 completed、trainer-consumed、queued 和 dropped 数量。
- generation pool 与 trainer-consumed batch 必须分开统计；没有 consumption lineage 时，只能分析生成成本，不能推断训练数据分布。
- A/B 使用相同 task manifest、seed、checkpoint 和资源。
- 至少报告绝对值和相对 baseline delta。
- 不同窗口不能在同一单元格中直接做 delta。

## Current Branch Coverage

当前 `origin/zbw/r2e-9b-128k` 已具备：

- `session_id`、`interaction_id`、`turn_idx`、`engine_id`。
- LLM request/response timestamp 和整次 RPC latency。
- prompt tokens、cached tokens、prompt chars 和 char-level LCP。
- agent turn、repeated/error action、patch、failure family 和 episode elapsed。
- proxy event lineage 和 generation begin/end 事件。

仍需补齐或验证：

- completion tokens 的统一输出。
- TTFT、prefill、decode/ITL 的可靠拆分。
- request-level queue/KV snapshot。
- tool 和 sandbox latency 分解。
- train step/cohort/trajectory/policy version 的端到端 join。
- exact code/image/model/data manifest。
- tracing 本身的性能开销和 observation-only 保证。

## Acceptance Check

一次实验只有同时满足下面条件，才能进入 dashboard：

1. Identity manifest 完整，或明确列出缺失项。
2. 统计窗口、count 和 warmup policy 明确。
3. baseline 与 candidate 的主要变量可解释。
4. 指标可以追溯到原始日志或结构化 artifact。
5. correctness guardrail 已判定。
6. Overlap-aware E2E step time、logical batch、最大序列长度和 32 卡总资源可直接比较。
7. 所有 `N/A` 都有原因，没有用 0 代替缺失值。

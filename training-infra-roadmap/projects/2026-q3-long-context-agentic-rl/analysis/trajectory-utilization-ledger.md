# Trajectory Utilization Ledger

Last updated: 2026-07-23

这份 living ledger 追踪一条 trajectory 从生成到产生梯度贡献的完整去向。它从 bs2 run 的 `2000 generated - 960 trainer consumed = 1040` 出发，但不把 1040 条统一称为浪费；每一类数据都必须有明确 disposition、证据等级、资源成本和实验结论。

当前首个审计对象是 [bs2-eqtraj-C1b-v2](../cases/bs2-eqtraj-C1b-v2.md)，机器可读源数据见 [bs2 metrics](../metrics/bs2-eqtraj-C1b-v2.json)。后续 `TU-E0a/E0b/E0c`、matched `TU-A0` 和 P1/P2 run 都在本文追加横向结果，不另建一套统计口径。

## 0. Owner Review

### 0.1 How to read the evidence labels

证据等级回答的是“这个数字可以被相信到什么粒度”，不是评价数字好坏。

| Label | 含义 | 本文中的例子 |
|---|---|---|
| `EXACT` | 日志直接记录了所描述的对象，并且对象语义明确。可以确认具体发生了多少次。 | `results.jsonl` 有 2000 条记录；trainer 明确消费了 960 条 trajectory。 |
| `EXACT_AGGREGATE` | 总数可以由同一窗口的 manager/cohort 账本精确闭合，但缺少逐 trajectory ID join。可以确认“这一类合计多少条”，不能确认“result ID X 最终属于哪一类”。 | 可以确认 798 条 rewarded trajectory 位于被拒绝的 cohort，但不能把 798 条逐一映射回 2000 个 result ID。 |
| `DERIVED_UPPER_BOUND` | 使用可靠计数和公开公式计算出的理想收益上限，不是已经观察到的收益。它依赖表中写明的假设。 | `407 / 89 = 4.57` 假设补齐后所有 group 都及时 export、未 stale、未被 uniform-reward filter，并最终被 trainer 消费。 |
| `UNKNOWN` | 当前 tracing 没有记录，或缺少必要 join，不能从现有数据可靠回答。 | 960 条 consumed trajectory 中有多少 loss-active/policy-gradient-active，当前未知。 |

最容易混淆的是 `EXACT_AGGREGATE`：它不是估算，**总数是准确的**；不准确或缺失的是每一条 trajectory 的身份映射和对应 token/GPU 成本。

### 0.2 Current data funnel

| Item | Current result | 代表什么 | 应该如何解读 | Evidence |
|---|---:|---|---|---|
| 生成结果记录 `Generated result artifacts` | 2000 | `results.jsonl` 中存在 2000 条 episode 结果记录，包含成功、截断和 error artifact | 只代表 eval runner 产出了结果文件记录，不代表 2000 条都成功 rollout、获得 reward 或进入训练 | `EXACT` |
| Manager 已接收 reward `Manager rewarded` | 1891 | AReaL manager 已为这些 session/member 接收到 final reward | Reward 完成仍不代表 group 完整；其中很多随后因 stale、partial 等原因被整组拒绝 | `EXACT_AGGREGATE` |
| Manager 已导出 `Manager exported` | 1088 | 136 个完整的 8-way cohort 被 manager 交给 workflow | 它们通过了 manager 的 group 完整性和 staleness 检查，但还可能被 workflow filter | `EXACT_AGGREGATE` |
| Workflow 全同 reward 过滤 | 128 | 16 个完整 group 的 8 条 reward 相同或近似相同 | 当前 GRPO group-relative advantage 为零，因此没有进入 trainer；它们不是 infra-invalid | `EXACT_AGGREGATE` |
| Trainer 实际消费 `Trainer consumed` | 960 | 120 个完整 group 进入 trainer 的 `prepare_batch -> compute_advantages -> ppo_update` 路径 | 这是当前可以确认实际参与训练的数据；reward <= 0 的 consumed trajectory 也计入 | `EXACT` |
| 未消费总量差 `Aggregate unconsumed gap` | `2000 - 960 = 1040` | Result artifact 数与 trainer-consumed 数之间的总量差 | 1040 不是一种统一原因，也不能全部叫作浪费；必须继续拆成 algorithm filter、manager reject、open 和未 reward | `EXACT_AGGREGATE` |
| 已 reward 但被 manager 拒绝 | 798 | 已经完成 reward，但所属 cohort 被 manager 拒绝的 trajectory | 这是完成计算后没有进入 workflow 的最大一类，包括 stale、partial、incomplete 和 infra reject | `EXACT_AGGREGATE` |
| Shutdown 时仍 open 且已 reward | 5 | Run 结束时仍在 open cohort 中的 rewarded trajectory | 窗口结束时不能直接判定浪费；只有确认不会在后续恢复或复用时才是 terminal unconsumed | `EXACT_AGGREGATE` |
| 截止日志终点尚未被 manager reward | 109 | Result artifact 存在，但截至日志终点没有进入 manager rewarded 总数 | 总数确定，但内部包含 pre-session failure、unfinished、error 等不同原因，逐条原因尚未闭环 | `EXACT_AGGREGATE`；breakdown `UNKNOWN` |

上表的守恒关系是：

```text
2000 generated artifacts
= 960 trainer consumed
 + 128 workflow filtered
 + 798 manager rejected after reward
 +   5 open-at-shutdown rewarded
 + 109 not manager-rewarded by cutoff
```

### 0.3 Current recovery focus

| Item | Current result | 代表什么 | 应该如何解读 | Evidence |
|---|---:|---|---|---|
| Partial/incomplete 中已 reward 的轨迹 | 407 | 62 个不完整 group 中已经成功结束并得到 reward、但因缺少其他 member 而未训练的 trajectory | 这是 cohort atomicity 带来的连带损失，也是当前最明确的 infra/control-plane 回收目标 | `EXACT_AGGREGATE` |
| 补齐这些 group 所缺的 slot | 89 | 要让上述 62 个 group 达到 8/8，合计还缺 89 个成功 rewarded member | 其中包括 never-started、session failure 和 early-close/409；不是 89 条都已经消耗了完整 rollout GPU 计算 | `EXACT_AGGREGATE` |
| 理论回收杠杆 `Recovery leverage` | `407 / 89 = 4.57` | 理想情况下，每补成 1 个缺失 slot，可以保住 4.57 条已经完成的既有 trajectory | 这是理论上限，不是预期 step-time 提升；补齐后仍可能 stale、uniform-filtered 或未被 trainer 消费 | `DERIVED_UPPER_BOUND` |
| 逐 trajectory 去向闭环 | 尚未闭环 | 还不能为 2000 个 result ID 各自标记 manager/workflow/trainer 的最终状态 | TU-E0a 必须补齐 ID join，之后才能计算每类真实 token、wall time 和 GPU-hour | `UNKNOWN` |
| 实际进入 loss 的轨迹/token `Loss active` | 未记录 | Consumed 数据中有多少 token 的最终 `loss_mask` 为真 | `trainer consumed` 不自动等于所有 token 都进入 loss | `UNKNOWN` |
| 实际产生梯度的轨迹/token `Gradient active` | 未记录 | Loss-active 数据中有多少具有非零 advantage/effective weight、产生直接梯度贡献 | 这是样本利用漏斗的最终系统指标，不能用 full-sequence token 代替 | `UNKNOWN` |

### 0.4 R5 six-update live closure

R5 是新的 bs2 P0 diagnostic，不是历史 `2000 -> 960` 窗口的重算。它首次用同一个
`trajectory_uid` 把 manager、workflow、trainer、loss 和 policy gradient 在真实训练 update 上连通。
本节冻结在 `global_step=5 -> version=6` 完成后的 strict audit：

```text
223 admitted attempts
 -> 180 generated/rewarded
 -> 96 manager/workflow exported
 -> 96 trainer consumed
 -> 94 policy-gradient-active
    2 trainer-consumed compact-filtered

84 generated but not trainer-consumed
= 35 partial + 16 stale + 33 rewarded waiting
```

| Item | R5 result | Interpretation | Evidence |
|---|---:|---|---|
| Trainer rows / exact join | `96 / 96` | 6 个 optimizer update 的每条 trainer trajectory 都能回溯到 manager/workflow；missing/conflict 为 0 | `EXACT` |
| Full-sequence tokens | 4,073,826 | Trainer 对 96 条 trajectory 的 full-sequence processing 口径 | `EXACT` |
| Response / loss / policy-gradient tokens | 518,340 / 518,340 / 518,340 | token conservation 成立；94 条 trajectory 有直接 policy-gradient contribution | `EXACT` |
| Fully masked | 2 trajectories / 159,330 full-sequence tokens | 两条都被 trainer 消费但 response/loss/gradient 为 0；占本窗口 trainer full-sequence tokens 的 3.91% | `EXACT` |
| Fully-masked cause | signal terminated；`AgentState.ERROR` | reward 时间戳与 eval failure/compact-filter 日志逐条同秒闭环 | `EXACT` |
| Partial cohorts | 5 cohorts / 35 rewarded trajectories | `idx-2/4/6/7/12` 都是 7/8 后 `partial_timeout`；分别缺 source rank `6/3/3/4/0` | `EXACT` |
| Stale cohorts | 2 cohorts / 16 rewarded trajectories | `idx-17/22` 都已 8/8 完成，rollout version 分别为 2/3，随后 terminal staleness | `EXACT` |
| Terminal-unconsumed rollout token cost | `UNKNOWN` | 当前 R5 源码没有为上述 51 条 trajectory 记录 rollout prompt/output token work | `UNKNOWN` |

两条 fully-masked trajectory 的 full-sequence 分别是 28,258 和 131,072 tokens。这个 159,330
可以解释 trainer 侧 processing，但不能直接叫作实际 rollout FLOPs；prefix cache、KV cache 和多轮
interaction 会改变物理计算。为补上这一层，后续 E0-on 已增加 session 级 logical
`rollout_prompt_tokens` / `rollout_output_tokens` 账本，并显式报告覆盖率；字段缺失继续标为
`UNKNOWN`，不得填 0。token 求和只在 `AREAL_TRAINING_PARTICIPATION_LOG` 开启时执行，E0-off
不访问 token payload，避免 matched overhead control 被观测逻辑污染。

R5 在这个 cutoff 内有 6 个 update completion timestamp、5 个 interval：

```text
828.114s, 1289.390s, 759.081s, 384.736s, 624.424s
median = 759.081s
mean = 777.149s
range = 384.736s .. 1289.390s
```

样本少且方差很大，不能替代至少 10 个 post-warmup interval 的 matched steady-state 性能门禁。

### 0.5 Current decision

先完成 `TU-E0a/TU-E0b`，将 generated、manager、workflow、trainer、loss 和 policy gradient 逐 trajectory 连通，并冻结真实 token-version 边界；再用 `TU-E0c/TU-A0` 判断关键路径价值和 matched control。P1 runtime/retry 继续 `HOLD`。

当前最重要的判断是：partial/incomplete recovery 有很强的系统杠杆，但“group 补齐”不等于“样本最终参与训练”。只有恢复后的 cohort 经过 export、workflow filter、trainer consumption，并具有 loss/policy-gradient-active token，才能计为真实回收成功。

## 1. Scope And Definitions

本文使用以下层级，不能互相替代：

```text
result artifact exists
  -> manager rewarded
  -> manager exported complete cohort
  -> workflow structurally eligible
  -> trainer consumed
  -> loss active
  -> gradient active
```

| Layer | Definition |
|---|---|
| Generated result | `results.jsonl` 存在 episode result artifact |
| Manager rewarded | AReaL session 成功接收 final reward |
| Manager exported | 完整 cohort 被 manager 交给 online workflow |
| Structurally eligible | group、reward、tensorization、lineage 和 trainable-token 检查通过 |
| Trainer consumed | trajectory 进入 `prepare_batch -> compute_advantages -> ppo_update` |
| Loss active | 至少一个 token 的最终 `loss_mask` 为真 |
| Gradient active | 至少一个 token 的 effective loss weight/advantage 非零 |

“System sample utilization”描述生成计算有多少转化为 trainer/gradient 输入；“algorithm sample efficiency”描述模型效果随 consumed sample 或 active token 的提升。本文首先解决前者，后者必须通过 checkpoint 下游评测验证。

## 2. bs2 Conservation Ledger

### 2.1 Top-level conservation

```text
2000 generated results
= 960 trainer consumed
 + 128 workflow uniform-reward filtered
 + 798 manager rejected after reward
 +   5 open-at-shutdown rewarded
 + 109 not manager-rewarded by cutoff
```

因此：

```text
2000 - 960 = 1040 aggregate unconsumed
1040 = 128 + 798 + 5 + 109
```

这条等式是同一 `idx-0..249` cohort ledger 上的 aggregate closure。Result episode ID 不连续，目前不能把每一条 `results.jsonl` 记录精确 join 到 manager disposition 和 trainer step。

### 2.2 Manager rejection decomposition

```text
798 manager-rejected rewarded trajectories
= 376 stale complete cohorts
 + 364 partial cohort deadline
 +  43 incomplete cohort
 +  14 sandbox internal exception
 +   1 cohort timeout
```

### 2.3 Full disposition table

| Disposition | Cohorts | Trajectories | Generated share | Current interpretation | Confidence |
|---|---:|---:|---:|---|---|
| Trainer consumed | 120 | 960 | 48.00% | 实际参与训练；reward 正负都视为有价值数据 | `EXACT` |
| Uniform-reward filtered | 16 | 128 | 6.40% | 完整 export，但当前 group-relative advantage 为零 | `EXACT_AGGREGATE` |
| Staleness rejected | 47 | 376 | 18.80% | 8/8 完成且 reward，因 version drift > 2 拒绝 | `EXACT_AGGREGATE` |
| Partial deadline | 54 | 364 rewarded | 18.20% | 缺 68 个 rewarded slot，连带拒绝既有完成成员 | `EXACT_AGGREGATE` |
| Incomplete cohort | 8 | 43 rewarded | 2.15% | 缺 21 个 rewarded slot；至少 2 组有 early-close/409 race | `EXACT_AGGREGATE` |
| Sandbox internal exception | 2 | 14 rewarded | 0.70% | cohort 内 infra-invalid member 导致整组拒绝 | `EXACT_AGGREGATE` |
| Cohort timeout | 1 | 1 rewarded | 0.05% | 其余成员未在 10800s 内形成完整 group | `EXACT_AGGREGATE` |
| Open at shutdown, rewarded | 2 open cohorts | 5 | 0.25% | 窗口末仍 open；若 run 不再恢复才是 terminal waste | `EXACT_AGGREGATE` |
| Not manager-rewarded by cutoff | Mixed | 109 | 5.45% | 包含 pre-session、unfinished、error 等，仍缺逐 ID disposition | `EXACT_AGGREGATE` / category `UNKNOWN` |

## 3. Funnel Metrics

| Conversion | Value | Interpretation |
|---|---:|---|
| Manager rewarded / generated | `1891 / 2000 = 94.55%` | 绝大多数 artifact 对应 reward 完成，但不代表可训练 |
| Manager exported / generated | `1088 / 2000 = 54.40%` | 136 个完整 cohort 进入 workflow |
| Workflow accepted / manager exported | `960 / 1088 = 88.24%` | 其余 128 条属于 uniform-reward group |
| Trainer consumed / generated | `960 / 2000 = 48.00%` | 当前 system sample utilization 的 aggregate proxy |
| Manager rejected after reward / generated | `798 / 2000 = 39.90%` | 最大的已确认完成计算损失 |
| Partial + incomplete rewarded / generated | `407 / 2000 = 20.35%` | 当前最明确的 cohort amplification |

`48.00%` 不能解释为“只有 48% token 有训练价值”。历史 2000 条窗口仍缺 per-disposition token、
wall time、GPU-hour，以及 960 条 consumed 数据的 loss-active/policy-gradient-active count。R5 已在
新的 6 个 update 窗口证明这些层级可逐 UID 精确记录，但不能用 R5 的 96 条结果回填历史 960 条。

## 4. Disposition Analysis Cards

### 4.1 Partial cohort deadline: 364 rewarded

**Observed mechanism**

- 54 个 group 期望 432 个 slot，365 admitted、364 rewarded。
- 67 个原始 member 有 result artifact 和 infra error，但从未进入 `areal_start_session_begin`。
- 另 1 个 admitted member 在 session 内发生 infra failure。
- 46/54 个 group 只缺 1 条，不存在固定 tail-rank 偏置。

**Recovery hypothesis**

Runtime-ready sandbox 加上首次 LLM request 前的 bounded replacement，可以用 68 个成功 member 补齐并保住 364 条已完成 trajectory。理论杠杆为：

```text
364 existing rewarded trajectories / 68 missing-or-failed slots = 5.35
```

**Correctness boundary**

- 同一个 task、prompt、sampling config 和 original group rank。
- Retry 只能发生在首次 LLM request 前，不能按模型 outcome 重采样。
- Replacement 不得比 baseline 引入更大的 actual token-version staleness。
- 补齐后的 group 仍需通过 uniform-reward、tensorization 和 trainer participation 检查。

**Required experiment:** `TU-E1`、`TU-E2`。

### 4.2 Incomplete cohort: 43 rewarded

**Observed mechanism**

8 个 group 共期望 64 个 slot，43 rewarded、21 未 rewarded。已精确定位的 idx 119 和 230 在只有 1 个 admitted slot 时提前关闭，随后各有 6 个 `/rl/start_session` 返回 409；其余 incomplete group 仍需 E0 逐 rank 归因。

**Recovery hypothesis**

完整 cohort manifest、member lease 和 producer-done/abort protocol 可以防止 manager 把“当前 admitted 都结束”误判成“后续不可能补齐”。

**Main risk**

Manifest 如果缺少 lease timeout 或 recovery priority，会把早关闭改成更长的 FIFO head-of-line blocking。

**Required experiment:** `TU-E0a`、`TU-E3`。

### 4.3 Staleness: 376 rewarded

**Observed mechanism**

47 个 group 已经 8/8 完成并 reward，但 cohort drift 为 3/4/5/7，超过 baseline `max_head_offpolicyness=2`。

**Optimization boundary**

不能通过直接接收 stale group 来提高表面利用率。优先验证：

- partial/HOL 修复能否缩短 cohort age，从而间接降低 drift；
- near-complete cohort 是否应获得 admission/recovery priority；
- 更快 update 是否反过来让长 trajectory 更容易跨越多个 version；
- actual token-version、importance ratio、clip fraction 是否仍在正确性边界内。

**Required experiment:** `TU-E4`。它必须在 recovery 实验中同步作为 guardrail，而不是等 recovery 完成后再看。

### 4.4 Uniform-reward filter: 128 trajectories

16 个完整 group 被过滤，其中 11 个 all-1、2 个 all-0，另 3 个具有相同 dense reward。它们不是 infra-invalid；在当前 GRPO group-relative normalization 下，group std 近零，没有直接 gradient signal。

可测试的是 DAPO 风格 dynamic sampling 或 task/data curriculum，不能简单绕过 filter。实验需要同时报告新增 generation cost、active-token goodput、reward composition 和下游效果。

**Required experiment:** `TU-E5`，优先级低于 infra/control-plane recovery。

### 4.5 Sandbox exception and cohort timeout: 15 rewarded

这 3 个 group 属于可能可恢复的 infra-invalid cohort，但当前数量较小。先纳入统一 failure classifier 和 member lifecycle tracing，不单独设计复杂机制；只有在 E1/E2 后仍稳定出现才升级优先级。

### 4.6 Open and not rewarded: 114 trajectories

5 条 rewarded trajectory 位于 run 结束时仍 open 的 cohort，不能在窗口结束时直接记为浪费。109 条未 manager-rewarded artifact 目前只有 aggregate closure，需要 E0 区分：

```text
pre-session terminal failure
session started but unfinished
session ended without reward
manager rejected before reward
queued/open at shutdown
result/manager join mismatch
```

只有确认 run 结束后不再复用，才能标记 `terminal_unconsumed`。

## 5. Recovery Opportunity Ranking

| Priority | Opportunity | Existing trajectories affected | Expected value | Main risk |
|---|---|---:|---|---|
| P0 | End-to-end lineage | 全部 2000 | 让 recovery 可以被证明，而不是依赖 aggregate coincidence | tracing overhead / join bug |
| P0 | Final participation + token-version audit | 960 consumed 和未来 recovered 数据 | 证明数据实际进入 loss/gradient，并冻结现有 off-policy correctness 边界 | packed reorder / CP duplicate / metric semantics |
| P0 | Critical-path replay + matched A0 | 407 partial/incomplete rewarded、89 missing slots | 判断 sample leverage 是否能转化为 update-interval 收益 | counterfactual 假设 / unmatched baseline |
| P1 | Runtime-ready sandbox | 主要覆盖 49 setup + 11 bootstrap failures | 降低 startup failure 和 partial amplification | runtime 行为与 baseline 不等价 |
| P1 | Bounded pre-session replacement | 364 partial rewarded | 少量补采保住已有 group | retry tail、capacity contention、version drift |
| P1 | Incomplete closure protocol | 43 incomplete rewarded | 消除 early close 和 409 cascade | manifest HOL / state leak |
| P1 | Staleness critical path | 376 stale rewarded | 潜在收益最大，但算法边界更强 | off-policy bias |
| P2 | Uniform-reward dynamic sampling | 128 exported | 提高 policy-gradient-active group 比例 | 更多 rollout 计算、样本分布变化 |
| P2 | Shutdown drain/reuse | 5 open + 109 unresolved | 减少 terminal overproduction | 延长 shutdown 或错误复用 stale data |

## 6. Experiment Matrix

每个实验使用 [scorecard template](../templates/experiment_scorecard.md)，并在 `metrics/` 保存同口径 JSON。一次只改变一个主要变量。

| ID | Decision question | Primary change | Required outputs | Acceptance gate | Status |
|---|---|---|---|---|---|
| `TU-E0a` | 能否逐 trajectory 解释 2000 条数据并确认最终训练参与层级？ | Observation-only six-layer lineage | generated -> manager -> workflow -> trainer -> loss/policy-gradient join | Unknown disposition = 0；join completeness = 100%；E2E overhead <= 2% | `FIRST_SIX_UPDATES_PASS / FULL_WINDOW_OVERHEAD_PENDING` |
| `TU-E0b` | 当前 async RL 的真实 version 边界是什么？ | Per-token version/no-EOS observation | manager head drift、token staleness、IS/clip/rejection、recomputed no-EOS | manager head drift <=2；token coverage=100%；no-EOS 可复算 | `FIRST_SIX_UPDATES_VERSION_COVERAGE_PASS / DISTRIBUTION_PENDING` |
| `TU-E0c` | 补齐 partial group 是否能缩短 trainer critical path？ | Historical timeline counterfactual replay | recovered cohorts、incremental members、update interval、capacity/stale feedback | optimistic interval >=5% 或 goodput >=10%；保守场景不回退 | `P0_BLOCKED_BY_E0A` |
| `TU-A0` | 后续 `32x8` candidate 应与谁比较？ | Matched observation-only control | commit/config/task/checkpoint/seed/resource hashes、>=10 intervals | 可复现；tracing overhead <=2% | `P0_BLOCKED_BY_E0A_E0B` |
| `TU-E1` | 预构建 runtime 是否减少 reset failure且不改变环境语义？ | Runtime-ready sandbox only | startup latency、failure class、task/image digest、repo/verifier smoke | 行为等价；无 episode-time install；failure 显著下降 | `P1_HOLD` |
| `TU-E2` | 有界 pre-session retry 是否用更少新增计算保住 partial group？ | Failure-class/time-budgeted reset replacement | attempt、recovered rank、incremental wall/GPU cost、final trainer disposition | 无 outcome resampling；manager head gate 不变；token staleness/IS/clip/rejection 不劣于 A0；recovery goodput 提升 | `P1_HOLD` |
| `TU-E3` | Manifest 是否消除 premature close/409，而不增加 HOL？ | Cohort plan/lease/producer protocol | 409、closure reason、lease age、FIFO head wait、state leak | 409 = 0；unique ranks = 100%；HOL 不回退 > 5% | `P1_HOLD` |
| `TU-E4` | 哪些 wall-clock/HOL 因素导致完整 cohort stale？ | P0 先 tracing/replay，P1 再评估 scheduling candidate | cohort age、actual token versions、drift、IS/clip、update interval | 不放宽 correctness；stale compute/GPU-hour 下降 | `P1_HOLD` |
| `TU-E5` | Dynamic sampling 是否提高 policy-gradient-active goodput且不损害效果？ | Uniform-reward replacement policy | replacement cost、active tokens、reward/task distribution、downstream eval | active-token goodput 提升；checkpoint non-inferior | `DEFERRED_ALGORITHM_AB` |
| `TU-E6` | Run 边界是否产生可避免的 terminal overproduction？ | Drain/reuse policy only | queued/open/terminal disposition、shutdown duration、staleness | terminal unknown = 0；无 stale reuse | `P2_HOLD` |

## 7. Required Metrics Per Experiment

### Data funnel

```text
attempted
generated_result
manager_rewarded
manager_exported
structurally_eligible
trainer_consumed
loss_active
policy_gradient_active
```

### Recovery accounting

```text
recovery_attempts
replacement_successes
cohorts_completed_by_replacement
existing_rewarded_trajectories_salvaged
replacement_trajectories_generated
replacement_wall_time / GPU-hours
final trainer-consumed salvaged trajectories
final policy-gradient-active salvaged trajectories/tokens
```

推荐同时报告：

```text
salvage_leverage
= final_trainer_consumed_existing_members / successful_replacement_members

recovery_goodput
= final_trainer_consumed_recovered_cohorts / incremental_rollout_GPU_hour
```

### Performance and correctness

- Overlap-aware update interval 和 window makespan。
- Rollout exposed wait、time-to-final-required-cohort、ready queue depth。
- Cohort age、wait-after-7th、FIFO blocker time。
- Actual token-version 分布、staleness、importance ratio、clip fraction。
- Duplicate original rank、mixed task/prompt/config、orphan session 和 sandbox leak。
- Reward、turn、context length、termination distribution。
- 下游 checkpoint non-inferiority。

## 8. Cross-run Comparison

| Run | Generated | Trainer consumed | Consumption ratio | Manager-rejected rewarded | Partial/incomplete rewarded | Stale rewarded | Uniform filtered | Unknown disposition | Update interval | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| bs2 C1b v2 | 2000 | 960 | 48.00% | 798 | 407 | 376 | 128 | 109 category-unresolved | 958.79s phase-sum proxy | Diagnostic baseline |
| R5 six-update snapshot | 180 | 96 | 53.33% | 51 terminal unconsumed | 35 partial | 16 | 0 observed in snapshot | 0 trainer participation UNKNOWN；33 rewarded waiting | 759.081s median over 5 intervals | P0 join pass；not matched steady-state |
| TU-E0a | | | | | | | | | | P0 planned |
| TU-E0b | | | | | | | | | | P0 planned |
| TU-A0 | | | | | | | | | | P0 blocked by E0a/E0b |
| TU-E1 | | | | | | | | | | P1 hold |
| TU-E2 | | | | | | | | | | P1 hold |
| TU-E3 | | | | | | | | | | P1 hold |
| Canonical 32x8 | | | | | | | | | | Pending |

不同 run 只有在 logical workload、最大序列长度、总资源、task manifest、sampling 和统计窗口可比时，才能计算相对 delta。bs2 只用于快速归因，不能替代 canonical `32x8` 性能结论。

## 9. Adjacent Costs Not Included In 1040

以下指标重要，但不能塞进 trajectory 守恒等式：

- 55397 个 retry/orphan interactions 是 request/turn-level event，不是 trajectory count。
- 80-turn、max-iteration、loop 或 reward <= 0 的 trajectory 如果已被 trainer 消费，就不是已确认浪费。
- Trainer blocked wait 是 GPU-hour 成本，不能作为 trajectory disposition。
- Full-sequence tokens、loss-active tokens 和 policy-gradient-active tokens 是不同计算层级。

这些成本与本 ledger 通过 ID 和时间线关联，但各自保留正确分母。

## 10. Open Questions

1. 109 条未 manager-rewarded artifact 的逐 ID 最终状态是什么？
2. 798 条 manager-rejected trajectory 分别消耗了多少 token、wall time 和 rollout GPU-hour？
3. 恢复 partial group 后，有多少会因为 uniform reward 或 staleness继续无法训练？
4. Manager cohort snapshot 与 trajectory 内 actual token versions 的关系是什么？
5. 历史 960 条 consumed trajectory 中，loss-active 和 policy-gradient-active 各有多少？R5 新窗口已回答 `96 consumed -> 94 gradient-active + 2 compact-filtered`，但不能外推回历史窗口。
6. Partial/incomplete cohort 在哪些 trainer step 的关键路径上，理论可降低多少 update interval？
7. 更快的 cohort supply 是否让 update 更频繁，从而提高长 trajectory 的 version drift？

## 11. Update Protocol

每次实验完成后必须更新：

1. Owner Review 的当前结论。
2. Full disposition table 和 funnel conversion。
3. 对应 analysis card 的 hypothesis 状态。
4. Experiment Matrix 的结果、scorecard 和 metrics 链接。
5. Cross-run comparison。
6. Open Questions 中已回答和新增的问题。

如果 tracing 只能支持 aggregate closure，继续标记 `EXACT_AGGREGATE`；只有同一个 trajectory ID 贯穿 result、manager、workflow 和 trainer，才升级为 per-trajectory `EXACT`。

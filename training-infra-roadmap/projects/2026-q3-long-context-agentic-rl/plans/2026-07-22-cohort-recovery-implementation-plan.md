# GRPO Cohort Recovery Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 先证明 cohort recovery 能在不改变现有 GRPO/async off-policy 语义的前提下，把 recovered trajectory 真正送入 loss/gradient，并能缩短训练关键路径；证据闭环后，才实施 pre-session recovery。

**Architecture:** P0 是 observation-only/decision phase：建立稳定 logical trajectory identity，把 generated -> manager -> workflow -> trainer -> final loss -> policy-gradient contribution 逐条连通；记录真实 per-token behavior version，复现当前 `max_head_offpolicyness=2` 的算法边界；对历史时间线做 counterfactual replay，并补一条 matched `32x8` control。P1 才包含 runtime-ready sandbox、bounded retry、manifest 和 replacement。第一阶段不修改 GRPO loss 数值、不改变采样/调度、不重复 valid trajectory。

**Tech Stack:** Python 3.11/3.12、asyncio、FastAPI/httpx、pytest/pytest-asyncio、AReaL online proxy、areal-evals bridge、Fuyao sandbox、SwanLab/JSONL tracing。

---

## 0. Scope Decision

当前只执行 **P0**。本文后半部分的 P1 recovery 设计保留为候选实现，但处于 `HOLD`；下面四项 P0 gate 没有全部通过前，不允许开启 retry、replacement、manifest closure 或 production default。

| P0 question | Answer encoded by this plan | Pass evidence |
|---|---|---|
| Recovered 样本是否真的参与训练？ | 不能用 manager `ready/exported` 代替训练参与；必须逐 logical trajectory 记录 `trainer_consumed`、`loss_active` 和 `policy_gradient_active` | 同一 `trajectory_uid` 可贯穿六层；join completeness 100% |
| Replacement 应满足什么 version 规则？ | P0 不采用 `drift=0` 新规则，也不放宽现有规则；manager 继续按 baseline `max_head_offpolicyness=2` 约束 cohort head，trainer 以 per-token `versions` 和现有 rejection/IS 处理真实数据 | manager head drift 越界数为 0；token staleness/IS/clip/rejection 可逐 token 审计 |
| 回收 407 条已 reward 成员是否一定提升性能？ | 不一定；只有补齐时间落在 trainer critical path 上才有收益。先做 timeline counterfactual，再决定是否投入 P1 | optimistic upper bound 有实质收益；保守场景不增加 update interval/staleness |
| 用什么基线声明 `32x8` 性能收益？ | 历史 R8b 只作背景，不是 matched control；先跑同 commit/config/task manifest/checkpoint/seed/resource 的 observation-only A0 | candidate 只能与 A0 做主比较，R8b 不用于归因 |

### P0/P1 boundary

**P0 includes:** identity/lineage、final participation metrics、token-version audit、no-EOS metric correction、offline critical-path replay、matched `32x8` A0、下游评测预注册。

**P1 deferred:** runtime image 发布、固定 retry budget、真实 replacement、producer manifest/lease、partial/incomplete closure、runtime canary、任何 GRPO estimator fallback。

## 1. Owner Decisions

实施前请确认以下默认选择。未修改即按推荐值执行。

| Decision | P0 decision | Reason |
|---|---|---|
| Behavior change | 关闭 | P0 只采集证据，不改变 scheduler、sampling、retry 或 loss |
| Existing async bound | 保持 `max_head_offpolicyness=2` | 这是本次 baseline 已在使用的算法/系统契约；P0 不擅自收紧为 0，也不放宽 |
| Version truth source | `InteractionWithTokenLogpReward.versions` | manager `cohort.rollout_version` 只是 cohort 创建时 snapshot，不能代表长 episode 中每个 token 的 behavior policy |
| Performance baseline | 新建 matched `32x8` A0 | R8b 的代码、任务窗口和 tracing 口径不完全匹配，只能作历史参考 |
| Recovery investment gate | 先通过 offline replay | 407/89 是 sample leverage upper bound，不等于 wall-clock speedup |
| Algorithm fallback | 关闭 | duplicate padding、partial-group estimator 和 dynamic sampling 都属于独立算法 A/B |

P0 明确不做：增加 partial deadline、把 infra failure 填成 reward=0、对 409 重试、按完成速度挑选 first-8、默认复制 valid sample、修改 `max_head_offpolicyness`。

## 2. Baseline And P0 Success Contract

基线使用 `bs2-eqtraj-C1b-v2`：54 个 partial cohort，365 admitted、364 rewarded、67 never-started members。46/54 组只缺 1 条。另有两个 `incomplete` control-plane race，每组 6 个 `/rl/start_session` 返回 409。

### Correctness gates

| Gate | Required result |
|---|---|
| Identity uniqueness | `trajectory_uid` 对 logical member 稳定；retry 使用新的 `attempt_uid`，不能伪装成新训练样本 |
| Trainer boundary | 每个 consumed cohort 恰好 8 个 unique original ranks，集合为 `{0..7}` |
| End-to-end join | generated、manager、workflow、trainer、loss、policy gradient 六层 join completeness = 100%；unknown disposition = 0 |
| Final participation | 每条 trainer-consumed trajectory 都能得到 `loss_active_tokens` 和 `policy_gradient_active_tokens`，缺失不能写成 0 |
| Policy lineage | manager admission/ready/export 的 cohort-head drift 保持 `<=2`；实际 `train_version - behavior_version[token]` 完整记录并沿用现有 rejection/IS，不把 `2` 错当成 per-token hard threshold |
| No behavior drift | P0 的 task order、sampling、scheduler、retry、deadline 和 loss 数值均与 control 一致 |
| Reward/effect | reward 只作 smoke guard；正式 behavior candidate 需通过预注册下游 non-inferiority |
| No duplicate padding | duplicate logical trajectory count 为 0 |

### Performance gates

| Metric | bs2 target | Canonical 32x8 target |
|---|---:|---:|
| Lineage join completeness | 100% | 100% |
| Unknown final disposition | 0 | 0 |
| Tracing overhead | E0-off/on interval delta <= 2% | A0 repeat delta <= 2% |
| Manager cohort-head drift beyond configured bound | 0 | 0 |
| Token-version telemetry coverage | 100% generated response tokens | Same |
| Counterfactual critical-path upper bound | >= 5% update interval or >= 10% consumed-cohort goodput | Same decision rule |
| Observation-only update interval | 不劣于 tracing-off control 2% | 同 task window paired comparison 不劣于 2% |

主效率指标仍是固定 logical workload 和 32 卡资源下的 steady-state update-completion interval。辅助指标是 consumed complete cohorts / rollout GPU-hour，而不是 raw generated results/s。

`407 / 89 = 4.57` 只表示“每新增一个成功缺失 member，理论上最多可保住多少已有 rewarded member”，不代表 `4.57x` 吞吐，也不代表 407 条都会落在当前 update 的最后一个 required cohort 上。

## 3. P0 File Map

Code root: `/Users/zengbw/Codebase/for_agentic_rl/trail-on-main`

| File | Responsibility |
|---|---|
| `areal/experimental/openai/proxy/cohort_manager.py` | 输出 cohort/member identity、manager snapshot version 和 terminal disposition；不改 closure 行为 |
| `areal/experimental/openai/proxy/server.py` | 在 `ReadyCohort` response 中返回 original rank 和 identity |
| `areal/experimental/openai/proxy/workflow.py` | 将 8 个 member 的 lineage sidecar 与 tensor batch 行顺序绑定 |
| `areal/experimental/openai/types.py` | 保持现有 per-token `versions`，定义 observation metadata contract |
| `areal/infra/workflow_executor.py` | 保证 workflow result 到 rollout result 不丢 lineage sidecar |
| `areal/infra/controller/rollout_controller.py` | controller batching 后保留 sidecar 顺序 |
| `areal/trainer/rl_trainer.py` | trainer boundary event、字符串 UID 到 local numeric sequence index 的映射、sidecar 隔离 |
| `areal/utils/data.py` | microbatch 分配时同步 split/reorder `[B]` sequence metadata；不得按 token 复制 ID |
| `areal/engine/megatron_engine.py` | lineage metadata 只进入 loss context，不传给 model forward |
| `areal/trainer/ppo/actor.py` | 依据最终 hard mask、advantage、weight 和 PPO branch 输出 per-sequence participation |
| `tools/analyze_online_cohort_lineage.py` | 生成逐 trajectory ledger 和 aggregate closure |
| `tools/replay_online_cohort_timeline.py` | observation log 的 critical-path counterfactual replay |
| `tests/test_proxy_event_lineage.py` | 六层 join、identity 和 conservation 契约 |
| `tests/test_training_participation_lineage.py` | padded/THD/CP 下的 sequence reorder 和 loss/policy-gradient-active 归属测试 |

Knowledge root: `/Users/zengbw/ReadBase/training-infra-roadmap/projects/2026-q3-long-context-agentic-rl`

代码仓与知识库是两个独立 Git 仓库。下面所有实现、测试和配置提交在 `trail-on-main` 完成；metric contract、实验 scorecard、dashboard 和 STATUS 更新在 `ReadBase` 单独提交。不得用一次 commit 混合两个仓库的变更，也不得为了执行本计划清理两个仓库中已有的无关修改。

| File | Responsibility |
|---|---|
| `cases/bs2-eqtraj-C1b-v2.md` | 现有根因基线和实验结果 |
| `metrics/bs2-eqtraj-C1b-v2.json` | 机器可读 baseline |
| `instrumentation/metric_contract.md` | 新 member/retry/version metrics 定义 |
| `experiments/cohort-recovery-bs2.md` | P0 observation、replay 和后续 P1 实验记录 |
| `experiments/cohort-recovery-32x8-a0.md` | matched `32x8` observation-only control |
| `STATUS.md` | Owner 只读入口和当前 decision |

## Chunk P0-A: Prove End-To-End Training Participation

Execution checkpoint (2026-07-22): 第一阶段代码和 synthetic feedback loop 见 [P0 lineage instrumentation validation](../experiments/2026-07-22_p0-lineage-instrumentation-validation.md)。当前状态是 `CODE_VALIDATED / E0_PENDING`，不是 P0 gate complete；下面未勾选项仍按真实 E0/A0 证据验收。

### P0 Task A1: Define stable logical identity

**Files:**
- Modify: `areal/experimental/openai/proxy/cohort_manager.py`
- Modify: `areal/experimental/openai/proxy/server.py`
- Modify: `areal/experimental/openai/proxy/workflow.py`
- Test: `tests/experimental/openai/test_proxy_gateway.py`

- [ ] **Step 1: Write failing identity tests**

Identity contract:

```text
trajectory_uid = run_id / sample_key / original_group_rank
attempt_uid    = trajectory_uid / attempt
session_id     = one actual backend session
cohort_id      = one manager group instance
```

`trajectory_uid` 在 retry 前后稳定，`attempt_uid` 必须变化；同一 logical member 的失败 attempt 不能被计成新训练 trajectory。测试 out-of-order admission `[6, 1, 7, 0, 4, 2, 5, 3]`，并保留现有 dense `cohort_rank` 行为。

- [ ] **Step 2: Add identity to manager and `ReadyCohort`**

manager terminal event 和 workflow export 必须同时带：

```text
trajectory_uid, attempt_uid, session_id, cohort_id
sample_key, original_group_rank, cohort_rank
manager_rollout_version, disposition, timestamp
```

本任务只增加字段和 event，不改 admission、closure、deadline 或 FIFO。

- [ ] **Step 3: Run focused tests**

```bash
uv run pytest tests/experimental/openai/test_proxy_gateway.py -k "identity or original_group_rank" -vv
```

### P0 Task A2: Carry sequence lineage through batching and microbatch reorder

**Files:**
- Modify: `areal/experimental/openai/proxy/workflow.py`
- Modify: `areal/infra/workflow_executor.py`
- Modify: `areal/infra/controller/rollout_controller.py`
- Modify: `areal/trainer/rl_trainer.py`
- Modify: `areal/utils/data.py`
- Modify: `areal/engine/megatron_engine.py`
- Test: `tests/test_training_participation_lineage.py`

- [ ] **Step 1: Write failing padded/THD reorder tests**

覆盖长度不同、microbatch balance 导致顺序变化、CP8 和 8-way cohort。断言 `forward_indices/backward_indices` 后，每个 sequence 的 numeric lineage index 仍映射到原 `trajectory_uid`。

- [ ] **Step 2: Keep strings out of model tensors**

workflow 返回与 batch row 对齐的 `trajectory_lineage` sidecar。trainer boundary 将 UID 映射为 compact `int64 sequence_lineage_idx[B]`；原始字符串 map 留在 trainer/control plane。

`split_padded_tensor_dict_into_mb_list()` 增加明确的 sequence-metadata 分支：`[B]` metadata 必须按 `group_indices` split/reorder，不能落入当前 `not_to_split` 路径，也不能扩成 `[B, S]` per-token ID。`megatron_engine` 将该 key 保留在 `orig_mb` loss context，并在 model forward 前显式移除。

- [ ] **Step 3: Verify concat and RPC boundaries**

测试 `concat_padded_tensors`、workflow executor、controller 和 actor RPC serialize/deserialize 后的顺序与数量，禁止依靠 list flatten 的偶然行为。

- [ ] **Step 4: Run focused tests**

```bash
uv run pytest tests/test_training_participation_lineage.py tests/experimental/openai/test_proxy_gateway.py -q
```

### P0 Task A3: Emit exact final participation

**Files:**
- Modify: `areal/trainer/rl_trainer.py`
- Modify: `areal/trainer/ppo/actor.py`
- Modify: `areal/utils/functional/functional.py`
- Modify: `areal/utils/stats_tracker.py`
- Test: `tests/test_training_participation_lineage.py`

- [ ] **Step 1: Freeze the metric semantics in tests**

对当前 PPO path，按 sequence 聚合以下 mask：

```text
trainer_consumed: row 进入 compute_advantages -> ppo_update
loss_active: final hard loss mask after M2PO and rejection-mask
policy_gradient_active:
  loss_active
  AND effective_advantage != 0
  AND effective_loss_weight != 0
  AND behave_importance_weight > 0 (when enabled)
  AND policy-ratio derivative is finite and non-zero
  AND NOT clip_mask
  AND NOT dual_clip_mask
```

`policy_gradient_active` 表示 actor objective 对当前 policy logprob 存在直接非零梯度的 token；它不是简单的原始 `loss_mask`。SAPO/KD 等其他 loss path 必须各自提供公式和测试；未实现的 path 输出 `UNKNOWN/unsupported`，不得伪装成 0 或 `EXACT`。

- [ ] **Step 2: Aggregate by `cu_seqlens`**

在 loss 已完成 hard mask、rejection 和 clipping branch 选择后，使用 `cu_seqlens` 把 token mask 聚合到 `sequence_lineage_idx`。CP rank 先 OR/sum，DP shard 产出 compact record，再由唯一 writer 汇总，避免每个 distributed rank 重复计数。

- [ ] **Step 3: Emit one terminal training record per UID**

```text
trajectory_uid, trainer_step, cohort_id
trainer_consumed, full_sequence_tokens
loss_active_tokens, policy_gradient_active_tokens
final_training_disposition
```

要求 conservation：`trainer_consumed = loss_active + fully_masked`，且 active token 数不能超过 response token 数。

- [ ] **Step 4: Add zero-advantage/rejection/clipping/CP tests**

至少覆盖 uniform reward、sequence-level rejection、PPO clip、dual clip、token/equal-trajectory weighting、THD padding 和 CP8 dedup。

- [ ] **Step 5: Run focused tests and overhead microbenchmark**

```bash
uv run pytest tests/test_training_participation_lineage.py -q
```

记录 tracing off/on 的 CPU、GPU memory 和 batch preparation delta；P0 gate 为 update interval overhead <= 2%。

### P0 Task A4: Make the audit reproducible

**Files:**
- Create: `tools/analyze_online_cohort_lineage.py`
- Modify: `tests/test_proxy_event_lineage.py`
- Modify: `training-infra-roadmap/projects/2026-q3-long-context-agentic-rl/instrumentation/metric_contract.md`

- [ ] 将 bs2 临时分析器固化为一个命令，输出逐 UID JSONL、aggregate JSON 和标准 Markdown block。
- [ ] fixture 覆盖 missing rank、409 race、uniform reward、trainer consumed but zero-gradient、shutdown open。
- [ ] 断言不能用 aggregate coincidence 冒充 ID join；无法 join 的记录必须明确列入 `unknown_disposition`。
- [ ] 运行：

```bash
uv run pytest tests/test_proxy_event_lineage.py tests/test_training_participation_lineage.py -q
```

**P0-A gate:** 所有 2000 条 result artifact 都得到明确 disposition；manager/workflow/trainer/loss/policy-gradient 六层 join completeness = 100%，unknown = 0，tracing overhead <= 2%。否则停止。

## Chunk P0-B: Preserve The Existing Algorithm Boundary

### P0 Task B1: Audit actual token versions instead of enforcing manager drift 0

**Files:**
- Modify: `areal/trainer/ppo/actor.py`
- Modify: `tools/analyze_online_cohort_lineage.py`
- Test: `tests/test_training_participation_lineage.py`

- [ ] **Step 1: Add a regression fixture where manager snapshot differs from token versions**

长 episode 可跨权重更新；测试必须证明 `cohort.rollout_version` 不能替代 `InteractionWithTokenLogpReward.versions`。

- [ ] **Step 2: Report two independent version views**

```text
manager_version_drift = current_version - cohort.rollout_version
token_staleness       = train_version - behavior_version[token]
```

按 cohort/trajectory/token 输出 min/p50/p95/max、超界 token 数、rejection fraction、importance ratio 和 clip fraction。

- [ ] **Step 3: Preserve baseline semantics**

P0 使用现有 `max_head_offpolicyness=2` 和现有 rejection/IS 配置。这里的 `2` 约束 cohort head 的 admission/ready/export，不是直接声明每个训练 token 的 staleness 都必须 `<=2`；后者必须从 `versions` 实测并解释 rejection/IS/clip 结果。P0 不新增 `drift=0` gate。是否为 replacement 引入 version lease，留到 P1 的算法/系统联合决策。

### P0 Task B2: Repair correctness guardrails before the run

- [ ] 修复 dynamic padding 下的 no-EOS 统计：以 unpadded response 边界和真实 terminal reason 为准，不能读取 padded tail token。
- [ ] 预注册下游 benchmark、主指标、评测 checkpoint/step 和 non-inferiority 容差；正式 behavior candidate 未通过下游 gate 时不能 promote。
- [ ] 将 task manifest hash、checkpoint hash、sampling config hash、off-policy config 和 code commit 写入 scorecard。

**P0-B gate:** manager cohort-head drift 不越过配置 bound；response token 的 version coverage=100%，staleness/IS/clip/rejection 可闭环；no-EOS 可由原始 response/terminal reason 复算；下游评测契约已冻结。

## Chunk P0-C: Prove Critical-Path Value And Freeze A Matched Control

### P0 Task C1: Replay the observed timeline before implementing recovery

**Files:**
- Create: `tools/replay_online_cohort_timeline.py`
- Create: `tests/test_replay_online_cohort_timeline.py`
- Create: `training-infra-roadmap/projects/2026-q3-long-context-agentic-rl/experiments/cohort-recovery-bs2.md`

- [ ] **Step 1: Reconstruct the factual timeline**

从 observation logs 重建 member planned/admitted/started/rewarded、cohort ready/reject、trainer dequeue/update boundary、policy update 和 in-flight capacity。原始事实与反事实输出必须分栏。

- [ ] **Step 2: Run bounded counterfactuals**

对 89 个 missing slots 使用 `30s / 120s / 300s / observed p50-p95 replacement latency` 场景；保持已有 trajectory duration/reward/task 不变，重放 FIFO、ready queue、trainer demand 和 staleness rejection。

- [ ] **Step 3: Report value and cost together**

至少输出：recovered cohorts、已有 rewarded members salvaged、incremental members、time-to-final-required-cohort、update interval、rollout GPU-hour、capacity contention、stale feedback。不得把 `407/89` 直接当 speedup。

- [ ] **Step 4: Apply the investment gate**

若 optimistic upper bound 仍无法达到 `>=5% update-interval reduction` 或 `>=10% consumed-cohort goodput improvement`，停止 P1 recovery；若 optimistic 通过但保守场景回退，则 P1 必须先解决 capacity/time-budget，而不是直接上固定 3 次 retry。

### P0 Task C2: Run observation-only bs2 E0 and matched `32x8` A0

**Files:**
- Create: `training-infra-roadmap/projects/2026-q3-long-context-agentic-rl/experiments/cohort-recovery-32x8-a0.md`
- Modify: `training-infra-roadmap/projects/2026-q3-long-context-agentic-rl/dashboard.md`
- Modify: `training-infra-roadmap/projects/2026-q3-long-context-agentic-rl/STATUS.md`

- [ ] **E0-off/on:** 同 checkpoint、task order、seed、bs2、group8、128K 和资源，唯一变化是 tracing；验证 join 与 <=2% overhead。
- [ ] **A0:** 用将来 candidate 相同的 code base、baseline behavior flags、task manifest、checkpoint、seed、sampling、32 A100-80GB 和 32x8 logical batch 跑 observation-only control。
- [ ] 主性能口径使用连续 update completion timestamps 和同窗口 makespan；logged phase sum 只作诊断。
- [ ] 至少采集 10 个有效 post-warmup update intervals；不足则标记 diagnostic，不能作为 production performance claim。
- [ ] 后续 candidate 与 A0 使用同 task slices 的 paired window，并报告 bootstrap confidence interval。历史 R8b 只列背景，不计算 recovery 因果 delta。

**P0-C gate:** replay 显示 recovery 有 critical-path 投资价值；A0 的配置/任务/代码/资源均可复现；P0 correctness gates 全过。Owner review 后才能把 P1 从 `HOLD` 改为 `READY`。

## P1 HOLD: Remove Episode-Time Runtime Installation

### Task 3: Audit and prepare runtime-ready task images

**Files:**
- Create: `fuyao_examples/swe_bench_rl/audit_r2e_runtime_images.py`
- Create: `tests/test_r2e_runtime_image_manifest.py`
- Create: `fuyao_examples/swe_bench_rl/eval_qwen35_9b_sftablb_native_qwen3coder_128k_dense_runtime_ready.yaml`

- [ ] **Step 1: Write parser tests for `images.yaml`**

断言 unique image digest 去重、任务到 image 映射保持不变、输出顺序确定、dry-run 不修改源 manifest。

- [ ] **Step 2: Implement image audit command**

每个 unique image 检查：`tmux -V`、`bash`、`git`、Python、`/usr/bin/execd`、OpenHands required runtime。输出：

```json
{"image": "...", "runtime_ready": false, "missing": ["tmux"]}
```

- [ ] **Step 3: Run audit against the actual R2E image manifest**

Artifact target:

```text
/dataset_rc_b1/zengbw1/r2e_gym_subset/runtime_image_audit.jsonl
```

- [ ] **Step 4: Publish derived images**

Preferred path: 按 source image digest 构建 derived tag，预装 `tmux` 和 OpenHands runtime，生成：

```text
/dataset_rc_b1/zengbw1/r2e_gym_subset/images.runtime-ready.yaml
```

如果平台暂不支持批量 derived image，允许使用版本化、只读的 portable runtime bundle，但必须先在 Debian/Ubuntu 两类 canary image 验证 glibc/libevent/ncurses 兼容性。禁止回退到每 episode `apt-get install`。

- [ ] **Step 5: Add fail-fast runtime-ready eval config**

新 YAML 指向 `images.runtime-ready.yaml`，删除 episode-time `apt-get update`，启动 execd 前做 `tmux -V` preflight；缺失时在 30 秒内返回明确 `runtime_preflight_failed`。

- [ ] **Step 6: Run config and manifest tests**

```bash
uv run pytest tests/test_r2e_runtime_image_manifest.py -q
```

- [ ] **Step 7: Run a 16-episode sandbox-only canary**

Acceptance: 16/16 sandbox ready；0 install；0 network bootstrap；无 sandbox leak。

- [ ] **Step 8: Commit without replacing baseline config**

```bash
git add fuyao_examples/swe_bench_rl/audit_r2e_runtime_images.py tests/test_r2e_runtime_image_manifest.py fuyao_examples/swe_bench_rl/eval_qwen35_9b_sftablb_native_qwen3coder_128k_dense_runtime_ready.yaml
git commit -m "feat: add runtime-ready R2E sandbox path"
```

**P1 runtime gate:** runtime-ready canary 未达到 100% 时，不开启 episode reset retry，先修工件。

## P1 HOLD: Bounded Pre-Session Reset Retry

### Task 4: Add an explicit infra retry classifier

**Files:**
- Create: `third_party/areal_evals_bridge/src/areal_evals_bridge/infra_retry.py`
- Modify: `third_party/areal_evals_bridge/tests/test_bridge_helpers.py`

- [ ] **Step 1: Write table-driven failing tests**

Retryable:

```text
sandbox create/ready timeout
HTTP 502/503/504
stale sandbox command HTTP 404 before first LLM request
httpx transport disconnect
runtime preflight failure caused by replaceable sandbox instance
```

Not retryable:

```text
agent max_iterations
verifier failure/reward=0
LLM context overflow
/rl/start_session 409
policy-version mismatch
```

- [ ] **Step 2: Implement `classify_pre_session_failure(exc)`**

返回 stable enum、retryable bool 和 sanitized reason。遍历 exception cause chain，不依赖单一错误字符串。

- [ ] **Step 3: Run classifier tests**

```bash
PYTHONPATH=third_party/areal_evals_bridge/src uv run pytest third_party/areal_evals_bridge/tests/test_bridge_helpers.py -k infra_retry -vv
```

- [ ] **Step 4: Commit classifier**

```bash
git add third_party/areal_evals_bridge/src/areal_evals_bridge/infra_retry.py third_party/areal_evals_bridge/tests/test_bridge_helpers.py
git commit -m "feat: classify pre-session infrastructure failures"
```

### Task 5: Retry reset with sandbox cleanup and the same original rank

**Files:**
- Modify: `third_party/areal_evals_bridge/src/areal_evals_bridge/env_wrapper.py`
- Modify: `third_party/areal_evals_bridge/src/areal_evals_bridge/ctx_timing.py`
- Modify: `third_party/areal_evals_bridge/src/areal_evals_bridge/runner.py`
- Modify: `third_party/areal_evals_bridge/tests/test_bridge_helpers.py`
- Modify: `fuyao_examples/swe_bench_rl/eval_qwen35_9b_sftablb_native_qwen3coder_128k_dense_runtime_ready.yaml`

- [ ] **Step 1: Inspect the installed `ResetWorker` and sandbox cleanup contract on Fuyao**

必须确认失败的 `inner.reset(ctx)` 后，调用哪一个 lifecycle API 能释放旧 sandbox；将结论写进代码注释和测试 fake。若无法安全 cleanup，同一 env object 不得直接二次 reset。

- [ ] **Step 2: Write failing async tests**

覆盖 fail-once-success、fail-twice-success、fatal-no-retry、cleanup failure、shutdown cancellation。成功路径必须只注册一个 AReaL session，并保持同一 `cohort_key/original_rank`。

- [ ] **Step 3: Add opt-in retry config**

```yaml
areal_infra_reset_retry:
  enabled: true
  max_attempts: 3
  backoff_seconds: [2, 5]
```

默认关闭；只在 runtime-ready canary 配置开启。

- [ ] **Step 4: Implement bounded retry**

每次失败顺序固定为：classify -> trace -> cleanup old sandbox -> backoff -> recreate/reset。达到预算后保留原始异常作为 result error。

- [ ] **Step 5: Emit attempt lifecycle**

```text
member_reset_begin
member_reset_failed
member_reset_cleanup_end
member_reset_retry_scheduled
member_reset_succeeded
```

每条必须带 `ctx_id/cohort_key/original_rank/attempt/failure_class`。

- [ ] **Step 6: Run bridge suite**

```bash
PYTHONPATH=third_party/areal_evals_bridge/src uv run pytest third_party/areal_evals_bridge/tests/test_bridge_helpers.py -q
```

- [ ] **Step 7: Commit reset retry**

```bash
git add third_party/areal_evals_bridge/src/areal_evals_bridge fuyao_examples/swe_bench_rl/eval_qwen35_9b_sftablb_native_qwen3coder_128k_dense_runtime_ready.yaml
git commit -m "feat: retry retryable sandbox resets before rollout"
```

**P1 retry gate:** 本地 failure-injection 测试必须证明 1/8 和 2/8 reset failure 最终仍形成 8 个 unique members，且没有重复 backend session。固定 `max_attempts=3` 只是初始测试参数，不能在 P0 replay 给出 tail/capacity 证据前成为 production default。

## P1 HOLD: Cohort Manifest And Closure Correctness

### Task 6: Add planned-member state without reserving inference capacity

**Files:**
- Modify: `areal/experimental/openai/proxy/server.py`
- Modify: `areal/experimental/openai/proxy/cohort_manager.py`
- Modify: `areal/experimental/openai/proxy/proxy_gateway.py`
- Modify: `areal/api/cli_args.py`
- Test: `tests/experimental/openai/test_proxy_gateway.py`

- [ ] **Step 1: Write manifest state-machine tests**

状态：

```text
planned -> leased -> started -> rewarded/ended
planned -> retryable_failed -> planned(next attempt)
planned/started -> terminal_failed
```

第一条 `plan_member` 创建 `expected_ranks={0..7}`，但不增加 worker load。version snapshot/lease 语义不在 manifest PR 内自行决定，必须复用 P0-B 冻结的 version contract。

- [ ] **Step 2: Add control-plane endpoints**

```text
POST /rl/plan_member
POST /rl/report_member_failure
```

Request identity：`task_id + cohort_key + original_group_rank + attempt`。重复相同 request 幂等；同 rank 并发不同 attempt 返回 conflict。

- [ ] **Step 3: Replace list-only membership with original-rank index**

保留现有 dense backend rank 以降低改动面，但 authoritative membership 使用 original rank。Ready 前断言 key set 恰好 `{0..7}`。

- [ ] **Step 4: Fix incomplete closure**

“当前 admitted slots 全 ended”不再等于“cohort 不可能补齐”。只有以下任一成立才允许 terminal reject：

```text
any member terminal_failed and no fallback enabled
all retry budgets exhausted
partial/cohort/session deadline reached
explicit producer_done with missing members
```

- [ ] **Step 5: Preserve the frozen version rule**

不能用 `current_version == cohort.rollout_version` 替换 baseline 语义。默认 candidate 是 `baseline_bounded_async`：manager 不绕过 `max_head_offpolicyness=2` 的 cohort-head gate，trainer 继续以 per-token `versions` 和现有 rejection/IS 处理。candidate 的 token staleness、IS、clip 和 rejection tail 必须不劣于 matched A0；不能把 head bound 误用成 token hard bound。若要实现严格 same-policy cohort，必须先提供 backend version lease/旧权重 serving，再作为独立策略 A/B，不能只比较 manager snapshot。

- [ ] **Step 6: Add regression tests for idx 119/230 race**

模拟一个 member 先 started+ended，其余 6 个稍后 admission，另一个 retrying。断言 cohort 保持 open、0 个 409，最终 ready 8/8。

- [ ] **Step 7: Run proxy suites**

```bash
uv run pytest tests/experimental/openai/test_proxy_gateway.py tests/experimental/openai/test_proxy_integration.py -q
```

- [ ] **Step 8: Commit manifest change**

```bash
git add areal/experimental/openai/proxy areal/api/cli_args.py tests/experimental/openai
git commit -m "feat: track planned members in online cohorts"
```

### Task 7: Connect bridge reset attempts to the manifest

**Files:**
- Modify: `third_party/areal_evals_bridge/src/areal_evals_bridge/session_manager.py`
- Modify: `third_party/areal_evals_bridge/src/areal_evals_bridge/env_wrapper.py`
- Modify: `third_party/areal_evals_bridge/tests/test_bridge_helpers.py`

- [ ] **Step 1: Write protocol tests**

断言 reset 前调用 `plan_member`；每个失败 attempt 调 `report_member_failure`；成功 attempt 的 `start_session` 携带同一个 attempt；最终失败标记 terminal。

- [ ] **Step 2: Add transient retry to control-plane transport**

`plan/report/start` 可以重试 transport error 和 5xx；409 由状态机解释，不做盲重试。所有 retry 复用同一个 idempotency identity。

- [ ] **Step 3: Run bridge and proxy integration tests**

```bash
PYTHONPATH=third_party/areal_evals_bridge/src uv run pytest third_party/areal_evals_bridge/tests/test_bridge_helpers.py tests/experimental/openai/test_proxy_integration.py -q
```

- [ ] **Step 4: Commit bridge-manifest integration**

```bash
git add third_party/areal_evals_bridge
git commit -m "feat: register eval members before sandbox reset"
```

**P1 manifest gate:** synthetic 1/8、2/8 transient failure 和 fast-finish race 全部得到 8 unique ranks；permanent failure 能快速、可解释地 reject，不等待 900s 才知道原因。

## P1 HOLD: Controlled Fuyao Experiments

### Task 8: Run one-variable-at-a-time bs2 behavior experiments

本任务在 `ReadBase` 仓库记录结果；Fuyao job 使用当阶段已冻结并通过测试的 `trail-on-main` commit。

**Files:**
- Modify: `training-infra-roadmap/projects/2026-q3-long-context-agentic-rl/experiments/cohort-recovery-bs2.md`
- Modify: `training-infra-roadmap/projects/2026-q3-long-context-agentic-rl/dashboard.md`
- Modify: `training-infra-roadmap/projects/2026-q3-long-context-agentic-rl/STATUS.md`

- [ ] **P1-E1: Runtime-ready sandbox only**

只替换 task image/runtime config。比较 pre-session failure、sandbox startup p50/p95、partial groups、step interval。

- [ ] **P1-E2: Runtime-ready + bounded reset retry**

只新增 reset retry。报告每类 attempts、replacement success、sandbox leak、retry wall cost。

- [ ] **P1-E3: Manifest + closure fix**

在 P1-E2 上开启 manifest。重点验证 409=0、unique rank=100%、manager head drift 越界=0、token staleness/IS/clip/rejection 不劣于 A0、partial <=1%。

- [ ] **Step 5: Produce the standard scorecard for every run**

每次必须使用同一模板，并额外报告：

```text
planned -> reset -> session -> rewarded funnel
attempt distribution
original-rank completeness
version drift
recovered existing trajectories
extra sandbox/rollout cost
consumed cohorts per rollout GPU-hour
overlap-aware update interval
```

- [ ] **Step 6: Apply rollback gates**

任一条件立即关闭新 behavior flag：duplicate rank >0、manager head drift 越界、replacement token staleness/IS/clip/rejection 显著劣于 A0、retry 超过 time/capacity budget、sandbox leak >0、409 增加、update interval 回退 >5%。

### Task 9: Candidate `32x8` validation against matched A0

- [ ] **Step 1: Freeze P1-E3 code/config and run 32x8 on 32 A100-80GB**

不同时修改 cache、WIP、OFP、session capacity 或 trainer topology。

- [ ] **Step 2: Compare against matched A0 with the canonical dashboard**

主指标：steady-state update interval。必须同时给出 reward、trajectory length、termination、per-token staleness、loss-active/policy-gradient-active count。R8b 只保留为历史背景。

- [ ] **Step 3: Run downstream evaluation at the pre-registered checkpoint**

Infra 指标通过但下游 non-inferiority 不通过时，不进入生产默认。

- [ ] **Step 4: Decide**

```text
Promote: correctness gates 全过，partial <=1%，下游不退化，update interval 改善 >=10%
Observe: correctness全过但性能改善 <10%
Rollback: 任一 correctness/downstream gate 失败
```

## P1/P2 HOLD: Deferred Algorithm Fallback

只有 P1-E3 后 partial 仍 >1%，并确认剩余原因无法通过真实 replacement 解决，才另开设计和计划。

候选为 GLM-5 风格 valid-member padding，但必须满足：valid unique members >4；statistics 基于 unique members；padding duplicate 使用 `1/repeat_count` 权重；记录 effective group size；不与 runtime/manifest PR 同时上线。需要独立短训和下游 A/B。

SAO/single-rollout 属于下一季度算法和系统联合课题，不进入本计划。

## P0 Final Verification

- [ ] Run P0-A/P0-B focused unit/integration suites listed above.
- [ ] Run `ruff check` on all P0 modified Python files.
- [ ] Parse every emitted metrics JSON.
- [ ] Verify Markdown links in ReadBase.
- [ ] Confirm git diff contains no scheduler/retry/deadline/loss behavior change and no unrelated changes.
- [ ] Record exact code commit、config hash、task manifest hash、checkpoint、seed、off-policy config 和 Fuyao job ID。
- [ ] Review E0 lineage/overhead、counterfactual replay、matched A0 和下游 preregistration as one P0 decision packet.
- [ ] Keep every P1 behavior flag disabled until owner changes P1 state from `HOLD` to `READY`.

## Expected Delivery Sequence

| Delivery | Estimated engineering time | Review gate |
|---|---:|---|
| P0 PR1 logical identity + six-layer lineage | 2-3 days | Per-UID join and reorder tests |
| P0 PR2 final participation + token-version audit | 2-4 days | Exact loss/policy-gradient-active and staleness tests |
| P0 replay tool + bs2 E0 | 1-2 days plus run time | 100% closure、<=2% overhead、critical-path investment gate |
| P0 matched `32x8` A0 | 1 run plus queue time | Reproducible observation control and >=10 post-warmup intervals |
| P0 owner decision packet | 0.5 day | `STOP` or authorize P1 |
| P1 runtime/retry/manifest | Deferred | Only after P0 authorization |

P0 预计 1-2 周，主要受 final participation instrumentation、CP8 验证和 Fuyao 排队影响。P0 的交付物不是“已经回收更多样本”，而是一个可以可靠回答 **回收后是否进入梯度、是否保持算法边界、是否值得占用关键路径资源** 的 decision packet。P1 工期在 P0 通过后重新估算。

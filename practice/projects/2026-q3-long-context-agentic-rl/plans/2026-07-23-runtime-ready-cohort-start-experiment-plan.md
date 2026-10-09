# Runtime-ready And Cohort Start Alignment Experiment Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不放宽 GRPO freshness、不改变采样和 loss 的前提下，验证 deterministic sandbox runtime 与 8-way group-aligned reset 能否提高训练样本利用率并缩短 steady-state update interval。

**Architecture:** 先冻结正在运行的 R5，只把它作为 root-cause evidence，不进入性能估计。随后在同一个 telemetry commit、同一份固定 512 条 task manifest 上依次运行 fresh A0、E1、E2、fresh A1；E1 只移除 episode-time network bootstrap，使用版本化 portable tmux runtime 和 fail-fast preflight，E2 在 E1 上只把 sandbox reset concurrency 从 4 提升到 8。A0/A1、E1/E1r 分别是相同 immutable source/config 的 bracket replay，所有运行保持模型、任务顺序、资源、WIP、OFP、group size、sampling 和 trainer topology 一致。

**Tech Stack:** Python 3.12、pytest、AReaL online proxy、areal-evals bridge、Fuyao sandbox、JSONL lineage、32 x A100-80GB。

---

## 0. Repository And Artifact Roots

Code worktree:

```text
/private/tmp/trail-p0-cohort-lineage
branch: codex/p0-cohort-lineage
base commit: cf1fc168fb5fb45da4e7890c49d9114b69a3fe12
```

Knowledge root:

```text
/Users/zengbw/ReadBase/training-infra-roadmap/projects/2026-q3-long-context-agentic-rl
```

R5 deployed source is a source-only snapshot without `.git`:

```text
/workspace/zengbw1@xiaopeng.com/code_r2e_p0_lineage_r5_20260723_133229/source
```

Before candidate deployment, all P0 code must pass the frozen related suite and be committed. E1/E2 use immutable commits and a generated source manifest containing relative path, size and SHA256. The deployed manifest must match the local commit artifact before launch.

## 1. Root-cause Evidence

R5 run:

```text
run_dir:
  /dataset_rc_b1/zengbw1/log/areal_r2e_gym_qwen35_9b/128k_bifrost-2026070622090200-zengbw1/
  experiments_zbw_128k_nativeqwen3coder_d1t2c4p1_bs2_g8_mb131072_p0lineage/
  20260723_135011
```

冻结 `global_step=5 -> version=6` 的事实：

```text
223 admitted -> 180 generated/rewarded -> 96 trainer consumed
generated but not consumed:
  partial 35
  stale 16
  waiting 33
```

五个 partial cohort 各缺一个 original rank，缺失 ctx 为 `22/35/51/60/96`。五条都已经 acquire sandbox，但都在首次 LLM 请求和 AReaL admission 前失败：

```text
RuntimeError: tmux is required but unavailable; install attempts failed
```

R5 后续窗口中，387 次 sandbox acquire 对应 12 个独立 tmux bootstrap failure，约 3.1%。`images.yaml` 有 4578 个 task image，逐 image 派生重建不适合作为第一轮验证方案。

两个 stale cohort 的精确时间线：

| Cohort | Rollout version | First admission -> reject | Admission spread | Reject-time version | Drift |
|---|---:|---:|---:|---:|---:|
| idx-17 | 2 | 28m52.9s | 3m19.5s | 5 | 3 |
| idx-22 | 3 | 17m37.4s | 4m57.9s | 6 | 3 |

`idx-22` 在 version 6 update 完成后约 1.47 秒才形成 ready cohort；若仍处于 version 5，drift=2，本可 export。当前 `group_size=8`，但 `worker_concurrency.reset_sandbox=4`，因此 group member 至少分两波 reset，是 admission spread 的可验证系统因素。

## 2. Frozen Matched Configuration

所有 candidate 必须保持：

```text
resources: 32 x A100-80GB
model: Qwen3.5-9B
logical batch: bs2 x group8
context: 128K
max response: 8192
rollout: vLLM d16t1
trainer: Megatron d1t2c8p1
max_concurrent_rollouts: 48
max_head_offpolicyness: 2
online_worker_capacity: 6 per vLLM worker, 96 total
eval concurrency: 128
max_active_sandboxes: 128
deploy_concurrency: 24
sampling: temperature=1.0, top_p=0.95, top_k=50
task order and seed: same as R5
initial checkpoint/model/tokenizer: same immutable SHA
resume: disabled; every run starts fresh from the same checkpoint
```

不得同时改变：prefix cache、WIP、OFP、partial/cohort deadline、group size、reward、agent max iterations、sampling、loss、trainer topology。

## 3. Fixed Workload And Statistical Contract

### Closed cohort window

主比较只使用同一组 logical cohorts：

```text
cohort_key: r2e_gym_train:1:idx-0 .. idx-63
episodes: ctx 0 .. 511
planned logical trajectories: 64 x 8 = 512
```

Candidate eval dataset 只包含前 64 个 source entry，每个 entry 由 `group_size=8` 展开为 8 个 original rank，共 512 个 episode。用 deterministic 生成工具产出 manifest，记录原始 `tasks.jsonl` SHA、原始 suite SHA、筛选规则、64 个 task ID、512 个 `(task_id, original_rank, ctx_id)` 和 manifest SHA；A0/E1/E2/A1/E1r 必须使用同一 manifest SHA。最后一个 episode 完成 reset 调度后停止新 admission，并 drain 到所有 64 个 cohort 进入以下闭合集合：

```text
trainer_consumed
uniform_reward_filtered
partial
stale
failed/rejected
open_at_drain_timeout
```

`rewarded_waiting_cohort` 和其他 open 状态在 drain 结束前不得计作浪费。Drain timeout 使用 baseline `online_cohort_timeout_seconds=10800` 加 600 秒观察余量；到期仍 open 的 cohort 单列，不强行归因。

### Run order and matched estimator

实验分为机制验证和性能确认：

```text
A0 = fresh baseline, fixed 512 tasks, current runtime bootstrap, reset concurrency 4
E1 = runtime-ready, reset concurrency 4
E2 = runtime-ready, reset concurrency 8
A1 = fresh baseline replay, byte-identical A0 source/config/task manifest
E1r = E1 replay, byte-identical E1 source/config/task manifest,
      only when E2 needs promotion-level confirmation
```

Historical R5 只用于确认 tmux failure、partial/stale 路径和 telemetry schema；不得通过过滤 `idx-0..63` 把它当 A0，因为同一进程后续 producer workload 会继续争用 rollout、sandbox 和 trainer 资源。

正式运行顺序为 `A0 -> E1 -> E2 -> A1`。A0/A1 使用同一个 baseline behavioral-config SHA，E1/E1r 使用同一个 E1 behavioral-config SHA；hash 计算前只规范化 trial name、output/run ID 等 launch metadata，其他字段必须逐字段相同。若 E2 进入 Promote 候选，再运行 E1r，形成 `E1 -> E2 -> E1r` bracket。

每个 run 的时间坐标使用其 steady-state update window 的 midpoint。严格为正的性能指标（update interval、makespan、rollout GPU-hour、goodput 的正值变换）在 log 空间做预注册的时间加权插值：

```text
log B_hat(t) =
  (1 - alpha) * log(metric_A0) + alpha * log(metric_A1)
alpha =
  (t - t_A0) / (t_A1 - t_A0)

log E1_hat(t_E2) =
  (1 - beta) * log(metric_E1) + beta * log(metric_E1r)
beta =
  (t_E2 - t_E1) / (t_E1r - t_E1)
```

估计器同时报告：

```text
E1 runtime-ready effect:
  metric_E1 / B_hat(t_E1)

E2 reset8 incremental effect:
  metric_E2 / E1_hat(t_E2)  # promotion decision
  metric_E2 / metric_E1     # diagnostic when E1r not run

baseline platform drift:
  metric_A1 / metric_A0
```

Fraction/count 指标不用 log 插值。对于 stale 等 fraction，以 `count / 64 planned cohorts` 在 probability natural scale 按同一 midpoint 时间权重插值；两侧均为 0 时 comparator 明确定义为 0，只有一侧为 0 时仍按 natural-scale interpolation 计算。

若 `A1/A0` 的 update interval 或 rollout GPU-hour 漂移超过 10%，性能结论降级为 `NO_CONCLUSION`，机制指标仍可报告。Moving-block/cluster bootstrap 每次 resample 都必须重算 run metric、相应 scale 的时间插值和最终 effect，不能先计算 point estimate 再对结果加误差。分析器必须用 synthetic fixtures 验证 bracket pairing、非对称时间坐标、run/config SHA equality、estimator 方向、zero/one-sided fraction comparator 和每次 resample 重算。

### Update interval window

排除 step 0 和其后的前两个 update intervals。主区间使用后续最多 20 个连续 successful weight-update completion intervals；少于 10 个只报告机制结果，10-19 个标记 diagnostic，20 个才允许 performance decision。

主性能指标：

```text
steady_state_update_interval =
  consecutive successful weight-update completion timestamp delta
```

同时报告固定 cohort window 的：

```text
time_to_10_updates
time_to_20_updates
window makespan
mean/p50/p95 update interval
```

比较使用 ratio-of-means 和 moving-block bootstrap（block length=3，10000 resamples）的 95% CI。CI 跨过 0 improvement 时为 `NO_CONCLUSION`，不能按点估计宣布收益。

样本效率指标：

```text
terminal trainer-consumed trajectories / 512 planned trajectories
complete consumed cohorts / 64 planned cohorts
complete consumed cohorts / rollout GPU-hour
partial trajectories and cohorts
stale trajectories and cohorts
policy_gradient_active_tokens / rollout response tokens
open_at_drain_timeout
```

机制指标：

```text
runtime bootstrap failure rate
sandbox startup p50/p95
cohort admission span p50/p95
first admission -> cohort ready p50/p95
policy drift at ready/export
```

正确性 guard：

```text
8 unique original ranks per consumed cohort
duplicate logical trajectory = 0
manager drift beyond OFP bound = 0
unknown disposition = 0
sandbox leak = 0
token version coverage = 100%
token staleness p50/p95/max
behavior-IS ratio and normalized ESS summary
optimizer PPO ratio summary
clip and rejection fraction
fully-masked and zero-gradient fraction
```

候选相对时间插值 matched baseline 的算法 guard：

```text
reward mean non-inferiority margin: -0.05 absolute
response-token p95 upper bound: 1.10x baseline
token staleness p95: no more than baseline + 0.25 version
clip fraction: no more than baseline + 2 percentage points
rejection fraction: no more than baseline + 2 percentage points
normalized ESS: at least 0.95x baseline
fully-masked/zero-gradient fraction: no more than baseline + 2 percentage points
```

所有 64 个 planned cohort 都进入 disposition/selection estimand；partial、stale、failed、filtered、consumed 和 open 均保留，candidate-only recovered cohort 不得因 baseline 中没有 consumed pair 而被删除。样本利用率和 terminal disposition 直接在完整 64-cohort universe 上比较。

Reward、response length、behavior-IS ESS、clip/rejection 和 fully-masked/zero-gradient 等算法 guard 的主分析，对每个 run 的全部 consumed cohort 做 independent cohort-cluster bootstrap，保留 cohort 内 8 条 trajectory/token 的相关性，并包含 candidate-only recovered consumed cohorts。相同 `cohort_key` 的 consumed intersection 另做 paired sensitivity analysis，但不能替代主分析。两种估计差异必须连同每个 missing pair 的 terminal disposition/reason 报告；存在 open、unknown 或无法解释的 metric missingness 时拒绝算法 non-inferiority 结论。

Reward margin、normalized behavior-IS ESS 下界、response-token p95、clip/rejection 和 fully-masked/zero-gradient fraction 上界均使用单侧 95% CI；只有完整置信界满足预注册 margin 才算通过。ESS 在每个 bootstrap resample 内从 `sum(w)`、`sum(w^2)` 和 `N` 重新计算。`duplicate=0`、`unknown=0`、8 unique ranks、version coverage 100% 和 leak=0 继续使用 exact hard gate。

若指标在当前 PPO path 不可获得，必须先补 telemetry，不能把缺失值当作通过。

## Chunk 1: Freeze R5 Evidence And Build Fresh Baseline

### Task 1: Produce the immutable R5 diagnostic scorecard

**Files:**
- Create: `fuyao_examples/swe_bench_rl/train_qwen35_9b_sftablb_native_qwen3coder_128k_dense_cp8_bs2_long.yaml`
- Verify: `fuyao_examples/swe_bench_rl/eval_qwen35_9b_sftablb_native_qwen3coder_128k_dense_amortized.yaml`
- Verify: `fuyao_examples/fuyao_deploy_bash/fuyao_online_run.sh`
- Create: `training-infra-roadmap/projects/2026-q3-long-context-agentic-rl/experiments/cohort-recovery-bs2.md`
- Modify: `training-infra-roadmap/projects/2026-q3-long-context-agentic-rl/dashboard.md`
- Modify: `training-infra-roadmap/projects/2026-q3-long-context-agentic-rl/STATUS.md`

- [ ] Label R5 `historical_diagnostic_only`; do not classify it as A0 or use it in the performance estimator.
- [ ] Record source path, config path, run directory, lineage sidecar and eval logs.
- [ ] Reconstruct the deployed bs2 train config in the code worktree and verify a normalized semantic diff against the deployed copy.
- [ ] Record the eval config and launcher that produced `.fuyao_run/evals_patched_config.yaml`.
- [ ] Record the exact deployed train/eval/launcher paths and generate a deployed source SHA256 manifest.
- [ ] Preserve the frozen version-6 and latest closed R5 lineage snapshots as root-cause fixtures.
- [ ] Run `tools/analyze_online_cohort_lineage.py` against the final sidecar.
- [ ] Export token-version/IS/clip telemetry availability and closed disposition counts as diagnostic evidence.
- [ ] Record 12 unique tmux bootstrap failures separately from duplicated traceback lines.
- [ ] Stop R5 only after all logs and process state are captured.
- [ ] Clean R5-created sandbox jobs and verify zero live leftovers.

### Task 2: Freeze one telemetry commit and one fixed 512-task workload

**Files:**
- Create: `fuyao_examples/r2e_rl/build_fixed_task_manifest.py`
- Create: `tests/test_build_fixed_task_manifest.py`
- Create: `fuyao_examples/r2e_rl/build_behavioral_manifest.py`
- Create: `tests/test_build_behavioral_manifest.py`
- Create: `fuyao_examples/r2e_rl/fixed_workloads/r2e_gym_train_512.manifest.json`
- Create: `fuyao_examples/swe_bench_rl/eval_qwen35_9b_sftablb_native_qwen3coder_128k_dense_fixed512_baseline.yaml`
- Create: `tests/test_r2e_fixed_window_config.py`

- [ ] Write failing tests for deterministic task selection, 64 unique source task IDs, exactly 512 expanded episodes, eight unique original ranks per cohort, source-data/suite SHA and manifest SHA.
- [ ] Generate the fixed workload from the same R5 task order without random sampling or NFS mutation.
- [ ] Create a fixed suite with `num_episodes: 64`; prove the evals expansion yields only ctx `0..511`, then let the producer naturally stop and enter an explicit drain phase.
- [ ] Add an exact-diff test proving baseline fixed512 differs from reconstructed R5 only in output/trial fields, fixed-task manifest and finite producer/drain control.
- [ ] Define and test a canonical `behavioral_manifest.v1` containing:

```text
source commit and full source manifest SHA
normalized train config and final patched eval config
behavior-affecting launcher args and environment
initial checkpoint/model/tokenizer path, size and SHA
fresh-start/resume=false assertion and RNG seeds
suite/tasks/images file SHA
64 selected task IDs and resolved task image digests
fixed task manifest SHA
all concurrency, timeout, sampling, loss, reward, WIP and OFP fields
evals package source/wheel SHA and applied patch manifest SHA
runtime mode plus startup-command SHA or runtime-bundle SHA
GPU topology and requested resource class
```

- [ ] The only normalized exclusions are explicit launch metadata: dynamic ports, credentials, PID, wall-clock start time, run ID, trial name and output directory. Unknown fields are included by default; there is no broad prefix-based exclusion.
- [ ] Record both the full behavioral manifest SHA and a comparison-domain SHA. A0/A1 must match in every behavior field; E1 may differ only in runtime mode/artifact/startup command; E2 may additionally differ only in `worker_concurrency.reset_sandbox`.
- [ ] Before launch, assert checkpoint/model/tokenizer SHAs match and no resume state, optimizer state or prior run directory is loaded.
- [ ] Freeze one telemetry commit only after Tasks 1, 5 and 6 tests pass; generate its source manifest.
- [ ] Require A0/A1/E1/E2/E1r source manifest SHA equality. Candidate behavior is selected only by eval config/runtime artifact, not by different code.
- [ ] Run fresh A0 before E1 and fresh A1 after E2. Require A0/A1 baseline behavioral-config SHA equality after launch-metadata normalization.

## Chunk 2: E1 Runtime-ready Sandbox

### Task 3: Build a deterministic portable tmux runtime

**Files:**
- Create: `fuyao_examples/r2e_rl/build_portable_tmux_runtime.py`
- Create: `tests/test_portable_tmux_runtime.py`
- Create: `fuyao_examples/r2e_rl/audit_runtime_image_abi.py`

- [ ] Write failing tests for deterministic manifest, dependency deduplication, missing dependency rejection, atomic publish, SHA verification and wrapper generation.
- [ ] Run:

```bash
uv run pytest tests/test_portable_tmux_runtime.py -vv
```

Expected: FAIL because the builder does not exist.

- [ ] Inventory unique image digests and classify architecture, distro, glibc, `/dev/pts`, `/tmp` and socket compatibility.
- [ ] Implement a builder that copies `tmux`, its resolved ELF dependencies and required terminfo into a staging directory.
- [ ] Generate a wrapper that invokes the bundled loader with a command-local `--library-path`; it must not export global `LD_LIBRARY_PATH`.
- [ ] Emit `runtime_manifest.json` with source paths, SHA256, bundle version, build OS/glibc/arch and supported ABI classes.
- [ ] Atomically rename the verified staging directory into the final versioned path.
- [ ] Set fixed owner and non-writable bundle permissions; startup verifies manifest SHA before use.
- [ ] Run the focused test again and require PASS.
- [ ] Build on the Fuyao training pod into:

```text
/workspace/training_common/sandbox/runtime/tmux-portable-v1
```

- [ ] Verify the bundle and its parent runtime-version directory cannot be modified through the normal candidate startup path.

### Task 4: Add E1 candidate config

**Files:**
- Create: `fuyao_examples/swe_bench_rl/eval_qwen35_9b_sftablb_native_qwen3coder_128k_dense_runtime_ready.yaml`
- Create: `fuyao_examples/r2e_rl/evals_suites_runtime_ready/r2e_gym_train.yaml`
- Test: `tests/test_r2e_runtime_ready_config.py`

- [ ] Write a failing config test requiring:

```text
same fixed512 dataset/task images/sampling/concurrency as A0
no apt-get update/install in sandbox startup
manifest SHA verification and command-local portable runtime wrapper
tmux -V and real create/send/capture/kill preflight before execd
explicit runtime_preflight_failed exit
reset_sandbox remains 4
retry is absent/disabled
```

- [ ] Implement the candidate YAML without modifying the R5 baseline YAML.
- [ ] Add an exact-diff test that allowlists only runtime suite/config changes relative to the fixed512 A0 config.
- [ ] Run the config test and the existing eval config tests.

### Task 5: Complete trainer-side correctness telemetry

**Files:**
- Modify: `areal/trainer/ppo/actor.py`
- Modify: `areal/utils/training_participation.py`
- Modify: `tests/test_training_participation_lineage.py`
- Modify: `tools/analyze_online_cohort_lineage.py`

- [ ] Write failing tests for token staleness p50/p95/max, behavior-IS and optimizer-PPO ratio first/second moments, normalized behavior-IS ESS, PPO clip fraction, rejection fraction, fully-masked trajectory fraction and zero-policy-gradient fraction.
- [ ] Emit a structured per-update sidecar after distributed reduction; do not estimate quantiles by parsing rounded console text.
- [ ] Use reducible counts/histograms or exact bounded version buckets for token staleness p50/p95; require 100% token-version coverage.
- [ ] Define freshness weight exactly as `w_behavior = π_proximal / π_behavior` (`behave_imp_weight`), never `π_theta / π_proximal` (`importance_weight`).
- [ ] Compute normalized behavior-IS ESS as `sum(w_behavior)^2 / (N * sum(w_behavior^2))` over post-rejection, pre-PPO-clip valid response tokens. This fixed population is the freshness guard population.
- [ ] Emit optimizer PPO ratio (`π_theta / π_proximal`) separately for optimization diagnostics; it must not satisfy the freshness ESS gate.
- [ ] Keep behavior-IS, optimizer PPO ratio, clip and rejection field names/populations/denominators explicit and stable across all runs.
- [ ] Join per-update telemetry to `trainer_step`, `train_version`, source manifest SHA and run ID.
- [ ] Add parity tests between structured aggregate output and the existing `behave_mask`, rejection mask, PPO clip mask and participation/actor masks, including a case where behavior-IS ESS degrades while optimizer PPO ESS stays near 1.
- [ ] Run the focused actor/participation suite and require the same telemetry implementation for A0/A1/E1/E2/E1r.

### Task 6: Add queue-boundary tracing, cleanup tooling and a closed-window analyzer

**Files:**
- Create: `tools/analyze_r2e_perf_experiment.py`
- Create: `tests/test_analyze_r2e_perf_experiment.py`
- Create: `tools/cleanup_run_sandboxes.py`
- Create: `tests/test_cleanup_run_sandboxes.py`
- Create: `third_party/patches/evals_sandbox_job_run_id.py`
- Create: `third_party/patches/evals_sandbox_reaper_manager_events.py`
- Modify: `third_party/patches/apply_evals_patches.py`
- Create: `third_party/areal_evals_bridge/src/areal_evals_bridge/reset_timing.py`
- Modify: `third_party/areal_evals_bridge/src/areal_evals_bridge/ctx_timing.py`
- Modify: `third_party/areal_evals_bridge/src/areal_evals_bridge/env_wrapper.py`
- Modify: `third_party/areal_evals_bridge/src/areal_evals_bridge/runner.py`
- Modify: `third_party/areal_evals_bridge/tests/test_bridge_helpers.py`

- [ ] Freeze the installed evals package/version and inspect its `Orchestrator.run -> setup -> _enqueue_initial -> _create_worker_tasks` ordering plus `BaseWorker.in_q.get` anchor.
- [ ] Add a bridge-owned Orchestrator subclass whose `setup()` calls upstream `setup()` and then replaces each still-empty reset queue with a tracing `asyncio.Queue`. Its overridden `put_nowait` is the actual `_enqueue_initial` boundary and its overridden `get` is the actual `BaseWorker` dequeue boundary.
- [ ] Emit `reset_enqueued` at queue `put_nowait` and `reset_dequeued` after queue `get`, retaining the trusted monotonic enqueue timestamp in a queue-owned `ctx_id` map. Assert the queue is empty when installed and every dequeue has exactly one enqueue.
- [ ] Emit `reset_started/reset_finished/reset_failed` around the actual reset execution, with `ctx_id/cohort_key/original_rank`, run ID and wall/monotonic timestamps.
- [ ] Add upstream-order/empty-queue/dequeue-parity tests that fail loudly if evals changes queue creation or run ordering. `env_wrapper.reset()` is only the execution start/finish boundary, never the enqueue boundary.
- [ ] Write failing analyzer fixtures for fixed-window censoring, duplicated traceback deduplication, clock alignment, cohort/rank join, queue/dequeue/reset timing, update interval extraction, non-symmetric A0/A1 and E1/E1r time-interpolated estimators, per-resample estimator recomputation, moving-block CI and cohort-clustered one-sided algorithm-guard CI.
- [ ] Parse lineage JSONL, ctx timing, sandbox jobs, evals log and areal log into one experiment scorecard.
- [ ] Assert conservation over 512 planned logical trajectories and refuse performance classification with open/unknown records.
- [ ] Assert source manifest, task manifest and matched behavioral-config SHAs before comparing runs; only launch metadata may be normalized away.
- [ ] Verify `sandbox_jobs.jsonl` is created in every run output directory and contains an append-only acquired/released lifecycle for each exact sandbox job name.
- [ ] Patch `SandboxJobRegistry` to include `AREAL_RUN_ID` in every record. Patch the upstream orphan reaper to recognize the actually emitted `manager.acquire/manager.release` events as well as legacy `lease.acquire/lease.release.end`; add idempotent anchor tests.
- [ ] Implement `cleanup_run_sandboxes.py` with dry-run default, explicit `--sandbox-jobs-jsonl`, exact recorded-name allowlist, idempotent release/delete and post-cleanup zero-live verification. It must reject missing, malformed or cross-run records.
- [ ] Test cleanup with acquired-only, already-released, duplicate, malformed and foreign-run fixtures; broad prefix/project cleanup is forbidden.
- [ ] Emit machine-readable JSON plus the canonical Markdown block.
- [ ] Run focused analyzer and bridge tests.

### Task 7: Run A0, E1 and their runtime mechanism checks

- [ ] Before each run, verify code source manifest SHA, fixed-task manifest SHA, config SHA and runtime artifact SHA against the launch record.
- [ ] Launch fresh A0 with baseline bootstrap and reset concurrency 4; close/drain the fixed 512-task window and clean only its recorded sandboxes.
- [ ] Canary every observed ABI class and at least 16 task images, including all five frozen-window failure images.
- [ ] Require real create/send/capture/kill success, 0 network bootstrap, 0 tmux failure, manifest SHA match and 0 leak.
- [ ] If canary fails, stop and fix the runtime artifact; do not launch training.
- [ ] Launch E1 with the frozen matched configuration.
- [ ] Restrict the producer to ctx `0..511`, then drain the fixed cohort window.
- [ ] Record E1's provisional mechanism result immediately; defer its final performance classification until A1 closes:

```text
PASS:
  tmux bootstrap failure = 0
  no correctness regression
  partial caused by pre-session runtime = 0
FAIL:
  any runtime incompatibility, duplicate rank, leak or correctness failure
```

E1 alone can establish a sample-readiness mechanism fix. It is not promoted as a performance optimization unless the final `metric_E1 / B_hat(t_E1)` bracket-bootstrap CI shows update-interval or consumed-cohort-goodput improvement without breaking the algorithm guards; point estimates alone cannot support promotion.

## Chunk 3: E2 Group-aligned Reset

### Task 8: Add the reset-concurrency candidate

**Files:**
- Create: `fuyao_examples/swe_bench_rl/eval_qwen35_9b_sftablb_native_qwen3coder_128k_dense_runtime_ready_reset8.yaml`
- Modify: `tests/test_r2e_runtime_ready_config.py`

- [ ] Write a failing test that E2 differs from E1 only in:

```yaml
worker_concurrency:
  reset_sandbox: 8
```

- [ ] Add the E2 YAML and pass the exact-diff test.
- [ ] Assert retry remains absent/disabled in both E1 and E2.
- [ ] Inspect and freeze the actual evals `ResetWorker` implementation/version; do not assume a global semaphore creates cohort alignment.
- [ ] Run a multi-cohort load canary near WIP=48 with reset concurrency 8.
- [ ] Prove from `reset_enqueued/started/finished` events that observed concurrency reaches 8 and report same-cohort rank aggregation.
- [ ] Inspect reset queue wait, sandbox create QPS, 429/5xx/create-timeout, ready p95, active sandbox peak and cleanup.
- [ ] Launch E2 only if the load canary has no new capacity/error pattern.

### Task 9: Run E2, A1, optional E1r and decide optimization value

- [ ] Launch E2 only after E1 is frozen and targeted cleanup reaches zero live jobs.
- [ ] Launch fresh A1 after E2 with the exact A0 source/config/task manifest SHAs.
- [ ] If E2 reaches the Promote mechanism gates, launch E1r with the exact E1 source/config/task manifest SHAs.
- [ ] Compare E1 against the log-space A0/A1 interpolation at `t_E1`, and E2 against the log-space E1/E1r interpolation at `t_E2` when E1r exists, on the same closed cohort window.
- [ ] Report `A1/A0` platform drift and refuse a performance decision when it exceeds 10%.
- [ ] Require cohort admission span p95 improvement >=30%.
- [ ] Apply the pre-registered E2 stale gate without increasing partial/runtime failure:

```text
if interpolated E1 stale fraction > 0:
  Promote requires one-sided 95% upper CI of
  (E2 stale fraction - interpolated E1 stale fraction) < 0
if interpolated E1 stale fraction == 0:
  Promote requires E2 stale cohorts == 0
```

这里的 interpolated E1 stale fraction 使用 natural-scale `stale_count / 64` 时间插值，并在每个 cohort-cluster bootstrap resample 内重算；不使用 log 插值。
- [ ] Mark:

```text
Promote:
  all correctness/coverage gates pass
  admission-span mechanism gate passes
  pre-registered stale gate passes
  95% CI lower bound shows >=5% interval improvement
  point estimate improves >=10%
Observe:
  correctness and mechanism pass
  CI shows positive improvement but below Promote threshold
No conclusion:
  correctness passes but CI crosses zero
Mechanism rejected:
  reset concurrency does not improve admission span >=30%
Safety rollback:
  any correctness, capacity or leak gate fails
Performance rollback:
  bracketed 95% CI shows interval regression
```

## Chunk 4: Conditional E3 Retry

E3 remains `HOLD`. E1 and E2 source/config tests must prove retry is disabled. E3 may be designed only after E1/E2/A1 are frozen and Owner separately approves a new branch, plan, config and trial.

The observation trigger is residual retryable pre-session failures divided by terminal pre-session attempts above 0.5%; it does not authorize implementation by itself.

Initial retry contract:

```text
pre-session infra failure only
cleanup old sandbox before retry
same cohort key and original rank
one retry
no retry for 409, context overflow, verifier/reward failure or started rollout
```

Retry may consume capacity and worsen staleness, so it cannot be bundled into E1 or E2.

## 4. Deployment And Rollback

Before each Fuyao launch, show the exact source commit/manifest, config, trial name, resource request, fixed task manifest and single changed variable. Reuse the existing machine only after the preceding run is frozen and sandbox cleanup is verified.

Each launch record must include the exact start/stop command, launcher PID, trainer PID, eval PID, run directory and lineage sidecar. Stop sequence is trainer/eval graceful termination, bounded wait, then force-kill only remaining matching PIDs. Cleanup is restricted to sandbox job names recorded by that run's `sandbox_jobs.jsonl`; broad project-wide deletion is forbidden. After cleanup, verify Ray actors, GPU processes, CPU eval workers and recorded sandbox jobs are zero before the next launch.

Runtime bundles are immutable artifacts and are not deleted during rollback. Rollback restores the baseline eval config and leaves E1/E2 behavior flags/configs unused.

Immediate rollback triggers:

```text
duplicate original rank > 0
unknown disposition > 0
manager drift beyond configured bound > 0
sandbox leak > 0
new 409 pattern
runtime incompatibility on any canary image
```

The triggers above are immediate safety rollback conditions. Performance regression is evaluated only after the closed window and bracketed CI are complete; noisy intermediate intervals never trigger destructive rollback by themselves.

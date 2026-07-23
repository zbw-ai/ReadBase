# P0 R5 Lineage Closure Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 R4 暴露的 `RTensor` lineage boundary 与 workflow event propagation，使 R5 完成首个 optimizer update 和逐 UID trainer participation 闭环。

**Architecture:** controller 只读取 `RTensor` 的 meta shape，不 materialize 远端 tensor；trace identity 通过 scheduling spec 传播到 actor 和 rollout workers。每个子问题独立 RED/GREEN，R5 不启用任何 P1 recovery behavior。

**Tech Stack:** Python、PyTorch、AReaL single-controller、RTensor、Ray、pytest、Fuyao。

---

## Chunk 1: P0.1 RTensor-Aware Row Alignment

### Task 1: Reproduce the R4 trainer-boundary failure

**Files:**
- Modify: `/private/tmp/trail-p0-cohort-lineage/tests/test_training_participation_lineage.py`
- Test: `/private/tmp/trail-p0-cohort-lineage/tests/test_training_participation_lineage.py`

- [x] Add a test constructing an `RTensor` whose meta tensor has shape `[8, S]`, plus eight lineage records.
- [x] Use a backend that raises on fetch; assert detach and subsequent attach both preserve eight rows without fetching.
- [x] Add an `RTensor` `cu_seqlens` case and assert batch size is `numel - 1`.
- [x] Run the single test and verify RED with `batch size 0`.

### Task 2: Add metadata-only RTensor batch-size support

**Files:**
- Modify: `/private/tmp/trail-p0-cohort-lineage/areal/utils/data.py`
- Test: `/private/tmp/trail-p0-cohort-lineage/tests/test_training_participation_lineage.py`

- [x] Teach `get_batch_size()` to recognize `RTensor.data.shape[0]` through a local import.
- [x] Keep tensor precedence `attention_mask -> cu_seqlens -> multimodal -> first tensor-like`.
- [x] Run the new test and existing trainer-boundary tests; verify GREEN.
- [x] Run targeted Ruff and `git diff --check`.

## Chunk 2: P0.2 Workflow Event Propagation

### Task 3: Reproduce missing rollout worker environment

**Files:**
- Modify: `/private/tmp/trail-p0-cohort-lineage/tests/test_training_participation_lineage.py`
- Modify: `/private/tmp/trail-p0-cohort-lineage/areal/trainer/rl_trainer.py`

- [x] Extend the existing propagation test with separate actor and rollout scheduling specs.
- [x] Assert both specs receive identical absolute `AREAL_TRAINING_PARTICIPATION_LOG` and `AREAL_RUN_ID`.
- [x] Add tracing-disabled and non-single-controller tests proving actor/rollout specs and parent environment remain unchanged.
- [x] Run the test and verify RED because rollout env vars are absent.
- [x] Update `_amend_training_participation_envvar()` to mutate both spec sets without changing configs when tracing is disabled.
- [x] Run the targeted test and surrounding lineage/proxy suites; verify GREEN.

## Chunk 3: P0.3 Runtime Hygiene

### Task 4: Close R4 and classify the residual process

**Files:**
- Update: `../experiments/2026-07-22_p0-lineage-instrumentation-validation.md`

- [x] Record final R4 funnel: `91 -> 55 -> 16 -> 0 -> 0`.
- [x] Capture launcher/trainer/eval parent-child relationships.
- [x] Stop the orphan wrapper/eval/tee process tree.
- [x] Clean only R4-created sandboxes. Owner-authorized targeted attempt completed: `248 attempted / 0 killed / 248 failed` because the IDs were not cancellable by the current identity; do not bypass ownership.
- [x] Confirm Ray returns to zero training GPU allocation.
- [x] Classify launcher without changing code: R4 used an older remote script hash that lacked the current cleanup logic, so there is no controlled reproduction against the reviewed launcher.

## Chunk 4: Verification and R5

### Task 5: Run code gates

- [x] Run P0.1/P0.2 tests in the Fuyao training environment.
- [x] Run proxy/workflow/lineage suites and redistribution tests.
- [x] Run Ruff, compile, and `git diff --check`.
- [x] Review the diff for observation-only equivalence.

### Task 6: Launch R5 after configuration confirmation

- [x] Present frozen R5 config: bs2, group size 8, 128K, 16 actor + 16 rollout GPUs, `max_head_offpolicyness=2`, tree training disabled. Owner confirmed on 2026-07-23.
- [x] Freeze the R4 task manifest/order, seed, checkpoint, sampling parameters and code source state; record hashes.
- [x] Generate unique R5 identity `p0-lineage-bs2-r5-20260723_134723` and a sidecar path that did not exist before launch; launcher fails closed on namespace reuse.
- [x] Deploy the exact reviewed source state to a new remote checkout; verify all 1130 source files, frozen input hashes, launcher hash, and 36 lineage tests.
- [x] Start one detached launcher and verify a unique trainer/eval process tree; G0-G2 pass with 16/16 vLLM backends and fresh sandbox/agent traffic.
- [x] Keep the stop guard active for traceback, UID conflict, partial lineage coverage, or training numerical-path change; none triggered before the first update.

### Task 7: Close G4-G5

- [x] Wait for two complete cohorts and the first optimizer update. Step 0 consumed `idx-0` and out-of-order-completed `idx-3`, then finished weight update to version 1.
- [x] Run strict analyzer against the live sidecar.
- [x] Require `trainer_consumed = 16`, and require those 16 UIDs to be exact subsets of both manager-exported and workflow-exported UIDs. Do not require aggregate equality: asynchronous prefetch had already exported `idx-5`, so the valid live aggregate was `24 / 24 / 16`.
- [x] Verify `policy_gradient_active <= loss_active <= response <= full_sequence` trajectory by trajectory.
- [x] Require behavior-version coverage exactly 100% and UNKNOWN participation exactly 0.
- [x] Fail R5 on any UID join error, token conservation violation, version coverage gap, UNKNOWN row, or cross-run event. No failure condition was observed for step 0.
- [x] Keep P1 on `HOLD` after G4-G5; matched tracing-overhead and observation-only live gates remain before any recovery behavior is enabled.

### R5 G4-G5 Result

| Gate | Result |
|---|---|
| Optimizer updates observed | `PASS`；审计截止到 `global_step=5 -> version=6`，共 6 个完成 update |
| Trainer logical batch | 6 个 update 共 `96` trajectories；每步 `2 cohorts x 8` |
| Strict UID join | `PASS`；trainer rows `96`，`EXACT=96`，missing/conflict `0` |
| Async upstream state | admitted `223`，generated/rewarded `180/180`，manager/workflow/trainer `96/96/96` |
| Token conservation | full sequence `4073826`，response/loss-active/policy-gradient-active 均为 `518340` |
| Participation | `94` policy-gradient-active；`2` trainer-consumed fully masked |
| Terminal unconsumed | `35` partial（5 个 7/8 cohort）+ `16` stale（2 个完整 cohort）；另有 `33` rewarded waiting |
| Version audit | coverage `1.0`，UNKNOWN `0` |
| Observed update intervals | 5 个；median `759.081s`，mean `777.149s`，range `384.736-1289.390s`；不是 matched post-warmup 性能结论 |
| Runtime errors | 未发现 Traceback、RuntimeError、empty batch/trajectory 或 CUDA OOM |

两个 fully-masked 样本的 full-sequence 分别为 `28258` 和 `131072`，合计 `159330`，占这 96 条
trainer full-sequence tokens 的 `3.91%`；response/loss-active/policy-gradient-active 都为 0。它证明
P0 能区分“被 trainer 消费”和“真正产生梯度”，不能在 P0 observation-only 阶段擅自过滤。这里的
full-sequence tokens 是 trainer processing 口径，不等于 prefix-cache/KV-cache 复用后的实际 rollout GPU work。

审计截止时 5 个 partial cohort 均为 7/8 后 `partial_timeout`：`idx-2/4/6/7/12` 分别缺 source
rank `6/3/3/4/0`。两个 stale cohort `idx-17/22` 都已 8/8 完成，rollout version 分别为 2/3。
当前 R5 没有 rollout token-work 字段，因此这 51 条 terminal-unconsumed trajectory 的 token 成本仍为
`UNKNOWN`。

### Task 8: Close the fully-masked cause

- [x] Correlate the trajectory timestamps with eval logs. Its `reward_recorded` timestamp is exactly `2026-07-23 14:21:22`, matching agent `exit=-1` and `signal: terminated`; the bridge assigned zero reward and compact-filtered the loss.
- [x] Add a bounded `compact_filter_reason_code`; raw free-form reason is not written to lineage. Known codes cover signal termination, AgentState.ERROR, max iterations and loop detector; other text collapses to `agent_failed:other` or `other`.
- [x] Propagate the bounded code through workflow export and trainer lineage without changing reward, advantage, loss mask or optimizer behavior.
- [x] Refine analyzer output to `trainer_consumed_compact_filtered` while retaining `final_training_disposition=trainer_consumed_fully_masked`.
- [x] Verify RED against the unmodified R5 snapshot, then GREEN with `8 passed`; run full proxy/lineage regression `65 passed`, Ruff, compile and `git diff --check`.
- [ ] Validate the new code in the next E0-on live sidecar. Do not restart the healthy R5 solely for this observation field.

### Task 9: Account for rollout work before trainer consumption

- [x] Add RED tests showing backend `end_session` lacks logical prompt/output token counts, gateway lineage drops the fields, and analyzer cannot report terminal-unconsumed token work.
- [x] Sum all recorded completion payload prompt/output tokens at rollout-server session close.
- [x] Propagate only non-negative exact integers through gateway and manager; missing or malformed observation fields fail open without changing session success, while analyzer requires the prompt/output pair for a valid audit.
- [x] Make analyzer fail closed on one-sided or negative counts and report generated/terminal-unconsumed count coverage plus logical prompt/output tokens.
- [x] Document that these are logical model-response token counts, not physical GPU FLOPs after APC/KV reuse and not trainer packed full-sequence tokens.
- [x] Add a non-UTF8 backend-body regression: observed RED `UnicodeDecodeError`, then GREEN by ignoring malformed observation payloads.
- [x] Bind token summation to `AREAL_TRAINING_PARTICIPATION_LOG`. The tracing-off RED proved token access still occurred; GREEN proves no token access and no token-work response fields when disabled.
- [x] Run the complete related regression: `143 passed` after the final tracing-switch test, plus Ruff, Python compile and `git diff --check`.
- [ ] Validate token-work coverage in the next E0-on live sidecar. Current R5 source was not hot-modified, so its 51 terminal partial/stale trajectories still have token cost `UNKNOWN`.

# P0 Lineage Instrumentation Validation

Status: `R4_FAILED_AT_TRAINER_BOUNDARY / P0.1_P0.2_CODE_GATES_PASS / R5_SIX_UPDATES_G3_G5_PASS / G6_PENDING`

Date: 2026-07-22 (updated 2026-07-23)

这份记录只回答一个问题：P0 的 observation-only tracing 是否已经具备进入真实训练实验的代码基础。它不宣称 cohort recovery 已带来性能收益，也不把 synthetic fixture 当成 `2000 -> 960 -> 1040` 的真实闭环。

## Scope

| Item | Value |
|---|---|
| Code repository | AReaL isolated worktree `/private/tmp/trail-p0-cohort-lineage` |
| Branch | `codex/p0-cohort-lineage` |
| Base commit | `cf1fc168fb5f` |
| Live job usage | R3 因 tracing 改变训练语义而停止；R4 因 `RTensor` row-alignment bug 在首个 trainer batch 失败；R5 已完成 6 个 optimizer update 的 G3-G5 strict lineage 闭环 |
| Behavior change | P0 必须 observation-only；P1 retry/replacement 未启用 |
| Trace switch | `AREAL_TRAINING_PARTICIPATION_LOG=<shared-jsonl-path>` |

## Implemented P0 Slice

1. manager admission 增加稳定 `trajectory_uid`、`attempt_uid`、原始 group rank 和 dense cohort rank；out-of-order admission 不再破坏 logical identity。
2. workflow 将 lineage 作为 row-aligned sidecar 送到 trainer；trainer 将字符串映射为 compact `int64 sequence_lineage_idx[B]`。
3. THD/BSHD microbatch reorder 显式重排 sequence metadata；model forward 前移除 metadata，loss context 保留。
4. PPO loss 在 hard mask、rejection、loss weight、advantage、PPO clip 和 dual clip 之后输出 `loss_active` 与 `policy_gradient_active`。
5. SAPO/KD 未实现精确 derivative semantics 时输出 `UNKNOWN/unsupported`，不能伪装成 0 或 `EXACT`。
6. actual token `versions` 用于 version coverage 和 staleness min/max；manager snapshot 不替代 token behavior version。
7. terminal reason 仅作为 lineage 观测字段，不进入 reward injection、GAE bootstrap 或其他训练数值路径。
8. analyzer 对 UID map 与 participation 做严格一一 join；缺失、冲突或空事件流直接失败。
9. lineage map 与 loss event 共用唯一 writer 选择：每个 DP replica 只允许 PP last stage、TP0、CP0 写入，避免非 loss rank 产生孤立 map。
10. manager、workflow 与 trainer 的 observation event 写入全部 fail-open；离线 analyzer 保持 fail-closed。写失败不会改变 admission、reward、recovery 或 optimizer 控制流。
11. manager 与 workflow 补齐 admission、started、reward、export、filter 和 terminal disposition；trainer-consumed 记录必须具备同一 `run_id + trajectory_uid + attempt_uid` 的完整前置闭环。
12. sidecar 权限固定为 `0600`，禁止 credential 字段；`admission_rejected` 只记录 bounded `rejection_code`，不写自由文本 message。
13. compact-filtered trajectory 只写 bounded `compact_filter_reason_code`，不把 agent 的自由文本 failure reason 写入 sidecar；analyzer 将 causal funnel disposition 与 trainer fully-masked training disposition 分开保留。

## Verification Evidence

### Regression suite

Fuyao job `bifrost-2026070622090200-zengbw1` 的 rank 0，使用远端已有 Python 环境和隔离 worktree：

初版曾通过 `200 passed in 14.54s`，但后续代码审查发现测试没有覆盖 observation-only 等价性，因此该结果不能作为启动依据。最终修正版在远端真实运行环境中的门禁为：

```text
118 passed in 13.47s  # lineage + proxy + workflow
62 passed in 9.72s    # PPO + rejection + Ray launcher + workflow detection
3 passed in 54.55s    # redistribution on 2/4/8 GPU
64 passed in 8.01s    # final sensitive-reason delta rerun; overlaps first suite
ruff: All checks passed!
```

覆盖：lineage/participation、proxy gateway、proxy event lineage、online trajectory GRPO MVP、rejection sampling、PPO stats 和 data utilities。新增门禁均先观察 RED，再实现 GREEN：

| Failure mode | RED evidence | GREEN condition |
|---|---|---|
| 非 loss model-parallel rank 只写 map、不写 participation | strict join 会出现 missing participation | map 与 loss callback 共用 topology writer |
| 空日志被误判为 0 unknown | empty stream 返回空 summary | empty stream 直接 `ValueError` |
| benchmark 固定 off->on 顺序产生 cache bias | delta 可出现无意义负数 | 多轮交替顺序，使用 paired median |
| tracing terminal reason 改变 reward/GAE | tracing on/off 训练语义不等价 | 恢复基线 no-EOS 逻辑，terminal reason 只写 sidecar |
| version 统计与 next-token loss 错位 | staleness 落在错误 token | 只在观测副本中逐序列 shift，禁止修改训练输入 |
| 全 mask microbatch 跳过 loss callback | trainer-consumed UID 缺 participation | loss 前注册所有序列，缺 callback 时精确输出 0/0 |
| SAPO/KD UNKNOWN 被累计成精确 0 | aggregate 产生伪精确结论 | UNKNOWN 的 loss/gradient token 为 `null`，排除出 exact aggregate |

代码静态检查：修改文件 `ruff check` 通过，`git diff --check` 通过，Python compile 通过。`test_serialization.py` 未计入门禁：其 collection 会访问 Hugging Face 下载 Qwen2.5-0.5B，而训练节点网络拒绝该下载；该失败不在本次修改路径，但仍作为未执行测试保留。

R4 暴露问题修复后的 2026-07-23 增量门禁：

```text
36 passed in 10.90s  # lineage；包含 RTensor metadata-only 与 actor/rollout env propagation
64 passed in 7.87s   # proxy gateway
23 passed in 9.53s   # workflow/cohort GRPO
80 passed in 9.94s   # data utilities + eval dispatch
1 passed in 5.84s    # Ray HTTP launcher
3 passed in 49.92s   # redistribution on 2/4/8 GPU
ruff: All checks passed!
compileall: PASS
git diff --check: PASS
```

共计 207 条测试。P0.1 的 RED 是 `RTensor` batch size 被误判为 0；P0.2 的 RED 是 rollout spec 缺少 participation log。修复后还补齐了 tracing disabled 和 non-single-controller 的无副作用契约测试，并通过独立规格审查与代码质量审查。

R5 闭环后新增 compact-filter cause 和 upstream rollout token-work instrumentation。测试先观察
bounded reason import RED、backend/gateway/analyzer 缺字段 RED，以及 non-UTF8 body 的
`UnicodeDecodeError` RED，再在隔离 snapshot 中通过：

```text
8 passed   # bounded reason + causal disposition focused tests
65 passed  # complete proxy workflow + lineage analyzer files
143 passed # final rollout server + gateway/manager + workflow + lineage regression
ruff: All checks passed!
py_compile: PASS
git diff --check: PASS
```

新字段在 session close 时记录 logical prompt/output token work，并报告 terminal-unconsumed coverage；
它不等于 APC/KV reuse 后的物理 GPU FLOPs，也不等于 trainer packed full-sequence tokens。正在运行的
R5 source 没有热改；这些 observation field 留到 matched E0-on 做 live 验证。token 求和也已绑定
到 `AREAL_TRAINING_PARTICIPATION_LOG`：E0-off 不访问 token payload、不返回 token-work 字段，避免
control 组承担这部分 instrumentation 开销。

### Analyzer CLI smoke

synthetic exact-lineage fixture 的严格 join 输出：

| Metric | Value |
|---|---:|
| Trainer-consumed trajectories | 2 |
| Full-sequence tokens | 224 |
| Response tokens | 56 |
| Loss-active tokens | 31 |
| Policy-gradient-active tokens | 20 |
| Behavior-version coverage | 100% |
| UNKNOWN participation | 0 |

这些数字只证明公式、join 和输出 schema 可工作，不是训练 case 的真实性能数据。

### CPU diagnostic microbenchmark

配置：`B=8`、`S=4096`、100 iterations x 7 alternating rounds。

| Metric | Paired median |
|---|---:|
| Mask collection off | 0.477 ms/batch |
| Mask collection on | 1.161 ms/batch |
| Incremental mask cost | 0.713 ms/batch |
| Collector + JSONL flush | 0.695 ms/batch |
| Writer cost | 86.9 us/trajectory |

这是 CPU 小张量 diagnostic，不能代替 GPU update interval，也不能用于宣称满足 `<=2%` tracing overhead gate。它的作用是提前排除同步逐 token I/O 等数量级错误。

## Live Integration Canary

本次运行验证了修改代码可在真实 32 卡链路启动并把 trajectory 送到 agent rollout，但代码审查发现 tracing 改变了 reward/GAE 语义，因此在首个 optimizer update 前主动停止。R3 是失败 canary，不得作为 E0-on 或性能结果。

| Item | Value |
|---|---|
| Fuyao job | `bifrost-2026070622090200-zengbw1` |
| Trial | `trial_oh_bs32_eqtraj_lr5e7_S64_WIP40_P0LINEAGE_R3` |
| Deployed code | `/workspace/zengbw1@xiaopeng.com/code_r2e_p0_lineage_20260722` |
| Run directory | `/dataset_rc_b1/zengbw1/log/areal_r2e_gym_qwen35_9b/128k_bifrost-2026070622090200-zengbw1/experiments_zbw_128k_nativeqwen3coder_d1t2c4p1_bs32_g8_mb131072/20260723_072555` |
| Participation sidecar | `.../p0_lineage/20260723_070100_training_participation.jsonl` |
| Resource state | 训练进程已停止；Ray 为 `0/32 GPU`、`0/320 CPU` 使用，可用于修复后重启 |
| Rollout serving | 16 backend ready；gateway `10.1.48.29:46659` 已启动并被 eval runner 发现 |
| Environment execution | 36,624 episodes 入队；sandbox 已创建、通过 health check、安装 OpenHands 并开始多轮 agent request |
| Stop reason | tracing terminal reason 被错误接入 reward/GAE，违反 observation-only gate |
| First update | 未发生 optimizer update，也未产生 participation sidecar；R3 数据不可用于性能比较 |
| Sandbox cleanup | 本次创建 23 个；正常 cleanup 终止 19 个，剩余 4 个因 service-account ownership 无法由当前用户取消 |

启动中发现并修复了一个独立于 lineage 语义的部署问题：`RayHTTPLauncher` 的 child process 原先从 `/code` 启动，导致它导入镜像内旧版 `areal`，而不是当前部署目录。现在 child `cwd` 固定为当前已导入 AReaL checkout root；rank 3 integration smoke 已确认 child 导入路径为本次部署目录，并增加 `test_ray_http_launcher.py` 回归测试。

### R4 final state

R4 使用 bs2 快速诊断场景，只验证 P0 tracing 的真实闭环，不与 canonical `32x8` R8b 直接比较吞吐。

| Item | Value |
|---|---|
| Fuyao job | `bifrost-2026070622090200-zengbw1` |
| Trial | `trial_oh_bs2_amortized_LONG_P0LINEAGE_R4` |
| Run ID | `p0-lineage-bs2-r4-20260723_103205` |
| Run directory | `/dataset_rc_b1/zengbw1/log/areal_r2e_gym_qwen35_9b/128k_bifrost-2026070622090200-zengbw1/experiments_zbw_128k_nativeqwen3coder_d1t2c4p1_bs2_g8_mb131072_p0lineage/20260723_103337` |
| Participation sidecar | `/dataset_rc_b1/zengbw1/log/areal_r2e_gym_qwen35_9b/128k_bifrost-2026070622090200-zengbw1/p0_lineage/20260723_103205_R4_training_participation.jsonl` |
| Frozen workload | `batch_size=2`、`n_samples=8`、`max_seq_len=128K`、`max_head_offpolicyness=2` |
| Topology | 16 Megatron actor GPUs + 16 vLLM TP1 rollout GPUs；`enable_tree_training=false` |
| G0-G2 | 4-node Ray healthy；16 actor + 16 rollout workers ready；gateway 和 evals/sandbox workflow 已连通 |
| Final lineage funnel | `admitted=91 -> generated=55 -> manager_exported=16 -> workflow_exported=0 -> trainer_consumed=0` |
| Final dispositions | `manager_exported_not_workflow_completed=16`、`partial_cohort_at_deadline=7`、`rewarded_waiting_cohort=32` |
| First update | 未发生；首个 trainer batch 在 lineage detach 阶段失败，不允许做性能解释 |
| Failure | `trajectory 0 has 8 records for batch size 0`；single-controller 收到 `RTensor`，旧 `get_batch_size()` 只识别本地 `torch.Tensor` |
| P0.1 fix | 仅读取 `RTensor.data` 的 meta shape；测试 backend 禁止 fetch，证明 controller 不会 materialize 128K tensor |
| P0.2 fix | participation log 与 `run_id` 同时传播到 actor 和 rollout scheduling specs；关闭 tracing 或非 single-controller 时不改 env |
| Process cleanup | trainer 已退出；残留 wrapper/eval/tee 三个 R4 PID 已精确终止；Ray 复核为 `0/32 GPU`、`0/320 CPU` |
| Launcher classification | R4 远端旧脚本 hash `ab44ad44...`，当前受审脚本 hash `d19774fe...`；旧版本不含现有双进程回收逻辑，因此本轮不再修改 launcher |
| Sandbox cleanup | owner 已授权使用 R4 launcher 的既有凭据做定向清理；248 个历史 ready sandbox ID 均被平台判定为当前身份不可取消，结果为 `attempted=248 / killed=0 / failed=248`。没有绕过 ownership，也没有扩大清理范围 |
| Known cleanup residue | R3 另有 4 个 CPU sandbox 由 service account 持有；当前账号无权取消，不占本 job 的 Ray/GPU 配额 |
| Newly exposed P0 gap | episode `r2e_gym_train:1:17` 因 sandbox 内 tmux 安装失败，在 gateway admission 前终止；当前 sidecar 无 `planned` 事件，无法从 manager trace 单独解释该 expected rank |

首个 cohort 曾在 age 120 秒时显示 `7/8 started`，到约 180 秒正常补齐 `8/8`。这次缺 1 条是 sandbox provisioning/admission lag，不是生成后丢失。相反，episode `:17` 的 tmux failure 发生在 admission 前，证明完整 expected-slot closure 还必须把 eval planner 的 `planned -> sandbox_started -> admission_attempt` 接入同一 `run_id/cohort/rank`；这是后续 P0 观测补充，不在 R5 前改变 recovery 行为。

### Canary feedback loop

| Gate | Evidence | Current result |
|---|---|---|
| G0 process | launcher、trainer、eval runner PID + 4-node Ray status | `PASS_R4` |
| G1 serving | 16 backend ready + gateway discovered | `PASS_R4` |
| G2 environment | sandbox ready + agent running + gateway traffic | `PASS_R4` |
| G3 trajectory | rollout completion、reward、manager admission | `PARTIAL_PASS_R4`；55 条 generated+rewarded，16 条 manager exported |
| G4 trainer | first complete logical batch enters PPO update | `FAIL_R4`；`RTensor` batch size 误判为 0 |
| G5 participation | JSONL strict join、conservation、version coverage | `FAIL_R4`；workflow/trainer 事件均为 0 |
| G6 observation-only | tracing on/off 保持 loss、gradient、reward/GAE 语义一致 | `CODE_REVIEW_PASS`；matched live A/B 未运行 |

如果 G3-G5 任一层失败，先按最后一个成功 gate 定位，不允许用存活进程或 GPU utilization 代替训练闭环证据。

## R5 Preflight Freeze

Status: `OWNER_CONFIRMED / REMOTE_SNAPSHOT_VERIFIED / LAUNCHED`

R5 只验证 P0 lineage 闭环，不启用 retry、replacement、partial-cohort training 或其他 P1 behavior。计划复用现有 Fuyao job 的 4 个空闲 A100 节点，不重新申请资源。

| Item | Frozen value |
|---|---|
| Resource | 4 nodes x 8 A100-80GB；16 actor GPU + 16 rollout GPU |
| Train config | `train_qwen35_9b_sftablb_native_qwen3coder_128k_dense_cp8_bs2_long.yaml`；SHA256 `5086ea47a23e5f8b3f7fbdcbb80cfc321050adc5568089688ca73a37a4e75c44` |
| Eval config | `eval_qwen35_9b_sftablb_native_qwen3coder_128k_dense.yaml`；SHA256 `59076ad23fe1dcb4e6713b7fe8cd5f08b9f4e21ca5da39c03cdfc4eb70067f84` |
| Logical batch | `train_dataset.batch_size=2` cohorts、`n_samples=8` trajectories/cohort，即每步 16 trajectories |
| Context/sampling | `max_tokens=131072`、`max_new_tokens=8192`、temperature `1.0`、top-p `0.95`、top-k `50`、greedy `false` |
| Topology | rollout `vllm:d16t1`；actor `megatron:d1t2c8p1`；tree training disabled |
| Online bounds | `max_concurrent_rollouts=48`、worker capacity `6`、`max_head_offpolicyness=2`、FIFO `96` cohorts |
| Timeouts | session/cohort `10800s`、partial cohort deadline `900s`、weight-update drain `1200s` |
| Seed | train/vLLM seed `1` |
| Checkpoint | `/dataset_rc_b1/suncy5/qwen35_9b_base_ablation_b_swehero_terminal_litecoder_gs4308_hf_eosfix` |
| Checkpoint manifests | config `946b51b2...`；weight index `3587e32c...`；tokenizer config `9c68b91e...`；chat template `a4aee8af...` |
| Task suite | `r2e_gym_train.yaml` SHA256 `fa2e3e30...`；4578 episodes |
| Task manifest | `tasks.jsonl` SHA256 `3dd27ba5...`；`images.yaml` SHA256 `ee2edad5...` |
| Source base | git `cf1fc168fb5fb45da4e7890c49d9114b69a3fe12` |
| Reviewed source tree | tracked + untracked source content hash `3ac6cdf507e262882a4cf03e05c26ae9c5b80de955cfde6170daf45efd5a71c5` |
| Runtime identity | 启动时生成唯一 `AREAL_RUN_ID` 与不存在的新 sidecar 路径；禁止复用 R4 文件 |
| Logging | SwanLab disabled；保留 AReaL/evals/participation sidecar 与 analyzer 输出 |

Owner 于 2026-07-23 确认上述 R5 配置，并授权定向清理 R4 sandbox。R5 远端 source snapshot 位于
`code_r2e_p0_lineage_r5_20260723_133229/source`：1130 个文件通过 manifest 逐文件校验；train/eval/launcher、
suite/tasks/images 的 SHA256 均与冻结值匹配；快照内 lineage 目标测试为 `36 passed`。启动前 Ray 为
`0/32 GPU`、`0/320 CPU`，未发现旧 launcher、trainer 或 `areal_evals_bridge.runner`。

## R5 Live Run

| Item | Value |
|---|---|
| Run ID | `p0-lineage-bs2-r5-20260723_134723` |
| Trial | `trial_oh_bs2_amortized_LONG_P0LINEAGE_R5` |
| Run directory | `.../experiments_zbw_128k_nativeqwen3coder_d1t2c4p1_bs2_g8_mb131072_p0lineage/20260723_135011` |
| Participation sidecar | `.../p0_lineage/20260723_134723_R5_training_participation.jsonl`；启动前不存在，权限 `0600` |
| Process identity | launcher PID `181807`；trainer PID `184079`；eval runner 唯一 |
| G0 | `PASS`：4 个 Ray 节点；单一 launcher/trainer/eval process tree |
| G1 | `PASS`：16/16 vLLM server ready；16 backend gateway 在 `10.1.48.29:46659` 就绪 |
| G2 | `PASS`：新 sandbox ready，OpenHands agent running，sidecar 开始接收 admission 事件 |
| G3 | `PASS`：version 6 strict snapshot 达到 admitted `223`、generated/rewarded `180`、manager/workflow exported `96` |
| G4 | `PASS`：6 个 update 共消费 96 条；审计截止到 `global_step=5 -> version=6` |
| G5 | `PASS`：trainer rows `96`、`EXACT=96`、version coverage `100%`、UNKNOWN `0`，无 UID join 或 token conservation error |
| Run isolation | analyzer 只发现 run ID `p0-lineage-bs2-r5-20260723_134723`；foreign `run_id` events `0` |

`global_step=5 -> version=6` 完成后的 live strict analyzer snapshot：

| Metric | Value |
|---|---:|
| Admitted attempts | 223 |
| Generated / rewarded | 180 / 180 |
| Manager / workflow exported | 96 / 96 |
| Trainer consumed | 96 |
| Trainer rows / exact trajectories | 96 / 96 |
| Full-sequence tokens | 4,073,826 |
| Response tokens | 518,340 |
| Loss-active tokens | 518,340 |
| Policy-gradient-active tokens | 518,340 |
| Behavior-version coverage | 100% |
| UNKNOWN participation | 0 |

异步流水线中，manager/workflow 可以领先 trainer，因此 gate 仍然使用 UID subset 而不是强制全局计数相等；
本次 snapshot 恰好是 `96/96/96`。96 个 trainer UID 均能一一回溯到 manager/workflow，且没有
missing/conflict/cross-run event。

96 条中 94 条产生 policy-gradient token，2 条被正确标记为
`trainer_consumed_fully_masked`，不是 UNKNOWN，也不是 tracing 丢失：

| Full-sequence tokens | Reward/eval timestamp | Exact cause |
|---:|---|---|
| 28,258 | `2026-07-23 14:21:22` | agent `exit=-1`、`signal: terminated`，随后 0 reward + compact-filter |
| 131,072 | `2026-07-23 14:52:13` | agent `exit=0`，但 OpenHands 进入 `AgentState.ERROR`，随后 0 reward + compact-filter |

两条合计 159,330 full-sequence tokens，占 6 个 update trainer full-sequence processing 的
`3.91%`，但 policy-gradient contribution 为 0。这个比例不能解释为 rollout GPU 浪费比例，因为
当前 R5 没有记录 APC/KV reuse 后的物理工作。

审计截止时的 84 条 generated-but-not-trainer-consumed 已完整分类：

| Disposition | Trajectories | Cohort evidence |
|---|---:|---|
| Partial | 35 | `idx-2/4/6/7/12` 均为 7/8 后 `partial_timeout`；分别缺 source rank `6/3/3/4/0` |
| Stale | 16 | `idx-17/22` 均为 8/8；rollout version 分别为 2/3，随后 terminal staleness |
| Rewarded waiting | 33 | 截止 snapshot 尚未 terminal，不得记作浪费 |

当前 R5 能逐 UID 确认 51 条 terminal-unconsumed 的数量和终态，但没有记录它们的 rollout token work，
所以 token/GPU 成本仍是 `UNKNOWN`；新代码已为 E0-on 增加 logical rollout token ledger。

6 个 update completion timestamp 形成 5 个 intervals：
`828.114s / 1289.390s / 759.081s / 384.736s / 624.424s`，median `759.081s`、mean
`777.149s`、range `384.736-1289.390s`。方差很大且尚未做 matched tracing-off/on，不能作为
steady-state 性能结论。

`tasks.jsonl` 的文件内容和字节顺序已由 hash 冻结；eval 配置未显式提供 shuffle seed，因此 R5 的可重复性声明限定为“相同 loader、环境、配置与 manifest”，不外推为跨 loader 实现的绝对调度顺序。

启动后验证只有一套 launcher/trainer/eval process tree，并依次通过 G0-G5。聚焦日志检索未发现
Traceback、RuntimeError、empty batch/trajectory 或 CUDA OOM。R5 当前证明 P0 tracing 在 6 个真实
update 上闭环；它没有证明 tracing overhead `<=2%`，也没有证明任何 recovery 性能收益。

## Remaining P0 Gaps

| Gate | Current status | Required evidence |
|---|---|---|
| `2000 -> 960 -> 1040` 六层逐 UID closure | `R5_SIX_UPDATES_PASS / FULL_RUN_PENDING` | 6 个真实 update 已闭环；仍需覆盖完整历史 run 的所有终态 |
| manager terminal disposition | `CODE_COMPLETE / LIVE_PARTIAL` | 已见 waiting、5 个 partial 和 2 个 stale；仍需 cancelled、incomplete、shutdown-open 的完整覆盖 |
| token-version audit | `R5_SIX_UPDATES_PASS / DISTRIBUTION_PENDING` | coverage 100%、UNKNOWN 0；仍需 p50/p95、超界 token、IS/clip/rejection 分布闭环 |
| compact-filter cause | `CODE_COMPLETE / LIVE_PENDING` | R5 已靠同秒日志闭环 signal termination 与 AgentState.ERROR；bounded reason code 待 E0-on sidecar 验证 |
| terminal-unconsumed rollout token work | `CODE_COMPLETE / LIVE_PENDING` | R5 的 51 条 partial/stale 只有 count，token cost UNKNOWN；E0-on 必须报告 logical prompt/output count coverage |
| tracing overhead `<=2%` | `UNKNOWN` | matched E0-off/on 的 post-warmup update interval |
| recovery critical-path value | `NOT_RUN` | timeline replay 与 bounded counterfactual |
| canonical `32x8` A0 | `NOT_RUN` | 同代码、任务、seed、checkpoint、sampling 和 32 A100-80GB |
| downstream non-inferiority | `NOT_FROZEN` | benchmark、checkpoint/step、主指标和容差预注册 |

因此当前结论是：**R3 已否决；R4 暴露的 `RTensor` 与 env propagation 问题已修复；R5 已通过 6 个真实 optimizer update 的 G3-G5 strict lineage 闭环。P0 observation 能精确区分 94 条 gradient-active、2 条 fully-masked、35 条 terminal partial、16 条 terminal stale 和 33 条 rewarded waiting。由于 matched E0-off/on overhead、terminal-unconsumed token coverage、完整终态覆盖和 G6 live equivalence 尚未完成，P1 继续 `HOLD`。**

## E0 Verifiable Feedback Loop

1. **Freeze inputs**：固定 bs2 task manifest、顺序、seed、checkpoint、sampling、group size 8、128K、资源和代码 commit；保存各项 hash。
2. **Run E0-off**：同一代码，不设置 participation log；保留 no-EOS correctness fix，采集至少 10 个 post-warmup update intervals。
3. **Run E0-on**：唯一变化是设置 `AREAL_TRAINING_PARTICIPATION_LOG`；使用新的空 JSONL 路径，禁止复用旧文件。
4. **Fail-closed audit**：analyzer 必须满足 map/participation join completeness 100%、version coverage 100%、unknown 0；任一失败先修 tracing，不分析性能。
5. **Conservation**：逐 trajectory 检查 `policy_gradient_active <= loss_active <= response <= full_sequence`；按 step、cohort 和全窗口三层闭合。
6. **Performance decision**：用 update completion timestamp 计算 paired post-warmup interval；on 相对 off 的 overhead 上界必须 `<=2%`，CPU microbenchmark 不参与 gate。
7. **Correctness decision**：抽样从原始 response stop reason 复算 no-EOS；核对 token behavior version、rejection、clip 和 final masks。
8. **Promote or stop**：上述 gate 全过才运行 timeline replay 和 `32x8` A0；否则回到对应采集层修复，P1 不解锁。

## Next Action

R4 sandbox 已完成 owner 授权的定向清理尝试，但 248 个历史 ID 均因 ownership/terminal-state
约束不可由当前身份取消；不做权限绕过。R5 截止 version 6 的 6 个 update 已通过 UID strict join、
token conservation、version coverage 与 UNKNOWN audit；两条 fully-masked、5 个 7/8 partial cohort
和 2 个 stale cohort 已逐 UID 闭环。
下一步准备 matched E0-off/on，在 E0-on live 验证 bounded reason code 与 logical rollout token ledger。
只有 paired post-warmup interval 证明 tracing overhead `<=2%`，且 terminal-unconsumed token coverage
闭合，才进入 timeline replay 和 canonical `32x8` A0。P1 继续 `HOLD`。

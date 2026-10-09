# Current Status

Last updated: 2026-07-24

这是项目 owner 默认需要阅读的唯一页面。详细数据和推导只在需要审计时进入链接文档。

## Goal

在固定 logical batch（32 cohorts x 8 trajectories）、最大序列长度（128K）、32 张 A100-80GB 总资源和算法正确性的条件下，降低 Qwen3.5-9B Agentic RL 的 overlap-aware post-warmup E2E step time。实际 token 数允许随轨迹自然变化。

## Canonical Baseline

| Run | Logged step phase sum | Rollout wait | Train | Rollout share |
|---|---:|---:|---:|---:|
| [R8b](baselines/R8b.md) | 83.89 min | 73.21 min | 10.46 min | 87.27% |

口径：step 1-6 的 logged `timeperf/*` phase sum，32 cohorts x 8 trajectories，`max_seq_len=128K`，32 张 A100-80GB，其中 rollout/train 为 16/16。它是当前历史 proxy；未来实验以 update-completion interval 和同窗口 makespan 复核真实 E2E 与异步重叠。

## Diagnostic Case

| Run | Workload | Logged phase-sum mean / p95 | Rollout wait share | Cohorts / rollout GPU-h | Decision |
|---|---:|---:|---:|---:|---|
| [bs2 C1b v2](cases/bs2-eqtraj-C1b-v2.md) | 2 x 8 | 15.98 / 32.54 min | 93.78% | 0.500 | 只用于快速归因，不替代 R8b |

bs2 的单 step 比 R8b 短 80.95%，但 logical batch 小 16 倍，work-normalized cohort goodput 反而低 69.5%。它证明了“小 batch 适合快速反馈”，没有证明正式训练吞吐更高。

## Current Judgement

1. Rollout 是确定的一阶瓶颈；bs2 中 93.78% step wall 暴露为 online dequeue wait。
2. 375 条 episode 跑到 80-turn cap，其中 reward > 0 只有 10 条；这是最明确的高成本长尾，但在训练效果实验前不把 consumed negative 判成浪费。
3. LLM RPC 累计时延与 episode elapsed 的相关系数为 0.906；context growth 和后期请求成本是一阶解释。
4. Request cache ratio 在 turn 10-19 为 95.44%，turn 20+ 降为 69.45%。cache 可能加速进入后期昂贵区间，但因果仍需 A/B。
5. bs2 的 `no_eos_ratios` 在 dynamic padding 下不可信，不能作为算法正确性 guardrail。

## Next Experiment

样本利用分析统一从 [Trajectory Utilization Ledger](analysis/trajectory-utilization-ledger.md) 进入。当前 `2000 -> 960 -> 1040` 只完成 aggregate closure。当前只推进 P0：先把 generated、manager、workflow、trainer、final loss 和 policy gradient 逐 trajectory 连通，再审计真实 token-version 边界和 recovery 的 critical-path 价值；P1 retry/replacement 暂停。

### 2026-07-23 Runtime-ready Freeze

按 [Runtime-ready experiment plan](plans/2026-07-23-runtime-ready-cohort-start-experiment-plan.md) 冻结 fresh A0/E1。R5 只作 root-cause evidence，已于 2026-07-24 优雅停止 trainer/evals，并停止四个 rank 的 R5 Ray cluster。

| Identity | Frozen value |
|---|---|
| Source commit | `ecc4af0436f15a5e4cb6566ccf537fa2c25b7913` |
| Source manifest file SHA | `8e36d1d16a6655a4f0e5731199d2abde8ddc6ed6527632e0af616e660ac957dc` |
| Source tree SHA | `c780b750af01ee894746901d1aa630fa8b826b9b6391092ab6da6701c3223965` |
| Fixed workload | 64 tasks x group 8 = 512 planned trajectories |
| Fixed workload manifest SHA | `fbc3af0792d64924deb4012a7061cf0e107d33f4d830f4098e48e96cb3b9409a` |
| Immutable evals manifest SHA | `dd6c9d968733a4f3938e1e7c94e848485214e8788ac6156eed796d9369d506a0` |
| Portable tmux manifest SHA | `df0700af53299ec1ecde6b02fd157f625ee53729fb51df8597fd52f129e5d198` |
| Prior clean-snapshot regression (`1a312e2b`) | `244 passed, 3 subtests passed` |
| `ecc4af04` focused contract regression | `67 passed, 3 subtests passed` |
| Runtime canary | 16/16 historical/representative task images passed full tmux lifecycle |
| A0 pre-launch behavioral/file SHA | `6889d8ef...60f259` / `fc71511a...d730ee`；已被 preflight mismatch 取代 |
| A0r1 behavioral/file SHA | `b4cc27e8...afec9d` / `6d4d8387...66659` |
| A0r1 comparison-domain SHA | `baa56029...04efe` |
| E1 pre-launch behavioral/file SHA | `1a27c5c9...9daa7` / `a6ef91ab...b43b5`；A0 drain 后按 deployed root 重建 |

A0/E1 的预启动 behavioral manifest 已完成 source、model、task order、sampling、trainer/rollout topology、reset concurrency 和 launcher 对账，目标唯一 intervention 仍是 `online_bootstrap -> portable_tmux`。实际部署时发现旧 A0 patched eval domain SHA `88c4f379...e3c6f1` 与同一 manifest 冻结的 deployed `PROJECT_ROOT` 不自洽；运行时正确值为 `15c9a433...d300c`。A0r1 已用原 builder 重新扫描全部冻结 artifact，旧 manifest 与 deploy-root 修正版只差 patched domain 和两个派生 digest；A0r1 再只改变 run metadata，comparison-domain SHA 保持不变。E1 必须在 A0 drain 后用相同 deployed-root 规则重建，不能继续使用旧 `9e33795a...4d34c8` 直接启动。

`ecc4af04` 相对上一版只修改 launcher 的 `export AREAL_RUN_ID="${RUN_ID}"` 及对应契约测试，防止 trial identity 与 sandbox registry identity 再次断裂。新旧 A0/E1 manifest transition 已逐字段验证，除 source/run metadata 和这两个源码文件外无其他变化；远端 1165 个 source 文件已重算 tree SHA 并通过。

### 2026-07-24 Execution Gate

Owner 已确认 A0 启动，并明确 sandbox 操作边界为仅检查、清理 zengbw owner 资源。R5 的 wrapper/trainer/evals PID 优雅退出，四节点 Ray 均已停止；训练相关 GPU 进程为 0，剩余低显存进程均识别为平台 `gpu_opt`。

R5 历史 `sandbox_jobs.jsonl` 早于 mandatory `run_id` patch。原文件保留不动，并用 trainer 环境和 lineage 双重确认唯一 run ID `p0-lineage-bs2-r5-20260723_134723` 后生成派生 cleanup registry：

| Cleanup evidence | Value |
|---|---|
| Final source registry SHA | `5cacd4a90a4c6c722a39ba84c740fae6d3401775a7ebbfee9c5b9c7a574d034b` |
| Derived registry SHA | `676bddfe0011c59e823caaea38f64ccd027165a6afe042922a9d9d9e17268ffb` |
| Unique acquired / released / orphan | `3538 / 3420 / 118` |
| Official sandbox SDK closure | `118 queried, 0 unknown` |
| Actual terminal / live | `20 JOB_CANCELLED / 98 JOB_RUNNING` |
| Exact cleanup attempt | `98 attempted / 98 permission denied / 0 cancelled` |
| Acquired owner audit | `3534 suncy5 / 4 unparseable` |
| Orphan owner audit | `118 suncy5 / 0 zengbw` |

98 个 live sandbox 尚未清理。2026-07-24 已临时注入 credentials 并用官方 sandbox SDK 做真实对象核验；读取权限有效，但精确 kill 的 98 次请求全部被平台拒绝，原因一致为当前 `AUTH_USER=zengbw1@xiaopeng.com` 不是这些 job 的 admin、queue/team/org owner。完整 owner 审计进一步确认：118 个 orphan 全部属于 `suncy5`，没有 zengbw owner 项；98 次取消均未成功，因此没有实际改动 suncy 资源。按 Owner 边界，这批资源只保留证据，不再查询式清理或取消，且不作为 A0 启动门槛。

R5 `evals.log` 中实际创建的 sandbox job 均带 `suncy5` owner；这与只读 checkpoint 路径中的 `suncy5` 无关，资源 owner 由 sandbox API key 决定。R5 source snapshot 的正式 `fuyao_debug_bash` launcher 没有 hardcoded key fallback，必须从进程环境读取；starter 会 source `/workspace/zengbw1@xiaopeng.com/soft/.r2e_env`。该私有文件自 2026-07-02 起未修改，并同时保存相同的 `FUYAO_API_KEY` 与 `FUYAO_SANDBOX_API_KEY`。由于缺少 R5 启动进程的环境快照，无法把 key 来源形式化证明到单一 shell 赋值；结合资源 owner、source snapshot 和文件时间线，高置信根因是旧私有 env 向 R5 注入了非 zengbw owner 的 key，而 `AUTH_USER=zengbw1@xiaopeng.com` 不能覆盖 key owner。

zengbw owner gate 已于 2026-07-24 通过。使用 1 CPU、2 GiB、无 GPU 的最小 canary 创建 `bifrost-2026072413135101-zengbw1`（sandbox `28544624`），backend job name 精确命中 `-zengbw1`；随后 kill 成功，并独立复核终态为 `JOB_CANCELLED`。首次 canary 汇总中的 `passed=false` 来自 verifier 未展开结构化 state 对象，不是资源未回收；修正解析后对同一 sandbox 复核通过，没有重复创建 canary。

已将通过 canary 的 key 原子写入 `/workspace/zengbw1@xiaopeng.com/soft/.r2e_env`：仅保留显式 `FUYAO_SANDBOX_API_KEY`，删除旧 `FUYAO_API_KEY` fallback，保留 `AUTH_USER` 与 SwanLab 配置，权限为 `0600`。轮换后布尔审计全部通过，临时 key 已从本地和 pod `/dev/shm` 删除。Frozen A0 实际调用的 `fuyao_debug_bash` launcher 在缺 key 时 fail closed，因此 sandbox owner 启动门槛已满足。

### 2026-07-24 A0 Execution

第一次 A0 attempt 在 run directory `20260724_133653` 被 patched eval domain 校验拦截：expected `88c4f379...e3c6f1`，actual `15c9a433...d300c`。eval runner 未进入 sandbox create；失败后 Ray 控制面退出但部分 vLLM 被 container init 接管，已按本次 A0 进程组定向清理，四节点 preflight 均恢复 `clean=true`。该 attempt 不计入性能样本。

A0r1 已于 2026-07-24 14:19 启动并进入真实 rollout：

| Runtime evidence | Value |
|---|---|
| Run directory | `/dataset_rc_b1/zengbw1/log/areal_r2e_gym_qwen35_9b/128k_bifrost-2026070622090200-zengbw1/experiments_zbw_128k_nativeqwen3coder_d1t2c8p1_bs2_g8_mb131072_p0fixed512_a0r1/20260724_141924` |
| SwanLab | [A0r1](https://swanlab.cn/@zengbw1/areal-experiments/runs/ycjjbmki9h9ls685xvvtx) |
| Cluster / engines | 4/4 Ray nodes；16 actor + 16 rollout engines ready |
| Patched eval domain | `15c9a433...d300c`，运行时校验通过 |
| Fixed producer | 512 episodes enqueued，64 cohorts x group 8 |
| First real sandboxes | `28548945-28548948`，独立 SDK 查询全部 owner suffix `-zengbw1`、`JOB_RUNNING` |
| First cohort lineage | `idx-0` 已 `8/8 started+confirmed+active`；`idx-1` 已开始 admission |
| Current conclusion | 启动、owner、固定 workload 和 cohort admission 已闭环；尚未形成首个 optimizer update 或性能收益结论 |

日志里的 `suncy5` 仅出现在只读 checkpoint/model 路径。A0r1 的 parent job、SwanLab identity、sandbox job name 和 API-key canary 均为 `zengbw1`，没有使用或清理 suncy owner sandbox。

### Historical P0 Evidence

P0 代码侧按 [lineage instrumentation validation](experiments/2026-07-22_p0-lineage-instrumentation-validation.md) 推进。R3 因 terminal reason 错误进入 reward/GAE 而作废；R4 形成 16 条 manager-exported trajectory 后，在首个 trainer batch 因 `RTensor` 被误判为 batch size 0 而失败。P0.1 已改为只读 `RTensor` meta shape，P0.2 已把 trace env 同时传播到 actor/rollout；两项通过 207 条远端测试、2/4/8 GPU redistribution、Ruff、compileall、`git diff --check` 和双 reviewer。

R5 已完成 6 个真实 optimizer update。冻结在 `global_step=5 -> version=6` 的 strict snapshot 得到
`223 admitted -> 180 generated/rewarded -> 96 manager/workflow exported -> 96 trainer consumed`；
96 条全部 `EXACT` join，version coverage `100%`、UNKNOWN `0`，full-sequence `4,073,826` tokens，
response/loss-active/policy-gradient-active 均为 `518,340`。5 个 update intervals 的 median 为
`759.081s`、mean `777.149s`、range `384.736-1289.390s`；尚不是 matched post-warmup 性能结论。

96 条里有 94 条产生 policy-gradient，2 条 compact-filtered：一条 28,258 full-sequence tokens，
对应 signal terminated；另一条 131,072 tokens，对应 OpenHands `AgentState.ERROR`。两条合计
159,330 tokens，占本窗口 trainer full-sequence processing 的 `3.91%`，但不产生梯度。另有 35 条
terminal partial（5 个 7/8 cohort）和 16 条 terminal stale（2 个 8/8 cohort）；33 条仍是
rewarded waiting，不能记作浪费。

代码已新增 bounded `compact_filter_reason_code`、causal disposition 和 session 级 logical rollout
prompt/output token ledger；tracing-off 不访问 token payload，保证 matched overhead control 口径。
隔离远端完整相关回归为 `143 passed`。正在运行的 R5 没有热改，因此当前
51 条 partial/stale 的 rollout token/GPU 成本仍为 `UNKNOWN`，留到 E0-on live 验证。G3-G5 已通过，但
matched tracing overhead、terminal-unconsumed token coverage 和 G6 live equivalence 尚未完成，
P1 继续 `HOLD`。

| Item | Value |
|---|---|
| Trigger | 按 [cohort recovery implementation plan](plans/2026-07-22-cohort-recovery-implementation-plan.md) 执行 P0；P1 保持 `HOLD` |
| Fast diagnostic | 固定 task/seed/checkpoint 的 bs2 `TU-E0a/TU-E0b` observation-only replay |
| Primary change | 稳定 logical trajectory ID、microbatch reorder-safe lineage、final loss/policy-gradient participation、per-token version audit |
| Diagnostic metrics | 六层 join、unknown disposition、loss/policy-gradient-active token、token staleness、IS/clip/rejection、tracing overhead |
| Critical-path gate | 用 `TU-E0c` timeline counterfactual 验证补齐 89 slots 是否能缩短 update interval，而不是用 `407/89` 代替性能收益 |
| Canonical control | P0 内运行 matched observation-only `32x8` A0；历史 R8b 只作背景 |
| Final effect gate | 训练所得 checkpoint 的下游任务评测 |
| P1 unlock | P0 首步 join 已过；仍需 overhead/version/no-EOS/replay/A0 全过且 Owner review 后，才允许 runtime/retry/manifest behavior experiment |

## Decisions

- 资源边界：固定 32 张 A100-80GB；rollout/train 划分和拓扑可以作为单一实验变量。
- 主性能指标：固定 workload 和总资源下，连续 update 完成点之间的 steady-state wall-clock interval。
- 异步统计：不能把 rollout active time 与 train active time直接相加；必须报告 overlap 和 exposed critical-path time。
- Participation goodput：同时报告 trainer-consumed cohort/trajectory、full-sequence token、loss-active token 和 policy-gradient-active token；缺失不能用 0 代替。
- Version correctness：manager cohort-head drift 与实际 per-token behavior staleness 分开；P0 保持 baseline `max_head_offpolicyness=2` 的 head gate 和现有 rejection/IS，不擅自改成 `drift=0`，也不把 head bound 误写成 token hard bound。
- Waste 口径：只有 terminal unconsumed、dropped/stale/cancelled、retry/duplicate、infra-invalid 和明确等待可以进入浪费账本；reward 正负只作训练数据组成。
- 训练中 reward 只作在线参考；最终效果以训练所得模型权重的下游任务评测为准。
- 每次实验只改变一个主要变量；否则只能标记 exploratory。

## Owner Attention

P0 owner review 检查四项：六层逐 trajectory join 和 final loss/policy-gradient count；dynamic-padding no-EOS 可由 unpadded response/terminal reason 复算；manager head drift 不越界且 token staleness/IS/clip/rejection 可闭环；下游 benchmark、主指标、checkpoint/step 和 non-inferiority 容差已预注册。任一项未完成，不进入 P1。

完整横向数据见 [dashboard](dashboard.md)，统一实验格式见 [scorecard](templates/experiment_scorecard.md)。

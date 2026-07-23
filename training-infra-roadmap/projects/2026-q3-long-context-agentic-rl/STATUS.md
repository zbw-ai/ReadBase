# Current Status

Last updated: 2026-07-23

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

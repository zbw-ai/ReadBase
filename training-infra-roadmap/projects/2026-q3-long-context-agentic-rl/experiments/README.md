# Experiments

这里保存 Q3 长上下文 Agentic RL 主线的受控实验。

## Naming

```text
YYYY-MM-DD_<baseline>_<single-change>.md
```

例如：

```text
2026-07-xx_R8b_observation-only-tracing.md
2026-07-xx_R8b_session-affinity-off.md
2026-07-xx_R8b_prefix-cache-ab.md
```

## Lifecycle

```text
PLANNED -> RUNNING -> ANALYZED
        -> PERF_ACCEPTED
        -> TRAINING_VALIDATED / REJECTED / INCONCLUSIVE
```

`PERF_ACCEPTED` 只表示 overlap-aware E2E step time、trainer participation 和在线 guardrail 通过；只有训练所得 checkpoint 的下游任务评测通过，才能标记 `TRAINING_VALIDATED`。每个实验从 [scorecard template](../templates/experiment_scorecard.md) 开始，同时在 `metrics/` 保存同 schema 的结构化指标，并在完成分析后更新 [performance dashboard](../dashboard.md)。

## Analyzed Diagnostic Cases

| Case | Role | Result |
|---|---|---|
| [bs2 C1b v2](../cases/bs2-eqtraj-C1b-v2.md) | 快速 tracing / long-tail diagnosis | 80-turn 高成本尾部明确；是否 early-stop 仍需训练效果验证，且不能替代 `32x8` baseline |

## Code Validation

| Record | Status | Conclusion |
|---|---|---|
| [P0 lineage instrumentation validation](2026-07-22_p0-lineage-instrumentation-validation.md) | `CODE_VALIDATED / E0_PENDING` | 200 tests、strict join 和 analyzer smoke 通过；真实六层 closure、E0 overhead、timeline replay 与 `32x8` A0 尚未运行 |

## Planned

| Experiment | Purpose | Dependency |
|---|---|---|
| [Trajectory utilization TU-E0](../analysis/trajectory-utilization-ledger.md#6-experiment-matrix) | 打通 generated -> manager -> workflow -> trainer -> loss/gradient 的逐 trajectory 去向 | Observation-only tracing |
| Fixed-task bs2 prefix-cache replay | 判断 early/mid-turn cache 收益是否导致 late-turn tail 扩大 | 当前 Fuyao 任务结束 |
| Agent horizon replay | 固定 cache，单独验证 max-iteration/loop/zero-progress control | prefix-cache replay |
| `32x8` observation-only reproduction | 补齐 canonical workload 的 consumption lineage、overlap 和 tracing overhead | 小实验确定 tracing schema |

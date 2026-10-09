# Structured Metrics

这里保存每个 baseline、diagnostic case 和受控实验的结构化性能数据。Markdown scorecard 用于人读，JSON 用于校验、自动对比和后续 dashboard 生成；二者必须使用同一个 `schema_version` 和统计窗口。

当前 schema：`agentic-rl-performance-scorecard/v2`。

规则：

- 原始缺失值使用 JSON `null`，并同时提供 `status` 和 `reason`；不能用 `0` 代替。
- Step performance 默认排除 warmup；whole-run rollout funnel 单独统计。
- `full_sequence_tokens` 不等于 `loss_active_tokens`。
- Reward 正负是 consumed-data composition，不是 waste disposition。
- 没有 trajectory-level join 时，`generated - consumed` 只能标记为 `INFERRED` aggregate surplus。

## Runs

- [R8b](R8b.json)
- [bs2-eqtraj-C1b-v2](bs2-eqtraj-C1b-v2.json)

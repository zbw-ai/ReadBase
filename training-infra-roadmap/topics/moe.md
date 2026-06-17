# Mixture of Experts

## 核心问题

用 sparse activation 扩展总参数规模：每个 token 只路由到少量 expert，降低 activated compute，但引入 routing、load balance 和 all-to-all 通信。

## 上游材料

- [GShard](../papers/gshard.md)
- [Switch Transformer](../papers/switch_transformer.md)
- [DeepSpeed-MoE](../papers/deepspeed_moe.md)
- [Mixtral](../tech_reports/mixtral.md)
- [DeepSeek-V3](../tech_reports/deepseek_v3.md)

## 关键机制

- router / gate
- top-1 / top-2 routing
- capacity factor
- expert parallel
- token dispatch/combine

## 生产关注

- expert hot spot
- all-to-all tail latency
- dropped token 或 padding waste
- expert checkpoint 重分布

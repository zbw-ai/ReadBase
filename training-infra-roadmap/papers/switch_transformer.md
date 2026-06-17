# Switch Transformer

- 时间：2021
- 链接：https://arxiv.org/abs/2101.03961
- 状态：TODO 深读

## 工程阅读重点

Switch Transformer 用 top-1 routing 简化 MoE，把 sparse model 扩展到 trillion 参数量级。重点看 routing、capacity factor、load balancing loss 和训练稳定性。

## 与知识图谱的关系

[GShard](gshard.md) → Switch Transformer → [DeepSpeed-MoE](deepspeed_moe.md) → [Mixtral](../tech_reports/mixtral.md)。

## 待补问题

- top-1 routing 为什么更简单？
- token dropping/capacity 如何影响质量和吞吐？
- MoE 稳定训练需要哪些系统监控？

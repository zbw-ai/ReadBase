# GShard

- 时间：2020
- 链接：https://arxiv.org/abs/2006.16668
- 状态：TODO 深读

## 工程阅读重点

GShard 是 MoE 和自动分片的重要早期系统。重点看 conditional computation、expert parallel、自动 sharding annotation 和大规模机器翻译训练。

## 与知识图谱的关系

GShard → [Switch Transformer](switch_transformer.md) → [Mixtral](../tech_reports/mixtral.md) → [DeepSeek-V3](../tech_reports/deepseek_v3.md)。

## 待补问题

- Expert Parallel 的早期系统设计是什么？
- 自动 sharding 对生产训练有什么帮助和限制？
- GShard 的 routing/load balance 如何影响后续 MoE？

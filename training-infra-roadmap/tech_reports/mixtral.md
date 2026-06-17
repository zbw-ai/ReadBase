# Mixtral of Experts

- 时间：2024
- 链接：https://arxiv.org/abs/2401.04088
- 状态：TODO 深读

## 工程阅读重点

Mixtral 8x7B 是开放 sparse MoE 的关键报告。重点看 top-2 routing、8 experts、13B activated parameters、32K context 和推理/训练成本结构。

## 与知识图谱的关系

[Switch Transformer](../papers/switch_transformer.md) → Mixtral → [DeepSeek-V3](deepseek_v3.md)。

## 待补问题

- top-2 routing 相比 top-1 的系统代价是什么？
- Mixtral 的 active parameters 如何影响吞吐？
- 开放 MoE 模型在 serving 上有哪些特殊问题？

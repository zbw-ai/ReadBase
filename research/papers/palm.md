# PaLM: Scaling Language Modeling with Pathways

- 时间：2022
- 链接：https://arxiv.org/abs/2204.02311
- 状态：TODO 深读

## 工程阅读重点

PaLM 是 540B dense model 的代表。重点看 Pathways、TPU pod、dense scaling、训练数据和超大模型评估。

## 与知识图谱的关系

GPT-3 → PaLM → Dense Scaling → [Llama 3](../tech_reports/llama3.md)。

## 待补问题

- PaLM 的训练系统与 GPU Megatron 路线有何不同？
- TPU pod 对并行策略有什么影响？
- Dense 540B 的 checkpoint/容错压力如何处理？

# DeepSeek-R1

- 时间：2025
- 链接：https://arxiv.org/abs/2501.12948
- 状态：TODO 深读

## 工程阅读重点

DeepSeek-R1 是 reasoning model 训练报告，重点不在预训练 infra，而在 RL reasoning pipeline、cold-start data、distillation、长链路训练任务管理。

## 与知识图谱的关系

[DeepSeek-V3](deepseek_v3.md) → DeepSeek-R1 → reasoning training infra。

## 待补问题

- RL 训练相对 pretraining 对 infra 有哪些新压力？
- rollout、reward、训练数据生成如何调度？
- reasoning model 的 checkpoint/eval cadence 如何设计？

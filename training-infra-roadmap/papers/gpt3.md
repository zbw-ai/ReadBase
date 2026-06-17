# GPT-3: Language Models are Few-Shot Learners

- 时间：2020
- 链接：https://arxiv.org/abs/2005.14165
- 状态：TODO 深读

## 工程阅读重点

GPT-3 代表 dense decoder-only scaling。重点看模型规模、数据规模、batch size、训练 compute，以及 scaling law 如何把训练系统推向更大 GPU 集群。

## 与知识图谱的关系

[Transformer](transformer.md) → GPT-style decoder → [Megatron-LM](megatron_lm.md) → [Llama 3](../tech_reports/llama3.md)。

## 待补问题

- GPT-3 规模下 DP/MP/PP 如何组合？
- dense scaling 对 checkpoint 和训练稳定性提出什么要求？
- GPT-3 与后续 PaLM/Llama 的 infra 差异是什么？

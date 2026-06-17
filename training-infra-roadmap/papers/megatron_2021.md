# Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM

- 时间：2021
- 链接：https://arxiv.org/abs/2104.04473
- 状态：TODO 深读

## 工程阅读重点

这是 Megatron 从 TP 走向 3D parallel 的关键论文。重点看 TP、PP、DP 如何组合，interleaved pipeline 如何减少 bubble，以及如何在 3072 GPU 上训练 1T 参数模型。

## 与知识图谱的关系

[Megatron-LM](megatron_lm.md) → 3D Parallel → [Pipeline Parallelism](../topics/pipeline_parallelism.md) → [MegaScale](../tech_reports/megascale.md)。

## 待补问题

- 如何选择 TP/PP/DP size？
- interleaved schedule 的收益和复杂度是什么？
- 为什么 naive parallelism 在千卡规模会失效？

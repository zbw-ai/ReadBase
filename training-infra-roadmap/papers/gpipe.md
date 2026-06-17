# GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism

- 时间：2018
- 链接：https://arxiv.org/abs/1811.06965
- 状态：TODO 深读

## 工程阅读重点

GPipe 是 pipeline parallel 的基础材料。重点看 micro-batch、pipeline bubble、activation 保存和同步更新语义。

## 与知识图谱的关系

GPipe → [Pipeline Parallelism](../topics/pipeline_parallelism.md) → [Megatron 2021](megatron_2021.md)。

## 待补问题

- micro-batch 数量如何影响 bubble？
- GPipe 与 1F1B 的区别是什么？
- pipeline stage imbalance 如何在生产中暴露？

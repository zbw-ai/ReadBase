# FlashAttention-2

- 时间：2023
- 链接：https://arxiv.org/abs/2307.08691
- 状态：TODO 深读

## 工程阅读重点

FlashAttention-2 继续优化 attention kernel 的 parallelism 和 work partitioning。重点看为什么 FlashAttention-1 仍有非 matmul FLOPs 和并行划分问题，以及 FA2 如何提升 Tensor Core 利用。

## 与知识图谱的关系

[FlashAttention](flashattention.md) → FlashAttention-2 → [FlashAttention-3](flashattention3.md)。

## 待补问题

- FA2 的 work partitioning 改了什么？
- 为什么减少 non-matmul FLOPs 很重要？
- 不同 head dim/seq length 下收益如何变化？

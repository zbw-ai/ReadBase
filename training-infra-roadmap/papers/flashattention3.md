# FlashAttention-3

- 时间：2024
- 链接：https://arxiv.org/abs/2407.08608
- 状态：TODO 深读

## 工程阅读重点

FlashAttention-3 面向 Hopper 架构，关注 asynchrony、低精度和更高硬件利用。重点看 FP8、Tensor Core、TMA/异步流水等硬件相关设计。

## 与知识图谱的关系

[FlashAttention-2](flashattention2.md) → FlashAttention-3 → [FP8](../topics/fp8.md) → [Transformer Engine](../topics/transformer_engine.md)。

## 待补问题

- Hopper 架构给 attention kernel 带来哪些新机会？
- FP8 attention 的数值风险是什么？
- FA3 与 Transformer Engine 如何配合？

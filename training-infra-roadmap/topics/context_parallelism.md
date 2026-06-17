# Context Parallelism

## 核心问题

长上下文训练中，单卡无法承载完整 sequence 的 attention/activation。Context Parallelism 沿 context 维切分序列，让多个 GPU 协作处理一个样本的长上下文。

## 上游材料

- [FlashAttention](flashattention.md)
- [Sequence Parallelism](sequence_parallelism.md)

## 关键机制

- context dimension partition
- attention KV exchange
- ring attention / block attention 类通信模式
- 与 FlashAttention 的 kernel 边界配合

## 生产关注

- 变长序列会造成 load imbalance。
- CP 的通信模式与普通 DP/TP 不同，需要单独 profiling。
- 长上下文阶段通常需要重新调 micro-batch、recompute 和 checkpoint。

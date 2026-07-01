# FlashAttention

## 核心问题

标准 attention materialize N x N attention matrix，长序列下 HBM IO 和 activation 显存成为瓶颈。FlashAttention 用 tiling + online softmax 降低数据搬运。

## 上游材料

- [Transformer](../papers/transformer.md)
- [FlashAttention](../papers/flashattention.md)
- [FlashAttention-2](../papers/flashattention2.md)
- [FlashAttention-3](../papers/flashattention3.md)

## 生产关注

- kernel fallback detection
- dtype/head_dim/mask 支持矩阵
- varlen sequence packing
- 与 activation checkpointing、context parallel 的边界

## 与其他主题的关系

- [Long-context Training](long_context_training.md)：长上下文训练会把 attention IO、varlen packing、kernel autotune 和 CP 边界一起推到性能主路径上。
- [Context Parallelism](context_parallelism.md)：CP 需要和 attention kernel 的分块、KV exchange、ring/block attention 边界配合。
- [Checkpointing](checkpointing.md)：activation checkpointing 和 recompute 策略会改变 FlashAttention 的显存收益与额外计算开销。

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

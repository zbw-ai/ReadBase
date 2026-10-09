# Interview: FlashAttention

## 高频面试题

1. FlashAttention 为什么能省显存？
2. 它是 approximate attention 吗？
3. Online softmax 解决了什么问题？
4. FlashAttention 主要优化 FLOPs 还是 IO？
5. FlashAttention 与长上下文训练有什么关系？

## 追问问题

- 如何确认没有走 fallback kernel？
- causal mask、varlen、head dim 对 kernel 有什么影响？
- FlashAttention 与 activation checkpointing 如何权衡？

## 生产环境案例

长上下文训练从 8K 扩到 32K 后 OOM，启用 FlashAttention 后显存下降，但吞吐仍不理想。下一步要看 sequence packing、context parallel、micro-batch 和数据 pipeline，而不是只盯 attention kernel。

## 常见错误回答

- “FlashAttention 把复杂度从 O(N^2) 变成 O(N)。”错误，它是 exact attention，主要减少 HBM IO 和中间显存。
- “它只是 fused softmax。”不完整，关键是 tiling + online softmax + IO-aware 数据流。

## 优秀回答示例

FlashAttention 通过分块把 Q/K/V 搬到 SRAM，用 online softmax 在不 materialize 完整 attention matrix 的情况下得到 exact attention。它主要优化 HBM 读写和 activation memory，对长上下文训练尤其关键。

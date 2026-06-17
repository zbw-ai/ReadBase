# Data Parallelism

## 核心问题

复制模型到多张 GPU，切分 batch，反向后同步梯度。它是吞吐扩展的基础，但会复制参数、梯度和 optimizer state。

## 上游概念

- [ZeRO](../papers/zero.md)
- [FSDP](fsdp.md)

## 关键机制

- all-reduce gradients
- gradient accumulation
- global batch size
- optimizer state replication 或 sharding

## 生产关注

- global batch 改变会影响收敛，需要学习率和 warmup 配合。
- DP group 跨节点通信较重，bucket 和 overlap 很关键。
- ZeRO/FSDP 本质是在 DP 语义下去掉状态冗余。

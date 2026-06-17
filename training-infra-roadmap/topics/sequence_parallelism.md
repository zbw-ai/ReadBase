# Sequence Parallelism

## 核心问题

Tensor Parallel 降低参数/计算压力，但 activation 中某些按 sequence 保存的部分仍会复制。Sequence Parallel 把部分 activation 沿 sequence 维切分，降低显存，并与 TP 通信模式配合。

## 上游材料

- [Megatron-LM](../papers/megatron_lm.md)
- [Tensor Parallelism](tensor_parallelism.md)

## 关键机制

- sequence dimension sharding
- reduce-scatter / all-gather activation
- LayerNorm、Dropout 等操作的切分边界

## 生产关注

- SP 通常不是单独打开的魔法开关，要与 TP size、micro-batch、recompute 一起评估。
- 如果通信 overlap 不好，省下的显存可能换来 step time 抖动。

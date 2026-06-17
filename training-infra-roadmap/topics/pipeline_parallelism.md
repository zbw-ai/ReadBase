# Pipeline Parallelism

## 核心问题

模型层数太多时，把 layer 分到不同 GPU/stage，用 micro-batch 流水化执行，降低单卡参数和 activation 压力。

## 上游概念

- [GPipe](../papers/gpipe.md)
- [PipeDream](../papers/pipedream.md)
- [Megatron 2021](../papers/megatron_2021.md)

## 关键机制

- Micro-batch
- Pipeline bubble
- 1F1B schedule
- Interleaved pipeline

## 生产关注

- stage imbalance 会直接制造 straggler。
- micro-batch 数量影响 bubble 和 activation memory。
- PP 与 checkpoint/recompute 绑定很深。

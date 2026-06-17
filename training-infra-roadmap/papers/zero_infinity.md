# ZeRO-Infinity

- 时间：2021
- 链接：https://arxiv.org/abs/2104.07857
- 状态：TODO 深读

## 工程阅读重点

ZeRO-Infinity 把 GPU/CPU/NVMe 组成分层内存系统。重点看 offload engine、prefetch、带宽隐藏和 NVMe IO 对训练吞吐的影响。

## 与知识图谱的关系

[ZeRO](zero.md) → [ZeRO-Offload](zero_offload.md) → ZeRO-Infinity → [Checkpointing](../topics/checkpointing.md)。

## 待补问题

- NVMe offload 如何避免训练完全 IO bound？
- 分层内存下参数生命周期如何调度？
- 这种方案今天在 H100 集群中还有哪些价值？

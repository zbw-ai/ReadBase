# Fully Sharded Data Parallel

- 时间：持续演进
- 链接：https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/
- 状态：TODO 深读

## 工程阅读重点

FSDP 是 PyTorch 原生的参数/梯度/optimizer state 分片方案。重点看 wrap policy、reshard、prefetch、state dict 和与 ZeRO-3 的关系。

## 与知识图谱的关系

[ZeRO](zero.md) → [FSDP Topic](../topics/fsdp.md) → [Checkpointing](../topics/checkpointing.md)。

## 待补问题

- FSDP 与 ZeRO-3 的实现差异是什么？
- auto wrap policy 如何影响通信粒度？
- sharded state dict 如何支持恢复和重分片？

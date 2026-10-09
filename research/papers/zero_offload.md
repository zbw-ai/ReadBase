# ZeRO-Offload

- 时间：2021
- 链接：https://arxiv.org/abs/2101.06840
- 状态：TODO 深读

## 工程阅读重点

ZeRO-Offload 把部分 optimizer/gradient 状态放到 CPU，降低 GPU 显存门槛。重点看 PCIe 带宽、CPU optimizer、overlap 和 offload 粒度。

## 与知识图谱的关系

[ZeRO](zero.md) → ZeRO-Offload → [ZeRO-Infinity](zero_infinity.md) → 分层内存训练。

## 待补问题

- Offload 什么时候省钱，什么时候拖慢？
- CPU 内存和 PCIe 带宽如何成为瓶颈？
- 单机低成本训练与大集群训练的取舍是什么？

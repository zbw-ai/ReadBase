# ZeRO

## 核心问题

Data Parallel 中参数、梯度、optimizer state 在每卡复制，显存冗余严重。ZeRO 通过分阶段分片这些状态，把集群总显存变成可用资源。

## 上游材料

- [ZeRO 论文](../papers/zero.md)
- [ZeRO-Offload](../papers/zero_offload.md)
- [ZeRO-Infinity](../papers/zero_infinity.md)

## 与 FSDP 的关系

FSDP 可以看作 PyTorch 原生化的 ZeRO-3 类方案：参数按需 all-gather，使用后释放，并结合 shard checkpoint。

## 生产关注

- 参数 prefetch 和通信 overlap
- bucket sizing
- checkpoint shard 保存与恢复
- offload 带宽瓶颈

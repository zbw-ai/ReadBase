# FSDP

## 核心问题

在 PyTorch 原生训练栈中按模块分片参数、梯度和 optimizer state，降低单卡显存，同时保持相对自然的模型代码。

## 上游材料

- [ZeRO](../papers/zero.md)
- [FSDP 论文/文档占位](../papers/fsdp.md)

## 关键机制

- Flatten parameters
- all-gather before forward
- reduce-scatter after backward
- reshard after forward/backward
- sharded state dict

## 生产关注

- auto wrap policy 决定通信粒度。
- prefetch 策略影响 overlap。
- checkpoint 需要支持 world size 和 sharding strategy 变化。

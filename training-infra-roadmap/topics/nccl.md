# NCCL

## 核心问题

分布式训练的 collective 通信层。TP、DP、FSDP、MoE 都依赖 NCCL 或等价通信库实现 all-reduce、reduce-scatter、all-gather、all-to-all 等通信。

## 相关材料

- [Megatron-LM](../papers/megatron_lm.md)
- [MegaScale](../tech_reports/megascale.md)

## 关键机制

- all-reduce
- reduce-scatter
- all-gather
- send/recv
- topology-aware algorithm

## 生产关注

- NCCL timeout 往往是结果，不是根因。
- 需要结合 GPU、NIC、交换机、路由、拓扑、其他租户一起看。
- TP/PP/DP group placement 会改变通信压力。

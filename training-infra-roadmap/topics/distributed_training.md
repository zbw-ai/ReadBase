# Distributed Training

## 核心问题

把一个大模型训练任务拆到多 GPU、多节点甚至万卡集群上，同时保持收敛、吞吐、稳定性和可恢复性。

## 相关材料

- [Megatron-LM](../papers/megatron_lm.md)
- [ZeRO](../papers/zero.md)
- [Llama 3](../tech_reports/llama3.md)
- [MegaScale](../tech_reports/megascale.md)

## 关键维度

- parallelism：TP / PP / DP / FSDP / EP / CP
- memory：activation / parameters / gradients / optimizer states
- communication：collective / p2p / all-to-all / overlap
- reliability：checkpoint / restart / straggler / observability

## 生产关注

- 配置不是静态最优，模型规模、seq length、GPU 拓扑和网络都会改变最优点。
- 大规模训练必须把性能工程和可靠性工程一起设计。

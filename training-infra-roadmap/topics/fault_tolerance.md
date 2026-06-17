# Fault Tolerance

## 核心问题

万卡训练中故障是常态。平台必须让训练作业在节点、网络、存储、数据服务异常后可诊断、可恢复、可继续。

## 相关材料

- [MegaScale](../tech_reports/megascale.md)
- [Llama 3](../tech_reports/llama3.md)
- [Checkpointing](checkpointing.md)

## 关键机制

- failure detection
- coordinated shutdown/restart
- elastic or fixed-world recovery
- checkpoint frequency
- straggler mitigation

## 生产关注

- 不是所有异常都该立即重启；慢节点和坏节点要区分。
- checkpoint 频率要平衡恢复损失和保存开销。
- 故障复盘必须能关联作业、节点、网络、数据和代码版本。

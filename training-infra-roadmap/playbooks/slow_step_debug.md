# Slow Step Debug

## 现象

训练没有失败，但 step time 变慢、抖动或长尾严重。

## 影响范围

GPU 利用率、训练成本、SLA、checkpoint cadence。

## 第一时间处理

区分是 compute、communication、input pipeline、checkpoint 还是 straggler。

## 排查顺序

1. 看 step time 分布。
2. 拆解 forward/backward/optimizer/communication/checkpoint。
3. 对比 rank-level trace。
4. 检查数据加载和存储抖动。

## 定位命令

待补充。

## 日志关键字

- `step_time`
- `data_time`
- `all_reduce`
- `checkpoint`

## 可能根因

- 慢 rank。
- 网络拥塞。
- checkpoint 写入抖动。
- 数据 pipeline 不稳定。

## 修复方案

待补充。

## 如何验证恢复

观察 p50/p95/p99 step time 和 GPU utilization。

## 如何避免再次发生

建立 rank-level telemetry 和 straggler dashboard。

## 关联 Topics

- [Distributed Training](../topics/distributed_training.md)
- [NCCL](../topics/nccl.md)
- [Fault Tolerance](../topics/fault_tolerance.md)

## 关联 Papers / Reports / Blogs

- [MegaScale](../tech_reports/megascale.md)

## 关联 Experiments

待补充。

## 复盘问题

- 这次慢 step 是否可以被提前观测到？

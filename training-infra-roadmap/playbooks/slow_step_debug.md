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

<a id="retrospective-observability"></a>

## 复盘补充：GPU 利用率很高但 step 不再推进

来源：[B300 field report](https://arxiv.org/abs/2608.05944)，2026-09-22 历史重评。该报告为小规模、硬件相关经验；下面迁移的是排查顺序，不复制其功耗阈值。

- **症状与影响：** GPU utilization 高，step/样本消费计数停住，尤其发生于 epoch 尾部。先确认是所有 rank、某个 DP group，还是个别任务。
- **第一响应：** 保存每个 rank 的最后一步、batch/packing 计数、NCCL 日志和功耗时间线，再按平台已有流程隔离故障；不要用单个 utilization 指标判断仍在有效计算。
- **调查顺序：** rank 迭代数是否一致 → packing 后各 rank 是否提前耗尽 → collective 是否对齐 → CPU/数据等待 → checkpoint I/O。对 NFS 与本地缓存做相同 page-cache 条件下的 A/B，再决定是否迁移存储。
- **观测命令：** `nvidia-smi --query-gpu=timestamp,index,utilization.gpu,power.draw,memory.used --format=csv -l 1`；结合现有分布式日志中的 `NCCL`、`Watchdog`、`timeout`、epoch/batch/step 边界。多节点需在相同时间轴收集，功耗只能辅助定位。
- **修复与验证：** 若确认是迭代数不一致，按训练语义选择等长数据、合法 padding/drop 策略或适用的 uneven-input 协议；跑过原来出错的 epoch 边界，并验证样本覆盖与 loss 归一化，不能只跑短 smoke test。
- **预防与复盘：** 启动前检查分布式数据边界，监控 rank 进度而非只监控存活；回答“为何烟测没有触及尾部”“修复是否改变有效样本”。关联[月度复盘](../tracking/monthly_signal_2026-08.md)和[状态实验](../experiments/rl_state_boundaries.md#retrospective-test-cases)。本仓库未复现该 B300 故障。

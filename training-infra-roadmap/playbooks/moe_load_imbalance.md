# MoE Load Imbalance

## 现象

MoE 训练中 expert 负载不均，部分 rank 计算或 all-to-all 变慢，step time 抖动。

## 影响范围

MoE 吞吐、收敛稳定性、straggler、显存峰值。

## 第一时间处理

查看 expert token count、capacity overflow、all-to-all 时间和 slow rank。

## 排查顺序

1. 检查 routing 分布。
2. 检查 capacity factor。
3. 检查 expert parallel group。
4. 检查 all-to-all 网络拓扑。

## 定位命令

待补充。

## 日志关键字

- `expert`
- `capacity`
- `all_to_all`
- `load balance`

## 可能根因

- routing 偏置。
- capacity 设置过低或过高。
- EP group 与网络拓扑不匹配。

## 修复方案

待补充。

## 如何验证恢复

观察 expert token histogram、drop rate、all-to-all 时间和 loss。

## 如何避免再次发生

建立 MoE 专用负载均衡监控。

## 关联 Topics

- [MoE](../topics/moe.md)
- [NCCL](../topics/nccl.md)

## 关联 Papers / Reports / Blogs

- [DeepSeek-V3](../tech_reports/deepseek_v3.md)

## 关联 Experiments

待补充。

## 复盘问题

- load imbalance 是数据导致，还是路由/拓扑导致？

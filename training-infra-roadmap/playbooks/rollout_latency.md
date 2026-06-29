# Rollout Latency

## 现象

Agentic RL / reasoning RL 中 rollout 生成变慢，policy update 等待样本，GPU 利用率下降。

## 影响范围

RL 训练吞吐、样本新鲜度、训练成本、评估周期。

## 第一时间处理

区分瓶颈在 inference worker、KV cache、reward/verifier、scheduler 还是数据写入。

## 排查顺序

1. 检查 rollout queue 长度。
2. 检查生成 token/s。
3. 检查 reward/verifier latency。
4. 检查 policy update 等待时间。

## 定位命令

待补充。

## 日志关键字

- `rollout`
- `queue`
- `reward`
- `verifier`
- `latency`

## 可能根因

- 长上下文导致 KV cache 压力。
- verifier 成为串行瓶颈。
- scheduler 没有平衡 freshness 和 throughput。

## 修复方案

待补充。

## 如何验证恢复

观察 rollout throughput、policy idle time、reward latency 和 end-to-end iteration time。

## 如何避免再次发生

为 rollout / verifier / update 建立分段 SLO。

## 关联 Topics

- [Distributed Training](../topics/distributed_training.md)
- [Context Parallelism](../topics/context_parallelism.md)

## 关联 Papers / Reports / Blogs

- [DeepSeek-R1](../tech_reports/deepseek_r1.md)

## 关联 Experiments

待补充。

## 复盘问题

- rollout 系统是否应该和训练系统解耦调度？

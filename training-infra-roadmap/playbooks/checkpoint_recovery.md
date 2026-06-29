# Checkpoint Recovery

## 现象

训练失败后需要从 checkpoint 恢复，或恢复后 loss spike / rank mismatch / optimizer state 不一致。

## 影响范围

训练连续性、实验可复现性、资源利用率。

## 第一时间处理

确认 checkpoint 完整性、global step、parallelism metadata 和 config/tokenizer 是否匹配。

## 排查顺序

1. 检查 checkpoint manifest。
2. 检查 shard 数量和 rank 数。
3. 检查 optimizer state。
4. 检查 RNG / dataloader state。

## 定位命令

待补充。

## 日志关键字

- `missing key`
- `unexpected key`
- `shape mismatch`
- `optimizer state`

## 可能根因

- checkpoint 写一半失败。
- 并行配置变化后没有 reshard。
- tokenizer/config 不匹配。

## 修复方案

待补充。

## 如何验证恢复

恢复后对比 loss、LR、global step、数据位置和短窗口 step time。

## 如何避免再次发生

定期恢复演练，保存 checksum，明确 checkpoint retention policy。

## 关联 Topics

- [Checkpointing](../topics/checkpointing.md)
- [Fault Tolerance](../topics/fault_tolerance.md)

## 关联 Papers / Reports / Blogs

- [MegaScale](../tech_reports/megascale.md)

## 关联 Experiments

- [Async Checkpoint](../experiments/checkpoint/async_checkpoint.md)

## 复盘问题

- 为什么直到恢复时才发现 checkpoint 不可用？

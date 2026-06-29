# TP NCCL Hang

## 现象

训练卡住，无明显 Python exception，部分 rank 停在 TP group collective。

## 影响范围

Tensor Parallel group 内所有 rank，可能扩散为全局 step hang。

## 第一时间处理

- 保留日志和 rank mapping。
- 不要只重启作业，先确认是否是确定性配置问题。

## 排查顺序

1. 检查 TP group 配置。
2. 检查 rank mapping 是否跨节点异常。
3. 检查 NCCL topology 和网络错误。
4. 检查 shape 是否在不同 rank 不一致。

## 定位命令

待补充。

## 日志关键字

- `NCCL WARN`
- `NET/IB`
- `all_reduce`
- `timeout`

## 可能根因

- TP group 跨慢速网络。
- rank mapping 错误。
- 某个 rank shape mismatch 导致 collective 等待。

## 修复方案

待补充。

## 如何验证恢复

待补充。

## 如何避免再次发生

待补充。

## 关联 Topics

- [Tensor Parallelism](../topics/tensor_parallelism.md)
- [NCCL](../topics/nccl.md)

## 关联 Papers / Reports / Blogs

- [Megatron-LM](../papers/megatron_lm.md)

## 关联 Experiments

- [TP vs DP](../experiments/tensor_parallelism/tp_vs_dp.md)

## 复盘问题

- 为什么监控没有提前发现通信异常？

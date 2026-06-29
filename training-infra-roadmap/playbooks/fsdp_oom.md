# FSDP OOM

## 现象

使用 FSDP 后仍然 OOM，或在 forward all-gather / backward reduce-scatter / optimizer step 阶段 OOM。

## 影响范围

单作业稳定性、batch size、吞吐和恢复成本。

## 第一时间处理

确认 OOM 发生阶段，而不是只看总显存峰值。

## 排查顺序

1. 检查 wrapping policy。
2. 检查 mixed precision。
3. 检查 activation checkpointing。
4. 检查 state_dict / checkpoint 导出路径。

## 定位命令

待补充。

## 日志关键字

- `CUDA out of memory`
- `all_gather`
- `FullStateDict`

## 可能根因

- wrap 粒度不合理。
- 临时 all-gather 峰值过高。
- activation 占比被低估。

## 修复方案

待补充。

## 如何验证恢复

记录 peak memory、step time、重启恢复是否稳定。

## 如何避免再次发生

为不同模型规模建立 FSDP 配置基线。

## 关联 Topics

- [FSDP](../topics/fsdp.md)
- [Checkpointing](../topics/checkpointing.md)

## 关联 Papers / Reports / Blogs

- [FSDP Paper](../papers/fsdp.md)

## 关联 Experiments

待补充。

## 复盘问题

- 为什么估算显存与真实峰值不一致？

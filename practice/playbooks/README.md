# Playbooks

`playbooks/` 是生产排障和操作指南。

它和 `topics/` 的区别：

- `topics/` 回答：这个技术是什么，为什么重要，如何设计。
- `playbooks/` 回答：线上出问题时，我应该按什么顺序排查、验证和恢复。

## Playbook Template

```text
# Title

## 现象

## 影响范围

## 第一时间处理

## 排查顺序

## 定位命令

## 日志关键字

## 可能根因

## 修复方案

## 如何验证恢复

## 如何避免再次发生

## 关联 Topics

## 关联 Papers / Reports / Blogs

## 关联 Experiments

## 复盘问题
```

## Current Playbooks

- [TP NCCL Hang](tp_nccl_hang.md)
- [Checkpoint Recovery](checkpoint_recovery.md)
- [FSDP OOM](fsdp_oom.md)
- [Slow Step Debug](slow_step_debug.md)
- [MoE Load Imbalance](moe_load_imbalance.md)
- [Rollout Latency](rollout_latency.md)

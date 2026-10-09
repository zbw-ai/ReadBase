# P0 R5 Lineage Closure Design

Status: `IMPLEMENTED / R5_G4_G5_PASS`

Date: 2026-07-23

## Problem

R4 已证明 manager 能形成并导出完整 cohort，但首个 trainer batch 在 lineage detach 阶段失败：

```text
trajectory 0 has 8 records for batch size 0
```

真实 batch 并不为空。single-controller 路径通过 RPC 返回 `RTensor`，其 `data` 是保留 shape 的 meta tensor；现有 `get_batch_size()` 只识别本地 `torch.Tensor`，因此误判为 0。与此同时，participation log 环境变量只传播给 actor workers，没有传播给 rollout workflow workers，导致 `manager_exported=16` 但 `workflow_exported=0`。

## Chosen Design

1. `get_batch_size()` 识别 `RTensor.data.shape[0]`，只读取 metadata，不调用 `RTensor.localize()`，避免把 128K tensor 拉回 controller。
2. `detach_trajectory_lineage()` 和 `attach_sequence_lineage_indices()` 继续执行严格 row-alignment 校验；不能用 lineage 长度反向伪造 batch size。
3. participation log path 与 run ID 同时传播给 actor 和 rollout scheduling specs，使 workflow 与 trainer 写入同一 sidecar。
4. R4 残留 eval runner 先清理并记录进程树。只有能在受控测试中复现 supervisor 泄漏，才修改 launcher，避免把重复启动或手工 detach 问题误诊为脚本缺陷。
5. R5 保持 bs2、group size 8、128K、32 卡和 sampling 不变，只验证 P0 closure。

## Rejected Alternatives

- **在 controller localize 整个 batch**：实现简单，但引入大规模 tensor 下载、controller 内存占用和额外同步，不符合训练控制面设计。
- **直接信任 `len(trajectory_lineage)`**：会让缺失 tensor 的坏 batch 通过，破坏 fail-closed correctness。
- **把 lineage detach 整体移动到 actor worker**：长期可能更整洁，但会扩大 controller/worker API 和 writer ownership 改动，不适合当前 P0。

## Acceptance Gates

| Gate | Requirement |
|---|---|
| P0.1 | `RTensor` batch size 正确识别；不触发 remote fetch |
| P0.2 | actor 与 rollout specs 获得相同 log path/run ID |
| P0.3 | R4 残留清理；R5 只存在一套 launcher/trainer/eval process tree |
| R5-G4 | 至少两个完整 cohort 进入首个 optimizer update |
| R5-G5 | `trainer_consumed = 16`；这 16 个 UID 必须逐一属于 manager/workflow exported 集合。异步 prefetch 允许上游 aggregate 大于 16 |
| Audit | UID strict join 100%；token conservation 成立；behavior-version coverage 100%；UNKNOWN participation 为 0 |

R5 必须使用唯一 `AREAL_RUN_ID` 和启动前不存在的全新 sidecar 路径；analyzer 禁止跨 run 聚合。上述任一 gate 不满足，R5 失败且 P1 retry/replacement 保持 `HOLD`。

R5 实测为 manager/workflow/trainer `24 / 24 / 16`：step 0 消费 `idx-0` 与 `idx-3`，额外 8 条是
prefetch 已导出的 `idx-5`。trainer 的 16 个 UID 均 strict join 到上游事件，证明集合 gate 能区分正常流水线
ahead 与真正的 UID 泄漏。

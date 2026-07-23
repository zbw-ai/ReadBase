# Infra Trends

用于记录 AI Training Infra 的技术演进时间线。这里写趋势，不写资料摘要。

## 关注方向

- Dense scaling → MoE scaling
- TP/PP/DP → SP/CP/EP 组合
- BF16 → FP8 / NVFP4
- synchronous checkpoint → async / distributed / elastic checkpoint
- pretraining-only platform → pretraining + SFT + RL + evaluation pipeline
- short context → long context / context parallel
- offline training → agentic rollout + verifier + scheduler
- trainer-centric RL → rollout / training / scheduler / weight-sync service architecture

## Timeline

### 2026 RL Framework Evolution

这张表不是 release 全量清单，只记录会改变 RL Infra 工程判断的节点。证据优先级为：official release > merged major PR > 有实现证据的 open PR。普通 commit、文档更新和仅有宣传材料的项目不进入主线。

| 时间 | Framework milestone | 主要子系统 | 工程信号 | 证据入口 |
|---|---|---|---|---|
| 2026-01 | verl v0.7.0；slime v0.2.2；NeMo RL v0.5.0 | rollout / scheduler / weight sync / precision | rollout server、fully async、Router Replay、inflight refit、FP8 rollout 开始成为框架正式能力 | [January audit](monthly_signal_2026-01.md#rl-framework-monthly-highlights-historical-audit) |
| 2026-02 | ROLL v0.2.0 | bidirectional resource scheduling / rollout | 训练空闲 GPU 可阶段性切到 rollout，异步 RL 的资源分区从静态配置走向运行时调度 | [February audit](monthly_signal_2026-02.md#rl-framework-monthly-highlights-historical-audit) |
| 2026-03 | AReaL v1.0；verl v0.7.1；slime v0.2.3/0.2.4；ROLL v0.2.1 | control plane / inference backend / routing / observability | single-controller、统一 checkpoint engine、PD/EPD、affinity routing 与 rollout timeline 同时成熟 | [March audit](monthly_signal_2026-03.md#rl-framework-monthly-highlights-historical-audit) |
| 2026-04 | AReaL v1.0.3；NeMo RL v0.6.0；OpenRLHF v0.10.0 | agent service / speculative rollout / long context / async sampling | RL pipeline 开始显式管理 agent service、drafter、长上下文内存和 oversampling | [April audit](monthly_signal_2026-04.md#rl-framework-monthly-highlights-historical-audit) |
| 2026-05 | AReaL v1.0.4；slime v0.3.0；OpenRLHF v0.10.3 | async training / trajectory path / weight sync / loss correctness | variable global batch、fully async、delta sync 进入正式路径；变长序列 loss aggregation 暴露为跨框架正确性风险 | [May audit](monthly_signal_2026-05.md#rl-framework-monthly-highlights-historical-audit) |
| 2026-06 | verl v0.8.0；ROLL v0.3.0；OpenRLHF v0.10.4；Miles | data/control plane / agent runtime / tracing / recovery | RL framework 从 recipe 集合转向可组合 backend、流式数据通道、Agent runtime、observability 与 fault tolerance | [June audit](monthly_signal_2026-06.md#rl-framework-monthly-highlights-historical-audit) |
| 2026-07-01 ~ 07-04 | AReaL v2.0；slime delta disk sync | service architecture / weight sync | 训练、推理、Agent、weight update 形成微服务边界；disaggregated rollout 开始采用 delta refit | [07-04 audit](frontier_scan_2026-07-04.md#rl-framework-watch-historical-audit-backfill) |
| 2026-07-07 ~ 07-10 | slime engine pull；verl receive-buffer correctness、PD rollout、Megatron Bridge | weight sync / inference backend / training | pull-based refit、NIXL/Mooncake PD rollout 与权重传输正确性成为一条连续工程主线 | [07-08](frontier_scan_2026-07-08.md#rl-framework-watch-historical-audit-backfill) / [07-10](frontier_scan_2026-07-10.md#rl-framework-watch-historical-audit-backfill) |
| 2026-07-10 ~ 07-20 | verl TransferQueue rollback；verl/NeMo RL checkpoint recovery；NCCL delta/refit | data path / checkpoint / recovery / weight sync | 架构抽象开始接受恢复语义检验：能传数据不等于能正确背压、重启和保留 policy version | [07-13](frontier_scan_2026-07-13.md#rl-framework-watch-historical-audit-backfill) / [07-20](frontier_scan_2026-07-20.md#rl-framework-watch-historical-audit-backfill) |
| 2026-07-20 ~ 07-23 | verl dynamic scheduling；NeMo RL NIXL refit/prompt-group streaming；AReaL HTTP scheduler | scheduler / rollout / control plane | 动态资源调度、流式 prompt-group 消费和 pluggable refit backend 开始汇合，直接面向长尾 rollout 与服务化控制面 | [07-22](frontier_scan_2026-07-22.md#rl-framework-watch-historical-audit-backfill) / [07-23](frontier_scan_2026-07-23.md#rl-framework-watch) |

### 当前框架判断

1. **AReaL** 最值得作为主研究对象：它同时暴露 async RL、service boundary、weight update、staleness 和 Agent workflow，适合做代码级优化。
2. **verl** 是最重要的对照组：backend 覆盖广、架构变化快，也提供了 TransferQueue 回滚这类非常有价值的负面证据。
3. **slime** 对 Agent-first、长轨迹、variable batch、SGLang/PD rollout 和 delta sync 的推进最集中。
4. **ROLL** 在双向资源借用、AgentRunner、RemoteBatch 与 OpenTelemetry 上提供了不同实现路线。
5. **NeMo RL** 适合跟踪 NVIDIA 训练栈下的 FP8、Megatron、refit、speculative rollout、checkpoint/recovery 与规模化运维。
6. **OpenRLHF** 架构更轻，但 token-level loss aggregation 的连续修复说明它对正确性回归仍有高参考价值。
7. **Emerging frameworks** 只有在具备真实代码、可运行训练路径或可复核 benchmark 时进入 watchlist；Miles 是当前已保留的代表。

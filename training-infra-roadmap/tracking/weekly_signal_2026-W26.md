# Weekly Signal Report, 2026-W26

- Window: 2026-06-22 00:00:00 ~ 2026-06-28 23:59:59
- Timezone: Asia/Shanghai
- Generated at: 2026-06-29
- Report type: weekly frontier radar

## 本周核心判断

本周没有收录合格的 **frontier weekly signal**。

原因不是“没有值得学习的材料”，而是候选材料主要属于历史补课：AReaL、verl / HybridFlow、Agent Lightning、OpenRLHF、DAPO、DeepSpeed-Chat、DeepSeek-R1、vLLM + OpenRLHF Integration 等都很重要，但它们不是本周新出现的前沿信号。

这些材料已经进入 [Historical Backfill](historical_backfill.md) 和 [Reading Queue](../reading_queue/README.md)，不再伪装成本周 signal。

## 本周筛选标准

一条材料要进入 weekly signal，至少满足以下条件：

- 最近 7 到 14 天发布、更新或引发明确讨论；最多放宽到 30 天。
- 有可核验来源：paper、repo、release note、official blog、benchmark 或技术报告。
- 不是单纯“主题相关”，而是可能改变未来 3 到 12 个月的工程判断。
- 能说明它影响哪个系统边界：parallelism、checkpoint、FP8、MoE、rollout、serving/training interface、scheduler、NCCL、observability。
- 如果证据弱，就标记为 Observe；如果质量不够，就不收录。

## Accepted Frontier Signals

本窗口未收录 accepted frontier signal。

本周不强行凑数量。宁缺毋滥。

## Deferred / Backfilled Candidates

以下材料重要，但更适合进入 historical backfill，而不是 weekly signal：

| 材料 | 处理方式 | 原因 |
|---|---|---|
| AReaL | [Historical Backfill](historical_backfill.md) / [P0](../reading_queue/P0.md) | 重要但不是本周新信号，补异步 rollout/train 解耦 |
| verl / HybridFlow | [Historical Backfill](historical_backfill.md) / [P0](../reading_queue/P0.md) | 重要但不是本周新信号，补 RLHF dataflow 和 actor resharding |
| Agent Lightning | [Historical Backfill](historical_backfill.md) / [P0](../reading_queue/P0.md) | 重要但不是本周新信号，补 agent runtime 与 trainer 解耦 |
| OpenRLHF | [Historical Backfill](historical_backfill.md) / [P1](../reading_queue/P1.md) | 补 Ray + vLLM + DeepSpeed 多组件调度 |
| vLLM + OpenRLHF Integration | [Historical Backfill](historical_backfill.md) / [P1](../reading_queue/P1.md) | 补 rollout inference / weight sync / placement group |
| SkyRL | [Historical Backfill](historical_backfill.md) / [P1](../reading_queue/P1.md) | 补 long-horizon tool-use agent training |
| DAPO | [Historical Backfill](historical_backfill.md) / [P1](../reading_queue/P1.md) | 补 reasoning RL recipe 如何落到系统栈 |
| NVIDIA NeMo RL | [Historical Backfill](historical_backfill.md) / [P1](../reading_queue/P1.md) | 补 NVIDIA post-training stack 演进 |
| RLHFless | [Historical Backfill](historical_backfill.md) / Observe | 方向有意思，但生产成熟度需要观察 |
| AReaL-Hex | Observe | 异构 GPU 异步 RL 调度值得跟踪，但应先读 AReaL 主线 |

## Reading Queue Updates

本周没有由 weekly signal 产生新的 P0。

当前 P0 来自 historical backfill：

- [AReaL](../reading_queue/P0.md)
- [HybridFlow / verl](../reading_queue/P0.md)
- [Agent Lightning](../reading_queue/P0.md)

当前 P1 来自 historical backfill：

- [OpenRLHF](../reading_queue/P1.md)
- [vLLM + OpenRLHF Integration](../reading_queue/P1.md)
- [SkyRL](../reading_queue/P1.md)
- [DAPO](../reading_queue/P1.md)
- [NVIDIA NeMo RL](../reading_queue/P1.md)

## 本周观察

这次修正明确了一条规则：weekly signal 是前沿雷达，不是补课清单。

Agentic RL / Rollout Infra 确实是当前最值得补的方向，但它应通过 historical backfill 进入系统。等未来出现新的论文、repo、release note 或官方技术博客，且它真的改变 rollout、scheduler、serving/training interface 或 post-training stack 的判断时，再进入 weekly signal。

## 下一步动作

- [x] 将 AReaL / verl / Agent Lightning 作为 historical backfill 的 P0，而不是 weekly signal 的 P0。
- [x] 将 OpenRLHF / vLLM + OpenRLHF / SkyRL / DAPO / NeMo RL 作为 historical backfill 的 P1。
- [x] 保留本文件作为 2026-W26 的 corrected weekly signal record。
- [ ] 下次生成 weekly signal 时只接受最近前沿材料，允许 0 条 accepted signal。

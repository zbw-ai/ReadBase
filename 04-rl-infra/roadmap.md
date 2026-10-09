# RL / Agentic RL 基础设施核心学习路线

[所属 Part IV：RL 基础设施](README.md) · [首页](../README.md)

## 学习目的

把 RL 系统理解为持续生产、评估、消费 trajectory 的分布式系统，并知道算法假设怎样约束数据与状态。
主线先保证离线固定轨迹的更新正确，再进入在线、异步和扩展；不从框架功能表或异步开关开始。
最终目标是解释有效训练供给、策略新鲜度与恢复正确性，而不是只优化 rollout token/s。

## 先修与起点

- 能说明 logprob、policy、reward、advantage 的角色，并理解一次梯度更新。
- 具备 [训练后端](../02-training-infra/README.md)与 [推理路线](../03-inference-infra/roadmap.md)基础，知道 actor update 和生成的资源需求不同。
- 先了解 [分布式系统](../01-systems/roadmaps/distributed-systems.md)中的身份、背压、部分失败和版本准入。

## 六阶段主线

“已有”表示机制或案例已写入正文，不代表迁移通过实验；项目历史证据与本路线的未执行练习分开记录。

| 阶段 | 核心问题 | 已有正文与覆盖状态 | 验证 / 掌握标准 |
|---|---|---|---|
| 1. 算法与数据契约 | PPO / GRPO 的更新需要什么信息，哪些 token 真正进入 loss？ | [PPO / GRPO / DAPO](topics/agentic_rl.md#ppo-grpo-dapo)有概念主线；[统计与状态边界](topics/agentic_rl.md#rl-state-boundaries)有不变量；完整目标函数推导与最小实现待补 | 对固定小轨迹列出 token、behavior logprob、reward、advantage、mask 与版本；分清 old policy、reference 与 critic，说明 group / token 归一化 |
| 2. 环境与轨迹生命周期 | 一次模型调用如何成为可训练 episode，失败与 reward 如何归属？ | [外部 Agent Gateway](topics/agentic_rl.md#external-agent-gateway)与 [环境契约](topics/agentic_rl.md#mimo-v26-environment-contract)已有案例；通用环境接口教程待补 | 画 reset → action / tool → observation → terminal / truncated → reward → export，保持 task / session / interaction 身份与 token lineage |
| 3. Rollout → reward → train | 哪个环节限制有效供给，哪里需要缓冲与背压？ | [Agentic RL 系统形态](topics/agentic_rl.md)、[供给与调度](topics/agentic_rl.md#gateway-streaming-refill)、[框架选型](topics/rl_framework_selection.md)已有 | 分开 generated、completed、accepted、consumed 与 gradient-active；按 role 和队列定位等待，不把低拒收率单独当作训练有效性 |
| 4. Sync / async 与 staleness | 去掉 barrier 后增加了哪些算法与状态成本？ | [异步、streaming、partial 与 staleness](topics/agentic_rl.md#async-streaming-partial-staleness)已有机制正文 | 画同步基线和异步时间线；明确 queue 上限、版本差、轨迹年龄、拒收 / mask / correction 规则，同时观察训练效果 |
| 5. Weight sync 与准入 | 训练态权重怎样成为可安全生成的新策略？ | [XCCL / disk 发布](topics/agentic_rl.md#areal-weight-sync-xccl-disk)与 [恢复准入](topics/agentic_rl.md#rl-state-boundaries)已有 | 分解 collect / convert、transfer、load / refit、exposed pause；核验每个 replica 的版本和 logprob，不把 transfer artifact 当恢复 checkpoint |
| 6. Recovery 与 goodput | 中断后怎样恢复同一数据 / 策略边界，长期有效产出如何计量？ | [状态不变量](topics/agentic_rl.md#monthly-retrospective-invariants)、[Checkpointing](../02-training-infra/topics/checkpointing.md)与 [项目指标契约](../practice/projects/2026-q3-long-context-agentic-rl/instrumentation/metric_contract.md)已有；本路线故障注入未执行 | 对齐数据 cursor、消费 frontier、buffer、policy / optimizer 状态；记录丢弃与重放语义，以质量门槛下的有效工作量和总资源成本验收 |

DAPO 等 recipe 在算法基础后比较；partial rollout、异步和复杂 placement 不是必选项，只有同步基线暴露明确问题后才增加复杂度。

## 最小练习与产物

统一入口为 [E08 RL 数据、策略版本与权重同步](../practice/experiments/a100_fsdp_io_lab.md#e08)，目前未执行。

1. **固定轨迹：先验证更新。**

   用小数据逐项核对 token / logprob / reward / mask；改变 padding 或 packing 后，有效样本语义应保持。
   先约定数值容差和统计分母，再考虑在线生成；loss 下降不是契约正确的充分证据。

2. **完整 episode：追踪身份。**

   设计含工具调用、超时和 terminal reward 的最小轨迹，区分任务失败、基础设施错误与正常截断。
   产物是生命周期表；接口返回文本不等于已保留可训练的 token / behavior logprob。

3. **同步在线：建立供给账本。**

   将 rollout、reward 和 train 串起来，按相同样本身份记录阶段时间、queue 与拒收原因。
   先解释 trainer idle 来自哪里，再评估增加并发、调资源比例或改 verifier 是否合理。

4. **异步对照：只改一个边界。**

   保持任务、算法与总资源口径可比，只改变供给调度或同步 cadence；记录版本差和样本年龄分布。
   不把版本差当成精确 policy divergence；结合 logprob / ratio、训练质量和独立评测解释结果。

5. **版本发布与恢复：注入一个失败点。**

   先用 mock 推演 refit 部分成功或 checkpoint 未提交，再依据 [状态边界计划](../practice/experiments/rl_state_boundaries.md)选择独立集成测试。
   必须说明哪些 worker 禁入、哪些样本重放 / 丢弃，以及恢复后的消费与版本账本如何重新对齐。

## 执行与覆盖边界

- A100 E00–E08、[RL 状态边界实验](../practice/experiments/rl_state_boundaries.md)和 [环境 / mixer 验证](../practice/experiments/mimo_v26_environment_and_mixer.md)均未执行。
- 本路线不申请 GPU、不创建集群任务；mock producer 也不等于已实现真实环境或机器人在线 RL。
- 框架与 backend 的支持矩阵受版本影响；[选型正文](topics/rl_framework_selection.md)中的历史判断不能直接推广到任意 release。
- 对 PPO / GRPO 变体，按实际 estimator、loss aggregation、normalization 与 correction 检查，不强行套同一个 group 下限。
- 单模型生成变快、样本接收更多、trainer 更忙，都可能没有增加有效更新或最终任务质量；各自统计分母必须公开。
- [历史项目](../practice/projects/2026-q3-long-context-agentic-rl/README.md)仅作证据组织示例；实际状态以其 [STATUS](../practice/projects/2026-q3-long-context-agentic-rl/STATUS.md)日期为准，不转写为本次练习结果。

## 主线之后去哪里

- 对 Teacher 服务和多领域蒸馏有具体需求后再读 [MOPD](topics/mopd.md)，不作为 RL 基础的必修前置。
- 进入具身环境前先补 [具身模型与基础设施](../05-embodied-infra/README.md)的动作、观测、评估与安全边界。
- 遇到供给长尾回到 [Rollout Latency](../practice/playbooks/rollout_latency.md)，遇到状态不连续回到 [Checkpoint Recovery](../practice/playbooks/checkpoint_recovery.md)。

最终产物是一份可追踪“样本从哪里来、由哪个策略生成、为何被消费、失败后怎样继续”的系统账本。

[返回 Part IV](README.md) · [返回首页](../README.md)

# Part V｜具身智能模型与基础设施

这一部分先补模型、任务与动作闭环的基础，再理解数据采集、仿真、训练、部署和安全边界。当前模型入门正文尚未创建；已有的 Agentic for Embodied 是进阶系统地图与平台设计，不能替代模型 primer，也未经过真实机器人或仿真实验验证。

## 从哪里开始

**当前主线：[具身 Models & Infra 六阶段路线](roadmap.md)**。按“模型与动作 → episode → 存储/I/O → 训练 → 评估 → 部署”推进；每阶段都有问题、正文入口、最小产出与证据边界。

1. [具身模型入门大纲](../docs/superpowers/specs/2026-09-18-embodied-models-primer-design.md)：这是待展开的大纲，先厘清 MLLM、VLA、policy、world model 分别解决什么问题。
2. [Agentic for Embodied](topics/agentic_for_embodied.md)：在具备最小机器人术语后，阅读 observation、action、trajectory 与实时闭环的系统边界。
3. [Agentic RL 数据与状态主线](../rl-infra/topics/agentic_rl.md)：比较哪些 rollout / version / recovery 能力可复用，哪些物理系统约束需要重建。
4. [实践与证据规则](../practice/README.md)：先定义可验证问题和安全边界；现阶段不把设计蓝图标为实验结果。

## 问题到验证

| 工程问题 | 正文 / 大纲入口 | 实验 / 排障入口 | 当前状态 |
|---|---|---|---|
| 模型理解指令，为什么不等于任务完成？ | [模型入门大纲](../docs/superpowers/specs/2026-09-18-embodied-models-primer-design.md) | 暂无 | 大纲草案；基础正文待写 |
| trajectory、时间戳与动作时限怎样进入系统设计？ | [Agentic for Embodied](topics/agentic_for_embodied.md) | 暂无具身专属实验 / runbook | `NEW`；已有系统地图，未验证 |
| 模型不大，是否还需要状态分片？ | [阶段④](roadmap.md#stage-4) · [FSDP/FSDP2/ZeRO](../training-infra/topics/fsdp.md) | [E01／E02](../practice/experiments/a100_fsdp_io_lab.md#e01) | 先 DDP 基线与显存账本；未执行 |
| 视频与低维信号怎样形成正确且高效的 batch？ | [阶段②](roadmap.md#stage-2) → [阶段③](roadmap.md#stage-3) | [E04 多模态 I/O](../practice/experiments/a100_fsdp_io_lab.md#e04) | 独立数据正文待补；计划未执行 |
| RL Infra 的状态管理能复用到哪里？ | [进阶平台设计](topics/agentic_for_embodied.md) · [RL 状态边界](../rl-infra/topics/agentic_rl.md#rl-state-boundaries) | [RL 故障注入计划](../practice/experiments/rl_state_boundaries.md)仅作方法参考 | RL 计划也未执行，不构成具身验证 |

## 继续深入

优先补齐模型与任务认识，再深化平台设计；相关来源从[研究阅读总表](../research/MASTER_READING_LIST.md)和[阅读队列](../research/reading_queue/README.md)进入。本页不新增模型 benchmark，也不将原有关注日期重置为目录迁移日期。

[返回总入口](../README.md) · [最近更新](../README.md#recent-updates) · [知识图谱](../KNOWLEDGE_GRAPH.md) · [上一部分：RL 基础设施](../rl-infra/README.md) · [下一部分：工程实践](../practice/README.md)

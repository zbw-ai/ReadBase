# Part 05｜具身智能模型与基础设施

**个人学习从[六阶段、八次学习指南](roadmap.md#study-sessions)开始。** 不必逐份读扫描：模型与动作 → 样本和 I/O → 训练 → 仿真评估 → 部署执行，最后按需进入平台设计。

这一部分先补模型、任务与动作闭环的基础，再理解数据采集、训练、仿真与部署。2026-10-23 已补成模型、数据、训练、评估与执行五块具体正文，包含计算示例、配置判断与排障；Agentic for Embodied 保留为进阶平台设计。正文整理、读者掌握与实验结果分开记录，目前未执行仿真或真机验证。

## 从哪里开始

**核心 topic：[具身 Infra：从一个动作到完整学习系统](topics/embodied_infra.md)**。先看整体地图，再从问题进入基础正文；[讨论起点](topics/embodied_infra.md#discussion)串起后续学习。

**当前主线：[个人学习指南](roadmap.md)**。每阶段都有问题、指定正文、最小产出与证据边界；[E1–E6 学习练习](../practice/experiments/embodied_contract_lab.md)提供样本、loss、评估分母与执行时间的自测。

| 编号 | 阶段入口 | 正文入口与真实覆盖 |
|---|---|---|
| 5.1 | [模型与动作](roadmap.md#stage-1) | [具身模型入门](topics/embodied_models_primer.md)：[VLM／VLT／VLA](topics/embodied_models_primer.md#model-roles)、学习算法、world model 与 [Infra 核心关注点](topics/embodied_models_primer.md#infra-priorities) |
| 5.2 | [Episode 数据契约](roadmap.md#stage-2) | [数据契约基础](topics/embodied_data_io.md#sample-contract) → [平台 schema](topics/agentic_for_embodied.md)；未验证 |
| 5.3 | [存储与多模态 I/O](roadmap.md#stage-3) | [数据与 I/O 正文](topics/embodied_data_io.md) → [E04](../practice/experiments/a100_fsdp_io_lab.md#e04)；实验未执行 |
| 5.4 | [训练与状态分片](roadmap.md#stage-4) | [具身训练](topics/embodied_training.md)：冻结/梯度、mask、DDP 全局 loss、状态分片与恢复；复用训练 Part 的通信机制 |
| 5.5 | [闭环评估](roadmap.md#stage-5) | [仿真与评估](topics/embodied_sim_evaluation.md)：环境与时钟、终止/截断、12-case 分母、reset 吞吐与 sim-to-real |
| 5.6 | [部署与反馈](roadmap.md#stage-6) | [部署与执行](topics/embodied_runtime.md)：80 ms 时间线、动作队列、迟到响应、提交边界与部署工件 |

[Agentic RL 数据与状态主线](../04-rl-infra/topics/agentic_rl.md)用于比较哪些 rollout / version / recovery 能力可复用、哪些物理系统约束需要重建。按[实践与证据规则](../practice/README.md)先定义可验证问题和安全边界，现阶段不把设计蓝图标为实验结果。

## 问题到验证

| 工程问题 | 正文 / 大纲入口 | 实验 / 排障入口 | 当前状态 |
|---|---|---|---|
| 模型理解指令，为什么不等于任务完成？ | [模型入门](topics/embodied_models_primer.md) | [纸面契约练习](../practice/experiments/embodied_contract_lab.md) | 正文已整理；讨论与验证待开展 |
| trajectory、时间戳与动作时限怎样进入系统设计？ | [数据契约](topics/embodied_data_io.md#sample-contract) | [契约练习](../practice/experiments/embodied_contract_lab.md)尚未执行；runbook 待补 | `NEW`；已有系统地图，未验证 |
| 模型不大，是否还需要状态分片？ | [阶段④](roadmap.md#stage-4) · [FSDP/FSDP2/ZeRO](../02-training-infra/topics/fsdp.md) | [E01／E02](../practice/experiments/a100_fsdp_io_lab.md#e01) | 先 DDP 基线与显存账本；未执行 |
| 为什么加卡以后 loss 或动作尺度变化？ | [全局有效 loss](topics/embodied_training.md#loss-normalization) · [恢复](topics/embodied_training.md#recovery) | [E4](../practice/experiments/embodied_contract_lab.md#e4) | 参考推导已写；训练对照未执行 |
| 成功率是否隐藏了超时、接管或缺失结果？ | [评估账本](topics/embodied_sim_evaluation.md#episode-ledger) | [E5](../practice/experiments/embodied_contract_lab.md#e5) | 教学例子；没有真实 benchmark 结果 |
| 为什么同一模型的响应也可能不能执行？ | [reset 与迟到请求](topics/embodied_runtime.md#stale-response) | [E6](../practice/experiments/embodied_contract_lab.md#e6) | 请求契约设计；未运行机器人 |
| 视频与低维信号怎样形成正确且高效的 batch？ | [阶段②](roadmap.md#stage-2) → [阶段③](roadmap.md#stage-3) | [E04 多模态 I/O](../practice/experiments/a100_fsdp_io_lab.md#e04) | [数据正文](topics/embodied_data_io.md)已建立；计划未执行 |
| RL Infra 的状态管理能复用到哪里？ | [进阶平台设计](topics/agentic_for_embodied.md) · [RL 状态边界](../04-rl-infra/topics/agentic_rl.md#rl-state-boundaries) | [RL 故障注入计划](../practice/experiments/rl_state_boundaries.md)仅作方法参考 | RL 计划也未执行，不构成具身验证 |

## 继续深入

优先补齐模型与任务认识，再深化平台设计。看历史演进用[2026 月报时间线](../research/tracking/monthly_reviews.md#embodied-2026)，理解技术用上方正文，选论文用[研究阅读总表](../research/MASTER_READING_LIST.md)和[共享队列](../research/reading_queue/README.md)。本页不新增模型 benchmark，也不将整理完成当个人掌握。

[返回总入口](../README.md) · [最近更新](../README.md#recent-updates) · [知识图谱](../KNOWLEDGE_GRAPH.md) · [上一部分：RL 基础设施](../04-rl-infra/README.md) · [下一部分：工程实践](../practice/README.md)

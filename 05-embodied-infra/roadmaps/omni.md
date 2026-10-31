# Omni 算法与 Infra 学习路线

[Part 05](../README.md) · [核心正文](../topics/omni_models_infra.md) · [报告笔记](../../research/tech_reports/qwen35_omni.md) · [实验册](../../practice/experiments/omni_sft_rl_lab.md)

**2026-10-31 建立。** 面向已有 LLM Training / Agentic RL 经验的 Infra 工程师，围绕 Omni 后训练和系统优化补模型知识。路线按掌握标准推进，不要求先完成整个具身方向；阅读优先级只在[共享 P1](../../research/reading_queue/P1.md#omni-20261031)维护，原 P0 在读状态保留。

本路线以音视频理解与语音交互为默认任务面；实际模型、硬件、SFT/RL 栈和是否有动作输出需在工作中确认。公开页只记录通用方法与公开来源，具体内部需求/实验数据沿用[项目手册的记录边界](../../practice/projects/2026-10-agibot-onboarding.md)。

## 第一轮学习：六阶段，每次带一个产出

| 阶段 / 建议投入 | 要回答的问题 | 指定正文与练习 | 掌握标准 |
|---|---|---|---|
| 1 模型和任务，2 次 × 60–90 min | 输入怎么变成表示，Thinker/Talker 分别生成什么？ARIA 为何出现？ | [报告架构](../../research/tech_reports/qwen35_omni.md) + [核心正文](../topics/omni_models_infra.md)；O0 | 画一个音视频问答与语音回复样本的 shape/时间线，区分 encoder frame、text token、codec 与物理动作 |
| 2 SFT 数据与目标，2 次 × 60–90 min | 哪些模块更新？哪些位置进入 loss？ | [SFT](../topics/omni_models_infra.md#sft) + [全局 loss](../topics/embodied_training.md#loss-normalization)；O1 | 手算不等长样本的全局 loss；定义可复现 media/processor/mask；列出 packed 与 unpacked 等价条件 |
| 3 SFT 系统基线，2 次 × 60–90 min | 数据、activation、MoE 通信哪个占关键路径？ | [长上下文](../../02-training-infra/topics/long_context_training.md)、[FSDP](../../02-training-infra/topics/fsdp.md)、[MoE](../../02-training-infra/topics/moe.md)按需；O2/O3 | 产出容量账本和按 stage 分解的 trace 方案；选择一个可证伪优化，不凭利用率直接加卡 |
| 4 蒸馏与 RL，3 次 × 60–90 min | OPD、DPO、在线 RL 分别训练什么分布、消耗什么资源？ | 报告 §4 + [RL 正文](../topics/omni_models_infra.md#rl) + [框架选型](../../04-rl-infra/topics/rl_framework_selection.md)；O4 | 写三张契约：paired query、preference pair、rollout group；解释 policy/logprob/reward/version |
| 5 流式推理，2 次 × 60–90 min | 首文本为什么不是首播？怎么定位音频卡顿？ | [Serving](../topics/omni_models_infra.md#serving) + [推理状态](../../03-inference-infra/topics/inference_state.md)；O5 | 画首包和稳态时间线，定义 RTF/chunk gap/underrun，能说明取消后哪些状态必须失效 |
| 6 集成与恢复，1–2 次 × 60–90 min | 优化怎样成为团队可持续使用的能力？ | [实验记录要求](../../practice/experiments/omni_sft_rl_lab.md) + [checkpoint](../../02-training-infra/topics/checkpointing.md)；O6 | 一页 baseline、一份可重放配置、一个恢复验收，以及可否上线的质量/成本对照 |

时间只用于安排自学，不是项目承诺。纸面练习与 CPU 数据验证可先做；GPU 实验需具备可运行工件与资源。读完一节不自动满足对应实验验收。

## 与算法同事对齐的最小问题单

1. 产品实际输入输出是什么：音频/视频、文本/语音、工具，是否存在机器人 action head？在线交互还是离线理解？
2. 当前阶段是 encoder alignment、SFT、OPD、DPO 还是在线 RL？这轮具体更新哪些模块？
3. 任务混合、采样分布、loss 分母与权重、stop/truncation、reward 版本由谁定义？
4. 模型/processor/codec/模板、训练 backend、rollout backend、checkpoint 格式分别是什么版本？
5. 业务质量门槛与真实瓶颈是什么？需要降低训练完成时间、GPU-hours，还是满足首包/稳态服务 SLO？
6. 现有 baseline 是否记录硬件/拓扑、数据版本、有效 token/音频时长、失败和完整端到端计时？

只需内部填写能影响近期决策的部分；未回答项保留“未知”，不由公开 Qwen 报告推断。

## 第一项优化怎么选

先完成 O0/O1，再从当前最大瓶颈选择一条：

| 候选交付 | 适合的观测 | 首个产出 | 成功判定 |
|---|---|---|---|
| 多模态数据和 SFT goodput | GPU 常等 decode/IO，step 长尾大 | 样本 manifest、展开成本、dataloader trace、分桶对照 | 固定样本和质量下，有效训练目标/秒提升，端到端训练时间下降 |
| Omni RL 有效轨迹供给 | trainer 等 group 或 reward，版本问题频发 | trajectory ledger、logprob parity、分阶段耗时 | 固定算法口径与质量下，完整有效 group 的供给提升，端到端成本下降 |
| 流式 serving | 首包/播放卡顿限制交互并发 | 分阶段 trace、队列容量、取消协议 | 固定资源和质量，SLO 内成功会话数提升且无旧轮次串音 |

首轮只推进一条。没有正确性和代表性基线时，暂不把异步 RL、跨机拆 stage、FP8 或大规模 CP 当作默认优化。

## 已覆盖与仍有缺口

- 已覆盖：公开报告的架构与后训练机制；SFT/RL/serving 的契约和优化问题；六阶段学习与 O0–O6 验证设计。
- 待补来源：ARIA 在线完整规则、准确 reward/loss 与配方、3.5 开放工件、框架实际兼容性。
- 待实践：指定模型的可运行最小训练、teacher/rollout scoring parity、多卡扩展、语音服务与故障恢复。全部未执行。
- 物理动作学习与闭环控制：如果工作包含这一接口，再接[具身六阶段](../roadmap.md)，不把语音交互指标当机器人任务成功率。

[回核心专题](../topics/omni_models_infra.md) · [研究来源索引](../../research/MASTER_READING_LIST.md) · [知识图谱](../../KNOWLEDGE_GRAPH.md)

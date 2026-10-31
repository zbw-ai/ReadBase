# Omni 优化：先固定目标与状态，再比较速度

**2026-10-31｜状态：工程假设，未实验验证。**

依据：[Qwen3.5-Omni 报告笔记](../tech_reports/qwen35_omni.md)中 Thinker/Talker 的不同后训练目标、ARIA 交错生成、Hybrid state 与首包关键路径。公开披露支持问题的存在；下面的决策规则是本库推断，不是作者性能结论。

## 判断

Omni Infra 的首个交付应是一套可重放的目标/状态基线。它能区分三种变化：在相同目标下执行更快、用更少信息或更低质量换成本、以及训练目标或状态语义发生变化。

原因是同一条链路至少有四种“长度”：音视频物理时长、输入表示长度、文本/codec 训练目标数、最终播放时长。某种长度变短，未必代表系统优化。例如减少视频帧可以提高 step/s，却改变了信息量；跳过慢 reward 可以提高组供给，却改变了经验分布；只优化首文本不一定缩短首播。

## 对工程决策的影响

| 决策 | 先固定的条件 | 再测什么 |
|---|---|---|
| SFT packing / 分桶 | 样本、全局目标、attention 与 recurrent 边界 | 梯度对照、E2E logical step time、资源成本 |
| 蒸馏 / reward cache | 原媒体、模态条件、teacher/reward 与前处理版本 | 命中质量、供给等待、完整成本 |
| RL rollout 加速 | 采样分布、action mask、完整 group、policy bundle | logprob parity、有效轨迹供给、训练质量 |
| serving batching / stage 拆分 | 质量、输出内容、首包与连续播放 SLO | 合格会话 goodput、客户端首播、queue 与取消 |

## 适用边界与反证

这不意味着每次 kernel 微优化都要从头训练模型。已有目标与数值基准覆盖所改路径时，复用局部回归即可；涉及数据、packing、采样、reward 或多组件版本时，再扩大验证范围。

如果任务实际是固定长度 text-only 输出、没有 Talker 与多阶段状态，很多额外约束可以去掉。如果测量证明端到端被单个 kernel 主导且目标完全相同，优化该 kernel 就可能直接产生业务收益。应以 trace 和对照实验确认，而不是把“Omni 很复杂”当作回避优化的理由。

## 验证方式

[O1](../../practice/experiments/omni_sft_rl_lab.md)检验目标等价，O2/O3 检验关键路径和收益，O4 检验概率与数据，O5/O6 检验交互和恢复。只有获得结果后，才将具体结论标 VERIFIED；目前仅有设计。

[回核心专题](../../05-embodied-infra/topics/omni_models_infra.md) · [学习路线](../../05-embodied-infra/roadmaps/omni.md) · [Insights](README.md)

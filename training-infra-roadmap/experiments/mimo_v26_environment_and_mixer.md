# MiMo-V2.6 / CodeMidas：环境与采样调度验证计划

> 2026-09-22；Status: NEW。仅完成实验设计，尚未生成环境、运行 RL 或执行故障注入，没有实测结果。

背景：[团队分享报告](../tech_reports/mimo_v26.md)、[Agentic RL 状态契约](../topics/agentic_rl.md#mimo-v26-environment-contract)。原始依据：[CodeMidas §3–5](https://arxiv.org/html/2609.22068v1)、[MiMo-V2.6 §5.5、§6.3](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL/blob/73875d00b30a89ef8cc353a0b60b0e9f9561952d/MiMo_V2_6_technical_report.pdf)。

## 假设与分阶段范围

先验证环境监督和调度机制，再决定是否花 GPU 时间做 RL。下列规模、阈值均为本仓库建议，并非 MiMo 原报告配置。

### A. 环境质量：不要把缺陷和难度混在一起

从获准使用的代码中选择 30 个功能任务，覆盖 CLI、纯函数、有状态 API；按仓库划分训练候选和保留测试任务，记录 commit、许可证、容器 digest、task/spec/verifier/harness 版本。原实现与 hidden verifier 放在 solver 看不到的边界外。

| 对照 | 操作 | 记录 |
|---|---|---|
| 起始状态 / reference | 分别在 2 / 4 个全新容器执行 | 期望 2 次 fail、4 次 pass；区分执行失败与断言失败 |
| 合法替代实现 | 人工构造满足规格但结构不同的解 | false negative 数及具体过严断言 |
| 有意缺陷实现 | 缺少边界行为、返回常量、错误状态清理 | false positive 数及未覆盖要求 |
| 环境故障注入 | grader 超时、依赖不可用、reset 失败 | 应输出 INFRA_ERROR，不进入任务 reward |
| 泄漏检查 | 在隔离测试容器中设置已知缓存残留并审核 | 是否检出以及清理后 reference 是否仍可构建 |

保留原始合成版本与清洗版本，对相同任务、相同固定策略各采样 4 条 rollout，盲审 submitted code 和 reward 是否一致。先报告误判数量与分母，再报告比例；任务数小，不宣称统计显著。发现一个确定 verifier 缺陷或答案泄漏就暂停该任务进入训练。

环境检查不是 RL 质量消融。未来比较清洗前后 RL 时，需控制任务身份、基座、rollout/token/compute budget 与 held-out eval；否则不同数据池的难度变化会混入质量效果。预算不足时只交付环境验收结果。

### B. 配比调度：固定总并发，比较有效供给

先用合成时间分布或真实小样本 trace 做离散事件模拟，不调用模型；设三个 source，分别代表快任务、长尾任务、低接受率任务。固定目标份额、并发上限与随机种子，比较：按目标比例投递、按缺口投递、按耗时/接受率与缺口联合调度。

记录每 step 的 accepted 与 consumed group 配比、收齐时间、并发占用、过量积压和成本代理指标。再加入 grader 延迟、超时重试、staleness 淘汰和恢复 replay，单独报告加入这些因素前后结果。MiMo Figure 16 未纳入全部这些代价，本实验不应复制这一证据缺口。

建议至少 5 个 seed；报告中位数和 p95、最差 source 偏差。若吞吐改善但长期消费分布偏离目标，视为未通过。计数配额允许的误差应按 batch group 数推导，而非用一个不适合小 batch 的固定百分比。

### C. 冷启动与恢复：检查完成顺序偏差

对同一个 source 设置两个长度差异明显的 harness；比较 source-only 与 source × harness 的长度先验，以及均值估计与保守分位数估计。对未完成样本保留运行年龄，避免只观察已完成短轨迹。

加入容量受限的 HBM / pinned-host KV pool，模拟重启后第一轮收集；分别记录容量超订、拒绝准入、吞吐恢复时间、每 source 消费配比、policy version 与重复消费。Replay 必须限定可用 policy/staleness 与幂等消费规则。

## 实验结果模板

```text
Run ID / source revisions:
Task / environment / verifier / harness versions:
Policy / seed / budget:
Baseline and treatment:
False positives / negatives (counts + denominators):
INFRA_ERROR handling:
Consumed mix / collection latency / stale discard:
HBM / host peak / packer RSS:
Failure evidence and counterexamples:
Conclusion and limits:
```

## 完成标准

产出环境清单、独立审核记录、模拟输入和完整结果后，才将对应阶段标为 VERIFIED。模拟验证不能代表 AReaL 生产路径验证；需另行接入小模型、真实环境及 runtime 后，再记录端到端证据。

回写目标：[Agentic RL](../topics/agentic_rl.md#mimo-v26-environment-contract)、[长期 insight](../insights/001_agentic_rl_will_change_training_infra.md#mimo-v26-evidence)。

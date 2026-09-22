# MiMo-V2.6 / CodeMidas：环境、调度与质量判断验证计划

> 2026-09-22；Status: NEW。以下 A–E 均为待执行设计；尚未生成环境、运行 RL、执行故障注入或完成 grader 审核，没有实测结果。

背景：[团队分享报告的四条观点](../tech_reports/mimo_v26.md#engineering-judgments)、[Agentic RL 状态契约](../topics/agentic_rl.md#mimo-v26-environment-contract)。原始依据：[CodeMidas §3–5](https://arxiv.org/html/2609.22068v1)、[MiMo-V2.6 §4.3、§5.3、§5.5、§6.3](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL/blob/73875d00b30a89ef8cc353a0b60b0e9f9561952d/MiMo_V2_6_technical_report.pdf)。

## 假设与分阶段范围

先验证环境监督、调度机制和 grader 可靠性，再决定是否花 GPU 时间做 RL。若已有可用的前后 checkpoint，可直接安排预算受控评测。下列规模、阈值均为本仓库建议，并非 MiMo 原报告配置。

| 阶段 | 要检验的判断 | 最小产出 | 不能据此声称 |
|---|---|---|---|
| A | 环境可信与当前训练准入值得分开记录 | 验收记录、不同策略/预算下的准入变化 | 环境审核已带来 RL 增益 |
| B/C | 配比与恢复验收需要看 source 内部 | 配比、长度/harness 分布、延迟与淘汰记录 | 模拟吞吐改善等于学习改善 |
| D | 质量排序需要独立验收 | 错误纠正与偏好排序的分层审核 | 离线 grader 一致性等于训练有效 |
| E | 相同上限与相同实际成本回答不同问题 | 前后 checkpoint 的成功率—成本曲线 | 将整条训练流程的增益归因于某个模块 |

<a id="environment-validity"></a>

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

**追加检验：同一可信环境是否应重新准入。**固定已审核的 task/env/verifier 版本，让两个能力不同的 checkpoint 在两档预算下分别尝试；固定 harness 和每任务 rollout 数，记录全通过、全失败、mixed-outcome，以及实际 reward 是否有差异。独立审核暂缓任务中的失败原因，不把基础设施失败算成模型失败。分别保存环境验收状态和 `policy × harness × budget × reward` 下的训练准入状态；发现新漏洞时更新前者，而非假设验收永久有效。

报告各条件下准入集合的变化和具体任务案例。若重新采样长期不改变准入，也没有改善覆盖或后续训练效果，则复杂生命周期管理的投入优先级下降。这个阶段只能检验准入变化；重新使用暂缓任务是否改善学习，需要后续等训练预算的 RL 对照。

单个 group 全通过或全失败不足以判定整个任务无价值。任务降频依据应包含多次采样的有效 group 比例、样本量与预算；有限采样也不能把真实成功概率断言为 0 或 1。

<a id="scheduler-distribution"></a>

### B. 配比调度：固定总并发，比较有效供给

先用合成时间分布或真实小样本 trace 做离散事件模拟，不调用模型；设三个 source，分别代表快任务、长尾任务、低接受率任务。固定目标份额、并发上限与随机种子，比较：按目标比例投递、按缺口投递、按耗时/接受率与缺口联合调度。

记录每 step 的 accepted 与 consumed group 配比、收齐时间、并发占用、过量积压和成本代理指标。再加入 grader 延迟、超时重试、staleness 淘汰和恢复 replay，单独报告加入这些因素前后结果。MiMo Figure 16 未纳入全部这些代价，本实验不应复制这一证据缺口。

增加同一 source 内的任务长度、harness、任务身份分层，对照 submitted → accepted → consumed。固定 trace 时保持任务身份、长度与 reward 结果的对应关系，检查“快样本更常进入梯度”是否存在。长度不是难度的替代标签；若要讨论难度偏差，需要独立的任务难度证据。所有任务最终都被消费时，应将短暂顺序差异与长期占比变化分开报告。

建议至少 5 个 seed；报告中位数和 p95、最差 source 偏差。若吞吐改善但长期消费分布偏离目标，视为未通过。计数配额允许的误差应按 batch group 数推导，而非用一个不适合小 batch 的固定百分比。

### C. 冷启动与恢复：检查完成顺序偏差

对同一个 source 设置两个长度差异明显的 harness；比较 source-only 与 source × harness 的长度先验，以及均值估计与保守分位数估计。对未完成样本保留运行年龄，避免只观察已完成短轨迹。

加入容量受限的 HBM / pinned-host KV pool，模拟重启后第一轮收集；分别记录容量超订、拒绝准入、吞吐恢复时间、每 source 消费配比、policy version 与重复消费。Replay 必须限定可用 policy/staleness 与幂等消费规则。

用同一份 trace 比较连续运行、仅恢复模型后冷启动、同时恢复合格 sample pool 和调度统计量三种条件。观察恢复后前几步的 source × harness 长度分布、replay 占比与 policy age，并记录恢复到预先定义的正常区间需要多少 step。若数据组成没有变化而等待时间下降，可先确认吞吐收益；样本更新仍可能影响训练，接入真实模型后还要检查 policy age、clip fraction 与固定评测。如果组成改变，先确定它是否符合训练意图，再用真实训练检验后果。

<a id="grader-calibration"></a>

### D. Grader 审核：分开评价错误纠正与正确解排序

从 A 中通过审核的任务收集一批固定候选解，包括明确缺陷、测试通过但漏满足规格、满足规格的不同实现，以及确实需要较大改动的任务。记录 rubric、grader、任务及候选版本；grader 看不到保留检查与独立审核结论。

| 对照层次 | 评分行为 | 独立验收关注点 |
|---|---|---|
| Binary-only | 只使用原始 executable tests | 已知 false positive / false negative 的基线 |
| 增加错误修正 | 在可复核证据下纠正错误 verdict | 纠正了多少真错误，又误杀了多少合法实现 |
| 再增加质量排序 | 在通过验收的解之间应用质量 rubrics | 排序是否符合该任务的维护、回归和规格要求 |

隐藏候选模型身份、随机交换候选顺序，由未参与训练评分的审核者盲审；使用保留规格检查和回归测试补充判断。先报告错误纠正的数量与分母，再报告排序一致性、分歧案例和顺序敏感性。把“审核者之间也不一致”单独记录，不能强行把某位审核者当作绝对 oracle。尤其检查 grader 是否一律偏爱短 patch，漏掉为满足要求而必要的大改动。

MiMo 原报告已经提到维护者视角的独立审核；本实验进一步要求可复查的量化记录。离线一致性只能说明评分可靠性的局部证据。若要复现 GAR 的训练收益，错误修正应按原文限定为 confirmed hacking 置零，比较 binary、仅 hacking correction、完整 GAR，并保持 length penalty、任务、基座等条件一致。上表包含普通遗漏、false negative 的广义 verdict 修正属于扩展实验，应单独命名和报告。若要决定预算分配，再另做“更多 grader / 更多 rollout”的等总训练成本对照，不把两个问题合在一次不受控实验中。

<a id="evaluation-budget"></a>

### E. 预算受控评测：分别看可达表现与实际效率

准备同一训练流程的初始与训练后 checkpoint、按仓库隔离的评测任务和版本固定的工具环境。先用同一个 harness，再独立使用一个未参与训练的 harness；不混合两组成绩。预先固定 attempts、停止规则与预算网格，设置多档生成 token、工具调用和墙钟上限，覆盖短任务及允许长程探索的范围。

每个配置记录成功率与不确定性、实际 token、工具调用次数、延迟和费用口径。预算耗尽是 agent 未完成，INFRA_ERROR 另列并报告分母及重试规则，不能悄悄删除失败。配对任务记录有助于区分普遍增益与少数长任务驱动的变化。

绘制成功率—实际成本曲线，分别标出 token、墙钟时间及可核算费用；不要将三者直接合成一个没有说明权重的分数。同上限比较回答“允许相同资源范围时能达到什么表现”；成本相近的点及不确定性回答效率问题。实际成本不完全匹配时，展示曲线和差异，不宣称严格等成本，也不通过事后截取成功样本来匹配预算。

若仅高预算区间出现增益，结论限定在该使用场景；有效利用长交互本身仍有价值。若相同成本下也更成功，或同一成功率下成本降低，则有更强的效率证据。跨 harness 保持增益可以加强迁移判断，但仍不能代替环境、grader 或 scheduler 的独立消融。

## 实验结果模板

```text
Run ID / source revisions:
Task / environment / verifier / harness versions:
Policy / seed / budget:
Baseline and treatment:
False positives / negatives (counts + denominators):
INFRA_ERROR handling:
Consumed mix / collection latency / stale discard:
Within-source mix / recovery steps / replay share / policy age:
Grader correction / false rejection / ranking disagreement:
Evaluation limits / actual token, tool, time and cost / uncertainty:
HBM / host peak / packer RSS:
Failure evidence and counterexamples:
Conclusion and limits:
```

## 完成标准

产出环境清单、独立审核记录、模拟输入和完整结果后，才将对应阶段标为 VERIFIED。模拟验证不能代表 AReaL 生产路径验证；需另行接入小模型、真实环境及 runtime 后，再记录端到端证据。

回写目标：[Agentic RL](../topics/agentic_rl.md#mimo-v26-environment-contract)、[长期 insight](../insights/001_agentic_rl_will_change_training_infra.md#mimo-v26-evidence)。

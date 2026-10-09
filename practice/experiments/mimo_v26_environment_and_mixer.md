# MiMo-V2.6 / CodeMidas：环境、调度与质量判断验证计划

> 初稿 2026-09-22；设计更新 2026-10-08；Status: NEW。以下 A–F 均为待执行设计；尚未生成环境、运行 RL、执行故障注入或完成 grader 审核，没有实测结果。

背景：[团队分享报告的四条观点](../../research/tech_reports/mimo_v26.md#6-工程判断与验证路径)、[Agentic RL 状态契约](../../04-rl-infra/topics/agentic_rl.md#mimo-v26-environment-contract)。原始依据：[CodeMidas §3–5](https://arxiv.org/html/2609.22068v1)、[MiMo-V2.6 §4.3、§5.3、§5.5、§6.3](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL/blob/73875d00b30a89ef8cc353a0b60b0e9f9561952d/MiMo_V2_6_technical_report.pdf)。

## 假设与分阶段范围

先验证环境监督、调度机制和 grader 可靠性，再决定是否花 GPU 时间做 RL。若已有可用的前后 checkpoint，可直接安排预算受控评测。下列规模、阈值均为本仓库建议，并非 MiMo 原报告配置。

| 阶段 | 要检验的判断 | 最小产出 | 不能据此声称 |
|---|---|---|---|
| A | 环境可信与当前训练准入值得分开记录 | 验收记录、不同策略/预算下的准入变化 | 环境审核已带来 RL 增益 |
| B/C | 配比与恢复验收需要看 source 内部 | 配比、长度/harness 分布、延迟与淘汰记录 | 模拟吞吐改善等于学习改善 |
| D | 质量排序需要独立验收 | 错误纠正与偏好排序的分层审核 | 离线 grader 一致性等于训练有效 |
| E | 相同上限与相同实际成本回答不同问题 | 前后 checkpoint 的成功率—成本曲线 | 将整条训练流程的增益归因于某个模块 |
| F | 模型 release 需要过程行为回归 | 固定版本下的重复、flooding、合理重试与任务成功对照 | 有限回放零重复代表线上已无风险 |

2026-10-08 已核验[公开资源与依赖](../../research/tech_reports/mimo_v26.md#8-开源资源与复现条件)。A 可从固定 revision 的已发布环境抽样，再检查 K8s、镜像、judge 与共享文件；不再以“先等待环境发布”为前提。代码静态核验没有改变任何实验的 NEW 状态。

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

GAGAR 新论文已披露 30 个任务、匿名随机排序及独立 Claude 裁判的审核；本实验将进一步引入人类维护者判断与跨裁判一致性。离线一致性只能说明评分可靠性的局部证据。若要复现 GAR 的训练收益，错误修正应按原文限定为 confirmed hacking 置零，比较 binary、仅 hacking correction、完整 GAR，并保持 length penalty、任务、基座等条件一致。上表包含普通遗漏、false negative 的广义 verdict 修正属于扩展实验，应单独命名和报告。若要决定预算分配，再另做“更多 grader / 更多 rollout”的等总训练成本对照，不把两个问题合在一次不受控实验中。

在训练前先用人工构造的有效 group 检查 [GAGAR Appendix A.2–A.3](https://arxiv.org/html/2609.32577v1)：无 cap 分支检查正 advantage 总和、失败项和比例；cap 触发分支检查重新中心化后的零均值，不要求保留原正 advantage 总和；非二元 length-adjusted rewards 另测正部截断与零分母 fallback。若实现等价 reward 形式，需同时固定 masks、token 权重、ratio 与 clipping 设置，比较最终 loss/gradient；不能只用组均值一致证明等价。以上检查尚未执行。

<a id="evaluation-budget"></a>

### E. 预算受控评测：分别看可达表现与实际效率

准备同一训练流程的初始与训练后 checkpoint、按仓库隔离的评测任务和版本固定的工具环境。先用同一个 harness，再独立使用一个未参与训练的 harness；不混合两组成绩。预先固定 attempts、停止规则与预算网格，设置多档生成 token、工具调用和墙钟上限，覆盖短任务及允许长程探索的范围。

每个配置记录成功率与不确定性、实际 token、工具调用次数、延迟和费用口径。预算耗尽是 agent 未完成，INFRA_ERROR 另列并报告分母及重试规则，不能悄悄删除失败。配对任务记录有助于区分普遍增益与少数长任务驱动的变化。

绘制成功率—实际成本曲线，分别标出 token、墙钟时间及可核算费用；不要将三者直接合成一个没有说明权重的分数。同上限比较回答“允许相同资源范围时能达到什么表现”；成本相近的点及不确定性回答效率问题。实际成本不完全匹配时，展示曲线和差异，不宣称严格等成本，也不通过事后截取成功样本来匹配预算。

若仅高预算区间出现增益，结论限定在该使用场景；有效利用长交互本身仍有价值。若相同成本下也更成功，或同一成功率下成本降低，则有更强的效率证据。跨 harness 保持增益可以加强迁移判断，但仍不能代替环境、grader 或 scheduler 的独立消融。

#### E 追加：把 serving 加速与 RL 有效供给分开测量

来自 [MiMo 演进对照](../../research/tech_reports/mimo_v26.md#76-跨版本的工程判断)的新问题：底层推理变快，收益最终落到用户请求、有效 group，还是只落到局部 kernel？这是待执行设计，不要求复现 TileRT，也不预设特定 GPU 或加速比。

先固定可用 checkpoint、任务池、harness、精度和采样语义，只切换一个已支持的推测解码配置，扫相同的请求长度与并发档位。分别做两条测量：

- **Serving**：记录 TTFT、ITL、p50 / p95 / p99 任务完成时延、超时、质量和全部分配设备时长；分母包括失败请求，成本比较附成功任务数，不能只统计快完成者。
- **RL 数据供给**：加入同一组工具与 grader 延迟 trace，固定 group 大小、各 source 目标份额和 staleness 限制，记录收齐并消费一个 batch 的时间、分层消费分布、重试、discard 与各池设备时长。模拟 trace 结果只能证明调度行为，不能证明模型学习收益。

如同时改变精度、模型或采样器，应另列对照并先验收质量 / 概率一致性；不能伪装成同模型单因素加速。若 serving 变快而 RL batch 时间不变，检查工具、评分与组尾等待；若 batch 变快却消费分布漂移，不能判定方案通过。只有后续实际训练在固定全链路预算下改善 held-out 结果，才升级为“学习效率改善”。所有阶段仍为 NEW。

<a id="release-behavior"></a>

### F. 发布行为回归：固定版本，并区分重复、调用量与必要重试

依据[官方修复复盘与本次观点修订](../../research/tech_reports/mimo_v26.md#51-工具调用重复与-mopd-修复)，先验证行为是否发生变化，再考虑训练修复。对照对象为同家族 RL / MOPD checkpoints；若只使用 API，记录 provider、route、模型名、请求时间及服务返回的版本字段，未提供固定 revision 时明确不能完全排除服务漂移。

| 回放集合 | 需要记录什么 | 保护的边界 |
|---|---|---|
| 曾出现无进展循环的失败案例 | 固定历史与 seed；按 harness、context、既有 flooding 历史分层 | 这类富集集合的发生率不能外推到生产流量 |
| 随机正常任务 | 任务成功、工具错误、延迟与资源消耗 | 避免只修好已知坏例，却损伤整体效果 |
| 合法并行与必要重试 | 同工具不同参数、超时后重试、改代码后复测 | 不将“调用变少”直接奖励为更好 |

分别统计：有重复的 response 比例；同一 turn 内工具名与规范化 JSON 参数完全相同的 `(N−U)/N`；跨轮/近似重复的人工审核；调用量分布与超过预先指定阈值的比例。Code-mode 若只能看到外层 exec，则标记底层调用未观测，不计为“零重复”。规范化 JSON 不等于理解语义，必要 polling/retry 是否合理仍需环境状态证据。

固定回放分母并报告失败/缺失原因；任何过滤都同时列出原始与有效分母。比较学生的最终表现，不能用专门 teacher 的零重复替代学生结果。小样本零观测只给出样本量与区间，不宣称线上问题已消失。若继续开展修复训练，分别核算 teacher 构造、合并蒸馏、验证与实际恢复成本，明确真实开销与全规模方案估算的区别。

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
Checkpoint / provider / route / API timestamp / version fields:
Repetition response and call denominators / flooding threshold / missing observations:
HBM / host peak / packer RSS:
Failure evidence and counterexamples:
Conclusion and limits:
```

## 完成标准

产出环境清单、独立审核记录、模拟输入和完整结果后，才将对应阶段标为 VERIFIED。模拟验证不能代表 AReaL 生产路径验证；需另行接入小模型、真实环境及 runtime 后，再记录端到端证据。

回写目标：[Agentic RL](../../04-rl-infra/topics/agentic_rl.md#mimo-v26-environment-contract)、[长期 insight](../../research/insights/001_agentic_rl_will_change_training_infra.md#mimo-v26-evidence)。

<a id="reproduction-entry"></a>

## 复现入口：先验证资源与最小训练闭环

[主文的开源与卡数说明](../../research/tech_reports/mimo_v26.md#81-复现范围与硬件规模)区分官方参考拓扑与工程估计。公开起点为 [MiMo-V2.6-Distill-Qwen-9B](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B)，代码固定 `a2ad9f6160b03ff2d47e59832bfb6b289f37c917`。这里仍为 NEW，没有任何 GPU 配置经过本仓库验证。

启动实际训练前，依次确认：

1. 固定训练仓库、子模块、权重、tokenizer / chat template、镜像与任务版本；确认选用 MiMo 的 9B SFT 起点，而非配置示例中可能出现的原始 Qwen 模型。
2. 用少量任务完成 reset、工具调用与 verifier 健康检查；至少覆盖真实成功、任务失败和 infra error，避免全零 reward 仍正常开训。
3. 以短上下文和低并发测量模型加载、rollout、prefill / logprob、backward / optimizer、checkpoint 各阶段峰值。单机 8 × 80GB 仅是规划预算，不是通过记录；不得直接向用户承诺最低卡数。
4. 跑通一个完整更新及 checkpoint 重载，验证 action mask、行为 logprob、有效 group 和参数变化；再跑少量 steps，记录实际成本与保留任务表现。闭环通过不等于论文指标复现。
5. 按上下文、batch、group、并发分别扩容，每次只改变一个主要因素；资源不足时保留失败证据，不以改成 LoRA 或缩小模型后的结果冒充原 recipe。

Code 优先验证多轮工具与可执行测试；Music 可作为较少外部依赖的训练接口验证。General / Webdev 的 judge 服务需单独计费或配卡，CPU 任务容器与镜像存储也不包含在训练 GPU 数内。

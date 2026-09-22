# 001: Agentic RL Will Change Training Infrastructure

## 核心判断

Agentic RL 会把训练基础设施从“高效执行大规模矩阵计算”推进到“高效运营一个持续产生经验数据的分布式系统”。

这不是一个小的 post-training 分支，而是训练平台职责边界的扩张。未来的 RL training platform 不仅要训练模型，还要调度 rollout、运行 agent runtime、管理 tool/environment、执行 verifier、控制样本新鲜度、同步权重、追踪 policy lineage，并把这些状态纳入 checkpoint 和 observability。

## 为什么我认为它重要

Pretraining 的系统目标相对清楚：让 GPU 以稳定 batch 消费 token。Agentic RL 的输入不是静态数据集，而是由当前 policy 在环境中生成的 trajectory。这意味着数据生产速度、样本质量、reward 延迟和训练稳定性互相耦合。

从工程角度看，Agentic RL 至少引入四个新的一等问题：

- rollout tail latency：长链推理和工具调用导致 batch 内样本时间差极大；
- sample freshness：样本来自哪个 policy version 会影响训练稳定性；
- runtime observability：prompt、tool call、reward、trace 必须能被复盘；
- train/inference dual state：同一个 actor 在训练布局和推理布局之间反复切换。

这些问题不是 PPO 参数能解决的。它们是平台问题。

## 关键信号来源

- [AReaL](https://arxiv.org/abs/2505.24298)：用 fully asynchronous RL 解耦 rollout 和 training，把 freshness 和 staleness 变成系统控制项。
- [HybridFlow / verl](https://arxiv.org/abs/2409.19256)：把 RLHF 表达为复杂 dataflow，并正面处理 actor training/generation 之间的 resharding。
- [Agent Lightning](https://arxiv.org/abs/2508.03680)：把 agent execution 和 trainer 解耦，让已有 agent runtime 通过 trace/transition 接入 RL。

这些来自 historical backfill 的信号共同说明：Agentic RL Infra 的主战场不是单个算法，而是 pipeline architecture。

## Historical Backfill 后的补充判断

[Historical Backfill](../tracking/historical_backfill.md) 让我更确定一件事：Agentic RL Infra 不是 2026 年突然冒出来的新方向，而是 RLHF pipeline、distributed RL dataflow、serving engine、Ray-style orchestration 和大型训练栈长期汇合的结果。

OpenRLHF / DeepSpeed-Chat 解释了早期 RLHF 为什么已经是多模型、多组件调度问题；vLLM + OpenRLHF 解释了推理引擎如何进入训练闭环；Ray RLlib/Ray Train 解释了底层调度思维；NeMo RL 则说明厂商训练栈正在把这些能力产品化。AReaL、verl、Agent Lightning 是当前最值得精读的三条主线，但不是孤立信号。

## 对工程决策的影响

如果我要设计下一代训练平台，我会提前预留这些能力：

1. Rollout worker pool 和 training worker pool 分离调度。
2. Trajectory store 支持 policy version、reward version、tokenizer version 和 tool/env version。
3. Reward/verifier 独立扩缩容，并暴露 queue depth 和 p99 latency。
4. Weight sync 有原子切换、版本确认和回滚能力。
5. Checkpoint 不只保存模型状态，还保存 RL pipeline 状态。
6. Observability 覆盖 token/s、request/s、sample freshness、policy idle、verifier backlog。
7. Inference engine 作为训练依赖纳入版本治理，而不是临时服务。

## 需要警惕的误区

- 误区一：把 Agentic RL 当成 SFT 后的一段小训练脚本。
- 误区二：只看训练 GPU utilization，不看 rollout/reward/trainer 的分段等待。
- 误区三：认为异步一定更好，忽略 stale sample 对算法稳定性的影响。
- 误区四：忽略 tokenizer、chat template、tool schema 变化带来的 correctness 问题。
- 误区五：不保存 trace，只保存 reward，导致失败样本无法复盘。

## 下一步

- 精读 AReaL，重点拆 rollout/training 解耦和 staleness 控制。
- 精读 HybridFlow，重点拆 RLHF dataflow 和 actor resharding。
- 阅读 Agent Lightning，重点看 agent trace 如何进入 trainer。
- 阅读 OpenRLHF 和 vLLM + OpenRLHF integration，补齐 Ray/vLLM/DeepSpeed 多组件调度细节。
- 用 [Rollout Latency Playbook](../playbooks/rollout_latency.md) 反推一套最小观测指标。

## 结论

Agentic RL 会让 training infra 和 inference infra、agent infra 汇合。谁能把 rollout、verifier、training、checkpoint 和 observability 做成稳定闭环，谁才真正拥有可扩展的 reasoning/agent post-training 能力。

<a id="mimo-v26-evidence"></a>

## 2026-09-22 工业证据与判断：监督定义和样本选择共同影响学习

[MiMo-V2.6 / CodeMidas 调研](../tech_reports/mimo_v26.md)补充了两类证据：[CodeMidas](https://arxiv.org/html/2609.22068v1) 的质量筛选实验表明，在报告设置中高质量 3k 任务优于未清洗 8k，但没有按构造总成本比较；[V2.6 报告 §4.3、§6.3](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL/blob/73875d00b30a89ef8cc353a0b60b0e9f9561952d/MiMo_V2_6_technical_report.pdf)展示了质量 grader 与异构采样调度怎样改变训练信号和数据供给。

**我的立场：**对尚未证明监督和数据路径可靠的系统，应先建立“为什么选中这条经验、评分依据是什么、实际带来什么收益”的证据，再扩吞吐。这里有四个具体选择，均为本仓库推断，尚未实测。

| 工程选择 | 适用理由 | 什么证据会让我调整判断 |
|---|---|---|
| [分开记录环境可信性和训练准入](../tech_reports/mimo_v26.md#judgment-environment) | 一次全通过/全失败只反映特定策略与预算下的有限尝试；新策略也可能揭露旧 verifier 漏洞 | 准入集合长期稳定且重新准入没有收益时，复杂管理可以后置 |
| [把 source 内部分布纳入恢复验收](../tech_reports/mimo_v26.md#judgment-scheduler) | 配额正常仍可能掩盖短轨迹优先、harness 偏差与 policy age 变化 | 若差异仅为短暂顺序、最终消费及训练结果不受影响，则降低长期偏差风险的判断 |
| [把 grader/rubric 作为训练目标来审核](../tech_reports/mimo_v26.md#judgment-grader) | 正确解排序在定义“更好的成功”，不能用同一个 grader 的高分自证价值 | 独立验收无收益或误杀增加时，收缩对应偏好；有效且划算才扩预算 |
| [同时报告可达表现与实际成本曲线](../tech_reports/mimo_v26.md#judgment-evaluation) | 相同推理上限未必产生相同实际成本，长程探索也可能有真实价值 | 若收益只出现在高预算范围，则限定场景；固定成本仍领先才支持效率提升 |

上述顺序针对缺少可信基线的团队；若已有证据表明 GPU 是主要瓶颈，计算优化应提前。CodeMidas 未隔离每个筛选步骤，V2.6 调度图包含模拟，生产大 run 也有人工干预，不能把它们合并成单因素 scaling 证明。已将判断写入 [Agentic RL](../topics/agentic_rl.md#mimo-v26-environment-contract)，对应[环境、调度、grader 和评测实验 A–E](../experiments/mimo_v26_environment_and_mixer.md)均为 NEW，没有 VERIFIED 结果。

## 2026-09-22 GitHub 补证据：有效容量由执行阶段决定

[补扫 G5/G8/G9](../tracking/github_audit_2026-09-22.md)分别暴露 prefill 临时矩阵、dummy graph 的真实 KV 写入，以及缓存命中后的跨 PP 等待。我的工程推断是：评估 rollout 容量时，应以阶段峰值、请求状态和可准入条件联合定义容量，不能只由持久 KV 大小或 cache-hit ratio 推导。若真实 trace 显示这些成本可忽略且 GPU GEMM 长期主导，则把计算优化放回首位；该判断需要[实验](../experiments/rl_state_boundaries.md)验证，目前没有本地性能结论。

## 2026-09-22 历史复盘：重新定义值得跟踪的工程信号

[7–9 月的逐月复盘](../tracking/monthly_reviews.md)修正了一个筛选偏差：只问“是否提出新系统/新调度”，会低估测量口径、静默数据错误和验证工具。[verl GPU 分母、NeMo sampled-token logprob、slime 完成队列、恢复 frontier](../tracking/github_retrospective_2026-07_to_2026-09.md)都没有靠新架构命名，却能改变训练成本或优化目标。

因此新增一个判断准则：**一项材料是否改变了我们对有效样本、状态所有权或可验证结果的定义？** 满足这个条件的“小修复”和负面 field report 可以比一般发布更值得读。反过来，相关性提高不代表实验结论被证明；BPO 的生产 snapshot 成本、环境合成的规模化、kernel verifier 的通用性仍需自己的验证。

这是复盘推断，落点见[工程不变量](../topics/agentic_rl.md#monthly-retrospective-invariants)和[候选实验](../experiments/rl_state_boundaries.md#retrospective-test-cases)，尚不属于 VERIFIED。已有个人阅读状态不变。

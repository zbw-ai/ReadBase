# 2025 Q3 训练基础设施复盘：数值一致性、低精度与稀疏化进入系统边界

> 整合说明（2026-10-08）：本文保留 9/22 的来源核验与复评日期；[GitHub 历史定向补证](github_history_2025_to_2026_h1.md)已保存，但全窗口事件索引重建仍因 API 连接失败待补，不能据此声称历史 GitHub 全覆盖。

> **2026-09-22 历史回看**，原始材料窗口为 2025-07-01 至 2025-09-30。本文重新选择当季值得继续读的训练系统材料，不是当时运行过的 frontier scan，也不改变历史 Accepted 数、扫描游标或已有阅读状态。下列 5 项是本次历史精选，First seen / 本次核验日期统一为 **2026-09-22**，Status 均为 **NEW**；Read / Deep Dive 表示建议投入，不代表用户已经读完。

## 这个季度应当留下什么判断

沿着上半年的 MoE 与 RL 扩展继续往后看，Q3 最值得保留的是：**运行方式会进入模型的数值行为，训练基础设施不能只负责把同一份权重搬到更多 GPU。** Kimi K2 把 optimizer 稳定性、EP overlap 和 engine switching 放进同一份工业报告；Thinking Machines 则从 reduction 顺序解释为什么固定权重、甚至 greedy decoding，都不足以保证稳定的输出。

另一条主线是计算预算的重新分配。NVFP4 用更窄的 GEMM 换取效率，却保留高精度状态与敏感路径；DeepSeek Sparse Attention 减少主 attention 访问的 token，却增加 indexer 与选取逻辑。**趋势推断：** 这些工作共同要求用“哪些路径变快、哪些状态仍然昂贵、哪些数值语义改变”评估优化，而不是用一种 dtype 或一种复杂度概括整套系统。

Agent Lightning 连接起下一季度的环境主线：当 agent 有自己的工具调用、控制流与执行框架，训练必须先获得可解释的 transition。到 2026 年，问题进一步变成 token/logprob、环境版本、终止原因和恢复位置能否共同追溯；不能把那些后来的完整协议倒写成 2025 年已经解决。

## 五份核心材料

### 1. Kimi K2：大 MoE 的稳定性与状态搬运需要一起设计

**来源：** [Kimi K2: Open Agentic Intelligence](https://arxiv.org/abs/2507.20534v1)，Kimi Team 等，technical report；本文采用报告 v1 的 **2025-07-28**，不是把它当作模型首次发布日。Source ID：`arxiv:2507.20534v1`。Impact：★★★★★；Decision：**Deep Dive**。

问题不只在于 MoE 能否容纳更多参数。Muon 扩大规模后，attention logits 可能失控；MuonClip 在 optimizer update 后按 head 调整 Q/K projection weights，以控制这种增长。报告还披露 EP 通信重叠、选择性重算、部分 activation 的 FP8 存储与 CPU offload，说明优化器的 token efficiency 必须由能长期稳定运行的系统兑现。作者报告的长程训练稳定性属于工业自报证据，不是对其他模型的保证。[机制与训练系统](https://arxiv.org/html/2507.20534v1#S2)

更贴近今天 RL 工程的是 engine switching：训练与推理共享资源后，换引擎仍要付出权重装载和布局转换成本。附录 G 记录了 H800 上 H2D 与 broadcast 争用 PCIe，导致理想的三阶段流水退化，最终采用两阶段安排。**工程判断：** 画出来能 overlap 的操作，实际可能争用同一条物理通路；应按 H2D、broadcast、reload 分段测时。[RL infrastructure 与附录 G](https://arxiv.org/html/2507.20534v1#A7)

**Reason：** 同一份报告同时提供数值失稳机制与状态搬运的负面工程案例。**Next：** 优先读 §2.4、§3.3 和附录 G，画出 colocated RL 的权重切换时间线，再回读 MuonClip。Related topics：[MoE](../../02-training-infra/topics/moe.md)、[Agentic RL](../../04-rl-infra/topics/agentic_rl.md)、[Pipeline Parallelism](../../02-training-infra/topics/pipeline_parallelism.md)。目标流向：tech report / experiment；当前 Status：NEW。

### 2. Agent Lightning：先把 agent 执行变成可训练的数据接口

**来源：** [Agent Lightning: Train ANY AI Agents with Reinforcement Learning](https://arxiv.org/abs/2508.03680v1)，Xufang Luo、Yuge Zhang、Zhiyuan He、Zilong Wang、Siyun Zhao、Dongsheng Li、Luna K. Qiu、Yuqing Yang；paper / framework，**2025-08-05**。Source ID：`arxiv:2508.03680v1`。Impact：★★★★☆；Decision：**Read**。

真实 agent 的控制流不一定属于 trainer：工具可能阻塞，工作流会分支，也可能由多个 agent 协作。原论文把 agent execution 与 RL training 分离，用统一的数据接口描述执行，并通过 LightningRL 的 credit assignment 将交互拆成训练 transition。它补的是 agent runtime 到训练数据之间的接口，不能仅理解成换一个 PPO/GRPO 实现。[原论文摘要与系统设计](https://arxiv.org/abs/2508.03680v1)

**工程判断：** 日志可观测不等于轨迹可训练；接入时仍要确认哪次模型调用对应哪段 action、reward 如何归属，以及工具结果如何进入下一状态。原工作在 SQL、RAG 和数学工具任务上的实验，不能推出所有复杂 harness 都已零成本接入。**版本边界：** 2025 年论文与[2026 年 8 月月报](monthly_signal_2026-08.md)里的 Agent Lightning v1.0 是不同时间点，不能引用当前 README 的新功能证明原版能力。

**Reason：** 它解释了为什么 agent 训练需要独立的数据协议，而不仅需要更快的 rollout。**Next：** 对照一个实际 AReaL rollout，标注 execution event、transition 与 reward 的对应关系，列出无法无损映射的字段。Related topics：[Agentic RL](../../04-rl-infra/topics/agentic_rl.md)、[RL Framework Selection](../../04-rl-infra/topics/rl_framework_selection.md)。目标流向：paper / topic；当前 Status：NEW。

### 3. Batch invariance：同一份权重不必然产生同一个 policy

**来源：** [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)，Horace He 与 Thinking Machines Lab，official engineering blog，**2025-09-10**。Source ID：`tml:20250910-nondeterminism`。Impact：★★★★★；Decision：**Deep Dive**。

文章把一个容易被“浮点误差”掩盖的问题拆开：请求的 batch size、位置和分块方式可能改变 reduction 次序；即使每个 kernel 在固定输入形状下确定，整个 serving 系统仍可能随 batching 改变结果。它分别处理 RMSNorm、GEMM 与 attention 的 batch invariance，并讨论 chunked prefill、KV 切分如何影响数值路径。核心是固定单个样本的数值计算，而不只是设 seed 或关闭随机采样。[原文机制与实现](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)

RL 的后果是 rollout 与 trainer 即使持有同版权重，也可能计算出不同 logprob。**工程判断：** 在排查 importance ratio 异常前，应先隔离相同 token、相同权重、不同 batching/切分下的差异，再处理真正的 policy staleness。文章中的实现和性能实验有指定模型与硬件边界；不能推出所有 TP/EP collective、低精度路径和 MoE routing 都已经一致。

**Reason：** 它把训推偏差落到可以复现的 kernel 行为，能直接改变 RL 故障定位顺序。**Next：** 设计单请求/混合 batch、prefill/decode、不同 chunk size 的 logprob 对照，分别记录误差与吞吐代价。Related topics：[Agentic RL](../../04-rl-infra/topics/agentic_rl.md)、[FlashAttention](../../01-systems/topics/flashattention.md)、[Distributed Training](../../02-training-infra/topics/distributed_training.md)。目标流向：engineering blog / experiment；当前 Status：NEW。

### 4. NVFP4：缩窄 GEMM 必须同时维护 forward/backward 的一致性

**来源：** [Pretraining Large Language Models with NVFP4](https://arxiv.org/abs/2509.25149v1)，NVIDIA、Felix Abecassis 等，technical report，**2025-09-29**。Source ID：`arxiv:2509.25149v1`。Impact：★★★★★；Decision：**Read**。

4-bit 训练的问题不只是动态范围变小。权重在 forward 与 backward 中沿不同方向量化，可能形成不一致的表示；outlier 和梯度量化偏差也会累积。报告组合使用二维 weight scaling、Wgrad 输入的 Random Hadamard Transform、stochastic rounding 与选择性高精度路径，并报告 12B 模型、10T tokens 的长程实验，与 FP8 baseline 比较收敛质量。[方法与实验](https://arxiv.org/html/2509.25149v1)

这不等于所有训练状态都变成 4-bit：报告保留 FP32 主权重、梯度累积和 optimizer state，attention 与若干敏感层也维持更高精度。**工程判断：** 显存预算仍要逐项计算，验收同时包含 loss 差距、异常值、cast/transform 成本和端到端 step time。该报告证明的是所述混合精度 recipe 的可行性，不能拿 FP4 Tensor Core 峰值当作训练加速承诺。

**Reason：** 它把“能否低精度训练”推进到长程收敛与梯度语义。**Next：** 先制作每类张量的存储/计算/累积精度表，再设计小规模 loss-parity 对照；[Q4](quarterly_signal_2025-Q4.md)继续跟踪 Transformer Engine 的实际 recipe 发布。Related topics：[FP8](../../01-systems/topics/fp8.md)、[Transformer Engine](../../01-systems/topics/transformer_engine.md)。目标流向：tech report / experiment；当前 Status：NEW。

### 5. DeepSeek-V3.2-Exp：稀疏 attention 改变访问模式，也增加新的训练对象

**来源：** [DeepSeek-V3.2-Exp: Boosting Long-Context Efficiency with DeepSeek Sparse Attention](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/840f3c924a6b1604b1998baebf0c5f167e10375a/DeepSeek_V3_2.pdf)，DeepSeek-AI，official report / weights / kernels；[官方发布公告](https://api-docs.deepseek.com/news/news250929/)为 **2025-09-29**，并核对[官方 Hugging Face model card](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp)。Source ID：`deepseek:v3.2-exp-20250929`。Impact：★★★★★；Decision：**Read**。

DSA 先用 lightning indexer 选择 token，再让主 attention 只访问选出的 KV。continued training 先保持 dense attention、训练 indexer 对齐 attention 分布，再转为稀疏训练并调整主模型。因而它既涉及 kernel，也涉及训练目标和架构迁移；不能看成给现有 dense checkpoint 随手加一个推理开关。[官方技术报告](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/840f3c924a6b1604b1998baebf0c5f167e10375a/DeepSeek_V3_2.pdf)

重要边界是主 attention 降为 `O(Lk)`，**indexer 仍为 `O(L²)`**；成本较低不等于总系统已经线性。**工程判断：** 应拆分 indexer、top-k、稀疏 KV 访问与主 attention 的成本，并验证短序列和不同长度分布下是否受益。厂商在特定 H800 服务配置下给出的成本曲线，也不能直接当成任意训练任务的 speedup。

**Reason：** 它提供了从 dense 模型继续训练到可运行稀疏模型的完整机制。**Next：** 先对照训练两阶段与 kernel 数据流，不急于追模型榜单；11 月的 indexer RoPE 修复属于[Q4 后续更新](quarterly_signal_2025-Q4.md)，不当作 9 月原始结论。Related topics：[Long-context Training](../../02-training-infra/topics/long_context_training.md)、[FlashAttention](../../01-systems/topics/flashattention.md)、[MoE](../../02-training-infra/topics/moe.md)。目标流向：tech report / experiment；当前 Status：NEW。

## OpenAI / Anthropic / NVIDIA / DeepSeek Watch

| 厂商 | 当季一手材料与处理 | 工程边界 |
|---|---|---|
| OpenAI | **Observed / Observe**：[Introducing gpt-oss](https://openai.com/index/introducing-gpt-oss/)，2025-08-05，OpenAI；开放权重与部署材料进入背景。 | 可用于研究开放 MoE 的推理路径；该公告不能证明完整预训练集群、恢复策略或 RL scheduler 已公开。产品发布本身不自动占核心阅读名额。 |
| Anthropic | **Observed / Observe**：[A postmortem of three recent issues](https://www.anthropic.com/engineering/a-postmortem-of-three-recent-issues)，2025-09-17，Anthropic。 | context routing、运行时优化与数值问题可以表现为质量退化。它是 serving 事故；迁移为 rollout parity 的检查项属于本文工程推断，不能称为已披露训练事故。 |
| NVIDIA | **Accepted / Read**：9 月 NVFP4 报告，见核心 4。 | 报告出现与 Transformer Engine recipe 的发布是两个事件；Q4 才按具体 release 补实现观察。 |
| DeepSeek | **Accepted / Read**：V3.2-Exp，见核心 5；API 公告与官方 HF 权重入口均已检查。 | 8 月 V3.1、9 月 Terminus 属版本背景；12 月 V3.2 的完整 RL 协议不倒灌进本季。 |

Watch 中的 Observed 项 Impact 为中，Reason 是有直接工程联系但不挤占本季五份核心；Next 是遇到对应实现问题时按指定 source 回查，Related topics 为 [Agentic RL](../../04-rl-infra/topics/agentic_rl.md) / [Fault Tolerance](../../02-training-infra/topics/fault_tolerance.md)，Status：NEW。

## Hugging Face Watch

[Vision Language Model Alignment in TRL](https://huggingface.co/blog/trl-vlm-alignment)（Sergio Paniego 等，2025-08-07）和 [Tricks from OpenAI gpt-oss YOU 🫵 can use with transformers](https://huggingface.co/blog/faster-transformers)（Aritra Roy Gosthipaty 等，2025-09-11）均是 **Hugging Face 官方团队文章，Observed / Observe**。前者提供多模态 trainer 与 vLLM 接入材料，后者把 kernel 分发、MXFP4、TP/EP 和 cache 路径放到具体框架实现中。它们是兼容性与实现参考，不能把示例演示外推成大型集群的性能验证。

TRL、Transformers、Accelerate、PEFT、Kernels 的当季 release / merged PR 入口统一见 [GitHub 历史索引](github_history_2025_to_2026_h1.md)。此处对 Accelerate / PEFT **未额外提升核心材料**，不表示当季没有变化；索引枚举与逐项代码审计是不同覆盖层级。社区文章未因出现在 Hub 自动接受。Impact：中；Reason：补框架兼容性背景；Next：需要复现时锁定当季版本核对 trainer/backend 接口；Related topics：[RL Framework Selection](../../04-rl-infra/topics/rl_framework_selection.md)、[FP8](../../01-systems/topics/fp8.md)；Status：NEW。

## RL Framework Watch

本表是 **Historical Review**，与当时 frontier 的判断分开。所有行均维持 **Observed / Observe、Status NEW**；没有因为 PR 名称包含 async 就推断吞吐、正确性和恢复协议全部得到验证。历史定向补证、待重建元数据范围与时间过滤见 [GitHub 历史索引](github_history_2025_to_2026_h1.md)。

| 框架 | 本季观察入口 | 子系统、价值与 AReaL 迁移问题 |
|---|---|---|
| AReaL | 当季 merged PR / release 历史索引；上半年异步路线继续作为背景。 | `scheduler / rollout`：Next 是区分设计首次披露、版本发布与后续修复，不从当前 README 反推当季成熟度。 |
| verl | 当季 merged PR / release 历史索引。 | `training / rollout`：本报告未核验出需要占第五份以外阅读名额的单项变化；迁移判断留到具体 diff。 |
| slime | [#258](https://github.com/THUDM/slime/pull/258)，2025-09-04，新增 fully async example；讨论中仍有依赖导入问题。 | `scheduler / rollout`：仅作为示例级证据，需要检查队列、policy version 和退出条件；可迁移到 AReaL 的是验证方法，不是默认配置或成熟度结论。 |
| ROLL | [#111](https://github.com/alibaba/ROLL/pull/111)，2025-07-31，async / agentic 设计入口。 | `scheduler / data/trajectory path`：关注多轮任务怎样进入训练；Next 是核对实现与失败路径，设计或 PR 合入本身不等于生产验证。 |
| OpenRLHF | 当季 merged PR / release 历史索引。 | `rollout / training`：保持框架比较背景，未对当季各路径逐一做代码验收，暂不提出 AReaL 的直接移植建议。 |
| NeMo RL | 当季 merged PR / release 历史索引。 | `training / inference backend`：后续 Q4 的 Nemotron 报告能提供实际使用证据；不提前把该报告中的配套实现归到本季。 |

Emerging framework：Agent Lightning 以论文和当时系统设计为 Accepted / Read，子系统为 `data/trajectory path`；迁移到 AReaL 的价值是执行与训练接口的分离，不能等同于直接替换 scheduler。上表其余观察 Impact 为中；Reason 是保留横向实现入口，Next 均为有实际问题时按历史 diff 定点核验，Related topics：[Agentic RL](../../04-rl-infra/topics/agentic_rl.md)、[RL Framework Selection](../../04-rl-infra/topics/rl_framework_selection.md)。

## 今天怎样继续，而不是把五份都排成必读

若正在做 RL infra，优先选择 **Kimi K2 的 engine switching + batch invariance**：一个解释状态搬运为何卡住，另一个解释训练为什么“看似同版权重却不一致”。若正在做低精度或长上下文，则先选择 **NVFP4 或 DSA 中与当前 workload 对应的一份**。Agent Lightning 留作环境/agent 接入时的接口参考。

覆盖缺口：本文核验了核心材料的标题、作者、原始日期与机制，没有复现其 benchmark，也没有逐条审计所有框架 PR。公开报告对大型作业故障率、checkpoint 完整性和网络拓扑披露仍不充分；不能据此比较厂商真实训练成本。来源原始元数据、方法锚点与证据限制见 [2025 H2 来源核验 JSON](audits/2026-09-22-history/2025-h2-sources.json)。

导航：[历史/月度回看总入口](monthly_reviews.md) · [GitHub 历史索引](github_history_2025_to_2026_h1.md) · [2025 Q2](quarterly_signal_2025-Q2.md) · [2025 Q4](quarterly_signal_2025-Q4.md) · [2026-08：后续状态与环境边界](monthly_signal_2026-08.md)

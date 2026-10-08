# 2025 Q4 训练基础设施复盘：环境与策略证据成为训练契约

> 整合说明（2026-10-08）：本文保留 9/22 的来源核验与复评日期；[GitHub 历史定向补证](github_history_2025_to_2026_h1.md)已保存，但全窗口事件索引重建仍因 API 连接失败待补，不能据此声称历史 GitHub 全覆盖。

> **2026-09-22 历史回看**，原始材料窗口为 2025-10-01 至 2025-12-31。以下 4 项是本次精选，不是补造当时的 frontier scan。First seen / 本次核验日期统一为 **2026-09-22**，Status 均为 **NEW**；Decision 是现在的阅读建议，不代替用户标记已读，也不改变旧扫描游标和 Accepted 数。

## 从“能跑 RL”走到“数据到底代表什么”

接续 [Q3](quarterly_signal_2025-Q3.md) 的 batch invariance，Q4 的材料把问题推进一层：**优化器收到的样本，是否来自它以为的 policy、动作空间和环境？** DeepSeek-V3.2 明确讨论 routing 与 sampling mask；OpenEnv 尝试给环境一个可共享的接口；Anthropic 用真实生产来源的编码环境研究 reward hacking；Nemotron 3 Nano 则展示多环境训练怎样接到实际的 trainer / inference backend。

**趋势推断：** Agentic RL 的扩展开始依赖两类基础设施：一类保留 token、logprob、route 与 policy 关系，另一类生产和验证环境、任务及奖励。两者需要一起工作；再快的 rollout 也不能修复错误的判分，规范化的环境接口也不会自动解决数值或 off-policy 偏差。

低精度路线同时从报告走向框架接入：Q3 的 NVFP4 方法在 Q4 的 Transformer Engine release 中出现具体训练 recipe。但“支持一种格式”与“某份模型报告使用该格式”仍须分开核对，尤其不能把 Nemotron 3 家族路线误写成 Nano 已完成 NVFP4 预训练。

## 四份核心材料

### 1. OpenEnv：环境开始成为可交换的训练构件

**来源：** [Building the Open Agent Ecosystem Together: Introducing OpenEnv](https://huggingface.co/blog/openenv)，Joseph Spisak、Davide Testuggine、Zach Wentz、Pierre Andrews、Sanyam Bhutani、Hamid Shojanazeri、Pankit Thapar、Emre Guven、Lewis Tunstall、Vaibhav Srivastav；Meta-PyTorch / Hugging Face 官方联合工程发布，**2025-10-23**。Source ID：`hf:openenv-20251023`。Impact：★★★★☆；Decision：**Read**。

Agent 训练的瓶颈不只有 GPU。每类任务都需要工具、依赖、执行上下文和观察结果；如果这些细节与某个 trainer 紧密绑定，换框架和复现实验都会很重。OpenEnv 发布时提供 `step / reset / close` 接口、Docker 环境示例，并以 0.1 RFC 讨论组件关系、打包、隔离和通信，让环境能作为共享构件进入 Hub。[发布时的实现与 RFC 边界](https://huggingface.co/blog/openenv)

它当时仍是早期接口与生态建设，文章明确把多项框架集成写作进行中。**工程判断：** 标准接口降低接入成本，却不自动证明环境可复现、判分可靠或租户隔离完整。对 AReaL 的实际价值是让 rollout adapter 与环境生命周期分开：先明确 reset 后状态、timeout 和 close 的行为，再讨论大规模调度。

**Reason：** 它让环境从一次性训练脚本走向可分发的接口和 artifact。**Next：** 用一个最小环境画出 agent、environment、reward 和 trainer 的边界，并记录 image / dependency / task / verifier 版本；这些记录项是本文建议，不是声称 0.1 已全部实现。Related topics：[Agentic RL](../topics/agentic_rl.md)、[Fault Tolerance](../topics/fault_tolerance.md)。目标流向：engineering blog / experiment；当前 Status：NEW。

### 2. Anthropic 的 reward-hacking 研究：判分路径本身会塑造训练结果

**来源：** [Natural Emergent Misalignment from Reward Hacking in Production RL](https://arxiv.org/abs/2511.18397v1)，Monte MacDiarmid、Benjamin Wright、Jonathan Uesato 等，Anthropic / Redwood Research；[官方研究文章](https://www.anthropic.com/research/emergent-misalignment-reward-hacking)首次发布 **2025-11-21**，arXiv v1 为 **2025-11-23**。Source ID：`arxiv:2511.18397v1`。Impact：★★★★★；Decision：**Read**。

该研究选择来自真实 Claude 训练的可被 reward hack 的编码环境，先让研究模型获得相应策略知识，再观察 RL 如何学到绕过任务目标的高奖励行为，以及是否泛化到其他评估。它的系统含义是：reward service 成功返回高分，不能替代“任务真的完成”的证据；training job 不 crash，也不能证明训练数据的语义正确。[实验设置](https://arxiv.org/abs/2511.18397v1)

必须保留实验边界：作者主动选择易被利用的环境并注入相关知识，这是受控研究，**不是宣告所有线上 Claude 发生了同样问题，也不是一次已披露的集群事故**。论文还比较若干缓解方法；不能仅凭单项有效就宣布环境已经可信。**工程判断：** 将 reward 聚合值与测试执行证据、失败原因分开保存，并对训练分数异常上升设置复核路径，比只看 reward 曲线更有诊断价值。

**Reason：** 它提供生产来源的环境证据，说明 verifier 错误会被优化器放大，值得进入 RL Infra 的故障分类。**Next：** 为一个编码环境列出“进程正常结束、测试实际执行、任务目标满足”的独立验收条件，并设计有明确隔离范围的负例测试。Related topics：[Agentic RL](../topics/agentic_rl.md)、[Fault Tolerance](../topics/fault_tolerance.md)。目标流向：paper / playbook；当前 Status：NEW。

### 3. DeepSeek-V3.2：MoE routing 与 sampling mask 进入训练数据契约

**来源：** [DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models](https://arxiv.org/abs/2512.02556v1)，DeepSeek-AI、Aixin Liu 等；technical report / weights。[官方 API 公告](https://api-docs.deepseek.com/news/news251201/)为 **2025-12-01**，arXiv v1 为 **2025-12-02**；同时核对[官方 Hugging Face model card](https://huggingface.co/deepseek-ai/DeepSeek-V3.2)。Source ID：`arxiv:2512.02556v1`。Impact：★★★★★；Decision：**Deep Dive**。

报告指出，多次更新复用 rollout，以及训练/推理实现差异，都会造成 off-policy 偏差。其稳定化措施包括对高偏差的负 advantage 序列做 masking，保留采样时的 expert routing，并让训练沿用 top-p/top-k 的 sampling mask。这里不是“再调一个 PPO clip 参数”就能替代的工作：route 和可选 token 集合也决定正在优化哪个计算路径和动作空间。[§3.1 Scaling GRPO](https://arxiv.org/html/2512.02556v1#S3.SS1)

**时间纠正：** 报告明确说 Keep Routing 自 DeepSeek-V3-0324 起已用于其 RL pipeline；本季度是该报告中的系统性披露，不能将机制首次使用时间写成 12 月。环境侧，报告又描述 issue/patch/test 的筛选、自动搭建可执行环境，以及联合生成 environment、tools、task、verifier 的流程。**工程判断：** 数据契约既要覆盖 policy evidence，也要覆盖任务如何被构造和验收，不能只保存 prompt 与最终 reward。[环境生产流程](https://arxiv.org/html/2512.02556v1#S3.SS2.SSS3)

**Reason：** 这是连接 MoE 计算路径、off-policy 稳定性与 agent 数据生产的工业报告。**Next：** 优先读 §3.1，再读 §3.2.3；对照 AReaL 的 trajectory schema 列出已保存、可重建与丢失的信息，不直接照搬 mask 阈值。Related topics：[Agentic RL](../topics/agentic_rl.md)、[MoE](../topics/moe.md)、[Distributed Training](../topics/distributed_training.md)。目标流向：tech report / topic / experiment；当前 Status：NEW。

### 4. Nemotron 3 Nano：多环境系统已落地，同步与精度边界仍需逐项读

**来源：** [Nemotron 3 Nano: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning](https://arxiv.org/abs/2512.20848v1)，NVIDIA、Aaron Blakeman 等；technical report / model recipe。家族与 Nano [官方公开发布](https://research.nvidia.com/labs/nemotron/Nemotron-3/)为 **2025-12-15**，本次固定采用的 arXiv v1 为 **2025-12-23**，不能把后一个日期当模型首发。Source ID：`arxiv:2512.20848v1`。Impact：★★★★★；Decision：**Read**。

报告的关键系统证据在 NeMo Gym / NeMo RL 分工：agent server 执行 rollout，model server 封装推理并保留 token/logprob 等元数据，resource server 负责验证；NeMo RL 控制训练，接入 Megatron-Core 与 vLLM。把不同环境共同用于 RL，要求这些接口能统一交付训练证据，而不仅是各自返回文本。[§3.2.4 Infrastructure](https://arxiv.org/html/2512.20848v1#S3.SS2.SSS4)

两条边界尤其值得记住：报告使用 **synchronous GRPO**，并非只要多环境就必须 fully async；它冻结 router weights，但仍更新 expert bias，因此不能说 routing 从此完全不变。精度部分则明确是 BF16 post-training 后进行选择性 FP8 PTQ，保留敏感 attention/Mamba 路径；这份 Nano 报告不能用来证明 NVFP4 预训练。[算法与量化](https://arxiv.org/html/2512.20848v1)

**Reason：** 有模型训练报告支撑的组件分工，比框架功能清单更能说明真实系统怎样组合。**Next：** 画出 NeMo Gym 到 trainer 的 token/logprob/reward 流，与 AReaL 对照；量化时分别验收模型质量、backend 支持和 rollout parity，不复用厂商局部吞吐倍数作容量承诺。Related topics：[Agentic RL](../topics/agentic_rl.md)、[MoE](../topics/moe.md)、[FP8](../topics/fp8.md)。目标流向：tech report / experiment；当前 Status：NEW。

## OpenAI / Anthropic / NVIDIA / DeepSeek Watch

| 厂商 | 当季材料与处理 | 应保留的边界 |
|---|---|---|
| OpenAI | **Observed / Observe**：[Update to GPT-5 System Card: GPT-5.2](https://openai.com/index/gpt-5-system-card-update-gpt-5-2/)，OpenAI，2025-12-11。 | 系统卡是工业评估资料；本次未核验到能据此复原训练调度、权重同步或 checkpoint 协议的细节，不能因模型发布自动 Accepted。 |
| Anthropic | **Accepted / Read**：reward-hacking 研究，见核心 2。 | 保留受控实验设置与生产来源环境的区别；不写成公开生产事故。 |
| NVIDIA | **Accepted / Read**：Nemotron 3 Nano，见核心 4；下述 Transformer Engine / NCCL releases **Observed / Observe**。 | 将模型实际 recipe、框架支持和家族未来路线分开。 |
| DeepSeek | **Accepted / Deep Dive**：V3.2，见核心 3；官方 API 公告与 HF 权重页均核验。 | [V3.2-Exp 2025-11-17 更新](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp#update)记录 indexer RoPE 要 non-interleaved，而 MLA 使用 interleaved layout；这是 9 月材料的实现修正，**Observed / Observe**，不再计一个新模型。 |

低精度的实际接入由 release 接续：[Transformer Engine v2.8](https://github.com/NVIDIA/TransformerEngine/releases/tag/v2.8)（2025-10-07）加入 NVFP4 training recipe，[v2.10](https://github.com/NVIDIA/TransformerEngine/releases/tag/v2.10)（2025-12-11）继续补 NVFP4 GroupedLinear 与 quantized CUDA graphs。通信侧，[NCCL v2.28.7-1](https://github.com/NVIDIA/nccl/releases/tag/v2.28.7-1)（2025-10-18）提供 Device API / GIN 方向，[v2.28.9-1](https://github.com/NVIDIA/nccl/releases/tag/v2.28.9-1)（2025-11-10）修正 main/proxy 操作排序以处理大规模 hang。这些是**官方 release 事实**；能否用于特定模型、硬件与网络，仍需兼容性和故障验证，不推断为无条件提速。

上述 Observed 项 Impact：中至高；Reason：证明论文到可用软件还存在版本、硬件和正确性边界；Next：复现时锁定 release 并核对限制，遇到 DSA 质量异常先排查 layout；Related topics：[Transformer Engine](../topics/transformer_engine.md)、[NCCL](../topics/nccl.md)、[Fault Tolerance](../topics/fault_tolerance.md)；Status：NEW。

## Hugging Face Watch

**Accepted / Read：OpenEnv**，见核心 1，明确属于 Meta-PyTorch 与 Hugging Face 联合官方发布，不是普通社区帖子。本季核心关心环境接口的形成；后续 2026 年的生态扩张不用于证明 2025-10 的集成已完成。

TRL、Transformers、Accelerate、PEFT、Kernels 的当季 release / merged PR 按 [GitHub 历史索引](github_history_2025_to_2026_h1.md)观察，均 **Observed / Observe**，不根据当前 docs 的功能清单补写当年能力。HF Blog 的已核验正文以 OpenEnv 为主；未对其所有当季文章逐篇审读。这一覆盖限制不等于上述库没有重要变化。Impact：中；Reason：控制阅读负担并保留实现入口；Next：与实际复现实验相关时，再按历史版本检查 adapter、distributed training 与 backend 接口；Related topics：[RL Framework Selection](../topics/rl_framework_selection.md)、[Distributed Training](../topics/distributed_training.md)；Status：NEW。

## RL Framework Watch

本表为 **Historical Review**。除已并入 Nemotron 核心条目的实际系统使用证据外，框架单项均为 **Observed / Observe、Status NEW**；即使后续补齐 PR/release 元数据，也不等于每个 diff 都已完成正确性审计。

| 框架 | 本季观察入口 | 子系统、工程影响及 AReaL 可迁移性 |
|---|---|---|
| AReaL | [#667](https://github.com/areal-project/AReaL/pull/667)，2025-12-04，vocab-parallel logprobs。 | `training / data/trajectory path`：关注 logits 分片下 logprob 的归约与目标 token 归属；Next 是同时核对数值与显存行为，不能只看输出 shape。 |
| verl | 当季 merged PR / release 历史索引。 | `scheduler / rollout`：保留作为框架横向参照；本页未额外选定已代码验收的变更，迁移建议待具体实现证据。 |
| slime | [#906](https://github.com/THUDM/slime/pull/906)，2025-11-25，题名为 FSDP2 true-on-policy。 | `training / inference backend`：作者正文明确该实现没有正确 backward，因此仅视为局部前向一致性尝试；Next 是先验证梯度，不能据标题声称完整训练 parity，更不能直接移植到 AReaL。 |
| ROLL | 当季 merged PR / release 历史索引。 | `scheduler / data/trajectory path`：继续跟踪 agentic 执行，但没有把上一季设计入口或当前功能当成本季度新发现。 |
| OpenRLHF | 当季 merged PR / release 历史索引。 | `rollout / training`：本页保留覆盖入口；未额外给出未经 diff 验证的吞吐或 AReaL 移植结论。 |
| NeMo RL | **Accepted as evidence within core 4**：Nemotron 3 Nano §3.2.4–3.2.5；框架本身非本季度首次诞生。 | `training / rollout / inference backend`：Gym 的 server 分工与 token/logprob 保真可用于 AReaL 接口对照；报告的同步训练不能改写为 fully async。 |

Emerging framework / subsystem：OpenEnv 被接受为环境接口证据，NeMo Gym 被纳入 Nano 的实际训练架构；两者都不是单凭 repo 出现就与成熟 RL trainer 等量齐观。观察项 Impact：中；Reason：为本季 policy evidence 与 environment 接口主线保留代码落点；Next：围绕实际问题定点审查并以实验判断迁移；Related topics：[Agentic RL](../topics/agentic_rl.md)、[RL Framework Selection](../topics/rl_framework_selection.md)。详见 [GitHub 历史索引](github_history_2025_to_2026_h1.md)。

## 向 2026 年带走的两个问题

维护 rollout/trainer 接口时，优先读 **DeepSeek-V3.2 §3.1 + Nemotron 的基础设施段落**，产出一张 policy evidence 字段表：采样概率从哪里来，route 是否保留，mask 是否一致，哪些元数据会在 adapter 中丢失。建设环境平台时，优先读 **OpenEnv + Anthropic 研究的实验设置**，区分环境可调用、可复现与判分可信。

这些问题后来在 [2026-08 月报](monthly_signal_2026-08.md)中扩展到 in-flight 样本、weight-sync admission、恢复 frontier 与环境变更治理。这里是后见的阅读联系，**不表示 2025 年四份材料已经实现 2026 年全部能力，也不证明它们之间存在直接继承关系**。

覆盖缺口：本次未复现性能或稳定性数字，没有逐 PR 审计各框架，也无法从公开资料还原厂商完整生产集群。部分发布页持续更新，故关键算法以 arXiv v1、明确日期的公告与 release 为锚；Nano 家族页当前新增产品不能回填 2025。完整标题、作者、日期和方法证据见 [2025 H2 来源核验 JSON](audits/2026-09-22-history/2025-h2-sources.json)。

导航：[历史/月度回看总入口](monthly_reviews.md) · [GitHub 历史索引](github_history_2025_to_2026_h1.md) · [2025 Q3](quarterly_signal_2025-Q3.md) · [2026-01](monthly_signal_2026-01.md) · [2026-08：后续状态与环境边界](monthly_signal_2026-08.md)

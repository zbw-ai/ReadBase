# 2025 Q1 训练基础设施复盘：Reasoning RL 改变了训练系统的工作负载

> 整合说明（2026-10-08）：本文保留 9/22 的来源核验与复评日期；[GitHub 历史定向补证](github_history_2025_to_2026_h1.md)已保存，但全窗口事件索引重建仍因 API 连接失败待补，不能据此声称历史 GitHub 全覆盖。

> 原始材料窗口：2025-01-01 至 2025-03-31，按首次公开版本归季。回看与补录时间：2026-09-22。本文是 Historical Retrospective，不是补造当时的 frontier scan，不改变扫描游标、原 Accepted 或用户阅读状态。
>
> 本季精选 **3 篇论文/报告、1 组官方实现**，整理为四条主线。R1 与 DAPO 承接既有 backfill；NSA、DeepEP/DeepGEMM 是这次复盘补齐的材料。下文 `Lifecycle: NEW` 表示本次阅读候选记录，不覆盖旧笔记的状态，也不代表用户已读。来源原始元数据见[核验记录](audits/2026-09-22-history/2025-h1-sources.json)。

## 先读这页

**趋势推断：2025 年第一季度，训练平台开始同时服务两种很不一样的计算。** 预训练仍强调大批量、规律的 GEMM 与 collective；reasoning RL 则不断生成长度不同的回答、执行验证、筛选样本，再把新权重送回生成端。后者的训练效率不能只靠 trainer 的 MFU 解释。即使去掉 critic，也不代表整个作业变得简单：成本会转移到 rollout、reward、样本分组以及权重交付。

本季另一条线是模型与硬件的共同设计。NSA 让 attention 的稀疏模式适合训练和 GPU 访存；DeepEP 与 DeepGEMM 把 MoE dispatch、grouped GEMM、FP8 scaling 的接口暴露为可读代码。**今天回读它们，应留下两个问题：训练所消费的样本有没有被实现细节改变？算法减少的 FLOPs，是否真的缩短了 GPU 上的关键路径？**

| 顺序 | 材料 | 本季解决的问题 | 今天留下的判断 |
|---|---|---|---|
| 1 | DeepSeek-R1 | 可验证奖励如何驱动 reasoning 训练 | rollout 与 verifier 是训练系统的一部分 |
| 2 | DAPO | 为什么照着 GRPO 公式仍复现失败 | sampling、truncation、loss reduction 会改变优化目标 |
| 3 | Native Sparse Attention | 长上下文 attention 的成本如何下降 | 稀疏比例必须转换成有效访存与 Tensor Core 工作 |
| 4 | DeepEP + DeepGEMM | MoE 的通信与 FP8 GEMM 如何落地 | routing、layout、量化和拓扑必须放在同一执行路径检查 |

## 1. DeepSeek-R1：平台需要生产可验证经验，而不只是读入静态 token

**原文机制。** [DeepSeek-R1 首版](https://arxiv.org/html/2501.12948v1) 区分两条路径：R1-Zero 从 base model 直接进行 RL；R1 则包含 cold-start SFT、reasoning RL、rejection sampling 后的 SFT 和后续 RL。GRPO 用同一 prompt 下的回答组估计相对优势，省掉通常与 policy 同量级的 critic。数学答案和代码测试等可验证反馈，成为其中的重要奖励来源。这是有具体训练阶段和开放权重的工业报告，但不是完整训练平台的开源交付。

**系统后果——本仓库推断。** 固定数量的 prompt 不能保证固定数量的生成 token，更不能保证稳定的 step time。长回答增加 decode 时间，回答组增加组内等待，验证器又带来 CPU、执行环境和失败处理需求。平台因此需要区分“生成了多少”“验证了多少”“真正进入更新多少”；只优化 backward 或只比较 serving tokens/s，都会漏掉一部分成本。R1-Zero 没有 SFT 冷启动，也不意味着从随机初始化训练，更不意味着没有任务数据和奖励设计。

**今天怎么读。** 优先读第 2 节的训练流程和奖励设计，再画出 actor、rollout、verifier、过滤与数据回流的边界。报告没有充分披露的调度、集群和恢复机制，不能根据模型效果反推。DeepSeek-V3 的预训练资源数字也不能拿来充当 R1 全流程成本。[既有 R1 笔记](../tech_reports/deepseek_r1.md)与[2025-01 backfill](backfill/2025-01.md)继续保留，本次补的是系统需求导读。

| 字段 | 本次判定 |
|---|---|
| Source ID / 类型 | `arxiv:2501.12948v1` / industrial technical report |
| 原始时间 / 本次核验 | 模型 2025-01-20 发布；论文 v1 为 2025-01-22；2026-09-22 回看 |
| Impact / Decision | 高 / **Read** |
| Reason / 一句话价值 | 解释 rollout、规则验证与多阶段数据生产为何进入训练平台的核心工作负载 |
| Related topics | [Agentic RL](../topics/agentic_rl.md)、[MoE](../topics/moe.md)、[Rollout Latency](../playbooks/rollout_latency.md) |
| Next / 目标去向 | 对照现有笔记补一张训练数据流与成本边界表；topic / insight 候选 |
| Lifecycle | **NEW**：本次复盘候选；旧笔记已有状态不变 |

## 2. DAPO：训练 recipe 的细节就是系统正确性

**原文机制。** [DAPO 首版](https://arxiv.org/html/2503.14476v1) 给出四个互相关联的改动：放宽上侧 clipping 以维持探索；dynamic sampling 丢掉组内全对或全错、相对优势没有区分度的 prompt 组，并补充采样；按 token 聚合 policy loss；为过长与被截断回答处理奖励噪声。它基于 verl，公开代码与数据，填补了“知道算法名称但复现不出训练曲线”的缺口。

**系统后果——本仓库推断。** Dynamic sampling 使“请求的 batch”和“参与更新的 batch”分离。若监控只数进入生成器的 prompt，就会高估有效吞吐；若 trainer 对每个 micro-batch 单独归一化，再简单平均，可能偏离预期的全局 token 权重。截断也不能只当作内存保护开关，因为它会改变 reward、mask 和训练分布。验收应同时观察组内 reward 方差、有效组比例、长度分布与 loss denominator。

**阅读取舍。** 作者在 Qwen2.5-32B 实验中报告 AIME 2024 达到 50 分，并与 R1-Zero-Qwen-32B 的 47 分及训练步数比较。这是特定 recipe 的结果，**训练步数减少不是总 GPU-hours 等比例减少**，dynamic sampling 的额外生成尤其需要计入。今天更值得复核四项改动的相互作用，而不是把超参数原样复制到不同 verifier 或 Agent 任务。[既有 DAPO backfill](backfill/2025-03.md)可作为后续深读入口。

| 字段 | 本次判定 |
|---|---|
| Source ID / 类型 | `arxiv:2503.14476v1` / paper + open training recipe |
| 原始时间 / 本次核验 | arXiv v1 2025-03-18；正文标注 March 17；2026-09-22 核验，均属 Q1 |
| Impact / Decision | 高 / **Deep Dive** |
| Reason / 一句话价值 | 让采样、长度限制和 loss reduction 从外围配置变成可验证的训练语义 |
| Related topics | [Agentic RL](../topics/agentic_rl.md)、[Distributed Training](../topics/distributed_training.md) |
| Next / 目标去向 | 固定同一批轨迹，比较不同 micro-batch 切分下的梯度；paper / experiment 候选 |
| Lifecycle | **NEW** |

## 3. Native Sparse Attention：少算 attention，先让稀疏模式适合 GPU

**原文机制。** [Native Sparse Attention（NSA）首版](https://arxiv.org/html/2502.11089v1) 同时使用压缩 token、选择重要 token block、sliding window 三条分支，并通过 gating 组合。压缩分支承担粗粒度全局信息，selection 保留细节，窗口分支处理局部模式。它从训练阶段采用稀疏结构；selection kernel 按共享 KV block 的 GQA query heads 组织工作，减少不规则访存带来的浪费。

**系统后果——本仓库推断。** 稀疏算法不是在 dense attention 外面加一个 mask 就完成了。block 大小、head sharing、索引生成、读取连续性和 backward 都会决定实际速度。若只报告被跳过的 token 比例，可能掩盖 gather、调度和小 GEMM 的成本。它也不是 FlashAttention 的无损替代：FlashAttention 改写相同 dense attention 的 IO 路径，NSA 改变模型实际使用的信息和训练结构。

**今天的价值。** 将它作为训练原生稀疏 attention 的设计样本：先理解三条分支的职责，再检查可训练性、kernel 工作粒度和模型质量的联动。论文中的 64K 序列实验不构成所有长度、模型和 GPU 上的端到端加速保证，也不能据此宣称后来某个 DeepSeek 产品采用了完全相同的 NSA。

| 字段 | 本次判定 |
|---|---|
| Source ID / 类型 | `arxiv:2502.11089v1` / attention architecture + kernel paper |
| 原始时间 / 本次核验 | 2025-02-16 / 2026-09-22 |
| Impact / Decision | 高 / **Read** |
| Reason / 一句话价值 | 把稀疏算法收益约束到训练、GQA 数据复用与实际访存路径上 |
| Related topics | [Long-context Training](../topics/long_context_training.md)、[FlashAttention](../topics/flashattention.md)、[Context Parallelism](../topics/context_parallelism.md) |
| Next / 目标去向 | 对照 dense attention，列出 forward/backward 工作量、索引与内存开销；paper / experiment 候选 |
| Lifecycle | **NEW** |

## 4. DeepEP + DeepGEMM：工业报告里的 MoE 联合优化成为可检查接口

**证据固定在当季代码。** 本次读取 [DeepEP 的 2025-03-28 快照](https://github.com/deepseek-ai/DeepEP/blob/26fa72d80f2ec3de21a596b0e47ffa1c19dc6ac4/README.md)及[DeepGEMM 的 2025-03-26 快照](https://github.com/deepseek-ai/DeepGEMM/blob/c57699ac933a93651c34d365797c2d8b41a4765b/README.md)。日期证明这些接口在 Q1 已存在，不将 commit 时间冒充首次公开发布时间，也不使用当前 main 的 Blackwell 等后续能力描述当季版本。

DeepEP 将 MoE dispatch/combine 分成训练与 prefill 需要的高吞吐路径、decode 需要的低延迟路径；前者处理 NVLink 与 RDMA 两类带宽域并支持控制通信占用的 SM。DeepGEMM 则实现细粒度 scaling 的 Hopper FP8 GEMM，提供普通与 grouped GEMM，并针对累加精度采用 promotion；当季 README 明确它只包含 GEMM kernel，cast、transpose 等可能需要由调用方完成或融合。

**系统后果——本仓库推断。** 两个库相接时，瓶颈可能落在 token permutation、alignment、scaling layout 或通信 buffer 生命周期上。单独测得更高带宽与更高 GEMM TFLOPS，不能直接相乘得到训练加速。训练、prefill 与 decode 的 shape 也不同，不能只拿小 batch decode 最优配置来配置训练集群。今天读它们的价值是检查 dispatch 输出怎样变成 grouped GEMM 输入，而不是把特定网络环境的参数当成通用调优规则。

| 字段 | 本次判定 |
|---|---|
| Source ID / 类型 | `github:deepseek-ai/DeepEP:2025-Q1-snapshot`、`github:deepseek-ai/DeepGEMM:2025-Q1-snapshot` / official implementation |
| 原始窗口 / 本次核验 | Q1 历史快照 / 2026-09-22；精确开源首日未在本次另行证明 |
| Impact / Decision | 高 / **Deep Dive** |
| Reason / 一句话价值 | 将 MoE 路由、网络域、FP8 scaling 与 GEMM 的接口约束变成可检查的工程对象 |
| Related topics | [MoE](../topics/moe.md)、[FP8](../topics/fp8.md)、[NCCL](../topics/nccl.md)、[DeepSeek-V3](../tech_reports/deepseek_v3.md) |
| Next / 目标去向 | 用固定 shape 画 dispatch → layout/cast → GEMM → combine 时间线；engineering blog / experiment 候选 |
| Lifecycle | **NEW** |

## OpenAI / Anthropic / NVIDIA / DeepSeek Watch

| 厂商与当季一手材料 | Triage | 工程取舍 |
|---|---|---|
| OpenAI：[o3-mini，2025-01-31](https://openai.com/index/openai-o3-mini/) | **Observed** | reasoning effort 可调体现推理计算预算的重要性；未由该发布页核验出足够训练调度、通信或恢复机制，不从能力榜单推导训练架构 |
| Anthropic：[Claude 3.7 Sonnet and Claude Code，2025-02-24](https://www.anthropic.com/news/claude-3-7-sonnet) | **Observed** | extended thinking 与 coding workload 说明长轨迹需求；发布页不是可复现的 RL infra 报告 |
| NVIDIA：[Blackwell 软件栈文章，2025-03-18](https://developer.nvidia.com/blog/nvidia-blackwell-delivers-world-record-deepseek-r1-inference-performance/)、[NeMo RL v0.1.0，2025-03-22 UTC](https://github.com/NVIDIA-NeMo/RL/releases/tag/v0.1.0) | **Observed** | 留意 cuDNN/CUTLASS 的低精度与架构绑定；NeMo RL 当季已公开 Ray、FSDP、vLLM、GRPO 路径。v0.1.0 明列 checkpoint gather 可造成大模型 OOM；有 release 不等于恢复路径已完整 |
| DeepSeek：[API changelog](https://api-docs.deepseek.com/updates/)、[官方 HF R1](https://huggingface.co/deepseek-ai/DeepSeek-R1)、[官方 HF V3-0324](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324)、上述报告与固定代码 | **Accepted / Observed** | Accepted 为 R1、NSA、DeepEP/DeepGEMM；3 月 24 日 V3-0324 权重/API 交付保持 Observed，不当成新训练系统论文。V3 原始报告首发于 2024-12，作为上游背景，不重复计入 Q1 首发论文 |

## Hugging Face Watch

HF 官方团队的 [Open-R1，2025-01-28](https://huggingface.co/blog/open-r1)（Elie Bakouch、Leandro von Werra、Lewis Tunstall）保持 **Observed**：它明确指出 R1 权重之外仍缺数据、训练代码和可复现 recipe。本季用它理解“开放模型”和“开放训练流程”的差距；项目计划不等于已完成全量复现。

| 官方组件抽样 | Triage | 与本季的关系 |
|---|---|---|
| [TRL v0.15.2，02-25](https://github.com/huggingface/trl/releases/tag/v0.15.2) | Observed | 修复与 pin Liger/vLLM 依赖，说明 rollout backend 的版本兼容也是 recipe 的组成；不把该小版本当作 GRPO 首发 |
| [Transformers v4.49.0，02-17](https://github.com/huggingface/transformers/releases/tag/v4.49.0) | Observed | FP8 quantization 与模型适配有运行价值，但不等于端到端 FP8 训练已经完成 |
| [Accelerate v1.5.0，03-12](https://github.com/huggingface/accelerate/releases/tag/v1.5.0)；[PEFT v0.15.0，03-19](https://github.com/huggingface/peft/releases/tag/v0.15.0) | Observed | 已核对官方 release；HPU 支持、adapter 方法更新不足以替换本季主要阅读问题 |
| Kernels | Not found / not verifiable in this scan | 本次没有确认 Q1 内足以独立收录的官方发布；不把 Q2 Kernel Hub 博客或 2026 的 Hub 仓库类型更新倒填到 Q1 |

这些是官方团队博客与官方 release 的抽样，不是 HF 社区文章全量回扫，也不宣称覆盖所有 PR。

## RL Framework Watch

本节为 **2026-09-22 Historical Audit**。它说明当季可证实的系统方向，不改变历史扫描计数。完整 GitHub 窗口与索引限制见[历史审计](github_history_2025_to_2026_h1.md)。

| 框架 | 当季证据与 Triage | 子系统 / 工程维度 | 对 AReaL 的参考 |
|---|---|---|---|
| AReaL | **Not found / not verifiable in this scan**：本次核验的异步系统论文首版在 5 月 | scheduler / staleness | 不把 Q2 的论文能力反填 Q1；前身系统和未核验内部历史不在此作结论 |
| verl | **Accepted**：DAPO 明确基于 verl；主线 2 已计入 | training、rollout、data/trajectory path / 有效组、loss reduction | 迁移 sampling 与 mask 语义，不能只移植配置名 |
| slime | **Not found / not verifiable in this scan**：未确认 Q1 公开版本 | rollout / 历史可用性 | 不以当前 README 判断 Q1 可用能力 |
| ROLL | **Not found / not verifiable in this scan**：本次确认的报告首版在 6 月 | scheduler、data/trajectory path | 当季历史能力未证实，不作负面的不存在判断 |
| OpenRLHF | **Observed**：[Ray collective weight sync #704，02-04](https://github.com/OpenRLHF/OpenRLHF/pull/704)、[universal checkpoint #891，03-20](https://github.com/OpenRLHF/OpenRLHF/pull/891)，由本次联合 GitHub 审计核验 | weight sync、checkpoint/recovery / 传输与恢复契约 | 对照不同并行布局的权重交付、状态恢复单位；未在此独立复现 |
| NeMo RL | **Observed**：[v0.1.0](https://github.com/NVIDIA-NeMo/RL/releases/tag/v0.1.0) | training、inference backend、checkpoint/recovery / FSDP+vLLM 与 checkpoint OOM | 把“能运行 GRPO”和“能保存并恢复完整训练状态”分开验收；release 的临时关闭 checkpoint 建议不是生产恢复方案 |

## 本季读法与承接

如果当前在做 RL，先读 **DAPO + R1 第 2 节**，留下数据流、分组、mask 和 denominator 的检查表；如果当前在做 MoE/长上下文，先读 **DeepEP/DeepGEMM 固定快照 + NSA**，留下可计时的 kernel/通信路径。两条路径无需同时展开成 P0 队列。

接着读 [2025 Q2](quarterly_signal_2025-Q2.md)：rollout 长尾会把系统推向异步，但异步又引出 policy lag、权重同步与样本生命周期；kernel 与通信的局部实现也开始被提升为可组合工具和整层调度。

## 来源核验与覆盖边界

| 论文首版 | 核验署名 | 日期 |
|---|---|---|
| DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning | 正文署名 DeepSeek-AI；arXiv citation_author 完整名单保存于 JSON | 2025-01-22 |
| Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention | Jingyang Yuan、Huazuo Gao、Damai Dai 等 15 位作者 | 2025-02-16 |
| DAPO: An Open-Source LLM Reinforcement Learning System at Scale | Qiying Yu、Zheng Zhang、Ruofei Zhu 等 35 位作者 | 2025-03-18 |

- 已从 arXiv v1 页面读取 `citation_title`、`citation_author`、`citation_date`，并核对方法正文；不同版本不重复计为首次发表。原始元数据与官方 release 时间保留在[JSON](audits/2026-09-22-history/2025-h1-sources.json)。
- Q1 原有 backfill 覆盖偏重 R1/DAPO；本次补入稀疏 attention 与 MoE kernel/communication，但没有宣称穷尽该季训练论文、storage/networking、硬件故障与基础栈全部发布。
- 官方博客和 HF model card 是本次可访问的页面，可能在发布后编辑；涉及当季代码能力时，优先引用固定 commit / release，网页不能证明的细节不补写。DeepSeek 已同时检查 API changelog 与官方 HF 权重页。
- 性能与稳定性结论均有原文归属，本次未开展 GPU 实验。建议的实验、topic 更新与深读是下一步，不是已完成的学习成果。

[返回按月/按季阅读入口](monthly_reviews.md)

# 2025 Q2 训练基础设施复盘：异步 RL 与通信调度走向系统化

> 整合说明（2026-10-08）：本文保留 9/22 的来源核验与复评日期；[GitHub 历史定向补证](github_history_2025_to_2026_h1.md)已保存，但全窗口事件索引重建仍因 API 连接失败待补，不能据此声称历史 GitHub 全覆盖。

> 原始材料窗口：2025-04-01 至 2025-06-30，按首次公开版本归季。回看与补录时间：2026-09-22。本文是 Historical Retrospective，不改 frontier scan 游标或原判定。
>
> 本季精选 **5 篇论文，合为四条主线**：AReaL 与 LlamaRL、MegaScale-MoE、TileLang、ROLL。AReaL 承接既有 2025-05 backfill，其余补齐异步系统、MoE 与 kernel 工具的比较视角。`Lifecycle: NEW` 只描述本次阅读候选，不覆盖已有笔记或替用户标记已读。[原始来源核验记录](audits/2026-09-22-history/2025-h1-sources.json)保留首版元数据。

## 先读这页

**趋势推断：Q2 的共同问题是如何让各个阶段持续交付有效工作。** RL 把 rollout 和 trainer 拆开后，要控制旧样本与新权重之间的距离；MoE 把更多时间花在传输上后，要重排整层通信与计算；新的 kernel 编程模型则尝试让这种硬件配合不再只能靠手写大量底层代码。ROLL 进一步把单条样本和环境交互纳入调度对象。

因此，本季不能用“异步更快”“FP8 更快”“稀疏更快”三个口号概括。**可迁移的设计应能回答：等待消失在什么位置，新增的状态由谁管理，性能收益是否改变训练目标或故障恢复语义？**

| 主线 | 材料 | 瓶颈怎样移动 | 今天最应追问 |
|---|---|---|---|
| 异步 RL | AReaL + LlamaRL | batch barrier → policy lag、队列与权重交付 | 样本由哪个 policy 产生，何时允许被消费？ |
| MoE 训练 | MegaScale-MoE | 局部 GEMM → 整层通信关键路径 | attention 与 FFN 是否需要相同并行布局？ |
| Kernel 工具 | TileLang | 手工实现调度 → 可组合的 dataflow 与 schedule | 编译器替你做了什么，哪些约束仍需显式控制？ |
| Agentic RL 框架 | ROLL | 整批 rollout → 每条样本和环境生命周期 | 超时、取消、部分完成和奖励如何返回 trainer？ |

## 1. AReaL + LlamaRL：异步不是开两个进程，而是定义两条流之间的关系

**原文机制。** [AReaL 首版](https://arxiv.org/html/2505.24298v1)让 rollout worker 持续生成，trainer 收够数据便更新，用 staleness-aware 控制限制生产与消费的版本差距，并通过 decoupled PPO 区分产生轨迹的 behavior policy 和约束更新的 proximal policy。[LlamaRL 首版](https://arxiv.org/html/2505.24034v1)则将 trainer、generator、reward 等组件置于 executor 与 communication channel 抽象下，使用不同资源布局，并通过 DDMA 在 GPU 间传输权重，降低大模型同步开销。

两者解决的等待相似，关注重点不同：AReaL 把异步样本的统计语义和 staleness 控制放在核心位置；LlamaRL 给出大模型、多组件放置与权重路径的工业设计。它们不是可直接横比的榜单项。AReaL 作者报告相同 GPU 数、结果匹配或更好时，最高 2.57× 的训练加速；LlamaRL 的最高 10.7× 来自 405B policy 与 DeepSpeed-Chat-like 基线的特定比较，包含资源布局、异步和其他优化，不能解释成“只开异步就提升 10.7×”。

**系统后果——本仓库推断。** 拆掉同步 barrier 后，必须显式管理权重发布、样本 policy version、队列长度和 backpressure。生成端很忙并不保证 trainer 收到的是可用数据；限制队列也不自动保证统计偏差已经消失。评测应同时报告达到同等质量的总 GPU-hours、被丢弃的样本和 lag 分布。长轨迹中途更新权重，还需要区分整条轨迹版本与逐 token 的行为证据，不能只给 batch 打一个最新版本号。

**今天怎么读。** 用 AReaL 第 4–5 节画生产/消费与 policy-version 时间线，再用 LlamaRL 第 5 节检查权重同步是否经过 CPU、是否集中聚合、并行分片如何对齐。论文证明了某些配置可行，不等于当前框架所有 backend、Agent 环境和 checkpoint 都具备同样保证。[既有 AReaL backfill](backfill/2025-05.md)继续保留原阅读状态。

| 字段 | AReaL | LlamaRL |
|---|---|---|
| Source ID / 类型 | `arxiv:2505.24298v1` / asynchronous RL system | `arxiv:2505.24034v1` / industrial RL system |
| 原始时间 / 本次核验 | 2025-05-30 / 2026-09-22 | 2025-05-29 / 2026-09-22 |
| Impact / Decision | 高 / **Deep Dive** | 高 / **Read** |
| Reason / 一句话价值 | 把异步效率与 staleness 控制放进同一个训练目标 | 暴露大模型 RL 的分布式组件放置和权重同步路径 |
| Related topics | [Agentic RL](../topics/agentic_rl.md)、[Rollout Latency](../playbooks/rollout_latency.md) | [Distributed Training](../topics/distributed_training.md)、[Checkpointing](../topics/checkpointing.md) |
| Next / 目标去向 | 追踪轨迹 admission、policy version 和 trainer consume；topic / experiment 候选 | 画 trainer shard 到 generator shard 的映射与传输时间线；paper / insight 候选 |
| Lifecycle | **NEW**：本次候选，既有状态不改 | **NEW** |

## 2. MegaScale-MoE：通信优化的单位是整个 MoE layer

**原文机制。** [MegaScale-MoE 首版](https://arxiv.org/html/2505.11432v1)分别选择 attention 与 FFN 的并行策略，配合跨算子的通信/计算重排、算子内 tile 级 overlap，以及 selective activation rematerialization。低精度通信也需要改变量化和 reduction 路径；原文在 FP8 通信方案中保留 FP32 reduction，不能简化为“把 collective dtype 改成 FP8”。

**系统后果——本仓库推断。** 单看 NCCL 带宽，无法判断某段通信是否拖慢训练：有的传输能藏在独立计算中，有的在依赖链上必须等待。另一方面，重计算和重通信可能降低显存却增加网络工作；只有把 forward、backward、buffer 释放和 layout 转换放在同一时间线上，才知道优化是否有效。Q1 的 DeepEP/DeepGEMM 提供局部工具，本篇提供如何把局部能力组织成整层执行的视角。

**证据与阅读取舍。** 作者报告 352B MoE、1,440 张 NVIDIA Hopper GPU 下吞吐 1.41M tokens/s，相对其 Megatron-LM 基线效率提升 1.88×。这是作者生产系统实验，不能泛化为任意 MoE 模型或当前 Megatron 的差距；本次也没有独立复现。今天优先读并行选择与 overlap 的依赖分析，性能表用于检查 workload、网络拓扑与基线是否可比。

| 字段 | 本次判定 |
|---|---|
| Source ID / 类型 | `arxiv:2505.11432v1` / production training system paper |
| 原始时间 / 本次核验 | 2025-05-16 / 2026-09-22 |
| Impact / Decision | 高 / **Deep Dive** |
| Reason / 一句话价值 | 联合决定并行布局、显存与通信调度，避免把每个 kernel 的最优当成作业最优 |
| Related topics | [MoE](../topics/moe.md)、[FP8](../topics/fp8.md)、[Sequence Parallelism](../topics/sequence_parallelism.md)、[NCCL](../topics/nccl.md) |
| Next / 目标去向 | 在固定 MoE workload 中分别计量可覆盖与暴露通信；paper / topic / experiment 候选 |
| Lifecycle | **NEW** |

## 3. TileLang：把 dataflow 和硬件调度分开表达，降低 kernel 试错成本

**原文机制。** [TileLang 首版](https://arxiv.org/html/2504.17577v1)以 tile 的移动和计算表达 dataflow，将 layout、thread binding、tensorization、pipeline 等调度选择交给可定制的 primitive、annotation 与编译过程。它不是去掉硬件约束，而是让常见调度可以自动推导，并在必要处允许开发者介入。

**系统后果——本仓库推断。** 研究者能更快表达新的 attention、GEMM 或融合算子，但编译成功并不代表 shape、dtype 和梯度路径都正确。kernel 的研究周期应包含正确性、边界 shape、编译/缓存成本和稳定计时；只有热点 shape 快，不足以证明完整训练 step 快。不同 GPU 上也不能仅凭同一份高层代码就宣称性能可移植。

**今天怎么读。** 将它与 Q1 的手写 DeepGEMM 路径比较，识别哪些 layout/pipeline 决策可交给工具，哪些还必须由 profiling 驱动。论文首发在 4 月，但 [v0.1.0 已于 2025-02-12 发布](https://github.com/tile-ai/tilelang/releases/tag/v0.1.0)；**这里收录的是 Q2 的论文与编程模型证据，不将 Q2 写成项目首次开源时间。** 当前 backend/API 能力不反推到首版论文。

| 字段 | 本次判定 |
|---|---|
| Source ID / 类型 | `arxiv:2504.17577v1` / kernel programming model paper |
| 原始时间 / 本次核验 | 论文 2025-04-24 / 2026-09-22 |
| Impact / Decision | 中高 / **Read** |
| Reason / 一句话价值 | 改变高性能 kernel 的实现和迭代方式，而不是提供某个 shape 的一次性性能冠军 |
| Related topics | [FlashAttention](../topics/flashattention.md)、[FP8](../topics/fp8.md)、[Transformer Engine](../topics/transformer_engine.md) |
| Next / 目标去向 | 为一个现有热点写最小 tile dataflow，并比较正确性、编译成本与全 step 收益；paper / experiment 候选 |
| Lifecycle | **NEW** |

## 4. ROLL：样本生命周期与环境交互成为框架接口

**原文机制。** [ROLL 首版](https://arxiv.org/html/2506.06122v1)采用 single-controller 与 Parallel Worker 抽象，把 parallel strategy、data transfer、rollout scheduler、environment/reward worker 和 AutoDeviceMapping 拆成明确模块。rollout scheduler 管理单条样本的生成生命周期，而不只处理整批完成；报告也包含多轮 Agent 任务的实验。

**系统后果——本仓库推断。** 环境交互使 rollout 的完成条件更复杂：模型返回、工具结束、reward 计算和训练可消费不是同一件事。框架需要为每条样本保留终止原因、环境状态和数据归属。否则取消、超时或重试可能静默改变组结构。这里的价值不是简单增加一个 environment callback，而是让调度对象与真正的训练样本对应。

**证据与取舍。** ROLL 作者在首版报告内部超过 200B 总参数 MoE、千卡级训练连续约两周的经验；这是厂商披露，不是所有故障模式下恢复正确性的证明。今天优先读第 4 节模块和 workflow，把可迁移的 sample lifecycle、worker 边界与资源映射对照 AReaL；不依据“用户友好”或 benchmark 单项高低决定迁移框架。

| 字段 | 本次判定 |
|---|---|
| Source ID / 类型 | `arxiv:2506.06122v1` / RL framework technical report |
| 原始时间 / 本次核验 | 2025-06-06 / 2026-09-22 |
| Impact / Decision | 中高 / **Read** |
| Reason / 一句话价值 | 将样本、环境与多模型放置放进明确的调度和数据接口 |
| Related topics | [RL Framework Selection](../topics/rl_framework_selection.md)、[Agentic RL](../topics/agentic_rl.md)、[Fault Tolerance](../topics/fault_tolerance.md) |
| Next / 目标去向 | 对比完成、取消、超时、奖励返回的状态转移与 AReaL 数据入口；paper / topic / experiment 候选 |
| Lifecycle | **NEW** |

## OpenAI / Anthropic / NVIDIA / DeepSeek Watch

| 厂商与当季一手材料 | Triage | 工程取舍 |
|---|---|---|
| OpenAI：[o3/o4-mini，2025-04-16](https://openai.com/index/introducing-o3-and-o4-mini/) | **Observed** | 将 reasoning 与工具使用结合，是 rollout 工作负载的方向证据；该发布页不足以证明其内部 scheduler、权重同步或 GPU 规模，不作为本季新增 infra 论文 |
| Anthropic：[Claude 4，2025-05-22](https://www.anthropic.com/news/claude-4) | **Observed** | 长时编码与 Agent 能力值得跟踪；公开能力结果与 training infrastructure disclosure 分开看 |
| NVIDIA：[long-context training，2025-06-02](https://developer.nvidia.com/blog/scaling-to-millions-of-tokens-with-efficient-long-context-llm-training/)、[NeMo-Skills，2025-06-25](https://developer.nvidia.com/blog/how-to-streamline-complex-llm-workflows-using-nvidia-nemo-skills/) | **Observed** | CP、activation recomputation/offload 与数据生成—训练—评估流程可作实现补充。前一篇把 FlashAttention 写成 O(n) 计算复杂度的表述不沿用：exact dense attention 的计算仍为二次量级，主要改进是 IO 与中间存储 |
| DeepSeek：[API changelog 的 2025-05-28](https://api-docs.deepseek.com/updates/)、[官方 HF DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) | **Observed** | 权重和 API 升级已核验；model card 描述 reasoning 改进，不据此补造异步 RL 架构或新训练成本。原 R1 报告属于 Q1，本季不重复计首发 |

## Hugging Face Watch

| 当季来源 | Triage | 具体信号与取舍 |
|---|---|---|
| [TRL v0.18.0，2025-05-28](https://github.com/huggingface/trl/releases/tag/v0.18.0) | **Observed** | 官方 release 包含 vLLM co-location、FSDP2、两侧 clipping、generation minibatch 与 gradient accumulation 解耦；同时列出 colocate 模式 prompt/completion 对齐修复。它支撑“吞吐配置与样本语义必须一起验收”的判断，不另加论文名额 |
| [Accelerate v1.8.0，2025-06-19](https://github.com/huggingface/accelerate/releases/tag/v1.8.0) | **Observed** | FSDP2 preparation 重构与 FP8 支持强调组合顺序；FP8、compile、activation checkpointing 的可组合性不是各自开关都能打开就成立 |
| [HF 官方团队 Kernel Hub 博客，2025-06-12](https://huggingface.co/blog/hello-hf-kernels) | **Observed** | 预编译 kernel 的版本匹配与分发降低部署摩擦，与 TileLang 的编写问题互补；本文未将 2026 新仓库类型、签名机制倒填到 2025 |
| [Transformers v4.52.1，2025-05-20](https://github.com/huggingface/transformers/releases/tag/v4.52.1)；[PEFT v0.15.2，2025-04-15](https://github.com/huggingface/peft/releases/tag/v0.15.2) | **Observed** | 前者以模型适配等变更为主，后者修复 prompt learning 方法；本次未提升为独立主线。另核对 PEFT v0.16.0 实际于 2025-07-03 发布，应归 Q3，不能用当前 docs 补写 Q2 能力 |

上述 HF 博客为官方团队文章，release 为官方仓库记录；社区 kernel 文章不自动提升为主线。该 Watch 是有边界的抽样，不等于缺席的库在当季没有进展。

## RL Framework Watch

以下为 **2026-09-22 Historical Audit**；实现级补漏与覆盖统计统一见[GitHub 历史审计](github_history_2025_to_2026_h1.md)。论文所描述的设计、release 已承诺能力与代码修复分开对待。

| 框架 | 证据与 Triage | 子系统 / 工程维度 | 对 AReaL 的参考 |
|---|---|---|---|
| AReaL | **Accepted**：2025-05-30 首版论文，主线 1 | rollout、training、scheduler / staleness 与有效吞吐 | 自身历史基线；后续 feature 应检查是否保留版本额度和行为策略证据 |
| verl | **Observed**：Q1 DAPO recipe 在本季继续作为比较背景 | training、data/trajectory path / group、mask、normalization | 延续有效组与 token 权重验收；没有把当前 verl 异步实现直接算作 Q2 成果 |
| slime | **Observed**：联合 GitHub 审计确认 [#2 于 2025-06-30 合并](https://github.com/THUDM/slime/pull/2)，主题为 partial rollout | rollout / 长尾与生成控制 | 作为进一步读代码的入口；本文不以 PR 标题推断中断、恢复、版本控制的具体保证 |
| ROLL | **Accepted**：2025-06-06 首版报告，主线 4 | scheduler、data/trajectory path、rollout / sample lifecycle、environment worker | 迁移接口与状态机设计，性能结论需要各自 workload 验证 |
| OpenRLHF | **Observed**：[2025-04-23 官方团队工程博客](https://vllm.ai/blog/2025-04-23-openrlhf-vllm)；联合审计确认 [#1015，2025-05-18](https://github.com/OpenRLHF/OpenRLHF/pull/1015) 的 Async/Agent RLHF 变更 | inference backend、weight sync、scheduler / Ray placement、CUDA IPC/NCCL、异步接口 | 对照同置/分置条件下资源所有权与权重发布；博客示例不是生产恢复证明 |
| NeMo RL | **Observed**：Q1 [v0.1.0](https://github.com/NVIDIA-NeMo/RL/releases/tag/v0.1.0) 与 Q2 [NeMo-Skills 集成文章](https://developer.nvidia.com/blog/how-to-streamline-complex-llm-workflows-using-nvidia-nemo-skills/) | training、inference backend / 数据、训练、评估连接 | 关注 backend 接口和 checkpoint 转换；未将 8 月 v0.3 的 Megatron backend、异步能力算作 Q2 |

OpenRLHF 博客在 Q2 有实际 Ray placement 与 CUDA IPC/NCCL 权重交付示例，值得作为 LlamaRL 权重路径的轻量对照；它已在[2025-04 backfill](backfill/2025-04.md)登记，因此保留承接关系，不假装首次发现。

## 本季只选两条深读路径

1. **做 RL 平台：AReaL → LlamaRL → 按需 ROLL。** 先解释 policy-version 与消费规则，再看大权重传输和组件放置，最后看单条样本的环境生命周期。完成标准是能画出一条轨迹从生成到更新、失败与恢复的路径。
2. **做预训练优化：MegaScale-MoE → TileLang。** 先找到可见通信，再决定是否需要自定义 kernel；不要在整层调度未明确时提前重写 GEMM。完成标准是能说明哪段 critical path 变短，以及量化、显存或代码维护增加了什么代价。

这些都是建议动作，本次没有扩充 P0 或将材料直接标为 SUMMARIZED/VERIFIED。[2025 Q1](quarterly_signal_2025-Q1.md)解释需求与局部机制从何而来；[2025 Q3](quarterly_signal_2025-Q3.md)继续追踪规模化实现和更长轨迹带来的约束。

## 来源核验与覆盖边界

| 论文首版 | 核验署名 | 日期 |
|---|---|---|
| TileLang: A Composable Tiled Programming Model for AI Systems | Lei Wang、Yu Cheng、Yining Shi 等 11 位作者 | 2025-04-24 |
| MegaScale-MoE: Large-Scale Communication-Efficient Training of Mixture-of-Experts Models in Production | Chao Jin、Ziheng Jiang、Zhihao Bai 等；**v1 为 18 位作者** | 2025-05-16 |
| LlamaRL: A Distributed Asynchronous Reinforcement Learning Framework for Efficient Large-scale LLM Training | Bo Wu、Sid Wang、Yunhao Tang 等 14 位作者；正文团队署名 Meta GenAI | 2025-05-29 |
| AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning | Wei Fu、Jiaxuan Gao、Xujie Shen 等 13 位作者 | 2025-05-30 |
| Reinforcement Learning Optimization for Large-Scale Learning: An Efficient and User-Friendly Scaling Library | Weixun Wang、Shaopan Xiong、Gengru Chen 等 41 位作者；正文署名 ROLL Team | 2025-06-06 |

- arXiv v1 的 `citation_title`、`citation_author`、`citation_date` 与正文方法已核对。[JSON](audits/2026-09-22-history/2025-h1-sources.json)原样保留 LlamaRL 的 citation_title 尾部 `Trainin` 截断；正文标题为完整 `Training`，不是两个不同论文。MegaScale-MoE 不混用后续修订版新增的作者。
- Q2 回看承接 AReaL 与 OpenRLHF 既有 backfill；本季另选的四篇补的是系统比较与工具链缺口。没有全量覆盖季度论文、checkpoint/storage、硬件可靠性或所有厂商报告，也没有把观察到的模型发布一律升级为训练系统材料。
- HF 组件按选定 release 抽样：Transformers v4.52.0 API 未取到后，改核可访问的 v4.52.1，未宣称覆盖所有中间版本；部分框架历史能力仍只得到有限证据。当前 docs/main 不能证明历史可用性；官网文章与 model card 可能有后续修改，代码能力优先以当季论文、release 或固定 commit 为准。
- 本次是阅读决策与历史核验，不是 GPU 复现。所有速度、规模与连续运行数字均来自作者/厂商陈述，不能作为当前生产容量或稳定性的保证。

[返回按月/按季阅读入口](monthly_reviews.md)

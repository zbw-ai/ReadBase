# Monthly Signal Report, 2026-04

> 整合说明（2026-10-08）：本文保留 9/22 的来源核验与复评日期；[GitHub 历史定向补证](github_history_2025_to_2026_h1.md)已保存，但全窗口事件索引重建仍因 API 连接失败待补，不能据此声称历史 GitHub 全覆盖。

## 2026-09-22 历史复盘：调度、数值、恢复与环境开始成为同一个问题

> 本节为 **2026-09-22 Historical Review**，把此前简短回看导读展开为阅读判断；下方原月报、原 Accepted / Decision 与 **2026-07-23 Historical Audit** 原文保留。这里的补选发生在 9 月，不冒充当月发现，不改 frontier cursor，也不表示已经完成阅读或实验。GitHub 覆盖已由前一轮的 7–9 月，扩展到 [2025—2026 H1 历史索引](github_history_2025_to_2026_h1.md)；代码枚举的完整性与论文、博客的定向核验分开计量。元数据、版本和 Decision 变化见 [H1 来源审计](audits/2026-09-22-history/2026-h1-sources.json)。

### 三条发展主线

1. **异步 RL 要守住样本的策略语义。** [DORA v1](https://arxiv.org/abs/2604.26256v1) 显式讨论 trajectory 内 policy 一致性、data integrity 与 bounded staleness，并采用多版本 streaming rollout；[NVIDIA FP8 RL](https://developer.nvidia.com/blog/run-high-throughput-reinforcement-learning-training-with-end-to-end-fp8-precision/) 则处理生成与训练 engine 的数值偏差。版本相同仍可能有 logprob mismatch，这是两份材料连读的原因。
2. **长上下文会同时改写显存布局与通信顺序。** 原 AutoSP/CommFuse 之外，重评 [TSP](https://arxiv.org/abs/2604.26294v1)：同一 device axis 同时放 weight shard 与 sequence shard，通过参数轮转/广播和 KV 交换换取显存空间。多维并行的名称不是最优布局证明，必须比较额外通信与减少的 activation。
3. **工业报告把底层优化连接到可恢复 rollout 与 sandbox。** 补入 [DeepSeek-V4 报告](https://arxiv.org/abs/2606.19348v1) 的 §3 与 §5.2：batch-invariant kernel、token WAL、metadata/per-token 数据分离以及 DSec 环境平台进入同一系统。核心启发是恢复未完成轨迹不能随便从头重采，否则可能改变长度分布；这是报告直接讨论的 correctness 问题。

### 本月两份可选深读

| 选择 | 核验信息 | 阅读时必须回答的问题 | 当前 Decision |
|---|---|---|---|
| 异步 correctness：DORA v1 | Tianhao Hu 等；2026-04-29；按 v1 作者表核验，未混用 7 月 v2 | 多版本并存时，谁负责接纳、完成、消费与清退 trajectory？ | Read → Deep Dive；高影响；与 AReaL 的 stale-budget 对照 |
| 工业系统：DeepSeek-V4 | DeepSeek-AI 等；4 月 24 日官方公开发布，arXiv v1 标注 4 月 26 日 | token WAL、KV 恢复、确定性 kernel 与 sandbox 供给如何共同保证长轨迹可继续？ | 未收录 → Deep Dive；高影响；优先读 §3、§5.2 |

### 补选、日期核验与未升级材料

- **新增精选：DeepSeek-V4，Deep Dive，★★★★★，状态 NEW；Source ID `arxiv:2606.19348v1`。** [官方发布页](https://deepseek.com/en/news/v4-preview/)与[官方权重卡](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)交叉核验；下一步把 §5.2.3 的续采、§5.2.4 的样本数据路径、§5.2.5 的环境供给对照 AReaL，目标为 [Agentic RL](../../rl-infra/topics/agentic_rl.md)、[Fault Tolerance](../../training-infra/topics/fault_tolerance.md)。公开机制可读，厂商规模与性能仍是自报证据，不表示仓库复现。
- **DeepSeek-V4 的 `2606` ID 前缀与日期不一致，已显式保留。** raw `citation_date` / `citation_online_date` 均为 `2026/04/26`，v1 submission history 为 `2026-04-26 14:49:33 UTC`，[PDF 扉页](https://arxiv.org/pdf/2606.19348v1)同日；4 月 24 日官方发布页目前直接链接同一报告，[HF 官方团队文章](https://huggingface.co/blog/deepseekv4)也记录当天发布及相同架构。故按最早可核验官方公开事件归 4 月；不推测编号差异的原因，也不把当前 HF 文件当作 4 月字节级快照。
- **新增精选：TSP，Observe → Read，★★★★☆，状态 NEW；Source ID `arxiv:2604.26294v1`。** Vasu Shyam、Anna Golubeva、Quentin Anthony；2026-04-29。下一步列出 attention/MLP 的数据移动顺序与 memory/communication trade-off，目标为 [Sequence Parallelism](../../training-infra/topics/sequence_parallelism.md)。升级原因是补充“同轴布局换显存”的独立判断，而非单纯再收一种并行简称。
- NVIDIA FP8 RL 原 Read 保留：Guyue Huang 等，2026-04-20；block-wise FP8 对齐、importance sampling 与 QKV scale 同步需要一起看。本文未把厂商的未来 kernel 优化预期当作实测结果。
- ZipCCL / TACO 仍 **Observe**：压缩字节不自动解决 critical-path 上的等待；先与 CommFuse 的 overlap 路径建立同配置对照。CacheFlow 仍 Observe：需先验证 KV 恢复语义与 rollout policy version 的对应关系。

### OpenAI / Anthropic / NVIDIA / DeepSeek Watch · 历史复盘

| 厂商 | 本次判断 | 覆盖与理由 |
|---|---|---|
| OpenAI | Observed（沿用原记录） | compute infrastructure 条目原有正文抓取限制仍保留，不把标题当机制证据 |
| Anthropic | Observed | [Managed Agents](https://www.anthropic.com/engineering/managed-agents) 是服务解耦相邻证据；尚未核验训练状态/权重协议，不升为本月精选 |
| NVIDIA | Accepted / Read | 4 月 20 日 FP8 RL 正文、署名和日期已核验；其余旧 NVIDIA Accepted 仍按原审计范围解读 |
| DeepSeek | Accepted / Deep Dive | 已查 [API changelog](https://api-docs.deepseek.com/updates/) 与 [官方 HF organization](https://huggingface.co/deepseek-ai)，补 V4 report 与权重公开事件；模型榜单不作为入选理由 |

### Hugging Face Watch · 历史复盘

[DeepSeek-V4 官方团队文章](https://huggingface.co/blog/deepseekv4)（Ben Burtenshaw，2026-04-24）作为 **Observed / 交叉来源**，不重复计一个精选；底层机制仍以 DeepSeek primary report 为准。已打开 [HF Blog](https://huggingface.co/blog) 和 [Transformers](https://github.com/huggingface/transformers/releases)、[Accelerate](https://github.com/huggingface/accelerate/releases)、[PEFT](https://github.com/huggingface/peft/releases)、[Kernels](https://github.com/huggingface/kernels/releases) 官方 release 入口；当月逐页代码覆盖以统一 GitHub 索引为准。当前页面不是历史快照，未单独确认的旧版本不据此生成新信号，社区文章也不借用官方团队身份。

**覆盖边界：** 定向补工业报告并重评 TSP；原 top-80 论文截断仍在。下方 2026-07-23 RL Framework Watch 审计保持原状；统一 GitHub 索引负责补证日期和重大 PR，本节不把 release 能力写成个人实现结果。

### RL Framework Watch · 2026-09-22 代码补证

[verl #6091](https://github.com/verl-project/verl/pull/6091) 对超大 tensor 分块以降低同步 buffer 峰值；[Megatron #4047](https://github.com/NVIDIA/Megatron-LM/pull/4047) 要求 async P2P send 完成后才释放 activation。它们分别约束 weight sync 与 pipeline training 的状态生命周期。[Transformer Engine v2.14.1](https://github.com/NVIDIA/TransformerEngine/releases/tag/v2.14.1) 修复 MXFP8 quantization + dbias fusion 的非确定性错误结果，进一步说明 FP8 验收不能只有吞吐。 这些是本轮 [GitHub 历史审计](github_history_2025_to_2026_h1.md) 核实的代码/版本证据；不改变下方 2026-07-23 Historical Audit 的原计数，不代表本仓库已运行回归实验。

---

- Window: 2026-04-01 00:00:00 ~ 2026-04-30 23:59:59
- Timezone: Asia/Shanghai
- Generated at: 2026-07-08
- Report type: monthly quality digest
- Sources scanned: arXiv cs.DC / cs.LG / cs.AI / cs.CL submittedDate window; NVIDIA / OpenAI / Microsoft Research / PyTorch official RSS; primary arXiv abstract pages and selected official blog pages.
- Scan completeness: arXiv API 覆盖 2026-04 全月四个重点分类各前 80 条按提交时间排序结果；NVIDIA / OpenAI / Microsoft Research / PyTorch RSS 可解析；NVIDIA 关键技术博客正文可抓取；OpenAI compute infrastructure 由 RSS 发现但正文抓取被站点挑战页阻断，因此未作为 accepted signal。

## 本月核心判断

2026 年 4 月的核心信号是：**RL post-training、long-context training 和 NVIDIA Training Stack 正在同时工程化**。这不是“又多了几篇优化论文”，而是几个原来分散的问题开始汇合：rollout 变长、生成尾延迟变大、低精度进入 RL 闭环、Sequence/Context Parallel 变成自动化编译问题，通信 overlap 也开始围绕 tail latency 做细粒度重排。

第一，**RL infra 的瓶颈从 trainer step 扩展到 rollout schedule 和数值一致性**。DORA 关注异步 rollout 的 bounded staleness，NVIDIA FP8 RL 关注 vLLM rollout 与 Megatron training 的低精度一致性。一个讲调度，一个讲数值/精度，但都在说明 RL 训练已经不是单纯 `generate -> train` 的脚本问题。

第二，**长上下文训练正在从手写并行策略变成系统能力**。AutoSP 直接把 automated sequence parallelism、long-context aware activation checkpointing 放进 compiler abstraction；NVIDIA BioNeMo 的 Context Parallelism 案例则说明 CP 不只是 LLM 文本模型技巧，而是面向超长结构输入的通用系统机制。

第三，**通信优化继续从“减少字节”走向“隐藏尾延迟”**。CommFuse、ZipCCL、TACO 都盯着分布式 LLM 的通信开销，但 CommFuse 更贴近生产判断：通信瓶颈不只是平均带宽，而是 overlap schedule 里的 tail latency。

## Accepted Signals

### Run High-Throughput Reinforcement Learning Training with End-to-End FP8 Precision

- Signal ID：2026-04-001
- Source ID：blog:nvidia/fp8-rl
- First seen：2026-07-08
- 来源窗口：official blog
- 类型：engineering blog
- 链接：https://developer.nvidia.com/blog/run-high-throughput-reinforcement-learning-training-with-end-to-end-fp8-precision/
- 影响等级：★★★★★
- Decision：Read
- Reason：它把 GRPO、rollout generation、Megatron training、NeMo RL、FP8 linear layers、FP8 KV cache/attention、importance sampling 和 vLLM/Megatron 数值对齐放到同一个 RL training loop 里讨论。
- 建议动作：进入 [P1](../reading_queue/P1.md)
- 关联主题：[Agentic RL](../../rl-infra/topics/agentic_rl.md), [FP8](../../systems/topics/fp8.md), [Transformer Engine](../../systems/topics/transformer_engine.md), [Rollout Latency](../../practice/playbooks/rollout_latency.md)
- 最终应流向：engineering blog / topic / experiment

这条是 4 月最贴你当前方向的工程博客。它的价值不在“FP8 能加速”，而在指出 RL 低精度训练有独特难点：rollout engine 和 trainer engine 不同，policy 每步更新，KV cache 和 attention 也会进入低精度路径，数值误差会影响 importance sampling 和训练稳定性。

### DORA: A Scalable Asynchronous Reinforcement Learning System for Language Model Training

- Signal ID：2026-04-002
- Source ID：arxiv:2604.26256
- First seen：2026-07-08
- 来源窗口：arXiv
- 类型：paper
- 链接：https://arxiv.org/abs/2604.26256
- 影响等级：★★★★★
- Decision：Read
- Reason：它直接指出 rollout phase 可占总 step time 的 50--80%，并把 long-tailed trajectories、MoE imbalance、intra-trajectory policy consistency、data integrity、bounded staleness 作为异步 RL 系统的核心约束。
- 建议动作：进入 [P1](../reading_queue/P1.md)
- 关联主题：[Agentic RL](../../rl-infra/topics/agentic_rl.md), [Rollout Latency](../../practice/playbooks/rollout_latency.md), [MoE](../../training-infra/topics/moe.md)
- 最终应流向：paper note / topic / playbook

DORA 和 AReaL / HybridFlow 应该放在一起读。它把“异步 rollout 提升吞吐”后面的代价讲得更工程化：只追求 overlap 会破坏策略一致性和样本新鲜度，必须显式限制 staleness，并处理长尾轨迹拖慢全局进度的问题。

### AutoSP: Unlocking Long-Context LLM Training Via Compiler-Based Sequence Parallelism

- Signal ID：2026-04-003
- Source ID：arxiv:2604.27089
- First seen：2026-07-08
- 来源窗口：arXiv
- 类型：paper
- 链接：https://arxiv.org/abs/2604.27089
- 影响等级：★★★★☆
- Decision：Read
- Reason：它把 long-context training 的 Sequence Parallelism 和 activation checkpointing 自动化，指出现有训练库更擅长大参数模型，而不是让用户容易组合长上下文优化。
- 建议动作：进入 [P1](../reading_queue/P1.md)
- 关联主题：[Long-context Training](../../training-infra/topics/long_context_training.md), [Sequence Parallelism](../../training-infra/topics/sequence_parallelism.md), [Context Parallelism](../../training-infra/topics/context_parallelism.md)
- 最终应流向：topic / experiment

这条适合和你正在看的 128k SFT 配置联系起来。长上下文训练的难点不是只把 `max_length` 调大，而是要让 sequence sharding、activation checkpoint、attention kernel、batch packing 和并行布局一起成立。AutoSP 的信号是：这些组合未来会越来越需要 compiler/runtime 帮忙。

### Advancing Emerging Optimizers for Accelerated LLM Training with NVIDIA Megatron

- Signal ID：2026-04-004
- Source ID：blog:nvidia/megatron-emerging-optimizers
- First seen：2026-07-08
- 来源窗口：official blog
- 类型：engineering blog
- 链接：https://developer.nvidia.com/blog/advancing-emerging-optimizers-for-accelerated-llm-training-with-nvidia-megatron/
- 影响等级：★★★★☆
- Decision：Read
- Reason：它把 Muon、MOP、REKLS 等 emerging optimizers 接入 Megatron Core / NeMo Megatron Bridge，并讨论 layer-wise distributed optimization、distributed Newton-Schulz、data/tensor parallelism 和 GB300 NVL72 上的训练吞吐。
- 建议动作：进入 [P1](../reading_queue/P1.md)
- 关联主题：[Distributed Training](../../training-infra/topics/distributed_training.md), [FSDP](../../training-infra/topics/fsdp.md), [ZeRO](../../training-infra/topics/zero.md), [Transformer Engine](../../systems/topics/transformer_engine.md)
- 最终应流向：engineering blog / topic / insight

这条和 7 月的 MatrixFSDP 可以形成一条线：新 optimizer 不只是算法 recipe，它会碰到 sharding、all-reduce、Newton-Schulz iteration、通信隐藏和 optimizer state 生命周期。训练 infra 工程师需要关心的是“这个 optimizer 如何在 3D parallel / ZeRO/FSDP / Megatron Core 下落地”。

### CommFuse: Hiding Tail Latency via Communication Decomposition and Fusion for Distributed LLM Training

- Signal ID：2026-04-005
- Source ID：arxiv:2604.24013
- First seen：2026-07-08
- 来源窗口：arXiv
- 类型：paper
- 链接：https://arxiv.org/abs/2604.24013
- 影响等级：★★★★☆
- Decision：Read
- Reason：它针对 TP/DP 中 reduce-scatter / all-gather overlap 的 tail latency，把 collective 拆成 P2P communication 并重新调度，关注的是通信隐藏失败时的尾部拖慢。
- 建议动作：进入 [P1](../reading_queue/P1.md)
- 关联主题：[Tensor Parallelism](../../training-infra/topics/tensor_parallelism.md), [Distributed Training](../../training-infra/topics/distributed_training.md), [NCCL](../../systems/topics/nccl.md)
- 最终应流向：paper note / topic / playbook

这条比单纯“压缩通信量”的论文更适合作为生产排障入口。真实训练里 step time 抖动经常不是平均通信时间，而是某些 collective 或 overlap schedule 的尾部没有藏住。CommFuse 可以作为后续分析 TP/DP overlap 的材料。

### Scaling Biomolecular Modeling Using Context Parallelism in NVIDIA BioNeMo

- Signal ID：2026-04-006
- Source ID：blog:nvidia/bionemo-context-parallelism
- First seen：2026-07-08
- 来源窗口：official blog
- 类型：engineering blog
- 链接：https://developer.nvidia.com/blog/scaling-biomolecular-modeling-using-context-parallelism-in-nvidia-bionemo/
- 影响等级：★★★★☆
- Decision：Read
- Reason：它展示 Context Parallelism 如何让超长生物结构输入跨 GPU 保留全局上下文，并包含 halo-exchange local attention、长序列 tiling/repartition 等实现细节。
- 建议动作：进入 [P1](../reading_queue/P1.md)
- 关联主题：[Context Parallelism](../../training-infra/topics/context_parallelism.md), [Long-context Training](../../training-infra/topics/long_context_training.md), [FlashAttention](../../systems/topics/flashattention.md)
- 最终应流向：engineering blog / topic

虽然它不是 LLM 文本训练文章，但它对 CP 的工程价值很强：CP 的本质是“单个样本的长上下文跨 GPU 保留全局依赖”，不是只服务 chat context。这个案例能帮助你跳出“CP=长文本”的窄视角。

## P0 / P1 更新

### P0

不调整。当前 P0 仍保持：

- AReaL
- HybridFlow / verl
- Rollout Infrastructure Tax

原因：4 月材料里 DORA 和 NVIDIA FP8 RL 都很重要，但它们更适合在读完 AReaL / HybridFlow 后作为对照：一个看异步调度，一个看低精度和 rollout/training 数值一致性。

### P1

新增或确认进入 P1：

- NVIDIA FP8 RL：RL post-training 低精度闭环。
- DORA：异步 rollout 与 bounded staleness。
- AutoSP：long-context Sequence Parallelism 自动化。
- NVIDIA Megatron Emerging Optimizers：新 optimizer 在 Megatron/NeMo 上的分布式落地。
- CommFuse：TP/DP communication overlap 的 tail latency。
- NVIDIA BioNeMo Context Parallelism：CP 的真实长序列工程案例。

## Observed / Rejected

| 材料 | Decision | 原因 |
|---|---|---|
| ZipCCL: Efficient Lossless Data Compression of Communication Collectives | Observe | 通信压缩方向有价值，但本月通信主线优先读 CommFuse；ZipCCL 可在 NCCL/communication 专题扩写时回看 |
| TACO: FP8 Communication Compression for Tensor Parallel LLM Training | Observe | 和 TP intermediate tensor 压缩强相关，但当前先读 CommFuse 建立 overlap/tail latency 判断 |
| Folding Tensor and Sequence Parallelism | Observe | TSP 同时折叠 TP/SP 很有意思，但需要先补完 TP/SP/CP 基础专题 |
| CacheFlow: 3D-Parallel KV Cache Restoration | Observe | agentic long-context serving 强相关，但推理 infra 主线还未正式展开 |
| DUAL-BLADE KV Cache Offloading | Observe | KV offload/NVMe-direct 对边缘推理有价值，但优先级低于 rollout/training 系统 |
| Beyond SFT-to-RL / PRISM | Observe | post-training recipe 有价值，但偏算法/对齐流程，系统边界弱于 DORA 和 NVIDIA FP8 RL |
| OpenAI Building the Compute Infrastructure for the Intelligence Age | Observe | RSS 标题高度相关，但正文抓取被站点挑战页阻断，未做 accepted signal |
| NVIDIA CUDA Tile / CompileIQ / Sparse Tensor posts | Observe | kernel/toolchain 方向有价值，当前不挤占 RL/long-context/communication 主线 |
| Generic OpenAI product / customer / academy posts | Ignore | 不改变当前 Training Infra / RL Infra 工程判断 |

## OpenAI / Anthropic / NVIDIA Watch

| Vendor | Sources checked | Decision | 结果 |
|---|---|---|---|
| OpenAI | official RSS / blog entry points | Observe | `Building the Compute Infrastructure for the Intelligence Age` 标题高度相关，但正文抓取被站点挑战页阻断；未做 accepted。其他 4 月 OpenAI 条目多为产品、客户、agent SDK 或安全叙事，不改变当前 Training/RL Infra 判断。 |
| Anthropic | official news/research/engineering RSS endpoints | Not verifiable | 2026-07-08 回补扫描时，尝试的 Anthropic RSS endpoint 返回 HTML error page，未形成可解析 feed；本月不把 Anthropic 缺失视为无信号，后续需要用稳定官方索引或手工 primary page 补查。 |
| NVIDIA | NVIDIA Technical Blog RSS / primary pages | Accepted | 本月 NVIDIA 有 3 条进入 accepted：FP8 RL、Megatron emerging optimizers、BioNeMo Context Parallelism；CUDA Tile / CompileIQ / sparse tensor 等放入 Observe。 |

## RL Framework Monthly Highlights: Historical Audit

> 本节于 2026-07-23 按 2026-04 自然月复核官方 release。只保留三条会改变服务边界、长上下文 rollout 或采样语义的更新。

| Framework / change | Subsystem | Primary evidence | Decision | 工程判断与 AReaL 参考 |
|---|---|---|---|---|
| AReaL [v1.0.3](https://github.com/areal-project/AReaL/releases/tag/v1.0.3) | agent service / rollout gateway / weight sync | official release；Agent Service、controller/router/data proxy、Megatron Bridge、pipelined distributed weight sync、vLLM inference service | Deep Dive | AReaL 开始显式形成 service architecture；后续应关注 gateway backpressure、跨服务 tracing 与 refit failure recovery，而不只是吞吐 |
| NeMo RL [v0.6.0](https://github.com/NVIDIA-NeMo/RL/releases/tag/v0.6.0) | rollout / long context / precision / fault tolerance | official release；speculative rollout、SGLang backend、YaRN、chunked CE、sequence packing、LoRA GRPO/DPO、fault-tolerance launcher | Deep Dive | online drafter refit、长上下文内存和 generation backend 已进入同一 RL pipeline；AReaL 可重点借鉴 policy+drafters 的联合版本管理 |
| OpenRLHF [v0.10.0](https://github.com/OpenRLHF/OpenRLHF/releases/tag/v0.10.0) | rollout / async sampling | official release；async mode 支持 `vLLM gen batch size > rollout batch size` oversampling，并加入 VLM RLHF | Observe | oversampling 能缓冲过滤/无效样本，但必须定义多生成样本如何进入 group normalization、如何限流以及未消费结果如何回收 |

## 对仓库的影响

- 需要更新的 topic：[Agentic RL](../../rl-infra/topics/agentic_rl.md), [Long-context Training](../../training-infra/topics/long_context_training.md), [Context Parallelism](../../training-infra/topics/context_parallelism.md), [FP8](../../systems/topics/fp8.md), [NCCL](../../systems/topics/nccl.md), [Distributed Training](../../training-infra/topics/distributed_training.md)
- 需要更新的 insight：可以后续补一篇“RL training stack 的瓶颈来自调度、精度和 serving engine 三方一致性”
- 需要更新的 playbook：[Rollout Latency](../../practice/playbooks/rollout_latency.md) 应加入 DORA 的 long-tailed trajectory / bounded staleness 视角；NCCL/TP 排障后续可加入 CommFuse 的 tail latency 思路
- 需要新增的 experiment：FP8 RL rollout/training numerical drift check、sequence parallel activation checkpoint benchmark、communication overlap tail latency probe
- 需要进入 historical backfill 的材料：无。本文件自身是 2026-04 月度前沿沉淀。

## 下月关注

- DORA / AReaL / HybridFlow / Miles 是否收敛到同一类异步 rollout-training 架构。
- FP8 / NVFP4 是否从 pretraining 进一步进入 RL rollout、KV cache、attention 和 reward/verifier pipeline。
- AutoSP / CP 是否把长上下文训练从手工并行配置推进到 compiler/runtime 自动化。
- 通信优化是否从压缩字节数转向重排 collective、隐藏尾延迟和提高 overlap 稳定性。


[返回月度阅读入口](monthly_reviews.md) · [2025—2026 H1 GitHub 历史索引](github_history_2025_to_2026_h1.md)

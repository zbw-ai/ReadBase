# Master Reading List

[首页](../README.md) · [最近更新](../README.md#recent-updates) · [研究入口](README.md) · [知识地图](../KNOWLEDGE_GRAPH.md)

这里是全库的来源材料索引，不是另一份当前学习计划。现有积累以训练与 Agentic RL 为主；保留其历史演进关系，再按当前问题扩展硬件、推理与具身材料。缺少完整笔记的方向明确标出，不用目录数量代表掌握程度。

找原训练手册里的技术正文、学习计划或面试资料，走[原文档集中入口](../02-training-infra/README.md#original-library)；找原始来源继续读本页。[NVIDIA MoE 五份中文 PDF](#megatron-core-moe-2026-zh-pdf)保留在下方原索引中。

想按阶段学习，先看 [具身六阶段](../05-embodied-infra/roadmap.md)、[GPU Systems](../01-systems/roadmaps/gpu-systems.md)、[Distributed Systems](../01-systems/roadmaps/distributed-systems.md)、[Inference](../03-inference-infra/roadmap.md)或 [Agentic RL](../04-rl-infra/roadmap.md)，再回本页查来源；路线中的候选材料不自动进入 P0/P1，也不计为已读。

| 方向 | 材料与正文入口 | 当前边界 |
|---|---|---|
| 硬件／系统 | [系统基础](../01-systems/README.md)、[ml-engineering 选读来源](https://github.com/stas00/ml-engineering/) | 原书按问题选读；不是已消化全集，新拓扑与 Host/I/O 章待建设 |
| 训练 | 下方模型／并行／状态／精度材料；[训练导航](../02-training-infra/README.md) | FSDP/ZeRO 与多模态 workload 是近期深化方向 |
| 推理 | [推理导航](../03-inference-infra/README.md)、[工程博客](engineering_blogs/README.md) | 现有内容局部覆盖，独立 serving 章节待建设 |
| RL／Agent | [框架与机制](../04-rl-infra/README.md)、[历史来源](tracking/historical_backfill.md) | 以已有来源笔记和代码分析为基线，不自动视作实验验证 |
| 具身 | [模型入门大纲](../docs/superpowers/specs/2026-09-18-embodied-models-primer-design.md)、[具身路径](../05-embodied-infra/README.md) | 模型 primer 与 episode 数据正文待建设；新材料进入统一雷达 |

## 0. 使用方式

- 每篇先读 `解决的问题`、`工程价值`、`生产环境思考题`，再决定是否深读公式和实验。
- 每周最多两篇核心材料，重点写下“如果我在维护训练平台，会改哪个模块”。
- 读完论文后同步更新对应 `topics/` 文档，避免知识停在单篇材料里。
- 面试准备走[独立入口](../interview/README.md)，机制看[技术地图](../KNOWLEDGE_GRAPH.md)，排障看 [playbooks](../practice/playbooks/README.md)，来源看本目录的 papers、tech_reports 和 engineering_blogs。
- 新资料先进入 [Tracking Radar](tracking/README.md)，再筛选进 reading queue 或 topics，避免 reading list 被热点淹没。

## 0.0 Tracking Radar

`tracking/` 是持续更新的研究雷达，不要求完整解读，只记录信号和判断。

| 入口 | 用途 |
|---|---|
| [Scan Log](tracking/scan_log.md) | 记录每次扫描窗口和下一次游标 |
| [Frontier Scan Template](tracking/frontier_scan_template.md) | 从上次扫描游标到现在的最新前沿扫描模板 |
| [Monthly Signal Report Template](tracking/monthly_signal_report_template.md) | 每月高质量正式信号沉淀模板 |
| [Historical Backfill](tracking/historical_backfill.md) | 历史精华补录：补当前工程判断缺口 |
| [Backfill By Month](tracking/backfill/README.md) | 历史材料按原始月份倒序归档 |
| [Engineering Blogs Tracking](tracking/engineering_blogs.md) | 大厂工程博客追踪 |
| [Release Notes](tracking/release_notes.md) | 模型、框架、训练栈发布记录 |
| [Infra Trends](tracking/infra_trends.md) | 训练基础设施演进时间线 |
| [Agentic RL](tracking/agentic_rl.md) | Agentic RL / rollout infra / verifier 专题追踪 |
| [月度阅读总览](tracking/monthly_reviews.md) | 按月份理解主线，再查原始证据 |
| [研究雷达](tracking/README.md) | 最新扫描和历史报告的统一入口 |
| [当前阅读队列](reading_queue/README.md) | 当前 P0/P1 只在此维护，不在索引复制名单 |

已建立的研究关联：

- [2025 季度／2026 月度复盘](tracking/monthly_reviews.md) → [2025—2026 H1 GitHub 补证](tracking/github_history_2025_to_2026_h1.md)与[7–9 月既有审计](tracking/github_retrospective_2026-07_to_2026-09.md)。两轮来源范围和覆盖缺口分别保留。
- [9 月正式月报](tracking/monthly_signal_2026-09.md) → [可信经验交付](../04-rl-infra/topics/agentic_rl.md#september-2026-monthly)。
- [10/08 扫描](tracking/frontier_scan_2026-10-08.md) → [MoE 联合设计](../02-training-infra/topics/moe.md#october-2026-joint-design)／[RL 三个边界](../04-rl-infra/topics/agentic_rl.md#october-2026-contracts) → [阅读决策原记录](reading_queue/P1.md#october-2026-priority)。这是固定来源关联，不在此更新“最新”名单。

## 0.2 Research OS 工作流

| 阶段 | 目录 | 作用 |
|---|---|---|
| Signals | [tracking](tracking/README.md) | frontier scan 扫描前沿，monthly 正式沉淀 |
| Queue | [reading_queue](reading_queue/README.md) | 决定 P0 / P1 |
| Notes | `papers/` / `tech_reports/` / `engineering_blogs/` | 消化原始资料 |
| Topics | [topics](../KNOWLEDGE_GRAPH.md) | 沉淀工程手册 |
| Insights | [insights](insights/README.md) | 形成个人判断 |
| Projects | [Q3 Long-context Agentic RL](../practice/projects/2026-q3-long-context-agentic-rl/README.md) | 用真实系统 baseline、tracing 和实验形成长期工程闭环 |
| Experiments | [experiments](../practice/experiments/README.md) | 实验验证 |
| Playbooks | [playbooks](../practice/playbooks/README.md) | 生产排障 runbook |
| Learning Log | [learning_log](learning_log/README.md) | 月度复盘 |

## 0.3 Active Closed Loops

这里记录已建立的材料、正文与实践关联；有完整链接不等于实验已经执行，实际完成状态以原记录为准。

| 主题 | Input | Queue | Topic | Insight | Project / experiment | Playbook | Log |
|---|---|---|---|---|---|---|---|
| Agentic RL Infra | [Historical Backfill](tracking/historical_backfill.md) | [P0](reading_queue/P0.md) | [Agentic RL](../04-rl-infra/topics/agentic_rl.md) / [Framework Selection](../04-rl-infra/topics/rl_framework_selection.md) | [001](insights/001_agentic_rl_will_change_training_infra.md) | [Q3 Long-context Agentic RL](../practice/projects/2026-q3-long-context-agentic-rl/README.md) | [Rollout Latency](../practice/playbooks/rollout_latency.md) | [2026-06](learning_log/2026/2026-06.md) |

## 0.4 Historical Backfill 入口

[Historical Backfill](tracking/historical_backfill.md) 用来补录历史精华材料。它不代表本周新趋势，而是回答：过去有哪些材料已经被行业验证重要，但当前仓库还没有充分吸收？

第一批补录主题是 Agentic RL / Rollout Infra Classics。以下保留材料用途，当前优先级以队列为准：

| 材料 | 为什么 |
|---|---|
| AReaL | 异步 rollout/train 解耦和 staleness 控制；见 [RL Framework Selection](../04-rl-infra/topics/rl_framework_selection.md) |
| HybridFlow / verl | RLHF dataflow 和 actor resharding；见同一框架章 |
| Agent Lightning | agent runtime 与 trainer 解耦 |
| OpenRLHF | Ray + vLLM + DeepSpeed 多组件调度 |
| vLLM + OpenRLHF Integration | rollout inference / weight sync / placement group |
| SkyRL | long-horizon tool-use agent training |
| DAPO | reasoning RL recipe 如何落到系统栈 |
| NVIDIA NeMo RL | NVIDIA post-training stack 演进 |

## 0.1 Engineering Blog 入口

2026 年以后，很多训练基础设施信息不会以 paper 形式出现，而是出现在工程博客、官方文档、release note 和厂商技术文章里。它们不替代论文，但会补足真实实现细节。

| 来源 | 入口 | 重点关注 |
|---|---|---|
| NVIDIA | [NVIDIA Engineering Blogs](engineering_blogs/nvidia/README.md) | Megatron-Core、Transformer Engine、NCCL、FP8、distributed checkpointing |
| OpenAI | [OpenAI Engineering Blogs](engineering_blogs/openai/README.md) | post-training、reasoning、evaluation、infra signals |
| Anthropic | [Anthropic Engineering Blogs](engineering_blogs/anthropic/README.md) | long context、post-training、安全和评估系统 |
| Hugging Face | [Hugging Face Blog](https://huggingface.co/blog) | TRL、Transformers、Accelerate、PEFT、Kernels、vLLM integration、rollout correctness |
| DeepSeek | [DeepSeek Engineering Blogs](engineering_blogs/deepseek/README.md) | DeepSeekMoE、FP8、DualPipe、reasoning RL |
| Google | [Google Engineering Blogs](engineering_blogs/google/README.md) | TPU、Pathways、Gemini、JAX/PAX |
| Meta | [Meta Engineering Blogs](engineering_blogs/meta/README.md) | Llama、PyTorch Distributed、FSDP、数据与训练平台 |
| Microsoft | [Microsoft Engineering Blogs](engineering_blogs/microsoft/README.md) | DeepSpeed、ZeRO、Azure training infra |
| ByteDance | [ByteDance Engineering Blogs](engineering_blogs/bytedance/README.md) | 训练平台、MoE、调度、稳定性 |
| Zhipu | [Zhipu Engineering Blogs](engineering_blogs/zhipu/README.md) | GLM、中文大模型、长上下文、国产集群适配 |

结构化索引：[Blogs CSV](references/blogs.csv)。

## 1. 基础模型

| 顺序 | 材料 | 仓库笔记 | 关注点 |
|---|---|---|---|
| 1 | Attention Is All You Need | [Transformer](papers/transformer.md) | Self-Attention 为什么让训练并行化 |
| 2 | BERT | [BERT](papers/bert.md) | Encoder-only 训练负载、MLM、预训练范式 |
| 3 | GPT-3 | [GPT-3](papers/gpt3.md) | Dense Decoder-only 扩展和 scaling law 工程压力 |

## 2. 并行训练

先读统一工程入口：[Megatron 5D 并行](../02-training-infra/topics/distributed_training.md)。它先建立 DP/TP/PP/CP/EP 的决策框架，再按需要进入各单项专题。

| 顺序 | 材料 | 仓库笔记 | 关注点 |
|---|---|---|---|
| 4 | Megatron-LM 2019 | [Megatron-LM](papers/megatron_lm.md) | Tensor Parallel 的工程起点 |
| 5 | Megatron-LM 2021 | [Megatron 2021](papers/megatron_2021.md) | TP + PP + DP 组合、interleaved pipeline |
| 6 | GPipe | [GPipe](papers/gpipe.md) | Micro-batch pipeline 和 bubble |
| 7 | PipeDream | [PipeDream](papers/pipedream.md) | 1F1B、weight stashing、pipeline 调度 |
| 8 | Alpa | [Alpa](papers/alpa.md) | 自动并行搜索和 production 可控性边界 |

### MoE 原文资料

<a id="megatron-core-moe-2026-zh-pdf"></a>
#### Megatron Core MoE 2026 中文翻译（PDF）

既有 NVIDIA 88 页报告 `Scalable Training of Mixture-of-Experts Models with Megatron Core` 中文翻译原文件归档。按顺序阅读，术语与关键数字回到[英文原始论文](https://arxiv.org/abs/2603.07685)核对：

1. [第一部分：摘要 + 第 1 节](<papers/sources/scalable-training-moe-megatron-core-2026/MoE训练论文翻译（第一部分：摘要 + 第1节）.pdf>)
2. [第二部分：第 2–3 节，架构与并行策略](<papers/sources/scalable-training-moe-megatron-core-2026/MoE训练论文翻译（第二部分：第2-3节 架构与并行策略）.pdf>)
3. [第三部分：第 4 节，突破三堵墙](<papers/sources/scalable-training-moe-megatron-core-2026/MoE训练论文翻译（第三部分：第4节 突破三堵墙）.pdf>)
4. [第四部分：低精度与长上下文训练](<papers/sources/scalable-training-moe-megatron-core-2026/第四部分：第5节——FP8_FP4低精度训练 & 第6节——长上下文训练.pdf>)
5. [第五部分：生产特性、性能评估、最佳实践与 RL 支持](<papers/sources/scalable-training-moe-megatron-core-2026/第五部分：第7–10节——生产特性、性能评估、最佳实践与RL支持.pdf>)

### Tensor Parallelism 支撑材料

先读工程手册章节：[Tensor Parallelism](../02-training-infra/topics/tensor_parallelism.md)。复习时沿 [TP 前后向与 SP 通信](../02-training-infra/topics/tensor_parallelism.md#tp-collective-derivation) → [Ring AllReduce 四卡六步](../01-systems/topics/nccl.md#ring-allreduce) → [主文档口述答案](../private_resume/2026-08-llm-infra-interview-prep.md#megatron-02)。它由以下材料支撑：

| 材料 | 为什么支撑 TP |
|---|---|
| [Transformer](papers/transformer.md) | TP 切分对象来自 QKV、Output Projection、MLP projection |
| [Megatron-LM](papers/megatron_lm.md) | Column/Row Parallel Linear 的核心来源 |
| [Megatron 2021](papers/megatron_2021.md) | TP 与 PP/DP 组成 3D parallel |
| [Sequence Parallelism](../02-training-infra/topics/sequence_parallelism.md) | TP 后续降低 activation 显存的直接演进 |
| [Context Parallelism](../02-training-infra/topics/context_parallelism.md) | 长上下文下与 TP 互补 |
| [NCCL / Network](../01-systems/topics/nccl.md) | TP 排障最终落到 collective 和拓扑 |

### Long-context Training 支撑材料

先读专题入口：[Long-context Training](../02-training-infra/topics/long_context_training.md)。它不是单一 paper 线，而是横跨 pretraining / SFT / RL 的系统主题。

| 材料 | 为什么支撑长上下文训练 |
|---|---|
| [Transformer](papers/transformer.md) | attention/MLP 是长上下文训练的基本计算图 |
| [FlashAttention](papers/flashattention.md) | 长上下文首先暴露 attention IO 和 kernel 瓶颈 |
| [Context Parallelism](../02-training-infra/topics/context_parallelism.md) | 单条长序列跨 GPU 切分的核心机制 |
| [Hierarchical CP](../02-training-infra/topics/context_parallelism.md#hierarchical-cp) | sequence/head 布局交换、两级 process groups、TP 后 KV heads 约束与实测取舍 |
| [Sequence Parallelism](../02-training-infra/topics/sequence_parallelism.md) | 降低 activation 显存，与 TP/CP 配合 |
| [选择性重计算](../02-training-infra/topics/long_context_training.md#selective-recompute) | 保存边界、MCore 参数、Dense/MoE/MLA 模块选择、FlashAttention 后的收益重估与排障 |
| [CP-local logits 案例](../02-training-infra/topics/long_context_training.md#cp-local-logits) | 解释 CP 已切分但 loss/logprob 又 materialize 全序列的静默显存问题 |
| [Transformer Engine / Fusion](../01-systems/topics/transformer_engine.md#fusion-map) | Attention、Norm、MLP、MoE 与 loss fusion 的接入和数值验收 |
| [Checkpointing](../02-training-infra/topics/checkpointing.md) | 长 step time 下保存、恢复和异步 checkpoint 更关键 |
| [Agentic RL](../04-rl-infra/topics/agentic_rl.md) | RL 阶段把长 prompt/response、rollout、KV cache 和 reward/verifier 带入训练系统 |
| [CompactionRL](papers/compactionrl.md) | long-horizon agent 在固定 context budget 下训练可压缩 trajectory |
| [Llama 3](tech_reports/llama3.md) | 128K context 的大规模训练报告入口 |

## 3. 显存优化

| 顺序 | 材料 | 仓库笔记 | 关注点 |
|---|---|---|---|
| 9 | ZeRO | [ZeRO](papers/zero.md) | Optimizer/gradient/parameter state 分片 |
| 10 | ZeRO-Offload | [ZeRO-Offload](papers/zero_offload.md) | CPU offload 的带宽和延迟代价 |
| 11 | ZeRO-Infinity | [ZeRO-Infinity](papers/zero_infinity.md) | NVMe/CPU/GPU 分层内存 |
| 12 | FSDP | [FSDP](papers/fsdp.md) | PyTorch 原生参数分片和 all-gather 生命周期 |

### Checkpointing 支撑材料

先读工程手册章节：[Checkpointing](../02-training-infra/topics/checkpointing.md)。它由以下材料支撑：

| 材料 | 为什么支撑 Checkpoint |
|---|---|
| [ZeRO](papers/zero.md) | optimizer/gradient/parameter 分片让 checkpoint 变成分布式状态问题 |
| [ZeRO-Offload](papers/zero_offload.md) | CPU/offload 状态影响保存和恢复路径 |
| [ZeRO-Infinity](papers/zero_infinity.md) | 分层内存训练与 checkpoint IO 边界相邻 |
| [FSDP](papers/fsdp.md) | full/sharded/local state dict 选择直接影响恢复和导出 |
| [Megatron-LM](papers/megatron_lm.md) | TP shard metadata 是 distributed checkpoint 的基本要求 |
| [Megatron 2021](papers/megatron_2021.md) | TP/PP/DP 多维并行要求 checkpoint 记录并行布局 |
| [OPT-175B](papers/opt_175b.md) | 开放训练日志提供真实故障和回滚经验 |
| [Llama 3](tech_reports/llama3.md) | 大规模 dense 模型生命周期需要 checkpoint lineage |
| [DeepSeek-V3](tech_reports/deepseek_v3.md) | MoE/FP8 训练要求保存 expert 和 precision metadata |
| [MegaScale](tech_reports/megascale.md) | 万卡训练把 checkpoint 变成吞吐和容错核心问题 |

## 4. Kernel 优化

| 顺序 | 材料 | 仓库笔记 | 关注点 |
|---|---|---|---|
| 13 | FlashAttention | [FlashAttention](papers/flashattention.md) | IO-aware exact attention |
| 14 | FlashAttention-2 | [FlashAttention-2](papers/flashattention2.md) | 更好的 work partitioning 和 Tensor Core 利用 |
| 15 | FlashAttention-3 | [FlashAttention-3](papers/flashattention3.md) | Hopper/FP8/异步流水 |

## 5. MoE

先读工程手册章节：[MoE 与 Parallel Folding](../02-training-infra/topics/moe.md#parallel-folding)，重点理解同一批物理 ranks 上的 Attention/Expert 双逻辑网格、token AllToAll 数据流和拓扑代价。

面试速查：[EP 带来的问题与解决方案](../private_resume/2026-08-llm-infra-interview-prep.md#megatron-06)——通信、负载倾斜、显存峰值、小 GEMM、overlap 与正确性。

| 顺序 | 材料 | 仓库笔记 | 关注点 |
|---|---|---|---|
| 16 | GShard | [GShard](papers/gshard.md) | Expert Parallel 和自动分片 |
| 17 | Switch Transformer | [Switch Transformer](papers/switch_transformer.md) | top-1 routing、capacity factor |
| 18 | DeepSpeed-MoE | [DeepSpeed-MoE](papers/deepspeed_moe.md) | MoE inference/training system |
| 19 | Mixtral | [Mixtral](tech_reports/mixtral.md) | top-2 sparse MoE 的开放模型实践 |
| 20 | DeepSeek-V3 | [DeepSeek-V3](tech_reports/deepseek_v3.md) | MLA + DeepSeekMoE + FP8 + 通信重叠 |

## 6. 超大规模训练

先读工程手册章节：[大规模训练稳定性与容错](../02-training-infra/topics/fault_tolerance.md#large-scale-training)，建立“故障概率、最慢 rank、拓扑、启动/存储惊群、checkpoint/recovery、goodput”的统一框架；再用 Llama 3 和 MegaScale 的公开生产数据校准规模判断。

| 顺序 | 材料 | 仓库笔记 | 关注点 |
|---|---|---|---|
| 21 | PaLM | [PaLM](papers/palm.md) | 540B dense 模型和 Pathways |
| 22 | OPT-175B | [OPT-175B](papers/opt_175b.md) | 开放复现、训练日志、失败经验 |
| 23 | Llama 2 | [Llama 2](tech_reports/llama2.md) | 开放模型训练配方 |
| 24 | Llama 3 | [Llama 3](tech_reports/llama3.md) | 405B、128K context、生产训练流程 |
| 25 | MegaScale | [MegaScale](tech_reports/megascale.md) | 10K+ GPU 训练系统稳定性 |

## 7. NVIDIA 高级主题

| 顺序 | 主题 | 仓库笔记 | 关注点 |
|---|---|---|---|
| 26 | Sequence Parallelism | [Sequence Parallelism](../02-training-infra/topics/sequence_parallelism.md) | activation 显存和 TP 配合 |
| 27 | Context Parallelism | [Context Parallelism](../02-training-infra/topics/context_parallelism.md) | 长上下文切分、attention 通信 |
| 28 | Transformer Engine / Fusion | [Transformer Engine 与 NVIDIA 融合算子](../01-systems/topics/transformer_engine.md) | FP8、Attention/Norm/MLP/MoE fusion、接入与数值验收 |
| 29 | Distributed Checkpointing | [Checkpointing](../02-training-infra/topics/checkpointing.md) | 异步保存、重分片、恢复时间 |
| 30 | NCCL / Network | [NCCL](../01-systems/topics/nccl.md)；[Ring AllReduce](../01-systems/topics/nccl.md#ring-allreduce) | collective 语义、逐步传块、通信量与算法选择、拓扑和 straggler 诊断 |

面试应用：[通用 Infra 与生产排障](../private_resume/2026-08-llm-infra-interview-prep.md#part-v)，串联 [Embedding / PS / 多级存储](../private_resume/2026-08-llm-infra-interview-prep.md#infra-10)、[数据 pipeline](../private_resume/2026-08-llm-infra-interview-prep.md#infra-11) 与 [Checkpoint 保存和恢复](../private_resume/2026-08-llm-infra-interview-prep.md#infra-08)；[一面手撕 LCA](../private_resume/2026-09-interview-coding.md#coding-03)收录在独立 Coding 题单。

后训练框架面试应用：[RL 框架与后训练通用题](../private_resume/2026-08-llm-infra-interview-prep.md#part-iii)，按框架源码、算法流程、异步 Rollout、训推并行、长轨迹显存、性能与平台能力组织；原理延伸回到[框架选型](../04-rl-infra/topics/rl_framework_selection.md)和 [Agentic RL](../04-rl-infra/topics/agentic_rl.md)。按技术主题与 P0/P1/P2 组织，可跨公司复用。

本人现场题目：[小红书一面 LRU 缓存 get / put](../private_resume/2026-09-interview-coding.md#coding-04)，保留 OrderedDict 解法，补充 Python3 可运行测试、复杂度和手写哈希表＋双向链表追问；不据此推断面试结果。

通用基础面试：[主文档 GPU / PyTorch / 低精度 Part](../private_resume/2026-08-llm-infra-interview-prep.md#part-foundations)，按 topic 再分 P0/P1；详细原理进入 [FP8](../01-systems/topics/fp8.md)与 [GPU 执行 / 编译 / Roofline](../01-systems/topics/transformer_engine.md#gpu-execution)，长代码进入 [Coding 梯度检查与性能练习](../private_resume/2026-09-interview-coding.md#coding-05)。[公司资料](../private_resume/2026-09-meshy-ml-system-interview-prep.md#meshy-public-work)在独立旧入口保存，不影响主文档通用题目的组织。[Meshy T2 历史补录](tracking/backfill/2026-07.md#meshy-t2)与 [P1 阅读](reading_queue/P1.md#meshy-t2-reading)保留公开架构和未复现边界。

## 8. Agentic RL / Rollout Infra

团队分享：[MiMo-V2.6 / CodeMidas：从环境工厂到大规模 Agentic RL](tech_reports/mimo_v26.md)，串联 [环境与消费配比契约](../04-rl-infra/topics/agentic_rl.md#mimo-v26-environment-contract)、[验证计划](../practice/experiments/mimo_v26_environment_and_mixer.md) 与 [P1 精读记录](reading_queue/P1.md#mimo-v26-reading)。已核验原报告和直播指标，尚未复现。

2026-10-08 补充 [GAGAR 论文与盲审证据](tech_reports/mimo_v26.md#25-gagar控制实验与独立质量审核)、[MOPD 行为修复](tech_reports/mimo_v26.md#51-工具调用重复与-mopd-修复)和[开源交付](tech_reports/mimo_v26.md#8-开源资源与复现条件)；新增[过程行为回归实验 F](../practice/experiments/mimo_v26_environment_and_mixer.md#release-behavior)。原始日期与研究决策见 [2026-09 backfill](tracking/backfill/2026-09.md)，实验仍未执行。

2026-10-08 演进补充：[MiMo 2025–2026 算法与 infra 主线](tech_reports/mimo_v26.md#7-算法与基础设施的演进) ↔ [MOPD 角色与状态覆盖](../04-rl-infra/topics/mopd.md#mimo-evolution-roles) ↔ [经验生命周期判断](insights/001_agentic_rl_will_change_training_infra.md#mimo-evolution-judgment)。时间线区分论文、产品与修订日期；Audio / Embodied 仅为背景核验，serving 与 RL 分开计量。

定向精读：[GLM Infra Agent 与 RSI 证据评估](engineering_blogs/zhipu/glm_infra_agent_recursive_self_improvement.md)，配合 [Agentic RL](../04-rl-infra/topics/agentic_rl.md) 阅读。下一步执行报告中的验证计划，当前未复现。

| 顺序 | 材料 | 仓库笔记 | 关注点 |
|---|---|---|---|
| 31 | CompactionRL | [CompactionRL](papers/compactionrl.md) | long-horizon agent 的 context compaction、segment loss 和 cross-trajectory credit assignment |
| 32 | AReaL | [RL Framework Selection](../04-rl-infra/topics/rl_framework_selection.md)；[项目 Gateway 源码审阅](../04-rl-infra/topics/agentic_rl.md#project-gateway-ownership)；[面试速答](../private_resume/2026-08-llm-infra-interview-prep.md#areal-09) | 异步 rollout/train 解耦；cohort 粘性路由、session 准入、engine 负载、版本预算与团队/个人边界 |
| 33 | HybridFlow / verl | [RL Framework Selection](../04-rl-infra/topics/rl_framework_selection.md)；[美团 Fully Async 实践](../04-rl-infra/topics/agentic_rl.md#meituan-fully-async-practice)（[历史来源](tracking/backfill/2026-01.md#meituan-fully-async)）；[原图与面试题专题](../private_resume/2026-08-llm-infra-interview-prep.md#fully-async-study) | RLHF dataflow、actor resharding；四模式、streaming、partial rollout、陈旧度预算与公开实验边界 |
| 34 | Agent Lightning | Tracking / P0 | agent runtime 与 trainer 解耦、trace schema |
| 35 | Traditional KD → OPD → MOPD | [MOPD（研究中 / 原理第一版）](../04-rl-infra/topics/mopd.md) | Student rollout、dense Teacher signal、domain routing、multi-teacher serving |

## 9. Agentic for Embodied / Physical Agent Infra

先读系统地图：[Agentic for Embodied](../05-embodied-infra/topics/agentic_for_embodied.md)。这条路线不要求先掌握机器人控制理论，而是先理解数据、仿真、实时 runtime、安全和 fleet feedback 怎样改变 Agentic RL 的工程边界。

| 顺序 | 材料 | 关注点 |
|---|---|---|
| 36 | [RT-2](https://robotics-transformer2.github.io/) | action tokenization 如何把 VLM 扩展为 VLA |
| 37 | [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) | continuous action distribution、action chunk 与推理时延 |
| 38 | [Open X-Embodiment](https://robotics-transformer-x.github.io/) | 跨 robot / sensor / action space 的数据标准化 |
| 39 | [OpenVLA](https://openvla.github.io/) | 开放 VLA 训练 pipeline、checkpoint 和 adaptation |
| 40 | [LeRobotDataset v3](https://huggingface.co/docs/lerobot/lerobot-dataset-v3) | video + Parquet + episode metadata 的工程数据布局 |
| 41 | [Isaac Lab](https://developer.nvidia.com/isaac/lab) | GPU simulation、parallel environment、reset 与 evaluation |
| 42 | [Real-Time Chunking](https://arxiv.org/abs/2506.07339) | action chunk 的异步实时执行与 deadline 问题 |
| 43 | [GR00T end-to-end workflow](https://developer.nvidia.com/blog/develop-humanoid-robot-policies-end-to-end-with-nvidia-isaac-gr00t/) | 厂商 data -> sim -> train -> eval -> deploy 平台路线 |

- [2026-09-22 Frontier Scan](tracking/frontier_scan_2026-09-22.md)：DSec → agent/sandbox 恢复，Conduit → 经验数据面，FP8 RL → 数值反馈；接入 [P1](reading_queue/P1.md) 与 [RL 状态实验](../practice/experiments/rl_state_boundaries.md)。

- [2026-09-22 GitHub 全面补扫](tracking/github_audit_2026-09-22.md)：闭合 GitHub 索引缺口；VLM CP/MTP、prefill workspace、KV lifecycle 与 release 验收，关联 [Agentic RL](../04-rl-infra/topics/agentic_rl.md#rl-state-boundaries) 和 [实验](../practice/experiments/rl_state_boundaries.md)。

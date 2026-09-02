# Master Reading List

这个 reading list 按“训练系统能力如何长出来”组织，而不是按发表时间机械排序。阅读目标是建立工程判断：一个训练平台在不同模型规模、集群规模、上下文长度和稳定性要求下，应该如何选择并行策略、显存策略、通信策略和容错策略。

## 0. 使用方式

- 每篇先读 `解决的问题`、`工程价值`、`生产环境思考题`，再决定是否深读公式和实验。
- 每周最多两篇核心材料，重点写下“如果我在维护训练平台，会改哪个模块”。
- 读完论文后同步更新对应 `topics/` 文档，避免知识停在单篇材料里。
- 面试准备走 `interview/`，生产复盘走 `topics/`，论文历史走 `papers/` 和 `tech_reports/`，工程实现细节走 `engineering_blogs/`。
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
| [Frontier Scan 2026-09-01](tracking/frontier_scan_2026-09-01.md) | 最新 frontier scan，覆盖到 2026-09-01 11:31:29 |
| [Frontier Scan 2026-08-30](tracking/frontier_scan_2026-08-30.md) | 上一次 frontier scan，聚焦异步恢复、trajectory correctness 与长上下文 VPP |
| [Monthly Signal 2026-08](tracking/monthly_signal_2026-08.md) | 2026 年 8 月五条高质量工程主线与工业证据汇总 |
| [Monthly Signal 2026-07](tracking/monthly_signal_2026-07.md) | 2026 年 7 月工程判断与工业证据月报 |
| [Monthly Signal 2026-06](tracking/monthly_signal_2026-06.md) | 2026 年 6 月高质量前沿信号沉淀 |
| [Monthly Signal 2026-05](tracking/monthly_signal_2026-05.md) | 2026 年 5 月高质量前沿信号沉淀 |
| [Monthly Signal 2026-04](tracking/monthly_signal_2026-04.md) | 2026 年 4 月高质量前沿信号沉淀 |
| [Monthly Signal 2026-03](tracking/monthly_signal_2026-03.md) | 2026 年 3 月高质量前沿信号沉淀 |
| [Monthly Signal 2026-02](tracking/monthly_signal_2026-02.md) | 2026 年 2 月高质量前沿信号沉淀 |
| [Monthly Signal 2026-01](tracking/monthly_signal_2026-01.md) | 2026 年 1 月高质量前沿信号沉淀 |
| [Weekly Signal 2026-W26](tracking/weekly_signal_2026-W26.md) | 旧 weekly 修正记录 |

## 0.2 Research OS 工作流

| 阶段 | 目录 | 作用 |
|---|---|---|
| Signals | [tracking](tracking/README.md) | frontier scan 扫描前沿，monthly 正式沉淀 |
| Queue | [reading_queue](reading_queue/README.md) | 决定 P0 / P1 |
| Notes | `papers/` / `tech_reports/` / `engineering_blogs/` | 消化原始资料 |
| Topics | [topics](topics/) | 沉淀工程手册 |
| Insights | [insights](insights/README.md) | 形成个人判断 |
| Projects | [Q3 Long-context Agentic RL](projects/2026-q3-long-context-agentic-rl/README.md) | 用真实系统 baseline、tracing 和实验形成长期工程闭环 |
| Experiments | [experiments](experiments/README.md) | 实验验证 |
| Playbooks | [playbooks](playbooks/README.md) | 生产排障 runbook |
| Learning Log | [learning_log](learning_log/README.md) | 月度复盘 |

## 0.3 Active Closed Loops

这里记录已经跑通的 Research OS 闭环。它们不是单篇论文阅读，而是从信号判断进入工程沉淀。

| 主题 | Input | Queue | Topic | Insight | Project / experiment | Playbook | Log |
|---|---|---|---|---|---|---|---|
| Agentic RL Infra | [Historical Backfill](tracking/historical_backfill.md) | [P0](reading_queue/P0.md) | [Agentic RL](topics/agentic_rl.md) / [Framework Selection](topics/rl_framework_selection.md) | [001](insights/001_agentic_rl_will_change_training_infra.md) | [Q3 Long-context Agentic RL](projects/2026-q3-long-context-agentic-rl/README.md) | [Rollout Latency](playbooks/rollout_latency.md) | [2026-06](learning_log/2026/2026-06.md) |

## 0.4 Historical Backfill 入口

[Historical Backfill](tracking/historical_backfill.md) 用来补录历史精华材料。它不代表本周新趋势，而是回答：过去有哪些材料已经被行业验证重要，但当前仓库还没有充分吸收？

第一批补录主题是 Agentic RL / Rollout Infra Classics。当前筛选结果：

| 队列 | 材料 | 为什么 |
|---|---|---|
| P0 | AReaL | 补异步 rollout/train 解耦和 staleness 控制；沉淀到 [RL Framework Selection](topics/rl_framework_selection.md) |
| P0 | HybridFlow / verl | 补 RLHF dataflow 和 actor resharding；沉淀到 [RL Framework Selection](topics/rl_framework_selection.md) |
| P0 | Agent Lightning | 补 agent runtime 与 trainer 解耦 |
| P1 | OpenRLHF | 补 Ray + vLLM + DeepSpeed 多组件调度 |
| P1 | vLLM + OpenRLHF Integration | 补 rollout inference / weight sync / placement group |
| P1 | SkyRL | 补 long-horizon tool-use agent training |
| P1 | DAPO | 补 reasoning RL recipe 如何落到系统栈 |
| P1 | NVIDIA NeMo RL | 补 NVIDIA post-training stack 演进 |

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

先读统一工程入口：[Megatron 5D 并行](topics/distributed_training.md)。它先建立 DP/TP/PP/CP/EP 的决策框架，再按需要进入各单项专题。

| 顺序 | 材料 | 仓库笔记 | 关注点 |
|---|---|---|---|
| 4 | Megatron-LM 2019 | [Megatron-LM](papers/megatron_lm.md) | Tensor Parallel 的工程起点 |
| 5 | Megatron-LM 2021 | [Megatron 2021](papers/megatron_2021.md) | TP + PP + DP 组合、interleaved pipeline |
| 6 | GPipe | [GPipe](papers/gpipe.md) | Micro-batch pipeline 和 bubble |
| 7 | PipeDream | [PipeDream](papers/pipedream.md) | 1F1B、weight stashing、pipeline 调度 |
| 8 | Alpa | [Alpa](papers/alpa.md) | 自动并行搜索和 production 可控性边界 |

### Tensor Parallelism 支撑材料

先读工程手册章节：[Tensor Parallelism](topics/tensor_parallelism.md)。它由以下材料支撑：

| 材料 | 为什么支撑 TP |
|---|---|
| [Transformer](papers/transformer.md) | TP 切分对象来自 QKV、Output Projection、MLP projection |
| [Megatron-LM](papers/megatron_lm.md) | Column/Row Parallel Linear 的核心来源 |
| [Megatron 2021](papers/megatron_2021.md) | TP 与 PP/DP 组成 3D parallel |
| [Sequence Parallelism](topics/sequence_parallelism.md) | TP 后续降低 activation 显存的直接演进 |
| [Context Parallelism](topics/context_parallelism.md) | 长上下文下与 TP 互补 |
| [NCCL / Network](topics/nccl.md) | TP 排障最终落到 collective 和拓扑 |

### Long-context Training 支撑材料

先读专题入口：[Long-context Training](topics/long_context_training.md)。它不是单一 paper 线，而是横跨 pretraining / SFT / RL 的系统主题。

| 材料 | 为什么支撑长上下文训练 |
|---|---|
| [Transformer](papers/transformer.md) | attention/MLP 是长上下文训练的基本计算图 |
| [FlashAttention](papers/flashattention.md) | 长上下文首先暴露 attention IO 和 kernel 瓶颈 |
| [Context Parallelism](topics/context_parallelism.md) | 单条长序列跨 GPU 切分的核心机制 |
| [Sequence Parallelism](topics/sequence_parallelism.md) | 降低 activation 显存，与 TP/CP 配合 |
| [Checkpointing](topics/checkpointing.md) | 长 step time 下保存、恢复和异步 checkpoint 更关键 |
| [Agentic RL](topics/agentic_rl.md) | RL 阶段把长 prompt/response、rollout、KV cache 和 reward/verifier 带入训练系统 |
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

先读工程手册章节：[Checkpointing](topics/checkpointing.md)。它由以下材料支撑：

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

先读工程手册章节：[MoE 与 Parallel Folding](topics/moe.md#parallel-folding)，重点理解同一批物理 ranks 上的 Attention/Expert 双逻辑网格、token AllToAll 数据流和拓扑代价。

| 顺序 | 材料 | 仓库笔记 | 关注点 |
|---|---|---|---|
| 16 | GShard | [GShard](papers/gshard.md) | Expert Parallel 和自动分片 |
| 17 | Switch Transformer | [Switch Transformer](papers/switch_transformer.md) | top-1 routing、capacity factor |
| 18 | DeepSpeed-MoE | [DeepSpeed-MoE](papers/deepspeed_moe.md) | MoE inference/training system |
| 19 | Mixtral | [Mixtral](tech_reports/mixtral.md) | top-2 sparse MoE 的开放模型实践 |
| 20 | DeepSeek-V3 | [DeepSeek-V3](tech_reports/deepseek_v3.md) | MLA + DeepSeekMoE + FP8 + 通信重叠 |

## 6. 超大规模训练

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
| 26 | Sequence Parallelism | [Sequence Parallelism](topics/sequence_parallelism.md) | activation 显存和 TP 配合 |
| 27 | Context Parallelism | [Context Parallelism](topics/context_parallelism.md) | 长上下文切分、attention 通信 |
| 28 | Transformer Engine FP8 | [Transformer Engine](topics/transformer_engine.md) | FP8 recipe、amax、scaling |
| 29 | Distributed Checkpointing | [Checkpointing](topics/checkpointing.md) | 异步保存、重分片、恢复时间 |
| 30 | NCCL / Network | [NCCL](topics/nccl.md) | collective、拓扑、straggler 诊断 |

## 8. Agentic RL / Rollout Infra

| 顺序 | 材料 | 仓库笔记 | 关注点 |
|---|---|---|---|
| 31 | CompactionRL | [CompactionRL](papers/compactionrl.md) | long-horizon agent 的 context compaction、segment loss 和 cross-trajectory credit assignment |
| 32 | AReaL | [RL Framework Selection](topics/rl_framework_selection.md) | 异步 rollout/train 解耦、staleness、sample freshness |
| 33 | HybridFlow / verl | [RL Framework Selection](topics/rl_framework_selection.md) | RLHF dataflow、actor training/generation resharding |
| 34 | Agent Lightning | Tracking / P0 | agent runtime 与 trainer 解耦、trace schema |
| 35 | Traditional KD → OPD → MOPD | [MOPD（研究中 / 原理第一版）](topics/mopd.md) | Student rollout、dense Teacher signal、domain routing、multi-teacher serving |

## 9. Agentic for Embodied / Physical Agent Infra

先读系统地图：[Agentic for Embodied](topics/agentic_for_embodied.md)。这条路线不要求先掌握机器人控制理论，而是先理解数据、仿真、实时 runtime、安全和 fleet feedback 怎样改变 Agentic RL 的工程边界。

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

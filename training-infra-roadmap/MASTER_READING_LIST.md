# Master Reading List

这个 reading list 按“训练系统能力如何长出来”组织，而不是按发表时间机械排序。阅读目标是建立工程判断：一个训练平台在不同模型规模、集群规模、上下文长度和稳定性要求下，应该如何选择并行策略、显存策略、通信策略和容错策略。

## 0. 使用方式

- 每篇先读 `解决的问题`、`工程价值`、`生产环境思考题`，再决定是否深读公式和实验。
- 每周最多两篇核心材料，重点写下“如果我在维护训练平台，会改哪个模块”。
- 读完论文后同步更新对应 `topics/` 文档，避免知识停在单篇材料里。
- 面试准备走 `interview/`，生产复盘走 `topics/`，论文历史走 `papers/` 和 `tech_reports/`。

## 1. 基础模型

| 顺序 | 材料 | 仓库笔记 | 关注点 |
|---|---|---|---|
| 1 | Attention Is All You Need | [Transformer](papers/transformer.md) | Self-Attention 为什么让训练并行化 |
| 2 | BERT | [BERT](papers/bert.md) | Encoder-only 训练负载、MLM、预训练范式 |
| 3 | GPT-3 | [GPT-3](papers/gpt3.md) | Dense Decoder-only 扩展和 scaling law 工程压力 |

## 2. 并行训练

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

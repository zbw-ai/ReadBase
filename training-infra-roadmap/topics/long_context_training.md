# Long-context Training

> 定位：长上下文训练工程章节，覆盖 LLM/MoE SFT、CP-local loss/logprob、视频 DiT Ulysses、配置选择与生产排障。
>
> 面试速答入口：[RESUME-05｜9B SFT 31s→9.3s](../../private_resume/2026-08-llm-infra-interview-prep.md#resume-05) · [RESUME-17｜35B-A3B 128K](../../private_resume/2026-08-llm-infra-interview-prep.md#resume-17) · [RESUME-07｜7.6GB CP-local logits](../../private_resume/2026-08-llm-infra-interview-prep.md#resume-07) · [RESUME-18｜视频 DiT/Ulysses](../../private_resume/2026-08-llm-infra-interview-prep.md#resume-18)

长上下文训练不是单一技术点，也不等于 Context Parallelism。

它是一条横跨 **Pretraining / SFT / RL** 的系统主线：序列长度变长以后，训练对象、数据分布、attention kernel、activation、checkpoint、rollout、reward/verifier、推理引擎和集群通信都会一起变化。

这个专题页先作为地图使用，帮助后续阅读和实验归位。

## 核心问题

短上下文训练中，主要矛盾通常是模型参数、optimizer state 和集群吞吐。长上下文训练把主要矛盾推向 token 维度：

- 单条样本可能无法放进一张 GPU。
- attention / hybrid attention / linear attention 的 kernel 行为会随 sequence length 出现性能悬崖。
- activation 和中间状态随长度放大，recompute、offload、checkpoint 需要重新取舍。
- 变长样本会造成 packing、load balance 和 step time 抖动。
- RL 阶段还会把长 prompt、长 response、KV cache、verifier 成本和 policy version 同步带进训练系统。

工程上不要问“能不能支持 128k”，而要问：

1. 数据里真实长样本占比是多少？
2. 每张卡实际处理多少 token？
3. 长样本是拖慢所有 step，还是被 packing 隔离？
4. checkpoint 和恢复是否能承受更长 step time？
5. RL rollout 的推理栈是否跟训练栈共享同一套 tokenizer、position scaling 和 policy version？

## 一套统一的优化顺序

长上下文优化最怕“先开遍所有性能开关”。更可靠的顺序是：

1. **固定 workload**：checkpoint、硬件、并行度、length/packing 分布、GBS/MBS、有效 token、精度、warmup 和统计窗口。
2. **建立显存账**：模型状态、saved activation、logits/loss、collective buffer、kernel workspace、graph pool 与 checkpoint 临时副本分开记。
3. **确认数据布局**：每个关键 tensor 是否真的按 TP/CP/EP 分片，THD/BSHD、padding、zigzag 与 loss mask 是否一致；先修静默回退和全量 materialization。
4. **拆 critical path**：data wait、attention、MLP/expert、TP/CP/EP communication、loss/logprob、backward、optimizer 与 checkpoint。
5. **再做联合搜索**：input pipeline → 并行策略 → recompute/offload → fusion/kernel → overlap。每解掉一个瓶颈都重新 profile。
6. **性能与正确性一起验收**：平均/p95 step、有效 tokens/s、peak allocated/reserved、loss/logprob/grad、save/resume 和长窗口稳定性。

<a id="qwen35-9b-sft"></a>
## Qwen3.5-9B SFT：31s → 9.3s 的工程解释

最新版简历记录的是联合结果：step time `31s→9.3s`，MFU `23%→45.2%`。可确认的优化方向是 DataLoader 并发与 prefetch、selective recompute、TP/CP 收敛；当前没有可公开的逐项同-workload 消融，不能给三项硬拆收益。

### 1. 先消除 input bubble

`num_workers=0` 时，数据读取、解码/tokenize、packing、collate 和 H2D 更容易串在训练主进程里。若 profiler 看到 GPU kernel 间存在与 `next(data_iter)` 对齐的空洞，应逐步验证：

- 增大 `num_workers`，项目底稿记录的方向是 `0→8`；
- `pin_memory` + non-blocking H2D；
- `persistent_workers` 避免 epoch/iterator 重建；
- 合理的 `prefetch_factor`，让 CPU 准备第 `n+1` 批时 GPU 计算第 `n` 批；
- tokenizer/packing 缓存、连续存储和减少小文件 metadata IO；
- 监控 CPU utilization、RSS、page fault、queue depth 和 data wait p95，防止 worker 过多反而争抢 CPU/内存。

这类优化不提高 GPU 峰值算力，而是减少 GPU 暴露的等待时间。

### 2. 从 full recompute 收敛到 selective recompute

Activation checkpointing 的目标不是“重算最便宜的模块”，而是最大化单位额外计算所释放的**峰值存活显存**：

```text
selection score ≈ bytes removed from peak live set / extra recompute FLOPs
```

Megatron-Core 当前 `recompute_granularity=selective` 默认 checkpoint `core_attn`。这并不与“Attention 很贵”矛盾：这里重算的是 core attention 子模块中内存密集但相对适合重建的中间状态，而不是机械地把整个 Attention/Transformer layer 重跑。现代版本还支持 `layernorm`、`moe_act`、`mla_up_proj`、`mlp`、`moe`、`shared_experts`、`gdn_norm_out` 等模块，其中部分使用 output-discarding checkpointing。

工程选择流程：

1. 在相同 workload 下跑 `none / selective / full` 三档，记录 peak live set 和 backward recompute time；
2. 从默认/已验证模块开始，不凭名称猜成本；
3. 用 memory snapshot 判断释放的 tensor 是否真正落在峰值窗口；
4. 显存有余量时优先减少 full-layer recompute，把省下的时间换成更大 MBS 或更少 microbatch；
5. 验证 dropout/RNG、loss、grad 与 checkpoint resume，不把“能跑”当作数值等价。

项目口径只确认从偏重 recompute 收敛到 selective；当时精确 `recompute_modules` 必须以配置为准。

### 3. 让 TP 解决权重，让 CP 解决长序列

9B 模型上 TP 过大可能让 GEMM 的 M/N/K 变小，并引入逐层高频 collective。128K 的一阶压力更多来自 sequence activation，因此在参数能够放下时，应比较“更小 TP、更大 CP”与原配置，而不是把所有卡给 TP。

比较时至少同时看：

- GEMM/FlashAttention kernel efficiency；
- TP all-reduce/all-gather/reduce-scatter exposed time；
- CP KV exchange/attention communication；
- 每 rank local sequence、activation peak 与 microbatch 数；
- 跨节点 group mapping。

另一 workload 的 `TP=4,CP=4 → TP=2,CP=8、163s→102s` 只能证明这种选择机制，不能作为 31s→9.3s 的消融项。

### 4. MFU 为什么必须做算术校验

若标准 MFU 使用相同的模型 FLOPs/step、有效 token 和 wall-clock step time，则应近似满足：

```text
MFU_new / MFU_old ≈ step_time_old / step_time_new
```

但 `31/9.3≈3.33`，`45.2/23≈1.97`，两者不能自动闭合。这不代表最新简历数字一定错误，但说明可能存在 estimator、有效 token、packing、data wait 是否计入或统计窗口差异。面试前应带上原始 MFU 公式和日志；补齐前分别陈述数字，不说它们来自完全相同的单一测量窗口。

<a id="qwen35-35b-a3b-128k"></a>
## Qwen3.5-35B-A3B 128K：为什么 active 3B 仍然难

MoE 的 active 参数决定单 token 的部分 FLOPs，但不能把整套系统当作 3B dense：

- 总参数、optimizer/main-weight 和 checkpoint 仍需按实际 sharding 放置；
- router、token dispatch/combine 与 Grouped GEMM 增加动态负载和 All-to-All；
- 128K 放大 attention/activation、CP communication 与 logits/loss 临时张量；
- expert token 不均衡会让最热 expert/rank 决定 step tail。

最新版简历的结果是平均 step time 降低约 `50%`。可验证的技术链应按以下顺序讲：

1. **修静默全量张量**：尤其是下一节的 CP full-logits gather；
2. **并行网格**：用 TP 解决单层权重/GEMM、CP 分摊 128K、EP 分布 expert；将高频 TP/EP/CP group 映射到合适拓扑；
3. **MoE kernel**：Grouped GEMM、permute/unpermute、router/top-k 与 shared-expert overlap，以 token histogram 验证负载；
4. **activation/loss**：packing/THD、selective recompute、vocab-parallel CE/logprob chunk，避免 FP32 full logits 常驻；
5. **供给与 overlap**：DataLoader、H2D、TP/CP/EP collective 与计算 overlap；
6. **重新配 batch**：释放显存后评估增大 MBS、减少 microbatch/recompute 是否更划算。

当前材料没有逐项 A/B，所以 50% 只能作为联合结果；具体 TP/CP/EP、绝对 step time 和测量窗口留在证据卡。

<a id="cp-local-logits"></a>
## CP-local logits：7.6GB 冗余分配的原理与修复

![CP-local logits 修复：保留 logits 分片，只聚合标量](../assets/topics/cp-local-logits.svg)

### 根因不是一个普通 chunk 参数

THD packed sequence、`CP>1` 的 actor 路径中，模型输出本来是：

```text
local logits: [T/CP, V/TP]
```

旧 postprocess 为恢复 packed sequence 顺序，先在 CP group all-gather 整个 logits：

```text
full logits on every CP rank: [T, V/TP]
```

因此主张量峰值从近似 `(T/CP)×(V/TP)×bytes` 回到 `T×(V/TP)×bytes`。后续 sequence chunk 只控制 log-softmax/gather 等计算产生的临时 tensor；full logits 已经存在时，chunk size 再小也不能挽回这部分显存。这就是“chunking 看起来已配置，但约 7.6GB 冗余分配仍出现”的原因。

### 正确的数据流

1. actor 的 THD+CP 路径让 `postprocess_packed_seqs_context_parallel(..., gather_thd_outputs=False)` 保持 logits local；
2. 将 padded packed labels 用和模型输入完全相同的 causal zigzag 切分到本 rank；
3. 在 `[T/CP,V/TP]` 上使用 vocab-parallel primitive 计算 selected-token logprob、entropy、vocab min/max；
4. 只 all-gather `[T/CP]` token scalar；
5. 按 zigzag 逆变换恢复 full packed order，再 unpad；
6. 用显式 `_pcp_output_layout` 标记 THD/BSHD，不能只根据 `cu_seqlens` 猜 layout。

项目提交 `be6fb98f` 对应这条路径。Critic 保留其已验证的 full-gather 路径，不能因为 actor 能 local-scalar 就默认所有输出头都同语义。

### 验证矩阵

| 维度 | 至少覆盖 |
|---|---|
| layout | THD / BSHD、packed / padded、带 `cu_seqlens` 的 BSHD |
| parallel | CP=1 reference、CP&gt;1、TP vocab shard、不同 CP rank |
| path | train、forward-only/compute-logp、actor、critic |
| numeric | token logprob、entropy、loss、grad、rank checksum |
| memory | all-gather 前后 shape、peak allocated/reserved、7.6GB 峰值消失 |
| lifecycle | padding/unpadding、checkpoint/recovery、长窗口无泄漏 |

<a id="video-dit-ulysses"></a>
## 视频 DiT：从 640×640×3×129 到 Ulysses/Ring 并行

![视频 DiT 的 Ulysses sequence parallel 数据流](../assets/topics/ulysses-video-cp.svg)

原始视频 shape 不能直接当 Attention sequence。通用链路是：

```text
[frames, channels, height, width]
  → VAE temporal/spatial compression
  → latent video
  → patchify
  → sequence tokens T' × H' × W'
  → DiT blocks
```

准确 token 数取决于 VAE temporal/spatial stride、latent channels 与 DiT patch size；没有模型配置就不要现场报死数。

### Ulysses 做了什么

Ulysses 初始让每个 rank 持有 `S/SP` token、全部 attention heads。Attention 前第一次 All-to-All 将数据重排为“全 sequence、部分 heads”；rank 在本地对 head shard 做 attention；第二次 All-to-All 再恢复 sequence-sharded output。它切 activation，不切参数，主要解决视频时空 token 过长导致的显存问题。

代价与限制：

- 两次 All-to-All 对拓扑和消息均衡敏感；
- Ulysses degree 通常受 Q/KV head 可整除性和 kernel layout 约束；
- degree 继续增大时，每 rank heads 太少，通信相对计算上升；
- 可用 `ulysses_degree × ring_degree` 组合更大序列并行网格，将高频 Ulysses A2A 放机内高速域，Ring 通信再跨节点；
- 与外层 DP/PP 组合时必须按真实 process group 算 world size，不能重复乘同一组 rank。

### 以 HunyuanVideo-14B 为面试例子的优化闭环

1. 固定 640×640×3×129、VAE/patch config、batch、precision、并行网格与质量输入；
2. 计算 latent token shape，先确认 activation 一阶峰值；
3. sweep Ulysses/Ring degree，记录 attention kernel、两次 A2A、peak memory 与 step p95；
4. 开 FlashAttention、mixed precision、selective recompute 和 QKV/RoPE/RMSNorm/MLP fusion；
5. 冻结的 VAE/text encoder 使用 no-grad、缓存或阶段化执行，减少重复计算与显存常驻；
6. 按分辨率/帧数 bucket，避免 padding 把最大视频 shape 扩散到全 batch；
7. 在国产卡场景将 Ulysses group 映射到 HCCS，跨节点通信再看 RoCE 带宽与长尾；
8. 用逐层输出、loss/grad、生成质量、checkpoint/recovery 和长稳共同验收。

HunyuanVideo 官方公开的 USP/xDiT 是推理实现和配置示例，只能支撑机制理解；不能把其 NVIDIA 推理数字说成本人国产卡训练收益。

## Pretraining 视角

预训练阶段关注的是大规模 token 吞吐和稳定性。

关键问题：

- 位置编码和上下文外推：RoPE scaling、YaRN、ALiBi、原生长上下文配置会影响训练稳定性和后续 SFT/RL 对齐。
- 数据长度分布：长文档比例、拼接策略、document boundary、packing 策略会直接影响有效 token 利用率。
- kernel 选择：FlashAttention、block/ring attention、hybrid attention、linear attention 在不同长度段表现可能完全不同。
- 并行布局：TP/PP/DP 之外，需要考虑 [Context Parallelism](context_parallelism.md) 和 [Sequence Parallelism](sequence_parallelism.md)。
- checkpoint 成本：长 step time 下，checkpoint 频率、async save 和恢复时间会影响集群有效利用率。

预训练里最容易踩的坑是把 max length 当成静态参数，只改 `max_position_embeddings` 或 rope scaling，却没有重新评估 kernel、packing、batch token budget 和 checkpoint 周期。

## SFT 视角

SFT 阶段关注的是样本质量、loss mask 和长尾样本处理。

关键问题：

- multi-turn 样本的 loss mask 是否正确，特别是 tool call、assistant-only loss、system prompt 和 long document grounding。
- packing 是否保持语义边界，是否会让长样本挤掉短样本吞吐。
- `max_length`、`max_token_len_per_gpu`、micro batch、global token batch 的关系是否清楚。
- 长样本是否需要单独 bucket，还是统一走 dynamic packing。
- activation recompute / optimizer offload / CPU offload 是否真的提升可训练性，还是吞吐损失过大。

SFT 里最常见的问题不是“放不下”，而是：

- padding 或 packing 造成大量无效 token；
- 少量超长样本把 step time 拉爆；
- tokenizer/chat template/loss mask 不一致导致 loss spike；
- 保存 checkpoint 后恢复时并行布局或 tokenizer/config 不匹配。

相关章节：

- [Checkpointing](checkpointing.md)
- [FlashAttention](flashattention.md)
- [Tensor Parallelism](tensor_parallelism.md)
- [Context Parallelism](context_parallelism.md)

## RL 视角

长上下文 RL 比 SFT 更难，因为训练系统外面又接了一套 rollout / inference / reward 系统。

关键问题：

- rollout prompt 和 response 都可能很长，KV cache 成本会成为瓶颈。
- vLLM / SGLang 等推理引擎的 batching、prefix cache、chunked prefill 会影响训练吞吐。
- policy update 后权重同步慢，会放大 staleness。
- verifier / reward model 如果也需要长上下文，奖励计算会成为独立瓶颈。
- 长 trajectory 的存储、回放、截断和去重会影响样本质量。

长上下文 RL 的核心不是单纯把模型训到 128k，而是让 **rollout producer、trainer consumer、reward/verifier、checkpoint/recovery** 在长序列下仍然能稳定协同。

[CompactionRL](../papers/compactionrl.md) 给出了一个很好的系统信号：长上下文 RL 可以不只依赖更长的 max context，而是训练 agent 在固定 context budget 下主动压缩历史状态。这样 long-context training 的问题就从“单条序列放不放得下”扩展成：

- 什么时候触发 compaction；
- summary 是否保留了 task-relevant state；
- compacted trajectory 如何保存和回放；
- summary segment 如何参与 RL loss；
- final reward 如何跨 compaction boundary 做 credit assignment。

因此，长上下文 RL 的基础设施不只需要 CP、KV cache 和 chunked prefill，还需要 compaction-aware trajectory schema 和 rollout observability。

相关章节：

- [Agentic RL Infrastructure](agentic_rl.md)
- [Rollout Latency Playbook](../playbooks/rollout_latency.md)
- [DeepSeek-R1](../tech_reports/deepseek_r1.md)

## 关键工程判断

- 如果只是少量样本超过长度上限，先评估是否过滤或单独 bucket，不要立刻把全局 max length 拉高。
- 如果每卡 token 超过 kernel 舒适区，优先考虑 CP 或重新 bucket，而不是盲目加 recompute。
- 如果 step time 抖动大，先看长度分布、packing shape 和 kernel autotune，再看 NCCL。
- 如果训练能跑但吞吐低，检查有效 token 比例、padding waste、cross-node CP 通信和 checkpoint spike。
- 如果 RL 长上下文慢，先拆 rollout latency：prefill、decode、weight sync、reward、trajectory IO，而不是只看 trainer MFU。

## 生产排障速查

| 症状 | 第一批检查 | 常见根因 | 验证修复 |
|---|---|---|---|
| CP 增大但显存不降 | 逐 rank tensor shape、logits/loss、THD/BSHD layout | postprocess full gather、FP32 loss upcast、CP 路径回退 | local shape 按 CP 缩放，loss/logprob 与 CP=1 对齐 |
| 不 OOM 但 step 很慢 | recompute scope、MBS/microbatch、kernel timeline | full recompute 过重、TP 切碎 GEMM、CP 跨慢链路 | `none/selective/full` 与 TP/CP sweep |
| step p99 很高 | 样本长度/packing、expert token histogram、rank skew | 长样本、padding、expert hotspot、A2A 拥塞 | bucket 后 p99 收敛且有效 token 不下降 |
| DataLoader worker 增大无收益 | CPU/RSS/page fault、queue depth、H2D overlap | CPU/内存已饱和、小文件/远端 IO、主线程 collate | data wait 与 GPU gap 同时下降 |
| 视频 SP 扩展性差 | Ulysses A2A、head shard、拓扑 mapping | heads 太少、跨节点 A2A、layout conversion | sweep Ulysses×Ring，比较 exposed A2A 与 attention time |
| MFU 与 step time 不闭合 | estimator、有效 token、统计窗口 | FLOPs 公式/packing/计时边界不同 | 统一计量后重新计算，不做口头补数 |

## 一手资料

- [Megatron-Core TransformerConfig：selective recompute 与融合配置](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.transformer.transformer_config.html)
- [Megatron-Core Fine-Grained Activation Offloading](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/fine_grained_activation_offloading.html)
- [HunyuanVideo：Unified Sequence Parallelism / xDiT](https://github.com/Tencent-Hunyuan/HunyuanVideo/blob/main/README.md)
- [NVIDIA MoE 报告中文分节 PDF 入口](../README.md#megatron-core-moe-2026-zh-pdf)

## 当前阅读路径

1. 先读 [Transformer](../papers/transformer.md)，理解为什么 attention/MLP 是训练系统的基本计算图。
2. 再读 [FlashAttention](../papers/flashattention.md)，理解长上下文为什么首先变成 IO 和 kernel 问题。
3. 接着读 [Tensor Parallelism](tensor_parallelism.md) 与 [Context Parallelism](context_parallelism.md)，理解序列和算子怎么切。
4. 然后读 [Checkpointing](checkpointing.md)，理解长 step time 下如何保存和恢复训练状态。
5. 最后读 [Agentic RL Infrastructure](agentic_rl.md)，把长上下文从 SFT 扩展到 rollout / RL 系统。
6. 再读 [CompactionRL](../papers/compactionrl.md)，理解 long-horizon agent 如何在固定 context budget 下训练可压缩的 trajectory。
7. [Agentic for Embodied](agentic_for_embodied.md) 把 long-horizon 扩展到视频历史、planner memory、动作反馈和物理环境状态。

## 待补实验

- 128k SFT 下不同 CP size 的 step time、MFU、显存和通信 profile。
- 长短样本混合时，packing 策略对 step time std 的影响。
- 关闭/开启 recompute、optimizer offload、activation offload 的吞吐与稳定性对比。
- RL rollout 中长 prompt prefill、long response decode、weight sync 的 latency breakdown。

## 我的总结

长上下文训练是一条系统主线，不是一个开关。Pretraining 关注 token 吞吐和稳定性，SFT 关注 packing、loss mask 和长尾样本，RL 关注 rollout、KV cache、reward 和 policy version。真正的工程判断来自端到端拆解：每卡 token、kernel 舒适区、通信拓扑、checkpoint 恢复、rollout latency 和有效 token 利用率必须一起看。

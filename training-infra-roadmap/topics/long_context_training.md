# Long-context Training

> 定位：这是地图型专题页，用来组织长上下文训练的跨主题关系。后续如果扩成完整工程手册章节，再补齐配置建议、排障流程、面试题和生产思考题。

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

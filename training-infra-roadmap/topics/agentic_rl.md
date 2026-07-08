# Agentic RL Infrastructure

## 这个主题解决什么问题

Agentic RL 不是“把 PPO/GRPO 换个任务继续跑”。它把训练系统从单一的 GPU batch training，变成一个持续生产、评估、消费 trajectory 的分布式系统。

在 pretraining 中，主要矛盾是 batch 是否喂得上 GPU、并行策略是否能稳定扩展、checkpoint 是否能恢复、NCCL 和存储是否拖慢 step time。

在 Agentic RL 中，新增的主要矛盾是：

- rollout 生成速度不稳定；
- 长上下文和多轮工具调用带来 tail latency；
- reward / verifier 可能成为串行瓶颈；
- policy training 等待样本，rollout worker 等待新权重；
- trajectory 有 policy version 和 freshness 问题；
- agent runtime、tool service、trainer、inference engine 之间的边界变复杂。

这意味着 Agentic RL Infra 的核心不是“更会调参”，而是设计一个能让 rollout、reward、training、weight sync、checkpoint 和 observability 长期稳定协同的系统。

## 为什么它会改变 Training Infra

传统 LLM 训练平台假设训练数据已经准备好，GPU 只需要持续消费 token。Agentic RL 打破了这个假设：训练数据是在训练过程中由当前或近似当前 policy 生成的。

这会带来三个结构性变化。

第一，训练系统从 batch synchronous 走向 producer-consumer。Rollout workers 是样本生产者，training workers 是样本消费者，reward/verifier 是中间加工环节。任何一个环节变慢，都会表现为 GPU 利用率下降或样本过旧。

第二，训练平台必须管理 policy version。样本不是普通 JSONL，而是带有生成模型版本、reward 版本、tokenizer 版本、tool 环境版本和 trace schema 的训练状态。样本越旧，RL 更新越可能偏离当前 policy。

第三，inference engine 成为训练系统的一部分。vLLM / SGLang / TensorRT-LLM 不再只是 serving 组件，而是 rollout workers 的核心执行引擎。KV cache、prefix cache、sampling 参数、token id 一致性都会影响训练正确性。

## 基本系统形态

一个最小 Agentic RL 训练系统通常包含：

```text
Prompt / Task Source
        ↓
Agent Runtime / Environment
        ↓
Rollout Inference Workers
        ↓
Trajectory Store
        ↓
Reward / Verifier Workers
        ↓
Training Workers
        ↓
Checkpoint / Policy Version
        ↓
Weight Sync / Inference Engine Update
```

工程上要避免把它理解成一个 for-loop。真实系统里这些模块通常异步运行，并且每个模块都有自己的并行策略、队列、失败模式和观测指标。

## Rollout 是新的系统瓶颈

Rollout 的难点在于延迟分布很差。Reasoning / tool-use / agentic task 的输出长度差异极大，某些样本会因为长链推理、工具超时、搜索失败或环境重试成为 straggler。

常见瓶颈包括：

- decode token/s 不够；
- KV cache 被长上下文撑爆；
- prefix cache 命中率低；
- batch 中最长 response 拖慢同步更新；
- tool call 或 browser/environment latency 不稳定；
- rollout queue 积压但 trainer 取不到可用样本；
- rollout 使用的 policy version 太旧。

AReaL 的核心价值就在这里：它把 rollout generation 和 policy training 解耦，用异步方式缓解“最长输出决定全局 step”的同步瓶颈。

## Reward / Verifier 是第二个瓶颈

Agentic RL 中 reward 不总是一个简单函数。它可能来自：

- rule-based checker；
- unit test；
- code execution sandbox；
- math verifier；
- judge model；
- retrieval / browser / environment feedback；
- human or synthetic preference model。

这些 reward source 的 latency、失败率、可复现性差异很大。生产环境里经常出现 rollout worker 很快，但 verifier backlog 越堆越高，最终 trainer 仍然没样本可训。

工程建议：

- reward/verifier 必须独立监控 queue depth、p50/p95/p99 latency、error rate；
- deterministic verifier 优先级高于 judge model；
- judge model 要记录版本，否则 reward drift 很难复盘；
- tool/environment reward 要设置 timeout、retry budget 和失败分类；
- verifier 输出要保留原始 trace，不能只保存 scalar reward。

## Context Compaction 进入 RL 训练

[CompactionRL](../papers/compactionrl.md) 提醒我们：long-horizon agent 的上下文管理不能只当成 inference-time prompt engineering。对于 coding agent、terminal agent 这类任务，trajectory 里包含错误日志、文件路径、失败命令、partial patch 和环境反馈；一旦 context 被压缩，summary 的质量会直接决定后续 action 是否还能继续有效探索。

工程上，compaction-aware RL 会改变 trajectory schema：

```text
execution segment
  ↓
summary segment
  ↓
reconstructed context = summary + recent turns
  ↓
execution segment
```

这意味着 rollout store 不能只保存 prompt/response/reward，还要保存 compaction trigger、summary tokens、segment boundary、full trace、compacted trace 和 policy version。trainer 也不能把每个 segment 当成独立 episode，否则会破坏 final reward 的 credit assignment。

CompactionRL 的核心判断是：summary generation 也是 policy action，应该和 task execution 一起接受最终任务 reward 的训练信号。这把“长上下文压缩”从推理优化问题推进到了 RL 训练系统问题。

## Training / Inference 双态模型

RL post-training 里同一个 actor model 会在两种状态之间切换：

- training state：FSDP / ZeRO / Megatron 分片，优化器状态完整，适合 backward/update；
- generation state：服务化推理布局，可能使用 vLLM / SGLang，适合 KV cache 和连续 decode。

verl / HybridFlow 的 3D-HybridEngine 之所以重要，是因为它把 actor model resharding 当成核心系统问题。权重从训练态同步到推理态，不只是拷贝参数，还涉及 dtype、并行布局、tokenizer、chat template、LoRA/adapter、推理引擎增量更新和 rollout 阻塞。

## 同步、半异步与全异步

### 同步 RL

流程简单：生成一批 rollout，算 reward，做一次 update，再生成下一批。优点是样本新鲜、算法直觉清楚；缺点是容易被最长 rollout 和最慢 verifier 拖住。

适合小模型、短上下文、快速 verifier 和初期复现实验。

### 半异步 RL

rollout、reward、training 有部分 overlap，但仍保留较强的 batch 边界或 policy version 边界。工程复杂度适中，适合多数从同步系统过渡的团队。

### 全异步 RL

rollout workers 持续生产，training workers 持续消费，系统通过 staleness bound、policy version 和 scheduler 控制稳定性。AReaL 是这条路线的代表。

适合 rollout latency tail 明显、GPU 利用率被同步 barrier 拖垮、且团队有能力建设完整 observability 和 freshness 控制的场景。

## 核心指标

Agentic RL 平台至少要监控这些指标：

- rollout token/s；
- rollout request/s；
- trajectory queue depth；
- reward/verifier queue depth；
- reward latency p50/p95/p99；
- policy update time；
- policy idle time；
- sample staleness；
- policy version lag；
- weight sync latency；
- rollout error rate；
- environment timeout rate；
- average response length 和 tail response length；
- effective training tokens/s；
- GPU utilization by role：rollout / verifier / training。

如果只看 GPU utilization，很容易误判。比如 rollout GPU 很忙，但 reward queue 堵住，trainer 仍然会 idle；或者 trainer 很忙，但用的是过旧样本，效果可能下降。

## 生产环境配置建议

- 初期优先选择同步或半异步，先把 correctness 跑通。
- 当 rollout p99 明显高于 p50，且 trainer 经常等待样本，再考虑异步化。
- rollout workers 和 training workers 最好分池管理，不要默认抢同一组 GPU。
- verifier 要独立扩缩容，不要嵌在 rollout worker 里变成隐藏瓶颈。
- trajectory store 要保存完整 metadata：policy version、reward version、tokenizer version、prompt version、tool/env version。
- weight sync 要有版本号和原子切换语义，避免部分 worker 使用半更新权重。
- 长上下文任务要尽早评估 KV cache、prefix cache 和 context truncation 策略。
- 评估集和训练 rollout 不要混用同一队列，避免在线训练把 eval cadence 拖乱。

## 常见故障

### GPU 忙但训练没进展

常见原因是 rollout 或 verifier 在忙，但 trainer 没有拿到足够可训练样本。先看 queue depth，再看 policy idle time。

### rollout latency 抖动

通常来自长输出 tail、tool timeout、KV cache pressure、batching 策略不合适或环境服务不稳定。详见 [Rollout Latency Playbook](../playbooks/rollout_latency.md)。

### reward drift

reward model、judge prompt、unit test、tool environment 任一变化都可能让 reward 不可比。必须记录 reward version。

### 样本过旧

异步系统中 rollout 使用旧 policy 生成，training 使用新 policy 更新。需要 staleness bound、importance ratio 或样本淘汰策略。

### tokenizer / token id 不一致

训练侧重新 tokenize rollout 文本，可能和推理侧采样 token 不一致，尤其在特殊 token、chat template、tool call schema 上容易出错。

### weight sync 失败

部分 inference workers 使用新权重，部分仍使用旧权重，导致同一批 trajectory 混入多个 policy version。必须让 weight update 可观测、可回滚。

## 与其他主题的关系

- [Distributed Training](distributed_training.md)：Agentic RL 把分布式训练扩展成训练、推理、reward、agent runtime 的复合系统。
- [Checkpointing](checkpointing.md)：checkpoint 不只保存 model/optimizer，还要保存 policy version、reward version、queue offset 和 rollout lineage。
- [FSDP](fsdp.md)：training state 常用 FSDP/ZeRO，和 inference state 的切换需要 resharding。
- [Tensor Parallelism](tensor_parallelism.md)：rollout inference 可能使用 TP，但跨节点 TP 会放大 decode latency。
- [Long-context Training](long_context_training.md)：长 prompt/response 会把 KV cache、chunked prefill、reward/verifier 成本和 policy staleness 一起带入 RL infra。
- [Context Parallelism](context_parallelism.md)：长上下文 trajectory 会推动 CP、KV cache 和 sequence 切分进入 RL 平台。
- [NCCL](nccl.md)：训练侧 collective 仍然重要，但 rollout/reward 系统还会引入更多 RPC 和存储流量。
- [DeepSeek-R1](../tech_reports/deepseek_r1.md)：reasoning RL 需求爆发的重要背景。

## 重点精读：来自 Historical Backfill

- [AReaL](https://arxiv.org/abs/2505.24298)：重点看 fully asynchronous RL、staleness、rollout/training worker balance。
- [HybridFlow / verl](https://arxiv.org/abs/2409.19256)：重点看 RLHF dataflow、hierarchical API、3D-HybridEngine、actor resharding。
- [Agent Lightning](https://arxiv.org/abs/2508.03680)：重点看 Training-Agent Disaggregation、trace schema、agent runtime integration。

## 前沿精读：来自 Frontier Scan

- [CompactionRL](../papers/compactionrl.md)：重点看 context compaction 如何进入 rollout collection、summary tokens 如何进入 RL objective、token-level loss normalization 和 cross-trajectory GAE 如何处理 compacted trajectory。

## Historical Backfill 发现的新关联

[Historical Backfill](../tracking/historical_backfill.md) 补充了几个不该混入 frontier scan、但对理解 Agentic RL Infra 很关键的历史材料：

- OpenRLHF 和 vLLM + OpenRLHF integration 说明 rollout inference、Ray placement group、vLLM engine、DeepSpeed ZeRO-3、weight sync 是一组系统问题。
- SkyRL 说明 long-horizon tool-use agent training 需要 environment、agent layer、training stack 和 evaluation 一起设计。
- DeepSpeed-Chat 是早期端到端 RLHF pipeline 的历史起点，适合用来理解 SFT/RM/RLHF 三阶段如何进入工程系统。
- Ray RLlib / Ray Train 提供了 actor/dataflow 调度背景，解释为什么许多 RLHF/Agentic RL 框架会建立在 Ray-style orchestration 上。
- NVIDIA NeMo RL 代表厂商训练栈开始把 GRPO、DAPO、reward environment、vLLM rollout、Megatron backend 统一进 post-training stack。

## 面试高频问题

1. 为什么 Agentic RL 不能简单复用 pretraining infra？
2. rollout latency 为什么会拖慢 policy update？
3. 同步 RL 和异步 RL 的核心 trade-off 是什么？
4. sample freshness 在 RL 训练里为什么重要？
5. training state 和 generation state 的模型布局有什么区别？
6. 为什么 actor resharding 会成为 RLHF 系统瓶颈？
7. reward/verifier 为什么要独立扩缩容？
8. 长上下文 trajectory 对 KV cache 和 checkpoint 有什么影响？
9. context compaction 为什么不能只当作 inference-time heuristic？
10. 如何判断 trainer idle 是 rollout 慢还是 reward 慢？
11. Agent runtime 和 RL trainer 解耦后，trace schema 应该记录什么？

## 生产环境思考题

1. 如果 rollout p99 是 p50 的 20 倍，同步 RL 会发生什么？
2. 如果异步 rollout 的样本太旧，如何限制 staleness？
3. 如果 reward model 更新了，历史 trajectory 是否还能复用？
4. 如果 inference workers 权重更新一半失败，如何避免污染训练样本？
5. 如果 verifier 依赖外部 tool service，如何设计 timeout 和 retry？
6. 如果训练侧用 FSDP，推理侧用 TP，weight sync 怎么做？
7. 如果 agent runtime 返回文本但不返回 token ids，会有哪些一致性风险？
8. 如果 rollout GPU 很忙但 trainer idle，第一步看什么指标？
9. 如果 policy update 很快但效果不涨，是否可能是样本质量或 freshness 问题？
10. 如果 compaction summary 丢掉关键错误日志，如何定位是 summary 失败还是 execution policy 失败？
11. 如果要支持多 agent task，trajectory storage schema 怎么设计？

## 我的总结

Agentic RL Infra 的关键转变是：训练平台开始承担在线数据生产系统的职责。过去我们优化的是单个 step 的计算效率；现在还要优化 trajectory 的生成、验证、排队、版本管理和消费效率。未来高级训练 infra 工程师需要同时理解训练并行、推理引擎、任务环境、队列调度和可观测性。这个方向值得长期跟踪。

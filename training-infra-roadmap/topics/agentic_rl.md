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

面试速答入口：[RL-ALGO-01｜PPO、GRPO、DAPO](../../private_resume/2026-08-llm-infra-interview-prep.md#rl-algo-01) · [VERL-04｜Fully Async、streaming、partial rollout 与 staleness](../../private_resume/2026-08-llm-infra-interview-prep.md#verl-04) · [RESUME-13｜CUDA Graph decode](../../private_resume/2026-08-llm-infra-interview-prep.md#resume-13) · [RESUME-19｜Gateway 调度收益](../../private_resume/2026-08-llm-infra-interview-prep.md#resume-19) · [AREAL-09｜Gateway 二次开发](../../private_resume/2026-08-llm-infra-interview-prep.md#areal-09) · [AREAL-10｜外部 Agent 接入](../../private_resume/2026-08-llm-infra-interview-prep.md#areal-10)

<a id="ppo-grpo-dapo"></a>
## PPO、GRPO、DAPO：先用最简单的话讲清楚

### PPO

> **PPO 用旧 policy 采样，再限制新 policy 一次不要改得太远。**

展开成训练链路：

1. Actor 用 behavior/old policy 生成 trajectory；
2. reward 与 Critic/value 估计共同产生 advantage，常见做法是 GAE；
3. 计算 `ratio = π_new(a|s) / π_old(a|s)`；
4. 用 clipped surrogate objective 限制 ratio，避免单次 update 过猛；
5. 对同一批样本做若干 epoch/minibatch 更新，同时训练 Critic。

LLM RLHF 中常加入 Reference model 的 KL penalty 约束策略不要偏离 base/SFT model，但 Reference/KL 是常见 RLHF recipe，不是 PPO 定义本身。面试时要把 old policy、Critic 和 Reference 三个角色分开。

### GRPO

> **GRPO 对同一个 prompt 采样一组答案，用组内相对好坏当 advantage，从而省掉 Critic。**

典型过程是：同一 prompt 生成 `G` 条 response，得到一组 reward；用组内均值/标准差标准化 reward，构造 relative advantage，再做 PPO 风格的 ratio clipping，并按实现选择 reference KL。核心收益是少维护一个和 Actor 同规模的 Critic，代价是每个 prompt 要采多条样本，组内 reward 没有区分度时学习信号会消失，且 normalization、loss aggregation 和长短样本权重会显著影响结果。

GRPO 是 PPO 的 group-relative 变体，不应说成“完全不用 old policy/logprob”，也不能说“没有 Critic 就没有 baseline”——组内统计量本身就是 baseline。

### DAPO

> **DAPO 是面向大规模 reasoning RL，把 GRPO 训练中容易塌掉的几个工程和目标函数细节系统修正的一套 recipe。**

DAPO 论文的四个关键点是：

- **Clip-Higher**：正负方向使用不对称 clip，上侧更宽，缓解低概率 token 难以被提升的问题；
- **Dynamic Sampling**：过滤组内 reward 全相同、没有 advantage 信号的 prompt，并继续采样补足有效 batch；
- **Token-level Policy Gradient Loss**：按有效 token 聚合，而不是先把每条 sequence 等权平均，避免长 response 的梯度被过度稀释；
- **Overlong Reward Shaping**：对接近/超过长度上限的 response 平滑惩罚，减少硬截断带来的噪声。

所以最稳妥的关系是：

```text
PPO：Critic/GAE + clipped policy update
  ↓ 去掉显式 Critic，改用同 prompt 组内相对 reward
GRPO
  ↓ 对 clip、有效采样、token loss、超长样本做系统修正
DAPO recipe
```

实际框架中的 GRPO/DAPO 可能在 KL 位置、loss aggregation、normalization、importance correction 上有变体。面试回答算法时，应把“论文定义”“框架实现”和“项目配置”分开。

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

<a id="cuda-graph-decode"></a>
## CUDA Graph：为什么 decode 会出现 6–8x 的局部收益

![CUDA Graph 将逐 kernel 提交变为静态执行图回放](../assets/topics/cuda-graph-decode.svg)

### 它优化的不是 FLOPs，而是提交路径

Autoregressive decode 每一步通常只处理少量新 token，但会重复执行相似的 model forward。小 batch 下，单个 kernel 很短，Python、runtime、driver 的逐 kernel launch 与同步可能占显著比例：

```text
Eager:
token n → CPU submit K1 → K2 → K3 ... → token n+1 再重复

CUDA Graph:
capture / instantiate once
token n → 更新静态 input buffer → graph replay
```

CUDA Graph 把 kernel、memcpy 和依赖关系作为一张可执行图预先实例化，之后低开销 replay。它不减少模型理论 FLOPs，也不是把所有 kernel 自动融合成一个 kernel；fusion 与 graph 可以叠加，但优化层次不同。

### continuous batching 和动态 KV 为什么还能用

推理引擎通常不是为任意 shape 捕获一张万能 graph，而是为若干 batch/sequence bucket 捕获多张图：

1. 为 bucket 预分配稳定地址的 token id、position、slot/KV block table 等 input buffer；
2. 每个 decode step 将新值 copy 到静态 buffer；
3. graph 中的 kernel 根据 metadata 访问 paged KV cache；
4. shape 和算子满足 bucket 就 replay，否则 fallback eager；
5. 记录 graph hit、fallback 和 padding waste，而不是只看“已开启 cudagraph”。

KV cache 的**内容**和 block mapping 可以变化，但 graph 捕获依赖的地址、shape 与控制结构必须满足实现契约。权重同步/refit 后，如果参数地址、module graph 或 kernel specialization 被破坏，需要重新 capture 或使用框架保证地址稳定。

### 显存与适用边界

Graph capture/private pool、多个 buckets 和 padding 可能额外占显存。GPU 已经被大 batch、大 GEMM 饱和时，host launch 占比很低，收益自然变小。动态 control flow、CPU callback、unsupported op 与频繁 graph break 也会侵蚀收益。

最新版简历中的主结果是 AReaL Qwen3.5-9B 128K Agentic RL **decode 6–8x**。另一个 verl 35B RLVR workload 是 **decode 约 14x**。两者模型、框架、batch/concurrency、graph coverage 和窗口不同，只能分开陈述；prefill、tool/sandbox、queue、reward、weight sync 和 trainer 都受 Amdahl 定律约束。

### 验证表

| 维度 | 固定/记录 |
|---|---|
| workload | model、gen-TP、batch/concurrency、prompt/response length、sampling |
| runtime | eager/graph、bucket、warmup、graph hit/fallback |
| local result | decode latency、inter-token latency、decode tokens/s、CPU launch gap |
| resource | GPU utilization、graph pool、KV cache、peak memory |
| end-to-end | rollout time、tool/env wait、trainer exposed wait；不把局部倍数直接外推 |

<a id="gateway-streaming-refill"></a>
## Gateway / Rollout 调度：分清供给、执行容量与训练接收

![Rollout 分层调度：session 容量、补位与 cohort 门禁](../assets/topics/gateway-streaming-refill.svg)

这是容量与回收的概念示意，不是代码部署拓扑。以下机制以 2026-09-09 核验的 `trail` 主线 `e9081cab` 为准，调用链和后续分支区别见[源码工程章节](#project-gateway-ownership)。

### 问题：GPU 有空闲，Trainer 为什么仍没数据？

一条 Agent episode 包含多次模型调用和工具等待；各 episode 长短不同，同一 prompt 的多个成员还必须凑成完整 cohort。因而有三种不同的等待：Agent 没准备好、session 没拿到执行容量、完整 cohort 还不能被训练器接收。只扩大 HTTP 并发，可能增加排队和过期样本，而不是有效训练数据。

### 三个预算，不能拿同一个 concurrency 解释

| 层次 | 任务单位与机制 | “完成”后发生什么 |
|---|---|---|
| 外部 Evals | reset / generation / step / agent-run；首个生成前才建物理 session | 多轮继续复用 session；episode 结束才 reward/end。仓内 runner 每个 epoch 仍 `await orch.run()` |
| Proxy 执行 | `C=rollout.max_concurrent_rollouts` 是总 active-session 上限，启动时精确拆给各 Proxy Worker | session 结束/清理释放 permit，唤醒当前等待成员；一次 HTTP/turn 完成不释放 |
| Trainer 接收 | 一个 dispatcher task 是完整 cohort 的接收预约；受并发和 staleness credit 双约束 | cohort 成功进入 accepted，仍占版本窗口；拒绝或版本推进才可能腾出/增加 credit |

设 `H=max_head_offpolicyness`、`B=consumer_batch_size`、`V=current_version`，online controller 的逻辑并发窗口 `L=(H+1)B`。`StalenessManager.get_capacity()` 的提交预算为：

```text
可新增 cohort 任务数 = min(
    L - running,
    (H + V + 1) × B - (accepted + running)
)
```

只有结果大于零且未暂停，dispatcher 才可继续提交。这里 `accepted` 是统计周期内的累计接受量，不是当前队列长度；成功任务从 running 转为 accepted，二者之和不变，所以“完成一个就必然再发一个”不成立。举例：`H=1、B=2、V=0、running=1、accepted=3` 时，物理执行即便空闲，新任务预算仍为零；版本推进后才可能继续。**物理 session 上限 C 不能除以 group_size，冒充这套 cohort 预算。**

代码可证明后台按容量持续调度、session 释放后触发放行；但外部 `evals` 包的 Orchestrator 实现不在这个仓库，不能据此证明“旧版完全固定 wave、新版全局消除 wave”。HTTP `stream=true` 的 SSE 转发更不能作为 episode 流式调度的证据。

### 均衡分发有两层，而不是一个 round-robin

Gateway 为新 cohort 轮询选择 owner Worker，此后 claim/session/生成/reward/end 都保持粘性。Worker 再为新 session 选 engine：优先避开 waiting，综合 running、waiting、active sessions，并对同 cohort 已用 engine 施加软惩罚。同一 session 后续固定 engine，有助于保留状态和复用前缀，但不保证一定命中 Prefix Cache。

各 Worker 都连接全部 inference engines，不过选择依据是本地 session 记录与定期抓取的 engine metrics，不是全局原子负载表。详细排序键和代价见[两层路由](#gateway-engine-routing)。

### 失败管理服务于有效供给，不是盲目提高重试次数

登记/启动可以按稳定身份重试；真正的模型生成在继承训练配置时设 `retry=0`，防止超时后重复生成。缺成员、缺 reward、stale、启动失败和整体超时要分桶。整组终结后清理 session/缓存并归还本地容量，但不能把本地清理等同于已确认远端 GPU 请求取消；更不能换 Worker 假装恢复一个丢失的 session。

### 结果如何解释

最新版简历记录：Rollout 阶段平均推理吞吐 `+60%`；Rejected Group `33.18%→2.73%`，绝对下降 `30.45pp`，同口径下相对下降约 `91.8%`。本次指定仓库中未找到这组数字对应的原始 A/B 日志和统计协议，因此保留为项目记录，不声称由本次代码审计复现。

机制的验收顺序：

- 容量：active sessions、等 permit 时间、空槽时间、pending depth；
- 均衡：每个 engine 的 running/waiting、队列偏斜、session affinity；
- 完整性：cohort ready latency、partial/overall timeout、missing reward、staleness；
- 有效性：exported/consumed/gradient-active trajectories、有效 token、trainer exposed wait、reward/eval。

没有独立消融就不拆分 60% 的贡献；拒绝比例的分母和原因分布未统一前，不把全部下降归因于长尾改善。团队主线、8 月异步分支、后续个人 quota 修复也不能强行归为同一次实验。

↩ [返回主文档 RESUME-19](../../private_resume/2026-08-llm-infra-interview-prep.md#resume-19)

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

<a id="async-streaming-partial-staleness"></a>
## Fully Async、streaming、partial rollout 与 staleness

这四个词经常被混用，但它们描述的是不同层次。

### Fully Async：执行关系

Fully Async 指 rollout producer 和 trainer consumer 不再以同一个 step barrier 串行推进：producer 持续写 queue/buffer，trainer 凑够可训练 batch 就更新，权重再按一定 cadence 发布给 rollout。它解决的是同步 phase bubble 和 trajectory 长尾阻塞。

它不保证单个 rollout 或 actor update 更快，也不自动保证 on-policy。异步越强，越要处理 queue、backpressure、version、safe retry、checkpoint 和恢复。

### Streaming：数据到达方式

在 RL Infra 语境里，streaming 至少有两种含义：

1. **token streaming**：OpenAI-compatible HTTP 请求逐 token/chunk 返回；它改善首 token 可见性和 Agent 交互，但不等于样本已可训练；
2. **sample/prompt streaming**：prompt、trajectory 或训练 batch 持续进入 data plane，不要求先生成一个完整 epoch/batch 文件。verl v0.9 release 中的 streaming dataloader 属于这一类。

面试时必须先说明所指对象。`stream=True` 只是 API 返回方式，不能据此宣称训练已经 Fully Async。

### Partial rollout：trajectory 生命周期

Partial rollout 指一条未完成 trajectory 可以在调度/权重更新边界被暂停、保存状态并继续，而不是为了等它而阻塞全局，也不是每次更新都丢弃整条长 episode。

需要保存：

- environment/session state；
- messages/token IDs、tool outputs 和 RNG/sampling metadata；
- segment boundary；
- 每段或每 token 的 behavior policy version/logprob；
- reward 是否 terminal、是否可跨 segment 回传。

它提升长 trajectory 利用率，但同一 trajectory 可能跨 policy version，credit assignment 和 correction 会更复杂。它不是“把字符串截成两段继续生成”这么简单。

### Staleness：算法距离的系统代理

最常见的离散指标是：

```text
version lag = current trainer version - behavior rollout version
```

它容易观测，但只是 policy divergence 的代理：同样落后 1 个 version，不同 learning rate、update size 和 token 状态上的 KL 可能完全不同。更完整的诊断还包括 trajectory age、importance ratio、behavior/current logprob gap 和 KL 分布。

处理策略通常是：

- `wait`：producer 等新权重或 trainer 等新鲜样本；
- `drop/reject`：丢弃超过阈值的 group/trajectory/token；
- `mask`：只让满足约束的 token 进入 loss；
- `correct/reweight`：用 importance sampling、rollout correction 或 decoupled loss；
- 调整 weight-sync cadence、producer/consumer 资源比和 queue capacity。

### 四者如何连起来

```text
Fully Async 去掉全局 barrier
  → sample streaming 持续供给 trainer
  → 长 episode 需要 partial rollout 才不被频繁丢弃
  → trajectory/token 跨版本，形成 staleness
  → wait/drop/mask/correct + weight sync 控制吞吐—效果前沿
```

不能只开一个 `max_staleness` 配置就宣布正确。必须同时看 effective-token goodput、version-lag 分布、stale rejection、importance ratio/KL、训练 reward 和 held-out eval。

<a id="meituan-fully-async-practice"></a>
## 美团 verl Fully Async 实践：把等待移出训练关键路径

**来源与阅读范围**：侯正罡，美团搜推 AI Infra 团队，《基于 verl 的 Fully Async Policy 训练架构》，2026 年 1 月，收录于 [2026-01-10 官方 meetup 目录](https://github.com/verl-project/verl-data/tree/main/verl_meetup_20260110)。依据[原始 PDF](https://github.com/verl-project/verl-data/blob/main/verl_meetup_20260110/4-%E4%BE%AF%E6%AD%A3%E7%BD%A1.pdf)第 5–20、24–28 页；2026-09-08 完成阅读与页面核对，状态 **DIGESTED，未复现实验**。这是历史实现案例，不代表当前 verl 所有路径的默认行为。

面试先读[主文档 Fully Async 专题：原图、配置与实验](../../private_resume/2026-08-llm-infra-interview-prep.md#fully-async-study)；这里补工程推理，不另起学习长文。可按问题直达：[四组件/四模式](../../private_resume/2026-08-llm-infra-interview-prep.md#verl-04) → [流式组批](../../private_resume/2026-08-llm-infra-interview-prep.md#verl-12) → [预算调优](../../private_resume/2026-08-llm-infra-interview-prep.md#verl-13) → [Partial/校正](../../private_resume/2026-08-llm-infra-interview-prep.md#verl-14) → [实验表](../../private_resume/2026-08-llm-infra-interview-prep.md#verl-15)。材料索引见[历史补录](../tracking/backfill/2026-01.md#meituan-fully-async)。

### 1. 为什么分池还不够：有三种不同的等待

1. **阶段等待**：同步模式先完成 rollout 再 update。分离 Trainer/Rollouter 资源，让两侧并行推进，减少阶段间空闲；代价是每侧可用卡数减少，因此并行度和资源比例需要重新调优。
2. **批次等待**：One Step Off Policy 可以重叠两轮工作，但固定批次仍会被最长轨迹拖住。改成完成即入队、凑够训练批量就消费，让短样本不再陪同一大批中的长样本一起等。单样本是调度/传输粒度，不意味着 batch size=1；GRPO 计算组内 advantage 时仍需完整 group，算好后可重新 packing 或切分训练 microbatch。
3. **权重发布等待**：即使持续生成，更新权重前若必须等所有在途请求自然结束，长尾仍会阻塞。Partial rollout 把在途任务暂停并保存已生成 token、behavior logprob，发布权重后续跑，减少 drain 等待和重复 decode；NCCL bucket 传输则缩短真正的数据传输阶段。**前者少等请求，后者快传权重，不是同一个优化。**

按这一机制做工程抽象：同步耗时近似是生成、训练与同步的串行和；充分流水后的稳态周期由较慢一侧及尚未隐藏的同步开销决定。这个近似必须在固定工作量、重新测量分池后的阶段耗时后使用，不能把原来全卡 rollout 和全卡 training 的耗时直接取 `max` 当预测结果。

### 2. staleness 在这份实现里是“超前生产额度”

分享第 15 页用样本预算控制异步程度。令 `B` 为一个权重发布周期内 Trainer 计划消费的 prompt/group 数，`C` 为本周期已计入预算的数量，`s` 为此处的 staleness 配置。分享用结转旧样本解释 `C`；补证的 v0.7.1 实现在同步时以队列与在途任务初始化计数，之后每提交一组就递增，Trainer 消费不会立即归还额度，直到下次同步再重设。因此不能把它理解成实时 `queue + active` 容量池：

```text
可用新增额度 = max(0, floor((1 + s) × B) − C)
例如 B=512、s=0.5、C=128，则本轮还可提交 640 个 prompt/group。
```

这是“最多能生产多少”的背压预算，不代表每轮必须生产满。`s=0.5` **不是落后 0.5 个 policy version，也不保证实际训练 batch 恰有 50% 的旧样本**。队列水位、样本年龄和逐 token policy lag 仍需独立观测。

[v0.7.1 recipe 文档](https://github.com/volcengine/verl/blob/v0.7.1/docs/advance/fully_async.md#parameter-description)将 `B` 具体写为 `trigger_parameter_sync_step × require_batches × ppo_mini_batch_size`；`require_batches` 控制一次取样量，trigger 计的是取样/训练循环，不一定是 optimizer.step 次数。这里计数单位是 prompt/group，未过滤前的 response 数再乘 `rollout.n`，不能把全局 batch 再乘 GPU 数。这是固定版本的参数补证，不将其默认值或快照实现倒推为 1 月实验的精确代码。

`s=0` 只意味着不借助这项额度跨发布周期超前生产，**不自动等价于整个系统严格 on-policy**：若一次发布之间 Trainer 连续更新多次，后面的 update 仍会消费此前参数生成的数据。该实现还要求 `s>0` 才让 partial rollout 实际生效。

### 3. Partial rollout 要保存什么，不能省掉什么

- **原始行为数据**：旧前缀由旧参数生成，续写由新参数生成，训练时必须保留各段真实的 token/logprob 对应关系。不能用新权重重算整条 logprob 后，把它冒充生成时的行为概率。
- **Agent 状态**：第 20 页要求在工具处理的安全边界暂停，保留轮次、工具指令与结果、对话上下文和多轮片段。不能只存字符串，导致恢复后重复执行有副作用的工具。
- **KV 边界**：复用 token 前缀不等于旧 KV cache 跨权重仍有效；权重变化后通常需要重新 prefill。节省的是从头自回归生成的成本，不是宣称恢复零成本。
- **校正边界**：第 18–19 页区分直接采用 rollout logprob 和 Decoupled PPO 式校正。后者把“训练策略相对近端策略的更新”与“近端策略相对实际采样策略的偏差”拆开处理；近端策略不是用于 KL penalty 的 Reference model。Clipping/correction 都不能修复错误的 mask、group 或行为数据，也不能保证任意陈旧度下收敛。

后两项中的 KV 有效性与数据契约检查是对分享机制的工程推论，不是分享披露的额外性能收益。

### 4. 怎么读效果表，避免把几种收益混起来

主文档已列总耗时，此处保留决定工程判断的对照：

| 分享中的证据 | 应得出的结论 |
|---|---|
| 128 卡 7B，400-step：同步 `40h48m`；stream off-policy `25h53m`；结合 staleness 与 partial rollout 后 `17h22m`（第 26 页） | 流式组批先减少大批次等待，进一步异步与轨迹续跑再减等待；后一步是联合配置，**不是 partial rollout 单因素消融**。 |
| 同组实验 acc/mean@1：同步 `max 0.3573 / last 0.2958`；最终 async `max 0.3521 / last 0.3094`（第 24 页） | max 略低而 last 较高，只能描述这次报告的结果，不能据此断言所有任务等精度或训练更稳定。 |
| staleness `0.3 / 0.5`：400-step 分别 `17h20m / 17h22m`，last acc `0.2865 / 0.3094`（第 27 页） | 本次从 0.3 增至 0.5 未观察到累计耗时继续下降；不能只看最小等待就继续加大 staleness，也不能据单次结果确定通用最佳值。 |
| 30B-A3B：同步 actor update `86.27s`，分离后 `206.63s`，400-step 总耗时仍从 `59h39m` 降到 `34h41m`（第 28 页） | 单个训练阶段变慢与端到端变快可以同时成立，符合分池后通过 overlap 减少等待的机制。不要直接相加异步计时项。 |
| 多轮工具：200-step `22h28m → 14h04m`；AIME 2025 acc/mean@30 的 last 为 `0.2056 / 0.2044`（第 28 页） | 此次约 1.60x，末点评分接近但并非完全相同；不能扩写成所有 Agent 任务都等效果加速。 |
| 参数同步耗时降低 60% 以上（第 13 页） | 是 bucket 化 NCCL 传输的阶段收益，不是端到端加速，也不能与上面倍数相乘。 |

**原源差异与统计边界**：PDF 将第一组简称为“Qwen2 7B Math”，[后续官方实验说明](https://verl.readthedocs.io/en/latest/advance/fully_async.html#experiments)写 `Qwen2.5-Math-7B`；因此主文档称“7B Math”，不把命名差异悄悄抹平。同一官方说明明确 30B-A3B 的 `96:32` 为 **Rollout:Trainer**，不是本人的 `3T+1R`。表中 `gen` 下降不能当作完整 Rollouter 推理时间下降；异步消费会隐藏后台工作，必须查看对应版本的计时范围。各模式的 `step` 还需按样本消费量/更新量对齐，累计耗时不是等质量 time-to-target，也不能换算成本人项目的 `tokens/s/GPU`。

### 5. 放到自己项目里，按什么顺序优化

1. **先测等待在哪里**：同步阶段占比、请求长度分布、两侧 idle、queue 水位与权重发布耗时，先区分供给不足、长尾、训练慢和同步慢。
2. **再配资源与批量**：联调 T:R、gen-TP、实例数与训练取样批量；减小批量会减少等待，但也可能改变数据顺序和训练效果，不能只追求 `require_batches=1`。
3. **再减少发布边界的停顿**：验证 partial rollout 的暂停/续跑正确性，分别测 drain 与 bucket 传输耗时，避免把两个收益合并归因。
4. **最后找吞吐与效果都可接受的配置**：逐步调 staleness 与发布频率，同时看有效 token goodput、response length、ratio/KL、拒绝率和 held-out eval。分享第 30 页的动态资源分配、token 级路由属于当时的后续规划，不能说成当时已实现。

这份材料解释的是可借鉴的机制。个人项目仍使用[RESUME-02 已有配置与指标](../../private_resume/2026-08-llm-infra-interview-prep.md#resume-02)，没有独立 A/B 的改动不拆贡献，没有复现的美团指标不写入个人成果。

<a id="external-agent-gateway"></a>
## 外部 Agent 如何通过 OpenAI-compatible Gateway 接入

### 基础协议

外部 Agent 不需要 import AReaL 内部 engine，只要能把模型调用指向 Gateway：

```text
Admin key
  POST /rl/start_session
      ↓ 返回 session_id + session API key
Session key
  POST /chat/completions | /responses | /v1/messages
      ↓ 可多轮调用 Tool/Sandbox
  POST /rl/set_reward
  POST /rl/end_session
```

两级 key 的职责不同：admin key 保护 session 创建/管理，session key 只允许访问该 session 的推理、reward 和结束接口。这样第三方 OpenAI/Anthropic-compatible Agent framework 只需替换 `base_url` 和 `api_key`，多轮控制流、Tool/Sandbox 状态仍留在框架外部。

这里列的是项目分支 `server.py` 的实际 route；OpenAI SDK 会在传入的 `base_url` 后追加 `chat/completions` 或 `responses`。如果外层 ingress 统一增加 `/v1` 前缀，应以部署路由为准，不能把 SDK 习惯路径直接写成项目服务端事实。

### 项目 online proxy/cohort 数据流

```text
External Evals / Agent
  → Gateway：cohort / session 粘性 HTTP 路由
  → Proxy Worker：鉴权、session 容量准入、engine 选择
      └→ worker-local CohortManager：reservation / rank / ready / staleness
  → vLLM/SGLang：生成 token/logprob
  → InteractionCache：interaction、token、behavior logp、version、reward
  → successful + rewarded + ended + complete + fresh cohort
  → workflow export/校验/ACK → executor tensorize/过滤 → loss/update
```

在该主线的 online 模式下，Trainer 先建立接收 reservation；bridge 在 reset sandbox 前登记逻辑 cohort member，首个生成请求才 `ensure_started()`，后续多轮复用 session key。上述简图中的 session 接口并不意味着 online 可以绕过 rank/claim 直接创建任意训练样本。

API compatibility 只解决“Agent 会不会调用”，不解决“轨迹能不能正确训练”。以下是完整系统需要守住的契约，其中 exact domain quota、权威 reward identity 和恢复公平性属于[后续个人分支](#gateway-personal-followups)，不要当成该 HEAD 已全部实现：

- session/interaction identity 不串；
- terminal reward 写到权威 completion；
- 重试不会重复创建或把一个 episode 路由到另一 cohort；
- cohort/group 完整性和 domain 配额正确；
- behavior logprob、token/version lineage 完整；
- update/checkpoint/recovery 后 queue 和 fairness state 一致。

<a id="project-gateway-ownership"></a>
## 项目 Gateway 二次开发：逻辑到底改了什么

阅读顺序：[版本与署名](#gateway-code-versions) → [调用链](#gateway-call-chain) → [两层选路](#gateway-engine-routing) → [失败与门禁](#gateway-lifecycle) → [配置与排障](#gateway-config-troubleshooting)。面试先讲[主文档 AREAL-09](../../private_resume/2026-08-llm-infra-interview-prep.md#areal-09)，被追问再看这里。

<a id="gateway-code-versions"></a>
### 1. 先锁定版本与署名，避免讲出一套不存在的架构

2026-09-09 静态核验代码阅读仓 `trail` 的 `main`：`e9081cabba3fcd43c0cce57faf3bdd9539250672`。下文未另作说明时，均指这个快照；没有部署服务或复现 GPU benchmark，也未以本地远程跟踪引用代替一次最新远端 fetch。

| 阶段 | 可核对的代码证据 | 架构与面试边界 |
|---|---|---|
| 7 月团队主线，已合入该 HEAD | `c9fa6925`，2026-07-24，`feat(online): refactor gateway scheduling (#34)`；作者 suran662，包含 wangxy74 的共同作者记录 | 改造前由 controller 持有 CohortManager；改后组生命周期与 session 准入下沉 Worker，Gateway 保留粘性路由。没有移除训练 controller |
| wangxy 的原分支工作 | `217b699b`（2026-07-20）、`b22da528`（2026-07-21），作者 wangxy74 | 原 SHA 非 HEAD 祖先；前者的 CohortManager 与 squash 提交内容一致，支持“工作被整合”，不说原提交逐笔合入 |
| 8 月异步演进，本地存在但未合入该 HEAD | `7f3f063f`、`026346ef`；分支顶端 `f1a5867a`，2026-08-19，作者 wangxy74 | **重新集中** Gateway 的 cohort assembly/admission，外部提供稳定身份，Trainer 改为 take terminal cohort；不是上面 worker-local 架构的另一种叫法 |
| 后续个人项目分支 | 在此前项目工作副本可核对 `10a3e264`、`9979a0f6`、`c83de5fa`、`21bb4862`，作者 zbw-ai；这些对象不在本次指定仓库 | exact quota、reward identity、安全重排等另列[历史分支说明](#gateway-personal-followups)，不当成当前 HEAD 的行为或上述收益的独立证明 |

因此可以说“团队完成了调度重构，wangxy 有明确代码贡献，我参与训练链路集成与优化”。不能仅凭 squash author、共同署名或代码量断定某人独立完成全部工作；个人设计、实现、验证和跨团队推进应分别举证。

**8 月分支具体改变了什么？** `7f3f063f/026346ef` 将稳定的 run/cohort/rank/session 身份与组装、准入集中到 Gateway；workflow 从 `reserve → wait_ready` 改为 `_take_cohort → _run_online_cohort`，消费外部已形成的终态 cohort。同时补充 receipt/batch 等生命周期观测。该分支的 take 标注为一次 collection 的 at-most-once，不是端到端 exactly-once。HEAD 已有 SessionTracer、engine metrics 和 cohort stats，不能说直到 8 月才有可观测性。

### 2. 改造的动机：把状态与执行放在能闭环的位置

Agent 多轮调用意味着一次模型响应结束，episode 仍可能去调用工具再回来；GRPO 又要求多个 episode 组成一组。原来的 controller 集中组状态、Worker 持有实际 session/轨迹，两侧需要反复协调。本次主线把组管理与执行准入下沉 Worker，让登记、启动、结束、奖励、导出和清理在同一 owner 上闭环。

| 组件 | 负责什么 | 不负责什么 |
|---|---|---|
| Gateway | 新 cohort 选 Worker；保存 cohort/reservation/claim/session 路由；HTTP/SSE 转发；聚合控制状态 | 不做 vLLM token batching；不持有当前 HEAD 的 CohortManager；不统一计算全局 session 优先级 |
| Proxy Worker | session permit、engine 选路、API 鉴权、InteractionCache、reward/end/export | 不是 Trainer optimizer；不把一次 chat 响应当成整个 group |
| worker-local CohortManager | 逻辑预约、成员身份、幂等、完整组、ready-time 版本门禁、导出状态 | reservation 本身不占物理执行 permit；不保证进程重启后持久化 exactly-once |
| Trainer controller/workflow/executor | 预提交接收任务、版本预算、export/校验/ACK，再做 tensorize/过滤；训练与权重更新协调 | 不直接执行外部 Agent 的工具控制流 |

<a id="gateway-call-chain"></a>
### 3. 一条 episode 怎样从外部 Agent 变成训练数据？

```text
Trainer dispatcher ── reserve 接收预约 ────────────┐
                                                   ↓
External Evals / Agent → Gateway → Proxy Worker + CohortManager
  ① register member                  claim：task/cohort/rank
  ② reset sandbox                    逻辑登记不占 session permit
  ③ 首个 generate → start_session    等 permit → 选 engine → session key
  ④ 多轮 chat / tool                 session 固定 Worker + engine
  ⑤ reward / end_session             释放物理 permit；保留可导出轨迹
                                                   ↓
                  完整 + successful + rewarded + ended + fresh
                                                   ↓
Trainer workflow ← wait_ready → export → 组/interaction 校验 → ACK
                                                   ↓
                           executor tensorize / 过滤 → 训练 update
```

关键实现：

- **先逻辑登记，后物理启动**：`ArealEnv.reset()` 先登记 member，再创建/reset sandbox；`ArealLLMClient.generate()` 首次调用 `ensure_started()` 才申请 session。多个 turn 共享同一 session，不重复占多个 permit。
- **两套容量解耦**：`CohortManager.reserve()` 建立 Trainer 接收身份，不占执行容量，也没有另一个独立的 reservation-capacity 旋钮；总 active-session 上限由 controller 拆给各 Worker。cohort 的提交窗口仍受[版本预算](#gateway-streaming-refill)约束。
- **有容量才启动**：`_CohortAwareSessionLimiter` 的 waiter 使用 Future，优先级是 `(reservation_seq, cohort_rank, ticket_seq)`；`release()` 通过 `call_soon(_drain)` 唤醒等待者。它只在当前已就绪成员中选最早者，不预留整组物理槽，也不因最早组成员尚未到达而阻塞所有后续组。
- **避免重复启动和泄漏**：`start_session()` 先解析幂等重放，再等待 permit；在短启动锁内再次核对 claim 后分配 engine/session。并发重试若发现 session 已存在，会释放多拿的 permit；取消也区分“未授权”与“已授权”两种回收路径。
- **执行资源与训练数据分开释放**：`end_session()` 归还物理 permit，但导出所需 cache 继续保留。workflow 完成组/interaction 校验并 ACK 后回收正常导出的缓存；executor 随后才做完整 tensor 整理与过滤。trajectory 模式的组 advantage 计算会在 ACK 前使用部分 tensor 转换，但不能因此说所有 tensorization 都在 ACK 前成功。ACK 后若下游失败，也不能假设 Proxy 仍保留数据可重导。

示意例子：若某 Worker 容量为 2、group_size 为 4，同一 cohort 的成员可以分批执行，两个 session 结束后再让另外两个进入。不能把这个策略描述成“四个成员必须同时获得四个 permit”的 gang scheduling。

<a id="gateway-engine-routing"></a>
### 4. 两层路由：为什么既有 RR，又有负载感知？

**第一层，Gateway 选 owner Worker。** `_assign_route()` 只对未绑定身份做 round-robin；`_resolve_route()` 用 cohort/reservation/claim/session 查 owner，身份指向不一致返回 409。cohort identity 使用 `task_id + cohort_key` 组合，后续 session key 也绑定原 Worker。这里的粘性保护 session/InteractionCache 状态，不是随时迁移的全局最小负载调度。

**第二层，Worker 为新 session 选 engine。** controller 给每个 Proxy Worker 建立到所有 inference engines 的客户端，不是一个 Worker 只能调用本机 engine。`_select_engine_for_session_locked()` 的评分逻辑是：

```text
effective_load = max(该 Worker 对当前候选 engine 记录的 active_sessions,
                     engine 的 vLLM running + waiting)
若 metrics 缺失：effective_load = active_sessions

按以下元组从小到大选：
(waiting > 0,
 waiting,
 effective_load + 同 reservation 已绑定该 engine 的 session 数,
 同 reservation 是否已使用该 engine,
 effective_load)
完全同分时再 round-robin。
```

先绕开已经出现 waiting 的 engine，再考虑负载和同组软反亲和。软反亲和让同组成员不轻易都堵在同一个 engine 上，但并非硬禁止共用 engine。选定后保存 `_session_to_engine_name`，多轮固定 engine；这有助于缓存复用，但不能替代 cache key/权重版本校验。

代价与边界：metrics 约每秒刷新，网络读取在锁外；本地 session 计数不覆盖其他 Worker 的全部在途状态，指标缺失会退化。同一时间各 Worker 可能看到相似旧负载并同时选择一个 engine，所以这不是全局原子均衡；请求数也不等于剩余 token 计算量。异构算力、长短请求混排要看 waiting、token 长度和实际队列偏斜，不能只看平均 GPU utilization。

### 5. 控制面不能被长请求挤死

Gateway 在 lifespan 中建立 **data 与 lifecycle 两个 aiohttp 连接池**：生成和 register/start 走数据侧，reward/end/abort/control 走生命周期侧。这样即使生成或登记长时间占连接，也不会直接占满同一个连接池、阻止 end 回收容量。它解决的是连接资源隔离，不保证网络/进程永不故障。

`/rl/control/state` 对并发查询复用一个 in-flight 聚合任务，向 Workers 获取状态；任何非 open 状态会阻挡新的 bridge 准入。Worker 的 `claim_member_when_available()` 可在 condition 上等接收 credit，等待时释放状态锁；短启动锁只保护最终分配。**当前主线允许内部 long-poll，与后续个人 domain-quota 分支“不在 domain lock 内长等”并不矛盾。**

<a id="gateway-lifecycle"></a>
### 6. 失败、版本和导出：具体守住哪些门槛？

| 场景 | 当前代码行为 | 不能扩大成什么承诺 |
|---|---|---|
| register/start 超时或暂时无 credit | bridge 复用稳定身份等待/重试；Worker 对 claim/start 做幂等检查 | 不是所有 HTTP 都能无限重试；原 session/cache 丢失不能透明复原 |
| 生成超时 | 继承训练配置时设 eval generation `retry=0`，避免旧请求尚未 abort 就重发；失败进入显式终结路径 | 控制请求可重试，不代表生成可随机重投；Gateway 的 502 不会触发跨 Worker failover |
| session 已结束但 reward 未到 | 保留 grace，等待奖励；超出 grace 则拒绝组 | end 成功不等于 group 已可训练 |
| cohort 成员没收齐 | `partial_cohort_deadline` 限制逻辑 claims 到齐时间；全部 claim 齐后不再因该 deadline 拒绝 | 它不是所有 session 的推理超时；仍有总 reservation deadline |
| 等待队列中的空 reservation 老化 | 默认版本在首次真实 claim 时绑定；ready 时再查 `current_version - rollout_version ≤ H` | 组级门禁不证明每个 turn/token 都来自同一权重；真实输出版本另写入 token tensor |
| 整组 reject/cancel | 终结 reservation，回收本地 orphan session/cache/key/permit，记录原因 | 回收函数未显式证明远端 engine abort；也不自动发一个 replacement credit |
| ready → export | 必须成员齐、全成功结束、全有 reward、通过新鲜度检查；workflow 再查空导出、重复 interaction ID、组大小和 final reward | 不把空组或缺 rank 的 partial group 当正常 GRPO batch |
| export ACK | 组/interaction 校验成功后确认导出、标记 exported 并回收缓存；executor 后续仍会 tensorize/过滤 | 不保证后续过滤通过，不是 optimizer 已提交，更不是跨进程故障的 durable exactly-once training |

路由表、CohortManager 和 export cache 是内存状态；Gateway 路由表还有容量上限和 LRU 淘汰。幂等依赖身份及相关状态仍可解析，不应承诺任意重启、任意延迟重试都安全恢复。需要持久恢复时应另设计状态落盘、恢复协议和端到端提交边界，不能靠给 HTTP 增加 retry 解决。

**权重更新是否还有 barrier？** 有。`H=0` 的 strict online 路径先暂停新准入、drain 物理 session、再暂停 rollout 和更新；`H>0` 允许在途 session 与 actor 更新重叠，但权重发布仍有协调屏障。当前 strict drain 超时只 warning 后继续，因此只能说“实现了 drain 协调”，不能说“必定排空后才更新”。`pause_for_weight_update()` 与取消 pending 的普通 pause 也不同，不能混用。

<a id="gateway-config-troubleshooting"></a>
### 7. 配置怎么调，遇到慢和拒绝怎么查？

| 配置/变量 | 单位与作用 | 调整时看什么 |
|---|---|---|
| `rollout.max_concurrent_rollouts` | 总 active sessions；controller 用商和余数精确分给各 Proxy，且总数须至少覆盖 Proxy 数 | session 占槽时间、engine waiting、KV 显存；不是每加一个 Worker 就把总容量再乘一遍 |
| `consumer_batch_size`、`gconfig.n_samples` | 前者在此 online 接收路径控制每批 cohort 数，后者为每组成员数 | 组完成时间、可训练 batch、长尾；不把 cohort 数与 HTTP 调用数混用 |
| `max_head_offpolicyness` | cohort 版本窗口 H，影响逻辑并发与 ready 门禁 | actor 消费速度、ready-time stale、有效样本；并非越大越快越好 |
| `partial_cohort_deadline_seconds` | 同 cohort 逻辑成员到齐时限 | reset/登记并发、成员到达分散；不是用延长它掩盖丢 rank |
| `cohort_timeout_seconds`、`reward_grace_seconds` | reservation 总寿命与 ended 后等 reward 时间 | 慢 episode、工具失败、reward 回传；需分原因调节 |
| `queue_size` / pending limit | dispatcher 输入预取与背压 | 排队年龄、credit 用量、内存；不能无限预取 |
| eval generation `retry` | 正式配置继承路径强制为 0；admission/control 重试独立 | 重复生成、失败清理、身份复用；绕开继承流程需另查实际配置 |

排查顺序：

1. **GPU 空闲**：先看外部 reset/tool 是否有任务可供给，再看 Worker 是否有 permit、Trainer credit 是否用完、admission 是否 paused。不能先认定 engine 调度有问题。
2. **部分 engine 排队**：对照各 engine running/waiting、metrics 新鲜度、session 固定路由和同组分布；只看 Gateway RR 无法解释第二层拥塞。
3. **Rejected Group 高**：拆 partial timeout、reservation timeout、missing reward、stale、backend failure。先定位产生阶段，再改并发/deadline，不以放宽完整性门禁降比例。
4. **结束后仍占槽**：追 register→start→reward/end→reject/ACK；检查重复释放保护、cancel 清理和 lifecycle 池。区分本地 active session 已清理与远端生成真正停止。
5. **吞吐好看但训练更慢**：对照 exported→consumed→gradient-active 数据、有效 token、update interval 和效果；指标口径见[RESUME-19](../../private_resume/2026-08-llm-infra-interview-prep.md#resume-19)。

<a id="gateway-source-index"></a>
### 8. 源码定位索引

路径相对 `trail` 根目录；行号固定于 `e9081cab`，后续版本以函数名定位。`proxy/` 是 `areal/experimental/openai/proxy/`，`bridge/` 是 `third_party/areal_evals_bridge/src/areal_evals_bridge/`。不把私有代码全文、运行端点或凭证复制到公开文档。

| 路径：起始行 / 函数 | 核对内容 |
|---|---|
| `areal/infra/controller/rollout_controller.py:59` `_split_proxy_capacity`；`:321`；`:570` `_async_start_proxy_impl` | 总容量拆分、cohort 逻辑窗口、每 Proxy 连接全部 engines |
| `proxy/proxy_gateway.py:150` `create_proxy_gateway_app`；`:186` `_resolve_route`；`:213` `_assign_route` | 四类身份路由、冲突与 RR |
| `proxy/proxy_gateway.py:225` lifespan；`:262` `_forward`；`:359` control state | 双连接池、SSE/失败转发、控制查询合并 |
| `proxy/proxy_rollout_server.py:142` `_CohortAwareSessionLimiter`；`:388` metrics；`:456` engine selection | session 排队、指标更新、负载和软反亲和 |
| `proxy/proxy_rollout_server.py:1182` register；`:1236` start；`:1470` end；`:1005` orphan cleanup | 逻辑/物理分离、幂等、释放与清理 |
| `proxy/cohort_manager.py:335` reserve；`:438` claim；`:671` condition wait | 接收预约、成员绑定、首 claim 版本、内部等待 |
| `proxy/cohort_manager.py:1354` ready；`:1412` reject；`:1448` expiry | 完整性、版本、partial/overall/reward 超时 |
| `proxy/workflow.py:524` `_run_online_cohort`；`proxy/proxy_rollout_server.py:1932` export；`areal/infra/workflow_executor.py:1085` | 等待、导出、组校验与 ACK；其后的 tensor 整理和过滤 |
| `areal/infra/staleness_manager.py:99` capacity；`areal/infra/workflow_executor.py:357` producer | 双预算、完成唤醒不等于无条件提交 |
| `bridge/env_wrapper.py:172` reset；`bridge/llm_client.py:126` generate；`bridge/session_manager.py:859` ensure_started | sandbox 前登记、懒启动、多 turn session |
| `bridge/config_inheritance.py:101`；`bridge/runner.py:330` | 关闭生成自动重试、epoch 串行、外部 Orchestrator 证据边界 |
| `areal/trainer/rl_trainer.py:755` strict online update | admission pause、drain、超时边界 |

↩ [返回主文档 AREAL-09](../../private_resume/2026-08-llm-infra-interview-prep.md#areal-09) · [返回知识关系](../KNOWLEDGE_GRAPH.md) · [阅读索引](../MASTER_READING_LIST.md)

<a id="gateway-personal-followups"></a>
### 9. 后续个人分支：与上述团队主线分开讲

以下保留此前工作副本的代码审阅记录。它描述 exact quota、reward identity、liveness 等后续演进，**不属于本次指定仓库的 HEAD**；具体是否用于某次性能对照，须另用部署 commit、配置和日志关联。个人实现不能反向覆盖团队基础架构的署名，团队成果也不妨碍说明自己真实完成的集成、调优与验证。

<details>
<summary>展开后续个人分支的四类改造与历史提交</summary>

#### 1. 从 supply-driven 变为 step-plan-driven admission

原始风险：外部 producer 哪个 domain 来得快，哪个 domain 就可能占满训练供给；多 Teacher MOPD 中会静默饿死某个 Teacher route。

逻辑变化：trainer 先生成本 step 的 exact domain quota plan；Gateway 用 reservation → claim → session → export 路由把 cohort 绑定到 domain、worker 和 step。optimizer update、weight sync 和 model save 成功后，trainer 才在内存中 `commit_pending()`；紧随其后的 recovery checkpoint 再持久化已推进的 fairness cursor。若在 commit 前失败，pending plan 不推进；若进程在 commit 后、recovery checkpoint 落盘前退出，恢复仍读取上一个持久化 cursor，从而重放这一步，而不是静默跳过配额。

证据：`10a3e264` 与 `9979a0f6` 是不同分支/演进阶段的同类 exact-quota 实现，答题时合并为一项能力，不累计成两个成果。

#### 2. 把 session lifecycle 与 reward identity 变成 fail-closed contract

原始风险：外部 marker 被误当作权威 completion、reward/end 到达顺序竞争、一个 rejected cohort 的 sibling 仍在运行却被过早清理，都会导致奖励写错、trajectory 丢失或串 session。

逻辑变化：Proxy 选择权威 final completion，外部 marker 只做诊断；reward 与 end 两种顺序都进入同一生命周期状态机；rejected cohort 不再获得 trainer credit，但仍保留 active sibling route 直到自己的 terminal cleanup；zero-interaction 继续 fail closed。

证据：`c83de5fa`、`e7373e8b`、`afb1882c`，以及对应 reward identity、session lifecycle 和 cohort rejection tests。

#### 3. 从“长时间等”改为“有边界、可证明安全的重试”

原始风险：registration 在 domain lock 内 long-poll，且复用 multi-hour streaming timeout；一个 abandoned handler 或坏 backend 就能锁住整个 domain。另一类风险是 closed-domain episode 占满 worker slots，open-domain episode 永远拿不到执行机会。

逻辑变化：

- quota miss 立即返回，不在 domain lock 内等待；backend forward 移到锁外；
- register/control 等小 RPC 使用 bounded timeout，真正的 streaming/ready wait 保留长 timeout；
- trainer 对 group size 和 wrong-domain 做第二道 fail-fast gate；
- Gateway 只对“尚未绑定、确定没有远端副作用”的 structured `quota_domain_closed` 返回 safe requeue；
- bare 408/429/5xx、transport error 或可能已经 claim 的模糊失败，必须复用同一 task/cohort/rank 原地重试，不能换身份；
- requeue 到队尾释放 worker slot，让当前开放 domain 获得执行机会。

证据：`eb8bd492`、`1162029d`、`b117b570`、`690816eb`、`30ab40c4` 及 fault-injection/behavioral tests。

#### 4. 正确性修复后继续保护 goodput

safe requeue 能打破死锁，但会产生 queue rotation tax；closed-domain 大队列反复轮转，还可能让同 cohort siblings 到达时间超过 partial deadline。

项目进一步调小 requeue throttle、扩 reset/admission worker，并扩大 partial cohort deadline，目标是让 sibling co-arrival time 显著小于 deadline。`21bb4862` 能证明配置与机制改动；若没有改动后的统一 benchmark/run log，只表述为“实现了吞吐保护机制”，不把 commit 标题直接当成“吞吐已恢复”的结果证据。

#### 只用于说明后续个人分支的总结

> 在后续个人分支中，我围绕 training-aware admission 补了 exact quota、公平性、liveness、session correctness 和 fault-injection 验证；OpenAI proxy 和 online cohort 基础架构属于团队已有能力。这些工作与 7 月团队调度重构分开讲，不把它们拼成同一个已部署版本或同一次性能实验。

</details>

<a id="areal-weight-sync-xccl-disk"></a>
## AReaL 权重同步：XCCL 与 disk 不是 checkpoint 的两种写法

### 问题与共同状态机

RL actor 在训练侧可能使用 Megatron/FSDP 的参数布局，rollout 在 vLLM/SGLang 中使用另一套 serving layout。一次更新不只是复制 `state_dict`，而是：训练参数收集/转换 → 传输 → inference engine load/refit → replica 验证 → policy version 切换。

两种模式都应服从同一条状态机：

```text
optimizer step(version=N)
  → pause/drain 需要隔离的 rollout
  → build WeightUpdateMeta(version=N+1)
  → convert + transfer + load/refit
  → verify all participating replicas
  → actor / critic / rollout set_version(N+1)
  → resume admission/generation
```

`version` 是 behavior-policy/staleness metadata，不是另一份模型产物；只有权重传输成功后才能推进。HF Saver/DCP recovery checkpoint 是按保存周期持久化训练恢复状态，和每次 actor→rollout 权重发布不是一回事。

### XCCL：直接分布式传输

XCCL 路径由训练侧参与发送的 rank(s) 与 rollout ranks 建立权重更新通信组。训练 engine 按参数映射收集/转换张量，切成 buckets，通过 collective 直接送到 rollout engine，再由后端 refit。

它的主要优势是避开共享文件系统、完整 HF 落盘和二次解析，适合高频同步；主要代价是：

- trainer/rollout rank、global rank 与 group member 必须精确一致；
- dtype、shape、参数顺序、tied weights、MoE expert identity 必须匹配；
- group 建立或某个 bucket hang 会把 rollout pause 直接暴露在关键路径；
- 后端需要提供 compatible distributed update/refit API；
- 部分 replica 成功时不能直接推进 version，否则会混合 behavior policy。

“trainer sender ranks”取决于训练分片和转换实现，不能默认所有 trainer ranks 都是 sender，也不能把它简化成 trainer rank 0 给所有 server 发一次普通 broadcast。

### disk：临时 HF transfer artifact

disk 路径先把当前 actor 权重转换并写入带版本的临时 HF transfer directory，rollout server 再通过 update/load endpoint 从该目录加载；LoRA 时也可能加载 versioned adapter path。

它的优势是生产者和消费者解耦，manifest/文件可以独立检查，loader 失败时也较易重试；代价包括 export、共享存储带宽、metadata/小文件、可见性等待、load/refit 与目录清理。需要验证：

- directory 是否以临时名写完后原子发布，避免读到半份权重；
- manifest、version、参数数量/shape/checksum 是否一致；
- 所有 rollout nodes 是否看到同一文件系统视图；
- 失败/恢复后临时目录是否泄漏或被错误复用；
- update 完成前旧 replica 是否继续服务，以及切换点是否一致。

这里的 disk artifact 只服务训练态→推理态转换。它通常不含 optimizer、scheduler、RNG、data cursor、queue/cohort state，不能承担训练恢复 checkpoint 的语义。

### 选择矩阵

| 约束 | 更倾向 XCCL | 更倾向 disk |
|---|---|---|
| sync cadence | 高频、exposed pause 敏感 | 低频或可容忍较长 pause |
| 网络/存储 | collective 域稳定、带宽充足 | 共享存储成熟，跨进程/跨故障域解耦更重要 |
| inference backend | 有稳定 distributed refit | 只有文件 load/refit 或该路径验证更成熟 |
| 调试/审计 | 已有 bucket/checksum/version telemetry | 需要保留可检查 transfer artifact |
| colocation/LoRA | 必须看具体分支约束 | 常是兼容性回退路径，但不能一概而论 |

项目口径：在相同项目 workload 下，verl 与 AReaL 最终都采用 XCCL，原因是实测权重更新时间更短；没有统一跨模型、后端、拓扑的 benchmark，就不说“XCCL 永远更快”。

### 当前项目分支的支持边界

- actor–rollout colocation 在该分支中显式要求 `weight_update_mode=disk`；这是这对 role 的调度/生命周期约束，不代表 ref/critic 其他 colocation 也同样受限；
- SGLang 的 LoRA distributed/XCCL update 在该分支中不支持，需要 disk；vLLM 与 full-weight/LoRA 的支持矩阵不同；
- XCCL group 只包含实际参与传输的 trainer sender rank(s) 与 rollout ranks；
- 这些是项目分支事实，不应外推为所有 AReaL release 的永久限制。

### 生产验收与排障

至少记录四段时间：training-side collect/convert、transfer/export、rollout load/refit、pause 后 exposed time。正确性上做 parameter checksum/抽样 tensor diff、same-prompt same-weight logprob check，并记录每个 replica 的 desired/loaded/active version。

故障时按边界排查：

1. **卡在 connect/group init**：检查 rank list、端口、world size、重复/缺失 member；
2. **卡在某个 bucket**：打印 parameter name/offset/shape/dtype、sender/receiver progress，判断顺序或尺寸不一致；
3. **传完但 logprob 不一致**：检查参数转换、tied weights、router/expert mapping、tokenizer/chat template 与 cache；
4. **disk load 看不到文件**：检查写完发布协议、共享挂载一致性、manifest 与目录权限；
5. **只有部分 replica 新版本**：保持/回退旧 active version 或隔离失败 replica，禁止把混合版本 cohort 当成同一 behavior policy；
6. **同步成功但 E2E 变慢**：拆开 sync latency 与 exposed pause，检查同步 cadence、drain、cache invalidation/re-prefill 和 producer/consumer 配平。

## 核心指标

Agentic RL 平台至少要监控这些指标：

- rollout token/s；
- rollout request/s；
- active/target concurrency 与 slot idle ratio；
- per-worker load skew 与 session-affinity hit；
- CUDA Graph hit/fallback、decode-only tokens/s 与 graph-pool memory；
- trajectory queue depth；
- reward/verifier queue depth；
- reward latency p50/p95/p99；
- policy update time；
- policy idle time；
- sample staleness；
- policy version lag；
- weight sync latency；
- rollout error rate；
- cohort ready latency、Rejected Group ratio 与 reason distribution；
- retryable/terminal failure、retry count 与 capacity leak；
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
- [MOPD](mopd.md)：把 Student rollout、Teacher scoring、domain routing 和多领域 capability integration 接入 post-training dataflow。
- [verl 与 AReaL：RL 框架架构选型](rl_framework_selection.md)：把 Agentic RL 的系统矛盾映射为框架选型，区分项目历史版本、当前能力、改造半径和公平 benchmark。
- [Agentic for Embodied](agentic_for_embodied.md)：把 Agentic RL 的 rollout、policy version、scheduler 和 tracing 扩展到 sensor-action trajectory、GPU simulation、robot fleet、edge deadline 与独立 safety authority。
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

1. 用最简单的话描述 PPO、GRPO、DAPO，它们是什么关系？
2. 为什么 Agentic RL 不能简单复用 pretraining infra？
3. rollout latency 为什么会拖慢 policy update？
4. Fully Async、streaming、partial rollout、staleness 分别是什么？
5. sample freshness 在 RL 训练里为什么重要？
6. training state 和 generation state 的模型布局有什么区别？
7. 为什么 actor resharding 会成为 RLHF 系统瓶颈？
8. reward/verifier 为什么要独立扩缩容？
9. 外部 Agent 如何通过 OpenAI-compatible Gateway 接入训练？
10. Gateway 兼容 OpenAI API 后，为什么仍可能“能跑但训错”？
11. 你对项目 Gateway 做了哪些改造，个人 ownership 到哪里？
12. 长上下文 trajectory 对 KV cache 和 checkpoint 有什么影响？
13. context compaction 为什么不能只当作 inference-time heuristic？
14. 如何判断 trainer idle 是 rollout 慢还是 reward 慢？
15. Agent runtime 和 RL trainer 解耦后，trace schema 应该记录什么？
16. AReaL 的 XCCL 与 disk 权重同步如何选择，为什么 disk transfer 不等于 recovery checkpoint？
17. CUDA Graph 为什么对 decode 收益大，continuous batching 和动态 KV 如何满足 capture 契约？
18. Gateway 的流式补位为什么不是 token streaming？如何同时守住并发、affinity 与幂等？

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
12. 如果 XCCL 某个 rollout rank 更新失败、其他 rank 已完成，version 和流量应该如何处理？

## 主要来源

- [PPO 原始论文](https://arxiv.org/abs/1707.06347)：clipped surrogate objective 与多 epoch minibatch update。
- [DeepSeekMath](https://arxiv.org/abs/2402.03300)：GRPO 的 group-relative advantage 与去 Critic 动机。
- [DAPO](https://arxiv.org/abs/2503.14476)：Clip-Higher、Dynamic Sampling、Token-level Policy Gradient Loss 与 Overlong Reward Shaping。
- [CUDA Programming Guide：CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)：graph definition、instantiation、replay 与 host launch overhead。
- [AReaL v2.1 Online Proxy](https://github.com/areal-project/AReaL/blob/v2.1.0/docs/en/tutorial/online_proxy.md)：外部应用的 session key、OpenAI-compatible endpoint、reward 与 end-session 协议。
- [AReaL v2.1 Agent Workflow](https://github.com/areal-project/AReaL/blob/v2.1.0/docs/en/reference/agent_workflow.md)：Proxy Worker、InteractionCache、token-level tracking 与 workflow export。
- [AReaL v2.1 Async Guide](https://github.com/areal-project/AReaL/blob/v2.1.0/docs/en/algorithms/async.md)：policy version、off-policyness 与 partial rollout。
- [verl v0.9.0 release](https://github.com/verl-project/verl/releases/tag/v0.9.0)：V1 trainer、streaming dataloader、staleness control 与 Agentic RL 的当前版本边界。

## 我的总结

Agentic RL Infra 的关键转变是：训练平台开始承担在线数据生产系统的职责。过去我们优化的是单个 step 的计算效率；现在还要优化 trajectory 的生成、验证、排队、版本管理和消费效率。未来高级训练 infra 工程师需要同时理解训练并行、推理引擎、任务环境、队列调度和可观测性。这个方向值得长期跟踪。

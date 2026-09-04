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
## Gateway 调度：从 wave barrier 到流式补位

![Gateway 流式补位、均衡分发与失败请求分流](../assets/topics/gateway-streaming-refill.svg)

这里的 **streaming refill** 是请求级调度：一个 rollout 完成、失败或释放 capacity 后，立即补入下一个待处理请求。它不是 OpenAI HTTP `stream=true` 的 token streaming。

### 原链路为什么浪费并发

固定 wave/batch 发出一批请求后，快请求已经完成，慢请求、tool timeout 或失败请求仍占据这批的 barrier。若调度器等整波结束才发下一波，配置并发为 `C`，实际 active sessions 会逐渐从 `C` 掉到很低。group-based RL 更糟：同一 prompt 的 sibling trajectories 必须满足 cohort 完整性；缺一个成员，前面已经生成的 sibling 也可能无法形成训练 group。

### 三个改造层

#### 1. 流式补位

- 维护 bounded pending queue 与 target concurrency；
- completion/failure/abort 统一释放 slot；
- 每释放一个 slot 就触发 dispatcher refill；
- 设置 backpressure，不用无限队列掩盖 rollout 供需失衡。

直接因果链是：`slot idle time 下降 → active concurrency 接近水位线 → 单位时间 completed trajectories 增加`。

#### 2. 均衡分发但保留 affinity

项目代码用 round-robin 给没有既有 owner 的新 route 分配 Proxy Worker，并由 capacity/backpressure 限制执行水位；reservation、cohort、claim、session 等 identity 一旦绑定，后续请求都解析到同一 worker。这样在静态 worker 能力接近时分散新负载，同时避免 InteractionCache、session state 和 prefix cache 在 worker 间漂移。

round-robin 不读取实时负载，因此不能称为 least-load；worker 异构或长尾严重时，可在保持 identity owner 不变的前提下为**新 route**扩展 capacity-aware/least-load 选择。已绑定 session 的正确性优先；worker 失效需要显式 lifecycle/recovery，而不是无条件随机改路由。

#### 3. 失败请求状态机

请求至少要区分：

```text
pending → admitted/in-flight → completed
                         ├→ retryable failure → same identity retry
                         └→ terminal/aborted → sibling/session cleanup
```

timeout、client disconnect、backend 5xx、reward 缺失、staleness 和显式取消不能全部走同一种 retry。安全重试需要 idempotency key/claim/session identity，防止重复生成、重复记 reward 或产生两个 cohort member；terminal failure 必须释放 capacity 并清理 sibling，避免半组永久占槽。

### 结果如何解释

最新版简历结果：Rollout 阶段平均推理吞吐 `+60%`；Rejected Group `33.18%→2.73%`，绝对下降 `30.45pp`，相对下降约 `91.8%`。

不能从这两个数字直接推出模型效果提升。至少还要同时看：

- active/target concurrency、slot idle、pending depth；
- per-worker load skew、session-affinity hit、prefix-cache reuse；
- retry/timeout/terminal failure 与 capacity leak；
- cohort ready latency、rejection reason、staleness；
- exported/consumed/gradient-active trajectories 与有效 token；
- trainer exposed wait、reward/eval distribution。

三项机制是联合改造，没有独立消融时不把 60% 拆成各自收益。早期性能重构与后续 [exact quota/liveness ownership](#project-gateway-ownership) 也要分开：后续提交强化的是正确性和可恢复性，不能把已有性能结果重新归因给它们。

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
  → Gateway：鉴权、路由、admission
  → CohortManager：task/cohort/group rank、capacity、rollout version、staleness
  → Proxy Worker：OpenAI-compatible endpoint
  → vLLM/SGLang：生成 token/logprob
  → InteractionCache：interaction、token、behavior logp、version、reward
  → rewarded + ended + complete cohort
  → Trainer export/tensorize → advantage/loss/update
```

API compatibility 只解决“Agent 会不会调用”，不解决“轨迹能不能正确训练”。训练系统仍必须保证：

- session/interaction identity 不串；
- terminal reward 写到权威 completion；
- 重试不会重复创建或把一个 episode 路由到另一 cohort；
- cohort/group 完整性和 domain 配额正确；
- behavior logprob、token/version lineage 完整；
- update/checkpoint/recovery 后 queue 和 fairness state 一致。

<a id="project-gateway-ownership"></a>
## 项目 Gateway 二次开发：逻辑到底改了什么

### 先划清 ownership

团队/上游已经提供 OpenAI-compatible proxy、online session/cohort 基础链路、Proxy Worker、InteractionCache、CohortManager 和 trainer consumer。项目基础提交 `64adce36` 的作者不是本人，因此面试中应说“项目基于这条链路”，不能说是自己从零设计。

个人代码 ownership 可以归纳为四类。

### 1. 从 supply-driven 变为 step-plan-driven admission

原始风险：外部 producer 哪个 domain 来得快，哪个 domain 就可能占满训练供给；多 Teacher MOPD 中会静默饿死某个 Teacher route。

逻辑变化：trainer 先生成本 step 的 exact domain quota plan；Gateway 用 reservation → claim → session → export 路由把 cohort 绑定到 domain、worker 和 step。optimizer update、weight sync 和 model save 成功后，trainer 才在内存中 `commit_pending()`；紧随其后的 recovery checkpoint 再持久化已推进的 fairness cursor。若在 commit 前失败，pending plan 不推进；若进程在 commit 后、recovery checkpoint 落盘前退出，恢复仍读取上一个持久化 cursor，从而重放这一步，而不是静默跳过配额。

证据：`10a3e264` 与 `9979a0f6` 是不同分支/演进阶段的同类 exact-quota 实现，答题时合并为一项能力，不累计成两个成果。

### 2. 把 session lifecycle 与 reward identity 变成 fail-closed contract

原始风险：外部 marker 被误当作权威 completion、reward/end 到达顺序竞争、一个 rejected cohort 的 sibling 仍在运行却被过早清理，都会导致奖励写错、trajectory 丢失或串 session。

逻辑变化：Proxy 选择权威 final completion，外部 marker 只做诊断；reward 与 end 两种顺序都进入同一生命周期状态机；rejected cohort 不再获得 trainer credit，但仍保留 active sibling route 直到自己的 terminal cleanup；zero-interaction 继续 fail closed。

证据：`c83de5fa`、`e7373e8b`、`afb1882c`，以及对应 reward identity、session lifecycle 和 cohort rejection tests。

### 3. 从“长时间等”改为“有边界、可证明安全的重试”

原始风险：registration 在 domain lock 内 long-poll，且复用 multi-hour streaming timeout；一个 abandoned handler 或坏 backend 就能锁住整个 domain。另一类风险是 closed-domain episode 占满 worker slots，open-domain episode 永远拿不到执行机会。

逻辑变化：

- quota miss 立即返回，不在 domain lock 内等待；backend forward 移到锁外；
- register/control 等小 RPC 使用 bounded timeout，真正的 streaming/ready wait 保留长 timeout；
- trainer 对 group size 和 wrong-domain 做第二道 fail-fast gate；
- Gateway 只对“尚未绑定、确定没有远端副作用”的 structured `quota_domain_closed` 返回 safe requeue；
- bare 408/429/5xx、transport error 或可能已经 claim 的模糊失败，必须复用同一 task/cohort/rank 原地重试，不能换身份；
- requeue 到队尾释放 worker slot，让当前开放 domain 获得执行机会。

证据：`eb8bd492`、`1162029d`、`b117b570`、`690816eb`、`30ab40c4` 及 fault-injection/behavioral tests。

### 4. 正确性修复后继续保护 goodput

safe requeue 能打破死锁，但会产生 queue rotation tax；closed-domain 大队列反复轮转，还可能让同 cohort siblings 到达时间超过 partial deadline。

项目进一步调小 requeue throttle、扩 reset/admission worker，并扩大 partial cohort deadline，目标是让 sibling co-arrival time 显著小于 deadline。`21bb4862` 能证明配置与机制改动；若没有改动后的统一 benchmark/run log，只表述为“实现了吞吐保护机制”，不把 commit 标题直接当成“吞吐已恢复”的结果证据。

### 最适合面试的总结

> 我没有把 Gateway 只当成 HTTP 转发层，而是把它改成 training-aware admission/control plane：它理解本 step 的 domain plan、cohort/session 生命周期、reward identity、policy version、safe retry 和 recovery。我的 ownership 主要是 exact quota、公平性、liveness、session correctness 和 fault-injection 验证；OpenAI proxy 和 online cohort 基础架构属于团队已有能力。

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

# verl 与 AReaL：RL 框架架构选型指南

> 定位：工程选型专题。主问题不是“哪个框架更强”，而是特定 workload、团队资产和版本约束下，哪个系统抽象能以更小改造半径交付目标能力。
>
> 核验日期：2026-09-02。项目判断以当时评估/使用的代码分支为准；当前官方基线为 verl v0.9.0（`483b8a0`）和 AReaL v2.1.0（`ecc8b0e`）。
>
> 面试速答入口：[AREAL-01｜为什么先选 verl，Agentic RL 阶段又转向 AReaL？](../../private_resume/2026-08-llm-infra-interview-prep.md#areal-01)

## 1. 先给结论

### 一句话区分

> **verl 的系统中心是灵活编排 RL 的多角色计算图和训练/推理后端；项目当时采用的 AReaL 路径，系统中心是持续生产 Agent trajectory，并在异步训练中管理 session、cohort、policy version 和 staleness。**

这句话描述的是项目做选型时最有区分度的架构重心，不是永久的功能边界：两个框架都能做同步和异步 RL，也都在补强 Agentic RL。尤其是 [verl v0.9.0](https://github.com/verl-project/verl/releases/tag/v0.9.0) 已加入更完整的 Fully Async、Agentic RL 和 Uni-Agent Gateway 路径，所以今天重新选型必须重新 benchmark。

### 最重要的选型原则

先定义 workload，再比较框架。不要先选一个热门框架，再把需求硬塞进去。

```text
任务形态
  → 模型与训练后端
  → rollout / Agent 接口
  → 资源 placement 与 weight sync
  → correctness / staleness
  → observability / recovery
  → 团队已有资产与二次开发半径
```

项目中的阶段性结论是：

- 标准 SFT/RLVR 阶段，团队更需要成熟的训练后端、算法 dataflow、vLLM/SGLang 集成和模型覆盖，因此选择 verl。
- 需求转为 128K、多轮 Tool/Sandbox、外部 Agent 在线接入后，session/cohort、长尾隔离、staleness 和 Gateway 改造成为主要矛盾，因此选择当时更贴合该 workload 的 AReaL 路径。
- 选择 AReaL 后并非“免费获得所有生产能力”；项目仍要补齐 Gateway 调度、online drain、lineage、监控、恢复、评测和多 Teacher 路由等交付能力。

## 2. 先识别 workload

| Workload | 主要矛盾 | 选型时最该问的问题 |
|---|---|---|
| SFT | 训练吞吐、显存、数据管线、checkpoint | 目标模型和 Megatron/FSDP 后端是否成熟？ |
| 标准同步 RLVR | Actor/Ref/Reward/Rollout 编排与 correctness | 算法 recipe、logprob、mask、weight sync 是否已经闭环？ |
| Fully Async RLVR | rollout/training overlap、长尾和样本新鲜度 | queue、partial rollout、staleness、恢复是否是一级抽象？ |
| 长时 Agentic RL | 多轮 session、tool/sandbox、外部环境、tail latency | Agent runtime 如何接入？trajectory 生命周期由谁管理？ |
| 外部 Agent 在线训练 | API 协议、鉴权、路由、session/cohort 和 tracing | Gateway 是否允许低成本扩展并保持 token-level lineage？ |
| 大规模 MoE RL | Megatron 后端、rollout engine、权重转换和通信 | 训练态与推理态的并行布局能否稳定转换？ |

框架差异往往不是“有没有某个 feature”，而是：这个 feature 是稳定主路径、实验 recipe，还是需要跨多个层次进行二次开发。

## 3. verl 的系统思想

### 3.1 Hybrid-Controller，而不是纯 single-controller

根据 [verl 0.7 架构说明](https://verl.readthedocs.io/en/latest/blog/v0.7.html)，HybridFlow/Hybrid-Controller 结合了两种编程模型：

- 高层使用 single-controller/MPMD：`RLTrainer` 描述 PPO、GRPO、DAPO 等多阶段、多模型 dataflow，决定何时 rollout、打 reward、算 logprob、更新 Actor，以及如何放置资源。
- 内部使用 SPMD/multi-controller：FSDP、Megatron、VeOmni 等训练 engine，以及 vLLM、SGLang、TensorRT-LLM 等 rollout engine，在各自 workers 上执行分布式计算和 collective。

因此，verl 的核心不是“Ray 启动了一批进程”，而是把 RL 算法控制流与底层分布式执行解耦：

```text
RLTrainer / algorithm dataflow
  → ResourcePool / WorkerGroup / placement
  → Actor / Rollout / Ref / Critic / Reward
  → Model Engine / Rollout Engine / Checkpoint Engine / TransferQueue
  → Megatron | FSDP | VeOmni / vLLM | SGLang | TensorRT-LLM
```

### 3.2 它擅长解决什么

1. **多角色编排**：Actor、Rollout、Reference、Critic、Reward 可以 colocate，也可以分离部署。
2. **后端复用**：上层 RL dataflow 不需要重写 Megatron/FSDP/vLLM/SGLang 的分布式实现。
3. **算法扩展**：PPO、GRPO、DAPO、OPD 等可以复用同一套角色和数据流抽象。
4. **资源布局**：通过 ResourcePool、WorkerGroup 和 placement 组合不同 GPU 分配。
5. **训练—推理转换**：权重同步、reshard、checkpoint/transfer 是显式系统问题，而不是简单 `state_dict` 拷贝。

截至 v0.9.0，verl 已进一步补强 server-based rollout、Fully Async、Agentic RL、Uni-Agent Gateway、TransferQueue 和 checkpoint engine；官方 release 还描述了通过 OpenAI/Anthropic-compatible endpoint 接入外部 harness，并支持 **1,000+ concurrent stateful sessions**。这是官方 release 披露的能力口径，不是本项目复测结果。verl 已经不能再被简单归类为“只适合同步 RLVR”。

### 3.3 优势

- 标准 SFT/RLVR 的模型、算法、训练后端和 rollout 生态完整度较高。
- 对 Megatron/FSDP 与 vLLM/SGLang 的组合路径较丰富，适合已有训练基础设施团队。
- colocate、disaggregate、sync、one-step-off-policy、fully async 等模式可以沿统一 dataflow 演进。
- 适合从同步 baseline 开始验证 correctness，再逐步打开异步和性能优化。

### 3.4 代价

- 抽象层较多：trainer、WorkerGroup、engine、rollout server、AgentLoop、TransferQueue、checkpoint engine 和配置会共同决定行为。
- 在项目当时版本里，若要深改外部 Agent ingress、Gateway 路由和 trajectory 生命周期，可能同时牵引 trainer/worker、数据协议、AgentLoop 和推理 engine 接口。
- 支持某个能力不等于该能力已覆盖目标模型、后端、硬件、恢复和长稳组合；版本与依赖矩阵仍需要实测。

这里的“框架较重”不是贬义，也不是说代码量大，而是说一次控制面改动的影响面可能跨越多个抽象层。对于标准 dataflow，这种完整抽象是优势；对于需要快速重写 Gateway/session 语义的项目，它可能增加改造成本。

## 4. AReaL 的系统思想

### 4.1 项目当时的 online proxy/cohort 路径

项目实际使用的是 AReaL online proxy/cohort 相关路径及公司二次开发，不应倒推成后续 AReaL 2.x 的完整微服务架构。面试时可按下面的控制链解释：

```text
External Agent / Evals + Tool / Sandbox
  → Gateway / CohortManager admission
  → session 绑定 cohort、group rank、rollout version、proxy worker
  → Proxy Worker → vLLM / SGLang
  → InteractionCache 记录 token、behavior logp、token version
  → terminal reward + end_session
  → cohort complete + ready-time staleness gate
  → Trainer 导出、tensorize、DP redistribute
  → Ref/Critic/Teacher/Prox logp + advantage
  → PPO/GRPO update
  → versioned weight transfer → version advance → resume rollout
```

这个路径的系统中心不是单次 `generate()`，而是 session 从 admission、推理、reward、结束，到成为可训练 cohort 的完整生命周期。它尤其适合外部 Agent 自己维护多轮环境状态、Tool/Sandbox 交互，而训练系统通过 Gateway 接管版本、轨迹和容量控制的场景。

### 4.2 为什么异步是一级问题

[AReaL v2.1 异步训练指南](https://github.com/areal-project/AReaL/blob/v2.1.0/docs/en/algorithms/async.md)把 rollout 和 training 放在分离 GPU 上并行执行，并用 `max_head_offpolicyness` 控制 rollout 相对训练 policy 的版本落后；partial rollout 还允许长 trajectory 跨 policy version 分段。

异步带来的不是纯性能收益，而是一组 correctness 责任：

- 每个样本由哪个 behavior policy 产生？
- trajectory 或 token 跨了多少个 version？
- 旧样本是训练、mask、reweight、reject，还是直接丢弃？
- weight sync 部分失败时，如何避免新旧权重混合服务？
- in-flight session、ready cohort 和 trainer state 如何一起恢复？

### 4.3 AReaL 2.0/2.1 不能倒推项目历史

[AReaL v2.0.0](https://github.com/areal-project/AReaL/releases/tag/v2.0.0) 在 2026-07-01 把 training、inference、agent 和 weight update 明确拆成微服务；[v2.1.0](https://github.com/areal-project/AReaL/releases/tag/v2.1.0) 于 2026-08-25 继续补充调度、proxy、版本归属、SWE workflow、Megatron 和 weight update 等能力。

因此需要区分三层：

| 层次 | 可以怎么说 | 不能怎么说 |
|---|---|---|
| 项目历史 | 使用 online proxy/cohort 路径并做 Gateway、lineage、恢复等二次开发 | “项目天然就是 AReaL 2.0 完整微服务” |
| 2.0 里程碑 | 官方把 training/inference/agent/weight update 服务化 | “2.0 代表所有生产能力都已完备” |
| 当前 2.1 | 当前官方版本继续演进 proxy、scheduler、version 和后端支持 | 用当前能力反向包装个人历史贡献 |

### 4.4 优势

- 长 trajectory、异步 producer-consumer、policy version 和 staleness 是架构主线，而不是事后补丁。
- external Agent 可以通过 OpenAI-compatible proxy/gateway 接入，session/interaction 更贴近在线 Agent runtime。
- rollout 与 trainer 可独立扩缩容，便于处理长尾、工具服务抖动和不同资源比例。
- 对需要深改 Gateway、cohort admission、trajectory lineage 的项目，项目当时的改造半径更符合团队需求。

### 4.5 两类代价必须分开

**异步架构固有复杂度**：

- off-policy correctness；
- behavior logprob 与 policy/token version；
- staleness、partial rollout 和样本拒绝；
- weight sync 原子性；
- in-flight trajectory 与跨服务恢复。

**项目当时需要补齐的外围能力**：

- Gateway 调度和 online drain；
- trajectory lineage、监控与审计；
- checkpoint/recovery、部署与评测工装；
- 多 Teacher 路由与数据域校验。

第二类是当时项目版本和团队交付的缺口，不能写成 AReaL 2.x 永久缺陷。

## 5. 在同一维度上比较

| 维度 | verl | AReaL | 工程判断 |
|---|---|---|---|
| 系统中心 | 多角色 RL dataflow、engine 与 placement 编排 | 异步 Agent trajectory、session/version 与服务反馈 | 看主要矛盾在训练编排还是在线 trajectory 生命周期 |
| 控制模型 | 高层 single-controller/MPMD，内部 SPMD engine | 项目路径以 proxy/cohort/controller 管理 online flow；2.x 转向服务化 | 不要把不同版本画成同一架构 |
| 标准 SFT/RLVR | 后端与 recipe 路径丰富 | 可以支持 RL，但不是项目选择它的首要原因 | 已有 verl 资产时通常先复用 |
| Fully Async | v0.7.1 时仍快速演进；v0.9 已显著补强 | 原生围绕异步、staleness、partial rollout 展开 | 当前必须用目标 workload 重测 |
| Agent 接入 | v0.9 已有 Agentic RL 与 Uni-Agent Gateway | online proxy/session/interaction 是项目核心路径 | 两者当前能力边界已明显收窄 |
| Training backend | Megatron/FSDP/VeOmni 等组合丰富 | 支持 Megatron/FSDP/Archon 等，随版本演进 | 按目标模型、精度和 checkpoint 实测 |
| Rollout backend | vLLM/SGLang/TensorRT-LLM 等 | vLLM/SGLang 等 | 不能只看“支持”，还要测 weight refit 和长稳 |
| Correctness | logprob、mask、rollout correction、version 等 | staleness、token/sample version、decoupled loss 等 | 都必须做 same-weight logp 和 lineage 校验 |
| 二次开发半径 | 标准 dataflow 扩展强；深改控制面可能跨多层 | 项目当时改 Gateway/session/cohort 更直接 | 这是版本和团队代码基础相关结论 |
| 运维复杂度 | Ray/WorkerGroup/engine/queue/checkpoint 组合复杂 | proxy/session/service/version/recovery 组合复杂 | 没有哪个框架天然“运维简单” |

## 6. 项目中的真实选型过程

### 6.1 第一阶段：为什么从 verl、slime、ROLL 中选择 verl

当时的目标是尽快交付 SFT 和标准 RLVR，不是先做一个通用 Agent 平台。比较维度应当这样讲：

1. 目标模型、Megatron/FSDP 后端和长上下文支持；
2. PPO/GRPO 等算法与 Actor/Ref/Reward/Rollout dataflow 完整度；
3. vLLM/SGLang rollout 与训练—推理权重同步；
4. correctness、checkpoint、可观测性和故障恢复；
5. 团队已有代码、上手成本和后续维护成本。

基于当时实际评估的版本和团队资产，verl 与已有 Megatron、vLLM/SGLang 体系更匹配，因此选择 verl。对 slime、ROLL 只说明参与过评估和比较维度，不编造没有实验记录支持的绝对排名。

### 6.2 第二阶段：为什么 Agentic RL 转向 AReaL

后来 workload 发生了结构性变化：

- 128K 长上下文；
- 多轮 Tool/Sandbox；
- 外部 evals/Agent client 持续发起请求；
- 同一任务需要多条 trajectory 组成 cohort；
- rollout 长尾、session 生命周期、policy version 和 staleness 成为主问题；
- Gateway 需要按项目控制面做深度修改。

在项目当时的代码基础上，AReaL online proxy/cohort 路径更贴近这组需求，改造 Gateway、admission、session、interaction 和 trajectory lineage 的半径更小。因此转向 AReaL，不是因为“verl 做不了异步”，也不是因为“AReaL 所有方面更先进”。

### 6.3 Ownership 应该怎么讲

准确的 ownership 是：

- 参与框架选型，定义与 workload 对齐的评价维度；
- 负责选定框架后的系统集成和关键二次开发；
- 用吞吐、正确性、模型效果、恢复和长稳指标完成交付闭环；
- 明确哪些是开源框架能力，哪些是团队改造，哪些是本人负责模块。

不要把 HybridFlow、AReaL async algorithm、vLLM/SGLang 或 Megatron 底层实现说成个人自研。

## 7. 如果今天重新选型

verl v0.9 已经显著补齐此前的部分差距，因此当前推荐不是静态的“verl 做 RLVR，AReaL 做 Agentic RL”，而是下面这张决策表：

| 当前场景 | 推荐起点 | 前提 |
|---|---|---|
| SFT、同步 GRPO/RLVR baseline | verl | 目标模型/后端在支持矩阵内，团队已有 verl 资产 |
| 基于现有 verl 系统演进 Fully Async | verl | 先验证当前 v0.9 trainer、queue、weight sync、recovery 是否覆盖目标组合 |
| 新建长时 Agent + 外部 harness 平台 | verl v0.9 与 AReaL 2.1 同时进入 PoC | 比较 gateway/session、trajectory contract 和生产恢复，不凭旧印象选择 |
| 深度定制 session/cohort/staleness 控制 | 倾向 AReaL PoC | 必须确认 2.1 服务接口与项目控制面匹配 |
| 希望统一已有 SFT/RLVR/OPD/Agentic recipe | 倾向 verl PoC | 统一生态收益大于 Gateway 定制成本 |
| 团队已有成熟 AReaL 平台与运维体系 | 倾向继续 AReaL | 迁移收益必须大于平台重建成本 |

框架选择的核心公式可以概括为：

```text
总交付成本
= 首次适配成本
+ 正确性验证成本
+ 性能优化成本
+ 长稳与恢复成本
+ 后续升级和团队维护成本
```

## 8. 如何做公平 benchmark

### 8.1 固定 workload

至少固定：

- 模型、dtype、训练和 rollout backend；
- prompt/response 长度分布、每 prompt 采样数和 tool latency；
- 总 GPU 数、trainer/rollout 资源比和网络拓扑；
- 算法、batch/token 口径、weight-sync cadence 和 staleness 约束；
- warmup、统计窗口、异常步处理、验证频率和 checkpoint 频率。

### 8.2 四层验收

| 层次 | 必测内容 |
|---|---|
| FUNCTIONAL | 能否跑通目标模型、Tool/Sandbox、长上下文、保存和恢复 |
| NUMERIC | same-weight logp、token/mask/reward 对齐、version 与 mixed-weight 检查 |
| PERFORMANCE | tokens/s/GPU、samples/hour、goodput、GPU utilization、p95/p99、weight-sync exposed time |
| EFFICACY | 相同 eval 协议下的 reward、目标任务能力、General 回归和不同 seed/checkpoint |

### 8.3 异步系统额外指标

- rollout queue depth、enqueue/dequeue rate；
- trainer idle 与 rollout idle；
- trajectory age 和 policy-version lag 分布；
- stale rejection/partial rollout 比例；
- Gateway、Tool/Sandbox、reward/verifier 的 p50/p95/p99；
- weight update 成功率、耗时和 version convergence；
- 故障后 in-flight session、ready cohort 和训练进度的恢复比例。

只比较框架官方 benchmark 没有意义；它们通常使用不同模型、长度、硬件、资源比和效果口径。真正的选型结果必须来自团队自己的 workload。

## 9. 面试口述模板

### 30 秒版本

> 我把 verl 看成多角色 RL dataflow 和训练/推理后端的编排框架，把项目当时的 AReaL 路径看成异步 Agent trajectory 的生产和版本控制系统。标准 SFT/RLVR 阶段，我们更需要成熟的 Megatron、vLLM/SGLang 和算法链路，因此选 verl；后来转向 128K、多轮 Tool/Sandbox 和外部 Agent，session/cohort、长尾、staleness 和 Gateway 改造成为主要矛盾，所以选 AReaL。这个结论基于当时版本，不是说 AReaL 全面优于 verl；当前 verl 0.9 已补强 Agentic RL，今天会重新做同 workload PoC。

### 2 分钟版本

> 我做选型时不会先给框架排总榜，而是先看 workload。第一阶段目标是稳定交付 SFT 和标准 RLVR，我比较了 verl、slime、ROLL 的模型和训练后端、算法 dataflow、vLLM/SGLang rollout、weight sync、correctness、checkpoint 和二开成本。基于当时的版本与团队 Megatron 资产，verl 的完整度和后端组合更匹配，所以选择 verl。
>
> 后来需求变成 128K 长上下文、多轮 Tool/Sandbox、外部 evals/Agent client 在线请求，同一个任务还要管理多条 trajectory。系统主要矛盾从标准 RL dataflow 变成 session/cohort 生命周期、rollout 长尾、policy version、staleness 和 Gateway 控制面。在项目当时的代码基础上，AReaL online proxy/cohort 路径更贴合，Gateway 和 trajectory lineage 的改造半径也更小，所以转向 AReaL。
>
> 代价是我们仍要补齐 Gateway 调度、online drain、监控、恢复、评测和多 Teacher 路由。我的结论不是“AReaL 异步、verl 同步”，也不是“AReaL 全面更先进”，而是架构要和主要矛盾匹配。现在 verl 0.9 已加入更完整的 Fully Async、Agentic RL 和 Uni-Agent Gateway，如果今天重新选，我会在同一模型、长度、资源和故障场景下同时做 PoC，再比较 goodput、staleness、correctness、恢复和维护成本。

## 10. 高频追问

### verl 现在也支持 Fully Async 和 Agentic RL，差异还成立吗？

历史选型仍成立，因为它描述的是当时版本、团队代码基础和改造成本；但不能把历史判断当作当前框架能力结论。当前差异已经收窄，应重新比较目标 workload 下的 Gateway/session contract、weight sync、staleness、恢复和模型支持。

### 你说 verl“重”，具体是什么意思？

不是启动慢或代码多，而是深改控制面时可能同时牵引 trainer、WorkerGroup、data protocol、AgentLoop、rollout engine、placement 和 checkpoint/transfer。对于标准 RL dataflow，这是完整抽象的价值；对于项目当时要快速深改 Gateway/session 语义的需求，改造半径相对较大。

### 为什么不一直使用 AReaL？

标准 SFT/RLVR 更看重训练后端、模型/算法 recipe、同步 baseline 和团队已有资产。框架统一也有价值，但不能为了统一牺牲成熟度和交付速度。Agentic RL 转向 AReaL 是 workload 改变后的局部最优，不代表所有阶段都应迁移。

### AReaL 还缺什么？

要先分版本。项目当时需要补 Gateway 调度、online drain、lineage、监控、恢复、评测和多 Teacher 路由；这些不能直接说成当前 AReaL 2.1 的永久缺陷。异步系统始终存在的复杂度则是 staleness、version、partial trajectory、weight sync 原子性和跨服务恢复。

### 如何证明选型不是主观偏好？

用相同 workload 做 PoC，至少比较 functional、numeric、performance、efficacy 四层，并加入 p99、staleness、weight-sync exposed time、故障恢复和团队维护成本。没有这些数据，只能叫技术调研，不能叫完成选型。

## 11. 常见错误回答

- “AReaL 是异步框架，verl 是同步框架。”
- “AReaL 架构全面更先进，所以后来替换 verl。”
- “verl 太重，AReaL 很轻。”
- 用当前 verl 0.9/AReaL 2.1 能力倒推项目当时实现。
- 把 slime、ROLL 写成永久能力排名，却没有保存当时的版本和实验记录。
- 只比较官方吞吐数字，不统一模型、长度、资源、warmup 和效果口径。
- 把开源框架原生能力说成个人设计，把团队能力全部说成个人 ownership。

## 12. 相邻主题

- [Agentic RL Infrastructure](agentic_rl.md)：理解 rollout、reward/verifier、weight sync、staleness 和 trajectory store 为什么改变训练系统边界。
- [Megatron 5D Parallelism](distributed_training.md)：理解训练 backend 的并行、显存和通信基础。
- [Long-context Training](long_context_training.md)：理解 128K–256K 训练与 rollout 的 KV、activation 和 tail latency。
- [Checkpointing](checkpointing.md)：理解 model/optimizer、policy version 和异步状态如何恢复。
- [MOPD](mopd.md)：理解多 Teacher online distillation 如何进入 RL dataflow。

## 13. 官方来源与版本边界

### verl

- [官方文档首页](https://verl.readthedocs.io/en/latest/)：HybridFlow、训练/推理后端、Agentic RL 和 Async Training 导航。
- [verl 0.7 架构说明](https://verl.readthedocs.io/en/latest/blog/v0.7.html)：Hybrid-Controller、Model/Rollout/Checkpoint Engine、TransferQueue 和 sync/async pipeline。
- [v0.9.0 Fully Async 文档](https://github.com/verl-project/verl/blob/v0.9.0/docs/advance/fully_async.md)：staleness、partial rollout、资源分离与实验口径。
- [v0.7.1 release](https://github.com/verl-project/verl/releases/tag/v0.7.1)：项目历史判断的公开版本参照，commit `bec9ef7`。
- [v0.9.0 release](https://github.com/verl-project/verl/releases/tag/v0.9.0)：当前重评基线，commit `483b8a0`。

### AReaL

- [官方仓库](https://github.com/areal-project/AReaL)。
- [v2.1.0 Asynchronous RL Guide](https://github.com/areal-project/AReaL/blob/v2.1.0/docs/en/algorithms/async.md)：`max_head_offpolicyness`、partial rollout 和 decoupled loss。
- [v2.1.0 Online Proxy](https://github.com/areal-project/AReaL/blob/v2.1.0/docs/en/tutorial/online_proxy.md)：外部 Agent/evals 与 proxy 的在线接入路径。
- [v2.1.0 Agent Workflow](https://github.com/areal-project/AReaL/blob/v2.1.0/docs/en/reference/agent_workflow.md)：Agent 与 rollout workflow 接口。
- [v2.0.0 release](https://github.com/areal-project/AReaL/releases/tag/v2.0.0)：微服务架构里程碑，commit `fee938e`。
- [v2.1.0 release](https://github.com/areal-project/AReaL/releases/tag/v2.1.0)：当前重评基线，commit `ecc8b0e`。

## 14. 我的总结

框架选型的本质不是比较 feature checklist，而是识别系统的主要矛盾。标准 SFT/RLVR 更关注多角色训练 dataflow、模型/后端生态和 correctness；长时 Agentic RL 会把 session、Tool/Sandbox、trajectory lifecycle、tail latency、policy version 和恢复推到控制面的中心。

项目先选 verl、后转 AReaL，是 workload 变化后的架构选择，不是对框架做永久排名。随着 verl v0.9 和 AReaL v2.1 持续演进，历史经验仍能证明选型能力，但当前决策必须重新用目标 workload、版本和生产门禁验证。

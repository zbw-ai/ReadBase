# Frontier Scan, 2026-09-05

- Previous scan：[2026-09-01](frontier_scan_2026-09-01.md)
- Window：2026-09-01 11:31:29 ~ 2026-09-05 00:21:28
- Timezone：Asia/Shanghai
- Generated at：2026-09-05 00:21:28
- Report type：flexible frontier scan
- Sources scanned：arXiv cs.AI / cs.LG / cs.CL / cs.DC；OpenAI / Anthropic / NVIDIA / DeepSeek official sources；Hugging Face Blog / TRL / Transformers / Accelerate / PEFT / Kernels；AReaL / verl / slime / ROLL / OpenRLHF / NeMo RL；Megatron-LM / vLLM / SGLang official changes
- Scan completeness：完整检查本窗口内 arXiv recent batch、核心厂商官方页面和重点框架 material changes。arXiv API 连接被上游重置后回退 official abstract/recent pages；部分 GitHub commit 页面回退官方 REST API、PR description、merged state、测试与 benchmark 记录核验。DeepSeek API changelog 本次无法稳定读取，已显式保留为下次补扫项。

## 本次核心判断

本次值得保留的不是模型发布，而是三个系统边界正在变得更清晰：

1. **Agentic RL rollout 不再是一次 prefill 后持续 decode 的静态 workload。** 环境交互会不断插入新 prefill；partial rollout 又要求请求级取消、保留 prefix 和恢复。AInfer-PD 与 slime 分别从 GPU collective 和 trajectory data contract 两侧处理这种动态性。
2. **昂贵 fresh rollout 的优化开始从“全部生成”转向“决定哪些旧样本仍值得学”。** Headroom-Drift Replay 把 replay 拆成剩余学习价值和 policy drift 两个判断，但这会把 policy/version/logprob provenance 变成训练系统必须记录的状态。
3. **长上下文与大规模训练的瓶颈越来越依赖 workload 和平台边界。** TRL 的 1M-token recipe 证明 CP 能把单序列摊到多卡，也明确暴露 full-causal/no-packing/hybrid-attention 不兼容等约束；2400-GPU multi-tenancy 研究则提醒我们，独占环境的 scaling curve 不能直接代表共享集群生产表现。

## Accepted Frontier Signals

### AInfer-PD：让 Distributed MoE Rollout 安全地并发 Prefill 与 Decode

- Signal ID：2026-09-05-001
- Source ID：arxiv:2609.00993
- First seen：2026-09-05 00:21:28
- 原始提交：2026-09-01 17:44:40，Asia/Shanghai
- Scan window：2026-09-01 11:31:29 ~ 2026-09-05 00:21:28
- Focus Match：P0 Focus
- 来源：arXiv primary page
- 类型：paper / Agentic RL inference / distributed MoE / prefill-decode multiplexing
- 链接：https://arxiv.org/abs/2609.00993
- Primary-source check：title、8 位作者、v1 timestamp、ADP/ATP 与 DeepEP communication-state isolation，以及全部 completion-time 数字均已对齐 arXiv metadata/abstract
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：它抓住 agentic rollout 与普通 serving 的关键差异：trajectory 在 generation 与 environment 间交替，新的 prefill 会持续插入正在 decode 的批次；MoE 下这种并发还会触发跨 rank collective order 和 DeepEP mutable state 冲突。
- Status：NEW
- 建议动作：优先阅读 collective ordering、P/D scheduling boundary 和 DeepEP state isolation；判断 AReaL 当前 rollout backend 是否存在同类 rank-order hazard
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [MoE](../topics/moe.md), [Distributed Training](../topics/distributed_training.md), [Rollout Latency](../playbooks/rollout_latency.md)

AInfer-PD 不建立独立 prefill/decode GPU pool，而是在同一批 accelerator 上共享权重和 KV state。它通过跨 rank 协调 P/D collective 发起顺序，并为 DeepEP 的 P/D path 隔离通信状态，使 attention parallel 与 expert communication 可以安全交错。

作者报告 fixed-workload rollout completion time：单节点相对关闭 P/D multiplexing 的同引擎降低 `7.1%-22.5%`，相对 SGLang 降低 `24.8%-32.9%`；双节点分别降低 `18.0%-35.3%` 与 `18.3%-31.8%`。这些是 paper-reported completion-time 结果，不等于完整 RL step 的同倍率加速。

### Multi-Tenancy Characterization：独占 Scaling Curve 不等于共享集群性能

- Signal ID：2026-09-05-002
- Source ID：arxiv:2609.00817
- First seen：2026-09-05 00:21:28
- 原始提交：2026-09-01 15:17:17，Asia/Shanghai；v2 2026-09-03
- Scan window：2026-09-01 11:31:29 ~ 2026-09-05 00:21:28
- Focus Match：P0 Focus
- 来源：arXiv primary page / SC26
- 类型：paper / distributed training / GPU cluster / multi-tenancy / network interference
- 链接：https://arxiv.org/abs/2609.00817
- Primary-source check：title、9 位作者、v1/v2 timestamp、SC26、最多 2400 GPUs、5 种并行策略、平台列表与 realistic noise model 均已对齐 arXiv metadata/abstract
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：这是直接面向生产集群的 scaling 证据。它同时比较 scale-up、scale-out、rack-scale、不同 parallelism 和 concurrent jobs，填补“benchmark 独占网络，但生产运行在 multi-tenant fabric”这一判断缺口。
- Status：NEW
- 建议动作：重点读 interference experiment、noise model、allocation scheme 和 parallelism sensitivity；提取能落到 scheduler、placement、network telemetry 与 noisy-neighbor diagnosis 的指标
- 关联主题：[Distributed Training](../topics/distributed_training.md), [NCCL](../topics/nccl.md), [Fault Tolerance](../topics/fault_tolerance.md), [Tensor Parallelism](../topics/tensor_parallelism.md)

论文覆盖 Alps、Leonardo、LUMI、JUPITER、NVL72 GB300 和 DGX A100，规模最高 2400 GPUs，并用五种 parallelization strategy 研究网络、计算能力和互连技术的联合作用。它的价值不是再给一条最好看的 MFU 曲线，而是回答哪些并行维度在共享网络下最容易被邻居任务放大抖动。

对平台工程的直接启示是：capacity planning 不能只存“模型 + GPU 数量”的基准；至少还要记录 placement、fabric domain、并行组跨域关系、并发 workload 和 rank arrival distribution，否则 scheduler 无法区分模型本身扩展差与网络干扰。

### TRL：把 1M-token SFT Recipe 与 Context Parallel 边界写成可运行入口

- Signal ID：2026-09-05-003
- Source ID：github:huggingface/trl@c61e556c
- First seen：2026-09-05 00:21:28
- 合入时间：2026-09-04 00:21:01，Asia/Shanghai
- Scan window：2026-09-01 11:31:29 ~ 2026-09-05 00:21:28
- Focus Match：P0 Focus
- 来源：Hugging Face TRL verified commit / code / documentation benchmark
- 类型：framework change / long-context SFT / context parallelism / FSDP2
- 链接：https://github.com/huggingface/trl/commit/c61e556c923e4a454a231b2caf224023eb57c195
- Primary-source check：verified merge commit、1M-token example、8xH100 config、step time/memory、RoPE scaling、chunked NLL、activation offload 与 model compatibility constraints 均已对齐 commit diff
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：这不是泛泛介绍 CP，而是一个能运行的 million-token SFT recipe，并明确写出哪些模型和 data semantics 不能用。它对当前 Qwen 128K SFT 配置比单纯的论文 benchmark 更有迁移价值。
- Status：NEW
- 建议动作：将配置与现有 Qwen3.5-9B 128K recipe 做逐项对照；先验证 attention pattern 兼容性，再考虑 CP size、activation offload 和 chunked loss
- 关联主题：[Long-context Training](../topics/long_context_training.md), [Context Parallelism](../topics/context_parallelism.md), [FSDP](../topics/fsdp.md), [FlashAttention](../topics/flashattention.md)

官方 example 用 8 张 H100 将 1,048,576-token Qwen3-8B 序列按 CP=8 切成每卡 131,072 tokens，并报告约 `373 s/step`、`56.2 GB/GPU`。配套 scaling 对照中，Qwen3-8B 从 1M/8 GPUs 的 `364 s` 扩到 2M/16 GPUs 的 `696 s`、4M/32 GPUs 的 `1346 s`；文档明确要求将这组三点用于观察 scaling ratio，不作为绝对时间横向比较。

更重要的是约束：当前 CP path 只能表达 full causal attention，不能保持 packed documents 的 block-diagonal mask，也拒绝 sliding-window、chunked 或 linear attention 模型。示例明确指出 Qwen3.5 及后续 hybrid-attention 模型不适用。这意味着对现有 Qwen3.5-9B 128K 任务，不能看到“TRL 支持 1M”就直接照搬；首先要证明每层 attention semantics 没被改变。

### slime：将 Streaming Rollout、请求级取消与 Prefix 恢复接入外部 SGLang

- Signal ID：2026-09-05-004
- Source ID：github:THUDM/slime#2272
- First seen：2026-09-05 00:21:28
- 合入时间：2026-09-03，GitHub merged PR
- Scan window：2026-09-01 11:31:29 ~ 2026-09-05 00:21:28
- Focus Match：P0 Focus
- 来源：slime merged PR / implementation / tests / live E2E training
- 类型：framework change / rollout / trajectory data path / cancellation / SGLang backend
- 链接：https://github.com/THUDM/slime/pull/2272
- Primary-source check：merge state/date、cumulative/incremental SSE、request-level abort、prefix retention、stream validation、45 focused tests 与 Qwen3-30B-A3B live E2E setup/results 均已对齐 PR description
- 影响等级：★★★★☆
- Decision：Read
- Reason：partial rollout 只有在 runtime 能“边生成边提交状态、精确取消单请求、保留已观察 prefix、拒绝残缺样本”时才是可靠能力；这项改动把它从 scheduler 概念落实成 trajectory contract。
- Status：NEW
- 建议动作：对照 AReaL trajectory schema、abort API、resume semantics 与 routed-expert metadata；优先复制 contract test，而不是先复制实现
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [Long-context Training](../topics/long_context_training.md), [Rollout Latency](../playbooks/rollout_latency.md)

改动让 external SGLang `/generate` 同时支持 cumulative 和 incremental stream；cancel 时只终止对应 HTTP request，已经收到的 token/logprob/top-p/routed-expert metadata 保留到 sample，之后只恢复被中断的 siblings。异常 EOF、长度不一致或 stream mode mismatch 会显式失败，避免不完整输出悄悄进入 reward/training。

Qwen3-30B-A3B live E2E 验证覆盖 SGLang TP4/EP4 inference、Megatron TP4/EP4 training、routing replay、真实 optimizer step 和 116/116 weight buckets 同步。`26.1 s` 与 `26.3 s` 表明 cumulative/incremental 模式性能基本持平；这项 signal 的证据是 correctness 和可组合性，不是已证明的端到端加速。

### NeMo RL：HybridEP 在 H100/B200 x86 多节点路径完成规模化兼容验证

- Signal ID：2026-09-05-005
- Source ID：github:NVIDIA-NeMo/RL#3436
- First seen：2026-09-05 00:21:28
- 合入时间：2026-09-02，GitHub merged PR
- Scan window：2026-09-01 11:31:29 ~ 2026-09-05 00:21:28
- Focus Match：P0 Focus
- 来源：NVIDIA NeMo RL merged PR / W&B validation records
- 类型：framework change / MoE / HybridEP / DeepEP / multi-node compatibility
- 链接：https://github.com/NVIDIA-NeMo/RL/pull/3436
- Primary-source check：merge state/date、x86_64/aarch64 dependency pin、`HYBRID_EP_MULTINODE` build-time requirement、NVLink-domain rule、H100/B200 validation matrix 与限制说明均已对齐 PR description
- 影响等级：★★★★☆
- Decision：Read
- Reason：它给出 HybridEP 从“能编译”到 32x8 H100、EP32 训练完成的工业级兼容证据，也暴露 build-time flag、NVLink-domain divisibility 和 dependency pin 这类真实部署边界。
- Status：NEW
- 建议动作：作为 DeepEP/HybridEP 部署清单阅读；不要把 20/20 steps 解读成独立性能收益，重点记录 binary provenance、topology constraint 与 validation matrix
- 关联主题：[MoE](../topics/moe.md), [Distributed Training](../topics/distributed_training.md), [NCCL](../topics/nccl.md), [Agentic RL](../topics/agentic_rl.md)

验证包括 Qwen3-30B-A3B 4x8 H100/EP8、Qwen3-235B-A22B 16x8 H100/EP16、Nemotron 3 Super 120B-A12B 32x8 H100/EP32，均完成 20/20 steps；Qwen3-30B-A3B 也在 4x8 B200 完成 20/20 steps。

PR 同时明确：多节点能力由 DeepEP build 时的 `HYBRID_EP_MULTINODE=1` 决定，recipe 不能事后改变已有 binary；每个 NVLink domain 以 8 ranks 计，EP group 必须能被 8 整除。作者也说明结果验证 dependency pin 的兼容性，不是对该 diff 的 isolated speedup benchmark。

### Headroom-Drift Replay：把 GRPO Replay 拆成“值得学”与“仍兼容”

- Signal ID：2026-09-05-006
- Source ID：arxiv:2609.03941
- First seen：2026-09-05 00:21:28
- 原始提交：2026-09-03 22:45:47，Asia/Shanghai
- Scan window：2026-09-01 11:31:29 ~ 2026-09-05 00:21:28
- Focus Match：P0 Focus
- 来源：arXiv primary page / COLM 2026
- 类型：paper / GRPO / replay / rollout cost / sample freshness
- 链接：https://arxiv.org/abs/2609.03941
- Primary-source check：title、2 位作者、v1 timestamp、COLM 2026、Headroom/Drift 两阶段选择、fresh stream 不变与 Agentic Search wall-clock claim 均已对齐 arXiv metadata/abstract
- 影响等级：★★★★☆
- Decision：Read
- Reason：它直接面对 agentic task 中 environment interaction 主导 wall-clock 的问题，并把 replay 本身从复杂 pipeline 中剥离出来；对 AReaL 来说，真正可迁移的是 replay buffer 的 selection/provenance contract。
- Status：NEW
- 建议动作：阅读 Headroom 和 Drift 的精确定义、replay ratio、importance correction 与 Agentic Search wall-clock breakdown；先验证训练语义，再谈省 rollout 成本
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [MOPD](../topics/mopd.md), [Rollout Latency](../playbooks/rollout_latency.md)

Headroom 对历史 group 按“剩余学习空间”排序，Drift 再判断它与当前 policy 是否仍兼容。fresh on-policy stream 保持不变，也不引入额外 generation/training machinery。作者报告在 math、multimodal 和 Agentic Search 上优于 naive replay，并在 Agentic Search 以更低 wall-clock 达到可比质量。

工程上不能只实现一个 replay queue。每个 group 至少要绑定 policy version、old logprob、reward/verifier version、environment version 和 tokenization provenance；否则 Drift 判断与训练 correction 没有可靠输入，省下的 rollout 可能换来不可解释的 off-policy 偏差。

## Observed / Rejected Candidates

| 材料 | Source ID | Focus Match | Decision | 原因 |
|---|---|---|---|---|
| [Hardware-Aware FP4 FlashAttention-4](https://arxiv.org/abs/2609.04105) | arxiv:2609.04105 | P0 | Observe | Direct-P 在 GB200 报告最高 `2.13x` BF16 forward、完整单卡 8B update 最高 `1.14x`，但所有测试的 MXFP4 probability/value distributed-training trajectory 均发散。重要判断是“FP4 GEMM 不自动等于 attention 提速”；目前单作者、缺少独立代码/复现证据，先观察。 |
| [Every Kernel Is a Join](https://arxiv.org/abs/2609.03905) | arxiv:2609.03905 | P1 | Observe | 用 join/aggregation decomposition 自动生成 multi-GPU exchange program，概念新颖且有单节点 LLaMA block 数据；距离生产级多节点并行编译器仍远。 |
| [DRACO](https://arxiv.org/abs/2609.04094) | arxiv:2609.04094 | P1 | Observe | 将 trajectory-level dynamic-rubric score 分配为 step-level GRPO advantage，能影响 reward pipeline，但当前主要贡献仍是 credit assignment 算法。 |
| [LeanGRPO](https://arxiv.org/abs/2609.03528) | arxiv:2609.03528 | P1 | Observe | 通过 Retain/Reweight 减少 diffusion RL 冗余 update 计算，最高报告 `1.83x`；不是当前 LLM/Agentic RL 主路径。 |
| [Latency-Aware Orchestration for Multi-Agent Workflows](https://arxiv.org/abs/2609.03335) | arxiv:2609.03335 | P1 | Observe | prediction-guided physical execution graph 与 heterogeneous GPU placement 有 serving 价值，但尚未改变 training/rollout 主线。 |
| [AReaL reserved-port ownership fix](https://github.com/areal-project/AReaL/commit/6feff6df3758907698b8ba7370594d80f5bbdfd4) | github:areal-project/AReaL@6feff6d | P1 | Observe | 将 forked worker 与 reserved port 的 ownership/rollback 做成原子 lifecycle，属于有价值的生产稳定性修复，但不是新架构。 |
| [verl material changes](https://github.com/verl-project/verl/commits/main/) | github:verl-project/verl@2026-09-01..04 | P1 | Observe | pluggable router、multi-node replica abort、VeOmni async activation offload 与新 GRPO example 值得后续按 release 聚合；本窗口缺少一项足够完整的独立 benchmark。 |
| [Megatron-FSDP heterogeneous MoE sharding](https://github.com/NVIDIA/Megatron-LM/commits/main/) | github:NVIDIA/Megatron-LM@2026-09-03 | P0 | Read | expert/non-expert parameter 可采用不同 ZeRO stage，并修复 MXFP8 offset correctness；非常贴近 MoE memory/communication，但本次先在 NVIDIA/runtime watch 保留，不扩成第七条 Accepted。 |

## OpenAI / Anthropic / NVIDIA / DeepSeek Watch

| 来源 | 本次结果 | Decision | 判断 |
|---|---|---|---|
| OpenAI | GPT-6 Astra safety overview/system card、Path to Astra safeguards、Daybreak security | Observed | 核心厂商报告需要看到，但本窗口公开页面重点是能力与安全治理，没有可迁移的 training、RL、serving runtime 或 cluster mechanism，因此不进入 Accepted。 |
| Anthropic | Claude Fable/Mythos 5.1 与 Enterprise Frontier Safeguards | Rejected / Observe | 模型与安全发布有工业信号，但公开信息未披露训练系统、post-training runtime 或规模化基础设施机制；不因厂商品牌自动收录。 |
| NVIDIA | NeMo RL HybridEP #3436；Megatron-FSDP heterogeneous MoE sharding；NVIDIA Developer Blog | **Accepted / Read** | HybridEP 有多节点 H100/B200 验证；Megatron-FSDP 的 expert/non-expert 分片策略值得继续读。Sep 1 博客以 inference sizing/agent application 为主，不另升 Accepted。 |
| DeepSeek | official Hugging Face organization model-card activity；API changelog | Observed / scan limitation | DeepSeek-V4-Flash-Vision-Exp model card 有 vLLM/SGLang 使用说明更新，但没有新的权重或技术报告；API changelog 本次无法稳定读取，下次从当前 cursor 回看该来源。 |

## Hugging Face Watch

- **TRL**：1M-token Context Parallel SFT recipe 是本次 Accepted/Deep Dive。它同时给出 benchmark、可运行 example 和模型兼容边界。
- **Hugging Face Blog**：窗口内 community/partner posts 较多，但未发现比 TRL primary code 更强的 Training/RL/Inference Infra 文章；不以 community 热度替代系统证据。
- **Transformers / Accelerate / PEFT / Kernels**：本窗口没有形成需要单独 Accepted 的 release。TRL example 依赖 Transformers main 的 checkpoint activation offload，这一依赖已在 signal 中保留。
- 判断：本次 HF 最值得读的是代码与 recipe，不是博客列表。

## RL Framework Watch

| Framework | Window 内可核验变化 | Decision | 对 AReaL 的判断 |
|---|---|---|---|
| AReaL | forked worker/reserved-port ownership、partial-start rollback 与 retryable lifecycle 修复 | Observe | 可直接借鉴 actor/process/port 三者的一致 ownership model；优先补 failure-injection test。 |
| verl | pluggable router、multi-node replica abort、VeOmni async activation offload、Qwen3.5 GRPO example | Observe | router plugin contract 和 abort propagation 值得对照；没有 benchmark 时不推断提速。 |
| slime | external SGLang streaming rollout + request-scoped abort + prefix resume | **Accepted / Read** | 最值得迁移的是 incremental/cumulative stream contract、unexpected EOF failure 和 partial-group resume tests。 |
| ROLL | 未发现 material release/merged change | Not found | 不用普通维护 commit 填充。 |
| OpenRLHF | 未发现 material release/merged change | Not found | 继续观察 vLLM integration、Ray placement 和 refit correctness。 |
| NeMo RL | HybridEP x86 multi-node；selected-token logprob 直接计算；async GRPO correctness fixes | **Accepted / Read** | HybridEP validation matrix 是最强证据；其余变更等待 release/benchmark 聚合。 |
| TRL | million-token SFT/CP recipe | **Accepted / Deep Dive** | 虽非 rollout framework 核心能力，但直接影响 post-training long-context data path；Qwen3.5 hybrid attention 是明确边界。 |

## Adjacent Runtime Watch

| Runtime | 变化 | Decision | 工程判断 |
|---|---|---|---|
| Megatron-LM | Megatron-FSDP 为 expert/non-expert parameter 配置不同 sharding strategy，并修复 MXFP8 offset 问题 | Read | MoE state 不应被迫采用单一 ZeRO stage；正确性修复比配置自由度更值得先核对。 |
| vLLM | fast-start、KV connector/P-D lifecycle、Qwen3.8/GLM5.3/DeepSeek-V4 follow-up | Observe | 变化较多但缺少单一足够强的新端到端证据；按 release 聚合。 |
| SGLang | HiCache sidecar/metrics、CPU simulator、decode CP、PD/KV lifecycle fixes | Observe | 与 slime streaming contract 相互作用最值得后续测试；本次不把常规 runtime 活跃度拆成多个 signal。 |

## Reading Queue 判断

- [ ] **第一优先：AInfer-PD。** 先看 Figure/algorithm 中 P/D collective ordering 与 DeepEP state isolation，再判断它是否能映射到 AReaL 的 SGLang/vLLM backend；预计 45-60 分钟。
- [ ] **第二优先：TRL 1M-token recipe。** 不需要通读文档，直接对照当前 Qwen3.5-9B 128K 脚本，回答“哪些优化可迁移、哪些因 hybrid attention 不能用”；预计 30 分钟。
- [ ] **第三优先：multi-tenancy characterization。** 只精读 interference methodology、topology/parallelism sensitivity 和 scheduler implication；预计 45 分钟。

现有 [P0 Reading Queue](../reading_queue/P0.md) 已满，本次不自动覆盖正在执行的阅读任务。Headroom-Drift Replay、slime streaming 和 NeMo HybridEP 作为后续 P1 替换候选，不把所有 Accepted 都塞进队列。

## 去重记录

- 新增 Accepted Source ID：`arxiv:2609.00993`、`arxiv:2609.00817`、`github:huggingface/trl@c61e556c`、`github:THUDM/slime#2272`、`github:NVIDIA-NeMo/RL#3436`、`arxiv:2609.03941`。
- TRL 1M recipe 与 8 月 Megatron variable-length packing 都属于 long-context training，但前者新增了 FSDP2/CP 的 runnable SFT entry、1M/4M benchmark 和 model compatibility boundary，因此独立收录。
- slime streaming 与既有 partial rollout/replay signal 不重复：本次新增的是 request-scoped abort、observed-prefix preservation 和 malformed-stream rejection 的具体 contract。
- NeMo HybridEP 的结果只解释为 pinned dependency/runtime compatibility，不重复包装为新的 HybridEP 算法或 isolated speedup。

## 下一步

- [ ] 下一次扫描从 `2026-09-05 00:21:28` 开始，按 Source ID 去重，并补查本次不稳定的 DeepSeek API changelog。
- [ ] 如果只读一篇，读 AInfer-PD；它最直接改变对 Agentic RL rollout workload 的建模方式。
- [ ] TRL recipe 读完后，把对 Qwen3.5 hybrid attention 的适用边界更新到 [Long-context Training](../topics/long_context_training.md)，但先验证现有 verl/Transformers path，不能只据 TRL 文档泛化。

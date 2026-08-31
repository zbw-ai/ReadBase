# Frontier Scan, 2026-08-24

- Previous scan：[2026-08-20](frontier_scan_2026-08-20.md)
- Window：2026-08-20 10:21:56 ~ 2026-08-24 09:32:12
- Timezone：Asia/Shanghai
- Generated at：2026-08-24 09:32:12
- Report type：flexible frontier scan
- Sources scanned：arXiv cs.AI / cs.LG / cs.CL / cs.DC / cs.PF / cs.AR / stat.ML；OpenAI / Anthropic / NVIDIA / DeepSeek official sources；Hugging Face Blog / TRL / Transformers / Accelerate / PEFT / Kernels；AReaL / verl / slime / ROLL / OpenRLHF / NeMo RL；Megatron-LM / vLLM / SGLang official default-branch changes
- Scan completeness：arXiv API 触发 rate limit，回退七个分类的 official recent pages，解析并去重 1,370 条候选记录；Accepted paper 的 title / authors / date / abstract claims 已逐条对齐 arXiv primary page。GitHub 以 official Atom、merged/default-branch commit、PR description 和测试/benchmark 交叉核验。扫描截止时刻冻结在检索开始前，晚于该时刻的变更留给下一次。

## 本次核心判断

这次最强的信号是：**RL 与长上下文 serving 正在从“固定资源分区 + 全量状态同步”转向可切换资源、稀疏状态传输和 workload-aware routing。** verl 已经把空闲 trainer GPU 临时切到 rollout，AReaL 与 vLLM 分别把 weight update 压成 AdamW delta 和 destination-owned shard，NeMo RL 则把 trajectory data plane 落到 CPU RDMA。与此同时，FlashPrefill V2、CacheRoute 和 ReCache 表明长上下文优化已经越过单 kernel 阶段，开始共同处理 sparse attention、paged KV、continuous batching、prefix routing 与 agent resource schema reuse。

另一个值得长期关注的变化来自 NVIDIA MaxLPS：AI factory 的 power headroom 不再只是机房规划参数，而开始成为可被 telemetry、policy 和 control loop 动态调度的系统资源。

## Accepted Frontier Signals

### FlashPrefill V2：Block-Sparse Prefill 开始具备 Production Backend 条件

- Signal ID：2026-08-24-001
- Source ID：arxiv:2608.19758
- First seen：2026-08-24 09:32:12
- 发布时间：2026-08-20 16:02:55，Asia/Shanghai
- Scan window：2026-08-20 10:21:56 ~ 2026-08-24 09:32:12
- Focus Match：P0 Focus
- 来源：arXiv primary page / paper
- 类型：paper / long-context serving / sparse attention kernel
- 链接：https://arxiv.org/abs/2608.19758
- Primary-source check：title、5 位作者、v1 date、PackGQA、warp specialization、ping-pong pipeline、FP8、paged KV cache、continuous batching 与 H20 数字已对齐 arXiv metadata/abstract
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：它补齐了 sparse-prefill 从算法原型走向 runtime backend 的关键缺口，不再只比较 attention microbenchmark，而是显式适配现代 serving 的 KV layout、batching 和量化约束。
- Status：NEW
- 建议动作：优先读 operator contract、paged KV integration 和 sparsity/accuracy gate，再判断是否值得进入 SGLang backend 对照实验
- 关联主题：[FlashAttention](../topics/flashattention.md), [Long-context Training](../topics/long_context_training.md), [Agentic RL](../topics/agentic_rl.md)

作者将第一版 FlashPrefill 的动态稀疏模式发现扩展为可部署路径：mean correction 控制高稀疏率误差，kernel 侧对齐 FlashAttention-3/4 的执行结构，并原生支持 FP8、paged KV cache 和 continuous batching。论文在 H20、128K context 上报告相对 FlashAttention-2 的最高 `47.26x` FP8 与 `27.19x` BF16 speedup；这些是作者报告的特定 attention workload 数字，不能直接等同于端到端 agent throughput。

### CacheRoute：Prefix Cache 命中率必须和负载倾斜一起规划

- Signal ID：2026-08-24-002
- Source ID：arxiv:2608.19677
- First seen：2026-08-24 09:32:12
- 发布时间：2026-08-20 14:12:47，Asia/Shanghai
- Scan window：2026-08-20 10:21:56 ~ 2026-08-24 09:32:12
- Focus Match：P0 Focus
- 来源：arXiv primary page / paper
- 类型：paper / LLM serving scheduler / prefix cache routing
- 链接：https://arxiv.org/abs/2608.19677
- Primary-source check：title、author、v1 date、periodic routing plan、warm set、60×H100、p99 SLO、QPS/hit-rate 数字与 counterexample 已对齐 arXiv metadata/abstract
- 影响等级：★★★★★
- Decision：Read
- Reason：它把 prefix-aware routing 从静态 hash/affinity 提升为受 load constraint 约束的周期计划，并主动给出 affinity 不值得开启的反例。
- Status：NEW
- 建议动作：重点看 shadow replay、hot-key replication threshold 和 plan refresh cadence，不只看 `2.3x` 数字
- 关联主题：[Distributed Training](../topics/distributed_training.md), [Long-context Training](../topics/long_context_training.md), [Agentic RL](../topics/agentic_rl.md)

在 60 张 H100、Llama-3.3-70B FP8 workload 中，论文报告 3.5 秒 p99 SLO 下 `176±11 QPS`，是最强 baseline 的 `2.3x`，served KV hit rate 从 `64.1%` 提升到 `93.2%`。更重要的是作者明确展示 32B workload 的负结果：当恢复的 KV 计算不足以抵消负载倾斜时，affinity 会降低收益，因此上线前必须用真实 trace 做 shadow replay。

### ReCache：Agent Tool Schema 不应被当成普通 Prefix

- Signal ID：2026-08-24-003
- Source ID：arxiv:2608.19662
- First seen：2026-08-24 09:32:12
- 发布时间：2026-08-20 13:57:24，Asia/Shanghai
- Scan window：2026-08-20 10:21:56 ~ 2026-08-24 09:32:12
- Focus Match：P0 Focus
- 来源：arXiv primary page / released code
- 类型：paper / agent inference / KV reuse and compression
- 链接：https://arxiv.org/abs/2608.19662
- Primary-source check：title、4 位作者、v1 date、resource-wise attention、resource-local position、layer/head-group routing、pruning、TTFT/KV-memory 数字与 code claim 已对齐 arXiv metadata/abstract
- 影响等级：★★★★☆
- Decision：Read
- Reason：tool/skill schema 会跨请求复用但顺序和组合不断变化，普通 prefix cache 无法命中；ReCache 把资源编码变成 composition-invariant KV block，直接命中长时 agent inference 的结构性浪费。
- Status：NEW
- 建议动作：检查 resource-wise attention 是否需要改模型、资源隔离如何影响 cross-resource interaction，以及代码能否接入现有 serving backend
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [Long-context Training](../topics/long_context_training.md)

论文报告 invocation F1 `82.3%` 对 dense baseline `82.4%`，同时 TTFT 提升 `3.655x`、allocated KV tensor memory 降低 `92.43%`、attention 加速 `1.423x`。这些数字说明“复用 schema 编码”和“选择性访问资源”可以拆开优化；但 resource-wise attention 改变了 attention visibility，不能只把它当成无侵入 cache plugin。

### verl Separate Async：让空闲 Trainer GPU 临时参与 Rollout

- Signal ID：2026-08-24-004
- Source ID：github:verl-project/verl#7373
- First seen：2026-08-24 09:32:12
- Scan window：2026-08-20 10:21:56 ~ 2026-08-24 09:32:12
- Focus Match：P0 Focus
- 来源：verl merged/default-branch implementation
- 类型：framework implementation / resource scheduling / rollout
- 链接：https://github.com/verl-project/verl/pull/7373
- Primary-source check：switch state machine、adaptive threshold、sleep/wake lifecycle、24×H100 配置、32K response、150-step wall-clock/tokens-per-second 与 staleness 数字已对齐 merged commit description/tests
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：这是 BiDiRL 类“训练资源借给生成”的思路进入主流 RL framework 的可运行实现，而且给出了明确的适用资源比例、staleness 变化和失效边界。
- Status：NEW
- 建议动作：与 AReaL 当前 colocate/separation scheduler 对照 switch cost、buffer starvation、weight freshness 和 vLLM sleep/wake 生命周期
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [Rollout Latency](../playbooks/rollout_latency.md), [Distributed Training](../topics/distributed_training.md)

该实现允许 `separate_async` trainer 在 prompts 提交后继续保持 trainer replicas 的 rollout mode，直到 replay buffer 拥有足够 sampleable groups，再回收 GPU 训练。作者在 3×8 H100、Qwen3.5-35B-A3B、最大 response 32K 的配置上报告 150 steps wall clock 从 18.80 小时降至 16.43 小时，tokens/s 提升 `10.1%`；mean staleness 从 `0.522` 增至 `0.556`。收益高度依赖 trainer GPU 占比和跨 step 生成速率稳定性，功能默认关闭且暂不兼容 PD disaggregation。

### AReaL AWEX：AdamW Delta Weight Transfer 合入 Separation Mode

- Signal ID：2026-08-24-005
- Source ID：github:areal-project/AReaL#1623
- First seen：2026-08-24 09:32:12
- Scan window：2026-08-20 10:21:56 ~ 2026-08-24 09:32:12
- Focus Match：P0 Focus
- 来源：AReaL merged/default-branch implementation
- 类型：framework implementation / weight sync / sparse delta transfer
- 链接：https://github.com/areal-project/AReaL/pull/1623
- Primary-source check：AdamW pre-step reconstruction、periodic full anchor、rank-consistent protocol、fallback conditions、103 focused tests 与 100-step Qwen3-30B-A3B run 已对齐 merged PR
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：它不再每次同步完整模型，而是从分布式 AdamW state 重构变化并传输 sparse delta；这正面处理大模型异步 RL 中越来越显著的 weight-sync data-plane 成本。
- Status：NEW
- 建议动作：先理解 correctness protocol 和 full-sync fallback，再测 delta sparsity、anchor interval 与 end-to-end sync 占比
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [Distributed Training](../topics/distributed_training.md), [Checkpointing](../topics/checkpointing.md)

初始同步和周期 anchor 仍走 full weight，中间连续版本才使用 delta；缺失/歧义 optimizer state、跳步、版本不连续或不支持的 precision state 都回退 full sync。接收端 apply 成功之后才推进 tracker 和 detector watermark，失败不会误提交版本。当前范围仅覆盖 Megatron-v2/SGLang-v2 separation、AdamW、无 LoRA、单 minibatch 和 power-of-two combined world size，因此它更像一个正确性优先的第一版，而不是通用 delta protocol。

### vLLM Sharded RDT：Inference Worker 只拉取自己真正消费的参数切片

- Signal ID：2026-08-24-006
- Source ID：github:vllm-project/vllm#43375
- First seen：2026-08-24 09:32:12
- Scan window：2026-08-20 10:21:56 ~ 2026-08-24 09:32:12
- Focus Match：P0 Focus
- 来源：vLLM merged/default-branch implementation
- 类型：runtime implementation / weight sync / NIXL and Ray Direct Transport
- 链接：https://github.com/vllm-project/vllm/pull/43375
- Primary-source check：lazy loader-op recording、destination slice pull、TP/EP ownership、NIXL/RDT transport、142 tests、Qwen3-235B/GLM-4.5-Air/Kimi-K2 validation 与 sync 数字已对齐 merged PR
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：它把 weight sync 从“每个 inference worker 接收整参数”改成 destination-owned slice transfer，对超大 MoE 的 train→serve 重分片尤其关键。
- Status：NEW
- 建议动作：精读 lazy op-chain wire format、unsupported-op fail-fast、trainer ownership metadata 和 layerwise buffer lifecycle
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [Distributed Training](../topics/distributed_training.md), [MoE](../topics/moe.md)

vLLM 在初始化阶段用 zero-storage placeholder 记录原生 weight loader 执行的 `narrow/t/view` 等操作链，后续 inference worker 通过 NIXL/RDT 只向持有对应参数的 trainer rank 拉取目标 slice，不在 worker 上物化完整 HF tensor。作者在 8 trainer GPU→8 inference GPU、Qwen3-235B-A22B 上报告 472 GB warm sync：DP8+EP8 为 3.3 秒，TP8 为 6.1 秒。数字来自单一硬件和 layout，但设计解决的是更普遍的 source/destination sharding mismatch。

### SGLang DeepSeek-V4：Q8KV8 Sparse MLA Prefill 真正接入 Runtime

- Signal ID：2026-08-24-007
- Source ID：github:sgl-project/sglang#32327
- First seen：2026-08-24 09:32:12
- Scan window：2026-08-20 10:21:56 ~ 2026-08-24 09:32:12
- Focus Match：P0 Focus
- 来源：SGLang merged PR / benchmark / regression tests
- 类型：runtime implementation / DeepSeek-V4 / FP8 sparse MLA
- 链接：https://github.com/sgl-project/sglang/pull/32327
- Primary-source check：merge date、runtime dispatch、Q-head/sink padding、input contract、TP8 setup、H20 throughput/TTFT 与 GSM8K/LongBench-v2 numbers 已对齐 merged PR
- 影响等级：★★★★☆
- Decision：Read
- Reason：DeepSeek-V4 已有 FP8 KV cache，但 prefill 仍可能回到 BF16 sparse path；该变更补上 FP8 query × FP8 KV 的完整 runtime dispatch 和真实 serving shape 防 hang 校验。
- Status：NEW
- 建议动作：对照 `flashmla_sparse` 与 `flashmla_sparse_q8` 的数据转换、workspace 和 padding 成本，关注长 context 下收益是否覆盖 dequant/requant
- 关联主题：[FlashAttention](../topics/flashattention.md), [FP8](../topics/fp8.md), [Long-context Training](../topics/long_context_training.md)

H20、DeepSeek-V4-Flash、TP8、FP8 KV、CUDA graph 默认 serving 下，PR 报告 Q8KV8 相对 BF16 sparse backend 的 input throughput/TTFT 提升约 `4.4%-8.2%`，且 10K-200K context 的 LongBench-v2 子集绝对分数变化为 `-0.006`。这里最有价值的不是个位数 speedup，而是 dtype/layout contract、C0/C4/C128 behavior 和此前真实 hang shape 都进入 regression coverage。

### NeMo RL Mooncake CPU RDMA：Trajectory Data Plane 在 256 GPU 上通过验证

- Signal ID：2026-08-24-008
- Source ID：github:NVIDIA-NeMo/RL@ffbf33f
- First seen：2026-08-24 09:32:12
- Scan window：2026-08-20 10:21:56 ~ 2026-08-24 09:32:12
- Focus Match：P0 Focus
- 来源：NeMo RL default-branch commit / PR validation
- 类型：framework implementation / data plane / CPU RDMA
- 链接：https://github.com/NVIDIA-NeMo/RL/commit/ffbf33f3099847b6685072d4eb251bae605ff8de
- Primary-source check：RoCE device detection、no-TCP-fallback semantics、per-process pinned-memory sizing、256-GPU DeepSeek-V3 run 与 metric checks 已对齐 default-branch commit/PR evidence
- 影响等级：★★★★☆
- Decision：Read
- Reason：它暴露了大规模 RL data plane 的两个生产坑：silent TCP fallback 会制造虚假 RDMA 覆盖，per-GPU pinned-memory 默认值会在节点级放大成 TiB 级常驻内存。
- Status：NEW
- 建议动作：把 transport identity、registered-memory budget 和 no-fallback health check 纳入 AReaL trajectory queue 的观测项
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [NCCL](../topics/nccl.md), [Distributed Training](../topics/distributed_training.md)

实现将 `mooncake_cpu` 明确定义为 CPU RDMA，找不到合适 RoCE device 直接失败，不再静默退化 TCP；同时把每 client process 的 segment/buffer 默认值从 512+64 GiB 修正为 64+4 GiB，避免 8 GPU 节点把注册内存放大到约 4.6 TiB。作者给出 2-node/8-GPU 与 32-node/256-GPU DeepSeek-V3 各 10 steps 的验证。长期稳定性仍未被 10-step run 证明，但这是一条足够具体的生产实现信号。

### NVIDIA DSX MaxLPS：Power Headroom 进入 AI Factory Control Loop

- Signal ID：2026-08-24-009
- Source ID：blog:nvidia/dsx-maxlps-2026-08-21
- First seen：2026-08-24 09:32:12
- Scan window：2026-08-20 10:21:56 ~ 2026-08-24 09:32:12
- Focus Match：P0 Focus
- 来源：NVIDIA Technical Blog / official docs
- 类型：industrial report / AI factory / power and thermal scheduling
- 链接：https://developer.nvidia.com/blog/maximizing-ai-factory-performance-per-watt-with-nvidia-dsx-maxlps/
- Primary-source check：publish date、Dynamic Power Software control loop、rack/site topology、Developer Preview 状态、GB200/Vera Rubin validation 与厂商数字已对齐 NVIDIA official post
- 影响等级：★★★★★
- Decision：Read
- Reason：它把 power、cooling 和 rack headroom 从静态 capacity planning 变成 fleet telemetry + policy 驱动的动态调度对象，直接影响未来超大集群 scheduler 与 observability 的边界。
- Status：NEW
- 建议动作：先读 DPS control loop、resource-group budget 和 failure policy，再把厂商 performance-per-watt 数字与独立验证分开记录
- 关联主题：[Distributed Training](../topics/distributed_training.md), [Fault Tolerance](../topics/fault_tolerance.md), [Checkpointing](../topics/checkpointing.md)

NVIDIA 描述 DPS 持续采集 GPU/rack/group power telemetry，在固定 site envelope 内重新分配未使用 headroom，并通过 policy 验证和 emergency response 保持约束。官方报告 GB200 NVL72 与 Vera Rubin NVL72 代表性 inference workload 在近似保持吞吐时，performance/W 提升约 `1.5x` 与 `1.3-1.4x`，并宣称固定 power budget 下可容纳更多 rack。DPS 当前为 Developer Preview，数字属于 vendor-reported industrial evidence，不能视为通用 workload 保证。

## Observed / Rejected Candidates

| 材料 | Source ID | Focus Match | Decision | 原因 |
|---|---|---|---|---|
| The Lazy Pod That Lies | arxiv:2608.19412 | P0 | Read | 对 eStargz/SOCI、2-140 GB image 做真实 KServe 测量，揭示 lazy pull 把成本推迟到 first read，并可能在 snapshotter cache 耗尽时让 Running pod 持续“假健康”；运维价值高，但离当前 RL data path 主线稍远。 |
| Compute-Efficient HPO Transfer for Large-Scale MoE | arxiv:2608.20061 | P0 | Read | 用 MoE-MLA-Muon 的 μP width transfer + token scaling law 预测 10T-token 最优 LR，并用于 155B/17B-active 预训练；值得作为大规模 recipe 证据精读，但暂未公开系统/代码细节。 |
| SAPO | arxiv:2608.19842 | P0 | Read | 单 backbone 在 causal boundary 分别输出 policy/value，以单 rollout、trajectory GAE 减少独立 critic memory；当前证据来自 1.5B/7B ALFWorld/WebShop，主要改变算法而非 runtime。 |
| EnvHarness | arxiv:2608.19880 | P1 | Read | 用可插拔 harness 包装静态环境并保留原 verifier，再由 EnvRigger 根据 trajectory 生成组件；与 Agent Lightning 的 harness ownership 相呼应，但不是 infra implementation。 |
| R2-OPD | arxiv:2608.19408 | P1 | Observe | 当 teacher reward 与 reasoning progress ranking 冲突时过滤 distillation reward；能补 MOPD reward diagnostics，但仍是算法层。 |
| DeepSeek-V4-Flash-Vision-Exp | vendor:deepseek-api-2026-08-21 | P0 | Read | DeepSeek API 发布 multimodal agent experimental model，并披露多个 agent benchmark；没有权重、model card 或训练/serving mechanism，作为核心厂商方向信号保留但不升级 Accepted。 |
| LFM2.5-DSpark | blog:huggingface/LiquidAI/lfm25-dspark | P1 | Read | 发布约 300M draft checkpoints，并在 H100/SGLang 与 M4 Max/llama.cpp 报告最高 3.18x/2.87x；属于厂商发布，机制与代码可核验，但不改变当前 RL runtime 主线。 |
| Transformers DTensor TP API | github:huggingface/transformers#47579 | P1 | Read | 将 TP inference/training API 做进 Transformers，并同步 evaluation CP；通用性重要，但本次未取得独立性能证据。 |
| Accelerate FSDP2 correctness | github:huggingface/accelerate@fd01e35+f801260 | P1 | Observe | 修正 activation checkpoint wrap 粒度与 tied-embedding DTensor placement；属于高价值 correctness 修复，不单独升级为前沿架构信号。 |
| Megatron heterogeneous DDP overlap for MIMO | github:NVIDIA/Megatron-LM#6284 | P1 | Read | 允许 multimodal submodule 使用不同 parameter-gather overlap policy，同时进入 checkpoint/rerun/paged-stash lifecycle；范围明确但暂无公开端到端 benchmark。 |

## OpenAI / Anthropic / NVIDIA / DeepSeek Watch

| Vendor | Sources checked | Decision | 结果 |
|---|---|---|---|
| OpenAI | official RSS / research index / releases | Observe | 本窗口发布 AI Futures 与企业案例，没有新的 Training/RL/Inference Infra 技术报告；AI Futures 属政策/研究议程，不进入当前主线。 |
| Anthropic | official research / newsroom | Not found | 最新可核验 research/news 仍早于本窗口；站点 `updatedAt` 不是文章发布时间，不据此制造新信号。 |
| NVIDIA | Technical Blog / NeMo RL / Megatron-LM | **Accepted / Read** | MaxLPS、NeMo RL CPU RDMA 进入 Accepted；Megatron MIMO heterogeneous DDP overlap 保留 Read。厂商数字均标注为 vendor-reported。 |
| DeepSeek | API changelog / official Hugging Face organization | **Observed / Read** | 8 月 21 日 API 发布 DeepSeek-V4-Flash-Vision-Exp；官方 HF 仍以 8 月 13 日 V4-Pro 为最新开放权重。本窗口无对应 model card/weights，SGLang Q8KV8 属第三方 runtime 实现。 |

## Hugging Face Watch

| Sources checked | Decision | 结果 |
|---|---|---|
| Hugging Face Blog | Read | LiquidAI 发布 LFM2.5-DSpark draft checkpoints 与 SGLang/llama.cpp 实现；HF 官方团队本窗口没有新的 Training/RL Infra 深度文章。 |
| TRL | Observe | 主要是 Async trainer/VLM regression tests、GRPO/RLOO multimodal correctness 与 config 修复；未出现继 AsyncDistillationTrainer 后的新架构变化。 |
| Transformers | Read | DTensor TP inference/training API 与 evaluation path CP 合入；值得后续与 Megatron/HF TP contract 对照。 |
| Accelerate / PEFT | Observe | FSDP2 activation-checkpoint/tied-embedding correctness、ZeRO-3 LoRA partitioned-shape 修复；属于重要 compatibility 信号。 |
| Kernels | Observe | kernel resolver infrastructure 进入 default branch feed，但关联 PR 仍显示 open；仅记录方向，不把未稳定的 resolver contract 当作已发布能力。 |

## RL Framework Watch

| Framework | Release / PR | 子系统 | 核心变化 | 证据 | 对 AReaL 的参考 | Decision |
|---|---|---|---|---|---|---|
| AReaL | [#1623](https://github.com/areal-project/AReaL/pull/1623) | weight sync | separation AdamW sparse delta + periodic full anchor | merged PR、103 tests、100-step run | 当前项目可直接测 delta sparsity、anchor interval 和 fallback ratio | **Accepted / Deep Dive** |
| verl | [#7373](https://github.com/verl-project/verl/pull/7373) | scheduler / rollout | idle trainer GPU 切换到 rollout，adaptive buffer threshold | merged implementation、24-H100 benchmark、staleness metrics | 与 AReaL resource scheduler/BiDiRL 做状态机对照 | **Accepted / Deep Dive** |
| slime | [#2085](https://github.com/THUDM/slime/pull/2085) | OPD correctness | teacher logprob 按 rollout temperature 而非 0 计算 | default-branch fix | 增加 teacher/student sampling-temperature parity check | Observe |
| ROLL | default branch / release | hardware | 仅有 Ascend install update | official feed | 不改变当前 GPU RL runtime | Ignore |
| OpenRLHF | default branch / release | - | 本窗口未发现重大架构、性能或 correctness 变化 | official feed | 无新增可迁移项 | Not found |
| NeMo RL | [ffbf33f](https://github.com/NVIDIA-NeMo/RL/commit/ffbf33f3099847b6685072d4eb251bae605ff8de), [#3410](https://github.com/NVIDIA-NeMo/RL/pull/3410), [#2884](https://github.com/NVIDIA-NeMo/RL/pull/2884) | data plane / async trainer | CPU RDMA、async PPO、async colocated GRPO with Megatron inference | default-branch commits、tests、256-GPU validation | 重点对照 transport identity、pinned-memory budget 与 PPO/GRPO 共用 scheduler contract | **Accepted / Read** |

## Adjacent Runtime Watch

| Runtime | 变化 | Decision | 工程判断 |
|---|---|---|---|
| vLLM | sharded RDT P2P weight sync；native sparse checkpoint update | **Accepted / Deep Dive** | destination-owned slice transfer 比通用 broadcast 更适合 TP/EP/MoE train→serve 重分片；sparse checkpoint update 是同一数据面演进的后续。 |
| SGLang | DeepSeek-V4 Q8KV8 sparse MLA prefill backend | **Accepted / Read** | kernel 只有进入 dtype/layout/runtime dispatch 和真实 serving regression 后才构成可用能力。 |
| Megatron-LM | MIMO heterogeneous DDP overlap；GTP symmetric-memory ReduceScatter；distributed Muon auto-TP | Read / Observe | 多模态训练开始允许 submodule-specific communication policy；GTP/Muon 变化需等待公开 benchmark 再升级。 |

## Reading Queue 判断

- [ ] **今天只读一个：verl #7373。** 它和刚读过的 BiDiRL 最接近，先回答“何时借 trainer GPU、何时归还、switch cost 如何计入、staleness 为何只小幅上升”。
- [ ] **第二优先：AReaL #1623。** 重点看 full anchor、AdamW inversion、version commit 和 unsafe fallback；这是最贴近当前项目代码的实现。
- [ ] FlashPrefill V2、vLLM Sharded RDT、MaxLPS 保留 P1，不把 9 条 Accepted 全部塞进本周队列。

## 去重记录

- 新增 Accepted Source ID：`arxiv:2608.19758`、`arxiv:2608.19677`、`arxiv:2608.19662`、`github:verl-project/verl#7373`、`github:areal-project/AReaL#1623`、`github:vllm-project/vllm#43375`、`github:sgl-project/sglang#32327`、`github:NVIDIA-NeMo/RL@ffbf33f`、`blog:nvidia/dsx-maxlps-2026-08-21`。
- SGLang #32327 在上一扫描 cutoff 后约两分钟合入，本次按 default-branch merge 正式收录；不回写 08-20 报告。
- NVIDIA Nemotron QAD、ModelExpress 等文章虽在本窗口的 feed 批量更新中出现，但其核心材料已在前序 scan 收录或观察，不重复计数。

## 扫描完整性

- arXiv：API rate limit 后回退 cs.AI / cs.LG / cs.CL / cs.DC / cs.PF / cs.AR / stat.ML official recent pages；解析 1,370 条去重候选，并按窗口、主题和 Source ID 过滤。Accepted paper 均核对 metadata、authors、date、abstract mechanism 和数字。
- Core vendors：OpenAI RSS/research、Anthropic research/news、NVIDIA Technical Blog/代码栈、DeepSeek API changelog/HF organization 均显式检查。
- Frameworks：AReaL、verl、slime、ROLL、OpenRLHF、NeMo RL 均检查 official default-branch feed；重大项进一步核对 PR、tests 和 benchmark。
- Hugging Face：Blog、TRL、Transformers、Accelerate、PEFT、Kernels 已检查；vendor/community post 不自动升级。
- Adjacent runtime：Megatron-LM、vLLM、SGLang 已检查；高活跃仓库 Atom 只保留有限 commit，普通未合并 PR 仍可能漏检。
- 边界：扫描截止时刻固定为 `2026-08-24 09:32:12`；晚于该时刻的 arXiv announcement、vendor update 或 default-branch merge 留给下一次。
- 下一游标：`2026-08-24 09:32:12`。

## 下一步动作

- [x] 更新 [Scan Log](scan_log.md) 与 [Tracking README](README.md)。
- [ ] 精读 verl #7373，并把 resource switch state machine 与 BiDiRL/AReaL 做一页对照。
- [ ] 阅读 AReaL #1623 的 protocol safety 与 fallback tests，决定是否在当前项目做 weight-sync microbenchmark。
- [ ] 后续月报只从本次 9 条中保留真正改变判断的 3-5 条，不按 Accepted 数量机械搬运。
- [ ] 下一次扫描从 `2026-08-24 09:32:12` 开始，继续按 Source ID 去重。

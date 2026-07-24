# Frontier Scan, 2026-07-24

- Previous scan：[2026-07-22](frontier_scan_2026-07-22.md)
- Window：2026-07-22 12:57 ~ 2026-07-24 09:52
- Timezone：Asia/Shanghai
- Generated at：2026-07-24 09:52
- Report type：flexible frontier scan
- Sources scanned：arXiv cs.DC / cs.LG / cs.AI / cs.CL / cs.AR / cs.PF / cs.NI recent records；OpenAI / Anthropic / NVIDIA / Hugging Face official sources；AReaL / verl / slime / ROLL / OpenRLHF / NeMo RL releases and major PRs
- Scan completeness：合并 7 月 23 日记录与从 `2026-07-23 15:55` 到本次结束时刻的增量扫描。覆盖当前可见的 2026-07-23 arXiv 公告，并逐条回到 arXiv abs 页核对 accepted signal。GitHub 框架检查覆盖正式 release、merged PR 与对应 patch；open PR 只作为候选能力观察，不写成已发布功能。GitHub REST API 在后半程触发匿名 rate limit，已回退到官方 PR patch、release Atom 与项目页面复核。

## 本次核心判断

合并窗口共有七个值得保留的系统信号，其中前五个来自 7 月 23 日记录，后两个来自本次增量：

1. **万亿参数 MoE 的 full-parameter post-training 已经成为独立的系统工程问题。** SLAI T-Rex 把并行策略、计算通信编排和 kernel 优化串成完整 Ascend SuperPOD 路径，而不是只给出单算子 benchmark。
2. **MoE overlap 正在从 layer/operator 粒度下沉到 tile 粒度。** persistent compute producer 与专用 SM communication consumer 可以在 expert tile 完成时立即发起 return all-to-all，缩短暴露在关键路径上的通信。
3. **MoE 网络优化未必需要动态重配物理拓扑。** MoX 说明 token-aware multicast 与离线静态路由也能适配 runtime-dependent expert traffic，为 direct-connect fabric 提供另一条工程路线。
4. **更激进的 KV cache reuse 会引入新的正确性和安全边界。** position-independent reuse 不能只按 token chunk 命中，因为缓存状态仍编码了产生它时的上文。
5. **MoE 性能越来越取决于 rack-scale hardware/software co-design。** NVIDIA 的 GB300 NVL72 数据把 NVLink all-to-all、scale-out gradient traffic 与 Megatron-Core/TorchTitan/JAX 软件优化放进了同一个 delivered-throughput 模型。
6. **KV-cache eviction 不能只报告“平均精度没掉”，还需要说明这一步究竟破坏了多少 attention state。** Randomized eviction 使 serving runtime 能对 cache-induced error 做逐步归因，为自适应重算和预算升级提供可观测信号。
7. **RL rollout backend 正在从 vLLM/SGLang 双选项扩展到 TensorRT-LLM，并把 colocated、disaggregated、async one-step-off 与 weight update 组合成正式 recipe。** 这比单独接入一个 inference engine 更重要，因为它扩大了可比较的 rollout runtime 设计空间。

## Accepted Frontier Signals

### SLAI T-Rex: Full-Parameter Post-training of the DeepSeek-V4 Family on Ascend SuperPOD

- Signal ID：2026-07-23-001
- Source ID：arxiv:2607.20145
- First seen：2026-07-23 15:55（Asia/Shanghai，本次扫描）
- Focus Match：P0 Focus
- 来源：arXiv
- 类型：technical report / trillion-parameter MoE / full-parameter post-training / Ascend
- 链接：https://arxiv.org/abs/2607.20145
- Primary-source check：title / 65 位 authors / v1 time / 34.22% MFU / 2.93x baseline improvement / 73-page report 已对齐 arXiv abs 页
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：这是少见的万亿参数 MoE 全参后训练系统报告，覆盖 memory、parallelism、communication orchestration、kernel 和稳定性，而不是只讨论算法或小规模 SFT。
- Status：NEW
- 建议动作：进入下一轮 P0 候选；优先读系统总览、parallel mapping、通信 overlap、kernel breakdown 与稳定性章节
- 关联主题：[MoE](../topics/moe.md), [Distributed Training](../topics/distributed_training.md), [FP8](../topics/fp8.md), [Long-context Training](../topics/long_context_training.md)

报告以 DeepSeek-V4 family 为 workload，在 Ascend NPU SuperPOD 上给出分层优化路径，并报告 34.22% MFU、相对 open-source baseline recipe 提升 2.93x。当前最值得核对的不是后半部分 OR 领域效果，而是 full-parameter post-training 如何安排参数/状态、并行组、通信 overlap 与低层 kernel，以及这些选择在 73 页正文中是否有足够可复现细节。

### Fine-grained Computation-Communication Overlap via Tile-level Signaling and Scheduling for Mixture-of-Experts

- Signal ID：2026-07-23-002
- Source ID：arxiv:2607.19539
- First seen：2026-07-23 15:55（Asia/Shanghai，本次扫描；boundary late-discovered）
- Focus Match：P0 Focus
- 来源：arXiv
- 类型：paper / MoE kernel / all-to-all overlap
- 链接：https://arxiv.org/abs/2607.19539
- Primary-source check：title / 3 位 authors / v1 time / producer-consumer design / 4x A100 / 2.64x and 2.74x claims 已对齐 arXiv abs 页
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：它直接优化 MoE expert compute 与第二次 all-to-all 的关键路径，并把调度粒度从完整 expert/kernel 降到 tile/segment。
- Status：NEW
- 建议动作：精读 persistent kernel、tile-ready signaling、SM partition 与 correctness；对照 DeepEP/Grouped GEMM 的 overlap 边界
- 关联主题：[MoE](../topics/moe.md), [NCCL](../topics/nccl.md), [Distributed Training](../topics/distributed_training.md)

方案让一个 persistent per-rank compute producer 覆盖本 rank 的所有 experts，并优先计算影响远端返回的 tiles；另一个驻留在少量专用 SM 上的 communication consumer，在 tile ready 后立即发起 segment-granular transfer。作者在 4x A100 上报告最高 2.64x 端到端、2.74x MoE layer speedup。下一步要确认专用 SM 的机会成本、不同 GEMM shape 下的收益稳定性，以及通信 consumer 是否依赖特定传输原语。

### MoX: Efficient MoE Routing on Direct-Connect Topologies

- Signal ID：2026-07-23-003
- Source ID：arxiv:2607.20220
- First seen：2026-07-23 15:55（Asia/Shanghai，本次扫描）
- Focus Match：P0 Focus
- 来源：arXiv
- 类型：paper / MoE networking / direct-connect topology
- 链接：https://arxiv.org/abs/2607.20220
- Primary-source check：title / 4 位 authors / v1 time / token-aware multicast / ASTRA-sim / 1.8x and 47% claims 已对齐 arXiv abs 页
- 影响等级：★★★★☆
- Decision：Read
- Reason：它针对 MoE 稀疏、动态的 token traffic，在不动态重配光网络拓扑的前提下优化 dispatch/return routing，直接连接 expert placement 与物理网络设计。
- Status：NEW
- 建议动作：进入 P1 候选；核对 trace 来源、tree-packing objective、拥塞模型与训练/推理 traffic 差异
- 关联主题：[MoE](../topics/moe.md), [NCCL](../topics/nccl.md), [Distributed Training](../topics/distributed_training.md)

MoX 用 token-aware multicast tree 降低复制带宽，再用离线预计算 link weights 做受限 multicast tree-packing。作者基于真实 MoE traffic/token traces 与 ASTRA-sim，报告完整 MoE block 相对 min-hop 最高 1.8x；在 1,024-TPU Boardfly 模型上，dispatch bottleneck link load 最多降低 47%。这说明 direct-connect fabric 的价值不只由拓扑决定，也取决于是否有适配 MoE traffic 的 routing layer。

### HijackKV: New Threat in Position-Independent KV Cache Reuse

- Signal ID：2026-07-23-004
- Source ID：arxiv:2607.19957
- First seen：2026-07-23 15:55（Asia/Shanghai，本次扫描）
- Focus Match：P0 Focus
- 来源：arXiv
- 类型：paper / KV cache reuse / serving correctness and security
- 链接：https://arxiv.org/abs/2607.19957
- Primary-source check：title / 4 位 authors / v1 time / position-independent reuse threat / 94% average success / 10% hit-rate and 50% recomputation claims 已对齐 arXiv abs 页
- 影响等级：★★★★☆
- Decision：Read
- Reason：它揭示了“相同 token chunk 可以跨位置复用 KV”这一性能优化的隐含前提不成立：KV 同时编码生成时的上下文，因此 cache key 与隔离边界必须覆盖更多状态。
- Status：NEW
- 建议动作：进入 P1 候选；重点核对攻击前提，并沉淀 KV cache tenant isolation、provenance、validation 与 recomputation 策略
- 关联主题：[Long-context Training](../topics/long_context_training.md), [Agentic RL](../topics/agentic_rl.md), [Fault Tolerance](../topics/fault_tolerance.md)

传统 prefix cache 要求 token 与 position 都匹配，position-independent reuse 试图扩大命中范围；HijackKV 表明，被命中的 benign text chunk 所对应 KV 可能携带攻击者此前上下文。作者报告单次攻击平均成功率 94%，并在低命中与频繁重算条件下仍有效。对 infra 的意义不是复述 attack，而是重新审视 cache identity、跨租户共享、状态 provenance 和安全回退路径。

### Setting a World Record for MoE Pre-Training on NVIDIA GB300 NVL72

- Signal ID：2026-07-23-005
- Source ID：blog:nvidia/gb300-nvl72-moe-pretraining-record
- First seen：2026-07-23 15:55（Asia/Shanghai，本次扫描；boundary late-discovered）
- Focus Match：P0 Focus
- 来源：NVIDIA Technical Blog
- 类型：official engineering blog / MoE pre-training / rack-scale co-design
- 链接：https://developer.nvidia.com/blog/setting-a-world-record-for-moe-pre-training-on-nvidia-gb300-nvl72/
- Primary-source check：publication date / 4 位 authors / DeepSeek-V3 671B / 1,648 TFLOPs per GPU / 256-to-1,024 GPU scaling claims 已对齐 NVIDIA 原文；性能数字均标记为 NVIDIA-reported
- 影响等级：★★★★★
- Decision：Read
- Reason：文章把 MoE per-layer all-to-all、rack 内 NVLink scale-up、rack 间 gradient scale-out 与三套训练框架的 delivered performance 放进同一组数据，适合建立硬件拓扑到训练吞吐的完整因果链。
- Status：NEW
- 建议动作：精读 Figure 1/3/4/7；把硬件代际、软件版本与框架差异分开，不直接把厂商报告倍数当作独立 benchmark
- 关联主题：[MoE](../topics/moe.md), [Megatron-LM](../papers/megatron_lm.md), [NCCL](../topics/nccl.md), [Distributed Training](../topics/distributed_training.md)

NVIDIA 在 DeepSeek-V3 671B、256 GPUs 上报告 Megatron Core 达到 1,648 TFLOPs/GPU，并称同一 GB300 NVL72 硬件六个月内依靠软件优化从 1,088 提升到 1,648 TFLOPs/GPU。扩展到 1,024 GPUs 时，Megatron Core 保持 98.5% per-GPU performance，TorchTitan/JAX 约 97%。最重要的工程判断是：MoE all-to-all 应尽量留在高带宽 scale-up domain，rack 间主要承载可被 compute 隐藏的梯度流量；但这些数据需要结合配置、precision 与 workload 细节复核。

### Error Certificates for KV-Cache Eviction via Randomized Design

- Signal ID：2026-07-24-001
- Source ID：arxiv:2607.21475
- First seen：2026-07-24 09:52（Asia/Shanghai，本次增量扫描；arXiv v1 为 2026-07-24 00:16）
- Focus Match：P1 Focus
- 来源：arXiv
- 类型：paper / KV-cache eviction / serving correctness
- 链接：https://arxiv.org/abs/2607.21475
- Primary-source check：title / author Peng Xie / v1 time / 0.97 empirical coverage / AUC 0.73–0.75 and 0.47–0.54 claims 已对齐 arXiv abs 页
- 影响等级：★★★★☆
- Decision：Read
- Reason：它把 KV eviction 的评估从平均 task accuracy 推进到逐步、可归因的 cache-induced error，并证明 deterministic top-k 无法从保留状态一致估计被删除内容造成的 attention error。
- Status：NEW
- 建议动作：进入 P1 候选；核对 randomized tail sampling、Hájek correction 的运行时成本，以及 certificate 如何驱动 recomputation
- 关联主题：[Long-context Training](../topics/long_context_training.md), [Agentic RL](../topics/agentic_rl.md), [Distributed Training](../topics/distributed_training.md)

论文的关键价值不在“随机淘汰比 top-k 更准”，而在于 observability：已删除的 KV 本来不可见，deterministic eviction 又不给估计器留下可校正的采样概率，因此 runtime 无法判断一次失败是 cache compression 造成的，还是模型本身就会失败。作者报告 randomized Poisson tail 在不降低 accuracy 的条件下取得 0.97 empirical coverage；certificate 对 cache-induced 与 inherent failure 的区分 AUC 为 0.73–0.75，而 output confidence 为 0.47–0.54。它更适合做 attribution 和 recomputation scheduling，作者也明确说它不是通用 failure predictor。

### NeMo RL: TensorRT-LLM Rollout Backend

- Signal ID：2026-07-24-002
- Source ID：github:NVIDIA-NeMo/RL#2420
- First seen：2026-07-24 09:52（Asia/Shanghai，本次增量扫描）
- Focus Match：P0 Focus
- 来源：NVIDIA NeMo RL GitHub
- 类型：merged PR / RL framework / rollout backend
- 链接：https://github.com/NVIDIA-NeMo/RL/pull/2420
- Primary-source check：PR title / merged time `2026-07-23 23:53`（Asia/Shanghai）/ TensorRT-LLM generation modules / colocated and non-colocated recipes / async one-step-off config 已对齐 merged PR patch；未独立复现实验
- 影响等级：★★★★★
- Decision：Read
- Reason：这不是只增加一个 engine adapter；它把 TensorRT-LLM 接进 GRPO、async trajectory collector、weight update 与 colocated/disaggregated deployment，形成可运行 recipe 边界。
- Status：NEW
- 建议动作：代码级对照 generation interface、pause/refit、KV invalidation 和 weight-version handling；评估其中哪些 backend contract 可迁移到 AReaL
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [Distributed Training](../topics/distributed_training.md), [NCCL](../topics/nccl.md), [Long-context Training](../topics/long_context_training.md)

merged patch 新增 `nemo_rl/models/generation/trtllm/` backend，并提供 Qwen3-1.7B colocated/non-colocated、Qwen3-8B colocated，以及 async one-step-off recipe。最值得读的是接口而不是“TensorRT-LLM 更快”这个未经独立验证的结论：同一 RL trainer 如何切换不同 generation backend，colocated 时如何暂停/释放和恢复资源，non-colocated 时如何同步权重，in-flight weight update 后为何要重建 KV cache，以及异步 rollout 如何携带 weight version。它为 AReaL 提供的是 backend contract 与 correctness checklist，而不是可直接照抄的性能倍数。

## Observed / Rejected Candidates

| 材料 | Source ID | Decision | 原因 |
|---|---|---|---|
| DGNA: Dissecting GPU NUMA Architecture through Microbenchmarking and Data Analysis | arxiv:2607.19922 | Observe | A100/H100 L2/DRAM NUMA 与 SM mapping 很有系统价值；先核对实验可复现性和对真实 kernel placement 的可操作性 |
| ELSAA | arxiv:2607.20214 | Observe | sparse + low-rank attention operator 连接长上下文训练，但摘要没有端到端训练结果，当前更像方法候选 |
| PRO-LONG | arxiv:2607.20064 | Observe | programmatic memory 让 Agent 搜索完整 interaction log，和 CompactionRL 构成有价值对照；主要信号在 harness/context management，不直接改变训练 runtime |
| How Fast Can Reward Models Score? | arxiv:2607.19712 | Observe | reward scoring 与 rollout 争抢资源的问题成立，但摘要显示 batching 影响大于语言/runtime，且缺少大规模 RL pipeline 证据 |
| Co-Located AI Training Jobs Synchronize? | arxiv:2607.19638 | Observe | shared power cap 导致 job phase-lock 的假设值得 GPU cluster 关注；目前是待实测的理论预测，不作为生产事实 |
| Controlled Periodic Synchronization for Efficient Data-Parallel Training | arxiv:2607.21224 | Observe | CPDP 在 16.6 ms WAN 与 ResNet-50/CIFAR100 上展示周期同步收益，但模型、网络与规模证据仍窄，不能外推到 LLM dense/MoE 训练 |
| The Dark Room in the Reward Channel | arxiv:2607.21273 | Observe | 指出 dense prediction reward 在 GRPO std normalization 下会驱动 agent 进入 0% success 的 absorbing state；当前为单作者、single-seed，先作为 reward-channel correctness 警报 |
| PATS: Policy-Aware Training Scaffolding for Agentic RL | arxiv:2607.21419 | Observe | 动态 scaffold 随 policy 能力调整且部署时移除，报告最多 +18.6%；主要改变训练策略与 context，不直接改变 rollout runtime |
| MemTools | arxiv:2607.21404 | Observe | 共享 memory runtime 与 lifecycle contract 值得 Agent Infra 关注，但当前不是 Training/RL Infra 的优先缺口 |
| Agentic Context Management | arxiv:2607.21503 | Reject | 方向匹配但工程证据主要来自自有 reference implementation 和 benchmark 汇总，尚不足以改变 long-context runtime 判断 |

## OpenAI / Anthropic / NVIDIA Watch

| Vendor | Sources checked | Decision | 结果 |
|---|---|---|---|
| OpenAI | official research index / newsroom | Not found / no core signal | 截至本次结束时刻，official Research index 最新技术条目仍为 7 月 15 日 GPT-Red；游标后未发现新的 training、RL、agent runtime 或 inference infra 正文。 |
| Anthropic | official news / research | Not found / no core signal | Newsroom 最新可见条目仍为 7 月 22 日 Economic Futures Research Fund / Economic Index；游标后没有改变当前 Training/RL/Inference Infra 判断的新材料。 |
| NVIDIA | Technical Blog / NeMo RL GitHub | Accepted + Rejected | 7 月 21 日 GB300 NVL72 MoE 文章保留 Accepted；本次新增 NeMo RL TensorRT-LLM backend。7 月 23 日 Prime Intellect hosted RL 教程有可运行命令，但主要证明产品可用性，不提供新的 runtime 架构或独立性能对照，因此不进入 Accepted。 |

## Hugging Face Watch

| Source | Source type | Decision | 结果 |
|---|---|---|---|
| Hugging Face Blog | official RSS / blog index | Rejected / no core signal | 截至本次结束时刻，最新仍为 7 月 23 日 Nunchaku 4-bit diffusion inference，发布时间早于增量游标且与当前 LLM Training/Agentic RL/serving 主线不匹配。 |
| TRL / Transformers / Accelerate / PEFT / Kernels | official release/docs entry points | Not found / no material change | 增量窗口未发现会改变 RL rollout、distributed training、long-context 或 inference backend 工程判断的正式 release。 |
| Community Articles | community index | Observe only | 未把社区文章热度当作系统信号；本窗口没有证据足够硬的新条目进入 Accepted。 |

## RL Framework Watch

本节只把 release、已合入的重大 PR 或证据充分的候选改动保留下来。`open` PR 表示方向信号，不表示当前版本已经可用。

| Framework | Change | Subsystem | Evidence / state | Decision | 对 AReaL 的参考 |
|---|---|---|---|---|---|
| AReaL | [PR #1554](https://github.com/areal-project/AReaL/pull/1554)：reject incomplete sampling evidence | rollout / train-inference correctness | merged；逐 token 校验 token/logprob 数量、finite value，并对 vLLM `-9999.0` ambiguous sentinel fail closed；patch 含对应单测 | Read | 这是 AReaL 自身应保留的 correctness contract：PPO/GRPO 不能把缺失或被 clamp 的 sampling logprob 当成真实行为策略证据 |
| AReaL | PR #1555 reduce Megatron training memory peaks | training / memory | open PR；给出 LM Head、vocab-parallel loss、optimizer grad-copy profile，峰值 95,174 MiB → 86,564 MiB | Observe | chunked LM Head loss、backward recompute 与 optimizer grad-copy accounting 值得代码级对照；合入前不记为正式能力 |
| verl | [PR #7095](https://github.com/verl-project/verl/pull/7095)：FSDP gradient accumulation 延迟 gradient sync | training / communication | merged；2026-07-22 14:49（Asia/Shanghai），位于本次游标窗口内 | Observe | AReaL 可对照 `no_sync`/defer-sync 边界，确认最后一个 micro-batch 才触发 collective，并验证 mixed precision、clip grad 与 optimizer step 的顺序 |
| verl | [PR #7082](https://github.com/verl-project/verl/pull/7082)：统一 stale / DAPO-filtered / failed group 的 eviction 与 refill | rollout / replay buffer / scheduler | merged；sync DAPO 按 `k → 2k` refill 后裁剪，async 三类终态均按 `k → k` refill；等待 group sessions settle 后再发布 terminal status | Read | 可对照 AReaL 的 group completion 与 retry 语义，避免一个 prompt 被多种 drop reason 重复补样，也避免 late trajectory write 污染已终结 group |
| verl | PR #7115 KV-cache-aware request load balancer | rollout / scheduler | open PR；prefix-cache hit + live load + sticky session，只有 smoke tests，尚缺真实 vLLM E2E throughput | Observe | AReaL rollout router 可借鉴 cache locality 与 overload-aware fallback 的联合 scoring |
| slime | PR #1709 Mooncake RDMA transport for rollout data | data / trajectory path | open PR；报告典型 payload trainer-side GET 约 3x，但 benchmark scripts 未随 PR 提供 | Observe | rollout tensor 跨节点传输可从 Ray object store 抽象为 pluggable structured/RDMA backend |
| slime | [PR #2228](https://github.com/THUDM/slime/pull/2228)：SGLang v0.5.15.post1 | inference backend | merged；仅依赖版本升级，本窗口没有 slime 自身架构或 benchmark 证据 | Reject / routine | 记录依赖基线即可，不把普通 dependency bump 升级成 framework signal |
| ROLL | PR #469 Mooncake DataProto transfer backend | data / trajectory path | open PR；保留 transfer backend boundary，包含 RDMA round-trip tests | Observe | 参考其后端接口与 lazy field materialization；需要独立性能数据后再判断价值 |
| OpenRLHF | release / major PR | — | 游标后未发现正式 release 或足以改变架构、性能、正确性的重大 PR | Not found | 不用普通 commit 补位 |
| NeMo RL | merged PR #3219 Single Controller async-GRPO rollout path | rollout / replay buffer / scheduler | merged；引入 group-granular TQReplayBuffer、reserve/commit、weight-version metadata 与 rollout pump；完整 SC path 仍依赖后续 PR | Observe | `reserve → generate → commit`、start/end weight version 与 per-version quota 很适合对照 AReaL trajectory queue；当前不能宣称端到端新路径已完成 |
| NeMo RL | [PR #2420](https://github.com/NVIDIA-NeMo/RL/pull/2420)：TensorRT-LLM rollout backend | rollout / inference backend / weight sync | merged；新增 colocated、non-colocated、async one-step-off recipes 和 generation modules；本次 Accepted，尚未独立 benchmark | Read | 对照 AReaL backend interface、pause/refit、in-flight update 后 KV invalidation 与 per-backend correctness tests |
| NeMo RL | [PR #3220](https://github.com/NVIDIA-NeMo/RL/pull/3220)：Single Controller train path with StalenessSampler | training / replay buffer / scheduler | merged；支持 max staleness、freshest-first、FIFO/force-order 与 stale group eviction；属于分阶段 SC 重构的一部分 | Observe | AReaL 可对照 staleness window 与 sample ordering 是否应是两个独立策略，以及 future-version group、完整 group 和 recovery step 的处理 |
| NeMo RL | [PR #3332](https://github.com/NVIDIA-NeMo/RL/pull/3332)：Qwen3-235B async rollout memory-util tuning | rollout / inference backend | merged；recipe 将 `gpu_memory_utilization` 从 0.80 调到 0.88；PR 报告约 11% E2E throughput，改动本身仅一项配置 | Observe | 说明 rollout buffer starvation 可能先由显存保守配置引起；迁移前必须在 AReaL 的真实 length distribution、KV pressure 与 OOM margin 上重测 |

### Framework Follow-up

- NeMo RL merged PR #3066 修复 async-GRPO checkpoint resume deadlock：恢复时不仅要有当前 step 的完整 batch，还必须准备 `step+1` lookahead，否则 refit 后会跳过一个 target weight version。该案例应进入未来 checkpoint/recovery playbook，但它是 correctness fix，不单独升级为 frontier Accepted。
- NeMo RL merged PR #3226 修复 colocated ZMQ-IPC refit 未传递 calibrated FP8 KV scales 的问题，说明 train-generation colocated path 需要逐字段验证 weight-sync metadata，而不能只验证主权重。
- AReaL #1555、verl #7115、slime #1709、ROLL #469 在本次结束时仍未形成可验证的正式能力；如果 PR 被关闭、重写或继续缺少 E2E 数据，不应沉淀成 topic 结论。

## 本次阅读决策

### P0 候选

1. **SLAI T-Rex**：当前最完整的万亿参数 MoE full-parameter post-training 系统报告，先读系统章节。
2. **Fine-grained MoE Tile-level Overlap**：直接连接 MoE kernel、all-to-all 与 SM partition，可快速形成明确工程判断。
3. **NeMo RL TensorRT-LLM backend**：不先追性能数字，先读 generation backend contract、colocation 与 weight/KV lifecycle，直接对照 AReaL。

### P1 候选

- MoX：补齐 MoE routing 与 direct-connect network 的关系。
- HijackKV：补齐 position-independent KV reuse 的 correctness/security 边界。
- NVIDIA GB300 NVL72 MoE pre-training：作为 topology + software co-design 案例阅读，但保留厂商 benchmark 审慎性。
- Error Certificates for KV-Cache Eviction：补齐 cache compression 的 attribution 与 recomputation observability。

本次不直接修改 reading queue。先完成当前 P0，再决定是否替换队列，避免 P1 继续膨胀。

## 下一次扫描起点

- Next cursor：2026-07-24 09:52
- 下次继续扫描：
  - 新增 arXiv v1；
  - OpenAI / Anthropic / NVIDIA / Hugging Face 官方更新；
  - AReaL / verl / slime / ROLL / OpenRLHF / NeMo RL 正式 release 与 open PR 的 merge/benchmark 状态；
  - NeMo RL TensorRT-LLM backend 是否出现公开 E2E benchmark，以及 Single Controller 后续 PR 是否形成完整 train/rollout/recovery path；
  - 是否出现可迁移到 AReaL 的 rollout scheduling、weight sync、trajectory transport 或 checkpoint/recovery 设计。

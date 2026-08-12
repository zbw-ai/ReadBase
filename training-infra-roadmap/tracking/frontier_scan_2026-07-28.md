# Frontier Scan, 2026-07-28

- Previous scan：[2026-07-27](frontier_scan_2026-07-27.md)
- Window：2026-07-27 12:23 ~ 2026-07-28 16:58
- Timezone：Asia/Shanghai
- Generated at：2026-07-28 16:58
- Report type：flexible frontier scan
- Sources scanned：arXiv cs.DC / cs.LG / cs.AI / cs.CL / cs.AR / cs.PF / cs.NI new records；OpenAI / Anthropic / NVIDIA / Hugging Face official sources；AReaL / verl / slime / ROLL / OpenRLHF / NeMo RL / Molt releases、default-branch changes 与 major PR
- Scan completeness：完整覆盖上一游标后的 arXiv 7 月 28 日公告、重点厂商官方源和 RL framework primary sources。arXiv 原始提交早于窗口、但本次才进入公告或首次发现的材料，显式标记为 `boundary late-discovered`。GitHub main commit 不等同正式 release；所有性能数字均保留作者或项目方 attribution，未独立复现。

## 本次核心判断

本次没有按数量凑榜单，而是保留了十条会改变当前工程判断的信号。它们集中指向五个趋势：

1. **长上下文的负载单位不能再只看 token。** Libra 证明相同 token 数的 packed microbatch 仍可能具有完全不同的 attention workload；`sum(sequence_length^2)` 才更接近真正的成本，错误的采样和放置会制造 DP straggler 与 PP bubble。
2. **百万 token、超稀疏 MoE 和 Agentic RL 已开始在同一系统里汇合。** Kimi K3 不只是 2.8T 模型发布；其报告同时暴露 KDA 算法系统协同、balanced EP、百万 token persistent rollout / sandbox state 和 MXFP4 QAT，值得按超大模型系统报告精读。
3. **推理优化的瓶颈正在从主 kernel 转移到旁路。** PIVOT 处理 sparse attention indexer，DA-MoE 处理 routing skew 下的 kernel dispatch，SpecBox 处理 sandbox cold start；只优化 attention/GEMM 主路径已经不够。
4. **Agentic RL correctness 正在穿过训练与推理边界。** TRL 的 vLLM sleep bug 会让 rollout worker 在 wake 后恢复初始 checkpoint，而不是当前 policy。系统没有报错，但 on-policy 数据语义已被破坏。
5. **长任务的状态管理、奖励判断和 all-failed group 正在成为一等基础设施。** CORVUS、SeekJudge 和 ProGPO 分别处理 stale trajectory state、长轨迹 reward server 与零奖励 group 的 credit trap；这些问题不能只靠扩 GPU 或加长 context 解决。

## Accepted Frontier Signals

### Kimi K3: Open Frontier Intelligence

- Signal ID：2026-07-28-001
- Source ID：arxiv:2607.24653
- First seen：2026-07-28 16:58（Asia/Shanghai，本次扫描）
- Focus Match：P0 Focus
- 来源：arXiv / Moonshot AI official repository / model weights
- 类型：technical report / 2.8T MoE / million-token context / agentic RL infra
- 链接：https://arxiv.org/abs/2607.24653
- 官方仓库：https://github.com/MoonshotAI/Kimi-K3
- Primary-source check：title / Kimi Team authorship / v1 time / 2.8T total / 104B activated / 1M context / 16-of-896 routed experts / approximately 2.5x scaling-efficiency claim 已对齐 arXiv 与官方仓库
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：这是本窗口最完整的超大模型系统报告信号，训练、MoE、长上下文、Agentic RL、量化和部署信息同时出现，不能按普通模型发布略过。
- Status：NEW
- 建议动作：优先精读 training infrastructure、KDA co-design、perfectly balanced EP、million-token agentic RL 和 deployment 章节，再决定是否新增 `tech_reports/kimi_k3.md`
- 关联主题：[MoE](../topics/moe.md), [Long-context Training](../topics/long_context_training.md), [Agentic RL](../topics/agentic_rl.md), [Distributed Training](../topics/distributed_training.md), [FP8](../topics/fp8.md)

Kimi K3 采用 2.8T MoE、104B activated parameters、93 层和 1,048,576 context。官方报告把基础设施贡献明确写成四条：KDA 的算法系统协同、带高效显存管理的 perfectly balanced expert-parallel training、保存 rollout 与 sandbox state 的 million-token agentic RL，以及 deployment innovations。

值得追的不是 benchmark 排名，而是系统组合：Stable LatentMoE 每 token 路由 16/896 experts；自 SFT 阶段起使用 MXFP4 weights / MXFP8 activations 的量化感知训练；长任务既支持完整 1M context，也在 BrowseComp 中使用 300K token 触发的 context compaction。这里已经形成“扩窗口 + 管理上下文 + 持久化环境状态”的完整工程闭环。

### Libra: Taming Attention Workload Skew in Long-Context LLM Training

- Signal ID：2026-07-28-002
- Source ID：arxiv:2607.23250
- First seen：2026-07-28 16:58（Asia/Shanghai，本次扫描）
- Focus Match：P0 Focus
- 来源：arXiv
- 类型：paper / long-context training / context parallel / straggler
- 链接：https://arxiv.org/abs/2607.23250
- Primary-source check：title / 19 位 authors / v1 time / mechanism / Qwen3-Turbo 256K-1M workloads / up to 2.54x E2E / up to 3.14x worst-step claim / production GPU-hours 已对齐 arXiv abs 页
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：它修正了长上下文训练中一个常见但危险的假设：packing 只平衡 token 和线性算子，不能平衡按平方增长的 attention workload。
- Status：NEW
- 建议动作：精读 workload model、Variance-Reduced Sequence Placement、Tiled Attention Pooling 和 bounded communication domain；对照当前 128K SFT 的长度分布与 step-time 抖动
- 关联主题：[Long-context Training](../topics/long_context_training.md), [Context Parallelism](../topics/context_parallelism.md), [Pipeline Parallelism](../topics/pipeline_parallelism.md), [Distributed Training](../topics/distributed_training.md)

Libra 把 packed sequence 和对应 CP group 组织进固定大小的 sequence pool。DP 扩容时增加 pool 数，而不是扩大每个 attention exchange domain；这样 attention balancing 不随 DP degree 无限扩张通信域。Variance-Reduced Sequence Placement 先把互补 workload 放在一起，Tiled Attention Pooling 再在 pool 内分发 sequence-head tiles，并把 tile exchange 与 attention 计算流水重叠。

作者报告在 Qwen3-Turbo 256K / 1M token workload 上，相对 Ulysses 端到端最高 2.54x，最差 step 的 straggler-attention microbenchmark 最高 3.14x；并称已在 32K-1M token 生产任务上运行数十万 GPU-hours。当前 arXiv 页面没有公开代码入口，因此先读机制和生产证据，不把它写成可立即接入的开源组件。

### TRL: Correct vLLM Sleep/Wake Weight Synchronization

- Signal ID：2026-07-28-003
- Source ID：github:huggingface/trl@c285dc17
- First seen：2026-07-28 16:58（Asia/Shanghai，本次扫描）
- Focus Match：P0 Focus
- 来源：Hugging Face TRL official repository
- 类型：main commit / rollout correctness / weight sync
- 链接：https://github.com/huggingface/trl/commit/c285dc17b17cc0847306a31ce5731373ef62d9b4
- Primary-source check：commit time / sleep-level-2 behavior / old reload path / new sync path 已对齐官方 diff；当前尚未进入正式 release
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：这是 silent correctness bug：训练可以继续、系统不会 crash，但 sleep/wake 后的 rollout 可能来自初始 checkpoint，而不是当前 policy。
- Status：NEW
- 建议动作：检查所有使用 vLLM sleep mode 的 GRPO recipe；验证 wake 后 parameter checksum、policy version、sample logprob 与 rollout metadata
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [Checkpointing](../topics/checkpointing.md), [Distributed Training](../topics/distributed_training.md), [Rollout Latency](../playbooks/rollout_latency.md)

vLLM sleep level 2 会丢弃 model weights。旧路径 wake 后调用 `reload_weights`，其语义是重新加载初始 checkpoint；修复后 TRL 显式记录 `_llm_weights_sleeping`，并在生成前调用 `sync_weights()` 推送当前训练 policy。

这类问题说明“显存回收成功”不等于“policy state 恢复正确”。生产系统需要把 policy version 和 weight checksum 作为 rollout 数据的一部分，不能只依赖 worker 已 wake、API 可调用这类活性信号。

### NVIDIA NOOA: Six Agent Harness Capabilities for Higher Model Performance

- Signal ID：2026-07-28-004
- Source ID：blog:nvidia/six-agent-harness-capabilities
- First seen：2026-07-28 16:58（Asia/Shanghai，本次扫描）
- Focus Match：P0 Focus
- 来源：NVIDIA Technical Blog / open research preview
- 类型：official engineering blog / agent harness / context and state management
- 链接：https://developer.nvidia.com/blog/six-agent-harness-capabilities-for-higher-model-performance/
- Primary-source check：publication time / typed I/O / pass-by-reference / code-as-action / programmable loop / explicit state / model-callable harness API / benchmark claims 已对齐 NVIDIA 官方正文
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：它把 harness 设计和 token 成本、cache validity、trajectory state 可检查性直接连接起来，适合与 CORVUS、CompactionRL 和 AReaL rollout contract 对照。
- Status：NEW
- 建议动作：阅读 pass-by-reference、SQLite typed memory、model-callable harness API 和 evaluation methodology；不要只看 SWE-bench 分数
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [Long-context Training](../topics/long_context_training.md), [Rollout Latency](../playbooks/rollout_latency.md)

NOOA 用单个 Python class 表达 agent：methods 是 capabilities，fields 是 state，docstrings / type annotations 是 prompt 与执行契约。最有 infra 价值的是 pass-by-reference：工具结果不必反复序列化进 transcript，长轨迹可以保持 append-only 与 prefix-cache friendly。

NVIDIA 报告其 SWE-bench Verified 达到 82.2%、约 1.1M tokens/task，对比 harness 为 78.2%、2.2M tokens/task。数字尚未独立复现，但代码、benchmark agent 和 methodology 已开放，证据强于只给结果的产品博客。

### SpecBox: Speculative Sandbox Scheduling for Efficient LLM Agent Serving

- Signal ID：2026-07-28-005
- Source ID：arxiv:2607.23933
- First seen：2026-07-28 16:58（Asia/Shanghai，本次扫描）
- Focus Match：P0 Focus
- 来源：arXiv
- 类型：paper / agent serving / sandbox scheduling / tail latency
- 链接：https://arxiv.org/abs/2607.23933
- Primary-source check：title / 10 位 authors / affiliations / v1 time / mechanisms / up to 2.9x P99 / 45.9% memory claim 已对齐 arXiv abs 页
- 影响等级：★★★★★
- Decision：Read
- Reason：Agent serving 的瓶颈开始从 LLM decode 扩展到 sandbox cold start、依赖预取、结果缓存和 artifact transport，这些同样会拖慢 RL rollout。
- Status：NEW
- 建议动作：进入 P1 候选；重点读 intent-driven prewarming 的误判成本、dependency graph、sandbox lifecycle 和 failure isolation
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [Rollout Latency](../playbooks/rollout_latency.md), [Fault Tolerance](../topics/fault_tolerance.md)

SpecBox 在 LLM 仍生成 token 时，根据 keyword 与 streaming embedding 推测即将发生的工具调用，并提前启动 sandbox；随后通过 dependency graph 做跨 step stochastic prefetch。系统还加入 semantic result cache 和 shared-memory out-of-band transport，减少重复执行和网络序列化。

作者在高并发多轮 agent traces 上报告，相对按需启动 sandbox 最多降低 2.9x P99；相对永久保留 sandbox 最多减少 45.9% peak memory。真正需要验证的是 false prewarm 是否会造成新的内存争用，以及 MCP/tool side effect 是否允许结果缓存。

### PIVOT: Query-Group Indexing for Token-Level Sparse Attention

- Signal ID：2026-07-28-006
- Source ID：arxiv:2607.24593
- First seen：2026-07-28 16:58（Asia/Shanghai，本次扫描）
- Focus Match：P0 Focus
- 来源：arXiv
- 类型：paper / sparse attention / indexer / long-context inference
- 链接：https://arxiv.org/abs/2607.24593
- Primary-source check：title / 8 位 authors / v1 time / PIVOT-Reuse and Refine / DeepSeek-V3.2 and GLM-5.1 evaluation / up to 4x indexer and 1.6x E2E claims 已对齐 arXiv abs 页
- 影响等级：★★★★☆
- Decision：Read
- Reason：它指出 token-level sparse attention 可能只把 O(L²) 瓶颈从 attention kernel 转移到 top-k indexer，并利用相邻 query 的候选重叠消除重复全前缀扫描。
- Status：NEW
- 建议动作：进入 P1 候选；确认 accuracy fidelity、query group size、MTP decode grouping 与训练可用性
- 关联主题：[FlashAttention](../topics/flashattention.md), [Long-context Training](../topics/long_context_training.md), [Agentic RL](../topics/agentic_rl.md)

PIVOT 把一组相邻 query 聚合为 proxy query，只做一次 full-prefix traversal 生成 candidate set。Reuse 直接共享 proxy top-k，Refine 再由每个 query 对 candidate set 重打分。prefill 用连续 query 分组，decode 则复用一个 MTP step 中共同生成的 queries。

作者报告在 DeepSeek-V3.2 和 GLM-5.1、LongBench / RULER 上保持 dense DSA indexer 的 accuracy，同时 indexer 最多 4x、long-context E2E latency 最多 1.6x。当前页面没有公开代码入口，先按算法与 kernel co-design 信号保留。

### DA-MoE: Distribution-Aware MoE Kernel Dispatch

- Signal ID：2026-07-28-007
- Source ID：arxiv:2607.23099
- First seen：2026-07-28 16:58（Asia/Shanghai，本次扫描）
- Focus Match：P1 Focus
- 来源：arXiv
- 类型：paper / MoE inference / kernel dispatch / routing skew
- 链接：https://arxiv.org/abs/2607.23099
- Primary-source check：title / 5 位 authors / v1 time / Effective Experts / Dirichlet reverse modeling / GPU-resident dispatch / latency numbers 已对齐 arXiv abs 页
- 影响等级：★★★★☆
- Decision：Read
- Reason：static token-count bucket 不能代表真实 fused-MoE GEMM 形状；expert routing skew 会改变 padding、reuse 和最佳 kernel。
- Status：NEW
- 建议动作：进入 P1 候选；重点读 routing histogram representation、offline tuning cost、GPU-side dispatch overhead 与 workload drift
- 关联主题：[MoE](../topics/moe.md), [Agentic RL](../topics/agentic_rl.md), [FP8](../topics/fp8.md)

DA-MoE 在 GPU 上读取 live routing histogram，并与 offline-tuned distributions 匹配，选择更合适的 fused-MoE kernel，避免 CPU-GPU synchronization。作者在 HumanEval-X serving traces 上报告：DeepSeek-V3 geomean fused-MoE latency 1.16x、Kimi K2 1.29x，峰值分别 1.40x 和 1.56x。

这条信号提醒训练和 rollout 系统：同一模型、同一 token batch size，并不意味着 expert GEMM 工作负载相同。profiling 与 autotuning 需要记录 routing distribution，而不是只记录 tokens/expert 的平均值。

### ProGPO: Progress-Conditioned Group Policy Optimization

- Signal ID：2026-07-28-008
- Source ID：arxiv:2607.22724
- First seen：2026-07-28 16:58（Asia/Shanghai，本次扫描）
- Focus Match：P0 Focus
- 来源：arXiv
- 类型：paper / group policy optimization / long-horizon agent / credit assignment
- 链接：https://arxiv.org/abs/2607.22724
- Primary-source check：title / 9 位 authors / v1 time / first-visit observation coverage / all-zero group condition / benchmarks and model scales 已对齐 arXiv abs 页
- 影响等级：★★★★☆
- Decision：Read
- Reason：它处理 GRPO 类方法在长任务中的 `all-failed group`：不是普遍替换 outcome reward，而是在组内没有任何成功信号时，用新状态覆盖度提供相对方向。
- Status：NEW
- 建议动作：进入 P1 候选；检查 observation equivalence、coverage gaming、额外 state storage 与 group scheduler 语义
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [Long-context Training](../topics/long_context_training.md), [Rollout Latency](../playbooks/rollout_latency.md)

长任务中 repeated / low-effect actions 往往占据 policy 高概率区域，真正改变环境状态的动作采样不足；当一组 rollout 全失败时，outcome reward 无法比较，形成 self-reinforcing credit trap。ProGPO 只在 group 全零时，用 first-visit observation coverage 给到达更多新状态的 trajectory/step 更高相对 advantage。

它与 VIGOR 的层次不同：VIGOR 决定 rollout 预算投向哪里，ProGPO 决定全失败 group 是否仍能产生方向。两者都要求 runtime 保留完整 group identity、observation fingerprints 和失败原因。

### SeekJudge: Practical Reward Infrastructure for Computer-Use Agent RL

- Signal ID：2026-07-28-009
- Source ID：arxiv:2607.23263
- First seen：2026-07-28 16:58（Asia/Shanghai，本次扫描）
- Focus Match：P0 Focus
- 来源：arXiv
- 类型：paper / reward server / long-horizon trajectory / computer-use agent
- 链接：https://arxiv.org/abs/2607.23263
- Primary-source check：title / 4 位 authors / v1 time / four-role judge / 9B distilled backbone / online-RL claim 已对齐 arXiv abs 页
- 影响等级：★★★★☆
- Decision：Read
- Reason：长任务 RL 的 reward 不只是一个模型调用，而是需要 trajectory condense、evidence grounding、主动查找、分析和低延迟 serving 的独立系统。
- Status：NEW
- 建议动作：进入 P1 候选；重点读 reward-server architecture、trajectory storage、step-level judgment、judge drift 与吞吐数据
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [Long-context Training](../topics/long_context_training.md), [Rollout Latency](../playbooks/rollout_latency.md)

SeekJudge 把 Condense、Ground、Seek、Analyze 四种角色组织成 Seek-Analyze loop，并蒸馏到共享 9B backbone。其目标不是给最终输出做一次 LLM-as-a-judge，而是对长 GUI trajectory 形成可用于在线 RL 的 step-level reward。

摘要声称其在 held-out RL goals 上达到或超过 native rule-based supervision，并有 reward server 加速设计，但没有给出摘要级端到端吞吐数字。后续必须核对 judge 成本是否会从 reward quality 问题变成新的 pipeline bottleneck。

### CORVUS: Synchronizing Agent Context with the Live Codebase

- Signal ID：2026-07-28-010
- Source ID：arxiv:2607.22711
- First seen：2026-07-28 16:58（Asia/Shanghai，本次扫描）
- Focus Match：P1 Focus
- 来源：arXiv
- 类型：paper / trajectory architecture / context reduction / coding agent
- 链接：https://arxiv.org/abs/2607.22711
- Primary-source check：title / 7 位 authors / v1 time / synchronized file registry / token and reasoning-cycle claims 已对齐 arXiv abs 页
- 影响等级：★★★★☆
- Decision：Read
- Reason：它把长轨迹中的 stale file snapshot 识别为数据结构问题，而不是单纯 context window 不够；这与 CompactionRL 和 NOOA 构成互补关系。
- Status：NEW
- 建议动作：进入 P1 候选；对照当前 coding-agent transcript，区分 immutable event、mutable world state 与 tool result reference
- 关联主题：[Long-context Training](../topics/long_context_training.md), [Agentic RL](../topics/agentic_rl.md), [CompactionRL](../papers/compactionrl.md)

传统 append-only trajectory 把 file-read action 和当时的 file content 永久绑定；文件被 agent 或人修改后，历史 snapshot 变旧，agent 会重复读并继续追加副本。CORVUS 把 file-read action 与 observation 解耦，用 synchronized registry 在每次 reasoning cycle 注入当前内容。

作者在两个 coding benchmark、四个模型上报告平均输入 token 减少 9%-50%、最终 prompt 缩短 15%-32%、reasoning cycles 最多减少 37%，pass rate 基本相当。它不是学习型 compaction，而是把“会变化的世界状态”从“不可变的事件日志”中拆出去。

## Observed / Rejected Candidates

### Observed

| 材料 | Source ID | Decision | 原因 |
|---|---|---|---|
| DynaResize: Runtime GPU Reallocation for Disaggregated LLM Post-Training | arxiv:2607.22614 | Observe / boundary late-discovered | 原始提交为 2026-06-15，本次首次发现；与 BiDiRL 高度相关，强调 communicator reuse、bounded state staging 和 hysteresis resize。作者报告相对最佳静态配置吞吐 +66.5%，适合后续做 BiDiRL/ROLL/verl dynamic scheduling 对照，但不伪装成今天新稿 |
| Scale Weight Decay and Train Better | arxiv:2607.23777 | Observe | Muon-SW 在 72M-930M MoE、约 600 tokens/active-parameter 下同 loss 最快 30%；实现很轻，但单作者、规模仍小于 frontier pretraining，先等待更大规模复现 |
| KAP: Knowledge Access Planning | arxiv:2607.24260 | Observe | 把 structured knowledge 编译成 runtime KV access plan，128K proposal-time KV access 降到 source state 的 5.5%；思路有价值，但需要确认 execution backend、质量边界和真实 decode speedup |
| Co-Harness | arxiv:2607.22688 | Observe | 联合优化 harness 与 model weights，并给出 200+ 小时 autonomous case；当前更像 agent research workflow，尚缺可迁移到 RL runtime 的公开实现细节 |
| Sparse Event-KV Memory Contract | arxiv:2607.23693 | Observe | 揭示保留 event KV 可能物化已被 eviction 的上游信息，对 Agent KV memory correctness 有价值；当前证据集中于 Qwen3-8B 与受控任务 |
| HeraSys | arxiv:2607.22578 | Observe | 跨 workflow node reuse、joint scheduling 和 resource skewing 最高降低 2.17x P99、提升 1.85x throughput；需要先确认 workload realism 与与现有 serving engine 的实现边界 |
| DomainPilot | arxiv:2607.22769 | Observe | domain-level loss monitor + mixture optimization 可通过约 30 行 adapter 接 MindSpeed/Megatron-LM；当前验证主要是 Qwen3-1.7B SFT，不足以改变大规模 recipe |
| PTStore | arxiv:2607.22648 | Observe / historical cross-list | 原始提交为 2026-06-25，当前因 cross-list 再出现；distributed prefix tensor replication 与 5-6x claim 值得进入 inference backfill，不计本次新信号 |
| Transformers model-aware FSDP plans | github:huggingface/transformers@3e9d3e50 | Observe | main commit 为多模型补 `_fsdp_plan`，把 wrapping/sharding policy 变成模型元数据；尚未进入 release，需等兼容性和 DCP/DTensor E2E |
| Transformers Mamba2-family cached decode fix | github:huggingface/transformers@9a223f15 | Observe | 修复 `seq_len=1` 的 dt/B/C axis、reshape 与 state broadcasting，TRL 随后恢复 NemotronH GRPO/RLOO 测试；重要 correctness 修复，但按未发布 commit 观察 |

### Rejected / No Core Signal

| 材料 | Decision | 原因 |
|---|---|---|
| FusionML: CPU+GPU co-execution on Apple Silicon | Reject for current focus | edge unified-memory prefill 优化有价值，但与当前数据中心训练/rollout 主线距离较远 |
| X-Stage for DiT inference | Reject for current focus | 通信计算重叠机制主要面向 diffusion transformer inference |
| Gleam CUDA API remoting over LANs | Observe only / no promotion | GPU remoting 与一致性有系统价值，但当前证据更偏 edge / cross-device sharing |
| Generic model, application, dataset and benchmark papers | Reject | 未提供会改变训练、rollout、serving、kernel、distributed runtime 或生产运维判断的机制 |

## OpenAI / Anthropic / NVIDIA Watch

| Vendor | Sources checked | Decision | 结果 |
|---|---|---|---|
| OpenAI | official news / research / API changelog / release notes | Not found / no core signal | 游标后未发现新的 training、RL、agent runtime 或 inference infra 技术正文。 |
| Anthropic | official newsroom / research / Claude Platform release notes | Not found / no core signal | 最新相关材料早于窗口起点；本次没有可验证的新系统信号。 |
| NVIDIA | Technical Blog / NeMo repositories | Accepted + Framework follow-up | NOOA agent harness 进入 Accepted；NeMo RL #3343 的 Gym submodule update 进入 framework watch。 |

## Hugging Face Watch

| Source | Source type | Decision | 结果 |
|---|---|---|---|
| TRL vLLM sleep/wake weight sync | official main commit | Accepted | 会把 rollout policy 错误恢复成初始 checkpoint 的 silent correctness bug，本次已提升为独立信号。 |
| Transformers model-aware FSDP plans | official main commit | Observe | 多模型补齐 `_fsdp_plan`，方向重要但尚未进入 release。 |
| Transformers Mamba2 cached decode | official main commit + TRL integration test | Observe | 修复单 token cached decode，并让 NemotronH GRPO/RLOO 测试恢复；等待正式 release。 |
| Hugging Face Blog | official blog index | Not found / no core signal | 窗口内文章未命中 Agentic RL、distributed training、long context 或 inference backend focus。 |
| Accelerate / PEFT / Kernels | official repositories and releases | Not found / no material release | PEFT 只有窄范围参数校验和 revision/token 转发修复；其余无高质量 signal。 |

## RL Framework Watch

本窗口没有新的 framework release。以下只保留会改变架构、correctness 或长期运维行为的 default-branch change / merged PR；Framework Watch 的 `Accepted` 表示值得保留的框架变更，不额外计入上方十条 Accepted Frontier Signals。

| Framework | Change | Subsystem | Evidence / state | Decision | 对 AReaL 的参考 |
|---|---|---|---|---|---|
| AReaL | [PR #1564](https://github.com/areal-project/AReaL/pull/1564)：backend-specific PP weight-sync groups | weight sync / inference backend | 已进入 default branch；SGLang PP 使用 per-stage NCCL group，vLLM 走 flat group | Accepted / Read | 已在 AReaL；应补 vLLM PP>1 的真实 update-dispatch E2E，而不只测试初始化分流 |
| verl | [PR #7157](https://github.com/verl-project/verl/pull/7157)：close TransferQueue event loop | data path / worker lifecycle | merged；旧实现 64 次泄漏 192 FD，新实现 1000 次零泄漏，并有 78-step E2E | Accepted / Read | 全面审计 sync-wrapper 创建的 loop、selector、thread 是否 stop/join/close；避免长期任务 `EMFILE` 后触发 Gloo reset |
| verl | [PR #7101](https://github.com/verl-project/verl/pull/7101)：vLLM 0.24 / Megatron Core 0.18 stack upgrade | training / rollout / deployment | 已进入 default branch；同步调整 TE、Megatron-Bridge、FP8 MoE mapping 与 async/sync CI matrix | Accepted / Observe | 可迁移 version-upgrade checklist 与跨 backend CI；PR 没有性能数据，不能写成性能提升 |
| verl | [PR #7173](https://github.com/verl-project/verl/pull/7173)：revert Qwen3.5 LoRA/MTP Megatron-Bridge support | weight sync / training / rollout | merged rollback；删除 name normalization、bucket transfer 和相关 tests；PR 未说明根因 | Accepted / Read | Qwen3.5、MTP、LoRA name mapping 应与 backend version 绑定，并保留 trainer-to-rollout parameter parity E2E |
| NeMo RL | [PR #3343](https://github.com/NVIDIA-NeMo/RL/pull/3343)：Gym submodule update | trajectory / environment / inference API | merged；修复 persisted-rollout metrics，并加入 Codex CLI、Responses SSE、MCP 和统一 trajectory mapping；新路径目前 eval-only | Accepted / Read | 只对已持久化 rollout 聚合指标；训练接入前必须补 token IDs、logprobs、reward profile 与 timeout cleanup |
| verl | [PR #7158](https://github.com/verl-project/verl/pull/7158)：partial-rollout resume MoE metadata fix | rollout / routing replay | merged；统一 `routed_experts` 为 NumPy，修复 resume concat crash | Observed | 将 interruption/resume metadata schema 显式版本化；当前属于窄范围修复 |
| Molt | [PR #44](https://github.com/NVIDIA-NeMo/labs-molt/pull/44)：CUDA compatibility selection | deployment / container | merged；按 host driver 选择是否启用 `cuda-compat-13-0` | Observed | AReaL container entrypoint 可借鉴，但不是 RL runtime 架构变化 |
| slime | no material default-branch change | - | 最新 change 早于本窗口 | Not found | 不用 routine commit 补数量 |
| ROLL | no material default-branch change | - | 最新 change 早于本窗口 | Not found | 继续跟踪 dynamic scheduling 与 weight update |
| OpenRLHF | no material default-branch change | - | 最新 change 早于本窗口 | Not found | 不用普通 issue / docs 代替工程信号 |

### Framework Follow-up

- AReaL #1564 修复了一个具体限制，但当前代码路径仍值得做负向测试：`gen_pp_size > 1` 时 update dispatcher 是否在 vLLM 与 SGLang 下选择了正确 communicator，而不是只验证对象初始化成功。
- verl #7157 是长期任务的典型 lesson：短任务看不出 file-descriptor leak，真正出错时往往表现为 `EMFILE`、Gloo reset 或随机 worker failure。资源生命周期需要 endurance test，而不只是单步 unit test。
- verl #7173 是重要的负向信号。功能被整体 rollback 比“新增模型支持”更值得看：MTP/LoRA/Megatron-Bridge 的 parameter-name 和 tensor-layout contract 还没有稳定，不能因 recipe 曾合入就视为生产可用。
- NeMo RL #3343 已打通 Codex CLI / Responses / MCP 的 eval trajectory path，但官方明确没有 token IDs / logprobs。把 eval adapter 宣称成 trainable rollout backend 会越过 correctness 边界。

## 本次阅读决策

### 建议今天先读

1. **Libra**：最贴当前长上下文训练实践，先理解为什么 token-balanced packing 仍会产生 attention straggler。
2. **Kimi K3 technical report**：先读系统章节，不看 benchmark 大表；重点是 balanced EP、million-token rollout state 和 KDA co-design。
3. **TRL vLLM sleep/wake fix**：十分钟看完 commit diff，再检查自己的 rollout backend 是否存在同类“恢复了服务但恢复错权重”问题。

### 后续按问题选择

- Agent serving tail latency：SpecBox。
- Sparse attention / MoE kernel：PIVOT、DA-MoE。
- GRPO all-failed group：ProGPO。
- Reward server：SeekJudge。
- Agent context state：NOOA、CORVUS。
- BiDiRL 对照：DynaResize 作为 historical/boundary comparison，不占最新阅读优先级。

本次不直接修改 reading queue。Frontier Scan 负责形成判断；读者选中一条开始阅读后，再把它提升到 P0/P1，避免队列自动膨胀。

## 下一次扫描起点

- Next cursor：2026-07-28 16:58
- 下次继续扫描：
  - 新增 arXiv v1、technical report 与高质量 engineering blog；
  - OpenAI / Anthropic / NVIDIA / Hugging Face 官方更新；
  - AReaL / verl / slime / ROLL / OpenRLHF / NeMo RL / Molt release、major merge 与 rollback follow-up；
  - Kimi K3 完整报告中的 EP、persistent rollout / sandbox state、MXFP4 QAT 和部署实现；
  - Libra 是否公开代码，以及 sequence-pool attention 对现有 CP backend 的接入边界；
  - TRL sleep/wake fix 是否进入 release、是否补 policy-version regression test；
  - verl Qwen3.5 LoRA/MTP rollback 根因与替代实现；
  - 可迁移到 AReaL 的 attention workload sampler、weight restore verification、harness state contract 与 sandbox prewarming。

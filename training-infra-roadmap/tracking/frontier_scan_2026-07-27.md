# Frontier Scan, 2026-07-27

- Previous scan：[2026-07-24](frontier_scan_2026-07-24.md)
- Window：2026-07-24 09:52 ~ 2026-07-27 12:23
- Timezone：Asia/Shanghai
- Generated at：2026-07-27 12:23
- Report type：flexible frontier scan
- Sources scanned：arXiv cs.DC / cs.LG / cs.AI / cs.CL / cs.AR / cs.PF / cs.NI recent records；OpenAI / Anthropic / NVIDIA / Hugging Face official sources；AReaL / verl / slime / ROLL / OpenRLHF / NeMo RL releases and major PRs；新增 Molt / TRL 作为高相关框架观察项
- Scan completeness：覆盖上一游标后的 arXiv 公告、官方技术博客与 GitHub release / merged PR。Molt、Windowed-MTP 的 arXiv v1 早于上一游标，但此前未被记录，本次以 `boundary late-discovered` 补入并与旧 scan 去重。open PR 只作为方向观察，不写成已发布能力。性能数字均保留论文或项目方 attribution，未独立复现。

## 本次核心判断

本次共有八条值得保留的系统信号。最重要的不是论文数量，而是四条正在汇合的工程主线：

1. **OPD 正在突破同 tokenizer 限制。** BPM 把 Teacher 的 next-token distribution 映射到共享 byte space，使不同模型家族之间可以做 full-vocabulary on-policy distillation；这直接补上了 [MOPD](../topics/mopd.md) 面向异构 Teacher 时最现实的接口缺口。
2. **Agentic RL 框架开始重新争夺“可修改性”。** Molt 选择 PyTorch-native、token-first contract 和单异步循环，试图证明研究者不必为大规模异步 RL 接受一套难以读懂的厚重 runtime。
3. **Rollout 优化不只发生在 scheduler。** VIGOR 从算法侧动态分配 group rollout 数；OpenForgeRL 把真实 harness 与训练后端拆开；TRL 则暴露了 generation batch 和 gradient accumulation 不对齐时可能静默改变梯度尺度。
4. **权重生命周期正在成为训练与推理的共同基础设施。** ModelExpress 统一冷启动、peer fan-out、JIT cache 迁移和 RL refit；verl 的 sharded delta sync 则证明 trainer layout 到 inference layout 的增量转换必须由 backend 显式声明，而不是让通用传输层猜测。
5. **百万 token 推理会让原本“便宜”的 draft head 变成新瓶颈。** Windowed-MTP 只限制 MTP draft 的可见 KV、保留 target 全量验证，说明长上下文优化可以在不改变输出分布的前提下缩小 speculative decoding 的工作集。

## Accepted Frontier Signals

### Cross-Tokenizer On-Policy Distillation via Byte-Prefix Marginalization

- Signal ID：2026-07-27-001
- Source ID：arxiv:2607.22334
- First seen：2026-07-27 12:23（Asia/Shanghai，本次扫描）
- Focus Match：P0 Focus
- 来源：arXiv / project page
- 类型：paper / OPD / tokenizer alignment / distillation
- 链接：https://arxiv.org/abs/2607.22334
- 项目页：https://bpm-opd.github.io/
- Primary-source check：title / 7 位 authors / v1 time / BPM mechanism / `>99%` exact-position condition / `+3.7~6.6 avg@8` 已对齐 arXiv abs 页；项目页显示代码预计 1–2 个月后开放，当前不能写成已有开源实现
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：它移除了 full-vocabulary OPD 的 same-tokenizer 前提，使 Qwen、GLM、MiniMax 等异构 Teacher 的 dense token supervision 有了质量守恒、内容对齐的转换方式。
- Status：NEW
- 建议动作：作为下一篇精读候选；读 byte-prefix marginal、跨 token 边界 lower bound、whitespace mask 与通信 payload，再决定如何更新 [MOPD](../topics/mopd.md)
- 关联主题：[MOPD](../topics/mopd.md), [Agentic RL](../topics/agentic_rl.md), [Long-context Training](../topics/long_context_training.md)

BPM 不把 Teacher token 生硬映射到“看起来相近”的 Student token，而是在共享 byte space 中，把每个 Teacher token 的概率分配给其 byte prefix 对应的最长 Student token；无法对齐的质量进入显式 residual category。作者报告超过 99% 的训练位置可精确恢复 Teacher-induced byte-prefix marginal，其余位置采用保持概率质量的 chain-factorized lower bound。

这篇论文也给了一个很重要的反例：byte 对齐本身并不等于语义监督正确。在纯空白代码位置，BPM 会忠实转移 Teacher 的 tokenizer segmentation habit，未加 mask 时 code pass rate 从 49% 降到 9%。因此生产实现必须把 tokenizer bridge 当成带 failure mode 的训练组件，而不是无损格式转换。项目页当前明确写着 `Code In 1–2 Months`。

### ModelExpress: Distributing Model Artifacts at the Speed of Light

- Signal ID：2026-07-27-002
- Source ID：blog:nvidia/modelexpress-model-artifacts
- First seen：2026-07-27 12:23（Asia/Shanghai，本次扫描）
- Focus Match：P0 Focus
- 来源：NVIDIA Technical Blog
- 类型：official engineering blog / weight distribution / cold start / RL refit
- 链接：https://developer.nvidia.com/blog/modelexpress-distributing-model-artifacts-at-the-speed-of-light/
- Primary-source check：publication date / 5 位 authors / path priority / NIXL registration optimizations / DeepSeek-V4 Pro startup numbers / receiver-driven RL refit stages 已对齐 NVIDIA 原文；数字均为 NVIDIA-reported
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：它把“checkpoint 下载”“serving replica 扩容”“kernel cache 复用”和“trainer-to-rollout weight refit”统一成同一个 artifact discovery、layout compatibility 与高速传输问题。
- Status：NEW
- 建议动作：精读 Figure 1/2/7；对照 AReaL 当前 weight update 路径，拆出 source identity、layout plan、receiver pull、partial failure 与 checksum contract
- 关联主题：[Checkpointing](../topics/checkpointing.md), [Agentic RL](../topics/agentic_rl.md), [Distributed Training](../topics/distributed_training.md), [NCCL](../topics/nccl.md)

ModelExpress 优先复用已经驻留 GPU、完成 runtime layout 转换的兼容权重副本，通过 NIXL 做 P2P RDMA；没有 peer 时再退回 ModelStreamer、GDS 或普通 loader。兼容性由模型和 runtime layout 共同生成的 `mx_source_id` 约束，避免把“模型名相同”错误地当作 tensor layout 相同。

文章对 RL refit 给出四阶段 receiver-driven 流程：trainer ranks 发布 shard ownership，rollout worker 发现目标版本，receiver 根据自身 layout 规划来源，再执行 one-sided pull、convert 和 load。需要保持审慎：博客称客户正在评估这些 building blocks，delta cross-cluster refit 仍在测试，不能把全部路线图写成已稳定交付的能力。

### Molt: A Scalable PyTorch-Native Training Framework for Agentic Reinforcement Learning

- Signal ID：2026-07-27-003
- Source ID：arxiv:2607.21653
- First seen：2026-07-27 12:23（Asia/Shanghai，本次扫描；boundary late-discovered）
- Focus Match：P0 Focus
- 来源：arXiv / NVIDIA-NeMo official repository
- 类型：technical report / open-source RL framework / fully async / PyTorch native
- 链接：https://arxiv.org/abs/2607.21653
- 代码：https://github.com/NVIDIA-NeMo/labs-molt
- Primary-source check：title / 11 位 authors / v1 time / fully asynchronous claim / open-source repository 已对齐 arXiv；仓库包含 package、tests、examples、recipes、container 与 Apache-2.0 license
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：它不是又包一层 trainer API，而是把 token correctness、policy version、partial rollout、weight sync 和 FSDP2/MoE 并行压进一条可读的 PyTorch-native async loop，适合与 AReaL/verl 做代码级架构对照。
- Status：NEW
- 建议动作：先读 architecture、agent contract、async queue 和 token-span 数据结构，再核对论文中“与 Megatron stack 统计相当”的 matched protocol
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [FSDP](../topics/fsdp.md), [MoE](../topics/moe.md), [Distributed Training](../topics/distributed_training.md)

Molt 使用 Ray 做 placement 与 async queue、vLLM 做 rollout、NVIDIA AutoModel + FSDP2 做训练；仓库把 token ids、logprobs、action ranges、reward 和 multimodal tensors 作为同一条 token-first contract。其价值在于研究修改路径短，而不是“代码行少”本身：算法、环境与系统状态是否能沿同一条数据流被检查，决定了 Agentic RL 新 estimator 和新 rollout mode 的迭代成本。

论文只声称在 matched fully-async protocol 下与 SOTA Megatron-based stack 统计可比，并未证明它在所有规模上更快。后续应独立核对 1T-class MoE recipe、router replay/freeze、partial rollout 与 context compaction 的真实可运行边界。

### Learning as Reasoning Unfolds: Progressive Rollout Allocation for Efficient Reinforcement Learning

- Signal ID：2026-07-27-004
- Source ID：arxiv:2607.22002
- First seen：2026-07-27 12:23（Asia/Shanghai，本次扫描）
- Focus Match：P0 Focus
- 来源：arXiv
- 类型：paper / GRPO / rollout allocation / sample efficiency
- 链接：https://arxiv.org/abs/2607.22002
- Primary-source check：title / 3 位 authors / v1 time / VIGOR mechanism / `2.3x` and `1.49x` rollout reduction / `+3.4` coding average pass claims 已对齐 arXiv abs 页
- 影响等级：★★★★☆
- Decision：Read
- Reason：它直接处理固定 `n` GRPO 的无效 rollout：不是先生成更多再过滤，而是先少量采样，再把剩余预算逐轮分配给 group reward variance 高的 prompt。
- Status：NEW
- 建议动作：进入 P1 候选；重点确认 progressive allocation 是否会放大长样本尾延迟，以及 scheduler 如何在不阻塞 batch 的前提下追加同组 rollout
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [Rollout Latency](../playbooks/rollout_latency.md), [Long-context Training](../topics/long_context_training.md)

VIGOR 从每个 prompt 少量 rollout 开始，在固定总预算内，把新增 rollout 分给 group reward variance 最高的 prompt。它从 sampling policy 上减少无梯度或低信息 group，而不是直接优化推理 kernel。作者报告数学任务达到目标精度最多少用 2.3x rollout，coding 达到 GRPO 最终 full pass rate 少用 1.49x rollout。

对系统实现而言，关键问题是动态追加是否形成新的碎片化和尾部 group。一个合理的 runtime 需要支持 group-level reserve、增量生成、超时终止和可取消任务，否则算法节省的 token 可能被调度开销与长尾等待吃掉。

### OpenForgeRL: Train Harness-native Agents in Any Environment

- Signal ID：2026-07-27-005
- Source ID：arxiv:2607.21557
- First seen：2026-07-27 12:23（Asia/Shanghai，本次扫描；v2 位于当前窗口）
- Focus Match：P0 Focus
- 来源：arXiv
- 类型：paper / harness-native agent RL / Kubernetes rollout isolation
- 链接：https://arxiv.org/abs/2607.21557
- Primary-source check：title / 10 位 authors / v1-v2 time / proxy + Kubernetes orchestrator / benchmark numbers 已对齐 arXiv abs 页
- 影响等级：★★★★☆
- Decision：Read
- Reason：它把 Claude Code、Codex、OpenClaw 一类 stateful/multi-process harness 当作真实 rollout program，而不是要求 agent 行为先被重写成训练框架内部 DSL。
- Status：NEW
- 建议动作：进入 P1 候选；等代码公开后再把它纳入 framework implementation 对比，当前先读 proxy trace contract、container isolation、failure/retry 和 token ownership
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [Rollout Latency](../playbooks/rollout_latency.md), [Fault Tolerance](../topics/fault_tolerance.md)

OpenForgeRL 用轻量 proxy 接管并记录 harness 的 model calls，再把数据交给标准 RL backend（论文以 veRL 为例）；Kubernetes orchestrator 为每条 rollout 启动独立远端 container。这个边界允许训练端不理解 harness 内部每个进程，但会把 trace completeness、工具副作用、容器恢复与 attribution 变成新的 correctness contract。

arXiv 页面称其为 open-source framework，但正文当前仍写代码、数据和模型将发布，本次未找到可验证的公开仓库。因此这条信号按论文保留，不计入“已有可运行框架”。

### Windowed-MTP: Removing the Full-Context Draft-KV Tax at Million-Token Context

- Signal ID：2026-07-27-006
- Source ID：arxiv:2607.21535
- First seen：2026-07-27 12:23（Asia/Shanghai，本次扫描；boundary late-discovered）
- Focus Match：P1 Focus
- 来源：arXiv
- 类型：paper / speculative decoding / MTP / million-token inference
- 链接：https://arxiv.org/abs/2607.21535
- Primary-source check：title / author / v1 time / draft-only sliding window / `~99%` KV reduction / `28%~44%` per-step claim 已对齐 arXiv abs 页
- 影响等级：★★★★☆
- Decision：Read
- Reason：它指出百万 token 场景中 native MTP draft 的 full-attention KV read 会压过 draft compute，并给出不改变 target verification/output distribution 的 bounded-working-set 解法。
- Status：NEW
- 建议动作：进入 P1 候选；核对 acceptance length、draft depth、hybrid attention target 与 SGLang implementation，再判断是否值得做本地复现
- 关联主题：[Long-context Training](../topics/long_context_training.md), [FlashAttention](../topics/flashattention.md), [Agentic RL](../topics/agentic_rl.md)

Windowed-MTP 只给 draft attention 加 StreamingLLM-style sliding window 和 attention sink，target 仍做 full-attention verification。因此它改变“候选 token 怎么提议”，不改变“哪些 token 被最终接受”。作者在三类架构、1M context、单 GPU SGLang 上报告 per-decode-step cost 降低 28%–44%，并通过 ring buffer 回收占总 KV 7.7%–11% 的 unread draft KV。

这是一条 inference infra 信号，但会直接影响 RL rollout：当 trajectory context 极长时，MTP draft 若仍读全量 KV，原本用于降低 decode latency 的 speculative path 可能反而放大 rollout 成本。

### verl: Sharded Delta Block Placements and Backend-Owned HF Export

- Signal ID：2026-07-27-007
- Source ID：github:verl-project/verl#7144
- First seen：2026-07-27 12:23（Asia/Shanghai，本次扫描）
- Focus Match：P0 Focus
- 来源：verl GitHub
- 类型：merged PR / weight sync / FSDP2 / sharded delta
- 链接：https://github.com/verl-project/verl/pull/7144
- Primary-source check：PR title / merged time / block-placement contract / backend-owned HF export / H100 E2E numbers / equivalence checks 已对齐 merged PR；未独立复现实验
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：它把 delta sync 从 flat `Shard(0)` 推进到 `Shard(k)`、multi-shard mesh 和 manual split，并明确“如何转换到 HF coordinates”应由训练 backend 负责。
- Status：NEW
- 建议动作：代码级阅读 `BlockPlacement`、HF delta export、seed/full fallback、receiver checksum；与 AReaL 的 weight sync layout contract 对照
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [FSDP](../topics/fsdp.md), [Checkpointing](../topics/checkpointing.md), [Distributed Training](../topics/distributed_training.md)

PR 把每个 rank 的本地 shard 描述为 full tensor 中的 hyper-rectangular block，并让 backend 输出最终 HF-coordinate delta entries；通用 delta engine 只负责 collectives、bucketing 和 wire protocol。这样可以 fail loud 地拒绝尚未实现的 expert/manual layout，而不是把本地位置误解释为全局参数位置。

项目方在 H100 GSM8K GRPO 上报告 7B weight update 约 1.3x、32B 1.55x、32B offload-off 2.3x、72B 2.3x speedup，并做了 200-step / 400-sync reward equivalence 与 receiver checksum 检查。需要注意 EP-aware veomni export 仍在后续 PR，本次不能外推到所有 MoE layout。

### Hugging Face TRL v1.9.1: Correct GRPO Loss Normalization

- Signal ID：2026-07-27-008
- Source ID：github:huggingface/trl@v1.9.1
- First seen：2026-07-27 12:23（Asia/Shanghai，本次扫描）
- Focus Match：P0 Focus
- 来源：Hugging Face TRL release / merged PR
- 类型：release / GRPO correctness / loss normalization
- 链接：https://github.com/huggingface/trl/releases/tag/v1.9.1
- 修复 PR：https://github.com/huggingface/trl/pull/6024
- Primary-source check：release time / affected loss types / mis-scaling factor / fix / regression-test configurations 已对齐 release 与 merged PR
- 影响等级：★★★★★
- Decision：Read
- Reason：这是一类不会 crash、甚至 loss curve 看起来仍可训练，但实际梯度被静默放大或缩小的 RL correctness bug；它直接影响实验可比性。
- Status：NEW
- 建议动作：检查所有自有 recipe 中 `steps_per_generation` 与 `gradient_accumulation_steps`；对 AReaL/verl 等实现核对 generation batch token denominator 的语义
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [Distributed Training](../topics/distributed_training.md), [Long-context Training](../topics/long_context_training.md)

在 DAPO/CISPO/VESPO loss 中，`num_items_in_batch` 统计整个 generation batch 的 completion tokens，覆盖 `steps_per_generation` 个 micro-steps；optimizer 却只在 `gradient_accumulation_steps` 个 micro-steps 后更新。两者不相等时，累计梯度会额外乘上 `gradient_accumulation_steps / steps_per_generation`：例如 `4/2` 配置得到预期梯度的 0.5x，`8/32` 得到 4x。

修复在 train mode 把 normalizer 乘以 `current_gradient_accumulation_steps / steps_per_generation`，并覆盖 final partial accumulation window。默认两者相等的配置不受影响，GRPO/SAPO/BNPO/DR-GRPO/LUSPO 的其他 normalization path 也不受本 bug 影响。

## Observed / Rejected Candidates

| 材料 | Source ID | Decision | 原因 |
|---|---|---|---|
| Scaling Native Multimodal Pre-Training From Scratch | arxiv:2607.22043 | Observe | compute scaling 与 modality allocation 有价值，但当前主贡献是 scaling law，不提供足以改变训练 runtime 的新机制 |
| Enough is as good as a feast: Does model merging scale with Reinforcement Learning? | arxiv:2607.22039 | Observe | 讨论 RL 后模型融合冲突，和 MOPD 有对照价值；当前主要是算法/经验结论，不先进入主线 |
| Adversarial Prompts for Acceptance Collapse in Speculative Decoding | arxiv:2607.21804 | Observe | 揭示 speculative decoding acceptance-rate attack 与 latency 风险；需要更多真实服务流量和防御开销证据 |
| Multi-turn RL with Structural and Performance Aware Rewards for CUDA Kernel Generation | arxiv:2607.20908 | Observe | CUDA kernel agent 与 reward design 相关，但重点是任务算法，不直接改变 RL runtime |
| HiKV | arxiv:2607.22389 | Observe | KV compression + custom accelerator 的速度/能耗数字亮眼，当前硬件依赖强，先看可复现性与通用 GPU 对照 |
| Unified Static-Dynamic Pruning for Scalable Acceleration of Large Language Model Inference | arxiv:2607.21985 | Observe | 自定义 sparse format/kernel 有 inference 价值；尚未形成训练/rollout 侧的直接判断 |
| RIS-Kernel | arxiv:2607.21927 | Reject | 单作者 CPU sparse attention 证据范围窄，当前不足以改变主线工程决策 |
| NVIDIA Nemotron agentic RTL material | official NVIDIA source | Reject / no core signal | 主要是领域模型与 benchmark 展示，没有新的训练、rollout 或 inference runtime 机制 |

## OpenAI / Anthropic / NVIDIA Watch

| Vendor | Sources checked | Decision | 结果 |
|---|---|---|---|
| OpenAI | official research index / newsroom | Not found / no core signal | 截至本次结束时刻，游标后未发现新的 training、RL、agent runtime 或 inference infra 技术正文。 |
| Anthropic | official news / research | Observed + Rejected | 7 月 24 日 Claude Opus 5 是重要模型发布，但公开材料未披露足以改变训练/runtime 判断的系统设计；Project Pilot 属 frontier red-team / drone control，超出当前主线。两者均未静默省略。 |
| NVIDIA | Technical Blog / NeMo repositories | Accepted + Observed | ModelExpress 与 Molt 进入 Accepted。NeMo RL 已合入 TRT-LLM SWE rollout 支持，验证目标是与 vLLM 收敛对齐，PR 明确性能优化留待后续；ModelExpress trainer integration 仍是 open umbrella，已有 11-step EFA run 但正式 parameter/generation parity 尚未完成，因此只作 Observe。 |

## Hugging Face Watch

| Source | Source type | Decision | 结果 |
|---|---|---|---|
| Hugging Face Blog | official blog index | Not found / no core signal | 游标后未发现新的 Agentic RL、distributed training、long-context 或 LLM inference backend 正文；Nunchaku diffusion 仍不进入当前主线。 |
| TRL v1.9.1 | official release | Accepted | DAPO/CISPO/VESPO accumulation-window normalization 修复会改变真实梯度尺度，属于训练正确性而非普通 patch。 |
| Transformers / Accelerate / PEFT / Kernels | official releases | Not found / no material release | 游标后没有符合当前 focus filter 的正式 release；不使用普通 commit 补数量。 |
| Community Articles | community index | Observe only | 本窗口没有证据强到足以进入 Accepted 的社区文章；官方团队文章与社区文章继续分开判断。 |

## RL Framework Watch

本节区分已合入能力、open direction 和普通维护。没有正式 release 的框架不使用 routine commit 补位。

| Framework | Change | Subsystem | Evidence / state | Decision | 对 AReaL 的参考 |
|---|---|---|---|---|---|
| AReaL | [PR #1516](https://github.com/areal-project/AReaL/pull/1516)：grouped reward normalization / drop incomplete group | rollout / group semantics | merged；37 tests；保留 original reward 并避免残缺 group 进入训练 | Read | group completeness 应是显式 trajectory contract；需要继续确认失败 rollout 的重试、drop 与统计口径 |
| AReaL | [PR #1444](https://github.com/areal-project/AReaL/pull/1444)：Qwen3.6 LoRA GRPO recipes | training / rollout / model support | merged；覆盖 27B 与 35B-A3B、8xA800 validation，并修复 LoRA save OOM/hot reload/offload | Observe | 作为模型接入与 LoRA lifecycle 案例；不是新的 runtime architecture |
| AReaL | PR #1564：SGLang PP weight-sync gate | weight sync / correctness | open；修复 pipeline-parallel layout mismatch 的方向合理 | Observe | 合入前不记为正式能力；应补 weight-layout compatibility test |
| verl | [PR #7144](https://github.com/verl-project/verl/pull/7144)：sharded delta block placements | weight sync / FSDP / checkpoint engine | merged；本次 Accepted，含 H100 E2E 与 checksum evidence | Deep Dive | 重点迁移 backend-owned layout export、seed/full fallback 和 fail-loud unsupported placement |
| verl | [PR #7139](https://github.com/verl-project/verl/pull/7139)：SGLang NCCL buffer-view race | weight sync / correctness | merged；recv buffer view 在下一次 broadcast 前被覆盖会造成参数 layer shift | Read | 发送完成不等于 consumer 已拥有稳定 storage；CUDA IPC/NCCL handoff 必须验证 tensor ownership 与 lifetime |
| slime | PR #1709 Mooncake RDMA / #2238 fully-async completed-group drop / #2235 CP advantage whitening | data path / rollout / training correctness | 均为 open PR | Observe | 方向分别对应 trajectory RDMA、group lifecycle 与 CP/DP normalization；未合入前不作为框架能力 |
| ROLL | PR #476 short-response advantage whitening fix | training correctness | merged；范围窄 | Observe / routine | 作为 mask/normalization edge case 记录，不升级为 frontier signal |
| OpenRLHF | PR #1272 variance-aware dynamic filtering | rollout / group sampling | open；constant continuous-reward group 即使 mean 合法也可能产生零 advantage | Observe | 与 VIGOR 互补：先过滤零方差 group，再决定是否追加高方差 group；合入和 E2E 前不写成正式功能 |
| NeMo RL | [PR #3130](https://github.com/NVIDIA-NeMo/RL/pull/3130)：TRT-LLM SWE rollout | rollout / inference backend | merged；在 4-node GB200/GB300 与 16-node H100 配置上核对前期 convergence，PR 明确性能优化后续进行 | Read | backend parity 先于 performance；应对照 token/logprob、stop reason、tool trace 与 refit version |
| NeMo RL | PR #3068 ModelExpress v2 trainer integration | weight sync / scheduler | open umbrella；11/11 EFA steps、framework refit median 2.965 s，但状态仍为 `PERF_PASS_CORRECTNESS_PENDING` | Observe | receiver-driven shard publishing 与 topology-aware RDMA 值得跟踪；formal parameter equality / generation parity 未完成 |
| Molt | PyTorch-native fully async framework | rollout / training / scheduler | open-source repo，含 package、tests、recipes 和 container；本次 Accepted | Deep Dive | 对照 AReaL 的 controller/worker 边界，看 token-first single-loop 能否降低算法改动扩散面 |
| Hugging Face TRL | v1.9.1 GRPO loss normalization | training correctness | official release；本次 Accepted | Read | 检查 AReaL 是否把 generation batch token count 与 optimizer accumulation window token count混为同一 denominator |

### Framework Follow-up

- verl #7139 是值得沉淀的 ownership lesson：NCCL receive buffer 的 tensor view 即使 shape/bytes 正好匹配，也可能在 CUDA IPC consumer 使用前被下一次 broadcast 覆盖；不能只用 `nbytes == numel * element_size` 判断是否需要 clone。
- NeMo RL #3068 已经把 ModelExpress refit 跑进真实 GRPO，但作者仍把状态标为 correctness pending。后续只有在 parameter equality、generation parity、committed-version agreement 与 failed-refit safety 完成后，才适合升级成 Accepted framework capability。
- OpenRLHF #1272 与 VIGOR 指向同一资源浪费源的两个层次：前者丢弃零方差 group，后者把追加 rollout 预算分给高方差 group。两者都可能改变 group completion 与 scheduler queue 语义。
- AReaL / verl / slime / ROLL / OpenRLHF / NeMo RL 在本窗口均无新的正式 release；Molt 是新增的可运行 emerging framework，TRL 因正式 release 和训练正确性被加入扩展观察。

## 本次阅读决策

### P0 候选

1. **BPM / Cross-Tokenizer OPD**：直接接上刚建立的 MOPD 主线，先回答异构 Teacher 如何给 Student 提供 dense token target。
2. **ModelExpress**：把 cold start 与 RL refit 放在同一套 artifact lifecycle 中，最适合形成 AReaL weight-sync 工程判断。
3. **Molt**：先读架构与 token contract，再判断它是“更薄的可研究框架”，还是把复杂度转移到了 AutoModel/Ray/vLLM。

### P1 候选

- VIGOR：与 GRPO 固定 `n`、零方差 group 和长尾等待直接相关。
- verl #7144：深入 backend-owned shard export 与 delta sync correctness。
- TRL v1.9.1：快速检查自有 recipe 是否受 silent gradient scaling 影响。
- Windowed-MTP：补齐 long-context rollout 的 speculative decoding 成本模型。
- OpenForgeRL：先读 harness trace 与 rollout isolation；等待公开代码后再做实现级判断。

本次不直接修改 reading queue。P0/P1 是候选，不是自动塞入队列；先由读者选择当前要读的一条，避免队列继续膨胀。

## 下一次扫描起点

- Next cursor：2026-07-27 12:23
- 下次继续扫描：
  - 新增 arXiv v1 与高价值技术报告；
  - OpenAI / Anthropic / NVIDIA / Hugging Face 官方更新；
  - AReaL / verl / slime / ROLL / OpenRLHF / NeMo RL 正式 release 与 major PR merge/benchmark 状态；
  - Molt 的真实规模配置、issue/PR 与 independent reproduction；
  - BPM 代码是否按项目页计划开放；
  - OpenForgeRL 是否发布可验证仓库；
  - NeMo RL ModelExpress integration 是否完成 parameter/generation parity；
  - 可迁移到 AReaL 的 rollout allocation、weight refit、token correctness、trajectory transport 与 recovery 设计。


# Frontier Scan, 2026-07-13

- Previous scan：[2026-07-10](frontier_scan_2026-07-10.md)
- Window：2026-07-10 10:48 ~ 2026-07-13 15:35
- Timezone：Asia/Shanghai
- Generated at：2026-07-13 15:35
- Report type：flexible frontier scan
- Sources scanned：arXiv cs.DC / cs.LG / cs.AI / cs.CL v1 records；OpenAI / Anthropic / NVIDIA / Hugging Face official sources；vLLM / SGLang / Transformers / TRL / Accelerate / PEFT / PyTorch / Megatron-Core / DeepSpeed / verl GitHub releases
- Scan completeness：扫描并去重 144 条窗口内 arXiv v1 记录；官方博客和主要框架 release 已核验。部分厂商博客只公开日期、不公开时分秒，已按 boundary late-discovered 记录，不虚构精确发布时间。

## 本次核心判断

本窗口出现了三条相互咬合的主线：**RL rollout/train 资源开始双向借用，verifier 开始直接消费 generation KV cache，serving runtime 则继续把 speculative decoding、PD disaggregation、collective fault detection 和长上下文 kernel 下沉到默认路径。** 这说明 Agentic RL Infra 的优化边界正在从“把 rollout 跑快”扩展为调度、验证、KV 状态、故障隔离和训练/推理运行时共同参与的闭环。

第二条强信号来自 NVIDIA：host offloading 的价值不再只是“省 HBM”，而是由 CPU-GPU coherent bandwidth、compiler scheduling、copy stream 和 NCCL overlap 共同决定的 activation placement 问题。模型结构本身也开始被 GEMM tile、精度格式和并行布局反向约束。

## Accepted Frontier Signals

### Bidirectional Resource Scheduling for Disaggregated and Asynchronous RL Post-Training

- Signal ID：2026-07-13-001
- Source ID：arxiv:2607.09207
- First seen：2026-07-13
- Scan window：2026-07-10 10:48 ~ 2026-07-13 15:35
- Focus Match：P0 Focus
- 来源：arXiv
- 类型：paper / Agentic RL system
- 链接：https://arxiv.org/abs/2607.09207
- 发布时间：2026-07-10
- Primary-source check：title / authors / submitted time / abstract / two 32-GPU testbeds / 1.94x claim 已对齐 arXiv abs 页
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：BiDiRL 不只拆开 rollout 与 training，而是允许两侧 GPU 在不同阶段双向借用，并用 hot-switch runtime 与 schedule-aware planner 管理资源切换。
- Status：NEW
- 建议动作：进入 [P1](../reading_queue/P1.md)，与 AReaL / DORA / HybridFlow 对照资源所有权、权重切换和 staleness 边界
- 预计阅读：1.5h
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [Rollout Latency](../playbooks/rollout_latency.md), [Distributed Training](../topics/distributed_training.md)

作者在两个 32-GPU testbed 上报告相对 veRL、AReaL 和 ROLL 最高 1.94x 加速，且不改变收敛结果。真正值得验证的不是峰值数字，而是资源借用期间 model state、KV cache、通信域和失败恢复的切换成本是否能在真实长尾 rollout 中稳定摊薄。

### KV-PRM: Efficient Process Reward Modeling via KV-Cache Transfer for Multi-Agent Test-Time Scaling

- Signal ID：2026-07-13-002
- Source ID：arxiv:2607.09153
- First seen：2026-07-13
- Scan window：2026-07-10 10:48 ~ 2026-07-13 15:35
- Focus Match：P0 Focus
- 来源：arXiv
- 类型：paper / verifier inference system
- 链接：https://arxiv.org/abs/2607.09153
- 发布时间：2026-07-10
- Primary-source check：title / authors / submitted time / O(L^2) to O(L) claim / 5000x FLOPs / 37x latency / 34x memory claims 已对齐 arXiv abs 页
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：它让 process reward model 直接读取 generation 阶段 KV cache，并用单个 verification token 完成评分，避免 verifier 重新编码整条 trajectory。
- Status：NEW
- 建议动作：进入 [P1](../reading_queue/P1.md)，重点检查 KV layout、模型兼容性、跨 worker 传输和 cache 生命周期
- 预计阅读：1.5h
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [Long-context Training](../topics/long_context_training.md), [Rollout Latency](../playbooks/rollout_latency.md)

作者报告 scoring FLOPs 最高减少 5000x、latency 降低 37x、memory 降低 34x。这些数字是特定实验结果，不能直接外推到生产；但“generation state 能否被 verifier 复用”是极强的系统问题，会直接影响 rollout / verifier co-location、KV 传输协议和多模型数值兼容性。

### COBS: Cumulant Order Block Sparse Attention

- Signal ID：2026-07-13-003
- Source ID：arxiv:2607.09052
- First seen：2026-07-13
- Scan window：2026-07-10 10:48 ~ 2026-07-13 15:35
- Focus Match：P1 Focus
- 来源：arXiv
- 类型：paper / sparse attention
- 链接：https://arxiv.org/abs/2607.09052
- 发布时间：2026-07-10
- Primary-source check：title / authors / submitted time / 32K RULER / KV traffic claims 已对齐 arXiv abs 页
- 影响等级：★★★★☆
- Decision：Read
- Reason：它用压缩的二阶统计量选择 block，在明显低于 dense KV traffic 的前提下，尝试补回现有 sparse selector 在 retrieval-heavy 长上下文任务上的质量缺口。
- Status：NEW
- 建议动作：暂不进入 P1；先等待 kernel / code 与真实 serving latency 数据，再决定是否升级为实验
- 关联主题：[Long-context Training](../topics/long_context_training.md), [FlashAttention](../topics/flashattention.md), [Rollout Latency](../playbooks/rollout_latency.md)

论文在 32K RULER 上报告均值 0.8195，dense 为 0.9040，NSA 为 0.2999；KV traffic 为 NSA 的 1.21x、dense 的约 1/15.15。当前最需要追问的是 selector 开销、prefill/decode 分别受益多少，以及 block pattern 是否能被现有 kernel 稳定利用。

### vLLM v0.25.0

- Signal ID：2026-07-13-004
- Source ID：github:vllm-project/vllm@v0.25.0
- First seen：2026-07-13
- Scan window：2026-07-10 10:48 ~ 2026-07-13 15:35
- Focus Match：P0 Focus
- 来源：vLLM GitHub Release
- 类型：release note / inference runtime
- 链接：https://github.com/vllm-project/vllm/releases/tag/v0.25.0
- 发布时间：2026-07-12 04:06（Asia/Shanghai）
- Primary-source check：tag / published time / Model Runner V2 / sequence parallel / collective fault detection / PD disaggregation 条目已对齐 GitHub release
- 影响等级：★★★★★
- Decision：Read
- Reason：Model Runner V2 成为 dense model 默认路径，同时补入 no-DP sequence parallel、NCCL symmetric-memory AG/RS、all-to-all peer fault detection、DP supervisor 和多项 PD/KV transfer 能力。
- Status：NEW
- 建议动作：与 SGLang v0.5.15 作为一个 release comparison 进入 [P1](../reading_queue/P1.md)
- 预计阅读：1h
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [NCCL](../topics/nccl.md), [Long-context Training](../topics/long_context_training.md), [Rollout Latency](../playbooks/rollout_latency.md)

官方 release 报告 no-DP sequence parallel 带来 1.9%~5.0% end-to-end throughput 改善，并加入 all-to-all peer fault detection 以避免 silent corrupted output。对 RL Infra 最关键的是 sleep/offload、weight sync、DP/PD 调度和 failure semantics 是否能被上层 trainer 可靠消费，而不只是单模型吞吐。

### SGLang v0.5.15

- Signal ID：2026-07-13-005
- Source ID：github:sgl-project/sglang@v0.5.15
- First seen：2026-07-13
- Scan window：2026-07-10 10:48 ~ 2026-07-13 15:35
- Focus Match：P0 Focus
- 来源：SGLang GitHub Release
- 类型：release note / inference runtime
- 链接：https://github.com/sgl-project/sglang/releases/tag/v0.5.15
- 发布时间：2026-07-11 06:58（Asia/Shanghai）
- Primary-source check：tag / published time / Spec V2 / IndexShare MTP / PD routing / RL sleep-wake 条目已对齐 GitHub release
- 影响等级：★★★★★
- Decision：Read
- Reason：Spec V2 默认启用、IndexShare MTP、decode CP、NIXL PD disaggregation、DP-aware routing 和 RL engine sleep/wake 同时推进，覆盖长上下文 rollout 的 decode、KV、调度和资源切换。
- Status：NEW
- 建议动作：与 vLLM v0.25.0 作为一个 release comparison 进入 [P1](../reading_queue/P1.md)
- 预计阅读：1h
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [Context Parallelism](../topics/context_parallelism.md), [Long-context Training](../topics/long_context_training.md), [Rollout Latency](../playbooks/rollout_latency.md)

官方 release 报告 Spec V2 带来 11% end-to-end TPS 改善，IndexShare MTP 在长上下文下最高降低 1.9x draft-step cost。应把这些数字视为 release benchmark，而不是通用结论；更重要的是与 vLLM 对照两者如何管理 CUDA Graph、PD routing、KV connector、RL sleep/wake 和故障恢复。

### Reducing HBM Bottlenecks in JAX-Based LLM Training with Host Offloading

- Signal ID：2026-07-13-006
- Source ID：blog:nvidia/jax-llm-host-offloading
- First seen：2026-07-13
- Scan window：boundary late-discovered；官方页面仅标注 2026-07-10，无精确发布时间
- Focus Match：P0 Focus
- 来源：NVIDIA Technical Blog
- 类型：engineering blog / training memory placement
- 链接：https://developer.nvidia.com/blog/reducing-high-bandwidth-memory-bottlenecks-in-jax-based-llm-training-with-host-offloading/
- 发布时间：2026-07-10（官方页面，仅日期）
- Primary-source check：title / authors / date / 128-GPU GB200 setup / throughput and memory claims 已对齐 NVIDIA 正文
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：它把 activation offloading、pinned host memory、NVLink-C2C、XLA latency hiding、copy stream 和 NCCL overlap 放到一套可测量的训练路径中。
- Status：NEW
- 建议动作：暂不扩充 P1；在下一次 P1 清理时优先提升，并与 activation checkpointing / ZeRO-Offload 做系统对照
- 预计阅读：1h
- 关联主题：[Long-context Training](../topics/long_context_training.md), [Checkpointing](../topics/checkpointing.md), [Distributed Training](../topics/distributed_training.md), [MoE](../topics/moe.md)

在 128 张 GB200 上，官方报告 DeepSeek-V3 671B 从 activation rematerialization 的 578.3 提升到 908.2 TFLOPs/s/device，提升 57%；Llama 3.1 405B 的固定配置只提升 2.9%。这个差异非常重要：offload 不是固定收益开关，取决于 activation 体积、重计算成本、可重叠工作和 CPU-GPU 互联。

### AI Model Co-Design: Hardware-Friendly LLM Design

- Signal ID：2026-07-13-007
- Source ID：blog:nvidia/hardware-friendly-llm-design
- First seen：2026-07-13
- Scan window：boundary late-discovered；官方页面仅标注 2026-07-10，无精确发布时间
- Focus Match：P1 Focus
- 来源：NVIDIA Technical Blog
- 类型：engineering blog / model-system co-design
- 链接：https://developer.nvidia.com/blog/ai-model-co-design-hardware-friendly-llm-design/
- 发布时间：2026-07-10（官方页面，仅日期）
- Primary-source check：title / authors / date / GB300 NVFP4 GEMM / 256K prefill PP claims 已对齐 NVIDIA 正文
- 影响等级：★★★★★
- Decision：Read
- Reason：它给出模型维度、tile alignment、NVFP4、TP/EP/PP 与 serving latency 的具体共同设计约束，说明模型 shape 本身就是 infra 参数。
- Status：NEW
- 建议动作：暂不进入 P1；后续沉淀到 TP / MoE / FP8 时作为硬件约束来源
- 关联主题：[Tensor Parallelism](../topics/tensor_parallelism.md), [MoE](../topics/moe.md), [FP8](../topics/fp8.md), [Long-context Training](../topics/long_context_training.md)

文章报告 GB300 NVFP4 GEMM 在 K/N 约 6144 附近饱和，达到 80% sustained throughput 约需 K > 3072、N > 2560，并建议维度至少按 128、优先 256/512 对齐。其 256K DeepSeek-R1 prefill 案例还展示 PP size 从 1 扩到 32 时降低 first-token latency；这些是硬件与特定 workload 相关的指导，不应机械变成所有模型的 shape 规则。

### Profiling in PyTorch (Part 3): Attention is all you profile

- Signal ID：2026-07-13-008
- Source ID：blog:huggingface/torch-attention-profile
- First seen：2026-07-13
- Scan window：boundary late-discovered；官方页面仅标注 2026-07-10，无精确发布时间
- Focus Match：P1 Focus
- 来源：Hugging Face Blog（official-team post）
- 类型：engineering blog / kernel profiling
- 链接：https://huggingface.co/blog/torch-attention-profile
- 发布时间：2026-07-10（官方页面，仅日期）
- Primary-source check：title / authors / date / tested A100 setup / profiler numbers 已对齐 Hugging Face 正文
- 影响等级：★★★★☆
- Decision：Read
- Reason：它用 profiler trace 解释 kernel count、dtype upcast、Tensor Core eligibility 与 memory traffic 为什么比单看 occupancy 更能定位 attention 性能问题。
- Status：NEW
- 建议动作：暂不进入 P1；作为 FlashAttention benchmark 与 128K SFT profiling 的实操参考
- 关联主题：[FlashAttention](../topics/flashattention.md), [Long-context Training](../topics/long_context_training.md), [Experiments](../experiments/README.md)

文章在特定 A100-SXM4-80GB shape 上报告 SDPA math backend forward 为 7.239 ms，而 Flash backend 为 146.8 us。数字不应跨 shape 外推；真正可复用的是排障方法：先看 trace 中实际选择了哪个 backend、发生了多少 kernel launch、是否隐式 upcast，再讨论 occupancy。

## Observed / Rejected Candidates

| 材料 | Source ID | Focus Match | Decision | 原因 |
|---|---|---|---|---|
| Mach-Mind-4-Flash Technical Report | arxiv:2607.09375 | P1 Focus | Observe | 35B-A3B MoE、OPD/RL infra 与 17% E2E speedup 有价值，但模型报告信息密度较大，先等系统实现和独立复现 |
| STEEL: Sparse Fused Attention on AMD XDNA NPU | arxiv:2607.09385 | P1 Focus | Observe | kernel/energy 结果具体，但硬件与编程栈较窄，迁移到当前 GPU infra 的直接价值有限 |
| Attention to Detail: vLLM Configurations | arxiv:2607.09172 | P1 Focus | Observe | 9000-run energy/performance/accuracy 测量有方法价值，先观察 benchmark 可复用性 |
| Scoped Verification for Long-Horizon Context Evolution | arxiv:2607.09175 | P1 Focus | Observe | verifier/harness 判断相关，但缺少足够强的 rollout system 细节 |
| Kernel Fusion in NVIDIA CUDA | blog:nvidia/kernel-fusion | P1 Focus | Observe | 手工 fusion / torch.compile / cuda.compute 对照有教学价值；3x 来自单个 RTX 4090 microbenchmark，不升级为前沿信号 |
| Claude Code v2.1.207 | github:anthropics/claude-code@v2.1.207 | P1 Focus | Observe | agent-team crash loop、worktree recovery、AWS stall guard 和 plugin shell injection 修复值得看，但更偏 agent runtime 产品可靠性 |
| Codex CLI 0.144.2 | github:openai/codex@rust-v0.144.2 | P1 Focus | Observe | Guardian auto-review 因 prompt regression 回滚，是 verifier policy 敏感性的真实案例；范围较窄，不进入主队列 |
| Transformers v5.13.1 | github:huggingface/transformers@v5.13.1 | P1 Focus | Observe | 明确服务 vLLM 0.25 的 custom model / linear layer 兼容性，属于重要配套 patch，不单独升级 |
| BioNeMo Agent Toolkit Co-Folding | blog:nvidia/bionemo-agent-cofolding | Outside core | Observe | Fold-CP 和 32K/64-way CP 有可迁移系统线索，但主要场景是 biomolecular co-folding |
| Robot Policy Evaluation for Deployment | blog:nvidia/robot-policy-evaluation | Outside core | Observe | rollout 数量、置信区间与 failure taxonomy 很扎实，但场景偏 robotics evaluation |
| Sovereign 30B-A3B Technical Report | arxiv:2607.09424 | Low match | Reject | 通用模型报告，当前没有足以改变训练/推理 infra 判断的独立系统细节 |

## OpenAI / Anthropic / NVIDIA Watch

| Vendor | Sources checked | Decision | 结果 |
|---|---|---|---|
| OpenAI | official news / research / Codex changelog and releases | Observe | 窗口内没有新的 training/RL/inference infra 正文；Codex CLI 0.144.2 的 Guardian prompt regression 回滚作为 verifier reliability 案例保留观察，0.144.3 无实际变更。 |
| Anthropic | official newsroom / research / Claude Platform notes / Claude Code releases | Observe | 没有新的 training/RL/inference research post；Claude Code v2.1.207 的 agent-team、session recovery 和 plugin trust-boundary 修复保留观察。 |
| NVIDIA | Technical Blog / research / NCCL / NeMo / TensorRT-related entry points | Accepted | JAX host offloading 与 hardware-friendly LLM design 进入 accepted；kernel fusion、BioNeMo Fold-CP 与机器人评估保留观察。7 月 10 日文章无精确时刻，按 boundary late-discovered 去重。 |

### Hugging Face Watch

| Sources checked | Decision | 结果 |
|---|---|---|
| Hugging Face Blog / TRL / Transformers / Accelerate / PEFT / Kernels releases | Accepted / Observe | official-team attention profiling 进入 accepted；Transformers v5.13.1 作为 vLLM 0.25 兼容 patch 观察。窗口内未发现新的 TRL / Accelerate / PEFT / Kernels release；community posts 未达到当前 infra 筛选门槛。 |

## Reading Queue Updates

- [ ] 保持 [P0](../reading_queue/P0.md) 不变，不用新鲜度替代当前学习优先级。
- [x] 加入 [P1](../reading_queue/P1.md)：BiDiRL，补 rollout/train 双向资源借用与 hot-switch scheduling。
- [x] 加入 [P1](../reading_queue/P1.md)：KV-PRM，补 generation KV cache 到 verifier 的状态复用路径。
- [x] 加入 [P1](../reading_queue/P1.md)：vLLM v0.25.0 / SGLang v0.5.15 对照阅读，作为一个任务而非两个 release 摘抄。
- [ ] 暂不扩队：NVIDIA host offloading / model co-design、COBS、HF attention profiling；先在下次 P1 清理时竞争优先级。

## 去重记录

- 本次新增 Source ID：arxiv:2607.09207, arxiv:2607.09153, arxiv:2607.09052, github:vllm-project/vllm@v0.25.0, github:sgl-project/sglang@v0.5.15, blog:nvidia/jax-llm-host-offloading, blog:nvidia/hardware-friendly-llm-design, blog:huggingface/torch-attention-profile
- `arxiv:2607.09052` 的 v1 时间为 2026-07-10 10:48:43（Asia/Shanghai），比上次 cursor 晚 43 秒，属于本窗口而不是重复项。
- NVIDIA / Hugging Face 的 7 月 10 日正文没有精确发布时间；上次 scan 未记录这些 Source ID，本次按 boundary late-discovered 补入，后续不得重复 accepted。
- `blog:huggingface/tito` 已在 [2026-05 Backfill](backfill/2026-05.md) 与 [P1](../reading_queue/P1.md) 记录，本次不重复收录。

## 扫描完整性

- 已扫描：窗口内 144 条 arXiv v1 记录；OpenAI / Anthropic / NVIDIA / Hugging Face 官方入口；vLLM / SGLang / Transformers / TRL / Accelerate / PEFT / PyTorch / Megatron-Core / DeepSpeed / verl release。
- 日期校验：arXiv 与 GitHub release 使用可核验时刻；只提供日期的官方博客明确标注 boundary，不假定其早于或晚于 10:48。
- 限制：GitHub API 请求期间出现 rate limit，重点 release 已回退到官方 release HTML / Atom 核验；未发 release 的 main-branch commit 不做全量追踪。
- 已知盲区：厂商内部未公开材料、无稳定 changelog 的 docs 增量、只在会议现场或社交平台传播且无 primary source 的信息不进入 accepted。

## 下一步动作

- [x] 更新 [Scan Log](scan_log.md)。
- [ ] 阅读 BiDiRL，画出 rollout pool / training pool 双向借用时的 model state 与 communicator 生命周期。
- [ ] 阅读 KV-PRM，确认 KV cache 跨 generation/verifier 复用所需的模型、位置编码和 layout 一致性。
- [ ] 对照 vLLM 0.25 与 SGLang 0.5.15 的 RL sleep/wake、PD routing、KV connector、collective fault handling 和 long-context optimization。
- [ ] P1 下次清理时，优先评估是否提升 NVIDIA JAX host offloading；不继续无上限堆积 release note。

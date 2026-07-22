# Frontier Scan, 2026-07-20

- Previous scan：[2026-07-13](frontier_scan_2026-07-13.md)
- Window：2026-07-13 15:35 ~ 2026-07-20 11:12
- Timezone：Asia/Shanghai
- Generated at：2026-07-20 11:12
- Report type：flexible frontier scan
- Sources scanned：arXiv cs.DC / cs.LG / cs.AI / cs.CL / cs.AR / cs.PF / cs.NI recent records；OpenAI / Anthropic / NVIDIA / Hugging Face official sources；vLLM / SGLang / Transformers / TRL / Accelerate / PEFT / Kernels / Megatron-Core / DeepSpeed / verl official releases
- Scan completeness：覆盖 arXiv 最近列表中 2026-07-14 至 2026-07-17 的可见公告，并逐条回到 abs 页核对 accepted signal。2026-07-20 上午尚未出现新的周一 arXiv 公告；厂商滚动文档与未发布 commit 不在本次完整性承诺内。

## 本次核心判断

本窗口最强的信号不是某个新模型，而是三类系统边界同时移动：

1. **Agentic RL 正把低精度数值一致性和百万 token execution capacity 变成训练基础设施问题。** QUADS 说明 rollout 用 NVFP4、trainer 用 BF16 时，activation quantization error 会扩大 rollout-trainer log-prob gap；LongStraw 则把 shared prompt、response branch replay 和模型状态保留带进 2M-token GRPO execution path。
2. **Inference Infra 的瓶颈继续下沉。** 小消息 collective 的微秒级延迟、Hybrid SWA 的 KV placement、RDMA cache、mixed-dtype fusion correctness 和长输入 FP4 MoE kernel 都直接影响 decode critical path。
3. **集群控制面开始从“预测”转向“排序与适应”。** Agora 证明异构、易失、互联网连接的节点可以完成真实 8.6B pretraining；HeaRank 则表明生产 GPU 故障更适合做风险排序，而不是执着于精确预测故障时刻。

## Accepted Frontier Signals

### Every Microsecond Matters: Achieving Near Speed-of-Light Latency in GPU Collectives

- Signal ID：2026-07-20-001
- Source ID：arxiv:2607.16100
- First seen：2026-07-18 00:36（Asia/Shanghai）
- Focus Match：P0 Focus
- 来源：arXiv
- 类型：paper / GPU collective
- 链接：https://arxiv.org/abs/2607.16100
- Primary-source check：title / 13 位 authors / v1 time / NCCL device-side API / symmetric memory / multicast / 7% SoL claim 已对齐 arXiv abs 页
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：decode-heavy LLM 的关键路径上存在大量小消息 collective，优化目标不再只是带宽，而是逼近 scale-up network 的绝对 latency lower bound。
- Status：NEW
- 建议动作：不立即扩 P1；先与 NCCL Device API、LL/LL128 和 vLLM/SGLang tensor-parallel decode trace 对照
- 关联主题：[NCCL](../topics/nccl.md), [Tensor Parallelism](../topics/tensor_parallelism.md), [Long-context Training](../topics/long_context_training.md)

作者基于 NCCL device-side API 构建 symmetric collectives，使用 barrier-free synchronization、symmetric memory 和 multicast，将小中消息的 overhead 压到 absolute hardware Speed-of-Light 下界的 7% 以内。最值得验证的是收益落在哪些 message size、GPU 数和 NVLink/NVSwitch topology，而不是只记住峰值。

### QUADS: Stabilizing NVFP4 Reinforcement Learning for MoE via Quantization-error Alignment across Dual Sides

- Signal ID：2026-07-20-002
- Source ID：arxiv:2607.15810
- First seen：2026-07-17 18:21（Asia/Shanghai）
- Focus Match：P0 Focus
- 来源：arXiv
- 类型：paper / low-precision RL system
- 链接：https://arxiv.org/abs/2607.15810
- Primary-source check：title / authors / v1 time / roughly 150-step collapse / 21.49-point pass@1 / ~16% throughput claims 已对齐 arXiv abs 页
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：它把 NVFP4 rollout 的失败定位为 training-inference activation error 与 log-prob gap，而不是笼统归因为 FP4 精度不足。
- Status：NEW
- 建议动作：作为 AReaL / veRL 低精度 rollout 的实验候选，先核对开源实现、支持模型和 quantization path
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [FP8](../topics/fp8.md), [MoE](../topics/moe.md), [Rollout Latency](../playbooks/rollout_latency.md)

QUADS 在 trainer 侧使用 asymmetric QAT，在 rollout 侧补偿高误差 activation channels，同时保留 native W4A4 GEMM。它给出的工程判断很明确：RL rollout 加速不能只测吞吐，还必须持续监控 rollout/trainer log-prob gap、policy consistency 和数值漂移。

### LongStraw: Long-Context RL Beyond 2M Tokens under a Fixed GPU Budget

- Signal ID：2026-07-20-003
- Source ID：arxiv:2607.14952
- First seen：2026-07-16 21:00（Asia/Shanghai）
- Focus Match：P0 Focus
- 来源：arXiv
- 类型：paper / long-context RL execution stack
- 链接：https://arxiv.org/abs/2607.14952
- Primary-source check：title / 20 位 authors / v1 time / 8xH20 / 32xH20 / 2.1M / 4.46M claims / correctness limitation 已对齐 arXiv abs 页
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：它将 shared-prompt no-autograd evaluation、model-state retention 和 response branch replay 组合成 million-token GRPO execution path。
- Status：NEW
- 建议动作：先读 execution graph 与 correctness limitation，不把“能跑通”误写成“能正确训练”
- 关联主题：[Long-context Training](../topics/long_context_training.md), [Agentic RL](../topics/agentic_rl.md), [Context Parallelism](../topics/context_parallelism.md)

作者在 8xH20 上走通 Qwen3.6-27B 的 2.1M-position grouped scoring/backward，并在 32xH20 上验证 GLM-5.2 全 78 层的 2.1M-token path。但论文明确承认 prompt state 被 detached，部分 distributed forward 和 gradient composition 尚未完成，因此当前价值是 execution capacity 证据，不是完整训练正确性。

### Agora: Collective and Permissionless Internet-Scale Pretraining of Large Language Models

- Signal ID：2026-07-20-004
- Source ID：arxiv:2607.13332
- First seen：2026-07-15 07:32（Asia/Shanghai）
- Focus Match：P1 Focus
- 来源：arXiv
- 类型：paper / distributed pretraining system
- 链接：https://arxiv.org/abs/2607.13332
- Primary-source check：title / authors / v1 time / 8.6B / 500B tokens / 40 days / 330 nodes / 63% centralized efficiency claims 已对齐 arXiv abs 页
- 影响等级：★★★★★
- Decision：Read
- Reason：它不是 simulator 或小规模 demo，而是在异构、易失、互联网连接的 330 个 contributor nodes 上完成真实 8.6B pretraining。
- Status：NEW
- 建议动作：观察其 pipeline sharding、asynchronous optimization、fault-tolerant collective 与 security/ownership 边界
- 关联主题：[Distributed Training](../topics/distributed_training.md), [Pipeline Parallelism](../topics/pipeline_parallelism.md), [Fault Tolerance](../topics/fault_tolerance.md)

Agora 报告约 170K tokens/s、4.2 tokens/TFLOP，并达到 centralized H100 baseline 约 63% 的效率。它不会替代数据中心训练，但为跨地域低带宽、preemptible compute 的 pipeline ownership 与恢复协议提供了少见的完整证据。

### Don't Predict, Prioritize: Rethinking GPU Reliability Assessment

- Signal ID：2026-07-20-005
- Source ID：arxiv:2607.15115
- First seen：2026-07-16 23:24（Asia/Shanghai）
- Focus Match：P0 Focus
- 来源：arXiv / KDD 2026
- 类型：paper / GPU cluster reliability
- 链接：https://arxiv.org/abs/2607.15115
- Primary-source check：title / authors / v1 time / production cluster / AUC 0.83 / top-5% captures 64% claims 已对齐 arXiv abs 页
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：它用数千 GPU 的生产数据说明 DBE / GPU Lost 具有强随机性，精确预测失败时刻不如对节点做相对风险排序。
- Status：NEW
- 建议动作：沉淀到 fault-tolerance 时重点讨论 risk-aware placement、preventive drain 与 spare priority
- 关联主题：[Fault Tolerance](../topics/fault_tolerance.md), [Distributed Training](../topics/distributed_training.md), [MegaScale](../tech_reports/megascale.md)

HeaRank 在线上将未来故障覆盖率从现网方案的 21% 提升到 top-5% 节点中的 64%。真正可迁移的不是模型本身，而是控制面决策：当 failure timing 不可预测时，用风险排序影响 job placement 和维护优先级。

### Full-Pipeline Inference Optimization for MiMo-V2.5 Series

- Signal ID：2026-07-20-006
- Source ID：arxiv:2607.13095
- First seen：2026-07-14 11:38（Asia/Shanghai）
- Focus Match：P0 Focus
- 来源：arXiv technical report
- 类型：technical report / production inference system
- 链接：https://arxiv.org/abs/2607.13095
- Primary-source check：title / Xiaomi MiMo Team authors / v1 time / Hybrid SWA / GCache / RDMA / KV-affinity router claims 已对齐 arXiv abs 页
- 影响等级：★★★★★
- Decision：Read
- Reason：它不是单点 kernel 优化，而是围绕 Hybrid SWA + MoE + multimodal composite architecture 重做 KV storage、prefetch、prefix tree、placement、distributed cache 和 routing。
- Status：NEW
- 建议动作：与 vLLM / SGLang 的 PD disaggregation、KV connector 和 cache-aware routing 对照
- 关联主题：[Long-context Training](../topics/long_context_training.md), [MoE](../topics/moe.md), [Agentic RL](../topics/agentic_rl.md)

报告把 SWA 的理论 O(W) 优势落实为 layerwise prefetch、SWA-aware prefix cache tree、GCache RDMA networking 和 KV-cache-affinity routing。对 RL Infra 的直接价值是：rollout serving 的 attention pattern、KV locality 与 router placement 必须联合设计。

### vLLM v0.25.1 / SGLang v0.5.15.post1 Correctness Patches

- Signal ID：2026-07-20-007
- Source IDs：github:vllm-project/vllm@v0.25.1；github:sgl-project/sglang@v0.5.15.post1
- First seen：2026-07-14
- Focus Match：P0 Focus
- 来源：official GitHub releases
- 类型：release note / inference correctness
- 链接：https://github.com/vllm-project/vllm/releases/tag/v0.25.1；https://github.com/sgl-project/sglang/releases/tag/v0.5.15.post1
- Primary-source check：release date / mixed-dtype fusion corruption / FP4 MoE long-input NaN / GLM-5.2 PD+CP IndexShare fixes 已对齐官方 release
- 影响等级：★★★★★
- Decision：Read
- Reason：两次 patch 都不是普通兼容性修复，而是会产生 hidden-state corruption、重复乱码或长输入 NaN 的静默正确性问题。
- Status：NEW
- 建议动作：升级前增加 mixed-dtype fusion、long-input FP4 MoE、PD disaggregation 和 CP 的数值回归
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [FP8](../topics/fp8.md), [Context Parallelism](../topics/context_parallelism.md), [Rollout Latency](../playbooks/rollout_latency.md)

这组 release 再次说明：推理框架性能升级不能只跑 throughput benchmark。graph fusion 的 dtype guard、低精度 MoE 长序列输出、PD/CP 状态共享都必须有 correctness oracle 和退化输出监控。

## Observed / Rejected Candidates

| 材料 | Source ID | Decision | 原因 |
|---|---|---|---|
| Branching Policy Optimization | arxiv:2607.14171 | Observe | sandbox snapshot/fork 改变 rollout topology 很有价值，但状态复制、并发调度和存储成本尚未系统量化 |
| Where Should RL Post-Training Compute Go? | arxiv:2607.13389 | Observe | GRPO FLOP accounting 将 search / learning / feedback 拆开很实用，但当前证据主要来自 LoRA Qwen2.5 pilot grid |
| M+Adam | arxiv:2607.10611 | Observe | additive + multiplicative update 正面处理低精度 master weight freeze；规模到 1B，距离大规模训练结论仍有距离 |
| GPU-Tile-Sim | arxiv:2607.11262 | Observe | tile graph 对 GEMM/attention modeling 很有价值，MICRO 2026 且 MAPE 1.22%~8.71%；当前更偏硬件设计工具 |
| PagedWeight | arxiv:2607.16184 | Observe | 最高 72% GPU memory saving 与 1.94x throughput 值得跟踪，但需核对动态量化成本与 workload 泛化 |
| Less Experts, Faster Decoding / EcoSpec | arxiv:2607.12696 | Observe | expert scattering 与 speculative MoE decoding 相关，但 peak speedup 尚不足以代表高并发 serving |
| AAFLOW+ | arxiv:2607.10987 | Observe | distributed KV cache 作为一等对象的方向正确；核心结果来自 empirical-microbenchmark-parameterized analytical model |
| MemoHarness | arxiv:2607.14159 | Observe | harness 从静态配置变为 experience-conditioned control layer 值得观察，但不是当前训练系统主线 |
| Quota Marketplace | arxiv:2607.09802 | Read / gap correction | Google 已部署、OSDI 2026，直接讨论异质价值下的 GPU quota allocation；v1 早于本次 cursor，作为上次扫描遗漏项记录，不冒充本窗口新提交 |
| NVIDIA NeMo RL Agentic Autoresearch | blog:nvidia/nemo-rl-autoresearch | Observe | 实验 ledger、分支、恢复和审批设计可复用，但当前更像 workflow reference，不单独占用阅读队列 |
| Transformers v5.14.0 / v5.14.1 | github:huggingface/transformers@v5.14 | Read | FSDP2、StaticCache+FlashAttention、MTP、MoE/DeepGEMM 变化较集中；升级前需做 checkpoint 和 kernel regression |
| NeMo Automodel x Hugging Face Diffusers | blog:huggingface/nemo-automodel-diffusers | Observe | DTensor-native FSDP2/TP/EP/CP/PP 与 checkpoint interoperability 有价值，但当前场景以 diffusion training 为主 |

## OpenAI / Anthropic / NVIDIA Watch

| Vendor | Sources checked | Decision | 结果 |
|---|---|---|---|
| OpenAI | official newsroom / research / engineering | Rejected / no core signal | 窗口内未发现新的 training、RL 或 inference infra 工程文章；安全与投资类内容不进入当前雷达。 |
| Anthropic | official news / research / engineering | Rejected / no core signal | 7 月 14 日的 education / grants / Economic Index 与当前 infra filter 不匹配；没有窗口内工程正文。 |
| NVIDIA | Technical Blog / NeMo / Megatron / NCCL related entry points | Accepted / Observe | NCCL device-side collective paper进入 accepted；NeMo RL Autoresearch 与 NeMo Automodel distributed training 保留观察。Megatron-Core 无窗口内正式 release。 |

## Hugging Face Watch

| Sources checked | Decision | 结果 |
|---|---|---|
| Hugging Face Blog / Transformers / TRL / Accelerate / PEFT / Kernels | Read / Observe | Transformers v5.14.0/v5.14.1 进入 Read；NVIDIA 联合 NeMo Automodel 文章保留观察。TRL、Accelerate、PEFT、Kernels 无可验证的窗口内正式 release。 |

## Framework Release Watch

| Framework | Decision | 结果 |
|---|---|---|
| vLLM | Accepted | v0.25.1 修复 mixed-dtype allreduce + RMSNorm + static quantization fusion 对 hidden state 的破坏。 |
| SGLang | Accepted | v0.5.15.post1 修复长输入 FP4 MoE NaN，以及 GLM-5.2 在 PD disaggregation / CP 下的 IndexShare 问题。 |
| Megatron-Core | Not found | 最新正式 Core release 早于窗口。 |
| DeepSpeed | Not found | 无窗口内正式 release。 |
| verl | Not found | 无窗口内正式 release。 |

## Reading Queue Updates

- [x] 保持 [P0](../reading_queue/P0.md) 不变：AReaL / HybridFlow / Rollout Infrastructure Tax 仍是当前学习主线。
- [x] 暂不向已经较长的 [P1](../reading_queue/P1.md) 增加条目；本次 accepted signal 先在 tracking 中竞争优先级。
- [ ] 当前一小时阅读建议：QUADS 或 LongStraw 二选一；前者补低精度 rollout correctness，后者补 million-token RL execution。

## 去重与窗口修正

- 本次新增 accepted Source ID：arxiv:2607.16100, arxiv:2607.15810, arxiv:2607.14952, arxiv:2607.13332, arxiv:2607.15115, arxiv:2607.13095, github:vllm-project/vllm@v0.25.1, github:sgl-project/sglang@v0.5.15.post1
- `arxiv:2607.09802` 的 v1 时间早于上次 cursor，但此前未记录。本次只作为 gap correction，不把它伪装为本窗口新提交。
- `arxiv:2607.10611` 的 v1 早于 cursor、v2 位于窗口内；当前只保留观察，不因普通修订重复 accepted。
- `arxiv:2607.14107` Polestar 的 arXiv ID 位于本月，但 v1 实际提交于 2026-05-07，不进入本次 frontier。

## 扫描完整性

- 已扫描：arXiv cs.DC / cs.LG / cs.AI / cs.CL / cs.AR / cs.PF / cs.NI recent listings；accepted signal 均回到 arXiv abs 或官方 release 核对 metadata 与数字。
- 官方来源：OpenAI / Anthropic / NVIDIA / Hugging Face，以及 vLLM / SGLang / Transformers / TRL / Accelerate / PEFT / Kernels / Megatron-Core / DeepSpeed / verl 的 release/blog 入口。
- 限制：GitHub API 触发匿名 rate limit 后回退到官方 Releases/Tags HTML；未扫描普通 PR、issue 或未发布 commit。
- 时间边界：本次截止到 2026-07-20 11:12，不包含之后发布的内容。此时 arXiv recent 页面最新公告仍为 2026-07-17。
- 证据边界：数字均为作者或官方 release 报告，不代表独立复现。LongStraw 已明确标注 execution capacity 不等于 complete training correctness。

## 下一步动作

- [x] 更新 [Scan Log](scan_log.md)。
- [ ] 先读 QUADS：画出 BF16 trainer、NVFP4 rollout、weight sync、activation quantization 与 log-prob gap 的因果链。
- [ ] 再读 LongStraw：确认 shared prompt no-grad、retained state、branch replay 与 GRPO loss 的正确性边界。
- [ ] 将 vLLM/SGLang 的 correctness patch 转为最小回归清单，而不是只记录 release note。
- [ ] 下次 monthly signal 再决定哪些条目进入正式沉淀，不因“新”自动挤入 P0/P1。

# Frontier Scan, 2026-08-18

- Previous scan：[2026-08-17](frontier_scan_2026-08-17.md)
- Window：2026-08-17 09:28:33 ~ 2026-08-18 09:31:44
- Timezone：Asia/Shanghai
- Generated at：2026-08-18 09:31:44
- Report type：flexible frontier scan
- Sources scanned：arXiv cs.DC / cs.LG / cs.AI / cs.CL / cs.PL recent announcement；OpenAI / Anthropic / NVIDIA / DeepSeek official sources；Hugging Face Blog / model cards / TRL / Transformers / Accelerate / PEFT / Kernels；AReaL / verl / slime / ROLL / OpenRLHF / NeMo RL；Megatron-LM / vLLM / SGLang / Molt official default-branch changes
- Scan completeness：完成本窗口 arXiv 新公告批次、四家核心厂商、Hugging Face 生态、六个 RL 框架和四个相邻 runtime 的逐项检查。高活跃 GitHub 仓库的 Atom feed 只保留最近约 20 个 commit，已结合 PR 页面交叉核验，但不能证明被 feed 截断的普通 commit 全部可见。

## 本次核心判断

这次真正值得留下的不是“又出现几个新模型”，而是 **RL post-training 正在同时重构计算时间线、长上下文并行边界和训推状态生命周期**。

1. **Rollplex 把同步 RL 的串行阶段拆开，但没有放弃 on-policy。** 它不是允许旧 policy 提前生成，而是把与 response 无关的 prefix computation 移入 rollout decode 的空闲算力窗口；这是 cross-phase spatial sharing，不是 staleness 换吞吐。
2. **长上下文 RL 开始得到原生 Context Parallel 支持。** Megatron-LM 已把 CP 接入 packed/unpacked RL path、Transformer Engine 和 CUDA Graph，说明 CP 不再只是 pretraining 配置项。
3. **Inference backend 的 sleep/resume 语义已经影响 weight sync 正确性。** verl 的修复说明 vLLM 和 SGLang 即使都叫 `sleep_level=1`，释放的 state 也不同；backend adapter 不能只统一函数名。
4. **Speculative decoding 必须返回真实 accepted-token logprob。** SGLang 为 DSpark 接入 output logprobs，直接决定 RL trajectory 中 behavior policy probability 是否可信。
5. **Quantization 正从 PTQ 后处理变成训练流程的一部分。** NVIDIA 的 Nemotron 3.5 QAD recipe 使用 BF16 teacher、NVFP4 student 和 KL distillation，并公开中间 checkpoint 的恢复曲线；但最终 checkpoint 并非所有汇总指标都优于 PTQ，不能只看宣传数字。
6. **MoE inference 正把迁移开销移出 critical path。** FreeBalance 在 router 之前预测 residual workload，将 expert migration 与 attention 等前置计算重叠；核心价值是提前调度，不只是重新摆放 expert。

## Accepted Frontier Signals

### Rollplex：在同步 On-Policy RL 内重叠 Prefix Compute 与 Rollout Decode

- Signal ID：2026-08-18-001
- Source ID：arxiv:2608.14498
- First seen：2026-08-18 09:31:44
- Announcement：2026-08-17；v1 submitted 2026-08-14
- Scan window：2026-08-17 09:28:33 ~ 2026-08-18 09:31:44
- Focus Match：P0 Focus
- 来源：arXiv primary page / paper
- 类型：paper / RL runtime / cross-phase GPU sharing
- 链接：https://arxiv.org/abs/2608.14498
- Primary-source check：title、11 位作者、submission time、165 GiB memory motivation、phase-aware HBM、TP-aware physical weight sharing、32×H800 benchmark 与 synchronous update claim 已对齐 arXiv abstract
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：它针对 VLM post-training 中 prefix/video processing 占比高、rollout decode compute utilization 低的问题，在不引入 policy staleness 的情况下重排三阶段时间线，和 BiDiRL/AWEX/TMax 形成很有价值的对照。
- Status：NEW
- 建议动作：先画 serial colocation、disaggregation、Rollplex 三条 timeline，再确认哪些 tensor 可共享物理存储、哪些 TP layout 必须重构
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [Rollout Latency](../playbooks/rollout_latency.md), [Long-context Training](../topics/long_context_training.md)

传统同步 runtime 按 rollout、reference scoring、actor training 串行推进。VLM 的 dense video/prompt prefix 在多个阶段重复出现，而 decode 阶段往往受 memory bandwidth 和串行 token generation 限制，留下未充分使用的 compute。Rollplex 将 reference/training 的 prefix 部分提前到 rollout decode window，与 decode 并发执行，但 response 仍来自当前 policy，因此不需要用更旧 rollout 换吞吐。

难点不只是并发 launch：论文指出 Qwen2.5-VL-32B 的 naive colocation 每卡约需 165 GiB，而且 rollout 与 training 偏好的 TP degree、weight layout 不同。phase-aware memory management 按 producer-consumer lifetime 控制 HBM residency；parallelism-aware weight sharing 让 layout-compatible tensors 共用物理存储，只重构不兼容部分。论文在 32×H800 上报告相对 serial colocation `1.23x–1.30x`、相对 disaggregation `1.57x–2.24x`，数字属于论文实验结果，尚未由本仓复现。

### NVIDIA Nemotron 3.5 Lightning：用 QAD 恢复 NVFP4 Agent 能力

- Signal ID：2026-08-18-002
- Source ID：blog:nvidia/nemotron-3-5-lightning-nvfp4-qad
- First seen：2026-08-18 09:31:44
- 发布时间：2026-08-18 02:12，Asia/Shanghai
- Scan window：2026-08-17 09:28:33 ~ 2026-08-18 09:31:44
- Focus Match：P0 Focus
- 来源：NVIDIA Technical Blog / official recipe
- 类型：engineering blog / quantization-aware distillation / NVFP4
- 链接：https://developer.nvidia.com/blog/developing-nemotron-3-5-lightning-nvfp4-with-qad-using-nvidia-model-optimizer/
- Primary-source check：PTQ→QAD pipeline、BF16 teacher、W4A16-NVFP4 student、KL loss、Megatron-LM/Bridge、Model Optimizer recipe、footprint 与 quality-recovery numbers 已对齐官方文章；数字明确标为 NVIDIA-reported
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：它提供的是从 BF16 checkpoint 到可部署 NVFP4 agent model 的完整工业 recipe，而不是只公布一个量化权重；尤其展示了不同 checkpoint 在质量恢复与部署约束之间的取舍。
- Status：NEW
- 建议动作：拆解 PTQ initialization、teacher/student forward、KL target、checkpoint selection 和 deployment validation 五个阶段，判断哪些环节可迁移到现有 Megatron/Model Optimizer 流程
- 关联主题：[FP8 / Low Precision](../topics/fp8.md), [Transformer Engine](../topics/transformer_engine.md), [Agentic RL](../topics/agentic_rl.md)

QAD 以 frozen BF16 teacher 为目标，让 W4A16-NVFP4 student 通过 KL distillation 恢复量化损失。NVIDIA 报告模型 footprint 从 65.85 GB 降至 21.19 GB，并声称最高约 4 倍吞吐；这些都是厂商结果，不能直接外推到其他模型、batch 或 runtime。

更值得注意的是 checkpoint selection：部分中间 checkpoint 的 median quality recovery 从 96.33% 提升到 99.72%、从 95.84% 提升到 98.53%；但最终 conservative checkpoint 的 QAD median 98.97% 略低于 PTQ 99.24%，只是若干 agentic benchmark 更好。工程结论不是“QAD 必胜 PTQ”，而是量化训练需要多指标、按 workload 选择 checkpoint。

### Megatron-LM 为 RL Training 接入 Context Parallelism

- Signal ID：2026-08-18-003
- Source ID：github:NVIDIA/Megatron-LM#5882
- First seen：2026-08-18 09:31:44
- Scan window：2026-08-17 09:28:33 ~ 2026-08-18 09:31:44
- Focus Match：P0 Focus
- 来源：Megatron-LM merged PR / default branch
- 类型：framework implementation / RL training / context parallelism
- 链接：https://github.com/NVIDIA/Megatron-LM/pull/5882
- Primary-source check：merged status、13 commits、packed/unpacked RL、TE、CUDA Graph 与 `CP>1` 修复已对齐官方 PR/commit history
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：长 trajectory 进入 RL training 后，attention activation 不再能只靠 TP/DP 处理；CP 必须同时维护 token partition、action mask、advantages、logprobs 与 loss reduction 的对齐。
- Status：NEW
- 建议动作：追踪 `sequence_packing_utils.py` 和 RL loss path，检查 CP scatter/gather 前后的 position、mask、advantage 与 denominator
- 关联主题：[Context Parallelism](../topics/context_parallelism.md), [Long-context Training](../topics/long_context_training.md), [Agentic RL](../topics/agentic_rl.md)

该合并不是简单放开一个 `context_parallel_size` 参数，而是覆盖 packed 与 unpacked RL path，并处理 Transformer Engine、CUDA Graph 与 `CP>1` 的组合。它标志着 long-context RL 正在进入 Megatron 的标准并行栈。

对 AReaL 的直接风险点是：模型 token 可以按 CP 切分，但 reward、advantage、action mask 和旧 policy logprob 不能因 scatter/gather 改变语义。实现阅读应优先检查有效 token denominator、跨 CP group reduction 以及 packed boundary，而不是只验证 forward 能跑通。

### FreeBalance：在 Router 之前预测 MoE Workload

- Signal ID：2026-08-18-004
- Source ID：arxiv:2608.14205
- First seen：2026-08-18 09:31:44
- Announcement：2026-08-17；v1 submitted 2026-08-14
- Scan window：2026-08-17 09:28:33 ~ 2026-08-18 09:31:44
- Focus Match：P0 Focus
- 来源：arXiv primary page / paper
- 类型：paper / MoE inference / online expert balancing
- 链接：https://arxiv.org/abs/2608.14205
- Primary-source check：title、5 位作者、submission date、residual workload predictor、cost model、load ratio 与 prefill latency 已对齐 arXiv abstract
- 影响等级：★★★★☆
- Decision：Read
- Reason：在线 expert balance 的主要问题不是能不能迁移，而是 routing 结果出来后再迁移已经落入 critical path；FreeBalance 用跨层 residual similarity 提前预测下一层负载。
- Status：NEW
- 建议动作：阅读 predictor input、migration granularity、错误预测成本和 expert state movement，判断对 DeepSeek-style EP serving 的可迁移性
- 关联主题：[MoE](../topics/moe.md), [NCCL](../topics/nccl.md), [Distributed Training](../topics/distributed_training.md)

FreeBalance 利用 residual network 中跨层 hidden representation 的相似性，在目标 router 执行前预测 rank/expert workload，并把 expert migration 与 attention 等前置计算重叠。cost model 限制 swap 数量，使同步开销不超过可隐藏窗口。

论文报告 max-to-mean rank load ratio 降低 32.8%、端到端 prefill latency 降低 13.1%，平均每层隐藏约 5.1 个 expert 的 balancing overhead。它是 inference 系统工作；是否适用于 training/rollout 仍取决于 expert weight 更新频率、迁移一致性和可用 overlap window。

### verl 修正 vLLM Sleep/Resume 与 Weight Sync 的状态语义

- Signal ID：2026-08-18-005
- Source ID：github:verl-project/verl#7434
- First seen：2026-08-18 09:31:44
- Scan window：2026-08-17 09:28:33 ~ 2026-08-18 09:31:44
- Focus Match：P0 Focus
- 来源：verl merged/default-branch fix
- 类型：framework implementation / weight sync / lifecycle correctness
- 链接：https://github.com/verl-project/verl/pull/7434
- Primary-source check：vLLM/SGLang level-1 sleep 差异、CuMemAllocator unmap、LoRA buffer、`cudaErrorInvalidValue` 与 resume-before-sync 路径已对齐官方 PR/commit
- 影响等级：★★★★☆
- Decision：Read
- Reason：同一个抽象参数在不同 inference backend 下释放的 state 不同；错误复用 SGLang 假设会让 vLLM weight/LoRA sync 写入已解除 VA mapping 的 buffer。
- Status：NEW
- 建议动作：把 AReaL 各 inference backend 的 sleep/resume contract 写成 state matrix：weights、LoRA、KV cache、CUDA VA、host backup、communicator 分别处于什么状态
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [Distributed Training](../topics/distributed_training.md), [Rollout Latency](../playbooks/rollout_latency.md)

SGLang 的 level-1 release 只释放 KV cache，base weights 仍映射，因此可以跳过 weights resume。vLLM level-1 sleep 则通过 CuMemAllocator 备份并 `unmap_and_release` 带 `weights` tag 的分配，其中包括 LoRA buffers；若直接执行 `copy_()` 同步，会写到未映射 VA 并触发 `cudaErrorInvalidValue`。

修复要求 vLLM 在同步前恢复 weights。真正可迁移的设计是让 backend 明确报告 state，而不是调度层根据相同的 sleep level 数字猜测生命周期。

### SGLang 为 DSpark Speculative Decoding 提供 Accepted-Token Logprobs

- Signal ID：2026-08-18-006
- Source ID：github:sgl-project/sglang#34478
- First seen：2026-08-18 09:31:44
- Scan window：2026-08-17 09:28:33 ~ 2026-08-18 09:31:44
- Focus Match：P0 Focus
- 来源：SGLang merged PR / default branch
- 类型：framework implementation / speculative decoding / rollout correctness
- 链接：https://github.com/sgl-project/sglang/pull/34478
- Primary-source check：merged status、output-logprob request、shared speculative-v2 processor、accepted-token logprob 与 integration coverage 已对齐官方 PR
- 影响等级：★★★★☆
- Decision：Read
- Reason：RL rollout 不能只返回正确 token；behavior policy logprob 必须对应最终 accepted tokens，否则 importance ratio、KL 或 off-policy correction 都可能被污染。
- Status：NEW
- 建议动作：对照 AReaL trajectory schema，确认 speculative draft/reject/accept 后保存的是 target distribution、draft distribution 还是实际 sampling distribution
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [Rollout Latency](../playbooks/rollout_latency.md), [Long-context Training](../topics/long_context_training.md)

实现允许 DSpark 请求 output logprobs，并统一使用 speculative-v2 processor 计算 accepted-token logprob，同时启用已有 grammar/logprob integration coverage。它的价值不是“多返回一个字段”，而是保持 trajectory 中 token 与 behavior probability 一一对应。

## Observed / Rejected Candidates

| 材料 | Source ID | Focus Match | Decision | 原因 |
|---|---|---|---|---|
| AReaL Qwen3-VL model-owned THD | github:areal-project/AReaL#1608 | P0 | Read | packing 前合并 multimodal embeddings，并加入 alignment/parity/distributed-forward tests；是重要 VLM trajectory/training 边界，但当前不改变 text-only RL 主线。 |
| vLLM FlashInfer NVLink one-sided All2All | github:vllm-project/vllm#51924 | P0 | Read | 增加 DeepSeek Blockwise FP8 payload/scale layout 校验与 MoE sequence parallelism，并在 8×B300 上做模型验证；实现扎实，但尚无独立端到端 speedup。 |
| SimpleOPD | arxiv:2608.14277 | P1 | Read | tokenizer-agnostic alignment 与 termination-token advantage masking 解决 OPD correctness 问题，直接关联 [MOPD](../topics/mopd.md)；缺少 rollout runtime 证据。 |
| The Integer Alibi | arxiv:2608.13756 | P1 | Read | 将 CUTLASS/Triton INT8 序列分歧定位到 scale application/output rounding，提醒 kernel deterministic 不等于 cross-kernel equivalent；当前规模与 workload 较窄。 |
| Envs-FORGE | arxiv:2608.14312 | P1 | Read | 联合生成 instruction、fixture、oracle、tests 和 Docker environment，并用 gold verification 过滤；更偏 environment synthesis/data pipeline，不是 runtime scheduling。 |
| NeMo RL fine-grained activation CPU offload | github:NVIDIA-NeMo/RL#2279 | P1 | Observe | 可按 `moe_act/core_attn/qkv_linear/...` 模块 offload，支持 dense/MoE；未给出峰值显存和 tokens/s，需评估与 rollout overlap 的资源竞争。 |
| verl `fsdp_turbo` backend | github:verl-project/verl#7362 | P1 | Observe | 注册式 trainer engine 支持 CUDA/NPU，适合对照 AReaL engine registry；暂未看到 checkpoint/rollout compatibility 的完整生产证据。 |
| Transformers inference correctness bundle | github:huggingface/transformers@2026-08-18 | P1 | Observe | 包含 sliding-window cache wraparound、Blackwell MoE grouped-mm crash 与 DFlash device placement 修复；重要但属于三项局部 correctness fix。 |
| NVIDIA PORTS-Pike AI-factory commitment | blog:nvidia/ports-pike-infrastructure | P1 | Observe | 4 GW 级土地/电力/shell 与长期租赁说明 power/site/financing 已成 scaling constraint；容量、GPU 数和收入均为 NVIDIA forecast，不是已完成部署。 |
| vLLM DeepEP-v2 receiver CPU overhead | github:vllm-project/vllm#51114 | P1 | Observe | 用 `repeat_interleave` 消除 per-expert host loop，报告每 engine step 约减少 13 ms CPU 开销；局部优化但提醒 rollout profile 要拆 host routing prep。 |
| KV Cache Compression Through the Lens of Transform Coding | arxiv:2608.14191 | P1 | Observe | attention-aware bit allocation 报告约 5.8x KV compression，但尚无 serving latency、throughput 或 kernel 实现证据。 |

明确拒绝：Barrier-Free Synchronization 虽有 AWS Neuron ISA 实现和 10%–45% kernel latency 报告，但离当前 GPU/RL 主线较远；DeaMoE、HBF/ReRAM serving、QUASAR 等保留在原始检索结果，不进入本次正式雷达；纯 CI、文档、模型适配和 community promotion 不单独记信号。

## OpenAI / Anthropic / NVIDIA / DeepSeek Watch

| Vendor | Sources checked | Decision | 结果 |
|---|---|---|---|
| OpenAI | official research / engineering / releases | Not found | 本窗口未发现新的 Training/RL/Inference Infra 技术报告或工程发布；sitemap 的批量 `lastmod` 不作为发布时间证据。 |
| Anthropic | official research / engineering / newsroom | Not found | 本窗口未发现新的训练系统、RL Infra、agent runtime 或 inference engineering 一手材料。 |
| NVIDIA | Technical Blog / Megatron-LM / NeMo RL / official model card | **Accepted / Deep Dive** | Nemotron 3.5 Lightning QAD recipe 与 Megatron RL Context Parallel 进入 Accepted；activation offload、model-card memory correction 与 AI-factory capacity commitment 保留 Observe。 |
| DeepSeek | API changelog / official Hugging Face organization | Not found | 本窗口没有新 API changelog、权重或 model card；DeepSeek-V4-Pro-0813 已在上一份 scan 收录，不重复记录。 |

## Hugging Face Watch

| Sources checked | Decision | 结果 |
|---|---|---|
| Hugging Face Blog | Observe / Reject | community post “Same Cluster, 33 Points More Utilization” 有调度建模参考，但 workload/权重/结果均由作者控制，未升级；没有新的 core-team 高质量 infra 文章。 |
| Transformers | Observe | sliding-window cache、Blackwell MoE grouped-mm 与 DFlash device placement 三项 merged fix 作为 correctness bundle 保留。 |
| TRL | Observe | OpenEnv trace/log-path compatibility fix 影响 `AsyncGRPO` example，但本窗口无 tagged release 或架构级变化。 |
| Accelerate / PEFT / Kernels | Routine only | 无重大 release；文档、ARM64 Windows build 与 bookkeeping changes 不进入雷达。 |
| NVIDIA official model card | Observe | Nemotron 3.5 Lightning NVFP4 的 DGX Spark vLLM recipe 将 `gpu-memory-utilization` 从 0.91 调至 0.85，属于部署稳定性修正。 |

## RL Framework Watch

| Framework | Release / PR | 子系统 | 核心变化 | 证据 | 对 AReaL 的参考 | Decision |
|---|---|---|---|---|---|---|
| AReaL | [#1608](https://github.com/areal-project/AReaL/pull/1608) | trajectory / training | Qwen3-VL dense/MoE 使用 model-owned THD，在 packing 前合并 multimodal embeddings | default-branch commit、alignment/parity/distributed tests | 直接检查 VLM token、embedding、position 与 loss mask 对齐 | Read |
| verl | [#7434](https://github.com/verl-project/verl/pull/7434) / [#7362](https://github.com/verl-project/verl/pull/7362) | weight sync / trainer backend | vLLM resume-before-sync correctness；注册式 `fsdp_turbo` engine | merged/default-branch commit | 为 backend adapter 建立 state contract；对照 trainer registry 抽象 | **Accepted / Read** + Observe |
| slime | default branch / release | - | 本窗口未发现改变架构、性能、正确性或生产行为的重大变化 | official feed | 无新增可迁移项 | Not found |
| ROLL | default branch / release | - | 本窗口未发现重大变化 | official feed | 无新增可迁移项 | Not found |
| OpenRLHF | default branch / release | - | 本窗口未发现重大变化 | official feed | 无新增可迁移项 | Not found |
| NeMo RL | [#2279](https://github.com/NVIDIA-NeMo/RL/pull/2279) | training memory | Megatron policy 支持按模块细粒度 activation CPU offload | default-branch commit；无性能 benchmark | 评估 actor memory gain 与 rollout/PCIe overlap 竞争 | Observe |

## Adjacent Runtime Watch

| Runtime | 变化 | Decision | 工程判断 |
|---|---|---|---|
| Megatron-LM | [#5882](https://github.com/NVIDIA/Megatron-LM/pull/5882) RL Context Parallel；另有 RL group filtering 与 explicit process-group RNG fixes | **Accepted / Deep Dive** | CP 是 long-context RL 的结构性能力；group filtering 与 RNG group binding 作为后续 correctness follow-up。 |
| vLLM | [#51924](https://github.com/vllm-project/vllm/pull/51924) one-sided All2All；[#51114](https://github.com/vllm-project/vllm/pull/51114) DeepEP-v2 CPU overhead | Read / Observe | 前者完善 DeepSeek Blockwise FP8 MoE communication，后者暴露 host routing prep 的可观测开销。 |
| SGLang | [#34478](https://github.com/sgl-project/sglang/pull/34478) DSpark output logprobs；[#35207](https://github.com/sgl-project/sglang/pull/35207) ngram host/D2H overlap | **Accepted / Read** + Observe | accepted-token logprob 是 RL 正确性边界；ngram 优化收益较窄，只保留 follow-up。 |
| Molt | default branch / release | Not found | 本窗口未发现可核验的重大变化。 |

## Reading Queue 判断

- [ ] **今天只读一个：Rollplex。** 回答“它和 BiDiRL 的本质区别是什么”：Rollplex 重叠同一 synchronous iteration 内的 phase，BiDiRL 动态借用 rollout/training 两侧资源，两者解决的 bubble 不同。
- [ ] **第二优先：Megatron-LM #5882。** 重点看 CP 后 advantage、action mask、old logprob 和 loss denominator 是否保持全局语义。
- [ ] NVIDIA QAD recipe 作为第三优先工业材料；暂不把本次所有 Accepted 自动塞入现有 P0。

## 去重记录

- 新增 Accepted Source ID：`arxiv:2608.14498`、`blog:nvidia/nemotron-3-5-lightning-nvfp4-qad`、`github:NVIDIA/Megatron-LM#5882`、`arxiv:2608.14205`、`github:verl-project/verl#7434`、`github:sgl-project/sglang#34478`。
- arXiv 论文的 v1 submission 在 8 月 13–14 日，但首次进入公开 announcement batch 的时间落在本扫描窗口；记录 announcement 与 submission 两个时间，避免误判为历史补录。
- DeepSeek-V4-Pro、Megatron RL generation lag、NeMo rollout failure containment 等上一份 Accepted 不重复进入本次。
- AReaL #1608、vLLM #51924、SimpleOPD 等只保留 Observe/Read，不因为主题匹配自动升级。

## 扫描完整性

- arXiv：扫描 8 月 17 日公告批次，核验 Accepted 的 title、authors、v1 time、abstract mechanism 与 numeric claims；replacement-only entries 未当成新论文。
- Core vendors：四家均显式检查。OpenAI/Anthropic sitemap `lastmod` 与真实发布日期可能不一致，使用 research/news index 交叉判断。
- Frameworks：官方 Atom feed、PR 页面与 default-branch commit 交叉核验；vLLM/SGLang 高活跃 feed 只保留最近约 20 个 commit，是本次最主要盲区。
- Vendor evidence：NVIDIA QAD、capacity、throughput、quality recovery 与 AI-factory capacity 数字均标记为 vendor-reported/forecast，未冒充独立复现。
- 下一游标：`2026-08-18 09:31:44`。本时刻之后出现的材料留给下一次扫描。

## 下一步动作

- [x] 更新 [Scan Log](scan_log.md) 与 [Tracking README](README.md)。
- [ ] 精读 Rollplex，和 BiDiRL/AWEX/TMax 画一张 scheduler taxonomy，不急着新建 topic。
- [ ] 阅读 Megatron-LM #5882 的 CP data path；若确认改变 long-context RL 工程判断，再更新 [Long-context Training](../topics/long_context_training.md) 与 [Agentic RL](../topics/agentic_rl.md)。
- [ ] 将 verl #7434 的 backend state matrix 用于后续 AReaL sleep/resume/weight-sync review。

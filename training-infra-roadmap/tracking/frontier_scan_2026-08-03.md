# Frontier Scan, 2026-08-03

- Previous scan：[2026-07-28](frontier_scan_2026-07-28.md)
- Window：2026-07-28 16:58 ~ 2026-08-04 13:22
- Timezone：Asia/Shanghai
- Generated at：2026-08-04 13:22（合并 08-04 增量扫描）
- Report type：flexible frontier scan
- Sources scanned：arXiv cs.DC / cs.LG / cs.AI / cs.CL / cs.AR / cs.PF / cs.NI new records；OpenAI / Anthropic / NVIDIA / DeepSeek / Hugging Face official sources；AReaL / verl / slime / ROLL / OpenRLHF / NeMo RL / TRL / Transformers / Accelerate / PEFT / Kernels releases、default-branch changes 与 major PR
- Scan completeness：完整覆盖上一游标后的 arXiv 新公告、重点厂商官方源与核心 RL framework primary sources。08-04 增量扫描只保留 PNPO 与 FutureBridge-OPD 两条最贴近 RL Infra / OPD-MOPD 主线的论文信号；其余候选经评估后不进入主记录。GitHub API 触发匿名 rate limit 后，回退到官方 Atom、release page 与 commit patch 核验；未用搜索摘要代替代码证据。

## 本次核心判断

本次保留十条高质量信号，不按数量凑榜单。它们共同指向六个变化：

1. **Agent serving 的前端状态已经进入关键路径。** TokTier 显示，在高 prompt-cache hit rate 下，tokenization 可占 TTFT 的大头；“重用 prefix”还必须保证 token IDs 与完整重分词逐位一致。
2. **系统指标必须从单次模型调用上升到完整 trajectory。** Aries 的生产 trace 表明，tool sandbox、context growth 与跨组件等待会让 tokens/s 掩盖真实瓶颈；Agentic RL 的 rollout 也需要同样的 trajectory-level observability。
3. **闲置 GPU 不是静态资源。** DeltaServe 把低峰 inference headroom 转成 LoRA fine-tuning；verl、NeMo RL 同期则在推进 decoupled PPO、MOPD teacher placement 与 async correctness。训推资源边界正在从固定分区走向受 SLO 和版本约束的动态复用。
4. **生产性能差距往往来自全栈配置，而不是模型代码。** NVIDIA Exemplar Cloud 的案例把 8%-31% 差距定位到 SMMU、CPU C-state、NUMA、NCCL QP 与容器拓扑文件；这类问题不会被一次 MFU 指标自动解释。
5. **Agent capability 的提升越来越依赖 post-training、harness 与 inference runtime 联合交付。** DeepSeek-V4-Flash-0731 没有改 backbone 架构，而是通过重新 post-training、reasoning-effort 控制、Responses API/Codex 适配与 DSpark speculative decoding，把同一 Flash 模型推向更强的 coding-agent workload。
6. **Rollout 复用和 teacher guidance 都需要看完整 causal trajectory。** PNPO 处理同一批 rollout 被多轮 learner update 复用后的 policy lag；FutureBridge-OPD 则不再只看当前位置的 teacher/student disagreement，而是验证 teacher bridge 是否真的改善后续 student trajectory。

## Accepted Frontier Signals

### OpenAI: How GPT-5.6 Fuses Frontier Intelligence with Frontier Efficiency

- Signal ID：2026-08-03-001
- Source ID：blog:openai/gpt-5-6-frontier-intelligence-efficiency
- First seen：2026-08-03 10:02（Asia/Shanghai，本次扫描）
- Focus Match：P0 Focus
- 来源：OpenAI official engineering blog
- 类型：engineering blog / inference stack / agent harness / autonomous systems optimization
- 链接：https://openai.com/index/gpt-5-6-frontier-intelligence-efficiency/
- Primary-source check：title / authors / 2026-07-29 publication date / serving-cost and token-efficiency claims / harness details 已对齐官方正文
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：这是少见的 frontier lab 生产栈披露，把 global/cluster/instance load balancing、kernel、speculative decoding、workload-specific engine tuning 和 agent harness 放进同一条效率链。
- Status：NEW
- 建议动作：精读 inference 与 harness 两节；把“append-only history + deterministic tool order + deferred discovery”对照 AReaL rollout request、prefix cache 与 tool schema 管理
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [Long-context Training](../topics/long_context_training.md), [FlashAttention](../topics/flashattention.md), [Distributed Training](../topics/distributed_training.md)

OpenAI 披露其优化覆盖请求路由、实例负载均衡、数据布局、kernel、speculative decoding 与 KV workload tuning。官方称 GPT-5.6 Sol 辅助重写生产 kernel 后端到端 serving cost 降低 20%，改进 draft model 后 token-generation efficiency 提升超过 15%；这些数字来自 OpenAI，尚无外部复现。

对 Agent Infra 更重要的是 repeated region：一次任务可能触发几十次模型和工具调用，任何每请求开销都会被放大。其 Rust harness 延迟发现 tools/skills/plugins、默认限制单次工具输出，并保持 model-visible history append-only、tool order deterministic，以维持稳定 prefix 与高 prompt-cache reuse。

### NVIDIA Exemplar Cloud: Lessons for Unlocking Full Performance on AI Infrastructure

- Signal ID：2026-08-03-002
- Source ID：blog:nvidia/exemplar-cloud-performance-lessons
- First seen：2026-08-03 10:02（Asia/Shanghai，本次扫描）
- Focus Match：P0 Focus
- 来源：NVIDIA Technical Blog
- 类型：official engineering blog / cluster performance / NCCL / topology / production troubleshooting
- 链接：https://developer.nvidia.com/blog/nvidia-exemplar-cloud-lessons-for-unlocking-full-performance-on-ai-infrastructure/
- Primary-source check：title / five authors / 2026-07-30 date / four case studies / iteration and collective measurements 已对齐官方正文
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：它不是硬件营销，而是四个可操作的真实排障案例，直接覆盖超大规模训练平台最容易被忽略的 host、VM、NUMA、fabric 与 container 边界。
- Status：NEW
- 建议动作：沉淀到 NCCL / slow-step playbook；提取 `perf`、`turbostat`、`numastat`、Nsight Systems 和 `nccl-tests` 的排查顺序
- 关联主题：[NCCL](../topics/nccl.md), [Distributed Training](../topics/distributed_training.md), [Fault Tolerance](../topics/fault_tolerance.md), [MegaScale](../tech_reports/megascale.md)

官方称相同 H100、GB200 NVL72 或 GB300 NVL72 系统上，partner cluster 常比 reference architecture 慢 8%-12%。案例包括：SMMU command queue 让虚拟化 DeepSeek-V3 MoE FP8 训练慢 12%-14%；错误 C-state 与 NUMA/process binding 让 H100 Llama 3 70B 慢 12%；512 GPU 上 `NCCL_IB_QPS_PER_CONNECTION=1` 暴露 AllGather/ReduceScatter，调到 4 后 iteration 从约 1.09s 降到 0.83s。

核心教训不是照抄某个环境变量，而是验证配置是否真正穿透 host、hypervisor、container 和 launcher。节点上存在 topology file，不代表训练容器能看到它；单机 healthy，也不代表 512 GPU collective path 正确。

### TokTier: Exact Stateful Tokenization for Agentic LLM Serving

- Signal ID：2026-08-03-003
- Source ID：arxiv:2607.29678
- First seen：2026-08-03 10:02（Asia/Shanghai，本次扫描）
- Focus Match：P0 Focus
- 来源：arXiv
- 类型：paper / agentic serving / tokenization correctness / TTFT
- 链接：https://arxiv.org/abs/2607.29678
- Primary-source check：title / Zhenyu Zhang and Zhichao Cao / 2026-07-31 date / workload scale / zero-divergence and latency claims 已对齐 arXiv abs 页
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：它把 tokenizer 从无状态 CPU 前处理变成带 exactness contract、增量修复、GPU fallback 与在线 shadow verification 的 serving subsystem。
- Status：NEW
- 建议动作：进入 P0 候选；重点读 stable-boundary check、fallback 条件、session state 生命周期和与 vLLM prefix cache 的接口
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [Long-context Training](../topics/long_context_training.md), [Distributed Training](../topics/distributed_training.md)

Coding agent 每次只追加少量工具结果，却可能重传百万字符 transcript。BPE token boundary 可能跨 append point 改变，简单缓存旧 token IDs 会 silent divergence。TokTier 在 append 附近重分词，只有通过 per-request stable-boundary check 才 splice；失败时扩大窗口或回退完整 tokenization。

论文在 17 个 tokenizer family、`1.5e10` split checks、12.4 TB real-text 与 93K+ agent steps 上报告零 divergence。作者称增量修复在 100K-3M characters 下为 0.5-1.1ms；接 vLLM 后 median TTFT 降低 16%-34%、P99 降低 23%。真正值得借鉴的是“优化必须带等价性证明和线上 shadow verifier”，不只是速度数字。

### Aries: Rethinking AI Cloud Infrastructure for Agentic Serving Systems

- Signal ID：2026-08-03-004
- Source ID：arxiv:2607.29069
- First seen：2026-08-03 10:02（Asia/Shanghai，本次扫描）
- Focus Match：P0 Focus
- 来源：arXiv
- 类型：paper / agent serving / full-stack observability / sandbox
- 链接：https://arxiv.org/abs/2607.29069
- Primary-source check：title / 16-author list / 2026-07-31 date / framework scope and three findings 已对齐 arXiv abs 页
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：它给 agent-native cloud 提供了更正确的实验单位：不是 isolated request，而是跨 model、tool、sandbox 和 state 的完整 trajectory。
- Status：NEW
- 建议动作：进入 P1 候选；重点读 trajectory reconstruction、telemetry correlation、sandbox abstraction 与 commercial trace methodology
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [Rollout Latency](../playbooks/rollout_latency.md), [Long-context Training](../topics/long_context_training.md)

Aries 把 task semantics 与 execution configuration 分开，重建跨组件 trajectory，并把 system telemetry 关联到同一条执行链。作者的实验与生产 trace 得出三点：token-centric metrics 会漏掉非推理瓶颈；保留更多 context 对准确率收益递减却持续降低 capacity；sandbox 常表现为长 idle + 短 burst，而 snapshot-based state management 让频繁 suspend/resume 过贵。

对 RL Infra 的直接映射是：rollout dashboard 不能只有 generation throughput。还应看到每条 trajectory 的 model wait、tool wait、sandbox cold/resume、context bytes、policy version、retry 与 terminal reason。

### DeltaServe: Host-Agnostic Co-Serving of Inference and Fine-Tuning for LLMs

- Signal ID：2026-08-03-005
- Source ID：arxiv:2607.28848
- First seen：2026-08-03 10:02（Asia/Shanghai，本次扫描）
- Focus Match：P1 Focus
- 来源：arXiv
- 类型：paper / inference-finetuning co-serving / SLO-aware scheduling / LoRA
- 链接：https://arxiv.org/abs/2607.28848
- Primary-source check：title / eight authors / 2026-07-30 date / vLLM-SGLang-S-LoRA integration / throughput and SLO claims 已对齐 arXiv abs 页
- 影响等级：★★★★☆
- Decision：Read
- Reason：它提供了比固定训推分区更温和的资源复用方式：只在 inference headroom 足够时注入 LoRA fine-tuning，并由 CUDA-graph-aware latency model 守住服务 SLO。
- Status：NEW
- 建议动作：进入 P1 候选；确认 admission granularity、preemption cost、optimizer-state residency 和 burst prediction failure mode
- 关联主题：[Distributed Training](../topics/distributed_training.md), [Agentic RL](../topics/agentic_rl.md), [Data Parallelism](../topics/data_parallelism.md)

DeltaServe 利用 inference prefill 与 LoRA fine-tuning forward 的共享执行结构，通过很小的 hook interface 接入已有 serving engine，要求 backend 支持 multi-LoRA batching。它没有把 online service 暂停后整卡交给训练，而是在 SLO-aware scheduler 判断存在 headroom 时才 admission fine-tuning work。

作者称在 production trace 上，vLLM 集成相对 LLMStation 提升 2.9x fine-tuning throughput，且保持 100% inference SLO compliance；相对 vLLM+torchtune 提升 39%。它对 RL 的启发是“借用资源”可以细化为可抢占的 execution slots，但只有 latency model、state residency 与回退机制齐全时才成立。

### Adaptive FastOPD: Progress-Aware Rollout Horizon Expansion for Efficient OPD

- Signal ID：2026-08-03-006
- Source ID：arxiv:2607.29494
- First seen：2026-08-03 10:02（Asia/Shanghai，本次扫描）
- Focus Match：P0 Focus
- 来源：arXiv
- 类型：paper / on-policy distillation / rollout tail / adaptive horizon
- 链接：https://arxiv.org/abs/2607.29494
- Primary-source check：title / five authors / 2026-07-31 date / mechanism / 49.1%-71.2% training-time claim 已对齐 arXiv abs 页
- 影响等级：★★★★☆
- Decision：Deep Dive
- Reason：它直接回答当前 MOPD/OPD 的系统问题：少数长 response 拖住 online rollout 时，如何让 horizon 随真实学习进展增长，而不是固定给满 15K。
- Status：NEW
- 建议动作：优先补入 [MOPD](../topics/mopd.md) 后续精读队列；核对四个 teacher-student signals、plateau detector 与 horizon-utilization guard
- 关联主题：[MOPD](../topics/mopd.md), [Agentic RL](../topics/agentic_rl.md), [Rollout Latency](../playbooks/rollout_latency.md), [Long-context Training](../topics/long_context_training.md)

Adaptive FastOPD 只在两个条件同时满足时扩大 rollout horizon：当前 boundary 附近的 teacher-student learning signals 已相对进入该 horizon 时的平台期，并且当前 horizon 被足够利用。第二个条件避免少数异常长样本触发整个训练阶段升档。

作者在两个 teacher-student pair 上报告，相对 OPD 15K 减少 49.1%-71.2% training time，同时取得最高平均性能。它不是异步调度器，却能从 workload source 端收紧长尾；可与 MOPD 的多教师选择、BiDiRL 的资源调度形成三层互补。

### PNPO: Reusing Rollouts under Policy Lag

- Signal ID：2026-08-03-009
- Source ID：arxiv:2608.01418
- First seen：2026-08-04 13:22（Asia/Shanghai，增量扫描）
- Focus Match：P0 Focus
- 来源：arXiv
- 类型：paper / RL post-training / rollout reuse / policy lag
- 链接：https://arxiv.org/abs/2608.01418
- Primary-source check：title / 12-author arXiv metadata / 2026-08-02 submission / one-vs-four epoch setup and Avg@32 claims 已对齐 arXiv primary metadata
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：rollout 通常是 RL 最贵的阶段；如果同一批 trajectory 能支持多轮 learner update，就能摊薄生成成本，但必须处理 behavior policy 到 current policy 的 causal-prefix drift。
- Status：NEW
- 建议动作：对照 AReaL decoupled loss、MinPRO、GSPO 与 importance-weight clipping；先验证 optimization correctness，再测 wall-clock goodput
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [MOPD](../topics/mopd.md), [Rollout Latency](../playbooks/rollout_latency.md)

严格的 off-policy correction 需要累计“到达当前 prefix 的概率”和“当前 action 的概率”，但 cumulative product 会造成极大的动态范围。PNPO 改用 causal prefix 上 likelihood ratio 的几何均值，在保留 prefix dependence 的同时压缩 log-weight scale。

作者报告：每批 rollout 只做一个 epoch 时，PNPO 并不稳定优于 GSPO；做四个 epoch、进入更强 off-policy 区间后，三项 benchmark 独立 peak 的非加权均值为 50.24，比 GSPO 高 3.00 个百分点。固定 2400 次 update 时，四 epoch PNPO 用 150 批 rollout 达到 49.66，与一 epoch 使用 600 批得到的 49.56 接近。它说明 rollout reuse 值得验证，但不能直接把“少 4 倍 rollout batch”写成“训练快 4 倍”。

### FutureBridge-OPD: Look Ahead Before You Distill

- Signal ID：2026-08-03-010
- Source ID：arxiv:2608.01953
- First seen：2026-08-04 13:22（Asia/Shanghai，增量扫描）
- Focus Match：P0 Focus
- 来源：arXiv
- 类型：paper / agentic OPD / teacher intervention / trajectory validation
- 链接：https://arxiv.org/abs/2608.01953
- Primary-source check：title / Chishui Chen and 13 coauthors / 2026-08-03 submission / three environments / teacher-student pair / gain claims and code link 已对齐 arXiv abs 页
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：它补上 OPD/MOPD 的关键缺口：teacher 与 student 在当前位置分歧大，不代表 teacher token 一定能让后续 student trajectory 变好。
- Status：NEW
- 建议动作：补入 [MOPD](../topics/mopd.md) 后续精读；重点核算 teacher bridge、student continuation 与多教师选择叠加后的 rollout 成本
- 关联主题：[MOPD](../topics/mopd.md), [Agentic RL](../topics/agentic_rl.md), [Long-context Training](../topics/long_context_training.md)

FutureBridge-OPD 在 high-disagreement state 执行一小段 teacher bridge，再切回 student continuation；只有当这段 bridge 提高后续正向 distillation signal 的密度时，才把指导视为有价值。判断单位从“一个 token 的 teacher/student disagreement”升级为“干预后的 future trajectory”。

论文在 ALFWorld、WebShop、ScienceWorld 上，以 Qwen3-32B teacher 和 Qwen3-1.7B student 为主设置，相对 vanilla OPD 和 TCOD 平均提升 16.6 与 7.6 分，并提供公开代码。系统侧真正要问的是：额外 teacher bridge 和 student continuation 增加了多少 rollout，能否由 MOPD teacher pool 与异步 scheduler 吸收。

### NVIDIA: Co-Designing Attention for Long-Context Inference

- Signal ID：2026-08-03-007
- Source ID：blog:nvidia/attention-codesign-long-context-inference
- First seen：2026-08-03 10:02（Asia/Shanghai，本次扫描）
- Focus Match：P0 Focus
- 来源：NVIDIA Technical Blog
- 类型：official engineering blog / attention architecture / GPU roofline / inference parallelism
- 链接：https://developer.nvidia.com/blog/co-designing-ai-model-attention-for-fast-interactive-long-context-inference/
- Primary-source check：title / five authors / 2026-07-31 date / FP8 measurements / GEMM-shape and parallelism guidance 已对齐官方正文
- 影响等级：★★★★☆
- Decision：Read
- Reason：它把 GQA group size、head dimension、sequence length 与 TP/KV duplication 放到同一套 GPU execution model 中，适合补齐训练视角之外的 long-context inference 判断。
- Status：NEW
- 建议动作：进入 P1 候选；重点读 prefill/decode roofline、`TP <= KV heads` 边界、ADP/KVP 与 hybrid parallelism
- 关联主题：[FlashAttention](../topics/flashattention.md), [Tensor Parallelism](../topics/tensor_parallelism.md), [Long-context Training](../topics/long_context_training.md)

文章指出 prefill 通常是 compute-bound，decode 通常被 KV read 的 HBM bandwidth 限制；prefix-cached agent turn 可能是短 ISL + 长 KVSL，因此其新一轮 prefill 会更像 decode。DeepSeek-R1 示例中，context 从 4K 增至 128K 时 attention 占 prefill 时间从 18% 升至 85%。

最有工程价值的约束是 `TP degree` 不应超过 KV head 数，否则 KV 会复制，显存与 bandwidth 收益被抵消。KV head 很少的模型应考虑 Attention Data Parallelism、KV Parallelism 或 Wide EP/Helix 一类混合方案，而不是机械扩大 TP。

### DeepSeek-V4-Flash-0731: Agentic Post-Training and DSpark Delivery

- Signal ID：2026-08-03-008
- Source ID：model:deepseek-ai/DeepSeek-V4-Flash-0731
- First seen：2026-08-03 16:34（Asia/Shanghai，本次扫描补漏）
- Focus Match：P0 Focus
- 来源：DeepSeek official API changelog + Hugging Face model card / weights
- 类型：model release / agentic post-training / speculative decoding / inference integration
- 链接：https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731
- Primary-source check：2026-07-31 release date / model identity / unchanged architecture / re-post-training statement / benchmark table / DSpark deployment flags / API scope 已对齐官方 changelog 与 model card
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：它证明 agent model 的版本升级不一定来自更大 backbone；同一 V4-Flash 架构通过 post-training、可控 reasoning effort、agent harness 与 speculative decoding 联合交付，可以大幅改变 coding-agent 的能力与 serving 形态。
- Status：NEW
- 建议动作：先读 model card 的 Introduction、Chat Template 与 vLLM/SGLang recipes；后续等待 DeepSeek Harness 和 post-training 细节公开，再判断提升主要来自 model、harness 还是 evaluation allocation
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [Long-context Training](../topics/long_context_training.md), [MOPD](../topics/mopd.md), [Distributed Training](../topics/distributed_training.md)

官方明确说明，0731 与 V4-Flash Preview 保持相同模型架构和规模，本次只做了重新 post-training；API 进入 public beta，APP/WEB 与 V4-Pro API 未同步升级。模型新增 `low`、`high`、`max` 三档 `reasoning_effort`，原生支持 Responses API 并专门适配 Codex。开放权重采用 MIT License。

部署侧不是只换一个 checkpoint：该版本附带 DSpark speculative decoding module。vLLM 可用 `method=dspark`、7 个 speculative tokens 启用；SGLang 直接从同一 checkpoint 读取 target/draft weights。官方 benchmark 中 Terminal Bench 2.1 从 Preview 的 61.8 升至 82.7，DeepSWE 从 7.3 升至 54.4，但这些结果使用尚未发布的 DeepSeek Harness minimal mode，且 DSBench 两项为内部集，因此当前只能确认“官方报告显著提升”，不能视为独立可复现结论。官方也没有披露此次 re-post-training 的数据构成、RL 算法或训练算力，禁止从结果反推具体训练方案。

## Observed / Rejected Candidates

### Observed

| 材料 | Source ID | Decision | 原因 |
|---|---|---|---|
| Mixture-of-Translators | arxiv:2607.28979 | Observe | 跨 Qwen2.5/GPT-2/OPT 翻译 KV cache，并在长上下文 case 中保留 direct-context 96.3% 质量；当前更偏 representation transfer，尚缺端到端 latency、translator cost 与生产 backend 集成 |
| Zero-Mem | arxiv:2607.29377 | Observe / Read later | memory operation 不调用 LLM、保留原始 trace，作者称相对最快 baseline 降低 57.6% memory-operation time；代码承诺 peer review 后开放，先等待实现 |
| ResKV | arxiv:2607.29591 | Observe | 用 main cache + residual cache 恢复被 eviction token 对 softmax numerator/denominator 的贡献；摘要未给可审计的速度数字与代码入口 |
| SLIM | arxiv:2607.29575 | Observe | decode attention saturation 的 semi-analytical model 与 Batching Configuration Advisor 有价值；当前验证集中于 OPT，需看跨现代 GQA/MoE backend 泛化 |
| WitCert | arxiv:2607.28699 | Observe | 给 KV quantization 加 runtime risk certificate 与 SGLang gate，且部分 theorem 用 Lean 4 验证；单篇需核对 patch、certificate overhead 与 adaptive-query 边界 |
| Data Turnstile | arxiv:2607.29250 | Observe | 公开 function-calling synthetic-data framework 与 100K+ interactions，但当前主要改变数据生产，不直接改变训练 runtime |
| SciDisco | arxiv:2607.28990 | Observe | process-verifiable scientific environment + turn-level credit 适合 Agentic RL environment 线；当前是特定 scientific discovery workload |
| ThinkReset | arxiv:2607.28642 | Historical / backfill candidate | 原始日期为 2026-05-26；bounded-context reusable interface 与 CompactionRL 互补，但不计本窗口新信号 |
| Topology-Aware Data Movement for Disaggregated GPU Inference | arxiv:2607.28633 | Observe / historical | 原始日期为 2026-04-19，且主要是 analytical projection；作者明确缺少真实 heterogeneous/CXL cluster 完整评估，不接受其 3-18x projection 为实测结果 |

### Rejected / No Core Signal

| 材料 | Decision | 原因 |
|---|---|---|
| OpenAI ARC-AGI-3 two-settings post | Observe only | harness settings 对 benchmark 成绩影响很大，但不是可迁移的训练/serving infra 机制；保留为“evaluation harness can dominate result”的旁证 |
| Hugging Face community GPU-management post | Reject for current scan | 话题相关，但缺少足以改变调度判断的可复现实验与框架实现；不因标题贴合就接受 |
| OlmoEarth infrastructure / LFM2.5 encoder posts | Reject for current focus | 分别偏 geospatial platform 与 CPU encoder deployment，不进入当前 Training/Agentic RL/large-scale inference 主线 |
| Generic model, application, dataset and benchmark papers | Reject | 未提供会改变训练、rollout、serving、kernel、distributed runtime 或生产运维判断的机制 |

## OpenAI / Anthropic / NVIDIA / DeepSeek Watch

| Vendor | Sources checked | Decision | 结果 |
|---|---|---|---|
| OpenAI | official news / engineering / research / API changelog | Accepted + Observed | GPT-5.6 efficiency engineering post 进入 Accepted；price-performance 产品页只作数字交叉验证；ARC-AGI-3 settings 进入 Observed，不把产品发布本身当 infra 信号。 |
| Anthropic | official newsroom / Claude Platform release notes | Not found / no new core signal | 官方 release notes 最新可见条目仍早于本窗口；没有可验证的新 training、RL 或 agent runtime 技术正文。 |
| NVIDIA | Technical Blog / NeMo RL official repository | Accepted + Framework follow-up | Exemplar Cloud 与 attention co-design 两篇进入 Accepted；NeMo RL 的 async KV correctness、MOPD teacher placement、router replay 和 rollout serialization 进入 framework watch。 |
| DeepSeek | official API changelog / Hugging Face model card and weights | Accepted | 7 月 31 日 DeepSeek-V4-Flash-0731 进入 Accepted：相同架构重新 post-training，开放权重，加入 reasoning-effort 控制、Responses API/Codex 适配与 DSpark speculative decoding；尚未公开 Harness 与 post-training 细节。 |

## Hugging Face Watch

| Source | Source type | Decision | 结果 |
|---|---|---|---|
| Hugging Face Blog | official feed, vendor/community authored posts | Observed / Rejected | 窗口内三篇为 GPU management、OlmoEarth 与 LFM2.5 Encoders；均未达到本次核心接受门槛。 |
| TRL | official release/main commits | Framework follow-up | 最新正式 release 仍为 07-28 的 v1.9.2；main 正在重构 DistillationTrainer、加入 chunked JSD，并修正 FSDP2-vLLM device hardcode，尚未形成稳定 release/benchmark。 |
| Transformers | official release/main commits | Observe | 最新 release 早于窗口；main 有 Kimi export、compressed tensors 与 tokenizer fixes，未形成当前主线的独立系统信号。 |
| Accelerate / PEFT / Kernels | official sources | Not found / no material signal | 未发现窗口内会改变 distributed training、rollout correctness 或 kernel strategy 的高质量 release。 |

## RL Framework Watch

本窗口没有新的核心 framework release。以下只保留会改变架构、correctness、性能或长期运维行为的 default-branch change；Framework Watch 不额外计入上方十条 Accepted Frontier Signals。

| Framework | Change | Subsystem | Evidence / state | Decision | 对 AReaL 的参考 |
|---|---|---|---|---|---|
| AReaL | [PR #1569](https://github.com/areal-project/AReaL/pull/1569)：output token 绑定实际 serving version | rollout / staleness correctness | merged；旧路径 request 前后读取两次 version，weight update race 会把旧 policy token 标成新 version；新增 race regression test | Accepted / Deep Dive | 已在 AReaL；应继续验证 multi-turn 每段 version、abort/resume 与 mixed-version loss mask |
| AReaL | [PR #1544](https://github.com/areal-project/AReaL/pull/1544)：Megatron deterministic mode 在 model build 前生效 | training / reproducibility | merged；部分 Megatron-Core / Transformer Engine 模块会在 construction 时缓存 determinism 配置，旧路径 build 后再设置会静默留下 nondeterministic kernel | Accepted / Deep Dive | 已在 AReaL；复现实验必须核对 initialization order、环境变量与 backend 选择，而不只是最终 config 值 |
| verl | [PR #7188](https://github.com/verl-project/verl/pull/7188)：`separate_async` 支持 Decoupled PPO | training / rollout correction | merged；在一个 `parameter_sync_step` 周期内用 CPU save/restore 保持稳定 `pi_old`，并按 version 重算 old logprob | Accepted / Read | 对比 AReaL decoupled loss：CPU snapshot 成本、版本粒度、worker failure 后 `pi_old` 恢复语义 |
| verl | [PR #7207](https://github.com/verl-project/verl/pull/7207)：partial rollout resume 不再重复发满 response budget | rollout / tail latency / correctness | merged；旧路径每次 resume 近似重新授予完整 budget，超出部分最终被截掉；新增多次 abort/resume E2E | Accepted / Read | AReaL interrupted trajectory 也应按 remaining budget 授权，记录 wasted generated tokens |
| verl | [PR #7225](https://github.com/verl-project/verl/pull/7225)：distillation loss 对 micro-batch size 保持不变 | distillation / correctness | merged；修复前固定 global batch、micro-batch=1 时 loss 可高 8x；修复后不同 gradient accumulation 划分一致 | Accepted / Deep Dive | MOPD/OPD 必须把 micro-batch invariance 设为 regression test，防止性能配置悄悄改变优化目标 |
| NeMo RL | [PR #3152](https://github.com/NVIDIA-NeMo/RL/pull/3152)：async weight update 后始终尊重 KV-cache recompute 开关 | weight sync / KV correctness | merged；配置为 true 时无论是否 in-flight update 都 invalidates backend KV；官方注释指出旧 KV 可累积 policy KL error | Accepted / Deep Dive | AReaL 应把 weight version 与 KV version 绑定，cache invalidation 失败不能只打印 warning 后继续 |
| NeMo RL | [PR #3292](https://github.com/NVIDIA-NeMo/RL/pull/3292)：移除 R3/NeMo-Gym routed-experts JSON bottleneck | trajectory data path / router replay | merged；长上下文 nested JSON 每个 serialize/parse hop 约消耗 1s 单线程 CPU，改成 versioned compact binary codec | Accepted / Read | MoE rollout metadata 应使用有版本的 typed/binary schema，并单独统计 encode/network/decode |
| NeMo RL | [PR #3340](https://github.com/NVIDIA-NeMo/RL/pull/3340)：Gym 启动前预留 MOPD teacher clusters | scheduler / teacher placement | merged；teacher placement group 提前、逐个 claim，并约束在 NVLink domain，避免 Gym 抢占资源后 teacher 初始化失败 | Accepted / Deep Dive | MOPD scheduler 必须先规划 student/inference/teacher 的 topology，再启动环境；失败时整体释放 reservation |
| NeMo RL | [PR #3378](https://github.com/NVIDIA-NeMo/RL/pull/3378)：router replay 穿过 TransferQueue | rollout / MoE routing replay | merged；携带 routed experts、weight version 与 validation，缺失 route 时显式 fallback 或报错 | Accepted / Read | AReaL MoE async rollout 应把 route metadata 纳入 trajectory contract，不能靠训练端重新路由假装同一 policy |
| TRL | DistillationTrainer refactor + chunked JSD main commits | distillation / memory | default branch in progress；尚无 release 与 E2E benchmark | Observe | 先关注 chunked loss 是否保持 token normalization 与 teacher/student mask 等价，不追逐未稳定 commit |
| verl | DeepSeek V4 FP8/FP4 refit、contiguous CP layout、delta-sharded mappings | training / weight sync / model support | merged；模型特定改动较多，尚无独立性能报告 | Observe | 只在真实 Qwen/DeepSeek recipe 需要时跟进，不把“支持新模型”自动升级为系统创新 |
| slime / ROLL / OpenRLHF | no material change after cursor | - | release pages与 official Atom 均无窗口内架构级更新 | Not found | 不用 routine commit 或文档更新补数量 |

### Framework Follow-up

- AReaL #1569 与 NeMo RL #3152 指向同一条 correctness invariant：trajectory token、KV cache、logprob 与 weight 必须能回答“由哪个 policy version 生成”。异步系统没有 versioned state contract，就无法可靠讨论 staleness。
- verl #7207 说明 partial rollout 的性能账不能只看最终保留 token。被生成后又截掉的 token 同样占用了 decode GPU；建议给 AReaL 增加 `generated_tokens_total / committed_tokens / discarded_tokens` 三项计数。
- NeMo RL #3340 是当前 [MOPD](../topics/mopd.md) 最值得立即看的实现补充：多教师首先是 topology-aware resource reservation 问题，其次才是 teacher routing 算法。
- NeMo RL #3292 说明长上下文 MoE rollout 中，HTTP/JSON metadata 可能比 GPU kernel 更先成为瓶颈。trajectory schema 设计应进入 performance review，而不是只做功能 review。
- AReaL #1544 与 verl #7225 共同说明：配置和 batch 划分属于 optimization semantics。训练任务“能跑完”不代表 deterministic setting 真正生效，也不代表不同 micro-batch 下优化的是同一个 loss。

## 本次阅读决策

### 建议今天先读

1. **Adaptive FastOPD**：最贴当前 OPD -> MOPD 专题，只读摘要、方法图和 horizon expansion 条件，先理解它如何从源头减少长尾。
2. **AReaL #1569 + NeMo RL #3152**：两份短 patch，一起建立异步 RL 的 policy/KV/token version correctness 判断。
3. **NVIDIA Exemplar Cloud**：跳过介绍，直接读四个 case study，把可复用命令和 profiler signal 摘出来。
4. **DeepSeek-V4-Flash-0731 model card**：重点区分 backbone、post-training、harness 和 speculative runtime 各自贡献；不要只看榜单涨幅。

### 后续按问题选择

- Agent serving front-end / TTFT：TokTier。
- Trajectory observability 与 sandbox：Aries。
- 训推资源复用：DeltaServe。
- Long-context inference 架构：NVIDIA attention co-design。
- Agent harness 与 prompt cache：OpenAI GPT-5.6 efficiency engineering post。
- Rollout 复用与 policy lag：PNPO。
- OPD/MOPD 的 teacher intervention：FutureBridge-OPD。

本次不自动修改 reading queue。只有用户实际开始阅读，才把对应材料提升到 P0/P1，避免队列因扫描自动膨胀。

## 下一次扫描起点

- Next cursor：2026-08-04 13:22
- 下次继续扫描：
  - 新增 arXiv v1、technical report 与高质量 engineering blog；
  - OpenAI / Anthropic / NVIDIA / DeepSeek / Hugging Face 官方更新；
  - AReaL / verl / slime / ROLL / OpenRLHF / NeMo RL / TRL release、major merge 与 rollback；
  - TokTier 是否公开代码，以及 exact incremental tokenizer 是否能接入 vLLM/SGLang/rollout service；
  - Adaptive FastOPD 与 MOPD 的组合边界；
  - PNPO 的 rollout reuse 是否能在 AReaL 上转化为 wall-clock goodput；
  - FutureBridge-OPD 的 future validation 是否会放大 teacher rollout 成本；
  - AReaL version attribution 是否覆盖 partial rollout、multi-turn 与 recovery；
  - NeMo RL MOPD teacher reservation、router replay 与 binary metadata codec 的 E2E 证据；
  - NVIDIA Exemplar Cloud 的 NCCL / NUMA / topology diagnostics 是否值得沉淀为独立 playbook。

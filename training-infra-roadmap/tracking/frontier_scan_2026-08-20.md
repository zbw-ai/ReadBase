# Frontier Scan, 2026-08-20

- Previous scan：[2026-08-18](frontier_scan_2026-08-18.md)
- Window：2026-08-18 09:31:44 ~ 2026-08-20 10:21:56
- Timezone：Asia/Shanghai
- Generated at：2026-08-20 10:21:56
- Report type：flexible frontier scan
- Sources scanned：arXiv cs.AI / cs.LG / cs.DC / cs.PF / cs.AR / stat.ML；OpenAI / Anthropic / NVIDIA / DeepSeek official sources；Hugging Face Blog / TRL / Transformers / Accelerate / PEFT / Kernels；AReaL / verl / slime / ROLL / OpenRLHF / NeMo RL；Megatron-LM / vLLM / SGLang official default-branch changes
- Scan completeness：arXiv API 返回本窗口六个分类并集 439 条记录，Accepted 的 title / authors / date / abstract claims 已逐条核对。GitHub 使用 official Atom、merged/default-branch commit 和 patch 交叉核验；高活跃仓库 feed 仍可能截断普通 commit。扫描截止时刻冻结在检索开始前，晚于该时刻的变更留给下一次。

## 本次核心判断

这次最强的信号不是某个新算法，而是 **harness、trajectory correctness 与异步容错正在成为 Agentic RL 的正式系统边界**。Agent Lightning v1.0 给出最小可复现框架，LEGO-RL 处理原生 coding harness 的 token 对齐和 sandbox 可靠性，NeMo RL 则把 staleness、fleet health 与 dropped rollout recovery 做进 runtime。与此同时，MOPD 已从概念进入可复现诊断和异步 trainer 实现。

## Accepted Frontier Signals

### Agent Lightning v1.0：把 Harnessed Agentic RL 定义成独立系统范式

- Signal ID：2026-08-20-001
- Source ID：arxiv:2608.17528
- First seen：2026-08-20 10:21:56
- 发布时间：2026-08-18 16:50:13，Asia/Shanghai
- Scan window：2026-08-18 09:31:44 ~ 2026-08-20 10:21:56
- Focus Match：P0 Focus
- 来源：arXiv primary page / paper
- 类型：paper / framework / harnessed agentic RL
- 链接：https://arxiv.org/abs/2608.17528
- Primary-source check：title、10 位作者、v1 date、约 3,500 行实现、AReaL 2.0 等采用关系、6K examples 与 SWE-bench Verified 数字已对齐 arXiv metadata/abstract
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：它明确把 agent harness 从“训练外部的应用代码”提升为 post-training data plane owner，并系统列出 retokenization、sample merging、advantage、loss normalization 和 backend scheduling 五类 correctness 边界。
- Status：NEW
- 建议动作：先读 framework contract 与 coding-agent pipeline，再对照 AReaL trajectory schema 和 endpoint proxy
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [Rollout Latency](../playbooks/rollout_latency.md), [Long-context Training](../topics/long_context_training.md)

传统 RL runtime 自己拥有 environment loop；harnessed RL 则由部署时 harness 管理工具、上下文和控制流，trainer 只观察一组 LLM request-response。这个边界变化解释了为什么“能接一个 OpenAI-compatible endpoint”并不等于能正确训练：多个请求如何合并成 sample、重新 tokenize 后 token 是否一致、每个请求的 advantage 如何回填，都需要显式契约。

论文报告用 6K examples 将 Qwen3.5-9B 在 SWE-bench Verified 从 41.8% 提升到 56.4%。该数字是论文结果，当前仓库未复现；更重要的信号是完整 workflow 和 training scripts 已发布，可作为 AReaL harness adapter 的对照实现。

### LEGO-RL：保留原生 Coding Harness，同时保证 Policy-Gradient 可训练性

- Signal ID：2026-08-20-002
- Source ID：arxiv:2608.17393
- First seen：2026-08-20 10:21:56
- 发布时间：2026-08-18 13:34:35，Asia/Shanghai
- Scan window：2026-08-18 09:31:44 ~ 2026-08-20 10:21:56
- Focus Match：P0 Focus
- 来源：arXiv primary page / project page
- 类型：paper / coding-agent RL / sandbox and trajectory correctness
- 链接：https://arxiv.org/abs/2608.17393
- Primary-source check：title、12 位作者、date、in-process proxy、trainer-side logprob recomputation、sandbox defenses、three-harness evaluation 与 probability-correlation claim 已对齐 arXiv metadata/abstract
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：它没有要求 OpenHands/Claude Code/OpenCode 改写内部 control flow，而是在 harness 边界捕获 raw generation stream，并把 compaction/re-serialization 后的 token 对齐交给 trainer 复算。
- Status：NEW
- 建议动作：重点看 raw stream capture、token alignment、reward-hacking defense 与 sandbox image cache，不先看榜单
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [Long-context Training](../topics/long_context_training.md), [Rollout Latency](../playbooks/rollout_latency.md)

LEGO-RL 的三层设计分别解决 faithful optimization、reliable execution 和 observability。最有迁移价值的是 in-process LLM proxy：harness 可以压缩、重序列化上下文，但 trainer 仍基于捕获的原始 generation stream 重算 logprob，避免应用层文本变换悄悄改变训练 token。

论文在 Qwen3.5-35B-A3B + GSPO 上跨三个 coding harness 报告提升，并称 rollout-training probability correlation 保持在 0.99 以上。这些是作者结果；对 AReaL 更重要的问题是相同一致性检查能否落到 trajectory ledger 和 failure taxonomy。

### NeMo RL SingleController：从“异步可跑”走向 Fleet-Level 可恢复

- Signal ID：2026-08-20-003
- Source ID：github:NVIDIA-NeMo/RL#3582+#3590+#3665
- First seen：2026-08-20 10:21:56
- Scan window：2026-08-18 09:31:44 ~ 2026-08-20 10:21:56
- Focus Match：P0 Focus
- 来源：NeMo RL merged/default-branch patches
- 类型：framework implementation / async scheduler / rollout recovery
- 链接：https://github.com/NVIDIA-NeMo/RL/pull/3582
- Primary-source check：AReaL-style admission sampler、ready-first mixed-version selection、IS guard、generation fleet health、shard quarantine、dropped rollout shrink/replace、batch floor 与 checkpoint tests 已对齐三个 default-branch patches
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：这组变更同时覆盖 sample freshness、慢/坏 shard 隔离和 prompt replacement，说明大规模异步 RL 的正确抽象不是一个 staleness 参数，而是一套 admission + health + recovery protocol。
- Status：NEW
- 建议动作：将 NeMo sampler/recovery state machine 与 AReaL manager 的 admission、retry、replacement 和 checkpoint 语义逐项对照
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [Fault Tolerance](../topics/fault_tolerance.md), [Checkpointing](../topics/checkpointing.md)

`ArealAdmissionSampler` 把 dispatch 限制在 trainer version 前 `eta` 个 batch，并允许 ready-first mixed-version selection；配置层强制要求 importance-sampling correction，避免“异步”只放宽新鲜度却不修正行为策略。generation fleet health 再对 shard 做 probe、quarantine、least-outstanding routing；dropped rollout 则允许 shrink 或用 spare prompt replace，并以 `min_step_batch_fraction` 保住训练 batch 下限。

对当前 AReaL 项目最值得复用的不是类名，而是三个独立状态机：sample 是否可训练、generation shard 是否可接流量、失败 prompt 是否替换。把三者揉成 retry count 会让恢复策略和算法语义互相污染。

### TRL AsyncDistillationTrainer：OPD/MOPD 获得异步 Rollout-Teacher-Training Pipeline

- Signal ID：2026-08-20-004
- Source ID：github:huggingface/trl#6705
- First seen：2026-08-20 10:21:56
- Scan window：2026-08-18 09:31:44 ~ 2026-08-20 10:21:56
- Focus Match：P0 Focus
- 来源：Hugging Face TRL merged/default-branch implementation
- 类型：framework implementation / asynchronous distillation / MOPD
- 链接：https://github.com/huggingface/trl/pull/6705
- Primary-source check：background rollout worker、student vLLM server、HTTP teacher scoring、FSDP2 constraint、generalized JS loss、multi-teacher `teacher_id` routing、examples 与 tests 已对齐官方 patch/docs
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：它把 generation、teacher forward 和 student update 从串行同进程拆开，并提供可运行 multi-teacher routing；MOPD 因而第一次进入 Hugging Face 的通用 post-training trainer 路线。
- Status：NEW
- 建议动作：对照 [MOPD](../topics/mopd.md) 检查 weight transfer、staleness、teacher routing failure 与 loss denominator
- 关联主题：[MOPD](../topics/mopd.md), [Agentic RL](../topics/agentic_rl.md), [Distributed Training](../topics/distributed_training.md)

该 trainer 让后台 worker 从 student vLLM server 生成 on-policy completion，再把完整序列发送给独立 teacher endpoint 做 teacher-forced scoring，training 可与 generation/teacher scoring 并发。多 teacher 模式由每条样本的 `teacher_id` 显式路由，缺失映射直接报错，不静默落到错误 teacher。

当前限制同样重要：distributed training 只支持 FSDP2，vLLM 与 Transformers 还存在依赖约束冲突。这说明它是值得精读的工程入口，但尚不是可直接替代 AReaL/verl 的生产方案。

### Open-MOPD：把多 Teacher 能力失衡定位为 Token Budget 分配问题

- Signal ID：2026-08-20-005
- Source ID：arxiv:2608.19098
- First seen：2026-08-20 10:21:56
- 发布时间：2026-08-20 00:50:39，Asia/Shanghai
- Scan window：2026-08-18 09:31:44 ~ 2026-08-20 10:21:56
- Focus Match：P0 Focus
- 来源：arXiv primary page / project page
- 类型：paper / multi-teacher OPD / optimization diagnostics
- 链接：https://arxiv.org/abs/2608.19098
- Primary-source check：title、10 位作者、date、oracle routing setup、three imbalance factors、35.6%/83.4% headroom recovery 与 open-source claim 已对齐 arXiv metadata/abstract
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：它直接回答当前 MOPD 专题最关键的疑问：能力整合失败未必是 gradient conflict，而可能是长短序列、收敛速度和异步 reward staleness 共同扭曲 token-level optimization budget。
- Status：NEW
- 建议动作：精读 token-share balancing、gap-aware budget 和 student reward refresh，再更新 [MOPD](../topics/mopd.md)
- 关联主题：[MOPD](../topics/mopd.md), [Agentic RL](../topics/agentic_rl.md), [Long-context Training](../topics/long_context_training.md)

Open-MOPD 在 SmolLM3-3B-Base 上使用 oracle routing，把 teacher routing ambiguity 从实验中拿掉。作者报告标准 M-OPD 只恢复 domain-routed oracle ensemble 可用 headroom 的 35.6%，并将问题拆为 sequence-length disparity、不同 capability 的 convergence drift 和异步更新造成的 reward staleness。

三项修正分别对应 token share、动态 capability budget 和 reward refresh，论文报告 headroom recovery 提升到 83.4%。数字仍需复现，但这套诊断框架可以直接转成训练监控：每 teacher 的 token share、gap、reward age 与 capability gain 不应只汇总成总 loss。

### Megatron RL：修正 Multi-Turn Packing 下的 Logprob 与 Generation Mask 对齐

- Signal ID：2026-08-20-006
- Source ID：github:NVIDIA/Megatron-LM#5887
- First seen：2026-08-20 10:21:56
- Scan window：2026-08-18 09:31:44 ~ 2026-08-20 10:21:56
- Focus Match：P0 Focus
- 来源：Megatron-LM merged/default-branch patch
- 类型：framework implementation / RL data path / sequence packing correctness
- 链接：https://github.com/NVIDIA/Megatron-LM/pull/5887
- Primary-source check：multi-region generation mask、per-turn logprob concatenation/scatter、float32 preservation、zero-turn placeholders、packed/unpacked tests 与 multi-turn fallback 已对齐官方 patch
- 影响等级：★★★★☆
- Decision：Read
- Reason：多轮 trajectory 的 generated tokens 不是单个连续区间；若仍按 first-generation offset 连续写入 old logprob，importance ratio 和 loss mask 会在 observation gap 后整体错位。
- Status：NEW
- 建议动作：把该 PR 的 multi-region mask 测试移植成 AReaL trajectory packing regression case
- 关联主题：[Long-context Training](../topics/long_context_training.md), [Agentic RL](../topics/agentic_rl.md), [Distributed Training](../topics/distributed_training.md)

这项变更虽然标题是 “Share prefill in sequence packing”，但主要工程价值是 correctness：每一轮生成区间分别形成 mask，per-turn logprobs 按实际 generated-token position scatter 到 packed tensor；无法安全折叠的 trajectory 回退到 per-turn rows。它还避免把 float32 wire logprobs 悄悄降为 bf16。

对长上下文 Agentic RL，这比一个新的 packing heuristic 更重要。只要 trajectory 中间插入 observation/tool result，连续区间假设就会失效，而训练通常不会立即 crash，只会静默优化错误 token。

### AReaL：Qwen3-VL Dense/MoE 接入 Native AWEX Colocation

- Signal ID：2026-08-20-007
- Source ID：github:areal-project/AReaL#1605
- First seen：2026-08-20 10:21:56
- Scan window：2026-08-18 09:31:44 ~ 2026-08-20 10:21:56
- Focus Match：P0 Focus
- 来源：AReaL merged/default-branch patch
- 类型：framework implementation / weight sync / multimodal colocation
- 链接：https://github.com/areal-project/AReaL/pull/1605
- Primary-source check：full multimodal HF config exchange、vision/text sharding contract、Dense/MoE registration、router dtype handling 与 colocate contract tests 已对齐官方 patch
- 影响等级：★★★★☆
- Decision：Read
- Reason：VLM colocation 不能把 `text_config` 当成完整 model contract；vision tower、nested config、router dtype 和 inference TP layout 都必须参与 train-to-inference metadata exchange。
- Status：NEW
- 建议动作：阅读 AWEX reader/writer contract tests，并检查 Qwen3.5-VL 项目是否仍有手写 metadata 假设
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [Distributed Training](../topics/distributed_training.md), [Long-context Training](../topics/long_context_training.md)

实现将完整 composite Hugging Face config 交给 native AWEX 解析，而不是只传一个扁平化 text config；Dense 和 MoE 路径均增加 contract tests。MoE 还需要从 nested `text_config` 读取 router dtype，否则 train-side converter 可能用错误 dtype 写入 inference engine。

它没有公开端到端 speedup，因此不应包装成性能突破；真正价值是把 Qwen3-VL colocation 的 metadata/sharding correctness 做成正式 contract。

## Observed / Rejected Candidates

| 材料 | Source ID | Focus Match | Decision | 原因 |
|---|---|---|---|---|
| GC-OPD | arxiv:2608.19181 | P0 | Read | 用 verifier 与 teacher score 的 group-calibrated residual 修正长上下文 OPD；五项 benchmark 有提升，但主要是算法/credit assignment，暂不改变 runtime。 |
| rl-triton | arxiv:2608.17641 | P1 | Read | 用 associative scan 统一七种 RL return/advantage 算法，报告相对 torch-compile `1.6x-5.70x`；收益集中在 thousands of envs + short rollouts，需验证 LLM RL shape。 |
| Agentic ESOpt | arxiv:2608.17310 | P1 | Read | 用 evolution strategies 规避 backprop memory 并做 trajectory-level attribution，能以 inference-level memory 全参优化；参数扰动评估成本和扩展效率仍需正文核验。 |
| TileMix | arxiv:2608.17336 | P1 | Read | 在 fused dense attention 内按 tile 路由 FP16/INT8，并共享 online-softmax state；有 A100 prefill benchmark 和代码，属于 inference kernel 方向。 |
| Batched Speculative Jacobi Rollouts for Visual OPD | arxiv:2608.18183 | P1 | Observe | HB-SJD 只替换 student rollout backend，异步推进不同 image；当前证据来自 visual autoregressive model，不直接外推到 LLM OPD。 |
| MoE router locality negative result | arxiv:2608.18261 | P1 | Observe | edge serving 测量扎实，并诚实报告 cache miss 与 perplexity trade-off；单 8GB GPU/SSD workload 离当前集群主线较远。 |
| Megatron MFSDP v2 sharding + unified communication stream | github:NVIDIA/Megatron-LM#6137+#6563 | P1 | Read | 支持 no-shard/ZeRO-1/ZeRO-2，并允许 AG/RS 共用 stream 复用 storage；实现规模大，但缺少公开训练 benchmark。 |
| NeMo RL BF16 to MXFP8 NCCL reshard | github:NVIDIA-NeMo/RL#3477 | P1 | Read | weight sync 可在 refit 时量化为 MXFP8，直连低精度 rollout；需要评估 scale layout、量化开销和更新频率。 |
| vLLM Mooncake Store decode KV offload | github:vllm-project/vllm#52466 | P1 | Read | decode 端把新完成 KV blocks 回写远端 store，利于跨轮 prefix reuse；当前要求相同 TP size 和兼容 KV topology。 |
| LiquidAI LFM2.5 QAD | blog:huggingface/LiquidAI/qad | P1 | Observe | 厂商在 HF Blog 发布 Q4_0 QAD checkpoints，可与 NVIDIA QAD 对照；尚未升级为通用训练系统信号。 |
| OpenAI Zero Data Retention | blog:openai/zero-data-retention | Out of Scope | Observe | 对企业 API 数据治理重要，但不改变当前 training/RL/inference infra 技术判断。 |

## OpenAI / Anthropic / NVIDIA / DeepSeek Watch

| Vendor | Sources checked | Decision | 结果 |
|---|---|---|---|
| OpenAI | official RSS / research / engineering / releases | Observe | 本窗口有 Zero Data Retention、cyber policy 和产品案例；没有新的 Training/RL/Inference Infra 技术报告。 |
| Anthropic | official research / newsroom / sitemap | Not found | 本窗口未发现新的训练系统、RL runtime 或 inference engineering 一手材料；站点配置更新时间不作为文章发布时间。 |
| NVIDIA | Technical Blog / NeMo RL / Megatron-LM | **Accepted / Deep Dive** | NeMo RL async admission/health/recovery 与 Megatron multi-turn packed-logprob correctness 进入 Accepted；MFSDP v2、MXFP8 reshard 保留 Read。 |
| DeepSeek | API changelog / official Hugging Face organization | Not found | 官方 HF 最新仍为 DeepSeek-V4-Pro-0813，本窗口无新权重、model card 或 API changelog；SGLang 的 V4 backend 属第三方 runtime 实现。 |

## Hugging Face Watch

| Sources checked | Decision | 结果 |
|---|---|---|
| Hugging Face Blog | Observe | LiquidAI QAD 与 IBM agent memory 均为 vendor/community post；前者保留对照，后者不进入 infra 主线。 |
| TRL | **Accepted / Deep Dive** | `AsyncDistillationTrainer` 提供异步 OPD/MOPD pipeline；另有 AsyncGRPO metrics 整理和 Liger normalizer parity fix。 |
| Transformers | Observe | 新增 NVFP4 quantization、Qwen Omni compilable-cache correctness 等；均属重要能力/修复，但本窗口不单独升级。 |
| Accelerate / PEFT / Kernels | Routine only | 没有改变分布式训练、RL runtime 或 kernel execution model 的重大 release。 |

## RL Framework Watch

| Framework | Release / PR | 子系统 | 核心变化 | 证据 | 对 AReaL 的参考 | Decision |
|---|---|---|---|---|---|---|
| AReaL | [#1605](https://github.com/areal-project/AReaL/pull/1605) | weight sync / inference backend | Qwen3-VL Dense/MoE native AWEX colocation contract | merged patch + Dense/MoE tests | 完整 composite config 与 nested dtype 必须进入 metadata exchange | **Accepted / Read** |
| verl | [#7347](https://github.com/verl-project/verl/pull/7347), [#7453](https://github.com/verl-project/verl/pull/7453) | weight transfer / LoRA sync | non-blocking FSDP2 transfer；修正 vLLM LoRA sync index mismatch | default-branch commits | 检查 async transfer completion 与 adapter index identity | Observe |
| slime | default branch / release | - | 本窗口未发现重大架构、性能或 correctness 变化 | official feed | 无新增可迁移项 | Not found |
| ROLL | default branch / release | hardware | Ascend image/docs update | official feed | 不改变当前 GPU runtime | Ignore |
| OpenRLHF | default branch / release | - | 本窗口未发现重大变化 | official feed | 无新增可迁移项 | Not found |
| NeMo RL | [#3582](https://github.com/NVIDIA-NeMo/RL/pull/3582), [#3590](https://github.com/NVIDIA-NeMo/RL/pull/3590), [#3665](https://github.com/NVIDIA-NeMo/RL/pull/3665) | scheduler / rollout / recovery | AReaL-style admission、fleet health、dropped rollout replacement | merged patches + extensive unit/functional tests | 可直接对照 AReaL admission/retry/recovery state machine | **Accepted / Deep Dive** |
| Agent Lightning | arxiv:2608.17528 / v1.0 | harness / trainer boundary | 约 3,500 行 harnessed RL reference framework | paper + released workflow/scripts | 作为 endpoint proxy 与 trajectory contract 的最小对照 | **Accepted / Deep Dive** |
| LEGO-RL | arxiv:2608.17393 | harness / sandbox / observability | 原生 coding harness 的 stream capture、logprob recompute 与 sandbox defenses | paper + project page + three-harness evaluation | 对照现有 R2E/OpenHands data path 与 failure taxonomy | **Accepted / Deep Dive** |

## Adjacent Runtime Watch

| Runtime | 变化 | Decision | 工程判断 |
|---|---|---|---|
| Megatron-LM | RL multi-turn packing correctness；MFSDP v2 sharding；unified comm stream | **Accepted / Read** | 前者直接决定训练语义，后两者是显存/通信能力扩展，不能因 PR 大小混为同一优先级。 |
| vLLM | Mooncake decode KV offload；FlashInfer all-reduce default；spec-decode stride fix | Read / Observe | decode KV 可回流远端 prefix store；其余为 backend capability/correctness follow-up。 |
| SGLang | DSpark custom draft worker、top-k int32 overflow fix | Observe | custom worker 提高实验扩展性，overflow 属 correctness；本窗口没有升级为独立 Accepted 的变化。 |

## Reading Queue 判断

- [ ] **今天只读一个：Agent Lightning v1.0。** 先回答“harness 拥有 environment loop 后，trainer 还必须拥有哪几项不可妥协的语义”。
- [ ] **第二优先：Open-MOPD。** 重点判断 token-share、capability gap 和 reward age 是否足以解释现有 MOPD 训练失衡。
- [ ] LEGO-RL 和 NeMo RL bundle 保留 P1/项目对照；不把 7 条 Accepted 全部塞进 P0。

## 去重记录

- 新增 Accepted Source ID：`arxiv:2608.17528`、`arxiv:2608.17393`、`github:NVIDIA-NeMo/RL#3582+#3590+#3665`、`github:huggingface/trl#6705`、`arxiv:2608.19098`、`github:NVIDIA/Megatron-LM#5887`、`github:areal-project/AReaL#1605`。
- `Agent Lightning` 早期论文已在 Historical Backfill；v1.0 是本窗口的新论文和实现升级，不视为重复收录。
- `MOPD` 与 `SimpleOPD` 已有历史记录；Open-MOPD 的 capability-budget diagnosis 和完整 recipe 属新增信号。

## 扫描完整性

- arXiv：按 submittedDate 查询六个重点分类并集，共 439 条；对 Accepted 逐项核对 title、authors、date、abstract mechanism 和数字。
- Core vendors：OpenAI RSS、Anthropic research/news、NVIDIA Technical Blog/代码栈、DeepSeek changelog/HF 均显式检查。
- Frameworks：AReaL、verl、slime、ROLL、OpenRLHF、NeMo RL 以及 Agent Lightning/LEGO-RL 均检查 official feed、patch 或 primary paper。
- Hugging Face：Blog、TRL、Transformers、Accelerate、PEFT、Kernels 已检查；community/vendor posts 不自动升级。
- 盲区：GitHub Atom 只保留有限 commit；普通未合并 PR 和被 feed 截断的 commit 不能保证完整。厂商页面动态站点的 `lastmod` 不作为发布时间证据。
- 下一游标：`2026-08-20 10:21:56`。晚于该时刻的内容留给下一次扫描。

## 下一步动作

- [x] 更新 [Scan Log](scan_log.md) 与 [Tracking README](README.md)。
- [ ] 精读 Agent Lightning v1.0，输出 harness/trainer ownership matrix，不急着新建 topic。
- [ ] 精读 Open-MOPD 后轻量更新 [MOPD](../topics/mopd.md)，重点补 token budget 和 reward age 诊断。
- [ ] 将 Megatron #5887 的 multi-region logprob test 转成 AReaL packing correctness 实验候选。
- [ ] 下一次扫描从 `2026-08-20 10:21:56` 开始，继续按 Source ID 去重。

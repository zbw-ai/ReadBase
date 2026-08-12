# Frontier Scan, 2026-08-12

- Previous scan：[2026-08-09](frontier_scan_2026-08-09.md)
- Window：2026-08-09 23:24 ~ 2026-08-12 09:51
- Timezone：Asia/Shanghai
- Generated at：2026-08-12 10:14
- Report type：flexible frontier scan
- Sources scanned：arXiv cs.DC / cs.LG / cs.AI / cs.CL / cs.AR / cs.PF / cs.NI；OpenAI / Anthropic / NVIDIA / DeepSeek official sources；Hugging Face Blog / TRL / Transformers / Accelerate / PEFT / Kernels；AReaL / verl / slime / ROLL / OpenRLHF / NeMo RL releases、default-branch changes 与 major PR
- Scan completeness：完成本窗口相关 arXiv 分类、四家核心厂商、Hugging Face 生态与六个 RL 框架的完整候选扫描。对 arXiv 周末投稿但在上一游标后才进入公告面的材料，统一标记为 `boundary late-discovered`，不伪装成窗口内投稿。

## 本次核心判断

本窗口保留七条信号，不凑榜单。最值得关注的不是又出现了几个模型，而是三个更扎实的系统方向：

1. **长尾 workload 正在反向重塑并行策略。** verl 的 Dynamic Context Parallel 不再让每个 micro-batch 固定支付最大 CP 通信成本。
2. **RL 训推一致性已经下沉到 kernel、路由顺序和低精度表示。** slime 的 GLM-5 对齐路径说明，训练与 rollout 使用不同执行栈时，数值契约必须覆盖 DeepEP、DeepGEMM、DSA 和 FP8 KV cache。
3. **弹性与长上下文服务开始围绕状态搬运重构。** FlashBoot 优化 weight materialization，OasisKV 优化 decode 期 KV placement；它们都在减少“为了算一点东西，先搬一大块状态”的代价。

## Accepted Frontier Signals

### Dynamic Context Parallel Scheduling Lands in verl

- Signal ID：2026-08-12-001
- Source ID：github:verl-project/verl@38f43722
- First seen：2026-08-12 09:51（Asia/Shanghai，本次扫描）
- Scan window：2026-08-09 23:24 ~ 2026-08-12 09:51
- Focus Match：P0 Focus
- 来源：verl default branch commit / original PR benchmark
- 类型：framework implementation / long-context training / context parallelism
- 链接：https://github.com/verl-project/verl/commit/38f43722531d8870ec9f9a918de4a80fe728a4ff
- 发布时间：2026-08-11
- Primary-source check：commit date / merged default-branch state / mechanism / benchmark setup / throughput and loss numbers 已对齐官方 commit message；旧 PR 页面仍可能显示缓存状态，因此以 default-branch commit 为准
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：它把 CP 从 job-level 固定配置推进为 micro-batch-level 调度决策，直接针对 1K-16K 长尾序列下的 padding 与通信浪费。
- Status：NEW
- 建议动作：代码级阅读 scheduler、local token ownership 和 loss normalization；评估是否可迁移到 AReaL 的 Megatron training path
- 关联主题：[Long-context Training](../topics/long_context_training.md), [Context Parallelism](../topics/context_parallelism.md), [Agentic RL](../topics/agentic_rl.md)

verl 保留静态 CP 拓扑与每 rank 最大 sequence budget，但允许每个 packed micro-batch 根据实际长度选择更小的 local CP group。短 batch 不再被迫使用 CP4，因而减少不必要的 CP communication、padding 和 output routing。

官方 commit 报告 Qwen3-30B-A3B SFT、TP1/PP1/EP8、静态 CP4、1K-16K 长尾序列下，固定 CP 从 `209.97K tok/s` 提升到 Dynamic CP 的 `337.16K tok/s`，即 `+60.6%`；25 步 mean loss 相对差异为 `0.0229%`。这是单一设置的项目 benchmark，不应外推为通用收益，但它明确说明：**平均长度不足以配置 CP，micro-batch 的长度分布才决定真实通信成本。**

### slime Aligns GLM-5 Megatron Training with SGLang Rollout

- Signal ID：2026-08-12-002
- Source ID：github:THUDM/slime@a74ae3a0
- First seen：2026-08-12 09:51（Asia/Shanghai，本次扫描）
- Scan window：2026-08-09 23:24 ~ 2026-08-12 09:51
- Focus Match：P0 Focus
- 来源：slime default branch commit / merged PR #2262
- 类型：framework implementation / train-rollout consistency / MoE / FP8
- 链接：https://github.com/THUDM/slime/commit/a74ae3a0ad16bd8b769d5386738e8ae3d1269d7e
- 发布时间：2026-08-11
- Primary-source check：commit date / merged state / subsystem scope / test count / logprob MAE / hidden-state comparison 已对齐官方 commit 与 PR
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：它把“训练和推理 logprob 要一致”从一句原则落实到 exact top-k order、DeepEP dispatch、batch-invariant FP8 kernel、DSA sparse attention 与 KV cache 表示。
- Status：NEW
- 建议动作：精读 route metadata capture、ordered gather、deterministic backward 和 CI alignment gate；与 AReaL 的 train-rollout consistency tests 对照
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [MoE](../topics/moe.md), [FP8](../topics/fp8.md)

这条改动覆盖 Megatron training、SGLang rollout、DeepGEMM、DeepEP、DSA sparse attention 和 FP8 KV cache。DeepEP 路径记录精确 top-k slot 顺序并传输 compact route metadata，按 token owner 顺序 gather，并为 backward 保持确定性；dense/MoE kernel 则对齐 accumulation order、router GEMM、activation quantization 与 padding 规则。

项目报告 100 个 focused tests；4096-token logprob MAE 为 `1.8943116231e-7`，decoder layers 0-5 在每层 4593 个 matched tokens 上 hidden-state diff 为零，并称已用于 750B production-level training。后者是项目方披露而非外部复现，但代码与 CI 证据足够强。对 AReaL 的直接启发是：一致性检查不能只比较最终 loss，必须能逐层定位到 router、kernel、quantization 和 token layout。

### FlashBoot: Sub-Second Weight Loading for Large Models at Rack Scale

- Signal ID：2026-08-12-003
- Source ID：arxiv:2608.08482
- First seen：2026-08-12 09:51（boundary late-discovered；投稿早于上一游标，公告面在本次窗口出现）
- Scan window：2026-08-09 23:24 ~ 2026-08-12 09:51
- Focus Match：P0 Focus
- 来源：arXiv / SGLang-based system
- 类型：paper / model loading / elasticity / rack-scale state movement
- 链接：https://arxiv.org/abs/2608.08482
- 发布时间：2026-08-09
- Primary-source check：title / six authors / date / mechanism / NCCL setup range / latency and bandwidth claims 已对齐 arXiv metadata 与摘要
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：它攻击的不是 steady-state kernel，而是大模型弹性扩缩容和 RL role switching 中越来越明显的 weight materialization 冷启动。
- Status：NEW
- 建议动作：精读 FabricArena memory layout、remote mapping lifetime、failure semantics 和安全隔离；等待 promised code 后再判断可复现性
- 关联主题：[Distributed Training](../topics/distributed_training.md), [Checkpointing](../topics/checkpointing.md), [Agentic RL](../topics/agentic_rl.md)

FlashBoot 将数万块 per-tensor allocation 变为 contiguous、exportable、inter-node addressable 的 `FabricArena`。`FlashLoad` 从 CPU 做 bulk zero-copy transfer；`FlashClone` 通过 remote mapping 复制 resident model，避开跨节点 clone 前 `10-110s` 的 NCCL communicator setup。

作者在 NVL72 上报告 remote weight mapping 约 `10ms`、每 clone 至少 `700GB/s`，单节点加载从 `20.1s` 降至 `0.4s`，并发 rack-level 从 `87s` 降至 `0.32s`。这些是作者在特定硬件与模型上的结果，且代码尚未公开。真正值得带回 RL Infra 的问题是：policy weight 是否能拥有稳定、可导出的连续布局，从而让 rollout replica 的启动/切换不再走 checkpoint-style materialization。

### OasisKV: Scaling In-Decode KV Cache Beyond HBM

- Signal ID：2026-08-12-004
- Source ID：arxiv:2608.08097
- First seen：2026-08-12 09:51（boundary late-discovered）
- Scan window：2026-08-09 23:24 ~ 2026-08-12 09:51
- Focus Match：P0 Focus
- 来源：arXiv / vLLM implementation
- 类型：paper / long-context serving / KV cache / prefill-decode disaggregation
- 链接：https://arxiv.org/abs/2608.08097
- 发布时间：2026-08-08
- Primary-source check：title / ten authors / date / vLLM implementation / KV budget / throughput and transfer claims 已对齐 arXiv metadata 与摘要
- 影响等级：★★★★★
- Decision：Read
- Reason：它把 speculative lookahead 从“减少 target decode 次数”转为“提前预测下一步真正需要的 KV block”，为 long-horizon rollout 的 KV 分层提供了新路径。
- Status：NEW
- 建议动作：核对 sparse-attention accuracy boundary、prefetch miss、host/remote tier bandwidth 和 speculative draft overhead
- 关联主题：[Long-context Training](../topics/long_context_training.md), [Agentic RL](../topics/agentic_rl.md), [Rollout Latency](../playbooks/rollout_latency.md)

OasisKV 不把 full KV cache 常驻 HBM，而是利用 speculative decoding 的 lookahead token 预测未来重要 token，后台识别 KV blocks，并从 host/remote memory 预取到 HBM。它因此把 capacity expansion、sparse attention 与 prefetch timing 合并成一个 decode pipeline。

作者在 vLLM 实现中报告：2048-token KV budget 下与 full attention 相差不超过 `0.7` accuracy point；reasoning workload 在 `0.1` point loss 时达到 dense vLLM 的 `1.69x`，多 GPU 长上下文最高 `2.1x`。PD disaggregation 下约 `2x` throughput，并减少 `6.5-9.7x` KV transfer。这里的核心风险不是平均 accuracy，而是 miss 是否集中在关键 reasoning step，以及长尾 prefetch 是否重新制造 decode stall。

### SwiftQK: Communication-Efficient TP for Query-Key Normalization

- Signal ID：2026-08-12-005
- Source ID：arxiv:2608.09160
- First seen：2026-08-12 09:51（Asia/Shanghai，本次扫描）
- Scan window：2026-08-09 23:24 ~ 2026-08-12 09:51
- Focus Match：P1 Focus
- 来源：arXiv
- 类型：paper / tensor parallelism / normalization kernel / inference communication
- 链接：https://arxiv.org/abs/2608.09160
- 发布时间：2026-08-10
- Primary-source check：title / three authors / date / scalar-statistics mechanism / QK-Norm latency and TPOT claims 已对齐 arXiv metadata 与摘要
- 影响等级：★★★★☆
- Decision：Read
- Reason：它展示了一个常被忽略的 TP 原则：算子需要全局统计量，不等于必须 all-gather 全向量。
- Status：NEW
- 建议动作：核对 persistent-kernel deadlock avoidance、P2P reduction topology 与不同 TP size 的收益曲线
- 关联主题：[Tensor Parallelism](../topics/tensor_parallelism.md), [NCCL](../topics/nccl.md), [Long-context Training](../topics/long_context_training.md)

QK-Norm 的 normalization factor 依赖完整 hidden vector，标准 TP 实现容易 all-gather 整个向量。SwiftQK 只交换 scalar normalization statistics，并在 deadlock-safe persistent kernel 中让剩余 P2P reduction 与 independent element-wise compute overlap。

作者报告相对 full-vector all-gather，QK-Norm latency 降低 `81.4%-93.9%`；端到端 serving 的 TPOT 平均降低 `29.5%`，相对优化后的 scalar aggregation 仍降低 `14.3%`。它给 Training Infra 的判断很直接：在引入一个 collective 前，先问下游真正需要的是完整 tensor、partial reduction，还是几个 scalar sufficient statistics。

### The Replay Gap: Static Evaluation Scores the Wrong Agent World

- Signal ID：2026-08-12-006
- Source ID：arxiv:2608.08239
- First seen：2026-08-12 09:51（boundary late-discovered）
- Scan window：2026-08-09 23:24 ~ 2026-08-12 09:51
- Focus Match：P0 Focus
- 来源：arXiv / released harness and trajectories
- 类型：paper / agent evaluation / model routing / rollout correctness
- 链接：https://arxiv.org/abs/2608.08239
- 发布时间：2026-08-08
- Primary-source check：title / author / date / six paired runs / rollout count / divergence and replay-validity claims 已对齐 arXiv metadata 与摘要
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：它推翻了 per-step agent router 的常见离线评估假设：换模型会改变后续环境状态，不能把新输出简单缝回旧 trajectory。
- Status：NEW
- 建议动作：与 NVIDIA Switchyard 一起读；把 live branching evaluation、same-model noise floor 和 serving determinism 纳入 router benchmark
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [Long-context Training](../topics/long_context_training.md)

论文在 SWE-bench agent trajectory 的控制点重建环境并 fork，由不同模型继续执行；same-model control 用于隔离 sampling/replay noise。约 900 rollouts 中，换模后 `61%-94%` action 被改写，只有 `3%` replayed state 仍有效；log-stitching 对所有 success-relevant outcome flip 都预测错误。

更重要的是，它还观察到所谓 temperature-0 determinism 依赖 serving configuration：FP8 与 AWQ 路径的 control divergence 显著不同。这意味着 router 评估必须记录 model、weights、quantization、engine 和 environment snapshot，不能只记 model name。对于 RL Infra，这和 off-policy trajectory replay 是同一类状态一致性问题。

### NVIDIA Nemotron 3.5 Lightning

- Signal ID：2026-08-12-007
- Source ID：hf:nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
- First seen：2026-08-12 09:51（Asia/Shanghai，本次扫描）
- Scan window：2026-08-09 23:24 ~ 2026-08-12 09:51
- Focus Match：P1 Focus
- 来源：NVIDIA official model card
- 类型：industrial release / efficient agent model / post-training base
- 链接：https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
- 发布时间：2026-08-11
- Primary-source check：model size / active parameters / architecture / context / intended post-training path / release date 已对齐 NVIDIA model card
- 影响等级：★★★★☆
- Decision：Read
- Reason：这不是单纯的轻量模型发布，而是把 BF16 customization base、NVFP4 deployment、MTP/DSpark、NeMo RL 与可复核 evaluation recipes 放到同一条工业交付路径。
- Status：NEW
- 建议动作：优先读 model card 的 architecture/training/deployment recipes；厂商 benchmark 需与外部 live workload 结果交叉看
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [MoE](../topics/moe.md), [FP8](../topics/fp8.md)

Nemotron 3.5 Lightning 是 `30B total / 3B active` 的 Mamba-2 + MoE + selective Attention hybrid，提供 BF16 reference weights、NVFP4 deployment path、MTP/DSpark 路径，并将 SFT、RL、distillation 与 domain adaptation 明确列为主要用途。官方 model card 标注最高 1M context，但单 H100 recipe 使用 256K；这两者不能混为同一部署承诺。

NVIDIA 同期继续强化 Switchyard 这一 OpenAI/Anthropic-compatible typed traffic control plane，但 Switchyard 早于本窗口已经公开，因此这里只作为工业栈关联，不计作新 Source ID。它支持 format translation、fallback、session affinity、classifier/stage routing 与 request/token statistics；`The Replay Gap` 提醒我们，router 仍必须通过 live fork 而不是静态 log stitching 验证。

## Observed / Rejected Candidates

| 材料 | Source ID | Focus Match | Decision | 原因 |
|---|---|---|---|---|
| LLMVisor | arxiv:2608.08382 | P1 | Observe | microsecond-scale per-request latency attribution 对 multi-tenant serving 很有用，p90/p99 error 也有数据；但目前主要解决 cost attribution/control primitive，先不挤占更直接的 RL/long-context 主线。 |
| Continuous Depth Batching | arxiv:2608.09444 | P1 | Observe | 真正实现 loop-iteration-level scheduling，并报告接近理论 adaptive-depth speedup；但依赖 looped LM 架构，迁移到主流 Transformer serving 的价值尚不明确。 |
| C2C-Explorer | arxiv:2608.08611 | P1 | Observe | workload-driven C2C design-space exploration、512-chip simulation 与 FPGA validation 很扎实，但更偏硬件架构研究，短期不改变 AReaL/训练平台实现。 |
| ZeroLock | arxiv:2608.07974 | Out of Scope | Observe only | BP-free local objective 面向 edge fine-tuning；4.9% throughput gain 尚不足以抵消训练语义与收敛风险，不进入当前主线。 |
| Anthropic Riemann proof update | anthropic:research/riemann | P1 | Observe | 能力信号重要，但本窗口材料主要是 formal reasoning/result，不含可迁移 Training/RL Infra 机制。 |
| Hugging Face community distillation post | hf-blog:cheap-distillation-at-scale | P1 | Observe | 与低成本 post-training 相关，但不是 Hugging Face 核心团队发布，且系统证据不足以单独进入 Accepted。 |
| NeMo RL MXFP8 refit + external judge pools | github:NVIDIA-NeMo/RL#3478/#3529 | P1 | Read | 前者将 MoE refit shuffle 从 per-expert 改为 batched gather，项目报告 transfer+update `4.021s -> 0.700s`；后者把 GenRM/NL2Bash 放到独立 Slurm/vLLM pools。二者都可迁移，但不再额外拆成 frontier 条目。 |

## OpenAI / Anthropic / NVIDIA / DeepSeek Watch

| Vendor | Sources checked | Decision | 结果 |
|---|---|---|---|
| OpenAI | official research / engineering / releases | Not found | 本窗口未发现可核验、且包含 Training/RL/Inference Infra 机制的新一手来源；近期 academic-research program 属使用与生态信息，不进入本轮。 |
| Anthropic | official research / newsroom | Observe | 发现 formal mathematics capability update，但没有新的 Training Infra / RL Infra 技术披露，因此仅记录能力信号。 |
| NVIDIA | official model card / NeMo docs / NeMo RL | **Accepted / Read** | Nemotron 3.5 Lightning 进入 Accepted；Switchyard 作为既有 routing control plane 关联观察，不冒充本窗口新发布；NeMo RL 的 batched MXFP8 refit 和 external judge pools 进入 framework watch。数字均按厂商/项目方披露处理。 |
| DeepSeek | API changelog / official Hugging Face organization / GitHub | Not found | 已补回上次 Hugging Face 可见性盲区；未发现 07-31 V4-Flash-0731 之后的新模型、技术报告、API 或训练栈发布。 |

## Hugging Face Watch

| Sources checked | Decision | 结果 |
|---|---|---|
| Hugging Face Blog | Observe | 本窗口有 distillation/community 内容，但未发现足够改变 Agentic RL、distributed training 或 inference backend 判断的 HF 核心团队文章。 |
| TRL / Transformers / Accelerate / PEFT / Kernels | Not found / routine only | 已检查 release/docs 与可见重大变化；本窗口未发现值得升级为独立 frontier signal 的架构、性能或 correctness 变化。 |
| NVIDIA model card hosted on Hugging Face | **Accepted / Read** | Nemotron 3.5 Lightning 作为 vendor-authored primary source 单独核验；托管平台不改变其 NVIDIA 官方来源属性。 |

## RL Framework Watch

| Framework | Release / PR | 子系统 | 核心变化 | 证据 | 对 AReaL 的参考 | Decision |
|---|---|---|---|---|---|---|
| AReaL | default branch dependency update | rollout runtime | AWEX 更新到 0.8.0；本窗口未看到配套架构说明或独立 benchmark | official default-branch activity | 等明确 changelog/behavior diff 后再判断，不把依赖版本变化当信号 | Observe |
| verl | [38f43722](https://github.com/verl-project/verl/commit/38f43722531d8870ec9f9a918de4a80fe728a4ff) | training / scheduler | Dynamic CP 按 micro-batch 长度选择 local CP group，减少 padding 与 CP communication | merged default-branch commit；tests；Qwen3-30B-A3B benchmark | AReaL 的长上下文 training scheduler 可借鉴 local CP selection 和 token ownership routing | **Accepted / Deep Dive** |
| slime | [a74ae3a0](https://github.com/THUDM/slime/commit/a74ae3a0ad16bd8b769d5386738e8ae3d1269d7e) | training / rollout / data path | GLM-5 Megatron-DeepEP-SGLang deterministic alignment，覆盖 FP8/DSA/MoE route | merged default-branch commit；100 tests；layerwise/logprob gate | 为 AReaL 建立 kernel-to-logprob 分层 consistency suite，而不只做 E2E loss check | **Accepted / Deep Dive** |
| ROLL | Not found | - | 未发现 release 或会改变架构、性能、正确性的重大合并项 | official repo activity | 继续观察 | Not found |
| OpenRLHF | Not found | - | 未发现正式 release 或足够强的 runtime 变化 | official repo activity | 继续观察 | Not found |
| NeMo RL | [#3478](https://github.com/NVIDIA-NeMo/RL/pull/3478), [#3529](https://github.com/NVIDIA-NeMo/RL/pull/3529) | weight sync / reward service | batched MXFP8 MoE refit shuffle；固定 GenRM/NL2Bash 解耦为 external vLLM pools | default-branch change / merged #3529 / code diff / isolated benchmark | AReaL 可检查 MoE refit layout transform 是否仍 per-expert；reward/judge 是否应独立资源池 | Read |

### Framework Watch 的工程结论

本窗口的框架信号比论文更贴近当前工作：**verl 在减少长尾序列的并行浪费，slime 在封死训练-推理数值漂移，NeMo RL 在优化 weight refit 并拆出固定 judge service。** 对 AReaL 最值得做的不是立刻复制 feature，而是先补三个可观测量：micro-batch length-to-CP decision、layerwise train/rollout diff、weight refit 各阶段耗时。

## Reading Queue Updates

- [ ] `P0.md` 仍保持上限 3，本次不机械追加。
- [ ] 若今天只读一项：先读 verl Dynamic CP commit/PR，再对照 AReaL 当前 128K SFT/RL micro-batch scheduler。
- [ ] 若有 90 分钟：读 slime #2262 的 mechanism 与 tests，重点理解为何 exact top-k order 和 batch-invariant kernel 会影响 PPO/GRPO logprob。
- [ ] FlashBoot、OasisKV、Replay Gap 保留为下一轮候选，完成已有 P0 后再替换进入队列。

## 去重记录

- 新增 Source ID：`github:verl-project/verl@38f43722`、`github:THUDM/slime@a74ae3a0`、`arxiv:2608.08482`、`arxiv:2608.08097`、`arxiv:2608.09160`、`arxiv:2608.08239`、`hf:nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`。
- 关联但不计作新信号：`github:NVIDIA-NeMo/Switchyard`，该项目早于本窗口已经公开。
- Boundary late-discovered：FlashBoot、OasisKV、The Replay Gap；它们投稿时间早于上一游标，但在本轮 arXiv 公告面出现，已显式标记，不回写上一份 scan。
- Follow-up：NeMo RL #3478/#3529 只进入 framework watch，不重复拆为 Accepted signal。

## 扫描完整性

- 已扫描：arXiv 七个相关分类 recent listings 与 accepted 候选 abstract page；四家核心厂商官方可见来源；Hugging Face Blog 与核心库；六个 RL 框架 official release/default branch/major PR。
- 网络说明：arXiv API 在本次扫描中响应不稳定，已回退分类 recent listing 和逐篇 primary-source page；部分 GitHub API 请求超时，已用 official commit/PR HTML 与 default-branch commit 交叉核验。
- 已知盲区：未公开的厂商内部报告、只有社交媒体转述而无一手页面的消息、未合并 draft PR。
- 下一游标：`2026-08-12 09:51`。本报告生成后的新材料留给下一次扫描，不虚构已覆盖到 10:14。

## 下一步动作

- [x] 更新 [Scan Log](scan_log.md)。
- [ ] 阅读：verl Dynamic CP implementation；slime GLM-5 train/rollout alignment；FlashBoot state layout。
- [ ] 形成工程判断后再更新 [Long-context Training](../topics/long_context_training.md) 与 [Agentic RL](../topics/agentic_rl.md)，不根据摘要抢跑沉淀。
- [ ] 可验证实验：在 AReaL 128K workload 上记录长度分布、固定 CP 通信占比与 hypothetical local CP；分解 refit 的 transfer、layout transform、update 时间。

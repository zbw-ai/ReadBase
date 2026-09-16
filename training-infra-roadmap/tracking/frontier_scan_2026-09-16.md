# Frontier Scan, 2026-09-16

- Previous scan：[2026-09-07](frontier_scan_2026-09-07.md)
- Window：2026-09-07 10:00:26 ~ 2026-09-16 14:54:26
- Timezone：Asia/Shanghai；结束时刻由本次时钟核验（UTC 06:54:26），不是日终时间。
- Report type：增量扫描 + 接手核验；Accepted 10，Observed 10（按下方独立候选行计数，Watch 不重复计数）。
- First seen：下列新条目统一登记为 2026-09-16 14:54:26；Scan window 均继承上述窗口。
- Status：Accepted 均为 NEW；读过摘要、相关机制、release 或 patch 不等于完成精读，更不等于 VERIFIED。

## 覆盖范围与证据边界

读取原任务交接，并核对 `0f698ac`、[Scan Log](scan_log.md)、[8 月月报](monthly_signal_2026-08.md)、P0/P1 和 learning log。工作基线为 `330176b`，包含后续面试手册更新。上轮五条信号保留为历史记录，本轮不重新背书其中未复核的数字。

arXiv 使用 cs.LG / cs.AI / cs.CL / cs.DC / cs.AR 的 9 月列表，补齐 LG/AI 的第二页；原始页分别返回 2149 / 2434 / 1317 / 265 / 143 条，跨分类去重后取得 4845 个标题。**这是整月标题池，不是窗口内论文数，也不是逐篇读完的数量**。按训练系统、并行、MoE、kernel、rollout、通信与 OPD 筛选，再对候选核对首次提交时间。Accepted 论文实际读取 `citation_title`、`citation_author`、`citation_date`、摘要与方法页面；未把 ID 可解析当成验证。早于窗口的 FP4 FlashAttention-4（2609.04105）不冒充本轮新增，待有明确学习缺口时走 backfill。

GitHub REST API 对七个核心仓库均返回匿名 rate limit。官方 HTML 部分缓存停在 9 月 10–11 日，随后用直接读取的 Atom 和固定 SHA patch 补查。**Atom 仅返回最近 20 条，`page=2/3/4` 仍返回同一批，不能视为成功分页。** AReaL/slime/ROLL 的 feed 已跨回起点；verl 的 HTML 与新 feed 可衔接；NeMo RL、Megatron-LM、OpenRLHF 尚不能证明整个窗口的所有 merged PR 均覆盖。release/index 检查也不覆盖尚未合并的全部 PR。

本次记录实际扫描终点，但对上述框架历史缺口及 HF 官方文章目录保留 **补扫起点 2026-09-07 10:00:26**。下次必须先补扫、按 Source ID 去重，不能仅从全局新游标向后看。没有发现与无法完整核验分别记录。

## 本次核心判断

1. **恢复能力必须连同训练语义一起验收。** 部分 group 是否可用、checkpoint 何时可见、重启 worker 何时重新接流量，是三个不同的提交边界。
2. **长上下文的显存瓶颈会迁移。** Attention 优化之后，expert dispatch、vocabulary projection、checkpoint 输入与 optimizer state 都可能成为下一个峰值。
3. **减少搬运不等于减少正确性检查。** hidden-state 蒸馏、压缩 KV、draft co-training 都需要验证重建分布、状态版本与并行布局。

## Accepted Frontier Signals

<a id="deepseek-v41"></a>
### 1. DeepSeek-V4.1-Flash：把 KV 状态和 Agent 数据生产放到架构中心

- Source ID：`hf:deepseek-ai/DeepSeek-V4.1-Flash`；类型：官方 model card / open weights / technical report 入口。
- 原题：DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression；作者：DeepSeek-AI；发布日期：2026-09-10，由 [API changelog](https://api-docs.deepseek.com/updates/) 与 [官方 HF model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) 交叉核对。
- Impact：★★★★★；Decision：Deep Dive。
- Reason / 一句话价值：CED、跨层 KV/index 复用与 SWA bounded replay 改变了长时 agent 的状态成本；这不是只更新 benchmark 分数的模型发布。
- 已核对：model card 正文披露 CED、CSA2、FP4 main KV，以及 SFT → RL → OPD 路线。其 **890 bytes/token 指 global KV**，不能当作全部 KV、模型显存或训练 activation 大小；数字为厂商自报。技术报告 PDF 有官方入口，本轮未完成全文精读。
- Next：先列清 encoder/decoder、global/SWA KV 与 replay 的状态边界，再核验训练侧 CP、重算和 rollout backend 支持；不直接套用 HF 自动生成的部署命令。
- 主题：[Long Context](../topics/long_context_training.md)、[MoE](../topics/moe.md)、[Agentic RL](../topics/agentic_rl.md)。

<a id="areal-groups"></a>
### 2. AReaL：Incomplete Group 的正确性跨越采样、归一化和 collective

- Source ID：`github:areal-project/AReaL@64f049f1e5b4cdedce57e3ae0a5091f4121f3fa5`。
- 原题：fix(rollout): train safely on incomplete groups (#1563)；作者：Maxwill Lin；时间：2026-09-15 14:08:34；[固定 commit / patch](https://github.com/areal-project/AReaL/commit/64f049f1e5b4cdedce57e3ae0a5091f4121f3fa5)。
- 类型：framework / rollout / data/trajectory path；Impact：★★★★★；Decision：Deep Dive。
- Reason / 一句话价值：保留部分成功样本时，logical rollout 数、tensor row 数与有效 token 数必须各自有定义。
- 代码证据：`min_usable_group_size` 根据 estimator 是否需要 group statistics 推导下限；该配置仅由 v1 rollout 路径消费。patch 同时覆盖 ragged gather、zero-trajectory rank、动态补采停止条件及相应 tests。不是把 `drop_incomplete_group` 改成 false 就完成支持。
- Next / AReaL 可迁移性：直接关联当前框架；逐项检查逻辑成员统计、padding loss mask、split trajectory 和 collective 次序，执行[故障注入方案](../experiments/rl_state_boundaries.md)。本轮只审阅测试，未运行上游测试。
- 主题：[Agentic RL 状态边界](../topics/agentic_rl.md#rl-state-boundaries)。

<a id="areal-checkpoint"></a>
### 3. AReaL：用不可变 generation 与 LATEST 发布 checkpoint

- Source ID：`github:areal-project/AReaL@e23a935362d8d535a4cad3a1d0a8100e0c0938fd`。
- 原题：fix(recover): publish immutable checkpoints via latest pointer (#1616)；作者：zhangjw（patch）；时间：2026-09-11 11:18:27；[commit](https://github.com/areal-project/AReaL/commit/e23a935362d8d535a4cad3a1d0a8100e0c0938fd)。
- 类型：framework / checkpoint/recovery；Impact：★★★★★；Decision：Read。
- Reason / 一句话价值：checkpoint payload 落盘和“可以被恢复端发现”必须分离。
- 代码证据：每步写入不可变 generation，payload 完成后才发布 LATEST；Megatron async save 延迟到 finalize；多个 engine 的非发布保存先 drain，再安排最终 publisher；actor/critic 保存与加载使用稳定映射。legacy SPMD layout 仍保留，不能泛化为所有路径都已事务化。
- Next / AReaL 可迁移性：检查 actor 完成但 critic 失败时 LATEST 是否保持上一个完整 generation；结合 [Checkpointing](../topics/checkpointing.md) 验证 storage 的 rename/可见性语义。

<a id="nemo-recovery"></a>
### 4. NeMo RL：重启 generation shard 后，等下一轮 refit 再准入

- Source ID：`github:NVIDIA-NeMo/RL@53bce0568e2a2a9d49985f2157050da5e59a70ce`。
- 原题：feat(sc): restart dead generation shards and re-admit them at the next refit (#3592)；作者：Terry Kong；co-author 含 asolergibert、Claude；时间：2026-09-15 07:22:54；[commit](https://github.com/NVIDIA-NeMo/RL/commit/53bce0568e2a2a9d49985f2157050da5e59a70ce)。
- 类型：framework / scheduler / weight sync / checkpoint/recovery；Impact：★★★★★；Decision：Deep Dive。
- Reason / 一句话价值：进程存活不是已持有当前 policy 的证据。
- 代码证据：replacement 先进入 stale，参与后续 refit 后才回到 serving；tests 明确覆盖 mid-refit 完成重启仍须等待下一轮、current weight version、restart budget/backoff 和 timeout。自动重启默认关闭；本轮未做 GPU 或 Ray 复现。
- Next / AReaL 可迁移性：迁移的是 incarnation、version 与 admission 联合约束，不能直接照搬 NeMo controller。对照 AReaL router 与 weight updater，注入“重启恰好发生在 refit 中途”。
- 主题：[Fault Tolerance](../topics/fault_tolerance.md)、[Agentic RL](../topics/agentic_rl.md#rl-state-boundaries)。

<a id="nemo-full-opd"></a>
### 5. NeMo RL：Full-vocabulary OPD 的数据边界与独立正确性检验

- Source ID：`github:NVIDIA-NeMo/RL@c49d53e2e46e95ba645870fead6d48e36c37259b`。
- 原题：feat(mopd): add full-vocabulary on-policy distillation to SingleController (#3978)；作者：Rayen，co-author Claude；时间：2026-09-10 16:10:06；[commit](https://github.com/NVIDIA-NeMo/RL/commit/c49d53e2e46e95ba645870fead6d48e36c37259b)。
- 类型：framework / training / data/trajectory path；Impact：★★★★☆；Decision：Read。
- Reason / 一句话价值：把 teacher hidden states 运到 student 再做词表投影，可以缩窄 payload，但必须正确重建 teacher 分布。
- 已读 patch：`docs/about/algorithms/mopd.md`、recipe 和测试入口。当前限 Megatron + SingleController、一个 teacher；hidden-state 路径另要求 student PP=1、temperature=1，并拒绝不能由线性 output head 重建的 post-logit transforms。chunk 控制词表工作集。
- Next / AReaL 可迁移性：用同一 checkpoint 做 self-distillation，并注入错位 token、错误 head shard、payload 损坏。KL 分解恒等式可能在错误输入下仍成立，不能替代独立 oracle；迁移时先定义 teacher/student token 与 vocab contract。
- 主题：[MOPD](../topics/mopd.md)、[Agentic RL](../topics/agentic_rl.md)。

<a id="draft-cotraining"></a>
### 6. Online Draft Co-Training：CP branch attention 与 PP feature transport

- Source ID：`arxiv:2609.07108v1`；类型：technical report；作者：Zili Wang、Zhaopeng Qiu、Yuekai Zhang、Shuang Yu、Junjie Lai。
- 原题：Online Draft Co-Training for Speculative Decoding in Large-Scale, Long-Context RL Post-Training；提交：2026-09-07 06:48:29 UTC，晚于起点；[元数据](https://arxiv.org/abs/2609.07108)、[方法正文](https://arxiv.org/html/2609.07108v1)。
- Impact：★★★★★；Decision：Read。
- Reason / 一句话价值：在线训练 drafter 时，不能假定它自然继承 policy 的 CP/PP。
- 已核对：causal 主序列 ring attention 与 rank-local branch attention 合并；TapChannel 在 PP stage 间单独搬运 target features。作者报告最高 122B 模型、256K context 的实验，未在此泛化为所有 RL 配置收益。
- 代码状态：[官方仓库 issue #3698](https://github.com/NVIDIA-NeMo/RL/issues/3698) 是 upstream 路线图，列出 stacked draft PR；**论文结果不等于 main/release 已完整具备该能力**。
- Next：为 AReaL 列出 draft feature owner、detach、CP mask、weight-version/refit contract，分别测 draft train、feature transport、rollout 和 E2E 开销。
- 主题：[Context Parallelism](../topics/context_parallelism.md)、[Agentic RL](../topics/agentic_rl.md)。

<a id="memory-peaks"></a>
### 7. Flattening Every Memory Peak in Long-Context Mixture-of-Experts Training

- Source ID：`arxiv:2609.14306v1`；类型：paper；作者：Shrey Pandit、Xuan-Phi Nguyen、Yiran Zhao、Shafiq Joty。
- 提交：2026-09-13 06:00:46 UTC；[元数据与摘要](https://arxiv.org/abs/2609.14306)、[正文](https://arxiv.org/html/2609.14306v1)。
- Impact：★★★★★；Decision：Read。
- Reason / 一句话价值：用四个独立峰值解释“已经做了 Attention 优化仍然 OOM”。
- 已核对：dispatch chunk 上限、Ring-DTP vocabulary projection、checkpoint 输入 CPU offload、bucket 化 optimizer offload 分别约束不同工作集。作者声称保持 loss/gradient 语义；本轮未验证数值等价或复现性能，不摘取最高倍率作为整体收益。
- Next：先用自己 workload 的 memory timeline 判断哪一峰值先溢出；比较 offload 的 PCIe/CPU 带宽代价，避免一次开启全部机制而失去归因能力。
- 主题：[Long Context](../topics/long_context_training.md)、[MoE](../topics/moe.md)、[FSDP](../topics/fsdp.md)。

<a id="mkernel"></a>
### 8. mKernel: Fast Multi-GPU, Multi-Node Fused Kernels

- Source ID：`arxiv:2609.13585v1`；类型：paper；作者：Ziming Mao、Yihan Zhang、Shawn Wei Chew、Shuang Ma、Costin Raiciu、Yang Zhou、Scott Shenker、Ion Stoica。
- 提交：2026-09-11 22:50:04 UTC；[元数据](https://arxiv.org/abs/2609.13585)、[正文](https://arxiv.org/html/2609.13585v1)。
- Impact：★★★★☆；Decision：Read。
- Reason / 一句话价值：把 compute/NVLink/RDMA overlap 的粒度降到 tile，同时考虑通信占用 SM 的代价。
- 已核对：persistent kernel 划分 compute/communication SM，运行时调节比例，分层搬运减少跨节点 bytes；host proxy 与 RDMA verbs 是重要实现选择。作者在两个 16-GPU H200 集群上的 kernel 测量，不是万卡训练 E2E 结论；不推导 IBGDA 普遍无用。
- Next：先对 GEMM+collective 建立 bytes、rank arrival skew、SM 竞争的基线，再评估 tile fusion；关注 shape 小或网络拥塞时退化。
- 主题：[NCCL](../topics/nccl.md)、[Tensor Parallelism](../topics/tensor_parallelism.md)。

<a id="hf-memory"></a>
### 9. HF 训练栈：Chunked Loss 与 FSDP2 checkpoint 边界应一起验收

- Source IDs：`github:huggingface/trl:release:v1.13.0`、`github:huggingface/accelerate:release:v1.15.0`；类型：官方 release notes；发布者：qgallouedec / SunMarc；日期：2026-09-10 / 2026-09-09。
- 原题：[TRL v1.13.0](https://github.com/huggingface/trl/releases/tag/v1.13.0)、[v1.15.0: FSDP2 activation memory, dtensor improvements](https://github.com/huggingface/accelerate/releases/tag/v1.15.0)。
- Impact：★★★★☆；Decision：Read。
- Reason / 一句话价值：loss projection dtype 与 checkpoint wrapping 层级都能让配置名相同的训练走出完全不同的显存和算力路径。
- 已核对：TRL 修正 chunked projection 的不必要 FP32 upcast；Accelerate 改为包住匹配的 transformer layer，并支持 checkpoint 输入 offload，另修复 FSDP2/PEFT full-state 保存遗漏 adapter shard。只按 release 声明登记，未本地运行。
- Next：记录实际 GEMM dtype、autocast、每层保存 activation、adapter round-trip；同时检查 loss/gradient parity。上轮已收过 1M-token recipe，本轮不重复计为新发现。
- 主题：[FSDP](../topics/fsdp.md)、[Long Context](../topics/long_context_training.md)、[Checkpointing](../topics/checkpointing.md)。

<a id="jax-moe"></a>
### 10. NVIDIA：JAX Dropless MoE 的 kernel、EP 与 offload 联合优化

- Source ID：`official:nvidia:accelerating-dropless-moe-training-in-jax-with-nvidia-transformer-engine`。
- 原题：[Accelerating Dropless MoE Training in JAX with NVIDIA Transformer Engine](https://developer.nvidia.com/blog/accelerating-dropless-moe-training-in-jax-with-nvidia-transformer-engine/)；作者：Seonghee Lee、Jeremy Berchtold、Phuong Nguyen、Teddy Do、Tejash Shah；日期：2026-09-14。
- 类型：官方 engineering blog；Impact：★★★★☆；Decision：Read。
- Reason / 一句话价值：dropless MoE 的性能取决于 ragged grouped GEMM、dispatch/combine 和训练内存计划共同配合。
- 已核对正文而非 AI-generated summary：TE、NCCL EP、host offload、XLA 配置均有工程入口。文中 GB200 的 10.4x 相对未优化 baseline，不能解读为比成熟 Megatron 快 10.4x；不把不同硬件的 scale-out 结果混为同一测量。
- Next：先逐项归因 kernel time、通信 bytes 和 overlap；只迁移配置背后的机制，不复制整个 YAML 到 AReaL。
- 主题：[MoE](../topics/moe.md)、[Transformer Engine](../topics/transformer_engine.md)、[NCCL](../topics/nccl.md)。

## Observed / Rejected Candidates

以下 10 行为 Observed 计数；Watch 的同源引用不另外计数。均为 Observe，未进入阅读队列。

| Source ID / 来源与类型 | Impact / Reason / 一句话价值 | 关联主题与下一步 |
|---|---|---|
| `arxiv:2609.16491`：[PipeSwift](https://arxiv.org/abs/2609.16491)，paper | ★★★★☆；把 JCT 放在 token SLO 前；目前核验到 deterministic replay，不能直接推出在线 RL time-to-quality | Agentic RL / PP；核验真实 environment interleaving 与 tail latency |
| `arxiv:2609.14636`：[STRIDE](https://arxiv.org/abs/2609.14636)，paper | ★★★☆☆；early stop + prefix buffer 缩短多轮 OPD；需先确认 restart 能否恢复 environment 状态 | MOPD；看 state snapshot、采样偏差及可运行实现 |
| `official:nvidia:nvlink6-resiliency`：[NVLink 6 resiliency](https://developer.nvidia.com/blog/how-nvidia-nvlink-6-delivers-multi-layer-resiliency-for-ai-factories/)，blog | ★★★★☆；硬件链路恢复与进程恢复分层；cuda-checkpoint 的 NCCL 支持明确仍为 prototype | Fault tolerance；待 GA/实现核验，避免把 inference shadow recovery 当训练恢复 |
| `github:vllm-project/vllm:release:v0.29.0`：[v0.29.0](https://github.com/vllm-project/vllm/releases/tag/v0.29.0)，release | ★★★★☆；MRV2 默认化与 KV sizing 改变 rollout backend；release 保留例外路径 | Rollout；固定版本做 logprob、sleep/wake、refit 回归 |
| `official:openai:scaling-storage-one-billion-users-part-one`：[Habitat](https://openai.com/index/scaling-storage-one-billion-users-part-one/)，blog | ★★★☆☆；backpressure/connection pool 有控制面借鉴价值，主体是产品在线存储 | Distributed systems；只在 rollout metadata 服务出现瓶颈时精读 |
| `official:anthropic:cyber-incidents-2026-09-09`：[研究目录](https://www.anthropic.com/research)，research post | ★★★☆☆；agent 权限边界相关，未核验到训练系统新机制 | Agent environment；需读原文后再决定是否提升 |
| `github:huggingface/peft:release:v0.21.0`：[v0.21.0](https://github.com/huggingface/peft/releases/tag/v0.21.0)，release | ★★☆☆☆；ShadowPEFT 等新方法，当前未形成训练 infra 优先级 | PEFT/FSDP；等待状态保存与分布式兼容证据 |
| `github:huggingface/kernels:release:v0.17.0`：[v0.17.0](https://github.com/huggingface/kernels/releases/tag/v0.17.0)，release | ★★★☆☆；kernel 加载时检查 device architecture，减少部署后才暴露不兼容 | Kernel；核对硬件/编译产物能力矩阵，兼查 v0.16.2 版本检查 |
| `github:verl-project/verl:pr:7764`：[Mooncake completion slots](https://github.com/verl-project/verl/pull/7764)，PR | ★★★☆☆；涉及跨 weight version 的完成槽清理，目前仅核对提交目录 | Weight sync；读 patch 再判断是否需改变 AReaL completion ownership |
| `github:OpenRLHF/OpenRLHF:window:2026-09-07..16`：[官方 feed](https://github.com/OpenRLHF/OpenRLHF/commits/main.atom)，framework | ★★★☆☆；可见 truncation、KL gradient、reward FP32、FlashREINFORCE 变化；尚未逐项核验 patch | Rollout/training correctness；先查 KL-as-loss 路径与截断语义 |

## OpenAI / Anthropic / NVIDIA / DeepSeek Watch

| Vendor | Triage | 本轮结论 |
|---|---|---|
| [OpenAI News](https://openai.com/news/) | Observed / Rejected | Habitat 进入观察；GPT-6 Astra 产品发布正文未形成本轮训练 infra 证据，Rejected / Ignore。9 月 7 日 automated research 信号不重复收录。 |
| [Anthropic Research](https://www.anthropic.com/research) | Observed / Rejected | 9 月 9 日安全事件评估仅观察；9 月 10 日能力评测不是当前 infra 主线。没有据此声称 Anthropic 没有新发布。 |
| [NVIDIA Blog](https://developer.nvidia.com/blog/) / NeMo / Megatron | Accepted / Observed | Dropless MoE、NeMo correctness/OPD Accepted；NVLink resiliency 观察；框架历史缺口须补扫。 |
| [DeepSeek API changelog](https://api-docs.deepseek.com/updates/) + [官方 HF org](https://huggingface.co/deepseek-ai) | Accepted | 两个渠道均看到 9 月 10 日 V4.1-Flash；model card 与技术报告入口相互对应，未只根据 API 模型名判断。 |

## Hugging Face Watch

| 来源 | Triage / Decision | 证据与下一步 |
|---|---|---|
| [Blog](https://huggingface.co/blog) | Observed / Observe；官方目录覆盖不完整 | 返回页面主要显示 Community Articles。Reef、sandbox-per-rollout 等社区标题不能当成 HF 官方团队结论；下轮补查官方文章索引与作者归属。 |
| [TRL releases](https://github.com/huggingface/trl/releases) | Accepted / Read | v1.13.0，聚焦 chunked loss；不重复上轮 1M-token recipe。 |
| [Transformers releases](https://github.com/huggingface/transformers/releases) | Observed / Observe | v5.17.0 的 HYV4 支持明确不执行 MTP layers；“能加载”不代表 speculative 路径完整。 |
| [Accelerate releases](https://github.com/huggingface/accelerate/releases) | Accepted / Read | v1.15.0 FSDP2 activation 与保存正确性，与 TRL 合为一次训练栈验收。 |
| [PEFT releases](https://github.com/huggingface/peft/releases) | Observed / Observe | v0.21.0；方法更新不自动成为 Infra Accepted。 |
| [Kernels releases](https://github.com/huggingface/kernels/releases) | Observed / Observe | v0.17.0 architecture checks 与 v0.16.2 compatibility check，先落实环境验收。 |

## RL Framework Watch

| Framework / 官方来源 | 子系统与工程维度 | Triage / AReaL 可迁移性 |
|---|---|---|
| [AReaL Atom](https://github.com/areal-project/AReaL/commits/main.atom) | rollout / trajectory / checkpoint：group 统计、ragged transport、发布原子性 | Accepted；两条固定 SHA 已读 patch/tests。AWEX、process reward、多流 rollout 先保留 follow-up；不把普通 commit 全部升格。 |
| [verl](https://github.com/verl-project/verl/commits/main.atom) | weight sync / training / checkpoint：refit GC、KV resume rank sync、router replay 的 revert/reland、inflight reissue | Observed；读取目录不等于验证所有实现；向 AReaL 迁移前先验版本 gate 与 collective 顺序。 |
| [slime](https://github.com/THUDM/slime/commits/main.atom) | rollout / inference backend | Not found；当前 feed 最新仍为 9 月 3 日，streaming external rollout 已在旧窗口，不重复。 |
| [ROLL](https://github.com/alibaba/ROLL/commits/main.atom) | scheduler / multimodal data | Not found；当前 feed 最新为 8 月 27 日，本窗口未见新 material change。 |
| [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF/commits/main.atom) | rollout / training：truncation、KL、reward precision | Observed；可见 9 月 10–14 日变化，历史缺口未全部覆盖；对 AReaL 有 correctness 对照价值，尚不推荐移植。 |
| [NeMo RL](https://github.com/NVIDIA-NeMo/RL/commits/main.atom) | scheduler / weight sync / recovery / training / data plane | Accepted；重新准入与 full-vocab OPD 固定 patch 已核验；Mooncake GDR、M-to-N refit 先留 follow-up，下一轮补齐历史分页。 |

相邻 runtime：[vLLM v0.29.0](https://github.com/vllm-project/vllm/releases/tag/v0.29.0) Observed；[SGLang releases](https://github.com/sgl-project/sglang/releases) 最新可见 v0.5.19 发布于 9 月 5 日，不作新 release 收录，不能据此排除新 PR；[Megatron Atom](https://github.com/NVIDIA/Megatron-LM/commits/main.atom) 可见 MTP CP-boundary、MFSDP metadata 和 GTP checkpoint 变化，但 feed 未覆盖全部窗口，标记 Not verifiable for full-window coverage。

## 学习推进与交接

- 本周 P0 仍保持最多三项，AReaL / HybridFlow 的 READING 状态不擅自改成已完成。
- 新增 [P1：RL 状态边界](../reading_queue/P1.md#rl-state-boundaries-reading)，把上面第 2–4 项作为一个阅读组合；这组比再加多篇算法论文更能补齐当前生产判断。
- 先用 30 分钟读 AReaL incomplete-group tests，再用 30 分钟读 LATEST publication，最后用 45 分钟读 NeMo stale → refit → serving tests。输出一页 invariant 表；接着才选 DeepSeek 或 draft co-training 精读。
- [9 月接手记录](../learning_log/2026/2026-09.md) 记录真实已核验内容和未做事项；[验证方案](../experiments/rl_state_boundaries.md) 当前为计划，不假造实验结果。
- 下次：先从旧游标补扫上述框架缺口与 HF 官方目录，再从本次终点增量；对固定 Source ID 去重。8 月月报已存在，不重复创建，9 月月报待自然月结束。
- 导航：[Master Reading List](../MASTER_READING_LIST.md)、[Knowledge Graph](../KNOWLEDGE_GRAPH.md)。本次没有配置定时任务。

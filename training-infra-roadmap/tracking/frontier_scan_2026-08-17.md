# Frontier Scan, 2026-08-17

- Previous confirmed cursor：[2026-08-14](frontier_scan_2026-08-14.md)
- Window：2026-08-14 17:39:50 ~ 2026-08-17 09:28:33
- Timezone：Asia/Shanghai
- Generated at：2026-08-17 09:28:33
- Report type：full rescan / supersedes the 2026-08-16 intermediate draft
- Sources scanned：arXiv cs.DC / cs.LG / cs.AI / cs.CL / cs.AR / cs.PF / cs.NI recent pages；OpenAI / Anthropic / NVIDIA / DeepSeek official sources；Hugging Face Blog / TRL / Transformers / Accelerate / PEFT / Kernels；AReaL / verl / slime / ROLL / OpenRLHF / NeMo RL；vLLM / SGLang / Megatron-LM / Molt official changes
- Scan completeness：从 2026-08-14 的最后确认游标完整回扫到本次实际结束时刻。arXiv API 遇到 rate limit，已回退各分类 official recent pages；页面最新公告批次仍为 8 月 14 日，没有遗漏新的周末/周一批次。

> 说明：8 月 16 日的中间扫描没有作为独立游标保留。本报告重新覆盖同一窗口并继续扫描到 8 月 17 日 09:28:33，避免后续出现“中间稿 + 正式稿”重复记账。

## 本次核心判断

这次复扫没有发现新的高质量论文，但发现了比普通论文增量更重要的工业信号：**Agentic RL 正在把 staleness、rollout failure 和 generation engine capacity 变成可配置、可推导、可观测的运行时对象；推理侧则继续沿着 KV state disaggregation 和低精度 speculative decoding 演进。**

1. **Megatron-LM 开始自动推导 RL generation lag。** 它把 inference DP、engine request capacity、GRPO group size 与 training batch size 放进同一个容量模型，并明确不同提交/消费粒度如何交换 GPU utilization 与 staleness。这比笼统地说“异步更快”前进了一步。
2. **NeMo RL 把 rollout failure containment 做成了独立控制面。** infra failure、deterministic data failure、deadline、row resend、watchdog 与 skip/abort 不再混成一个 retry 参数。
3. **DeepSeek-V4-Pro-0813 是本窗口最重要的厂商信号。** 正式 model card 同时给出 1.7T 模型、DSpark speculative decoding、FP4/indexer cache、vLLM/SGLang 部署参数和最高 384K 输出建议，说明长 reasoning 与大规模 MoE serving 已作为一个整体工程问题交付。
4. **Megatron-LM 的 KV handoff 已不是概念 PR。** 实现包含 NIXL transfer、prefill pinning、decode import、prefix reuse、容量队列、跨模型并行完成跟踪和失败隔离；它暴露的限制也很重要：hybrid recurrent state 尚不能只靠 KV handoff 解决。
5. **“没有新 arXiv”不等于没有前沿进展。** 本窗口真正值得读的内容主要来自 core-vendor model card 和 framework default-branch implementation，而不是新的 paper batch。

## Accepted Frontier Signals

### DeepSeek-V4-Pro-0813：长 Reasoning、FP4 与 Speculative Decoding 的工业交付

- Signal ID：2026-08-17-001
- Source ID：hf:deepseek-ai/DeepSeek-V4-Pro-0813
- First seen：2026-08-17 09:28:33
- Original release：2026-08-13，`boundary late-discovered`
- Scan window：2026-08-14 17:39:50 ~ 2026-08-17 09:28:33
- Focus Match：P0 Focus
- 来源：DeepSeek official Hugging Face model card
- 类型：official model release / industrial inference evidence / agentic model
- 链接：https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813
- Primary-source check：模型规模、reasoning effort、DSpark、vLLM/SGLang recipes、精度/kernel 参数与 max output length 均对齐官方 model card；benchmark 明确标为 vendor-reported
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：它不是只有权重与榜单的模型发布，而是公开了大规模 MoE 在 FP4、sparse indexer、speculative decoding、chunked prefill 和超长 reasoning 场景下的可运行 serving 配方。
- Status：NEW
- 建议动作：先读 model card 的 deployment recipes，再分别追 vLLM `deep_gemm_mega_moe` / FP4 indexer cache 与 SGLang DSPARK / chunked prefill 的实现边界
- 关联主题：[Long-context Training](../topics/long_context_training.md), [FP8 / Low Precision](../topics/fp8.md), [MoE](../topics/moe.md), [Agentic RL](../topics/agentic_rl.md)

正式版本为 1.7T 参数模型，基于 V4-Pro Preview 结构并附加 DSpark speculative decoding。官方将 `reasoning_effort` 暴露为 low/high/max，并建议 high/max 场景允许最高 384K output；这意味着 serving 系统必须同时处理长 decode、KV/state 容量和 speculative path 的稳定性，而不能只优化短请求吞吐。

vLLM recipe 使用 4 个 GB300 节点、DP4、expert parallel、`deep_gemm_mega_moe`、FP4 indexer cache 与 7 个 speculative tokens；SGLang recipe 给出 TP4、`flashinfer_mxfp4`、DSPARK、4096 chunked prefill 与 SWA full-token ratio。这里最值得学习的是“模型能力如何被 runtime 参数化”，不是照抄某组硬件配置。

官方报告 TerminalBench、DeepSWE 等 agent/coding benchmark 相比 preview 明显提升，但这些数字仍是厂商自报结果，其中部分 benchmark 为内部版本。工程上应把它们视为“值得进一步复核的工业证据”，而不是直接当成跨框架可复现结论。

### Megatron-LM Autotunes RL Generation Lag

- Signal ID：2026-08-17-002
- Source ID：github:NVIDIA/Megatron-LM#4127
- First seen：2026-08-17 09:28:33
- Scan window：2026-08-14 17:39:50 ~ 2026-08-17 09:28:33
- Focus Match：P0 Focus
- 来源：Megatron-LM default-branch commit derived from PR #4127
- 类型：framework implementation / asynchronous RL / staleness control
- 链接：https://github.com/NVIDIA/Megatron-LM/pull/4127
- Primary-source check：default-branch commit、lag 定义、capacity 公式、submission/consumption granularity、GRPO group atomicity、tail factor 与 oversubscription warning 已对齐官方 patch；PR 页面状态不作为“已合并”依据
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：它把用户最关心的“异步 rollout 能提前多少、何时会过时、为什么长尾导致 bubble”从经验配置变成 engine-capacity-aware 的可推导参数。
- Status：NEW
- 建议动作：画出 B/B、G/G、G/B、R/G、R/B 五种模式的 timeline，并对照 AReaL 当前 rollout queue、policy version 和 GRPO group consumption 语义
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [Rollout Latency](../playbooks/rollout_latency.md), [MOPD](../topics/mopd.md)

实现定义 collection lag `L`：当前消费的 rollout 是在多少个 training step 之前生成的。若用户未显式设置 `--rl-generation-lag` 或 inflight request 上限，系统根据 inference DP、engine `max_requests`、GRPO group size 与每步 prompt groups 自动估计可用容量。官方 patch 给出的上界形态是：

```text
max_effective_lag = DP * engine.max_requests / (G * P) - 1
```

其中 `G` 是每个 prompt 的 GRPO rollout 数，`P` 是每个 training batch 的 prompt group 数。这个式子不是训练正确性的保证，而是“生成端最多能提前积累多少工作”的容量边界。

submission 可按 rollout、group 或 batch，consumption 可按 group 或 batch。GRPO 不能逐 rollout 消费，因为同组 reward/advantage 未齐时无法保持 group-level 语义。B/B 的 staleness 可预测但更容易出现 idle；R/G 可以更充分地填满 engine，但会提高 policy staleness，需要 importance sampling 等算法机制配合。工程上真正要调的是粒度、容量、长尾与 staleness budget 的联合边界。

### NeMo RL Contains Rollout Failures in the SingleController Path

- Signal ID：2026-08-17-003
- Source ID：github:NVIDIA-NeMo/RL#3589
- First seen：2026-08-16 22:52:28
- Scan window：2026-08-14 17:39:50 ~ 2026-08-17 09:28:33
- Focus Match：P0 Focus
- 来源：NeMo RL merged PR / default branch
- 类型：framework implementation / rollout failure containment / watchdog
- 链接：https://github.com/NVIDIA-NeMo/RL/pull/3589
- Primary-source check：PR、commit、配置字段、retry/deadline/watchdog 逻辑与测试已对齐官方 diff；未把未披露的生产收益写成事实
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：它解决的不是某次 rollout exception，而是长 horizon 环境中少数坏 prompt、环境进程卡死、部分 row 丢失或 backpressure 无进展如何被隔离、分类和有界恢复。
- Status：NEW
- 建议动作：精读 failure budget、row-level resend、progress watchdog 和 permit release；对照 AReaL rollout worker 的 timeout/retry/skip 语义
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [Fault Tolerance](../topics/fault_tolerance.md), [Rollout Latency](../playbooks/rollout_latency.md)

实现分别为基础设施失败与 deterministic prompt/data failure 设置重试预算；native generation、environment step 与 whole rollout 也有独立 deadline。系统优先重发缺失 row，再决定是否重试整个 group，避免一个局部失败放大全局重算。

watchdog 以 `(committed groups, train steps)` 作为进度信号，同时观察 inflight 与 idle counter，因此能够发现“没有异常但系统已经不再前进”的 zero-inflight backpressure deadlock。对 AReaL 最值得迁移的是 failure-semantics matrix：谁检测、重试是否换 worker、哪些 permit/state 必须释放、何时 skip、何时 abort，以及 skip 后 batch 语义如何闭环。

### Megatron-LM Adds Disaggregated KV State Handoff

- Signal ID：2026-08-17-004
- Source ID：github:NVIDIA/Megatron-LM#6222
- First seen：2026-08-16 22:52:28（先记为 Observe，本次完成实现核验后升级）
- Scan window：2026-08-14 17:39:50 ~ 2026-08-17 09:28:33
- Focus Match：P0 Focus
- 来源：Megatron-LM default-branch implementation
- 类型：framework implementation / disaggregated inference / KV state transfer
- 链接：https://github.com/NVIDIA/Megatron-LM/pull/6222
- Primary-source check：NIXL backend、prefill pin/release、decode import、prefix reuse、capacity queue、completion tracker、failure quarantine、reset/drain、TP/PP metadata 与 unsupported cases 已逐项对齐官方 patch/tests；官方未提供端到端 benchmark
- 影响等级：★★★★☆
- Decision：Read
- Reason：这是 training stack 向 disaggregated inference 延伸的完整实现证据，特别适合判断 KV handoff 的 state ownership、生命周期与 failure boundary，而不是只看 P/D 分离概念图。
- Status：NEW
- 建议动作：沿 request lifecycle 阅读 `prefill complete -> pin -> send -> decode import -> complete -> release`，再和 vLLM/SGLang 的 KV transfer connector 对照
- 关联主题：[Distributed Training](../topics/distributed_training.md), [Long-context Training](../topics/long_context_training.md), [Fault Tolerance](../topics/fault_tolerance.md)

prefill 侧会在完成后 pin 住 KV blocks，直到 coordinator 发出 `RELEASE_KV`；decode 侧分配或导入本地 blocks，并可利用已有 prefix cache，减少源端实际需要传输的范围。handoff 按目标容量进入 FIFO 队列，NIXL 是默认 transfer backend。

完成跟踪不只看单 rank：CPU ZMQ tracker 会聚合 model-parallel peers，任一 peer 失败即可让请求快速失败；reset/drain 会清理状态，并隔离可能残留不完整传输的 destination。实现还明确拒绝 hybrid recurrent models，因为只有 KV state 不足以恢复 recurrent state。这类“明确说不能支持什么”的边界，比一张 throughput 图更有工程价值。

### Megatron-LM Caches FullyParallel Checkpoint Shard Distribution

- Signal ID：2026-08-17-005
- Source ID：github:NVIDIA/Megatron-LM#5553
- First seen：2026-08-16 22:52:28
- Scan window：2026-08-14 17:39:50 ~ 2026-08-17 09:28:33
- Focus Match：P0 Focus
- 来源：Megatron-LM merged PR / default branch
- 类型：framework implementation / distributed checkpoint / metadata scaling
- 链接：https://github.com/NVIDIA/Megatron-LM/pull/5553
- Primary-source check：配置接口、cache key、atomic write、read/write distribution reuse 与 caveat 已对齐官方 diff；官方未提供稳定 benchmark，因此不声称具体加速比
- 影响等级：★★★★☆
- Decision：Read
- Reason：当 state dict、parallel layout 与 world size 固定时，FullyParallel shard distribution 是确定的，重复 world-wide metadata all-gather 属于可消除的控制面开销。
- Status：NEW
- 建议动作：把 checkpoint profile 拆成 `metadata gather -> distribution plan -> tensor IO`，确认当前瓶颈是否真的在 planner/control plane
- 关联主题：[Checkpointing](../topics/checkpointing.md), [Fault Tolerance](../topics/fault_tolerance.md), [Distributed Training](../topics/distributed_training.md)

实现按 process group 缓存 FullyParallel distribution，磁盘文件以最小 global rank 标识，并采用临时文件加 `os.replace`。同一次 gather 可同时生成 save/load distribution；READ 路径可以跳过后续 collective。

它没有优化参数和 optimizer state 的真实写入带宽，也不会替用户校验 elastic restart 后的 layout 是否仍兼容。若 world size 或 sharding layout 改变，缓存必须显式失效。因此它是 checkpoint control-plane optimization，不是“checkpoint 整体加速”的无条件结论。

## Observed / Rejected Candidates

| 材料 | Source ID | Focus Match | Decision | 原因 |
|---|---|---|---|---|
| Anthropic: Patterns and problems in emerging multiagent systems | web:anthropic/multiagent-systems | P1 | Read | 45-agent VM/forum 实验暴露协作收益、conformity、资源轮询风暴和目标冲突，能补 agent harness、evaluation 与 resource isolation 判断；不是直接的 RL runtime 实现，保留 Observe。 |
| NeMo RL exposes training CUDA Graph configuration | github:NVIDIA-NeMo/RL#3483 | P1 | Observe | CUDA Graph 可降低稳定 shape 下的 launch overhead，但 RL 动态 batch/sequence 的 capture/fallback 边界尚需单独验证。 |
| verl stores routed expert IDs as `torch.int16` | github:verl-project/verl#7407 | P1 | Observe | 缩小 MoE routing metadata 的显存与传输体积，属于清晰局部优化；缺少对 step time 或 trajectory path 的独立证据。 |
| slime adds ROCm INT4 QAT kernel support | github:THUDM/slime#2274 | P1 | Observe | 增强异构硬件与低比特训练覆盖，当前主要是 portability / feature enablement。 |
| vLLM stabilizes DeepSeek-V4 backend | github:vllm-project/vllm@deepseek-v4-followups | P1 | Observe | sparse MLA、MTP 与 partial-prefill 连续修复说明 backend 正处于 correctness/performance 收敛期；没有把一组 routine follow-up commits 拆成多个信号。 |
| SGLang GLM-5 / DSPARK / Kimi follow-ups | github:sgl-project/sglang@2026-08-16 | P1 | Observe | 包含 DSA indexer fallback、DSPARK logprobs、prefill-breakable CUDA Graph 与 EPLB reporting；有参考价值但还不足以改变整体 runtime 判断。 |
| Megatron HFSDP deferred DP-outer gradient reduction | github:NVIDIA/Megatron-LM@2026-08-16-hfsdp | P1 | Observe | 涉及 gradient communication placement，值得后续观察 benchmark 与适用并行布局；本次不单独升级。 |

Anthropic 文章原始时间为 2026-08-13，上一份 08-14 scan 错误写成 `Not found`，本次按 `boundary late-discovered` 纠正。文章对 45 个 agent、每个独占 VM 的漏洞发现 swarm 进行实验：协作提高了发现覆盖，但在限制到相同核心目录后 token efficiency 未必优于独立运行；另一个资源实验出现 agent 高频轮询造成 240 万次 job request、最终仅 117 次 accepted 的系统性拥塞。它提供的是多 agent 系统设计证据，而不是“agent 越多越快”的宣传结论。

## OpenAI / Anthropic / NVIDIA / DeepSeek Watch

| Vendor | Sources checked | Decision | 结果 |
|---|---|---|---|
| OpenAI | official research / engineering / releases | Not found | 本窗口未发现可核验、且包含 Training/RL/Inference Infra 新机制的一手材料。 |
| Anthropic | official research | **Observed / Read** | 补入 8 月 13 日 [Patterns and problems in emerging multiagent systems](https://www.anthropic.com/research/multiagent-systems)，用于观察多 agent 协作、资源争用与 evaluation harness；属于上一轮漏项，不伪装成本日新发布。 |
| NVIDIA | Megatron-LM / NeMo RL official implementation | **Accepted** | RL lag autotuning、rollout failure containment、disaggregated KV handoff 与 checkpoint distribution cache 均有代码/测试证据，分别覆盖 scheduler、recovery、inference state 与 checkpoint control plane。 |
| DeepSeek | official Hugging Face organization / model card；API changelog endpoint | **Accepted / Deep Dive** | 补入 [DeepSeek-V4-Pro-0813](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813)；model card 已核验，API changelog 端点本次未能稳定读取，因此不声称无其他 API 变化。 |

## Hugging Face Watch

| Sources checked | Decision | 结果 |
|---|---|---|
| Official/vendor model cards | **Accepted** | DeepSeek-V4-Pro-0813 official model card 提供了本窗口最完整的工业部署信号。 |
| Hugging Face Blog | Rejected / routine | 最新可见文章没有出现足以改变 Agentic RL、distributed training、long context 或 inference backend 判断的新机制；8 月 14 日 open-model roundup 不以综述热度替代工程证据。 |
| TRL / Transformers / Accelerate / PEFT / Kernels | Routine only | 未发现需要升级为 frontier signal 的 architecture/performance/correctness release。 |

## RL Framework Watch

| Framework | Release / PR | 子系统 | 核心变化 | 证据 | 对 AReaL 的参考 | Decision |
|---|---|---|---|---|---|---|
| AReaL | default branch / release | - | 上一游标后未发现改变架构、性能、正确性或生产行为的重大变化 | official commit feed | 无新增可迁移项 | Not found |
| verl | [#7407](https://github.com/verl-project/verl/pull/7407) | training / MoE metadata | routed expert IDs 使用 `torch.int16`；同期有 SGLang LoRA correctness fix | merged PR / default branch | 检查 route metadata dtype、跨进程传输与 trajectory storage 成本 | Observe |
| slime | [#2274](https://github.com/THUDM/slime/pull/2274) | training / kernel | ROCm INT4 QAT kernel；同期 converter/debug-data 修复 | merged PR / default branch | 参考异构硬件与低比特训练覆盖，不改变 scheduler/rollout 设计 | Observe |
| ROLL | default branch / release | - | 上一游标后未发现重大变化 | official commit feed | 无新增可迁移项 | Not found |
| OpenRLHF | default branch / release | - | 上一游标后未发现重大变化 | official commit feed | 无新增可迁移项 | Not found |
| NeMo RL | [#3589](https://github.com/NVIDIA-NeMo/RL/pull/3589) / [#3483](https://github.com/NVIDIA-NeMo/RL/pull/3483) | rollout recovery / training runtime | failure budgets、deadlines、row resend、watchdog；另暴露 CUDA Graph training config | merged PR、diff、tests | 建立 failure taxonomy、bounded retry、no-progress watchdog；评估动态 shape 下 CUDA Graph 边界 | **Accepted / Deep Dive** + Observe |

## Adjacent Runtime Watch

| Runtime | 变化 | Decision | 工程判断 |
|---|---|---|---|
| Megatron-LM | [#4127](https://github.com/NVIDIA/Megatron-LM/pull/4127) RL generation lag、[#6222](https://github.com/NVIDIA/Megatron-LM/pull/6222) KV handoff、[#5553](https://github.com/NVIDIA/Megatron-LM/pull/5553) checkpoint distribution cache | **Accepted** | 三项分别对应 RL scheduler、disaggregated inference state lifecycle 与 checkpoint control plane，均已核到实现边界。 |
| vLLM | DeepSeek-V4 sparse MLA / MTP / partial-prefill 与 Kimi backend follow-ups | Observe | 新模型 backend 仍在 correctness 与 performance 共同收敛期；不把 routine fixes 包装成多个 frontier signals。 |
| SGLang | GLM-5 DSA fallback、DSPARK logprobs、Kimi prefill-breakable graph、EPLB reporting | Observe | 方向与 DeepSeek/Kimi serving 相关，但本窗口缺少独立端到端证据。 |
| Molt | default-branch feed endpoint unavailable | Not verifiable | 未发现可由其他一手来源确认的新重大变化；下一轮继续回看，不以搜索摘要代替代码证据。 |

## Reading Queue 判断

- [ ] **今天只读一个：Megatron-LM #4127。** 目标不是记参数名，而是回答 `engine capacity -> inflight work -> generation lag -> policy staleness -> GPU bubble` 的因果链。
- [ ] **第二优先：DeepSeek-V4-Pro-0813 model card。** 只看 serving recipes 与 stated limitations，先建立 1.7T / FP4 / DSpark / 384K output 的系统约束图。
- [ ] NeMo RL #3589 与 Megatron-LM #6222 保留为下一轮工程精读，不自动把本次 5 条 Accepted 全塞进 P0。

## 去重与纠错记录

- 8 月 16 日中间扫描已被本报告替代，不再单独占据 scan cursor。
- 保留原 First seen：NeMo RL #3589、Megatron-LM #5553 与 #6222 最早在 8 月 16 日中间扫描被发现。
- `hf:deepseek-ai/DeepSeek-V4-Pro-0813` 与 `web:anthropic/multiagent-systems` 均为 8 月 13 日原始材料，因上一轮 Watch 漏检而按 `boundary late-discovered` 补入。
- Megatron-LM #6222 从 Observe 升级为 Accepted，因为本次已取得并核验完整 patch/tests，不再保留“无法核实现细节”的旧结论。
- 本窗口没有新的 arXiv announcement batch，不重复收录 8 月 14 日已记录的 TideRL、MISA-T、RoutePack 与 vToken。

## 扫描完整性

- arXiv：API 返回 rate limit；已逐项回退 cs.DC / cs.LG / cs.AI / cs.CL 等 official recent pages，最新公告仍为 8 月 14 日，本窗口没有新的周末/周一 paper batch。
- Core vendors：OpenAI、Anthropic、NVIDIA、DeepSeek 均显式检查。DeepSeek API changelog 未稳定读取，但 official Hugging Face model card 已核验；该盲区已写明。
- Frameworks：六个 RL 框架与 Megatron-LM、vLLM、SGLang 的 official default-branch activity 已检查；Molt feed endpoint 本次不可用。
- 证据边界：vendor benchmark 明确按 vendor-reported 处理；没有为 checkpoint cache、KV handoff 或 RL lag autotuning 编造官方未披露的 speedup。
- 下一游标：`2026-08-17 09:28:33`。本时刻之后出现的材料留给下一次扫描。

## 下一步动作

- [x] 更新 [Scan Log](scan_log.md) 与 [Tracking README](README.md)。
- [ ] 精读 Megatron-LM #4127，若改变对 AReaL staleness/queue 的判断，再更新 [Agentic RL](../topics/agentic_rl.md) 或 AReaL 工程分析。
- [ ] 阅读 DeepSeek-V4-Pro-0813 serving recipes，区分模型结构约束、厂商推荐配置与可迁移 runtime 机制。
- [ ] 后续将 NeMo RL #3589 作为 rollout failure semantics 案例，将 #5553 作为 checkpoint metadata scaling 案例沉淀；不在未读完时提前扩写 topic。

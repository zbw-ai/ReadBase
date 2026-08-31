# Frontier Scan, 2026-08-26

- Previous scan：[2026-08-24](frontier_scan_2026-08-24.md)
- Window：2026-08-24 09:32:12 ~ 2026-08-26 10:22:05
- Timezone：Asia/Shanghai
- Generated at：2026-08-26 10:22:05
- Report type：flexible frontier scan
- Sources scanned：arXiv cs.AI / cs.LG / cs.CL / cs.DC / cs.PF / cs.AR / stat.ML；OpenAI / Anthropic / NVIDIA / DeepSeek official sources；Hugging Face Blog / TRL / Transformers / Accelerate / PEFT / Kernels；AReaL / verl / slime / ROLL / OpenRLHF / NeMo RL；Megatron-LM / vLLM / SGLang official default-branch changes
- Scan completeness：arXiv 以 official recent pages 覆盖 8 月 24 日游标后的公告批次，并对 Accepted paper 逐条核验 title、authors、submission date 与 abstract claims。厂商文章以官方页面为准；GitHub 以 merged PR、default-branch commit、测试和 benchmark 为准。扫描截止时刻冻结在检索开始前，晚于该时刻的变更留给下一次。

## 本次核心判断

本次只有三条足够强的 Accepted，但它们指向同一个变化：**AI systems 的竞争正在从“堆更多峰值 FLOPS 和链路带宽”转向“让数据放对位置、让同步等待可解释、让软件直接编排数据移动”。**

OpenAI Jalapeño 与 Microsoft Maia 200 都把 locality、memory、network 和 software-controlled dataflow 放到芯片设计中心；Synchronization Tax 则说明即使在 8-GPU scale-up 域内，collective 也可能有超过一半时间浪费在 rank 到达不齐，而主要来源不是网络本身，是跨 rank 的 GEMM runtime variation。对训练与 RL Infra 的含义很直接：未来只看 kernel 均值、理论带宽和 MFU 不够，必须把 per-rank timeline、arrival skew、state placement 和 data-movement policy 作为一等可观测对象。

## Accepted Frontier Signals

### OpenAI Jalapeño：Inference Chip 的目标从峰值吞吐转向整条 Pareto Frontier

- Signal ID：2026-08-26-001
- Source ID：blog:openai/jalapeno-first-results-2026-08-25
- First seen：2026-08-26 10:22:05
- 发布时间：2026-08-25
- Scan window：2026-08-24 09:32:12 ~ 2026-08-26 10:22:05
- Focus Match：P0 Focus
- 来源：OpenAI official engineering post
- 类型：industrial report / inference accelerator / hardware-software co-design
- 链接：https://openai.com/index/jalapeno-first-results/
- Primary-source check：发布时间、首款自研 inference chip、InferenceX workload、700W rated / tested sustained power、三组模型的 throughput-per-watt 与 latency 数字、deployment/qualification 状态均已对齐官方文章
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：这不是单一 kernel 或模型优化，而是 OpenAI 首次用公开 workload 展示自研 inference silicon、memory、network、serving software 与模型协同设计的路线。
- Status：NEW
- 建议动作：先读 benchmark methodology、prefill/decode 资源设计与 memory locality，再把 package TDP 归一化和真实整机能耗的证据边界分开记录
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [Long-context Training](../topics/long_context_training.md), [Distributed Training](../topics/distributed_training.md)

官方在 GPT-OSS 120B、DeepSeek R1 670B 和 Kimi K2.5 1T 上报告峰值 throughput-per-watt 提升 `1.5-1.9x`、端到端 latency 降低 `1.7-3.6x`；Jalapeño rated power 为 700W，测试 workload 的 sustained power 不超过 550W。文章强调把 KV cache 和模型状态尽量保留在本地，并让同一类 accelerator 同时服务 prefill/decode，而不是为单一 operating point 做专用分区。

这些数字属于 OpenAI 自报的 InferenceX 测试，功耗比较使用公开 package TDP 归一化，且芯片仍在 production qualification、计划年底开始部署。工程上应把它视为强工业方向证据，不应直接外推为任意模型和整机系统的普遍优势。

### Maia 200：从 Thread-Centric 转向 Software-Defined Data Movement

- Signal ID：2026-08-26-002
- Source ID：arxiv:2608.24664
- First seen：2026-08-26 10:22:05
- 发布时间：2026-08-25 23:05:40，Asia/Shanghai
- Scan window：2026-08-24 09:32:12 ~ 2026-08-26 10:22:05
- Focus Match：P0 Focus
- 来源：arXiv primary page / industrial system report
- 类型：paper / AI accelerator / software-defined dataflow
- 链接：https://arxiv.org/abs/2608.24664
- Primary-source check：title、17 位作者、v1 timestamp、SDLA 定义、FP4/FP8 throughput、TDP 与 HBM bandwidth 已对齐 arXiv metadata/abstract
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：它给出了一条比“增加 Tensor Core 峰值”更重要的架构路线：由软件显式编排 specialized memory 与 data-movement engine，让算子实现围绕数据流而不是线程抽象组织。
- Status：NEW
- 建议动作：精读 memory hierarchy、dataflow programming model、collective/network interface 和 workload mapping，判断哪些思想能迁移到 compiler/runtime 层
- 关联主题：[Distributed Training](../topics/distributed_training.md), [Transformer Engine](../topics/transformer_engine.md), [FP8](../topics/fp8.md)

Maia 200 报告 `10,145 TFLOP/s FP4`、`5,072 TFLOP/s FP8`、750W TDP 与 7 TB/s HBM bandwidth，并把架构定义为 Software Defined Locally Accessed Dataflow Architecture。真正值得读的不是峰值数字，而是它如何让软件控制 memory/data movement engine，以及这种控制面如何避免通用 thread-centric execution 在大模型 inference 上产生的数据搬运浪费。

Maia 200 硬件已于 2026 年 1 月由 Microsoft 公布；本次新信号是 8 月 25 日发布的体系结构论文，而不是一次新的芯片发布。论文的增量价值在于把此前的产品规格进一步抽象成 SDLA 和 data-movement-centric architecture。

### Synchronization Tax：Scale-Up 域内的等待可能比通信本身更贵

- Signal ID：2026-08-26-003
- Source ID：arxiv:2608.22503
- First seen：2026-08-26 10:22:05
- 发布时间：2026-08-24 00:45:58，Asia/Shanghai；上一游标后进入 arXiv announcement/recent batch，本次按 boundary late-discovered 收录
- Scan window：2026-08-24 09:32:12 ~ 2026-08-26 10:22:05
- Focus Match：P0 Focus
- 来源：arXiv primary page / systems measurement paper
- 类型：paper / GPU scale-up / collective communication / straggler
- 链接：https://arxiv.org/abs/2608.22503
- Primary-source check：title、3 位作者、v1 timestamp、workload/architecture 覆盖、arrival gap、50%/78% 数字与 EVT/Hockney model claims 已对齐 arXiv metadata/abstract
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：它把“collective 慢”拆成网络传输和 barrier 前等待，并证明相同硬件、相同 kernel、统一 fabric 上仍会出现系统性 rank arrival skew；这会直接改变 NCCL、TP/EP 和 straggler 的诊断顺序。
- Status：NEW
- 建议动作：优先读 trace attribution method、GEMM variance 分布与 augmented Hockney model，再对照自己的训练 trace 检查 collective enqueue/arrival skew
- 关联主题：[NCCL](../topics/nccl.md), [Tensor Parallelism](../topics/tensor_parallelism.md), [Distributed Training](../topics/distributed_training.md), [Fault Tolerance](../topics/fault_tolerance.md)

作者分析 4 个 language models、3 代 GPU architecture 上数十万次 collective，观察到 rank 到达 barrier 的差距可达数百到数千微秒；在 8-GPU scale-up domain 中，等待可占 collective communication time 的 50% 以上。基于 per-rank kernel trace 的归因结果显示，跨 rank GEMM runtime variation 解释了 78% 的该类开销。

最重要的工程判断是：升级 NVLink/NVSwitch bandwidth 不会自动消除 collective bottleneck。如果 rank 在 collective 前就已经错位，网络越快，等待占比反而可能越突出。排障时应先区分 `arrival skew` 与 `transport time`，而不是看到 NCCL span 变长就直接归因网络。

## Observed / Rejected Candidates

| 材料 | Source ID | Focus Match | Decision | 原因 |
|---|---|---|---|---|
| [Megatron-LM CP layout integration](https://github.com/NVIDIA/Megatron-LM/pull/6776) | github:NVIDIA/Megatron-LM#6776 | P0 | Read | TE softmax attention 需要 zigzag CP，而 linear attention 不需要；混合模型若统一使用 zigzag，会让 linear-attention layer 的 chunkwise CP communication 翻倍。实现允许 contiguous/zigzag 按层转换，系统价值明确，但本次没有端到端性能数字。 |
| [vLLM Blackwell batch-invariance Triton autotune](https://github.com/vllm-project/vllm/pull/53649) | github:vllm-project/vllm#53649 | P1 | Read | merged PR 在单个 Qwen3-1.7B latency benchmark 上把平均延迟从约 0.770s 降到 0.576s，约 25% latency reduction；标题的 33.6% 属于 speedup 口径。局部收益真实，但 workload/硬件覆盖窄，不升级为系统级信号。 |
| [ShardMeter](https://arxiv.org/abs/2608.23840) | arxiv:2608.23840 | P1 | Observe | 关注 geo-distributed sharded training 的 measurement/planning，和跨地域训练有关，但当前不是训练主线，先保留索引。 |
| [Sigmoid Attention for Learned KV Eviction](https://arxiv.org/abs/2608.23296) | arxiv:2608.23296 | P1 | Observe | 把 sigmoid attention 作为 learned KV eviction 的 substrate，触及长上下文 memory policy，但当前证据偏 workshop/算法原型。 |
| [How to Train a Critic Stably and Efficiently](https://arxiv.org/abs/2608.23566) | arxiv:2608.23566 | P1 | Observe | 与 RL critic 稳定性相关，但目前更偏算法与 recipe，没有足够 runtime/infra 机制进入 Accepted。 |
| [Prime Agent](https://arxiv.org/abs/2608.23552) | arxiv:2608.23552 | P1 | Observe | self-improving RLM harness 与 long-horizon agent 相关，先观察代码、任务真实性和持续运行证据。 |
| [AutoSaddler](https://arxiv.org/abs/2608.23041) | arxiv:2608.23041 | P1 | Observe | 自动优化 agent harness/scaffold，方向相关但尚未形成可迁移的 RL runtime 设计。 |
| [TRL DistillationTrainer tool calling](https://github.com/huggingface/trl/pull/6723) | github:huggingface/trl#6723 | P1 | Observe | tool-call trajectory 进入 distillation trainer 是有价值的接口扩展，但没有改变 rollout/training architecture。 |
| [SGLang disaggregated-prefill bootstrap fixes](https://github.com/sgl-project/sglang/pull/36029) | github:sgl-project/sglang#36029 | P1 | Observe | 修复 bootstrap metadata/room lifecycle 的 correctness 问题，对 PD deployment 有生产价值，但属于局部修复。 |

## OpenAI / Anthropic / NVIDIA / DeepSeek Watch

| 来源 | 本次结果 | Decision | 判断 |
|---|---|---|---|
| OpenAI | Jalapeño first results | **Accepted / Deep Dive** | 自研 inference silicon 已从 roadmap 进入可公开比较的 working system；重点是 local state、prefill/decode fungibility 和 full-stack co-design，不只看厂商 benchmark 倍数。 |
| Anthropic | 未发现窗口内可核验的新 paper、technical report 或 infra engineering post | Not found | 不用产品更新补位；下次继续检查 research/news/engineering primary source。 |
| NVIDIA | Megatron-LM CP layout conversion/integration；未发现新的 NVIDIA Technical Blog 系统报告 | Read / Observe | hybrid attention 需要 layer-aware CP layout，这是一条值得迁移的实现信号；缺少端到端 benchmark，暂不升级 Accepted。 |
| DeepSeek | API changelog 与 official Hugging Face organization 未发现晚于上一扫描已记录发布的新技术报告/权重 | Not found | V4-Pro 与 V4-Flash-Vision-Exp 已在前序 scan 记录，本次不重复计数。 |

## Hugging Face Watch

- **Blog**：未发现窗口内足以改变 Agentic RL、distributed training 或 inference backend 判断的新官方文章。
- **TRL**：`DistillationTrainer` 增加 tool calling 支持，保留为 Observed；它改善数据/trajectory contract，但不是新的调度或训练架构。
- **Transformers / Accelerate / PEFT / Kernels**：检查了窗口内 release/default-branch 变化，主要是测试、兼容性与局部 correctness 调整，没有强行升级 Accepted。

## RL Framework Watch

| Framework | Window 内可核验变化 | Decision | 对 AReaL 的判断 |
|---|---|---|---|
| AReaL | 未发现晚于上一游标、且达到 architecture/performance/correctness 门槛的新 merged change | Not found | 继续跟进 AWEX 与 separation mode 的后续 benchmark，不用普通 commit 填充扫描。 |
| verl | 窗口内主要为 CI/config/docs 与常规修复 | Ignore | Separate Async 已在 08-24 scan 收录，本次没有新的 scheduler/weight-sync 机制。 |
| slime | rollout 代码整理和文档更新 | Ignore | 没有改变 rollout lifecycle、resource scheduling 或 weight sync contract。 |
| ROLL | 未发现窗口内 material release/merged change | Not found | 继续观察异步调度与 backend integration。 |
| OpenRLHF | official default-branch feed 未提供可核验的新重大变化 | Not found / limited | 本次来源可见性有限，下次回看；不把旧 change 当新信号。 |
| NeMo RL | 未发现晚于上一游标的新重大 merged change | Not found | CPU RDMA trajectory data plane 已在 08-24 scan 收录，不重复。 |

## Adjacent Runtime Watch

| Runtime | 变化 | Decision | 工程判断 |
|---|---|---|---|
| Megatron-LM | hybrid model CP layout conversion/integration | Read | Attention 类型不同，最优 CP layout 也不同；layout policy 必须进入 layer graph，不能只按 model-global flag 配置。 |
| vLLM | Blackwell batch-invariance kernel autotune；TP determinism correctness guard | Read / Observe | 前者是局部性能信号，后者说明 batch invariance 与 fused collective/kernel 组合仍有 determinism 边界。 |
| SGLang | disaggregated prefill metadata/room lifecycle 与 FP8 MoE correctness fixes | Observe | 高活跃 serving runtime 当前更值得跟踪 correctness contract，而不是把每个小 benchmark 当架构突破。 |

## Reading Queue 判断

- [ ] **今天只读一个：Synchronization Tax。** 先读 abstract、Figure 1/2、trace attribution 和 augmented Hockney model，回答“collective span 里到底有多少是网络，多少是 rank 等待”。
- [ ] **第二优先：Maia 200。** 不背峰值数字，只理解 SDLA 如何把 data movement 变成软件可编排对象。
- [ ] Jalapeño 保留为工业路线精读：重点审 benchmark boundary、local state 和 prefill/decode fungibility，不把厂商结论直接当通用结论。

## 去重记录

- 新增 Accepted Source ID：`blog:openai/jalapeno-first-results-2026-08-25`、`arxiv:2608.24664`、`arxiv:2608.22503`。
- Synchronization Tax 的 submission time 早于上一游标，但在上一 scan 后进入可见 announcement/recent batch，本次标记为 boundary late-discovered；后续不重复收录。
- DeepSeek V4 系列、verl Separate Async、AReaL AWEX、vLLM Sharded RDT 与 NeMo RL CPU RDMA 已在 08-24 或更早 scan 记录，本次只做 watch 去重。

## 扫描完整性

- arXiv：检查 cs.AI / cs.LG / cs.CL / cs.DC / cs.PF / cs.AR / stat.ML official recent pages；Accepted paper 均核对 title、authors、submission timestamp、abstract mechanism 与数字。
- Core vendors：OpenAI engineering/research、Anthropic research/news、NVIDIA Technical Blog/Megatron-LM、DeepSeek API changelog/HF organization 均显式检查。
- Hugging Face：Blog、TRL、Transformers、Accelerate、PEFT、Kernels 已检查；没有因 watch 义务强行凑 Accepted。
- RL frameworks：AReaL、verl、slime、ROLL、OpenRLHF、NeMo RL 已检查；部分 GitHub feed 的可见性/更新时间有限，已在表中标注，不用旧提交填补窗口。
- Adjacent runtime：Megatron-LM、vLLM、SGLang 已检查 merged/default-branch changes。
- 边界：扫描截止时刻固定为 `2026-08-26 10:22:05`；晚于该时刻的 arXiv announcement、vendor update 或 merge 留给下一次。
- 下一游标：`2026-08-26 10:22:05`。

## 下一步动作

- [x] 更新 [Scan Log](scan_log.md) 与 [Tracking README](README.md)。
- [ ] 阅读 Synchronization Tax，形成一条关于 `arrival skew != network time` 的排障判断。
- [ ] 阅读 Maia 200 的 dataflow programming model，判断是否值得进入 inference-infra topic/backfill。
- [ ] 下一次扫描从 `2026-08-26 10:22:05` 开始，继续按 Source ID 去重。

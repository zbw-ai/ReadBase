# 大模型训练推理 Infra 高级工程师：三天面试冲刺手册

> - 适用对象：社招大模型训练/推理 Infra 高级工程师
> - 目标档位：当前年薪约 80 万，目标 100–120 万
> - 使用窗口：首轮面试前 3 天
> - 核验日期：2026-08-30
> - 依据：[2026 版简历](output/曾柏炜-大模型训练推理Infra高级工程师-原格式版-2026.docx)、[项目事实底稿](2026-08-xpeng-infra-resume-materials.md)及文末官方资料

## 0. 先看结论：面试官会如何评估你

这个薪资档位不会只考“知不知道 TP、GRPO、vLLM”。面试官通常在同时验证六件事：

1. **真实性**：数字是不是亲自做出来的，能否说清 workload、基线、变量和统计口径。
2. **机制理解**：知道配置怎么写，还是理解数据流、通信、显存和正确性边界。
3. **工程判断**：能否从现象收敛到一阶瓶颈，并设计最小可证伪实验。
4. **系统能力**：能否跨训练、推理、调度、存储、网络和故障恢复看完整链路。
5. **结果意识**：优化是否守住数值正确性和模型效果，而不是只把 GPU 跑满。
6. **高级工程师成熟度**：是否能定义问题、做取舍、推动交付、复盘并沉淀平台能力。

回答任何技术题都尽量使用“五句法”：

> **结论**是什么 → **机制**为什么 → 当时做了什么**选择** → 用什么**证据**验证 → 结论的**边界**是什么。

如果一个问题没做过，不要把文档知识包装成项目经验。推荐说法是：“这个能力在项目里我主要是使用和调优，不是底层实现 owner；我能从机制和生产排障角度回答。”

## 1. 三天冲刺安排

### Day 1：先保证简历问不倒（约 4 小时）

- 30 分钟：背熟 90 秒自我介绍和 30 秒职业定位。
- 2 小时：完成 `RESUME-01` 至 `RESUME-10`，把每个数字补齐 workload 卡片。
- 60 分钟：口述 async RLVR、长上下文 SFT、Agentic RL/MOPD 三个故事，每个控制在 3 分钟。
- 30 分钟：处理本手册“口径风险清单”。

### Day 2：三大框架主线（约 4 小时）

- 90 分钟：Megatron-Core 六道 P0，重点是并行策略选择而非背名词。
- 75 分钟：verl 五道 P0，能画出 Actor/Rollout/Ref/Reward/Trainer 数据流。
- 75 分钟：AReaL 四道 P0，讲清异步收益、staleness 和 Agentic RL 正确性。

### Day 3：通用 Infra + 模拟面试（约 4 小时）

- 90 分钟：MFU、OOM、NCCL、checkpoint、推理吞吐。
- 60 分钟：从 P1 中选择与你目标 JD 最贴近的 10 题。
- 60 分钟：完整模拟一轮“自我介绍 → 项目深挖 → 框架 → 故障题 → 反问”。
- 30 分钟：只看最后一小时清单，不再扩展新知识。

## 2. 面试前必须校准的简历口径

### 2.1 双 Teacher MOPD 效果口径不一致

当前简历写了“双 Teacher MOPD 在 SWE、Terminal 双域提升且 General 不下降”；但项目事实底稿仍将 MOPD 定义为 `FUNCTIONAL` 已过、`NUMERIC/EFFICACY` 未完全闭环。面试前必须二选一：

- 如果已有正式 held-out paired evaluation：准备 checkpoint、样本数、seed、baseline、置信区间或 bootstrap、污染排查和 General 回归数据。
- 如果没有：口头主动收敛为“链路和 early canary 已验证，最终效果仍在严格评测”，不要继续强化已闭环的印象。

### 2.2 不同 CUDA Graph 数字不能混用

- 简历中的“35B 真实 RL decode 约 14x”是一个特定场景的 decode 阶段收益。
- 项目底稿中的“6–8x”来自另一 Agentic RL 场景。
- 二者都不能直接说成端到端 step 提速；必须带模型、引擎、batch/concurrency 和测量区间。

### 2.3 SFT `31s → 9.3s` 必须拆分贡献

必须补齐：GPU 数量、sequence length、global/micro batch、packed ratio、模型版本、前后是否相同有效 token 数，以及 `num_workers` 和选择性重计算分别贡献多少。否则 3.3x 很容易被认为是 workload 变化。

### 2.4 “交付 checkpoint”不等于“长期稳定训练”

对 128K/256K 任务分别说明：跑了多少 step、覆盖了怎样的长度分布、是否过长尾样本、checkpoint 是否完成下游验证。不要用“跑通 smoke test”替代生产稳定性。

### 2.5 Megatron-Core 的个人边界

简历写的是“了解 Megatron 5D 并行、基于 Megatron-Core 交付和二次开发”。面试时应把自己定位为**训练系统集成、性能与正确性优化者**；只有确实改过底层 collective、parallel state 或调度实现时，才说自己是 Megatron 核心机制实现者。

## 3. P0 Top 28 学习路径

| 顺序 | 题目 | 预计复习 | 通过标准 |
|---:|---|---:|---|
| 1 | [RESUME-01 自我介绍](#resume-01) | 12 分钟 | 90 秒内讲完，两条主线清楚 |
| 2 | [RESUME-02 async RLVR 优化](#resume-02) | 20 分钟 | 能从瓶颈证据讲到资源重配 |
| 3 | [RESUME-03 gen-TP 与资源配比](#resume-03) | 15 分钟 | 能解释为何减 TP 反而更快 |
| 4 | [RESUME-04 吞吐指标可信度](#resume-04) | 15 分钟 | 能说明统计窗口和可比性 |
| 5 | [RESUME-05 SFT 3.3x](#resume-05) | 20 分钟 | 能拆分数据和计算瓶颈 |
| 6 | [RESUME-06 128K/256K 长上下文](#resume-06) | 18 分钟 | 能列显存账和并行选择 |
| 7 | [RESUME-07 CP chunking 静默失效](#resume-07) | 15 分钟 | 能讲症状、根因、验证 |
| 8 | [RESUME-08 Agentic RL 架构](#resume-08) | 20 分钟 | 能画完整数据流和关键路径 |
| 9 | [RESUME-09 OPD/MOPD](#resume-09) | 20 分钟 | 能守住正确性与效果边界 |
| 10 | [RESUME-10 千卡集群交付](#resume-10) | 15 分钟 | 能讲个人 ownership 和故障体系 |
| 11 | [MEGATRON-01 5D 并行](#megatron-01) | 15 分钟 | 每个维度解决什么都说清 |
| 12 | [MEGATRON-02 Row/Column TP](#megatron-02) | 18 分钟 | 能说通信点和原因 |
| 13 | [MEGATRON-03 TP 负优化](#megatron-03) | 15 分钟 | 能结合 9B 项目解释 |
| 14 | [MEGATRON-04 SP 与 CP](#megatron-04) | 15 分钟 | 不混淆两个 sequence 切分 |
| 15 | [MEGATRON-05 Distributed Optimizer](#megatron-05) | 15 分钟 | 能说明分片对象和通信 |
| 16 | [MEGATRON-06 MoE/EP](#megatron-06) | 18 分钟 | 能解释 all-to-all 和负载均衡 |
| 17 | [VERL-01 HybridFlow 架构](#verl-01) | 18 分钟 | 能画 role/data/control flow |
| 18 | [VERL-02 Colocate 与 Disaggregate](#verl-02) | 15 分钟 | 能做资源取舍 |
| 19 | [VERL-03 训练/推理权重同步](#verl-03) | 18 分钟 | 能说布局转换和原子性 |
| 20 | [VERL-04 Fully Async](#verl-04) | 18 分钟 | 能讲 producer-consumer 与 staleness |
| 21 | [VERL-05 GRPO 正确性](#verl-05) | 18 分钟 | 能识别 logprob/mask/normalization 风险 |
| 22 | [AREAL-01 为什么选 AReaL](#areal-01) | 15 分钟 | 不停留在“更异步” |
| 23 | [AREAL-02 Off-policy 与 staleness](#areal-02) | 18 分钟 | 能讲性能-稳定性权衡 |
| 24 | [AREAL-03 Agentic RL 服务链](#areal-03) | 18 分钟 | 能讲 agent/env/reward/trainer 边界 |
| 25 | [AREAL-04 Trajectory lineage](#areal-04) | 18 分钟 | 能说明样本到底有没有训练贡献 |
| 26 | [INFRA-01 MFU](#infra-01) | 15 分钟 | 会算、会解释、会识别假提升 |
| 27 | [INFRA-02 OOM 定位](#infra-02) | 18 分钟 | 能按张量生命周期定位 |
| 28 | [INFRA-03 NCCL/Checkpoint 故障](#infra-03) | 18 分钟 | 能给出生产排查顺序 |

---

## 4. P0：简历项目深挖

<a id="resume-01"></a>
### RESUME-01｜请做一个 1–2 分钟自我介绍（P0，12 分钟）

- **问题**：请介绍一下你自己，重点讲与大模型训练推理 Infra 相关的经历。
- **面试官意图**：判断你的职业主线、表达能力和 seniority；同时选择后续深挖入口。
- **精准回答**：

  > 我目前在小鹏机器人负责大模型后训练基础设施，主要有两条主线。第一条是基于 verl 和 Megatron-Core 建设 SFT/RLVR 能力，覆盖 Qwen3/Qwen3.5 dense/MoE、32K–256K 长上下文，以及 vLLM/SGLang rollout；我做过 fully async RLVR 资源解耦，把代表性稳态吞吐从 76 提升到 211–255 tokens/s/GPU，也做过 128K SFT 的数据、重计算和显存优化。第二条是基于 AReaL 建设 Agentic RL 和在线蒸馏链路，重点解决 rollout 长尾、trajectory 利用、policy staleness、跨引擎权重同步和多 Teacher 路由正确性。此前在华为负责大模型迁移、性能/精度优化和千卡级集群长稳交付。我擅长的不只是把任务跑通，而是用指标和实验同时闭环性能、数值正确性、模型效果与故障恢复。

- **项目证据或知识边界**：所有数字必须能回到固定 workload；不要在自我介绍里主动说尚未闭环的 MOPD 最终效果。
- **高概率追问**：最有代表性的优化是什么？你在项目中的 ownership？为什么从华为到小鹏、现在又看机会？
- **危险回答**：连续罗列十几个框架；花一分钟讲学校和论文；说“全栈负责”却说不清代码和实验边界。

<a id="resume-02"></a>
### RESUME-02｜你如何把 fully async RLVR 吞吐从 76 提升到 211–255 tokens/s/GPU？（P0，20 分钟）

- **问题**：请完整讲一次这个优化，为什么一开始 async 反而很慢？
- **面试官意图**：验证数字真实性、性能分析方法和端到端系统判断，而不只是调参能力。
- **精准回答**：

  > 我先按阶段拆解同步基线，发现约 79% 时间在 rollout，因此 async 的关键不是“打开异步开关”，而是平衡样本生产率和 trainer 消费率。初始 3T+1R、gen-TP=4 时，8 张 rollout GPU 只能形成 2 个实例，稳态约 76 tokens/s/GPU，trainer idle ratio 0.41，说明 rollout 供给不足。第一步把 gen-TP 从 4 调到 2，在相同 rollout GPU 上增加实例数，吞吐到 211–255；第二步尝试 2T+2R，达到 236–293，idle ratio 降到 0.10–0.14，此时瓶颈转向 actor update。我的判断依据是 gen wait、actor update、ref、parameter sync、queue depth、idle ratio 和显存，而不是单看 GPU utilization。

- **项目证据或知识边界**：项目底稿记录的 workload 是 Qwen3-30B-A3B、32K、4×8 A100-80GB；`211–255` 是代表性稳态 step 区间，不是全程平均。
- **高概率追问**：为什么 gen-TP=2 更快？2T+2R 为什么不是最终答案？staleness 怎么控制？parameter sync 占多少？
- **危险回答**：把不同资源配比的单卡吞吐直接横比；只报最高点 293；把 async 等价为严格 on-policy。

<a id="resume-03"></a>
### RESUME-03｜为什么减小 gen-TP、增加实例数会提高 rollout 吞吐？（P0，15 分钟）

- **问题**：TP 越大单模型越快，为什么你的场景反而选择更小 TP？
- **面试官意图**：检查你是否理解 decode 的计算/通信特征、并发和集群拓扑。
- **精准回答**：

  > rollout 的目标是总 token 生产率，不是单请求最低延迟。TP 增大后，每卡 GEMM 变小、每层 collective 更频繁，decode 又是小 batch、逐 token 的 memory/latency-sensitive 阶段，未必能吃满 GPU。gen-TP 从 4 降到 2 后，同样 8 张卡可以从 2 个实例变成 4 个实例，增加独立 continuous batching 的并发池；只要单实例显存能容纳权重和 KV cache，总吞吐可能显著提升。选择点要联合看每实例 token/s、KV cache 容量、请求长度分布、跨机通信和尾延迟，而不是固定偏好某个 TP。

- **项目证据或知识边界**：你有直接项目证据；但面试前应补一张 `TP × 实例数 × 并发 × token/s × p95` 表。
- **高概率追问**：何时 TP=1 更好？什么时候必须增大 TP？长上下文 KV cache 会怎样改变结论？
- **危险回答**：“TP 通信多，所以越小越好。”模型放不下、KV cache 不够或单实例计算太慢时并不成立。

<a id="resume-04"></a>
### RESUME-04｜如何证明 76→211–255 的数据可信且可比？（P0，15 分钟）

- **问题**：你的吞吐口径是什么？是否换了 batch、数据或硬件？
- **面试官意图**：识别 benchmark cherry-pick，判断你有没有实验设计和数据治理能力。
- **精准回答**：

  > 我会先声明分子、分母和窗口：生成或有效训练 token 数，除以 GPU 数和稳态墙钟时间；再固定模型、checkpoint、prompt/response 长度分布、采样参数、硬件、并发上限和统计区间。warmup、checkpoint、validation、异常重试要单独剔除或明确包含。对 async 系统还要同时给 queue depth、trainer idle、policy version lag 和 rejected/stale ratio，避免用堆积旧样本换表面吞吐。最后至少重复多窗口，并报告区间而不是只报峰值。

- **项目证据或知识边界**：简历数字是“代表性稳态吞吐”；不要说成完整训练平均或端到端 cost reduction。
- **高概率追问**：为什么用 tokens/s/GPU？generated token 与 effective training token 有什么区别？如何做固定 logical batch A/B？
- **危险回答**：只展示 GPU utilization；把输出长度变长带来的 token/s 变化当作系统优化；不记录失败/过滤样本。

<a id="resume-05"></a>
### RESUME-05｜Qwen3.5-9B SFT 为什么能从 31s 降到 9.3s？（P0，20 分钟）

- **问题**：`num_workers=0→8` 和选择性重计算各解决了什么？
- **面试官意图**：验证你能否区分 input pipeline、GPU compute、显存与配置变化，并证明 3.3x 不是换 workload。
- **精准回答**：

  > 我先把 step 拆成 data wait、forward、backward、optimizer 和通信。`num_workers=0` 导致主进程同步取数和预处理，GPU 出现明显 data bubble；提升 worker 并配合预取后先消除输入供给瓶颈。显存侧原先采用更重的 recompute，虽然省显存但重复计算过多；通过张量级显存账确认余量后改为选择性重计算，只重算高显存/低重算代价模块。最终相同 workload 下 step time 31s→9.3s，MFU 23%→29.6%。回答时我会把两项改动分别做 A/B，避免把总收益错误归因给一个开关。

- **项目证据或知识边界**：当前简历给出了总结果，但没有展示逐项贡献；面试前必须补齐硬件、序列长度、batch 和有效 token 一致性。
- **高概率追问**：为什么 MFU 只升了 6.6pp、step 却快 3.3x？data wait 是否计入 MFU？怎样避免 workers 过多造成 CPU/内存压力？
- **危险回答**：说“num_workers 提升了 GPU 算力”；不说明前后 batch/token 数；把峰值 MFU 当平均 MFU。

<a id="resume-06"></a>
### RESUME-06｜128K/256K 长上下文训练的显存主要花在哪里？（P0，18 分钟）

- **问题**：你如何设计长上下文 SFT 的并行和显存方案？
- **面试官意图**：检查是否真正做过长序列训练，能否从张量维度做 memory accounting。
- **精准回答**：

  > 长上下文首先放大 activation，attention 中间量、logits、loss upcast 和 packed sequence 元数据也可能形成峰值。我的顺序是：先做参数/梯度/optimizer/KV 或 activation/logits/临时 buffer 的显存账；再确认 FlashAttention、THD/packing 和 CP 是否真的生效；然后在 TP、CP、recompute、offload 之间做取舍。TP 解决权重和大 GEMM 切分，但 TP 过大会缩小 GEMM 并增加通信；CP 按序列切 activation，更适合 128K/256K，但 attention 需要交换 KV。最后用真实长度分布而非只用 max length 验证，覆盖长尾样本、checkpoint 和恢复。

- **项目证据或知识边界**：简历包含 35B-MoE 256K、27B 128K/256K checkpoint 交付；应区分 working recipe、短跑交付和长期稳定基线。
- **高概率追问**：activation 为何近似随 sequence length 增长？attention memory 是否仍是二次？CP 和 Ulysses SP 区别？offload 为什么可能严重拖慢？
- **危险回答**：只说“开 FlashAttention 和重计算”；默认 max length 就是平均 workload；用更多 TP 机械解决所有 OOM。

<a id="resume-07"></a>
### RESUME-07｜CP chunking 静默失效为什么会分配 7.6GB 全量 logits buffer？（P0，15 分钟）

- **问题**：没有报错但显存异常，你怎么发现并证明 chunking 没生效？
- **面试官意图**：验证源码阅读、张量形状推导和静默正确性/性能问题定位能力。
- **精准回答**：

  > 症状是显存峰值与按 CP 分片后的理论值不一致，但任务没有显式异常。我先用 dtype × batch × sequence × vocab 估算 logits buffer，7.6GB 更接近全序列 logits，而不是 CP local chunk；再通过 rank 级 shape/log、allocation snapshot 和关键函数 tracing 确认 loss/CE 路径没有继承 CP chunk 信息。修复后我会同时验证 local tensor shape、峰值显存、loss 数值和多 rank 一致性，避免只凭 OOM 消失判断正确。

- **项目证据或知识边界**：简历只给出结论；面试前准备实际 tensor shape、dtype、vocab size 和修复位置。若无法公开代码，至少能画调用链。
- **高概率追问**：为什么是“静默”失效？如果 logits 用 fp32 会怎样？fused CE 能如何避免 materialize 全量 logits？
- **危险回答**：把 7.6GB 当固定公式背诵；无法解释任何维度；只说“看 profiler 找到的”。

<a id="resume-08"></a>
### RESUME-08｜请画出你的 Agentic RL 训练链路，最大瓶颈在哪里？（P0，20 分钟）

- **问题**：从 task 到 policy update，一条 trajectory 经历哪些系统？
- **面试官意图**：判断你是否拥有端到端视角，以及能否区分 agent、inference、reward、training 和 control plane。
- **精准回答**：

  > 链路是 task/prompt source → agent workflow → tool/sandbox environment → vLLM/SGLang rollout → trajectory/reward → trainer → policy version/checkpoint → weight sync → 新一轮 rollout。项目初期一阶瓶颈是 rollout wait，历史基线中约占 step 的 87%；根因不是单一 decode 慢，而是长上下文 late-turn cost、8-way cohort straggler、sandbox 并发和样本供给。优化时我分别处理 decode（CUDA Graph）、prefill（prefix cache）、environment 并发和 Gateway 调度，并以固定 logical batch 的端到端 update interval、effective token goodput、rejected ratio 和 staleness 验收。

- **项目证据或知识边界**：底稿记录 DeepSWE `6467s→2301s`、Seta Terminal `2240s→770s` 等更强数据，但它们未全部进入当前简历；使用前确认可对外披露和统计口径。
- **高概率追问**：为什么 cache hit 高不一定 E2E 更快？cohort 为什么放大 tail？如何区分 reward 慢和 rollout 慢？
- **危险回答**：把 Agentic RL 描述成“PPO 加 tool call”；只看模型服务器 token/s；忽略环境失败和样本版本。

<a id="resume-09"></a>
### RESUME-09｜OPD/MOPD 解决什么问题？你如何证明它正确且有效？（P0，20 分钟）

- **问题**：多 Teacher 在线蒸馏的完整数据流、loss 和验证门禁是什么？
- **面试官意图**：检查算法-系统联合能力，并重点识别效果夸大和 distributed correctness 风险。
- **精准回答**：

  > OPD 让 Student 用当前 policy 生成 trajectory，再由 Teacher 对同一 token 路径打分，训练侧使用 teacher/student logp 构造蒸馏信号；MOPD 增加了按 `data_source` 的 Teacher 路由、scatter/scoring/gather、混域配额和 trajectory 级权重。我的验收分三层：FUNCTIONAL 看 rollout、score、backward、weight sync、checkpoint 能否闭环；NUMERIC 看同权重 logp、mask、token 对齐、normalization 和各 rank fail-consistent；EFFICACY 才看 held-out paired evaluation、置信区间和 General 回归。系统跑通不能替代数值正确，更不能替代模型效果。

- **项目证据或知识边界**：这是当前最大口径风险。若无法拿出双 Teacher 正式评测，就明确说 early canary，不重复简历中的最终效果结论。
- **高概率追问**：为什么 Teacher 弱于 Student 会失败？reverse KL 的 token advantage 怎么构造？为什么 equal-token weighting 可能偏向长 trajectory？
- **危险回答**：用训练 loss 下降证明能力提升；把单 Teacher 和双 Teacher 结果混在一起；忽略数据污染和 Teacher headroom。

<a id="resume-10"></a>
### RESUME-10｜你在千卡/万卡级交付里具体负责什么？（P0，15 分钟）

- **问题**：不要讲团队成果，请讲你亲自做的决策、代码和故障闭环。
- **面试官意图**：拆分个人贡献与平台/团队红利，评估项目 owner 和跨团队推动能力。
- **精准回答**：

  > 在华为阶段我的职责不是单点模型适配，而是从需求澄清、并行/精度/性能方案、POC 到现网长稳保障的端到端接口。技术上我会把问题拆成模型图与算子、并行策略、HCCS/RoCE 通信、内存和数据链路；项目上负责风险清单、问题分级、人员协同和客户复现环境。以 MoE 项目为例，我参与/负责从功能打通、精度对齐到把相对性能从 0.16x 优化到 0.95x、MFU 达 35%，并保障 3K 卡训练。我的个人边界会明确到负责模块、关键实验和最终交付件，不把整个客户项目都算成个人代码成果。

- **项目证据或知识边界**：客户、卡数和模型规模继续按简历 `x/xx` 脱敏；面试前准备一个可公开的故障案例和一张职责 RACI。
- **高概率追问**：3K 卡最常见故障是什么？如何区分硬件、网络和框架问题？带 4–5 人如何分工？
- **危险回答**：只说“协调资源、推动闭环”；把客户业务效果归因给自己；泄露客户或集群敏感信息。

---

## 5. P0：Megatron-Core 高频题

<a id="megatron-01"></a>
### MEGATRON-01｜Megatron 的“5D 并行”分别解决什么问题？（P0，15 分钟）

- **问题**：TP、PP、DP、CP、EP 如何组合？总 GPU 数怎么计算？
- **面试官意图**：检查分布式训练基本盘，以及你能否按模型/序列/拓扑选择并行策略。
- **精准回答**：

  > DP 切 batch、复制模型，用 collective 同步梯度；TP 切层内大矩阵，解决单层权重/计算，但通信频繁；PP 按层切深度，解决整模型容量但引入 bubble；CP 切 sequence 和 activation，主要服务长上下文，attention 需要跨 CP rank 交换 KV；EP 把 MoE experts 分布到不同 rank，引入动态 token dispatch/all-to-all。Dense 场景可写成 `world_size = TP × PP × CP × DP`。MoE 场景若把 expert data parallel 记作 EDP，则可写成 `world_size = TP × PP × CP × EP × EDP`；实际配置里 EP 常从数据并行域继续分组，所以不能把 EP 再乘到一个已经包含 EP 的“总 DP”上。选择时先满足容量，再把高频通信限制在高速拓扑域，最后用 profile 优化吞吐。

- **项目证据或知识边界**：你有 Megatron 后端的配置、集成与调优经验；不要声称设计了全部并行算法。
- **高概率追问**：官方示例中的 DP 是 dense DP 还是 expert DP？expert-DP group 如何得到？TP/CP 哪个优先放单机？
- **危险回答**：把 SP 当成第六个独立 world-size 维度；认为各维完全正交；只背定义不谈通信。

<a id="megatron-02"></a>
### MEGATRON-02｜Column Parallel 和 Row Parallel Linear 怎么切？通信在哪里？（P0，18 分钟）

- **问题**：以 MLP 或 Attention projection 说明 forward/backward collective。
- **面试官意图**：判断 TP 是否停留在“把模型切到多卡”的表层。
- **精准回答**：

  > 对 `Y=XW`，Column Parallel 沿 `W` 的输出维切，每个 rank 得到部分输出特征；如果下一个算子能继续消费分片输出，就不必立即 all-gather。Row Parallel 沿 `W` 的输入维切，每个 rank 计算 partial output，forward 需要 reduce-sum 合并。Megatron 把 MLP 的 up projection 设计成 Column Parallel、down projection 设计成 Row Parallel，让中间 hidden 分片直接流动，只在 block 边界做必要规约。backward 的通信与 forward 对偶：Column Parallel 需要为 input gradient 做规约，Row Parallel 需要把 output gradient 切给各 rank。实际实现可能用 all-reduce，配合 sequence parallel 后常拆成 reduce-scatter/all-gather。

- **项目证据或知识边界**：这是框架机制题；简历只有使用/调优证据，无需假装亲自实现 TP layer。
- **高概率追问**：QKV projection 如何切 head？为什么 TP 要求 hidden/head 数可整除？sequence parallel 如何改变通信？
- **危险回答**：只说“按行/按列平均切”；混淆权重矩阵的逻辑维度与代码存储布局；说 TP 没有通信。

<a id="megatron-03"></a>
### MEGATRON-03｜为什么 TP 从 2 增到 4 可能更慢？（P0，15 分钟）

- **问题**：结合你的 9B 长上下文项目解释 TP 负优化。
- **面试官意图**：评估工程取舍和性能模型，而非配置记忆。
- **精准回答**：

  > TP 增大能降低每卡权重和部分 activation，但代价是每层通信更频繁、单 rank GEMM 的 M/N/K 变小，kernel efficiency 下降。对 9B 这种 hidden size 相对较小的模型，TP=4 可能把 GEMM 切得过碎，且长上下文问题本质更多在 sequence activation，此时把 GPU 预算给 CP 往往更合适。项目里应比较 `TP=2, CP=8` 与 `TP=4, CP=4` 的峰值显存、GEMM 时间、TP collective、CP attention 通信和 step time，而不是只看能否启动。

- **项目证据或知识边界**：项目底稿记录过约 `163s→102s` 的相关对比；该数字不在当前简历，使用前确认 workload 与披露范围。
- **高概率追问**：为什么 TP 通常放 NVLink 域内？如果 TP=2 放不下怎么办？怎样用 Nsight/NCCL trace 证明？
- **危险回答**：“TP 越大通信越大”但说不出通信频率和张量；忽略 batch/GEMM shape；把一次结果普适化。

<a id="megatron-04"></a>
### MEGATRON-04｜Sequence Parallel 和 Context Parallel 有什么区别？（P0，15 分钟）

- **问题**：它们都切 sequence，为什么不是同一件事？
- **面试官意图**：这是 Megatron 高频辨析题，能快速筛掉只背 5D 名词的人。
- **精准回答**：

  > Megatron 的 sequence parallel 通常和 TP 绑定，主要把 TP 区域中原本复制的 LayerNorm、Dropout 等 activation 沿 sequence 维分片，降低冗余，并把部分 TP all-reduce 拆成 reduce-scatter/all-gather；attention 的全序列语义并没有因此被完整分布。Context Parallel 则把网络输入和几乎所有 activation 沿 sequence 长度持久切分，每个 CP rank 只持有一段 token；attention 为让本地 Q 看到全局 KV，需要 ring/P2P/all-gather 等通信。因此 SP 是 TP 的 activation 去重优化，CP 是长上下文的独立并行轴。

- **项目证据或知识边界**：你有 CP/THD/packed 配置经验；底层通信算法若未改过，应定位为使用与诊断。
- **高概率追问**：CP 为什么能替代一部分 full recompute？GQA/MQA 下 KV 通信怎样变化？packed sequence 对 CP load balance 有何影响？
- **危险回答**：“SP 切短序列，CP 切长序列”；认为 SP 会分片所有 attention activation；忽略 CP 的 KV 通信。

<a id="megatron-05"></a>
### MEGATRON-05｜Megatron Distributed Optimizer 与 ZeRO-1/2/3 怎么对应？（P0，15 分钟）

- **问题**：它分片了什么、每步有哪些通信、能省多少显存？
- **面试官意图**：验证 model-state memory accounting 和 DP 通信理解。
- **精准回答**：

  > 经典 Megatron distributed optimizer 主要在 DP 维分片 optimizer state 和 FP32 main parameters，梯度通过 reduce-scatter 让各 rank 得到自己负责的 shard，更新后再 all-gather 参数视图，思想接近 ZeRO-1，并通过 contiguous param/grad buffer 提高通信效率。现代 Megatron-FSDP 又可配置 `optim`、`optim_grads`、`optim_grads_params`，分别对应 ZeRO-1/2/3 式分片。显存不能只背 `16/d`，要看 param/grad dtype；官方示例中 fp16 param+grad 从每参数约 20 bytes 变为 `4+16/d`，bf16 param+fp32 grad 则是 `6+12/d`。

- **项目证据或知识边界**：你做过 distributed checkpoint 和 optimizer 相关故障；若没改 optimizer 核心，明确为集成/排障经验。
- **高概率追问**：DP=1 时还有什么冗余 buffer？overlap grad reduce 如何实现？ZeRO-3 与 TP/PP 怎么组合？
- **危险回答**：把 Megatron distributed optimizer 直接等同 ZeRO-3；忽略 main param 和 dtype；认为分片没有通信成本。

<a id="megatron-06"></a>
### MEGATRON-06｜MoE 为什么需要 EP？all-to-all 为什么难优化？（P0，18 分钟）

- **问题**：请从 router、dispatch、expert compute、combine 讲一层 MoE。
- **面试官意图**：验证简历中 dense/MoE 经验，以及动态通信和负载均衡能力。
- **精准回答**：

  > Router 为每个 token 选择 top-k expert；EP 把 experts 放到不同 rank，token 先按目的 expert 做 permute/dispatch 和 all-to-all，到本地 grouped GEMM 计算，再 all-to-all combine 并恢复顺序。难点是 token 路由动态、每个 rank 发送量不均，热点 expert 会让最快 rank 等最慢 rank；小 expert batch 还会降低 GEMM 效率。优化要联合看 expert load、capacity/dropped token、A2A p95、permutation、grouped GEMM、expert placement 和网络拓扑。TP+EP 组合时官方要求启用 sequence parallel，避免相关 activation 复制和布局问题。

- **项目证据或知识边界**：你有 Qwen3/Qwen3.5 MoE recipe 和华为大 MoE 优化经验；准备一个具体的 expert imbalance 或 A2A 案例。
- **高概率追问**：top-1 与 top-2 的代价？capacity factor 如何影响效果和性能？EP 跨节点怎么放？MoE checkpoint 如何 reshuffle？
- **危险回答**：“MoE 每 token 只算少数 expert，所以一定更快”；只谈参数量，不谈动态通信和负载尾部。

---

## 6. P0：verl 高频题

> 版本提醒：截至核验日，verl 官方最新 release 页面显示 `v0.7.1`；统一 Engine Worker 架构已取代旧 worker 实现，`fully_async_policy` 仍位于 `verl.experimental`。面试时先说明你项目使用的具体分支/版本，再谈当前 upstream。

<a id="verl-01"></a>
### VERL-01｜verl/HybridFlow 的核心架构是什么？（P0，18 分钟）

- **问题**：RayPPOTrainer、WorkerGroup、Actor/Rollout/Ref/Critic/Reward 如何协同？
- **面试官意图**：判断你是否真正读过/改过框架，而不是只会运行 recipe。
- **精准回答**：

  > verl 把 RL dataflow 的控制逻辑和各模型计算后端分开。单 controller 的 RayPPOTrainer 负责主训练循环、Worker/WorkerGroup 构造和资源池；WorkerGroup 是远端 workers 的代理，负责数据 dispatch/collect。ActorRolloutRef 可以按角色独立或 colocate，Critic/Reward 也有独立 worker group；DataProto 在方法间传递 tensor 和 metadata。新架构中 TrainingWorker 通过 BaseEngine/EngineRegistry 对接 FSDP、Megatron 等训练后端，rollout 对接 vLLM/SGLang。它的价值是用一套上层 RL dataflow 组合不同计算和 placement 策略。

- **项目证据或知识边界**：你有 verl 二次开发经验；面试前至少能指出自己改过的 trainer/worker/config 路径和一个 upstream 差异。
- **高概率追问**：controller 是否会成为瓶颈？DataProto 如何跨 rank dispatch？旧 `megatron_workers` 与新 Engine Workers 有何变化？
- **危险回答**：只说“verl 基于 Ray”；混淆 trainer control plane 与每 GPU worker；背旧版本类名却不说明版本。

<a id="verl-02"></a>
### VERL-02｜Actor/Rollout 应该 colocate 还是分离部署？（P0，15 分钟）

- **问题**：什么场景选择 3T+1R、2T+2R 或 colocated hybrid engine？
- **面试官意图**：评估资源建模、权重同步和不同 workload 下的系统取舍。
- **精准回答**：

  > Colocate 的优点是 GPU 复用和本地/高速权重切换，适合资源有限、同步边界清晰的场景；缺点是训练态参数/optimizer/activation 与推理 KV cache 争显存，还需要 sleep/wakeup/reshard。分离部署可以让 rollout 和 trainer 真正 overlap、独立扩缩容，适合 agentic 长尾明显的场景，但需要跨池权重同步、队列、staleness 和故障恢复。资源比不按模型大小拍脑袋，而由 rollout producer rate、trainer consumer rate、update cadence 和 tail latency 决定；最优点是两侧 exposed idle 最小且样本新鲜度可接受。

- **项目证据或知识边界**：直接对应你的 fully async 项目；3T+1R/2T+2R 是本项目布局，不是通用最佳实践。
- **高概率追问**：为什么 2T+2R 后瓶颈转向 actor？动态资源调度何时更优？colocate 如何释放 KV/optimizer 显存？
- **危险回答**：“分离一定吞吐更高”；只算 GPU 数，不算参数同步和 queue；忽略故障域扩大。

<a id="verl-03"></a>
### VERL-03｜训练态 Megatron 权重如何同步到 vLLM/SGLang？（P0，18 分钟）

- **问题**：为什么不是简单 `state_dict` 拷贝？
- **面试官意图**：检查训练-推理双态模型、并行布局转换和一致性保证。
- **精准回答**：

  > 训练侧可能按 TP/PP/CP/EP 和 distributed optimizer 分片，推理侧则按 serving TP、量化和 engine-specific layout 存权重；同步需要参数命名/shape/dtype 映射、必要的 gather/reshard，再通过 NCCL/XCCL、CUDA IPC 或其他 backend refit。正确性上要给每次更新单调 policy version，所有 inference replicas 完成后原子切换；失败时不能让同一训练 batch 混入半新半旧权重。性能上关注导出、传输、load/refit、cache invalidation/re-prefill 和 rollout pause 的 exposed time。

- **项目证据或知识边界**：你有跨引擎同步和 final parameter sync 故障经验；准备一次 keyword mismatch 或部分 worker 失败的真实排查。
- **高概率追问**：TP size 不同如何 reshard？LoRA 只同步 adapter 有何差异？如何做 same-weight logp check？
- **危险回答**：认为 NCCL broadcast 完成就代表所有 engine 已可服务；忽略 tokenizer/chat template 和 tied weights；没有 version barrier。

<a id="verl-04"></a>
### VERL-04｜verl fully async 的 producer-consumer 流程是什么？（P0，18 分钟）

- **问题**：Rollouter、MessageQueue、Trainer、ParameterSynchronizer 各做什么？
- **面试官意图**：验证你对自己最强项目的框架层理解，并观察是否认识到 async 并非天然 on-policy。
- **精准回答**：

  > Rollouter 持续逐样本生成并按 freshness/capacity 写入 MessageQueue；Trainer 按训练所需 mini-batch 从队列消费，完成若干 update 后由 ParameterSynchronizer 把新权重同步到 rollout。收益来自训练与生成时间重叠、隔离长尾，而不是让单个阶段本身更快。关键控制量包括 queue depth、staleness threshold、parameter-sync cadence、partial rollout 和 actor/rollout 资源比。严格 on-policy 需要更强 barrier；fully async 常会产生 one-step 或 bounded off-policy，因此必须记录 behavior policy version/logprob 并使用 correction、拒绝或阈值控制。

- **项目证据或知识边界**：截至核验日该功能仍在 `experimental`；你的项目分支可能与 upstream `v0.7.1` 不同，应明确版本。
- **高概率追问**：queue 满/空分别说明什么？怎么 checkpoint in-flight samples？staleness=0 是否自动严格 on-policy？
- **危险回答**：说“完全异步但没有陈旧样本”；只调队列大小；忽略恢复后的 pending/running prompt。

<a id="verl-05"></a>
### VERL-05｜GRPO/RLVR 链路最容易出现哪些“能跑但训错”的问题？（P0，18 分钟）

- **问题**：为什么 loss 正常、reward 也涨，结果仍可能不可信？
- **面试官意图**：检查数值正确性和 RL 系统经验，这是高级岗位的重要分水岭。
- **精准回答**：

  > 我会按 token、trajectory、group、policy version 四层检查。token 层看 tokenizer/chat template、response mask、rollout 与 trainer logprob、padding/packing；trajectory 层看 reward 对齐、截断、tool trace 和有效 token normalization；group 层看 GRPO 同 prompt samples 是否完整、reward std=0、partial/rejected group；policy 层看 behavior version、importance ratio、weight sync 和 stale rejection。验证方法包括 same-weight logp、tiny deterministic batch、per-token diff、single-rank/多-rank对照、loss 手算和 held-out eval。训练不 NaN 只证明 functional，不证明 numeric 或 efficacy。

- **项目证据或知识边界**：直接对应你的 OPD/MOPD、rollout correction 和 tracing 经验。
- **高概率追问**：rollout logprob 和 trainer recompute logprob 为什么会不一致？group std=0 怎么处理？response length normalization 有何偏差？
- **危险回答**：只看最终 reward；把 KL/loss 曲线平滑当作正确性证据；不记录原始 token ids。

---

## 7. P0：AReaL 高频题

> 版本提醒：AReaL 在 2026-07-01 发布 2.0，官方描述为 training、inference、agent、weight-update 独立微服务。你的项目经历可能基于 1.x/内部演进版本，回答时先给版本边界，不要把 2.0 新架构倒推为当时已经使用。

<a id="areal-01"></a>
### AREAL-01｜已经有 verl，为什么还要选择 AReaL？（P0，15 分钟）

- **问题**：两个框架的核心定位和适用场景有什么差别？
- **面试官意图**：评估选型能力；也会验证你是否只是同时列出两个热门框架。
- **精准回答**：

  > 两者都覆盖大模型 RL，但设计重心不同。verl/HybridFlow 强在用统一 RL dataflow、WorkerGroup 和多训练/推理后端组合 colocated 或 separated placement，生态与算法 recipe 丰富；AReaL 从一开始更强调 disaggregated fully asynchronous RL、bounded off-policyness，以及把外部 agent runtime/online clients 接入训练。我的选择不是谁“更先进”，而是按 workload：常规 SFT/RLVR 和已有 verl 生态优先 verl；长时、多轮、外部 sandbox、rollout tail 明显且需要持续在线供给时，AReaL 的异步和 agent integration 更自然。代价是 staleness、trajectory lifecycle、微服务故障与可观测性更复杂。

- **项目证据或知识边界**：你分别有 verl RLVR 和 AReaL Agentic RL 项目，是强项目证据；但要说明当时的版本和公司二次开发。
- **高概率追问**：verl fully async 后差异是否还成立？AReaL 2.0 与旧架构有何变化？同一任务怎么做公平选型 benchmark？
- **危险回答**：“AReaL 异步、verl 同步”；2026 年的 verl 已有 fully async；也不要用社区 benchmark 代替本项目证据。

<a id="areal-02"></a>
### AREAL-02｜AReaL 如何控制异步训练的 off-policyness？（P0，18 分钟）

- **问题**：`max_head_offpolicyness`、policy version 和 partial rollout 如何协同？
- **面试官意图**：验证你是否理解 async 的算法代价，而不只是吞吐收益。
- **精准回答**：

  > 异步时 rollout 由旧 policy 产生，trainer 已更新到新版本。AReaL 用版本化 capacity/staleness manager 限制 trajectory head 相对当前 policy 的最大落后；`max_head_offpolicyness=0` 可退化到同步，增大阈值通常提高 overlap 和吞吐，但可能增加训练偏差。partial rollout 还能让一条长 trajectory 跨 policy version 分段，因此不能只给整条 trajectory 一个粗粒度版本；最好保留 per-turn/per-token behavior metadata，并结合 importance ratio、rejection/masking 和效果 A/B 验证阈值。

- **项目证据或知识边界**：你有 staleness manager、policy version 和 rejection diagnostics 经验；不要引用官方“通常 2–8”当作项目最优值。
- **高概率追问**：manager head drift 和真实 behavior staleness 区别？stale 样本直接丢弃会有什么系统后果？
- **危险回答**：把 off-policy 只当数据过期问题；认为版本差 1 的所有 token 偏差相同；忽略 throughput-quality frontier。

<a id="areal-03"></a>
### AREAL-03｜AReaL 2.0 的微服务化对 Agentic RL 有什么价值？（P0，18 分钟）

- **问题**：training、inference、agent、weight-update 为什么要拆开？
- **面试官意图**：考系统边界、扩缩容、故障隔离和当前框架演进敏感度。
- **精准回答**：

  > 四类服务的负载和失败模式不同：inference 受 KV cache、batching 和长尾影响；agent 受 tool/sandbox IO、session 和 retry 影响；training 受 backward/collective/checkpoint 影响；weight update 是跨布局的数据面。拆开后可以独立扩缩容、替换 backend、隔离故障，并让外部 agent 通过 OpenAI-compatible gateway 接入。但代价是多服务版本、backpressure、幂等重试、session affinity、原子权重切换和跨服务 tracing。微服务不是目的，只有当它降低资源耦合并提高可观测/可恢复性时才值得。

- **项目证据或知识边界**：AReaL 2.0 发布晚于你部分项目；可以用项目中的 Gateway/online session 经验类比，但不要说项目天然就是完整 2.0。
- **高概率追问**：控制面和数据面如何分离？weight update 服务失败怎么办？HTTP 会不会成为瓶颈？
- **危险回答**：“微服务更解耦、更高性能”而没有状态一致性设计；忽略服务间背压和恢复。

<a id="areal-04"></a>
### AREAL-04｜如何证明 generated trajectory 最终真的产生了梯度？（P0，18 分钟）

- **问题**：为什么 `generated - consumed` 不能直接叫作浪费？
- **面试官意图**：考数据 lineage、样本利用率定义和跨系统可观测性。
- **精准回答**：

  > 一条 trajectory 可能生成后等待、被 reward 过滤、因 staleness 拒绝、只完成 partial cohort、进入 trainer 但 loss mask 为零，或者训练完成但未进入当前统计窗口。因此需要 stable logical trajectory ID，把 generated → manager → workflow/reward → exported → consumed → loss-active → policy-gradient-active 逐层 join，并记录 full-sequence、loss-active 和 gradient-active token。项目 tracing 曾闭环 `223 admitted→180 generated/rewarded→96 exported→96 consumed`，其中还有 2 条 compact-filtered 消耗 token 但不产梯度。正确指标应按原因归因，而不是把差值一刀切成 waste。

- **项目证据或知识边界**：这是项目底稿中的直接证据；如不可对外披露精确数字，保留方法和比例定义。
- **高概率追问**：如何处理 microbatch reorder？tracing 本身会不会拖慢？最终 drain 时 waiting 样本算什么？
- **危险回答**：用队列长度代替 lineage；只追踪 trajectory 数而不追踪 token；忽略 tracing overhead 对 A/B 的污染。

---

## 8. P0：训练推理 Infra 通用题

<a id="infra-01"></a>
### INFRA-01｜MFU 是什么？如何正确计算和使用？（P0，15 分钟）

- **问题**：为什么 MFU 提升不一定代表用户吞吐提升？
- **面试官意图**：验证性能指标基本功和对“指标游戏”的警惕。
- **精准回答**：

  > MFU 是实际训练吞吐对应的模型理论 FLOPs 与硬件峰值 FLOPs 的比值，常写成 `model_FLOPs_per_token × tokens_per_second / aggregate_peak_FLOPs`。关键是 FLOPs 公式要匹配 dense/MoE、attention、activation checkpointing 的统计约定，峰值要匹配 dtype/Tensor Core 和硬件，tokens/s 要用真实有效 token。MFU 适合比较同模型同 workload 的计算利用，但它会忽略数据质量、padding、被 mask token、rollout 等非训练阶段；通过减少有效工作量也可能“提高”MFU。因此同时报告 effective tokens/s、step time、阶段 breakdown、显存和端到端 cost。

- **项目证据或知识边界**：你有 MFU estimator 和 SFT `23%→29.6%` 的直接经验；准备公式和模型参数。
- **高概率追问**：MoE FLOPs 按 total parameters 还是 activated parameters？recompute FLOPs 是否计入 numerator？为什么 achieved TFLOPs 和 MFU 不等价？
- **危险回答**：MFU=GPU utilization；使用不同 FLOPs 公式横比；只报百分比不报 throughput。

<a id="infra-02"></a>
### INFRA-02｜遇到 OOM，你的标准定位顺序是什么？（P0，18 分钟）

- **问题**：不要只列开关，请给出可复用排查方法。
- **面试官意图**：检查生产问题定位和张量生命周期理解。
- **精准回答**：

  > 第一步固定复现并区分 initialization、forward、backward、optimizer、checkpoint/weight sync 哪个阶段 OOM；第二步读取 allocated/reserved/peak、各 rank 差异和 fragmentation；第三步按参数、梯度、optimizer、activation、logits、通信/临时 buffer、KV cache 做理论账并对照 snapshot；第四步检查 shape 异常、padding/packing、dtype upcast、CP/TP 是否生效、长尾样本和内存泄漏；第五步才按收益/代价选择减 batch、recompute、CP/TP、offload、fused op 或 allocator 调整。修复后验证同 workload 的 loss、吞吐和长稳，不以“没 OOM”结束。

- **项目证据或知识边界**：可结合 fp32 logits、7.6GB buffer、长样本 OOM、CP=1 回退 CP=2 等案例。
- **高概率追问**：reserved 很高但 allocated 不高怎么办？为什么某一 rank 单独 OOM？offload 为什么可能造成 step 巨慢？
- **危险回答**：第一反应 `empty_cache()`；直接减 batch；把 fragmentation 当所有 OOM 的解释。

<a id="infra-03"></a>
### INFRA-03｜多机训练 NCCL hang 或 checkpoint 恢复失败怎么排查？（P0，18 分钟）

- **问题**：给出生产环境的调查顺序和止损方案。
- **面试官意图**：评估千卡经验、故障域判断、日志证据和恢复设计。
- **精准回答**：

  > 先止损：保存 job/rank/host/topology/checkpoint 证据，判断是否需要隔离节点或从上一个已验证 checkpoint 恢复。NCCL hang 按“代码一致性 → rank 健康 → 网络/硬件 → 环境版本”排查：确认所有 rank 进入相同 collective、count/dtype/group/顺序一致；找 first bad rank 和 CUDA/Xid/进程退出；再看 IB/RoCE/HCCS link、packet/error counter、拓扑和 NCCL debug trace。Checkpoint 则核对 model/optimizer/scheduler/RNG/data cursor/parallel metadata、写入原子性和 shard 完整性；恢复后用 loss continuity、参数/optimizer checksum、data position 和短窗口数值对照验证。

- **项目证据或知识边界**：你有 checkpoint deadlock、distributed optimizer checkpoint crash 和千卡交付经历；准备一个明确的 first bad event 案例。
- **高概率追问**：为什么一个 rank 提前异常会表现成其他 rank NCCL timeout？world size/TP 改变后如何恢复？async checkpoint 如何保证一致性？
- **危险回答**：一看到 hang 就重启；只看最后报错 rank；checkpoint 只保存 model weights。

---

## 9. P1：简历二级追问

### RESUME-11｜你如何定义自己在项目中的 ownership？（P1，8 分钟）

- **问题**：哪些是你设计/编码的，哪些来自开源框架或团队？
- **面试官意图**：拆分个人能力、团队协作和平台红利。
- **精准回答**：按“我定义的问题 → 我负责的模块/实验 → 依赖的团队能力 → 我推动的上线/验收”回答；给出一个代码改动、一个关键判断和一个交付结果。
- **项目证据或知识边界**：可选 fully async 资源模型、CP chunking、trajectory lineage 或 MOPD 路由；不要把开源能力描述成自研。
- **高概率追问**：关键设计谁拍板？如果没有你项目会怎样？你 review 过哪些核心模块？
- **危险回答**：反复使用“我们”；用 PR 数代替技术贡献；把所有收益都归因给自己。

### RESUME-12｜精度对齐问题通常怎么定位？（P1，10 分钟）

- **问题**：模型迁移后 loss/logits 不一致，你从哪里开始？
- **面试官意图**：验证华为阶段的精度调优不是黑盒试参。
- **精准回答**：先固定 seed/input/checkpoint 和 eval mode，从输入、embedding、逐层 hidden、attention/MLP、logits、loss 到 backward 梯度做分层 dump；区分 dtype/算子实现、mask/position、随机性、数据与优化器状态；用 first-divergence 而非最终 loss 找根因。
- **项目证据或知识边界**：可讲 NPU/CPU AIT 对比或 YOLO/LLAMA 迁移，但继续保持客户信息脱敏。
- **高概率追问**：容许误差怎么设？多机不确定性如何处理？forward 对齐但训练发散怎么办？
- **危险回答**：只比较最终输出；直接调 learning rate；把 FP16 误差都视为正常。

### RESUME-13｜CUDA Graph 为什么能让 decode 快 14x，却不能说 E2E 快 14x？（P1，8 分钟）

- **问题**：它消除了什么开销，什么情况下收益小？
- **面试官意图**：检查局部指标与端到端收益边界。
- **精准回答**：CUDA Graph 复用静态执行图，降低逐 token decode 的 CPU launch、Python 和同步开销；小 batch、小 kernel、频繁 decode 时相对收益大。但 prefill、tool/env、queue、weight sync、trainer 都不在该阶段，且动态 shape/控制流会造成 graph miss 或多 graph 管理，因此 E2E 加速由 Amdahl 定律约束。
- **项目证据或知识边界**：14x 与 6–8x 属于不同场景，必须分别带 workload。
- **高概率追问**：动态 batch 怎么 capture？权重更新后 graph 是否失效？graph 会额外占多少显存？
- **危险回答**：把 kernel/阶段加速直接乘到 step；只报最大加速；忽略 graph capture 条件。

### RESUME-14｜为什么 Prefix Cache 命中率更高，训练可能反而更慢？（P1，10 分钟）

- **问题**：缓存指标与 Agentic RL E2E 指标为什么可能反向？
- **面试官意图**：判断是否具备反直觉的系统思考和因果实验能力。
- **精准回答**：cache 只缩短重复 prefix 的 prefill；它可能让更多 episode 更快进入长上下文 late turns，增加总生成 token、cohort straggler 和环境交互，最终 trainer exposed wait 反而增加。应固定 task/seed/logical batch，对比 update interval、effective tokens、episode completion 和下游效果。
- **项目证据或知识边界**：底稿仅确认 prefill 阶段下降 44%，不能单独声称 E2E 收益。
- **高概率追问**：cache key 包含什么？session affinity 为什么重要？如何测真正 reuse ratio？
- **危险回答**：cache hit 越高系统越快；不区分 prefill time 与 episode time。

### RESUME-15｜Rejected Group 从 33.18% 降到 2.73%意味着什么？（P1，10 分钟）

- **问题**：group 为什么会被拒绝，降低拒绝率是否一定提高训练质量？
- **面试官意图**：验证 group-based RL 的数据完整性和指标解释。
- **精准回答**：先定义 rejection 原因，如 cohort 未完整、timeout、stale、reward 无效或路由失败；调度补位/均衡可以降低因长尾导致的 incomplete groups，提高样本供给。但若为了凑齐 group 放宽超时或接受更旧样本，质量可能下降，需同时看 staleness、reward distribution、effective tokens 和 eval。
- **项目证据或知识边界**：该数字来自项目底稿而非当前简历；对外使用前确认。
- **高概率追问**：partial group 能不能训练？uniform reward group 怎么处理？
- **危险回答**：把 rejected 全称为坏样本；只优化比例不看原因分布。

### RESUME-16｜你如何带 4–5 人完成复杂交付？（P1，8 分钟）

- **问题**：请讲一次技术负责人/项目负责人的具体做法。
- **面试官意图**：评估高级工程师的带项目能力，而不仅是个人贡献。
- **精准回答**：按目标/验收拆成模型适配、精度、性能、集群/现网问题；定义 owner、接口、优先级和风险；用统一复现模板和 daily blocker 收敛；关键问题亲自下钻，最终沉淀基线 recipe、故障库和客户交付件。
- **项目证据或知识边界**：华为 TX 项目有 4–5 人团队经验；说明是项目协同还是正式 people management。
- **高概率追问**：成员意见冲突怎么办？如何判断自己下钻还是授权？怎样评价交付质量？
- **危险回答**：只讲开会和催进度；把协调当管理的全部；没有技术验收机制。

## 10. P1：Megatron-Core 二级题

### MEGATRON-07｜Pipeline Parallel bubble 怎么估算和优化？（P1，10 分钟）

- **问题**：microbatch、stage balance、1F1B 和 interleaving 有什么关系？
- **面试官意图**：检查 PP 的调度与吞吐理解。
- **精准回答**：PP 把层分 stage，fill/drain 产生 bubble；基础近似中 bubble 随 stage 数增加、随 microbatch 数增加而下降。优化包括增加 microbatch、均衡每 stage 计算/显存、1F1B、virtual/interleaved pipeline 和通信 overlap，但 microbatch 过多会改变 batch/optimizer 与内存。
- **项目证据或知识边界**：技能栏“了解/使用”；如项目未重点调 PP，明确无直接性能案例。
- **高概率追问**：为什么 first/last stage 更容易不平衡？长上下文下 PP 是否更划算？
- **危险回答**：只背 bubble 公式；认为 microbatch 越多越好；忽略不均衡 stage。

### MEGATRON-08｜Packed Sequence 为什么能提吞吐，又会带来哪些风险？（P1，10 分钟）

- **问题**：padding waste、attention mask、position id 和 loss mask 如何处理？
- **面试官意图**：验证长序列训练的数据-算子接口能力。
- **精准回答**：packing 把多个样本拼入连续 token，减少 padding、提高有效 token 比；需要 cu_seqlens/segment boundary 保证 attention 不跨样本，正确生成 position/loss mask，并处理 tool/多轮边界。与 CP、FlashAttention、dynamic batch 组合时还要避免 rank load imbalance 和 shape 不兼容。
- **项目证据或知识边界**：你有 packed sequence/THD 经验；准备一个边界 bug 或验证方法。
- **高概率追问**：packing 后 batch size 怎么定义？长短样本混排如何平衡 CP rank？
- **危险回答**：只说“拼起来就行”；忽略跨样本 attention 泄漏；用总 token 代替有效 token。

### MEGATRON-09｜Recompute 和 Offload 应该怎么选？（P1，10 分钟）

- **问题**：显存不够时，为什么不是两个都开满？
- **面试官意图**：考计算、PCIe/NVLink、显存和吞吐的 trade-off。
- **精准回答**：recompute 用额外 forward FLOPs 换 activation 显存，适合计算可承受且互联/CPU 慢的场景；offload 把 param/optimizer/activation 放 CPU，节省更多 HBM，但受 PCIe、CPU 内存和 overlap 影响。应按峰值来源选择 selective recompute 或指定对象 offload，并用 exposed transfer time 而非是否异步判断成本。
- **项目证据或知识边界**：直接对应 selective recompute 与 offload PCIe 诊断。
- **高概率追问**：full recompute 约增加多少计算？哪些层适合 selective？NVMe offload 何时可用？
- **危险回答**：把 offload 当免费显存；全开后只看能跑；不做 memory snapshot。

### MEGATRON-10｜分布式 checkpoint 如何支持并行度变化恢复？（P1，10 分钟）

- **问题**：TP/PP/DP 改变时为什么普通 rank-local 文件不够？
- **面试官意图**：检查 checkpoint schema、全局 tensor metadata 和恢复验证。
- **精准回答**：checkpoint 需描述 global tensor、每 shard offset/shape/replica 和并行 metadata；加载时按新拓扑重新规划 shard，而不是让 rank 号绑定文件。还要处理 optimizer state、RNG、scheduler、data cursor 和 tied/shared weights。保存应原子提交 manifest，恢复后做 loss continuity 和短窗口对照。
- **项目证据或知识边界**：你有 Megatron distributed checkpoint crash/deadlock 经验；准备 `flattened_range` 案例边界。
- **高概率追问**：PP stage 改变如何映射层？async save 如何防半成品？optimizer reshard 为什么更难？
- **危险回答**：只转换 model weights；忽略 optimizer 和 data cursor；以能 load 作为正确。

## 11. P1：verl 二级题

### VERL-06｜DataProto 和 WorkerGroup 解决了什么问题？（P1，8 分钟）

- **问题**：为什么不用普通 Python dict + Ray actor？
- **面试官意图**：检查框架接口层和分布式数据 dispatch 理解。
- **精准回答**：DataProto 为 tensor batch 与 non-tensor metadata 提供统一协议，支持按 DP/TP 等语义 dispatch、collect 和 reorder；WorkerGroup 把多远端 worker 暴露成集体调用接口。二者降低上层算法流对具体 backend/SPMD layout 的耦合，但 schema、batch 维和 metadata 对齐错误会产生静默 bug。
- **项目证据或知识边界**：若未直接改 DataProto，说明主要从调用和故障层理解。
- **高概率追问**：non-tensor 数据如何广播？microbatch reorder 后 ID 怎么保持？
- **危险回答**：只说“序列化”；忽略 dispatch semantics 和数据对齐。

### VERL-07｜Actor、Reference、Critic、Reward 各自为什么存在？（P1，8 分钟）

- **问题**：GRPO 为什么可以没有 Critic？Reference 是否总需要？
- **面试官意图**：确认 RL 基础与系统资源角色对应。
- **精准回答**：Actor 生成并更新 policy；Reference 提供 KL/约束基线；Critic 估计 value 用于 advantage，GRPO 可用 group-relative reward 替代 learned critic；Reward 可能是模型或 rule/verifier。是否 colocate、offload 或省略取决于算法和显存/吞吐，不是固定四模型。
- **项目证据或知识边界**：有 RLVR/GRPO 使用经验；算法推导若不是主责可保持工程视角。
- **高概率追问**：DAPO 相对 GRPO 改了什么？Reference logprob 何时可预计算？
- **危险回答**：把 reward model 等同 critic；认为 GRPO 完全不需 baseline/normalization。

### VERL-08｜Ray 在 verl 中最常见的生产故障有哪些？（P1，10 分钟）

- **问题**：资源够但 worker 起不来、RPC 卡住或进程残留怎么办？
- **面试官意图**：验证多机 orchestration 实战。
- **精准回答**：从 placement group/resource pool、runtime env、端口/网络、object store、actor lifecycle 和底层 worker 异常分层；先找第一个失败 actor/rank，再看 Ray 状态与 GPU 进程，避免被 RPC timeout 末端错误误导。任务退出要幂等清理 sandbox、server、NCCL communicator 和临时资源。
- **项目证据或知识边界**：你有 Fuyao/Ray RPC/failure cleanup 经验；选一个具体案例。
- **高概率追问**：controller 挂了如何恢复？object store pressure 表现？placement group 为什么 pending？
- **危险回答**：只会 `ray stop --force`；不保存现场；把 Ray 错误当根因。

### VERL-09｜vLLM 与 SGLang rollout 后端怎么选？（P1，10 分钟）

- **问题**：不要只比 benchmark，给出训练系统选型维度。
- **面试官意图**：评估推理引擎与 RL dataflow 的集成能力。
- **精准回答**：比较目标模型支持、TP/EP/量化、continuous batching、prefix cache、structured/tool calling、sleep/wakeup、weight refit、rollout logprob 一致性、metrics、故障恢复和团队维护成本；再用项目 prompt/length/concurrency 做固定 workload A/B。训练 rollout 更重视 token/logprob 正确性和权重更新，而非纯 serving 榜单。
- **项目证据或知识边界**：你接入过两个后端；准备各自一次兼容性或稳定性问题。
- **高概率追问**：为什么同权重 logprob 会不一致？SGLang/vLLM 权重更新如何处理 cache？
- **危险回答**：按“谁更快”一刀切；只看公开榜单；忽略版本兼容矩阵。

## 12. P1：AReaL 二级题

### AREAL-05｜Partial Rollout 的收益和风险是什么？（P1，10 分钟）

- **问题**：一条 trajectory 跨多个 policy version 是否还能训练？
- **面试官意图**：考长 trajectory 调度和算法语义。
- **精准回答**：partial rollout 可避免权重更新时丢弃长 episode，提升利用率并减少 barrier；风险是同一 trajectory 内 behavior policy 不一致，credit assignment、logprob 和版本记录复杂。必须保存 segment/turn/token 级边界与 behavior metadata，并由算法决定 mask、correction 或 rejection。
- **项目证据或知识边界**：有 online session/trajectory 经验；如果项目没启用跨版本 partial，明确为机制理解。
- **高概率追问**：environment state 怎么恢复？segment reward 如何分配？
- **危险回答**：把 partial rollout 当字符串续写；整条只记一个 policy version。

### AREAL-06｜权重同步如何做到原子、可观测、可回滚？（P1，10 分钟）

- **问题**：部分 inference worker 更新失败时怎么办？
- **面试官意图**：检查分布式一致性和生产设计。
- **精准回答**：采用 prepare/transfer/validate/commit 状态机：发布 version 和 manifest，worker 接收并校验 checksum/shape，全部 ready 后 gateway 原子切流；超时则保持旧版本或隔离失败 replica，不允许混合服务。记录每 replica active version、耗时和失败原因，并保留上一稳定版本回滚。
- **项目证据或知识边界**：有 XCCL/NCCL broadcast 和 re-prefill diagnostics；说明当时实现到哪一级。
- **高概率追问**：大模型双缓冲显存不够怎么办？滚动更新能否用于训练 rollout？
- **危险回答**：一次 broadcast 即原子；失败后简单重试而不看是否部分生效。

### AREAL-07｜Online Proxy 与 session drain 为什么重要？（P1，8 分钟）

- **问题**：外部 agent client 接入训练时如何安全更新/关停？
- **面试官意图**：评估在线 Agentic RL 的 session 生命周期管理。
- **精准回答**：gateway 接受 OpenAI-compatible 请求并绑定 session/trajectory；更新或 checkpoint 前停止接收新 session，让 in-flight session 在 deadline 内完成或显式标记 partial/cancelled，再保存 queue/session/data cursor。否则会丢轨迹、重复训练或混入跨版本请求。
- **项目证据或知识边界**：你做过 online session drain 和 shutdown contract；可作为直接证据。
- **高概率追问**：客户端断线怎么处理？retry 如何幂等？session affinity 丢失会影响 cache 吗？
- **危险回答**：直接 kill server；不区分 request 完成与 trajectory 完成。

### AREAL-08｜FUNCTIONAL、NUMERIC、EFFICACY 三层门禁分别是什么？（P1，8 分钟）

- **问题**：为什么系统跑完 100 step 仍不能证明算法有效？
- **面试官意图**：考严谨性与研发验收方法。
- **精准回答**：FUNCTIONAL 验证流程和恢复能闭环；NUMERIC 验证 token、logp、mask、loss、跨 rank 一致性；EFFICACY 用无污染 held-out evaluation 验证能力收益和回归。三者有先后依赖但不能互相替代。
- **项目证据或知识边界**：这是你 MOPD 项目的核心方法论，也用于解释为什么部分结果必须谨慎。
- **高概率追问**：每层最小测试是什么？什么时候可以进入长跑？
- **危险回答**：用 loss 不 NaN 通过 numeric；用训练 reward 通过 efficacy。

## 13. P1：通用 Infra 与高级工程师题

### INFRA-04｜AllReduce、ReduceScatter、AllGather、AllToAll 分别用于哪里？（P1，10 分钟）

- **问题**：请结合 DP/TP/ZeRO/MoE，而不是只给定义。
- **面试官意图**：检查集合通信基本功和框架映射。
- **精准回答**：AllReduce 让所有 rank 得到规约结果，常见于复制参数的梯度同步；ReduceScatter 把规约结果分片，适合 sharded gradient/optimizer；AllGather 重建分片参数/activation；AllToAll 每 rank 向不同 rank 交换不同数据，是 MoE token dispatch 的核心。性能判断要看消息大小、频率、拓扑和 straggler。
- **项目证据或知识边界**：有 NCCL/XCCL 与 MoE 经验；底层算法实现若无直接经验需说明。
- **高概率追问**：为什么 RS+AG 等价于 AR 的语义组合？ring/tree 怎么选？
- **危险回答**：把 broadcast 当 all-gather；只背 API；忽略所有 rank 必须调用一致 collective。

### INFRA-05｜给你 64 张 A100，如何为 35B MoE 128K 选择并行策略？（P1，15 分钟）

- **问题**：没有完整参数时请先问哪些问题，再给初始方案。
- **面试官意图**：考需求澄清、容量模型和系统设计，不期待唯一答案。
- **精准回答**：先问 total/activated params、hidden/layers/experts/top-k、dtype、batch、长度分布、节点拓扑、目标吞吐和训练阶段；再做 model state/activation/logits 账。初始原则是 TP 留在节点高速域、CP 解决 128K、EP 按 expert 数和网络放置、PP 仅在容量/深度需要时引入，剩余形成 DP；随后用 smoke→单节点→多节点 scale curve 校正。
- **项目证据或知识边界**：可绑定 35B-A3B 与 128K/256K 交付，但不要假装题目参数已知。
- **高概率追问**：为什么不直接 TP=8？EP 是否跨节点？global batch 不可整除怎么办？
- **危险回答**：立刻报一组数字；不问模型结构和拓扑；忽略有效 batch/收敛约束。

### INFRA-06｜推理吞吐、延迟和 KV cache 如何权衡？（P1，10 分钟）

- **问题**：怎样同时解释 TTFT、TPOT、tokens/s 和 p99？
- **面试官意图**：验证推理基础与 rollout 性能模型。
- **精准回答**：prefill 主导 TTFT、计算密集；decode 逐 token、受 KV cache/内存带宽和调度影响，TPOT/p99 更关键。continuous batching 提吞吐但可能加排队；增大 batch 提 token/s 但增加延迟与 KV 占用。Agentic rollout 还需把 tool/env wait、session affinity 和 prefix reuse 纳入 E2E。
- **项目证据或知识边界**：有 vLLM/SGLang、CUDA Graph、prefix cache 和长上下文经验。
- **高概率追问**：为什么长上下文降低可并发数？chunked prefill 有何取舍？
- **危险回答**：只有 token/s 一个指标；把模型服务器延迟等同 agent episode 延迟。

### INFRA-07｜你会怎样设计训练系统的可观测性指标树？（P1，10 分钟）

- **问题**：GPU utilization 高但训练没进展，如何快速定位？
- **面试官意图**：考端到端 observability 和值班效率。
- **精准回答**：顶层用 time-to-update、effective tokens/s、cost 和 success rate；向下分 data/rollout/reward/trainer/weight-sync/checkpoint；每层有 rate、latency p50/p95/p99、queue、error、resource。再用 trajectory ID、policy version、rank/host 关联 trace，先找 critical path 和 first divergence。
- **项目证据或知识边界**：有 MFU、阶段耗时、lineage 和 DeepInsight/SwanLab 类指标经验。
- **高概率追问**：高基数 label 如何控制？如何避免 profiling 污染？
- **危险回答**：堆很多指标但没有层级；只看 GPU utilization；无跨服务 correlation ID。

### INFRA-08｜一个可恢复训练 checkpoint 必须保存什么？（P1，8 分钟）

- **问题**：Agentic RL 相比 SFT 还要多保存哪些状态？
- **面试官意图**：检查训练状态机与恢复语义。
- **精准回答**：基础包括 model、optimizer、scheduler/scaler、RNG、global step、data sampler/cursor、parallel metadata；RL/Agentic 还需 policy/reward/tokenizer/prompt/env version、queue offset、in-flight/partial trajectory、session/cohort state 和 rollout backend provenance。恢复后验证不是从“能启动”，而是 loss/data/version 连续。
- **项目证据或知识边界**：有 StatefulDataLoader、online drain、checkpoint/recovery 经验。
- **高概率追问**：哪些状态可重建？如何避免重复消费？保存 queue 会不会太大？
- **危险回答**：只保存权重；忽略 data cursor；恢复后不做数值检查。

### BEHAVIOR-01｜为什么你匹配 100–120 万档位？（P1，10 分钟）

- **问题**：相比普通训练工程师，你的不可替代性是什么？
- **面试官意图**：评估价值密度、稳定性、动机与薪资合理性，不是邀请你直接报数字。
- **精准回答**：强调三类复合能力：Megatron/verl/AReaL 的框架集成与二次开发；从长上下文、异步 rollout 到千卡集群的性能/稳定性闭环；能把模型效果、数值正确性和 Infra 交付放在同一验收体系。用 2–3 个量化项目证明，并说明下一岗位希望承担平台/核心模块 owner，而非只寻求涨薪。
- **项目证据或知识边界**：当前工作年限约 3 年多，高薪档位会追问深度和影响范围；补齐服务用户数、GPU-hours、默认 recipe/主干贡献等业务影响数据。
- **高概率追问**：为什么现在换工作？期望总包结构？若达不到怎么办？
- **危险回答**：只用学历/大厂背景论证；把目标薪资作为换工作唯一原因；虚构团队影响。

---

## 14. P2：时间允许再看

### P2-01｜FSDP 与 ZeRO 有什么关系，和 Megatron TP 有何不同？（P2，6 分钟）

- **问题**：三者都省显存，为什么不能互相替代？
- **面试官意图**：检查分片概念是否清晰。
- **精准回答**：ZeRO/FSDP 沿 DP 维分片 model states，并在需要时 gather/reshard；TP 沿层内计算维切权重和算子。前者偏状态分片，后者改变单层计算图；生产上常组合。
- **项目证据或知识边界**：技能栏声明了解 ZeRO/FSDP，主要项目更偏 Megatron；可明确无 FSDP 核心实现经验。
- **高概率追问**：FULL_SHARD 对应 ZeRO 几？FSDP 与 Megatron-FSDP 区别？
- **危险回答**：FSDP 就是 TP；ZeRO-3 没有 all-gather；认为组合一定更快。

### P2-02｜FlashAttention 为什么更省显存、更快？（P2，6 分钟）

- **问题**：它是否改变 attention 数学结果或复杂度？
- **面试官意图**：检查 kernel/IO 基础。
- **精准回答**：通过 tiling 和 online softmax 在 SRAM/register 中分块计算，避免 materialize 完整 `S×S` attention matrix，减少 HBM IO；精确 attention 的计算复杂度仍近似二次，但显存/IO 大幅改善。
- **项目证据或知识边界**：有长上下文使用经验；若没写 kernel，定位为机制和集成调优。
- **高概率追问**：为什么仍可能在 256K OOM？如何与 CP/packed sequence 组合？
- **危险回答**：把复杂度说成线性；认为它消除所有 attention activation。

### P2-03｜如何判断瓶颈在 CUDA kernel、内存带宽还是通信？（P2，8 分钟）

- **问题**：给一个 profile 方法而非工具列表。
- **面试官意图**：检查性能工程基本方法。
- **精准回答**：先做阶段 breakdown，再看 kernel occupancy/SM、Tensor Core、dram throughput、launch gap 和 collective overlap；结合 roofline、GEMM shape 与通信 trace 判断。用单机/多机、不同 batch/TP 的 scale experiment 证伪。
- **项目证据或知识边界**：你有 tracing/MFU/通信优化经验；CUDA kernel 手写深度需诚实说明。
- **高概率追问**：GPU util 高为什么仍可能低效？小 GEMM 有什么特征？
- **危险回答**：看到 util 100% 就认为 compute-bound；只说用 Nsight。

### P2-04｜设计一个 256K、多轮 Agentic RL 平台（P2，12 分钟）

- **问题**：从 API、调度、数据、正确性、恢复和指标设计。
- **面试官意图**：综合考高级工程师系统设计与取舍。
- **精准回答**：按 task source、agent/env、inference pool、trajectory/reward store、trainer、weight update、checkpoint/observability 分层；优先讲 backpressure、session affinity、staleness、原子版本、lineage 和 failure recovery，再讨论 TP/CP/KV cache。
- **项目证据或知识边界**：高度贴合你的经历；未知 SLA/规模时先提问，不急于报架构。
- **高概率追问**：哪层是 source of truth？外部 env 不稳定怎么办？如何多租户？
- **危险回答**：画一条理想流水线无失败状态；只谈模型并行；没有容量模型。

### P2-05｜如果现场让你写 producer-consumer/并发队列代码，会考什么？（P2，8 分钟）

- **问题**：如何实现有界队列、取消、重试、幂等和优雅退出？
- **面试官意图**：验证 Python/C++ 工程基本功，不让框架经验掩盖编码能力。
- **精准回答**：先定义 ownership、backpressure 和 shutdown protocol；实现 bounded queue、超时、状态机、idempotency key、异常传播和 drain；测试空/满、生产者死亡、消费者慢、重复消息和取消竞态。
- **项目证据或知识边界**：可绑定 async rollout/message queue/session drain。
- **高概率追问**：exactly-once 是否可能？锁与 async event loop 如何选择？
- **危险回答**：只给 happy path；吞掉异常；用无限队列解决阻塞。

### P2-06｜为什么从算法研究转向训练 Infra？（P2，6 分钟）

- **问题**：你的 MICCAI/AAAI 经历如何帮助当前工作？
- **面试官意图**：评估职业动机、学习能力和长期稳定性。
- **精准回答**：研究经历训练了实验设计、数值验证和论文阅读；华为/小鹏经历让你确认更擅长把模型方法转成可扩展、可恢复、可验证的系统。Infra 不是离开算法，而是用系统能力缩短算法迭代周期并守住正确性。
- **项目证据或知识边界**：可引用论文和两个阶段的职业转变，不需展开病理图像算法细节。
- **高概率追问**：未来更想做训练还是推理？是否愿意写底层 C++/CUDA？
- **危险回答**：“算法太卷所以转 Infra”；把 Infra 描述成部署运维；职业方向摇摆。

---

## 15. 三框架对比速查

| 维度 | Megatron-Core | verl | AReaL |
|---|---|---|---|
| 核心定位 | 大模型高性能训练组件与并行/模型实现 | LLM RL post-training dataflow 与多后端编排 | 面向 reasoning/agent 的异步 RL 与在线服务桥接 |
| 主要抽象 | Transformer/parallel state/distributed optimizer/checkpoint | Trainer、WorkerGroup、DataProto、BaseEngine、Rollout | training/inference/agent/weight-update、staleness、online gateway |
| 训练后端 | 自身提供 Megatron 训练栈 | 可选 Megatron、FSDP/FSDP2 等 | 可接 Megatron/FSDP 等，版本相关 |
| 推理角色 | 不是主要目标 | 集成 vLLM/SGLang 等 rollout | 独立 inference service/rollout，强调在线 agent 接入 |
| 强项 | TP/PP/CP/EP、MoE、长上下文、规模扩展 | 算法流灵活、placement、多 engine/recipe 生态 | fully async、bounded off-policy、长时 Agentic RL、服务解耦 |
| 核心代价 | 配置/模型适配复杂、通信与拓扑敏感 | role/版本/依赖矩阵复杂，async 部分仍 experimental | staleness、trajectory 状态、微服务一致性与运维复杂 |
| 你的证据 | Megatron 后端 SFT/RLVR、长上下文、MoE、checkpoint | SFT/RLVR、fully async、vLLM/SGLang、性能/稳定性 | 128K Agentic RL、在线蒸馏、lineage、weight sync |
| 诚实边界 | 主要是集成/调优，不默认是核心并行算法作者 | 项目分支可能不同于 upstream v0.7.1 | 项目版本可能早于 2.0，不能倒推使用新微服务架构 |

一句话区分：

> **Megatron-Core 决定“一个大模型如何高效训练”，verl 决定“RL 的多个模型与计算阶段如何编排”，AReaL 更强调“长时 agent 数据如何异步生产、控陈旧并在线接入训练”。**

## 16. 四张项目证据卡：面试前必须手写补齐

### 卡 1：Fully Async RLVR

```text
模型/版本：Qwen3-30B-A3B（确认）
硬件/拓扑：4×8 A100-80GB（确认是否可披露）
上下文/长度分布：32K max；平均/p95 = ______
基线：同步约 ______；async 初始 76 tok/s/GPU
变量：gen-TP、实例数、3T+1R / 2T+2R
结果窗口：211–255 / 236–293 的步数与时间 = ______
正确性：staleness、reward、eval 是否一致 = ______
个人贡献：代码模块/实验/决策 = ______
```

### 卡 2：Qwen3.5-9B SFT 3.3x

```text
GPU/并行：______
sequence length / packed ratio：______
global/micro batch / effective tokens：______
31s→9.3s 的逐项 A/B：num_workers ______；recompute ______
MFU 公式和峰值硬件 FLOPs：______
峰值显存与 loss 对齐：______
```

### 卡 3：Agentic RL / Rollout

```text
任务：DeepSWE / Terminal（按可披露范围）
模型/硬件/并发：______
端到端 critical path：______
decode 6–8x 或 14x 的具体 workload：______
prefix cache 44% 的测量阶段：prefill only
Gateway/rejected group 的基线与窗口：______
效果/正确性护栏：______
```

### 卡 4：OPD/MOPD

```text
Student / Teacher：______
训练数据与 held-out 数据：______
路由字段和多 Teacher mapping：______
loss / mask / normalization：______
FUNCTIONAL 证据：______
NUMERIC 证据：______
EFFICACY 证据与置信区间：______
当前能说/不能说：______
```

## 17. 一轮首面模拟顺序

按下面顺序录音，控制在 45–60 分钟：

1. `RESUME-01` 自我介绍（90 秒）。
2. `RESUME-02` async RLVR 主故事（3 分钟）→ `RESUME-03/04` 连续追问。
3. `RESUME-08` Agentic RL 架构（3 分钟）→ `AREAL-02/04` 连续追问。
4. `MEGATRON-01` 5D 并行 → `MEGATRON-03/04/06` 三选二。
5. `VERL-01` 架构 → `VERL-03/05`。
6. `INFRA-02` OOM 或 `INFRA-03` NCCL/checkpoint 故障题。
7. `BEHAVIOR-01` 岗位匹配度和换工作动机。
8. 向面试官反问两题。

录音复盘只检查四点：是否先说结论；是否有数字但也有口径；是否说清个人贡献；是否主动限定证据边界。

## 18. 建议反问面试官

优先问能判断岗位真实含金量的问题：

1. 团队当前主要瓶颈在 pretraining、post-training、rollout inference，还是集群稳定性？
2. Megatron/verl/AReaL 是直接使用、深度二次开发，还是自研框架？候选人入职后负责哪一层？
3. 当前训练规模、主要模型形态和长上下文范围是什么？最大痛点是吞吐、成本、正确性还是恢复？
4. 高级工程师的成功标准是什么：核心模块 ownership、平台 adoption、GPU cost、训练成功率，还是带项目？
5. 算法团队与 Infra 团队如何共同验收性能、数值正确性和模型效果？

不建议首轮一开始就只问加班、晋升和薪资结构；这些可以在 HR/后续轮次系统确认。

## 19. 面试前最后一小时清单

- [ ] 自我介绍能在 90 秒内完成，且只保留两条主线。
- [ ] async RLVR 能解释 76、211–255、236–293、0.41、0.10–0.14 各自口径。
- [ ] SFT 31→9.3s 的前后 workload 完全一致，并能拆分两项改动贡献。
- [ ] 双 Teacher MOPD 的最新效果证据已与简历口径统一。
- [ ] CUDA Graph 14x 与 6–8x 不混用，且明确是 decode 阶段。
- [ ] 能画 Megatron TP/PP/CP/DP/EP，以及 verl/AReaL 两张数据流图。
- [ ] 能用一句话区分 SP 与 CP、distributed optimizer 与 ZeRO-3、verl 与 AReaL。
- [ ] 准备一个 OOM、一个 NCCL/checkpoint、一个精度对齐真实案例。
- [ ] 每个故事能说清“我做了什么”，不只说“团队做了什么”。
- [ ] 不泄露联系方式、客户名、内部仓库、未公开模型和未脱敏集群数据。

## 20. 继续阅读：仓库内现有材料

- [Agentic RL Infrastructure](../training-infra-roadmap/topics/agentic_rl.md)
- [Tensor Parallelism 面试题](../training-infra-roadmap/interview/tensor_parallelism.md)
- [MoE 面试题](../training-infra-roadmap/interview/moe.md)
- [Checkpoint 面试题](../training-infra-roadmap/interview/checkpoint.md)
- [FSDP 面试题](../training-infra-roadmap/interview/fsdp.md)
- [FlashAttention 面试题](../training-infra-roadmap/interview/flashattention.md)
- [Megatron-LM 论文笔记](../training-infra-roadmap/papers/megatron_lm.md)

## 21. 资料来源与版本边界

技术结论优先使用官方资料；岗位题目概率来自当前公开 JD 与本简历暴露面，是面试准备判断，不是统计学结论。

### 官方框架资料（核验于 2026-08-30）

- NVIDIA Megatron-Core：[Parallelism Strategies Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)、[Context Parallelism](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)、[Distributed Optimizer](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/dist_optimizer.html)。Release 页面核验到 `core_v0.18.2`，commit `571370c`。
- verl：[GitHub](https://github.com/verl-project/verl)、[HybridFlow Programming Guide](https://verl.readthedocs.io/en/latest/hybrid_flow.html)、[Engine Workers](https://verl.readthedocs.io/en/latest/workers/engine_workers.html)、[Fully Async](https://github.com/verl-project/verl/blob/main/docs/advance/fully_async.md)。Release 页面核验到 `v0.7.1`，commit `bec9ef7`；`fully_async_policy` 仍在 `verl.experimental`。
- AReaL：[GitHub](https://github.com/areal-project/AReaL)、[Asynchronous RL Guide](https://github.com/areal-project/AReaL/blob/main/docs/en/algorithms/async.md)、[Online Proxy](https://github.com/areal-project/AReaL/blob/main/docs/en/tutorial/online_proxy.md)、[Releases](https://github.com/areal-project/AReaL/releases)。核验到 `v2.0.0`/AReaL 2.0（2026-07-01）；2.0 将 training、inference、agent、weight-update 拆为独立服务。
- NVIDIA NCCL：[Collective Operations, NCCL 2.31.2](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)。
- PyTorch：[Distributed Checkpoint Tutorial](https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html)。

### 当前岗位信号（动态页面，核验于 2026-08-30）

- [华为社招：大模型训练/强化学习/推理相关岗位](https://career.huawei.com/reccampportal/portal5/social-recruitment-detail.html?dataSource=1&jobId=28183)：强调独立系统设计、训练/RL 原理、精度调优、vLLM/SGLang 和软硬件协同。
- [华为社招：AI 底层软件栈与训推性能](https://career.huawei.com/reccampportal/portal5/social-recruitment-detail.html?dataSource=1&jobId=32189)：强调 runtime、显存、集合通信、profiling、疑难问题攻坚和稳定交付。
- BOSS 公开职位聚合中的腾讯/美团等岗位把 Megatron、verl、vLLM/SGLang、RL Infra、规模训练和系统优化列为核心职责；聚合页会变动，只用于判断常见考察方向，不用于技术事实。

## 22. 题量与时间预算

| 优先级 | 题量 | 建议投入 | 用法 |
|---|---:|---:|---|
| P0 | 28 | 6–8 小时 | 首轮前全部口述一遍 |
| P1 | 24 | 3–4 小时 | 选择与目标 JD 最相关的 10–15 题 |
| P2 | 6 | 不超过 1 小时 | 查漏补缺，不挤占项目复盘 |

最终原则：**三天内宁可把 10 个项目问题答到可追问三层，也不要浅背 100 个名词。**

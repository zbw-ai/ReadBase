# Tracking

`tracking/` 是 AI Training Infra 的研究雷达。

它不追求完整解读，也不替代 `papers/`、`tech_reports/`、`engineering_blogs/` 或 `topics/`。它的职责是记录三类输入：frontier scan 负责从上次游标到现在的最新扫描，monthly signal 负责高质量正式沉淀，historical backfill 负责按原始月份补录过去已经证明重要但仓库还没吸收的历史精华材料。

## 知识流转

```text
发现 tracking
  ↓
筛选 reading_queue
  ↓
阅读 papers / tech_reports / engineering_blogs
  ↓
理解 learning_log
  ↓
沉淀 topics / insights
  ↓
验证 experiments
```

## 文件说明

- [Scan Log](scan_log.md)：每次前沿扫描的账本，记录窗口、来源、accepted / observed 数量和下一次扫描游标。
- [Frontier Scan 2026-08-30](frontier_scan_2026-08-30.md)：当前最新扫描，覆盖到 2026-08-30 21:04:46；重点包括 NeMo RL generation-shard recovery、AReaL truncation/GAE correctness、RL-for-LLM 并行性能方法论与长上下文 VPP。
- [Frontier Scan 2026-08-28](frontier_scan_2026-08-28.md)：上一份扫描，覆盖到 2026-08-28 10:24:25；重点包括 OpenAI-Hugging Face incident technical report、psRL、Granite 4.2 异步 GRPO/128K 工业配方与 verl Liger fused PPO kernel。
- [Frontier Scan 2026-08-26](frontier_scan_2026-08-26.md)：前一份扫描，覆盖到 2026-08-26 10:22:05；收录 OpenAI Jalapeño、Microsoft Maia 200 与 GPU Synchronization Tax，核心判断聚焦 hardware-software co-design、data movement 与 rank arrival skew。
- [Frontier Scan 2026-08-24](frontier_scan_2026-08-24.md)：前一份扫描，覆盖到 2026-08-24 09:32:12；重点包括 FlashPrefill V2、CacheRoute、ReCache、verl trainer-GPU lending、AReaL AdamW delta transfer、vLLM Sharded RDT、SGLang DeepSeek-V4 Q8KV8、NeMo RL CPU RDMA 与 NVIDIA MaxLPS。
- [Frontier Scan 2026-08-20](frontier_scan_2026-08-20.md)：前一份扫描，覆盖到 2026-08-20 10:21:56；重点包括 Agent Lightning v1.0、LEGO-RL、NeMo RL async recovery、TRL Async Distillation、Open-MOPD、Megatron multi-turn packing correctness 与 AReaL Qwen3-VL AWEX colocation。
- [Frontier Scan 2026-08-18](frontier_scan_2026-08-18.md)：上一份扫描，覆盖到 2026-08-18 09:31:44；重点包括 Rollplex、NVIDIA Nemotron QAD、Megatron RL Context Parallel、FreeBalance、verl vLLM state-lifecycle fix 与 SGLang DSpark accepted-token logprobs。
- [Frontier Scan 2026-08-17](frontier_scan_2026-08-17.md)：前一份扫描，完整重扫 2026-08-14 17:39:50 到 2026-08-17 09:28:33；重点包括 DeepSeek-V4-Pro-0813、Megatron RL generation lag autotuning、NeMo rollout failure containment、Megatron disaggregated KV handoff 与 checkpoint distribution cache。
- [Frontier Scan 2026-08-14](frontier_scan_2026-08-14.md)：上一份确认游标，覆盖到 2026-08-14 17:39:50；重点包括 TideRL、MISA-T、RoutePack、AReaL grouped colocation、verl multi-sender weight sync、NeMo RL async checkpoint 与 vToken。
- [Frontier Scan 2026-08-12](frontier_scan_2026-08-12.md)：前一份扫描，覆盖到 2026-08-12 09:51；重点包括 verl Dynamic CP、slime GLM-5 训推对齐、FlashBoot、OasisKV、Replay Gap 与 NVIDIA Nemotron 3.5。
- [Frontier Scan 2026-08-09](frontier_scan_2026-08-09.md)：前一份扫描，覆盖到 2026-08-09 23:24；重点包括 K-EXAONE 2.0、TensorCast、SpecRoll、slime v0.3.1 与 AReaL AWEX colocation。
- [Frontier Scan 2026-08-03](frontier_scan_2026-08-03.md)：前一份扫描，合并 8 月 4 日增量，覆盖到 2026-08-04 13:22。
- [Frontier Scan 2026-07-28](frontier_scan_2026-07-28.md)：7 月最后一份独立扫描，覆盖到 2026-07-28 16:58。
- [Frontier Scan Template](frontier_scan_template.md)：灵活执行的最新前沿扫描模板，从上次扫描游标扫到本次实际扫描结束时刻。
- [Monthly Signal Report Template](monthly_signal_report_template.md)：每月输出上个月的高质量正式信号沉淀。
- [Monthly Signal 2026-07](monthly_signal_2026-07.md)：2026 年 7 月工程判断与工业证据月报。
- [Monthly Signal 2026-06](monthly_signal_2026-06.md)：2026 年 6 月高质量前沿信号沉淀。
- [Monthly Signal 2026-05](monthly_signal_2026-05.md)：2026 年 5 月高质量前沿信号沉淀。
- [Monthly Signal 2026-04](monthly_signal_2026-04.md)：2026 年 4 月高质量前沿信号沉淀。
- [Monthly Signal 2026-03](monthly_signal_2026-03.md)：2026 年 3 月高质量前沿信号沉淀。
- [Monthly Signal 2026-02](monthly_signal_2026-02.md)：2026 年 2 月高质量前沿信号沉淀。
- [Monthly Signal 2026-01](monthly_signal_2026-01.md)：2026 年 1 月高质量前沿信号沉淀。
- [Historical Backfill](historical_backfill.md)：历史补录总入口。
- [Backfill By Month](backfill/README.md)：按材料原始发布时间月份倒序补录历史精华材料，每个月一个文件。
- [Engineering Blogs](engineering_blogs.md)：大厂技术博客和官方文档追踪。
- [Release Notes](release_notes.md)：模型、框架、训练栈发布记录。
- [Infra Trends](infra_trends.md)：训练基础设施技术演进时间线。
- [Agentic RL](agentic_rl.md)：Agentic RL、long-context RL、rollout infra、verifier/reward pipeline 专题追踪。

## Frontier / Monthly / Historical Backfill

| 类型 | 作用 | 时间窗口 | 是否正式收录 |
|---|---|---|---|
| Frontier Scan | 捕捉从上次扫描游标到现在的新前沿信号 | 上次 `Next cursor` 到本次实际扫描结束时刻 | 否，主要是雷达 |
| Monthly Signal | 从当月 frontier/backfill/reading 中筛选高质量信号 | 上月 1 日到月末 | 是，正式沉淀 |
| Historical Backfill | 补录过去已经证明重要、但仓库还没吸收的经典材料 | 按材料原始发布时间月份归档 | 视质量进入队列 |
| Reading Queue | 从 frontier/monthly/backfill 中筛选本周真正要读的 P0/P1 | 当前学习周期 | 是，决定阅读 |

Backfill 不按时间补，按“它能补哪个工程判断缺口”来补。

## 扫描窗口与游标

- Frontier Scan：从 [Scan Log](scan_log.md) 上一次 `Next cursor` 开始，到本次实际扫描结束时刻为止。文件名为 `frontier_scan_YYYY-MM-DD.md`。
- Scan Log：每次扫描后必须更新，记录 `Window`、`Sources`、`Accepted`、`Observed`、`Next cursor` 和完整性说明。
- `Window` 结束时间和 `Next cursor` 不能预填未来时间。白天扫描就写白天的实际时刻；如果精确时刻缺失，下次扫描应回退到最后可确认时间点并去重。
- Monthly Signal：上月 1 日 00:00:00 到上月最后一天 23:59:59，时区 `Asia/Shanghai`。文件名为 `monthly_signal_YYYY-MM.md`。
- Monthly 不重新发现材料，只从当月 frontier scans、backfill、release note 和实际阅读结果中筛选。
- Historical Backfill：按材料原始发布时间月份归档到 `backfill/YYYY-MM.md`，另记录“补录时间”，不和 frontier scan 混。
- Weekly Signal：只保留历史记录，不再维护固定周报模板或 weekly papers 占位文件。需要看最新内容时使用 Frontier Scan，需要正式沉淀时使用 Monthly Signal。

## 记录原则

- 不追求全量，只记录可能改变工程判断的信号。
- 只抓取当前最关心的 AI Systems / Training Infra / Agentic RL Infra 内容，不做通用 AI newsletter。
- 不把历史材料混入 frontier scan，避免污染“最新趋势”判断。
- Frontier scan 不强行凑数；0 条 accepted signal 是合法结果。
- Monthly signal 才是正式高质量收录，通常只保留 3 到 5 条，允许更少。
- 每条 accepted signal 建议记录 `Signal ID`、`Source ID`、`First seen` 和 `Window`，避免重复收录。
- 每条材料必须有“一句话价值”。
- 每条材料必须给出 `Decision`：`Ignore`、`Observe`、`Read`、`Deep Dive`。
- 每条材料必须给出 `Reason`：为什么做这个决策。
- 每条材料建议标记 `Status`：`NEW`、`READING`、`SUMMARIZED`、`DIGESTED`、`VERIFIED`、`IMPLEMENTED`、`OBSOLETE`。
- 每条材料必须给出建议动作：`进入 P0`、`进入 P1`、`观察`、`忽略`。
- 影响等级用 `★★★★★` 到 `★`，帮助筛选。
- P0 不超过 3 条，但不是每次扫描都必须产生 P0。
- tracking 里的内容可以粗糙，但不能没有判断。

## Vendor Watch

每次 frontier scan 和 monthly signal 都必须显式维护 `OpenAI / Anthropic / NVIDIA / DeepSeek Watch`。

- OpenAI / Anthropic / NVIDIA / DeepSeek 是一级关注源：paper、technical report、official docs、engineering blog、model card、weight release、release note、research post 都要进入扫描视野。
- DeepSeek 需要同时检查官方 API changelog 与 Hugging Face organization；重要开放权重更新不一定配套独立博客。
- 四家来源不是自动进入 Accepted；仍然按照是否改变 Training/RL/Inference Infra 工程判断来筛选。
- 核心模型厂商的 technical report、model card、工程博客和规模化部署报告按一级工业证据处理：优先读其系统边界和生产证据，同时明确区分公开事实、厂商自报数字与仓库推断。
- 如果材料重要，进入 Accepted；如果相关但不够硬，进入 Observed；如果没有系统细节，写明 Rejected / Ignore。
- 如果本次未发现可核验高质量信号，或者来源端点不可用，也要在 Vendor Watch 写出来，避免三家动态在记录里“隐身”。
- NVIDIA Training Stack 相关内容优先看 Megatron-Core、Transformer Engine、NCCL、FP8/NVFP4、MoE kernel、distributed checkpointing、scheduling、observability。
- OpenAI / Anthropic 相关内容优先看 training infrastructure、post-training/RL、agent runtime、evaluation/verifier、安全训练、推理/serving、compute/network/cluster 线索。

Hugging Face 作为独立重点生态源，每次扫描还应显式维护 `Hugging Face Watch`：

- 优先扫描 Hugging Face Blog，以及 TRL、Transformers、Accelerate、PEFT、Kernels 等官方 release / docs。
- 重点关注 agentic RL、rollout correctness、training-serving integration、long context、distributed training 和 inference backend。
- 区分 Hugging Face 官方团队文章、厂商联合文章与 community post；来源级别不等于自动 Accepted，仍按工程信号筛选。

## RL Framework Watch

每次 frontier scan 必须显式维护 `RL Framework Watch`。它和厂商 Watch 的分工不同：厂商 Watch 判断技术方向，框架 Watch 判断代码、runtime 和工程能力是否已经发生可用变化。

- 核心名单：AReaL、verl、slime、ROLL、OpenRLHF、NeMo RL。
- 动态名单：新出现且具备真实代码、可运行训练链路或可复核 benchmark 的 RL Infra 框架。
- 跟踪正式 release，以及会改变架构、性能、正确性或生产行为的重大 PR。
- 不跟踪普通 commit、文档修正、小型 bugfix，避免 tracking 退化成 GitHub activity feed。
- 宣传文章、仓库 README 或未经复核的 benchmark 不能单独构成 Accepted signal。

每项保留变化至少回答四个问题：

1. 改动发生在 `rollout`、`training`、`scheduler`、`weight sync`、`data/trajectory path`、`checkpoint/recovery` 还是 `inference backend`？
2. 它解决性能、显存、稳定性、正确性还是可运维性问题？
3. 证据来自 release note、代码 diff、测试、benchmark 还是 production report？
4. 对 AReaL 当前架构是否存在可迁移的设计或实现？

Monthly Signal 不重新扫描 GitHub，只汇总当月 frontier scans 已经筛出的框架变化。

历史例外：2026-07-23 曾按用户明确要求，对 2026 年 1–6 月 Monthly Signal 和 7 月既有 Frontier Scan 做过一次 RL framework historical audit。所有回补段落都标记为 `Historical Audit`，不修改原 Accepted 数量、阅读决策或 cursor。后续不要把这种一次性迁移变成常规流程。

## Personal Focus Filter

Frontier scan 优先看这些方向：

- 训练系统：Megatron-Core、DeepSpeed、FSDP、PyTorch Distributed、NVIDIA NeMo/Megatron。
- 分布式训练：TP / PP / DP / EP / SP / CP、通信 overlap、rank mapping、拓扑。
- GPU 集群：NCCL、NVLink/NVSwitch、InfiniBand、RoCE、straggler、fault tolerance。
- 显存与状态：ZeRO、FSDP、optimizer state、distributed checkpointing、recovery。
- Kernel 与精度：FlashAttention、Transformer Engine、FP8 / NVFP4、CUTLASS、Grouped GEMM。
- MoE 与大规模训练：expert parallel、load balance、DeepSeekMoE、MegaScale、Llama/DeepSeek/Gemini 训练系统。
- Agentic RL / post-training infra：rollout、verifier/reward、RLHF/GRPO/DAPO 系统、training-serving disaggregation、weight sync、sample freshness。

以下内容通常拒绝：纯模型榜单、应用论文、领域数据集、prompt 技巧、产品新闻、没有系统细节的模型发布、无法连接到训练/推理/rollout infra 的纯算法改进。

## 从 Tracking 到沉淀

- 值得立刻读的内容进入 `reading_queue/P0.md`。
- 值得以后读的内容进入 `reading_queue/P1.md`。
- 已读完并形成判断的内容进入 `learning_log/`。
- 形成工程观点后进入 `insights/`。
- 可以实验验证的内容进入 `experiments/`。
- 被消化为可复用工程知识后进入 `topics/`。
- 进入真实工程实践或生产方案后，状态可以标记为 `IMPLEMENTED`。

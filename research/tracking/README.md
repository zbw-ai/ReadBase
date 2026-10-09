# Tracking

`research/tracking/` 是 Systems、Training、Inference、RL、Embodied 共用的研究雷达，不为每条主线复制完整雷达或阅读队列。

它不追求完整解读，也不替代共享 research notes 或各主线的 `topics/`。它的职责是记录三类输入：frontier scan 负责从上次游标到现在的最新扫描，monthly signal 负责高质量正式沉淀，historical backfill 负责按原始月份补录能填补当前模型/学习或工程判断缺口的历史材料。

## 2026-09-22 起的未来覆盖政策

保留 Training / RL 深度，同时将独立 Inference Systems、Embodied 模型机制与基础设施纳入后续扫描。此次调整只修改规则和模板，不代表已经扫描新来源，不推进 cursor，不改历史 Accepted 数量、阅读决策或完成状态。过去未覆盖的新主线要明确标记历史缺口；补读旧材料按原始月份进入 `backfill/YYYY-MM.md`，不伪装成当天新闻。

候选满足以下任一通道，才有资格继续评估；它们都不等于自动 Accepted：

1. **模型/学习价值**：直接回答当前 embodied 学习缺口，例如模型输入输出、action representation、训练目标、泛化边界或评估方法；可以尚无已证明的 infra 增益。
2. **系统后果**：改变性能、正确性、状态、IO 或部署约束，适用于 Systems / Training / Inference / RL / Embodied。

每项说明 `Target question` 和 `Evidence`。公开机制、源码/测试、可复核 benchmark、厂商自报和仓库推断分层标注；无代码不能宣称已可运行/复现，未验证效果不能写成已证实收益。通用 demo、融资、产品新闻、未经核验榜单不自动升级。这里的 MLLM 指 multimodal large language model，不是 vLLM 推理引擎。

## 默认阅读入口：按月复盘

不必逐份补读 frontier scans。先看[2026 年月度总览](monthly_reviews.md)，再按问题选择[7 月](monthly_signal_2026-07.md)、[8 月](monthly_signal_2026-08.md)或[9 月阶段报告](monthly_signal_2026-09.md)。月报先解释技术主线和工程后果，原记录保留作证据。

本次[GitHub 历史复盘](github_retrospective_2026-07_to_2026-09.md)覆盖 15 库、7–9 月完整事件索引，并对 20 项 PR 深入复核；[历史材料重评](monthly_reviews.md)明确区分原来 Observe 后升级与原来已 Read 的重新提要。

## 知识流转

```text
发现 research/tracking
  ↓
筛选 research/reading_queue
  ↓
阅读 research/{papers,tech_reports,engineering_blogs}
  ↓
理解 research/learning_log
  ↓
沉淀各主线 topics / research/insights
  ↓
验证 practice/experiments，积累 practice/projects / playbooks
```

## 原始扫描与专题索引

<details>
<summary>展开扫描、旧月报与专题文件列表</summary>

- [GitHub 全面补扫 2026-09-22](github_audit_2026-09-22.md)：15 库、374 commits / 383 merged PR / 3 releases；新增 9 组信号，关闭 vLLM/SGLang 索引缺口，附可审计账本。

- [Scan Log](scan_log.md)：每次前沿扫描的账本，记录窗口、来源、accepted / observed 数量和下一次扫描游标。
- [Frontier Scan 2026-09-22](frontier_scan_2026-09-22.md)：最新至 16:52:05，7 Accepted；DSec、Conduit、FP8 RL 稳定性与框架状态契约；原运行时覆盖缺口已由同日 GitHub 专项补扫关闭。
- [Frontier Scan 2026-09-20](frontier_scan_2026-09-20.md)：此前至 10:27:43，8 Accepted；DeepSeek 报告补漏、NVFP4 端到端成本、原子准入、KV 存储背压与隔离。
- [Frontier Scan 2026-09-18](frontier_scan_2026-09-18.md)：此前扫描至 10:42:01，8 条 Accepted；BP/CSBP、GeoMesh、集群碎片与 RL 状态提交。补齐旧来源索引缺口，保留运行时补扫项。
- [Frontier Scan 2026-09-16](frontier_scan_2026-09-16.md)：此前扫描至 14:54:26（Asia/Shanghai），10 条 Accepted；重点是 DeepSeek-V4.1、RL 状态边界、长上下文显存与 kernel。框架历史缺口需回退补扫。
- [Frontier Scan 2026-09-07](frontier_scan_2026-09-07.md)：此前扫描，覆盖到 2026-09-07 10:00:26；arXiv 无新公告批次，重点包括 OpenAI automated research 工业报告、NeMo RL rollout token ledger / vLLM reload refit，以及 Megatron GDP CuTeDSL CP / per-rank RNG resume correctness。
- [Frontier Scan 2026-09-05](frontier_scan_2026-09-05.md)：上一份扫描，覆盖到 2026-09-05 00:21:28；重点包括 AInfer-PD、2400-GPU multi-tenancy characterization、TRL 1M-token CP recipe、slime streaming rollout、NeMo RL HybridEP 与 Headroom-Drift Replay。
- [Frontier Scan 2026-09-01](frontier_scan_2026-09-01.md)：前一份扫描，覆盖到 2026-09-01 11:31:29；重点包括 Anthropic RL environment 治理、HARTS rollout-tree prefix sharing、CE-MoE、verl weight-sync admission gate 与 Megatron variable-length packing。
- [Frontier Scan 2026-08-30](frontier_scan_2026-08-30.md)：上一份扫描，覆盖到 2026-08-30 21:04:46；重点包括 NeMo RL generation-shard recovery、AReaL truncation/GAE correctness、RL-for-LLM 并行性能方法论与长上下文 VPP。
- [Frontier Scan 2026-08-28](frontier_scan_2026-08-28.md)：前一份扫描，覆盖到 2026-08-28 10:24:25；重点包括 OpenAI-Hugging Face incident technical report、psRL、Granite 4.2 异步 GRPO/128K 工业配方与 verl Liger fused PPO kernel。
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
- [Monthly Signal 2026-08](monthly_signal_2026-08.md)：2026 年 8 月工程判断，聚焦 Agent environment、异步正确性/恢复、长上下文调度、状态搬运与 MoE 联合设计。
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


</details>

## Frontier / Monthly / Historical Backfill

| 类型 | 作用 | 时间窗口 | 是否正式收录 |
|---|---|---|---|
| Frontier Scan | 捕捉从上次扫描游标到现在的新前沿信号 | 上次 `Next cursor` 到本次实际扫描结束时刻 | 否，主要是雷达 |
| Monthly Signal | 从当月 frontier/backfill/reading 中筛选高质量信号 | 上月 1 日到月末 | 是，正式沉淀 |
| Historical Backfill | 补录过去已经证明重要、但仓库还没吸收的经典材料 | 按材料原始发布时间月份归档 | 视质量进入队列 |
| Reading Queue | 从 frontier/monthly/backfill 中筛选本周真正要读的 P0/P1 | 当前学习周期 | 是，决定阅读 |

Backfill 不按时间补，按“它能补哪个模型/学习或工程判断缺口”来补。

## 扫描窗口与游标

- Frontier Scan：从 [Scan Log](scan_log.md) 上一次 `Next cursor` 开始，到本次实际扫描结束时刻为止。文件名为 `frontier_scan_YYYY-MM-DD.md`。
- Scan Log：每次真实扫描后必须更新，记录 `Window`、`Sources`、`Accepted`、`Observed`、`Next cursor` 和完整性说明。唯一写入路径是仓库根下的 `research/tracking/scan_log.md`；不要写旧目录或 legacy stub，规则迁移本身不更新游标。
- `Window` 结束时间和 `Next cursor` 不能预填未来时间。白天扫描就写白天的实际时刻；如果精确时刻缺失，下次扫描应回退到最后可确认时间点并去重。
- Monthly Signal：上月 1 日 00:00:00 到上月最后一天 23:59:59，时区 `Asia/Shanghai`。文件名为 `monthly_signal_YYYY-MM.md`。
- Monthly 不重新发现材料，只从当月 frontier scans、backfill、release note 和实际阅读结果中筛选。
- Historical Backfill：按材料原始发布时间月份归档到 `backfill/YYYY-MM.md`，另记录“补录时间”，不和 frontier scan 混。
- Weekly Signal：只保留历史记录，不再维护固定周报模板或 weekly papers 占位文件。需要看最新内容时使用 Frontier Scan，需要正式沉淀时使用 Monthly Signal。

## 记录原则

- 不追求全量，只记录能填补具体模型/学习缺口或改变工程判断的信号。
- 按五条学习主线与上述双通道筛选，不做通用 AI newsletter。
- 不把历史材料混入 frontier scan，避免污染“最新趋势”判断。
- Frontier scan 不强行凑数；0 条 accepted signal 是合法结果。
- Monthly signal 才是正式高质量收录，通常只保留 3 到 5 条，允许更少。
- 每条 accepted signal 必须记录 `Source ID`、`First seen`、`Scan window`，建议补充 `Signal ID`。新增信号同时记录 `Learning track`（Systems / Training / Inference / RL / Embodied，可多选）、`Signal type`（model / algorithm / system）、`Evidence` 和 `Target question`。
- 同一 `Source ID` 可以出现在多个 Watch，但 Accepted 只计一次；follow-up 要说明相对于已收录内容的新证据。
- 每条 accepted signal 及新 paper/report note 的标题、作者、发布时间、关键数字必须与一手来源核对。arXiv ID 能打开不代表核验完成，还须匹配 `citation_title`、`citation_author`、`citation_date` 和摘要/方法；推断须明确标注。
- 每条材料必须有“一句话价值”。
- 每条材料必须给出 `Decision`：`Ignore`、`Observe`、`Read`、`Deep Dive`。
- 每条材料必须给出 `Reason`：为什么做这个决策。
- 每条材料建议标记 `Status`：`NEW`、`READING`、`SUMMARIZED`、`DIGESTED`、`VERIFIED`、`IMPLEMENTED`、`OBSOLETE`。
- 每条材料必须给出建议动作：`进入 P0`、`进入 P1`、`观察`、`忽略`。
- 影响等级用 `★★★★★` 到 `★`，帮助筛选。
- 未来 triage 的目标是当前活跃 P0 不超过 3 条，但不是每次扫描都必须产生 P0；迁移不得自动删减、降级或把既有队列标为 Done。
- tracking 里的内容可以粗糙，但不能没有判断。

## Vendor Watch

每次 frontier scan 和 monthly signal 都必须显式维护 `OpenAI / Anthropic / NVIDIA / DeepSeek Watch`。

- OpenAI / Anthropic / NVIDIA / DeepSeek 是一级关注源：paper、technical report、official docs、engineering blog、model card、weight release、release note、research post 都要进入扫描视野。
- DeepSeek 需要同时检查官方 API changelog 与 Hugging Face organization；重要开放权重更新不一定配套独立博客。
- 四家来源不是自动进入 Accepted；仍按具体模型/学习缺口或系统后果筛选。
- 核心模型厂商的 technical report、model card、工程博客和规模化部署报告按一级工业证据处理：优先读其系统边界和生产证据，同时明确区分公开事实、厂商自报数字与仓库推断。
- 如果材料通过准入与核验，进入 Accepted；如果相关但证据不足，进入 Observed；若两条准入通道均不满足，写明 Rejected / Ignore。
- 如果本次未发现可核验高质量信号，或者来源端点不可用，写明 `Not found / not verifiable in this scan`，避免四家动态在记录里“隐身”。月报只汇总本月已有记录，不为填表重新扫描。
- NVIDIA Training Stack 相关内容优先看 Megatron-Core、Transformer Engine、NCCL、FP8/NVFP4、MoE kernel、distributed checkpointing、scheduling、observability。
- OpenAI / Anthropic 相关内容优先看 training infrastructure、post-training/RL、agent runtime、evaluation/verifier、安全训练、推理/serving、compute/network/cluster 线索。

Hugging Face 作为独立重点生态源，每次扫描还应显式维护 `Hugging Face Watch`：

- 优先扫描 Hugging Face Blog，以及 TRL、Transformers、Accelerate、PEFT、Kernels、LeRobot 等官方 release / docs。
- 重点关注 agentic RL、rollout correctness、training-serving integration、long context、distributed training、独立 inference backend、embodied 学习和 dataset IO。
- 区分 Hugging Face 官方团队文章、厂商联合文章与 community post；来源级别不等于自动 Accepted，仍按两条准入通道筛选。月报复用已筛选记录。

## RL Framework Watch

每次新的 frontier scan 和 monthly signal 必须显式维护 `RL Framework Watch`；月报只总结已有记录。它和厂商 Watch 的分工不同：厂商 Watch 判断技术方向，框架 Watch 判断代码、runtime 和工程能力是否已经发生可用变化。

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

## Inference Systems Watch

每次新的 frontier scan 和 monthly signal 必须显式保留本节，独立于 RL rollout 判断推理系统价值。

- 候选来源包括 vLLM、SGLang、TensorRT-LLM 等的一手 release/docs、重大 PR、测试、benchmark 和生产报告。
- 关注 serving/scheduler、KV cache 与状态搬运、长上下文、多模态输入、kernel/precision、分布式推理、延迟/吞吐/成本及部署正确性。
- 写明实际检查的来源、证据、目标问题和 `Accepted` / `Observed` / `Rejected` / `Not found / not verifiable in this scan`。没有变化或无法核验也是有效结果。
- 月报仅归纳本月已有扫描/阅读记录，不重新发现材料；历史缺口单独标注，不回写旧扫描制造覆盖。

## Embodied Models & Infra Watch

每次新的 frontier scan 和 monthly signal 必须显式保留本节，分别判断模型/学习价值与 infra 后果。

- 候选来源包括 openpi / Physical Intelligence、OpenVLA、NVIDIA GR00T / Isaac Lab、Hugging Face LeRobot、PyTorch / TorchCodec，以及 dataset/IO 社区。这是检查候选，不是工具推荐或自动收录名单。
- 模型通道关注 observation/action 接口、action representation、训练目标、泛化与 evaluation；系统通道关注数据解码与 IO、训练资源/状态、仿真与评估吞吐、部署接口与正确性。
- 每项保留具体来源、实际可核验的证据、`Target question` 和决定；无代码或未复核数字要写证据边界，不将推断当作机制或收益事实。
- 可为零条或写 `Not found / not verifiable in this scan`；月报只汇总已有记录。同一 Source ID 与 NVIDIA/HF 等 Watch 交叉出现时不重复计数。

## Personal Focus Filter

Frontier scan 优先看这些方向：

- 训练系统：Megatron-Core、DeepSpeed、FSDP、PyTorch Distributed、NVIDIA NeMo/Megatron。
- 分布式训练：TP / PP / DP / EP / SP / CP、通信 overlap、rank mapping、拓扑。
- GPU 集群：NCCL、NVLink/NVSwitch、InfiniBand、RoCE、straggler、fault tolerance。
- 显存与状态：ZeRO、FSDP、optimizer state、distributed checkpointing、recovery。
- Kernel 与精度：FlashAttention、Transformer Engine、FP8 / NVFP4、CUTLASS、Grouped GEMM。
- MoE 与大规模训练：expert parallel、load balance、DeepSeekMoE、MegaScale、Llama/DeepSeek/Gemini 训练系统。
- Agentic RL / post-training infra：rollout、verifier/reward、RLHF/GRPO/DAPO 系统、training-serving disaggregation、weight sync、sample freshness。
- 独立 Inference Systems：serving、scheduler、KV/state、long context、multimodal serving、latency/throughput/cost、部署与正确性，不要求先证明影响 rollout。
- Embodied Models & Infra：具体学习缺口驱动的模型机制、action representation、objectives、generalization/evaluation，以及数据/仿真/训练/部署系统。
- Hardware / Systems 学习来源按待解决问题选取；ml-engineering 等外部材料作为来源，不整本镜像，不运行未审查脚本。

以下内容通常拒绝：纯榜单、通用 demo、融资、prompt 技巧、产品新闻，以及既无具体学习价值也无系统后果的材料。不要仅因 embodied 模型/算法尚无 infra 增益就拒绝；也不要仅因贴上 embodied 标签就接受。

## 从 Tracking 到沉淀

- 值得立刻读的内容进入 [P0](../reading_queue/P0.md)。
- 值得以后读的内容进入 [P1](../reading_queue/P1.md)。
- 已读完并形成判断的内容进入共享 `research/learning_log/`。
- 形成观点后进入共享 `research/insights/`。
- 可以实验验证的内容进入 `practice/experiments/`，项目证据进入 `practice/projects/`，生产流程进入 `practice/playbooks/`（此处均为仓库根相对路径）。
- 被消化为可复用知识后进入对应主线的 `topics/`，同步根 [Knowledge Graph](../../KNOWLEDGE_GRAPH.md) 与 [Master Reading List](../MASTER_READING_LIST.md) 的导航关系。
- 进入真实工程实践或生产方案后，状态可以标记为 `IMPLEMENTED`。

## 交付与维护边界

每次 frontier/月报交付都附完整报告链接、一小段中文总体进展与趋势判断，以及每条 Accepted 的来源链接和一句话看点；说明瓶颈、机制或后果，不只重复标题。区分新发表与迟发现材料，标注趋势推断和重要覆盖缺口；没有合格信号就明确写无。

2026-09-22 的自动化路径核查仅覆盖四份本地配置，未发现 ReadBase 硬编码路径匹配；这不能证明其他机器或远程调度器已覆盖。后续维护必须检查实际配置，且扫描游标只写 `research/tracking/scan_log.md`。

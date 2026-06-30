# Tracking

`tracking/` 是 AI Training Infra 的研究雷达。

它不追求完整解读，也不替代 `papers/`、`tech_reports/`、`engineering_blogs/` 或 `topics/`。它的职责是记录三类输入：weekly 负责固定窗口扫描前沿，monthly 负责高质量正式沉淀，historical backfill 负责补录过去已经证明重要但仓库还没吸收的历史精华材料。

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

- [Weekly Signal Report Template](weekly_signal_report_template.md)：每周日高质量信号判断模板，适合 HuggingFace、arXiv、GitHub、厂商博客和 release note。
- [Monthly Signal Report Template](monthly_signal_report_template.md)：每月 1 日输出上个月的高质量正式信号沉淀。
- [Historical Backfill](historical_backfill.md)：补录过去几年已被验证有长期价值、但仓库还没吸收的经典材料。
- [Weekly Papers](weekly_papers.md)：每周值得关注的新论文。
- [Engineering Blogs](engineering_blogs.md)：大厂技术博客和官方文档追踪。
- [Release Notes](release_notes.md)：模型、框架、训练栈发布记录。
- [Infra Trends](infra_trends.md)：训练基础设施技术演进时间线。
- [Agentic RL](agentic_rl.md)：Agentic RL、long-context RL、rollout infra、verifier/reward pipeline 专题追踪。

## Weekly / Monthly / Historical Backfill

| 类型 | 作用 | 时间窗口 | 是否正式收录 |
|---|---|---|---|
| Weekly Signal | 捕捉新出现、可能改变工程判断的前沿信号 | 上周一到上周日 | 否，主要是雷达 |
| Monthly Signal | 从当月 weekly/backfill/reading 中筛选高质量信号 | 上月 1 日到月末 | 是，正式沉淀 |
| Historical Backfill | 补录过去已经证明重要、但仓库还没吸收的经典材料 | 按原始发布时间归档 | 视质量进入队列 |
| Reading Queue | 从 weekly/monthly/backfill 中筛选本周真正要读的 P0/P1 | 当前学习周期 | 是，决定阅读 |

Backfill 不按时间补，按“它能补哪个工程判断缺口”来补。

## 固定时间窗口

- Weekly 统计窗口：上周一 00:00:00 到上周日 23:59:59，时区 `Asia/Shanghai`。每周一生成，文件名为 `weekly_signal_YYYY-Www.md`。
- Monthly 统计窗口：上月 1 日 00:00:00 到上月最后一天 23:59:59，时区 `Asia/Shanghai`。每月 1 日生成，文件名为 `monthly_signal_YYYY-MM.md`。
- Monthly 不重新发现材料，只从当月 weekly、backfill、release note 和实际阅读结果中筛选。
- Historical backfill 按“原始发布时间”归档，另记录“补录时间”，不和 weekly 混。

## 记录原则

- 不追求全量，只记录可能改变工程判断的信号。
- 只抓取当前最关心的 AI Systems / Training Infra / Agentic RL Infra 内容，不做通用 AI newsletter。
- 不把历史材料混入 weekly signal，避免污染“本周趋势”判断。
- Weekly signal 不强行凑数；0 条 accepted signal 是合法结果。
- Monthly signal 才是正式高质量收录，通常只保留 3 到 5 条，允许更少。
- 每条 accepted signal 建议记录 `Signal ID`、`Source ID`、`First seen` 和 `Window`，避免重复收录。
- 每条材料必须有“一句话价值”。
- 每条材料必须给出 `Decision`：`Ignore`、`Observe`、`Read`、`Deep Dive`。
- 每条材料必须给出 `Reason`：为什么做这个决策。
- 每条材料建议标记 `Status`：`NEW`、`READING`、`SUMMARIZED`、`DIGESTED`、`VERIFIED`、`IMPLEMENTED`、`OBSOLETE`。
- 每条材料必须给出建议动作：`进入 P0`、`进入 P1`、`观察`、`忽略`。
- 影响等级用 `★★★★★` 到 `★`，帮助每周筛选。
- P0 不超过 3 条，但不是每周都必须产生 P0。
- tracking 里的内容可以粗糙，但不能没有判断。

## Personal Focus Filter

Weekly 扫描优先看这些方向：

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

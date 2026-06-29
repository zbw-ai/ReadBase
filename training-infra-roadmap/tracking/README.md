# Tracking

`tracking/` 是 AI Training Infra 的研究雷达。

它不追求完整解读，也不替代 `papers/`、`tech_reports/`、`engineering_blogs/` 或 `topics/`。它的职责是持续记录最近值得关注的新论文、工程博客、release note、模型发布和 infra 趋势，然后把少数真正重要的内容推进到阅读队列和工程手册。

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
- [Weekly Papers](weekly_papers.md)：每周值得关注的新论文。
- [Engineering Blogs](engineering_blogs.md)：大厂技术博客和官方文档追踪。
- [Release Notes](release_notes.md)：模型、框架、训练栈发布记录。
- [Infra Trends](infra_trends.md)：训练基础设施技术演进时间线。
- [Agentic RL](agentic_rl.md)：Agentic RL、long-context RL、rollout infra、verifier/reward pipeline 专题追踪。

## 记录原则

- 不追求全量，只记录可能改变工程判断的信号。
- 每条材料必须有“一句话价值”。
- 每条材料必须给出 `Decision`：`Ignore`、`Observe`、`Read`、`Deep Dive`。
- 每条材料必须给出 `Reason`：为什么做这个决策。
- 每条材料建议标记 `Status`：`NEW`、`READING`、`SUMMARIZED`、`DIGESTED`、`VERIFIED`、`IMPLEMENTED`、`OBSOLETE`。
- 每条材料必须给出建议动作：`进入 P0`、`进入 P1`、`观察`、`忽略`。
- 影响等级用 `★★★★★` 到 `★`，帮助每周筛选。
- 每周最多 Top 10 signals，其中最多 3 条进入 P0。
- tracking 里的内容可以粗糙，但不能没有判断。

## 从 Tracking 到沉淀

- 值得立刻读的内容进入 `reading_queue/P0.md`。
- 值得以后读的内容进入 `reading_queue/P1.md`。
- 已读完并形成判断的内容进入 `learning_log/`。
- 形成工程观点后进入 `insights/`。
- 可以实验验证的内容进入 `experiments/`。
- 被消化为可复用工程知识后进入 `topics/`。
- 进入真实工程实践或生产方案后，状态可以标记为 `IMPLEMENTED`。

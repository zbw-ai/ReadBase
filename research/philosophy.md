# Philosophy

[研究入口](README.md) · [首页](../README.md) · [知识地图](../KNOWLEDGE_GRAPH.md)

## North Star

**Build a production-grade understanding of Large-Scale AI Systems through a human-AI maintained Personal Research Operating System.**

目标不是收藏论文，而是建立能够支撑大规模 AI 系统设计、性能优化、生产排障和长期技术判断的工程知识体系。

## Long-Term Scope

硬件／系统、训练、推理、RL 与具身是并列且相互复用的学习方向，实验和排障为它们提供验证。不是先完成 Training，再依次解锁 Inference 和 Agent。

当前学习重点见[首页](../README.md)，具体阅读决策只在[统一队列](reading_queue/README.md)维护。资料、追踪和生命周期统一归 `research/`；技术正文归各 Part，验证与项目归 `practice/`，面试仅按需使用。

这个仓库不是短期学习笔记，而是一个可以持续演进 5 年以上的 Personal Research Operating System。

## What This Repo Is Not

- 不是 paper collection。
- 不是普通 AI notebook。
- 不是只按文件类型堆资料的 PKM。
- 不是追热点的链接仓库。

## What This Repo Is

这是一个持续输入、筛选、消化、验证和复盘的工程研究系统：

```text
Signals
  ↓
Decisions
  ↓
Topics
  ↓
Experiments
  ↓
Playbooks
  ↓
Learning Log
```

## Reading Principle

不是记住论文，而是建立知识图谱。

不是收藏链接，而是形成工程判断。

不是追热点，而是理解演进路线。

任何知识最终都应该回答：

- Why：为什么这个问题重要？
- How：系统是如何实现的？
- Tradeoff：代价和边界是什么？
- Production：真实生产环境会踩什么坑？
- Decision：它是否改变我的工程决策？

## Lifecycle

每条重要资料都应该有生命周期状态：

```text
NEW         刚发现
READING     正在阅读
SUMMARIZED  已形成论文/报告/博客笔记
DIGESTED    已沉淀到 topics 或 insights
VERIFIED    已通过实验或复现验证
IMPLEMENTED 已进入真实工程实践或生产方案
OBSOLETE    已过时或被更好的方案替代
```

`IMPLEMENTED` 是最重要的状态之一。读懂和验证还不够，真正进入工程实践后，知识才完成了最后一段路。

## Completion Standard

**If a paper does not change my engineering judgment, experiment design, or system implementation, I have not really finished reading it.**

一篇重要论文、技术报告或工程博客，最终至少应该产出以下之一：

- 更新一个 `topic`
- 写一篇 `insight`
- 创建一个 `experiment`
- 形成一个 `playbook`
- 改变一个工程 decision

## Decision-First

以后真正值钱的不是“我读过什么”，而是“我为什么决定读它、忽略它、验证它或采用它”。

Tracking 中每条重要信号都应该包含：

```text
Decision: Ignore / Observe / Read / Deep Dive
Reason: 为什么？
```

这会保留当时的技术判断上下文，避免资料变成信息墓地。

未来如果真实工程决策积累足够多，可以新增 `decisions/` 作为 Engineering Decision Records：

```text
decisions/
  2026-07-fsdp-vs-zero.md
  2026-08-why-we-use-context-parallel.md
  2026-09-agent-rollout-scheduler.md
```

现在先不创建空目录，避免在没有真实 decision 前形成形式主义。

## Motto

**Knowledge compounds only when it changes decisions.**

**只有能够改变工程决策的知识，才会产生长期复利。**

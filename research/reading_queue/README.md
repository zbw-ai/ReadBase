# Reading Queue

过去扫描不需要逐篇补读：先走[按季度 / 月度复盘](../tracking/monthly_reviews.md)，每月只选择与当前工作最相关的一两份。历史重评不会自动增加当前 P0，也不把已收录视作已读。

`research/reading_queue/` 是 Systems / Training / Inference / RL / Embodied 共用的筛选层，用来把 [Tracking](../tracking/README.md) 中的信号转化为明确阅读计划，不为每个 Part 复制完整队列。独立的 `reading_inbox/` 仍是个人 intake，不因迁移而自动进入精读队列。

它解决的问题是：tracking 会越来越多，但真正值得精读的材料永远只能是少数。

## 文件

- [P0](P0.md)：本周必须读，数量严格控制。
- [P1](P1.md)：以后值得读，但不进入本周。
- [Done](Done.md)：已读完，并记录去向。

## 规则

- **未来 triage 目标**：当前活跃 P0 不超过 3 条；不是每次扫描都必须新增 P0。现有队列即使超过目标，也要逐项判断，不能用目录迁移自动删减、降级或标为 Done。
- P1 可以更多，但每月复盘一次；一条材料连续一个月没有动作时，重新检查目标问题、时机与证据，再明确决定保留、降级观察或移出，不能把迁移当作清理授权。
- 队列候选记录 `Learning track`（可多选）、`Signal type`（model / algorithm / system）、`Evidence` 和 `Target question`。同一材料跨主线只保留一个队列项，以关联主题表达多重价值。
- 既保留有系统后果的材料，也允许回答具体 embodied 模型/学习缺口的材料；后者无需虚构已证明的 infra 收益。名单、demo 或未经核验榜单不是自动晋级依据。
- Done 必须建立在实际阅读与产出上，写清去向：共享 research note / 对应主线 topic / research insight / practice experiment、project 或 playbook；收录、改路径、历史重评都不等于读完。
- 导航关系改变时更新根 [Knowledge Graph](../../KNOWLEDGE_GRAPH.md) 和共享 [Master Reading List](../MASTER_READING_LIST.md)。

2026-09-22 的结构与政策迁移不调整 `P0.md`、`P1.md`、`Done.md` 的阅读决策或完成状态；这里只制定后续维护规则。

历史候选先读[2025 季度 / 2026 月度复盘](../tracking/monthly_reviews.md)，按当前问题选择一至两篇。本轮只更新候选与证据，不批量加入 P0/P1，也不替用户标记已读。

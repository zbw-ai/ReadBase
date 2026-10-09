# Research｜研究来源与知识生命周期

这里保存论文、工业报告、工程资料及其发现、取舍、精读和消化记录，为各 Part 提供证据，不另起一条按论文数量推进的学习路线。研究雷达是 inbox，正文是工程解释，实践是验证；三者不能互相替代。

## 从哪里开始

1. [研究方法](philosophy.md)：先明确一个材料能改变什么工程判断，而不是只问是否读过。
2. [研究雷达与扫描规则](tracking/README.md)：新信号走 frontier scan，历史材料走 backfill，时间边界以 [scan log](tracking/scan_log.md)为准。
3. [阅读队列](reading_queue/README.md)：按 [P0](reading_queue/P0.md) / [P1](reading_queue/P1.md)决定投入，不在各 Part 重复维护优先级。
4. [阅读总表](MASTER_READING_LIST.md)：进入论文与 technical report；工程实现资料另见[工程博客目录](engineering_blogs/README.md)。
5. [洞察](insights/README.md)与[学习日志](learning_log/README.md)：保存判断变化、未解问题和下一步，而非复制更新公告。

## 从信号到工程判断

| 当前问题 | 证据 / 正文入口 | 验证 / 反馈去向 | 状态规则 |
|---|---|---|---|
| 新材料是否值得读？ | [Tracking](tracking/README.md) → [阅读队列](reading_queue/README.md) | 明确 Decision、Reason 与下一步 | 被发现不等于被接受或读完 |
| 材料解决了什么系统瓶颈？ | [阅读总表](MASTER_READING_LIST.md) · [工程资料](engineering_blogs/README.md) | 回写相关 Part，必要时形成 [Insight](insights/README.md) | note 的完成不自动等于消化 |
| 判断能否被验证？ | 对应 Part 的正文 | [实践入口](../practice/README.md) | 计划不得标为 `VERIFIED` |
| 一个月究竟学到了什么？ | [学习日志](learning_log/README.md) · [月度复盘入口](tracking/monthly_reviews.md) | 调整阅读与实验决策 | 保留原始日期、结果和证据缺口 |

## 继续深入

生命周期沿用 `NEW → READING → SUMMARIZED → DIGESTED → VERIFIED → IMPLEMENTED`，被替代的材料可标记 `OBSOLETE`；不是每条材料都必须走完全程。独立的 [Reading Inbox](../reading_inbox/README.md)保持原用途，不因本次目录迁移被合并；全局结构变化只在[总入口](../README.md)记录。

[返回总入口](../README.md) · [最近更新](../README.md#recent-updates) · [知识图谱](../KNOWLEDGE_GRAPH.md) · [回到 Part I：系统基础](../01-systems/README.md) · [进入 Part VI：工程实践](../practice/README.md)

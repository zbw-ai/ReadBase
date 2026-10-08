# 2025—2026 H1 历史复盘证据包

内容核验批次为 2026-09-22，任务于 2026-10-08 继续整合。阅读正文见[历史补证报告](../../github_history_2025_to_2026_h1.md)，主入口见[季度/月度复盘](../../monthly_reviews.md)。

| 文件 | 保存什么 | 不代表什么 |
|---|---|---|
| [2025 H1](2025-h1-sources.json) | 34 个来源的引用元数据、日期和核验边界 | 全季度论文穷尽检索 |
| [2025 H2](2025-h2-sources.json) | 6 篇 arXiv、3 个官方核心及 Watch 证据 | 所有更新在原季度都已可用 |
| [2026 H1](2026-h1-sources.json) | 24 条来源/版本记录，7 项新增或提高优先级的精选 | 用户已读或原扫描判定被覆盖 |
| [reviewed_prs.json](reviewed_prs.json) | 24 个 PR 的 merge 元数据、改动文件清单、定向 diff 判断 | 审阅所有文件、运行 GPU 测试或生产验收 |
| [selected_release_evidence.json](selected_release_evidence.json) | 7 个具体 release 的历史摘要来源 | 全量 release 索引 |
| [github_manifest.json](github_manifest.json) | 当前索引重建缺口、目标范围和失败原因 | 已关闭 15 库全窗口覆盖缺口 |

## 尚未完成的索引

9/22 的临时 API 原始缓存未保留到本次环境。10/8 重新获取元数据时，多数 GitHub API 请求遇到 TCP/TLS 超时、EOF、连接重置；批量脚本两次自动审批超时。没有用先前中途进度的计数填充最终 CSV，也没有因为少数仓库元数据读取成功就宣布枚举完成。

[rebuild_github_indexes.py](rebuild_github_indexes.py)保存明确的重建方法：固定默认分支 head，按历史 committer / merged_at / published_at 时间过滤；PR 倒序翻页越过下界；release 端点翻到末尾；按上海时区分季度/月份，输出 CSV、计数与 SHA256。它只读 GitHub，写本目录与 `/tmp/readbase-history-recheck/cache`，需要 Python 3、gh 已登录和可用网络。源代码通过语法解析；**本次没有成功运行完整采集流程**。

后续运行成功后，仍需核对分页停止条件、窗口归属、重复项和所有 CSV 列数，再更新正文与导航的覆盖声明。当前报告中保留的历史结论与后续恢复的索引要分开记录核验日期；不要推进 frontier cursor。

## 本次内容检查

季度/月度文档、backfill 和导航的本地链接已检查；JSON 均可解析；24 个 PR 的 Source ID 唯一，文件数量与清单长度匹配。2026 年 1–6 月原 `Window` 以下正文保留，原 scan、scan_log、P0/P1 与个人阅读状态未改。这里的检查是文档完整性检查，不是实验验证。

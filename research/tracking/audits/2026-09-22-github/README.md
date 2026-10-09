# GitHub 补扫证据索引

对应[补扫报告](../../github_audit_2026-09-22.md)。这是元数据与筛选账本，不是逐行源码审计或上游测试结果。

- [manifest.json](manifest.json)：请求范围、分页数量、完成标记与 CSV 校验值。
- [commits.csv](commits.csv)：374 条默认分支提交，含日期、作者、固定 SHA、标题初筛及接纳组。
- [merged_prs.csv](merged_prs.csv)：383 条窗口内 merged PR；base 字段区分默认分支与开发/功能分支，与 commits 有重叠，不能相加当成独立变更。
- [releases.csv](releases.csv)：3 个窗口内 release，标明 prerelease。
- [open_prs.csv](open_prs.csv)：2531 条近期更新的 open PR 元数据；只是观察时状态，不声称逐篇精读，也不把批量元数据更新当新功能。
- [reviewed_prs.csv](reviewed_prs.csv)：57 个重点 PR 的正文、文件目录和选定关键 diff 核验记录；files_received 不是“全部行已读”，patch 字段也不能替代完整 checkout。

原始 API 响应保留于本次本地临时目录，公开仓库保留可重查 URL、SHA、时间与紧凑索引。没有获取私有仓库或执行上游代码。

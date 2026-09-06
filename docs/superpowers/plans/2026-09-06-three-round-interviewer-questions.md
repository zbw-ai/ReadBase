# 三轮技术面反问体系实施计划

**Goal:** 将 `VI.4 建议反问面试官` 改造成按一面、二面、三面分层的现场速查体系。

**Source of truth:** `docs/superpowers/specs/2026-09-06-three-round-interviewer-questions-design.md`

## Task 1：重写 VI.4

- 保留 `vi-questions-to-ask` 锚点。
- 增加三轮总览表和现场使用原则。
- 每轮写入 3 个主问题、2 个备选问题、1 个 30 秒问题。
- 每轮补充正向信号、风险信号和不建议提问项。

## Task 2：验证

- 检查原有控制台链接仍能命中 `VI.4`。
- 检查三轮结构和题目数量符合规格。
- 确认未新增正式题号、未改变全局题目统计。
- 运行 Markdown 本地链接、图片路径和 `git diff --check` 检查。

## Task 3：发布

- 审查最终 diff，只提交 spec、plan 和主面试文档。
- 推送变更到 GitHub `main`。
- 将本地 `main` 快进至同一提交，并保留主工作区中其他未提交内容。

# Monthly Signal Report Template

这个模板用于输出上个月的高质量 AI Systems / AI Training Infra 信号沉淀。

Monthly signal 不是更长的 frontier scan，也不是链接汇总。它只从当月 frontier scans、release note、backfill 和真实阅读结果中筛选少量高价值材料，回答：

> 这个月真正值得改变工程判断的信号是什么？

原则：

- 固定统计窗口：上月 1 日 00:00:00 到上月最后一天 23:59:59，时区 `Asia/Shanghai`。
- 文件名使用月份：`monthly_signal_YYYY-MM.md`。
- Accepted signals 通常 3 到 5 条；可以更少，可以为 0。
- P0 通常不超过 1 到 2 条，P1 通常不超过 3 到 5 条。
- Monthly 不重新发现材料，只汇总当月 frontier scans / backfill / release note / 已读材料。
- 宁缺毋滥。没有高质量判断就明确写无。

---

# Monthly Signal Report, YYYY-MM

- Window: YYYY-MM-01 00:00:00 ~ YYYY-MM-DD 23:59:59
- Timezone: Asia/Shanghai
- Generated at: YYYY-MM-DD
- Report type: monthly quality digest

## 本月核心判断

用 1-3 段说明本月哪些技术方向真正值得进入长期关注。

## Accepted Signals

本节只放高质量内容，可以为空。

### 标题

- Signal ID：YYYY-MM-001
- Source ID：
- First seen：
- 来源窗口：frontier scan / backfill / release note / reading
- 类型：paper / repo / engineering blog / release note / report
- 链接：
- 影响等级：★★★★★ / ★★★★☆ / ★★★☆☆
- Decision：Read / Deep Dive / Observe
- Reason：
- 建议动作：进入 P0 / 进入 P1 / 观察 / 直接沉淀到 topic
- 关联主题：
- 最终应流向：topic / insight / playbook / experiment / note

## P0 / P1 更新

### P0

- 材料：
- 为什么现在必须读：
- 目标产物：

### P1

- 材料：
- 为什么以后值得读：
- 目标产物：

## Observed / Rejected

| 材料 | Decision | 原因 |
|---|---|---|
| 标题 | Observe / Ignore / Backfill | 为什么没有进入 accepted signals |

## 对仓库的影响

- 需要更新的 topic：
- 需要更新的 insight：
- 需要更新的 playbook：
- 需要新增的 experiment：
- 需要进入 historical backfill 的材料：

## 下月关注

- 方向 1：
- 方向 2：
- 方向 3：

# Reading Queue

过去扫描不需要逐篇补读：先走[按月复盘](../tracking/monthly_reviews.md)，每月只选择与当前工作最相关的一两份。历史重评不会自动增加当前 P0，也不把已收录视作已读。

`reading_queue/` 是筛选层，用来把 `tracking/` 中的信号转化为明确阅读计划。

它解决的问题是：tracking 会越来越多，但真正值得精读的材料永远只能是少数。

## 文件

- [P0](P0.md)：本周必须读，数量严格控制。
- [P1](P1.md)：以后值得读，但不进入本周。
- [Done](Done.md)：已读完，并记录去向。

## 规则

- 每周 P0 不超过 3 条。
- P1 可以更多，但每月清理一次。
- Done 里必须写清楚产出：paper note / topic update / insight / experiment / playbook。
- 如果一条材料连续一个月没有动作，要么降级观察，要么删除。

# 按月读懂训练基础设施：2026 年研究复盘

**这是过去扫描的默认阅读入口。** 不需要把每次 frontier scan 逐份补读。每个月先读“回看导读 / 先读这页”，得到问题、机制、工程后果，再按需打开原始记录。7–8 月原月报放在可展开的来源明细里，9 月明确标为阶段报告。

2026-09-22 整理范围：1–6 月既有月报的跨月导读，7–9 月 28 次扫描的重点综合、历史 GitHub 补扫与六篇候选重新评估。原始扫描、原 Accepted、原 Decision 和游标保留；“仓库收录/写笔记”不等于“用户已读/掌握”。

## 一眼看懂连续进展

**趋势推断：** 前半年的并行、显存和通信能力，是后半年 Agentic RL 的底座；7 月重点是辨认 rollout 的真实成本，8 月是异步系统的样本和状态正确性，9 月进一步连接环境质量、数据交付、数值反馈和恢复协议。读月报时始终问：哪个瓶颈移动了，代价转移到哪里，如何证明优化没有改变训练语义？

| 月份 | 主线 | 读完应留下的判断 |
|---|---|---|
| [2026-01](monthly_signal_2026-01.md) | 状态、staleness 与通信的共同边界 | 先理解权重、数据和时间归属 |
| [2026-02](monthly_signal_2026-02.md) | 规模化可靠性与动态并行 | 先确定故障和恢复单位 |
| [2026-03](monthly_signal_2026-03.md) | 局部优化怎样进入调度 | 写清 workload 与测量边界 |
| [2026-04](monthly_signal_2026-04.md) | 并行、通信、精度的联合设计 | 把计算与通信放到同一时间线 |
| [2026-05](monthly_signal_2026-05.md) | 共享数据与网络观测 | 区分重复计算、拥塞和到达偏差 |
| [2026-06](monthly_signal_2026-06.md) | 可组合 rollout 与 kernel 栈 | 明确训练和推理的接口契约 |
| [2026-07](monthly_signal_2026-07.md) | Rollout 成本与长轨迹状态 | 统一成本口径，验证 token/logprob/resume |
| [2026-08](monthly_signal_2026-08.md) | 异步化之后的样本与状态守恒 | 检查队列、消费 frontier、环境和权重发布 |
| [2026-09](monthly_signal_2026-09.md) | 可信经验的生产、交付与恢复（截至 9/22） | 连读 MiMo、DSec 和框架历史复盘 |

## 现在只看三处

1. [8 月复盘](monthly_signal_2026-08.md)：解释为什么异步系统即使不断产出 token，仍可能丢样本、恢复错位或使用旧状态。
2. [9 月阶段报告](monthly_signal_2026-09.md)：把环境生产、DSec、Conduit、低精度与真实框架问题连起来。
3. [7–9 月 GitHub 复盘](github_retrospective_2026-07_to_2026-09.md)：直接看 14 组历史补漏及其工程后果，不需要打开两万条索引记录。

先不扩充 P0，也不要求把 P1 清空。如果要深读，月报中的“只选两份 / 三份”按当前问题选择即可。

## 这次回看修正了什么

| 类别 | 原记录 | 现在怎么处理 | 原因 |
|---|---|---|---|
| BPO / Harness Engineering | 7 月 Observe | Read，进入月报 | 可恢复环境与可靠 kernel 验收已成为当前系统设计能力 |
| B300 field report / Contract-Grade Verifier | 8 月 Observe 或 Read later | Read，进入月报 | 小规模排障与语义验证可直接改变生产判断，不能只按论文规模筛选 |
| Envs-FORGE / Lazy Pod | 8 月已经 Read，但留在观察区 | 保持 Read，提升正文优先级 | 环境生产和延后失败与现在的 Agentic RL 主线连接更清楚 |
| GitHub 历史具体实现 | 原月报未展开 | 14 组 Read 补漏 | 统一成本分母、采样证据、样本守恒、状态与恢复边界 |
| 已知 AReaL checkpoint / NeMo backport | 已见 commit 或同源实现 | 去重，补关联 | PR 链接新出现不等于新技术信号 |
| QEffect / MoSim / 条件性 entropy 缺口 | Observe | 保持 Observe | 有相关性但还不足以改变当前阅读排序，避免事后全部升级 |

重评完整依据：[7 月原始材料](backfill/2026-07.md)、[8 月原始材料](backfill/2026-08.md)。后续报告提供了新的关注理由，不等于替早期材料完成了独立验证。

## 覆盖与证据边界

- GitHub：15 库，2026-07-01 00:00 至 09-22 17:06:16（上海）；11,114 commits、11,728 merged PR、55 releases，集合有重叠。逐页索引完成，重点复核 20 个 PR；不把全量索引称为全量代码审计。
- 论文：重新核验选定 35 篇的标题、作者、日期与摘要，对重评候选进一步读方法/限制；并非对三个月 arXiv 全站穷尽检索。[书目证据](audits/2026-09-22-retrospective/paper_metadata.json)。
- 1–6 月：重新组织已有月报，保留当时 arXiv 截断、厂商页面不可达与 GitHub 覆盖不足，不声称完成这些月份的全量历史 API 补扫。
- 原始发布日期、第一次扫描发现日期、9/22 重新评价日期分开记；跨月发现归材料原始月份，GitHub 合并月按 Asia/Shanghai，9 月未结束。
- 当前是资料整理与工程判断，没有新跑 GPU 实验，没有替用户修改个人已读状态。

## 原始扫描对应关系（仅在需要核对时展开）

扫描文件按执行月分组；涉及跨月来源时按原始发表月进入对应月报，因此文件名月份与归档材料月份不一定相同。

<details>
<summary>2026-07：10 次扫描</summary>

- [2026-07-04](frontier_scan_2026-07-04.md)
- [2026-07-07](frontier_scan_2026-07-07.md)
- [2026-07-08](frontier_scan_2026-07-08.md)
- [2026-07-10](frontier_scan_2026-07-10.md)
- [2026-07-13](frontier_scan_2026-07-13.md)
- [2026-07-20](frontier_scan_2026-07-20.md)
- [2026-07-22](frontier_scan_2026-07-22.md)
- [2026-07-24](frontier_scan_2026-07-24.md)
- [2026-07-27](frontier_scan_2026-07-27.md)
- [2026-07-28](frontier_scan_2026-07-28.md)

</details>

<details>
<summary>2026-08：11 次扫描</summary>

- [2026-08-03](frontier_scan_2026-08-03.md)
- [2026-08-09](frontier_scan_2026-08-09.md)
- [2026-08-12](frontier_scan_2026-08-12.md)
- [2026-08-14](frontier_scan_2026-08-14.md)
- [2026-08-17](frontier_scan_2026-08-17.md)
- [2026-08-18](frontier_scan_2026-08-18.md)
- [2026-08-20](frontier_scan_2026-08-20.md)
- [2026-08-24](frontier_scan_2026-08-24.md)
- [2026-08-26](frontier_scan_2026-08-26.md)
- [2026-08-28](frontier_scan_2026-08-28.md)
- [2026-08-30](frontier_scan_2026-08-30.md)

</details>

<details>
<summary>2026-09：7 次扫描</summary>

- [2026-09-01](frontier_scan_2026-09-01.md)
- [2026-09-05](frontier_scan_2026-09-05.md)
- [2026-09-07](frontier_scan_2026-09-07.md)
- [2026-09-16](frontier_scan_2026-09-16.md)
- [2026-09-18](frontier_scan_2026-09-18.md)
- [2026-09-20](frontier_scan_2026-09-20.md)
- [2026-09-22](frontier_scan_2026-09-22.md)

</details>

更早月份及一次性 Historical Audit 沿用原月报的引用；[旧周报](weekly_signal_2026-W26.md)仅保留审计意义。[扫描账本](scan_log.md)继续用于下一次增量边界。

## 与知识库的连接

月报中的判断进入 [Agentic RL](../topics/agentic_rl.md#monthly-retrospective-invariants)、[状态边界实验设计](../experiments/rl_state_boundaries.md#retrospective-test-cases)和[慢步排障](../playbooks/slow_step_debug.md#retrospective-observability)。它们是后续验证入口，不是已完成实验。[Master Reading List](../MASTER_READING_LIST.md)与[Knowledge Graph](../KNOWLEDGE_GRAPH.md)均以本页作为月度导航。

# Weekly Signal Report Template

这个模板用于每周日输出高质量 AI Systems / AI Training Infra 前沿信号判断。它适合 HuggingFace Trending、arXiv、GitHub Trending、NVIDIA/Meta/Google/Microsoft/OpenAI/Anthropic/DeepSeek 等工程博客和 release note。

不要把它写成新闻摘要，也不要把它写成补课清单。它只回答：

> 本周有哪些新出现的技术信号，可能改变未来 3 到 12 个月的工程判断？

原则：

- 不强行凑数。0 条 accepted signal 是合法结果。
- 只收最近 7 到 14 天的材料；最多放宽到 30 天，但必须解释为什么现在是前沿信号。
- 只有高质量、可核验、会改变工程判断的材料才进入 `Accepted Frontier Signals`。
- P0 不超过 3 条，但不是每周都必须有 P0。
- 历史重要材料进入 [Historical Backfill](historical_backfill.md)，不要混入 weekly signal。
- 每条 accepted signal 必须给出 `Decision` 和 `Reason`。

---

# Weekly Signal Report, YYYY-WW

## 本周核心判断

用 1 段话总结本周是否有值得采纳的前沿信号。不超过 150 字。

如果没有合格信号，直接写：

> 本周没有收录合格的 frontier weekly signal。

## 筛选标准

说明本周采用的筛选口径。默认标准：

- 最近 7 到 14 天发布、更新或引发明确讨论；最多放宽到 30 天。
- 有可核验来源：paper、repo、release note、official blog、benchmark 或技术报告。
- 不是单纯“主题相关”，而是可能改变工程判断。
- 能说明它影响哪个系统边界：parallelism、checkpoint、FP8、MoE、rollout、serving/training interface、scheduler、NCCL、observability。

## Accepted Frontier Signals

本节可以为空。不要强行填满。

### 标题

- 来源：
- 类型：paper / model / engineering blog / release note / repo / report
- 链接：
- 发布时间：
- 影响等级：★★★★★ / ★★★★☆ / ★★★☆☆
- Decision：Ignore / Observe / Read / Deep Dive
- Reason：
- Status：NEW / READING / SUMMARIZED / DIGESTED / VERIFIED / IMPLEMENTED / OBSOLETE
- 建议动作：进入 P0 / 进入 P1 / 观察 / 忽略
- 预计阅读：30min / 1h / 2h / 4h
- 关联主题：

正文写 1-3 段，解释为什么它是“前沿信号”，不是为什么它“重要”。

重点回答：

- 它改变了哪个系统约束？
- 它是否暴露新的训练/推理瓶颈？
- 它是否说明某个方向从 paper 走向 production？
- 它会影响哪些主题：MoE、FP8、context parallel、checkpoint、rollout、scheduler、NCCL？

## Deferred / Rejected Candidates

记录被拒绝或延后的候选，保持 decision 可追溯。

| 材料 | Decision | 原因 |
|---|---|---|
| 标题 | Observe / Ignore / Historical Backfill | 为什么没有进入 accepted signals |

## Reading Queue Updates

- [ ] 加入 `reading_queue/P0.md`：
- [ ] 加入 `reading_queue/P1.md`：
- [ ] 仅观察：
- [ ] 转入 `historical_backfill.md`：

## 本周观察

用 1-2 段写自己的判断。这里最重要。

可以回答：

- 本周有没有真正值得采纳的前沿信号？
- 哪些内容虽然重要，但属于 historical backfill？
- 哪些信号证据不够，应继续观察？
- 哪些 accepted signal 值得进入 P0？

## 下一步动作

- [ ] 需要阅读：
- [ ] 需要更新的 topic：
- [ ] 需要新增的 engineering blog 笔记：
- [ ] 需要做实验验证的方向：

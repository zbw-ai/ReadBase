# Weekly Signal Report Template

这个模板用于每周日输出高质量 AI Systems / AI Training Infra 信号判断。它适合 HuggingFace Trending、arXiv、GitHub Trending、NVIDIA/Meta/Google/Microsoft/OpenAI/Anthropic/DeepSeek 等工程博客和 release note。

不要把它写成新闻摘要。它应该回答：本周有哪些东西可能改变工程决策？

原则：

- 每周最多记录 Top 10 signals。
- 最多 3 条进入 `reading_queue/P0.md`。
- 每条必须给出 `Decision` 和 `Reason`。
- 数量在精不在多。

---

# Weekly Signal Report, YYYY-WW

## 本周核心信号

用 1 段话总结本周最重要的趋势，不超过 150 字。

示例写法：

> MoE 正在从模型创新变成训练/推理基础设施的默认约束；agentic 能力不再是 demo 卖点，而开始反向影响模型结构、上下文长度、rollout 系统和评估方式。

## Top 3: 进入 P0

### 项目 / 论文 / 博客标题

- 来源：
- 类型：paper / model / engineering blog / release note / repo / report
- 链接：
- 影响等级：★★★★★
- Decision：Ignore / Observe / Read / Deep Dive
- Reason：
- Status：NEW / READING / SUMMARIZED / DIGESTED / VERIFIED / IMPLEMENTED / OBSOLETE
- 建议动作：进入 P0 / 进入 P1 / 观察 / 忽略
- 预计阅读：30min / 1h / 2h / 4h
- 关联主题：
  - `topics/...`
  - `engineering_blogs/...`
  - `tech_reports/...`

正文写 1-3 段，解释为什么它值得关注。重点不是复述发布说明，而是提炼工程信号：

- 它改变了哪个系统约束？
- 它是否暴露了新的训练/推理瓶颈？
- 它是否说明某个方向从 paper 走向 production？
- 它会影响哪些主题：MoE、FP8、context parallel、checkpoint、rollout、scheduler、NCCL？

## Top 10 Signals

### 1. 标题

- 来源：
- 类型：
- 链接：
- 影响等级：★★★★
- Decision：Ignore / Observe / Read / Deep Dive
- Reason：
- Status：
- 建议动作：
- 关联主题：

一句话价值：

### 2. 标题

- 来源：
- 类型：
- 链接：
- 影响等级：★★★
- Decision：
- Reason：
- Status：
- 建议动作：
- 关联主题：

一句话价值：

### 3. 标题

- 来源：
- 类型：
- 链接：
- 影响等级：★★★
- Decision：
- Reason：
- Status：
- 建议动作：
- 关联主题：

一句话价值：

## 本周观察

用 1-2 段写自己的判断。这里最重要。

可以回答：

- 本周的信号说明哪个方向在加速？
- 哪些技术正在从 paper 进入可运行系统？
- 哪些能力正在从模型卖点变成基础设施约束？
- 哪些内容值得进入 `reading_queue/P0.md`？

## 下一步动作

- [ ] 加入 `reading_queue/P0.md`：
- [ ] 加入 `reading_queue/P1.md`：
- [ ] 需要更新的 topic：
- [ ] 需要新增的 engineering blog 笔记：
- [ ] 需要做实验验证的方向：

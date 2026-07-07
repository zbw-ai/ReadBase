# Weekly Signal Report Template

> 兼容说明：默认日常扫描请使用 [Frontier Scan Template](frontier_scan_template.md) 和 [Scan Log](scan_log.md)。本模板只在确实需要按自然周回顾时使用，例如固定周报、审计某一周是否漏扫、或保留旧 weekly 记录。

这个模板用于每周一输出上一周的高质量 AI Systems / AI Training Infra 前沿信号判断。它适合 HuggingFace Trending、arXiv、GitHub Trending、NVIDIA/Meta/Google/Microsoft/OpenAI/Anthropic/DeepSeek 等工程博客和 release note。

不要把它写成新闻摘要，也不要把它写成补课清单。它只回答：

> 本周有哪些新出现的技术信号，可能改变未来 3 到 12 个月的工程判断？

这个 weekly signal 不是通用 AI newsletter。它只抓取当前最关心的 AI Systems / Training Infra / Agentic RL Infra 内容。

原则：

- 不强行凑数。0 条 accepted signal 是合法结果。
- 固定统计窗口：上周一 00:00:00 到上周日 23:59:59，时区 `Asia/Shanghai`。
- 文件名使用 ISO 周编号：`weekly_signal_YYYY-Www.md`，不要使用生成日期命名。
- 只收最近 7 到 14 天的材料；最多放宽到 30 天，但必须解释为什么现在是前沿信号。
- 只有高质量、可核验、会改变工程判断的材料才进入 `Accepted Frontier Signals`。
- P0 不超过 3 条，但不是每周都必须有 P0。
- 历史重要材料进入 [Historical Backfill](historical_backfill.md) 或 `backfill/YYYY-MM.md`，不要混入 weekly signal。
- 每条 accepted signal 必须给出 `Decision` 和 `Reason`。

## Personal Focus Filter

扫描时优先关注以下方向。只有命中这些方向，才有资格进入 accepted signals 或 observed candidates。

### P0 Focus

- 大模型训练系统：Megatron-Core、DeepSpeed、FSDP、PyTorch Distributed、NVIDIA NeMo/Megatron。
- 分布式训练：TP / PP / DP / EP / SP / CP、并行策略组合、rank mapping、通信 overlap。
- GPU 集群与网络：NCCL、NVLink/NVSwitch、InfiniBand、RoCE、straggler、故障恢复。
- 显存与状态管理：ZeRO、FSDP、optimizer state、activation checkpointing、distributed checkpointing、checkpoint recovery。
- Kernel 与精度：FlashAttention、FlashAttention variants、Transformer Engine、FP8 / NVFP4、CUTLASS、Grouped GEMM。
- MoE 与超大规模训练：expert parallel、router/load balance、DeepSeekMoE、MegaScale、Llama/DeepSeek/Gemini 训练系统报告。
- Agentic RL / post-training infra：rollout、verifier/reward pipeline、RLHF/GRPO/DAPO 系统、training-serving disaggregation、weight sync、sample freshness。

### P1 Focus

- 推理系统中会反向影响训练或 rollout 的内容：vLLM、SGLang、TensorRT-LLM、KV cache、prefix cache、serving/training interface。
- Evaluation / benchmark infra 中会改变训练闭环的内容：自动评测、verifier、reward model、agent evaluation pipeline。
- 数据管线、合成数据、过滤、去重，如果它们直接影响大规模训练或 post-training 系统。

### Usually Reject

- 只有模型榜单提升、但没有训练/推理系统细节的模型发布。
- 纯应用论文、领域数据集、prompt 技巧、产品新闻。
- 纯算法改进但无法连接到 rollout、scheduler、checkpoint、parallelism、kernel、serving/training interface 的材料。
- 没有可核验来源的社交媒体传闻。

宁可漏掉泛 AI 热点，也不要污染这个雷达。

---

---

# Weekly Signal Report, YYYY-WW

- Window: YYYY-MM-DD 00:00:00 ~ YYYY-MM-DD 23:59:59
- Timezone: Asia/Shanghai
- Generated at: YYYY-MM-DD
- Report type: weekly frontier radar

## 本周核心判断

用 1 段话总结本周是否有值得采纳的前沿信号。不超过 150 字。

如果没有合格信号，直接写：

> 本周没有收录合格的 frontier weekly signal。

## 筛选标准

说明本周采用的筛选口径。默认标准：

- 必须命中 `Personal Focus Filter`。
- 最近 7 到 14 天发布、更新或引发明确讨论；最多放宽到 30 天。
- 有可核验来源：paper、repo、release note、official blog、benchmark 或技术报告。
- 不是单纯“主题相关”，而是可能改变工程判断。
- 能说明它影响哪个系统边界：parallelism、checkpoint、FP8、MoE、rollout、serving/training interface、scheduler、NCCL、observability。

## Accepted Frontier Signals

本节可以为空。不要强行填满。

### 标题

- Signal ID：YYYY-Www-001
- Source ID：arxiv:xxxx.xxxxx / github:org/repo@tag / blog:vendor/slug / hf:org/model
- Focus Match：P0 Focus / P1 Focus
- 来源：
- 类型：paper / model / engineering blog / release note / repo / report
- 链接：
- 发布时间：
- First seen：
- Window：YYYY-WW
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

| 材料 | Source ID | Focus Match | Decision | 原因 |
|---|---|---|---|---|
| 标题 | arxiv:xxxx.xxxxx | P0 / P1 / Out of Scope | Observe / Ignore / Historical Backfill | 为什么没有进入 accepted signals |

## Reading Queue Updates

- [ ] 加入 `reading_queue/P0.md`：
- [ ] 加入 `reading_queue/P1.md`：
- [ ] 仅观察：
- [ ] 转入 `historical_backfill.md`：

## 去重记录

- 本周新增 Source ID：
- Follow-up Source ID：
- 与历史 backfill 重复但未收录：

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

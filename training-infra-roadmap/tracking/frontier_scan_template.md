# Frontier Scan Template

这个模板用于灵活执行的最新前沿扫描。它替代固定 weekly 作为日常扫描入口。

它只回答：

> 从上一次扫描游标到现在，有没有新出现、可核验、会改变工程判断的 AI Systems / Training Infra / Agentic RL Infra 信号？

## 使用原则

- 扫描窗口不强行按自然周。窗口从 [Scan Log](scan_log.md) 的上一条 `Next cursor` 开始，到本次扫描结束时间为止。
- 文件名使用生成日期：`frontier_scan_YYYY-MM-DD.md`。
- 每次扫描必须更新 [Scan Log](scan_log.md)。
- 可以 0 条 accepted signal；宁缺毋滥。
- 不把历史经典材料塞进 frontier scan。重要但不新的材料进入 [Historical Backfill](historical_backfill.md) 或 `backfill/YYYY-MM.md`。
- 不硬选 Top 3 / Top 10。只有真正改变工程判断的材料才进入 accepted。
- 每条 accepted signal 必须有 `Source ID`、`First seen`、`Scan window`、`Decision`、`Reason`。
- 如果来源扫描不完整，例如 arXiv API rate limit、GitHub 不可访问、博客站点超时，必须写入“扫描完整性”。

## Focus Filter

优先扫描：

- Agentic RL / post-training infra：rollout、verifier/reward、training-serving disaggregation、weight sync、sample freshness、trajectory store、sandbox。
- Long-context training / inference infra：context parallel、attention IO、KV cache、chunked prefill、prefix cache、长序列数据管线。
- Training stack：Megatron-Core、DeepSpeed、FSDP、PyTorch Distributed、NeMo、Transformer Engine。
- Distributed systems：TP / PP / DP / EP / SP / CP、通信 overlap、rank mapping、NCCL、NVLink/NVSwitch、IB/RoCE、straggler。
- Memory / state：ZeRO、FSDP、activation checkpointing、distributed checkpointing、fault tolerance、elastic recovery。
- Kernel / precision：FlashAttention、FP8 / NVFP4、CUTLASS、Grouped GEMM、MoE kernel。
- Inference infra if it affects RL rollout：vLLM、SGLang、TensorRT-LLM、serving/training interface。

通常拒绝：

- 纯模型榜单、没有系统细节的模型发布。
- 通用 AI 产品新闻、prompt 技巧、应用论文、领域数据集。
- 纯算法改进但无法连接到 rollout、scheduler、parallelism、checkpoint、kernel、serving/training interface 的材料。
- 不能核验来源的社交媒体传闻。

---

# Frontier Scan, YYYY-MM-DD

- Previous scan:
- Window:
- Timezone: Asia/Shanghai
- Generated at:
- Report type: flexible frontier scan
- Sources scanned:
- Scan completeness:

## 本次核心判断

用 1 段话总结本次扫描是否有值得采纳的前沿信号。不超过 150 字。

如果没有合格信号，直接写：

> 本次扫描没有收录合格的 frontier signal。

## Accepted Frontier Signals

本节可以为空。不要强行填满。

### 标题

- Signal ID：YYYY-MM-DD-001
- Source ID：arxiv:xxxx.xxxxx / github:org/repo@tag / blog:vendor/slug / hf:org/model
- First seen：
- Scan window：
- Focus Match：P0 Focus / P1 Focus
- 来源：
- 类型：paper / repo / engineering blog / release note / report / model
- 链接：
- 发布时间：
- 影响等级：★★★★★ / ★★★★☆ / ★★★☆☆
- Decision：Ignore / Observe / Read / Deep Dive
- Reason：
- Status：NEW / READING / SUMMARIZED / DIGESTED / VERIFIED / IMPLEMENTED / OBSOLETE
- 建议动作：进入 P0 / 进入 P1 / 观察 / 忽略 / 转入 backfill
- 预计阅读：30min / 1h / 2h / 4h
- 关联主题：

正文写 1-3 段，解释为什么它是“前沿信号”，不是为什么它只是“主题相关”。

重点回答：

- 它改变了哪个系统约束？
- 它是否暴露新的训练/推理瓶颈？
- 它是否说明某个方向从 paper 走向 production？
- 它影响哪些主题：MoE、FP8、context parallel、checkpoint、rollout、scheduler、NCCL、observability？

## Observed / Rejected Candidates

记录被拒绝、延后或仅观察的候选，保持 decision 可追溯。

| 材料 | Source ID | Focus Match | Decision | 原因 |
|---|---|---|---|---|
| 标题 | arxiv:xxxx.xxxxx | P0 / P1 / Out of Scope | Observe / Ignore / Backfill | 为什么没有进入 accepted signals |

## Reading Queue Updates

- [ ] 加入 `reading_queue/P0.md`：
- [ ] 加入 `reading_queue/P1.md`：
- [ ] 仅观察：
- [ ] 转入 `tracking/backfill/YYYY-MM.md`：

## 去重记录

- 本次新增 Source ID：
- Follow-up Source ID：
- 与历史 backfill 重复但未收录：

## 扫描完整性

- 已扫描来源：
- 未完整扫描来源：
- 已知盲区：
- 下次优先补扫：

## 下一步动作

- [ ] 更新 [Scan Log](scan_log.md)
- [ ] 需要阅读：
- [ ] 需要更新的 topic：
- [ ] 需要新增的 engineering blog / paper / report note：
- [ ] 需要做实验验证的方向：

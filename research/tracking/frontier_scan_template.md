# Frontier Scan Template

这个模板用于灵活执行的最新前沿扫描。它替代固定 weekly 作为日常扫描入口。

它只回答：

> 从上一次扫描游标到现在，有没有新出现、可核验、能填补具体模型/学习缺口或改变系统判断的 Systems / Training / Inference / RL / Embodied 信号？

## 使用原则

- 扫描窗口不强行按自然周。窗口从 [Scan Log](scan_log.md) 的上一条 `Next cursor` 开始，到本次实际扫描结束时刻为止。
- `Window` 的结束时间必须是已经实际扫描过的时间点，例如 `2026-07-07 14:30`。不要在白天扫描时把结束时间写成当天 `23:59`，否则会假装覆盖了尚未发生的文章。
- `Next cursor` 必须等于本次实际扫描结束时刻。若当次没有记录精确结束时刻，下次扫描应回退到最后一个可确认时间点，宁可重复观察，也不要漏扫。
- 文件名使用生成日期：`frontier_scan_YYYY-MM-DD.md`。
- 每次真实扫描必须更新 [Scan Log](scan_log.md)，唯一写入目标是仓库根下的 `research/tracking/scan_log.md`，不写旧路径或 legacy stub。
- 可以 0 条 accepted signal；宁缺毋滥。
- 不把历史经典材料塞进 frontier scan。重要但不新的材料进入 [Historical Backfill](historical_backfill.md) 或 `backfill/YYYY-MM.md`。
- 不硬选 Top 3 / Top 10。候选必须满足具体 embodied 模型/学习价值或系统后果之一，再做证据核验与收录判断。
- 每条 accepted signal 必须有 `Source ID`、`First seen`、`Scan window`、`Decision`、`Reason`、`Learning track`（可多选）、`Signal type`、`Evidence`、`Target question`。
- 核验每条 Accepted 的标题、作者、发布时间和关键数字；arXiv 必须匹配 `citation_title`、`citation_author`、`citation_date` 及摘要/方法，而非只确认 ID 存在。标注厂商自报、未核验数字与推断。
- 同一 Source ID 跨 Watch 引用只计一次 Accepted；Watch 的 `Accepted / Observed / Rejected / Not found / not verifiable in this scan` 是筛选状态，不替代材料的阅读 `Decision`。
- 扩展范围自 2026-09-22 起约束未来扫描。模板更新不是执行扫描，不推进 cursor，也不重写历史记录；新主线历史缺口单独标注，旧材料按原始月份 backfill。
- 如果来源扫描不完整，例如 arXiv API rate limit、GitHub 不可访问、博客站点超时，必须写入“扫描完整性”。

## 返回用户时的摘要

每次交付都包含以下内容，不只返回完整报告链接或 Git 发布状态：

1. **整体进展与趋势**：用一小段中文交代本轮新增、延续或补扫的重点，以及它们共同反映的工程方向；区分来源事实与趋势推断，简述影响判断的覆盖缺口。没有合格信号时直接说明，不强造趋势。
2. **逐篇一句话看点**：每篇 Accepted 文章或信号附来源链接，用一句话说明学习缺口、关键机制、瓶颈或工程后果，不只复述标题；补扫旧材料明确标记。
3. **完整报告**：附报告链接，详细证据、数值边界与观察项留在报告内。

## Focus Filter

优先扫描：

- Agentic RL / post-training infra：rollout、verifier/reward、training-serving disaggregation、weight sync、sample freshness、trajectory store、sandbox。
- Long-context training / inference infra：context parallel、attention IO、KV cache、chunked prefill、prefix cache、长序列数据管线。
- Training stack：Megatron-Core、DeepSpeed、FSDP、PyTorch Distributed、NeMo、Transformer Engine。
- Distributed systems：TP / PP / DP / EP / SP / CP、通信 overlap、rank mapping、NCCL、NVLink/NVSwitch、IB/RoCE、straggler。
- Memory / state：ZeRO、FSDP、activation checkpointing、distributed checkpointing、fault tolerance、elastic recovery。
- Kernel / precision：FlashAttention、FP8 / NVFP4、CUTLASS、Grouped GEMM、MoE kernel。
- Independent Inference Systems：vLLM、SGLang、TensorRT-LLM、serving/scheduler、KV/state、multimodal serving、latency/throughput/cost、部署正确性；不要求先影响 RL rollout。
- RL framework runtime：AReaL、verl、slime、ROLL、OpenRLHF、NeMo RL，以及具备真实代码和可运行训练链路的新框架；重点看 release 与重大架构 PR。
- Embodied Models & Infra：inputs/outputs、action representations、objectives、generalization/evaluation，以及数据 IO、仿真、训练状态与部署链路。
- Hugging Face ecosystem：Hugging Face Blog、TRL、Transformers、Accelerate、PEFT、Kernels、LeRobot 及其与 training / inference / RL / embodied 的集成。
- Hardware / Systems 来源按目标问题选取；ml-engineering 等材料作为来源，不整本镜像、不执行未审查脚本。

准入通道：一是回答当前 embodied 模型/学习缺口，即使尚无已证明 infra 增益；二是存在性能、正确性、状态、IO 或部署等系统后果。分别写清价值和证据，不强行为模型笔记推断 infra 收益。MLLM 指 multimodal large language model，不能与 vLLM 推理引擎混写。

通常拒绝：

- 纯榜单、通用 demo、融资、无具体学习价值或系统细节的发布。
- 通用 AI 产品新闻、prompt 技巧，以及不能回答当前问题的应用/数据材料。
- 两条准入通道都不满足的算法改进；不能仅因 embodied 模型机制尚无 infra 增益而拒绝。
- 不能核验来源的社交媒体传闻。

无代码的来源可以提供已公开模型机制，但不能写成已可运行或可复现；未经验证的效果与性能数字必须标明证据限制。

---

# Frontier Scan, YYYY-MM-DD

- Previous scan:
- Window:
- Timezone: Asia/Shanghai
- Generated at:
- Report type: flexible frontier scan
- Sources scanned:
- Scan completeness:
- Coverage policy: 2026-09-22 expanded scope; actual coverage documented below

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
- Learning track：Systems / Training / Inference / RL / Embodied（可多选）
- Signal type：model / algorithm / system
- Target question：当前要填补的具体学习缺口或系统问题
- Evidence：一手方法/模型卡、代码/测试、可复核 benchmark、生产报告；另标厂商自报、未验证部分与推断
- Focus Match：P0 Focus / P1 Focus
- 来源：
- 类型：paper / repo / engineering blog / release note / report / model
- 链接：
- 作者 / 发布主体：
- 发布时间：
- 影响等级：★★★★★ / ★★★★☆ / ★★★☆☆
- Decision：Ignore / Observe / Read / Deep Dive
- Reason：
- Status：NEW / READING / SUMMARIZED / DIGESTED / VERIFIED / IMPLEMENTED / OBSOLETE
- 建议动作：进入 P0 / 进入 P1 / 观察 / 忽略 / 转入 backfill
- 预计阅读：30min / 1h / 2h / 4h
- 关联主题：
- 一句话价值：

正文写 1-3 段，解释为什么它是“前沿信号”，不是为什么它只是“主题相关”。

重点回答：

- 它回答了哪个模型/学习缺口，或改变了哪个系统约束？
- 如果是 embodied 模型机制：inputs/outputs、action representation、objective、generalization/evaluation 的哪一项有新信息？
- 它是否暴露新的训练/推理瓶颈？
- 它是否说明某个方向从 paper 走向 production？
- 它影响哪些主题：MoE、FP8、context parallel、checkpoint、rollout、scheduler、NCCL、observability？

## Observed / Rejected Candidates

记录被拒绝、延后或仅观察的候选，保持 decision 可追溯。

| 材料 | Source ID | Focus Match | Decision | 原因 |
|---|---|---|---|---|
| 标题 | arxiv:xxxx.xxxxx | P0 / P1 / Out of Scope | Observe / Ignore / Backfill | 为什么没有进入 accepted signals |

## OpenAI / Anthropic / NVIDIA / DeepSeek Watch

四家一手来源必须显式出现。不要自动收录；填写实际检查的链接、证据和结果，不把来源名单当作已扫描证明。

| Vendor | Sources checked / Evidence | Source ID | Watch decision | 结果 / 覆盖限制 |
|---|---|---|---|---|
| OpenAI | official blog / research / docs / reports 的实际链接与证据 | 如有 | Accepted / Observed / Rejected / Not found / not verifiable in this scan | 本次发现或限制 |
| Anthropic | official blog / research / docs / reports 的实际链接与证据 | 如有 | Accepted / Observed / Rejected / Not found / not verifiable in this scan | 本次发现或限制 |
| NVIDIA | technical blog / docs / developer posts / reports 的实际链接与证据 | 如有 | Accepted / Observed / Rejected / Not found / not verifiable in this scan | 本次发现或限制 |
| DeepSeek | official API changelog 与官方 Hugging Face organization 均检查；附 reports/weight releases 等证据 | 如有 | Accepted / Observed / Rejected / Not found / not verifiable in this scan | 哪些机制公开，哪些仍不可核验 |

## Hugging Face Watch

Hugging Face Blog 与核心框架 release 必须显式出现，但不自动进入 Accepted。

| Sources checked / Evidence | Source ID | Watch decision | 结果 / 覆盖限制 |
|---|---|---|---|
| Hugging Face Blog / TRL / Transformers / Accelerate / PEFT / Kernels / LeRobot 的实际链接与证据 | 如有 | Accepted / Observed / Rejected / Not found / not verifiable in this scan | training / inference / RL / embodied / dataset IO 信号；标明官方团队、厂商联合或 community post |

## RL Framework Watch

核心检查 AReaL、verl、slime、ROLL、OpenRLHF、NeMo RL，并动态加入有真实实现和证据的新框架。只跟踪正式 release 与重大 PR，不罗列普通 commit。

| Framework | Release / PR / Source ID | 子系统 | 核心变化 / 工程维度 | 证据 | 对 AReaL 的参考 | Watch decision |
|---|---|---|---|---|---|---|
| AReaL / verl / slime / ROLL / OpenRLHF / NeMo RL / emerging（实际扫描逐项记录） | tag / PR / Source ID / Not found | rollout / training / scheduler / weight sync / data/trajectory path / checkpoint/recovery / inference backend | 行为变化及性能/显存/稳定性/正确性/运维后果 | release note / diff / test / benchmark / production report | 可复用设计、需验证或无直接关系 | Accepted / Observed / Rejected / Not found / not verifiable in this scan |

重大 PR 至少满足一项：改变进程或资源拓扑、训练与推理解耦方式、调度语义、权重同步、sample freshness、trajectory 数据流、并行/显存策略、checkpoint/recovery、核心 backend 或公开性能/正确性边界。PR 规模大不等于信号重要。

## Inference Systems Watch

独立判断推理价值，不把本节限于 rollout backend。候选来源是 vLLM、SGLang、TensorRT-LLM 等官方 release/docs、重大 PR、测试、benchmark 和生产报告；名单不是自动推荐或收录。

| Source / Source ID | 实际检查的来源与 Evidence | Target question / 系统后果 | Watch decision | 限制 / 下一步 |
|---|---|---|---|---|
| runtime / paper / production report | 具体链接；机制、代码、测试与数字边界 | serving/scheduler、KV/state、long context、multimodal、kernel/precision、latency/throughput/cost、部署正确性 | Accepted / Observed / Rejected / Not found / not verifiable in this scan | 可为零条；未检查或不可验证要写明 |

## Embodied Models & Infra Watch

候选来源：openpi / Physical Intelligence、OpenVLA、NVIDIA GR00T / Isaac Lab、Hugging Face LeRobot、PyTorch / TorchCodec、dataset/IO 社区。这是待核验来源，不是推荐或自动 Accepted。模型/学习通道不要求先证明 infra 增益；系统后果另列证据。

| Source / Source ID | 实际检查的来源与 Evidence | Learning value / Target question | System consequence（如有） | Watch decision / 限制 |
|---|---|---|---|---|
| paper / model / repo / report / community source | 具体链接；方法、代码、测试或公开评估，区分社区与官方 | inputs/outputs、action representation、objective、generalization/evaluation | 数据 IO、仿真/训练/部署机制；无证据就写未披露或待验证 | Accepted / Observed / Rejected / Not found / not verifiable in this scan；无代码/未核验数字明确标注 |

五类 Watch 允许零 Accepted。交叉出现的 Source ID 只计一次；不要为了补齐 Watch 回写历史扫描。

## Reading Queue Updates

- [ ] 加入 [P0](../reading_queue/P0.md)（未来 triage 目标为活跃项 ≤ 3；不自动删改旧条目）：
- [ ] 加入 [P1](../reading_queue/P1.md)：
- [ ] 仅观察：
- [ ] 转入本目录下 `backfill/YYYY-MM.md`（原始发表月份）：

## 去重记录

- 本次新增 Source ID：
- Follow-up Source ID：
- 与历史 backfill 重复但未收录：
- 跨 Watch 同一 Source ID 的引用与唯一计数：

## 扫描完整性

- 已扫描来源：
- 未完整扫描来源：
- 已知盲区：
- 新增主线的历史覆盖缺口（不得写成已补扫）：
- 下次优先补扫：

## 下一步动作

- [ ] 实际扫描后更新 [Scan Log](scan_log.md)；只写 `research/tracking/scan_log.md`
- [ ] 需要阅读：
- [ ] 需要更新的 topic：
- [ ] 需要新增的 engineering blog / paper / report note：
- [ ] 需要做实验验证的方向：

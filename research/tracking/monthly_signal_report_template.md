# Monthly Signal Report Template

这个模板用于输出上个月的高质量 Systems / Training / Inference / RL / Embodied 共享研究信号沉淀。

Monthly signal 不是更长的 frontier scan，也不是链接汇总。它只从当月 frontier scans、release note、backfill 和真实阅读结果中筛选少量高价值材料，回答：

> 这个月真正填补了模型/学习缺口或改变了系统判断的信号是什么？

原则：

- 固定统计窗口：上月 1 日 00:00:00 到上月最后一天 23:59:59，时区 `Asia/Shanghai`。
- 文件名使用月份：`monthly_signal_YYYY-MM.md`。
- Accepted signals 通常 3 到 5 条；可以更少，可以为 0。
- P0 通常不超过 1 到 2 条，P1 通常不超过 3 到 5 条。
- Monthly 不重新发现材料，只汇总当月 frontier scans / backfill / release note / 已读材料。
- 扩展范围自 2026-09-22 起约束未来报告；本模板修改不表示新扫描、不推进 cursor，也不更改历史 Accepted 数量、阅读决策或完成状态。新增主线的历史覆盖不足应写成缺口，旧材料后续按原始月份 backfill，不在写月报时临时重新发现。
- 准入有两条通道：回答当前 embodied 学习问题（inputs/outputs、action representation、objective、generalization/evaluation），即使尚无已证明 infra 增益；或存在性能、正确性、状态、IO、部署等系统后果。独立 inference 不受“是否影响 rollout”限制。
- 通用 demo、融资、产品新闻、未经核验榜单不自动晋级；无代码不能宣称已可运行或复现，效果/性能未验证必须标注。MLLM 指多模态大语言模型，不是 vLLM 推理引擎。
- 核对每条 Accepted 的一手标题、作者、发布时间和关键数字；arXiv 必须匹配 `citation_title`、`citation_author`、`citation_date` 及摘要/方法。若已有记录不足以支持结论，写明待核验而非补造证据；核验既有材料不等于重新发现新来源。
- 每项新增 Accepted 记录 `Learning track`、`Signal type`、`Evidence`、`Target question`；同一 Source ID 跨 Watch 只计一次。Watch 筛选状态与阅读 `Decision` 分开。
- 核心模型厂商的 technical report、model card、工程博客和规模化部署报告属于一级工业证据，必须显式审视；但厂商身份不等于自动 Accepted，公开机制、代码、可复核 benchmark 和厂商自报数字要分层标注。
- 宁缺毋滥。没有高质量判断就明确写无。

---

# Monthly Signal Report, YYYY-MM

- Window: YYYY-MM-01 00:00:00 ~ YYYY-MM-DD 23:59:59
- Timezone: Asia/Shanghai
- Generated at: YYYY-MM-DD
- Report type: monthly quality digest
- Input reports / reading records: 本月实际使用的仓库记录链接
- Coverage policy: 2026-09-22 expanded scope; gaps retained, not backdated coverage

## 本月核心判断

用 1-3 段说明本月哪些模型/学习或系统方向真正值得进入长期关注；区分来源事实、厂商自报与趋势推断。

## Accepted Signals

本节只放高质量内容，可以为空。

### 标题

- Signal ID：YYYY-MM-001
- Source ID：
- First seen：
- Scan window / 来源窗口：对应 frontier scan 的真实窗口，或 backfill / release note / reading 的原始时间与记录时间
- 来源记录：已有 frontier / backfill / release note / reading 的仓库链接
- Learning track：Systems / Training / Inference / RL / Embodied（可多选）
- Signal type：model / algorithm / system
- Target question：当前模型/学习缺口或系统问题
- Evidence：公开方法、代码/测试、可复核 benchmark 或生产报告；明确自报、未验证与推断的边界
- 类型：paper / repo / engineering blog / release note / report
- 链接：
- 作者 / 发布主体：
- 发布时间：
- 影响等级：★★★★★ / ★★★★☆ / ★★★☆☆
- Decision：Read / Deep Dive / Observe
- Reason：
- 建议动作：进入 P0 / 进入 P1 / 观察 / 直接沉淀到 topic
- 关联主题：
- 一句话价值：
- 最终应流向：对应主线 topic / research insight 或 note / practice playbook、experiment 或 project
- Status：NEW / READING / SUMMARIZED / DIGESTED / VERIFIED / IMPLEMENTED / OBSOLETE

## P0 / P1 更新

使用共享 [P0](../reading_queue/P0.md) / [P1](../reading_queue/P1.md)。未来 triage 目标为活跃 P0 ≤ 3；月报可提议调整，但不能因为结构迁移自动删减、降级或标记 Done。

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

## Industrial Evidence Watch

核心厂商的技术报告和生产材料应单独说明证据等级，以及它验证或改变了哪条工程路线。

## OpenAI / Anthropic / NVIDIA / DeepSeek Watch

四家一手来源必须显式出现。只汇总本月已有扫描与阅读记录，不为填写 Watch 重扫来源；未覆盖写缺口，不自动收录。

| Vendor | 已有来源记录 / Evidence / Source ID | Watch decision | 本月判断 / 覆盖限制 |
|---|---|---|---|
| OpenAI | official blog / research / docs / reports；附当月记录与一手证据 | Accepted / Observed / Rejected / Not found / not verifiable in this scan | 有何判断或未覆盖原因 |
| Anthropic | official blog / research / docs / reports；附当月记录与一手证据 | Accepted / Observed / Rejected / Not found / not verifiable in this scan | 有何判断或未覆盖原因 |
| NVIDIA | technical blog / docs / developer posts / reports；附当月记录与一手证据 | Accepted / Observed / Rejected / Not found / not verifiable in this scan | 有何判断或未覆盖原因 |
| DeepSeek | official API changelog / 官方 Hugging Face organization / reports；明确当月是否覆盖两处 | Accepted / Observed / Rejected / Not found / not verifiable in this scan | 哪些机制公开，哪些仍不可核验 |

## Hugging Face Watch

| 已有来源记录 / Evidence / Source ID | Watch decision | 本月判断 / 覆盖限制 |
|---|---|---|
| Hugging Face Blog / TRL / Transformers / Accelerate / PEFT / Kernels / LeRobot 的当月记录 | Accepted / Observed / Rejected / Not found / not verifiable in this scan | training / inference / RL / embodied / dataset IO；区分官方团队、厂商联合和 community post，未覆盖不补造 |

## RL Framework Watch

本节不重新扫描 GitHub，只汇总本月 frontier scans 已筛选出的正式 release 与重大 PR。没有改变工程判断的框架活动不进入月报。

| Framework | 已有记录 / Release / PR / Source ID | 子系统 | 本月判断 / Evidence | 对 AReaL 的参考 | Watch decision |
|---|---|---|---|---|---|
| AReaL / verl / slime / ROLL / OpenRLHF / NeMo RL / emerging | 当月 frontier 记录及实际 tag / PR | rollout / training / scheduler / weight sync / data/trajectory path / checkpoint/recovery / inference backend | 系统边界与性能/显存/稳定性/正确性/运维维度；附 diff/test/benchmark/production 证据 | 可复用设计、需验证或无直接关系 | Accepted / Observed / Rejected / Not found / not verifiable in this scan |

## Inference Systems Watch

只归纳本月既有扫描/阅读中的独立 inference 信号，包括但不限于 vLLM、SGLang、TensorRT-LLM；不以是否影响 RL rollout 为前提，不在月报重扫 GitHub。

| Source / Source ID | 已有记录与 Evidence | Target question / 本月判断 | Watch decision / 覆盖限制 |
|---|---|---|---|
| runtime / paper / production report | 当月记录和公开机制/代码/测试/benchmark | serving/scheduler、KV/state、long context、multimodal、kernel/precision、latency/throughput/cost、部署正确性 | Accepted / Observed / Rejected / Not found / not verifiable in this scan；可以零条 |

## Embodied Models & Infra Watch

候选来源包括 openpi / Physical Intelligence、OpenVLA、GR00T / Isaac Lab、Hugging Face LeRobot、PyTorch / TorchCodec 和 dataset/IO 社区。这里只总结已有记录；名单不是推荐，也不表示当月已检查所有来源。

| Source / Source ID | 已有记录与 Evidence | Learning value / Target question | System consequence（如有） | Watch decision / 覆盖限制 |
|---|---|---|---|---|
| paper / model / repo / report / community source | 当月记录、公开方法/代码/评估；标注无代码与未核验数字 | inputs/outputs、action representation、objective、generalization/evaluation | 数据/仿真/训练/部署边界；无证据就写未披露或待验证 | Accepted / Observed / Rejected / Not found / not verifiable in this scan；可以零条 |

同一 Source ID 在五类 Watch 中交叉出现时，Accepted 只计一次。新主线没有历史记录是覆盖缺口，不得回写旧报告制造覆盖。

## 对仓库的影响

- 需要更新的 topic：
- 需要更新的 insight：
- 需要更新的 playbook：
- 需要新增的 experiment：
- 需要进入 historical backfill 的材料：
- 新主线历史覆盖缺口与后续补录计划（按原始发表月份）：
- 需要更新的根 Knowledge Graph / 共享 Master Reading List：

## 下月关注

- 方向 1：
- 方向 2：
- 方向 3：

## 返回用户时的摘要

附完整报告链接、一小段中文总体进展与工程趋势，以及每条 Accepted 的来源链接和一句话机制/学习价值/后果。区分新发表与迟发现材料，明确趋势推断与覆盖缺口；零条 Accepted 就说明无合格信号，不凑趋势。

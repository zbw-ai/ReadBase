# Monthly Signal Report · 2026-09

- 统计窗口：**2026-09-01 00:00:00 → 2026-09-30 23:59:59，Asia/Shanghai**。
- 编制日期：2026-10-08；类型：月度高质量复盘，替代原截至 9/22 的阶段版。
- 来源：7 次 9 月 frontier scans、9/22 GitHub 专项与历史复盘、已有专题阅读，以及 [10/08 scan](frontier_scan_2026-10-08.md) 中原始发表/合并于 9 月的材料。**不重新发现文章，不推进 frontier 游标。**
- 核心结论：保留 **5 组月度 Accepted 主线**；9 月下旬增加 HAPMoE、AReaL partial group、verl layout refit、ROLL NCCL suspend 与 NeMo SGLang recovery 的证据。
- 覆盖边界：自然月已经结束，但不代表来源已穷尽。六个 RL 框架 9/22 17:06:16 至月末的 merged PR 索引补入 **82 条**；外围仓库、直接 commit 和 arXiv 全分类发现仍有缺口。见[归档账本](audits/2026-09-monthly/aggregation.json)。

用户不需要逐份读完扫描。先看下面五条判断，再按自己的问题选择一份材料。Status 区分“仓库已写出笔记”和“个人已掌握”，本报告不替用户更新学习进度。

## 本月进展与趋势

7 月开始看清 rollout 的成本，8 月集中暴露异步执行与恢复的边界，9 月更明确地把环境、经验数据、数值状态和请求准入连接成训练系统。**趋势推断：有效训练吞吐取决于交付了多少可信、可消费、可恢复的经验，而不只是生成了多少 token。** 这是本仓库对连续材料的归纳，不是全行业统计。

<a id="m1"></a>

## 1. DeepSeek 报告要与 DSec 一起读：环境状态和模型状态同样重要

月度 Signal ID：2026-09-M1；Impact：★★★★★；类型：vendor technical reports；Source IDs：`arxiv:2609.19969v1`、`arxiv:2609.22978v1`。首次收录分别见 [9/20](frontier_scan_2026-09-20.md)、[9/22](frontier_scan_2026-09-22.md)；模型权重更早见于 9/16，按同一主线去重。

[DeepSeek-V4.1-Flash](https://arxiv.org/abs/2609.19969) 将 KV 压缩与 Agentic 能力放在同一份工业报告中；配套 [DSec](https://arxiv.org/abs/2609.22978) 披露 sandbox 平台的接口、镜像供给和 pause/resume。最值得跟进的是长程任务被打断后，agent loop 与环境状态如何保留，而不只是模型 checkpoint 是否存在。

这里不能混淆三件事：压缩上下文降低模型侧成本；冻结容器保留运行现场；VM snapshot 还涉及另一套恢复机制。DSec 的平台规模属于厂商披露，本仓库没有独立复现。DeepSeek 模型报告首次登记晚于其原始发布日，按 late-discovered 保留，不能算成 9 月 20 日新发表。

**Decision：Deep Dive。** 只读一份工业材料时，优先从 DSec 的生命周期机制切入，再回到模型报告的训练设计。现有[DeepSeek 阅读入口](../reading_queue/P1.md#deepseek-v41-report)已承接，避免重复建任务。

<a id="m2"></a>

## 2. MiMo / CodeMidas 解释经验怎样生产，Conduit 解释它怎样交付

月度 Signal ID：2026-09-M2；Impact：★★★★★；类型：technical report / paper / systems paper；Source IDs：`hf:XiaomiMiMo/MiMo-V2.6-Pro-RL@73875d00b30a89ef8cc353a0b60b0e9f9561952d`、`arxiv:2609.22068v1`、`arxiv:2609.24456v1`。来源：[9/22 扫描](frontier_scan_2026-09-22.md)与 [MiMo 定向研究记录](agentic_rl.md#mimo-v26-research)，原始日期、署名和阅读状态沿用专题笔记。

已有 [MiMo-V2.6 / CodeMidas 笔记](../tech_reports/mimo_v26.md)覆盖源码驱动环境、评分、Sample Mixer 和运行故障。它带来的判断是：环境通过测试不等于有学习价值，采样配比和任务难度也会改变 GPU 的有效产出。[CodeMidas](https://arxiv.org/abs/2609.22068v1) 的独立实验与 [MiMo-V2.6 报告](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL/blob/73875d00b30a89ef8cc353a0b60b0e9f9561952d/MiMo_V2_6_technical_report.pdf)不是同一训练实验，增益不能相加。

[Conduit](https://arxiv.org/abs/2609.24456) 将经验数据的放置、容量与交付时机显式化。它适合回答 learner 为何等数据；主评估基于 RLlib，LLM post-training 属于扩展评估，不能直接套用全部性能结论。

**Decision：MiMo / CodeMidas 已有仓库笔记，Conduit Read。** 迁移时先画清任务身份、policy version、驻留位置、消费确认和可重放边界，而不是直接更换 replay buffer。

<a id="m3"></a>

## 3. 恢复的核心是“什么时候可以重新接请求”

月度 Signal ID：2026-09-M3；Impact：★★★★★；类型：merged PR / official release；下旬主 Source IDs：`github:areal-project/AReaL#1721`、`github:alibaba/ROLL@v0.4.0`、`github:NVIDIA-NeMo/RL#3613`。First seen：2026-10-08，**late-discovered，归入原合并/发布月**；旧材料的首次收录时间保留在原扫描。

本月 AReaL 的不可变 generation + `LATEST`、NeMo RL 的 generation shard 恢复、verl 的 admission gate 共同说明：数据写完、权重装完、状态重建完、允许新请求进入，是不同的事件。

9 月历史补漏又找到 [TRL #7175](https://github.com/huggingface/trl/pull/7175)：一个慢同步工具原来就能卡住整个事件循环和 heartbeat。修复把同步工具放入线程池，同时保留同一 turn 内的顺序。与 8 月队列丢 group、7 月 logprob 错配连起来，不能把失败都归咎于 GPU 或 NCCL。

月末补充使判断更具体：[AReaL #1721](https://github.com/areal-project/AReaL/pull/1721) 在 9/23 合并，将 v2 offline 的有效导出子集与全组 session 清理分开；它是 v1 思路进入 v2 的实现变化，不是第一次提出 partial group。[ROLL v0.4.0](https://github.com/alibaba/ROLL/releases/tag/v0.4.0) 在 9/29 发布，用原生 NCCL suspend/resume 回收动态显存，同时保留 ProcessGroup 身份、排除不能共同 idle 的跨角色组。[NeMo RL #3613](https://github.com/NVIDIA-NeMo/RL/pull/3613) 在 9/29 合并，失效 SGLang logical engine group 等到 refit 边界替换；已开始的 stream、active weight transfer 等不在保证恢复范围。

证据分层：AReaL #1721 已读 patch，ROLL 已读 tag 内实现；NeMo 为 PR 说明和作者报告的故障实验，未独立审代码或复现。三者都不能证明任意故障下训练等价。

**Decision：Read，重点读状态转换。** 将“拒绝”“等待”“取消”“部分完成”和“准入恢复”作为不同结果验收。[GitHub 专项](github_audit_2026-09-22.md)的 parallel sampling、teacher identity、dummy-state 和 CP 补充证据，以及[历史复盘](github_retrospective_2026-07_to_2026-09.md)，都收敛到这组问题。

<a id="m4"></a>

## 4. 低精度要同时核对速度、反馈和状态表示

月度 Signal ID：2026-09-M4；Impact：★★★★★；类型：paper / training-stack implementation；主 Source IDs：`arxiv:2609.22870v1`、`github:NVIDIA-NeMo/RL:3491eed5772425acec5edb3fa5d7adccb23ff6f2`、`github:verl-project/verl#7987`。来源：[9/20](frontier_scan_2026-09-20.md)、[9/22](frontier_scan_2026-09-22.md)、[10/08 A11](frontier_scan_2026-10-08.md#a11)；最后一项为 10/08 发现的 9/24 合并记录。

[Full Pipeline FP8 RL](https://arxiv.org/abs/2609.22870) 提出量化误差可能通过 importance ratio 与 clipping 消除本应保留的负反馈；本轮仍保留“作者机制主张、尚未完整审阅消融”的证据等级。NeMo RL 的 [NVFP4 训练与 refit 实现](https://github.com/NVIDIA-NeMo/RL/commit/3491eed5772425acec5edb3fa5d7adccb23ff6f2)则表明 rollout 局部收益必须扣除训练与重载成本。

9/22 历史补查 [Megatron #6666](https://github.com/NVIDIA/Megatron-LM/pull/6666)得到一个重要反例：量化 checkpoint 的编码不一致，未必意味着表示数值或 GEMM 已变差。该 PR 从 FP32 main parameters 重建量化权重，解决特定 MXFP8 round-trip parity；不能把它写成已确认的模型精度退化事故。

9/24 的 [verl #7987](https://github.com/verl-project/verl/pull/7987) 又补上一层：即使同为 BF16，checkpoint layout 也可能与 serving kernel 的重排布局不同。PR 描述在 live storage 上 staging、逐层 fold，并保持更新期间关闭 forward。这里只核对了详细说明和合并信息，patch 下载失败；不能把布局转换方案写成已独立验证。

**Decision：Read。** 统一记录端到端有效吞吐、refit 时间、训练反馈分布，并把 bitwise parity、数值等价与模型质量分开。承接现有[低精度阅读](../reading_queue/P1.md#nvfp4-refit-reading)。

<a id="m5"></a>

## 5. 并行配置、长上下文与 kernel 需要联合验收

月度 Signal ID：2026-09-M5；Impact：★★★★☆；类型：paper / distributed-training implementation；主 Source IDs：`arxiv:2609.39350v1`、`github:huggingface/accelerate#4177`。HAPMoE 的 citation 元数据与方法由 [10/08 A3](frontier_scan_2026-10-08.md#a3)核验，First seen 为 10/08、原发表日期为 9/30；Accelerate 属历史审计承接。

前期长上下文材料在本月延伸到 CP、packing、共享 prefix 与混合注意力适配。9/22 历史补查 [Accelerate #4177](https://github.com/huggingface/accelerate/pull/4177)发现，有些 hook 会把更严格的 attention mask 替换成普通 causal 语义，因此实现选择直接改变训练目标；它不是“所有 CP 都不能支持 sliding attention”的结论。该 PR UTC 合并于 8/31，按本仓库上海时区归入 9/1。

9/30 的 [HAPMoE](https://arxiv.org/abs/2609.39350v1) 将异构硬件 profiling、路由不均、非均匀 pipeline 和重计算放入同一规划问题。它值得读的不是一个最大加速数字，而是怎样把通信、显存和最慢 stage 纳入可测成本。其六维搜索变量不能直接相乘当 world size，稳定 profile 窗口也不保证动态路由长期不变；跨厂商 launcher 与实际部署尚未复现。

**Decision：Read。** 对照 CP=1 与 CP>1 的受控输入、输出和梯度，而不是只用一个能下降的 loss 验收。7–8 月重新提到正文的 Harness Engineering 与 Contract-Grade Verifier 则提供 kernel 层的验证思路。

[QEffect](https://arxiv.org/abs/2609.23536)与 [MoSim](https://arxiv.org/abs/2609.23278)仍 **Observe**：前者适合后续检查 FP8/captured-graph 的资源与数值状态契约，后者关注网络争用下的模拟可信度；目前没有充分的新阅读结果支持挤占前三个重点，不为了复盘数量强行升级。

## 如果现在只愿意读三份

| 顺序 | 材料 | 最应该回答的问题 |
|---|---|---|
| 1 | [MiMo / CodeMidas 的现有中文笔记](../tech_reports/mimo_v26.md) | 哪些环境和样本值得送进昂贵的训练系统？ |
| 2 | [DSec](https://arxiv.org/abs/2609.22978) | GPU 作业被抢占时，长程环境与 agent 进度怎样保留？ |
| 3 | [本月状态边界与下旬实现](#m3) | partial group、NCCL offload 和引擎恢复，各在什么条件下才允许继续训练？ |

这是一条可选阅读路径，不扩充当前 P0，不把其他材料变成必须清空的待办。

## OpenAI / Anthropic / NVIDIA / DeepSeek Watch

本节汇总本月扫描已核验条目，不声称今天重新穷尽四家全部站点。

| 来源 | 月度判定 | 值得记住的内容与边界 |
|---|---|---|
| OpenAI | Accepted / Observed | [Automated research 工业报告](https://openai.com/index/research-acceleration-view-inside-openai/)把评估、人工介入、安全与资源调度放在同一研究流程中；具体数据沿用[9/7 核验](frontier_scan_2026-09-07.md)，不把 agent 工作量等同于独立成功产出；9/28 training safety cases 由 10/08 扫描保留 Observe，尚未深读执行机制 |
| Anthropic | Observed / carried forward | [环境治理报告](https://www.anthropic.com/news/improving-alignment-security-efforts)原始发布于 8 月，9/1 才进入扫描；归入 8 月复盘，本月仅承接；已查窗口的领域应用内容 Rejected，不推断站点没有任何新文章 |
| NVIDIA | Accepted | NeMo RL / Megatron 的恢复、数值表示和 logprob 实现，加上 [AIPerf](https://developer.nvidia.com/blog/benchmarking-llm-inference-at-scale-with-aiperf/)的压测方法：同时校验训练端、backend 和发压客户端 |
| DeepSeek | Accepted / Deep Dive | V4.1-Flash 与 DSec 连读；[API changelog](https://api-docs.deepseek.com/updates)、[官方 HF](https://huggingface.co/deepseek-ai)由本月扫描交叉核验，报告时间、权重发布和 API 更新不混为一个事件；10/08 发现的 HF chat-template #68 只有相对时间，不强行归入 9 月 |

## Hugging Face Watch

[HF Blog](https://huggingface.co/blog)、TRL、Transformers、Accelerate、PEFT、Kernels 和 tokenizers 均在原扫描或 GitHub 15 库历史索引内，目录覆盖不等于全部正文精读。

- **Accepted / Read：** TRL 工具循环 #7175、Accelerate CP 边界 #4177，以及既有 [tokenizers v1 文章](https://huggingface.co/blog/tokenizers-v1)；后者是 RC，CPU tokenizer 局部指标不等于 Python/GPU pipeline 同幅收益。
- **Observed：** TRL #6625 entropy backward 的条件性缺口，不夸大成已有内置 GRPO 普遍错误；9/28 [RL Environments Hub](https://huggingface.co/blog/rl-environments)由 HF 官方团队与 guest 作者发布，10/08 才收录，关注 taskset 版本化，标签不等于 runtime 互通。TRL v1.14.2、ThinkingBox 等 10 月材料不纳入本月。
- **Rejected / 不追加：** 普通模型集成、文档与重复发布不因来自 HF 自动接纳。官方团队文章与 community post 沿用原扫描来源标记。

## RL Framework Watch

| 框架 | 承接本月证据 | 子系统 / 工程维度 | 对 AReaL 的迁移判断 |
|---|---|---|---|
| AReaL | Accepted：完整 group、checkpoint generation、AWEX idle collective；下旬 #1721 / #1749 | data path / training / checkpoint；正确性、liveness | v2 offline partial group 与全组 cleanup 分离；#1750 未关闭的 receipt 风险仍 Observe |
| verl | Accepted：gate 关闭、恢复后准入、v0.9.1；下旬 #7987 | scheduler / weight sync；退出、layout 与显存 | 区分 reject 与 park；refit 完成需包含 kernel layout，不能只看收包完成 |
| slime | Accepted：本月 streaming/cancel；Observed：下旬 #2410 / #2427 背景线索 | rollout / data path；样本守恒、回放 | 10/08 扫描已携带 Straw 线索，但不能用 10/07 #2444 的完整恢复能力倒推 9 月已成熟 |
| ROLL | Accepted：9/29 v0.4.0，10/08 发现 | training / weight sync；NCCL 显存与生命周期 | 原生 suspend/resume 有 runtime 版本和 collective 同序要求；跨角色组需独立处理 |
| OpenRLHF | Observed：REST 可见 v0.11.2（9/14）；下旬查询 0 条 merged PR | 无新增月度主线；release 网页可能滞后 | 保留架构对照；0 条 merged PR 不证明没有直接 commit 或 open PR |
| NeMo RL | Accepted：token ledger、refit、generation recovery、多 teacher；下旬 #3613；Observed #4129 | data path / recovery / backend；身份与状态一致性 | refit 前恢复引擎；ledger 还要核依赖 pins。10 月 #4410 ready-first 不归入本月 |
| TRL（补充） | Accepted：异步工具调度、实验 harness | rollout / scheduler；环境顺序、heartbeat | 借鉴隔离方式，不把轻量示例当集群规模证明 |

## 已有阅读成果与未完成事项

本月 [MiMo / CodeMidas](../tech_reports/mimo_v26.md)与 [GLM Infra Agent](../engineering_blogs/zhipu/glm_infra_agent_recursive_self_improvement.md)已有专题内容。这里的“已有”只说明仓库完成了整理，不代表用户已吸收、实验已 VERIFIED。DeepSeek / DSec、Conduit、低精度和状态边界仍有深读或实验工作。

## 月末归档与覆盖账本

本月正式窗口完整，发现覆盖仍分来源记录，不能把日历完整与检索完整混为一谈。

| 来源区间 | 已有证据 | 不能据此声称 |
|---|---|---|
| 至 9/22 17:06:16 的 GitHub | [15 库历史复盘](github_retrospective_2026-07_to_2026-09.md)与[9/22 专项](github_audit_2026-09-22.md)，两份集合有重叠 | 不能把两份数量相加当独立信号，也不代表所有代码 diff 已读 |
| 9/22 17:06:16 至 9/30 月末 | 从 [10/08 索引](audits/2026-10-08-frontier/merged_pr_index.json)按 `merged_at` 转上海时区筛出 82 条：AReaL 7、verl 32、slime 8、ROLL 1、OpenRLHF 0、NeMo RL 34 | 这是六库合并 PR 集合，含 cherry-pick；不是全月总 PR 数，也不覆盖其他库或直接 commit |
| 论文 / 文章 | 9 月扫描、已选元数据复核，以及 10/08 新发现的 9 月材料 | 未全分类枚举 arXiv；厂商缺失正文和相对时间条目不能用推测补齐 |

**月份边界复核：** HAPMoE（9/30）、ROLL v0.4.0（9/29）、AReaL #1721（9/23）、verl #7987（9/24）、NeMo #3613（9/29）回归 9 月，保留 First seen=10/08。Olmo-core 3、VenusRL、TRANSIT、ThinkingBox、AICR、GPUNetIO、TRL v1.14.2、slime #2444、NeMo #4410 留在 10 月，不能把后续能力算作 9 月成果。日期取来源原始日期或 `merged_at`，不取 PR 创建时间或 patch 作者日期。

逐项选择及 82 条 ID 见[月度归档账本](audits/2026-09-monthly/aggregation.json)。源标题、作者、日期与机制核验沿用原扫描/专题及[10/08 来源账本](audits/2026-10-08-frontier/evidence.json)，没有新增未经核验的论文事实。既有 2025—2026 H1 历史 GitHub 缺口继续保留。

## 10 月关注与知识流转

1. **环境是否值得训练：** 沿 MiMo / CodeMidas、DSec 检查任务质量、失败归因、状态隔离和恢复成本；先复用已有笔记，不重复扩充 P0。
2. **样本是否可消费：** 以 partial group、token lineage、refit admission 为线索，选一个实际 backend 做受控失败实验；关注长度分布与 policy age，不只看 tokens/s。
3. **优化是否真正省成本：** HAPMoE 的配置误差、低精度 refit 峰值、kernel layout 与端到端成本一起量；10 月新材料只作为下一阶段阅读，不能补作 9 月已验证结果。

本月判断承接 [Agentic RL topic](../../04-rl-infra/topics/agentic_rl.md#september-2026-monthly)、[MoE 联合设计](../../02-training-infra/topics/moe.md#october-2026-joint-design)与[未执行的验证候选](../../practice/experiments/rl_state_boundaries.md#october-2026-cases)。P1 沿用 DeepSeek、NVFP4 与 CSBP 等已有入口，HAPMoE 先按集群需求选读；本次完成月度归纳与导航，不改变用户的个人已读状态，也不将仓库笔记标成 VERIFIED。

[返回月度入口](monthly_reviews.md) · [历史补扫与证据](github_retrospective_2026-07_to_2026-09.md)

# Monthly Signal Report · 2026-09（阶段复盘）

**截至 2026-09-22，尚非完整自然月。** 本文汇总 7 次 9 月扫描、已有专题阅读和历史 GitHub 补扫。一般来源沿用已确认的 9/22 16:52:05 游标；GitHub 补扫到 17:06:16（Asia/Shanghai）；另纳入当天已完成的 MiMo 专题笔记，不据此推进全局游标。月末还需补齐余下时段。

用户不需要逐份读完扫描。先看下面五条判断，再按自己的问题选择一份材料。Status 区分“仓库已写出笔记”和“个人已掌握”，本报告不替用户更新学习进度。

## 本月进展与趋势

7 月开始看清 rollout 的成本，8 月集中暴露异步执行与恢复的边界，9 月更明确地把环境、经验数据、数值状态和请求准入连接成训练系统。**趋势推断：有效训练吞吐取决于交付了多少可信、可消费、可恢复的经验，而不只是生成了多少 token。** 这是本仓库对连续材料的归纳，不是全行业统计。

## 1. DeepSeek 报告要与 DSec 一起读：环境状态和模型状态同样重要

[DeepSeek-V4.1-Flash](https://arxiv.org/abs/2609.19969) 将 KV 压缩与 Agentic 能力放在同一份工业报告中；配套 [DSec](https://arxiv.org/abs/2609.22978) 披露 sandbox 平台的接口、镜像供给和 pause/resume。最值得跟进的是长程任务被打断后，agent loop 与环境状态如何保留，而不只是模型 checkpoint 是否存在。

这里不能混淆三件事：压缩上下文降低模型侧成本；冻结容器保留运行现场；VM snapshot 还涉及另一套恢复机制。DSec 的平台规模属于厂商披露，本仓库没有独立复现。DeepSeek 模型报告首次登记晚于其原始发布日，按 late-discovered 保留，不能算成 9 月 20 日新发表。

**Decision：Deep Dive。** 只读一份工业材料时，优先从 DSec 的生命周期机制切入，再回到模型报告的训练设计。现有[DeepSeek 阅读入口](../reading_queue/P1.md#deepseek-v41-report)已承接，避免重复建任务。

## 2. MiMo / CodeMidas 解释经验怎样生产，Conduit 解释它怎样交付

已有 [MiMo-V2.6 / CodeMidas 笔记](../tech_reports/mimo_v26.md)覆盖源码驱动环境、评分、Sample Mixer 和运行故障。它带来的判断是：环境通过测试不等于有学习价值，采样配比和任务难度也会改变 GPU 的有效产出。[CodeMidas](https://arxiv.org/abs/2609.22068v1) 的独立实验与 [MiMo-V2.6 报告](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL/blob/73875d00b30a89ef8cc353a0b60b0e9f9561952d/MiMo_V2_6_technical_report.pdf)不是同一训练实验，增益不能相加。

[Conduit](https://arxiv.org/abs/2609.24456) 将经验数据的放置、容量与交付时机显式化。它适合回答 learner 为何等数据；主评估基于 RLlib，LLM post-training 属于扩展评估，不能直接套用全部性能结论。

**Decision：MiMo / CodeMidas 已有仓库笔记，Conduit Read。** 迁移时先画清任务身份、policy version、驻留位置、消费确认和可重放边界，而不是直接更换 replay buffer。

## 3. 恢复的核心是“什么时候可以重新接请求”

本月 AReaL 的不可变 generation + `LATEST`、NeMo RL 的 generation shard 恢复、verl 的 admission gate 共同说明：数据写完、权重装完、状态重建完、允许新请求进入，是不同的事件。

9 月历史补漏又找到 [TRL #7175](https://github.com/huggingface/trl/pull/7175)：一个慢同步工具原来就能卡住整个事件循环和 heartbeat。修复把同步工具放入线程池，同时保留同一 turn 内的顺序。与 8 月队列丢 group、7 月 logprob 错配连起来，不能把失败都归咎于 GPU 或 NCCL。

**Decision：Read，重点读状态转换。** 将“拒绝”“等待”“取消”“部分完成”和“准入恢复”作为不同结果验收。[GitHub 专项](github_audit_2026-09-22.md)的 parallel sampling、teacher identity、dummy-state 和 CP 补充证据，以及[历史复盘](github_retrospective_2026-07_to_2026-09.md)，都收敛到这组问题。

## 4. 低精度要同时核对速度、反馈和状态表示

[Full Pipeline FP8 RL](https://arxiv.org/abs/2609.22870) 提出量化误差可能通过 importance ratio 与 clipping 消除本应保留的负反馈；本轮仍保留“作者机制主张、尚未完整审阅消融”的证据等级。NeMo RL 的 [NVFP4 训练与 refit 实现](https://github.com/NVIDIA-NeMo/RL/commit/3491eed5772425acec5edb3fa5d7adccb23ff6f2)则表明 rollout 局部收益必须扣除训练与重载成本。

这次补查 [Megatron #6666](https://github.com/NVIDIA/Megatron-LM/pull/6666)得到一个重要反例：量化 checkpoint 的编码不一致，未必意味着表示数值或 GEMM 已变差。该 PR 从 FP32 main parameters 重建量化权重，解决特定 MXFP8 round-trip parity；不能把它写成已确认的模型精度退化事故。

**Decision：Read。** 统一记录端到端有效吞吐、refit 时间、训练反馈分布，并把 bitwise parity、数值等价与模型质量分开。承接现有[低精度阅读](../reading_queue/P1.md#nvfp4-refit-reading)。

## 5. 更长上下文、更快 kernel 需要更强的验收

前期长上下文材料在本月延伸到 CP、packing、共享 prefix 与混合注意力适配。补查 [Accelerate #4177](https://github.com/huggingface/accelerate/pull/4177)发现，有些 hook 会把更严格的 attention mask 替换成普通 causal 语义，因此实现选择直接改变训练目标；它不是“所有 CP 都不能支持 sliding attention”的结论。该 PR UTC 合并于 8/31，按本仓库上海时区归入 9/1。

**Decision：Read。** 对照 CP=1 与 CP>1 的受控输入、输出和梯度，而不是只用一个能下降的 loss 验收。7–8 月重新提到正文的 Harness Engineering 与 Contract-Grade Verifier 则提供 kernel 层的验证思路。

[QEffect](https://arxiv.org/abs/2609.23536)与 [MoSim](https://arxiv.org/abs/2609.23278)仍 **Observe**：前者适合后续检查 FP8/captured-graph 的资源与数值状态契约，后者关注网络争用下的模拟可信度；目前没有充分的新阅读结果支持挤占前三个重点，不为了复盘数量强行升级。

## 如果现在只愿意读三份

| 顺序 | 材料 | 最应该回答的问题 |
|---|---|---|
| 1 | [MiMo / CodeMidas 的现有中文笔记](../tech_reports/mimo_v26.md) | 哪些环境和样本值得送进昂贵的训练系统？ |
| 2 | [DSec](https://arxiv.org/abs/2609.22978) | GPU 作业被抢占时，长程环境与 agent 进度怎样保留？ |
| 3 | [7–9 月 GitHub 复盘](github_retrospective_2026-07_to_2026-09.md) | 哪些“不 crash”的状态错误会悄悄改变训练？ |

这是一条可选阅读路径，不扩充当前 P0，不把其他材料变成必须清空的待办。

## OpenAI / Anthropic / NVIDIA / DeepSeek Watch

本节汇总本月扫描已核验条目，不声称今天重新穷尽四家全部站点。

| 来源 | 月度判定 | 值得记住的内容与边界 |
|---|---|---|
| OpenAI | Accepted | [Automated research 工业报告](https://openai.com/index/research-acceleration-view-inside-openai/)把评估、人工介入、安全与资源调度放在同一研究流程中；具体数据沿用[9/7 核验](frontier_scan_2026-09-07.md)，不把 agent 工作量等同于独立成功产出 |
| Anthropic | Observed / carried forward | [环境治理报告](https://www.anthropic.com/news/improving-alignment-security-efforts)原始发布于 8 月，9/1 才进入扫描；归入 8 月复盘，本月仅承接；已查窗口的领域应用内容 Rejected，不推断站点没有任何新文章 |
| NVIDIA | Accepted | NeMo RL / Megatron 的恢复、数值表示和 logprob 实现，加上 [AIPerf](https://developer.nvidia.com/blog/benchmarking-llm-inference-at-scale-with-aiperf/)的压测方法：同时校验训练端、backend 和发压客户端 |
| DeepSeek | Accepted / Deep Dive | V4.1-Flash 与 DSec 连读；[API changelog](https://api-docs.deepseek.com/updates)、[官方 HF](https://huggingface.co/deepseek-ai)由本月扫描交叉核验，报告时间、权重发布和 API 更新不混为一个事件 |

## Hugging Face Watch

[HF Blog](https://huggingface.co/blog)、TRL、Transformers、Accelerate、PEFT、Kernels 和 tokenizers 均在原扫描或 GitHub 15 库历史索引内，目录覆盖不等于全部正文精读。

- **Accepted / Read：** TRL 工具循环 #7175、Accelerate CP 边界 #4177，以及既有 [tokenizers v1 文章](https://huggingface.co/blog/tokenizers-v1)；后者是 RC，CPU tokenizer 局部指标不等于 Python/GPU pipeline 同幅收益。
- **Observed：** TRL #6625 entropy backward 的条件性缺口，不夸大成已有内置 GRPO 普遍错误。
- **Rejected / 不追加：** 普通模型集成、文档与重复发布不因来自 HF 自动接纳。官方团队文章与 community post 沿用原扫描来源标记。

## RL Framework Watch

| 框架 | 承接本月证据 | 子系统 / 工程维度 | 对 AReaL 的迁移判断 |
|---|---|---|---|
| AReaL | Accepted：完整 group、checkpoint generation、AWEX idle collective | training / checkpoint / scheduler；正确性、liveness | 自身基线，重点测不完整 group 与重启后策略版本 |
| verl | Accepted：gate 关闭、恢复后准入、v0.9.1 | scheduler / weight sync；退出轮转与临时暂停语义 | 区分 reject 与 park；稳定发布和 main 补丁分开验收 |
| slime | Accepted，承接本月 streaming/cancel；8 月丢 group 修复历史补录 | rollout / training；样本守恒、统计域 | 长短轨迹和 CP 空分片都进入测试，避免只移植快路径 |
| ROLL | Observed | 本窗口未选出改变月度结论的新增实现 | 保留对照；已完成索引，不用活跃度补位 |
| OpenRLHF | Observed | 同上 | 保留 Ray/vLLM/DeepSpeed 架构对照 |
| NeMo RL | Accepted：token ledger、refit、generation recovery、多 teacher | data path / recovery / backend；身份与状态一致性 | 可借鉴契约，格式与 backend 兼容仍需单独验证 |
| TRL（补充） | Accepted：异步工具调度、实验 harness | rollout / scheduler；环境顺序、heartbeat | 借鉴隔离方式，不把轻量示例当集群规模证明 |

## 已有阅读成果与未完成事项

本月 [MiMo / CodeMidas](../tech_reports/mimo_v26.md)与 [GLM Infra Agent](../engineering_blogs/zhipu/glm_infra_agent_recursive_self_improvement.md)已有专题内容。这里的“已有”只说明仓库完成了整理，不代表用户已吸收、实验已 VERIFIED。DeepSeek / DSec、Conduit、低精度和状态边界仍有深读或实验工作。

GitHub 历史覆盖已从片段 feed 扩展为 15 库逐页索引；论文复核为选定 35 篇的元数据与部分机制，未重扫 arXiv 三个月全分类，也未覆盖任意新兴 repo。1–6 月历史 GitHub 仍保留原覆盖限制。9 月剩余日期留待后续自然扫描，现有游标不因这次复盘前移。

[返回月度入口](monthly_reviews.md) · [历史补扫与证据](github_retrospective_2026-07_to_2026-09.md)

# Frontier Scan 2026-09-22

> 后续更新：[GitHub 全面补扫](github_audit_2026-09-22.md)已关闭下文记录的 vLLM/SGLang 提交索引缺口，新增 9 组 GitHub 信号并核验 release/分支归属。下文保留原扫描时刻的 7 Accepted / 5 Observed 口径；GitHub 专项游标为 17:06:16，其他来源仍为原游标。

- Window：2026-09-20 10:27:43 → 2026-09-22 16:52:05（Asia/Shanghai）。Next cursor：2026-09-22 16:52:05。
- Accepted：7；Observed：5 组（O1–O5；Watch 不重复计数）。A1/A3 于 9/19 提交，为 announcement-boundary late-discovered；其余五项发表于窗口内。
- 每个 Accepted 的 scan window 均为上述窗口；First seen 均登记为本轮核验结束 2026-09-22 16:52:05，不代表网络首发。Status：NEW，局部机制阅读不等于完整精读或本地复现。
- 覆盖限制：GitHub REST 本轮限流，回退 main Atom 和固定 patch；vLLM/SGLang feed 未跨过起点，下轮必须从 **2026-09-20 10:27:43** 补扫这两个来源并去重。其他来源使用全局游标；扫描是重点筛选，不承诺穷尽所有论文。

## 整体进展与趋势

本轮新增重点从模型内部优化延伸到环境、经验数据和数值反馈三个环节。DSec 补充 DeepSeek-V4.1-Flash 报告背后的 sandbox 平台；Conduit 将经验数据放置与交付独立成可优化的数据面；FP8 RL 论文提醒我们，量化误差也可能让应当受到惩罚的 token 失去梯度。结合 AReaL 与 NeMo 实现，工程趋势推断是：有效训练吞吐取决于状态能否正确保留、交付并消费，单独降低 GPU 计算时间不足以保证收益。此结论是材料归纳，不是行业统计。

## Accepted

### A1 · DSec：把 agent 环境状态与 GPU 作业生命周期拆开

- Source ID：`arxiv:2609.22978v1`；Type：vendor technical report；[原文](https://arxiv.org/abs/2609.22978v1) / [全文](https://arxiv.org/html/2609.22978v1)。
- 核验标题：DeepSeek Elastic Compute (DSec): A Sandbox Infrastructure for Effective Agentic Training at Scale。Authors：Jialiang Huang、Hongxuan Tang、Jingchang Chen 等，完整成员见原文；citation_date：2026/09/19。原始 citation_title/author/date/abstract 已核对，早于主窗口，按公告边界补漏收录。
- Impact：★★★★★；Decision：Deep Dive；Reason：披露真实 agent sandbox 平台如何处理突发创建、低利用率、镜像分发与训练抢占，是 V4.1-Flash §5.2 的高价值配套阅读。
- 一句话看点：独立 agent loop、环境分层和 pause/resume 让长程 rollout 在 GPU 作业被抢占后仍能保留进展。
- 已读机制：统一 FnCall/container/microVM/VM 接口；3FS 按需载入镜像；§6.2 保留 agent 执行状态与 sandbox 状态两者，§6.3 容器冻结后 reclaim/swap，microVM 则 snapshot 后退出进程。不能把容器暂停等同于完整 VM snapshot。
- 证据边界：平台规模和效果属于厂商生产披露，未独立复现；本文只读取架构、生命周期和关键机制，尚未完成 31 页全文精读。
- Related topics：Agentic RL / checkpoint / storage / CPU-memory scheduling。Next：纳入 [DeepSeek P1](../reading_queue/P1.md#deepseek-v41-report)，优先 §6.2–6.3，再读 §5.2–5.3；对照 [MiMo 环境研究](../tech_reports/mimo_v26.md)区分环境供给与运行态恢复。

### A2 · Conduit：经验数据路径成为独立优化对象

- Source ID：`arxiv:2609.24456v1`；Type：systems paper；[原文](https://arxiv.org/abs/2609.24456v1) / [全文](https://arxiv.org/html/2609.24456v1)。
- 核验标题：Conduit: An Experience Data Plane for Distributed Reinforcement Learning。Authors：Sitong Zhang、Tuo Shi、Mario Di Francesco、Zeke Wang、Bo Zhao；citation_date：2026/09/21；citation metadata 与方法章节已核对。
- Impact：★★★★☆；Decision：Read；Reason：CPU/GPU 分层容量、网络带宽和交付时机共同决定 learner 等待时间，不应把 replay buffer 仅当队列实现。
- 一句话看点：把经验的写入、放置和交付拆成显式控制点，减少 learner 暴露在关键路径上的数据等待。
- 证据：§4–6 定义 EDP、容量受限的带宽感知放置及 latency-aware scheduling；主评估是 RLlib，Appendix A 扩展到 verl LLM post-training，不能把传统 RL 规模结果直接当作 LLM RL 的结果。
- 边界：未复现性能，也未完成代码和 Appendix C 正确性不变量审计；迁移时必须保留 on-policy freshness、off-policy replay 和样本身份语义。
- Subsystem / Dimension：data/trajectory path / scheduler；延迟、容量、数据一致性。Transfer to AReaL：可借鉴显式 residency/delivery 与 consumer readiness，但需先核查当前 ownership 与 sample version。
- Related topics：Agentic RL / data movement / memory tiering。Next：进入 [P1 数据与状态阅读](../reading_queue/P1.md#rl-state-boundaries-reading)，画 actor→buffer→learner 成本图并读 Appendix A/C。

### A3 · Full Pipeline FP8 RL：被 clipping 丢掉的负反馈

- Source ID：`arxiv:2609.22870v1`；Type：paper；[原文](https://arxiv.org/abs/2609.22870v1)。
- 核验标题：Towards Full Pipeline FP8 Reinforcement Learning for LLMs。Authors：Fanchao Chen、Ziheng Jiang、Ziyun Wei、Zheng Zhong、Du Li、Chi Zhang、Haibin Lin、Shivaram Venkataraman；citation_date：2026/09/19。原始 citation metadata/abstract 已核对；公告边界 late-discovered。
- Impact：★★★★★；Decision：Read；Reason：给出低精度训练失稳的具体机制候选，能改变 precision 验收指标，不是仅算法分数变化。
- 一句话看点：FP8 噪声可能扭曲 importance ratio，使负 advantage token 被错误裁剪、失去惩罚，从而出现 entropy surge 和乱码。
- 来源提出 Calibrated Clipping，以 BF16 分布的下界裁剪分位校准 FP8，并相应调整上界。证据边界：本轮只完成摘要与元数据核验，HTML 全文不可用；该因果链是作者主张，尚未审阅消融，不外推到 NVFP4 或所有 estimator。
- Related topics：FP8 / GRPO / DAPO / train-inference mismatch。Next：作为 [NVFP4/低精度 P1](../reading_queue/P1.md#nvfp4-refit-reading)配套，获取 PDF 后审阅校准开销、BF16 参考采样频率和负 advantage 的 clip fraction，再决定实验实现。

### A4 · tokenizers v1 RC：CPU 数据供给也可能卡住 GPU

- Source ID：`hf-blog:tokenizers-v1`；Type：official-team engineering blog；[原文](https://huggingface.co/blog/tokenizers-v1)。
- 标题：tokenizers v1: encode, decode and scaling, measured。作者页面署名：Arthur Zucker、Simon Brandeis、Luc Georges、Lysandre；发布日期 2026-09-21。
- Impact：★★★★☆；Decision：Read；Reason：正文提供具体拆词、缓存、内存分配和多线程机制，并明确 benchmark 条件。
- 一句话看点：SIMD splitter、线程局部词缓存与可复用 scratch buffer 减少 CPU tokenization 开销，让更快的模型不被输入供给拖慢。
- 证据边界：是 release candidate，不是稳定版 v1 已发布；未知 split grammar 仍走 regex，收益取决于重复词分布；正文的 Rust 测量不包含 Python 调用开销，不能直接承诺 Python 端同幅收益。
- Related topics：input pipeline / CPU bottleneck / benchmarking。Next：按真实多语言与代码语料测 distinct-document workload、输出 token IDs 一致性和端到端 GPU idle；保持雷达 Read，不另占 P0。

### A5 · AReaL AWEX：idle 不构成跨 rank 同步点

- Source ID：`github:areal-project/AReaL:574bc6a708176c049cab5102b39e706e992f79c4`；Type：main commit；[固定补丁](https://github.com/areal-project/AReaL/commit/574bc6a708176c049cab5102b39e706e992f79c4)。
- 标题：fix(engine): avoid AWEX idle-loop collective deadlock (#1737)。作者：yulangz；patch date：2026-09-21 10:30:24 +0800；main Atom 核对归属。
- Impact：★★★★★；Decision：Read；Reason：TP event loop 漂移时，readiness all-reduce 可能与下一个请求 broadcast 交错，导致训练停止推进；不是普通日志修复。
- 一句话看点：权重更新只在显式 SGLang pause 同步 scheduler ranks 后处理，避免局部 idle 判断破坏 collective 顺序。
- Subsystem / Dimension：weight sync / scheduler；分布式 liveness。证据：固定 diff 删除 idle-loop update path，commit message 解释竞争原因；本仓库未运行多 rank 重现。
- Transfer to AReaL：直接适用于该 AWEX 路径；其他 backend 只迁移“显式同步点”原则，不能机械复制调用。
- Related topics：NCCL / TP / pause-resume。Next：加入 [状态边界实验](../experiments/rl_state_boundaries.md)，注入 rank loop skew，同时检查通信序列与恢复后的策略版本。

### A6 · NeMo RL：多 teacher 全词表蒸馏需要逐行身份

- Source ID：`github:NVIDIA-NeMo/RL:93b1c1d03474f6e6e122ec84394a5ce07c672b3c`；Type：main commit #4045；[固定补丁](https://github.com/NVIDIA-NeMo/RL/commit/93b1c1d03474f6e6e122ec84394a5ce07c672b3c)。
- 上游 subject 为截断标题「feat(mopd): support multiple teacher checkpoints in full-vocabulary o… (#4045)」；作者：Rayen / ruit，另含 co-author trailers；patch date：2026-09-21 10:33:37 -0700。
- Impact：★★★★☆；Decision：Read；Reason：teacher identity 贯穿 payload、replay、codec 和 loss 重建，影响正确性和 LM-head 显存预算。
- 一句话看点：同一 microbatch 混合多个 teacher 时，每行必须携带 teacher_index，才能用对应 LM-head 重建目标分布。
- 已核验 docs 与 diff：hidden_states 路径按 teacher 装载 LM-head shard；teacher 数增加会线性增加这部分常驻开销，必须检查共同的形状/词表约束与 lifecycle 策略。
- Subsystem / Dimension：data/trajectory path / training；身份正确性、显存。Transfer to AReaL：迁移 row-level provenance 和重排后身份验证，不假设现有 AReaL 已兼容此 wire format。
- Related topics：OPD/MOPD / TP / memory lifecycle。Next：并入 [P1 状态阅读](../reading_queue/P1.md#rl-state-boundaries-reading)，先检查混合 teacher 的重排、batching 与 checkpoint 后身份保持；未执行上游 tests。

### A7 · vLLM level-2 sleep：冻结权重仍需恢复契约

- Source ID：`github:vllm-project/vllm:639461eda424debe7bbdf2b1fc577bf83e29c643`；Type：main commit；[固定补丁](https://github.com/vllm-project/vllm/commit/639461eda424debe7bbdf2b1fc577bf83e29c643)。
- 标题：[RL][Sleep] Retain frozen weights across level-2 sleep (#57891)。作者：aoshen02；patch date：2026-09-22 12:36:55 +0800。
- Impact：★★★★☆；Decision：Read；Reason：只同步可训练参数时，不能默认 level-2 sleep 后冻结参数仍正确存在。
- 一句话看点：按运行时参数名备份冻结 GPU 权重到 CPU，并在 wake-up 原位恢复，让部分参数训练有明确的保留边界。
- Subsystem / Dimension：inference backend / weight sync；恢复正确性、host memory。证据：docs/config/gpu_worker diff；匹配 named_parameters 的 glob，不读 requires_grad；功能不会替 trainer 过滤更新，loader/post-processing 也必须保留值和 storage；仅 target model。
- Transfer to AReaL：适合借鉴冻结集合、更新集合与 loader ownership 的联合验收；需预算 pageable CPU backup，不能只看释放的 GPU 显存。
- Related topics：sleep/wake / frozen weights / refit。Next：在[实验计划](../experiments/rl_state_boundaries.md)比较冻结权重校验值、storage、host 峰值和 wake 延迟；不声称已正式 release。

## Observed

| ID | 来源 / Type | Impact / Decision | Reason / 一句话价值 | Related topics / Next |
|---|---|---|---|---|
| O1 | [QEffect](https://arxiv.org/abs/2609.23536)，paper，Genlang Chen / Junyi Zhu，2026-09-20 | 高 / Observe | 摘要提出 scaling 更新、backward owner、weight cache version 与 graph stream 同步四类契约；与现有低精度状态主线契合 | FP8 / PP / CUDA Graph；已核 metadata/abstract，待全文与代码证据再提升 |
| O2 | [MoSim](https://arxiv.org/abs/2609.23278)，paper，Yeonho Yoo 等，2026-09-20 | 中高 / Observe | 调度改变 NIC 竞争，固定网络惩罚难以预测 JCT | GPU clusters / networking；核验摘要，后续读网络模型适用范围，不把模拟结果当集群实测 |
| O3 | [verl main](https://github.com/verl-project/verl/commits/main/)、[Megatron main](https://github.com/NVIDIA/Megatron-LM/commits/main/)、[SGLang main](https://github.com/sgl-project/sglang/commits/main/)，commit indexes | 中 / Observe | 发现 VLM THD、PreGDR fusion、DSpark stream 与请求取消候选；本轮未充分审 patch，不按标题接纳 | training / inference backend；补固定版本证据及 SGLang 历史缺口 |
| O4 | [NVIDIA Blog](https://developer.nvidia.com/blog/)，官方目录 | 中 / Observe | 9/21 TensorRT Multi-Device 与 Dynamo-Triton 集成条目值得检查，正文抓取失败 | serving / multi-GPU；下轮回查 9/21 正文，暂不使用性能主张 |
| O5 | [Async GRPO HF Jobs](https://huggingface.co/blog/asyncgrpo-lora-hfjobs) / [TRL releases](https://github.com/huggingface/trl/releases)，旧条目 follow-up | 高 / Observe | 文章仍声称 v1.14，而 release feed 最新仍为 v1.13.0；版本差异未解决 | adapter sync / reproducibility；固定 PR/commit 验证，不能据博客直接声称该配方已在稳定 release 可用 |

## OpenAI / Anthropic / NVIDIA / DeepSeek Watch

| Vendor | Triage | 本轮来源与判断 |
|---|---|---|
| OpenAI | Rejected（本轮所见非 infra 条目） | [News](https://openai.com/news/)：数学顾问组、Academy 学习路径等不符合训练系统筛选；本轮未核验到新的可接纳 infra 报告 |
| Anthropic | Not found / not verifiable in this scan | [Research](https://www.anthropic.com/research)：所见最新仍为 9/17 生物分子相关内容；无新增合格系统材料，不代表整个站点无更新 |
| NVIDIA | Accepted + Observed | NeMo A6；官方 Blog 的新多 GPU serving 条目 O4，Megatron O3；旧 AIPerf 不重复计数 |
| DeepSeek | Accepted + Observed | DSec A1 来自 arXiv；同时查 [API changelog](https://api-docs.deepseek.com/updates) 与 [官方 HF](https://huggingface.co/deepseek-ai)，所见 changelog 最新 9/10，权重/报告入口未见比已有 V4.1 更新的可接纳发布；不以 API 无更新否定新报告 |

## Hugging Face Watch

- [HF Blog](https://huggingface.co/blog) / [RSS](https://huggingface.co/blog/feed.xml)：Accepted A4 为官方团队文章；oMLX 人员加入消息与模型剪枝文章未提供本轮需要的训练系统证据，Rejected，不因刊在 HF 自动接受社区/厂商文章。
- 官方 release feeds 已查：[TRL](https://github.com/huggingface/trl/releases)、[Transformers](https://github.com/huggingface/transformers/releases)、[Accelerate](https://github.com/huggingface/accelerate/releases)、[PEFT](https://github.com/huggingface/peft/releases)、[Kernels](https://github.com/huggingface/kernels/releases)。所见最新分别 v1.13.0 / v5.17.0 / v1.15.0 / v0.21.0 / v0.17.1，均早于主窗口，没有据此新接纳 release。release feed 不等于 main/docs 全量审计；TRL 差异 O5 待续。

## RL Framework Watch

| Framework | Triage / evidence | Subsystem / dimension / 对 AReaL 的意义 |
|---|---|---|
| [AReaL](https://github.com/areal-project/AReaL/commits/main/) | Accepted A5；VLM CP 等其他候选 Observed，未计独立信号 | weight sync / scheduler；通信顺序直接适用；其他候选需 patch 审阅 |
| [verl](https://github.com/verl-project/verl/commits/main/) | Observed O3；看到 THD/memory budget/backend upgrade | training / rollout；先核 shape、容量与 backend 行为再迁移 |
| [slime](https://github.com/THUDM/slime/commits/main/) | Not found：feed 未见主窗口内新提交 | 保留旧 backend 对齐主线，本轮无新设计结论 |
| [ROLL](https://github.com/alibaba/ROLL/commits/main/) | Not found：feed 未见主窗口内新提交 | 不用旧提交补新信号数量 |
| [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF/commits/main/) | Not found：feed 未见主窗口内新提交 | 同上；限 main feed 口径 |
| [NeMo RL](https://github.com/NVIDIA-NeMo/RL/commits/main/) | Accepted A6；其他修复未达到新增阅读阈值 | data/trajectory path / training；迁移 teacher identity 与 head lifecycle 验收 |
| Conduit（动态候选） | Accepted A2 论文；未单独宣称 framework readiness | data/trajectory path；需代码可运行性与 freshness 证据后再纳入持续 framework watch |

补充 runtime：vLLM Accepted A7；SGLang Observed O3；Megatron Observed O3。九个仓库的 REST 均受限，main Atom 每个返回 20 条：AReaL/verl/slime/ROLL/OpenRLHF/NeMo/Megatron 最旧条目跨过起点，提供窗口提交索引，不等于逐个 patch 已读。vLLM 最旧仅至 9/22 01:36:45 UTC、SGLang 至 9/22 04:11:27 UTC，不能宣布窗口完整。

## 阅读落点与下次动作

1. 先读 DSec §6.2–6.3，与 V4.1-Flash §5.2 和 MiMo 环境/恢复报告交叉理解；保持现有 P0 与用户阅读状态。
2. Conduit 加入数据/状态 P1；FP8 论文加入低精度 P1。tokenizers 暂留雷达，避免队列继续无界增长。
3. 本轮局部判断已接入 [Agentic RL](../topics/agentic_rl.md#rl-state-boundaries) 与 [故障注入计划](../experiments/rl_state_boundaries.md)，没有本地 GPU 或上游 tests 结果。
4. 下轮 vLLM/SGLang 从 9/20 10:27:43 回退去重；回查 NVIDIA 9/21 正文与 TRL 版本差异。arXiv 本轮检查 cs.DC/cs.LG/cs.CL/cs.AI/cs.AR recent 和选中论文原始页面，未做全分类穷尽审阅。

导航：[Scan Log](scan_log.md) · [Tracking](README.md) · [Knowledge Graph](../KNOWLEDGE_GRAPH.md) · [Master Reading List](../MASTER_READING_LIST.md)。

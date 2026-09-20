# Frontier Scan 2026-09-20

- Window：2026-09-18 10:42:01 → 2026-09-20 10:27:43（Asia/Shanghai）。Next cursor：2026-09-20 10:27:43。
- Runtime 补扫窗口：2026-09-16 14:54:26 → 2026-09-20 10:27:43；按固定 Source ID 与[9/18 scan](frontier_scan_2026-09-18.md)去重。
- Accepted：8；Observed：5 组（O1–O5，Watch 不重复计数）。其中 5 条 commit 位于主窗口内；A1/A8 为 late-discovered，A2 只有 9/18 日期，边界归属未定。
- First seen：以下条目均于本轮首次收录，登记时间 2026-09-20 10:27:43，不代表互联网首次出现。Status：NEW；源码、文档核验不等于执行测试或完成整篇精读。

## 整体进展与趋势

本轮 arXiv recent 仍是 9/18 公告批次，新增重点来自官方工程文章和框架实现；通过 DeepSeek 官方 HF 入口补到上轮漏掉的技术报告，运行时历史索引缺口也已补齐。工程趋势判断：长上下文与 Agentic RL 的成本优化越来越需要同时管理缓存容量、权重重载、请求准入和状态归属。低精度 rollout 的局部加速会被 refit/训练成本抵消，缓存压缩也必须说明是否引入近似；因此应以有效端到端吞吐和数值一致性评价，而不是只看 kernel 或生成速度。这是基于本轮材料的归纳，不代表全行业统计结论。

## Accepted

### A1 · DeepSeek-V4.1-Flash 技术报告（late-discovered）

- Source ID：`arxiv:2609.19969v1`；Type：vendor technical report；[原文](https://arxiv.org/abs/2609.19969) / [全文](https://arxiv.org/html/2609.19969v1)。
- 核验标题：DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression。Authors：DeepSeek-AI 团队，成员列表见原文 Appendix A；已读取原始 citation_author 列表，HTML 元数据另含单独冒号这一分隔项，不当作作者。citation_date：2026/09/17；v1：2026-09-17 09:43:10 UTC。
- 与旧信号区别：9/16 接受的是 9/10 模型发布；本次接受的是新技术报告证据。报告早于本轮起点，明确作为边界补漏，不把它写成周末新论文。
- Impact：★★★★★；Decision：Deep Dive；Reason：CED、CSA2 跨层 KV/index 复用、FP4 KV 与 SWA bounded replay 共同改变 prefill、持久缓存和训练并行的约束。
- 一句话看点：通过架构、缓存精度与 replay 联合降低长程 agent 的 KV 成本，并披露跨 PP 共享状态和异步 post-training 的实现设计。
- 证据与边界：原文称 global KV 为 890 bytes/token，约为 V4-Flash 的 1/4；这不是完整进程显存。§3.2.2 明确 bounded replay 得到近似 SWA 状态，结果依赖 cache-hit 位置，不能声称数学等价。数字为厂商报告，本仓库未复现。
- 已读机制：§3.1.2 用 shadow indexer、单一逻辑 owner、跨 PP payload 和 microbatch 生命周期管理支持 CSA2；§5.2 描述 colocated/time-shared 异步 rollout、sample-level dispatch、routing replay 与 off-policy 控制，不将“异步”误解为必然分离硬件。
- Related topics：KV cache / PP / CP / Agentic RL / checkpoint。Next：进入 [P1 优先深读](../reading_queue/P1.md#deepseek-v41-report)，先核查状态 owner 和 replay 精度，再展开技术报告笔记；不把本次局部阅读标为 SUMMARIZED。

### A2 · NVIDIA AIPerf：先保证压测客户端不成为瓶颈

- Source ID：`nvidia-blog:benchmarking-llm-inference-at-scale-with-aiperf`；Type：official engineering blog；[原文](https://developer.nvidia.com/blog/benchmarking-llm-inference-at-scale-with-aiperf/)。
- 标题：Benchmarking LLM Inference at Scale with AIPerf。作者：Francesco Di Natale, Elias Bermudez, Anthony Casagrande, Matthew Kotila, Harshini Komali, Ganesh Kudleppanavar。页面日期：2026-09-18；无可核验精确发布时间，保留窗口边界不确定性。
- Impact：★★★★☆；Decision：Read；Reason：正文说明多进程 load worker、独立 record processor 与 ZMQ 协调，并提供 arrival pattern、长度分布和 trace replay 来控制负载形态。
- 一句话看点：可信的 rollout/serving 容量评估要同时控制发压能力、请求到达模式和尾延迟，单一 tokens/s 不够。
- 证据边界：已读正文而非页面 AI-generated summary；多进程设计旨在缓解 client bottleneck，不意味着任意部署下自动消除它。静态固定 OSL 与生产 trace 应分别测，TTFT/ITL 需 streaming；本仓库未运行配方。
- Related topics：rollout latency / benchmarking / goodput。Next：先作为现有 rollout 实验的测量参考，记录 client CPU、实际 dispatch rate、p99、失败率和缓存命中条件，不另开当前 P0。

### A3 · NeMo RL：per-token NVFP4 训推链路

- Source ID：`github:NVIDIA-NeMo/RL:3491eed5772425acec5edb3fa5d7adccb23ff6f2`；Type：merged commit；[固定补丁](https://github.com/NVIDIA-NeMo/RL/commit/3491eed5772425acec5edb3fa5d7adccb23ff6f2)。
- 核验标题：feat: add end-to-end TE NVFP4 training with per-token vLLM rollout (#3566)。作者：Zhang Shuai；committer date：2026-09-19T20:58:29Z（UTC），不混同 release 日期。
- Impact：★★★★★；Decision：Read；Reason：selected routed-expert MLP 采用 per-token scaling；BF16 权重经 IPC refit，rollout 端量化并完成 native reload，训练不直接传 serving 表示。
- 一句话看点：低精度收益必须扣除训练和 refit 成本：上游报告 rollout 1.40×，token-normalized 端到端约 1.15×。
- Subsystem / Dimension：training / weight sync / inference backend。证据与边界：已读固定版本 design doc、配置及 patch。上游 Qwen3-30B-A3B-Base、8×4 GB200、900 shared logged steps；rollout EP=PP=1、colocated、GRPO 已验证，backward 为 dequantized 路径，attention/routers/shared experts 等保留 BF16。Refit 从 1.8 s 到 19.2 s；这些是来源报告，不是本地结果，也不能宣称全训练进程峰值显存降低。
- Transfer to AReaL / Next：适合 AReaL 借鉴 source/serving 表示解耦、refit 完成 fence 与失败关闭契约；硬件、量化 granularity、router replay 须重验。补入 P1 低精度阅读入口。

### A4 · verl：关闭 gate 后区分拒绝与等待

- Source ID：`github:verl-project/verl:8050ff113b5b2346e2709ee82c209e03b7c82901`；Type：merged commit；[固定补丁](https://github.com/verl-project/verl/commit/8050ff113b5b2346e2709ee82c209e03b7c82901)。
- 核验标题：[fully_async] fix: reject requests arriving behind the closed gate stead of parking them (#7912)。作者：Joel；committer date：2026-09-18T03:54:09Z（UTC），不混同 release 日期。
- Impact：★★★★☆；Decision：Read；Reason：replica 离开 load balancer 去训练且不会立即 resume 时，迟到请求不能一直 parked；新增 reject_request=True，使其返回 aborted。
- 一句话看点：请求准入不仅要知道 gate 是否关闭，还必须知道 replica 是短暂同步还是已退出服务轮转。
- Subsystem / Dimension：scheduler / rollout / liveness。证据与边界：读取 CPU gate 和 separate-async tests：默认 weight-sync abort 恢复 parking，退出轮转使用 rejection；不要将所有暂停一律改为拒绝。
- Transfer to AReaL / Next：可迁移到 AReaL colocation 的状态机；承接 9/18 resume 完成后才开放 gate 的信号，新增 rejection/parking 验收用例。

### A5 · vLLM：n>1 sampling 原子准入

- Source ID：`github:vllm-project/vllm:a7156060c1fc17c5b31d46d3760829bed8ac1a58`；Type：merged commit；[固定补丁](https://github.com/vllm-project/vllm/commit/a7156060c1fc17c5b31d46d3760829bed8ac1a58)。
- 核验标题：[Core] Make parallel sampling (n>1) reqs admission atomic (#53936)。作者：Nicolò Lucchesi；committer date：2026-09-18T16:08:44Z（UTC），不混同 release 日期。
- Impact：★★★★☆；Decision：Read；Reason：在首个 await 之前登记所有 child request 以占用完整容量，并在提交取消时清理所有 child，防止并发准入交错。
- 一句话看点：GRPO 多样本请求应在准入容量上全进或全拒，取消也要释放整个 child 集合。
- Subsystem / Dimension：inference backend / scheduler / correctness。证据与边界：已核验 all-or-nothing 与 cancellation cleanup 测试；原子 admission 不保证生成完成时所有样本都成功，不替代训练侧 partial-group estimator 规则。
- Transfer to AReaL / Next：AReaL 可借鉴 parent/child slot 记账；后续实验联合检查 capacity reservation、abort 和 group completeness。

### A6 · vLLM：KV offload 背压

- Source ID：`github:vllm-project/vllm:23e26e058839fb2a3e77c78fbf2184f563593227`；Type：merged commit；[固定补丁](https://github.com/vllm-project/vllm/commit/23e26e058839fb2a3e77c78fbf2184f563593227)。
- 核验标题：[KV Offloading] Back-pressure detection and remediation (#50045)。作者：bnellnm；committer date：2026-09-18T18:47:48Z（UTC），不混同 release 日期。
- Impact：★★★★☆；Decision：Read；Reason：新增 store-latency detector 和 drop/throttled-drop policy，对拥塞 secondary tier 少写或跳过写入，记录 stores/blocks dropped。
- 一句话看点：KV 外存扩容必须有背压，否则缓存写入拥塞会转化为服务尾延迟。
- Subsystem / Dimension：inference backend / cache storage / tail latency。证据与边界：已核验 detector/policy、配置继承和 P2P 拒绝测试；P2P 的 rendezvous 时间使 store latency 信号不可靠，该配置会报错，不是通用适配所有传输。未测性能；少存缓存还会增加 miss/recompute 代价。
- Transfer to AReaL / Next：AReaL 接入外部 cache 时可借鉴分层队列和降载指标，需同时比较 TTFT、ITL、miss 与有效 rollout tokens。

### A7 · Megatron：MFSDP v2 接入细粒度 1F1B

- Source ID：`github:NVIDIA/Megatron-LM:ba5e63cff038a201a4bc5adf142b99e8a935c75b`；Type：merged commit；[固定补丁](https://github.com/NVIDIA/Megatron-LM/commit/ba5e63cff038a201a4bc5adf142b99e8a935c75b)。
- 核验标题：Integrate MFSDP v2 with fine-grained 1F1B scheduling (#7112)。作者：Jianbin Chang；committer date：2026-09-20T00:15:28Z（UTC），不混同 release 日期。
- Impact：★★★★☆；Decision：Read；Reason：在细粒度调度中安装 owner-aware unshard/reshard hooks，backward 后归约梯度，并在 optimizer 前等待 reduce-scatter stream。
- 一句话看点：FSDP 与 MoE overlap 的组合需要重新对齐参数生命周期和梯度完成边界，不能仅叠加两个开关。
- Subsystem / Dimension：training / scheduler / communication。证据与边界：已读 adapter/custom hooks 与功能测试配置；有 H100 golden/config 文件不代表本仓库跑过。未发现足以引用的通用速度结论，不给加速百分比。
- Transfer to AReaL / Next：对 AReaL 的 Megatron backend 有间接价值；先确认 scheduler 支持矩阵、内存峰值及 optimizer fence，再考虑启用。

### A8 · SGLang：修复共享 SSD offload 路径的静默损坏（late-discovered）

- Source ID：`github:sgl-project/sglang:329ffc89b9129be3ef9135108cd74f7a699e819b`；Type：merged commit；[固定补丁](https://github.com/sgl-project/sglang/commit/329ffc89b9129be3ef9135108cd74f7a699e819b)。
- 核验标题：[Mooncake] Fix silent SSD offload corruption when TP/PP ranks share ssd_offload_path (#31926)。作者：Michele Palazzi；committer date：2026-09-17T02:51:03Z（UTC），不混同 release 日期。
- Impact：★★★★☆；Decision：Read；Reason：独立 Mooncake client 在同一目录中产生冲突 bucket 文件，O_TRUNC 可截断其他 rank 数据；修复按 DP/TP/PP/attention-CP rank 建私有子目录。
- 一句话看点：缓存的文件命名空间必须对应真实并行 owner，不能让多个 rank 共享可冲突的 bucket 文件。
- Subsystem / Dimension：inference backend / cache storage / correctness。证据与边界：9/17 commit，属于 9/16 起的补扫缺口，非本轮新发布；已读新增路径隔离 CPU tests。此修复不提供跨独立作业的全局目录唯一性保证，部署仍应分离 job 根目录。
- Transfer to AReaL / Next：AReaL 使用 SGLang/Mooncake rollout 时应核查路径隔离；将 owner 身份扩展到 cache 存储检查，不把 cache 文件存在视为数据正确。

A3 的数字来自[固定版本 design doc](https://github.com/NVIDIA-NeMo/RL/blob/3491eed5772425acec5edb3fa5d7adccb23ff6f2/docs/design-docs/te-nvfp4-per-token-rollout.md)，与 main 后续修订区分。

## Observed（5 组）

以下均为 Decision Observe / Status NEW，保留后续审计，不因标题或合并状态自动 Accepted。

| ID / primary source | Impact / Reason / 一句话价值 | Related topics / Next |
|---|---|---|
| O1：[AReaL main](https://github.com/areal-project/AReaL/commits/main/)，#1723/#1724 | ★★★☆☆；AWEX scheduler API 适配与 attributed Arena failure 零 reward 保留可能改变 rollout 可用性和失败样本语义 | rollout/data；先检查支持版本与 failure attribution 测试，不外推为全框架行为 |
| O2：[NeMo main](https://github.com/NVIDIA-NeMo/RL/commits/main/)，#3898/#3935/#4106/#4107 | ★★★★☆；Mooncake storage checkpoint、consolidated save lifecycle、PackedTensor 和避免本地复制是后续数据路径主线 | data/checkpoint；对齐存储持久性与恢复语义，标题审查不足以证明 fault tolerance |
| O3：[slime main](https://github.com/THUDM/slime/commits/main/)，#2390/#2391/#2394 | ★★★☆☆；internal sync、移除 legacy async 入口和 reloadable process group 调整需要按最终支持路径评估 | scheduler/weight sync；先读迁移边界及大提交 diff，不能将 entrypoint 删除等同于删除所有异步能力 |
| O4：[SGLang runtime](https://github.com/sgl-project/sglang/commits/main/)，#40024/#40034/#39464/#34012 | ★★★★☆；shortest-prefill-first、agentic simulator、backpressure 分类及 tail-aware cache 值得围绕真实 rollout trace 联合看 | scheduler/cache；先做负载公平性与尾延迟审计，#34012 为补扫旧项 |
| O5：[Megatron main](https://github.com/NVIDIA/Megatron-LM/commits/main/)，#7265/#7302/#6885 | ★★★★☆；MXFP8 storage/batch invariance 与 FSDP stream memory leak 属重要候选，但尚未完整读 patch | precision/memory；核对适用模型与测试，未测收益，不抢占现有阅读主线 |

## OpenAI / Anthropic / NVIDIA / DeepSeek Watch

| Vendor / source | Triage | 本轮结论 |
|---|---|---|
| [OpenAI News](https://openai.com/news/) | Rejected / Not found | 9/18 Australian Youth Safety Blueprint 为政策内容，Ignore；未在所查目录发现新的可核验训练 infra 文章。之前 online storage 信号不重复。 |
| [Anthropic Research](https://www.anthropic.com/research) | Not found / Rejected | 最新可见 research 仍为 9/17 biomolecular modeling，领域应用不进入主线；未见窗口内新的训练技术报告。 |
| [NVIDIA Blog](https://developer.nvidia.com/blog/) / NeMo / Megatron | Accepted | A2 AIPerf、A3 NVFP4、A7 MFSDP；首页置顶旧文不按新发布日期重复接受。 |
| [DeepSeek changelog](https://api-docs.deepseek.com/updates) + [官方 HF](https://huggingface.co/deepseek-ai) | Accepted late-discovered | changelog 仍为 9/10 发布，HF 最新权重未变，但组织 paper 入口发现 9/17 报告 A1。说明只看权重/更新日志会漏掉后发技术报告。 |

## Hugging Face Watch

- [Blog RSS](https://huggingface.co/blog/feed.xml) 最新可见仍为 9/15 IBM Research 内容；它是 vendor-authored，不等同 HF 团队文章。官方 Async GRPO 实践已于 9/18 收录，本轮 follow-up 不重复计数。
- [TRL](https://github.com/huggingface/trl/releases) v1.13.0、[Transformers](https://github.com/huggingface/transformers/releases) v5.17.0、[Accelerate](https://github.com/huggingface/accelerate/releases) v1.15.0、[PEFT](https://github.com/huggingface/peft/releases) v0.21.0、[Kernels](https://github.com/huggingface/kernels/releases) v0.17.1：release feed 均未见更新。
- 重新读取 [Async GRPO 博客](https://huggingface.co/blog/asyncgrpo-lora-hfjobs)，仍称 TRL v1.14；官方 `releases/tags/v1.14.0` API 本轮返回 404。差异未解决，不能将其写成已发布稳定版，也不能仅凭 release 不存在推断代码未合并；运行前固定代码 SHA。
- 覆盖 Blog、release 及命中文档；未遍历五个项目全部 main PR。No new accepted HF signal，不为了更新数量重复上一轮。

## RL Framework Watch

| Framework / source | Triage / subsystem | Evidence / AReaL transferable / Next |
|---|---|---|
| [AReaL](https://github.com/areal-project/AReaL/commits/main/) | Observed O1；rollout/data | 主窗口 2 commits；AWEX 和 failure reward 先补 tests 核验，不把小修默认接受。 |
| [verl](https://github.com/verl-project/verl/commits/main/) | Accepted A4；scheduler | 主窗口 6 commits；rejection 与 parking 的状态区分可迁移；Ascend refit 与 trainer breaking 配置留在索引待审。 |
| [slime](https://github.com/THUDM/slime/commits/main/) | Observed O3；scheduler/weight sync | 主窗口 5 commits；internal sync 不能只看标题得出架构结论。 |
| [ROLL](https://github.com/alibaba/ROLL/commits/main/) | Not found in this scan | 主窗口 REST 返回 0；不等于没有在其他分支开发。 |
| [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF/commits/main/) | Not found in this scan | 主窗口 REST 返回 0；既有 KL/truncation 深读待办继续保留。 |
| [NeMo RL](https://github.com/NVIDIA-NeMo/RL/commits/main/) | Accepted A3 / Observed O2；training/refit/data | 主窗口 15 commits；量化 owner 与失败关闭契约可借鉴，但 GRPO/GB200 支持边界不可省略。 |

相邻 runtime：Megatron 主窗口 31 commits；vLLM/SGLang 从 9/16 回退分别 218/201 commits，补齐上轮索引缺口。A5–A8 与 O4/O5 记录本轮决策；未发现已完成 runnable-path 核验、需要加入核心名单的新 RL 框架。

## 覆盖、补漏与下一步

- arXiv cs.DC/cs.LG/cs.CL/cs.AI/cs.AR recent 最新日期仍为 9/18，未见周末新公告；不因此推断没有新提交。A1 在旧公告中漏筛，本轮通过官方组织入口发现，已核验 citation_title / citation_author / citation_date / abstract 与 §3/§5；不回写旧 Accepted 计数。
- 所列 GitHub REST commits 以 since 限定，vLLM 3 页（100+100+18）、SGLang 3 页（100+100+1），其余一页，均以不足 100 条结束。证明所选默认分支提交索引覆盖，不证明全部 PR、其他分支或全部 patch 已精读；O1–O5 和旧 NeMo/OpenRLHF 深读项继续保留。
- OpenAI/Anthropic 直接 HTTP 403，已回退网页工具读取官方目录。厂商 Watch 是目录与命中页定向核查，不声称整个站点穷尽；AIPerf 发布日缺少时分秒，单独标记边界。
- 下轮从本轮 Next cursor 继续，arXiv 回看 9/18 公告边界并按 Source ID 去重；HF TRL 版本口径仍需追查。
- [P1](../reading_queue/P1.md)新增 DeepSeek 报告和 NVFP4 优先阅读；保留原 P0 三条与用户学习状态。将 gate mode 与原子容量契约补入 [Agentic RL](../topics/agentic_rl.md#rl-state-boundaries) 和[实验计划](../experiments/rl_state_boundaries.md)，尚未执行，不标 VERIFIED。

导航：[Tracking](README.md) · [Scan Log](scan_log.md) · [Master Reading List](../MASTER_READING_LIST.md) · [Knowledge Graph](../KNOWLEDGE_GRAPH.md)。

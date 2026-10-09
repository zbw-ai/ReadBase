# Frontier Scan · 2026-10-08

## 先读这一段

本轮收录 **13 组 Accepted**：7 篇论文/工程文章、ROLL 与 TRL 两项版本变化、4 组 RL 框架实现进展。建议先看 **Olmo-core 3、VenusRL、TRL 结束 token 修复**；如果正在维护 AReaL，先插入 A10 的 partial-group 审阅。

**整体进展与趋势（仓库推断）**：MoE 优化正在把并行布局、expert 常驻、路由与精度放在一起设计；Agentic RL 的关键问题进一步落到“什么时候有可训练的组、什么时候算真正结束、故障后恢复哪份状态”。这些材料提供的是不同层次的证据，不代表整个行业已经解决相关问题。集群配置验证与任务最终状态验证也值得持续关注。

- 全局窗口：**2026-09-22 16:52:05 → 2026-10-08 10:52:31，Asia/Shanghai**。
- GitHub 合并 PR 索引窗口：**2026-09-22 17:06:16 → 2026-10-08 10:43:00，Asia/Shanghai**；六个核心 RL 框架共 **169 条**，分页完成，含跨分支 cherry-pick，不能当作 169 项独立能力。
- First seen：以下信号均为 **2026-10-08 本仓库首次收录**；不把本次发现日期当发表日期。生命周期均为 **NEW**，不代表精读、复现或迁移完成。
- 证据：[来源与覆盖账本](audits/2026-10-08-frontier/evidence.json)、[169 条合并 PR 索引](audits/2026-10-08-frontier/merged_pr_index.json)。arXiv 三篇已逐项核对 `citation_title / citation_author / citation_date` 和方法正文。
- **部分覆盖**：本轮为定向研究扫描；arXiv 未全分类枚举，GitHub 未完成默认分支直接提交和所有外围仓库 PR 的全量索引。release 网页存在缓存滞后，已用可获得的 REST 结果纠正；不能把“未找到”写成“没有变化”。详见文末续扫游标。

## Accepted：文章与系统设计

<a id="a1"></a>
### A1 · Olmo-core 3：MoE 的瓶颈需要跨层联合处理

- Source ID：`blog:allenai/olmocore3`；来源/类型：Ai2 官方工程文章；作者：Ai2；发布时间：2026-10-01。
- 原文：[Introducing Olmo-core 3: Open, scalable training infrastructure for large MoEs](https://allenai.org/blog/olmocore3)；[官方代码仓库](https://github.com/allenai/Olmo-core)。
- First seen：2026-10-08；Scan window：本报告全局窗口；Impact：★★★★★；Decision：**Deep Dive**；Status：NEW。
- 一句话价值：把 expert 常驻、activation 路由、并行布局与低精度一起优化，解释为什么单独打开 overlap 未必改善训练吞吐。
- Reason：公开训练栈的设计取舍比孤立 kernel 加速更适合建立 MoE 工程判断。

重点读 expert 参数的驻留方式、GPU 上的路由元数据、EP/PP 和 distributed optimizer 如何配合，以及 MXFP8 的转换成本。文章报告的多组对比采用不同模型、硬件与负载；不能混成一个通用加速比。本轮核对官方文章与代码入口，tech report 链接抓取失败，**未完成报告全文和代码路径审计**。

关联：[MoE](../../training-infra/topics/moe.md#october-2026-joint-design)、[FP8](../../systems/topics/fp8.md)。下一步：进入 [P1](../reading_queue/P1.md#october-2026-priority)，先画参数/activation 的驻留和迁移图，再补读技术报告的消融；预留 2h。

<a id="a2"></a>
### A2 · VenusRL：优化凑齐训练组的时间

- Source ID：`arxiv:2610.03286v1`；来源/类型：arXiv systems paper；作者：Mingjun Zhang、Yucheng Li 等（完整 citation 作者见账本）；发布时间：2026-10-02。
- 原文：[VenusRL: A Fully Disaggregated Agentic RL System with Priority Scheduling and Scalable Interaction](https://arxiv.org/abs/2610.03286v1)；[方法正文](https://arxiv.org/html/2610.03286v1)。
- First seen：2026-10-08；Scan window：全局窗口；Impact：★★★★★；Decision：**Deep Dive**；Status：NEW。
- 一句话价值：GPU 一直忙不等于 trainer 有数据，调度应关注完整训练组的就绪时间。
- Reason：将长短轨迹、环境等待、KV 占用和 sample freshness 放入同一条训练关键路径。

训练、生成、环境交互解耦后，长度预测与组级优先级影响 GPU slot、prefill 和 KV 分配；接近 staleness 边界的组获得更高优先级。环境侧另用 microVM 共享与 COW 控制内存。作者的性能结果只适用于论文实验；本轮未验证公开代码、稳定性或可直接迁移性。

关联：[Agentic RL](../../rl-infra/topics/agentic_rl.md#october-2026-contracts)。下一步：进入 [P1](../reading_queue/P1.md#october-2026-priority)，对照 AReaL 记录 ready-group latency、policy age、环境 OOM 和 KV eviction，先判断瓶颈是否相同；预留 2h。

<a id="a3"></a>
### A3 · HAPMoE：异构硬件下联合搜索并行配置

- Source ID：`arxiv:2609.39350v1`；来源/类型：arXiv paper；作者：Mengyuan Fan、Peizhuang Cong 等（完整作者见账本）；发布时间：2026-09-30。
- 原文：[HAPMoE: Heterogeneity-Aware Automatic Parallelism Planning for Mixture-of-Experts Models Training](https://arxiv.org/abs/2609.39350v1)；[方法正文](https://arxiv.org/html/2609.39350v1)。
- First seen：2026-10-08；Scan window：全局窗口；Impact：★★★★☆；Decision：**Read**；Status：NEW。
- 一句话价值：把设备能力、路由不均和通信代价纳入规划，允许不同 pipeline stage 承担不同层数。
- Reason：解决“配置数学上能整除，实际仍被最慢 stage 限制”的训练部署问题。

profile–model–search 联合选择 PP/TP/DP/EP/TPE/CP、stage 放置和重计算，并生成 Megatron 配置。这里的六维是搜索变量，**不能直接相乘当 world size**。方法针对相对稳定的部署窗口；路由分布持续变化需要重新 profile。跨厂商通信实现、模型误差及 launcher 的实用性仍需代码和实机验证。

关联：[MoE](../../training-infra/topics/moe.md#october-2026-joint-design)、[Pipeline Parallelism](../../training-infra/topics/pipeline_parallelism.md)。下一步：按当前集群构造算力/带宽/显存表，检查规划输入是否可测；暂留雷达，60min 选读 §3–5。

<a id="a4"></a>
### A4 · TRANSIT：用 CPU DRAM 换取更少的 GPU

- Source ID：`arxiv:2610.07593v1`；来源/类型：arXiv systems paper；作者：Hyungyo Kim、Nicholas Satchanov 等（完整作者见账本）；发布时间：2026-10-06。
- 原文：[TRANSIT: Transparent Scale-in for Multi-Node LLM Training](https://arxiv.org/abs/2610.07593v1)；[方法正文](https://arxiv.org/html/2610.07593v1)。
- First seen：2026-10-08；Scan window：全局窗口；Impact：★★★★☆；Decision：**Read**；Status：NEW。
- 一句话价值：在显存迫使任务扩到更多节点时，细粒度 host-memory 访问可能同时减少 GPU 需求和跨节点通信。
- Reason：把显存容量、PCIe 与网络代价放入同一资源选择问题。

用户态 interposition 将分配引向 UVM，利用 warm-up 访存记录选择预取或 zero-copy；低复用数据可从 CPU DRAM 直接进入 GPU cache。论文的 **per-GPU throughput 不等于总吞吐**，缩卡保持每卡效率不能推出总训练时间不变。透明接入也不自动证明所有 allocator、CUDA graph 和 runtime 组合兼容。

关联：[ZeRO](../../training-infra/topics/zero.md)、[FSDP](../../training-infra/topics/fsdp.md)。下一步：先列 NUMA/PCIe、host-memory 压力、总 tokens/s、GPU-hours 与排队时间的比较表；暂留雷达，60min 阅读 §3–5，未复现。

<a id="a5"></a>
### A5 · ThinkingBox：验证任务最终状态

- Source ID：`blog:microsoft/thinkingbox`；来源/类型：Microsoft / Hugging Face 联合工程文章；署名：Tuhin Kundu；发布时间：2026-10-03。
- 原文：[ThinkingBox](https://huggingface.co/blog/microsoft/thinkingbox)；[官方仓库](https://github.com/microsoft/thinkingbox)。
- First seen：2026-10-08；Scan window：全局窗口；Impact：★★★★☆；Decision：**Read**；Status：NEW。
- 一句话价值：工具调用合法、agent 自报完成，都不能代替对最终业务状态与副作用的检验。
- Reason：reward/verifier 的错误会直接污染训练样本；它与环境吞吐同样属于 RL infra。

每次尝试使用隔离的任务状态，主要用确定性状态检查，必要时加限定 rubric；重复尝试衡量稳定性。多次均成功是有限样本的观测，不是可靠性保证。发布的评测集不能直接拿来训练后再报告同一评测成绩。

关联：[Agentic RL](../../rl-infra/topics/agentic_rl.md#october-2026-contracts)。下一步：沿已有 MiMo/CodeMidas 环境主线选读 45min，给一个任务增加最终状态、额外副作用、system failure 三类验收；暂留雷达。

<a id="a6"></a>
### A6 · NVIDIA AICR v1.0：配置需要可核验的实际状态

- Source ID：`blog:nvidia/aicr-v1-0`；来源/类型：NVIDIA 官方工程文章；作者：Mark Chmarny、Nathan Taber；发布时间：2026-10-06。
- 原文：[AICR v1.0: Open, Stable, and Verifiable GPU Cluster Configuration](https://developer.nvidia.com/blog/aicr-v1-0-open-stable-and-verifiable-gpu-cluster-configuration/)；[官方仓库](https://github.com/NVIDIA/aicr)。
- First seen：2026-10-08；Scan window：全局窗口；Impact：★★★★☆；Decision：**Read**；Status：NEW。
- 一句话价值：把期望配置、集群实际状态和验证结果分别记录，使“部署完成”有可检查的运行环境依据。
- Reason：升级后性能和稳定性退化，经常需要先确认硬件、驱动、网络及软件版本组合。

Snapshot、Recipe、Bundle、Validation 分别表达观测、期望、交付物和验证。Recipe 本身不负责持续 reconcile，验证证据也只覆盖其声明的环境与检查项。本轮读的是人工正文与官方仓库入口，没有运行工具，不依据页面 AI-generated summary 推导能力。

关联：[Fault Tolerance](../../training-infra/topics/fault_tolerance.md)、[NCCL](../../systems/topics/nccl.md)。下一步：为一次实际集群升级拟定前后 snapshot 与通信验证表；暂留雷达，45min 选读。

<a id="a7"></a>
### A7 · DOCA GPUNetIO：GPU 发起网络操作的公共底座

- Source ID：`blog:nvidia/doca-gpunetio-gda-ki-unified-gpu-networking`；来源/类型：NVIDIA 官方工程文章；作者：Elena Agostini、Simon Schwitanski、Pak Markthub；发布时间：2026-10-06。
- 原文：[How DOCA GPUNetIO Unifies GPU-Initiated Networking Across the NVIDIA Software Stack](https://developer.nvidia.com/blog/doca-gpunetio-gda-ki-unified-gpu-networking)。
- First seen：2026-10-08；Scan window：全局窗口；Impact：★★★★☆；Decision：**Read**；Status：NEW。
- 一句话价值：理解 NCCL、NVSHMEM、NIXL/UCX 的 GPU 发起通信路径时，可以从共享的队列与完成语义入手。
- Reason：训练 overlap 和通信进展机制需要落实到谁提交、谁轮询、何时可复用 buffer。

CPU 仍负责初始化 GPU/NIC 资源，GPU 执行部分提交与 completion 操作。开源 Verbs 路径和完整 DOCA SDK 的能力范围不同；不能写成所有网络功能都开源、都不需要 CPU。本轮核对官方正文，未对通信库的具体版本组合做兼容验证。

关联：[NCCL](../../systems/topics/nccl.md)。下一步：画 CPU setup / GPU doorbell / completion / buffer lifetime 时序，再对照自己使用的 NCCL backend；暂留雷达，60min。

## Accepted：GitHub 实现进展

<a id="a8"></a>
### A8 · ROLL v0.4.0：NCCL 显存释放有明确生命周期

- Source ID：`github:alibaba/ROLL@v0.4.0`；类型：official release + tag code；发布账号：gaow0007；发布：2026-09-29 04:09:29 UTC。
- 原文：[release](https://github.com/alibaba/ROLL/releases/tag/v0.4.0)；[tag 内 nccl_suspend.py](https://github.com/alibaba/ROLL/blob/v0.4.0/roll/utils/nccl_suspend.py)。
- First seen：2026-10-08；Scan window：全局窗口；Impact：★★★★☆；Decision：**Read**；Status：NEW。
- 一句话价值：保留 ProcessGroup 身份，通过原生 suspend/resume 回收 NCCL 动态显存，减少 colocated 阶段切换的状态重建。
- Reason：代码显式处理 runtime 库、collective 顺序与跨角色通信组，比“支持 offload”更具体。

已读 tag 文件：检查实际映射的 libnccl 和符号，代码要求 NCCL ≥ 2.29.7；不支持时明确报错。collective 调用必须在各 rank 同序且 communicator 空闲；`model_update/` 跨角色组不一起 idle，因此跳过。回滚只属 best effort，不能保证任意部分失败都恢复。release 另含 router replay 等变化，本轮不逐项认证。

子系统：training / weight sync；维度：显存、正确性、可运维性。对 AReaL：可借鉴状态边界，不直接复制私有 handle 访问。下一步：先检查真实 NCCL runtime 和跨角色 group 清单，再设计暂停/恢复故障注入；关联 [RL 状态主题](../../rl-infra/topics/agentic_rl.md#october-2026-contracts)，60min 选读。

<a id="a9"></a>
### A9 · TRL v1.14.2：结束 token 会改变实际训练样本

- Source ID：`github:huggingface/trl@v1.14.2`；类型：official release / merged PR；发布账号：qgallouedec；核心 PR 作者：albertvillanova；日期：2026-10-06。
- 原文：[release](https://github.com/huggingface/trl/releases/tag/v1.14.2)；[PR #7505](https://github.com/huggingface/trl/pull/7505)（页面核实 10/06 merged，merge short SHA `062ad83`）。
- First seen：2026-10-08；Scan window：全局窗口；Impact：★★★★★；Decision：**Read**；Status：NEW。
- 一句话价值：只看 tokenizer 的 EOS，可能漏掉 generation config 的结束标记，导致多训尾部文本或错误丢弃完整样本。
- Reason：这是 sampling、mask、指标与 loss 的共同语义错误，不能当作普通小修补。

覆盖 GRPO、RLOO、Distillation；修复将模型声明的结束 ID 用于生成、completion mask 和截断判断。vLLM 生成停止正确，也不代表 trainer 的 mask 正确。来源包含 PR 说明与测试变更记录；未本地运行 TRL 测试，不把 PR 页部分 checks 通过写成完整 CI 通过。

子系统：rollout / data path / training；维度：正确性。对 AReaL：检查生成端、奖励端、训练端对终止和截断的统一定义。下一步：加入 [P1 优先组合](../reading_queue/P1.md#october-2026-priority)的 30min 检查项；[实验设计](../../practice/experiments/rl_state_boundaries.md#october-2026-cases)未执行。

<a id="a10"></a>
### A10 · AReaL v2 partial group：导出集合与清理集合分开

- Source ID：`github:areal-project/AReaL#1721`；类型：merged PR + patch；作者：sitabulaixizawaluduo；合并：2026-09-23 08:09:13 UTC。
- 原文：[PR #1721](https://github.com/areal-project/AReaL/pull/1721)；已读 patch commit `9cfaae1d500c60af329f3109f9c7739835eebc61`。
- First seen：2026-10-08；Scan window：GitHub 索引窗口；Impact：★★★★★；Decision：**Read**；Status：NEW。
- 一句话价值：失败成员可以不参加训练，但原组的所有 session 都必须清理，且成功导出后还要重新检查组是否足够大。
- Reason：同时影响资源泄漏、group normalization 和失败样本选择偏差。

适用 v2 **offline** 路径，支持 strict drop 和 `min_usable_group_size`；online 行为不可类推。patch 中可见 controller 参数传递、data proxy 导出数量复核及测试改动。作者说明未本地运行 pytest 或多机 GPU 集成，本轮也未运行；新增测试代码不等于测试已通过。

子系统：data/trajectory path / rollout；维度：正确性、资源生命周期。相关 [#1749](https://github.com/areal-project/AReaL/pull/1749) 处理 offload/recovery 所有权，属于同一主线的说明级补充证据。下一步：对照当前部署版本核对 partial acceptance、reward denominator 和 cleanup；关联 [topic](../../rl-infra/topics/agentic_rl.md#october-2026-contracts) / [实验](../../practice/experiments/rl_state_boundaries.md#october-2026-cases)，45min。

<a id="a11"></a>
### A11 · verl MoE refit：checkpoint layout 与 kernel layout 不是同一份语义

- Source ID：`github:verl-project/verl#7987`；类型：merged PR；作者：wengeezhang；合并：2026-09-24 05:18:38 UTC。
- 原文：[PR #7987](https://github.com/verl-project/verl/pull/7987)。
- First seen：2026-10-08；Scan window：GitHub 索引窗口；Impact：★★★★☆；Decision：**Read**；Status：NEW。
- 一句话价值：权重同步必须处理 kernel 重排后的参数布局，同时控制 staging 峰值和 CUDA graph 地址稳定性。
- Reason：后端更换可使相同 checkpoint tensor 无法直接写进运行中的 expert 参数。

PR 描述在 live storage 上建立 checkpoint-layout view，再逐层 fold 回运行布局；中间不能运行 forward。完整 layer buffering 还可能保存未分片权重，不能仅按网络 bucket 估峰值。本轮核对 API 的合并时间、作者与详细说明；patch 下载超时，**未独立审计实现**。PR 中内存估计、微型模型测量不能外推所有模型。

子系统：weight sync / inference backend；维度：显存、正确性。对 AReaL：先建立 canonical weight → runtime layout → refit commit 的契约。下一步：补 patch，并做两轮 refit、失败重试和固定输入 logprob 对照；关联 [RL topic](../../rl-infra/topics/agentic_rl.md#october-2026-contracts)，60min。

<a id="a12"></a>
### A12 · slime：训练重启时保留 serving

- Source ID：`github:THUDM/slime#2444`；类型：merged PR；作者：zhuzilin；合并：2026-10-07 01:46:32 UTC。
- 原文：[PR #2444](https://github.com/THUDM/slime/pull/2444)；背景：[distributed async rollout #2410](https://github.com/THUDM/slime/pull/2410)、[Straw checkpoint #2427](https://github.com/THUDM/slime/pull/2427)。
- First seen：2026-10-08；Scan window：GitHub 索引窗口；Impact：★★★★★；Decision：**Read**；Status：NEW。
- 一句话价值：trainer attempt 与 serving 可以有不同寿命，但 replay 的释放必须等到 durable checkpoint，而非仅等训练步骤完成。
- Reason：恢复逻辑开始显式处理 manager 丢失、接收凭据、并行布局变化和权重 baseline。

PR 描述独立 ServingCluster、存活 driver fencing、queue receipts 和共享 batch publication。需要重新提交到同一个存活 Ray 集群；不是自动重试，更不保证集群全失效后保留 serving。作者报告 GPU 故障实验，但 batch 的局部阶段加速不等于端到端加速。本轮读取详细说明；patch 获取失败，未独立复核测试和代码。

子系统：checkpoint/recovery / data path / weight sync；维度：可靠性、内存、恢复成本。对 AReaL：借鉴 attempt/session 所有者分离。下一步：先列 serving owner、trainer、queue、checkpoint 的失效矩阵；关联 [topic](../../rl-infra/topics/agentic_rl.md#october-2026-contracts) / [实验](../../practice/experiments/rl_state_boundaries.md#october-2026-cases)，60min。

<a id="a13"></a>
### A13 · NeMo RL：ready-first 准入与引擎恢复各有边界

- Source ID：`github:NVIDIA-NeMo/RL#4410`，关联 `github:NVIDIA-NeMo/RL#3613`；类型：merged PR；作者：youngeunkwon0405 / Kh4L（#3613 保留原作者 xiuhu17）；合并：2026-10-06 02:20:10 / 2026-09-29 13:44:47 UTC。
- 原文：[ready-first #4410](https://github.com/NVIDIA-NeMo/RL/pull/4410)、[SGLang fault tolerance #3613](https://github.com/NVIDIA-NeMo/RL/pull/3613)。
- First seen：2026-10-08；Scan window：GitHub 索引窗口；Impact：★★★★★；Decision：**Read**；Status：NEW。
- 一句话价值：完成顺序、准入 lookahead 和故障恢复时机要分别建模，一个 staleness 配置不能包办三者。
- Reason：直接影响异步 PPO 的数据选择与训练/生成切换的可恢复性。

#4410 将 ready-first 用于 SingleController full-batch PPO，保留 importance correction 等要求；`max_staleness_versions` 限制准入 lookahead，**不驱逐已经准入的迟到 rollout**。#3613 在下一次 refit 前替换失效的 logical engine group，健康引擎在既有非流式 retry 边界内继续工作；active transfer、已开始的 stream 和 trainer death 不属于保证恢复范围。

证据为 API 合并记录和 PR 详细说明，非独立 diff 审计。#4410 的相关 GPU study 包含其他改动，不能当此 PR revision 的 GPU 验证；#3613 的作者 chaos 结果是生成/恢复测试，不是完整 GRPO 收敛验证。

子系统：scheduler / rollout / checkpoint/recovery；维度：吞吐、freshness、可靠性。对 AReaL：分开测组就绪、准入与恢复，不能直接迁移一个参数名。下一步：对照 A2、A10 阅读并补故障矩阵；关联 [topic](../../rl-infra/topics/agentic_rl.md#october-2026-contracts)，60min。

## Observed / Rejected Candidates

Observed 共 **13 项**；下表每行一项。这些内容保留线索，不自动增加精读队列。共同 First seen 为 2026-10-08，Status 为 NEW；关联主题和下一步写在判断中。

| ID / 来源 | Impact / Decision | 一句话价值、Reason 与下一步 |
|---|---|---|
| O1 [HF RL Environments Hub](https://huggingface.co/blog/rl-environments)，09/28，官方团队含 guest；`blog:hf/rl-environments` | ★★★☆☆ / Observe | taskset 的发现、托管与版本化有用；标签和 load snippet 不证明跨框架 runtime 兼容。关联环境供给，下一步固定 taskset、verifier 和镜像 revision。 |
| O2 [OpenAI / Ironclad](https://openai.com/index/advancing-computer-use-with-ironclad/)，10/06；`blog:openai/advancing-computer-use-with-ironclad` | ★★★★☆ / Observe | 领域专家把完整业务流程变成多条件验收；缺少训练 infra 遥测，时间收益是估计。关联环境/reward，先读 rubric，等待系统成本披露。 |
| O3 [OpenAI training safety cases](https://openai.com/index/towards-safety-cases-for-frontier-ai-training/)，09/28；`blog:openai/towards-safety-cases-for-frontier-ai-training` | ★★★☆☆ / Observe | 保留训练治理线索；尚未深读工程执行机制，不推出集群性能结论。下一步核验 monitoring 与执行控制接口。 |
| O4 [Transformers v5.19.0](https://github.com/huggingface/transformers/releases/tag/v5.19.0)，10/06；`github:huggingface/transformers@v5.19.0` | ★★★☆☆ / Observe | MoE router logits 输出与 attention 配置接口变化值得做兼容检查；未审代码。关联 MoE，下一步核对当前 trainer 的输出读取路径。 |
| O5 [PEFT v0.21.2](https://github.com/huggingface/peft/releases/tag/v0.21.2)，10/01；`github:huggingface/peft@v0.21.2` | ★★☆☆☆ / Observe | encoder-decoder 与 Transformers 的兼容修复，只在相应 workload 下重要。下一步核对依赖矩阵。 |
| O6 [DeepSeek-V4.1-Flash HF #68](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/discussions/68)；`hf:deepseek-ai/DeepSeek-V4.1-Flash#68` | ★★★☆☆ / Observe | 页面确认 chat template PR 已合并，影响 token 序列；只有相对时间，不作为新模型发布。关联训推对齐，下一步固定 revision 对照 encoding。 |
| O7 [vLLM v0.31.0](https://github.com/vllm-project/vllm/releases/tag/v0.31.0)，10/05；`github:vllm-project/vllm@v0.31.0` | ★★★★☆ / Observe | DeepSeek-V4.1 的 KV/路由和启动显存实现继续变化；release 不是本地兼容证据。关联 inference backend，下一步审固定 diff、核支持硬件。 |
| O8 [SGLang v0.5.21](https://github.com/sgl-project/sglang/releases/tag/v0.5.21)，10/02；`github:sgl-project/sglang@v0.5.21` | ★★★★☆ / Observe | 包含 DeepSeek-V4.1 等模型支持；未读完全部实现，不能把版本名当 correctness 验收。下一步按实际 rollout backend 补审。 |
| O9 [AReaL #1750](https://github.com/areal-project/AReaL/pull/1750)，09/26；`github:areal-project/AReaL#1750` | ★★★★☆ / Observe | mean-only reward 与失败归因相关；PR 正文仍列 native receipt fallback 的未解决问题，合并不代表问题已关闭。下一步追查后续修复和真实 harness 合同。 |
| O10 [NeMo RL #3724](https://github.com/NVIDIA-NeMo/RL/pull/3724)，10/07；`github:NVIDIA-NeMo/RL#3724` | ★★★★☆ / Observe | grouped MoE MXFP8 refit 的逻辑 weight、padding 和量化尺度需要一致；作者报告短程 GPU 验证，未独立复核。下一步接到现有低精度 refit 阅读项。 |
| O11 [slime #2443](https://github.com/THUDM/slime/pull/2443)；`github:THUDM/slime#2443` | ★★★★☆ / Observe | 标题提示 linear attention 在 CP>1 的梯度问题；当前仅索引级证据，不据标题断言根因。下一步审 diff 与 gradient parity tests。 |
| O12 [NeMo RL #4129](https://github.com/NVIDIA-NeMo/RL/pull/4129)，09/28；`github:NVIDIA-NeMo/RL#4129` | ★★★★☆ / Observe | token lineage 要等 durable staging 确认后才发布记录；依赖 Gym/Megatron/Bridge 组合，不能只看单库 merged。下一步核实际 submodule pins 与恢复测试。 |
| O13 [NeMo RL #4067](https://github.com/NVIDIA-NeMo/RL/pull/4067)，10/07；`github:NVIDIA-NeMo/RL#4067` | ★★★☆☆ / Observe | 恢复时可以保留 prompt 身份而重新生成，和 replay 原轨迹语义不同。下一步检查数据分布与本 revision 的测试执行证据。 |

Rejected：OpenAI 产品发布和 Anthropic research 目录中的科学/经济应用材料未显示足够的训练系统披露，本轮不因厂商身份自动收录。更早的 OPEN-1B 等候选不计入本轮新增；未核验的论文候选不进入 Accepted。

## OpenAI / Anthropic / NVIDIA / DeepSeek Watch

| Vendor | Sources checked | 结果 |
|---|---|---|
| OpenAI | [Research](https://openai.com/news/research/)、Ironclad 正文、training safety cases 入口 | **Observed**：O2/O3；产品发布 **Rejected**。未获得可新增的集群/训练吞吐证据。 |
| Anthropic | [Research](https://www.anthropic.com/research)、[Engineering](https://www.anthropic.com/engineering) | research 目录可读，近期条目主要是应用/经济/科学，按 focus filter **Rejected**；engineering 的 featured containment 正文抓取失败，标记 **Not found / not verifiable in this scan**，不声称无更新。 |
| NVIDIA | AICR、GPUNetIO 正文；Megatron / NeMo RL release 与 NeMo merged PR 索引 | **Accepted**：A6/A7/A13；Megatron Core 0.19.2 release 页面为 09/18，窗口外；默认分支其他直接提交未全扫。 |
| DeepSeek | [API changelog](https://api-docs.deepseek.com/updates/)、[官方 HF organization](https://huggingface.co/deepseek-ai)、HF #68 | API 页面最新可见为 09/10 V4.1-Flash；HF 确有 chat-template 合并活动，**Observed**：O6。没有把旧技术报告再次当新发布；HF 全模型 commit 历史未枚举。 |

## Hugging Face Watch

| 来源 | 核验结果 / Decision |
|---|---|
| [HF Blog](https://huggingface.co/blog) | **Accepted**：Microsoft 联合文章 ThinkingBox；**Observed**：官方团队含 guest 的 RL Environments Hub。Ai2 官方文章与 HF 同源转载合并计数；社区文章不因登上 Blog 自动收录。 |
| [TRL](https://github.com/huggingface/trl/releases/tag/v1.14.2) | **Accepted A9**。确认 v1.14.2 release；这不能反向证明旧审计中的 v1.14.0 tag 内容，旧版本差异仍保留。 |
| [Transformers](https://github.com/huggingface/transformers/releases/tag/v5.19.0) / [PEFT](https://github.com/huggingface/peft/releases/tag/v0.21.2) | **Observed O4/O5**，不重复计 Accepted。 |
| [Accelerate](https://github.com/huggingface/accelerate/releases/tag/v1.15.0) | 可见 v1.15.0 日期 09/09，窗口外；FSDP2 内存与 checkpoint 改动作为旧版本背景，本轮不冒充新条目。 |
| [Kernels](https://github.com/huggingface/kernels/releases/tag/v0.16.0) | 可见 v0.16.0 页面日期 06/26，且检索缓存较旧；**Not verifiable**：不能由该页断言窗口无新版本，下载时签名强制执行与否也不可从“支持签名”推断。 |

## RL Framework Watch

六库合并 PR 查询采用同一 UTC 窗口，各页 `incomplete_results=false`，NeMo 两页，其余一页；按 PR URL 去重后数量与 `total_count` 相等。索引包含所有命中的合并分支，不等同默认分支 release 覆盖，也不等同全部 diff 已审阅。

| Framework / PR 索引数 | Release / 重点证据 | 子系统 / 工程维度 | 对 AReaL 的参考 / Decision |
|---|---|---|---|
| AReaL / 9 | release 网页 v2.1.0（08/25），REST 请求超时；#1721 patch、#1749/#1750 说明 | data path / recovery；正确性 | **Accepted A10**；成功导出子集与完整组 cleanup 分离。O9 的 receipt 风险保留，不替作者关闭。 |
| verl / 35 | release 网页 v0.9.1（09/20）；REST 下载截断不可作完整结果；#7987 | weight sync / backend；显存与正确性 | **Accepted A11**；逐层 layout 转换与 admission 边界。#7986/#8064 的 FP8 staging/alias 修复可连读，未作独立 Accepted。 |
| slime / 13 | release 网页 v0.3.2（08/28），REST 连接超时；#2444、#2410、#2427 | recovery / data path / weight sync；可靠性 | **Accepted A12**；attempt 与 serving 生命周期分离；CP 梯度问题 **Observed O11**。 |
| ROLL / 1 | REST v0.4.0 09/29；#504 为发布 PR；已读 tag 内 NCCL 文件 | training / weight sync；显存与正确性 | **Accepted A8**；collective 空闲边界、runtime 库匹配、跨角色组排除。 |
| OpenRLHF / 0 | REST 最新 v0.11.2 为 09/14，纠正缓存网页仍显示 v0.11.0 | 无本窗口可审的 merged PR | **Not found in queried index**；不等于没有直接 commit/open PR 活动。 |
| NeMo RL / 111 | REST 最新 v0.7.0 为 07/29；#4410/#3613、O10/O12/O13 | scheduler / rollout / recovery / refit；freshness 与可靠性 | **Accepted A13**，其余保留观察；不能仅扫 release 判断框架停滞。 |
| Emerging：VenusRL | 论文方法已读，公开训练代码未核验 | scheduler / environment；组就绪与内存 | **Accepted paper A2**；尚不作为成熟可运行框架推荐。 |

## 阅读队列与知识回写

- [P1](../reading_queue/P1.md#october-2026-priority)：新增 Olmo-core 3 / VenusRL 两个优先精读项，附 TRL/AReaL correctness 短检查；不扩张 P0。
- [MoE topic](../../training-infra/topics/moe.md#october-2026-joint-design)：补“布局、路由、精度联合决策”，明确规划假设。
- [Agentic RL topic](../../rl-infra/topics/agentic_rl.md#october-2026-contracts)：补 complete/partial group、终止 token、durable recovery 的不同边界。
- [实验候选](../../practice/experiments/rl_state_boundaries.md#october-2026-cases)：只设计对照与验收，未运行；不标 VERIFIED。

## 去重、覆盖与下一游标

13 组 Accepted 的主 Source ID 均为新增；NeMo A13 包含两个 PR，相关支撑 PR 和转载不再加计组数。DeepSeek 技术报告、9 月上旬 release 与更早论文不重复当新材料。GitHub PR 使用 **merged_at** 判断窗口，不能使用 patch 作者日期；例如 AReaL #1721 的 patch 作者日期早于窗口，但 09/23 才合并。

| 来源 | 本轮完成范围 | 下次起点与补扫项 |
|---|---|---|
| 定向文章 / 厂商 / HF | primary 页面、元数据与可读正文；截止 10/08 10:52:31 | 正常增量从该时间继续；Anthropic containment、Olmo 技术报告全文等失败正文按原链接补读。 |
| arXiv 全量发现 | 三篇 Accepted 元数据与方法已核；未全分类枚举 | **发现完整性游标仍从 09/22 16:52:05 回扫、去重**，不能以本次检索代替所有论文已覆盖。 |
| 六个核心 RL 框架 merged PR | 169 条完整查询索引，到 10/08 10:43:00；重点说明审阅，AReaL #1721 patch 深读 | 下一次 merged PR 索引从 **10/08 10:43:00** 接续；动态 PR 正文不是历史内容快照。 |
| GitHub 默认分支直接提交 / 外围仓库 PR / releases 全量 | release 与重点 PR 抽查，部分 REST 超时、截断或网页旧缓存 | **仍从 09/22 17:06:16 回扫**；优先 vLLM/SGLang/Megatron、HF 框架、AReaL/verl/slime release API，以及本轮尚未取得的 patch。 |
| DeepSeek HF 历史 | org 与单个 merged discussion 已核验 | 全模型 commit 历史未关闭；从 09/22 的原边界补查，不能按 relative “updated” 认定新模型。 |

全局 `Next cursor` 为 **2026-10-08 10:52:31**，但以上分来源回退约束优先。既有 2025—2026 H1 历史 GitHub 补证的未完成项也不因本轮扫描而自动关闭。

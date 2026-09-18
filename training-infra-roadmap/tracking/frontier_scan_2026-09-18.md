# Frontier Scan 2026-09-18

- Window：2026-09-16 14:54:26 → 2026-09-18 10:42:01（Asia/Shanghai）。
- 历史补扫：NeMo RL / Megatron-LM / OpenRLHF / HF Blog 从 2026-09-07 10:00:26 回退，按 Source ID 与[上一轮](frontier_scan_2026-09-16.md)去重。
- Accepted：8（7 条窗口内新增，1 条 HF 官方博客 late-discovered）；Observed：9 组，以下 O1–O9 为唯一计数口径，Watch 不重复计数。
- First seen：以下 Accepted 均为本仓库本轮首次发现，登记时间 2026-09-18 10:42:01；不代表互联网首次出现。Status 均为 NEW，未运行上游代码或 GPU benchmark。
- 筛选范围：Training Infra、并行训练、GPU 集群、kernel/precision、Agentic RL 的执行与数据正确性；不按模型热度收录。

## 本轮判断

优先读 BP/CSBP：并行切分可以依据 loss 的依赖结构，而不只是 tensor 的形状。RL 主线继续收敛到“恢复完成后什么时候可以接流量”：版本正确之外，所有 rank 的 scheduler resume 也必须完成。异构训练与集群调度各保留一篇，分别检查 time-to-quality 和碎片整理代价，避免把通信压缩比、空闲 GPU 数直接等同于有效吞吐。

## Accepted

### A1 · Block Parallelism / CSBP

- Source ID：`arxiv:2609.19242v1`；Type：paper；[原文](https://arxiv.org/abs/2609.19242) / [方法与实验](https://arxiv.org/html/2609.19242v1) / [作者代码入口](https://github.com/ScalingIntelligence/Turbo-dLLM)。
- 核验标题：Block Parallelism For Efficient Distributed Long-Context Diffusion Language Model Training。
- Authors：Tarun Suresh, Pranshu Chaturvedi, Hangoo Kang, Parth Shroff, Ishan S. Khare, Hermann Kumbong, Azalia Mirhoseini。citation_date：2026/09/16；v1：2026-09-16 17:35:09 UTC。
- Impact：★★★★★；Decision：Read；Reason：利用 BDLM target-block loss 可分性，将 corrupted block 计算留在 owner rank，CSBP 再切 shared clean context，减少无复用 K/V 的跨 rank 传输。
- 一句话价值：决定“什么根本不需要通信”可以比继续优化同一 collective 更有效。
- 数字边界：作者报告 16 H200、256K 上 SFT 吞吐相对最佳被测 baseline 为 1.18–1.45×；8 H100、1M 的 7.59× 是 DFlash2 drafter training，不能宣传为通用 full-model 或推理加速。原文 §5.1 使用 BF16，baseline 与 CSBP 各自优化并行配置和 batch。
- Related topics：长上下文 / CP / attention / speculative training。Next：进入 [P1](../reading_queue/P1.md#csbp-reading)，先画 clean/corrupted K/V 与梯度归属，再读数值一致性和 load balancing；推理未知 token 不满足该训练前提。

### A2 · GeoMesh

- Source ID：`arxiv:2609.18388v1`；Type：paper；[原文](https://arxiv.org/abs/2609.18388) / [全文](https://arxiv.org/html/2609.18388v1)。
- 核验标题：GeoMesh: Workload-Balanced and Sign-Compressed Geo-Distributed LLM Training。
- Authors：Changyong Shin, Jaerim Park, Minchul Kang, Younghun Go, Zhixiong Niu, Yongqiang Xiong, Gyeongsik Yang, Chuck Yoo。citation_date：2026/09/16；v1：2026-09-16 09:41:39 UTC。
- Impact：★★★★☆；Decision：Read；Reason：同时处理异构 worker 等待和 WAN 同步，以 per-worker batch/inner steps 分配工作，以 sign-based pseudo-gradient 加 magnitude/token count 减少轮间通信。
- 一句话价值：通信与 straggler 要共同计入 time-to-target，而不只比较每轮传输字节。
- 证据边界：§7 明确 WAN 用 Azure 实测带宽加 Linux TC 模拟，没有复现 latency、jitter、丢包或运行中带宽变化；真实跨地域验证仍是 future work。主要实验四个单 GPU worker，附录扩至八个；inner optimizer 与部分 baseline 不同，不能将全部收益归因于压缩。
- Related topics：distributed training / WAN / straggler。Next：保留在 radar，先核对优化器公平性与稳定性，再决定复现；不挤占现有 P0。

### A3 · COMPASS-ABS

- Source ID：`arxiv:2609.18519v1`；Type：paper；[原文](https://arxiv.org/abs/2609.18519) / [全文](https://arxiv.org/html/2609.18519v1)。
- 核验标题：COMPASS-ABS: Reducing Fragmentation in Shared GPU Clusters for Deep Learning Training Workloads。
- Authors：Yukai Zhou, Hongfan Wu。citation_date：2026/09/16；v1：2026-09-16 11:49:06 UTC。
- Impact：★★★★☆；Decision：Read；Reason：用 SIF 区分调度器造成的额外碎片，结合 placement / removal / compaction 约束可达集群状态，而非只在任务到来时 best-fit。
- 一句话价值：总空闲 GPU 足够仍可能无法满足同节点与 gang placement，碎片整理必须计入迁移代价。
- 证据边界：理论 SIF ≤ 2/N 依赖 Workload Composition Condition，并非任意 GPU 需求和拓扑都成立；论文有物理集群合成 workload 与 trace-driven simulation，不能替代用户集群上的收益测量。
- Related topics：GPU cluster scheduling / topology / checkpoint migration。Next：先读 WCC、迁移成本敏感性和非 2 的幂需求处理，保持 radar，不新增当前周任务。

### A4 · verl：先恢复 engine，再开放请求

- Source ID：`github:verl-project/verl:e2ac8f6222801d5e8ce50447b0c3c2d9237e4770`；Type：merged commit；[补丁](https://github.com/verl-project/verl/commit/e2ac8f6222801d5e8ce50447b0c3c2d9237e4770)。作者 lxb007981；2026-09-18 01:51:07 UTC。
- 标题：`[vllm] fix: prevent scheduler resume race during async weight sync (#7846)`。
- Impact：★★★★★；Decision：Read；Reason：KV wake-up 隐式 resume 后再次显式 resume，新 DP wave 可让各 rank 的 engines_running 不一致，一部分进 resume collective，另一部分进 model collective，最终 Gloo timeout。
- 已核验代码：replica 先 await head server 的 `resume_engine_generation()`，返回后才向所有 server 发 `open_submission_gate()`；不能简单删掉显式 resume，因为跳过 KV restore 的路径仍需要它。
- 一句话价值：weight version 一致还不够，请求准入必须晚于跨 rank 恢复完成。
- Subsystem：scheduler / weight sync / inference backend；Dimension：liveness / correctness。Transfer to AReaL：可迁移两阶段准入契约，不能直接照搬 vLLM 版本相关 RPC。
- Next：并入 [P1 状态边界组合](../reading_queue/P1.md#rl-state-boundaries-reading)、[topic](../topics/agentic_rl.md#rl-state-boundaries)及[实验计划](../experiments/rl_state_boundaries.md)。上游描述给出 fully async GLM-5.2 验证，本仓库未复现。

### A5 · AReaL：逐 interaction reward 一次提交

- Source ID：`github:areal-project/AReaL:179ff1bf80796ec8797cea3dc2dcf8f46beef6cb`；Type：merged commit；[补丁](https://github.com/areal-project/AReaL/commit/179ff1bf80796ec8797cea3dc2dcf8f46beef6cb)。作者 Henry Su；2026-09-16 07:06:49 UTC。
- 标题：`fix(v2): preserve per-interaction agent rewards (#1662)`。
- Impact：★★★★☆；Decision：Read；Reason：默认 `set_reward_finish_timeout=0` 下逐条发送 reward 会让第一条请求先 finalize，后续出现 HTTP 400 并丢弃 rollout group；新实现将 rewards map 在一次请求、一次锁内写入，最多 finalize 一次。
- 一句话价值：reward 的提交粒度必须与 trajectory 生命周期一致，否则吞吐和统计语义会同时受损。
- Subsystem：data/trajectory path；Dimension：reward attribution / atomicity。Transfer to AReaL：直接适用于该 v2 路径；已读 controller/data proxy/session 与新增测试，不能外推为全部旧版本行为。
- Next：加入已有状态边界实验的 reward transaction 用例；记录 interaction IDs、各步 reward、finalize 次数，尚未执行。

### A6 · NeMo RL：流式 vocab-parallel draft soft CE

- Source ID：`github:NVIDIA-NeMo/RL:8241b9f6f4caf162e35b785142922d934e108d86`；Type：merged commit；[补丁](https://github.com/NVIDIA-NeMo/RL/commit/8241b9f6f4caf162e35b785142922d934e108d86)。作者 Seonjin（Git 签名 seonjinn）；2026-09-17 09:38:19 UTC。
- 标题：`feat(draft): stream vocab-parallel soft CE (#3703)`。
- Impact：★★★★☆；Decision：Read；Reason：token tile 内生成 FP32 distribution，跨 TP 归约归一化项，新增 loss/wrapper 与 distributed tests，把 soft-label loss 的临时张量纳入显存控制。
- 一句话价值：长上下文 draft 训练不能只优化 attention，还要限制 vocab 维概率临时量。
- Subsystem：training；Dimension：activation memory / numerical correctness。Transfer to AReaL：可借鉴分块 CE 与并行归一化，但需匹配 TP group、mask 和 reduction contract。
- Next：作为上轮 draft co-training 的实现 follow-up；`policy.draft.token_chunk_size` 默认 4096 是配置值，较小值与更多 launch 的取舍需实测。该组件已合并不代表整套 co-training pipeline 已发布或验证。

### A7 · Megatron-LM：dataloader checkpoint 归属与写入顺序

- Source ID：`github:NVIDIA/Megatron-LM:919d381b28b5936e27bc6504b0277f67c3e9242e`；Type：merged commit；[补丁](https://github.com/NVIDIA/Megatron-LM/commit/919d381b28b5936e27bc6504b0277f67c3e9242e)。作者 Deepak Narayanan（patch 另列 AI co-author）；2026-09-17 10:40:31 UTC。
- 标题：`Fix dataloader checkpoint rank indexing and write ordering (#7413)`。
- Impact：★★★★☆；Decision：Read；Reason：loader 按 DP × gtp_remat 分片，用 replicate DP rank 命名会使不同数据 shard 写同一文件；将 loader state 保存移到 checkpoint 写入调用之后，避免预先占用同一 iteration 目录触发 partial-checkpoint 判断。
- 一句话价值：恢复模型还不够，数据游标的 owner 必须匹配真实分片轴。
- Related topics：checkpointing / data parallelism / fault tolerance。Next：核对 `dp_gtp_remat_group`、rank-independent loader 的单写者契约与测试；这个调用顺序本身不证明所有异步存储具有事务性。

### A8 · HF Async GRPO + LoRA 跨 Jobs 实践（late-discovered）

- Source ID：`hf-blog:asyncgrpo-lora-hfjobs`；Type：HF 官方团队 engineering blog；[原文](https://huggingface.co/blog/asyncgrpo-lora-hfjobs)。
- 标题：Async GRPO with LoRA across HF Jobs: a bucket, a proxy, and no NCCL。作者：Amine Dirhoussi, Quentin Gallouédec, Kashif Rasul, Sergio Paniego；页面日期 2026-09-10，属于上轮来源覆盖缺口，本轮补扫首次收录，不算 9 月 18 日新发布。
- Impact：★★★★☆；Decision：Read；Reason：adapter 经共享 bucket mount 发布，proxy 分发加载并维持 prefix affinity；每个 policy version 使用不同 adapter 名，避免旧 KV 与新权重混用，并给 in-flight rollout 留旧版本。
- 一句话价值：小 adapter 降低 weight sync 成本后，版本命名、缓存身份、存储可见性和 adapter 槽位成为关键正确性条件。
- 证据边界：文章标注 TRL v1.14，但本轮官方 releases feed 最新仍为 v1.13.0；按文章配方记录，不声称 v1.14 已稳定发布。文中 atomic rename 也不能自动证明任意 FUSE/object-store mount 的跨节点原子可见性。
- Related topics：Agentic RL / LoRA / policy staleness / KV cache。Next：与已有 P1 状态边界阅读组合交叉阅读，运行前固定含所需代码的 SHA 和 vLLM 版本；未复现文中运行时长，不搬运为通用加速结论。

## Observed（9 组）

以下 Decision 均为 Observe、Status NEW；保留问题，不等于已完成源码审计。

| ID / Source / Type | Impact / Reason / 一句话价值 | Related topics / Next |
|---|---|---|
| O1：[AReaL VLM CPU broadcast](https://github.com/areal-project/AReaL/commit/518d2ff983983e22cbf4ffa18debaa57af021f89)，commit，9/17 | ★★★☆☆；VLM CPU broadcast 与 microbatch memory 可成为非 GPU 算子瓶颈，标题级筛选 | data path；连同 processor reuse 跟读 patch 后判断，不凭 perf 标题接受 |
| O2：[NeMo Energon packing](https://github.com/NVIDIA-NeMo/RL/commit/946b1970f50ff5714a2ec9dfacbdcc1017cf27de)，commit，9/18 | ★★★★☆；SFT packing 的 owner 转入数据层，需核对边界与恢复游标 | training/data；下次先看 packed sample 对齐与断点恢复 tests |
| O3：[NeMo data-plane telemetry](https://github.com/NVIDIA-NeMo/RL/commit/fb121a00d6b49cffd59e9e13e6c92215008b948e)，commit，9/17 | ★★★☆☆；时间、分位数、字节量有助于区分序列化与传输成本，尚无收益验证 | data/trajectory path；核对计时边界，再迁移到 AReaL tracing |
| O4：[MFSDP owner planning](https://github.com/NVIDIA/Megatron-LM/commit/2c897b8bf104be4f5795737dbeff9bd54741f99a) 与 [hybrid FSDP](https://github.com/NVIDIA/Megatron-LM/commit/29694c26ec7efdeb5b9cd3aed4501ddb23be8f9b)，commits | ★★★★☆；新规划 API 与 reduction/padding 变更需放在完整执行计划中理解 | training/communication；连同补扫的 GTP/FP4/CP 历史一起审计，避免逐提交自动 Accepted |
| O5：[NeMo 历史提交](https://github.com/NVIDIA-NeMo/RL/commits/main/)，9/7–9/16 gap replay | ★★★★☆；补到 rollout checkpoint、Mooncake GDR、M-to-N refit 及 actor teardown ownership，不能仅凭目录宣布恢复可靠 | checkpoint/recovery、weight sync；优先 #3923/#3924 与 #3739，按固定 SHA 深读；上轮已接受项不重复 |
| O6：[OpenRLHF 历史提交](https://github.com/OpenRLHF/OpenRLHF/commits/main/)，9/7–9/16 gap replay | ★★★★☆；reverse-KL gradient、terminal oversampling buffer 和 truncation 传播具有训练正确性后果 | training/data；复核 #1309/#1321/#1327 tests，并与既有信号去重；当前窗口 README 更新 Ignore |
| O7：[vLLM KV release API](https://github.com/vllm-project/vllm/commit/092bdd6d57ac7c1cd5272339c80372053fd51bbe) / [SGLang P-D role switching](https://github.com/sgl-project/sglang/commit/1f60ddef5dc2ae3bbfbe0c5cea45690c4b60a251)，commits | ★★★★☆；涉及显存生命周期和运行中角色切换，但未读完契约，不能把 API 出现等同于 RL 可直接接入 | inference backend/scheduler；检查 quiesce、KV ownership、in-flight request 与失败恢复 |
| O8：[OPEN-1B](https://arxiv.org/abs/2609.17380)，paper，v1 9/15，boundary late-discovered | ★★★★☆；公开逐步可审计训练轨迹，涉及跨硬件确定性；尚未完成代码与代价审计 | reproducibility；检查 kernel/collective 顺序约束、存储成本，元数据未按 Accepted 全流程核验故只 Observe |
| O9：[CUDA Toolkit 13.4](https://developer.nvidia.com/blog/cuda-toolkit-13-4-adds-windows-on-arm-support-and-greater-control-over-shared-gpus/)，官方 blog，9/9，补扫观察 | ★★★★☆；正文涉及 MPS V3、locality domains、CFT，训练共享资源控制值得跟进 | GPU resource isolation / communication；先核对对应正式 API、支持硬件与限制，不把首页置顶当新发布日期 |

## OpenAI / Anthropic / NVIDIA / DeepSeek Watch

| Vendor / primary entry | Triage | 核验结果与下一步 |
|---|---|---|
| [OpenAI News](https://openai.com/news/) | Rejected / Observed | 9/17 Astra for Law、广告/商业价值文章无直接训练 infra 后果，Ignore；9/16 model misalignment reporting 仅目录级观察，本轮不作为 infra Accepted，不推断未披露训练细节。 |
| [Anthropic Research](https://www.anthropic.com/research) | Rejected | 9/17 biomolecular modeling 为领域应用信号，未发现足够训练系统机制，不纳入主线；之前 security 材料不重复。 |
| [NVIDIA Blog](https://developer.nvidia.com/blog/) / NeMo / Megatron | Accepted / Observed | A6/A7 为固定源码证据；CUDA 13.4 为 O9。NVLink 6 resiliency 已在上一轮 Observe，NVHBM/BlueField 首页卡片为旧文章，不当本轮新品。未发现已核验的窗口内新训练技术报告。 |
| [DeepSeek API changelog](https://api-docs.deepseek.com/updates) + [官方 HF 组织](https://huggingface.co/deepseek-ai) | Observed | 两处最新可见相关发布仍为 9/10 V4.1-Flash，承接上轮 A1；本轮未见可核验的新权重或训练报告。运行时 DSv4 支持修复不等于厂商又发布了模型。 |

## Hugging Face Watch

- [官方 Blog RSS](https://huggingface.co/blog/feed.xml) 与正文：回退读到 9/7 之前，补入 A8；9/15 IBM Research agent consistency 为 vendor-authored 内容，不能等同 HF 团队自己的研究，也不因首页展示自动接受。Gradio 与领域 safety 文章无本轮核心 infra 后果，Ignore。
- [TRL releases](https://github.com/huggingface/trl/releases)：最新可见 v1.13.0（9/10），与 A8 正文 v1.14 口径不一致，保留版本核验项。
- [Transformers](https://github.com/huggingface/transformers/releases)：v5.17.0（9/10）；[Accelerate](https://github.com/huggingface/accelerate/releases)：v1.15.0（9/9）；[PEFT](https://github.com/huggingface/peft/releases)：v0.21.0（9/15），均在主窗口前，不重复接受。
- [Kernels v0.17.1](https://github.com/huggingface/kernels/releases/tag/v0.17.1)（9/16 UTC）：已读 release body，仅 `egg_info.writers` entry-point module path 修复；Decision Ignore，不当作新 kernel 性能信号。
- 覆盖口径是 Blog 与 release/docs-linked evidence；未声称遍历五个项目全部 main 提交。

## RL Framework Watch

| Framework | Triage / subsystem / dimension | Evidence / Transfer to AReaL / Next |
|---|---|---|
| [AReaL](https://github.com/areal-project/AReaL/commits/main/) | Accepted A5；Observed O1；data/trajectory correctness、VLM memory | 已读 reward patch/tests；batch reward commit 适用该 v2 路径，VLM 性能尚待量化。 |
| [verl](https://github.com/verl-project/verl/commits/main/) | Accepted A4；scheduler / weight sync / liveness | engine-resume 完成后再开放所有 submission gate，可迁移状态契约。 |
| [slime](https://github.com/THUDM/slime/commits/main/) | Not found in this scan | Atom 最新可见仍为 9/3，未见窗口内 material change；不代表框架停止维护。 |
| [ROLL](https://github.com/alibaba/ROLL/commits/main/) | Not found in this scan | Atom 最新可见仍为 8/27，未见窗口内 material change。 |
| [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF/commits/main/) | 当前 README Ignore；历史 O6 Observe；training/data correctness | REST 回退覆盖 28 条，KL/oversampling/truncation 契约待逐项源码核验；迁移前匹配 estimator。 |
| [NeMo RL](https://github.com/NVIDIA-NeMo/RL/commits/main/) | Accepted A6；O2/O3/O5 Observe；training、data、recovery | REST 回退覆盖 55 条；流式 CE 可借鉴，历史 checkpoint/refit 不能凭 PR 标题认定可靠。 |

相邻运行时 [Megatron-LM](https://github.com/NVIDIA/Megatron-LM/commits/main/) 为 A7/O4；vLLM/SGLang 为 O7。未发现已完成 runnable-path 核验、值得本轮新增到核心 watchlist 的独立 RL 框架。

## Coverage 与下一次起点

1. arXiv：读取 cs.DC/cs.LG/cs.CL/cs.AI/cs.AR recent 列表，相关候选再开 abs；A1–A3 对照原始 HTML 的 citation_title / citation_author / citation_date / abstract 与全文方法。recent 是公告快照，不保证覆盖截止前所有尚未公告提交；下轮回看边界并去重。
2. GitHub：本轮 REST 恢复。Megatron 从 9/7 10:00:26 起两页 100+22 条，NeMo 55 条、OpenRLHF 28 条；分页返回不足 100，已补齐这些来源的主分支提交索引缺口。**索引覆盖不等于所有 material patch 已深读**；O4–O6 保存待审内容。
3. AReaL/verl/slime/ROLL Atom 的最旧记录跨过主窗口起点。vLLM/SGLang 仅最新 20 条，最旧记录仍晚于起点，当前窗口仍有历史缺口；下一轮二者必须从 2026-09-16 14:54:26 回退，优先 REST 分页，不能用本轮最新时间替代完整覆盖。
4. HF RSS 回退覆盖上轮官方目录缺口，A8 明示 late-discovered；NVIDIA 首页置顶与分类卡片不能当时间排序。网页工具超时时已回退直接读取官方 HTML，厂商观察是所列目录与命中页的定向检查，不声称所有域名页面穷尽。
5. Next cursor：2026-09-18 10:42:01；新窗口从这里续扫，另按第 1/3 项回退。未解决：vLLM/SGLang 完整 main 历史、TRL 博客与 release 版本差异、O4–O6 深读。

## 流转

- 新增 [CSBP P1](../reading_queue/P1.md#csbp-reading)，其余主线不扩张 P0。
- A4/A5 的状态契约补入 [Agentic RL](../topics/agentic_rl.md#rl-state-boundaries) 与[故障注入计划](../experiments/rl_state_boundaries.md)，没有运行结果，不标 VERIFIED。
- [9 月维护记录](../learning_log/2026/2026-09.md) · [Scan Log](scan_log.md) · [Tracking](README.md) · [Knowledge Graph](../KNOWLEDGE_GRAPH.md) · [Master Reading List](../MASTER_READING_LIST.md)。

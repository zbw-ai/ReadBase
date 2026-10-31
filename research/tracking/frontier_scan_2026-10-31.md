# Frontier Scan · 2026-10-31

[研究雷达](README.md) · [扫描账本](scan_log.md) · [证据与覆盖](audits/2026-10-31-frontier/README.md) · [P1 阅读动作](../reading_queue/P1.md#oct31-frontier)

- Window / 下一游标：**2026-10-23 11:09:27 → 2026-10-31 22:20:20，Asia/Shanghai**（UTC 03:09:27 → 14:20:20）。查询使用固定事件 cutoff；后续正文核验时间见审计请求记录，不扩大事件窗口。
- **15 组 Accepted、18 组 Observed；另 1 篇晚发现材料按 10 月 backfill 保存**。组内来源及跨 Watch 引用不重复计数。First seen 均为 2026-10-31；以下 Accepted 均 Status **NEW**，没有运行上游训练、GPU benchmark、仿真或真机，不表示用户已精读。
- 实际覆盖：arXiv cs.RO / cs.LG / cs.DC / cs.AR recent 目录 **2,002 个去重 ID**，按题名筛选、重点摘要与方法核验；不是 2,002 篇全文阅读。目录列出 10/23、26、27、28、30 批次，不能据此声称已覆盖缺失公告日。cs.CV / cs.CL 多次超时或连接失败。
- GitHub：24 个指定仓库 release 各取最多 30 条；10 仓 merged search 返回 153 条且 returned=total、incomplete=false；其余 search 限流。ROLL / OpenRLHF / NeMo RL 补取 closed PR 首 100 条，runtime / TorchCodec 等补取 Atom 快照；**不称完整全窗口 PR 审计**。重点 PR 阅读页面及 patch，patch 含中间提交，最终行为以已合并页面与后续修正为边界。

## 本次核心判断

本轮证据支持的学习方向是：先区分 latent motion、可执行动作和真实闭环，再决定如何扩展数据。World Model 的视频指标、熟悉任务收益、失败边界准确性需要分开验收。工程上，分片训练开始覆盖 adapter 与 EMA，RL 则继续修正采样支持集、权重布局和恢复身份。以上为本轮材料归纳，不是行业全景或性能实测结论。

## Accepted Frontier Signals

所有条目的 Scan window / First seen / Status 使用页首共同字段；作者完整列表及 citation metadata 见 [papers.json](audits/2026-10-31-frontier/papers.json)。日期按原始来源标示；PR 记录的是合并事件，不把旧提案日期当本轮首发。

<a id="a1"></a>
### A1 · RynnVLA-Latent / RynnWorld-Latent：共享运动监督，不等于共享机器人控制

- Source ID：`arxiv:2610.34356v1`、`arxiv:2610.36791v1`；paper + 官方代码 README；作者均 Haoyu Zhao 等 14 人；v1 分别 **10/25 15:05:32 UTC、10/27 07:54:16 UTC**。
- Title / 来源：[RynnVLA-Latent: Scaling Robotic Manipulation Foundation Models with 3D-Grounded Latent Actions](https://arxiv.org/abs/2610.34356v1)；[RynnWorld-Latent: A Cross-Embodiment World Model Conditioned on Universal Latent Actions](https://arxiv.org/abs/2610.36791v1)。[VLA 代码](https://github.com/alibaba-damo-academy/RynnVLA-Latent) / [World 代码](https://github.com/alibaba-damo-academy/RynnWorld-Latent)。
- Impact：High；Decision：**Deep Dive**；Learning track：Embodied / Training；Signal type：model / algorithm。
- Target question / Reason：没有机器人 action label 的人类视频，怎样成为 policy 和 action-conditioned world model 的共同监督？RynnLAM 以未来特征残差、相机补偿 3D flow、相机运动解耦构造连续 latent；VLA 用 flow matching 预测 latent chunk，再以目标机器人动作和 proprioception 做 post-training；World 分支以全局 action tokens 与逐帧对齐局部调制预测未来视频。
- Evidence：两篇正文 §3–4、局部消融及官方 README 已核。模型价值是把视觉变化变为中间监督；**latent 不是 native command**。World 的合成路线还需要首帧编辑、retargeting / IK、时间对齐与过滤，真实动作和生成观察并非同一物理过程产生。
- 边界：World 论文的 23 数据集 / 553.5M frames 与当前 README 的 32 / >265M 不一致，不能合并成统一规模事实。VLA 摘要与正文的 RoboTwin 数字也有口径差异，本轮不引用排行榜数字。README 明确公开的 VLA Stage-1 权重不能直接当闭环控制 policy；未下载权重或跑代码。World 真机每任务 35 trials、额外合成训练同时增加算力，不能归因成纯数据收益；视觉 action following 不证明物理正确。
- Related topics：[模型与动作](../../05-embodied-infra/topics/embodied_models_primer.md)、[数据 I/O](../../05-embodied-infra/topics/embodied_data_io.md)、[机制回写](../../05-embodied-infra/topics/embodied_infra.md#oct31-learning)。Next：P1，画 video→latent→native action 与 video→latent→synthetic episode 两条链，分别标记监督来源、时间跨度和验收边界。
- 一句话价值：跨具身扩展可以先统一“运动监督”，但执行接口与合成数据的动作一致性仍须单独证明。

<a id="a2"></a>
### A2 · World Modeling 的收益：训练辅助目标与泛化要分开

- Source ID：`arxiv:2610.34593v1`；[What Does World Modeling Actually Improve in Robot Action Learning?](https://arxiv.org/abs/2610.34593v1)；Hanwen Cui 等 7 人；paper；v1 **10/25 21:47:19 UTC**，另有 10/27 v2，本轮方法判断限定已读 v1。
- Impact：High；Decision：**Read**；Learning track：Embodied / Training；Signal type：algorithm。
- Target question / Reason：WAM 的提升来自更好的训练监督，还是必须在部署时生成视频？正文在 FastWAM / OpenWAM 各自内部固定结构、初始化、数据与步数，开关 future-video loss；另给 π0.5 增加仅训练期使用的轻量 video predictor，推理时删除。
- Evidence：§3、§5、§6。轻量分支使用冻结 Wan VAE 的 latent 作为目标，并非不使用预训练视频表示。已见任务收益与 unseen task / 环境随机化表现分开报告。
- 边界：全部在 RoboTwin 2.0 仿真；单训练 seed，许多 rollout 不能代替多 seed。轻量分支与大型 Wan 分支同时改变结构、初始化和 loss 权重，不能据此证明“小模型胜过大模型”；未给系统训练成本和推理延迟的完整对照。
- Related topics：[具身模型](../../05-embodied-infra/topics/embodied_models_primer.md)、[评估](../../05-embodied-infra/topics/embodied_sim_evaluation.md)。Next：P1，用固定 action pathway、相同样本与预算设计辅助预测消融；区分 seen / unseen / perturbation。
- 一句话价值：future prediction 可以只在训练时提供监督，但熟悉任务成功率提高并不保证分布外控制更可靠。

<a id="a3"></a>
### A3 · WorldFailBench：不能只检查世界模型是否猜对成功或失败

- Source ID：`arxiv:2610.33565v1`；[WorldFailBench: Do World Models Understand Policy Failures?](https://arxiv.org/abs/2610.33565v1)；Zheng Zhang 等 14 人；paper / benchmark；v1 **10/24 09:33:52 UTC**。
- Impact：High；Decision：**Read**；Learning track：Embodied；Signal type：algorithm / system。
- Target question / Reason：生成环境若把“抓住后掉落”模拟成“抓错物体”，最终都失败，是否仍能用来诊断 policy？Evidence：方法将 failure reason、failure stage、failure boundary 分开，与参考执行的频率和跨 policy 相关性比较；边界任务使用仿真中受控场景扰动。
- 边界：LIBERO、RoboTwin 与 G1 实验不支持所有任务泛化；failure boundary 来自仿真，不能改写成真机安全边界证明。失败原因标签描述可观察行为，不是模型内部因果解释；预算、标注和 reference noise 也影响结果。
- Related topics：[仿真评估](../../05-embodied-infra/topics/embodied_sim_evaluation.md)、[机制回写](../../05-embodied-infra/topics/embodied_infra.md#oct31-learning)。Next：P1，与 A2 配对阅读；给评估表增加原因、阶段、扰动方向和成功率变化幅度。
- 一句话价值：一个可用的生成式评估器，必须保留 policy 怎样失败以及在哪些条件下开始失败。

<a id="a4"></a>
### A4 · Same Label, Different Interface：action 名称不足以定义实验

- Source ID：`arxiv:2610.38104v1`；[Same Label, Different Interface: Beyond Action-Space Labels in Robot Learning](https://arxiv.org/abs/2610.38104v1)；Arshia Sangwan、Zhongyang Zhang；paper；v1 **10/28 00:41:57 UTC**。
- Impact：High；Decision：**Read**；Learning track：Embodied；Signal type：algorithm。
- Target question / Reason：两个模型都输出 joint position，为何迁移后行为完全不同？Evidence：固定 learner 与预算，在 Factory 接触任务改变物理 scale 和动作 EMA，再比较匹配局部 raw-torque command gain 的 position / velocity 接口；匹配一个局部量仍不匹配 feedback 与 held-action dynamics。
- 边界：单仿真 Franka、三任务、同一 PPO 配方，非真机或通用控制器排名；动作 EMA 与模型参数 EMA 是两个不同概念。
- Related topics：[动作执行](../../05-embodied-infra/topics/embodied_runtime.md)、[数据契约](../../05-embodied-infra/topics/embodied_data_io.md)。Next：Read，补写输出归一化→物理缩放→滤波→参考保持→feedback→effort limit 的完整接口表，再比较学习效果。
- 一句话价值：action-space label 相同，不代表机器人收到相同的物理控制信号。

<a id="a5"></a>
### A5 · LeRobot：FSDP2 / PEFT / EMA 开始形成完整训练与恢复链

- Source ID：`github:huggingface/lerobot#5086`、`#5095`、`#5117`；HaomingSong / Hugging Face LeRobot；merged PR；合并 **10/29、10/30、10/30 UTC**，merge SHA 分别 `89cc0a8`、`f337b8c`、`724b1b6`。
- 来源：[PEFT under sharded training](https://github.com/huggingface/lerobot/pull/5086)、[MixedPrecisionPolicy / sync_each_batch](https://github.com/huggingface/lerobot/pull/5095)、[native EMA](https://github.com/huggingface/lerobot/pull/5117)。Impact：High；Decision：**Deep Dive**；Learning track：Embodied / Training / Systems；Signal type：system。
- Target question / Reason：模型分片后 adapter、混合精度、EMA 和 checkpoint 是否仍有一致语义？Evidence：页面和 patch 的配置、checkpoint、训练与测试路径已核。PEFT 先构造 base / adapter 再在 prepare 后载入分片；checkpoint 存 adapter 并保留 base 路径。精度配置分开 param / reduce / output / cast_forward_inputs；sync_each_batch 用更多通信换梯度分片驻留。EMA 使用与参数相同 placement 的 FP32 average，DCP 保存恢复。
- 边界：这些是 main 合并事件，不是新的 LeRobot release。EMA 默认关闭，启用后周期评估和发布权重选择会改变；需要核对 online / EMA 版本。完整 safetensors 导出仍有 gather 成本；可训练状态减少不等于 activation 成本减少。上游测试为来源报告，本仓未运行。
- Related topics：[具身训练回写](../../05-embodied-infra/topics/embodied_training.md#oct31-precision-state)、[FSDP](../../02-training-infra/topics/fsdp.md)。Next：P1，验证 resume 前后 adapter、EMA update count、optimizer 与样本顺序；锁定具体 commit。
- 一句话价值：能把模型放进多卡只是起点，adapter、EMA、精度与恢复状态必须一起闭合。

<a id="a6"></a>
### A6 · LeRobot policy 集成：模型能力与可执行工件需要分别验收

- Source ID：`github:huggingface/lerobot#5008`、`#5049`、`#4897`；LeRobot contributors；merged PR；合并 **10/29、10/23、10/23 UTC**；merge SHA `c386ba8`、`d3793b7`、`8cd02c1`。
- 来源：[native Long-WAM](https://github.com/huggingface/lerobot/pull/5008)、[Faster-WAM](https://github.com/huggingface/lerobot/pull/5049)、[LingBot-VLA 2.0](https://github.com/huggingface/lerobot/pull/4897)。Impact：High；Decision：**Read**；Learning track：Embodied / Inference / Training；Signal type：model / system。
- Target question / Reason：公开模型进入统一 trainer/runtime 后，什么会破坏训练—执行一致性？Evidence：Long-WAM 区分先视频后动作 IDM 与联合 denoising COD，使用标准 processor、保存统计，reset / 指令变化清理历史和动作队列；Faster-WAM 将多层 video KV 做融合，浅 action expert 重复 denoise；LingBot 保留训练期 teacher 监督与推理期 query tokens，并修复 canonical action slot 与 checkpoint 转换。
- 边界：Long-WAM 当前核验路径为同步，不能声称 async / RTC 已验证。Faster-WAM PR 的速度测试使用随机 video/action 权重，不能拿其倍数当训练后 policy 成功率或端到端机器人收益；架构改变需要对应训练权重。LingBot 的 upstream parity、CUDA graph 和 smoke 数字属于提交者指定版本报告，本仓只做源码核查。
- Related topics：[模型](../../05-embodied-infra/topics/embodied_models_primer.md)、[执行](../../05-embodied-infra/topics/embodied_runtime.md)。Next：Read，先做 checkpoint→processor→action slots→reset parity，再评估 latency 与闭环质量；不自动把三种模型都升入 P1 主读。
- 一句话价值：集成价值在于保留模型的监督、预处理和动作语义，不能用“能加载”代替“能正确执行”。

<a id="a7"></a>
### A7 · Isaac Lab：reset 后的缓存与采样时钟影响观测正确性

- Source ID：`github:isaac-sim/IsaacLab#8780`；ooctipus / Isaac Lab；[Fix sensor freshness, reset cadence, and native coverage](https://github.com/isaac-sim/IsaacLab/pull/8780)；merged **10/23 05:46:27 UTC**，`1d1a808`。
- Impact：High；Decision：**Read**；Learning track：Embodied / Systems；Signal type：system。
- Target question / Reason：reset 或 root/joint write 后首次 observation 是否仍是旧姿态？Evidence：lazy accessor 处理 pending kinematics，sensor 初始化/reset 失效派生缓存；读已清空输出不推进采样时钟，首次正时间 update 才请求测量；native binding 检查 environment coverage。patch 包含 frame / camera / reset 测试。
- 边界：main 修复与 release backport 分开；页面记录特定 Newton compiler failures，不能泛称所有测试全绿。本仓未运行仿真。
- Related topics：[仿真评估](../../05-embodied-infra/topics/embodied_sim_evaluation.md)。Next：Read，按 write→first read→zero-time render→positive-time update 验证姿态和时间戳，避免把陈旧观察误判成 policy 失败。
- 一句话价值：观测新鲜度既依赖物理状态更新，也依赖缓存失效和采样时钟。

<a id="a8"></a>
### A8 · TRL 1.17：采样支持集与混合精度改变 RL loss 的真实含义

- Source ID：`github:huggingface/trl@v1.17.0`；Hugging Face TRL；[release](https://github.com/huggingface/trl/releases/tag/v1.17.0)，**10/29 15:51:15 UTC**；核查支撑 PR [#7866](https://github.com/huggingface/trl/pull/7866)、[#7760](https://github.com/huggingface/trl/pull/7760)，不重复计数。
- Impact：High；Decision：**Deep Dive**；Learning track：RL / Training / Inference；Signal type：algorithm / system。
- Target question / Reason：同一 token 的 rollout logprob 与 trainer logprob 是否在同一分布上？Evidence：top-k/p/min-p 后 sampler 在保留支持集上归一化，trainer 若仍用全词表会引入偏置；GRPO colocate 路径取 sampling mask 后在同一支持集重算。AsyncGRPO 改为 score centering 与 masked importance sampling，额外传输 sampler top-logprobs。
- 边界：GRPO 修复依赖 vLLM >=0.28 与 colocate 等条件，server mode 本 PR 未补齐；AsyncGRPO 默认目标改变，不能当无语义变化升级。额外 logprob 序列化有吞吐代价，release 的 reward / 性能数字为指定 sanity bed 自报；并非保证所有 FP8 训练等价。
- Subsystem：training / rollout / data/trajectory path；维度：正确性、数值稳定性、payload 成本。对 AReaL：借鉴支持集、温度、processor、权重版本与数值路径共同记录，不直接替换算法。
- Related topics：[RL 机制回写](../../04-rl-infra/topics/agentic_rl.md#oct31-contracts)。Next：P1，固定同权重、同样本，比较 full-vocab 与 support-normalized logprob，再评估校正与吞吐。
- 一句话价值：重要性权重首先要求比较的是同一采样分布，不能把所有差异都当 policy staleness。

<a id="a9"></a>
### A9 · AReaL：权重字节更新成功不代表 kernel 布局正确

- Source ID：`github:areal-project/AReaL#1834`；snie / AMD；[restore aiter MoE weight layout after online weight updates](https://github.com/areal-project/AReaL/pull/1834)；merged **10/28 08:08:51 UTC**，`8539527`。
- Impact：High；Decision：**Read**；Learning track：RL / Inference；Signal type：system。
- Target question / Reason：为何第一次 online weight update 后 rollout 失真，但训练进程仍运行？Evidence：ROCm aiter 的 unquantized MoE expert weights 在加载时 tile-shuffle；直接写 checkpoint layout 破坏运行布局。最终页面包括 SGLang expert slice 的 unshuffle/load/reshuffle、AWEX 后处理及 vLLM layerwise reload 路径。
- 边界：dense / CUDA 不属该问题范围；SGLang >=0.5.21 已有 begin/end update 后处理，不应照搬旧版本 patch。该兼容 patch 带文件 hash 守卫；本仓未执行。
- Subsystem：weight sync / inference backend；维度：正确性与版本兼容；对 AReaL：直接适用，需按当前依赖版本检查。
- Related topics：[Agentic RL](../../04-rl-infra/topics/agentic_rl.md#oct31-contracts)。Next：Read，weight sync 后核 layout、graph storage 与同输入输出，不只核 tensor shape/checksum。
- 一句话价值：在线更新的提交边界应包含运行时权重变换，而不止收到新权重字节。

<a id="a10"></a>
### A10 · verl：暂停门闩应等待请求入队，而非首 token

- Source ID：`github:verl-project/verl#8397`；Shawn Lu / xiaoxlu；[release the submission-gate barrier when a request lands in the engine](https://github.com/verl-project/verl/pull/8397)；merged **10/28 04:14:13 UTC**，`081c76b`；提案与 patch 作者时间更早，不计为本轮首次提出。
- Impact：High；Decision：**Read**；Learning track：RL / Inference；Signal type：system。
- Target question / Reason：weight sync pause 为何等待长队列完成 prefill？Evidence：旧 `_admitting` 直到首输出才减，暂停必须等 waiting requests 被调度；改为 `add_request()` 完成有序发送后释放门闩，再消费 collector，异常/取消用内部 request ID abort；有 CPU barrier/cancellation 回归测试。
- 边界：结论依赖所用 vLLM 版本的有序 socket 与 pause 语义；不能泛化所有 backend。PR checks 并非全绿，本仓未运行上游 tests。
- Subsystem：scheduler / weight sync；维度：暂停延迟、取消正确性；对 AReaL：可迁移“已接纳”与“已产出”的状态拆分。
- Related topics：[Agentic RL](../../04-rl-infra/topics/agentic_rl.md#oct31-contracts)。Next：Read，构造 queued→pause→cancel 时序，记录每种状态的计数与 owner。
- 一句话价值：等待首 token 才承认请求已入队，会把模型执行时间错误地放进权重切换屏障。

<a id="a11"></a>
### A11 · NeMo RL：恢复要保存逻辑选择与 attempt 身份

- Source ID：`github:NVIDIA-NeMo/RL#4200`、`#4245`；Ali Roshan Ghias、Akash Mehra / NVIDIA；[SingleController checkpoint/resume](https://github.com/NVIDIA-NeMo/RL/pull/4200)、[monotonic Gym task IDs](https://github.com/NVIDIA-NeMo/RL/pull/4245)；均 merged **10/30 UTC**，`d415d45` / `2bb62a8`。
- Impact：High；Decision：**Read**；Learning track：RL / Systems；Signal type：system。
- Target question / Reason：重启后怎样避免选择集合变化、未确认写入被重复派发，或外部任务 ID 重用？Evidence：#4200 保存 forest selections、receipts、reward/owner 元数据与 trainer RNG，并校验 schema/forest mode/padding；#4245 将单调 task counter 与 data-plane snapshot 同一切面保存，同 attempt sibling retry 保留 ID，新 attempt 获取新 ID。
- 边界：旧 experimental forest checkpoint 不一定可迁移；普通旧 schema 与 forest schema 规则不同。不把页面后续讨论的其他 schema 提案当已合并功能。上游主要 CPU tests、GPU/distributed qualification 未在本轮完成。
- Subsystem：checkpoint/recovery / data/trajectory path；维度：恢复正确性、重试隔离；对 AReaL：借鉴逻辑样本、attempt、外部执行 ID 三层身份，具体 schema 不照搬。
- Related topics：[Agentic RL](../../04-rl-infra/topics/agentic_rl.md#oct31-contracts)。Next：Read，设计 crash-before/after snapshot 与不确定写入确认的故障矩阵。
- 一句话价值：能恢复参数还不够，训练使用哪批轨迹以及每次外部执行的身份也必须恢复。

<a id="a12"></a>
### A12 · FlatCake：KV 的存储布局与排队比“启用 GDS”更关键

- Source ID：`arxiv:2610.38355v1`；[FlatCake: A Flat KVCache Storage Architecture for LLM Serving](https://arxiv.org/abs/2610.38355v1)；Xin Huang 等 10 人；systems paper；v1 **10/28 06:54:51 UTC**。
- Impact：High；Decision：**Read**；Learning track：Inference / Systems；Signal type：system。
- Target question / Reason：SSD 有 KV 命中，为何仍不能及时恢复、反而重算？Evidence：§3–5 将 DRAM 与 NVMe 作为 peer media，利用 KV write-once/immutable、session 粒度复用，按 session 连续与 layer-major 布局放置；coalesced SSD read 经 GDS 进入有界 GPU window，分开 DRAM / SSD queue，减少 staging contention 与读放大。
- 边界：8×B200 / 指定 DRAM 与 SSD 预算上的生产 trace replay，另有 A100 / Llama context sweep；不是作者生产服务上线证明。Mooncake-GDS 对照说明仅换传输原语不等于布局收益；不外推其他 workload、可变状态或 storage reliability。未定位并运行完整实现，不宣称可复现。
- Related topics：[推理状态](../../03-inference-infra/topics/inference_state.md#oct31-storage)。Next：Read，分别测逻辑命中、按 deadline 交付命中、read amplification、queue time 与重算比例。
- 一句话价值：KV 命中只有在请求需要前送到 GPU 才有价值，磁盘布局和恢复队列决定这一步能否完成。

<a id="a13"></a>
### A13 · Mistral Large 4：超大 MoE 的加载、量化与可训练子集共同约束部署

- Source ID：`hf:mistralai/Mistral-Large-4-1T-A52B`；Mistral AI；official model card + HF 团队 integration post；公开发布按 **10/29** [HF 文章](https://huggingface.co/blog/mistral-large4)及[官方模型卡](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B)核验，模型 metadata 时间未另取，不将 lastModified 当首发。
- Impact：High；Decision：**Read**；Learning track：Training / Inference / Systems；Signal type：model / system。
- Target question / Reason：超大 MoE 的 active parameter 数怎样变成真实显存与微调预算？Evidence：官方卡区分约 1T total、49B active、含 embedding/output 的 52B；HF 文章说明 MLA / sparse-attention indexer、expert routing、跨层 KV 与 FP8/NVFP4 交付。TRL 1.17 提供 FP8 base、EP + FSDP load-time sharding、限定模块 LoRA 的单节点配方。
- 边界：**单节点是 8×B300 的限定 LoRA 配方，不是全参数训练**；显存、吞吐与厂商训练规模未复测。HF 博客与模型卡部分 benchmark 数值不同，本轮不转抄排行榜；NVFP4 不默认等质量。许可证是带 Model-as-a-Service 限制的 open-weight license，不能写成无条件 MIT。
- Related topics：[MoE](../../02-training-infra/topics/moe.md)、[FSDP](../../02-training-infra/topics/fsdp.md)。Next：Read，先列 resident weights / trainable states / activation / KV 四本账，再看 backend 的实际精度与 checkpoint 支持。
- 一句话价值：少量 active experts 降低每 token 计算，不会自动消除全部权重的加载和存储成本。

<a id="a14"></a>
### A14 · Megatron Core 0.20：新并行能力必须连同限制一起读

- Source ID：`github:NVIDIA/Megatron-LM@core_v0.20.0`；NVIDIA Megatron；[release](https://github.com/NVIDIA/Megatron-LM/releases/tag/core_v0.20.0)；**10/29 21:06:49 UTC**。
- Impact：High；Decision：**Read**；Learning track：Training / Inference / Systems；Signal type：system。
- Target question / Reason：Megatron-FSDP v2、EP、mixed precision 与 checkpoint 能否任意组合？Evidence：release 明确实验性 adapter 的 TP=1 / expert TP=1，以及 optimizer checkpoint 当前排除 EP layouts；同时列出 quantized gathers、full-iteration graph 的适用配置、DeepEP v2 / NCCL EP MXFP8 的实验状态。
- 边界：这是本轮 release 汇总事件，包含早前 PR，不将所有特性称 10 月 29 日首次出现；未逐审全部 diff。ShortcutMoE 与 wide residual 的 graph/PP 限制必须保留，不能给出统一“全部支持”矩阵。
- Related topics：[FSDP](../../02-training-infra/topics/fsdp.md)、[MoE](../../02-training-infra/topics/moe.md)。Next：Read，按模型、TP/EP/PP、精度、graph、optimizer resume 建兼容性表，再做小 run。
- 一句话价值：新并行栈的价值不止扩展组合，更在于明确哪些组合能保存、恢复和正确执行。

<a id="a15"></a>
### A15 · CUDA GPU Fractioning：显存配额与错误隔离是不同维度

- Source ID：`blog:nvidia/memory-enforced-gpu-sharing-in-kubernetes-with-nvidia-cuda-13-4`；[Memory-Enforced GPU Sharing in Kubernetes with NVIDIA CUDA 13.4](https://developer.nvidia.com/blog/memory-enforced-gpu-sharing-in-kubernetes-with-nvidia-cuda-13-4/)；Ekin Karabulut、David Lifshitz、Hagay Sharon；official engineering blog；**10/27**。
- Impact：High；Decision：**Read**；Learning track：Systems / Training / Inference；Signal type：system。
- Target question / Reason：Kubernetes 的 GPU fraction 怎样落实为进程可见显存边界？Evidence：正文将 scheduler placement、per-container request/limit annotations、node-side GPU Fractioning Operator 与 CUDA enforcement 分开；KAI NvFractions 在调度时避免同 GPU 混用 compute modes。
- 边界：正文区分 time-slicing 与 sm-sharing 的 error domain；显存限制不等于 MIG 的硬件隔离，也不保证尾延迟。博客的 annotation 示例存在不同命名写法，应核当前 operator schema 后再配置。本轮未读完整 operator 实现或实测支持矩阵，不复制命令到生产 runbook。
- Related topics：[Systems](../../01-systems/README.md)、[GPU 路线](../../01-systems/roadmaps/gpu-systems.md)。Next：Read，先核硬件/驱动/模式与 annotation schema，再测超限、邻居故障、突发回收和尾延迟。
- 一句话价值：共享 GPU 需要分别验收显存可用量、调度份额、错误域与服务时延。

## Observed / Rejected Candidates

下表 **18 组 Observed**，Decision 均 Observe、生命周期 NEW；不进入 Accepted / P0。目录候选未取到主源时只作为发现线索，不把题名对应机制当已核事实。非重点商业消息另在 Watch 标 Rejected，不计 Observed 数。

| 组 / 来源 | 判断与 Evidence | Target question / 下一步 |
|---|---|---|
| O1 · [RynnUnify](https://arxiv.org/abs/2610.32965v1)，Minghao Zhu 等，10/23；paper | abs metadata / HTML 已核；理解、交错视频语言规划、agentic 与 action extensions 应分开评估，不能共享一个“大脑”标签即认定控制统一 | Embodied / model；先核各 extension 的训练数据、输出与任务条件，与 A1 去重后再排期 |
| O2 · [Continuous-Action World Models](https://arxiv.org/abs/2610.40401v1)，Liangyu Li、Qingwen Liu，10/29 | abs / HTML 已取得；latent function + query-time action decoding 有动作频率价值，但本轮未完成实验与可执行接口审查 | Embodied / model；核任意 query time 与真实 controller hold/latency 的区别 |
| O3 · [RoboAtom](https://arxiv.org/abs/2610.40207v1) / [UMI-Live](https://arxiv.org/abs/2610.39385v1)，目录线索 | abs/HTML 请求失败，题名来自 cs.RO 目录；不能声明作者、方法或数字已核 | Embodied / algorithm、system；重试一手元数据与数据收集协议 |
| O4 · [ScaleRLT](https://arxiv.org/abs/2610.34427v1)，Dezhen Lu 等，10/25 | abs/HTML 已取；schedule replay value + dense gradient 的 hybrid logprob 需要区分数值一致与梯度目标等价 | RL / system；阅读全文论证、训练对照与代码再决定升级 |
| O5 · [LRMoE](https://arxiv.org/abs/2610.37720v1)，Jiaming Hu 等，10/27 | abs/HTML 已取；通过选择 backward token-expert pairs 减少 demand，改变更新近似，不能只看 MoE layer time | Training / system；核 fixed-loss 与端到端收敛、筛选开销 |
| O6 · [TAO](https://arxiv.org/abs/2610.37230v1)，Shuhuai Li 等，10/27 | abs/HTML 已取；tool readiness + cache credit + token footprint 联合 admission/placement | Inference / system；核 tool readiness 预测误差与公平性，不能只照搬最大收益 |
| O7 · [Rimuru](https://arxiv.org/abs/2610.40305v1)，Zhiyuan Fang 等，10/29 | abs/HTML 已取；共享 page pool、异构 memory views 与 expert offload 值得跟进，巨大收益需核 baseline 是否受容量限制 | Inference / system；核吞吐与 TTFT 对照、搬运成本及实现 |
| O8 · [EnvSched](https://arxiv.org/abs/2610.37609v1)，LG 目录线索 | abs / HTML 均访问失败，不按标题接受 | RL / system；重试元数据、环境调度算法与 benchmark |
| O9 · [AReaL #1825](https://github.com/areal-project/AReaL/pull/1825) | Qwen VLM 训练/推理图像预处理对齐，页面和 patch 已核；与现有 preprocessing 契约关联 | RL / system；按具体 processor 和图像尺寸核 parity，未测 |
| O10 · [LeRobot #5107](https://github.com/huggingface/lerobot/pull/5107)、[#4974](https://github.com/huggingface/lerobot/pull/4974)、[#4948](https://github.com/huggingface/lerobot/pull/4948)、[#4808](https://github.com/huggingface/lerobot/pull/4808) | #5107 核到 float64 running statistics 修正，其他仅 merged 标题索引；Lance episode index、sidecar 内存与 depth batch decoding 需进一步看 diff | Embodied / system；检查 normalization 正确性、dataset IO 与 worker RSS，标题速度数字不转述为事实 |
| O11 · [NVIDIA gds-diag](https://developer.nvidia.com/blog/how-to-check-and-configure-nvidia-gpudirect-storage-for-your-platform/)，Phil Carns，10/30 | 正文核到 config/container/mount 检查与结构化 findings；工具自身不自动实施 mitigation | Systems / system；先核源码与支持矩阵，再接已有 I/O 实验；未运行或安装其 skill |
| O12 · [NVIDIA Dynamo Kubernetes runtime](https://developer.nvidia.com/blog/build-a-validated-kubernetes-runtime-for-nvidia-dynamo/)，Stefan Schimanski 等，10/30 | 正文已取，pinned recipes / runtime dependencies 有部署价值；未审全配方实现与支持矩阵 | Inference / system；核 request / KV-event / data plane 的依赖与 readiness |
| O13 · [Granite Switch 4.1.1](https://huggingface.co/blog/ibm-granite/granite-switch-4-1-1)，IBM vendor post，10/29 | 正文是模块化 adapter patch release，声称不重训 base；兼容性未独立验证，不将修补包装成架构首发 | Inference / model；核 adapter ABI、缓存与回滚契约 |
| O14 · [LFM3](https://huggingface.co/blog/LiquidAI/lfm3)，LiquidAI vendor post，10/27；[Nemotron Halloween](https://huggingface.co/blog/nvidia/nemotron-halloween)，NVIDIA vendor post，10/30 | HF 正文已取；模型发布与展示不直接替代系统机制或生产证据 | Training / Inference / model；按模型卡/方法/部署证据继续 triage，暂不抄榜单 |
| O15 · [OpenAI misalignment detection](https://openai.com/index/how-we-detect-frontier-model-misalignment/)，10/29 | RSS 可验证目录项，正文 403，未核方法；不能依据标题做技术归纳 | RL / system 潜在相关；补正文再判定是否改变评估与运行契约 |
| O16 · [Anthropic Research](https://www.anthropic.com/research) 的 10/23 alignment assessment 与 10/27 expert judgment / drug development | 目录已核，正文未逐项深读；与本轮具身/Infra 缺口的直接联系不足 | RL 潜在评估价值；有具体环境隔离或评估问题时再深读 |
| O17 · [TensorRT-LLM 1.4.0rc2](https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.4.0rc2) / [Transformers 5.22](https://github.com/huggingface/transformers/releases/tag/v5.22.0) | 已读 release；前者 KV lifecycle / breaking changes、后者模型集成需按部署版本核验；不把 release 全部 PR 算成新增 accepted | Inference / system；先核 TRT TP+RPC 初始化已知 hang、wheel `1.4.0rc2.post1` 与 container `1.4.0rc2` 区别 |
| O18 · [TorchCodec commits](https://github.com/meta-pytorch/torchcodec/commits/main)、[Kernels 0.18](https://github.com/huggingface/kernels/releases/tag/v0.18.0)、[PyTorch 2.15](https://github.com/pytorch/pytorch/releases/tag/v2.15.0)、[DeepSpeed 0.19.9](https://github.com/deepspeedai/DeepSpeed/releases/tag/v0.19.9) | release/Atom 索引已核；TorchCodec missing PTS→NaN 与 image metadata 是需回看线索；未审各变更 diff | Embodied / Systems / Training；先定时间戳、分片或 kernel 问题，再做版本差异审计 |

## OpenAI / Anthropic / NVIDIA / DeepSeek Watch

| Vendor | Sources checked / Evidence | Watch decision / 结果与限制 |
|---|---|---|
| OpenAI | [Research](https://openai.com/news/research/) / [Engineering](https://openai.com/news/engineering/) 均 403；[官方 RSS](https://openai.com/news/rss.xml) 成功；O15 正文仍 403 | **Observed / not verifiable in this scan**：O15 待补。合作、客户案例与一般政策发布按本库 focus **Rejected**，不靠 RSS 描述接受技术事实 |
| Anthropic | [Research](https://www.anthropic.com/research) / [Engineering](https://www.anthropic.com/engineering) 成功，工程目录最新仍 10/13；研究目录见 O16 | **Observed**：有新研究目录项，未完成其正文核查；工程目录未见本窗口新增，不等于全部 docs 无更新 |
| NVIDIA | [Technical Blog](https://developer.nvidia.com/blog/) 目录、A15 / O11 / O12 正文，Megatron / TRT / NeMo / GR00T / Isaac releases / PR | **Accepted** A7/A11/A14/A15；O11/O12/O14/O17 **Observed**。发布机制与自报性能分开，未做 GPU 实测 |
| DeepSeek | [API changelog](https://api-docs.deepseek.com/updates/) 与 [官方 HF 最新 30 模型 metadata](https://huggingface.co/api/models?author=deepseek-ai&sort=lastModified&direction=-1&limit=30) 均成功 | **Not found in this scan**：所查 changelog 最新 10/17，HF 最新修改也是 10/17；未见窗口内新接受项。非全量 org 历史与所有 model card 审计 |

## Hugging Face Watch

| 来源 | Evidence / Watch decision |
|---|---|
| [HF Blog](https://huggingface.co/blog) + [feed](https://huggingface.co/blog/feed.xml) | **Accepted** A13：HF 团队集成文与 Mistral 官方模型卡交叉；IBM / LiquidAI / NVIDIA vendor-authored posts 分别 O13/O14 **Observed**。不是 HF 官方团队原创，社区文章也非全量扫描 |
| [TRL releases](https://github.com/huggingface/trl/releases) | **Accepted** A8；1.17.1 patch 也已列入 release 索引，不另计；Mistral LoRA 作为 A13 证据 |
| [Transformers](https://github.com/huggingface/transformers/releases) | **Observed** O17：5.21、5.21.1、5.22.0、5.22.1 在窗口；非全部 PR/diff 审计 |
| [Accelerate](https://github.com/huggingface/accelerate/releases) / [PEFT](https://github.com/huggingface/peft/releases) | **Not found in checked release list**：各首 30 releases 未见窗口新增；merged search 限流，不能声称没有 main 变化 |
| [Kernels](https://github.com/huggingface/kernels/releases) | **Observed** O18，0.18.0 release；未完成实现深审 |
| [LeRobot](https://github.com/huggingface/lerobot/pulls) | **Accepted** A5/A6；42 merged 索引，重点 patch 核查；O10 **Observed**。不把当前 main 能力误写成新 release |

## RL Framework Watch

| Framework / 实际来源 | 子系统与工程维度 | Evidence / 对 AReaL 的参考 | Watch decision |
|---|---|---|---|
| [AReaL](https://github.com/areal-project/AReaL/pulls) | weight sync / inference backend：布局正确性 | 28 merged，release 26 条；A9 页面/patch，O9 图像 preprocessing；直接适用版本需核 | **Accepted** A9；**Observed** O9 |
| [verl](https://github.com/verl-project/verl/pulls) | scheduler / weight sync：pause 边界与取消 | 24 merged，release 16 条；A10 核代码与测试；可迁移 admission 状态拆分 | **Accepted** A10 |
| [slime](https://github.com/THUDM/slime/pulls) | router registration / CI | 2 merged（#2483 保留注册、#2497 CI），release 11 条；未见足够材料证明重大系统改变；不把常规修复升级 | **Rejected** 本轮小改；无新增 Accepted |
| [ROLL](https://github.com/alibaba/ROLL/pulls) | 无已核窗口重大变化 | search 限流；closed PR 首100未见窗口合并，Atom 最新10/19，release 5 条；仅此范围 | **Not found in checked sources** |
| [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF/pulls) | 无已核窗口重大变化 | search 限流；closed PR 首100未见窗口合并，Atom 最新9/17，release 首30；不推断所有分支无更新 | **Not found in checked sources** |
| [NeMo RL](https://github.com/NVIDIA-NeMo/RL/pulls) | checkpoint/recovery / data/trajectory path | closed PR 首100含70条窗口 merged、release10条；#4200/#4245 patch 核查；可迁移逻辑样本与 attempt 身份 | **Accepted** A11；其余仅标题筛选，分页缺口保留 |
| [TRL](https://github.com/huggingface/trl/releases/tag/v1.17.0) | training / rollout：分布、数值与 payload | A8 release + 两项 PR；支持集与数值误差分开 | **Accepted** A8 |
| [Megatron](https://github.com/NVIDIA/Megatron-LM/releases/tag/core_v0.20.0) | training / inference backend / checkpoint | A14 release 提供兼容性边界，未完成所有 PR 审计 | **Accepted** A14 |

新发现框架未达到本轮可核实现门槛，不凭宣传仓库扩展 core watchlist。

## Inference Systems Watch

| 来源 | Evidence / Target question | Watch decision / 限制 |
|---|---|---|
| [vLLM releases](https://github.com/vllm-project/vllm/releases) / [main Atom](https://github.com/vllm-project/vllm/commits/main.atom) | 首30 releases、最新20 commits；weight reload、NIXL layer pairing 等标题线索；另有 A8–A10 的已核集成后果 | **Observed** 快照；search/closed API 限流，**10/23→cutoff 全窗口 PR 不完整**，不能以最新20覆盖8天 |
| [SGLang releases](https://github.com/sgl-project/sglang/releases) / [main Atom](https://github.com/sgl-project/sglang/commits/main.atom) | 首30 releases、最新20 commits；host pools、diffusion Ulysses 等线索；A9 指定版本与 A12 论文实现分开 | **Observed** 快照；全窗口 PR 缺口保留，未审所有 kernel |
| [TensorRT-LLM rc2](https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.4.0rc2) / [main](https://github.com/NVIDIA/TensorRT-LLM/commits/main.atom) | KV transfer lifecycle、zero-copy、跨 turn reuse 与部署已知问题；release 与后续 main 不混算 | **Observed** O17；未完整审 major PR，部署前需版本核查 |
| [FlatCake](https://arxiv.org/abs/2610.38355v1) | immutable KV 的介质布局、及时恢复与读放大，独立 serving 价值 | **Accepted** A12；trace replay，非生产复现 |
| [Mistral 模型卡](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B) | precision、KV 与可部署权重形态 | **Accepted** A13；backend 配方有硬件/版本条件 |
| NVIDIA Kubernetes runtime / GPU fractioning | O12 / A15；依赖版本、GPU配额、错误域与 readiness | A15 **Accepted**；O12 **Observed**；不能以配置成功代替 SLO 测量 |

## Embodied Models & Infra Watch

两栏各自判断模型/学习与系统后果；公开方法不要求先证明 Infra 提速，代码存在也不等于实验跑通。

| 来源 / Watch decision | Models / Learning：实际证据与缺口 | Infra：实际证据与缺口 |
|---|---|---|
| [Physical Intelligence blog](https://www.pi.website/blog) / [openpi](https://github.com/Physical-Intelligence/openpi)；**Not found in checked sources** | 博客最新10/13，窗口未见新机制；不重复接受旧 Pi API / π0.7 | merged search 0、release空；未审 direct commits，旧 PyTorch precision 风险见补录，不称本周新修复 |
| [OpenVLA](https://github.com/openvla/openvla) / [OFT](https://github.com/moojink/openvla-oft)；**Not found in checked sources** | 未见窗口新模型接受项，未重读历史方法 | 两仓 merged=0、release空；direct commits 非全量 |
| [RynnVLA](https://arxiv.org/abs/2610.34356v1) / [RynnWorld](https://arxiv.org/abs/2610.36791v1)；**Accepted A1** | action latent、flow objective、target adaptation 与合成训练价值已核；泛化与物理因果有边界 | 官方 README 已核，数据统计/工件阶段有差异；未运行 labeling、训练、retargeting |
| [WAM 消融](https://arxiv.org/abs/2610.34593v1) / [WorldFailBench](https://arxiv.org/abs/2610.33565v1) / [Action interface](https://arxiv.org/abs/2610.38104v1)；**Accepted A2–A4** | 分开辅助目标、泛化、失败原因/阶段/边界与执行 action interface | 提供评估设计，不提供本仓吞吐收益；动作滤波、控制周期应与 runtime 一起记录 |
| [GR00T](https://github.com/NVIDIA/Isaac-GR00T)；**Observed** | #798 标题 Update N1.7，未深读，不当新模型首发；LeRobot LingBot/Long-WAM 为 A6 | 1 merged、5 releases 列表；未完成 #798 diff 审查 |
| [Isaac Lab](https://github.com/isaac-sim/IsaacLab)；**Accepted A7** | 未因此声明 policy 学习进步；观测错误会污染训练/评估 | 56 merged、26 releases；重点 #8780 缓存/reset/freshness，其他 major PR 仍待核 |
| [LeRobot](https://github.com/huggingface/lerobot)；**Accepted A5/A6** | Long-WAM / Faster-WAM / LingBot 的目标、输入、动作与推理边界已核；未找到本窗口 SmolVLA 新模型首发 | 分片 PEFT / EMA / mixed precision 有代码证据；Lance/sidecar/decode 为 O10，尚未实测 |
| [TorchCodec](https://github.com/meta-pytorch/torchcodec) / [PyTorch](https://github.com/pytorch/pytorch/releases/tag/v2.15.0)；**Observed O18** | 无本轮已核新模型机制 | release / Atom 有界核查，missing PTS 与 metadata 待审 diff；不能把缺时间戳当0 |
| arXiv cs.RO / cs.LG / cs.CV | RO/LG 成功，CV两种 page size 均失败；O1–O3 为模型候选；**未检查与查后无新增分开** | DC/AR 成功；标题筛选后只深读重点候选，不承诺穷尽 dataset/IO 社区 |

## 晚发现与下一步

[FP32 master copy 的 VLA 精度审计](backfill/2026-10.md#oct31-vla-precision)于 **10/22 19:45:06 UTC** 已发表，早于上次游标，本次才发现；独立 backfill，**不计本轮 15 组**。已核 citation metadata 与正文，不把“Hub checkpoint 存 bf16”当训练全过程无 FP32 master 的证明。

先读 [P1 本轮三项学习动作](../reading_queue/P1.md#oct31-frontier)：latent→native 接口、WAM 评估、分片训练/精度恢复。旧 P0 保持状态；RL/Inference 材料按现有任务选读，不把全部 Accepted 都变成在读。

下一轮从本页 cutoff 去重扫描，但以下来源必须保留更早补扫起点：

1. **本轮缺口**：cs.CV/cs.CL、未显示的公告日期、GitHub 限流仓的完整 merged PR / direct commits 从 **10/23 11:09:27** 回扫；不以 Atom 快照替代历史范围。
2. **继承缺口**：arXiv 发现覆盖自 **9/22 16:52:05**；GitHub direct commits / 历史外围 PR / 未核 release history 自 **9/22 17:06:16**。本轮并未关闭这些历史缺口。
3. OpenAI 正文、RoboAtom/UMI-Live/EnvSched 主源失败，补核后再决定接受。RynnWorld 文档统计差异与 release/main 配方边界须继续核对。

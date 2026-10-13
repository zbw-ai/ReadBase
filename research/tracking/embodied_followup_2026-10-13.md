# 具身模型与 Infra：后续模型、GitHub 与学习入口补齐

日期：2026-10-13。类型：定向 Historical Backfill / GitHub audit；来源按原始月份归档，不是新一轮 frontier scan，不推进 scan_log，也不改旧 Accepted 数量。接续[第一批 13 项](embodied_retrospective_2026-10-13.md)。

**本轮主入口：[具身 Infra 核心 topic](../../05-embodied-infra/topics/embodied_infra.md)**。从整体循环进入模型基础、数据 I/O、训练、评估和执行；[阶段路线](../../05-embodied-infra/roadmap.md)负责学习顺序，[共享队列](../reading_queue/P1.md#embodied-20261013)负责待办，[学习记录](../learning_log/2026-10.md#embodied-discussion)保存讨论。

## 进展与趋势判断

模型侧已经从单段动作生成延伸到经验学习、短长记忆与世界预测，Infra 侧的重要变化集中于时间和状态：训练样本身份、视频像素、flow 方向、共享 view 以及已提交动作都需要可核验契约。**趋势推断**：接下来值得投入的不是更长的模型名单，而是让数据、模型状态与实际执行说同一种语言，再用闭环协议判断优化是否有效。

本轮接受 **16 项：8 项模型／评估来源（7 篇论文、1 篇厂商文章）和 8 项 GitHub 事件**；另记录 FBFM 等观察项。它们是晚发现／重新核验的历史材料，按原始月份归档。模型／评估来源与 GitHub 事件分开列出；同一材料在多处出现不重复计数。条目全部保持 NEW，既不把整理当用户读完，也不把上游测试当本地复现。

## Models / Learning：接受的来源

| 原始日期 | 来源与账本 | 一句话看点 |
|---|---|---|
| 2025-11-18 | [RECAP / π*0.6](https://arxiv.org/abs/2511.14759v1) · [Deep Dive](backfill/2025-11.md#embodied-followup-2511-14759) | 利用 value 估计的 advantage 条件，把示范、自主经验与人工纠错接到策略学习。 |
| 2026-02-17 | [DreamZero](https://arxiv.org/abs/2602.15922v1) · [Deep Dive](backfill/2026-02.md#embodied-followup-2602-15922) | 联合预测未来视频与动作，并用执行后的真实观察更新缓存；视频先验必须服从闭环状态与延迟约束。 |
| 2026-03-04 | [MEM](https://arxiv.org/abs/2603.03596v1) · [Read](backfill/2026-03.md#embodied-followup-2603-03596) | 用近期视觉与长期文字摘要分担不同时间尺度的记忆，让策略状态不再仅是一张当前图像。 |
| 2026-04-16 | [π0.7](https://arxiv.org/abs/2604.15483v1) · [Read](backfill/2026-04.md#embodied-followup-2604-15483) | 把质量、速度、控制模式和视觉子目标等条件纳入训练，使推理中的策略调整有学习依据。 |
| 2026-04-17 | [GR00T N1.7 EA 文章](https://huggingface.co/blog/nvidia/gr00t-n1-7) · [Read](backfill/2026-04.md#embodied-followup-groot-n17) | 模型与公开适配入口升级需要绑定具体代码和 checkpoint，不能只按模型名称迁移。 |
| 2026-05-28 | [MiraBench](https://arxiv.org/abs/2605.29360v1) · [Read](backfill/2026-05.md#embodied-followup-2605-29360) | 用物理一致性、动作遵循与失败保留评估世界模型，避免把逼真成功视频当可信模拟。 |
| 2026-06-03 | [OSCAR](https://arxiv.org/abs/2606.04463v1) · [Read](backfill/2026-06.md#embodied-followup-2606-04463) | 以骨架渲染表达跨本体动作条件，再生成未来视频；其评估使用记录动作重放，不能等同在线策略闭环。 |
| 2026-09-17 | [VLA-Feedback](https://arxiv.org/abs/2609.21022v1) · [Read](backfill/2026-09.md#embodied-followup-2609-21022) | 以新观测完成最后一步动作去噪，局部修正已有 chunk；传感器到执行的延迟仍可能支配响应。 |

## Infra：接受的 GitHub 事件

| 原始日期 | 原始事件与账本 | 一句话看点 |
|---|---|---|
| 2025-04-28 | [moojink/openvla-oft 94d345fe8ede223821f8cd8ddd6b3f5997ef8b65](https://github.com/moojink/openvla-oft/commit/94d345fe8ede223821f8cd8ddd6b3f5997ef8b65) · [Read](backfill/2025-04.md#embodied-github-2) | 分离训练噪声时间表与推理步数，修正可选 diffusion head 的减步推理。 |
| 2025-12-18 | [Physical-Intelligence/openpi 821](https://github.com/Physical-Intelligence/openpi/pull/821) · [Read](backfill/2025-12.md#embodied-github-1) | 加权混采、过滤和 chunk 共同决定训练分布；配置比例仍须用实际样本核验。 |
| 2026-07-29 | [NVIDIA/Isaac-GR00T 727](https://github.com/NVIDIA/Isaac-GR00T/pull/727) · [Read](backfill/2026-07.md#embodied-github-5) | 减少图像格式往返的同时提供输入 parity 测试，但 gated skip 和公开 lint 失败必须保留。 |
| 2026-09-16 | [isaac-sim/IsaacLab v3.0.0-EA](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0-EA) · [Deep Dive](backfill/2026-09.md#embodied-github-6) | 零拷贝 Warp/Torch view 带来共享可变状态，迁移还需检查 quaternion 顺序。 |
| 2026-09-30 | [meta-pytorch/torchcodec v0.17.0](https://github.com/meta-pytorch/torchcodec/releases/tag/v0.17.0) · [Deep Dive](backfill/2026-09.md#embodied-github-7) | 分阶段解码让并行更灵活，同时把 CUDA stream 依赖和内存寿命交给调用方。 |
| 2026-10-06 | [huggingface/lerobot 4075](https://github.com/huggingface/lerobot/pull/4075) · [Read](backfill/2026-10.md#embodied-github-4) | 统一 solver 前先明确 flow 方向、速度符号和已提交前缀，避免静默改变动作。 |
| 2026-10-06 | [meta-pytorch/torchcodec 1758](https://github.com/meta-pytorch/torchcodec/pull/1758) · [Read](backfill/2026-10.md#embodied-github-8) | 视频时间戳正确仍可能拿到被覆盖的像素；该变更只是 surface 复用问题的缓解。 |
| 2026-10-07 | [huggingface/lerobot 3917](https://github.com/huggingface/lerobot/pull/3917) · [Deep Dive](backfill/2026-10.md#embodied-github-3) | 远端 episode 池将内存预算、rank 覆盖、样本顺序和恢复变成可测试契约。 |

## GitHub 实际覆盖

六库均检查官方 release/tag 首批页，并检查按 updated 倒序的 closed PR 页；PR 只代表有界候选集合，不是完整历史。另查 OFT 最近 100 个 commit 及重点变更的文件、测试与公开检查。

| 仓库 | release / tag 本页数量 | closed PR 本页数量（merged） | PR updated 范围 |
|---|---|---|---|
| Physical-Intelligence/openpi | 0 / 0 | 100（35） | 2025-07-09T05:04:40Z → 2026-10-08T18:11:00Z |
| moojink/openvla-oft | 0 / 0 | 2（0） | 2025-09-08T15:49:08Z → 2026-07-21T12:18:42Z |
| huggingface/lerobot | 11 / 11 | 100（40） | 2026-09-28T09:06:02Z → 2026-10-12T14:46:49Z |
| isaac-sim/IsaacLab | 24 / 30 | 100（76） | 2026-10-10T19:44:49Z → 2026-10-13T08:25:30Z |
| NVIDIA/Isaac-GR00T | 5 / 5 | 100（48） | 2025-09-26T17:38:11Z → 2026-10-07T22:35:39Z |
| meta-pytorch/torchcodec | 24 / 30 | 100（88） | 2026-08-18T18:11:13Z → 2026-10-08T11:35:51Z |

30 个 tag 是分页上限，不是仓库 tag 总数。updated 范围不等于事件首发区间或该区间已穷尽。OFT 无 release/tag、可见最新实现 commit 为 2025-09-09，未据此发明 2026 新版本。[机器可读审计](audits/2026-10-13-embodied-followup/github_evidence.json)保存实际来源、固定 SHA、测试边界及排除项。

## 观察、撤回与证据边界

- **Observed**：[FBFM](https://arxiv.org/abs/2607.29235v1)已核 metadata 和方法，但 pseudo-asynchronous clock 不能证明 wall-clock 实时性，求导开销与控制验证仍待补；见[7 月条目](backfill/2026-07.md#embodied-followup-2607-29235)。

- **Observed**：openpi #634 的 π0.5/PyTorch 支持未逐文件验证数值 parity；OFT 多 GPU barrier 修复、LeRobot decoder LRU cache、GR00T letterbox 作为后续线索，未另列 Accepted。原始链接见审计索引。
- **Rejected / Ignore**：[LeRobot #4836](https://github.com/huggingface/lerobot/pull/4836) 的 remote inference 很快被 [#4936](https://github.com/huggingface/lerobot/pull/4936) 撤回，不当作当前已交付功能。
- GR00T N1.6 GitHub release 记录发布日期为 2026-04-15，代码合并已在 2025-12；不能用 release 日期当模型首发。N1.7 EA 文章与当前模型卡的冲突字段不当事实引用。
- LeRobot #3917 在固定历史 commit 上限制 worker；后续 [#4874](https://github.com/huggingface/lerobot/pull/4874) 已改变这一点。报告没有把历史限制冒充当前 main 配置。
- GR00T #727 有 gated parity 测试，公开检查只见 lint 失败；TorchCodec #1758 的检查有成败混合。测试存在、作者报告通过、公开 CI 通过、本地复现四者分开。
- TorchCodec #1758 晚于 v0.17，且只是缓解，不宣称发布包已包含它或 NVDEC 问题彻底解决。Isaac Lab 3.0 是 EA；不把兼容性设计当生产验证。

## 后半年官方来源与剩余范围

| 来源 | 实际核查 | 结果与范围 |
|---|---|---|
| PI Blog | 直接获取返回 429，辅以域名限定的 6–10 月搜索 | Not found / not verifiable；检索未见不等于没有发布，接受依据仍是固定论文 |
| NVIDIA GR00T 标签页 | 页面可读，7/7 端到端工作流文章已查看 | Observed，已有 7 月账本；标签页不是 NVIDIA 全来源 |
| NVIDIA GEAR publications | 搜索索引可发现条目，直接页面为动态壳 | 部分可核验，未遍历全部论文 |
| Hugging Face Blog | 实际首页及官方 LeRobot v0.6.0 文章 | 首页检查无额外 Accepted；未遍历所有分页，框架集成不替代模型论文核验 |

模型方法核验推进到 9 月；GitHub 有实际 10 月事件。但 8 月及 10 月前半的模型来源仍不构成系统穷尽覆盖。触觉、灵巧手、移动操作与真实部署恢复仍是独立缺口；没有按月份凑材料。后半年查询、原始日期与方法证据见[核查记录](audits/2026-10-13-embodied-followup/late_model_sources.json)。整体“补齐”指补上前次明确的来源和学习入口缺口，不代表全领域全历史收录完成。

## 接下来怎么读

先从核心 topic 的一个动作例子开始，再读模型与数据正文。之后选两条深化线：经验与记忆（RECAP→MEM→π0.7），世界预测与评估（MiraBench→DreamZero）。工程先选 LeRobot streaming、flow parity 与 TorchCodec 的样本正确性；其余按真实瓶颈展开，不同时升级为 P0。

全文来源证据：[论文 metadata](audits/2026-10-13-embodied-followup/model_sources.json)、[方法页面哈希](audits/2026-10-13-embodied-followup/method_pages.json)、[GitHub 证据](audits/2026-10-13-embodied-followup/github_evidence.json)。历史分月记录保留原有阅读状态；新信号只代表本次核验范围。

[返回雷达](README.md) · [历史补录](backfill/README.md) · [具身核心入口](../../05-embodied-infra/topics/embodied_infra.md)

# 具身模型与 Infra：历史补扫和下一阶段重点

后续进展：[模型、GitHub 与核心学习入口补齐](embodied_followup_2026-10-13.md)已开展；以下保留第一批时的覆盖状态，未宣称两批穷尽历史。

日期：2026-10-13。类型：Historical Backfill / priority review；不是 frontier scan，不推进扫描游标，也不改历史 Accepted 数量。

本轮核验 12 篇原始论文和 1 篇官方工程文章，按原始月份补录：**6 项新增、7 项已有材料复评**。保留历史条目的状态，本次新增学习问题仍为 NEW；核验和整理不等于用户已读、全文精读或实验完成。

## 整体进展与趋势判断

下一阶段同时重视模型怎么生成动作和系统怎样保证这些动作及时、正确地执行。历史脉络从 action chunk / diffusion，发展到视觉语言预训练与连续动作专家，再走向异构数据共同训练和异步执行。**趋势推断**：工程重点会从单次模型调用性能转向数据语义、观测年龄、动作提交和闭环评估的联合约束；不能仅凭更快生成或展示视频判断泛化。

## 时间线与一句话看点

| 原始日期 | 材料／本轮处理 | 主要看点 |
|---|---|---|
| 2023-03-07 | [Diffusion Policy](backfill/2023-03.md#embodied-20261013-2303-04137) · 复评；[原文](https://arxiv.org/abs/2303.04137v1) | 以条件去噪生成连续动作序列，结合 receding horizon 执行；动作多模态和推理迭代成本需要一起理解。 |
| 2023-04-23 | [ACT](backfill/2023-04.md#embodied-20261013-2304-13705) · 新增；[原文](https://arxiv.org/abs/2304.13705v1) | 以条件变分模型学习 action chunk，temporal ensembling 汇合同一时刻的多次预测；chunk 长度同时影响误差累积与反馈速度。 |
| 2023-10-13 | [Open X-Embodiment / RT-X](backfill/2023-10.md#embodied-20261013-2310-08864) · 复评；[原文](https://arxiv.org/abs/2310.08864v1) | 跨机器人数据与共同训练提供迁移证据，但统一文件格式不代表动作坐标、单位和控制语义相同。 |
| 2024-03-19 | [DROID](backfill/2024-03.md#embodied-20261013-2403-12945) · 新增；[原文](https://arxiv.org/abs/2403.12945v1) | 分布式真实环境采集扩大场景多样性；工程上应把采集质量、环境身份和数据划分纳入数据治理。 |
| 2024-06-13 | [OpenVLA](backfill/2024-06.md#embodied-20261013-2406-09246) · 复评；[原文](https://arxiv.org/abs/2406.09246v1) | 将视觉语言模型接到离散动作表示与机器人微调，是理解 VLM 怎样成为控制策略的基线。 |
| 2024-10-31 | [π0](backfill/2024-10.md#embodied-20261013-2410-24164) · 复评；[原文](https://arxiv.org/abs/2410.24164v1) | 将视觉语言条件与 flow matching 连续动作生成结合；动作专家与迭代采样改变训练和部署的计算路径。 |
| 2025-02-27 | [OpenVLA-OFT](backfill/2025-02.md#embodied-20261013-2502-19645) · 新增；[原文](https://arxiv.org/abs/2502.19645v1) | 结合并行解码、action chunk 与连续 L1 回归优化 VLA 微调；生成吞吐提升不能直接替代闭环任务成功率。 |
| 2025-03-18 | [GR00T N1](backfill/2025-03.md#embodied-20261013-2503-14734) · 新增；[原文](https://arxiv.org/abs/2503.14734v1) | 联合训练的视觉语言 System 2 与动作 diffusion transformer System 1 连接语义理解和控制，并混合真实、视频与合成数据。 |
| 2025-04-22 | [π0.5](backfill/2025-04.md#embodied-20261013-2504-16054) · 新增；[原文](https://arxiv.org/abs/2504.16054v1) | 异构数据共同训练把高层子任务语义与低层动作连接起来；泛化结论应结合训练配方和消融边界阅读。 |
| 2025-06-02 | [SmolVLA](backfill/2025-06.md#embodied-20261013-2506-01844) · 新增；[原文](https://arxiv.org/abs/2506.01844v1) | 小型 VLA 与社区数据训练结合，并将动作预测和执行异步化；适合作为可负担实验路径的候选。 |
| 2025-06-09 | [RTC](backfill/2025-06.md#embodied-20261013-2506-07339) · 复评；[原文](https://arxiv.org/abs/2506.07339v1) | 生成下一段动作时保持已提交前缀，以 inpainting 衔接执行中的动作，直接处理推理延迟造成的控制不连续。 |
| 2025-09-16 | [LeRobotDataset v3](backfill/2025-09.md#embodied-20261013-lerobot-v3) · 复评；[原文](https://huggingface.co/blog/lerobot-datasets-v3) | 将多个 episode 打包到共享数据与视频文件，通过元数据定位边界，缓解小文件与大规模数据访问问题。 |
| 2025-11-06 | [Isaac Lab](backfill/2025-11.md#embodied-20261013-2511-04831) · 复评；[原文](https://arxiv.org/abs/2511.04831v1) | 将 GPU 物理仿真、渲染、传感器与学习工作流连接起来；吞吐评价必须同时记录环境配置与仿真保真度。 |

## 阅读顺序：只启动一组

1. **动作模型**：ACT / Diffusion Policy → OpenVLA 与 π0 对照 → OFT。产出输入、动作表示、训练目标、解码、执行前缀的对照表。GR00T N1、π0.5、SmolVLA 用作架构与数据配方扩展。
2. **数据契约**：OXE / DROID → LeRobotDataset v3。产出 episode schema 与单位、坐标、时间、归一化、拆分策略检查表。
3. **执行与评估**：RTC → Isaac Lab。产出带观测年龄和提交边界的时间线，以及固定仿真条件的评估设计。

待办统一进入 [P1 三阶段](../reading_queue/P1.md#embodied-20261013)，原 P0 在读项不自动退出。[正文](../../05-embodied-infra/topics/agentic_for_embodied.md#embodied-model-contracts)给出机制和生产风险的连接；[六阶段路线](../../05-embodied-infra/roadmap.md)继续负责掌握顺序。

## 实际覆盖与缺口

- 论文：上表 12 篇固定 v1 的标题、作者、日期和摘要机制已核；ACT、OFT、π0.5 另查方法。其余不声称全文精读。完整署名及页面哈希见[证据索引](audits/2026-10-13-embodied/sources.json)。
- 工程来源：LeRobot v3 官方博客已核。补充阅读 [openpi README](https://github.com/Physical-Intelligence/openpi)、[OpenVLA-OFT README](https://github.com/moojink/openvla-oft)、[LeRobotDataset 当前文档](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3)，仅代表 2026-10-13 的入口核查，不是历史版本复原或源码审计。norm stats、动作映射、多相机／proprio 配置值得追踪，未运行代码。
- 本轮覆盖 2023–2025 的一组基础材料，**不声称全部具身历史已补齐**。2026 模型演进、World Model 的 action-conditioned dynamics 与规划评估、触觉／灵巧手、移动操作、多机器人部署仍需独立补扫。
- GitHub 尚缺 openpi / OpenVLA-OFT / LeRobot / Isaac Lab / GR00T / TorchCodec 的固定版本、重大 PR、测试与兼容性变更时间线；此次未把入口访问写成完成全面 GitHub 审计。下一批按实际 release 日期补录，无法确认的保持待核验。
- 闭环 benchmark、失败恢复、数据授权／版本治理和真实机器人运行证据仍不完整；厂商或作者报告的性能不能当成本地复现。

## 后续扫描规则

具身是下一阶段优先项。每轮 Embodied Watch 分别写 Models / Learning 与 Infra 两栏，分别报告实际检查的来源、判断、证据和缺口；零 Accepted 有效。继续保留 Training、RL、Inference 和四厂商等既有 Watch。来源检查失败与检查后无重要新增必须分开写。月报只汇总已有记录，本次历史补扫按补录月份进入后续总结，不追改旧月报的历史覆盖声明。

[返回研究雷达](README.md) · [历史月份索引](backfill/README.md) · [具身 Part](../../05-embodied-infra/README.md)

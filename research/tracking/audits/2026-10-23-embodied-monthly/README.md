# 2026-10-23：2026 年 1—9 月具身月报补充账本

[阅读入口](../../monthly_reviews.md#embodied-2026) · [逐来源机器账本](aggregation.json)

本次响应指定的历史月报补齐请求：先归并已核验的历史来源，对明确缺口另做定向核验并写入原始月份 backfill，再汇入月报。不是一次 frontier scan，也不是重新枚举九个月全部论文与 GitHub 事件。

**共 24 个唯一来源：9 项新增历史 Read、11 项已有 Accepted 归并、4 项已有 Observe 保留。** 同一个来源同时出现在 Models / Learning 和 Infra 只计一次；统计只描述本次补充，不与原月报 Accepted 相加。同日其他任务已登记的 VLT 也算已有材料，不能再算新增。

| 月份 | 新增历史接受 | 已有接受归并 | 已有 Observe | 合计 |
|---|---:|---:|---:|---:|
| [2026-01](../../monthly_signal_2026-01.md#embodied-20261023) | 2 | 0 | 0 | 2 |
| [2026-02](../../monthly_signal_2026-02.md#embodied-20261023) | 1 | 1 | 0 | 2 |
| [2026-03](../../monthly_signal_2026-03.md#embodied-20261023) | 1 | 1 | 0 | 2 |
| [2026-04](../../monthly_signal_2026-04.md#embodied-20261023) | 1 | 2 | 1 | 4 |
| [2026-05](../../monthly_signal_2026-05.md#embodied-20261023) | 1 | 1 | 1 | 3 |
| [2026-06](../../monthly_signal_2026-06.md#embodied-20261023) | 1 | 1 | 0 | 2 |
| [2026-07](../../monthly_signal_2026-07.md#embodied-20261023) | 0 | 2 | 2 | 4 |
| [2026-08](../../monthly_signal_2026-08.md#embodied-20261023) | 2 | 0 | 0 | 2 |
| [2026-09](../../monthly_signal_2026-09.md#embodied-20261023) | 0 | 3 | 0 | 3 |

## 本次新增的九项：一句话判断

| 原始月份 | 来源与 Decision | 解决当前什么问题 |
|---|---|---|
| 2026-01 | [TorchCodec 0.10](https://github.com/meta-pytorch/torchcodec/releases/tag/v0.10.0) · Read · [账本](../../backfill/2026-01.md#embodied-20261023-torchcodec-010) | Decoder transforms 与 GPU encoding 让视频变换位置和设备路径可以单独检查，但不能直接外推为具身训练吞吐收益。 |
| 2026-01 | [A Pragmatic VLA Foundation Model](https://arxiv.org/abs/2601.18692v1) · Read · [账本](../../backfill/2026-01.md#embodied-20261023-lingbot-vla) | 多视角语言与本体状态经 MoT / flow matching 生成 action chunk，depth distillation 与 sharding 设计把泛化、几何监督和训练效率连起来。 |
| 2026-02 | [feat(dataset): add streaming video encoding + HW encoder support](https://github.com/huggingface/lerobot/pull/2974) · Read · [账本](../../backfill/2026-02.md#embodied-20261023-lerobot-streaming) | 按相机线程持续编码减少 episode 间 PNG 回读等待，但有限队列丢帧要求把观察完整性纳入验收。 |
| 2026-03 | [v3.0.0-beta](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0-beta) · Read · [账本](../../backfill/2026-03.md#embodied-20261023-isaaclab-3beta) | 多物理后端与 Warp 数据路径拆开仿真依赖，同时引入四元数、写入和 tensor 接口的正确性迁移要求。 |
| 2026-04 | [LeRobot v0.5.1](https://github.com/huggingface/lerobot/releases/tag/v0.5.1) · Read · [账本](../../backfill/2026-04.md#embodied-20261023-lerobot-v051) | revision-safe Hub cache将数据版本落到实际路径；4月交付不代表4月首次实现。 |
| 2026-05 | [TorchCodec v0.13.0](https://github.com/meta-pytorch/torchcodec/releases/tag/v0.13.0) · Read · [账本](../../backfill/2026-05.md#embodied-20261023-torchcodec-v013) | 增量多流编码需核frame count与finalization，不能自动保证机器人传感器同步。 |
| 2026-06 | [Isaac Lab v3.0.0-beta2](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0-beta2) · Read · [账本](../../backfill/2026-06.md#embodied-20261023-isaaclab-beta2) | MCAP回放以控制事件和终止语义检查升级兼容性，不能证明真机或物理确定性。 |
| 2026-08 | [ChainVLA](https://arxiv.org/abs/2608.02326v1) · Read · [账本](../../backfill/2026-08.md#chainvla-2608-02326) | 任务记忆和未执行计划共同进入下一段动作生成。 |
| 2026-08 | [LeRobot v0.6.1](https://github.com/huggingface/lerobot/releases/tag/v0.6.1) · Read · [账本](../../backfill/2026-08.md#lerobot-v061-20260803) | 发布中的统计量恢复与错误传播修复共同保护训练输入语义。 |

## 日期、版本与证据边界

- GitHub 按上海事件月归档：GR00T #714 的 6/30 UTC 合并为上海 7/1，计入 7 月。论文保留 v1 原始 citation 日期，不由编号猜时间。
- LeRobot streaming encoding 在 2 月 #2974 / v0.4.4 已交付；3 月发布文章不是首次。v0.5.1 的缓存机制在 3/27 UTC（上海 3/28）合入、4/7 发布；v0.6.1 的两项实现 7/30 合入、8/3 发布。release 作为交付节点，PR 不重复计数。
- Isaac Lab 分别保留 3 月 beta、6 月 Beta 2、9 月 EA，不能倒写成 10 月 GA；3 月 `.data` 的 Warp 接口也不借用后来的 ProxyArray。
- 公开代码、测试文件与作者实验是不同证据；本仓库未运行 GPU、仿真、机器人或编解码复现。新增项均 NEW；GR00T 工作流已有 DIGESTED 等历史状态不重新评级。

## 核查范围与可复核文件

- [Q1 范围说明](q1/README.md)与[逐来源记录](q1/summary.json)：LingBot-VLA citation/方法、TorchCodec 0.10 固定版本、LeRobot #2974 和 Isaac Lab beta；复用 DreamZero/MEM。
- [Q2 逐来源与范围](q2/summary.json)：三个 release、相关 PR 与固定版本实现／测试；复用 π0.7、GR00T N1.7、MiraBench、OSCAR 及两个观察项。
- [Q3 逐来源与范围](q3/summary.json)：复用 7/9 月记录，定向新增 ChainVLA 和 LeRobot v0.6.1，核对 GR00T #714 月份。
- 保存 primary 元数据、固定 SHA、内容 hash、核验位置与原创摘要；不复制完整网页、论文摘要或源码 diff。跨月引用和复用来源在聚合账本中可追溯。

## 保留与未覆盖

九份原月报正文逐字保留，新增补充统一锚点为 `embodied-20261023`。旧 Accepted、Decision、Historical Audit 与 frontier cursor 保留；只给共享 P1 原具身阶段增加按问题选读，不改 P0 或个人已读状态。月报的新 Inference Systems Watch 只说明此次未重扫独立推理引擎，不能借具身部署内容宣称覆盖 vLLM / SGLang / TensorRT-LLM。

仍未完成每月全分类论文枚举、openpi / OpenVLA / OFT / SmolVLA 全事件历史、全量 PR 代码审计或独立实测。已有四厂商、HF、RL 等 Watch 沿用各自原核验范围。10 月未结束，本次不生成 10 月正式月报。

[链接、格式与原文保留验证](validation.json)记录本次发布前检查；这些是文档检查，不是外部实现测试。

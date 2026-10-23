# 2026-10-23 具身月报历史补充 · Q1 来源审计

范围为 2026 年 1–3 月的定向历史补课。先核验并写入原始月份 backfill，再由月报汇总；不是 frontier scan，也不是当月全量来源覆盖。本批新增 4 个 Accepted 来源，归并 2 个已有来源；一个来源出现在两栏仍只计一次，不修改历史 Accepted 或扫描游标。

- [1 月月报](../../../monthly_signal_2026-01.md#embodied-20261023) / [1 月账本](../../../backfill/2026-01.md#embodied-20261023-lingbot-vla)
- [2 月月报](../../../monthly_signal_2026-02.md#embodied-20261023) / [2 月账本](../../../backfill/2026-02.md#embodied-20261023-lerobot-streaming)
- [3 月月报](../../../monthly_signal_2026-03.md#embodied-20261023) / [3 月账本](../../../backfill/2026-03.md#embodied-20261023-isaaclab-3beta)
- [机器可读归并结果](summary.json)、[来源元数据及文件 hash](sources.json)、[编辑前原文 hash](baseline_hashes.json)

## 核验方法与判断

| 来源 | 原始时间与本次核验范围 | 原创核验摘要 / 边界 |
|---|---|---|
| [LingBot-VLA v1](https://arxiv.org/abs/2601.18692v1) | 01-26 17:08:04 UTC；抓取原始 abs metadata 与 v1 方法页面，核 citation_title / citation_author / citation_date，§3–5 | 核验多视角/语言/本体状态输入、连续 action chunk、MoT、flow matching 与 depth distillation。§4.2 的 sharding/kernel 机制可作为训练读物；评估底盘固定，不能推断移动操作泛化。未引用后续 v4 或当前仓库更新替代 v1。 |
| [TorchCodec 0.10](https://github.com/meta-pytorch/torchcodec/releases/tag/v0.10.0) | 01-22 15:56:23 UTC；tag `0b261b98080925f2b709712a5491a1e8dd817065` | 核 release、decoder transforms、encoder 接口与性能建议。它提供视频阶段的工程边界；没有把 release 的编码加速数字改写成机器人训练收益。 |
| [LeRobot #2974](https://github.com/huggingface/lerobot/pull/2974) | 02-23 12:57:43 UTC merge `e96339a3b49ac080e11664d2be4737987d861a57`；v0.4.4 于 02-27 18:07:31 UTC 发布 | 核队列、线程与测试文件。有限等待后丢帧是明确实现行为，不能称完全无阻塞或无损。3 月 v0.5 博客是汇总，不能据此把该机制首次交付改成 3 月。 |
| [Isaac Lab 3.0 beta](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0-beta) | 03-17 06:38:37 UTC；tag `a4a7602f29e755e2673fe0022ea35566df6dd7d5` | 核 release、固定迁移说明与 articulation 数据接口。beta 的 `wp.array`、XYZW、index/mask 是这次历史对象；后续正式版与 ProxyArray 不能倒写进 3 月。 |
| [DreamZero](https://arxiv.org/abs/2602.15922v1) / [MEM](https://arxiv.org/abs/2603.03596v1) | 复用 [10-13 元数据与方法核验](../../2026-10-13-embodied-followup/model_sources.json) | 既有账本归并；不新增来源数量、不改变 Deep Dive/Read 与 NEW 状态。部署的缓存、记忆生命周期建议标为工程推断。 |

审计文件只保留 metadata、hash、固定版本信息与原创核验摘要，不保留第三方正文、abstract 或代码 patch。临时原文用于核查但不纳入仓库。所有性能与能力描述仍是作者/项目披露；本批未运行外部脚本、训练、仿真或数据处理实验。

## 实际覆盖与缺口

已定向检索 LingBot-VLA，检查 PI 官方博客日期索引，以及 LeRobot、TorchCodec、Isaac Lab release 索引；对上表来源继续固定版本与方法/代码核验。索引检查与 Accepted 分开：LeRobot v0.4.3、v0.5.0 和 Isaac Lab 2.3.2 等索引可见不构成新增接受；v0.5.0 元数据仅为版本边界辅助证据。OpenVLA/OFT、GR00T 与 openpi 当月全部变更没有完成枚举，cs.RO/cs.LG/cs.CV 也没有全量重扫，不将这些缺口写作“检查后没有重要新增”。

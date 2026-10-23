# 具身数据与 I/O：从 Episode 到正确的 GPU Batch

[核心入口](embodied_infra.md) · [模型入门](embodied_models_primer.md) · [六阶段路线](../roadmap.md#stage-2)

## 1. 问题：为什么一个 shape 正确的 batch 仍可能是错的

机器人示范同时包含视频、低维状态、动作和任务信息。它们不是天然同一时刻采集；磁盘中的一行也不一定对应模型要消费的整段观察与动作。目标是让每个训练样本能追溯到一条 episode，并在优化读取速度时保持语义不变。

[OXE](https://arxiv.org/abs/2310.08864v1)把跨机器人学习带入数据讨论，[DROID](https://arxiv.org/abs/2403.12945v1)强调真实环境多样性；[LeRobotDataset v3](https://huggingface.co/blog/lerobot-datasets-v3)处理多 episode 的存储与索引。前两者的数据与学习价值，不能由一种文件格式自动实现。

<a id="sample-contract"></a>
## 2. 机制：先定义逻辑样本，再选择存储

最小样本身份建议包含 dataset revision、episode ID、anchor time、observation offsets、action horizon 和变换版本。字段名称可随实现变化，但语义必须可解释。

| 层次 | 需要表达的信息 | 常见错误 |
|---|---|---|
| Episode | 任务、开始／结束、成功／失败／超时、机器人与采集身份 | 把两次 reset 两侧的帧接成一段 |
| Observation | 各相机时间戳、本体状态、语言、标定 | 用文件顺序冒充传感器同步 |
| Action | 目标／增量、坐标系、单位、控制模式、有效时刻 | 将动作记录时间当实际执行时间 |
| 训练窗口 | 历史观察、未来动作、缺失与 padding mask | padding 被当真实示范参与 loss |
| Lineage | 原始数据、processor、normalization 与 split 版本 | 更新 stats 后旧 checkpoint 悄悄换含义 |

以纯教学样例说明边界：episode A 在 1.00 秒结束，样本锚点为 0.90 秒，动作间隔为 0.05 秒，预测四步需要 0.90、0.95、1.00、1.05 秒。若定义 1.00 为最后有效动作，则 1.05 必须按约定 mask／拒绝采样，不能读取下一个 episode 的第一帧；若结束时间是右开边界，1.00 也应无效。先写清边界约定再讨论实现。

视频帧可能有自己的 presentation timestamp；不应把帧编号除以名义 FPS 当作所有媒体的精确时间。插值、最近邻、容差和缺帧策略会改变训练目标，必须在数据契约中声明。

想把这些字段接回训练，可看[一个样本到 loss 与动作的完整算例](embodied_models_primer.md#sample-to-action)：两路图像、状态、四步动作标签、有效 mask 与 L1 分母，再接到在线执行。这是教学推演，不是数据实验结果。

## 3. 存储机制：逻辑 Episode 与物理 Shard 分开

LeRobotDataset v3 将多个 episode 的低维数据与视频存进共享文件，通过 metadata 定位片段；读取者必须同时理解 schema、episode offset 与视频时间。迁移后“每个 episode 一个文件”的假设不再可靠。官方[当前格式文档](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3)是实现入口，具体 API 以固定版本为准，不能把今天文档里的所有能力倒推到最初版本。

工程取舍是减少小文件请求与保留随机窗口访问能力。更大 shard 可能减少 metadata 开销，却扩大随机读取和解码范围；顺序预取可能更有效率，但若改变任务或场景采样分布，就已经改变训练实验。

视频通常依赖关键帧与中间帧编码。需要某个时间窗口，不等于只读同等数量的压缩帧；应测真正读取字节、解码帧数和返回有效帧数。TorchCodec 的版本／测试核查见[GitHub 补扫](../../research/tracking/embodied_followup_2026-10-13.md)，解码后端替换属于正确性与性能共同验收。

<a id="pipeline"></a>
## 4. 配置指导：把等待拆成六段

**索引／读取 → 视频解码 → CPU 变换 → collate → H2D → GPU 计算**。并发后各段可能重叠，不能简单把六个平均值相加当成端到端延迟。

| 可调配置 | 先测的量 | 调整前保留的约束 |
|---|---|---|
| Shard / cache | 请求数、读取字节、冷／热缓存延迟 | episode 身份和采样分布 |
| Decoder / batch decode | seek 开销、解码帧数、返回时间戳 | 帧精度、色彩／范围、shape 与 dtype |
| Worker / prefetch | CPU、内存、队列深度、p95 取样时间 | 有界内存与确定的样本流 |
| Augmentation | CPU 变换开销、RNG 与跨帧一致性 | 不引入无意的时序或几何矛盾 |
| Pinned memory / H2D | 拷贝与计算重叠、host 内存压力 | 生命周期与同步正确性 |
| Distributed sampler | 各 rank 样本量、重复／遗漏、慢 rank | 全局分母、split、epoch 与 resume 位置 |

先测单进程基线，再逐个改变配置。增加 worker 可能只是把瓶颈转到 CPU、存储或内存；平均吞吐提高也可能仍被慢 rank 拖住。同时报告窗口数/s、有效动作步数/s 与 padding 比例；含部分 padding 的窗口仍可能有有效监督，不能只看原始窗口数，也不能把整窗一概剔除。

<a id="io-case"></a>
### 算一个 batch：为什么加 worker 也会先耗尽内存

下面是教学假设，没有读取真实数据或执行 benchmark。在两相机案例上增加 2 帧观察历史：`B=16`、2 路 RGB、每路 2 帧、分辨率 `224×224`。假设 CPU worker 已将图像转为 FP32，则图像张量为 `[16,2,2,3,224,224]`：

```text
image_bytes_per_batch = 16 × 2 × 2 × 3 × 224 × 224 × 4
                      = 38,535,168 bytes = 36.75 MiB
8 workers × 每 worker 2 个预取 batch 的图像容量 = 588 MiB
```

这只是这些预取槽位全部占满时的图像 payload 估算；不包含 worker 正在处理的数据、压缩视频、decoder 工作集、Python 对象、pinned staging 或主进程当前 batch。实际队列组织与 dtype 需按实现核对，不能把 588 MiB 当 RSS 上限；多个训练进程还会各自创建 loader。

再假设同一批 64 个请求帧只有 32 个唯一的 `(video, frame)`，由于 seek 与关键帧依赖，decoder 实际处理 96 帧。对唯一需求的解码放大是 `96/32=3`；按最终 tensor 槽位计算则是 `96/64=1.5`，两者分母不同。复用帧可以减少重复解码，但必须 scatter 回原来的样本、相机、时间和重复位置。

| 一次串行 batch 的教学计时 | 读取 | 解码 | 变换 | Collate | H2D | GPU 训练 |
|---|---:|---:|---:|---:|---:|---:|
| 耗时 | 8 ms | 24 ms | 6 ms | 2 ms | 4 ms | 30 ms |

串行合计是 74 ms。为单独解释流水线重叠，此处另设一个串行 CPU 供给 stage，四段合计 40 ms；上面的 8-worker 配置仅用于预取内存算例。若 CPU 供给、传输、GPU 计算能够完整重叠、资源也不互相争用，该假设下的稳态理想间隔下界是 `max(40,4,30)=40 ms/batch`。多 worker 能跨 batch 并行时，应使用汇总后的实际供给间隔，不能用单 batch 的 CPU 延迟代替它。这只是容量推导，不是预取后的实测结果；真实 warmup、队列波动、内存拷贝和解码争用还会改变间隔。仅增大 prefetch 无法把持续 40 ms 的供给能力变成 30 ms。

这时合理的下一步是固定同样的 sample IDs，对照窗口合并解码或 CPU 处理，检查像素、时间戳、action/mask，再测稳态间隔与 RSS。训练侧最终消费多少有效动作元素，还要接[全局 loss 分母](embodied_training.md#loss-normalization)；吞吐不能由窗口数独自代表。

### 从代码变化理解“等价”的具体含义

[LeRobot #3917](https://github.com/huggingface/lerobot/pull/3917)用有界 episode 池处理远端视频，并测试 rank 覆盖、顺序和恢复；池内混合不代表全局均匀随机，历史 worker 限制也已被后续版本改变。[TorchCodec v0.17](https://github.com/meta-pytorch/torchcodec/releases/tag/v0.17.0)允许拆开 demux、decode 和颜色转换，跨 CUDA stream 的执行依赖与底层存储寿命都须验收；只等 event 不自动解决 allocator 复用。

另一个反例是 [TorchCodec #1758](https://github.com/meta-pytorch/torchcodec/pull/1758)：时间戳可能没错，底层 decode surface 却被复用而导致像素错误。新增 surface 和 CPU/CUDA 像素对照提供缓解与回归证据，不能据此声称所有编码场景已彻底修复；该 PR 晚于 v0.17，使用时要查目标 wheel 是否包含它。以上代码与测试均未在本仓库运行。

<a id="oct14-io"></a>
### 2026-10-14：并行读取的三个独立验收条件

[本轮源码信号](../../research/tracking/frontier_scan_2026-10-14.md)使“更快且等价”可以拆成具体断言：

- [LeRobot #4874](https://github.com/huggingface/lerobot/pull/4874)：每 rank 再分给多个 worker，并按 DataLoader 轮转恢复。exact anchor order 要求同 dataset/seed/sampler/pool/worker/分布式拓扑；不保证随机增强完全回放。除 samples/s，还要看 RAM、batch 内 episode 多样性与训练端是否真的在等数据。
- [LeRobot #4804](https://github.com/huggingface/lerobot/pull/4804)：同视频窗口汇总 unique frame indices 解码，再 scatter 回原顺序和重复；64 MiB 是单次多样本解码估算，不是整个 worker RSS 上限。对照相邻窗与随机窗，并比较 timestamp、像素、dtype 和峰值。
- [Isaac Lab #8546](https://github.com/isaac-sim/IsaacLab/pull/8546)：交错 world prototype 时把 env ID、frame/pose 与 renderer destination 一起重排，否则张量 shape 正确也可能配错环境。作者有定向回归，公开 CI 仍有一项失败，不能直接当生产稳定性结论。

顺序、样本覆盖和观测身份必须分别验证；上述上游代码/测试未在本仓库运行。

<a id="deployment"></a>
## 5. 生产陷阱：训练端与执行端必须用同一种动作语言

归一化可以改变尺度，却不会自动对齐机器人动作语义。例如两个 7 维向量可能分别是绝对末端目标与相对位姿增量；拼接前必须有明确 adapter，不能由统一维数判断兼容。

把 checkpoint、输入尺寸／裁剪、tokenizer 或 processor、normalization stats、action representation、adapter 和 robot profile 一起管理。训练数据位置和 RNG 属于恢复状态；部署控制配置属于执行状态，二者有关但不相同。

[OpenVLA-OFT](https://github.com/moojink/openvla-oft)与 [openpi](https://github.com/Physical-Intelligence/openpi)的公开入口提供具体输入、动作与 normalization 接口参照。它们的 README 是实现线索，不是对任意新机器人直接可用的证明；具体版本证据见补扫账本。

## 6. 排查顺序：先证明没读错，再证明读得快

1. 抽一个失败样本，定位原始 episode 与确切帧，回放时间窗口。
2. 核相机／状态／动作时刻、坐标、单位、边界与 mask，检查反归一化。
3. 比较训练与部署 processor，核版本与异常缺失输入。
4. 核场景、任务或采集分组的 train / eval split，排相邻窗口泄漏。
5. 固定同一组样本比较 decoder、worker、cache 与 H2D；记录冷暖状态与分位数。
6. 最后检查跨 rank 样本重复、遗漏、重启位置和长尾；没有样本身份记录就先补证据。

这是 topic 中的诊断顺序，不代替已有具体系统的生产 runbook，也不代表已经跑过数据实验。

## 7. 相邻系统与小练习

数据多样性影响泛化；采样分布影响训练；解码与传输影响 GPU 利用率；processor 与 action 语义影响部署。优化任何一层都应保留其他层的验收条件。

数据进入[trainer step](embodied_training.md#step)，评估保留[case 与 episode 账本](embodied_sim_evaluation.md#episode-ledger)，部署核[processor 与动作工件](embodied_runtime.md#artifact)。这三个消费者共享数据身份，但各自检查不同边界。

先完成[纸面契约练习](../../practice/experiments/embodied_contract_lab.md)，再按 [A100 E04](../../practice/experiments/a100_fsdp_io_lab.md#e04)执行真实多模态 I/O 对照。两者当前均没有运行结果。我们第一轮讨论可以只解释上面的 episode 边界反例，确认样本正确性，再进入吞吐问题。


<a id="oct23-data-contract"></a>
## 2026-10-23 补充：先证明样本属于谁，再证明字节怎样进GPU

[LeRobot #4890](https://github.com/huggingface/lerobot/pull/4890)暴露了一个分布式边界：trainer知道自己的DP rank，spawn出的DataLoader worker却未必继承process group或正确环境变量。显式把trainer的DP shard序列化进dataset，才不会在自建launcher下各rank重复读全部episode。验收应检查跨rank sample ID的交集与并集，并故意让环境变量和trainer配置冲突；loader有输出、长度正确、GPU忙都不充分。

[AReaL #1812](https://github.com/areal-project/AReaL/pull/1812)处理另一个边界：视觉payload按trajectory的batch row归属，不能按token长度或把整批payload复制给每个分组。concat→计算advantage→split→筛选/重排→microbatch之后，图像仍须与对应token行一致；组间隔离和组内共享也要分别定义。这是可迁移的多模态数据契约，不是机器人性能实测。

[NVIDIA KvikIO教程](https://developer.nvidia.com/blog/how-to-read-data-into-gpu-memory-in-python-with-nvidia-gpudirect-storage/)则提醒，读到GPU数组只证明结果所在位置：KvikIO可能绕过cuFile，cuFile也可能走CPU compatibility path。应同时记录两层配置，并在适用平台用nvfs/posix运行时计数核路径；GDS也可能使用GPU bounce buffer。随后再测解码、时间对齐和batch-ready，不能从直传字节推断端到端训练提速。具体版本与核验边界见[扫描](../../research/tracking/frontier_scan_2026-10-23.md)。

以上检查均是待执行建议。优先按“样本覆盖/所有权正确→实际IO路径→吞吐瓶颈”排查，避免先优化一个产生重复或错配样本的loader。

<a id="oct19-memory"></a>
## 2026-10-19 补充：缓存预算与进程内存不是同一个账本

[LeRobot #4903](https://github.com/huggingface/lerobot/pull/4903)将 HTTP `response.read()` 改成流式迭代后拼接，避免 response/stream 引用环在 cyclic GC 前多留一份完整视频。它仍会分配最终 bytes，不能称零复制；业务 compressed-byte budget 和 decoder cap 都守住，也不能保证 worker RSS 有界。

排查顺序应是先固定依赖版本、worker 数和输入，再分别记录业务缓存、活跃 decoder、retained HTTP body、native live allocation 与 RSS。该 PR 作者在原支持栈的长时负载仍触发内存 guard；另一组新版 Torch/TorchCodec 的稳定性观察不属于本 PR 已验证的 locked runtime。因此正确下一步是同负载长跑与内存归因，而不是从单项修复推断 OOM 已彻底解决。来源与具体边界见[本轮扫描](../../research/tracking/frontier_scan_2026-10-19.md)；这里没有本地实测结果。

# Checkpointing

Checkpointing 是大规模训练里的“保险系统”和“生产流水线接口”。它不仅决定故障后能不能恢复，也决定训练能不能跨阶段迁移、能不能评估、能不能改并行度、能不能复现实验。

如果只能记一句话：checkpoint 不是保存权重，而是保存一次可验证、可恢复、可重分片的分布式训练状态。

## 1. Checkpoint 为什么是训练 infra 核心问题

Checkpoint 是训练系统从“能跑”走向“能长期跑”的分界线。模型规模、训练时长和集群规模上来后，它会同时影响可靠性、吞吐、存储成本和事故恢复时间。

### 模型越大 checkpoint 越难

模型规模变大后，checkpoint 难度不是线性增加。因为同时变大的还有：

- 参数文件大小
- optimizer states
- 并行 shard 数量
- rank 数量
- metadata 数量
- 写入请求数
- 恢复时的重分片成本
- 失败概率和恢复频率

百卡训练里 checkpoint 慢一点只是烦；万卡训练里 checkpoint 慢会直接吞掉有效训练时间。更糟的是，保存越慢，故障窗口越大，故障后重复计算越多。

### Checkpoint 不只是保存参数

只保存 model weights 只能用于推理或 finetune 起点，不能保证继续训练。继续训练需要恢复训练动态：

- optimizer momentum/variance 决定下一步更新方向
- LR scheduler 决定学习率
- RNG state 决定 dropout、data shuffle、sampling
- dataloader state 决定数据位置
- global step 决定日志、评估、scheduler、保存策略
- parallelism metadata 决定 shard 如何拼回模型

恢复后 loss spike，很多时候不是权重坏了，而是 optimizer、RNG、data cursor 或并行 metadata 不一致。

### 为什么 checkpoint 会影响训练吞吐

Checkpoint 影响吞吐的路径：

- rank 同步等待保存点
- GPU tensor 拷贝到 CPU 或 staging buffer
- CPU serialization
- 本地 NVMe 写入
- 共享文件系统或对象存储写入
- metadata commit
- 校验和 manifest 写入
- 旧 checkpoint 清理

如果这些动作阻塞训练 step，step time 会出现周期性 spike。异步 checkpoint 可以缓解，但它引入一致性和资源竞争问题：后台 IO 可能与 dataloader、NCCL 或下一次 checkpoint 抢资源。

## 2. Checkpoint 类型

不同 checkpoint 类型解决的问题不同。生产系统里通常不会只保留一种形态，而是训练恢复用 sharded/distributed，发布和评估用 consolidated/full，容错用 async 或 elastic。

### Full checkpoint

把完整模型状态聚合成一个或少数文件。优点是简单、容易给下游加载；缺点是大规模训练不可扩展，聚合和写入会成为瓶颈。

适合：

- 小模型
- 单机或少量 GPU
- 发布推理权重
- 训练结束后的 consolidated export

### Sharded checkpoint

每个 rank 或每组 rank 保存自己持有的 shard。优点是保存快、无需集中聚合；缺点是与并行配置绑定更紧，恢复和转换更复杂。

适合：

- TP/PP/DP/FSDP/ZeRO 训练中间保存
- 大模型频繁保存
- 训练恢复优先，而不是人工阅读权重文件

### Distributed checkpoint

分布式 checkpoint 不只是“很多 shard 文件”，而是包含 shard layout、tensor metadata、parallelism metadata、load/save planner 和一致性协议。Megatron Core 提供 `dist_checkpointing` 包，PyTorch 也有 distributed checkpoint 方向，核心都是让分布式状态保存/加载成为框架能力。

适合：

- 多维并行训练
- 保存和加载并行度可能不同
- 需要跨训练阶段迁移
- 需要高性能恢复

### Async checkpoint

训练主流程把状态快照交给后台线程/进程/服务写入，尽量减少前台停顿。关键不是“异步”两个字，而是 snapshot consistency：后台写的必须是某个 step 的一致状态，不能一边训练更新一边写脏数据。

适合：

- checkpoint 文件很大
- 保存频率较高
- 有本地 NVMe 或独立 checkpoint worker

### Incremental checkpoint

只保存相对上一次 checkpoint 改变的部分。理论上可降低写入量，但训练中 optimizer state 每步都变，真正增量空间不总是大。更适合配合存储系统做 block-level dedup 或针对部分状态做增量。

适合：

- 存储成本敏感
- checkpoint 间差异可控
- 有专门存储系统支持

### Elastic checkpoint

支持恢复时 world size、TP/PP/DP/FSDP/EP 配置变化。它要求 checkpoint 表示尽量 parallelism-agnostic，或者至少包含足够 metadata 做 reshard。

适合：

- 弹性训练
- 故障后缩容继续
- 从 pretraining 到 finetuning/post-training 切换并行配置

## 3. 保存内容

一个可继续训练的 checkpoint 至少应考虑：

| 内容 | 为什么重要 |
|---|---|
| model parameters | 模型权重主体 |
| optimizer states | Adam `m/v`、master weights，决定继续训练轨迹 |
| gradients | 通常不必每次保存，但 pipeline/gradient accumulation 中断点可能需要 |
| LR scheduler | 恢复学习率和 warmup/decay 位置 |
| RNG states | dropout、初始化、数据采样、MoE routing 随机性 |
| data loader progress | 避免重复/跳过数据 |
| global step | 对齐日志、评估、scheduler、checkpoint cadence |
| parallelism metadata | TP/PP/DP/FSDP/EP/CP size、rank mapping、tensor shard offset |
| tokenizer / config | vocab、special tokens、hidden size、heads、RoPE、dtype 等 |
| MoE expert state | expert weights、router、expert placement、load balance 相关状态 |
| scaler / precision metadata | AMP/FP8 scale、amax history、loss scaler |
| experiment metadata | git commit、data version、framework version、CUDA/NCCL 版本 |

生产建议：把“训练恢复必需状态”和“审计/复现实验状态”都纳入 checkpoint manifest。前者保证能跑，后者保证知道自己跑的是什么。

## 4. 大规模训练中的瓶颈

大规模 checkpoint 的瓶颈通常不在 Python API，而在全链路 IO、metadata、同步等待和恢复路径。要把保存和加载都当成分布式系统问题看。

### 写入带宽瓶颈

如果每次 checkpoint 是 TB 级，写入带宽会决定保存时间。单个共享文件系统路径很容易成为瓶颈。常见做法是先写本地 NVMe，再异步搬到远端存储。

### Metadata storm

成千上万 rank 同时创建小文件、写 manifest、更新目录，会把 metadata server 打爆。对象存储也会遇到 list/put/head 请求风暴。

缓解：

- 合并小 shard
- 分层目录
- manifest 原子提交
- 限流上传
- 避免每个 rank 写大量小 metadata

### Rank 同步等待

保存点通常需要所有 rank 到达一致 step。如果某些 rank 慢，其他 rank 会等；如果某些 rank 保存慢，下一步训练也可能等。checkpoint 于是放大 straggler。

### Object storage 延迟

对象存储适合容量，不一定适合低延迟、大量小对象。保存到 S3/OSS/GCS 时要关注 multipart upload、重试、最终一致性、list 性能和凭证过期。

### Shared filesystem 抖动

NFS/Lustre/GPFS 等共享文件系统在大规模并发写下可能出现周期性抖动。训练侧看到的是 step time spike 或 checkpoint timeout，存储侧看到的是 metadata server 或 OST/OSS hotspot。

### Checkpoint 过大导致 step time spike

即使平均吞吐看起来不错，保存点的 step time spike 也会影响训练效率和监控告警。要单独记录 `save_start/save_end/sync_time/upload_time/manifest_time`。

### 恢复时间过长

训练恢复不只是读文件：

- 调度重新拉起作业
- 初始化分布式进程组
- 读取 manifest
- 下载或读取 shard
- reshard
- 恢复 optimizer/RNG/data
- warmup dataloader/cache

如果恢复要几个小时，故障成本会压垮训练计划。

## 5. Megatron / DeepSpeed / FSDP 的 checkpoint 差异

| 方案 | 核心形态 | 适合场景 | 主要风险 |
|---|---|---|---|
| 普通 `torch.save` | 单进程保存 Python/PyTorch state dict | 小模型、debug、导出权重 | 大模型聚合慢、单点 IO、缺少并行 metadata |
| Megatron distributed checkpoint | TP/PP/DP/EP aware sharded state + dist checkpointing | Megatron-Core 多维并行训练 | 需要正确维护 shard metadata 和并行配置 |
| DeepSpeed ZeRO checkpoint | ZeRO optimizer/parameter partitions 按 rank 保存 | DeepSpeed ZeRO 训练恢复 | ZeRO stage 与 world size 绑定，转换 full weights 可能慢 |
| PyTorch FSDP state_dict | full / sharded / local state dict 等模式 | PyTorch 原生 FSDP 训练和导出 | state dict 类型选择错误会导致 OOM 或难恢复 |

### Megatron distributed checkpoint

Megatron 场景里，checkpoint 必须理解 TP/PP/DP/EP 切分。一个 tensor 的 shard 不只是第几个文件，还包括 partition dim、offset、rank group 和可能的 pipeline layer mapping。优点是贴合 Megatron 训练恢复，缺点是要严格维护 metadata。

### DeepSpeed ZeRO checkpoint

DeepSpeed ZeRO checkpoint 保存 ZeRO 分片后的 optimizer、parameter 等状态。ZeRO-2/3 下，每个 rank 只持有部分状态。恢复训练很方便，但如果要导出完整 FP32 权重，通常需要额外 consolidation 流程。

### PyTorch FSDP state_dict

FSDP 有多种 state dict 形态：full、sharded、local 等。full state dict 方便下游使用，但可能 OOM；sharded state dict 更适合大规模训练恢复；local state dict 更贴近 rank 本地状态。生产中要按用途保存，而不是只存一种。

### 普通 torch.save

适合小模型和单机调试。大规模训练中直接 `torch.save(model.state_dict())` 常见问题是 rank0 聚合 OOM、保存慢、缺少 optimizer/RNG/data/parallel metadata。

## 6. 容错与恢复

容错不是 checkpoint 的同义词。Checkpoint 提供恢复锚点，训练平台还需要故障检测、作业重启、状态一致性验证和恢复后的健康检查。

### Failure detection

需要区分：

- 进程失败
- GPU XID/ECC
- 网络 timeout
- 存储 timeout
- 数据服务失败
- straggler 但未失败

不同故障对应不同动作。慢节点不一定该立即重启，checkpoint 写慢也不一定是训练代码问题。

### Rank restart

固定 world size 训练通常要求所有 rank 协调重启。elastic training 允许部分变化，但前提是 checkpoint 能支持新的 rank layout。

### Partial failure

部分 rank 保存成功、部分 rank 保存失败是最危险状态。不能让 manifest 指向半成品 checkpoint。生产上应采用 staging path + atomic commit：只有所有 shard 校验通过，才发布 latest。

### Checkpoint consistency

一致性至少包括：

- 所有 rank 来自同一 global step
- model 与 optimizer 对应
- scheduler 与 global step 对应
- RNG/data cursor 对应
- manifest 完整且可校验

### Exactly-once vs at-least-once

数据消费语义很关键。很多训练系统实际做到的是 at-least-once 数据读取：恢复后可能重复少量样本。是否可接受取决于训练规模和数据 pipeline。对需要严格复现实验的场景，要保存 dataloader cursor 和 shuffle seed。

### Elastic training

Elastic checkpoint 的难点在 reshard。TP=8 保存的权重要能用 TP=4 加载，FSDP world size 改变后 optimizer state 要能重新分布，MoE expert placement 改变后 expert shard 要能映射到新 rank。

## 7. 生产环境设计建议

一个可用的 checkpoint 方案必须同时回答四个问题：多久保存一次、保存到哪里、怎么证明它能恢复、坏了以后怎么回滚。

### Checkpoint 频率如何设计

用一个简单模型思考：

```text
期望损失 = 保存开销 + 故障概率 * 重算开销
```

实践建议：

- 小规模实验：按时间或 step 高频保存，方便回滚。
- 大规模预训练：常用 30-120 分钟一个 checkpoint，具体看故障率和保存成本。
- 不稳定新配置：先提高频率，稳定后降低。
- 关键阶段切换前后：额外保存 milestone checkpoint。

### Checkpoint 保留策略

建议分层保留：

- `latest`: 最近 1-3 个，用于故障恢复。
- `hourly`: 最近一天或几天，用于短期回滚。
- `milestone`: 阶段切换、数据配比变化、LR 变化前后。
- `release`: 可用于评估、推理、下游训练的清理版本。

不要只保留 latest。latest 如果在 loss spike 后覆盖了好状态，会很被动。

### 本地盘 + 远端存储如何分层

常见架构：

1. rank 写本地 NVMe staging。
2. 节点或 checkpoint worker 聚合/校验。
3. 后台上传远端对象存储或共享存储。
4. manifest 原子更新。
5. lifecycle 任务清理旧版本。

本地盘用于吞吐，远端存储用于可靠性。需要监控本地盘空间，否则异步上传慢会把节点写满。

### 异步保存如何避免一致性问题

- snapshot 后再让训练继续，后台只读 snapshot。
- 禁止后台直接读会被 optimizer 更新的 live tensor。
- 给每次 checkpoint 独立 staging 目录。
- manifest 最后写，并带校验和。
- 后台失败不能悄悄覆盖 latest。

### Checkpoint naming convention

推荐包含：

```text
ckpt/global_step=000123456/tp=4_pp=8_dp=64_ep=1_cp=1/manifest.json
```

目录名应包含 global step 和关键并行配置。详细 metadata 放 manifest，不要把所有信息塞目录名。

### Checksum / validation

至少做：

- shard 文件大小检查
- checksum 或 hash
- manifest 完整性
- tensor shape/dtype 校验
- rank count 校验
- load smoke test
- 恢复后首个 step loss continuity 检查

### 恢复演练

没有恢复演练的 checkpoint 只是心理安慰。建议：

- 每天或每 N 个 checkpoint 自动抽样恢复。
- 在小规模集群上做不同 world size 的 load test。
- 定期模拟 rank failure、存储失败、latest 损坏。
- 把恢复时间作为 SLO。

## 8. 常见故障与排障

Checkpoint 故障最麻烦的地方是：保存时可能不报错，恢复时才暴露；或者恢复能跑，但训练轨迹已经偏了。排障要把文件完整性和训练状态一致性分开看。

### Checkpoint 写一半失败

处理原则：半成品不能成为 latest。检查 staging 目录、manifest 是否原子提交、失败后是否清理或标记 bad checkpoint。

### 恢复后 loss spike

优先检查：

- optimizer state 是否加载
- LR scheduler step 是否正确
- RNG/data cursor 是否恢复
- mixed precision scaler/FP8 metadata 是否恢复
- 模型 config/tokenizer 是否一致
- 是否从错误 global step 恢复

### Optimizer state 不一致

现象：loss 连续性差、更新幅度异常、某些 rank gradient/param norm 异常。检查 optimizer shard shape、dtype、param id mapping、FSDP/ZeRO flatten order。

### Rank 数变化后无法恢复

原因通常是 checkpoint 与原并行配置强绑定。需要 distributed/elastic checkpoint 支持 reshard，或者先离线转换成 universal/full representation。

### Tokenizer/config 不匹配

现象：embedding/lm head shape mismatch、special token 错、评估异常。必须把 tokenizer 文件、vocab size、rope config、model config 纳入 checkpoint 或 release manifest。

### MoE expert shard 丢失

MoE checkpoint 不仅有 dense weights，还有 expert weights、router、expert placement。某个 expert shard 丢失可能只在路由到该 expert 时暴露，排查很痛。需要 expert-level manifest。

### Object storage 慢

看 multipart upload 延迟、失败重试、list/head 请求、限流、跨地域流量、凭证刷新。训练侧要有 upload queue depth 和 backlog 指标。

### Checkpoint 太多撑爆存储

需要 lifecycle policy，而不是人工删。删除前确认没有作业依赖该 checkpoint、没有评估任务正在读取、milestone/release 不被误删。

## 9. 与其他主题的关系

- [ZeRO](zero.md)：ZeRO checkpoint 保存分片 optimizer/parameter 状态，stage 越高恢复越依赖 metadata。
- [FSDP](fsdp.md)：FSDP state dict 类型决定保存是 full、sharded 还是 local。
- [Megatron-LM](../papers/megatron_lm.md)：TP/PP/DP/EP 多维切分要求 checkpoint 理解 tensor shard 和 layer mapping。
- [Tensor Parallelism](tensor_parallelism.md)：TP shard 保存和恢复必须记录 partition dim、offset、TP rank。
- [Pipeline Parallelism](pipeline_parallelism.md)：PP 按 layer/stage 切分，恢复时 stage layout 改变需要 layer remap。
- [Data Parallelism](data_parallelism.md)：DP 影响 optimizer/gradient 同步和数据进度保存。
- [MoE](moe.md)：expert state、router、expert placement 和 load balance metadata 都要保存。
- [MegaScale](../tech_reports/megascale.md)：万卡训练把 checkpoint 从功能问题升级成吞吐、容错和运维问题。
- [Fault Tolerance](fault_tolerance.md)：checkpoint 是故障恢复的锚点，但不是完整容错系统。
- [Llama 3](../tech_reports/llama3.md)：大规模 dense 训练需要多阶段 checkpoint、评估和发布 lineage。
- [DeepSeek-V3](../tech_reports/deepseek_v3.md)：MoE + FP8 + 大规模训练要求 checkpoint 覆盖 expert 和 precision metadata。

## 10. 面试题

Checkpoint 面试不应停在“保存哪些字段”，更要能解释为什么这些字段影响继续训练，以及在多维并行和存储抖动下如何设计恢复系统。

### 基础题

1. Checkpoint 和 model weights 有什么区别？
2. 继续训练需要保存哪些状态？
3. Optimizer state 丢失会有什么后果？
4. RNG state 为什么重要？
5. Full checkpoint 和 sharded checkpoint 的区别是什么？
6. Async checkpoint 解决什么问题？
7. global step 为什么必须保存？
8. tokenizer/config 为什么属于 checkpoint 相关元数据？
9. checkpoint 频率如何影响训练吞吐？
10. 恢复后如何判断 checkpoint 是正确的？

### 进阶题

1. Distributed checkpoint 与普通多文件保存有什么区别？
2. FSDP full state dict 为什么可能 OOM？
3. ZeRO-3 checkpoint 为什么难以直接给推理使用？
4. TP=8 checkpoint 如何恢复到 TP=4？
5. PP stage layout 改变后如何加载 checkpoint？
6. MoE expert checkpoint 有哪些特殊问题？
7. FP8 训练 checkpoint 需要保存哪些 precision metadata？
8. checkpoint manifest 应包含哪些字段？
9. 如何设计 checkpoint 的 atomic commit？
10. incremental checkpoint 在 LLM 训练中为什么不总是有效？

### 生产环境题

1. checkpoint 写一半失败，系统应该如何处理？
2. 恢复后 loss spike，你第一小时查什么？
3. object storage p99 变慢，训练侧会看到什么现象？
4. checkpoint 保存导致 step time 周期性 spike，如何定位？
5. rank 数变化后无法恢复，如何补救？
6. tokenizer 不匹配会如何暴露？
7. MoE expert shard 丢失如何发现？
8. checkpoint 太多撑爆存储，如何设计保留策略？
9. dataloader state 没保存会造成什么训练风险？
10. 如何做恢复演练？

### 系统设计题

1. 设计一个 10K GPU 训练平台的 checkpoint 系统。
2. 如何设计本地 NVMe + 远端对象存储的分层 checkpoint？
3. 如何支持 Megatron TP/PP/DP checkpoint reshard？
4. 如何支持 FSDP 训练和 HuggingFace 推理之间的权重转换？
5. 如何为 MoE 模型设计 expert-aware checkpoint manifest？
6. 如何把 checkpoint 与 fault tolerance scheduler 集成？
7. 如何定义 checkpoint SLO？
8. 如何设计 checkpoint 监控面板？
9. 如何在 checkpoint 后自动触发评估任务？
10. 如何设计跨阶段 pretraining -> SFT -> RL 的 checkpoint lineage？

## 11. 生产环境思考题

1. 如果一个 10K GPU 作业每次 checkpoint 停 10 分钟，训练效率损失是多少？
2. 如果 latest checkpoint 损坏，但旧 checkpoint 被清理了，平台哪里设计错了？
3. 如果恢复后数据重复消费 0.1%，是否可接受？
4. 如果 TP size 改变后 optimizer state 无法加载，应该在线 reshard 还是离线转换？
5. 如果异步 checkpoint 上传积压，是否应该继续训练？
6. 如果对象存储 list 操作很慢，manifest 设计如何避免依赖 list？
7. 如果 checkpoint 保存与 dataloader 共享本地 NVMe，会发生什么？
8. 如果 MoE expert placement 改变，旧 checkpoint 如何映射到新 expert ranks？
9. 如果 FP8 amax history 没恢复，短期 loss 会如何表现？
10. 如果只保存 model weights，不保存 scheduler，长训练恢复会有什么偏差？
11. 如果 checkpoint 文件很多，metadata server 打满，如何重构文件布局？
12. 如果某个 rank checkpoint 比其他 rank 慢 5 倍，如何定位？
13. 如果训练中途升级框架版本，checkpoint compatibility 如何验证？
14. 如果 checkpoint load 很慢，如何区分存储慢、reshard 慢和 CPU deserialize 慢？
15. 如果要做弹性训练，checkpoint 格式需要从第一天支持哪些 metadata？

## 参考入口

- [ZeRO](../papers/zero.md)
- [MegaScale](../tech_reports/megascale.md)
- [Checkpoint 面试手册](../interview/checkpoint.md)
- [Megatron Core dist_checkpointing 官方文档](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/dist_checkpointing.html)
- [DeepSpeed Model Checkpointing 官方文档](https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html)
- [PyTorch FSDP 官方文档](https://docs.pytorch.org/docs/stable/fsdp.html)

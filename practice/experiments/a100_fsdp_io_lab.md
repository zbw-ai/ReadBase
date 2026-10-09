# A100 实验课程：从状态分片到具身数据与 RL 链路

[实验入口](README.md) · [工程实践](../README.md) · [首页](../../README.md) · [知识地图](../../KNOWLEDGE_GRAPH.md)

状态：**设计，E00–E08 全部未执行**。本页不包含实测结论或可直接提交的集群作业。已知可用 A100、最多 64 卡；具体卡型、互联、存储、软件版本及使用授权尚未核对。

## 从哪里开始

学习路线回链：[具身六阶段](../../05-embodied-infra/roadmap.md) · [GPU Systems](../../01-systems/roadmaps/gpu-systems.md) · [Distributed Systems](../../01-systems/roadmaps/distributed-systems.md) · [Inference](../../03-inference-infra/roadmap.md) · [Agentic RL](../../04-rl-infra/roadmap.md)。本页统一维护实验设计，不在各路线重复写一份结果。

建议顺序：**E00a 环境画像 → E01 状态账本 → E02 通信生命周期 → E04 多模态 I/O**。遇到显存或恢复问题再做 E03/E05；单机假设解释清楚后才扩到 E06。E07/E08 保留推理与 RL 能力，不把整个课程缩成训练 benchmark。

先读 [FSDP](../../02-training-infra/topics/fsdp.md)、[ZeRO](../../02-training-infra/topics/zero.md)、[NCCL](../../01-systems/topics/nccl.md)；具身模型背景从[具身入口](../../05-embodied-infra/README.md)补齐。不要求读完整本硬件参考书再开始实验。

## 共同实验契约

每项实验执行前建立独立实验卡，填齐：问题、假设、环境和前置条件、控制变量、资源上限（节点／卡数／时长／GPU-hour）、步骤、数值门槛、指标、停止条件、实际结果、结论边界和回链。本页是课程与未执行卡，不替代具体运行计划。

- 环境清单：A100 40/80GB、SXM/PCIe、每机卡数、NVLink/NVSwitch、GPU–CPU–NIC 拓扑、IB/RoCE、MIG/MPS／共享情况、CPU/NUMA、容器资源和 `/dev/shm`、存储、驱动与框架版本。
- 公平对照：固定模型、初始化、样本身份、有效 global batch/tokens、loss normalization、optimizer 数学定义、精度、seed 与 backend；一次只改变目标变量。无法对齐的 dtype/master weight/fusion 等差异必须列出。
- 机制对照与完整 recipe 对照分开；后者只能说“该 recipe 更快”，不能把全部收益归因 FSDP 或某个参数。
- 计时与 profiler 分开运行；记录 warmup、有效窗口、重复次数、最慢 rank 的 step time、显存 allocated/reserved、有效 samples/tokens/frames、CPU/存储负载与异常步。
- 正确性阈值在执行前依据 dtype 和参考路径确定：loss、梯度／更新误差、样本身份、mask 与恢复状态；不能测完后为了通过而放宽标准。
- 统一停止条件：超过批准资源／时间预算，影响共享任务，出现非有限值、数值或数据对齐超阈值，或失去可靠测量依据。单项 OOM 记录为容量失败，不持续重试撑大资源。
- 当前每项卡的实际环境、执行日期、预算批准与结果均为空，因此不能标为 VERIFIED。

## 实验总览

| ID | 要回答的问题 | 起步规模 | 主要产物 |
|---|---|---|---|
| E00 | 手上的机器有什么真实边界？ | 单机只读画像；微基准另行授权 | 环境清单与按需链路曲线 |
| E01 | 状态分片到底省了哪部分显存？ | 1 卡参考，2/4/8 卡对照 | dtype/状态账本与更新误差 |
| E02 | 参数何时聚齐，通信能隐藏多少？ | 2–8 卡 | 参数／显存时间线与 trace |
| E03 | 显存换计算，怎样选重计算与 microbatch？ | 2–8 卡 | 独立消融结果 |
| E04 | episode 到 batch 慢在哪里？ | 单卡到单机，按需跨机 | I/O、decode、H2D 与 batch 等待分解 |
| E05 | checkpoint 能否恢复同一训练过程？ | 4→4 / 8→8；支持后再测 8→4 | 恢复验收与状态清单 |
| E06 | 扩到跨机时哪种分片布局划算？ | 单机到 16/32/64 卡，按问题选择 | 强／弱／容量扩展分开的曲线 |
| E07 | 推理并发、KV 与 Graph 怎样影响尾延迟？ | 1–4 卡 | TTFT/TPOT/p95 与有效吞吐 |
| E08 | rollout 到训练的数据与版本是否可信？ | 总预算 4–8 卡起步 | 版本账本、同步成本与有效供给 |

卡数是上限建议而非执行授权，不要求每档都跑；真实单机边界以环境为准。状态实验用受控小 Transformer，数据实验再接公开具身 workload，不拿不同模型的吞吐直接排名。

<a id="e00"></a>
## E00｜环境画像与必要微基准

- **问题／假设**：GPU 等待可能来自 Host、拓扑或 I/O，而非矩阵计算本身；先分辨资源与链路边界。
- **前置与控制**：先获得访问范围授权，区分共享／独占和分区实例；记录上方环境清单。只读查询不代表已获准访问任意集群。
- **步骤**：E00a 只收集必要画像；E00b 在单独批准的负载／文件／时长内，按当前问题选择代表性 BF16 GEMM、pageable/pinned H2D、P2P、collective 或指定文件读取测试。
- **指标／验收**：明确 dtype、shape、消息大小、单／双向、GPU／整机口径；FSDP 需测 AG/RS，不用单个 all-reduce 值代表全部通信。曲线最终必须解释 E02/E04/E06 的真实 workload。
- **安全／边界**：不改 BIOS、时钟／功率、ACS/IOMMU、网络全局配置；不 reset GPU、不清共享 page cache、不扫全盘。社区脚本需先审代码与版本，读文件压测也属于有负载操作。
- **结果／后续**：未执行。先填 E00a 环境表，之后才决定哪些 E00b 曲线必要。

<a id="e01"></a>
## E01｜DDP、ZeRO-1/2/3、FSDP1/2 的状态账本

- **问题／假设**：状态分片降低常驻显存，但 peak 还取决于临时参数、activation、缓冲与首个 optimizer step；不能只按参数量除卡数。
- **前置与控制**：固定可共同运行的小模型、样本与有效 batch；包含 DDP 和 DeepSpeed stage 0 基线；锁定支持 FSDP1/2 的版本。FP32 数值对照和 BF16 性能对照分开。
- **步骤**：先单卡确认一步更新；再在 2/4/8 卡中选必要规模，依次记录初始化、forward、backward、optimizer step 的状态 dtype/形状/驻留与峰值。
- **指标／验收**：参数、梯度、master weight、optimizer 状态、activation、buffer 分开计账；比较 loss 与更新误差。OOM 单独记容量界限，不换模型后继续比较速度。
- **结果／后续**：未执行。用实测账本回链 [FSDP](../../02-training-infra/topics/fsdp.md) 与 [ZeRO](../../02-training-infra/topics/zero.md)，不预填节省比例。

<a id="e02"></a>
## E02｜参数生命周期、分组、reshard 与 prefetch

- **问题／假设**：分组与预取改变 all-gather/reduce-scatter 的时机及临时工作集，小组不一定更快，保留参数也不一定最合算。
- **前置与控制**：复用 E01 配置；先固定拓扑、mixed precision、有效 batch 和重计算设置，只改一个参数。
- **步骤**：FSDP2 先对比 root-only／按 block 分组，再单独对比 reshard、prefetch；DeepSpeed ZeRO-3 的参数驻留设置另做消融，不把 API 名字直接等同。
- **指标／验收**：记录每个参数组的驻留区间、AG/RS 时间、exposed communication、峰值显存和 step time；trace 中的判断必须由不带 profiler 的计时复核。
- **结果／后续**：未执行。产出一层 forward/backward 的时间线；具体 API 以固定版本为准。

<a id="e03"></a>
## E03｜重计算、精度与梯度累积

- **问题／假设**：节省 activation 可能允许更大的 microbatch，但额外计算和通信时机也可能抵消收益。
- **前置与控制**：复用共同契约，事先确定容差；每项子实验有独立基线，不能同时打开多个开关后声称单项贡献。
- **步骤**：E03a 只改重计算范围；E03b 只改 mixed precision；E03c 固定有效 global batch，改 microbatch／累积组合，核验同步次数与 loss normalization。
- **指标／验收**：峰值显存、更新误差、有效吞吐、最慢 rank step time、额外重算与通信暴露量。
- **结果／后续**：未执行。关联[选择性重计算](../../02-training-infra/topics/long_context_training.md#selective-recompute)；A100 不作为原生 FP8/FP4 Tensor Core 加速验证平台。

<a id="e04"></a>
## E04｜具身多模态 I/O：从 episode 到 batch

- **问题／假设**：瓶颈可能是小文件／远端请求、视频 seek/decode、预处理、H2D 或采样不均；增加 worker 数不一定解决根因。
- **前置与控制**：选固定公开 episode/sample manifest，明确 camera、timestamp、state/action、mask 与 normalization；确认数据许可、存储后端和 codec 支持。
- **步骤**：先测 loader，再接真实训练；按层比较文件布局、读取缓存、CPU/CUDA 解码、batch prefetch，不把 WebDataset、DALI、LeRobot 当成同一层互斥工具。
- **指标／验收**：读取字节／请求数、seek/decode 时间、CPU/GPU 占用、H2D、batch 等待和有效 frames/s；校验样本身份、时间对齐与像素误差。JPEG/MP4 变换可能改变像素与空间成本，不能全归因 I/O。
- **缓存／安全**：分别记录应用、OS page cache、远端缓存；只隔离应用目录不等于全链路冷缓存，未知层标未知，不清共享缓存。CUDA 可用不意味着该卡支持目标硬件解码路径。
- **结果／后续**：未执行。episode 数据正文待建设，先从[具身入口](../../05-embodied-infra/README.md)与[系统基础](../../01-systems/README.md)建立问题清单。

<a id="e05"></a>
## E05｜保存成功之后，真的能续训吗？

- **问题／假设**：权重能加载不等于恢复 optimizer、RNG、数据游标和同一训练进度。
- **前置与控制**：固定 backend／版本与数据顺序，独立实验输出目录；明确保存提交完成点，禁止覆盖生产 checkpoint。
- **步骤**：连续训练作为参考，对比同拓扑 4→4 或 8→8 的保存重启；只有确认支持后，再做 8→4 的 reshard 恢复。故障注入仅针对获准的实验进程。
- **指标／验收**：样本身份／顺序、optimizer/RNG/step、loss 与更新误差、保存 stall、恢复时间。改变 world size 的对照单独标注，不预设 bitwise 一致。
- **结果／后续**：未执行。回链 [checkpoint](../../02-training-infra/topics/checkpointing.md) 与[恢复排障](../playbooks/checkpoint_recovery.md)。

<a id="e06"></a>
## E06｜跨机分片和规模扩展

- **问题／假设**：跨机全分片与机内分片／机间复制有不同通信和显存代价，单机最快配置未必能直接扩展。
- **前置与控制**：E01/E02 已解释单机结果；经批准后再选 16/32/64 卡，不要求全部档位；记录进程组与实际 GPU–NIC 放置。
- **步骤**：先固定模型和有效 global batch/tokens 做强扩展；再独立固定每卡工作量做弱扩展；增大模型是容量实验，另列结果。
- **指标／验收**：最慢 rank step、吞吐／卡、通信暴露、rank skew、存储/Host 争用；不能把弱扩展总吞吐增长说成同一任务加速。
- **结果／后续**：未执行。以可解释的规模拐点为目标，而非用满 64 卡。

<a id="e07"></a>
## E07｜推理并发、KV 与 CUDA Graph

- **问题／假设**：batching、缓存与 launch 开销共同决定吞吐和尾延迟，提高并发可能牺牲交互时延。
- **前置与控制**：固定 checkpoint、backend、请求集合、输入／输出长度分布、计时窗口与 warmup；1–4 卡内选满足问题的规模。
- **步骤**：先测并发曲线，再逐项启用 KV/prefix cache、CUDA Graph 等支持的特性；区分 prefill、decode 和排队。命中／未命中缓存分别记录，不能混成一个加速比例。
- **指标／验收**：TTFT、TPOT、p95、有效吞吐、显存、输出一致性；Graph 降低 launch 开销不代表矩阵乘法本身更快。具身观测回放可以作为后续独立 workload。
- **结果／后续**：未执行。回链[推理入口](../../03-inference-infra/README.md)；A100 回放结果不能证明端侧实时性或真机安全。

<a id="e08"></a>
## E08｜RL 数据、策略版本与权重同步

- **问题／假设**：生成吞吐提高不保证有效训练供给增加；版本、轨迹过滤、同步和 trainer idle 可能成为主要约束。
- **前置与控制**：总预算 4–8 卡起步，固定模型、轨迹契约、算法参数和统计分母；先确保离线固定轨迹更新正确。
- **步骤**：先比较固定轨迹下的 loss/logprob/mask 和更新，再接同步在线供给，最后针对权重同步或供给调度做单变量对照；记录生成与消费 policy version。
- **指标／验收**：权重同步耗时、trainer idle、有效轨迹／tokens、拒收原因、版本差、数值对齐；恢复时核验数据与策略状态边界。
- **结果／后续**：未执行。对照 [RL 状态边界](../../04-rl-infra/topics/agentic_rl.md#rl-state-boundaries)；模拟 producer 不等于已实现机器人在线 RL。

## 产物与公开边界

执行某项实验后才新增该项脚本／结果目录。仓库只保留可公开的环境摘要、配置、汇总指标、必要图表与复现说明；大 trace、权重、视频保留在获准存储，不上传内部数据、代码、节点地址或凭据。

后续结论必须注明 workload、版本、统计窗口及无法控制的变量，再回链相应 topic／playbook。没有实际运行，就一直保留“未执行”。

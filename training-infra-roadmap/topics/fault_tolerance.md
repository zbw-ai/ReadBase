<a id="large-scale-training"></a>
# 大规模训练稳定性与容错：从千卡到万卡

> 面试速答入口：[INFRA-09｜万卡规模效应与优化](../../private_resume/2026-08-llm-infra-interview-prep.md#infra-09) · [INFRA-03｜NCCL hang/checkpoint 恢复](../../private_resume/2026-08-llm-infra-interview-prep.md#infra-03)

## 0. 先说结论

万卡训练不是“把千卡参数乘十”。很多问题在小规模已经存在，但到万卡会跨过阈值，成为决定训练能否按期完成的一阶矛盾：

1. 单个组件的小概率故障，聚合到整个作业后变成高概率事件；
2. 同步训练由最慢 rank 决定，局部 p99 长尾会扩散为全局 step slowdown；
3. 通信、进程启动、数据读取与 checkpoint 会形成跨节点、跨机架的拥塞和惊群；
4. 一个 rank 的 first failure，常在远端表现为大量 collective timeout；
5. 只优化峰值 MFU 不够，最终要优化长期 **goodput** 和 **time-to-train**。

一句话概括优化目标：

> **降低故障发生率，缩小故障影响面，缩短检测定位时间，降低恢复与重算成本；同时治理 topology 和 tail latency，守住稳态性能。**

---

## 1. 为什么规模会带来质变

### 1.1 故障概率随故障单元数量累积

若将 GPU、host 或 node 抽象为独立故障单元，每个单元在一个时间窗内出错的概率为 `p`，`N` 个单元至少发生一次故障的概率为：

```text
P(any failure) = 1 - (1 - p)^N
```

当 `p` 很小时，可用 `Np` 理解一阶趋势。该模型只是直觉工具，生产集群并不真正独立：同一交换机、机架、电源、镜像、driver 或数据服务会产生相关故障，故障域可能一次影响一组节点。

因此，万卡作业不能以“正常情况下不会坏”为前提，而必须把检测、隔离、替换和恢复放进正常控制流。

### 1.2 同步训练放大最慢 rank，而不是平均 rank

对存在 collective 或 pipeline 同步边界的训练 step，可以用下面的近似建立直觉：

```text
T_step ≈ max(T_rank_0, T_rank_1, ..., T_rank_N-1)
```

某张卡降频、一个 NIC 重传、一个 dataloader shard 抖动，或者某个 MoE rank 收到更多 token，都可能让其余 ranks 等待。规模越大，采到尾部事件的概率越高，所以“平均 GPU utilization 正常”不能证明训练没有 straggler。

### 1.3 并发操作会形成系统级惊群

小规模下可以接受的动作，到万卡会变成风暴：

- 万级进程同时拉镜像、解析配置、rendezvous、创建 communicator；
- 数千 worker 同时读取大量小文件或请求 metadata service；
- 所有 ranks 同时保存 checkpoint shard 或在恢复时同时拉取；
- 所有进程同时上报高基数日志、trace 和 metrics。

所以优化不只是让单个请求更快，还要通过分层、聚合、缓存、限流、错峰和 backpressure 控制并发形态。

### 1.4 峰值 throughput 不等于长期产出

建议把四类指标分开：

| 指标 | 回答什么问题 | 不能说明什么 |
|---|---|---|
| MFU | 有效模型 FLOPs 占硬件峰值的比例 | 不包含故障停机、回滚和模型效果 |
| throughput | 稳态时每秒处理多少 tokens/samples | 不包含失败后未提交的工作 |
| effective training time | wall-clock 中真正执行有效 iteration 的比例 | 不直接说明每个有效 iteration 是否足够快 |
| goodput | 单位已分配资源时间内成功提交且满足约束的有效训练工作 | 需要明确“有效”和分母口径 |

训练场景可用下面的工程定义：

```text
goodput = 已成功提交、数值有效的训练 tokens
          --------------------------------
          已分配 GPU 数 × wall-clock time
```

它会把排队后的启动时间、checkpoint stall、故障检测、节点替换、恢复、回滚重算和 straggler 都计入。万卡系统要同时报告 MFU/throughput 和 job MTBF、MTTR、恢复损失、有效训练时间占比。

---

## 2. 六类规模化问题总览

| 领域 | 典型表象 | 根因方向 | 第一优先动作 |
|---|---|---|---|
| 故障与恢复 | 进程退出、Xid/ECC、节点失联、collective timeout | GPU/host/NIC/交换机/软件/数据服务 | 保存 first-failure 证据，确定故障域并隔离 |
| straggler | step p99 变长、同配置不同 run 性能漂移 | 降频、NUMA/PCIe、网络重传、数据或 expert 倾斜 | 做 rank/stage heatmap，找 first divergence |
| 通信与拓扑 | scale efficiency 下降、AllToAll/AllReduce 尾部恶化 | 跨慢速域、过订阅、路径冲突、group 映射错误 | 对齐逻辑并行组与物理拓扑，测 exposed time |
| 调度与启动 | 分配后长时间没有 first step、初始化偶发 hang | gang scheduling、镜像/配置、rendezvous、communicator storm | 分阶段启动、health gate、fail-fast |
| 数据与 checkpoint | GPU 周期性空转、保存/恢复击穿存储 | 小文件/metadata storm、并发 shard IO、数据倾斜 | 预分片/缓存、分布式异步保存、限流与原子 manifest |
| 可观测与正确性 | 最后日志全是 timeout，恢复后 loss/data 不连续 | 根因传播、日志风暴、状态不完整或 silent corruption | 统一身份与事件序列，恢复后做连续性门禁 |

下面逐项展开。重点不是背故障清单，而是建立“症状 → 故障域 → 证据 → 动作 → 验证”的闭环。

---

## 3. 故障与恢复：把异常当作正常控制流

### 3.1 常见故障域

- **GPU/device**：Xid、ECC、HBM、device lost、异常降频、kernel hang；
- **host**：进程退出、OOM、CPU/内存压力、NUMA 放置、机器重启；
- **节点内互联**：PCIe、NVLink/NVSwitch 或国产卡对应高速互联；
- **节点间网络**：NIC、光模块、IB/RoCE/HCCS、交换机、拥塞与重传；
- **软件**：driver/runtime/通信库版本漂移、死锁、资源泄漏、非法 shape；
- **外部服务**：存储、metadata、对象存储、数据服务、调度和控制面。

### 3.2 训练前：减少故障进入作业

1. **预检**：GPU/HBM、节点内 P2P、节点间 pair test、存储读写和版本一致性；
2. **健康评分**：不要只判断 alive/dead，记录慢卡、链路错误和历史故障；
3. **故障域感知分配**：避免把关键通信组随机跨越过订阅或已知不稳定域；
4. **隔离与冷却**：反复触发异常的节点进入 quarantine，修复并通过 burn-in 后再归队；
5. **小规模门禁**：同一 artifact 先过功能、数值、单机与多机 smoke，再放大。

### 3.3 训练中：快速失败优于长时间半死不活

鲁棒训练状态机应至少覆盖：

```text
RUNNING
  -> SUSPECTED          heartbeat / progress / error anomaly
  -> DIAGNOSING         stop global progress, collect lightweight evidence
  -> QUARANTINING       identify and evict bad failure units
  -> REALLOCATING       refill with healthy nodes
  -> RESTORING          load the latest committed checkpoint
  -> VALIDATING         check loss/data/version continuity
  -> RUNNING
```

一个 rank 已退出后，其他 ranks 往往只是卡在 collective。无限延长 timeout 不会恢复进展，反而扩大 GPU 空耗。更好的策略是识别“全局已不可能继续”的条件，快速协调退出，同时保存 first-failure、collective sequence 和拓扑证据。

### 3.4 fixed-world 与 elastic recovery 怎么选

- **fixed-world replacement**：保持 TP/PP/CP/EP/DP 大小和 shard layout 不变，用健康节点替换故障节点，再从 checkpoint 重启。对大模型多维并行最稳妥，恢复路径也更容易验证。
- **elastic data-parallel recovery**：只改变可弹性的 DP 副本数，模型并行组和 shard 语义保持不变。需要重新计算 global batch、data sampler、scheduler 和梯度归一化语义。
- **改变模型并行维度恢复**：例如 TP/PP/EP 改变，必须支持 checkpoint reshard、rank mapping 和 optimizer state 变换，风险远高于“换节点不换 world-size”。

不能因为调度器支持 elastic，就宣称训练语义天然可弹性恢复。先确认哪些并行轴、batch 语义和状态允许变化。

### 3.5 衡量恢复能力

- `MTTD`：故障发生到检测；
- 诊断与隔离时间；
- `MTTR`：故障到恢复有效 step；
- 自动归因/自动恢复覆盖率；
- 每次故障丢失的 steps/tokens；
- restart success rate；
- 恢复后 loss、optimizer、data cursor 和 version 是否连续。

---

## 4. Straggler：同步系统的尾部放大器

### 4.1 先区分持续慢与偶发慢

- **持续慢 host**：同一机器跨作业、跨 step 都慢，优先查时钟、功耗、温度、NUMA、PCIe、GPU/HBM 或 NIC；
- **周期性慢**：checkpoint、日志 flush、data refill、GC、内存整理或其他周期任务；
- **网络型长尾**：collective 或 P2P 某些 peer/rack 反复偏慢，查重传、拥塞、路由和 oversubscription；
- **workload 型长尾**：variable sequence、packing 不均、MoE token imbalance、不同 pipeline stage 计算量不同；
- **偶发全局慢**：共享存储、交换机、控制面或集群其他租户引起。

### 4.2 诊断顺序

1. 按 step 将时间拆成 data、forward、backward、optimizer、collective、checkpoint；
2. 每个阶段按 rank/host/device 输出 p50/p95/p99/max，而不是只看平均；
3. 用 heatmap 找最先偏离的 rank 和持续性；
4. 对照逻辑位置：它属于哪个 TP/PP/CP/EP/DP group，等待会传到哪里；
5. 将计算事件与 NIC/交换机、GPU clock、温度、ECC/Xid、CPU/IO 时间对齐；
6. 通过换节点、换 rank mapping、单机/节点对/小 group microbenchmark 做最小证伪。

### 4.3 常见优化

- 自动识别并隔离 persistent slow node；
- 固定 CPU affinity、NUMA 和 NIC 绑定，避免不同 run 漂移；
- 处理 variable-length 数据的 token-based batching 和 packing imbalance；
- MoE 观察每个 expert/rank token count、drop rate、capacity 和 dispatch time；
- 平衡 PP stage layer 与 virtual pipeline chunk，降低 stage skew；
- 降低周期性 checkpoint/日志与关键训练路径的资源争用；
- 通过 QoS 或资源隔离减少 noisy neighbor。

不要看到某个 NCCL kernel 很长就直接判定“网络慢”。它可能只是等待上游计算、数据或 peer 进入 collective。要找 first divergence，而不是最后一个阻塞点。

---

## 5. 通信与拓扑：先画故障域，再配并行组

### 5.1 两张图必须同时存在

1. **物理拓扑图**：device → 节点内互联 → NIC/rail → leaf/spine → rack/pod；
2. **逻辑并行图**：每个 rank 属于哪些 TP、PP、CP、EP、DP group，以及每种 group 的 tensor、消息大小、频率和同步敏感度。

只有 world-size 公式而没有 rank placement，不足以设计万卡训练。

### 5.2 拓扑映射原则

- 高频、强同步、latency-sensitive 的 group 优先放入最快且稳定的拓扑域；TP 通常优先留在单机高速互联内；
- EP 的动态 token exchange 和 CP 的 KV/context 交换也可能非常吃网络，不能简单都扔到跨机架域；
- PP 的通信 peer 较少但在 critical path，stage 邻接关系要匹配物理路径；
- DP 可能跨更慢域，但大 bucket 的 ReduceScatter/AllGather 仍需高带宽与有效 overlap；
- Parallel Folding 下 Attention 与 Expert 的逻辑网格并不要求所有轴机械正交，必须按真实 group construction 分析。

这些是初始原则，不是固定排序。最终 placement 取决于模型 shape、消息模式、链路层级和集群 oversubscription，必须用 scale curve 与通信 trace 验证。

### 5.3 典型优化手段

- hierarchical collective：先节点内规约，再跨节点/机架，最后节点内分发；
- topology/rail-aware rank mapping，减少热点路径和跨慢速域次数；
- 为训练流量设置可验证的 QoS、隔离或 admission control；
- 调整 bucket/message size，避免过多小消息或过晚通信；
- DP/TP/PP 的 prefetch、chunking、优先级和 async overlap；
- MoE dispatcher 的 load balance、AllToAllV count 校验和热点 expert 治理。

性能验收看的是 **exposed communication**：

```text
exposed_comm = step critical path 中未被计算隐藏的通信时间
```

通信总时长增加并不必然让 step 变慢；反之，即使 profiler 显示 async launch，也不等于通信真正被隐藏。要比较 overlap 开关前后的 critical path、计算是否被带宽争抢拖慢，以及 step time。

完整 collective 语义与排障见 [NCCL 与分布式通信算子](nccl.md#collective-map)，MoE 的逻辑网格与 Parallel Folding 见 [MoE](moe.md#parallel-folding)。

---

## 6. 调度与启动：从“起进程”变成分布式控制面

### 6.1 为什么 first step 可能等很久

万卡作业启动通常包含：

```text
resource allocation
-> node health gate
-> image/environment readiness
-> rank mapping and rendezvous
-> process spawn
-> communicator initialization
-> model/checkpoint/data initialization
-> first effective step
```

任何阶段只有总耗时而没有分段 trace，都会把问题表现成“调度慢”或“初始化 hang”。

### 6.2 优化方向

- 将 allocation-to-first-step 拆成明确阶段并设置独立 timeout；
- 镜像、依赖、模型 metadata 和常用数据 shard 预热；
- 分层控制：global driver 管状态机，node agent 管本机进程和健康；
- metadata 批量分发，避免所有 worker 打同一服务；
- 确定性 rank mapping，启动前校验 world-size 与各 process group；
- 先启动小 canary group 或分批建 communicator，再进入全量 barrier；
- gang scheduling 无法凑齐时快速反馈，而不是占着大部分 GPU 无限等待；
- 预留少量健康 spare capacity，缩短坏节点替换路径。

应记录 allocation、pod ready、process ready、rendezvous、communicator init、checkpoint restore、data ready 和 first step 的独立耗时与失败率。

---

## 7. 数据与 Checkpoint：避免 IO 和 metadata storm

### 7.1 数据路径

典型问题包括：

- millions of small files 导致 metadata service 成为瓶颈；
- 所有 ranks 在 step 边界同步 refill，形成周期性 GPU idle；
- shard 大小、样本长度或压缩解码成本不均；
- 重启后所有 worker 同时读取同一 checkpoint/data hot set；
- 数据损坏、重复消费或 cursor 回退造成 silent correctness issue。

优化手段：离线 pack/预分片、顺序大块读取、节点本地 cache、异步 prefetch、分层缓存、token-balanced shard，以及可重现的 sampler/data cursor。

### 7.2 Checkpoint 路径

万卡 checkpoint 必须被视为分布式事务，而不只是 `torch.save`：

1. 每个 rank 保存与其并行布局对应的 model/optimizer shard；
2. 使用分布式并行 IO，避免 gather 到单 root；
3. 将设备到 host/local storage 的关键 pause 与后台上传分开；
4. node-level aggregation、限速和错峰，避免存储惊群；
5. 写临时版本，所有必要 shard/checksum 成功后原子提交 manifest；
6. 保存 scheduler/scaler、RNG、global step、data cursor 和 parallel metadata；
7. 恢复后再验证 loss、optimizer、数据位置与并行布局，而不只验证“能 load”。

### 7.3 Checkpoint 间隔怎么定

间隔不是固定经验值，要用以下量实测：

- 有效 checkpoint pause `C`；
- 作业或故障域的 MTBF `M`；
- restore、诊断和资源重分配时间 `R`；
- 每次回滚丢失的平均训练进度；
- checkpoint 对存储和稳态 step 的后台干扰。

在“故障率恒定、保存开销固定、故障独立”的简化模型中，最优间隔量级常用 `sqrt(2CM)` 建立直觉；生产上要用真实故障分布、相关故障和 storage contention 校正。保存更频繁会增加正常开销，保存太稀会增加回滚浪费。

完整状态与存储设计见 [Checkpointing](checkpointing.md)。

---

## 8. 可观测、正确性与规模门禁

### 8.1 最小关联键

至少要能用以下键将应用与基础设施事件拼到同一时间线：

```text
job/run ID
model/config/data/checkpoint version
global step / microbatch / collective sequence
global rank + TP/PP/CP/EP/DP coordinates
host + device + NIC + rack/pod
```

没有逻辑并行坐标，就很难解释一个坏 rank 为什么让哪些 peers 等待；没有物理坐标，就很难把错误归因到 NIC、交换机或故障域。

### 8.2 分层 telemetry，避免日志反噬

- 常态只保留低成本 counters、阶段时间和代表性采样；
- 异常触发后提高采样率，保存 ring buffer/flight recorder；
- worker 日志聚合、去重并按 error signature 分类；
- 高基数 label 进入 trace/log，不全部塞入 metrics；
- profiler 只在短窗口或抽样 rank 开启，并验证观测开销。

### 8.3 正确性门禁

万卡“跑得动”不等于“训得对”。至少分三层：

1. **FUNCTIONAL**：collective count/order、checkpoint shard、data cursor 和 state machine 正确；
2. **NUMERIC**：小规模 reference 对齐、loss/grad/overflow、跨 rank invariant 和恢复连续性；
3. **EFFICACY**：代表性长窗口收敛与下游评测没有回归。

对 silent data corruption、版本漂移或错误恢复，只看进程存活和 throughput 无法发现。训练 artifact 应绑定 code、config、container、topology、data 和 checkpoint lineage。

### 8.4 分阶段扩容，而不是一步跳到万卡

示例门禁：

```text
1–8 cards       functional + numeric baseline
32–64 cards     process-group / topology / node-pair communication
hundreds        scale curve + overlap + checkpoint
thousands       tail latency + failure/recovery + storage pressure
10K             correlated failure domains + control-plane and operational goodput
```

具体卡数由集群形态决定。关键是每一级都要定义准入指标、失败退出条件和可复现 artifact；不能用更大规模掩盖小规模尚未解决的错误。

---

## 9. 生产排障 Runbook 摘要

概念章节只给调查顺序；执行命令和日志关键词应进入独立 playbook。

### 症状 A：作业 hang / collective timeout

1. 保存 job、step、collective sequence、rank/host/device/NIC、拓扑和首个异常时间；
2. 判断是否有 rank 进程退出、OOM、Xid/ECC、heartbeat 消失；
3. 核对所有 ranks 的 collective、group、count、dtype 和调用顺序；
4. 找 first bad rank，不把最后报 timeout 的 rank 当根因；
5. 检查节点内互联、NIC/交换机 counter、重传和拓扑；
6. 通过单机、节点对或小 group 测试缩小故障域；
7. 隔离坏节点，从已验证 checkpoint 恢复并检查连续性。

### 症状 B：没有 hang，但 step p99 持续恶化

1. 确认是所有 rank 同时慢，还是少数 rank 先慢；
2. 分解 data/compute/collective/checkpoint；
3. 对齐 GPU clock/temperature、CPU/NUMA、NIC 和 storage；
4. 检查 sequence/expert/stage load imbalance；
5. 进行换节点、换映射或关掉周期任务的单变量实验；
6. 用相同 workload 的 scale curve 和长窗口重新验收。

### 症状 C：恢复很慢或恢复后结果不连续

1. 区分 resource allocation、download、load、reshard、data init 和 first-step 时间；
2. 验证 manifest/checksum 和必要 shard 完整性；
3. 核对 model/optimizer/scheduler/RNG/data cursor/parallel metadata；
4. 限制恢复并发，防止 worker 同时打爆存储；
5. 检查 loss、optimizer step、LR、数据位置和版本连续性；
6. 将本次恢复时间和丢失训练量回填 checkpoint policy。

进一步排障见 [Slow Step Debug](../playbooks/slow_step_debug.md) 与 [NCCL](nccl.md#hang-diagnosis)。

---

## 10. 项目经验如何映射：万卡项目背景与 3K 直接证据分开

### 10.1 可以直接说的边界

> TX、X1 项目所在集群总规模分别约 1.4 万卡和 1.2 万卡，这是交付平台背景，不等于个人 owner 了完整万卡训练系统。我的直接规模证据是 X1 200B MoE 模型的 3K 卡连续稳定训练两个月；职责是模型侧从跑通、采集 profile、瓶颈归因，到并行策略、算子与通信优化、规模回归和性能验收的闭环。底层编译器、算子库、集合通信、网络和集群运维由对应团队实现，我负责提供稳定复现和 rank 级证据，并完成最终模型侧验证。

3K 实践可以支撑以下工程判断：

- 小规模通过后，必须在目标规模重新画 scale curve，不能线性外推；
- 平均 step time 不够，需要看 rank/stage tail 与 physical/logical topology；
- 优化一个瓶颈后要重新 profile，因为瓶颈会迁移；
- 性能达标之外还要验证 loss、精度、checkpoint 和连续训练窗口；
- 跨团队问题要给出可复现 workload、first bad event 和修复前后 A/B。

万卡部分应明确表述为“基于规模效应与公开生产系统总结出的设计判断”，不要说成自己管理过完整万卡集群。

### 10.2 面试前必须补的一张真实故障卡

只选一个亲历且允许披露的 3K 事件：

```text
workload 与规模：____________________
表面症状：____________________________
first bad step/event/rank：____________
逻辑并行坐标 + 物理节点：_____________
排除过哪些方向：______________________
我亲自采集/修改的证据：________________
底层团队修改了什么：__________________
同 workload A/B：_____________________
恢复/长窗口验证：______________________
个人边界：____________________________
```

如果细节没有确认，就宁可讲调查框架，也不要现场创造 NIC、交换机、checkpoint 或数值故障。

---

## 11. 面试回答模板

### 30 秒版

> 万卡训练有三个质变：小概率单点故障变成全局高概率事件；同步 step 被最慢 rank 的 p99 拖住；启动、通信、数据和 checkpoint 形成跨机架惊群。因此不能只看峰值 MFU，要看 goodput。设计上我会做 health check 和故障隔离、rank 级 straggler 诊断、拓扑感知并行、分层启动、分布式异步 checkpoint，以及 first-failure 可观测和自动恢复。

### 3 分钟版结构

1. 先讲三个规模质变；
2. 按“故障、长尾、拓扑、控制面、数据/checkpoint、可观测”各讲一个机制和一个动作；
3. 用 goodput 收口，而不是只报 MFU；
4. 最后限定个人边界：1.2 万/1.4 万卡是项目所在集群总规模，X1 3K 模型侧稳定训练是直接证据；整套万卡平台 ownership 不属于个人。

### 高频追问

1. 为什么同步训练看 `max(rank time)`，不是平均？
2. NCCL/XCCL kernel 很长，如何判断是网络根因还是等待慢 rank？
3. TP/PP/CP/EP/DP 如何映射到节点、rail、机架和 pod？
4. fixed-world replacement 与 elastic recovery 怎么选？
5. checkpoint 间隔如何结合 MTBF、保存成本与回滚损失？
6. 如何防止启动和恢复时的 metadata/storage storm？
7. 峰值 MFU、effective training time 与 goodput 有什么区别？
8. 你在 X1 3K 中亲自处理过哪个规模问题，证据是什么？

### 常见错误回答

- 只说“卡多了容易坏、通信慢”，没有 scale mechanism；
- 只看平均 MFU/GPU utilization，不看 p99、恢复和有效训练时间；
- 把 collective timeout 的最后报错 rank 当根因；
- 认为加大 timeout、加 barrier 或多打日志就能解决 hang；
- 把所有高频并行轴都机械映射到同一拓扑层级；
- checkpoint 只保存 model weights，恢复后只验证能启动；
- 把行业论文中的 10K/16K 数字或平台团队能力包装成个人经历。

---

## 12. 相邻系统关系与资料

- [MegaScale](../tech_reports/megascale.md)：万卡训练的效率、故障诊断、straggler 与恢复生产经验；
- [The Llama 3 Herd of Models](../tech_reports/llama3.md)：16K H100 训练中断与长期训练效率案例入口；
- [NCCL 与分布式通信算子](nccl.md#collective-map)：collective 语义、5D 映射和 hang 排障；
- [Checkpointing](checkpointing.md)：分布式状态、异步保存、原子提交和恢复验证；
- [Megatron 5D 并行](distributed_training.md)：逻辑并行组与 world-size；
- [MoE 与 Parallel Folding](moe.md#parallel-folding)：Attention/Expert 双逻辑网格与 EP dataflow；
- [Slow Step Debug](../playbooks/slow_step_debug.md)：慢 step 的执行型排障入口。

主要事实来源：

- [MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs](https://arxiv.org/abs/2402.15627)：论文报告 12,288 GPUs 上 175B 训练达到 55.2% MFU；其万卡生产作业数周内经历超过 100 次重启，并将故障、straggler、深度可观测和恢复视为生产稳定性的核心问题。
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)：论文报告 405B 训练最多使用 16K H100，并记录 54 天窗口内 466 次训练中断，其中 419 次为非计划中断。

以上公开数字用于说明规模化规律，不是 X1 项目的个人指标。

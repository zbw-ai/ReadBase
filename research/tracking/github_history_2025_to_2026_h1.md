# GitHub 历史补扫：2025 全年—2026 上半年

> 2026-09-22 历史复盘批次；2026-10-08 恢复任务；完整事件索引重建尚未完成。材料窗口为北京时间 **2025-01-01 00:00:00—2026-06-30 23:59:59**，是过去完整窗口，不是把今天的 frontier 游标填到日末。已保存的 24 项 PR 定向审阅结论来自 9 月 22 日；保留证据与尚未完成的覆盖项见 [manifest](audits/2026-09-22-history/github_manifest.json)。本页不推进 scan_log，不改当时 Accepted 数、Decision 或用户阅读状态。

从 [2025 季度 / 2026 月度入口](monthly_reviews.md)读主线；本页用于追问“这些变化究竟何时进入了哪个实现”。后续窗口接 [2026 年 7–9 月 GitHub 复盘](github_retrospective_2026-07_to_2026-09.md)。

## 覆盖方法与证据边界

计划对 15 个核心仓库分别枚举 release、固定默认分支 head 可达的历史 commit、已合并 PR。以下是重建方法，不是本次完成声明。commit 以 committer 时间过滤，PR 以 merged_at 过滤（包含非默认目标分支），release 以 published_at 过滤；按北京时间归到 2025 Q1–Q4 与 2026 1–6 月。PR 按 updated_at 倒序分页越过下界，再筛 merged_at，避免只检查 latest-50/top-80。另 8 个基础栈仓库计划只补 release，不能写成同等深度的代码补扫。

**索引枚举不等于逐条代码审计。** GitHub 的实时分页不是原子快照；后续更新、rebase、删除或私有历史可能改变可见结果。commit 时间不是首次公开或首次合入时间；目标为 release 分支的 PR 不能自动宣称已进入 main；无 release 不代表无代码发布。论文、博客仍为定向核验，不声称覆盖全网历史。

本次定向阅读 **24 个 PR：21 Read、3 Observe**，只读了与判断有关的 diff，未运行上游 GPU 测试。slime #2 已在旧资料中出现，属于补证而非首次发现；其余“未找到旧引用”也不证明过去从未见过。`Read` 是现在的阅读建议，不是生产验收。24 项的标题、作者、合并时间、SHA、分支和完整改动文件列表保存在 [reviewed_prs.json](audits/2026-09-22-history/reviewed_prs.json)。

## 本次已完成与尚未完成

| 层次 | 当前状态 | 可检查产物 |
|---|---|---|
| 季度/月度内容与版本核验 | 已保存 | 2025 四份季度报告、2026 H1 六份月报及三份来源 JSON |
| 历史 PR 定向审阅 | 24 项已保存；21 Read、3 Observe | [标题、作者、merge SHA、文件列表与判断](audits/2026-09-22-history/reviewed_prs.json) |
| 基础栈 release 精选 | 7 项历史摘要保留；不是端点全量索引 | [精选 release 证据](audits/2026-09-22-history/selected_release_evidence.json) |
| 15 核心库 commit / merged PR / release 全窗口索引 | **待重建**：旧临时缓存已失，10/8 API 多次连接失败 | [逐库待补状态](audits/2026-09-22-history/github_manifest.json) |
| 8 基础栈 release 全窗口索引 | **待重建**，不使用失效缓存里的旧计数作完成证明 | 同一 manifest；未生成虚假的 CSV |

核心库范围：AReaL、verl、slime、ROLL、OpenRLHF、NeMo RL、Megatron-LM、vLLM、SGLang、TRL、Transformers、Accelerate、PEFT、Kernels、Tokenizers。基础栈范围：Transformer Engine、NCCL、FlashAttention、DeepSpeed、PyTorch、DeepEP、FlashMLA、DeepGEMM。

10/8 已成功 fetch 本仓库远端，但公开 GitHub API 查询主要出现 TCP/TLS 超时、EOF、连接重置；少数仓库元数据成功不等于事件覆盖完成。自动审批对批量脚本两次超时，这不是认定读取行为不安全。保留[重建脚本](audits/2026-09-22-history/rebuild_github_indexes.py)便于后续继续；脚本需要 gh 的 API 读取权限和正常网络，只读远端并写本审计目录，当前未成功跑完整窗口。


## 按时间线应当吸收的实现变化

以下均是 **Historical Review**。Source type 为官方 GitHub merged PR；Source ID、合并时间与作者列在条目下。First seen 应解释为本轮证据登记 **2026-09-22**，不是原始发现日；slime #2 是已知材料的复核。Status：NEW（不覆盖已有个人状态）。Read 项 Impact 为高，Observe 项 Impact 为中。每项给出机制、Reason 与迁移边界；共同 Next 是先按固定 merge SHA 检查对应路径，再做所列最小对照，不能直接移植当前 HEAD。

### [2025-Q1 阅读主线](quarterly_signal_2025-Q1.md)

- **[OpenRLHF/OpenRLHF #704](https://github.com/OpenRLHF/OpenRLHF/pull/704) · Read · `weight sync`**：Ray collective 提供另一条 vLLM 权重同步路径，配置和进程组初始化是实际工程问题。
  - Source ID：`github:OpenRLHF/OpenRLHF#704`；作者：HollowMan6；合并：`2025-02-04T01:01:51Z`（UTC）；目标分支：`main`。
  - Reason / 工程维度：训练与推理通信域。AReaL 迁移 / Next：能借鉴同步域封装；需要对应 NCCL/RCCL、Ray 与 CuPy 条件，不能据此保证任意集群无 hang。
- **[OpenRLHF/OpenRLHF #891](https://github.com/OpenRLHF/OpenRLHF/pull/891) · Read · `checkpoint/recovery`**：接入 DeepSpeed universal checkpoint 转换，支持在受支持的配置变化后加载。
  - Source ID：`github:OpenRLHF/OpenRLHF#891`；作者：HollowMan6；合并：`2025-03-20T01:55:13Z`（UTC）；目标分支：`main`。
  - Reason / 工程维度：拓扑变化后的恢复。AReaL 迁移 / Next：区分checkpoint格式可转换和数据/optimizer/随机状态连续性，不能宣称任意并行布局通用。

### [2025-Q2 阅读主线](quarterly_signal_2025-Q2.md)

- **[OpenRLHF/OpenRLHF #1015](https://github.com/OpenRLHF/OpenRLHF/pull/1015) · Read · `rollout / scheduler`**：新增异步 trainer、generation actor 和信号协调，使 agent 执行成为训练链路的一部分。
  - Source ID：`github:OpenRLHF/OpenRLHF#1015`；作者：xiaoxigua999；合并：`2025-05-18T02:07:01Z`（UTC）；目标分支：`main`。
  - Reason / 工程维度：异步运行路径进入代码。AReaL 迁移 / Next：看SignalActor与暂停/更新边界；代码可运行入口不证明大规模稳定性。
- **[THUDM/slime #2](https://github.com/THUDM/slime/pull/2) · Read · `rollout / data path`**：在获得足够样本后中断其余请求，保留中断或超额样本，并维护状态与buffer。
  - Source ID：`github:THUDM/slime#2`；作者：guapisolo；合并：`2025-06-30T02:14:33Z`（UTC）；目标分支：`main`。
  - Reason / 工程维度：partial rollout 的早期实现。AReaL 迁移 / Next：迁移前检查跨版本续采样、sample身份和过滤语义；当时PR自称first part，不写成完整端到端解法。

### [2025-Q3 阅读主线](quarterly_signal_2025-Q3.md)

- **[alibaba/ROLL #111](https://github.com/alibaba/ROLL/pull/111) · Read · `rollout / scheduler`**：Agentic RL 重构包含group队列、异步训练路径与abort并发处理，代码显式组织环境和样本批次。
  - Source ID：`github:alibaba/ROLL#111`；作者：PanAndy；合并：`2025-07-31T03:31:20Z`（UTC）；目标分支：`main`。
  - Reason / 工程维度：group队列与异步环境。AReaL 迁移 / Next：复核GroupQueue的容量、等待和退出协议；大批量同步PR不意味着所有列出功能均经本仓库验证。
- **[THUDM/slime #258](https://github.com/THUDM/slime/pull/258) · Observe · `rollout`**：有fully_async示例文件，但PR正文集中于autonomy导入故障，标题不能单独证明系统能力。
  - Source ID：`github:THUDM/slime#258`；作者：zhuzilin；合并：`2025-09-04T02:57:50Z`（UTC）；目标分支：`main`。
  - Reason / 工程维度：示例证据与成熟度分开。AReaL 迁移 / Next：保留示例级观察，等待端到端行为和版本证据，不据标题升级成熟框架。
- **[NVIDIA-NeMo/RL #1035](https://github.com/NVIDIA-NeMo/RL/pull/1035) · Read · `weight sync / data path`**：更新generation_weight_version并在恢复轨迹采集前发布正确版本，避免新轨迹带旧标记。
  - Source ID：`github:NVIDIA-NeMo/RL#1035`；作者：RahulSChand；合并：`2025-09-03T21:18:29Z`（UTC）；目标分支：`faster-strictfifo`。
  - Reason / 工程维度：refit之后的版本发布。AReaL 迁移 / Next：AReaL对照version发布与准入顺序；补丁是具体时序修复，不是全系统事务证明。

### [2025-Q4 阅读主线](quarterly_signal_2025-Q4.md)

- **[THUDM/slime #906](https://github.com/THUDM/slime/pull/906) · Observe · `training / inference backend`**：尝试在FSDP2训练中复用推理MoE路径，但作者明确写着without the correct backward。
  - Source ID：`github:THUDM/slime#906`；作者：zhuzilin；合并：`2025-11-25T12:33:09Z`（UTC）；目标分支：`main`。
  - Reason / 工程维度：true-on-policy的证据限度。AReaL 迁移 / Next：只能作为局部对齐尝试，不能写完整训练parity；还要对照2026年FSDP路径移除。
- **[areal-project/AReaL #667](https://github.com/areal-project/AReaL/pull/667) · Read · `training`**：vocab-parallel logprob/entropy提供自定义autograd，减少保存的中间状态并带数值/梯度测试。
  - Source ID：`github:areal-project/AReaL#667`；作者：rchardx；合并：`2025-12-04T06:40:22Z`（UTC）；目标分支：`main`。
  - Reason / 工程维度：logprob也是分布式算子。AReaL 迁移 / Next：检查vocab分片、全局normalizer与backward，而不只比forward输出或显存。
- **[NVIDIA-NeMo/RL #1627](https://github.com/NVIDIA-NeMo/RL/pull/1627) · Read · `data/trajectory path`**：Gym rollout按完成时间返回时，用显式row index恢复输入顺序。
  - Source ID：`github:NVIDIA-NeMo/RL#1627`；作者：yfw；合并：`2025-12-14T02:25:31Z`（UTC）；目标分支：`main`。
  - Reason / 工程维度：完成顺序不能替代样本身份。AReaL 迁移 / Next：AReaL的数据重排与packing应保留身份；避免将reward、prompt或trajectory错配。

### [2026-01 阅读主线](monthly_signal_2026-01.md)

- **[OpenRLHF/OpenRLHF #1152](https://github.com/OpenRLHF/OpenRLHF/pull/1152) · Read · `rollout / scheduler`**：重组streaming async sampling与队列控制，移除重复的独立async实现文件。
  - Source ID：`github:OpenRLHF/OpenRLHF#1152`；作者：Freder-chen；合并：`2026-01-04T23:33:56Z`（UTC）；目标分支：`main`。
  - Reason / 工程维度：streaming采样结构演进。AReaL 迁移 / Next：先核对该revision的新入口与队列参数；不能根据旧路径不存在断言功能被删除。
- **[areal-project/AReaL #804](https://github.com/areal-project/AReaL/pull/804) · Read · `training / data path`**：把共享前缀序列组织成trie并用FlexAttention树mask，在训练路径复用前缀计算。
  - Source ID：`github:areal-project/AReaL#804`；作者：nuzant；合并：`2026-01-07T07:17:27Z`（UTC）；目标分支：`main`。
  - Reason / 工程维度：共享prefix进入训练。AReaL 迁移 / Next：区分完整树mask语义、packing和普通拼接；需要共享率与梯度对照才能判断收益。

### [2026-02 阅读主线](monthly_signal_2026-02.md)

- **[areal-project/AReaL #926](https://github.com/areal-project/AReaL/pull/926) · Read · `checkpoint/recovery`**：Archon将pinned-CPU staging与后台DCP写入/合并分开，并处理后台collective的进程组归属。
  - Source ID：`github:areal-project/AReaL#926`；作者：rchardx；合并：`2026-02-15T01:15:18Z`（UTC）；目标分支：`main`。
  - Reason / 工程维度：异步save有多个完成阶段。AReaL 迁移 / Next：量出staging暂停、后台写与最终可恢复时间；不把async API返回当作checkpoint已经持久化。
- **[THUDM/slime #1624](https://github.com/THUDM/slime/pull/1624) · Read · `weight sync / inference backend`**：导出HF权重后按quantization_config处理，再形成同步chunk，补上表示转换。
  - Source ID：`github:THUDM/slime#1624`；作者：GeLee-Q；合并：`2026-02-25T06:50:36Z`（UTC）；目标分支：`main`。
  - Reason / 工程维度：Megatron bridge携带量化语义。AReaL 迁移 / Next：验收不只比tensor名字和shape，还要检查scale、dtype、排列与对应backend loader。

### [2026-03 阅读主线](monthly_signal_2026-03.md)

- **[THUDM/slime #1664](https://github.com/THUDM/slime/pull/1664) · Read · `training / inference backend`**：该revision删除FSDP实现和相应示例；2025年存在某backend不代表未来仍受维护。
  - Source ID：`github:THUDM/slime#1664`；作者：zhuzilin；合并：`2026-03-04T04:08:33Z`（UTC）；目标分支：`main`。
  - Reason / 工程维度：框架支持范围会收缩。AReaL 迁移 / Next：选型固定commit/tag并检查迁移成本；不能把这一项目的取舍推成FSDP本身无价值。
- **[areal-project/AReaL #990](https://github.com/areal-project/AReaL/pull/990) · Observe · `training`**：PPO actor/critic日志的token计数改为全batch metadata，避免CP局部分片造成口径偏差。
  - Source ID：`github:areal-project/AReaL#990`；作者：yash27-lab；合并：`2026-03-06T03:23:14Z`（UTC）；目标分支：`main`。
  - Reason / 工程维度：统计日志不等于训练目标。AReaL 迁移 / Next：作者只验证CPU日志/相邻测试；保留为测量参考，不宣称此PR改变了loss或完成GPU正确性证明。

### [2026-04 阅读主线](monthly_signal_2026-04.md)

- **[NVIDIA/Megatron-LM #4047](https://github.com/NVIDIA/Megatron-LM/pull/4047) · Read · `training / scheduler`**：PP source buffer在异步send完成前不能释放，否则下阶段可能读到已被复用的内存。
  - Source ID：`github:NVIDIA/Megatron-LM#4047`；作者：ZhiyuLi-Nvidia；合并：`2026-04-16T23:35:25Z`（UTC）；目标分支：`main`。
  - Reason / 工程维度：异步P2P的buffer生命周期。AReaL 迁移 / Next：迁移send完成fence→释放的所有权规则；不要为修复一条路径粗暴关闭全部overlap。
- **[verl-project/verl #6091](https://github.com/verl-project/verl/pull/6091) · Read · `weight sync`**：允许NCCL/NIXL传输bucket小于最大权重，把大tensor分块，并区分CUDA IPC直传路径。
  - Source ID：`github:verl-project/verl#6091`；作者：wuxibin89；合并：`2026-04-27T11:49:54Z`（UTC）；目标分支：`main`。
  - Reason / 工程维度：超大权重与buffer峰值。AReaL 迁移 / Next：同时记录最大tensor、bucket、聚合与接收staging；缩小bucket不等于整个权重常驻消失。

### [2026-05 阅读主线](monthly_signal_2026-05.md)

- **[areal-project/AReaL #1345](https://github.com/areal-project/AReaL/pull/1345) · Read · `checkpoint/recovery / scheduler`**：恢复较高model version时同步校正accepted计数，避免初始额度被错误放大并突发提交rollout。
  - Source ID：`github:areal-project/AReaL#1345`；作者：daihaowz；合并：`2026-05-20T09:04:36Z`（UTC）；目标分支：`main`。
  - Reason / 工程维度：恢复也必须恢复staleness预算。AReaL 迁移 / Next：验证恢复前后可接受样本容量一致；将控制计数器和模型版本一起纳入checkpoint契约。
- **[THUDM/slime #1806](https://github.com/THUDM/slime/pull/1806) · Read · `weight sync`**：比较权重字节，发送变化位置与新值，接收端覆盖；disk和NCCL共用线格式。
  - Source ID：`github:THUDM/slime#1806`；作者：nanjiangwill；合并：`2026-05-26T03:52:41Z`（UTC）；目标分支：`main`。
  - Reason / 工程维度：delta不是浮点增量累加。AReaL 迁移 / Next：依赖一致初始baseline，host snapshot/staging也有成本；来源声称无漂移不等于网络中断和半更新自动恢复。

### [2026-06 阅读主线](monthly_signal_2026-06.md)

- **[NVIDIA-NeMo/RL #2651](https://github.com/NVIDIA-NeMo/RL/pull/2651) · Read · `checkpoint/recovery / data path`**：replay buffer进入checkpoint；target在训练消费完整batch后推进，reservation等所有worker结束再释放。
  - Source ID：`github:NVIDIA-NeMo/RL#2651`；作者：macandro96；合并：`2026-06-09T23:15:46Z`（UTC）；目标分支：`main`。
  - Reason / 工程维度：消费才推进完成边界。AReaL 迁移 / Next：这是8月trained-frontier更完整设计之前的一步，不能写成6月已解决所有resume丢样本。
- **[NVIDIA/Megatron-LM #5047](https://github.com/NVIDIA/Megatron-LM/pull/5047) · Read · `training`**：在per-token-loss路径校正aux_loss/z_loss的TP/CP相关缩放，保持梯度贡献口径。
  - Source ID：`github:NVIDIA/Megatron-LM#5047`；作者：deepakn94；合并：`2026-06-03T00:42:40Z`（UTC）；目标分支：`main`。
  - Reason / 工程维度：MoE loss缩放必须跨并行度一致。AReaL 迁移 / Next：固定有效token与batch做不同TP/CP对照；不把所有配置都判为受影响。
- **[areal-project/AReaL #1393](https://github.com/areal-project/AReaL/pull/1393) · Read · `training / scheduler`**：共置offload时可释放可重算的grad buffers，恢复时重新分配和清零，减少host备份负担。
  - Source ID：`github:areal-project/AReaL#1393`；作者：HT-Yuan；合并：`2026-06-23T06:13:52Z`（UTC）；目标分支：`main`。
  - Reason / 工程维度：临时梯度不必备份到host。AReaL 迁移 / Next：区分可重算临时状态和跨step持久状态；默认开关及兼容性按固定版本核对，不外推整作业内存倍数。
- **[THUDM/slime #2143](https://github.com/THUDM/slime/pull/2143) · Read · `weight sync / scheduler`**：相同model_path的并发更新共享future，不同目标冲突返回错误，已加载版本可跳过。
  - Source ID：`github:THUDM/slime#2143`；作者：zhuzilin；合并：`2026-06-29T04:03:28Z`（UTC）；目标分支：`main`。
  - Reason / 工程维度：并发更新请求要有明确归并语义。AReaL 迁移 / Next：AReaL可借鉴幂等与冲突拒绝；路径相同是否内容不可变仍需部署约束，不等于全局事务。

## 基础栈：报告出现与代码可用是两个时间点

下列为版本实现背景，均为官方 release notes；本轮记录为 **Observed / Observe、Impact 中、Status NEW**。Reason 是补充季度/月度材料的实现边界；Next 是只在对应硬件与 workload 上锁版本验证。它们不计入上面的 24 个 PR，也不把 release 中所有改动都提升为 Accepted。

| 版本与原始时间（UTC） | 一句话价值与限制 | 相关主题 / 建议行动 |
|---|---|---|
| [Transformer Engine v2.0，2025-02-13](https://github.com/NVIDIA/TransformerEngine/releases/tag/v2.0) | MXFP8 cast/GEMM 与 FSDP2 路径进入版本；当时 Userbuffers overlap 的 MXFP8 限制需单独检查。 | [FP8](../../01-systems/topics/fp8.md)：按格式、并行与 overlap 组合验收，不能读成 NVFP4 已成熟。 |
| [Transformer Engine v2.8，2025-10-07](https://github.com/NVIDIA/TransformerEngine/releases/tag/v2.8) | NVFP4 recipe 从 Q3 报告进入训练库；支持格式不等于所有模块或模型都有相同收益。 | [Transformer Engine](../../01-systems/topics/transformer_engine.md)：检查 recipe、cast 与同步开销。 |
| [NCCL v2.28.7-1，2025-10-18](https://github.com/NVIDIA/nccl/releases/tag/v2.28.7-1) | GIN device API 与 communicator revoke 扩展设备通信和恢复控制，但 API 与硬件条件仍有限制。 | [NCCL](../../01-systems/topics/nccl.md)：分别验证数据路径和故障退出；后续 [v2.28.9-1](https://github.com/NVIDIA/nccl/releases/tag/v2.28.9-1) 的 ordering 修复也要检查。 |
| [Transformer Engine v2.10，2025-12-11](https://github.com/NVIDIA/TransformerEngine/releases/tag/v2.10) | NVFP4 GroupedLinear 与 graph 路径继续落地，graph 与进程组销毁顺序也成为正确性条件。 | [MoE](../../02-training-infra/topics/moe.md)：检查 expert GEMM 与 graph 生命周期，不只看峰值吞吐。 |
| [DeepSpeed v0.18.9，2026-03-30](https://github.com/deepspeedai/DeepSpeed/releases/tag/v0.18.9) | AutoSP 已有代码版本，早于后续论文阅读时间；并行配置与 checkpoint 转换要一起看。 | [Distributed Training](../../02-training-infra/topics/distributed_training.md)：区分首次实现、论文与本仓库发现日。 |
| [Transformer Engine v2.14.1，2026-04-24](https://github.com/NVIDIA/TransformerEngine/releases/tag/v2.14.1) | MXFP8 quantize + dbias fusion 的非确定性错误说明 fused 路径也需要数值回归。 | [FP8](../../01-systems/topics/fp8.md)：相同输入重复运行并对照未融合基线。 |
| [PyTorch v2.12.1，2026-06-18](https://github.com/pytorch/pytorch/releases/tag/v2.12.1) | B200 FLASH_ATTN batch invariance 修复针对特定后端/硬件，不代表所有训推误差消失。 | [FlashAttention](../../01-systems/topics/flashattention.md)：按 backend、batch shape 与 dtype 构造对照。 |

Source ID 使用 `github-release:<repo>@<tag>`，原始日期、仓库与 tag 见 [精选 release 证据](audits/2026-09-22-history/selected_release_evidence.json)；以上背景源于 9 月 22 日定向审阅，10 月 8 日全量元数据重建仍待完成。DeepGEMM 的 `nv_dev_*` tag 不能自动解释为稳定版；FlashAttention 的空 release body 不足以推断新增机制；FlashMLA 无 release 记录时仍应查历史代码与报告。

## 这轮回看改变的判断

**趋势推断：** 2025 上半年把异步 rollout、权重交付与环境接口变成可运行路径；下半年补数值一致性与数据身份；到 2026 上半年，checkpoint、消费位置、版本计数和内存生命周期开始明确进入实现。但这不是所有框架同步变成熟的线性过程：backend 会被移除，示例可能缺关键梯度，指标修复也不能误写成训练算法修复。

最值得保留的筛选规则是：遇到 `async`、`on-policy`、`checkpoint`、`FP4` 等名称，继续追问实际保留了哪些状态、在哪个版本可用、缺什么失败路径证据。可执行的后续验证见 [RL 状态实验](../../practice/experiments/rl_state_boundaries.md)，历史脉络已同步到 [Agentic RL 主题](../../04-rl-infra/topics/agentic_rl.md#history-2025-h1-2026)。

论文/官方文章的原始引用与核验边界分别见 [2025 H1](audits/2026-09-22-history/2025-h1-sources.json)、[2025 H2](audits/2026-09-22-history/2025-h2-sources.json)、[2026 H1](audits/2026-09-22-history/2026-h1-sources.json)。这些核验不替代实际阅读或实验；旧 scan 原文、历史 Accepted 数、学习状态与 frontier cursor 均保留。

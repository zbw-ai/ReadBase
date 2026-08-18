# Frontier Scan, 2026-08-14

- Previous scan：[2026-08-12](frontier_scan_2026-08-12.md)
- Window：2026-08-12 09:51 ~ 2026-08-14 17:39:50
- Timezone：Asia/Shanghai
- Generated at：2026-08-14 17:39:50
- Report type：flexible frontier scan
- Sources scanned：arXiv cs.DC / cs.LG / cs.AI / cs.CL / cs.AR / cs.PF / cs.NI；OpenAI / Anthropic / NVIDIA / DeepSeek official sources；Hugging Face Blog / TRL / Transformers / Accelerate / PEFT / Kernels；AReaL / verl / slime / ROLL / OpenRLHF / NeMo RL default-branch changes、major PR 与 release
- Scan completeness：扫描六个主分类 API 返回的 553 条候选，并补查 cs.AR recent 的 48 条记录；完成四家核心厂商、Hugging Face 生态和六个 RL 框架的逐项检查。不同 arXiv 分类存在 cross-list 重复，不把两组数量相加冒充 unique papers。GitHub API 不稳定时回退到官方 Atom、commit patch 和 PR 页面。两篇 8 月 11 日投稿、在上一游标之后才进入公告面的论文标记为 `boundary late-discovered`。

## 本次核心判断

本窗口保留七条信号，不按数量凑榜单。它们共同指向一件事：**Agentic RL 的优化对象正在从单个 kernel 或单个训练 step，转向跨 rollout、training、KV cache、weight sync 和 recovery 的状态调度。**

1. **Readiness 正在成为 RL scheduler 的一等信号。** TideRL 根据 ready backlog 调整 batching、Ref/Actor 执行方式和训推资源；MISA-T 则在 inference admission 层控制不同 rollout workload 对 KV cache 的竞争。
2. **MoE RL 的负载平衡不能再拆成 attention packing 与 expert routing 两个孤立问题。** RoutePack 利用 rollout 已知的 routing demand，在 optimizer-step window 内联合规划。
3. **开源框架开始补齐真正影响大规模运行的基础能力。** AReaL 补 grouped colocation，verl 将 weight sync 做成 node-local multi-sender，NeMo RL 开始保存异步 rollout controller 的可恢复状态。
4. **Inference Infra 仍是 RL Infra 的组成部分。** vToken 说明 KV eviction 只有算法还不够，allocator 和 attention kernel 之间的粒度错配同样会吞掉收益。

## Accepted Frontier Signals

### TideRL: Boosting Agentic RL Goodput with Readiness-Aware Scheduling

- Signal ID：2026-08-14-001
- Source ID：arxiv:2608.10402
- First seen：2026-08-14 17:39:50（boundary late-discovered）
- Scan window：2026-08-12 09:51 ~ 2026-08-14 17:39:50
- Focus Match：P0 Focus
- 来源：arXiv primary source
- 类型：paper / Agentic RL scheduler / rollout-training elasticity
- 链接：https://arxiv.org/abs/2608.10402
- 发布时间：2026-08-11
- Primary-source check：title / eleven authors / submission date / CTB / RA2P / ERS / headline results 已对齐 arXiv metadata、摘要与 HTML 正文
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：它没有只把 synchronous RL 改成 asynchronous RL，而是让 scheduler 根据“现在有多少可训练样本”同时决定 rollout batching、Ref/Actor pipeline 和 GPU 角色分配。
- Status：NEW
- 建议动作：进入下一轮 P0 候选；重点读 readiness metric、rank migration boundary、KV state preservation 和 convergence comparison
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [Long-context Training](../topics/long_context_training.md), [Rollout Latency](../playbooks/rollout_latency.md)

TideRL 包含三个互相咬合的机制：`Continuous Task Batching` 在多轮 task pause/resume 之间保留 KV 状态；`Resource-Aware Ref-Actor Pipelining` 根据 ready backlog 和 arrival interval，在 decoupled streaming 与 colocated aggregation 之间选择；`Elastic Resource Scaling` 用相同 readiness 信号在 rollout 与 training 之间移动 rank。

作者在 text-only 和 multimodal agentic workload 上报告，相对同步 baseline 最高 `5.6x` goodput、相对异步 baseline 超过 `33%`，KV cache hit rate 提升 `1.58x`，per-step training time 最多下降 `44.3%`，总等待时间最多下降 `77.6%`。这些是作者在最多 32 张 H100 上的结果；真正需要核对的是 ERS 的迁移成本是否被自然的 weight-sync/cache-flush boundary 吸收，以及收益对 rollout 长尾分布是否敏感。

### Scheduling Mixed RL Rollouts Beyond Prefix Locality (MISA-T)

- Signal ID：2026-08-14-002
- Source ID：arxiv:2608.11152
- First seen：2026-08-14 17:39:50（boundary late-discovered）
- Scan window：2026-08-12 09:51 ~ 2026-08-14 17:39:50
- Focus Match：P0 Focus
- 来源：arXiv primary source
- 类型：paper / mixed rollout serving / KV-cache admission
- 链接：https://arxiv.org/abs/2608.11152
- 发布时间：2026-08-11
- Primary-source check：title / seven authors / submission date / admission mechanism / Step3.7 and Qwen3.6 numbers 已对齐 arXiv metadata 与摘要
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：它指出 prefix-aware routing 只能决定“发到哪里”，却不能控制 RLVR、RLHF 和 agentic sessions 如何争夺有限 KV capacity，也不能保证 trainer 指定的 workload mixture。
- Status：NEW
- 建议动作：与 vLLM Router、AReaL rollout admission 一起读；重点验证 workload cap、residency accounting 和 weight-sync rewarm
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [Long-context Training](../topics/long_context_training.md), [Rollout Latency](../playbooks/rollout_latency.md)

MISA-T 位于现有 cache-aware router 之前，增加 adaptive session admission、workload-aware KV-capacity allocation 和 residency-time-aware accounting。它不是简单把请求发给 prefix hit 最大的 worker，而是限制不同 workload 占用多少 KV、存活多久，并尽量保持 trainer 要求的数据比例。

作者报告，相对 sweep-tuned cache-aware vLLM Router，Step3.7 与 Qwen3.6-35B-A3B rollout throughput 分别提升 `53.3%` 和 `43.6%`；50-iteration Step3.7 实验中 rollout throughput 提升 `35.6%`，mean iteration time 下降 `22.8%`，同时 workload mixture 和 task score 保持可比。对 AReaL 的直接问题是：当前 scheduler 是否只看 engine load/prefix locality，而没有把 session residence time 与训练采样配额纳入 admission。

### RoutePack: Expert Placement and Attention-Aware Data Packing for MoE RL

- Signal ID：2026-08-14-003
- Source ID：arxiv:2608.12146
- First seen：2026-08-14 17:39:50
- Scan window：2026-08-12 09:51 ~ 2026-08-14 17:39:50
- Focus Match：P0 Focus
- 来源：arXiv primary source
- 类型：paper / MoE RL / expert placement / data packing
- 链接：https://arxiv.org/abs/2608.12146
- 发布时间：2026-08-12
- Primary-source check：title / five authors / date / hierarchical planner / Ling-3.0 results 已对齐 arXiv metadata 与摘要
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：MoE RL 的 step time 同时由 dense attention token composition 和 sparse expert peak 决定；单独优化 packing 或 routing 可能只是把瓶颈推给另一侧。
- Status：NEW
- 建议动作：读 optimizer-step window planner、routing replay、EDP shard objective；评估是否可复用 AReaL 已有 sample packing 与 route metadata
- 关联主题：[MoE](../topics/moe.md), [Agentic RL](../topics/agentic_rl.md), [Distributed Training](../topics/distributed_training.md)

RoutePack 利用 rollout 阶段已经产生的 sequence length 和逐层 expert demand，先做 layer-wise expert placement/rerouting，再在整个 optimizer-step window 内联合优化 DP packing。它保持 logical top-k routing 与既有 MoE kernel，不依赖 microbatch-level expert replication。

作者在 Ling-3.0-Tiny 与 Ling-3.0-Flash 上报告：expert rerouting 单独带来 `3.80%` / `10.50%`，routing-aware packing 再增加 `4.86%` / `3.98%`，整体 throughput 提升 `8.85%` / `14.89%`。这组数字不算夸张，反而更像可落地优化；关键是 planner 成本、routing demand 在 policy update 后的漂移，以及执行窗口是否与异步 RL 的 freshness boundary 冲突。

### AReaL Adds Grouped Colocation to the Ray Scheduler

- Signal ID：2026-08-14-004
- Source ID：github:areal-project/AReaL#1575
- First seen：2026-08-14 17:39:50
- Scan window：2026-08-12 09:51 ~ 2026-08-14 17:39:50
- Focus Match：P0 Focus
- 来源：AReaL merged PR / default branch
- 类型：framework implementation / placement / colocation
- 链接：https://github.com/areal-project/AReaL/pull/1575
- 发布时间：2026-08-13
- Primary-source check：merged state / placement constraints / grouped-worker example / tests 已对齐官方 PR 与 patch
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：它解除 colocated roles 必须 replica 数完全相等的限制，使少量 multi-GPU inference worker 可以复用大量 single-GPU trainer rank 的同一批物理 GPU。
- Status：NEW
- 建议动作：代码级阅读 Ray scheduler 与 physical GPU mapping；区分 grouped colocation primitive 和 TideRL/BiDiRL 式动态资源迁移
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [Distributed Training](../topics/distributed_training.md)

典型配置可以是 `16 x 4-GPU SGLang workers` 与 `64 x 1-GPU actor workers` colocate。实现要求两侧总 GPU demand 完全相等，每个 inference group 的物理 GPU 在同一节点内连续且不跨节点，并通过 NodeAffinity 和显式 GPU IDs 启动 zero-GPU launcher。

这条改动的价值是把 heterogeneous role shape 变成可表达的 placement，而不是自动解决动态借卡。它为 AWEX、训推 colocate 和未来 role switching 提供了更稳的资源映射基础；若要迁移 TideRL 的 readiness-aware scaling，仍需要额外的 state handoff、lifecycle 和 scheduler policy。

### verl Adds Node-Local Multi-Sender NCCL Weight Broadcast

- Signal ID：2026-08-14-005
- Source ID：github:verl-project/verl#7291
- First seen：2026-08-14 17:39:50
- Scan window：2026-08-12 09:51 ~ 2026-08-14 17:39:50
- Focus Match：P0 Focus
- 来源：verl merged PR / default branch
- 类型：framework implementation / weight sync / NCCL topology
- 链接：https://github.com/verl-project/verl/pull/7291
- 发布时间：2026-08-14
- Primary-source check：broadcast topology / opt-in flag / benchmark models and timings 已对齐官方 PR 与 commit message
- 影响等级：★★★★★
- Decision：Deep Dive
- Reason：它没有做参数分片，而是让 rank 0 的 NVLink-local peers 充当跨节点 relay，使多个 NIC 并行发送同一份 model bucket，直接针对 colocated RL 的 weight-sync wall time。
- Status：NEW
- 建议动作：对照 AReaL weight update path，核对 topology discovery、bucket scheduling、failure semantics 和 full-param gather 剩余开销
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [NCCL](../topics/nccl.md), [Tensor Parallelism](../topics/tensor_parallelism.md)

rank 0 仍是唯一 source，但同节点 actor peers 进入 broadcast tree 作为 relay，各自可以驱动一张 NIC；remote actor workers 不作为 sender。每个 bucket 仍只执行一次 broadcast，因此不改变 parameter semantics，也不是把 tensor 切成多份发送。

项目报告 Qwen3.5-27B weight sync 从 `5.026s` 降至 `1.791s`（`2.81x`），Qwen3-30B-A3B 从 `5.570s` 降至 `1.892s`（`2.94x`）。在 8 trainer / 8 rollout、跨 4 节点设置中约为 `2.25x`；剩余主要成本是 trainer 侧 full-parameter gather。该功能为 opt-in，配置 `engine_kwargs={"nccl":{"multi_sender":true}}`。

### NeMo RL Checkpoints SingleController Asynchronous Rollout State

- Signal ID：2026-08-14-006
- Source ID：github:NVIDIA-NeMo/RL#3429
- First seen：2026-08-14 17:39:50
- Scan window：2026-08-12 09:51 ~ 2026-08-14 17:39:50
- Focus Match：P0 Focus
- 来源：NeMo RL merged PR / default branch
- 类型：framework implementation / async RL / checkpoint recovery
- 链接：https://github.com/NVIDIA-NeMo/RL/pull/3429
- 发布时间：2026-08-13
- Primary-source check：saved state / restore validation / capacity-change behavior / test coverage 已对齐官方 PR 与 patch
- 影响等级：★★★★★
- Decision：Read
- Reason：异步 RL 的 checkpoint 不能只保存 policy/optimizer；若 controller、ready trajectories、policy-version metadata 丢失，恢复后会改变采样语义并浪费长 rollout。
- Status：NEW
- 建议动作：与仓库 [Checkpointing](../topics/checkpointing.md) 章节对照，梳理 ready/in-flight trajectory 的 exactly-once 边界
- 关联主题：[Checkpointing](../topics/checkpointing.md), [Fault Tolerance](../topics/fault_tolerance.md), [Agentic RL](../topics/agentic_rl.md)

SingleController checkpoint 会保存 ready replay-buffer groups 及其 metadata，包括 start/end weight、target step、group ID 和 DataPlane field data；未完成的 in-flight rollout 被明确丢弃。恢复时校验 partition ID、group size、sample ID duplication，并在 capacity 变化时保留更新鲜的数据。

这里最重要的不是“能 serialize 一个 Python object”，而是故障恢复语义：什么状态已经对 trainer 可见、什么状态仍可重算、staleness 淘汰由谁负责。它值得沉淀到未来的 async RL checkpoint playbook。

### vToken: Token-Level Virtualization for Reclaimable KV Caches

- Signal ID：2026-08-14-007
- Source ID：arxiv:2608.13263
- First seen：2026-08-14 17:39:50
- Scan window：2026-08-12 09:51 ~ 2026-08-14 17:39:50
- Focus Match：P1 Focus
- 来源：arXiv primary source / vLLM implementation
- 类型：paper / inference infra / KV-cache memory management
- 链接：https://arxiv.org/abs/2608.13263
- 发布时间：2026-08-13
- Primary-source check：title / seven authors / date / token-table mechanism / vLLM implementation / results 已对齐 arXiv metadata 与摘要
- 影响等级：★★★★☆
- Decision：Read
- Reason：PagedAttention 解决 allocator-level fragmentation，但 token-level eviction 仍受 block-level physical layout 限制；这类粒度错配会直接限制长 horizon rollout concurrency。
- Status：NEW
- 建议动作：核对 async repacking cost、CUDA Graph stability、eviction policy integration 和 attention kernel indirection overhead
- 关联主题：[Long-context Training](../topics/long_context_training.md), [Agentic RL](../topics/agentic_rl.md), [Rollout Latency](../playbooks/rollout_latency.md)

vToken 用 token-table indirection 解耦 logical token liveness 与 physical block placement，并异步 repack live tokens。它保留现有 PagedAttention kernel 与 CUDA Graph compatibility，因此更像一层可复用的 KV virtualization，而不是重写 attention backend。

作者在 vLLM 上报告每请求 retained KV blocks 减少 `27.2%-72.3%`，SLA-constrained throughput 最高 `1.37x`，受限 active-KV budget 下 concurrency 最高 `2x`；接入一个 eviction policy 的代码从 500+ 行降到 50 行以内。对 RL rollout 的价值取决于 eviction/recompute 是否破坏长 agent session 的 tail latency，而不只是平均 memory saving。

## Observed / Rejected Candidates

| 材料 | Source ID | Focus Match | Decision | 原因 |
|---|---|---|---|---|
| TEMPO | arxiv:2608.13057 | P1 | Observe | 用 `max(a+bG, c+betaN)` 区分 EP memory-bound / compute-bound，并在 Qwen3-235B 报告 4%-6% throughput；但 DeepSeek-V3 通信主导时无收益，先保留为 EP scheduler 参考。 |
| OpScale | arxiv:2608.13499 | P1 | Observe | operator-level autoscaling 有 40 A100 与 24 GB200 traces，资源与吞吐数字扎实；当前更偏 serving control plane，等与 rollout elastic pool 形成直接关联再精读。 |
| CAKE | arxiv:2608.12629 | P1 | Observe | compiler-agent co-design 与 verifier-guided kernel generation 很强，但短期主要改变 kernel engineering workflow，不直接改变 RL runtime。 |
| Contract-Grade Verifier for GPU Kernels | arxiv:2608.12700 | P1 | Observe | 对 2638 个已接受 kernel 的审计发现大量 correctness violation，提醒 benchmark 必须先过语义门；适合与 CAKE 一起回看。 |
| RealisticTritonBench | arxiv:2608.12004 | P1 | Observe | 从真实 PR 提取 end-to-end kernel tasks，比孤立 micro-kernel benchmark 更可信；暂未给出直接可迁移训练系统机制。 |
| LazyTrain | arxiv:2608.11919 | P1 | Observe | layer streaming、checkpoint/recompute/offload 联合规划有约 1.24x sustained TFLOPS，但目标是 limited-hardware training，不挤占集群级主线。 |
| Total Recall at What Cost? | arxiv:2608.11879 | P1 | Observe | agent memory 的 serving cost benchmark 与长 horizon 相关，但暂时更偏评测体系。 |
| HBF Sucks! | arxiv:2608.11668 | P1 | Observe | 四条 production traces 与 H100/B200 profile 显示 HBF 直接替换 SSD 会让 latency 上升 2-5.5x、SLO goodput 下降 1.1-2.7x；这是 KV 分层不能只看设备峰值带宽的高质量反例，后续与 OasisKV 一起读。 |

## OpenAI / Anthropic / NVIDIA / DeepSeek Watch

| Vendor | Sources checked | Decision | 结果 |
|---|---|---|---|
| OpenAI | official research / engineering / releases | Not found | 本窗口未发现可核验、且包含 Training/RL/Inference Infra 机制的新一手材料。 |
| Anthropic | official research / engineering / newsroom | Not found | 本窗口未发现新的 RL Infra、训练系统或推理系统技术披露。 |
| NVIDIA | official technical sources / NeMo RL | **Accepted / Read** | NeMo RL SingleController checkpoint 进入 Accepted；另有 Nemotron Super Omni 120B-A12B GRPO recipe、MTP speculative decoding 与对应 refit correctness 修复，作为工业集成证据记录在 framework watch。 |
| DeepSeek | API changelog / official Hugging Face organization / GitHub | Not found | 未发现上一游标后的新技术报告、模型权重、API 变更或训练栈发布。 |

## Hugging Face Watch

| Sources checked | Decision | 结果 |
|---|---|---|
| Hugging Face Blog | Not found / routine only | 本窗口未发现足以改变 Agentic RL、distributed training、long context 或 inference backend 判断的 HF core-team 新文章。 |
| TRL / Transformers / Accelerate / PEFT / Kernels | Not found / routine only | 已检查可见 release、docs 和重要变更，未发现需要升级为独立 frontier signal 的新机制。 |
| Community / vendor-authored posts | Observe | 未将社区文章或托管在 Hugging Face 的厂商材料自动视为 HF 官方技术信号。 |

## RL Framework Watch

| Framework | Release / PR | 子系统 | 核心变化 | 证据 | 对 AReaL 的参考 | Decision |
|---|---|---|---|---|---|---|
| AReaL | [#1575](https://github.com/areal-project/AReaL/pull/1575) | scheduler / placement | grouped colocation 支持少量 multi-GPU inference workers 复用大量 single-GPU trainer ranks | merged PR、placement constraints、tests | 这是 AReaL 自身的重要 placement primitive；后续可在其上研究 readiness-aware role scaling | **Accepted / Deep Dive** |
| verl | [#7291](https://github.com/verl-project/verl/pull/7291) | weight sync | node-local actor peers 作为 NCCL relay，多 NIC 并行广播完整参数 bucket | merged PR、两组模型与多节点 benchmark | 检查 AReaL 当前 weight update 是否由单 sender/NIC 限制，以及 trainer full-gather 占比 | **Accepted / Deep Dive** |
| verl | [#6804](https://github.com/verl-project/verl/pull/6804) | rollout / multimodal correctness | Continuous Token path 统一 full history processor encoding，修复 legacy tool path 的 image-pad/boundary mismatch | merged PR、Qwen/GLM/Gemma VL tests | AReaL multimodal rollout 必须测试 canonical token path 与 full-history processor 一致性 | Read |
| slime | default branch fixes | rollout / training correctness | 修复 fully-async rollout 丢 completed groups、DP+CP advantage whitening 与 FP8 conversion | official commit feed / patches | 将 async sample accounting、CP-aware whitening 和 low-precision conversion 加入回归测试 | Read |
| ROLL | Not found | - | 未发现本窗口内会改变架构、性能或正确性的重大合并项 | official repo activity | 继续观察 | Not found |
| OpenRLHF | [#1299](https://github.com/OpenRLHF/OpenRLHF/pull/1299) | data / trajectory path | ragged experience partitions 从 `zip` 改为 `zip_longest`，避免 uneven batches 被静默截断 | merged PR / sample-count tests | 检查所有跨 rank experience merge 是否可能因 ragged partition 丢样本 | Read |
| NeMo RL | [#3429](https://github.com/NVIDIA-NeMo/RL/pull/3429) | checkpoint / recovery | 保存并恢复 SingleController ready rollout state，明确丢弃 in-flight work | merged PR / restore validation tests | 为 AReaL async controller 定义 ready、visible、recomputable 三类状态 | **Accepted / Read** |
| NeMo RL | [#3494](https://github.com/NVIDIA-NeMo/RL/pull/3494) | training / rollout / recipe | 16 节点 Nemotron Super Omni 120B-A12B GRPO，TP8/EP16/CP2，含 MTP speculative decoding | official config / smoke validation；厂商 recipe | 关注 MTP weight refit、CP、async Gym 和 checkpoint volume 的组合，而非直接接受性能结论 | Read |

### Framework Watch 的工程结论

这两天框架侧最值得带回 AReaL 的不是某个统一“大框架”，而是四个窄但关键的 contract：**异构 role 如何映射物理 GPU、参数如何利用节点内拓扑同步、ragged trajectories 如何保证不丢样本、异步 controller 如何恢复。** 它们分别对应 placement、weight sync、data correctness 和 fault tolerance，是长期 agent workload 能稳定运行的底座。

## Reading Queue Updates

- [ ] `P0.md` 继续保持上限 3，本次不自动挤入全部新信号。
- [ ] 若只有 60 分钟：先读 TideRL 的 Figure 1-4 与 scheduler sections，回答 readiness signal 如何驱动三层决策。
- [ ] 若有第二个 60 分钟：读 MISA-T，重点区分 prefix locality、KV admission 和 trainer workload mixture。
- [ ] RoutePack、AReaL #1575、verl #7291 作为代码/机制精读候选；完成现有 P0 后再替换。
- [ ] NeMo RL #3429 与 vToken 保留 P1，分别支撑 async recovery 与 long-horizon KV management。

## 去重记录

- 新增 Source ID：`arxiv:2608.10402`、`arxiv:2608.11152`、`arxiv:2608.12146`、`github:areal-project/AReaL#1575`、`github:verl-project/verl#7291`、`github:NVIDIA-NeMo/RL#3429`、`arxiv:2608.13263`。
- Boundary late-discovered：TideRL、MISA-T。二者提交时间早于本窗口，但在上一游标后的公告扫描中首次可见，不回写 08-12 scan。
- Framework follow-up：verl multimodal CT、slime correctness、OpenRLHF ragged batch 与 NeMo RL 120B recipe 只记录在 framework watch，不重复拆成 Accepted signal。

## 扫描完整性

- 已扫描：六个 arXiv 主分类 API 返回的 553 条候选，并补查 cs.AR recent 48 条记录；四家核心厂商官方可见来源；Hugging Face Blog 与核心训练库；六个 RL 框架的 official default branch、major PR 与 release。
- 网络说明：GitHub API 响应不稳定，已回退 official Atom、commit patch 和 PR 页面；这不会把未验证的社交媒体信息写成事实。
- 已知盲区：未公开的厂商内部工作、只有二手转述而无 primary source 的材料、尚未合并且没有可复核 benchmark 的 draft PR。
- 下一游标：`2026-08-14 17:39:50`。本报告之后出现的材料留给下一次扫描。

## 下一步动作

- [x] 更新 [Scan Log](scan_log.md)。
- [ ] 精读 TideRL；若它改变对 async RL scheduler 的判断，再更新 [Agentic RL](../topics/agentic_rl.md) 与 rollout playbook。
- [ ] 对 AReaL #1575 和 verl #7291 做代码级对照，区分 placement、dynamic borrowing 与 weight-sync topology 三件事。
- [ ] 将 RoutePack 作为 MoE RL 专题候选，不根据摘要直接改写 [MoE](../topics/moe.md)。

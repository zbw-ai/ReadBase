# 大模型训练推理 Infra 面试主文档与专题知识设计

## 目标

把当前会话中所有与面试有关的项目表达、框架选型和训练 Infra 知识完整汇总进 `private_resume/2026-08-llm-infra-interview-prep.md`，把它建设成一份按优先级和三天准备节奏组织的系统面试框架；同时把可复用原理沉淀到 handbook：`training-infra-roadmap/topics/distributed_training.md` 负责 5D 总览，`training-infra-roadmap/topics/moe.md` 负责 Parallel Folding，`training-infra-roadmap/topics/nccl.md` 负责通信算子，公开 interview 文档提供精炼回答与回链。

章节需要让读者回答六个核心问题：

1. 为什么 Attention 和 MoE 不能总是共用一套最优并行布局？
2. Parallel Folding 如何让同一批物理 ranks 同时承载两套逻辑网格？
3. SP 和 CP 都切 sequence，为什么 SP 不是独立的 world-size 维度？
4. 配置在数学上成立之后，如何判断 process group、通信拓扑、负载和 checkpoint 是否真的正确？
5. DP、TP、PP、CP、EP 分别切什么，动机、实现、通信代价和面试考点是什么？
6. 分布式通信算子分别执行什么数据变换，为什么用于不同并行维度，如何判断使用是否正确？

## 范围与边界

### 5D 并行详细知识源

`training-infra-roadmap/topics/distributed_training.md` 是 5D 并行的唯一详细总览。它负责把已有 DP、TP、PP、CP、SP、MoE 专题串成一套统一决策框架，而不复制每个专题的全部实现细节。新增内容覆盖：

- 5D 的统一定义：DP 切 batch、TP 切层内 tensor、PP 切模型深度、CP 切 context、EP 切 routed expert；
- 每一维的动机、解决的问题、具体切分方法、核心 collective/P2P、显存收益和主要代价；
- Dense world size、MoE rank 复用和 Parallel Folding 的计算边界；
- 配置选择顺序：容量 -> GEMM 效率 -> 通信拓扑 -> profile 验证；
- 面试从定义、通信、组合、场景设计到项目证据的五层考察方式；
- 指向各单项 topic 的相对链接。

其中，5D 总览只维护 SP/CP 的简要比较和组合边界；`sequence_parallelism.md` 与 `context_parallelism.md` 分别维护 activation 切分、tensor layout、KV exchange 等机制细节。

### Parallel Folding 详细知识源

`training-infra-roadmap/topics/moe.md` 是本知识点的唯一详细知识源。新增内容覆盖：

- 解释 Folding 等式所需的 SP/CP 前置结论及指向 5D、SP、CP 专题的回链，不维护第三套机制细节；
- Dense/Attention 与 MoE/Expert 两套逻辑网格；
- world-size 等式和传统 nested layout 的适用边界；
- 8-rank 概念例和 NVIDIA 256-GPU 官方例；
- ProcessGroupCollection、梯度规约域和运行时 token 数据流；
- 节点内/节点间拓扑选择；
- Parallel Folding 的收益、代价、失效模式和排障检查单。

### 面试入口

`training-infra-roadmap/interview/moe.md` 只增加：

- 一道 Megatron 5D 并行综合题；
- 一道 Parallel Folding 高频题；
- 一道 CP 与 SP 区分题；
- 每题的考察意图、3-5 分钟回答、追问、错误回答；
- 指向 topic 详细章节的相对链接。

本次不新建额外 topic/interview 文件。

### 主面试文档汇总

`private_resume/2026-08-llm-infra-interview-prep.md` 是面向本次求职的完整阅读入口和系统准备框架，而不是专题原理的全文合集。它需要让读者在不跳转的情况下完成第一轮口述准备，但不能把 topic 的长篇原理机械复制进来。

主文档采用以下信息架构：

1. **开篇使用说明**：明确岗位、三天倒计时、回答口径和数字边界；给出“今天先看什么”的最短路径。
2. **P0：最高频、最关键、最贴简历**：自我介绍、代表性优化、Ownership、职业选择、Fully Async RLVR、AReaL Agentic RL、MOPD/TILE、千卡/万卡交付、Megatron 5D/显存/通信基本盘。P0 必须排在最前，支持第一天完整过一遍。
3. **P1：高概率技术深挖**：VeRL/AReaL 框架选型、SP/CP、PP/VPP、Parallel Folding、collective、checkpoint/容错、性能分析方法等，供第二天补强。
4. **P2：扩展与压力追问**：更细的框架源码、边界场景、系统设计和开放题，供第三天查漏补缺。
5. **项目证据卡与最后清单**：统一数字、个人贡献边界、危险说法、反问面试官以及面试前快速复习入口。

每道题统一使用紧凑模板：`问题 -> 优先级/建议时长 -> 面试官意图 -> 30 秒结论 -> 2-5 分钟精准回答 -> 项目证据或知识边界 -> 高频追问 -> 危险回答 -> 深入阅读`。简单问题不强行填满全部字段；复杂题把首屏答案控制在可口述范围，推导、图解、配置表、排障细节放入对应 topic，并在题目结尾提供语义明确的就近链接。主文档链接必须是补充阅读，不能代替核心答案。

排序以“命中率 × 简历相关性 × 区分度 × 临场失分风险”为准，而不是按知识学科顺序排列。相同知识只保留一个主答案，其余题目用锚点回链，避免重复和口径漂移。P0/P1/P2 标记、目录、三天学习路径和正文顺序必须一致；不能出现目录写 P0、正文却埋在文末的情况。

专题文档承担完整机制、公式推导、运行时数据流、配置建议、生产陷阱和排障；主文档只保留面试所需的结论、关键公式、项目证据和回答边界。主文档中的每个“深入阅读”链接都要能自然落到对应小节，而不是只链接到一个很长文件的顶部。

实施前按当前会话做逐项覆盖审计：

1. **自我介绍与职业选择**：教育背景一句话；华为 X1 约 200B MoE 代表性优化；Ownership 的定义和 `Scope -> Decision -> Execution -> Coordination -> Outcome`；华为部门搬迁上海、深圳长期规划、技术栈扩展；小鹏组织调整和只看深圳的边界表达。
2. **简历项目**：Fully Async RLVR 相比同步的系统优势、`76 -> 211-255 tokens/s/GPU` 的正确口径和配置配平；AReaL Agentic RL 三层泳道链路及瓶颈；OPD/MOPD 的多 expert 能力汇聚背景、用户明确的 TILE merge 基线命名和训练/验证方法（不得擅自改写成 TIES）；X1/TX 千卡/万卡交付中的模型跑通、profile、优化和迭代达标闭环。
3. **框架选型**：VeRL 与 AReaL 各一句话；最初比较 VeRL、slime、ROLL 后选择 VeRL 的时间点与评估维度；Agentic RL 阶段转向 AReaL 的 fully async、proxy/gateway 和 session 数据链优势；避免声称 VeRL 不支持 async；明确 AReaL gateway 更易改造但外围生产能力仍需补齐。
4. **Megatron 基础与进阶**：5D 各维的含义、动机、实现、通信和考察方式；Dense/MoE world-size；SP 与 CP；PP bubble、1F1B 和 VPP；Parallel Folding；Megatron 训练显存账本和 OOM 生命周期定位。
5. **通信算子**：Broadcast、Reduce、AllReduce、Scatter、Gather、AllGather、ReduceScatter、AllToAll、Send/Recv、Barrier、AllToAllV 的输入输出、版本/API 边界、5D/DistOpt/FSDP 场景、性能与正确性排障。
6. **资料与版本边界**：仓库内五份 NVIDIA MoE 译文入口；Megatron、VeRL、AReaL、NCCL 官方资料；项目发生版本与当前 upstream 能力分开描述。

仓库命名、Git 操作、文档存储位置等元讨论不属于面试材料，不进入主面试手册。

主文档已有题目采用“补强而非重复新增”的方式：例如扩写现有 `AREAL-01` 的 VeRL/slime/ROLL 选型背景，扩写现有 `INFRA-04` 的通信算子答案，复核 `MEGATRON-01/04/07`、`RESUME-01A/01B/01C/02/08/09/10` 是否覆盖本会话最终口径。

#### TILE merge 事实门禁

主文档当前仍存在多处 `TIES-Merging`、TIES 三步机制和对应论文引用，不能做字符串级重命名。实施时必须：

1. 全文件把项目 baseline 统一为用户确认的 `TILE merge`；
2. 删除所有只属于 TIES-Merging 的 trim/elect/sign merge 机制、论文引用和比较结论；
3. TILE 只写用户已确认的项目事实：多个分别 RL 训练得到的 experts 需要汇聚能力，直接 model merge 效果不佳，因而转向以不同 expert 为 teachers、RL 前模型为 student、在对应 RL 数据上做 OPD/MOPD；
4. 未得到用户进一步确认前，不发明 TILE 的算法展开、权重公式或论文来源；
5. 仅针对主面试文档执行 `rg -n "TIES" private_resume/2026-08-llm-infra-interview-prep.md` 并要求零残留；逐项检查 P0 索引、`RESUME-09` 正文、追问、危险回答、项目证据卡、最后清单和资料来源。规格中的迁移说明可以保留 `TIES`，不能对全仓库错误执行零残留门禁。

#### AReaL 三层泳道防回归门禁

`RESUME-08` 保持三层语义，不退化成一条串行 pipeline 或含混的双泳道：

```text
External evals/Agent producer
    -> AReaL online proxy/control plane
    -> Trainer/policy feedback
```

必须保留以下顺序与状态不变量：

- 外部 Agent/Tool/Sandbox 通过 OpenAI-compatible gateway 建立 session；
- CohortManager 在 admission 时快照 rollout version；只有 session 同时 rewarded 和 ended、cohort 完整且通过 ready-time staleness gate 才进入 ready；
- trainer 侧 `OpenAIProxyWorkflow` 消费 ready cohort 并导出/tensorize interactions；
- optimizer/PPO update 后先完成 versioned weight transfer，成功后再按项目实际 API 推进 policy/rollout version；不在缺少当前版本源码证据时枚举 actor/critic/rollout 对象，checkpoint 是独立旁路；
- SGLang/vLLM 是 `RemoteInfEngine` 后端，Tool/Sandbox 属于外部 Agent/environment，不写成 AReaL 固定 stage。

#### 当前会话原子覆盖矩阵

| 原子项 | 目标题号/章节 | 必须保留的事实、数字或公式 | 事实来源 | 禁止声称 | 实施前状态 |
| --- | --- | --- | --- | --- | --- |
| 教育背景与自我介绍 | `RESUME-01` | 厦大本科、清华硕士一句话；两条 Infra 主线 | 简历/用户确认 | 教育背景展开过长 | 已有，复核 |
| X1 代表性优化 | `RESUME-01A` | 约 200B MoE、`0.16x -> 0.95x`、MFU 35%、3K 卡；并行、Grouped MatMul、融合、overlap 闭环 | 简历/用户确认 | 把 2026 新技术倒灌为当时事实 | 已有，复核 |
| Ownership | `RESUME-01B` | `Scope -> Decision -> Execution -> Coordination -> Outcome`，拆开个人/团队/开源贡献 | 用户确认 | “所有代码都是我写的” | 已有，复核 |
| 职业选择 | `RESUME-01C` | 华为部门搬迁上海、深圳长期规划、扩展通用 GPU/RL 技术栈；当前组织调整；只看深圳 | 用户确认 | 主动展开结婚生娃买房；负面评价原公司 | 已有，复核 |
| Fully Async 主故事 | `RESUME-02/03`、`VERL-04` | 初始 `76`，优化稳态 `211-255 tokens/s/GPU`；`236-293` 仅为 `2T+2R` 候选窗口；先讲 sync barrier/长尾，再讲资源与配置配平 | 用户/项目底稿 | 把 `76 -> 211-255` 说成 sync-to-async 3x；同步约 200 未同 workload 闭环前报倍数 | 已有，补门禁 |
| AReaL Agentic RL 链路 | `RESUME-08`、`AREAL-02/03/04` | 三层泳道、rewarded+ended complete cohort、ready staleness、OpenAIProxyWorkflow export、weight sync 后 set_version | AReaL 源码审查/项目事实 | 串行单链；version 在 sync 前；Tool/Sandbox 是固定组件 | 已有，防回归 |
| MOPD 背景与方法 | `RESUME-09`、`AREAL-08`、卡 5 | 不同数据分别 RL 得到多个 experts；TILE merge 效果不佳；RL 前模型为 Student、experts 为冻结 Teachers、对应 RL 数据做 OPD/MOPD；Student rollout，Teacher 对 Student 同一 token path scoring；Teacher 按 `data_source` 路由且只存在于训练期，最终 Student 推理不依赖 Teacher；保留 FUNCTIONAL/NUMERIC/EFFICACY 三层门禁、项目限定的 Teacher headroom Go/No-Go 和个人 PR/设计/实验贡献证据 | 用户确认/项目底稿 | TIES 机制/论文；Teacher 重新生成答案；最终推理依赖路由；用 loss 或 early canary 声称最终多域效果；把项目能力全部说成个人实现 | 待迁移 |
| 千卡/万卡交付 | `RESUME-10/16` | X1/TX 国产卡适配：跑通 -> 采集 -> 定位 -> 优化 -> 验证 -> 迭代达标 | 用户确认 | 只说“保障集群”不讲个人动作 | 已有，复核 |
| 5D 并行 | `MEGATRON-01` | DP/TP/PP/CP/EP 的动机、切分、通信、代价；Dense `W=TP*PP*CP*DP` | 官方资料/用户要求 | 无条件写 `W=TP*PP*CP*DP*EP`；SP 作为独立维度 | 已有，补展开 |
| SP 与 CP | `MEGATRON-04` | SP 依附 TP、局部 activation layout；CP 独立、切全 context/activation、Attention 交换 KV | 官方文档/用户确认回答 | “二者都切 sequence 所以等价” | 已有，复核原句 |
| PP/VPP | `MEGATRON-01/07` | classic interleaved 1F1B、forward/backward 与 model chunks 近似均衡时，`bubble/useful=(p-1)/m`，总时间占比 `(p-1)/(m+p-1)`，VPP 理想再除 `v`；同时保留 microbatch/layer divisibility 或 custom pipeline layout 条件，以及增加 P2P/调度/负载不均的代价 | Megatron 论文/用户要求 | 把理想公式用于不均衡 stage；microbatch 越多永远越好；忽略 layout 整除约束 | 已有，补条件 |
| Parallel Folding | `MEGATRON-01`、`topics/moe.md` | `TP*CP*DP=ETP*EP*EDP`（每 PP stage）；8-rank `2*2*2=1*8*1`；256 GPU `4*2*8*4=1*64*1*4` | NVIDIA 报告/官方指南 | 两套 mesh 相乘；整个 MoE layer 都属于 expert mesh | 已有，补 topic |
| Megatron 显存账本 | `INFRA-02` | per-rank persistent/transient；保留官方 dtype 表：FP16/FP16 `20` 与 `4+16/d`、BF16/FP32 `18` 与 `6+12/d`、FP32/FP32 `16` 与 `8+8/d` bytes/param；单 Distributed Optimizer instance 时 Dense `d=DP*CP`、Expert `d=EDP`，多 instances 分别取实际 `intra_dp_cp`/`intra_expt_dp` group size；再计 activation/PP in-flight、通信/workspace 和生命周期峰值 | 官方代码/用户确认方案 A | 无条件写 `d_dense=DP*CP`；机械使用 `16/d`；把所有阶段峰值相加 | 已有，复核 |
| VeRL 初始选型 | `AREAL-01`、三框架速查 | 最初比较 VeRL/slime/ROLL；当时按训练后端、RLVR 完整度、rollout、权重同步、稳定性、二开成本选择 VeRL | 用户确认/对应版本官方资料 | 对今天的 slime/ROLL 作永久排名 | 待补 |
| VeRL -> AReaL | `AREAL-01/03` | VeRL 标准后训练成熟；Agentic RL 时 AReaL fully async、OpenAI proxy/gateway、session/trajectory 更匹配；gateway 好改但外围能力需补 | 用户确认/官方资料 | “VeRL 只能同步”“AReaL 所有方面更先进” | 待补 |
| 通信算子 | `INFRA-04`、`topics/nccl.md` | 常见 collective/P2P 的输入输出；NCCL 2.31.2 API；DistOpt/FSDP 生命周期；AllToAllV/Barrier 边界 | NCCL/PyTorch 官方资料 | 把语义等价当 bitwise 等价；混淆 gradient 与 parameter tensor | 待扩写 |
| PDF 与版本资料 | `继续阅读/资料来源` | 五份 NVIDIA MoE 译文入口；Megatron/VeRL/AReaL/NCCL 版本边界 | 仓库文件/官方资料 | 孤立文件无入口；把当前版本倒推项目版本 | 已有，补 NCCL |

实施完成后将“实施前状态”逐项更新为已验证，且用 `rg` 检查关键数字、术语、禁止词和锚点；不能只凭人工通读声明无遗漏。

#### 主文档结构验收门禁

- P0 内容必须覆盖上表中全部简历主线，并位于 P1/P2 之前；前三小时至少能完成自我介绍、X1、Ownership、Fully Async、AReaL、MOPD/TILE、5D 与显存八个核心主题。
- 每道 P0 题必须有一句可先说出口的结论；完整回答默认 2-5 分钟，只有标注为系统设计/深挖的题目才允许更长。
- 主文档不得用一整页原理背景挤压回答本身；超过口述需要的机制细节迁入 topic，并保留关键结论和定向锚点链接。
- 对同一事实只维护一个权威口径：吞吐数字、world-size 公式、PP bubble、显存 bytes/param、AReaL version 顺序和 TILE 命名必须全文件一致。
- `目录 -> 三天计划 -> 正文 -> 项目证据卡 -> 最后清单` 五处的题号、优先级和术语必须互相校验；随机从任一题进入时，都能回到上一级主题或进入对应专题。
- 所有新增相对链接和锚点必须通过脚本检查；深入链接至少抽查一次目标小节，而不只检查目标文件存在。

### 通信算子详细知识源

`training-infra-roadmap/topics/nccl.md` 是通信算子的唯一详细知识源；`distributed_training.md` 只维护 5D 并行到通信模式的映射，不重复 collective 语义。章节以 NCCL `2.31.2` 和对应 PyTorch Distributed 在线文档（访问日期 `2026-09-01`）为版本基线，并明确区分“数学通信语义、框架 API、NCCL 原生 API、组合/派生实现”。`nccl.md` 新增内容覆盖：

- Collective 与 point-to-point 的区别；communicator/process group、rank、root 的基本语义；
- Broadcast、Reduce、AllReduce、Scatter、Gather、AllGather、ReduceScatter、AllToAll 的输入输出变换；在 NCCL 2.31.2 中它们均有 host collective API，其中 `ncclAlltoAll`、`ncclGather`、`ncclScatter` 是较新版本能力，必须标注版本边界；
- Send/Recv 与 Barrier 的用途；NCCL 2.31.2 有 host Send/Recv，但没有与 PyTorch `dist.barrier()` 等价的通用 host collective Barrier，后者属于框架同步语义或由后端组合实现；
- `AllReduce = ReduceScatter + AllGather`、`AllReduce = Reduce + Broadcast` 只在 count/partition、dtype、reduction op 等条件兼容时具有数学语义等价；底层不一定机械调用两个 API，浮点归约顺序不同也不保证 bitwise 或数值完全一致；
- NCCL fixed-count AllToAll 与框架/dispatcher 的 variable-count AllToAllV 边界；AllToAllV 必须校验 split 总量和每对 peer 的 send/recv count 一致；
- DP/TP/PP/CP/EP/FSDP/Distributed Optimizer 中的典型使用位置；
- latency/bandwidth、ring/tree、消息大小、拓扑、异步 stream/overlap 等性能判断；
- group membership、count/dtype、调用顺序、shape、stream/wait 等正确性不变量和 hang 排障。

`training-infra-roadmap/interview/tensor_parallelism.md` 增加一道跨并行维度的“常见通信算子及使用场景”面试题，包含考察意图、3-5 分钟回答、追问和错误回答，并回链 `topics/nccl.md`。`interview/moe.md` 只在 EP/Parallel Folding 问题中引用 AllToAll，不复制完整通信算子答案。

### 双向链接与导航

由于 `distributed_training.md` 被提升为 5D 组合总览，本次允许对以下现有文件做最小导航修改：

- `data_parallelism.md`、`tensor_parallelism.md`、`pipeline_parallelism.md`、`context_parallelism.md`、`sequence_parallelism.md`、`moe.md` 各增加一条回到 5D 总览的链接；
- `distributed_training.md` 增加到 `nccl.md` 的通信语言入口，`nccl.md` 回链 5D 总览；
- `topics/nccl.md` 与 `interview/tensor_parallelism.md` 的通信算子综合题保持双向链接；
- `KNOWLEDGE_GRAPH.md` 增加 5D 总览与各单项 topic、Parallel Folding 的关系；
- `MASTER_READING_LIST.md` 收录 5D 总览入口；
- 如 `training-infra-roadmap/README.md` 尚未提供该入口，增加一条导航链接。

这些修改只改变导航，不在各单项 topic 重复 5D 正文。

## 内容架构

### 1. 5D 总览：统一问题框架

`distributed_training.md` 先用一张总表建立五维认知：

| 维度 | 切分对象 | 主要动机 | 核心通信 | 主要代价 |
| --- | --- | --- | --- | --- |
| DP | batch/sample | 扩展吞吐 | classic DP 使用 gradient AllReduce；sharded DP 见 Distributed Optimizer/FSDP 生命周期 | 模型状态复制或分片、跨节点带宽 |
| TP | 单层 hidden/head/tensor | 单层容量和计算 | 每层 all-reduce/all-gather/reduce-scatter | 高频通信、小 GEMM |
| PP | Transformer layers | 整体模型容量 | stage 间 activation/gradient P2P | bubble、stage imbalance |
| CP | context/sequence | 长上下文 activation | Attention KV exchange | KV 通信和序列负载均衡 |
| EP | routed expert identity | MoE 专家参数和计算分布 | token dispatch/combine all-to-all | load imbalance、小 expert GEMM |

每一维统一按“含义 -> 动机 -> 具体做法 -> 通信 -> 代价 -> 配置判断 -> 面试追问”展开，但具体 Row/Column Parallel、pipeline schedule、CP 通信实现和 MoE dispatcher 细节回链已有专题。

5D 到通信算子的简明映射为：

```text
classic DP -> gradient AllReduce
Distributed Optimizer -> gradient ReduceScatter -> local optimizer update -> parameter AllGather
FSDP（取决于 sharding strategy；FULL_SHARD 典型路径）-> pre-forward parameter AllGather -> optional post-forward parameter reshard -> pre-backward parameter AllGather -> post-backward gradient ReduceScatter / reshard
TP  -> layer-level AllReduce / AllGather / ReduceScatter
PP  -> stage-boundary Send/Recv
CP  -> Attention KV 的 P2P / AllGather / AllToAll
EP  -> token dispatch/combine AllToAll；特定 dispatcher 可使用 AllGather 或 variable-count exchange
```

Dense 场景明确：

```text
world_size = TP x PP x CP x DP
```

同时明确 `SP` 不乘入 world size，`EP` 在传统 nested layout 或 Parallel Folding 下也不能不加判断地再乘一次。

### 2. SP 与 CP：内容所有权和面试回答

`distributed_training.md` 用简表回答两者“都切 sequence，为什么不是一回事”：

- SP 是 TP group 内的 activation layout 优化，没有独立 size，不乘入 world size；
- SP 主要分摊 LayerNorm、Dropout、Residual 等在 TP ranks 间重复的 activation，并用 all-gather/reduce-scatter 衔接 TP Linear；
- CP 是独立并行维度，从输入开始持久切分 context 和网络 activation；
- Attention 的跨 token 依赖要求 CP ranks 交换 KV；
- TP=T、CP=C 且开启 SP 时，部分 sequence-parallel activation 可抽象为 `[S/(C*T), B, H]`，但 Attention 的有效上下文仍为全局 `S`；具体 tensor layout 和通信实现回链 SP/CP 单项专题。

`moe.md` 只保留“SP 不进入 world size、CP 进入 Attention mesh”的一段结论，用于解释 Parallel Folding 公式，不放完整对比表和 SP/CP Mermaid。

面试手册保留下列可直接口述的核心回答：

> SP 和 CP 虽然都在 sequence 维度切 activation，但 SP 不是独立并行轴，它依附 TP，主要把 LayerNorm、Dropout、Residual 等位置原本在 TP ranks 上重复的 activation 沿 sequence 分摊，并用 all-gather/reduce-scatter 衔接 TP Linear，因此不计入 world size。CP 是独立并行轴，从输入开始把整个 context 和全部 activation 分给不同 CP ranks；因为 Attention 存在跨 token 依赖，需要在 CP group 内交换 KV。简单说，SP 是 TP 内部的显存和通信布局优化，CP 是面向长上下文的模型并行策略。

### 3. 问题定义：Dense-Sparse Mismatch

解释单个 Transformer block 中两种不同的性能诉求：

- Attention 的大 QKV/投影矩阵可从较高 TP 获益，长上下文可从 CP 获益；
- MoE 专家 GEMM 通常更小，过高 ETP 会继续碎片化 GEMM，而高 EP 有利于专家参数分布和 token 聚合；
- 传统 `EP ⊆ DP` 布局把 expert group 限制在 dense DP 域内，可能导致 GPU 数量乘法膨胀或次优配置。

### 4. 核心机制：同一 rank pool 上的双逻辑网格

使用 Mermaid 表达同一个 PP stage 内的物理 ranks 被两套 process-group mapping 重新解释：

```text
Attention mesh: TP x CP x DP
Expert mesh:    ETP x EP x EDP
```

完整等式为：

```text
world_size = TP x CP x DP x PP
world_size = ETP x EP x EDP x PP
TP x CP x DP = ETP x EP x EDP   # 每个 PP stage
```

明确禁止把两套网格相乘成：

```text
TP x CP x DP x ETP x EP x EDP x PP
```

PP 在两套布局中必须一致；同时说明“唯一结构约束”不等于没有模型 shape、专家数、grouped GEMM、通信实现等可整除约束。

### 5. 两个算例

#### 8-rank 概念例

同一组 8 ranks：

```text
Attention: TP2 x CP2 x DP2 = 8
Expert:    ETP1 x EP8 x EDP1 = 8
```

若 `PP=2`，完整作业为 16 GPUs。此例强调“复用同一批卡”，不把两套布局相乘。

#### 256-GPU 官方例

```text
Attention: TP4 x CP2 x DP8 x PP4 = 256
Expert:    ETP1 x EP64 x EDP1 x PP4 = 256
```

每个 PP stage 都有 64 ranks。Attention 用 TP/CP/DP 解释这些 ranks，MoE 则把同一 rank pool 重新映射为 EP64。

### 6. Process groups 与运行时数据流

区分以下语义域，并明确 Expert mesh 只描述 routed expert 权重与计算，不能把整个 MoE layer 都归入该网格：

- Attention：`tp`、`cp`、`dp`、`dp_cp`、`pp`；
- Routed experts：`ep`、`expt_tp`、`expt_dp`、`pp`；
- Router、shared expert、LayerNorm 和 auxiliary-loss 等组件按其参数语义继续使用 TP/CP 或 dense 相关 group；具体映射以所记录 Megatron-Core 版本的 `ProcessGroupCollection` 和模块实现为准；
- Dense 参数通常在 `dp_cp` 域规约；expert 参数在 `expt_dp` 域规约；
- Model weights 的放置由 TP/PP 或 ETP/EP/PP 决定，并在对应 DP/CP 或 EDP 副本域复制；Distributed Optimizer 进一步分片 optimizer state、master parameters 和更新 shard，更新后再 all-gather model-parameter buffer；FSDP 的按需参数 all-gather 属于另一套参数生命周期，不能与 Distributed Optimizer 混写；
- Parallel Folding 不要求在 Attention 与 Expert 层之间动态搬迁完整权重；跨布局流动的主要是 activation/token；
- 跨布局流动的是 activation/token：Attention output -> Router -> EP dispatch all-to-all -> Expert compute -> combine all-to-all -> 下一层。

### 7. 配置与拓扑判断

先求出包含 TP、CP、PP、ETP、EP、派生 DP/EDP 的容量可行解，同时检查 expert 数、sequence/model shape 和 kernel 的整除约束，再按性能调整：

1. Attention 的 TP/CP 高频通信优先置于 NVLink/NVSwitch 域；
2. Expert 的 EP all-to-all 尽可能限制在高带宽域；
3. ETP 通常从 1 开始搜索，但是否提高必须同时根据单专家显存、expert GEMM shape、TP/ETP 通信和实测 profile 决定，不能归结为单一容量条件；
4. 当 `TP x CP` 或 EP 不能全部放入单个高速域时，先根据通信频率与消息规模确定拓扑优先级，再评估 hierarchical CP；Parallel Folding 的价值之一是允许 Attention 的 CP 与 Expert 的 EP 在不同时刻复用同一高速 rank pool；
5. 通过 profile 验证 GEMM shape、all-to-all 暴露时间、KV 通信、load imbalance、straggler 和跨节点流量，而不是只验证公式；
6. 对 Dense 与 Expert 参数分别验证梯度规约 group、optimizer state 和 distributed checkpoint metadata。

### 8. 代价与排障

明确 Parallel Folding 不是免费优化：它没有消除 CP 的 KV 通信和 EP 的 all-to-all，还增加双网格 process-group 创建、dispatcher 映射、配置搜索以及 checkpoint/recovery metadata 的复杂度。检查顺序为：

1. 两套 world-size 分解与 PP 是否一致；
2. rank membership 和 process-group cardinality 是否正确；
3. dispatcher 的 source/destination rank 和 inverse mapping 是否在双网格之间可逆；
4. dropless 模式下 assignment 数是否等于有效 token 数乘 `top-k`；capacity/drop 模式下是否满足 `accepted + dropped = assignments`，padding token 是否被正确 mask，combine 是否完整恢复原 token 顺序；
5. expert load balance、dropped tokens 和 padding/capacity 开销；
6. EP all-to-all、CP KV 通信是否跨越慢链路；
7. ETP/EP 是否造成过小 GEMM；
8. Dense/Expert 梯度规约、optimizer state 和参数 all-gather 是否使用正确 group；
9. checkpoint 及其配套加载配置是否包含足以重建 Dense/Expert sharding、expert identity、optimizer shard 和 replica mapping 的信息，恢复和并行配置变更后的 reshard 是否一致；具体信息位于 checkpoint 还是外部配置，以实际 Megatron-Core 版本格式为准。

## 图示设计

采用一个主 Mermaid 图，保持浅层、少交叉：

```text
同一 PP stage 的物理 rank pool
        |                       |
Attention logical mesh     Expert logical mesh
   TP x CP x DP            ETP x EP x EDP
        |                       |
 Attention output -> router/dispatch -> experts -> combine
```

颜色只区分三类语义：物理资源、Attention 逻辑组、Expert 逻辑组。公式和算例放在图外，避免图中过多文本。

## 资料来源

事实与数字优先引用：

1. NVIDIA, *Scalable Training of Mixture-of-Experts Models with Megatron Core*；
2. NVIDIA, *MoE Parallel Folding: Heterogeneous Parallelism Mappings for Efficient Large-Scale MoE Model Training with Megatron Core*（arXiv:2504.14960；8-rank 示例标注为该论文附录中的映射示例）；
3. Megatron Core 官方 MoE Parallel Folding 用户指南；
4. Megatron-LM 官方 Context Parallelism 和 Parallelism Strategies 文档；
5. Megatron-Core 官方 `parallel_state.py`、`ProcessGroupCollection` 和 Distributed Optimizer 相关 API/源码；记录引用的 tag/commit 或访问日期 `2026-09-01`，把实现细节标记为版本相关；
6. 本仓库保存的论文中文译文第二部分；
7. NVIDIA NCCL `2.31.2` 官方 Collective Operations、C API 与 point-to-point 文档（访问日期 `2026-09-01`），并参考 NCCL `2.28.3` release notes 标注 AllToAll/Gather/Scatter host API 的版本边界；
8. PyTorch Distributed 官方 collective API 文档（访问日期 `2026-09-01`），用于框架层异步 handle、Barrier 和 variable-size API 边界。

对代码实现相关描述使用“当前 Megatron-Core 实现”表述，避免把版本相关细节写成永恒定义。

## 验收标准

- `topics/distributed_training.md` 能独立回答 5D 各维的含义、动机、切分方法、通信、代价、组合和面试考察方式，并回链单项专题；
- `topics/moe.md` 能独立回答 Parallel Folding、MoE world size、process group、运行时流和排障问题，并回链 5D 总览与 CP/SP，但不复制 SP/CP 机制细节；
- 8-rank 与 256-GPU 示例算术正确，两套逻辑网格没有被误乘；
- `interview/moe.md` 的 5D、Parallel Folding、CP/SP 三道题可在 3-5 分钟内口述，并回链到 topic；
- `topics/nccl.md` 能用输入输出语义解释常见 collective/P2P，区分 NCCL 2.31.2 原生 API 与框架/组合语义，给出 5D、Distributed Optimizer、带 sharding-strategy 边界的 FSDP 使用映射、性能模型和正确性排障；等价关系明确兼容条件和浮点归约边界，AllToAllV 验证 split 总量与逐 peer 收发一致性；`interview/tensor_parallelism.md` 包含可口述的通信算子综合题，且与 `topics/nccl.md` 双向链接；
- DP/TP/PP/CP/SP/MoE 单项 topic 都能回到 5D 总览；`KNOWLEDGE_GRAPH.md`、`MASTER_READING_LIST.md` 和必要时的 handbook README 已更新导航；
- 所有本地 Markdown/PDF 链接存在，外部链接指向官方来源；
- Mermaid 语法闭合，节点文字不拥挤；
- `private_resume/2026-08-llm-infra-interview-prep.md` 通过上述六类覆盖审计；每项都有可独立口述答案，链接只承担延伸阅读；
- 保留工作区既有未提交内容，不覆盖或误提交来源不明的修改；不改动无关文档。

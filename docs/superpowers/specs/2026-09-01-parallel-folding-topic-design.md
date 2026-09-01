# Parallel Folding 工程知识章节设计

## 目标

把 Megatron 5D 并行及 Parallel Folding 从面试文档中的零散解释，提升为可长期维护的工程知识章节：`training-infra-roadmap/topics/distributed_training.md` 负责 5D 总览，`training-infra-roadmap/topics/moe.md` 负责 Parallel Folding，并在 `training-infra-roadmap/interview/moe.md` 提供 3-5 分钟面试回答与回链。

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

已有 `private_resume/2026-08-llm-infra-interview-prep.md` 不重复扩写，避免形成多个内容近似但容易漂移的详细版本。本次不新建额外 topic/interview 文件。

### 通信算子详细知识源

`training-infra-roadmap/topics/nccl.md` 是通信算子的唯一详细知识源；`distributed_training.md` 只维护 5D 并行到通信模式的映射，不重复 collective 语义。`nccl.md` 新增内容覆盖：

- Collective 与 point-to-point 的区别；communicator/process group、rank、root 的基本语义；
- Broadcast、Reduce、AllReduce、Scatter、Gather、AllGather、ReduceScatter、AllToAll 的输入输出变换；
- Send/Recv 与 Barrier 的用途，并明确 Barrier 是同步语义，不等同于张量重分布；
- `AllReduce = ReduceScatter + AllGather`、`AllReduce = Reduce + Broadcast` 的语义等价关系，以及“语义等价不代表底层一定机械执行两个算子”；
- fixed-count AllToAll 与 variable-count AllToAllV/dispatcher 的边界；
- DP/TP/PP/CP/EP/FSDP/Distributed Optimizer 中的典型使用位置；
- latency/bandwidth、ring/tree、消息大小、拓扑、异步 stream/overlap 等性能判断；
- group membership、count/dtype、调用顺序、shape、stream/wait 等正确性不变量和 hang 排障。

`training-infra-roadmap/interview/tensor_parallelism.md` 增加一道跨并行维度的“常见通信算子及使用场景”面试题，包含考察意图、3-5 分钟回答、追问和错误回答，并回链 `topics/nccl.md`。`interview/moe.md` 只在 EP/Parallel Folding 问题中引用 AllToAll，不复制完整通信算子答案。

### 双向链接与导航

由于 `distributed_training.md` 被提升为 5D 组合总览，本次允许对以下现有文件做最小导航修改：

- `data_parallelism.md`、`tensor_parallelism.md`、`pipeline_parallelism.md`、`context_parallelism.md`、`sequence_parallelism.md`、`moe.md` 各增加一条回到 5D 总览的链接；
- `distributed_training.md` 增加到 `nccl.md` 的通信语言入口，`nccl.md` 回链 5D 总览；
- `KNOWLEDGE_GRAPH.md` 增加 5D 总览与各单项 topic、Parallel Folding 的关系；
- `MASTER_READING_LIST.md` 收录 5D 总览入口；
- 如 `training-infra-roadmap/README.md` 尚未提供该入口，增加一条导航链接。

这些修改只改变导航，不在各单项 topic 重复 5D 正文。

## 内容架构

### 1. 5D 总览：统一问题框架

`distributed_training.md` 先用一张总表建立五维认知：

| 维度 | 切分对象 | 主要动机 | 核心通信 | 主要代价 |
| --- | --- | --- | --- | --- |
| DP | batch/sample | 扩展吞吐 | gradient all-reduce，或 reduce-scatter + all-gather | 模型状态复制、跨节点带宽 |
| TP | 单层 hidden/head/tensor | 单层容量和计算 | 每层 all-reduce/all-gather/reduce-scatter | 高频通信、小 GEMM |
| PP | Transformer layers | 整体模型容量 | stage 间 activation/gradient P2P | bubble、stage imbalance |
| CP | context/sequence | 长上下文 activation | Attention KV exchange | KV 通信和序列负载均衡 |
| EP | routed expert identity | MoE 专家参数和计算分布 | token dispatch/combine all-to-all | load imbalance、小 expert GEMM |

每一维统一按“含义 -> 动机 -> 具体做法 -> 通信 -> 代价 -> 配置判断 -> 面试追问”展开，但具体 Row/Column Parallel、pipeline schedule、CP 通信实现和 MoE dispatcher 细节回链已有专题。

5D 到通信算子的简明映射为：

```text
DP  -> gradient AllReduce，或 ReduceScatter + parameter AllGather
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
7. NVIDIA NCCL 官方 Collective Operations 与 point-to-point 文档（记录访问日期 `2026-09-01`）；
8. PyTorch Distributed 官方 collective API 文档，用于框架层异步 handle、Barrier 和 variable-size API 边界。

对代码实现相关描述使用“当前 Megatron-Core 实现”表述，避免把版本相关细节写成永恒定义。

## 验收标准

- `topics/distributed_training.md` 能独立回答 5D 各维的含义、动机、切分方法、通信、代价、组合和面试考察方式，并回链单项专题；
- `topics/moe.md` 能独立回答 Parallel Folding、MoE world size、process group、运行时流和排障问题，并回链 5D 总览与 CP/SP，但不复制 SP/CP 机制细节；
- 8-rank 与 256-GPU 示例算术正确，两套逻辑网格没有被误乘；
- `interview/moe.md` 的 5D、Parallel Folding、CP/SP 三道题可在 3-5 分钟内口述，并回链到 topic；
- `topics/nccl.md` 能用输入输出语义解释常见 collective/P2P，给出 5D/FSDP 使用映射、性能模型和正确性排障；`interview/tensor_parallelism.md` 包含可口述的通信算子综合题；
- DP/TP/PP/CP/SP/MoE 单项 topic 都能回到 5D 总览；`KNOWLEDGE_GRAPH.md`、`MASTER_READING_LIST.md` 和必要时的 handbook README 已更新导航；
- 所有本地 Markdown/PDF 链接存在，外部链接指向官方来源；
- Mermaid 语法闭合，节点文字不拥挤；
- 不改动无关文档，不重复扩写 private resume 主面试文档。

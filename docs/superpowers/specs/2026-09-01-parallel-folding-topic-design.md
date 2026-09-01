# Parallel Folding 工程知识章节设计

## 目标

把 Parallel Folding 从面试文档中的零散解释，提升为 `training-infra-roadmap/topics/moe.md` 中可长期维护的工程知识章节，并在 `training-infra-roadmap/interview/moe.md` 提供 3-5 分钟面试回答与回链。

章节需要让读者回答四个核心问题：

1. 为什么 Attention 和 MoE 不能总是共用一套最优并行布局？
2. Parallel Folding 如何让同一批物理 ranks 同时承载两套逻辑网格？
3. SP 和 CP 都切 sequence，为什么 SP 不是独立的 world-size 维度？
4. 配置在数学上成立之后，如何判断 process group、通信拓扑、负载和 checkpoint 是否真的正确？

## 范围与边界

### 详细知识源

`training-infra-roadmap/topics/moe.md` 是本知识点的唯一详细知识源。新增内容覆盖：

- SP 与 CP 的语义边界及组合后的 tensor shape；
- Dense/Attention 与 MoE/Expert 两套逻辑网格；
- world-size 等式和传统 nested layout 的适用边界；
- 8-rank 概念例和 NVIDIA 256-GPU 官方例；
- ProcessGroupCollection、梯度规约域和运行时 token 数据流；
- 节点内/节点间拓扑选择；
- Parallel Folding 的收益、代价、失效模式和排障检查单。

### 面试入口

`training-infra-roadmap/interview/moe.md` 只增加：

- 一道 Parallel Folding 高频题；
- 一道 CP 与 SP 区分题；
- 每题的考察意图、3-5 分钟回答、追问、错误回答；
- 指向 topic 详细章节的相对链接。

已有 `private_resume/2026-08-llm-infra-interview-prep.md` 不重复扩写，避免形成多个内容近似但容易漂移的详细版本。

## 内容架构

### 1. 前置概念：SP 与 CP

先回答两者“都切 sequence，为什么不是一回事”：

- SP 是 TP group 内的 activation layout 优化，没有独立 size，不乘入 world size；
- SP 主要分摊 LayerNorm、Dropout、Residual 等在 TP ranks 间重复的 activation，并用 all-gather/reduce-scatter 衔接 TP Linear；
- CP 是独立并行维度，从输入开始持久切分 context 和网络 activation；
- Attention 的跨 token 依赖要求 CP ranks 交换 KV；
- TP=T、CP=C 且开启 SP 时，部分 sequence-parallel activation 可抽象为 `[S/(C*T), B, H]`，但 Attention 的有效上下文仍为全局 `S`。

这里用一张对比表和一个小型 Mermaid 数据流图解释，不把具体 kernel/fusion 实现固定成唯一 tensor layout。

### 2. 问题定义：Dense-Sparse Mismatch

解释单个 Transformer block 中两种不同的性能诉求：

- Attention 的大 QKV/投影矩阵可从较高 TP 获益，长上下文可从 CP 获益；
- MoE 专家 GEMM 通常更小，过高 ETP 会继续碎片化 GEMM，而高 EP 有利于专家参数分布和 token 聚合；
- 传统 `EP ⊆ DP` 布局把 expert group 限制在 dense DP 域内，可能导致 GPU 数量乘法膨胀或次优配置。

### 3. 核心机制：同一 rank pool 上的双逻辑网格

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

### 4. 两个算例

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

### 5. Process groups 与运行时数据流

区分以下语义域，并明确 Expert mesh 只描述 routed expert 权重与计算，不能把整个 MoE layer 都归入该网格：

- Attention：`tp`、`cp`、`dp`、`dp_cp`、`pp`；
- Routed experts：`ep`、`expt_tp`、`expt_dp`、`pp`；
- Router、shared expert、LayerNorm 和 auxiliary-loss 等组件按其参数语义继续使用 TP/CP 或 dense 相关 group；具体映射以所记录 Megatron-Core 版本的 `ProcessGroupCollection` 和模块实现为准；
- Dense 参数通常在 `dp_cp` 域规约；expert 参数在 `expt_dp` 域规约；
- Model weights 的放置由 TP/PP 或 ETP/EP/PP 决定，并在对应 DP/CP 或 EDP 副本域复制；Distributed Optimizer 进一步分片 optimizer state、master parameters 和更新 shard，更新后再 all-gather model-parameter buffer；FSDP 的按需参数 all-gather 属于另一套参数生命周期，不能与 Distributed Optimizer 混写；
- Parallel Folding 不要求在 Attention 与 Expert 层之间动态搬迁完整权重；跨布局流动的主要是 activation/token；
- 跨布局流动的是 activation/token：Attention output -> Router -> EP dispatch all-to-all -> Expert compute -> combine all-to-all -> 下一层。

### 6. 配置与拓扑判断

先求出包含 TP、CP、PP、ETP、EP、派生 DP/EDP 的容量可行解，同时检查 expert 数、sequence/model shape 和 kernel 的整除约束，再按性能调整：

1. Attention 的 TP/CP 高频通信优先置于 NVLink/NVSwitch 域；
2. Expert 的 EP all-to-all 尽可能限制在高带宽域；
3. ETP 通常从 1 开始搜索，但是否提高必须同时根据单专家显存、expert GEMM shape、TP/ETP 通信和实测 profile 决定，不能归结为单一容量条件；
4. 当 `TP x CP` 或 EP 不能全部放入单个高速域时，先根据通信频率与消息规模确定拓扑优先级，再评估 hierarchical CP；Parallel Folding 的价值之一是允许 Attention 的 CP 与 Expert 的 EP 在不同时刻复用同一高速 rank pool；
5. 通过 profile 验证 GEMM shape、all-to-all 暴露时间、KV 通信、load imbalance、straggler 和跨节点流量，而不是只验证公式；
6. 对 Dense 与 Expert 参数分别验证梯度规约 group、optimizer state 和 distributed checkpoint metadata。

### 7. 代价与排障

明确 Parallel Folding 不是免费优化：它没有消除 CP 的 KV 通信和 EP 的 all-to-all，还增加双网格 process-group 创建、dispatcher 映射、配置搜索以及 checkpoint/recovery metadata 的复杂度。检查顺序为：

1. 两套 world-size 分解与 PP 是否一致；
2. rank membership 和 process-group cardinality 是否正确；
3. dispatcher 的 source/destination rank 和 inverse mapping 是否在双网格之间可逆；
4. dropless 模式下 assignment 数是否等于有效 token 数乘 `top-k`；capacity/drop 模式下是否满足 `accepted + dropped = assignments`，padding token 是否被正确 mask，combine 是否完整恢复原 token 顺序；
5. expert load balance、dropped tokens 和 padding/capacity 开销；
6. EP all-to-all、CP KV 通信是否跨越慢链路；
7. ETP/EP 是否造成过小 GEMM；
8. Dense/Expert 梯度规约、optimizer state 和参数 all-gather 是否使用正确 group；
9. checkpoint 是否完整保存两套 mesh metadata，恢复和并行配置变更后的 reshard 是否一致。

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
2. NVIDIA, *MoE Parallel Folding: Heterogeneous Parallelism for Training Giant Mixture-of-Experts Models*（arXiv:2504.14960；8-rank 示例标注为该论文附录中的映射示例）；
3. Megatron Core 官方 MoE Parallel Folding 用户指南；
4. Megatron-LM 官方 Context Parallelism 和 Parallelism Strategies 文档；
5. Megatron-Core 官方 `parallel_state.py`、`ProcessGroupCollection` 和 Distributed Optimizer 相关 API/源码；记录引用的 tag/commit 或访问日期 `2026-09-01`，把实现细节标记为版本相关；
6. 本仓库保存的论文中文译文第二部分。

对代码实现相关描述使用“当前 Megatron-Core 实现”表述，避免把版本相关细节写成永恒定义。

## 验收标准

- `topics/moe.md` 能独立回答 Parallel Folding、world size、CP/SP、process group、运行时流和排障问题；
- 8-rank 与 256-GPU 示例算术正确，两套逻辑网格没有被误乘；
- `interview/moe.md` 的两道题可在 3-5 分钟内口述，并回链到 topic；
- 所有本地 Markdown/PDF 链接存在，外部链接指向官方来源；
- Mermaid 语法闭合，节点文字不拥挤；
- 不改动无关文档，不重复扩写 private resume 主面试文档。

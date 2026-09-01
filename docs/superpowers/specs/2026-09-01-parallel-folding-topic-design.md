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

区分以下语义域：

- Attention：`tp`、`cp`、`dp`、`dp_cp`、`pp`；
- Expert：`ep`、`expt_tp`、`expt_dp`、`pp`；
- Dense 参数通常在 `dp_cp` 域规约；expert 参数在 `expt_dp` 域规约；
- 权重和 optimizer state 按各自参数归属长期分片，不是在每层间动态搬迁完整权重；
- 跨布局流动的是 activation/token：Attention output -> Router -> EP dispatch all-to-all -> Expert compute -> combine all-to-all -> 下一层。

### 6. 配置与拓扑判断

先以容量约束确定 TP/PP/EP，再按性能调整：

1. Attention 的 TP/CP 高频通信优先置于 NVLink/NVSwitch 域；
2. Expert 的 EP all-to-all 尽可能限制在高带宽域；
3. ETP 默认优先尝试 1，只有单专家权重或 GEMM 无法容纳时再提高；
4. 通过 profile 验证 GEMM shape、all-to-all 暴露时间、load imbalance、straggler 和跨节点流量，而不是只验证公式；
5. 对 Dense 与 Expert 参数分别验证梯度规约 group、optimizer state 和 distributed checkpoint metadata。

### 7. 代价与排障

明确 Parallel Folding 不是免费优化。检查顺序为：

1. 两套 world-size 分解与 PP 是否一致；
2. rank membership 和 process-group cardinality 是否正确；
3. 每个 token 的 dispatch/combine 数量及路由容量是否守恒；
4. expert load balance、dropped tokens 和 padding/capacity 开销；
5. EP all-to-all、CP KV 通信是否跨越慢链路；
6. ETP/EP 是否造成过小 GEMM；
7. Dense/Expert 梯度规约和 optimizer state 是否使用正确 group；
8. checkpoint 保存、恢复和并行配置变更后的 reshard 是否一致。

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
2. Megatron Core 官方 MoE Parallel Folding 用户指南；
3. Megatron-LM 官方 Context Parallelism 文档；
4. 本仓库保存的论文中文译文第二部分。

对代码实现相关描述使用“当前 Megatron-Core 实现”表述，避免把版本相关细节写成永恒定义。

## 验收标准

- `topics/moe.md` 能独立回答 Parallel Folding、world size、CP/SP、process group、运行时流和排障问题；
- 8-rank 与 256-GPU 示例算术正确，两套逻辑网格没有被误乘；
- `interview/moe.md` 的两道题可在 3-5 分钟内口述，并回链到 topic；
- 所有本地 Markdown/PDF 链接存在，外部链接指向官方来源；
- Mermaid 语法闭合，节点文字不拥挤；
- 不改动无关文档，不重复扩写 private resume 主面试文档。

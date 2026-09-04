# Megatron 5D 并行：从容量约束到拓扑映射

<a id="five-d-framework"></a>
## 问题框架

单卡训练大模型同时受四类约束：参数和 optimizer state 放不下、长序列 activation 放不下、单步算力不够、单一并行维度的通信或计算粒度已经恶化。Megatron 的 5D 并行不是五个缩写的列表，而是一套把不同对象切到不同 process group 的组合方法：

> DP 切样本，TP 切层内 tensor，PP 切网络深度，CP 切上下文，EP 切 routed experts。

面试和生产配置都应按同一条逻辑展开：**切什么 → 为什么切 → rank 上实际保存/计算什么 → 引入什么通信 → 瓶颈会迁移到哪里**。

## 30 秒总览

| 维度 | 切分对象 | 主要动机 | 典型通信 | 主要代价 |
| --- | --- | --- | --- | --- |
| DP | batch / sample | 扩展吞吐 | gradient AllReduce；分片 optimizer 使用 ReduceScatter/AllGather | 模型状态复制或跨节点规约 |
| TP | hidden、head、MLP tensor | 单层容量与计算 | 每层 AllReduce、AllGather、ReduceScatter | 高频通信、小 GEMM |
| PP | Transformer layers | 整模型容量 | stage 间 Send/Recv | bubble、stage imbalance |
| CP | context / sequence | 长上下文 activation | Attention KV 的 P2P、AllGather 或 AllToAll | KV 通信、变长负载不均 |
| EP | expert identity | 分散专家参数与计算 | token dispatch/combine AllToAll | 动态负载、小 expert GEMM、尾延迟 |

Dense Transformer 的基本公式是：

```text
world_size = TP × PP × CP × DP
```

`SP` 不增加 rank，不能乘进 world size。MoE 中 `EP` 是否能继续乘入该式取决于 rank mapping；Parallel Folding 下 Attention 和 Expert 使用同一批物理 rank 的两套逻辑网格，详见 [MoE 与 Parallel Folding](moe.md#parallel-folding)。

## 1. Data Parallelism：切 batch

### 动机与做法

每个 DP rank 保存相同的逻辑模型，处理不同 microbatch，反向后聚合相同参数的梯度。它的第一目标是扩大样本吞吐，而不是让模型本身变小。

经典复制式 DP：

```text
rank 0: batch A -> g0 ┐
rank 1: batch B -> g1 ├─ gradient AllReduce -> every rank obtains reduced g
rank 2: batch C -> g2 ┘
```

Megatron Distributed Optimizer、ZeRO 和 FSDP 仍保留 DP 的数学语义，只是进一步分片 model states。典型 Megatron Distributed Optimizer 路径是：

```text
gradient ReduceScatter
    -> each rank updates its optimizer/parameter shard
    -> parameter AllGather
```

### 解决什么问题

- 增加 global batch 下的吞吐和总算力；
- 让 optimizer state、gradient 或 parameter 可以在 DP domain 中分片；
- 为不同 pipeline replica 提供独立数据供给。

### 代价和配置边界

`GBS = MBS × DP × gradient_accumulation_steps`。扩大 DP 会改变 global batch，必须一起考虑学习率、warmup 和收敛，不是纯系统参数。Megatron 开启 CP 时，microbatch 数仍只除以纯 DP；但 Dense 参数的梯度规约和 Distributed Optimizer sharding domain 默认可能覆盖 `DP×CP`，因为 CP ranks 持有相同 Dense 参数并分别贡献局部 context 的梯度。

完整实现对比见 [Data Parallelism](data_parallelism.md)。

## 2. Tensor Parallelism：切单层大矩阵

### 动机与做法

TP 在 Transformer layer 内切 Linear、Attention head 或 embedding，使单个大矩阵不必完整放在一张 GPU。以 `Y=XW` 为例：

- Column Parallel 沿输出维切 `W`，各 rank 产生不同输出 feature；
- Row Parallel 沿输入维切 `W`，各 rank 产生 partial sum，随后规约；
- Megatron 常把 MLP/Attention 的 Column 与 Row Parallel 成对布置，让中间分片直接传递，减少不必要的重建。

### 解决什么问题

- 单层参数和 GEMM 无法放入或高效执行；
- hidden size、head 数或 vocabulary 需要跨卡分片；
- 为更大的 microbatch 腾出模型状态显存。

### 代价和配置边界

TP collective 几乎每层、每个 microbatch 都发生，因此通常优先放进 NVLink/NVSwitch 域。TP 过大会同时缩小 GEMM 的 M/N/K 并增加通信参与 rank，9B 或小 expert GEMM 尤其容易出现“卡更多反而更慢”。

完整 Row/Column Parallel、通信和性能模型见 [Tensor Parallelism](tensor_parallelism.md)。

<a id="pipeline-vpp"></a>
## 3. Pipeline Parallelism：切模型深度

### 动机与做法

PP 把连续或自定义布局的 layer 放到不同 stage。microbatch 的 activation 在 forward 中由前一 stage Send 到后一 stage，gradient 在 backward 中反向 Send/Recv。它直接减少每个 rank 保存的层数，但引入 pipeline warmup、steady、cooldown 和 stage 间等待。

对 stage 均衡、忽略通信的 non-interleaved 1F1B，设物理 stage 数为 `p`、每 iteration 的 microbatch 数为 `m`：

```text
useful_time = m × (t_f + t_b)
bubble_time = (p - 1) × (t_f + t_b)

bubble / useful = (p - 1) / m
bubble / total  = (p - 1) / (m + p - 1)
```

第一个式子回答“相对有效计算的额外开销”，第二个式子回答“总 step 中有多少比例是 bubble”。

### VPP 解决什么问题

Virtual Pipeline Parallelism 不增加 GPU，而是令每个物理 stage 持有 `v` 个不连续 model chunks，并用 interleaved 1F1B 更频繁地穿插执行。若 chunks、forward/backward 近似均衡，microbatch/layer layout 满足调度约束，且先忽略额外通信，理想 bubble 相对 non-interleaved 基线约缩小 `v` 倍：

```text
bubble / useful ≈ (p - 1) / (m × v)
```

它不是免费优化：P2P 次数通常上升，activation 生命周期和调度更复杂，chunk 过小会降低 kernel efficiency。现代 custom pipeline layout 可以缓解传统 layer divisibility 约束，但仍必须按真实 layer 成本检查 stage balance。

完整调度背景见 [Pipeline Parallelism](pipeline_parallelism.md)。

## 4. Context Parallelism：切完整上下文

### 动机与做法

长上下文训练时，参数可能放得下，但 activation、Attention 中间量、logits 或 workspace 已经 OOM。CP 是独立并行轴，从输入开始把 context 和几乎全部网络 activation 持久分给 CP ranks；每个 rank 只保存大约 `S/CP` 的 local query 和 activation。

Attention 存在跨 token 依赖，因此 local Q 仍需访问全局 KV。Megatron 可以使用 ring/P2P、AllGather 或 AllToAll 类路径，具体取决于 CP communication type、Attention 实现和 GQA/MQA 布局。

### 解决什么问题

- 128K/256K 等长上下文的 activation 容量；
- 将 attention 计算和 KV 存储分摊到多 rank；
- 在不继续放大 TP 的情况下扩展 sequence。

### 代价和配置边界

CP 不切 Dense 参数；它节省的是与 sequence 相关的 activation。变长样本会制造 CP load imbalance，KV 通信还可能跨节点。若 `TP×CP` 能放入单机高速域通常一起放；放不下时一般优先保证高频 TP 本地，再考虑 hierarchical CP，但最终要按真实消息量 profile。

完整机制见 [Context Parallelism](context_parallelism.md)。

## 5. Expert Parallelism：切 routed experts

### 动机与做法

MoE layer 的 router 为 token 选择 top-k experts。EP rank 保存不同 experts；token 经 permute 后按目标 expert dispatch 到对应 rank，执行 Grouped GEMM，再通过 combine exchange 返回并恢复 token 顺序：

```text
tokens -> router/top-k -> permute -> AllToAll dispatch
       -> local experts / Grouped GEMM
       -> AllToAll combine -> unpermute
```

### 解决什么问题

- 分散总 expert 参数和 expert compute；
- 避免每个 rank 复制所有 experts；
- 让高总参数、低 activated parameters 的 sparse 模型扩展到更多 GPU。

### 代价和配置边界

EP 把 Dense 的规则通信变成动态 token exchange。expert hot spot、每 peer count 不均、跨节点 AllToAll 和小 expert batch 会共同放大尾延迟。ETP 是 expert tensor parallel，可以与 Attention TP 不同；EDP 则复制同一 expert shard并同步其梯度。完整机制和 Parallel Folding 见 [MoE](moe.md)。

<a id="sp-vs-cp"></a>
## 6. SP 为什么不是第六个 world-size 轴

SP 和 CP 都会在 tensor shape 上切 sequence，但系统语义不同：

| 对比项 | Sequence Parallelism | Context Parallelism |
| --- | --- | --- |
| rank 来源 | 复用 TP group | 独立 CP group |
| 切分范围 | TP 区域之间原本重复的 LayerNorm、Dropout、Residual 等 activation | 从输入开始的 context 和几乎全部 activation |
| Attention 全局语义 | 不独立分布完整 context | local Q 通过 KV exchange 访问全局 context |
| world size | 不增加 | 乘入公式 |
| 主要目标 | TP 内 activation 去重和通信布局优化 | 扩展长上下文 |

一句话回答：

> SP 是依附 TP 的局部 activation layout 优化；CP 是面向长上下文的独立模型并行策略。

TP=`T`、CP=`C` 且启用 SP 时，部分 SP 区域的 local activation 可以近似写为 `[S/(C×T), B, H]`，但 Attention 的语义上下文仍是全局 `S`。完整细节见 [Sequence Parallelism](sequence_parallelism.md) 和 [Context Parallelism](context_parallelism.md)。

<a id="five-d-config"></a>
## 7. 5D 如何组合

### 第一步：满足容量

1. 计算每个 PP stage 的 Dense、Expert 参数与 optimizer state；
2. 计算 local sequence、saved activation、PP in-flight microbatches；
3. 单列 logits/loss、collective bucket、kernel workspace 和 checkpoint/weight-sync 临时副本；
4. 先找到能运行的 TP/PP/CP/EP 候选集合。

### 第二步：恢复计算粒度

- TP、ETP 是否把 GEMM 切得过小；
- EP 后每个 expert 实际获得多少 token；
- MBS 和 packing 后的有效 token 是否足够；
- PP/VPP chunk 是否太小或不均衡。

### 第三步：映射拓扑

通用优先级不是绝对规则，但常见起点是：

1. 高频 TP 留在 NVLink/NVSwitch 域；
2. 能放下时把 `TP×CP` 一起留在节点内；
3. EP 是否跨节点取决于 experts 数、token traffic、dispatcher 和网络；
4. DP 常承担较低频、大消息通信，可扩展到节点间；
5. PP 边界应减少跨慢链路次数并避免 stage 热点。

### 第四步：用 profile 收敛

容量可行不等于性能可行。至少同时看：

- GEMM shape、Tensor Core 利用率和 kernel launch；
- TP/CP/EP/DP 各 process group 的 exposed communication；
- pipeline bubble 和各 stage p50/p99；
- expert load、token drop/padding、AllToAll per-peer counts；
- effective tokens/s、step time、peak memory、loss 与收敛。

## 8. 高频面试考法

### 第一层：定义

“五个维度分别切什么？”必须用 batch、tensor、layer、context、expert 五个对象回答，而不是只展开缩写。

### 第二层：通信

“每个维度主要用什么 collective？”要能说明张量语义：DP 同步 gradient、TP 组合 layer partial result、PP 传 activation/gradient、CP 交换 KV、EP dispatch/combine token。详见 [NCCL 与分布式通信算子](nccl.md)。

### 第三层：world size

先问 Dense 还是 MoE、传统 nested layout 还是 Parallel Folding。不能无条件回答 `TP×PP×CP×EP×DP`。

### 第四层：场景设计

面试官给 GPU 数、模型大小和 sequence length 时，先追问 total/activated parameters、hidden/layers/experts/top-k、dtype、MBS/GBS、长度分布和节点拓扑，再给初始方案和验证实验。

### 第五层：项目证据

高级工程师还会被追问：实际配置是什么、瓶颈怎样定位、哪个方案被否决、优化后瓶颈迁移到哪里、性能提升是否守住 loss/精度/长稳。

## 9. 生产排障检查单

### 能跑但很慢

1. 对比各维 communication exposed time，而不是只看 NCCL 总时间；
2. 检查 TP/ETP 后 GEMM 是否过碎；
3. 检查 PP stage balance 和 microbatch 数；
4. 检查 CP 变长负载和 KV traffic；
5. 检查 EP token imbalance、dispatcher 和 Grouped GEMM batch；
6. 固定 workload 做单变量 A/B，再重新 profile 瓶颈迁移。

### 某些 rank OOM

检查 uneven PP layout、embedding/LM head、shared expert、hottest expert rank、PP in-flight activation、collective bucket、CUDA Graph private pool 和 checkpoint/weight sync 临时副本。总参数除以 world size 通常会低估峰值。

### 多机 hang

先确认各 rank 的 process group、collective 顺序、count、dtype 和 shape 一致，再找 first bad rank、CUDA/Xid、NIC/link 和拓扑问题。NCCL timeout 经常是上游 rank 失败的结果，不是根因。

## 相关材料

- [Megatron-LM](../papers/megatron_lm.md)
- [ZeRO](../papers/zero.md)
- [Llama 3](../tech_reports/llama3.md)
- [MegaScale](../tech_reports/megascale.md)
- [Data Parallelism](data_parallelism.md)
- [Tensor Parallelism](tensor_parallelism.md)
- [Pipeline Parallelism](pipeline_parallelism.md)
- [Context Parallelism](context_parallelism.md)
- [Sequence Parallelism](sequence_parallelism.md)
- [MoE 与 Parallel Folding](moe.md#parallel-folding)
- [NCCL 与分布式通信算子](nccl.md)
- [Megatron 5D / MoE 面试题](../interview/moe.md#megatron-5d)
- [Agentic for Embodied](agentic_for_embodied.md)

## 参考资料

- [Megatron Core Parallelism Strategies Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
- [Megatron Core Context Parallelism](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)
- [Megatron Core MoE Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html)
- [Megatron Core Distributed Optimizer](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/dist_optimizer.html)

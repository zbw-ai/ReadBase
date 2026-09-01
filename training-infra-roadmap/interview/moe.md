# Interview: MoE、5D 并行与 Parallel Folding

深入原理：[Megatron 5D 并行](../topics/distributed_training.md)、[MoE 与 Parallel Folding](../topics/moe.md#parallel-folding)。

<a id="megatron-5d"></a>
## 高频题 1：Megatron 的 5D 并行分别解决什么问题

### 面试官意图

检查分布式训练基本盘，以及候选人能否从容量、计算粒度、通信和拓扑做组合，而不是只背 DP/TP/PP/CP/EP 定义。

### 3–5 分钟优秀回答

> DP 切 batch，不同副本处理不同数据，通过 gradient AllReduce 或 sharded optimizer 通信扩展吞吐；TP 切层内 hidden/head/大矩阵，解决单层容量，但每层有高频 collective，过大会让 GEMM 变碎；PP 按 layer 切模型深度，解决整模型容量，但引入 bubble 和 stage imbalance；CP 从输入开始持久切 context 和几乎全部 activation，服务长上下文，Attention 需要交换 KV；EP 切 routed expert identity，token 经 AllToAll 到目标 expert 计算再返回，难点是动态负载和尾延迟。
>
> Dense 模型先写 `world_size=TP×PP×CP×DP`，SP 依附 TP，不乘入 world size。MoE 不能无条件再乘 EP；Parallel Folding 下，Attention 和 Expert 是同一批物理 ranks 的两套 mapping，每个 PP stage 满足 `TP×CP×DP=ETP×EP×EDP`。
>
> 配置时先满足参数和 activation 容量，再检查 TP/ETP 后的 GEMM 粒度，把高频 TP 优先放入 NVLink/NVSwitch 域，联合评估 CP KV、EP AllToAll 和 PP bubble，最后用固定 workload profile 收敛。容量可行不代表性能最优。

### 高频追问

- 为什么 Dense optimizer sharding group 可能是 `DP×CP`，microbatch 数却只除以 DP？
- TP 与 CP 哪个优先放单机？
- 为什么 SP 不算第六维？
- 64 张 GPU、35B MoE、128K context 如何给初始并行方案？

### 常见错误回答

- 无条件写 `world_size=TP×PP×CP×EP×DP`；
- 把 ETP 默认等于 Attention TP；
- 把 SP/VPP 乘进 world size；
- 只说显存收益，不说通信、GEMM 和 topology。

## 高频题 2：Parallel Folding 解决什么问题

### 面试官意图

检查是否真正理解 NVIDIA Megatron Core 新一代 MoE parallel mapping，以及能否区分 logical mesh 与 physical GPU。

### 3–5 分钟优秀回答

> Attention 和 routed experts 的最优并行度不同：Attention 的大 GEMM 和长上下文可能需要较高 TP/CP；expert GEMM 本来较小，过高 ETP 会继续碎片化，更高 EP 则有利于分散 experts。Parallel Folding 让同一 PP stage 的物理 ranks 同时拥有两套坐标：Attention 使用 `TP×CP×DP`，expert 使用 `ETP×EP×EDP`，并满足两边乘积相等。
>
> 例如 8 ranks 上，Attention 可以是 `2×2×2`，expert 是 `1×8×1`；不是 64 张卡。NVIDIA 的 256 GPU 例子是 `4×2×8×4 = 1×64×1×4`。这样 Attention 保留 TP/CP，expert 把 ETP 降到 1、EP 扩到 64，恢复 expert GEMM 粒度。
>
> 实现上依赖两套 process groups。运行时 Attention 在 TP/CP mesh 上执行；router 后 token 按 expert permute，经 EP AllToAll、ETP/Grouped GEMM，再 combine。Expert mesh 只属于 routed expert 子图，不代表 router、residual 和整个 MoE layer 都使用 expert mapping。最终还要验证 group membership、token inverse mapping、per-peer count、expert load、AllToAll p99 和 checkpoint expert reshuffle。

### 高频追问

- 为什么当前 Megatron 定义下通常是 `CP×DP=EP×EDP`？`DP=EP×EDP` 的简写何时才成立？
- ProcessGroupCollection 解决什么问题？
- 为什么两套 mesh 不能相乘？
- Parallel Folding 一定更快吗？
- expert optimizer state 在哪个 group 分片/规约？

### 常见错误回答

- 把 Parallel Folding 说成新增一维并行；
- 把 `TP×CP×DP` 与 `ETP×EP×EDP` 相乘；
- 只背公式，不讲 token 数据流；
- 认为整个 MoE block 都属于 expert mesh。

## 高频题 3：SP 和 CP 都切 sequence，区别是什么

### 面试官意图

快速验证候选人是否理解 tensor layout、Attention 跨 token 依赖和 world-size 公式。

### 优秀回答

> SP 和 CP 虽然都在 sequence 维度切 activation，但 SP 不是独立并行轴，它依附 TP，主要把 LayerNorm、Dropout、Residual 等位置原本在 TP ranks 上重复的 activation 沿 sequence 分摊，并用 AllGather/ReduceScatter 衔接 TP Linear，因此不计入 world size。CP 是独立并行轴，从输入开始把整个 context 和几乎全部 activation 分给不同 CP ranks；因为 Attention 存在跨 token 依赖，需要在 CP group 内交换 KV。简单说，SP 是 TP 内部的显存和通信布局优化，CP 是面向长上下文的模型并行策略。

### 高频追问

- TP、CP、SP 同开时 local sequence shape 如何理解？
- 为什么 TP+EP 时 Megatron 要求启用 SP？
- GQA/MQA 如何改变 CP 的 KV 通信？

### 常见错误回答

- “SP 切短序列，CP 切长序列”；
- 把 SP 和 CP 都乘入 world size；
- 认为 SP 已经把 Attention 的完整 context 独立分布。

## 高频题 4：MoE 的 AllToAll 为什么难优化

### 优秀回答

> Router 让每个 token 选择 top-k experts；token 按目标 expert permute 后 dispatch 到 EP ranks，本地执行 Grouped GEMM，再 combine 返回。AllToAll 难点是每个 peer 的 token count 动态且不均，hot expert 会制造 rank straggler；EP 过大时每个 expert 的 token batch 又变小，GEMM efficiency 下降。优化要联合看 per-expert tokens、per-peer send/recv count、dispatch/combine p95/p99、Grouped GEMM shape、expert placement、dropless/padding 策略和网络拓扑。

### 生产案例

平均 step time 正常但 p99 周期性升高时，先把 MoE layer 拆成 router、permute、dispatch、Grouped GEMM、combine、unpermute，再按 rank 关联 expert load 与 NIC traffic。只看 NCCL 总时间无法判断是网络慢还是上游 token imbalance。

## 补充追问清单

- top-1 与 top-2 routing 对效果、通信量和 expert compute 有何影响？
- capacity factor、dropless routing、padding 和 dropped token 如何取舍？
- router auxiliary loss / load balancing 如何影响训练稳定性？
- expert hot spot 应监控哪些 histogram 和 p99 指标？
- MoE checkpoint 改变 EP/ETP 时如何按 global expert identity reshuffle？
- 为什么“每 token 只激活少数 experts”不代表 MoE 一定比 Dense 更快？

## 相关专题

- [Megatron 5D 并行：完整工程章节](../topics/distributed_training.md)
- [MoE 与 Parallel Folding：双网格、运行时和排障](../topics/moe.md#parallel-folding)
- [NCCL 与通信算子](../topics/nccl.md)
- [Sequence Parallelism](../topics/sequence_parallelism.md)
- [Context Parallelism](../topics/context_parallelism.md)

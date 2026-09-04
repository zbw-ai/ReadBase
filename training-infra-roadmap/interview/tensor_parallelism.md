# Interview: Tensor Parallelism

相关手册章节：[Tensor Parallelism](../topics/tensor_parallelism.md)

## 高频面试题

1. Tensor Parallel 解决什么问题？
2. Column Parallel 和 Row Parallel 怎么切？
3. 为什么 TP 通常放在单机 NVLink 内？
4. TP forward/backward 有哪些 collective？
5. TP 和 PP/DP/FSDP 如何组合？

<a id="communication-operators"></a>
## 高频面试题：常见通信算子执行什么操作，用在哪里

### 面试官意图

检查候选人能否把 collective 的输入输出布局映射到 gradient、parameter、activation、KV 和 routed token，而不是只会背 AllReduce 等名词。

### 3–5 分钟回答

> Broadcast 把 root 的一份 tensor 复制给所有 ranks；Reduce 把所有输入规约到 root；AllReduce 让每个 rank 都得到完整规约结果，常用于 classic DP gradient 或 TP partial sum；Scatter 把 root 的不同分片发给不同 ranks，Gather 只在 root 收集所有分片；AllGather 让每个 rank 重建全部 shards，常用于 sharded parameter/activation；ReduceScatter 规约后每 rank 只保留一片，常用于 Distributed Optimizer/FSDP gradient 和 SP；AllToAll 让每个 rank 给不同 peer 发送不同数据，是 MoE token dispatch/combine 的核心；PP 与 ring CP 常用 Send/Recv。
>
> Megatron Distributed Optimizer 的典型生命周期是 gradient ReduceScatter → local optimizer update → parameter AllGather。FSDP FULL_SHARD 常在计算前 AllGather parameter，之后 reshard，backward 后 ReduceScatter gradient，但具体顺序取决于 sharding strategy 和 reshard/prefetch 配置。
>
> Correctness 先检查 process group、调用顺序、count/shape、dtype/op/root/peer 和 stream wait；性能再看消息大小、频率、topology、p95/p99 与 exposed time。`ReduceScatter+AllGather` 只在分片、dtype、op 和 layout 兼容时与 AllReduce 数学等价，浮点归约顺序也不保证 bitwise 一致。

### 高频追问

- Broadcast 与 AllGather 都会复制数据，区别是什么？
- gradient ReduceScatter 后为什么还要 parameter AllGather？
- fixed-count AllToAll 与 AllToAllV 有什么边界？
- Barrier 能否解决异步 race？
- 为什么 NCCL timeout 往往不是根因？

### 常见错误回答

- 把 Broadcast 说成收集所有 ranks 的输入；
- 把 gradient 和 parameter 的通信顺序说反；
- 认为异步 collective 返回就代表通信完成；
- 把 AllToAllV 当成所有 NCCL 版本都有的通用 host API；
- 只说 ring/tree，不说消息、group 和 tensor 语义。

完整四卡示例、NCCL 2.31.2 API 边界、5D/FSDP 映射和 hang 排障见 [NCCL 与分布式通信算子](../topics/nccl.md#collective-map)。

## 追问问题

- TP size 从 4 增加到 8，吞吐一定提升吗？
- checkpoint 如何从 TP=8 转成 TP=4？
- 如果 TP all-reduce p99 抖动，怎么定位？

## 生产环境案例

8 卡 H100 节点训练 70B dense model，TP=8 单节点通信快但 DP degree 低；TP=4 可增加 DP，但单卡显存更紧。应结合 hidden size、micro-batch、activation checkpointing 和网络拓扑试算。

## 常见错误回答

- “TP 就是把模型平均切开。”错在忽略矩阵维度、通信点和计算粒度。
- “TP 越大越好。”错在忽略通信频率和小 GEMM 效率。

## 优秀回答示例

Tensor Parallel 是 intra-layer parallelism，通常围绕 Transformer 中的大 Linear/GEMM 做列切和行切。它用额外 collective 换取单卡显存和算力扩展，但通信频繁，所以生产上优先放在高速互联域内，并与 PP/DP/FSDP 共同决定整体并行布局。

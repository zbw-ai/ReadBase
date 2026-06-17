# Tensor Parallelism

Tensor Parallelism, TP, 是把 Transformer 单层里的大算子切到多张 GPU 上执行。它不是“多卡训练”的泛称，而是 intra-layer parallelism：一个 Linear、Attention projection、MLP projection 的权重和中间结果被多个 rank 共同持有、共同计算、共同通信。

如果只能记一句话：TP 用频繁的高速 collective 换单层模型容量和单层 GEMM 吞吐，所以它最怕慢网络、错拓扑、错 rank mapping 和过大的 TP size。

## 1. Tensor Parallelism 解决什么问题

TP 的出现不是为了让配置表更复杂，而是因为大模型训练同时撞上了单卡显存、单卡算力和单算子尺寸三个边界。理解 TP 前，要先把它和 DP、ZeRO/FSDP 的职责拆开。

### 单卡为什么放不下大模型

一个 Transformer 训练 step 的显存不只是参数。真实显存通常包括：

- model parameters
- gradients
- optimizer states, Adam 通常有 `m` 和 `v`
- forward activations
- attention 临时 buffer
- communication buckets
- framework allocator fragmentation

以 BF16 参数 + FP32 Adam 状态粗略估算，单个参数可能对应 2 bytes parameter、2 bytes gradient、4 bytes master weight、8 bytes Adam states，合计十几 bytes 级别。模型从 7B 到 70B 再到 400B，单卡显存很快不是“紧一点”，而是根本不可能。

### 为什么 Data Parallel 不够

Data Parallel, DP, 做的是复制模型、切分 batch、同步梯度。它提升吞吐，但不降低单卡模型副本的大小。普通 DDP 下每张 GPU 都有完整参数和 optimizer state，所以如果单卡放不下一个模型副本，DP 无法解决。

ZeRO/FSDP 可以减少 DP 状态冗余，但它们主要解决“训练状态如何分片”，不是“单个超大矩阵乘如何在多卡上执行”。当一个 layer 的 hidden size、FFN size、vocab projection 或 attention projection 太大时，仍然需要 TP 把算子本身拆开。

### 为什么需要把一个算子拆到多张 GPU

Transformer 里最重的计算是 GEMM：

- Attention: Q/K/V projection, output projection
- MLP: up projection, gate projection, down projection
- Vocab projection / cross entropy

这些 GEMM 既吃显存，也吃算力。TP 的思路是保留 GEMM 的规则性，但把权重矩阵按维度切开，让每张 GPU 做其中一块。与把每个 op 乱切相比，Megatron-LM 的 TP 只在少数稳定边界通信，工程可控性更好。

## 2. Megatron-LM 的核心思想

Megatron-LM 的关键贡献是把 Transformer 中的大 Linear 成对切分。NVIDIA Megatron Core 现在仍把 `tensor_parallel` 作为 Transformer 模型并行的核心包，官方文档也把它与 Megatron-LM 论文和 activation recomputation 论文关联起来。

### Column Parallel Linear

假设 Linear 写成：

```text
Y = X A
```

其中 `X` shape 约为 `[tokens, hidden]`，`A` shape 约为 `[hidden, output]`。Column Parallel 是沿 `A` 的 output 维切：

```text
A = [A1, A2, ..., Ap]
Yi = X Ai
Y = concat(Y1, Y2, ..., Yp)
```

每个 TP rank 持有一部分输出列，只算自己的 `Yi`。如果下游可以消费 shard output，就不立刻 all-gather；如果下游需要完整 hidden，就 all-gather。Megatron 的工程价值就在这里：尽量让成对算子之间传 shard，避免每个 Linear 后都聚合。

### Row Parallel Linear

Row Parallel 是沿 `A` 的 input 维切：

```text
X = [X1, X2, ..., Xp]
A = [A1; A2; ...; Ap]
Yi = Xi Ai
Y = sum(Y1, Y2, ..., Yp)
```

每个 rank 做 partial output，最后需要把 partial sum 合起来。这个合并通常是 TP group 内的 all-reduce 或等价 reduce。

### MLP 中如何切分

Transformer MLP 常见形态：

```text
H -> 4H / intermediate -> H
```

Megatron 风格切法：

1. 第一个 projection 用 Column Parallel，把 intermediate hidden 按列切开。
2. activation 函数在 shard 上本地执行。
3. 第二个 projection 用 Row Parallel，把 shard intermediate 投回 hidden。
4. Row Parallel 的输出需要 all-reduce，得到完整 hidden 或进入后续并行区域。

SwiGLU/GEGLU 这类 gated MLP 只是多一个 gate projection，本质仍是 Column Parallel 产出 shard intermediate，然后 Row Parallel 合并。

### Attention 中 QKV / Output Projection 如何切分

Attention 通常包括：

```text
X -> QKV projection -> attention -> output projection
```

常见切法：

- QKV projection 用 Column Parallel：按 head 或 hidden shard 切分 Q/K/V。
- 每个 rank 只计算自己负责的 attention heads。
- Attention 内部 softmax/value aggregation 在本地 head shard 上做。
- Output projection 用 Row Parallel：把各 rank 的 head shard 投回 hidden，并通过 all-reduce 合并 partial output。

这个设计非常自然：multi-head attention 本来就有 head 维度，按 head 切分可以减少跨 rank 依赖。但要注意 GQA/MQA/MLA 这类 attention 变体可能改变 Q/K/V 的切分粒度，不能机械套老配置。

### 为什么需要 AllReduce / ReduceScatter / AllGather

TP 的通信来自数学依赖：

- Column Parallel 后，如果后续需要完整输出，就要 all-gather。
- Row Parallel 后，每个 rank 只有 partial sum，就要 all-reduce。
- Sequence Parallel 打开后，部分 activation 沿 sequence 维切分，forward 可能需要 all-gather，backward 可能用 reduce-scatter。
- Vocab Parallel cross entropy 需要在 vocab shard 间处理 logits max/sum 等归约。

工程判断：通信不是 TP 的副作用，而是 TP 语义的一部分。配置 TP 时必须同时估算 compute saving 和 collective cost。

## 3. TP 通信模式

TP 的通信要按 forward、backward 和 overlap 三个层面看。只知道“有 all-reduce”不够，生产排障时必须知道 collective 出现在第几类层、哪个方向、是否和 GEMM 重叠。

### Forward 通信

典型 Transformer block 中，forward 通信集中在：

- Row Parallel Linear 输出合并：常见 all-reduce。
- Column Parallel 需要完整输出时：all-gather。
- Sequence Parallel 下，某些 op 前需要 gather sequence shard。
- Vocab Parallel logits/cross entropy 可能有 max/sum reduction。

如果只看 profiler 里的 GEMM 时间，会低估 TP 的成本。TP size 越大，单卡 GEMM 变小，但 collective 次数不一定变少，通信延迟占比会上升。

### Backward 通信

Backward 通信主要来自：

- input gradient 的 all-reduce 或 reduce-scatter
- weight gradient 计算前后的同步
- sequence parallel 下的 activation gradient reduce-scatter
- DP/FSDP 的 gradient/state 通信与 TP 通信交织

Megatron Core 源码里有异步 all-reduce / reduce-scatter 与 weight gradient GEMM 重叠的路径；这说明生产优化不是“通信少一点”这么简单，而是要让通信在正确 stream 上与计算重叠。

### AllReduce 发生在哪里

最典型的 all-reduce 在 Row Parallel Linear 之后，因为每个 rank 只算出输出的一部分贡献。MLP down projection 和 Attention output projection 都属于这个模式。

如果 sequence parallel 打开，一些原来的 all-reduce 会变成 reduce-scatter/all-gather 组合，以换取 activation 显存下降。此时 profiler 里 collective 名称变了，但本质仍是 TP rank 间交换 partial tensor。

### TP size 增大后通信瓶颈如何变化

TP size 增大有三个效果：

- 单卡参数和 GEMM 规模下降。
- TP collective 参与 rank 增多，通信延迟和带宽压力上升。
- 单卡 GEMM 可能变得太小，Tensor Core 利用率下降。

所以 TP=8 不一定比 TP=4 快。典型坏信号是：GPU 利用率下降、NCCL 时间占比上升、step time p99 变差、GEMM kernel 变碎、DP 扩展反而更好。

## 4. TP 与硬件拓扑

TP 对硬件拓扑比 DP 更敏感，因为它把每层内部依赖变成了跨 GPU collective。拓扑放错时，模型可能还能跑，但吞吐和稳定性会明显变差。

### TP group 最好放在 NVLink / NVSwitch 域内

TP 通信发生在每层，频率远高于 DP 梯度同步。它对 latency 和 bandwidth 都敏感，所以 TP group 最好放在单节点高速互联内：

- 8x H100/H800 NVSwitch 节点：TP=8 常见，但仍要看模型形状。
- 4-GPU NVLink island：TP=4 更自然。
- PCIe-only 节点：TP 需要更谨慎，可能 TP=2 都会明显吃通信。

### 跨节点 TP 为什么危险

跨节点 TP 把每层 collective 放到 IB/RoCE 网络上，代价会被层数和 micro-batch 次数放大。危险不只是平均慢，还有 tail latency：

- 某个 rank 网络抖动会拖住整个 TP group。
- 小 tensor collective 更受 latency 影响。
- TP collective 与 DP/FSDP/MoE 通信抢网络。
- NCCL hang 排障难度显著增加，因为问题可能来自 rank mapping、网卡、交换机、路由、GID、RoCE PFC/ECN 配置或其他作业干扰。

### IB / RoCE 环境下 TP 的代价

IB/RoCE 更适合承载 DP/FSDP 这类较低频的大块通信，或者 pipeline stage 间 p2p。把 TP 放到跨节点网络上，等于把每层内部同步暴露给网络尾延迟。

只有在模型单层太大、节点内 TP 不够时，才考虑跨节点 TP。此时要做两件事：第一，用 profiler 证明瓶颈可接受；第二，固定 rank mapping，让同一 TP group 尽量走最短、最稳定路径。

### TP 与 PP / DP / FSDP 的组合方式

常见组合：

- TP + DP：单层切分 + batch 扩展，适合中等规模 dense model。
- TP + PP + DP：大 dense model 经典 3D parallel。
- TP + FSDP/ZeRO：TP 解决单层算子，FSDP/ZeRO 解决状态冗余。
- TP + EP + DP：MoE 模型常见，TP 切 expert 或 dense block，EP 切 expert placement。
- TP + CP：长上下文训练中，TP 切 hidden/head，CP 切 sequence/context。

配置原则：把最频繁、最细粒度的通信放在最快拓扑内。通常优先级是 TP/EP 节点内，PP 可以跨节点，DP/FSDP 跨更多节点。

## 5. 生产环境配置建议

TP size 的选择本质是一个 trade-off：用更多通信换更小单卡 GEMM 和更低显存。不要从模型参数量直接推 TP，而要同时看 hidden size、head 数、seq length、micro-batch、节点拓扑和目标 DP degree。

### TP size 选择

| TP size | 适合场景 | 风险 |
|---|---|---|
| TP=1 | 小模型、FSDP/ZeRO 足够、省通信 | 单卡放不下时不可用 |
| TP=2 | 13B-70B 某些配置、PCIe 或弱 NVLink 环境 | 收益可能被通信抵消 |
| TP=4 | 很常见的折中，适合 8 卡节点内留出 DP/PP 组合空间 | hidden/head 数必须可整除 |
| TP=8 | 单节点 NVSwitch、大 hidden dense model、较大 GEMM | DP degree 下降，collective 更重，跨节点更危险 |

### 什么时候优先扩大 DP

优先扩大 DP 的条件：

- 单卡或小 TP 已能放下模型。
- GEMM 足够大，单卡计算效率高。
- 网络足以支撑梯度同步。
- 目标是提高吞吐，而不是解决单层容量。

DP 扩大也不是免费：global batch 会变，需要调 learning rate、warmup、gradient accumulation 和数据 shuffle。

### 什么时候引入 PP

引入 PP 的信号：

- 层数多，总参数/activation 超出节点内 TP 能力。
- TP 继续增大导致通信过重。
- 可以把 layer 较均匀地分 stage。
- 有足够 micro-batch 隐藏 pipeline bubble。

PP 的代价是调度复杂、bubble、stage imbalance、activation send/recv 和 checkpoint metadata 更复杂。

### 什么时候 TP 会拖慢训练

TP 可能拖慢训练的场景：

- TP group 跨节点。
- hidden size/head 数太小，切完 GEMM 太碎。
- TP 与 EP/FSDP 通信同时打满网络。
- rank mapping 把 TP group 分散到不同 PCIe root complex 或不同机架。
- sequence parallel/async overlap 配置不正确。
- micro-batch 太小，通信 latency 无法被 compute 掩盖。

## 6. 常见故障与排障

TP 故障通常表现为 NCCL、shape 或性能问题，但根因可能在 rank mapping、并行配置、网络、checkpoint 或模型结构。排障时先确认 group 和 shape，再看网络和性能。

### NCCL Hang

典型现象：训练卡住、无报错、NCCL timeout、某些 rank 停在 collective。

排查顺序：

1. 确认所有 rank 的 TP/PP/DP group 构造一致。
2. 检查是否有某个 rank 在 collective 前 shape mismatch 或提前异常。
3. 打开 NCCL debug 和框架 debug，定位卡住的 collective。
4. 检查 rank mapping 是否把 TP group 跨了非预期节点。
5. 检查网络健康、IB/RoCE counters、链路错误和其他作业干扰。

### TP group 配错

典型现象：loss 异常、shape 对不上、rank 数不整除、checkpoint load 后权重错位。

重点检查：

- `world_size % (tp * pp * dp)` 是否成立。
- hidden size、num heads、kv heads、intermediate size 是否可被 TP 整除。
- expert TP 与 dense TP 是否混用同一 group。
- checkpoint metadata 中 TP size 是否与当前运行一致。

### Rank mapping 错误

典型现象：TP=4 本该节点内，却跨机器通信；同一节点 GPU 利用不均；NCCL topo 显示路径绕远。

排查建议：

- 打印每个 global rank 的 hostname、local rank、CUDA device、TP group id。
- 对照 scheduler 分配和 NCCL topo。
- 用小规模 all-reduce benchmark 验证 group 内带宽。

### 跨节点通信过多

典型现象：step time 随节点数增加非线性变差，NCCL 占比显著上升。

处理思路：

- 把 TP group 收回节点内。
- 增大 PP/DP，而不是继续增大 TP。
- 调整 rank order，让 TP 邻近。
- 检查是否 MoE all-to-all 与 TP all-reduce 叠加。

### GPU 利用率低

可能原因：

- TP 切得太细，GEMM 太小。
- micro-batch 太小。
- communication overlap 没生效。
- data pipeline starvation 被误判为 TP 问题。
- pipeline bubble 或 stage imbalance。

先用 profiler 区分 compute idle、NCCL wait、data wait，再改配置。

### Step time 抖动

TP step time 抖动常来自尾延迟：

- 某个 rank 网络慢。
- 某个 GPU ECC/XID/温度降频。
- 共享存储或 dataloader 抖动。
- MoE routing 造成 expert load 不均。
- checkpoint 保存与训练通信抢资源。

不要只看平均 step time。至少看 p50/p95/p99、每层耗时、每个 collective 耗时。

### Activation shape mismatch

常见触发点：

- sequence parallel 开关与 layer 实现不匹配。
- context parallel 后 sequence 维 shard 处理错误。
- QKV head 数不能被 TP 整除。
- GQA/MQA 的 `num_key_value_heads` 与 TP 配置冲突。
- checkpoint 由不同 TP/CP 配置保存，恢复时没有正确 reshard。

## 7. 与其他技术的关系

- [Megatron-LM](../papers/megatron_lm.md)：TP 的经典工程实现，Column/Row Parallel Linear 是主线。
- [Sequence Parallel](sequence_parallelism.md)：在 TP 基础上切 sequence 维 activation，减少显存，但引入 all-gather/reduce-scatter。
- [Context Parallel](context_parallelism.md)：长上下文下切 context/sequence，与 TP 的 hidden/head 切分互补。
- [FSDP](fsdp.md)：FSDP 管训练状态分片，TP 管单层算子切分。两者组合时 checkpoint 和通信 overlap 更复杂。
- [MoE Expert Parallel](moe.md)：EP 切 expert，TP 切 expert 内部或 dense block。TP group 和 EP group 重叠会显著影响 all-to-all 与 all-reduce 的网络压力。
- [FlashAttention](flashattention.md)：FlashAttention 优化 attention kernel 内 IO，TP 优化 head/hidden 维分布。两者一起决定 attention 的单层吞吐。
- [Distributed Checkpointing](checkpointing.md)：TP 保存的是 shard 权重，恢复时必须知道 TP rank、partition dim、offset 和目标 TP size。
- [NCCL](nccl.md)：TP 的生命线。TP 排障多数最终会落到 collective、拓扑和 rank mapping。

## 8. 面试题

TP 面试的核心不是背概念，而是能从矩阵切分推导通信，从通信推导拓扑要求，再从拓扑要求推导生产配置和排障路径。

### 基础题

1. Tensor Parallelism 和 Data Parallelism 的区别是什么？
2. 为什么普通 DDP 不能解决单卡放不下模型的问题？
3. Column Parallel Linear 切的是哪个维度？
4. Row Parallel Linear 为什么需要 all-reduce？
5. Transformer MLP 中两个 Linear 通常如何配合切分？
6. Attention 的 QKV projection 为什么适合按 head/hidden 切？
7. TP size 必须满足哪些整除条件？
8. TP group 是什么？
9. TP 和 Model Parallel 是什么关系？
10. TP checkpoint 与普通 checkpoint 有什么不同？

### 进阶题

1. 为什么 Column Parallel 后不一定马上 all-gather？
2. Sequence Parallel 如何改变 TP 的通信模式？
3. GQA/MQA 会如何影响 TP 配置？
4. TP size 增大为什么可能降低 Tensor Core 利用率？
5. TP all-reduce 与 DP all-reduce 的频率差异是什么？
6. TP 如何与 activation recompute 配合？
7. Vocab Parallel cross entropy 为什么需要跨 rank reduction？
8. Megatron 中异步 grad all-reduce 的收益来自哪里？
9. TP 和 FSDP 组合时参数生命周期如何变复杂？
10. TP 与 pipeline parallel 的切分边界如何选择？

### 生产环境题

1. TP=8 跨两个节点后 step time 抖动，如何排查？
2. 训练卡在 NCCL all-reduce，如何判断是 group 配错还是网络问题？
3. rank mapping 错误会造成哪些现象？
4. TP group 内某张 GPU 降频会如何影响整体训练？
5. 如何验证 TP group 是否真的在 NVSwitch 域内？
6. TP 开大后 GPU 利用率下降，你会看哪些 profiler 指标？
7. checkpoint 从 TP=8 恢复到 TP=4 需要哪些 metadata？
8. sequence parallel 打开后 shape mismatch，如何定位？
9. MoE 训练中 TP all-reduce 和 EP all-to-all 互相抢网络，如何缓解？
10. 如何设计 TP 配置的灰度验证？

### 系统设计题

1. 给定 70B dense model 和 8xH100 节点，如何选择 TP/PP/DP？
2. 给定 405B dense model，如何规划 3D parallel 和 checkpoint？
3. 给定 671B MoE 模型，TP、EP、DP 的优先级如何排？
4. 设计一个自动并行配置搜索器，需要哪些输入和指标？
5. 如何在 scheduler 层保证 TP group 拿到正确拓扑？
6. 如何设计 TP-aware distributed checkpoint 格式？
7. 如何构建 TP collective 的监控面板？
8. 如何设计跨节点 TP 的风险评估实验？
9. 长上下文训练中 TP 与 CP 如何组合？
10. 如何在多租户 GPU 集群中减少 TP 作业被网络噪声影响？

## 9. 生产环境思考题

1. 如果 TP=8 跨两个节点部署，哪些 collective 会最先暴露问题？
2. 如果 NVLink 拓扑不是全互联，TP group 应该如何放置？
3. 如果 TP group 和 MoE EP group 重叠，all-reduce 与 all-to-all 会如何互相影响？
4. 如果 TP=4 时吞吐高于 TP=8，应该如何解释？
5. 如果某次升级 FlashAttention 后 TP step time 变差，如何判断是 kernel 还是通信？
6. 如果 checkpoint 恢复后只有 TP rank 2 的权重异常，如何定位 shard offset？
7. 如果 pipeline stage 内 layer 数不均，TP 能否掩盖 stage imbalance？
8. 如果 DP 扩大导致 global batch 太大，是否应该转而增大 TP？
9. 如果 GQA 的 kv heads 小于 TP size，会发生什么？
10. 如果 TP group 内 rank 的 hostname 不连续，是否一定有问题？
11. 如果 NCCL timeout 只发生在 backward，优先看哪些 TP 通信点？
12. 如果 micro-batch 从 4 降到 1，TP 通信占比会如何变化？
13. 如果 sequence parallel 降低显存但 step time 变差，如何决定是否保留？
14. 如果训练平台支持弹性恢复，TP size 是否应该允许动态变化？
15. 如果 RoCE 集群 PFC/ECN 配置不稳，为什么 TP 比 DP 更容易受伤？

## 参考入口

- [Megatron-LM 论文](../papers/megatron_lm.md)
- [Megatron Core tensor_parallel 官方文档](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/tensor_parallel.html)
- [Megatron-LM tensor_parallel 源码](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/tensor_parallel)
- [NCCL Topic](nccl.md)

# NCCL 与分布式通信算子：语义、场景与排障

## 问题框架

分布式训练里的通信题不能只背 API 名。工程上真正需要回答的是：

1. 每个 rank 输入什么，最终在哪些 rank 得到什么；
2. 传的是 gradient、parameter、activation、KV，还是 routed token；
3. 哪个 process group 参与，count/dtype/调用顺序是否一致；
4. 通信能否与计算重叠，是否只是把时间藏进了另一个瓶颈；
5. hang、慢或数值异常时，如何找到 first divergence。

本文以 **NCCL 2.31.2** 的 host API 为版本基线。Broadcast、Reduce、AllReduce、Gather、Scatter、AllGather、ReduceScatter、AllToAll 均有对应 host collective API；`ncclAlltoAll`、`ncclGather`、`ncclScatter` 是 NCCL 2.28.3 引入的较新 host collective API。框架文档中的 Barrier、AllToAllV 等语义不能无条件等同为同名 NCCL host API。

<a id="collective-map"></a>
## 1. 先区分 Collective 与 P2P

- **Collective**：communicator/process group 中所有参与 ranks 必须以一致顺序进入同一个 collective，输入输出由 group 语义共同定义。
- **Point-to-point**：Send/Recv 指定 peer，适合 PP stage 邻接传输、ring KV exchange 或不规则协议；多组 P2P 往往使用 grouped calls 提交。
- **Communicator/process group**：定义参与 ranks 及 rank 顺序。同一 global rank 可以同时属于 TP、DP、CP、EP 等多个 group。
- **Root**：Broadcast、Reduce、Gather、Scatter 的特殊 rank；root 的 send/recv buffer layout 通常与非 root 不同。

通信正确性的第一原则是：**先确定 group 和 tensor 语义，再选择算子。**

## 2. 常见 Collective 一览

设 group size 为 `N`，每个 rank 为 `r`。

| 算子 | 输入 → 输出 | 哪些 rank 得到完整结果 | 典型场景 |
| --- | --- | --- | --- |
| Broadcast | root 的一份 tensor → 所有 ranks 同一 tensor | 所有 ranks | 初始化参数、配置/metadata 分发 |
| Reduce | 所有 ranks tensor 按 op 规约 → root | 仅 root | 只需 root 汇总的统计量 |
| AllReduce | 所有 ranks tensor 规约 → 每个 rank | 所有 ranks | DDP gradient 同步、TP partial sum |
| Scatter | root 的 `N` 个分片 → 每 rank 一个分片 | 每 rank 仅自己的 shard | root 分发固定大小分片 |
| Gather | 每 rank 一个分片 → root 按 rank 拼接 | 仅 root | root 收集固定大小结果 |
| AllGather | 每 rank 一个分片 → 每 rank 得到全部分片 | 所有 ranks | 重建 sharded parameter/activation |
| ReduceScatter | 先逐元素规约，再把结果分成 `N` 片 | 每 rank 一个 reduced shard | sharded gradient、SP activation |
| AllToAll | 每 rank 向每个 peer 发送一个等长分片 | 每 rank 收到来自所有 peers 的分片 | MoE token exchange、layout transpose |
| Send/Recv | sender tensor → 指定 receiver | receiver | PP activation/gradient、ring CP |

`sum`、`max`、`min` 等 reduction op 只适用于 Reduce、AllReduce、ReduceScatter 一类规约算子；Gather/AllGather/AllToAll 只搬运和重排数据，不做数值规约。

## 3. 每个算子到底做了什么

### 3.1 Broadcast：一份复制到所有 ranks

```text
before: rank0=[A], rank1=[?], rank2=[?], rank3=[?]
root=0 broadcast
after:  rank0=[A], rank1=[A], rank2=[A], rank3=[A]
```

它解决“一份权威数据要复制给所有参与者”，不聚合其他 rank 的输入。常见于模型初始化、随机种子/控制 metadata 或从 root 发布小状态。它不是 AllGather：AllGather 会保留每个 rank 的不同输入。

### 3.2 Reduce：多份规约到 root

```text
rank0=[1], rank1=[2], rank2=[3], rank3=[4]
Reduce(sum, root=0) -> rank0=[10]
```

只在 root 需要总和、最大值等结果时使用。若所有 rank 后续都需要结果，直接 AllReduce 通常比“Reduce 后再手动处理”更自然。

### 3.3 AllReduce：所有人得到同一规约结果

```text
rank r input: xr
every rank output: y = reduce(x0, x1, ..., xN-1)
```

典型用途：

- DDP 中复制参数的 gradient 同步；
- TP Row Parallel Linear 的 partial output 求和；
- 标量 loss、overflow flag 或统计量在 group 内一致化。

成本直觉：ring AllReduce 每 rank 的大消息数据量近似 `2×(N-1)/N×message_size`，但真实性能还取决于拓扑、算法/协议、channel、消息大小和并发流。

### 3.4 Scatter：root 把不同分片发给不同 ranks

```text
root input: [A0 | A1 | A2 | A3]
after: rank0=A0, rank1=A1, rank2=A2, rank3=A3
```

它适合 root 已经拥有完整、固定大小的 rank-ordered buffer。分布式训练的大 tensor 通常本来就在各 rank 上，因此 Scatter 不像 AllReduce/AllGather 那么常见。

### 3.5 Gather：不同分片只收集到 root

```text
before: rank0=A0, rank1=A1, rank2=A2, rank3=A3
after on root: [A0 | A1 | A2 | A3]
```

适合 root 汇总结果、调试或保存小规模 metadata。大模型中无节制 Gather 到 root 会制造 root 内存和带宽瓶颈。

### 3.6 AllGather：每个人都重建完整数据

```text
rank r input: Ar
every rank output: [A0 | A1 | ... | AN-1]
```

典型用途：

- Distributed Optimizer 更新 shard 后重建参数视图；
- FSDP 在计算前 materialize 当前 module 的 full parameter；
- TP/SP 在需要完整 activation 的算子前重建 sequence/feature；
- 收集 variable-length 数据时，框架层通常先交换 sizes，再 pad 或使用支持变长的接口。

AllGather 增加每 rank 瞬时显存。OOM 不一定发生在计算层，也可能发生在 parameter/activation materialization 阶段。

### 3.7 ReduceScatter：规约后每个人只保留一片

若每个 rank 输入都逻辑分成 `N` 个 chunks：

```text
rank r input: [xr,0 | xr,1 | ... | xr,N-1]
rank k output: reduce(x0,k, x1,k, ..., xN-1,k)
```

典型用途：

- Megatron Distributed Optimizer 让每 rank 得到自己负责的 reduced gradient shard；
- FSDP 对分片参数的 gradient 做规约并重新分片；
- Sequence Parallel 用 ReduceScatter 将规约后的 activation/gradient 同时切到 TP ranks。

它与 AllReduce 的关键差别不是“少算一次”，而是输出布局不同：AllReduce 每 rank 得到完整结果，ReduceScatter 每 rank 只得到一个 shard。

### 3.8 AllToAll：每个人都给每个 peer 不同数据

将每 rank 输入切成 `N` 个目的分片：

```text
rank i sends chunk[i,j] to rank j
rank j receives chunk[0,j], chunk[1,j], ..., chunk[N-1,j]
```

它常用于：

- MoE 根据 expert destination dispatch token，再反向 combine；
- Ulysses 类 sequence/head layout transpose；
- 某些 CP 或 dispatcher 的维度重排。

AllToAll 难优化的原因不是 API 特殊，而是每个 peer 的数据量可能动态且不均。fixed-count NCCL AllToAll 要求每 peer 固定 count；MoE dropless routing 常需要框架/dispatcher 的 variable-count exchange。

### 3.9 Send/Recv：明确 peer 的数据通道

PP 中，stage `i` forward Send activation 到 `i+1`，backward 接收 gradient；CP ring 中，每步把 KV block 发给相邻 rank。NCCL `ncclSend/ncclRecv` 没有 tag 参数；NCCL P2P 的 communicator、peer、count、dtype 或调用顺序不兼容时很容易 deadlock。NCCL grouped P2P 还需要所有参与 rank 保持兼容的批次调用顺序。其他 backend/framework 即使暴露 tag，也不能把 tag 语义套到 NCCL host API 上。

## 4. AllToAllV 与 Barrier 的边界

### AllToAllV

AllToAllV 表示每个 `(src,dst)` pair 可以有不同 count。MoE 的实际 token routing常符合这个语义，但 NCCL 2.31.2 没有通用的 `ncclAlltoallv` host collective API。框架或 dispatcher 可以用多组 `ncclSend/ncclRecv`、padding 到 fixed count，或专用通信库实现。

正确性必须满足：

```text
sum(send_splits[rank]) == local_send_elements
sum(recv_splits[rank]) == local_recv_elements
send_count[src][dst] == recv_count[dst][src]
```

只检查本 rank 总量相等不够；任意 peer pair 不一致都可能 hang、越界或静默错位。

### Barrier

PyTorch `dist.barrier()` 表达“process group 中所有 ranks 到达后才能继续”的框架同步语义。NCCL 2.31.2 没有与之完全对应的通用 host collective Barrier API；后端可用小 tensor collective 等方式实现同步效果。

不要把 Barrier 当成修复 race/hang 的万能方法。它会把上游不一致转移到新的等待点，并可能破坏 overlap。正确做法是定义数据依赖和 stream/event，再用 barrier 做必要的阶段边界或诊断。

## 5. 常见数学等价为什么不能滥用

### `AllReduce = ReduceScatter + AllGather`

在 count 可均匀分片、dtype/reduction op 一致、布局兼容时，它们具有数学语义等价：先得到 reduced shards，再收集为完整结果。高性能实现可以采用 reduce-scatter/all-gather phases，但不能据此断言底层机械调用了两个 host API。

### `AllReduce = Reduce + Broadcast`

数学上可先把规约结果放到 root，再广播。但 topology、算法、buffer layout 和性能可能不同。

### 为什么不保证 bitwise 一致

浮点加法不满足严格结合律。不同 ring/tree、chunk、rank order 或 fusion 会改变归约顺序，因此结果可能在容差内一致但非 bitwise identical。Numeric validation 应定义 dtype 对应的误差阈值，而不是无条件要求逐 bit 相同。

## 6. 5D 并行中的通信映射

| 系统位置 | 典型数据 | 常见通信 | 为什么 |
| --- | --- | --- | --- |
| classic DP/DDP backward | gradient | AllReduce | 所有复制参数 ranks 需要相同 gradient |
| Megatron Distributed Optimizer | gradient → parameter | ReduceScatter → local update → AllGather | 每 rank 只更新 state shard，随后重建参数视图 |
| FSDP FULL_SHARD | parameter → gradient | pre-forward AllGather；可选 post-forward reshard；pre-backward AllGather；post-backward ReduceScatter/reshard | 仅计算时 materialize full parameter，之后恢复分片 |
| TP Linear | activation/partial result | AllReduce、AllGather、ReduceScatter | 合并层内分片结果或改变 layout |
| SP | activation/gradient | AllGather、ReduceScatter | 在 TP group 内重建/分摊 sequence activation |
| PP | activation/gradient | Send/Recv | 相邻 stage 传递边界 tensor |
| CP Attention | KV / layout | P2P、AllGather、AllToAll | local Q 访问全局 context |
| EP MoE | routed tokens | AllToAll/variable-count exchange | token 去目标 expert 并返回 |

FSDP 路径取决于 sharding strategy、reshard policy、prefetch 和模块粒度；上表描述 FULL_SHARD 的典型生命周期，不应当作所有 FSDP 配置的唯一调用序列。

完整并行框架见 [Megatron 5D 并行](distributed_training.md)，Parallel Folding 与 EP 数据流见 [MoE](moe.md#parallel-folding)。

## 7. 性能判断：看 exposed time，不只看通信总时长

### Latency 与 bandwidth

- 小消息更受 launch、同步和网络 latency 影响；
- 大消息更接近链路 bandwidth 上限；
- group 越大不代表必然更慢，算法和拓扑会改变并行路径；
- p50 正常、p99 抖动常来自 rank/host/NIC straggler 或动态 count imbalance。

### Ring、Tree 与 topology

Ring 通常适合大消息带宽利用，Tree 类算法可能降低小消息或大规模 group 的步骤深度；NCCL 会根据 topology、消息和版本选择算法/协议，也可通过环境变量做诊断性强制。生产结论必须来自目标硬件实测，不能把“ring 一定更快”当定理。

### Overlap

异步发起不等于真正隐藏：

```text
exposed_comm = communication_end - max(communication_start, dependent_compute_end)
```

需要检查：

- compute 与 communication 是否真无依赖；
- stream priority、event/wait 是否正确；
- NCCL kernel 是否与 GEMM 争 SM；
- 两者是否同时打满 HBM 或 NIC；
- bucket/chunk 太小是否增加 launch，太大是否缩小隐藏窗口。

## 8. 正确性不变量

每次 collective 至少核对：

1. **group membership**：所有参与 rank 对 communicator 和 rank order 的理解一致；
2. **call order**：同一 group 上 collective 顺序一致；
3. **count/shape**：底层元素数匹配，不能只看高层 shape 名称；
4. **dtype/op/root/peer**：所有参与者的协议参数兼容；
5. **buffer lifetime**：异步完成前 input/output buffer 不能被复用；
6. **stream dependency**：consumer 必须等待通信完成，producer 必须先产出数据；
7. **layout metadata**：gather/scatter/all-to-all 后的 rank order、offset 和 inverse mapping 正确。

“能跑完”并不证明 numeric 正确。需要 tiny deterministic tensor、手算 expected output、single-rank/多-rank对照和端到端 loss validation。

<a id="hang-diagnosis"></a>
## 9. NCCL hang 的生产排查顺序

### 第一步：保存现场并找 first bad event

记录 job、rank、host、GPU、NIC、topology、最近 checkpoint、collective sequence 和各 rank 最后进度。不要只看最后报 timeout 的 rank；它往往只是等待先失败的 peer。

### 第二步：验证代码/协议一致性

- 是否某 rank 因数据分支跳过 collective；
- count/dtype/group/root/peer 是否一致；
- gradient accumulation、empty batch、MoE variable splits 是否导致不同调用序列；
- 是否有 rank 提前 OOM、assert、CUDA illegal access 或进程退出。

### 第三步：检查 stream 和异步错误

确认 producer event、collective stream、consumer wait，打开 PyTorch/NCCL 的 async error 与 flight recorder 类诊断能力；查看 watchdog 报告的 sequence number，向前追第一处不一致。

### 第四步：检查硬件与网络

查看 GPU Xid/ECC、NVLink、NIC port、IB/RoCE counter、packet retry/drop、交换机和拓扑变化；做单节点、节点对和目标 group 的最小通信测试以缩小故障域。

### 第五步：验证修复

同一 workload 回归功能、numeric、p50/p99、长稳和故障恢复。单次不再 hang 不能证明修复完成。

## 10. 面试回答模板

### 30 秒版本

> AllReduce 是所有 rank 规约后都拿完整结果，常用于 DDP gradient 或 TP partial sum；ReduceScatter 是规约后每 rank 只拿一片，适合 sharded gradient；AllGather 把各 rank 分片重建到所有 rank，常用于参数或 activation materialization；AllToAll 是每 rank 给不同 peer 发不同数据，典型是 MoE token dispatch；PP 常用 Send/Recv。判断算子不能只背名字，要同时说传的 tensor、process group、输出布局和生命周期。

### 3–5 分钟展开顺序

1. 先按“复制、规约、分片、全交换、P2P”分类；
2. 用一个四 rank 例子说明输入输出；
3. 映射 DP/TP/PP/CP/EP、Distributed Optimizer/FSDP；
4. 说明 `RS+AG` 的条件等价和 floating reduction 边界；
5. 最后讲 group/order/count/stream 四类 correctness 与 hang 排障。

### 常见错误回答

- 把 Broadcast 和 AllGather 都说成“复制数据”而不区分输入来源；
- 把 ReduceScatter 说成“先 reduce 到 root 再 scatter”，却说不清每个 rank 的输出 shard；
- 把 gradient ReduceScatter 与 parameter AllGather 的 tensor 类型说反；
- 认为所有 AllToAll 都支持任意 variable splits；
- 用 Barrier 掩盖错误的数据依赖；
- 认为异步 API 返回就代表通信已经完成。

## 相关材料

- [Megatron 5D 并行](distributed_training.md)
- [MoE 与 Parallel Folding](moe.md#parallel-folding)
- [Tensor Parallelism 面试题：通信算子](../interview/tensor_parallelism.md#communication-operators)
- [MegaScale](../tech_reports/megascale.md)

## 参考资料

- [NCCL 2.31.2 Collective Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)
- [NCCL API](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api.html)
- [NCCL 2.28.3 Release Notes](https://docs.nvidia.com/deeplearning/nccl/release-notes/rel_2-28-3.html)
- [PyTorch Distributed collectives](https://docs.pytorch.org/docs/stable/distributed.html)
- [PyTorch FSDP](https://docs.pytorch.org/docs/stable/fsdp.html)

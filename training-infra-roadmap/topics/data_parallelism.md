# Data Parallelism

5D 组合入口：[Megatron 5D 并行总览](distributed_training.md)。

## 核心问题

复制模型到多张 GPU，切分 batch，反向后同步梯度。它是吞吐扩展的基础，但会复制参数、梯度和 optimizer state。

## 上游概念

- [ZeRO](../papers/zero.md)
- [FSDP](fsdp.md)

## 关键机制

- all-reduce gradients
- gradient accumulation
- global batch size
- optimizer state replication 或 sharding

<a id="dp-concept-and-implementations"></a>
## DP 是策略，DataParallel/DDP/FSDP/Megatron DP 是实现

先区分三个层次：

```text
算法策略：Data Parallelism
    │
    ├── PyTorch API：nn.DataParallel、DistributedDataParallel、FSDP
    │
    └── 大模型框架中的逻辑维度：Megatron DP group
```

一句话结论是：

> Data Parallelism 是一种并行策略；PyTorch 的 `nn.DataParallel`、`DistributedDataParallel`、FSDP，以及 Megatron 的 DP group，都是这种策略在不同规模和 model-state 分片方式下的实现。它们共享“不同副本处理不同数据、聚合同一参数的梯度”这一数学语义，但进程模型、参数放置和通信路径不同。

### 1. Data Parallelism 的不变语义

无论使用哪个框架，数据并行的基本目标都是：

```text
相同的逻辑模型
    + 不同的数据分片
    + 聚合同一参数的梯度
    = 等价的 global batch 训练
```

以两个数据副本为例：

```text
replica 0: parameters θ + batch A -> local gradient g0
replica 1: parameters θ + batch B -> local gradient g1

global gradient = (g0 + g1) / 2
```

真正发生变化的是：一个 replica 是否必须在一张 GPU 上保存完整模型、哪些 model states 被复制或分片，以及聚合结果是否需要在每个 rank 上完整存在。

### 2. `torch.nn.DataParallel`：单进程多 GPU

早期 PyTorch 常见写法是：

```python
model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
```

它通常由一个 Python 进程控制多张 GPU：

```text
主进程 / 主 GPU
    -> scatter input
    -> forward 时把 module replicate 到各 GPU
    -> 各 GPU 处理一个 batch chunk
    -> backward 时把 replica gradients 求和到原始 module
    -> 更新主 module
```

主要特点是：

- API 自动沿 batch dimension 切分 tensor input；
- 原始 module 位于 `device_ids[0]`，其他设备使用 forward replica；
- 输出汇总、梯度累加和原始参数放置使主 GPU 承担额外负载；
- 单进程控制、多设备 replication 和主卡不均衡限制了扩展性；
- 通常只用于单机，PyTorch 当前建议即使单机多卡也优先使用 DDP。

这是一种 Data Parallelism 实现，但“PyTorch `DataParallel`”这个类不等于数据并行概念本身。[PyTorch DataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.DataParallel.html)

### 3. `DistributedDataParallel`：多进程复制式 DP

DDP 通常采用一张 GPU 一个进程：

```text
process 0 -> GPU 0 -> persistent model replica 0
process 1 -> GPU 1 -> persistent model replica 1
process 2 -> GPU 2 -> persistent model replica 2
process 3 -> GPU 3 -> persistent model replica 3
```

每个进程保存完整模型和 optimizer，独立完成 forward/backward；用户通过 `DistributedSampler` 等方式给不同 rank 分发不同数据。参数初始化保持一致后，训练主路径同步的是梯度，而不是每一步重新广播全部参数：

```text
different data
    -> local backward
    -> bucketized gradient AllReduce
    -> every rank obtains the same reduced gradient
    -> every rank performs the same optimizer step
```

DDP 的优势包括：

- 支持单机和多机；
- 不存在集中式主 GPU 汇总所有训练工作的瓶颈；
- gradient buckets ready 后可以异步 AllReduce，与剩余 backward compute 重叠；
- 每个进程拥有独立 autograd engine 和 optimizer，扩展性通常明显优于 `nn.DataParallel`。

因此，`nn.DataParallel` 与 DDP 的关系是：

> 两者实现同一个 Data Parallelism 策略，但前者是单进程主卡式实现，后者是多进程、对等 replica 和 collective 通信实现；它们不是同一个 PyTorch API。[PyTorch DistributedDataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)

### 4. FSDP：逻辑上数据并行，物理分片取决于 sharding strategy

标准 DDP 的每个 rank 都复制完整的 parameters、gradients 和 optimizer states。FSDP 仍然让不同 ranks 处理不同数据并聚合同一逻辑模型的梯度，但可以根据 sharding strategy 分片不同的 model states。只有 `FULL_SHARD` 才在“分什么”以及主要生命周期上对应 ZeRO-3：

```text
标准 DDP：
每个 rank 长期保存完整 parameters / gradients / optimizer states

FSDP FULL_SHARD / ZeRO-3：
每个 rank 长期保存 parameters / gradients / optimizer states 的 shard
计算前临时 AllGather 所需参数
反向后 ReduceScatter 梯度
```

其他策略不能沿用这条完整生命周期：`SHARD_GRAD_OP` 不在 forward 后立即 reshard 参数；`HYBRID_SHARD` 只在子 group 内做 FULL_SHARD、在 group 间复制；`NO_SHARD` 则保留复制式 model states。它们的常驻显存、AllGather 次数、通信 group 和峰值都不同。Megatron-FSDP 中，`optim_grads_params` 才是 ZeRO-3/FULL_SHARD 对应路径，`optim` 与 `optim_grads` 分别接近 ZeRO-1/2。

所以“数据并行”不等于“每张 GPU 必须永久保存完整模型”。更本质的判断标准是：

```text
不同 ranks 是否处理不同数据分片
并共同更新同一个逻辑模型
```

FSDP 改变了 model-state placement 和通信方式，没有改变 Data Parallelism 的训练语义。

### 5. Megatron DP：模型副本本身可以横跨多张 GPU

Megatron 中的 `DP` 不是调用 `torch.nn.DataParallel`，而是 process-group mesh 中的一个逻辑坐标。一个完整逻辑模型副本可能已经由 TP、PP、CP 共同承载：

```text
Dense world_size = TP × PP × CP × DP
```

例如：

```text
TP=2, PP=4, CP=2, DP=8
world_size = 2 × 4 × 2 × 8 = 128
```

一个 model-parallel replica 使用：

```text
TP × PP × CP = 16 GPUs
```

整个作业存在 8 个语义上的数据副本：

```text
128 GPUs
    -> 8 个 DP replicas
    -> 每个 replica 由 TP2 × PP4 × CP2 的 16 GPUs 共同承载
```

纯 DP group 固定 `(tp_rank, pp_rank, cp_rank)`，只改变 `dp_rank`。组内 ranks 持有相同的 model-parallel parameter shard、处理不同数据，因此需要规约对应 shard 的梯度。

这与普通 PyTorch DDP 的数学语义相同，但 replica 粒度不同：

```text
普通 DDP：    一张 GPU 通常是一份完整模型 replica
Megatron DP： 一组 TP×PP×CP GPUs 共同组成一份模型 replica
```

### 6. 为什么现代 DP 不一定只用 AllReduce

通信算子取决于 model states 是否分片，而不是 Data Parallelism 的名字：

| 实现 | model-state 放置 | 主通信路径 |
|---|---|---|
| `nn.DataParallel` | 原始 module 在主设备，forward replicas 在其他设备 | replica gradient 累加到原始 module |
| PyTorch DDP | 每个 rank 复制完整 model states | gradient AllReduce |
| Distributed Optimizer / ZeRO-1 | optimizer state 和更新责任分片 | gradient ReduceScatter + parameter AllGather |
| FSDP FULL_SHARD / ZeRO-3 | parameters、gradients、optimizer states 分片 | parameter AllGather + gradient ReduceScatter |
| Megatron DP | 对相同 model-parallel shard 建 DP group | 根据 DDP/Distributed Optimizer/FSDP 后端选择 AR 或 RS+AG |

因此从早期 DDP 的 AllReduce 演进到 ReduceScatter + AllGather，不是换了一个并行概念，而是利用 Data Parallel ranks 分摊 model states 和 optimizer update。

### 7. CP、EP 会进一步细化“谁和谁同步”

Megatron 的实际通信 group 不总是纯 DP group：

- CP ranks 复制 Dense 参数，但分别处理同一 sequence 的不同 context；完整 Dense 梯度需要合并 DP replicas 与 CP context shards 的贡献，因此当前 Megatron 常使用 `dp_cp` group；
- routed experts 分布在 EP ranks 上，只有持有同一 expert shard 的 EDP ranks 才构成 expert gradient replica group；
- Parallel Folding 下 Dense/Attention 与 routed experts 使用两套逻辑 mesh，不能用一个全局 DDP group 粗暴同步所有参数。

这里必须区分：

```text
语义 DP：处理多少份独立数据，影响 global batch
通信 group：哪些 ranks 复制同一 parameter shard，需要合并梯度
```

开启 CP 后，Dense 通信 group 可能是 `DP × CP`，但 global batch 和 microbatch 数仍只除以语义 DP，因为 CP ranks 处理的是同一批样本的不同 context，而不是更多独立样本。

### 8. 演进关系

```text
Data Parallelism
    │
    ├── nn.DataParallel
    │      单进程、多 GPU、主设备聚合
    │
    ├── DistributedDataParallel
    │      多进程、完整模型副本、gradient AllReduce
    │
    ├── DDP + Distributed Optimizer / ZeRO
    │      optimizer/gradient 分片、ReduceScatter + AllGather
    │
    ├── FSDP FULL_SHARD / ZeRO-3
    │      parameters/gradients/optimizer states 全分片
    │
    └── Megatron DP group
           每个 DP replica 内部再组合 TP/PP/CP/EP
```

### 9. 面试精炼回答

> DP 是数据并行策略，PyTorch 的 `nn.DataParallel`、DDP、FSDP 和 Megatron DP group 都是它的实现。早期 `nn.DataParallel` 是单进程控制多 GPU，由主设备 scatter input、创建 forward replicas 并累加梯度，存在主卡和单进程瓶颈；DDP 通常一张 GPU 一个进程，每个进程保存完整模型副本，通过 bucketized gradient AllReduce 同步，能够扩展到多机。FSDP 保留不同 ranks 处理不同数据的语义，但具体分片和通信由 sharding strategy 决定；只有 FULL_SHARD 才同时分片 parameters、gradients 和 optimizer states，对应 ZeRO-3。Megatron 的 DP 则是逻辑 process-group 维度，一个模型 replica 可能由 TP×PP×CP 多张 GPU 共同组成，DP group 只同步相同 model-parallel shard。使用 Distributed Optimizer 或 FSDP 分片策略后，通信可能由单纯 gradient AllReduce 演进为 ReduceScatter 和 parameter AllGather，但仍然属于 Data Parallelism。

### 10. 常见错误回答

- “DP 就是 `torch.nn.DataParallel`”：把算法策略与一个早期 PyTorch 类混为一谈。
- “DDP 会在每一步广播参数”：标准 DDP 训练主路径是 gradient AllReduce；参数初始化同步后，各 rank 通过相同梯度和 optimizer step 保持一致。
- “FSDP 不是数据并行，因为参数没有复制”：FSDP 的逻辑训练语义仍是数据并行；model states 是否以及如何物理分片取决于 sharding strategy。
- “Megatron DP=8 就是 8 张 GPU”：每个 DP replica 可能还包含 TP×PP×CP ranks，总 GPU 数需按完整 mesh 计算。
- “DP 通信永远是 AllReduce”：Distributed Optimizer 和 FSDP 分片策略会使用 ReduceScatter、AllGather 等通信。

<a id="dp-communication"></a>
## DP 通信算子：AllReduce、ReduceScatter 与 AllGather

DP 的通信路径取决于 model state 是否分片。最常见的三种模式是：

```text
标准 DDP：                gradient AllReduce
Distributed Optimizer：  gradient ReduceScatter + parameter AllGather
FSDP FULL_SHARD / ZeRO-3：parameter AllGather + gradient ReduceScatter
```

### 1. 标准 DDP：Gradient AllReduce

每个 DP rank 保存相同模型参数、处理不同数据，因此反向传播后会得到不同的局部梯度：

```text
rank 0: local gradient g0 ─┐
rank 1: local gradient g1 ─┼─ AllReduce(SUM/AVG)
rank 2: local gradient g2 ─┤
rank 3: local gradient g3 ─┘
                           ▼
               每个 rank 都得到完整聚合梯度 g
```

各 rank 随后独立执行相同的 optimizer step，参数因而继续保持一致。实际框架通常把梯度放入 contiguous buckets：某个 bucket 的梯度一旦 ready，就异步发起 AllReduce，与剩余 backward compute 重叠，而不是等整个 backward 结束后一次性通信。

标准 DDP 的核心语义是：

> 每个 rank 都需要完整聚合梯度，因此使用 gradient AllReduce；普通 DP 本身不减少参数、梯度和 optimizer state 的副本数。

### 2. Distributed Optimizer：ReduceScatter + AllGather

当 optimizer state、FP32 main parameters 和更新计算由 DP ranks 分片负责时，不需要让每个 rank 都保留完整聚合梯度并重复执行完整 optimizer update。

第一步是对梯度执行 ReduceScatter：

```text
各 rank 的完整局部梯度
          │
    ReduceScatter
          ▼
rank 0: reduced gradient shard 0
rank 1: reduced gradient shard 1
rank 2: reduced gradient shard 2
rank 3: reduced gradient shard 3
```

每个 rank 使用自己负责的 gradient shard、optimizer-state shard 和 master-parameter shard 完成本地更新：

```text
gradient shard
    + optimizer-state shard
    + master-parameter shard
          │
          ▼
updated parameter shard
```

更新后再对低精度 model-parameter shards 执行 AllGather：

```text
rank 0: updated parameter shard 0 ─┐
rank 1: updated parameter shard 1 ─┼─ AllGather
rank 2: updated parameter shard 2 ─┤
rank 3: updated parameter shard 3 ─┘
                                   ▼
               每个 rank 恢复完整 model-parameter buffer
```

完整关键路径是：

```text
Backward
   -> gradient ReduceScatter
   -> local optimizer step
   -> parameter AllGather
   -> next Forward
```

这里最容易答错的是 AllGather 对象：

> Distributed Optimizer 通常 AllGather 的是更新后的 parameter shards，不是把 gradient shards 重新拼成完整梯度。

Megatron-Core 官方 Distributed Optimizer 的数据流也是 gradient ReduceScatter、分片 optimizer step、parameter AllGather，并使用连续 parameter/main-gradient buffers 组织通信。[官方文档](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/dist_optimizer.html)

### 3. AllReduce 与 ReduceScatter + AllGather 的关系

从规约结果的数据分布语义看，可以写成：

```text
AllReduce(x) ≈ AllGather(ReduceScatter(x))
```

`ReduceScatter` 先完成规约并把结果分片，`AllGather` 再让所有 rank 获得全部规约结果。这个等价关系不表示底层实现一定机械调用两个 collective；实际 NCCL algorithm、chunking、ring/tree 路径和 overlap 策略可能不同。

Distributed Optimizer 的关键优化，是利用 ReduceScatter 产生的中间分片直接完成 optimizer update：

```text
标准 DDP：
local gradients -> AllReduce -> 每卡完整梯度 -> 每卡完整更新

Distributed Optimizer：
local gradients -> ReduceScatter -> 每卡梯度分片
                -> 每卡分片更新 -> AllGather updated parameters
```

### 4. FSDP FULL_SHARD / ZeRO-3：参数也长期分片

FSDP FULL_SHARD/ZeRO-3 进一步把 model parameters 本身长期分片。每个 FSDP unit 通常需要在计算前临时恢复参数：

```text
Forward 前：
parameter AllGather -> Forward -> parameter reshard/free

Backward：
parameter AllGather -> Backward -> gradient ReduceScatter
```

因此它的主通信路径是：

```text
parameter AllGather
    -> Forward / Backward compute
gradient ReduceScatter
    -> sharded optimizer update
```

与经典 Megatron Distributed Optimizer 的区别是：

- Distributed Optimizer 主要分片 optimizer state、FP32 main parameters 和更新责任；低精度 model-parameter buffer 在参数同步后通常仍在 DP ranks 上复制。
- FSDP FULL_SHARD/ZeRO-3 还长期分片 model parameters，并在 forward/backward 所需位置按 FSDP unit 做 parameter AllGather。

Megatron-FSDP 的 `optim_grads_params` 路径通过 pre-forward/pre-backward hooks unshard parameters，通过 post-forward/post-backward hooks reshard parameters 并规约梯度；具体通信粒度受 FSDP unit、bucket 和 prefetch 配置影响。其他 sharding mode 不应直接套用这条 FULL_SHARD 生命周期。[Megatron-FSDP API](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.distributed.fsdp.src.megatron_fsdp.megatron_fsdp.html)

### 5. Gradient Accumulation 与通信重叠

若一次 optimizer step 包含多个 microbatches，通常不应在每个 microbatch 后都做完整 DP 同步：

```text
microbatch 0 ─┐
microbatch 1 ─┼─ local gradient accumulation
microbatch 2 ─┤
microbatch 3 ─┘
               -> final grad synchronization
               -> optimizer step
```

常见实现使用 `no_sync` 或 `is_last_microbatch` 控制同步时机。开启通信重叠后：

- gradient bucket ready 后异步发起 AllReduce/ReduceScatter，与剩余 backward 重叠；
- parameter AllGather 可以预取并与下一层 forward 重叠；
- bucket 太小会变得 latency-bound，太大则会延迟 collective 启动并增加峰值 buffer。

所以性能诊断不能只看 collective 总时长，还要区分：

```text
通信总时间
通信被计算隐藏的时间
真正暴露在 critical path 上的时间
```

### 6. 其他辅助通信

DP 还可能使用以下操作，但它们通常不是主要吞吐路径：

- `Broadcast`：初始化时从指定 rank 同步参数或状态；
- `Barrier`：初始化、checkpoint、调试或阶段切换时同步；
- `AllReduce`：同步 loss、token count、gradient norm、overflow flag；
- `AllGather`：收集变长 metadata、指标或数据进度；
- `Reduce`：只把统计结果规约到一个 rank。

普通 DP 不以 point-to-point 作为主通信方式；P2P 更常见于 PP stage 间 activation/gradient 传输。

### 7. 与 CP、EP 组合后的 group 边界

概念上的 DP 是“处理不同数据的模型副本”，但 Megatron-Core 中实际 gradient/optimizer group 还会受到 CP 和 EP 影响：

- CP ranks 复制 Dense 参数，却分别处理同一 sequence 的不同 context，因此 Dense 参数的梯度规约和 Distributed Optimizer 分片常使用 `dp_cp` group；
- routed expert 参数只在持有同一 expert shard 的 EDP ranks 间同步，不能错误地使用 Dense DP group；
- Parallel Folding 下 Dense/Attention 和 routed experts 是两套逻辑 mesh，必须分别核对 gradient ReduceScatter 和 parameter AllGather 的 group membership。

这也是面试中不能只回答“DP 就是 AllReduce”的原因：高级岗位通常会继续追问究竟同步什么 tensor、在哪个 process group、何时发起、是否分片以及能否与计算重叠。

### 8. 面试精炼回答

> 标准 DP 在反向传播中使用 gradient AllReduce，让不同数据副本得到相同的聚合梯度，通常按 bucket 发起并与 backward overlap。使用 Distributed Optimizer 后，通信一般拆成 gradient ReduceScatter 和 updated-parameter AllGather：每个 DP rank 只保留一份聚合梯度及 optimizer shard，局部更新对应参数分片，再 AllGather 恢复低精度模型参数。FSDP FULL_SHARD/ZeRO-3 进一步长期分片 model parameters，因此还要在 forward/backward 前按 FSDP unit AllGather 参数，反向后 ReduceScatter 梯度；其他 FSDP sharding strategy 的驻留和通信不同。回答时还要说明 gradient accumulation 的同步时机，以及 CP 下的 `dp_cp`、MoE 下的 EDP group 边界。

### 9. 高频追问

1. 为什么 `AllReduce ≈ ReduceScatter + AllGather`，但 Distributed Optimizer 不会重新 AllGather 完整梯度？
2. gradient bucket 大小如何影响 overlap、延迟和峰值显存？
3. ZeRO-1/2/3 分别分片哪些 model states，通信路径如何变化？
4. gradient accumulation 时为什么通常只在最后一个 microbatch 同步？
5. 开启 CP 后，为什么 Dense optimizer group 可能是 `DP × CP`，而 global batch 计算仍只除以语义 DP？
6. routed expert gradients 应该在哪个 group 中规约？

## 生产关注

- global batch 改变会影响收敛，需要学习率和 warmup 配合。
- DP group 跨节点通信较重，bucket 和 overlap 很关键。
- ZeRO/FSDP 分片策略本质是在 DP 语义下去掉不同范围的状态冗余。

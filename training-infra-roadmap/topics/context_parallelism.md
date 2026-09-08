# Context Parallelism

5D 组合入口：[Megatron 5D 并行总览](distributed_training.md)。

面试入口：[MEGATRON-01｜5D 与 TP/CP 拓扑选择](../../private_resume/2026-08-llm-infra-interview-prep.md#megatron-01)；本页重点：[Hierarchical CP：机内 A2A、机间 Ring](#hierarchical-cp)。

> 所属专题：[Long-context Training](long_context_training.md)。CP 是长上下文训练的核心机制之一，但长上下文训练还包括 pretraining/SFT/RL 的数据、kernel、checkpoint、rollout 和稳定性问题。

## 核心问题

长上下文训练中，单卡无法承载完整 sequence 的 attention/activation。Context Parallelism 沿 context 维切分序列，让多个 GPU 协作处理一个样本的长上下文。

## 上游材料

- [Long-context Training](long_context_training.md)
- [FlashAttention](flashattention.md)
- [Sequence Parallelism](sequence_parallelism.md)

## 关键机制

- context dimension partition
- attention KV exchange
- ring attention / block attention 类通信模式
- 与 FlashAttention 的 kernel 边界配合

<a id="hierarchical-cp"></a>
## Hierarchical CP：机内交换布局，机间流动 KV

> 核验日期：2026-09-08。以下解释 Megatron-Core / Transformer Engine 的 `cp_comm_type="a2a+p2p"` 路径，不把它泛化为所有名为 hybrid/hierarchical CP 的实现；配置需以实际安装版本为准。下面的 GPU/head 数是说明机制的自拟例子，不是个人项目配置或性能实测。

### 1. 一句话和动机

**把一个 CP group 分成两级：内层用 all-to-all（A2A）把“短序列、较多 heads”换成“较长序列、较少 heads”；外层用 ring/P2P 逐块交换 KV，算完再用逆向 A2A 恢复原布局。** 内层通常映射 NVLink/NVSwitch，外层映射节点间网络，但这需要正确的 rank placement，不是配置名字自动保证的。[MCore `cp_comm_type` 定义](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.transformer.transformer_config.html#core.transformer.transformer_config.TransformerConfig.cp_comm_type)

为什么组合两种方法？纯 A2A 要继续切 heads，受可分的 KV heads 数限制；纯 ring 可以沿序列扩展，但大 CP 的逐块计算/通信轮次和跨节点等待可能增多。分层允许把高带宽域用于布局交换，再通过外层序列分片扩容；它是一个需要比较的候选策略，不是“跨节点 CP 必开项”。[NVIDIA HCP 说明](https://docs.nvidia.com/nemo/megatron-bridge/nightly/training/hierarchical-context-parallel.html)

### 2. 两级分别做什么

固定一个 TP rank 来看，令总序列长度为 `S`，该 rank 经 TP 投影后有 `h_q` 个 Q heads、`h_kv` 个 KV heads；内层大小 `A`，外层大小 `R`，`CP=A×R`。先忽略 batch、head dimension 和 causal chunk 的重排，只看逻辑元素数：

| 阶段 | 每 rank 的 Q | 每 rank 的 K/V | 实际工作 |
|---|---|---|---|
| 进入 Attention 通信前 | `S/(A×R)` tokens × `h_q` heads | `S/(A×R)` tokens × `h_kv` heads | 保留自身 CP token 分片 |
| 内层 A2A 后 | `S/R` tokens × `h_q/A` heads | `S/R` tokens × `h_kv/A` heads | 从 A 个成员收集对应 heads 的 token，发送其他 heads；不是复制一份全量 QKV |
| 外层 Ring 计算 | Q 留在本 rank | 同一 head 分片的 KV 按 outer ring 流入/流出 | local Q 与各 KV block 计算 Attention，并合并 softmax 统计 |
| 输出逆向 A2A 后 | 输出恢复为 `S/(A×R)` tokens × `h_q` heads | — | 把 token 归还原 CP owner、拼回该 TP rank 的 heads，再接输出投影 |

这张表中的“较长序列”只有 **内层组覆盖的 `S/R`**，并不是每卡提前拿到全局 `S`。GQA 下 `h_q` 与 `h_kv` 不同，两个头数都必须符合切分约束。TE 的 `flash_attn_a2a_communicate` 与 `AttnFuncWithCPAndKVP2P` 对应上述交换和 ring 路径。[TE 实现](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/attention/dot_product_attention/context_parallel.py)

**为什么仍然是全局 Attention？** 对每个 query，依次覆盖 mask 允许的 KV blocks；各块输出用对应 softmax 的 log-sum-exp 统计合并，不能直接平均“每块单独 softmax”后的结果。反向也要累积远端 query 对 K/V 的梯度，并把梯度送回对应 owner。它改变计算和数据布局，不改变模型的 Attention 连接关系；浮点归约顺序变化仍需数值验收。[FlashAttention 的分块与 online softmax 原理](https://arxiv.org/abs/2205.14135)

### 3. 具体例子：2 台机器 × 每台 8 GPU，TP=2、CP=8

设 `PP=1、DP=1`，标准 TP 布局下总 Q heads=32、KV heads=8，序列 128K。则每个 TP rank 有 `h_q=16、h_kv=4`，可选 `A=4、R=2`：

```text
world_size = TP × PP × CP × DP = 2 × 1 × 8 × 1 = 16
CP = A × R = 4 × 2 = 8
机内占用 = TP × A = 2 × 4 = 8 GPU
```

以下假设物理节点分别放 global ranks `0–7` 和 `8–15`，且 TP rank 变化最快。TP groups 是 `[0,1]、[2,3]、…、[14,15]`。只展示 TP rank=0 对应的那一个 CP group：

| 分组 | global ranks | 通信 |
|---|---|---|
| 完整 CP group | `[0,2,4,6,8,10,12,14]` | 8 个 CP 成员共同承载同一上下文 |
| 节点 0 的 inner group | `[0,2,4,6]` | 机内 A2A |
| 节点 1 的 inner group | `[8,10,12,14]` | 机内 A2A |
| 四个 outer groups | `[0,8]`、`[2,10]`、`[4,12]`、`[6,14]` | 对应 head 分片之间的跨节点 P2P |

TP rank=1 的奇数 ranks 组成另一套相同结构；**不是挑一张卡替整台机器做跨节点通信**。这些分组是给定 rank 顺序下的推导；换启动顺序后应打印 group 的 global ranks 与 hostname 检查。[MCore `create_hierarchical_groups` / `initialize_model_parallel`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py)

把一次 Attention 的元素数代进去就更直观（`K=1024`）：

```text
A2A 前：Q = 16K tokens × 16 heads；K/V 各 = 16K tokens × 4 heads
A2A 后：Q = 64K tokens ×  4 heads；K/V 各 = 64K tokens × 1 head
外层计算：Q 不动，读取本地及另一节点相应的 KV blocks
逆 A2A：输出恢复 16K tokens × 16 heads
```

每次布局交换前后的逻辑 Q/K/V 元素数守恒，但实际重排 buffer、通信 buffer 和 backward saved tensors 仍可能提高峰值显存。Causal 负载均衡常使用前后配对的 chunks，因此这里的 16K/64K 是 token 数量，不保证为连续位置区间。

### 4. 怎么配，哪些条件会卡住

Megatron-LM 参数片段如下；它不是完整启动命令，也不替代模型/数据/精度等配置：

```bash
--tensor-model-parallel-size 2 \
--context-parallel-size 8 \
--hierarchical-context-parallel-sizes 4 2 \
--cp-comm-type a2a+p2p
```

`hierarchical_context_parallel_sizes=[4,2]` 的第一项是 A2A group 大小，第二项是 P2P group 大小；二者乘积必须等于 CP，不能再把它们乘到一个已含 CP 的 world size 上。[配置字段](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/model_parallel_config.py) · [CLI 校验](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/arguments.py)

使用前检查：

- **Heads**：当前 TE 路径需要 `A` 整除送进 Attention 的本地 Q/K/V heads；不能默认 KV 不足时自动复制。标准均匀 TP 下通常检查 `h_kv=总 KV heads/TP`，而不是只看全模型 head 数。例子里 `h_kv=4`，inner 取 8 就不成立；outer ring 的大小不受这个 head 切分约束。
- **拓扑**：要让 inner group 全部本地，常见起点是 `TP×A≤每节点 GPU 数`，还要确认这些卡处于同一高速互联域。TP 已占满单机时，不能仅靠声明 `A>1` 获得额外机内 CP 卡位。
- **数据布局**：CP 的 sequence/chunk 整除、causal mask、位置编号和 packed `cu_seqlens` 仍要满足后端规则；本例不能直接当作所有 packed/变长布局的索引实现。
- **版本与接入路径**：MCore、TE、Attention backend 要匹配；当前 TE HCP 分支还有 attention bias 等支持限制。若经 Megatron Bridge 接入，核验时的官方页面只保证 MPU 初始化路径，不能假设 decentralized process-group 路径一定创建 HCP groups。[Bridge 限制](https://docs.nvidia.com/nemo/megatron-bridge/nightly/training/hierarchical-context-parallel.html#stable-bridge-limitation)

### 5. 收益、代价与排障

收益来自**把两种通信方式放到合适的链路，并保留通信/计算重叠机会**。对同一个总 CP，外层 ring 处理的是 `R` 个较大的序列分片，而不是 `A×R` 个原始小分片；但每块的 token/head 布局也变了，不能据此声称网络字节数或训练时间固定降低 A 倍。

| 现象 | 优先检查 | 验收方式 |
|---|---|---|
| 初始化或 reshape 报错 | 两级乘积、TP 后 Q/KV heads、实际 TE 支持范围 | 先用可整除的小配置完成 forward/backward |
| 开了分层却更慢 | inner 是否跨节点；A2A 重排/等待；ring 的 exposed communication；并发网络争用 | 相同 workload 对比 `p2p`、合法的 `a2a` 与 `a2a+p2p`，看端到端 step time，不只看带宽 |
| 长序列 OOM | A2A layout 临时副本、P2P buffers、workspace 和激活生命周期 | 记录峰值时刻的 tensor shape 和 memory snapshot，不能只按 `S/CP` 估总显存 |
| loss/梯度不对 | token 重排与 causal mask 是否一致；softmax 合并；K/V 梯度 owner | 与小规模参考实现对齐输出、loss、梯度，再扩到真实长度 |

### 6. 与 TP、SP、Parallel Folding 的关系

- **不是新并行轴**：HCP 把既有 CP 分解成 `A×R`，不增加一层额外 GPU 倍数。
- **不是 TP**：内层暂时按 heads 分配 Attention 计算，不等于继续切 QKV/MLP 权重；输出会恢复 CP 布局。
- **不是 SP**：SP 复用 TP ranks，减少部分重复 activation；HCP 使用 CP ranks 组织 Attention 通信。二者可组合，但作用位置不同。
- **不是 Parallel Folding**：Folding 为 Attention 与 MoE experts 选择两套逻辑网格；HCP 在 Attention 的 CP group 内再组织两级通信。

### 7. 面试直接回答（约 45 秒）

> Hierarchical CP 是一种适配网络层级的 CP 通信实现。Megatron 的 a2a+p2p 先在机内用 all-to-all，把每卡的短序列、多 heads 换成更长序列、少 heads；然后跨节点用 ring 流动 KV，让本地 Q 分块完成全局 Attention，最后再通过逆 all-to-all 恢复原来的 token 布局。这样能利用 NVLink，也能让总 CP 超过单纯 head 切分的限制。但内层大小受 TP 后本地 KV heads 和机内卡位约束，是否更快要看重排、通信重叠和端到端 profile，不能默认开启就有收益。

↑ [返回本节](#hierarchical-cp) · [返回主文档原问题](../../private_resume/2026-08-llm-infra-interview-prep.md#megatron-hierarchical-cp) · [返回 5D 总览](distributed_training.md)

## 生产关注

- 变长序列会造成 load imbalance。
- CP 的通信模式与普通 DP/TP 不同，需要单独 profiling。
- 长上下文阶段通常需要重新调 micro-batch、recompute 和 checkpoint。

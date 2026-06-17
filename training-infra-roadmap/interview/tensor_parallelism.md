# Interview: Tensor Parallelism

## 高频面试题

1. Tensor Parallel 解决什么问题？
2. Column Parallel 和 Row Parallel 怎么切？
3. 为什么 TP 通常放在单机 NVLink 内？
4. TP forward/backward 有哪些 collective？
5. TP 和 PP/DP/FSDP 如何组合？

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

# Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism

## 论文信息

- 作者：Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, Bryan Catanzaro
- 时间：2019
- 链接：https://arxiv.org/abs/1909.08053
- 相关主题：[Tensor Parallelism](../topics/tensor_parallelism.md), [Distributed Training](../topics/distributed_training.md), [NCCL](../topics/nccl.md)

---

## 解决的问题

单靠 Data Parallel 无法把数十亿参数 Transformer 放进单卡显存，传统模型并行又容易引入复杂编译器或低效通信。Megatron-LM 提出一种直接在 PyTorch 里实现的 intra-layer model parallel，把 Transformer 的核心矩阵乘切开，让 billion-scale dense model 能在多 GPU 上高效训练。

---

## 背景与瓶颈

2019 年的瓶颈很清楚：模型参数、梯度和 optimizer state 已经超过单卡容量；但如果随意切层或切 op，会导致 GPU 等待和通信开销吞掉收益。Megatron 的选择非常工程化：只切最规则、最大头的 GEMM，让通信点少而稳定，保留大矩阵乘的计算粒度。

---

## 核心创新

- Column Parallel Linear：把输出维切开，常用于 QKV 和 MLP up projection。
- Row Parallel Linear：把输入维切开，常用于 attention output 和 MLP down projection。
- 在 Transformer block 内成对设计切分，减少不必要 all-gather。
- 用少量 all-reduce 在关键边界同步 partial result。
- 保持 PyTorch 可实现性，不依赖新的编译器栈。

---

## 关键图表解读

最重要的是 MLP 和 attention projection 的切分图。工程上要看通信插入点：第一层 GEMM 按列切，每张卡产出一部分 hidden；第二层 GEMM 按行切，每张卡消费自己的 hidden slice，最后 all-reduce 合并结果。这种模式解释了为什么 TP group 内网络必须足够快，也解释了为什么 TP 通常优先放在单节点 NVLink/NVSwitch 内。

---

## 工程价值

Megatron-LM 给训练系统提供了一个可落地的原则：不要为了“并行”把计算打碎到失去 GEMM 效率；要围绕 Transformer block 的大矩阵乘设计切分，让通信和计算边界稳定可预测。

---

## 对训练基础设施的影响

- 奠定现代 Tensor Parallel 的基本形式。
- 影响 Megatron-Core、NeMo、DeepSpeed hybrid parallel、Colossal-AI 等系统。
- 让 TP + PP + DP 成为超大模型训练默认组合。
- 推动 sequence parallel、context parallel 等后续降低 activation 显存和长上下文成本的技术。

---

## 今天的应用场景

当 dense model 或 MoE activated parameters 超出单卡容量，或者单卡 GEMM 已经不足以提供目标吞吐时，TP 仍是基础工具。生产上 TP size 不宜盲目增大，因为 TP 通信频繁，跨节点 TP 会显著吃网络。

---

## 后续演进

Megatron-LM 2019 → Megatron 2021 3D parallel → Sequence Parallel → Context Parallel → Megatron-Core distributed checkpointing。

---

## 相关论文

- [Transformer](transformer.md)
- [Megatron 2021](megatron_2021.md)
- [GPipe](gpipe.md)
- [ZeRO](zero.md)

---

## 相关代码

- GitHub：https://github.com/NVIDIA/Megatron-LM
- 关键模块：`megatron/core/tensor_parallel`
- 推荐阅读入口：`ColumnParallelLinear`、`RowParallelLinear`、parallel state 初始化、distributed checkpoint。

---

## 面试高频问题

1. Tensor Parallel 和 Data Parallel 的本质区别是什么？
2. Column Parallel Linear 和 Row Parallel Linear 分别切哪个维度？
3. Megatron 为什么能减少 all-gather？
4. TP 中 forward 和 backward 分别有哪些 collective？
5. TP size 增大为什么可能降低吞吐？
6. 为什么 TP group 通常放在节点内？
7. Attention QKV projection 如何做 TP？
8. MLP 两个 projection 如何配合切分？
9. TP 和 sequence parallel 的关系是什么？
10. TP checkpoint 如何支持不同 TP size 的恢复？

---

## 生产环境思考题

1. 8 卡节点上 TP=8 和 TP=4+DP=2 如何选？
2. 如果 TP all-reduce 成为瓶颈，如何定位是 NCCL、拓扑还是算子粒度问题？
3. TP group 跨机后吞吐下降，如何通过 profiling 证明原因？
4. TP 和 activation recompute 如何一起估算显存？
5. 不同层 hidden size 不一致时，TP 切分要如何处理？
6. MoE 模型中 TP 与 expert parallel 如何组合？
7. 保存 checkpoint 时如何记录 TP metadata？
8. 推理服务是否还应保留训练时 TP 切分？
9. 当 batch size 很小时，TP 为什么可能让 GPU 更空？
10. TP 训练中某张卡慢会如何影响整个 step？

---

## 我的总结

Megatron-LM 的价值在于把“模型并行”从概念变成了工程范式：围绕 Transformer 的大 GEMM 做规则切分，用少量 collective 换取显存和计算扩展。今天很多训练平台仍然在调同一个旋钮：TP 放多大、放在哪个拓扑域、如何与 PP/DP/FSDP/MoE 组合。

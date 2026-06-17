# 30 天计划：恢复论文阅读习惯

目标：每周两篇，总投入不超过 4 小时；读完能用自己的话解释工程价值，而不是复述摘要。

## Week 1：训练对象

- Day 1：读 [Transformer](../papers/transformer.md)，画出 Transformer block 的计算图。
- Day 3：读 [BERT](../papers/bert.md)，重点比较 encoder-only 与 decoder-only 训练负载。
- Day 6：复盘：attention、MLP、LayerNorm、embedding 分别对应哪些 kernel 和显存。

## Week 2：并行训练起点

- Day 8：读 [Megatron-LM](../papers/megatron_lm.md)，手写 Column/Row Parallel Linear 的通信点。
- Day 11：读 [GPipe](../papers/gpipe.md)，理解 micro-batch、bubble、activation 保存。
- Day 13：复盘：TP、PP、DP 分别解决什么问题，分别引入什么问题。

## Week 3：显存优化

- Day 15：读 [ZeRO](../papers/zero.md)，整理 ZeRO-1/2/3 状态生命周期。
- Day 18：读 [FSDP](../papers/fsdp.md)，对比 ZeRO-3 与 PyTorch FSDP。
- Day 20：复盘：参数、梯度、optimizer state、activation、checkpoint 各自如何省。

## Week 4：Kernel 与生产系统

- Day 22：读 [FlashAttention](../papers/flashattention.md)，重点理解 HBM/SRAM 数据移动。
- Day 25：读 [MegaScale](../tech_reports/megascale.md)，整理 straggler 和 fault tolerance。
- Day 28-30：写一页总结：如果设计一个 1K GPU 训练平台，最先建设哪些能力。

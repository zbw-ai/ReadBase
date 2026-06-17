# ZeRO: Memory Optimizations Toward Training Trillion Parameter Models

## 论文信息

- 作者：Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He
- 时间：2019
- 链接：https://arxiv.org/abs/1910.02054
- 相关主题：[ZeRO](../topics/zero.md), [FSDP](../topics/fsdp.md), [Checkpointing](../topics/checkpointing.md)

---

## 解决的问题

Data Parallel 简单好用，但每张 GPU 都复制完整参数、梯度和 optimizer state，显存浪费巨大。ZeRO 的核心目标是消除 DP 里的冗余状态，让模型规模随 GPU 数量增长，同时尽量保留 DP 的易用性和计算粒度。

---

## 背景与瓶颈

Adam 训练时，参数、梯度、momentum、variance 加起来通常远超模型权重本身。传统 DP 每卡一份完整副本，显存随模型规模线性爆炸，却没有利用多卡总显存。Model Parallel 可以解决部分显存，但开发复杂、通信敏感。ZeRO 则从“状态冗余”切入。

---

## 核心创新

- ZeRO-1：分片 optimizer states。
- ZeRO-2：进一步分片 gradients。
- ZeRO-3：进一步分片 parameters，需要按需 all-gather。
- 保留 DP 训练语义，但用 reduce-scatter/all-gather 管理状态生命周期。
- 将显存优化从模型结构切分转成训练状态切分。

---

## 关键图表解读

最重要的是三阶段显存拆分图。工程上要看每个 stage 改变了哪个生命周期：ZeRO-1 主要影响 optimizer step；ZeRO-2 影响 backward gradient reduce；ZeRO-3 影响 forward/backward 参数 materialization。Stage 越高，显存越省，但 runtime 通信和参数预取调度越复杂。

---

## 工程价值

ZeRO 把大模型训练的显存问题拆成可管理的三类状态，直接影响 DeepSpeed、FSDP 和后来的 distributed checkpoint。它让平台工程师可以独立思考：哪些状态需要常驻 GPU，哪些可以分片，哪些可以 offload，哪些可以 recompute。

---

## 对训练基础设施的影响

- FSDP 与 ZeRO-3 在思想上高度相近：参数按需 all-gather，使用后释放。
- 训练框架必须管理参数 shard、通信 bucket、prefetch、overlap。
- Checkpoint 不再是单文件权重保存，而是 sharded state 保存和重分片恢复。
- Offload/Infinity 推动 CPU/NVMe 分层内存训练。

---

## 今天的应用场景

中大规模训练常用 ZeRO/FSDP 降低显存占用，尤其是 optimizer state 很大的 Adam 系训练。选择 stage 时要看瓶颈：如果显存压力不大，ZeRO-1/2 更稳；如果单卡放不下模型，ZeRO-3/FSDP 才是必要复杂度。

---

## 后续演进

ZeRO → ZeRO-Offload → ZeRO-Infinity → PyTorch FSDP → Distributed Checkpointing → Hybrid Sharding。

---

## 相关论文

- [Megatron-LM](megatron_lm.md)
- [FSDP](fsdp.md)
- [ZeRO-Offload](zero_offload.md)
- [ZeRO-Infinity](zero_infinity.md)

---

## 相关代码

- GitHub：https://github.com/microsoft/DeepSpeed
- 关键模块：DeepSpeed ZeRO optimizer、partitioned parameters、offload engine。
- 推荐阅读入口：ZeRO stage config、bucket size、overlap communication、checkpoint load/save。

---

## 面试高频问题

1. ZeRO 解决的是哪类显存冗余？
2. ZeRO-1/2/3 分别分片什么？
3. ZeRO-3 和 FSDP 有什么相似和不同？
4. ZeRO 为什么能保持 Data Parallel 语义？
5. ZeRO-3 forward 时参数如何被使用？
6. reduce-scatter 和 all-gather 在 ZeRO 中分别做什么？
7. ZeRO 与 Tensor Parallel 可以如何组合？
8. ZeRO-Offload 为什么不总是更快？
9. ZeRO checkpoint 为什么比普通 DP checkpoint 复杂？
10. ZeRO stage 越高为什么不一定越好？

---

## 生产环境思考题

1. ZeRO-3 训练中 all-gather 抖动如何定位？
2. bucket size 过大或过小分别会导致什么问题？
3. 参数 prefetch 如何与 forward 计算重叠？
4. 如果 checkpoint 恢复后 loss 异常，如何检查 shard 对齐？
5. Offload 到 CPU 时 PCIe 带宽如何估算？
6. ZeRO 与 activation checkpointing 如何组合选型？
7. 多节点训练中 ZeRO 的通信 group 如何布局？
8. optimizer state sharding 对学习率调度有什么影响？
9. ZeRO-3 下如何做参数统计和 debug？
10. 训练中断后如何验证恢复点包含 optimizer 和 RNG 状态？

---

## 我的总结

ZeRO 的关键不是“省显存”四个字，而是把 DP 里的训练状态从复制品变成分布式资源。它迫使训练框架管理参数、梯度、optimizer state 的生命周期，也让 checkpoint、通信重叠、offload 和恢复流程成为平台能力的一部分。

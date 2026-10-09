# Part 01｜AI Systems 基础

这一部分建立 GPU 执行、数据搬运、分布式通信与低精度的共同语言，服务后续训练、推理和 RL 学习。先判断瓶颈落在计算、显存还是通信，再讨论优化开关；这里不是完整的硬件选型手册。

## 从哪里开始

按阶段学习先选 [GPU Systems 核心路线](roadmaps/gpu-systems.md)或 [Distributed Systems 核心路线](roadmaps/distributed-systems.md)。前者解释执行与搬运，后者解释通信、部分失败和状态；后者不等于训练 5D 并行。

| 编号 | 主题入口 | 学习问题与覆盖 |
|---|---|---|
| 1.1 | [GPU 执行与片上复用](topics/transformer_engine.md#gpu-execution) | 先具备 tensor shape、矩阵乘基础，回答 warp、tiling 与数据复用改变了什么 |
| 1.2 | [PyTorch 执行与布局](topics/transformer_engine.md#pytorch-execution) | 区分 view、复制与 Autograd，追问额外分配和同步发生在哪里 |
| 1.3 | [NCCL 与通信语义](topics/nccl.md) | 在进入分布式训练前，画清每个 rank 的输入、输出和 process group |
| 1.4 | [FP8 与数值格式](topics/fp8.md) | 先分清范围、精度和状态存储，再读 scaling 与稳定性边界 |
| 1.5 | [FlashAttention](topics/flashattention.md) | 带着 Attention shape 与 HBM IO 问题读；topic 仍是骨架，机制展开见所链论文 |

## 问题到验证

| 工程问题 | 正文入口 | 实验 / 排障入口 | 当前状态 |
|---|---|---|---|
| GPU 忙，为什么 step 仍慢？ | [Roofline 与测量](topics/transformer_engine.md#roofline) | [Slow Step](../practice/playbooks/slow_step_debug.md) | 已有解释；排障命令仍需结合环境补齐 |
| collective 为什么 hang？ | [NCCL](topics/nccl.md) | [TP NCCL Hang](../practice/playbooks/tp_nccl_hang.md) | 已有语义与排障路径；不代表本仓库完成复现 |
| Attention 的 IO 成本如何变化？ | [FlashAttention](topics/flashattention.md) | [Benchmark](../practice/experiments/flashattention/benchmark.md) | topic 为骨架；实验 `NEW`，无结果 |
| 低精度为什么更慢或不稳定？ | [FP8](topics/fp8.md) · [Transformer Engine](topics/transformer_engine.md) | [A100 实验路线](../practice/experiments/a100_fsdp_io_lab.md) | 正文已有；路线未执行，不把 A100 当作原生 FP8 实验平台 |

## 继续深入

[Transformer 论文工程解读](../research/papers/transformer.md)连接模型计算与系统成本；[研究入口](../research/README.md)统一维护来源和阅读决策。硬件、存储与网络的独立基础章节尚未创建；[stas00/ml-engineering](https://github.com/stas00/ml-engineering)仅作为选读来源，不代表本仓库已完成对应章节。

[返回总入口](../README.md) · [最近更新](../README.md#recent-updates) · [知识图谱](../KNOWLEDGE_GRAPH.md) · [下一部分：训练基础设施](../02-training-infra/README.md)

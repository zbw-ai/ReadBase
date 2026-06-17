# 一年计划：成长为高级 AI Training Infra 工程师

目标：不仅能读懂系统论文，还能设计、上线、优化和排障超大规模训练平台。

## Q1：补齐基础模型与分布式训练

- 精读 Transformer、GPT-3、Megatron-LM、Megatron 2021、ZeRO、FSDP。
- 能力目标：给定模型规模和 GPU 拓扑，能提出 TP/PP/DP/FSDP 初始配置。

## Q2：掌握 Kernel、显存与长上下文

- 精读 FlashAttention 1/2/3、Sequence Parallel、Context Parallel、Transformer Engine FP8。
- 能力目标：能用 profiler 判断 attention/MLP/communication/data pipeline 瓶颈。

## Q3：掌握 MoE 与超大规模训练

- 精读 GShard、Switch、DeepSpeed-MoE、Mixtral、DeepSeek-V3。
- 能力目标：能设计 MoE expert placement、load balance、all-to-all 观测和 checkpoint 策略。

## Q4：系统化生产能力

- 精读 MegaScale、OPT-175B 训练日志、NVIDIA distributed checkpointing 相关资料。
- 能力目标：能设计 1K-10K GPU 训练平台的监控、容错、调度、checkpoint 和事故复盘流程。

## 年终项目

写一份完整设计文档：《10K GPU LLM Training Platform Design》。必须覆盖：

- 集群拓扑和通信域划分
- 并行策略搜索与配置管理
- 显存预算和 checkpoint 策略
- NCCL/RDMA/RoCE/InfiniBand 诊断
- Straggler detection
- Fault tolerance 和自动恢复
- 训练作业可观测性

# Engineering Blogs

这个目录收录工程博客、官方技术文档、release note、系统实践文章和厂商技术解读。

它和 `papers/`、`tech_reports/` 的定位不同：

- `papers/`：理解技术思想的来源和历史演进。
- `tech_reports/`：理解模型/系统报告中的训练设计。
- `engineering_blogs/`：理解真实工程栈正在解决什么问题，尤其是没有正式论文的训练基础设施能力。

2026 年以后，很多高价值信息不会以 paper 形式出现，而是出现在 Megatron-Core、Transformer Engine、NCCL、FSDP、checkpoint、serving/training platform、RL pipeline 等工程博客或官方文档里。这里的目标不是收藏链接，而是把博客转化成工程判断。

当前 `references/blogs.csv` 仍是 backlog 索引，尚未代表已经完成的工程博客笔记。第一批真实条目建议从 NVIDIA Training Stack 和 vLLM/OpenRLHF 这类已经影响当前 topic/playbook 的材料开始补。

## 收录原则

优先收录能回答以下问题的文章：

- 它解决了哪个真实训练系统瓶颈？
- 它影响了哪个工程边界：显存、通信、kernel、checkpoint、调度、容错、数据管线？
- 它和 Megatron、DeepSpeed、FSDP、NCCL、Transformer Engine、FlashAttention、MoE、RL training 的关系是什么？
- 它是否提供了生产环境配置、排障或性能数据？
- 它是否补充了 paper 中没有写清楚的实现细节？

## 厂商入口

- [NVIDIA](nvidia/README.md)：Megatron-Core、Transformer Engine、NCCL、CUDA、FP8、Blackwell/Hopper training stack。
- [OpenAI](openai/README.md)：scaling、post-training、reasoning、system safety、infrastructure signals。
- [Anthropic](anthropic/README.md)：Claude training/post-training、interpretability、safety infra、long-context signals。
- [DeepSeek](deepseek/README.md)：DeepSeek-V3/R1、MoE、FP8、DualPipe、reasoning training。
- [Google](google/README.md)：TPU/Pathways、Gemini、JAX/PAX、large-scale training systems。
- [Meta](meta/README.md)：Llama、PyTorch/FSDP、distributed training、data/infra practice。
- [Microsoft](microsoft/README.md)：DeepSpeed、ZeRO、Megatron-DeepSpeed、Azure AI training infrastructure。
- [ByteDance](bytedance/README.md)：大规模训练平台、MoE、推荐/LLM 训练系统实践。
- [Zhipu](zhipu/README.md)：GLM 系列、中文大模型训练、国产集群与工程实践。

## 建议模板

每篇工程博客笔记建议使用：

```text
# 标题

## 来源信息

## 解决的问题

## 工程背景

## 核心机制

## 系统设计要点

## 性能与稳定性信息

## 生产环境启发

## 和现有主题的关系

## 值得追问的问题

## 我的总结
```

不是所有博客都需要单独成文。短博客、发布说明、API 文档可以只进入 `references/blogs.csv`，再被 `topics/` 章节引用。

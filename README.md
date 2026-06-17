# ReadBase

ReadBase 是我的长期工程知识库，用来沉淀 AI Infra、LLM Infra、分布式训练、GPU 集群、推理系统和相关工程实践。

这个仓库不是论文收藏夹，而是面向工程判断、系统设计、生产排障和面试复习的 handbook 集合。

## 知识地图

![ReadBase Knowledge Map](assets/readbase-knowledge-map.svg)

## 当前专题

| 专题 | 状态 | 入口 |
|---|---|---|
| AI Training Infrastructure Handbook | 建设中 | [training-infra-roadmap](training-infra-roadmap/README.md) |

## 学习入口

- [30 天计划](training-infra-roadmap/roadmaps/30_day_plan.md)：恢复论文阅读习惯，每周两篇。
- [90 天计划](training-infra-roadmap/roadmaps/90_day_plan.md)：建立完整训练系统知识图谱。
- [一年计划](training-infra-roadmap/roadmaps/yearly_plan.md)：面向高级 AI Training Infra 工程师的长期路线。
- [面试手册](training-infra-roadmap/interview/tensor_parallelism.md)：从 Tensor Parallelism 开始建立可回答、可追问的面试表达。

## 推荐结构

当前采用“一个总仓库，多个专题手册”的结构：

```text
ReadBase/
  README.md
  training-infra-roadmap/
    README.md
    MASTER_READING_LIST.md
    KNOWLEDGE_GRAPH.md
    papers/
    tech_reports/
    topics/
    roadmaps/
    interview/
    references/
```

未来如果继续扩展，可以自然增加：

```text
ReadBase/
  inference-infra-roadmap/
  agentic-rl-roadmap/
  gpu-systems-roadmap/
  distributed-systems-roadmap/
```

## 维护原则

- 每个专题目录都应该有自己的 `README.md`、reading list 和知识图谱。
- `topics/` 写成工程手册章节，而不是概念摘抄。
- `papers/` 和 `tech_reports/` 服务于系统理解，不追求学术摘要式覆盖。
- `interview/` 聚焦高频题、追问、生产案例、错误回答和优秀回答。
- 尽量保持内部 Markdown 链接可点击、可长期维护。

## 当前重点

近期优先建设 [AI Training Infrastructure Handbook](training-infra-roadmap/README.md)，特别是：

- [Tensor Parallelism](training-infra-roadmap/topics/tensor_parallelism.md)
- [Checkpointing](training-infra-roadmap/topics/checkpointing.md)
- [FSDP](training-infra-roadmap/topics/fsdp.md)（建设中）
- [MoE](training-infra-roadmap/topics/moe.md)（建设中）
- [NCCL](training-infra-roadmap/topics/nccl.md)（建设中）

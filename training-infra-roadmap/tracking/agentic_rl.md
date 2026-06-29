# Agentic RL Tracking

用于追踪 Agentic RL、long-context RL、rollout infra、reward/verifier pipeline、异步采样和训练调度。

这个文件不等同于 RL 算法笔记。这里关注训练基础设施问题：

- rollout 如何调度？
- 长上下文 trajectory 如何存储和切分？
- policy training 与 inference worker 如何解耦？
- reward / verifier 如何成为系统瓶颈？
- PPO / GRPO / DAPO / agentic RL 对 checkpoint lineage 有什么新要求？
- scheduler 如何在 sample efficiency、GPU utilization、freshness 之间取舍？

## 模板

```text
## YYYY-MM-DD

### 标题

- 来源：
- 类型：paper / blog / repo / report
- 链接：
- 影响等级：
- Decision：Ignore / Observe / Read / Deep Dive
- Reason：
- 建议动作：
- Status：
- 关联主题：rollout / verifier / reward / scheduler / long context / checkpoint / distributed training
- 一句话价值：
- 需要追问：
```

## Backlog

暂无。

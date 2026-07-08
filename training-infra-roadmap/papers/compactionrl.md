# CompactionRL: Reinforcement Learning with Context Compaction for Long-Horizon Agents

## 论文信息

- 作者：Yujiang Li, Zhenyu Hou, Yi Jing, Jie Tang, Yuxiao Dong
- 机构：Tsinghua University
- 时间：2026
- 链接：https://arxiv.org/abs/2607.05378
- 相关主题：[Agentic RL](../topics/agentic_rl.md), [Long-context Training](../topics/long_context_training.md), [Rollout Latency](../playbooks/rollout_latency.md)
- 核验说明：标题、作者、日期和摘要已按 arXiv abstract 页元信息核验；方法细节来自论文 HTML 正文。

---

## 解决的问题

Long-horizon agent 在 coding、terminal、tool-use 任务里会不断产生交互历史：reasoning、tool call、stdout/stderr、错误日志、文件路径、失败尝试和中间方案都会进入 trajectory。只靠扩大 context window 有两个问题：第一，长上下文训练和推理成本很高；第二，真实 rollout 的长度分布有长尾，少量超长任务会拖垮 KV cache、prefill、decode 和训练样本构造。

CompactionRL 解决的是：在固定 working context budget 下，如何让 agent 继续执行超过单窗口长度的任务，并且让“压缩上下文”本身成为 RL 可训练行为。

---

## 背景与瓶颈

传统做法通常有三类：

1. 直接增加 max context length；
2. 在推理时做 external memory / summary / prompt compression；
3. 在 rollout 超长时截断历史或失败退出。

这些方法对训练系统都不够友好。增加 context length 会放大 attention、KV cache、activation 和 checkpoint 成本；推理时外部 summary 是 heuristic，summary 质量不一定服务最终任务 reward；截断历史则会丢掉关键状态，尤其是 coding task 中的文件路径、错误日志和已经尝试过的 patch。

更深的瓶颈是 RL 数据结构变了：一个原本完整的 trajectory 会被 compaction 切成多个 execution segment 和 summary segment。这样会破坏很多 group-wise RL 方法的假设，也会引入 segment 长度不均、reward 共享、跨 segment credit assignment 等问题。

---

## 核心创新

### 1. Trainable Context Compaction

当当前 history 接近 context budget 时，policy 会生成一段 summary，然后用：

```text
summary + recent turns
```

重建新的上下文继续 rollout。论文默认保留最近 `k=2` 个 interaction steps；如果放不下，会进一步减少 recent turns。

关键点是 summary 不是外部固定模块，而是同一个 trainable policy 生成的 token。summary token 与普通 assistant response 一样进入 RL objective。

### 2. Execution 和 Summary 共享最终任务 Reward

每条 rollout 最终只有 task correctness reward，例如 SWE-bench / Terminal-Bench 是否通过。CompactionRL 不额外设计 summary-quality reward，因为手工 summary metric 很难判断哪些细节对最终解题有用。

也就是说，summary 的训练信号来自最终任务成败：如果 summary 保留了正确的错误日志、文件路径、部分修复方案，后续执行更可能成功；如果 summary 丢了关键信息，任务失败，summary action 也会被惩罚。

### 3. 从 Group-wise 方法转向 PPO

论文认为 GRPO 这类 group-wise advantage estimator 不适合 compacted rollout。原因是一个 rollout 被切成 variable number of segments 后，固定 group 结构会被破坏；如果按 segment 做样本，compaction 多的 rollout 会在 group statistics 里被重复加权；如果只按完整 rollout 做 normalization，又缺少 segment-level advantage。

因此 CompactionRL 使用 PPO 和 critic/value function，让 value-based advantage 支持 variable segments。

### 4. Token-level Loss Normalization

compaction 会制造不同长度的 execution segment 和 summary segment。如果按 segment 或 sample 平均 loss，短 summary segment 和长 execution segment 的权重会不公平，也会让 compaction 次数多的 trajectory 过度影响训练。

论文改为对 generated tokens 做 token-level loss normalization，让每个 trainable token 的权重更稳定。

### 5. Cross-trajectory GAE

如果每个 segment 独立算 GAE，会把最终 reward 错误地放到每个 segment 末尾，导致早期 action / summary 看起来离 final reward 很近，credit assignment 偏乐观。

CompactionRL 引入 cross-trajectory GAE，用后续 segment 的 token 数修正 advantage，让 summary 前后的 segment 仍然按同一条 trajectory 的时间距离传播 reward。

---

## 关键图表解读

最重要的是 Figure 2：CompactionRL 的 rollout / training 流程。

工程上应把图理解成一个新的 trajectory schema：

```text
execution segment
  ↓ context budget 不足
summary segment
  ↓ reconstructed context = summary + recent turns
execution segment
  ↓
summary segment
  ↓
...
final reward
```

这个图的意义不是“中间插入摘要”，而是说明 RL sample 不再是单段 response。每条样本携带：

- 原始完整 trace；
- compacted trace；
- compaction trigger points；
- summary tokens；
- segment boundaries；
- shared final reward；
- cross-segment advantage。

这会直接影响 trajectory store、loss 计算、rollout replay、debug 工具和评估复现。

---

## 工程价值

CompactionRL 把 long-context agent training 从“把上下文窗口拉大”转成“在固定上下文预算下训练可压缩的交互状态”。这对 infra 的价值很大：

- 降低 long-horizon rollout 对超长 context window 的硬依赖；
- 让 summary / memory 不再只是 inference-time heuristic；
- 把 trajectory segmentation、summary generation、credit assignment 纳入训练系统；
- 为 coding agent / terminal agent 这类长时程任务提供更可控的训练方式；
- 说明 long-context RL 的核心瓶颈不只是 KV cache，还有训练数据结构和 reward 传播。

对平台工程师来说，最值得关注的是：如果引入 compaction，数据格式、loss 计算、sample replay、debug trace、checkpoint metadata 都要跟着变。

---

## 对训练基础设施的影响

### Rollout Runtime

rollout worker 需要支持动态 compaction trigger，而不是固定生成直到结束。每次 compaction 都要记录：

- 触发时剩余 context budget；
- 被压缩的历史范围；
- summary text / tokens；
- recent turns；
- reconstructed context；
- 后续执行结果。

### Trajectory Store

trajectory 不能只存 prompt/response/reward。至少要能恢复完整 trace 和 compacted trace，否则无法复盘 summary 是否丢信息。

### Trainer

trainer 需要按 segment 构造 batch，但又不能把 segment 当独立 episode。loss normalization 和 advantage estimation 必须知道 segment boundary 和原始 rollout 归属。

### Evaluator

评估必须区分 single-window 和 compaction-enabled。论文也指出 CompactionRL 的收益不一定迁移到禁用 compaction 的 single-window evaluation，这说明 train/test execution mode 必须一致。

### Observability

需要新增指标：

- average compactions per trace；
- summary length；
- compaction-triggered task pass rate；
- post-compaction failure rate；
- summary token ratio；
- per-segment advantage distribution；
- overlong rate；
- tool calls per trace。

---

## 今天的应用场景

最直接的场景是 coding agent RL：

- SWE-bench / Terminal-Bench；
- 长时间 terminal interaction；
- 多轮 tool call；
- 需要记住错误日志、文件路径、patch history 的任务；
- long-horizon rollout 超出固定 context budget 的任务。

它也可能扩展到 browser agent、research agent、data analysis agent 和任何长时程多步骤任务。但论文实验主要集中在代码类 benchmark，迁移到其他 agent domain 时需要重新验证 summary 是否能保留 task-relevant state。

---

## 后续演进

这篇论文把几条线接在一起：

```text
Long-context Agent
  ↓
Context Compaction / Memory / Summary
  ↓
Rollout Segmentation
  ↓
PPO + Token-level Loss
  ↓
Cross-trajectory GAE
  ↓
Agentic RL Infra
```

后续值得跟：

- CompactionRL 与 ReSum / SUPO / Context-Folding 的关系；
- compaction-aware replay buffer；
- summary quality 的自动诊断；
- verifier / reward 是否能直接评估 summary 可用性；
- 在 vLLM/SGLang rollout engine 中如何支持 compaction trigger；
- 与 long-context training、CP、KV cache、prefix cache 的组合。

---

## 相关论文

- [Transformer](transformer.md)
- [FlashAttention](flashattention.md)
- [Megatron-LM](megatron_lm.md)
- [DeepSeek-R1](../tech_reports/deepseek_r1.md)
- [GLM-4.5: Agentic, Reasoning, and Coding Foundation Models](https://arxiv.org/abs/2508.06471)
- AReaL: https://arxiv.org/abs/2505.24298
- HybridFlow / verl: https://arxiv.org/abs/2409.19256

---

## 相关代码

- 论文训练框架：使用 `slime`，论文描述为 open-source asynchronous RL framework。
- 评估环境：Harbor environment、Terminus-KIRA agent scaffold。
- 目前未在论文页确认独立的 CompactionRL 官方代码仓库。

推荐阅读入口：

- rollout collection 中 compaction trigger 和 reconstructed context；
- training data schema 中 segment boundary；
- PPO loss 的 token-level normalization；
- cross-trajectory GAE；
- evaluation 中 single-window vs compacted setting。

---

## 面试高频问题

1. CompactionRL 解决的核心问题是什么？
2. 为什么 long-horizon agent 不能只靠扩大 context window？
3. context compaction 和普通 prompt summarization 有什么区别？
4. 为什么 summary token 要进入 RL objective？
5. 为什么论文选择 PPO，而不是 GRPO 这类 group-wise 方法？
6. compaction 为什么会破坏 group-wise advantage estimator 的假设？
7. token-level loss normalization 解决什么偏差？
8. cross-trajectory GAE 为什么必要？
9. execution segment 和 summary segment 的 reward 如何分配？
10. 为什么 CompactionRL 的收益不一定迁移到 single-window evaluation？
11. 它和 long-context training / CP / KV cache 的关系是什么？
12. 如果 summary 丢掉关键错误日志，训练系统如何暴露？
13. 在 rollout infra 中如何记录 compacted trajectory？
14. 如果 compaction 次数过多，会有哪些副作用？
15. 它对 coding agent RL 的价值为什么特别大？

---

## 生产环境思考题

1. trajectory store 如何同时保存 full trace 和 compacted trace？
2. rollout worker 何时触发 compaction：固定阈值、动态阈值，还是基于任务状态？
3. summary 过短导致丢信息，summary 过长导致节省不明显，如何监控？
4. compaction 后任务失败，如何判断是 summary 错还是 execution policy 错？
5. 如果 trainer 只看到 segment，看不到完整 rollout，会不会破坏 reward attribution？
6. 如何把 compaction metadata 写进 checkpoint / replay buffer？
7. vLLM / SGLang rollout engine 是否需要暴露 context budget 和 KV cache 状态？
8. 多次 compaction 后如何避免 summary drift？
9. 如果 reward/verifier 很慢，compaction 是否会进一步放大 rollout latency tail？
10. 如果不同模型版本生成 summary，policy version 和 summary version 如何记录？
11. compaction-enabled training 和 normal serving 模式不一致，线上部署如何处理？
12. 如果任务包含敏感上下文，summary 是否会改变审计和数据合规要求？
13. 如何设计 dashboard 观察 compaction count、summary length、post-compaction success rate？
14. 是否应该给 summary 单独做 SFT warmup，再进入 RL？
15. 如果要在自研 RL pipeline 中复现，最小改动应先落在哪：rollout schema、loss、还是 evaluator？

---

## 我的总结

CompactionRL 的价值不在“会总结上下文”，而在把 context compaction 变成 long-horizon agent RL 的训练对象。它让 summary、execution、segment boundary 和 final reward 进入同一个训练系统，迫使 infra 重新设计 trajectory schema、loss normalization、credit assignment 和 rollout observability。对做 RL Infra 的工程师来说，这篇是理解“长上下文 agent 训练为什么不只是调大 context length”的关键材料。

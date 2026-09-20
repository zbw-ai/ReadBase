# Engineering Blogs Tracking

用于追踪近期大厂工程博客、官方文档和 release note。完整沉淀入口在 [Engineering Blogs](../engineering_blogs/README.md)。

## 模板

```text
## YYYY-MM-DD

### 标题

- 来源：NVIDIA / OpenAI / Anthropic / Hugging Face / DeepSeek / Google / Meta / Microsoft / ByteDance / Zhipu / Other
- 链接：
- 类型：engineering blog / official docs / release note
- 影响等级：
- Decision：Ignore / Observe / Read / Deep Dive
- Reason：
- 建议动作：
- Status：
- 关联主题：
- 一句话价值：
- 是否需要进入 engineering_blogs/：
```

## Backlog

暂无。

## 2026-09-20 定向精读：Z.ai Infra Agent

- Source ID: `zai:glm-built-its-inference-infrastructure`
- First seen: 2026-09-20（本记录首次核验，不声称全网或仓库首次发现）。
- 来源 / 类型 / 日期：[Z.ai 官方工程博客](https://z.ai/blog/glm-built-its-inference-infrastructure)，2026-09-17；无个人署名。
- 范围：用户指定材料及相关官方页面、代码；不是全域扫描，不变更 `scan_log.md` 游标或既有 Accepted 计数。
- 影响等级：高；Decision: Deep Dive。
- Reason: 将 Infra Agent 效果拆成可检查的数值、并发与性能问题，沉淀证据分层方法。
- 关联主题：[Agentic RL](../topics/agentic_rl.md)、[Context Parallelism](../topics/context_parallelism.md)。
- 一句话价值：用可证伪实验区分机制可信、生产收益可信与自主能力可信。
- 下一步：[报告与待执行验证计划](../engineering_blogs/zhipu/glm_infra_agent_recursive_self_improvement.md)；[P1](../reading_queue/P1.md) 跟进源码和实验。
- Status: DIGESTED；未复现。

# AReaL Agentic RL 面试流程图改版设计

## 目标

重绘 `RESUME-08` 的 Mermaid 图，使其适合在 Markdown 文档中完整阅读，同时满足：

- 对齐项目实际的 AReaL online proxy + controller-owned cohort admission 链路；
- 第一眼能看懂三类系统边界和主数据流；
- 能直接定位最大瓶颈 `ready cohort barrier`；
- 保留异步训练中的 weight feedback、policy version 和 staleness 语义；
- 避免把源码类名、HTTP endpoint 和所有异常路径都堆进主图。

## 采用方案

采用用户选择的 **A：三层泳道**，自上而下组织为：

1. **External Episode Producer**：Task、Evals Agent、Tool/Sandbox、terminal reward；
2. **AReaL Rollout & Control**：Gateway/CohortManager、Proxy/InteractionCache、vLLM/SGLang、Ready Cohort Barrier；
3. **Trainer & Policy Feedback**：prepare/export、trajectory batch、score/update、versioned weight sync。

读者先沿每层从左到右阅读，再沿竖向箭头理解 episode 如何形成可训练 cohort，最后沿一条反馈箭头理解新权重如何回到 rollout backend。

## 视觉层级

- 每个泳道使用同色系浅色背景，不给同级节点随机着色；
- 正常主链使用实线箭头；异步等待和权重反馈使用虚线或强调色箭头；
- `Ready Cohort Barrier` 使用唯一的红色强调，旁边附三项原因：long-context late turns、last-of-8 straggler、sandbox/retry/rejection；
- 节点标题使用业务概念，源码类名作为第二行小字，例如 `Batch Preparation` 下标注 `actor.prepare_batch()`；
- Checkpoint/Eval 降级为 weight update 后的灰色旁路，不进入主反馈环。

## 信息架构

### 第一层：External Episode Producer

主链：

`Task / Dataset → Evals Agent Runtime ↔ Tool / Sandbox → Terminal Reward`

向下通过 Gateway 的 session API 连接 AReaL。图中明确说明 agent/env 状态属于外部 evals runtime，而不是 AReaL trainer 内部固定阶段。

### 第二层：AReaL Rollout & Control

主链：

`Gateway + Cohort Admission → Proxy + InteractionCache ↔ vLLM/SGLang → Ready Cohort Barrier`

Gateway/CohortManager 节点标注 grouping、capacity、rollout version 和 staleness；Proxy 节点标注 token、behavior logp、token version 和 reward 的记录职责。Ready Cohort 只有在完整 group rewarded/ended 并通过 ready-time staleness gate 后形成。

### 第三层：Trainer & Policy Feedback

主链：

`Wait & Export → Trajectory Redistribute → Ref/Teacher/Advantage → PPO/GRPO Update → Versioned Weight Sync`

训练后端以 `Megatron / FSDP / Archon` 作为小字边界说明。权重同步箭头返回 inference backend，并明确语义为 `transfer succeeds → set_version`。

## 正确性约束

- Online 模式的 rollout producer 是外部 evals/agent runtime，不画成 trainer 内部 `Agent Workflow`；
- Tool/Sandbox 属于 agent/environment，不画成 AReaL 固定服务；
- vLLM/SGLang 是 inference backend，不是位于 workflow 之后的独立 rollout 阶段；
- reward 在 episode terminal 由环境提交，并与 interaction/session 数据合并导出；
- 不创造统一的 `Training Queue` 节点；训练侧表述为等待 ready cohort 并形成 batch；
- 顺序必须是 optimizer update → versioned weight transfer → 成功后 set version；
- checkpoint/eval 是旁路；
- 最大瓶颈表述为 trainer 暴露的 ready-cohort wait，而不是单独归因于 decode。

## 验证

- Mermaid fence 和 Git diff 无格式错误；
- 图中节点与本地 AReaL 项目代码调用顺序逐项核对；
- 在 GitHub/Markdown 渲染宽度下保持自上而下阅读，不依赖横向滚动；
- 文本回答与图中术语一致；
- 改版只触及 `RESUME-08`，不改其他面试题。

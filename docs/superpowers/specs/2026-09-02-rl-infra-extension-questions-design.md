# RL Infra 延伸题与 Gateway 项目口径设计

## 1. 目标

在现有面试准备体系内补齐 RL Infra 下一阶段的高频问题，并保持用户已确认的阅读方式：

1. 先按 Topic 分 Part；
2. 每个 Part 内再按 Core / P0 / P1 / P2 排序；
3. 主文档只保留能直接口述的简明答案；
4. 原理、代码证据和选型细节沉淀到已有专题文档，通过相对链接回链；
5. 题目必须区分框架事实、项目事实和个人 ownership，不能把团队或上游工作写成个人实现。

本轮不修改简历，不新增分散的 Topic 文档，不重做能力地图。

## 2. 修改范围

### 2.1 主入口

- `private_resume/2026-08-llm-infra-interview-prep.md`

### 2.2 详细专题

- `training-infra-roadmap/topics/fsdp.md`
- `training-infra-roadmap/topics/agentic_rl.md`
- `training-infra-roadmap/topics/rl_framework_selection.md`

### 2.3 代码证据来源

- `/Users/zengbw/Codebase/for_agentic_rl/trail_renew_0720`

代码仓仅用于只读核验，不修改其中任何文件，也不处理其现有未提交改动。

## 3. 题目迁移与优先级

### 3.1 Part II：Megatron、MoE 与训练后端

| 动作 | 优先级 | 题目 | 处理方式 |
|---|---:|---|---|
| 重写并前移 | P0 | FSDP/FSDP2 与 ZeRO-1/2/3 的区别和联系 | 用状态分片对象、执行模型、参数生命周期和适用规模建立统一坐标系；替换现有过短的 P2-01，避免重复题 |
| 新增 | P0 | Megatron 与 FSDP/FSDP2 如何选型 | 从模型是否单卡可容纳、是否需要 TP/PP/CP/EP、模型改造成本、弹性与生态、checkpoint 和故障排查回答 |
| 新增 | P1 | MBridge 是什么，与 Megatron Bridge 有何关系 | 明确二者是不同 package/implementation，而非别名；说明 HF 与 Megatron-Core 之间模型构建和权重转换的桥接作用，以及当前代码仓为何同时保留 |

详细内容进入 `topics/fsdp.md`。MBridge 题同时回链框架选型专题，但不新建独立文档。

### 3.2 Part III：RL 算法、verl 架构与推理后端

| 动作 | 优先级 | 题目 | 处理方式 |
|---|---:|---|---|
| 新增 | P0 | 用最简单的话描述 PPO、GRPO、DAPO | 先给一句话，再说明 advantage、reference/KL、group-relative 和 DAPO 的工程改进；避免把 DAPO说成完全独立于 PPO/GRPO 的新范式 |
| 扩写 | P0 | verl 如何组织不同训练/推理后端与任务 | 扩写现有 VERL-01，解释 high-level single-controller、WorkerGroup/Ray 资源编排和 worker 内 SPMD engine 的职责边界 |
| 扩写 | P0 | colocate 与 disaggregate 是什么 | 扩写现有 VERL-02，区分逻辑角色、物理 GPU 放置和时间复用；说明它相关但不等同于“训推共卡 vs 异构部署” |
| 前移并扩写 | P0 | vLLM 与 SGLang 如何选型 | 将现有 VERL-09 从 P1 前移到 P0，按 rollout 接入、并发/缓存、Agent 工作负载、版本兼容与运维成熟度回答 |
| 新增 | P1 | verl 0.7.0 及以后主要迭代 | 以官方 release/tag 为准，按 rollout、scheduler、weight sync、data/trajectory、checkpoint/recovery 和 backend 分类；只写能核验的主要变化 |
| 扩写 | P1 | streaming、partial rollout、staleness 分别是什么 | 以现有 VERL-04 为入口，补齐三个概念的对象、触发时机、收益、正确性代价和相互关系，并回链 AReaL 的相关题 |

算法基础与异步语义进入 `topics/agentic_rl.md`；controller、placement、推理后端和版本演进进入 `topics/rl_framework_selection.md`。

### 3.3 Part IV：AReaL、Gateway 与 Agentic RL

| 动作 | 优先级 | 题目 | 处理方式 |
|---|---:|---|---|
| 新增 | P0 | 结合代码仓，你对 Gateway 做了哪些修改，逻辑发生了什么变化 | 用“原始问题 → 机制变化 → 正确性约束 → 性能/稳定性结果 → 个人边界”回答；引用个人 commit 和对应代码路径，不把上游 online proxy/cohort 基础架构算作个人原创 |
| 新增 | P1 | Agent 如何通过 OpenAI-compatible Gateway 从框架外部接入 | 讲清 `start_session → chat/completions → set_reward → end_session`，以及 admin/session key、Proxy Worker、InteractionCache、CohortManager、Trainer 的连接关系 |

详细内容进入 `topics/agentic_rl.md` 和 `topics/rl_framework_selection.md`，不新建 Gateway 专题。

## 4. 主文档题量调整

本轮预计新增 6 道唯一问题，并对 2 道现有问题扩写、1 道现有问题前移，不重复计数：

| Part | 修改前 | 修改后 | 变化 |
|---|---:|---:|---:|
| Part I：个人定位与项目口径 | 7 | 7 | 0 |
| Part II：Megatron、MoE 与训练后端 | 19 | 21 | +2 |
| Part III：RL 算法、verl 与推理后端 | 12 | 14 | +2 |
| Part IV：AReaL、Gateway 与 Agentic RL | 14 | 16 | +2 |
| Part V：交付、排障与综合设计 | 8 | 8 | 0 |
| 合计 | 60 | 66 | +6 |

预计优先级分布：

- Part II：Core 3 / P0 15 / P1 5 / P2 1；
- Part III：Core 1 / P0 9 / P1 4 / P2 1；
- Part IV：Core 2 / P0 7 / P1 8 / P2 1。

实施时必须根据实际标题和锚点重新统计，不能机械沿用预计值。

## 5. 统一答题结构

每道新增或重写题保持主文档现有结构，最低包含：

1. **问题**：面试官原始问法；
2. **面试官意图**：他在验证什么能力；
3. **先说结论**：20–40 秒能说完；
4. **展开回答**：3–5 个有因果关系的要点；
5. **项目口径或知识边界**：哪些是实际做过，哪些只是理解；
6. **高概率追问**；
7. **危险回答**；
8. **延伸阅读**：只链到三个已有专题中的相关锚点。

主文档避免长篇 release notes、API 清单和源码逐行解释。详细专题按“问题 framing → 机制 → 选型/配置 → 生产风险 → 排障 → 面试问法”展开。

## 6. MBridge 内容边界

回答必须明确区分三个概念：

- **Megatron-Core**：分布式训练核心库，负责 Transformer 模型并行、执行与训练状态；
- **mbridge**：代码仓当前保留的兼容性 bridge package；
- **NVIDIA Megatron Bridge**：NVIDIA 维护的独立 bridge package，负责 Hugging Face 与 Megatron-Core 模型/权重互操作，并提供较新的架构和 PEFT/LoRA 能力。

项目口径只说：在 AReaL 的 Megatron engine 中按 `bridge_type` 选择实现，用于模型构建、HF/Megatron 权重导入导出和相关 rollout weight flow 的适配。不能声称设计了任一 bridge，也不能把 bridge 说成一种并行策略。

选型要点以本地代码和官方文档双重核验：

- 新 GPU workflow 通常优先 NVIDIA Megatron Bridge；
- `mbridge` 保留向后兼容，在部分 disk-based weight broadcast 和 tree-attention 路径仍有现实约束；
- 具体支持矩阵随版本变化，主文档只保留稳定结论，版本细节放专题并标注来源时间。

## 7. Gateway 项目口径与 ownership

### 7.1 总体叙事

把 Gateway 的变化概括为：

> 从 OpenAI-compatible 请求路由器，演进为理解训练 step、domain quota、cohort/session 生命周期、policy version、staleness、重试和恢复语义的 training-aware admission/control plane。

### 7.2 团队/上游基础能力

以下作为背景描述，不归为个人原创：

- OpenAI-compatible proxy；
- online session 与 cohort 基础链路；
- Proxy Worker、InteractionCache、CohortManager 和 trainer consumer 的总体结构；
- 团队提交 `64adce36` 引入的异步 rollout/cohort/tracing 基础能力。

### 7.3 可归入个人 ownership 的代码变化

以 `zbw-ai` authored commits 和实际 diff 为证据，围绕以下四类组织，不逐条背 commit：

1. **精确配额与公平性**：step-level exact domain quotas、reservation/claim/session/export、避免 supply-driven starvation，成功更新后才提交 plan；
2. **session/cohort 正确性**：reward 与 proxy completion 绑定、rejected cohort 的 in-flight session 保留、reward/end 顺序兼容、zero-interaction fail closed；
3. **liveness 与安全重试**：禁止持锁 long-poll、控制 RPC bounded timeout、错误 domain fail-fast、端口重试、仅 structured pre-binding quota miss 可换 slot/requeue；
4. **吞吐恢复**：rotation admission 下调节 requeue throttle、worker reset 和 partial deadline，避免正确性修复造成 step throughput 退化。

核心个人提交证据包括：`10a3e264`、`9979a0f6`、`c83de5fa`、`e7373e8b`、`afb1882c`、`eb8bd492`、`1162029d`、`b117b570`、`690816eb`、`30ab40c4`、`21bb4862`。实施时需再次核对 author、diff 和文件路径。

### 7.4 面试表达边界

- 使用“我负责/我修改”时，必须能落到个人 commit、测试或生产问题；
- 团队完成的基础架构使用“项目基于/团队已有”；
- 不把 API 兼容等同于 RL correctness：训练还要求 session identity、reward attribution、trajectory completeness、policy version/staleness 和 safe retry；
- 不暴露业务敏感数据、内部域名或未脱敏配置。

## 8. 技术事实与来源策略

对可能随版本变化的内容，写入前核验官方 primary source：

- PyTorch 官方 FSDP/FSDP2 文档；
- DeepSpeed ZeRO 官方文档或论文；
- NVIDIA Megatron-Core / Megatron Bridge 官方文档；
- `mbridge` 官方仓库或本地已锁版本文档；
- verl 官方 v0.7.0、v0.8.0、v0.9.0 release/tag；
- vLLM 与 SGLang 官方文档；
- PPO、GRPO、DAPO 原始论文或官方项目材料；
- AReaL v2.1 官方代码/文档和本地项目代码。

版本演进题不把普通 bugfix 堆成“主要特性”，只保留对架构、性能、正确性或运维方式有明显影响的变化。无法直接核验的内容标为项目推断，不写成框架事实。

## 9. 导航与链接

实施后需要同步更新：

- 主文档顶部 Part 题量、P0 学习路径和三天学习顺序；
- Part II/III/IV 的内部目录或快速跳转；
- 三个专题之间的双向链接；
- `training-infra-roadmap/README.md` 或 `MASTER_READING_LIST.md` 仅在现有入口无法到达相关专题时补链，不为了本轮改动扩大导航范围。

## 10. 验收标准

完成前必须验证：

1. 主文档共有 66 道唯一问题，实际数字与顶部统计一致；
2. 所有新增题先按 Topic 分 Part，再按 Core/P0/P1/P2 排列；
3. `P2-01` 已重写前移而非复制，`VERL-09` 已前移且没有残留重复标题；
4. 三个详细专题均可从主文档点击到达，并能回到主文档相关题；
5. MBridge、Megatron Bridge、Megatron-Core 三者没有混淆；
6. Gateway 回答能明确区分团队基础能力和个人 ownership；
7. 个人代码事实可由 commit author、diff 和测试定位；
8. verl 版本、框架选型和算法描述均有 primary source 支撑；
9. Markdown 相对链接和本地图片路径全部可解析；
10. Git diff 不包含代码仓、简历 DOCX 或无关文件改动。

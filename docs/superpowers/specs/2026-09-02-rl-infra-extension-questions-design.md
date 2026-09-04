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
| 重写并前移 | P0 | `DIST-01`｜FSDP/FSDP2 与 ZeRO-1/2/3 的区别和联系 | 用状态分片对象、执行模型、参数生命周期和适用规模建立统一坐标系；替换现有过短的 `P2-01`，避免重复题 |
| 新增 | P0 | `MEGATRON-11`｜Megatron 与 FSDP/FSDP2 如何选型 | 从模型是否单卡可容纳、是否需要 TP/PP/CP/EP、模型改造成本、弹性与生态、checkpoint 和故障排查回答 |
| 新增 | P1 | `BRIDGE-01`｜MBridge 是什么，与 Megatron Bridge 有何关系 | 明确二者是不同 package/implementation，而非别名；说明 HF 与 Megatron-Core 之间模型构建和权重转换的桥接作用，以及当前代码仓为何同时保留 |

详细内容进入 `topics/fsdp.md`。MBridge 题同时回链框架选型专题，但不新建独立文档。

### 3.2 Part III：RL 算法、verl 架构与推理后端

| 动作 | 优先级 | 题目 | 处理方式 |
|---|---:|---|---|
| 新增 | P0 | `RL-ALGO-01`｜用最简单的话描述 PPO、GRPO、DAPO | 先给一句话，再说明 advantage、reference/KL、group-relative 和 DAPO 的工程改进；避免把 DAPO 说成完全独立于 PPO/GRPO 的新范式 |
| 扩写 | P0 | `VERL-01`｜verl 如何组织不同训练/推理后端与任务 | 解释 high-level single-controller、WorkerGroup/Ray 资源编排和 worker 内 SPMD engine 的职责边界 |
| 扩写 | P0 | `VERL-02`｜colocate 与 disaggregate 是什么 | 区分逻辑角色、物理 GPU 放置和时间复用；说明它相关但不等同于“训推共卡 vs 异构部署” |
| 扩写 | P0 | `VERL-04`｜fully async、streaming、partial rollout 与 staleness 如何配合 | 保留为现有 P0 题，不新增、不降为 P1；补齐四个概念的对象、触发时机、收益、正确性代价和相互关系，并回链 AReaL 的相关题 |
| 前移并扩写 | P0 | `VERL-09`｜vLLM 与 SGLang 如何选型 | 从 P1 前移到 P0，按 rollout 接入、并发/缓存、Agent 工作负载、版本兼容与运维成熟度回答 |
| 新增 | P1 | `VERL-10`｜verl 0.7.0 及以后主要迭代 | 以官方 release/tag 为准，按 rollout、scheduler、weight sync、data/trajectory、checkpoint/recovery 和 backend 分类；只写能核验的主要变化 |

算法基础与异步语义进入 `topics/agentic_rl.md`；controller、placement、推理后端和版本演进进入 `topics/rl_framework_selection.md`。

### 3.3 Part IV：AReaL、Gateway 与 Agentic RL

| 动作 | 优先级 | 题目 | 处理方式 |
|---|---:|---|---|
| 新增 | P0 | `AREAL-09`｜结合代码仓，你对 Gateway 做了哪些修改，逻辑发生了什么变化 | 用“原始问题 → 机制变化 → 正确性约束 → 可核验结果 → 个人边界”回答；引用个人 commit 和对应代码路径，不把上游 online proxy/cohort 基础架构算作个人原创 |
| 新增 | P1 | `AREAL-10`｜Agent 如何通过 OpenAI-compatible Gateway 从框架外部接入 | 讲清 `start_session → chat/completions → set_reward → end_session`，以及 admin/session key、Proxy Worker、InteractionCache、CohortManager、Trainer 的连接关系 |

详细内容进入 `topics/agentic_rl.md` 和 `topics/rl_framework_selection.md`，不新建 Gateway 专题。

## 4. 主文档题量调整

本轮净新增 6 道唯一问题，现有题的完整迁移动作为：

- `P2-01` 重写并前移，改号为 `DIST-01`；
- `VERL-01`、`VERL-02`、`VERL-04` 扩写，原 ID 不变；
- `VERL-09` 前移并扩写，原 ID 不变。

六道净新增题为：`MEGATRON-11`、`BRIDGE-01`、`RL-ALGO-01`、`VERL-10`、`AREAL-09`、`AREAL-10`。

最终 Part 题量是规范性验收值：

| Part | 修改前 | 修改后 | 变化 |
|---|---:|---:|---:|
| Part I：个人定位、Ownership 与职业选择 | 7 | 7 | 0 |
| Part II：Megatron、MoE、训练后端与长上下文 | 19 | 21 | +2 |
| Part III：RL 算法、verl 与 Fully Async RLVR | 12 | 14 | +2 |
| Part IV：AReaL、Gateway、Agentic RL 与 MOPD | 14 | 16 | +2 |
| Part V：通用 Infra 与生产排障 | 8 | 8 | 0 |
| 合计 | 60 | 66 | +6 |

最终优先级分布也是规范性验收值：

- Part I：Core 3 / P0 3 / P1 3 / P2 1；
- Part II：Core 3 / P0 15 / P1 5 / P2 1；
- Part III：Core 1 / P0 9 / P1 4 / P2 1；
- Part IV：Core 2 / P0 7 / P1 8 / P2 1；
- Part V：Core 1 / P0 2 / P1 5 / P2 1。

这里的 **Core 是 P0 的子集**，不是额外题目；例如 Part III 的 14 道题按唯一题计数为 `P0 9 + P1 4 + P2 1`，其中 1 道 P0 同时标为 Core。若实施时发现现有题量基线或唯一题定义有误，必须先修正规格并重新审查，不能自行偏离这些验收值。

### 4.1 最终 Part 标题

六个 Part 的最终标题固定为：

1. Part I｜个人定位、Ownership 与职业选择；
2. Part II｜Megatron、MoE、训练后端与长上下文；
3. Part III｜RL 算法、verl 与 Fully Async RLVR；
4. Part IV｜AReaL、Gateway、Agentic RL 与 MOPD；
5. Part V｜通用 Infra 与生产排障；
6. Part VI｜面试应变与查漏补缺。

Part VI 是学习和应变方法，不包含独立题目，因此不计入 66 道题。

### 4.2 题号与 anchor 兼容

- `P2-01` 改号为 `DIST-01`，主导航和所有仓库内入链改到 `#dist-01`；
- 在 `DIST-01` 标题前保留显式兼容 anchor `<a id="p2-01"></a>`，避免旧书签立即失效；
- 其余现有题保留原 ID；
- 六道新题使用第 4 节声明的唯一 ID，不能在别处复用；
- 验收同时检查题号唯一、显式 anchor 唯一、仓库内链接可达和旧 `#p2-01` 兼容入口可达。

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

- 对本地 AReaL 锁定版本而言，且新增 workflow 不受 disk-based HF I/O 或 tree-attention 限制时，优先评估 NVIDIA Megatron Bridge；这不是跨项目、跨版本的普遍规则；
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
4. **吞吐保护机制**：rotation admission 下调节 requeue throttle、worker reset 和 partial deadline，目标是避免正确性修复造成 step throughput 退化；只有在存在 run log、指标或 benchmark 时，才能进一步表述为“吞吐恢复”。

核心个人提交证据包括：`10a3e264`、`9979a0f6`、`c83de5fa`、`e7373e8b`、`afb1882c`、`eb8bd492`、`1162029d`、`b117b570`、`690816eb`、`30ab40c4`、`21bb4862`。实施时需再次核对 author、diff 和文件路径；其中 `10a3e264` 与 `9979a0f6` 先按“可能位于不同分支或演进阶段的同类 exact-quota 实现”处理，比较 patch-id/改动范围后再决定是否合并为一项证据，不能按两项独立成果累计。

证据必须分层使用：

- **commit author、diff、test**：证明个人 ownership、机制和回归覆盖；
- **run log、指标、benchmark**：证明性能或稳定性结果；
- 缺少第二层证据时，只说“实现了保护/缓解机制”或“修复了可复现故障”，不声称吞吐已经恢复或生产指标已经提升。

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

1. 主文档共有 66 道唯一问题，Part 分布严格为 `7 / 21 / 14 / 16 / 8`，实际数字与顶部统计一致；
2. 所有新增题先按 Topic 分 Part，再按 Core/P0/P1/P2 排列；
3. `P2-01` 已重写前移为 `DIST-01` 而非复制，旧 anchor 可达；`VERL-09` 已前移且没有残留重复标题；
4. 三个详细专题均可从主文档点击到达，并能回到主文档相关题；
5. MBridge、Megatron Bridge、Megatron-Core 三者没有混淆；
6. Gateway 回答能明确区分团队基础能力和个人 ownership；
7. 六道新题 ID、全部题号和显式 anchor 均唯一，仓库内入链已更新；
8. 个人代码 ownership 可由 commit author、diff 和测试定位；性能/稳定性结果只在有 run log、指标或 benchmark 时陈述；
9. verl 版本、框架选型和算法描述均有 primary source 支撑；
10. Markdown 相对链接和本地图片路径全部可解析；
11. Git diff 不包含代码仓、简历 DOCX 或无关文件改动。

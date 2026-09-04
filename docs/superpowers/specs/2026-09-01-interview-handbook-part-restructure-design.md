# 面试手册 Part 化重构与简历口径校准设计

## 目标

把 `private_resume/2026-08-llm-infra-interview-prep.md` 从“先按优先级分块、再按框架分块”的混合结构，重构为一条可以按主题连续学习、也可以在面试中沿追问链展开的路径；同时以最新版简历和本人确认信息校准 MOPD、CUDA Graph、SFT、checkpoint、Megatron-Core 和 Fully Async 六项口径。

## 修改范围

- 主文档：`private_resume/2026-08-llm-infra-interview-prep.md`
- 项目事实底稿：`private_resume/2026-08-xpeng-infra-resume-materials.md`
- 能力地图文字：`private_resume/assets/llm-infra-personal-capability-map.svg`
- 不修改最新版简历 DOCX，不批量改历史简历草稿，不重做能力地图版式。

## 信息架构

正文按主题 Part 组织，而不是先把全部 P0 放在一起、再把全部 P1 放在一起：

1. **Part I｜个人定位、Ownership 与职业选择**：自我介绍、Ownership、职业选择、薪资档位和高级工程师协作能力。
2. **Part II｜Megatron、大规模训练与长上下文**：X1 约 200B MoE、5D 并行、TP/SP/CP/PP/EP、SFT 31s→9.3s、128K/256K、显存账本、规模交付。
3. **Part III｜verl 与 Fully Async RLVR**：框架选型、Actor/Rollout 编排、同步与异步、资源供需配平、权重同步和 RLVR 正确性。
4. **Part IV｜AReaL、Agentic RL 与 MOPD**：online proxy/cohort 链路、rollout 长尾、staleness、trajectory lineage、CUDA Graph、Prefix Cache、TILE→MOPD 和三层门禁。
5. **Part V｜通用 Infra 与生产排障**：通信算子、NCCL、checkpoint/recovery、推理/KV cache、可观测性和系统设计题。MFU 作为训练性能口径唯一归入 Part II。
6. **Part VI｜面试应变与查漏补缺**：三框架速查、项目证据卡、模拟面试、反问、最后一小时清单和延伸阅读。

每个技术 Part 内固定采用：`Part 目标 → Core（核心十题在本 Part 的子集）→ P0 扩展 → P1 深挖 → P2 选学 → 追问路线`。Core 是 P0 的优先子集，不是另一套题库。某个层级没有题目时明确写“本 Part 无额外题目”，不复制题目凑层级。Part VI 是演练层，按 Core/P0/P1/P2 四种模拟强度组织，不新增技术题。

### 唯一归属与跨 Part 边界

- 每个题号只在一个 Part 有完整正文，题号和 HTML anchor 保持稳定；全局导航与其他 Part 只链接，不复制答案。
- Part II 负责训练态机制与项目：训练显存、Megatron distributed checkpoint 的拓扑变化、长上下文和规模训练。Part V 负责跨系统生产能力：collective 语义、NCCL hang、完整恢复状态、推理/KV cache 和可观测性。
- Part III 负责 verl/RLVR 的 role 编排、同步/异步资源模型、训练—推理权重转换和 GRPO 数据正确性。Part IV 负责 agent/tool/session/cohort 的 online 数据生产、AReaL staleness、trajectory lineage、MOPD 和 Agentic RL 阶段优化。
- CUDA Graph、Prefix Cache 的项目问法只放 Part IV；通用推理吞吐/KV cache 设计只放 Part V；vLLM/SGLang 作为 verl rollout backend 的选型只放 Part III。
- 相邻主题通过“建议追问”链接串联。例如 Megatron collective 题链接 Part V 通信算子，Fully Async 链接 AReaL staleness，但不复制完整回答。

### 现有题目的完整迁移表

| Part | Core（同时属于 P0） | P0 扩展 | P1 深挖 | P2 选学 |
|---|---|---|---|---|
| I 个人定位 | RESUME-01、RESUME-01B、RESUME-01C | 无额外题目 | RESUME-11、RESUME-16、BEHAVIOR-01 | P2-06 |
| II Megatron/训练 | RESUME-01A、MEGATRON-01、INFRA-02 | RESUME-05/06/07/10、MEGATRON-02/03/04/05/06、INFRA-01 | MEGATRON-07/08/09/10 | P2-01、P2-02 |
| III verl/RLVR | RESUME-02 | RESUME-03、VERL-01/02/03/04/05 | VERL-06/07/08/09 | P2-05 |
| IV AReaL/Agentic RL | RESUME-08、RESUME-09 | AREAL-01/02/03/04 | RESUME-13/14/15、AREAL-05/06/07/08 | P2-04 |
| V 通用 Infra | INFRA-04 | INFRA-03 | RESUME-12、INFRA-05/06/07/08 | P2-03 |
| VI 面试演练 | 核心十题串讲 | 全部 P0 模拟 | JD 定向 P1 追问 | P2 查漏与 coding/system design |

上述映射覆盖现有 31 道 P0、23 道 P1 和 6 道 P2，共 60 道；实施时不改题号，避免锚点和既有链接失效。

## “核心十题”的准确定义

“核心十题”只表示只剩 3 小时时必须能口述的十个入口，不等于全部 P0：

1. RESUME-01 自我介绍
2. RESUME-01A X1 约 200B MoE 性能优化
3. RESUME-01B Ownership
4. RESUME-01C 职业选择
5. RESUME-02 Fully Async RLVR
6. RESUME-08 AReaL Agentic RL 链路
7. RESUME-09 MOPD
8. MEGATRON-01 5D 并行
9. INFRA-02 Megatron 显存账本
10. INFRA-04 通信算子

主导航把这十题单独列出；每题的完整正文只保留在所属 Part 的 Core 区域。其余问题放入各 Part 的 P0/P1/P2 路线，不再接在同一张“核心十题”顺序表后面造成题量歧义。

## 六项事实口径

### MOPD

最终允许对外使用的精确表述是：“最新版双 Teacher MOPD 结果在 SWE、Terminal 双域提升，General 不下降。”这表示 EFFICACY 已有方向性结论，但在 checkpoint、样本量、seed、baseline、评测窗口和统计置信信息补齐前，不追加“双域显著提升”“稳定提升 X pp”或“完整统计闭环”等更强表述。FUNCTIONAL、NUMERIC、EFFICACY 三层门禁仍保留。单 Teacher 的 7.9pp/7.0pp 不外推为双 Teacher 数值。Model Merge baseline 固定写 `TILE merge`，不扩写未确认机制或论文来源。

### CUDA Graph

唯一对外数字是最新版简历中的“35B 真实 RL decode 约 14x”。删除 6–8x 口径；明确这是 decode 阶段收益，不是 rollout 或端到端训练加速。

### SFT 31s→9.3s

写成一个按 profile 推进的联合优化过程：固定 workload 与统计口径；先用 `num_workers=0→8` 和 data prefetch 去掉 Host/DataLoader bubble；再从过重 recompute 收敛到 selective recompute，优先保留重算代价高的 attention/SDPA 结果，重算更便宜且显存收益高的部分；最后收敛 TP/CP，使 TP 不因过大而切碎 GEMM、放大高频 collective，并用 CP 更直接分摊长序列 activation。最终 `31s→9.3s、MFU 23%→29.6%` 是联合结果，不虚构单项贡献。删除现文中“必须拆出每项 A/B”“能拆分两项改动贡献”等无法由现有证据支持的要求，改成“能解释各项解决的瓶颈，独立收益待同 workload A/B 补齐”。`TP=4,CP=4 → TP=2,CP=8` 的 `163s→102s` 是另一条已确认长上下文证据，未确认同一 workload 前不并入 31s→9.3s 的贡献拆分。

### checkpoint 交付

“交付 checkpoint”定义为训练框架和 recipe 已达到稳定训练验收：算法团队可以据此持续开展实验并产出经下游验证的有效模型权重。它明显强于 smoke test，但不自动等于无限期无人值守长稳。回答时用代表性长度分布、连续训练窗口、loss/grad 稳定、save/resume、下游质量和 recipe 可复现说明验收。

### Megatron-Core 个人边界

定位为 feature integration/application layer：掌握 5D 配置、process group 与拓扑推理，完成 Megatron-Core/MBridge 后端在 SFT/RLVR/长上下文/MoE 中的接入、调优和正确性排障。没有实现 collective kernel，没有修改 `parallel_state`/process-group construction，也没有编写 pipeline scheduler。暂不修改简历；面试口述避免“Megatron 核心机制作者”的暗示。

### Fully Async

先区分架构收益与测得收益：同步基线的阶段拆解显示 rollout 占主要时间，async 通过 Trainer/Rollouter 分池、队列解耦、阶段 overlap 和独立扩缩容减少 exposed wait，并降低长 trajectory 对整个同步 step 的阻塞；它不降低单条 trajectory 的生成时延。

项目优化链按事实写：初始 async `3T+1R、gen-TP=4、2 个实例` 只有 76，trainer idle 0.41；调整到 `gen-TP=2、4 个实例`，并配合 `require_batches`、cache 生命周期和 rollout serving 常见能力，使生产率达到 211–255；`2T+2R、8 个实例` 的候选窗口为 236–293，idle 0.10–0.14，瓶颈迁移到 actor update。dynamic batch、chunked prefill、prefix cache、CUDA Graph path、partial rollout、bounded staleness、rollout correction、validation frequency 和 serving limit 等，只列为已启用/联合配置，缺少独立 A/B 时不分摊收益。同步约 200 仅作为初始诊断背景；协议补齐前不声称 async 相比 sync 的提升比例。

## 验收

### 跨文件落点

| 口径 | 主文档 | 项目事实底稿 | SVG |
|---|---|---|---|
| MOPD | 校准区、Part IV、证据卡、清单 | 使用原则、推荐写法、4.5、6.6、故事/风险项 | 将“待闭环”改为“SWE/Terminal ↑ · General ↔”等短标签 |
| CUDA Graph | 校准区、RESUME-13、证据卡、清单 | 推荐写法、4.4 | 不新增数字 |
| SFT 31s→9.3s | 校准区、RESUME-05、证据卡、清单 | 在 SFT 推荐写法/事实展开中补联合过程与边界 | 不承载 |
| checkpoint 交付 | 校准区、RESUME-06、证据卡/清单 | 长上下文与稳定性段落补验收定义 | 不承载 |
| Megatron 边界 | 校准区、相关项目证据、框架速查 | 分布式训练/技能与项目边界 | 不承载 |
| Fully Async | 校准区、RESUME-02、证据卡、清单 | 推荐写法、4.2、故事与待确认项 | 不新增数字 |

证据卡中的 `______` 是刻意保留给本人面试前填写的非阻塞演练字段；实现者不得虚构缺失数据。已确认的方向性结论应直接填入，缺少 A/B 或统计信息的字段标注“待本人补齐”，不影响本次文档结构与口径校准完成。

### 检查标准

- 文档中的“核心十题”正好十道，十个题号在各自 Part 只有一个完整正文位置；其他题目按迁移表唯一归入 Part 和优先级。
- 六个 Part 均有学习目标和 Core/P0/P1/P2 导航；Part I 可明确无额外 P0，Part VI 按四种模拟强度组织。技术 Part 均有追问路线。
- 全部 60 个现有题号都有且仅有一个归属，题号和 anchor 不变；跨 Part 只链接不复制。
- 主文档和底稿不再残留 `6–8x`、`6-8x` 等旧数字排版变体，并正向出现“35B 真实 RL decode 约 14x”及“不是端到端加速”。
- 主文档、底稿和 SVG 不再把 MOPD 写成“EFFICACY 待闭环”“NUMERIC/EFFICACY 未完全闭环”“没有证明多域模型效果提升”“多域最终效果仍待严格评测”或“early canary 仍是最新结论”，并正向出现允许口述的双域/General 结论；不把单 Teacher pp 数字当成双 Teacher 结果。
- `31s→9.3s` 正向包含 DataLoader/prefetch、selective recompute、TP/CP 联合过程和“无单项 A/B”边界；不借用另一 workload 的 `163s→102s` 作为独立贡献。
- checkpoint 正向定义为“可支持算法实验并产出有质量权重的稳定训练验收”，同时保留“不等于无限期无人值守”的边界。
- Megatron 正向定义 feature integration/application layer，并明确未实现 collective kernel、未修改 `parallel_state`/process-group construction、未编写 scheduler。
- Fully Async 正向保留 `76→211–255`、`236–293`、idle 变化、架构收益和配置优化链，同时删除“超过同步”及任何未闭环的 sync 提升倍数。
- Markdown 相对链接可解析，SVG 可通过 XML 解析，`git diff --check` 无错误。

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
5. **Part V｜通用 Infra、性能与生产排障**：MFU、通信算子、NCCL、checkpoint/recovery、推理/KV cache、可观测性和系统设计题。
6. **Part VI｜面试应变与查漏补缺**：三框架速查、项目证据卡、模拟面试、反问、最后一小时清单和延伸阅读。

每个技术 Part 内固定采用：`Part 目标 → 首问入口 → P0 扩展 → P1 深挖 → P2 选学 → 追问路线`。同一主题的问题不再分散在多个优先级章节。

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

主导航把这十题单独列出；其余问题放入各 Part 的 P0/P1/P2 路线，不再接在同一张“核心十题”顺序表后面造成题量歧义。

## 六项事实口径

### MOPD

最终口径为：双 Teacher MOPD 在 SWE、Terminal 双域提升，General 不下降。FUNCTIONAL、NUMERIC、EFFICACY 三层门禁仍保留；EFFICACY 已有方向性结论，但面试前仍需在证据卡补齐 checkpoint、样本量、seed、baseline、评测窗口和统计置信信息。单 Teacher 的 7.9pp/7.0pp 不外推为双 Teacher 数值。Model Merge baseline 固定写 `TILE merge`，不扩写未确认机制或论文来源。

### CUDA Graph

唯一对外数字是最新版简历中的“35B 真实 RL decode 约 14x”。删除 6–8x 口径；明确这是 decode 阶段收益，不是 rollout 或端到端训练加速。

### SFT 31s→9.3s

写成一个按 profile 推进的联合优化过程：固定 workload 与统计口径；先用 `num_workers=0→8` 和 data prefetch 去掉 Host/DataLoader bubble；再从过重 recompute 收敛到 selective recompute，优先保留重算代价高的 attention/SDPA 结果，重算更便宜且显存收益高的部分；最后收敛 TP/CP，使 TP 不因过大而切碎 GEMM、放大高频 collective，并用 CP 更直接分摊长序列 activation。最终 `31s→9.3s、MFU 23%→29.6%` 是联合结果，不虚构单项贡献。`TP=4,CP=4 → TP=2,CP=8` 的 `163s→102s` 是另一条已确认长上下文证据，未确认同一 workload 前不并入 31s→9.3s 的贡献拆分。

### checkpoint 交付

“交付 checkpoint”定义为训练框架和 recipe 已达到稳定训练验收：算法团队可以据此持续开展实验并产出经下游验证的有效模型权重。它明显强于 smoke test，但不自动等于无限期无人值守长稳。回答时用代表性长度分布、连续训练窗口、loss/grad 稳定、save/resume、下游质量和 recipe 可复现说明验收。

### Megatron-Core 个人边界

定位为 feature integration/application layer：掌握 5D 配置、process group 与拓扑推理，完成 Megatron-Core/MBridge 后端在 SFT/RLVR/长上下文/MoE 中的接入、调优和正确性排障。没有实现 collective kernel，没有修改 `parallel_state`/process-group construction，也没有编写 pipeline scheduler。暂不修改简历；面试口述避免“Megatron 核心机制作者”的暗示。

### Fully Async

先区分架构收益与测得收益：同步基线的阶段拆解显示 rollout 占主要时间，async 通过 Trainer/Rollouter 分池、队列解耦、阶段 overlap 和独立扩缩容减少 exposed wait，并降低长 trajectory 对整个同步 step 的阻塞；它不降低单条 trajectory 的生成时延。

项目优化链按事实写：初始 async `3T+1R、gen-TP=4、2 个实例` 只有 76，trainer idle 0.41；调整到 `gen-TP=2、4 个实例`，并配合 `require_batches`、cache 生命周期和 rollout serving 常见能力，使生产率达到 211–255；`2T+2R、8 个实例` 的候选窗口为 236–293，idle 0.10–0.14，瓶颈迁移到 actor update。dynamic batch、chunked prefill、prefix cache、CUDA Graph path、partial rollout、bounded staleness、rollout correction、validation frequency 和 serving limit 等，只列为已启用/联合配置，缺少独立 A/B 时不分摊收益。同步约 200 仅作为初始诊断背景；协议补齐前不声称 async 相比 sync 的提升比例。

## 验收

- 文档中的“核心十题”正好十道，其他题目明确归入 Part 和优先级。
- 六个 Part 均有学习目标、题目层级和追问路线；同一主题不再被 P0/P1 主章节割裂。
- 主文档、事实底稿和能力地图中不再残留 `6–8x`、MOPD “EFFICACY 待闭环”或 async “超过同步”结论。
- `31s→9.3s` 不借用另一 workload 的 `163s→102s` 作为独立贡献。
- Markdown 相对链接可解析，SVG 可通过 XML 解析，`git diff --check` 无错误。

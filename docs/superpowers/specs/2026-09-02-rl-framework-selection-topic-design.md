# RL 框架选型专题设计

## 目标

把“为什么标准 SFT/RLVR 选择 verl、为什么 Agentic RL 转向 AReaL”沉淀成两层内容：

1. 面试主文档只保留可在 1–2 分钟内直接口述的选型结论、决策维度、知识边界和专题链接。
2. 工程手册专题完整解释两个框架的系统思想、架构差异、优劣、选型矩阵和真实项目决策过程。

目标不是给框架做静态排名，而是回答：在特定 workload、团队资产和版本约束下，哪种系统抽象能以更小改造半径交付目标能力。

## 采用方案

新增独立专题 `training-infra-roadmap/topics/rl_framework_selection.md`，不把详细内容继续堆进面试主文档，也不把通用框架选型混入已有 `agentic_rl.md` 的原理主线。

## 信息架构

专题按以下顺序组织：

1. **先给结论**：一句话区分 verl 与 AReaL，并强调两者都能做异步 RL，差异是系统中心和适配成本。
2. **从 workload 出发**：区分 SFT、标准 RLVR、Fully Async RLVR、长时 Agentic RL 和外部 Agent 在线接入。
3. **verl 架构**：Hybrid-Controller 在高层用 single-controller/MPMD 编排，在内部训练与推理 engine 中用 SPMD/multi-controller 执行；同时解释 WorkerGroup、多角色 dataflow、后端与 placement。
4. **AReaL 架构**：先拆开项目当时的 online proxy/cohort 二次开发链路，再解释 AReaL 2.0 的 training、inference、agent、weight-update 微服务里程碑，最后标注当前 2.1 的后续演进；不把三者混成一张无版本架构图。
5. **同一维度比较**：控制面、数据面、Agent 接入、异步机制、后端生态、正确性、恢复和二次开发半径。
6. **优劣与选型矩阵**：明确不同任务的推荐起点以及选择成立的前提。
7. **个人项目决策复盘**：先比较 verl、slime、ROLL 后选择 verl；需求切换到 128K、多轮 tool/sandbox、外部 Agent 和 Gateway 改造后选择 AReaL；同时补齐外围生产能力。
8. **验证方法**：如何用相同 workload 比较 goodput、tail latency、freshness、weight-sync exposed time、恢复与效果。
9. **面试口述**：30 秒和 2 分钟两个版本，高频追问与危险回答。
10. **版本与来源**：只引用官方文档/仓库，区分项目当时版本、verl 0.7 的 experimental Fully Async 路径、AReaL 2.0 架构里程碑和当前 AReaL 2.1；记录 2026-09-02 核验日和对应 tag/commit。

## 主文档边界

`private_resume/2026-08-llm-infra-interview-prep.md` 的 `AREAL-01` 保留：

- 一句话区分；
- 约 90 秒精准回答；
- 选型维度；
- 项目证据或知识边界；
- 高频追问和危险回答；
- 指向专题的单一明确链接。

不在主文档重复架构拆解、完整优劣表、选型矩阵和 benchmark 设计。Part VI 的框架速查表增加专题入口，不复制专题正文。

## 关键事实与口径

- 不说“AReaL 异步、verl 同步”；verl 0.7 已提供 Fully Async 路径，但截至 2026-09-02 仍位于 `verl.experimental`，不能写成稳定默认主路径。
- 不把 verl 简化为纯 single-controller；准确口径是 Hybrid-Controller：高层 single-controller/MPMD 编排，内部 model/rollout engine 以 SPMD/multi-controller 执行。
- 不说“AReaL 全面更先进”；准确说法是它在长时 Agentic RL workload 下的架构起点更贴近项目问题。
- “verl 较重”必须落到改动牵引范围：trainer、worker、数据协议、agent loop、推理后端和 placement 等层的联动，而不是笼统批评。
- “AReaL 更好修改 Gateway”只描述项目当时版本和团队改造路径，不外推成永久结论。
- AReaL 的代价分两层：off-policy correctness、token/trajectory version、staleness、weight sync、partial trajectory 与跨服务恢复属于异步架构固有复杂度；监控、评测、lineage、checkpoint/部署等是项目当时版本和团队交付中需要补齐的外围能力，不能写成当前 2.x 的永久缺陷。
- 项目 online proxy/cohort 链路、AReaL 2.0 微服务架构与 2.1 当前 release 分节表达，不能用 2.x 架构倒推项目历史实现。
- 对 slime、ROLL 只记录当时比较维度，不编造未经项目记录支持的能力排名。

## 导航设计

- `training-infra-roadmap/README.md` 的工程手册章节增加专题入口。
- `training-infra-roadmap/topics/agentic_rl.md` 在相邻系统处链接到选型专题。
- 新专题回链到 Agentic RL 原理章和面试主文档 `AREAL-01`。
- `training-infra-roadmap/KNOWLEDGE_GRAPH.md` 在 Agentic RL 能力分区增加框架选型节点；不扩大全局第一轮主线。
- `training-infra-roadmap/MASTER_READING_LIST.md` 的 AReaL/HybridFlow 条目链接到专题，形成来源到判断的入口。

## 验收标准

- 主文档 `AREAL-01` 能在 1–2 分钟内口述完，且存在明确专题链接。
- 专题能够独立回答架构、优劣、选择原因、版本边界和公平比较方法。
- 所有新增相对链接和锚点可解析，Markdown 图片/图表不引入外部本地依赖。
- 官方链接来自 verl/AReaL 官方文档或 GitHub 仓库。
- 所有动态能力结论标注 `项目当时版本 / 当前官方版本 / 核验日期或 tag/commit`；选型矩阵的关键判断可追溯到官方来源，或明确标成个人项目判断。
- 变更不改动简历文件，不夸大个人对框架底层实现的 ownership。

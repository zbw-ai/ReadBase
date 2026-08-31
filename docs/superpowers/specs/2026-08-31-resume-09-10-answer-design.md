# RESUME-09/10 回答纠偏设计

## 1. 目标

更新 [`private_resume/2026-08-llm-infra-interview-prep.md`](../../../private_resume/2026-08-llm-infra-interview-prep.md) 中两道 P0 项目题：

1. `RESUME-09` 从算法机制起手改为真实项目因果链：多个领域数据分别 RL 得到 Expert，TIES-Merging 未达到多领域能力汇聚目标，因此改用 MOPD。
2. `RESUME-10` 从泛化的集群运维回答改为本人在华为阶段的真实职责：负责客户模型在国产卡上的功能适配与性能达标闭环。

本次只修改私人面试准备 Markdown，不修改对外简历、DOCX 或项目事实底稿。客户名称继续使用 `X1`、`TX` 代号，不在 Markdown 中展开真实名称。

## 2. 共通设计原则

### 2.1 先限定个人边界

- 不把多个 RL Expert、TIES-Merging 或 MOPD 的全部算法研究归为个人原创。
- 不把千卡/万卡项目背景等同于本人负责整个集群平台、硬件运维和全部稳定性问题。
- 主回答明确“我负责的交付对象、工作范围和验收目标”，再展开具体动作。

### 2.2 用闭环证明 ownership

技术回答遵循：

```text
业务/交付目标
  → 基线方案或初始状态
  → 暴露问题
  → 技术选择
  → 实现与验证
  → 结果与证据边界
```

避免只列技术名词，或者用团队整体成果代替个人贡献。

## 3. RESUME-09：为什么从 TIES-Merging 转向 MOPD

### 3.1 面试官意图

验证候选人是否能同时讲清：

- 多领域能力汇聚的真实业务问题；
- 为什么参数空间 Model Merge 不满足目标；
- OPD/MOPD 的 Student rollout、Teacher scoring、路由和 loss 数据流；
- 如何区分系统跑通、数值正确和能力有效；
- 是否会夸大尚未完成的双 Teacher 评测结果。

### 3.2 项目因果链

1. 使用不同领域的数据分别进行 RL，得到多个领域 Expert。
2. 交付目标不是部署多个模型，而是把多个 Expert 的能力汇聚到一个统一模型。
3. 先尝试 `TIES-Merging`。准确名称为 TIES-Merging，其中 TIES 表示 `Trim、Elect Sign、Merge`；不得写成 `tile merge`。
4. 项目中的 TIES-Merging 初步实验没有达到“一个模型同时接近各领域 Expert”的目标。没有确认具体数字前，不写能力下降幅度或 coefficient 敏感度的实测结论。
5. 技术判断：TIES 仍是在参数空间一次性静态合并多个 RL task vector；它可以缓解冗余参数与符号冲突，但不能在能力迁移过程中根据训练样本领域选择监督来源，也不保证各领域行为都被稳定继承。
6. 因此采用 MOPD：各 RL Expert 作为冻结 Teacher，Student 从共同的 RL 前模型初始化；继续使用各领域原 RL 数据训练，并保留 `data_source` 路由信息。
7. Student 用当前 policy 在相应环境中生成 trajectory；系统按 `data_source` 将轨迹路由给对应 Teacher。Teacher 不重新生成答案，而是对 Student 实际走过的相同 token 路径计算 logp。
8. 训练侧根据 Teacher/Student token-level logp 差异构造 OPD 信号，把多个领域 Teacher 的行为能力写入同一个 Student。Teacher 路由只发生在 MOPD 训练期；训练完成后部署的是无需 Teacher 和路由的统一 Student。

### 3.3 为什么 MOPD 更匹配目标

- TIES-Merging 在权重空间做一次静态合并；MOPD 在 Student 实际访问的状态分布上迁移行为。
- MOPD 在训练期间可以按 `data_source` 选择监督 Student 的 Teacher；TIES-Merging 没有这种数据条件化的训练过程。这里不是说最终 Student 推理时仍需动态路由 Teacher。
- MOPD 提供 token-level dense supervision；但它仍可能遭遇共享参数上的跨域梯度冲突，不能被描述为天然解决灾难性遗忘。
- Student 从 RL 前共同模型初始化，避免直接选择某一个领域 Expert 作为起点而先验偏向该领域，同时保留统一的 tokenizer、词表和 chat template 血缘。

### 3.4 验证门禁

回答按三层组织：

1. **FUNCTIONAL**：混域数据、`data_source` 路由、Teacher scoring、backward、weight sync、checkpoint/recovery 完成闭环；每个 Teacher 路由都有非零样本，失败不能静默串域。
2. **NUMERIC**：token、mask、Teacher/Student logp、scatter/gather 和 normalization 对齐；same-weight 条件下蒸馏信号应接近零；各 rank 对异常 fail-consistent。
3. **EFFICACY**：在相同评测协议下比较 RL 前 Student、各领域 Expert、TIES-Merging、单 Teacher OPD 和多 Teacher MOPD；分别报告领域能力和 General 回归，并使用逐题配对、多个 checkpoint/seed 和置信区间。

Teacher headroom 是本项目的 Go/No-Go 门，而不是普遍不可能性定理：如果对应 Teacher 在目标领域、相同评测协议下没有可测 headroom，且 same-path token-level 信号也没有显示稳定的局部互补能力，就不启动该域的正式蒸馏，也不声明 EFFICACY；先检查 Teacher、数据或评测协议。Teacher 总分不高于 Student 并不严格排除它在局部状态上提供有效监督。训练 loss 下降不能证明模型能力提升。

### 3.5 证据边界

- 只有在本人能映射到亲自负责的 PR/commit、设计文档或实验记录时，才说“我设计并实现了”多 Teacher 路由、score validation、`mopd_pg`、mixed-domain data、trajectory weighting、online drain、recovery 或评测工装；其余降级为“项目/系统支持”，并明确个人参与边界。
- TIES-Merging 初步实验未达到多领域能力同时保留的目标，可以作为项目路线背景；只有本人确认亲自设计或执行了该实验时，才把实验动作写成个人贡献。
- 多个 Teacher 与 Student 是否来自同一 RL 前模型、tokenizer、词表和 chat template，需要用 checkpoint/config 或实验记录核实；未核实前只作为方案要求，不作为已完成事实。
- 双 Teacher MOPD 的正式多域效果未完全闭环时，只能说 FUNCTIONAL、NUMERIC 或 early canary 到哪一层，不能声称已经提升多域能力。
- 不把受数据、环境或协议污染的探索性结果升级为正式 EFFICACY 结论。

## 4. RESUME-10：华为阶段的千卡/万卡交付职责

### 4.1 面试官意图

重点不是让候选人泛讲大规模集群运维，而是确认：

- “千卡/万卡”是项目背景还是本人承担了可验证职责；
- 能否独立完成模型从跑通到性能验收的工程闭环；
- 能否区分个人 ownership、团队协作和底层平台贡献；
- 单机优化扩展到大规模后，是否会重新测量和处理瓶颈迁移。

### 4.2 个人职责边界

本题发生在华为阶段。本人主要负责 X1、TX 客户模型在国产卡上的：

1. 功能适配与模型跑通；
2. 性能数据采集与基线固化；
3. 性能瓶颈分析；
4. 优化方案选择、实施或推动；
5. 同 workload A/B 和精度回归；
6. 持续重复测量与优化，直到达到客户性能验收目标。

本人不是整个万卡集群平台、硬件、网络和所有稳定性问题的总 owner。编译器、融合算子、集合通信、硬件或集群环境的问题由对应团队实现修复时，本人负责提供稳定复现、profiling 证据、模型侧验收条件并完成最终回归。

### 4.3 主回答闭环

1. **固定验收口径**：模型版本、并行配置、global/micro batch、sequence length、precision、卡数、warmup、统计窗口、精度阈值和目标性能。
2. **模型跑通**：完成算子兼容、分布式并行、checkpoint/data 和精度链路适配。
3. **采集证据**：记录 step time、吞吐、MFU/硬件利用、算子时间、collective exposed time、pipeline idle、显存峰值等数据。
4. **定位主瓶颈**：区分并行切分、kernel/小 GEMM、通信暴露、显存与重计算、Host/data 或规模化 straggler。
5. **选择优化措施**：根据证据调整 TP/PP/DP/EP 等并行策略，接入 Grouped MatMul/融合算子，或进行计算通信 overlap；只讲项目实际使用的措施。
6. **验证与迭代**：相同 workload 做 A/B，同时验证 loss/精度和稳定性；重新 profile，观察瓶颈迁移，再进入下一轮，直到性能达标。

X1 约 200B MoE 是代表案例，可交叉引用 `RESUME-01A` 的 `0.16x → 0.95x`、MFU 35% 和 3K 卡训练证据。TX 没有确认可披露的模型与数字时，只作为第二个客户交付背景，不补造指标。

### 4.4 与原回答的取舍

- 删除将主回答写成“软件/固件/节点健康、故障域、SLA、集群巡检与自动恢复”的泛化平台运维叙事。
- 保留大规模意识作为追问：规模放大后 collective、拓扑、straggler、故障概率和性能抖动会被放大，必须重新 profile，不能把单机收益线性外推。
- checkpoint、NCCL hang、节点隔离等内容留给通用 Infra 排障题，除非本人能准备一个直接负责的华为交付案例。

## 5. 文档落地要求

- 保留现有锚点 `resume-09`、`resume-10` 和 P0 数量，不新增题目。
- 两题均包含：问题、面试官意图、60–120 秒精准回答、项目证据或知识边界、高概率追问、危险回答。
- `RESUME-09` 的高概率追问增加：TIES 如何工作、为什么仍可能失败、为什么 Student 从 RL 前模型初始化、Teacher 为什么必须有 headroom。
- `RESUME-10` 的高概率追问增加：你亲自做了什么、如何固定 benchmark、一次完整瓶颈迁移、X1 与 TX 的职责是否相同。
- 私人 Markdown 只保留客户代号，不展开真实客户名。

## 6. 验收标准

- TIES-Merging 拼写和 `Trim、Elect Sign、Merge` 展开准确。
- MOPD 数据流明确为 Student rollout、训练期按 `data_source` 路由、Teacher 对同一 token path scoring，而不是 Teacher 重新生成轨迹；同时明确最终 Student 推理不依赖 Teacher 路由。
- FUNCTIONAL、NUMERIC、EFFICACY 三层证据边界明确，未闭环效果不被夸大。
- Teacher headroom 仅作为本项目 Go/No-Go 门禁，不被表述为“Teacher 总分不高于 Student 就不可能提供任何学习信号”。
- `mopd_pg`、score validation、online drain、recovery、评测工装、TIES 实验和模型初始化血缘等事实，在写成个人贡献前均能映射到本人确认的 PR/commit、设计文档或实验记录；否则降级为团队/系统能力或方案要求。
- RESUME-10 明确个人负责模型适配和性能达标，不暗示本人负责整个万卡集群平台。
- X1/TX 仅使用代号，未写客户真实名称。
- Markdown fence、标题层级和锚点有效；内部链接可解析；`git diff --check` 无 whitespace error。

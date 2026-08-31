# 简历核心追问回答设计

## 1. 目标

在现有 [`private_resume/2026-08-llm-infra-interview-prep.md`](../../../private_resume/2026-08-llm-infra-interview-prep.md) 中补强四道高频问题，使回答既能在 60–120 秒内给出主结论，也能承受 10–20 分钟技术深挖：

1. 最有代表性的性能优化：以华为 X1 约 200B MoE 预训练性能优化为主案例。
2. 项目 ownership：解释概念，并用同一 X1 项目证明个人责任边界。
3. 职业选择：独立解释华为到小鹏、现在看机会的原因。
4. Fully Async RLVR：先说明相比同步链路的系统优势，再说明如何通过配置把优势兑现。

本次只修改私人面试准备 Markdown，不修改或重新生成对外简历、DOCX、PDF，也不改变简历中的客户脱敏口径。

## 2. 设计原则

### 2.1 一问只证明一种能力

- X1 MoE 题证明大模型预训练性能分析和系统优化能力。
- Ownership 题证明高级工程师的责任边界、判断力和推动能力。
- 职业选择题证明动机稳定、职业主线清晰，不混入技术细节展开。
- Fully Async 题证明对同步屏障、流水线供需和 off-policy 代价的理解。

四题可以使用相同项目证据，但不能把四种意图揉成一个冗长回答。

### 2.2 先给因果链，再列优化项

每道技术题遵循：

```text
固定 workload 与指标
  → profile 找到 exposed bottleneck
  → 提出假设和单变量实验
  → 改变系统配置/实现
  → 再 profile，确认瓶颈迁移
  → 性能、精度、稳定性共同验收
```

避免只背开关、只罗列框架名，或把并行策略、融合算子、通信重叠说成彼此独立的“优化清单”。

### 2.3 严格区分项目事实与后验知识

- **项目事实**：X1 约 200B MoE、Megatron/MindSpeed 技术栈、并行策略、Grouped MatMul、融合算子、计算通信重叠、内存优化、精度对齐、相对性能 `0.16x → 0.95x`、MFU 35%、3K 卡训练保障。
- **需要本人补齐的事实**：相对性能分母、模型层数/专家数/top-k、训练精度、TP/PP/DP/EP 具体组合、HCCS/RoCE 拓扑、每项优化的独立收益、实际融合算子名、通信重叠的具体调度方式。
- **2026 NVIDIA 报告补充**：Parallel Folding、DeepEP/HybridEP、merged FWD-BWD + W/D split、FP8/FP4、fine-grained activation offloading、partial/full CUDA Graphs 等只能表述为“今天在 NVIDIA/Megatron Core 上继续优化时会评估的演进方向”，不得表述为当时已经落地。

主要参考：

- [Scalable Training of Mixture-of-Experts Models with Megatron Core](https://arxiv.org/abs/2603.07685)
- [Megatron Core MoE 官方 README](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md)

### 2.4 私密信息边界

- 对外简历继续使用 `X1`、`xxxB`、`x/xx` 等脱敏口径。
- 私人准备文档可以写 `约 200B`、`0.16x → 0.95x`、MFU 35% 和 3K 卡，帮助口述。
- 客户真实名称不写入 Markdown；面试现场是否口述由本人根据保密要求判断。

## 3. 文档结构

在 `RESUME-01` 自我介绍之后增加三道 P0 独立追问，使用字母后缀避免大范围重排现有编号和锚点：

1. `RESUME-01A｜最有代表性的性能优化是什么？`
2. `RESUME-01B｜你在这个项目中的 ownership 是什么？`
3. `RESUME-01C｜为什么从华为到小鹏，现在为什么又看机会？`

删除后文与 `RESUME-01B` 重复的简短 `RESUME-11` ownership 条目，或将其改成指向 P0 完整答案的交叉引用，避免维护两份相互漂移的答案。

现有 `RESUME-10` 只保留千卡交付、长稳保障、故障域判断和团队协同，不再重复 ownership 的完整定义；需要说明个人责任边界时交叉引用 `RESUME-01B`。

现有 `RESUME-02` 保留编号，但修改标题和回答重心：

> `RESUME-02｜Fully Async 相比同步 RLVR 有什么优势？你如何把初始吞吐从 76 优化到 211–255 tokens/s/GPU？`

同步更新开头的 3 天优先级表、问题索引和四张项目证据卡，不改动无关题目。

## 4. RESUME-01A：X1 约 200B MoE 性能优化

### 4.1 面试官意图

验证候选人能否把超大 MoE 性能问题拆成模型结构、并行映射、硬件拓扑、kernel、通信、显存和稳定性问题；同时确认 `0.16x → 0.95x` 是可解释、可复现的工程结果，而不是团队数字。

### 4.2 主回答结构

回答控制在 2–3 分钟，采用“背景与指标 → profile → 四层优化 → 结果与复盘”：

1. **背景和口径**
   - X1 约 200B MoE 预训练模型，从功能打通、精度对齐进入性能优化。
   - 先说明 `0.16x/0.95x` 的分母、固定的 global batch、sequence length、precision、卡数和统计窗口。
2. **瓶颈模型**
   - 用 NVIDIA 报告的 Three Walls 作为后验总结：Memory Wall、Communication Wall、Compute Efficiency Wall。
   - 强调当时是通过 profiler、算子耗时、collective 时间、pipeline bubble、tokens-per-expert 和显存峰值逐层定位，不是事后套概念。
3. **并行与拓扑**
   - 在容量可行的候选中联合选择 TP/PP/DP/EP/sequence parallel。
   - 先解释通用判断：避免把 expert GEMM 切得过碎，并根据实际 HCCS/RoCE 域、collective 频率和消息量放置 TP/EP/PP/DP；不能在项目配置尚未补齐时直接声称“TP/EP 留在 HCCS、PP/DP 跨节点”就是当时方案。
   - 面试前补齐实际并行组合、通信 group 到物理拓扑的映射和关键 collective；只有核对完成后，才能把某个 topology-aware mapping 写入项目主答案。
   - 调整 micro-batch、gradient accumulation、VPP/层划分以降低 pipeline bubble，同时守住显存。
4. **计算与 kernel**
   - 将多个 expert 的小 GEMM 聚合成 Grouped MatMul，提高矩阵规模和硬件利用率。
   - 使能实际使用过的融合算子，重点解释减少中间张量、内存读写和 kernel launch，而不是只报开关名。
   - 结合 token permutation、expert padding/load imbalance 解释为什么 MoE 的 MFU 不能只看理论 FLOPs。
5. **通信与调度**
   - 对 TP/DP collective、EP token dispatch/all-to-all 和 PP P2P 分别 profile。
   - 用独立 stream、chunking 或调度重排将无依赖的通信隐藏在 attention/expert/反向计算后面；回答必须明确当时真实采用的一种 overlap 机制。
   - overlap 后重新检查带宽争用、额外 buffer 和 GEMM 降速，不能用 timeline 重叠直接等价为收益。
6. **内存、精度和稳定性**
   - 使用实际采用的 distributed optimizer、sequence parallel、activation recomputation/重计算粒度或 buffer 复用来降低峰值显存，从而避免被迫使用过高 TP/PP。
   - 对融合算子和低精度路径做逐层 dump/first-divergence，对齐 loss 和梯度。
   - 从小规模功能/精度验证扩到 3K 卡长稳训练，补充 checkpoint、故障隔离和性能回归门禁。
7. **结果和方法论**
   - 相对性能从 `0.16x` 提升到 `0.95x`、MFU 35%，达到上线目标并保障 3K 卡训练。
   - 结论不是“某个融合算子带来 6 倍”，而是通过并行映射、kernel、通信、显存和集群稳定性的多轮瓶颈迁移完成系统优化。

### 4.3 NVIDIA 2026 报告的追问补充

仅在面试官追问“如果今天继续做，还有什么优化”时使用：

- **Parallel Folding**：将 attention 的 TP/CP/DP 与 MoE 的 ETP/EP/EDP 解耦，避免同一并行配置同时伤害 dense attention 和 sparse expert。
- **Dispatcher**：根据硬件拓扑评估 DeepEP/HybridEP 等 optimized dispatcher，减少跨节点冗余搬运并提升 EP 带宽利用率。
- **EP overlap**：用 merged FWD-BWD、独立 compute/comm stream 和 Wgrad/Dgrad split 扩大 all-to-all 隐藏窗口。
- **Memory**：fine-grained recomputation、pipeline-aware activation offloading、precision-aware optimizer，目标是用较低开销换取更小 TP/PP 和更高效 GEMM。
- **Compute**：router/permutation fusion、FP8/FP4 grouped quantization + Grouped GEMM、partial/full CUDA Graphs；特别说明 dropless MoE 动态 expert shape 与 graph 静态 shape 的冲突。
- **方法**：仍按“先内存可行、再选拓扑友好的并行映射、最后 profile 当前瓶颈”的迭代顺序，而不是一次性全开。

## 5. RESUME-01B：Ownership

### 5.1 概念定义

文档首先给出中文解释：

> Ownership 不是“所有代码都是我写的”，而是我对一个明确问题从目标定义、技术方案、关键实现、跨团队推进到上线验收承担端到端责任；遇到风险时我负责暴露问题、组织决策并把结果闭环。

### 5.2 回答框架

使用五个维度回答，避免只说“负责”或反复使用“我们”：

1. **Scope**：我承诺解决的具体问题和成功指标是什么。
2. **Decision**：我亲自做出的关键技术判断是什么，舍弃了哪些方案。
3. **Execution**：我亲自实现、调试或主导实验的模块是什么。
4. **Coordination**：依赖哪些框架、算子、硬件和客户团队，我如何推动问题闭环。
5. **Outcome**：结果如何验收；上线后出了问题，谁负责 first response、回归和复盘。

X1 例子需要明确：本人负责/核心负责从功能打通、精度对齐、性能方案到上线保障；个人关键贡献是并行配置搜索、Grouped MatMul/融合算子接入与验证、通信性能分析和关键实验；框架基础能力、底层算子实现和集群运维中哪些属于其他团队必须如实说明。

### 5.3 高质量边界

- 不把 Megatron/MindSpeed 原生能力说成自研算法。
- 不把 3K 卡项目的全部成果归为个人代码成果。
- 能回答“如果没有你，项目最可能卡在哪里”：缺少统一瓶颈模型、配置选择和跨团队收敛接口，而不是“没人能写代码”。

## 6. RESUME-01C：职业选择

### 6.1 独立成题的原因

这是稳定性、动机和岗位匹配题，与 MoE 技术深挖的评价维度不同。回答控制在 60–90 秒，不主动展开技术实现。

### 6.2 回答结构

1. **先承认每段经历的获得**：华为提供了大模型迁移、昇腾性能/精度优化和千卡交付经验；小鹏提供了 GPU、Megatron-Core、verl、AReaL 和 RL Infra 深入实践。
2. **华为到小鹏的客观因素**：部门整体搬迁上海，而本人的家庭和长期定居规划在深圳。
3. **职业因素**：希望从与单一国产硬件和客户交付强绑定的职责，扩展到通用 GPU 生态、训练后训练和平台型 Infra ownership。
4. **现在看机会的触发因素**：当前部门较大组织调整使方向和岗位边界存在不确定性。
5. **长期诉求**：只考虑深圳，希望在训练、后训练和训练推理 Infra 上长期承担清晰的核心系统责任。

### 6.3 表述边界

- 主动说“家庭和长期定居规划在深圳”，不主动列举结婚、生娃、买房。
- 不主动说只考虑宝安、南山；办公地点和通勤范围留给 HR 确认。
- 不使用“大厂螺丝钉”“自由度低”“部门不稳定”等负面措辞。
- 组织调整只作为触发因素，长期技术方向、岗位责任和深圳稳定发展才是选择标准。
- 不承诺“当前公司不调整就绝不离职”等无法验证的假设。

## 7. RESUME-02：Fully Async RLVR

### 7.1 面试官意图

验证候选人是否理解异步架构为何能提高端到端 GPU 利用率、为什么架构开关本身不产生收益，以及如何同时处理供需失衡、长尾、staleness、weight sync 和正确性。

### 7.2 主回答结构

1. **同步基线**
   - 限定为“本项目同步基线或典型 phased baseline”，不能泛化为所有同步 RLVR 实现都将 rollout、reward/ref、actor update 和 weight sync 完全串行执行。
   - 核心同步边界是：当前 logical batch 的 rollout/reward 数据未组装完成前不能进入对应 update；要用新 policy 继续生成时，需要先完成相应的 weight sync。reward/ref 是否能并行或与其他阶段重叠取决于具体实现。
   - 在本项目 phased baseline 中，Trainer 在主要 rollout 窗口存在等待，Rollouter 在主要 update/sync 窗口存在等待；同一 logical batch 的长 trajectory 会放大 exposed wait。
   - 优点是 policy freshness 和 step 语义清晰，不能把同步说成一无是处。
2. **Fully Async 的系统优势**
   - 将 Rollouter 作为持续 producer、Trainer 作为持续 consumer，通过 queue 解耦生命周期。
   - rollout 与训练在时间上重叠，减少阶段 bubble，并降低长尾 trajectory 对整个同步 step 的阻塞。
   - Trainer/Rollouter 可以独立扩缩容；优化目标变为平衡 producer rate、consumer rate、weight update cadence 和样本新鲜度。
3. **为什么初始 async 只有 76**
   - 同步 profile 显示约 79% 时间在 rollout，但初始使用 `3T+1R`，24 张卡训练、8 张卡 rollout，与实际耗时结构不匹配。
   - `gen-TP=4` 使 8 张 rollout GPU 只能形成 2 个 vLLM 实例，producer rate 低于 trainer consumer rate，trainer idle ratio 为 0.41。
   - 这说明异步只提供 overlap 机会；资源未配平时，queue 仍会空、Trainer 仍会等待。
4. **如何兑现优势**
   - `gen-TP 4 → 2`：相同 8 张 rollout GPU 的实例数从 2 增到 4，提升独立 continuous batching 的并发池，达到 `211–255 tokens/s/GPU`。
   - `3T+1R → 2T+2R`：rollout 使用 16 张 GPU、实例数增至 8，候选窗口达到 `236–293 tokens/s/GPU`，trainer idle ratio 降到 `0.10–0.14`。
   - 瓶颈随后转移到 actor update，说明继续增加 rollout 资源已不是最优方向。
5. **代价与最终选择**
   - queue 太空表示 rollout 供给不足；queue 持续增长表示 rollout 过供给并增加 staleness。
   - 最终配置联合观察端到端 throughput、gen wait、actor/ref/update、parameter sync、queue depth、idle ratio、显存、policy version 和效果回归。
   - Fully async 的收益来自减少 exposed idle，不是让单个 rollout 或 update kernel 自动变快；代价是 bounded off-policy、跨池权重同步和更复杂的恢复语义。

### 7.3 数字口径

- `76 → 211–255`：fully async 初始错误/不匹配配置与调优后配置的比较。
- `同步约 200`：目前只是待核实线索。同步精确值、相同 workload、统计窗口、warmup/异常步处理和 `tokens/s/GPU` 分母全部补齐前，不得进入可口述主答案，也不得声称存在严格可比的 sync/async 提升倍数。
- `76 → 211–255` 和 `236–293`：在口径确认前仅用于 fully async 不同配置之间的比较。
- `236–293`：`2T+2R` 候选窗口，不能自动写成正式长期平均或最终生产配置。
- 禁止说“fully async 相比同步提升 3 倍”。
- 面试前必须确认 `tokens/s/GPU` 的分母是全集群 GPU、rollout GPU，还是框架特定口径。

## 8. 验收标准

### 8.1 内容验收

- 四道题均包含：问题、面试官意图、精准回答、追问、项目证据/知识边界、危险回答。
- 每道题第一段可以独立作为 60–120 秒回答，后续内容可以承接深挖。
- X1 回答以真实项目为主，NVIDIA 2026 技术明确标为后验演进方向。
- Ownership 明确区分个人贡献、开源框架和团队依赖。
- 职业选择不与技术题混写，地点诉求体现长期稳定性。
- Fully Async 先讲 sync/async 机制，再用配置和数据验证，不把 `76` 误当同步基线。
- 同步与 async 的 workload、统计窗口和吞吐分母未完全对齐前，主答案不包含 sync/async 倍数或“超过同步”的结论。

### 8.2 文件验收

- Markdown fence 平衡，标题层级和锚点有效。
- 新增的目录链接可以跳转。
- 相对内部链接和外部主来源链接可解析。
- `git diff --check` 无 whitespace error。
- Git 状态能够区分本次文档修改与此前未提交的流程图修改。

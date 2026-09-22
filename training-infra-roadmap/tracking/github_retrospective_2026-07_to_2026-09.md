# GitHub 历史补扫与复盘：2026 年 7–9 月

- 复盘日期：2026-09-22；性质：**Historical Review**，不是新 frontier scan。
- 冻结事件窗口：2026-07-01 00:00:00 → 2026-09-22 17:06:16，Asia/Shanghai；与已有 GitHub 专项确认边界一致。9 月仍是未完成月份。
- 15 个核心仓库，逐页枚举 **11,114 main-history commits、11,728 merged PR、55 releases**。三种集合有重叠，不能相加当成独立信号数。
- 深入复核 20 个 PR 的正文及变更文件，形成 **14 组 Read 历史补漏、1 组 Observe、1 组已有信号去重**。这不是对一万多个 PR 逐项进行完整代码审计，也没有运行上游 GPU 测试。
- 未修改原扫描 Accepted 数、历史 Decision 或任何扫描游标；不把回移 PR、已知 commit 的 PR 链接重复算新增。

## 覆盖账本

| 月份（上海） | main-history commits | merged PR | releases |
|---|---:|---:|---:|
| 2026-07 | 3,616 | 3,841 | 19 |
| 2026-08 | 4,142 | 4,360 | 19 |
| 2026-09，截至确认边界 | 3,356 | 3,527 | 17 |

| 仓库 | commits | merged PR | releases | 索引边界 |
|---|---:|---:|---:|---|
| [NVIDIA-NeMo/RL](https://github.com/NVIDIA-NeMo/RL) | 393 | 538 | 1 | 完整跨过起点 / 到短页 |
| [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | 783 | 917 | 3 | 完整跨过起点 / 到短页 |
| [OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | 40 | 27 | 3 | 完整跨过起点 / 到短页 |
| [THUDM/slime](https://github.com/THUDM/slime) | 88 | 84 | 2 | 完整跨过起点 / 到短页 |
| [alibaba/ROLL](https://github.com/alibaba/ROLL) | 8 | 8 | 0 | 完整跨过起点 / 到短页 |
| [areal-project/AReaL](https://github.com/areal-project/AReaL) | 132 | 147 | 2 | 完整跨过起点 / 到短页 |
| [huggingface/accelerate](https://github.com/huggingface/accelerate) | 49 | 48 | 1 | 完整跨过起点 / 到短页 |
| [huggingface/kernels](https://github.com/huggingface/kernels) | 95 | 99 | 4 | 完整跨过起点 / 到短页 |
| [huggingface/peft](https://github.com/huggingface/peft) | 175 | 174 | 2 | 完整跨过起点 / 到短页 |
| [huggingface/tokenizers](https://github.com/huggingface/tokenizers) | 152 | 139 | 2 | 完整跨过起点 / 到短页 |
| [huggingface/transformers](https://github.com/huggingface/transformers) | 793 | 795 | 9 | 完整跨过起点 / 到短页 |
| [huggingface/trl](https://github.com/huggingface/trl) | 563 | 568 | 9 | 完整跨过起点 / 到短页 |
| [sgl-project/sglang](https://github.com/sgl-project/sglang) | 4012 | 4292 | 7 | 完整跨过起点 / 到短页 |
| [verl-project/verl](https://github.com/verl-project/verl) | 335 | 378 | 2 | 完整跨过起点 / 到短页 |
| [vllm-project/vllm](https://github.com/vllm-project/vllm) | 3496 | 3514 | 8 | 完整跨过起点 / 到短页 |

认证 REST 每页 100 项。commits 固定 main 并用 since/until 限窗，直到短页；closed PR 按 updated 降序翻页，直到越过起点，再按 merged_at 筛选，因此包含非 main 分支。release 复用同日已读到末页的索引并按 published_at 筛选。未纳入未合并 PR、issue、未合并分支、私有仓库；main-history 指当前 main 可达历史，不保证每个 commit 的提交日期等于首次进入 main 日期。更早的 1–6 月并未用这一方法全量重扫。

可核对：[manifest 与分页数](audits/2026-09-22-retrospective/manifest.json)、[commit 索引](audits/2026-09-22-retrospective/commits.csv)、[merged PR 索引](audits/2026-09-22-retrospective/pulls.csv)、[release 索引](audits/2026-09-22-retrospective/releases.csv)、[20 项人工复核元数据](audits/2026-09-22-retrospective/reviewed_prs.json)。索引中的 previous_url_or_sha_match 是去重线索，不是“此前从未涉及该问题”的证明。

## 历史补漏与重新判断

每组统一：First seen in this retrospective = 2026-09-22；检索窗口如上；Impact = 高（改变正确性、生产运行或成本判断，O1 为中）；Status = NEW（K1 保留原状态）。完整标题、GitHub 作者、合并时刻、branch、SHA 和文件证据见元数据。Decision 不表示用户已读。以下是相对于原记录**新增的具体证据**，不将既有主题再次当作新主题。

<a id="j1"></a>

### J1 · 07 月：成本与统计口径

- Source ID / Type：`github:verl-project/verl:7049,6977`；merged PR。[#7049](https://github.com/verl-project/verl/pull/7049)（2026-07-20T03:33:52Z；main） / [#6977](https://github.com/verl-project/verl/pull/6977)（2026-07-08T09:44:35Z；main）。
- Subsystem：scheduler / training。Decision：**Read**。
- Reason / 一句话看点：独立 rollout GPU 必须进入吞吐分母；separate-async 的一个 step 要覆盖一致的参数同步周期。
- 对 AReaL 的迁移与下一步：统一 AReaL 与对照框架的总 GPU、step、有效 token 和更新次数口径。

<a id="j2"></a>

### J2 · 07 月：训练证据正确性

- Source ID / Type：`github:NVIDIA-NeMo/RL:3297`；merged PR。[#3297](https://github.com/NVIDIA-NeMo/RL/pull/3297)（2026-07-20T22:47:17Z；main）。
- Subsystem：rollout / data path。Decision：**Read**。
- Reason / 一句话看点：用实际 sampled token ID 查 logprob；字典首项可能属于另一个候选 token。
- 对 AReaL 的迁移与下一步：在不同 backend、多候选 logprob 和重排路径上验证 token→logprob 一一对应。

<a id="j3"></a>

### J3 · 07 月：优化轨迹连续性

- Source ID / Type：`github:NVIDIA-NeMo/RL:3171`；merged PR。[#3171](https://github.com/NVIDIA-NeMo/RL/pull/3171)（2026-07-14T04:57:55Z；main）。
- Subsystem：checkpoint/recovery。Decision：**Read**。
- Reason / 一句话看点：PPO 的 actor/critic 要识别 DCP 内嵌 optimizer/scheduler；目录探测失败不应静默重置优化器。
- 对 AReaL 的迁移与下一步：验收恢复后的 Adam moments、scheduler 和 critic warmup 状态，不只比权重。

<a id="j4"></a>

### J4 · 07 月：环境与 trace 所有权

- Source ID / Type：`github:huggingface/trl:6420`；merged PR。[#6420](https://github.com/huggingface/trl/pull/6420)（2026-07-24T07:41:30Z；main）。
- Subsystem：rollout / data/trajectory path。Decision：**Read**。
- Reason / 一句话看点：实验 OpenEnv harness 允许外部 agent 持有循环，框架读取 proxy token/logprob trace，并过滤非 agent 辅助调用。
- 对 AReaL 的迁移与下一步：借鉴 session 隔离、trace provenance、verify 与训练 reward 分离；不声称已规模验证。

<a id="j5"></a>

### J5 · 07 月：recurrent state 生命周期

- Source ID / Type：`github:vllm-project/vllm:49757`；merged PR。[#49757](https://github.com/vllm-project/vllm/pull/49757)（2026-07-29T01:09:41Z；main）。
- Subsystem：inference backend / checkpoint/recovery。Decision：**Read**。
- Reason / 一句话看点：dummy run 也执行真实状态写入；清除旧 block-table 行，避免写坏被重新分配的 recurrent state。
- 对 AReaL 的迁移与下一步：在 AReaL backend 验收中加入 dummy/real 交错与 PD receive；对应 SGLang #30986 的失败撤销。

<a id="j6"></a>

### J6 · 07 月：失败原子性

- Source ID / Type：`github:sgl-project/sglang:30986`；merged PR。[#30986](https://github.com/sgl-project/sglang/pull/30986)（2026-07-23T10:48:32Z；main）。
- Subsystem：inference backend。Decision：**Read**。
- Reason / 一句话看点：load-back 在真正复制前发布 mamba slot，失败后必须撤销引用并回收；否则下一步读未初始化或旧状态。
- 对 AReaL 的迁移与下一步：迁移 prepare→copy→publish / abort 的状态不变量，不照抄 SGLang pool 实现。

<a id="a1"></a>

### A1 · 08 月：样本守恒与统计域

- Source ID / Type：`github:THUDM/slime:2238,2235`；merged PR。[#2238](https://github.com/THUDM/slime/pull/2238)（2026-08-12T06:05:20Z；main） / [#2235](https://github.com/THUDM/slime/pull/2235)（2026-08-12T05:49:26Z；main）。
- Subsystem：rollout / scheduler / training。Decision：**Read**。
- Reason / 一句话看点：按需取完成 group 并保留余量；避免 blocking put 卡事件循环；advantage whitening 统计域包含 CP，空分片也参与 collective。
- 对 AReaL 的迁移与下一步：分别测队列拥塞、无重复/丢失、不同 CP 分片的目标一致性。

<a id="a2"></a>

### A2 · 08 月：消费 frontier

- Source ID / Type：`github:NVIDIA-NeMo/RL:3599,3820`；merged PR。[#3599](https://github.com/NVIDIA-NeMo/RL/pull/3599)（2026-08-25T09:56:00Z；main） / [#3820](https://github.com/NVIDIA-NeMo/RL/pull/3820)（2026-08-25T14:23:55Z；super-v3.5-posttraining）。
- Subsystem：checkpoint/recovery / data path。Decision：**Read**。
- Reason / 一句话看点：保存训练消费 frontier 及覆盖元数据，恢复时补回 in-flight/未消费 prompt；#3820 是 super-v3.5-posttraining 分支回移。
- 对 AReaL 的迁移与下一步：对 AReaL 恢复做长短轨迹分层统计，区分重采样、重复训练和漏样本。两个 PR 只计一组。

<a id="a3"></a>

### A3 · 08 月：坐标映射与失败边界

- Source ID / Type：`github:vllm-project/vllm:50723,53751`；merged PR。[#50723](https://github.com/vllm-project/vllm/pull/50723)（2026-08-22T17:39:59Z；main） / [#53751](https://github.com/vllm-project/vllm/pull/53751)（2026-08-27T10:19:52Z；main）。
- Subsystem：weight sync / inference backend。Decision：**Read**。
- Reason / 一句话看点：native loader 将 checkpoint 坐标映射到 TP/EP/packed layout；稀疏 wire 为 O(nnz)，staging/apply 仍 O(N)，原地更新非事务。
- 对 AReaL 的迁移与下一步：借鉴 backend-owned 映射；先验 shared baseline、loader 支持范围和失败后禁入，再测总 refit。

<a id="a4"></a>

### A4 · 08 月：衍生权重一致性

- Source ID / Type：`github:sgl-project/sglang:35883`；merged PR。[#35883](https://github.com/sgl-project/sglang/pull/35883)（2026-08-30T21:13:35Z；main）。
- Subsystem：weight sync / inference backend。Decision：**Read**。
- Reason / 一句话看点：GLM gate 直接使用 canonical FP32 parameter，避免运行时更新后继续使用旧 FP32 shadow。
- 对 AReaL 的迁移与下一步：恢复/更新验收覆盖派生缓存，不能把 canonical weight checksum 当全链路证明。

<a id="a5"></a>

### A5 · 08 月：logprob 峰值显存

- Source ID / Type：`github:sgl-project/sglang:31958`；merged PR。[#31958](https://github.com/sgl-project/sglang/pull/31958)（2026-08-07T21:01:08Z；main）。
- Subsystem：inference backend / data path。Decision：**Read**。
- Reason / 一句话看点：用逐行 logsumexp normalizer 与 gather/top-k 代替完整 log-softmax 中间量，减少 teacher 路径临时存储。
- 对 AReaL 的迁移与下一步：检查 tie、非有限值、精度和 top-k 分布后测整条 teacher pipeline；上游局部速度不是 E2E。

<a id="s1"></a>

### S1 · 09 月：attention 语义边界

- Source ID / Type：`github:huggingface/accelerate:4177`；merged PR。[#4177](https://github.com/huggingface/accelerate/pull/4177)（2026-08-31T19:32:35Z；main）。
- Subsystem：training。Decision：**Read**。
- Reason / 一句话看点：拒绝不能保持 sliding/chunked mask 的 CP 路径，并防止不交换 recurrent state 的 sequence parallel 配置静默运行。
- 对 AReaL 的迁移与下一步：CP=1 与 CP>1 做受控输入/梯度对照；不是说所有实现都不支持这些模型。UTC 8/31、上海 9/1，归 9 月。

<a id="s2"></a>

### S2 · 09 月：事件循环 liveness

- Source ID / Type：`github:huggingface/trl:7175`；merged PR。[#7175](https://github.com/huggingface/trl/pull/7175)（2026-09-17T14:25:03Z；main）。
- Subsystem：rollout / scheduler。Decision：**Read**。
- Reason / 一句话看点：同步工具移入线程池，异步工具 await；同一 turn 的工具仍顺序执行，跨 rollout 共享对象需线程安全。
- 对 AReaL 的迁移与下一步：AReaL 对照 heartbeat 与工具耗时，保留有状态环境顺序，并测试超时/取消。

<a id="s3"></a>

### S3 · 09 月：低精度表示可复现性

- Source ID / Type：`github:NVIDIA/Megatron-LM:6666`；merged PR。[#6666](https://github.com/NVIDIA/Megatron-LM/pull/6666)（2026-09-10T19:50:16Z；main）。
- Subsystem：checkpoint/recovery / training。Decision：**Read**。
- Reason / 一句话看点：恢复时从 FP32 main params 重建量化权重，修正 MXFP8 codes/scale 的 round-trip parity；作者明确不是已证明的精度修复。
- 对 AReaL 的迁移与下一步：分开验收编码一致、数值一致、GEMM/梯度一致，不用 bit mismatch 直接推断模型质量下降。

<a id="o1"></a>

### O1 · 08 月：条件性梯度缺口

- Source ID / Type：`github:huggingface/trl:6625`；merged PR。[#6625](https://github.com/huggingface/trl/pull/6625)（2026-08-27T14:27:17Z；main）。
- Subsystem：training。Decision：**Observe**。
- Reason / 一句话看点：补齐 chunked entropy backward，但当时内置调用仅在 no_grad 下记录 entropy。
- 对 AReaL 的迁移与下一步：只有使用该输出构造可微 entropy loss 时才需要按此路径检查；不宣称普遍影响已有 GRPO。

<a id="k1"></a>

### K1 · 09 月：已有信号去重

- Source ID / Type：`github:areal-project/AReaL:1616`；merged PR。[#1616](https://github.com/areal-project/AReaL/pull/1616)（2026-09-11T03:18:28Z；main）。
- Subsystem：checkpoint/recovery。Decision：**Read / already tracked**。
- Reason / 一句话看点：不可变 generation 与 LATEST 原子发布已被 9/16 扫描以 commit 记录。
- 对 AReaL 的迁移与下一步：本次补上 PR 关联，不算新漏项，不重复加入队列。

## 发布边界与仍不升级的内容

main 合并不是 release 发布。55 条 release 只证明索引覆盖，不承诺每个版本逐项回归；以上补漏按 PR/commit 使用，不暗示已进入当时安装包。9 月稳定版与 RC 的具体确认沿用[同日专项审计](github_audit_2026-09-22.md)：verl v0.9.1、vLLM v0.30.0 与 tokenizers v1 RC 要分别处理；TRL main 的后续提交不能因博客表述就归到不存在的已核验 tag。

ROLL、OpenRLHF、Transformers、PEFT、Kernels 与 tokenizers 同样完成索引枚举；没有为了凑覆盖数给每库增加 Accepted。普通模型接入、文档、CI、硬件适配和重复 backport 不自动升级。没有用新增条目数衡量这次复盘的质量。

## 对过去判断的修正

1. “小修复不如新系统论文重要”需要改为“是否改变样本、状态或测量语义”：logprob 错配、group 丢弃和 optimizer 恢复都直接改变训练。
2. “传输稀疏就代表更新便宜”需要拆成 wire、staging、apply、rebuild 和 admission；这里只验证机制边界，尚无本地端到端收益结论。
3. “有 checkpoint 就可恢复”需要消费 frontier 和状态发布协议；“有 CP”需要 attention、recurrent state 和 loss 统计域匹配。

月度阅读：[7 月](monthly_signal_2026-07.md)、[8 月](monthly_signal_2026-08.md)、[9 月阶段报告](monthly_signal_2026-09.md)；[总入口](monthly_reviews.md)。判断进入 [Agentic RL 章节](../topics/agentic_rl.md#monthly-retrospective-invariants)与[实验设计](../experiments/rl_state_boundaries.md#retrospective-test-cases)，并未标为 VERIFIED。

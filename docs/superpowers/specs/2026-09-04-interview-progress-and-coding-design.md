# 面试进度附录与 Coding 题单设计

## 目标

在不建立独立复盘体系的前提下，完成三件事：

1. 在面试准备主文档附录维护极简的公司、轮次、时间和状态；
2. 把实际暴露的技术缺口吸收到整体面试题体系，不记录逐场详细复盘；
3. 把手撕代码集中到一个独立文档，主文档只提供稳定入口。

完整信息允许提交公开 GitHub `main`。

## 主文档改动

目标文件：`private_resume/2026-08-llm-infra-interview-prep.md`。

### 进度附录

在全文末尾（现有 Part VI 的 VI.8 及收束原则之后）增加 `<a id="interview-progress"></a>` 与 `## Appendix A｜面试流程进度台账`。标注“更新截至 2026-09-04”，只保留：公司、岗位、面试时间、当前轮次、状态、下一节点。不记录问题、自评、答案或复盘。

初始数据：

| 公司 | 岗位 | 时间 | 当前轮次 | 状态 | 下一节点 |
|---|---|---|---|---|---|
| 灵动时刻 | 训练 Infra | 2026-09-03 下午 | 一面完成 | 待反馈 | 等待结果 |
| 智元机器人 | 训练 Infra | 2026-09-04 下午 | 一面完成 | 已通过 | 2026-09-08 下午二面 |
| 字节跳动 | 训练 Infra | 2026-09-08 晚上 | 一面待进行 | 已排期 | 完成一面 |
| 小红书中台 | 训练 Infra | 待定 | 待约面 | 时间未定 | 确认面试时间 |

顶部现场救急增加 `[面试进度](#interview-progress)` 与 `[Coding 手撕题](2026-09-interview-coding.md)`；Part V 导航附近再提供一次 Coding 入口。附录回链到总控制台。

### 整体题库补强

- 扩展 `MEGATRON-01`：增加 Megatron-Core 的框架级 SPMD、parallel state/process group、组件编排和一行端到端数据流；不新增题目。schedule、distributed optimizer、checkpoint 的机制、公式和排障分别只在 `MEGATRON-05`、`MEGATRON-07`、`MEGATRON-10` 维护，此处只回链。
- 扩展 `MEGATRON-02`：用 shape 说明 MLP 的 gate/up Column Parallel + down Row Parallel，以及 Attention 的 QKV Column Parallel + output Row Parallel；区分 TP head/hidden 切分与 CP sequence/KV 交换；不新增题目。
- 扩展 `DIST-01`：补充 FSDP/FSDP2 在 forward/backward/optimizer 生命周期中的 parameter AllGather、gradient ReduceScatter、reshard/prefetch；不新增题目。raw TP/PP shard checksum 的排障正文只在 `TRAIN-ANOMALY-01` 维护，此处仅回链。
- 新增 `ROLLOUT-01`（Part III，P0）：统一回答 rollout 在模型执行、KV/cache、请求调度、训推协同、异步长尾、正确性与 goodput 六层的优化，并绑定已有项目证据。
- Verl/AReaL 选型、通信算子、CUDA Graph、Prefix Cache 继续复用现有题，不复制答案。

今天暴露的九个技术缺口采用唯一归属，避免同一答案在多处漂移：

| 缺口 | 唯一主答案 | 维护边界 |
|---|---|---|
| verl 与 AReaL 选型 | `AREAL-01` | 只维护选型判断 |
| Megatron 与 FSDP 选型 | `MEGATRON-11` | 不新增重复题 |
| FSDP 原理及通信 | `DIST-01` | 只讲 FSDP 生命周期；通用 collective 语义回链 `INFRA-04` |
| 通信算子语义和原理 | `INFRA-04` | collective 唯一详解 |
| Rollout 优化全景 | `ROLLOUT-01` | 只做分层诊断/决策树；细节回链 `RESUME-02/03/13/14/19`、`VERL-02/04/05/09` |
| CUDA Graph | `RESUME-13` | 不在 Rollout 总题重复原理 |
| Prefix Cache | `RESUME-14` | 不在 Rollout 总题重复原理 |
| Megatron 框架原理 | `MEGATRON-01` | 框架级数据流唯一主答案 |
| TP 切分细节 | `MEGATRON-02` | MLP/Attention shape 与通信唯一主答案 |

`DIST-01` 不重复 raw shard checksum 排障，相关边界回链 `TRAIN-ANOMALY-01`。`ROLLOUT-01` 的每一层标注证据等级：本人项目直接证据、联合配置结果但无单因素 A/B、机制理解、今天会评估。

主文档题量：78 → 79；P0 47 → 48；Part III 16 → 17。其他优先级与 Part 数量不变，Core 10 不变。

## Coding 文档

新建 `private_resume/2026-09-interview-coding.md`，顶部回链主文档控制台，内容仅包含：

1. PyTorch 手写 Multi-Head Self-Attention：输入/输出统一为 `[B,S,D]`；shape 推导、`d_model % num_heads`、QKV projection、split/merge heads、scale、causal/padding mask、FP32 softmax、output projection、复杂度与易错点；`key_padding_mask` 为 `[B,S]` 且 `True` 表示屏蔽，明确 causal 与 padding mask 的 broadcast/合并。实现禁止直接调用 `nn.MultiheadAttention`，测试可以使用独立的 `nn.MultiheadAttention` 或 SDPA/reference 实现对照。
2. `N×N` 矩阵原地顺时针旋转 90°：主对角线原地 transpose，再对每行调用 `row.reverse()`；`O(N²)` 时间、`O(1)` 额外空间。修改矩阵前验证 square/non-ragged，非方阵抛 `ValueError`；不用 NumPy、切片拷贝或额外矩阵。`3×3` 的期望结果固定为 `[[7,4,1],[8,5,2],[9,6,3]]`。

Coding 题不计入主文档 79 道知识题统计，避免跨文件计数混淆。

## 验收

- 主文档 79 题；P0/P1/P2=`48/26/5`；Part I–V=`7/27/17/18/10`；79 条逐题正确双回链。同步更新顶部总览、Part III 摘要/局部导航、VI.8 优先级表与结尾题量。
- 新增 `rollout-01`、`interview-progress` 锚点唯一；`ROLLOUT-01` 同时出现在顶部“推理 / Rollout”技术行、全量索引、Part III 局部导航和正文。全部变更文档的 same-file/local-file/fragment 链接与图片可解析，Coding 文档可回到主文档 `#interview-console`；Core 10 不变。
- 进度附录只有约定的六列和四行，不出现面试问题或详细复盘。
- Coding 文档两段代码可直接运行。使用 `/Users/zengbw/miniconda3/bin/python`（PyTorch 2.12）验证：MHA 固定随机种子，与独立 reference 做无 mask、causal、padding、causal+padding 数值对照；覆盖 `d_model` 不可整除异常，以及输入/参数梯度 finite；矩阵旋转覆盖 0×0、1×1、3×3 精确结果和 ragged/非方阵异常。验证器从 Markdown code fence 抽取代码运行，避免文档与测试代码漂移。
- 项目边界不变：不把机制理解包装成底层框架/kernel ownership；Rollout 未做过的优化使用“通用方法/今天会评估”。
- 相对发布基线只允许新增本 spec、实施 plan、Coding 文档并修改主文档。
- 主工作树现有 dirty status、binary diff hash 和目标主文档 hash 前后保持不变；不得在 dirty 主工作树执行 switch/merge/rebase/cherry-pick/commit/push，所有实施与发布只在隔离 worktree 完成。
- 发布前 fetch，确认 `origin/main` 是待发布 HEAD 的祖先并核对 changed-files allowlist；只将隔离 worktree 的 HEAD fast-forward 推送到远端 `main`。发布后 `git ls-remote --heads origin` 只能有 `refs/heads/main`，且其 SHA 等于验收 HEAD。

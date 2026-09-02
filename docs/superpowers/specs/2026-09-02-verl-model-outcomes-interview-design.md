# 自研版 verl 真实模型落地题设计

## 1. 目标

在已确认的面试手册结构中新增一道独立 P1 题，用两项真实模型工作说明自研版 verl 的生产落地价值：

1. LLM 路线：[Athena-Brain Technical Report](https://arxiv.org/abs/2607.18985)；
2. MLLM/VLM 路线：[Capek 0.5](https://arxiv.org/abs/2608.06756)。

本题不展开论文公式、完整实验指标或算法创新史，只要求候选人能看图讲清两条后训练链路、它们的共同系统抽象和关键差异，并准确界定个人 ownership：

> 我参与自研版 verl 的框架建设；组内算法同学基于该框架完成训练 recipe、模型实验与论文产出。我的重点是解释框架如何稳定承载这些后训练链路，不把算法创新或论文贡献归到个人名下。

## 2. 已核验来源

### 2.1 Athena-Brain

- arXiv：`2607.18985v2`，2026-07-25；
- 标题：*Athena-Brain Technical Report: An Efficient Robot Brain for General Intelligence and Embodied Interaction*；
- 图：Figure 3，论文 PDF 第 4 页；
- 模型定位：8B LLM；
- 图中主线：`Athena SFT anchor → same-origin domain-specialized RL experts → TIES merge → low-dose linear interpolation with a selected alternative-lineage checkpoint → Athena Brain`。

### 2.2 Capek 0.5

- arXiv：`2608.06756v1`，2026-08-07；
- 标题：*Capek 0.5: An Execution-Centric Vision-Language Model for Embodied Intelligence*；
- 图：Figure 6，论文 PDF 第 9 页；
- 模型定位：2B dense 与 35B-A3B MoE 两条 VLM track；
- 图中主线：`shared VLM → four capability specialists → TIES student initialization → routed MOPD on student-generated prefixes → one inference checkpoint`。

用户上传图片的对话顺序与论文顺序相反：第一张是 Capek Figure 6，第二张是 Athena Figure 3。实施时必须按论文名和图号重新命名，不能沿用“图片 1/图片 2”的临时顺序。

## 3. 修改范围

### 3.1 主文档

- `private_resume/2026-08-llm-infra-interview-prep.md`

### 3.2 图片资产

- `private_resume/assets/papers/athena-brain-post-training-figure-3.png`
- `private_resume/assets/papers/capek-0.5-specialization-consolidation-figure-6.png`

两张图片直接取自用户本轮上传的 PNG，保持原始像素和宽高比，不二次压缩、不裁掉原始 Figure caption。主文档在图片下明确标注论文、Figure 编号和 arXiv 链接，不把论文原图描述成候选人个人绘制。

### 3.3 可选专题回链

- `training-infra-roadmap/topics/rl_framework_selection.md`
- `training-infra-roadmap/topics/mopd.md`

专题只在已有相关段落补充一条案例回链：框架选型页承接“自研版 verl 的真实 workload”，MOPD 页承接 Capek 的 `TIES + routed MOPD` 案例。不得复制主文档的两张大图，也不为本轮新建 paper note、tracking 条目或额外专题。

## 4. 新题位置、编号与计数

### 4.1 题目

新增：

> **VERL-11｜自研版 verl 支撑了哪些真实后训练工作？请结合 Athena-Brain 与 Capek 0.5 说明。（P1，15 分钟）**

归入 Part III｜RL 算法、verl 与 Fully Async RLVR 的 P1 区域，放在 `VERL-10` 之后。顶部全量索引、Part III 局部导航和 Part III 追问路线都加入 `VERL-11`。

### 4.2 最终计数

在上一份已确认规格的 68 题基础上新增一道 P1，最终规范性计数是：

| Part | 唯一问题数 |
|---|---:|
| Part I | 7 |
| Part II | 22 |
| Part III | 15 |
| Part IV | 17 |
| Part V | 8 |
| 合计 | 69 |

最终优先级分布：

- P0：38；
- P1：26；
- P2：5；
- Core 10 不变，仍是 P0 的子集。

三天冲刺表不把 `VERL-11` 提升到 Core 或必背 P0；Part/题量总览、全量索引和 P1 时间预算需要同步到 `69 / 38 / 26 / 5`。

## 5. 主文档版式

`VERL-11` 使用用户已确认的上下堆叠版式，而不是把两张宽图左右压缩：

1. 题目、面试官意图；
2. ownership 边界提示；
3. Athena-Brain 小节：一句话链路 → Figure 3 全宽图 → 四步解释；
4. Capek 0.5 小节：一句话链路 → Figure 6 全宽图 → 四步解释；
5. 共同系统主线与关键差异；
6. 项目边界、高概率追问、危险回答和论文链接。

图片使用相对 Markdown 路径，并保留足够空行，确保 GitHub/本地 Markdown 页面以原宽高比渲染。图片 `alt` 必须包含论文名、Figure 编号和链路用途，不能只写“图 1”。

## 6. 回答内容契约

### 6.1 先说结论

允许口述的总括是：

> 我参与建设的自研版 verl 不只跑过单一算法 demo，而是支撑了 LLM 和 MLLM 两类真实后训练工作。两条路线都体现了“先把不同能力独立做强，再汇聚成一个部署模型”，但 Athena 主要在 parameter space 做分层 model merge，Capek 则先用 TIES 初始化，再通过 routed MOPD 在 Student 自己访问的状态分布上做 policy-space consolidation。算法 recipe 和论文由组内算法同学负责；我的贡献边界是框架建设及其训练、rollout、任务与后端承载能力。

不能把“基于我参与建设的框架产出论文”说成“我提出了论文算法”，也不能把所有图中阶段都说成在 verl 内完成。尤其 Athena 的最终 parameter-space merge 是后训练 pipeline 的一部分，但不自动属于 RL trainer 的执行路径。

### 6.2 Athena-Brain：LLM 路线

主文档讲清四步即可：

1. **建立统一 anchor**：从 open-weight base 做 General SFT，形成 Athena SFT，作为后续能力分叉和 task vector 的共同参考；
2. **领域专门化**：从同一 Athena SFT 起点分别训练 Agent、Science、Instruction Following、Code、Embodied 等 domain-specialized RL experts；
3. **同源能力汇聚**：共享起点的 RL experts 以该 anchor 定义 task vectors，通过 TIES 合成 multi-domain trunk；
4. **异源能力补充**：另一条训练 lineage 的候选 checkpoint 不进入同一 TIES voting pool，而是以低权重线性插值注入，形成最终单一 Athena-Brain checkpoint。

可补一句：General RL 使用 GRPO，结合 correctness reward 和 token-budget reward 兼顾推理质量与简洁性。但这不是本题重点，不展开预算、数据量或评测数字。

必须避免：

- 把 Athena 说成使用 MOPD；
- 把 TIES 简化成无条件平均所有 checkpoint；
- 忽略 same-origin expert 与 alternative-lineage checkpoint 使用不同 merge operator 的原因；
- 声称 framework 本身“自动产出”了最终算法和模型结果。

### 6.3 Capek 0.5：MLLM/VLM 路线

主文档讲清四步即可：

1. **同源能力专门化**：从共享 Base VLM 独立训练 Spatial Reasoning、Temporal Understanding、Action Guidance、State Verification 四个 specialist；
2. **异构任务统一承载**：各能力有不同数据、输出格式、parser/verifier 和 reward，但都以 autoregressive text generation 接口和 GRPO 路径训练，checkpoint 保持 parameter-compatible；
3. **两阶段能力汇聚**：先把四个 expert 的 task vectors 用 TIES 合成 Student 初始化，再让 Student 自己 rollout；每个样本按 capability route 选择对应的冻结 Teacher，在 Student-generated prefixes 上提供 token-level 蒸馏信号；
4. **单模型部署**：Teacher 与 routing 只在 consolidation 训练期存在，推理只保留一个 autoregressive checkpoint。

可用一句话解释为什么不是只做 TIES：parameter merge 能提供较好的统一初始化，但不保证每个 specialist 的行为都被保留；routed MOPD 继续在 Student 自己访问的前缀上补 policy-space transfer。

本题不展开 reverse-KL 公式、Top-k approximation、TIES trimming 比例、全部 benchmark 和具体 gain；若面试官追问，再跳转到 `RESUME-09` 与 MOPD 专题。

### 6.4 两条路线的统一系统视角

| 维度 | Athena-Brain | Capek 0.5 |
|---|---|---|
| 模型类型 | 8B LLM | 2B dense / 35B-A3B MoE VLM |
| 能力生产 | 同一 SFT anchor 分叉多个 domain RL experts | 同一 VLM 分叉四个 capability specialists |
| 能力汇聚 | TIES + alternative-lineage low-dose interpolation | TIES initialization + routed MOPD |
| consolidation 空间 | 主要是 parameter space | parameter space 后继续进入 policy space |
| 推理形态 | 一个 checkpoint | 一个 checkpoint |

回答最终收束到自研框架需要提供的稳定能力，而不是复述论文摘要：

- heterogeneous dataset/task/reward 的统一 schema；
- SFT/GRPO 与多 expert recipe 的可复用执行路径；
- rollout、verifier/reward、训练 backend 和 checkpoint lineage 的正确关联；
- 多模态场景下 image/video metadata、mask、position 与 rollout sample identity 不错位；
- 训练结果可复现、可恢复，并能把多个独立 experiment 稳定交付给算法团队。

上面是可迁移的框架能力清单。只有能由个人代码、配置、日志或故障案例证明的具体项，才使用“我负责”；其余使用“框架/团队支持”。

## 7. 面试官意图与追问

### 7.1 面试官意图

1. 你参与建设的自研框架是否被真实模型工作使用，而不只是跑通 benchmark；
2. 你能否从算法图反推训练系统需要承载的数据流、任务抽象和 checkpoint lineage；
3. 你是否能区分 parameter-space merge、policy-space distillation 和 RL expert production；
4. 你能否诚实划分 Infra ownership 与算法/论文 ownership。

### 7.2 高概率追问

- Athena 为什么同源 RL experts 用 TIES，异源 checkpoint 改用低剂量线性插值？
- Capek 为什么先 TIES 再 MOPD，直接从 Base VLM 做 MOPD 行不行？
- 多模态 GRPO 相比文本 GRPO，数据 contract 和 rollout correctness 多了什么？
- 自研版 verl 为这些 heterogeneous tasks 具体补了哪些模块？请落到本人代码路径、配置和生产问题。
- 两个项目哪些阶段运行在 verl，哪些是训练完成后的离线 merge/evaluation？

最后两问若缺少本地代码或项目底稿证据，只保留为面试前待补卡片，不虚构答案。

## 8. 图片、来源与版权边界

- 两张图片由用户明确要求保存到主文档，实施时保留原始截图，不从 PDF 再批量复制其他图；
- Athena HTML 页面标注 CC BY 4.0；Capek arXiv 页面标注 arXiv perpetual non-exclusive license。无论许可证差异，主文档均按论文原图处理：注明论文标题、Figure 编号和 arXiv 原文链接；
- 不擦除用户截图中的现有 watermark，不添加“本人绘制”等归属；
- 图片只服务于面试讲解，不把论文全文或大量图表复制进仓库。

## 9. 验收标准

1. 主文档新增且只新增一道唯一问题 `VERL-11`，anchor 为 `#verl-11`；
2. 题量严格为 Part `7 / 22 / 15 / 17 / 8`，优先级 `38 / 26 / 5`，总计 69；
3. 顶部全量索引、Part III 导航、Part III P1 和追问路线均可跳转到 `VERL-11`；
4. Athena Figure 3 与 Capek Figure 6 没有配反，图片文件名、alt、caption 和论文链接一致；
5. 两张 PNG 保持原始尺寸，可被图片库正常解码，Markdown 相对路径可解析；
6. Athena 回答正向出现 `SFT anchor → domain RL experts → TIES → low-dose interpolation`，且不出现 Athena 使用 MOPD 的错误；
7. Capek 回答正向出现 `four specialists → TIES init → Student rollout → routed frozen Teacher → one checkpoint`；
8. ownership 明确写成“参与自研版 verl 框架建设，算法同学完成算法与论文”，不暗示本人是论文作者或算法提出者；
9. 不在无代码证据时虚构自研框架的具体新增模块，也不把离线 model merge 自动归入 verl trainer；
10. 主文档不堆论文公式、完整实验表和无关 benchmark；
11. `git diff --check`、Markdown 链接检查和 PNG 解码检查通过，Git diff 不包含下载的临时 PDF、渲染页或 `.superpowers` 视觉草稿。

## 10. 提交

本规格增补与上一份已确认的 68 题导航规格共同实施。实现提交仍保持聚焦，建议提交信息：

```text
Expand interview handbook with verl model outcomes
```

推送沿用当前 `codex/personal-infra-map` 分支和仓库既有安全同步流程。

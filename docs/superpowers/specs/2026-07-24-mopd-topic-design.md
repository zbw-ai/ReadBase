# MOPD 原理专题设计

## 背景

ReadBase 已有 [Agentic RL Infrastructure](../../../training-infra-roadmap/topics/agentic_rl.md) 作为 RL Infra 总入口，但尚未系统解释 On-Policy Distillation（OPD）以及它如何演进为 Multi-Teacher On-Policy Distillation（MOPD）。

MOPD 同时涉及 post-training 算法与训练系统：Student rollout、Teacher prefill、token-level supervision、domain routing、异步服务和多 Teacher 资源组织。如果直接从框架实现切入，容易记住配置却没有建立算法因果链。

## 目标

创建单篇专题：

```text
training-infra-roadmap/topics/mopd.md
```

第一版只建立从传统蒸馏到 OPD、再到 MOPD 的原理主线，使读者能够回答：

1. fixed-corpus logit KD、Teacher-generated trajectory distillation 与 OPD 有什么区别？
2. 为什么训练时使用固定或 Teacher prefix、推理时使用 Student prefix 会产生 state-distribution mismatch？
3. 为什么 Student 必须在自己的 rollout 上接受 Teacher 监督？
4. Teacher 的 token-level log-prob 如何转化为 Student 的更新信号？
5. MOPD 为什么采用 per-prompt routing，而不是对多个 Teacher 做 logits ensemble？
6. MOPD 相比 Mix-RL、Cascade RL、Off-Policy Finetune 和 Param-Merge 改变了什么？
7. Same-origin Teacher 为什么影响训练稳定性？

## 非目标

第一版暂不承担以下任务：

- 不做 verl、NeMo RL、AReaL、slime、ROLL 的源码逐行分析。
- 不提供生产配置、GPU 数量和性能调优参数。
- 不建立 MOPD 实验或 AReaL 原型。
- 不创建独立 OPD 专题；内容明显膨胀后再拆分。
- 不写成 MOPD 独立论文的章节翻译或完整论文摘要。

## 内容结构

`topics/mopd.md` 按以下顺序组织：

1. **专题定位与问题**：说明第一版状态为 `READING`，并提出多领域 RL Teacher 难以合并成统一模型的矛盾。
2. **从传统蒸馏到 OPD**：区分 fixed-corpus logit KD、Teacher-generated trajectory distillation 和 Student-rollout OPD；解释 prefix 来源变化导致的 state-distribution mismatch。
3. **OPD 的最小闭环**：Student rollout → Teacher 在相同 Student prefix 上 prefill/scoring → token-level log-prob → Student update。
4. **Reverse KL 与 token-level advantage**：只保留理解 sampled-token policy-gradient 所需的公式和直觉。
5. **从 OPD 到 MOPD**：Domain Teacher、由 prompt/domain metadata 决定的 routing、单样本单 Teacher；明确不是 learned router，也不是 logits ensemble。
6. **MOPD 论文 recipe**：General SFT → Domain-specialized RL → MOPD integration，并用紧凑表格比较 Mix-RL、Cascade RL、Off-Policy Finetune、Param-Merge 和 MOPD。
7. **为什么有效，以及边界在哪里**：on-policy、dense supervision、same-origin 与模块化领域训练；同时说明 Teacher 冲突、错误路由、控制 token 放大和资源增长。Same-origin 的论文事实是共同 SFT 起点带来更小的初始 policy gap/KL 和更稳定的优化；“token advantage 极值和梯度方差可能更可控”只作为由较小初始 KL 推出的机制解释，必须标注为推断。
8. **实现变体侧栏**：简要比较 sampled-token policy-gradient 与 Top-k distillation 的信号、方差和载荷，不展开推导。
9. **与 RL Infra 的关系**：区分算法要求与实现选择，简述 Teacher prefill service、异步 overlap、trajectory metadata 和 freshness。
10. **下一步与相关材料**：合并开源实现、AReaL 迁移、实验问题、一手来源和 300 字以内阶段总结。

## 核心叙事

专题不从“MOPD 有三个阶段”开始，而从一个工程矛盾开始：

```text
每个领域都能训练出更强的 RL Teacher
                    ↓
为什么一个统一模型仍然难以同时继承所有能力？
```

随后依次建立：

```text
固定语料 / Teacher prefix 上的蒸馏
    ↓ 训练 prefix 与推理时 Student prefix 不一致
Student 自己 rollout
    ↓ Teacher 在相同 Student prefix 上评分 sampled token
OPD
    ↓ 不同领域需要不同 Teacher
MOPD
```

每一节必须回答“上一种方案为什么还不够”，避免只罗列定义。

## 三层边界

正文必须显式区分以下三层，避免把论文工程实现误认为算法必要条件：

1. **算法核心**：Student rollout、prompt-selected Teacher scoring、distillation update。
2. **论文完整 recipe**：General SFT → same-origin domain RL Teachers → MOPD integration。
3. **框架实现选择**：Teacher server、async overlap、resource pool、policy-version 与 staleness 管理。

`On-policy` 指训练样本来自 Student rollout 分布，不等于异步系统消费样本时绝对没有 policy lag。NeMo RL 的 async GRPO、ICE-POP 等属于框架层的异步修正，不应倒推为 MOPD 算法定义。

## 可选图示

图示不阻塞第一版正文。若正文评审确认确有必要，再增加一张浅色、科研论文风格 SVG，放在 OPD 到 MOPD 的过渡位置。图中包含两个层次：

1. 下半部分：Student rollout、Teacher prefill、token signal、Student update 的 OPD 闭环。
2. 上半部分：Math / Code / Agent Teacher 通过 domain routing 接入同一个闭环。

约束：

- 自左向右主流程，避免折线和箭头交叉。
- 只使用英文图例，正文提供中文解读。
- 不使用深色大背景。
- 不放大圆圈或装饰性加号。
- 不复制论文原图，重新设计数据流。
- 图中文字不得重叠，次要注释放在下方说明区。

## 来源纪律

事实与公式优先使用以下一手来源：

- MiMo-V2-Flash Technical Report，arXiv:2601.02780。
- MOPD: Multi-Teacher On-Policy Distillation for Capability Integration in LLM Post-Training，arXiv:2606.30406。
- verl Multi-Teacher OPD PR #6051 及合并后的示例。
- NVIDIA NeMo RL MOPD 官方文档。

必须明确区分：

- 2026-01-06：MOPD 在 MiMo-V2-Flash 技术报告中首次公开。
- 2026-04-20：verl Multi-Teacher OPD 实现合并。
- 2026-06-29：MOPD 独立论文发布。

独立论文新增的对比实验、Top-k distillation 实现及其 bias correction、same-origin 稳定性和 multi-round evolution，不应倒推成 verl 早期实现已经完整复现。

## 导航接入

第一版修改以下入口：

- `training-infra-roadmap/topics/agentic_rl.md`：增加双向关系。
- `training-infra-roadmap/MASTER_READING_LIST.md`：增加 OPD → MOPD 阅读路线。
- `training-infra-roadmap/KNOWLEDGE_GRAPH.md`：增加文字关系索引，不扩大主图。

专题状态转为 `DIGESTED` 后，再评估是否加入 `training-infra-roadmap/README.md` 的工程手册入口。暂不修改 `papers.csv`，因为本轮不创建独立 paper note。

## 验收标准

- 能沿一个 prompt 写出 Student、Teacher、Trainer 各自的输入和输出。
- 能指出 sampled token 的 Student/Teacher log-prob 如何形成 token-level advantage。
- 能区分 fixed-corpus logit KD、Teacher-generated trajectory、Student trajectory 和 Teacher prefill scoring。
- 能明确指出一个样本由 domain metadata 路由到一个 Teacher，不是 learned router 或 Teacher ensemble。
- 对照表完整覆盖 Mix-RL、Cascade RL、Off-Policy Finetune、Param-Merge 和 MOPD，并比较监督密度、数据分布、串行耦合与融合空间。
- Same-origin、Top-k 和 multi-round evolution 均标注其来源阶段，不倒推成早期框架已完整实现。
- 从论文事实延伸出的机制解释明确标注为推断。
- 所有时间、标题、关键结论与一手来源一致。
- 所有新增相对链接有效。
- 若生成 SVG，则 XML 可解析，渲染后无文字或箭头重叠。
- 实施前记录 `git status --short` 基线；实施后同时检查 `git diff --name-only` 与 `git ls-files --others --exclude-standard`，确认相对基线只新增专题、可选资产、设计说明和已批准的导航文件，不触碰当前工作区中与本专题无关的修改。

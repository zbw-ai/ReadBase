# 大模型训练推理 Infra 高级工程师：面试题库与现场速查

> - 适用对象：社招大模型训练/推理 Infra 高级工程师
> - 目标档位：当前年薪约 80 万，目标 100–150 万
> - 使用方式：按简历查题；面试前按薄弱项复习，现场先读「直接回答」
> - 修订日期：2026-09-08；官方资料的核验日期与项目版本见文末
> - 依据：最新投递版 PDF 简历（2026-08-30，本地核验且不在公开仓库记录含手机号文件名）、[项目事实底稿](2026-08-xpeng-infra-resume-materials.md)及文末官方资料

<a id="interview-console"></a>
## 0. 考场速查

### 0.1 面试现场速查控制台

> **怎么用**：沿「教育背景 → 工作技能 → 项目经历」找到对应题目，先讲直接回答，被追问时再看展开。题尾可返回本 Part 或本控制台；浏览器返回按钮、macOS `⌘ + [`、Windows/Linux `Alt + ←` 可回到上一次跳转位置。题头的分钟数是完整准备时间，答案里的秒数是口述参考时长。

**快速入口**：[自我介绍](#resume-01) · [框架选型](#areal-01) · [Coding 手撕题](2026-09-interview-coding.md) · [Meshy 笔试专项](2026-09-meshy-ml-system-written-prep.md#meshy-top) · [技术面反问](#vi-questions-to-ask) · [面试前复习](#vi-0) · [面试进度](#interview-progress)

| 简历区块 | 简历内容 / 面试切入点 | 高频题目入口 |
|---|---|---|
| **教育背景** | 厦门大学本科、清华大学硕士，研究方向为人工智能 | [自我介绍](#resume-01) |
| **工作技能** | **Megatron / 分布式训练**：5D 并行、TP/SP/CP、Distributed Optimizer、FSDP/DeepSpeed/Accelerate | **[整体优化方案](#megatron-optimization-overview)** · [5D 并行](#megatron-01) · [TP 切分](#megatron-02) · [SP/CP](#megatron-04) · [Distributed Optimizer](#megatron-05) · [FSDP](#dist-01) · [框架选型](#megatron-11) |
|  | **MoE / 长上下文 / 显存性能**：EP、Grouped GEMM、融合算子、显存账本 | [Dense/MoE](#moe-01) · [EP/A2A](#megatron-06) · [显存账本](#infra-02) · [融合算子](#kernel-01) |
|  | **RL / verl / AReaL**：PPO/GRPO/DAPO、Fully Async、Agentic RL | [RL 算法](#rl-algo-01) · [verl/AReaL 选型](#areal-01) · [HybridFlow](#verl-01) · [资源部署](#verl-02) · [Async/Streaming/Staleness](#verl-04) |
|  | **Rollout / 通信 / 稳定性**：vLLM/SGLang、CUDA Graph、Prefix Cache、Collective、异常排障 | [Rollout 优化](#rollout-01) · [后端选型](#verl-09) · [CUDA Graph](#resume-13) · [Prefix Cache](#resume-14) · [通信算子](#infra-04) · [万卡问题](#infra-09) · [训练异常](#train-anomaly-01) |
| **项目经历（核心）** | **X1 200B MoE**：**`0.16x→0.95x / MFU 35% / 3K 卡连续稳定训练两个月`** | **[代表性优化](#resume-01a) · [Ownership](#resume-01b) · [5D 并行](#megatron-01) · [Dense/MoE](#moe-01) · [规模交付](#resume-10)** |
|  | **Long Context SFT**：**`31s→9.3s；MFU 23%→45.2%`**（独立简历口径，不据此互相反推）；**`128K / 7.6GB`** | **[9B SFT](#resume-05) · [35B-A3B/128K](#resume-17) · [长上下文显存](#resume-06) · [CP-local logits](#resume-07)** |
|  | **Fully Async RLVR**：async 内部配置优化 **`76→211–255 tokens/s/GPU`** | **[同步与异步](#resume-02) · [gen-TP/实例数](#resume-03) · [Rollout 优化](#rollout-01) · [资源部署](#verl-02) · [Async/Staleness](#verl-04) · [正确性](#verl-05)** |
|  | **AReaL Agentic RL / Gateway**：**decode `6–8x`；Rollout `+60%`；Rejected Group `33.18%→2.73%`** | **[训练链路](#resume-08) · [CUDA Graph](#resume-13) · [Gateway 收益](#resume-19) · [Gateway Ownership](#areal-09) · [XCCL/Disk](#areal-11)** |
|  | **OPD / MOPD**：**双 Teacher 在 SWE、Terminal 双域提升且 General 不下降（方向性结论）** | **[MOPD 主问题](#resume-09) · [Trajectory→Gradient](#areal-04) · [三层正确性门禁](#areal-08)** |
|  | **TX 文生视频 / 国产卡规模交付**：**模型跑通、精度、性能、扩容与交付闭环** | **[HunyuanVideo/Ulysses](#resume-18) · [千卡/万卡交付](#resume-10) · [精度对齐](#resume-12) · [融合算子](#kernel-01) · [万卡规模效应](#infra-09)** |

---

### 0.2 一张图看懂我的能力主线

![大模型训练推理 Infra 个人能力地图：六个能力域与脱敏项目证据](assets/llm-infra-personal-capability-map.svg)

> 图例：实心节点表示有项目证据的集成、调优或交付经验，但不自动等于底层算法/kernel 的实现者；空心节点表示原理掌握、证据尚待补齐或今天会评估的能力延伸。

**20–30 秒口述版**：

> 我的主线有两条：一是基于 Megatron 的大模型训练，做过 X1 200B MoE 模型、长上下文和国产卡性能闭环；二是基于 verl/AReaL 的后训练，做过 Fully Async RLVR、Agentic RL 和 MOPD。我主要负责模型侧系统集成、性能与正确性优化，以及从跑通到性能达标的交付闭环。

## 1. 复习导航（按需展开）

<details>
<summary><strong>展开：六个 Part、Core 10 与全量题目索引</strong></summary>

### 1.1 六个 Part：先知道每一部分解决什么问题

能力图给出个人主线；下面这张表把主线映射到可直接进入的面试 Part。数字按唯一问题计数，Core 是 P0 的子集。

| Part | 解决的核心问题 | 关键入口 | 优先级与题量 |
|---|---|---|---:|
| [Part I](#part-i) | 你是谁、做了什么、为什么值得信任 | 自我介绍、Ownership、职业选择 | Core 3 / P0 3 / P1 3 / P2 1，共 7 |
| [Part II](#part-ii) | 大模型如何放得下、跑得快、扩得稳 | Megatron、5D、MoE、显存、长上下文 | Core 3 / P0 20 / P1 6 / P2 1，共 27 |
| [Part III](#part-iii) | RL dataflow 如何被框架和训练/推理后端承载 | PPO/GRPO/DPO、verl、Fully Async、Rollout 优化、真实模型落地 | Core 1 / P0 11 / P1 5 / P2 1，共 17 |
| [Part IV](#part-iv) | Agent trajectory 如何在线生产、校验和消费 | AReaL、Gateway、staleness、MOPD、weight sync | Core 2 / P0 11 / P1 6 / P2 1，共 18 |
| [Part V](#part-v) | 跨框架的通信、恢复、推理与生产排障 | Collective、万卡稳定性、训练异常、NCCL、checkpoint | Core 1 / P0 4 / P1 5 / P2 1，共 10 |
| [Part VI](#part-vi) | 如何把知识变成首面表现 | 三天冲刺、口径校准、证据卡、模拟面试 | 不新增问题 |

全文共 **79 道唯一问题**：P0 49 道、P1 25 道、P2 5 道。Core 10 已计入 P0，不重复计数；Coding 手撕题单独维护，不计入这里。

### 1.2 Core 10：建立个人项目主线的十个入口

Core 10 用于建立自我介绍、项目和机制之间的回答链，已计入 P0。已有基础、准备下一轮时，优先按 [VI.0](#vi-0) 补实际薄弱项；各题的完整答案只保留在所属 Part。

| 顺序 | 所属 Part | 题目 |
|---:|---|---|
| 1 | Part I | [RESUME-01｜请做一个 1–2 分钟自我介绍](#resume-01) |
| 2 | Part I | [RESUME-01B｜你在项目中的 Ownership 是什么？](#resume-01b) |
| 3 | Part I | [RESUME-01C｜为什么从华为到小鹏，现在为什么又看机会？](#resume-01c) |
| 4 | Part II | [RESUME-01A｜最有代表性的性能优化是什么？](#resume-01a) |
| 5 | Part II | [MEGATRON-01｜Megatron 的“5D 并行”分别解决什么问题？](#megatron-01) |
| 6 | Part II | [INFRA-02｜Megatron 训练显存如何计算？遇到 OOM 怎么定位？](#infra-02) |
| 7 | Part III | [RESUME-02｜Fully Async 相比同步 RLVR 有什么优势？](#resume-02) |
| 8 | Part IV | [RESUME-08｜请画出你的 Agentic RL 训练链路，最大瓶颈在哪里？](#resume-08) |
| 9 | Part IV | [RESUME-09｜OPD/MOPD 解决什么问题？](#resume-09) |
| 10 | Part V | [INFRA-04｜常见通信算子执行什么操作，分别用在哪里？](#infra-04) |

### 1.3 全量问题索引：按 Part 定位，按优先级学习

<details>
<summary><strong>Part I｜个人定位、Ownership 与职业选择（7）</strong></summary>

- **P0 / Core**：[RESUME-01 自我介绍](#resume-01) · [RESUME-01B Ownership](#resume-01b) · [RESUME-01C 职业选择](#resume-01c)
- **P1**：[RESUME-11 第二个 Ownership 案例](#resume-11) · [RESUME-16 带 4–5 人交付](#resume-16) · [BEHAVIOR-01 为什么匹配薪资档位](#behavior-01)
- **P2**：[P2-06 为什么从算法研究转向训练 Infra](#p2-06)

</details>

<details>
<summary><strong>Part II｜Megatron、MoE、训练后端与长上下文（27）</strong></summary>

- **P0 / Core**：[RESUME-01A X1 200B MoE 模型性能优化](#resume-01a) · [MEGATRON-01 5D 并行](#megatron-01) · [INFRA-02 Megatron 显存账本](#infra-02)
- **P0 扩展**：[RESUME-05 SFT 31s→9.3s](#resume-05) · [RESUME-17 35B-A3B 128K](#resume-17) · [RESUME-06 128K/256K 显存](#resume-06) · [RESUME-07 CP-local logits](#resume-07) · [KERNEL-01 NVIDIA 融合算子](#kernel-01) · [RESUME-10 千卡/万卡交付](#resume-10) · [MEGATRON-02 Column/Row Parallel](#megatron-02) · [MEGATRON-03 TP 变大为什么更慢](#megatron-03) · [MEGATRON-04 SP 与 CP](#megatron-04) · [MEGATRON-05 Distributed Optimizer](#megatron-05) · [MOE-01 Dense 与 MoE](#moe-01) · [MEGATRON-06 EP 与 all-to-all](#megatron-06) · [INFRA-01 MFU](#infra-01) · [DIST-01 FSDP/FSDP2 与 ZeRO](#dist-01) · [MEGATRON-11 训练框架分层与选型](#megatron-11) · [SFT-DATA-01 数据到 loss 正确性](#sft-data-01) · [MLLM-01 多模态与具身训练差异](#mllm-01)
- **P1**：[RESUME-18 视频 DiT/Ulysses](#resume-18) · [MEGATRON-07 PP bubble](#megatron-07) · [MEGATRON-08 Packed Sequence](#megatron-08) · [MEGATRON-09 Recompute/Offload](#megatron-09) · [MEGATRON-10 Distributed checkpoint](#megatron-10) · [BRIDGE-01 MBridge/Megatron Bridge](#bridge-01)
- **P2**：[P2-02 FlashAttention](#p2-02)

</details>

<details>
<summary><strong>Part III｜RL 算法、verl 与 Fully Async RLVR（17）</strong></summary>

- **P0 / Core**：[RESUME-02 Fully Async RLVR](#resume-02)
- **P0 扩展**：[RL-ALGO-01 PPO/GRPO/DAPO](#rl-algo-01) · [DPO-01 DPO 与 SFT/PPO/GRPO](#dpo-01) · [RESUME-03 gen-TP 与实例数](#resume-03) · [ROLLOUT-01 Rollout 优化全景](#rollout-01) · [VERL-01 HybridFlow 架构](#verl-01) · [VERL-02 colocate/disaggregate](#verl-02) · [VERL-03 训练到 rollout 权重同步](#verl-03) · [VERL-04 async/streaming/partial/staleness](#verl-04) · [VERL-05 RLVR 正确性](#verl-05) · [VERL-09 vLLM/SGLang 选型](#verl-09)
- **P1**：[VERL-06 DataProto/WorkerGroup](#verl-06) · [VERL-07 Actor/Ref/Critic/Reward](#verl-07) · [VERL-08 Ray 故障](#verl-08) · [VERL-10 v0.7 以后演进](#verl-10) · [VERL-11 自研版 verl 模型落地](#verl-11)
- **P2**：[P2-05 producer-consumer coding](#p2-05)

</details>

<details>
<summary><strong>Part IV｜AReaL、Gateway、Agentic RL 与 MOPD（18）</strong></summary>

- **P0 / Core**：[RESUME-08 Agentic RL 链路](#resume-08) · [RESUME-09 OPD/MOPD](#resume-09)
- **P0 扩展**：[AREAL-01 verl/AReaL 选型](#areal-01) · [AREAL-02 off-policyness](#areal-02) · [AREAL-03 微服务化](#areal-03) · [AREAL-04 trajectory→gradient](#areal-04) · [AREAL-09 Gateway 改造](#areal-09) · [AREAL-11 XCCL 与 disk](#areal-11) · [RESUME-13 CUDA Graph](#resume-13) · [RESUME-19 Gateway 调度收益](#resume-19) · [RESUME-14 Prefix Cache](#resume-14)
- **P1**：[RESUME-15 Rejected Group](#resume-15) · [AREAL-05 Partial Rollout](#areal-05) · [AREAL-06 原子 weight sync](#areal-06) · [AREAL-07 Online Proxy/session drain](#areal-07) · [AREAL-10 外部 Agent 接入](#areal-10) · [AREAL-08 三层门禁](#areal-08)
- **P2**：[P2-04 设计 256K Agentic RL 平台](#p2-04)

</details>

<details>
<summary><strong>Part V｜通用 Infra 与生产排障（10）</strong></summary>

- **P0 / Core**：[INFRA-04 通信算子](#infra-04)
- **P0 扩展**：[TRAIN-ANOMALY-01 loss/NaN/梯度/收敛排障](#train-anomaly-01) · [INFRA-09 万卡规模效应与优化](#infra-09) · [INFRA-03 NCCL hang/checkpoint 恢复](#infra-03)
- **P1**：[RESUME-12 精度对齐](#resume-12) · [INFRA-05 64 卡 35B MoE 128K](#infra-05) · [INFRA-06 推理吞吐/延迟/KV](#infra-06) · [INFRA-07 可观测性指标树](#infra-07) · [INFRA-08 可恢复 checkpoint](#infra-08)
- **P2**：[P2-03 kernel/带宽/通信瓶颈](#p2-03)

</details>

初次准备先用 Core 10 建立主线，之后按岗位和实际薄弱项选择 Part。P0 表示高频基础或核心项目，P1 表示深入追问，P2 表示按需补充；不要求在一次复习里精读全部 P0。现场用顶部简历入口表，日常学习用这里的全量索引；时间安排与项目口径见 [Part VI](#vi-0)。

</details>

---

<a id="part-i"></a>
## Part I｜个人定位、Ownership 与职业选择

**学习目标**：先建立面试官对你的职业主线、个人贡献和稳定性的判断；所有技术深挖都从这里选择入口。

**本 Part 导航**：P0 / Core：[自我介绍](#resume-01) · [Ownership](#resume-01b) · [职业选择](#resume-01c)；P1：[第二个贡献案例](#resume-11) · [4–5 人协同交付](#resume-16) · [岗位价值](#behavior-01)；P2：[从研究到 Infra](#p2-06)。

### Core｜最高优先入口

<a id="resume-01"></a>
#### RESUME-01｜请做一个 1–2 分钟自我介绍（P0，12 分钟）

- **直接回答（60–90 秒）**：

  > 面试官您好，我叫曾柏炜，本科毕业于厦门大学，硕士毕业于清华大学，研究方向是人工智能。我目前在小鹏机器人做大模型后训练基础设施，主要负责两块：基于 verl 和 Megatron-Core 的 SFT、RLVR 训练，以及基于 AReaL 的 Agentic RL 和多 Teacher 蒸馏。具体工作集中在长上下文、rollout 调度、显存性能和训练正确性。
  >
  > 之前在华为，我主要做国产卡上的模型适配、精度和性能优化，代表项目是 X1 200B MoE，并支撑过 3K 卡连续稳定训练两个月。两个阶段共同的工作方式是：先把模型跑通，再通过 profiling 找瓶颈，验证优化效果，最终支持算法实验和模型产出。我希望继续深入训练 Infra，承担核心模块的技术工作和交付责任。

- **被要求量化时再补**：在 Fully Async 路径内部，通过 rollout 资源配比与联合配置优化，代表性稳态吞吐从 `76` 提升到 `211–255 tokens/s/GPU`；这是异步初始配置和优化配置的比较，具体 workload 与统计边界见 [Fully Async 项目](#resume-02)。教育专业追问再补“厦大电气工程及其自动化、清华电子信息”。

- **项目证据或知识边界**：所有数字必须能回到固定 workload。自我介绍只说“多 Teacher 在线蒸馏链路和方向性效果结论”；被追问时再使用“最新版双 Teacher MOPD 在 SWE、Terminal 双域提升且 General 不下降”，不主动混入单 Teacher pp 数字，也不暗示统计信息已全部补齐。
- **高概率追问**：[最有代表性的优化](#resume-01a)是什么？你在项目中的 [ownership](#resume-01b)？[为什么从华为到小鹏、现在又看机会](#resume-01c)？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：请介绍一下你自己，重点讲与大模型训练推理 Infra 相关的经历。

- **面试官意图**：判断你的职业主线、表达能力和 seniority；同时选择后续深挖入口。

- **危险回答**：连续罗列十几个框架；教育背景超过 10–15 秒或展开课程、论文和奖项；说“全栈负责”却说不清代码和实验边界。

</details>

↩ [返回本 Part 导航](#part-i) · ↑ [返回面试速查控制台](#interview-console)

<a id="resume-01b"></a>
#### RESUME-01B｜你在项目中的 Ownership 是什么？（P0，10 分钟）

- **直接回答（60 秒）**：

  > 我理解的 ownership，是对一段明确的工作从问题定位、方案选择到最终验收负责。以 X1 200B MoE 为例，我负责模型侧从功能、精度可用到性能达标，再到规模训练的验证。
  >
  > 我具体做的是采集性能数据、定位瓶颈，调整并行配置，接入 Grouped MatMul 和融合算子，再验证通信 overlap、性能和精度。遇到算子、编译器或集群问题，我提供可复现的输入和 profiling 证据，推动对应团队解决，最后完成模型侧回归。框架和底层能力由相关团队提供，我对这些能力在目标模型上能否正确、高效地工作负责。

- **追问怎么展开**：按“负责范围 → 亲自改动 → 关键取舍 → 团队依赖 → 验收结果”展开，[X1 性能优化](#resume-01a)给出主故事，[千卡交付](#resume-10)说明规模边界。是否担任正式第一接口人、是否制定组织级门禁等职责，需有本人事实再补。
- **项目证据或知识边界**：准备一项亲自改动、一项关键实验、一项被你否决的方案和一次跨团队闭环。明确哪些融合算子是直接使用、哪些是适配或修改，不能把 Megatron/MindSpeed 原生能力说成自研。
- **高概率追问**：最终方案谁拍板？你写了哪些模块？底层算子不是你写的，为什么结果算你的？如果没有你项目最可能卡在哪里？失败时你承担什么责任？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：Ownership 是什么意思？X1 项目里哪些事情是你负责的，哪些是框架或团队完成的？

- **面试官意图**：判断你是否达到高级工程师所需的端到端责任能力，同时拆分个人贡献、开源能力和团队红利。

- **危险回答**：“我全栈负责”“基本都是我做的”；只讲协调不讲技术判断；只讲代码不讲上线结果；用团队总成果替代个人边界。

</details>

↩ [返回本 Part 导航](#part-i) · ↑ [返回面试速查控制台](#interview-console)

<a id="resume-01c"></a>
#### RESUME-01C｜为什么从华为到小鹏，现在为什么又看机会？（P0，8 分钟）

- **直接回答（60 秒）**：

  > 从华为到小鹏，首先是地点因素：当时部门计划整体搬迁上海，我的家庭和长期定居规划在深圳。技术上，我已经积累了国产卡模型适配和性能优化经验，希望进一步做 GPU 上的训练框架与后训练系统，小鹏当时的岗位正好匹配。
  >
  > 这次看机会，是因为部门有比较大的组织调整，团队方向和岗位职责存在不确定性。我仍希望在深圳长期做大模型训练 Infra，重点看团队方向、技术工作内容，以及能否对核心模块承担明确责任。这几个方面匹配的话，我希望长期发展。

- **项目证据或知识边界**：面试只说“家庭和长期定居规划在深圳”，不主动展开结婚、生娃和买房；只看深圳可以坦诚，但宝安、南山及周边的通勤范围留到 HR 确认办公地点时再说。
- **高概率追问**：如果小鹏组织稳定是否还会看机会？为什么入职不到一年？你只看深圳会不会限制发展？什么条件能让你长期留下？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：两次职业选择的原因是什么？如何证明你加入后会稳定发展？

- **面试官意图**：判断离职动机是否客观、职业主线是否连续，以及地点、组织变化和岗位期望是否与招聘岗位匹配。

- **危险回答**：“在华为是螺丝钉、自由度低、会的太少”“小鹏现在很不稳定”；过度讨论家庭安排；把组织调整说成唯一原因；表示只要薪资更高就离开。

</details>

↩ [返回本 Part 导航](#part-i) · ↑ [返回面试速查控制台](#interview-console)

### P0 扩展｜首轮前应掌握

本 Part 无额外 P0；完成 Core 后直接进入 P1 深挖。

### P1 深挖｜面试官继续追问

<a id="resume-11"></a>
#### RESUME-11｜如何用第二个项目证明 Ownership 不是背模板？（P1，8 分钟）

- **直接回答（60 秒）**：

  > 我会用小鹏的 CP-local logits 修复来说明。长上下文 actor 路径已经把序列分到了不同 CP rank，但后处理又把 full-sequence logits 聚合回来，后面再做 chunking 也省不掉这部分显存。我负责定位这条多余的聚合路径，并将处理改为在本地 logits 上计算 logprob，最后只聚合每个 token 的标量结果，消除了简历记录的约 7.6GB 冗余分配。
  >
  > 这个工作的重点是集成路径的正确性：labels、mask 和 logits 必须按同一规则分片，修复后还要验证 logprob、loss 和梯度。CP、TP 本身由 Megatron 提供，我负责它们接入上层训练链路后是否真正起效。

- **追问展开**：[CP-local logits 的形状、修复接口与验证](#resume-07)。如改讲其他项目，必须有同样具体的“问题 → 亲自改动 → 验证”证据。
- **项目证据或知识边界**：不要重复 X1 故事；不要把 verl/AReaL/Megatron 开源能力描述成自研。若选 MOPD，效果只使用“最新版双 Teacher 在 SWE、Terminal 双域提升且 General 不下降”，并把个人工程贡献与算法/评测团队贡献拆开。
- **高概率追问**：关键设计谁拍板？如果没有你项目会怎样？你 review 过哪些核心模块？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：除了 X1，再用小鹏项目说明一次你的 ownership，哪些来自开源框架或团队？

- **面试官意图**：验证 [RESUME-01B](#resume-01b) 的定义是否可以迁移到不同项目，而不是只会背一个华为案例。

- **危险回答**：反复使用“我们”；用 PR 数代替技术贡献；把所有收益都归因给自己。

</details>

↩ [返回本 Part 导航](#part-i) · ↑ [返回面试速查控制台](#interview-console)

<a id="resume-16"></a>
#### RESUME-16｜你如何带 4–5 人完成复杂交付？（P1，8 分钟）

- **直接回答（45–60 秒）**：

  > 在华为 TX 项目中，我有协同 4–5 人做模型交付的经历。我的核心工作仍然是模型适配、精度和性能问题定位；问题涉及算子、框架或集群时，需要把模型侧的复现条件、性能数据和验收结果交给对应同事，一起推进。
  >
  > 对我来说，技术协同最重要的是把问题说清楚：现在卡在哪一层，谁能解决，修复后用什么数据证明有效。我的个人贡献以模型侧定位、优化和回归为主，4–5 人是项目协同规模。

- **被追问冲突或授权时**：需要补本人真实事件，再讲“当时分歧、判断依据、最终选择和结果”；当前材料不足以写成具体管理故事。
- **项目证据或知识边界**：华为 TX 项目有 4–5 人团队经验；说明是项目协同还是正式 people management。
- **高概率追问**：成员意见冲突怎么办？如何判断自己下钻还是授权？怎样评价交付质量？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：请讲一次技术负责人/项目负责人的具体做法。

- **面试官意图**：评估高级工程师的带项目能力，而不仅是个人贡献。

- **危险回答**：只讲开会和催进度；把协调当管理的全部；没有技术验收机制。

</details>

↩ [返回本 Part 导航](#part-i) · ↑ [返回面试速查控制台](#interview-console)

<a id="behavior-01"></a>
#### BEHAVIOR-01｜为什么你匹配 100–150 万档位？（P1，10 分钟）

- **直接回答（45–60 秒）**：

  > 我能带来的价值，是把训练框架能力接到真实模型上，并持续解决性能和正确性问题。在华为，我做过 X1 200B MoE 的模型侧优化和规模训练交付；在小鹏，我做过 Megatron 后端的长上下文 SFT、verl 异步 RLVR，以及 AReaL 的 Agentic RL 和蒸馏链路。
  >
  > 这些经历让我能从模型计算、通信、显存一直追到 rollout 和训练数据，判断瓶颈在哪里、该改哪一层、怎样验证。我希望下一岗位继续对核心模块承担明确责任，薪资也结合岗位职责和实际影响来讨论。

- **项目证据或知识边界**：当前工作年限约 3 年多，高薪档位会追问深度和影响范围；补齐服务用户数、GPU-hours、默认 recipe/主干贡献等业务影响数据。
- **高概率追问**：为什么现在换工作？期望总包结构？若达不到怎么办？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：相比普通训练工程师，你的不可替代性是什么？

- **面试官意图**：评估价值密度、稳定性、动机与薪资合理性，不是邀请你直接报数字。

- **危险回答**：只用学历/大厂背景论证；把目标薪资作为换工作唯一原因；虚构团队影响。

</details>

↩ [返回本 Part 导航](#part-i) · ↑ [返回面试速查控制台](#interview-console)

### P2 选学｜时间允许再补

<a id="p2-06"></a>
#### P2-06｜为什么从算法研究转向训练 Infra？（P2，6 分钟）

- **直接回答（30–45 秒）**：

  > 研究经历让我养成了读论文、设计对照实验和验证结果的习惯。后来在华为和小鹏做项目，我发现自己更愿意深入模型背后的系统问题：为什么跑得慢、哪里占显存、分布式之后为什么结果变了。训练 Infra 能直接影响算法同学的实验效率，也需要理解模型和数值机制，所以我希望长期沿这个方向深入。

- **项目证据或知识边界**：可引用论文和两个阶段的职业转变，不需展开病理图像算法细节。
- **高概率追问**：未来更想做训练还是推理？是否愿意写底层 C++/CUDA？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：你的 MICCAI/AAAI 经历如何帮助当前工作？

- **面试官意图**：评估职业动机、学习能力和长期稳定性。

- **危险回答**：“算法太卷所以转 Infra”；把 Infra 描述成部署运维；职业方向摇摆。

</details>

↩ [返回本 Part 导航](#part-i) · ↑ [返回面试速查控制台](#interview-console)

### 本 Part 追问路线

自我介绍 → X1/Fully Async/AReaL 三选一 → Ownership → 职业选择；谈薪时再进入岗位价值与 100–150 万档位匹配度。

---

<a id="part-ii"></a>
## Part II｜Megatron、MoE、训练后端与长上下文

**学习目标**：用 X1 200B MoE 模型和长上下文 SFT 证明训练系统基本盘：框架选型、数据契约、并行、显存、算子、通信、精度与规模交付，并能把能力迁移到多模态/具身训练。

**本 Part 导航**：

- **先看全景**：[Megatron 训练整体优化方案](#megatron-optimization-overview)
- **Core**：[X1 MoE 优化](#resume-01a) · [5D 并行](#megatron-01) · [显存账与 OOM](#infra-02)
- **P0 项目**：[9B SFT 加速](#resume-05) · [35B-A3B 128K](#resume-17) · [长上下文显存](#resume-06) · [CP-local logits](#resume-07) · [融合算子](#kernel-01) · [千卡规模交付](#resume-10)
- **P0 机制**：[TP Linear](#megatron-02) · [TP 负优化](#megatron-03) · [SP 与 CP](#megatron-04) · [Distributed Optimizer](#megatron-05) · [Dense 与 MoE](#moe-01) · [EP 与 All-to-All](#megatron-06) · [MFU](#infra-01) · [FSDP 与 ZeRO](#dist-01) · [训练后端选型](#megatron-11) · [SFT 数据正确性](#sft-data-01) · [多模态与具身](#mllm-01)
- **P1**：[视频 Ulysses](#resume-18) · [PP bubble](#megatron-07) · [Packed Sequence](#megatron-08) · [Recompute 与 Offload](#megatron-09) · [Checkpoint 换并行度](#megatron-10) · [两种 Bridge](#bridge-01)
- **P2**：[FlashAttention 原理](#p2-02)

<a id="megatron-optimization-overview"></a>
### 训练优化总览｜如何系统优化一个 Megatron 训练任务？

> **P0 复习入口**：先掌握这条回答主线，再按表格跳转到具体题；本节是已有问题的总览，不另增题号。适用于 Dense/MoE 的预训练与 SFT，EP、router、dispatcher 等是 MoE 特有项。

**60–90 秒口述**：

> 我会先固定模型、数据、序列长度、batch、精度和硬件，建立性能与正确性基线。然后沿一个训练 step 做 profiling，先确认是否在等数据，再看显存、通信和计算效率哪个是主要瓶颈。
>
> 并行策略先保证模型放得下，再比较不同 TP、CP、PP、EP 组合下的实际吞吐，避免切分过细导致小 GEMM 和通信开销。显存侧用状态分片、选择性重计算和必要的 offload；计算侧看 Grouped GEMM、融合算子和 Attention kernel；通信侧看拓扑、token dispatcher，以及没有被计算掩盖的等待。MoE 还要看专家负载是否均衡，不能只看平均利用率。
>
> 每次针对一个瓶颈做可归因的 A/B，再重新 profile，因为省下显存后，最优并行度和 batch 也可能变。最终要同时验证 step time、有效吞吐、显存、loss 和恢复能力，而不是只证明某个算子更快。

#### 1. 用“三堵墙”定位，用五类手段优化

**Memory Wall 是装不下或被迫缩小 batch；Communication Wall 是跨卡等待过多；Compute Efficiency Wall 是算力没有高效转成有效计算。** 三者会互相影响，并行配置横跨三者，生产能力保障长期可用；数据供给与 checkpoint I/O 也要单独检查，不能因为不在“三堵墙”名称里就忽略。

| 优化维度 | 具体怎么做 | 验证什么，避免什么代价 | 继续查题 |
|---|---|---|---|
| **① 并行与拓扑** | 用 TP/PP/CP/DP/EP 分摊计算与状态；MoE 用 Parallel Folding 分别选择 Attention 与 Expert 网格；用 microbatch/VPP 调整流水线 | 每卡峰值、GEMM 大小、通信暴露时间、PP bubble；并行度大不一定更快 | [5D/Folding](#megatron-01) · [TP](#megatron-02) · [SP/CP](#megatron-04) · [PP/VPP](#megatron-07) |
| **② 显存** | 区分参数、梯度、optimizer、activation 和临时 buffer；选择状态分片、细粒度 recompute、memory-efficient permutation、precision-aware optimizer、activation offload | 看峰值发生在哪一段；重算增加计算，offload 增加传输，状态降精度需验证数值。优先消除非预期全量张量 | [显存账本](#infra-02) · [Recompute/Offload](#megatron-09) · [FSDP/ZeRO](#dist-01) · [CP logits](#resume-07) |
| **③ 通信** | 分开看 TP/DP collective、CP KV、EP dispatch/combine 和 PP P2P；匹配拓扑和 dispatcher，按依赖安排重叠；NVIDIA 路径可评估 DeepEP/HybridEP，以及 Dgrad/Wgrad 拆分与延后 Wgrad 的 EP overlap | 看未被掩盖的通信和最慢 rank；overlap 可能争抢 SM、HBM 或链路，必须复测端到端 | [通信算子](#infra-04) · [EP/A2A](#megatron-06) · [千卡/万卡](#infra-09) |
| **④ 计算与执行** | Grouped GEMM 提高多 expert 计算效率；融合 router、permute/unpermute、激活、归一化或 loss，减少中间读写；选合适 Attention kernel；评估 CUDA Graph 与 sync-free 路径 | 区分 GEMM 效率低、HBM 受限、CPU launch/同步空洞；Graph 不减少数学 FLOPs，动态形状也不能假定全图可捕获 | [融合算子](#kernel-01) · [FlashAttention](#p2-02) · [Profiler](#p2-03) · [Graph 原理](#resume-13) |
| **⑤ 生产与模型初始化** | 监控 router/expert 负载，评估 dropless 或容量策略；用 distributed optimizer/FSDP 管理状态，用 distributed checkpoint 支持保存恢复；需要时评估 dense→MoE upcycling | token dropping 会改变训练语义；upcycling 是初始化方案，不是 step 加速开关。验收 loss、效果、恢复和长期有效训练时间 | [MoE 路由](#moe-01) · [Distributed Optimizer](#megatron-05) · [Checkpoint](#megatron-10) · [训练异常](#train-anomaly-01) |

五类能力对应你提供的报告 §1.3；“瓶颈—动作—验证”是面试中的工程组织方式。机制可对照 [NVIDIA MoE 报告 §1.3](https://arxiv.org/html/2603.07685v1#S1.SS3) 与 [Megatron Core MoE 官方指南](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html)（本节核验于 2026-09-07）；具体开关、组合与硬件支持以所用版本为准。

#### 2. 真正动手时的顺序

1. **固定基线、保证算对**：锁定 global batch/有效 token 口径、精度、卡数、软件版本、warmup 和统计窗口；保留配置与 loss 基线。MoE 按总参数算权重容量，不能只拿激活参数量估显存。
2. **先能放下，再选更快的切法**：估算显存，找可行并行网格；在同一 workload 下比较吞吐。显存不够先处理容量，能跑以后按实际瓶颈排序，不强制先优化某一种算子。
3. **逐项归因、循环调优**：数据等待高就调 DataLoader workers/prefetch，padding 浪费高再评估 packing；计算差就看 kernel/GEMM；通信暴露高就看拓扑、dispatcher/overlap。释放显存后重新评估 microbatch、TP/CP 和 recompute，但全局训练语义不能悄悄改变。
4. **从局部收益走到规模验收**：同时记录 step time 的均值/尾部、有效 tokens/s/GPU、MFU、峰值显存、expert 负载与错误率；回归 loss/模型效果、checkpoint 恢复及多机稳定性。不把单 kernel 加速直接当作整步收益。

#### 3. 三个容易被追问的边界

- **Folding 不是多出一批卡**：同一 PP 划分下，`world_size = PP × TP × CP × DP = PP × ETP × EP × EDP`。它解除 Attention/Expert 切分绑定；截图的“打破 EP≤DP”针对传统受限布局，不是说所有版本都用同一种 DP 定义，更不能把 EP 额外乘到总卡数上。[双网格、8/256 GPU 例子](../training-infra-roadmap/topics/moe.md#parallel-folding)
- **Sync-free 不是取消分布式同步**：这里主要减少为了获知动态 expert token 数而发生的 CPU–GPU 同步，让调度信息尽量留在设备端；collective、数据依赖和梯度语义仍需保证。Dropless MoE 的 Graph 捕获范围取决于 kernel、内存与框架支持，可能只捕获静态子图。[报告 §4.3.7](https://arxiv.org/html/2603.07685v1#S4.SS3.SSS7)
- **低精度是跨维度手段，不是全部改成 FP8/FP4**：它可能同时影响 activation、GEMM 和部分通信，但要核对硬件/算子支持、量化额外开销与敏感路径精度；precision-aware optimizer 也不等于把所有 optimizer state 无条件降精度。[低精度与收敛边界](https://arxiv.org/html/2603.07685v1#S5)

#### 4. 接回自己的项目

- **X1 200B MoE**：重点落在并行策略、Grouped MatMul、融合算子、通信掩盖与模型侧性能交付，接 [代表性优化](#resume-01a)。DeepEP、HybridEP、Parallel Folding、FP8/FP4 等未确认方案只能说“今天会评估”；NVIDIA 实现不能直接套成国产卡当年的配置。
- **9B SFT / 长上下文**：重点落在 DataLoader 并发与预取、选择性重计算、TP/CP 调整，接 [31s→9.3s](#resume-05)。[35B/128K](#resume-17) 与 [CP-local logits](#resume-07) 分别讲，不补造联合消融。
- **适用范围**：上表是技术工具箱，不代表本人全部实现或使用。训练 CUDA Graph 的候选收益也不能套用 [Agentic RL decode 6–8x](#resume-13) 的测量结果。

↩ [返回本 Part 导航](#part-ii) · ↑ [返回面试速查控制台](#interview-console)

### Core｜最高优先入口

<a id="resume-01a"></a>
#### RESUME-01A｜最有代表性的性能优化是什么？（P0，20 分钟）

- **直接回答（60–90 秒）**：

  > 最有代表性的是华为 X1 的 200B MoE 预训练优化。我负责模型侧从功能打通、精度对齐到性能达标，并支撑 3K 卡训练。优化主要有三类：调整并行组合，在模型放得下的前提下避免 expert GEMM 被切得过碎；接入 Grouped MatMul 和融合算子，改善小矩阵效率、减少中间读写；分析通信暴露时间，利用框架的 overlap 能力减少等待。每轮改动后重新 profile，并回归精度和稳定性。
  >
  > 最终按客户对标口径，相对性能从 `0.16x` 提升到 `0.95x`，MFU 达到 35%，支撑 3K 卡连续稳定训练两个月。我的直接贡献是模型集成、性能归因和调优闭环；底层算子库、集合通信或硬件问题由对应团队协同解决，不都算成我个人的实现。

- **2–3 分钟展开版**：按下面五层展开，不要把未确认项说成已经实施。

  1. **先固定 benchmark**：补齐 `0.16x/0.95x` 的分母，以及模型层数、专家数/top-k、global/micro batch、sequence length、precision、卡数、warmup 和统计窗口；否则数字没有解释力。
  2. **并行与拓扑**：从容量可行的 TP/PP/DP/EP 候选中选择吞吐更高的组合。MoE 的 expert 通常已经是小 GEMM，过高 expert-TP 可能进一步碎片化计算；但最终映射必须结合当时实际 HCCS/RoCE 拓扑、collective 频率和消息量说明。面试前补齐真实并行度和通信 group 到物理拓扑的映射。
  3. **计算效率**：Grouped MatMul 把不同 expert 的可变 token batch 聚合调度，提高硬件利用率；融合算子要解释具体融合了什么、减少了哪些读写或 launch。准备实际使用过的 2–3 个算子名，以及各自 A/B 收益，不能泛称“各种融合算子”。
  4. **通信掩盖**：按 TP/DP collective、EP dispatch/combine 和 PP P2P 分别看 exposed time。准备一条当时真实的 overlap timeline，说明 compute/communication 的依赖如何解除、使用了什么 stream/chunk/schedule，以及为什么 overlap 后没有因资源争用拖慢 GEMM。
  5. **显存、精度和规模化**：只讲确认使用过的显存手段，例如实际的 optimizer 分片、sequence parallel、recompute 或 buffer 复用；融合和低精度路径用逐层 dump 找 first divergence。最后从小规模功能/精度基线扩到 3K 卡，验证 loss、吞吐、checkpoint 和故障恢复。

- **结合 NVIDIA 2026 MoE 报告可以补充什么**：按 [训练整体优化总览](#megatron-optimization-overview) 的三堵墙与五类手段展开；Parallel Folding、DeepEP/HybridEP、更细的 EP overlap、低精度与 Graph 路径是今天继续演进时会评估的方向，不是 X1 当时已经落地的成果。

- **项目证据或知识边界**：可以口述 X1、200B MoE 模型、`0.16x → 0.95x`、MFU 35% 和 3K 卡连续稳定训练两个月；对外简历继续脱敏。客户真实名称不写入或展示。上述 NVIDIA 新方案必须使用“今天会评估”，不能倒灌成 2023–2024 年项目事实。
- **高概率追问**：`0.16x` 的分母是什么？实际 TP/PP/DP/EP 怎么配？Grouped MatMul 为什么有效？EP all-to-all 占比多少？load imbalance 怎么测？哪项优化收益最大？为什么 MFU 只有 35%？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：请讲一个你最有代表性的优化案例，最好能体现大模型训练 Infra 的系统能力。

- **面试官意图**：验证你能否把超大 MoE 的性能问题拆成并行映射、kernel、通信、显存、精度和规模化稳定性问题；同时检查 `0.16x → 0.95x` 是否有明确口径和个人贡献。

- **危险回答**：从头到尾罗列开关；把总收益全部归因给 Grouped MatMul；无法给出真实并行配置和融合算子名；把 NVIDIA 2026 的 DeepEP、Parallel Folding 或 CUDA Graph 说成当时已实施。

</details>

↩ [返回本 Part 导航](#part-ii) · ↑ [返回面试速查控制台](#interview-console)

<a id="megatron-01"></a>
#### MEGATRON-01｜Megatron 的“5D 并行”分别解决什么问题？（P0，15 分钟）

- **直接回答（60–90 秒）**：

  > 可以按“切什么”记：DP 切 batch，TP 切层内矩阵，PP 切模型层，CP 切长序列，EP 切 MoE experts。对应的代价也不同：TP 是逐层 collective，PP 是 pipeline bubble，CP 需要交换 KV，EP 需要搬运路由后的 token。
  >
  > “5D”不等于五个数字机械相乘。Parallel Folding 中，Attention 和 Expert 是同一批 GPU 的两套逻辑网格，分别计算后都必须等于 world size；EP 不能再乘到 Attention 的 GPU 数上。实际选型先满足参数和 activation 容量，再结合拓扑比较通信与 GEMM 效率。我在项目中主要使用、集成和调优这些能力，没有实现底层 process group 或并行调度算法。

- **如果面试官问“Megatron 的原理是什么”**：Megatron-Core 本质上是把一个 Transformer step 表达成多维 rank mesh 上的 SPMD 执行。初始化阶段由 `parallel_state` 按 TP/PP/CP/DP/EP 建 process groups；模型层把 Linear、Attention、MoE 和 sequence layout 绑定到对应 group；pipeline schedule 排列 microbatch 的 forward/backward；optimizer、checkpoint 再按同一分片元数据管理训练状态。每个进程运行同一套训练程序，但只持有和计算自己所属 shard，并通过 collective/P2P 恢复完整数学语义。

  ```text
  config / rank mesh
      -> process groups
      -> sharded Transformer modules
      -> pipeline-scheduled forward/backward
      -> gradient reduction + local optimizer step
      -> distributed checkpoint
  ```

  这段只用于建立框架全景；TP 层内数据流见 [MEGATRON-02](#megatron-02)，pipeline schedule/bubble 见 [MEGATRON-07](#megatron-07)，optimizer state 分片见 [MEGATRON-05](#megatron-05)，checkpoint 见 [MEGATRON-10](#megatron-10)。

- **MoE world-size 的正确口径**：

  ```text
  Dense / Attention：world_size = TP × CP × DP × PP
  MoE / Expert：     world_size = ETP × EP × EDP × PP

  每个 PP stage 内：TP × CP × DP = ETP × EP × EDP
  ```

  左右两边是同一批 rank 的两套 process-group mapping，不能再彼此相乘。`ETP` 是 Expert Tensor Parallel，不能默认等于 Attention TP。Parallel Folding 论文的单 PP-stage 示例是：

  ```text
  Attention：TP=2 × CP=2 × DP=2 = 8
  Expert：   ETP=1 × EP=8 × EDP=1 = 8
  ```

  论文端到端配置若再取 `PP=2`，完整作业就是 16 GPU。Megatron 官方 MoE Guide 还给出 256 GPU 示例：Attention 为 `TP4×CP2×DP8×PP4`，Expert 为 `ETP1×EP64×EDP1×PP4`，两边都等于 256。

- **为什么有时又会看到 `DP=EP×EDP`**：按当前 Megatron 的 Expert Data Parallel 定义，若传统布局取 `ETP=TP`，expert rank pool 还包含 CP，因此严格关系是：

  ```text
  CP × DP = EP × EDP
  world_size = TP × CP × PP × DP
             = TP × PP × EP × EDP       # when ETP = TP
  ```

  即 `EDP=CP×DP/EP`。只有 `CP=1`，或 legacy 资料把 expert-DP 定义为不含 CP 的子维度时，才可简写 `DP=EP×EDP`；回答前必须先声明定义。它仍只是传统 nested layout 的特例，不是通用 MoE 公式。

- **官方示例中的 DP、EP、EDP group 怎么理解**：

  - 纯 Dense DP 轴：固定 `(PP, TP, CP)`，只改变 `DP`，表示不同数据副本；`m=GBS/(MBS×DP)` 中使用这个 DP。
  - Dense `dp_cp` group：固定 `(PP, TP)`，覆盖 `DP×CP` ranks。CP ranks 也复制 Dense 参数并贡献局部 context 的梯度；Megatron 默认在该 group 上做 Dense 梯度归约和 Distributed Optimizer 分片。
  - EP group：固定 `(PP, ETP, EDP)`，改变 `EP`，持有不同 expert shard，负责 token all-to-all。
  - EDP group：固定 `(PP, ETP, EP)`，改变 `EDP`，持有同一 expert shard 的副本并同步梯度。

- **TP/CP 哪个优先放单机**：TP 通信发生在每层、每个 microbatch，通常先保证 TP 在 NVLink/NVSwitch 域；若 `TP×CP` 能放进单机，再把二者一起留在高速域。放不下时优先保持 TP 单机，并考虑下面的 hierarchical CP，让内层 CP 本地、外层 CP 跨节点；MoE 还要同时评估 EP all-to-all，不能脱离消息量、overlap 和实测 profile 给绝对答案。

<a id="megatron-hierarchical-cp"></a>

- **hierarchical CP 是什么（追问约 45 秒）**：Megatron 的 `a2a+p2p` 把 CP 分成内外两级。内层用 all-to-all，把 Q/K/V 从“短序列、较多 heads”换成“较长序列、较少 heads”；外层让对应 head 分片的 KV 沿 ring 流动，本地 Q 分块计算全局 Attention；最后逆向 all-to-all 恢复原 CP token 布局。不是简单做两次 all-gather，也不是新增一个 world-size 维度。

  例：2 台 × 8 GPU，`TP=2、CP=8、PP=DP=1`；可取 `hierarchical_context_parallel_sizes=[4,2]`，即机内 4 路 A2A、机间 2 路 P2P，`TP×4=8` 恰好占满单机。前提是 **TP 后本地 Q/KV heads 能被 4 整除**；例如总 Q heads=32、KV heads=8，TP 后分别为 16 和 4。配置合法不代表更快，还要测 A2A 重排、P2P overlap 和显存峰值。

  [展开原理：QKV shape、16 GPU 分组、配置、收益与排障](../training-infra-roadmap/topics/context_parallelism.md#hierarchical-cp) · [官方通信类型定义](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.transformer.transformer_config.html#core.transformer.transformer_config.TransformerConfig.cp_comm_type)

- **PP bubble 怎么算，VPP 解决什么问题**：设 `p` 是物理 PP stages，`m=GBS/(MBS×DP)` 是每次 iteration 的 microbatch 数，stage 均衡且忽略通信时，non-interleaved 1F1B 有：

  ```text
  useful_time = m × (t_f + t_b)
  bubble_time = (p - 1) × (t_f + t_b)
  bubble / useful = (p - 1) / m
  bubble / total  = (p - 1) / (m + p - 1)
  ```

  面试官问“额外开销”常用第一式，问“占总时间比例”用第二式。优化顺序是减小 `p`、在 GEMM 和收敛允许时增加 `m`、按真实计算量平衡 stage、再使用 VPP/interleaved 1F1B 和 P2P overlap。`VPP=v` 不增加 GPU，而是把模型切成 `p×v` 个 virtual chunks，每个物理 rank 持有多个不连续 chunk；在 interleaved 1F1B 中，若 `v` 个 chunks 的 forward/backward 近似均衡，microbatch/layer divisibility 或 custom pipeline layout 满足调度约束，并先忽略新增通信，理想 `bubble/useful≈(p-1)/(m×v)`。代价是 P2P 次数约增大 `v` 倍、activation 生命周期和调度更复杂，chunk 太小还会损害 kernel efficiency。

- **深入阅读**：[Megatron 5D 并行：每一维的动机、实现、通信、组合和考察方式](../training-infra-roadmap/topics/distributed_training.md#five-d-framework)；[MoE Parallel Folding：双逻辑网格、公式、8/256 GPU 示例和排障](../training-infra-roadmap/topics/moe.md#parallel-folding)。
- **项目证据或知识边界**：你有 Megatron 后端的配置、集成与调优经验；不要声称设计了全部并行算法。
- **高概率追问**：为什么 Dense optimizer shard group 可能是 `DP×CP`，但 microbatch 数只除以 DP？ETP 为什么不一定等于 TP？VPP 与 Zero-Bubble 有何区别？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：TP、PP、DP、CP、EP 如何组合？总 GPU 数怎么计算？

- **面试官意图**：检查分布式训练基本盘，以及你能否按模型/序列/拓扑选择并行策略。

- **危险回答**：把 `TP×CP×EP×DP` 当通用公式；把 Attention TP 和 ETP 混为一谈；把 SP/VPP 当成额外 world-size 维度；只背定义不谈通信、GEMM 粒度和拓扑。

</details>

↩ [返回本 Part 导航](#part-ii) · ↑ [返回面试速查控制台](#interview-console)

<a id="infra-02"></a>
#### INFRA-02｜Megatron 训练显存如何计算？遇到 OOM 怎么定位？（P0，18 分钟）

- **直接回答（60–90 秒）**：

  > 我按每个 rank 建两本账：一本是参数、梯度、main parameter 和 optimizer state，另一本是随阶段变化的 activation、logits、通信 buffer 和 workspace。模型状态按这个 rank 实际持有的参数及分片组算；activation 按 shape、dtype 和同时存活的份数算。最后取真正同时出现的峰值，不能把 forward、optimizer、checkpoint 的独立峰值全部相加。
  >
  > OOM 时先看发生在哪个阶段，再对比理论账和实际分配。如果是 full logits 或 dtype upcast，就先修对应张量；如果是正常 activation 峰值，再选择 MBS、CP、recompute 等手段。PyTorch snapshot 只覆盖它自己的 allocator；设备占用仍有缺口时，还要查 NCCL 等外部分配。修复后同时验显存、loss 和吞吐。

- **总账公式：先按生命周期去重**：

  ```text
  M_peak ≈ M_persistent
         + max_phase(
             M_saved_activation
           + M_phase_temp
           + M_phase_workspace
           + M_phase_comm
           )
         + M_allocator_overhead_at_peak

  tensor_bytes = product(tensor_shape) × dtype_bytes × live_copies
  ```

  `phase` 至少区分 initialization、forward、backward、optimizer、checkpoint/weight sync。每块内存只归入一个生命周期；例如 forward workspace 不能再叠加到 optimizer 峰值。`reserved-allocated` 包含 allocator cache、rounding 和不可用碎片，不能全部叫 fragmentation。

- **第一本账：参数、梯度和 Adam 状态**。Megatron Core Distributed Optimizer 官方理论值如下，`d` 是该类参数实际使用的 optimizer sharding group size：

  | 参数/梯度 dtype | 普通 optimizer | Distributed Optimizer |
  |---|---:|---:|
  | FP16 param + FP16 grad | 20 bytes/param | `4 + 16/d` |
  | BF16 param + FP32 grad | 18 bytes/param | `6 + 12/d` |
  | FP32 param + FP32 grad | 16 bytes/param | `8 + 8/d` |

  表里的 `/param` 乘的是**本 rank 在模型并行之后持有的参数量**，不是全模型参数量：

  - Dense/Attention 参数按 TP、PP 和真实 layer placement 切；普通 DP/CP 不切参数。
  - Expert 参数按 ETP、EP、PP 和 expert placement 切。
  - Dense 状态默认 `d_dense=DP×CP`；若配置多个 distributed optimizer instances，取实际 `intra_dp_cp` group size。
  - Expert 状态默认 `d_expert=EDP`；多个 instances 时取实际 `intra_expt_dp` group size。
  - embedding、LM head、router、shared expert、MTP 和 uneven PP layout 单列；first/last stage 往往不能用“总参数/PP”估算。

- **第二本账：activation**：

  1. 从每层保存到 backward 的 tensor shape 开始，乘 dtype 和 live copies，再乘该 PP rank 同时在途的 microbatch 数。
  2. CP 将 local sequence 降为 `S/CP`，参数显存不随 CP 降低；SP 只在 TP 配套区域分片部分重复 activation，不能把全部 activation 无条件再除以 TP。
  3. PP 只减少本 stage 的层数；warmup/steady/cooldown 的 activation peak 取决于 schedule 和 in-flight microbatches，不能简单除以 PP。
  4. selective/full recompute 减少 saved tensors，但会在 backward 前重算；FlashAttention 避免显式 materialize 完整 `S×S` attention matrix，但其他 activation 和 workspace 仍存在。

  Megatron 的 [`theoretical_memory_usage.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/theoretical_memory_usage.py) 可以作为估算起点，但它的 activation 公式带有 BF16、SP、selective recompute 和特定模型结构等假设；复杂 MoE 仍要回到真实 tensor shape。

- **第三本账：logits、loss 和容易漏掉的 buffer**：

  ```text
  M_logits = local_logit_positions × local_vocab_size
           × dtype_bytes × live_copies
  ```

  `local_logit_positions` 是实际生成 logits 的位置数，不自动等于 loss mask 选中的监督 token 数。要确认它是否已按 CP/packing/chunking 变成本地值、vocab 是否仍是 TP shard、loss 是否把 BF16 logits upcast 到 FP32。Vocab-parallel CE 可以避免 full-vocab gather，但不自动消除本地所有位置的 logits；是否连 LM head 一起分块或融合，要看真实实现。

  再核查 gradient accumulation、通信 overlap、GEMM/FlashAttention/Grouped GEMM workspace、临时 cast、CUDA Graph private pool、checkpoint/权重转换副本和 allocator overhead。**按物理存储去重**：contiguous param/main-grad buffer 若已经承载表中参数/梯度，bucket 又只是其 view，就不能再加一份；只追加实际 padding、额外副本和独立临时分配。[Megatron buffer API](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.distributed.param_and_grad_buffer.html)

- **面试现场的手算顺序**：

  1. 写出 parallel map，先算每个 PP rank 的 Dense 与 Expert `P_local`，找参数最重的 stage/rank。
  2. 选 dtype 行，用 `P_local × bytes/param(d)` 算模型状态；注意 Dense 的 `d_dense` 和 Expert 的 `d_expert` 不同。
  3. 用 `B_local、S/CP、H、num_layers_local、in-flight microbatches` 计算 saved activation，再应用 SP/recompute 的真实作用范围。
  4. 单列 logits/loss、通信 bucket、kernel workspace、CUDA Graph 和 checkpoint 临时峰值。
  5. 按 phase 取最大并发组合，加 allocator headroom；最后逐 rank 比较，重点看 first/last PP stage 和 hottest expert rank。

- **理论账如何闭环 OOM**：固定 workload 后，按 initialization、forward、backward、optimizer、checkpoint 记录 `allocated/reserved/max_memory_allocated`；分阶段测量时重置 peak 统计，另保留整步峰值。用 memory snapshot、tensor shape log 和 profiler 找 PyTorch 管理的未入账 allocation。若设备级占用明显高于 PyTorch reserved，再查 NCCL、CUDA context、第三方 kernel 等外部分配，不能要求 snapshot 解释全部显存。[PyTorch 显存观测边界](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html)

  先修 shape 回退、dtype upcast、buffer 生命周期或泄漏，再根据峰值来源选择 MBS、recompute、CP/TP/PP、offload、fused op 或 allocator 配置。修复后同时验证 tensor shape、峰值显存、loss、吞吐、checkpoint 和长稳，不以“不再 OOM”为结束。

- **深入阅读**：[5D 并行如何改变每-rank 参数、activation 和通信](../training-infra-roadmap/topics/distributed_training.md#five-d-config)、[Data Parallelism](../training-infra-roadmap/topics/data_parallelism.md)。
- **项目证据或知识边界**：可结合 [RESUME-07](#resume-07) 的 full-sequence logits 7.6GB 冗余分配、长样本 OOM 和 checkpoint/weight sync 峰值；dtype 与 `tokens×vocab×dtype×live copies` 精确拆解必须以原始 shape/日志为准，不默认说成 FP32。
- **高概率追问**：为什么 CP 不切参数却能参与 Dense optimizer sharding？为什么 microbatch 数只除以 DP，而 `d_dense` 默认是 `DP×CP`？某一 rank 单独 OOM有哪些原因？reserved 很高但 allocated 不高怎么办？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：请建立一份 Megatron 训练显存账本，并说明它如何指导 OOM 定位。

- **面试官意图**：检查你能否把 dtype、并行组、张量 shape 和生命周期统一到每-rank peak memory，而不是只会尝试减 batch、重计算和 offload。

- **危险回答**：用全模型参数直接乘 bytes/param；把所有显存都除以 world-size；把每个 phase 峰值和 workspace 全部相加；第一反应 `empty_cache()` 或直接减 batch；把 `reserved-allocated` 全部解释为 fragmentation。

</details>

↩ [返回本 Part 导航](#part-ii) · ↑ [返回面试速查控制台](#interview-console)

### P0 扩展｜首轮前应掌握

<a id="resume-05"></a>
#### RESUME-05｜Qwen3.5-9B SFT 为什么能从 31s 降到 9.3s？（P0，20 分钟）

- **直接回答（60–90 秒）**：

  > 这是数据供给、重算策略和并行配置共同优化的结果。我提高 DataLoader 并发并做预取，减少输入等待；把偏重的 full recompute 收敛到 selective recompute，减少不必要的重算；再调整 TP/CP，平衡本地 GEMM 效率、长序列 activation 和通信成本。
  >
  > Step time 的记录是 `31s→9.3s`，MFU 的记录是 `23%→45.2%`。不过这两组比值按标准 MFU 不能直接闭合，统计窗口和 estimator 还需回查，所以不能宣称它们来自完全相同的单一测量窗口。我也没有可靠的逐项消融，不会把总收益拆给某一个开关。另一次 `TP=4,CP=4 → TP=2,CP=8、163s→102s` 属于不同 workload，只用于解释并行策略的取舍。

- **DataLoader 追问怎么展开**：底稿确认 `num_workers=0→8` 与 prefetch。Workers 主要并行准备 CPU 数据；`num_workers=0` 本身不证明 GPU 一定在等数据，需看 `next(data_iter)` 与 GPU 空洞是否对齐。Pinned memory、`persistent_workers`、`prefetch_factor` 和 non-blocking H2D 是需要核查的配置，不在未确认时说成这次全部启用。CPU 准备下一批与 GPU 计算重叠，和 H2D copy 与 kernel 真正重叠，是两件事；后者还需要合适的独立 stream、pinned memory、可用 DMA engine 和正确依赖，不能只凭 `non_blocking=True` 判断。[PyTorch 官方教程](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)

- **Selective recompute 怎么选**：不要回答成“Attention 贵，所以不重算”。Megatron-Core 当前默认 selective 模块是 `core_attn`，原因是该区域保存的中间 activation 相对重算 FLOPs 更划算；现代版本还支持 `layernorm`、`moe_act`、`mlp`、`moe`、`shared_experts` 等模块。项目回答只确认“从偏重 full recompute 收敛到 selective”，具体 module list 必须以当时配置为准。选择方法是比较 `释放的峰值 bytes / 额外重算 FLOPs`，并确认它是否位于峰值存活窗口，再用 `none/selective/full` 三档同 workload sweep 验证 step time、peak memory、loss/grad 和 dropout/RNG 一致性。
- **MFU 算术门禁**：模型 FLOPs/step、硬件峰值口径和计时范围相同时，标准 MFU 应近似与 step time 成反比；`31/9.3≈3.33` 与 `45.2/23≈1.97` 不能自动闭合。因此需回查 estimator、data wait 是否计时、模型实际处理的 token/长度分布、packing、microbatch 与平均窗口，并单列 loss-mask 选中的监督 token。补齐前保留两组数字，但不声明同一单一计时窗口，也不用其中一个反推另一个。
- **深入阅读**：[长上下文训练：SFT 优化、selective recompute 与验证顺序](../training-infra-roadmap/topics/long_context_training.md#qwen35-9b-sft)、[Megatron-Core TransformerConfig](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.transformer.transformer_config.html)。
- **项目证据或知识边界**：最新版简历确认总结果、DataLoader 并发、selective recompute 与 TP/CP 调整方向，但没有逐项贡献。`num_workers=0→8` 来自底稿；若面试只按公开简历回答，可说“提高 DataLoader 并发并预取”。
- **高概率追问**：为什么 MFU 与 step time 比值不闭合？prefetch 如何证明真的重叠？为什么 selective 默认会重算 `core_attn`？workers 过多有什么反作用？为什么不继续增大 TP？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：DataLoader/prefetch、选择性重计算和 TP/CP 调整各解决了什么？如何证明 `31s→9.3s` 不是换 workload？

- **面试官意图**：验证你能否区分 input pipeline、GPU compute、显存与配置变化，并证明 3.3x 不是换 workload。

- **危险回答**：说“num_workers 提升 GPU 算力”；把 standard MFU 的算术矛盾糊过去；把总收益硬拆成未经 A/B 的百分比；虚构具体 `recompute_modules`；把另一 workload 的 `TP=2,CP=8` 说成这次最终配置。

</details>

↩ [返回本 Part 导航](#part-ii) · ↑ [返回面试速查控制台](#interview-console)

<a id="resume-17"></a>
#### RESUME-17｜Qwen3.5-35B-A3B 在 128K 下为什么能把平均 step time 降低约 50%？（P0，18 分钟）

- **直接回答（60–90 秒）**：

  > 这个场景的联合结果是平均 step time 降低约 50%。35B-A3B 不能按 3B Dense 来做显存和性能判断：全部 experts 的参数和 optimizer state 仍要存放，128K 会放大 activation、通信和 logits/loss 张量，路由还会造成动态负载。分析时要同时看这些成本，再决定并行、重算和 kernel 的取舍。
  >
  > 我没有完整的逐项消融，所以不把 50% 分摊给某一个开关。另外有一项能讲清机制的长上下文修复：actor logprob 保留 CP-local logits、只聚合 token 标量，消除了约 7.6GB 冗余分配。但它是否落在这组 50% 的测量窗口，仍需原始配置和日志确认；这两项先分开陈述，不拼成已证实的因果链。

- **追问展开：如何选择优化顺序**：先按实际分片计算参数、activation、logits/loss 的峰值，排除非预期 full materialization；再比较 Attention TP/CP 与 Expert ETP/EP 的映射，避免 expert GEMM 过碎，并检查 expert token histogram；最后按 profile 选择 packing、selective recompute、Grouped GEMM/融合算子、loss chunk 和通信 overlap。这是分析与选型顺序，不表示每项都已在 `-50%` benchmark 中验证过。

- **验证顺序**：先看 peak allocated 与 tensor shape 是否符合 CP/TP/EP 理论；再看 `data wait / attention / expert GEMM / dispatch A2A / TP-CP collective / loss / backward / optimizer`；最后以相同有效 token、长度分布和统计窗口比较平均值与 p95，并验 loss、logprob、expert load、checkpoint/recovery。
- **深入阅读**：[128K MoE 的优化账本与 CP-local logits](../training-infra-roadmap/topics/long_context_training.md#qwen35-35b-a3b-128k)。
- **项目证据或知识边界**：确定事实分别是 Qwen3.5-35B-A3B/128K 的平均 step time 约降低 50%，以及 [RESUME-07](#resume-07) 的代码级 CP-local logits 修复。两者是否属于同一 benchmark、具体 TP/CP/EP、各项贡献和最终绝对 step time，都须以原始配置/日志补齐。
- **高概率追问**：A3B 为什么仍会 OOM？TP 与 EP 怎样避免重复乘 world size？为什么 loss/logprob 会成为 128K 峰值？平均下降 50% 是否掩盖 p99？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：MoE、长上下文和 loss/logprob 路径叠加后，你按什么顺序优化？

- **面试官意图**：检查你能否把 headline 拆成“内存可行性 → 并行效率 → kernel/通信 → 数据供给”的因果链，而不是套用 9B 方案。

- **危险回答**：因为 active 参数只有 3B，所以按 3B dense 估显存；把 50% 全归给 CP 或 chunking；混用 9B 的 31s→9.3s；不给 workload 与统计窗口。

</details>

↩ [返回本 Part 导航](#part-ii) · ↑ [返回面试速查控制台](#interview-console)

<a id="resume-06"></a>
#### RESUME-06｜128K/256K 长上下文训练的显存主要花在哪里？（P0，18 分钟）

- **直接回答（60–90 秒）**：

  > 我会先按 [INFRA-02](#infra-02) 的 Megatron 显存账本确认模型状态能否放下，再单独计算长序列放大的 activation、logits、loss upcast 和临时 workspace。对 128K/256K，CP 把每个 rank 的 local sequence 降为 `S/CP`，SP 在 TP 区域还能继续去掉部分重复 activation；PP 只减少本 stage 的层数，activation 峰值仍取决于同时在途的 microbatch，不能机械除以 PP。然后确认 FlashAttention、THD/packing、CP 和 fused cross entropy 的真实 tensor shape，避免配置写了但实际回退。最后才比较 selective/full recompute、CP、TP 和 offload：TP 解决权重和大 GEMM，但过大会让 GEMM 变碎；CP 更直接解决长序列 activation，但需要 KV 通信；offload 则可能把瓶颈转移到 PCIe。最终用真实长度分布验证峰值显存、loss、吞吐、checkpoint 和恢复，而不是只跑一个 max-length step。

- **项目证据或知识边界**：简历中的 35B-MoE 256K、27B 128K/256K checkpoint 交付，指训练框架和 recipe 已达到稳定训练验收，可支持算法团队继续实验并产出经下游验证的有效模型权重，不只是 smoke test。回答时准备代表性长度分布、连续训练窗口、loss/grad、save/resume、下游质量和 recipe 复现证据；同时不要外推为无限期、无人值守长稳。
- **高概率追问**：activation 为何近似随 sequence length 增长？attention memory 是否仍是二次？CP 和 Ulysses SP 区别？offload 为什么可能严重拖慢？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：你如何设计长上下文 SFT 的并行和显存方案？

- **面试官意图**：检查是否真正做过长序列训练，能否从张量维度做 memory accounting。

- **危险回答**：只说“开 FlashAttention 和重计算”；默认 max length 就是平均 workload；用更多 TP 机械解决所有 OOM。

</details>

↩ [返回本 Part 导航](#part-ii) · ↑ [返回面试速查控制台](#interview-console)

<a id="resume-07"></a>
#### RESUME-07｜CP chunking 静默失效为什么会分配 7.6GB 冗余 logits？（P0，18 分钟）

- **直接回答（60–90 秒）**：

  > 问题在 chunk 之前：THD packed、CP 大于 1 的 actor 路径，原本已经输出本地 logits，但 postprocess 又沿 CP 做了一次 full-sequence all-gather。后面再按 chunk 算 logprob，也释放不了前面已经分配的完整 logits，所以没有报错，却多出约 7.6GB 冗余分配。
  >
  > 修复是保留 CP-local logits，把 labels 和 mask 按相同规则切到本地，算出每个 token 的 logprob/entropy 后只 gather 标量。关键验收不只是 OOM 消失，还要确认 token 顺序、logprob、loss 和梯度对齐。7.6GB 是这次记录的结果，不是所有模型都会出现的固定值。

- **代码路径展开**：actor 的 THD+CP 路径设置 `gather_thd_outputs=False`，保留 `[T/CP,V/TP]` logits；labels 和 loss mask 按相同 causal zigzag 规则切分，在 TP vocab shard 上计算 selected-token logprob/entropy，再沿 CP gather `[T/CP]` 标量、恢复 packed 顺序并 unpad。Critic 保留原有已验证的 full-gather 路径。显式 `_pcp_output_layout` 标记还避免了只凭 `cu_seqlens` 把 BSHD 误判成 THD。

![CP-local logits 修复：保留分片 logits，只聚合标量](../training-infra-roadmap/assets/topics/cp-local-logits.svg)

- **显存公式**：旧路径主张量近似 `T × (V/TP) × dtype_bytes × live_copies`；新路径是 `(T/CP) × (V/TP) × dtype_bytes` 加上可忽略得多的 `[T]` scalar gather。7.6GB 是最新简历确认的冗余分配结果，不在缺少原始 shape 日志时倒推出唯一 `T/V/dtype/live_copies` 组合。
- **如何证明**：在 all-gather 前后记录 per-rank shape/stride/dtype 与 `max_memory_allocated`；确认 labels、mask、logits 的 zigzag 对齐；用 CP=1 参考和 CP>1 修复版比较 token logprob、entropy、loss、grad 与多 rank checksum；同时跑 train 和 forward-only/compute-logp 路径，并验证 BSHD/THD、padding/unpadding 与 critic 回归。
- **代码证据**：本地项目提交 `be6fb98f`；核心接口是 `gather_thd_outputs=False`、`split_packed_labels_for_thd_cp`、`gather_packed_scalar_from_thd_cp` 与 `_pcp_output_layout`。
- **深入阅读**：[CP-local logits、chunked logprob 与排障流程](../training-infra-roadmap/topics/long_context_training.md#cp-local-logits)。
- **项目证据或知识边界**：可以把“为什么 chunk 无法挽救 full logits gather”和“只聚合标量”作为直接源码证据；没有公开原始 shape 记录时不虚构 7.6GB 的精确拆解。
- **高概率追问**：为什么 sequence chunking 救不了已分配的 full logits？TP vocab shard 如何计算 exact logprob？zigzag split 为什么取头尾两段？为什么 critic 不复用 actor 路径？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：没有报错但显存异常，你怎么发现并证明 chunking 没生效？

- **面试官意图**：验证源码阅读、张量形状推导和静默正确性/性能问题定位能力。

- **危险回答**：把 bug 说成 chunk size 没传进去；只改 allocator 或 `empty_cache()`；在 CP-local logits 上配 full-sequence labels；只看 OOM 消失，不验 logprob/loss。

</details>

↩ [返回本 Part 导航](#part-ii) · ↑ [返回面试速查控制台](#interview-console)

<a id="kernel-01"></a>
#### KERNEL-01｜NVIDIA 卡上为什么还需要融合算子？常见融合如何接入？（P0，18 分钟）

- **直接回答（60–90 秒）**：

  > GEMM 快，不代表它前后的整条链都快。很多小算子的成本在反复读写 HBM 和 kernel launch；融合可以减少这些开销，例如 bias+activation、residual+norm，MoE 还可以优化 token 重排和多个小 expert 的计算。FlashAttention 则进一步通过分块和 online softmax，避免写出完整 attention matrix。
  >
  > 接入时我优先使用 Megatron-Core、Transformer Engine 等已有配置和 layer spec，然后看 profiler 是否真的命中目标 kernel，并对照 loss、梯度、显存和吞吐。必要的模块替换还要保持权重布局和 checkpoint 兼容。我的经验是接入与调优，不是编写底层 CUDA kernel；融合也可能因额外布局转换或资源压力而变慢。

- **速查分类**：Attention fusion 主要减少 `QKᵀ → scale/mask → softmax → dropout → PV` 的 HBM 往返；elementwise/norm fusion 合并短小链路；training fusion 合并多 tensor 更新或梯度累加；MoE fusion 把多个小 expert GEMM 聚合并减少 token 搬运。
- **常见接入点**：QKV/RoPE、scaled masked softmax、bias+GeLU/SwiGLU、bias+dropout+residual、residual+RMSNorm、gradient accumulation、multi-tensor optimizer、vocab-parallel CE，以及 MoE router/permute/unpermute/Grouped GEMM。Grouped GEMM 与 shared-expert overlap 是同一优化链上的配套能力，不都等同于 elementwise fusion。具体启用项以 layer spec、配置和 kernel trace 为准。
- **何时可能负优化**：shape 太小/太怪触发 fallback；为 fusion 做额外 layout conversion；register/shared-memory 压力降低 occupancy；graph/compiler 频繁重编译；数值精度或 dropout RNG 语义不一致。
- **深入阅读**：[Transformer Engine 与 NVIDIA 融合算子工程清单](../training-infra-roadmap/topics/transformer_engine.md#fusion-map)。
- **项目证据或知识边界**：项目可以讲 Grouped MatMul、融合算子接入与性能验证；不声称自己编写了底层 CUDA kernel。具体启用了哪些开关，以项目配置和 profiler kernel name 为准。
- **高概率追问**：FlashAttention 与普通 elementwise fusion 的本质差异？fused cross entropy 如何避免 full-vocab logits？为什么开了 fused flag 可能没生效？如何做数值验收？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：GEMM 已经由 cuBLAS/Transformer Engine 优化，为什么还要 fusion？是开参数还是替换接口？

- **面试官意图**：检查你能否区分 compute-bound GEMM 与 memory/launch-bound 小算子，并能把融合落到框架接入、兼容和验证。

- **危险回答**：“融合减少计算量，所以一定更快”；只列名词；把接口替换等同于 kernel 命中；不提供 unfused fallback 和精度对照。

</details>

↩ [返回本 Part 导航](#part-ii) · ↑ [返回面试速查控制台](#interview-console)

<a id="resume-10"></a>
#### RESUME-10｜你在千卡/万卡级交付里具体负责什么？（P0，15 分钟）

- **直接回答（60–90 秒）**：

  > 我负责的是模型侧交付，不是整个万卡平台。TX、X1 所在集群总规模分别约 1.4 万卡和 1.2 万卡；我的直接训练证据是 X1 200B MoE 的 3K 卡连续稳定训练两个月，这两个规模口径要分开。
  >
  > 具体工作是让客户模型在国产卡上跑通、对齐精度，再通过 profiling 定位瓶颈，调整并行配置、接入融合算子、优化通信暴露，持续回归到性能验收。如果根因在编译器、算子库、集合通信或硬件，我提供复现和证据，推动对应团队解决，最后负责模型侧验证。我的 ownership 是这条交付闭环，不能把底层团队实现也归成我个人开发。

- **六步展开版**：

  1. **固定验收口径**：模型版本、global/micro batch、sequence length、precision、卡数、warmup、统计窗口、精度阈值和性能目标。
  2. **完成模型跑通**：处理算子兼容、分布式并行、checkpoint/data 和精度链路，先建立小规模可复现基线。
  3. **采集性能证据**：记录 step time、吞吐、MFU/硬件利用、算子耗时、collective exposed time、pipeline idle 和显存峰值。
  4. **识别当前主瓶颈**：区分并行切分、kernel/小 GEMM、通信暴露、显存与重计算、Host/data 和规模化 straggler。
  5. **最小变量验证**：调整并行策略、融合算子或 overlap 时，保持 workload 不变，验证性能、loss、精度与稳定性。
  6. **重新 profile 并继续迭代**：不能把单机收益线性外推到千卡规模；collective、拓扑和 straggler 会随规模放大，必须在目标规模重新验收。

- **项目证据或知识边界**：X1 200B MoE 模型可以交叉引用 [RESUME-01A](#resume-01a) 的 `0.16x→0.95x`、MFU 35% 和 3K 卡训练证据；TX 可说文生视频、文生图和 389B MoE 的迁移/功能/性能工作，并用最新简历的“开局性能提升 30%–50%、10+ 模型交付、80+ 生产问题、协同 4–5 人”作为项目总结果，但要能区分个人直接动作和团队总成果。如何把集群总规模与直接训练规模分开，见 [INFRA-09](#infra-09)。客户继续使用代号。
- **高概率追问**：你亲自改了什么、推动了什么？性能 benchmark 如何固定？讲一次“优化后瓶颈迁移”的完整迭代？为什么单机收益扩到千卡可能消失？X1 和 TX 中你的职责是否完全相同？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：你说参与过千卡/万卡级交付，个人具体负责哪一段？请不要只讲团队整体做了什么。

- **面试官意图**：确认“千卡/万卡”是项目背景还是你承担了可验证职责；检查你能否独立完成模型从跑通到性能验收的闭环，并区分个人、框架及底层团队贡献。

- **危险回答**：把整个万卡平台、硬件运维和稳定性体系说成个人 ownership；只说“协调资源、推动闭环”而没有 profiling 和 A/B；把底层团队实现的算子或通信优化说成自己开发；泄露客户和集群敏感信息。

</details>

↩ [返回本 Part 导航](#part-ii) · ↑ [返回面试速查控制台](#interview-console)

<a id="megatron-02"></a>
#### MEGATRON-02｜Column Parallel 和 Row Parallel Linear 怎么切？通信在哪里？（P0，18 分钟）

- **直接回答（60 秒）**：

  > TP 切的是一个 layer 内部的 hidden/output channel 或 attention head，不是 sequence。对 `Y=XW`，Column Parallel 沿 `W` 的输出维切，每个 rank 产生一部分输出特征；Row Parallel 沿输入维切，每个 rank 产生同 shape 的 partial output，再做 reduce-sum。Megatron 把 MLP 的 gate/up 做 Column Parallel、down 做 Row Parallel；Attention 的 QKV projection 做 Column Parallel，把 heads 分给各 TP rank，output projection 再做 Row Parallel。这样中间张量一直保持分片，只在必要边界通信，而不是每个 Linear 后 all-gather。

- **用 shape 展开 MLP**：令输入 `X:[N,H]`，FFN intermediate size 为 `I`，TP size 为 `t`。以下按数学权重 `W:[in,out]` 表示，代码中的存储转置不改变切分语义。

  ```text
  Gate/Up（Column Parallel）:
      W_gate^r, W_up^r : [H, I/t]
      G^r, U^r         : [N, I/t]
      Z^r = SiLU(G^r) * U^r
      forward 无需合并；每个 rank 本地完成激活

  Down（Row Parallel）:
      W_down^r : [I/t, H]
      Y_partial^r = Z^r @ W_down^r : [N, H]
      Y = sum_r(Y_partial^r)
  ```

  forward/backward 的完整通信要分是否开启 SP：

  | TP Linear | 无 SP | 有 SP |
  |---|---|---|
  | Column Parallel forward | 无 TP collective，输入在 TP ranks 复制 | `AllGather(X)`，把 sequence shard 临时拼回后再做 Column GEMM |
  | Column Parallel backward | `AllReduce(dX)`，合并各输出 shard 对输入的 partial gradient | `ReduceScatter(dX)`，规约后仍按 sequence 分片 |
  | Row Parallel forward | `AllReduce(Y_partial)`，每个 rank 得到完整 `Y` | `ReduceScatter(Y_partial)`，规约并沿 sequence 分片 |
  | Row Parallel backward | 无 TP collective，本地得到 intermediate shard 的 `dZ` | `AllGather(dY)` 还原 Row Linear 所需输入，再本地得到分片 `dZ` |

  因此，无 SP 的经典简写是 “Column forward 不通信、backward AllReduce；Row forward AllReduce、backward 不通信”；有 SP 则是 “Column forward AllGather、backward ReduceScatter；Row forward ReduceScatter、backward AllGather”。

- **用 shape 展开 Attention**：MHA 有 `n_h` 个 query heads、每头维度 `d_h`，`H=n_h×d_h`。QKV 的 Column Parallel 让每个 rank 持有 `n_h/t` 个 heads：

  ```text
  QKV projection:  [N,H] -> Q^r/K^r/V^r:[N,n_h/t,d_h]
  local attention: 只计算本 rank 的 heads
  context^r:       [N,H/t]
  output projection（Row Parallel）:
      context^r @ W_o^r -> [N,H] partial -> ReduceScatter / AllReduce
  ```

  GQA/MQA 还要检查 `num_query_heads`、`num_kv_heads` 与 TP 的可整除/复制规则：当 KV heads 少于 TP size 时，部分实现会复制 KV head，而不能机械写成 `n_kv/t`。TP 切 head/hidden；CP 才切 `S`，并让本地 Q 通过 P2P/ring/all-gather/all-to-all 访问跨 rank KV。完整 SP/CP 区别见 [MEGATRON-04](#megatron-04)。

- **项目证据或知识边界**：这是框架机制题；简历只有使用/调优证据，无需假装亲自实现 TP layer。
- **高概率追问**：QKV projection 如何切 head？为什么 TP 要求 hidden/head 数可整除？sequence parallel 如何改变通信？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：以 MLP 或 Attention projection 说明 forward/backward collective。

- **面试官意图**：判断 TP 是否停留在“把模型切到多卡”的表层。

- **危险回答**：只说“按行/按列平均切”；混淆权重矩阵的逻辑维度与代码存储布局；说 TP 没有通信。

</details>

↩ [返回本 Part 导航](#part-ii) · ↑ [返回面试速查控制台](#interview-console)

<a id="megatron-03"></a>
#### MEGATRON-03｜为什么 TP 从 2 增到 4 可能更慢？（P0，15 分钟）

- **直接回答（60 秒）**：

  > TP 增大能减少每卡权重和部分 activation，但本地 GEMM 也会变窄，计算效率可能下降；逐层 collective 的相对成本则可能上升。这里不是说 TP 从 2 变 4 就让每层 collective 调用次数翻倍，而是参与 rank、通信算法、消息量和计算通信比例变了。
  >
  > 对 9B 长上下文，若参数已经放得下，可以比较更小 TP、更大 CP：让 CP 分摊长序列 activation，避免 TP 过度切分矩阵。我的另一组 workload 比较过 `TP=2,CP=8` 与 `TP=4,CP=4`；判断依据要同时看 GEMM、TP/CP 通信和峰值显存，不能把一次结果推广到所有模型。

- **次数与成本不要混淆**：固定模型、SP 开关和 microbatch/schedule 后，TP degree 不自动增加模型图中的 collective 调用点。经典无 SP Transformer layer 前后向共四个 TP AllReduce；开启 SP 后通信形式改变，完整对应见 [TP Linear 通信表](#megatron-02)。底层算法的通信轮数和耗时仍可随 group size 改变。[Megatron 原论文 §3](https://arxiv.org/html/1909.08053v4)

- **项目证据或知识边界**：项目底稿记录过约 `163s→102s` 的相关对比；该数字不在当前简历，使用前确认 workload 与披露范围。
- **高概率追问**：为什么 TP 通常放 NVLink 域内？如果 TP=2 放不下怎么办？怎样用 Nsight/NCCL trace 证明？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：结合你的 9B 长上下文项目解释 TP 负优化。

- **面试官意图**：评估工程取舍和性能模型，而非配置记忆。

- **危险回答**：“TP 越大通信越大”但说不出通信频率和张量；忽略 batch/GEMM shape；把一次结果普适化。

</details>

↩ [返回本 Part 导航](#part-ii) · ↑ [返回面试速查控制台](#interview-console)

<a id="megatron-04"></a>
#### MEGATRON-04｜Sequence Parallel 和 Context Parallel 有什么区别？（P0，15 分钟）

- **直接回答（60–90 秒）**：

  > 两者在 tensor shape 上都沿 sequence dimension 切，但系统含义不同。Megatron 的 SP 复用 TP process group，只分片 TP 区域之间原本重复的 LayerNorm、Dropout、Residual 等 activation，并用 reduce-scatter/all-gather 替代部分 TP all-reduce；它不增加 world-size，也没有把 Attention 的完整上下文独立分布。CP 则是独立 mesh 轴，从网络输入开始持久切分 sequence 和几乎全部 activation，每个 CP rank 只持有 `S/CP` token；Attention 中本地 Q 要通过 P2P/ring/all-gather/all-to-all 等方式访问全局 KV。因此 SP 是 TP 配套的 activation 去重，CP 是长上下文的独立并行。

  | 对比项 | SP | CP |
  |---|---|---|
  | process group | 复用 TP group | 独立 CP group |
  | 切分范围 | LayerNorm、Dropout、Residual 等非 TP 区域的重复 activation | 输入和几乎全部 activation |
  | Attention 语义 | 不独立分布完整 context | 本地 Q 通过 KV 通信访问全局 context |
  | world-size | 不增加 | 乘入 world-size |
  | 主要目标 | 减少 TP rank 的 activation 冗余 | 扩展长上下文显存与计算 |

  Megatron Core 的 `sequence_parallel` 并不是字面意义上的默认开启，但官方建议 TP 时启用，并要求 TP 与 EP 同时使用时启用。TP、CP、SP 同开时，CP 先把语义 context 切为 `S/CP`；在 SP 覆盖的区域，activation 还可沿 TP group 形成近似 `S/(CP×TP)` 的本地分片，但 Attention 的全局上下文仍由 CP 通信保证。

- **深入阅读**：[5D 总览中的 SP/CP 对比](../training-infra-roadmap/topics/distributed_training.md#sp-vs-cp)、[Sequence Parallelism](../training-infra-roadmap/topics/sequence_parallelism.md)、[Context Parallelism](../training-infra-roadmap/topics/context_parallelism.md)。
- **项目证据或知识边界**：你有 CP/THD/packed 配置经验；底层通信算法若未改过，应定位为使用与诊断。
- **高概率追问**：为什么 SP 不进入 world-size？为什么 TP+EP 要启用 SP？CP 为什么能替代一部分 full recompute？GQA/MQA 下 KV 通信怎样变化？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：它们都切 sequence，为什么不是同一件事？

- **面试官意图**：这是 Megatron 高频辨析题，能快速筛掉只背 5D 名词的人。

- **危险回答**：“SP 切短序列，CP 切长序列”；把 `SP×CP` 都乘进 world-size；认为 SP 会持久分片全部 attention activation；忽略 CP 的 KV 通信。

</details>

↩ [返回本 Part 导航](#part-ii) · ↑ [返回面试速查控制台](#interview-console)

<a id="megatron-05"></a>
#### MEGATRON-05｜Megatron Distributed Optimizer 与 ZeRO-1/2/3 怎么对应？（P0，15 分钟）

- **直接回答（60–90 秒）**：

  > 经典 Megatron distributed optimizer 主要分片 optimizer state 和 FP32 main parameters，梯度通过 reduce-scatter 让各 rank 得到自己负责的 shard，更新后再 all-gather 参数视图，思想接近 ZeRO-1，并通过 contiguous param/grad buffer 提高通信效率。开启 CP 时不能把 shard group 简化成纯 DP：Dense 参数默认使用 `DP×CP` 的 `dp_cp` group，Expert 参数使用 EDP group。现代 Megatron-FSDP 又可配置 `optim`、`optim_grads`、`optim_grads_params`，分别对应 ZeRO-1/2/3 式分片。显存不能只背 `16/d`，完整 dtype 表和每-rank 算法见 [INFRA-02](#infra-02)。

- **延伸阅读**：[DP 策略、PyTorch DP/DDP/FSDP、Megatron DP group 与通信算子](../training-infra-roadmap/topics/data_parallelism.md#dp-concept-and-implementations)。

- **项目证据或知识边界**：你做过 distributed checkpoint 和 optimizer 相关故障；若没改 optimizer 核心，明确为集成/排障经验。
- **高概率追问**：DP=1 时还有什么冗余 buffer？overlap grad reduce 如何实现？ZeRO-3 与 TP/PP 怎么组合？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：它分片了什么、每步有哪些通信、能省多少显存？

- **面试官意图**：验证 model-state memory accounting 和 DP 通信理解。

- **危险回答**：把 Megatron distributed optimizer 直接等同 ZeRO-3；忽略 main param 和 dtype；认为分片没有通信成本。

</details>

↩ [返回本 Part 导航](#part-ii) · ↑ [返回面试速查控制台](#interview-console)

<a id="moe-01"></a>
#### MOE-01｜Dense 和 MoE 的主要区别是什么？expert 如何路由，是大专家还是小专家，有多少专家，是否有 shared expert？（P0，15 分钟）

- **直接回答（60–90 秒）**：

  > Dense 模型中，每个 token 都经过同一套 FFN 参数，执行规则、GEMM 形状和负载比较稳定；MoE 把 FFN 换成多个 expert，由 router 为每个 token 选择 top-k expert，因此总参数可以很大，但单 token 只激活少量参数。代价是多出 router、token 重排、all-to-all、Grouped GEMM、负载均衡和更复杂的 checkpoint/并行映射。总专家数 `E` 和每个 token 激活的 `top-k` 是两个概念；所谓大专家或小专家主要看单个 expert 的 FFN intermediate size。更多、更窄的 expert 能细化专业化，但也更容易产生小 GEMM、通信和负载不均。shared expert 是每个 token 都会经过的公共 FFN，用来承载共性能力，routed experts 再负责专业化；它不是所有 MoE 都必有的结构。

- **Router 的最短数据流**：`hidden states → router score → top-k expert IDs/weights → token dispatch/permute → expert FFN → weighted combine → restore token order`。常见 token-choice routing 是每个 token 选 expert；capacity、dropless、aux loss 或 bias-based balance 决定过载如何处理，但具体机制必须以模型配置为准。
- **先问清这四个量**：总专家数 `E`、每 token 的 `top-k`、单 expert 的 `FFN intermediate size`、shared expert 数量。再补 router/balance、capacity/dropless、EP/ETP/EDP 和物理拓扑，才能判断 activated parameters、GEMM 粒度和通信量。
- **深入阅读**：[Dense 与 MoE：结构、路由、专家粒度和 shared expert](../training-infra-roadmap/topics/moe.md#dense-vs-moe)；继续追问系统代价时进入 [MEGATRON-06](#megatron-06)。
- **项目证据或知识边界**：项目可确认的是 X1 200B MoE 模型的适配与性能优化；当前材料没有已核验、可公开的 `E / top-k / expert FFN intermediate size / shared expert` 配置，面试前按证据卡补齐，不从相似模型猜。
- **高概率追问**：total parameters 与 activated parameters 怎么算？top-1/top-2 的效果和成本？fine-grained expert 为什么可能更难跑快？shared expert 是否参与 EP？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：Dense 与 MoE 在结构、计算和系统代价上有什么区别？拿到一个 MoE 配置时先看哪些字段？

- **面试官意图**：检查你是否真正理解 MoE 的模型结构和配置坐标，而不是一上来只讲 EP/all-to-all。

- **危险回答**：“MoE 参数大但计算量不变，所以一定比 Dense 快”；把 `E` 当 `top-k`；认为专家越多越好；把 shared expert 说成所有 MoE 标配。

</details>

↩ [返回本 Part 导航](#part-ii) · ↑ [返回面试速查控制台](#interview-console)

<a id="megatron-06"></a>
#### MEGATRON-06｜MoE 为什么需要 EP？all-to-all 为什么难优化？（P0，18 分钟）

- **直接回答（60 秒）**：

  > Router 为每个 token 选择 top-k expert；EP 把 experts 放到不同 rank，token 先按目的 expert 做 permute/dispatch 和 all-to-all，到本地 grouped GEMM 计算，再 all-to-all combine 并恢复顺序。难点是 token 路由动态、每个 rank 发送量不均，热点 expert 会让最快 rank 等最慢 rank；小 expert batch 还会降低 GEMM 效率。优化要联合看 expert load、capacity/dropped token、A2A p95、permutation、grouped GEMM、expert placement 和网络拓扑。TP+EP 组合时官方要求启用 sequence parallel，避免相关 activation 复制和布局问题。

- **项目证据或知识边界**：你有 Qwen3/Qwen3.5 MoE recipe 和华为大 MoE 优化经验；准备一个具体的 expert imbalance 或 A2A 案例。
- **高概率追问**：top-1 与 top-2 的代价？capacity factor 如何影响效果和性能？EP 跨节点怎么放？MoE checkpoint 如何 reshuffle？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：请从 router、dispatch、expert compute、combine 讲一层 MoE。

- **面试官意图**：验证简历中 dense/MoE 经验，以及动态通信和负载均衡能力。

- **危险回答**：“MoE 每 token 只算少数 expert，所以一定更快”；只谈参数量，不谈动态通信和负载尾部。

</details>

↩ [返回本 Part 导航](#part-ii) · ↑ [返回面试速查控制台](#interview-console)

<a id="infra-01"></a>
#### INFRA-01｜MFU 是什么？如何正确计算和使用？（P0，15 分钟）

- **直接回答（60–90 秒）**：

  > MFU 衡量的是模型所需计算占硬件理论算力的比例，可以写成“每 step 的模型 FLOPs，除以 step 时间和总峰值算力”。标准 MFU 只计必要的 forward/backward，不把额外 recompute 算成有效模型工作；把重算也计进去，更接近 HFU 的口径。
  >
  > 我会先核对模型结构、实际计算长度、dtype 和计时范围。SFT 中 prompt 不进 loss，不代表它没有计算成本；模型处理的 token 与直接监督的 token 要分开报。同 workload、同公式、同计时下，MFU 应随吞吐一起变化；如果看起来不一致，先查统计口径。训练段 MFU 也不能代替 rollout 等待、数据质量和端到端成本。

- **公式与三个统计边界**：

  ```text
  MFU = required_model_FLOPs_per_step / (step_time × aggregate_peak_FLOPs)
  ```

  1. **模型工作量**：按 Dense/MoE、attention 结构与长度分布估必要 forward/backward，排除额外 rematerialization；长序列不能无条件只用参数量近似。标准定义见 [PaLM §4.1](https://arxiv.org/html/2204.02311v5)。
  2. **Token 口径**：区分实际模型计算位置、non-padding input token、loss-mask-selected token。Prompt 通常仍参与 forward，并通过上下文影响 response 的梯度；不能直接把监督 token 数代入全模型 FLOPs 公式。Padding 是否计入，要看 estimator 和 kernel 实际执行路径；另报 padding ratio 与监督 token 吞吐。
  3. **硬件与时间**：峰值匹配硬件、dtype 及 dense/sparse Tensor Core 口径；明确 step 时间是否包含 data wait、optimizer、checkpoint，以及是否只统计训练阶段。更换这些边界后，MFU 不能直接横比。

- **项目证据或知识边界**：最新版简历数字是 SFT `23%→45.2%`；必须准备 MFU estimator、有效 token 和统计窗口，并能解释它为何不能直接由 `31s→9.3s` 反推。
- **高概率追问**：MoE FLOPs 按 total parameters 还是 activated parameters？recompute FLOPs 是否计入 numerator？为什么 achieved TFLOPs 和 MFU 不等价？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：为什么 MFU 提升不一定代表用户吞吐提升？

- **面试官意图**：验证性能指标基本功和对“指标游戏”的警惕。

- **危险回答**：MFU=GPU utilization；使用不同 FLOPs 公式横比；只报百分比不报 throughput。

</details>

↩ [返回本 Part 导航](#part-ii) · ↑ [返回面试速查控制台](#interview-console)

<a id="p2-01"></a>
<a id="dist-01"></a>
#### DIST-01｜FSDP/FSDP2 与 ZeRO-1/2/3 有什么区别和联系？（P0，15 分钟）

- **直接回答（60–90 秒）**：

  > ZeRO 是按 DP 维消除 model-state 冗余的方法族：Stage 1 分 optimizer state，Stage 2 再分 gradient，Stage 3 连 parameter 也分。PyTorch FSDP 的 `FULL_SHARD` 在“分片哪些状态”上接近 ZeRO-3，但不是同一套实现；forward/backward 前按模块 all-gather 参数，backward 后 reduce-scatter gradient，并按 reshard policy 释放完整参数。FSDP1 以 wrapper/FlatParameter 为核心；FSDP2 使用 `fully_shard` 和 per-parameter DTensor，FQN、状态管理和 composability 更自然。`SHARD_GRAD_OP` 只能粗略类比 ZeRO-2，因为参数驻留和 reshard 语义并不完全相同。它们与 TP 不互斥：FSDP/ZeRO 沿 data-parallel replica 分状态，TP 则直接改变层内 GEMM 和 activation 的计算图；大模型训练经常组合使用。

- **`FULL_SHARD` 一层在一个 step 内怎么走**：

  ```text
  steady state: 每个 DP rank 只持有本层 parameter shard
      -> pre-forward Parameter AllGather，临时 materialize 完整参数
      -> forward compute
      -> 按 reshard_after_forward 策略释放/保留完整参数
      -> pre-backward 再次 Parameter AllGather（若此前已 reshard）
      -> backward compute
      -> Gradient ReduceScatter，每个 rank 只留下本地 gradient shard
      -> local optimizer 用本地 parameter/gradient/optimizer-state shard 更新
  ```

  FSDP 的 prefetch 是让下一层 parameter AllGather 与当前层计算重叠，不是消灭通信；prefetch 太激进会同时 materialize 多层参数，反而推高峰值显存。`reshard_after_forward=False` 能减少 backward 前的第二次 AllGather，但用参数驻留换通信，语义更接近 ZeRO-2 式取舍，仍要按具体 API/版本说明。

- **FSDP1 与 FSDP2 的执行骨架**：二者都有“按模块 gather 参数—计算—reshard—reduce-scatter 梯度”的核心生命周期。FSDP1 通常由 wrapper 把参数展平为 `FlatParameter` 后切 shard；FSDP2 的 `fully_shard` 在原参数上使用 DTensor 分片，并用 module hooks 组织通信，因而保留 per-parameter FQN、组合其他 parallelism 和 checkpoint 更自然。通用 collective 的输入输出语义见 [INFRA-04](#infra-04)；raw TP/PP/CP/EP shard 的 checksum 不能直接要求相等，排障口径见 [TRAIN-ANOMALY-01](#train-anomaly-01)。

- **现场画账**：先写 `P/G/O` 三类 model state：ZeRO-1=`O`，ZeRO-2=`O+G`，ZeRO-3/FSDP FULL_SHARD=`O+G+P`；再补 activation、通信 buffer 和 workspace，避免说成“总显存除以 DP”。
- **深入阅读**：[FSDP/FSDP2、ZeRO 与 Megatron 训练后端选型](../training-infra-roadmap/topics/fsdp.md#fsdp-zero-map)。
- **项目证据或知识边界**：你的主项目以 Megatron-Core 后端为主，对 FSDP/FSDP2 的口径是机制理解、框架选型与集成判断；不声称实现过 FSDP 核心 sharding/hooks。
- **高概率追问**：FSDP2 为什么不用 FlatParameter？`FULL_SHARD` 每个阶段有哪些 collective？`SHARD_GRAD_OP` 为什么不能严格等同 ZeRO-2？FSDP 与 TP 能否组合？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：FSDP、FSDP2 和 ZeRO 都在“分片”，它们分别分什么？与 TP 有何本质区别？

- **面试官意图**：检查你能否从参数、梯度、优化器状态和运行时通信解释 DP state sharding，而不是只做名词映射。

- **危险回答**：FSDP 就是 TP；ZeRO-3 没有 all-gather；把 Stage 1/2/3 说反；认为状态分片一定更快。

</details>

↩ [返回本 Part 导航](#part-ii) · ↑ [返回面试速查控制台](#interview-console)

<a id="megatron-11"></a>
#### MEGATRON-11｜Megatron、FSDP/FSDP2、DeepSpeed 与 Accelerate 如何分层和选型？（P0，15 分钟）

- **直接回答（30 秒）**：

  > Accelerate 更像 Hugging Face 上层的启动与分布式编排 facade，可以通过 plugin 接 FSDP 或 DeepSpeed；FSDP/FSDP2 是 PyTorch-native 的 DP state sharding；DeepSpeed 是包含 ZeRO、CPU/NVMe offload、pipeline 等能力的训练 runtime；Megatron-Core 的优势是 TP/PP/CP/EP 多维模型并行、模型实现和高性能 kernel。它们不是简单四选一，先确定需要哪一层能力，再选择经过目标模型和硬件验证的组合。

- **追问展开：如何选型**：

  > 我不会按“哪个框架更先进”选，而是先看单层和全模型能否放下、是否必须 TP/PP，再看长序列和 MoE 是否需要 CP/EP，然后评估 offload、拓扑、融合 kernel、checkpoint/权重转换、模型接入速度和团队已有资产。Hugging Face 模型快速适配、中等规模且主要需要 DP 分片时，Accelerate+FSDP2 通常更自然；已有 ZeRO/offload 资产或 CPU/NVMe 分层需求时会重点评估 DeepSpeed；超大 Dense/MoE、长上下文且必须联合 TP/PP/CP/EP 时更倾向 Megatron。最终要在固定 workload 下比较 effective tokens/s、峰值显存、scale efficiency、恢复时间和维护成本，而不是仅看能否启动。

- **项目口径**：选择 Megatron 不是因为其他框架“不行”，而是项目需要 MoE/长上下文多维并行，并且已有 SFT/RLVR、MBridge、checkpoint 和权重同步资产更贴合。你的直接生产证据在 Megatron-Core 的特性使用、集成和调优；DeepSpeed、Accelerate、FSDP/FSDP2 只按机制理解与选型判断回答，不声称修改过底层 sharding、hook 或 runtime。
- **深入阅读**：[训练后端决策树与显存账](../training-infra-roadmap/topics/fsdp.md#backend-selection) · [Hugging Face Accelerate：FSDP 与 DeepSpeed](https://huggingface.co/docs/accelerate/concept_guides/fsdp_and_deepspeed)。
- **高概率追问**：Accelerate 自己是否实现了 ZeRO？30B Dense、8 卡怎么选？200B MoE 呢？FSDP2+TP 的代价是什么？为什么团队熟悉度是技术指标？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：这四者分别处在哪一层，是互斥的四选一吗？什么场景选择哪种组合？

- **面试官意图**：评估你能否区分上层编排、DP state sharding 和模型并行，并把模型规模、生态成熟度与团队成本转成架构决策。

- **危险回答**：把 Accelerate 和 Megatron 当作同一层的四选一；说 Accelerate 自己实现 ZeRO；“小模型 FSDP、大模型 Megatron”一句话结束；只看能否 OOM。

</details>

↩ [返回本 Part 导航](#part-ii) · ↑ [返回面试速查控制台](#interview-console)

<a id="sft-data-01"></a>
#### SFT-DATA-01｜SFT 数据从原始样本到 loss，如何保证没有训错？（P0，15 分钟）

- **直接回答（60–90 秒）**：

  > 我会沿着“样本变成什么 token、哪些位置算 loss、这些样本最终被谁消费”来检查。先确认 chat template、BOS/EOS 和截断策略；再核对 next-token labels 与 loss mask，确保 prompt、padding 和样本边界没有误进监督。Packing 后还要保证 attention 不跨样本，DP 加载和恢复不重复、不遗漏。
  >
  > 验证时拿几条小样本逐 token 展示角色、label 和 mask，手算一段 CE，再比较 packing 前后、单卡与多卡、save/resume 的结果。Loss 下降不等于训对了，还需要 held-out eval 和回归集。我的直接经验在数据接口、加载性能和 mask/packing 验证；质量标注、清洗规则和算法评测要区分团队职责。

- **完整链路速查**：数据格式/质量/去重与污染检查 → schema/chat template → tokenizer/BOS/EOS/长度策略 → input/position/attention/loss-mask contract → packing/bucket/shuffle/DP sampler → 数值对照与 held-out eval。特别核对 `logits[t]` 是否预测正确的下一个 token，以及 label shift 由数据层还是模型 loss 实现，避免重复 shift。

- **最容易训错的四处**：chat template 与 rollout/inference 不一致；prompt 或 padding token 误进 loss；packing 后 `cu_seqlens`、position reset 或 segment boundary 错；数据并行 sampler/recovery 导致样本重复、遗漏或顺序漂移。
- **性能与正确性一起看**：DataLoader worker、prefetch、缓存和 pinned memory 解决 host/data wait；packing/bucketing 提高有效 token 比。但每次优化都要同时报告 data wait p95/p99、effective tokens/s、padding ratio，并重跑 token/mask 与恢复一致性测试。可回链 [SFT 31s→9.3s](#resume-05) 和 [Packed Sequence](#megatron-08)。
- **项目证据或知识边界**：你有 SFT 数据加载、prefetch、packing/mask 对齐与训练性能闭环经验；数据清洗规则、质量标注或算法评测若由数据/算法团队负责，要明确自己的接口、验证与协作边界。
- **高概率追问**：多轮 SFT 为什么通常只训 assistant token？packing 如何阻止跨样本 attention？恢复后怎样证明不重不漏？perplexity 为什么不足以评估 instruction following？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：请讲清数据清洗、分词、掩码、packing、分布式加载和评估的完整链路。

- **面试官意图**：检查你是否理解训练吞吐背后的 data contract，能否发现“loss 正常但监督位置错误”的静默问题。

- **危险回答**：只说“用 Hugging Face Datasets”；认为 loss 下降就是数据正确；优化 DataLoader 后不复查顺序、mask 和恢复语义。

</details>

↩ [返回本 Part 导航](#part-ii) · ↑ [返回面试速查控制台](#interview-console)

<a id="mllm-01"></a>
#### MLLM-01｜多模态/具身训练与纯 LLM 训练有什么不同？（P0，15 分钟）

- **直接回答（60–90 秒）**：

  > Infra 主链路仍是数据、forward/backward、并行、optimizer 和 checkpoint，但多模态多了媒体处理与跨模态对齐。图像/视频需要 decode、采样、resize 和 augmentation，CPU、存储和 host-to-device 很容易先成为瓶颈；视觉 encoder、projector 与 LLM 可能采用不同冻结策略、dtype 和 optimizer group；分辨率、帧数和文本长度让 visual token 数与 shape 动态变化，容易造成 rank 负载不均、attention activation 膨胀和编译/CUDA Graph shape 爆炸，因此要做 bucket、动态 batching、selective recompute、FlashAttention 以及 Ulysses/CP 等序列切分。正确性上必须守住媒体样本、placeholder/token、时空 position、attention/loss mask 和 label 的一一对应。
  > 具身训练不能直接等同 MLLM：它还增加 observation、action、reward、environment state 和 episode 的时间同步，可能包含连续动作、action chunk 和 simulator/real-world 闭环。评测也不能只看离线 loss，要看任务成功率、轨迹质量、安全约束和闭环回归。

- **项目映射**：TX 阶段的直接证据是文生视频/文生图模型国产卡迁移，以及功能、精度、性能闭环，可用 [HunyuanVideo/Ulysses](#resume-18) 解释视频 token 与通信；自研 verl 支撑 Capek MLLM 后训练说明你理解 multimodal data contract。它们不能升级为机器人真机数据、VLA 或具身算法 ownership。
- **高概率追问**：视频 DataLoader 为什么更容易成为瓶颈？不同帧数如何组 batch？视觉 encoder 冻结后还需要保存什么 checkpoint？具身任务为什么必须做 closed-loop eval？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：从数据 pipeline、模型结构、并行显存、正确性和评测说明新增约束。

- **面试官意图**：判断你能否把已有视频/MLLM Infra 经验迁移到机器人场景，同时守住没有直接做过具身算法训练的事实边界。

- **危险回答**：把多模态训练说成“LLM 多一个 encoder”后结束；把 MLLM 项目直接包装成具身/VLA 训练经验；只谈 GPU，不谈媒体 IO 和样本对齐。

</details>

↩ [返回本 Part 导航](#part-ii) · ↑ [返回面试速查控制台](#interview-console)

### P1 深挖｜面试官继续追问

<a id="resume-18"></a>
#### RESUME-18｜以 HunyuanVideo-14B 为例，640×640×3×129 帧如何用 Ulysses CP 优化？（P1，18 分钟）

- **直接回答（60–90 秒）**：

  > 先区分原始视频和模型 token：`640×640×3×129` 经过 VAE 压缩、patchify 后，才变成 DiT 的长序列，不能直接把原始像素数当 sequence length。这里我用 HunyuanVideo 解释机制，不把这个 shape 和具体配置说成当年的项目实绩。
  >
  > Ulysses 的核心是两次布局转换。每个 rank 先持有部分 sequence、全部 heads；Attention 前用 All-to-All 换成全部 sequence、部分 heads，本地完成 attention；再用 All-to-All 换回 sequence shard。它能分摊长序列 activation，但通信成本和 head 数限制都要考虑。我的国产卡项目经验可用于判断拓扑和性能；实际并行 degree、VAE stride 和分项收益必须看真实配置。

- **追问展开：拓扑与配套优化**：优先评估将 Ulysses All-to-All 放在 HCCS 等机内高速域；更大规模可考虑 Ulysses+Ring 的层级组合，但 Ring 也在 attention 层通信，不能称为“低频通信”。是否适合跨 RoCE/IB，要看消息量、带宽和 overlap。其他候选包括冻结 text encoder/VAE 后使用 no-grad、缓存或阶段化运行，BF16/FP16、FlashAttention 与 fusion、selective recompute、按分辨率/帧数 bucket。最后按 attention、MLP、All-to-All、VAE/text/data 和 step p95 验证，不把候选清单当作已落地配置。

![视频 DiT 的 Ulysses sequence parallel 数据流](../training-infra-roadmap/assets/topics/ulysses-video-cp.svg)

- **配置约束**：`world_size = ulysses_degree × ring_degree` 只描述这组序列并行网格，不应与外层 DP/PP 重复计算；Ulysses degree 通常受 attention heads/KV heads 以及 kernel layout 约束。官方 HunyuanVideo 公开的是 xDiT/USP 推理示例，项目训练方案只能作为机制参照，不能拿官方推理数字冒充训练收益。
- **深入阅读**：[视频 DiT：从原始帧到 Ulysses/Ring 并行](../training-infra-roadmap/topics/long_context_training.md#video-dit-ulysses)。
- **项目证据或知识边界**：确认事实是华为 TX 阶段负责文生视频/文生图模型的国产卡迁移、功能/精度/性能闭环；HunyuanVideo-14B 与该输入 shape 是面试讲解例子。未核验的 VAE stride、patch size、head 数、并行 degree 和分项收益不要说成项目事实。
- **高概率追问**：为什么原始 129 帧不等于 129 tokens？Ulysses 与 Megatron CP/Ring Attention 的区别？All-to-All 为什么容易受跨节点拓扑影响？head 数不够怎么扩？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：视频 DiT 的“长序列”从哪里来？Ulysses 具体切什么、通信什么，还能配哪些优化？

- **面试官意图**：核对文生视频项目是否真正理解时空 token、attention 并行和国产卡拓扑，而不是把 LLM CP 原样套过来。

- **危险回答**：直接把 `640×640×3×129` 相乘当 Attention sequence；把 Ulysses 说成切模型参数；把官方推理配置当项目训练配置；声称所有通信都被完全 overlap。

</details>

↩ [返回本 Part 导航](#part-ii) · ↑ [返回面试速查控制台](#interview-console)

<a id="megatron-07"></a>
#### MEGATRON-07｜Pipeline Parallel bubble 怎么估算和优化？（P1，10 分钟）

- **直接回答（60 秒）**：

  > PP 的 bubble 来自流水线填充和排空。设 `p` 个 stage、`m` 个 microbatch，在 stage 均衡且忽略通信时，普通 1F1B 的额外开销相对有效计算是 `(p-1)/m`；如果问占总时间的比例，则是 `(p-1)/(m+p-1)`，这两个分母不能混。
  >
  > 优化先看能否减少 stage、增加合理的 microbatch 数并平衡各 stage，再评估 VPP。VPP 让一个物理 rank 持有多个 model chunks，能缩小理想 bubble，但会增加 P2P 和调度复杂度，不保证真实 step 一定更快。

- **VPP 公式的前提**：只有 chunks 的 forward/backward 近似均衡、microbatch/layer divisibility 或 custom layout 满足调度约束，且先忽略新增通信时，理想 bubble 才约再除以 VPP size。还需检查 activation 生命周期和小 chunk 的 kernel efficiency；完整推导见 [MEGATRON-01](#megatron-01)。
- **深入阅读**：[5D 总览：Pipeline Parallelism 与 VPP](../training-infra-roadmap/topics/distributed_training.md#pipeline-vpp)、[Pipeline Parallelism](../training-infra-roadmap/topics/pipeline_parallelism.md)。
- **项目证据或知识边界**：技能栏“了解/使用”；如项目未重点调 PP，明确无直接性能案例。
- **高概率追问**：为什么 first/last stage 更容易不平衡？长上下文下 PP 是否更划算？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：microbatch、stage balance、1F1B 和 interleaving 有什么关系？

- **面试官意图**：检查 PP 的调度与吞吐理解。

- **危险回答**：只背 bubble 公式；认为 microbatch 越多越好；忽略不均衡 stage。

</details>

↩ [返回本 Part 导航](#part-ii) · ↑ [返回面试速查控制台](#interview-console)

<a id="megatron-08"></a>
#### MEGATRON-08｜Packed Sequence 为什么能提吞吐，又会带来哪些风险？（P1，10 分钟）

- **直接回答（30–60 秒）**：

  > Packing 把多个样本放进连续 token 存储，主要收益是少算 padding。它不能只把文本拼起来：要用 `cu_seqlens` 或 segment boundary 隔离样本的 attention，并正确处理 position、labels 和 loss mask。和 CP、FlashAttention、dynamic batch 组合后，还要检查每个 rank 的负载与布局兼容。验证时比较同一组样本 packing 前后的逐 token loss，而不是只看吞吐上涨。

- **项目证据或知识边界**：你有 packed sequence/THD 经验；准备一个边界 bug 或验证方法。
- **高概率追问**：packing 后 batch size 怎么定义？长短样本混排如何平衡 CP rank？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：padding waste、attention mask、position id 和 loss mask 如何处理？

- **面试官意图**：验证长序列训练的数据-算子接口能力。

- **危险回答**：只说“拼起来就行”；忽略跨样本 attention 泄漏；用总 token 代替有效 token。

</details>

↩ [返回本 Part 导航](#part-ii) · ↑ [返回面试速查控制台](#interview-console)

<a id="megatron-09"></a>
#### MEGATRON-09｜Recompute 和 Offload 应该怎么选？（P1，10 分钟）

- **直接回答（30–60 秒）**：

  > Recompute 是少存 activation、到 backward 时重新计算；offload 是把指定状态搬到 CPU 等外部存储，需要时再取回。一个主要花额外计算，一个主要花传输和主机资源，省多少显存取决于处理的对象。
  >
  > 所以先找峰值来源：activation 很重就比较 selective/full recompute；参数或 optimizer state 很重则评估对应 offload。最终看峰值显存和 step time，尤其是没有被计算掩盖的传输时间，不能因为用了异步 copy 就认为成本消失。

- **项目证据或知识边界**：直接对应 selective recompute 与 offload PCIe 诊断。
- **高概率追问**：full recompute 约增加多少计算？哪些层适合 selective？NVMe offload 何时可用？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：显存不够时，为什么不是两个都开满？

- **面试官意图**：考计算、PCIe/NVLink、显存和吞吐的 trade-off。

- **危险回答**：把 offload 当免费显存；全开后只看能跑；不做 memory snapshot。

</details>

↩ [返回本 Part 导航](#part-ii) · ↑ [返回面试速查控制台](#interview-console)

<a id="megatron-10"></a>
#### MEGATRON-10｜分布式 checkpoint 如何支持并行度变化恢复？（P1，10 分钟）

- **直接回答（30–60 秒）**：

  > 换并行度后，同一个 rank 不再负责原来的那块参数，所以 checkpoint 不能只保存“rank 号对应哪个文件”。它需要记录全局 tensor，以及每个 shard 的 offset、shape 和 replica 信息，让加载器按新拓扑重新切分。
  >
  > 真正恢复训练还要处理 optimizer、RNG、scheduler、data cursor 和共享权重；这些状态是否支持换并行度，要看具体 checkpoint 格式和版本。保存要有完整性提交点，恢复后验证状态、loss 连续性和短窗口训练，不能以能 load 为结束。

- **项目证据或知识边界**：你有 Megatron distributed checkpoint crash/deadlock 经验；准备 `flattened_range` 案例边界。
- **高概率追问**：PP stage 改变如何映射层？async save 如何防半成品？optimizer reshard 为什么更难？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：TP/PP/DP 改变时为什么普通 rank-local 文件不够？

- **面试官意图**：检查 checkpoint schema、全局 tensor metadata 和恢复验证。

- **危险回答**：只转换 model weights；忽略 optimizer 和 data cursor；以能 load 作为正确。

</details>

↩ [返回本 Part 导航](#part-ii) · ↑ [返回面试速查控制台](#interview-console)

<a id="bridge-01"></a>
#### BRIDGE-01｜MBridge 是什么？与 Megatron Bridge 是什么关系？（P1，10 分钟）

- **直接回答（60 秒）**：

  > Bridge 负责把 Hugging Face 的模型配置和权重，映射到 Megatron-Core 的分布式布局，并支持导出；它不负责替代 Megatron 的并行执行。`mbridge` 和 NVIDIA 的 `Megatron Bridge` 是两个独立 package，不只是改名。
  >
  > 我们的 AReaL 分支保留两种 backend，通过 `bridge_type` 选择。我负责的是在 RL 框架里集成、选择并验证转换路径，不是这两个库的作者。迁移时不能只看新模型能否 load，还要验证 logits/loss、训练和 save/resume，以及已有 disk I/O、tree-attention 路径是否兼容。

- **项目版本与选择背景**：该分支固定 `mbridge==0.15.1` 和 `megatron-bridge==0.3.0`，旧链路默认 mbridge。`mbridge` 是较早的原型，其思想后来被官方 Megatron Bridge 采用；官方库提供更新模型及 PEFT/LoRA 支持，但实际迁移仍需逐模型验证，不能仅凭 package 更新作决定。

- **项目证据或知识边界**：你的 ownership 是在 RL 框架里使用、选择和集成 Bridge backend，并验证权重转换/恢复；不是 mbridge 或 NVIDIA Megatron Bridge 的作者。
- **深入阅读**：[Bridge 层的职责、两套实现和项目落点](../training-infra-roadmap/topics/fsdp.md#bridge-layer)；[verl/AReaL 训练后端与 rollout 布局选型](../training-infra-roadmap/topics/rl_framework_selection.md)。
- **高概率追问**：Bridge 与 Megatron-Core 谁负责 process group？为什么不能直接 `load_state_dict`？转换正确性如何验证？为什么项目仍保留旧 mbridge？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：为什么 RL/SFT 框架需要 Bridge 层？两个同名近似项目如何区分？

- **面试官意图**：验证你是否真正处理过 Hugging Face 与 Megatron 之间的模型配置、权重和 checkpoint 边界。

- **危险回答**：把 MBridge 说成 Megatron-Core 的并行模块；认为两个 package 只是改名；只验证能 load，不验证 logits/loss/save-resume。

</details>

↩ [返回本 Part 导航](#part-ii) · ↑ [返回面试速查控制台](#interview-console)

### P2 选学｜时间允许再补

<a id="p2-02"></a>
#### P2-02｜FlashAttention 为什么更省显存、更快？（P2，6 分钟）

- **直接回答（30 秒）**：

  > FlashAttention 主要优化的是 IO，不是把标准 attention 换成近似算法。它用 tiling 和 online softmax 分块计算，避免把完整 `S×S` attention matrix 写到 HBM，再读回来。因此显存和访存开销明显下降，但精确 attention 的计算复杂度仍近似二次；其他 activation、参数和 workspace 也仍然存在。

- **项目证据或知识边界**：有长上下文使用经验；若没写 kernel，定位为机制和集成调优。
- **高概率追问**：为什么仍可能在 256K OOM？如何与 CP/packed sequence 组合？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：它是否改变 attention 数学结果或复杂度？

- **面试官意图**：检查 kernel/IO 基础。

- **危险回答**：把复杂度说成线性；认为它消除所有 attention activation。

</details>

↩ [返回本 Part 导航](#part-ii) · ↑ [返回面试速查控制台](#interview-console)

### 本 Part 追问路线

X1 MoE 优化 → Dense/MoE 结构与 router → 5D 并行选择 → 本 rank 显存账 → 训练框架分层选型 → SFT data contract → 长上下文/多模态项目 → 千卡规模交付。

---

<a id="part-iii"></a>
## Part III｜RL 算法、verl 与 Fully Async RLVR

**学习目标**：讲清 RLVR 的 role/data/control flow，以及同步到异步后如何做生产者—消费者配平、权重同步和正确性控制。

**本 Part 导航**：Core：[RESUME-02 异步收益与配平](#resume-02)；P0 扩展：[RL-ALGO-01 PPO/GRPO/DAPO](#rl-algo-01) · [DPO-01 偏好优化](#dpo-01) · [RESUME-03 gen-TP 与实例数](#resume-03) · [ROLLOUT-01 Rollout 优化](#rollout-01) · [VERL-01 框架分层](#verl-01) · [VERL-02 共置与分离](#verl-02) · [VERL-03 权重同步](#verl-03) · [VERL-04 异步与样本新鲜度](#verl-04) · [VERL-05 训练正确性](#verl-05) · [VERL-09 推理后端选型](#verl-09)；P1：[VERL-06 数据与分发](#verl-06) · [VERL-07 RL 角色](#verl-07) · [VERL-08 Ray 故障](#verl-08) · [VERL-10 版本演进](#verl-10) · [VERL-11 真实后训练工作](#verl-11)；P2：[P2-05 并发队列编码](#p2-05)。

### Core｜最高优先入口

<a id="resume-02"></a>
#### RESUME-02｜Fully Async 相比同步 RLVR 有什么优势？你如何把初始吞吐从 76 优化到 211–255 tokens/s/GPU？（P0，20 分钟）

- **直接回答（60–90 秒）**：

  > Fully Async 的收益是让生成和训练重叠，不是让单条回答自动生成得更快。Rollouter 持续生产，Trainer 凑够 batch 就消费，两侧通过队列解耦；代价是旧样本、背压和权重更新更难管理。
  >
  > 我们的 30B-A3B、32K、32 张 A100 场景，最初异步配置把 24 张卡给训练、8 张卡给生成，生成 TP 为 4，只有两个推理实例，Trainer 经常等数据，吞吐只有 76。随后我降低 gen-TP、增加实例，并联合调整 batch 触发、缓存生命周期和调度配置，优化窗口达到 211–255 tokens/s/GPU。这个结果属于 async 内部的联合优化，不能说成同步切异步提升三倍，也不能全部归因于降 TP。
  >
  > 再增加 rollout 资源后，Trainer 空闲减少，但瓶颈逐渐转到 actor update。所以我用两侧等待时间、队列水位和 policy lag 配平资源，同时确认有效训练 token 和模型效果没有退化。

- **项目展开：配置、观测与边界**：

  | 配置阶段 | 资源与实例 | 观测 |
  |---|---|---|
  | 同步诊断 | Qwen3-30B-A3B、32K、32 张 A100 | rollout 约占阶段时间 79%，用于识别可重叠空间 |
  | 初始 async | `3T+1R`；24 张 Trainer GPU、8 张 Rollouter GPU；`gen-TP=4`、2 个 vLLM 实例 | 吞吐 76；trainer idle ratio 0.41，producer 供给不足 |
  | 联合优化 | `3T+1R`；`gen-TP=2`、4 个实例；结合下述供给与 serving 配置 | 优化窗口 211–255，不拆分无独立 A/B 的贡献 |
  | 更多 rollout 资源 | `2T+2R`；16 张 Rollouter GPU、8 个实例 | 候选窗口 236–293；idle ratio 0.10–0.14；瓶颈转向 actor update |

  联合配置包括 `require_batches`/trigger、`free_cache_engine`、dynamic batch、chunked prefill、prefix cache、CUDA Graph path、partial rollout、bounded staleness、rollout correction、validation frequency，以及 `max_model_len`、`max_num_batched_tokens`。追问时按它们减少哪段暴露等待展开，不把配置名称堆进主答，也不虚构单因素收益。

- **Benchmark 门禁**：先声明分子、分母和窗口，固定模型/checkpoint、prompt-response 长度分布、采样参数、硬件、并发上限和统计区间；warmup、checkpoint、validation、失败重试和过滤样本要明确是否包含。除吞吐外同时报告 queue depth、trainer idle、policy version lag 和 rejected/stale ratio，防止用堆积旧样本换表面吞吐。
- **项目证据或知识边界**：`76 → 211–255` 是 async 初始配置与优化配置的比较；`236–293` 是 `2T+2R` 候选窗口，二者都不是全程平均。同步“约 200”只用于说明最初的阶段拆解和选型背景，只有在相同 workload、窗口和 `tokens/s/GPU` 分母确认后才能做性能比较；确认前不要说 Fully Async 超过同步，更不能说相比同步提升三倍。CUDA Graph 的 `14x` 来自另一项 35B 真实 RL decode 证据，也不能用于解释这里的 211–255。
- **高概率追问**：同步链路中哪些阶段真的串行、哪些可以重叠？为什么 gen-TP=2 更快？queue 空/满分别说明什么？2T+2R 为什么不是最终答案？staleness 怎么控制？parameter sync 占多少？generated token 和 effective training token 有何区别？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：为什么要从同步改为 Fully Async？它解决了哪些等待？为什么初始 async 反而只有 76？

- **面试官意图**：验证你是否理解异步架构的系统收益、供需模型和 off-policy 代价，而不只是调了 gen-TP 和资源配比。

| 维度 | 本项目/典型 phased 同步基线 | Fully Async |
|---|---|---|
| 调度 | logical batch 组装完成后才能进入对应 update，新 policy rollout 前完成相应 weight sync | Rollouter 持续生产，Trainer 按可用 batch 消费 |
| 暴露等待 | 主要 rollout 窗口 Trainer 等待，主要 update/sync 窗口 Rollouter 等待；长 trajectory 放大 logical batch wait | rollout 与训练时间重叠，长尾不再阻塞整个同步 step，但仍受 group/batch 完整性约束 |
| 资源 | 阶段共享或固定编排，单阶段资源利用高但容易产生 phase bubble | Trainer/Rollouter 可独立扩缩容，但固定分池错误会造成一侧长期空闲 |
| Policy 语义 | policy freshness 和 step 边界更直接 | 需要 queue、policy version、staleness、backpressure 和恢复协议 |

- **危险回答**：“异步一定比同步快”；把所有同步实现说成完全串行；把不同资源配比的单卡吞吐直接横比；只报最高点 293；把 async 等价为严格 on-policy；用堆积旧 policy 样本换吞吐。

</details>

↩ [返回本 Part 导航](#part-iii) · ↑ [返回面试速查控制台](#interview-console)

### P0 扩展｜首轮前应掌握

<a id="rl-algo-01"></a>
#### RL-ALGO-01｜请用最简单的话描述 PPO、GRPO 和 DAPO（P0，15 分钟）

- **直接回答（60 秒）**：

  > PPO 是“让高 advantage 的回答更可能出现，但用 clipped ratio 避免新策略一步走太远”；通常需要 Critic 估 value，LLM RLHF recipe 还常另加 Reference KL。GRPO 是“同一个 prompt 采样一组回答，用组内 reward 相对高低做 advantage”，因此可以省掉 Critic，但依赖完整且可比较的 group。DAPO 可以理解为面向大模型 RL 的 GRPO 工程增强：用 Clip-Higher、dynamic sampling、token-level policy-gradient loss 和 overlong reward shaping 处理探索不足、无效 group、长短样本权重和硬截断问题，目标是提高训练效率与稳定性。

- **系统映射**：PPO 重点守住 `value/GAE/old_logp`；GRPO 重点守住 `prompt-group membership/reward std/response mask`；DAPO 还要求动态采样、有效 group 过滤、token-level normalization 和 overlong 标记不能在异步队列里错位。
- **深入阅读**：[PPO、GRPO、DAPO：从公式到 RL Infra 数据契约](../training-infra-roadmap/topics/agentic_rl.md#ppo-grpo-dapo)。
- **项目证据或知识边界**：你以 RL Infra 的算法落地和正确性为主，不必把自己包装成算法提出者；重点回答算法变化如何改变 rollout、资源和校验。
- **高概率追问**：GRPO 没有 Critic 为什么仍有 baseline？group reward std=0 怎么办？DAPO 的 dynamic sampling 为什么影响吞吐？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：三种算法各自解决什么问题？系统侧需要提供哪些数据？

- **面试官意图**：确认你能把算法目标翻译成 rollout、logprob、reward、group 和 update 的工程数据契约。

- **危险回答**：“PPO 有 Critic、GRPO 没 Critic、DAPO 更好”后结束；只背 loss 名称，不讲数据契约和系统代价。

</details>

↩ [返回本 Part 导航](#part-iii) · ↑ [返回面试速查控制台](#interview-console)

<a id="dpo-01"></a>
#### DPO-01｜DPO 如何工作，与 SFT、PPO/GRPO 怎么选？（P0，12 分钟）

- **直接回答（60–90 秒）**：

  > SFT 是模仿标准回答，DPO 是学会更偏好一对回答中的好答案，PPO/GRPO 则是在线采样后优化 reward。DPO 比较 preferred 和 rejected 在当前模型、reference 模型下的 response logprob，让当前模型相对 reference 更偏向 preferred。它不需要在线 rollout、单独训练 Reward Model 或 Critic，因此训练链路通常更简单；但效果依赖偏好对的质量和覆盖范围。已有可靠离线偏好数据时我会优先评估 DPO；需要模型自己探索、调用工具并根据环境结果学习时，我会重点评估在线 RL，而不是只比较 loss 名称。

- **公式只说到这一步**：令 `Δ=((log πθ(yw|x)-log πref(yw|x))-(log πθ(yl|x)-log πref(yl|x)))`，DPO 最小化 `-log σ(βΔ)`；`β` 控制相对 reference 的偏离强度。这里的 logprob 必须只聚合 response 有效 token，并保证 chosen/rejected 使用同一 prompt、tokenizer 和 chat template。
- **系统侧检查**：paired sample identity 不能被 shuffle/packing 拆散；chosen/rejected 截断策略和 response mask 必须对称；reference checkpoint/version 要固定；ref logprob 可离线预计算或在线计算，但要校验精度和 lineage；同时关注长度偏置、全拒绝/低质量 preference、data contamination 和 held-out win rate。
- **深入阅读**：[Direct Preference Optimization 原论文](https://arxiv.org/abs/2305.18290)。
- **项目证据或知识边界**：你的强项是 PPO/GRPO/RLVR Infra、数据契约和正确性；如果没有 DPO 生产项目，就明确按算法机制与系统选型回答，不把知识理解说成落地 ownership。
- **高概率追问**：为什么 DPO 仍然需要 reference policy？`β` 太大或太小会怎样？chosen/rejected 长度不同怎么处理？什么时候必须转向 PPO/GRPO？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：DPO 为什么不需要在线 rollout、Reward Model 和 Critic？它比 PPO 简单在哪里，又损失了什么？

- **面试官意图**：确认你能把偏好优化算法翻译成数据、logprob、reference model、mask 和训练系统成本。

- **危险回答**：“DPO 就是不需要 reward 的 PPO”；认为完全不需要 reference logprob；忽略 response mask、偏好数据质量与离线分布覆盖。

</details>

↩ [返回本 Part 导航](#part-iii) · ↑ [返回面试速查控制台](#interview-console)

<a id="resume-03"></a>
#### RESUME-03｜为什么减小 gen-TP、增加实例数会提高 rollout 吞吐？（P0，15 分钟）

- **直接回答（60–90 秒）**：

  > rollout 追求的是所有卡合起来每秒产出多少有效 token，不只是单个请求有多快。TP 增大后，每卡 GEMM 更小，而每层通信需要更多 rank 参与，通信时延可能更难被计算摊薄；固定实现下，每层 collective 的调用次数并不会因此自动增加。我们把 gen-TP 从 4 降到 2，同样 8 张卡能部署 4 个而不是 2 个实例，增加独立并发池。但前提是每个实例仍有足够显存放权重和 KV cache，且有足够请求把它喂满。我会同时看实例吞吐、KV 容量、尾延迟和通信暴露时间，决定 TP 与实例数的平衡点。

- **机制依据**：经典 Megatron TP 的 Attention/MLP 前向各有一次 all-reduce；应区分调用次数、collective 内部通信轮数和通信占比。[NVIDIA Megatron 原始说明](https://research.nvidia.com/labs/adlr/MegatronLM/)。

- **项目证据或知识边界**：你有直接项目证据；但面试前应补一张 `TP × 实例数 × 并发 × token/s × p95` 表。
- **高概率追问**：何时 TP=1 更好？什么时候必须增大 TP？长上下文 KV cache 会怎样改变结论？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：TP 越大单模型越快，为什么你的场景反而选择更小 TP？

- **面试官意图**：检查你是否理解 decode 的计算/通信特征、并发和集群拓扑。

- **危险回答**：“TP 通信多，所以越小越好。”模型放不下、KV cache 不够或单实例计算太慢时并不成立。

</details>

↩ [返回本 Part 导航](#part-iii) · ↑ [返回面试速查控制台](#interview-console)

<a id="rollout-01"></a>
#### ROLLOUT-01｜Rollout 做了哪些优化？如何系统定位瓶颈？（P0，18 分钟）

- **直接回答（60–90 秒）**：

  > 我先把 rollout 时间拆成模型执行、排队、环境交互和权重更新，再看哪段真正让训练等数据。模型执行慢，就区分 prefill 和 decode，分别检查 prefix reuse、batch、TP 通信和 CUDA Graph；GPU 有空隙但请求没完成，就检查长尾、补位和 worker 分发；生成很快但 Trainer 仍等，就检查完整 group、拒绝原因和两侧资源配比。
  >
  > 我的直接项目工作包括 gen-TP/实例数配平、异步生产消费和 Gateway 调度。验收时不仅看推理服务器 token/s，还看真正参与训练的有效 token、完整 cohort 供给、weight-sync pause 和端到端 update interval。否则可能只是生成了更多被拒绝或过旧的样本。

- **先按症状分流，不要盲调开关**：

  | 现象 | 一阶判断 | 优先动作 | 唯一细节入口 |
  |---|---|---|---|
  | decode kernel 碎、CPU launch 高 | 执行/launch bound | CUDA Graph、固定 bucket、融合 sampling；再测 graph miss | [RESUME-13](#resume-13) |
  | 单实例不满、TP collective 暴露 | 并行粒度不合适 | 降 gen-TP、增实例和并发；同时守住权重/KV 显存 | [RESUME-03](#resume-03) |
  | prefill/KV 占满或 cache 抖动 | memory/cache bound | prefix/Paged KV、block/eviction、chunked prefill；分开统计 prefill/decode | [RESUME-14](#resume-14) |
  | GPU 间歇空闲、短请求被长请求阻塞 | scheduler/tail bound | continuous batching、完成即补位、长度/负载感知路由 | [RESUME-19](#resume-19) |
  | trainer 或 rollouter 长时间等对方 | pipeline rate mismatch | T:R 配平、colocate/disaggregate、bounded queue 与 overlap | [RESUME-02](#resume-02)、[VERL-02](#verl-02) |
  | 旧样本、半组、失败请求多 | freshness/correctness bound | partial/stale 策略、group quota、版本与 reward identity、safe retry | [VERL-04](#verl-04)、[VERL-05](#verl-05) |

- **项目证据分层**：

  - **本人项目直接证据**：gen-TP/实例数与 T:R 配平、Fully Async producer-consumer、Gateway 完成即补位/均衡分发/失败请求管理；对应 `76→211–255 tokens/s/GPU` 的优化窗口，以及 Gateway rollout throughput `+60%`、Rejected Group `33.18%→2.73%`。这些数字属于不同 workload，不能拼成一个总倍数。
  - **本人项目直接证据，但只代表局部阶段**：AReaL 9B 128K 的 CUDA Graph 是 decode `6–8x`，不是 rollout E2E；Prefix Cache 的 `44%` 是 prefill 阶段测量，不自动代表端到端收益。
  - **联合配置结果、没有单因素 A/B**：dynamic batch、chunked prefill、partial rollout、staleness/correction 等常以组合配置生效；没有受控实验时只讲机制和联合结果，不虚构贡献拆分。
  - **机制理解/今天会评估**：speculative decoding、KV offload、量化 rollout、跨机分层调度等若项目未落地，就明确按模型、硬件、正确性和 E2E goodput 做评估，不说成已交付成果。

- **收口方法**：先固定模型、prompt/response 长度分布、采样参数、并发、硬件和统计窗口，建立 `prefill / decode / scheduler wait / weight sync / trainer wait / rejected` breakdown；一次只改变一个主要变量，并回归同权重 token/logprob、reward/group 完整性和 held-out eval。后端选择见 [VERL-09](#verl-09)，AReaL online Gateway 的实现边界见 [AREAL-09](#areal-09)。
- **高概率追问**：为什么 engine tokens/s 涨但 RL step 变慢？Prefix Cache 何时负收益？CUDA Graph 为什么只加速 decode？如何证明流式补位没有破坏 group？异步如何限制 off-policy？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：不要只列 CUDA Graph、Prefix Cache 等开关；请按端到端链路说明 rollout 的关键优化、指标和项目证据。

- **面试官意图**：检查你能否把推理引擎、KV cache、请求调度、训推协同和异步正确性统一成 goodput，而不是只会调单个 backend。

- **危险回答**：只报峰值 token/s；把所有优化收益相加；用堆积 stale 样本换吞吐；忽略失败/拒绝样本和权重更新停顿；把“今天会评估”说成个人已落地。

</details>

↩ [返回本 Part 导航](#part-iii) · ↑ [返回面试速查控制台](#interview-console)

<a id="verl-01"></a>
#### VERL-01｜verl/HybridFlow 的核心架构是什么？（P0，18 分钟）

- **直接回答（60–90 秒）**：

  > verl 可以分成“上层决定做什么，下层分布式执行”。上层 controller 编排 rollout、reward、advantage、actor update 和权重同步；资源层决定每个角色占哪些 GPU；数据协议保证 token、logprob 和样本身份能跨阶段对齐。真正进入模型计算后，再由 Megatron、FSDP 或推理引擎在各自的进程组里执行。Ray 负责远端 actor 和资源调度，不理解 PPO；推理引擎调度的是请求和 batch，也不决定 RL 更新顺序。这个分层允许上层算法流程复用不同训练后端、推理后端和资源布局。

- **代码展开**：`RayPPOTrainer` 对应算法控制，`ResourcePool/WorkerGroup` 对应资源与集体调用，`DataProto/TensorDict` 对应 batch 协议；TransferQueue 是后续数据传输/存储链路的一部分，不与 TensorDict 当成同类容器。具体 Worker/Engine 类名按项目分支说明。

- **四层画法**：`Algorithm Controller → ResourcePool/WorkerGroup → Data Contract → Backend SPMD Engine`。面试时分别说明“谁决定下一阶段”“谁占哪些 GPU”“传什么 batch/metadata”“谁执行 collective”。
- **深入阅读**：[verl 的 single-controller、SPMD engine 与数据/资源边界](../training-infra-roadmap/topics/rl_framework_selection.md#verl-controller-spmd)。

- **项目证据或知识边界**：你有 verl 二次开发经验；面试前至少能指出自己改过的 trainer/worker/config 路径和一个 upstream 差异。
- **高概率追问**：controller 是否会成为瓶颈？DataProto 如何跨 rank dispatch？旧 `megatron_workers` 与新 Engine Workers 有何变化？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：RayPPOTrainer、WorkerGroup、Actor/Rollout/Ref/Critic/Reward 如何协同？

- **面试官意图**：判断你是否真正读过/改过框架，而不是只会运行 recipe。

- **危险回答**：只说“verl 基于 Ray”；混淆 trainer control plane 与每 GPU worker；背旧版本类名却不说明版本。

</details>

↩ [返回本 Part 导航](#part-iii) · ↑ [返回面试速查控制台](#interview-console)

<a id="verl-02"></a>
#### VERL-02｜Actor/Rollout 应该 colocate 还是分离部署？（P0，15 分钟）

- **直接回答（60–90 秒）**：

  > 我先分清三个问题：角色是否独立、是否共用 GPU、执行时是否重叠。Colocate 是训练和生成复用同一批 GPU，通常靠分时运行、offload 和状态切换节省资源，但要处理训练状态与 KV cache 的显存竞争。分离部署使用独立资源池，适合让训练和生成重叠、分别扩容，代价是跨池权重同步、样本陈旧和恢复更复杂；它并不要求异构硬件。我们选 3T+1R 还是 2T+2R，依据是两侧产消速率和等待时间，而不是固定认为分离更快或 rollout 卡越多越好。

- **版本补充**：当前 verl v0.9.0 的 Unified V1 将 `sync / colocate_async / separate_async` 放进一套执行模型；这是 upstream 演进，不应倒推成项目 v0.7.1 已经具备同样实现。
- **深入阅读**：[colocate、disaggregate 与时间并发的三轴判断](../training-infra-roadmap/topics/rl_framework_selection.md#placement-three-axes)。

- **项目证据或知识边界**：直接对应你的 fully async 项目；3T+1R/2T+2R 是本项目布局，不是通用最佳实践。
- **高概率追问**：为什么 2T+2R 后瓶颈转向 actor？动态资源调度何时更优？colocate 如何释放 KV/optimizer 显存？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：什么场景选择 3T+1R、2T+2R 或 colocated hybrid engine？

- **面试官意图**：评估资源建模、权重同步和不同 workload 下的系统取舍。

- **危险回答**：“分离一定吞吐更高”；只算 GPU 数，不算参数同步和 queue；忽略故障域扩大。

</details>

↩ [返回本 Part 导航](#part-iii) · ↑ [返回面试速查控制台](#interview-console)

<a id="verl-03"></a>
#### VERL-03｜训练态 Megatron 权重如何同步到 vLLM/SGLang？（P0，18 分钟）

- **直接回答（60–90 秒）**：

  > 训练态和推理态的参数名字、分片方式和内存布局可能不同，所以不是直接拷贝一个 state_dict。训练侧先收集或重排参数，完成名字、shape 和 dtype 映射，再传给推理后端做 refit；必要时还要清理旧权重对应的 cache。正确性上，我会暂停需要隔离的推理，确认参与服务的 replica 已完成加载，再发布对应版本。禁止的是一次 forward 读到一部分新、一部分旧的参数，或者实际权重和版本标记不符；异步训练 batch 本身可以包含多个完整、可追溯的 behavior versions，前提是满足算法和 staleness 约束。性能看的是收集、传输、refit 和暂停中真正暴露出来的时间。

- **布局与一致性展开**：TP/PP/EP 可能改变参数布局，CP 主要切 activation/context，不应直接当成参数分片轴；distributed optimizer 的 optimizer 分片也不等于要同步 optimizer 到 rollout。跨 replica 的原子切流是可选发布协议，不等于所有 async 样本必须同版本。[AReaL 原论文](https://arxiv.org/html/2505.24298v1)明确允许同一训练 batch 含不同 behavior versions。

- **项目路径追问**：AReaL 项目中 XCCL 直接 bucket transfer 与 disk 临时 HF transfer 的差异、支持边界和选择见 [AREAL-11](#areal-11)。
- **项目证据或知识边界**：你有跨引擎同步和 final parameter sync 故障经验；准备一次 keyword mismatch 或部分 worker 失败的真实排查。
- **高概率追问**：TP size 不同如何 reshard？LoRA 只同步 adapter 有何差异？如何做 same-weight logp check？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：为什么不是简单 `state_dict` 拷贝？

- **面试官意图**：检查训练-推理双态模型、并行布局转换和一致性保证。

- **危险回答**：认为 NCCL broadcast 完成就代表所有 engine 已可服务；忽略 tokenizer/chat template 和 tied weights；没有 version barrier。

</details>

↩ [返回本 Part 导航](#part-iii) · ↑ [返回面试速查控制台](#interview-console)

<a id="verl-04"></a>
#### VERL-04｜Fully Async、streaming、partial rollout 与 staleness 如何配合？（P0，18 分钟）

- **直接回答（60–90 秒）**：

  > 这四个词描述不同层次。Fully Async 是生成和训练的执行关系，二者不再按每个 step 全局等待。Streaming 是数据如何到达，要说明是 token 流还是样本持续到达。Partial rollout 是未完成的轨迹可以暂停后续跑，必须连同环境状态和 behavior 信息保存，不是把字符串接起来。Staleness 则描述样本对应的策略离当前训练策略有多远；版本差只是代理，还要看 logprob ratio 或 KL。系统用有界队列、权重发布频率、过旧样本处理和 correction，把重叠收益与训练偏差约束在可接受范围。

- **项目流程**：项目 v0.7.1/公司分支里，Rollouter 按 freshness/capacity 写队列，Trainer 拼训练 batch，更新后同步新权重；`require_batches`、partial rollout、bounded staleness 和 correction 一起决定 goodput。当前 v0.9.0 的 unified async/replay/stale-drop 是后续 upstream 能力，必须分开表述。
- **深入阅读**：[Fully Async、streaming、partial rollout 与 staleness 的统一状态机](../training-infra-roadmap/topics/agentic_rl.md#async-streaming-partial-staleness)。

- **项目证据或知识边界**：你的项目基于当时的 v0.7.1/公司分支，Fully Async 仍在快速演进；当前官方已到 v0.9.0，并对 trainer、Agentic RL 和相关数据/权重链路继续重构。面试时必须区分项目实现与当前 upstream，不能把两者类名和能力直接混用。
- **高概率追问**：queue 满/空分别说明什么？怎么 checkpoint in-flight samples？staleness=0 是否自动严格 on-policy？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：这些词各自描述什么，为什么不能互相替代？Rollouter、queue、Trainer 和 ParameterSynchronizer 如何组成闭环？

- **面试官意图**：验证你对自己最强项目的框架层理解，并观察是否认识到 async 并非天然 on-policy。

- **危险回答**：说“完全异步但没有陈旧样本”；只调队列大小；忽略恢复后的 pending/running prompt。

</details>

↩ [返回本 Part 导航](#part-iii) · ↑ [返回面试速查控制台](#interview-console)

<a id="verl-05"></a>
#### VERL-05｜GRPO/RLVR 链路最容易出现哪些“能跑但训错”的问题？（P0，18 分钟）

- **直接回答（60–90 秒）**：

  > 我会按 token、trajectory、group、policy version 四层检查。token 层看 tokenizer/chat template、response mask、rollout 与 trainer logprob、padding/packing；trajectory 层看 reward 对齐、截断、tool trace 和有效 token normalization；group 层看 GRPO 同 prompt samples 是否完整、reward std=0、partial/rejected group；policy 层看 behavior version、importance ratio、weight sync 和 stale rejection。验证方法包括 same-weight logp、tiny deterministic batch、per-token diff、single-rank/多-rank对照、loss 手算和 held-out eval。训练不 NaN 只证明 functional，不证明 numeric 或 efficacy。

- **项目证据或知识边界**：直接对应你的 OPD/MOPD、rollout correction 和 tracing 经验。
- **高概率追问**：rollout logprob 和 trainer recompute logprob 为什么会不一致？group std=0 怎么处理？response length normalization 有何偏差？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：为什么 loss 正常、reward 也涨，结果仍可能不可信？

- **面试官意图**：检查数值正确性和 RL 系统经验，这是高级岗位的重要分水岭。

- **危险回答**：只看最终 reward；把 KL/loss 曲线平滑当作正确性证据；不记录原始 token ids。

</details>

↩ [返回本 Part 导航](#part-iii) · ↑ [返回面试速查控制台](#interview-console)

<a id="verl-09"></a>
#### VERL-09｜vLLM 与 SGLang rollout 后端怎么选？（P0，12 分钟）

- **直接回答（60–90 秒）**：

  > 我不会先给两者排固定名次，而是先锁定项目版本、模型和硬件。第一关是训练接口能否闭环：权重能否更新，token 和 rollout logprob 能否与 Trainer 对齐，暂停恢复后 cache 是否正确。通过这一关，再用真实长度和并发比较吞吐、尾延迟、更新停顿和故障恢复。多轮 Agent 的 prefix reuse、session 与 tool calling 是重点测试项，两种后端都要实际验证。最终选的是在我们 workload 下有效训练供给稳定、维护成本可控的后端，不是 serving 榜单最高的一项。

- **深入阅读**：[vLLM 与 SGLang：面向 RL rollout 的选型矩阵](../training-infra-roadmap/topics/rl_framework_selection.md#vllm-sglang-selection)。
- **项目证据或知识边界**：你接入过两个后端；准备各自一次兼容性或稳定性问题，并明确比较对应的版本、模型和硬件。
- **高概率追问**：为什么同权重 logprob 会不一致？SGLang/vLLM 权重更新如何处理 cache？prefix 命中率高为何仍可能 E2E 更慢？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：不要只比 benchmark，给出训练系统选型维度。

- **面试官意图**：评估推理引擎与 RL dataflow 的集成能力，以及你能否拒绝“固定赢家”叙事。

- **危险回答**：按“谁更快”一刀切；只看公开榜单；忽略版本兼容、weight update 与 token correctness。

</details>

↩ [返回本 Part 导航](#part-iii) · ↑ [返回面试速查控制台](#interview-console)

### P1 深挖｜面试官继续追问

<a id="verl-06"></a>
#### VERL-06｜DataProto 和 WorkerGroup 解决了什么问题？（P1，8 分钟）

- **直接回答（45–60 秒）**：

  > DataProto 解决“一个 batch 的 tensor 和 metadata 如何一起流转”，WorkerGroup 解决“如何把一组远端 worker 当成一个分布式角色调用”。普通 dict 和 Ray actor 也能拼出这些能力，但分片、收集和重排规则会散落在算法代码里。verl 把数据协议与调用分发收进统一接口，后端可以按自己的 DP/TP 布局执行。最容易出错的是只重排 tensor，没同步重排样本 ID、reward 或其他 metadata；形状看起来正确，训练却已经错位。

- **项目证据或知识边界**：若未直接改 DataProto，说明主要从调用和故障层理解。
- **高概率追问**：non-tensor 数据如何广播？microbatch reorder 后 ID 怎么保持？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：为什么不用普通 Python dict + Ray actor？

- **面试官意图**：检查框架接口层和分布式数据 dispatch 理解。

- **危险回答**：只说“序列化”；忽略 dispatch semantics 和数据对齐。

</details>

↩ [返回本 Part 导航](#part-iii) · ↑ [返回面试速查控制台](#interview-console)

<a id="verl-07"></a>
#### VERL-07｜Actor、Reference、Critic、Reward 各自为什么存在？（P1，8 分钟）

- **直接回答（45–60 秒）**：

  > Actor 是要被优化的策略，Rollout 用它的某个版本采样。Reward 判断回答好不好，可以是模型，也可以是规则或测试；Critic 估计状态 value，用来构造 advantage，它不是 reward model。Reference 则提供固定的比较基线，例如限制模型别偏离 SFT 太远。GRPO 用同 prompt 的组内 reward 构造 baseline，所以可以省掉 learned Critic；如果算法不使用 reference KL，也可以不部署 Reference。这些是逻辑职责，不代表每个任务都必须放四份模型。

- **项目证据或知识边界**：有 RLVR/GRPO 使用经验；算法推导若不是主责可保持工程视角。
- **高概率追问**：DAPO 相对 GRPO 改了什么？Reference logprob 何时可预计算？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：GRPO 为什么可以没有 Critic？Reference 是否总需要？

- **面试官意图**：确认 RL 基础与系统资源角色对应。

- **危险回答**：把 reward model 等同 critic；认为 GRPO 完全不需 baseline/normalization。

</details>

↩ [返回本 Part 导航](#part-iii) · ↑ [返回面试速查控制台](#interview-console)

<a id="verl-08"></a>
#### VERL-08｜Ray 在 verl 中最常见的生产故障有哪些？（P1，10 分钟）

- **直接回答（45–60 秒）**：

  > 我先区分是 worker 没启动，还是启动后卡住。前者查 placement group、资源声明和 runtime env；后者先找最早失败的 actor/rank，再检查底层 GPU 进程、网络、object store 和 collective。控制端最后报的 RPC timeout 往往只是下游异常的结果，不能直接当根因。保留日志和进程现场后再清理；退出路径还要能重复执行，避免留下推理 server、sandbox 或 communicator，影响下一次任务。

- **项目证据或知识边界**：你有 Fuyao/Ray RPC/failure cleanup 经验；选一个具体案例。
- **高概率追问**：controller 挂了如何恢复？object store pressure 表现？placement group 为什么 pending？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：资源够但 worker 起不来、RPC 卡住或进程残留怎么办？

- **面试官意图**：验证多机 orchestration 实战。

- **危险回答**：只会 `ray stop --force`；不保存现场；把 Ray 错误当根因。

</details>

↩ [返回本 Part 导航](#part-iii) · ↑ [返回面试速查控制台](#interview-console)

<a id="verl-10"></a>
#### VERL-10｜verl v0.7.0 以后几个大版本发生了什么系统性变化？（P1，12 分钟）

- **直接回答（60–90 秒）**：

  > 我把演进归纳为“从多条实验路径收敛到统一 engine 和统一 async 执行模型”。v0.7 开始强化 engine abstraction、server-based rollout 和 TransferQueue，Fully Async 的 producer-consumer 形态逐步成型；v0.8 推进 Unified Engine 迁移，把 sync trainer 也接入 TransferQueue，并扩展 OPD/Uni-Agent 等能力，但多条新旧路径仍在过渡；v0.9 的 Unified V1 进一步统一 `sync / colocate_async / separate_async`，补 replay/stale drop/wait、streaming dataloader/recovery、Uni-Agent Gateway 和 `delta_sharded` 等权重更新能力。这里会有 breaking change，不能拿当前类名解释旧项目。我实际项目基于 v0.7.1/公司分支，当前 upstream 只用于重评和迁移判断。

- **深入阅读**：[verl v0.7–v0.9 的架构演进与迁移风险](../training-infra-roadmap/topics/rl_framework_selection.md#verl-release-evolution)。
- **高概率追问**：TransferQueue 为什么重要？Unified Engine 解决什么重复？`colocate_async` 与 `separate_async` 的差别？升级如何做 numeric regression？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：请按架构主线说明 v0.7、v0.8、v0.9，而不是罗列 release note。

- **面试官意图**：检查你能否区分项目版本与当前 upstream，并从版本演进提炼可迁移的系统判断。

- **危险回答**：把后续功能说成项目当时已使用；背 feature list 不讲控制流/数据流；看到大版本就直接升级生产。

</details>

↩ [返回本 Part 导航](#part-iii) · ↑ [返回面试速查控制台](#interview-console)

<a id="verl-11"></a>
#### VERL-11｜自研版 verl 支撑了哪些真实后训练工作？请结合 Athena-Brain 与 Capek 0.5 说明。（P1，15 分钟）

- **直接回答（60–90 秒）**：

  > 我参与建设的自研版 verl 支撑了 LLM 和 MLLM 两类真实后训练工作。两条路线都体现了“先把不同能力独立做强，再汇聚成一个部署模型”，但 Athena 主要在 parameter space 做分层 model merge，Capek 则先用 TIES 初始化，再通过 routed MOPD 在 Student 自己访问的状态分布上做 policy-space consolidation。算法 recipe、模型实验和论文由组内算法同学负责；我的贡献边界是自研版 verl 的框架建设，以及训练、rollout、异构任务和后端承载能力，不把论文算法创新归到个人名下。

##### LLM 路线：Athena-Brain-8B

**一句话链路**：`Athena SFT anchor → 同源多域 RL Experts → TIES → 异源 checkpoint 低剂量线性插值 → 单一 Athena-Brain checkpoint`。

![Athena-Brain Figure 3：从 Athena SFT、多域 RL Experts 到 TIES 与低剂量线性插值的 LLM 后训练链路](assets/papers/athena-brain-post-training-figure-3.png)

> 图源：[Athena-Brain Technical Report v2，Figure 3](https://arxiv.org/abs/2607.18985v2)，作者原图；按 CC BY 4.0 保留出处。

1. 从 open-weight base 做 General SFT，形成后续能力分叉和 task vector 的共同 `Athena SFT` anchor；
2. 从同一 anchor 分别训练 Agent、Science、Instruction Following、Code、Embodied 等 domain-specialized RL experts；
3. 共享 lineage 的 experts 相对同一 anchor 定义 task vectors，用 TIES 汇聚为 multi-domain trunk；
4. 另一训练 lineage 的候选 checkpoint 不进入同一 TIES voting pool，而以低权重线性插值补充能力，最终部署一个 8B LLM。

Athena 的这张图是 **parameter-space consolidation**，不要说成用了 MOPD。General RL 可补充为使用 GRPO、correctness reward 和 token-budget reward 兼顾正确性与简洁性，但本题不展开论文数据量和评测数字。最终 model merge 属于完整后训练 pipeline，不自动等于在 verl trainer 内执行。

##### MLLM/VLM 路线：Capek 0.5

**一句话链路**：`Shared Base VLM → 四类能力 Expert GRPO → TIES 初始化 → Student rollout + routed MOPD → 单一推理 checkpoint`。

![Capek 0.5 Figure 6：四类能力专家经 TIES 初始化和 routed MOPD 汇聚为单一 VLM 的后训练链路](assets/papers/capek-0.5-specialization-consolidation-figure-6.png)

> 图源：[Capek 0.5 v1，Figure 6](https://arxiv.org/abs/2608.06756v1)，作者原图；组内论文图片已获公开使用确认，保留原始 watermark 与出处。

1. 从共享 Base VLM 独立训练 Spatial Reasoning、Temporal Understanding、Action Guidance、State Verification 四个 specialist；
2. 各能力的数据、输出格式、parser/verifier 和 reward 不同，但统一到 autoregressive generation 与 GRPO 训练路径，checkpoint 保持 parameter-compatible；
3. 先用四个 expert 的 task vectors 做 TIES，得到统一 Student 初始化；Student 再自己 rollout，每个样本按 capability route 选择对应冻结 Teacher，在 Student-generated prefixes 上提供 token-level 蒸馏信号；
4. Teacher 和 routing 只在 consolidation 训练期存在，最终推理使用一个 autoregressive checkpoint。

Capek 不是“把四个 Teacher 的答案混在一起做 SFT”。TIES 先给出较好的权重空间初始化；routed MOPD 再在 Student 自身访问的前缀上补 policy-space behavior transfer。完整 reverse-KL、student-sampled token surrogate 和 current/behavior/proximal logp 的追问见 [RESUME-09 公式展开](#mopd-loss-followup)；Capek 论文 recipe 与个人 AReaL 项目实现仍需分开说明。

| 维度 | Athena-Brain | Capek 0.5 |
|---|---|---|
| 模型类型 | 8B LLM | 2B dense / 35B-A3B MoE VLM |
| 能力生产 | 同一 SFT anchor 分叉多个 domain RL experts | 同一 VLM 分叉四个 capability specialists |
| 能力汇聚 | TIES + alternative-lineage low-dose interpolation | TIES initialization + routed MOPD |
| consolidation 空间 | 主要是 parameter space | parameter space 后继续进入 policy space |
| 推理形态 | 一个 checkpoint | 一个 checkpoint |

- **统一 Infra 视角**：框架要稳定承载 heterogeneous dataset/task/reward schema、SFT/GRPO 和多 expert recipe、rollout/verifier/backend/checkpoint lineage；MLLM 还要守住 image/video metadata、mask、position 与 sample identity。只有能落到本人代码、配置、日志或故障案例的项才说“我负责”，其余说“框架/团队支持”。
- **高概率追问**：Athena 为什么同源 expert 用 TIES、异源 checkpoint 用插值？Capek 为什么不是只做 TIES？多模态 GRPO 的 data contract 多了什么？两个项目哪些阶段运行在 verl、哪些是离线 merge/evaluation？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：除了框架架构和性能数字，你参与建设的自研版 verl 实际支撑了哪些 LLM/MLLM 后训练工作？

- **面试官意图**：确认自研框架不是单一算法 demo；检查你能否从算法图反推训练系统的数据流、任务抽象和 checkpoint lineage，并诚实拆分 Infra 与算法论文 ownership。

- **危险回答**：说自己提出了论文算法；把 Athena 说成 MOPD；把 Capek 说成 Teacher 生成答案再 SFT；把离线 merge 全部归入 verl trainer；用论文结果替代个人框架改动证据。

</details>

↩ [返回本 Part 导航](#part-iii) · ↑ [返回面试速查控制台](#interview-console)

### P2 选学｜时间允许再补

<a id="p2-05"></a>
#### P2-05｜如果现场让你写 producer-consumer/并发队列代码，会考什么？（P2，8 分钟）

- **直接回答（45–60 秒）**：

  > 我会先明确谁拥有任务、成功后谁确认、失败后谁重试。队列必须有界，满时让 producer 等待或拒绝，不能无限积压。任务要有稳定 ID 和状态，重试不能重复产生副作用；取消和异常要传到上游。退出时先停新任务，再等待或取消 in-flight，最后关闭 consumer。写完不仅测正常收发，还测队列空满、producer 死亡、慢 consumer、重复消息和取消竞态。是否做到 exactly-once，要看确认与副作用能否原子提交，不能靠一个内存队列承诺。

- **项目证据或知识边界**：可绑定 async rollout/message queue/session drain。
- **高概率追问**：exactly-once 是否可能？锁与 async event loop 如何选择？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：如何实现有界队列、取消、重试、幂等和优雅退出？

- **面试官意图**：验证 Python/C++ 工程基本功，不让框架经验掩盖编码能力。

- **危险回答**：只给 happy path；吞掉异常；用无限队列解决阻塞。

</details>

↩ [返回本 Part 导航](#part-iii) · ↑ [返回面试速查控制台](#interview-console)

### 本 Part 追问路线

同步瓶颈 → Fully Async 架构收益 → PPO/GRPO/DAPO 与 DPO 选型 → Rollout 六层优化 → gen-TP/实例/T:R 配平 → staleness 与 correction → 权重同步和恢复 → Athena/Capek 真实模型落地与 ownership。

---

<a id="part-iv"></a>
## Part IV｜AReaL、Gateway、Agentic RL 与 MOPD

**学习目标**：从 online proxy/cohort 数据流出发，回答长时 agent rollout、版本控制、trajectory lineage、阶段优化和多 Teacher 蒸馏。

**本 Part 导航**：Core：[RESUME-08 训练链路与瓶颈](#resume-08) · [RESUME-09 OPD/MOPD](#resume-09)；P0 扩展：[AREAL-01 框架选择](#areal-01) · [AREAL-02 异步偏差控制](#areal-02) · [AREAL-03 微服务边界](#areal-03) · [AREAL-04 轨迹到梯度](#areal-04) · [AREAL-09 Gateway 改造](#areal-09) · [AREAL-11 XCCL 与 disk](#areal-11) · [RESUME-13 CUDA Graph](#resume-13) · [RESUME-19 Gateway 收益](#resume-19) · [RESUME-14 Prefix Cache](#resume-14)；P1：[RESUME-15 拒绝率口径](#resume-15) · [AREAL-05 Partial Rollout](#areal-05) · [AREAL-06 原子权重发布](#areal-06) · [AREAL-07 Session drain](#areal-07) · [AREAL-10 外部 Agent 接入](#areal-10) · [AREAL-08 三层验收](#areal-08)；P2：[P2-04 长上下文平台设计](#p2-04)。

### Core｜最高优先入口

<a id="resume-08"></a>
#### RESUME-08｜请画出你的 Agentic RL 训练链路，最大瓶颈在哪里？（P0，20 分钟）

- **直接回答（60–90 秒）**：

  > 我的项目由外部 Agent 持续生产轨迹，AReaL 负责把这些交互变成可训练数据。一个任务先在 Gateway 创建 session，然后多轮调用模型和工具；系统同时记录 token、生成时的 logprob 和版本。任务结束后提交 reward，同一 prompt 的多条轨迹完整、结束且不过旧，才组成可以训练的 cohort。Trainer 取出数据，计算 advantage、更新模型，再把新权重发布给推理侧。
  >
  > 最大的暴露等待是“等完整 cohort”，不只是模型算得慢：128K 后期推理、同组最后一条长尾、sandbox 和失败重试都会影响供给。我分别优化 decode、prefill 和 Gateway 调度，最后用固定 logical batch 的 update interval、有效训练 token、拒绝原因和样本新鲜度验收，不能只看推理服务器 token/s。

- **版本边界**：下面对齐的是项目实际使用的 **AReaL online proxy + controller-owned cohort admission 二次开发链路**。它不是普通离线 `RolloutWorkflow` 的串行图，也不要把后续 AReaL 2.0 的独立微服务架构倒推为项目当时的实现。
- **系统流程图**：

```mermaid
%%{init: {
  "theme": "base",
  "themeVariables": {
    "fontFamily": "Inter, PingFang SC, Microsoft YaHei, sans-serif",
    "fontSize": "15px",
    "lineColor": "#64748b"
  },
  "flowchart": {
    "curve": "basis",
    "nodeSpacing": 32,
    "rankSpacing": 48
  }
}}%%
flowchart TB

    subgraph EPISODE["① External Episode Producer · 外部并发采样"]
        direction LR
        TASK["Task / Dataset"]
        AGENT["Evals Agent Runtime<br/>启动 cohort × n trajectories"]
        ENV["Tool / Sandbox<br/>维护多轮环境状态"]
        REWARD["Terminal Reward<br/>结束 episode"]

        TASK --> AGENT
        AGENT <-->|"multi-turn"| ENV
        ENV --> REWARD
    end

    subgraph AREAL["② AReaL Online Rollout & Control · 会话、推理与 Cohort 状态"]
        direction LR
        GATEWAY["Gateway + Cohort Admission<br/>grouping · capacity · rollout version · staleness"]
        PROXY["Proxy + InteractionCache<br/>session · token · behavior logp · reward"]
        INFERENCE["Inference Backend<br/>vLLM / SGLang"]
        BARRIER{{"Ready Cohort Barrier<br/>完整 n_samples · rewarded + ended<br/>ready-time staleness gate"}}

        GATEWAY -->|"bind session / route"| PROXY
        PROXY <-->|"agenerate / model response"| INFERENCE
        PROXY -->|"exportable interactions"| BARRIER
    end

    subgraph TRAIN["③ Trainer & Policy Feedback · 消费、更新与新策略发布"]
        direction LR
        PREPARE["Wait & Export<br/>actor.prepare_batch()<br/>OpenAIProxyWorkflow"]
        BATCH["Trajectory Batch<br/>tensorize · redistribute · broadcast"]
        OPTIMIZE["Score & Policy Update<br/>Ref / Teacher / Advantage<br/>PPO / GRPO"]
        WEIGHTS["Versioned Weight Sync<br/>XCCL / transient disk<br/>transfer succeeds → set_version"]

        PREPARE --> BATCH --> OPTIMIZE --> WEIGHTS
    end

    AGENT -->|"1. start_session → session API key"| GATEWAY
    AGENT -->|"2. LLM requests with session key"| GATEWAY
    INFERENCE -. "model responses" .-> AGENT
    REWARD -->|"set_reward + end_session"| GATEWAY
    GATEWAY -. "cohort completeness / version gate" .-> BARRIER

    BARRIER -->|"ready cohort"| PREPARE
    WEIGHTS ==>|"publish new policy"| INFERENCE
    WEIGHTS -.-> CKPT["Checkpoint / Eval<br/>旁路，非 trajectory 主链"]

    BOTTLENECK["最大瓶颈 · ready-cohort wait<br/>① 128K late-turn inference<br/>② last-of-8 cohort straggler<br/>③ sandbox / retry / rejection"]
    BARRIER --- BOTTLENECK

    classDef producer fill:#eff6ff,stroke:#60a5fa,color:#1e3a5f,stroke-width:1.5px;
    classDef rollout fill:#ecfdf5,stroke:#4caf78,color:#164e3b,stroke-width:1.5px;
    classDef trainer fill:#fff7ed,stroke:#e7a23b,color:#6b3b0a,stroke-width:1.5px;
    classDef bottleneck fill:#fff1f2,stroke:#e05260,color:#7f1d2d,stroke-width:2px;
    classDef side fill:#f8fafc,stroke:#94a3b8,color:#475569,stroke-dasharray:4 3;

    class TASK,AGENT,ENV,REWARD producer;
    class GATEWAY,PROXY,INFERENCE rollout;
    class PREPARE,BATCH,OPTIMIZE,WEIGHTS trainer;
    class BARRIER,BOTTLENECK bottleneck;
    class CKPT side;

    style EPISODE fill:#f8fbff,stroke:#bfdbfe,stroke-width:1px
    style AREAL fill:#f5fdf9,stroke:#bbf7d0,stroke-width:1px
    style TRAIN fill:#fffbf5,stroke:#fed7aa,stroke-width:1px
```

- **代码与瓶颈展开**：

  - admission：外部 producer 独立调用 `start_session`；CohortManager 校验 capacity/staleness，绑定 cohort、group rank、rollout version 和 proxy worker，返回 session API key。
  - interaction：后续 LLM 请求携带 session key；Proxy/InteractionCache 记录 token、behavior logp 和 token version，Tool/Sandbox 状态由外部 Agent 维护。
  - ready/export：同组成员 rewarded、ended 且通过 ready-time staleness gate 后，`actor.prepare_batch()` 经 `OpenAIProxyWorkflow` 等待并导出完整 cohort，tensorize 后按 DP 重分配。
  - update/publish：按 recipe 计算 Ref/Critic/Teacher/Prox logp 与 advantage，执行 PPO/GRPO update；权重传输成功后才推进实际版本。checkpoint/eval 不是每条 trajectory 的数据主链，但执行时仍可能占用资源、增加 update 间隔，benchmark 要说明是否包含。
  - 历史基线记录 rollout wait 约占 step 的 87%；项目的 8-way cohort 会放大 last-of-8 straggler。这个观测只解释对应基线，不外推为所有 Agentic RL 的固定占比。

- **项目证据或知识边界**：底稿记录 DeepSWE `6467s→2301s`、Seta Terminal `2240s→770s` 等更强数据，但它们未全部进入当前简历；使用前确认可对外披露和统计口径。
- **高概率追问**：`start_session` 与后续 LLM request 的 API 边界是什么？为什么 online 模式没有 trainer 内部 Agent Workflow？cohort 为什么放大 tail？reward、session 和 trajectory 在哪里落盘/导出？为什么 weight sync 完成后才能推进 version？cache hit 高为什么不一定让 E2E 更快？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：从 task 到 policy update，一条 trajectory 经历哪些系统？

- **面试官意图**：判断你是否拥有端到端视角，以及能否区分 agent、inference、reward、training 和 control plane。

- **危险回答**：把链路画成 `Agent Workflow → vLLM → Reward → Training Queue` 的固定串行管线；把 `policy version` 当成权重同步前独立生成的模型产物；把 checkpoint 画进每条 trajectory 的关键路径；只看模型服务器 token/s，忽略 session/cohort、环境失败和样本版本。

</details>

↩ [返回本 Part 导航](#part-iv) · ↑ [返回面试速查控制台](#interview-console)

<a id="resume-09"></a>
#### RESUME-09｜OPD/MOPD 解决什么问题？你如何证明它正确且有效？（P0，20 分钟）

- **直接回答（60–90 秒）**：

  > 我们希望把不同领域 RL Expert 的能力汇聚到一个模型。项目先试了 TILE merge，但初步效果没达到目标，于是改用 MOPD。Student 从 RL 前模型初始化，自己在各领域环境中生成轨迹；系统按数据领域选择冻结 Teacher，让 Teacher 对 Student 实际生成的同一条 token 路径打分，而不是另生成标准答案。训练用 Teacher 与当前 Student 的逐 token logprob 差构造蒸馏信号，最终只部署一个 Student。
  >
  > 我把验收分成三层：流程能否闭环、token/logprob/路由是否算对、无污染评测上是否真有提升。当前双 Teacher 的项目结论是 SWE、Terminal 双域提升且 General 不下降；完整统计信息仍需补齐，不能把单 Teacher 的提升数字当成双 Teacher 结果，也不能用 loss 下降证明能力提升。

- **数据流与选型展开**：领域 RL Expert 是冻结 Teacher；Student 使用各领域原 RL 数据，保留 `data_source`，按领域路由 scoring。MOPD 在 Student 自己访问的前缀上继续学习，与静态 parameter merge 不同，但不自动消除共享参数的跨域梯度冲突；混域配额、trajectory 权重和 General 回归仍要控制。

<a id="mopd-loss-followup"></a>
##### 追问：reverse-KL、sampled-token surrogate 与三份 Student logp

对固定的 Student-generated prefix `s_t`，完整词表上的局部目标是：

```text
KL(π_current || π_teacher)
  = Σ_v π_current(v | s_t) [log π_current(v | s_t) − log π_teacher(v | s_t)]
```

sampled-token OPD 不必传完整词表分布，而是在 Student 采出的 action 上构造 stop-gradient 信号。项目 `mopd_pg` 的核心是：

```text
A_mopd,t = clamp(stop_gradient(logp_teacher,t − logp_current,t), −c, +c)
A_total,t = A_mopd,t + λ × A_task,t
```

`logp_current` 必须来自**本次训练 forward**；正 advantage 表示 Teacher 比当前 Student 更认可这个 sampled token。随后它进入 PPO-style clipped surrogate，并按有效 action mask 和配置的 token/trajectory 权重归一化。`λ=0` 是纯蒸馏，不应再让 task reward 路径影响梯度。

| 名称 | 来源 | 用途 |
|---|---|---|
| behavior logp | 当时真正采样该 token 的 rollout policy | 记录采样分布，诊断 staleness；需要时参与 correction |
| proximal logp | 由 recipe 指定的近端参考 policy，常在本轮更新前 recompute 并固定 | PPO ratio 的分母；不能与冻结 Teacher 或 SFT Reference 混为一谈 |
| current Student logp | 每个训练 forward 的当前参数 | ratio 分子，以及项目 MOPD advantage 中被 stop-gradient 的 Student 项 |

在标准 token-level 路径中，`r_t = exp(logp_current,t − logp_prox,t)`；behavior 与 proximal 的差异由配置的 correction/rejection 处理，不能把二者混掉。严格 fresh sampling、固定 prefix 且无额外 clipping 时，sampled-token PG 对应局部 reverse-KL 的梯度方向；异步旧样本、PPO clipping、advantage clipping 和 trajectory weighting 会改变估计，不能宣称项目 loss 就是完整轨迹 reverse-KL 的精确无偏梯度。

代码核验位置是项目 `areal/trainer/ppo/actor.py` 的 `mopd_advantage` 与 `grpo_loss_fn`、`areal/utils/functional/functional.py` 的 `ppo_actor_loss_fn`。机制续读：[MOPD 专题：最小数据流与算法边界](../training-infra-roadmap/topics/mopd.md#opd-的最小数据流)。

- **为什么不继续依赖 TILE merge**：项目已确认的事实只有“初步效果没有达到多个领域能力同时保留的目标”。可以从系统选型角度说，静态 model merge 不会自动利用原 RL 数据在 Student 的访问分布上继续学习；但在没有实验记录前，不补造 TILE 的内部机制、系数敏感性、具体掉点或论文归属。
- **三层验证门禁**：

  1. **FUNCTIONAL**：混域数据、`data_source` 路由、Teacher scoring、backward、weight sync、checkpoint/recovery 能闭环；各 Teacher 路由都有非零样本，失败不能静默串域。
  2. **NUMERIC**：token、mask、Teacher/Student logp、scatter/gather 和 normalization 对齐；same-weight 条件下蒸馏信号应接近零；异常在各 rank 上 fail-consistent。
  3. **EFFICACY**：在相同协议下比较 RL 前 Student、各领域 Expert、TILE merge、单 Teacher OPD 和多 Teacher MOPD；分别评测各领域能力和 General 回归，并看逐题配对、多个 checkpoint/seed 与置信区间。训练 loss 下降不能替代下游效果。

- **当前效果结论**：最新版双 Teacher MOPD 结果是 SWE、Terminal 双域提升且 General 不下降。当前允许口述的是这个方向性结论；checkpoint、样本量、seed、baseline、评测窗口和置信信息仍需在证据卡补齐，补齐前不说“显著提升”或虚构双 Teacher 的具体 pp。简历中的单 Teacher `Terminal +7.9pp`、`SWE +7.0pp` 只属于各自单 Teacher 实验。

- **Teacher headroom 的准确说法**：这是本项目的 Go/No-Go 门，不是普遍定理。如果 Teacher 在目标领域没有可测 headroom，same-path token 信号也没有显示稳定的局部互补能力，就先检查 Teacher、数据和评测协议，而不是直接增加蒸馏步数；但 Teacher 总分不高于 Student，并不严格排除它在部分状态上仍能提供有效监督。
- **项目证据或知识边界**：多 Teacher 路由、score validation、`mopd_pg`、mixed-domain data、trajectory weighting、online drain、recovery 和评测工装，只有能映射到本人负责的 PR、设计或实验记录时才说“我设计并实现”；其余说成项目能力。效果结论以最新版简历为准，但不把早期受污染 run、单 Teacher pp 或 early canary 当成双 Teacher 最终数字。
- **高概率追问**：TILE merge 当时如何评估？为什么 Student 从 RL 前模型而不是某个 Expert 初始化？Teacher 为什么对 Student 的同一 token path 打分？`mopd_pg` 的 token advantage 怎么构造？如何防止 `data_source` 串域？equal-token weighting 为什么可能偏向长 trajectory？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：为什么不用 Model Merge 汇聚多个 RL Expert，而要做 MOPD？完整数据流和验证门禁是什么？

- **面试官意图**：检查你能否从真实业务问题推导技术选型，讲清 model merge 与 on-policy 行为蒸馏的差别，并识别多 Teacher 路由、distributed correctness 和效果夸大风险。

- **危险回答**：把 TILE 的未确认机制和论文来源讲成事实；说 Teacher 重新生成答案；暗示最终推理仍需动态路由 Teacher；用训练 loss 下降证明能力提升；把单 Teacher、受污染的探索实验和双 Teacher 正式结果混在一起；把代码仓库已有功能全部说成个人实现。

</details>

↩ [返回本 Part 导航](#part-iv) · ↑ [返回面试速查控制台](#interview-console)

### P0 扩展｜首轮前应掌握

<a id="areal-01"></a>
#### AREAL-01｜为什么先选 verl，Agentic RL 阶段又转向 AReaL？（P0，15 分钟）

- **直接回答（60–90 秒）**：

  > 两次选择对应不同问题。最初要稳定交付 SFT 和标准 RLVR，团队已有 Megatron 资产，当时 verl 的角色编排、模型和后端适配更匹配。后来转向 128K、多轮工具和外部 Agent 在线请求，主要矛盾变成 session/cohort 生命周期、长尾和策略版本管理；基于当时的代码，AReaL online 路径更贴合，控制面改造也更集中，所以我们转向 AReaL，并补齐 Gateway、恢复和多 Teacher 路由。
  >
  > 这不是“AReaL 异步、verl 同步”的永久分类。当前 verl 也在补强异步和 Agentic RL，如果重新选型，我会锁定版本，用同一 workload 和故障场景做 PoC，再比较正确性、有效供给和维护成本。

- **架构展开**：verl 的重点是灵活编排 RL 多角色计算与后端；项目所用 AReaL online 路径的重点是持续生产 trajectory，并管理 session、cohort、policy version 和 staleness。当前 upstream 的能力与项目采用时的分支必须分开。

- **选型维度**：`workload 形态 → 训练后端/模型支持 → rollout/agent 接口 → placement 与 weight sync → correctness → 可观测/恢复 → 二开半径与团队维护成本`。公开 benchmark 只能提供候选，最终要用自己的模型、长度分布、并发和故障场景做 A/B。
- **详细专题**：[verl 与 AReaL：RL 框架架构选型指南](../training-infra-roadmap/topics/rl_framework_selection.md)——包含架构、优劣、当前选型矩阵、公平 benchmark 和 2 分钟回答。
- **项目证据或知识边界**：你分别有 verl RLVR 和 AReaL Agentic RL 项目，是强项目证据；说明“当时评估的版本”和公司二次开发。对 slime、ROLL 只说当时评估维度与选择，不编造没有记录的排名或缺陷。
- **高概率追问**：verl v0.9 后差异是否还成立？所谓“框架重”具体体现在哪里？AReaL 项目链路与 2.x 有何区别？同一任务怎么做公平选型 benchmark？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：你最初如何比较 verl、slime、ROLL？两个框架的核心思想有什么差别，为什么不同阶段做了不同选择？

- **面试官意图**：评估你的选型方法、版本意识和二次开发判断；也会验证你是否只是同时列出多个热门框架。

- **危险回答**：“AReaL 异步、verl 同步”；“AReaL 所有方面更先进”；把当时版本结论外推到当前 slime/ROLL；只用社区 benchmark，不讲团队已有资产和改造成本。

</details>

↩ [返回本 Part 导航](#part-iv) · ↑ [返回面试速查控制台](#interview-console)

<a id="areal-02"></a>
#### AREAL-02｜AReaL 如何控制异步训练的 off-policyness？（P0，18 分钟）

- **直接回答（60–90 秒）**：

  > 异步的风险是生成还在用旧策略，Trainer 已经更新。AReaL 一方面按版本与消费进度限制生成容量，避免无限超前生产；另一方面在项目 online 路径里检查 cohort 的版本差，过旧数据不能直接交给训练。版本差只是代理，同样落后一步，真实分布偏离也可能不同，所以还要看 logprob ratio、correction/masking 和效果评测。阈值为零时，项目会配合 strict drain 和更新边界退化到同步；正阈值允许更多重叠。若长轨迹跨版本续跑，必须保存实际生成 token 的 behavior 信息，不能只用一个 session 创建版本代替。

- **配置展开**：`max_head_offpolicyness` 同时影响 capacity/admission 与项目 ready-time stale gate；capacity 计数不是逐 token 的真实 policy divergence。权重版本检查、token logp 对齐与算法 correction 是互补门禁，不是设置一个阈值就自动严格 on-policy。

- **项目证据或知识边界**：你有 staleness manager、policy version 和 rejection diagnostics 经验；不要引用官方“通常 2–8”当作项目最优值。
- **高概率追问**：manager head drift 和真实 behavior staleness 区别？stale 样本直接丢弃会有什么系统后果？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：`max_head_offpolicyness`、policy version 和 partial rollout 如何协同？

- **面试官意图**：验证你是否理解 async 的算法代价，而不只是吞吐收益。

- **危险回答**：把 off-policy 只当数据过期问题；认为版本差 1 的所有 token 偏差相同；忽略 throughput-quality frontier。

</details>

↩ [返回本 Part 导航](#part-iv) · ↑ [返回面试速查控制台](#interview-console)

<a id="areal-03"></a>
#### AREAL-03｜AReaL 2.0 的微服务化对 Agentic RL 有什么价值？（P0，18 分钟）

- **直接回答（60–90 秒）**：

  > 这些服务需要的资源和失败方式不一样。Inference 主要受 KV cache、batching 和长尾影响，Agent 更多在等工具和 sandbox，Training 则需要密集计算与 collective，权重更新负责跨布局搬运参数。拆开以后，可以分别扩容、替换后端或处理故障，不必让整个系统一起变化。但状态不能因此丢失：请求身份、背压、版本发布和恢复协议仍要贯通。我会把微服务看成管理不同生命周期的手段，而不是“拆开自然更快”；如果规模还小、边界不稳定，额外运维成本也可能不划算。

- **项目证据或知识边界**：AReaL 2.0 发布晚于你部分项目；可以用项目中的 Gateway/online session 经验类比，但不要说项目天然就是完整 2.0。
- **高概率追问**：控制面和数据面如何分离？weight update 服务失败怎么办？HTTP 会不会成为瓶颈？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：training、inference、agent、weight-update 为什么要拆开？

- **面试官意图**：考系统边界、扩缩容、故障隔离和当前框架演进敏感度。

- **危险回答**：“微服务更解耦、更高性能”而没有状态一致性设计；忽略服务间背压和恢复。

</details>

↩ [返回本 Part 导航](#part-iv) · ↑ [返回面试速查控制台](#interview-console)

<a id="areal-04"></a>
#### AREAL-04｜如何证明 generated trajectory 最终真的产生了梯度？（P0，18 分钟）

- **直接回答（60–90 秒）**：

  > 我会给 trajectory 一个稳定 ID，把生成、reward、导出、训练消费和 loss-active 状态连起来。生成后没消费，不一定浪费，可能还在排队、等同组成员，或者不在当前统计窗口；进入 Trainer 也不代表产生 policy gradient，mask 可能全零，advantage 也可能为零。因此要分别记录 trajectory 数和有效 token 数，再按 stale、失败、过滤、等待等原因归因。这个链路证明的是样本是否进入实际训练信号；不能只拿 generated 减 consumed，就把差值都称为浪费。

- **项目 tracing 展开**：底稿曾闭环 `223 admitted→180 generated/rewarded→96 exported→96 consumed`，另有 2 条 compact-filtered 样本消耗生成 token 但不产生梯度。应以原始 ID join、mask/advantage 和窗口定义解释，不能把这些累计数直接转成长期利用率。

- **项目证据或知识边界**：这是项目底稿中的直接证据；如不可对外披露精确数字，保留方法和比例定义。
- **高概率追问**：如何处理 microbatch reorder？tracing 本身会不会拖慢？最终 drain 时 waiting 样本算什么？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：为什么 `generated - consumed` 不能直接叫作浪费？

- **面试官意图**：考数据 lineage、样本利用率定义和跨系统可观测性。

- **危险回答**：用队列长度代替 lineage；只追踪 trajectory 数而不追踪 token；忽略 tracing overhead 对 A/B 的污染。

</details>

↩ [返回本 Part 导航](#part-iv) · ↑ [返回面试速查控制台](#interview-console)

<a id="areal-09"></a>
#### AREAL-09｜结合代码仓说说你对 Gateway 做了哪些改造（P0，20 分钟）

- **直接回答（60–90 秒）**：

  > 基础 online proxy 和 session/cohort 链路是团队已有的，我主要改训练相关的控制逻辑。第一，让每个训练 step 按计划拿到不同领域的完整 group，避免哪个领域来得快就占满 batch；配额进度还要与 checkpoint 恢复一致。第二，把 reward、session 结束和失败清理放进明确状态机，避免奖励写错或轨迹丢失。第三，处理超时与重试：有远端副作用不确定的请求必须沿用原身份，只有证明尚未绑定的配额拒绝才能换任务重排。
  >
  > 我还为重排引入的吞吐损失加了保护，并用 fault-injection 验证失败路径。代码能证明机制，但没有统一 benchmark 的部分，我只说实现了吞吐保护，不说性能已经恢复。

- **代码与不变量展开**：

  1. 团队基础提交 `64adce36` 已有 Proxy Worker、InteractionCache、CohortManager 与 trainer consumer，不归为个人从零实现。
  2. exact quota：Trainer 生成本 step 的 domain plan，Gateway 通过 reservation → claim → session → export 绑定 domain/worker/step；optimizer、weight sync、model save 成功后才内存 commit，再由 recovery checkpoint 持久化 fairness cursor。commit 前失败不推进；commit 后但 checkpoint 落盘前失败，从上次持久状态重放，不静默跳过配额。
  3. reward/session：权威 reward identity 与 session lifecycle fail-closed，兼容 reward/end 到达顺序；rejected cohort 的 active sibling 仍须正确 terminal cleanup。
  4. liveness/safe retry：domain lock 内不 long-poll，小 RPC 用 bounded timeout，group-size/wrong-domain fail-fast；只有尚未绑定且确定无远端副作用的结构化 quota miss 可换身份 requeue，模糊 408/429/5xx 沿用 identity 重试。
  5. goodput protection：调整 requeue throttle、worker 和 partial deadline，控制 queue-rotation tax，让 sibling co-arrival 不轻易触发半组超时；配置变更不是性能实验。

- **代码证据归类**：exact quota `10a3e264/9979a0f6` 是同一能力的演进，不重复算成果；reward/session `c83de5fa/e7373e8b/afb1882c`；liveness/safe retry `eb8bd492/1162029d/b117b570/690816eb/30ab40c4`；goodput protection `21bb4862`。
- **深入阅读**：[项目 Gateway 二次开发：原始能力、状态机、safe retry 与证据边界](../training-infra-roadmap/topics/agentic_rl.md#project-gateway-ownership)。
- **项目证据或知识边界**：强调个人是上述控制逻辑和验证的 owner，不把团队已有 online proxy/cohort 架构说成从零自研，也不拿 commit message 代替性能实验。
- **高概率追问**：为什么 quota plan 要等 optimizer、weight sync、model save 后才 commit？commit 与 recovery checkpoint 之间失败如何恢复？408/429 为什么不能直接换 task 重试？requeue 为什么会伤害 goodput？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：原始 Gateway 已经有什么？你亲自修改了什么控制逻辑，系统行为发生了哪些变化？

- **面试官意图**：核对代码级 ownership，判断你能否从 HTTP proxy 上升到训练一致性、调度公平性与 liveness。

- **危险回答**：“我重写了 Gateway”；只讲加接口，不讲 invariant；所有失败都随机换 worker 重试；用 PR 数量代替系统结果。

</details>

↩ [返回本 Part 导航](#part-iv) · ↑ [返回面试速查控制台](#interview-console)

<a id="areal-11"></a>
#### AREAL-11｜AReaL 的 XCCL 与 disk 权重同步有什么区别？为什么项目最终选择 XCCL？（P0，15–18 分钟）

- **直接回答（60–90 秒）**：

  > 两者都是把更新后的 actor 权重发布给推理引擎，不是训练恢复 checkpoint。XCCL 把参数转换成 bucket，通过参与发送的 trainer rank 与 rollout rank 之间的通信组直接传输，再由推理侧 refit；它省去落盘和文件加载，但对通信组、布局与后端支持要求更高。disk 先导出到带版本的临时 HF 目录，再让推理侧加载，更方便检查和重试，但要付存储与加载开销。我们在固定项目 workload 下最终选 XCCL，因为测到的权重更新更快；不把这个结论外推到所有拓扑。无论哪条路径，加载成功前都不能把新版本标记成可服务。

| 维度 | XCCL | disk |
|---|---|---|
| 数据路径 | trainer sender rank(s) → collective/bucket transfer → rollout refit | trainer → 临时 HF transfer path → rollout loader/refit |
| 优势 | 少一次落盘与解析，低延迟，适合高频同步 | 解耦清晰，产物可检查，失败后易重载 |
| 代价 | 建组、rank 映射和后端支持复杂；部分失败要防混合版本 | 共享存储带宽、metadata/小文件、load 和清理可能暴露在关键路径 |
| 故障观察 | group 建立、collective hang、bucket/checksum、各 replica active version | export 完整性、manifest/version、文件可见性、loader/refit、残留目录 |

- **版本/支持边界**：在本地项目分支中，actor–rollout colocate 显式要求 disk；这不等于所有 colocate role 都只能用 disk，ref/critic 的共置条件不同。该分支的 SGLang LoRA 路径不支持 XCCL。XCCL group 包含参与传输的 trainer sender rank(s) 和 rollout ranks，不是默认把全部 trainer ranks 都拉进一个组。
- **深入阅读**：[AReaL XCCL 与 disk 权重同步：数据路径、状态机与选型](../training-infra-roadmap/topics/agentic_rl.md#areal-weight-sync-xccl-disk)；相邻问题：[VERL-03 训练到 rollout 权重同步](#verl-03)、[AREAL-06 原子发布与回滚](#areal-06)。
- **项目证据或知识边界**：可以说“verl/AReaL 在项目固定 workload 下最终都采用 XCCL，权重同步更快”；未形成统一公开 benchmark 时不报倍数，不把 disk 临时权重目录说成训练恢复 checkpoint。
- **高概率追问**：为什么 set_version 必须在传输成功后？部分 rollout rank 失败怎么办？何时宁可选 disk？colocate 为什么可能限制传输路径？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：两条链路分别传什么，如何切版本，各自适合什么资源布局？

- **面试官意图**：检查你是否真正理解训练态到推理态的权重搬运、故障边界和项目选型，而不是只记住“XCCL 更快”。

- **危险回答**：“XCCL 就是 NCCL，一定比 disk 快”；把 version 当成另一份权重；把所有 trainer rank 都说成 sender；把 disk transfer 和持久 checkpoint 混为一谈。

</details>

↩ [返回本 Part 导航](#part-iv) · ↑ [返回面试速查控制台](#interview-console)

<a id="resume-13"></a>
#### RESUME-13｜CUDA Graph 为什么能让 Agentic RL decode 加速 6–8x？（P0，12 分钟）

- **直接回答（60–90 秒）**：

  > decode 每一步只处理少量新 token，却反复执行很多短 kernel。小 batch 时，CPU 提交和框架调度开销可能让 GPU 出现空隙。CUDA Graph 先捕获一组 kernel 与依赖，之后更新静态输入 buffer、回放执行图，减少重复提交开销；它没有凭空减少 GEMM 的计算量。
  >
  > continuous batching 虽然动态变化，但引擎可以把请求映射到预捕获的 bucket，靠 padding 和更新元数据复用图。不能落入支持路径时才回退 eager。我的 AReaL 9B、128K 项目记录是 decode 加速 6–8x；prefill、工具、reward、排队和训练都不在这个分母里，所以不能说 E2E 也快 6–8x。

- **capture/refit 展开**：token id、position、KV block table 等内容可以变，但图引用的地址及执行约束需要保持有效；graph private pool 也会占显存。权重 refit 后若仍原地写入受支持的地址，不必一律 recapture；地址或图结构失效则需要重新 capture。收益大小要结合 graph coverage、CPU launch gap 和计算瓶颈判断。

![CUDA Graph 将逐 kernel 提交变为静态执行图回放](../training-infra-roadmap/assets/topics/cuda-graph-decode.svg)

- **两组数字不得混用**：AReaL Agentic RL decode 是 `6–8x`；另一个 verl 35B RLVR workload 的 decode 记录为约 `14x`。它们的模型、框架、batch/concurrency、graph coverage 和统计窗口不同，不能拼成同一结论。
- **验证方法**：同模型、gen-TP、batch/concurrency、输入/输出长度和 sampling 配置，warmup 后比较 eager 与 graph 的 decode-only latency/token throughput；记录 graph hit/fallback、CPU launch gap、GPU utilization、private-pool 显存与 E2E rollout/step time。
- **深入阅读**：[Agentic RL 中的 CUDA Graph：capture、bucket、失效与指标边界](../training-infra-roadmap/topics/agentic_rl.md#cuda-graph-decode)。
- **项目证据或知识边界**：对外主数字使用最新简历 `6–8x`；`14x` 只能在明确说“另一项 verl 35B RLVR workload”时补充。
- **高概率追问**：continuous batching 为什么还能用 graph？权重同步后是否必须 recapture？graph 为什么可能额外 OOM？GPU 已经 compute-bound 时收益多大？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：它消除了什么开销？动态 batch/KV cache 如何 capture？为什么不能说 E2E 也快 6–8x？

- **面试官意图**：验证 GPU execution model、推理引擎接入和局部指标边界；也会核对最新版简历数字。

- **危险回答**：说 CUDA Graph 融合了所有 kernel 或减少模型计算量；把 decode 倍数外推到 rollout/E2E；忽略 graph miss、静态地址和额外显存。

</details>

↩ [返回本 Part 导航](#part-iv) · ↑ [返回面试速查控制台](#interview-console)

<a id="resume-19"></a>
#### RESUME-19｜Gateway 如何通过流式补位、均衡分发和失败管理把 Rollout 吞吐提升 60%？（P0，20 分钟）

- **直接回答（60–90 秒）**：

  > 原来的调度更接近一波一波发请求：短请求结束后槽位空着，还在等这一波的长尾。我们改成完成一个、在容量允许时马上补一个，维持有效并发；新任务均衡分到 worker，已经绑定的多轮 session 保持原路由。失败请求则明确区分重试与终结，避免槽位泄漏或同组轨迹长期凑不齐。这三项是联合改造，不能分别拆收益。
  >
  > 项目记录的 Rollout 阶段平均推理吞吐提升了 60%，Rejected Group 从 33.18% 降到 2.73%。前者不是训练端到端提升；后者的原始分母、原因分布和对照窗口还待核验，所以目前只说观察到拒绝比例下降，不能把下降全部归因于长尾改善，更不能直接说训练质量提高。

- **调度与失败状态展开**：

  - 流式补位是 request/rollout slot 释放后从有界 pending queue 补任务，不是 HTTP token streaming；worker capacity/backpressure 决定能否补位。
  - 对应项目历史分支的新 route 用 round-robin 分配 Proxy Worker，不是实时 least-load；reservation、cohort、claim、session 一旦绑定保持 affinity，后续请求不随意迁移 InteractionCache/session 状态。
  - in-flight、completed、retryable、terminal/aborted 必须区分；网络超时不证明远端已取消，不能未经确认就释放远端资源并换身份。模糊失败沿用 identity 重试；确定终结后清理 sibling/session 与 capacity。

![Gateway 流式补位、均衡分发与失败请求分流](../training-infra-roadmap/assets/topics/gateway-streaming-refill.svg)

- **待观测验证的机制链**：及时释放/补位应减少空槽；均衡新路由与保持 affinity 应减少 worker skew 和状态迁移；失败显式终结/幂等重试应减少 capacity leak 与 incomplete cohorts。分别用 active concurrency、worker skew、reason breakdown 和 ready-cohort rate 检验，不能仅凭总拒绝率认定每条链都已证实。
- **与 AREAL-09 的区别**：本题回答早期 Rollout 性能重构和量化结果；[AREAL-09](#areal-09) 回答后续 exact quota、reward/session fail-closed、safe retry 与 liveness 的个人代码 ownership。不要把不同阶段的提交和收益强行归为一次 A/B。
- **深入阅读**：[Gateway 调度：从 wave barrier 到流式补位](../training-infra-roadmap/topics/agentic_rl.md#gateway-streaming-refill)。
- **项目证据或知识边界**：保留记录值 `+60%`、`33.18%→2.73%`，但需补原始统计协议；分母若包含不同的 admitted、terminal 或 attempted cohorts，就不能直接横比。数值相减为 `-30.45pp`，若同口径相对降幅约 `91.8%`；算术换算不能替代分母核验。三项缺少独立消融，不拆贡献。
- **高概率追问**：round-robin 为什么不等于实时 least-load？session affinity 为什么优先于重新均衡？失败后为什么必须复用 identity？partial group 能不能训练？补位会不会让样本更 stale？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：三项机制分别解决什么？Rejected Group 为什么会从 33.18% 降到 2.73%？

- **面试官意图**：核对你是否理解调度 critical path、cohort 完整性和失败状态机，而不是只复述结果数字。

- **危险回答**：把 token streaming 当调度补位；每次失败随机换 worker；只追求并发而无 bounded queue/backpressure；把 Rejected Group 下降直接等同于训练效果提升。

</details>

↩ [返回本 Part 导航](#part-iv) · ↑ [返回面试速查控制台](#interview-console)

<a id="resume-14"></a>
#### RESUME-14｜Prefix Cache 如何工作，什么时候有效？（P0，10 分钟）

- **直接回答（60–90 秒）**：

  > Prefix Cache 缓存的是已经算过的前缀 KV，不是最终答案。对 causal Transformer，在模型权重和相关输入条件相同时，相同 token 前缀会产生可复用的 KV；新请求命中后不用重算这段，只计算新增 suffix。多轮对话重复历史、多个请求共享 system prompt 或长文档时比较有效，主要节省 prefill，不直接减少生成新 token 的 decode 工作。
  >
  > 命中要求 token 和相关模型状态一致，不能只看字符串相似。RL 频繁更新权重后，旧 KV 必须失效或按版本隔离，否则用错缓存会影响输出与 logprob。我们记录的 44% 是 prefill 阶段耗时下降；如果主要时间在 decode、工具或完整 cohort 等待，端到端收益就会小得多。

- **实现展开**：引擎通常以 block/hash 或 radix tree 组织可复用前缀；匹配身份至少涉及精确 token prefix，具体后端还需处理 adapter、multimodal 输入及其他影响 KV 的配置。paged allocation 管理存储块，prefix caching 决定哪些已有 KV 可以复用，二者不是同一个机制。session affinity 有助于命中本地缓存，但不能替代 cache key 与版本校验。[vLLM 官方说明](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)。
- **反直觉追问：命中率升高，为什么 E2E 可能仍变慢？** 固定完整 trajectory、生成路径与终止条件时，缓存本身不会让总 token 或工具交互变多。若观察到变慢，先查 workload/采样路径、admission、并发、wall-time budget、超时完成率和样本组成是否变化，再查缓存管理开销与资源竞争。只有这些额外条件成立，才可能出现“更多 episode 完成后期长回合、总工作量上升”等二阶效应；没有对照实验不能把它当作既定因果。按相同 task/seed/logical batch 比较 update interval、有效 token、episode completion 与下游效果。[APC 的作用边界](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/#limits)。
- **项目证据或知识边界**：底稿记录 prefill 耗时下降 44%，不能单独声称 E2E 收益；cache 命中率、节省的 prefill token 和端到端 goodput 是不同指标。
- **高概率追问**：cache key 包含什么？weight refit 后如何失效？共享 prefix 是否要求同一个 session？命中率与节省计算量为什么不同？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：缓存的是什么？为什么相同前缀可以复用，主要省哪一段计算？权重更新后还能继续命中吗？

- **面试官意图**：先验证 prefill/decode 与 KV cache 基础，再追问 Agentic RL 的失效、调度与 E2E 收益。

- **危险回答**：说缓存的是答案；相似字符串就能命中；跨权重版本复用旧 KV；命中率越高系统一定越快。

</details>

↩ [返回本 Part 导航](#part-iv) · ↑ [返回面试速查控制台](#interview-console)

### P1 深挖｜面试官继续追问

<a id="resume-15"></a>
#### RESUME-15｜Rejected Group 从 33.18% 降到 2.73%意味着什么？（P1，10 分钟）

- **直接回答（45–60 秒）**：

  > 这个比例下降，首先表示被记为 rejected 的 group 占比变了，还不能直接等于“坏样本更少”或“模型效果更好”。我会先确认分母是哪些 group、是否去重、统计窗口怎么处理未完成样本，再把拒绝拆成半组超时、整体超时、stale、reward 缺失和后端失败。只有相同口径的原因分布改善，才能具体归因。我们的记录是 33.18% 到 2.73%，相差 30.45 个百分点，但原始分母和原因分布仍待核验；不能为了降低比例，就放宽旧样本或不完整 group 的准入。

- **分母门禁**：正式报告应写 `N_rejected / N_eligible`，并说明 eligible 是唯一 admitted cohorts、同窗口 terminal cohorts，还是其他集合；这些集合不能混用。还应报告绝对 cohort 数、未完成/跨窗口样本、reason breakdown 与 effective training tokens，不只给百分比。
- **项目证据或知识边界**：保留 `33.18%→2.73%（-30.45pp）` 作为待原始协议复核的项目记录；联合改造和指标边界见 [RESUME-19](#resume-19)，不得把总降幅全归因于 incomplete groups。
- **高概率追问**：partial group 能不能训练？uniform reward group 怎么处理？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：group 为什么会被拒绝，降低拒绝率是否一定提高训练质量？

- **面试官意图**：验证 group-based RL 的数据完整性和指标解释。

- **危险回答**：把 rejected 全称为坏样本；只优化比例不看原因分布。

</details>

↩ [返回本 Part 导航](#part-iv) · ↑ [返回面试速查控制台](#interview-console)

<a id="areal-05"></a>
#### AREAL-05｜Partial Rollout 的收益和风险是什么？（P1，10 分钟）

- **直接回答（45–60 秒）**：

  > Partial rollout 是让未完成的轨迹可以暂停并续跑，不必每次权重更新都等它结束或丢掉已有生成。收益是减少长尾等待和重复工作，风险是一条轨迹可能由多个 policy versions 生成，所以必须保存 segment 边界、token 与 behavior logprob/version，环境状态也要能继续。暂停后最终凑成完整样本再训练，与直接拿截断片段训练，是两种不同算法契约；后者还要定义 reward、bootstrap 和 mask。它也不等于 GRPO 可以随意拿半个 group 训练。

- **项目证据或知识边界**：有 online session/trajectory 经验；如果项目没启用跨版本 partial，明确为机制理解。
- **高概率追问**：environment state 怎么恢复？segment reward 如何分配？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：一条 trajectory 跨多个 policy version 是否还能训练？

- **面试官意图**：考长 trajectory 调度和算法语义。

- **危险回答**：把 partial rollout 当字符串续写；整条只记一个 policy version。

</details>

↩ [返回本 Part 导航](#part-iv) · ↑ [返回面试速查控制台](#interview-console)

<a id="areal-06"></a>
#### AREAL-06｜权重同步如何做到原子、可观测、可回滚？（P1，10 分钟）

- **直接回答（60–90 秒，设计方案）**：

  > 如果让我设计，我会把权重发布分成准备、传输、校验、激活四步：先声明候选版本，worker 完成加载和必要校验后，才允许用这个版本服务。某个 replica 部分加载失败，就先隔离或停止它，不能继续用“半新半旧”的参数生成。记录每个 replica 的目标、已加载和活动版本，才能判断更新卡在哪一步。
  >
  > 回滚还取决于有没有保留旧权重。如果是原地 refit，又没有双缓冲，就不能承诺超时后旧版本自然还在；需要暂停服务，重新加载已知正确版本。我的项目证据覆盖 XCCL 同步、更新后推进 version 和诊断；跨 replica 原子切流、完整校验与自动回滚是这里的生产设计，不冒充已经全部实现。

- **设计展开**：可用 prepare/transfer/validate/commit 与 manifest、shape/dtype/checksum 校验构成发布协议；双缓冲允许保留旧版服务，但有显存成本，原地更新则需要 pause/refit/validate/resume。训练 batch 是否允许多个完整 behavior versions，由 async 算法与 lineage 约束决定；这与禁止 torn weights 是两件事。
- **传输实现追问**：具体的 XCCL 与 disk 数据路径、支持边界和项目选型见 [AREAL-11](#areal-11)；本题重点仍是跨 replica 的原子发布语义。
- **项目证据或知识边界**：已核实项目调用顺序为 pause admission/inference → weight transfer/refit → set_version；有 XCCL/NCCL broadcast 和 re-prefill diagnostics。仅凭这些不能宣称具备跨 replica 的双缓冲原子切流、checksum 全量验证和自动回滚。
- **高概率追问**：大模型双缓冲显存不够怎么办？滚动更新能否用于训练 rollout？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：部分 inference worker 更新失败时怎么办？

- **面试官意图**：检查分布式一致性和生产设计。

- **危险回答**：一次 broadcast 即原子；失败后简单重试而不看是否部分生效。

</details>

↩ [返回本 Part 导航](#part-iv) · ↑ [返回面试速查控制台](#interview-console)

<a id="areal-07"></a>
#### AREAL-07｜Online Proxy 与 session drain 为什么重要？（P1，8 分钟）

- **直接回答（45–60 秒）**：

  > 我区分关停 drain、严格同步 drain 和异步更新暂停。关停时先停止接收新 session，再让 in-flight 在 deadline 内完成或显式取消，并按支持的恢复协议处理队列与 cursor，不能直接 kill。项目中 staleness 阈值为零时，会在训练更新边界做 strict drain；正阈值的异步模式则保留 active/open sessions，权重更新时暂停新 admission 和推理执行，更新后恢复，再用 ready-time stale gate 与 token metadata 控制跨版本样本。每次更新都等所有长 episode 结束，会重新引入我们原本想减少的等待。

- **恢复边界**：session drain 不等于外部 Tool/Sandbox 状态已经可恢复；只有实际持久化的 session、queue、cursor 和环境状态才可以宣称支持续跑，否则要明确取消并按 identity/replay 协议重建。
- **项目证据或知识边界**：你做过 online session drain 和 shutdown contract；可作为直接证据。
- **高概率追问**：客户端断线怎么处理？retry 如何幂等？session affinity 丢失会影响 cache 吗？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：外部 agent client 接入训练时如何安全更新/关停？

- **面试官意图**：评估在线 Agentic RL 的 session 生命周期管理。

- **危险回答**：直接 kill server；不区分 request 完成与 trajectory 完成。

</details>

↩ [返回本 Part 导航](#part-iv) · ↑ [返回面试速查控制台](#interview-console)

<a id="areal-10"></a>
#### AREAL-10｜外部 Agent 如何通过 OpenAI-compatible Gateway 接入训练？（P1，12 分钟）

- **直接回答（60–90 秒）**：

  > Agent 把模型 client 的地址指向 Gateway，就可以继续用熟悉的模型 API，工具和 sandbox 的控制流仍在外部。但接入训练还多一层生命周期：管理端先创建 session，Agent 带 session key 多轮请求，任务结束后提交 reward 并结束 session。Proxy 记录 token、behavior logprob 与版本，同组数据完整且通过新鲜度检查后，Trainer 才能导出、计算 loss 并更新。因此“API 能调用”只是第一步；session 身份、重试幂等、reward 权威来源和 group 完整性才决定这些交互能否正确训练。

- **项目 API 展开**：admin key 调 `/rl/start_session` 获取 session ID/key；session key 请求项目分支的 `/chat/completions`、`/responses` 或 `/v1/messages`，终结时写 `/rl/set_reward`、`/rl/end_session`。外层 ingress 是否添加 `/v1` 前缀以部署路由为准，不能把 SDK 常用路径直接当服务端事实。InteractionCache 保存交互数据，CohortManager 管理 group/capacity/staleness，Trainer export/tensorize 后消费。

- **项目链路**：`External Agent/Evals → Gateway admission → session-bound Proxy Worker → vLLM/SGLang → InteractionCache → rewarded/ended cohort → trainer export/update`。项目二次开发还在 admission、safe retry 和 lifecycle 上增加了 [AREAL-09](#areal-09) 的约束。
- **深入阅读**：[外部 Agent 接入协议与 online proxy/cohort 数据流](../training-infra-roadmap/topics/agentic_rl.md#external-agent-gateway)。
- **项目证据或知识边界**：这是项目使用的 online proxy/cohort 路径；AReaL 2.1 的具体 API 文档可用于解释协议，但不要把后续独立微服务实现倒推到项目版本。
- **高概率追问**：为什么要 admin/session 两级 key？客户端重试如何不生成重复 trajectory？reward 先于 end 或晚于 end 怎么办？Tool state 由谁恢复？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：框架外的 Agent/Tool/Sandbox 如何接入 AReaL？一次 session 从创建到成为训练样本经历什么？

- **面试官意图**：检查 API 兼容层、session 状态、trajectory 数据和训练消费之间是否真正闭环。

- **危险回答**：“兼容 OpenAI API，所以任意 Agent 可直接训练”；把 Tool/Sandbox 说成 Gateway 内部固定模块；忽略 reward/session/group 的状态机。

</details>

↩ [返回本 Part 导航](#part-iv) · ↑ [返回面试速查控制台](#interview-console)

<a id="areal-08"></a>
#### AREAL-08｜FUNCTIONAL、NUMERIC、EFFICACY 三层门禁分别是什么？（P1，8 分钟）

- **直接回答（45–60 秒）**：

  > FUNCTIONAL 问“流程是否闭环”：数据能进来、更新能完成、失败能按约定恢复。NUMERIC 问“是不是算对了”：同一 token、mask、logprob、loss 和跨 rank 结果能否对齐，same-weight 信号是否符合预期。EFFICACY 才问“模型有没有变好”：用无污染、同协议的 held-out evaluation 对比目标领域与 General 回归。跑完一百步最多提供部分 functional 证据，loss 不 NaN 也不能通过 numeric；训练 reward 上升更不能替代 efficacy。

- **项目证据或知识边界**：这是 MOPD 项目的核心方法论。当前双 Teacher 已有“SWE、Terminal 双域提升且 General 不下降”的 EFFICACY 方向性结论；统计细节仍按证据卡补齐，三层门禁不能因为已有结论而省略。
- **高概率追问**：每层最小测试是什么？什么时候可以进入长跑？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：为什么系统跑完 100 step 仍不能证明算法有效？

- **面试官意图**：考严谨性与研发验收方法。

- **危险回答**：用 loss 不 NaN 通过 numeric；用训练 reward 通过 efficacy。

</details>

↩ [返回本 Part 导航](#part-iv) · ↑ [返回面试速查控制台](#interview-console)

### P2 选学｜时间允许再补

<a id="p2-04"></a>
#### P2-04｜设计一个 256K、多轮 Agentic RL 平台（P2，12 分钟）

- **直接回答（60–90 秒）**：

  > 我会先确认并发、长度分布、训练频率和故障恢复要求，再把平台分成外部 Agent/环境、推理池、轨迹与 reward 存储、Trainer 和权重发布。任务和 trajectory 用稳定身份贯穿各层，队列有界，session 保持路由；样本能否消费由完整性、reward 与 staleness 决定。长上下文下再分别估算推理 KV 与训练 activation，选择 TP、CP、batch 和重计算。故障设计上，我会明确谁保存权威轨迹、外部环境能否恢复、部分权重更新失败如何隔离，并用有效训练 token、ready-cohort wait 和 E2E update interval 验收，而不是只画正常路径。

- **项目证据或知识边界**：高度贴合你的经历；未知 SLA/规模时先提问，不急于报架构。
- **高概率追问**：哪层是 source of truth？外部 env 不稳定怎么办？如何多租户？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：从 API、调度、数据、正确性、恢复和指标设计。

- **面试官意图**：综合考高级工程师系统设计与取舍。

- **危险回答**：画一条理想流水线无失败状态；只谈模型并行；没有容量模型。

</details>

↩ [返回本 Part 导航](#part-iv) · ↑ [返回面试速查控制台](#interview-console)

### 本 Part 追问路线

AReaL online 链路 → ready-cohort wait/长尾 → staleness 与 weight version → XCCL/disk 权重发布 → CUDA Graph/Prefix Cache → TILE baseline → MOPD 三层验收。

---

<a id="part-v"></a>
## Part V｜通用 Infra 与生产排障

**学习目标**：把训练与 rollout 项目上升为可迁移的生产能力：训练数值异常、通信协议、故障定位、恢复、推理容量与可观测性。

**本 Part 导航**：Core：[通信算子](#infra-04)；P0 扩展：[训练数值异常](#train-anomaly-01) · [万卡规模效应](#infra-09) · [NCCL 与恢复排障](#infra-03)；P1：[精度对齐](#resume-12) · [64 卡并行选型](#infra-05) · [推理与 KV cache](#infra-06) · [可观测性](#infra-07) · [Checkpoint 状态](#infra-08)；P2：[性能瓶颈定位](#p2-03)。

**Coding 实战**：[PyTorch MHA 与 `N×N` 矩阵原地顺时针旋转](2026-09-interview-coding.md)（独立题单，不计入本 Part 题量）。

### Core｜最高优先入口

<a id="infra-04"></a>
#### INFRA-04｜常见通信算子执行什么操作，分别用在哪里？（P0，15 分钟）

- **直接回答（60–90 秒）**：

  > 我会从输入输出和参与的 process group 来区分。AllReduce 把各 rank 的输入规约后，让所有 rank 得到完整结果，典型是 DP 梯度同步；ReduceScatter 同样做规约，但每个 rank 只留下自己的分片；AllGather 则把这些分片收集到每个 rank，本身不做求和。比如 Megatron Distributed Optimizer 是先对梯度 ReduceScatter，各 rank 更新本地 optimizer shard，再 AllGather 更新后的参数。AllToAll 是每个 rank 给不同 peer 发不同数据，MoE 用它做 token dispatch 和 combine。Broadcast 是 root 向所有 rank 复制同一份输入，Reduce 是只有 root 拿到规约结果；Scatter 和 Gather 分别是 root 分发和收集不同分片。PP 的 activation 和反向 gradient 通常用 Send/Recv。具体到一条通信，我会先确认传的是梯度、参数、activation 还是 token，再确认 group 和输出布局。

- **输入输出速查**：

  | 算子 | 每个 rank 最终得到什么 | 典型场景 |
  |---|---|---|
  | Broadcast | root 的同一份 tensor | 参数/metadata 初始化 |
  | Reduce | 只有 root 得到规约结果 | root 汇总统计量 |
  | AllReduce | 每 rank 得到完整规约结果 | classic DP gradient、TP partial sum |
  | Scatter | 每 rank 得到 root 输入中的不同 shard | root 分发固定分片 |
  | Gather | 只有 root 得到各 rank shard 的拼接 | root 收集结果/调试 |
  | AllGather | 每 rank 得到所有 shards 的拼接 | parameter/activation 重建 |
  | ReduceScatter | 每 rank 得到规约结果中的一个 shard | Distributed Optimizer/FSDP gradient、SP |
  | AllToAll | 每 rank 收到来自所有 peers 的不同分片 | MoE dispatch/combine、layout transpose |
  | Send/Recv | receiver 得到指定 sender 的 tensor | PP activation/gradient、ring CP |

- **框架生命周期**：

  ```text
  classic DP:
      local gradient -> AllReduce -> every replica updates the same parameter

  Megatron Distributed Optimizer:
      gradient ReduceScatter -> local optimizer update -> parameter AllGather

  FSDP FULL_SHARD（典型路径，取决于 strategy/reshard policy）:
      pre-forward parameter AllGather
      -> optional post-forward reshard
      -> pre-backward parameter AllGather（若 forward 后已 reshard）
      -> post-backward gradient ReduceScatter / reshard

  TP -> layer-level AllReduce / AllGather / ReduceScatter
  PP -> Send / Recv
  CP -> KV P2P / AllGather / AllToAll
  EP -> token dispatch/combine AllToAll 或 variable-count exchange
  ```

- **两个边界**：`AllReduce = ReduceScatter + AllGather` 只在 count 可分片、dtype、reduction op 和 layout 兼容时数学等价，底层不一定机械调用两个 API，浮点归约顺序也不保证 bitwise 一致。NCCL 2.31.2 有 fixed-count `ncclAlltoall`，但没有通用 `ncclAlltoallv` host API；框架/dispatcher 的 AllToAllV 必须校验每对 peer 的 send/recv count。PyTorch `dist.barrier()` 是框架同步语义，也不能简单当作 NCCL 通用 host Barrier API。
- **正确性与性能**：正确性先查 group membership、collective 顺序、count/shape、dtype/op/root/peer、buffer lifetime 和 stream wait；性能再看消息大小、频率、ring/tree/topology、p95/p99 和 exposed communication。异步发起不等于已经与计算重叠。
- **深入阅读**：[NCCL 与分布式通信算子：逐算子四卡示例、5D 映射和 hang 排障](../training-infra-roadmap/topics/nccl.md#collective-map)。
- **项目证据或知识边界**：你有 NCCL/XCCL、MoE AllToAll、weight sync 和大规模故障定位经验；若没有实现 NCCL kernel/算法，明确个人边界是使用、集成、性能分析和排障。
- **高概率追问**：Broadcast 与 AllGather 有何区别？为什么 RS+AG 与 AR 只说语义等价？gradient 和 parameter 分别在哪一步通信？AllToAllV 如何避免 count 不一致 hang？ring/tree 怎么选？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：请解释 Broadcast、Reduce、AllReduce、Scatter、Gather、AllGather、ReduceScatter、AllToAll 和 Send/Recv，并结合 DP/TP/PP/CP/EP、Distributed Optimizer/FSDP 说明场景。

- **面试官意图**：检查集合通信基本功、tensor 语义、process group 和训练生命周期；区分“背 API”与真正理解数据布局。

- **危险回答**：只背中文定义；把 gradient ReduceScatter 与 parameter AllGather 说反；认为 Broadcast 会收集每个 rank 的输入；把 Barrier 当修复 race 的万能方法；忽略所有 ranks 必须以一致协议调用 collective。

</details>

↩ [返回本 Part 导航](#part-v) · ↑ [返回面试速查控制台](#interview-console)

### P0 扩展｜首轮前应掌握

<a id="train-anomaly-01"></a>
#### TRAIN-ANOMALY-01｜loss 震荡、NaN、梯度爆炸或收敛慢怎么定位？（P0，18 分钟）

- **直接回答（60–90 秒）**：

  > 我先保留失败 step、异常 batch、rank 信息和最后一个正常 checkpoint，把复现条件固定下来。然后沿数据、forward、backward、optimizer 找首次分歧：先确认 token、label 和 mask 正确，再看哪一层最先出现非有限值或异常梯度，最后核对学习率、梯度累积和恢复的 optimizer 状态。如果问题只在分布式出现，我会对比 collective 前后的 tensor，但按照分片语义检查，不能要求所有 rank 的不同分片数值相同。定位后，用最小可复现并行规模、高精度或禁用可疑 fused kernel 做对照，再逐步恢复优化特性。验收看同一失败 batch 的数值、短窗口 loss/grad 和 held-out eval；loss 震荡还要排除数据分布和有效 token 数变化，不能只靠调小 LR。

- **追问时展开的核对细节**：现场证据还包括 optimizer/scheduler、RNG、data cursor、rank/host 和环境版本；forward 记录 activation/logits/loss 的 min/max/mean 与 NaN/Inf，backward 记录逐层 grad norm、零梯度与 finite status。Optimizer 核对 LR/warmup、有效 global batch、gradient accumulation、clip、Adam epsilon/betas 和 weight decay。分布式对照记录 sample ID、collective 前后 tensor 的 shape/layout；只有同一逻辑复制态或固定同输入的 debug 对照才比较 checksum，TP/PP/CP/EP 分片必要时重建后再比较。模型无法放入单卡时，保留复现所需的最小并行规模。

- **按症状分流**：
  - **立即 NaN/Inf**：优先查坏数据、除零/`log(0)`、softmax overflow、norm、低精度 cast、FP16 loss scale 和 fused kernel；BF16 通常不需要 GradScaler，FP16 应先 unscale 再做 gradient clipping。
  - **grad norm 突然尖峰**：定位首个异常 layer/step，检查异常 token 长度、loss normalization、梯度累积语义、学习率跳变和跨 rank reduce；clipping 是保护措施，不是根因解释。
  - **loss 震荡或收敛慢**：先确认 effective tokens、mask 和数据分布正确，再查 global batch/LR scaling、warmup/decay、optimizer state、样本重复污染、过强 regularization 或精度损失，并用 held-out eval 区分优化慢与数据/目标错误。
- **交叉排障**：峰值显存和 OOM 进入 [Megatron 显存账本](#infra-02)；collective hang、网络和 checkpoint 恢复进入 [NCCL/恢复排障](#infra-03)。本题聚焦数值和收敛，不重复两套系统故障正文。
- **项目证据或知识边界**：优先绑定你做过的精度对齐、长上下文 mask/logits、checkpoint crash 和大规模训练验收；没有真实案例的异常类型按诊断方法回答，不虚构生产事故。
- **高概率追问**：为什么 loss 正常仍可能训错？forward 正常、backward NaN 怎么办？gradient clipping 放在什么位置？如何复现仅某个 rank 出现的 NaN？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：训练能启动但数值异常或长期不收敛，你如何止损、缩小范围并证明修复有效？

- **面试官意图**：检查你是否能独立调试训练故障，区分数据、数值精度、梯度、optimizer、分布式一致性和算法问题，而不是靠试参碰运气。

- **危险回答**：直接降低 LR 或加大 clipping；只开 `detect_anomaly` 跑全量集群；看到最后报错 rank 就当根因；以“不再报错”代替数值与效果回归。

</details>

↩ [返回本 Part 导航](#part-v) · ↑ [返回面试速查控制台](#interview-console)

<a id="infra-09"></a>
#### INFRA-09｜万卡训练相比千卡以下有哪些规模特有问题？如何优化？（P0，20 分钟）

- **直接回答（60 秒）**：

  > 万卡规模主要放大三类问题：组件小概率故障累积成作业频繁中断；同步训练被慢 rank 拖住，局部长尾变成全局等待；启动、通信和 checkpoint 的集中并发形成跨机架拥塞。因此我会用长期 goodput 和 time-to-train 验收，同时看 MFU、MTBF、MTTR 和 checkpoint 开销。方案上，先用健康检查和 rank 级观测定位故障与 straggler，再做拓扑感知并行、分层启动和限速 IO，并验证 checkpoint 与自动恢复。我的直接训练规模证据是 X1 的 3K 卡长稳；完整万卡平台的控制面和容错设计，我按系统原理展开。

- **追问展开（3–5 分钟）**：

  | 规模化问题 | 为什么到万卡会质变 | 主要优化 | 必看指标 |
  |---|---|---|---|
  | 故障成为常态 | 若把单个故障单元在一个时间窗内出错概率记作 `p`，独立近似下全作业至少一处故障概率为 `1-(1-p)^N`；真实集群还存在机架、交换机、电源和软件版本导致的相关故障 | 训练前 health check、节点健康评分与隔离；heartbeat/first-failure detection；故障节点替换；从已验证 checkpoint 协调恢复；定期 recovery drill | job MTBF、MTTR、自动归因率、恢复成功率、丢失 step/GPU-hours |
  | straggler 放大全局尾部 | 同步训练近似满足 `T_step≈max(T_rank)`；单卡降频、NUMA/PCIe、NIC 重传、数据抖动或 MoE expert 热点都会拖慢所有 rank | 分阶段、分 rank 记录 p50/p95/p99/max；找 first divergence；隔离慢节点；平衡数据和 expert load；减少 noisy neighbor | step `p99-p50`、rank skew、collective p99、pipeline bubble |
  | 通信跨越多级拓扑 | 并行组会跨 NVLink/HCCS、节点、rail、机架甚至 pod；过订阅、路径冲突和动态 AllToAll 不均衡会让平均带宽失去意义 | topology-aware rank/group mapping；把高频、强耦合通信尽量限制在高速域；hierarchical collective、rail-aware routing；用 exposed communication 验证 overlap | scale efficiency、链路利用率、重传/丢包、collective p99、exposed communication |
  | 调度与启动形成控制面风暴 | 万级进程同时分配资源、拉镜像、读配置、rendezvous 和初始化 communicator；一个 late node 就可能卡住 gang scheduling | 分阶段启动和 health gate；镜像/依赖预热；分层编排、批量 metadata；确定性 rank mapping；timeout/fail-fast；预留替换节点 | allocation-to-first-step、各初始化阶段耗时、启动失败率、communicator init 时间 |
  | 数据与 checkpoint 形成 IO/metadata storm | 数千 worker 同时访问小文件、保存 shard 或恢复，会打爆 metadata service、网络和对象存储；checkpoint pause 的 GPU 成本被卡数放大 | 数据预分片与本地缓存；sharded/distributed async checkpoint；节点级聚合、分层落盘、限速/错峰；原子 manifest、checksum、data cursor 和恢复重分片 | dataloader p99、save pause/E2E time、restore time、存储带宽与 IOPS、重复/丢失样本 |
  | 故障症状远离根因 | 首个异常 rank 可能无日志，其他 rank 最终只报 NCCL/XCCL timeout；全量高频日志本身又会压垮观测系统 | 统一 job/step/collective sequence 与 rank/host/device/NIC 身份；分层 telemetry；超时保存 flight recorder；从 first bad event 而非 last error 归因；恢复后验证 loss/data/version 连续 | 检测时间、根因覆盖率、日志丢失率、恢复后首步/loss 连续性 |

- **统一优化框架**：用四个动作记忆稳定性闭环——**降低故障发生率 → 缩小故障影响面 → 缩短检测和定位时间 → 降低恢复与重算成本**；再用 topology 和 tail-latency 治理守住稳态性能。
- **为什么看 goodput**：峰值 throughput/MFU 只描述“跑起来时有多快”；`goodput = 已成功提交且有效的训练 token / 已分配 GPU wall-clock`，会把启动、checkpoint、故障停机、回滚重算和慢节点一起计入。万卡优化应同时报告 MFU/throughput 与 job MTBF、MTTR、checkpoint pause、有效训练时间占比。
- **项目映射与边界**：

  > TX、X1 所在集群总规模分别约 1.4 万卡和 1.2 万卡；我直接参与的规模证据是 X1 200B MoE 的 3K 卡连续稳定训练两个月。我负责模型侧跑通、profile、瓶颈归因、并行/算子/通信优化和规模回归，向对应团队提供稳定复现与 rank 级证据，再完成模型侧验收。集群、网络、编译器和底层集合通信由各自团队实现。我的经历能支撑模型侧长稳与性能判断，万卡控制面、存储惊群和全平台容错属于进一步的系统设计分析。

- **公开系统证据，不作为个人项目数字**：MegaScale 在 12,288 GPUs 上训练 175B 模型，并披露一个万卡生产作业数周内重启超过 100 次；Llama 3 论文披露 405B 训练最多使用 16K H100，在 54 天观测窗内发生 466 次中断，其中 419 次为非计划中断。它们共同说明故障处理和长期有效训练时间是万卡系统的一等指标。
- **待本人补证（不作为口述事实）**：从 X1 3K 经历中选一个确有证据的事件，记录“表面症状 → first bad event/rank → 故障域 → 本人提供的 profile/复现 → 对应团队修复 → 同 workload 回归 → 长窗口验证”。具体事件未确认前，保留上述系统方法回答，不补写 GPU/NIC/checkpoint 事故。
- **深入阅读**：[大规模训练稳定性与容错：从千卡到万卡](../training-infra-roadmap/topics/fault_tolerance.md#large-scale-training)；具体 collective hang 继续看 [INFRA-03](#infra-03) 与 [NCCL 专题](../training-infra-roadmap/topics/nccl.md#hang-diagnosis)。
- **高概率追问**：为什么 `T_step` 看 max 而不是平均？慢 rank 和网络拥塞怎么区分？TP/EP/CP/DP 如何映射拓扑？固定 world-size 与 elastic recovery 怎么选？checkpoint 间隔怎么定？如何避免恢复时击穿存储？goodput 怎么计算？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：为什么万卡训练不能理解为“把千卡配置线性放大”？请从故障、性能、通信、控制面、存储和恢复说明。

- **面试官意图**：检查你是否具备大规模系统视角，能否把 MFU、straggler、拓扑、checkpoint 和运维串成完整闭环；同时核验你的实际规模与个人 ownership。

- **危险回答**：只回答“机器更容易坏、通信更慢”；只看平均 GPU utilization/MFU；把所有 hang 都归因于 NCCL；认为多打日志就能定位；把公开万卡经验或平台团队能力包装成个人 ownership。

</details>

↩ [返回本 Part 导航](#part-v) · ↑ [返回面试速查控制台](#interview-console)

<a id="infra-03"></a>
#### INFRA-03｜多机训练 NCCL hang 或 checkpoint 恢复失败怎么排查？（P0，18 分钟）

- **直接回答（60–90 秒）**：

  > 我先保存 job、rank、host 和最后正常 step 的证据，再决定是否隔离节点、回到已验证 checkpoint。NCCL hang 的关键是找首次异常：先检查各 rank 的 collective 协议和进程健康，再查链路、硬件和环境版本，最后报 timeout 的 rank 经常只是等待方。恢复失败则先检查 checkpoint 是否完整，以及 model、optimizer 和数据位点是否属于同一个提交 step，再看并行布局与加载代码是否兼容。恢复后，我会对齐逻辑参数和 optimizer 状态，核对 data cursor，并用固定输入和短窗口 loss/grad 验证恢复语义。

- **分支 A：NCCL hang**。按“代码一致性 → rank 健康 → 网络/硬件 → 环境版本”缩小故障域。核对 group membership、collective sequence、dtype/op/root 与 count 协议；variable-count exchange 要逐 peer 配对 send/recv count。结合 first bad rank 的进程退出、CUDA/Xid、flight recorder/NCCL trace，区分某 rank 未进入 collective 与通信路径本身变慢。GPU 环境检查 NVLink、IB/RoCE 链路和 packet/error counter；昇腾项目按 HCCS/RoCE 及对应通信栈的日志定位，避免混用工具与故障码。
- **分支 B：Checkpoint 恢复**。先检查 manifest/完成标记、shard 完整性和 checksum，再核对 model、optimizer、scheduler/scaler、RNG、data cursor 与 parallel metadata。它们必须组成同一提交 step 的一致快照；改变 world size/TP 后，需要框架支持的 reshard/转换，不能只改启动参数。加载后的 checksum 比较同一逻辑状态或相同分片布局，重分片后不要求原始 shard 文件逐个相等；loss continuity 也要结合数据位点、RNG 和固定输入对照解释。

- **异步保存追问**：一致快照、staging 和持久化完成是三个不同边界。先让训练状态对应同一提交 step，再将状态复制到不会被后续更新修改的 buffer；若使用异步 staging，必须在相关参数/状态再次修改前等复制完成。后台落盘全部成功并验证 shard 后，才能发布新的可恢复点；保存失败时继续保留上一份有效 checkpoint，同时限制并发保存数量和 CPU/pinned-memory 压力。[PyTorch DCP 异步保存](https://docs.pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html)

- **项目证据或知识边界**：你有 checkpoint deadlock、distributed optimizer checkpoint crash 和千卡交付经历；准备一个明确的 first bad event 案例。
- **高概率追问**：为什么一个 rank 提前异常会表现成其他 rank NCCL timeout？world size/TP 改变后如何恢复？async checkpoint 如何保证一致性？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：给出生产环境的调查顺序和止损方案。

- **面试官意图**：评估千卡经验、故障域判断、日志证据和恢复设计。

- **危险回答**：一看到 hang 就重启；只看最后报错 rank；checkpoint 只保存 model weights。

</details>

↩ [返回本 Part 导航](#part-v) · ↑ [返回面试速查控制台](#interview-console)

### P1 深挖｜面试官继续追问

<a id="resume-12"></a>
#### RESUME-12｜精度对齐问题通常怎么定位？（P1，10 分钟）

- **直接回答（60 秒）**：

  > 我会先固定输入、tokenizer、checkpoint、seed 和 dtype，用 eval mode 排除 dropout 等随机因素。从 embedding 开始，逐层比较 hidden state、attention、MLP、logits 和 loss，找到第一个明显偏离的算子，再检查 mask、position、精度转换和实现差异。误差阈值要按 dtype 和算子设定，同时看绝对误差、相对误差和误差分布。Forward 对齐后，再在受控随机性下比较训练态的梯度与 optimizer update；如果只有多机出现差异，就进一步检查分片布局和通信前后数据。这样能把“最后 loss 不同”缩小到具体计算边界。

- **项目证据或知识边界**：可讲 NPU/CPU AIT 对比或 YOLO/LLAMA 迁移，但继续保持客户信息脱敏。
- **高概率追问**：容许误差怎么设？多机不确定性如何处理？forward 对齐但训练发散怎么办？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：模型迁移后 loss/logits 不一致，你从哪里开始？

- **面试官意图**：验证华为阶段的精度调优不是黑盒试参。

- **危险回答**：只比较最终输出；直接调 learning rate；把 FP16 误差都视为正常。

</details>

↩ [返回本 Part 导航](#part-v) · ↑ [返回面试速查控制台](#interview-console)

<a id="infra-05"></a>
#### INFRA-05｜给你 64 张 A100，如何为 35B MoE 128K 选择并行策略？（P1，15 分钟）

- **直接回答（60–90 秒）**：

  > 我先确认 A100 是 40GB 还是 80GB、每节点卡数和互联，再问训练阶段、模型总参数与激活参数、hidden/layers、专家数和 top-k，以及 batch、实际长度分布和目标吞吐。随后按每个 rank 的 model state、activation 和 logits 算峰值显存。初始方向是把 TP 放在节点高速域，用 CP 分担长序列 activation；EP 根据专家数、路由负载和网络选择，PP 在容量或模型深度需要时引入。Attention 和 Expert 两种布局要分别核算，再确定有效 DP 和 global batch，不能把所有并行度机械相乘。最后从能放下模型的最小规模 smoke test 起，比较候选配置的 step breakdown、峰值显存和 scale curve，再决定 64 卡配置。

- **项目证据或知识边界**：可绑定 35B-A3B 与 128K/256K 交付，但不要假装题目参数已知。
- **高概率追问**：为什么不直接 TP=8？EP 是否跨节点？global batch 不可整除怎么办？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：没有完整参数时请先问哪些问题，再给初始方案。

- **面试官意图**：考需求澄清、容量模型和系统设计，不期待唯一答案。

- **危险回答**：立刻报一组数字；不问模型结构和拓扑；忽略有效 batch/收敛约束。

</details>

↩ [返回本 Part 导航](#part-v) · ↑ [返回面试速查控制台](#interview-console)

<a id="infra-06"></a>
#### INFRA-06｜推理吞吐、延迟和 KV cache 如何权衡？（P1，10 分钟）

- **直接回答（60 秒）**：

  > 我先看延迟目标，再找满足目标时的最大有效吞吐。TTFT 包含排队、prefill 和首 token 生成，高负载时排队可能占很大比例；TPOT 描述后续 token 的生成间隔。Prefill 通常更偏计算密集，decode 更容易受权重/KV 访存和调度影响。Continuous batching 能填补空闲槽位，但提高并发也会增加 KV 占用和尾延迟，所以要一起看 tokens/s、TTFT/TPOT 的 p95/p99、队列和 KV 使用率。长上下文会减少可同时驻留的请求数；chunked prefill 能改善与 decode 的调度共存，也需要衡量 prefill 完成时间。Agentic rollout 最后还要加上 tool/env wait、session affinity 和 prefix reuse，以完整 episode 或训练更新周期验收。

- **项目证据或知识边界**：有 vLLM/SGLang、CUDA Graph、prefix cache 和长上下文经验。
- **高概率追问**：为什么长上下文降低可并发数？chunked prefill 有何取舍？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：怎样同时解释 TTFT、TPOT、tokens/s 和 p99？

- **面试官意图**：验证推理基础与 rollout 性能模型。

- **危险回答**：只有 token/s 一个指标；把模型服务器延迟等同 agent episode 延迟。

</details>

↩ [返回本 Part 导航](#part-v) · ↑ [返回面试速查控制台](#interview-console)

<a id="infra-07"></a>
#### INFRA-07｜你会怎样设计训练系统的可观测性指标树？（P1，10 分钟）

- **直接回答（60 秒）**：

  > 我会先看训练是否真的在推进：顶层是 time-to-update、有效训练 tokens/s、成功率和成本，GPU utilization 放在资源层。发现更新变慢后，沿 data、rollout、reward、trainer、weight sync 和 checkpoint 拆开，看各阶段的吞吐、延迟分位数、队列、错误和资源占用，找出当前 critical path。比如 rollout 队列供给不足和 trainer 更新慢，需要完全不同的处理。定位单条样本时，用 trajectory ID、policy version、rank/host 串联 trace；这些高基数 ID 留在日志或 trace，聚合指标只保留有限维度。开启详细 tracing 后还要做开销对照，确保观测没有改变被测瓶颈。

- **项目证据或知识边界**：有 MFU、阶段耗时、lineage 和 DeepInsight/SwanLab 类指标经验。
- **高概率追问**：高基数 label 如何控制？如何避免 profiling 污染？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：GPU utilization 高但训练没进展，如何快速定位？

- **面试官意图**：考端到端 observability 和值班效率。

- **危险回答**：堆很多指标但没有层级；只看 GPU utilization；无跨服务 correlation ID。

</details>

↩ [返回本 Part 导航](#part-v) · ↑ [返回面试速查控制台](#interview-console)

<a id="infra-08"></a>
#### INFRA-08｜一个可恢复训练 checkpoint 必须保存什么？（P1，8 分钟）

- **直接回答（60–90 秒）**：

  > 我会先定义恢复到哪个提交 step，再保存与它一致的 model、optimizer、scheduler/scaler、RNG、global step、data sampler/cursor 和 parallel metadata。通常选 optimizer step 完成后的边界；如果要在梯度累积中间恢复，还要处理未提交梯度和 microstep 状态。Agentic RL 还要记录 policy、reward、tokenizer、prompt、env 的版本，以及 rollout backend provenance，保证恢复后数据的解释方式一致。队列和在途 trajectory 则有两种策略：能保存必要环境状态时继续恢复 session/cohort；否则显式取消或丢弃未提交工作，按约定重采样。两种策略都要把消费位点和训练提交状态对齐，防止重复消费、跳过数据或混入错误 policy version。最后用数据位点、固定输入和短窗口数值回归验证恢复结果。

- **在途状态的取舍**：queue offset 必须与队列内容或可重放日志配套；in-flight/partial trajectory、session/cohort 是否持久化取决于恢复承诺，不能只保存几个 ID 就假定可续跑。选择丢弃重采样时，要记录丢弃范围、更新消费/提交账本，并说明重算成本与随机路径可能变化，不承诺 bitwise 等价。异步 staging 与落盘的一致性边界见 [NCCL 与恢复排障](#infra-03)。
- **项目证据或知识边界**：有 StatefulDataLoader、online drain、checkpoint/recovery 经验。
- **高概率追问**：哪些状态可重建？如何避免重复消费？保存 queue 会不会太大？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：Agentic RL 相比 SFT 还要多保存哪些状态？

- **面试官意图**：检查训练状态机与恢复语义。

- **危险回答**：只保存权重；忽略 data cursor；恢复后不做数值检查。

</details>

↩ [返回本 Part 导航](#part-v) · ↑ [返回面试速查控制台](#interview-console)

### P2 选学｜时间允许再补

<a id="p2-03"></a>
#### P2-03｜如何判断瓶颈在 CUDA kernel、内存带宽还是通信？（P2，8 分钟）

- **直接回答（60 秒）**：

  > 我先在固定 workload 下拆 step timeline，确认时间花在数据等待、计算、通信还是同步，再分析关键路径上的 kernel。判断 compute-bound，要结合 GEMM shape、Tensor Core 吞吐和 roofline；判断 memory-bound，要看实际 DRAM 带宽、访存量和算术强度。小 GEMM 或大量碎 kernel 还可能受 launch gap 和调度开销限制，occupancy 高也不代表计算有效率高。通信方面，我会看 collective 的消息量、频率和未被计算覆盖的时间。最后用改变 batch、TP 或节点数的对照实验验证：如果预测的瓶颈没有随变量变化，就要重新检查假设。

- **项目证据或知识边界**：你有 tracing/MFU/通信优化经验；CUDA kernel 手写深度需诚实说明。
- **高概率追问**：GPU util 高为什么仍可能低效？小 GEMM 有什么特征？

<details>
<summary>面试意图与回答提醒</summary>

- **问题**：给一个 profile 方法而非工具列表。

- **面试官意图**：检查性能工程基本方法。

- **危险回答**：看到 util 100% 就认为 compute-bound；只说用 Nsight。

</details>

↩ [返回本 Part 导航](#part-v) · ↑ [返回面试速查控制台](#interview-console)

### 本 Part 追问路线

collective 输入输出 → loss/NaN/梯度/收敛异常 → 万卡规模效应/goodput → process group/拓扑 → NCCL hang → checkpoint/recovery → inference/KV cache → 指标树与 first divergence。

**性能项目必备追问**：[性能瓶颈定位](#p2-03) 虽保留 P2 编号与分级，但 X1、SFT、Fully Async 和 Agentic rollout 的性能故事都应能用它说明 profile 证据与对照实验；复习这些项目时一并口述。

---

<a id="part-vi"></a>
## Part VI｜面试应变与查漏补缺

**怎么用**：先按下一轮面试复习薄弱项，再核对项目口径。题目答案在前五个 Part，这里只放学习顺序、证据卡和反问入口。

**本 Part 导航**：[下一轮复习](#vi-0) · [智元 JD 补题](#vi-0a) · [项目证据卡](#vi-evidence-cards) · [模拟面试](#vi-mock) · [三轮反问](#vi-questions-to-ask) · [最后一小时](#vi-last-hour)

<a id="vi-0"></a>
### VI.0 下一轮复习与口径校准

按目前台账，9 月 8 日下午是智元二面，晚上是字节一面；小红书中台一面调整至 9 月 9 日 17:00，避免同日下午场次过密。下面安排优先覆盖已暴露的薄弱项；后续面试也可沿用，按岗位调整项目比重。

| 时间 | 复习入口 | 完成标准 |
|---:|---|---|
| 35 分钟 | [verl/AReaL 选型](#areal-01) · [Megatron/FSDP 选型](#megatron-11) · [FSDP 原理](#dist-01) | 能讲清约束、取舍和两次参数 AllGather/梯度 ReduceScatter 的条件 |
| 35 分钟 | [5D](#megatron-01) · [TP 的 MLP/Attention 切分](#megatron-02) · [通信算子](#infra-04) | 能画张量 shape 与通信方向；不只背并行维度 |
| 35 分钟 | [Rollout 全景](#rollout-01) · [CUDA Graph](#resume-13) · [Prefix Cache](#resume-14) · [Gateway](#resume-19) | 每项说出瓶颈、机制、收益范围和副作用 |
| 25 分钟 | [个人贡献](#resume-01b) · [X1](#resume-01a) · [SFT](#resume-05) · [万卡问题](#infra-09) | 选一个主项目讲清亲自做的动作、技术取舍和证据；另一个只备追问 |
| 35 分钟 | [手写 MHA](2026-09-interview-coding.md#coding-01) · [矩阵旋转](2026-09-interview-coding.md#coding-02) | 脱离答案写主路径，再检查 shape、mask、边界和复杂度 |
| 15 分钟 | [自我介绍](#resume-01) · [项目卡](#vi-evidence-cards) · [反问](#vi-questions-to-ask) | 自我介绍一分钟、数字不混用、按面试官角色选一到两问 |

合计 **3 小时**，是已有基础上的定向复习。技术同事面偏机制与实现；直属主管面在同一项目上多准备“为什么这样选、替代方案、个人贡献、怎样验收”。

<a id="vi-0a"></a>
#### VI.0A｜智元机器人训练 Infra：30 分钟补题

这份清单只补当前 JD 与既有题库的差集。按顺序读，每题先记住一句话，再点击进入完整答案：

| 时间 | 入口 | 必须记住的一句话 |
|---:|---|---|
| 8 分钟 | [TRAIN-ANOMALY-01｜loss/NaN/梯度/收敛排障](#train-anomaly-01) | 保护现场，按数据→forward→backward→optimizer→distributed 找 first divergence；OOM 与通信走已有专项题。 |
| 6 分钟 | [MEGATRON-11｜Megatron/FSDP/DeepSpeed/Accelerate](#megatron-11) | Accelerate 是上层编排，FSDP/ZeRO 是 DP 分片，Megatron 是多维模型并行；按约束选组合。 |
| 6 分钟 | [SFT-DATA-01｜数据到 loss 正确性](#sft-data-01) | chat template、token、position/attention/loss mask、packing 和 data cursor 必须一起验证。 |
| 5 分钟 | [DPO-01｜DPO 与 SFT/PPO/GRPO](#dpo-01) | DPO 用离线偏好对优化 policy-reference log-ratio，链路简单但缺少在线探索。 |
| 5 分钟 | [MLLM-01｜多模态与具身训练差异](#mllm-01) | 多模态新增媒体 IO、动态 visual token 和跨模态对齐；具身再增加时序 action 与闭环评测。 |

合计 **30 分钟**。你的直接项目证据仍以 Megatron 训练、长上下文 SFT、RLVR/AReaL、TX 视频/图像模型迁移和 Capek MLLM Infra 承载为准；不要把机制理解扩写成 DeepSpeed/FSDP 底层实现或机器人具身算法 ownership。

<details>
<summary><strong>还有三天：展开完整学习路线</strong></summary>

#### 三天冲刺安排

##### Core 10：建立项目主线时使用

按 Part 顺序口述：[自我介绍](#resume-01) → [Ownership](#resume-01b) → [职业选择](#resume-01c) → [X1 200B MoE 模型](#resume-01a) → [5D 并行](#megatron-01) → [Megatron 显存](#infra-02) → [Fully Async](#resume-02) → [AReaL 链路](#resume-08) → [MOPD/TILE](#resume-09) → [通信算子](#infra-04)。每题先说 30 秒结论，再展开到 2–5 分钟。

##### Day 1：Part I + Part II Core（约 4 小时）

- 45 分钟：完成 Part I 的自我介绍、Ownership 和职业选择。
- 90 分钟：完成 Part II Core：X1 MoE、5D 并行和 Megatron 显存账本。
- 60 分钟：补 Part II 的 SP/CP、PP/VPP、Dense/MoE、EP 与通信关系。
- 30 分钟：处理下面六项“口径校准”，统一数字和个人边界。
- 15 分钟：补项目证据卡中的 workload、数字分母和个人贡献。

##### Day 2：Part II 扩展 + Part III（4 小时 15 分钟）

- 90 分钟：浏览 Part II P0 的直接回答，重点深挖 SFT、CP-local logits 与训练框架选型；融合算子、规模交付、SFT data contract、多模态按 JD 选择。
- 30 分钟：选择性完成 Part II P1：视频 DiT/Ulysses、PP bubble、packing、recompute/offload、distributed checkpoint 和 Bridge 迁移层。
- 90 分钟：浏览 Part III P0 的直接回答，重点深挖 Fully Async 主故事、verl controller/SPMD、资源部署与权重同步；算法、staleness 和后端选型按薄弱项补充。
- 45 分钟：把 Fully Async 从 30 秒结论逐步展开到 3 分钟，并用 [VERL-11](#verl-11) 补充真实 LLM/MLLM 后训练落地证据。

##### Day 3：Part IV + Part V + Part VI 模拟（4 小时 15 分钟）

- 105 分钟：浏览 Part IV Core/P0，重点深挖 AReaL 链路、CUDA Graph、Prefix Cache、Gateway 与 MOPD；XCCL/disk 和 trajectory lineage 作为连续追问。
- 60 分钟：完成 Part V Core/P0：通信算子、loss/NaN/梯度/收敛排障、万卡规模效应与 NCCL/checkpoint 故障排查。
- 30 分钟：从 Part IV/V 的 P1 中选择与目标 JD 最相关的题。
- 60 分钟：按 Part VI 完成“自我介绍 → 项目 → 框架 → 故障 → 职业选择 → 反问”的完整模拟。

这份路线包含浏览和选择性精读，不代表在三天内完成全部题头标注的准备时长。

</details>

#### 面试前必须校准的六项简历口径

##### 口径 1：双 Teacher MOPD

统一为：**最新版双 Teacher MOPD 结果在 SWE、Terminal 双域提升，General 不下降。**这表示 EFFICACY 已有方向性结论，但在 checkpoint、样本数、seed、baseline、评测窗口和统计置信信息补齐前，不额外说“显著提升”“稳定提升 X pp”或“完成统计闭环”。单 Teacher 的 `Terminal +7.9pp`、`SWE +7.0pp` 不能当作双 Teacher 的分项数字。

##### 口径 2：CUDA Graph 的 6–8x 与约 14x 分属两个 workload

最新版投递简历的主口径是：**AReaL Qwen3.5-9B 128K Agentic RL 中，CUDA Graph 将 decode 阶段加速 6–8x。**另有 **verl 35B RLVR workload 的 decode 约 14x**，只能作为另一套模型/框架/并发和统计窗口下的独立证据。二者都只是 decode 局部收益，不是 rollout、单步训练或端到端同倍数加速。

##### 口径 3：SFT `31s → 9.3s` 讲联合优化链

已确认的动作是 `num_workers=0→8` 与 data prefetch、从偏重 full recompute 收敛到 selective recompute，以及 TP/CP 调整。pinned memory、persistent workers、独立 copy stream 等属于可检查的工程手段，不能在缺少当时配置时追加为实绩。选择重算模块要看释放的峰值显存和额外计算成本，经典 selective 的 `core_attn` 与现代 fused attention 路径需具体分析。TP/CP 调整要平衡本地 GEMM、逐层 collective 的相对开销与长序列 activation。

最新版数字仍是 `31s→9.3s、MFU 23%→45.2%`；当前没有逐项 A/B，两组指标也尚不能按标准 MFU 算术闭合。需核对 FLOPs estimator、实际计算 token、data wait 和统计窗口；补齐前不声称属于同一单一测量窗口。另一 workload 的 `TP=4,CP=4 → TP=2,CP=8、163s→102s` 只用于解释并行选择。`35B-A3B/128K 平均 step -50%` 与 actor CP-local logits 修复的 benchmark 归属尚未确认，先作为两项独立证据。

##### 口径 4：“交付 checkpoint”的准确含义

这里的“交付 checkpoint”不是 smoke test 产出一个可保存文件，而是**训练框架和 recipe 达到稳定训练验收，能够支持算法团队持续实验并产出经下游验证的有效模型权重**。回答时用代表性长度分布、连续训练窗口、loss/grad 稳定、save/resume、下游质量验证和 recipe 可复现说明交付；同时保留边界：它不自动等于无限期、无人值守的生产长稳。

##### 口径 5：Megatron-Core 的个人边界

面试定位是 **Megatron-Core feature integration/application layer 的训练系统集成、性能与正确性优化者**：能做 5D 配置、process group/拓扑推理，以及 Megatron-Core/MBridge 后端在 SFT、RLVR、长上下文和 MoE 中的接入、调优与排障。没有实现 collective kernel，没有修改 `parallel_state`/process-group construction，也没有编写 pipeline scheduler；不要暗示自己是这些底层机制的作者。简历暂时不改。

##### 口径 6：Fully Async 的同步对照尚未闭环

开箱同步基线用于发现约 79% 时间在 rollout，说明存在 overlap 空间；但“同步约 200”仍需补齐完全一致的 workload、统计窗口、warmup/异常步处理和 `tokens/s/GPU` 分母。`76 → 211–255` 是 Fully Async **内部**从初始配置到优化配置的比较，`236–293` 是 `2T+2R` 候选窗口。补齐协议前不要声称 Fully Async 相比同步提升了多少，更不能把 76→211–255 说成“同步切异步后的三倍提升”。

### VI.1 三框架对比速查

| 维度 | Megatron-Core | verl | AReaL |
|---|---|---|---|
| 核心定位 | 大模型高性能训练组件与并行/模型实现 | LLM RL post-training dataflow 与多后端编排 | 面向 reasoning/agent 的异步 RL 与在线服务桥接 |
| 主要抽象 | Transformer/parallel state/distributed optimizer/checkpoint | Trainer、WorkerGroup、TensorDict/DataProto、Engine、Rollout/TransferQueue | training/inference/agent/weight-update、staleness、online gateway |
| 训练后端 | 自身提供 Megatron 训练栈 | 可选 Megatron、FSDP/FSDP2 等 | 可接 Megatron/FSDP 等，版本相关 |
| 推理角色 | 不是主要目标 | 集成 vLLM/SGLang 等 rollout | 独立 inference service/rollout，强调在线 agent 接入 |
| 强项 | TP/PP/CP/EP、MoE、长上下文、规模扩展 | 算法流、placement、多 engine/recipe、sync/async/agent 生态 | async、bounded off-policy、session/trajectory、服务解耦 |
| 核心代价 | 配置/模型适配复杂、通信与拓扑敏感 | role/service/版本/依赖矩阵复杂，多条新路径持续演进 | staleness、trajectory 状态、微服务一致性与运维复杂 |
| 你的证据 | Megatron 后端 SFT/RLVR、长上下文、MoE、checkpoint | SFT/RLVR、fully async、vLLM/SGLang、性能/稳定性 | 128K Agentic RL、在线蒸馏、lineage、weight sync |
| 诚实边界 | 位于 feature integration/application layer；未实现 collective kernel，未改 `parallel_state`/process-group construction，未写 pipeline scheduler | 项目判断基于当时代码；当前官方已到 v0.9.0 | 项目版本早于 2.x，不能倒推使用当前微服务架构 |

一句话区分：

> **Megatron-Core 决定“一个大模型如何高效训练”，verl 决定“RL 的多个模型与计算阶段如何编排”，AReaL 更强调“长时 agent 数据如何异步生产、控陈旧并在线接入训练”。**

你的选型口径：

> **标准 SFT/RLVR 阶段，在当时比较 verl、slime、ROLL 后选择了完整度和后端生态更匹配的 verl；Agentic RL 阶段因长时 session、外部 Agent、fully async 和 Gateway 改造需求转向 AReaL，同时自行补齐外围生产能力。**

详细比较与当前版本重评：[verl 与 AReaL：RL 框架架构选型指南](../training-infra-roadmap/topics/rl_framework_selection.md)。

<a id="vi-evidence-cards"></a>
### VI.2 六张项目证据卡：已确认事实与待核验项

每卡先查 workload、比较双方和个人动作，再看待核验项。数字来自本人确认和[项目底稿](2026-08-xpeng-infra-resume-materials.md)；没有原始日志的部分不补造配置或因果关系。

#### 卡 1：X1 200B MoE 模型

| 项目 | 面试口径 |
|---|---|
| Workload | X1 200B MoE 预训练；直接规模训练证据为 3K 卡连续稳定训练两个月 |
| 比较双方 | 客户对标口径下相对性能 `0.16x→0.95x`，最终 MFU `35%` |
| 个人动作 | 模型跑通、性能采集与瓶颈分析；并行配置、Grouped MatMul、融合算子接入与通信 overlap 验证 |
| 验收 | 模型侧功能、精度、性能达标及规模回归；底层算子、网络和平台由对应团队负责 |
| 待核验 | 对标分母、模型层数/专家数/top-k、具体并行配置和融合算子、统计窗口；不凭 200B 反推专家结构 |
| 深挖入口 | [代表性优化](#resume-01a) · [个人贡献](#resume-01b) · [规模交付](#resume-10) |

#### 卡 2：Fully Async RLVR

| 项目 | 面试口径 |
|---|---|
| Workload | Qwen3-30B-A3B，32K 上限，4 节点 × 8 张 A100-80GB |
| 比较双方 | async 初始 `3T+1R / gen-TP=4 / 2 实例`：`76 tok/s/GPU`；联合优化后的代表性稳态窗口：`211–255` |
| 个人动作 | 调整 gen-TP、rollout 实例数、训练/推理资源配比及供给、暂停、cache 生命周期相关配置 |
| 验收与候选 | `2T+2R / 8 实例`候选窗口为 `236–293`，trainer idle 由 `0.41` 降至 `0.10–0.14`；同时检查 staleness、reward/eval |
| 待核验 | tok/s/GPU 的 token 定义、GPU 分母和窗口步数；同步“约 200”尚无完整同-workload 对照。不能报 sync→async 的提升倍数，也不分摊单项配置收益 |
| 深挖入口 | [异步优化主故事](#resume-02) · [gen-TP](#resume-03) · [数据正确性](#verl-05) |

#### 卡 3：Qwen3.5-9B SFT 3.3x

| 项目 | 面试口径 |
|---|---|
| Workload | Qwen3.5-9B 长上下文 SFT；具体卡数、长度分布、packing 与 batch 配置待原始日志确认，16–64 卡仅是项目覆盖范围 |
| 比较双方 | 最新简历分别记录 `31s→9.3s`、MFU `23%→45.2%`；当前不能证明两组指标来自同一计时窗口 |
| 个人动作 | DataLoader worker `0→8` 与预取、selective recompute、TP/CP 调整；没有逐项消融，不分拆贡献 |
| 验收 | 检查 loss/grad、显存峰值与稳定训练；还需以相同 FLOPs、计算 token 和统计窗口核对指标 |
| 待核验与独立结果 | 具体配置和 MFU estimator；`163s→102s` 是另一 workload。35B-A3B/128K 的 `step -50%` 与 actor CP-local logits 的 `7.6GB` 修复先分别陈述，未证实属于同次 benchmark |
| 深挖入口 | [9B SFT](#resume-05) · [35B/128K](#resume-17) · [CP-local logits](#resume-07) |

#### 卡 4：TX 视频 DiT / Ulysses

| 项目 | 面试口径 |
|---|---|
| Workload | TX 的文生视频、文生图和 MoE 模型国产卡适配；HunyuanVideo-14B 只作机制说明，不冒充已确认的项目 checkpoint |
| 比较双方 | 最新简历的项目总结果：开局性能提升 `30%–50%`、10+ 模型交付、80+ 问题；各模型的具体基线与窗口需分别说明 |
| 个人动作 | 模型侧功能、精度与性能适配，采集数据、分析瓶颈、推进对应团队修复并回归；协同 4–5 人 |
| 验收 | 功能、精度、性能和交付；Ulysses 示例重点验证 token/head 切分及 AllToAll，不把公开推理数字当训练实绩 |
| 待核验 | 实际模型与并行配置、具体个人事故案例；640×640×3×129 示例需由 VAE stride 和 patch size 推导 token 数 |
| 深挖入口 | [视频模型/Ulysses](#resume-18) · [交付职责](#resume-10) · [融合算子](#kernel-01) |

#### 卡 5：Agentic RL / Rollout

| 项目 | 面试口径 |
|---|---|
| Workload | Qwen3.5-9B、128K 的 Agentic RL；底稿的 R2E/SWE 场景为 32 张 A100，其他实验仍各自确认配置 |
| 端到端结果 | 底稿记录 DeepSWE 稳态 step `6467s→2301s`，有效 token 吞吐 `146.8 tok/s/GPU`；Seta Terminal `2240s→770s`，`233 tok/s/GPU`。未全部进入投递简历，展开时说明具体 workload |
| 分阶段结果 | AReaL decode `6–8x`；prefill 时间 `-44%`；Gateway 联合改造后 Rollout 推理吞吐 `+60%`，Rejected Group `33.18%→2.73%`。这些是不同阶段指标，不能相乘 |
| 个人动作与验收 | CUDA Graph、Prefix Cache、Sandbox 并发和 Gateway 调度；后续 quota/retry/lifecycle 改造按代码证据单独讲，验有效训练 token、staleness、轨迹与模型效果 |
| 待核验 | E2E 吞吐口径为实际参与训练的有效 token / GPU / 稳态时间；GPU 分母、基线窗口、Rejected Group 分母和拒绝原因分布仍需携带原始统计。verl 35B decode 约 `14x` 是独立 workload |
| 深挖入口 | [训练链路](#resume-08) · [调度收益](#resume-19) · [后续 Gateway 代码贡献](#areal-09) |

#### 卡 6：OPD/MOPD

| 项目 | 面试口径 |
|---|---|
| Workload | 不同领域分别 RL 得到 Expert；领域 Expert 作为冻结 Teacher，RL 前模型作为 Student，使用各领域原 RL 数据 |
| 比较双方 | TILE merge 初步未达到多域能力保留目标；最新版双 Teacher MOPD 在 SWE、Terminal 双域提升且 General 不下降 |
| 个人动作 | 按 data source 路由 Teacher、同 token 路径打分、loss/mask/normalization、混域权重、恢复与评测链路；算法 recipe 和个人工程贡献分开 |
| 验收 | FUNCTIONAL：链路可用；NUMERIC：计算正确；EFFICACY：目标域与 General 下游评测 |
| 待核验 | TILE 配置、各 Teacher/Student 血缘、最终 checkpoint/样本量/seed/baseline/窗口；不追加显著性或双 Teacher 的具体 pp，单 Teacher `7.9pp/7.0pp` 不混用 |
| 深挖入口 | [MOPD 主故事](#resume-09) · [轨迹到梯度](#areal-04) · [三层验收](#areal-08) |

<a id="vi-mock"></a>
### VI.3 一轮首面模拟顺序

按下面顺序录音，控制在 45–60 分钟：

1. [自我介绍](#resume-01)（一分钟）→ [个人贡献](#resume-01b) → [职业选择](#resume-01c)。
2. 训练项目：[X1 MoE](#resume-01a) 或 [长上下文 SFT](#resume-05) 选一个（3 分钟）→ [5D](#megatron-01) → [TP](#megatron-02) → [显存/OOM](#infra-02)。
3. RL 项目：[Fully Async](#resume-02) 或 [Agentic RL 链路](#resume-08) 选一个（3 分钟）→ [框架选型](#areal-01) → [Rollout 优化](#rollout-01) → [权重同步](#areal-11)。
4. 机制与排障：[FSDP](#dist-01) → [通信算子](#infra-04) → [训练异常](#train-anomaly-01) 或 [万卡问题](#infra-09) 选一题。
5. 深入追问按岗位选择：[MOPD](#resume-09)、[Gateway 代码贡献](#areal-09)、[CUDA Graph](#resume-13)、[Prefix Cache](#resume-14)。
6. 剩余时间讲 [岗位价值](#behavior-01)，再从 [三轮反问](#vi-questions-to-ask) 选一到两题。

录音复盘只检查四点：是否先说结论；是否有数字但也有口径；是否说清个人贡献；是否主动限定证据边界。

<a id="vi-questions-to-ask"></a>
### VI.4 建议反问面试官

**一面看实际工作和协作方式，二面看直属主管与岗位职责，三面看技术方向和 Leader 是否值得长期跟随。** 先确认面试官负责的范围，按角色选题；实际轮次与职级可能不同。每次选 1–2 题，并沿对方的回答追问一次。

| 轮次 | 面试官 | 核心判断 |
|---|---|---|
| 一面 | 组内同事 / 未来协作伙伴 | 工作是否真实、有技术含量，工程质量与协作是否健康 |
| 二面 | 直接主管 / +1 | 职责、权限、资源和成功标准是否匹配 |
| 三面 | 部门负责人 / Leader | 技术判断、长期愿景和团队理念是否值得跟随 |

#### 一面：组内同事 / 未来协作伙伴

**要判断什么**：实际工作的一阶技术问题、技术栈使用深度、工程质量标准，以及跨团队排障是否顺畅。

**主问 3 题**

1. **当前一阶瓶颈**：团队当前训练 Infra 最消耗工程时间的问题是什么——性能、稳定性、正确性、数据链路还是 rollout？现在通常如何定位和推进？
2. **技术栈与岗位落点**：团队主要使用什么训练框架，哪些地方需要自己修改？这个岗位最可能负责哪一层？
3. **复杂故障协作**：遇到跨训练框架、算子、通信和集群的复杂故障时，组内通常如何复现、分工和验收？

**备选 2 题**

1. 最近半年团队解决过最棘手的训练问题是什么，真正的一阶根因在哪里？
2. 团队如何做性能 benchmark、数值正确性回归、code review 和故障复盘？

> **只剩 30 秒**：团队当前最希望新同事尽快接手并解决的一个技术问题是什么？

**如何判断回答**

- **正向信号**：有具体 workload、规模、指标和定位工具；能区分框架能力、团队二开与业务 glue code；性能优化同时有正确性和模型效果验收；复杂故障强调共同复现、证据链和复盘。
- **风险信号**：只罗列框架名和卡数；长期依赖人工救火；没有 benchmark、监控或回归；问题长期在算法、性能和平台团队之间流转。

**本轮不建议问**：只问薪资、晋升、加班等应由 HR 或主管回答的问题；也不要把宝贵时间用在组内同事难以回答的宏观公司战略上。

#### 二面：直接主管 / +1

**要判断什么**：岗位的真实 ownership、入职后的成功标准、决策权限和资源是否匹配，以及主管如何处理优先级冲突。

**主问 3 题**

1. **成功标准**：如果我加入，前三个月和半年最希望我解决什么问题？您会用哪些结果判断我做得好？
2. **Ownership 边界**：这个岗位的职责边界和决策权限是什么？除了交付任务，是否需要主导框架选型、架构改造和跨团队推进？
3. **优先级取舍**：当训练效果、交付时间、GPU 成本和系统稳定性发生冲突时，团队通常如何排序，您会在哪些节点介入？

**备选 2 题**

1. 团队当前最需要补齐的能力是什么？主要卡在技术不确定性、资源还是跨团队协作？
2. 对高级工程师的评价更看重个人难题攻坚、平台影响力、项目 owner，还是团队协作和人才培养？

> **只剩 30 秒**：如果我入职半年，做到什么结果会让您认为这次招聘非常成功？为此我能获得哪些资源和协作支持？

**如何判断回答**

- **正向信号**：成功标准具体，责任、权限和资源基本对等；能诚实说明当前短板和哪些问题暂时不做；主管能保护核心目标并协调跨团队资源；既关注交付，也认可可复用的平台沉淀。
- **风险信号**：只给结果责任却不给决策权和协调支持；目标频繁变化且没有取舍机制；ownership 等同于无限兜底和长期救火；评价标准依赖主管主观印象。

**本轮不建议问**：不要沉入过细的底层实现，除非主管主动深入；可以询问评价机制，但不要在技术面索要晋升或薪资保证。

#### 三面：部门负责人 / Leader

**要判断什么**：负责人对未来两三年的核心技术判断、下注和放弃的依据、建设团队的标准，以及短期交付与长期能力之间的取舍。是否值得跟随，应从这些答案中推断。

**主问 3 题**

1. **长期技术方向**：结合团队的模型方向，未来两三年您认为最值得持续建设的训练系统能力是什么？为什么优先投入这一层？
2. **技术品味与判断**：面对很多新框架和技术路线，您如何判断哪些值得下注、哪些应该放弃？能否分享一次重要技术取舍，以及后来什么证据改变或强化了判断？
3. **团队理念与取舍**：您希望建设一支什么样的团队、培养什么样的高级工程师？当短期交付与长期平台质量冲突时，您通常如何决策？

**备选 2 题**

1. 目前您认为部门最大的长期技术风险是什么？哪件难而重要的事情最值得团队一起攻克？
2. 一两年后出现什么技术或组织结果，会让您认为这支团队真正建立了壁垒？

> **只剩 30 秒**：未来两三年您最希望这支团队建立哪项不可替代的能力，为什么值得现在开始长期投入？

**如何判断回答**

- **正向信号**：能从业务目标、模型演进和系统约束推导技术方向；能说明下注、放弃和修正判断的证据；愿景能落到人才、资源和里程碑；尊重专业判断，也清楚何时必须为交付妥协。
- **风险信号**：只有宏大口号、规模和 headcount，没有关键取舍；从不谈失败或判断修正；强调英雄主义和无限投入；愿景与可用资源、业务节奏明显脱节。

**本轮不建议问**：不要问细碎的框架参数或单点实现；也不要直接问“您为什么值得跟随”，这会把判断题变成恭维题。

<a id="vi-last-hour"></a>
### VI.5 面试前最后一小时清单

这一小时只复述已准备过的内容。发现证据缺口时，使用已经确认的口径，把补证任务记到项目卡。

| 时间 | 做什么 | 入口 |
|---:|---|---|
| 5 分钟 | 说一遍自我介绍，确认地点和职业选择的简短表达 | [自我介绍](#resume-01) · [职业选择](#resume-01c) |
| 15 分钟 | 讲一个核心项目：问题、本人动作、结果、边界；再接两层技术追问 | [项目入口表](#interview-console) · [项目证据卡](#vi-evidence-cards) |
| 10 分钟 | 串讲两类框架选型；画 FSDP 或 TP 的通信流程 | [verl/AReaL](#areal-01) · [Megatron/FSDP](#megatron-11) · [TP](#megatron-02) |
| 10 分钟 | 复述 rollout 三项优化及收益范围 | [Rollout](#rollout-01) · [CUDA Graph](#resume-13) · [Prefix Cache](#resume-14) |
| 15 分钟 | 手写一道已练过的题，检查 shape、mask 或矩阵边界 | [Coding](2026-09-interview-coding.md) |
| 5 分钟 | 选一到两道反问，确认页面跳转和返回操作 | [反问](#vi-questions-to-ask) · [速查控制台](#interview-console) |

<details>
<summary><strong>完整知识自检：平时复习时逐项检查</strong></summary>

- [ ] 自我介绍能在 90 秒内完成，且只保留两条主线。
- [ ] X1 MoE 能在 90 秒内讲清 Three Walls、关键动作和结果，并能补齐真实并行配置、融合算子和 overlap timeline。
- [ ] Dense/MoE 能区分总专家数 `E`、每 token 激活数 `top-k`、expert FFN intermediate size 与 shared expert，不猜测 X1 未核验配置。
- [ ] Ownership 能区分个人决策、亲自实现、开源框架和团队依赖。
- [ ] Fully Async 先解释同步边界和 producer-consumer overlap，再解释 76、211–255、236–293、0.41、0.10–0.14；同步口径未补齐前不报提升倍数。
- [ ] 职业选择能在 60–90 秒内讲清上海搬迁、深圳长期规划、技术栈扩展和当前组织调整，不使用负面措辞。
- [ ] SFT 使用最新版 `31s→9.3s、MFU 23%→45.2%`；能解释 DataLoader、selective recompute、TP/CP，但在 MFU 算术闭环前不声称两组数字来自同一单一窗口。
- [ ] 双 Teacher MOPD 统一为“SWE、Terminal 双域提升且 General 不下降”，不混用单 Teacher pp。
- [ ] TILE merge 只说项目确认的 baseline 与评测结果，不扩写未确认机制或论文来源。
- [ ] CUDA Graph 主答使用“AReaL 9B 128K Agentic RL decode 6–8x”；verl 35B RLVR 约 14x 只作为独立 workload，二者都不外推端到端。
- [ ] Gateway 能区分请求完成即补位与 token streaming，并说明 `+60%`、`33.18%→2.73%` 的阶段、分母和正确性护栏。
- [ ] CP-local logits 能画出 `[T/CP,V/TP] → local scalar → CP gather [T]`，并解释 sequence chunk 为什么救不了已 materialize 的 full logits。
- [ ] 能用 Attention/Expert 双视图算 MoE world-size，不再机械相乘 TP、CP、EP、DP。
- [ ] 能用 `P/G/O` 解释 ZeRO-1/2/3 与 FSDP/FSDP2，并按模型、并行、拓扑和团队资产选择 Megatron 或 FSDP2。
- [ ] 能用一句话讲 PPO、GRPO、DAPO，并说明算法变化如何改变 rollout 数据契约。
- [ ] 能解释 verl 的 controller、ResourcePool/WorkerGroup、TransferQueue 与 backend SPMD engine 分别调度什么。
- [ ] 能区分 Fully Async、streaming、partial rollout、staleness，以及 colocate、disaggregate、异构部署三个概念。
- [ ] 能从代码证据说清 Gateway 团队基线与个人四层改造，不把 OpenAI proxy/cohort 基础架构说成自研。
- [ ] 能区分 XCCL 直接 bucket transfer 与 disk 临时 HF transfer，明确两者都不是 recovery checkpoint，并能说清本地分支的 colocation/LoRA 支持边界。
- [ ] 能用 Athena 与 Capek 两张图讲清 LLM/MLLM 后训练链路，并把个人 Infra ownership 与算法同学的 recipe/论文成果分开。
- [ ] 能画 Megatron TP/PP/CP/DP/EP，以及 verl/AReaL 两张数据流图。
- [ ] 能用一句话区分 SP 与 CP、distributed optimizer 与 ZeRO-3、verl 与 AReaL。
- [ ] 能从输入输出解释 AllReduce、ReduceScatter、AllGather、AllToAll 和 Send/Recv，并说清 gradient、parameter、activation、token 分别在哪一步传输。
- [ ] 能用“小概率故障变高概率、最慢 rank 放大 p99、并发操作形成惊群”解释万卡规模质变，并用 goodput、MTBF/MTTR、topology、checkpoint/recovery 收口；明确 1.2 万/1.4 万是集群总规模，直接证据是 X1 3K 连续稳定训练两个月，不是完整万卡平台 ownership。
- [ ] 能从本 rank 参数量、bytes/param、activation、logits 和 phase peak 手算一遍 Megatron 显存账。
- [ ] 准备一个 OOM、一个 NCCL/checkpoint、一个精度对齐真实案例。
- [ ] 每个故事能说清“我做了什么”，不只说“团队做了什么”。
- [ ] 不泄露联系方式、客户名、内部仓库、未公开模型和未脱敏集群数据。

</details>

### VI.6 继续阅读：仓库内现有材料

- [Agentic RL Infrastructure](../training-infra-roadmap/topics/agentic_rl.md)
- [Megatron 5D 并行总览](../training-infra-roadmap/topics/distributed_training.md)
- [Dense/MoE、Expert 路由与 Parallel Folding 工程章节](../training-infra-roadmap/topics/moe.md#dense-vs-moe)
- [NCCL 与分布式通信算子](../training-infra-roadmap/topics/nccl.md#collective-map)
- [大规模训练稳定性与容错：从千卡到万卡](../training-infra-roadmap/topics/fault_tolerance.md#large-scale-training)
- [Tensor Parallelism 面试题](../training-infra-roadmap/interview/tensor_parallelism.md)
- [MoE 面试题](../training-infra-roadmap/interview/moe.md)
- [Checkpoint 面试题](../training-infra-roadmap/interview/checkpoint.md)
- [FSDP 面试题](../training-infra-roadmap/interview/fsdp.md)
- [FSDP/FSDP2、ZeRO、Megatron 与 Bridge 选型](../training-infra-roadmap/topics/fsdp.md)
- [verl 与 AReaL：RL 框架架构选型](../training-infra-roadmap/topics/rl_framework_selection.md)
- [FlashAttention 面试题](../training-infra-roadmap/interview/flashattention.md)
- [Megatron-LM 论文笔记](../training-infra-roadmap/papers/megatron_lm.md)
- [Megatron Core MoE 2026 中文翻译（5 部分 PDF）](../training-infra-roadmap/README.md#megatron-core-moe-2026-zh-pdf)

### VI.7 资料来源与版本边界

技术结论优先使用官方资料；岗位题目概率来自当前公开 JD 与本简历暴露面，是面试准备判断，不是统计学结论。

#### 官方框架资料（核验于 2026-09-02）

- NVIDIA Megatron-Core：[Scalable Training of Mixture-of-Experts Models with Megatron Core](https://arxiv.org/abs/2603.07685)、[MoE Parallel Folding](https://arxiv.org/abs/2504.14960)、[MoE Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html)、[Parallelism Strategies Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)、[Context Parallelism](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)、[Distributed Optimizer](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/dist_optimizer.html)、[Pipeline Schedules](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.pipeline_parallel.schedules.html)、[`theoretical_memory_usage.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/theoretical_memory_usage.py)。Release 页面核验到 `core_v0.18.2`，commit `571370c`；MoE 技术报告和上述公式补充核验于 2026-09-01。
- PyTorch/DeepSpeed/Bridge：[FSDP2 `fully_shard`](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html)、[FSDP1](https://docs.pytorch.org/docs/stable/fsdp.html)、[DeepSpeed ZeRO Tutorial](https://www.deepspeed.ai/tutorials/zero/)、[`mbridge`](https://pypi.org/project/mbridge/)、[NVIDIA Megatron Bridge](https://docs.nvidia.com/nemo/megatron-bridge/latest/)。FSDP/ZeRO 是 DP state sharding，Megatron 多维并行解决的约束更广；两类能力可以组合。`mbridge` 与 NVIDIA `megatron-bridge` 是独立 package。
- RL 算法：[PPO](https://arxiv.org/abs/1707.06347)、[DeepSeekMath/GRPO](https://arxiv.org/abs/2402.03300)、[DAPO](https://arxiv.org/abs/2503.14476)。主文档只保留工程口述，公式和数据契约见 [Agentic RL topic](../training-infra-roadmap/topics/agentic_rl.md#ppo-grpo-dapo)。
- verl：[GitHub](https://github.com/verl-project/verl)、[HybridFlow Programming Guide](https://verl.readthedocs.io/en/latest/hybrid_flow.html)、[0.7 Architecture](https://verl.readthedocs.io/en/latest/blog/v0.7.html)、[v0.7.0](https://github.com/verl-project/verl/releases/tag/v0.7.0)、[v0.8.0](https://github.com/verl-project/verl/releases/tag/v0.8.0)、[v0.9.0](https://github.com/verl-project/verl/releases/tag/v0.9.0)、[v0.9.0 Fully Async](https://github.com/verl-project/verl/blob/v0.9.0/docs/advance/fully_async.md)。项目历史参照为 `v0.7.1`（`bec9ef7`）；当前重评基线为 `v0.9.0`（`483b8a0`），不能把后续能力倒推到项目版本。
- 项目产出图示来源：[Athena-Brain v2, Figure 3](https://arxiv.org/pdf/2607.18985v2)、[Capek 0.5 v1, Figure 6](https://arxiv.org/pdf/2608.06756v1)。两图用于说明自研版 verl 支撑的 LLM/MLLM 后训练链路；个人 ownership 是框架建设、集成、性能与正确性保障，不把算法 recipe、模型产出或论文 authorship 归到个人名下。
- Rollout backend：[vLLM OpenAI-compatible server](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/)、[SGLang docs](https://docs.sglang.io/)。后端选型必须锁定版本、模型、硬件与真实 RL workload。
- AReaL：[GitHub](https://github.com/areal-project/AReaL)、[v2.1.0 Asynchronous RL Guide](https://github.com/areal-project/AReaL/blob/v2.1.0/docs/en/algorithms/async.md)、[v2.1.0 Online Proxy](https://github.com/areal-project/AReaL/blob/v2.1.0/docs/en/tutorial/online_proxy.md)、[Releases](https://github.com/areal-project/AReaL/releases)。`v2.0.0`（`fee938e`，2026-07-01）把 training、inference、agent、weight-update 拆为独立服务；截至 2026-09-02，当前 release 为 `v2.1.0`（`ecc8b0e`）。项目 online proxy/cohort 链路早于 2.x，不能倒推为当前架构。
- NVIDIA 训练与执行：[Megatron-Core TransformerConfig / selective recompute](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.transformer.transformer_config.html)、[Megatron-Core fused bias-dropout-add](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.fusions.fused_bias_dropout.html)、[CUDA Graphs Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)。文档说明当前可用机制，不倒灌成项目当时已启用的具体开关。
- 视频序列并行：[Tencent HunyuanVideo / Unified Sequence Parallelism](https://github.com/Tencent-Hunyuan/HunyuanVideo#parallel-inference-on-multiple-gpus-by-xdit)。官方示例用于解释 Ulysses/Ring 机制，不把其推理配置或性能数字当作本人训练项目结果。
- NVIDIA NCCL：[Collective Operations, NCCL 2.31.2](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)。
- PyTorch：[Distributed Checkpoint Tutorial](https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html)。
- 大规模生产训练：[MegaScale](https://arxiv.org/abs/2402.15627)、[The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)。两篇材料用于支撑万卡规模的故障、straggler、观测和恢复判断；其中公开集群数字不是个人项目证据。

#### 当前岗位信号（动态页面，核验于 2026-08-30）

- [华为社招：大模型训练/强化学习/推理相关岗位](https://career.huawei.com/reccampportal/portal5/social-recruitment-detail.html?dataSource=1&jobId=28183)：强调独立系统设计、训练/RL 原理、精度调优、vLLM/SGLang 和软硬件协同。
- [华为社招：AI 底层软件栈与训推性能](https://career.huawei.com/reccampportal/portal5/social-recruitment-detail.html?dataSource=1&jobId=32189)：强调 runtime、显存、集合通信、profiling、疑难问题攻坚和稳定交付。
- BOSS 公开职位聚合中的腾讯/美团等岗位把 Megatron、verl、vLLM/SGLang、RL Infra、规模训练和系统优化列为核心职责；聚合页会变动，只用于判断常见考察方向，不用于技术事实。

### VI.8 题量与时间预算

| 优先级 | 题量 | 全量准备时间（按题头累加） | 用法 |
|---|---:|---:|---|
| P0 | 49 | 约 13 小时 20 分钟 | 优先练实际薄弱项；Core 10 用来串联个人项目主线 |
| P1 | 25 | 约 4 小时 20 分钟 | 按目标 JD 和项目追问选择，不要求一次学完 |
| P2 | 5 | 40 分钟 | 按需补充；profiler 是性能项目的前置工具，可提前看 |

上表是把每题完整学习一遍的估算，不含 coding、做实验和重复口述。已有基础时，按 [下一轮 3 小时复习](#vi-0) 或 [最后一小时清单](#vi-last-hour) 选题；现场只查「直接回答」，被追问再向下展开。

---

<a id="interview-progress"></a>
## Appendix A｜面试流程进度台账

> 更新截至 2026-09-07；时间为北京时间（UTC+8）。这里只维护时间、轮次和状态；技术问题统一归入正文题库，不做逐场面试复盘。

| 公司 | 岗位 | 面试时间 | 当前轮次 | 状态 | 下一节点 |
|---|---|---|---|---|---|
| 灵动时刻 | 训练 Infra | 2026-09-03 下午 | 一面完成 | 未通过 | 本轮流程结束 |
| 智元机器人 | 训练 Infra | 2026-09-04 下午 | 一面完成 | 已通过 | 2026-09-08 下午二面 |
| 字节跳动 | 训练 Infra | 2026-09-08 晚上 | 一面待进行 | 已排期 | 完成一面 |
| 小红书中台 | 训练 Infra | 2026-09-09 17:00 | 一面待进行 | 已排期 | 完成一面 |

↑ [返回面试速查控制台](#interview-console)

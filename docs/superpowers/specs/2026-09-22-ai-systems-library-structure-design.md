# ReadBase 结构调整：共享基础、训练／推理／RL 与具身学习路径

> 日期：2026-09-22。
> 状态：用户已确认总体结构及五项补充，并要求面试准备独立成 Part、增加最近更新入口；本文件按审阅意见修订，尚未重组正式入口。
> 本轮交付：结构、追踪规则、内容边界和实验课程设计。不启动 GPU 作业，不宣称已完成教程、精读或性能验证。
> 读者：已有 LLM Training 与 Agentic RL 工程经验，接下来重点学习具身模型、具身 Infra、FSDP/FSDP2 和 ZeRO 的工程师。

## 1. 已确认的目标与范围

### 1.1 用户决策

1. 长期技术学习优先；面试主文档、题库、Coding 和准备材料独立为按需使用的 Part，不纳入日常学习必经路线。
2. 延续“共享系统基础＋应用学习路径”，保留 LLM Training、Agentic RL 积累，具身成为当前重点。
3. 具身模型与具身 Infra 都进入统一最新材料追踪范围；这里的文章追踪称 `tracking`，区别于运行时 performance tracing。
4. 训练、推理、RL 必须兼顾；推理不再仅因影响 RL rollout 才被关注。
5. 深入 FSDP/FSDP2、ZeRO-1/2/3，以及具身场景的多模态存储、读取、解码和数据正确性。
6. 根 README 和各 Part 提供简短概括、明确顺序和便捷跳转，避免多个入口重复维护正文。
7. 有 A100、64 卡以内的实验资源；先设计、后确认环境与执行预算，不把资源可用视为启动任务授权。
8. 首页新增“最近更新”，快速发现仓库新内容和项目进展；区别于外部文章 tracking。个人最新项目经历通过面试 Part 进入既有项目素材与主文档，不重复建一套履历。

### 1.2 本规格的交付边界

本规格完整定义的是**知识库组织与近期学习课程**。导航与追踪规则可以作为一个有界实施批次；深入教程、实验代码和 GPU 执行是后续分别验收的内容工作，不在结构调整时一次性实现。

本次不改仓库名称，不批量迁移历史文档，不重写面试题库，不调整定时任务，不回填历史扫描计数或游标，不安装训练环境，不访问远端集群。

## 2. 当前问题与设计选择

- [根入口](../../../README.md)及[手册入口](../../../training-infra-roadmap/README.md)仍使用 Training → Inference → Agent 的阶段式描述，不能体现实际并存的知识方向。
- [现有具身系统章](../../../training-infra-roadmap/topics/agentic_for_embodied.md)已有较完整蓝图，但尚未真实实验验证；[模型入门设计](2026-09-18-embodied-models-primer-design.md)仍是大纲，不能当成已完成教程。
- [FSDP 专题](../../../training-infra-roadmap/topics/fsdp.md)已有机制和选型，应在此深化，而不是新增相同原理的“具身版 FSDP”。
- [扫描模板](../../../training-infra-roadmap/tracking/frontier_scan_template.md)把独立 inference 限定为影响 rollout，且厂商 Watch 与当前四家规则存在不一致。
- README、Master List、Knowledge Graph 重复维护最新扫描、优先级和学习顺序，增加入口漂移。

比较过三种方案：

| 方案 | 判断 |
|---|---|
| 保留物理路径，重建共享基础与学习导航 | 采用；知识复用、旧链接稳定，新增文件限于确实有内容的导航／章节 |
| 各方向复制一套 papers/topics/tracking/queue | 不采用；FSDP、数据、推理和 RL 容易重复维护 |
| 全仓改名搬迁后重新分目录 | 当前不采用；链接迁移成本高，无法直接解决内容深度问题 |

总原则：**导航按读者的问题组织，正文按知识职责存放，优先级按当前阶段管理。**

## 3. 阅读结构与 Part 边界

### 3.1 六个日常学习 Part＋一个按需面试 Part

Part I–VI 是日常学习视图，不是互斥的学科分类：具身是综合应用路径，复用训练、推理、RL 及公共基础；实验为所有方向服务。Part VII 是独立的面试使用场景，只在需要复习、准备或现场速查时进入，不设为前六个 Part 的学习前置。

| Part | 核心问题 | 主要内容 | 复用边界 |
|---|---|---|---|
| I. 系统基础 | 计算、数据和状态如何流动？ | PyTorch/autograd、GPU 执行与内存、通信、存储、profiling、正确性 | 提供各方向共用的概念，不复制框架完整教程 |
| II. 训练系统 | 如何正确、高效、可靠地更新模型？ | DDP、ZeRO、FSDP1/2、Megatron、并行、重计算、精度、checkpoint、恢复 | FSDP/ZeRO 完整正文只维护一处；覆盖 LLM 与多模态训练 |
| III. 推理系统 | 如何在质量约束下满足吞吐与时延？ | prefill/decode、KV/prefix cache、batching、量化、编译、CUDA Graph、服务调度、实时推理 | 不限于 RL rollout，也不把所有推理都简化为 token decode |
| IV. RL 与 Agent 系统 | 环境、生成、训练如何形成可信闭环？ | 算法最小基础、verl/AReaL、rollout、reward、调度、权重版本、轨迹与恢复 | 文本／工具 Agent 为已有基线；物理动作的特有风险回链具身路径 |
| V. 具身模型与 Infra | 理解、交互、动作和环境反馈如何连接系统？ | MLLM、多模态交互、VLA、World Model、BC/RL、episode、数据 I/O、仿真、部署评估 | 不作为 RL 子集；先模型和样本，再训练、执行与学习系统 |
| VI. 实验与排障 | 如何验证认识，并复现、解释故障？ | A100 学习实验、项目、原始指标、runbook、结论边界 | 链接到相应 topic，不另写一套理论教材 |
| VII. 面试复习与准备（按需） | 如何复习、组织项目表达，并快速找到题目答案？ | 面试主文档、专题题库、Coding、项目经历与准备材料 | 聚合既有文档，技术原理回链日常学习正文；不作为当前学习重点 |

LLM Training 对应 Part II 的主要应用路径；Agentic RL 对应 Part IV；Embodied 对应 Part V。Part III 将此前散落的 inference 能力显式组织起来，不要求先完成训练学习才能进入。

### 3.2 文件位置

保留 `training-infra-roadmap/` 的历史路径；显示标题可使用“AI Systems 工程学习手册”，并说明目录名沿用历史路径，不再意味着只维护训练。

Part I–V 的轻量导航放在现有 `roadmaps/` 中；Part VI 和 VII 分别使用既有实验、面试目录的 README 作为入口：

```text
ReadBase/
  README.md                                  # 首页：当前重点＋最近更新＋六个日常 Part＋按需面试
  training-infra-roadmap/
    README.md                                # 全库学习地图和文档使用方法
    roadmaps/
      foundations/README.md                  # Part I，拟新增
      training/README.md                     # Part II，拟新增
      inference/README.md                    # Part III，拟新增
      rl/README.md                           # Part IV，拟新增
      embodied/README.md                     # Part V，拟新增
      30_day_plan.md                         # 既有基础补课路线，明确适用人群
      90_day_plan.md                         # 既有系统训练路线，不默认为当前待办
      yearly_plan.md                         # 既有长期能力路线，保留可复用部分
    experiments/README.md                    # 复用为 Part VI，不新增重复入口
    interview/README.md                      # Part VII，拟新增；统一聚合面试准备
    topics/                                  # 技术正文：每个核心问题一个主要维护位置
    papers/ / tech_reports/ / engineering_blogs/
    tracking/ / reading_queue/ / learning_log/
    projects/ / insights/ / playbooks/ / references/
  private_resume/                             # 保留主文档、Coding、项目素材等旧路径
```

新增的五份学习 README 和一份面试 README 都是薄导航，不复制 `topics/`、不拥有独立扫描或队列；没有匹配内容的项写明“待建设”，不创建空目录或死链接。面试材料的“独立”指统一 Part 入口与使用场景，不为此搬迁文件或改动旧问题锚点。

## 4. 首页、Part 与跳转规则

### 4.1 根 README

按以下顺序组织，尽量一屏能找到主要入口：

1. 一句话定位：面向训练、推理、RL 与具身的 AI Systems 工程学习库。
2. 当前学习重点：具身模型基础、FSDP/ZeRO、多模态 I/O；只描述方向，具体待办链接统一 P0/P1。
3. 最近更新：最多 5 条有阅读价值的新增／实质更新，直接跳转正文或项目状态。
4. 日常学习 Part I–VI 的简短表格：解决什么问题、从哪里开始。
5. 按需入口 Part VII：面试复习与准备；只保留一个简短入口，不在首页展开复习计划或题库。
6. 常用入口：知识关系、资料索引、外部资料追踪、实验与排障。
7. 简短维护原则；详细目录职责移至手册入口。

不在首页同时堆放全部论文、完整研究工作流和历史扫描列表。保留 ReadBase 名称和既有面试文件路径。

### 4.2 每份 Part README

日常学习 Part I–VI 使用以下结构；Part VII 按 §4.3 采用面试场景导航，不机械套用前置知识、实验与排障表格：

- **本 Part 回答什么**：2–3 句，不写宣传性愿景。
- **从这里开始**：优先 3–5 个有顺序的入口，每项注明前置知识及读后应能解释的问题。
- **按问题查找**：问题／正文／实验或排障／内容完成程度。
- **继续阅读**：进阶材料按需展开，避免所有材料都标必读。
- **返回与相邻 Part**：返回首页、返回学习地图、必要的跨 Part 链接。

每篇被本批次触及的正文增加简短导航栏；不为了加返回链接一次性修改全仓旧笔记。

GitHub Markdown 的明确回链用于回到 Part 或首页；要回到刚才的精确滚动位置，使用浏览器后退。不开新网页控制台，不依赖本地预览服务或 JavaScript。

### 4.3 Part VII：面试时才打开的统一入口

拟新增 `training-infra-roadmap/interview/README.md`，在一个页面内提供以下分组：

| 入口 | 内容落点与使用方式 |
|---|---|
| 面试主文档／现场速查 | [既有主文档](../../../private_resume/2026-08-llm-infra-interview-prep.md)，保留原有题号、锚点与考场导航 |
| 按技术主题复习 | 既有 `interview/` 专题题库；从短答案回链 `topics/` 机制正文 |
| Coding／手撕代码 | [既有 Python 3 编程题库](../../../private_resume/2026-09-interview-coding.md)，不混入技术正文学习路径 |
| 项目经历与最新素材 | [既有项目素材底稿](../../../private_resume/2026-08-xpeng-infra-resume-materials.md)和主文档的项目问题；标明事实、贡献边界和指标口径 |
| 专项准备（可选） | 已有笔试／公司专项准备，标明历史适用背景，不替代通用主文档 |

Part VII 保留“快速复习 → 项目问答 → 技术追问 → Coding”的使用提示，但不安排日常刷题待办；主文档内部原有 Part 编号也不随知识库新增 Part 重新编号。链接标签区分“知识库 Part VII”和“面试主文档内的章节”，避免两个层级混淆。

后续新增真实项目经历时，先更新对应事实与证据记录，再更新面试主文档中需要讲述的版本；入口只链接和概括，不复制技术方案和实验数字。没有新材料时，不根据学习计划或代码阅读编造个人经历，也不把未运行实验写成项目成果。不得借导航调整恢复此前要求删除的薪资、股权等信息。

### 4.4 最近更新：仓库变化，不是另一份资讯雷达

根 README 是唯一的全库“最近更新”列表，使用显式锚点 `recent-updates`。手册、Part、Master List 与 Knowledge Graph 链接回这里，不重复维护全库滚动列表；各 Part 可以说明自己的内容覆盖，不另抄一份全局更新清单。

每条使用 `文档更新日期｜类别／所属 Part｜新增或改变了什么｜直达链接`，按日期倒序，最多 5 条。同一主题同一批修改合并为一条；只收有阅读价值的新增或实质修改，不记录排版、改名、修链接等琐碎提交。首版从实际已提交内容中核对选取，不为凑满条目生成更新。

- **知识内容**：新增／深化了哪个章节、论文解读、实验设计或排障方法，链接具体正文或章节。
- **项目进展**：链接既有 `training-infra-roadmap/projects/.../STATUS.md` 的最近记录或实际里程碑，注明原记录截至日期；文档更新不意味着里程碑完成，旧状态也不能推断为当前运行事实。个人面试项目素材更新则链接对应素材／问答，并标明类别，不与实验项目混称。
- **状态边界**：沿用来源里的“设计／待验证／实测”等标识；文档提交日期不替代项目发生日期、实验执行日期或原始材料发表日期。不在摘要中添加原文没有的结论。
- **与 tracking 的区别**：tracking 回答外部出现了什么；最近更新回答仓库中现在新增了什么值得读。一份扫描报告可作为仓库更新，但不把报告中每条外部新闻搬到首页。
- **维护方式**：本次有实质内容更新时一并调整首页条目并检查链接，随正文同批发布；历史通过仓库 Git 提交记录查询，不新增 changelog 文件、自动化或网站。

只曝光已获准公开的技术内容和必要进展摘要；不把面试台账、联系方式等信息自动提升到首页，也不因目录名 `private_resume/` 就假定其在 GitHub 上私密。

### 4.5 唯一信息来源

| 信息 | 主要维护位置 | 其他入口如何引用 |
|---|---|---|
| 本阶段学习方向 | 根 README | Part 概括与其一致，不维护另一套全局优先级 |
| 仓库最近实质更新 | 根 README 的“最近更新” | 其他导航回链；最多 5 条，直接链接来源正文 |
| 当前具体阅读优先级 | `reading_queue/P0.md`、`P1.md` | 链接队列，不复制易过期的 P0 名单 |
| 最新扫描与扫描状态 | `tracking/README.md`、`scan_log.md` | Master List/KG 链接稳定入口，不重复滚动列表 |
| 一项技术的机制与判断 | 对应 `topics/` 正文 | Part 只摘要，面试题只保留必要速答 |
| 来源证据 | papers/reports/blog notes 与原始材料 | topic 引用；来源核验不等于实验复现 |
| 实验结果 | 对应实验记录 | topic 引用结果和边界，不抄一套不同数字 |
| 学习项目进度 | 对应 `projects/.../STATUS.md` | 最近更新仅摘要实际变化，里程碑依原记录判断 |
| 面试表达与题库 | 既有面试主文档、专题题库和 Coding 文档 | Part VII 聚合；项目素材作为事实依据，技术机制回链 topic |

内容完成程度使用“提纲／可阅读／需深化”等编辑描述；已有 `NEW → READING → ... → VERIFIED` 是材料生命周期。两者不混用，文件长不等于实验验证完成。

## 5. 统一 Tracking：兼顾模型认识和系统进展

### 5.1 范围调整

保留同一 `tracking/`、Source ID、scan log、backfill 和全局 P0/P1，不额外建立“具身新闻池”。

| 方向 | 必须能进入扫描视野的变化 |
|---|---|
| Training | PyTorch Distributed/FSDP、DeepSpeed、Megatron、数据训练栈、并行、精度、kernel、checkpoint/recovery |
| Inference | 独立 serving、vLLM/SGLang/TensorRT-LLM、KV 与调度、编译/量化、MLLM/生成式模型推理、实时执行 |
| RL/Agent | 现有框架 Watch、环境、reward、rollout、策略同步、样本与状态边界 |
| Embodied Models | MLLM 理解与交互、VLA、动作表示、Diffusion/Flow、World Model、模仿学习与 RL 的关键机制 |
| Embodied Infra | episode/trajectory、视频和时序数据、多模态 I/O、训练适配、仿真、评估、实时部署与反馈 |
| Shared Systems | GPU/网络/存储、可观测性、可靠性、资源利用与成本边界 |

### 5.2 接受标准有两条，而不再只有系统优化一条

1. **模型／学习机制价值**：材料能改变对模型输入输出、action 表示、训练目标、泛化或评估的理解，补当前具身模型认识的明确缺口；不要求事先证明 GPU 加速收益。
2. **系统价值**：材料改变数据契约、内存、通信、执行效率、时延、正确性、恢复或部署边界。

任一成立即可进入候选；最终 Decision 仍看重要性、可核验程度和当前问题。纯 demo、融资、无方法细节的榜单、营销内容通常不收录。发布开放权重本身也不等于新的系统能力。

模型论文可以因学习价值进入 Read，即使尚无开源代码；必须标注证据限制，不能称“可复现”或“已验证”。工程 PR 必须说明实际代码／测试／性能边界的变化，不能只依据 README 宣传。

每条候选仅新增必要字段：`Learning track`（可多选）、`Signal type`（model/algorithm/system）、`Evidence`、`Target question`。原有 Source ID、First seen、Window、Decision、Reason、Status、去向等继续保留；优先级与方向分开。

### 5.3 Watch 与覆盖

- 保留 OpenAI/Anthropic/NVIDIA/DeepSeek、Hugging Face、RL Framework Watch；修复 frontier 模板缺少 DeepSeek 的漂移。
- 增加简短 `Inference Systems Watch` 和 `Embodied Models & Infra Watch`，记录实际检查的来源、变化、证据和 Accepted/Observed/Rejected/Not found 或未能核验的情况，不强行凑信号。
- HF Watch 扩展到 LeRobot；具身来源覆盖 openpi/Physical Intelligence、OpenVLA、GR00T/Isaac Lab 等模型和代码，以及 PyTorch/TorchCodec、数据 I/O 社区的重要变更。名单用于发现，不构成自动推荐或强制采用。
- 多处 Watch 引用同一 Source ID，不重复计 Accepted。扫描资源不足时说明覆盖缺口，不假装全面覆盖。
- Monthly 仍只汇总对应月份实际扫描、backfill 和阅读结果，不为新范围重写历史报告。
- 新增追踪范围不等于补扫已完成；下次扫描须记录扩展生效时间及此前具身／推理覆盖缺口。旧材料单独走 backfill，或作为明示的补扫保留实际发布时间，不改旧游标伪装已经扫描。

更新范围包括 AGENTS/CLAUDE 的相关定位与筛选条款、tracking README 和两类模板；不创建或修改任何 scheduler/automation。

## 6. 近期重点一：FSDP、FSDP2 与 ZeRO 的深入路径

### 6.1 正文组织

现有 `topics/fsdp.md` 扩为共享的状态分片与训练后端章节；保留已有锚点和 Bridge 引用。`topics/zero.md` 保留 ZeRO 阶段速览与来源入口，并回链完整的状态账本和框架对照，不另维护一份重复教程。原理不在具身路径重复书写，具身只增加 workload 对照。

建议阅读顺序：

1. **DDP 基线与显存账本**：参数、梯度、optimizer、activation、通信缓冲、初始化与首个 optimizer step。
2. **ZeRO-1/2/3**：分别分什么，如何完成一次参数更新，常驻状态与临时工作集如何区别。
3. **FSDP1**：FlatParameter、wrapping、FULL_SHARD/SHARD_GRAD_OP/HYBRID_SHARD，能够读懂已有工程。
4. **FSDP2**：fully_shard、DTensor/DeviceMesh、hooks、bottom-up grouping、参数生命周期。
5. **性能取舍**：分组粒度、prefetch、reshard、gradient sync、重计算、mixed precision、topology、offload。
6. **正确性与恢复**：有效 batch、loss normalization、冻结/共享参数、梯度累积、checkpoint、随机与数据状态。
7. **框架选型与具身映射**：小模型不自动选 FSDP；区分模型状态与多图/视频 activation，DDP 可满足时先保留简单基线。
8. **源码与实验**：用固定版本追踪一层 forward/backward，再用 trace 与实测账本核对机制。

不能把 FSDP2 的 `reshard_after_forward=False` 写成与 DeepSpeed ZeRO-2 完全相同；不能把 PyTorch FSDP2 与 Megatron-FSDP 混称同一实现。

### 6.2 来源与产出边界

已核对的官方入口为 [PyTorch FSDP2](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html)和 [DeepSpeed ZeRO](https://www.deepspeed.ai/tutorials/zero/)。前者用于参数分组与执行生命周期，后者用于分片阶段及参数驻留配置。实施时锁定文档对应的软件版本，不直接照搬 main/stable 的最新接口到旧环境。

本规格只确定章节与验收；后续精读需依照仓库流程登记来源，更新现有 paper/topic/playbook 中的不足，不因新增标题把 TODO 笔记标成已完成。

## 7. 近期重点二：具身数据与 I/O

### 7.1 四层分析，不做工具名投票

| 层次 | 需要解释的问题 | 优先核对的实现 |
|---|---|---|
| 逻辑数据契约 | episode、camera、时间、state/action、mask、normalization 是否一致 | LeRobotDataset 的 schema 与 temporal sampling |
| 物理布局 | 小文件、较大视频/列式分片、TAR 的访问粒度和存储放大 | Parquet+MP4、WebDataset TAR；Lance 仅作有理由时的可选对照 |
| 读取、解码与预处理 | 随机时间窗口、seek、批量解码、resize、CPU/GPU 和 H2D 成本 | TorchCodec；DALI 作为 profile 指向该瓶颈后的候选 |
| 远端访问与缓存 | 本地 NVMe、共享文件系统、对象存储、整文件/块缓存 | 实际存储后端、fsspec、reader/decoder 的缓存能力 |

四层可以组合；WebDataset 与 DALI 不互斥，LeRobot 的数据语义也不等于只有一种读取实现。对象缓存、decoder 缓存和 DataLoader prefetch 分别讨论。

### 7.2 已核对的社区机制与研究问题

- [LeRobot v3](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)：多个 episode 共享较大 Parquet/MP4 文件，用 metadata 重建逻辑边界。用来研究文件数量、时序随机访问和元数据正确性。
- [TorchCodec](https://github.com/meta-pytorch/torchcodec)：媒体解码为 PyTorch tensor 的入口；CPU/CUDA、寻帧方式、批量请求和初始化成本需要在固定版本实测。
- [WebDataset](https://github.com/webdataset/webdataset)：TAR 分片与流式读取；另行验证 rank/worker 切分、shuffle、重复与遗漏，不能当成天然具备 episode 语义。
- [DALI video reader](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/operations/nvidia.dali.fn.readers.video.html)：预处理流水线候选；标准 reader 的 CFR 限制须先核对，不能用跳过检查替代时间正确性。
- [fsspec 本地缓存](https://filesystem-spec.readthedocs.io/en/latest/features.html#caching-files-locally)：整文件与块缓存的取舍；实际 reader/backend 是否兼容需验证，不预设任意组合均可插拔。

这些是设计参考来源，不是本次 frontier Accepted 清单。正式纳入时核对原始发布时间或具体变更，旧材料走 backfill；不以网页抓取日期代替发布日期。

### 7.3 内容落点

先按既有大纲完成 `topics/embodied_models_primer.md`；数据路径写入拟新增 `topics/embodied_data_pipeline.md`，以一个固定公开 episode 为例贯穿到训练 batch。

现有 `agentic_for_embodied.md` 保持高级系统蓝图身份，数据深入部分回链新章；不复制 FSDP 机制、通用 checkpoint 原理或已有 Agentic RL 调度全文。

“存储优化”必须同时回答：采样分布、像素/时间误差、标签对齐是否改变。将 MP4 改成 JPEG 的结果不能全部归因 I/O，因为编码、空间与像素都可能变化。

## 8. A100 实验课程：从机制验证到 64 卡

### 8.1 环境与授权

已知：A100，用户表示 64 卡以内资源充足。未知：40/80GB、SXM/PCIe、每机卡数、NVLink/NVSwitch、IB/RoCE/NIC、CPU/NUMA、存储和软件版本。

这些未知不阻塞课程设计，但阻塞具体容量、拓扑映射和启动命令。执行前必须补环境清单、允许节点／队列、数据与模型许可、时长和 GPU-hour 上限、停止条件。

原则：1 卡先验收，2–8 卡理解机制，需要验证跨机假设时再使用 16/32/64 卡。卡数是上限而非目标，不要求 1、8、16、32、64 每档都跑；真实单机边界以环境为准。

A100 以 FP32/BF16/FP16、通信、编译与 I/O 实验为主，不作为原生 FP8/FP4 Tensor Core 加速的验证平台。视频另核验 codec/profile 与 [NVIDIA 编解码支持矩阵](https://developer.nvidia.com/video-encode-decode-support-matrix)，不因 CUDA 可用就假设硬件编解码可用。

### 8.2 实验矩阵（全部未执行）

| ID | 要回答的问题 | 规模建议 | 对照及产物 |
|---|---|---|---|
| E01 状态分片账本 | DDP、ZeRO-1/2/3、FSDP1/2 到底省了哪些状态？ | 1 卡参考，2/4/8 卡 | 同一小模型，含 DeepSpeed stage 0；FP32 数值参考、BF16 性能分开，记录更新误差与实际显存 |
| E02 通信与生命周期 | 何时 all-gather/reduce-scatter，分组、reshard、prefetch 改变了什么？ | 2–8 卡 | FSDP2 root-only/逐 block、reshard 单变量；ZeRO-3 参数驻留另作消融，输出 trace/显存时间线 |
| E03 显存与计算取舍 | 重计算、mixed precision、梯度累积分别怎样影响吞吐和正确性？ | 2–8 卡 | E03a 只变重计算；E03b 只变精度配置；E03c 固定有效 global batch 改 microbatch/累积组合；每个子实验独立对照 |
| E04 多模态 I/O | 慢的是远端读取、seek、decode、预处理还是 GPU 等待？ | 单卡到单机，必要时跨机 | 同一 episode/sample manifest，比较文件布局、CPU/CUDA 解码、冷/热缓存；先 loader，再真实训练 |
| E05 恢复 | 保存成功是否真的能继续相同训练？ | 4→4 / 8→8；支持时再测 8→4 | 连续训练对照重启续训，核对样本游标、RNG、optimizer、step/loss/更新误差；不假设导出权重等于完整恢复 |
| E06 跨机扩展 | 全局分片与机内分片/机间复制何时划算？ | 单机起点到 16/32/64 卡 | 固定 recipe，分开测强扩展、弱扩展和容量扩展，记录慢 rank、exposed communication、吞吐/卡 |
| E07 推理 | 并发、KV、CUDA Graph 如何影响吞吐和尾延迟？ | 1–4 卡 | 固定 checkpoint/backend，先并发曲线再逐项启用特性；TTFT、TPOT、p95、有效吞吐；具身观测回放为可选子项 |
| E08 RL 链路 | 权重同步、策略版本和有效轨迹供给能否正确衔接？ | 总预算 4–8 卡起步 | 固定轨迹更新对照，再同步在线供给；记录生成/消费版本、mask/logprob、同步时间和 trainer idle |

执行优先序：E01 → E02 → E04；根据问题再展开 E03/E05，E06 不抢在单机解释清楚之前。E07/E08 保留训练以外的能力验证，不因本阶段重点而长期省略。

矩阵不强迫所有实验共享一个模型：状态分片用受控小 Transformer，E04 再接公开具身模型；不同 workload 结果不能放在同一吞吐榜上。

### 8.3 公平比较与指标

每项实验先固定模型实现、初始化、样本身份、loss normalization、有效 global batch/tokens、optimizer 数学定义、精度、kernel/backend 和 seed，再一次只改变目标变量。

分开两类结论：

- **机制对照**：控制额外 fusion/compile/offload 等变量；DeepSpeed stage 0 帮助观察框架额外成本，无法完全对齐的 dtype/master weight/optimizer 行为明示。
- **完整 recipe 对照**：允许各框架优化，但完整登记差异；只能结论“该 recipe 在这个 workload 更快”，不能把所有收益归因分片算法。

共同能运行的模型用于速度对比；某方案 OOM 时记录失败点并另做容量测试，不用不同模型的 tokens/s 排名。

强扩展固定模型和每次更新有效 global batch/tokens；弱扩展固定每卡工作量、允许 global batch 增长；模型本身变大则属于容量扩展。三类分别画图，不把弱扩展吞吐提升说成固定训练任务加速。

每次记录：最大 rank 的 step time、测量窗口、warmup、重复次数、allocated/reserved 峰值、有效 samples/tokens/frames、通信关键路径、CPU/存储负载和异常步。正式计时与 profiler 采样分开。

E04 额外记录时间戳/camera/mask/归一化一致性、读取字节与请求数、缓存条件、batch 等待。缓存按应用缓存、OS page cache、远端服务缓存分别记录：隔离缓存目录只能控制应用缓存，不能称全链路冷缓存；无法控制的层级标记未知，不清空共享机器全局缓存。

### 8.4 验证边界与安全

- 数值阈值按 dtype 与固定参考在执行前定义，不能看到结果后调整通过标准。
- 同拓扑恢复和改变 world size 的恢复分开；后者涉及数据重新分配、随机性和 backend 支持，不要求未经证明的 bitwise 一致。
- checkpoint 在独立实验目录保存，不能覆盖已有生产权重；故障注入只作用于已批准的实验进程，不影响共享节点其他任务。
- 数据使用公开或有明确授权的材料，不上传内部代码、权重、数据、节点地址或凭据到公开仓库。
- A100 上的视频回放和 action latency 不能证明端侧实时性或真机安全；模拟 producer 的轨迹实验不等于已完成机器人在线 RL。
- 原始大 trace、权重、视频留在获准实验存储；仓库只放可公开的配置、环境摘要、汇总指标、必要图表和复现说明。

### 8.5 实验文档组织

第一批先新增一份 `experiments/a100_fsdp_io_lab.md` 维护课程、环境门槛和 E01–E08 的未执行实验卡，由既有 `experiments/README.md` 进入。开始某项实验后才建立该项结果/脚本目录，避免一次创建八套空文件。

实验卡固定包含：问题、假设、前置条件、控制变量、资源上限、执行步骤、正确性门槛、指标、停止条件、实际结果、结论边界、回链。设计阶段“实际结果”明确写“未执行”，不能生成示例数字冒充实测。

## 9. 分批实施与文件清单

### 批次 A：导航、规则与课程入口

本规格审阅后可单独编制实施计划，目标是不搬旧正文、让阅读入口和后续采集规则先一致。

| 文件或区域 | 改动 |
|---|---|
| 根 README、手册 README、philosophy | 更新定位、当前重点、六个日常 Part＋一个按需面试 Part；首页增加最近更新，去掉互相等待的 Phase 描述 |
| `roadmaps/` 五份 Part README | 新增薄导航，只链接现有正文或明确标注待建设 |
| `interview/README.md` | 新增 Part VII，聚合主文档、专题题库、Coding、项目素材和专项准备；不移动旧文档 |
| 30/90/年度计划 | 明确既有基础路线身份，增加当前 Part 回链；不重写为另一套周待办 |
| MASTER_READING_LIST、KNOWLEDGE_GRAPH | 分离材料索引与知识依赖，补具身/推理路径，清理重复动态导航 |
| tracking README、frontier/monthly 模板 | 扩展范围、具身模型接受标准、Watch、覆盖缺口与去重规则 |
| AGENTS、CLAUDE | 同步定位、目录职责、未来扫描范围及“实质内容更新同步首页摘要”规则，保留既有发布/来源核验约束 |
| reading_queue README | 明确方向标签和全局 P0≤3；不凭结构调整改变材料阅读状态 |
| experiments README、A100 lab | 增加未执行课程入口与实验卡，保留既有实验和状态 |
| 根知识地图资产 | 检查旧 Phase 是否冲突；若冲突，在原资产内更新标签/关系并验证，不另开网站 |

本批次不直接删除或重排在读 P0 条目；后续当前学习队列调整须保留已有进度并说明转移去向，不把降优先级误记为读完。

### 批次 B：近期内容深化

分别验收：具身模型 primer → FSDP/ZeRO 深入 → episode 到 batch 的数据路径。每项遵守来源核验、正文唯一维护位置和双向链接规则；补正文时同步 Part 的“待建设”状态。

推理入口首版可指向现有 RL 选型章中的 serving 内容和 CUDA Graph 等章节，但必须承认是局部覆盖，不称已经拥有完整推理手册。独立推理正文根据后续研究问题建设，迁移时保留旧锚点或回链。

### 批次 C：实验实现与执行

先选 E01/E02/E04 中的一项编写独立可执行计划；核对 A100 环境与版本、取得具体执行授权后再运行。从数值正确性与小规模开始，记录真实结果再扩卡。

结构验收不依赖 GPU 结果；实验课程验收也不等于代码已实现或性能已验证。

## 10. 验收标准

### 结构与导航

- 首页可直接进入六个日常 Part 和一个独立的按需面试 Part；由 Part 到主正文通常再点击一次。
- 每个 Part 明确阅读顺序、前置知识、已有内容与缺口；返回首页和学习地图链接有效。
- FSDP/ZeRO、通信、checkpoint 等公共机制没有产生不同方向的重复全文。
- 面试主文档路径及已有问题锚点保留，不引入公司定制复习作为学习主线。
- Part VII 能找到主文档、专题题库、Coding、项目素材和专项准备；日常学习无需经过面试准备。
- 首页最近更新最多 5 条，内容和日期能追溯到实际提交及正文；包含项目更新时不得把计划写成完成或误用项目发生日期。
- 全库最近更新只有首页一份；项目状态、个人素材、实验结果和外部 tracking 各保留原有职责，不新增重复正文。
- 不把具身等同 RL，不把推理仅当 rollout backend，不默认“小模型必须用 FSDP”。

### 追踪与证据

- 新规则覆盖训练、独立推理、RL、具身模型和具身系统；模型学习价值可以独立构成阅读理由。
- Watch 重复引用不重复计数；历史 scan、月报、Accepted 数量与游标不因结构升级改变。
- 原有在读状态未被批量完成；来源核验、教程完成与实验验证严格区分。

### 实验与发布

- A100 课程有正确性门槛、资源/环境前置项、停止条件与未执行标识。
- 不存在未经授权的作业启动、环境安装、数据下载、共享缓存清理或 checkpoint 覆盖。
- 修改范围内 Markdown 路径、锚点和图片有效；如修改 SVG，检查 XML 与实际布局。
- 发布前查看工作区、fetch origin/main，保护用户及其他任务的未提交改动；默认只提交本任务文件到 main，不新建远端分支或 PR。若分支保护强制要求 PR，停止直接发布并说明限制、请求方向，不绕过保护规则。
- 与本任务无关的既有或并发修改不覆盖、不暂存、不代为发布。

## 11. 本轮实际交付状态

本轮只修订这份结构与课程规格，已纳入独立的面试 Part 和首页最近更新设计。此前官方来源已做针对性核对，但没有执行全量 frontier scan，没有改动主 README、Part、队列、追踪规则或实验入口，也没有运行任何 GPU 实验。

后续顺序：用户审阅本规格 → 为批次 A 制定实施计划并落地 → 分别深化批次 B 内容 → 根据环境与授权实施批次 C 实验。

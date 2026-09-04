# 简历驱动速查控制台与双主线能力图设计

## 目标

把面试准备主文档进一步收敛为可反复复习、考场可快速跳题的实战题库：

1. 删除开头关于面试官评估维度、回答原则等抽象说明，正文直接进入速查入口；
2. 将控制台从“项目 / 技术 / 数字”三套并列索引改为一套简历驱动索引；
3. 以教育背景、工作技能、项目经历为顺序，其中项目经历是主体，所有入口落到唯一题目正文；
4. 保留能力图现有知识内容和项目证据，重做为主线清晰、留白充分、色彩素雅的博客/论文级 SVG；
5. 将目标档位更新为“当前年薪约 80 万，目标 100–150 万”。

目标文件：

- `private_resume/2026-08-llm-infra-interview-prep.md`
- `private_resume/assets/llm-infra-personal-capability-map.svg`

不修改 79 道问题的题号、优先级、正文答案、数字口径和 Part 归属。

## 主文档开头

### 删除内容

仅删除现有主文档第 9–24 行、即旧 `## 0. 先看结论：面试官会如何评估你` 标题到 `<a id="interview-console"></a>` 之前的抽象说明，包括：

- 六项抽象评估维度；
- “五句法”说明；
- 没做过的能力如何表达的通用提示。

不得整体误删或改变 `<a id="interview-console"></a>`、控制台、能力图和题库的先后位置；控制台内容和能力图仅按后续章节重构，题库不得移动或修改。被删抽象内容不在别处复制；每道题现有的“面试官意图、项目证据或知识边界、危险回答”已经承担具体提醒。

### 新的开头层级

元信息之后直接进入：

```text
<a id="interview-console"></a>
## 0. 考场速查
### 0.1 面试现场速查控制台
### 0.2 一张图看懂我的能力主线
## 1. 整体视野与问题导航
```

保留 `#interview-console` 稳定锚点，避免 79 道题的返回链接失效。保留简短使用说明和“现场救急”行；它们是操作入口，不是另一套索引。

## 0.1 简历驱动的唯一索引

删除现有三块并列导航：

- 从项目经历进入；
- 从技术主题进入；
- 按关键数字反查。

替换为一张三列主表：

| 简历区块 | 简历内容 / 面试切入点 | 高频题目入口 |
|---|---|---|

第一列只允许出现用户确认的三个区块：`教育背景`、`工作技能`、`项目经历（核心）`。所有内容按真实简历阅读顺序组织，答案仍只存在于原题正文。

### 教育背景

控制在 1 行：

1. 厦门大学本科、清华大学硕士、AI 研究方向 → `RESUME-01` 自我介绍。

教育信息只作面试入口，不展开课程、论文或奖项。`RESUME-01C` 职业选择继续保留在“现场救急”和完整索引，不硬塞进教育背景；`RESUME-01B` Ownership 进入 X1 项目行。

### 工作技能

控制在 4 行，每行按简历技能域链接唯一题目：

1. Megatron / 分布式训练：`MEGATRON-01/02/04/05/11`、`DIST-01`；
2. MoE / 长上下文 / 显存性能：`MOE-01`、`MEGATRON-06`、`INFRA-02`、`KERNEL-01`；
3. RL / verl / AReaL：`RL-ALGO-01`、`AREAL-01`、`VERL-01/02/04`；
4. Rollout / 通信 / 稳定性：`ROLLOUT-01`、`VERL-09`、`RESUME-13/14`、`INFRA-04/09`、`TRAIN-ANOMALY-01`。

这里回答“简历上写了会什么”，只保留框架和机制入口；不复制项目数字。

### 项目经历（核心）

控制在 6 行，并占主表主要视觉篇幅。关键数字合并进所属项目行，不再单独反查：

1. X1 200B MoE：`0.16x→0.95x / MFU 35% / 3K 卡两个月`，链接 `RESUME-01A/01B`、`MEGATRON-01`、`MOE-01`、`RESUME-10`；
2. Long Context SFT：`31s→9.3s；MFU 23%→45.2%（独立简历口径，不据此互相反推）/ 128K / 7.6GB`，链接 `RESUME-05/17/06/07`；
3. Fully Async RLVR：`async 内部配置优化：76→211–255 tokens/s/GPU`，链接 `RESUME-02/03`、`ROLLOUT-01`、`VERL-02/04/05`；
4. Agentic RL / Gateway：`decode 6–8x / rollout +60% / Rejected Group 33.18%→2.73%`，链接 `RESUME-08/13/19`、`AREAL-09/11`；
5. OPD / MOPD：方向性结论为“双 Teacher 在 SWE、Terminal 双域提升且 General 不下降”，不增加“显著”等统计措辞；链接 `RESUME-09`、`AREAL-04/08`；
6. TX 文生视频 / 国产卡规模交付：链接 `RESUME-18/10/12`、`KERNEL-01`、`INFRA-09`。

同一题可以被简历技能和项目经历同时引用，但正文只有一份；控制台不再提供第三套“按数字反查”视图。

## 0.2 双主线能力图

### 信息架构

能力图采用从上到下的三层结构，替代六色放射图：

1. **定位层**：顶部居中只保留 `LLM Infra`、`TRAINING · RL · ROLLOUT`；不再使用拥挤的中心大卡片和曲线；
2. **能力层**：两条并列主线加一层共同工程底座：
   - `Megatron 训练主线`：训练系统 / Megatron、通信与拓扑；
   - `RL / Agentic 主线`：RL 与后训练、推理与 Rollout；
   - `共同工程底座`：显存与性能、正确性与规模化交付。两张底座卡横跨两条主线，不归入任意一侧；其中国产卡适配/千卡性能交付属于 Training 项目证据，但规模化交付方法是共享能力；
3. **证据层**：底部保留六张项目证据卡，并按两条主线使用 `2 列 × 3 行` 排列：
   - 左列 Training：X1 200B MoE → SFT / Long Context → X1 / TX 国产卡规模交付；
   - 右列 RL / Agentic：Fully Async RLVR → AReaL Agentic RL → MOPD。

共同工程底座使用一条横跨两列的细窄标题带：

```text
性能 × 正确性 × 可恢复性 × 规模化交付
```

标题带下方并列放置“显存与性能”和“正确性与规模化交付”两张中性卡。它表示这些目标同时约束训练与 RL/Agentic 两条主线，不是第三条主线。

### 布局

- 保持横向阅读友好的 SVG，建议 `viewBox="0 0 1680 1360"`；最终尺寸可在不改变结构的前提下微调。
- 顶部定位层约占 12%；两条主线核心能力约占 30%；共同工程底座约占 18%；证据层约占 36%；底部留白约占 4%。
- 两条主线各占约一半宽度，中间留 56–72 px gutter。
- 六个能力卡片仍使用 2 列 × 3 行严格对齐：前两行分别归属 Training 与 RL/Agentic；第三行放在横跨两列的共同工程底座容器内，两个子卡不继承左右主线归属。卡片高度、标题基线、项目符号和行距统一。
- 六张证据卡使用与能力层一致的 2 列 × 3 行；关键数字最多两行，说明最多两行，不允许文本贴边或相互覆盖。
- 取消所有曲线、同心圆、箭头/marker、大阴影和装饰波形；只保留无箭头的细分隔线、小圆点和非常轻的层次阴影。

### 色彩与字体

使用同一冷色族的两档低饱和主线强调色：

- 页面背景：暖灰白 `#F7F7F4`；
- 主文字：深灰绿 `#1F2A28`；
- 训练主线：灰蓝 `#526B73`；
- RL / Agentic 主线：深青 `#2F7169`；
- 卡片背景：白色或极浅灰绿 `#FFFFFF / #F0F4F2`；
- 边框和辅助线：`#D5DEDB`；
- 次要文字：`#64716E`。

灰蓝和深青只用于区分两条主线，不构成多彩配色；不再为六个能力域分别使用蓝、橙、红、绿、紫、青。字体继续使用系统无衬线字体栈，标题、标签、正文只保留三档字号和三档字重。

### 必须保留的内容清单

允许为了换行与整齐调整标点、缩写和词序，但以下语义必须完整保留：

| 能力域 | 实心项目证据关键词 | 空心原理/延伸关键词 |
|---|---|---|
| 训练系统 / Megatron | `5D 并行配置与调优`；`Grouped MatMul / 融合算子`；`128K–256K / SP/CP / recompute`；`packing / DataLoader / prefetch` | `Folding / DeepEP / FP8/FP4` |
| 通信与拓扑 | `AR / RS / AG / A2A / P2P`；`NCCL / XCCL 集成与排障`；`compute–comm overlap / exposed` | `NVLink / IB / RoCE / HCCS 精细映射` |
| 显存与性能（共同底座） | `参数 / 梯度 / 优化器 / activation`；`logits / KV / workspace / OOM`；`profiling / MFU / 瓶颈迁移`；`recompute / fusion` | `offload / Three Walls` |
| RL 与后训练 | `verl / SFT/RLVR / Fully Async`；`AReaL / Gateway / Session / Cohort`；`Agent / Tool / Sandbox / goodput`；`MOPD / 双域提升 / General 不下降` | 无 |
| 推理与 Rollout | `vLLM / SGLang rollout`；`gen-TP / 多实例 / batching`；`KV / Prefix Cache / CUDA Graph`；`weight sync / policy version` | 无 |
| 正确性与规模化交付（共同底座） | `loss / logprob / mask / 精度对齐`；`FUNCTIONAL / NUMERIC / EFFICACY`；`checkpoint / recovery / lineage`；`国产卡适配 / 千卡性能交付闭环` | 无 |

证据卡必须保留以下内容和边界注记：

| 归属 | 证据卡 | 必须保留 |
|---|---|---|
| Training | X1 200B MoE | `0.16x→0.95x`、`MFU 35% / 3K 卡`、`跑通→性能达标闭环` |
| Training | SFT / Long Context | `Qwen3 / Qwen3.5`、`32K–256K`、`data / recompute / memory / ckpt`、`特定 workload 口径` |
| Training | X1 / TX 国产卡规模交付 | `适配→Profile→优化→验证→扩容`、`模型跑通与性能达标闭环`、`非集群平台 / 运维总 owner` |
| RL / Agentic | Fully Async RLVR | `76` 初始 `3T+1R / gen-TP=4`、`211–255 tokens/s/GPU`、`gen-TP 4→2`、`Async 内；236–293=2T+2R 候选` |
| RL / Agentic | AReaL Agentic RL | `Gateway / Session / Cohort`、`Version / ready-cohort`、`长尾 / trajectory goodput`、`外部 Agent / Env 边界清晰` |
| RL / Agentic | MOPD | `RL Experts→Teachers`、`Pre-RL model→Student`、`原 RL 数据 OPD / FUNCTIONAL`、`SWE / Terminal 提升；General 不下降` |

实心圆点继续表示项目实战；空心圆点仅用于上表三类原理/延伸项。图例必须明确这两种语义。

### 内容边界

- 不新增未经简历或项目底稿确认的技术、数字和 ownership。
- `X1 200B MoE` 不加“约”；MOPD 继续使用“双域提升且 General 不下降”的最新口径。
- 图片正文不放联系方式、客户全名、内部仓库和未脱敏信息。

## 保留内容

- Core 10、Part I–VI、全量问题索引和所有 79 道正文保持原位置与内容；
- 79 个题尾 `返回本 Part 导航 / 返回面试速查控制台` 保持可用；
- Coding 题单、面试进度附录和现场救急入口保留；
- 图片路径和 Markdown alt text 不变，避免外部引用失效。

## 验收标准

### 主文档

- 目标档位精确为“当前年薪约 80 万，目标 100–150 万”；旧 `100–120 万` 不再出现。
- `## 0. 先看结论：面试官会如何评估你`、六项评估维度、“五句法”和通用提示从主文档删除。
- `#interview-console` 锚点唯一，且位于 `## 0. 考场速查` 前；`0.1`、`0.2` 层级连续。
- 控制台只有一张简历驱动三列表格：1 行教育背景、4 行工作技能、6 行项目经历；第一列只出现三个确认区块，不再出现三套旧标题。项目经历为 6/11 行，项目名、数字和链接加粗，形成主体。
- 项目数字只在所属项目行出现；`async 内部配置优化`、SFT 两组数字不可互推、MOPD 方向性结论三项口径边界不得丢失；所有控制台链接指向现有唯一答案。
- 题目 ID 集合、Part ID 集合必须与基线 `3498ed5` 完全一致；79 道题、P0/P1/P2=`48/26/5`、Part I–V=`7/27/17/18/10` 保持不变。
- 所有显式 `id` 唯一；Core 10 目标 ID 与顺序不变；所有内部锚点、题尾双回链、相对文件链接和图片路径均可解析。

### SVG

- SVG 可解析为 XML；title/desc 与双主线结构一致。
- 只有灰蓝与深青两种主线强调色，其他颜色是中性背景/文字/边框。
- 无放射曲线、同心圆、箭头/marker 和拥挤中心卡片；前两行主线能力、第三行共同底座以及 2×3 证据卡严格对齐，共同底座没有 RL/Training 单侧归属暗示。
- SVG self-contained：不引用外部字体、图片、脚本或 `foreignObject`。
- 分别按桌面 1400 px 和窄屏 900 px 宽度生成预览；两种尺度下均检查无文字溢出、遮挡、截断，主线标题、关键数字、边界注记和图例可读，列间 gutter 清楚。
- 实心/空心圆点含义仍有简洁图例。

### Git 与安全

- 发布基线固定为 `3498ed5`；相对该基线只新增本设计、后续实施计划，并修改主文档与能力图 SVG。复审通过后的设计 HEAD 由实施计划记录，避免在规格正文中维护自引用 commit hash。
- 开工前主工作树基线必须记录并在完成、发布后逐项比对：
  - `status --porcelain`：仅三个 tracked dirty 文件；
  - `git diff --binary` SHA-256：`d01efc17b7ebb0c2be22c762e5fa228823f3c60e2e5354f2a1d11a468a7540f8`；
  - `docs/superpowers/specs/2026-09-01-parallel-folding-topic-design.md` SHA-256：`ff96f2533a4e2ce37639f5b0700e9a979c5e0fbabda5c53925ae03f2f7ecc715`；
  - `private_resume/2026-08-llm-infra-interview-prep.md` SHA-256：`a4af2de96006c02e27205eb49e0ca2a7beee7502163088bfda31bccc7976aaee`；
  - `training-infra-roadmap/topics/data_parallelism.md` SHA-256：`11049a96578780e8b82f011cf725f2bb4c2cd618d01da4721470d75aa58ca0eb`。
- 所有改动和发布在隔离 worktree 完成；禁止在 `/Users/zengbw/ReadBase` 主工作树执行写文件、switch、merge、rebase、pull、reset、commit 或 push。
- 发布前 fetch 并确认 `origin/main` 是待发布 HEAD 的祖先；只推送 `HEAD:main`，不创建或推送功能分支。
- 只允许隔离 worktree 执行 `git push origin HEAD:main`。发布后确认本地 `origin/main` 与 HEAD 一致，并复核主工作树 status、binary diff hash 与三个 dirty 文件内容 hash 未变化。

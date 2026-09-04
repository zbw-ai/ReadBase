# 智元机器人训练 Infra JD 临场补题设计

## 目标

把已经确认的岗位差集并入 `private_resume/2026-08-llm-infra-interview-prep.md`，让内容既能服务今天的 30 分钟冲刺，也能长期留在系统化题库中。本次唯一 JD source of truth 是用户在当前对话中粘贴的“训练 Infra 岗位”职责与任职要求，不把其他公开招聘页面中未出现的 LoRA、结构化输出或端侧量化等要求扩入本轮范围。

## 设计原则

1. 不建立新的内容分支文档；完整答案只进入主文档。
2. 按知识归属分散到 Part II、III、V，而不是建立孤立的“智元专题”。
3. 新增四题均按当前岗位 P0 处理；已有 `MEGATRON-11` 扩展 DeepSpeed/Accelerate 的分层与组合关系，不新增重复 ID；Core 10 不变。
4. 在控制台“现场救急”增加 `智元训练 Infra（30min）`，跳转到 Part VI 的 30 分钟冲刺清单；清单再链接五道完整答案。
5. 每道新题继续使用现有结构：问题、面试官意图、精准回答、追问、边界/危险回答、双回链。
6. 同步更新题量、Part 计数、全量索引和本 Part 导航。

## 问题归属

| ID | 问题 | Part | 原因 |
|---|---|---|---|
| `SFT-DATA-01` | SFT 数据从原始样本到 loss，如何保证没有训错？ | Part II | 数据、tokenizer、mask、packing 与训练正确性 |
| `MLLM-01` | 多模态/具身训练与纯 LLM 训练有什么不同？ | Part II | 多模态 token、数据 pipeline、显存与并行 |
| `DPO-01` | DPO 如何工作，与 SFT、PPO/GRPO 怎么选？ | Part III | 后训练算法及其系统数据契约 |
| `TRAIN-ANOMALY-01` | loss 震荡、NaN、梯度爆炸或收敛慢怎么定位？ | Part V | 生产训练异常与故障定位 |

另扩展现有 `MEGATRON-11`：先说明 Accelerate 是接入/编排 facade，DeepSpeed、FSDP/FSDP2、Megatron 是不同层次且可组合的训练能力，再按模型规模、多维并行、offload、生态和团队资产做决策。个人边界固定为：Megatron-Core 有生产使用与集成证据；DeepSpeed、Accelerate、FSDP/FSDP2 只表述机制理解和选型判断，不声称未经确认的生产落地。

`MLLM-01` 必须区分 MLLM 与具身/VLA：直接项目证据仅包括 TX 视频/图像模型的国产卡迁移、功能/精度/性能闭环，以及自研 verl 对 Capek MLLM 后训练的 Infra 承载；不得升级为机器人真机数据、VLA 或具身算法训练 ownership。具身部分只从工程机制说明 observation/action/episode 时序、控制输出和闭环评测带来的新增约束。

## 计数变化

- 总题量：74 → 78
- P0：43 → 47；P1 26、P2 5 不变
- Part II：25 → 27
- Part III：15 → 16
- Part V：9 → 10
- Part I 7、Part IV 18 不变

完整矩阵：Part II 为 Core 3 / P0 20 / P1 6 / P2 1，共 27；Part III 为 Core 1 / P0 10 / P1 5 / P2 1，共 16；Part V 为 Core 1 / P0 4 / P1 5 / P2 1，共 10。

## 30 分钟冲刺入口

在现有 VI.0 标题后、三天计划之前增加稳定锚点 `<a id="vi-0a"></a>` 和 `VI.0A｜智元机器人训练 Infra：30 分钟补题`，顺序与配额固定为：

1. `TRAIN-ANOMALY-01`：8 分钟。JD 明确枚举 OOM、loss 震荡/NaN、梯度爆炸、收敛慢和通信瓶颈，是最高概率追问；新题补数值与收敛异常，并交叉链接已有 OOM 与通信排障答案，避免重复正文。
2. `MEGATRON-11`：6 分钟。补 DeepSpeed/Accelerate/FSDP/Megatron 分层与组合。
3. `SFT-DATA-01`：6 分钟。覆盖数据清洗、分词、掩码、对齐与评估指标。
4. `DPO-01`：5 分钟。覆盖 JD 明确列出的 DPO/PPO/RL 算法选择。
5. `MLLM-01`：5 分钟。把已有视频/MLLM 证据迁移到多模态/具身工程判断，但守住 ownership 边界。

总计必须恰好 30 分钟。顶部现场救急入口使用不过期名称 `智元训练 Infra（30min）` 并只跳到 `#vi-0a`；VI.0A 只写时间分配和跳转，不复制答案。

## 验收条件

- 四个新题锚点与 `vi-0a` 存在且唯一；顶部入口唯一指向 VI.0A，VI.0A 唯一链接四个新题和扩展后的 `MEGATRON-11`。
- 78 道问题逐个按题目 body 校验：题尾各有且仅有一条所属 Part/总控制台双回链，且不位于 fenced code/Mermaid 中。
- 题量为 P0/P1/P2=`47/26/5`，Part I–V=`7/27/16/18/10`；同时校验每个 Part 的 Core/P0/P1/P2 矩阵，Core 10 内容与顺序不变。
- 四个新题分别在正文、全量索引、本 Part 导航与 VI.0A 中出现且指向一致；`MEGATRON-11` 仍为原唯一题，只扩展回答与入口描述。
- 所有 Markdown 本地链接、图片和显式锚点可解析，SVG 可解析为 XML。
- 除主文档与本设计/实施计划外不修改其他路径。
- 实施前记录主工作树 `status -uall`、dirty binary diff SHA-256、目标主文档 SHA-256，并在最终发布后确认三项仍相等，不能只比较是否仍显示 `M`；`main...origin/main` 只作为推送前背景快照，不要求推送后保持相等。
- 发布前 `git fetch origin`，确认 `origin/main` 是 HEAD 祖先且只做 fast-forward；发布后确认 `origin/main == HEAD`、远端只有 `main`。不创建远端功能分支，不 force push。

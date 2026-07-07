# Historical Backfill

`historical_backfill.md` 是历史补录总入口。它不直接承载越来越长的材料清单；具体条目按材料原始发布时间月份放入 [backfill/](backfill/README.md)。

Historical Backfill 不是前沿扫描，也不是 weekly/monthly signal。

| 类型 | 作用 |
|---|---|
| [Frontier Scan](frontier_scan_template.md) | 从上次扫描游标到现在，捕捉最新前沿信号 |
| [Scan Log](scan_log.md) | 记录每次扫描窗口和下一次游标，保证不重不漏 |
| Historical Backfill | 补录过去已经证明重要、但仓库还没吸收的经典材料 |
| [Monthly Signal](monthly_signal_report_template.md) | 从当月扫描、补录和阅读结果中精选正式判断 |
| [Reading Queue](../reading_queue/README.md) | 从 frontier/backfill/monthly 中筛选真正要读的 P0/P1 |
| Topics / Insights / Playbooks | 最终沉淀位置 |

## 定位

历史材料不要按“最近看到什么”来补，也不要按主题拆得过细。

更好的方式是：

```text
按原始发布时间月份倒序补课
  ↓
每个月一个 backfill/YYYY-MM.md
  ↓
每条材料说明它补哪个工程判断缺口
  ↓
重要材料进入 P0/P1
  ↓
读完后沉淀到 topic / insight / playbook / experiment
```

例如今天发现一篇 2025-05 的经典 rollout infra 论文，不放到今天的 frontier scan，也不塞进一个巨大的 historical_backfill 文件，而是放到：

```text
tracking/backfill/2025-05.md
```

并在条目中记录：

```text
原始时间：2025-05
补录时间：2026-07-07
```

## 月份索引

已开始整理：

- [2026-02](backfill/2026-02.md)：RLHF 资源弹性
- [2025-08](backfill/2025-08.md)：agent runtime 与 RL trainer 解耦
- [2025-05](backfill/2025-05.md)：异步 RL 系统
- [2025-04](backfill/2025-04.md)：vLLM 进入 RLHF rollout
- [2025-03](backfill/2025-03.md)：reasoning RL recipe 与系统栈
- [2025-01](backfill/2025-01.md)：DeepSeek-R1 与 reasoning RL 需求爆发
- [2024-12](backfill/2024-12.md)：DeepSeek-V3 与大规模训练工程底座
- [2024-09](backfill/2024-09.md)：verl / HybridFlow 与 RLHF dataflow
- [2024-05](backfill/2024-05.md)：OpenRLHF 与开源 RLHF pipeline
- [2023-08](backfill/2023-08.md)：DeepSpeed-Chat 与早期 RLHF 工程化

待补：

- 2026-06：NVIDIA NeMo RL / Megatron RL 等持续更新文档，需按具体 release 或文档版本确认。
- Ray RLlib / Ray Train：属于长期演进材料，需要拆到具体版本节点后再归档。

## 待确认月份材料

这些材料已经被识别为有价值，但还需要确认原始发布时间月份或具体版本节点后再放入 `backfill/YYYY-MM.md`。

### SkyRL

- 原始时间：2025，月份待确认
- 补录时间：2026-06
- 类型：repo
- 链接：https://github.com/NovaSky-AI/SkyRL
- 为什么现在补录：AReaL/verl 之外，需要一个 long-horizon tool-use agent training 栈作为横向参照。
- Decision：Read
- Reason：它补的是“多轮工具调用 agent training 如何工程化”的判断缺口。
- 建议动作：进入 P1
- 关联主题：[Agentic RL](../topics/agentic_rl.md), rollout / environment / evaluation
- 最终应流向：engineering blog / topic / experiment
- 生命周期状态：NEW

### NVIDIA NeMo RL

- 原始时间：2026，具体 release 月份待确认
- 补录时间：2026-06
- 类型：doc / repo
- 链接：https://docs.nvidia.com/nemo/rl/latest/index.html
- 为什么现在补录：NVIDIA 已经把 GRPO、DAPO、reward environment、vLLM rollout、Megatron backend 放进同一个 post-training stack。
- Decision：Read
- Reason：它补的是“NVIDIA training stack 如何进入 RL/post-training”的判断缺口。
- 建议动作：进入 P1
- 关联主题：[Agentic RL](../topics/agentic_rl.md), [Transformer Engine](../topics/transformer_engine.md), [NCCL](../topics/nccl.md)
- 最终应流向：engineering blog / topic / playbook
- 生命周期状态：NEW

### Ray RLlib / Ray Train

- 原始时间：2020-2026，需拆到具体版本节点
- 补录时间：2026-06
- 类型：doc / paper
- 链接：https://docs.ray.io/en/latest/rllib/index.html
- 为什么现在补录：OpenRLHF、AReaL、SkyRL 都大量依赖 Ray-style actor/scheduler 思维。
- Decision：Observe
- Reason：它补的是“RL pipeline 底层调度抽象”的背景缺口，但不应抢占 P0。
- 建议动作：仅索引
- 关联主题：[Distributed Training](../topics/distributed_training.md), [Agentic RL](../topics/agentic_rl.md)
- 最终应流向：engineering blog / topic
- 生命周期状态：NEW

## 记录模板

每个 `backfill/YYYY-MM.md` 建议使用：

```markdown
# Historical Backfill, YYYY-MM

## 本月核心判断

这个月的历史材料补齐了哪个工程判断缺口？

## 方向小节

### Title

- 原始时间：
- 补录时间：
- 类型：paper / blog / report / repo / talk / doc
- 链接：
- 为什么现在补录：
- 历史影响：
- 今天是否仍有价值：★★★★★ / ★★★★☆ / ★★★☆☆ / ★★☆☆☆ / ★☆☆☆☆
- Decision：Ignore / Observe / Read / Deep Dive
- Reason：
- 建议动作：进入 P0 / P1 / 直接沉淀到 topic / 仅索引
- 关联主题：
- 最终应流向：paper note / engineering blog / topic / insight / playbook / experiment
- 生命周期状态：NEW / READING / SUMMARIZED / DIGESTED / VERIFIED / IMPLEMENTED / OBSOLETE
```

## 质量规则

- 不追求补全历史，只补当前工程判断缺口。
- 不按主题拆文件，避免碎片化；主题关系写在条目字段里。
- 不把历史材料混入 frontier scan。
- 不因为材料经典就自动进入 P0；必须解释为什么现在必须读。
- 每条材料控制在 10 到 15 行左右，避免把 backfill 写成论文笔记。
- 读完后应减少 backfill 的负担，把长期价值沉淀到 `topics/`、`insights/`、`playbooks/` 或 `experiments/`。

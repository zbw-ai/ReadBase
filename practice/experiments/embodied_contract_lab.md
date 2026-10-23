# 具身学习练习：从样本、训练到评估与执行

[个人学习安排](../../05-embodied-infra/roadmap.md#study-sessions) · [具身核心入口](../../05-embodied-infra/topics/embodied_infra.md#discussion) · [数据正文](../../05-embodied-infra/topics/embodied_data_io.md)

状态：NEW / 练习设计；尚未执行，不是机器人控制脚本。先纸面或只读数据检查，再决定是否运行 GPU / 仿真实验。真实硬件与远端训练不在本记录的执行范围内。

## 问题与假设

观察与动作对不齐、padding 跨 episode、或已经提交的动作被迟到预测覆盖，都可能让局部模型指标无法代表闭环表现。这是待验证假设；练习旨在定位能观察到的证据，而非预设所有故障都由它们造成。

<a id="e1"></a>
## E1：一条 Episode 的窗口

使用纯教学数据：episode A 有动作时刻 0.00、0.05、0.10、0.15 秒，0.15 是最后有效动作；episode B 从 0.00 重新计时。A 的 anchor=0.10，预测三步。列出每个动作位置、来源 episode 和有效 mask；再令一个相机缺少 0.10 秒帧，明确拒绝、最近邻或其他策略。

验收：不得把 B 的第一帧填入 A；能说明 padding 是否参与 loss，以及相机对齐策略的误差边界。保存预期和实际结果，执行前不填“通过”。

<a id="e2"></a>
## E2：迟到的 Action Chunk

用假设时钟：在 0 ms 采样，50 ms 完成 chunk 预测，动作间隔 20 ms；到 90 ms 收到基于 20 ms 观察的新预测。先定义旧 chunk 从哪一时刻开始执行，再画哪些位置已提交、哪些允许替换，明确动作目标时间以观察时刻还是执行起点锚定，并分别计算两份预测返回时和实际执行时的 observation age；比较时必须使用相同参考时刻。

验收：不能只用请求完成顺序覆盖队列；同时说明时钟域和安全回退由谁负责。例子不设通用正确控制策略，评价的是是否给出完整契约与可检验结果。

<a id="e3"></a>
## E3：更换 Decoder / 数据布局

具备小型授权数据集后，固定 dataset revision、episode、时间窗口、变换和 RNG；比较优化前后的返回时间戳、帧内容误差、action、mask、有效样本数量，再测读取／解码／H2D。容差必须预先声明。

验收：语义与 split 一致才比较性能；记录冷暖缓存、worker 数量和 rank 分布。具体测量复用 [A100 E04](a100_fsdp_io_lab.md#e04)，不另建训练框架。

纸面预习可以先重算[batch 内存与流水线例子](../../05-embodied-infra/topics/embodied_data_io.md#io-case)，不需要数据集；该算例不填入性能实验结果。

<a id="e4"></a>
## E4：两个 rank 怎样学到同一个目标

对应[训练正文](../../05-embodied-infra/topics/embodied_training.md#loss-normalization)。教学任务固定等权、归一化动作的 L1：rank 0 有 8 个有效动作步，每步 7 维，均值误差 0.2；rank 1 有 2 个有效步，均值误差 0.6。

先独立完成四项：算本地误差和与全局 mean；解释为什么平均两个 local mean 不对；在 DDP 默认梯度平均下写本地反传缩放；将同一批有效元素拆成两个 microbatch 后保持目标不变。

<details>
<summary>完成后展开参考推导（不是实验结果）</summary>

有效数是 56 与 14，误差和是 11.2 与 8.4；全局目标为 `19.6/70=0.28`。`W=2` 时本地反传标量为 `2×11.2/70=0.32`、`2×8.4/70=0.24`；DDP 平均梯度后对应全局有效元素目标。累积时按整个 optimizer step 的有效计数缩放，不能每个 microbatch 先各自平均后再默认除以累积次数。

</details>

验收：说明这个公式的目标、process group 与默认梯度规约假设。数值扩展再对比单进程合并 batch 和两 rank 的一步参数更新，固定输入、初始权重及随机性，并先使用没有 batch-dependent 行为的小模型；普通 BatchNorm 或分组变化后的 Dropout 会引入额外差异，仅固定 seed 不足以隔离它们。当前未运行。另画 trainable adapter → frozen block → action head，说明哪个参数不更新、哪段仍需传递输入梯度。

<a id="e5"></a>
## E5：一份成功率报告漏掉了谁

对应[评估账本](../../05-embodied-infra/topics/embodied_sim_evaluation.md#episode-ledger)。教学计划 12 个 case：5 自主成功、2 任务失败、1 timeout、1 人工接管后完成、1 执行中 sim crash、1 reset 失败、1 未启动。

分别计算可判定结果上的条件自主成功率、计划 case 的成功交付比例、结果覆盖率与 policy 启动覆盖率。再规定重试时 `scenario_id` 与 `attempt_id` 怎样记录，解释为什么不能只保留重试成功的结果。

<details>
<summary>完成后展开参考口径（不是 benchmark 标准）</summary>

按正文协议，可判定结果 9 个、policy 启动 10 个；四个比例分别是 `5/9≈55.6%`、`5/12≈41.7%`、`9/12=75%`、`10/12≈83.3%`。3 个缺失任务结果保留原故障或未启动原因；`5/12` 不等于已证明缺失 case 全部策略失败，`5/9` 也不能代表完整评估已完成。接管完成单独记 assisted，不能加到自主成功。

</details>

验收：给出版本、case 集合、成功判据、时间预算和重试规则；指出日志动作评分、录制动作回放、policy 仿真闭环各自不能证明什么。实际仿真与真机均未运行。

<a id="e6"></a>
## E6：相同模型版本，为什么仍要拒绝响应

对应[执行时间表](../../05-embodied-infra/topics/embodied_runtime.md#time-budget)与[reset 反例](../../05-embodied-infra/topics/embodied_runtime.md#stale-response)。先重算 forward 40 ms、候选接纳 70 ms、实际执行 80 ms 三种口径；再画 Q17 发出、Q18 返回、reset、Q17 迟到的顺序。

验收：解释旧响应为何失效、队列接纳/提交/物理执行的区别，以及相对增量动作为什么不能随便删前缀。画出谁负责清理模型记忆、请求、候选队列和 controller 状态；给出最小 artifact 清单。

这是请求与状态的纸面练习，不提供自动控制机器人命令。若以后做软件回归，记录拒绝原因和状态转换；若做闭环实验，结果独立保存，不能用软件断言代替物理效果。

## 结果区（待执行）

| 练习 | 实际输入与版本 | 结果／证据 | 状态 |
|---|---|---|---|
| E1 | 待填写 | 无运行或讨论结果 | NEW |
| E2 | 待填写 | 无运行或讨论结果 | NEW |
| E3 | 待选择授权数据集与环境 | 无性能或正确性结果 | NEW |
| E4 | 两 rank 的教学样本；实际后端待选 | 仅参考推导，未完成读者作答或训练运行 | NEW |
| E5 | 虚构 12-case 账本；实际 taskset 待选 | 仅参考口径，无仿真或读者作答记录 | NEW |
| E6 | 假设时钟与请求序列 | 无软件回归、读者作答或闭环结果 | NEW |

讨论之后把解释回写[模型入门](../../05-embodied-infra/topics/embodied_models_primer.md)或[数据正文](../../05-embodied-infra/topics/embodied_data_io.md)，进度写入[共享学习记录](../../research/learning_log/2026-10.md#embodied-discussion)。

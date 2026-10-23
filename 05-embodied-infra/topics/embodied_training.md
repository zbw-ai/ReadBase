# 具身训练：从正确的 batch 到可恢复的参数更新

[核心入口](embodied_infra.md) · [路线阶段④](../roadmap.md#stage-4) · [模型入门](embodied_models_primer.md) · [数据与 I/O](embodied_data_io.md)

本文解决一个具体问题：拿到 episode 数据和预训练模型以后，怎样证明 trainer 更新了该更新的参数、按预期权重消费了样本，并且能从中断处继续训练？模型结构先读[动作生成路径](embodied_models_primer.md#action-compute)；本文承接[单样本算例](embodied_models_primer.md#sample-to-action)，不再重讲动作含义。

状态：**NEW / 教学与工程设计**。算例是人为设定，不是性能实测；文中的训练、恢复与 GPU 练习均未执行。API 行为核对于 2026-10-23，运行时仍须锁定自己的 PyTorch、模型与数据版本。

<a id="step"></a>
## 1. 先解释一个 trainer step

一次 optimizer step 可以消费多个 microbatch。先画出下面这条链，再考虑加卡：

```text
固定数据 revision + episode/window IDs
  → decode / transform / collate
  → observation、state、action target、各类 mask
  → vision / language / action 模块前向
  → 任务 loss 的分子、有效权重分母与反传缩放
  → backward + 跨 rank 梯度聚合
  → 按约定完成梯度缩放、必要的 unscale / clipping
  → optimizer、scheduler、可选 EMA 更新
  → 记录已提交 step 与实际消费的样本
```

每一步至少留下可核对的证据：样本身份、有效动作元素数、loss 分子／分母、各模块梯度是否存在、参数更新量和 step 是否真正提交。只看 loss 曲线或 GPU 利用率，不能判断上述链路是否正确。

BC 与交互 RL 可以复用训练后端，但训练输入与完成条件不同；完整概念对照在[BC／RL 基础](embodied_models_primer.md#bc-rl-infra)。

| 路径 | trainer 从哪里取数据 | 一次更新前还需确认什么 |
|---|---|---|
| 离线 BC | 固定示范版本中的 observation/action 窗口 | 标签、normalization、mask 与采样权重；不必等待新机器人交互 |
| 带交互收数的 BC | 策略部署后产生的数据，经筛选或人工纠正形成监督样本 | 数据接纳和版本发布；有交互不自动变成 RL |
| 离线／在线 RL | 带奖励及状态转移等算法所需字段的经验 | reward、终止与截断、行为策略版本，以及算法需要的 replay／价值估计或校正状态 |

连续动作 RL 不一定使用 LLM 的 token/logprob 契约，也不一定是 PPO。先固定具体算法，再决定保存 critic、target network、replay 或其他状态；不要给所有具身训练强行添加同一套字段。环境采样与 reset 见[仿真与评估](embodied_sim_evaluation.md)，运行时权重切换见[动作执行](embodied_runtime.md)。

<a id="trainable"></a>
## 2. 冻结的是参数，还是整条梯度路径？

先列出模块清单，不能只写“训练 VLA”：

| 模块或工件 | 配置时回答的问题 | 一步更新验收 |
|---|---|---|
| Vision encoder | 全冻结、部分 block 训练，还是插入 adapter？ | 预期冻结参数不更新；可训练视觉模块的梯度与更新可见 |
| Language/VLM backbone | 全量微调、LoRA 或冻结？ | 记录具体可训练参数名，核 adapter 是否实际进入 forward |
| Action expert/head | 学 token、连续回归、噪声还是速度场？ | 对应目标参与 loss，shape、有效维度及更新量正确 |
| Processor / norm stats / action adapter | 固定预处理，还是训练中的可学习变换？ | 固定工件有版本；可学习参数纳入 optimizer 与 checkpoint |
| EMA（方法采用时） | 用于评估还是发布，在哪个 step 更新？ | online weights 与 EMA 不混淆，保存更新计数 |

冻结通常通过参数的 `requires_grad=False` 表达，但它与 `no_grad()`、`eval()` 是不同开关：

- **全冻结的前置 vision encoder**：输入是普通图像张量，上游没有需要训练的模块；不需要对输入求梯度时，可以不为这段构造反传图，其输出仍可供后续可训练 head 使用。
- **夹在可训练模块之间的冻结 block**：例如 trainable adapter → frozen block → action head。block 权重不求梯度，adapter 仍需要通过它获得输入梯度；把整段包进 `no_grad()` 会切断这条路径。
- **`eval()`**：控制 Dropout、BatchNorm 等模块的行为，不负责关闭 autograd。冻结后是否切换模式，要按配方决定并在验证中保持一致。

这些区别来自 [PyTorch 2.14 Autograd 语义](https://docs.pytorch.org/docs/2.14/notes/autograd.html#locally-disabling-gradient-computation)。冻结中间层仍可能保留反传所需 activation；不能按可训练参数比例缩放整次训练显存。

启动时输出每个模块的总参数／可训练参数、dtype、optimizer group、学习率与 weight decay。optimizer 只接收计划更新的参数；改变冻结策略时，要处理旧梯度和 optimizer 状态，并重新核对分布式包装约束。只把参数从 optimizer 移除，却继续计算并累积其梯度，也不是完整冻结配置。

**最小验收**：用同一小 batch 做一次更新，比较参数前后差值；冻结参数应符合“不更新”的约定，可训练但没变的模块需要区分零梯度、未被使用、学习率为零或 AMP 跳步。不要用“梯度张量存在”替代“目标参数确实更新”。

<a id="loss-normalization"></a>
## 3. Mask 与全局 loss：先定义谁应该获得多少权重

一个 batch 可以同时有多种 mask，不能复用一个布尔张量代表全部语义：

| mask | 解决什么问题 | 不能替代什么 |
|---|---|---|
| Observation / attention mask | 历史帧、相机或文本位置是否可作为输入 | 不自动屏蔽动作监督 |
| Action time mask | horizon 尾部是否超出 episode 的有效动作 | 不说明每个动作维度是否存在 |
| Action dimension mask | 不同 embodiment 的动作维度是否有效 | 不说明该时间步是否真实存在 |
| Loss weight / sample weight | 任务、轨迹、来源或动作维度怎样加权 | 不是“缺失值均填零”的理由 |

先确定目标：按有效动作元素平均、按轨迹平均、按任务均衡，都会产生不同梯度。没有一种归一化适合所有方法；下面**仅演示归一化连续动作的等权 L1、按有效元素平均**，不是通用 BC／flow／RL 公式。

### 两个 rank 的可计算例子

设每个 rank 有 2 个样本，`H=4`、`D=7`，所有有效动作维度等权。rank 0 的两个样本均有 4 步；rank 1 只有 2 步有效，另一个样本为全 padding。为展示边界暂保留空样本，真实采样器通常应提前拒收。

| rank | 有效动作步数 | 有效元素数 `N_r` | 本地有效元素平均 L1 | 误差和 `S_r` |
|---|---:|---:|---:|---:|
| 0 | 8 | 56 | 0.2 | 11.2 |
| 1 | 2 | 14 | 0.6 | 8.4 |

```text
错误地平均两个 local mean： (0.2 + 0.6) / 2 = 0.4
本例所需全局目标：           (11.2 + 8.4) / (56 + 14) = 0.28
```

rank 1 有效元素少，却被“先各自平均、再等权平均”赋予了过大权重。即使每 rank 的 batch shape 一样，padding 比例不同也会改变训练目标。

在 **DDP 默认对梯度取平均、没有自定义通信 hook 或特殊 join 缩放** 的前提下，设当前 DDP process group 的 rank 数为 `W`、组内全局有效数为 `N=Σ_r N_r`，本地反传目标应为：

```text
L_r_for_backward = W × S_r / N
DDP 平均后的梯度 = (1/W) × Σ_r ∇(W × S_r/N)
                 = ∇[(Σ_r S_r)/N]
本例：rank 0 反传标量 0.32，rank 1 为 0.24；对应全局目标 0.28。
```

`S_r` 保留计算图；`N` 是按所有 rank 的 mask 汇总的计数，不求梯度。日志另行归约 detached 的 `S_r` 与 `N_r`，显示 `ΣS/ΣN`；不要把未说明口径的本地反传标量当全局 loss。缩放是由 [DDP 梯度平均规则](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html)推导的教学结果，并非框架自动处理动作 mask。

若某 rank 没有有效元素，但全局 `N>0`，仍需按既定模型路径参与兼容的 backward／collective，以连接计算图的零贡献处理本地监督；单个 rank 擅自跳过一步可能让其他 rank 等待。全局 `N=0` 时，所有 rank 一致跳过更新并记录原因，不能除零；也不能先产生 NaN，再期待乘零 mask 消除它。

### 梯度累积的分母跨整个 optimizer step

设一次更新积累 `A` 个 microbatch，本例目标应使用 `N=Σ_r Σ_a N_ra`；每个 microbatch 的分子 `S_ra` 都按同一个 `N` 缩放。分别计算各 microbatch mean 后再除以 `A`，只有权重条件吻合时才等价。

一种直观教学实现是先拿到这一组 microbatch 的有效计数，再依次反传 `W×S_ra/N`，最后更新一次；已经按总 `N` 缩放时，不要再额外除以 `A`。这要求控制预取内存。另一种方案是累积未归一化梯度，最后按全局计数统一缩放；它要同时正确处理 AMP unscale、clipping 和后端规约语义，不能直接套用成通用代码。

DDP 的 `no_sync()` 只控制梯度同步时机，**不修正 loss 权重**；使用它时，相关 forward 与 backward 都应在上下文内。FSDP 的累积接口、梯度驻留与通信行为需另核固定版本，不能直接抄 DDP 开关。先用单进程合并 batch 作 reference，再比两 rank、不同 padding 与不同累积分组的一步参数更新。这个等价检查先选没有 batch-dependent 行为、随机性受控的小模型：普通 BatchNorm 的局部统计会随分组变化，仅固定 seed 也未必保持逐样本 Dropout mask 相同，不能把这些模型行为差异归因于分母错误。练习接到 [E4](../../practice/experiments/embodied_contract_lab.md#e4)。

<a id="memory"></a>
## 4. 按状态账本选单卡、DDP 或 FSDP

详细通信机制复用 [DP](../../02-training-infra/topics/data_parallelism.md#dp-communication) 与 [FSDP/FSDP2](../../02-training-infra/topics/fsdp.md#fsdp-zero-map)。具身训练增加的是多相机、历史窗口、动作 horizon 和不同冻结范围，选型仍先回答哪类状态超预算。

```text
每 rank 峰值 = 实际驻留的参数 + 梯度 + optimizer 状态
             + 可选 master weights / EMA
             + activation + 临时完整参数 + 通信/解码/框架缓冲
```

不要把所有项都除以卡数。冻结参数仍要参与 forward；FSDP 参数分片不等于 activation 按比例缩小；若 GPU 解码与训练共卡，还要单独计入帧和解码器工作集。

| 测量结果 | 更合理的第一步 | 验收重点 |
|---|---|---|
| 单卡能容纳目标 microbatch | 先建立单卡数值基线 | loss、mask、梯度和一步更新可解释 |
| 单卡状态能放下，需要更多样本吞吐 | 以 DDP 作扩展基线 | 样本不重复／遗漏、全局目标不变、最慢 rank 耗时 |
| 参数／梯度／optimizer 状态主导 OOM | 评估 FSDP/ZeRO 状态分片 | 参数聚合峰值、通信暴露、保存／恢复支持 |
| 多视角或长历史使 activation 主导 | 减 microbatch，测选择性重计算等方案 | 输入覆盖与有效 global batch 是否改变 |
| GPU 等 batch 或少数 rank 拖尾 | 先测数据、decode 与采样分布 | 不用分片掩盖 I/O 空洞 |

冻结范围、并行组和 optimizer 创建顺序共同决定真实账本。[FSDP2 官方教程](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html#model-initialization)要求先应用 `fully_shard` 再创建 optimizer；本次核到的版本支持同组混合冻结与非冻结参数，但不能把这个结论套到所有 FSDP1 配置。

分组从可解释的模块/block 边界开始，记录 all-gather、reduce-scatter、临时完整参数与 compute 的重叠。vision、VLM、action expert 计算比例不同，不能只按参数量平均切组；[LingBot-VLA v1 §4.2](https://arxiv.org/html/2601.18692v1#S4.SS2)提供 action-expert 分组的已收录案例，不是所有 VLA 的默认配置。

第一次测量至少覆盖初始化、forward、backward 和首个 optimizer step；optimizer 状态可能在首次更新才建立。记录实际 dtype 和 optimizer 实现，再算字节数，不套用未经核验的“每参数固定若干字节”。具体实验复用 [E01–E03](../../practice/experiments/a100_fsdp_io_lab.md#e01)，不另建一套训练框架。

<a id="recovery"></a>
## 5. Checkpoint 保存的是一次学习状态

区分三件事：**能加载权重、能按同一语义续训、能逐 bit 重现后续结果**。后两者逐级增加条件，变更 world size、kernel 或数据调度时尤其不能默认一致。通用持久化机制见 [Checkpointing](../../02-training-infra/topics/checkpointing.md)。

| 状态组 | 应保存或固定什么 | 丢失后的典型后果 |
|---|---|---|
| 参数与优化 | model、optimizer、scheduler、step、采用时的 AMP scaler/EMA 及其计数 | 用旧权重开始另一条优化轨迹 |
| 输入语义 | processor、norm stats、action adapter、相机/关节顺序、数据 revision | loss 看似连续，但输入或动作尺度改变 |
| 消费位置与随机性 | sampler/shuffle/mixture 状态、数据游标、augmentation RNG、框架 RNG | 重复、漏读、增强或采样比例改变 |
| 训练结构 | trainable 参数清单、optimizer groups、模型/config、并行布局与依赖 | 错配 optimizer state，或恢复后的参数不再更新 |
| 算法附加状态 | 方法实际使用的 replay、critic/target、记忆或 curriculum 状态 | 经验、目标或跨窗口上下文与权重不一致 |

有限数据集可以用明确样本顺序验收；流式数据还要区分“worker 已预取”“训练已消费”和“step 已提交”。若只能从最近提交的 step 重放，记录允许重放的样本范围；不要声称恢复了无法追踪的 worker 内部位置。

**选择清楚的保存边界。** 教学起点是在 optimizer step、scheduler 和约定的 EMA 更新完成后，保存一致快照。若要在梯度累积中途恢复，还要保存累积梯度、microstep、总计数与对应数据位置；仅保存 `global_step` 不够。异步持久化的提交与完整性规则复用训练基础，不把“后台写入已启动”当作可恢复。

[LeRobot v0.6.1 的既有核验](../../research/tracking/backfill/2026-08.md#lerobot-v061-20260803)是具体反例入口：resume 路径需保留 checkpoint processor stats，不能被当前数据集重新算出的统计覆盖。改统计量属于改变输入语义，应作为明确的新训练配置，而不是悄悄发生在恢复过程。

恢复验收用一条连续训练分支作 reference，另一条在相同边界保存、重启并继续：

1. 先核权重、processor/stats 摘要、可训练参数与 optimizer group 是否一致。
2. 对照下一组样本身份、mask、随机增强和有效元素数，再比 forward/loss。
3. 对照梯度、参数更新、scheduler/EMA 计数；容差与不确定性在运行前声明。
4. 同拓扑恢复解释清楚后，再按后端支持做 reshard 恢复；不预设 bitwise 一致。

复用 [E05 恢复实验](../../practice/experiments/a100_fsdp_io_lab.md#e05)。部署只使用其中部分工件，但机器人记忆、执行队列与过期动作不能当 trainer checkpoint 盲目重放；边界见[运行时恢复](embodied_runtime.md)。

## 6. 配置与指标：每个开关应对应一个假设

| 配置面 | 首轮固定项 | 优化前后的对照 |
|---|---|---|
| 数据与目标 | revision、split、mixture、H、动作维度、mask、归一化口径 | 有效元素／轨迹权重与样本身份 |
| 可训练范围 | vision/VLM/action 的参数清单、LoRA 配置、模式、optimizer groups | 模块梯度和实际更新 |
| 数值 | compute/reduction/optimizer dtype、scaler、clipping 时点 | 合并 batch reference 的更新误差 |
| 并行 | microbatch、累积次数、DP group、shard 分组、recompute | 最慢 rank step、显存峰值、暴露通信 |
| 恢复 | 保存提交点、数据消费位置、stats/EMA 与版本清单 | 重启前后同义输入和同一学习进度 |

同时报告 sample/s、有效 action elements/s、每任务采样比例和 optimizer steps/s。增加 padding 后 tensor 吞吐可能提高，有效监督却没增加；改变 trajectory 权重后 loss 数字也不再可直接横比。性能测量应与[闭环评估](embodied_sim_evaluation.md)共同检查，离线一步等价不证明任务成功率等价。

<a id="troubleshooting"></a>
## 7. 常见坑与排障顺序

| 现象 | 优先取证 | 分辨与修复方向 |
|---|---|---|
| 加卡后 loss 尺度或更新量改变 | 各 rank 的 `S/N`、mask 数、梯度归约与累积配置 | 先排 local mean 的错误等权，再查精度与学习率 |
| 冻结 backbone 后 adapter 不学习 | 可训练参数表、`grad is None`、forward 中的 detach/no-grad | 核梯度是否被冻结中间块截断；不要先加学习率 |
| head 很小仍 OOM | forward/backward/首个 step 峰值，多相机 activation 与解码缓冲 | 按状态类别改配置，不能直接把整模型除以 rank 数 |
| 训练到 episode 尾部或坏帧后挂住 | 各 rank 迭代数、有效数、数据异常与 collective 时间线 | 不允许单 rank 静默丢 batch 或提前退出；统一失败策略 |
| resume 后动作尺度突然偏移 | processor、norm stats、adapter 与数据版本摘要 | 先排统计被覆盖，再排模型更新与 domain shift |
| 权重恢复成功但后续 loss 跑偏 | optimizer/scheduler、RNG、mixture、样本位置和累积边界 | 分清优化状态丢失、样本改变与可接受的数值差异 |

固定一个小样本集合的检查顺序是：**输入身份与语义 → loss 与权重 → 梯度路径 → optimizer 更新 → 分布式聚合 → 恢复 → 性能**。数值正确后，才用阶段耗时判断瓶颈在 loader、H2D、compute 还是 collective；数据专项见 [I/O 流水线](embodied_data_io.md#pipeline)。

<a id="mastery"></a>
## 8. 学完后应能独立完成什么

- 画出一个真实配置的 forward/backward 路径，标明冻结参数与仍需输入梯度的模块。
- 从两个 rank 的有效计数重算全局 loss，解释为什么例子结果是 0.28 而不是 0.4。
- 改变 microbatch／累积分组时，保持同一目标与有效 batch；说明哪里同步、哪里归一化。
- 列出显存账本，解释 DDP 放不下的具体原因，以及 FSDP 没有解决的那部分状态。
- 给出能检测 stats 被覆盖、数据重复和 optimizer 丢失的恢复对照，而非只调用 load。

先完成 [E4 纸面与数值练习](../../practice/experiments/embodied_contract_lab.md#e4)，再按问题选择 A100 课程；本文不新开路线或阅读队列。实际学习结果写回共享学习记录，实验结果出现后才更新验证状态。

## 9. 来源与相邻章节

- [PyTorch 2.14 DDP](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html)：核对默认梯度平均及 `no_sync`；本文的有效元素公式为基于该语义的推导。
- [PyTorch 2.14 Autograd](https://docs.pytorch.org/docs/2.14/notes/autograd.html#locally-disabling-gradient-computation)：核对参数冻结、grad mode 与 eval 的区别；算例不是官方推荐配方。
- [FSDP2 官方教程](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)：本次读取页面标为 2.14，核对分片、混合冻结与 optimizer 初始化；具体参数以执行环境版本为准。
- [LingBot-VLA 原始账本](../../research/tracking/backfill/2026-01.md#embodied-20261023-lingbot-vla)、[LeRobot 恢复账本](../../research/tracking/backfill/2026-08.md#lerobot-v061-20260803)：分别作为训练分组和 processor 恢复的已核案例，不外推作者性能。
- [数据](embodied_data_io.md)、[仿真与评估](embodied_sim_evaluation.md)、[执行](embodied_runtime.md)分别提供训练输入、闭环验收与动作提交约束；[平台章](agentic_for_embodied.md)保留跨资源池设计。

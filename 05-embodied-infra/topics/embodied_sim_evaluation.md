# 具身仿真与评估：从一次 reset 到可信的成功率

[核心入口](embodied_infra.md) · [所属 Part](../README.md) · [评估学习阶段](../roadmap.md#stage-5) · [E5 评估账本练习](../../practice/experiments/embodied_contract_lab.md#e5)

本章回答：怎样确认一个 policy 在给定任务上可靠，以及失败来自模型、环境还是运行系统。学习顺序是固定 taskset → 环境生命周期 → 结果账本 → 吞吐 → 证据边界 → sim-to-real 诊断。

状态：**NEW / 教学设计，未执行仿真、GPU 或真机实验**。下文所有杯子任务、频率、case 数和耗时都是明确的教学假设，不是公开 benchmark 配置或本仓库实测。平台服务与 fleet 拓扑沿用[进阶平台章](agentic_for_embodied.md)。

<a id="env-contract"></a>
## 1. 先固定评估问题，再创建环境

教学任务是“把杯子放入托盘”。输入为两个相机视角、语言任务与本体状态；policy 每次预测 `H=4`、`D=7` 的动作段，执行前 `K=2` 步后重新预测。动作仍须约定坐标、单位、夹爪语义与 controller 接口，`[4,7]` 本身不足以执行。

本例成功判据：杯子位于预先定义的托盘区域内、满足约定姿态阈值、夹爪已经释放，并连续保持 0.5 秒仿真时间。评估器读取哪些状态、容差与持续时间必须事先固定；画面看起来“差不多放好了”不能临时替代判据。

### 1.1 一个 environment 包含哪些东西

| 组成 | 决定什么 | 最容易遗漏的版本 |
|---|---|---|
| 场景与资产 | 杯子几何、托盘、碰撞体、初始姿态 | asset digest、场景参数、碰撞配置 |
| Physics | 接触、关节、质量、摩擦与数值推进 | backend、步长、求解器及相关配置 |
| Sensor / render | 模型何时看到什么图像和状态 | 相机位置、标定、曝光／噪声、采样时刻 |
| Action adapter / controller | 模型向量怎样改变机械系统 | 单位、坐标、控制模式、限幅、控制周期 |
| Task / evaluator | 什么叫成功、失败、超时 | 任务定义、判据代码、容差、时间预算 |

[Isaac Lab 原论文](https://arxiv.org/abs/2511.04831v1)将物理、渲染、执行器、传感器与学习工作流放在一起，适合据此建立组件地图；论文中的框架能力不能证明某套自定义环境已经正确。

### 1.2 固定 taskset，而不只固定一个 seed

本例先建立 12 个 `scenario_id`：训练分布内、未见杯子外观、未见初始位置各 4 个；这些组是教学划分，不代表任意物体替换都构成同等难度的泛化。A/B policy 使用同一 case 集合，并分别保存结果。

最小 evaluation manifest 是下面的**教学契约，不是官方 API**：

```text
suite_version / evaluator_version / scenario_manifest_digest
scenario_id / distribution_group / initial_state / randomization_spec
environment_version / asset_digest / physics_config / sensor_config
controller_version / action_adapter / action_period / policy_schedule
policy_artifact / processor / normalization / camera_order
seed_set / rng_policy / sim_time_budget / wall_time_budget
success_rule / intervention_rule / retry_rule / aggregation_rule
```

同 seed 只是复跑条件之一；资产、求解器、随机数消费顺序和并行执行变化都可能影响轨迹。比较时同时保存初始状态与版本，而不把“设了 seed”当跨实现逐位确定性的保证。

### 1.3 从计划 case 到一条 episode

```text
PENDING → RESETTING → READY → RUNNING → RESULT_COMMITTED
             ↓          ↓        ↓
         reset失败    未启动    任务结束／外部截断／运行故障
```

这里的状态名也是教学约定。`scenario_id` 标识待测条件；`attempt_id` 标识某次尝试；真正开始交互后再绑定 `episode_id`。重试生成新的 attempt，不能覆盖旧记录或选择最好的一次当唯一结果。

Reset 要恢复的不只是杯子位置：

- 环境侧恢复物体、关节、速度、接触相关状态和 task 计数器。
- Policy/runtime 清理上一 episode 的记忆、action chunk、已提交位置与迟到请求。
- Sensor/controller 重建有效观测与控制初态；第一帧不能沿用 reset 前的画面。
- 调度与记录侧确认 case 身份、版本和 readiness，再允许首个动作执行。

这些职责不一定都由 `env.reset()` 实现，应按 owner 检查。随机化参数在何时采样、失败 reset 是否消耗随机数、snapshot 是否包含外部状态，也要写入协议。

<a id="simulation-clock"></a>
## 2. Physics、render、action 与 policy 不是同一个频率

本例用以下频率解释计算边界，实际配置应由任务和实现决定：

| 环节 | 教学假设 | 一次更新的含义 |
|---|---|---|
| Physics | 240 Hz，`dt=1/240 s` | 推进一步动力学；不是一次 VLA 查询 |
| Action / controller 接口 | 20 Hz，每 50 ms 一个目标 | 在两个目标之间推进 12 个 physics substeps |
| 两路相机 | 各 20 Hz | 生成带采样时刻的新观察；不保证与 policy 同时完成 |
| Policy 查询 | 同步示意为 10 Hz | 每次预测 4 步、执行 2 步，执行覆盖 100 ms |

因此一次 `[4,7]` 预测覆盖 200 ms 的动作时间；执行前 2 步后再根据新观察调整。控制器怎样在 50 ms 内跟踪目标属于其接口契约，不能假设所有实现都简单保持动作不变。

`env.step(action)` 推进多少 physics substeps、是否渲染、是否立即 reset，取决于具体环境及 wrapper；不能从函数名推出上述教学时序。Gymnasium 的通用 `step` 返回 observation、reward、`terminated`、`truncated` 与 info，具体时间组织仍由环境实现。[官方 Env API](https://gymnasium.farama.org/api/env/)

再区分两种时间：**仿真时间**描述被模拟系统经过多久；**wall-clock**描述真实计算耗时。离线仿真可以暂停等待 policy，而真实机器人在等待时仍处于物理世界中。

若仿真总是等 GPU 返回才推进，慢 policy 可能得到很好的成功率，却没有验证实时执行。延迟实验要声明：推理期间环境是否继续推进、传感器是否继续采样、controller 如何消费旧动作；这些规则接到[运行时正文](embodied_runtime.md)。

<a id="termination"></a>
## 3. Episode 结束，不等于所有学习目标都应该归零

Gymnasium 区分 MDP 定义内的终止 `terminated` 与外部条件造成的截断 `truncated`。两者都会结束当前采样段，但对依赖 value estimate 的算法，bootstrap 处理不同。[官方 time-limit 教程](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)

| 事件 | 先解释语义 | 需要保存的证据 |
|---|---|---|
| 成功或任务定义内失败 | 到达任务的终止状态 | 成功判据／失败原因与最终 observation |
| 外加采样时间上限 | 任务本可继续，但本次收集停止 | 时间预算类型、截断位置、有效最终状态 |
| 任务自身的有限时域结束 | 截止时间就是任务定义的一部分 | horizon 定义；保留 Markov 性所需的剩余时间信息 |
| 进程崩溃、损坏帧、reset 失败 | 运行故障，未必存在可用 transition | fault stage、缺失字段与受影响 case |

例如本章将 20 秒仿真时间设为**外部评估上限**：未成功即记 timeout；这说明评估没有达成目标，不自动决定训练时应如何设置 value target。Wall-clock 作业预算又是另一回事，不能与仿真任务时限互换。

对持续任务的一步 value target，且截断前最后 observation 有效时，可用 `y = r + γ × (1 − terminated) × V(o_final)` 理解为什么不能用笼统 `done` 去掉所有 bootstrap。这里是教学形式，不是所有 RL 算法的统一实现。

具体算法还须分别规定 return/GAE 的轨迹边界、bootstrap mask 与缺失 transition 的处理；不能让优势递推跨到 reset 后的新 episode，也不能把 auto-reset 返回的下一 episode 首帧误当 `o_final`。检查 wrapper 对最终 observation 的交付方式。

BC 的动作监督没有上述 value bootstrap；episode 边界主要决定动作窗口、padding 与 mask。把成功率记账、数据有效性和 RL target 分开，再与[训练正文](embodied_training.md)核对，而不是用一个 `done` 字段承担三种职责。

<a id="episode-ledger"></a>
## 4. 十二个 case：成功率的分子、分母和缺失结果

下面是一轮**虚构结果**。为便于算账，终态类别互斥；真实事件日志仍可保留多个故障标志，并事先规定主分类优先级。

| Case | 数量 | 记录结果 | 是否有可判定的自主任务结果 |
|---|---|---|---|
| C01–C05 | 5 | 自主成功 | 有，成功 |
| C06–C07 | 2 | 任务失败，例如杯子掉落 | 有，失败 |
| C08 | 1 | 达到外部仿真时间上限 | 有，本预算内未成功 |
| C09 | 1 | 人工接管后完成 | 有，自主评估未成功；另记 assisted completion |
| C10 | 1 | 执行中 simulator crash | 没有完整任务结果；保留故障记录 |
| C11 | 1 | Reset 失败，未进入 policy loop | 没有任务结果；保留 reset 失败 |
| C12 | 1 | 到作业截止仍未启动 | 没有任务结果；保留未启动原因 |

共计划 12 个 case，10 个启动了 policy，9 个得到本协议可判定结果；剩余 3 个的任务结局缺失。假设 C10/C11 已经初步归为基础设施失败；若故障由 policy 非法动作触发，应另查输入与责任，不能为了提高分数全部标成“无效环境”。

### 4.1 同时给出条件表现与整套流程的产出

| 本章教学指标 | 算法 | 能回答什么 |
|---|---|---|
| **可判定结果上的条件自主成功率** | `5 / 9 ≈ 55.6%` | 已取得可判定结果的条件子集中，policy 多常自主完成 |
| **固定计划上的整套评估可靠性成功率** | `5 / 12 ≈ 41.7%` | 这批计划 case 有多少被系统交付为可确认的自主成功 |
| 可判定结果覆盖率 | `9 / 12 = 75%` | 计划覆盖缺了多少；缺失集中在哪类条件 |
| Policy 启动覆盖率 | `10 / 12 ≈ 83.3%` | 有多少 case 真正进入交互阶段 |

这些名称和分母是本章的教学协议，**不是统一 benchmark 标准**。第二项把未知结局保留为“尚未交付成功”，不是声称缺失 case 的 policy 一定失败；第一项则可能因非随机缺失而偏乐观，必须连覆盖率与分组结果一起看。

如果既定 benchmark 要求完整 12 个结果，当前应报告“未完成”，而不是把 `5/9` 与另一个模型的完整结果直接排名。任务失败、timeout、人工介入不能因为不好看就移出有效分母；不能把 C09 改成自主成功。

### 4.2 重试、缺失与聚合

- 重试规则先定：哪些 fault 可重试、次数上限、相同初态如何恢复，以及最后怎样汇总 attempts。
- 重跑 C10–C12 时保留原记录；不能给失败 policy 反复抽新 seed，直到成功才提交。
- 分开报告每个分布组、每个任务的分子和分母；task 宏平均与 episode 微平均权重不同，应先选再比较。
- Case 少时，单次比例不代表稳定差异；后续按协议增加重复并报告不确定性，不只展示最佳 seed。
- 自动成功判据也要验收：保留边界失败和人工抽查，检查 reward、评估器与视频解释是否一致。

最小结果表应包含 `scenario_id、attempt_id、episode_id、版本、起止时间、outcome、termination_reason、intervention、fault、日志索引`。Missing 是需要解释的状态，不能填 success，也不能从导出的 CSV 静默消失。

<a id="throughput"></a>
## 5. Rollout 吞吐：reset 和失败也占真实时间

教学假设：4 个环境并行，每个生成 200 个有效 action transitions；一轮计算／交互用 10 秒 wall-clock，reset 用 2 秒，调度间隙用 1 秒，三个阶段不重叠。

```text
只算活跃阶段：     4 × 200 / 10       = 80 transitions/s
完整一轮：         4 × 200 / (10+2+1) ≈ 61.5 transitions/s
reset 从 2s 降到 1s：800 / 12         ≈ 66.7 transitions/s
```

最后一项只提高约 8.3%，不是“reset 快一倍，系统快一倍”。若其中一个环境的 200 条 transition 因观测损坏不可用，本轮有效吞吐变成 `600/13≈46.2`；正常任务失败轨迹不应仅因没有成功就被排除，它仍可能提供有效评估或学习证据。

实际异步系统不必服从这个整轮模型，应直接用总有效 transition 数除以测量窗口 wall-clock，再分解等待。记录 reset p50/p95、各环境完成时间、render 等待、policy 等待、fault 与有效样本率。

Physics steps/s、相机 frames/s、action transitions/s、完整 episodes/s 是不同指标。若工具已经报整个 GPU 的聚合 steps/s，不要再次乘环境数量；相机数增加也不能直接按相同比例推导闭环吞吐。

<a id="evidence-boundaries"></a>
## 6. 看到什么证据，才能作什么结论

| 评估方式 | 动作与下一观察来自哪里 | 能证明的范围／主要缺口 |
|---|---|---|
| 日志上的动作预测评分 | 新 policy 预测动作，但下一观察仍取日志 | 固定数据上的拟合；没有检验自己失误后的观察与恢复 |
| 录制动作在环境中回放 | 旧轨迹提供动作，环境产生响应 | 环境／adapter／动作重放的一致性；不是新 policy 在线决策 |
| 仿真 policy 闭环 | Policy 根据它造成的新观察继续选动作 | 指定 simulator、分布与时序协议下的任务表现 |
| World-model 预测或闭环 | 分清录制动作条件预测，还是 policy 与 learned backend 交互 | 前者检验预测；后者仍受 learned dynamics 误差与可利用漏洞约束 |
| 视频观感评分 | 未必有明确动作输入或可执行控制接口 | 画面质量与部分一致性；不能单独证明动作遵循和任务能力 |
| 真机闭环 | 实际执行动作，再由传感器反馈 | 实际测试分布中的表现；有限测试不代表所有场景 |

[MiraBench](https://arxiv.org/html/2605.29360v1)提供了有用的反例问题：动作本应失败时，world model 会不会仍生成成功视频？阅读时分开物理一致性、动作遵循与失败保留，不把视觉逼真归并为一个“可信环境”分数；评委和任务覆盖仍有边界。

[OSCAR](https://arxiv.org/html/2606.04463v1)将骨架渲染用作动作条件，其策略评估使用录制动作重放，并有标定筛选与视频评委。它支持研究评估代理，却不等于 policy 每一步都基于生成观察重新行动；相关性不能自动升级为真机闭环替代能力。

落到杯子例子：给成功／提前松爪两组动作，检查预测是否保留相应后果；再换成 policy-in-simulation，让掉杯之后的新观察真正进入下一次决策。两项实验回答的问题不同，结果要分表保存。

<a id="sim-to-real"></a>
## 7. Sim-to-real：用受控对照缩小差异来源

先固定基线、taskset 与成功判据，再改变一个可解释的因素；确认单因素行为后才组合扰动。这里是诊断顺序，不是保证 sim-to-real 成功的配方。

| 层次 | 固定／核对 | 一个受控对照 | 不应直接归因于 |
|---|---|---|---|
| 接口语义 | 单位、坐标、旋转、夹爪、关节顺序与 normalization | 同一目标在 sim 和真机 adapter 的解释 | 模型泛化不足 |
| 观测条件 | 相机位姿、曝光、分辨率、遮挡、state 对齐 | 固定任务，仅改变相机或视觉条件 | 所有视觉偏差都靠数据增强解决 |
| 时间条件 | 采样、推理、排队、动作提交与 controller 周期 | 固定场景，逐档注入延迟／抖动 | 更高离线 FPS 必然更可靠 |
| 动力学／接触 | 质量、摩擦、执行器响应、限幅与接触几何 | 固定 policy，改变一组已说明范围的参数 | 提高渲染质量即可补齐物理差异 |
| 任务分布 | 物体、位置、指令、任务组合与本体 | 分组 held-out 集，不混入训练调参 | 一个平均成功率代表所有泛化 |

Domain randomization 需要记录变量、范围、采样相关性和依据；随机得更宽不自动更真实。真实回归集应按物体／采集场景等隔离，避免相邻窗口或同一事件泄漏到训练与评估两侧。

逐项记录 sim 与 real 的同协议结果，并核人工介入、reset 难度和时间预算；否则所谓 gap 可能来自评估改变。没有可用真机证据时，只写“仿真扰动鲁棒性”，不改写成“已经实现 sim-to-real”。

<a id="troubleshooting"></a>
## 8. 配置与排障：先找失效边界，再调模型

| 现象 | 第一检查 | 下一步证据 |
|---|---|---|
| 相同 checkpoint 换环境就失败 | action/observation 契约、controller 与版本 | 固定单 case，保存进入模型与进入控制器的数值 |
| Success 突然提高，任务却没有改善 | 分母、timeout、接管、missing、判据版本 | 逐 case 对账，比较原始事件与聚合结果 |
| 仿真很快，完整评估很慢 | reset、render、policy 等待与调度间隙 | 按 wall-clock 分阶段计时，不用 physics FPS 代替 |
| 只在 reset 后第一步出错 | 旧帧、旧 chunk、记忆或 controller 状态残留 | 核 episode generation 与首帧／首动作身份 |
| 固定 seed 仍无法复跑 | 初态、随机化、资产、后端和 wrapper 版本 | 对照第一个分歧时刻，而非只比较最终成功率 |
| World model 看着合理却高估成功 | 动作条件是否生效、失败是否被抹平 | 成功/失败动作对照、评委抽查与真实失败案例 |

先处理确定的接口或账本错误，再讨论模型容量与训练数据。成功判据改动应生成新的报告版本；性能优化也要保留同义输入、相同 case 和终止协议。

<a id="mastery"></a>
## 9. 自测：交付一份能被另一个人复核的报告

1. 画清一个 case 的 reset、首帧、首动作、终止与结果提交；指出 policy、环境、controller 各清理什么。
2. 用本章频率算出一次 action 对应多少 physics substeps、H/K 覆盖多久；解释 wall-clock 延迟为何另算。
3. 为真实终止、外部截断和损坏 transition 分别写数据处理规则；说明所选 RL 算法怎样 bootstrap。
4. 重算 12-case 账本的四个比例；指出人工接管、sim crash 和未启动怎样保留，何时应报告未完成。
5. 给出 A/B 固定版本与分组清单，说明哪个对照检验动作语义、哪个检验时间、哪个检验动力学。
6. 用一句话区分日志评分、动作回放、world-model 闭环和真机闭环的证据范围。

将结果填入 [E5](../../practice/experiments/embodied_contract_lab.md#e5)，未执行就保留预期与待验证项。训练侧读[训练机制](embodied_training.md)，执行时限读[运行时](embodied_runtime.md)，数据身份与时序读[数据正文](embodied_data_io.md)。

## 10. Primary 来源与证据范围

- [Gymnasium Env API](https://gymnasium.farama.org/api/env/)与[Handling Time Limits](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)：2026-10-23 核对接口和终止语义；实际运行仍应固定安装版本和 wrapper。
- [Isaac Lab 原论文](https://arxiv.org/abs/2511.04831v1)：仿真组件与学习工作流；[既有核验记录](../../research/tracking/backfill/2025-11.md#embodied-20261013-2511-04831)。
- [MiraBench 原文](https://arxiv.org/html/2605.29360v1)：动作条件可靠性与失败保留；[既有核验记录](../../research/tracking/backfill/2026-05.md#embodied-followup-2605-29360)。
- [OSCAR 原文](https://arxiv.org/html/2606.04463v1)：骨架动作条件与录制动作评估边界；[既有核验记录](../../research/tracking/backfill/2026-06.md#embodied-followup-2606-04463)。

本章整理机制与教学协议，没有新增热点扫描、推进游标或声称实验完成。

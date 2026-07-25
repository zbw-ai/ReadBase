# MOPD：从 On-Policy Distillation 到多领域能力集成

> 状态：READING  
> 定位：原理型专题页。第一版先回答 Distillation → OPD → MOPD 为什么逐步产生；框架代码、性能实验和 AReaL 原型留到后续验证。

## 先记住一个问题

每个领域都能独立训练出更强的 RL Teacher，为什么一个统一模型仍然难以同时继承所有能力？

困难不在于“有没有专家”，而在于**如何把多个专家的能力稳定地写回同一个 Student**：

- 把所有领域混在一次 RL 里，reward、数据分布和超参数会相互干扰。
- 按领域串行训练，后训练的能力可能覆盖前面的能力。
- 收集 Teacher 答案再做 SFT，Student 学到的是 Teacher 的状态分布，不是自己推理时真正会访问的状态。
- 合并 Teacher 参数，冲突发生在 weight space，融合结果对系数和模型距离很敏感。

MOPD（Multi-Teacher On-Policy Distillation）的核心判断是：

> 领域能力可以分别生产，但应在 Student 自己的 rollout 分布上，通过 prompt routing 和逐 token Teacher 信号完成统一集成。

## 从传统 Distillation 到 OPD

### 1. Fixed-corpus logit KD

经典 Knowledge Distillation 在一批固定输入或固定序列上，让 Student 拟合 Teacher 的概率分布。

```text
fixed corpus
    ↓
Teacher logits
    ↓
Student matching
```

它比只学习 hard label 信息更密，但训练时看到的 prefix 仍由固定语料决定。Student 真正自回归推理后，一旦早期 token 与语料偏离，后续会进入训练时没有覆盖的状态。

### 2. Teacher-generated trajectory distillation

更直接的做法是让 Teacher 先生成答案，再把这些 trajectory 当作 SFT 数据训练 Student。

```text
Teacher rollout → offline trajectories → Student SFT
```

这能转移 Teacher 的解题路径，但仍是 off-policy imitation：

- 训练 prefix 来自 Teacher。
- 推理 prefix 来自 Student 自己。
- 两种 prefix 对应的 state distribution 不一致。

因此，不能笼统地说“静态 trajectory 导致 exposure bias”。真正的问题是：

> Student 训练时所条件化的 prefix，与推理时由 Student 自己产生的 prefix 不同。

### 3. Student-rollout On-Policy Distillation

OPD 把数据生产权交回 Student：

```text
Student samples
    ↓
Teacher scores the same sampled trajectory
    ↓
Student updates on its own state distribution
```

Teacher 不重新生成另一条“标准答案”，而是在 Student 已经产生的 token prefix 上做 prefill/scoring。这样训练和推理访问的是同一类 Student 状态，同时 Teacher 又能为每个训练 token 提供比 trajectory-level reward 更密的监督。

## OPD 的最小数据流

以 prompt `x` 和 Student rollout `y = (y1, ..., yT)` 为例：

| 组件 | 输入 | 输出 | 关键约束 |
|---|---|---|---|
| Student rollout | prompt `x`、当前 rollout policy | sampled tokens `y`、Student logprobs、action mask、policy version | `y` 必须由 Student 产生 |
| Teacher scoring | 同一个 `x` 和同一条 `y` | sampled-token logprobs，或 Top-k logits | Teacher 做 prefill/scoring，不重新 rollout |
| Trainer | `y`、Student/Teacher logprobs、mask、版本元数据 | token-level advantage 与 Student gradient | 只训练有效 assistant/action tokens |

 sampled-token policy-gradient 形式可以写成：

```text
A_t = stop_gradient(
  log π_teacher(y_t | x, y_<t)
  - log π_student(y_t | x, y_<t)
)
```

其中：

- `y_t` 是 Student 实际采样的 token。
- `A_t > 0` 表示 Teacher 比 Student 更认可这个 sampled token。
- 该差值可以作为 per-token advantage 进入 policy-gradient loss。
- 从目标方向看，它推动 Student 在自己的 rollout 分布上逼近 Teacher，对应 reverse-KL minimization。

这一页不展开完整梯度推导。工程上更重要的是：**Teacher payload 可以只传 sampled token 的 logprob，而不必传整个 vocabulary distribution。**

## OPD 为什么还不够

单 Teacher OPD 只回答“如何把一个 Teacher 的分布稳定蒸馏给 Student”，没有回答：

- Math、Code、SWE、Instruction Following 使用不同 RL recipe 时，谁来统一调度？
- 一个 batch 内的不同 prompt 应该找哪个 Teacher？
- 多个 Teacher 是做 logits ensemble，还是分别负责自己的领域？
- 如何避免多领域 joint RL 的 seesaw effect？

MOPD 在 OPD 上增加的关键能力不是更复杂的 loss，而是**多领域能力生产与统一能力集成的解耦**。

## MOPD 的算法核心

```text
multi-domain prompts
        ↓
Student rollout
        ↓
prompt/domain metadata routing
        ↓
selected frozen Teacher scoring
        ↓
token-level distillation update
        ↓
one unified Student
```

需要明确四个边界：

1. 每个领域有一个独立训练得到的 domain Teacher。
2. 一个样本根据 prompt/domain metadata 路由到一个 Teacher。
3. routing 不是训练出来的 MoE router。
4. 不对多个 Teacher 的 logits 做 ensemble averaging。

因此，MOPD 的“Multi-Teacher”不是让所有 Teacher 同时评价每个样本，而是让每个 Teacher 在自己负责的样本分布上提供监督。

## 算法核心、论文 Recipe 与框架实现

这三层经常被混写，需要分开理解。

### 算法核心

```text
Student rollout
→ selected Teacher scoring
→ distillation update
```

只要满足 Student 自采样、正确路由和 Teacher 对同轨迹打分，就抓住了 MOPD 的算法本体。

### 论文 Recipe

[MOPD 独立论文](https://arxiv.org/abs/2606.30406)采用三阶段流程：

```text
General SFT
    ↓
same-origin domain RL Teachers
    ↓
MOPD integration
```

- General SFT checkpoint 同时初始化 Student 和各领域 Teacher。
- 各领域从共同起点独立做 RL，可以采用不同 reward、环境、算法和超参数。
- 冻结领域 Teacher，再把能力蒸馏回从同一 SFT checkpoint 初始化的统一 Student。

这是论文验证过的稳定 recipe，但“必须先按这三阶段训练”不是 MOPD 算法定义本身。

### 框架实现

Teacher server、独立 GPU pool、async overlap、Ray actor、ICE-POP 等属于系统实现选择。它们决定吞吐、资源成本和异步正确性，但不能反过来定义 MOPD。

## 为什么 MOPD 比其他融合方式更合理

| 方案 | 监督密度 | Student 训练分布 | 领域训练耦合 | 融合空间 | 主要风险 |
|---|---|---|---|---|---|
| Mix-RL | 多为 trajectory-level reward | on-policy | 强耦合，所有领域同一次 RL | policy update | reward/梯度干扰、seesaw |
| Cascade RL | 多为 trajectory-level reward | on-policy | 串行耦合 | policy update | 遗忘、长链路稳定性、重跑成本 |
| Off-Policy Finetune | token-level hard target | Teacher trajectory | Teacher 可并行训练 | static dataset | exposure bias、能力转移不均匀 |
| Param-Merge | 无额外数据监督 | 不适用 | Teacher 可并行训练 | weight space | 参数冲突、系数敏感、结果不稳定 |
| MOPD | dense token-level signal | Student rollout | Teacher 独立，最终统一集成 | policy space | routing、Teacher 距离、资源与异步一致性 |

MOPD 的工程优势来自两个组合：

- **dense supervision**：Teacher 在每个有效训练 token 上提供信号，不只在 trajectory 结束后返回一个 scalar reward。
- **模块化领域训练**：Math、Code、SWE 等团队可以独立选择 RL recipe，并行产出 Teacher，再通过统一 MOPD stage 集成。

前者提高每条 trajectory 的学习信号密度，后者减少跨团队和跨领域的串行依赖。

## Same-origin Teacher 为什么重要

论文中，Student 和各领域 Teacher 共享 General SFT 起点。领域 Teacher 只是在这个起点上继续做专门 RL，因此初始 Teacher/Student policy gap 较小。

论文直接观察到：

- same-origin Teacher 对应较低的初始 KL 和更平滑的优化过程；
- 换成能力更强但分布差异更大的外部 Teacher，训练反而明显不稳定；
- policy-gradient 和 Top-k 两种实现都可能在大 policy gap 下退化。

**工程推断：**较小的初始 KL 可能让 sampled-token advantage 的极值和梯度方差更可控。但这是对观测结果的机制解释，不应写成论文已经严格证明的定理。

这意味着“找更强 Teacher”不是无条件收益。生产中至少要先测：

- 初始 per-token KL、advantage 分位数和 entropy 变化；
- Teacher 与 Student 的 tokenizer、chat template、control token 是否一致；
- 每个 domain 的路由准确率；
- 外部 Teacher 是否持续给 Student 低概率区域施加惩罚信号。

## Top-k：实现变体，不是 MOPD 定义

### Sampled-token policy-gradient

- 每个位置只使用 Student 实际采样 token 的 Teacher logprob。
- 网络 payload 小，容易接入 PPO/GRPO dataflow。
- 单 token estimator 的信息量较少，方差可能更高。

### Top-k distillation

- Teacher 返回每个位置的 Top-k token 分布。
- 利用更多 Teacher 概率信息，但通信和存储 payload 更大。
- 独立论文的 Top-k objective 加入了 bias-correction term，避免 naive Top-k truncation 改变目标最优点。

论文把 Top-k 作为 lower-variance alternative，但在 same-origin Teacher 实验中，两种形式表现接近，Top-k 没显示出额外稳定性收益。不要把“理论上利用更多分布信息”直接等价为“生产上一定更快或更稳”。

## 与 RL Infra 的关系

MOPD 在 Student rollout 和 Trainer 之间插入了新的 Teacher scoring stage：

```text
rollout → route → teacher prefill → trajectory store → train
```

### Teacher Prefill Service

Teacher 接收已经完成的 Student trajectory，只需要做 prefill 并返回 token-level logprobs 或 Top-k logits，不需要 autoregressive decode。独立论文把 Teacher 部署为外部 prefill service，并让已完成序列的 Teacher scoring 与其他序列的 Student sampling 异步重叠。

这是性能优化，不是算法必要条件。若 Teacher prefill 被串行塞进 Trainer critical path，MOPD 仍然成立，但 step latency 会明显增加。

### Teacher Resource Pool

Teacher 数量增加时，显存和 GPU 成本可能近似线性增长。工程上需要考虑：

- 相同 checkpoint 的 Teacher alias 能否复用一个 worker group；
- Teacher 是否常驻、按需加载或分时复用；
- Teacher prefill arrival rate 与 service throughput 是否匹配；
- Student rollout 慢时，Teacher GPU 是否长期空闲；
- Top-k payload 是否造成 trajectory store 和网络压力。

### Routing 与 Trajectory Metadata

每条 trajectory 至少应携带：

- `domain_id` / `teacher_id`
- rollout `policy_version`
- prompt、token ids、Student logprobs
- assistant/action mask
- Teacher scoring version 和完成状态

错误路由会把“专家监督”变成系统性错误标签。tool/environment token 若未正确 mask，也可能把不可学习的外部行为写进 loss。

### On-policy 不等于 Zero Staleness

算法层的 on-policy 表示 trajectory 由 Student policy 产生，而不是来自固定 Teacher 数据集。但在异步系统中：

- rollout policy 生成 trajectory 后，Trainer 可能已更新到新版本；
- Teacher scoring、排队和训练消费会引入时间差；
- 因而仍需记录 policy version，并设置 staleness bound、importance correction 或丢弃策略。

[NeMo RL 的 MOPD 文档](https://docs.nvidia.com/nemo/rl/nightly/about/algorithms/mopd.html)把 MOPD 接入 async GRPO，用独立 non-colocated Teacher group，并通过 ICE-POP gate 修正异步 off-policy drift。这是框架层的正确性处理，不是 MOPD 原论文公式的组成部分。

### 当前框架落地

- [verl PR #6051](https://github.com/verl-project/verl/pull/6051)于 2026-04-20 合并 Multi-Teacher OPD，加入按样本来源选择 Teacher 的 routing、Teacher 管理和示例 recipe。
- NeMo RL 提供 async GRPO 上的 MOPD 配置、基于 agent name 的 Teacher routing、独立 Teacher worker group 和 correctness smoke test。
- 对 AReaL/slime/ROLL，当前更适合先核对其 OPD loss、Teacher scoring 和 trajectory metadata 能力，再判断补齐 multi-teacher routing 的改造量；不能只因支持单 Teacher distillation 就宣称已经完整支持 MOPD。

## 生产环境边界

### Teacher 冲突

即使一个样本只路由到一个 Teacher，不同 Teacher 对共享行为、格式和 control token 的偏好仍可能冲突。最终 Student 是共享参数，领域梯度仍会在参数空间相遇。

### Routing 错误

数据集标签或 agent metadata 错误会让 prompt 被错误 Teacher 评分。必须监控每个 Teacher 的流量占比、unknown/fallback 比例和按 domain 的 KL 分布。

### Control Token 放大

Teacher 对格式 token、tool schema 或拒答模板的强偏好，可能通过 dense token signal 被快速放大。需要按 token 类型观察 advantage、loss 和 entropy，而不只看整体 reward。

### Teacher 服务成为瓶颈

Teacher prefill 若跟不上 rollout 完成速率，会形成新的 queue backlog。异步 overlap 只能隐藏“足够快”的 Teacher 成本，不能消灭容量不足。

### 可恢复性

恢复训练时，不能只恢复 Student checkpoint。还要校验 Teacher 版本、routing config、trajectory queue offset、policy-version window 和未完成 scoring 请求，否则恢复后的数据语义可能改变。

## Multi-round Evolution

2026-06-29 的独立论文进一步研究了多轮 Student–Teacher 演进：

```text
Round-1 MOPD Student
        ↓
作为新起点重新训练 domain Teachers
        ↓
Round-2 MOPD integration
```

它说明 MOPD 不一定只是一次性能力融合，还可以形成“统一 Student → 新一轮领域能力生产 → 再集成”的循环。

这是独立论文的扩展分析，不能倒推为 verl 在 2026-04-20 合并的 PR 已经完整实现 multi-round orchestration、Top-k bias correction 或论文中的稳定性实验。

## 时间线与一手来源

| 时间 | 事件 | 工程意义 |
|---|---|---|
| 2026-01-06 | [MiMo-V2-Flash Technical Report](https://arxiv.org/abs/2601.02780)首次公开 MOPD | MOPD 先作为工业级模型 post-training recipe 出现 |
| 2026-04-20 | [verl Multi-Teacher OPD PR #6051](https://github.com/verl-project/verl/pull/6051)合并 | 开源 RL 框架开始补齐 multi-teacher routing 与执行路径 |
| 2026-06-29 | [MOPD 独立论文](https://arxiv.org/abs/2606.30406)发布 | 补充受控 baseline、Top-k、same-origin 稳定性和 multi-round 分析 |

补充实现资料：

- [NeMo RL: Multi-Teacher On-Policy Distillation](https://docs.nvidia.com/nemo/rl/nightly/about/algorithms/mopd.html)
- [Agentic RL Infrastructure](agentic_rl.md)

## 当前工程判断

MOPD 的价值不只是“多 Teacher 蒸馏”。它把能力生产和能力集成拆成两个阶段：各领域团队独立做最适合自己的 RL，再由统一 Student 在自身 rollout 分布上吸收 dense token-level supervision。它避免了 Teacher trajectory SFT 的状态分布错位，也减少了 joint RL 和 cascade RL 的跨领域耦合。

但 MOPD 没有消除所有冲突，只是把主要矛盾从“共同训练一条 RL 链路”转移到 Teacher 距离、routing correctness、Teacher prefill 容量、异步 freshness 和共享 Student 参数更新上。真正落地时，算法收益必须和新增 Teacher 资源及系统复杂度一起评估。

## 下一步

1. 对照 verl 与 NeMo RL 的数据结构、Teacher 生命周期和 routing 实现。
2. 设计 AReaL 上的 MOPD 最小原型，不先引入 Top-k。
3. 测量 Teacher prefill throughput、队列积压和与 rollout 的 overlap 比例。
4. 建立 routing correctness、policy freshness、advantage 分位数和 control-token behavior 监控。

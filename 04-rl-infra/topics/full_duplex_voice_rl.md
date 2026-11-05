# 全双工多模态语音 Agent：从训练数据到 RL 基建

**2026-11-05｜公开来源定向研究；工程方案未实验。** 面向连续音视频输入、语音输出、多轮任务与工具调用。本文不推定任何内部模型的结构、数据、效果或资源配置；物理动作 policy 需另外定义，不能由语音模型直接外推。

[RL Part](../README.md) · [Omni 模型基础](../../05-embodied-infra/topics/omni_models_infra.md) · [项目交付路线](../../05-embodied-infra/roadmaps/omni.md#duplex-project) · [实验册](../../practice/experiments/omni_sft_rl_lab.md#duplex-o4) · [来源核验](../../research/tracking/audits/2026-11-05-duplex/sources.json)

## 问题：训练的是回答，也是在共享时间线上做决定

目标可以用一句话说明：**听懂用户与场景，在合适的时候说完整的话，必要时停下，经过多轮交互把事情办成。**

| 能力 | 具体例子 | 需要区分的失败 |
|---|---|---|
| 内容与感知 | 看见用户指的物体，听懂语义、情绪与环境声音 | 转写对了，但指代、语气或画面理解错了 |
| 发言时机 | 用户在思考时等一下，说完后及时接话 | 抢话、长时间沉默、把“嗯”当作夺话 |
| 被打断与接续 | 用户纠正地点后，停止旧回复并更新计划 | 音频停了，旧工具请求仍执行；下一轮引用用户没听到的内容 |
| 多轮任务 | 澄清条件、调用工具、处理失败并确认结果 | 单轮回答很好，但状态、承诺与实际工具结果不一致 |
| 输出体验 | 声音自然，内容完整，播放连续 | 低延迟来自提前结束；音质好却答非所问 |

**流式输出、支持打断、模型原生全双工是三个不同维度。** 外部 VAD/调度器可以实现系统层的边听边说；模型层全双工要求输入在生成期间持续影响决策。两条路线都能作为工程候选，应以同一业务任务比较，不能仅凭架构名称判断优劣。

## 公开前沿：按问题选参照，不做统一 SOTA 排名

以下核到原始论文或官方文档。不同语言、任务、音轨、硬件和评测方式不能横比总分；作者的效果结论没有在本库复现。日期、完整作者、版本与摘录见[证据表](../../research/tracking/audits/2026-11-05-duplex/sources.json)。这不是全行业穷尽扫描，也不推进 frontier cursor。

| 来源与核验版本 | 已披露的训练／机制 | 最值得借鉴什么 | 不能据此声称什么 |
|---|---|---|---|
| [Moshi](https://arxiv.org/html/2410.00037v2)，v1 2024-09-17，v2 10-02 | 文本底座→单流语音预训练→模拟多流→真实双人对话→合成助手指令；用户／助手音频分流，Inner Monologue 对齐文本与语音 | 全双工能力先依赖表示与对话 SFT；模拟分流缺少真实 overlap，需要真实对话补齐 | 该基础流程不是在线 RL recipe，也不证明视觉／工具能力 |
| [PersonaPlex](https://arxiv.org/html/2602.06053v1)，2026-01-14 | Moshi 初始化；文本角色＋音频音色的 hybrid prompt；合成角色／服务对话训练，发布模型另加入 Fisher 自然对话 | 真实对话学节奏，定向合成数据补角色与任务，两者分工 | 原文结尾把 alignment 与外部工具列为后续方向，不能将其写成完整 Agent RL |
| [MiniCPM-o 4.5 官方说明](https://github.com/OpenBMB/MiniCPM-o/blob/main/docs/minicpm_o4dot5_en.md)，动态页，核验 2026-11-05 | 音视频与输出在统一时间线上组织；TDM、在线 encoder/decoder、主动发言决策；提供 duplex 使用入口 | 最贴近“持续看、听、说”的可运行接口候选 | 本次文档未建立完整 RL loss／reward／训练数据配方，也未验证实际兼容性 |
| [Qwen3.5-Omni](https://arxiv.org/html/2604.15804v2)，v1 2026-04-17，v2 04-21 | Thinker specialist distillation、paired audio/text OPD、interaction-aligned RL；Talker 偏好 DPO 后接规则奖励与 GSPO | 理解、声音、多轮行为可分阶段对齐；具体边界见[既有笔记](../../research/tech_reports/qwen35_omni.md) | 流式 Thinker/Talker 与 ARIA 披露，不能单独证明原生全双工训练或完整开放 recipe |
| [Step-Audio 2](https://arxiv.org/html/2507.16632v3)，v1 2025-07-22，v3 08-27 | 模态对齐→混合预训练→多任务 SFT；先两阶段 PPO 优化推理长度与质量，再以 GRPO 提升音频感知；包含工具与语音合成数据 | 可把语音理解／推理和交互时机分成不同训练问题，工具数据需有调用与不调用的对照 | 音频 RL 或工具调用能力不等于训练过全双工让话 |
| [Multi-Faceted Interactivity Alignment](https://arxiv.org/html/2606.11167v1)，2026-06-09 | Fisher／Seamless 独立双音轨抽取短段；pause、turn、backchannel、interrupt 四类 reward；GRPO，加语义 judge，历史 context 不进 loss；policy objective 用 text stream | 四类行为一起优化，避免“快接话”和“不抢话”互相损害；短窗口训练仍要多轮评测 | 论文的 text-action 优化依据来自对应模型，不能搬到任意 codec／控制头 |
| [DuplexPO](https://arxiv.org/html/2607.07148v1)，2026-07-08 | 长对话抽 dynamics-critical windows；历史 teacher forcing，窗口内采样与更新；FCDR 分解接话、附和、让话与参与模式；GRPO-style；末阶段仅训练 speech generation 模块 | 把“何时说”作为独立优化目标，集中算力到关键事件，并检查推理／指令能力 | 局部窗口不能替代完整闭环任务训练；其 KL 指向文中的 rollout behavior policy，不能一律改称固定 SFT reference |
| [SteerDuplex](https://arxiv.org/html/2609.12623v1)，2026-09-11 | 自然＋合成指令 SFT→两阶段交互 RL；分别标准化 reward；增加回答连续性和用户附和后的继续发言；text-stream policy loss，含承载时序的 sampled padding | 直接暴露“为了让话而不回答”的 reward hacking；把完整性纳入验收 | 各阶段预算／数据也变化，不能把所有收益归因给单项 reward；评测数据重叠项只作诊断 |
| [OpenAI Realtime VAD](https://developers.openai.com/api/docs/guides/realtime-vad)／[会话](https://developers.openai.com/api/docs/guides/realtime-conversations)，动态文档，核验 2026-11-05 | VAD／semantic VAD 控制，取消与未播音频截断；WebSocket 客户端负责记录已播放边界 | 生产协议明确区分“模型生成了”和“用户听到了” | API 行为文档不公开底座训练 recipe，不能据此推断使用 PPO／GRPO 或训练数据 |

**来源顺序建议**：先读四类交互对齐与 DuplexPO 的数据、mask 和 reward，再看 SteerDuplex 的失败分析；架构看 Moshi，角色数据看 PersonaPlex，音视频执行接口看 MiniCPM-o，Omni 模块后训练复用 Qwen。没有必要先逐篇完整复现。

## 机制：推荐怎样训练

以下是综合公开方法提出的**候选工程流程**，不是任何单一团队的内部配方。若已有可用 Omni checkpoint，先做能力诊断，跳过已满足的阶段；不默认从零预训练。

| 阶段 | 数据与目标 | 更新范围与关键验收 |
|---|---|---|
| A 模态基础与对齐 | 音频／视觉感知、ASR、TTS、音视频问答；已有底座不足时才补 CPT | 先 adapter／相关 encoder，再按缺口解冻；确认输入确实被使用，保留文本能力回归 |
| B 任务与声音 SFT | 口语任务对话、工具轨迹、音色／风格与自然对话混合 | 分开语义与声学生成 loss；固定输入、目标、stop、mask 与分母；合成数据补可控覆盖 |
| C 全双工 SFT | 连续双音轨＋视频时间线，含重叠、沉默、纠正、附和、打断与接续 | 训练因果输入：只见当前可用观测；学等待／开始／继续／停止，不能删除所有静音时间格 |
| D 定向蒸馏／偏好 | 语义配对 audio/text、声音偏好对、坏例修复 | 文本 teacher 可补语义，无法凭转写补语气、背景声与视觉证据；DPO 需固定可比条件 |
| E 窗口交互 RL | 从长历史抽关键窗口，同一因果上下文采样候选，计算时机＋内容／完整性 reward | 首先同步；根据真实模型选 text stream／控制头／audio action 的更新边界；复查所有受共享参数影响的能力 |
| F 完整会话 RL | 用户 simulator＋工具环境，包含目标变更、失败和多轮恢复 | 评估最终任务与中间交互；跟踪整个 episode 状态。窗口 RL 通过不代表这一阶段通过 |
| G 回归与迭代 | 固定 held-out、真实失败、基础能力 replay | 比较完整性、任务成功、自然度、时机、成本；按退化来源决定继续 RL、补数据还是修执行协议 |

**先把高价值行为分开学，再验证能否一起工作。** 时机可由显式 head、文本／静音符号或联合流隐式承载；不要求为了这条路线强改模型。mask 只决定哪些 action 受直接监督，共享 backbone 变化仍可能影响声音和理解。

### 算法真正困难在哪里

1. **Action 是什么？** 要区分内容 token、speech codebook、开始／停止符号、无新文本的时间格、工具调用。客户端取消如果不是模型采样动作，就不能伪造一个 action logprob。环境输入、强制历史和普通 batch padding 不参与 policy loss；承载“继续等待”的模型采样符号则可能必须参与。
2. **看到了什么？** 全双工是部分可观测的时间过程。离线 forced alignment 可以产生标注，不能把未来语音、整句最终转写或后到视频送进当前策略输入。训练侧重算 logprob 必须重建 rollout 当时实际可见的前缀和模态状态。
3. **奖励该给谁？** 整段最终分数会把成功／失败扩散到大量无关音频 token。候选方案是事件窗口／turn／episode 分层评分；局部窗口补可解释信用，完整 episode 补长期任务后果。时间步长可变时要明确 discount 与 return 的时间单位，不能直接沿用 token 步数解释真实等待成本。
4. **GRPO 的组可比吗？** 固定相同 prefix、任务、初始环境状态及 reward 版本；缺候选、超时与零方差组按 estimator 的明确规则处理。闭环用户会因回答不同而改变后续输入，不强求候选未来用户音轨相同；固定录音只检验条件响应，无法证明策略的反事实交互效果。
5. **长会话怎样更新？** 首轮固定 episode／窗口内 policy bundle；先在边界切版本。若后续支持跨版本续采，要逐片段保存 behavior identity 和实际概率，并采用已确认的算法校正，不能只补一个 episode 末尾版本号。
6. **何时 PPO，何时 GRPO／DPO？** 有可靠 preference pair 时可做 DPO；在线组采样与评分成本可控时可评估 GRPO-style；需要 learned value 的候选方案可评估 PPO。选择依据是 action、奖励方差、轨迹长度与成本，不能仅因节省 critic 就预判 GRPO 效果更好。

## 数据：从“问答样本”升级为“双方事件时间线”

### 数据分层与优先采集

| 数据层 | 解决什么 | 准备方式与风险 |
|---|---|---|
| 模态基础 | 听清、看懂、说清 | ASR、音频事件、带时间的视觉 grounding、语音生成；按当前能力缺口补齐 |
| 自然交互 | 学真实停顿、重叠、语调、附和 | 独立双方音轨＋统一时钟；Fisher／Seamless 是机制参照，需核许可、中文与业务域差异 |
| 业务任务 | 澄清、改目标、工具失败后继续 | 定向脚本＋TTS 扩覆盖，真实录音校准；不能只合成理想用户与成功调用 |
| 全双工难例 | 区分相似声音背后的不同意图 | 成对构造：思考停顿／说完，附和／夺话，环境噪声／指令，自我纠正／新任务；配完整上下文 |
| 闭环经验 | 发现模型输出引起的用户反应与任务失败 | simulator 采样，再用真实用户回归；记录 simulator 版本、场景与失败类型 |

合成数据适合扩业务意图、工具 schema、角色和可控扰动；真实数据校准接话节奏、口音、情绪和重叠。先小批审核，再按错误桶补数，不预设一个通用真实／合成比例。把评测集按 speaker、session、场景／脚本模板划开后再切 clip，避免同一长会话或合成模板泄漏。

### 最小记录：四条时间线，两个视图

保存 **采集／到达、模型实际消费、生成／送出、客户端播放／工具执行** 四条时间线；跨进程同时记 clock domain、同步误差或可比较的相对时间，不能直接减不同机器的裸时间戳。

| 字段组 | 最小内容 |
|---|---|
| 身份与条件 | session／episode／turn／event／parent ID，任务、speaker、原媒体 hash、processor／codec／模板版本 |
| 连续观测 | 双方独立音轨、video frame 时间、采样率、clip offset、chunk 到达／可见边界；增强与随机种子 |
| 策略与输出 | policy bundle、模型实际 action、逐位置 behavior logprob、sampling config、loss/action mask、text/audio 对齐 |
| 执行反馈 | generated／sent／played frontier，取消／打断，工具 request/result/commit ID，terminal 与 truncation 原因 |
| 奖励与消费 | reward 各分量、评分版本、validity、group 与分支身份、重复／超时原因、是否进入 update 及消费 checkpoint |

**审计视图保存完整生成，策略上下文按实际经历重建。** 未播放的输出仍可保留以分析浪费和模型尝试，但不能默认用户听过；是否对被取消动作施加 loss 是算法选择，需明确。工具已经提交的副作用则不能因为音频取消而当作从未发生。

示例：助手已生成三句，用户只听到第一句就改了地点。下一轮要根据“已播第一句＋用户纠正＋实际工具状态”继续；若把三句全算进用户历史，模型会以为用户已知尚未听到的确认。OpenAI 会话文档的未播音频截断提供了明确的生产协议参照；训练采用相同边界是本文工程建议。

### Reward：先拆分，再做汇总

建议保留 `task / grounding / content / timing / completeness / voice / execution / cost` 分量，不在数据层只留一个总分。业务成功和完整性先作为护栏，再比较时机与成本；具体权重需用反例和人评校准。

| 奖励／约束 | 应测什么 | 必须加入的反例 |
|---|---|---|
| 接话与等待 | 真正说完后响应；句中停顿不抢话 | 一直沉默也有低抢话率；所有停顿都接话也有低延迟 |
| 让话与继续 | 真正夺话后及时停；普通附和后继续完成 | 从未开口却拿到“打断成功”；听到“嗯”就终止回答 |
| 内容与任务 | 语义、视觉证据、工具结果及最终目标 | 自信复述、假装调用成功、忽略纠正 |
| 声音与完整性 | 可懂度、自然度、所需音色、回答完整 | 只输出很短且清晰的声音；ASR 低错误却信息缺失 |
| 执行与成本 | 播放／取消正确、计算与延迟成本 | 增大预生成缓存缩短局部指标，却增加打断后废弃音频 |

规则／VAD 适合初步时序标签，ASR＋文本 judge 适合内容，音频 judge／抽样人评检验韵律与体验，工具状态检验真实完成。文本 judge 看不到声音，不应独自判定“自然度”。保留 judge 置信度／分歧与版本；评分前先用固定反例检查偏差。Infra 故障和正常任务失败分开归因，不能统一补零后当成学习信号，也不能从运维成功率分母中删除。

## 配置与 Infra：在原有闭环上补齐时间、执行和提交边界

通用链路仍是环境→流式观测→rollout→trajectory→reward→update→发布。每个阶段都应输出同一 episode 的证据；reward 与 trajectory 通过身份 join 后，只有满足 estimator 准入要求的记录才能进入 update。

| 子系统 | 建设优先项 | 容易破坏的语义 |
|---|---|---|
| 环境／会话 | 双向 event scheduler、reset、工具 mock、用户 simulator、virtual clock、确定性 seed 与回放 | 固定音轨回放不是会响应模型的用户；快速仿真不能提前交付未来观测 |
| 媒体处理 | 按原媒体与 processor 版本缓存，解码／H2D overlap，有限队列；只在可等价条件下复用特征 | encoder 更新、随机增强、抽帧或时序一变，旧 feature cache 可能失效 |
| 流式 rollout | 会话级状态隔离，按 chunk deadline 调度，prefill／decode 资源与限额，取消回收 | KV 之外还可能有 recurrent／encoder／codec state；长 prefill 不得阻断活跃音频流 |
| 轨迹缓冲 | 原媒体对象引用＋不可变 event log；训练前展开为 packed tensors；设 RAM、对象存储、队列上限 | 只有音频／文本字符串，无法重建可见前缀；重试产生重复消费；静音时间格被删除 |
| 评分服务 | 便宜确定性检查与昂贵 judge 分级；保留抽查；版本化缓存和异步 join | 只给快速／高分样本评分而改变经验分布；超时候选悄悄丢失 |
| Updater | 先小路径 scoring／gradient parity，再引入 DP/FSDP、TP/CP、recompute 与精度优化 | 跨模态 mask、loss 分母、Hybrid state 边界和行为概率改变 |
| 权重发布 | Thinker／Talker／head／codec／processor 依赖 manifest，转换校验、replica refit 与流量准入 | 部分更新后混用新旧模块；worker 收完 bytes 却未完成 refit 就接流量 |
| 恢复 | model／optimizer／RNG／sampler、consumed frontier、待评分条目与环境 checkpoint 的一致边界 | 重放已提交工具副作用，或丢掉尚未消费的 reward；训练恢复与在线会话续接混为一谈 |

异步适合掩盖 rollout、工具与 reward 长尾，但会增加版本陈旧和缓存积压。先同步基线，再一次只放开一个阶段；队列上限、样本年龄、版本差和拒收规则要与算法一起定义。不要同时引入复杂并行、异步、低精度和联合模块 RL。

### 性能指标怎样连到业务效果

建议同时保留三层看板，统计同一固定 workload、质量门槛与资源边界。

| 层级 | 主指标 | 分母／口径 |
|---|---|---|
| 业务 | 多轮任务成功率、完成所需轮数、打断后正确接续、用户偏好 | 全部任务尝试，区分模型失败与 Infra 故障，按语言／噪声／场景分桶 |
| 交互 | 首播时延、用户夺话到最后一帧旧音频的时延、抢话率、漏接率、附和后继续率、chunk gap／underrun | 分别记录事件检测时间与实际播放时间；缺响应计失败，不能只报成功样本延迟 |
| 训练系统 | 完整有效 group／秒、实际 gradient-active 工作量、完成一次质量达标迭代的时间与总 accelerator-hours | 环境、rollout、codec、judge、refit、update、失败与重试成本都入账；group 数不能单独代表学习收益 |

MFU 用于解释 **updater** 的计算效率；明确有效模型 FLOPs、MoE active 路径、padding、重计算和硬件精度峰值的计算口径。RL 整体有真实等待、声音解码和评分服务，不能用 updater MFU 推出全链路效率。预训练的 strong／weak scaling、固定模型下的训练加速与 RL goodput 分开验收。

离线窗口仿真若各阶段可持续超实时，候选可用 virtual clock 加速；需要真实墙钟反馈、网络、音频设备或物理动作时另测实时路径。只在执行语义相同时比较“多少训练音频秒／墙钟秒”，不承诺通用倍数。

## 生产陷阱与排障顺序

| 症状 | 先查什么 | 再决定什么 |
|---|---|---|
| RL 后接话变快但任务更差 | 空／短回复、语义 reward、终止规则、附和误判、基础能力 held-out | 补完整性约束与对照数据，还是限制更新范围 |
| ratio／KL 异常 | 原媒体／processor→可见前缀→action mask→behavior 权重／采样概率→train scoring | 修契约后再检查 kernel／精度差异 |
| 用户打断后旧声音继续 | VAD／语义判断→取消传播→生成／发送／播放缓冲→turn epoch | 属于模型没让话，还是执行器没停播 |
| trainer 等数据、rollout 又很忙 | completed→reward-ready→accepted→consumed→gradient-active 的数量与年龄 | 优化解码、评分、完整组等待或版本拒收，避免盲加 rollout 并发 |
| 扩容后 MFU 升但迭代变慢 | rank skew、queue、媒体展开成本、judge 饱和、weight sync 暴露停顿 | 用一次端到端 trace 判断资源配置 |
| 恢复后行为变化／重复动作 | 完整 bundle、环境／工具提交、reward 去重、消费 frontier、cache 身份 | 回退、重放或丢弃，留下可解释的恢复账本 |

## 相邻系统与证据边界

复用 [Agentic RL 状态与调度](agentic_rl.md)、[框架选型](rl_framework_selection.md)、[Omni 模型／loss／serving](../../05-embodied-infra/topics/omni_models_infra.md)和[推理状态](../../03-inference-infra/topics/inference_state.md)。本章只补连续时间交互、全双工数据与 RL 的特有约束，不复制完整训练／推理手册。

本轮已经核原始来源并形成方案；**没有运行模型、实现 simulator、训练、测吞吐或完成业务评测**。闭源训练细节、中文业务迁移、视频 grounding 与 full-duplex 联合 RL 的完整开放 recipe，以及所有目标 backend 的训练／scoring／发布兼容性仍需补证。公开资料的局部结果不构成某个业务模型已达到相同效果的证据。

[回交付路线](../../05-embodied-infra/roadmaps/omni.md#duplex-project) · [O4 全双工验证](../../practice/experiments/omni_sft_rl_lab.md#duplex-o4) · [共享 P1](../../research/reading_queue/P1.md#omni-20261031) · [材料索引](../../research/MASTER_READING_LIST.md)

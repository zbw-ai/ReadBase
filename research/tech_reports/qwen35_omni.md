# Qwen3.5-Omni：从模态对齐到可训练、可流式服务的系统

## 论文信息

| 字段 | 核验结果 |
|---|---|
| 标题 / 作者 | *Qwen3.5-Omni Technical Report* / Qwen Team |
| 版本日期 | v1：2026-04-17；本笔记依据 v2：2026-04-21 |
| 原始来源 | [arXiv 元数据](https://arxiv.org/abs/2604.15804v2) · [全文](https://arxiv.org/html/2604.15804v2) |
| 核验 | 2026-10-31，核对 citation title / author / date、§2–4 机制、§2.5 性能口径及 §5 评估分类；[证据索引](../tracking/audits/2026-10-31-omni/sources.json) |
| 状态 | 核心机制笔记已整理（SUMMARIZED）；用户阅读、源码验证与实验未完成 |
| 学习入口 | [Omni 专题](../../05-embodied-infra/topics/omni_models_infra.md) · [学习路线](../../05-embodied-infra/roadmaps/omni.md) · [P1](../reading_queue/P1.md#omni-20261031) |

文中的“报告披露”来自该版本；“工程推断”是本库的迁移建议。模型能力、规模与性能数字均为作者报告，本库没有复现。本文关注算法如何改变 Infra，不推定任何公司内部模型采用相同架构。

## 架构概览

报告 §2 延续 Thinker–Talker 分工：文本、视觉、音频经 tokenizer / Vision Encoder / AuT 转成输入表示；Thinker 负责理解、推理与文本输出；Talker 利用上下文与 Thinker 表示生成语音 token，MTP 补充 residual codebooks，因果 ConvNet Code2Wav 增量输出波形。

- **Hybrid Attention MoE**：Thinker 与 Talker 均采用混合骨干，包含 Gated Delta Net（GDN）。工程上要同时管理 attention KV 与 recurrent state，不能用一个“KV token 数”描述所有层。
- **时间对齐**：TM-RoPE 配合显式文本时间戳。仅依靠绝对时间位置编号会让长视频索引过稀；时间戳提供更直接的 timecode 表示，但也占用上下文。抽帧、音频切片和位置生成必须保持同一时间轴。
- **ARIA（Adaptive Rate Interleave Alignment）**：把文本和语音由双轨组织为自适应交错流。§2.4 给出的约束是任意前缀的累计 speech/text token 比不超过该样本整体比例。它解决 tokenizer 编码速率不匹配带来的漏词、读音和数字表达问题，不等价于固定“每几个文本 token 发一帧音频”。
- **任务边界**：文本/语音输出、工具调用与音视频编码任务，不构成机器人关节动作 policy 的证据。这里的 agentic action 不能直接解释成执行器动作。

报告没有给出足以重建 ARIA 全部训练和在线边界处理的代码。本库不假设在线已知未来完整语音长度；特殊 token、起始前缀与末尾 flush 的实际规则仍需实现证据。

## 训练系统设计

### 预训练是理解 SFT 的前提

§3 分三阶段：S1 冻结 LLM，先训练 adapter 再训练各 encoder；S2 全参数多模态学习；S3 增加长音视频及上下文长度。后训练接收到的表示已具备跨模态基础，不能把少量文本 SFT 当成从零获得 Omni 能力。

### Thinker：能力整合、模态一致性、交互质量

| §4.1 阶段 | 报告披露 | Infra 需要承接的契约（工程推断） |
|---|---|---|
| Specialist Distillation | 各领域 teacher 从 Qwen3.5 base 经独立 SFT/RL 得到，覆盖文本、视觉、音频；生成领域数据整合到统一模型 | teacher/checkpoint、输入模态、生成条件与数据版本可追踪；不能只保存答案字符串 |
| On-Policy Distillation | 成对 audio/text query，使用文本条件下生成的较高质量回答，作为对应音频条件的蒸馏目标 | paired query 必须语义对应；分别记录 teacher/text 条件与 student/audio 条件，评估音频特有信息是否丢失 |
| Interaction-Aligned RL | 多轮轨迹围绕语言切换、persona 一致性、长对话指令遵循设计 reward | 会话/轮次身份、历史上下文、reward 版本与轨迹终止原因一起保存 |

报告将第二阶段称为 OPD，并给出上述文字流程；它没有在该节完整披露 loss、teacher 更新频率、采样与重要性校正细节。不能直接改写成“标准 reverse-KL token distillation”或假定已有可复现的 OPD recipe。第三阶段也未在该节指定完整优化器与奖励系数，不能因 Talker 使用 GSPO 就断言 Thinker 同样如此。

### Talker：声音能力与偏好单独成链

§4.2 给出四阶段：General → 高质量数据 CPT / Long-Context → Reinforcement Learning Stage → Speaker Fine-tuning。第三阶段先用人标多语言偏好对进行 DPO，再引入规则奖励与 GSPO；最后做轻量说话人微调。**DPO 是偏好优化，和需要在线 rollout 的 GSPO 有不同的数据生产与系统成本**，虽然报告将两者放在同一阶段。

Talker 输入有多模态上下文与语音控制条件，不能简化为独立 TTS 的“文本→声音”。偏好对需固定内容与音色等条件；质量评价要拆开可懂度、内容一致性、自然度、说话人相似度和响应时延。具体奖励函数、采样组配置、loss mask、冻结范围与 optimizer recipe 均需进一步证据。

## 并行策略

**未披露完整训练拓扑**：报告不足以确定训练 TP / PP / CP / EP / DP 组合、节点布局及实际 batch。§2.5 说明 Plus 与 Flash 部署资源和并行策略不同，但没有提供可直接复用的完整部署清单。

工程推断：先单卡/最小副本做 loss 与梯度基准，再逐项引入状态分片、EP 和 CP。尤其 Hybrid Attention 需要确认 GDN 的序列边界、state reset、CP 传递及 backward 支持；full-attention 的 CP 示例不自动适用。通用机制复用 [5D 并行](../../02-training-infra/topics/distributed_training.md)、[MoE](../../02-training-infra/topics/moe.md)与 [FSDP](../../02-training-infra/topics/fsdp.md)。

## 显存优化

报告披露 GDN 有助于降低长上下文 KV I/O；这不是整机显存与长度无关的证据。剩余 attention 层的 KV、recurrent state、encoder features、Talker state 和输出缓冲仍占显存。

工程推断：分开建立训练状态、activation、音视频临时张量与 serving state 的账本。冻结模块仅消除相应梯度/optimizer 状态；如果梯度仍须穿过该模块到上游可训练参数，不能随意使用 `no_grad`。缓存 frozen encoder features 只有在权重、前处理、裁剪和增强固定时才可复用。

## 通信优化

报告未披露训练 collective 的分解、overlap 收益或跨节点通信实测。不能据此声称某种 EP 布局已验证。

工程推断：训练同时观察 MoE dispatch/combine、梯度同步与变长样本造成的 rank skew；serving 则关注 encoder→Thinker→Talker→Code2Wav 的队列和状态传递。模块拆卡之前先量化中间表示字节数、发送频率和回压传播，避免把低 GPU 利用率误判为需要更多并行。

## 集群规模

**训练 GPU 型号/数量、网络拓扑、总训练时长与完整算力成本未在本次核验章节中披露**。摘要的数千亿级参数描述不提供可用于精确容量规划的配置。

报告 §3 给出 S2 长度 32,768、S3 长度 262,144；§4.2 Talker 长上下文阶段为 64k。这些是不同阶段和模块的作者配置，不能统一写成“所有模块均为 256k”。

## 工程经验

### 延迟必须测到可播放音频

§2.5 Table 2 使用 **internal vLLM**，并对 MTP 和 codec decoder 使用 `torch.compile`、CUDA Graph；表格标为 theoretical first-packet latency。单并发音频输入下 Flash / Plus 的 Overall Latency 分别为 **235 / 435 ms**。这些是作者在特定条件下的数字，资源分配不同，不能据此做同硬件性价比结论，也不是公网客户端 SLO。

Thinker TTFT、Talker TTFC 和 Code2Wav 时间不是简单相加关系：交错流与并行构成关键路径。工程上从输入到达、排队、prefill、首文本、首 codec、首波形直至客户端首播分别打点；稳态还需记录播放 underrun、chunk gap 和取消后的残留输出。

### 对齐错误会伪装成性能问题

长音视频、文本时间戳与 ARIA 改变了有效序列结构。只优化 padded tokens/s 可能增加 padding，或通过减少帧、截断输入降低真实任务难度。性能报告必须固定数据、有效训练目标和输出质量，并把数据处理失败、丢样、超时计入分母。

## 对行业的影响

**本库判断**：Omni 后训练要求把跨模态数据、teacher 供给、交互轨迹和多阶段流式推理看成一套系统。算法接口决定状态边界和关键路径，训练后端的单点吞吐已不足以描述完整工作成本。这里是从报告机制得到的工程推断，不是已测得的行业通用收益。

## 我的收获

对 Infra 岗位，优先能解释四件事：哪一部分参数在学、哪些 token/帧进入 loss、rollout 的采样策略和 reward 针对什么输出、优化后是否保持相同任务语义。下一步落到 [SFT/RL 正文](../../05-embodied-infra/topics/omni_models_infra.md#sft)与 [O0–O6 实验设计](../../practice/experiments/omni_sft_rl_lab.md)，目前未实验。

## 后续演进

以本报告为架构和算法参考；公开 [Qwen3-Omni 仓库](https://github.com/QwenLM/Qwen3-Omni)只能作为旧代可运行实现的比较入口，不能代表 3.5 的 Hybrid Attention、ARIA 或完整后训练实现。本次官方 HF 模型查询未见 3.5-Omni 开放权重；报告说明可经 API 访问，不据此推断已经开源或全渠道不存在权重。

需要后续补证的是：3.5 实现/权重与许可、ARIA 在线规则、各训练后端的 module coverage、Thinker/Talker 联合更新边界、准确 reward 与 checkpoint 格式。旧代实验若用于机制学习，要明确架构差异，不冠以“3.5 复现”。

## 面试高频问题

1. **为什么 Omni 不只是 VLM 加 TTS？** Talker 使用多模态上下文，语音的内容和韵律受交互状态约束；数据、loss、实时调度都发生变化。
2. **为什么长上下文需要 KV 之外的账本？** Hybrid 模型还维护 recurrent state，音视频特征、音频 renderer 和会话缓冲也有生命周期。
3. **Thinker 和 Talker 可以共用同一套 RL 吗？** 通用调度可复用，但 action、logprob、reward、序列长度和评价目标不同，需逐项验证。
4. **235 ms 能直接当目标吗？** 它是作者 internal vLLM 的特定理论口径，需重新定义本地端到端测量边界和资源配置。

## 生产环境思考题

- 音频条件 OPD 的文本配对丢失语气、背景声音或说话人信息时，怎样避免蒸馏错误能力？
- Talker 更新而 codec/Thinker 不更新时，哪个版本元组决定一条 rollout 可否进入训练？
- Hybrid state 与 attention KV 在请求取消、重试、前缀复用时是否一并隔离？
- 实际吞吐提高来自计算优化，还是帧率降低、样本丢弃、mask 改变？怎样用固定数据回放区分？

[回专题](../../05-embodied-infra/topics/omni_models_infra.md) · [回补录](../tracking/backfill/2026-04.md#qwen35-omni) · [来源索引](../MASTER_READING_LIST.md)

相关判断：[Omni 优化的目标与状态契约](../insights/omni_optimization_contracts.md)，为基于本报告的工程假设，未实验。

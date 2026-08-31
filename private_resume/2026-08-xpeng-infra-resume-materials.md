# 小鹏大模型训练推理 Infra 简历素材底稿

更新时间：2026-08-29
目标岗位：外部社招，大模型训练 / RL / Agentic RL Infra 高级工程师
状态：第一版事实底稿，尚未修改原 PDF

## 0. 使用原则

1. 对外只写开源技术底座 `verl`、`AReaL`，不写内部仓库名 `llm_train`、`Trail`。
2. 简历正文突出问题、技术决策和结果，不罗列分支、脚本和 commit。
3. `已验证` 的数字可以进入简历；`实验性` 内容只能描述为“设计、实现、打通、定位”，不能写成性能或模型效果收益。
4. 性能收益必须和固定 workload、硬件规模、统计窗口绑定；算法收益必须有 checkpoint 下游评测。
5. 对 MOPD、prefix cache 等尚未完成效果闭环的工作，保留技术深度，但不夸大结论。

## 1. 推荐职业定位

### 一句话定位

大模型训练推理 Infra 高级工程师，聚焦基于 Megatron-Core、verl、AReaL 的 SFT、RLVR 与 Agentic RL 系统，具备长上下文训练、异步 rollout、训练推理解耦、在线蒸馏、轨迹数据链路和性能正确性联合优化经验。

### 30 秒自我介绍素材

我目前在小鹏机器人负责大模型后训练基础设施，工作分为两条主线：一条是基于 verl 和 Megatron-Core 建设 SFT、RLVR 训练能力，覆盖 Qwen3/Qwen3.5 dense、MoE、32K-256K 长上下文，以及 vLLM/SGLang rollout 和 fully async 训练；另一条是基于 AReaL 建设 Agentic RL 系统，重点解决 9B/128K 场景下 rollout 长尾、cohort 等待、轨迹利用、policy staleness、在线多 Teacher 蒸馏和端到端可观测性问题。我的特点不是只让任务跑起来，而是通过可验证实验同时守住性能、数值正确性和模型效果。

## 2. 工作时间线

### 华为

华为 - 计算产品线 - 昇腾计算 - AI 训练优化工程师
`2023.07 - 2025.11`

原简历中的华为职责、项目和成果暂不改动，仅将截止时间由“至今”更新为 `2025.11`。

### 小鹏机器人

小鹏机器人 - 大模型训练推理 Infra 高级工程师
`2025.11 - 至今`

建议部门总述：

> 负责大模型后训练基础设施研发，基于 verl、Megatron-Core 和 AReaL 二次开发 SFT、RLVR 与 Agentic RL 训练系统，重点建设长上下文训练、异步 rollout、训练推理解耦、在线蒸馏、性能可观测性和轨迹数据治理能力，支持 Qwen3/Qwen3.5 dense/MoE 模型在多机 A100 集群上的训练与实验验证。

## 3. 推荐对外写法 v2

参考“项目标题 + 职责概述 + 粗体能力标签 + 技术机制 + 量化结果”的表达方式。对外不出现内部仓库名，只写开源技术底座和个人承担的工作。

### 基于 verl 的 SFT / RLVR 训练系统

负责基于 verl、Megatron-Core/MBridge 二次开发大模型 SFT/RLVR 训练链路，支持 Qwen3/Qwen3.5 dense、MoE、32K-256K 长上下文和多机 A100 训练，主要工作如下：

- **异步 RLVR 吞吐优化**：面向 Qwen3-30B-A3B 32K、32 张 A100-80GB 场景适配 fully async policy，解耦 Trainer 与 Rollouter；基于 rollout 生产率和 trainer 消费率调整 gen-TP、vLLM 实例数及 3T+1R/2T+2R 资源配比，将异步初始吞吐由 76 提升至 211-255 tokens/s/GPU，超过约 200 tokens/s/GPU 的同步基线；2T+2R 配置进一步达到 236-293 tokens/s/GPU，trainer idle ratio 由 0.41 降至 0.10-0.14。
- **长上下文 SFT 优化**：完成 Qwen3.5 9B/27B/32B 等模型 128K SFT 训练适配，覆盖 16-64 张 A100；基于张量级显存账定位 fp32 logits、gradient buffer、offload PCIe、CP 通信和长样本激活峰值，验证 9B 场景 TP=2/CP=8 相比 TP=4/CP=4 将 step time 由约 163s 降至 102s，并建设 grad-norm spike 的 batch 级 tracing、超长样本截断检查和 checkpoint deadlock 修复能力。
- **训练后端与稳定性**：接入 vLLM/SGLang rollout 后端，支持 MoE、DAPO、rule/code/model reward 和异步 parameter sync；修复 uneven dataset split、reward-loop 初始化、final parameter sync、Megatron distributed checkpoint 等问题，建设 Qwen3.5 FLOPs/MFU、阶段耗时和显存监控，为多机训练提供可复现 recipe、失败诊断和恢复能力。

### 基于 AReaL 的 Agentic RL 与在线蒸馏

负责 Qwen3.5-9B 128K 长上下文与多轮 Agentic RL 训练链路建设和性能优化，覆盖 online rollout、tool/sandbox environment、训练推理异步调度及在线蒸馏，主要工作如下：

- **长上下文与端到端性能**：面向 DeepSWE 与 Seta Terminal 多轮环境交互场景，交付 Qwen3.5-9B 128K Agentic RL 训练链路；DeepSWE 端到端稳态单步耗时由 6467s 降至 2301s（-64.4%），有效 Token 吞吐达到 146.8 tok/s/GPU；Seta Terminal 由 2240s 降至 770s（-65.6%），有效 Token 吞吐达到 233 tok/s/GPU。
- **Rollout 与调度优化**：通过 CUDA Graph 将 decode 阶段加速 6-8x，Prefix Cache 优化使 prefill 阶段耗时降低 44%，并优化 Sandbox 并发以提升 vLLM 有效并发；重构 Rollout 调度链路，通过 Gateway 实现流式补位、均衡分发与失败请求管理，使 Rollout 阶段平均推理吞吐提升 60%，Rejected Group 比例由 33.18% 降至 2.73%（-30.45pp）。
- **多 Teacher 在线蒸馏**：设计并实现 On-Policy Distillation/MOPD，支持 trajectory 按 data source 路由至对应 Teacher 计算 logp，以及 teacher score 校验、mopd_pg loss、mixed-domain data、equal-trajectory weighting、online session drain、断点续训和 held-out paired evaluation；完成 9B Terminal 单 Teacher 30-step 与双 Teacher early canary，采用 FUNCTIONAL/NUMERIC/EFFICACY 分层门禁，避免将系统跑通误判为数值正确或模型效果提升。

建议最终简历正文优先采用以上 6 条。篇幅不足时，先合并“训练后端与稳定性”到前两条，再视目标岗位删除长上下文 SFT 或多 Teacher 蒸馏中的一条。

## 4. 事实展开与证据边界

以下 5 条保留更完整的证据说明，供修改简历和准备面试使用。

### 4.1 后训练框架建设

基于 `verl + Megatron-Core/MBridge` 二次开发统一 SFT/RLVR 训练链路，支持 Qwen3/Qwen3.5 dense、MoE 模型及 32K-256K 长上下文场景，完善 TP/PP/CP/DP/EP、packed sequence、dynamic batch、recompute、offload、checkpoint 和 rule/model reward 配置，并接入 vLLM、SGLang rollout 后端及多机 Fuyao 任务部署。

证据等级：`可写`。其中“32K-256K”表示覆盖过的工程配置范围，不等于所有模型和长度均已形成稳定生产基线。

### 4.2 Fully Async RLVR 性能优化

面向 Qwen3-30B-A3B 32K、32 张 A100-80GB 场景，完成 verl fully async policy 适配与训练/rollout 资源解耦；通过 gen-TP、vLLM 实例数、Trainer/Rollouter 配比和 batch supply 调优，将异步初始吞吐由 `76` 提升至 `211-255 tokens/s/GPU`，超过约 `200 tokens/s/GPU` 的同步基线，并在 2T+2R 配置下观测到 `236-293 tokens/s/GPU`、trainer idle ratio 由 `0.41` 降至 `0.10-0.14`。

证据等级：`强可写`。面试时必须说明这些是代表性稳态 step 区间，不包装成全程平均值；验证硬件为 `4 节点 x 8 A100-80GB`。

### 4.3 长上下文 SFT 与显存/稳定性分析

负责 Qwen3.5 长上下文 SFT 适配和性能诊断，在 16-64 张 A100 上打通 9B/27B/32B 等模型的 128K 训练路径；基于张量级显存账定位 fp32 logits、gradient buffer、offload PCIe、CP 通信和长样本激活峰值，验证 9B 场景 `TP=2, CP=8` 相比 `TP=4, CP=4` 的 step time 从约 `163s` 降至 `102s`，并建设 grad-norm spike 的 batch 级 tracing、超长样本截断检查和 checkpoint deadlock 修复能力。

证据等级：`可写但需精修`。9B 对比数字来自 v1/v2 实验记录；32B 128K 跑通约 50 step 后仍因长样本 OOM，不能写成稳定生产运行；27B 128K 有 working recipe 证据。

### 4.4 128K Agentic RL 性能与轨迹分析

面向 Qwen3.5-9B、128K、32 张 A100 的 R2E/SWE Agentic RL 场景，建立 overlap-aware step、rollout supply、cohort tail、cache、token participation 和 trajectory lineage 性能体系；量化历史基线 step 均值 `83.89 min`，其中 rollout wait `73.21 min / 87.27%`，进一步定位长上下文后期 LLM 调用、8-way cohort straggler 和轨迹供给不足为一阶瓶颈，而非仅以 prefix-cache 命中率判断性能。

性能闭环结果：

- DeepSWE 场景端到端稳态单步耗时由 `6467s` 降至 `2301s`（`-64.4%`），有效 Token 吞吐达到 `146.8 tok/s/GPU`。
- Seta Terminal 场景端到端稳态单步耗时由 `2240s` 降至 `770s`（`-65.6%`），有效 Token 吞吐达到 `233 tok/s/GPU`。
- CUDA Graph 将 `decode` 阶段加速 `6-8x`；Prefix Cache 使 `prefill` 阶段耗时降低 `44%`；Sandbox 并发优化提升 vLLM 有效并发。
- 重构 Rollout 调度链路，通过 Gateway 实现流式补位、均衡分发和失败请求管理，使 vLLM 并发稳定在理论值附近，Rollout 阶段平均推理吞吐提升 `60%`，Rejected Group 比例由 `33.18%` 降至 `2.73%`（`-30.45pp`）。

补充技术动作：

- 打通 generated -> manager -> workflow -> trainer -> loss -> policy gradient 六层逐轨迹 join。
- 在真实 6-step tracing run 中闭环 `223 admitted -> 180 generated/rewarded -> 96 exported -> 96 consumed`，96 条全部 exact join。
- 识别 94 条 gradient-active、2 条 compact-filtered；后者消耗 `159,330` full-sequence tokens，占该窗口 trainer token processing 的 `3.91%`，但不产生梯度。
- 将 partial、stale、waiting/final-drain 与 terminal waste 分开，避免把 `generated - consumed` 直接判为浪费。

证据等级：`强可写`。端到端结果统一表述为“稳态单步耗时”；CUDA Graph、Prefix Cache 和 Rollout 平均推理吞吐必须分别限定为 decode、prefill 和 Rollout 阶段收益，不能单独外推为端到端加速。有效 Token 吞吐的面试口径固定为“实际参与训练的有效 Token / GPU / 端到端稳态时间”。

### 4.5 OPD/MOPD 算法与 Infra 联合建设

设计并实现 on-policy distillation 与多 Teacher MOPD 训练链路，覆盖 trajectory `data_source` 路由、Teacher scatter/gather、teacher-logp 校验、`mopd_pg` loss、mixed-domain data、equal-trajectory weighting、在线 session drain、断点续训和 held-out paired evaluation；完成 9B Terminal 单 Teacher 30-step、SWE 100-step 系统稳定性验证及双 Teacher early canary，形成“FUNCTIONAL、NUMERIC、EFFICACY”分层验收机制。

算法判断素材：早期 OPD 实验修复 reverse-KL advantage 缺失、token filter、kept-token normalization 和 thinking template 等问题，将 6-benchmark 能力退化从约 `10-20pp` 收敛到 `1.07pp`；最终通过 SFT/Teacher 能力对比确认 Teacher 弱于 Student，及时否决“继续调 Infra 即可获得模型提升”的错误假设，并将后续工作转向更强 Teacher、GRPO 混合目标和严格下游评测。

证据等级：`前半段可写，效果结论谨慎写`。MOPD 当前证明了训练链路和早期稳定性，没有证明多域模型效果提升；SWE 100-step 历史 run 存在 mapping-500 数据污染，只能作为系统稳定性证据。

## 5. 推荐的简历压缩版本

如果小鹏经历只能保留 4 条，建议使用以下组合：

1. **框架与模型覆盖**：基于 verl/Megatron-Core 和 AReaL 建设 SFT、RLVR、Agentic RL 后训练系统，支持 Qwen3/Qwen3.5 dense/MoE、32K-256K、多机 A100 和 vLLM/SGLang。
2. **性能硬结果**：32 A100 上将 Qwen3-30B-A3B 32K fully async RLVR 吞吐从 76 提升至 211-255 tokens/s/GPU，2T+2R 场景达到 236-293 tokens/s/GPU，idle ratio 降至 0.10-0.14。
3. **Agentic RL 性能优化**：DeepSWE 端到端稳态单步耗时由 6467s 降至 2301s（-64.4%），有效 Token 吞吐达到 146.8 tok/s/GPU；Seta Terminal 由 2240s 降至 770s（-65.6%），有效 Token 吞吐达到 233 tok/s/GPU；Gateway 调度重构使 Rollout 阶段平均推理吞吐提升 60%，Rejected Group 比例由 33.18% 降至 2.73%。
4. **算法-Infra 联合能力**：实现 OPD/MOPD 多 Teacher 路由、打分、loss、混域训练、恢复和评测链路，以 FUNCTIONAL/NUMERIC/EFFICACY 分层门禁防止用系统跑通冒充模型有效。

若能保留第 5 条，再加入长上下文 SFT 的 TP/CP、显存和稳定性诊断。

## 6. 完整项目素材库

### 6.1 verl SFT / RLVR 基础能力

- 新增 Qwen3-8B 8K/32K、Qwen3-30B-A3B 32K RLVR recipes，支持 MoE、expert tensor parallel 和 Megatron 后端。
- 参数化 rule reward、code sandbox reward、LLM judge，修复 reward batch 分流、dataset split 不整除和 final parameter sync 调用问题。
- 完成 DAPO 的 FSDP/Megatron 训练入口和参数配置。
- 适配 SGLang rollout 后端，处理 async mode、multi-stage wakeup、rollout correction、A100 torchao compatibility 和镜像依赖问题。
- 建立同步、fully async 双路径，尽量通过上层参数化和幂等 patch 保持与 verl 上游的升级兼容。

### 6.2 Fully Async RLVR

问题：同步训练约 79% 时间消耗在 rollout；最初把 32 GPU 分为 24 Trainer + 8 Rollouter，造成 rollout 算力不足。

关键判断：

- `3T+1R, gen-TP=4` 只有 2 个 vLLM 实例，吞吐约 76 tokens/s/GPU，trainer idle ratio 0.41。
- 将 gen-TP 从 4 调为 2，vLLM 实例增加到 4，吞吐提升到 211-255 tokens/s/GPU。
- 调整到 `2T+2R` 后 rollout 实例增至 8，吞吐达到 236-293 tokens/s/GPU，瓶颈由 rollout 转移到 actor update。
- 通过 gen_wait、update_actor、ref、param_sync、idle ratio 和 GPU memory 的阶段分解，而不是只看 GPU utilization。

工程修复：

- uneven dataset split；
- reward loop manager 初始化；
- final parameter sync keyword mismatch；
- streaming ref during gen-wait；
- Megatron-Core/MBridge 版本验证与升级；
- CP=1 OOM 后回退 CP=2；
- offload/recompute/token cap 的显存边界。

### 6.3 长上下文 SFT

覆盖素材：

- Qwen3.5-35B-A3B SFT 和 256K recipe；
- Qwen3.5-27B 32K smoke + 128K/64GPU working recipe；
- Qwen3.5-9B 128K/16GPU v1-v3；
- Qwen3-32B 128K/64GPU OOM 和性能诊断；
- context parallel、THD/full recompute、activation/optimizer offload；
- tool-call argument parsing 和长样本 truncation；
- Megatron checkpoint non-tensor args、chat-template fallback 和 tokenizer copy；
- cross entropy fp32 upcast 与 fused CE 显存分析；
- grad_norm spike 时的样本 ID、长度和 batch trace。

典型高级工程判断：

- 9B hidden size 较小时，盲目增加 TP 会缩小 per-rank GEMM 并放大通信，TP=4 是负优化。
- DP=1 时需要重新审视为 DP collective 准备的 fp32 gradient buffer 和 DDP bucket，而不是机械沿用默认值。
- 128K 最大长度不等于平均长度；CP 按最坏长度配置会让典型 47K 样本的通信/计算比恶化。
- “任务跑过若干 step”与“长尾样本稳定运行”是两种不同验收。

### 6.4 AReaL/Agentic RL 框架与性能

框架能力：

- Qwen3.5 dense/MoE/VL 的 Megatron converter、GDN/full attention、MTP、THD packed mode 和 CP 支持。
- XCCL/NCCL weight broadcast、model weight sync、stats tracker 和 staleness manager 稳定性修复。
- Qwen3.5 FLOPs/MFU estimator、training throughput、per-rank peak memory 和 DeepInsight metrics。
- 128K R2E/SWE production recipes、sandbox lifecycle、Fuyao multi-node launcher 和 failure cleanup。
- session-affinity rollout routing、prefix caching、per-request cache hit、per-turn LLM metrics和 episode metrics。
- logp mismatch、weight update re-prefill、policy version 和 importance/rejection diagnostics。

性能结论：

- 正式 32x8 historical baseline 的 logged step mean 为 83.89 min，rollout wait 占 87.27%。
- trajectory elapsed 与最大 request size 的相关系数约 0.845；长上下文增长和 late-turn cost 是主要解释。
- cohort wait-after-7th 的 p95 约 51.5 min，8-way group 的 straggler 放大了单条长轨迹成本。
- 小 batch case 虽然单 step 更短，但 work-normalized goodput 更差，不能把“小 batch 跑得快”写成整体吞吐提升。
- prefix cache 提高可能让 trajectory 更快进入昂贵后期，必须以固定 logical batch 的 overlap-aware update interval 和下游效果验收。

### 6.5 Trajectory Lineage 与样本利用

- 设计 stable logical trajectory ID，支持 microbatch reorder-safe join。
- 记录 manager head drift 和真实 per-token behavior staleness，避免把两者混为同一个 freshness 指标。
- 记录 full-sequence、loss-active 和 policy-gradient-active trajectory/token。
- 将 compact filter、uniform reward、stale、partial cohort、cancelled、waiting 和 final drain 分开归因。
- tracing-off 不访问 token payload，并设置 matched overhead control，防止 tracing 本身污染性能实验。
- 使用 fixed task/seed/checkpoint/manifest 和 artifact SHA 建立可重复 A/B 反馈回路。

### 6.6 OPD / MOPD

OPD 工程能力：

- 同步 on-policy distillation 训练框架；
- REINFORCE-style reverse KL；
- advantage end-to-end wiring 和 monitoring；
- positive-token filter、kept-token normalization；
- Teacher thinking template 一致性；
- dense+sparse advantage，结合 OPD 与 GRPO outcome；
- Level-2 gradient verification 和可证伪实验设计。

MOPD 工程能力：

- trajectory 级 `data_source` 路由和 batch-aligned route field；
- 多 Teacher 配置、按域 scatter/scoring/gather；
- worker-side teacher-score validation 和 all-rank fail-consistent，避免 distributed hang；
- `mopd_pg` actor loss、task advantage 组合和 equal-trajectory weighting；
- HF-native multi-domain mixture、domain preflight 和 deterministic quota planning；
- online reservation domain authority、route conflict fail-fast；
- session/cohort drain、HTTP shutdown contract、runtime provenance；
- checkpoint recovery、StatefulDataLoader 位点恢复；
- MATH500/SWE/TerminalBench held-out paired evaluation 和 bootstrap。

当前诚实边界：

- FUNCTIONAL：已证明单 Teacher、多 Teacher、agentic sandbox、backward、weight sync 和 checkpoint 链路可工作。
- NUMERIC：9B 生产布局 same-weight 数值门尚未完全关闭，历史跨节点 logp 存在差异。
- EFFICACY：多域最终效果仍待严格评测；不能声称 MOPD 已提升模型能力。

## 7. 已合入或有明确工程落点的代表性贡献

对外简历不写 PR 号，面试中可用于证明工程落地。

- AReaL 内部主干：Qwen3.5-35B-A3B 128K Terminal RL 配置与 Fuyao 部署。
- AReaL 内部主干：Qwen3.5 FLOPs、training throughput/MFU 和 DeepInsight metrics。
- AReaL 内部主干：Megatron distributed optimizer checkpoint `flattened_range` crash 修复。
- AReaL 内部主干：Ray RPC 对非 Tensor 返回值的类型处理修复。
- verl 内部主线/分支：Qwen3 RLVR、MoE、fully async、SGLang、OPD、Qwen3.5 SFT 和 checkpoint 稳定性能力。
- MOPD 交付分支：路由、loss、score validation、mixed-domain、online drain、recovery 和 held-out eval 工装。

## 8. 技能栏建议

### 大模型训练与算法

`SFT`、`PPO/GRPO`、`RLVR`、`DAPO`、`On-Policy Distillation`、`MOPD`、rule/model reward、importance sampling、policy staleness、trajectory/cohort sampling。

### 分布式训练

`Megatron-Core`、`MBridge`、`verl`、`AReaL`、`PyTorch`、`Ray`；TP/PP/CP/DP/EP、sequence parallel、packed sequence、dynamic batch、activation recompute、optimizer/parameter/gradient offload、distributed checkpoint。

### 推理与 Agentic RL

`vLLM`、`SGLang`、prefix cache、CUDA Graph、weight sync、OpenAI-compatible serving、tool/sandbox rollout、R2E-Gym、SWE-Bench、TerminalBench。

### 性能与稳定性

MFU/FLOPs、step critical path、GPU memory accounting、NCCL、communication overlap、profile/tracing、SwanLab、DeepInsight、Fuyao multi-node deployment、checkpoint/recovery、failure disposition。

## 9. 面试展开故事

### 故事 A：为什么 async 反而比 sync 慢

从 76 tokens/s/GPU 的 async 初始结果出发，用 rollout producer/trainer consumer 供需模型发现 3T+1R 资源配比与实际 79% rollout 时间相反；通过减小 gen-TP 增加实例数，再调整为 2T+2R，使吞吐超过同步基线并将瓶颈转移到 actor update。

### 故事 B：为什么 prefix cache 命中更高，训练反而可能更慢

命中率只是局部推理指标。cache 加速前期 turn 后，更多 trajectory 进入长上下文昂贵后期，可能增加 episode 长度、cohort straggler 和 trainer exposed wait。最终应以固定 logical batch 的 update interval、参与训练 token 和下游质量判断，而非 cache hit 单点判断。

### 故事 C：性能优化必须守住算法正确性

从 `2000 generated -> 960 consumed` 出发，拒绝把差值 1040 直接判为浪费；建立六层 lineage 后，将 stale、partial、waiting、filter 和 gradient participation 分开，发现一部分 token 虽进入 trainer，却因 final mask 不产生梯度。

### 故事 D：一个失败的 OPD 实验为什么有价值

修复 reverse-KL 和 mode collapse 后，最终 6-benchmark 仍比 SFT 低 1.07pp。通过验证 Teacher 本身弱于 Student，证明继续扩大训练或调 Infra 不可能得到目标收益，从而停止错误方向，并形成 Teacher headroom 和 downstream eval 的前置门禁。

### 故事 E：如何定义 Agentic RL 系统“跑通”

区分 FUNCTIONAL、NUMERIC、EFFICACY：能完成 rollout/backward/checkpoint 不代表 logp 数值一致，更不代表下游效果提升。多 Teacher MOPD 工作以这三层门禁组织代码、实验与交付。

## 10. 当前不能直接写成成果的内容

1. “Prefix Cache 单独提升了 Agentic RL 端到端性能”：当前已验证 prefill 阶段耗时降低 44%，但不能将该局部收益单独外推为 E2E 收益。
2. “cohort recovery 提升了样本利用率或 step time”：P0 tracing 已建立，P1 recovery 效果尚未验证。
3. “MOPD 提升了多域模型能力”：FUNCTIONAL 已过，NUMERIC/EFFICACY 未完全闭环。
4. “Qwen3.5-9B 128K SFT v3 step time 降低 15%-25%”：这是设计预期，不是当前可确认实测结果。
5. “Qwen3-32B 128K 已稳定训练”：只跑通约 50 step，长样本仍触发 OOM。
6. “SWE 100-step 证明模型有效”：历史训练数据存在 mapping-500 污染，只能证明系统稳定。

## 11. 下一轮需要本人确认的数据

按优先级确认，后续逐项补，不需要一次回答完。

### P0：决定简历强度

1. 小鹏正式部门名称和 HR 系统中的职位名称。
2. Fully Async RLVR 最终采用哪一个窗口作为正式收益：`76 -> 211-255`，还是 `sync 200 -> 236-293`；是否有全程平均值。
3. Qwen3.5-9B 128K SFT v3 是否实际跑完、最终 step time/MFU/训练步数和下游结果。
4. 9B/128K Agentic RL 当前最新正式实验是否已经超越 R8b 83.89 min baseline。
5. MOPD 在 2026-08-05 之后的双 Teacher 30/100-step 与 held-out eval 最终结论。

### P1：增强业务影响

1. 这些训练能力服务了多少模型、算法同事或业务项目。
2. 你在小鹏是 owner、核心开发还是多人协作；是否带人或负责项目排期。
3. 合入公司主干的 PR 数、代码评审责任和发布版本。
4. 节省的 GPU-hours、实验周转时间或故障恢复时间。
5. 哪些能力已经成为团队默认 recipe、平台能力或生产基线。

## 12. 对旧简历整体结构的初步建议

旧简历当前把“工作经历”和“项目经历”分开，华为内容重复较多。后续正式改版建议：

1. 保留教育经历，但压缩荣誉和基础技能。
2. 工作经历按“公司 + 职位 + 4-5 条成果”展开，不再另设重复项目经历。
3. 小鹏经历放在最前，占整页约 40%-50%，华为压缩为 3-4 条。
4. 科研经历压缩为 2 行，保留 MICCAI 第一作者、AAAI 共同一作。
5. 删除年龄、性别、民族、籍贯等对技术社招没有帮助的信息。
6. 技能栏从“掌握什么”改成“训练算法、分布式系统、推理/Agentic、性能工具”四组关键词。
7. 最终控制在 1-2 页；高级工程师版本优先 2 页，确保每个强项目有可面试展开空间。

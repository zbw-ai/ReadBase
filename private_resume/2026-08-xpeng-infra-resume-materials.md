# 小鹏大模型训练推理 Infra 简历素材底稿

更新时间：2026-09-03
目标岗位：外部社招，大模型训练 / RL / Agentic RL Infra 高级工程师
状态：已按最新版 2026 简历与本人确认口径校准；本轮不修改简历 DOCX

## 0. 使用原则

1. 对外只写开源技术底座 `verl`、`AReaL`，不写内部仓库名 `llm_train`、`Trail`。
2. 简历正文突出问题、技术决策和结果，不罗列分支、脚本和 commit。
3. `已验证` 的数字可以进入简历；`实验性` 内容只能描述为“设计、实现、打通、定位”，不能写成性能或模型效果收益。
4. 性能收益必须和固定 workload、硬件规模、统计窗口绑定；算法收益必须有 checkpoint 下游评测。
5. 双 Teacher MOPD 可表述为“SWE、Terminal 双域提升且 General 不下降”；统计细节未补齐前不追加“显著提升”或具体 pp。Prefix Cache 只写已验证的 prefill 局部收益，不外推端到端。
6. Megatron-Core 个人边界位于 feature integration/application layer：使用和集成 5D 并行、训练后端与相关特性，不声称实现 collective kernel、修改 `parallel_state`/process-group construction 或编写 pipeline scheduler。

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

最新版投递简历的规模边界：TX、X1 项目所在集群总规模分别约 `1.4 万卡`、`1.2 万卡`；个人直接模型训练证据是 X1 200B MoE 的 `3K 卡稳定训练两个月`。TX 可口述文生视频、文生图和 389B MoE 的迁移/功能/性能闭环，以及开局性能提升 `30%–50%`、10+ 模型交付、80+ 生产问题、协同 4–5 人；必须区分个人动作与团队总结果。

### 小鹏机器人

小鹏机器人 - 大模型训练推理 Infra 高级工程师
`2025.11 - 至今`

建议部门总述：

> 负责大模型后训练基础设施研发，基于 verl、Megatron-Core 和 AReaL 二次开发 SFT、RLVR 与 Agentic RL 训练系统，重点建设长上下文训练、异步 rollout、训练推理解耦、在线蒸馏、性能可观测性和轨迹数据治理能力，支持 Qwen3/Qwen3.5 dense/MoE 模型在多机 A100 集群上的训练与实验验证。

## 3. 推荐对外写法 v2

参考“项目标题 + 职责概述 + 粗体能力标签 + 技术机制 + 量化结果”的表达方式。对外不出现内部仓库名，只写开源技术底座和个人承担的工作。

### 基于 verl 的 SFT / RLVR 训练系统

负责基于 verl、Megatron-Core/MBridge 二次开发大模型 SFT/RLVR 训练链路，支持 Qwen3/Qwen3.5 dense、MoE、32K-256K 长上下文和多机 A100 训练，主要工作如下：

- **异步 RLVR 吞吐优化**：面向 Qwen3-30B-A3B 32K、32 张 A100-80GB 场景适配 fully async policy，解耦 Trainer 与 Rollouter；基于 rollout 生产率和 trainer 消费率调整 gen-TP、vLLM 实例数及 3T+1R/2T+2R 资源配比，将异步初始吞吐由 76 提升至 211-255 tokens/s/GPU；2T+2R 候选窗口达到 236-293 tokens/s/GPU，trainer idle ratio 由 0.41 降至 0.10-0.14。同步约 200 仅作早期阶段诊断背景，对照协议补齐前不声明 async 相比 sync 的提升比例。
- **长上下文 SFT 优化**：完成 Qwen3.5 9B/27B/32B 等模型 128K SFT 训练适配，覆盖 16-64 张 A100；通过 `num_workers=0→8` 与 data prefetch、selective recompute 和 TP/CP 收敛，将代表性 9B SFT step time 由 31s 降至 9.3s、MFU 由 23% 提升到 45.2%。该结果是联合优化，当前不分摊单项收益；两组比值尚不能按标准 MFU 公式直接闭合，面试前需补 estimator、effective-token 和统计窗口。另一个长上下文 workload 中，`TP=2/CP=8` 相比 `TP=4/CP=4` 将 step time 由约 163s 降至 102s，两组数字不混用。
- **长上下文 MoE 与显存优化**：在 Qwen3.5-35B-A3B 128K 场景中将平均 step time 降低约 50%；修复 THD+CP actor 路径重复 all-gather full-sequence logits 的静默问题，保持 `[T/CP,V/TP]` logits 本地计算、只聚合 token scalar，消除约 7.6GB 冗余分配。前者是联合结果，当前不拆分单项贡献；后者有提交 `be6fb98f` 的源码证据。
- **训练后端与稳定性**：接入 vLLM/SGLang rollout 后端，支持 MoE、DAPO、rule/code/model reward 和异步 parameter sync；在独立的 verl 35B RLVR workload 中，CUDA Graph 将 decode 阶段加速约 14x；修复 uneven dataset split、reward-loop 初始化、final parameter sync、Megatron distributed checkpoint 等问题，建设 Qwen3.5 FLOPs/MFU、阶段耗时和显存监控，为多机训练提供可复现 recipe、失败诊断和恢复能力。

### 基于 AReaL 的 Agentic RL 与在线蒸馏

负责 Qwen3.5-9B 128K 长上下文与多轮 Agentic RL 训练链路建设和性能优化，覆盖 online rollout、tool/sandbox environment、训练推理异步调度及在线蒸馏，主要工作如下：

- **长上下文与端到端性能**：面向 DeepSWE 与 Seta Terminal 多轮环境交互场景，交付 Qwen3.5-9B 128K Agentic RL 训练链路；DeepSWE 端到端稳态单步耗时由 6467s 降至 2301s（-64.4%），有效 Token 吞吐达到 146.8 tok/s/GPU；Seta Terminal 由 2240s 降至 770s（-65.6%），有效 Token 吞吐达到 233 tok/s/GPU。
- **Rollout 与调度优化**：在 AReaL Qwen3.5-9B 128K Agentic RL workload 中通过 CUDA Graph 将 decode 阶段加速 6–8x；Prefix Cache 优化使 prefill 阶段耗时降低 44%，并优化 Sandbox 并发以提升 vLLM 有效并发；重构 Rollout 调度链路，通过 Gateway 实现流式补位、均衡分发与失败请求管理，使 Rollout 阶段平均推理吞吐提升 60%，Rejected Group 比例由 33.18% 降至 2.73%（-30.45pp）。各数字分别限定在 decode、prefill 和 Rollout 阶段，不外推为端到端同倍数加速；verl 35B RLVR decode 约 14x 是另一套独立 workload。
- **多 Teacher 在线蒸馏**：设计并实现 On-Policy Distillation/MOPD，支持 trajectory 按 data source 路由至对应 Teacher 计算 logp，以及 teacher score 校验、`mopd_pg` loss、mixed-domain data、equal-trajectory weighting、online session drain、断点续训和 held-out paired evaluation；采用 FUNCTIONAL/NUMERIC/EFFICACY 分层门禁，最新版双 Teacher 结果在 SWE、Terminal 双域提升且 General 不下降。具体 checkpoint、样本量、seed、baseline、评测窗口和统计置信信息仍需随面试证据卡携带。

建议最终简历正文优先采用以上 6 条。篇幅不足时，先合并“训练后端与稳定性”到前两条，再视目标岗位删除长上下文 SFT 或多 Teacher 蒸馏中的一条。

## 4. 事实展开与证据边界

以下 5 条保留更完整的证据说明，供修改简历和准备面试使用。

### 4.1 后训练框架建设

基于 `verl + Megatron-Core/MBridge` 二次开发统一 SFT/RLVR 训练链路，支持 Qwen3/Qwen3.5 dense、MoE 模型及 32K-256K 长上下文场景，完善 TP/PP/CP/DP/EP、packed sequence、dynamic batch、recompute、offload、checkpoint 和 rule/model reward 配置，并接入 vLLM、SGLang rollout 后端及多机 Fuyao 任务部署。

个人边界：工作重点是 Megatron-Core/MBridge 的 feature 使用、训练后端集成、配置与拓扑推理、性能和正确性排障；没有实现底层 collective kernel，没有修改 `parallel_state`/process-group construction，也没有编写 pipeline scheduler。

证据等级：`可写`。其中“32K-256K”表示覆盖过的工程配置范围，不等于所有模型和长度均已形成稳定生产基线。

### 4.2 Fully Async RLVR 性能优化

面向 Qwen3-30B-A3B 32K、32 张 A100-80GB 场景，完成 verl fully async policy 适配与训练/rollout 资源解耦。同步阶段拆解显示约 79% 时间消耗在 rollout，说明存在通过 producer-consumer overlap、独立扩缩容和长尾解耦减少 exposed wait 的空间；这不代表单条 trajectory latency 自动降低。

初始 async `3T+1R、gen-TP=4、2 个 vLLM 实例` 只有 `76 tokens/s/GPU`、trainer idle ratio `0.41`。将 gen-TP 降至 2、实例增至 4，并联合配置 `require_batches`/trigger、`free_cache_engine`、dynamic batch、chunked prefill、prefix cache、CUDA Graph path、partial rollout、bounded staleness、rollout correction、validation frequency 和 serving limits 后，async 内部吞吐达到 `211-255 tokens/s/GPU`；其中缺少独立同-workload A/B 的能力不单独分摊收益。`2T+2R、8 个实例` 候选窗口观测到 `236-293 tokens/s/GPU`，idle ratio 降至 `0.10-0.14`，瓶颈迁移到 actor update。

证据等级：`强可写`。面试时必须说明这些是代表性稳态 step 区间，不包装成全程平均值；验证硬件为 `4 节点 x 8 A100-80GB`。同步约 `200 tokens/s/GPU` 只作为初始诊断背景；相同 workload、统计窗口、warmup/异常步处理和分母补齐前，不声明 Fully Async 相比同步的提升比例。

### 4.3 长上下文 SFT 与显存/稳定性分析

负责 Qwen3.5 长上下文 SFT 适配和性能诊断，在 16-64 张 A100 上打通 9B/27B/32B 等模型的 128K 训练路径。对代表性 9B SFT workload，通过 `num_workers=0→8` 与 data prefetch 消除输入 bubble，从偏重 full recompute 收敛到 selective recompute，并调整 TP/CP 以避免 TP 切碎 GEMM、用 CP 分摊长序列 activation，最终将 step time 从 `31s` 降至 `9.3s`、MFU 从 `23%` 提升至 `45.2%`。这是最新版简历联合结果，没有逐项同-workload A/B，不拆分虚构收益。由于 `31/9.3` 与 `45.2/23` 在标准 MFU 定义下不能自动闭合，面试前必须补齐 MFU estimator、有效 token、是否计入 data wait 和统计窗口；补齐前不宣称两组数字来自同一单一测量窗口。

Qwen3.5-35B-A3B 128K 场景的最新版结果是平均 step time 降低约 `50%`。可确认的代码级机制包括修复 THD+CP actor 路径的 full-sequence logits all-gather：保留 CP-local、TP-vocab-sharded logits，本地计算 logprob/entropy 后只 gather token scalar，消除约 `7.6GB` 冗余分配。其余 TP/CP/EP、packing、recompute、fusion 和 overlap 属于联合优化路径；没有逐项 A/B 时不分摊 50%。

另一长上下文 workload 基于张量级显存账定位 fp32 logits、gradient buffer、offload PCIe、CP 通信和长样本激活峰值，验证 `TP=2, CP=8` 相比 `TP=4, CP=4` 将 step time 从约 `163s` 降至 `102s`。该证据只用于说明并行策略机制，不与 31s→9.3s 合并。

checkpoint 交付定义：训练框架和 recipe 达到稳定训练验收，可支持算法团队持续实验并产出经下游验证的有效模型权重；它强于 smoke test，但不自动等于无限期、无人值守长稳。需要用代表性长度分布、连续训练窗口、loss/grad、save/resume、下游质量和 recipe 可复现证明。

证据等级：`可写但需精修`。9B 对比数字来自 v1/v2 实验记录；32B 128K 跑通约 50 step 后仍因长样本 OOM，不能写成稳定训练交付；已声明交付的 35B-MoE 256K、27B 128K/256K checkpoint 按上述验收定义口述。

### 4.4 128K Agentic RL 性能与轨迹分析

面向 Qwen3.5-9B、128K、32 张 A100 的 R2E/SWE Agentic RL 场景，建立 overlap-aware step、rollout supply、cohort tail、cache、token participation 和 trajectory lineage 性能体系；量化历史基线 step 均值 `83.89 min`，其中 rollout wait `73.21 min / 87.27%`，进一步定位长上下文后期 LLM 调用、8-way cohort straggler 和轨迹供给不足为一阶瓶颈，而非仅以 prefix-cache 命中率判断性能。

性能闭环结果：

- DeepSWE 场景端到端稳态单步耗时由 `6467s` 降至 `2301s`（`-64.4%`），有效 Token 吞吐达到 `146.8 tok/s/GPU`。
- Seta Terminal 场景端到端稳态单步耗时由 `2240s` 降至 `770s`（`-65.6%`），有效 Token 吞吐达到 `233 tok/s/GPU`。
- 在 AReaL Qwen3.5-9B 128K Agentic RL workload 中，CUDA Graph 将 `decode` 阶段加速 `6–8x`；Prefix Cache 使 `prefill` 阶段耗时降低 `44%`；Sandbox 并发优化提升 vLLM 有效并发。verl 35B RLVR 的 decode 约 `14x` 是独立 workload，不混用。
- 重构 Rollout 调度链路，通过 Gateway 实现流式补位、均衡分发和失败请求管理，使 vLLM 并发稳定在理论值附近，Rollout 阶段平均推理吞吐提升 `60%`，Rejected Group 比例由 `33.18%` 降至 `2.73%`（`-30.45pp`）。

补充技术动作：

- 打通 generated -> manager -> workflow -> trainer -> loss -> policy gradient 六层逐轨迹 join。
- 在真实 6-step tracing run 中闭环 `223 admitted -> 180 generated/rewarded -> 96 exported -> 96 consumed`，96 条全部 exact join。
- 识别 94 条 gradient-active、2 条 compact-filtered；后者消耗 `159,330` full-sequence tokens，占该窗口 trainer token processing 的 `3.91%`，但不产生梯度。
- 将 partial、stale、waiting/final-drain 与 terminal waste 分开，避免把 `generated - consumed` 直接判为浪费。

证据等级：`强可写`。端到端结果统一表述为“稳态单步耗时”；AReaL 的 CUDA Graph 主口径使用 `6–8x`，verl 35B RLVR 的约 `14x` 仅在明确独立 workload 时补充。CUDA Graph、Prefix Cache 和 Rollout 平均推理吞吐必须分别限定为 decode、prefill 和 Rollout 阶段收益，不能单独外推为端到端加速。有效 Token 吞吐的面试口径固定为“实际参与训练的有效 Token / GPU / 端到端稳态时间”。

### 4.5 OPD/MOPD 算法与 Infra 联合建设

设计并实现 on-policy distillation 与多 Teacher MOPD 训练链路，覆盖 trajectory `data_source` 路由、Teacher scatter/gather、teacher-logp 校验、`mopd_pg` loss、mixed-domain data、equal-trajectory weighting、在线 session drain、断点续训和 held-out paired evaluation；形成“FUNCTIONAL、NUMERIC、EFFICACY”分层验收机制。最新版双 Teacher MOPD 结果在 SWE、Terminal 双域提升且 General 不下降。

算法判断素材：早期 OPD 实验修复 reverse-KL advantage 缺失、token filter、kept-token normalization 和 thinking template 等问题，将 6-benchmark 能力退化从约 `10-20pp` 收敛到 `1.07pp`；最终通过 SFT/Teacher 能力对比确认 Teacher 弱于 Student，及时否决“继续调 Infra 即可获得模型提升”的错误假设，并将后续工作转向更强 Teacher、GRPO 混合目标和严格下游评测。

证据等级：`可写但必须带边界`。允许口述的效果结论只有“SWE、Terminal 双域提升且 General 不下降”；checkpoint、样本量、seed、baseline、评测窗口和置信信息补齐前，不追加“显著提升”或双 Teacher 具体 pp。单 Teacher `Terminal +7.9pp`、`SWE +7.0pp` 不能外推到双 Teacher；SWE 100-step 历史污染 run 只作为系统稳定性证据，不作为最终效果证据。

## 5. 推荐的简历压缩版本

如果小鹏经历只能保留 4 条，建议使用以下组合：

1. **框架与模型覆盖**：基于 verl/Megatron-Core 和 AReaL 建设 SFT、RLVR、Agentic RL 后训练系统，支持 Qwen3/Qwen3.5 dense/MoE、32K-256K、多机 A100 和 vLLM/SGLang。
2. **性能硬结果**：32 A100 上将 Qwen3-30B-A3B 32K fully async RLVR 吞吐从 76 提升至 211-255 tokens/s/GPU；2T+2R 候选窗口观测到 236-293 tokens/s/GPU，idle ratio 降至 0.10-0.14。
3. **Agentic RL 性能优化**：DeepSWE 端到端稳态单步耗时由 6467s 降至 2301s（-64.4%），有效 Token 吞吐达到 146.8 tok/s/GPU；Seta Terminal 由 2240s 降至 770s（-65.6%），有效 Token 吞吐达到 233 tok/s/GPU；Gateway 调度重构使 Rollout 阶段平均推理吞吐提升 60%，Rejected Group 比例由 33.18% 降至 2.73%。
4. **算法-Infra 联合能力**：实现 OPD/MOPD 多 Teacher 路由、打分、loss、混域训练、恢复和评测链路，以 FUNCTIONAL/NUMERIC/EFFICACY 分层门禁验收；最新版双 Teacher 结果在 SWE、Terminal 双域提升且 General 不下降。

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
- 调整到 `2T+2R` 后 rollout 实例增至 8，候选窗口观测到 236-293 tokens/s/GPU，瓶颈由 rollout 转移到 actor update。
- 通过 gen_wait、update_actor、ref、param_sync、idle ratio 和 GPU memory 的阶段分解，而不是只看 GPU utilization。
- 结合 `require_batches`/trigger、`free_cache_engine`、dynamic batch、chunked prefill、prefix cache、CUDA Graph path、partial rollout、bounded staleness、rollout correction、validation frequency 和 serving limits 稳定生产/消费；缺少独立 A/B 时不拆分各项收益。
- sync 约 200 的 workload/统计窗口/分母未完全对齐，不能写成 async 已超过 sync，更不能用 76→211-255 代表 sync→async 提升。

工程修复：

- uneven dataset split；
- reward loop manager 初始化；
- final parameter sync keyword mismatch；
- streaming ref during gen-wait；
- Megatron-Core/MBridge 版本验证与升级；
- CP=1 OOM 后回退 CP=2；
- offload/recompute/token cap 的显存边界。

### 6.3 长上下文 SFT

代表性性能结果：通过 `num_workers=0→8` 与 prefetch、selective recompute、TP/CP 收敛，最新版简历记录 step time 从 `31s` 降到 `9.3s`、MFU 从 `23%` 提升到 `45.2%`。这是联合结果，不拆分没有 A/B 的单项贡献；两组数字在 MFU estimator、effective-token 和计时窗口核对前不声明为完全相同的单一测量窗口。`TP=4/CP=4 → TP=2/CP=8、163s→102s` 属于另一 workload。

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
- “交付 checkpoint”指 recipe 达到稳定训练验收，可支持算法团队实验并产出经下游验证的有效权重；这强于跑过若干 step，但仍不等于无限期无人值守长稳。

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
- NUMERIC：以 token/logp/mask/loss、same-weight 和跨 rank fail-consistent 为持续回归门禁；历史跨节点 logp 差异不作为效果结论。
- EFFICACY：最新版双 Teacher MOPD 在 SWE、Terminal 双域提升且 General 不下降。统计细节补齐前不声称“显著提升”，也不把单 Teacher `7.9pp/7.0pp` 当成双 Teacher 数字。

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

口述边界：Megatron-Core 以 5D 配置、特性使用、训练后端集成和性能/正确性排障为主；没有实现 collective kernel、修改 `parallel_state`/process-group construction 或编写 pipeline scheduler。

### 推理与 Agentic RL

`vLLM`、`SGLang`、prefix cache、CUDA Graph、weight sync、OpenAI-compatible serving、tool/sandbox rollout、R2E-Gym、SWE-Bench、TerminalBench。

### 性能与稳定性

MFU/FLOPs、step critical path、GPU memory accounting、NCCL、communication overlap、profile/tracing、SwanLab、DeepInsight、Fuyao multi-node deployment、checkpoint/recovery、failure disposition。

## 9. 面试展开故事

### 故事 A：为什么 async 反而比 sync 慢

从 76 tokens/s/GPU 的 async 初始结果出发，用 rollout producer/trainer consumer 供需模型发现 3T+1R 资源配比与实际 79% rollout 时间相反；通过减小 gen-TP、增加实例数和稳定 batch supply，再尝试 2T+2R，使 async 内部吞吐提高并将瓶颈转移到 actor update。同步对照协议未闭环前不报 async 相对提升倍数。

### 故事 B：为什么 prefix cache 命中更高，训练反而可能更慢

命中率只是局部推理指标。cache 加速前期 turn 后，更多 trajectory 进入长上下文昂贵后期，可能增加 episode 长度、cohort straggler 和 trainer exposed wait。最终应以固定 logical batch 的 update interval、参与训练 token 和下游质量判断，而非 cache hit 单点判断。

### 故事 C：性能优化必须守住算法正确性

从 `2000 generated -> 960 consumed` 出发，拒绝把差值 1040 直接判为浪费；建立六层 lineage 后，将 stale、partial、waiting、filter 和 gradient participation 分开，发现一部分 token 虽进入 trainer，却因 final mask 不产生梯度。

### 故事 D：一个失败的 OPD 实验为什么有价值

修复 reverse-KL 和 mode collapse 后，最终 6-benchmark 仍比 SFT 低 1.07pp。通过验证 Teacher 本身弱于 Student，证明继续扩大训练或调 Infra 不可能得到目标收益，从而停止错误方向，并形成 Teacher headroom 和 downstream eval 的前置门禁。

### 故事 E：如何定义 Agentic RL 系统“跑通”

区分 FUNCTIONAL、NUMERIC、EFFICACY：能完成 rollout/backward/checkpoint 不代表 logp 数值一致，数值一致也不替代 held-out 下游效果。多 Teacher MOPD 以三层门禁组织代码、实验与交付；最新版 EFFICACY 结论为 SWE、Terminal 双域提升且 General 不下降。

## 10. 当前不能直接写成成果的内容

1. “Prefix Cache 单独提升了 Agentic RL 端到端性能”：当前已验证 prefill 阶段耗时降低 44%，但不能将该局部收益单独外推为 E2E 收益。
2. “cohort recovery 提升了样本利用率或 step time”：P0 tracing 已建立，P1 recovery 效果尚未验证。
3. “双 Teacher MOPD 在 SWE/Terminal 显著提升 X pp”：当前只确认双域提升且 General 不下降；显著性与具体双 Teacher pp 没有随证据卡补齐前不能写。
4. “Qwen3.5-9B 128K SFT v3 step time 降低 15%-25%”：这是设计预期，不是当前可确认实测结果。
5. “Qwen3-32B 128K 已稳定训练”：只跑通约 50 step，长样本仍触发 OOM。
6. “SWE 100-step 证明模型有效”：历史训练数据存在 mapping-500 污染，只能证明系统稳定。

## 11. 下一轮需要本人确认的数据

按优先级确认，后续逐项补，不需要一次回答完。

### P0：决定简历强度

1. 小鹏正式部门名称和 HR 系统中的职位名称。
2. Fully Async RLVR 的同步对照是否已补齐相同 workload、统计窗口、warmup/异常步处理和 `tokens/s/GPU` 分母；未补齐前正式收益只用 async 内部 `76 -> 211-255`。
3. Qwen3.5-9B SFT `31s→9.3s` 的硬件、sequence/packing、GBS/MBS、有效 token、统计窗口，以及是否存在逐项同-workload A/B。
4. 9B/128K Agentic RL 当前最新正式实验是否已经超越 R8b 83.89 min baseline。
5. MOPD 双 Teacher held-out eval 的 checkpoint、样本量、seed、baseline、评测窗口和置信信息；最终方向性结论已确认为 SWE、Terminal 双域提升且 General 不下降。

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

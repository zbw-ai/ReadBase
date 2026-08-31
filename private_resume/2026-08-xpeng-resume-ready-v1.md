# 小鹏经历 - 简历正文候选 v1

## 小鹏机器人｜大模型训练推理 Infra 高级工程师｜2025.11 - 至今

负责大模型后训练基础设施研发，基于 verl、Megatron-Core/MBridge 和 AReaL 二次开发 SFT、RLVR 与 Agentic RL 训练系统，支持 Qwen3/Qwen3.5 dense、MoE 模型及多机 A100 长上下文训练，重点解决训练推理解耦、rollout 性能、轨迹利用和在线蒸馏问题。

### 大模型 SFT / RLVR 训练系统（基于 verl）

- **异步 RLVR 吞吐优化**：面向 Qwen3-30B-A3B 32K、32 张 A100-80GB 场景适配 fully async policy，解耦 Trainer 与 Rollouter；基于 rollout 生产率和 trainer 消费率调整 gen-TP、vLLM 实例数及 3T+1R/2T+2R 资源配比，将代表性稳态吞吐由 76 提升至 211-255 tokens/s/GPU，超过约 200 tokens/s/GPU 的同步基线；2T+2R 配置进一步达到 236-293 tokens/s/GPU，trainer idle ratio 由 0.41 降至 0.10-0.14。
- **长上下文 SFT 优化**：完成 Qwen3.5 9B/27B/32B 等模型 128K SFT 训练适配与性能诊断，覆盖 16-64 张 A100；基于张量级显存账定位 fp32 logits、gradient buffer、offload PCIe、CP 通信和长样本激活峰值，验证 9B 场景 TP=2/CP=8 相比 TP=4/CP=4 将 step time 由约 163s 降至 102s，并建设 grad-norm spike 的 batch 级 tracing 与超长样本检查能力。
- **训练后端与稳定性**：接入 vLLM/SGLang rollout 后端，支持 MoE、DAPO、rule/code/model reward 和异步 parameter sync；修复 uneven dataset split、reward-loop 初始化、final parameter sync、checkpoint deadlock 和 Megatron distributed checkpoint 等问题，建设 Qwen3.5 FLOPs/MFU、阶段耗时及显存监控，形成可复现 recipe、故障诊断与恢复链路。

### 长上下文 Agentic RL 系统（基于 AReaL）

- **128K Rollout 性能诊断**：面向 Qwen3.5-9B、32 张 A100、32 cohorts x 8 trajectories 的 R2E/SWE 场景，建立 overlap-aware step、rollout supply、cohort tail、prefix cache 和 per-turn LLM 指标体系；量化历史基线 step 均值 83.89 min，其中 rollout wait 为 73.21 min、占 87.27%，结合 cohort wait-after-7th p95 约 51.5 min，定位长上下文后期推理和 8-way cohort straggler 为一阶瓶颈。
- **轨迹利用与算法正确性**：打通 generated -> manager -> workflow -> trainer -> loss -> policy gradient 六层逐轨迹 lineage，在真实 6-step run 中闭环 223 admitted -> 180 generated/rewarded -> 96 exported -> 96 consumed，96 条全部 exact join；识别 94 条 gradient-active 和 2 条 compact-filtered trajectory，后者消耗 159,330 full-sequence tokens、占 trainer token processing 的 3.91% 但不产生梯度，并将 stale、partial、waiting/final-drain 与 terminal waste 分开归因。
- **多 Teacher 在线蒸馏**：设计并实现 On-Policy Distillation/MOPD，支持 trajectory 按 data source 路由至对应 Teacher 计算 logp，以及 teacher-score 校验、mopd_pg loss、mixed-domain data、equal-trajectory weighting、online session drain、断点续训和 held-out paired evaluation；完成 9B Terminal 单 Teacher 30-step 与双 Teacher early canary，建立 FUNCTIONAL/NUMERIC/EFFICACY 分层验收机制。

# 小鹏经历 - 简历正文候选 v2（精简版）

## 小鹏机器人｜大模型训练推理 Infra 高级工程师｜2025.11 - 至今

负责基于 verl、Megatron-Core 和 AReaL 的大模型后训练 Infra 建设，聚焦长上下文 SFT/RLVR、异步 rollout、Agentic RL 性能及在线蒸馏。

- **异步 RLVR 吞吐优化**：面向 Qwen3-30B-A3B 32K、32 张 A100-80GB 场景适配 fully async 训练，基于 rollout 生产率与 trainer 消费率重构训练/推理资源配比（gen-TP 4->2、3T+1R->2T+2R），将吞吐由 76 提升至 211-255 tokens/s/GPU；2T+2R 配置达到 236-293 tokens/s/GPU，trainer idle ratio 由 0.41 降至 0.10-0.14。
- **128K SFT 性能优化**：完成 Qwen3.5 9B/27B/32B 在 16-64 张 A100 上的 128K SFT 适配；通过张量级显存建模定位 fp32 logits、gradient buffer、CP 通信和 offload 开销，9B 场景采用 TP=2/CP=8，相比 TP=4/CP=4 将 step time 由约 163s 降至 102s，并补齐长样本 OOM、grad-norm spike 与 checkpoint deadlock 诊断能力。
- **Agentic RL Rollout 性能分析**：面向 Qwen3.5-9B 128K、32 cohorts x 8 trajectories、32 张 A100 场景，建立 overlap-aware step、cohort tail、prefix cache 与 per-turn LLM 指标体系；量化基线 step 均值 83.89 min，其中 rollout wait 占 87.27%，定位长上下文后期推理与 8-way cohort straggler 为一阶瓶颈。
- **轨迹利用与算法正确性**：打通 generated -> manager -> workflow -> trainer -> loss -> policy gradient 六层 lineage；在真实 6-step run 中实现 96 条训练轨迹 100% 精确关联，识别 94 条 gradient-active 轨迹及 2 条无梯度贡献轨迹，后者消耗 3.91% 的 trainer full-sequence token 计算量，并将 stale、partial、waiting 与 terminal waste 分开归因。
- **多 Teacher 在线蒸馏**：设计并实现 On-Policy Distillation/MOPD，支持 trajectory 按数据域路由至对应 Teacher 计算 logp，完成 teacher-score 校验、mopd_pg loss、混域训练、session drain、断点续训及 held-out paired evaluation；跑通 9B Terminal 单 Teacher 30-step 与双 Teacher canary，并建立 FUNCTIONAL/NUMERIC/EFFICACY 分层验收机制。

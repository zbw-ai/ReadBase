# 推理基础设施核心学习路线

[所属 Part III：推理基础设施](README.md) · [首页](../README.md)

## 学习目的

独立理解在线推理服务如何把请求变成 GPU 工作，如何在质量、延迟、吞吐、容量与成本之间做取舍。
rollout 只是推理的一种下游 workload；这里的验收单位是请求与服务目标，不以 trainer 是否缺数据替代 serving 评价。
当前尚无独立推理 topic，主线明确列出局部可复用内容和仍需补齐的正文。

## 先修与起点

- 理解自回归生成、Attention shape 和基本 tensor layout，可从 [Transformer 解读](../research/papers/transformer.md)补起。
- 完成 [GPU Systems](../01-systems/roadmaps/gpu-systems.md)的执行 / 内存 / 测量基础，能区分设备执行与 Host 提交。
- 学过 [分布式系统路线](../01-systems/roadmaps/distributed-systems.md)的 timeout、背压与版本准入，有助于后半段服务设计。

## 五阶段主线

“局部覆盖”不等于已有完整 serving 手册；“待补”表示学习问题尚无成章材料，不创建占位 topic。

| 阶段 | 核心问题 | 已有正文与覆盖状态 | 验证 / 掌握标准 |
|---|---|---|---|
| 1. Prefill / decode 成本 | 一个请求的排队、首 token 与后续 token 成本分别来自哪里？ | [Roofline](../01-systems/topics/transformer_engine.md#roofline)和 [decode 提交路径](../04-rl-infra/topics/agentic_rl.md#cuda-graph-decode)局部覆盖；独立成本模型待补 | 画请求时间线，区分 TTFT、TPOT、端到端延迟与有效吞吐；给定输入 / 输出长度分布，解释为何平均 token/s 不够 |
| 2. KV、分页与前缀复用 | KV 随什么增长，谁拥有 block，何时可共享 / 回收？ | [decode 与动态 KV](../04-rl-infra/topics/agentic_rl.md#cuda-graph-decode)及 [后端选型](../04-rl-infra/topics/rl_framework_selection.md#vllm-sglang-selection)只作场景入口；paged KV / prefix cache 正文待补 | 按模型 KV 结构、长度、dtype 与并发建账，追踪逻辑序列到物理 block；定义命中、失效、隔离与取消后的回收条件 |
| 3. Batching 与 chunked prefill | 怎样同时接纳新请求又不让长 prefill 拖坏 decode 尾延迟？ | [供给与执行容量](../04-rl-infra/topics/agentic_rl.md#gateway-streaming-refill)仅覆盖 rollout；continuous batching / chunked prefill 正文待补 | 给混合长短请求画调度时间线，说明 token budget、等待、公平性与抢占代价；不能只按并发数选配置 |
| 4. Kernel、Graph 与量化 | 哪些优化减少 IO，哪些减少提交，哪些改变数值与内存？ | [融合地图](../01-systems/topics/transformer_engine.md#fusion-map)、[compile / Graph](../01-systems/topics/transformer_engine.md#torch-compile)、[FP8 验证](../01-systems/topics/fp8.md#precision-validation)局部覆盖；推理量化专项待补 | 固定 workload 做单变量对照，确认实际 kernel / fallback；同时检查 prefill、decode、显存和输出质量，不能从训练精度直接推出推理量化结论 |
| 5. SLO 与分布式服务 | 在满足延迟和质量门槛时能承载多少请求？多副本 / TP / 跨机怎样选？ | [NCCL](../01-systems/topics/nccl.md)、[版本准入](../04-rl-infra/topics/agentic_rl.md#rl-state-boundaries)提供底层机制；SLO、容量规划、分布式 serving 正文待补 | 固定到达率和长度分布，报告达标吞吐、尾延迟与资源成本；设计限流、隔离、发布 / 回滚，说明新增分布式状态与故障域 |

先建立单副本的请求与 KV 账本，再讨论多副本、模型并行或 prefill / decode 分离。后端选型不能先于 workload 定义。

## 最小练习与产物

统一入口为 [E07 推理并发、KV 与 CUDA Graph](../practice/experiments/a100_fsdp_io_lab.md#e07)，目前未执行；下面是缩小后的学习步骤。

1. **先写 workload 卡。**

   固定模型 / revision、tokenizer、输入输出长度、采样参数、请求到达方式与统计窗口。
   写清 TTFT / TPOT 的计时定义和服务目标；不要把闭环固定并发结果当成任意线上流量容量。

2. **再写 KV 容量卡。**

   估算不同上下文与并发下的 KV 需求，区分逻辑 token、已分配 block、缓存保留与实际显存。
   对共享前缀设计命中 / 未命中两组输入；记录缓存条件，不把热缓存收益混成普遍加速。

3. **画混合请求调度。**

   放入长 prefill、短交互和持续 decode，分别推演整段 prefill 与 chunked prefill 的排队影响。
   产物是时间线和可证伪假设；没有可运行 backend 时只标“设计”，不填性能数字。

4. **只改变一项执行优化。**

   从 Graph、attention backend 或一种获支持的量化路径中选一项；量化还需单独的质量基线。
   区分 warmup / capture / steady-state、kernel 时间和服务时间；先验证输出与 fallback，再解释加速来源。

5. **最后画容量边界。**

   在同一请求分布下改变负载，记录排队、拒绝、成功且达标的请求和资源占用。
   将限流与取消后的 KV / slot 回收纳入正确性；达到吞吐峰值不代表仍满足 SLO。

## 执行与覆盖边界

- A100 E00–E08 全部未执行；本路线没有启动推理服务、GPU benchmark 或线上压测。
- E07 是实验设计，不是现成脚本；执行前需补实际后端版本、预算、计时协议、正确性门槛与停止条件。
- A100 不提供原生 FP8 / FP4 Tensor Core 验证条件；模型支持某 dtype 也不证明目标 kernel 生效。
- [vLLM / SGLang 选型章节](../04-rl-infra/topics/rl_framework_selection.md#vllm-sglang-selection)保留 RL 集成与历史版本边界，不能当成通用 serving 排名。
- [Rollout Latency playbook](../practice/playbooks/rollout_latency.md)可借鉴分层定位，但 verifier、cohort 与 staleness 不是普通在线服务的必需组件。
- Paged KV、prefix cache、continuous batching、chunked prefill、推理量化和 SLO 容量规划均仍需独立成章。

## 与其他路线的关系

- [GPU Systems](../01-systems/roadmaps/gpu-systems.md)解释单设备成本，[分布式系统](../01-systems/roadmaps/distributed-systems.md)解释请求、版本与失败边界。
- [训练基础设施](../02-training-infra/README.md)提供模型分片背景，但训练吞吐目标不能直接替代推理 SLO。
- [RL 路线](../04-rl-infra/roadmap.md)增加 behavior logprob、policy version 和训练消费契约，应在推理基础之后学习。

最终要能交付一份“明确 workload / SLO 下，性能与正确性同时成立”的服务验收卡，而不是只报最高 tokens/s。

[返回 Part III](README.md) · [返回首页](../README.md)

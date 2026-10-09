# 分布式系统核心学习路线

[所属 Part I：AI Systems 基础](../README.md) · [首页](../../README.md)

## 学习目的

围绕 AI 作业的通信、控制、队列、版本与恢复，建立“哪些状态可继续、哪些工作可重试”的判断力。
主线不等于多维训练并行：TP / DP 等只是一类消费者，rollout、模型发布与推理服务同样需要这些基础。
本路线不扩展成数据库百科；只学习能解释当前 AI 系统边界的经典机制。

## 先修与起点

- 理解进程、线程、网络请求、文件与基本并发；能区分本地函数返回和远端工作完成。
- 能画生产者、消费者与状态所有者，说明消息中包含什么身份信息。
- [GPU Systems 路线](gpu-systems.md)有助于理解异步设备执行，但不要求先完成 GPU 实测。

## 五阶段主线

“已有”表示存在机制正文；“局部”表示来自训练 / RL 的案例；“待补”不表示已验证。每阶段都先写正常路径，再写一个失败反例。

| 阶段 | 核心问题 | 已有正文与覆盖状态 | 验证 / 掌握标准 |
|---|---|---|---|
| 1. Collective、P2P 与 RPC | group 协同和请求—响应的完成语义有何不同？ | [NCCL 语义](../topics/nccl.md#collective-map)已有；[外部 Agent Gateway](../../rl-infra/topics/agentic_rl.md#external-agent-gateway)提供 RPC 场景；通用 RPC 语义待补 | 对同一工作画出参与者、输入输出、调用顺序和完成点；不能把 barrier 当作跨服务事务提交 |
| 2. 部分失败、timeout、retry 与幂等 | 对端超时，究竟未执行、执行中还是已完成但回复丢失？ | [hang 排查](../topics/nccl.md#hang-diagnosis)、[Gateway 生命周期](../../rl-infra/topics/agentic_rl.md#gateway-lifecycle)与 [状态边界](../../rl-infra/topics/agentic_rl.md#rl-state-boundaries)局部覆盖；统一重试教程待补 | 为一次有副作用的请求定义稳定 ID、deadline、重试预算与重复处理规则；解释“超时”为什么不能直接等同“失败且无副作用” |
| 3. 背压与长尾 | 为什么更多并发可能只增加排队、过期和资源占用？ | [供给 / 执行 / 接收预算](../../rl-infra/topics/agentic_rl.md#gateway-streaming-refill)与 [大规模训练容错](../../training-infra/topics/fault_tolerance.md)已有场景正文；通用排队模型待补 | 分开到达、执行、完成、消费速率；定位 first slow event，给出有界队列、准入与取消后的资源回收设计 |
| 4. 版本协调与流量准入 | 进程存活、权重加载和可接流量为何不是同一状态？ | [权重发布状态机](../../rl-infra/topics/agentic_rl.md#areal-weight-sync-xccl-disk)与 [部分失败恢复](../../rl-infra/topics/agentic_rl.md#rl-state-boundaries)已有 | 记录 desired / loaded / active version 与 worker incarnation；部分成功时能说明谁被隔离、谁继续服务、何时切换 |
| 5. 持久化、原子发布与恢复 | 文件写完、latest 更新与恢复同一逻辑进度分别意味着什么？ | [Checkpointing](../../training-infra/topics/checkpointing.md)与 [恢复不变量](../../rl-infra/topics/agentic_rl.md#monthly-retrospective-invariants)已有；跨存储语义实测待补 | 画 snapshot → payload → validate → publish → restore；列出未提交状态的处理与数据重放 / 去重边界，验证恢复后继续正确推进 |

通信正确性先于性能；失败语义先于重试；状态边界先于自动恢复。不要靠增加 timeout 或无界重试掩盖协议不一致。

## 最小练习与产物

以下均为建议设计，不代表本仓库已经实现或执行这些练习。

1. **通信卡：手算一个小 collective。**

   给每个 rank 分配不同的小数组，列出 AllReduce、AllGather 与 ReduceScatter 的预期结果。
   再画一个 RPC 的 request / execution / response，指出哪些 collective 假设不能搬到 RPC 上。

2. **重试卡：回复丢失但工作已完成。**

   用纸笔或 mock 枚举发送前、执行中、提交后回复前的失败；为每个状态写可否重试与结果查询办法。
   幂等键必须绑定逻辑操作，不能每次 retry 生成一个新身份后仍宣称消重。

3. **背压卡：慢消费者与一个慢请求。**

   为 submitted / inflight / completed / consumed / discarded 建立守恒账本，加入取消和超时。
   说明容量单位是请求、session、cohort 还是 bytes；不要用一个 concurrency 值解释所有层次。

4. **发布卡：一份新版本只更新了部分 worker。**

   画出准备、加载、校验、准入与回退；让一个 replacement 中途加入，检查 incarnation 和版本身份。
   对照 [RL 状态边界实验](../../practice/experiments/rl_state_boundaries.md)设计验收，不把 health check 成功视为发布成功。

5. **恢复卡：payload 完成但 manifest 未发布。**

   列出可恢复版本、孤立文件和未消费工作；明确是在重放、重新采样还是丢弃，并说明允许的重复语义。
   后续获准运行时可衔接 [E05 checkpoint](../../practice/experiments/a100_fsdp_io_lab.md#e05)，先固定拓扑再讨论 reshard。

## 执行与覆盖边界

- [RL 状态边界实验](../../practice/experiments/rl_state_boundaries.md)与 A100 E00–E08 都未执行；[Async Checkpoint](../../practice/experiments/checkpoint/async_checkpoint.md)目前只是 `NEW` 问题卡。
- mock 只能验证局部状态机，不能证明真实网络、跨进程顺序、文件系统持久性或多 rank 数值正确。
- 不默认所有 storage 都支持相同的 rename、可见性和原子发布语义；实现前核对实际后端，再选择提交协议。
- deadline、重试预算、幂等与去重保留期应共同设计；“at-least-once 传递”不自动等于“exactly-once 业务效果”。
- 不把一次 NCCL timeout 都归因网络；先找首个失败 rank / 请求 / 状态转换。
- 所有故障注入限独立、获准的测试进程与输出目录；不能向生产训练或共享存储随意注入故障。

## 主线之后去哪里

- 集体通信与并行布局：进入 [训练基础设施](../../training-infra/README.md)，再读 TP / FSDP 等具体消费者。
- 请求调度、KV 与服务尾延迟：进入 [推理核心路线](../../inference-infra/roadmap.md)。
- trajectory、policy version 与有效供给：进入 [RL 核心路线](../../rl-infra/roadmap.md)。

最终产物应是一张能解释失败与恢复的状态机，不只是一张正常运行的系统架构图。

[返回 Part I](../README.md) · [返回首页](../../README.md)

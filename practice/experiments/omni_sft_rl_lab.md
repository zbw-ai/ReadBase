# Omni SFT / RL / Streaming 验证实验册

**状态：NEW，O0–O6 全部未执行；本页是实验设计，不含实测结果。**

[核心专题](../../05-embodied-infra/topics/omni_models_infra.md) · [学习路线](../../05-embodied-infra/roadmaps/omni.md) · [Qwen3.5-Omni 依据与边界](../../research/tech_reports/qwen35_omni.md) · [实验入口](README.md)

## 实验目标与统一环境卡

问题：在固定任务与质量约束下，哪些变化真正减少 Omni 后训练/交互的端到端成本，同时保持样本、目标、policy 与状态语义？

运行前填模型及所有子模块 revision、数据 manifest/hash、processor/tokenizer/codec/template、训练/rollout 框架 commit、CUDA/driver/kernel 版本、GPU/CPU/RAM/存储/网络、拓扑、精度、并行组、batch 与 accumulation、seed、采样/reward/停止规则。环境暂未提供，因此不写假定可运行的训练命令或资源规模。

每个 run 保存实际 command、config digest、输入 manifest、原始日志和 trace、失败样本、结果计算脚本及结果表；命令空缺就不能标执行完成。公司实验保存在内部批准位置，公开只回写不含内部信息的通用结论。

## O0｜样本展开与媒体时间轴

- **Question**：同一个媒体样本在离线训练和在线 rollout 中是否对应相同内容？
- **Setup**：少量覆盖 text-only、audio、video、audio-video 的许可样本，含不等长、非零 clip offset、空/损坏输入；人工标记音视频同一事件。固定 processor 与增强种子。
- **执行动作**：保存实际抽帧时间、音频片段区间、模态 span、position、target/mask；对两条路径逐字段比较。另做有意错位输入，验证检查会失败。
- **验收**：正确路径没有跨 clip、漏模态或不明偏移；错误输入被显式拒绝/标注；每个 token/frame 可追溯原媒体。容差以 processor 的采样规则定义，不用任意时间常数。
- **Results**：未执行。产出将用于锁定 O1/O4 的输入契约。

## O1｜SFT loss、packing 与分布式等价

- **Question**：优化 layout/分布式执行有没有改变实际训练目标？
- **Setup**：固定 logical batch，包含不同长度、模态缺失、不同 target_count；先用框架支持的小模型/最小路径建立 oracle，再迁移目标模型。关闭或固定随机增强/dropout。
- **对照**：unpacked vs packed；单副本 vs DP；无 accumulation vs 等价 accumulation；CP 仅在声明支持 Hybrid state 的路径上另测。
- **测量**：全局有效目标数、各 loss 分子/分母、逐模块梯度 max-abs/relative error、边界样本输出；记录 norm 接近零时的绝对误差。
- **验收**：mask/计数/边界完全一致，数值误差符合预先基于 dtype 与重复运行确定的容差；GDN state 跨样本重置。用不等长 rank 故意触发 rank-mean 错误，确认测试能检出。
- **Results**：未执行。未通过前不进入吞吐结论。

## O2｜数据与 SFT 性能基线

- **Question**：step 长尾由 I/O/解码、encoder、主干计算还是 collective 主导？
- **Setup**：固定代表性样本混合和训练目标，记录实际 processor 展开长度，保留冷启动和 warmup 边界；稳定窗口多次测量。
- **分解**：读取→解码/预处理→H2D→encoder→Thinker/Talker→backward→optimizer；通信与计算重叠用 timeline 解读，不相加阶段时间冒充 E2E。
- **指标**：E2E logical step time p50/p95、有效目标/秒、有效媒体时长/秒、padding ratio、peak memory、Host RAM、queue wait、rank skew、失败/重试。不同模态 token 不直接合成跨 workload 的统一效率排名。
- **验收**：有可重放 baseline 和一个被 trace 支持的瓶颈假设；还未要求加速。
- **Results**：未执行。

## O3｜一次一种 SFT 优化

- **Question**：bucketing/prefetch/cache/recompute/并行策略中，哪项改善关键路径？
- **Setup**：从 O2 选择一项；A/B 固定样本、logical batch、质量和资源，记录所有有意变化。冻结特征缓存只能在 encoder 与前处理不变时启用。
- **测量**：O2 全指标 + O1 正确性 + held-out 分模态质量；同时记录 warm cache 与 cold cache。若改变帧率、序列截断或采样混合，改称质量/成本消融。
- **验收**：质量在预设容差内，正确性通过，端到端耗时/成本确有改善；没有收益也保留负结果与瓶颈迁移。
- **Results**：未执行，不预设加速百分比。

## O4｜蒸馏与 RL 闭环正确性

- **Question**：进入 update 的数据是否来自声明的输入、策略和 reward？
- **Setup**：先独立构建 paired audio/text、chosen/rejected preference、在线 rollout group 三类样本。先同步 text-output 路径，再扩 Talker；不把 DPO 和在线 RL 混成一个实验。
- **测量**：训练端与 rollout 端同权重同输入的 teacher-forced logprob；target/action mask、sampling config、policy bundle；teacher cache 身份、偏好条件一致性、reward 各分量和 group 完整性。
- **故障注入**：错 processor、错旧 logprob、丢候选、重复 reward、超时、旧 policy/refit、zero-variance group。检查系统按既定算法拒绝、补齐或跳过并留下原因。
- **验收**：概率/梯度对照在预定容差内；错误样本不静默进入 loss；失败/超时留在分母。再计 rollout、waveform、tool、reward、group wait、train、weight sync 的关键路径。
- **Results**：未执行。只有闭环稳定后才另立异步/staleness 对照。

## O5｜流式服务质量与吞吐

- **Question**：增加并发或拆分 stage 是否让用户更快听到正确、连续的回复？
- **Setup**：固定媒体、输出任务/质量、资源与到达模式；同时记录 open-loop 到达率、队列/拒绝和 closed-loop 并发，避免被自节流负载掩盖过载。分别测冷启动/稳定期。
- **测量**：输入到达→首文本→首 codec→首 waveform→客户端首播；p50/p95/p99、RTF、chunk gap、underrun、SLO 内成功会话/秒和完整资源成本。音频输出时长与内容质量一起对比。
- **故障注入**：打断/取消、stage 变慢、断连重试、重复/乱序 chunk；旧 turn 不可继续播放，未完成传输的 buffer 不提前复用。
- **验收**：预设质量和首包/播放 SLO 都满足，再比较 goodput；不得把报告理论首包数值当本地目标已达成。
- **Results**：未执行。

## O6｜Checkpoint、权重发布与恢复

- **Question**：训练或服务部分失败后能否恢复一致状态？
- **Setup**：保存完整模块/optimizer/RNG/sampler 与数据游标清单；发布包含所有相关 Thinker/Talker/MTP/codec/processor/adapter 依赖的版本元组。
- **故障注入**：checkpoint 写入中断、只更新 Talker、tensor 映射错误、旧 cache 重用、发布前 worker 失败；演练拒绝不完整工件及回退上一已验证版本。
- **验收**：恢复后的下一 logical batch、样本 ID、目标数、loss/梯度与无故障路径可解释地一致；新流量仅在校验和 barrier/准入协议完成后进入。训练恢复不默认包含在线播放会话续接，后者另定协议。
- **Results**：未执行。

## 结果记录与决策

| Experiment / run | 实际命令与环境 | 正确性 / 质量 | E2E 性能 / 资源 | 决策 |
|---|---|---|---|---|
| O0–O6 | 待执行后填写 | 无结果 | 无结果 | 不宣称 VERIFIED / IMPLEMENTED |

结果形成后回写[核心专题](../../05-embodied-infra/topics/omni_models_infra.md)；长期项目另存 `practice/projects/`，不把本实验计划当作已完成的项目证据。

# Monthly Signal Report · 2026-10

[月度入口](monthly_reviews.md) · [历史月报缺漏检查](monthly_audit_2026-10-31.md) · [归并账本](audits/2026-10-monthly/aggregation.json)

- 编制日期：2026-10-31，Asia/Shanghai；应用户月底请求提前编制本月摘要，区别于默认次月编制上月报告。
- 月份范围：2026-10；**实际证据截止 10/31 22:20:20**。月底余下时段尚未扫描，本文为月底截至该时刻版，不声称自然月已经闭合；后续扫描有晚发现材料时按原始事件月份归回。
- 输入：[10/08](frontier_scan_2026-10-08.md)、[10/12](frontier_scan_2026-10-12.md)、[10/14](frontier_scan_2026-10-14.md)、[10/19](frontier_scan_2026-10-19.md)、[10/23](frontier_scan_2026-10-23.md)、[10/31](frontier_scan_2026-10-31.md)，以及[十月 backfill](backfill/2026-10.md)、[release 记录](release_notes.md)、[实际学习记录](../learning_log/2026-10.md)与现有专题。
- **5 组月度 Accepted 主线**。六份扫描的 90 组 Accepted 是原报告计数合计，包含跨月及同主线后续事件，不能当作 90 篇十月新文章。月报不新增发现、不修改扫描游标、不重复建立阅读任务。

## 本月核心判断

**趋势推断：** 具身学习正在把视频里的运动信息、动作影响力、历史记忆和反馈分别变为可建模对象；这要求我们区分中间表示、机器人命令和闭环评估，而不是把所有未来视频预测都称为可用的 world model。10/14 的 ACCRUE / AXIS-Bench、10/19 的 REKI / T2-z，以及 10/23 的 CEL / AIM / Multi-view LAM / CADRE，共同构成月底两条模型主线的背景，不能只看最后一周。

工程上，数据输入、数值更新、采样分布和状态交付越来越直接决定模型是否学到了预期目标。LeRobot 的 worker shard、FSDP/EMA，RL 的 sampling support 与 refit，serving 的 KV 布局与队列都说明：吞吐优化必须保留明确的语义和生命周期。以上是已有记录的归纳，不是全领域统计，也不是本仓实测。

## Accepted Signals

以下均为已有 Accepted 或已核 backfill 的归并；Source ID、作者、发布日期与具体证据沿用链接的原记录，不重新声称核查一手网页。未写出完整论文笔记的材料保持 NEW；本页综合判断不等于用户已精读。

<a id="m1"></a>
### M1 · 先统一运动监督，再证明它能成为动作

- Signal ID：2026-10-M1；Source ID：`arxiv:2610.34356v1`、`arxiv:2610.36791v1`；paper / model；Haoyu Zhao 等，原始 v1 10/25、10/27 UTC；First seen 10/31，来源窗口见 [A1](frontier_scan_2026-10-31.md#a1)。
- Learning track：Embodied / Training；Signal type：model / algorithm；Impact：High；Decision：Deep Dive；Status：NEW。
- Target question / Reason：没有 native action label 的视频如何帮助控制学习？[RynnVLA-Latent](https://arxiv.org/abs/2610.34356v1)与 [RynnWorld-Latent](https://arxiv.org/abs/2610.36791v1)利用共享 latent motion 连接 policy 与未来视频，但 VLA Stage-1 权重不能直接当闭环控制器。
- Evidence：原扫描已核方法、消融和官方 README；论文与 README 数据规模存在差异，本月不引用混合规模或榜单。结合 [10/23 Multi-view LAM](frontier_scan_2026-10-23.md)的动作可观测性问题，视觉变化不等于可识别的物理动作。
- 一句话价值：表示复用能扩大监督来源，native action 的对齐和执行接口仍要单独验收。
- Next / 最终去向：沿 [P1](../reading_queue/P1.md#oct31-frontier)画 video→latent→native action 与合成 episode 两条链，回到[具身机制](../../05-embodied-infra/topics/embodied_infra.md#oct31-learning)，不增加新的 P0。

<a id="m2"></a>
### M2 · World Model 必须分别通过训练收益、泛化与失败保留三关

- Signal ID：2026-10-M2；Source ID：`arxiv:2610.34593v1`、`arxiv:2610.33565v1`；paper / benchmark；Hanwen Cui 等、Zheng Zhang 等；原始 v1 10/25、10/24 UTC；First seen 10/31；[A2/A3 原记录](frontier_scan_2026-10-31.md#a2)。
- Learning track：Embodied / Training；Signal type：algorithm / system；Impact：High；Decision：Read；Status：NEW。
- Target question：预测未来为什么能帮助动作学习，又为什么不能据此信任生成式评估？[WAM 消融](https://arxiv.org/abs/2610.34593v1)区分训练期辅助监督与部署视频生成；[WorldFailBench](https://arxiv.org/abs/2610.33565v1)把失败原因、阶段与边界分开。
- Evidence / Reason：原记录已核方法；前者仅 RoboTwin 仿真且单训练 seed，后者的仿真边界不能转写成真机安全保证。与 [10/23 AIM / CADRE](frontier_scan_2026-10-23.md)连读，数据是否包含动作影响、预测能否及时返回，也是独立变量。
- 一句话价值：熟悉任务成功率、分布外控制和失败诊断是不同指标，不能互相代替。
- Next / 去向：复用 P1 评估任务，填 seen / unseen / perturbation 与 reason / stage / boundary 表；关联[仿真评估](../../05-embodied-infra/topics/embodied_sim_evaluation.md)。

<a id="m3"></a>
### M3 · 训练验收从 loss 扩展到参数更新、数据身份和恢复状态

- Signal ID：2026-10-M3；Source ID：`github:huggingface/lerobot#5086`、`#5095`、`#5117`、`arxiv:2610.31859v1`；merged PR / paper；LeRobot contributors、Shubham Singh。PR 10/29–30 UTC 合并，论文 10/22 UTC，First seen 均 10/31；[A5](frontier_scan_2026-10-31.md#a5)与[晚发现 backfill](backfill/2026-10.md#oct31-vla-precision)。
- Learning track：Embodied / Training / Systems；Signal type：system；Impact：High；Decision：Deep Dive（精度论文 Read）；Status：NEW。
- Target question / Reason：梯度非零、作业运行，为什么参数仍可能没有有效更新？[精度研究](https://arxiv.org/abs/2610.31859v1)区分 BF16 compute、parameter storage 和 FP32 master；[分片 EMA](https://github.com/huggingface/lerobot/pull/5117)把平均权重与 checkpoint 纳入训练生命周期。
- Evidence：原论文方法、固定配方与 PR 实现/测试已核，未本地运行。不能从 BF16 导出权重推定训练没有 master copy，也不能泛称所有 VLA 默认配置错误。10/14 streaming worker、10/19 cache RSS、10/23 DP shard 与 Transformers expert gradient 修复提供本月较早证据：输入唯一性和梯度路径同样需要验收。
- 一句话价值：loss 降低只是一个观测点，必须追踪样本、实际参数差和 resume 后的状态连续性。
- Next / 去向：[训练章节](../../05-embodied-infra/topics/embodied_training.md#oct31-precision-state)与 [E7](../../practice/experiments/embodied_contract_lab.md#e7)，实验仅设计，未执行。

<a id="m4"></a>
### M4 · RL 正确性需要贯穿采样分布、权重布局与请求身份

- Signal ID：2026-10-M4；主 Source ID：`github:huggingface/trl@v1.17.0`、`github:areal-project/AReaL#1834`、`github:verl-project/verl#8397`；release / merged PR；HF TRL、AReaL、verl 团队；First seen 10/31；事件日期和窗口见 [A8–A11](frontier_scan_2026-10-31.md#a8)。
- Learning track：RL / Training / Inference；Signal type：algorithm / system；Impact：High；Decision：Deep Dive；Status：NEW。
- Target question：同一 token 的 rollout/training 概率是否属于同一支持集，同一权重版本是否已经成为可执行 layout？[TRL 1.17](https://github.com/huggingface/trl/releases/tag/v1.17.0)明确 sampling support，[AReaL #1834](https://github.com/areal-project/AReaL/pull/1834)恢复在线更新后的 aiter 布局，[verl #8397](https://github.com/verl-project/verl/pull/8397)区分准入等待、首 token 和取消身份。
- Evidence / Reason：原记录已读 release/patch；GRPO 修复受 colocate 和 backend 版本约束，AsyncGRPO 默认目标有变化。承接 10/12 group identity、10/14 loss 分母、10/19 route replay、10/23 活跃请求 KV 与 refit，不能把全部误差归为 staleness。
- 一句话价值：概率、版本和身份都一致之后，有效吞吐才是可比较的指标。
- Next / 去向：[RL 章节](../../04-rl-infra/topics/agentic_rl.md#oct31-contracts)；维护 RL 链路时选读，不挤占当前具身三项任务；先做同权重、同输入、同支持集对照。

<a id="m5"></a>
### M5 · 独立 serving 的 KV 命中必须能按时交付

- Signal ID：2026-10-M5；Source ID：`arxiv:2610.38355v1`；systems paper；Xin Huang 等；v1 10/28 UTC，First seen 10/31；[A12](frontier_scan_2026-10-31.md#a12)。
- Learning track：Inference / Systems；Signal type：system；Impact：High；Decision：Read；Status：NEW。
- Target question / Reason：SSD 中存在 KV，为什么请求还要重算？[FlatCake](https://arxiv.org/abs/2610.38355v1)通过不可变 session/layer 布局、独立队列和有界 GPU window，把逻辑命中变为及时交付。
- Evidence：原记录核对 §3–5，主要是指定配置上的生产 trace replay，不能写成生产上线或任意 workload 收益。与 [10/23 uGDS / UlaCache / SGLang](frontier_scan_2026-10-23.md)以及 [10/14 ReplaySSM / RDMA registration](frontier_scan_2026-10-14.md)连读，稀疏读取、状态表示、DMA 生命周期和释放一致性是不同层次。
- 一句话价值：状态能存下来不够，排队、读放大和 deadline 决定复用是否真正省成本。
- Next / 去向：[推理状态章节](../../03-inference-infra/topics/inference_state.md#oct31-storage)，用 delivered hit、queue time 和重算比例设计 serving 验收；不依附于 RL rollout 才成立。

## OpenAI / Anthropic / NVIDIA / DeepSeek Watch

本表为原扫描 Watch 的归并，Accepted 表示已被原扫描接纳，并非额外月度主线；外部来源本次未重扫。

| Vendor | 原记录 / 月度判断 | Evidence 与边界 |
|---|---|---|
| OpenAI | Accepted / Observed：[10/23 广告服务工程](frontier_scan_2026-10-23.md)；早期 Ironclad / safety cases 留观察 | 主服务时限约束旁路 agent，关注 deadline 与失败隔离；工业自报不等于开源可复现，产品发布不自动接纳 |
| Anthropic | Accepted / Observed：[10/12 环境故障](frontier_scan_2026-10-12.md)、[10/23 形式验证](frontier_scan_2026-10-23.md) | 环境边界与证明适用范围需要分别界定；10/08 正文失败不能视为无更新，后续核验仅关闭具体条目缺口 |
| NVIDIA | Accepted：[10/08 AICR / GPUNetIO](frontier_scan_2026-10-08.md)、10/19 Nemotron、10/23 GDS、[10/31 CUDA/Megatron](frontier_scan_2026-10-31.md) | 配置实际状态、IO 生命周期、训练/推理工件值得优先读；GPU placement、显存限制和故障隔离不同；报告数字保留自报边界 |
| DeepSeek | Accepted / Observed：[10/14 cache](frontier_scan_2026-10-14.md)、[10/19 encoder / Flash-1017](frontier_scan_2026-10-19.md) | 原扫描检查 API changelog 和官方 HF 两处；编码顺序和轮次身份影响输入语义，旧报告不重算首发；全模型历史未穷尽 |

## Hugging Face Watch

| 来源与原记录 | Decision / Evidence / 限制 |
|---|---|
| TRL 1.14.2→1.15→1.16→1.17，六份扫描 | Accepted：结束 token、logits 峰值、共享前缀 loss 多重性、sampling support 是不同变化；不按版本数量重复算创新 |
| LeRobot、Transformers、PEFT / Accelerate / Kernels | Accepted：M3 与 10/23 EP gradient；其余沿原扫描 Observed，不能把仓库列表当全部 PR 已审 |
| HF Blog / 社区作者，10/19–31 Watch | 官方团队、AWS/NVIDIA/Microsoft 厂商社区文章分别标识；Mistral Large 4 的部署与微调边界已进 10/31 Accepted，未升级为本月第六主线；普通发布 Rejected |

## RL Framework Watch

| Framework / 原记录 | 判定与子系统 | 对 AReaL 的可迁移判断与证据限制 |
|---|---|---|
| AReaL：10/14、19、23、31 | Accepted；training / weight sync / data path | 分母、CPU staging、route replay、多模态 row、MoE layout；需要固定 backend，不能仅检查 checksum |
| verl：10/12、19、23、31 | Accepted；scheduler / training / inference backend | CPU snapshot ownership、offload 同步、低精度 refit、准入取消；迁移状态转换，不直接照搬配置 |
| slime：10/08、12 | Accepted；checkpoint/recovery / data path | v0.4.0 的训练恢复与保留 serving 要区分 attempt 和持久状态；其余复杂 PR 保持原 Observe |
| ROLL：10/08、19、31 | Carried forward / Observed / Rejected | v0.4.0 原发布在9月，不能再算10月新发；ROCm窄适配不提升为月度机制；有限索引无新增不是停更证明 |
| OpenRLHF：各轮 Watch | Observed / Not found in queried indexes | release 与零 merged PR 仅覆盖已查询集合，直接提交未完整核查；保留架构对照 |
| NeMo RL：10/12–31 | Accepted；rollout / training / checkpoint/recovery | group identity、route data、活跃 KV refit、receipt/RNG/task identity；作者测试、CPU证据与真实GPU恢复分别标注 |

上述性能、正确性和可靠性证据回到各次 PR/patch 记录；没有在本次月报执行框架测试。

## Inference Systems Watch

- **Accepted：** M5 的独立存储与服务主线；[10/14 vLLM](frontier_scan_2026-10-14.md) sleep 后 RDMA registration、[10/19 SGLang](frontier_scan_2026-10-19.md) write-back producer 同步、[10/23 SGLang](frontier_scan_2026-10-23.md)跨 TP rank KV 释放，分别对应资源生命周期与部署正确性。
- **Observed / coverage limited：** TensorRT-LLM 与其他 runtime 沿原 Watch；10/31 部分 runtime 仅 Atom 快照，不能推断全窗口无关键 PR。
- Release、合并事件和已运行的生产版本分开，论文 trace replay 与上线指标分开。

## Embodied Models & Infra Watch

| Models / Learning | Infra |
|---|---|
| Accepted：ACCRUE、AXIS-Bench、REKI、T2-z、CEL、AIM、Multi-view LAM、CADRE、M1/M2；证据回到 10/14–31 扫描。输入、表示、目标、泛化和失败评估各有适用范围，未独立复现 | Accepted：LeRobot streaming/shard/cache/FSDP/EMA，Isaac Lab world/camera 槽位、布局与 reset sensor freshness；来源为固定 PR/测试，不能推断真机或仿真已通过 |
| PI/openpi、OpenVLA/OFT、GR00T/SmolVLA 沿各轮 Observed 或 Not found / not verifiable，候选名字出现不等于当月机制全覆盖 | TorchCodec #1758 的晚发现缓解保留在十月 backfill，不声称彻底修复 NVDEC；数据像素、时间戳和消费者生命周期分别验收 |

## Observed / Rejected 与跨月处理

- [10/23 晚发现](backfill/2026-10.md#oct23-late) CusADi2、BaGuaLu-2、PlaneZip 保持 Observe；方法或实验对照未充分核清，不因标题/规模晋级。
- [10/14 晚发现](backfill/2026-10.md#oct14-late) ReplaySSM、KV4-RL、ObsSpec 按原 Read 承接，PlaceMoE 仍 Observe；文章原始时间与公告/发现时间分开。
- HAPMoE、ROLL v0.4.0、AReaL #1721、verl #7987、NeMo #3613 已归[9月正式月报](monthly_signal_2026-09.md)，不在10月重复记新发。
- Qwen3.5-Omni 原论文属4月，10/31形成笔记；本月记录阅读产物，[4月补充](monthly_signal_2026-04.md#omni-20261031)记录来源归属。

## 实际阅读成果、队列与下月动作

[十月学习日志](../learning_log/2026-10.md)记录概念讨论和教学推演，掌握仍待用户复述；[Omni 报告笔记](../tech_reports/qwen35_omni.md)为 SUMMARIZED，训练、源码验证与 O0–O6 未完成。具身章节、路线与 E1–E7 形成学习材料，不是 VERIFIED 实验成果。Release 专页没有十月新条目，但六次扫描已内嵌 release 核验，不能因此误判十月未查发布。

P0 不动，P1 复用 [三项任务](../reading_queue/P1.md#oct31-frontier)：表示→动作；WAM 泛化/失败评估；精度/恢复。Omni 延续现有独立入口，不重复新增待办。11月先选择一个问题完成状态图与最小验收，再扩充来源；维护 RL 或 serving 时分别选 M4/M5。

## 覆盖缺口与收尾规则

- 10/31 22:20:20 后至月末尚未覆盖；下一次 frontier 从原游标继续，月报不推进游标。
- arXiv 发现旧缺口仍从9/22 16:52:05回看；GitHub直接提交、外围PR和未完整release历史从9/22 17:06:16回看。10/31 CV/CL失败、缺失公告日和限流仓全窗口PR还需补查。
- 2025—2026 H1历史GitHub全集、独立inference与具身来源穷举均未由本次月报解决。[历史审计](monthly_audit_2026-10-31.md)区分报告遗漏与检索覆盖不足。
- 没有新外部发现、没有GPU或机器人实验；月报归并不改变历史Accepted、原Decision或个人阅读状态。

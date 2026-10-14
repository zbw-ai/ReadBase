# Frontier Scan · 2026-10-14

[研究雷达](README.md) · [扫描账本](scan_log.md) · [具身核心入口](../../05-embodied-infra/topics/embodied_infra.md) · [阅读队列](../reading_queue/P1.md#embodied-20261013)

## 本轮结论与选读

本轮 **16 组 Accepted**：具身模型/学习 2、具身 Infra 4、RL 框架 4、Systems/Inference 3、厂商工业/代码信号 3；另 **27 组 Observed**，在后文逐组列出。数量按 triage 组计，联合 PR 保留各自 Source ID；同一信号出现在多个 Watch 只计一次。

**建议先读 A1 ACCRUE → A2 AXIS-Bench → A3 LeRobot remote inference**。前两项帮助理解旧技能保存与次优示范的使用，第三项把模型输出接到本地执行。做数据工程时加读 A4/A5；维护 RL 时先查 A7–A10；推理状态与通信重点看 A13/A14/A16。无需把 16 项同时变成在读任务。

**整体进展与趋势（本轮样本推断）**：具身方向从扩大数据量进一步走向经验复用、质量条件与闭环部署；系统方向持续把性能优化绑定到状态身份和数值语义。多 worker、紧凑 route、EP 弹性和状态缓存都可能省资源，但前提是样本、分母、动作与网络注册仍属于正确的版本。模型论文没有公开代码不等于机制没有学习价值；代码合并也不等于已经证明真机或生产可靠。

## 时间、计数与证据

- 主事件窗口：**2026-10-12 10:34:41 → 2026-10-14 14:20:25 Asia/Shanghai**；六 RL 框架 merged PR 独立起点为 **10/12 10:27:28**，同一终点。固定 cutoff 后继续核验页面，未将后续核验时间冒充事件覆盖。下一轮仍从 **14:20:25** 回看。
- First seen：除 A4 于 **10/13** 已作线索提及、本轮首次正式接受外，全部 Accepted 均为 **2026-10-14 本轮核验**。每项 scan window 继承本段，A7–A10 使用 RL 独立窗口。全部生命周期 **NEW**；本仓库没有运行上游代码、GPU benchmark、仿真或真机实验。
- GitHub：六 RL 框架 **54** 条 merged PR，三个推理运行时 **427** 条，共 **481** 条完整查询索引；标题全筛、重点正文/patch 核对，绝不是 481 个 diff 全审。具身七库为有界 release/closed-PR 检查，单独保留边界。
- 证据：[具身](audits/2026-10-14-frontier/embodied_evidence.json)、[RL](audits/2026-10-14-frontier/rl_evidence.json)、[Systems/Inference](audits/2026-10-14-frontier/systems_evidence.json)、[厂商/HF](audits/2026-10-14-frontier/root_evidence.json)；索引：[54 条 RL PR](audits/2026-10-14-frontier/rl_merged_pr_index.json)、[427 条推理 PR](audits/2026-10-14-frontier/systems_merged_pr_index.json)、[arXiv 目录](audits/2026-10-14-frontier/arxiv_index_summary.json)。归档保留元数据、页面哈希与核验结论，不镜像论文/博客全文。
- 窗口前材料：[十月补录](backfill/2026-10.md#oct14-late)单列 ReplaySSM 原论文、KV4-RL、ObsSpec 和 PlaceMoE；前三项 Read，后一项 Observe，不计本轮新发布 Accepted。原论文首次提交与近期公告/博客日期分开。

## Accepted

### A1 · ACCRUE：让 actor–critic 经验跨任务累积

- Source ID：`arxiv:2610.18596v1`；Type：paper；Signal type：algorithm；来源：[abs](https://arxiv.org/abs/2610.18596v1)、[方法正文](https://arxiv.org/html/2610.18596v1)。
- 标题：*ACCRUE: Continual Off-Policy Reinforcement Learning for Robot Foundation Models*；作者：Changyeon Kim、Kimin Lee、Yashraj Narang、Yijie Guo。`citation_title` / 全部 `citation_author` / `citation_date=2026/10/13` 已由原始 HTML 核验；v1 提交 `2026-10-13 15:53:26 UTC`。
- First seen：本次核查 `2026-10-14`；scan window：上述窗口；Impact：High；Decision：**Read**；Learning track：Embodied / RL / Training；Status：NEW。
- Target question：任务逐个到达、旧轨迹只能保留少量时，如何复用 value/behavior 而不让新任务 critic 破坏旧技能？
- Evidence：方法 §4 将轻量 actor/critic 从上一阶段继承；critic 只用当前任务 transition 更新，当前共享 actor 在旧任务保留样本上向前阶段缓存输出蒸馏。外部视觉 encoder 始终冻结，使 replay 的视觉表示在基础模型阶段性微调之后仍一致；reference action chunk 提供任务条件。每阶段 base policy 先用累计 demonstrations 微调，再在该阶段 RL 期间固定。
- 模型/学习价值：保留旧技能不必依赖已经被新任务改写的 critic；把“新任务 value 学习”和“旧行为保存”拆开是可迁移设计。
- 系统影响（推断）：replay 需要保存稳定 encoder 特征、reference/action 条件与旧 actor target 的对应版本；actor/critic/base model 的 checkpoint 不能独立任意恢复。
- 评估与边界：§5.3 的真实 DROID 实验使用 π0.5 + FastTD3，6D end-effector delta action、稀疏人工成功标签、异步 actor/learner；比较对象是同框架每任务重启 actor–critic。论文报告后续阶段 robot RL interaction time 降低 31.0%–68.5%，停止条件为最近 20 次 19 次成功；这是作者报告的特定接触装配结果，不是总训练 wall-clock 或通用机器人提升。§6 明确最多五阶段，replay 按任务有界、总内存仍随任务增长；不同 embodiment/action space 和更不相关任务留待验证。未本地复现；项目页经直接 HTTP 访问可核验，并标注 Code (coming soon)，因此未声称代码已公开。
- Reason：直接填补持续学习、replay 兼容与任务切换的现有学习缺口，方法和有边界的真机实验都已披露。
- Related topics：[具身学习入口](../../05-embodied-infra/topics/embodied_infra.md#oct14-learning)、具身策略学习、RL post-training、checkpoint/recovery、trajectory/replay；Next：进入共享 P1 的具身阶段候选，先画出三类模型状态与 replay 字段，再设计“更新 base 后旧 replay 是否兼容”的小实验；不要改变既有 P0。
- 一句话：固定 replay 的视觉表示，并把当前 critic 学习与旧 actor 行为蒸馏分开，让有限经验支持跨任务的 off-policy RL。

### A2 · AXIS-Bench：质量定义、标签粒度与微调方式需要一起比较

- Source ID：`arxiv:2610.18869v1`；Type：paper / model-learning evaluation；来源：[abs](https://arxiv.org/abs/2610.18869v1)、[方法正文](https://arxiv.org/html/2610.18869v1)、[项目](https://axis-bench.github.io/)。
- 标题：*What Matters in Quality-Aware VLA Pretraining: A Benchmark and Empirical Study*；作者：Mingxuan Yan、Litian Gong、Dihong Huang、Zhixu Li、Tianyu Zhang、Wenqian Zhang、Hefeifei Jiang、Peihao Li、Jian Zhang、Mengfei Zhao、Yikai Tang、Hai Zhai、Zehao Wang、Ruijian Liang、Yanjia Huang、Lin Shao、Changliu Liu、Kaiyu Hang、Zhiwen Fan、Masayoshi Tomizuka、Jianfei Yang、Jiachen Li。原始 HTML 三类 citation metadata 完整核验；`citation_date=2026/10/13`；v1 `2026-10-13 17:59:52 UTC`。
- First seen：2026-10-14；scan window：上述窗口；Impact：High；Decision：**Read**；Learning track：Embodied / Training；Signal type：model；Status：NEW。
- Target question：同样的次优示范，是过滤、加权还是条件学习更合适？episode/chunk 质量标签如何与目标任务微调协议相互作用？
- Evidence：§3–5 基于可 replay 的模拟 demonstrations，保留物体和 contact state 以重算质量标签；§4 以接近物体与接触后进展构造 approach-aware reward，再生成 chunk/episode 标签，对比 weighting/filtering/conditioning；每个 VLA 内对齐数据和训练预算。§5.1 比较 π0.5、SmolVLA、X-VLA，但各自初始化与可训练模块不同。§5.4 固定质量 tag 的微调也要测试带/不带 tag 的推理，tag 本身不是通用增益开关。
- 模型/学习价值：数据“质量”不是一个脱离目标的分数；质量定义、粒度、训练使用方式及微调输入契约共同决定收益。conditioning 平均偏向 episode、filtering 平均偏向 chunk，论文同时报告反例。
- 系统影响（推断）：可 replay 数据需保留 state/contact 与相机配置；质量标签生成与动作监督窗口要版本化，避免离线打分与在线样本不同步。
- 关键数字防误读：论文的真实任务 `9.6% → 31.7%` 是**普通 BC 模拟预训练相对无预训练**的收益，不是质量 conditioning 的收益；conditioning 为 `32.5%`，仅比 plain BC 高 `0.83 percentage points`，十项任务中五升四降一平（§6.2）。这些是作者数据，不是本地复现。
- 边界：共享 tabletop simulation；真实评估仅一种 VLA、一个参考 workspace。进展标签可能低估准备/恢复动作；跨模型比较不能分离初始化原因。项目页明确 **Code (Coming Soon)、Dataset (Coming Soon)**，不能宣称已可复现。
- Reason：方法控制和反例能改变“过滤低分轨迹就一定更好”的工程判断；无代码不抹掉已披露学习机制。
- Related topics：[数据 I/O](../../05-embodied-infra/topics/embodied_data_io.md#oct14-io)、具身数据配方、动作 chunk、训练分布与评估；Next：与已有质量/混采材料对照，先定义自己的 label/selection/FT/inference 对照矩阵；共享 P1 候选，保持 NEW。
- 一句话：次优示范的价值取决于质量标签怎么定义、在何种粒度进入训练，以及微调是否保留条件格式，不能只按一个质量分数删数据。


### A3 · LeRobot #4937：远程推理需要本地动作执行契约

- Source ID：`github:huggingface/lerobot:pull:4937`；[官方 PR](https://github.com/huggingface/lerobot/pull/4937)；标题 *feat(rollout): add remote inference engine*；作者 imstevenpmwork；合并 `2026-10-13 14:07:49 UTC`，固定 merge SHA `ae2b57f12223c8c5184fcd814eb63cd9ce7de203`。
- First seen：2026-10-14；窗口如上；Impact：High；Decision：**Deep Dive**；Learning track：Embodied / Inference；Signal type：system；Status：NEW。
- Target question：网络推理期间动作已被执行、指令被切换或队列耗尽，如何防止旧结果恢复已经失效的机器人动作？
- Evidence：PR 完整 146 文件列表已取两页；重点读 `inference/execution.py`、`remote_inference/client.py`、`engine.py`、`chunk_contract.py` 及相关回归差异。prediction/执行分离，robot 端持有队列与 motor 权限；observation age、request identity、execution generation、committed cursor 绑定 chunk，迟到结果不能重新授权旧动作。耗尽后要求本地 hold acknowledgment 和 fresh capture 才恢复，deadline 不因反复检查续期。
- 测试证据：存在 `test_reset_and_timeout_make_late_results_ineligible`、`test_replace_future_requires_current_cursor_and_generation`、`test_starvation_requires_hold_ack_and_fresh_capture_before_recovery`、remote/local parity、session ownership 和 shutdown 测试。作者报告 540 passed / 4 skipped；本次 head SHA 的公开 Full Tests / GPU Tests 等为 success（保存 check-runs），没有本地运行。
- Reason：改变真实部署执行语义，且可以沿代码和回归核验，不是只添加远程 API。
- 边界：一 server/checkpoint、一 active client；自动重连、并发多 client、history-dependent policies 不在实现范围；远端 hosted GPU、routed 硬件与 trained RTC 真机验收仍为作者待办。驱动必须声明 position-hold 支持。旧 async 命令被替换，Unitree G1/SONIC 暂无支持的 remote replacement。
- 历史区别：#4836 于 10-12 合并后被 #4936 撤回，昨天已记录；**#4937 是 10-13 的重新引入事件**。不可继续写成“当前无 remote inference”，也不可重复算 #4836。
- Related topics：[执行契约](../../05-embodied-infra/topics/embodied_infra.md#execution)、实时 action chunk、observation age、部署/恢复；Next：对照现有动作提交边界建立故障表：迟到、切换、断流、hold 失败；不要将 software tests 视为真机安全证明。
- 一句话：把远端返回的 chunk 绑定到观测与本地执行代际，使延迟、暂停和队列耗尽都不能让失效动作重新生效。

### A4 · LeRobot #4874：streaming 多 worker 的吞吐来自更多 CPU，恢复仍需相同拓扑

- Source ID：`github:huggingface/lerobot:pull:4874`；[官方 PR](https://github.com/huggingface/lerobot/pull/4874)；标题 *feat(streaming): use --num_workers with streaming datasets, up to 2.5x faster data loading*；作者 Hadrien-Cornier；合并 `2026-10-12 09:22:07 UTC`；SHA `75cd2ba0ce9ff47686bb2bd7b7f65840606c02ee`。
- First seen：**2026-10-13（已有后续线索）**；本轮完整 triage 2026-10-14；窗口如上；Impact：High；Decision：**Read**；Learning track：Embodied / Training；Signal type：system；Status：NEW。
- Target question：如何让 rank 内多个 DataLoader worker 不重复/遗漏样本，并从训练 step 恢复原 batch 顺序？
- Evidence：`streaming_dataset.py` 的 `_worker_share` 再划分 frame-balanced episode shard；`_batches_per_worker` 模拟 in-order DataLoader turn，恢复时旋转 worker 角色；byte budget/cache 按 rank 划分，显式 episode pool 改成每 worker。代码注明 exact anchor order 需要相同 dataset/seed/sampler/pool/分布式拓扑/worker 数，随机图像变换不属于该恢复保证。
- 回归：`test_workers_yield_every_frame_once`、rank shard、epoch 内/跨 epoch resume、seed、budget 划分、whole batch offset、worker turn 对照。公开 Full Tests/Fast Tests success，GPU Tests skipped；未本地复现。
- 性能边界：作者两组 AV1 数据 CPU loader 报告约 2.1–2.5×，伴随 RAM 上升和单 batch episode diversity 下降；更多 worker 并非降低每样本 CPU 工作。作者明确 A10G/H200 上 ACT 的 GPU-bound 训练无端到端收益。无需把数字当所有训练任务的提速。
- Reason：接续昨天 #3917 的数据契约，新增的并行与恢复条件会改变真实配置判断。
- Related topics：[数据 I/O](../../05-embodied-infra/topics/embodied_data_io.md#oct14-io)、episode IO、DDP rank、checkpoint/recovery；Next：在自己的数据上同时量 samples/s、RSS、batch episode diversity 与恢复顺序；保留昨天 #3917 历史限制的时间边界。
- 一句话：多 worker 通过 CPU 并行提高数据供给，同时把 worker 拓扑、batch 轮转和 episode 多样性变成恢复与配置条件。

### A5 · LeRobot #4804：重叠时间窗可以去重解码，但不能丢掉原始帧语义

- Source ID：`github:huggingface/lerobot:pull:4804`；[官方 PR](https://github.com/huggingface/lerobot/pull/4804)；标题 *perf(datasets): optimize Lance RGB batch decoding*；作者 yanghua；合并 `2026-10-12 14:46:49 UTC`；SHA `24f15b92b556cb861ca65e989462e23505a701bd`。
- First seen：2026-10-14；窗口如上；Impact：Medium；Decision：**Read**；Learning track：Embodied / Training；Signal type：system；Status：NEW。
- Target question：相邻样本时间窗大量重叠时，怎样减少重复解码并控制峰值解码内存，仍保留重复帧、顺序和 timestamp tolerance？
- Evidence：`lance_backend.py` 对同文件 RGB 请求汇总 sorted unique frame indices，解码后 scatter 到原样本顺序；单请求保留直通路径。multi-sample call 按最多 64 unique frames 与估计 64 MiB 预算分块（至少一帧，超大单帧仍可超预算），测试覆盖 duplicates/order、PTS 检查、float32 byte limit、单请求不复制。公开 Full Tests/Fast Tests success，GPU Tests skipped；未本地复现。
- 边界：64 MiB 是单次活跃多样本文件解码的估算，不是 worker RSS 上限；最终输出、decoder cache 和并行文件都不包括，metadata shape 不正确会影响估计。PyAV depth path 不变。作者 benchmark 在 V100/PyTorch 2.9.1/TorchCodec 0.8.1/LanceDB 0.37.1；相邻窗收益显著高于随机窗，不能外推任意随机训练数据。
- Reason：直接改变多模态 IO 重复工作，并有值、顺序和时戳回归，适合与已收录 TorchCodec surface/lifetime 正确性连读。
- Related topics：[数据 I/O](../../05-embodied-infra/topics/embodied_data_io.md#oct14-io)、视频 IO、action history/window、数据正确性；Next：用相邻/随机 window 两类负载测试吞吐、RSS 与逐帧 parity。
- 一句话：同视频的重叠帧先去重再有限批量解码，最后恢复样本原顺序与重复，避免吞吐优化静默改变训练输入。

### A6 · Isaac Lab #8546：不同 world prototype 不能改变环境与相机槽位配对

- Source ID：`github:isaac-sim/IsaacLab:pull:8546`；[官方 PR](https://github.com/isaac-sim/IsaacLab/pull/8546)；标题 *Preserve environment ordering in Newton frame views*；作者 nblauch；固定 merge SHA `76215316eba829b3453f5d20e4e1d537982a774a`；合并 `2026-10-14 02:08:17 UTC`。
- First seen：2026-10-14；窗口如上；Impact：High；Decision：**Read**；Learning track：Embodied / Systems；Signal type：system；Status：NEW。
- Target question：vectorized simulation 混合 prototype 后，相机/pose 是否仍属于同一个 env，还是发生静默跨环境错配？
- Evidence：`newton_site_frame_view.py` 对来源 expression index 与显式 env ID 做 stable order；在 GPU buffers 创建前一起重排 body indices、local transforms、scale 与 renderer destination，保留每 env 内源 frame 顺序。新增 CUDA regression `test_interleaved_clone_prototypes_preserve_frame_order` 构造交错 prototype、多个 frame 与自定义 namespace，同时断言 world pose 和 renderer path。
- 测试边界：作者报告 focused regression 在修复前失败、45 个本地 frame-view tests 通过，并报告 24 environments × 360 steps 的 FoviDex camera replay。公开 head check-runs 的 `isaaclab_newton` 与 rendering checks success，但 **`isaaclab (core) [2/3]` failure**，因此不能写“CI 全绿”。learning recovery 与 contact solver correctness 作者仍标未验证；未本地复现。
- Reason：这是 observation identity / label correctness 问题；比一条普通兼容性修复更影响训练有效性。
- Related topics：[具身学习入口](../../05-embodied-infra/topics/embodied_infra.md#evaluation)、vectorized simulation、camera/pose ownership、数据正确性；Next：把 env ID、frame slot 和 renderer destination 作为异构仿真回归的成组断言。
- 一句话：在 GPU buffer 建立前按环境编号稳定重排 frame 与 renderer 目标，防止交错 prototype 把另一环境的 pose 喂给当前相机。


### A7 · AReaL：loss aggregation 必须与跨 step / DP 分母一起定义

- Source ID：`github:areal-project/AReaL:pr:1546`；Type：official merged PR；Title：`feat(ppo): make loss aggregation configurable`；Author：`EazyReal`；Merged：**2026-10-12 09:40:16 UTC**。
- [PR #1546](https://github.com/areal-project/AReaL/pull/1546) / [固定合并提交 aedc21ec](https://github.com/areal-project/AReaL/commit/aedc21ec66ff03f329dcd4139945da8087e0d0e7)。
- Signal type：system；Learning track：RL / Training；Impact：★★★★★；Decision：**Read**；Subsystem：`training`、`data/trajectory path`；工程维度：objective weighting / distributed normalization correctness。
- Target question：一个 prompt 的回答跨 microbatch、DP rank 与 optimizer step 后，token mean、sequence mean、prompt mean 是否仍对应所选目标？
- Reason / 一句话价值：把原始分母与 token 系数在响应级切分前准备，再与 engine 的 microbatch 权重绑定，避免 filtering 或拆批悄悄改变目标。
- Evidence：实际 patch 新增 `TokenMean`、`SequenceMean`、`PromptMean`、`ConstantLength`；`PromptMean` 在完整 group 上准备系数，以实际 step 数 K / 全局有效 group 数 G 定标，真实全 mask response 保留 step 权重，transport dummy 权重为零；代表测试比较 padded/packed、不同 step 数及双进程 Gloo 的 **固定参数 loss/gradient** 与完整目标。
- 重要边界：这不是多次 optimizer 更新与一次整批更新的轨迹等价证明；默认仍是 token mean。非 token aggregation 与 distillation / tree training 组合被拒绝；M2PO selector 仍依赖 microbatch。上游未跑 native GPU、多节点训练，Megatron-Core 环境缺失限制了完整验证。
- AReaL 迁移价值：原生功能；先把自身期望目标、filter 后的 numerator 与原始 denominator 写成契约，再选择 aggregation，而不是只改配置名。
- Related topics：[Agentic RL](../../04-rl-infra/topics/agentic_rl.md#oct14-objective-contract)、[框架选型](../../04-rl-infra/topics/rl_framework_selection.md)；Next：构造长短响应混合、group 跨 step、全 mask 与 dummy case，验证 loss/gradient 及指标的分母；不提升现有 P0。

### A8 · NeMo RL：零 train/gen mismatch 是严格受限的执行契约

- Source ID：`github:NVIDIA-NeMo/RL:pr:3208`；Type：official merged PR；Title：`feat(megatron): zero train-gen mismatch with non-colocated Megatron inference`；Author：`YigongQin`；Merged：**2026-10-12 04:58:15 UTC**。
- [PR #3208](https://github.com/NVIDIA-NeMo/RL/pull/3208) / [固定合并提交 18eb86d5](https://github.com/NVIDIA-NeMo/RL/commit/18eb86d51b08790248779a939aad952b035f215d)。
- Signal type：system；Learning track：RL / Training / Inference；Impact：★★★★★；Decision：**Read**；Subsystem：`training`、`rollout`、`inference backend`；工程维度：numerical correctness / configuration validation。
- Target question：在模型权重一致时，怎样把由 batch shape、GEMM/attention kernel 与 logprob scoring 引起的偏差，与异步 policy lag 分开？
- Reason / 一句话价值：用配置 gate 强制训练与非共置 Megatron Inference 使用 batch-invariant kernel / scoring 契约，把“零误差”从口头目标变为可拒绝不支持配置的测试条件。
- Evidence：核对 validator、token multiple helper 与实际 nightly 脚本；脚本要求开启 kernel 的 marker，并逐步检查 `gen_kl_error` 的 min/max 均为 0、`token_mult_prob_error` min/max 均为 1、loss/grad norm 有限。上游报告 Qwen3-30B-A3B、2×4 GB200、同步 GRPO 五步通过。
- 重要边界：支持声明限定 **BF16 / Blackwell / TP=1 / CP=1 / 非共置 Megatron generation**，需要 FA4、TE batch-invariant、ordered collectives、raw logprobs、temperature=1、无 top-k/top-p、无 sequence packing；dynamic batching 没有对应运行验证。FP8、Hopper、TP/CP>1、MLA/SSM-hybrid 不在此次证明范围。共置试验后续 step 出现失配且根因未明；异步旧轨迹对当前权重有 KL 属正常 policy lag，不应承诺为零。
- AReaL 迁移价值：迁移配置验证器与“同权重、同 token”的最小对照实验设计；Megatron 双端方案不能直接等同于 AReaL 的 SGLang/vLLM 后端收益。
- Related topics：[Agentic RL](../../04-rl-infra/topics/agentic_rl.md#oct14-objective-contract)、[推理系统](../../03-inference-infra/README.md)；Next：先固定权重与采样分布，对照 batch/precision/kernel/weight-version 四个维度，再讨论 async 误差预算。

### A9 · NeMo RL：支持集外的 token 不能先把 -inf 清零再算 ratio

- Source ID：`github:NVIDIA-NeMo/RL:pr:4367`；Type：official merged PR；Title：`fix(loss): exclude invalid policy support from actor ratios`；Author：`wdykas`；Merged：**2026-10-12 16:41:52 UTC**。
- [PR #4367](https://github.com/NVIDIA-NeMo/RL/pull/4367) / [固定合并提交 557c117d](https://github.com/NVIDIA-NeMo/RL/commit/557c117dce37ba72e520ec9eab841d3033232c0e)。
- Signal type：system；Learning track：RL；Impact：★★★★★；Decision：**Read**；Subsystem：`training`、`data/trajectory path`；工程维度：importance-ratio / gradient correctness。
- Target question：top-k/top-p 后的 current、previous、generation 分布支持集不一致时，为何一个无效 token 会影响整条响应的 GSPO ratio 和其他 token 的梯度？
- Reason / 一句话价值：保留支持集 mask 到 actor loss，再联合三份 logprob 判定有效 token；过早把 `-inf` 替换成 0 会把不可能事件当作概率 1，污染 ratio。
- Evidence：核对普通与 packed loss preparation、loss ratio 与代表性梯度测试；测试将自动排除与显式 mask 对照，覆盖 token/sequence ratio、current/previous/generation 无效来源与完全排除响应；reference KL 另保留原来的未过滤 token 集合。全局 rollout-batch 分母不因各 microbatch 剩余 token 不同而隐式重算。
- 重要边界：只把启用 top-k/top-p 时的负无穷视为预期支持集差异，NaN/+inf 保持可见；无过滤时的 -inf 仍应失败。该修复处理 mismatch 后的损失语义，不使训练/推理 logits 或 top-p 实现相同。上游称 targeted tests 通过，完整分布式收敛、部分 Automodel/compiled CUDA 组合仍待验证。
- AReaL 迁移价值：检查 logprob 清洗、IS/GSPO ratio、KL-in-reward 和 reference KL 是否误共享 mask；把“全部 token 都被排除”列入梯度与指标测试。
- Related topics：[Agentic RL](../../04-rl-infra/topics/agentic_rl.md#oct14-objective-contract)；Next：用一个支持集外 token + 一个有效 token 构造最小回归，分别验证 actor 梯度和 reference-KL 不被误删。

### A10 · NeMo RL：route 数据面同时需要紧凑表示与跨 turn 边界一致性

本组是 **两个独立子机制** 的联合 triage：①广播表示与显存；②多轮 token route 正确性。它们不是同一个修复，也不互为性能因果证据；计为一组便于共享数据路径复查，三个 Source ID 保留。

- Source IDs / 标题 / 作者 / 合并时间：
  - `github:NVIDIA-NeMo/RL:pr:4526`；`fix(data-plane): broadcast route fragments instead of the dense routes table`；`ZhiyuLi-Nvidia`；**2026-10-12 15:03:36 UTC**；[PR](https://github.com/NVIDIA-NeMo/RL/pull/4526) / [固定提交 a5fba402](https://github.com/NVIDIA-NeMo/RL/commit/a5fba40266a2d696d4fb63c8b6e1278104840f90)。
  - `github:NVIDIA-NeMo/RL:pr:4530`；`perf(data-plane): broadcast unpadded rows and pad on each rank`；`ZhiyuLi-Nvidia`；**2026-10-12 20:10:15 UTC**；[PR](https://github.com/NVIDIA-NeMo/RL/pull/4530) / [固定提交 3df254d3](https://github.com/NVIDIA-NeMo/RL/commit/3df254d30675093603cd48c61d812b590ff415ec)。
  - `github:NVIDIA-NeMo/RL:pr:4610`；`fix(sc): Fix router replay with token capture for multi-turn rollouts`；`mk-0`；**2026-10-13 06:00:51 UTC**；[PR](https://github.com/NVIDIA-NeMo/RL/pull/4610) / [固定提交 fe1d97ce](https://github.com/NVIDIA-NeMo/RL/commit/fe1d97ce2b82c2c443a44634b44a894dcf98f568)。
- Type：official merged PR group；Signal type：system；Learning track：RL / Systems；Impact：★★★★★；Decision：**Read**；Subsystem：`data/trajectory path`、`rollout`、`training`；工程维度：GPU/host memory、wire bytes、MoE replay correctness。
- Target question：MoE route 为什么在传给训练前就可能因 padding OOM，而多轮轨迹即便 route coverage 看似正常仍能包含错误专家？
- Reason / 一句话价值：跨 rank 只传真实 route fragments / jagged rows，并在消费时补齐；跨 turn 则要保留下一轮 prefill 才得到的上一轮末 token 路由，数据量和数据身份需要分开验证。
- Evidence A（#4526/#4530）：实际 patch 用 `uint8` blob + row index 搬运去重 route fragments；后续广播 nested values+offsets，`int16` 按 bytes 传输；route 保留 jagged 到 microbatch 再 pad。代表测试检查 fragment round-trip、NCCL、补齐后等于旧 sentinel table。上游 12-node A/B 的 bytes/row 从 24.1 MB 到 7.9 MB、leader fetch 新增 GPU 峰值从 18.7 GiB 到 1.9 GiB；总 step time 1121s 对 1110s 被作者明确判断在噪声内，异步 batch 不配对，不能写成训练吞吐 3 倍。86-node 验证仍待完成；其他普通字段仍存在整批 CPU padding，广播仍逐完整 field 暂存，无 chunking。
- Evidence B（#4610）：原来仅取 `experts[prev_len:]` 会丢掉下一轮才算出的 `prev_len-1` 路由；patch 携带 boundary row，并回填上一 span 的末 token，同时将未记录 route 从合法默认 expert 改为 sentinel。测试断言两次调用拼接后的 routes 等于最后一次 full route。**route plan schema 从 2 升到 3，decoder 明确拒绝旧版本**，升级/恢复必须检查在途旧 plan。
- 重要边界：多轮修复限定 token capture + router replay + vLLM single controller；token echo / single turn 不受此前同一问题影响。最后一个 sampled token 没有后继 target/attention，不需要虚构实际 route。全量训练或恢复兼容并未在本仓库复现。
- AReaL 迁移价值：可迁移的检查项是“实际 token 数 vs padded shape 的通信账本”及“输入 token 才产生路由的跨 turn 边界”；对 schema 变更应检查 in-flight buffer / checkpoint，不能只把一个切片修复移植过去。
- Related topics：[Agentic RL](../../04-rl-infra/topics/agentic_rl.md#oct14-objective-contract)、[框架选型](../../04-rl-infra/topics/rl_framework_selection.md)；Next：绘制 token / route / staging span 对齐表，分别做字节量/峰值与跨轮 route-equality 测试。


### A11 · FlashMask-3：mask 不只决定算哪些元素，也决定 CP 的通信与负载

- Source ID：`arxiv:2610.17901v1`；Type：systems paper；[原文](https://arxiv.org/abs/2610.17901v1) / [方法](https://arxiv.org/html/2610.17901v1)。
- Title：FlashMask-3: Efficient and Expressive Mask-Aware Distributed Attention。
- Authors：Guoxia Wang、Qianyue He、Siming Wu、Haoyang Xie、Qiwen Bao、Jinle Zeng、Jiabin Yang、Tian Wu、Dianhai Yu。
- v1：2026-10-13 07:55:51 UTC；citation_title / citation_author / citation_date 已逐项核。First seen 与窗口见上。
- Learning track：Systems / Training；Signal type：system；Impact：★★★★☆；Decision：Read。
- Target question：packed document / prefix mask 接到 context parallel 后，如何同时避免局部 mask 重建、rank 负载失衡和无效 KV 通信？
- Reason / 一句话价值：通过列式边界的 clip–shift–max 本地变换复用 mask，再把有效块信息用于 Hopper kernel、稀疏负载均衡与分层通信，使逻辑稀疏真正减少分布式工作。
- Evidence：已读 §3.1/3.2/3.4、§4 主要实验、Appendix A；§3.3 局部审阅。作者在 128 H100、118B-A14B、128K context 的完整训练中报告 7.3%–11.0% throughput 提升。
- 数字边界：该完整训练基线是 **FlashMask-3 采用 Dual Chunk + all-gather、关闭新 balancing/overlap 的配置**，不能改写成“比完整 Megatron 训练快 3 倍”。microbenchmark 与端到端训练分开。每组跑 70 steps，去掉最高/最低各 10 steps，剩余 50 报均值与标准差；不是长期收敛/生产稳定性验证。
- 机制边界：“zero communication”只指 **mask localization**，KV 仍需通信；受限于每列可由两段 masked interval 表示的 mask family，Hopper TMA/WGMMA，分层路由假设每节点 8 GPU NVLink 与对应 inter-node 拓扑。小上下文不一定摊销调度开销。
- Related topics：[FlashAttention](../../01-systems/topics/flashattention.md)、CP / packed training；Next：与既有 CompileMask 对照“表达方式—预处理—分布式局部化—通信剪枝”，先检查 mask 与梯度语义，不直接外推性能。无需新增 P0。

### A12 · FastEEP：通过局部状态复用缩短实例内部的 EP 扩缩容

- Source ID：`arxiv:2610.18451v1`；Type：systems paper；[原文](https://arxiv.org/abs/2610.18451v1) / [方法](https://arxiv.org/html/2610.18451v1)。
- Title：FastEEP: Fast Intra-Instance Elastic Scaling for Expert-Parallel MoE Serving。
- Authors：Yunfei Feng、Jingkai He、Shiqi Liu、Cheng Zhang、Dong Du、Erhu Feng。
- v1：2026-10-13 14:22:00 UTC；citation 元数据已核。
- Learning track：Inference / Systems；Signal type：system；Impact：★★★★☆；Decision：Read。
- Target question：MoE 流量变化时，能否回收或增加实例内部的 EP workers，而不重启进程、重建所有通信和 CUDA Graph？
- Reason / 一句话价值：保留最大拓扑的父通信域、worker 与虚拟地址，切换 NCCL active subgroup / NIXL mask，再按语义恢复权重、KV 和 graph buffers；专家映射、通信可达性与请求路由共同跨越一次状态切换。
- Evidence：已读 §4–6。作者使用两节点共 16 个 H20-3e GPU、vLLM 0.27.0、六个 MoE 模型；对能够运行的 vLLM Elastic EP 比较，报告扩缩容耗时加速 4.5×–11.7×。
- 数字边界：是 **扩缩容 latency**，不是 steady-state tokens/s；正常 FastEEP transitions 4.2–13.4s。文中 live serving 的 8→4→8 仍有 4.2s / 7.3s zero-output 区间，不可说零停顿。Elastic EP 无法运行的两个模型未纳入对应比较。
- 适用边界：只能在启动时预建的最大拓扑内缩放，scale-up 需要可用 GPU memory；多于 16 GPU 是设计推断，未实测。inactive buffers 转移到 CPU 带来 host-memory 开销。作者声称开源并提交 vLLM PR，但本轮未核代码链接或该 PR，不能宣称已进入 vLLM release。
- Related topics：[推理系统](../../03-inference-infra/README.md)，MoE EP / memory / serving scheduler；Next：画出 drain→撤销远端可达性→释放显存，以及恢复内存→注册网络→发布路由的状态机，判断自己负载是否值得秒级扩缩容成本。

### A13 · vLLM #59625：sleep 后虚拟地址不变，RDMA 注册仍必须重建

- Source ID：`github:vllm-project/vllm:pr:59625`；Type：merged PR；[官方 PR](https://github.com/vllm-project/vllm/pull/59625)。
- Title：[KV Connector] Support sleep mode with MooncakeConnector over RDMA；Author：aoshen02；merged：2026-10-14 04:18:48 UTC。
- 固定 head `4d45158a3fca8fe081ae961ea615965f1e5434cb`；merge `3ccc0a90b67321bee83a17c9d50f881452057995`。14 个 changed files 已完整取得，实际 patch 核对重点 hooks、Mooncake、配置 gate 与单测。
- Learning track：Inference / RL / Systems；Signal type：system；Impact：★★★★★；Decision：Read。
- Target question：为何 P/D worker sleep/wake 后没有显式错误，KV 内容仍可能错，显存也可能继续被 NIC pin 住？
- Reason / 一句话价值：CUDA VMM 可在同一虚拟地址映射不同物理页，NIC 注册不能沿用旧映射；睡眠前注销 KV memory，恢复映射后重新注册，并在启动阶段拒绝不支持该流程的 connector 组合。
- Evidence：当前 diff 的 `unregister_kv_caches()` / `restore_kv_caches()`、worker 调用顺序、Mooncake RDMA-only gate、幂等 unregister/register 单测都已核对。支持范围排除 NVLink/TENT 等路径；NIXL 等组合拒绝 sleep。
- **重要边界**：PR body 还描述较宽的 pause/abort/recompute 生命周期及旧名称 `release_kv_caches`，但当前合并 diff 只有窄的 registration lifecycle；这里不将正文全部功能当本 PR 已交付。硬件矩阵/大量测试数字来自作者对若干早期 head 的报告，未独立核其适用于当前 head；本轮未运行测试，不复述为已验证保证。
- Related topics：[推理系统](../../03-inference-infra/README.md)、KV transfer、CUDA VMM、rollout sleep / weight sync；Next：检查自用版本的 connector×transport×sleep 组合，并把远端写入清空与 RDMA re-registration 纳入验收，不能仅以地址相等作恢复成功依据。


### A14 · ReplaySSM：把状态写回从每一步解码中移开

- Source ID：`nvidia:replayssm-blog:2026-10-12`；Type：official engineering blog；[原文](https://developer.nvidia.com/blog/replayssm-rethinking-state-storage-for-faster-hybrid-model-decoding/)。Title：ReplaySSM: Rethinking State Storage for Faster Hybrid-Model Decoding；Authors：Ze-Wei Liou、Harris Nover、Amey Naik、Andrii Skliar、Igor Shovkun、Ido Shahaf、Tri Dao；网页 `datePublished` 为 **2026-10-12 16:00:54 UTC**，在主窗口内。
- Learning track：Systems / Inference；Signal type：system；Impact：★★★★★；Decision：Read。
- Target question：SSM 已经使用固定大小状态，为什么 decoding 仍被 HBM 流量、投机回滚和低精度误差限制？
- Reason / 一句话价值：以 checkpoint 加近期输入缓冲表示状态，输出每步计算、状态周期性合并，把状态写回和投机回滚成本降下来。
- Evidence：正文方法、实验条件及关联论文的 arXiv 元数据与摘要已核对，未依赖文章自动生成摘要。作者报告的收益主要在较大 batch；batch=1 的该组测试无提速。低精度读取副本与高精度递归 checkpoint 分离，避免读取量化误差进入后续递归；不等于完全没有精度损失。
- 新旧边界：配套 [arXiv:2610.13778v1](https://arxiv.org/abs/2610.13778v1) 首发为 **10/09 15:59:40 UTC**，属于[十月补录](backfill/2026-10.md#oct14-late)；本项计数的是窗口内工业文章，论文不再算新发。框架集成和生产采用为作者披露，本轮未逐一复核所列集成 PR、未执行 benchmark。
- Related topics：[推理系统](../../03-inference-infra/README.md)；Next：先画 checkpoint / buffer / committed prefix 的状态图，再核对应框架版本，比较含 flush 成本的延迟。

### A15 · Run:ai 开源：分清资源配额、调度和 GPU 共享的责任

- Source ID：`nvidia:runai-open-source:2026-10-13`；Type：official engineering blog + public repository；[原文](https://developer.nvidia.com/blog/nvidia-runai-is-now-open-source-as-part-of-nvidia-dsx-ai-factory-platform/) / [仓库](https://github.com/dsx-ai-factory/run-ai)。Title：NVIDIA Run:ai Is Now Open Source as Part of NVIDIA DSX AI Factory Platform；Authors：Ekin Karabulut、Sam Heywood；`datePublished`：**2026-10-13 14:19:40 UTC**。
- Learning track：Systems / Training / Inference；Signal type：system；Impact：★★★★☆；Decision：Read。
- Target question：多租户 GPU 平台如何把组织配额、成组准入、放置和设备侧资源限制连起来？
- Reason / 一句话价值：公开控制面、集群服务和模块边界，适合沿一次 workload 的生命周期学习，而非只比较 scheduler 的利用率。
- Evidence：官方正文、Run:ai / KAI Scheduler / Workload Map 的公开 README 已读；区分 KAI 的 gang / topology placement、资源管理层的组织策略、GPU Fractioning Operator 对 CUDA 13.4 原生分片的 Kubernetes 集成（显存硬限制由 CUDA driver 执行）。Workload Map / Karta 提供 CRD 结构解析与通用修改，不凭它本身获得完整调度或 autoscaling。
- 边界：KAI 等模块早已公开，不把所有能力写成 10/13 新发明；本轮未部署、未审全仓源码、未验证隔离强度或吞吐收益。CUDA / driver 与设备适配应按具体版本验证。
- Related topics：[GPU 集群](../../01-systems/README.md)、[训练系统](../../02-training-infra/README.md)；Next：以 PyTorchJob 和 disaggregated serving 各画一条提交→配额→准入→放置→抢占/恢复链，核清每个状态的拥有者。

### A16 · DeepSeek-V4.1-Flash：未完成压缩时也要恢复正确的 index K cache

- Source ID：`hf:deepseek-ai/DeepSeek-V4.1-Flash:commit:8b3cec7`；Type：official model-repository commit；[固定 diff](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/commit/8b3cec70c307624fb098c3a04bdbfc118e7abda9)。Title：Restore each indexer's K cache on incomplete compression steps (#12)；发布账户 GeeeekExplorer，联合作者 Ziming Wang / ZenAlexa；**2026-10-12 08:37:23 UTC**。
- Learning track：Inference；Signal type：system；Impact：★★★★★；Decision：Read。
- Target question：当前 token 没有生成新的 compressed latent，是否仍须切回本层拥有的索引缓存？
- Reason / 一句话价值：将 `shared_attn.index_k = self.k_cache` 从“压缩组完成”的分支移到每个 `owns_k` 步骤，避免共享指针沿用其他 owner 的缓存。
- Evidence：实际 diff 与[固定 model.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/8b3cec70c307624fb098c3a04bdbfc118e7abda9/inference/model.py) 的 Indexer 上下文核对；新增 K 仍要求 `latent is not None`，恢复 owner 指针则不再依赖该条件。故障后果为依据代码的判断，本轮没有测得输出差异或运行回归。
- 边界：这是公开参考推理实现的修复，不代表 API 模型升级、权重变更或所有 serving backend 已同步。后续 [`4dd75a8`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/commit/4dd75a8b448d95ae8e284d5c5a1e4b8808bd583b) 还给 Attention / DSparkAttention 的 `wo_a` 投影前加入 activation quantize/dequantize，并调整 TileLang 依赖；作为同仓后续观察，不从 commit 标题 `update` 推断性能收益。
- Related topics：[独立推理系统](../../03-inference-infra/README.md)；Next：设计跨 owner、压缩边界前后和 speculative path 的逐 token parity 检查；未执行。

## Observed：27 组

统一 Decision：Observe；未列入本轮 Accepted，后续动作见最后一列。旧 release 版本回看、常规 commit 和未核实历史线索不进入这里的计数。

| ID / Source | Learning track | Reason、Evidence 边界与 Next |
|---|---|---|
| O1 · [Anthropic Scion](https://www.anthropic.com/engineering/redesigning-source-control) | Systems | 内容寻址、稀疏 checkout 与节点/区域缓存有工业价值；当前优先具身和训练运行时，先观察。厂商扩容数字未复现；需要时再读故障域隔离设计。 |
| O2 · [Anthropic self-replicating AI](https://www.anthropic.com/research/self-replicating-ai) | Inference / RL | 异构硬件部署与评估 harness 有线索，但整体偏能力/安全评估；不将简化实验或趋势预测当真实自主复制结论。后续只追部署正确性机制。 |
| O3 · [LiquidAI LFM2.5-24B-A2B](https://huggingface.co/blog/LiquidAI/lfm25-24b-a2b) | RL / Inference | 10/13 vendor-authored Team Article 披露 MOPD→Agentic RL 配方与本地推理；缺少足以分离阶段收益的消融，速度/榜单为作者自报，先追报告与运行时约束。 |
| O4 · [Megatron 26.09 alpha/lambda RC](https://github.com/NVIDIA/Megatron-LM/releases) | Training | 10/13 两个 rc 标签发布，但 release body 为空；元数据可确认，不凭标签推断新机制。待 tag diff 与版本说明。 |
| O5 · [DeepSeek 4dd75a8 后续数值路径](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/commit/4dd75a8b448d95ae8e284d5c5a1e4b8808bd583b) | Inference | 实际 diff 在 Attention/DSparkAttention 输出投影前补 activation quantize/dequantize；需测逐 token parity，不据此声称质量/速度提升。 |
| O6 · [AReaL #1774](https://github.com/areal-project/AReaL/pull/1774) | RL | 多模态 token/image、frozen state、重试与 colocated pause 契约；63-file 变更未全审，CPU suite 不能替代 GPU/多节点验证。 |
| O7 · [verl TPU GRPO #8155/#8225/#8226](https://github.com/verl-project/verl/pull/8155) | RL / Training | TorchTitan+vLLM TPU 配方；硬件 plugin 和基准等价性未深核，迁移 backend contract 前补硬件验证。 |
| O8 · [verl MTP #8052/#8059](https://github.com/verl-project/verl/pull/8052) | RL / Training | reference layout 与 MoE loader 兼容修复；先对照实际 checkpoint 参数布局，不升级为新架构。 |
| O9 · [NeMo streaming PPO #4341](https://github.com/NVIDIA-NeMo/RL/pull/4341) | RL | pending gradient、tail gating 与一轮 policy 限制有调度价值；上游数值 parity 未解决，先验证目标一致性。 |
| O10 · [NeMo NumPy staging #4654](https://github.com/NVIDIA-NeMo/RL/pull/4654) | RL | CPU preparation 优化；可迁移 ownership/位级等价测试，不能把阶段收益写成完整训练收益。 |
| O11 · [NeMo refit/Reshard #3984/#3865](https://github.com/NVIDIA-NeMo/RL/pull/3984) | RL | weight sync 前显存释放与 NCCL Reshard 配方；先核 export/transfer 生命周期及拓扑限制。 |
| O12 · [NeMo DSpark #3711](https://github.com/NVIDIA-NeMo/RL/pull/3711) | RL | heads/checkpoint 支持仅为一部分，尚不能声称完整训练/runtime；等端到端证据。 |
| O13 · [Frontier Models for Every Robot: Announcing our API beta](https://www.pi.website/blog/api-beta) | Embodied | 披露上传数据、托管微调与 hosted inference，承认 connectivity 问题，但未披露模型机制/可复现评估或网络deadline/fallback契约。 等待技术docs/model card与部署时限/失败恢复说明。 |
| O14 · [Trust-Calibrated VLA Policy Post-Training with Evolving Imagination](https://arxiv.org/abs/2610.16517v1) | Embodied | metadata和方法全文已获取；reliability控制imagination比例/horizon、chunk trust控制replay/weight有潜在价值，尚未完成全部实验协议复核。 核查真实fixed-data adaptation与在线RL的区别、world-model错误度量。 |
| O15 · [Value-Preserving Action Quantization](https://arxiv.org/abs/2610.17252v1) | Embodied | metadata和全文已获取；reachability/value与动作重构误差区分值得读，尚未核proof假设和search成本。 后续action representation专题，核成本/任务范围，不外推pushing到generalist VLA。 |
| O16 · [Reserve CUDA graph memory when the simulation context is created](https://github.com/isaac-sim/IsaacLab/pull/8539) | Embodied | 预留CUDA graph虚拟地址改善Windows renderer启动后capture stall，作者未测试Linux/L40/长训。 遇到同类Windows stall再定向验证，保持平台边界。 |
| O17 · [[RL] Fix training continuation and task configuration overrides](https://github.com/isaac-sim/IsaacLab/pull/8465) | Embodied | 训练resume optimizer选择、task发现与override修复，属于已有训练配置契约的后续实现。 与checkpoint恢复专题连读，不据修复数量自动晋级。 |
| O18 · [[PhysX/OVPhysX] Preserve compliant contacts during randomization and cloning](https://github.com/isaac-sim/IsaacLab/pull/8464) | Embodied | 材料随机化/clone保留compliant contact；native PhysX test尚有作者未验边界。 后续contact fidelity排障时再核CPU/GPU与真实材料参数。 |
| O19 · [Keep recording targets fixed and preserve replay success checks](https://github.com/isaac-sim/IsaacLab/pull/8425) | Embodied | record目标固定与replay success callback有用；replay仍未恢复原始command state。 后续演示replay协议专题核查目标/成功条件恢复。 |
| O20 · [Fix heap buffer overflow (read) in GIF decoder](https://github.com/meta-pytorch/torchcodec/pull/1764) | Embodied | 本页已核合并与说明，窄GIF decoder heap buffer read修复；未建立机器人视频主路径新的系统判断，本轮不升格。 使用GIF输入时定向核修复与release包含关系。 |
| O21 · [RankLens](https://arxiv.org/abs/2610.18009v1) | Systems / Training | 单 GPU 重放目标 rank 的执行；metadata/scope 已核，依赖原控制流/通信记录，待查 trace 成本和修改后有效性。 |
| O22 · [vLLM SHM ownership #61000](https://github.com/vllm-project/vllm/pull/61000) | Inference / RL | 实际 patch 在 SHM slot 释放前取得独立 buffer，并有单槽回归；与已有 snapshot ownership 主题接近，先按实际使用路径复查。 |
| O23 · [vLLM peer registration #57139](https://github.com/vllm-project/vllm/pull/57139) | Inference | 将昂贵注册移出 scheduler 路径；正文性能验证尚缺，待测 cold/warm peer TTFT 与 scheduler gap。 |
| O24 · [SGLang hybrid HiCache #40223](https://github.com/sgl-project/sglang/pull/40223) | Inference | 恢复 target 之外还须恢复各 MTP draft 的 KV/recurrent/conv state；目前描述及文件索引核查，待深读 patch 与训练模型验证。 |
| O25 · [SGLang weight spans #44164](https://github.com/sgl-project/sglang/pull/44164) | Inference / RL | prompt KV weight spans 提供版本可观测性；CI 失败原因未追完，metadata 存在不保证版本一致。 |
| O26 · [TensorRT-LLM 默认迁移 #17495](https://github.com/NVIDIA/TensorRT-LLM/pull/17495) | Inference | 实际 diff 默认 KV manager V2 / Python transceiver；main 合并不等于已进 release，升级前记录两端解析后的配置并验证支持矩阵。 |
| O27 · [TensorRT-LLM readiness fence #19857](https://github.com/NVIDIA/TensorRT-LLM/pull/19857) | Inference | 有条件地 poll ready event 后发布接收，保留 RDMA 次序；限 V2/overlap/PP1/无connector，待测自身负载下 host blocking 与 GPU 空洞。 |

## OpenAI / Anthropic / NVIDIA / DeepSeek Watch

| 来源 | 实际检查与判断 | 边界 |
|---|---|---|
| [OpenAI Research](https://openai.com/news/research/) / [Engineering](https://openai.com/news/engineering/) | Not found in checked sources：可见目录中没有本窗口值得接受的新条目 | 只查可见研究/工程索引，未穷举 API docs、所有产品 changelog 与隐藏分页；不等于整个机构没有更新 |
| [Anthropic Research](https://www.anthropic.com/research) / [Engineering](https://www.anthropic.com/engineering) | Observed：O1 Scion、O2 异构部署评估；Rejected：[10/12 distillation defenses](https://www.anthropic.com/research/distillation-defenses)，本轮优先问题外，未建立可复现 Infra 收益 | Scion 的工业机制有价值，保留后续入口；安全/政策与厂商指控不转写成已验证技术事实 |
| [NVIDIA Technical Blog](https://developer.nvidia.com/blog/) / GitHub | Accepted A14 ReplaySSM、A15 Run:ai；Megatron 两个 RC 为 O4；Embodied / RL / runtime 变更分别在下方 Watch | 工业正文与自动生成摘要分开；已有模块的重新发布不改写为新发明；性能和生产使用声明未本地验证 |
| [DeepSeek API changelog](https://api-docs.deepseek.com/updates) + [官方 HF org](https://huggingface.co/deepseek-ai) + 模型 commits | Accepted A16；Observed O5。API changelog 最新有日期条目仍为 09/10；HF org 按 lastModified 返回前 10 个模型，并核唯一窗口内更新模型的提交史 | web 初次超时后直接 HTTP 成功，不能写成未扫。HF 10/12 更新包含真正 inference patch；README 社区讨论、lastModified 与新权重发布不能混同。本轮没有新权重发布证据 |

## Hugging Face Watch

- **Blog**：[首页](https://huggingface.co/blog)官方/团队文章与社区区分别浏览；O3 为 LiquidAI 署名的 Team Article，不能当 Hugging Face 自研模型或独立评测。一般社区工具/宣传不自动接受。
- **TRL / Transformers / Accelerate / PEFT / Kernels**：分别获取官方 releases 最新 30 条，窗口内均无新发布；可见最新为 TRL v1.15.0、Transformers v5.19.0、Accelerate v1.15.0、PEFT v0.21.2、Kernels v0.17.2。TRL v1.15 已于 10/12 接受，不重复。此处是 release 边界核查，未穷举这些仓库的全部 PR/docs 差异。
- **LeRobot / SmolVLA**：Accepted A3–A5；SmolVLA model card 和格式/实现线索已检查，未确认窗口内新的模型版本。具体 episode/执行证据归下面两栏，不重复计数。
- **Training stack 补充**：Megatron-LM / DeepSpeed / PyTorch releases 最新 30 条已查；Megatron 见 O4，后两者无窗口新发布（可见 v0.19.7 / v2.14.1）。不是对 Megatron/FSDP/DeepSpeed 全部源码活动的完整审计。

## RL Framework Watch

54 条 merged PR 查询 `total_count=returned`、`incomplete_results=false`；实际 patch 深读 6 个 PR，形成 A7–A10 四组。六库 release 历史分页共 178 条，本窗口无新 release。

| 框架 | merged PR | 本轮判断 | Subsystem / AReaL 可迁移性 |
|---|---:|---|---|
| [AReaL](https://github.com/areal-project/AReaL/pulls) | 5 | Accepted A7；Observed O6 | training / data path；原生 aggregation 与多模态状态契约，先核 loss 分母与版本 |
| [verl](https://github.com/verl-project/verl/pulls) | 8 | Observed O7/O8；无新 Accepted | training / inference backend / weight sync；迁移 TPU backend 契约与 checkpoint layout 验收，不直接抄硬件参数 |
| [slime](https://github.com/THUDM/slime/pulls) | 2 | Rejected：两项 index 文档变化；v0.4.0 前轮已收录 | 无新的实质 subsystem 变化；不因活动数量收录 |
| [ROLL](https://github.com/alibaba/ROLL/pulls) | 2 | Rejected：pyarrow pin 与 vLLM-Omni collective backend 缺参修复，均属窄适配 | 后者涉及 weight sync rendezvous，未建立独立重要新机制；可做配置完整性检查 |
| [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF/pulls) | 0 | Not found in queried merged PR/releases | 不覆盖 direct commits；最新 v0.11.2 实为 09/14，纠正元数据，不写成本轮新发布 |
| [NeMo RL](https://github.com/NVIDIA-NeMo/RL/pulls) | 37 | Accepted A8–A10；Observed O9–O12 | rollout / training / data path / weight sync；迁移数值 gate、支持集 mask、route schema 与内存账本，不把配置受限的测试外推所有后端 |

ROLL release 缓存冲突经 REST 核对，最新 v0.4.0 为 09/29；六框架发布历史访问缺口在本次范围内已补证，但直接 commits、其他仓库及早期 PR 的专项缺口仍保留。

## Inference Systems Watch

| 运行时 | 实际覆盖 | Decision / 证据与缺口 |
|---|---|---|
| [vLLM](https://github.com/vllm-project/vllm/pulls) | 166 merged PR 全标题筛选，重点 #59625/#61000/#57139，release 首30项 | Accepted A13；Observed O22/O23。核对 registration hooks 与 ownership 关键 patch；不是全 diff 审阅，也不将 PR 正文早期设计当当前实现 |
| [SGLang](https://github.com/sgl-project/sglang/pulls) | 170 merged PR 全标题，#40223/#44164/#43936 detail/file index，release 首30项 | Observed O24/O25；#43936 只保留候选。draft state 与 weight spans 尚未完整代码/硬件复现 |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/pulls) | 91 merged PR 全标题，#17495/#19857/#20095 detail/file index，前两项关键 patch，release 首30项 | Observed O26/O27；#20095 仅候选；默认 manager/runtime 变更须与具体发布版本分开 |

427 条查询无截页、`incomplete_results=false`；三个 release 列表均无本窗口新发布。可见 vLLM v0.31.0、SGLang v0.5.21；TRT-LLM 顶部为预发布 v1.3.0rc29，不与 stable v1.2.1 混同。独立 serving 的 KV/状态、延迟、扩缩容与部署正确性均已纳入；没有把它们缩成 RL rollout 的附属项。

## Embodied Models & Infra Watch

### Models / Learning

| 来源与实际核查 | 判断 / 学习问题 | 剩余缺口 |
|---|---|---|
| arXiv cs.RO / cs.LG / cs.CV 官方 10/14 new 索引，候选 metadata + 方法 | Accepted A1/A2；Observed O14 MERL / O15 VAQ：持续经验、数据质量、想象偏差与 action representation | 全索引不等于逐篇全文 triage；触觉、灵巧手、移动操作仍不足。ACCRUE / AXIS 项目代码均 Coming soon |
| [PI Blog](https://www.pi.website/blog) / [Research](https://www.pi.website/research) / openpi | Observed O13 API beta：托管训练/推理，未披露可评估的新目标和延迟/回退契约 | 新域名本轮可访问，不延续昨日访问失败；未因视频 demo 接受 |
| OpenVLA / OFT GitHub，GR00T GitHub + NVIDIA 标签页 | Not found：已检查窗口内 release/merged PR，未找到值得接纳的新模型事件 | 不代表开放 PR、未发布论文和其他渠道全部覆盖；ACCRUE 不冒充 GR00T 新版 |
| LeRobot / SmolVLA model card、HF Blog | Infra 三项 Accepted；未确认新 SmolVLA 模型版本 | AXIS 比较中的初始化/训练模块不同，不将表格排名当模型因果比较 |

World Model 本轮 MERL 为 Observe：只在可靠性、action-conditioned imagined data 与学习评估上提问，不用视频观感代表可用于控制。

### Infra

七库均检查 releases 首30项、closed PR 按 updated 倒序至窗口前的有界页面（最多首100条）；详情和返回边界在具身证据中。不是 direct commit 全量审计。

| 仓库 | 实际返回 releases / closed PR | 判断与证据 |
|---|---:|---|
| Physical-Intelligence/openpi | 0 / 100 | Not found：窗口无合并/release；可见 PR 更新已早于窗口 |
| openvla/openvla | 0 / 20 | Not found：窗口无合并/release |
| moojink/openvla-oft | 0 / 2 | Not found：窗口无合并/release |
| NVIDIA/Isaac-GR00T | 5 / 100 | Not found：窗口无合并/release；可见最新 n1.6.1-release 为 04/23 |
| isaac-sim/IsaacLab | 27 / 100 | Accepted A6；Observed O16–O19。simulation frame/environment identity 与恢复/物理参数分别核查；A6 公开 CI 有一项失败 |
| huggingface/lerobot | 11 / 100 | Accepted A3–A5；#4874 为已知线索首次正式接受；#4937 是撤回后的重新引入，#4836 不重复。最新 release v0.6.1 为 08/03 |
| meta-pytorch/torchcodec | 24 / 100 | Observed O20 GIF 窄修复；可见最新仍 09/30 v0.17.0，旧 surface/lifetime 问题沿原账本追踪 |

Isaac Lab browser-demos-v1/v2/v3 为 Rejected / Ignore：是演示资源发布，不当新训练平台主版本或评估证据。软件测试、作者本机结果、公开 CI 和真机验证保持分层。

## 已完成、未完成与下一轮

- 本轮形成雷达记录、[具身学习判断](../../05-embodied-infra/topics/embodied_infra.md#oct14-learning)、[数据 I/O 契约](../../05-embodied-infra/topics/embodied_data_io.md#oct14-io)、[RL 目标与数据表示](../../04-rl-infra/topics/agentic_rl.md#oct14-objective-contract)；共享 P1 只接到已有具身阶段，不改 P0、不标用户已读。
- arXiv 直接 HTTP 已拿到 10/14 官方索引，解决了 web 缓存停留 10/12 的当前访问问题；Systems 四分类 new/recent 共8页、3480个去重 ID，含 replacements。cs.LG recent 达2000条，不能推断覆盖全部历史；公告分组缺 10/13 也不改变论文实际 submission date。
- **旧 arXiv 发现缺口继续从 2026-09-22 16:52:05 上海时间保留**，尤其 9/22–10/06 尚未系统补完；本次有限目录与定向回补不能关闭它。
- **GitHub direct commits / 历史外围 PR / 未完整分页的 release 历史继续从 2026-09-22 17:06:16 保留**。六 RL 框架的 release-list 已单独完成，不能连带抹去其他仓库缺口。
- 下一轮主窗口与本轮已查询的 RL/runtime/具身来源均从 **2026-10-14 14:20:25** 回看去重。之后核验时读取了页面，不表示覆盖所有稍后合并事件；此段时间不跳过。
- 仍未完成的内容包括新模型代码可复现性、PR 的全部硬件矩阵、更多触觉/移动机器人学习来源与历史补漏；不是“已经全面验证了所有实现”。

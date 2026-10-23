# Frontier Scan · 2026-10-23

[研究雷达](README.md) · [扫描账本](scan_log.md) · [具身核心入口](../../05-embodied-infra/topics/embodied_infra.md#oct23-learning) · [共享阅读队列](../reading_queue/P1.md#embodied-20261013)

## 本轮结论与选读

本轮 **19组 Accepted、54组 Observed**：具身模型/学习4、具身Infra1、RL/训练6、Systems/独立推理4、厂商工程4。按判断组计数，联合PR保留各自Source ID；19组含22个主Source ID，同项在多个Watch出现只计一次。Observed为具身17、RL19、Systems10、厂商/HF8；另有3篇晚发现材料归[10月backfill](backfill/2026-10.md#oct23-late)，不混入窗口内数量。

**先读 CEL（A1）→ AIM（A2）→ CADRE（A4）**：分别回答机器人经验怎样回流、什么数据让world model真正跟随动作、为什么高FPS仍会交互滞后。想衔接训练/RL经验，选 **NeMo活跃KV（A6）或Transformers EP梯度（A14）**。其余按当前问题查，不要求逐篇读完；原P0保持不变。

**整体进展与趋势（本轮样本推断）**：具身学习的关注点从“更多示范/更好画质”走向动作后果是否可辨识、失败怎样指导经验复用，以及压缩是否损害交互。Infra则把episode、payload、梯度、权重与KV的身份和生命周期细化为契约；单个算子跑得快或loss正常，都不足以说明系统正确。原生PyTorch后端和更多仿真交付降低接入成本，但兼容性、真实闭环与故障恢复仍需分别验收。

## 时间、范围与证据

- 固定事件窗口：**2026-10-19 09:40:52 → 2026-10-23 11:09:27 Asia/Shanghai**，即 `2026-10-19T01:40:52Z → 2026-10-23T03:09:27Z`。后续取证时间另记[manifest](audits/2026-10-23-frontier/manifest.json)，不拿核验结束跨过未查询事件；下轮从11:09:27回看去重。
- First seen通常为10/23；A6提案9/22已见，本轮新核10/19合并。生命周期均NEW，做了来源/关键方法核查与topic回写，未运行上游代码、证明checker、GPU benchmark、仿真或真机，不表示用户已精读。
- GitHub **23仓、1320条窗口merged PR索引**：七具身仓80；六RL框架91+Megatron/DeepSpeed/TRL110；三runtime893+Transformers/Accelerate/PEFT/Kernels146。各查询returned=total、incomplete=false，标题筛选后核重点diff/tests；**不是1320个diff全审，也不是所有direct commits/所有历史全覆盖**。
- release列表：具身66条、RL/训练9仓448条分页耗尽；Systems7仓首100项/仓为有界核查。窗口内核到IsaacLab3.0.0、DeepSpeed0.19.8、SGLang0.5.22、TRT-LLM1.4.0rc1、Transformers5.20.2；release、main与后续PR不混算。
- arXiv：具身RO/LG/CV目录2756去重ID；Systems LG/CL/DC/AR目录2339；RL限定API查询467/467。**来源重叠不可相加成独立论文总数**，也不是全文全读。具体筛选、metadata、未取到正文和旧缺口见各Watch与[审计入口](audits/2026-10-23-frontier/README.md)。
- 新论文不等于新代码：CEL的abs/HTML题名与索引存在差异，以主源为准；CADRE README仍链接含10/07的blog URL，但首次代码/博客公开时间未据此确认；uGDS README已有7–8月代码记录。A1等论文按v1日期核窗口，涉及新版观察单独注明。

## Accepted

<a id="a1"></a>
### A1 · CEL：失败检索成功经验，车队局部恢复保护 episode 身份

- **Source ID**：`arxiv:2610.25736v1`；Type：paper / industrial-scale real-robot system；[主源](https://arxiv.org/abs/2610.25736v1) / [方法与实验](https://arxiv.org/html/2610.25736v1) / [项目](https://cel-robot-learning.github.io)。Title：*Learning from Fleet Experience: A Collective Learning Paradigm for Robots*；Authors：Chao Yu、Shu'ang Yu 等 **42 位作者**，完整 `citation_author` 已留证；`citation_title/author/date` 三类均已核，v1 **2026-10-19 03:41:27 UTC**，v2 10/21。索引显示另一个题名，本条使用实际 abs 与 HTML 一致的主源题名，不靠索引猜题名。
- First seen：2026-10-23；scan window：上述固定窗口；Impact：High；Decision：**Deep Dive**；Learning track：Embodied / RL / Training / Systems；Signal type：algorithm / system；Status：NEW。
- **Target question / Reason**：机器人越多，如何不同比例增加专家纠错，又不让局部故障污染动作—观察配对或中断整个收数循环？该材料同时给出经验选择算法和真实车队运行约束。
- **Evidence**：§3 的 FERT 用冻结视觉编码与本体状态、时序匹配，从成功轨迹检索失败上下文附近片段，再提高成功 action chunk 的采样优先级；失败动作不作监督。FATS 把实时 inference 与完整 episode 交付分开：station 身份持久、worker incarnation 更换，连接带 epoch，旧回复丢弃，必须完整字段和 outcome 到齐才放行训练。
- **模型/学习价值**：初始化已用 SFT + HG-DAgger，之后无需追加专家动作；仍需人工 outcome、reset 和安全监看，不能称无人监督。系统价值：robot-local recovery 与 learner checkpoint recovery 分层，数据面和控制面的身份检查共同维护正确性。
- **边界**：五任务、同平台两台 held-out AgiBot G2，每策略每任务120次；96机器人/48 A800/专线条件下的运行记录为作者报告。更多机器人同时增加总经验，非固定数据量规模消融；同步/全局重启对照是 trace replay 与模型，非实际部署 A/B。本文项目链接 RLinf，但本轮只核其 README，未定位并逐审 FERT/FATS 完整实现，不能声明代码已复现论文。
- Related topics：[具身模型](../../05-embodied-infra/topics/embodied_models_primer.md)、[具身数据IO](../../05-embodied-infra/topics/embodied_data_io.md)、[Agentic for Embodied](../../05-embodied-infra/topics/agentic_for_embodied.md)；Next：P1阶段候选，先画 station→worker→episode→policy-version 状态图，再区分 retrieval gain 与 fleet data gain；迁移到 AReaL 时优先借鉴 incarnation fencing 与 completed-trajectory admission，不直接套用其监督目标。
- **一句话**：失败只告诉系统该复习哪段成功经验；异步车队还必须保证重启前后的动作与轨迹不会串号。

<a id="a2"></a>
### A2 · AIM：动作影响力比成功轨迹数量更接近 world model 的数据需求

- **Source ID**：`arxiv:2610.27387v1`；Type：paper；[主源](https://arxiv.org/abs/2610.27387v1) / [正文](https://arxiv.org/html/2610.27387v1)。Title：*Training Controllable World Models via Action Influence Maximization*；Authors：Samuel M. Bateman、Tenny Yin、Chongyi Zheng、Lei Huang、Brian Wang、Benjamin Eysenbach、Jaime Fernández Fisac、Dhruv Shah；citation metadata 已核，v1 **2026-10-19 23:21:18 UTC**。
- First seen：2026-10-23；scan window：上述固定窗口；Impact：High；Decision：**Read**；Learning track：Embodied / RL；Signal type：algorithm；Status：NEW。
- **Target question / Reason**：world model 为什么能生成“像成功示范”的未来，却忽略实际输入动作？训练集是否覆盖了同状态下不同动作的可辨识后果，是独立于视频规模的学习缺口。
- **Evidence**：§3 将 action-conditioned 与 action-marginal prediction 的差异联系到条件互信息；AIM 以 replay-density 的状态覆盖项与动作可辨识项指导批量在线采样，实际用 learned density models、IQL/AWR 近似。§4 保持各环境 world-model recipe 固定，比较同新增 rollout 预算，并分别测 end-effector、物体与 joint recall。
- **边界**：Four Rooms / Push-T 仿真；摘要的“85×”只对应 **3k expert + 500 AIM** 与300k expert在特定 end-effector controllability上的匹配，不能写成任务成功率/训练速度85倍。长时评估中某些单项reward与不重估reward的消融反而更好；一般 sequential RL 近似不继承 contextual-bandit 的改善保证。密度模型重训成本不小，本轮未定位官方实现或真机证据。
- Related topics：[具身模型](../../05-embodied-infra/topics/embodied_models_primer.md)、[具身数据IO](../../05-embodied-infra/topics/embodied_data_io.md)；Next：给 world-model 数据审查增加 action shuffle/替代动作反事实、coverage 与 recall 分项，不用 FID 或成功示范数替代；P1候选。
- **一句话**：采集能区分不同动作后果的数据，可能比继续重复同一种成功动作更有助于 world model 跟随控制。

<a id="a3"></a>
### A3 · Multi-view LAM：第二视角补的是动作可观测性

- **Source ID**：`arxiv:2610.27794v1`；Type：paper；[主源](https://arxiv.org/abs/2610.27794v1) / [正文](https://arxiv.org/html/2610.27794v1)。Title：*Multi-View Latent Action Pretraining for Cross-Embodiment Robot Learning from Videos*；Authors：Oliver Hausdörfer、Felix Minzenmay、Mani Majd、Maximilian P. Christof、Shiyuan Zhang、Angela P. Schoellig；citation metadata 已核，v1 **2026-10-20 07:55:43 UTC**。
- First seen：2026-10-23；scan window：上述固定窗口；Impact：High；Decision：**Read**；Learning track：Embodied / Training；Signal type：model；Status：NEW。
- **Target question / Reason**：从无 action-label 视频学习 latent action 时，为什么更多单视角数据仍不能识别离开成像平面的运动？它直接补跨 embodiment 数据、动作表示与几何可观测性的学习缺口。
- **Evidence**：§III 先跨同步相机 patch 做 attention，再时序融合，利用两帧 bottleneck 差值构造 latent action；冻结 VLM 的 latent expert 先预测 plan，action expert 再条件生成 action chunk。视频样本只监督 latent，robot样本另有action监督。§V在目标 embodiment 缺少任务动作示范时单列cross-embodiment transfer，并用 frozen-latent action probe与实际rollout区分表示质量和控制价值。
- **边界**：多视角用于 LAM 训练，不等于部署只有一个相机：policy仍用一个外部相机与wrist view。论文的“17%”是R² **0.54→0.63 的相对提升**；仿真平均成功率42.7→56.2是**13.5个百分点**。probe不能替代闭环成功，精密抓取仍受稀疏latent/缺动作监督限制。项目页只有匿名标题和视频，本轮未找到可核源码，不声称已开源可复现。
- Related topics：[具身模型](../../05-embodied-infra/topics/embodied_models_primer.md)、[具身数据IO](../../05-embodied-infra/topics/embodied_data_io.md)；Next：P1数据配方候选；把相机同步、view identity、latent时间跨度、目标embodiment动作预算列入对照矩阵。
- **一句话**：先让两路同步视角共同解释运动，再压成 latent action，才能减少单视角对深度动作和遮挡的歧义。

<a id="a4"></a>
### A4 · CADRE：视频压缩直接决定 action-to-observation 延迟

- **Source ID**：`arxiv:2610.30129v1`；Type：paper + public code；[主源](https://arxiv.org/abs/2610.30129v1) / [正文](https://arxiv.org/html/2610.30129v1) / [代码](https://github.com/AdrienRR/cadre-autoencoders-and-world-models)。Title：*Rethinking Video Autoencoders for Interactive World Models*；Authors：Adrien Ramanana Rahary、Yutong Bai、Nicolas Dufour、Patrick Pérez、David Picard；citation metadata 已核，v1 **2026-10-21 17:58:11 UTC**。
- First seen：2026-10-23；scan window：上述固定窗口；Impact：High；Decision：**Read**；Learning track：Embodied / Inference / Systems；Signal type：model / system；Status：NEW。
- **Target question / Reason**：world model FPS够高，为什么控制仍明显滞后？时空压缩既改变算力需求，也决定等待多少个动作后才返回观察，不能只按重建图像质量挑VAE。
- **Evidence**：§2–5用冻结DINOv3特征、小bottleneck与因果diffusion decoder构造多压缩点，配相同world-model recipe比较四环境。时间latent要等其覆盖的动作到齐，因此chunk等待不能靠更快kernel消除；空间压缩又可能掩盖小区域手部运动。评价分FID/FVD与从rollout解码动作的ARR，而非只看PSNR。
- **边界**：ARR是probe相对真实视频的可恢复度，**不是闭环task success或力学正确性**；静相机/移动相机解释仍与任务、动作、场景差异混杂。H100低延迟属于论文指定streaming配置。代码README的双用户profile仅约14 FPS且排除网络/browser，不能继承单用户20 FPS结论；multires autoencoder不能直接替换已发world-model配套latent。README仍链接含10/07的blog URL且有额外配置，但该URL不证明实际公开日期；审计另取到10/22博客，故此处只计10/21论文事件，不称本轮首次开源；未运行代码。
- Related topics：[具身模型](../../05-embodied-infra/topics/embodied_models_primer.md)、[具身基础设施](../../05-embodied-infra/topics/embodied_infra.md)、[推理系统](../../03-inference-infra/README.md)；Next：P1 world-model路线候选；将“等待action chunk + WM计算 + decode + 网络显示”拆账，并分别验收motion、action following与下游控制。
- **一句话**：时间压缩省token却延迟动作反馈，空间压缩可能保住画质却丢掉细小动作，必须以交互预算选压缩方式。

<a id="a5"></a>
### A5 · LeRobot #4890：DP shard 身份必须由 trainer 传给 worker

- **Source ID**：`github:huggingface/lerobot:pull:4890`；Type：official merged PR；Title：*fix(streaming): give the trainer's data-parallel rank to the streaming dataset*；Author：Hadrien-Cornier；[PR](https://github.com/huggingface/lerobot/pull/4890) / [固定提交](https://github.com/huggingface/lerobot/commit/8a125584153240d7511dddfbda440bbee56b134a)；合并 **2026-10-21 19:26:55 UTC**，base=`main`。
- First seen：2026-10-23；scan window：上述固定窗口；Impact：High；Decision：**Read**；Learning track：Embodied / Training / Systems；Signal type：system；Status：NEW。
- **Target question / Reason**：trainer已初始化分布式，为什么spawn出来的DataLoader worker仍把全部episode读一遍？这是数据分片正确性问题，训练“跑通”与GPU utilization不能发现重复曝光。
- **Evidence**：4文件完整diff已读。trainer用`parallel_dims.dp_rank/dp_world_size`设置dataset的显式shard，pickle进spawn worker；显式值优先于worker中的process group和环境变量。新增回归故意令环境变量与trainer rank冲突，在真实spawn loader检查样本集合；另测trainer wiring。
- **边界**：原问题影响未设置`RANK/WORLD_SIZE`的自建launcher；标准torchrun/accelerate launch不应被统称全部受影响。没有显式配置时仍保留旧fallback。head checks可见6 success/1 skipped，但不是本仓库执行，也不证明所有DP/TP组合、resume与episode均衡已验证；main merge尚非新stable release。
- Related topics：[具身数据IO](../../05-embodied-infra/topics/embodied_data_io.md)、[分布式训练](../../02-training-infra/topics/distributed_training.md)；Next：加入“trainer rank与环境变量故意冲突、跨rank样本交并集”的契约检查，不只验loader长度。
- **一句话**：把trainer的DP身份显式序列化给数据worker，才能避免自建launcher下无报错的全量重复读数。

<a id="a6"></a>
### A6 · 活跃请求 KV 是 refit 状态的一部分，prefix cache reset 不足以保证重算

- Source ID：`github:NVIDIA-NeMo/RL:pull:3929`；[feat: pause SingleController refits and enable KV recomputation by default](https://github.com/NVIDIA-NeMo/RL/pull/3929)；Author：`youngeunkwon0405`；Merged：`2026-10-19T20:45:38Z`；base：`main`；[固定merge](https://github.com/NVIDIA-NeMo/RL/commit/7344ae75271a7474e87214f4e1de46792fa0d058)。
- First seen：2026-09-22（open proposal）；本轮2026-10-23核实合并事件；Scan window：同本报告；Type：official merged PR；Impact：high；Decision：Read；Learning track：RL / Inference；Signal type：system；Status：NEW。
- Target question：异步请求跨越权重更新时，如何使KV、engine membership和dispatch gate在同一生命周期内变化？
- Reason：默认行为改变且补齐active-request KV与故障恢复契约，不能仅按普通pause接口适配处理。
- 一句话价值：以pause→refit→resume围住权重更新，默认清除活跃请求KV后重算；失败时保持dispatch关闭，重建fleet后重新pause参与者。
- Subsystem：weight sync / rollout / inference backend；Engineering dimension：correctness / state / recovery。
- Evidence：已读single_controller.py、generation/interfaces.py、vllm_generation.py与SGLang lifecycle hooks实际diff，并核refit recovery/resume/controller回归；测试涵盖pause前失败、transfer后partial state、替换actor、共享resume deadline、失败不重开dispatch和SGLang只在pause内flush。
- Evidence limits：38-file PR为关键路径审查；未运行GPU或上游测试。SGLang synchronizer本来就flush，hooks只是确认，flag=false也不保留KV；不额外提供其timeout。旧head约3% step-time差异不作为当前性能结论，PR正文注明当前head GPU恢复/性能待测。同步vLLM等无native hook路径不自动获得等价保证。
- AReaL transfer：可迁移生命周期与失败gate；需按AReaL具体backend重写active-KV处理，不能只调用prefix-cache reset。
- Next：在AReaL建立request/weight/KV version状态表，注入pause timeout、transfer中断、replacement actor与resume失联，分别量重prefill开销和accepted fresh tokens。
- Related topics：[agentic_rl](../../04-rl-infra/topics/agentic_rl.md)；[rl_framework_selection](../../04-rl-infra/topics/rl_framework_selection.md)。

<a id="a7"></a>
### A7 · 低精度 RL：trainer量化与bucket refit共同定义权重/scale/layout契约

- Source ID：`github:verl-project/verl:pull:8287`；[[BREAKING][megatron, vllm, sglang, rollout, doc] refactor: unify QAT and low-precision rollout](https://github.com/verl-project/verl/pull/8287)；Author：`wuxibin89`；Merged：`2026-10-19T14:01:44Z`；base：`main`；[固定merge](https://github.com/verl-project/verl/commit/8f016846e3010e225072a947270dc0c1b98c1cd0)。
- Source ID：`github:verl-project/verl:pull:8306`；[[2/3][sglang] feat: support mxfp8/mxfp4/nvfp4 weight refit](https://github.com/verl-project/verl/pull/8306)；Author：`wuxibin89`；Merged：`2026-10-21T08:39:31Z`；base：`main`；[固定merge](https://github.com/verl-project/verl/commit/5265d561a3bf29f3af3d0aeae8634fe8bf4617c3)。
- Source ID：`github:verl-project/verl:pull:8308`；[[3/3][vllm] feat: standardize low-precision kernels on DeepGEMM](https://github.com/verl-project/verl/pull/8308)；Author：`wuxibin89`；Merged：`2026-10-21T10:23:29Z`；base：`main`；[固定merge](https://github.com/verl-project/verl/commit/619c397dc9917fed278b6e47c52a6620aef05466)。
- First seen：2026-10-23；Scan window：同本报告；Type：official merged PR；Impact：high；Decision：Read；Learning track：RL / Training / Inference；Signal type：system；Status：NEW。
- Target question：QAT或低精度rollout如何避免trainer与engine重复量化、scale布局漂移以及CUDA graph地址失效？
- Reason：量化责任从rollout端移到trainer exporter，跨backend refit协议和配置均改变；格式支持本身不足以概括工程价值。
- 一句话价值：trainer输出预量化权重与scale，engine累计所有sync buckets后重建kernel布局并写回已capture存储；适用FP8/MX格式统一DeepGEMM策略。
- Subsystem：training / weight sync / inference backend；Engineering dimension：precision correctness / state / memory。
- Evidence：已读engine quantizer注册/export路径、结构化RolloutQuantizationConfig、SGLang staging/finalize/abandon及MXFP4/NVFP4存储恢复、MXFP8派生scale重建，核vLLM后端选择与CPU restage测试；实际测试验证weight/derived scale地址与对象保留。
- Evidence limits：60/9/8-file PR仅选读关键路径，未逐行全审、未运行GPU。abandon恢复存储绑定不等于事务回滚旧数值；in-place staging可能已有部分新值。NVFP4保留专用backend，不可概括全部格式都DeepGEMM；硬件/shape/版本限制仍需核。不能从共同kernel推断logprob bitwise一致或收敛收益。
- AReaL transfer：可迁移quantization ownership、bucket-finalization和CUDA graph指针契约，不能直接复制全backend矩阵。
- Next：AReaL评估trainer exporter/schema、final-bucket marker、失败后的generation gate与storage pointer；比较同token raw logprob tail而非只看均值；迁移旧字符串quantization配置须显式。
- Related topics：[rl_framework_selection](../../04-rl-infra/topics/rl_framework_selection.md)；[agentic_rl](../../04-rl-infra/topics/agentic_rl.md)。

<a id="a8"></a>
### A8 · 多模态轨迹重组必须按batch row切图像payload

- Source ID：`github:areal-project/AReaL:pull:1812`；[fix(utils): split multimodal payloads by trajectory rows](https://github.com/areal-project/AReaL/pull/1812)；Author：`zmuxuny`；Merged：`2026-10-23T02:46:10Z`；base：`main`；[固定merge](https://github.com/areal-project/AReaL/commit/86843b5001c6633f8dfe753c05f94dfba495c418)。
- First seen：2026-10-23；Scan window：同本报告；Type：official merged PR；Impact：high；Decision：Read；Learning track：RL / Training / Embodied；Signal type：system；Status：NEW。
- Target question：concat→advantages→split→选择/重排轨迹后，视觉payload如何仍与本轨迹token行对齐？
- Reason：不仅重复拷贝图像造成内存增长，也破坏轨迹级所有权；对后续多模态/具身训练数据管道有直接约束。
- 一句话价值：split仅对multi_modal_input*列表按trajectory group行数切片后deepcopy，并拒绝payload行数与batch不符，保留组内alias与组间隔离。
- Subsystem：data/trajectory path / training；Engineering dimension：correctness / memory / multimodal ownership。
- Evidence：完整读取2-file实际diff与CPU回归：不等长/空组、无图像占位、多模态key变体、错误行数、unique storage线性增长，以及真实PPOActor.compute_advantages后选择/重排再microbatch的图文对齐。
- Evidence limits：上游173项CPU通过和storage probe为作者报告，本轮未执行；CPU storage bytes不是RSS/GPU OOM或吞吐。仅list且匹配multi_modal key，普通metadata list不按行切。Megatron controller已有strip/restore workaround；不表示该路径此前无保护，也不证明机器人实验。
- AReaL transfer：AReaL原生修复；具身迁移是数据所有权与重组验收，不是动作模型或机器人性能结论。
- Next：AReaL多模态路径加入concat/split/reorder所有权回归，将图像、视频或其他payload的row cardinality与token sequence长度分开检查。
- Related topics：[agentic_rl](../../04-rl-infra/topics/agentic_rl.md)；[rl_framework_selection](../../04-rl-infra/topics/rl_framework_selection.md)。

<a id="a9"></a>
### A9 · AsyncGRPO admission按KV压力调整，吞吐必须扣除stale丢弃

- Source ID：`github:huggingface/trl:pull:7610`；[Adaptive in-flight cap from vLLM load in AsyncGRPO](https://github.com/huggingface/trl/pull/7610)；Author：`AmineDiro`；Merged：`2026-10-19T08:40:23Z`；base：`main`；[固定merge](https://github.com/huggingface/trl/commit/d296691ae5ca2db37887942fa6b60265f2581595)。
- First seen：2026-10-23；Scan window：同本报告；Type：official merged PR；Impact：medium-high；Decision：Read；Learning track：RL / Inference；Signal type：system；Status：NEW。
- Target question：固定max_inflight造成服务端排队时，怎样减少stale rollout而不把raw tokens/s误作有效训练供给？
- Reason：从静态并发上限转为KV反馈控制，具可迁移调度价值；同PR改变KV保留语义，必须同时暴露。
- 一句话价值：按KV capacity/max_model_len初始化cap，周期性读取usage/queue，超阈值减小admission且不取消在途请求，低负载随完成量渐增。
- Subsystem：scheduler / rollout / weight sync；Engineering dimension：latency / policy staleness / cache state。
- Evidence：完整读取7-file diff：usage>0.8降30%，<0.6且无queue并完成一个cap量后增25%，5秒poll；缺metrics不误当idle。fake metrics server测试真实控制/生成循环的扩容、spike、drain和group完整性。
- Evidence limits：未运行上游测试。作者2+2 H200受限8GiB KV对照显示raw生成下降而stale丢弃下降，不泛化其数字。该PR在async GRPO/distillation pause中显式clear_cache=false保留旧KV，不能视为与A6重算方案等价；metadata缺失/服务不可达有fallback，KV cap也未解决满cache下纯policy-lag问题。
- AReaL transfer：可迁移admission控制与有效供给口径；不直接采纳clear_cache=false或固定阈值。
- Next：在AReaL先对齐metrics/DP聚合语义，再记录raw/fresh/accepted token速率、age分布与trainer等待；独立验证跨weight update旧KV的policy一致性。
- Related topics：[agentic_rl](../../04-rl-infra/topics/agentic_rl.md)；[rl_framework_selection](../../04-rl-infra/topics/rl_framework_selection.md)。

<a id="a10"></a>
### A10 · BIRD-RL: Scaling Agentic Reinforcement Learning over Stateful Data-Centric Environments

- Source ID：`arxiv:2610.29426v1`；[官方abs](https://arxiv.org/abs/2610.29426v1) / [全文方法](https://arxiv.org/html/2610.29426v1)；Authors：Ge Qu, Jinyang Li, Xiaohan Xu, Xiaolong Li, Bowen Qin, Nan Huo, Ziwei Tang, Chenhao Ma, Reynold Cheng；v1：`2026-10-21T08:50:39Z`。
- First seen：2026-10-23；Scan window同本报告；Type：systems paper；Impact：high；Decision：Read；Learning track：RL / Training / Systems；Signal type：system；Status：NEW。
- Target question / Reason：多轮有状态数据库环境怎样复用replica，同时保持轨迹内连续性和轨迹间隔离？该工作对具体state/IO/部署瓶颈披露机制与测量，值得进一步阅读。
- 一句话价值：用worker分区pool和trajectory→replica绑定跨tool调用持有环境，终止/timeout回收，dirty副本在下次租用前lazy reset。
- Subsystem：rollout / data/trajectory path / scheduler；Engineering dimension：state isolation / IO / correctness。
- Evidence：官方citation_title/author/date与abs一致；已读§4、§5.3/Table3、Appendix D.2及limitations。SQLite B=16/R=16条件下线上DB lifecycle I/O 2.94TiB→236GiB（12.8×），另有69GiB离线pool；训练实现描述基于verl、8 H100。
- Evidence limits：12.8×不是吞吐/显存倍数，排除了pool物化且dirty rollout仅9.25%；hot DB耗尽、worker崩溃、timeout后迟到写入的fencing未在本轮验证。仅SQL场景；代码链接公开但尚未执行，性能/效果均作者自报。
- AReaL transfer / Next：把pool exhaustion、lease epoch、late tool result、dirty reset失败纳入AReaL环境验收；先用SQLite故障注入再谈其他有状态环境。
- Related topics：[agentic_rl](../../04-rl-infra/topics/agentic_rl.md)。

<a id="a11"></a>
### A11 · FlashNext-RL: A System Optimization Study for Long-Horizon Agentic RL of Qwen3.8-Flash-Next

- Source ID：`arxiv:2610.28998v1`；[官方abs](https://arxiv.org/abs/2610.28998v1) / [全文方法](https://arxiv.org/html/2610.28998v1)；Authors：Zhewei Yao, Yuxiong He；v1：`2026-10-20T23:53:09Z`。
- First seen：2026-10-23；Scan window同本报告；Type：systems paper；Impact：high；Decision：Read；Learning track：RL / Training / Systems；Signal type：system；Status：NEW。
- Target question / Reason：长上下文agent RL在host offload、routing payload、refit与checkpoint上怎样共同限住内存和停顿？该工作对具体state/IO/部署瓶颈披露机制与测量，值得进一步阅读。
- 一句话价值：先减少host copy/gradient staging，再优化稀疏attention与MoE；按CP shard和microbatch传routing payload，layerwise refit与有界buffer复用导出checkpoint。
- Subsystem：training / data/trajectory path / weight sync / checkpoint/recovery；Engineering dimension：host memory / throughput / checkpoint state。
- Evidence：官方citation metadata与全文相符；已读§3 memory plan/CP/QSA/8-layer ladder、§4 host/batch/refit/checkpoint及§5实训口径。8层同批次累计对照2391→17678 REAL tok/s/GPU（7.4×）；完整48层开发行配置变化，不可直接作受控speedup。
- Evidence limits：作者自报，一训三推共4个8×B200节点；首两层、n-gram/indexer/MTP frozen。实训仅steps5–15，22.4min平均迭代，含padding trainer速率与REAL不可混用；稳定KL不是任务质量/长期收敛。未找到本文专属实现链接；引用prime-rl等不等于改动已开源。local NVMe DCP不能耐节点丢失。
- AReaL transfer / Next：AReaL先量host PSS、routing bytes/token、pack/decode同时存活份数和checkpoint clone；保持同shape同batch做单变量试验，不复用论文推算8–11训练节点结论。
- Related topics：[agentic_rl](../../04-rl-infra/topics/agentic_rl.md)；[long_context_training](../../02-training-infra/topics/long_context_training.md)；[checkpointing](../../02-training-infra/topics/checkpointing.md)。

<a id="a12"></a>
### A12 · uGDS：GPU direct 数据路径之外，控制路径和DMA生命周期也要设计

- Source ID：`arxiv:2610.30868v1`；Type：system paper + public repository；[论文](https://arxiv.org/abs/2610.30868v1) / [方法与实验](https://arxiv.org/html/2610.30868v1) / [官方代码](https://github.com/ScaleX-IO/uGDS)。Title：*uGDS: A User-Space Development Kit for High-Performance GPU Direct Storage*；Authors：Jing Wei、Guanyi Chen、Qi Chen、Chuyu Zhang、Zheng Zhou、Chutian Wang、Jian Zhang；citation_title/author/date=`2026/10/22`已核，v1 **2026-10-22 07:45:58 UTC**。
- First seen：2026-10-23；scan window：上述固定窗口；Impact：High；Decision：**Read**；Learning track：Systems / Inference / Training；Signal type：system；Status：NEW。
- Target question / Reason：SSD→GPU已经直传后，为何小IO和stream-ordered接口仍慢；timeout后能否立刻释放映射？它把host控制开销、不同流量的队列隔离和DMA安全生命周期放到同一设计里。
- Evidence：§2–3由CPU在用户态构造NVMe命令、管理SQ/CQ，payload经P2P DMA；minimal kernel helper用于特权setup，不等于完全不需要内核。同步与batch流量分硬件queue；stream接口通过host callback衔接同一IO engine。in-flight引用和registration generation防止过早释放及旧请求解除新映射；不可恢复timeout将handle标wedged、隔离queue/memory，外部controller reset停止DMA后才允许强制回收。应用仍负责NVMe/RDMA/GPU之间的producer-consumer依赖。
- 数字和边界：§4.6 的LMCache实验为A100-SXM4-40GB+Samsung990PRO、Qwen3-0.6B、vLLM0.20.1+cu129、LMCache0.5.1.dev35、Torch2.11/CUDA12.9，关闭vLLM内存APC、强制SSD-only hits。32,768-token prompt作者median TTFT为1915→715ms；不能外推混合命中率、现代runtime或大型模型。文中goodput图也称logical KV throughput，不宜当带SLO的标准goodput。设计绕开文件系统，需独占/重新绑定存储设备；不是一般Python读文件的无条件替换。
- Code boundary：只核官方README与固定SHA `2ae25d078f7a51e1f50042d61848431d2ed15512`，未做实现全审或运行。README已有7月HIP、8月LMCache适配器记录；本轮事件是**新论文**，不是新项目或首次集成。
- 一句话：把NVMe控制搬到用户态可减少提交开销，但缓存收益必须连同CPU占用、stream推进和DMA停止证明一起验收。
- Related topics：[Systems主题](../../01-systems/topics/) GPU/IO、[inference_state](../../03-inference-infra/topics/inference_state.md)、KV offload；Next：固定cache-hit条件，分别测提交、DMA、scatter与TTFT，设计超时隔离/重注册负向测试；不执行上游设备setup脚本。

<a id="a13"></a>
### A13 · UlaCache：稀疏读取不等于历史可以丢弃

- Source ID：`arxiv:2610.29820v1`；Type：system paper；[论文](https://arxiv.org/abs/2610.29820v1) / [正文](https://arxiv.org/html/2610.29820v1)。Title：*UlaCache: Unified Logical Addressing for Sparse LLM Serving*；Authors（citation顺序）：Isaac Zhu、Alan Cao、Hugh Yin、Rocky Wang、Orion Wang、Hscos Zhang、Jack Li、Jason Zhao；citation_title/author/date=`2026/10/21`及方法已核，v1 **2026-10-21 14:46:11 UTC**。
- First seen：2026-10-23；scan window：同上；Impact：High；Decision：**Read**；Learning track：Inference / Systems；Signal type：system；Status：NEW。
- Target question / Reason：每步只访问少数KV，为什么decode仍受总历史容量限制；如何共同利用CPU/GPU容量、减少MLA跨TP复制，又不把迁移中的页发布给attention？
- Evidence：§3–5稳定logical address解耦选取和物理位置；DP本地管理，TP按逻辑token位置分布存储，并保留每计算rank的local selected-KV buffer，仅fetch misses。CPU扩容时PB-LRU以共享GPU池和并行recency buckets/bitmaps替代固定每请求配额；保护当前访问，等待旧readers结束，再交换CPU/GPU KV，**全部传输完成后提交mapping并发完成信号**。相同selection的层组复用plan并预取；不是丢弃旧KV的近似压缩。
- 数字和边界：§6为GLM-5.3、8×H200或8×B300、FP8权重/KV与BF16 activation；Multi-LexSum输入64K–128K、每请求greedy生成2048 tokens，MTP两draft steps；内存预算匹配，但各方法比较各自实测batch范围的峰值。B300 TP+spec作者peak decode throughput为HiSparse的2.51×，**先完成整批prefill再计decode**，不是在线到达/SLO/TTFT结果。BS1八种配置均慢于GPU-resident SGLang、快于HiSparse；扩容量和管理成本有取舍。没有据此宣称跨模型精度/所有稀疏attention等价；正文未定位本方法公开实现入口，不声称可复现。
- 一句话：稳定逻辑地址与共享缓存能把更多物理容量用于独立历史，但每次地址更新都要遵守读者结束→数据搬移→映射发布的顺序。
- Related topics：[inference_state](../../03-inference-infra/topics/inference_state.md)、sparse attention、MLA/TP、KV tiering；Next：画独立history/replica/cache副本的容量账本，分开测BS1延迟、峰值decode吞吐与在线SLO。

<a id="a14"></a>
### A14 · Transformers EP：forward正常仍可能让experts完全不训练

- Source ID：`github:huggingface/transformers:pull:49688`；[PR](https://github.com/huggingface/transformers/pull/49688)；Title：*Fix zero expert gradients in bfloat16 with expert parallelism and ep_size < fsdp_size*；Author：qgallouedec；merged **2026-10-19 02:45:19 UTC**，merge `d88946453f945ecdc7bc5cf4427084a5d6c215fe`。
- Source ID：`github:huggingface/transformers:pull:49710`；[PR](https://github.com/huggingface/transformers/pull/49710)；Title：*Fix expert-parallel token dispatch gradients on torch 2.7 to 2.10*；Author：digit50；merged **2026-10-20 07:14:05 UTC**，merge `9f5750c67ad4c2e077dd3883d20503f3d77b2fe5`。
- First seen：2026-10-23；scan window：同上；Type：official merged PRs；Impact：High；Decision：**Read**；Learning track：Training / Systems；Signal type：system；Status：NEW。
- Target question / Reason：expert的梯度分母不等于其FSDP subgroup大小，或token dispatch collective缺autograd时，怎样发现静默训练失败？两项是不同失效机制，统一指向逐模块backward契约。
- Evidence：两个PR各2文件的实际diff全核。#49688让expert reduction无条件SUM后本地除，避免BF16 PREMUL_SUM路径；4GPU `fsdp_size=4,ep_size=2`回归比较整体batch参考的loss和各参数gradient norm。#49710把dispatch/combine都切至`all_to_all_single_autograd`，`wait_tensor`包identity backward，现有EP backward测试新增按shard对应的逐参数梯度比较与None检查。
- 边界：#49688不是逐元素gradient parity；其PyTorch≤2.13问题范围来自作者与链接上游，不能写所有NCCL版本均有同缺陷。#49710 PR正文描述一个双Gloo CPU回归，但最终diff只显示加强既有backward测试，不能把未在最终diff定位的测试当证据。两者目标head首100 check-runs均有失败，失败原因本轮未诊断；未本地运行。v5.20.2 release明确包含#49688，发布早于#49710 merge，不能称该release已经含后者。
- 一句话：用可微collective保住梯度路径，再按实际数据覆盖定义reduction分母；loss能算出来并不能证明experts得到正确更新。
- Related topics：[fsdp](../../02-training-infra/topics/fsdp.md)、MoE / expert parallelism；Next：建立相同global batch的单卡/分片backward对照，检查None、norm和逐元素误差，覆盖dtype×Torch版本×EP/FSDP布局。

<a id="a15"></a>
### A15 · SGLang #45453：失败传输的容量释放也必须跨rank达成一致

- Source ID：`github:sgl-project/sglang:pull:45453`；Type：official merged PR；[PR](https://github.com/sgl-project/sglang/pull/45453) / [固定merge](https://github.com/sgl-project/sglang/commit/b57b5e22867f110ff577a9da18a9ff7e6619100c)。Title：*[Fix] disaggregation: Synchronize deferred decode KV release across TP ranks*；Author：paulzhang-tm；merged **2026-10-21 17:37:29 UTC**。
- First seen：2026-10-23；scan window：同上；Impact：High；Decision：**Read**；Learning track：Inference / Systems；Signal type：system；Status：NEW。
- Target question / Reason：PD失败后每rank ACK和deadline不同，为什么局部释放会进一步造成admission分歧和polling collective形状错位？这是分布式资源生命周期，不仅是单个请求错误处理。
- Evidence：3文件全部diff已核。失败hold flags用MAX归约，只要一个attention-TP rank需要保留，所有rank均hold；逐请求ready用MIN归约，全部rank达到其排空/已有device-timeout条件才共同释放。仅为peer保留的rank不虚构本地ACK依赖。host destinations始终要求drain，timeout不能替代远端写入结束。新增双进程真实Gloo回归覆盖错开的ACK/deadline、非对称通知、host timeout及独立请求推进。
- 边界：GPU cache和transport通知在回归中mock；作者明确无GPU/RDMA A2E或model accuracy测试。device保留既有timeout分支，本PR不能证明所有device remote writes已停止。目标head check-runs首100/共126中有失败，未核完整原因；本轮未执行。main合并不等于stable已发。
- 一句话：释放的不是某rank私有的一个页，而是决定下一批admission和collective次序的共同容量，必须保留和释放都一致。
- Related topics：[inference_state](../../03-inference-infra/topics/inference_state.md)、PD disaggregation、failure handling；Next：把交错ACK/timeout与host不因timeout释放加入故障注入，并在真实transport核远端DMA终止边界。

<a id="a16"></a>
### A16 · KvikIO / GDS：Python 读进 GPU，不代表走了直接存储路径

- Source ID：`nvidia-blog:2026-10-22:kvikio-gds-python`；Type：official engineering tutorial；[原文](https://developer.nvidia.com/blog/how-to-read-data-into-gpu-memory-in-python-with-nvidia-gpudirect-storage/)。Title：*How to Read Data into GPU Memory in Python with NVIDIA GPUDirect Storage*；Author：Andrew Hoyle；发布 **2026-10-22 16:39:09 UTC**。First seen：2026-10-23；scan window：本报告固定窗口；Impact：High；Decision：**Read**；Learning track：Systems / Training / Embodied；Signal type：system；Status：NEW。
- Target question / Reason：为什么用了 `CuFile`、读完的字节数也正确，CPU staging 仍可能存在？数据管线调优先要证明实际路径。
- Evidence：正文把 KvikIO→cuFile 的选择与 cuFile→nvidia-fs/compatibility 的选择分开；`KVIKIO_COMPAT_MODE=off` 只约束前一层。教程用进程存活时的 `CUFILE_STATS_LEVEL=3` / `gds_stats` 区分 nvfs、posix；KvikIO 自己走 POSIX 则不出现在 cuFile 计数中。task 分块与 cuFile 内部分块也是两级；未注册目标 buffer 即便走 GDS，仍可能通过 GPU bounce buffer 再复制。
- 边界：教程为 H100、CUDA13.3/cuFile1.18、本地 ext4 NVMe、指定 PCIe 拓扑与系统配置；本文没有给出机器人视频端到端提速。文件字节直达 GPU 不代替视频解码、时间对齐或训练样本构造。uGDS 是另一条用户态块设备控制路线，不能把两篇当同一个实现。本仓未安装依赖或改系统参数。
- Related topics：[具身数据 I/O](../../05-embodied-infra/topics/embodied_data_io.md)、GPU/存储路径；Next：先记录实际库/驱动/文件系统，再测读路径、解码和 batch-ready 时间各自占比。
- 一句话：先用运行时计数证明数据究竟怎样进入 GPU，再讨论 GDS 是否缓解了实际供给瓶颈。

<a id="a17"></a>
### A17 · TorchNeuron：原生 PyTorch 后端减少迁移工作，执行语义仍需核对

- Source ID：`pytorch-blog:2026-10-19:torchneuron-native`；Type：official vendor-authored technical post + public repository；[PyTorch 原文](https://pytorch.org/blog/introducing-torchneuron-native-pytorch-backend-for-aws-trainium/) / [固定 README](https://github.com/aws-neuron/torch-neuronx/blob/50b14171affaa365f1df6aa015232427336f6a4d/README.md) / [AWS profiling 教程](https://huggingface.co/blog/amazon/profiling-torchneuron-part-1)。Title：*Introducing TorchNeuron: Native PyTorch backend for AWS Trainium*；byline：Nafea Bshara 等 AWS 团队（完整署名见来源）；发布 **2026-10-19 12:45:23 UTC**，正文预告10/20 public beta；本轮已核公开仓库及 Beta README，不将仓库创建日当新发布日。profiling 教程为 Josh Longenecker，10/21。
- First seen：2026-10-23；scan window：本报告固定窗口；Impact：High；Decision：**Read**；Learning track：Systems / Training / Inference；Signal type：system；Status：NEW。
- Target question / Reason：跨硬件仍用 PyTorch 时，哪些成本被接口统一隐藏，哪些需 Infra 独立验收？
- Evidence：PrivateUse1 注册 `neuron`；eager 按 op 执行，compiled 路径经过 FX/AOTAutograd→Torch-MLIR/StableHLO→Neuron executable；`torch.profiler` 关联框架、runtime 与设备 trace。profiling 文用同输入比较拆散 RMSNorm、融合算子和 compile，并要求同步后导出。README 明确静态 shape、非连续输入复制、首次编译成本；stream/event 维持顺序不代表每条 stream 独占硬件引擎。
- 边界：固定 README 为 Trn2/Trn3、Python3.12–3.14、torch/torch-neuronx2.14 配套；adaptive eager 默认关闭；FlexAttention有限，float64按float32执行，本公开版本未提供TorchAO集成（用于MXFP8等复合类型），博客另称相关FP8能力仍在private beta。不能把宣传中的“改 device 即可”解释成所有模型、dtype、动态 shape 和分布式组合已等价验证。源码公开已核，未全审或运行。
- Related topics：[分布式训练](../../02-training-infra/topics/distributed_training.md)、[Systems](../../01-systems/README.md)、profiling；Next：建立 op/dtype/shape/backward/collective 兼容表，再比较冷编译、热路径与完整 step；HF TorchTPU 另列 Observe。
- 一句话：原生后端让模型定义更易复用，但迁移验收必须包含数值、动态 shape、通信与编译成本，不能只看 `.to(device)` 是否成功。

<a id="a18"></a>
### A18 · Anthropic 形式验证：可机检的证明也有精确适用范围

- Source ID：`anthropic-research:2026-10-22:formal-verification`；Type：official research post + proof artifacts；[原文](https://www.anthropic.com/research/formal-verification) / [固定证明库](https://github.com/theorem-labs/oss-verify-proofs/tree/8f5f46768858af236a4270f9628e6beed9aa0e72)。Title：*LLMs are making formal software verification feasible at a much larger scale*；发布机构：Anthropic，与 Theorem 合作；**2026-10-22**，页面日期字段为00:00占位，不当真实发布时间。First seen：2026-10-23；scan window：本报告固定窗口；Impact：High；Decision：**Read**；Learning track：Systems / RL；Signal type：system；Status：NEW。
- Target question / Reason：测试覆盖不足时能否把某个可靠性要求变成可机检目标；checker通过究竟证明了什么？这直接关系到基础库和 agent verifier 的验收边界。
- Evidence：把 OSS-Fuzz entry point 编译为 RISC-V binary，以 Linux/ISA semantics 生成定理，再由模型写 Rocq proof、kernel检查。允许修补代码后重生成命题，因此仍需审核行为改变，避免把程序改成无行为来轻易满足目标。共享证明可复用；不是信任模型口头声称正确。
- 数字与边界：博客自报239 entry points/123 libraries；本轮固定公开 README 是 **60 entry points/25 libraries**，不能把全部研究成果当已开放工件。博客主要研究案例限定输入<64MB、单线程静态 RISC-V/Linux binary等假设；公开README则逐项定义plain/sanitizer构建的可观察行为一致及无error，并排除plain超内存预算的输入（ASan 2GiB、UBSan 16GiB）。二者不是同一个统一验收协议；ASan也不覆盖所有内存安全错误，LPS明确不完整。可信计算基础还包括 semantics、定理、checker及二进制导入工具；不覆盖任意硬件/并发行为/全库逻辑。没有运行证明检查。
- Related topics：[Systems](../../01-systems/README.md)、[Agentic RL](../../04-rl-infra/topics/agentic_rl.md)、verifier correctness；Next：选一个小 entry point，列 spec/assumptions/patch provenance/checker 边界，再决定是否做可负向验证的实验。
- 一句话：让模型产出由独立 checker 验证的证明很有价值，但验证对象、环境假设和实际公开工件的范围仍必须逐项说清。

<a id="a19"></a>
### A19 · OpenAI 广告服务：旁路 agent 如何服从主响应时限

- Source ID：`openai-engineering:2026-10-21:chatgpt-ad-stack`；Type：official production engineering post；[原文](https://openai.com/index/building-chatgpt-ad-stack/)。Title：*How we built ChatGPT’s novel ad stack for user trust and relevance*；Authors：Ritendra Datta、Karthik Gopalratnam；**2026-10-21**，页面只给日期。First seen：2026-10-23；scan window：本报告固定窗口；Impact：High；Decision：**Read**；Learning track：Inference / Systems / RL；Signal type：system；Status：NEW。
- Target question / Reason：一个多轮、可反复调用工具的辅助 agent，怎样在不断变化的输入与时限下工作，又不阻塞用户的主响应？这是已部署服务的架构披露，具有可迁移的状态与故障隔离经验。
- Evidence：独立模型/服务接收会话上下文；orchestrator 发起检索、过滤、打分与拍卖工具，拍卖而非 LLM 直接定价。候选可随上下文更新；主响应就绪而广告未就绪时丢弃机会，返回无广告是正常结果。日志拆分不合格、未检索、无候选、交付与实际互动，避免把不同流失环节混为一个成功率；高能力模型评估辅助小模型的优化。
- 边界：文章自报生产架构，未公开源码、负载/成本账本、p99、取消传播实现或训练目标，不能推出旁路在所有共享资源上绝无干扰。无广告不等于训练负奖励；server delivery也不等于真正展示。收入与市场规模不作为本仓接受依据。
- Related topics：[推理状态](../../03-inference-infra/topics/inference_state.md)、[Agentic RL](../../04-rl-infra/topics/agentic_rl.md)、deadline/admission；Next：画主响应、辅助任务、过期结果和最终展示的生命周期，核实际实现是否有取消与资源隔离。
- 一句话：多轮 agent 的服务设计要明确何时停止等待、哪种“空结果”算成功，以及每个阶段到底交付了什么。

## Embodied Models & Infra Watch

### Models / Learning

| 来源 | 实际核查与判断 | 缺口 |
|---|---|---|
| arXiv [cs.RO](https://arxiv.org/list/cs.RO/recent) / [cs.LG](https://arxiv.org/list/cs.LG/recent) / [cs.CV](https://arxiv.org/list/cs.CV/recent) | 6个new/recent页面，2,756去重ID含replacements；recent分别463/1427/907，完整显示10/19–23五个announcement日，均低于2000 cap。按具身标题与问题筛选，16篇metadata/abstract；15篇HTML可取，4篇方法/评估重点读。Accepted A1–A4；Observed EO1–EO7；CusADi2另列late | 不是2,756篇全文triage；只覆盖本轮筛选的类目与标题。SuperMimic HTML404，未PDF深读；保留既有9/22起旧发现缺口，不反称补齐历史 |
| [PI Blog](https://www.pi.website/blog) / [Research](https://www.pi.website/research) / openpi | **Not found in checked window**；直接HTTP核索引，最新仍10/13 API beta；openpi窗口merged/release均0 | 不涵盖未公开API内部变化、open PR及direct commits |
| OpenVLA / OFT | **Not found in queried merged PR/releases**，均0 | 未查所有direct commits/外围论文 |
| GR00T / Isaac Lab / NVIDIA具身工业资料 | GR00T #797 **Observed**；IsaacLab3.0 **Observed**；Cosmos/H-Dreams/SimReady/Microduck **Observed**，日期、作者与交付层分开记录 | 不能把GA与模型首发混为一项，也未证明所有资产/backend/生成视频物理等价 |
| [SmolVLA card](https://huggingface.co/lerobot/smolvla_base) | card/API/最近10 commits已核，sha=`5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6`，lastModified=10/06；权重卡本窗口**Not found**，LeRobot #4292输入修复**Observed** | 模型卡没更新不等于runtime没变；不覆盖全部派生SmolVLA |

### Infra

七库exact-UTC merged查询 **80条**，均`returned=total_count`、`incomplete_results=false`；每库1页（100/page、search最多1000条，本轮均远低于cap）。**全部80条title筛选与base/merge/head SHA补核**；12个PR读取详情及完整文件清单，只有A5与EO8等标明者读完全部diff，不称全库code audit。release-list返回66项、每库1页均少于100，因此列表耗尽；窗口内仅IsaacLab v3.0.0一个release。

| Repository | merged total / returned / pages | release返回 / pages / 窗口内 | 判断 |
|---|---:|---:|---|
| Physical-Intelligence/openpi | 0 / 0 / 1 | 0 / 1 / 0 | Not found in queried surfaces |
| openvla/openvla | 0 / 0 / 1 | 0 / 1 / 0 | Not found in queried surfaces |
| moojink/openvla-oft | 0 / 0 / 1 | 0 / 1 / 0 | Not found in queried surfaces |
| NVIDIA/Isaac-GR00T | 1 / 1 / 1 | 5 / 1 / 0 | Observed EO13，旧N1.7 release不重复 |
| isaac-sim/IsaacLab | 37 / 37 / 1 | 26 / 1 / 1 | Observed EO12，v3.0.0已GA；main更新未自动算入tag |
| huggingface/lerobot | 42 / 42 / 1 | 11 / 1 / 0 | Accepted A5；Observed EO8–EO11；发布仍v0.6.1，merged main不等同stable |
| meta-pytorch/torchcodec | 0 / 0 / 1 | 24 / 1 / 0 | Not found in queried merged/releases；最新v0.17.0（9/30），不重复收录 |

Routine doc/CI/视频宣传与窄拼写修复 **Rejected / Ignore**，未因量大提升Accepted。保留 **2026-09-22 17:06:16 上海起的direct commits/历史外围PR缺口**；本轮完备性只限七库指定窗口的merged search与release-list，不覆盖所有commit。完整索引、metadata、文件数量/patch hash与what-read边界见[具身审计](audits/2026-10-23-frontier/embodied/README.md)。

## RL Framework Watch

| 框架 | 实际coverage与Decision | 仍有缺口 |
|---|---|---|
| AReaL | 11/11 merged，release26/26；Accepted A8，Observed RO1/RO2，其余常规infra/CI/文档不独立接受 | 默认分支direct commits未扫；#1816 Ascend aggregation为分支适配；GPU未复现 |
| verl | 22/22 merged，release16/16；Accepted A7，Observed RO5/RO6 | 大PR仅关键路径核查，QAT与CP组合矩阵未复现 |
| slime | 4/4 merged，release11/11；Observed RO3/RO4；README与MUSA bootstrap适配不独立接受 | recovery与独立runtime未做真实GPU/多节点重放 |
| ROLL | 0/0 merged，release5/5；Not found in queried merged/releases | 不等于direct commits/open PR无变化 |
| OpenRLHF | 0/0 merged，release110/110（100+10）；Not found in queried merged/releases | 同上，未扫direct commits |
| NeMo RL | 54/54 merged，release10/10；Accepted A6，Observed RO7–RO10；cp/nightly/docs不重复计机制 | 复杂draft/xtoken/MSC未完整深审，branch backport单列，GPU fault injection未跑 |

### 训练库、TRL与发现范围

| 仓库 | merged returned/total | pages | release历史返回 | 窗口release |
|---|---:|---|---:|---|
| NVIDIA-NeMo/RL | 54/54 | 54 | 10（10） | 0 |
| NVIDIA/Megatron-LM | 58/58 | 58 | 51（51） | 0 |
| OpenRLHF/OpenRLHF | 0/0 | 0 | 110（100,10） | 0 |
| THUDM/slime | 4/4 | 4 | 11（11） | 0 |
| alibaba/ROLL | 0/0 | 0 | 5（5） | 0 |
| areal-project/AReaL | 11/11 | 11 | 26（26） | 0 |
| deepspeedai/DeepSpeed | 12/12 | 12 | 118（100,18） | v0.19.8 |
| huggingface/trl | 40/40 | 40 | 101（100,1） | 0 |
| verl-project/verl | 22/22 | 22 | 16（16） | 0 |

六核心框架91条merged PR；附加Megatron-LM/DeepSpeed/TRL110条，共201条。所有query incomplete_results=false、单页不足100条；9仓release历史448条已取完（OpenRLHF 100+10、DeepSpeed 100+18、TRL 100+1）。窗口只有DeepSpeed v0.19.8（2026-10-20T15:21:47Z，loadams）。初次sandbox网络失败，获授权只读重试成功，未留下未完成分页。

arXiv API当前窗口cs.DC/cs.LG + training/reinforcement查询：467/467，分页300+167；属于限定查询完整索引，不等于全arXiv、cs.CL-only或历史发现完整。额外定向搜索发现HarnessRL。

## Inference Systems Watch

| Runtime | Actual coverage | Decision / gaps |
|---|---|---|
| vLLM | 332窗口merged PR，4页100/100/100/32；releases首100，latest v0.32.0=10/18早于窗口 | 3Observed；窗口无新release。索引完整不等于332个diff全审；direct commits、未合并proposal、旧历史缺口仍未覆盖。 |
| SGLang | 385窗口merged PR，4页100/100/100/85；releases首100，latest v0.5.22=10/19 02:55:19Z在窗口内 | Accepted A15，另3个PR Observed与release组；0.5.22早于A15 merge，不能称已包含A15。 |
| TensorRT-LLM | 176窗口merged PR，2页100/76；releases首100 | 3PR合并为1Observed组，rc1并入release组；未宣称production-ready或完整GPU CI。 |

三runtime共 **893**；七库共 **1039** merged PR，各查询returned=total_count、incomplete=false。标题关键词优先筛选后核14个PR详情/完整文件索引；实际深读边界逐项写明，**不是1039个patch全审**。

## OpenAI / Anthropic / NVIDIA / DeepSeek Watch

| 厂商 | 实际检查来源 | Triage / 证据与缺口 |
|---|---|---|
| OpenAI | research / engineering / news 三个官方目录；10/19低资源语言、10/21生物agents与广告服务、10/22 Decisions 四篇正文 | **Accepted A19**；其余三篇Observed。生产服务机制优先于榜单与公司新闻；未声称所有API文档/产品changelog全扫。 |
| Anthropic | research / engineering / news 三目录；10/22 formal verification正文、Theorem公开证明和semantics README | **Accepted A18**；10/20蛋白设计仅目录核查，Observe；engineering最新可见Scion10/13为旧项。公开证明库仅研究成果子集，未运行checker。 |
| NVIDIA | Technical Blog、Newsroom目录；GDS、Mamba-2、OpenShell、Sentry正文；Cosmos/H-Dreams/Microduck/SimReady由具身分栏核查 | **Accepted A16**；Mamba与agent runtime治理Observed。模型/资产发布与可用训练路径分开；未全面审查GTC所有硬件及合作新闻。 |
| DeepSeek | [官方API更新](https://api-docs.deepseek.com/updates/) + [官方HF模型metadata](https://huggingface.co/api/models?author=deepseek-ai&sort=lastModified&direction=-1&limit=20) | **Not found in checked sources**：API最新可见10/17；HF按lastModified倒序首20项，最新Flash-1017为10/17 05:46:55Z，原Flash10/16。上轮Search/Fetch与Flash-1017不重复Accepted。只说明已查目录未发现窗口内更新，不代表所有权重文件/论坛全核。 |

## Hugging Face Watch

实际检查 [HF Blog](https://huggingface.co/blog) 主列表与可见社区卡片。主列表可见本窗口及之前日期，社区卡片为精选而非全量；AWS/NVIDIA/Microsoft发布是**厂商作者社区文章**，不一概当HF官方团队文章。HF团队 TorchNeuron/TorchTPU 与 train-agents 单独标注。Blog和下表代码线索按Source ID去重。

| 来源 | Triage / 实际检查 |
|---|---|
| Blog / TorchNeuron | A17关联HF官方介绍与AWS10/21 profiling正文，后者配套脚本公开但未执行；HF10/19介绍时间字段为00:00占位，本组新增日期以PyTorch10/19 12:45:23Z主文核定。 |
| Blog / TorchTPU | O-ROOT7 Observe：官方团队原文揭示原生backend/compile/Pallas，明确测试并非全绿；不能由示例推全部模型已可运行。 |
| Blog / RL | Microsoft ThinkingBox训练篇、HF train-agents课程见RL Observed，旧benchmark不重复计新信号；课程入口不是新框架性能证据。 |
| Blog / Embodied | NVIDIA Cosmos3 H-Dreams、Microduck sim-to-real、SimReady Central见具身Observed；分别检查动作条件、评估与资产/版本依赖。 |
| TRL | 本轮40 merged PR与release元数据，完整判断见RL分栏；LeRobot42 merged PR与release全历史见具身分栏。 |
| Transformers / Accelerate / PEFT / Kernels | 下表保留实际PR/release/docs核查范围；动态main docs不能充当新发布的时间证据。 |

| Library | Actual coverage | Decision / gaps |
|---|---|---|
| Transformers | 112窗口merged PR，2页；releases首100；main AttentionInterface docs | Accepted A14；FP8 cache和5.20.2 Observed。动态main docs是接口核查，不是新增事件日期证据。 |
| Accelerate | 3窗口merged PR；releases首100，latest1.15.0=9/9；main FSDP guide | 无新release；3标题为DeepSpeed zero-clipping、AlignDevicesHook compile、next-token/accumulation weighting，未深入diff，不升级Accepted；FSDP docs仅背景接口核查。 |
| PEFT | 23窗口merged PR；releases首100，latest0.21.2=10/1；LoRA docs | #3963 Observe；窗口无新release；其他adapter修复只标题筛选。docs导航stable标签与release API不同，版本以发布记录为准。 |
| Kernels | 8窗口merged PR；releases首100，latest0.17.2=10/2；main index/docs目录 | 无新release；有TPU builder、stable ABI shim/ABI符号与安全依赖更新，未深入实现，不升级Accepted；并未运行kernel下载/build。 |

## Observed 明细

保留待核问题与判断依据，不因来源显眼或代码公开就自动进入Accepted。

### 具身：17组

| ID / 来源 | Reason、已核证据、边界与下一步 |
|---|---|
| EO1 · [PaceVLA](https://arxiv.org/abs/2610.25961v1) | 10/19 v1 metadata/abstract/HTML已取；多个时间粒度decoder与context router区分粗执行和精细接触，值得补temporal action表示。尚未完整核量化定义、任务耗时/成功率与训练预算，Next核切换时的动作连续性和控制周期，不能称普遍实时提速。 |
| EO2 · [HAF-VLA](https://arxiv.org/abs/2610.26584v1) / [CW-WAM](https://arxiv.org/abs/2610.30117v1) | 分别10/19与10/21；两篇metadata/abstract/HTML已核。前者分阶段whole-body denoising和cross-stage KV条件，后者联合motion/wrench预测接外层admittance；是不同动作接口，不能混为一种“层级控制”。Next读完整ablation与控制频率，拆模型输出与低层controller贡献。 |
| EO3 · [SuperMimic](https://arxiv.org/abs/2610.28480v1) / [DiffDreamer](https://arxiv.org/abs/2610.31617v1) | 10/20、10/22；metadata/abstract核验，SuperMimic HTML404未读PDF，DiffDreamer HTML已取。共享可微动力学与解析gradient有训练价值，但前者大幅GPU-hour数字暂不引用；后者分开环境latent和显式机体动力学。Next核算力/数据/return对照与model bias，非已复现提速。 |
| EO4 · [Physical Canvas](https://arxiv.org/abs/2610.28613v1) / [DRILL](https://arxiv.org/abs/2610.30814v1) | 10/20、10/22；两篇metadata/abstract/HTML核验；前者visuo-tactile一致性+reward监督，后者以imagined update后的policy return训练world model。能补“画质≠policy utility”，但未完整审reward训练/评价泄漏、meta-gradient预算与真实收益，Next优先核独立real-world验证和同budget对照。 |
| EO5 · [UniWALM](https://arxiv.org/abs/2610.28626v1) | 10/20 metadata/abstract/HTML已核；video/action/language同步生成，下一步再消费text memory，另有action correction目标。Next拆language objective、memory loop、FMC三项独立消融；不把单frame输入等同没有历史状态。 |
| EO6 · [Post-hoc Epistemic Uncertainty](https://arxiv.org/abs/2610.29730v1) | 10/21 metadata/abstract/HTML已核；GP conditional-variance head从冻结policy embedding估计coverage而不改policy。论文主动区分task间coverage与task内failure排序值得读；尚未核kernel/参考集大小、OOD protocol与supervisor训练暴露，Next核线上referral和误报成本。 |
| EO7 · [Echo](https://arxiv.org/abs/2610.31569v1) / [Progress Probes P²](https://arxiv.org/abs/2610.31613v1) | 均10/22；metadata/abstract/HTML已核。前者把stop/failure signal加入action chunk联合denoise，后者用内部activation的linear progress probe监测；不能把两者等同failure recovery。Next核状态监督构造、阈值calibration、错误停机/漏报和任务外泛化。 |
| EO8 · [LeRobot sampled stats #4949](https://github.com/huggingface/lerobot/pull/4949) | 10/21 main；完整两文件diff已读。episode image/video统计用所代表总帧数加权，不再以采样帧数加权，真实PNG+JSON round-trip回归有解析oracle。旧保存stats不迁移，append会混合旧新语义，recompute_stats也不能据此声称可修图像stats。作为数据账本窄修复观察，Next核metadata迁移与估计方差。 |
| EO9 · [LeRobot noise #4813](https://github.com/huggingface/lerobot/pull/4813) / [device #4856](https://github.com/huggingface/lerobot/pull/4856) / [SmolVLA uint8 #4292](https://github.com/huggingface/lerobot/pull/4292) | 正文、文件列表与测试范围已核，非所有生产patch全审；分别是外部噪声注入、cuda:index round-trip、resize前uint8→[0,1]。DDPM后续步骤仍有随机噪声，不能把起始noise接口称全局确定性；设备tests含mock，非真GPU证明。Next建立raw input→processor→sampler→action parity，保留各policy边界。 |
| EO10 · [LeRobot VLA-JEPA RoPE #4835](https://github.com/huggingface/lerobot/pull/4835) | 10/22 main；正文及两文件diff范围已核。adjacent-pair rotation与频率layout匹配影响training-only world predictor；shape相同也改变旧checkpoint输出。作者25-case harness未加入repo，不当成新CI回归；Next核旧checkpoint行为和可持续测试，不宣称模型质量提升。 |
| EO11 · [LeRobot Wall-OSS-0.5 #4200](https://github.com/huggingface/lerobot/pull/4200) | 10/22 main；正文与13文件完整列表已核，未完整审990行model和654行processor。native routed text/action experts、q01/q99与recipe supervision有模型接口价值；限定canonical26D，parity数字是作者RTX5090指定输入，Next核joint text/action mask与checkpoint转换，不能称任意机器人可部署。 |
| EO12 · [Isaac Lab v3.0.0](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0)；[#8765](https://github.com/isaac-sim/IsaacLab/pull/8765)、[#8773](https://github.com/isaac-sim/IsaacLab/pull/8773)、[#8782](https://github.com/isaac-sim/IsaacLab/pull/8782)、[#8681](https://github.com/isaac-sim/IsaacLab/pull/8681) | **正式GA于10/20 02:05:51 UTC发布**，与上一轮仅EA的时间点不同。release全文已核：multi-backend/task API、Warp/PyTorch显式view、Kit-less和迁移契约；后续四PR主要是shared binding、teleop/native frame、静态ray target环境归属，详情/完整文件列表已核，未全面diff审计。release不是各后台物理等价保证；#8765微基准有启动成本与单target回退，非training FPS。Next固定tag做ordering/contact/reset/pose parity，再选后台。 |
| EO13 · [GR00T #797](https://github.com/NVIDIA/Isaac-GR00T/pull/797) | 10/22 main；四文件diff范围已核，是部署文档和tests调整，延续上轮#790，不是新N1.7首发或新量化优化。Next按#790既定输入/工件/完整policy-call验收，不从PR标题重复接受。 |
| EO14 · [NVIDIA Cosmos tools](https://blogs.nvidia.com/blog/cosmos-open-train-evaluate-deploy-world-models/) | Pranjali Joshi，10/21官方公告全文已核；CDF blueprints、Framework适配、Nano simulation variants及NIM交付分开陈述。没有逐审cosmos-framework/data-factory diff和配套benchmark；“2×”“90%”等仅vendor/partner口径且条件不同，不进入趋势定量结论。Next核action-conditioned闭环示例、数据处理契约和版本固定。 |
| EO15 · [Cosmos3-H-Dreams](https://huggingface.co/blog/nvidia/cosmos3-h-dreams) | **10/20 NVIDIA作者Community Article**，Lukas Zbinden、Javier Gamazo、Nigel Nelson、Mostafa Toloui、Sean Huver。文章+官方repo README已核：28D动作、10Hz对齐、bidirectional teacher→causal student、自生成context训练，2步是另训checkpoint，FadeMem管理KV。具备action-conditioned机制，非仅视频美化；但临床力学/任务验证仍为future work，未核训练recipe/weights全链路与实时benchmark。Next核action约定、teacher/student交付版本和控制误差，不能称临床可用。 |
| EO16 · [SimReady Central](https://huggingface.co/blog/nvidia/simready-central-assets-robotics-ai-factories) | 10/21 NVIDIA作者Community Article，John Kosnik、Frank DeLise、Pomi Lee；全文+Foundation README已核。profile将碰撞/关节/材料/语义能力显式化；内容通过某profile不证明目标solver物理准确。Next选一个资产核profile版本、依赖/单位/运行时契约和真参数来源；未下载2,000资产或执行验证器。 |
| EO17 · [Microduck Isaac Lab](https://huggingface.co/blog/nvidia/microduck-isaac-lab-sim-to-real) | 10/21 NVIDIA作者Community Article，Oluwaseun Doherty、Oyindamola Omotuyi、Richard AntRi31、Asier Arranz；web open失败后直接HTTP取得全文。教程明确BAM/backlash、domain randomization、61 observations→14 joint targets、ONNX handoff。16GPU吞吐是作者报告，未核硬件/benchmark原始记录与真机统计；Next固定GA tag/任务配置核导出normalization、ordering、action scale，未执行文中安装/部署命令。 |

### RL / Training：19组

统一Decision=Observe；不计Accepted，不等于逐行完整代码审查。

| 组 | Sources | 机制、证据边界与Next |
|---|---|---|
| RO1 AReaL checkpoint progress | [AReaL #1801](https://github.com/areal-project/AReaL/pull/1801) / [AReaL #1802](https://github.com/areal-project/AReaL/pull/1802) | checkpoint/recovery；scheduler state补存/后加载与同step共享Saver频率决策是两个机制；已读元数据/文件与小diff，不合成一个恢复保证。后续核完整resume多步LR曲线和actor/critic joint checkpoint。 |
| RO2 AReaL consumer cache / HCCL P2P | [AReaL #1800](https://github.com/areal-project/AReaL/pull/1800) / [AReaL #1807](https://github.com/areal-project/AReaL/pull/1807) | data/trajectory path / weight sync；teacher RTensor本地cache排空、按directed peer分轮HCCL各有具体修复；元数据和文件范围已核，后者硬件故障仅作者复现，优先追真实AWEX多节点与错误drain。 |
| RO3 slime Straw rewind | [slime #2482](https://github.com/THUDM/slime/pull/2482) | checkpoint/recovery / data/trajectory path；读rewind与recovery实际diff：保留queue identity、重开epoch fence旧lease、prompt恢复后重新生成，删除未来rollout索引/owner并保留model payload。队列恢复失败退到可读dataset cursor或0会改变数据重复语义；作者两节点测试未独立执行。 |
| RO4 slime isolated SGLang runtime | [slime #2484](https://github.com/THUDM/slime/pull/2484) | inference backend；独立Python runtime和启动/控制路径有实际文件，未深核隔离依赖与端口/进程退出，暂Observe。 |
| RO5 verl distillation objective and CP alignment | [verl #7855](https://github.com/verl-project/verl/pull/7855) / [verl #8325](https://github.com/verl-project/verl/pull/8325) / [verl #8332](https://github.com/verl-project/verl/pull/8332) | training / data/trajectory path；tail-aware top-k KL将剩余mass作为第K+1类，另两项对齐FP8 THD padding及teacher top-k与student CP split。三项各自机制不可混称一次修复；元数据/测试范围已核，需验证TP梯度与packing/CP组合。 |
| RO6 verl MXFP8 padding / raw mismatch telemetry | [verl #8127](https://github.com/verl-project/verl/pull/8127) / [verl #8025](https://github.com/verl-project/verl/pull/8025) | training；MXFP8 per-sequence alignment与raw logprob mismatch metric分别补齐低精度训练和观测，不独立宣称性能或数值对齐。前者后续#8325重构，迁移应看最终树。 |
| RO7 NeMo speculative draft co-training | [RL #3960](https://github.com/NVIDIA-NeMo/RL/pull/3960) | training / rollout / weight sync / checkpoint/recovery；Automodel携带DSpark/DFlash/EAGLE3训练与draft.* refit、checkpoint metadata/CP rescale；86-file清单/正文已核，未完整读实现；不能由短canary宣称端到端speculative收益。 |
| RO8 NeMo cross-tokenizer offset alignment | [RL #3286](https://github.com/NVIDIA-NeMo/RL/pull/3286) | data/trajectory path / training；offset cluster替代DP对齐并扩展chat/native template，span/eot/gold-loss partition成为契约；45-file范围与测试清单已核，需进一步审char boundary和不同template的可比性。 |
| RO9 NeMo object-store commit / state cleanup | [RL #4743](https://github.com/NVIDIA-NeMo/RL/pull/4743) / [RL #4433](https://github.com/NVIDIA-NeMo/RL/pull/4433) / [RL #4511](https://github.com/NVIDIA-NeMo/RL/pull/4511) | checkpoint/recovery / weight sync；MSC对象存储用_SUCCESS在async shard完成后publish；safe sleep丢rollout权重后标stale强制refit；SFT恢复初始化/optimizer。元数据和文件已核，各为不同状态机，需分别验证故障注入；不把作者host OOM节省当通用收益。 |
| RO10 NeMo refit stream / jagged A18 follow-ups | [RL #4720](https://github.com/NVIDIA-NeMo/RL/pull/4720) / [RL #4842](https://github.com/NVIDIA-NeMo/RL/pull/4842) | weight sync / data/trajectory path；packed buffer跨stream携带gather依赖，nested routes留到microbatch才pad，均有针对性回归；前者#4866是release分支backport不重复计，后者不重收10/19 A18机制。 |
| RO11 Megatron recompute / optimizer memory | [Megatron-LM #6429](https://github.com/NVIDIA/Megatron-LM/pull/6429) / [Megatron-LM #8233](https://github.com/NVIDIA/Megatron-LM/pull/8233) / [Megatron-LM #8232](https://github.com/NVIDIA/Megatron-LM/pull/8232) / [Megatron-LM #8313](https://github.com/NVIDIA/Megatron-LM/pull/8313) / [Megatron-LM #7394](https://github.com/NVIDIA/Megatron-LM/pull/7394) | training / checkpoint/recovery；EP A2A overlap按layer segment recompute；dev分支chunked optimizer限制chain staging和optimizer-owned master窗口；MFSDP Muon shard momentum/owner orthogonalization；global quantile balancing。已核metadata/file inventory，未深审大diff，不能宣称main都可用或给复现收益。 |
| RO12 DeepSpeed v0.19.8 and new kernels | [DeepSpeed #8732](https://github.com/deepspeedai/DeepSpeed/pull/8732) / [DeepSpeed #8838](https://github.com/deepspeedai/DeepSpeed/pull/8838) / [DeepSpeed #8839](https://github.com/deepspeedai/DeepSpeed/pull/8839) | training / rollout / inference backend；CUDA graph continuous rollout固定physical slots并受greedy/layout限制；vocab-parallel Triton复用MAX/SUM collectives、融合本地entropy/KL；prefill仅last-position logits。元数据/正文/文件核查，未全审或运行。v0.19.8 release是独立事件，不默认包含本窗口所有PR。 |
| RO13 TRL tree packing follow-up | [trl #7793](https://github.com/huggingface/trl/pull/7793) | training / scheduler；按TreeRow实际增量cost将样本spill到不提高max-row load的空余row；实际小diff/测试已读，属于10/19 tree packing后续，不重复接受新tree机制。 |

### HF blogs

- **Observed / Observe**：[ThinkingBox新训练篇](https://huggingface.co/blog/microsoft/thinkingbox-training)，2026-10-22，Tuhin Kundu，Microsoft署名并说明Microsoft/HF联合。新披露的是固定离线policy-relative筛选、独立episode后端、系统错误refill、共享训练/评估解释器与verl训练栈；数据库终态reward及旧benchmark不重复接受。187任务训练池私有，公开sample不是报告数据；作者20/20与pass@1上升同时pass@20下降，成本不含训练/probe，不宣称可重现或绝对可靠性。Next：核公开pipeline schema和失败分类。
- **Observed / Observe**：[HF课程导航](https://huggingface.co/blog/huggingface/train-agents)，2026-10-20，Ben Burtenshaw / Sergio Paniego / merve，页面为Community Article但作者属HF团队。SFT→distillation→RL→environment→harness的资源导航，未披露独立新增系统机制；可作学习路线参考，不自动推广现有材料。

### Other papers

- **Observed / Observe**：[HarnessRL](https://arxiv.org/abs/2610.26426v1)，10/19；已核citation metadata/abstract且取得方法HTML，但尚未深核online harness proposer/critic开销和奖励漂移验收。
- **Observed / Observe**：[Adaptive Decoding](https://arxiv.org/abs/2610.25860v1)，10/19；已核primary metadata/abstract，support schedule比较噪声与test replay相同点估计不支持稳定系统收益；不把validation小幅改善泛化为算力节省。
- **Observed / Observe**：[Harness Scaling](https://arxiv.org/abs/2610.29458v1)，10/21；已核primary metadata/abstract，factorized harness diversity可能有agent学习价值，但未深核训练成本与black-box迁移边界。
- **Observed / Observe**：[Measure First, Then Allocate](https://arxiv.org/abs/2610.28785v1)，10/20；已核primary metadata/abstract，BRACE预测mixed-reward组的预算价值值得跟进；需把historical bank成本与新增rollout分开，尚未读全文。

### Systems / Inference：10组

| Source ID / 来源 | 实际核查、Decision / Reason | 下一步与边界 |
|---|---|---|
| `github:vllm-project/vllm:pull:56430` · [FT KV leak](https://github.com/vllm-project/vllm/pull/56430) | Observe；10/22 merge；10-file索引，实际核worker exception携带KV output、aggregator salvage、batch queue drain和测试。失败采样不应丢掉transfer完成事件。 | `kv_cache_events`在salvage中明确丢弃，避免污染live quorum；测试名称/部分描述仍称salvaged votes，需要以实际实现为准。补完整故障恢复和真实PD验证。 |
| `github:vllm-project/vllm:pull:59635` · [NIXL pull sleep](https://github.com/vllm-project/vllm/pull/59635) | Observe；10/19 merge；核正文、14-file列表与接口范围。sleep/remap可能保留VA但改变物理注册，需重新注册并传播epoch。 | 描述包含父PR/早期revision，未逐条映射全部正文到当前patch；仅UCX pull范围，push仍不支持，UCX环境设置进程级副作用需验。 |
| `github:vllm-project/vllm:pull:52670` · [pluggable KV dtype](https://github.com/vllm-project/vllm/pull/52670) | Observe；10/20 merge；50-file列表与config/tests存在性已核；KV dtype走插件化抽象。 | PR无实测结果；未全审50文件，不据接口扩展宣称新精度可直接部署。 |
| `github:sgl-project/sglang:pull:45559` · [linker offload ordering](https://github.com/sgl-project/sglang/pull/45559) | Observe；10/21 merge；7文件全diff。external linker offload同时等待scheduler页索引事件与forward写完事件，延续10/19 HiCache #44800的不同代码路径。 | 作者由代码阅读发现，未在main复现；单测为event/调用顺序mock，非GPU race reproduction。读两个路径可避免把一次HiCache修复当全缓存安全。 |
| `github:sgl-project/sglang:pull:42557` · [paged experts](https://github.com/sgl-project/sglang/pull/42557) | Observe；10/19 merge；23文件索引和正文，host保存完整expert，GPU保持K-slot working set，只搬路由命中的缺页。 | 首版single-GPU、指定BF16/FP8/GPTQ，不含TP/EP/NVFP4；prefill与高并发/小K可劣化。未全审实现与benchmark，不抄headline倍数。 |
| `github:sgl-project/sglang:pull:44280` · [KVShard DSA](https://github.com/sgl-project/sglang/pull/44280) | Observe；10/22 merge；6-file索引、正文；GLM-5.2 page-sharded DSA区分logical page和scratch地址。 | 与UlaCache主题相关但不能未经核验当其代码；正文accuracy/speed栏未填，未完整patch审阅。 |
| `github:NVIDIA/TensorRT-LLM:pull:19710` + `:pull:20108` + `:pull:20117` · [generation reuse](https://github.com/NVIDIA/TensorRT-LLM/pull/19710)、[zero-copy](https://github.com/NVIDIA/TensorRT-LLM/pull/20108)、[backend contract](https://github.com/NVIDIA/TensorRT-LLM/pull/20117) | 一组Observe；10/20–21 merge；官方正文和11/24/24文件索引。围绕KVCM2增量稳定prefix提交、物理allocation边界注册、backend/native类型边界。 | 未全diff审阅，不把自动生成review当验收；#19710正文自动review称只有import变化，与实际11文件列表冲突，按真实patch后续核。#20108默认回到zero-copy是breaking变化，不假定rc1包含所有main merge。 |
| `github:huggingface/transformers:pull:48677` · [FP8 QuantizedCache](https://github.com/huggingface/transformers/pull/48677) | Observe；10/22 merge；6-file列表/正文，固定首update scale、连续FP8存储、attention前全量反量化。 | 缓存变小不等于峰值显存/延迟同步改善；正文称附benchmark脚本，但最终文件列表未包含该脚本，本轮未接受其数字为验证结果。 |
| `github:huggingface/peft:pull:3963` · [gather adapters only](https://github.com/huggingface/peft/pull/3963) | Observe；10/19 merge；2文件全diff，DTensor TP保存先筛需要的adapter/embedding再gather，避免每rank物化整个base。 | test用spy检查gather keys并断言返回非DTensor；未本地执行。只覆盖指定DTensor分支，不泛化全部TP保存路径。 |
| `github:NVIDIA/TensorRT-LLM:release:v1.4.0rc1` + `github:huggingface/transformers:release:v5.20.2` + `github:sgl-project/sglang:release:v0.5.22` · [rc1](https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.4.0rc1)、[5.20.2](https://github.com/huggingface/transformers/releases/tag/v5.20.2)、[0.5.22](https://github.com/sgl-project/sglang/releases/tag/v0.5.22) | 一组Observe；发布时间分别10/22 01:38:13Z、10/19 17:45:37Z、10/19 02:55:19Z；官方release metadata/notes。rc1仍prerelease；5.20.2含#49688和generation输入dtype修复；SGLang0.5.22为stable、Python最低3.11，notes含HiSparse spec和hybrid Mamba提交边界修复等旧merge汇总。 | release聚合旧提交，不按每个release-note条目算窗口新merge；#49710不在5.20.2清单。 |

### 厂商 / HF：8组

以下 **8组**，均Decision=Observe，不计Accepted；其余标题筛选后拒绝的公司/商业/垂直应用新闻另记末行。

| ID / 来源 | Evidence / Reason | Next / 边界 |
|---|---|---|
| O-ROOT1 · [OpenAI biological agents](https://openai.com/index/scaling-ai-research-for-complex-biological-data/)（10/21） | 正文披露按gene perturbation分工、汇总reasoning atlas，并比较单agent与并行agents；与任务分解/评估有关。 | 论文与资源预算细节未深核，领域发现需实验检验；不把“32分钟”当通用分布式加速比。补读原始实验再决定。 |
| O-ROOT2 · [OpenAI低资源语言](https://openai.com/index/low-resource-programming-languages/)（10/19） | 正文以SQLite数据库状态作跨语言oracle、迁移用输出byte parity，提供两类eval概念。 | 未审公开test suite、训练机制未披露；不按模型分数自动升优先级。后续用于对照RL环境reward设计。 |
| O-ROOT3 · [Decisions API](https://openai.com/index/decisions-api/)（10/22） | 正文披露bounded answers、复用上下文cache及低时延接口；属于接口产品发布。 | 未独立核延迟/质量/容量，接口约束未全读；不照抄最高倍数，也不据此宣称通用推理新架构。 |
| O-ROOT4 · [Anthropic蛋白设计](https://www.anthropic.com/research/claude-scorpion-toxin-binder-design)（10/20） | research目录确认新文；本轮仅目录筛选，非正文核验。 | 主要是生命科学应用；是否有可迁移agent系统机制尚待正文，不能写已全面读完。 |
| O-ROOT5 · [NVIDIA Mamba-2 CUDA Tile](https://developer.nvidia.com/blog/accelerating-mamba-2-with-high-performance-forward-pass-kernels-in-nvidia-cuda-tile/)（10/19） | 正文给chunk cumsum→chunk state→output三kernel及融合中间量，适合学习SSM计算与数据移动。 | 本轮核原文未运行kernel/核全benchmark；这是forward tutorial，不把局部kernel提升当完整训练/serving收益。公式符号与维度还需对照代码后再回写教学章。 |
| O-ROOT6 · [OpenShell](https://developer.nvidia.com/blog/how-nvidia-openshell-reference-architecture-secures-and-governs-ai-agent-workloads-at-scale/) + [Sentry](https://developer.nvidia.com/blog/inside-nvidia-sentry-secure-ai-agents-at-runtime/)（10/21） | 两篇正文区分gateway/supervisor/sandbox与BlueField独立policy/observability域；有实际部署边界价值。 | reference架构与硬件性能数字未独立验证、未审实现；不把policy proof解释为任意agent行为安全。后续核故障、凭据和旁路约束。 |
| O-ROOT7 · [HF TorchTPU](https://huggingface.co/blog/hf-torchtpu)（10/20） | 官方团队正文：native device、eager/compile、Pallas kernels和TP；明确有未通过测试及未完成nightly CI。 | 本轮未审TorchTPU实现；示例吞吐限定模型/batch/chips。与A17共同提示迁移成本变化，但不把beta兼容性等同生产成熟度。 |
| O-ROOT8 · [HF MPS](https://huggingface.co/blog/mps-pytorch-2-15) + [Dharma-AI VocabX](https://huggingface.co/blog/Dharma-AI/vocabx-vocabulary-specialization)（10/20–21） | HF主目录可见，前者后端支持、后者词表专门化；本轮只做目录筛选。 | 正文/实现未核，当前优先具身与大规模系统；不接受标题带来的性能推断。若后端/词表成为实际问题再读。 |

Rejected / Ignore：目录中的融资、合作、地区上线、岗位介绍与缺少当前模型/系统问题的垂直应用发布；不因来自核心厂商或登上HF列表而自动收录，也不将它们计入上述Observed。

## 晚发现材料、缺口与下一步

- [10月独立补录](backfill/2026-10.md#oct23-late)：CusADi2（10/18）、BaGuaLu-2（10/16）、PlaneZip（10/18），均Observe/NEW，补低延迟动力学、动态MoE显存和activation offload问题。尚未完成方法/实验深核，不计本轮新Accepted。
- Systems目录另发现[KVLearn](https://arxiv.org/abs/2610.28637v1)、[PolyGraph](https://arxiv.org/abs/2610.23861v1)：ID前缀2610与页面citation/history的8/10、8/14日期冲突，前者HTML404。保留审计候选，待澄清后按原始月份归档，不据ID猜日期或当10月新文。
- **旧缺口继续保留**：arXiv发现从2026-09-22 16:52:05上海回扫；GitHub direct commits、历史外围PR、未完整核查的release history从2026-09-22 17:06:16回扫。当前窗口查询完整不关闭历史缺口。
- 当前限制：SuperMimic HTML不可取；HF社区卡片非全量，Anthropic蛋白/HF MPS/VocabX只目录筛选；大型PR仅重点代码深核，有CI failure/skip或作者未做GPU验证。机器人真机泛化、触觉/灵巧操作的控制评价和world-model物理真实性仍有阅读缺口。
- 回写入口：[具身经验与动作信息](../../05-embodied-infra/topics/embodied_infra.md#oct23-learning)、[数据所有权与读路径](../../05-embodied-infra/topics/embodied_data_io.md#oct23-data-contract)、[推理容量与地址生命周期](../../03-inference-infra/topics/inference_state.md#oct23-capacity)、[RL更新与环境租约](../../04-rl-infra/topics/agentic_rl.md#oct23-update-contract)。选读进入原[共享P1](../reading_queue/P1.md#embodied-20261013)，不新建雷达或并行P0。

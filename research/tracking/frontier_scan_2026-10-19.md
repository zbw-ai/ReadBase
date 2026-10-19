# Frontier Scan · 2026-10-19

[研究雷达](README.md) · [扫描账本](scan_log.md) · [具身核心入口](../../05-embodied-infra/topics/embodied_infra.md) · [共享阅读队列](../reading_queue/P1.md#embodied-20261013)

## 本轮结论与选读

本轮 **19 组 Accepted、45 组 Observed**：具身模型/学习3、具身Infra3、RL/训练6、Systems/独立推理4、厂商机制与代码3。数量按 triage 组计算，联合PR保留各自 Source ID；同项出现在多个 Watch 只计一次。Observed 分为具身13、RL14、Systems10、厂商/HF8，后文可逐行核对。

**优先读 REKI（A1）→ T2-z（A2）→ GR00T 部署改动（A4）**：分别理解短历史、隐藏环境条件与模型执行成本。想接续已有大模型训练经验，再读 **Nemotron 3.5（A17）**，重点是 teacher/student/MTP 与推理状态如何共同设计。A3 用于反思监督目标；其余按当前数据、训练、服务问题选读，无需逐项精读。

**整体进展与趋势（本轮样本推断）**：具身工作正在把“记住历史”“适应环境”“预测动作”分成不同机制和评测问题；Infra 同时从局部提速转向完整状态生命周期：内存仍由谁持有、CPU何时能读、动作对应哪个关节、缓存何时真正就绪。更高吞吐可能来自经验复用或共享前缀，必须同步核训练目标、数据分母与恢复语义。这不是这些瓶颈已经解决的结论。

## 时间、范围与证据

- 固定事件窗口：**2026-10-14 14:20:25 → 2026-10-19 09:40:52 Asia/Shanghai**，即 `2026-10-14T06:20:25Z → 2026-10-19T01:40:52Z`。固定查询终点后继续取证；核验结束时间另记 manifest，不拿它跨过未查询的事件。下一轮从 **09:40:52** 回看去重。
- First seen：一般为10/19本轮核验；A7 的提案早在9/22 open快照出现，本次首次核验10/16合并。生命周期均为 **NEW**；本仓库没有运行上游代码、GPU benchmark、仿真或真机实验。
- GitHub窗口 merged PR 索引共 **958条**：六RL框架97、三推理runtime750、七具身仓111；各仓查询 returned=total、incomplete=false。标题全筛，重点项核实际diff/tests；**不是958个diff全审，也不是仓库所有文件审计**。core RL release全历史178条、具身67条；推理首100项/仓及root首30项/仓只宣称覆盖实际核查范围。
- [审计入口](audits/2026-10-19-frontier/README.md)保留元数据、页面哈希、文件索引、核查结论与限制，不镜像完整论文/博客/PR正文。arXiv RO/LG/CV 与 LG/CL/DC/AR 的重叠目录分别留证，**3713与3109不可相加当作全局独立论文数**；最新可见公告为10/16。
- 本轮没有新增必须独立补录的窗口前论文。A12 是10/15 TRL release事件，其tree packing PR在10/13已合并；A17为10/15公告，报告首页10/14但小时未知；A4不是N1.7首次发布。这些日期边界明确保留，不把旧材料伪装成新merge/新模型。

## Accepted

<a id="a1"></a>
### A1 · REKI：把短历史放进 action expert 的 register cache

- Source ID：`arxiv:2610.22178v1`；Type：paper + public repository；[论文](https://arxiv.org/abs/2610.22178v1) / [正文](https://arxiv.org/html/2610.22178v1) / [代码](https://github.com/kyutai-labs/reki)。Title：*REKI: Registering Knowledge in VLAs*；Authors：Juliette Marrie、Paul Pacaud、Adrien Ramanana Rahary、Shizhe Chen、Moritz Böhle。citation_title / 全部 citation_author / citation_date=`2026/10/15` 已核；v1 **2026-10-15 16:34:06 UTC**。
- First seen：2026-10-19；scan window：上述主窗口；Impact：High；Decision：**Read**；Learning track：Embodied / Inference / Training；Signal type：model；Status：NEW。
- Target question：仅看当前帧为什么会在两种姿态间反复振荡？短历史能否进入动作模型，而不重复计算历史图像的 VLM？
- Evidence：§3–4 以当前图像、语言和 robot state 预测 flow-matching action chunk；每帧 32 个 registers 读取本帧 VLM 输出与旧 registers，不读取 action token、跳过 adaptive normalization，因此不随 denoising time/state 改变；其 KV 在 Euler steps 之间复用，并保存前两帧。训练窗口内每帧都有动作监督，encoder 可批量处理；推理缓存形成滚动窗口。§6.2 用相同权重移除历史的对照支持“读取历史”贡献，不能把所有跨模型优势归给 memory。
- 模型/学习价值：将 observation aliasing 与长程推理区分；只需少量最近历史就可能消除反复动作。系统影响：register cache 是 episode 状态，官方 README 明确每 episode 调用 `policy.reset()`，相机角色、state 顺序和调用 cadence 也属于部署输入契约。
- 评估边界：§6.3 的 **21.8 ms/chunk** 是作者 H100、batch=1、两张 256×256 图像、compiled、含预处理与传输的本地调用测量；不是网络/机器人闭环延迟。§7 真机仅 UR5、五任务、每任务十次测试，部分结果给 partial credit；跨模型训练数据、chunk 长度不完全一致。更长时间的记忆仍待验证。公开 repo 和 README 已核（固定 SHA `12dd1e24acfc01c817e03ecd43f37cc5f9f6b2d1`），没有运行或声称复现。
- Reason：直接补短历史、action conditioning、可缓存计算与泛化评估的学习缺口。
- Related topics：[具身学习入口](../../05-embodied-infra/topics/embodied_infra.md)、action chunk、memory、推理 state；Next：共享 P1 具身阶段候选；画 register/action attention mask，验证 episode reset 与训练/推理窗口差异，不改原 P0。
- 一句话：让与 denoising 无关的 registers 保存最近观察，既避免反复编码旧帧，也帮助策略区分“刚从另一姿态到达”的相同视觉状态。

<a id="a2"></a>
### A2 · T2-z：交互先校准环境 latent，再比较动作后果

- Source ID：`arxiv:2610.22364v1`；Type：paper + public repository；[论文](https://arxiv.org/abs/2610.22364v1) / [正文](https://arxiv.org/html/2610.22364v1) / [代码](https://github.com/T2-z-latent-optimization/T2-z)。Title：*T2-z: Test-Time Optimization of Environment Latents for Robotic World Models*；Authors：Chenyuan Zhou、Yining Hong、Huang Huang、Yunfan Jiang、Wenlong Huang、David Held、Vitor Campagnolo Guizilini、Yue Wang、Li Fei-Fei、Jiajun Wu、Leonidas Guibas、Yejin Choi。三类 citation metadata 已核；citation_date=`2026/10/15`；v1 **2026-10-15 17:59:11 UTC**。
- First seen：2026-10-19；scan window：上述主窗口；Impact：High；Decision：**Read**；Learning track：Embodied；Signal type：algorithm；Status：NEW。
- Target question：同一可见场景下摩擦、质量或交互规则不同，world model 怎样利用少量真实交互修正 action-conditioned prediction？
- Evidence：§3 交替优化共享 world model 与每环境 latent，另一方固定；测试时冻结 world model，仅用 support interactions 的预测误差更新 32D latent，随后用于不同 query actions。未给环境因子类型或真值监督。§4 与附录 C 将视觉指标、物理预测误差、实际执行选择动作的成功率分开；不是只评视频观感。
- 模型/学习价值：把共享动态与环境特定信息拆开，区分直接优化 latent 和通过固定 history encoder 推断。系统影响（推断）：support/query 的环境身份和 latent 生命周期需绑定；环境变更后的缓存失效、探测动作选择与更新时间预算需要另验。
- 评估边界：七模拟、四真实任务；适配方法使用相同 support，query outcome 不提前暴露，但**每组候选动作保证至少有一个成功动作**，并非开放动作空间自主规划。部分误差单位跨任务不同，不照抄总平均“物理误差”为同一物理量。附录 E.4 作者在一 H200 的 Friction workload 上，40 次 latent 更新加五条 query 视频生成总计 **18.09 s**，不含加载/数据读取/写 MP4；因此不能当实时控制循环。代码 README 和固定 SHA `c72f051f5cffcb1f372a32bc488df273556d39b9` 已核，未运行。
- Reason：补上隐含环境、test-time adaptation 和 action-conditioned world-model 评价的核心缺口。
- Related topics：[具身学习入口](../../05-embodied-infra/topics/embodied_infra.md)、world model、adaptation、support/query；Next：先定义环境变化→新 support→latent 更新→query 的状态图，再比较动作选择质量与适配耗时；P1 候选。
- 一句话：固定共享 world model，用交互误差更新小型环境 latent，使后续动作预测适应看不见的环境条件，同时付出明确的测试时优化成本。

<a id="a3"></a>
### A3 · Supervision allocation：动作损失的样本份额不是固定答案

- Source ID：`arxiv:2610.21915v1`；Type：paper；[论文](https://arxiv.org/abs/2610.21915v1) / [正文](https://arxiv.org/html/2610.21915v1)。Title：*What Should Robot Policies Learn to Predict? Rethinking Supervision Targets in Vision-Language-Action Learning*；Authors：Haitao Shen、Hanyu Liu、Yihan Lin、Huiping Liu、Junbo Cui、Daoheng Li、Yang Li、Jing Zhang、Wenxin Li、Yuan Yao。三类 citation metadata 已核；citation_date=`2026/10/15`；v1 **2026-10-15 14:01:06 UTC**。
- First seen：2026-10-19；scan window：上述主窗口；Impact：High；Decision：**Read**；Learning track：Embodied / Training；Signal type：algorithm；Status：NEW。
- Target question：相同样本曝光预算下，继续做 action prediction，还是把部分样本用于 latent-action / future-vision 目标，更有助于策略学习？
- Evidence：§3 两流 MMDiT 的 sample router 让每个样本只计一个目标损失；比较三种 LA target 与三种 FV representation。FV route 仍输入干净 demonstration actions，推理只输出动作但两 expert 仍交互。§4 与附录对 FV 使用保留同 encoder/input/architecture 的 P100 对照，以区分目标分配与额外视觉输入；LIBERO 上 P25 都低于各自 P100，CALVIN 又有相反配置，不能推出辅助目标越多越好。
- 模型/学习价值：target representation、监督份额、source-matched control 都应是实验变量。系统影响（推断）：训练账本需同时记录 route 比例、目标构造及 action-label 使用；仅记录 action-loss 数量会低估监督。
- 评估边界：预算是 updates×effective batch 的 **sample occurrences**，不是等 FLOPs/等 wall-clock；“半数 action route”不等于只使用一半 action labels。四真机任务每项二十次、按组成动作计 partial credit，不可写成整任务成功率。正文未定位本方法官方代码；LARYBench 链接是参考资料，不能当该方法实现。没有本地复现。
- Reason：直接改变“动作输出就应主要训练动作损失”的默认判断，同时披露控制和反例。
- Related topics：[具身学习入口](../../05-embodied-infra/topics/embodied_infra.md)、数据配方、objective、训练预算；Next：P1 候选；建立 target×route×budget 矩阵，先加 source-matched P100 和耗时/显存账本。
- 一句话：用显式 router 比较动作与辅助预测目标的预算分配，并分别控制额外输入与计算成本，才能判断收益来自何处。

<a id="a4"></a>
### A4 · GR00T #790：缓存、量化与计时必须落到同一部署工件

- Source ID：`github:NVIDIA/Isaac-GR00T:pull:790`；Type：official merged PR；[PR](https://github.com/NVIDIA/Isaac-GR00T/pull/790) / [固定 diff](https://github.com/NVIDIA/Isaac-GR00T/commit/dac54b313160c201c450f97df0fd74ebd379c892)。Title：*Update GR00T N1.7*；Author：gr00t-agent[bot]；合并 **2026-10-15 21:46:25 UTC**，base `main`；merge SHA `dac54b313160c201c450f97df0fd74ebd379c892`。
- First seen：2026-10-19；scan window：上述主窗口；Impact：High；Decision：**Read**；Learning track：Embodied / Inference / Systems；Signal type：system；Status：NEW。
- Target question：多步 denoising 中不变的 cross-attention K/V 如何复用，量化/engine/runtime 的 precision 和 shape 契约怎样保持一致？
- Evidence：36 文件索引已核；重点读 optimized exporter/runtime、calibration/config、benchmark 与回归 diff。新增 cross-KV cache、timestep modulation 预计算、projection fusion 与 CUDA graph；先从**未量化** policy 捕获所有 component calibration inputs，避免逐个 fake-quant 原地修改污染后续校准。shared precision recipe/metadata 同时驱动导出、构建和 runtime 文件解析；缺少量化 engine 不默默替换成其他精度。graph 只捕获第一输入 signature，其他 signature 走临时 buffer eager path，拒绝 RTC action overlap。
- Reason / 价值：这是部署执行路径更新，**不是 N1.7 本轮首次发布**；release API 的 `n1.7-release` 为 2026-04-18。
- 数字与边界：新表改用同步完整 policy call，包含预处理与动作解码、排除 RPC；旧组件时间相加表被移除。作者 Thor120W 特定输入测试报告 baseline TRT 77.6 ms、optimized mixed NVFP4 35.9 ms；冻结 NPZ 输入未随仓库提供，无法声称 identical-input 复现。0.99 final-action cosine gate 不证明闭环 task success。batch baked static；默认 baseline/unquantized，RTC overlap 不支持。公开 head checks 只返回 lint success，不能说 GPU CI 全绿。
- Related topics：[具身学习入口](../../05-embodied-infra/topics/embodied_infra.md)、[推理系统](../../03-inference-infra/README.md)、precision、CUDA graph、latency；Next：固定 checkpoint/输入/功耗/shape 比较完整调用，再测闭环任务、首请求与 signature 切换；不运行上游安装脚本。
- 一句话：把跨 denoising 的不变计算缓存下来，并用共同 metadata 约束量化工件和 runtime；延迟收益仍需完整调用与闭环质量两种验收。

<a id="a5"></a>
### A5 · LeRobot #4903：有界视频缓存不等于有界 worker RSS

- Source ID：`github:huggingface/lerobot:pull:4903`；Type：official merged PR；[PR](https://github.com/huggingface/lerobot/pull/4903) / [固定提交](https://github.com/huggingface/lerobot/commit/8975047fbb76e7fef3f482988ceccc17e2102a90)。Title：*fix(streaming): release HTTP response bodies without waiting for cyclic GC*；Author：pkooij；合并 **2026-10-15 10:19:31 UTC**，base `main`；merge SHA `8975047fbb76e7fef3f482988ceccc17e2102a90`。
- First seen：2026-10-19；scan window：上述主窗口；Impact：High；Decision：**Read**；Learning track：Embodied / Training / Systems；Signal type：system；Status：NEW。
- Target question：compressed-byte budget 与 decoder cap 都守住时，为什么多 worker streaming RSS 仍继续上升？
- Evidence：四文件 diff 已核。`response.read()` 把 body 放入 HTTPX response/stream cycle，关闭连接后仍可能保留第二份视频；改为 `b''.join(response.iter_bytes())`，保留原 retry/context close。回归关闭 cyclic GC，覆盖单 range/四子 range，检查返回字节和 response body retained bytes。不要把 join 当零复制。
- Reason / 学习价值：延续上轮 streaming worker 线索，区分业务 cache、Python ownership cycle、native live allocation 与 allocator-free heap。
- 边界：作者完整原支持栈负载仍在 **100.87 GB** guard 失败；HTTP 修复仅消除一个机制。Torch2.14.1/TorchCodec0.17 的单独 30 分钟实验较稳定，但**超出此 PR 恢复后的依赖范围**，并非本 PR 已升级/验证的 locked runtime；同时 host load 改变，不能算受控速度提升。最新公开 head Fast Pytest/lint success，多个 sim-eval skipped；作者历史全量测试不自动验证当前 head。没有本地运行。
- Related topics：[具身数据 I/O](../../05-embodied-infra/topics/embodied_data_io.md)、worker lifetime、memory accounting；Next：对同负载分别记录 compressed bytes/live decoders/retained response body/RSS/native allocator，按实际依赖版本做长时验证。
- 一句话：逐块读取避免响应对象在 GC 前额外持有视频副本，但 worker RSS 仍受其他 native/allocator 行为影响，不能用一个 cache 上限证明整体内存稳定。

<a id="a6"></a>
### A6 · Isaac Lab #8480：同名 physx 顺序必须成为相同的张量布局

- Source ID：`github:isaac-sim/IsaacLab:pull:8480`；Type：official merged PR；[PR](https://github.com/isaac-sim/IsaacLab/pull/8480) / [固定提交](https://github.com/isaac-sim/IsaacLab/commit/c8a06a87c7de8c9bcfd44f1bb511f927e48a87f0)。Title：*[Core] Make "physx" mean the OVPhysX order on every backend*；Author：AntoineRichard；合并 **2026-10-15 14:48:39 UTC**，base `develop`；merge SHA `c8a06a87c7de8c9bcfd44f1bb511f927e48a87f0`。
- First seen：2026-10-19；scan window：上述主窗口；Impact：High；Decision：**Read**；Learning track：Embodied / Systems；Signal type：system；Status：NEW。
- Target question：checkpoint 参数未变，为什么换 PhysX/OVPhysX/Newton 后 action/state 张量可能被映射到不同关节？
- Evidence：15 文件索引，ordering/resolver/PhysX articulation 与单测 diff 已核。之前三个 backend 对 `physx` 的 child ordering/root selection 不同；现在显式 `physx` 统一为 OVPhysX breadth-first order（joint type→prim path），在 active backend 自己的 names 上生成映射；不能模拟的 root tie、nested bodies、loop 等报错。测试覆盖 root 优先级、多类型分支和拒绝路径。保持 `None` 为各自 native order，且不改变 solver order。
- Reason / 价值：直接影响 policy transfer 与 checkpoint 解释，不是单纯 API 命名清理。
- 边界：**breaking migration**：旧 Isaac Sim PhysX 的 `physx` 曾代表 authored order；重放旧 checkpoint 应从旧 native 运行记录 `joint_names/body_names` 并显式传入，不能直接相信同名配置。两个 ArticulationRootAPI 等仍有已知限制，部分资产无法导入 Newton。自动随 checkpoint 保存 ordering 仍是 follow-up。公开 head physics/core/multi-GPU smoke 等 check success，但不是本仓库复现或任意资产跨后端等价保证；develop 合并不等同已发 stable release。
- Related topics：[具身学习入口](../../05-embodied-infra/topics/embodied_infra.md)、action representation、sim-to-sim、checkpoint/recovery；Next：把 joint/body names 与单位、control mode 一起列入 checkpoint manifest，做逐关节映射验收。
- 一句话：统一 symbolic ordering 并对无法复现的资产拒绝猜测，避免相同 shape 的张量在换仿真后静默控制不同关节。

<a id="a7"></a>
### A7 · CPU-staged AdamW：CPU authoritative state 需要可等待的 staging 与恢复边界

- Source ID：`github:areal-project/AReaL:pr:1613`；[官方PR](https://github.com/areal-project/AReaL/pull/1613) / [固定merge](https://github.com/areal-project/AReaL/commit/89400da9aa7606b7b5fbc8fd46f576f689e3dc06)；Title：feat(engine): add CPU-staged Megatron AdamW；Author：`yulangz`；Merged：`2026-10-16T02:15:52Z`；base：`main`。
- First seen：2026-09-22（open proposal）；Merged event first verified：2026-10-19；Scan window：同本报告固定窗口；Type：official merged PR；Impact：high；Decision：Read；Learning track：RL / Training；Signal type：system；Status：NEW。
- Target question：optimizer state 留在 CPU 后，如何保证 CUDA slot 重用、checkpoint 和 model-only reload 都读到最新一致状态？
- Reason：将显存优化落实到状态所有权和 checkpoint 失败语义，直接影响 colocated RL 的可用容量。
- 一句话价值：FP32 master/moments 常驻 pinned CPU，以有限 CUDA slots 执行 H2D→AdamW→D2H；slot 重用、保存和恢复必须遵守同一状态生命周期。
- Subsystem：training / weight sync / checkpoint/recovery；Engineering dimension：correctness / state / memory。
- Evidence：实际读取 gpu_staged_optimizer.py 的 CPU slabs、三 stream/event 链与 checkpoint state_dict；staged_optimizer_runtime.py 的 FREE/D2H_PENDING 状态和 drain/release；实际读取 checkpointer.py 的 begin/complete load、model-only reset、load_failed fail-stop；AWEX writer 保留 IPC export 到 reader ACK；检查 tests/test_gpu_staged_optimizer.py 的 delayed-gradient stream、slot memory bound、save drain、resume parity、failed-load 和 model-only reset；另有 DP2/MCore 与 AWEX 相关测试文件。
- Evidence limits：本轮未执行上游测试；46-file PR 仅深核上述关键路径，不声称逐行全审；PR 作者的早期 actor memory 降幅来自拆分前 snapshot；不能作为当前合并版本的整卡峰值收益；作者近期八卡/三次更新 smoke 仅单节点短测；v2 residency、长跑收敛和最新 merge 的多节点尚无独立验证；09/22 open PR snapshot 已观察该提案；本次接受的是 10/16 merged 事件，非重新发现旧提案。
- AReaL transfer / Next：AReaL 原生能力；先做 checkpoint freshness、HF reload 后 master/moments 重建与 ACK 超时的故障注入，并分别量 actor allocated、全阶段整卡峰值和吞吐。
- Related topics：[RL 框架](../../04-rl-infra/topics/rl_framework_selection.md)、[RL 状态生命周期](../../04-rl-infra/topics/agentic_rl.md#oct19-state-lifecycle)

<a id="a8"></a>
### A8 · R3：route replay 的单位是 token/layer 布局和 forward/backward 生命周期

- Source ID：`github:areal-project/AReaL:pr:1496`；[官方PR](https://github.com/areal-project/AReaL/pull/1496) / [固定merge](https://github.com/areal-project/AReaL/commit/6cad8710bb5740a5480f8fba3a5970dd08638d01)；Title：feat(engine): add native R3 router replay；Author：`yulangz`；Merged：`2026-10-15T09:01:24Z`；base：`main`。
- Source ID：`github:NVIDIA-NeMo/RL:pr:4458`；[官方PR](https://github.com/NVIDIA-NeMo/RL/pull/4458) / [固定merge](https://github.com/NVIDIA-NeMo/RL/commit/c7aca812b741321d7d735e0c3175012f16064a87)；Title：feat(automodel): support rollout routing replay (R3) on the Automodel policy；Author：`joyang-nv`；Merged：`2026-10-17T11:32:00Z`；base：`main`。
- First seen：2026-10-19；Scan window：同本报告固定窗口；Type：official merged PR；Impact：high；Decision：Read；Learning track：RL / Training；Signal type：system；Status：NEW。
- Target question：同权重下 MoE expert selection 漂移时，如何保证 rollout、重算 logprob 与 activation-checkpoint backward 使用对齐 routes？
- Reason：两个后端新增落地揭示可迁移 correctness contract，不能把开启 replay 当成全模型数值等价。
- 一句话价值：R3 重放离散 expert IDs，同时保留实时 router weights 的学习；route 缺失、padding、layer mapping 和 backward recompute 都必须显式处理。
- Subsystem：rollout / training / data/trajectory path；Engineering dimension：correctness / state / memory。
- Evidence：AReaL 实际读取 r3/config.py、preprocess.py、orchestration.py 及 megatron_engine.py 关键路径：连续 MoE 后缀限制、input-token 对齐、无效 microbatch 的 live/record fallback、REPLAY_FORWARD/REPLAY_BACKWARD 与 finally 清理；NeMo 实际读取 automodel/router_replay.py、train.py、data.py：decoder-layer/MoE-layer 两种 mapping、all -1 缺失行 live fallback、损坏行拒绝、gate 未消费 replay handle 报错、训练与 backward 位于 replay context；测试文件覆盖 native FIFO/forward-backward replay、真实 Automodel Gate 的 gradient、activation checkpoint recompute、无效 routes、padding 与未消费 gate。
- Evidence limits：AReaL 仅选读71-file PR关键路径；SGLang+Megatron，限制 sparse/interleaved/trailing dense MoE、padded等不支持布局，不能推断 vLLM 已支持；NeMo Automodel 要求 vLLM，拒绝 CP>1、sequence packing、sequence parallel 与 HF fallback；这些限制不同于 AReaL；作者 canary/20-step 数值为上游自报；PR 更新段落仍留 current-head smoke 与 step-time 对比待补，不能把早期跑分或降低logprob差异写成收敛提升；本轮未运行 GPU/多节点测试；R3 与 experience replay 是不同概念。
- AReaL transfer / Next：迁移 route schema、missing-vs-corrupt 分支、packing/CP/SP 映射以及异常后 replay 队列清空检查；为 AReaL 比较逐 token mismatch 的 tail 指标。
- Related topics：[agentic_rl](../../04-rl-infra/topics/agentic_rl.md)；[rl_framework_selection](../../04-rl-infra/topics/rl_framework_selection.md)

<a id="a9"></a>
### A9 · FSDP2 CPU offload：host 读取发生在 model.to 内，末尾同步已太迟

- Source ID：`github:verl-project/verl:pr:8187`；[官方PR](https://github.com/verl-project/verl/pull/8187) / [固定merge](https://github.com/verl-project/verl/commit/edd2c63b6a77b551c20cdded1c802d82a9cc751d)；Title：[fsdp] fix: prevent parameter corruption during FSDP2 CPU offload；Author：`SelfDLer`；Merged：`2026-10-16T03:18:32Z`；base：`main`。
- First seen：2026-10-19；Scan window：同本报告固定窗口；Type：official merged PR；Impact：high；Decision：Read；Learning track：RL / Training；Signal type：system；Status：NEW。
- Target question：为何没有 optimizer update 的 offload/onload 也能改变参数，而一次末尾 synchronize 仍无法修复？
- Reason：uneven sharding 的 CPU repadding 会提前消费异步 D2H destination，造成持久参数损坏；是通用 offload correctness 问题。
- 一句话价值：把 FSDP2 runtime 与初始化 offload 都改为 blocking D2H，确保 FSDP 内部 CPU repadding 读到完成的传输。
- Subsystem：training / weight sync；Engineering dimension：correctness / state / memory。
- Evidence：实际 diff：offload_fsdp2_model_to_cpu 使用 model.to(cpu, non_blocking=False)，fsdp2_load_full_state_dict 复用该 helper(empty_cache=False)；分布式 regression 涵盖 nested FSDP2、even/uneven shards、conversion stream 上预排计算、default/non-default streams，以及 CPU/恢复后逐元素精确对照；CUDA/NPU CI入口与 CPU transfer-policy mock 检查均在 actual diff；本轮只核代码/测试设计，没有执行。
- Evidence limits：H2D helper 保留 non_blocking=True 不代表 overlap 性能无损；blocking D2H可能改变 host 等待及 pinned storage；作者的 CUDA/NPU reproducer 与 NPU 两卡回归为上游自报；无本轮独立复现或受控吞吐对比。
- AReaL transfer / Next：在 AReaL 的 FSDP/offload 路径加入“无 forward/backward/step 的参数 round-trip”验收，并审计库内部 host consumers；不要以 empty_cache 或末尾同步代替依赖边界。
- Related topics：[rl_framework_selection](../../04-rl-infra/topics/rl_framework_selection.md)；[README](../../02-training-infra/README.md)

<a id="a10"></a>
### A10 · 多轮 RL：原始 prompt 的 assistant 历史不属于本轮生成

- Source ID：`github:NVIDIA-NeMo/RL:pr:4487`；[官方PR](https://github.com/NVIDIA-NeMo/RL/pull/4487) / [固定merge](https://github.com/NVIDIA-NeMo/RL/commit/e12d18ba8e115fdcb957329740dae1f12864cb53)；Title：fix(training): exclude prompt history from reward lengths and loss masks；Author：`linj-glitch`；Merged：`2026-10-18T23:27:53Z`；base：`main`。
- First seen：2026-10-19；Scan window：同本报告固定窗口；Type：official merged PR；Impact：high；Decision：Read；Learning track：RL / Training；Signal type：system；Status：NEW。
- Target question：history 中已有 assistant turns 时，长度奖励和 loss mask 如何避免惩罚/训练旧文本并漏计后续生成 turn？
- Reason：role==assistant 不是 generated-token 的充分条件；错误 provenance 会同时改变 reward shaping 和训练目标。
- 一句话价值：先以 generation_logprobs 标记区分本轮生成，再累加所有生成 assistant turns；必须在 flatten 补齐缺失 logprobs 之前做统计。
- Subsystem：training / data/trajectory path；Engineering dimension：correctness / data/trajectory semantics。
- Evidence：实际 diff：reward_functions.py 从首个 assistant 长度改为所有带 generation_logprobs 的 assistant 长度之和；decompose_message_log 使用同一规则；sync_rollout_actor.py 将 decomposition 移至 flatten 之前；sync/async PPO与distillation loss-mask 分支都要求 generation marker；回归涵盖100个history tokens+15个新tokens、空history placeholder、多turn sum、actor操作顺序以及PPO/distillation训练循环mask。
- Evidence limits：本轮未执行上游测试；作者说明 full training 与 CI 未跑；该marker是当前schema约定，不能泛化为所有外部message log都自带可靠provenance；必须在字段补全前消费。
- AReaL transfer / Next：为 AReaL 的多轮工具轨迹建立 prompt/generated/tool 三类 token provenance，并让 reward长度、advantage/loss mask、数据面序列化共享契约。
- Related topics：[agentic_rl](../../04-rl-infra/topics/agentic_rl.md)；[mopd](../../04-rl-infra/topics/mopd.md)

<a id="a11"></a>
### A11 · ReFlow：将经验复用、loss 权重与 policy age 分开

- Source ID：`arxiv:2610.21640v1`；[abs](https://arxiv.org/abs/2610.21640v1) / [方法与评估](https://arxiv.org/html/2610.21640v1)。Title：ReFlow: Replay-Driven Decoupling for Efficient RL Post-Training。
- Authors：Yukun Tian, Junwei Su, Haofeng Xu, Lansong Diao, Zhengping Qian, Weibo Zhao, Wenchao Zhou, Ruiting Zhou, Chuan Wu；v1提交：2026-10-15 10:21:41 UTC；10/16公告；First seen：2026-10-19；Scan window同本报告；Type：systems paper；Impact：high；Decision：Read；Learning track：RL / Training / Systems；Signal type：system；Status：NEW。
- Target question / Reason：有界replay能否减少新rollout需求，又控制policy lag和弹性切换成本？该机制将学习目标、fresh-data供给和资源调度一起定义。
- 一句话价值：把replay比例上限、replay loss权重和policy lag分开控制，再按完整replica组调整预热worker角色，避免用更高staleness单独换吞吐。
- Evidence：原始citation metadata与abs、方法、评估相符；σ/τ/δ分别管理batch reuse、loss贡献和age；hybrid workers预热并按replica组切角色，新增trainer用shard-local broadcast；KV/preemption反馈控制admission。
- 数字/边界：作者在最多128 H800上报告最高2.06×；指标是trainer processed tokens/s，包含replay，并非新轨迹生成率。reasoning只5 warmup+5测量step、coding10+10；实验σ=τ=0.5、δ=1。理论结论依赖局部假设；公开代码链接未核到，全部性能为作者自报。
- Subsystem：rollout / scheduler / data/trajectory path / weight sync；Engineering dimension：fresh-sample供给、policy staleness、状态转移与吞吐口径。
- AReaL transfer / Next：先用已有version ledger模拟σ/τ/δ与fresh supply，再验证切角色后optimizer shard、in-flight trajectory和collective成员；不直接复制benchmark参数。 作为P1候选精读replay目标与fresh-token吞吐口径；不自动改P0。
- Related topics：[agentic_rl](../../04-rl-infra/topics/agentic_rl.md)；[rl_framework_selection](../../04-rl-infra/topics/rl_framework_selection.md)。

<a id="a12"></a>
### A12 · TRL v1.16.0：共享前缀的计算可复用，rollout loss 多重性仍要保留

- Source ID：`github:huggingface/trl:release:v1.16.0`；[官方release](https://github.com/huggingface/trl/releases/tag/v1.16.0)；发布账户：qgallouedec；Published：2026-10-15 18:00:33 UTC；First seen：2026-10-19；Scan window同本报告；Type：official release；Impact：high；Decision：Read；Learning track：RL / Training；Signal type：system；Status：NEW。
- 较早机制证据：`github:huggingface/trl:pr:7328`；[Tree attention](https://github.com/huggingface/trl/pull/7328)；作者AmineDiro；[固定merge ea4d310](https://github.com/huggingface/trl/commit/ea4d31033e216377bdf66576d426e48187d3eb06)，2026-10-13 16:47:19 UTC（**早于窗口**）。该PR于09/22已在draft/open快照出现；不是本轮新merge。API compare确认v1.16.0包含它（ahead=33、behind=0、merge-base同SHA），未逐审其后33提交。
- Target question / Reason：多条rollout共享长前缀时，能否只forward一次并保留每条trajectory的训练目标？这能减少Agentic RL重复计算，但必须维护branch隔离和loss多重性。
- 一句话价值：以`(token,next_token)`建trie，非attention层按unique tokens执行；每个segment只gather根到自身路径的K/V，由FA3 varlen处理，`token_index`将共享位置的logprob映射回各rollout loss terms。
- Evidence：实际读取`packing.py`完整实现、trainer planner/compute_loss/架构限制和测试diff。预算同时约束unique tokens及training loss terms；测试检查branch causal mask、shared位置多loss项、group布局和rank不饥饿，并对真实compute_loss/backward和sequence packing做loss/gradient parity。
- 证据边界：测试loss容差为atol/rtol 1e-3，全参数拼接gradient相对norm<5%，不是逐参数bitwise保证。约3倍来自作者单H100 BF16共享前缀微基准；完整训练对比来自**旧FlexAttention实现**，不能当作当前FA3端到端结果。收益依赖packing ratio/fork threshold。本轮未运行测试。
- 限制：Hopper/FA3；拒绝linear attention/convolution和attention sinks；VLM parity只覆盖text-only。GRPO主loss保留rollout multiplicity，但MoE auxiliary load-balancing以unique forwarded tokens计，目标并非处处不变。
- Subsystem：training / data/trajectory path；Engineering dimension：compute reuse / memory / loss correctness / branch-state isolation。
- AReaL transfer / Next：比较shared-forward与per-rollout objective分离、双预算、branch mask和auxiliary weighting；含GDN/recurrent state模型须先定义分支状态恢复。列P1候选，不改P0。
- Related topics：[agentic_rl](../../04-rl-infra/topics/agentic_rl.md)；[rl_framework_selection](../../04-rl-infra/topics/rl_framework_selection.md)。

<a id="a13"></a>
### A13 · Remote KV 的成本模型与 context-parallel 一致性

- **Source ID**：`arxiv:2610.21500v1`；**First seen**：2026-10-19；**scan window**：同本报告。
- **来源/类型**：[When the KV Cache Is Slower Than the GPU: Scaling Remote KV Tiers for Context-Parallel Prefill](https://arxiv.org/abs/2610.21500v1)，论文；Muhammad Ibrahim；v1 **2026-10-15 08:37:22 UTC**，10/16 官方公告。已核 `citation_title/author/date` 和方法/实验正文。
- **Impact / Decision**：High / **Read**；**Learning track**：Inference / Systems；**Signal type**：system；**Status**：NEW。
- **Reason / Target question**：远端命中为何比重算慢，PCP 中单 rank IO stall 为何拖住整个 engine；如何同时保持最长连续 prefix 和所有 rank 的 collective 次序？
- **Evidence**：正文把加载成本拆为固定开销与传输，按 chunk index 分配 owner，并在 load 前交换失败/metadata、达成共同 prefix，避免某 rank 单独跳过 collective。8×B300、GLM-5.3 FP8、PCP8/EP8 的 450 请求热缓存回放中，作者报告 p99 83.32s→0.67s；但优化后 median 0.40s 仍慢于重算 0.31s。每条件仅一次；回放只测单 prefill engine，约17k-token阈值是单点拟合后的预测，未测多 engine 争用。不能把五节点部署对比视为只改变 cache 的消融。[方法和实验](https://arxiv.org/html/2610.21500v1)
- **相关主题**：独立推理 serving / KV tiering / context parallelism / NCCL；正文中的 throughput +34% 还包含另一项 FlashInfer kernel 改动，不归功于 KV sharding 单项。
- **一句话价值**：GPU 更快会提高远端 KV 的最低有效带宽，并让 scheduler 固定开销成为无法靠扩容存储消除的成本。
- **Next step**：在共享推理阶段问题中加入 load-vs-recompute、跨 rank prefix 一致性与 timeout fallback 的验收设计；不直接改 runtime，也不标已复现。

<a id="a14"></a>
### A14 · BCL：固定样本数下的 rank-local workload balancing

- **Source ID**：`arxiv:2610.21162v1`；**First seen**：2026-10-19；**scan window**：同本报告。
- **来源/类型**：[Coordination-Free Workload Regularization for Variable-Cost Distributed Training](https://arxiv.org/abs/2610.21162v1)，论文；Tao Huang、Chen Zhu、Antonio Alonso、Daming Li、Adarsh Goura、Jaideep Ray、Yitong Zhou；v1 **2026-10-15 03:22:50 UTC**，10/16 公告。citation metadata 与算法、Table 5 对照条件已核。
- **Impact / Decision**：Medium–High / **Read**；**Learning track**：Training / Systems；**Signal type**：system；**Status**：NEW。
- **Reason / Target question**：packing 消除 padding 后，如何在不跨 rank 交换样本的前提下减小同步 straggler，并保留 trainer 每步固定 record-count 契约？
- **Evidence**：BCL 在每 rank 的 K 个 batch 窗口内按长度成本作 LPT 分配及交换，保证每样本一次、每包 B 个、重排有界。Llama-3.2-1B / NeMo-RL v0.7 / FSDP2 / bf16 / FA3、8×H200，三次300步、剔除前20步：同 input path 的11语料 lookahead 对照平均提速0.4%–30.5%；与 pooled MFFD 的13语料5.0%–16.7%同时包含输入路径和跨 campaign 差异，不能称纯调度收益。各对照固定同一 cap；单条超长样本 OOM、收敛质量与跨 rank 长期偏斜仍不由 BCL 保证。[算法和实验](https://arxiv.org/html/2610.21162v1)
- **相关主题**：[data_parallelism](../../02-training-infra/topics/data_parallelism.md)、[distributed_training](../../02-training-infra/topics/distributed_training.md)、[long_context_training](../../02-training-infra/topics/long_context_training.md)；可迁移到 RL/SFT data path 的成本账本，不能把 SFT 结果外推 RL 更新目标。
- **一句话价值**：固定 token 数、固定 record 数与固定计算量是不同契约，局部 lookahead 可减少同步尾部，但需把调度收益和输入管线收益拆开测。
- **Next step**：训练 handbook 增加同数据/同cap/同input-path对照，以及 record/token loss 分母、顺序变化和最长单样本边界。

<a id="a15"></a>
### A15 · Meta AMD 训练生产化的完整成本与可靠性账本

- **Source ID**：`arxiv:2610.22230v1`；**First seen**：2026-10-19；**scan window**：同本报告。
- **来源/类型**：[Architecting for Hardware Diversity: Meta's End-to-End Co-Design for High-Performance AI Training on AMD GPUs](https://arxiv.org/abs/2610.22230v1)，工业论文；Haoyu Zhang、Arshad Rahman 等，完整58位作者已核 metadata；v1 **2026-10-15 17:09:08 UTC**，10/16 公告。
- **Impact / Decision**：High / **Read**；**Learning track**：Systems / Training；**Signal type**：system；**Status**：NEW。
- **Reason / Target question**：训练从 CUDA 移植到另一 GPU 后，哪些差距来自模型/shape、kernel、collective、功耗与可靠性，而不是简单“能运行”？
- **Evidence**：正文披露利用 HBM 容量调整训练配方、分析式 kernel 选择、按 chiplet/atomic/occupancy 改 kernel，并加入 checksum/replay、GPU 间投票与隔离。10个生产推荐模型、8–128 GPU、同卡数/同 batch 的 MI350X/B200 throughput 几何均值为101.2%；但分别部署在1000W与750W功耗，表中有快有慢，不能外推为等功耗排名或大模型训练通用优势。训练 failure、SDC、数值 parity 均是工业作者报告；本轮核正文，未复现其生产模型。[Table I / III、方法和运维](https://arxiv.org/html/2610.22230v1)
- **相关主题**：[fp8](../../01-systems/topics/fp8.md)、[nccl](../../01-systems/topics/nccl.md)、[distributed_training](../../02-training-infra/topics/distributed_training.md)、[fault_tolerance](../../02-training-infra/topics/fault_tolerance.md)。
- **一句话价值**：跨 GPU 的性能可移植性需要从模型配方到故障隔离的共同设计，单个 kernel 峰值不能替代训练吞吐和正确性验收。
- **Next step**：系统学习路线补 cross-vendor 对照矩阵：功耗/shape/精度/通信重叠/故障率；生产数字保持作者自报标签。

<a id="a16"></a>
### A16 · SGLang HiCache write-back 必须等 KV producer 完成

- **Source ID**：`github:sgl-project/sglang#44800`；**First seen**：2026-10-19；**scan window**：同本报告。
- **来源/类型**：[Order write-back host copies after the forward that produced the KV](https://github.com/sgl-project/sglang/pull/44800)，官方仓库 merged PR；提交账号 `metamergebot`；合并 **2026-10-17 08:31:41 UTC**。merge/head SHA、11个文件索引已留证。
- **Impact / Decision**：High / **Read**；**Learning track**：Inference / Systems；**Signal type**：system；**Status**：NEW。
- **Reason / Target question**：HiCache 备份被标作完成，却保存了尚未被 forward 更新的旧页时，如何建立可验证的 producer→copy→publish 顺序？
- **Evidence**：已读5个生产文件全部 diff 和核心 CUDA 回归：Python/Rust tree 在 write-back insert 时不提前发布 SWA backup；blocking write-back drain 设置 `wait_for_forward`，D2H stream 先等 model forward stream。回归故意延迟 forward、把页从-1改42，再驱逐同一请求较旧 SWA 窗口，断言 host snapshot 为42。作者报告单CUDA与931项Rust测试；PR所链接 Extra CI 有失败，本轮未核全失败原因、未本地跑测试。该改变不证明其他 async write-through 路径或所有硬件矩阵已安全。
- **相关主题**：独立推理 KV/state 生命周期、SWA/hybrid cache、CUDA stream ordering；与 remote KV 成本项独立，不能以缓存 hit rate 代替内容正确性。
- **一句话价值**：cache 页的 ownership 与其内容就绪是两个条件，拿到页索引并不代表它已可安全复制给下一个请求。
- **Next step**：把“延迟 producer 后驱逐再回读”的负向回归模式纳入 KV/cache 验收；补查 CI 与目标运行时版本，main 合并不等于已经进入 stable release。

<a id="a17"></a>
### A17 · Nemotron 3.5：把多教师训练、draft 对齐与混合状态缓存一起设计

- Source ID：`nvidia-blog:nemotron-3-5-super-vl:2026-10-15`；Type：official technical blog / technical report；标题 *NVIDIA Nemotron 3.5 Super VL Powers Specialized Agents with Multimodal Reasoning*；作者 Chris Alexiuk、Chintan Patel；发布时间 `2026-10-15T18:01:00Z`。[官方文章](https://developer.nvidia.com/blog/nvidia-nemotron-3-5-super-vl-powers-specialized-agents-with-multimodal-reasoning/)链接的 [NVIDIA 技术报告](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3.5-Super-and-Lightning-Technical-Report.pdf)题为 *Nemotron 3.5 Super and Lightning: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Models for Agentic AI*，署名 NVIDIA、首页日期 10/14，未披露首次公开时刻；本组以 10/15 公告为事件，不把报告虚构为精确窗口内首发。
- Impact：High；Decision：**Deep Dive**；Learning track：Training / RL / Inference；Signal type：model / system；First seen：2026-10-19；scan window：本报告窗口；Status：NEW。
- Target question：能力更强的 student，为什么仍需重训 drafter；混合模型命中 prefix cache 后，为什么还会重复计算？
- Evidence：报告 §3.6 仅 Super 用 student rollout 接域专家 teacher 打分，两阶段 MOPD 配独立 teacher fleets，并显式处理一步 stale 的 student importance weighting，Lightning 没有该 MOPD 阶段；§3.7 在 backbone 完成后冻结它，单独训练 MTP head，使用 hidden-state mixing 与冻结 backbone 的分布监督，teacher 角色不同。§5.3 指出 KV/Mamba checkpoint 不含末 token 的中间 hidden state，draft 需要恢复它；在边界附近多存 Mamba checkpoint，才能减少 replay。已读这些机制节，未完成全报告精读。
- Reason：工业报告把后训练目标、actor/teacher 调度与推理状态成本接起来，能补现有 MOPD 与 rollout/inference 判断；不是因榜单名次收录。
- 边界：token-level student importance weighting 不能消除 context 分布偏差；报告承认 masking 有 bias。性能为 NVIDIA 指定硬件/代理负载结果，未复现。2.83T Indic 是生成语料池，不能当作本模型实际训练 token。LiLiCorr、QAD 仅作为后续精读点，不据公告宣称全栈实测收益。
- Related topics：[MOPD](../../04-rl-infra/topics/mopd.md)、[推理状态](../../03-inference-infra/topics/inference_state.md)；Next：先画 student→teacher→learner 管道及 prefix-hit→draft 的状态依赖图，再核对应版本的部署配置；共享 P1 按需选读。
- 一句话：模型、drafter 和缓存要共享一致的训练与状态边界，单独提升模型能力或 prefix 命中率还不足以提高完整任务效率。

<a id="a18"></a>
### A18 · Beam：1M context 的成本由 attention 层布局与完整权重驻留决定

- Source ID：`hf-blog:reflection-beam:2026-10-16`；Type：HF official-team engineering post / vendor model card；标题 *Welcome Beam: a 501B open model you can own, customize, and scale*；作者 ben burtenshaw、merve、Pedro Cuenca；10/16。[HF 文章](https://huggingface.co/blog/reflection-beam)、[Reflection 模型卡](https://huggingface.co/reflection/Beam-1.0-501B-A23B)、[v5.20.1 实现](https://github.com/huggingface/transformers/blob/v5.20.1/src/transformers/models/beam/modeling_beam.py)。HF 模型历史首条公开记录 `0664107` 为 `2026-10-16T10:34:28Z`；配置与模型卡核验快照 `78f8f654d40a40919b8e9954b40dd879ecb9a692`。
- Impact：High；Decision：**Read**；Learning track：Inference / Training；Signal type：model / system；First seen：2026-10-19；scan window：本报告窗口；Status：NEW。
- Target question：两个同样支持长上下文的 MoE，为什么单请求 KV 预算与最小部署规模会不同？
- Evidence：config 的 52 层中 43 层为 512-token sliding attention、9 层 full attention；代码给局部层 RoPE、全局层 NoPE + 随位置变化的 query scaling，并采用随深度缩放的 residual。按 8 KV heads、128 head dim、BF16 推导，只有全局层的缓存随长度持续增长，局部层有窗口上限。501B 总参数、23B active 是官方口径；所有 expert 仍需驻留或明确 offload，active 参数不能用来估权重内存。
- Reason：公开 config/实现能把模型机制转换为容量和算子兼容判断；同时提供 residual/精度语义的学习案例。
- 边界：模型卡明确 **Technical Report (coming)**；预训练配方与集群细节未核验。HF 的约 36 KiB/token 是特定 BF16 dynamic-cache 结构估算，不是所有 backend 的实测 RSS；稀疏计算也不保证等同 23B dense 模型速度。未执行模型、SFT 或 RL recipe。
- Related topics：[推理状态预算](../../03-inference-infra/topics/inference_state.md)、[MoE](../../02-training-infra/topics/moe.md)；Next：按 layer type、KV dtype、并发和 backend 分页开销建立容量表，再检查 sliding-cache 真正截断与算子数值 parity。
- 一句话：长上下文容量不能只看“1M”和激活参数量，应分别计算全局 KV、局部窗口及全部专家权重。

<a id="a19"></a>
### A19 · DeepSeek 编码修复：tool result 的顺序与回合身份都是模型输入语义

- Source ID：`hf:deepseek-ai/DeepSeek-V4.1-Flash:commit:314113f8a7dadb388215396ba5fbee2c610a04d6`；Type：official reference-code commit；标题 *fix(encoding): preserve tool image order and current-turn thinking*；作者 Qizhou Guo；`2026-10-16T09:27:28Z`。[固定提交与测试差异](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/commit/314113f8a7dadb388215396ba5fbee2c610a04d6)。
- Impact：High；Decision：**Read**；Learning track：Inference / RL；Signal type：system；First seen：2026-10-19；scan window：本报告窗口；Status：NEW。
- Target question：工具返回图片乱序到达，或只追加工具结果时，prompt 和 image payload、当前回合 reasoning 会不会被悄悄改写？
- Evidence：与 parent `5254448` 的 `encoding.py` 对照：先规范化/按 tool call 排序，再提取图像；保留 list/content_blocks 输入；只有工具结果的 user block 不作为新用户回合。校验放在 merge 前，避免文本字段被丢弃后漏检。三文件 diff 含多输入格式图像顺序、当前回合 thinking 与占位符回归；测试存在，未本地运行。
- Reason：这会静默改变模型真正看到的输入，直接影响多模态 agent rollout 的正确性与可重复性。
- 边界：该提交属于原 V4.1-Flash 仓库，不能冒称 1017 权重性能增益的原因；也未证明所有托管 API/backend 都已采用它。10/17 新权重与 API 公告另列 Observed。
- Related topics：[推理输入与状态](../../03-inference-infra/topics/inference_state.md)、[Agentic RL](../../04-rl-infra/topics/agentic_rl.md)；Next：将乱序 tool results、夹带图片、无 tools 声明的同回合续写纳入生产 adapter parity 检查，并固定模型与 encoder revision。
- 一句话：先确定工具结果的最终顺序与回合边界，再编码图像和 reasoning，才能保证格式转换没有改变任务语义。


## Embodied Models & Infra Watch

### Models / Learning

| 来源 | 实际检查与状态 | 缺口 |
|---|---|---|
| [arXiv cs.RO](https://arxiv.org/list/cs.RO/recent) / [cs.LG](https://arxiv.org/list/cs.LG/recent) / [cs.CV](https://arxiv.org/list/cs.CV/recent) | 六个官方 new/recent 页面直接 HTTP 核验；最新 announcement heading 为 10/16；3,713 去重 IDs 含 replacements；按具身词筛选并人工查看 10/14 附近以后的相关标题，九候选 citation+abstract，八候选 HTML 获取，三篇方法/评估重点读。Accepted A1–A3；Observed EO1–EO6 | 标题筛选不是 3,713 篇全文 triage；cs.LG recent 达 2,000 返回上限，历史不可声称完整。Spark0 HTML失败，触觉/灵巧手/移动控制仍未深入 |
| [PI Blog](https://www.pi.website/blog) / [Research](https://www.pi.website/research) / [openpi](https://github.com/Physical-Intelligence/openpi) | Not found in checked window：博客最新可见 10/13 API beta，上一轮已 Observe，未重复；openpi merged PR/release 为0 | 索引可访问；不等于模型内部没有变化，也不补称 API 已有完整 deadline/fallback 技术披露 |
| [OpenVLA](https://github.com/openvla/openvla) / [OFT](https://github.com/moojink/openvla-oft) | Not found：窗口 merged PR/release 均0 | 不涵盖开放 PR、直接 commits 和所有论文渠道 |
| [GR00T](https://github.com/NVIDIA/Isaac-GR00T) / [NVIDIA robotics](https://developer.nvidia.com/blog/tag/robotics/) | Infra A4 Accepted；没有本窗口新 N1.7 release 证据；robotics 标签首页16项最新10/07，未发现窗口内模型帖子 | 不把 N1.7旧release 或本轮部署优化当新模型首发；只检查该标签首页 |
| [SmolVLA card](https://huggingface.co/lerobot/smolvla_base) | Not found：card+API+最近10 commits；lastModified 10/06，SHA `5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6`，不在窗口 | 不覆盖所有派生 SmolVLA 模型；LeRobot Infra 另列 |

World Model 本轮 A2 按环境校准、action-conditioned prediction 和实际动作选择接受；EO3/EO5/EO6 保留待核，不用视频质量代替控制价值。

### Infra

七库用 exact UTC `merged:` 查询，**111条、total_count=returned、incomplete_results=false**；67 条 release-list，均单页少于100而耗尽。111条为标题筛选，15个PR获取详情/完整文件列表（14具身仓+1 SGLang），不是111个完整 diff 全审。具体 hashes、head/merge SHA、what_read 和 limits 见 evidence.json。

| 仓库 | merged PR / releases返回 | 判断 |
|---|---:|---|
| Physical-Intelligence/openpi | 0 / 0 | Not found in queried merged PR/releases |
| openvla/openvla | 0 / 0 | Not found in queried merged PR/releases |
| moojink/openvla-oft | 0 / 0 | Not found in queried merged PR/releases |
| NVIDIA/Isaac-GR00T | 1 / 5 | Accepted A4；最新 N1.6.1-release4/23，N1.7-release4/18，均非新发 |
| isaac-sim/IsaacLab | 88 / 27 | Accepted A6；Observed EO10–EO12；10/15 `interactive-demos` release为演示资源，Rejected/Ignore；没有从docs标题推断正式3.0已发布 |
| huggingface/lerobot | 21 / 11 | Accepted A5；Observed EO7–EO9。**8 main、13 integration-only**，已逐一核 base。当前 release仍v0.6.1（8/03） |
| meta-pytorch/torchcodec | 1 / 24 | #1769 revert rocjpeg tolerance：Rejected/Ignore，窄测试容差回退，没有新的主路径机制；最新v0.17.0为9/30 |

保留 **2026-09-22 16:52:05 上海时间起的旧 arXiv 发现缺口**及 **2026-09-22 17:06:16 起的 direct commits / 历史外围 PR 缺口**。本轮七库窗口 merged PR 与 release-list 的完备性只适用于上述查询，不关闭跨来源历史缺口。本轮未新发现必须写为 backfill 的独立旧材料；现有10/13观察和4月release只去重，不重新接受。

## RL Framework Watch

| Repository | merged total / returned | incomplete | release pages | release total | window releases |
|---|---:|---|---|---:|---:|
| NVIDIA-NeMo/RL | 60 / 60 | false | 10 | 10 | 0 |
| OpenRLHF/OpenRLHF | 0 / 0 | false | 100, 10 | 110 | 0 |
| THUDM/slime | 6 / 6 | false | 11 | 11 | 0 |
| alibaba/ROLL | 1 / 1 | false | 5 | 5 | 0 |
| areal-project/AReaL | 16 / 16 | false | 26 | 26 | 0 |
| verl-project/verl | 14 / 14 | false | 16 | 16 | 0 |

共97条merged PR，全部题名初筛，查询均在单个100条page内完整返回；六仓release历史178条，OpenRLHF明确读取第2页10条。无窗口内新release。PR query包含所有base branches；cherry-pick按事件索引保留但不重复计系统机制。

| 框架 | Decision | 工程判断 |
|---|---|---|
| AReaL | Accepted A7/A8；Observed RO1–RO4 | CPU state/R3关键路径；Ascend PR须单列目标分支。#1791为旧v1机制follow-up。 |
| verl | Accepted A9；Observed RO5/RO6 | offload correctness优先；硬件/配方与配置适配不独立接受。 |
| slime | Observed RO7/RO8 | long-context agent与新模型backend有实质代码，但本轮未完成复杂patch和GPU证据核查。 |
| ROLL | Rejected / Ignore | 唯一#507是ROCm默认TRITON_ATTN窄backend适配；无独立新机制。 |
| OpenRLHF | Not found in queried merged PR/releases | 0 merged/0 release；不等于direct commits无更新。 |
| NeMo RL | Accepted A8/A10；Observed RO9–RO14 | replay/trajectory provenance/状态生命周期；tests-only与cherry-picks不重复收录。 |

TRL为扩展框架：本轮release的tree packing单列A12；ReFlow A11是论文，不计六仓PR数量。每条RL Accepted均记录 subsystem、工程维度及可迁移到AReaL的检查。

## Inference Systems Watch

| 运行时 | 实际覆盖 | 结果与边界 |
|---|---|---|
| [vLLM](https://github.com/vllm-project/vllm/pulls) | 289 merged PR，3页100/100/89；官方releases首100项 | Accepted无新增runtime PR；Observed #59790/#56836/#58168与v0.32.0；#43202仅候选。全标题筛选不等于全diff。 |
| [SGLang](https://github.com/sgl-project/sglang/pulls) | 319 merged PR，4页100/100/100/19；官方releases首100项 | Accepted A16；Observed #43108；#33728仅候选。窗口无release，最新官方release仍v0.5.21，不能凭代码内版本/cookbook猜v0.5.22已发布。 |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/pulls) | 142 merged PR，2页100/42；官方releases首100项 | Observed #19940/#19722与v1.4.0rc0；#19016仅候选。release已知问题、merged main与stable分开。 |

**750条**窗口merged PR全部分页，三个查询`incomplete_results=false`且`returned=total_count`。选取10个PR详情/完整文件列表，代码深读范围如evidence所记，不以自动生成review或正文长期设计替代当前patch。三个release首100项都越过窗口起点，足以核本窗口；不声称完整历史发布清单。例行CI waive、文档、例行依赖更新与纯model-support标题未自动接受。

## OpenAI / Anthropic / NVIDIA / DeepSeek Watch

| 来源与实际检查 | 结论 | 证据、判断与缺口 |
|---|---|---|
| OpenAI [Research](https://openai.com/news/research/) / [Engineering](https://openai.com/news/engineering/) 索引 | **Not found in checked indexes** | 可见最新分别仍为 10/7、9/11；在所检查索引中未发现窗口内符合筛选的新文章。不是覆盖全部产品 changelog、模型卡与 docs 的证明。 |
| Anthropic [Research](https://www.anthropic.com/research) / [Engineering](https://www.anthropic.com/engineering) | **Observed / Rejected** | 10/16 quantum circuit 外部 verifier 闭环见 O-ROOT1；ECDSA 起源、AI 预期/就业文章不回答当前重点问题，Rejected / Ignore。工程索引仍为 10/13 Scion，不重复接受。 |
| NVIDIA [技术博客](https://developer.nvidia.com/blog/) / Nemotron 报告 | **Accepted / Observed** | A17 为 Accepted；10/14 NeMo Gym 芯片设计、10/15 Muse、10/16 RTX Spark 分别见观察或低优先级筛除；未把产品宣传当作新的训练系统证据。 |
| DeepSeek [API changelog](https://api-docs.deepseek.com/updates/) + [官方 HF 组织](https://huggingface.co/deepseek-ai) | **Accepted / Observed** | 原模型 10/16 encoder 提交 A19；10/17 Flash-1017 权重/API更新与 Search API 分别观察。检查了两个模型的 HF commit 历史和固定配置；原模型报告仍为 9/10 文档，没有把旧技术报告当作新报告。 |

## Hugging Face Watch

检查 [HF Blog](https://huggingface.co/blog)主列表与可见 community 列表，并读取下列 repo 最近 30 条 release（不足 30 时取全部）。这是当前窗口 release 核查，不是全量 PR/commit 审计；[证据元数据](audits/2026-10-19-frontier/root_evidence.json)保留实际返回边界。LeRobot 另见具身 Watch。

| 来源 | 结论 | 判断与下一步 |
|---|---|---|
| official-team Beam / Muse Spark Open | **Accepted / Observed** | Beam A18；Muse O-ROOT3；有模型机制/实现价值，但不能补造厂商未公开的训练配方。 |
| NVIDIA-authored Indic；HF team-authored community Tokenmaxxing | **Observed** | O-ROOT4/5；分别区分语料池与训练预算、PR 阶段与正式发行。标为 community 并不自动排除，也不按页面位置自动升级。 |
| TRL v1.16.0，10/15 | **Accepted component / Observed remainder** | AsyncGRPO tree packing 见 RL 对应条；其余 Trainium、ROCm、compile/weight-sync 修复按 O-ROOT6 观察，不把整版全部变化都算 Accepted。 |
| Transformers v5.20.0 / v5.20.1，10/15–16 | **Observed + A18 implementation evidence** | release 明示 EP 速度回退，缓存 API 变更可能影响 pi0；模型可加载不等于训练/部署全链路通过。O-ROOT7 追踪。 |
| Accelerate / PEFT / Kernels | **Not found in checked release feeds** | 窗口无新 release；未据此声称 main 无新 PR。 |
| Megatron-LM / DeepSpeed / PyTorch | **Not found in checked release feeds** | 训练主线补查窗口无新 release；direct commits/未展开 PR 保留缺口。 |

## Observed 明细

以下均为 Decision=Observe，未进入本轮Accepted；每组保留来源、判断、证据边界与下一步。

### 具身：13组

| ID / 来源 | 实际阅读与判断、Next |
|---|---|
| EO1 · [Spark-0](https://arxiv.org/abs/2610.22342v1) | 10/15 首发 metadata/abstract 已核；touch 同时用于视觉预训练监督与 tactile-conditioned action，适合补触觉缺口；HTML 404，本轮未读 PDF 方法，**不能断言已全文核验或数据已可下载**。后续核人手触觉→机器人动作对齐与同数据同预算对照。 |
| EO2 · [Value as Memory](https://arxiv.org/abs/2610.22368v1) | metadata/abstract 已核、正文已获取但未完整审核协议；value-history 作为 progress memory 与在线 policy iteration 值得比较。后续拆分 memory、online schedule、gradient budget 与并发收数带来的收益，不复述摘要大数字。 |
| EO3 · [Dexterity from Supervised Play](https://arxiv.org/abs/2610.22340v1) | metadata/abstract 已核；play policy 采集数据并给 world-model MPC 提 candidate，另做 policy distillation。后续核 contact exploration 的覆盖与模型误差；不把与 RL 的不同数据来源等同公平样本效率结论。 |
| EO4 · [Mem4Act](https://arxiv.org/abs/2610.22323v1) | metadata/abstract 已核；多粒度聚合、QA+视觉重构及 memory-first training 可与 REKI 对照，但本轮未完整审 ablation/真机协议。Next：分清压缩接口和长期信息是否实际保留。 |
| EO5 · [VeriVLA](https://arxiv.org/abs/2610.21985v1) | metadata/abstract 已核；trigger 后由 latent world model 排序多个 action chunks。Next：完整核 trigger calibration、selection bias 与新增决策延迟；不能只称 training-free，因为 verifier 仍需训练。 |
| EO6 · [CoDA](https://arxiv.org/abs/2610.22362v1) | metadata/abstract 已核；失败诊断驱动环境生成和 potential shaping 的循环有价值。Next：核 shaping 的条件/终止语义、critic 误诊与生成预算，不从 memory 框图推断开放世界泛化。 |
| EO7 · [LeRobot FineRobotics integration](https://github.com/huggingface/lerobot/pull/4984) / [#4986](https://github.com/huggingface/lerobot/pull/4986) / [#4963](https://github.com/huggingface/lerobot/pull/4963) | 13 条合并均指向 `codex/finerobotics`；正文/关键文件列表核验，主 PR #4892 仍被明确标为未合主线。恢复、publication、endpoint admission 有系统价值，先观察集成，不能写成 LeRobot main 已交付；未来核主线合并和 live Slurm/Hub 边界。 |
| EO8 · [LeRobot G1 SONIC #4955](https://github.com/huggingface/lerobot/pull/4955) | **上轮 remote inference 的明确 follow-up**：正文、token echo/neutral seed/hold gate 和测试 diff 已核；现在 SONIC token controller 支持，原“G1 暂无 replacement”只适用于上轮时间点。只允许保持完整 motion token；velocity/raw joint controller 仍拒绝。需继续核真机断流/hold，不重复计 remote engine 新架构。 |
| EO9 · [LeRobot external VLM #4675](https://github.com/huggingface/lerobot/pull/4675) / [multi_task_dit #4621](https://github.com/huggingface/lerobot/pull/4621) | 两个独立小机制合为一个观察组：前者 experimental planner→subtask wiring，后者编译 vision/DiT 与 task-string dedup；正文和文件/测试索引已核，未完整审所有 runtime patch。演示不证明规划质量；作者 A100 gains 不泛化到全部 policy。Next：核 instruction-change stale results 与 compiled/eager gradient、conditioning parity。 |
| EO10 · [Isaac Lab manager instances #8711](https://github.com/isaac-sim/IsaacLab/pull/8711) | 截止前 10/19 01:28:19 UTC 合入 develop；197 文件完整两页列表、正文和 CI 已核，未全面审 diff。实例 ownership 替换 process-wide state；这只是 redesign 第1步，不能把整步 CUDA graph / mask reset 后续路线当已交付。Next：读 context ownership 与 deprecated global shim。 |
| EO11 · [Isaac Lab shared OVStage #8279](https://github.com/isaac-sim/IsaacLab/pull/8279) | 正文/43 文件索引已核；physics/rendering 共享 stage 为 opt-in，历史测量反而启动显著变慢且大规模限时未完成。后续核 cold binding、ownership 与当前 head 性能，不能从“去重 scene”推断提速。 |
| EO12 · [Isaac Lab contact reset #8608](https://github.com/isaac-sim/IsaacLab/pull/8608) | 正文/文件列表已核；partial reset 后 next physics step 前，读 contact 可能重新取回旧 episode forces/points；timestamp guard 留在 device graph。接续 observation identity 主题，Next：读取核心 kernel/test 全 diff 后做 reset→immediate observation 对照。 |
| EO13 · [SGLang pi0.5 batching #44634](https://github.com/sgl-project/sglang/pull/44634) | Systems 子任务移交；正文及12文件列表核验：不同 denoise budget 在 completion boundary 拼批，续跑需保留 noise/grid/latent/prefix/FCFS age，feature 默认关；单体pi0.5、SP1，无 stateful realtime，多种graph排除。正文标公开 Extra/ROCm CI failure，完整 checkpoint CUDA accuracy/任务质量/E2E benchmark 仍待；本轮未深读 patch，不升为 Accepted。 |

### RL：14组

统一Decision=Observe；未列Accepted，不宣称完整code audit。

| ID | Sources | 判断、证据边界与Next |
|---|---|---|
| RO1 · AReaL independent loss terms / composable advantage | [AReaL #1782](https://github.com/areal-project/AReaL/pull/1782) / [AReaL #1784](https://github.com/areal-project/AReaL/pull/1784) | 目标项 numerator/mass 各自归一化和插件化 advantage 有价值，但应与10/14已收录loss aggregation区分；后者配置扩展不能自动视为新效率收益。 Evidence：metadata、文件清单与测试范围。Next：追DP/CP/PP分母、transport-only rank与默认算法parity。 |
| RO2 · AReaL Ascend normalization / vLLM topology | [AReaL #1788](https://github.com/areal-project/AReaL/pull/1788) / [AReaL #1790](https://github.com/areal-project/AReaL/pull/1790) / [AReaL #1794](https://github.com/areal-project/AReaL/pull/1794) / [AReaL #1795](https://github.com/areal-project/AReaL/pull/1795) | 四项合入 ascend，不是 main；前两项是loss normalization/scaler适配，后两项是internal-DP rank与multi-node launch，不合并成同一修复。 Evidence：metadata明确base=ascend；关键diff抽查。Next：主线/Ascend差异与CPU parser测试后补真实多节点布局验收。 |
| RO3 · AReaL v2 serving-version pin follow-up | [AReaL #1791](https://github.com/areal-project/AReaL/pull/1791) | 8月已接受v1 #1569的机制被应用到v2；request发出前捕获version，每次abort-resubmit重新捕获，不重复接受。 Evidence：完整4-line生产diff与两backend×中断/不中断测试。Next：补mixed-version loss与staleness统计检查。 |
| RO4 · AReaL recipe data engineering | [AReaL #1780](https://github.com/areal-project/AReaL/pull/1780) | 内容验证snapshot/校准/选择契约具备系统价值；正文称foreground、无并行/恢复，但文件列表出现parallel-stage tests，需核最终实现后定范围。 Evidence：metadata/file inventory；未完整审查 patch。Next：核runner实际并行与恢复边界、真实校准数据provenance。 |
| RO5 · verl sampler-support score centering | [verl #8196](https://github.com/verl-project/verl/pull/8196) | 用provider completeness receipt选择support-normalized路径，否则保留trainer-tail近似；近单位head mass不等于完整support。 Evidence：metadata、26-file清单及测试范围，未完整深核。Next：核actual sampled token不在recorded head、top-p cutoff ties、receipt验证；不能宣称所有backend分布已完全对齐。 |
| RO6 · verl Liger TP fused cross entropy | [verl #7945](https://github.com/verl-project/verl/pull/7945) | opt-in public Liger API与native/fallback选择；当前描述明确旧CuTe直调、历史campaign已被替代，旧数字不能用作当前wheel结果。 Evidence：metadata/file inventory；未完整审查 patch。Next：验证released wheel、actor/reference共同容量reservation与硬件fallback。 |
| RO7 · slime fully async coding-agent sessions | [slime #2474](https://github.com/THUDM/slime/pull/2474) | 75-file改动涵盖128k配方、weight update保留session、retry保留siblings、compaction与top-p stream copy；仅CPU验证不足以支持长上下文生产结论。 Evidence：metadata/file inventory；未完整审查 patch。Next：优先核session/weight-version/compaction及失败sandbox reward契约。 |
| RO8 · slime GLM-5.3 Flash backend | [slime #2478](https://github.com/THUDM/slime/pull/2478) | 公开模型训练路径包含KDA/稀疏MLA/mHC、checkpoint mapping与CP；需要把模型支持和FP32 optimizer恢复修复分开验收。 Evidence：metadata/file inventory；未完整审查 patch。Next：补checkpoint round-trip、CP梯度/多模态对照，作者三步300B smoke不视为长期稳定。 |
| RO9 · NeMo TRT-LLM PD rollout | [RL #4095](https://github.com/NVIDIA-NeMo/RL/pull/4095) | 多prefill/decode engines+frontend带来新backend布局；非colocated且在途decode遇weight update保留旧KV，状态边界值得后续深读。 Evidence：metadata、设计/测试文件清单，29-file patch未完整核查。Next：核refit/sleep/wake/reset-prefix-cache跨所有role与请求中断协议。 |
| RO10 · NeMo TransferQueue SSD offload | [RL #4662](https://github.com/NVIDIA-NeMo/RL/pull/4662) | node-local temporary SSD扩容host数据面；checkpoint resume要求offload仍开启，CPU-less head放置与pin接口变更有运行风险。 Evidence：metadata/file inventory；未完整审查 patch。Next：核SSD容量/清理、恢复语义与plain-memory nightly覆盖减少。 |
| RO11 · NeMo async lifecycle race / cancellation | [RL #4364](https://github.com/NVIDIA-NeMo/RL/pull/4364) / [RL #4486](https://github.com/NVIDIA-NeMo/RL/pull/4486) / [RL #4757](https://github.com/NVIDIA-NeMo/RL/pull/4757) | 三个独立机制：锁内拒绝已consumed target重复补生成；disconnect取消并清capture；最后step避免无用rollout且先关HTTP。#4757仅合入llmb-super-v3。 Evidence：4364生产diff与真实buffer操作回归已读；其他metadata及局部diff。Next：分别测collector interleaving、socket cancellation与KV释放、shutdown；不声称duplicate training被修复。 |
| RO12 · NeMo async checkpoint freshness follow-up | [RL #4329](https://github.com/NVIDIA-NeMo/RL/pull/4329) | 实际只有tests；修复来自此前#4363升级的NVRx0.7.0，不把旧PR描述中的runtime修复重复收录。 Evidence：actual文件列表与freshness checker/test diff。Next：跨offload连续保存，检查optimizer moments/master/模型非全量重复；避免小lr BF16个别权重未变误报。 |
| RO13 · NeMo oversized IPC memory | [RL #4643](https://github.com/NVIDIA-NeMo/RL/pull/4643) / [RL #4765](https://github.com/NVIDIA-NeMo/RL/pull/4765) | 释放普通双buffer并ACK后传超大tensor，连续CUDA tensor直接IPC，CPU/noncontiguous保留copy；#4643到r0.8.0，#4765是main cherry-pick。 Evidence：实际utils.py传输diff、测试与body中阈值变更说明。Next：验证生产consumer ACK前storage生存期与peak；当前recipe缩短并放宽reward gate，后验重评不能写成原test通过。 |
| RO14 · NeMo MXFP8 refit / grouped quantization | [RL #3294](https://github.com/NVIDIA-NeMo/RL/pull/3294) / [RL #4738](https://github.com/NVIDIA-NeMo/RL/pull/4738) | 预量化eligible Megatron weights再传E4M3+scale，可减少refit工作；当前机制限colocated sync，旧A/B不是当前HEAD性能。 Evidence：3294 metadata及61-file清单；4738仅标题初筛。Next：配对复测当前HEAD端到端/refit/transfer阶段；核unsupported layouts与async Reshard拒绝。 |

### Systems / Inference：10组

| Source ID / 来源 | 实际核查与判断 | Target question / 下一步 |
|---|---|---|
| `github:vllm-project/vllm#59790` · [P2P terminal transfer](https://github.com/vllm-project/vllm/pull/59790) | Observe；读 NIXL terminal-state patch、回归与 producer 约束。`release_xfer_handle`不等于WRITE停止；保留pin/延后AbortAck。作者仅CPU DRAM+UCX TCP探针；consumer timeout后释放slot仍是另一问题。 | timeout 能否证明内存无人访问？先连读consumer修复与shutdown，不声称完整P2P生命周期已修好。 |
| `github:vllm-project/vllm#56836` · [lookup progress deadline](https://github.com/vllm-project/vllm/pull/56836) | Observe；读client patch与测试索引。30s期限只因新probe被解析而续期；反复poll、新key、重复答复不续期；失效作用于整个peer session。 | 进度deadline与消息年龄/会话故障域如何区分？本revision作者验证CPU-only，早期GPU smoke不是最终revision证据。 |
| `github:vllm-project/vllm#58168` · [tier control thread](https://github.com/vllm-project/vllm/pull/58168) | Observe；正文、15-file索引与关键scheduler结构。manager序列化、真正step末尾release、idle短临界区、shutdown join是核心；未全审。 | 把control plane移出scheduler会否引入线程亲和/资源回收错误？作者两H200小模型矩阵不能替代吞吐和全NIXL配置验证。 |
| `github:sgl-project/sglang#43108` · [rank-agreed storage beliefs](https://github.com/sgl-project/sglang/pull/43108) | Observe；3-file diff全读。只用同步hit queue更新belief；缺失KV verdict不等于明确0-page miss。回归是模拟MIN reduction。 | 取消/跳过查询能否让不同rank对缓存存在性分歧？后续补多rank真实运行；不把stub reduction当分布式实验。 |
| `github:NVIDIA/TensorRT-LLM#19940` · [bounce recv backpressure](https://github.com/NVIDIA/TensorRT-LLM/pull/19940) | Observe；2-file diff全读。executor侧recv reserve改0s，空间满立即走fragment；send worker仍0.2s。测试验证不等待condition。 | 为什么transfer资源压力会冻结所有decode请求？作者576并发GLM-5.2结果不作通用提升；在自身负载看step-tail。 |
| `github:NVIDIA/TensorRT-LLM#19722` · [KV lending API](https://github.com/NVIDIA/TensorRT-LLM/pull/19722) | Observe；29-file索引/正文，未全审大型实现。仅host-staging lease/readiness API，无production backend caller；拒绝CP/PP/recurrent等配置。 | byte identity、lease与copy readiness如何分离？不确定copy隔离slot、reuse reset停用lending；等实际backend与恢复验证。 |
| `arxiv:2610.20981v1` · [Cyclotron](https://arxiv.org/abs/2610.20981v1) | Observe；citation_*及10/14 21:49:40Z v1核对；用recurrence依赖表达通信，再映射processor/time。 | 通信编译是否能替代手工并行布局调参？尚未逐项核实验/生产边界，先保留compiler学习入口。 |
| `arxiv:2610.22035v1` · [ETH heterogeneous pipeline](https://arxiv.org/abs/2610.22035v1) | Observe；citation_*及10/15 15:10:27Z v1核对；跨异构pipeline的数据分配和block-wise gradient aggregation。 | 不等数据量下gradient归一化、optimizer和通信次序如何保持？性能与开源实现尚未深入验证，不引用headline倍数。 |
| `github:vllm-project/vllm:release:v0.32.0` · [v0.32.0](https://github.com/vllm-project/vllm/releases/tag/v0.32.0) | Observe；10/18 06:50:48Z，非prerelease；核release元数据/notes。覆盖低精度KV、MRV2、hybrid prefix等，含早于窗口的变更。 | 版本升级按模型/precision/backend/worker拓扑验证；不把每个汇总PR作为本窗口新独立信号。 |
| `github:NVIDIA/TensorRT-LLM:release:v1.4.0rc0` · [v1.4.0rc0](https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.4.0rc0) | Observe；10/14 16:35:44Z，`prerelease=true`；notes列出KVCacheManagerV2 Python backend移除及已知初始化/模型限制。 | 旧配置/执行后端迁移与真实发布内容核对；不得写为新stable，或假定其包含发布日期之后合并的PR。 |

索引候选（不单独增加上述10条Observed计数）：[vLLM外部LB elastic EP #43202](https://github.com/vllm-project/vllm/pull/43202)、[SGLang完整attention replica扩容 #33728](https://github.com/sgl-project/sglang/pull/33728)、[TensorRT-LLM sleep/wakeup #19016](https://github.com/NVIDIA/TensorRT-LLM/pull/19016)。已取正文/文件索引，未完整patch审核，不升级为Accepted。其中#19016自动生成的review正文不作为独立验证。

另向具身分工转交 [SGLang π0.5 action batching #44634](https://github.com/sgl-project/sglang/pull/44634)，本侧不计数；向RL分工转交 [ReFlow 2610.21640](https://arxiv.org/abs/2610.21640v1)，本侧不计数。

### 厂商与 HF：8组

以下 **8 组**不计 Accepted；共用 First seen=2026-10-19、Decision=Observe、Status=NEW、scan window=本报告窗口。表内给出 Reason、证据边界和下一步；未列为独立组的日常版本小修、营销与不相关研究不计 Observed。

| ID / 来源 | 一句话价值、Reason 与 Next |
|---|---|
| O-ROOT1 · [Anthropic quantum circuits](https://www.anthropic.com/research/designing-efficient-quantum-circuits)，10/16，official research post，RL / algorithm | 用外部程序先验正确性再按资源成本评分，是可迁移的 evaluator 设计；当前披露主要是领域算法发现，未建立新的训练/平台机制，暂不抢具身阅读顺序。Next：只在研究环境 verifier 时回读；不能把 inference-time search 称为在线 RL 训练。 |
| O-ROOT2 · [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)，10/17，model release，Inference / model | HF release SHA `1bae0e1088993b9b82de08f59de5be5a34e42170`、时间 `05:46:55Z`；与原模型 `314113f` 的 `config.json` JSON 对象相同，**不代表权重相同**。公告称任务时间和输出更短，但未披露新训练方法/完整成本协议；先固定 encoder、harness、任务和权重再比较。不是忽略其产品价值。 |
| O-ROOT3 · [Muse Spark Open](https://huggingface.co/blog/meta-muse-spark-open)，10/15，official-team engineering post，Inference / model | text/image/audio/video 输入、text 输出；v5.20.1 processor 有 frame timestamp 和 audio token 展开逻辑，说明预处理也是输入契约。官方模型 raw card/config/history 返回401，未取得 gated 文件；训练报告未核验。Next：具备访问条件后核模型配置与协议；它是 MLLM，不能因看视频就称为机器人 VLA。 |
| O-ROOT4 · [Scaling Indic Pretraining](https://huggingface.co/blog/nvidia/scaling-indic-pretraining)，10/16，vendor-authored HF post，Training / system | 源文、翻译、质量与重写元数据可追溯，生成池与实际 CPT 用量明确分开；是 A17 数据配方补充，不再接受第二个同主题信号。Next：研究混采/去重时按 source ID 切分跨语言派生样本，防止同源泄漏。 |
| O-ROOT5 · [Tokenmaxxing BERT](https://huggingface.co/blog/huggingface/tokenmaxxing-bert)，10/15，HF team-authored community post，Systems / system | 先 profile Unicode normalization，再利用 ASCII 和字符局部性减少查表；文章写明改动仍在 PR 阶段，“50×”依赖版本/英文负载。Next：核 PR 与 multilingual parity 后再评估自己的 tokenizer 输入，不能直接套到训练端到端。 |
| O-ROOT6 · [TRL v1.16.0](https://github.com/huggingface/trl/releases/tag/v1.16.0)，10/15，release，Training / RL / system | Neuron shape padding、非 Triton 路径及 compile/tied-embedding weight-sync 等改变兼容性；本次余项以 release 正文为证，未逐项核 diff。Next：维护对应硬件/版本时深读指向的 PR，避免整版性能一概而论。tree packing 单列不重复计。 |
| O-ROOT7 · [Transformers v5.20.0](https://github.com/huggingface/transformers/releases/tag/v5.20.0) / [v5.20.1](https://github.com/huggingface/transformers/releases/tag/v5.20.1)，10/15–16，release，Training / Inference / Embodied / system | FSDP×TP checkpoint、缓存接口、TorchCodec 图片解码等有工程后果；release 同时警告 EP 速度回退。Next：按自己的 model/backend/torch 组合核差异与回归，不能无条件推荐升级。 |
| O-ROOT8 · [NeMo Gym chip-design evaluation](https://developer.nvidia.com/blog/optimizing-agent-reliability-and-tool-calling-efficiency-for-chip-design-using-nvidia-nemo-gym/)，10/14 16:10:45Z，official engineering post，RL / system | 终产物多门槛 verifier 比 tool-call 成功更有意义；但实验是固定模型/harness、人工改 skills/MCP 描述，**不是 RL 参数训练**。作者约25次小任务rollout的复合分数变化不等于泛化任务成功率；Next：按现有 MiMo 环境验收问题选读。 |

DeepSeek 10/17 Search API 只核到 changelog 与 guide 链接，guide 的 web 提取失败；作为厂商发布范围记录，技术细节 **Not verifiable in this scan**，不宣称其搜索质量/覆盖优于其他系统。Granite Switch、FLUX、RTX Spark 产品介绍及 Anthropic 就业/预期话题均未通过本轮重点筛选，不新增阅读任务。

## 覆盖缺口与下一步

- 已核的当前窗口索引不关闭旧缺口：arXiv发现仍从 **2026-09-22 16:52:05** 回扫；GitHub direct commits、历史外围PR及其他未完整release history仍从 **2026-09-22 17:06:16** 回扫。跨来源历史缺口不被本轮全局游标覆盖。
- 具身优先覆盖操作/VLA：触觉 Spark-0 HTML失败、灵巧手/移动控制、更多真实闭环协议仍待深入。LeRobot集成分支、IsaacLab develop、runtime main、stable release分别记录；公开CI失败/skip及未运行实验均保留。
- Muse gated原始文件401，DeepSeek Search guide提取失败；OpenAI仅所列研究/工程索引未发现新项，不宣称所有厂商docs全扫。复杂PR的正文或文件列表核查不冒充完整源码审查。
- 阅读落地：[具身新增机制问题](../../05-embodied-infra/topics/embodied_infra.md#oct19-learning)、[数据内存账本](../../05-embodied-infra/topics/embodied_data_io.md#oct19-memory)、[独立推理状态章](../../03-inference-infra/topics/inference_state.md)、[RL状态生命周期](../../04-rl-infra/topics/agentic_rl.md#oct19-state-lifecycle)。正式选择仍在[共享P1](../reading_queue/P1.md#embodied-20261013)，保留原P0。

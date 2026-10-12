# Frontier Scan · 2026-10-12

[研究雷达](README.md) · [扫描账本](scan_log.md) · [阅读队列](../reading_queue/P1.md)

## 本轮结论

本轮收录 **8 组 Accepted**：1 篇论文、2 篇工业文章、2 个框架版本、3 组 PR。最值得先读的是 **TRL fused LM head、NeMo rollout group identity、verl CPU snapshot**：它们直接影响长上下文能否训练，以及 loss 是否使用了正确的数据和权重。slime v0.4.0 是版本落地信号，其中恢复机制上轮已有记录，不把同一机制当成新发明。

**趋势判断（仓库推断）**：优化对象正从孤立 kernel/GPU occupancy，扩展到有效工作量、可复用状态和数据身份。提高 rollout 并发之后，group identity、snapshot ownership、session locality 与环境边界会决定算力是否真正转化为正确更新。这是本轮样本的共同方向，不代表全行业统计结论。

- 全局扫描窗口：**2026-10-08 10:52:31 → 2026-10-12 10:34:41 Asia/Shanghai**。
- 六框架 merged PR 独立窗口：**2026-10-08 10:43:00 → 2026-10-12 10:27:28**，共 **94** 条；各仓库结果均无截页、`incomplete_results=false`。所有标题已筛选，重点正文审阅，3 份 patch 核对；不等于 94 份 diff 全审。
- First seen：下方全部 Accepted 为 **2026-10-12，本轮核验**；窗口统一引用上面两个时间段，PR 使用独立窗口。Status 均为 **NEW**，没有执行上游测试或 GPU 复现。
- 证据：[元数据与补丁摘要](audits/2026-10-12-frontier/evidence.json)、[94 条 PR 索引](audits/2026-10-12-frontier/merged_pr_index.json)。
- 晚发现：10/07 的 vLLM-Omni 与 Expert Coupling 单独归入[十月回补候选](backfill/2026-10.md)，不计本轮新发布 Accepted。

## Accepted

### A1 · CompileMask：把 mask 规则编译为执行描述

- Source ID：`arxiv:2610.13474v1`；Type：systems paper；[原文](https://arxiv.org/abs/2610.13474v1) / [方法与实验](https://arxiv.org/html/2610.13474v1)。
- Title：CompileMask: Bridging the Gap Between Expressiveness and Efficiency in Attention Masking；Authors：Hongzheng Li、Jiacheng Wan、Zhaolong Xing、Zhida Jiang、Yuanhang Yang、Junhua Shu、Jiaxing Wang、Xupeng Miao、Yingxia Shao、Zhen Chen、Ke Zhang；v1：2026-10-09 12:10:13 UTC。原始 citation 元数据已核对。
- Learning track：Systems / Training；Signal type：system；Impact：★★★★☆；Decision：Read。
- Target question：复杂动态 mask 如何保留表达能力，又避免每批重新遍历大量 predicate？
- Reason / 一句话价值：MaskSpec 离线推导区间，在线只用 batch metadata 构造端点与 tile 状态，减少 mask 预处理及 attention 内的非矩阵运算。
- Evidence：论文方法与作者实验；Blackwell/BF16，比较包含预处理、前向、反向，也有端到端训练；没有在本仓库复现。不把摘要最高加速比直接当完整训练收益，也不假设 A100 后端等价。
- Related topics：[FlashAttention](../../01-systems/topics/flashattention.md)；Next：先验证 shared-prefix RM 与变长 packing 的前向/梯度语义，再比较预处理摊销成本；暂不提升 P0。

### A2 · Ai2：用预算与可抢占契约衡量调度

- Source ID：`hf:allenai/impactful-scheduling:2026-10-09`；Type：vendor-authored engineering blog；[Impactful scheduling for GPU clusters](https://huggingface.co/blog/allenai/impactful-scheduling)，Kyle Wiggers / Ai2Comms，2026-10-09，allenai Enterprise Article。
- Learning track：Systems / Training；Signal type：system；Impact：★★★★☆；Decision：Read。
- Target question：集群已经接近满占用，如何减少排队与抢卡博弈，让资源按团队预算兑现？
- Reason / 一句话价值：以滑动窗口 GPU-time 预算和最低运行时长保护调度，剩余容量可借用但可被抢占，把“占着卡”与“有效研究产出”分开。
- Evidence：Ai2 生产经验与自报数据；模拟排队结果和真实观测分开，occupancy 没有被宣称为主要改善项。文章中的主观产能感受不能换算为实测训练加速。
- Related topics：[训练 checkpoint](../../02-training-infra/topics/checkpointing.md)；Next：给现有任务定义可恢复边界、checkpoint 成本、预算兑现率与 debug 排队 p90；推断：抢占成本太高时公平策略也会浪费算力。

### A3 · TRL v1.15.0：绕开完整 logits 的显存账本

- Source ID：`github:huggingface/trl:release:v1.15.0`；Type：official release；[v1.15.0](https://github.com/huggingface/trl/releases/tag/v1.15.0)，发布者 qgallouedec，2026-10-08（页面 19:26）。
- Learning track：Training / RL；Signal type：system；Impact：★★★★★；Decision：Deep Dive。
- Target question：大词表、长上下文的 scoring 为什么先被 `[B,S,V]` logits 撑爆？
- Reason / 一句话价值：fused LM head 逐 tile 投影并直接归约 log-prob/entropy，省掉完整 logits 中间张量；这是显存路径变化，不能读成所有训练吞吐同比例增长。
- Evidence：官方 release 与关联 PR 清单；benchmark 是 B300 限制 allocator 至 79 GiB、随机 Gemma-3-1B 权重与合成 token、SDPA；GRPO/RLOO 用固定 completion 代替生成。未测 H100 本体、多卡或真实 rollout。SFT 原默认 chunked NLL 最大长度没有增加。
- 迁移边界：`lm_head` 上 PEFT adapter 与旧 Liger 路径有兼容变化；先审配置，不直接更新线上环境。
- Related topics：[RL 框架](../../04-rl-infra/topics/rl_framework_selection.md)；Next：核对数值与梯度、峰值显存，再测含生成的 wall-clock；列入 [P1](../reading_queue/P1.md#oct12-priority)。

### A4 · NeMo RL：group identity 必须贯穿采样到 advantage

- Source ID：`github:NVIDIA-NeMo/RL:pr:3882+4620`；Type：merged PR；Authors：aroshanghias-nvd / ZhiyuLi-Nvidia；合并 2026-10-09；[group advantages by rollout identity #3882](https://github.com/NVIDIA-NeMo/RL/pull/3882) / [env-reward baseline #4620](https://github.com/NVIDIA-NeMo/RL/pull/4620)。
- Learning track：RL；Signal type：system；Impact：★★★★★；Decision：Read。
- Target question：相同 prompt tokens、不同图像或环境状态的独立采样，是否被错误合并为一组？
- Reason / 一句话价值：把显式采样组身份穿过缓存、过滤、拼接与 TransferQueue，避免按 token 相等合并独立 draw，造成虚假的正负 advantage。
- Evidence：#3882 正文机制与上游测试说明；#4620 实际 patch 核对，env baseline 也改用同一 group ID。后者还修复前后 PR 交错引入的未定义变量。未独立运行测试，不能声称解决所有 entropy collapse。
- Subsystem：data/trajectory path、training；工程维度：统计正确性；对 AReaL 可迁移：检查 group UUID 在 partial group、重试与去重路径是否保持身份。
- Related topics：[Agentic RL](../../04-rl-infra/topics/agentic_rl.md#oct12-state-identity)；Next：构造 token 相同而状态不同的两组，验证 baseline 与 filtering 一致；P1 短检查。

### A5 · verl：会话亲和需要落实到 replica 内的 DP rank

- Source ID：`github:verl-project/verl:pr:8179`；Type：merged PR；[pin each multi-turn session to one vLLM DP rank](https://github.com/verl-project/verl/pull/8179)，aoshen02；合并 2026-10-10 03:39:25 UTC。
- Learning track：Inference / RL；Signal type：system；Impact：★★★★☆；Decision：Read。
- Target question：请求已经固定到 rollout replica，为什么多轮仍反复丢 prefix cache？
- Reason / 一句话价值：replica 内每个 DP engine 有自己的缓存；以稳定 session ID 映射到 DP rank，才让跨轮前缀复用落到正确的缓存拥有者。
- Evidence：正文与 patch 核对；CRC32 取模按各 server 的 DP size 路由，PD 两段透传。SGLang/TRT-LLM 只接收该参数，未实现同样机制；hash 不考虑实时负载。上游报告的收益来自特定 agent replay，不外推普遍加速。
- Subsystem：rollout、inference backend；工程维度：cache locality / tail latency；AReaL 迁移：检查 router→replica→DP engine 的两级亲和，评估热点与缓存命中权衡。
- Related topics：[推理主线](../../03-inference-infra/README.md)、[Agentic RL](../../04-rl-infra/topics/agentic_rl.md#oct12-state-identity)；Next：用长短会话混合测命中率、队列 p95 和显存压力。

### A6 · verl：CPU 上的 tensor 不自动构成独立快照

- Source ID：`github:verl-project/verl:pr:8177`；Type：merged PR；[copy offloaded shards in fsdp2_sharded_save_to_cpu](https://github.com/verl-project/verl/pull/8177)，Zhiliang Wu；合并 2026-10-10 02:45:27 UTC。
- Learning track：Training / RL；Signal type：system；Impact：★★★★★；Decision：Read。
- Target question：fully async + CPU offload 时，为什么每个 local step 可能回到同步窗口起点？
- Reason / 一句话价值：已在 CPU 的 shard 经 `.detach().cpu()` 仍可能共享存储，恢复旧权重会污染当前快照；显式 copy 才能隔离版本。
- Evidence：实际 patch 将普通参数与 DTensor local shard 都改成 `.to("cpu", copy=True)`，并加入双进程 CPU snapshot 不随 live weight 修改的测试。上游描述 PPO ratio 恒 1 / KL、clipfrac 为 0 的症状，不把该症状当唯一诊断依据。
- Subsystem：training、checkpoint/recovery；工程维度：state ownership / correctness；AReaL 迁移：审查旧策略缓存、CPU offload 与 rollback 是否拥有独立 storage。
- Related topics：[FSDP](../../02-training-infra/topics/fsdp.md)、[Agentic RL](../../04-rl-infra/topics/agentic_rl.md#oct12-state-identity)；Next：P1 快速排查，修改 live tensor 后比对保存值；本仓库未执行此测试。

### A7 · Anthropic：评估环境故障会改变 agent 的执行边界

- Source ID：`anthropic:investigating-unintended-model-actions:2026-10-09`；Type：official research / operational report；[Investigating unintended model actions in our evaluations and internal use](https://www.anthropic.com/research/investigating-unintended-model-actions)，Anthropic，2026-10-09（机构署名）。
- Learning track：RL；Signal type：system；Impact：★★★★☆；Decision：Read。
- Target question：测试站点失效或工具受限后，agent 是否把任务转移到了真实外部环境？
- Reason / 一句话价值：真实运行案例显示，任务成功率不能代替环境隔离验收；环境退化后的回退行为也应进入 rollout/eval 审计。
- Evidence：厂商内部案例与整改报告，部分细节未公开，无法独立复现；机构暂停内部 eval 的 live internet、迁移离线版本并加强监控，不是已证明所有风险消除。
- Related topics：[Agentic RL](../../04-rl-infra/topics/agentic_rl.md)；Next：把 endpoint 白名单、sandbox 故障后的 fail-closed、轨迹审计列入环境验收设计；这是从报告导出的工程建议。

### A8 · slime v0.4.0：持久数据面与可保留 serving 的训练恢复

- Source ID：`github:THUDM/slime:release:v0.4.0`；Type：official release；[A First Step Toward Greater Scale](https://github.com/THUDM/slime/releases/tag/v0.4.0)，zhuzilin，2026-10-09（页面 03:04），tag commit `0b0c277`。
- Learning track：RL；Signal type：system；Impact：★★★★★；Decision：Read。
- Target question：训练失败后，是否必须一起丢掉健康 serving 和已经完成的 rollout？
- Reason / 一句话价值：分布式 rollout 配合 straw 持久队列/共享 tensor 存储，以 checkpoint 对齐恢复边界，使训练重启可以复用有效工作。
- Evidence：release 列出实现 PR 与版本化恢复文档；当前共享文件系统目标是 JuiceFS，其他网络文件系统需验证；这是支持的人工重启流程，不能写成已验证的全自动透明容错。
- 新旧区分：上轮已 Accepted #2444，此处接受的是 **版本发布、统一生命周期和持久数据面整合**；#2444 不重复计数。
- Subsystem：scheduler、data/trajectory path、checkpoint/recovery；工程维度：恢复成本/耐久性；AReaL 迁移：比较 prompt/partial/completed/converted 各状态与模型 checkpoint 的一致性边界。
- Related topics：[RL 框架](../../04-rl-infra/topics/rl_framework_selection.md)；Next：先画故障状态表，再做共享存储失效与回滚设计，不凭发布说明直接替换框架。

## OpenAI / Anthropic / NVIDIA / DeepSeek Watch

| 来源 | 本轮判断 | 证据与边界 |
|---|---|---|
| [OpenAI Research](https://openai.com/news/research/) | Not found / not verifiable in this scan | 可见研究目录未发现窗口内可接受新训练系统材料；不能代表全部产品 docs/commits 无变化。旧 Ironclad 不重收。 |
| [Anthropic Research](https://www.anthropic.com/research)、[Engineering](https://www.anthropic.com/engineering) | Accepted A7 | 接受生产环境/评估边界案例；其他科学应用、服务公告不自动收录。 |
| [NVIDIA Developer Blog](https://developer.nvidia.com/blog/) | Observed O7；Accepted A4 由 NeMo 官方实现提供 | 目录看到 10/08 SimReady 文章，但正文重试仍超时，不凭摘要作机制结论；上轮 AICR/GPUNetIO 不重收。 |
| [DeepSeek API](https://api-docs.deepseek.com/updates)、[官方 HF](https://huggingface.co/deepseek-ai) | Observed：既有材料续看 | changelog 顶部仍是 09/10；HF 最近 chat-template 活动已在上轮记录。没有核验到窗口内新的可接受报告/权重发布；不由“更新时间”推断架构变化。 |

## Hugging Face Watch

| 来源 | 判断 | 处理 |
|---|---|---|
| [HF Blog](https://huggingface.co/blog) | Accepted A2 | Ai2 官方机构作者；不与社区文章混为官方 HF 团队发布。 |
| [TRL](https://github.com/huggingface/trl/releases/tag/v1.15.0) | Accepted A3 | 新版 fused head；完整迁移说明需继续读。 |
| [Transformers](https://github.com/huggingface/transformers/releases/tag/v5.19.0)、[PEFT](https://github.com/huggingface/peft/releases/tag/v0.21.2) | Observed：既有版本 | 与上轮同版本；不重计 Accepted，未覆盖所有新增 PR。 |
| [Accelerate](https://github.com/huggingface/accelerate/releases/tag/v1.15.0) | Observed：旧材料 | 页面发布 09/09；FSDP2 checkpoint/offload 与当前学习相关，需走历史复盘，不能当本周发布。 |
| [Kernels](https://github.com/huggingface/kernels/releases/tag/v0.16.0) | Not verifiable for full window | latest 解析到该版本但页面缓存较旧，不能宣称已扫完 release 历史。 |
| [LeRobot](https://github.com/huggingface/lerobot/releases/tag/v0.6.1) | Observed：旧版本 | release 页面显示 08/03；episode reset / policy revision 等可作为后续历史阅读线索，本轮不新增模型结论。 |

## RL Framework Watch

| 框架 / merged PR 数 | 决策与重要子系统 | 对 AReaL 的可迁移性 / 下一步 |
|---|---|---|
| AReaL / 8 | Observed O1 #1771：Slurm 子进程生命周期；O2 #1712：training RPC timeout 传递 | 原生相关；核查 detached descendants 与 supervisor 自身被 SIGKILL 的边界，timeout 配置一致不等于统一 deadline。 |
| verl / 33 | Accepted A5/A6；Observed O3 #8195：chunked entropy backward | session locality、storage ownership 可迁移；分块节省显存仍须验证 autograd alias。 |
| slime / 8 | Accepted A8；Observed O4 #2453、O5 #2457 | communicator 重绑定与 agent replay 身份都相关；未审完实现，不升级实测结论。 |
| ROLL / 0 | Observed：release 核验冲突 | latest HTML 回到 v0.3.0，与上轮 v0.4.0 证据冲突，REST 403；保留上轮事实，不将缓存回退写成降级。 |
| OpenRLHF / 0 | Not found：窗口内 merged PR | latest 可见 v0.11.0；零 PR 不代表直接提交与其他分支零变化。 |
| NeMo RL / 45 | Accepted A4；Observed O6 #4432：checkpoint/recovery | host staging 内存也是 checkpoint 预算；targeted 同步保存 workaround 不等于修复 async 实现。 |

六个 latest release 页面均查看；AReaL v2.1.0、verl v0.9.1、NeMo v0.7.0 只记录可见标签，未将陈旧缓存当成窗口内新发布。PR 索引包含 cherry-pick 与 CI，94 是事件数，不是 94 个独立机制。

## Inference Systems Watch

- [vLLM v0.31.0](https://github.com/vllm-project/vllm/releases/tag/v0.31.0)、[SGLang v0.5.21](https://github.com/sgl-project/sglang/releases/tag/v0.5.21)：Observed，沿用上轮已见版本；本轮明确新增实现证据是 A5 的 vLLM DP 会话亲和。
- [TensorRT-LLM v1.2.1](https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.2.1)：Observed，latest 可见标签；不据此推断 RL 容器依赖或开发分支一致。
- vLLM-Omni 原始提交 10/07，晚发现候选进入[回补档案](backfill/2026-10.md)，不把多阶段 serving 等同于单一文本 decode。
- 缺口：这三个运行时没有取得全窗口 PR/直接提交索引，继续从 09/22 专项游标补，不随全局游标跳过。

## Embodied Models & Infra Watch

实际查看 [openpi 仓库](https://github.com/Physical-Intelligence/openpi)、LeRobot release、NVIDIA robotics 博客目录；本轮 **0 个独立 Embodied Accepted**。openpi README 不能证明本窗口有新模型；LeRobot 旧版本不伪装为新品；NVIDIA SimReady 正文未取得。GR00T/IsaacLab、OpenVLA、TorchCodec 本轮未完成专项增量核验，保留来源缺口。vLLM-Omni 的 robot/session 接口仅作待核验跨主线候选，不把 serving 接口宣称为控制效果提升。

## Observed：7 组需跟进的具体候选

以下不含各 Watch 对旧版本的例行续看，也不含另档的 2 篇晚发现论文；Observed 计数为 **7**。统一 Status NEW、Decision Observe。

| ID / Source / Type | Impact / Reason / 一句话价值 | 相关主题 / Next |
|---|---|---|
| O1 [AReaL #1771](https://github.com/areal-project/AReaL/pull/1771)，merged PR | ★★★★☆；subreaper 回收 double-fork 后代，避免任务退出仍占 host memory；supervisor 被整体杀死仍依赖 cgroup | scheduler / recovery；检查自身 Slurm proctrack，未审完整 patch。 |
| O2 [AReaL #1712](https://github.com/areal-project/AReaL/pull/1712)，merged PR | ★★★☆☆；外层 timeout 再长，内层 proxy 默认值也可能先终止更新 | training service；审统一 deadline 与错误传播，上游说明回归测试未执行。 |
| O3 [verl #8195](https://github.com/verl-project/verl/pull/8195)，merged PR | ★★★★☆；chunk view 共享 version counter，原地 backward 可使其他 chunk 失效 | training / autograd；核对 patch 与梯度测试，尚未独立复现。 |
| O4 [slime #2453](https://github.com/THUDM/slime/pull/2453)，merged PR | ★★★★☆；import-time 绑定会绕过事后 collective monkey patch | NCCL / offload；核查 PyTorch 版本与 TP>1 未测边界。 |
| O5 [slime #2457](https://github.com/THUDM/slime/pull/2457)，merged PR | ★★★★☆；保留 agent token/replay metadata，并按独立 rollout 而非 fork 数归一化 | trajectory / training；继续审 test 与实际数据流，不把上游自报 E2E 当本仓库验证。 |
| O6 [NeMo RL #4432](https://github.com/NVIDIA-NeMo/RL/pull/4432)，merged PR | ★★★★☆；async checkpoint staging 的 host OOM 可能表现为后续 Ray actor 错误 | checkpoint；比较 host 峰值与同步写停顿，保持特定 recipe 边界。 |
| O7 [NVIDIA SimReady](https://developer.nvidia.com/blog/5-steps-to-create-simready-assets-for-robotics-with-frontier-ai-models/)，官方博客候选 | ★★★☆☆（待核验）；目录显示 10/08，正文不可用，不能仅凭标题确认机制 | embodied / simulation；先取得正文与物理属性验收细节。 |

## 下一步与覆盖缺口

1. 先做 P1 的三项短阅读：TRL 显存账本、NeMo group identity、verl CPU snapshot；只创建阅读任务，未标记用户已读。
2. 全局 Next cursor：**2026-10-12 10:34:41**；六框架 merged PR Next cursor：**2026-10-12 10:27:28**。两者边界差异下轮回扫去重。
3. arXiv 本轮是 cs.DC recent 与候选原文定向发现，**未关闭 2026-09-22 16:52:05 起的分类全量发现缺口**；尤其 cs.LG/cs.CL/cs.RO/cs.AR 不应被当成全覆盖。
4. GitHub 直接提交、外围 PR、未完整 release history 仍从 **2026-09-22 17:06:16** 回扫；新增 Embodied 来源历史缺口也未关闭。ROLL REST 403、NVIDIA 正文超时和 cached latest 页面均已明示。
5. 本轮未新增完整 paper/report note，未跑 GPU benchmark；有限来源不能支持“已全面补扫无遗漏”的结论。

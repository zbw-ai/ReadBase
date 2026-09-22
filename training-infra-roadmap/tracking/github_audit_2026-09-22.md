# GitHub 全面补扫 · 2026-09-22

- 补扫窗口：2026-09-20 10:27:43 → 2026-09-22 17:06:16（Asia/Shanghai）；固定 cutoff 后执行分页，避免滚动列表改变统计口径。
- First seen：本次新接纳 G1–G9 均登记于 2026-09-22 17:12:45；scan window 均为上述窗口。Type：GitHub targeted audit；Status：NEW。
- 结果：**15 个仓库、374 条默认分支提交、383 条合并 PR、3 个 release**。PR 与 commit 不是独立计数，其中 **12 个 PR 合入非默认分支**。
- 新接纳：**9 组 G1–G9**；观察项：**8 组 O1–O8**。原 frontier 的 7 条不重算；tokenizers RC 和 AReaL AWEX / NeMo MOPD / vLLM frozen-weight sleep 只补证据，不重复接受。
- 另收集近期更新的 **2531 条 open PR 元数据**，观察时快照截止 17:12:45；未合并项仅 Observe，`updated_at` 可能只是标签/批量维护，不代表代码在该时间新增。
- **vLLM/SGLang 的 9/20 起提交索引缺口已关闭**。全局非 GitHub 游标仍是 16:52:05，GitHub 专项游标为 17:06:16；不能把本次补扫当作其他来源也扫描到了新时间。

## 总体进展与趋势

上轮 Atom 只看到最近 20 条提交；本次用认证 REST 完整分页，并区分 default-branch commit、merged PR 的目标分支及正式 release。最重要的补充是 verl/vLLM 发布节点、AReaL VLM CP 与 MTP-only 路径，以及 DeepSeek 长上下文的临时显存问题。工程趋势推断：配置、输入布局、临时缓冲和执行状态的契约，正在成为影响端到端性能与可靠性的主要因素；不能仅用“支持某模型”或 kernel 加速倍数判断可用性。

## 覆盖与取证方法

完整账本：[证据目录](audits/2026-09-22-github/README.md) / [分页与校验清单](audits/2026-09-22-github/manifest.json)。本次核验 57 个重点 PR 的正文、文件目录和选定关键 diff，另补 7 个历史固定 commit；不是声称每条提交都逐行审计。release 按 `published_at`、PR 按 `merged_at`、commit 按 committer date 筛选；所有时间在 CSV 中保留 UTC。

| Repo | 默认分支 commits | merged PR（含非默认） | releases | 结论 |
|---|---:|---:|---:|---|
| [NVIDIA-NeMo/RL](https://github.com/NVIDIA-NeMo/RL) | 10 | 11 | 0 | 窗口分页结束；main commit 与 PR 分开核对 |
| [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | 12 | 18 | 0 | 窗口分页结束；main commit 与 PR 分开核对 |
| [OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | 0 | 0 | 0 | 窗口分页结束；main commit 与 PR 分开核对 |
| [THUDM/slime](https://github.com/THUDM/slime) | 0 | 0 | 0 | 窗口分页结束；main commit 与 PR 分开核对 |
| [alibaba/ROLL](https://github.com/alibaba/ROLL) | 0 | 0 | 0 | 窗口分页结束；main commit 与 PR 分开核对 |
| [areal-project/AReaL](https://github.com/areal-project/AReaL) | 13 | 13 | 0 | 窗口分页结束；main commit 与 PR 分开核对 |
| [huggingface/accelerate](https://github.com/huggingface/accelerate) | 1 | 1 | 0 | 窗口分页结束；main commit 与 PR 分开核对 |
| [huggingface/kernels](https://github.com/huggingface/kernels) | 0 | 1 | 0 | 窗口分页结束；main commit 与 PR 分开核对 |
| [huggingface/peft](https://github.com/huggingface/peft) | 6 | 6 | 0 | 窗口分页结束；main commit 与 PR 分开核对 |
| [huggingface/tokenizers](https://github.com/huggingface/tokenizers) | 15 | 13 | 1 | 窗口分页结束；main commit 与 PR 分开核对 |
| [huggingface/transformers](https://github.com/huggingface/transformers) | 21 | 21 | 0 | 窗口分页结束；main commit 与 PR 分开核对 |
| [huggingface/trl](https://github.com/huggingface/trl) | 8 | 9 | 0 | 窗口分页结束；main commit 与 PR 分开核对 |
| [sgl-project/sglang](https://github.com/sgl-project/sglang) | 164 | 165 | 0 | 窗口分页结束；main commit 与 PR 分开核对 |
| [verl-project/verl](https://github.com/verl-project/verl) | 12 | 13 | 1 | 窗口分页结束；main commit 与 PR 分开核对 |
| [vllm-project/vllm](https://github.com/vllm-project/vllm) | 112 | 112 | 1 | 窗口分页结束；main commit 与 PR 分开核对 |

窗口内三项 release：verl v0.9.1（9/20）、vLLM v0.30.0（9/22）和 tokenizers v1.0.0-rc.2（9/21，prerelease）。HF 其他五库未见窗口内新 release。slime/ROLL/OpenRLHF 在本窗口没有 main 新提交，不代表无 open PR 或旧机制需要复核。

## 新接纳

### G1 · verl v0.9.1：把异步训练与权重同步修复落到版本上

- Source ID：`github-release:verl-project/verl:v0.9.1`；Type：official release；[原文](https://github.com/verl-project/verl/releases/tag/v0.9.1)；发布账号 `wuxibin89`；published `2026-09-20T07:24:43Z`；prerelease=`False`。

- Impact：高；Decision：Read；Reason：release 将分散的 V1 trainer、准入、refit 和 packing 行为汇合成升级检查面。
- 一句话看点：重点核对 V1 异步迁移、trainer GPU 借用、weight-sync gate 与 IPC/refit 生命周期，而不是只更新版本号。
- 边界：GPU lending 默认关闭；旧 fully_async/one_step_off_policy 入口是 deprecated，不等于已在本版删除。已接纳过的单个修复不算本轮新机制；release 数字均为上游报告。
- Subsystem / Dimension：scheduler / weight sync / training；兼容性与恢复正确性。Transfer to AReaL：借鉴升级验收矩阵与 pause/resume/ACK 顺序，不假设配置一一对应。
- Related topics：Agentic RL / refit。Next：作为现有 P1 的版本基线，检查当前实际依赖再决定升级；本任务没有升级任何运行环境。

### G2 · vLLM v0.30.0：检查 rollout backend 的可用版本与破坏性变化

- Source ID：`github-release:vllm-project/vllm:v0.30.0`；Type：official release；[原文](https://github.com/vllm-project/vllm/releases/tag/v0.30.0)；发布账号 `khluu`；published `2026-09-22T05:20:54Z`；prerelease=`False`。

- Impact：高；Decision：Read；Reason：涉及 PP 下 speculative decoding、低精度、KV 管理和服务入口兼容性，直接影响 RL rollout backend。
- 一句话看点：将版本能力与部署契约一起读，尤其 scale-out endpoints 的显式开启要求及废弃配置清理。
- 边界：release 汇总包含更早已合并机制；本轮 main 新 PR 不自动视为包含在这个 tag，使用前需要 tag ancestry 检查。未复现其性能数字。
- Subsystem / Dimension：inference backend / scheduler；兼容性与 rollout 稳定性。Transfer to AReaL：检查 backend pin、控制 API 与支持矩阵，避免单独追最新版本。
- Related topics：serving / PP / precision。Next：对照当前 AReaL/verl 依赖做能力表；verl main 此次所见 pin 为 vLLM 0.29.0，不能推断已支持 0.30.0。

### G3 · VLM context parallelism：模型拥有切分，metadata 必须跟到底

- Source ID：`github:areal-project/AReaL:c74b343c53c785eb620943fcabbd6b8263e28051`；[#1672 feat(engine): support VLM context parallelism](https://github.com/areal-project/AReaL/pull/1672)；作者账号 `sitabulaixizawaluduo`；merged `2026-09-21T12:57:42Z` → `main`。
- Source ID：`github:verl-project/verl:8e03c039f9a70750490c93196b759b332fc029fd`；[#7948 [megatron] fix: preserve bucketed VLM THD lengths](https://github.com/verl-project/verl/pull/7948)；作者账号 `qinyi810`；merged `2026-09-22T02:10:27Z` → `main`。

- Type：merged code；Impact：高；Decision：Read；Reason：多模态 embedding 融合、THD repacking 与 CP 分区相互依赖，提前切分或丢失 padded length 都会破坏执行。
- 一句话看点：AReaL 将 Qwen VLM 的 THD/CP 切分留给模型，verl 则修复 bucketed 长度在重新打包时丢失的问题。
- 核验：AReaL 保留完整多模态输入直到 vision embedding 融合，修正 DP/CP token normalization；新 guarded path 需要 MCore≥0.18.2、Bridge≥0.5.1。Qwen3.5-VL 包括 CP=1；Qwen3-VL 旧 CP=1 路径例外。verl 将 padded metadata 传入两个 VLM forward 路径，并新增 repacking 回归测试。
- Subsystem / Dimension：training / data path；shape、梯度缩放、长序列显存。Transfer to AReaL：直接相关；借鉴跨层 physical length 不变量。Next：P1 做 CP1/CP2 loss-gradient 对照与 padding bucket 测试；未在本地执行上游测试。

### G4 · AReaL MTP-only：冻结主干训练 drafter 的完整运行约束

- Source ID：`github:areal-project/AReaL:aba2f207c742143afd41e058404ec37d6ecced83`；[#1719 feat(engine): support native MTP-only and packed Qwen SFT](https://github.com/areal-project/AReaL/pull/1719)；作者账号 `dingzhiqiang`；merged `2026-09-21T09:47:20Z` → `main`。

- Type：merged code；Impact：高；Decision：Read；Reason：覆盖无可训练参数的 PP stage、独立 MTP label channel、导出与恢复，而非只增加一个冻结开关。
- 一句话看点：支持原生 MTP-only 与 packed Qwen SFT，并明确冻结权重、PP 阶段和 runtime 版本的约束。
- 核验：DDP/optimizer 构造前冻结非 MTP 参数，shared embedding/output 保持冻结；不支持该路径下 LoRA、critic 或 FSDP wrappers。代码检查实际加载的 cuDNN≥9.19.0，以及 MCore≥0.18.2、Bridge≥0.5.1。
- 边界：PR 提供短程功能验证和离线 coding trajectory SFT 记录；不能据 loss 下降断言 speculative acceptance 或 coding-task 质量提高。上游报告不等于本地 VERIFIED。
- Subsystem / Dimension：training / checkpoint；冻结语义、packed layout、数值稳定性。Transfer to AReaL：直接相关。Next：与[现有 RL 状态 P1](../reading_queue/P1.md#rl-state-boundaries-reading)合并，先核导出后 frozen tensor 完全一致及 MTP loss mask，再考虑长程实验。

### G5 · SGLang DeepSeek-V4.1：KV 能装下，prefill 仍可能 OOM

- Source ID：`github:sgl-project/sglang:95521da18df4780e9c63f5e2ddd284ecd9ca9b1c`；[#40217 [DeepSeek-V4.1] Bound dense prefill indexer memory](https://github.com/sgl-project/sglang/pull/40217)；作者账号 `harmya`；merged `2026-09-20T21:43:20Z` → `main`。

- Type：merged code；Impact：高；Decision：Deep Dive；Reason：补齐模型报告 global KV 指标无法描述的临时工作区峰值。
- 一句话看点：按 query rows 分块计算 FP32 indexer scores，并跨层传 compact candidate block IDs，压低长上下文 prefill 的临时显存。
- 核验：原实现 scores 与分片/拼接 Boolean masks 可能同时存活，主要项约 `6*T*N` bytes；这些是每 rank 临时分配，不能简单按 TP 除掉。新代码给 scores 设 2 GiB budget，每行仍看完整 context；该 budget 不是整个 indexer 或进程峰值上限。
- 边界：候选 block IDs 无损表示所选 mask；不表示 DeepSeek 所有 replay 都精确。上游 B200 indexer microbenchmark 和 full-model cold-prefill 分开报告，后者每长度单次，不应泛化速度提升。
- Subsystem / Dimension：inference backend / kernel；临时显存与可运行 context。Transfer to AReaL：rollout 容量预算同时计入 prefill workspace，不能只用 bytes/token。
- Related topics：KV cache / long context / DeepSeek。Next：并入 [DeepSeek P1](../reading_queue/P1.md#deepseek-v41-report)，绘制权重、KV、indexer、graph 与 staging 的峰值组成。

### G6 · vLLM KV hints：先有协议，尚未有动作执行

- Source ID：`github:vllm-project/vllm:986e217158a9ed54eca8307c6bb9abac7c5b2278`；[#53423 [Feature] Add first-class KV hints request envelope for programmatic KV management](https://github.com/vllm-project/vllm/pull/53423)；作者账号 `karen-sy`；merged `2026-09-21T16:56:12Z` → `main`。

- Type：merged code；Impact：中高；Decision：Read；Reason：为 orchestrator 与缓存管理后端建立明确请求契约，具有系统边界意义。
- 一句话看点：KV hints 接通 Python/Rust、HTTP/gRPC 和 scheduler 的请求传递，但具体 hint 的定义与执行仍由后续后端负责。
- 核验：typed envelope/action 与 MessagePack wire compatibility；PR 明确只建立 envelope/plumbing，没有执行 individual actions。不能写成 vLLM 已支持通用缓存 pin/prefetch/evict 控制。
- Subsystem / Dimension：inference backend / data path；协议兼容性。Transfer to AReaL：可评估在 rollout 请求中携带缓存意图，但必须另查接收 backend 是否消费。
- Related topics：KV management / scheduler。Next：保持 P1 配套，追踪 consumer 实现和无效/未知 hint 的处理。

### G7 · NeMo Gym shards：环境路由要保留 replica 身份

- Source ID：`github:NVIDIA-NeMo/RL:350fab73de72c001c88fb581a34ffd98444f72ee`；[#3373 feat(nemo-gym): wire sharded stacks into training](https://github.com/NVIDIA-NeMo/RL/pull/3373)；作者账号 `ananthsub`；merged `2026-09-21T17:53:13Z` → `main`。
- Source ID：`github:NVIDIA-NeMo/RL:fd7112c374af23b3bcc392637fb737495f8a3f31`；[#3374 fix(nemo-gym): preserve replica identity during routing](https://github.com/NVIDIA-NeMo/RL/pull/3374)；作者账号 `ananthsub`；merged `2026-09-21T19:45:15Z` → `main`。

- Type：merged code；Impact：高；Decision：Read；Reason：environment scale-out 同时影响数据覆盖检查、group 重试、端口隔离和启动失败清理。
- 一句话看点：把 sharded Gym 接入训练，并让一个 prompt group 的处理、重试与观测保持明确的 replica 归属。
- 核验：entrypoints 保存 ShardSet，验证 dataset agent_ref 被 shard 覆盖；后续 setup 失败会取回并关闭已经启动的 shard set。副本共享端口配置时要求 STRICT_SPREAD；round-robin 是 dispatch/group 口径，不是 completion-aware load balance。
- Subsystem / Dimension：rollout / scheduler；identity、failure containment、可运维性。Transfer to AReaL：借鉴 environment ownership 和每 replica 的指标，不把多个副本视为无状态池。
- Related topics：Agentic RL / environment / recovery。Next：并入状态 P1，检查重试不串 replica、部分启动失败无 orphan process；与 DSec 生命周期阅读联动。

### G8 · Graph replay 的 dummy 请求和通信流也有真实副作用

- Source ID：`github:vllm-project/vllm:d2983f2f163249acbe3053c073cb9830290984a0`；[#56734 [Bugfix][Spec Decode] Stop dummy draft decode steps from writing KV through stale block-table rows](https://github.com/vllm-project/vllm/pull/56734)；作者账号 `ivanium`；merged `2026-09-21T01:06:25Z` → `main`。
- Source ID：`github:sgl-project/sglang:bc22e1de9e3734d264933eb3b14301768ac7eb92`；[#40658 [DSpark] Fix draft CUDA graph stream explosion](https://github.com/sgl-project/sglang/pull/40658)；作者账号 `kpham-sgl`；merged `2026-09-22T06:05:45Z` → `main`。

- Type：merged code；Impact：高；Decision：Read；Reason：改变 speculative rollout 的 KV 正确性及 graph 执行资源，而非表面代码清理。
- 一句话看点：idle DP rank 的 dummy draft 不能写入旧请求的 KV，capture 内广播也应使用已启用的正确 communicator。
- 核验：vLLM 以 `idx_mapping=-1` 在 kernel 内屏蔽 dummy/padding row，生成 PAD slot，防止 persistent block table 的旧映射被重放写入。SGLang CUDA broadcast 在 PyNccl enabled 时沿当前 capture stream，保留不可用时 fallback。
- 边界：vLLM PR 的 GPU probe 是作者证据；SGLang stream 数降低不等于吞吐加速，其报告明确没有建立端到端速度收益，且 dummy weights/simulated acceptance 不验证模型质量。
- Subsystem / Dimension：inference backend / scheduler；KV correctness、graph lifecycle。Transfer to AReaL：将 idle-rank dummy 与多轮 prefix-hit 纳入 rollout 验收。
- Related topics：CUDA Graph / speculative decoding / KV cache。Next：追加[实验计划](../experiments/rl_state_boundaries.md)，检查 dummy 不写真实 block、cache-hit 后 acceptance 不异常；尚未运行。

### G9 · PP HiCache prefetch tickets：让后续 stage 提前开始 I/O

- Source ID：`github:sgl-project/sglang:020703923dba90d531fe8ec472de720584b69f80`；[#36700 [PP + HiCache] Add PP Prefetch Tickets for eager cross-stage storage prefetch](https://github.com/sgl-project/sglang/pull/36700)；作者账号 `huangtingwei9988`；merged `2026-09-20T03:19:32Z` → `main`。

- Type：merged code；Impact：高；Decision：Read；Reason：命中缓存不保证 TTFT 更低，顺序传播的调度等待仍可能阻塞下游预取。
- 一句话看点：PP0 发送轻量 ticket，让多个 stage 在真实请求到达前启动存储预取，再按共同 ready prefix 准入。
- 核验：ticket 的创建、绑定、取消/释放和跨 stage readiness 是主要新增状态；不是简单提前发一个 GET。上游 benchmark 使用 Mooncake/RDMA，基线有用于启动的验证 allocator guard，不能称无条件通用加速。
- Subsystem / Dimension：scheduler / inference backend / storage；TTFT、IO overlap、状态释放。Transfer to AReaL：长程 agent 的命中 trace 应拆解 lookup、I/O、跨 stage 等待与 admission，避免只看 cache-hit ratio。
- Related topics：PP / HiCache / storage。Next：作为现有 rollout 成本实验配套，先注入 ticket 发出后取消及单 stage 延迟，再测真实 trace。

## Observed 与纠偏（8 组）

| ID | 来源 / Impact / Decision | 核验后的判断、相关主题与下一步 |
|---|---|---|
| O1 | [tokenizers rc.2](https://github.com/huggingface/tokenizers/releases/tag/v1.0.0-rc.2)，中 / Observe | 确认 9/21 发布 prerelease，补原 A4 证据，不再算新信号；CPU pipeline，待 Python 与真实语料测量 |
| O2 | [TRL #7017](https://github.com/huggingface/trl/pull/7017)，高 / Observe | adapter-only sync 已于 9/10 17:37:08 UTC 合入 main；compare 显示 merge 在 v1.13.0 之后 12 commits；v1.14.0 ref 返回 404，release 最新 v1.13.0。已定位“代码存在”与“稳定 release 归属”差异，博客 release 声明仍不可证实；用固定 SHA，而非未核实 tag |
| O3 | [Megatron #5532](https://github.com/NVIDIA/Megatron-LM/pull/5532)、[#7245](https://github.com/NVIDIA/Megatron-LM/pull/7245)、[#7435](https://github.com/NVIDIA/Megatron-LM/pull/7435)，中高 / Observe | Pre-GDR fusion 默认关闭且 E2E 是 proxy estimate；GDP speculative 是独立集成；MIMO non-colocated CP 的该 PR 没跑 GPU oracle。training/CP/kernel；保留三者不同证据等级，不按 merge 自动接受 |
| O4 | [NeMo #4217](https://github.com/NVIDIA-NeMo/RL/pull/4217)，高 / Observe | shard-to-shard TRT-LLM refit 仅合入 `opt/dev-rubin-pin`；PR 报告限制 rollout TEP 与 PP=1。weight sync / quantization；先追 main/tag 与支持矩阵，不宣称 NeMo main 已可用 |
| O5 | [SGLang #40256](https://github.com/sgl-project/sglang/pull/40256)、[#40278](https://github.com/sgl-project/sglang/pull/40278)、[#27265](https://github.com/sgl-project/sglang/pull/27265)，中高 / Observe | staging 应在最终 KV sizing 前预留；TMA transfer 的 host link 条件不能泛化；TensorCast 默认 allocator 与 scratch 模式复制成本不同。memory/storage；先测端到端与资源占用 |
| O6 | [SGLang #39461](https://github.com/sgl-project/sglang/pull/39461)、[#40391](https://github.com/sgl-project/sglang/pull/40391)、[#40313](https://github.com/sgl-project/sglang/pull/40313)，中 / Observe | disconnect cancellation 是 best-effort 且跳过 fan-out/PD；stream idle timeout 独立于 engine abort；删除旧 SWA/Mamba cache 类是已迁移 UnifiedRadixCache 的清理，不是删除支持。scheduler/cache；按支持路径做集成测试 |
| O7 | [Transformers #47809](https://github.com/huggingface/transformers/pull/47809)、[#48421](https://github.com/huggingface/transformers/pull/48421)，中高 / Observe | cache allocator/sector/pool 分工支持混合 attention；Switch router 修复 raw logits、capacity 与 loss，不外推到全部 MoE。cache/MoE；待实际使用该 backend 时提升 |
| O8 | [verl #7927](https://github.com/verl-project/verl/pull/7927)、[AReaL #1738](https://github.com/areal-project/AReaL/pull/1738)、[NeMo #3706](https://github.com/NVIDIA-NeMo/RL/pull/3706)，中高 / Observe | teacher lanes 是 tracing/Prometheus 归属，不是新调度隔离；AWEX 在 auxiliary scoring 前 offload 的 tests 是 mock ordering；draft split-step 按 DP-global draft tokens 归一化。observability/memory/loss；作为已有状态主线配套 |

## 历史待复核项回看

下面是 9/18、9/20 扫描遗留候选的补证据，不伪装成本窗口新提交，不改变历史 Accepted 数量或原游标。Decision 均保持 Observe；完成的是来源与机制初步核验，未声称生产验证或每行代码都已读。

| 来源 | 本次明确的工程后果 / 下一步 |
|---|---|
| [AReaL #1697](https://github.com/areal-project/AReaL/commit/518d2ff983983e22cbf4ffa18debaa57af021f89)、[#1723](https://github.com/areal-project/AReaL/pull/1723)、[#1724](https://github.com/areal-project/AReaL/pull/1724) | VLM CPU staging 保留 alias、AWEX scheduler API 适配、attributed failure 零 reward 都有特定路径；分别验证复制成本、版本边界和失败归因 |
| [NeMo #4105](https://github.com/NVIDIA-NeMo/RL/commit/946b1970f50ff5714a2ec9dfacbdcc1017cf27de)、[#3616](https://github.com/NVIDIA-NeMo/RL/commit/fb121a00d6b49cffd59e9e13e6c92215008b948e) | Energon packs 的 num_valid_samples 计物理 pack，NLL 仍按 valid tokens；dynamic batching/HybridEP flex 受限。Data-plane timing/bytes 是计时边界证据，不是吞吐收益证明 |
| [NeMo #3923](https://github.com/NVIDIA-NeMo/RL/pull/3923)、[#3924](https://github.com/NVIDIA-NeMo/RL/pull/3924)、[#3898](https://github.com/NVIDIA-NeMo/RL/pull/3898)、[#3935](https://github.com/NVIDIA-NeMo/RL/pull/3935) | sibling ledger 补 TQ 不具备的 lineage；周期 rollout snapshot 锚定 durable trainer state；Mooncake 先恢复 storage 后 queue metadata；Automodel 写完才能 mutate/promote。下一步是组合故障注入，单个 save 成功不证明端到端恢复 |
| [NeMo #3739](https://github.com/NVIDIA-NeMo/RL/pull/3739)、[#4106](https://github.com/NVIDIA-NeMo/RL/pull/4106)、[#4107](https://github.com/NVIDIA-NeMo/RL/pull/4107) | Megatron nccl_reshard 是 opt-in 分片传输；PackedTensor 将 preprocessing 随切片/传输保存；local fetch 共享 storage 不等于跨进程 wire 一律 zero-copy |
| [OpenRLHF #1309](https://github.com/OpenRLHF/OpenRLHF/pull/1309)、[#1321](https://github.com/OpenRLHF/OpenRLHF/pull/1321)、[#1327](https://github.com/OpenRLHF/OpenRLHF/pull/1327) | KL unbiased gradient 默认关闭；terminal buffer 清空后才能重启 loader；多轮 length 截断须传到 penalty。下一步检查 estimator 和 sampling/recovery 组合，不据旧标题认定所有默认行为已改变 |
| [slime #2390](https://github.com/THUDM/slime/pull/2390)、[#2391](https://github.com/THUDM/slime/pull/2391)、[#2394](https://github.com/THUDM/slime/pull/2394) | internal sync 带入 async filtering/staleness/replay；删除 legacy entrypoint 不等于删除所有异步训练；不 offload train 时避免 reloadable process group。迁移需以保留入口为准 |
| [Megatron #6597](https://github.com/NVIDIA/Megatron-LM/commit/2c897b8bf104be4f5795737dbeff9bd54741f99a)、[#6773](https://github.com/NVIDIA/Megatron-LM/commit/29694c26ec7efdeb5b9cd3aed4501ddb23be8f9b)、[#7265](https://github.com/NVIDIA/Megatron-LM/pull/7265)、[#7302](https://github.com/NVIDIA/Megatron-LM/pull/7302)、[#6885](https://github.com/NVIDIA/Megatron-LM/pull/6885) | owner plan 是布局/packing API；Hybrid FSDP 要区分 weight replication 与 gradient sharding；QuantizedDBuffer 本身不是完整 MFSDP 集成；batch invariance 有 backend 限制；one-shot stream 问题影响 reserved memory，不能写成 live tensors 同幅减少 |
| [vLLM #44890](https://github.com/vllm-project/vllm/commit/092bdd6d57ac7c1cd5272339c80372053fd51bbe)、[SGLang #28403](https://github.com/sgl-project/sglang/commit/1f60ddef5dc2ae3bbfbe0c5cea45690c4b60a251) | selective KV discard 需 completed pause、memory resident 和支持 backend，kept requests 醒来重算；runtime PD role switching 的端到端 quiesce/资源回收仍需集成验证 |
| [SGLang #40024](https://github.com/sgl-project/sglang/pull/40024)、[#40034](https://github.com/sgl-project/sglang/pull/40034)、[#39464](https://github.com/sgl-project/sglang/pull/39464)、[#34012](https://github.com/sgl-project/sglang/pull/34012) | shortest-prefill-first 按未缓存剩余 tokens；agentic simulator 提供带 tool delay 的合成 trace；503/429 不自动当 breaker fault；T-LRU opt-in。下一步联合测公平性、tail latency 和 backpressure，不把合成负载当真实训练 |

## Open PR 与分支边界

Open snapshot 只做标题/状态检索，不占 Accepted，完整列表见 CSV。近期应继续看 [AReaL #1721](https://github.com/areal-project/AReaL/pull/1721) partial groups、[#1722](https://github.com/areal-project/AReaL/pull/1722) sample refill、[OpenRLHF #1323](https://github.com/OpenRLHF/OpenRLHF/pull/1323) oversampling checkpoint、[Megatron #7114](https://github.com/NVIDIA/Megatron-LM/pull/7114) MXFP8 MFSDP integration、[NeMo #4129](https://github.com/NVIDIA-NeMo/RL/pull/4129) MInf ledger checkpoint 和 [TRL #6623](https://github.com/huggingface/trl/pull/6623) FSDP2 ignored-parameter sync。它们是待核验实现，不是已发布能力。

12 个非默认分支 merged PR 的 base 全部保留在账本：尤其 NeMo `opt/dev-rubin-pin` 和 Megatron `dev` 不能用 merged 标签代替 main 可用性；其余 CI/功能分支合并也不自动成为用户功能信号。

## 落点与限制

- G5 接入 DeepSeek P1；G3/G4/G7/G8/G9 接入现有状态/rollout 主线；G1/G2 是版本验收参考，保持当前 P0 和用户阅读状态。
- 更新[Agentic RL 状态章节](../topics/agentic_rl.md#rl-state-boundaries)与[实验计划](../experiments/rl_state_boundaries.md)。本次没有运行上游测试、GPU workload 或自动升级依赖。
- GitHub **已声明窗口的索引缺口关闭**；源码穷尽审计、open PR 全文阅读和测试复现并不在这一结论内。后续 GitHub 从 17:06:16 扫描；非 GitHub 从原 16:52:05 继续。NVIDIA 博客正文仍是另一来源的待办。

导航：[原 9/22 全源报告](frontier_scan_2026-09-22.md) · [Scan Log](scan_log.md) · [Tracking](README.md) · [P1](../reading_queue/P1.md)。

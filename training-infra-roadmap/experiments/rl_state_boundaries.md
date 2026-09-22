# RL 状态边界：故障注入验证计划

## Question

部分 rollout、async checkpoint 和 generation worker 重启后，系统是否只消费完整、版本正确且有明确定义的状态？来源见 [2026-09-16 scan](../tracking/frontier_scan_2026-09-16.md)，阅读顺序见 [P1](../reading_queue/P1.md#rl-state-boundaries-reading)。

## Environment

- **计划，尚未执行**；未绑定用户集群，不申请 GPU 或启动付费任务。
- AReaL 基线需包含 `64f049f1` 与 `e23a9353`；NeMo RL 基线需包含 `53bce056`。正式运行时记录完整 SHA、依赖 lock、backend、GPU/NCCL/Ray 版本。
- 第一阶段 CPU 或 mock contract tests；第二阶段在独立测试环境做最小多 worker 集成，不在生产训练注入故障。

## Setup

| 实验 | 干预 | 必须保持的不变量 | 观测与失败判据 |
|---|---|---|---|
| Partial group | 原计划四个 rollout，分别留下 0/1/2/4 个可用成员；将一条轨迹拆成多 row，并改变 padding | logical member 数不随 row 拆分增加；无效 token 不进入 loss；group-stat estimator 遵守最小样本规则 | 记录 raw/usable/logical counts、mask sum、advantage；出现 NaN、padding 改变有效 token 的 loss/grad 或统计重复计数则失败 |
| Ragged transport | 某 rank 零轨迹，其他 rank 不等长；本地 packing 抛异常 | 所有参与 rank 按相同 collective 顺序退出或完成；不能仅一侧进入新的 collective | 记录 collective 序号、超时和 rank 错误；无界等待即失败 |
| Checkpoint publication | actor payload 完成后令 critic 保存失败；另一次在 finalization 前中断 publisher | LATEST 仍指上一个完整 generation；未提交的新 generation 不可被当作 latest 恢复 | 对比 pointer、manifest 和 actor/critic global step；混合 step 或指向残缺 payload 即失败 |
| Refit admission | 杀死 generation shard；让 replacement 在 refit 启动前或中途完成重启 | 中途回来的 shard 保持 stale；只有参与成功 refit 的 incarnation/version 可以接流量 | 记录 shard ID、incarnation、weight version、refit participant set、routing decision；旧版本获准 serving 即失败 |

9/18 新增，来源见[本轮 A4/A5](../tracking/frontier_scan_2026-09-18.md)：

| 实验 | 干预 | 必须保持的不变量 | 观测与失败判据 |
|---|---|---|---|
| Resume admission | 模拟一个 DP rank 的 resume 延迟，同时排队下一波请求；覆盖有/无 KV restore 两条路径 | 全 replica engine resume 返回前，所有 submission gate 保持关闭 | 记录每 rank 的 resume collective 和 wave 序号；提前 admission 或无界等待即失败 |
| Reward transaction | 在默认 finish timeout 下提交含多个 interaction 的 reward map，再测试非法 ID | 合法 map 全部应用后才 finalize，最多一次；非法输入处理符合上游契约 | 对照每步 reward、HTTP 状态和 finalize count；只保留末步或丢弃合法 group 即失败 |

基线分别固定 verl `e2ac8f6222801d5e8ce50447b0c3c2d9237e4770`、AReaL `179ff1bf80796ec8797cea3dc2dcf8f46beef6cb` 或包含它们的后续版本。仍未执行，不编造 GPU 结果。

9/20 补充计划，来源见[本轮 A4/A5](../tracking/frontier_scan_2026-09-20.md)：在“短暂同步”和“退出服务轮转”两种 gate mode 下分别注入迟到请求，验证 parking 后可恢复、rejection 可及时返回；并发提交两个总容量超过上限的 `n>1` 请求，确认 all-or-nothing reservation，再在首个 child 提交后取消，检查所有 slot 释放。容量原子性与最终样本完整性分开验收。本仓库尚未执行这些测试。

这些是根据 upstream 行为设计的验收用例，不是对所有 estimator、storage 和 backend 通用的实现规范。最小组大小尤其需要按配置检查。

## Commands

进入事先安装好依赖的独立 AReaL checkout 后，可从固定 patch 中已核实存在的测试文件起步：

```bash
git rev-parse HEAD
git status --short
python -m pytest tests/test_incomplete_rollout_groups.py -q
```

运行前核对该 checkout 包含上述修复并阅读测试的环境要求。其余 checkpoint / NeMo 测试入口从固定 patch 的文件树选择，记录实际命令后再执行；这里不编造尚未验证的 test selector 或集群启动参数。

## Results

未执行。暂无 pass/fail、吞吐、恢复时长或 GPU 利用率数据。阅读 upstream tests 不等于在本地跑通过。

## Analysis / Decision Impact

只有 contract 与集成测试都满足预期，才将结论提升为 VERIFIED，并据实际 backend 限制修改排障手册。若只在 mock 下通过，只能说明局部状态转换；不能证明跨进程通信、存储持久性和模型数值一致性。

## Follow-up

- 将实际结果回填本页和 [9 月 Learning Log](../learning_log/2026/2026-09.md)。
- 关联：[Agentic RL](../topics/agentic_rl.md#rl-state-boundaries)、[Checkpointing](../topics/checkpointing.md)、[Knowledge Graph](../KNOWLEDGE_GRAPH.md)。

## 2026-09-22 补充计划：AWEX 与冻结权重

来源：[9/22 A5/A7](../tracking/frontier_scan_2026-09-22.md)。以下均未执行，不新增未经验证的测试命令。

- AWEX：在一个 TP rank 注入 event-loop 延迟，同时排队推理请求和权重更新。固定修复前后版本，对比 collective 序列，确认显式 pause 后才更新、无死锁，并检查恢复后的 policy version。单进程 mock 不能代替多 rank 验证。
- Frozen weights：用明确的 named_parameters glob 选冻结集合，记录 sleep 前后的值校验与恢复 storage；trainer 只更新其他参数。检查 pageable CPU 备份峰值、wake 延迟，以及 loader 不替换冻结 storage。额外注入 trainer 错误更新冻结参数，确认外层契约能检测，而不是期待 vLLM 自动过滤。
- DSec 思路迁移：分别模拟 GPU job 终止、agent loop 终止、sandbox 终止，记录各自能否恢复；不把磁盘 checkpoint 当成进程执行态 checkpoint。尚未部署 DSec，不声称复现其平台。

## GitHub 补扫后的新增验收计划

来源：[2026-09-22 GitHub G3/G5/G8/G9](../tracking/github_audit_2026-09-22.md)。Status：NEW；以下没有执行结果。

| 场景 | 控制变量 / 干预 | 验收指标 |
|---|---|---|
| DeepSeek cold prefill 峰值 | 固定权重、context、query chunk、cache hit=0，分别记录原实现与 row tiling；不要在每次调用前清空 allocator 来改变比较条件 | allocated/reserved 峰值、indexer scope 与全模型峰值分开；score budget 之外记录 mask/top-k scratch；同时比较输出选择与 TTFT |
| VLM THD/CP | 同一 batch 在 CP1/CP2、bucket padding 开/关下核验 physical length、label/mask 和 token count | fused vision 后的布局一致、loss/gradient 在指定容差内；记录实际 runtime 版本 |
| Dummy draft KV | 在 DP 某 rank idle 时触发多步 draft，再用旧 prefix 发真实请求；比较修复前后固定 SHA | dummy/padding mapping 全 PAD，不写原真实 block；记录 acceptance、输出正确性和 KV sentinel，不能只看吞吐 |
| PP prefetch ticket | 发 ticket 后取消、单 stage 延迟、storage miss 与部分 ready | KV/sidecar 共同安全前缀决定 admission；取消后无泄漏；TTFT 拆分 I/O 与跨 stage 等待 |

先准备能运行的独立 backend checkout 再记录实际命令；不把本仓库文档检查当成上述实验通过。

<a id="retrospective-test-cases"></a>

## 2026-09-22 月度复盘补充：最小验证场景

状态：**设计候选，未执行**。来源为[14 组历史补漏](../tracking/github_retrospective_2026-07_to_2026-09.md)与[7–8 月重评](../tracking/monthly_reviews.md)。不新建一批必做任务，按正在使用的 backend 选择一项。

| 场景 | 对照与注入 | 必须记录 | 通过条件 / 会推翻什么判断 |
|---|---|---|---|
| 异步队列守恒 | 相同 task IDs，完成速度高于消费速度；分批取走后重启 | submitted / inflight / completed / trained / discarded IDs、discard reason、heartbeat | 无无法解释的漏项；不重复训练；重采样有显式语义；若只降并发就好转，继续查背压而非直接归因 GPU |
| checkpoint cut | 混合长短轨迹，在 cursor 前移但训练未消费时保存并恢复 | 数据 cursor、trained frontier、buffer coverage、样本长度分布 | 未消费窗口可恢复或重生成；已消费项不重复；验证对长样本是否有选择性丢失 |
| sparse refit | 相同 dense baseline，分别 dense/sparse 更新；中途让一个 worker 失败 | wire bytes、staging 峰值、apply/rebuild 时间、版本、admission | 总时长和正确性均满足目标才算收益；失败 worker 禁入并恢复，不能只报告 wire 变小 |
| CP 语义 | 固定权重与输入，CP=1 对照 CP>1；包含 sliding mask、packed 文档和无 response 分片 | 输出、梯度、归一化统计、collective 参与序列 | 输出/梯度在预设容差内一致或明确拒绝不支持配置；loss 可下降不是通过条件 |
| lazy environment | eager/lazy 对照，同一镜像；增加缓存占用并注入 snapshotter 重启 | Ready、首请求、全文件读取、后续错误、节点缓存水位 | 不把 Ready 当可用证明；失败可检测、可隔离、可恢复；不将 KServe 结果直接外推 DSec |
| kernel 合同 | 一个已有算子，对照可信高精度参考；异常值、边界 shape、重复执行和 backward | 规格、误差、NaN/Inf 语义、gradient、计时条件 | 先验证适用规格再计性能；不能为追求一致性随意修改数学语义或容差 |

代码与版本选择前先读 [Agentic RL 不变量](../topics/agentic_rl.md#monthly-retrospective-invariants)。上游 PR 的测试通过声明是来源证据，不是本实验的结果。

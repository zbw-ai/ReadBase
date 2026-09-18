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

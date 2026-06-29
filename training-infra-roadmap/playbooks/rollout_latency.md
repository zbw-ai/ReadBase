# Rollout Latency

## 现象

Agentic RL / reasoning RL 中 rollout 生成变慢，policy update 等待样本，GPU 利用率下降。

## 影响范围

RL 训练吞吐、样本新鲜度、训练成本、评估周期。

## 第一时间处理

先不要只看训练 GPU 利用率。把端到端 pipeline 拆成 rollout、reward/verifier、trajectory store、policy update、weight sync 五段，确认到底是哪一段让 trainer 拿不到可训练样本。

## 排查顺序

1. 检查 trainer 是否在等样本：policy idle time、update interval、effective batch fill time。
2. 检查 rollout queue：pending requests、finished trajectories、dropped/timeout trajectories。
3. 检查 rollout latency 分布：p50/p95/p99、平均输出长度、最长输出长度。
4. 检查生成吞吐：tokens/s、requests/s、KV cache usage、prefix cache hit rate。
5. 检查 reward/verifier：queue depth、p95/p99 latency、timeout rate、error rate。
6. 检查 trajectory store：写入延迟、读取延迟、序列化开销、对象大小。
7. 检查 weight sync：policy version lag、inference worker 更新耗时、失败重试。
8. 检查调度策略：rollout/training GPU 配比、是否被长尾任务拖住、是否有 staleness bound。

## 定位命令

不同系统命令不一样，但排障时至少要拿到这些视图：

```bash
# 1. 看各角色 GPU 是否真的在忙
nvidia-smi dmon

# 2. 看 rollout / reward / trainer 关键日志
grep -E "rollout|reward|verifier|policy|weight|queue|staleness|timeout|token" train.log

# 3. 看 NCCL 或分布式训练是否顺带异常
grep -E "NCCL|timeout|hang|rank|all_reduce|broadcast" train.log

# 4. 看对象存储或 trajectory store 是否有慢写
grep -E "trajectory|store|upload|download|serialize|deserialize" train.log
```

如果系统接了 Prometheus/Grafana，优先看 dashboard，不要从散落日志里猜。

## 日志关键字

- `rollout`
- `queue`
- `reward`
- `verifier`
- `latency`
- `staleness`
- `policy_version`
- `weight_sync`
- `trajectory`
- `timeout`
- `kv_cache`
- `prefix_cache`
- `token_per_second`
- `idle`

## 可能根因

- 长上下文导致 KV cache 压力。
- verifier 成为串行瓶颈。
- scheduler 没有平衡 freshness 和 throughput。
- batch 内 response length 差异太大，同步 RL 被最长样本拖住。
- tool call / browser / environment 服务 p99 延迟过高。
- reward model 或 judge model 资源不足。
- trajectory 序列化太重，写入存储慢。
- rollout worker 使用旧 policy 太久，样本被 trainer 丢弃。
- weight sync 阻塞 rollout worker，导致生成停顿。
- vLLM engine 和 training actor 的 placement group / GPU fraction 配置不合理，导致资源碎片或互相抢占。
- tokenizer / chat template 不一致，导致 rollout 结果被训练侧过滤。
- 训练 GPU 和 rollout GPU 配比不合理。

## 修复方案

短期止血：

1. 降低单个 rollout 的最大生成长度，先确认 tail latency 是否下降。
2. 给 tool/environment 设置 timeout 和 retry budget，避免无限等待。
3. 单独扩 verifier/reward worker，确认 trainer idle 是否下降。
4. 暂时增大 rollout queue buffer，但同步设置最大 staleness，避免旧样本污染。
5. 降低每轮 update 所需 rollout 数量，减少 batch fill time。
6. 对长任务做分桶，避免短任务被长任务拖住。

中期优化：

1. 引入半异步或全异步 rollout/training 解耦。
2. 将 rollout worker pool、reward worker pool、training worker pool 分离调度。
3. 对 trajectory store 做分片写入，避免单点 metadata 或对象存储瓶颈。
4. 建立 policy version 和 reward version 追踪。
5. 为 inference worker 做增量或低停顿 weight sync，参考 vLLM + OpenRLHF 中 CUDA IPC / NCCL 的同步路径。
6. 对高频 prompt / system prefix 使用 prefix cache。

长期设计：

1. 采用 AReaL 类 producer-consumer 架构，通过 staleness bound 控制异步样本。
2. 采用 verl / HybridFlow 类 dataflow 抽象，把 RL pipeline 各阶段变成可编排节点。
3. 把 rollout、reward、training、checkpoint、eval 纳入统一 observability。

## 如何验证恢复

至少验证以下指标：

- rollout token/s 恢复；
- rollout p95/p99 latency 下降；
- reward/verifier queue depth 不再持续增长；
- policy idle time 下降；
- end-to-end update interval 缩短；
- sample staleness 没有超出阈值；
- eval reward / pass rate 没有因为使用旧样本而恶化。

## 如何避免再次发生

为 rollout / verifier / update 建立分段 SLO：

- rollout p95 latency；
- reward p95 latency；
- trainer max idle time；
- max policy version lag；
- max trajectory queue age；
- weight sync max duration；
- environment timeout rate。

每次新任务上线前，用小规模 dry run 测 response length 分布和 verifier latency，不要等大规模训练时才发现 tail latency。

## 关联 Topics

- [Distributed Training](../topics/distributed_training.md)
- [Context Parallelism](../topics/context_parallelism.md)
- [Agentic RL](../topics/agentic_rl.md)
- [Checkpointing](../topics/checkpointing.md)

## 关联 Papers / Reports / Blogs

- [DeepSeek-R1](../tech_reports/deepseek_r1.md)
- [AReaL](https://arxiv.org/abs/2505.24298)
- [HybridFlow / verl](https://arxiv.org/abs/2409.19256)
- [Agent Lightning](https://arxiv.org/abs/2508.03680)
- [Historical Backfill](../tracking/historical_backfill.md)
- [vLLM + OpenRLHF Integration](https://vllm.ai/blog/2025-04-23-openrlhf-vllm)
- [OpenRLHF](https://arxiv.org/abs/2405.11143)
- [NVIDIA NeMo RL](https://docs.nvidia.com/nemo/rl/latest/index.html)

## 关联 Experiments

- 建议新增：`experiments/agentic_rl/rollout_latency.md`

## 复盘问题

- rollout 系统是否应该和训练系统解耦调度？
- 这次问题是 compute-bound、memory-bound、network-bound、storage-bound 还是 external-service-bound？
- 是否记录了 policy version、reward version、tokenizer version 和 environment version？
- 如果同样任务扩大 10 倍 rollout worker，瓶颈会转移到哪里？
- 这次修复是否降低了 sample freshness 或训练稳定性？

# Interview: Checkpoint

相关手册章节：[Checkpointing](../topics/checkpointing.md)

## 高频面试题

1. 大模型 checkpoint 需要保存哪些状态？
2. sharded checkpoint 和 full checkpoint 有什么区别？
3. distributed checkpoint 为什么重要？
4. checkpoint frequency 如何选择？
5. 恢复训练后如何验证正确性？

## 追问问题

- world size 变化后如何 load checkpoint？
- optimizer state 丢失会有什么影响？
- 异步 checkpoint 如何避免不一致？

## 生产环境案例

10K GPU 作业每 30 分钟保存一次 checkpoint，但保存耗时 8 分钟，严重影响有效训练时间。需要引入 async checkpoint、分层存储、限流、增量或分片保存，并评估故障恢复损失。

## 常见错误回答

- “checkpoint 就是保存 model weights。”错误，大规模训练必须保存 optimizer、scheduler、RNG、data cursor、parallel metadata。
- “保存越频繁越安全。”不一定，保存本身会消耗 IO 和训练时间。

## 优秀回答示例

训练 checkpoint 是可恢复训练状态的快照，不只是权重。生产系统需要 sharded/async/distributed checkpoint，支持并行度变化恢复，并通过 loss continuity、optimizer state、RNG 和数据位置验证恢复正确性。

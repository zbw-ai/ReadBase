# Interview: FSDP

## 高频面试题

1. FSDP 和 ZeRO-3 的关系是什么？
2. FSDP forward 前为什么需要 all-gather 参数？
3. backward 后为什么 reduce-scatter？
4. auto wrap policy 如何影响性能？
5. FSDP checkpoint 与普通 checkpoint 有何不同？

## 追问问题

- FSDP OOM 时你先调 wrap 粒度、micro-batch 还是 recompute？
- FSDP all-gather 抖动如何定位？
- full state dict 和 sharded state dict 如何选择？

## 生产环境案例

训练 30B 模型时，普通 DDP 因 Adam state OOM。切到 FSDP 后显存下降，但 step time 增加。应检查 wrap 粒度、bucket size、prefetch、overlap 和 checkpoint 保存策略。

## 常见错误回答

- “FSDP 只是省显存版本 DDP。”不完整，它改变了参数 materialization 生命周期。
- “FSDP 总比 DDP 好。”错误，模型较小或网络较弱时 FSDP 额外通信可能不划算。

## 优秀回答示例

FSDP 把参数、梯度和 optimizer state 分片到 data parallel ranks 上，需要时 all-gather，用完释放，并在 backward 中 reduce-scatter 梯度。它适合显存成为瓶颈的训练，但生产中要调 wrap、prefetch、bucket、checkpoint，避免省显存换来过多通信。

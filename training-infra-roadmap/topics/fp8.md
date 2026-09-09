# FP8

## 核心问题

在 H100/H800 等硬件上用 FP8 降低矩阵乘的存储和带宽压力，提高吞吐，但必须控制数值稳定性。

## 相关材料

- [DeepSeek-V3](../tech_reports/deepseek_v3.md)
- [Transformer Engine](transformer_engine.md)
- [FlashAttention-3](../papers/flashattention3.md)

## 关键机制

- E4M3 / E5M2
- scaling factor
- amax history
- delayed scaling
- FP8 GEMM accumulation

## 生产关注

- FP8 不是全模型无脑替换；哪些层用、哪些状态保留 BF16/FP32 很关键。
- 需要监控 scale/amax 异常、loss spike 和溢出。
- 不同硬件和 kernel 的支持矩阵要严格验证。

<a id="precision-validation"></a>
## 从机制到验证：不止改变 dtype

低精度评估先拆四层：**格式可表示范围 → scale/量化误差 → 实际 GEMM 与数据搬运 → 端到端训练质量**。BF16 适合作为对照，但也不是无限精确的真值；FP8/FP4 则不能只凭 loss 不出现 NaN 验收。

配置前先明确 scale 方向：若 `q=Q(x/s)`、`x_hat=s*q`，这里 s 是反量化 scale；API 可能存它的倒数。Current scaling 根据当前 amax 选择范围，Delayed 用历史值减少当前数据读取，但分布突变可能导致饱和。[Current](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/fp8_current_scaling/fp8_current_scaling.html) / [Delayed](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/fp8_delayed_scaling/fp8_delayed_scaling.html)

工程排障顺序：

1. 固定 workload 和高精度对照，找第一次偏离的层/步骤；分别观察 fprop、dgrad、wgrad 的误差、饱和和零值比例。
2. 用按层/按 GEMM 回退缩小范围；恢复才异常时检查 scale/history，分布式才异常时检查 scale 对应的 tensor 与通信组。
3. 用 profiler 拆量化、layout、GEMM、通信与其他算子；小 GEMM 可能无法摊薄量化和 launch 成本。
4. 通过短窗口数值检查后，再看长期收敛与任务质量。对于 3D/视频，增加几何/拓扑或时序回归是工程建议，不是“某一格式已证明适用于全部模型”的结论。

教学手算、GPU 硬件边界与练习集中见 [Meshy 数值与低精度题组](../../private_resume/2026-09-meshy-ml-system-interview-prep.md#meshy-a1)；主文档仅保留[面试速答](../../private_resume/2026-08-llm-infra-interview-prep.md#meshy-interview-sprint)。这些验证方案不代表已完成本人 FP8/FP4 生产落地。

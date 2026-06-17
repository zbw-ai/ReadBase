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

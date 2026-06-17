# Transformer Engine

## 核心问题

NVIDIA Transformer Engine 提供面向 Transformer 训练的 FP8/mixed precision building blocks，让模型以更高吞吐运行，同时管理 scaling、amax 和数值稳定性。

## 相关材料

- [FP8](fp8.md)
- [FlashAttention-3](../papers/flashattention3.md)
- [DeepSeek-V3](../tech_reports/deepseek_v3.md)

## 关键机制

- FP8 Linear
- LayerNorm/RMSNorm 融合
- attention backend selection
- amax/scaling metadata

## 生产关注

- 版本、CUDA、driver、GPU 架构强相关。
- 模型 checkpoint 需要保存或兼容 precision recipe metadata。
- 性能回退时要检查 kernel path，而不是只看 dtype。

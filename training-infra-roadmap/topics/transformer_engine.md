# Transformer Engine 与 NVIDIA 融合算子

> 面试速答入口：[KERNEL-01｜NVIDIA 卡上为什么还需要融合算子？](../../private_resume/2026-08-llm-infra-interview-prep.md#kernel-01)

## 问题 framing

GPU 性能不只取决于 Tensor Core 峰值。Transformer block 中有两类完全不同的瓶颈：

- 大 GEMM 往往更接近 compute-bound，核心是矩阵 shape、precision、Tensor Core 与并行切分；
- norm、bias、activation、dropout、mask、softmax、cast、optimizer update 等短链路更容易 memory-bound 或 launch-bound。

融合算子的价值通常不是减少模型定义中的 FLOPs，而是：

1. 中间结果留在 register/shared memory，少写回和重读 HBM；
2. 减少 kernel launch、Python/runtime/driver 提交与 stream 同步；
3. 缩短临时 tensor 生命周期，降低峰值显存；
4. 让 compiler/library 针对完整 pattern 选择更合适的 tile、epilogue 与流水。

所以正确问题不是“要不要 fusion”，而是“这一段究竟受 FLOPs、HBM、launch 还是通信限制，融合是否命中真实 kernel”。

<a id="fusion-map"></a>
## 常见融合算子地图

| 类别 | 典型 pattern | 主要收益 | 常见实现/入口 |
|---|---|---|---|
| Attention | QKᵀ → scale/mask → softmax/dropout → PV | IO-aware tiling，避免显式 attention matrix，减少 HBM 往返 | FlashAttention、Transformer Engine attention backend |
| Projection/position | fused QKV、QKV + RoPE、single-QKV RoPE | 减少 split/concat、position kernel 和临时 tensor | Megatron-Core/TE layer spec、`fused_single_qkv_rope`、`apply_rope_fusion` |
| Activation/epilogue | bias + GeLU/GEGLU/SwiGLU、weighted activation | GEMM epilogue 与激活少一次读写/launch | TE、Megatron-Core fused bias activation、Triton |
| Residual/norm | bias + dropout + residual、residual + RMSNorm、fused LayerNorm | memory-bound 链路合并，减少 activation 常驻 | fused bias-dropout-add、TE fused RMSNorm |
| Softmax/loss | scaled masked softmax、vocab-parallel cross entropy | 避免 full-vocab/full-attention materialization | Megatron-Core fused softmax、vocab-parallel CE |
| Backward | gradient accumulation fusion、dgrad/wgrad overlap | 将权重梯度累加并入 kernel，减少额外读写 | Megatron tensor-parallel linear + Apex extension |
| Optimizer | multi-tensor Adam/SGD、fused unscale/clip/update | 一次处理多个参数 tensor，降低 launch 与遍历开销 | Apex/TE/PyTorch fused optimizer |
| MoE | token permute/unpermute、router/top-k、Grouped GEMM、shared expert overlap | 合并大量小 expert 工作，减少 token 搬运与 launch | Megatron-Core MoE、Transformer Engine GroupedLinear |

### FlashAttention 为什么不只是普通 fusion

普通 elementwise fusion 主要把相邻小算子并在一个 kernel 里。FlashAttention 进一步改变 Attention 的 IO 算法：分块读 Q/K/V，在线维护 softmax 统计量，不把完整 `S×S` attention matrix 写入 HBM。它既是 fusion，也是 IO-aware algorithm；长上下文收益通常比简单 bias/activation fusion 更结构化。

### Grouped GEMM 为什么适合 MoE

MoE 每个 expert 接收的 token 数不同，单 expert GEMM 可能太小。如果逐 expert launch，会出现大量小 GEMM、launch overhead 和低 Tensor Core 利用率。Grouped GEMM 把多个不同 `M`、共享或相近 `N/K` 的 expert matmul 编排为一组执行，提高设备占用并减少 launch。它不能消除 router imbalance；若某个 expert 特别热，最慢 expert/rank 仍决定尾部。

## 如何接入：先配置，再替换抽象，最后才写 kernel

### 路径 1：框架原生配置

优先使用 Megatron-Core/Transformer Engine 已有的 TransformerConfig、layer spec 和 backend selection。例如 selective fusion、RoPE、RMSNorm、bias-dropout-add、gradient accumulation fusion、Grouped GEMM 等。优点是 TP/PP/CP/EP、checkpoint 和 state-dict 语义通常已经集成。

但“flag=true”不等于目标 kernel 生效。仍要核对：

- GPU architecture、CUDA/driver、PyTorch、Transformer Engine/Apex 版本；
- BF16/FP16/FP8 dtype 与 dimension alignment；
- contiguous/layout、sequence/attention backend 和训练/推理模式；
- 是否因 head dim、mask、dropout、dynamic shape 或不支持的 model module fallback。

### 路径 2：通过 layer spec/module factory 替换接口

如果模型使用标准 PyTorch module，但框架识别不到对应 pattern，应在 model construction 层替换，而不是到处改 forward：

```text
HF / custom module
  → model adapter / layer spec
  → TE/Megatron fused module
  → 保持参数名、shape、dtype、并行 layout
  → checkpoint load/save compatibility
```

接口替换必须守住：state-dict key、weight transposition、QKV packing 顺序、bias 语义、dropout RNG、TP/EP shard、FP8 metadata、autograd 与 checkpoint conversion。

### 路径 3：Triton/CUDA 自定义 kernel

只有在 profiler 证明现有库没有命中、pattern 稳定、收益足够且团队能承担多架构维护时才进入。最低验收包括 shape/dtype matrix、forward/backward gradcheck、极值/NaN、determinism、fallback、benchmark 与不同 GPU architecture。

用户的项目边界应表述为“融合特性接入、配置调优、kernel 命中确认和数值/性能验收”，没有证据时不说“实现了底层 CUDA kernel”。

## 性能验证

### 先判断属于哪种 wall

| 观测 | 更可能的瓶颈 | 下一步 |
|---|---|---|
| 大 GEMM Tensor Core 利用低，shape 很碎 | compute efficiency / 并行切分 | 调 TP/EP、batch/token、Grouped GEMM |
| 大量短 kernel，GPU 中间有 launch gap | launch-bound | fusion、CUDA Graph、减少 Python/control path |
| DRAM throughput 高、算术强度低 | memory-bound | elementwise/norm fusion、减少 cast/layout copy |
| fused kernel 很快但前后 transpose 占主导 | layout conversion | 改数据布局或取消该 fusion |
| 单卡快，多卡收益消失 | communication exposed | group/topology/overlap，不继续堆 fusion |

### 同 workload A/B 最少记录

- 每 step/每 token 延迟、有效 tokens/s 与 MFU；
- kernel name/count、CPU launch gap、SM/Tensor Core/DRAM 指标；
- peak allocated/reserved 与临时 tensor shape；
- TP/CP/EP communication exposed time；
- loss、logits、grad、optimizer state 与长窗口稳定性。

不能把 microbenchmark 的 kernel speedup 直接外推到 step，更不能外推到端到端 RL。

## 常见失败模式与排障

### 开了开关但没有收益

1. profiler 确认是否命中 fused kernel，而不是读配置；
2. 检查 dtype、shape alignment、contiguous、mask/dropout 和 backend；
3. 比较为融合引入的 transpose/cast/copy；
4. 判断原路径是否已被 `torch.compile`、TE 或其他 backend 融合；
5. 若 GPU 已接近 compute-bound，大 GEMM 周边的小 fusion 对 step 贡献本来就有限。

### 性能更快但 loss 漂移

优先检查 softmax accumulation precision、QKV packing/RoPE 顺序、dropout seed/offset、LayerNorm/RMSNorm epsilon、FP8 scaling/amax recipe、loss reduction 与 TP/CP mask。必须用 first-divergence 逐层对齐，而不是只比较最终 loss。

### 显存反而增加

检查 workspace、autotune cache、FP8 metadata、CUDA Graph private pool、fallback 是否同时保留 unfused tensor，以及 activation/checkpoint 生命周期是否改变。

## 与相邻系统的关系

- [FlashAttention](flashattention.md)：Attention IO 算法与长上下文；
- [MoE](moe.md)：Grouped GEMM、token dispatcher 与 expert overlap；
- [Long-context Training](long_context_training.md)：selective recompute、CP-local loss；
- [Megatron 5D 并行](distributed_training.md#five-d-config)：并行切分决定 GEMM shape 与 collective；
- [CUDA Graph / Agentic RL](agentic_rl.md#cuda-graph-decode)：fusion 优化 kernel 内/间 IO，CUDA Graph 主要优化重复提交，两者互补。

## 面试回答模板

> NVIDIA 卡上仍需要融合算子，主要针对 memory-bound 和 launch-bound 链路，而不是假设大 GEMM 也能无限加速。常见的有 FlashAttention、QKV/RoPE、bias-activation、bias-dropout-residual、RMSNorm、vocab-parallel CE、gradient accumulation、fused optimizer，以及 MoE 的 permute/Grouped GEMM。接入时先用 Megatron-Core/Transformer Engine 的 config 和 layer spec；模型抽象不匹配才替换 module interface。开关之后用 profiler 看是否真的命中 kernel，同时做 unfused 数值对照、peak memory 和 fallback 测试。我的项目职责是接入、调优和验收，不把底层 kernel 实现归为个人贡献。

## 一手资料

- [Megatron-Core TransformerConfig](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.transformer.transformer_config.html)
- [Megatron-Core fused bias-dropout-add](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.fusions.fused_bias_dropout.html)
- [Megatron-Core tensor-parallel linear / gradient accumulation fusion](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.tensor_parallel.layers.html)
- [Megatron-Core MoE](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html)
- [FP8](fp8.md)
- [FlashAttention-3](../papers/flashattention3.md)
- [DeepSeek-V3](../tech_reports/deepseek_v3.md)

## 我的总结

融合算子是“数据移动与提交开销优化”，不是一个统一开关。先用 profile 判断 wall，再选择 framework-native fusion、module adaptation 或自定义 kernel；最后用性能、显存和数值三条证据闭环。真正高级的回答不是列出算子名，而是能说明为何融合、在哪个 layout 命中、何时 fallback，以及收益如何跨越不了通信和端到端 Amdahl 边界。

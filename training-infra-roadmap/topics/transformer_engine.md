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

<a id="gpu-execution"></a>
## GPU 执行与片上数据复用

Kernel 的 grid 被分成 thread blocks，block 调度到 SM；NVIDIA warp 通常是 32 个线程。数据来自显存（例如 HBM/GDDR），经缓存和片上存储供计算使用。GEMM 通过 tile 将数据放入 shared memory/寄存器并复用，减少慢速数据搬运。更大的 tile 或更多 pipeline stages 会消耗更多片上资源，可能减少驻留 blocks，所以 occupancy 不是越高越好。

| 机制 | 需要区分的边界 | 排查方向 |
|---|---|---|
| Coalescing | 同一 warp 的地址尽量落在少量内存事务中；连续访问通常有利，但还受对齐、访问宽度和 stride 影响 | 先看真实布局和访问模式，不只看 tensor shape |
| Register spill | 寄存器不够时部分数据落到 local memory；local 是线程私有地址空间，不代表片上 | 检查寄存器压力与额外 device-memory 访问 |
| Shared memory bank conflict | 不同线程访问同一 bank 的不同地址可能串行化 | 与显存访问的 coalescing 分开诊断 |
| Occupancy | 驻留活跃 warps 占硬件上限的比例，可以帮助隐藏延迟 | 资源压力、访存限制和执行依赖仍会限速，不能当作最终性能目标 |

### Tile 容量手算

BF16 GEMM 若 `BM=128, BN=128, BK=64`，A/B 单 stage 共 `(128*64+64*128)*2=32 KiB`；三 stage 约 96 KiB，尚未计其他开销。增加 stage 必须同时核算片上资源与驻留数量，不能先把 stage 拉满再期望更快。

来源：[CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)、[Hopper Tuning](https://docs.nvidia.com/cuda/archive/11.8.0/hopper-tuning-guide/index.html)。[面试短答：GPU 执行与访存](../../private_resume/2026-08-llm-infra-interview-prep.md#gpu-01)。

### 卡型与互联：先确认实际执行平台

比较型号时先明确 PCIe/SXM 等形态，再看可用显存、显存带宽/L2、dense Tensor Core 能力、SM 资源、GPU 互联和软件支持。A100 没有 Hopper 那样的原生 FP8 Tensor Core 路径；H100/H200 的差异也不能只看算力，容量和带宽会改变 workload 的瓶颈。Blackwell 的具体低精度路径仍需按 SKU 和 recipe 查支持矩阵。[Ampere](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/) · [Hopper](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) · [H200](https://www.nvidia.com/en-us/data-center/h200/)

通信粗估可以先写 `T ≈ 次数×启动延迟 + bytes/有效带宽`，再看 overlap、拥塞和慢 rank：小模型的频繁小消息更可能受 latency 影响，大张量更需关注带宽。先区分 rank 晚到 collective 与 collective 本身慢，具体原语和 ring 的数据流只在 [NCCL 章节](nccl.md#ring-allreduce)及[主文档通信题](../../private_resume/2026-08-llm-infra-interview-prep.md#infra-04)维护。

<a id="pytorch-execution"></a>
## PyTorch 执行：Autograd 与 tensor layout

Autograd 在前向记录需要梯度的运算与依赖，反向按链式法则使用保存的中间量。Tensor 还包括 storage、shape、stride 和 offset，shape 一样不代表布局一样。`view` 共享存储且要求 stride 兼容；`reshape` 可能复制；`transpose` 常只改变元数据，但后续 kernel 可能因此需要不同访存或额外 `contiguous()`。这也是融合算子接入必须验证 layout，而不只是核对 shape 的原因。

连续 `x:[2,3]` 的 stride 是 `(3,1)`；`x.T:[3,2]` 的 stride 是 `(1,3)`，元素地址可写成：

```text
base + (i * stride0 + j * stride1 + offset) * element_size
```

转置本身便宜，不代表后续 matmul 前的数据整理免费。验证时同时检查共享存储关系、stride、实际复制与 kernel 路径。

正确性还涉及反向保存状态：in-place 改写反向所需 tensor 可能触发 version-counter 错误；`detach()` 切断梯度链，不等于深拷贝；`.grad` 会累积，训练循环需要有意地清零或累计。不要用到处加 `contiguous()` 和 `retain_graph=True` 掩盖布局或图生命周期问题。Linear 的具体前后向 shape、归约轴和 dtype 边界见 [三次 GEMM](fp8.md#fp8-gemm)。

来源：[Tensor Views](https://docs.pytorch.org/docs/stable/tensor_view.html)、[Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)。[面试短答：Autograd / layout](../../private_resume/2026-08-llm-infra-interview-prep.md#pytorch-01) · [Linear 与 AMP](../../private_resume/2026-08-llm-infra-interview-prep.md#pytorch-03)。

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

<a id="torch-compile"></a>
### 编译路径：torch.compile 与 CUDA Graph

典型 PyTorch 编译路径由 Dynamo 捕获 Python 中的 tensor 运算和 guards，AOTAutograd 处理前后向图，再由 Inductor 优化并生成代码或调用已有 kernel。它可能做融合、减少中间量，不是把整个程序编成一个 kernel。CUDA Graph 主要是捕获并重放稳定的 GPU 工作流，减少 CPU launch 开销，不会自动重写数学算法；两者可以组合。

| 现象/配置 | 机制与边界 |
|---|---|
| Graph break | 无法继续捕获一段程序，可能退回 eager，之后再捕获 |
| Recompile | 输入 shape/stride、Python 值等使所有已有缓存版本的 guards 都不满足，才需要重新编译；另一缓存版本匹配时可以直接复用 |
| `fullgraph=True` | 遇到 graph break 通常报错，适合定位，不等于保证产出单个 GPU kernel |
| `dynamic=True` | 可用于动态 shape 场景，但不保证完全不重编译 |

用 `TORCH_LOGS="graph_breaks,recompiles,guards"` 定位捕获与缓存问题；可依次用 backend `eager`、`aot_eager`、`inductor` 缩小失败层级，它们不是三个性能档位。预热、编译和 steady-state 时间应分开报告。

对于不同 vertex budget 或视频尺寸等变长场景，先统计 shape 分布，再评估 bucketing/padding、动态 shape 和编译缓存。不能为了命中图把无效 padding 当成有效工作量，最后只报 kernel 时间。已有某个 decode workload 的 CUDA Graph 收益，也不能直接外推为 Diffusion 加速。

来源：[torch.compiler](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler.html)、[Troubleshooting](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_troubleshooting.html)。[面试短答：torch.compile](../../private_resume/2026-08-llm-infra-interview-prep.md#pytorch-02) · [eager / compile 计时练习](../../private_resume/2026-09-interview-coding.md#coding-06)。

## 性能验证

<a id="roofline"></a>
### Roofline：同一个 Linear 为什么随 batch 改变瓶颈

小 batch 对同一份权重的复用少，可能受带宽或 launch 限制；batch 增大后，一次权重读取能服务更多计算。先用 Roofline 估算，再看真实 DRAM/L2 流量和 kernel。模型小不等于一定快，也不等于必然 memory-bound。

设 `X[M,K] @ W[K,N]`，BF16 输入/输出，忽略 bias、输出旧值读取和其他中间量，且 A/B 各读一次、C 写一次：

```text
FLOPs ≈ 2MNK
理想最低 bytes ≈ 2(MK + KN + MN)
Arithmetic intensity I ≈ MNK / (MK + KN + MN)
算力上界 ≈ min(峰值计算吞吐, 对应层级带宽 × I)
```

手算 `K=N=4096`：M=1 时约 **1 FLOP/byte**；M=512 时约 **409.6 FLOP/byte**。若假设一张教学 GPU 的 dense BF16 峰值是 200 TFLOPS、HBM 带宽 2 TB/s，其 ridge point 是 100 FLOP/byte。后者越过理想 ridge 不等于实测能跑满，还要考虑并行度、tile、padding 和调度；这里是模型推导，不是特定 GPU 或真实 workload 的测量结果。

这份权重约 32 MiB，反复微基准可能命中 L2，此时不能用冷 HBM 的模型解释结果。不同 dtype、dense/sparse 峰值、PCIe/SXM 卡型不能混用。报告“有效 GB/s”时要说明是理论 bytes 还是 profiler 实际流量。低精度还可能引入额外量化、转置和临时 buffer，需结合 [FP8 端到端验证](fp8.md#fp8-debug) 重新记账。

来源：[NVIDIA GEMM 性能模型](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)。[面试短答：Roofline 与 batch](../../private_resume/2026-08-llm-infra-interview-prep.md#gpu-02) · [GPU 计时练习](../../private_resume/2026-09-interview-coding.md#coding-06)。

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

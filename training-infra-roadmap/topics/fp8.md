# FP8

## 核心问题

在 H100/H800 等硬件上用 FP8 降低矩阵乘的存储和带宽压力，提高吞吐，但必须控制数值稳定性。

## 相关材料

- [DeepSeek-V3](../tech_reports/deepseek_v3.md)
- [Transformer Engine](transformer_engine.md)
- [FlashAttention-3](../papers/flashattention3.md)

## 关键机制

<a id="float-formats"></a>
### 浮点格式：范围、间隔与状态存储

Exponent 决定动态范围，fraction 决定相邻可表示值的间隔。BF16 的范围接近 FP32，但有效精度比 FP16 低；它通常不需要 FP16 那样的 loss scaling。TF32 是处理 FP32 矩阵计算的一种 Tensor Core 精度模式，不是把模型存成 19-bit。FP8/FP4 则必须结合 scale、accumulation 和训练 recipe，不能让整个训练链路一律降精度。

| 格式 | sign / exponent / fraction | 未 scaling 的最大有限值 | `[1,2)` 内的间隔 |
|---|---:|---:|---:|
| FP16 | 1 / 5 / 10 | 65504 | `2^-10` |
| BF16 | 1 / 8 / 7 | 约 `3.39×10^38` | `2^-7` |
| TF32 运算精度 | 1 / 8 / 10 | FP32 数量级 | `2^-10` |
| NVIDIA FP8 E4M3 | 1 / 4 / 3 | 448 | `2^-3` |
| FP8 E5M2 | 1 / 5 / 2 | 57344 | `2^-2` |
| FP4 E2M1 | 1 / 2 / 1 | 6 | `2^-1` |

表中间隔可由 fraction 位数手算：`1 + 2^-8` 可由 FP16 精确表示，却位于 BF16 的两个相邻值正中间；round-to-nearest-even 得到 1。因此“BF16 精度比 FP16 高”混淆了范围和精度。FP16 最小 normal 是 `2^-14`，最小 subnormal 是 `2^-24`；subnormal 没有 normal 数隐含的前导 1，实际执行还需确认是否 flush-to-zero。

格式只解释一个数如何表示，不能直接推出训练总显存。低精度量化副本、master weights、梯度和 optimizer states 必须分别记账；临时 layout、workspace 与 activation 生命周期也可能抵消节省。

来源：[TE 混合精度](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/introduction/introduction.html)、[TF32](https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/)、[NVIDIA 混合精度指南](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)、[PTX 数值格式](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)。[面试短答：浮点格式](../../private_resume/2026-08-llm-infra-interview-prep.md#precision-01)。

<a id="fp8-scaling"></a>
### Scale 约定、Current 与 Delayed scaling

先统一符号：

```text
q = Q(x / s)
x_hat = s * q
r = 1 / s
```

这里 `s` 是反量化 scale，`r` 是量化乘数；有些 API 存的是 `r`，不能只看变量名判断乘除方向。Current scaling 从当前 tensor 的 amax 计算 scale，再量化，需要先观察当前数据；Delayed scaling 用历史 amax 预测 scale，量化时同时记录新 amax，减少额外读取，但分布突变时可能饱和。两者的代价比较必须同时考虑数据搬运和数值误差。[Current](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/fp8_current_scaling/fp8_current_scaling.html) / [Delayed](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/fp8_delayed_scaling/fp8_delayed_scaling.html)

**手算历史滞后的饱和**：E4M3 最大有限值为 448，历史 `amax=2`，则 `r=448/2=224`。若当前出现 4，`4*r=896` 超出范围；采用饱和量化时截到 448，反量化只有 `448/224=2`。预留 margin 可以降低饱和风险，但会牺牲小数值的表示能力。全零 tensor 需要实现定义安全 scale，不能直接除零。

配置与验证还需记录 scale 的分组、amax history 更新方式和 checkpoint 恢复语义。观察 amax 不够，还要看非零值量化成零的比例、饱和比例和相对误差。训练 loss scaling 与 FP8 tensor scaling 不是一回事：前者沿链式法则缩放梯度，后者为各 tensor 选择表示范围。

[面试短答：Scale 与量化](../../private_resume/2026-08-llm-infra-interview-prep.md#precision-02)。

<a id="fp8-gemm"></a>
### Linear 的三次 GEMM 与精度边界

按 PyTorch 权重布局，`X:[M,K]`、`W:[N,K]`，前向 `Y=XWᵀ`；上游梯度 `G=dY:[M,N]`，反向 `dX=GW`、`dW=GᵀX`、`db=sum(G, dim=0)`。三次 GEMM 的归约轴不同，不能只测 forward 就宣称训练加速。

| 运算 | shape | reduction 轴 | 经典 FP8 HYBRID 的量化输入 | 示例 `M=256,K=1024,N=4096` |
|---|---|---|---|---|
| Forward | `[M,K] @ [K,N]` | K | X：E4M3；W：E4M3 | 每次约 `2MNK=2^31` FLOPs |
| dX | `[M,N] @ [N,K]` | N | G：E5M2；W：E4M3 | 同上 |
| dW | `[N,M] @ [M,K]` | M | G：E5M2；X：E4M3 | 同上；不包含跨 DP 的梯度归约 |

所以“backward 的两个输入全是 E5M2”不准确：反向仍使用前向 activation/weight 的低精度表示。具体格式取决于 recipe，不能将 HYBRID 的表格推广到所有 block-scaling 路径。

实现中应分开检查五层：**GEMM 输入 dtype → 内部累加路径 → 输出 dtype → 跨 microbatch 的梯度累积 → optimizer states**。输出 BF16 不等于内部仅用 BF16 累加；内部累加路径还受硬件/kernel 配置影响。FP32 master copy 的作用之一，是累积低精度权重无法表示的小更新，但 `autocast` 不会自动创建独立 master weights，是否保留由训练/optimizer 实现决定。[TE GEMM 与布局](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/performance_considerations/performance_considerations.html)

**FP16＋GradScaler 的更新顺序**：

```text
前向 autocast
  → scaler.scale(loss).backward()
  → scaler.unscale_(optimizer)
  → 按需裁剪梯度
  → scaler.step(optimizer)
  → scaler.update()
```

发现 Inf/NaN 时由 scaler 跳过 optimizer 更新。梯度累积期间 scale 保持不变，到完整 effective batch 后才 unscale/update；BF16 通常不需要 GradScaler。[PyTorch AMP 示例](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)

[面试短答：Linear 前后向与 AMP](../../private_resume/2026-08-llm-infra-interview-prep.md#pytorch-03) · [NumPy 方向梯度校验](../../private_resume/2026-09-interview-coding.md#coding-05)。

<a id="block-scaling"></a>
### Block scaling：MXFP8 / NVFP4 与版本约束

整 tensor 共用 scale 时，少数 outlier 会占据范围，使其余数值表示变粗；block scaling 将影响限制到局部。更细的 scale 有额外存储、计算和 layout 约束，不是免费精度。

| 路径 | 数据与 scale | 分组及转置边界 |
|---|---|---|
| MXFP8 | FP8 数据，每 32 元素共享 E8M0 二次幂 scale | rowwise 与 columnwise 分组不同，需从高精度源分别量化，不能只转置量化值就认为等价 |
| NVFP4 | E2M1 数据、E4M3 block scale、FP32 全局 scale | 基本分组为 16 元素；TE 2.18 训练 recipe 默认 weights 用 16×16 的 2D scaling，activation/gradient 用 1×16 |

不能混淆 **原生指令能力、CUDA/TE 支持、训练 recipe** 三层约束。核对于 2026-09-09 的 **TE 2.18** MXFP8/NVFP4 训练文档明确列出 SM 10.0/10.3，不能将支持范围外推到所有 Blackwell SKU。执行前先确认实际卡型，再查所用版本矩阵和 shape 限制。[MXFP8](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/mxfp8/mxfp8.html) / [NVFP4](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html)

NVFP4 训练还结合随机舍入、随机 Hadamard 变换等减少量化误差的机制。完整训练效果不能全部归功于四位编码本身；在比较 recipe 时，应区分数值表示、额外稳定性机制与实际 kernel 路径。

[面试短答：MXFP8 / NVFP4](../../private_resume/2026-08-llm-infra-interview-prep.md#precision-04)。

## 生产关注

- FP8 不是全模型无脑替换；哪些层用、哪些状态保留 BF16/FP32 很关键。
- 需要监控 scale/amax 异常、loss spike 和溢出。
- 不同硬件和 kernel 的支持矩阵要严格验证。

<a id="precision-validation"></a>
<a id="fp8-debug"></a>
## 从机制到验证：不止改变 dtype

低精度评估先拆四层：**格式可表示范围 → scale/量化误差 → 实际 GEMM 与数据搬运 → 端到端训练质量**。BF16 适合作为对照，但也不是无限精确的真值；FP8/FP4 则不能只凭 loss 不出现 NaN 验收。

工程排障顺序：

1. 固定数据、checkpoint 和配置，建立可复现 BF16 对照，找第一次偏离的层/步骤；分别观察 fprop、dgrad、wgrad 的 amax、scale、误差、饱和和零值比例。
2. 用按层/按 GEMM 回退缩小范围；恢复才异常时检查 scale/history，分布式才异常时检查 amax reduction group、row/column scale 对应的 tensor，不能一律归咎于学习率。
3. 用 profiler 拆量化、layout、GEMM、通信与其他算子；小 GEMM 可能无法摊薄量化和 launch 成本。
4. 通过短窗口数值检查后，再看长期收敛与任务质量。对于 3D/视频，增加几何/拓扑或时序回归是工程建议，不是“某一格式已证明适用于全部模型”的结论。

### 数值误差与执行路径分开验证

对比 `||x_hat-x||₂/(||x||₂+eps)` 与梯度 cosine，避免只看全模型 loss 或 gradient norm。保留量化误差、使用高精度 GEMM 的对照，可以帮助区分表示误差和低精度执行路径问题；TE 提供 tensor stats、FakeQuant、按层/按 GEMM 禁用量化等诊断机制。BF16 对照同样受浮点舍入和执行顺序影响，不能视为无限精确的真值。[TE Debug API](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/debug/3_api_features.html) / [PyTorch Numerical Accuracy](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html)

### GEMM 加速不能直接外推到 step

假设原 step 60% 是可加速 GEMM，GEMM 快 2 倍，则总加速最多 `1/(0.4+0.6/2)=1.43x`；若新增量化开销又占原 step 的 10%，只剩 `1/(0.4+0.6/2+0.1)=1.25x`。这是 Amdahl 教学计算，不是实测结果。还需检查实际 kernel 命中及端到端开销，相关性能模型见 [Roofline 与小 GEMM](transformer_engine.md#roofline)。

[面试短答：低精度数值与性能排障](../../private_resume/2026-08-llm-infra-interview-prep.md#precision-03)。本页是机制与验证方法，不代表已完成个人 FP8/FP4 生产落地；实践结论需另有 workload、配置和测量证据。

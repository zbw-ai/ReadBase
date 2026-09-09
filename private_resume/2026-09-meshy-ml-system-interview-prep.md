<a id="meshy-interview-top"></a>
# Meshy ML System 技术面：低精度、GPU、PyTorch 与 3D / 视频生成

> [主文档 Meshy 速查](2026-08-llm-infra-interview-prep.md#meshy-interview-sprint) · [笔试基础题与 NumPy 解答](2026-09-meshy-ml-system-written-prep.md#meshy-top) · [通用 Coding](2026-09-interview-coding.md#coding-top)
>
> 更新：2026-09-09。笔试已通过；技术一面拟约 **2026-09-15（周二）15:00，待正式确认**。本页是依据 JD、用户转述反馈和官方技术资料设计的复习题，**不是 Meshy 真题或必考清单**。

## 0. 范围与使用方法

本轮先补 **数值计算 → GPU 执行 → PyTorch 机制 → 动手验证**，再把自己的数据、显存、并行和视频模型适配经验接上去。RL 是已有长项，不占这次准备的大头。每题至少做到：**一句话说明机制 → 手算一个例子 → 写出关键代码/数据流 → 说明如何验证**。

### 已知流程与待确认项

- **本人收到的信息**：笔试之后两轮技术面，包含编码、环境配置和共享屏幕；终面线下见主管，CEO 线上参与。暂按自带电脑准备，不代表对方已确认设备安排。
- **转述反馈**：低精度、GPU 细节和理论基础可能问得深入；负责人较 hands-on；近期在组建视频团队。作为准备信号使用，不写成经独立核实的公司制度。“毕业久就很难通过”是个人判断，不是筛选规则。
- **公开招聘**：[官方湾区 ML System 岗](https://jobs.ashbyhq.com/meshy/90988ed5-f767-4c0d-9cbc-b69d792db1a9)的搜索可见招聘正文提到 GPU/Jupyter 和现场编码，但不能把海外流程、时长或资料权限套到本次国内面试；网页直接打开可能仅显示动态加载壳。
- **面前确认**：是否提供远程 NVIDIA GPU、具体型号/系统/PyTorch 版本、允许使用哪些资料和工具、屏幕共享范围、工作地点及线下面试地点。允许上网不等于允许 AI；未明确时按独立编码练习。求职地域仍以深圳及既定通勤范围为准。

### 按 topic 查题，再按优先级学习

| Topic | P0：先掌握 | P1：追问与迁移 | P2：终面延伸 |
|---|---|---|---|
| A｜数值与低精度 | [A1 格式](#meshy-a1) · [A2 Scale](#meshy-a2) · [A3 三次 GEMM](#meshy-a3) · [A4 收敛/性能排障](#meshy-a4) | [A5 MXFP8/NVFP4](#meshy-a5) | — |
| B｜GPU 与性能模型 | [B1 执行与存储](#meshy-b1) · [B2 Roofline 手算](#meshy-b2) | [B3 卡型与通信](#meshy-b3) | — |
| C｜现代 PyTorch | [C1 Autograd/Stride](#meshy-c1) · [C2 Compile](#meshy-c2) · [C3 FSDP2](#meshy-c3) | — | — |
| D｜数据与规模训练 | [D1 数据 pipeline](#meshy-d1) | [D2 扩容与恢复](#meshy-d2) | — |
| E｜3D / Diffusion | [E1 训练与推理](#meshy-e1) | [E2 多阶段优化](#meshy-e2) · [E3 3D 表征](#meshy-e3) | — |
| F｜Python3 动手 | [F1 数学到代码](#meshy-f1) · [F2 性能测量](#meshy-f2) | [F3 Triton 融合](#meshy-f3) | — |
| G｜现场与项目交流 | [G1 配置环境](#meshy-g1) | [G2 项目映射](#meshy-g2) · [G3 公开工作](#meshy-g3) | [G4 主管/CEO 交流](#meshy-g4) |

题尾返回本表；要回到刚才的滚动位置，用浏览器后退：macOS `⌘ + [`，Windows/Linux `Alt + ←`。

<a id="meshy-plan"></a>
### 从现在到拟约面试的安排

| 日期 | 主任务 | 当天验收 |
|---|---|---|
| 9/10 周四 | A1–A4：格式、scale、反向、低精度诊断；避开已排的 Infix 16:00 面试 | 脱稿解释 BF16 精度/范围；写出 Linear 三次 GEMM；手算一次量化饱和 |
| 9/11 周五 | B1–B3＋F3：GPU、Roofline、小融合 kernel | 手算 FLOPs/bytes；说明 mask、tiling、occupancy；有 GPU 才做实测 |
| 9/12 周六 | C1–C3＋F2：Autograd、compile、FSDP2 | 独立写 benchmark；画出参数 AG/释放/梯度 RS；定位一次 graph break |
| 9/13 周日 | D/E：数据加载、扩容、Diffusion、3D | 画数据到 GPU 的链路；说明非自回归生成；完成一个端到端性能方案 |
| 9/14 周一 | 45–60 分钟独立编码模拟＋项目追问＋G3 公开工作 | 不看答案完成 Attention/梯度检查；把真实优化讲到 tensor 和 timeline |
| 9/15 周二 | 面前 30–45 分钟热身，15:00 为拟约时间 | 检查环境与分享范围，复述三道薄弱题；不临时升级驱动/依赖 |

**只有两小时时**：低精度 40 分钟 → GPU/Roofline 25 分钟 → PyTorch 25 分钟 → 数据/Diffusion 15 分钟 → 编码与环境 15 分钟。不是两小时学完所有内容，优先读 P0 的直接回答与例子。

## A｜数值计算与低精度训练

<a id="meshy-a1"></a>
### A1｜FP16、BF16、TF32、FP8、FP4 差在哪里？（P0）

**考察意图**：能否从浮点表示推出范围、误差和训练行为，而不是只背 dtype 名称。

**直接回答**：Exponent 决定范围，fraction 决定相邻可表示值的间隔。BF16 范围接近 FP32，但有效精度比 FP16 低；它通常不需要 FP16 那样的 loss scaling。TF32 是处理 FP32 矩阵计算的一种 Tensor Core 精度模式，不是把模型存成 19-bit。FP8/FP4 还要结合 scale、accumulation 和训练 recipe，不能让整个训练链路都一律降精度。

| 格式 | sign / exponent / fraction | 未 scaling 的最大有限值 | `[1,2)` 内的间隔 |
|---|---:|---:|---:|
| FP16 | 1 / 5 / 10 | 65504 | `2^-10` |
| BF16 | 1 / 8 / 7 | 约 `3.39×10^38` | `2^-7` |
| TF32 运算精度 | 1 / 8 / 10 | FP32 数量级 | `2^-10` |
| NVIDIA FP8 E4M3 | 1 / 4 / 3 | 448 | `2^-3` |
| FP8 E5M2 | 1 / 5 / 2 | 57344 | `2^-2` |
| FP4 E2M1 | 1 / 2 / 1 | 6 | `2^-1` |

**手算**：`1 + 2^-8` 可由 FP16 精确表示，却在 BF16 的两个相邻值正中间；round-to-nearest-even 得到 1。FP16 最小 normal 是 `2^-14`，最小 subnormal 是 `2^-24`；subnormal 没有 normal 数隐含的前导 1，实际执行还需确认是否 flush-to-zero。

**危险回答**：“BF16 精度比 FP16 高”“FP8 自动让训练总显存减半”。低精度量化副本、master weights、梯度和 optimizer states 要分别记账。

来源：[TE 混合精度](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/introduction/introduction.html)、[TF32](https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/)、[NVIDIA 混合精度指南](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)、[PTX 数值格式](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)。

↑ [返回速查](#meshy-interview-top)

<a id="meshy-a2"></a>
### A2｜FP8 的 scale 怎么计算？Current 与 Delayed 有何区别？（P0）

**考察意图**：理解量化误差与数据读取代价，能自己检查代码里 scale 的方向。

**直接回答**：先统一符号：`q = Q(x/s)`，反量化 `x_hat = s*q`，这里 `s` 是反量化 scale。有些 API 存的是它的倒数。Current scaling 从当前 tensor 的 amax 算 scale，再量化，需要先观察当前数据；Delayed 使用历史 amax 预测 scale，量化时同时记录新 amax，减少额外读取，但分布突变时可能饱和。不能只说后者快而不提这个数值代价。

**手算饱和**：E4M3 最大有限值 448，历史 amax=2，则量化乘数 `r=1/s=448/2=224`。若新值变成 4，`4*r=896` 超出范围；采用饱和量化时截到 448，反量化只有 2。预留 margin 可以降低饱和风险，但会牺牲小数值的表示能力。全零 tensor 需要实现定义安全 scale，不能除零。

**追问**：观察 amax 不够，还要看量化成零的比例、饱和比例和相对误差。Scale 的分组、更新历史、checkpoint 恢复必须与实际 recipe 对齐。训练 loss scaling 与 FP8 tensor scaling 不是一回事：前者沿链式法则缩放梯度，后者为各 tensor 选择表示范围。

来源：[Current scaling](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/fp8_current_scaling/fp8_current_scaling.html)、[Delayed scaling](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/fp8_delayed_scaling/fp8_delayed_scaling.html)。

↑ [返回速查](#meshy-interview-top)

<a id="meshy-a3"></a>
### A3｜Linear 前向和反向怎么算？低精度用在哪？（P0）

**考察意图**：数学、shape、dtype 和训练系统是否能连起来。

**直接回答**：按 PyTorch 权重布局，`X:[M,K]`、`W:[N,K]`，前向 `Y=XWᵀ`；上游梯度 `G=dY:[M,N]`，反向 `dX=GW`、`dW=GᵀX`、`db=sum(G, dim=0)`。三次 GEMM 的归约轴不同，不能只测 forward 就宣称训练加速。还要分开说明输入 dtype、内部累加、输出 dtype、跨 microbatch 的梯度累积和 optimizer states。

| 运算 | shape | reduction 轴 | 示例 `M=256,K=1024,N=4096` |
|---|---|---|---|
| Forward | `[M,K] @ [K,N]` | K | 每次约 `2MNK=2^31` FLOPs |
| dX | `[M,N] @ [N,K]` | N | 同上 |
| dW | `[N,M] @ [M,K]` | M | 同上；不包含跨 DP 的梯度归约 |

经典 FP8 HYBRID 中，X/W 可采用 E4M3，dY 采用 E5M2，因此不能说“backward 的两个输入全是 E5M2”。具体内部累加路径还受硬件/kernel 配置影响，输出 BF16 不等于内部仅用 BF16 累加。

**FP16＋GradScaler 的更新顺序**：前向 autocast → `scaler.scale(loss).backward()` → `scaler.unscale_(optimizer)` → 按需裁剪梯度 → `scaler.step(optimizer)` → `scaler.update()`。发现 Inf/NaN 时由 scaler 跳过 optimizer 更新；梯度累积期间 scale 保持不变，到完整 effective batch 后才 unscale/update。BF16 通常不需要 GradScaler。`autocast` 不会自动创建一份独立 FP32 master weights；是否有 master copy 由训练/optimizer 实现决定。FP32 master copy 的作用之一是累积低精度权重无法表示的小更新。

来源：[TE GEMM 与布局](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/performance_considerations/performance_considerations.html)、[PyTorch AMP](https://docs.pytorch.org/docs/stable/amp.html)。动手：[F1 梯度校验](#meshy-f1)。

↑ [返回速查](#meshy-interview-top)

<a id="meshy-a4"></a>
### A4｜低精度不收敛，或者反而更慢，怎么排查？（P0）

**考察意图**：能否把“数值问题”和“没有执行到高效路径”拆开。

**直接回答**：先固定数据、checkpoint 和配置，建立可复现 BF16 对照；从第一次偏离定位到层，再区分 fprop、dgrad、wgrad。观察 activation/gradient 的 amax、scale、零值/饱和比例、相对误差和梯度方向，逐层回退精度。性能上确认实际 kernel，再拆量化、layout/transpose、GEMM、通信和剩余算子。小 GEMM 的量化与 launch 成本可能吃掉全部收益。

- **数值实验**：对比 `||x_hat-x||₂/(||x||₂+eps)` 和梯度 cosine；用保留量化误差的高精度计算区分表示误差与执行路径问题。TE 提供 tensor stats、FakeQuant、按层/按 GEMM 禁用量化的诊断机制。
- **恢复/多卡才坏**：检查 scale/history 是否正确恢复，amax reduction group 和 row/column scale 是否对应正确 tensor；不能一律归咎于学习率。
- **性能手算**：假设原 step 60% 是可加速 GEMM，GEMM 快 2 倍，总加速最多 `1/(0.4+0.6/2)=1.43x`；若新量化开销又占原 step 的 10%，只剩 `1.25x`。这是 Amdahl 示例，不是个人实测。
- **验收**：短窗口 loss/gradient 对齐，再做足够长的收敛与任务评估；3D/视频还要验证几何、拓扑、纹理或时序一致性。无 NaN 不等于效果没有退化。

**项目边界**：FP8/FP4 在这里是原理与验证方案，不把它写成自己已负责过完整生产收敛。实际证据优先讲已有的性能与精度对齐项目。

来源：[TE Debug API](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/debug/3_api_features.html)、[PyTorch Numerical Accuracy](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html)。

↑ [返回速查](#meshy-interview-top)

<a id="meshy-a5"></a>
### A5｜MXFP8、NVFP4 为什么采用 block scaling？（P1）

**直接回答**：整 tensor 共用 scale 时，少数 outlier 会占据范围，使其余数值表示变粗；block scaling 将影响限制到局部。MXFP8 每 32 元素共享 E8M0 二次幂 scale。NVFP4 用 E2M1 数据、E4M3 block scale 和 FP32 全局 scale。更细的 scale 有额外存储、计算和 layout 约束，不是免费精度。

**三个重点追问**：

1. **NVFP4 都是 1×16 吗？** 基本编码按 16 元素分组，但核对的 TE 2.18 训练 recipe 默认 weights 用 **16×16 的 2D scaling**，activation/gradient 用 1×16；不要把编码规则和训练 recipe 混为一谈。
2. **转置后 scale 还成立吗？** MXFP8 沿归约方向分组，rowwise 与 columnwise 的组不相同；常需从高精度源分别量化，不能仅转置量化值就假设等价。
3. **Blackwell 都一样吗？** 原生指令能力、CUDA/TE 支持和训练 recipe 是三层约束。核对的 TE 2.18 MXFP8/NVFP4 训练页明确列出 SM 10.0/10.3；不能把这一支持范围外推到所有 Blackwell SKU。先问实际卡型，再查所用版本矩阵和 shape 限制。

更深一层：NVFP4 训练还结合随机舍入、随机 Hadamard 变换等减少量化误差的机制，不能把完整训练效果归功于四位编码本身。这部分先会解释动机，不必无依据地背内部实现参数。

来源：[MXFP8](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/mxfp8/mxfp8.html)、[NVFP4](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html)。

↑ [返回速查](#meshy-interview-top)

## B｜GPU 执行、内存与性能模型

<a id="meshy-b1"></a>
### B1｜一个 GPU kernel 怎样执行？为什么要 tiling？（P0）

**直接回答**：Kernel 的 grid 被分成 thread blocks，block 调度到 SM；NVIDIA warp 通常是 32 个线程。数据来自显存（例如 HBM/GDDR），经缓存和片上存储供计算使用。GEMM 通过 tile 将数据放入 shared memory/寄存器并复用，减少慢速数据搬运。更大的 tile 或更多 pipeline stages 会消耗更多片上资源，可能减少驻留 blocks，所以 occupancy 不是越高越好。

**必须展开的术语**：

- **Coalescing**：同一 warp 的地址尽量落在少量内存事务中；连续访问通常有利，但还受对齐、访问宽度和 stride 影响。
- **Register spill**：寄存器不够时部分数据落到 local memory；“local”是线程私有地址空间，不代表片上，可能产生 device-memory 访问。
- **Shared memory bank conflict**：不同线程访问同一 bank 的不同地址可能串行化；不是 HBM 的 coalescing 问题。
- **Occupancy**：驻留活跃 warps 占硬件上限的比例；帮助隐藏延迟，但过大资源压力、访存限制或执行依赖仍会限速。

**手算 tile**：BF16，`BM=128, BN=128, BK=64`，A/B 单 stage 共 `(128*64+64*128)*2=32 KiB`，三 stage 约 96 KiB，尚未计其他开销。不能先把 stage 拉满再期望更快。

来源：[CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)、[Hopper Tuning](https://docs.nvidia.com/cuda/archive/11.8.0/hopper-tuning-guide/index.html)。

↑ [返回速查](#meshy-interview-top)

<a id="meshy-b2"></a>
### B2｜同一个 Linear，batch=1 和大 batch 为什么不一样？（P0）

**直接回答**：小 batch 对同一份权重的复用少，可能受带宽或 launch 限制；batch 增大后，一次权重读取能服务更多计算。先用 Roofline 估算，再看真实 DRAM/L2 流量和 kernel。模型小不等于一定快，也不等于必然 memory-bound。

设 `X[M,K] @ W[K,N]`，BF16 输入/输出，忽略 bias、输出旧值读取和其他中间量，且 A/B 各读一次、C 写一次：

```text
FLOPs ≈ 2MNK
理想最低 bytes ≈ 2(MK + KN + MN)
Arithmetic intensity I ≈ MNK / (MK + KN + MN)
算力上界 ≈ min(峰值计算吞吐, 对应层级带宽 × I)
```

**手算**：`K=N=4096`，M=1 时约 **1 FLOP/byte**；M=512 时约 **409.6 FLOP/byte**。若假设一张教学 GPU 的 dense BF16 峰值是 200 TFLOPS、HBM 带宽 2 TB/s，其 ridge point 是 100 FLOP/byte。后者越过理想 ridge 不等于实测就能跑满，还要考虑并行度、tile、padding 和调度。

**追问**：这份权重约 32 MiB，反复微基准可能命中 L2，此时不能用冷 HBM 的模型解释结果。不同 dtype、dense/sparse 峰值、PCIe/SXM 卡型不能混用。报告“有效 GB/s”时要说明是理论 bytes 还是 profiler 实际流量。

来源：[NVIDIA GEMM 性能模型](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)。

↑ [返回速查](#meshy-interview-top)

<a id="meshy-b3"></a>
### B3｜问到具体 GPU 和通信，应该比较什么？（P1）

**直接回答**：先明确型号与形态，再比较可用显存、HBM/L2、dense Tensor Core 能力、SM 片上资源、GPU 间互联和软件支持。比如 A100 不具备 Hopper 那样的原生 FP8 Tensor Core 路径；H100 与 H200 不能仅按峰值 FLOPS 理解，内存容量/带宽也会影响 workload。Blackwell 的低精度仍要核查具体 SKU 和 recipe，不能用一个架构名代替兼容表。

**通信如何估算**：`T ≈ 次数×启动延迟 + bytes/有效带宽`，然后考虑 overlap 和拥塞。小模型的频繁小 all-reduce 更容易受 latency 影响；大张量更需看带宽。先确认 slow rank 是晚到 collective，还是 collective 本身传得慢。

**Ring 复习**：典型 Ring AllReduce = ReduceScatter + AllGather，每阶段 N−1 步；每 rank 发送约 `2(N−1)M/N` bytes，接收同量，M 是每 rank 输入大小。NCCL 不总选 ring；算法和协议会受拓扑、消息量、版本影响。[主文档 Ring](2026-08-llm-infra-interview-prep.md#ring-allreduce-quick)

来源：[Ampere 架构](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/)、[Hopper 架构](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)、[H200](https://www.nvidia.com/en-us/data-center/h200/)、[NCCL 性能口径](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)。

↑ [返回速查](#meshy-interview-top)

## C｜PyTorch 机制与框架选型

<a id="meshy-c1"></a>
### C1｜Autograd、view、stride、in-place 怎样影响正确性和性能？（P0）

**直接回答**：Autograd 在前向记录需要梯度的运算与依赖，反向按链式法则使用保存的中间量。Tensor 还包括 storage、shape、stride 和 offset，shape 一样不代表布局一样。view 共享存储且要求 stride 兼容；reshape 可能复制；transpose 常只改变元数据，但后续 kernel 可能因此需要不同访存或额外 contiguous。

**手算**：连续 `x:[2,3]` 的 stride 是 `(3,1)`；`x.T:[3,2]` 的 stride 是 `(1,3)`，地址可写成 `base + (i*stride0+j*stride1+offset)*element_size`。转置本身便宜，不代表后续 matmul 前的数据整理免费。

**追问**：in-place 改写反向所需 tensor 可能触发 version-counter 错误；`detach()` 切断梯度链，不等于深拷贝；`.grad` 会累积，训练循环需要有意地清零或累计。不要用到处加 `contiguous()` 和 `retain_graph=True` 的方式掩盖问题。

来源：[Tensor Views](https://docs.pytorch.org/docs/stable/tensor_view.html)、[Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)。

↑ [返回速查](#meshy-interview-top)

<a id="meshy-c2"></a>
### C2｜torch.compile 做了什么？和 CUDA Graph 有什么不同？（P0）

**直接回答**：典型 PyTorch 编译路径由 Dynamo 捕获 Python 中的 tensor 运算和 guards，AOTAutograd 处理前后向图，再由 Inductor 优化并生成代码或调用已有 kernel。它可能做融合、减少中间量，不是把整个程序编成一个 kernel。CUDA Graph 主要是捕获并重放稳定的 GPU 工作流，减少 CPU launch 开销，不会自动替你重写数学算法；两者可以组合。

- **Graph break**：无法继续捕获一段程序，可能退回 eager，之后再捕获。
- **Recompile**：输入 shape/stride、Python 值等使所有已有缓存版本的 guards 都不满足，才需要重新编译；另一缓存版本匹配时可以直接复用。
- **定位**：`TORCH_LOGS="graph_breaks,recompiles,guards"`；可依次用 backend `eager`、`aot_eager`、`inductor` 缩小失败层级，它们不是三个性能档位。
- **配置边界**：`fullgraph=True` 遇到 graph break 通常报错，适合定位；`dynamic=True` 不保证完全不重编译。预热、编译和 steady-state 时间分开报告。

**场景题**：不同 vertex budget 或视频尺寸变化时，先统计 shape 分布，再评估 bucketing/padding、动态 shape 和编译缓存；不能为了命中图把无效 padding 算进去，最后只报 kernel 时间。

来源：[torch.compiler](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler.html)、[Troubleshooting](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_troubleshooting.html)。个人 CUDA Graph 证据仍限于 [Agentic decode](2026-08-llm-infra-interview-prep.md#resume-13)，不外推为 Diffusion `6–8x`。

↑ [返回速查](#meshy-interview-top)

<a id="meshy-c3"></a>
### C3｜FSDP2 的参数在哪里？什么时候通信？小模型要用吗？（P0）

**直接回答**：PyTorch FSDP2 用 `fully_shard` 对参数做 DTensor 分片，通过 hooks 在计算前 all-gather 所需参数，反向后 reduce-scatter 梯度。它主要减少模型状态常驻显存，不会自动把单个样本的全部 activation 按 DP 切开。小模型能放下时先比较 DDP；若分片通信比省下的显存更贵，FSDP2 不一定更快。

```text
reshard_after_forward=True 的典型单组路径：
参数 shard → AG 完整参数 → forward → 释放完整参数
          → AG 完整参数 → backward → RS 梯度 → 更新本地 shard
```

“释放/reshard”在这里不意味着额外执行一个参数 ReduceScatter。`reshard_after_forward=False` 可以保留完整参数，省掉 backward 前的参数 AG，但占用更多显存。以 **PyTorch 2.11 文档**为例，默认 `None` 对非 root 通常取 True，root 取 False；必须按实际版本和配置说，不笼统宣称每层每次都执行两次 AG。

**粒度选择**：通常先对 blocks 自底向上 `fully_shard`，再处理 root；过大组峰值高、overlap 差，过小组通信启动多。Optimizer 在分片变换后创建。Root 管理未被子组覆盖的参数，不等于重新分片一遍所有子参数。

**与 Megatron 选型**：模型结构持续变化、希望贴近原生 PyTorch 时优先评估 FSDP2；需要成熟 TP/PP/CP/EP、已有 Megatron 模型适配时评估 Megatron-Core。不是“大就 Megatron、小就 FSDP”的绝对规则。[主文档完整选型](2026-08-llm-infra-interview-prep.md#megatron-11)

来源：[PyTorch 2.11 fully_shard](https://docs.pytorch.org/docs/2.11/distributed.fsdp.fully_shard.html)、[FSDP2 Tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)。

↑ [返回速查](#meshy-interview-top)

## D｜数据 pipeline 与分布式训练

<a id="meshy-d1"></a>
### D1｜3D 数据导致 GPU 等待，怎么定位与优化？（P0）

**直接回答**：先拆远端读取、解压/解析、采样/增强、组批、H2D 和 GPU 计算，记录每段时间与队列状态；用固定 GPU tensor 排除数据路径，再逐段加回来。按实际瓶颈做 shard 合并、缓存、并行预处理、预取和 pinned memory；不是直接把 `num_workers` 开到最大。

**可落地的顺序**：

1. 测各 rank 的 data wait、CPU/磁盘/网络、p50/p95；3D 资产大小差异、视频解码和坏样本可能造成长尾。
2. 扫描 `num_workers` 与 `prefetch_factor`，结合 `persistent_workers`；总并发要乘 rank 数。粗略预取内存随 `ranks×workers×prefetch×batch_bytes` 增长，还不包含解析副本与其他 buffer。
3. 合并大量小文件、预计算稳定的元数据/转换结果；对长度或 vertex 数分桶，减少 padding，同时检查随机性和样本分布。
4. `pin_memory=True` 配合 `non_blocking=True` 可为异步 H2D 创造条件；真正与计算重叠还需要正确 stream、依赖和 buffer 生命周期，不能只凭参数已开启宣称实现 overlap。
5. 验证样本 ID、mask、增强种子与恢复位置。IterableDataset 要按 rank 和 worker 分片，否则可能重复读；DistributedSampler 要处理 epoch 更新、补齐/丢尾语义。

**个人案例怎么接**：用 [SFT 31s→9.3s](2026-08-llm-infra-interview-prep.md#resume-05)说明观察数据等待与配置联调；总收益包括重计算和 TP/CP 调整，不把全部收益归给 DataLoader，也不编造未留存的单项消融数字。

来源：[PyTorch DataLoader](https://docs.pytorch.org/docs/stable/data.html)、[数据加载性能教程](https://docs.pytorch.org/tutorials/intermediate/intermediate_data_loading_tutorial.html)。

↑ [返回速查](#meshy-interview-top)

<a id="meshy-d2"></a>
### D2｜模型不大，扩到数百卡为什么仍可能低效？（P1）

**直接回答**：固定总 batch 的强扩展会使每卡计算变少，梯度通信和数据长尾占比升高；如果增大全局 batch 又改变了优化过程，不能只比较 step time。先保持 workload/训练口径可比，评估有效吞吐与 time-to-quality，再选择 DDP/FSDP 或必要的模型并行。

**优化与验收**：平衡可变长度样本的计算负载；合并合理大小的梯度 bucket 并 overlap；确认拓扑和慢 rank；按一致快照保存模型、optimizer、随机状态与 data cursor，故障恢复不能重复或漏掉训练数据而不知情。扩容收益需要扣除 checkpoint、失败重试和空转成本。

展开：[万卡特有问题](2026-08-llm-infra-interview-prep.md#infra-09) · [Checkpoint](2026-08-llm-infra-interview-prep.md#infra-08) · [通信原语](2026-08-llm-infra-interview-prep.md#infra-04)。

↑ [返回速查](#meshy-interview-top)

## E｜3D、视频与多阶段生成

<a id="meshy-e1"></a>
### E1｜Diffusion / Flow Matching 训练与推理在算什么？（P0）

**直接回答**：它们通常在带噪的连续表示上预测噪声、数据或速度等目标；训练会采样时间和扰动构造监督，不是每个训练样本都展开完整推理去噪链。推理则从噪声出发，重复调用网络按求解器逐步更新。每步可能并行处理大量空间/时空 tokens，不是 LLM 的逐 token 自回归 decode。

**最简单的 Flow Matching 教学例子**：取噪声 `x0`、数据 `x1`，`t∈[0,1]`，定义 `xt=(1-t)x0+t*x1`；监督速度是 `u=x1-x0`，最小化 `||vθ(xt,t,c)-u||²`。推理从 t=0 积分到 1。实际方法的路径、时间方向、参数化和 loss weighting 可能不同，先声明这个约定。

**追问**：去噪主干 self-attention 的输入随 latent/时间步变化，不能把 LLM 不变前缀的 KV reuse 原封不动搬来。但若 cross-attention 的文本条件固定，且 K/V 投影不依赖时间、权重也不变，其 K/V 可以精确缓存；Q 变化时 attention 输出仍需重算。joint-attention 中的文本状态是否更新则要看结构。跨步复用会变化的中间特征通常属于近似优化，和前述固定 K/V 缓存不同；减步、量化或近似 cache 都需要重新评估质量。[Cross-attention 实现参考](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_ltx.py) · [多阶段推理](#meshy-e2)

来源：[Flow Matching 原始论文](https://arxiv.org/abs/2210.02747)、[Diffusers 推理优化教程](https://huggingface.co/docs/diffusers/main/tutorials/fast_diffusion)。

↑ [返回速查](#meshy-interview-top)

<a id="meshy-e2"></a>
### E2｜小型多阶段生成服务怎么优化？（P1）

**直接回答**：先画 DAG，测预处理、编码、去噪/flow、解码、后处理与导出的时间、显存峰值和队列。优化用户等待最长的一段，不预设大矩阵乘一定主导。稳定 shape 的计算段评估 compile/CUDA Graph/fusion；跨阶段考虑共享常量、资源池、并发与背压，避免同时启动多个高显存阶段造成 OOM。

| 现象 | 先试什么 | 不能忽略的代价 |
|---|---|---|
| 很多短 kernel，CPU launch 间隙明显 | 融合、compile、适合的 graph capture | 首次编译、动态 shape、图内存、正确性 |
| 不同尺寸/vertex budget 频繁重编译 | shape bucket、缓存编译产物、动态 shape | padding 的无效计算、缓存增长 |
| 多步网络调用占主导 | 高效 Attention、低精度、减少不必要计算 | 减步/特征 cache 若改变算法需验证质量 |
| 编解码/几何后处理长尾 | 批处理、并行 CPU/GPU 实现、独立资源池 | CPU oversubscription、拷贝、阶段排队 |

**验收**：报告冷启动/稳态、p50/p95、并发与每个合格资产的成本；3D 结果还要可用，视频要关注时序质量。平均 latency 降低但失败率或重试升高，未必降低有效成本。

来源：[PyTorch Compile 与 Diffusers](https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/)。这里是工程设计题，不暗示 Meshy 内部采用这些具体实现。

↑ [返回速查](#meshy-interview-top)

<a id="meshy-e3"></a>
### E3｜Mesh、Point Cloud、Voxel、SDF、NeRF、3DGS 有什么区别？（P1）

**直接回答**：先看最终需要的是可编辑表面、可打印几何，还是视图合成。不同表征优化的是不同目标，互转通常不是无损的。

| 表征 | 存的是什么 | 系统代价与主要边界 |
|---|---|---|
| Mesh | 顶点、边/面及属性 | 直接描述表面；拓扑、法线、面朝向、自交影响编辑与打印 |
| Point Cloud | 离散位置及颜色/法线等 | 采样/邻域查询重要；点本身没有封闭表面连接 |
| Voxel | 三维网格单元上的占据或特征 | 稠密网格随线性分辨率呈立方增长；稀疏结构另有索引开销 |
| SDF / 隐式场 | 空间点到表面的有符号距离等函数 | 可提取零等值面；质量依赖采样、函数误差与提取分辨率 |
| NeRF | 空间密度与视角相关辐射颜色 | 主要面向体渲染；密度场不等同于 SDF 或直接可打印 mesh |
| 3D Gaussian Splatting | 位置、协方差、不透明度、颜色等高斯参数 | 高效视图渲染；高斯集合不自动定义 watertight mesh |

**与低精度关联**：坐标的小偏移可能破坏细薄结构、法线或表面接合，所以 loss 接近不代表资产可用。先确认坐标单位、归一化、左右手系和 camera convention，再讨论数值容忍度。不要未经证据声称“3D 全部必须 FP32”或“FP8 肉眼无差”。

来源：[Open3D Mesh](https://www.open3d.org/docs/release/tutorial/geometry/mesh.html)、[NeRF 原始项目](https://www.matthewtancik.com/nerf)、[3DGS 原始项目](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)。

↑ [返回速查](#meshy-interview-top)

## F｜Python 3 现场编码与测试

<a id="meshy-f1"></a>
### F1｜把数学写成可验证的代码：Attention 与 Linear 反向（P0）

**考察意图**：不是记忆答案，而是独立说明轴、广播、数值稳定和边界。

优先闭卷写 [稳定 softmax](2026-09-meshy-ml-system-written-prep.md#meshy-a02)、[NumPy Attention](2026-09-meshy-ml-system-written-prep.md#meshy-a03) 和 [MHA](2026-09-interview-coding.md#coding-01)。特别说清 mask 的 bool 含义、softmax 沿 key 轴、全 mask 行的约定；不要把不同题目的 mask 语义混用。

下面用方向有限差分检查 `Y=XWᵀ+b` 的梯度。取固定上游梯度 G，定义标量 `L=sum(Y*G)`；比较数值方向导数和解析梯度内积。这是 **CPU/NumPy 教学测试**，不是训练 benchmark。

```python
import numpy as np

rng = np.random.default_rng(0)
x = rng.normal(size=(3, 4))
w = rng.normal(size=(5, 4))
b = rng.normal(size=(5,))
g = rng.normal(size=(3, 5))

def loss(x, w, b):
    return np.sum((x @ w.T + b) * g)

grads = (g @ w, g.T @ x, g.sum(axis=0))
args = [x, w, b]
eps = 1e-6
for i, grad in enumerate(grads):
    direction = rng.normal(size=args[i].shape)
    plus, minus = args.copy(), args.copy()
    plus[i] = args[i] + eps * direction
    minus[i] = args[i] - eps * direction
    numerical = (loss(*plus) - loss(*minus)) / (2 * eps)
    analytical = np.sum(grad * direction)
    np.testing.assert_allclose(numerical, analytical, rtol=1e-6, atol=1e-7)
print("Linear directional gradient checks: PASS")
```

**追问**：有限差分有截断与浮点误差，不是 eps 越小越好；随机方向测试不是穷举证明。再测 batch=1、非方阵、bias 广播和目标 shape。涉及 ReLU 的差分检查要避开不可导的零点，或明确导数约定。

↑ [返回速查](#meshy-interview-top)

<a id="meshy-f2"></a>
### F2｜现场怎样公平比较 eager 与 compile？（P0）

**直接回答**：先测正确性，再区分首次调用与稳态；CUDA 异步执行，必须用 events 或正确同步计时。固定输入/shape/dtype，预热，多轮取中位数，说明是否包含数据拷贝、分配、反向和 optimizer。没有 GPU 时只做 CPU 正确性和语法检查，不报告 GPU 加速比。

下面是 **待在 NVIDIA CUDA 环境实测**的 Python3 模板；输入已在 GPU，包含表达式产生输出的分配，不含 H2D、反向或 optimizer。首次调用包含编译/初始化等成本，不是纯 compiler 耗时。

```python
import sys
import time
import statistics
from importlib import metadata

print("Python:", sys.version.split()[0], "executable:", sys.executable)
try:
    import torch
except ImportError as exc:
    raise SystemExit("当前 python3 没有 PyTorch；先核对解释器") from exc
print("PyTorch:", torch.__version__, "built CUDA:", torch.version.cuda)
try:
    print("Triton:", metadata.version("triton"))
except metadata.PackageNotFoundError:
    print("Triton distribution: not found")
if torch.version.cuda is None or not torch.cuda.is_available():
    raise SystemExit("需要可用 NVIDIA CUDA GPU；跳过 GPU benchmark")
device = torch.device("cuda:0")
torch.cuda.set_device(device)
print("GPU:", torch.cuda.get_device_name(device))
print("capability:", torch.cuda.get_device_capability(device))
torch.manual_seed(0)

def eager(x, bias):
    return torch.relu(x + bias) * 0.5 + x

compiled = torch.compile(eager, backend="inductor", fullgraph=True)

def bench_ms(fn, x, bias, warmup=20, repeats=100, rounds=5):
    for _ in range(warmup):
        fn(x, bias)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(rounds):
        start.record()
        for _ in range(repeats):
            fn(x, bias)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / repeats)
    return statistics.median(samples)

with torch.inference_mode():
    for shape in [(1, 257), (32, 1024), (1024, 4096)]:
        x = torch.randn(shape, device=device, dtype=torch.float32)
        bias = torch.randn(shape[-1], device=device, dtype=x.dtype)
        reference = eager(x, bias)
        torch.cuda.synchronize()
        before = time.perf_counter()
        actual = compiled(x, bias)
        torch.cuda.synchronize()
        first_call_s = time.perf_counter() - before
        torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
        eager_ms = bench_ms(eager, x, bias)
        compiled_ms = bench_ms(compiled, x, bias)
        print(shape, "correctness=PASS", f"first_call={first_call_s:.3f}s",
              f"eager={eager_ms:.4f}ms compiled={compiled_ms:.4f}ms",
              f"speedup={eager_ms / compiled_ms:.2f}x")
```

**必须会解释限制**：反复用同一数据可能命中缓存；小 shape 下 events 覆盖的区间也可能包含 CPU 发射间隙。调换测试顺序、多轮测量、换 buffer 和 profiler 能帮助定位，不能单凭一次 speedup 宣称内存带宽提高。扩成训练 benchmark 时，要补 backward/optimizer 和梯度正确性。

来源：[PyTorch Benchmark](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html)、[Compile 排障](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_troubleshooting.html)。

↑ [返回速查](#meshy-interview-top)

<a id="meshy-f3"></a>
### F3｜怎样把 bias＋ReLU＋residual 写成融合 kernel？（P1）

**题目**：实现 `y=relu(x+bias)*0.5+x`，`x:[M,N]`，`bias:[N]`。先解释下面的参考实现，再遮住答案重写。它是 **FP32、连续布局、仅 forward 的教学 kernel，GPU 路径未在本地实测**，不是生产 Triton 经历。

```python
import torch
import triton
import triton.language as tl

@triton.jit
def fused_kernel(X, B, Y, N: tl.constexpr, TOTAL: tl.constexpr,
                 BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < TOTAL
    x = tl.load(X + offsets, mask=valid, other=0)
    b = tl.load(B + offsets % N, mask=valid, other=0)
    y = tl.maximum(x + b, 0.0) * 0.5 + x
    tl.store(Y + offsets, y, mask=valid)

def fused_forward(x, bias):
    if (not x.is_cuda or x.ndim != 2 or bias.ndim != 1
            or bias.device != x.device or x.dtype != torch.float32
            or bias.dtype != x.dtype or not x.is_contiguous()
            or not bias.is_contiguous() or bias.numel() != x.shape[1]):
        raise ValueError("要求同设备 CUDA、FP32、连续 [M,N] 与 [N]")
    if x.requires_grad or bias.requires_grad:
        raise ValueError("教学版本没有注册 backward")
    if x.numel() >= 2**31:
        raise ValueError("教学版本限定元素数小于 2**31，避免 int32 索引溢出")
    y = torch.empty_like(x)
    if x.numel() == 0:
        return y
    with torch.cuda.device(x.device):
        fused_kernel[(triton.cdiv(x.numel(), 256),)](
            x, bias, y, x.shape[1], x.numel(), BLOCK=256)
    return y
```

**为什么可能快**：减少中间量写回/读出和多次 launch，但 compile 也可能做同类融合，要同时比较 eager 与 compile。理想最低读写量可近似 `(2MN+N)*4` bytes，前提是 bias 被有效复用；不是实际 HBM 流量。

**测试清单**：N=1/31/32/33/255/256/257/1023/1024/1025；M=1/3/大值；负值、零、尾块、空输入；不支持的 stride/dtype 和超出 int32 索引范围的输入应明确拒绝。先限定有限输入，再约定 NaN/Inf 语义。照 F2 的正确性与计时方法测试，不预填加速结果。

**反向追问**：令 `z=x+bias`，上游梯度 g，约定 ReLU 在 0 的梯度为 0：`dx=g*(1+0.5*(z>0))`，`db=sum_rows(g*0.5*(z>0))`。训练可用还要注册 autograd、处理归约/累加 dtype，并验证与 compile 的集成，不能把 forward kernel 当作完整训练算子。

来源：[Triton Vector Add](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)、[Fused Softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)。

↑ [返回速查](#meshy-interview-top)

## G｜环境、真实项目与公司工作交流

<a id="meshy-g1"></a>
### G1｜共享屏幕配置环境，第一步做什么？（P0）

**直接回答**：先确定运行机器和解释器，再检查 PyTorch build、driver、GPU 可见性和最小 CUDA 操作，最后才处理 Triton/compile。先读报错定位层级，不立即重装全部环境。

```bash
python3 --version
python3 -m pip --version
python3 -m pip show torch triton
nvidia-smi
nvcc --version
```

命令是检查清单，不要求每台电脑都有：Mac 没有 `nvidia-smi`/CUDA 属正常情况，需远端 NVIDIA 环境。`nvidia-smi` 的 CUDA 字段主要反映 driver 可支持的 CUDA 版本范围；`torch.version.cuda` 是 PyTorch build 对应版本；`nvcc` 是本机 toolkit 编译器。三者不必完全相同，wheel 能运行也不等于能编译任意自定义扩展。

**诊断顺序**：解释器/pip 是否同环境 → driver与GPU可见性/容器权限 → CUDA wheel 与库加载 → 最小 tensor运算 → compiler/headers/ABI → Triton缓存目录和 shape/kernel。不要在正式面试前临时升级系统驱动。

**自带电脑准备**：Python3/NumPy 可离线运行；提前测试共享指定窗口、字体与终端。关闭无关通知，避免暴露公司代码、token、聊天或私人文件。工具和参考资料权限以面试说明为准，不把公开岗位“可上网”解释为可用 AI。

来源：[PyTorch 安装说明](https://pytorch.org/get-started/locally/)、[CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/)。

↑ [返回速查](#meshy-interview-top)

<a id="meshy-g2"></a>
### G2｜怎样把自己的经历对齐 Meshy，而不是只讲 RL？（P1）

**可口述开场**：

> 我本科在厦门大学、硕士在清华，主要做训练系统集成和性能优化。在华为做过 200B MoE，以及文生视频/图像模型的国产卡适配、精度和性能优化；在小鹏做长上下文 SFT 和后训练框架。我比较擅长沿数据、显存、并行和 kernel timeline 找到瓶颈，再通过配置和代码调整验证结果。这次我特别关注 3D 生成及岗位可能覆盖的视频方向中的变长数据、低精度与多阶段执行，希望把已有经验迁移过来，也在补 GPU 和数值计算细节。

| 面试切入点 | 用哪个真实项目回答 | 必须准备的细节 |
|---|---|---|
| 数据/训练效率 | [9B SFT](2026-08-llm-infra-interview-prep.md#resume-05) | 数据等待如何看；workers/prefetch；选择重算与 TP/CP；31s→9.3s 不编单项贡献 |
| 显存/源码正确性 | [CP-local logits](2026-08-llm-infra-interview-prep.md#resume-07) | 哪个 tensor 提前聚合；为什么 chunk 没救峰值；local labels/mask 与 loss 对齐 |
| Diffusion/视频适配 | [TX 视频模型](2026-08-llm-infra-interview-prep.md#resume-18) | 模型跑通、数据流、精度基线、通信/算子瓶颈、复测流程 |
| 并行与算子 | [200B MoE](2026-08-llm-infra-interview-prep.md#resume-01a) | Grouped GEMM 的 M 从哪里来；EP token 数、算通重叠、瓶颈迁移 |
| launch/调度开销 | [CUDA Graph](2026-08-llm-infra-interview-prep.md#resume-13) | 为什么 decode 受 launch 影响；shape/capture/warmup；6–8x 的局部口径 |

**三条边界**：HunyuanVideo-14B/640×640×3×129 是主文档教学示例，不冒充已确认的实际交付配置；Megatron 的特性集成不冒充底层 collective/调度开发；FP8/FP4、FSDP2/compile/Triton 若只有学习或小实验，就明确说明，讲自己的验证方法。

**跨时区协作准备**：用英语做一次 30 秒项目摘要，重点说 workload、bottleneck、change、validation，不需要编海外合作经历。比如：

> My work focuses on training-system integration and performance optimization. I usually start with an end-to-end profile, identify the dominant bottleneck, and validate both throughput and numerical correctness. My experience includes long-context SFT, MoE training, and post-training systems.

↑ [返回速查](#meshy-interview-top)

<a id="meshy-g3"></a>
### G3｜了解 Meshy 哪些公开工作，可以怎样自然聊起来？（P1）

**用法**：选一个真正理解的点，只用 20–30 秒说明观察，再问系统取舍。不需要泛泛赞美，也不要把对方尚未公开的组织安排说成你已确认的事实。

#### 1. Meshy T2：从逐 token 生成转向并行 Flow Matching

公开论文 **Meshy T2: Fast Native Mesh Generation with Flow Matching** 首发 2026-07-28，v3 更新于 2026-08-12。它以每顶点一个连续 latent 的 mesh VAE 表示几何，先用 voxel flow 生成粗占据结构，再由 mesh flow 结合图像、结构和 vertex budget 生成 mesh latent；不是逐 mesh token 自回归。[论文与作者元数据](https://arxiv.org/abs/2607.28675)

**可以这样问（工程推论，不是论文实测结论）**：

> 我看了 T2 的公开架构。它把生成变成 coarse-to-fine 的 flow，系统优化的重点会从逐 token decode 转到多步网络调用、可变 vertex 数和最终解码。我比较好奇，实际最值得优化的是去噪/flow 主干、VAE 解码，还是不同 vertex budget 下的 shape 和组批？

不背未经复现的速度数字。核验时[官方仓库](https://github.com/meshy-dev/meshy-t2)仍主要提供介绍/素材及待发布信息，不能声称已经运行其完整代码。公开研究路线也不等于公司所有线上模型。[历史补录与阅读边界](../training-infra-roadmap/tracking/backfill/2026-07.md#meshy-t2)

#### 2. MakerWorld / Bambu：从生成完成到资产可用

Meshy 2026-03-17 发布的合作消息介绍了 Image-to-3D 接入 MakerWorld/MakerLab，并将多色 3MF 接入 Bambu Studio 工作流。这能支持“生成结果需要服务实际打印流程”，**不能推出模型在打印机上推理、联合设计 GPU 或终端芯片**。[公司发布的合作消息](https://www.prnewswire.com/news-releases/how-to-turn-any-image-into-a-full-color-3d-print-in-one-click--meshys-multi-color-printing-powered-by-meshy-6-is-now-live-on-makerworld-302714800.html)

> 我关注到你们和 MakerWorld 的合作。对 ML Systems 来说，我理解指标不应只看生成速度，还要看模型能否顺利进入切片和打印流程。做低精度或推理优化时，团队会用什么几何/可打印性指标做回归门禁？下游失败会反馈到数据还是模型评估里？

这是从个人“性能＋正确性”经验出发的讨论，不假定自己懂打印机硬件。

#### 3. 视频方向：联系真实经验，再确认岗位覆盖

截至核验时，[官网视频入口](https://www.meshy.ai/video)可见 Image-to-Video/Text-to-Video 等产品选项；**仅凭入口无法确认模型是否自研、所用后端或视频团队成立时间**。用户转述的“新视频团队”保留为待面试确认的信息。

> 我也注意到官网的视频生成入口。我以前做过文生视频模型的适配和性能优化，比较熟悉从模型跑通、精度对齐到并行和算子瓶颈定位的过程。想了解这个岗位会主要服务 3D，还是也覆盖视频方向？两边在数据读取、编译、低精度和性能回归上，会更倾向共用基础设施还是先独立迭代？

**不要说**：“听说新组了视频团队，你们一定用某某 DiT”“和拓竹合作，所以一定会考端侧芯片”“T2 就是你们生产模型”。公开资料用于提出好问题，不用于填补未知事实。

↑ [返回速查](#meshy-interview-top)

<a id="meshy-g4"></a>
### G4｜和一线主管、CEO 交流，问什么有内容？（P2）

先把自己的技术问题答扎实，再从对方实际取舍判断团队，而不是准备一段“愿景赞美”。三题选一两题即可：

1. **对主管**：如果入职前三个月只能解决一个系统问题，您最希望是训练实验周转、数据供给、低精度收敛，还是在线推理成本？现在有哪些 baseline，结果由什么指标验收？
2. **对主管/CEO**：3D 与视频都在发展时，哪些基础设施值得统一，哪些应保留研究迭代自由？能否举一个最近为了长期技术质量而没有选择最快上线方案的例子？
3. **对 CEO/负责人**：在生成速度、资产可用性和研究新能力之间，公司未来一年最想建立什么优势？ML Systems 团队能参与哪些技术决策，如何从业务反馈影响研究优先级？

听具体案例、验收方式、资源与职责，不只听价值观用词。地域与协作方式是实际约束，需在进入线下流程前确认，不暗示愿意迁居。

↑ [返回速查](#meshy-interview-top)

## 验证与继续阅读

- 本页 CPU Linear 方向梯度测试可直接运行；GPU benchmark 与 Triton 部分是待 GPU 实测练习，不提供虚构结果。各项目数字沿用主文档，不将课堂推导写成个人生产成绩。
- 官方资料核验于 2026-09-09；TE/PyTorch 的具体支持矩阵和默认参数必须绑定版本。准备时先理解机制，拿到面试环境再对照版本。
- [FP8 工程入口](../training-infra-roadmap/topics/fp8.md) · [知识图谱](../training-infra-roadmap/KNOWLEDGE_GRAPH.md) · [阅读索引](../training-infra-roadmap/MASTER_READING_LIST.md) · [返回主文档](2026-08-llm-infra-interview-prep.md#meshy-interview-sprint)

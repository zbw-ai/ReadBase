# GPU Systems 核心学习路线

[所属 Part I：AI Systems 基础](../README.md) · [首页](../../README.md)

## 学习目的

从“GPU 很忙但程序不快”出发，建立执行、内存、数据复用与测量的共同模型。
完成主线后，应能解释一个 Transformer 算子的瓶颈，选择有依据的优化，并守住数值与端到端收益边界。
本路线复用已有正文，不以从零实现完整 CUDA 算子库为目标，也不要求先读完整本硬件手册。

## 先修与起点

- 能读 tensor shape，理解矩阵乘、归约和基本链式法则。
- 能用 PyTorch 检查 dtype、stride、device 与张量生命周期；不足时先读 [PyTorch 执行与布局](../topics/transformer_engine.md#pytorch-execution)。
- GPU 型号、内存与互联尚未核实时，先看 [E00 环境画像](../../practice/experiments/a100_fsdp_io_lab.md#e00)，不要套用另一架构的峰值或功能。

## 五阶段主线

“已有”仅描述正文覆盖，不代表已经实验；“局部”表示只能支持部分问题；“待补”不是一个已经存在的空章节。

| 阶段 | 核心问题 | 已有正文与覆盖状态 | 验证 / 掌握标准 |
|---|---|---|---|
| 1. 执行与内存层级 | grid/block/warp 如何执行？register、shared memory、cache、HBM 分别承担什么？ | [GPU 执行](../topics/transformer_engine.md#gpu-execution)已有机制摘要；完整访存微基准待补 | 画出一次读—算—写路径，分清 coalescing、bank conflict、spill 与 occupancy，不能把它们都叫“带宽不足” |
| 2. Tiling 与 layout | 为什么复用可以少搬数据？tile 变大为何未必更快？ | [片上复用](../topics/transformer_engine.md#gpu-execution)与 [tensor layout](../topics/transformer_engine.md#pytorch-execution)已有手算；自写 tiled kernel 实验待补 | 给定 GEMM shape/dtype，核算单 tile 的片上工作集；能沿 stride 找出额外 copy 与潜在资源压力 |
| 3. IO-aware 与 fusion | 普通算子融合和改变 IO 算法有什么不同？ | [融合算子地图](../topics/transformer_engine.md#fusion-map)已有；[FlashAttention 论文解读](../../research/papers/flashattention.md)解释机制；[topic](../topics/flashattention.md)仍是骨架 | 画出 unfused/fused 两条数据路径，标注不再 materialize 的中间量；说明 online softmax 的作用与数值对照边界 |
| 4. Streams 与 CUDA Graph | 异步提交是否真正重叠？Graph 消除了哪一段开销？ | [通信 overlap 与依赖](../topics/nccl.md)、[compile / Graph](../topics/transformer_engine.md#torch-compile)与 [decode 示例](../../04-rl-infra/topics/agentic_rl.md#cuda-graph-decode)局部覆盖；通用 stream/event 教程待补 | 画 producer → event → consumer；分开 CPU 提交、GPU 执行与等待，解释 capture/replay、动态 shape 和额外内存约束 |
| 5. Roofline 与数值验收 | 这是 compute、memory 还是 launch 瓶颈？更快是否仍然算对？ | [Roofline](../topics/transformer_engine.md#roofline)、[浮点格式](../topics/fp8.md#float-formats)与 [精度验证](../topics/fp8.md#precision-validation)已有正文 | 先估 FLOPs/bytes 再对照 trace；按层定位数值偏离；同时交付 kernel、端到端、内存与正确性证据 |

顺序是“看清执行 → 改变复用 → 减少 IO → 管理提交与依赖 → 统一验收”。Roofline 在最后汇总，不表示前四阶段可以不测量。

## 最小练习与产物

以下是建议练习，尚无新增执行结果；先做纸笔 / CPU 检查，再按授权决定是否使用 GPU。

1. **执行账本：一个 GEMM。**

   写清输入、输出、dtype、理论读写量与复用位置，指出哪些是假设、哪些需要 profiler 核验。
   产物是一页数据流图；不要只列硬件规格。

2. **布局账本：同一 tensor 的 view / transpose / copy。**

   检查 shape、stride、storage 关系，说明后续算子是否可能引入 layout conversion。
   产物是一张操作前后表；“转置本身不复制”不等于整段执行没有复制。

3. **IO 账本：Attention 与一条 elementwise 链。**

   对照 [FlashAttention 解读](../../research/papers/flashattention.md)与融合地图，分别标出算法变化和普通 fusion 的中间量节省。
   [FlashAttention benchmark](../../practice/experiments/flashattention/benchmark.md)目前只有 `NEW` 问题卡，不当作可复用结果。

4. **时间线：一次提交与重复 replay。**

   先明确输入 shape、buffer 生命周期与依赖，再拟定 eager / compile / Graph 的单变量对照。
   若进入实测，预热、编译、capture 与 steady-state 分开；使用 [E07](../../practice/experiments/a100_fsdp_io_lab.md#e07)作为推理练习入口。

5. **验收卡：同一 Linear 的两种 workload。**

   改变 batch 后重新估算算术强度，写出“预期瓶颈 → 支持证据 → 反例”。
   对精度、layout 或 fusion 的变更，先定义误差容限，再测完整调用路径，不在看到结果后放宽标准。

## 执行与覆盖边界

- [A100 实验课程](../../practice/experiments/a100_fsdp_io_lab.md)的 E00–E08 全部未执行；本路线没有申请资源或运行 GPU。
- A100 不作为原生 FP8 / FP4 Tensor Core 加速验证平台；可先学习表示、scale 和误差机制，执行路径另查实际硬件支持。
- streams/events、tiling kernel 的可运行最小实现和 bank-conflict 对照尚待补充；现有短节不能替代完整实操。
- `torch.compile`、fusion 与 CUDA Graph 是不同优化层，开关开启不证明命中预期 kernel。
- microbenchmark、profiler 窗口与端到端计时分别报告；缓存、动态 shape、fallback 和测量开销都要记录。
- 不用高 occupancy、GPU utilization 或单 kernel speedup 代替工作负载的有效吞吐与数值正确性。

## 主线之后去哪里

- 计算分片与通信成本：进入 [训练基础设施](../../02-training-infra/README.md)。
- Prefill / decode、KV 与请求调度：进入 [推理核心路线](../../03-inference-infra/roadmap.md)。
- 多进程正确性与失败恢复：进入 [分布式系统核心路线](distributed-systems.md)。

每完成一个阶段，保留“机制解释、反例、验证设计、实际证据或未执行标记”四项，再决定是否深入底层 kernel。

[返回 Part I](../README.md) · [返回首页](../../README.md)

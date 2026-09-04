# FSDP/FSDP2、ZeRO 与 Megatron 训练后端选型

> 定位：训练后端与状态分片专题。先用统一“显存账本 + 运行时数据流”理解 FSDP、ZeRO，再讨论什么时候需要 Megatron 的模型并行，以及 MBridge 为什么出现在 RL 框架里。
>
> 面试速答：[DIST-01｜FSDP/FSDP2 与 ZeRO-1/2/3](../../private_resume/2026-08-llm-infra-interview-prep.md#dist-01) · [MEGATRON-11｜Megatron 与 FSDP/FSDP2 选型](../../private_resume/2026-08-llm-infra-interview-prep.md#megatron-11) · [BRIDGE-01｜MBridge 与 Megatron Bridge](../../private_resume/2026-08-llm-infra-interview-prep.md#bridge-01)

## 1. 先建立统一坐标系

训练态显存先粗分为：

```text
模型状态 = 参数 P + 梯度 G + optimizer state O
运行时状态 = activation A + 临时 buffer B + 通信/碎片/框架开销 R
```

以 mixed-precision Adam 为例，每参数字节数取决于是否保留 FP32 master weight、梯度精度和 optimizer 实现，不能机械背一个常数。关键判断是：

- ZeRO/FSDP 主要减少数据并行 ranks 之间重复的 `P/G/O`；
- activation 不会因为 ZeRO-3 自动按 DP 等比例消失，仍要靠 micro-batch、recompute、SP/CP、packing 等处理；
- TP/PP/CP/EP 解决的是模型某个计算或容量维度，和 DP state sharding 不是同一个层次；
- offload 把状态移到 CPU/NVMe，只是换了一堵墙，会引入 PCIe、CPU 或存储带宽瓶颈。

因此，面试里先问“哪类状态超预算”，再选分片策略，而不是见到 OOM 就直接上 ZeRO-3/FSDP FULL_SHARD。

<a id="fsdp-zero-map"></a>
## 2. ZeRO-1/2/3 到底分了什么

DeepSpeed ZeRO 的三阶段是递进的状态去冗余：

| 方案 | 参数 P | 梯度 G | optimizer O | 核心收益 | 主要新增代价 |
|---|---|---|---|---|---|
| DDP / ZeRO-0 | 复制 | 复制 | 复制 | 实现直接 | 状态显存随 DP 重复 |
| ZeRO-1 | 复制 | 复制 | 分片 | 优先消除 Adam state 冗余 | update 后要让参数副本一致 |
| ZeRO-2 | 复制 | 分片 | 分片 | 进一步消除 gradient 冗余 | reduce-scatter/分片更新的数据流更复杂 |
| ZeRO-3 | 分片 | 分片 | 分片 | 完整模型状态近似按 DP size 分摊 | forward/backward 期间按需 all-gather 参数，通信和调度最重 |

一句话：**Stage 1 分 optimizer，Stage 2 再分 gradient，Stage 3 再分 parameter。**

但“分片”不表示计算时永远只有 shard。ZeRO-3 必须在某个 layer/module 计算前收集所需参数，计算后再决定何时释放；bucket、prefetch、persistence threshold、live parameter budget 和 reuse distance 会影响通信次数与峰值显存。

## 3. FSDP1 与 FSDP2 的联系和区别

### 3.1 FSDP 与 ZeRO 的关系

PyTorch 官方明确说明 FSDP 受到 ZeRO-3 启发。`FULL_SHARD` 同样分片参数、梯度和 optimizer state：

```text
pre-forward     parameter all-gather
forward         使用完整参数计算
post-forward    可选 reshard/free
pre-backward    必要时再次 all-gather
post-backward   gradient reduce-scatter + parameter reshard
optimizer step  每个 rank 更新本地 shard
```

所以可以说：**FSDP FULL_SHARD 和 ZeRO-3 在“分什么”上对应，但它们不是同一套 runtime。** DeepSpeed ZeRO 有自己的 engine、配置、offload 和 checkpoint 生态；FSDP 是 PyTorch 原生实现，module wrapping、hooks、state dict 和 composability 不同。

`SHARD_GRAD_OP` 有时被口头类比为 ZeRO-2，但要加限定：它在计算窗口保留完整参数、窗口外仍可把参数 reshard；ZeRO-2 的参数通常保持复制。两者“梯度和 optimizer state 分片”的显存目标相近，参数驻留生命周期并不完全一样。

### 3.2 FSDP1

FSDP1 的典型心智模型是 wrapper + FlatParameter：

- 用 `FullyShardedDataParallel(module)` 包装 module；
- 把一组参数 flatten 后进行分片与 collective；
- `auto_wrap_policy` 决定 FSDP unit，也就决定 all-gather/reduce-scatter 粒度；
- `FULL_SHARD`、`SHARD_GRAD_OP`、`HYBRID_SHARD` 等策略改变分片域与参数生命周期；
- `use_orig_params`、shared parameters、mixed precision 和 state dict API 会影响易用性。

它已经能生产使用，但 wrapper/flattened parameter 语义会让原参数、optimizer、checkpoint 和其他并行技术的组合更难理解。

### 3.3 FSDP2

FSDP2 的入口是 `fully_shard(module, mesh=...)`，重点变化不是“多分了一种状态”，而是表示与可组合性：

- 参数原地变成 per-parameter `DTensor`，默认在 mesh 上按 dim-0 shard；
- 不再用一个外层 wrapper 替换模型，原 parameter FQN 保持不变；
- pre-forward/pre-backward hook all-gather，post hook free/reshard，gradient 通过 reduce-scatter 回到 shard；
- bottom-up 对 Transformer block 调用 `fully_shard`，每个 FSDP unit 是一个 collective group；
- optimizer 必须在 `fully_shard` 后基于 DTensor parameters 创建；
- checkpoint 更自然地与 DTensor、DeviceMesh、Distributed Checkpoint 组合；
- 可在二维 DeviceMesh 上组合 TP 与 FSDP2，但这不代表所有模型/算子组合都天然可用。

一句话：**FSDP1 主要把 module 包进分片 runtime；FSDP2 主要把参数本身表示为 DTensor shard，并让分片成为可组合的原地变换。**

## 4. 性能为什么不只取决于“分片倍数”

### 4.1 FSDP unit 太大或太小

- 只在 root 上做一个大 unit：collective 粒度大、峰值 full parameter 高，难与 layer compute overlap；
- unit 切得过碎：collective 次数和 launch latency 增多，小消息效率差；
- 常见起点是一个 Transformer block 一个 unit，再用 profile 调整 prefetch 和 reshard。

### 4.2 `reshard_after_forward`

- `True`：forward 后立即释放 full parameter，省峰值显存；backward 前要再次 all-gather；
- `False`：把 full parameter 留到 backward，少一次 all-gather，但显存更高；
- 这是显存换通信，不存在固定最优值。

### 4.3 跨节点 collective

FULL_SHARD 会频繁 all-gather/reduce-scatter。若 DP group 跨慢网络，理论显存收益可能被通信暴露吞掉。`HYBRID_SHARD` 或 2D mesh 可以在节点内 shard、节点间 replicate，用更多显存换更少的跨节点高频通信。

### 4.4 activation 仍可能是主瓶颈

长上下文时，即使 `P/G/O` 已经分片，attention/MLP activation、logits、loss buffer 仍可能 OOM。此时继续扩大 FSDP shard size 未必有用，要联动 selective recompute、SP/CP、sequence packing 和 fused loss。

<a id="backend-selection"></a>
## 5. Megatron 与 FSDP/FSDP2 如何选

### 5.1 不是互斥概念

- FSDP/FSDP2 是 PyTorch 数据并行状态分片实现；
- Megatron-Core 是面向大 Transformer 的训练/模型并行栈，提供 TP、PP、CP、EP、SP、distributed optimizer、pipeline schedule、fused kernels 和 distributed checkpoint；
- 大系统可以把模型并行与 sharded DP 组合；现代栈也存在 FSDP2 + TP 的二维组合。

面试时不要说“Megatron 是 TP，FSDP 是 DP，所以二选一”。真正选择的是整套模型定义、并行能力、kernel、checkpoint 和团队运维生态。

### 5.2 决策顺序

```text
目标模型与单层是否能放下
  → 单卡/单机能否容纳训练状态
  → 是否需要 TP/PP/CP/EP
  → 高频通信能否放进 NVLink 域
  → 目标模型/精度/kernel/ckpt 是否成熟
  → 团队调试和升级成本
```

### 5.3 典型选择

| 场景 | 更合理的起点 | 原因 |
|---|---|---|
| 中小模型 SFT/RL、模型代码变化快 | FSDP2 | PyTorch 原生、HF 生态接入自然、per-parameter DTensor 和 DCP 易组合 |
| 模型状态超单卡，但单层和序列仍可处理 | FSDP2/ZeRO-3 | 主要矛盾是 DP 状态冗余，暂不需要重模型并行 |
| 超大 dense/MoE、长上下文，需要 TP/PP/CP/EP | Megatron-Core | 多维模型并行、MoE/长上下文和 fused kernel 是主矛盾 |
| 已有成熟 Megatron 模型、配置、ckpt 和排障体系 | Megatron-Core | 迁移成本和生产确定性通常比理论易用性更重要 |
| 快速适配大量 HF 新模型 | FSDP2 优先 PoC | 模型改造半径通常更小，但仍要验证 rollout/weight sync 与长稳 |
| 既要 TP 又希望 PyTorch-native sharded DP | FSDP2 + TP PoC | 技术上可组合；支持矩阵、checkpoint 和算子正确性必须实测 |

### 5.4 项目口径

本项目对 Megatron 的 ownership 是 **feature integration/application layer**：使用和集成 5D 并行、distributed optimizer、MBridge/Megatron Bridge、checkpoint 与 RL backend，做配置、模型适配、性能/正确性优化和交付。没有实现底层 collective kernel，没有修改 `parallel_state`/process-group construction，也没有编写 pipeline scheduler。

<a id="bridge-layer"></a>
## 6. MBridge 是什么

### 6.1 它解决的不是“并行”，而是“语义翻译”

Hugging Face 和 Megatron-Core 对同一模型的描述并不相同：

- config 字段和模型类不同；
- QKV、MLP、MoE expert 等参数命名与布局不同；
- Megatron 权重可能按 TP/PP/EP/VPP 分布；
- rollout 或交付通常又需要 HF/inference-engine 可识别的布局。

Bridge 的职责是：

```text
HF config/model/checkpoint
  ↔ config mapping + model provider + parameter mapping
Megatron-Core distributed model/checkpoint
```

这包括按目标并行布局加载 HF 权重、构建 Megatron model、把 Megatron 分布式权重导回 HF，以及做转换正确性验证。它不是 TP/PP 调度器，也不替代 Megatron-Core。

### 6.2 `mbridge` 与 NVIDIA `megatron-bridge`

二者是不同 package，不是同一个库的两种拼写：

| 项目 | `mbridge` | NVIDIA `megatron-bridge` |
|---|---|---|
| 定位 | 原型/社区先行实现 | NVIDIA 维护的官方后续实现 |
| 核心能力 | HF ↔ Megatron-Core config/权重转换，覆盖 RL 常用 export/import | conversion + verification，并扩展训练 recipe、模型与 PEFT/LoRA |
| 当前方向 | 保留兼容和已有工作流，不再主推高级能力 | 新能力和更广模型支持的主要演进方向 |
| 本地锁定版本 | `0.15.1` | `0.3.0` |

`mbridge` 项目自身说明其思想已被 Megatron Bridge 采用，并建议高级能力转向官方实现。不能把这个关系说成“改名”，因为两者包名、API、版本和支持矩阵都不同。

### 6.3 为什么本地 AReaL 同时保留

本地代码通过 `actor.megatron.bridge_type` 选择：

```text
bridge_type=mbridge
  → mbridge.AutoBridge.from_pretrained
  → 本地 native/fast HF load-save 路径

bridge_type=megatron-bridge
  → megatron.bridge.AutoBridge.from_hf_pretrained
  → provider/model + native HF load-save + PEFT/LoRA 路径
```

保留两条路径的工程原因是：

- 老 checkpoint、模型适配和部署路径仍依赖 `mbridge`；
- 本地 disk-based weight broadcast 对现有 fast HF I/O 路径有现实依赖；
- 本地 tree-attention training 当前只支持 `mbridge`；
- 新增 GPU workflow 若不受上述限制，可优先评估 NVIDIA Megatron Bridge；
- 使用 XCCL 直接 weight broadcast 时，HF disk I/O 的权重相对下降。

这是本地锁定版本的选型结论，不应外推成所有项目、所有新版 Megatron Bridge 的固定规则。

## 7. 训练后端选型的验证清单

### FUNCTIONAL

- 目标 dense/MoE/VLM 模型可构建；
- SFT/RL forward、backward、optimizer step、save/load 可闭环；
- 目标 TP/PP/CP/EP/FSDP mesh 合法；
- rollout weight export/refit 能工作。

### NUMERIC

- 单卡 HF、FSDP2、Megatron same-weight logits/logprob 对齐；
- TP/EP gather 后参数与 HF checkpoint 对齐；
- loss mask、gradient scale、optimizer state 和 resume step 对齐；
- bridge round-trip 不丢参数、不静默 transpose/reshape。

### PERFORMANCE

- peak/reserved memory 按 `P/G/O/A/B/R` 分账；
- all-gather/reduce-scatter/all-to-all exposed time；
- MFU、tokens/s/GPU、data stall、checkpoint 与 weight-sync 时间；
- 不只比较能跑的最大 batch，还要固定 global token batch。

### RECOVERY

- 相同并行度 resume；
- 允许时验证 DP world size 变化；
- checkpoint 不完整、某 rank 失败时 fail-consistent；
- RL 系统还要恢复 policy version、queue/data cursor 和 rollout 权重版本。

## 8. 高频追问与危险回答

### 为什么 FSDP FULL_SHARD 和 ZeRO-3 不能直接画等号？

分片对象相同，但参数表示、module/runtime hooks、bucket/prefetch、offload、checkpoint 和框架集成不同。正确说法是“在状态分片层次对应”，不是“实现等价”。

### FSDP2 为什么更容易和 TP 组合？

参数显式表示为 DTensor，并通过 DeviceMesh 描述 placement；但这只提供组合基础，模型算子、2D layout、checkpoint 和性能仍需验证。

### 参数都分片了，计算时为什么还能做 GEMM？

FSDP/ZeRO-3 在 module 计算前 all-gather 当前工作集的 full parameter，完成后再 reshard；省的是长期驻留，不是让普通 GEMM 直接消费任意 parameter shard。

### 危险回答

- “FSDP 就是 PyTorch 版 ZeRO-3”；
- “SHARD_GRAD_OP 完全等于 ZeRO-2”；
- “上 FSDP 后 activation 也按 DP size 分掉”；
- “Megatron 和 FSDP 只能二选一”；
- “模型大就用 Megatron，模型小就用 FSDP”，但不谈 layer、sequence、MoE、拓扑和团队资产；
- 把 MBridge 说成并行策略，或说 `mbridge` 只是 `megatron-bridge` 的旧名字；
- 把使用 bridge 和 Megatron feature integration 说成自己实现了 Megatron 底层。

## 9. 相邻主题

- [verl 与 AReaL：RL 框架架构选型](rl_framework_selection.md)：Bridge 如何进入训练后端、rollout backend 和 weight-update 决策。
- [Agentic RL Infrastructure](agentic_rl.md)：训练态/推理态模型、policy version、staleness 与外部 Agent 数据流。
- [Megatron 5D Parallelism](distributed_training.md)：TP/PP/CP/EP 与分片 DP 如何组合。

## 10. 官方来源

- [PyTorch FSDP2 `fully_shard`](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html)：per-parameter DTensor、all-gather/reduce-scatter、bottom-up grouping 与 state dict 语义。
- [PyTorch FSDP1](https://docs.pytorch.org/docs/stable/fsdp.html)：`FULL_SHARD`、`SHARD_GRAD_OP`、`HYBRID_SHARD` 与 wrap policy。
- [DeepSpeed ZeRO Tutorial](https://www.deepspeed.ai/tutorials/zero/)：ZeRO-1/2/3 的官方分片定义和 offload。
- [ZeRO 论文](https://arxiv.org/abs/1910.02054)：状态冗余与通信/显存动机。
- [Megatron-LM 大规模训练论文](https://arxiv.org/abs/2104.04473)：模型并行与大规模训练系统背景。
- [`mbridge` PyPI 项目说明](https://pypi.org/project/mbridge/)：原型定位、HF/Megatron-Core 互操作与迁移方向。
- [NVIDIA Megatron Bridge](https://docs.nvidia.com/nemo/megatron-bridge/latest/)：官方 conversion、verification、training 与 PEFT/LoRA 能力。

## 11. 我的总结

FSDP、ZeRO 和 Megatron 的选择不是框架名投票，而是显存与通信问题分解：先判断 `P/G/O/A` 谁超预算，再判断是否需要拆单层、拆深度、拆序列或拆 expert，最后把高频通信放到合适拓扑。

MBridge 出现在 RL 系统中，是因为训练态 Megatron 和 rollout/交付态 HF layout 之间存在真实的 config 与权重语义鸿沟。能把这条转换链的正确性、性能和版本边界讲清楚，比只会背 API 更能体现训练 Infra ownership。

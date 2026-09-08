# Long-context Training

> 定位：长上下文训练工程章节，覆盖选择性重计算、LLM/MoE SFT、CP-local loss/logprob、视频 DiT Ulysses、配置选择与生产排障。
>
> 面试速答入口：[RESUME-05｜9B SFT 31s→9.3s](../../private_resume/2026-08-llm-infra-interview-prep.md#resume-05) · [RESUME-17｜35B-A3B 128K](../../private_resume/2026-08-llm-infra-interview-prep.md#resume-17) · [RESUME-07｜7.6GB CP-local logits](../../private_resume/2026-08-llm-infra-interview-prep.md#resume-07) · [RESUME-18｜视频 DiT/Ulysses](../../private_resume/2026-08-llm-infra-interview-prep.md#resume-18)

长上下文训练不是单一技术点，也不等于 Context Parallelism。

它是一条横跨 **Pretraining / SFT / RL** 的系统主线：序列长度变长以后，训练对象、数据分布、attention kernel、activation、checkpoint、rollout、reward/verifier、推理引擎和集群通信都会一起变化。

这个专题页先作为地图使用，帮助后续阅读和实验归位。

## 核心问题

短上下文训练中，主要矛盾通常是模型参数、optimizer state 和集群吞吐。长上下文训练把主要矛盾推向 token 维度：

- 单条样本可能无法放进一张 GPU。
- attention / hybrid attention / linear attention 的 kernel 行为会随 sequence length 出现性能悬崖。
- activation 和中间状态随长度放大，recompute、offload、checkpoint 需要重新取舍。
- 变长样本会造成 packing、load balance 和 step time 抖动。
- RL 阶段还会把长 prompt、长 response、KV cache、verifier 成本和 policy version 同步带进训练系统。

工程上不要问“能不能支持 128k”，而要问：

1. 数据里真实长样本占比是多少？
2. 每张卡实际处理多少 token？
3. 长样本是拖慢所有 step，还是被 packing 隔离？
4. checkpoint 和恢复是否能承受更长 step time？
5. RL rollout 的推理栈是否跟训练栈共享同一套 tokenizer、position scaling 和 policy version？

## 一套统一的优化顺序

长上下文优化最怕“先开遍所有性能开关”。更可靠的顺序是：

1. **固定 workload**：checkpoint、硬件、并行度、length/packing 分布、GBS/MBS、有效 token、精度、warmup 和统计窗口。
2. **建立显存账**：模型状态、saved activation、logits/loss、collective buffer、kernel workspace、graph pool 与 checkpoint 临时副本分开记。
3. **确认数据布局**：每个关键 tensor 是否真的按 TP/CP/EP 分片，THD/BSHD、padding、zigzag 与 loss mask 是否一致；先修静默回退和全量 materialization。
4. **拆 critical path**：data wait、attention、MLP/expert、TP/CP/EP communication、loss/logprob、backward、optimizer 与 checkpoint。
5. **再做联合搜索**：input pipeline → 并行策略 → recompute/offload → fusion/kernel → overlap。每解掉一个瓶颈都重新 profile。
6. **性能与正确性一起验收**：平均/p95 step、有效 tokens/s、peak allocated/reserved、loss/logprob/grad、save/resume 和长窗口稳定性。

<a id="selective-recompute"></a>
## 选择性重计算：哪些 activation 值得少存、多算？

**先记一句话**：前向不长期保存选中模块的部分中间结果，反向需要时再算出来；优先选“确实占据显存峰值、重建代价又低”的部分，而不是把整层都重跑。

**阅读顺序**：[原理](#recompute-mechanism) → [参数](#recompute-config) → [选哪些结构](#recompute-modules) → [配置示例](#recompute-recipes) → [调优与排障](#recompute-tuning)。面试现场看[主文档速答](../../private_resume/2026-08-llm-infra-interview-prep.md#megatron-selective-recompute)。

版本边界：参数与约束以 **Megatron-Core 0.17.0 / Megatron-LM `core_r0.17.0`** 为基准，核验于 2026-09-08。下面是通用机制与候选配置，**不是个人项目最终配置的补录**；通过 verl 等上层框架使用时，还要确认参数确实传给了 MCore。

<a id="recompute-mechanism"></a>
### 1. 为什么反向需要前向的 activation？

以 `Y = XW` 为例，计算 `dW = XᵀdY` 需要前向输入 `X`；激活函数的反向也需要相应输入或输出。所以正常训练不只是保存最后的输出，还会保留许多中间 tensor，直到对应 backward 用完。

例如一个 MLP 是 `X → FC1 → activation → FC2 → Y`：

- **正常执行**：autograd 保存反向所需的中间状态，计算开销小，但这些 tensor 会跨越较长的 forward/backward 间隔。
- **整个 MLP checkpoint**：主要保留区域输入等边界状态，不长期保存区域内部的全部中间状态；backward 到来时重新执行所需前向，再求梯度。代价包含大 GEMM，若边界内有通信也可能重放。
- **只重算 activation**：保留 FC1 的输出作为重建输入，在合适时机释放激活函数输出；在 FC2 backward 需要它前恢复。重算边界不包含 FC1/FC2，成本更小，但省下的显存也更有限。

它省的是 **saved activation**，不会自动减少参数、梯度、Adam 状态，也不是把模型 checkpoint 写到磁盘。如果 OOM 的根因是 optimizer 初始化或额外 full logits，重算 Transformer 中间层未必对症。

**为什么 MCore 还需要 output-discarding checkpoint？** 普通 checkpoint 不保存区域内部状态，但区域输出可能仍被下游 backward 保存。例如 `activation` 的输出是 FC2 的输入，FC2 算权重梯度时还要用它。MCore 的 `CheckpointWithoutOutput` 会在下游前向使用完后释放该输出的 storage，并挂 hook，在下游 backward 消费前重算恢复。它不是简单 `del tensor`：autograd 或 view 仍可能持有同一块 storage。具体输入保存、RNG 恢复与 hook 生命周期由框架管理，不建议在业务层手工删除 tensor 模仿。[MCore checkpoint 实现](https://github.com/NVIDIA/Megatron-LM/blob/core_r0.17.0/megatron/core/tensor_parallel/random.py)

一个只用于建立量级感的例子：BF16 的本地 tensor `[16384, 1, 4096]` 占 `128 MiB`。如果重算确实让它不再跨越峰值窗口，才可能减少这部分峰值；如果别的消费者还持有它，或真实峰值发生在 optimizer 阶段，就不能直接计为 `128 MiB` 的整步收益。多层、多在途 microbatch 还会放大保存量，但必须按实际生命周期去重。

<a id="recompute-config"></a>
### 2. 参数：selective 选模块，full 再决定怎样选层

| MCore 字段 / Megatron-LM CLI | 含义 | 一般怎么设 |
|---|---|---|
| `recompute_granularity` / `--recompute-granularity` | `None` 不额外启用这一机制；`selective` 重算选定子模块；`full` 重算整层范围 | 能放下先测无额外重算基线；不够再试 selective，最后比较 full |
| `recompute_modules` / `--recompute-modules` | selective 的模块种类列表；默认 `core_attn`，不是层号列表 | 按实际模型选择；会作用于各层中存在且实现支持的对应模块 |
| `recompute_method` / `--recompute-method` | full 的分层方法：`uniform` 或 `block` | selective 不设置；full 必须明确设置 |
| `recompute_num_layers` / `--recompute-num-layers` | uniform 的每组层数，或 block 的重算层数 | selective 必须为 `None`；full 根据本地层数设置 |
| `distribute_saved_activations` / `--distribute-saved-activations` | 将 checkpoint 保存的第一个输入 hidden states 沿 TP 分片，重算前 gather 恢复 | 默认关；0.17 CLI 要求 TP>1、full，且不能与 SP 同开 |

最后一个参数不是“分摊重算任务”，也不把全部 activation 再除以 TP。若已经启用 `sequence_parallel`，不要继续套用这条节省公式。[0.17 配置文档](https://docs.nvidia.com/megatron-core/developer-guide/0.17.0/apidocs/core/core.transformer.transformer_config.html)、[CLI 校验](https://github.com/NVIDIA/Megatron-LM/blob/core_r0.17.0/megatron/training/arguments.py)、[输入分片实现](https://github.com/NVIDIA/Megatron-LM/blob/core_r0.17.0/megatron/core/tensor_parallel/random.py)

**最容易考错的是 `full + uniform` 与 `full + block`。** 假设当前 model chunk 有 8 层：

- `full + uniform + num_layers=2`：分成 `[1,2] [3,4] [5,6] [7,8]` 四个 checkpoint 区域，8 层全部参与重算；不是“只重算 2 层”。增大组长减少长期保存的组间边界，但重建期间的临时峰值也可能上升，不能认为越大越省、越快。
- `full + block + num_layers=2`：通常只对本地前 2 层分别做整层 checkpoint，其余 6 层正常执行。这里的 full 指**被选中层的重算范围**，不意味着所有层都重算。
- 有 PP/VPP 时按**当前 model chunk 的本地层数**理解，不按全模型层数，也不能把一个物理 PP rank 上所有 VPP chunks 混算。0.17 的 uniform 实现支持最后不足一组的尾部，不要求层数整除；FP8/FP4 路径还可能因输入梯度条件调整 block 的起点。[TransformerBlock 实现](https://github.com/NVIDIA/Megatron-LM/blob/core_r0.17.0/megatron/core/transformer/transformer_block.py)

<a id="recompute-modules"></a>
### 3. 哪些结构适合重算？先看实际保存边界

以下七项是 0.17 的模块名。表中“优先比较”是候选实验顺序，不是跨模型通用的性能排名。

| 模块名 | 实际重算范围 | 适用结构与取舍 |
|---|---|---|
| `core_attn` | core attention；不包括普通 Attention 的 QKV 和输出投影 GEMM | 非 Flash 路径可优先比较；Flash/TE fused attention 下额外收益可能很小，长序列和 CP 又可能增加重放成本 |
| `layernorm` | 独立的 `input_layernorm`、`pre_mlp_layernorm`，采用 output-discarding | Norm 的重建相对便宜；但若已融合进 LayerNormLinear、外层变成 `IdentityOp`，这个外层边界会跳过 |
| `moe_act` | expert MLP 中的激活函数，采用 output-discarding | grouped MoE 常用候选；不重跑 FC1/FC2，也不是丢掉全部 expert expansion |
| `mla_up_proj` | MLA up projection 与 RoPE 部分，采用 output-discarding | 对比保存低维输入并重建展开 Q/KV 的收益；只适用于 MLA，不能套到普通 GQA |
| `mlp` | 整个 Dense MLP，普通 checkpoint | Dense 的扩展维 activation 若是峰值主因可试；需要付出两次大 Linear 的前向重算代价 |
| `moe` | 整个 MoE 层，普通 checkpoint | 显存很紧时比较；可能重放 router、dispatch/combine 与 expert GEMM，通信成本不能漏算 |
| `shared_experts` | shared expert 子模块，普通 checkpoint | shared expert 保存量占比高时评估；还需检查与 shared-expert overlap 的兼容性 |

模块名及类型见[0.17 配置文档](https://docs.nvidia.com/megatron-core/developer-guide/0.17.0/apidocs/core/core.transformer.transformer_config.html)；边界分别可核对 [Attention](https://github.com/NVIDIA/Megatron-LM/blob/core_r0.17.0/megatron/core/transformer/attention.py)、[TransformerLayer](https://github.com/NVIDIA/Megatron-LM/blob/core_r0.17.0/megatron/core/transformer/transformer_layer.py)、[Experts](https://github.com/NVIDIA/Megatron-LM/blob/core_r0.17.0/megatron/core/transformer/moe/experts.py) 与 [MoELayer](https://github.com/NVIDIA/Megatron-LM/blob/core_r0.17.0/megatron/core/transformer/moe/moe_layer.py)。

**按模型结构记四种情况：**

1. **Dense / GQA + FlashAttention**：先比较无额外重算和 `core_attn`，同时查独立 norm、MLP 的真实保存量。FlashAttention 已避免保存完整 `S×S` 矩阵，并在 backward 内部分块重建相关量；外层 `core_attn` checkpoint 则再次调用 core-attention forward。两者不是同一层次，不能沿用未融合 Attention 的显存节省预期。[TE Attention 机制](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/attention/attention.html)
2. **Grouped MoE**：先比较 `moe_act`，再看是否有独立 `layernorm` 可省。TEGroupedMLP 的 `moe_act` 保留 FC1 输出，只重建 activation 输出；FC1/FC2 仍正常执行。若启用整个 `moe`，不要把 `moe_act` 的收益再累加一次。[Grouped expert 实现](https://github.com/NVIDIA/Megatron-LM/blob/core_r0.17.0/megatron/core/transformer/moe/experts.py)
3. **MLA**：额外评估 `mla_up_proj`，看展开后的 Q/KV、RoPE 中间结果是否在峰值存活。不要只因模型是 MoE 就认为它使用 MLA；Attention 结构和 expert 结构是两件事。
4. **长上下文 / Hybrid 模型**：区分 full attention、linear attention/GatedDeltaNet 等实际层类型；CP/SP 已改变本地 shape，重算要基于切分后的峰值重新选。新版本存在 `gdn_norm_out` 等扩展，但不属于这里的 0.17 七项；既不能照抄给旧版本，也不能拿 `core_attn` 解释所有混合层。[当前配置 API](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.transformer.transformer_config.html)

这些候选顺序也与 NVIDIA 的[重算调优指南](https://docs.nvidia.com/nemo/megatron-bridge/nightly/training/activation-recomputation.html)相符；该页是滚动 nightly 文档，其建议不代表 0.17 支持所有新组合。

<a id="recompute-recipes"></a>
### 4. 常见配置：作为 A/B 起点，不是万能模板

下面均为 Megatron-LM CLI **参数片段**，追加到已跑通的训练命令；不是完整启动脚本，也不是某次个人项目的最终参数。无额外重算的基线应移除这组 recompute 参数，使 `recompute_granularity=None`，不要传字符串 `none`。

**标准 core-attention selective 对照：**

```bash
--recompute-granularity selective \
--recompute-modules core_attn
```

**Grouped MoE 的低成本边界候选**：先单测 `moe_act`，再比较添加 `layernorm`。前提是 grouped expert 路径支持该实现，并且实际存在可重算的独立 norm。

```bash
--moe-grouped-gemm \
--recompute-granularity selective \
--recompute-modules moe_act layernorm
```

**Selective 仍不够时，比较部分整层重算**：此例假设当前 chunk 至少 2 层；逐步增加层数，找到满足 headroom 的最小范围。

```bash
--recompute-granularity full \
--recompute-method block \
--recompute-num-layers 2
```

**若需要所有层按组重算**：此例每 2 层构成一个 checkpoint 区域；和上例二选一比较，不是叠加。

```bash
--recompute-granularity full \
--recompute-method uniform \
--recompute-num-layers 2
```

**配置门禁，0.17 尤其注意：**

- 显式设置 `recompute_modules` 是替换默认列表，不是自动追加。例如只写 `moe_act layernorm`，不会自动保留 `core_attn`；查看最终有效配置确认实际列表。
- selective 不配 `recompute_method/recompute_num_layers`；它们是 full 的控制项。切换 recipe 时清理旧配置，避免上层 merge 后残留。
- `moe_act` 要求 `moe_grouped_gemm=True`；`mla_up_proj` 要求实际启用 MLA。
- `shared_experts` 重算与启用 shared experts 的 `moe_shared_expert_overlap=True` 冲突；别同时开启后只看最终是否 OOM。
- FP8 下 `moe_act/layernorm` 有额外 TE 版本和 recipe 约束：0.17 校验要求 TE ≥ `2.6.0dev0`，且不支持 delayed scaling。BF16 上有效不能证明切到 FP8 也兼容。
- `moe` 包含 `moe_act` 的计算区域，默认不要嵌套叠加；这不是声称所有版本必报错，而是避免未经验证的重复重算与生命周期问题。`mlp` 与 `moe` 则可能分别服务混合模型里的 Dense/MoE 层。
- 不直接复制 HF 的 `gradient_checkpointing_enable()` 或 PyTorch checkpoint 参数给 MCore；它们的配置入口和 checkpoint 实现不同。旧 `moe_layer_recompute` 迁移后使用合法模块名 `moe`，不是 `moe_layer`。[配置校验与迁移源码](https://github.com/NVIDIA/Megatron-LM/blob/core_r0.17.0/megatron/core/transformer/transformer_config.py)

<a id="recompute-tuning"></a>
### 5. 工程上怎么选：先过显存线，再选吞吐最好的配置

选择准则不是“重算越多越好”，而是：**在正确性和最坏样本显存余量都满足的候选中，选择有效吞吐最高的配置。**

```text
先看：ΔM_peak = 基线整步峰值 − 候选整步峰值
再看：ΔT_step = 候选稳态 step time − 基线稳态 step time
单位时间换得的显存 ≈ ΔM_peak / ΔT_step     （ΔT_step > 0 时）
```

`saved bytes / recompute FLOPs` 可用于筛选，但最终用 wall time：重算会改变通信、kernel 融合、调度和 overlap，不只有 FLOPs。这个比值也不是唯一优化目标；释放显存后可能允许更大 MBS、更少 microbatch 或更合适的并行配置，需要另外测端到端收益。

**建议的最小实验流程：**

1. **定位峰值**：先排除 full logits、dtype upcast 和异常常驻副本；确认问题确实是 saved activation，而不是参数或 optimizer。记录哪个 rank、PP/VPP chunk、哪个阶段最重。
2. **固定 workload**：保持模型、tokens/length 分布、packing、精度、GBS/MBS、TP/CP/EP/PP、kernel/backend 和统计窗口一致。原配置无法完整跑通时，可先缩小 workload 找候选，但最终必须回到目标 workload 验证。
3. **窄边界逐项 A/B**：按结构测试单模块，再测试组合。每次记录有效配置、peak allocated/reserved、设备总占用、step median/p95 和额外 backward 前向/collective 时间。
4. **不够再扩大边界**：按峰值来源比较整个 `mlp`、整个 `moe` 或 `full + block`；不是每个模型都要把这些依次开一遍。仍不够时比较 full uniform、MBS、CP/SP 或 activation offload。
5. **确认稳态和最坏情况**：完成 optimizer state 的初始化，覆盖完整 forward/backward/梯度同步/optimizer，以及长样本、PP 在途 microbatch 和 MoE 热 rank。不要只看 forward 结束后的 `allocated`。
6. **验正确性再联合调优**：对照 loss、梯度、短窗口收敛；检查 dropout/RNG、routing 副作用和 FP8 scale/amax 状态。重放若走了不同分支或重复更新状态，可能产生静默梯度问题。随后才调整 MBS/并行度，单独报告其联合收益。[PyTorch checkpoint 的重放一致性说明](https://docs.pytorch.org/docs/2.9/checkpoint.html)

**显存余量如何考虑？** 根据最长样本、动态 routing、临时 workspace 与运行波动留空间，不用一个固定百分比套所有任务。降低 MBS 后若调整梯度累积保持 GBS，要同时观察 GEMM 效率和 PP bubble。启用 CUDA Graph、切 FP8、换 TP/CP 或更换 TE kernel 后，都需要重新做上述对照，旧最优值不保证仍然最优。

| 现象 | 优先核查 | 下一步 |
|---|---|---|
| 开了 `layernorm`，显存几乎没变 | norm 是否融合、外层是否 `IdentityOp`，输出是否位于真正峰值 | 查看实际 module spec 和 snapshot，不凭配置名判定生效 |
| 开 `core_attn` 后变慢、省得很少 | 已用 Flash/TE fused attention？是否重放 CP 通信？ | 对照关闭外层重算，转查 MLP/expert activation |
| 开整个 `moe` 后 A2A 增多、吞吐下降 | checkpoint 是否覆盖 dispatch/combine；是否与子边界重复 | 比较 `moe_act` 等窄边界，或重新权衡通信与显存 |
| 重算后仍在同一处 OOM | 峰值是否来自 optimizer、logits、graph pool、重建临时张量 | 回到每-rank 显存账，不盲目追加模块 |
| loss/grad 偏离或恢复后异常 | RNG、状态更新副作用、FP8 recipe、重放控制流 | 缩小到单模块对照，验证正确性后再扩大范围 |

### 6. 面试追问与项目表达

**为什么 full recompute 慢？** 因为被 checkpoint 的区域多执行了前向。简化地，原训练若是 `F+B≈3F`，完整重跑一次模型前向会变成约 `4F`，即计算量约增加三分之一；这只是不含通信/IO等的估算，不能说 step time 必然增加 33%。Selectively checkpoint 的只是部分区域，代价取决于边界。

**为什么你从 full 改成 selective，反而训练更快？**

> 重算本身是用计算换显存，不是天然加速。我原来采用了偏保守的整层重算；联合优化后，在满足显存余量的前提下缩小重算范围，就能减少重复前向。选择时看实际峰值和重建代价，再比较 step time、显存、loss 和梯度。具体重算哪些模块需要回查当时配置，不能拿今天框架支持的参数冒充项目事实。

在个人 [9B SFT 项目](#qwen35-9b-sft)里，`31s→9.3s` 是数据供给、重算和并行配置的联合结果，不把全部加速归给 selective。常见错误回答包括“只重算最贵的 Attention”“selective 就是选几层”“FlashAttention 开了再 checkpoint 一定更省”“输出多大就一定能省多少显存”。

**相关机制**：[FlashAttention](flashattention.md) · [Transformer Engine / Fusion](transformer_engine.md#fusion-map) · [CP](context_parallelism.md) · [SP](sequence_parallelism.md) · [MoE](moe.md)。

↩ [返回主文档：选择性重计算](../../private_resume/2026-08-llm-infra-interview-prep.md#megatron-selective-recompute) · [返回 SFT 项目回答](../../private_resume/2026-08-llm-infra-interview-prep.md#resume-05) · [知识图谱](../KNOWLEDGE_GRAPH.md) · [阅读总索引](../MASTER_READING_LIST.md)

<a id="qwen35-9b-sft"></a>
## Qwen3.5-9B SFT：31s → 9.3s 的工程解释

最新版简历记录的是联合结果：step time `31s→9.3s`，MFU `23%→45.2%`。可确认的优化方向是 DataLoader 并发与 prefetch、selective recompute、TP/CP 收敛；当前没有可公开的逐项同-workload 消融，不能给三项硬拆收益。

### 1. 先消除 input bubble

`num_workers=0` 表示 DataLoader 不使用 worker 子进程，并不单凭这个值就能断定 GPU 在等数据。先看 CPU 数据准备、H2D 与 GPU 计算的实际 timeline；若 GPU 空洞与 `next(data_iter)` 对齐，再逐步验证以下候选。当前项目只确认 worker `0→8` 与 prefetch，不代表其余配置都已在这次 benchmark 中启用：

- 增大 `num_workers`，项目底稿记录的方向是 `0→8`；
- `pin_memory` + non-blocking H2D；
- `persistent_workers` 避免 epoch/iterator 重建；
- 合理的 `prefetch_factor`，让 CPU 准备第 `n+1` 批时 GPU 计算第 `n` 批；
- tokenizer/packing 缓存、连续存储和减少小文件 metadata IO；
- 监控 CPU utilization、RSS、page fault、queue depth 和 data wait p95，防止 worker 过多反而争抢 CPU/内存。

这类优化不提高 GPU 峰值算力，而是减少 GPU 暴露的等待时间。CPU 预取下一批与 GPU 计算重叠，不等于 H2D copy 与 kernel 已经重叠；后者还需要 pinned memory、合适的独立 CUDA stream、可用 DMA engine 和正确依赖，不能只看 `non_blocking=True`。[PyTorch 官方说明](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)

### 2. 从 full recompute 收敛到 selective recompute

这一步的逻辑是：在显存仍能容纳目标 workload 的前提下，缩小 checkpoint 范围，减少重复前向。先看哪些保存到 backward 的 tensor 占据实际峰值，再比较无额外重算、selective 和 full 的显存、step time 与数值结果；重建成本包含计算及可能重放的通信。

`core_attn` 是 MCore selective 的默认候选，但在 Flash/TE fused attention 下不保证最划算；需按真实层结构、fusion 与 TP/CP 布局选择。完整原理、模块表、参数和 A/B 顺序见[选择性重计算](#selective-recompute)。项目口径只确认从偏重 recompute 收敛到 selective；当时精确 `recompute_modules` 必须以配置为准，不将通用示例写成项目历史。

### 3. 让 TP 解决权重，让 CP 解决长序列

9B 模型上 TP 过大可能让 GEMM 的 M/N/K 变小，并引入逐层高频 collective。128K 的一阶压力更多来自 sequence activation，因此在参数能够放下时，应比较“更小 TP、更大 CP”与原配置，而不是把所有卡给 TP。

比较时至少同时看：

- GEMM/FlashAttention kernel efficiency；
- TP all-reduce/all-gather/reduce-scatter exposed time；
- CP KV exchange/attention communication；
- 每 rank local sequence、activation peak 与 microbatch 数；
- 跨节点 group mapping。

另一 workload 的 `TP=4,CP=4 → TP=2,CP=8、163s→102s` 只能证明这种选择机制，不能作为 31s→9.3s 的消融项。

### 4. MFU 为什么必须做算术校验

若标准 MFU 使用相同的模型 FLOPs/step、有效 token 和 wall-clock step time，则应近似满足：

```text
MFU_new / MFU_old ≈ step_time_old / step_time_new
```

但 `31/9.3≈3.33`，`45.2/23≈1.97`，两者不能自动闭合。这不代表最新简历数字一定错误，但说明可能存在 estimator、有效 token、packing、data wait 是否计入或统计窗口差异。面试前应带上原始 MFU 公式和日志；补齐前分别陈述数字，不说它们来自完全相同的单一测量窗口。

<a id="qwen35-35b-a3b-128k"></a>
## Qwen3.5-35B-A3B 128K：为什么 active 3B 仍然难

MoE 的 active 参数决定单 token 的部分 FLOPs，但不能把整套系统当作 3B dense：

- 总参数、optimizer/main-weight 和 checkpoint 仍需按实际 sharding 放置；
- router、token dispatch/combine 与 Grouped GEMM 增加动态负载和 All-to-All；
- 128K 放大 attention/activation、CP communication 与 logits/loss 临时张量；
- expert token 不均衡会让最热 expert/rank 决定 step tail。

最新版简历的结果是平均 step time 降低约 `50%`。以下是面对这类 workload 的分析与选型顺序，不是已经核实的逐项优化时间线或消融：

1. **排查非预期全量张量**：检查 logits/loss 是否意外 full materialize；具体 workload 是否走 actor logprob 分支须由调用链确认；
2. **并行网格**：用 TP 解决单层权重/GEMM、CP 分摊 128K、EP 分布 expert；将高频 TP/EP/CP group 映射到合适拓扑；
3. **MoE kernel**：Grouped GEMM、permute/unpermute、router/top-k 与 shared-expert overlap，以 token histogram 验证负载；
4. **activation/loss**：packing/THD、selective recompute、vocab-parallel CE/logprob chunk，避免 FP32 full logits 常驻；
5. **供给与 overlap**：DataLoader、H2D、TP/CP/EP collective 与计算 overlap；
6. **重新配 batch**：释放显存后评估增大 MBS、减少 microbatch/recompute 是否更划算。

当前材料没有逐项 A/B，所以 50% 只能作为联合结果；具体 TP/CP/EP、绝对 step time 和测量窗口留在证据卡。下一节的 actor CP-local logits 修复及约 7.6GB 冗余分配是另一项机制证据；它是否包含在这次 `-50%` benchmark 中，还需要原始配置和日志确认。补齐前两项分别陈述，不把它们拼成已证实的因果链。

<a id="cp-local-logits"></a>
## CP-local logits：7.6GB 冗余分配的原理与修复

![CP-local logits 修复：保留 logits 分片，只聚合标量](../assets/topics/cp-local-logits.svg)

### 根因不是一个普通 chunk 参数

THD packed sequence、`CP>1` 的 actor 路径中，模型输出本来是：

```text
local logits: [T/CP, V/TP]
```

旧 postprocess 为恢复 packed sequence 顺序，先在 CP group all-gather 整个 logits：

```text
full logits on every CP rank: [T, V/TP]
```

因此主张量峰值从近似 `(T/CP)×(V/TP)×bytes` 回到 `T×(V/TP)×bytes`。后续 sequence chunk 只控制 log-softmax/gather 等计算产生的临时 tensor；full logits 已经存在时，chunk size 再小也不能挽回这部分显存。这就是“chunking 看起来已配置，但约 7.6GB 冗余分配仍出现”的原因。

### 正确的数据流

1. actor 的 THD+CP 路径让 `postprocess_packed_seqs_context_parallel(..., gather_thd_outputs=False)` 保持 logits local；
2. 将 padded packed labels 用和模型输入完全相同的 causal zigzag 切分到本 rank；
3. 在 `[T/CP,V/TP]` 上使用 vocab-parallel primitive 计算 selected-token logprob、entropy、vocab min/max；
4. 只 all-gather `[T/CP]` token scalar；
5. 按 zigzag 逆变换恢复 full packed order，再 unpad；
6. 用显式 `_pcp_output_layout` 标记 THD/BSHD，不能只根据 `cu_seqlens` 猜 layout。

项目提交 `be6fb98f` 对应这条路径。Critic 保留其已验证的 full-gather 路径，不能因为 actor 能 local-scalar 就默认所有输出头都同语义。

### 验证矩阵

| 维度 | 至少覆盖 |
|---|---|
| layout | THD / BSHD、packed / padded、带 `cu_seqlens` 的 BSHD |
| parallel | CP=1 reference、CP&gt;1、TP vocab shard、不同 CP rank |
| path | train、forward-only/compute-logp、actor、critic |
| numeric | token logprob、entropy、loss、grad、rank checksum |
| memory | all-gather 前后 shape、peak allocated/reserved、7.6GB 峰值消失 |
| lifecycle | padding/unpadding、checkpoint/recovery、长窗口无泄漏 |

<a id="video-dit-ulysses"></a>
## 视频 DiT：从 640×640×3×129 到 Ulysses/Ring 并行

![视频 DiT 的 Ulysses sequence parallel 数据流](../assets/topics/ulysses-video-cp.svg)

原始视频 shape 不能直接当 Attention sequence。通用链路是：

```text
[frames, channels, height, width]
  → VAE temporal/spatial compression
  → latent video
  → patchify
  → sequence tokens T' × H' × W'
  → DiT blocks
```

准确 token 数取决于 VAE temporal/spatial stride、latent channels 与 DiT patch size；没有模型配置就不要现场报死数。

### Ulysses 做了什么

Ulysses 初始让每个 rank 持有 `S/SP` token、全部 attention heads。Attention 前第一次 All-to-All 将数据重排为“全 sequence、部分 heads”；rank 在本地对 head shard 做 attention；第二次 All-to-All 再恢复 sequence-sharded output。它切 activation，不切参数，主要解决视频时空 token 过长导致的显存问题。

代价与限制：

- 两次 All-to-All 对拓扑和消息均衡敏感；
- Ulysses degree 通常受 Q/KV head 可整除性和 kernel layout 约束；
- degree 继续增大时，每 rank heads 太少，通信相对计算上升；
- 可用 `ulysses_degree × ring_degree` 组合更大序列并行网格，将高频 Ulysses A2A 放机内高速域，Ring 通信再跨节点；
- 与外层 DP/PP 组合时必须按真实 process group 算 world size，不能重复乘同一组 rank。

### 以 HunyuanVideo-14B 为面试例子的优化闭环

1. 固定 640×640×3×129、VAE/patch config、batch、precision、并行网格与质量输入；
2. 计算 latent token shape，先确认 activation 一阶峰值；
3. sweep Ulysses/Ring degree，记录 attention kernel、两次 A2A、peak memory 与 step p95；
4. 开 FlashAttention、mixed precision、selective recompute 和 QKV/RoPE/RMSNorm/MLP fusion；
5. 冻结的 VAE/text encoder 使用 no-grad、缓存或阶段化执行，减少重复计算与显存常驻；
6. 按分辨率/帧数 bucket，避免 padding 把最大视频 shape 扩散到全 batch；
7. 在国产卡场景将 Ulysses group 映射到 HCCS，跨节点通信再看 RoCE 带宽与长尾；
8. 用逐层输出、loss/grad、生成质量、checkpoint/recovery 和长稳共同验收。

HunyuanVideo 官方公开的 USP/xDiT 是推理实现和配置示例，只能支撑机制理解；不能把其 NVIDIA 推理数字说成本人国产卡训练收益。

## Pretraining 视角

预训练阶段关注的是大规模 token 吞吐和稳定性。

关键问题：

- 位置编码和上下文外推：RoPE scaling、YaRN、ALiBi、原生长上下文配置会影响训练稳定性和后续 SFT/RL 对齐。
- 数据长度分布：长文档比例、拼接策略、document boundary、packing 策略会直接影响有效 token 利用率。
- kernel 选择：FlashAttention、block/ring attention、hybrid attention、linear attention 在不同长度段表现可能完全不同。
- 并行布局：TP/PP/DP 之外，需要考虑 [Context Parallelism](context_parallelism.md) 和 [Sequence Parallelism](sequence_parallelism.md)。
- checkpoint 成本：长 step time 下，checkpoint 频率、async save 和恢复时间会影响集群有效利用率。

预训练里最容易踩的坑是把 max length 当成静态参数，只改 `max_position_embeddings` 或 rope scaling，却没有重新评估 kernel、packing、batch token budget 和 checkpoint 周期。

## SFT 视角

SFT 阶段关注的是样本质量、loss mask 和长尾样本处理。

关键问题：

- multi-turn 样本的 loss mask 是否正确，特别是 tool call、assistant-only loss、system prompt 和 long document grounding。
- packing 是否保持语义边界，是否会让长样本挤掉短样本吞吐。
- `max_length`、`max_token_len_per_gpu`、micro batch、global token batch 的关系是否清楚。
- 长样本是否需要单独 bucket，还是统一走 dynamic packing。
- activation recompute / optimizer offload / CPU offload 是否真的提升可训练性，还是吞吐损失过大。

SFT 里最常见的问题不是“放不下”，而是：

- padding 或 packing 造成大量无效 token；
- 少量超长样本把 step time 拉爆；
- tokenizer/chat template/loss mask 不一致导致 loss spike；
- 保存 checkpoint 后恢复时并行布局或 tokenizer/config 不匹配。

相关章节：

- [Checkpointing](checkpointing.md)
- [FlashAttention](flashattention.md)
- [Tensor Parallelism](tensor_parallelism.md)
- [Context Parallelism](context_parallelism.md)

## RL 视角

长上下文 RL 比 SFT 更难，因为训练系统外面又接了一套 rollout / inference / reward 系统。

关键问题：

- rollout prompt 和 response 都可能很长，KV cache 成本会成为瓶颈。
- vLLM / SGLang 等推理引擎的 batching、prefix cache、chunked prefill 会影响训练吞吐。
- policy update 后权重同步慢，会放大 staleness。
- verifier / reward model 如果也需要长上下文，奖励计算会成为独立瓶颈。
- 长 trajectory 的存储、回放、截断和去重会影响样本质量。

长上下文 RL 的核心不是单纯把模型训到 128k，而是让 **rollout producer、trainer consumer、reward/verifier、checkpoint/recovery** 在长序列下仍然能稳定协同。

[CompactionRL](../papers/compactionrl.md) 给出了一个很好的系统信号：长上下文 RL 可以不只依赖更长的 max context，而是训练 agent 在固定 context budget 下主动压缩历史状态。这样 long-context training 的问题就从“单条序列放不放得下”扩展成：

- 什么时候触发 compaction；
- summary 是否保留了 task-relevant state；
- compacted trajectory 如何保存和回放；
- summary segment 如何参与 RL loss；
- final reward 如何跨 compaction boundary 做 credit assignment。

因此，长上下文 RL 的基础设施不只需要 CP、KV cache 和 chunked prefill，还需要 compaction-aware trajectory schema 和 rollout observability。

相关章节：

- [Agentic RL Infrastructure](agentic_rl.md)
- [Rollout Latency Playbook](../playbooks/rollout_latency.md)
- [DeepSeek-R1](../tech_reports/deepseek_r1.md)

## 关键工程判断

- 如果只是少量样本超过长度上限，先评估是否过滤或单独 bucket，不要立刻把全局 max length 拉高。
- 如果每卡 token 超过 kernel 舒适区，优先考虑 CP 或重新 bucket，而不是盲目加 recompute。
- 如果 step time 抖动大，先看长度分布、packing shape 和 kernel autotune，再看 NCCL。
- 如果训练能跑但吞吐低，检查有效 token 比例、padding waste、cross-node CP 通信和 checkpoint spike。
- 如果 RL 长上下文慢，先拆 rollout latency：prefill、decode、weight sync、reward、trajectory IO，而不是只看 trainer MFU。

## 生产排障速查

| 症状 | 第一批检查 | 常见根因 | 验证修复 |
|---|---|---|---|
| CP 增大但显存不降 | 逐 rank tensor shape、logits/loss、THD/BSHD layout | postprocess full gather、FP32 loss upcast、CP 路径回退 | local shape 按 CP 缩放，loss/logprob 与 CP=1 对齐 |
| 不 OOM 但 step 很慢 | recompute scope、MBS/microbatch、kernel timeline | full recompute 过重、TP 切碎 GEMM、CP 跨慢链路 | `none/selective/full` 与 TP/CP sweep |
| step p99 很高 | 样本长度/packing、expert token histogram、rank skew | 长样本、padding、expert hotspot、A2A 拥塞 | bucket 后 p99 收敛且有效 token 不下降 |
| DataLoader worker 增大无收益 | CPU/RSS/page fault、queue depth、H2D overlap | CPU/内存已饱和、小文件/远端 IO、主线程 collate | data wait 与 GPU gap 同时下降 |
| 视频 SP 扩展性差 | Ulysses A2A、head shard、拓扑 mapping | heads 太少、跨节点 A2A、layout conversion | sweep Ulysses×Ring，比较 exposed A2A 与 attention time |
| MFU 与 step time 不闭合 | estimator、有效 token、统计窗口 | FLOPs 公式/packing/计时边界不同 | 统一计量后重新计算，不做口头补数 |

## 一手资料

- [Megatron-Core TransformerConfig：selective recompute 与融合配置](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.transformer.transformer_config.html)
- [Megatron-Core Fine-Grained Activation Offloading](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/fine_grained_activation_offloading.html)
- [HunyuanVideo：Unified Sequence Parallelism / xDiT](https://github.com/Tencent-Hunyuan/HunyuanVideo/blob/main/README.md)
- [NVIDIA MoE 报告中文分节 PDF 入口](../README.md#megatron-core-moe-2026-zh-pdf)

## 当前阅读路径

1. 先读 [Transformer](../papers/transformer.md)，理解为什么 attention/MLP 是训练系统的基本计算图。
2. 再读 [FlashAttention](../papers/flashattention.md)，理解长上下文为什么首先变成 IO 和 kernel 问题。
3. 接着读 [Tensor Parallelism](tensor_parallelism.md) 与 [Context Parallelism](context_parallelism.md)，理解序列和算子怎么切。
4. 然后读 [Checkpointing](checkpointing.md)，理解长 step time 下如何保存和恢复训练状态。
5. 最后读 [Agentic RL Infrastructure](agentic_rl.md)，把长上下文从 SFT 扩展到 rollout / RL 系统。
6. 再读 [CompactionRL](../papers/compactionrl.md)，理解 long-horizon agent 如何在固定 context budget 下训练可压缩的 trajectory。
7. [Agentic for Embodied](agentic_for_embodied.md) 把 long-horizon 扩展到视频历史、planner memory、动作反馈和物理环境状态。

## 待补实验

- 128k SFT 下不同 CP size 的 step time、MFU、显存和通信 profile。
- 长短样本混合时，packing 策略对 step time std 的影响。
- 关闭/开启 recompute、optimizer offload、activation offload 的吞吐与稳定性对比。
- RL rollout 中长 prompt prefill、long response decode、weight sync 的 latency breakdown。

## 我的总结

长上下文训练是一条系统主线，不是一个开关。Pretraining 关注 token 吞吐和稳定性，SFT 关注 packing、loss mask 和长尾样本，RL 关注 rollout、KV cache、reward 和 policy version。真正的工程判断来自端到端拆解：每卡 token、kernel 舒适区、通信拓扑、checkpoint 恢复、rollout latency 和有效 token 利用率必须一起看。

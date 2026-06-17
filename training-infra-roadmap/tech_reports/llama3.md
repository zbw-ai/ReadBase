# The Llama 3 Herd of Models

## 论文信息

- 作者：Meta Llama Team
- 时间：2024
- 链接：https://arxiv.org/abs/2407.21783
- 相关主题：[Distributed Training](../topics/distributed_training.md), [Checkpointing](../topics/checkpointing.md), [Fault Tolerance](../topics/fault_tolerance.md)

---

## 架构概览

Llama 3 是一组 foundation models，最大公开模型为 405B dense Transformer，支持多语言、代码、推理和工具使用，context window 可到 128K。它不是 MoE 路线，而是 dense scaling 的高质量工程实践。

---

## 训练系统设计

Llama 3 的训练报告价值在于展示大规模 dense model 训练的全流程：数据过滤和配比、预训练、长上下文扩展、post-training、安全模型、评估和发布。对 infra 来说，405B dense 模型意味着参数、activation、optimizer state、checkpoint 和吞吐都必须被系统性规划。

---

## 并行策略

公开报告没有把全部内部训练栈细节展开，但从模型规模可推断需要 3D parallel 组合：Tensor Parallel 承担单层矩阵切分，Pipeline Parallel 承担层切分，Data Parallel/FSDP 类技术承担吞吐和状态分片。长上下文阶段还会放大 sequence/context parallel 的价值。

---

## 显存优化

405B dense 模型不像 MoE 那样只激活少量参数，每 step 的 dense compute 和 activation 压力更直接。显存优化重点包括 activation checkpointing、optimizer state sharding、mixed precision、合理 micro-batch、attention kernel 和 checkpoint shard 管理。

---

## 通信优化

Dense 训练通信路径以 TP collective、PP activation send/recv、DP/FSDP gradient/state 通信为主。规模上去后，通信优化不只是减少 bytes，还包括 overlap、拓扑布局、bucket sizing、pipeline bubble 控制和 straggler 诊断。

---

## 集群规模

Llama 3 最大 405B 模型属于超大规模 dense 训练。公开材料更强调模型和训练流程，而不是像 MegaScale 那样细讲集群系统；因此本仓库把 Llama 3 与 [MegaScale](megascale.md) 组合阅读：前者理解模型训练目标，后者理解万卡生产系统。

---

## 工程经验

- Dense 模型更容易建模吞吐，但总 compute 成本极高。
- 长上下文不是只改 RoPE/position scaling，还会放大 attention、activation、checkpoint 和数据 pipeline 压力。
- 训练平台需要支持从 pretraining 到 post-training 的多阶段作业，而不是单一大作业。
- 发布级模型需要可复现评估、安全数据和模型 lineage。

---

## 对行业的影响

Llama 3 把开放模型的规模和质量推到接近闭源前沿，也让社区看到 dense 405B 仍有强竞争力。对 infra 行业，它提示：MoE 很重要，但 dense 训练的稳定性、数据质量和系统成熟度仍然是基础能力。

---

## 我的收获

Llama 3 是理解现代开放模型工程化的必读材料。它的价值不在某个并行 trick，而在把大规模预训练、长上下文、post-training、安全和发布打通。作为训练 infra 工程师，应重点追问：405B dense 模型如何稳定跑完，如何保存和恢复，如何发现数据/系统异常，如何把训练系统服务于完整模型生命周期。

---

## 后续演进

- [Transformer](../papers/transformer.md)
- [Megatron-LM](../papers/megatron_lm.md)
- [ZeRO](../papers/zero.md)
- [FlashAttention](../papers/flashattention.md)
- [MegaScale](megascale.md)

---

## 面试高频问题

1. Llama 3 405B dense 与 DeepSeek-V3 MoE 的训练成本结构有什么不同？
2. Dense Transformer 的主要通信路径是什么？
3. 长上下文会如何改变显存和 kernel 瓶颈？
4. 为什么 post-training 也需要训练平台能力？
5. 405B dense 模型如何选择 TP/PP/DP？
6. Activation checkpointing 对 dense model 的收益和代价是什么？
7. Llama 系模型常见的 attention 结构优化有哪些？
8. 训练和推理的 KV cache 压力有什么不同？
9. 大规模模型发布为什么需要 lineage 和评估系统？
10. Llama 3 与 Llama 2 在 infra 视角下的主要升级是什么？

---

## 生产环境思考题

1. 405B dense 模型 checkpoint 一次需要多久才可接受？
2. 长上下文扩展阶段是否应与基础预训练使用同一并行配置？
3. 如何发现数据 pipeline 造成的 GPU starvation？
4. 如何在训练中监控每层耗时漂移？
5. 如果 pipeline stage imbalance，如何重切 layer？
6. 大模型多阶段训练如何保证 tokenizer、data mix、checkpoint lineage 一致？
7. 训练平台如何支持安全/对齐阶段的小规模高频实验？
8. context length 增加后 attention kernel fallback 如何发现？
9. 如何为发布模型保存可审计元数据？
10. 训练失败恢复后如何确认评估曲线连续？

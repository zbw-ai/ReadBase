# DeepSpeed-MoE

- 时间：2022
- 链接：https://arxiv.org/abs/2201.05596
- 状态：TODO 深读

## 工程阅读重点

DeepSpeed-MoE 关注 MoE 的训练和推理系统优化。重点看 expert parallel、MoE inference latency、通信优化和与 ZeRO/DeepSpeed 生态的组合。

## 与知识图谱的关系

[Switch Transformer](switch_transformer.md) → DeepSpeed-MoE → [MoE Topic](../topics/moe.md)。

## 待补问题

- MoE inference 为什么和 training 有不同瓶颈？
- expert slicing/parallelism 如何影响延迟？
- DeepSpeed-MoE 与 DeepSeekMoE 有哪些系统差异？

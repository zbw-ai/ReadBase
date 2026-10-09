# PipeDream: Generalized Pipeline Parallelism for DNN Training

- 时间：2018
- 链接：https://arxiv.org/abs/1806.03377
- 状态：TODO 深读

## 工程阅读重点

PipeDream 关注 pipeline 调度和吞吐，重点看 1F1B、weight stashing、异步/同步语义，以及 pipeline 并行对收敛和显存的影响。

## 与知识图谱的关系

[GPipe](gpipe.md) → PipeDream → 1F1B → [Megatron 2021](megatron_2021.md)。

## 待补问题

- weight stashing 解决什么问题？
- 1F1B 为什么能降低 activation memory？
- 生产训练为什么更偏好可控同步语义？

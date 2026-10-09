# Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning

- 时间：2022
- 链接：https://arxiv.org/abs/2201.12023
- 状态：TODO 深读

## 工程阅读重点

Alpa 代表自动并行搜索路线。重点看 inter-operator 与 intra-operator parallelism 的分层搜索，以及自动化方案在生产可控性、debuggability、稳定性上的边界。

## 与知识图谱的关系

Alpa → Auto Parallel → TP/PP/DP 配置搜索 → 训练平台调参自动化。

## 待补问题

- 自动并行搜索需要哪些 cost model？
- 为什么生产平台仍常保留人工配置能力？
- 自动并行如何处理 heterogeneous cluster？

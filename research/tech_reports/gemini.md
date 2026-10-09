# Gemini

- 时间：2023
- 链接：https://arxiv.org/abs/2312.11805
- 状态：TODO 深读

## 工程阅读重点

Gemini 是多模态 foundation model 报告。重点看多模态数据 pipeline、TPU/GPU 训练系统、模型规模、评估和安全流程。

## 在本手册中的定位

Gemini 不是第一轮 training-infra 主线材料。它保留在 `tech_reports/` 中，是为了后续扩展“多模态训练基础设施”时使用：重点会放在多模态数据 pipeline、跨模态 batch construction、评估系统和大规模训练组织方式，而不是 Gemini 模型能力本身。

## 与知识图谱的关系

Transformer → multimodal foundation model → 大规模数据/训练/评估系统。

## 待补问题

- 多模态训练的数据 pipeline 与纯文本有何不同？
- 图像/音频/视频 token 对显存和吞吐有什么影响？
- 多模态训练如何做 checkpoint 和评估？

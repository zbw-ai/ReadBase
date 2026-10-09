# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

- 时间：2018
- 链接：https://arxiv.org/abs/1810.04805
- 状态：TODO 深读

## 工程阅读重点

BERT 是 encoder-only Transformer 预训练代表。阅读时不要只看 NLP benchmark，要关注 MLM/NSP 训练负载、双向 attention 对 mask/kernel 的影响、sequence length 与 batch size 的显存权衡。

## 与知识图谱的关系

[Transformer](transformer.md) → BERT → Encoder-only pretraining → 大 batch 数据并行训练。

## 待补问题

- BERT 训练对数据 pipeline 和 batch construction 有什么要求？
- Encoder-only 与 decoder-only 的 activation/attention pattern 有何差异？
- 今天哪些 infra 技术仍受 BERT/MLPerf 训练影响？

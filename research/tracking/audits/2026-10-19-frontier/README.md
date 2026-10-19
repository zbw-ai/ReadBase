# 2026-10-19 Frontier 证据边界

[完整扫描](../../frontier_scan_2026-10-19.md) · [扫描账本](../../scan_log.md)

事件窗口UTC 10/14 06:20:25 → 10/19 01:40:52；后续取证不延长此窗口。索引完整只表示该查询返回完整，不代表全部历史、direct commits或逐文件源码审计。

| 证据 | 范围 |
|---|---|
| [具身核验](embodied_evidence.json) / [PR与release索引](embodied_merged_pr_release_index.json) / [arXiv目录](embodied_arxiv_index_summary.json) | 111 merged PR，67 release，3篇方法重点核查；branch/CI/runtime边界保留 |
| [RL核验](rl_evidence.json) / [索引](rl_merged_pr_index.json) | 六核心仓97 merged PR、178 release历史；另核TRL发布与ReFlow |
| [Systems核验](systems_evidence.json) / [索引](systems_merged_pr_index.json) / [arXiv目录](systems_arxiv_coverage.json) | 750 runtime merged PR；3篇重点论文与4个Accepted组 |
| [厂商/HF核验](root_evidence.json) | 四厂商、HF博客与模型元数据、8仓有界release；401/提取失败保留 |
| [Manifest](manifest.json) | 事件窗口、组数、文件hash与验证记录 |

只归档必要元数据、文件名/哈希、独立核验摘要，不归档完整abstract、文章、PR body或diff。没有执行上游代码或训练/仿真/真机实验。当前窗口覆盖不关闭9/22起的历史缺口，来源间重叠arXiv数不相加。

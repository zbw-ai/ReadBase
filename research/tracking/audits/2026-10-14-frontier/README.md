# 2026-10-14 扫描证据

对应[完整报告](../../frontier_scan_2026-10-14.md)。这里保存来源 URL、原始日期、作者、固定提交、页面哈希、索引边界和核查结论，不保存论文/博客全文或临时下载路径。

- [manifest](manifest.json)：16 个接受组与 27 个观察组的唯一计数；Watch 重复不再计数。
- [具身证据](embodied_evidence.json)：模型 metadata / 方法、七库有界查询、代表性代码和公开 CI；不是本地实验记录。
- [RL 证据](rl_evidence.json)与[54 条 PR 索引](rl_merged_pr_index.json)：六库完整查询及重点 patch；未归档完整原始 API payload，可从各项官方 URL / commit 继续核对。
- [Systems 证据](systems_evidence.json)与[427 条 PR 索引](systems_merged_pr_index.json)：merged 查询结果的标题索引，`closed_at` 保留 Search API 原字段；Accepted PR 的精确 `merged_at` 在 evidence 中，不将全部 Search 行冒充逐 PR detail 核验。
- [arXiv 目录摘要](arxiv_index_summary.json)：发现范围，不等于所有论文全文审阅。
- [厂商/HF 证据](root_evidence.json)：目录和正文核查、DeepSeek 模型 history/diff、八库 release 列表；没有覆盖所有官方文档历史。

事件 cutoff 为 2026-10-14 06:20:25 UTC，部分页面在之后获取；只纳入截止前事件。报告核验时间与下一轮事件游标分开，旧历史缺口继续保留。本文档校验不代表上游代码、GPU、仿真或真机测试通过。

# 2026-10-23 Embodied audit

事件窗口：2026-10-19T01:40:52Z（不含）→2026-10-23T03:09:27Z（含）。核验在cutoff后继续，不接受cutoff后事件。

- [Merged PR / release索引](merged_pr_release_index.json)：七库80条merged，66条release-list；窗口release仅Isaac Lab v3.0.0。各库返回/分页/限额/base/SHA已记录。
- [arXiv索引](arxiv_index_summary.json)：六页2,756去重ID含replacements；标题过滤是发现范围，不是全篇核验。
- [论文metadata与阅读范围](paper_metadata.json)：16候选；4 Accepted、11窗口内Observed、1窗口前backfill候选；15篇HTML可取，SuperMimic HTML404。
- [PR核验边界](pr_verification.json)：12个详情/完整文件清单；LeRobot #4890与#4949完整diff，其余不声称全patch audit。
- [页面与代码工件metadata](source_pages.json)：官方页/README hashes；只有引用元数据，不镜像全文。

全部Accepted First seen为2026-10-23、Status NEW。此次没有执行上游脚本、测试、训练、仿真或真机。作者实验、上游CI和本仓库实测严格区分。

5组Accepted（4篇论文+LeRobot #4890），17组Observed见主报告。CusADi2 2610.25476v1首次提交为10/18 19:03:28 UTC，作为late-discovered/backfill候选，不计新信号。保留9/22起历史论文发现及direct commits/外围PR缺口。

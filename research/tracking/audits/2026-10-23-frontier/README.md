# 2026-10-23 Frontier 审计入口

[完整报告](../../frontier_scan_2026-10-23.md) · [扫描账本](../../scan_log.md) · [Manifest](manifest.json)

固定事件窗口：2026-10-19 01:40:52 → 2026-10-23 03:09:27 UTC。之后只核验已选事件；核验完成时间不代替下一游标。19组Accepted含22个主Source ID、54组Observed，另3篇旧材料按10月backfill保存，2篇日期异常待核。

| 证据 | 实际范围与限制 |
|---|---|
| [具身说明](embodied/README.md) / [PR与release索引](embodied/merged_pr_release_index.json) / [论文metadata](embodied/paper_metadata.json) | 7库80 merged；66 release列表耗尽，窗口内IsaacLab3.0.0；2756 arXiv去重ID。仅重点方法/代码深读。 |
| [RL说明](rl/README.md) / [索引](rl/merged_pr_index.json) / [Accepted证据](rl/accepted_evidence.json) | 六核心仓91 merged，Megatron/DeepSpeed/TRL110；release列表448；限定arXiv查询467。33重点PR核详情/文件清单，不是201个diff全审。 |
| [Systems manifest](systems/manifest.json) / [索引](systems/merged_pr_index.json) / [证据](systems/evidence.json) | 三runtime893 merged +四HF库146；release首100项/仓；2339 arXiv去重ID。检查存在CI失败，未冒充GPU验证。 |
| [厂商目录](root/evidence.json) / [正文metadata](root/details.json) / [公开工件](root/repos.json) / [核验摘要](root/notes.json) | 四厂商/HF可见索引、选定文章、DeepSeek最新20模型metadata；HF社区非全量。TorchNeuron与证明工件使用固定SHA。 |

23仓共1320窗口merged PR，query完整只说明所列仓库/窗口/查询完整；不覆盖direct commits、所有旧PR或未合并提案。不同arXiv目录及查询重叠，不得相加。9/22起历史发现与GitHub缺口继续保留。

仅存必要metadata、hash、文件索引和独立核验判断，不镜像论文/博客/PR全文或diff。未执行上游代码、训练、证明checker、GPU测试、仿真或真机。链接与JSON等本仓文档检查结果见manifest的validation记录。

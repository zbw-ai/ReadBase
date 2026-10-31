# 2026-10-31 Frontier 审计

[完整报告](../../frontier_scan_2026-10-31.md) · [唯一游标账本](../../scan_log.md) · [Manifest](manifest.json)

事件窗口固定为 **2026-10-23 03:09:27 → 2026-10-31 14:20:20 UTC**，对应 Asia/Shanghai 10/23 11:09:27 → 10/31 22:20:20。之后的请求只补核 cutoff 前的材料，不扩展事件窗口。15组Accepted、18组Observed，1篇10/22精度审计独立backfill。

| 记录 | 实际证据与边界 |
|---|---|
| [requests.json](requests.json) | 204次请求的URL、请求开始时间、HTTP/失败信息、成功响应hash。正文/patch仅临时阅读，不镜像全文；`retrieved_at`是请求发起时间，不是事件发布日期。OpenAI研究/工程页403后RSS成功；CV/CL重试失败；GitHub search/REST限流后使用页面、patch和Atom。 |
| [discovery_titles.json](discovery_titles.json) | RO/LG/DC/AR recent目录2002个去重ID与题名；目录分别485/1438/125/72，交叉收录不能相加。目录显示10/23、26、27、28、30批次，缺失日期不宣称覆盖。仅重点材料进一步核查，不是全文阅读清单。 |
| [papers.json](papers.json) | 成功取得的13篇候选arXiv `citation_title`、完整`citation_author`、`citation_date`及submission history；包括Observed与backfill，不等于13篇Accepted。版本差异在报告注明。 |
| [repositories.json](repositories.json) | 24仓每仓首30 releases；10仓merged search共153条，均returned=total、incomplete=false。其他search限流；ROLL/OpenRLHF/NeMo closed PR首100，分别0/0/70条窗口merged。Atom快照非全窗口；release首30不等于历史耗尽。 |
| [pr_evidence.json](pr_evidence.json) | 15个重点PR的merge时间/SHA及patch文件索引；阅读了相关机制、配置、checkpoint/调度与测试段，不声称所有patch全审。多提交patch有旧中间状态，判断以最终合并说明及修正为准。 |

未安装/执行外部仓库脚本、上游skills、模型、训练、GPU benchmark、仿真或机器人控制。上游CI与论文数字仅为来源证据，不是本仓测试结果。保留旧arXiv 9/22 16:52:05 Asia/Shanghai与GitHub direct commits/历史外围PR/未核release history 9/22 17:06:16缺口；本轮受限来源另从10/23回扫。

本仓Markdown目标/anchor、JSON和diff检查的实际结果记录于manifest；这里只做文档验证，不能提升研究生命周期到VERIFIED。

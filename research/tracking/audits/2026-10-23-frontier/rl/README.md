# RL / Training 子扫描审计（2026-10-23）

事件窗口：2026-10-19T01:40:52Z → 2026-10-23T03:09:27Z。核验延续至下列 JSON 各自的 queried/verified timestamp，不扩展事件 cutoff。

- [merged PR / release 索引](merged_pr_index.json)：9 仓 201 条合并 PR；六个核心 RL 框架 91 条；查询全部 `incomplete_results=false`。release 历史返回 448 条，其中核心六仓 178 条；OpenRLHF 100+10、DeepSpeed 100+18、TRL 100+1，其余一页。窗口唯一 release 是 DeepSpeed v0.19.8，不能据此推断它包含窗口全部 PR。
- [重点 PR 元数据](selected_pr_metadata.json)：33 个 PR 的官方题名、作者、merge 时间/SHA/base 和实际文件清单；各文件查询均为一页且 returned=changed_files。仅取文件列表不等于逐行审阅。
- [Accepted 证据和边界](accepted_evidence.json)：6 组（3 组核心框架 5 个 PR，1 组 TRL 1 个 PR，2 篇论文）；明确实际读取路径与本轮未执行上游测试。
- [arXiv 限定查询索引](arxiv_query_index.json)：当前窗口 cs.DC/cs.LG、training/reinforcement 查询 467/467，300+167；仅元数据。分钟粒度发现后，Accepted 按 abs 秒级 v1 时间核窗。题名/关键词筛选，不能称全 arXiv 覆盖。
- [其他论文及 HF 博客 triage](supplemental_triage.json)：4 篇论文和 2 篇博客 Observe。另有 13 组 PR Observe，详见主扫描报告（草稿交接只存 /tmp，不作为仓库链接）。

初次沙箱网络访问失败；授权只读重试完成，无未完成 GitHub 分页。没有执行远程脚本或上游测试，没有下载运行模型。持久文件只保存源元数据、文件清单、短原创判断，不含 PR 正文、patch 或论文全文。正文/patch 的临时读取材料不作为持久证据链接。

范围限制：所有 base branch 都进入 merged query，特别是 Megatron dev 与 NeMo release cherry-pick 不能冒充 main 新机制。默认分支 direct commits、历史外围 PR、旧 release 全文和旧 arXiv 发现缺口自 2026-09-22 17:06:16 Asia/Shanghai 保留；本轮访问 release 历史列表不等于回填旧 release 内容。未扫描纯 cs.CL 的完整索引；HarnessRL 来自补充定向检索。没有新增 backfill 事件，不推进任何游标。

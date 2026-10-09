# 两层编号与旧训练入口收编 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用户已确认两层编号，并要求收编 `training-infra-roadmap/`，让原来的文档重新容易找到。

**Architecture:** 五个技术 Part 仅加固定数字前缀；第二层编号只放在各 Part 的 README 导航，不改变 Topic basename 或正文标题。原训练入口的功能并入训练 README；跨领域正文继续只有一个维护位置。遵循用户只维护 main 的要求，不新建分支或另一套站点。

**Tech Stack:** Git、Markdown、现有链接/锚点检查器；不改运行环境、不执行 GPU 实验。

## 已确认的范围

| 原目录 | 唯一新目录 |
|---|---|
| `systems/` | `01-systems/` |
| `training-infra/` | `02-training-infra/` |
| `inference-infra/` | `03-inference-infra/` |
| `rl-infra/` | `04-rl-infra/` |
| `embodied-infra/` | `05-embodied-infra/` |

`research/`、`practice/`、`interview/`、`private_resume/`、`reading_inbox/`、附件与配置保持原位。编号表示分类位置，不表示学习优先级；具身仍是首页当前重点。

`training-infra-roadmap/` 只剩 README 与 7 个 Topic 跳转页，没有独立正文。确认目标正文存在后，撤下这 8 个冗余跳转页：保留它们的导航用途到 `02-training-infra/README.md#original-library`，不再保留平行的旧根目录。文件历史可通过 Git 查询；GitHub 不会为普通目录重命名自动重定向，外部旧书签需要更新。

## Task 1：保全与机械迁移

- [x] 检查 main 工作区干净，fetch 后 HEAD 与 origin/main 一致；基线为 `ab6e192`。
- [x] 运行现有 Markdown 链接/锚点检查，记录迁移前结果。
- [x] 基线中的 33 个技术 Part 文件按上表一一映射（以 Git 文件清单为准），内容只做路径替换；不改变研究决策、游标、实验状态、数值或附件。
- [x] 更新当前有效的相对链接、GitHub main 链接和 AGENTS 中的目录约定。历史迁移 JSON、提交固定的 URL、来源快照保留原样；历史规格中的原路径说明不伪改成当时已存在的新路径。
- [x] 旧目录 8 个跳转页的正文目标与显式锚点逐一核对，入口表覆盖后撤下。

## Task 2：找回入口与两层编号

- [x] 根 README 前部增加“原文档快捷入口”，提供原训练手册、Megatron/5D、FSDP、MoE、RL/Gateway、PDF、论文索引、扫描/月报、学习计划、实验/项目、面试主文档入口。保留 `recent-updates`、`learning-parts`、`core-roadmaps` 锚点；最近更新仍不超过 5 条。
- [x] 五个 Part README 标题统一为 Part 01–05，重点 Topic 用 `1.1`／`2.1` 等编号，原有内容边界和未完成状态保留，不用空文件凑章节。
- [x] 训练 README 的 `original-library` 表收编原训练入口；同页直接可达原 30/90 天及年度计划、PDF 与跨 Part 技术正文，不让用户再追着迁移说明跳转。
- [x] 知识图、材料索引同步当前入口；实践/面试仍分别作为 Part 06/07 导航，不改目录名。其他正文仅机械改路径，不扩写技术内容。

## Task 3：独立验收与发布

- [x] 验证所有原始受跟踪文件都有去向；除批准的 8 个跳转页之外无丢失。非导航正文逐字等价于基线加路径替换，所有二进制附件哈希不变。
- [x] 全库 Markdown 路径/锚点及 SVG/JSON/CSV 验证通过；检查无冲突标记、无活跃导航指向已删除目录。
- [x] 独立规格审阅通过，再独立质量审阅；修复发现的问题。
- [x] 显式暂存本轮文件，commit 并 push main，核对本地/远端 SHA 与干净状态；如保护规则拒绝，不绕过规则。
- [x] 实际打开 GitHub 首页及训练入口，验证编号、旧入口表与正文链接。

## 验收记录

- 迁移前：268 个 Markdown，4,594 条本地引用，1,536 条锚点引用，0 错误。
- 旧 8 个跳转页的 62 条本地链接、38 条显式锚点已在迁移前核对可达；其导航用途集中到训练 README，历史保留在 Git 中。
- 迁移后：261 个 Markdown，4,601 条本地引用，1,522 条锚点引用，0 错误；20 个 JSON、7 个 SVG、11 个 CSV 解析通过。
- 文件守恒：基线 339 个文件中保留 331 个，删除批准的 8 个跳转页；33 个 Part 文件迁移。296 个非导航文本仅改路径，730 个 Part 正文标题不变；24 个二进制文件字节一致，120 个研究记录与扫描游标内容保留。
- 独立规格审阅与质量审阅均通过，无待修复项。不会将目录整理记为阅读完成或实验验证。
- 正文与导航发布提交：`2524a6d`，正常 push 至 `origin/main` 后通过 `git ls-remote` 确认 SHA 一致；未创建分支或 PR，未改仓库保护规则。
- Chrome 实页验收：GitHub 首页显示 01–05 目录和“原文档快捷入口”；训练 README 显示常用入口、2.1–2.7、学习计划、五份 PDF 和面试链接；从该页点击 FSDP 可正常显示技术正文。普通目录重命名不会重定向旧书签，应改存新的首页或训练入口。

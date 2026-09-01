# ReadBase · AI Systems Research OS 品牌软升级设计

## 目标

保留 GitHub 仓库名、本地目录名和 `ReadBase` 品牌，将用户可见定位统一为 `ReadBase · AI Systems Research OS`，明确长期覆盖 Training Infrastructure、Inference Infrastructure、Agentic RL、GPU Systems 与生产工程。

## 修改范围

1. `README.md`
   - 一级标题改为 `ReadBase · AI Systems Research OS`。
   - 首段继续解释 Personal Research Operating System，并增加长期覆盖范围。
2. `AGENTS.md`
   - Repository Purpose 的首句使用新的展示名称；不改变工作流规则。
3. `CLAUDE.md`
   - Repository Overview 的首句使用新的展示名称；不改变工具或验证规则。
4. GitHub repository description
   - 设置为：`Personal Research OS for Large-Scale AI Systems: training, inference, Agentic RL, GPU systems and production engineering.`

## 明确不修改

- 不修改 GitHub repository slug `ReadBase`、remote URL 或本地目录名。
- 不批量改写历史 tracking、plans、specs 和绝对路径。
- 不修改 `assets/readbase-knowledge-map.svg` 的 `ReadBase` 主品牌；该图仍是 Training Infra 阶段的知识地图。
- 不触碰当前工作区中 `training-infra-roadmap/` 下另一组未提交的 frontier scan、monthly signal 和导航改动。
- 不新建永久品牌说明文档；本规格在最终品牌提交前删除并从最终提交树中消失。

## 验证

- `git diff --check` 通过。
- 最终品牌提交只包含 `README.md`、`AGENTS.md`、`CLAUDE.md`。
- 现有未提交的 `training-infra-roadmap/` 文件保持原状、保持未暂存。
- GitHub repository slug 仍为 `zbw-ai/ReadBase`，Description 与设计文本一致。
- 推送后本地 HEAD、`origin/main` 和 GitHub ref 一致。

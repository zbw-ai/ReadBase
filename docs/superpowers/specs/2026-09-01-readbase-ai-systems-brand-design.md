# ReadBase · AI Systems Research OS 品牌软升级设计

## 目标

保留 GitHub 仓库名、本地目录名和 `ReadBase` 品牌，将用户可见定位统一为 `ReadBase · AI Systems Research OS`，明确长期覆盖 Training Infrastructure、Inference Infrastructure、Agentic RL、GPU Systems 与生产工程。

## 修改范围

1. `README.md`
   - 一级标题改为 `ReadBase · AI Systems Research OS`。
   - 首段改为：`ReadBase · AI Systems Research OS 是一个面向 Large-Scale AI Systems 的中文 Personal Research Operating System。`
   - 增加一句：`长期覆盖 Training Infrastructure、Inference Infrastructure、Agentic RL、GPU Systems 与生产工程。`
   - 将“当前专题”中的 Phase 3 改为 `Agentic RL / Agent Infrastructure`，Phase 4 改为 `GPU Systems / AI Engineering`；production engineering 作为横向工程原则，不新增 Phase。
2. `AGENTS.md`
   - `## Repository Purpose` 首句改为：`ReadBase · AI Systems Research OS is a Chinese-language, content-first Personal Research Operating System for Large-Scale AI Systems.`；不改变工作流规则。
3. `CLAUDE.md`
   - `## What this repo is` 首句使用与 `AGENTS.md` 相同的展示名称和定位，保留其后“not a software application”等 Claude 专用说明；不改变工具或验证规则。
4. GitHub repository description
   - 设置为：`Personal Research OS for Large-Scale AI Systems: training, inference, Agentic RL, GPU systems and production engineering.`

## 明确不修改

- 不修改 GitHub repository slug `ReadBase`、remote URL 或本地目录名。
- 不批量改写历史 tracking、plans、specs 和绝对路径。
- 不修改 `assets/readbase-knowledge-map.svg` 的 `ReadBase` 主品牌；该图仍是 Training Infra 阶段的知识地图。
- 不触碰当前工作区中 `training-infra-roadmap/` 下另一组未提交的 frontier scan、monthly signal 和导航改动。
- 不新建永久品牌说明文档；本规格已经进入远端历史，最终品牌提交会将它从当前文件树删除，但不重写已发布历史。

## 验证

- `git diff --check` 通过。
- 最终品牌提交只包含 `README.md`、`AGENTS.md`、`CLAUDE.md` 的修改和本规格文件的删除。
- 实施前记录 `training-infra-roadmap/` 的 porcelain status、tracked binary diff SHA-256 和 untracked file SHA-256；实施后必须完全一致。
- 只使用显式 path staging，并核对 `git diff --cached --name-status`；禁止 `git add -A`，不得暂存 `training-infra-roadmap/` 下的并发工作。
- GitHub repository slug 仍为 `zbw-ai/ReadBase`，Description 与设计文本一致。
- 推送后本地 HEAD、`origin/main` 和 GitHub ref 一致。

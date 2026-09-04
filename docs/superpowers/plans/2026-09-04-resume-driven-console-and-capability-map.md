# 简历驱动速查控制台与双主线能力图实施计划

## 范围与基线

按已批准规格 `../specs/2026-09-04-resume-driven-console-and-capability-map-design.md` 实施。

- 发布基线：`3498ed5`
- 已批准规格 HEAD：`aa7e124`
- 允许新增：本计划、上述设计规格
- 允许修改：
  - `private_resume/2026-08-llm-infra-interview-prep.md`
  - `private_resume/assets/llm-infra-personal-capability-map.svg`
- 禁止修改 79 道题的 ID、正文、优先级、Part 归属和 Core 10 顺序。

## 主工作树保护

不得在 `/Users/zengbw/ReadBase` 执行写文件、switch、merge、rebase、pull、reset、commit 或 push。开工基线：

- dirty files：
  - `docs/superpowers/specs/2026-09-01-parallel-folding-topic-design.md`
  - `private_resume/2026-08-llm-infra-interview-prep.md`
  - `training-infra-roadmap/topics/data_parallelism.md`
- `git diff --binary` SHA-256：`d01efc17b7ebb0c2be22c762e5fa228823f3c60e2e5354f2a1d11a468a7540f8`
- 文件 SHA-256：
  - `ff96f2533a4e2ce37639f5b0700e9a979c5e0fbabda5c53925ae03f2f7ecc715`
  - `a4af2de96006c02e27205eb49e0ca2a7beee7502163088bfda31bccc7976aaee`
  - `11049a96578780e8b82f011cf725f2bb4c2cd618d01da4721470d75aa58ca0eb`

## 实施步骤

1. 主文档目标薪资改为 `100–150 万`，删除旧开场抽象说明，保留稳定 `#interview-console` 锚点。
2. 将控制台三套表合并为一张三列表格：1 行教育背景、4 行工作技能、6 行项目经历；把关键数字及最短边界放回所属项目。
3. 保持 0.2 图片引用位置不变，将 SVG 重构为：定位层 → 两条主线前两行 → 共同工程底座第三行 → 两列三行证据层。
4. 统一为暖灰白背景、灰蓝 Training、深青 RL/Agentic、中性 shared；移除曲线、同心圆、箭头和拥挤中心卡。
5. 使用本机 SVG 渲染能力生成 1400px、900px 预览，检查文字、留白、gutter、边界注记和图例；若默认工具不可用，使用浏览器或已安装 Python SVG 库，不安装新依赖。
6. 运行 fence-aware 题目/回链验证、基线 ID 集合对比、控制台结构与数字边界断言、Markdown 本地链接、SVG XML/self-contained/颜色/marker 检查及 `git diff --check`。
7. 比对 changed-files allowlist 和 dirty 主工作树全部指纹；提交后 fetch，确认 `origin/main` 是 HEAD 祖先，只执行 `git push origin HEAD:main`。
8. 发布后确认 `origin/main==HEAD`、隔离 worktree clean、远端未创建功能分支，并再次比对主工作树指纹。

## 完成定义

- 文档从元信息直接进入 `## 0. 考场速查`，目标薪资为 `100–150 万`。
- 控制台只有一套简历驱动入口，项目经历为主体；旧三套标题不再出现。
- 能力图在 1400px 和 900px 预览下无溢出、遮挡或截断，两条主线和共同工程底座一眼可辨。
- 六个能力域、实心/空心语义、六张证据卡及其数字/边界全部保留。
- 79 道问题、`48/26/5`、Part `7/27/17/18/10`、Core 10 和逐题双回链完全不变。
- GitHub 只更新 `main`，主工作树原有修改保持逐字节不变。

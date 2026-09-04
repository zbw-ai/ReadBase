# 面试进度附录与 Coding 题单实施计划

## 范围

按已批准规格 `../specs/2026-09-04-interview-progress-and-coding-design.md` 实施，只修改：

- `private_resume/2026-08-llm-infra-interview-prep.md`
- `private_resume/2026-09-interview-coding.md`
- 本计划与对应设计规格

不在 dirty 主工作树执行 Git 写操作，不新增面试复盘文档，不复制已有题目的完整答案。

## 实施步骤

1. 更新主文档控制台、Part III 全量索引和局部导航，引入 `ROLLOUT-01`；同步题量为 79、P0 为 48、Part III 为 17。
2. 扩写 `MEGATRON-01` 的框架总览、`MEGATRON-02` 的 MLP/Attention TP shape、`DIST-01` 的 FSDP/FSDP2 生命周期；专项机制继续回链已有唯一主答案。
3. 新建 Coding 题单，提供 PyTorch MHA 与 `N×N` 矩阵原地顺时针旋转的可运行实现、复杂度、易错点和测试。
4. 在主文档全文末尾加入截至 2026-09-04 的六列四行进度台账，并加入控制台/Coding 双向导航。
5. 从 Markdown code fence 抽取代码，用 PyTorch 2.12 环境运行数值对照、梯度和异常测试；运行题量、优先级、Part、逐题回链、锚点、链接、SVG/XML 与 `git diff --check` 检查。
6. 复核 changed-files allowlist 和 dirty 主工作树指纹；在隔离 worktree 提交，fetch 后确保 fast-forward，只推送 `HEAD:main`；发布后确认远端仅有 `main` 且 SHA 一致。

## 完成定义

- 主文档恰有 79 道唯一问题，P0/P1/P2 为 `48/26/5`，Part I–V 为 `7/27/17/18/10`。
- `ROLLOUT-01`、Coding 入口和进度附录均可从顶部到达，并可回到控制台。
- MHA 与矩阵题的文档内代码由真实测试覆盖。
- 技术内容遵守项目证据边界；进度附录不包含逐场复盘。
- 远端只维护 `main`，dirty 主工作树完全不变。

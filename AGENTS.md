# AGENTS.md

This file is the shared working guide for AI coding agents in this repository. It applies to Codex, Claude Code, and other agents unless a tool-specific file says otherwise.

## Repository Purpose

ReadBase is a Chinese-language, content-first engineering knowledge base. It is not an application repo and has no product build pipeline. The current primary handbook is `training-infra-roadmap/`, whose goal is to become an **AI Training Infrastructure Engineering Handbook**.

Write for a software engineer growing into a senior AI/LLM training infrastructure engineer. Prefer engineering judgment, system design, production troubleshooting, and interview readiness over academic-style summaries.

## Working Style

- Use Chinese for authored handbook content, keeping common technical terms in English: TP, ZeRO, FSDP, FlashAttention, GEMM, NCCL, all-reduce, checkpoint, etc.
- Do not turn papers into translation notes. Explain what problem the work solved, which training-system bottleneck moved, what breaks in production, and how modern systems inherited the idea.
- Keep changes scoped. Do not rewrite unrelated files or reorder large sections unless the task explicitly asks for it.
- Treat existing uncommitted files as user or other-agent work. Do not revert them without explicit permission.

## Structure

- `README.md` is the ReadBase umbrella entry point.
- `training-infra-roadmap/README.md` is the Training Infra Handbook entry point.
- `training-infra-roadmap/papers/` contains engineering-perspective paper notes.
- `training-infra-roadmap/tech_reports/` contains model/system technical reports.
- `training-infra-roadmap/topics/` contains long-form engineering handbook chapters.
- `training-infra-roadmap/interview/` contains interview handbook notes.
- `training-infra-roadmap/roadmaps/` contains staged learning plans.
- `training-infra-roadmap/references/` contains CSV indexes.
- `training-infra-roadmap/assets/` contains handbook-specific visual assets.
- `assets/` contains root-level shared assets.

## Document Templates

For `papers/`, follow this section order:

论文信息 → 解决的问题 → 背景与瓶颈 → 核心创新 → 关键图表解读 → 工程价值 → 对训练基础设施的影响 → 今天的应用场景 → 后续演进 → 相关论文 → 相关代码 → 面试高频问题 → 生产环境思考题 → 我的总结.

For `tech_reports/`, follow this section order:

论文信息 → 架构概览 → 训练系统设计 → 并行策略 → 显存优化 → 通信优化 → 集群规模 → 工程经验 → 对行业的影响 → 我的收获 → 后续演进 → 面试高频问题 → 生产环境思考题.

For `topics/`, write as engineering handbook chapters: problem framing → mechanism → config guidance → production pitfalls → troubleshooting → adjacent-system relationships.

For `interview/`, include: 高频面试题 → 追问问题 → 生产环境案例 → 常见错误回答 → 优秀回答示例.

## Linking Rules

- Use relative Markdown links between files.
- Keep links bidirectional when adding important relationships.
- Update `training-infra-roadmap/KNOWLEDGE_GRAPH.md` and `training-infra-roadmap/MASTER_READING_LIST.md` when adding a relationship that changes navigation.
- Before claiming completion, verify internal Markdown links and image paths.

## Diagram Rules

- Use Mermaid for lightweight navigation diagrams in README files and `KNOWLEDGE_GRAPH.md`.
- For core paper/topic explanations, prefer research-paper-style SVG figures when the diagram carries conceptual weight: model blocks, data flow, communication patterns, checkpoint layouts, parallel groups, and kernel IO paths.
- SVG figures should be light-toned, readable on GitHub, and visually calm.
- Avoid overlapping elements, especially overlapping text.
- Put secondary annotations in whitespace or side callouts instead of crowding the main flow.
- Keep arrow crossings rare and meaningful.
- Use consistent colors for semantic categories such as GEMM/Linear, Attention/kernel, state boundary, residual, communication, and checkpoint.
- When embedding original paper figures in this public repo, use them sparingly, cite the source clearly, and prefer key figures that anchor the reader's understanding.

## Verification

Typical checks for this repo:

- Markdown links and local image paths resolve.
- SVG files parse as XML.
- JSON files parse.
- CSV files parse with the expected number of columns.
- Git status clearly separates tracked edits from new assets.

Do not say a change is complete until the relevant checks have actually run.

## Git And Publishing

Before returning commit commands, check local status and fetch the remote reference so the diff against GitHub is clear.

Prefer commit messages that describe the learning artifact, for example:

```bash
git commit -m "Improve Transformer figures with research-style system view"
```

If GitHub rejects a direct push because branch rules require a pull request, create or suggest a branch-and-PR flow instead of weakening repository protection rules.

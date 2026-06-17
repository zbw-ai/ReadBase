# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

ReadBase is a **content-first knowledge base** (Chinese-language Markdown), not a software application. There is no product build, unit-test suite, lint pipeline, or dependency tooling. Most files under `training-infra-roadmap/` are hand-authored Markdown or CSV. "Working in this repo" means writing, restructuring, and cross-linking engineering handbook notes; verification usually means checking Markdown links, SVG/XML validity, CSV parseability, and Git status.

The stated purpose (README.md): this is an engineering *handbook* for AI/LLM training infrastructure, oriented toward engineering judgment, system design, production troubleshooting, and interview prep — explicitly **not** an academic paper collection or concept-summary dump.

## Structure

Top level is `ReadBase/` with one self-contained "专题手册" (topic handbook) subdirectory, `training-infra-roadmap/`. The README anticipates sibling handbooks later (`inference-infra-roadmap/`, `agentic-rl-roadmap/`, etc.); each future handbook is expected to carry its own `README.md`, reading list, and knowledge graph.

Inside `training-infra-roadmap/`:

- `papers/` — paper notes, all using the same engineering-perspective template (see below).
- `tech_reports/` — model/system reports (Llama, DeepSeek, MegaScale...); emphasis on training-system design and operational lessons, not the model itself.
- `topics/` — cross-cutting engineering handbook chapters that weave multiple sources together. These are the highest-value docs and the intended "deep" reference (e.g. `tensor_parallelism.md`, `checkpointing.md` are the quality templates to imitate).
- `interview/` — interview handbook: high-frequency questions, follow-ups (追问), production cases, wrong-answer vs. good-answer examples.
- `roadmaps/` — `30_day_plan.md`, `90_day_plan.md`, `yearly_plan.md`; pace reading so material gets digested, not just collected.
- `references/` — `papers.csv` and `reports.csv`, structured indexes (columns: `category,title,year,primary_link,notes,status`; status ∈ `draft`/`todo`).
- `MASTER_READING_LIST.md`, `KNOWLEDGE_GRAPH.md` — the navigation/index layer.

`assets/` holds shared images (e.g. the homepage knowledge-map SVG).

## Document conventions (follow these when adding or editing notes)

**Reading philosophy is the core constraint.** Every note answers engineering questions first: what real problem did this solve, what was the training-system bottleneck at the time, which system boundary did it move (显存 / 通信 / 调度 / kernel / 容错 / 运维), how does it shape today's Megatron/DeepSpeed/FSDP/TE/DeepSeek/Llama stack, and what breaks in production. Formulas and proofs appear only when they serve engineering judgment.

**`papers/` template** (see `papers/transformer.md` as the canonical example) — sections in order, separated by `---`:
论文信息 → 解决的问题 → 背景与瓶颈 → 核心创新 → 关键图表解读 → 工程价值 → 对训练基础设施的影响 → 今天的应用场景 → 后续演进 → 相关论文 → 相关代码 → 面试高频问题 → 生产环境思考题 → 我的总结.

**`interview/` template** (see `interview/tensor_parallelism.md`): 高频面试题 → 追问问题 → 生产环境案例 → 常见错误回答 → 优秀回答示例.

**`topics/` chapters** are long-form numbered sections (`## 1.`, `## 2.` ...), typically opening with a one-sentence "如果只能记一句话" takeaway, then problem framing → mechanism → config guidance → troubleshooting → relationships to adjacent topics.

## Linking discipline (the thing most likely to break)

This repo lives and dies on its internal Markdown links — keep them clickable and correct.

- Use **relative** links between docs. Cross-directory links from inside a subfolder go up one level: from `papers/x.md` to a topic, use a path like `../topics/tensor_parallelism.md`; same-folder links omit the path. From the repo root, use paths like `training-infra-roadmap/topics/tensor_parallelism.md`.
- Links are **bidirectional by design**. `KNOWLEDGE_GRAPH.md` maintains a "双向索引" section and `MASTER_READING_LIST.md` maps each source to the topic notes it supports. When you add a doc or a relationship, update both the source note's "相关..." sections **and** these two index files so the graph stays consistent.
- Diagrams in `KNOWLEDGE_GRAPH.md` and READMEs are **Mermaid** (`flowchart`/`graph`). The graph is intentionally kept "少而清楚" (sparse and clear) — do not re-bloat it into a full dependency mesh; the textual index, not the main diagram, carries fine-grained relationships.

## Maintenance principles (from README.md)

- `topics/` must read as engineering handbook chapters, not concept excerpts.
- `papers/` and `tech_reports/` serve system understanding, not exhaustive academic coverage.
- Each handbook directory should have its own `README.md`, reading list, and knowledge graph.
- Current build priority: deepen the ~7 seed drafts to be recitable/interview-ready/design-grade, using `tensor_parallelism.md` and `checkpointing.md` as the bar; then extend FSDP, MoE, FP8, NCCL; then fill `interview/` (NCCL, RDMA, RoCE, InfiniBand); then turn the reading list / CSVs / graph into a searchable entry point.

## Language

All content is written in Chinese (technical terms kept in English: TP, ZeRO, FSDP, FlashAttention, GEMM, all-reduce, etc.). Match this mixed style when authoring or editing.

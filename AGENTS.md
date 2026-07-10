# AGENTS.md

This file is the shared working guide for AI coding agents in this repository. It applies to Codex, Claude Code, and other agents unless a tool-specific file says otherwise.

## Repository Purpose

ReadBase is a Chinese-language, content-first Personal Research Operating System for Large-Scale AI Systems. It is not an application repo and has no product build pipeline. The current primary handbook is `training-infra-roadmap/`, which is Phase 1: Training Infrastructure.

Write for a software engineer growing into a senior AI/LLM training infrastructure engineer. Prefer engineering judgment, system design, production troubleshooting, and interview readiness over academic-style summaries.

North Star: build a production-grade understanding of Large-Scale AI Systems through a human-AI maintained Personal Research Operating System.

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
- `training-infra-roadmap/engineering_blogs/` contains engineering blogs, official docs, release notes, and vendor technical posts that expose implementation details not captured by papers.
- `training-infra-roadmap/tracking/` is the research radar: frontier scans, scan logs, recent papers, engineering blogs, release notes, infra trends, agentic RL signals, monthly digests, and historical backfill. It records signal and triage, not full notes.
- `training-infra-roadmap/reading_queue/` turns tracking signals into P0/P1 reading decisions.
- `training-infra-roadmap/learning_log/` records monthly learning progress, questions, and next steps.
- `training-infra-roadmap/insights/` stores original engineering judgments.
- `training-infra-roadmap/experiments/` records practical verification and benchmarks.
- `training-infra-roadmap/playbooks/` stores production troubleshooting runbooks.
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

For `engineering_blogs/`, do not summarize marketing copy. Extract the engineering signal: source information → solved problem → engineering background → core mechanism → system design details → performance/stability information → production lessons → related topics → questions to pursue → short summary.

For `tracking/`, keep entries lightweight and judgment-heavy. Each item should include source/type/link, impact level, `Decision` (`Ignore` / `Observe` / `Read` / `Deep Dive`), `Reason`, related topics, one-sentence value, and next step. Tracking is an inbox/radar; promote only important items into `reading_queue/`, `papers/`, `tech_reports/`, `engineering_blogs/`, `topics/`, `insights/`, `experiments/`, or `playbooks/`.

For `tracking/frontier_scan_YYYY-MM-DD.md`, scan from the previous cursor in `tracking/scan_log.md` to the actual scan end timestamp. Do not force a natural week. Do not write an end-of-day timestamp such as `23:59` unless that time has actually been scanned. If the exact scan end timestamp was not recorded, the next scan should backtrack to the last confirmed timestamp and dedupe. A frontier scan can have zero accepted signals. Every accepted signal needs `Source ID`, `First seen`, scan window, `Decision`, and `Reason`. After each scan, update `tracking/scan_log.md` with the next cursor.

For every accepted signal and every new paper/report note, verify title, authors, publication date, and key numeric claims against the primary source page before writing them as facts. For arXiv sources, the arXiv ID resolving is not enough: the `citation_title`, `citation_author`, `citation_date`, and abstract/method details must match the note. If a detail is inferred rather than directly sourced, label it as an inference.

For `tracking/historical_backfill.md`, do not chase recency. It is an index and rules page. Backfill entries should live in `tracking/backfill/YYYY-MM.md`, where `YYYY-MM` is the material's original publication month. Backfill only past materials that fill a current engineering judgment gap. Each entry should explain original time, backfill time, why it is backfilled now, historical impact, current value, Decision, Reason, suggested action, related topics, target destination, and lifecycle status. Do not mix historical backfill into frontier scans.

Weekly signal reports and weekly papers templates are retired. Keep existing weekly files only as historical audit records. For current updates, use frontier scans plus `tracking/scan_log.md`; for formal summaries, use monthly signal reports.

Monthly reports use the previous calendar month, named `monthly_signal_YYYY-MM.md`. Monthly reports are the high-quality digest and should summarize frontier scans, backfill, release notes, and actual reading results; they should not rediscover material from scratch.

Frontier/monthly signal scanning must use the repository owner's focus filter, not generic AI popularity. Prioritize AI Systems, Training Infra, distributed training, GPU clusters/networking, Megatron/DeepSpeed/FSDP, MoE, FlashAttention/kernel/precision, NVIDIA training stack, large-scale training reports, and Agentic RL/post-training infra. Usually reject generic model releases, application papers, domain datasets, prompt tricks, product news, and algorithm-only items with no infra consequence.

Vendor watch rule: every frontier scan and monthly signal report must include an explicit `OpenAI / Anthropic / NVIDIA Watch` section. These three vendors are not automatic accepts, but their papers, technical reports, official docs, engineering blogs, release notes, and research posts must be visibly triaged as `Accepted`, `Observed`, `Rejected`, or `Not found / not verifiable in this scan`. If a vendor source cannot be scanned or verified, state that limitation instead of silently omitting the vendor.

Hugging Face watch rule: every frontier scan and monthly signal report must also include an explicit `Hugging Face Watch`. Scan the Hugging Face Blog plus relevant TRL, Transformers, Accelerate, PEFT, and Kernels releases/docs, especially for Agentic RL, rollout correctness, training-serving integration, long context, distributed training, and inference backends. Distinguish official-team or vendor-authored posts from community posts, and do not auto-accept either category.

For `playbooks/`, write runbooks, not concept explanations: symptom → impact scope → first response → investigation order → commands → log keywords → likely root causes → fixes → validation → prevention → related topics/sources/experiments → postmortem questions.

For `topics/`, write as engineering handbook chapters: problem framing → mechanism → config guidance → production pitfalls → troubleshooting → adjacent-system relationships.

For `interview/`, include: 高频面试题 → 追问问题 → 生产环境案例 → 常见错误回答 → 优秀回答示例.

## Knowledge Lifecycle

Use these statuses when tracking important materials:

```text
NEW         just discovered
READING     actively reading
SUMMARIZED  converted into a paper/report/blog note
DIGESTED    reflected in topics or insights
VERIFIED    validated by experiment or reproduction
IMPLEMENTED used in real engineering practice or production design
OBSOLETE    outdated or superseded
```

## Human-AI Workflow Protocol

When adding a new material:

1. Add it to `tracking/` with impact, Decision, and Reason. Use frontier scan for new material and `tracking/backfill/YYYY-MM.md` for older classics.
2. If important, move it to `reading_queue/P0.md` or `reading_queue/P1.md`.
3. After reading, create or update the relevant paper/report/blog note.
4. Update at least one `topics/` chapter if the material changes system understanding.
5. If it forms a technical judgment, add or update `insights/`.
6. If it can be validated, create or update an `experiments/` record.
7. If it changes production troubleshooting, add or update a `playbooks/` runbook.
8. Update `KNOWLEDGE_GRAPH.md` / `MASTER_READING_LIST.md` when navigation changes.

Important: if a paper does not change engineering judgment, experiment design, or system implementation, it is not really finished.

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
